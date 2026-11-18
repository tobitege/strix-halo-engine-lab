// deepseek41 CUDA decode kernels, v1 vs v2 on real tensors of one layer: bitwise equality and interleaved timing.
//   strix-cuda-ds41-kern-bench moe <model.gguf> <routes.txt> [layer = 20] [windows = 64] [rows = 1,2,4,8] [cold = 1]
// moe: the routed experts (routes + gate/up + down) of windows of R consecutive recorded tokens, random A8 inputs per
// window; v1 = enqueue_routed_gate_up + enqueue_routed_down, v2 = the _v2 pair. Every window compares raw (and the A8
// of the SwiGLU outputs) bitwise; timings alternate v1 / v2 per window, cold L2 (a 256 MB read before each) unless
// cold = 0; GB/s = union experts' bytes / time.

#include "strix/cuda/ds41_common.hpp"
#include "strix/cuda/ds41_moe.hpp"
#include "strix/cuda/ds41_moe_v2.hpp"
#include "strix/cuda/ds41_core.hpp"
#include "strix/cuda/ds41_router_v2.hpp"
#include "strix/cuda/ds41_parts_host.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/ds41_attention.hpp"
#include "strix/ds41_layer.hpp"
#include "../../src/cuda/models/ds41/ds41_moe_device.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace dc = strix::ds41::cuda;

namespace {

__global__ void flush_kernel(const int4* p, std::size_t n, int* sink) {
    int acc = 0;
    for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x) {
        const int4 v = __ldcg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678) *sink = acc;
}

std::vector<std::vector<std::uint32_t>> read_routes(const char* path, std::uint32_t layer) {
    std::vector<std::vector<std::uint32_t>> tokens;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string item;
        while (ls >> item) {
            const auto colon = item.find(':');
            if (colon == std::string::npos || std::stoul(item.substr(0, colon)) != layer) continue;
            std::vector<std::uint32_t> ids;
            std::istringstream is(item.substr(colon + 1));
            std::string v;
            while (std::getline(is, v, ',')) ids.push_back(static_cast<std::uint32_t>(std::stoul(v)));
            tokens.push_back(ids);
        }
    }
    return tokens;
}

void load(std::ifstream& in, std::uint64_t offset, std::uint64_t bytes, std::uint8_t* device) {
    constexpr std::uint64_t chunk = 64ULL << 20U;
    std::uint8_t* host = nullptr;
    STRIX_CUDA_CHECK(cudaMallocHost(&host, chunk));
    in.seekg(static_cast<std::streamoff>(offset));
    for (std::uint64_t at = 0; at < bytes; at += chunk) {
        const auto n = std::min(chunk, bytes - at);
        in.read(reinterpret_cast<char*>(host), static_cast<std::streamsize>(n));
        STRIX_CUDA_CHECK(cudaMemcpy(device + at, host, n, cudaMemcpyHostToDevice));
    }
    STRIX_CUDA_CHECK(cudaFreeHost(host));
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

int moe(int argc, char** argv) {
    if (argc < 4) throw std::runtime_error("moe <model.gguf> <routes.txt> [layer] [windows] [rows] [cold]");
    const std::uint32_t layer = argc > 4 ? static_cast<std::uint32_t>(std::atoi(argv[4])) : 20U;
    const std::uint32_t windows = argc > 5 ? static_cast<std::uint32_t>(std::atoi(argv[5])) : 64U;
    std::vector<std::uint32_t> rows_list{1, 2, 4, 8};
    if (argc > 6) {
        rows_list.clear();
        std::istringstream is(argv[6]);
        std::string v;
        while (std::getline(is, v, ',')) rows_list.push_back(static_cast<std::uint32_t>(std::stoul(v)));
    }
    const int cold_mode = argc > 7 ? std::atoi(argv[7]) : 1;  // 1 = flush before each arm, 0 = none, 2 = hot (arm run twice, the second timed)
    const bool cold = cold_mode == 1, hot = cold_mode == 2;
    if (argc > 8) dc::set_routed_v2_diagnostic(std::atoi(argv[8]));
    // the B arm: 2 = v2 pair, 3 = v3 pair, 13 = v1 gate/up + v3 down, 31 = v3 gate/up + v1 down
    int variant = argc > 9 ? std::atoi(argv[9]) : 3;
    if (variant >= 100) { dc::set_routed_v3_one_item(true); variant -= 100; }  // 1xx: v3 gate/up one item per block
    // 43 = v1b gate/up + v3 down, 41 = v1b gate/up + v1 down
    const bool b_v1b = variant == 43 || variant == 41;
    const bool b_gate_v3 = variant == 3 || variant == 31, b_down_v3 = variant == 3 || variant == 13 || variant == 43, b_v1_gate = variant == 13,
               b_v1_down = variant == 31 || variant == 41;
    std::printf("B arm = v%d\n", variant);
    if (const char* e = std::getenv("STRIX_DSRTXK_V1B_R1")) {  // the v1b R1 kernel form (diagnostics)
        dc::set_v1b_r1_form(std::atoi(e));
        std::printf("v1b R1 form %s\n", e);
    }
    const auto device = dc::select_device(0);
    std::printf("device %s, %d SMs\n", device.name.c_str(), device.sms);
    const auto file = strix::gguf::read_file(argv[2]);
    const auto find = [&](const std::string& name) -> const strix::gguf::TensorInfo& {
        for (const auto& t : file.tensors)
            if (t.name == name) return t;
        throw std::runtime_error("missing tensor " + name);
    };
    const std::string p = "blk." + std::to_string(layer) + ".";
    const auto& tg = find(p + "ffn_gate_exps.weight");
    const auto& tu = find(p + "ffn_up_exps.weight");
    const auto& td = find(p + "ffn_down_exps.weight");
    dc::MoeShape shape;
    dc::DeviceBuffer<std::uint8_t> gate(tg.byte_size), up(tu.byte_size), down(td.byte_size);
    {
        std::ifstream in(argv[2], std::ios::binary);
        load(in, tg.absolute_offset, tg.byte_size, gate.get());
        load(in, tu.absolute_offset, tu.byte_size, up.get());
        load(in, td.absolute_offset, td.byte_size, down.get());
    }
    const auto tokens = read_routes(argv[3], layer);
    if (tokens.empty()) throw std::runtime_error("no routes for the layer");
    const std::uint32_t cap = 8U, H = shape.hidden, I = shape.inter, K = shape.top_k;
    std::mt19937 rng(11);
    dc::DeviceBuffer<std::int8_t> xc(std::size_t{cap} * H);
    dc::DeviceBuffer<float> xd(std::size_t{cap} * H / 32U), raw1(std::size_t{cap} * K * H), raw2(std::size_t{cap} * K * H),
        y(dc::routed_v2_y_floats(cap * K, shape));
    dc::DeviceBuffer<std::uint32_t> ids(std::size_t{cap} * K), counters(dc::routed_v3_counter_words(cap * K));
    counters.zero();
    dc::DeviceBuffer<std::uint8_t> ws1(dc::moe_layer_workspace_bytes(cap, shape)), ws2(dc::moe_layer_workspace_bytes(cap, shape));
    dc::DeviceBuffer<int4> flush((256ULL << 20U) / sizeof(int4));
    dc::DeviceBuffer<int> sink(1);
    STRIX_CUDA_CHECK(cudaMemset(flush.get(), 1, flush.size() * sizeof(int4)));
    dc::Stream stream;
    const auto st = stream.get();
    cudaEvent_t e0, e1, em;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventCreate(&em);
    const auto parts1 = dc::moe_layer_workspace_views(ws1.get(), cap, shape);
    const auto parts2 = dc::moe_layer_workspace_views(ws2.get(), cap, shape);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> scale(0.002F, 0.05F);
    if (argc > 10 && variant == 3) {  // clock64 stamps of one hot R1 gate/up v3 launch
        const std::uint32_t R = 1;
        std::vector<std::uint32_t> h(tokens[3].begin(), tokens[3].begin() + K);
        ids.upload(h);
        dc::MoeRoutesView rv;
        rv.ids = ids.get(); rv.rows = R; rv.top_k = K; rv.experts = shape.experts; rv.split = 192; rv.out = parts2.routes;
        dc::enqueue_moe_routes(rv, st);
        dc::RoutedGateUpView g;
        g.shape = shape; g.weights = {gate.get(), up.get(), down.get()}; g.routes = parts2.routes; g.max_union = K; g.max_rows = R;
        g.x = {xc.get(), xd.get(), nullptr, R, H, H}; g.act = parts2.act;
        dc::DeviceBuffer<unsigned long long> stamps(1024U * 32U);
        for (int rep = 0; rep < 3; ++rep) {
            stamps.zero();
            dc::set_routed_v3_stamps(rep == 2 ? stamps.get() : nullptr);
            dc::enqueue_routed_gate_up_v3(g, y.get(), counters.get(), st);
            stream.sync();
        }
        dc::set_routed_v3_stamps(nullptr);
        const auto hs = stamps.download(stamps.size());
        std::vector<double> wait, comp, fin, total_k;
        int blocks = 0;
        for (std::uint32_t blk = 0; blk < 1024U; ++blk) {
            if (!hs[blk * 32U]) continue;
            ++blocks;
            for (std::uint32_t k = 0; k < 8U; ++k) {
                const auto* q = &hs[blk * 32U + k * 4U];
                if (!q[0] || !q[3]) continue;
                wait.push_back(static_cast<double>(q[1] - q[0]));
                comp.push_back(static_cast<double>(q[2] - q[1]));
                fin.push_back(static_cast<double>(q[3] - q[2]));
            }
            std::uint64_t first = hs[blk * 32U], last = 0;
            for (std::uint32_t k = 0; k < 8U; ++k) if (hs[blk * 32U + k * 4U + 3U]) last = hs[blk * 32U + k * 4U + 3U];
            total_k.push_back(static_cast<double>(last - first));
        }
        std::printf("v3 stamps (cycles, median): blocks %d items %zu | slab wait %.0f | compute %.0f | finisher %.0f | block loop total %.0f\n",
                    blocks, wait.size(), median(wait), median(comp), median(fin), median(total_k));
    }
    std::printf("R  windows  union/win  v1 us   GB/s  | v2 us   GB/s  | v2/v1   mismatches\n");
    bool all_ok = true;
    for (const std::uint32_t R : rows_list) {
        std::vector<double> t1, t2, g1t, g2t;
        double unions = 0;
        std::uint64_t mism = 0;
        for (std::uint32_t win = 0; win < windows + 2U; ++win) {
            std::vector<std::uint32_t> h;
            for (std::uint32_t r = 0; r < R; ++r) {
                const auto& t = tokens[(win * R + r) % tokens.size()];
                h.insert(h.end(), t.begin(), t.begin() + K);
            }
            ids.upload(h);
            const auto u = static_cast<std::uint32_t>(std::set<std::uint32_t>(h.begin(), h.end()).size());
            std::vector<std::int8_t> hc(std::size_t{R} * H);
            for (auto& c : hc) c = static_cast<std::int8_t>(code(rng));
            std::vector<float> hd(std::size_t{R} * H / 32U);
            for (auto& d : hd) d = scale(rng);
            STRIX_CUDA_CHECK(cudaMemcpy(xc.get(), hc.data(), hc.size(), cudaMemcpyHostToDevice));
            STRIX_CUDA_CHECK(cudaMemcpy(xd.get(), hd.data(), hd.size() * 4U, cudaMemcpyHostToDevice));
            for (const auto* parts : {&parts1, &parts2}) {
                dc::MoeRoutesView rv;
                rv.ids = ids.get();
                rv.rows = R;
                rv.top_k = K;
                rv.experts = shape.experts;
                rv.split = 192;
                rv.out = parts->routes;
                dc::enqueue_moe_routes(rv, st);
            }
            const auto views = [&](const dc::MoeLayerScratch& parts, dc::RoutedGateUpView& g, dc::RoutedDownView& d) {
                g.shape = d.shape = shape;
                g.weights = d.weights = {gate.get(), up.get(), down.get()};
                g.routes = d.routes = parts.routes;
                g.max_union = d.max_union = std::min(R * K, shape.experts);
                g.max_rows = d.max_rows = R;
                g.x = {xc.get(), xd.get(), nullptr, R, H, H};
                g.act = d.act = parts.act;
            };
            dc::RoutedGateUpView g1, g2;
            dc::RoutedDownView d1, d2;
            views(parts1, g1, d1);
            views(parts2, g2, d2);
            d1.raw = raw1.get();
            d2.raw = raw2.get();
            STRIX_CUDA_CHECK(cudaMemsetAsync(raw2.get(), 0xff, raw2.size() * 4U, st));
            STRIX_CUDA_CHECK(cudaMemsetAsync(parts2.act.codes, 0x55, std::size_t{R} * K * I, st));
            STRIX_CUDA_CHECK(cudaMemsetAsync(parts2.act.deltas, 0x55, std::size_t{R} * K * I / 8U, st));
            if (parts2.act.sums) STRIX_CUDA_CHECK(cudaMemsetAsync(parts2.act.sums, 0x55, std::size_t{R} * K * I / 8U, st));
            float ms[2] = {0, 0}, mg[2] = {0, 0};
            for (int k = 0; k < 2; ++k) {
                const int which = (win & 1U) ? 1 - k : k;  // alternate the order
                if (cold) flush_kernel<<<1024, 256, 0, st>>>(flush.get(), flush.size(), sink.get());
                if (hot) {  // the same window once untimed: its expert bytes are in L2 for the timed run
                    if (which == 0) {
                        dc::enqueue_routed_gate_up(g1, st);
                        dc::enqueue_routed_down(d1, st);
                    } else if (variant != 2) {
                        if (b_v1b) dc::enqueue_routed_gate_up_v1b(g2, st);
                        else if (b_v1_gate) dc::enqueue_routed_gate_up(g2, st);
                        else dc::enqueue_routed_gate_up_v3(g2, y.get(), counters.get(), st);
                        if (b_v1_down) dc::enqueue_routed_down(d2, st);
                        else dc::enqueue_routed_down_v3(d2, st);
                    } else {
                        dc::enqueue_routed_gate_up_v2(g2, y.get(), st);
                        dc::enqueue_routed_down_v2(d2, y.get(), st);
                    }
                }
                cudaEventRecord(e0, st);
                if (which == 0) {
                    dc::enqueue_routed_gate_up(g1, st);
                    cudaEventRecord(em, st);
                    dc::enqueue_routed_down(d1, st);
                } else {
                    if (b_v1b) dc::enqueue_routed_gate_up_v1b(g2, st);
                    else if (b_gate_v3) dc::enqueue_routed_gate_up_v3(g2, y.get(), counters.get(), st);
                    else if (b_v1_gate) dc::enqueue_routed_gate_up(g2, st);
                    else dc::enqueue_routed_gate_up_v2(g2, y.get(), st);
                    cudaEventRecord(em, st);
                    if (b_down_v3) dc::enqueue_routed_down_v3(d2, st);
                    else if (b_v1_down) dc::enqueue_routed_down(d2, st);
                    else dc::enqueue_routed_down_v2(d2, y.get(), st);
                }
                cudaEventRecord(e1, st);
                stream.sync();
                cudaEventElapsedTime(&ms[which], e0, e1);
                cudaEventElapsedTime(&mg[which], e0, em);
            }
            const auto a = raw1.download(std::size_t{R} * K * H), b = raw2.download(std::size_t{R} * K * H);
            std::vector<std::int8_t> ca(std::size_t{R} * K * I), cb(ca.size());
            STRIX_CUDA_CHECK(cudaMemcpy(ca.data(), parts1.act.codes, ca.size(), cudaMemcpyDeviceToHost));
            STRIX_CUDA_CHECK(cudaMemcpy(cb.data(), parts2.act.codes, cb.size(), cudaMemcpyDeviceToHost));
            mism += std::memcmp(a.data(), b.data(), a.size() * 4U) != 0 ? 1U : 0U;
            mism += std::memcmp(ca.data(), cb.data(), ca.size()) != 0 ? 1U : 0U;
            {  // the rest of the A8 state: deltas and sums of every slot's groups
                const std::size_t ng = std::size_t{R} * K * I / 32U;
                std::vector<float> da(ng), db(ng);
                std::vector<std::int32_t> sa(ng), sb(ng);
                STRIX_CUDA_CHECK(cudaMemcpy(da.data(), parts1.act.deltas, ng * 4U, cudaMemcpyDeviceToHost));
                STRIX_CUDA_CHECK(cudaMemcpy(db.data(), parts2.act.deltas, ng * 4U, cudaMemcpyDeviceToHost));
                if (variant != 2 && parts1.act.sums && parts2.act.sums) {
                    STRIX_CUDA_CHECK(cudaMemcpy(sa.data(), parts1.act.sums, ng * 4U, cudaMemcpyDeviceToHost));
                    STRIX_CUDA_CHECK(cudaMemcpy(sb.data(), parts2.act.sums, ng * 4U, cudaMemcpyDeviceToHost));
                }
                mism += std::memcmp(da.data(), db.data(), ng * 4U) != 0 ? 1U : 0U;
                mism += std::memcmp(sa.data(), sb.data(), ng * 4U) != 0 ? 1U : 0U;
                const auto cz = counters.download(counters.size());
                for (const auto c : cz) if (c) { ++mism; break; }  // every launch leaves the counters zero
            }
            if (win < 2U) continue;
            t1.push_back(ms[0] * 1e3);
            t2.push_back(ms[1] * 1e3);
            g1t.push_back(mg[0] * 1e3);
            g2t.push_back(mg[1] * 1e3);
            unions += u;
        }
        const double m1 = median(t1), m2 = median(t2), bytes = unions / windows * static_cast<double>(shape.expert_bytes());
        std::printf("%u  %6u   %6.1f    %7.1f  %5.0f | %7.1f  %5.0f | %5.3f   %llu\n", R, windows, unions / windows, m1, bytes / (m1 * 1e3), m2,
                    bytes / (m2 * 1e3), m2 / m1, static_cast<unsigned long long>(mism));
        std::vector<double> d1t, d2t, dt, dg, dd;
        for (std::size_t i = 0; i < t1.size(); ++i) {
            d1t.push_back(t1[i] - g1t[i]);
            d2t.push_back(t2[i] - g2t[i]);
            dt.push_back(t2[i] - t1[i]);
            dg.push_back(g2t[i] - g1t[i]);
            dd.push_back((t2[i] - g2t[i]) - (t1[i] - g1t[i]));
        }
        const auto q = [](std::vector<double> v, double f) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[static_cast<std::size_t>(f * (v.size() - 1))]; };
        std::printf("   gate/up us A %.1f B %.1f | down us A %.1f B %.1f (per-window medians)\n", median(g1t), median(g2t), median(d1t), median(d2t));
        std::printf("   paired B-A us: total %.1f [%.1f, %.1f] gate/up %.1f [%.1f, %.1f] down %.1f [%.1f, %.1f] (median [q25, q75])\n", q(dt, 0.5),
                    q(dt, 0.25), q(dt, 0.75), q(dg, 0.5), q(dg, 0.25), q(dg, 0.75), q(dd, 0.5), q(dd, 0.25), q(dd, 0.75));
        if (mism) all_ok = false;
    }
    std::printf(all_ok ? "BITWISE PASS\n" : "BITWISE FAIL\n");
    return all_ok ? 0 : 1;
}


// router: enqueue_router vs enqueue_router_v2 on the real F32 router of a layer, rows 1..8, random x per window
// (plus a NaN row and an all-tie case: zero x and zero bias), cold L2; compares scores, ids and weights bitwise.
int router(int argc, char** argv) {
    if (argc < 3) throw std::runtime_error("router <model.gguf> [layer] [windows] [cold]");
    const std::uint32_t layer = argc > 3 ? static_cast<std::uint32_t>(std::atoi(argv[3])) : 20U;
    const std::uint32_t windows = argc > 4 ? static_cast<std::uint32_t>(std::atoi(argv[4])) : 64U;
    const bool cold = argc > 5 ? std::atoi(argv[5]) != 0 : true;
    const auto device = dc::select_device(0);
    std::printf("device %s, %d SMs\n", device.name.c_str(), device.sms);
    if (const char* f = std::getenv("STRIX_DSRTXK_ROUTER_FORM")) {  // router v2 load schedule (same bits)
        dc::set_router_v2_form(std::atoi(f));
        std::printf("router v2 form %d\n", std::atoi(f));
    }
    const auto file = strix::gguf::read_file(argv[2]);
    const auto find = [&](const std::string& name) -> const strix::gguf::TensorInfo& {
        for (const auto& t : file.tensors)
            if (t.name == name) return t;
        throw std::runtime_error("missing tensor " + name);
    };
    const std::string p = "blk." + std::to_string(layer) + ".";
    const auto& tw = find(p + "ffn_gate_inp.weight");
    const auto& tb = find(p + "exp_probs_b.bias");
    const std::uint32_t E = 384, H = 5120, K = 6;
    dc::DeviceBuffer<std::uint8_t> w(tw.byte_size), b(tb.byte_size), zb(tb.byte_size);
    {
        std::ifstream in(argv[2], std::ios::binary);
        load(in, tw.absolute_offset, tw.byte_size, w.get());
        load(in, tb.absolute_offset, tb.byte_size, b.get());
    }
    STRIX_CUDA_CHECK(cudaMemset(zb.get(), 0, zb.size()));
    dc::DeviceBuffer<float> x(8U * H), s1(8U * E), s2(8U * E), w1(8U * K), w2(8U * K);
    dc::DeviceBuffer<std::uint32_t> i1(8U * K), i2(8U * K), counter(1);
    counter.zero();
    dc::DeviceBuffer<int4> flush((256ULL << 20U) / sizeof(int4));
    dc::DeviceBuffer<int> sink(1);
    STRIX_CUDA_CHECK(cudaMemset(flush.get(), 1, flush.size() * sizeof(int4)));
    dc::Stream stream;
    const auto st = stream.get();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    bool all_ok = true;
    if (argc > 6) {  // stamps of one R1 launch
        dc::DeviceBuffer<unsigned long long> stamps(192U * 8U);
        stamps.zero();
        dc::RouterView v;
        v.x = x.get(); v.rows = 1; v.weight = {reinterpret_cast<const float*>(w.get()), E, H}; v.bias = reinterpret_cast<const float*>(b.get());
        v.top_k = K; v.scores = s2.get(); v.ids = i2.get(); v.weights = w2.get();
        for (int rep = 0; rep < 3; ++rep) {
            flush_kernel<<<1024, 256, 0, st>>>(flush.get(), flush.size(), sink.get());
            dc::set_router_v2_stamps(rep == 2 ? stamps.get() : nullptr);
            dc::enqueue_router_v2(v, counter.get(), st);
            stream.sync();
        }
        dc::set_router_v2_stamps(nullptr);
        const auto h = stamps.download(192U * 8U);
        unsigned long long t0 = ~0ULL;
        for (std::uint32_t i = 0; i < 192U; ++i) t0 = std::min(t0, h[i * 8U]);
        std::vector<double> st0, st1, st2;
        double last_end = 0, last_arrived = 0;
        for (std::uint32_t i = 0; i < 192U; ++i) {
            st0.push_back((h[i * 8U] - t0) / 1e3); st1.push_back((h[i * 8U + 1U] - t0) / 1e3); st2.push_back((h[i * 8U + 2U] - t0) / 1e3);
            if (h[i * 8U + 3U]) last_end = (h[i * 8U + 3U] - t0) / 1e3;
            if (h[i * 8U + 4U]) last_arrived = (h[i * 8U + 4U] - t0) / 1e3;
        }
        std::printf("stamps us: start med %.1f max %.1f | loaded med %.1f max %.1f | scores med %.1f max %.1f | finisher past counter %.1f"
                    " | select end %.1f\n",
                    median(st0), *std::max_element(st0.begin(), st0.end()), median(st1), *std::max_element(st1.begin(), st1.end()), median(st2),
                    *std::max_element(st2.begin(), st2.end()), last_arrived, last_end);
    }
    std::printf("rows  v1 us  v2 us  v2/v1  mismatches\n");
    for (std::uint32_t R = 1; R <= 8; ++R) {
        std::vector<double> t1, t2;
        std::uint64_t mism = 0;
        for (std::uint32_t win = 0; win < windows + 2U; ++win) {
            std::vector<float> hx(std::size_t{R} * H);
            const int kind = win % 16U == 7U ? 1 : win % 16U == 11U ? 2 : 0;  // 1: NaN in row 0, 2: all ties
            for (auto& v : hx) v = kind == 2 ? 0.0F : nd(rng) * 0.05F;
            if (kind == 1) hx[17] = std::nanf("");
            STRIX_CUDA_CHECK(cudaMemcpy(x.get(), hx.data(), hx.size() * 4U, cudaMemcpyHostToDevice));
            dc::RouterView v;
            v.x = x.get();
            v.rows = R;
            v.weight = {reinterpret_cast<const float*>(w.get()), E, H};
            v.bias = reinterpret_cast<const float*>(kind == 2 ? zb.get() : b.get());
            v.top_k = K;
            v.route_scale = 1.5F;
            dc::RouterView v2 = v;
            v.scores = s1.get(); v.ids = i1.get(); v.weights = w1.get();
            v2.scores = s2.get(); v2.ids = i2.get(); v2.weights = w2.get();
            STRIX_CUDA_CHECK(cudaMemsetAsync(s2.get(), 0x55, s2.size() * 4U, st));
            STRIX_CUDA_CHECK(cudaMemsetAsync(i2.get(), 0x55, i2.size() * 4U, st));
            STRIX_CUDA_CHECK(cudaMemsetAsync(w2.get(), 0x55, w2.size() * 4U, st));
            float ms[2] = {0, 0};
            for (int k = 0; k < 2; ++k) {
                const int which = (win & 1U) ? 1 - k : k;
                if (cold) flush_kernel<<<1024, 256, 0, st>>>(flush.get(), flush.size(), sink.get());
                cudaEventRecord(e0, st);
                if (which == 0) dc::enqueue_router(v, st);
                else dc::enqueue_router_v2(v2, counter.get(), st);
                cudaEventRecord(e1, st);
                stream.sync();
                cudaEventElapsedTime(&ms[which], e0, e1);
            }
            const auto a = s1.download(std::size_t{R} * E), bb = s2.download(std::size_t{R} * E);
            const auto ia = i1.download(std::size_t{R} * K), ib = i2.download(std::size_t{R} * K);
            const auto wa = w1.download(std::size_t{R} * K), wb = w2.download(std::size_t{R} * K);
            if (std::memcmp(a.data(), bb.data(), a.size() * 4U) || std::memcmp(ia.data(), ib.data(), ia.size() * 4U) ||
                std::memcmp(wa.data(), wb.data(), wa.size() * 4U) || counter.download(1)[0] != 0U)
                ++mism;
            if (win < 2U) continue;
            t1.push_back(ms[0] * 1e3);
            t2.push_back(ms[1] * 1e3);
        }
        const double m1 = median(t1), m2 = median(t2);
        std::printf("%u     %6.1f %6.1f  %5.3f  %llu\n", R, m1, m2, m2 / m1, static_cast<unsigned long long>(mism));
        if (mism) all_ok = false;
    }
    std::printf(all_ok ? "BITWISE PASS\n" : "BITWISE FAIL\n");
    return all_ok ? 0 : 1;
}


// routes: enqueue_resident_routes (one warp up to 32 slots) vs enqueue_resident_routes_block on random windows:
// rows 1..5 x top_k 6 of distinct ids per row (repeats across rows), a random part of the experts not resident
int routes(int argc, char** argv) {
    const std::uint32_t windows = argc > 2 ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 2000U;
    const std::uint32_t E = 384, K = 6;
    std::mt19937 rng(9);
    dc::DeviceBuffer<std::uint32_t> ids(32), a(6U * 64U), b(6U * 64U);
    dc::DeviceBuffer<std::int32_t> slot_of(E);
    dc::Stream stream;
    const auto st = stream.get();
    const auto views = [&](std::uint32_t* base, std::uint32_t rows) {
        dc::ResidentRoutesView v;
        v.ids = ids.get(); v.slot_of = slot_of.get(); v.rows = rows; v.top_k = K; v.experts = E; v.split = 192;
        v.out.experts = base; v.out.begins = base + 64; v.out.rows = base + 128; v.out.slots = base + 192;
        v.out.order = base + 256; v.out.low_counts = base + 320; v.out.counts = base + 352;
        return v;
    };
    std::uint64_t mism = 0;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::vector<double> ta, tb;
    for (std::uint32_t w = 0; w < windows; ++w) {
        const std::uint32_t rows = 1U + w % 5U;
        std::vector<std::uint32_t> h;
        for (std::uint32_t r = 0; r < rows; ++r) {
            std::vector<std::uint32_t> all(E);
            for (std::uint32_t e = 0; e < E; ++e) all[e] = e;
            std::shuffle(all.begin(), all.end(), rng);
            const std::uint32_t span = w % 3U == 0U ? 12U : E;  // small spans repeat experts across rows
            std::vector<std::uint32_t> pick(all.begin(), all.end());
            if (span < E) { for (std::uint32_t k = 0; k < K; ++k) pick[k] = (all[k] % span); std::sort(pick.begin(), pick.begin() + K);
                            pick.erase(std::unique(pick.begin(), pick.begin() + K), pick.begin() + K); }
            for (std::uint32_t k = 0; k < K; ++k) h.push_back(k < pick.size() ? pick[k] : (pick[0] + 100U + k) % E);
        }
        std::vector<std::int32_t> so(E);
        const int miss_pct = static_cast<int>(w % 4U) * 20;
        for (std::uint32_t e = 0; e < E; ++e) so[e] = static_cast<int>(rng() % 100U) < miss_pct ? -1 : static_cast<std::int32_t>(rng() % 9000U);
        ids.upload(h);
        slot_of.upload(so);
        STRIX_CUDA_CHECK(cudaMemsetAsync(a.get(), 0x5a, a.size() * 4U, st));
        STRIX_CUDA_CHECK(cudaMemsetAsync(b.get(), 0x5a, b.size() * 4U, st));
        cudaEventRecord(e0, st);
        dc::enqueue_resident_routes_block(views(a.get(), rows), st);
        cudaEventRecord(e1, st);
        stream.sync();
        float m1 = 0;
        cudaEventElapsedTime(&m1, e0, e1);
        cudaEventRecord(e0, st);
        dc::enqueue_resident_routes(views(b.get(), rows), st);
        cudaEventRecord(e1, st);
        stream.sync();
        float m2 = 0;
        cudaEventElapsedTime(&m2, e0, e1);
        ta.push_back(m1 * 1e3);
        tb.push_back(m2 * 1e3);
        const auto ha = a.download(a.size()), hb = b.download(b.size());
        if (std::memcmp(ha.data(), hb.data(), ha.size() * 4U) != 0) ++mism;
    }
    std::printf("routes windows %u: block %.1f us, warp %.1f us (median event time), mismatching windows %llu\n", windows, median(ta), median(tb),
                static_cast<unsigned long long>(mism));
    std::printf(mism ? "BITWISE FAIL\n" : "BITWISE PASS\n");
    return mism ? 1 : 0;
}


int ob(int argc, char** argv) {
    if (argc < 3) throw std::runtime_error("ob <model.gguf> [layer] [windows]");
    const std::uint32_t layer = argc > 3 ? static_cast<std::uint32_t>(std::atoi(argv[3])) : 20U;
    const std::uint32_t windows = argc > 4 ? static_cast<std::uint32_t>(std::atoi(argv[4])) : 64U;
    const auto file = strix::gguf::read_file(argv[2]);
    const strix::gguf::TensorInfo* tw = nullptr;
    for (const auto& t : file.tensors)
        if (t.name == "blk." + std::to_string(layer) + ".attn_output_b.weight") tw = &t;
    if (!tw) throw std::runtime_error("missing attn_output_b");
    const std::uint32_t H = 5120, OL = 8192, half = OL / 2U;
    dc::DeviceBuffer<std::uint8_t> raw(tw->byte_size);
    {
        std::ifstream in(argv[2], std::ios::binary);
        load(in, tw->absolute_offset, tw->byte_size, raw.get());
    }
    dc::Stream stream;
    const auto st = stream.get();
    dc::Q8TiledBuffer tiled;
    tiled.repack(raw.get(), H, OL, st);
    stream.sync();
    dc::DeviceBuffer<std::int8_t> codes(8U * OL);
    dc::DeviceBuffer<float> deltas(8U * OL / 32U), y1(8U * H), y2(8U * H);
    dc::DeviceBuffer<std::int32_t> sums(8U * OL / 32U);
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> scale(0.001F, 0.1F);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::uint64_t mism = 0;
    std::printf("rows  two launches us  one launch us (median, warm)\n");
    for (std::uint32_t R = 1; R <= 8; ++R) {
        std::vector<double> ta, tb;
        for (std::uint32_t w = 0; w < windows; ++w) {
            std::vector<std::int8_t> hc(8U * OL);
            for (auto& c : hc) c = static_cast<std::int8_t>(code(rng));
            std::vector<float> hd(8U * OL / 32U);
            for (auto& d : hd) d = scale(rng);
            codes.upload(hc);
            deltas.upload(hd);
            const auto view = [&](std::uint32_t h, float* out) {
                dc::Q8A8View v;
                v.tiled = tiled.view();
                v.count = H;
                v.k_offset = h * half;
                v.width = half;
                v.x = {codes.get() + h * half, deltas.get() + h * half / 32U, sums.get() + h * half / 32U, R, half, OL};
                v.out = out;
                v.accumulate = h > 0;
                return v;
            };
            STRIX_CUDA_CHECK(cudaMemsetAsync(y2.get(), 0x7f, y2.size() * 4U, st));
            float ma = 0, mb = 0;
            cudaEventRecord(e0, st);
            dc::enqueue_q8_0_a8(view(0, y1.get()), st);
            dc::enqueue_q8_0_a8(view(1, y1.get()), st);
            cudaEventRecord(e1, st);
            stream.sync();
            cudaEventElapsedTime(&ma, e0, e1);
            auto one = view(0, y2.get());
            one.k_halves = 2U;
            cudaEventRecord(e0, st);
            dc::enqueue_q8_0_a8(one, st);
            cudaEventRecord(e1, st);
            stream.sync();
            cudaEventElapsedTime(&mb, e0, e1);
            const auto a = y1.download(std::size_t{R} * H), b = y2.download(std::size_t{R} * H);
            if (std::memcmp(a.data(), b.data(), a.size() * 4U) != 0) ++mism;
            if (w >= 2U) { ta.push_back(ma * 1e3); tb.push_back(mb * 1e3); }
        }
        std::printf("%u     %8.1f          %8.1f\n", R, median(ta), median(tb));
    }
    std::printf("mismatching windows %llu\n", static_cast<unsigned long long>(mism));
    std::printf(mism ? "BITWISE FAIL\n" : "BITWISE PASS\n");
    return mism ? 1 : 0;
}


// q8: dense sites of one layer, rows 1 (decode), split 1 / 2 / 4, cold L2
int q8(int argc, char** argv) {
    if (argc < 3) throw std::runtime_error("q8 <model.gguf> [layer] [reps] [rows]");
    const std::uint32_t layer = argc > 3 ? static_cast<std::uint32_t>(std::atoi(argv[3])) : 20U;
    const int reps = argc > 4 ? std::atoi(argv[4]) : 40;
    const std::uint32_t R = argc > 5 ? static_cast<std::uint32_t>(std::atoi(argv[5])) : 1U;
    const auto file = strix::gguf::read_file(argv[2]);
    const auto find = [&](const std::string& name) -> const strix::gguf::TensorInfo& {
        for (const auto& t : file.tensors)
            if (t.name == name) return t;
        throw std::runtime_error("missing tensor " + name);
    };
    struct Site { const char* name; const char* tensor; std::uint32_t rows, width, count, per_group, xwidth, x_stride; };
    const Site sites[] = {
        {"q_a", "attn_q_a.weight", 1280, 5120, 1280, 1280, 5120, 5120},
        {"q_b", "attn_q_b.weight", 32768, 1280, 32768, 32768, 1280, 1280},
        {"o_a", "attn_output_a.weight", 8192, 4096, 8192, 1024, 4096, 32768},
        {"shexp_gate", "ffn_gate_shexp.weight", 2304, 5120, 2304, 2304, 5120, 5120},
        {"shexp_down", "ffn_down_shexp.weight", 5120, 2304, 5120, 5120, 2304, 2304},
        {"o_b_half", "attn_output_b.weight", 5120, 8192, 5120, 5120, 4096, 8192},
    };
    dc::Stream stream;
    const auto st = stream.get();
    dc::DeviceBuffer<int4> flush((256ULL << 20U) / sizeof(int4));
    dc::DeviceBuffer<int> sink(1);
    STRIX_CUDA_CHECK(cudaMemset(flush.get(), 1, flush.size() * sizeof(int4)));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::mt19937 rng(17);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> scale(0.001F, 0.1F);
    std::uint64_t mism = 0;
    std::printf("site        MB      S=1 us   S=2 us   S=4 us   (cold, rows %u; GB/s at S=1)\n", R);
    for (const auto& site : sites) {
        const auto& t = find("blk." + std::to_string(layer) + "." + site.tensor);
        dc::DeviceBuffer<std::uint8_t> raw(t.byte_size);
        {
            std::ifstream in(argv[2], std::ios::binary);
            load(in, t.absolute_offset, t.byte_size, raw.get());
        }
        dc::Q8TiledBuffer tiled;
        tiled.repack(raw.get(), site.rows, site.width, st);
        dc::DeviceBuffer<std::int8_t> xc(std::size_t{R} * site.x_stride);
        dc::DeviceBuffer<float> xd(std::size_t{R} * site.x_stride / 32U), out(std::size_t{R} * site.count), ref(std::size_t{R} * site.count),
            scratch(dc::q8_0_a8_split_scratch_floats(site.count, R, 4));
        dc::DeviceBuffer<std::int32_t> xs(std::size_t{R} * site.x_stride / 32U);
        std::vector<double> t1, t2, t4;
        for (int rep = 0; rep < reps + 2; ++rep) {
            std::vector<std::int8_t> hc(xc.size());
            for (auto& c : hc) c = static_cast<std::int8_t>(code(rng));
            std::vector<float> hd(xd.size());
            for (auto& d : hd) d = scale(rng);
            xc.upload(hc);
            xd.upload(hd);
            for (const std::uint32_t S : {1U, 2U, 4U}) {
                dc::Q8A8View v;
                v.tiled = tiled.view();
                v.count = site.count;
                v.per_group = site.per_group;
                v.width = site.xwidth;
                v.x = {xc.get(), xd.get(), xs.get(), R, site.xwidth, site.x_stride};
                v.out = S == 1U ? ref.get() : out.get();
                v.split = S;
                v.split_scratch = scratch.get();
                v.split_scratch_floats = scratch.size();
                flush_kernel<<<1024, 256, 0, st>>>(flush.get(), flush.size(), sink.get());
                cudaEventRecord(e0, st);
                dc::enqueue_q8_0_a8(v, st);
                cudaEventRecord(e1, st);
                stream.sync();
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                if (rep >= 2) (S == 1U ? t1 : S == 2U ? t2 : t4).push_back(ms * 1e3);
                if (S != 1U) {
                    const auto a = ref.download(ref.size()), b = out.download(out.size());
                    if (std::memcmp(a.data(), b.data(), a.size() * 4U) != 0) ++mism;
                }
            }
        }
        const double mb = static_cast<double>(t.byte_size) / 1e6 * (std::string(site.name) == "o_b_half" ? 0.5 : 1.0);
        std::printf("%-10s %6.1f  %7.1f  %7.1f  %7.1f   %6.0f\n", site.name, mb, median(t1), median(t2), median(t4), mb * 1e3 / median(t1));
    }
    std::printf("mismatches %llu\n", static_cast<unsigned long long>(mism));
    std::printf(mism ? "BITWISE FAIL\n" : "BITWISE PASS\n");
    return mism ? 1 : 0;
}


// hc: hc mixes + norm of one site at decode, 4 arms (partial warps 25/5 x split-norm legacy/cluster), cold L2
int hc(int argc, char** argv) {
    const int reps = argc > 2 ? std::atoi(argv[2]) : 200;
    const std::uint32_t R = argc > 3 ? static_cast<std::uint32_t>(std::atoi(argv[3])) : 1U, H = 5120U;
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> uni(-1.0F, 1.0F);
    std::uniform_int_distribution<int> mant(0, 1023), expo(10, 16), sign(0, 1);
    std::vector<float> hs(std::size_t{R} * 4U * H), hpre(std::size_t{R} * 4U), hw(H), hbase(24), hscale(3);
    for (auto& v : hs) v = uni(rng);
    for (auto& v : hpre) v = uni(rng);
    for (auto& v : hw) v = 1.0F + 0.25F * uni(rng);
    for (auto& v : hbase) v = uni(rng);
    for (auto& v : hscale) v = 0.5F * uni(rng);
    std::vector<std::uint16_t> hfn(std::size_t{24} * 4U * H);
    for (auto& v : hfn) v = static_cast<std::uint16_t>((sign(rng) << 15) | (expo(rng) << 10) | mant(rng));
    dc::DeviceBuffer<float> stream_d(hs.size()), pre(hpre.size()), w(H), base(24), scale(3);
    dc::DeviceBuffer<std::uint16_t> fn(hfn.size());
    stream_d.upload(hs); pre.upload(hpre); w.upload(hw); base.upload(hbase); scale.upload(hscale); fn.upload(hfn);
    dc::DeviceBuffer<std::uint8_t> partial(dc::hc_mix_workspace_bytes(R, H));
    struct Arm { dc::DeviceBuffer<float> mix, collapse, out, deltas; dc::DeviceBuffer<std::int8_t> codes; dc::DeviceBuffer<std::int32_t> sums; };
    std::vector<Arm> arms;
    for (int a = 0; a < 4; ++a)
        arms.push_back({dc::DeviceBuffer<float>(R * 24U), dc::DeviceBuffer<float>(std::size_t{R} * H), dc::DeviceBuffer<float>(std::size_t{R} * H),
                        dc::DeviceBuffer<float>(std::size_t{R} * H / 32U), dc::DeviceBuffer<std::int8_t>(std::size_t{R} * H),
                        dc::DeviceBuffer<std::int32_t>(std::size_t{R} * H / 32U)});
    dc::Stream stream;
    const auto st = stream.get();
    dc::DeviceBuffer<int4> flush((256ULL << 20U) / sizeof(int4));
    dc::DeviceBuffer<int> sink(1);
    STRIX_CUDA_CHECK(cudaMemset(flush.get(), 1, flush.size() * sizeof(int4)));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const auto run = [&](int a) {
        dc::set_hc_partial_warps(a & 1 ? 5U : 25U);
        dc::set_hc_norm_cluster((a & 2) != 0);
        dc::HcMixView m;
        m.stream = stream_d.get(); m.rows = R; m.hidden = H; m.fn = fn.get(); m.base = base.get(); m.scale = scale.get();
        m.mixes = arms[a].mix.get(); m.partial = reinterpret_cast<float*>(partial.get());
        dc::NormView n;
        n.hc_stream = stream_d.get(); n.pre = pre.get(); n.pre_stride = 4U; n.weight = w.get(); n.rows = R; n.n = H;
        n.out = arms[a].out.get(); n.collapsed = arms[a].collapse.get();
        n.a8 = {arms[a].codes.get(), arms[a].deltas.get(), arms[a].sums.get(), R, H, H};
        dc::enqueue_hc_mixes_norm(m, n, st);
    };
    std::vector<double> t[4];
    std::uint64_t mism = 0;
    for (int rep = 0; rep < reps + 2; ++rep)
        for (int k = 0; k < 4; ++k) {
            const int a = (rep & 1) ? 3 - k : k;
            for (auto* b : {arms[a].mix.get(), arms[a].collapse.get(), arms[a].out.get(), arms[a].deltas.get()})
                STRIX_CUDA_CHECK(cudaMemsetAsync(b, 0xff, (b == arms[a].mix.get() ? R * 24U : b == arms[a].deltas.get() ? std::size_t{R} * H / 32U
                                                                                                     : std::size_t{R} * H) * sizeof(float), st));
            STRIX_CUDA_CHECK(cudaMemsetAsync(arms[a].codes.get(), 0x5a, std::size_t{R} * H, st));
            STRIX_CUDA_CHECK(cudaMemsetAsync(arms[a].sums.get(), 0xff, std::size_t{R} * H / 32U * sizeof(std::int32_t), st));
            flush_kernel<<<1024, 256, 0, st>>>(flush.get(), flush.size(), sink.get());
            cudaEventRecord(e0, st);
            run(a);
            cudaEventRecord(e1, st);
            stream.sync();
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            if (rep >= 2) t[a].push_back(ms * 1e3);
            if (k == 3)
                for (int b = 1; b < 4; ++b) {
                    const auto eq = [&](const auto& x, const auto& y) {
                        const auto u = x.download(x.size()), v = y.download(y.size());
                        return std::memcmp(u.data(), v.data(), u.size() * sizeof(u[0])) == 0;
                    };
                    if (!eq(arms[0].mix, arms[b].mix) || !eq(arms[0].collapse, arms[b].collapse) || !eq(arms[0].out, arms[b].out) ||
                        !eq(arms[0].deltas, arms[b].deltas) || !eq(arms[0].codes, arms[b].codes) || !eq(arms[0].sums, arms[b].sums))
                        ++mism;
                }
        }
    dc::set_hc_partial_warps(25U);
    dc::set_hc_norm_cluster(false);
    const char* names[4] = {"partial25+legacy (parent)", "partial5+legacy", "partial25+cluster", "partial5+cluster"};
    std::printf("hc mixes+norm, rows %u, hidden %u, cold L2, %d reps, median us (cluster arms fall back below sm_90)\n", R, H, reps);
    for (int a = 0; a < 4; ++a) std::printf("  %-26s %8.2f\n", names[a], median(t[a]));
    std::printf("mismatches %llu\n", static_cast<unsigned long long>(mism));
    std::printf(mism ? "BITWISE FAIL\n" : "BITWISE PASS\n");
    return mism ? 1 : 0;
}


__global__ void f16_kernel(std::uint32_t* layer, std::uint32_t* attn, std::uint32_t* moe) {
    const std::uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= 65536U) return;
    layer[h] = __float_as_uint(strix::ds41::f16_to_float(static_cast<std::uint16_t>(h)));
    attn[h] = __float_as_uint(strix::ds41::attn::f16_to_float(static_cast<std::uint16_t>(h)));
    moe[h] = __float_as_uint(strix::ds41::cuda::moe_detail::half_bits(h));
}

// f16: all 65536 halves on the device against the previous branchy formulas (host)
int f16(int, char**) {
    const auto bits = [](float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; };
    const auto old_layer = [&](std::uint32_t h) -> std::uint32_t {
        const std::uint32_t sign = (h & 0x8000U) << 16U, e = (h >> 10U) & 0x1fU, m = h & 0x3ffU;
        if (e == 0x1fU) return sign | 0x7f800000U | (m << 13U);
        if (e == 0) {
            if (m == 0) return sign;
            const float v = static_cast<float>(m) * 5.9604644775390625e-8F;
            return bits(sign ? -v : v);
        }
        return sign | ((e + 112U) << 23U) | (m << 13U);
    };
    const auto old_attn = [&](std::uint32_t h) -> std::uint32_t {
        const std::uint32_t sgn = (h >> 15U) << 31U, e = (h >> 10U) & 31U, m = h & 1023U;
        if (e == 0U) { const float v = static_cast<float>(m) * 5.9604644775390625e-08F; return bits(sgn != 0U ? -v : v); }
        if (e == 31U) return sgn | 0x7F800000U | (m << 13U);
        return sgn | ((e + 112U) << 23U) | (m << 13U);
    };
    dc::DeviceBuffer<std::uint32_t> a(65536), b(65536), m(65536);
    f16_kernel<<<256, 256>>>(a.get(), b.get(), m.get());
    STRIX_CUDA_CHECK(cudaDeviceSynchronize());
    const auto ha = a.download(65536), hb = b.download(65536), hm = m.download(65536);
    std::uint64_t mism = 0;
    for (std::uint32_t h = 0; h < 65536U; ++h) {
        mism += ha[h] != old_layer(h);
        mism += hb[h] != old_attn(h);
        mism += hm[h] != old_layer(h);  // moe_detail::half_bits: the host half_to_float bits (== the layer formula)
        mism += bits(strix::ds41::f16_to_float(static_cast<std::uint16_t>(h))) != old_layer(h);
        mism += bits(strix::ds41::attn::f16_to_float(static_cast<std::uint16_t>(h))) != old_attn(h);
    }
    std::printf("f16: 65536 halves x (device, host) x (layer, attention, moe half_bits) conversions, mismatches %llu\n", static_cast<unsigned long long>(mism));
    std::printf(mism ? "BITWISE FAIL\n" : "BITWISE PASS\n");
    return mism ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("usage: strix-cuda-ds41-kern-bench moe ...");
        if (std::strcmp(argv[1], "moe") == 0) return moe(argc, argv);
        if (std::strcmp(argv[1], "router") == 0) return router(argc, argv);
        if (std::strcmp(argv[1], "routes") == 0) return routes(argc, argv);
        if (std::strcmp(argv[1], "ob") == 0) return ob(argc, argv);
        if (std::strcmp(argv[1], "q8") == 0) return q8(argc, argv);
        if (std::strcmp(argv[1], "hc") == 0) return hc(argc, argv);
        if (std::strcmp(argv[1], "f16") == 0) return f16(argc, argv);
        throw std::runtime_error("unknown mode");
    } catch (const std::exception& e) {
        std::printf("FAIL %s\n", e.what());
        return 1;
    }
}
