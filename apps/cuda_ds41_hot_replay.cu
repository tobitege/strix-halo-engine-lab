// deepseek41 dense + hot experts decode replay (CUDA, sm_120): the MoE sites of a 40-layer R1 decode with the routed
// experts placed by a hot-expert placement (static rtx-pro experts + optional dynamic slots refilled from pinned host
// memory after misses) and the others computed by expert servers (strix/ds41-expert-rpc-v1), driven by recorded
// routes token by token. The dense part of each layer is a timed GPU stand-in (--dense-us); the router's ids are
// the recorded ones, the routing weights / A8 input / shared output are synthetic. Measures per token: remote
// layers, remote slots, answer items, RTT, server time, refills, GPU and host time; per layer with --timeline.
// No bitwise check here (strix-cuda-ds41-parts-host-check gates the layer bits; this measures the token).
//
//   strix-cuda-ds41-hot-replay MODEL.gguf PLACEMENT.json ROUTES.txt --connect HOST:PORT[,HOST:PORT] --token-file F
//       [--seqs I,J,..|--seqmod M:K] [--tokens N] [--dense-us 280] [--shared-us 15] [--dyn-gib 0] [--source-gib 40]
//       [--dyn-budget 16] [--busy-poll] [--null-compute] [--no-load] [--out FILE.jsonl] [--timeline FILE.jsonl]
//       [--warmup 4] [--budgets 0,16] [--dense-list 280,200] [--connect-wait-s 30]
// --budgets / --dense-list: run the test sequences once per (copy budget, dense stand-in) pair on the same residency
// (budget 0 = the static placement alone: the dynamic slots stay empty).
// ROUTES: "# seq NAME rows N" headers then one line per token "layer:id,id,.. ..." (the --routes-seqs recording).
// --seqmod M:K: test sequences = index % M == K (the others form the source's frequency fit).
// Dynamic slots (--dyn-gib): split over layers in proportion to static experts + 1; source: the warmest non-static
// experts of the non-test sequences, --source-gib of pinned host memory (or the placement's pc-ram lists if any).

#include "strix/device/cuda_enumerator.hpp"
#include "strix/cuda_ds41_core.hpp"
#include "strix/cuda_ds41_hot_residency.hpp"
#include "strix/cuda_ds41_parts_host_driver.hpp"
#include "strix/ds41_hot_placement.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dc = strix::ds41::cuda;
namespace x = strix::ds41::xrpc;
namespace sd = strix::ds41;

namespace {

__global__ void spin_kernel(unsigned long long ns) {
    unsigned long long t0 = 0, t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}
void spin(double us, cudaStream_t s) {
    if (us <= 0.0) return;
    spin_kernel<<<1, 1, 0, s>>>(static_cast<unsigned long long>(us * 1000.0));
    STRIX_CUDA_CHECK(cudaGetLastError());
}

struct Seq {
    std::string name;
    std::vector<std::vector<std::uint32_t>> rows;  // [token][layer * top_k + k]
};

std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> out;
    std::istringstream is(s);
    std::string v;
    while (std::getline(is, v, c)) if (!v.empty()) out.push_back(v);
    return out;
}

std::vector<Seq> read_seqs(const std::string& path, std::uint32_t layers, std::uint32_t K) {
    std::vector<Seq> seqs;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("# seq", 0) == 0) {
            seqs.push_back({});
            std::istringstream ls(line.substr(5));
            ls >> seqs.back().name;
            continue;
        }
        if (line.empty() || line[0] == '#') continue;
        if (seqs.empty()) seqs.push_back({"routes", {}});
        std::vector<std::uint32_t> row(std::size_t{layers} * K, 0U);
        std::vector<char> seen(layers, 0);
        std::istringstream ls(line);
        std::string item;
        while (ls >> item) {
            const auto colon = item.find(':');
            if (colon == std::string::npos) continue;
            const auto l = static_cast<std::uint32_t>(std::stoul(item.substr(0, colon)));
            if (l >= layers) continue;
            const auto ids = split(item.substr(colon + 1), ',');
            if (ids.size() < K) throw std::runtime_error("short route item");
            for (std::uint32_t k = 0; k < K; ++k) row[std::size_t{l} * K + k] = static_cast<std::uint32_t>(std::stoul(ids[k]));
            seen[l] = 1;
        }
        if (std::count(seen.begin(), seen.end(), 1) != static_cast<long>(layers)) throw std::runtime_error("route line without every layer");
        seqs.back().rows.push_back(std::move(row));
    }
    return seqs;
}

double pct(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<std::size_t>(q * static_cast<double>(v.size())))];
}
double mean(const std::vector<double>& v) { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); }

}  // namespace

int main(int argc, char** argv) {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared card, restricts the process to the chosen one and prints the "device:" receipt line.
    strix::device::bind_cuda_main(argc, argv);
    std::string model, placement_path, routes, connect, token_file, out_path, timeline_path, seqs_arg, seqmod, budgets_arg, dense_arg, layers_arg;
    double connect_wait_s = 30.0;
    std::uint32_t tokens_per_seq = 0, dyn_budget = 16, warmup = 4;
    double dense_us = 280.0, shared_us = 15.0, dyn_gib = 0.0, source_gib = 40.0;
    bool busy_poll = false, null_compute = false, no_load = false;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string s = argv[i];
            const auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
                return argv[++i];
            };
            if (s == "--connect") connect = next();
            else if (s == "--token-file") token_file = next();
            else if (s == "--seqs") seqs_arg = next();
            else if (s == "--seqmod") seqmod = next();
            else if (s == "--tokens") tokens_per_seq = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--dense-us") dense_us = std::stod(next());
            else if (s == "--shared-us") shared_us = std::stod(next());
            else if (s == "--dyn-gib") dyn_gib = std::stod(next());
            else if (s == "--source-gib") source_gib = std::stod(next());
            else if (s == "--dyn-budget") dyn_budget = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--warmup") warmup = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--budgets") budgets_arg = next();
            else if (s == "--layers") layers_arg = next();
            else if (s == "--dense-list") dense_arg = next();
            else if (s == "--connect-wait-s") connect_wait_s = std::stod(next());
            else if (s == "--busy-poll") busy_poll = true;
            else if (s == "--null-compute") null_compute = true;
            else if (s == "--no-load") no_load = true;
            else if (s == "--out") out_path = next();
            else if (s == "--timeline") timeline_path = next();
            else if (!s.empty() && s[0] == '-') throw std::invalid_argument("unknown option " + s);
            else if (model.empty()) model = s;
            else if (placement_path.empty()) placement_path = s;
            else if (routes.empty()) routes = s;
            else throw std::invalid_argument("unexpected argument " + s);
        }
        if (routes.empty()) throw std::invalid_argument("usage: MODEL.gguf PLACEMENT.json ROUTES.txt [options]");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
    try {
        const auto device = dc::select_device(0);
        std::printf("device %s (cc %d.%d, %d SMs)\n", device.name.c_str(), device.major, device.minor, device.sms);
        const auto t_load0 = std::chrono::steady_clock::now();
        dc::GgufExpertReader reader(model);
        auto shape = reader.shape();
        if (shape.top_k == 0) shape.top_k = 6;
        const std::uint32_t E = shape.experts, K = shape.top_k, H = shape.hidden;
        sd::HotPlacementExpect ex;
        ex.geometry.experts = E;
        ex.geometry.top_k = K;
        ex.geometry.canonical_split = E / 2U;
        ex.geometry.expert_bytes = shape.expert_bytes();
        ex.geometry.layers = 40;
        const auto placement = sd::load_hot_placement(placement_path, ex);
        const std::uint32_t L = placement.geometry.layers;
        auto seqs = read_seqs(routes, L, K);

        // test sequences
        std::vector<char> test(seqs.size(), 0);
        if (!seqs_arg.empty()) for (const auto& v : split(seqs_arg, ',')) test.at(std::stoul(v)) = 1;
        else if (!seqmod.empty()) {
            const auto mk = split(seqmod, ':');
            const auto M = std::stoul(mk.at(0)), Kf = std::stoul(mk.at(1));
            for (std::size_t i = 0; i < seqs.size(); ++i) test[i] = (i % M) == Kf ? 1 : 0;
        } else std::fill(test.begin(), test.end(), 1);

        // dynamic slots and source
        dc::HotResidencyOptions ho;
        ho.shape = shape;
        if (layers_arg.empty()) for (std::uint32_t l = 0; l < L; ++l) ho.layers.push_back(l);
        else for (const auto& v : split(layers_arg, ',')) ho.layers.push_back(static_cast<std::uint32_t>(std::stoul(v)));
        std::vector<char> held(L, 0);
        for (const auto l : ho.layers) held.at(l) = 1;
        ho.copies_per_token = dyn_budget;
        ho.load_weights = !no_load;
        const std::uint64_t eb = shape.expert_bytes();
        if (dyn_gib > 0.0) {
            const auto n_dyn = static_cast<std::uint64_t>(dyn_gib * 1073741824.0 / static_cast<double>(eb));
            std::vector<std::uint64_t> cnt(L);
            std::uint64_t sum = 0;
            for (std::uint32_t l = 0; l < L; ++l) { cnt[l] = placement.layers[l].experts[0].size() + 1U; sum += cnt[l]; }
            ho.dynamic_slots.assign(L, 0U);
            for (std::uint32_t l = 0; l < L; ++l)
                ho.dynamic_slots[l] = static_cast<std::uint32_t>(std::min<std::uint64_t>(n_dyn * cnt[l] / sum, E - cnt[l] + 1U));
            ho.source.assign(L, {});
            bool listed = false;
            for (std::uint32_t l = 0; l < L; ++l) {
                for (const auto e : placement.layers[l].experts[static_cast<std::size_t>(sd::HotDevice::pc_ram)]) { ho.source[l].push_back(e); listed = true; }
            }
            if (!listed) {  // the warmest non-static experts of the non-test sequences
                std::vector<double> freq(std::size_t{L} * E, 0.0);
                for (std::size_t i = 0; i < seqs.size(); ++i)
                    if (!test[i])
                        for (const auto& row : seqs[i].rows)
                            for (std::uint32_t l = 0; l < L; ++l)
                                for (std::uint32_t k = 0; k < K; ++k) freq[std::size_t{l} * E + row[std::size_t{l} * K + k]] += 1.0;
                std::vector<std::uint32_t> idx;
                for (std::uint32_t l = 0; l < L; ++l)
                    for (std::uint32_t e = 0; e < E; ++e)
                        if (held[l] && placement.device_of(l, e) != sd::HotDevice::rtx_pro && ho.dynamic_slots[l] > 0) idx.push_back(l * E + e);
                std::stable_sort(idx.begin(), idx.end(), [&](std::uint32_t a, std::uint32_t b) { return freq[a] > freq[b]; });
                const auto n_src = std::min<std::uint64_t>(idx.size(), static_cast<std::uint64_t>(source_gib * 1073741824.0 / static_cast<double>(eb)));
                idx.resize(n_src);
                std::sort(idx.begin(), idx.end());
                for (const auto v : idx) ho.source[v / E].push_back(v % E);
            }
        }
        dc::HotResidency res(placement, reader, ho);
        std::uint32_t statics = 0, dyns = 0;
        for (const auto l : ho.layers) { statics += res.static_slots(l); dyns += res.dynamic_slots(l); }
        const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
        std::printf("residency: %u static experts + %u dynamic slots = %.2f GB on the device, source %.2f GB pinned, budget %u copies/token, loaded in %.1f s%s\n",
                    statics, dyns, static_cast<double>(res.device_bytes()) / 1e9, static_cast<double>(res.source_bytes()) / 1e9, dyn_budget, load_s,
                    no_load ? " (weights NOT loaded)" : "");

        dc::PartsHostOptions po;
        po.shape = shape;
        po.split = E / 2U;
        po.capacity = 1;
        po.endpoints = split(connect, ',');
        po.busy_poll = busy_poll;
        po.connect_wait = std::chrono::milliseconds(static_cast<long long>(connect_wait_s * 1000.0));
        po.request_flags = null_compute ? x::kNullCompute : 0U;
        po.hello.geometry.layers = L;
        po.hello.geometry.experts = E;
        po.hello.geometry.top_k = K;
        po.hello.geometry.hidden = H;
        po.hello.geometry.inter = shape.inter;
        po.hello.geometry.split = E / 2U;
        if (!token_file.empty()) po.hello.token = x::read_token_file(token_file);
        dc::PartsHost host(po);
        for (std::size_t i = 0; i < host.acks().size(); ++i) {
            const auto& ack = host.acks()[i];
            std::printf("server %s: engine %s, experts [%u, %u), layer mask 0x%llx, numeric %s\n", po.endpoints[i].c_str(), ack.engine.c_str(),
                        ack.residency.expert_first, ack.residency.expert_end, static_cast<unsigned long long>(ack.residency.layer_mask),
                        ack.numeric_identity.c_str());
        }
        std::fflush(stdout);

        // buffers: one R1 window per layer; ids / weights of the whole token uploaded at its start
        dc::DeviceBuffer<std::uint32_t> d_ids(std::size_t{L} * K);
        dc::DeviceBuffer<float> d_w(std::size_t{L} * K), xf(H), shared(H), raw(std::size_t{K} * H), y(H);
        std::uint32_t* h_ids = nullptr;
        float* h_w = nullptr;
        STRIX_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_ids), std::size_t{L} * K * 4U, cudaHostAllocPortable));
        STRIX_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_w), std::size_t{L} * K * 4U, cudaHostAllocPortable));
        dc::A8Buffer xa;
        xa.resize(1, H);
        std::mt19937 rng(20261021);
        std::normal_distribution<float> nd(0.0F, 1.0F);
        {
            std::vector<float> hx(H), hs(H);
            for (auto& v : hx) v = nd(rng) * 0.5F;
            for (auto& v : hs) v = nd(rng) * 0.05F;
            xf.upload(hx.data(), H);
            shared.upload(hs.data(), H);
        }
        dc::Stream stream;
        dc::QuantizeA8View q;
        q.x = xf.get();
        q.rows = 1;
        q.width = H;
        q.out = xa.view();
        q.out.rows = 1;
        dc::enqueue_quantize_a8(q, stream.get());
        stream.sync();
        const dc::A8Rows x8{xa.codes.get(), xa.deltas.get(), xa.sums.get(), 1, H, H};
        std::vector<cudaEvent_t> ev(L + 1U), ready(L);
        for (auto& e : ev) STRIX_CUDA_CHECK(cudaEventCreate(&e));
        for (auto& e : ready) STRIX_CUDA_CHECK(cudaEventCreate(&e));

        std::FILE* out = out_path.empty() ? nullptr : std::fopen(out_path.c_str(), "w");
        std::FILE* tl = timeline_path.empty() ? nullptr : std::fopen(timeline_path.c_str(), "w");
        std::vector<std::uint32_t> budgets{dyn_budget};
        std::vector<double> denses{dense_us};
        if (!budgets_arg.empty()) { budgets.clear(); for (const auto& v : split(budgets_arg, ',')) budgets.push_back(static_cast<std::uint32_t>(std::stoul(v))); }
        if (!dense_arg.empty()) { denses.clear(); for (const auto& v : split(dense_arg, ',')) denses.push_back(std::stod(v)); }
        for (const double dense_cfg : denses)
        for (const std::uint32_t budget_cfg : budgets) {
        const double dense_us = dense_cfg;
        const std::uint32_t dyn_budget = budget_cfg;
        res.set_copies_per_token(dyn_budget);
        std::vector<double> tok_ms, host_ms, rl_tok, slots_tok, items_tok, rtt_all, rtt1, rtt2, srv_all, pen_tok, copies_tok, hits_tok, span_local, span_remote;
        std::uint64_t measured = 0;
        for (std::size_t si = 0; si < seqs.size(); ++si) {
            if (!test[si]) continue;
            const auto& sq = seqs[si];
            res.reset_dynamic();
            const std::size_t n = tokens_per_seq ? std::min<std::size_t>(tokens_per_seq, sq.rows.size()) : sq.rows.size();
            for (std::size_t ti = 0; ti < n; ++ti) {
                const auto& row = sq.rows[ti];
                std::copy(row.begin(), row.end(), h_ids);
                for (std::uint32_t l = 0; l < L; ++l) {
                    float sum = 0.0F;
                    for (std::uint32_t k = 0; k < K; ++k) { h_w[l * K + k] = std::uniform_real_distribution<float>(0.05F, 1.0F)(rng); sum += h_w[l * K + k]; }
                    for (std::uint32_t k = 0; k < K; ++k) h_w[l * K + k] = h_w[l * K + k] / sum * 2.5F;
                }
                res.begin_token();
                const auto th0 = std::chrono::steady_clock::now();
                STRIX_CUDA_CHECK(cudaMemcpyAsync(d_ids.get(), h_ids, std::size_t{L} * K * 4U, cudaMemcpyHostToDevice, stream.get()));
                STRIX_CUDA_CHECK(cudaMemcpyAsync(d_w.get(), h_w, std::size_t{L} * K * 4U, cudaMemcpyHostToDevice, stream.get()));
                STRIX_CUDA_CHECK(cudaEventRecord(ev[0], stream.get()));
                std::vector<dc::PartsHostTiming> lt(L);
                for (std::uint32_t l = 0; l < L; ++l) {
                    if (!held[l]) { STRIX_CUDA_CHECK(cudaEventRecord(ev[l + 1U], stream.get())); continue; }
                    spin(dense_us, stream.get());  // the layer's dense part up to the router (stand-in)
                    STRIX_CUDA_CHECK(cudaEventRecord(ready[l], stream.get()));
                    res.before_layer(l, stream.get());
                    dc::PartsHost::Window w;
                    w.rows = 1;
                    w.x = x8;
                    w.ids = d_ids.get() + std::size_t{l} * K;
                    w.weights = d_w.get() + std::size_t{l} * K;
                    w.shared = shared.get();
                    w.raw = raw.get();
                    w.y = y.get();
                    w.router_ready = ready[l];
                    lt[l] = host.run_layer(res.layer(l), w, stream.get(), false, [&](cudaStream_t s) { spin(shared_us, s); });
                    res.after_layer(l, host.last_host_ids(), K, host.last_done_event());
                    STRIX_CUDA_CHECK(cudaEventRecord(ev[l + 1U], stream.get()));
                }
                stream.sync();
                const double hms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - th0).count();
                float gms = 0.0F;
                STRIX_CUDA_CHECK(cudaEventElapsedTime(&gms, ev[0], ev[L]));
                std::uint32_t rlayers = 0, slots = 0, items = 0;
                double rtt_sum = 0.0;
                std::vector<double> spans(L);
                for (std::uint32_t l = 0; l < L; ++l) {
                    float ms = 0.0F;
                    STRIX_CUDA_CHECK(cudaEventElapsedTime(&ms, ev[l], ev[l + 1U]));
                    spans[l] = ms * 1000.0;
                    if (!held[l]) continue;
                    if (lt[l].remote_slots) { ++rlayers; slots += lt[l].remote_slots; items += lt[l].items; rtt_sum += lt[l].rtt_us; }
                }
                const auto& st = res.token_stats();
                const bool warm = ti >= warmup;
                if (warm) {
                    ++measured;
                    tok_ms.push_back(gms);
                    host_ms.push_back(hms);
                    rl_tok.push_back(rlayers);
                    slots_tok.push_back(slots);
                    items_tok.push_back(items);
                    copies_tok.push_back(st.copies);
                    hits_tok.push_back(st.dynamic_hits);
                    for (std::uint32_t l = 0; l < L; ++l) {
                        if (!held[l]) continue;
                        if (lt[l].remote_slots) {
                            rtt_all.push_back(lt[l].rtt_us);
                            srv_all.push_back(lt[l].server_us);
                            (lt[l].items <= 1 ? rtt1 : rtt2).push_back(lt[l].rtt_us);
                            span_remote.push_back(spans[l]);
                        } else {
                            span_local.push_back(spans[l]);
                        }
                    }
                }
                if (out)
                    std::fprintf(out, "{\"budget\":%u,\"dense_us\":%.0f,\"seq\":%zu,\"name\":\"%s\",\"t\":%zu,\"warm\":%s,\"gpu_ms\":%.3f,\"host_ms\":%.3f,\"remote_layers\":%u,\"remote_slots\":%u,\"items\":%u,"
                                 "\"rtt_sum_us\":%.1f,\"dyn_hits\":%u,\"copies\":%u,\"not_in_source\":%u,\"over_budget\":%u}\n",
                                 dyn_budget, dense_us, si, sq.name.c_str(), ti, warm ? "true" : "false", gms, hms, rlayers, slots, items, rtt_sum, st.dynamic_hits, st.copies, st.not_in_source,
                                 st.over_budget);
                if (tl && warm)
                    for (std::uint32_t l = 0; l < L; ++l)
                        if (held[l]) std::fprintf(tl, "{\"budget\":%u,\"dense_us\":%.0f,\"seq\":%zu,\"t\":%zu,\"layer\":%u,\"span_us\":%.1f,\"local\":%u,\"remote\":%u,\"items\":%u,\"servers\":%u,\"enqueue\":%.1f,\"host_wait\":%.1f,"
                                     "\"prep\":%.1f,\"rtt\":%.1f,\"server\":%.1f,\"pack\":%.1f,\"plan\":%.1f,\"local_first\":%s}\n",
                                     dyn_budget, dense_us, si, ti, l, spans[l], lt[l].local_slots, lt[l].remote_slots, lt[l].items, lt[l].servers, lt[l].enqueue_us, lt[l].host_wait_us,
                                     lt[l].prep_us, lt[l].rtt_us, lt[l].server_us, lt[l].pack_us, lt[l].plan_us, lt[l].local_done_before_answers ? "true" : "false");
            }
            std::printf("seq %zu %s: %zu tokens\n", si, sq.name.c_str(), n);
            std::fflush(stdout);
        }
        const double lspan = pct(span_local, 0.5);
        std::printf("{\"replay\":\"ds41-hot\",\"placement\":\"%s\",\"tokens\":%llu,\"dense_us\":%.1f,\"shared_us\":%.1f,\"dyn_slots\":%u,\"source_gb\":%.2f,\"dyn_budget\":%u,"
                    "\"endpoints\":%zu,\"null_compute\":%s,\"gpu_ms\":{\"mean\":%.3f,\"p50\":%.3f,\"p90\":%.3f},\"host_ms_mean\":%.3f,\"tok_s\":%.2f,"
                    "\"remote_layers\":%.2f,\"remote_slots\":%.2f,\"items\":%.2f,\"dyn_hits\":%.2f,\"copies\":%.2f,"
                    "\"rtt_us\":{\"p50\":%.1f,\"p90\":%.1f,\"p99\":%.1f,\"p50_1item\":%.1f,\"p50_2plus\":%.1f},\"server_us_p50\":%.1f,"
                    "\"span_us\":{\"local_p50\":%.1f,\"remote_p50\":%.1f,\"remote_p90\":%.1f},\"remote_penalty_us_per_layer\":%.1f,"
                    "\"all_local_token_ms_est\":%.3f}\n",
                    placement_path.c_str(), static_cast<unsigned long long>(measured), dense_us, shared_us, dyns, static_cast<double>(res.source_bytes()) / 1e9, dyn_budget,
                    po.endpoints.size(), null_compute ? "true" : "false", mean(tok_ms), pct(tok_ms, 0.5), pct(tok_ms, 0.9), mean(host_ms),
                    tok_ms.empty() ? 0.0 : 1000.0 / mean(tok_ms), mean(rl_tok), mean(slots_tok), mean(items_tok), mean(hits_tok), mean(copies_tok), pct(rtt_all, 0.5),
                    pct(rtt_all, 0.9), pct(rtt_all, 0.99), pct(rtt1, 0.5), pct(rtt2, 0.5), pct(srv_all, 0.5), lspan, pct(span_remote, 0.5), pct(span_remote, 0.9),
                    pct(span_remote, 0.5) - lspan, lspan * static_cast<double>(ho.layers.size()) / 1000.0);
        }
        if (out) std::fclose(out);
        if (tl) std::fclose(tl);
        return 0;
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
}
