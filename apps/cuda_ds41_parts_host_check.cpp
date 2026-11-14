// deepseek41 parts-host gate (CUDA, sm_120): one real MoE layer with a resident subset of its routed experts on this
// device and the others computed by expert servers (strix/ds41-expert-rpc-v1), against the same layer with all
// experts resident: y and the local slots' raw vectors must be bitwise equal. Prints one JSON line per window size
// with the parts-host timings (medians).
//
//   strix-cuda-ds41-parts-host-check EXPERTS.gguf ROUTES.txt [--layer 20] [--rows 1,2,4,8] [--windows 3] [--warmup 1]
//       [--hot-fraction 0.5] [--hot-pick freq|hash] [--connect HOST:PORT[,HOST:PORT]] [--token-file F] [--busy-poll]
//       [--seed N] [--route-offset N] [--null-compute] [--copies] [--in-place]
//       [--placement PLACEMENT.json [--dyn-slots D] [--dyn-budget N] [--consecutive]]
// --placement: the layer's resident experts are the placement's rtx-pro experts (tensormeld/hot-expert-placement-v1,
// slot order of the file) instead of --hot-fraction/--hot-pick. --dyn-slots D adds D dynamic slots refilled after
// each window from a pinned host copy of the layer's other experts (HotResidency, at most --dyn-budget copies per
// window); --consecutive takes the windows' tokens consecutively from ROUTES (decode order) instead of a stride of 37.
// --copies: event + cudaMemcpyAsync D2H instead of the export kernel and flag. --in-place: the combine reads the
// answers from mapped host memory (no H2D).
// --null-compute: the servers answer zeros without computing (wire and parts-host overheads only; no bitwise check).
// ROUTES: recorded routes ("layer:id,id,..." items per token line). --hot-pick freq keeps the experts most often
// routed in ROUTES (the windows come from the same file: an in-sample residency), hash a seeded random subset.

#include "strix/device/cuda_enumerator.hpp"
#include "strix/cuda_ds41_core.hpp"
#include "strix/cuda_ds41_parts_host_driver.hpp"
#include "strix/cuda_ds41_hot_residency.hpp"
#include "strix/ds41_hot_placement.hpp"
#include "strix/gguf_reader.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dc = strix::ds41::cuda;
namespace x = strix::ds41::xrpc;

namespace {

struct Args {
    std::string model, routes, connect, token_file, hot_pick = "freq", placement;
    std::uint32_t dyn_slots = 0, dyn_budget = 6;
    bool consecutive = false;
    std::uint32_t layer = 20, windows = 3, warmup = 1, seed = 20261021, route_offset = 0;
    std::vector<std::uint32_t> rows{1, 2, 4, 8};
    double hot_fraction = 0.5;
    bool busy_poll = false, null_compute = false, copies = false, inplace = false;
};

std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> out;
    std::istringstream is(s);
    std::string v;
    while (std::getline(is, v, c)) if (!v.empty()) out.push_back(v);
    return out;
}

std::vector<std::vector<std::uint32_t>> read_routes(const std::string& path, std::uint32_t layer, std::uint32_t top_k) {
    std::vector<std::vector<std::uint32_t>> tokens;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line[0] == '#') continue;  // "# seq NAME rows N" headers of a --routes-seqs recording
        std::istringstream ls(line);
        std::string item;
        while (ls >> item) {
            const auto colon = item.find(':');
            if (colon == std::string::npos || std::stoul(item.substr(0, colon)) != layer) continue;
            std::vector<std::uint32_t> ids;
            for (const auto& v : split(item.substr(colon + 1), ',')) ids.push_back(static_cast<std::uint32_t>(std::stoul(v)));
            if (ids.size() >= top_k) { ids.resize(top_k); tokens.push_back(ids); }
        }
    }
    return tokens;
}

void load(std::ifstream& in, std::uint64_t offset, std::uint64_t bytes, std::uint8_t* device) {
    constexpr std::uint64_t chunk = 64ULL << 20U;
    std::uint8_t* host = nullptr;
    STRIX_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host), chunk));
    in.seekg(static_cast<std::streamoff>(offset));
    for (std::uint64_t at = 0; at < bytes; at += chunk) {
        const auto n = std::min(chunk, bytes - at);
        in.read(reinterpret_cast<char*>(host), static_cast<std::streamsize>(n));
        if (!in) throw std::runtime_error("short read of the expert tensors");
        STRIX_CUDA_CHECK(cudaMemcpy(device + at, host, n, cudaMemcpyHostToDevice));
    }
    STRIX_CUDA_CHECK(cudaFreeHost(host));
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const auto h = v.size() / 2;
    return v.size() % 2 ? v[h] : (v[h - 1] + v[h]) / 2.0;
}
double mean(const std::vector<double>& v) { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); }

}  // namespace

int main(int argc, char** argv) {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared card, restricts the process to the chosen one and prints the "device:" receipt line.
    strix::device::bind_cuda_main(argc, argv);
    Args a;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string s = argv[i];
            const auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
                return argv[++i];
            };
            if (s == "--layer") a.layer = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--rows") { a.rows.clear(); for (const auto& v : split(next(), ',')) a.rows.push_back(static_cast<std::uint32_t>(std::stoul(v))); }
            else if (s == "--windows") a.windows = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--warmup") a.warmup = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--hot-fraction") a.hot_fraction = std::stod(next());
            else if (s == "--hot-pick") a.hot_pick = next();
            else if (s == "--connect") a.connect = next();
            else if (s == "--token-file") a.token_file = next();
            else if (s == "--busy-poll") a.busy_poll = true;
            else if (s == "--null-compute") a.null_compute = true;
            else if (s == "--copies") a.copies = true;
            else if (s == "--in-place") a.inplace = true;
            else if (s == "--placement") a.placement = next();
            else if (s == "--dyn-slots") a.dyn_slots = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--dyn-budget") a.dyn_budget = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--consecutive") a.consecutive = true;
            else if (s == "--seed") a.seed = static_cast<std::uint32_t>(std::stoul(next()));
            else if (s == "--route-offset") a.route_offset = static_cast<std::uint32_t>(std::stoul(next()));
            else if (!s.empty() && s[0] == '-') throw std::invalid_argument("unknown option " + s);
            else if (a.model.empty()) a.model = s;
            else if (a.routes.empty()) a.routes = s;
            else throw std::invalid_argument("unexpected argument " + s);
        }
        if (a.model.empty() || a.routes.empty() || a.rows.empty()) throw std::invalid_argument("usage: EXPERTS.gguf ROUTES.txt [options]");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
    try {
        const auto device = dc::select_device(0);
        std::printf("device %s (cc %d.%d, %d SMs)\n", device.name.c_str(), device.major, device.minor, device.sms);
        const auto file = strix::gguf::read_file(a.model);
        const auto find = [&](const std::string& name) -> const strix::gguf::TensorInfo& {
            for (const auto& t : file.tensors)
                if (t.name == name) return t;
            throw std::runtime_error("missing tensor " + name);
        };
        // the geometry as the expert server derives it from the same file
        x::Geometry geo;
        std::uint32_t layers = 0;
        for (const auto& t : file.tensors) {
            if (t.name.rfind("blk.", 0) != 0 || t.name.find("_exps.weight") == std::string::npos) continue;
            const auto dot = t.name.find('.', 4);
            layers = std::max(layers, static_cast<std::uint32_t>(std::stoul(t.name.substr(4, dot - 4))) + 1U);
        }
        const std::string p = "blk." + std::to_string(a.layer) + ".";
        const auto& tg = find(p + "ffn_gate_exps.weight");
        const auto& tu = find(p + "ffn_up_exps.weight");
        const auto& td = find(p + "ffn_down_exps.weight");
        if (tg.ggml_type != 16 || tu.ggml_type != 16 || td.ggml_type != 10) throw std::runtime_error("expected IQ2_XXS gate/up and Q2_K down");
        geo.hidden = static_cast<std::uint32_t>(tg.dimensions[0]);
        geo.inter = static_cast<std::uint32_t>(tg.dimensions[1]);
        geo.experts = static_cast<std::uint32_t>(tg.dimensions[2]);
        if (const auto* m = file.find_metadata("deepseek41.num_hidden_layers"); m && m->unsigned_value) layers = static_cast<std::uint32_t>(*m->unsigned_value);
        if (const auto* m = file.find_metadata("deepseek41.num_experts_per_tok"); m && m->unsigned_value) geo.top_k = static_cast<std::uint32_t>(*m->unsigned_value);
        geo.layers = layers;
        geo.split = geo.experts / 2U;
        dc::MoeShape shape;
        shape.hidden = geo.hidden;
        shape.inter = geo.inter;
        shape.experts = geo.experts;
        shape.top_k = geo.top_k;
        const std::uint32_t E = shape.experts, K = shape.top_k, H = shape.hidden;

        // all experts resident: the reference
        dc::DeviceBuffer<std::uint8_t> gate(tg.byte_size), up(tu.byte_size), down(td.byte_size);
        {
            std::ifstream in(a.model, std::ios::binary);
            load(in, tg.absolute_offset, tg.byte_size, gate.get());
            load(in, tu.absolute_offset, tu.byte_size, up.get());
            load(in, td.absolute_offset, td.byte_size, down.get());
        }
        const auto tokens = read_routes(a.routes, a.layer, K);
        if (tokens.empty()) throw std::runtime_error("no routes for the layer");

        // residency
        std::unique_ptr<strix::ds41::HotPlacement> placement;
        std::unique_ptr<dc::GgufExpertReader> reader;
        std::unique_ptr<dc::HotResidency> hotres;
        dc::ResidentLayer rl_own;
        const dc::ResidentLayer* rlp = &rl_own;
        std::uint32_t hot = 0;
        dc::DeviceBuffer<std::uint8_t> rgate, rup, rdown;
        dc::DeviceBuffer<std::int32_t> d_slot_of;
        std::uint64_t covered = 0, all = 0;
        if (!a.placement.empty()) {
            strix::ds41::HotPlacementExpect ex;
            ex.geometry.layers = layers;
            ex.geometry.experts = E;
            ex.geometry.top_k = K;
            ex.geometry.canonical_split = geo.split;
            ex.geometry.expert_bytes = shape.expert_bytes();
            placement = std::make_unique<strix::ds41::HotPlacement>(strix::ds41::load_hot_placement(a.placement, ex));
            reader = std::make_unique<dc::GgufExpertReader>(a.model);
            dc::HotResidencyOptions ho;
            ho.shape = shape;
            ho.layers = {a.layer};
            ho.copies_per_token = a.dyn_budget;
            if (a.dyn_slots) {
                ho.dynamic_slots.assign(layers, 0U);
                ho.dynamic_slots[a.layer] = a.dyn_slots;
                ho.source.assign(layers, {});
                for (std::uint32_t e = 0; e < E; ++e)
                    if (placement->device_of(a.layer, e) != strix::ds41::HotDevice::rtx_pro) ho.source[a.layer].push_back(e);
            }
            hotres = std::make_unique<dc::HotResidency>(*placement, *reader, ho);
            rlp = &hotres->layer(a.layer);
            hot = hotres->static_slots(a.layer);
            for (const auto& t : tokens) for (const auto e : t) { ++all; covered += rlp->host_slot_of[e] >= 0 ? 1U : 0U; }
            std::printf("layer %u: placement %s: %u static experts + %u dynamic slots (source %.2f GB pinned, budget %u/window), static route coverage of the file %.3f, %zu tokens\n",
                        a.layer, a.placement.c_str(), hot, hotres->dynamic_slots(a.layer), static_cast<double>(hotres->source_bytes()) / 1e9, a.dyn_budget,
                        all ? static_cast<double>(covered) / static_cast<double>(all) : 0.0, tokens.size());
        } else {
            std::vector<std::uint64_t> freq(E, 0);
            for (const auto& t : tokens) for (const auto e : t) if (e < E) ++freq[e];
            std::vector<std::uint32_t> pick(E);
            std::iota(pick.begin(), pick.end(), 0U);
            if (a.hot_pick == "freq") std::stable_sort(pick.begin(), pick.end(), [&](std::uint32_t p1, std::uint32_t p2) { return freq[p1] > freq[p2]; });
            else if (a.hot_pick == "hash") { std::mt19937 r(a.seed ^ 0x5eedU); std::shuffle(pick.begin(), pick.end(), r); }
            else throw std::invalid_argument("--hot-pick freq|hash");
            hot = static_cast<std::uint32_t>(std::lround(std::clamp(a.hot_fraction, 0.0, 1.0) * E));
            std::vector<std::uint32_t> resident(pick.begin(), pick.begin() + hot);
            std::sort(resident.begin(), resident.end());
            auto& rl = rl_own;
            rl.layer = a.layer;
            rl.host_slot_of.assign(E, -1);
            for (std::uint32_t i = 0; i < hot; ++i) rl.host_slot_of[resident[i]] = static_cast<std::int32_t>(i);
            for (const auto& t : tokens) for (const auto e : t) { ++all; covered += rl.host_slot_of[e] >= 0 ? 1U : 0U; }
            const std::size_t gb = shape.gate_expert_bytes(), db = shape.down_expert_bytes();
            rgate.resize(std::max<std::size_t>(1, gb * hot));
            rup.resize(std::max<std::size_t>(1, gb * hot));
            rdown.resize(std::max<std::size_t>(1, db * hot));
            for (std::uint32_t i = 0; i < hot; ++i) {
                const std::size_t e = resident[i];
                STRIX_CUDA_CHECK(cudaMemcpy(rgate.get() + i * gb, gate.get() + e * gb, gb, cudaMemcpyDeviceToDevice));
                STRIX_CUDA_CHECK(cudaMemcpy(rup.get() + i * gb, up.get() + e * gb, gb, cudaMemcpyDeviceToDevice));
                STRIX_CUDA_CHECK(cudaMemcpy(rdown.get() + i * db, down.get() + e * db, db, cudaMemcpyDeviceToDevice));
            }
            d_slot_of.upload(rl.host_slot_of);
            rl.weights = {rgate.get(), rup.get(), rdown.get()};
            rl.slot_of = d_slot_of.get();
            std::printf("layer %u: %u experts resident of %u (%.2f GB of %.2f GB, pick %s), route coverage of the file %.3f, %zu tokens\n", a.layer,
                        hot, E, static_cast<double>(hot) * static_cast<double>(shape.expert_bytes()) / 1e9,
                        static_cast<double>(E) * static_cast<double>(shape.expert_bytes()) / 1e9, a.hot_pick.c_str(),
                        static_cast<double>(covered) / static_cast<double>(all), tokens.size());
        }
        // parts-host
        const std::uint32_t cap = *std::max_element(a.rows.begin(), a.rows.end());
        dc::PartsHostOptions po;
        po.shape = shape;
        po.split = geo.split;
        po.capacity = cap;
        po.endpoints = split(a.connect, ',');
        po.busy_poll = a.busy_poll;
        po.export_flag = !a.copies;
        po.answers_in_place = a.inplace;
        po.request_flags = a.null_compute ? x::kNullCompute : 0U;
        po.hello.geometry = geo;
        if (!a.token_file.empty()) po.hello.token = x::read_token_file(a.token_file);
        else if (!po.endpoints.empty()) throw std::invalid_argument("--connect needs --token-file");
        dc::PartsHost host(po);
        for (std::size_t i = 0; i < host.acks().size(); ++i) {
            const auto& ack = host.acks()[i];
            std::printf("server %s: engine %s, experts [%u, %u), layer mask 0x%llx, world %u, max rows %u, numeric %s\n", po.endpoints[i].c_str(),
                        ack.engine.c_str(), ack.residency.expert_first, ack.residency.expert_end,
                        static_cast<unsigned long long>(ack.residency.layer_mask), ack.world, ack.max_rows, ack.numeric_identity.c_str());
        }
        std::fflush(stdout);

        // window buffers
        dc::DeviceBuffer<std::uint32_t> ids(std::size_t{cap} * K);
        dc::DeviceBuffer<float> rw(std::size_t{cap} * K), xf(std::size_t{cap} * H), shared(std::size_t{cap} * H);
        dc::DeviceBuffer<float> raw_ref(std::size_t{cap} * K * H), y_ref(std::size_t{cap} * H), raw_h(std::size_t{cap} * K * H), y_h(std::size_t{cap} * H);
        dc::A8Buffer xa;
        xa.resize(cap, H);
        dc::DeviceBuffer<std::uint8_t> ws(dc::moe_layer_workspace_bytes(cap, shape));
        dc::Stream stream;
        cudaEvent_t router_ready, r0, r1;
        STRIX_CUDA_CHECK(cudaEventCreate(&router_ready));
        STRIX_CUDA_CHECK(cudaEventCreate(&r0));
        STRIX_CUDA_CHECK(cudaEventCreate(&r1));
        std::mt19937 rng(a.seed);
        std::normal_distribution<float> nd(0.0F, 1.0F);
        bool all_ok = true;
        std::uint32_t win_counter = a.route_offset;
        std::size_t consec_at = a.route_offset;
        for (const std::uint32_t R : a.rows) {
            std::vector<double> enq, pack, d2h, wait_us, prep, rtt, server, plan, h2d, comb, local, total, overlap, ref_us, remote_slots, local_slots, items, qbytes,
                abytes, servers;
            bool pinned = false;
            std::uint64_t dyn_hits = 0, dyn_copies = 0;
            std::uint32_t y_bad = 0, raw_bad = 0, local_first = 0, checked = 0;
            for (std::uint32_t w = 0; w < a.warmup + a.windows; ++w, ++win_counter) {
                // the window: R consecutive recorded tokens, random A8 input / weights / shared output
                std::vector<std::uint32_t> h_ids;
                const std::size_t t0 = a.consecutive ? consec_at % tokens.size() : (std::size_t{win_counter} * 37U) % tokens.size();
                if (a.consecutive) consec_at += R;
                for (std::uint32_t r = 0; r < R; ++r) {
                    const auto& t = tokens[(t0 + r) % tokens.size()];
                    h_ids.insert(h_ids.end(), t.begin(), t.end());
                }
                std::vector<float> h_w(std::size_t{R} * K), h_x(std::size_t{R} * H), h_s(std::size_t{R} * H);
                for (auto& v : h_w) v = std::ldexp(std::uniform_real_distribution<float>(0.05F, 1.0F)(rng), -1);
                for (auto& v : h_x) v = nd(rng) * 0.5F;
                for (auto& v : h_s) v = nd(rng) * 0.05F;
                ids.upload(h_ids.data(), h_ids.size());
                rw.upload(h_w.data(), h_w.size());
                xf.upload(h_x.data(), h_x.size());
                shared.upload(h_s.data(), h_s.size());
                dc::QuantizeA8View q;
                q.x = xf.get();
                q.rows = R;
                q.width = H;
                q.out = xa.view();
                q.out.rows = R;
                dc::enqueue_quantize_a8(q, stream.get());
                const dc::A8Rows x8{xa.codes.get(), xa.deltas.get(), xa.sums.get(), R, H, H};
                // reference: every expert resident
                dc::MoeLayerView v;
                v.shape = shape;
                v.weights = {gate.get(), up.get(), down.get()};
                v.rows = R;
                v.split = geo.split;
                v.x = x8;
                v.ids = ids.get();
                v.route_weights = rw.get();
                v.shared = shared.get();
                v.raw = raw_ref.get();
                v.y = y_ref.get();
                v.workspace = ws.get();
                v.capacity = cap;
                STRIX_CUDA_CHECK(cudaEventRecord(r0, stream.get()));
                dc::enqueue_moe_layer(v, stream.get());
                STRIX_CUDA_CHECK(cudaEventRecord(r1, stream.get()));
                stream.sync();
                float ref_ms = 0.0F;
                STRIX_CUDA_CHECK(cudaEventElapsedTime(&ref_ms, r0, r1));
                // hybrid
                STRIX_CUDA_CHECK(cudaMemsetAsync(raw_h.get(), 0xff, raw_h.size() * 4U, stream.get()));
                STRIX_CUDA_CHECK(cudaMemsetAsync(y_h.get(), 0xff, y_h.size() * 4U, stream.get()));
                stream.sync();
                STRIX_CUDA_CHECK(cudaEventRecord(router_ready, stream.get()));
                dc::PartsHost::Window win;
                win.rows = R;
                win.x = x8;
                win.ids = ids.get();
                win.weights = rw.get();
                win.shared = shared.get();
                win.raw = raw_h.get();
                win.y = y_h.get();
                win.router_ready = router_ready;
                std::vector<char> local_before(std::size_t{R} * K);
                for (std::uint32_t s2 = 0; s2 < R * K; ++s2) local_before[s2] = rlp->host_slot_of[h_ids[s2]] >= 0 ? 1 : 0;
                if (hotres) hotres->before_layer(a.layer, stream.get());
                const auto t = host.run_layer(*rlp, win, stream.get(), true);
                stream.sync();
                if (hotres) {
                    hotres->begin_token();
                    for (std::uint32_t r = 0; r < R; ++r) hotres->after_layer(a.layer, host.last_host_ids() + std::size_t{r} * K, K, host.last_done_event());
                    dyn_hits += hotres->token_stats().dynamic_hits;
                    dyn_copies += hotres->token_stats().copies;
                }
                // bitwise checks
                const auto yr = y_ref.download(std::size_t{R} * H), yh = y_h.download(std::size_t{R} * H);
                bool y_ok = std::memcmp(yr.data(), yh.data(), yr.size() * 4U) == 0;
                const auto rr = raw_ref.download(std::size_t{R} * K * H), rh = raw_h.download(std::size_t{R} * K * H);
                bool raw_ok = true;
                for (std::uint32_t s = 0; s < R * K; ++s)
                    if (local_before[s] && std::memcmp(rr.data() + std::size_t{s} * H, rh.data() + std::size_t{s} * H, H * 4U) != 0) raw_ok = false;
                if (a.null_compute) { y_ok = true; raw_ok = true; }  // wire probe: the answers are zeros
                ++checked;
                if (!y_ok) ++y_bad;
                if (!raw_ok) ++raw_bad;
                if (!y_ok || !raw_ok)
                    std::printf("MISMATCH R %u window %u: y %s raw(local) %s (remote slots %u, fnv y ref %016llx hybrid %016llx)\n", R, w,
                                y_ok ? "ok" : "DIFF", raw_ok ? "ok" : "DIFF", t.remote_slots,
                                static_cast<unsigned long long>(dc::fnv1a(yr.data(), yr.size())),
                                static_cast<unsigned long long>(dc::fnv1a(yh.data(), yh.size())));
                if (w < a.warmup) continue;
                ref_us.push_back(ref_ms * 1000.0);
                enq.push_back(t.enqueue_us); pack.push_back(t.pack_us); if (t.remote_slots) pinned = t.pinned_answers; d2h.push_back(t.d2h_us); wait_us.push_back(t.host_wait_us); prep.push_back(t.prep_us); rtt.push_back(t.rtt_us);
                server.push_back(t.server_us); plan.push_back(t.plan_us); h2d.push_back(t.h2d_us); comb.push_back(t.combine_us);
                local.push_back(t.local_us); total.push_back(t.total_us); overlap.push_back(t.local_overlap_us);
                remote_slots.push_back(t.remote_slots); local_slots.push_back(t.local_slots); items.push_back(t.items);
                qbytes.push_back(static_cast<double>(t.request_bytes)); abytes.push_back(static_cast<double>(t.answer_bytes)); servers.push_back(t.servers);
                if (t.local_done_before_answers) ++local_first;
            }
            const bool ok = y_bad == 0 && raw_bad == 0;
            all_ok = all_ok && ok;
            std::printf("{\"gate\":\"ds41-parts-host\",\"layer\":%u,\"rows\":%u,\"windows\":%u,\"checked\":%u,\"hot_fraction\":%.3f,\"hot_pick\":\"%s\","
                        "\"resident\":%u,\"endpoints\":%zu,\"export_flag\":%s,\"in_place\":%s,\"bitwise_y\":%s,\"bitwise_raw_local\":%s,\"y_mismatch\":%u,\"raw_mismatch\":%u,"
                        "\"slots\":{\"local\":%.2f,\"remote\":%.2f,\"items\":%.2f,\"servers\":%.2f},\"bytes\":{\"request\":%.0f,\"answer\":%.0f},"
                        "\"pinned_answers\":%s,\"median_us\":{\"enqueue\":%.1f,\"pack\":%.1f,\"d2h\":%.1f,\"host_wait\":%.1f,\"prep\":%.1f,\"rtt\":%.1f,\"server\":%.1f,\"plan\":%.1f,\"h2d\":%.1f,\"combine\":%.1f,"
                        "\"local\":%.1f,\"total\":%.1f,\"local_overlap\":%.1f,\"all_resident_layer\":%.1f},\"local_done_before_answers\":%u,\"dyn_slots\":%u,\"dyn_hits\":%llu,\"dyn_copies\":%llu,\"placement\":\"%s\"}\n",
                        a.layer, R, a.windows, checked, a.hot_fraction, a.hot_pick.c_str(), hot, po.endpoints.size(), po.export_flag ? "true" : "false", po.answers_in_place ? "true" : "false", y_bad == 0 ? "true" : "false",
                        raw_bad == 0 ? "true" : "false", y_bad, raw_bad, mean(local_slots), mean(remote_slots), mean(items), mean(servers), mean(qbytes),
                        mean(abytes), pinned ? "true" : "false", median(enq), median(pack), median(d2h), median(wait_us), median(prep), median(rtt), median(server), median(plan), median(h2d), median(comb),
                        median(local), median(total), median(overlap), median(ref_us), local_first, hotres ? hotres->dynamic_slots(a.layer) : 0U,
                        static_cast<unsigned long long>(dyn_hits), static_cast<unsigned long long>(dyn_copies), a.placement.c_str());
            std::fflush(stdout);
        }
        std::printf(all_ok ? "PASS parts-host bitwise\n" : "FAIL parts-host bitwise\n");
        return all_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
}
