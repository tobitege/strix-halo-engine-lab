// strix-cuda-ds41-hot-decode: the deepseek41 R1 decode on the CUDA parts-host with dense + hot experts: every layer's
// dense part on this device (CudaPrefill::step), the routed experts of a hot-expert placement resident here (static
// slots + optional dynamic LRU slots refilled from pinned host memory after misses), the others computed by expert
// servers (strix/ds41-expert-rpc-v1) through PartsHost::run_layer at each MoE site (CudaStepIo::moe_site).
//
//   strix-cuda-ds41-hot-decode MODEL.gguf PLACEMENT.json --import-state STATE [--ref PAIR_LOG] [--steps N]
//       --connect HOST:PORT[,HOST:PORT] --token-file F [--busy-poll] [--dyn-gib G --source-gib S --budget N
//       --routes ROUTES [--seqmod M:K]] [--chats I,J,.. --chat-prompt P --chat-steps N --seqs-file SEQS]
//       [--timing] [--json OUT] [--timeline OUT] [--no-load]
//
// Greedy steps from the imported state; with --ref, every step's fed token, argmax and logits FNV are compared with
// the pair's greedy log ("pos P fed T argmax A ... logits_fnv F"). --chats: for each W1 sequence I (a line "NAME\tids"
// of SEQS), re-import the state, feed its first P tokens in windows of up to 8 rows (teacher forcing), then N greedy
// steps (timing; no reference). One JSON line per step: tokens, FNV, match, wall / device ms, remote layers, slots,
// answer items, RTT (sum, max), server time, dynamic hits / copies; --timeline: one line per (step, layer).
#include "strix/device/cuda_enumerator.hpp"
#include "strix/cuda_ds41_hot_residency.hpp"
#include "strix/cuda_ds41_hot_site.hpp"
#include "strix/cuda_ds41_parts_host_driver.hpp"
#include "strix/cuda_ds41_prefill.hpp"
#include "strix/deepseek41_attention_state.hpp"
#include "strix/ds41_hot_placement.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace strix::ds41::cuda;
namespace x = strix::ds41::xrpc;
namespace sd = strix::ds41;

namespace {

[[noreturn]] void usage(const char* why) {
    std::fprintf(stderr, "strix-cuda-ds41-hot-decode: %s\n", why);
    std::exit(2);
}

std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> out;
    std::istringstream is(s);
    std::string v;
    while (std::getline(is, v, c)) if (!v.empty()) out.push_back(v);
    return out;
}

struct RefStep {
    std::uint32_t pos = 0, fed = 0, argmax = 0;
    std::uint64_t fnv = 0;
};
std::map<std::uint32_t, RefStep> read_ref(const std::string& path) {
    std::ifstream in(path);
    if (!in) usage("cannot read the reference log");
    std::map<std::uint32_t, RefStep> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("pos ", 0) != 0) continue;
        std::istringstream s(line);
        std::string w;
        RefStep r;
        while (s >> w) {
            if (w == "pos") s >> r.pos;
            else if (w == "fed") s >> r.fed;
            else if (w == "argmax") s >> r.argmax;
            else if (w == "logits_fnv") s >> std::hex >> r.fnv >> std::dec;
        }
        out.emplace(r.pos, r);
    }
    return out;
}

std::string hex(std::uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
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
    // lazy module loading + a kernel spinning on a host flag can deadlock the first launch of another kernel
#ifdef _WIN32
    _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
#endif
    if (argc < 3) usage("MODEL.gguf PLACEMENT.json [options]");
    const std::string model = argv[1], placement_path = argv[2];
    std::string budgets_arg, state_file, ref_file, connect, token_file, routes, seqmod = "5:0", chats_arg, seqs_file, json_out, timeline_out;
    std::uint32_t steps = 64, budget = 16, chat_prompt = 96, chat_steps = 128;
    double dyn_gib = 0.0, source_gib = 40.0;
    bool busy_poll = false, timing = false, no_load = false, graph_site = true, keep_reader = false;
    double heartbeat_us = 200.0;
    CudaPrefillOptions o;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage(("missing value of " + a).c_str());
            return argv[++i];
        };
        if (a == "--import-state") state_file = next();
        else if (a == "--ref") ref_file = next();
        else if (a == "--steps") steps = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--connect") connect = next();
        else if (a == "--token-file") token_file = next();
        else if (a == "--busy-poll") busy_poll = true;
        else if (a == "--dyn-gib") dyn_gib = std::stod(next());
        else if (a == "--source-gib") source_gib = std::stod(next());
        else if (a == "--budgets") budgets_arg = next();
        else if (a == "--budget") budget = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--routes") routes = next();
        else if (a == "--seqmod") seqmod = next();
        else if (a == "--chats") chats_arg = next();
        else if (a == "--chat-prompt") chat_prompt = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--chat-steps") chat_steps = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--seqs-file") seqs_file = next();
        else if (a == "--timing") timing = true;
        else if (a == "--json") json_out = next();
        else if (a == "--timeline") timeline_out = next();
        else if (a == "--no-load") no_load = true;
        else if (a == "--eager-site") graph_site = false;
        else if (a == "--heartbeat-us") heartbeat_us = std::stod(next());  // PartsHost::run_layer at every site, eager steps (the first version)
        else if (a == "--engram") o.engram = next();
        else if (a == "--no-engram-prefetch") o.engram_prefetch = false;
        else if (a == "--keep-reader") keep_reader = true;  // the expert loader's cached handle stays open (as before)
        else usage(("unknown option " + a).c_str());
    }
    if (state_file.empty()) usage("--import-state is required");
    std::ofstream js, tl;
    if (!json_out.empty()) js.open(json_out, std::ios::app);
    if (!timeline_out.empty()) tl.open(timeline_out, std::ios::app);
    const auto emit = [&](const std::string& line) {
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
        if (js) js << line << "\n";
    };
    int failures = 0;
    try {
        const auto t_load = std::chrono::steady_clock::now();
        const auto imported = strix::deepseek41::load_attention_state(state_file);
        std::uint32_t extra = steps + 16U;
        if (!chats_arg.empty()) extra = std::max(extra, chat_prompt + chat_steps + 16U);
        o.max_positions = imported.position + extra;
        o.tiers.assign(40, ExpertTier::file);
        o.slots = 0;           // decode-only object: no expert bytes of its own (the MoE sites are the parts-host's)
        o.decode_rows = 8;
        o.decode_graphs = graph_site;
        o.timing = timing;
        o.timing_ops = false;
        CudaPrefill p(model, o);

        // residency (static placement + dynamic slots) and the expert servers
        GgufExpertReader reader(model);
        auto shape = reader.shape();
        if (shape.top_k == 0) shape.top_k = 6;
        const std::uint32_t E = shape.experts, K = shape.top_k, H = shape.hidden;
        sd::HotPlacementExpect ex;
        ex.geometry.experts = E;
        ex.geometry.top_k = K;
        ex.geometry.canonical_split = E / 2U;
        ex.geometry.expert_bytes = shape.expert_bytes();
        const auto placement = sd::load_hot_placement(placement_path, ex);
        const std::uint32_t L = placement.geometry.layers;
        HotResidencyOptions ho;
        ho.shape = shape;
        for (std::uint32_t l = 0; l < L; ++l) ho.layers.push_back(l);
        ho.copies_per_token = budget;
        ho.load_weights = !no_load;
        const std::uint64_t eb = shape.expert_bytes();
        if (dyn_gib > 0.0) {
            const auto n_dyn = static_cast<std::uint64_t>(dyn_gib * 1073741824.0 / static_cast<double>(eb));
            std::vector<std::uint64_t> cnt(L);
            std::uint64_t sum = 0;
            for (std::uint32_t l = 0; l < L; ++l) { cnt[l] = placement.layers[l].experts[0].size() + 1U; sum += cnt[l]; }
            ho.dynamic_slots.assign(L, 0U);
            for (std::uint32_t l = 0; l < L; ++l) ho.dynamic_slots[l] = static_cast<std::uint32_t>(std::min<std::uint64_t>(n_dyn * cnt[l] / sum, E - cnt[l] + 1U));
            // source: the warmest non-static experts of the recorded routes outside the test fold
            if (routes.empty()) usage("--dyn-gib needs --routes (the source's frequencies)");
            std::vector<double> freq(std::size_t{L} * E, 0.0);
            const auto mk = split(seqmod, ':');
            const auto M = std::stoul(mk.at(0)), Kf = std::stoul(mk.at(1));
            std::ifstream rf(routes);
            std::string line;
            long seq = -1;
            while (std::getline(rf, line)) {
                if (line.rfind("# seq", 0) == 0) { ++seq; continue; }
                if (line.empty() || line[0] == '#' || (seq >= 0 && static_cast<unsigned long>(seq) % M == Kf)) continue;
                std::istringstream ls(line);
                std::string item;
                while (ls >> item) {
                    const auto colon = item.find(':');
                    if (colon == std::string::npos) continue;
                    const auto l = static_cast<std::uint32_t>(std::stoul(item.substr(0, colon)));
                    if (l >= L) continue;
                    for (const auto& v : split(item.substr(colon + 1), ',')) {
                        const auto e = static_cast<std::uint32_t>(std::stoul(v));
                        if (e < E) freq[std::size_t{l} * E + e] += 1.0;
                    }
                }
            }
            std::vector<std::uint32_t> idx;
            for (std::uint32_t l = 0; l < L; ++l)
                for (std::uint32_t e = 0; e < E; ++e)
                    if (placement.device_of(l, e) != sd::HotDevice::rtx_pro && ho.dynamic_slots[l] > 0) idx.push_back(l * E + e);
            std::stable_sort(idx.begin(), idx.end(), [&](std::uint32_t a2, std::uint32_t b2) { return freq[a2] > freq[b2]; });
            idx.resize(std::min<std::size_t>(idx.size(), static_cast<std::size_t>(source_gib * 1073741824.0 / static_cast<double>(eb))));
            std::sort(idx.begin(), idx.end());
            ho.source.assign(L, {});
            for (const auto v : idx) ho.source[v / E].push_back(v % E);
        }
        HotResidency res(placement, reader, ho);
        if (!keep_reader) reader.close();  // loaded: the source is in pinned memory, the file is not read again
        std::uint32_t statics = 0, dyns = 0;
        for (std::uint32_t l = 0; l < L; ++l) { statics += res.static_slots(l); dyns += res.dynamic_slots(l); }

        PartsHostOptions po;
        po.shape = shape;
        po.split = E / 2U;
        po.capacity = 8;
        po.endpoints = split(connect, ',');
        po.busy_poll = busy_poll;
        po.connect_wait = std::chrono::milliseconds(900000);
        po.hello.geometry.layers = L;
        po.hello.geometry.experts = E;
        po.hello.geometry.top_k = K;
        po.hello.geometry.hidden = H;
        po.hello.geometry.inter = shape.inter;
        po.hello.geometry.split = E / 2U;
        if (!token_file.empty()) po.hello.token = x::read_token_file(token_file);
        const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();
        {
            std::ostringstream os;
            os << "{\"event\":\"loaded\",\"describe\":" << p.describe() << ",\"static_experts\":" << statics << ",\"dynamic_slots\":" << dyns
               << ",\"residency_gb\":" << static_cast<double>(res.device_bytes()) / 1e9 << ",\"source_gb\":" << static_cast<double>(res.source_bytes()) / 1e9
               << ",\"budget\":" << budget << ",\"load_s\":" << load_s << "}";
            emit(os.str());
        }
        std::unique_ptr<PartsHost> host_ptr;
        std::unique_ptr<HotSite> site_ptr;
        if (graph_site) site_ptr = std::make_unique<HotSite>(res, po, L);
        else host_ptr = std::make_unique<PartsHost>(po);
        const auto& acks = graph_site ? site_ptr->acks() : host_ptr->acks();
        if (graph_site) {
            site_ptr->set_heartbeat_us(heartbeat_us);
            site_ptr->start(0, L);
        }
        for (std::size_t i = 0; i < acks.size(); ++i) {
            const auto& ack = acks[i];
            emit("{\"event\":\"server\",\"endpoint\":\"" + po.endpoints[i] + "\",\"engine\":\"" + ack.engine + "\",\"experts\":[" +
                 std::to_string(ack.residency.expert_first) + "," + std::to_string(ack.residency.expert_end) + "],\"numeric\":\"" + ack.numeric_identity + "\"}");
        }

        // the MoE site hook
        cudaEvent_t router_ready = nullptr;
        STRIX_CUDA_CHECK(cudaEventCreate(&router_ready));
        std::vector<PartsHostTiming> lt(L);
        std::vector<char> seen(L, 0);
        CudaStepIo io;
        io.moe_site_graphs = graph_site;
        io.moe_site = [&](const CudaMoeSite& site) {
            if (graph_site) {
                site_ptr->enqueue(site.layer, site.x, site.ids, site.weights, site.shared, site.raw, site.y, site.shared_expert, site.stream,
                                  site.layer == 0U);
                return;
            }
            auto& host = *host_ptr;
            res.before_layer(site.layer, site.stream);
            STRIX_CUDA_CHECK(cudaEventRecord(router_ready, site.stream));
            PartsHost::Window w;
            w.rows = site.rows;
            w.x = site.x;
            w.ids = site.ids;
            w.weights = site.weights;
            w.shared = site.shared;
            w.raw = site.raw;
            w.y = site.y;
            w.router_ready = router_ready;
            lt[site.layer] = host.run_layer(res.layer(site.layer), w, site.stream, false, site.shared_expert);
            seen[site.layer] = 1;
            for (std::uint32_t r = 0; r < site.rows; ++r) res.after_layer(site.layer, host.last_host_ids() + std::size_t{r} * K, K, host.last_done_event());
        };
        const std::map<std::uint32_t, RefStep> ref = ref_file.empty() ? std::map<std::uint32_t, RefStep>{} : read_ref(ref_file);

        struct Agg {
            std::vector<double> ms, dev, rl, slots, items, rtt, rtt_layer, srv, copies, hits, eng, eng_wait, pf, wall;
        };
        // wall time between the starts of consecutive counted steps of a phase (includes this app's own work)
        std::string wall_phase;
        auto wall_t = std::chrono::steady_clock::now();
        std::map<std::string, Agg> agg;
        std::uint32_t compared = 0, matched = 0;
        const auto do_step = [&](const std::vector<std::uint32_t>& fed, const std::string& phase, std::uint32_t k, bool count) {
            std::fill(seen.begin(), seen.end(), 0);
            if (!graph_site) res.begin_token();
            const std::uint32_t pos = p.position();
            const auto t_entry = std::chrono::steady_clock::now();
            const double wall_ms = wall_phase == phase ? std::chrono::duration<double, std::milli>(t_entry - wall_t).count() : 0.0;
            wall_phase = count ? phase : std::string();
            wall_t = t_entry;
            const auto s = p.step(fed, io);
            HotRefillStats st;
            if (graph_site) {
                const auto t = site_ptr->last_token();
                for (std::uint32_t l = 0; l < L && l < t.size(); ++l) {
                    auto& x = lt[l];
                    x = PartsHostTiming{};
                    x.rows = 1;
                    x.local_slots = t[l].local;
                    x.remote_slots = t[l].remote;
                    x.items = t[l].items;
                    x.servers = t[l].servers;
                    x.answer_bytes = t[l].answer_bytes;
                    x.host_wait_us = t[l].export_seen_us;
                    x.prep_us = t[l].prep_us;
                    x.rtt_us = t[l].rtt_us;
                    x.server_us = t[l].server_us;
                    x.plan_us = t[l].plan_us;
                    seen[l] = 1;
                    st.copies += static_cast<std::uint32_t>(t[l].copies);
                    st.dynamic_hits += static_cast<std::uint32_t>(t[l].hits);
                }
            } else {
                st = res.token_stats();
            }
            std::uint32_t rl = 0, slots = 0, items = 0;
            double rtt_sum = 0, rtt_max = 0, srv_sum = 0;
            for (std::uint32_t l = 0; l < L; ++l) {
                if (!seen[l] || !lt[l].remote_slots) continue;
                ++rl;
                slots += lt[l].remote_slots;
                items += lt[l].items;
                rtt_sum += lt[l].rtt_us;
                rtt_max = std::max(rtt_max, lt[l].rtt_us);
                srv_sum += lt[l].server_us;
            }
            std::ostringstream os;
            os << "{\"event\":\"" << phase << "\",\"k\":" << k << ",\"pos\":" << pos << ",\"rows\":" << fed.size() << ",\"fed\":" << fed.back()
               << ",\"argmax\":" << (s.tokens.empty() ? 0U : s.tokens.back()) << ",\"logits_fnv\":\"" << (s.logits_fnv.empty() ? std::string() : hex(s.logits_fnv.back()))
               << "\",\"ms\":" << s.seconds * 1e3 << ",\"enqueue_ms\":" << s.enqueue_seconds * 1e3 << ",\"device_ms\":" << s.device_ms
               << ",\"engram_ms\":" << s.engram_seconds * 1e3 << ",\"engram_wait_ms\":" << s.engram_wait_seconds * 1e3 << ",\"engram_issue_ms\":" << s.engram_issue_seconds * 1e3
               << ",\"engram_pf\":" << (s.engram_prefetched ? 1 : 0) << ",\"wall_ms\":" << wall_ms << ",\"remote_layers\":" << rl << ",\"remote_slots\":" << slots << ",\"items\":" << items
               << ",\"rtt_sum_us\":" << rtt_sum << ",\"rtt_max_us\":" << rtt_max << ",\"server_sum_us\":" << srv_sum << ",\"dyn_hits\":" << st.dynamic_hits
               << ",\"copies\":" << st.copies << ",\"not_in_source\":" << st.not_in_source << ",\"over_budget\":" << st.over_budget;
            if (fed.size() == 1U && !ref.empty() && !s.tokens.empty() && phase.rfind("step", 0) == 0) {
                const auto it = ref.find(pos);
                if (it != ref.end()) {
                    const bool ok = it->second.fed == fed.back() && it->second.argmax == s.tokens.back() && (s.logits_fnv.empty() || it->second.fnv == s.logits_fnv.back());
                    ++compared;
                    matched += ok ? 1U : 0U;
                    if (!ok) ++failures;
                    os << ",\"ref_argmax\":" << it->second.argmax << ",\"ref_fnv\":\"" << hex(it->second.fnv) << "\",\"match\":" << (ok ? "true" : "false");
                }
            }
            os << "}";
            emit(os.str());
            if (tl)
                for (std::uint32_t l = 0; l < L; ++l) {
                    if (!seen[l]) continue;
                    const auto& t = lt[l];
                    tl << "{\"phase\":\"" << phase << "\",\"k\":" << k << ",\"pos\":" << pos << ",\"layer\":" << l << ",\"rows\":" << t.rows;
                    if (l < s.layers.size()) tl << ",\"layer_ms\":" << s.layers[l].total_ms << ",\"attn_ms\":" << s.layers[l].attention_ms << ",\"moe_ms\":" << s.layers[l].moe_ms;
                    tl << ",\"local\":" << t.local_slots << ",\"remote\":" << t.remote_slots << ",\"items\":" << t.items << ",\"servers\":" << t.servers
                       << ",\"enqueue_us\":" << t.enqueue_us << ",\"host_wait_us\":" << t.host_wait_us << ",\"prep_us\":" << t.prep_us << ",\"rtt_us\":" << t.rtt_us
                       << ",\"server_us\":" << t.server_us << ",\"pack_us\":" << t.pack_us << ",\"plan_us\":" << t.plan_us
                       << ",\"answer_bytes\":" << t.answer_bytes << "}\n";
                }
            if (count && fed.size() == 1U) {
                auto& g = agg[phase];
                g.ms.push_back(s.seconds * 1e3);
                g.dev.push_back(s.device_ms);
                g.rl.push_back(rl);
                g.slots.push_back(slots);
                g.items.push_back(items);
                g.rtt.push_back(rtt_sum);
                g.srv.push_back(srv_sum);
                g.copies.push_back(st.copies);
                g.hits.push_back(st.dynamic_hits);
                g.eng.push_back(s.engram_seconds * 1e3);
                g.eng_wait.push_back(s.engram_wait_seconds * 1e3);
                g.pf.push_back(s.engram_prefetched ? 1.0 : 0.0);
                if (wall_ms > 0.0) g.wall.push_back(wall_ms);
                for (std::uint32_t l = 0; l < L; ++l)
                    if (seen[l] && lt[l].remote_slots) g.rtt_layer.push_back(lt[l].rtt_us);
            }
            return s.tokens.empty() ? fed.back() : s.tokens.back();
        };

        std::vector<std::uint32_t> budgets{budget};
        if (!budgets_arg.empty()) { budgets.clear(); for (const auto& v : split(budgets_arg, ',')) budgets.push_back(static_cast<std::uint32_t>(std::stoul(v))); }
        for (const std::uint32_t bud : budgets) {
        res.set_copies_per_token(bud);
        const std::string sfx = budgets.size() > 1U ? "-b" + std::to_string(bud) : std::string();
        // ---- greedy from the imported state (the exactness run)
        p.import_state(imported);
        res.reset_dynamic();
        std::uint32_t next = imported.next_token;
        for (std::uint32_t k = 0; k < steps; ++k) next = do_step({next}, "step" + sfx, k, k >= 4U);

        // ---- chats: the imported context, a W1 sequence's first tokens forced in windows of 8, then greedy
        if (!chats_arg.empty()) {
            if (seqs_file.empty()) usage("--chats needs --seqs-file");
            std::vector<std::pair<std::string, std::vector<std::uint32_t>>> seqs;
            std::ifstream sf(seqs_file);
            std::string line;
            while (std::getline(sf, line)) {
                const auto tab = line.find('\t');
                if (tab == std::string::npos) continue;
                std::vector<std::uint32_t> ids;
                for (const auto& v : split(line.substr(tab + 1), ',')) ids.push_back(static_cast<std::uint32_t>(std::stoul(v)));
                seqs.emplace_back(line.substr(0, tab), std::move(ids));
            }
            for (const auto& c : split(chats_arg, ',')) {
                const auto& sq = seqs.at(std::stoul(c));
                p.import_state(imported);
                res.reset_dynamic();
                const auto parts = split(sq.first, ':');
                const std::string phase = "chat" + c + "-" + (parts.size() > 1 ? parts[1] : std::string("x")) + sfx;
                const std::uint32_t P = std::min<std::uint32_t>(chat_prompt, static_cast<std::uint32_t>(sq.second.size()));
                std::uint32_t at = 0, kk = 0;
                while (at < P) {
                    const std::uint32_t r = graph_site ? 1U : std::min(8U, P - at);  // the graph site is R1
                    std::vector<std::uint32_t> fed(sq.second.begin() + at, sq.second.begin() + at + r);
                    next = do_step(fed, phase + "-prompt", kk++, false);
                    at += r;
                }
                for (std::uint32_t k = 0; k < chat_steps; ++k) next = do_step({next}, phase, k, k >= 4U);
            }
        }
        }  // budgets
        if (graph_site) {
            emit("{\"event\":\"site\",\"device_wait_timeouts\":" + std::to_string(site_ptr->errors()) + "}");
            site_ptr->stop();
        }
        for (const auto& [phase, g] : agg) {
            std::ostringstream os;
            os << "{\"event\":\"summary\",\"phase\":\"" << phase << "\",\"steps\":" << g.ms.size() << ",\"ms_mean\":" << mean(g.ms) << ",\"ms_p50\":" << pct(g.ms, 0.5)
               << ",\"ms_p90\":" << pct(g.ms, 0.9) << ",\"device_ms_mean\":" << mean(g.dev) << ",\"tok_s\":" << (mean(g.ms) > 0 ? 1e3 / mean(g.ms) : 0.0)
               << ",\"remote_layers\":" << mean(g.rl) << ",\"remote_slots\":" << mean(g.slots) << ",\"items\":" << mean(g.items) << ",\"rtt_sum_us\":" << mean(g.rtt)
               << ",\"rtt_layer_p50\":" << pct(g.rtt_layer, 0.5) << ",\"rtt_layer_p90\":" << pct(g.rtt_layer, 0.9) << ",\"server_sum_us\":" << mean(g.srv)
               << ",\"copies\":" << mean(g.copies) << ",\"dyn_hits\":" << mean(g.hits) << ",\"engram_ms\":" << mean(g.eng)
               << ",\"engram_wait_ms\":" << mean(g.eng_wait) << ",\"engram_wait_p90\":" << pct(g.eng_wait, 0.9) << ",\"prefetched\":" << mean(g.pf)
               << ",\"wall_ms_mean\":" << mean(g.wall) << ",\"wall_tok_s\":" << (mean(g.wall) > 0 ? 1e3 / mean(g.wall) : 0.0);
            if (phase.rfind("step", 0) == 0) os << ",\"compared_all\":" << compared << ",\"matched_all\":" << matched;
            os << "}";
            emit(os.str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-cuda-ds41-hot-decode: %s\n", e.what());
        return 1;
    }
    return failures ? 1 : 0;
}
