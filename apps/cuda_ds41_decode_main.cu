// strix-cuda-ds41-decode: incremental decode steps of the deepseek41 CUDA backend (CudaPrefill::step), checked and timed.
//
//   strix-cuda-ds41-decode MODEL.gguf (--prompt-ids FILE [--tokens T] | --import-state FILE) [--positions N]
//       [--layers END] [--steps N] [--rows R] [--ref PAIR_LOG] [--check-run] [--hashes] [--split K]
//       [--tiers DHF.. | --device-gib G] [--slots S] [--eager | --graphs-ab] [--graph-warmup N] [--bucket B]
//       [--timing | --timing-layers] [--stops K1,K2,..] [--stop-steps N] [--no-spin] [--no-lean-state] [--engram file|zero] [--json OUT]
//       [--expert-tp] [--overlay OVERLAY.gguf] [--weight-dirs LIST] [--drive-readers N] [--weight-verify cached|always]
// --fuse-hc / --fuse-shared: opt-in exact decode launch fusions (default off).
// --expert-tp + --overlay c-u6.gguf: the pair's P15 target identity (ds41-expert-tp2-i1152-v1 on the c-u6 dense weights).
//
// The sequence starts from a prefill of the prompt (run) or an imported decode-format state, then decodes greedily:
// every step feeds the previous step's token (R rows: the next R tokens of the reference continuation when --ref is
// given, else the greedy token repeated; R > 1 is a timing shape). One JSON line per step: position, fed tokens,
// argmax, logits FNV, wall and device milliseconds, and the comparison with the reference log (lines "pos P fed T
// argmax A top X logits_fnv F" of the pair's greedy run).
//
// --check-run: run(prompt[0, T)) + step(next) against run(prompt[0, T) + next): token, logits FNV and the last row's
//   stream FNV after every layer (prints both; exit 1 on a difference).
// --split K: the same greedy run through two objects, FRONT [0, K) (stream out) and TAIL [K, END) (stream in, the
//   boundary sections copied FRONT -> TAIL every step): the pipeline API's exactness check.
// --dump-boundary FILE (FRONT [0, K) via --layers K, tokens from --ref) / --tail-from FILE (TAIL via --first K): the same
//   pipeline across two processes (every step's boundary in a file).
// --commit-test R1,R2,..: the partial commit gate (CudaPrefill::commit_window) from --import-state: --commit-skip N R1
//   steps, then for every R of the list and every accepted depth a = 0..R-1 one window of R rows (the sequence's next
//   token, then R - 1 other tokens) committed with kept = a + 1, followed by --commit-cont R1 steps; then the same
//   object re-imports the state and replays every committed token as R1 steps. Compared per committed row: logits FNV
//   (with the head) or the FNV of the stream + mix rows (a FRONT, --layers K); after every window's commit: the
//   exported state's per-layer hashes (rings, KV / index-key entries, ratio-2 carries, top-k) and the position.
// --from-zero (with --prompt-ids, the head): no prefill; reset_sequence(), the prompt fed as decode windows of --rows
//   rows from position 0 (every row compared with --ref: fed, argmax, logits FNV), then --steps greedy R1 steps.
// --stops K1,K2,..: after the greedy steps, --stop-steps steps (default 16) that stop after layer K (FRONT prefixes:
//   the stream comes out, no head); the sequence's later layers fall behind (timing only).
#include "strix/device/cuda_enumerator.hpp"
#include "strix/cuda_ds41_prefill.hpp"
#include "strix/deepseek41_attention_state.hpp"
#include "strix/ds41_file_reader.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace strix::ds41::cuda;

namespace {

std::uint64_t fnv(const float* v, std::size_t n) {
    std::uint64_t h = 14695981039346656037ULL;
    const auto* p = reinterpret_cast<const std::uint8_t*>(v);
    for (std::size_t i = 0; i < n * 4U; ++i) h = (h ^ p[i]) * 1099511628211ULL;
    return h;
}

[[noreturn]] void usage(const char* why) {
    std::fprintf(stderr, "strix-cuda-ds41-decode: %s\n", why);
    std::fprintf(stderr, "usage: strix-cuda-ds41-decode MODEL.gguf (--prompt-ids FILE | --import-state FILE) [options]\n"
                         "  --weight-dirs LIST --drive-readers N (1..32) --weight-verify cached|always\n");
    std::exit(2);
}

std::vector<std::uint32_t> read_ids(const std::string& path) {
    std::ifstream in(path);
    if (!in) usage("cannot read the prompt ids");
    std::vector<std::uint32_t> ids;
    std::string tok;
    while (in >> tok) {
        for (auto& ch : tok)
            if (ch == ',' || ch == '[' || ch == ']') ch = ' ';
        std::istringstream s(tok);
        std::uint64_t v = 0;
        while (s >> v) ids.push_back(static_cast<std::uint32_t>(v));
    }
    if (ids.empty()) usage("empty prompt ids");
    return ids;
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
        out.emplace(r.pos, r);  // first occurrence (the logs repeat rank 0's lines)
    }
    return out;
}

std::string hex(std::uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v.size() % 2U ? v[v.size() / 2U] : 0.5 * (v[v.size() / 2U - 1U] + v[v.size() / 2U]);
}

std::string step_json(const CudaStepResult& s, const std::vector<std::uint32_t>& fed) {
    std::ostringstream o;
    o << "\"pos\":" << s.position << ",\"rows\":" << s.rows << ",\"fed\":[";
    for (std::size_t i = 0; i < fed.size(); ++i) o << (i ? "," : "") << fed[i];
    o << "],\"argmax\":[";
    for (std::size_t i = 0; i < s.tokens.size(); ++i) o << (i ? "," : "") << s.tokens[i];
    o << "],\"logits_fnv\":[";
    for (std::size_t i = 0; i < s.logits_fnv.size(); ++i) o << (i ? "," : "") << "\"" << hex(s.logits_fnv[i]) << "\"";
    o << "],\"ms\":" << s.seconds * 1e3 << ",\"enqueue_ms\":" << s.enqueue_seconds * 1e3 << ",\"device_ms\":" << s.device_ms
      << ",\"engram_ms\":" << s.engram_seconds * 1e3 << ",\"engram_wait_ms\":" << s.engram_wait_seconds * 1e3
      << ",\"engram_pf\":" << (s.engram_prefetched ? 1 : 0) << ",\"expert_io_ms\":" << s.expert_io_seconds * 1e3 << ",\"graphs\":" << (s.graphs ? "true" : "false")
      << ",\"captures\":" << s.graph_captures;
    if (!s.layers.empty()) {
        o << ",\"layers\":[";
        bool f = true;
        for (std::size_t l = 0; l < s.layers.size(); ++l) {
            const auto& t = s.layers[l];
            if (t.rows == 0U) continue;
            o << (f ? "" : ",") << "[" << l << "," << t.total_ms << "," << t.attention_ms << "," << t.moe_ms << "]";
            f = false;
        }
        o << "]";
    }
    if (!s.ops.empty()) {
        o << ",\"ops\":{";
        bool f = true;
        for (const auto& [k, v] : s.ops) {
            o << (f ? "" : ",") << "\"" << k << "\":" << v;
            f = false;
        }
        o << "}";
    }
    if (!s.stream_hashes.empty()) {
        o << ",\"stream_hashes\":[";
        for (std::size_t i = 0; i < s.stream_hashes.size(); ++i) o << (i ? "," : "") << "\"" << hex(s.stream_hashes[i]) << "\"";
        o << "]";
    }
    return o.str();
}


// One step's boundary of a cut after layer K - 1 (strix/ds41-pipeline-v1 content): tokens, stream, FFN mix, and the
// sections a TAIL [K, 40) reads (KV entries the step completed on the sources < K, top-k rows of the index sources < K,
// the candidate rows).
struct Boundary {
    std::uint32_t pos = 0;
    std::vector<std::uint32_t> tokens;
    std::vector<float> stream, mix;
    struct Kv {
        std::uint32_t layer = 0, first = 0, count = 0;
        std::vector<std::uint8_t> kv, idx;
    };
    struct Topk {
        std::uint32_t layer = 0;
        std::vector<std::uint32_t> counts, ids;
    };
    std::vector<Kv> kv;
    std::vector<Topk> topk;
    std::vector<std::uint8_t> cand;
};

void collect_sections(const CudaPrefill& front, Boundary& b, std::uint32_t K) {
    const auto R = static_cast<std::uint32_t>(b.tokens.size());
    for (const std::uint32_t src : {2U, 8U, 14U, 20U}) {  // the KV sources (ratio 2, 2, 2, 1)
        if (src >= K) continue;
        const std::uint32_t ratio = src == 20U ? 1U : 2U;
        Boundary::Kv e;
        e.layer = src;
        e.first = b.pos / ratio;
        e.count = (b.pos + R) / ratio - e.first;
        e.kv.resize(std::size_t{e.count} * 288U);
        e.idx.resize(std::size_t{e.count} * 68U);
        front.copy_kv_entries(src, e.first, e.count, e.kv.data(), e.idx.data());
        b.kv.push_back(std::move(e));
    }
    for (std::uint32_t src = 0; src < K; ++src) {
        Boundary::Topk t;
        t.layer = src;
        t.counts.resize(R);
        t.ids.resize(std::size_t{R} * 512U);
        try {
            front.copy_topk(src, R, t.counts.data(), t.ids.data());
        } catch (const std::invalid_argument&) {  // not an index source
            continue;
        }
        b.topk.push_back(std::move(t));
    }
    if (K > 20U) {
        b.cand.resize(std::size_t{R} * front.cand_stride());
        front.copy_candidates(R, b.cand.data());
    }
}

void apply_sections(CudaPrefill& tail, const Boundary& b) {
    const auto R = static_cast<std::uint32_t>(b.tokens.size());
    for (const auto& e : b.kv) {
        try {
            tail.write_kv_entries(e.layer, e.first, e.count, e.kv.data(), e.idx.data());
        } catch (const std::invalid_argument&) {  // not read by the TAIL's layers
        }
    }
    for (const auto& t : b.topk) {
        try {
            tail.write_topk(t.layer, R, t.counts.data(), t.ids.data());
        } catch (const std::invalid_argument&) {
        }
    }
    if (!b.cand.empty()) {
        try {
            tail.write_candidates(R, b.cand.data());
        } catch (const std::invalid_argument&) {
        }
    }
}

template <class T>
void put_vec(std::FILE* f, const std::vector<T>& v) {
    const std::uint64_t n = v.size();
    std::fwrite(&n, 8, 1, f);
    if (n) std::fwrite(v.data(), sizeof(T), v.size(), f);
}
template <class T>
bool get_vec(std::FILE* f, std::vector<T>& v) {
    std::uint64_t n = 0;
    if (std::fread(&n, 8, 1, f) != 1) return false;
    v.resize(n);
    return n == 0 || std::fread(v.data(), sizeof(T), n, f) == n;
}
void write_boundary(std::FILE* f, const Boundary& b) {
    std::fwrite(&b.pos, 4, 1, f);
    put_vec(f, b.tokens);
    put_vec(f, b.stream);
    put_vec(f, b.mix);
    const std::uint32_t nk = static_cast<std::uint32_t>(b.kv.size()), nt = static_cast<std::uint32_t>(b.topk.size());
    std::fwrite(&nk, 4, 1, f);
    for (const auto& e : b.kv) {
        std::fwrite(&e.layer, 4, 1, f);
        std::fwrite(&e.first, 4, 1, f);
        std::fwrite(&e.count, 4, 1, f);
        put_vec(f, e.kv);
        put_vec(f, e.idx);
    }
    std::fwrite(&nt, 4, 1, f);
    for (const auto& t : b.topk) {
        std::fwrite(&t.layer, 4, 1, f);
        put_vec(f, t.counts);
        put_vec(f, t.ids);
    }
    put_vec(f, b.cand);
    std::fflush(f);
}
bool read_boundary(std::FILE* f, Boundary& b) {
    if (std::fread(&b.pos, 4, 1, f) != 1) return false;
    if (!get_vec(f, b.tokens) || !get_vec(f, b.stream) || !get_vec(f, b.mix)) return false;
    std::uint32_t nk = 0, nt = 0;
    if (std::fread(&nk, 4, 1, f) != 1) return false;
    b.kv.assign(nk, {});
    for (auto& e : b.kv)
        if (std::fread(&e.layer, 4, 1, f) != 1 || std::fread(&e.first, 4, 1, f) != 1 || std::fread(&e.count, 4, 1, f) != 1 || !get_vec(f, e.kv) ||
            !get_vec(f, e.idx))
            return false;
    if (std::fread(&nt, 4, 1, f) != 1) return false;
    b.topk.assign(nt, {});
    for (auto& t : b.topk)
        if (std::fread(&t.layer, 4, 1, f) != 1 || !get_vec(f, t.counts) || !get_vec(f, t.ids)) return false;
    return get_vec(f, b.cand);
}

}  // namespace

int main(int argc, char** argv) {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared card, restricts the process to the chosen one and prints the "device:" receipt line.
    strix::device::bind_cuda_main(argc, argv);
    if (argc < 2) usage("MODEL.gguf [options]");
    const std::string model = argv[1];
    CudaPrefillOptions o;
    o.max_positions = 0;
    std::uint32_t n_tokens = 0, steps = 32, rows = 1, split = 0, stop_steps = 16, force_every = 0;
    std::string ids_file, state_file, ref_file, tiers, json_out, stops_arg, dump_file, tail_file;
    std::string commit_test;  // --commit-test R1,R2,..: partial-commit gate (see the header)
    std::uint32_t commit_cont = 1, commit_skip = 0;
    bool from_zero = false;  // --from-zero: reset_sequence() + the prompt as decode windows of --rows rows
    std::uint32_t dump_layers = 0;
    bool check_run = false, graphs_ab = false, io_bench = false, graph_sections = false, ref_feed = false;
    bool weight_dirs_given = false, weight_verify_given = false;
    try {
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            const auto next = [&]() -> std::string {
                if (i + 1 >= argc) usage(("missing value of " + a).c_str());
                return argv[++i];
            };
            if (a == "--prompt-ids") ids_file = next();
            else if (a == "--tokens") n_tokens = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--import-state") state_file = next();
            else if (a == "--positions") o.max_positions = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--layers") o.end_layer = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--steps") steps = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--rows") rows = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--ref") ref_file = next();
            else if (a == "--check-run") check_run = true;
            else if (a == "--io-bench") io_bench = true;
            else if (a == "--graph-sections") graph_sections = true;  // the step copies the cut's sections itself (io.sections_cut)
            else if (a == "--hashes") o.stream_hashes = true;
            else if (a == "--split") split = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--dump-boundary") dump_file = next();
            else if (a == "--tail-from") tail_file = next();
            else if (a == "--first") o.first_layer = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--tiers") tiers = next();
            else if (a == "--device-gib") o.device_expert_gib = std::stod(next());
            else if (a == "--slots") o.slots = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--eager") o.decode_graphs = false;
            else if (a == "--graphs-ab") graphs_ab = true;
            else if (a == "--graph-warmup") o.decode_graph_warmup = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--bucket") o.decode_bucket = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--timing") o.timing = true;
            else if (a == "--timing-layers") { o.timing = true; o.timing_ops = false; }
            else if (a == "--stops") stops_arg = next();
            else if (a == "--stop-steps") stop_steps = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--no-lean-state") o.lean_state = false;
            else if (a == "--no-spin") o.decode_spin = false;
            else if (a == "--engram") o.engram = next();
            else if (a == "--buffered") o.engram_direct = false;
            else if (a == "--no-engram-prefetch") o.engram_prefetch = false;
            else if (a == "--force-every") force_every = static_cast<std::uint32_t>(std::stoul(next()));  // R1: every N-th step feeds argmax + 1
            else if (a == "--engram-threads") o.engram_threads = static_cast<unsigned>(std::stoul(next()));
            else if (a == "--chunk") o.chunk_rows = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--moe-chunk") o.moe_chunk_rows = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--json") json_out = next();
            else if (a == "--fuse-hc") o.decode_fuse_hc = true;
            else if (a == "--fuse-shared") o.decode_fuse_shared = true;
            else if (a == "--expert-tp") o.expert_tp = true;
            else if (a == "--ref-feed") ref_feed = true;  // R1: feed the reference's fed token of each position (teacher forcing)  // the P15 target identity's routed-expert sum order (decode steps)
            else if (a == "--overlay") o.weights_overlay = next();
            else if (a == "--weight-dirs") { o.weight_read.weight_dirs = next(); weight_dirs_given = true; }
            else if (a == "--drive-readers") o.weight_read.set_drive_readers(next());
            else if (a == "--weight-verify") { o.weight_read.weight_verify = next(); weight_verify_given = true; }
            else if (a == "--commit-test") commit_test = next();
            else if (a == "--from-zero") from_zero = true;
            else if (a == "--commit-cont") commit_cont = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--commit-skip") commit_skip = static_cast<std::uint32_t>(std::stoul(next()));
            else if (a == "--trellis-shexp") o.trellis_shexp = next();
            else usage(("unknown option " + a).c_str());
        }
        o.weight_read.environment(weight_dirs_given, weight_verify_given, model);
        o.weight_read.require_supported(model);
    } catch (const std::exception& e) { usage(e.what()); }
    if (io_bench) {
        // random 264-byte records in the second half of the file (the Engram tables): latency of one decode step's reads
        for (const bool direct : {true, false})
            for (const unsigned threads : {1U, 8U, 64U})
                for (const std::size_t per_job : {std::size_t{0}, std::size_t{1}, std::size_t{32}})
                    for (const std::size_t n : {std::size_t{24}, std::size_t{48}, std::size_t{192}}) {
                        strix::ds41::FileReader fr(model, direct, threads, o.weight_read);
                        std::uint64_t x = 0x9E3779B97F4A7C15ULL ^ (n * 131U + threads);
                        std::vector<double> t;
                        std::vector<std::uint8_t> out(n * 264U);
                        for (int rep = 0; rep < 30; ++rep) {
                            std::vector<std::uint64_t> off(n);
                            for (auto& v : off) {
                                x = x * 6364136223846793005ULL + 1442695040888963407ULL;
                                v = fr.size() / 2U + (x >> 20U) % (fr.size() / 2U - 4096U) / 264U * 264U;
                            }
                            const auto t0 = std::chrono::steady_clock::now();
                            if (per_job == 0U) fr.read_records_queued(off, 264U, out.data());  // per_job 0: queued from this thread
                            else fr.read_records(off, 264U, out.data(), per_job);
                            t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3);
                        }
                        std::printf("{\"event\":\"io_bench\",\"direct\":%s,\"threads\":%u,\"per_job\":%zu,\"records\":%zu,\"median_ms\":%.4f%s}\n",
                                    direct ? "true" : "false", threads, per_job, n, median(t), (fr.multi() ? ",\"weight_io\":" + fr.multi_stats_json() : std::string()).c_str());
                        std::fflush(stdout);
                    }
        return 0;
    }
    if (ids_file.empty() == state_file.empty()) usage("one of --prompt-ids or --import-state");
    if (rows < 1U || rows > 8U) usage("--rows 1..8");
    o.decode_rows = std::max<std::uint32_t>(rows, 1U);
    // Import and from-zero use only step(); prompt/run comparisons retain the prefill object.
    o.decode_only = !state_file.empty() || from_zero;
    std::vector<std::uint32_t> prompt;
    strix::deepseek41::AttentionState imported;
    if (!ids_file.empty()) {
        prompt = read_ids(ids_file);
        if (n_tokens != 0U) prompt.resize(std::min<std::size_t>(n_tokens, prompt.size()));
    } else {
        imported = strix::deepseek41::load_attention_state(state_file);
        prompt = imported.tokens;
    }
    const std::uint32_t T = static_cast<std::uint32_t>(prompt.size());
    if (o.max_positions == 0U) o.max_positions = T + steps * rows + 64U;
    if (!tiers.empty()) {
        o.tiers.assign(40, ExpertTier::file);
        for (std::size_t l = 0; l < tiers.size() && l < 40; ++l)
            o.tiers[l] = tiers[l] == 'D' ? ExpertTier::device : tiers[l] == 'H' ? ExpertTier::host : ExpertTier::file;
    }
    std::map<std::uint32_t, RefStep> ref;
    if (!ref_file.empty()) ref = read_ref(ref_file);
    std::ofstream js;
    if (!json_out.empty()) js.open(json_out, std::ios::app);
    const auto emit = [&](const std::string& line) {
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
        if (js) js << line << "\n";
    };
    int failures = 0;
    std::FILE* dump = nullptr;
    std::FILE* tail_in = nullptr;
    if (!dump_file.empty()) {
        if (state_file.empty() || o.end_layer >= 40U || o.first_layer != 0U) usage("--dump-boundary: a FRONT [0, K) object (--layers K) from --import-state");
        dump_layers = o.end_layer;
        dump = std::fopen(dump_file.c_str(), "wb");
        if (!dump) usage("cannot write the boundary file");
    }
    if (!tail_file.empty()) {
        if (state_file.empty() || o.first_layer == 0U) usage("--tail-from: a TAIL [K, END) object (--first K) from --import-state");
        tail_in = std::fopen(tail_file.c_str(), "rb");
        if (!tail_in) usage("cannot read the boundary file");
    }
    try {
        const auto t_load = std::chrono::steady_clock::now();
        // the objects: one over [0, END) or the FRONT [0, K) + TAIL [K, END)
        CudaPrefillOptions fo = o, to = o;
        if (split) {
            fo.end_layer = split;
            to.first_layer = split;
            fo.slots = to.slots = 0;
        }
        CudaPrefill p(model, fo);
        std::unique_ptr<CudaPrefill> tail;
        if (split) tail = std::make_unique<CudaPrefill>(model, to);
        emit("{\"event\":\"loaded\",\"describe\":" + p.describe() + (tail ? ",\"tail\":" + tail->describe() : std::string()) +
             ",\"load_s\":" + std::to_string(std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count()) +
             p.weight_io_fields() + "}");

        // ---- --from-zero: the prompt as decode windows from position 0
        if (from_zero) {
            if (ids_file.empty() || o.end_layer != 40U || split) usage("--from-zero needs --prompt-ids and the head (one object)");
            p.reset_sequence();
            std::uint32_t compared = 0, matched = 0, nxt = 0;
            const auto check = [&](std::uint32_t pos, std::uint32_t fed, std::uint32_t arg, std::uint64_t f) {
                const auto it = ref.find(pos);
                if (it == ref.end()) return;
                const bool ok = it->second.fed == fed && it->second.argmax == arg && it->second.fnv == f;
                ++compared;
                matched += ok ? 1U : 0U;
                if (!ok) {
                    ++failures;
                    std::printf("{\"event\":\"mismatch\",\"pos\":%u,\"fed\":%u,\"argmax\":%u,\"fnv\":\"%s\",\"ref_argmax\":%u,\"ref_fnv\":\"%s\"}\n", pos, fed, arg,
                                hex(f).c_str(), it->second.argmax, hex(it->second.fnv).c_str());
                }
            };
            for (std::uint32_t at = 0; at < T; at += rows) {
                const std::uint32_t n = std::min(rows, T - at);
                std::vector<std::uint32_t> fed(prompt.begin() + at, prompt.begin() + at + n);
                const std::uint32_t pos = p.position();
                const auto s = p.step(fed, CudaStepIo{});
                for (std::uint32_t r = 0; r < n; ++r) check(pos + r, fed[r], s.tokens.at(r), s.logits_fnv.at(r));
                nxt = s.tokens.back();
                emit("{\"event\":\"fill\"," + step_json(s, fed) + "}");
            }
            for (std::uint32_t k = 0; k < steps; ++k) {
                const std::uint32_t pos = p.position();
                const std::vector<std::uint32_t> fed{nxt};
                const auto s = p.step(fed, CudaStepIo{});
                check(pos, nxt, s.tokens.at(0), s.logits_fnv.at(0));
                nxt = s.tokens.at(0);
            }
            emit("{\"event\":\"from_zero\",\"prompt\":" + std::to_string(T) + ",\"rows\":" + std::to_string(rows) + ",\"steps\":" + std::to_string(steps) +
                 ",\"compared\":" + std::to_string(compared) + ",\"matched\":" + std::to_string(matched) + "}");
            return failures ? 1 : 0;
        }

        // ---- the sequence's start
        std::uint32_t next = 0;
        if (!state_file.empty()) {
            const auto t0 = std::chrono::steady_clock::now();
            p.import_state(imported);
            if (tail) tail->import_state(imported);
            next = imported.next_token;
            emit("{\"event\":\"imported\",\"position\":" + std::to_string(imported.position) + ",\"next_token\":" + std::to_string(next) +
                 ",\"digest\":\"" + hex(imported.digest()) + "\",\"s\":" +
                 std::to_string(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()) + "}");
        } else {
            if (split) usage("--split needs --import-state (the TAIL starts from a state)");
            const auto r = p.run(prompt);
            next = r.next_token;
            emit("{\"event\":\"prefill\",\"tokens\":" + std::to_string(T) + ",\"seconds\":" + std::to_string(r.seconds) + ",\"next_token\":" +
                 std::to_string(next) + ",\"logits_fnv\":\"" + hex(fnv(r.logits.data(), r.logits.size())) + "\"}");
        }

        // ---- --check-run: step(next) after run(T) == run(T + 1)
        if (check_run) {
            if (state_file.empty() == false) usage("--check-run needs --prompt-ids");
            CudaStepIo io;
            const std::vector<std::uint32_t> fed{next};
            const auto s = p.step(fed, io);
            auto longer = prompt;
            longer.push_back(next);
            const auto r = p.run(longer);  // a fresh sequence of T + 1 tokens
            const std::uint64_t f_step = s.logits_fnv.at(0), f_run = fnv(r.logits.data(), r.logits.size());
            bool same = f_step == f_run && s.tokens.at(0) == r.next_token;
            std::ostringstream os;
            os << "{\"event\":\"check_run\",\"T\":" << T << ",\"step_token\":" << s.tokens[0] << ",\"run_token\":" << r.next_token << ",\"step_fnv\":\""
               << hex(f_step) << "\",\"run_fnv\":\"" << hex(f_run) << "\"";
            if (!s.stream_hashes.empty() && !r.stream_hashes.empty()) {
                std::uint32_t diff = 0;
                for (std::size_t l = 0; l < std::min(s.stream_hashes.size(), r.stream_hashes.size()); ++l) diff += s.stream_hashes[l] != r.stream_hashes[l];
                os << ",\"layers_compared\":" << std::min(s.stream_hashes.size(), r.stream_hashes.size()) << ",\"layers_differ\":" << diff;
                same = same && diff == 0U;
            }
            os << ",\"identical\":" << (same ? "true" : "false") << "}";
            emit(os.str());
            if (!same) ++failures;
            return failures ? 1 : 0;
        }

        // ---- partial commit gate
        if (!commit_test.empty()) {
            if (state_file.empty() || split) usage("--commit-test needs --import-state (one object)");
            std::vector<std::uint32_t> Rs;
            {
                std::stringstream ss(commit_test);
                std::string item;
                while (std::getline(ss, item, ',')) Rs.push_back(static_cast<std::uint32_t>(std::stoul(item)));
            }
            for (const auto r : Rs)
                if (r < 1U || r > o.decode_rows) usage("--commit-test rows 1..--rows");
            const bool has_head = o.end_layer == 40U;
            std::uint64_t lcg = 0x243F6A8885A308D3ULL;
            const auto other = [&]() {
                lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
                return static_cast<std::uint32_t>((lcg >> 33U) % 129280U);
            };
            std::vector<float> so(std::size_t{o.decode_rows} * 4U * 5120U), mo(std::size_t{o.decode_rows} * 24U);
            // one step; per-row FNV of its outputs
            const auto run_step = [&](const std::vector<std::uint32_t>& fed, std::vector<std::uint64_t>& rf, std::vector<std::uint32_t>& toks) {
                CudaStepIo io;
                if (!has_head) {
                    io.stream_out = so.data();
                    io.mix_out = mo.data();
                }
                const auto r = p.step(fed, io);
                rf.clear();
                for (std::size_t i = 0; i < fed.size(); ++i) {
                    if (has_head) rf.push_back(r.logits_fnv.at(i));
                    else rf.push_back(fnv(so.data() + i * 4U * 5120U, 4U * 5120U) ^ (fnv(mo.data() + i * 24U, 24U) * 31U));
                }
                toks = r.tokens;
            };
            struct Check {
                std::uint32_t pos;
                std::vector<std::pair<std::string, std::uint64_t>> h;
            };
            std::vector<std::uint32_t> committed;  // fed tokens of every committed position, in order
            std::vector<std::uint64_t> fnv_win;    // their row FNVs in the windowed run
            std::vector<Check> checks;
            std::vector<std::uint64_t> rf;
            std::vector<std::uint32_t> toks;
            std::uint32_t nxt = next, windows = 0;
            const auto r1 = [&]() {
                run_step({nxt}, rf, toks);
                committed.push_back(nxt);
                fnv_win.push_back(rf[0]);
                nxt = has_head ? toks.at(0) : other();
                p.commit_window(1U, nxt);  // a whole step (kept == R): only the next token
            };
            const auto t0 = std::chrono::steady_clock::now();
            for (std::uint32_t k = 0; k < commit_skip; ++k) r1();
            for (const auto R : Rs)
                for (std::uint32_t a = 0; a < R; ++a) {
                    std::vector<std::uint32_t> fed(R);
                    fed[0] = nxt;
                    for (std::uint32_t j = 1; j < R; ++j) fed[j] = other();
                    const std::uint32_t pos = p.position();
                    run_step(fed, rf, toks);
                    const std::uint32_t kept = a + 1U;
                    for (std::uint32_t j = 0; j < kept; ++j) {
                        committed.push_back(fed[j]);
                        fnv_win.push_back(rf[j]);
                    }
                    nxt = has_head ? toks.at(a) : other();
                    p.commit_window(kept, nxt);
                    if (p.position() != pos + kept) usage("commit_window: position");
                    checks.push_back({p.position(), p.export_state().hashes()});
                    ++windows;
                    for (std::uint32_t k = 0; k < commit_cont; ++k) r1();
                }
            const double win_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            // the R1 replay of the committed tokens
            p.import_state(imported);
            std::uint32_t row_bad = 0, state_bad = 0, states = 0;
            std::size_t ci = 0;
            std::string first_bad;
            for (std::size_t i = 0; i < committed.size(); ++i) {
                const std::uint32_t pos = p.position();
                run_step({committed[i]}, rf, toks);
                p.commit_window(1U, i + 1U < committed.size() ? committed[i + 1U] : 0U);
                if (rf[0] != fnv_win[i]) {
                    ++row_bad;
                    if (first_bad.empty()) first_bad = "row pos " + std::to_string(pos);
                }
                while (ci < checks.size() && checks[ci].pos == pos + 1U) {
                    ++states;
                    if (p.export_state().hashes() != checks[ci].h) {
                        ++state_bad;
                        if (first_bad.empty()) first_bad = "state pos " + std::to_string(pos + 1U);
                    }
                    ++ci;
                }
            }
            const bool same = row_bad == 0U && state_bad == 0U && states == checks.size();
            std::ostringstream os;
            os << "{\"event\":\"commit_test\",\"head\":" << (has_head ? "true" : "false") << ",\"layers\":[" << o.first_layer << "," << o.end_layer
               << "],\"windows\":" << windows << ",\"rows_compared\":" << committed.size() << ",\"rows_differ\":" << row_bad
               << ",\"states_compared\":" << states << ",\"states_differ\":" << state_bad << ",\"checks\":" << checks.size()
               << ",\"first_pos\":" << imported.position << ",\"last_pos\":" << imported.position + committed.size() << ",\"windowed_s\":" << win_s
               << ",\"first_bad\":\"" << first_bad << "\",\"identical\":" << (same ? "true" : "false") << "}";
            emit(os.str());
            return same ? 0 : 1;
        }

        // ---- greedy steps
        std::vector<double> ms, dev_ms;
        std::uint32_t matched = 0, compared = 0;
        const std::uint32_t K = split ? split : dump_layers;
        const bool front_mode = split != 0U || dump != nullptr;
        std::vector<float> s_out, m_out;
        for (std::uint32_t k = 0; k < steps; ++k) {
            if (graphs_ab) p.set_decode_graphs(k % 2U == 1U);
            const std::uint32_t pos = p.position();
            std::vector<std::uint32_t> fed(rows, next);
            if (force_every != 0U && rows == 1U && k > 0U && k % force_every == 0U) fed[0] = (next + 1U) % 129280U;  // not the greedy token
            if (ref_feed && rows == 1U && ref.count(pos) != 0U) fed[0] = ref.at(pos).fed;
            if (rows > 1U && !ref.empty())
                for (std::uint32_t r = 1; r < rows; ++r) {
                    const auto it = ref.find(pos + r);
                    if (it != ref.end()) fed[r] = it->second.fed;
                }
            CudaStepIo io;
            CudaStepResult s;
            if (front_mode) {
                // FRONT [0, K): stream + mix out, then the boundary sections (into the TAIL object or the dump file)
                Boundary bd;
                bd.pos = pos;
                bd.tokens = fed;
                bd.stream.resize(std::size_t{rows} * 4U * 5120U);
                bd.mix.resize(std::size_t{rows} * 24U);
                io.stream_out = bd.stream.data();
                io.mix_out = bd.mix.data();
                if (graph_sections) io.sections_cut = K;
                s = p.step(fed, io);
                if (graph_sections) {
                    for (const auto& e : s.sections.kv) bd.kv.push_back({e.layer, e.first, e.count, e.kv, e.idx_keys});
                    for (const auto& t : s.sections.topk) bd.topk.push_back({t.layer, t.counts, t.ids});
                    bd.cand = s.sections.candidates;
                } else {
                    collect_sections(p, bd, K);
                }
                if (tail) {
                    apply_sections(*tail, bd);
                    CudaStepIo ti;
                    ti.stream_in = bd.stream.data();
                    ti.mix_in = bd.mix.data();
                    const auto t = tail->step(fed, ti);
                    const double front_s = s.seconds;
                    const float front_dev = s.device_ms;
                    s.tokens = t.tokens;
                    s.logits_fnv = t.logits_fnv;
                    s.seconds = front_s + t.seconds;
                    s.device_ms = front_dev + t.device_ms;
                }
                if (dump) write_boundary(dump, bd);
            } else if (tail_in) {
                Boundary bd;
                if (!read_boundary(tail_in, bd)) usage("boundary file ended");
                if (bd.pos != pos || bd.tokens != fed) usage("boundary file: position / tokens differ from the sequence");
                apply_sections(p, bd);
                io.stream_in = bd.stream.data();
                io.mix_in = bd.mix.data();
                s = p.step(fed, io);
            } else {
                s = p.step(fed, io);
            }
            if (!s.tokens.empty()) next = s.tokens.back();
            else if (!ref.empty() && ref.count(pos + rows) != 0U) next = ref.at(pos + rows).fed;  // no head: the reference's greedy tokens
            std::string cmp;
            if (!ref.empty() && !s.tokens.empty()) {
                const auto it = ref.find(pos + rows - 1U);
                if (it != ref.end()) {
                    const bool ok = it->second.fed == fed.back() && it->second.argmax == s.tokens.back() &&
                                    (s.logits_fnv.empty() || it->second.fnv == s.logits_fnv.back());
                    ++compared;
                    matched += ok ? 1U : 0U;
                    if (!ok) ++failures;
                    cmp = ",\"ref_argmax\":" + std::to_string(it->second.argmax) + ",\"ref_fnv\":\"" + hex(it->second.fnv) + "\",\"match\":" + (ok ? "true" : "false");
                }
            }
            emit("{\"event\":\"step\"," + step_json(s, fed) + cmp + "}");
            if (k >= 2U) {
                ms.push_back(s.seconds * 1e3);
                dev_ms.push_back(s.device_ms);
            }
            if (rows > 1U && ref.empty()) next = s.tokens.back();
        }
        {
            std::ostringstream os;
            os << "{\"event\":\"summary\",\"steps\":" << steps << ",\"rows\":" << rows << ",\"median_ms\":" << median(ms) << ",\"median_device_ms\":" << median(dev_ms)
               << ",\"tok_s\":" << (median(ms) > 0 ? 1e3 * rows / median(ms) : 0.0) << ",\"compared\":" << compared << ",\"matched\":" << matched
               << p.weight_io_fields() << "}";
            emit(os.str());
        }

        // ---- FRONT prefixes [0, K)
        if (!stops_arg.empty() && !tail) {
            std::stringstream ss(stops_arg);
            std::string item;
            while (std::getline(ss, item, ',')) {
                const auto stop = static_cast<std::uint32_t>(std::stoul(item));
                std::vector<double> w, d;
                for (std::uint32_t k = 0; k < stop_steps; ++k) {
                    std::vector<std::uint32_t> fed(rows, next);
                    CudaStepIo io;
                    s_out.resize(std::size_t{rows} * 4U * 5120U);
                    m_out.resize(std::size_t{rows} * 24U);
                    io.stream_out = s_out.data();
                    io.mix_out = m_out.data();
                    io.stop_layer = stop;
                    const auto s = p.step(fed, io);
                    if (k >= 2U) {
                        w.push_back(s.seconds * 1e3);
                        d.push_back(s.device_ms);
                    }
                    if (k + 1U == stop_steps) emit("{\"event\":\"front_step\",\"stop\":" + std::to_string(stop) + "," + step_json(s, fed) + "}");
                }
                std::ostringstream os;
                os << "{\"event\":\"front\",\"stop\":" << stop << ",\"rows\":" << rows << ",\"median_ms\":" << median(w) << ",\"median_device_ms\":" << median(d)
                   << "}";
                emit(os.str());
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-cuda-ds41-decode: %s\n", e.what());
        return 1;
    }
    return failures ? 1 : 0;
}
