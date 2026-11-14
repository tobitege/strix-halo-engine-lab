#include "strix/device/hip_gate.hpp"
#include "strix/dist/open_transport.hpp"
#include "strix/deepseek41_batch_forward.hpp"
#include "strix/deepseek41_device_forward.hpp"
#include "strix/deepseek41_pair_defaults.hpp"
#include "strix/deepseek41_prefill.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/dist/gpu_direct.hpp"
#include "strix/ds41_expert_tp.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

// deepseek41 prefill of a prompt, one node (world 1, every expert of the layer range resident) or one rank of two
// (expert halves, half sums exchanged per chunk), or both ranks in one process (--transport emulate).
//   strix-deepseek41-prefill MODEL.gguf --prompt FILE (comma-separated ids) [--rows N] [--layers L]
//       [--world 1 | --rank R --transport usb4-auto | usb4-gpu | tcp-listen:PORT | tcp-connect:HOST:PORT | emulate]
//       [--chunk C] [--window W] [--lanes N] [--moe-chunk R] [--no-tp] [--row-split | --no-row-split] [--stream-experts] [--timing] [--reference] [--repeat K] [--connect-timeout S] [--no-cone] [--warmup N] [--split-min-rows N] [--joint-max-rows N] [--state-timing] [--loads] [--loads-out CSV] [--stream-hashes-out FILE] [--draft-rows N --seed-out PREFIX]
//       [--ttft-timing [--ttft-warmup R] [--ttft-decode G]]
//       [--task-levels whole,layer,phase,finest [--task-rows N,...] [--task-units K | --task-budget MS] [--task-moe-chunk R] [--task-table]
//        [--interleave G [--bg-rows N]] [--mem-guard-gib G]]
// --embed-identity A:B (world 1): the identity gate of provided embedding rows, see embed_identity_gate.
// --task-levels: after the runs, the prompt again as resumable PrefillTasks, one per listed unit level ("whole": one
// step() of every unit; "level:R": with MoE chunks of R rows; "level[:R]@MS": units_within(MS) units per step), K units per step() (default 1) or as many as units_within(MS) predicts (rank 0 decides, rank 1
// follows); each task's next token, logits, last row's streams, state hashes and exported state digest must equal run()'s.
// --interleave G (two ranks): a BatchForward (the prefill reads its weights in place, as the serve) decodes another
// request (slot 0: the prompt's first --bg-rows rows, default 16, one per step, then G greedy tokens) alone first, then again with one
// of its steps after every step() of each task: its tokens must equal the alone run's. Prints the unit table (device ms
// per unit, by kind and by layer) and the step() wall times.
// --ttft-timing: one "ttft" line per stage from the process start to the first token (exec to main, prompt read, transport,
// communicator, weights, barrier, each run's wall time on the host); --ttft-warmup R: one run of the prompt's first R rows
// before the measured runs (the first run of a process pays one-time costs); --ttft-decode G: after the runs, the state
// goes to a decode forward (export, prefill freed, decode weights, import) and G greedy steps from the prefill's next
// token are timed one by one (the first step pays the decode forward's one-time costs).
// --reference (world 1 or emulate): the decode forward fed the same tokens one by one; the last row's logits, the last
// row's stream after every layer and every layer's attention state must be equal bit for bit.
namespace {

// Both ranks here before either goes on: a 4-byte exchange, waited for.
void barrier(strix::dist::Communicator& comm, std::uint32_t round) {
    // one buffer and stream per thread for the process (emulate: one per rank thread), not one per call
    thread_local void* buffers = nullptr;
    thread_local hipStream_t stream = nullptr;
    if (buffers == nullptr && (hipMalloc(&buffers, 8) != hipSuccess || hipStreamCreate(&stream) != hipSuccess)) throw std::runtime_error("barrier buffers");
    (void)hipMemsetAsync(buffers, 0, 8, stream);
    comm.exchange(stream, buffers, static_cast<char*>(buffers) + 4, 4,
                  strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 2U, round});
    (void)hipStreamSynchronize(stream);
}

struct Args {
    std::string model, prompt_path, transport, loads_out, hashes_out, seed_out;
    std::uint32_t rank = 0, world = 1, layers = 40, rows = 0, chunk = 1024, window = 8, lanes = 4, moe_chunk = 4096, repeat = 1, warmup = 0,
                  split_min_rows = strix::deepseek41::PrefillOptions{}.split_min_rows,
                  joint_max_rows = strix::deepseek41::PrefillOptions{}.joint_max_rows;
    unsigned connect_timeout = 60;
    bool embed_identity = false;  // --embed-identity A:B
    std::uint32_t embed_a = 0, embed_b = 0;
    bool select_ab = false;  // --select-ab: run k with the exact dense selector off (k even) / on (k odd), one load
    std::uint32_t draft_rows = 0; // --draft-rows N, optional --seed-out PREFIX for raw seed gates
    std::uint32_t extend_check = 0;  // --extend-check S: after the runs, run() + extend() in S-row pieces == the whole run
    std::string select_pattern;  // --select-pattern OSSO...: run k off ('O') / on ('S') by the k-th letter (cyclic)
    bool timing = false, reference = false, cone = true, loads = false, tp = true, stream_experts = false, row_split = true, decode_reference = false,
         state_timing = false;
    bool ttft = false;  // --ttft-timing
    std::string task_levels;  // --task-levels
    std::vector<std::uint32_t> task_rows;  // --task-rows: prompt prefixes gated in turn (default: the whole prompt)
    std::uint32_t task_units = 1, interleave = 0, bg_rows = 16;
    std::uint32_t task_moe_chunk = strix::deepseek41::PrefillTaskOptions{}.moe_chunk_rows;  // --task-moe-chunk (0: the prefill's)
    std::uint32_t task_engram_rows = strix::deepseek41::PrefillTaskOptions{}.engram_rows;  // --task-engram-rows
    double task_budget = 0.0;
    double mem_guard_gib = 16.0;  // --mem-guard-gib (0: off)
    bool task_table = false;
    std::uint32_t ttft_warmup = 0, ttft_decode = 0;
};

void validate_embed_identity_args(const Args& a) {
    if (a.embed_identity && (a.ttft || a.world != 1U))
        throw std::invalid_argument("--embed-identity requires --world 1 without --ttft-timing");
}

// --ttft-timing: wall time of each stage (host clock), printed as "ttft <stage>: <ms since the previous line> ms (at <ms
// since main> ms)".
struct TtftClock {
    bool on = false;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now(), last = t0;
    static double ms(std::chrono::steady_clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
    void line(const char* what) {
        const auto now = std::chrono::steady_clock::now();
        if (on) {
            std::printf("ttft %s: %.3f ms (at %.3f ms)\n", what, ms(now - last), ms(now - t0));
            std::fflush(stdout);
        }
        last = now;
    }
    // exec to main: the process start time of /proc/self/stat (clock ticks since boot) against /proc/uptime
    void exec_to_main() const {
        if (!on) return;
        std::ifstream st("/proc/self/stat"), up("/proc/uptime");
        const std::string s((std::istreambuf_iterator<char>(st)), std::istreambuf_iterator<char>());
        double uptime = 0.0;
        up >> uptime;
        const auto close = s.rfind(')');
        if (close == std::string::npos || uptime <= 0.0) return;
        std::istringstream rest(s.substr(close + 2));
        std::string field;
        const double tick = static_cast<double>(::sysconf(_SC_CLK_TCK));
        for (int i = 3; i <= 22 && rest >> field; ++i)
            if (i == 22) std::printf("ttft exec to main: %.0f ms (clock ticks of %.0f ms)\n", (uptime - std::stod(field) / tick) * 1e3, 1e3 / tick);
    }
};
TtftClock ttft_clock;

std::uint64_t fnv(const std::vector<float>& v) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const float f : v) {
        std::uint32_t b;
        std::memcpy(&b, &f, 4);
        for (int k = 0; k < 4; ++k, b >>= 8U) h = (h ^ (b & 0xffU)) * 1099511628211ULL;
    }
    return h;
}

std::vector<std::uint32_t> read_prompt(const std::string& path, std::uint32_t rows) {
    std::ifstream in(path);
    if (!in) throw std::invalid_argument("cannot read " + path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<std::uint32_t> ids;
    std::stringstream ss(text);
    for (std::string t; std::getline(ss, t, ',');)
        if (!t.empty()) ids.push_back(static_cast<std::uint32_t>(std::stoul(t)));
    if (rows != 0U) {
        if (rows > ids.size()) throw std::invalid_argument("prompt shorter than --rows");
        ids.resize(rows);
    }
    return ids;
}

void print_result(const char* who, const strix::deepseek41::PrefillResult& r, std::size_t rows, bool timing, const std::string& loads_out = {}) {
    std::printf("%s: %zu rows in %.3f s = %.1f tok/s (engram read %.3f s, head %.3f s; %llu row-layers); next token %u, logits_fnv %016llx\n",
                who, rows, r.seconds, static_cast<double>(rows) / r.seconds, r.engram_seconds, r.head_seconds,
                static_cast<unsigned long long>(r.row_layers), r.next_token, static_cast<unsigned long long>(fnv(r.logits)));
    if (r.engram_wait_seconds > 0.0) std::printf("  engram read overlapped: waited %.3f s for it\n", r.engram_wait_seconds);
    if (r.expert_read_seconds > 0.0)
        std::printf("  streamed experts: %.3f s of file reads on the prefetch thread, %.3f s waited for them\n", r.expert_read_seconds,
                    r.expert_wait_seconds);
    if (!r.routing_loads.empty() && !loads_out.empty()) {  // one line per layer: the entries of every expert
        std::FILE* f = std::fopen(loads_out.c_str(), "w");
        if (f == nullptr) throw std::runtime_error("cannot write " + loads_out);
        for (const auto& e : r.routing_loads) {
            for (std::size_t i = 0; i < e.size(); ++i) std::fprintf(f, i ? ",%u" : "%u", e[i]);
            std::fputc('\n', f);
        }
        std::fclose(f);
    }
    if (!r.routing_loads.empty()) {  // entries per expert half of every layer's output rows, and the largest expert
        std::uint64_t all_low = 0, all_high = 0;
        for (std::size_t l = 0; l < r.routing_loads.size(); ++l) {
            const auto& e = r.routing_loads[l];
            std::uint64_t low = 0, high = 0, top = 0;
            for (std::size_t i = 0; i < e.size(); ++i) {
                (i < e.size() / 2U ? low : high) += e[i];
                top = std::max<std::uint64_t>(top, e[i]);
            }
            all_low += low;
            all_high += high;
            std::printf("  loads layer %2zu: low %7llu high %7llu (high/low %.3f), largest expert %llu\n", l, static_cast<unsigned long long>(low),
                        static_cast<unsigned long long>(high), low ? static_cast<double>(high) / static_cast<double>(low) : 0.0,
                        static_cast<unsigned long long>(top));
        }
        std::printf("  loads all layers: low %llu high %llu (high/low %.3f)\n", static_cast<unsigned long long>(all_low),
                    static_cast<unsigned long long>(all_high), all_low ? static_cast<double>(all_high) / static_cast<double>(all_low) : 0.0);
    }
    if (!timing) return;
    // attention and moe include their own exchanges; other = layer - attention - moe; the exchange columns are
    // inside those intervals (wall, and CPU blocked in finish()), not additional time
    double tot = 0, at = 0, mo = 0, ex = 0, bl = 0;
    std::printf("  (ms; attention and moe include their exchanges; exchange wall / blocked are inside them, do not add)\n");
    for (std::size_t l = 0; l < r.layers.size(); ++l) {
        const auto& t = r.layers[l];
        std::printf("  layer %2zu %8.2f  attention %8.2f  moe %8.2f  other %7.2f  | exchange wall %7.2f blocked %7.2f\n", l, t.seconds * 1e3,
                    t.attention * 1e3, t.moe * 1e3, (t.seconds - t.attention - t.moe) * 1e3, t.exchange * 1e3, t.exchange_blocked * 1e3);
        tot += t.seconds;
        at += t.attention;
        mo += t.moe;
        ex += t.exchange;
        bl += t.exchange_blocked;
    }
    std::printf("  sum      %8.2f  attention %8.2f  moe %8.2f  other %7.2f  | exchange wall %7.2f blocked %7.2f\n", tot * 1e3, at * 1e3,
                mo * 1e3, (tot - at - mo) * 1e3, ex * 1e3, bl * 1e3);
    const auto& ap = r.attention_parts;
    std::printf("  attention parts: projections %.2f  window K/V + compressor %.2f  q_b %.2f  windows %.2f  o_a + wo_b %.2f ms\n",
                ap[0] * 1e3, ap[1] * 1e3, ap[2] * 1e3, ap[3] * 1e3, ap[4] * 1e3);
}

void write_receipts(const Args& a, const strix::deepseek41::DevicePrefill& pre,
                    const strix::deepseek41::PrefillResult& r, unsigned rank) {
    if (!a.hashes_out.empty()) {  // one line per layer stream, then the attention state (runs compared with diff)
        std::FILE* f = std::fopen(a.hashes_out.c_str(), "w");
        if (f == nullptr) throw std::runtime_error("cannot write " + a.hashes_out);
        for (std::size_t l = 0; l < r.stream_hashes.size(); ++l)
            std::fprintf(f, "stream@%zu %016llx\n", l, static_cast<unsigned long long>(r.stream_hashes[l]));
        for (const auto& [name, h] : pre.state_hashes()) std::fprintf(f, "%s %016llx\n", name.c_str(), static_cast<unsigned long long>(h));
        std::fprintf(f, "logits %016llx\n", static_cast<unsigned long long>(fnv(r.logits)));
        std::fclose(f);
    }
    if (a.draft_rows != 0U) {
        const auto seed = pre.draft_seed();
        for (unsigned j = 0; j < seed.streams.size(); ++j) {
            std::vector<float> data(std::size_t{seed.rows} * seed.row_elements);
            if (!data.empty() && hipMemcpy(data.data(), seed.streams[j], data.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess)
                throw std::runtime_error("seed receipt copy failed");
            std::printf("rank %u seed %u rows %u first %u elements %u hash %016llx\n", rank, j, seed.rows, seed.first,
                        seed.row_elements, static_cast<unsigned long long>(fnv(data)));
            if (!a.seed_out.empty()) {
                const auto path = a.seed_out + "." + std::to_string(j) + ".bin";
                std::FILE* f = std::fopen(path.c_str(), "wb");
                if (!f) throw std::runtime_error("cannot write seed receipt " + path);
                const auto written = std::fwrite(data.data(), sizeof(float), data.size(), f);
                const int closed = std::fclose(f);
                if (written != data.size() || closed) throw std::runtime_error("seed receipt write failed");
            }
        }
    }
}

strix::deepseek41::PrefillOptions options(const Args& a, std::size_t rows) {
    strix::deepseek41::PrefillOptions o;
    o.layers = a.layers;
    o.draft_rows = a.draft_rows;
    o.max_positions = static_cast<std::uint32_t>(rows);
    o.chunk_rows = a.chunk;
    o.window_rows = a.window;
    o.attention_lanes = a.lanes;
    o.moe_chunk_rows = a.moe_chunk;
    o.tp_attention = a.tp;
    o.row_split = a.row_split;
    o.stream_experts = a.stream_experts;
    o.stream_hashes = a.embed_identity || a.reference || !a.hashes_out.empty();
    o.verify_embed_extras = a.embed_identity;
    o.timing = a.timing;
    o.cone = a.cone;
    o.routing_loads = a.loads;
    o.warmup_rows = a.warmup;
    o.split_min_rows = a.split_min_rows;
    o.joint_max_rows = a.joint_max_rows;
    return o;
}

// --ttft-timing: the optional warm-up run, then --repeat runs of the prompt, each from the barrier (two ranks) to the
// result on the host; returns the last run's result.
strix::deepseek41::PrefillResult ttft_runs(const Args& a, strix::deepseek41::DevicePrefill& pre, const std::vector<std::uint32_t>& prompt,
                                           strix::dist::Communicator* comm, bool print, const std::string& loads_out) {
    strix::deepseek41::PrefillResult r;
    const auto one = [&](std::span<const std::uint32_t> rows, std::uint32_t round, const char* label) {
        if (comm != nullptr) barrier(*comm, round);
        ttft_clock.line("barrier");
        const auto t0 = std::chrono::steady_clock::now();
        r = pre.run(rows);
        const double wall = TtftClock::ms(std::chrono::steady_clock::now() - t0);
        ttft_clock.line(label);
        if (print) {
            std::printf("ttft %s: %zu rows, wall %.3f ms (run %.3f ms, engram read %.3f ms, head %.3f ms), next token %u, logits_fnv %016llx\n", label,
                        rows.size(), wall, r.seconds * 1e3, r.engram_seconds * 1e3, r.head_seconds * 1e3, r.next_token,
                        static_cast<unsigned long long>(fnv(r.logits)));
            std::fflush(stdout);
        }
    };
    if (a.ttft_warmup != 0U)
        one(std::span<const std::uint32_t>(prompt).first(std::min<std::size_t>(a.ttft_warmup, prompt.size())), 900U, "warm-up run");
    for (std::uint32_t k = 0; k < a.repeat; ++k) {
        char label[32];
        std::snprintf(label, sizeof(label), "run %u", k);
        one(prompt, k, label);
        if (print) print_result(comm == nullptr ? "prefill" : "rank prefill", r, prompt.size(), a.timing, loads_out);
    }
    return r;
}

// --ttft-decode: the prefill's state into a decode forward (the prefill already freed), then G greedy steps, each timed.
void ttft_decode(const Args& a, const std::vector<std::uint32_t>& prompt, strix::deepseek41::AttentionState state, std::uint32_t next,
                 strix::dist::Communicator* comm, std::uint32_t rank, bool print) {
    strix::deepseek41::DeviceForwardOptions fo;
    fo.layers = a.layers;
    fo.max_positions = static_cast<std::uint32_t>(prompt.size() + a.ttft_decode + 8U);
    fo.rank = rank;
    fo.world = comm != nullptr ? 2U : 1U;
    fo.communicator = comm;
    fo.op_hashes = false;
    fo.gather_logits = true;
    fo.engram_direct_io = true;
    fo.tp_attention = comm != nullptr;
    strix::deepseek41::DeviceForward fwd(a.model, fo);
    ttft_clock.line("decode weights load");
    if (print)
        std::printf("ttft decode forward: device weights %.2f GiB loaded in %.1f s\n", static_cast<double>(fwd.weight_bytes()) / 1073741824.0,
                    fwd.load_seconds());
    state.tokens = prompt;
    state.next_token = next;
    fwd.import_state(state);
    ttft_clock.line("import state");
    if (comm != nullptr) barrier(*comm, 950U);
    ttft_clock.line("barrier");
    std::uint32_t token = next;
    for (std::uint32_t g = 0; g < a.ttft_decode; ++g) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& logits = fwd.step(token);
        const double wall = TtftClock::ms(std::chrono::steady_clock::now() - t0);
        ttft_clock.line(g == 0U ? "first decode step" : "decode step");
        if (print) {
            std::printf("ttft decode step %u: pos %zu fed %u argmax %u, wall %.3f ms (step %.3f ms), logits_fnv %016llx\n", g, prompt.size() + g, token,
                        fwd.next_token(), wall, fwd.last_step_seconds() * 1e3, static_cast<unsigned long long>(fnv(logits)));
            std::fflush(stdout);
        }
        token = fwd.next_token();
    }
}

// ---- resumable prefill gates (--task-levels) ----

std::uint64_t meminfo_kb(const char* path, const char* key) {
    std::ifstream in(path);
    const std::size_t k = std::strlen(key);
    for (std::string line; std::getline(in, line);)
        if (line.compare(0, k, key) == 0) return std::stoull(line.substr(k));
    return 0;
}

// memory of this process and node: device used (hipMemGetInfo), VmRSS / VmHWM, MemAvailable
void mem_line(std::uint32_t rank, const char* stage) {
    std::size_t free_b = 0, total_b = 0;
    (void)hipMemGetInfo(&free_b, &total_b);
    const double gib = 1073741824.0, kib = 1048576.0;
    std::printf("rank %u memory [%s]: device used %.2f GiB of %.2f, VmRSS %.2f GiB, VmHWM %.2f GiB, MemAvailable %.2f GiB\n", rank, stage,
                static_cast<double>(total_b - free_b) / gib, static_cast<double>(total_b) / gib,
                static_cast<double>(meminfo_kb("/proc/self/status", "VmRSS:")) / kib, static_cast<double>(meminfo_kb("/proc/self/status", "VmHWM:")) / kib,
                static_cast<double>(meminfo_kb("/proc/meminfo", "MemAvailable:")) / kib);
    std::fflush(stdout);
}

// --mem-guard-gib G: abort before a load that would leave less than G GiB of MemAvailable (estimate: the serve's measured
// footprint per rank, BatchForward 78.2 + prefill 7.6 GiB at 40 layers, scaled by layers, + 4 GiB of scratch); the
// prefill also checks its exact bytes
void mem_guard(const Args& a, std::uint32_t world, const char* what) {
    if (a.mem_guard_gib <= 0.0) return;
    const double need = (world == 2U ? 86.0 : 160.0) * a.layers / 40.0 + 4.0;
    const double avail = static_cast<double>(meminfo_kb("/proc/meminfo", "MemAvailable:")) / 1048576.0;
    std::printf("memory guard (%s): estimate %.1f GiB + margin %.1f GiB, MemAvailable %.1f GiB\n", what, need, a.mem_guard_gib, avail);
    std::fflush(stdout);
    if (need + a.mem_guard_gib > avail) throw std::runtime_error(std::string("memory guard: not enough MemAvailable for ") + what);
}

// rank 0's value of k on both ranks (a 4-byte exchange)
std::uint32_t agree(strix::dist::Communicator& comm, std::uint32_t k, std::uint32_t rank, std::uint32_t round) {
    thread_local std::uint32_t* buffers = nullptr;
    thread_local hipStream_t stream = nullptr;
    if (buffers == nullptr && (hipMalloc(reinterpret_cast<void**>(&buffers), 8) != hipSuccess || hipStreamCreate(&stream) != hipSuccess))
        throw std::runtime_error("agree buffers");
    std::uint32_t host[2] = {k, 0};
    (void)hipMemcpyAsync(buffers, host, 8, hipMemcpyHostToDevice, stream);
    comm.exchange(stream, buffers, buffers + 1, 4, strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 4U, round});
    (void)hipMemcpyAsync(host, buffers, 8, hipMemcpyDeviceToHost, stream);
    (void)hipStreamSynchronize(stream);
    return rank == 0U ? host[0] : host[1];
}

// another request on a BatchForward: its prompt one row per step, then greedy tokens
struct Background {
    strix::deepseek41::BatchForward* fwd = nullptr;
    std::vector<std::uint32_t> prompt, out;
    std::uint32_t want = 0, fed = 0, next = 0;
    double max_ms = 0.0, sum_ms = 0.0;
    std::uint32_t steps = 0;
    void start(strix::deepseek41::BatchForward& f, std::vector<std::uint32_t> p, std::uint32_t g) {
        fwd = &f;
        prompt = std::move(p);
        want = g;
        fed = next = 0;
        out.clear();
        max_ms = sum_ms = 0.0;
        steps = 0;
        fwd->reset(0U);
    }
    [[nodiscard]] bool done() const { return fwd == nullptr || out.size() >= want; }
    void step() {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<strix::deepseek41::BatchRow> rows;
        if (fed < prompt.size()) {
            const std::uint32_t n = 1U;  // one row per step (a multi-row segment may not start before the first compressed entry)
            for (std::uint32_t i = 0; i < n; ++i) rows.push_back({0U, prompt[fed + i]});
            fed += n;
            const auto& t = fwd->step(rows);
            if (fed == prompt.size()) out.push_back(next = t.back());
        } else {
            rows.push_back({0U, next});
            out.push_back(next = fwd->step(rows)[0]);
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        max_ms = std::max(max_ms, ms);
        sum_ms += ms;
        ++steps;
    }
};

std::uint64_t state_digest(const strix::deepseek41::AttentionState& s) {
    std::uint64_t d = 14695981039346656037ULL;
    for (const auto& [name, h] : s.hashes()) d = (d ^ h) * 1099511628211ULL;
    return d;
}

// run() as the reference, then each level of --task-levels as a PrefillTask; returns the mismatches
int task_gate(const Args& a, strix::deepseek41::DevicePrefill& pre, const std::vector<std::uint32_t>& prompt, strix::dist::Communicator* comm,
              std::uint32_t rank, strix::deepseek41::BatchForward* fwd, bool print) {
    namespace ds = strix::deepseek41;
    if (comm != nullptr) barrier(*comm, 700U);
    const auto ref = pre.run(prompt);
    const auto ref_state = pre.state_hashes();
    const auto ref_digest = state_digest(pre.export_state());
    const std::vector<std::uint32_t> bg_prompt(prompt.begin(), prompt.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(a.bg_rows, prompt.size())));
    Background bg;
    std::vector<std::uint32_t> bg_ref;
    if (fwd != nullptr && a.interleave != 0U) {
        bg.start(*fwd, bg_prompt, a.interleave);
        while (!bg.done()) bg.step();
        bg_ref = bg.out;
        std::printf("rank %u background alone: %u steps, step wall max %.2f ms mean %.2f ms\n", rank, bg.steps, bg.max_ms,
                    bg.steps ? bg.sum_ms / bg.steps : 0.0);
    }
    std::printf("rank %u task reference run(): %zu rows %.3f s, next token %u, logits %016llx, state digest %016llx\n", rank, prompt.size(),
                ref.seconds, ref.next_token, static_cast<unsigned long long>(fnv(ref.logits)), static_cast<unsigned long long>(ref_digest));
    int bad_all = 0;
    std::stringstream levels(a.task_levels);
    std::uint32_t li = 0;
    for (std::string level; std::getline(levels, level, ','); ++li) {
        ds::PrefillTaskOptions o;
        o.moe_chunk_rows = a.task_moe_chunk;
        o.engram_rows = a.task_engram_rows;
        double budget = a.task_budget;
        if (const auto at = level.find('@'); at != std::string::npos) {  // level[:R]@MS = units_within(MS) per step
            budget = std::stod(level.substr(at + 1U));
            level.resize(at);
        }
        if (const auto colon = level.find(':'); colon != std::string::npos) {  // level:R = this level with MoE chunks of R rows
            o.moe_chunk_rows = static_cast<std::uint32_t>(std::stoul(level.substr(colon + 1U)));
            level.resize(colon);
        }
        const bool whole = level == "whole";
        o.level = level == "layer" ? ds::PrefillUnitLevel::layer : level == "phase" ? ds::PrefillUnitLevel::phase : ds::PrefillUnitLevel::finest;
        if (!whole && level != "layer" && level != "phase" && level != "finest") throw std::invalid_argument("unknown task level " + level);
        if (comm != nullptr) barrier(*comm, 710U + li);
        if (fwd != nullptr && a.interleave != 0U) bg.start(*fwd, bg_prompt, a.interleave);
        const auto t0 = std::chrono::steady_clock::now();
        auto task = pre.begin_run(prompt, o);
        std::uint32_t round = 0, steps = 0;
        while (!task->done()) {
            std::uint32_t k = whole ? std::numeric_limits<std::uint32_t>::max() : a.task_units;
            if (!whole && budget > 0.0) {
                k = task->units_within(budget);
                if (comm != nullptr) k = agree(*comm, k, rank, 1000U + li * 100000U + round);
            }
            ++round;
            (void)task->step(k);
            ++steps;
            if (!bg.done()) bg.step();  // a decode step of another request between two steps of the prefill
        }
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const std::uint32_t bg_during = bg.steps;
        const double bg_during_max = bg.max_ms;
        while (!bg.done()) bg.step();
        const auto r = task->result();
        const auto units = task->units_done();
        const auto timings = task->unit_timings();
        const auto walls = task->step_walls();
        task.reset();
        int bad = 0;
        if (r.next_token != ref.next_token || fnv(r.logits) != fnv(ref.logits)) ++bad;
        if (r.last_row_local && r.stream_hashes != ref.stream_hashes) ++bad;
        if (pre.state_hashes() != ref_state) ++bad;
        const auto dg = state_digest(pre.export_state());
        if (dg != ref_digest) ++bad;
        const bool bg_ok = bg_ref.empty() || bg.out == bg_ref;
        if (!bg_ok) ++bad;
        double dev_sum = 0.0, dev_max = 0.0, wall_max = 0.0;
        const ds::PrefillUnitTiming* top = nullptr;
        std::map<std::string, std::pair<double, double>> kinds;  // kind: sum, max
        std::map<std::int32_t, double> per_layer;
        for (const auto& u : timings) {
            dev_sum += u.device_ms;
            if (u.device_ms > dev_max) {
                dev_max = u.device_ms;
                top = &u;
            }
            auto& kd = kinds[u.kind];
            kd.first += u.device_ms;
            kd.second = std::max(kd.second, u.device_ms);
            per_layer[u.layer] += u.device_ms;
        }
        for (const double w : walls) wall_max = std::max(wall_max, w);
        std::printf("rank %u task %zu rows %s/moe%u@%.0f: %u units in %u steps, %.3f s wall (%.1f tok/s); unit device ms sum %.1f max %.2f (%s, layer %d); "
                    "step() wall max %.2f ms; next token %u logits %016llx digest %016llx: %s%s\n",
                    rank, prompt.size(), level.c_str(), o.moe_chunk_rows, budget, units, steps, wall, static_cast<double>(prompt.size()) / wall, dev_sum, dev_max, top ? top->kind : "-",
                    top ? top->layer : -1, wall_max, r.next_token, static_cast<unsigned long long>(fnv(r.logits)), static_cast<unsigned long long>(dg),
                    bad == 0 ? "bitwise EQUAL to run()" : "DIFFERENT from run()",
                    bg_ref.empty() ? "" : (bg_ok ? ", background tokens EQUAL" : ", background tokens DIFFERENT"));
        if (!bg_ref.empty())
            std::printf("rank %u task %zu rows %s: background %u steps during the task (step wall max %.2f ms), %u in all\n", rank, prompt.size(), level.c_str(), bg_during,
                        bg_during_max, bg.steps);
        if (print) {
            for (const auto& [k, v] : kinds) std::printf("  kind %-10s sum %9.2f ms  max %8.2f ms\n", k.c_str(), v.first, v.second);
            std::string line;
            for (const auto& [l, v] : per_layer) {
                char b[48];
                std::snprintf(b, sizeof(b), " L%d %.1f", l, v);
                line += b;
            }
            std::printf("  per layer (device ms):%s\n", line.c_str());
            if (a.task_table)
                for (const auto& u : timings) std::printf("  unit %4u layer %3d %-10s %8.2f ms\n", u.index, u.layer, u.kind, u.device_ms);
        }
        std::fflush(stdout);
        bad_all += bad;
    }
    return bad_all;
}

// --embed-identity A:B (world 1): the prompt again with rows [A, B) taken from provided embedding rows
// (DevicePrefill::set_embed_rows) instead of the embedding table.
//   identity : the table's own rows, as plain rows -> every bit of the plain run (logits, next token, the last row's
//              stream hashes, the attention state hashes)
//   control  : the rows of the neighbouring tokens -> must differ (the splice is what the layers read)
//   bf16     : rows cut to 8 significant bits, once as binary16 and once as BF16 -> equal to each other
//   image    : finite logits, independent image-router and dead-hash evidence, and an observable image effect
//   text     : the plain prompt after the image run -> every bit of the first plain run
// Exit code 0 only when all of them hold.
int embed_identity_gate(const Args& a, strix::deepseek41::DevicePrefill& pre, const std::vector<std::uint32_t>& prompt,
                        const strix::deepseek41::PrefillResult& ref) {
    namespace ss = strix::session;
    const std::uint32_t A = a.embed_a, B = a.embed_b;
    if (A >= B || B > prompt.size()) throw std::invalid_argument("--embed-identity: need A < B <= rows");
    if (a.layers == 0U || ref.stream_hashes.size() != a.layers)
        throw std::runtime_error("--embed-identity: missing reference stream hashes");
    const auto fnv = [](const auto& v) -> std::uint64_t {  // logits, or the labelled state hashes
        std::uint64_t h = 14695981039346656037ULL;
        const auto mix = [&h](std::uint64_t b, int bytes) {
            for (int k = 0; k < bytes; ++k, b >>= 8U) h = (h ^ (b & 0xffU)) * 1099511628211ULL;
        };
        for (const auto& e : v) {
            if constexpr (requires { e.second; }) {
                mix(e.second, 8);
            } else {
                std::uint32_t b;
                std::memcpy(&b, &e, 4);
                mix(b, 4);
            }
        }
        return h;
    };
    const auto ref_state = pre.state_hashes();
    const std::span<const std::uint32_t> span_tokens(prompt.data() + A, B - A);
    const std::vector<std::uint16_t> rows = pre.embedding_rows(span_tokens);
    const std::uint32_t width = static_cast<std::uint32_t>(rows.size() / (B - A));
    struct Out {
        strix::deepseek41::PrefillResult r;
        std::vector<std::pair<std::string, std::uint64_t>> state;
    };
    const auto run = [&](const char* name, const std::vector<std::uint16_t>* data, ss::EmbedRowFormat format, bool extras) {
        if (data != nullptr) {
            ss::EmbedSplice sp;
            sp.width = width;
            sp.format = format;
            ss::EmbedSegment seg;
            seg.run_row = A;
            seg.rows = B - A;
            seg.data = data->data();
            seg.extras = extras;
            sp.segments.push_back(seg);
            pre.set_embed_rows(std::move(sp));
        }
        Out o;
        o.r = pre.run(prompt);
        if (o.r.stream_hashes.size() != a.layers)
            throw std::runtime_error("--embed-identity: missing arm stream hashes");
        o.state = pre.state_hashes();
        std::printf("embed-identity %-8s next %u logits %016llx state %016llx\n", name, o.r.next_token,
                    static_cast<unsigned long long>(fnv(o.r.logits)), static_cast<unsigned long long>(fnv(o.state)));
        return o;
    };
    const auto same = [](const Out& x, const strix::deepseek41::PrefillResult& r, const std::vector<std::pair<std::string, std::uint64_t>>& st) {
        return x.r.next_token == r.next_token && x.r.logits.size() == r.logits.size() &&
               std::memcmp(x.r.logits.data(), r.logits.data(), r.logits.size() * sizeof(float)) == 0 &&
               x.r.stream_hashes == r.stream_hashes && x.state == st;
    };
    int bad = 0;
    const auto verdict = [&](const char* name, bool ok) {
        std::printf("embed-identity %-8s %s\n", name, ok ? "PASS" : "FAIL");
        if (!ok) ++bad;
    };
    const Out identity = run("identity", &rows, ss::EmbedRowFormat::f16, false);
    verdict("identity", same(identity, ref, ref_state));

    std::vector<std::uint16_t> shifted(rows.size());  // row i <- the row of token i + 1 (cyclic)
    for (std::uint32_t i = 0; i < B - A; ++i)
        std::memcpy(shifted.data() + std::size_t{i} * width, rows.data() + std::size_t{(i + 1U) % (B - A)} * width, std::size_t{width} * 2U);
    const bool movable = shifted != rows;  // an unchanged control cannot establish substitution
    const Out control = run("control", &shifted, ss::EmbedRowFormat::f16, false);
    verdict("control", movable && control.r.logits.size() == ref.logits.size() &&
                       std::memcmp(control.r.logits.data(), ref.logits.data(), ref.logits.size() * sizeof(float)) != 0);
    if (!movable) std::printf("embed-identity control inconclusive: choose an interval with distinct embedding rows\n");

    std::vector<std::uint16_t> cut(rows.size()), cut_bf16(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        cut[i] = static_cast<std::uint16_t>(rows[i] & 0xFFF8U);  // 1 + 7 significant bits: exact in both formats
        const std::uint32_t sign = (cut[i] & 0x8000U) << 16U, e = (cut[i] >> 10U) & 31U;
        std::uint32_t mant = cut[i] & 0x3FFU, bits = 0;
        if (e == 31U) throw std::runtime_error("--embed-identity: the embedding table holds a non-finite value");
        if (e != 0U) {
            bits = ((e + 112U) << 23U) | (mant << 13U);
        } else if (mant != 0U) {
            std::uint32_t ex = 113U;
            while ((mant & 0x400U) == 0U) {
                mant <<= 1U;
                --ex;
            }
            bits = (ex << 23U) | ((mant & 0x3FFU) << 13U);
        }
        cut_bf16[i] = static_cast<std::uint16_t>((sign | bits) >> 16U);
    }
    const Out as_f16 = run("cut-f16", &cut, ss::EmbedRowFormat::f16, false);
    const Out as_bf16 = run("cut-bf16", &cut_bf16, ss::EmbedRowFormat::bf16, false);
    verdict("bf16", same(as_bf16, as_f16.r, as_f16.state));

    const Out image = run("image", &rows, ss::EmbedRowFormat::f16, true);
    bool finite = !image.r.logits.empty();
    for (const float v : image.r.logits) finite = finite && std::isfinite(v);
    verdict("image-router", image.r.embed_image_router_checked);
    verdict("image-dead", image.r.embed_dead_hash_checked);
    verdict("image", finite && !same(image, ref, ref_state));
    std::printf("embed-identity image    logits %s the plain run\n",
                fnv(image.r.logits) == fnv(ref.logits) ? "equal to" : "differ from");

    // Keep the same allocation: long image -> short text -> long text must restore the original result.
    if (prompt.size() > 1U) (void)pre.run(std::span<const std::uint32_t>(prompt.data(), prompt.size() / 2U));
    const Out text = run("text", nullptr, ss::EmbedRowFormat::bf16, false);
    verdict("text", same(text, ref, ref_state));
    (void)run("image-again", &rows, ss::EmbedRowFormat::f16, true);
    if (prompt.size() > 1U) {
        ss::EmbedSplice short_splice;
        short_splice.width = width;
        short_splice.format = ss::EmbedRowFormat::f16;
        short_splice.segments.push_back({0, 1, rows.data(), false});
        pre.set_embed_rows(std::move(short_splice));
        (void)pre.run(std::span<const std::uint32_t>(prompt.data(), prompt.size() / 2U));
    }
    verdict("text-after-short-splice", same(run("text", nullptr, ss::EmbedRowFormat::bf16, false), ref, ref_state));
    std::printf("embed-identity rows [%u, %u) of %zu, width %u, staging %.2f MiB: %s\n", A, B, prompt.size(), width,
                static_cast<double>(strix::deepseek41::DevicePrefill::embed_staging_bytes(B - A, width)) / 1048576.0,
                bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}

int run_world1(const Args& a, const std::vector<std::uint32_t>& prompt) {
    auto o = options(a, prompt.size());
    if (a.ttft) {
        strix::deepseek41::AttentionState state;
        std::uint32_t next = 0;
        {
            strix::deepseek41::DevicePrefill pre(a.model, o);
            ttft_clock.line("prefill weights load");
            std::printf("world 1: layers [0, %u), device weights %.2f GiB loaded in %.1f s\n", a.layers,
                        static_cast<double>(pre.weight_bytes()) / 1073741824.0, pre.load_seconds());
            next = ttft_runs(a, pre, prompt, nullptr, true, a.loads_out).next_token;
            if (a.ttft_decode == 0U) return 0;
            state = pre.export_state();
            ttft_clock.line("export state");
        }
        ttft_clock.line("free prefill");
        ttft_decode(a, prompt, std::move(state), next, nullptr, 0U, true);
        return 0;
    }
    strix::deepseek41::DevicePrefill pre(a.model, o);
    std::printf("world 1: layers [0, %u), device weights %.2f GiB loaded in %.1f s\n", a.layers,
                static_cast<double>(pre.weight_bytes()) / 1073741824.0, pre.load_seconds());
    if (const auto ov = pre.overlay_description(); !ov.empty()) std::printf("weights overlay: %s\n", ov.c_str());
    strix::deepseek41::PrefillResult r;
    for (std::uint32_t k = 0; k < a.repeat; ++k) {
        r = pre.run(prompt);
        print_result("prefill", r, prompt.size(), a.timing, a.loads_out);
    }
    write_receipts(a, pre, r, 0U);
    if (a.embed_identity) return embed_identity_gate(a, pre, prompt, r);
    if (!a.task_levels.empty()) {
        auto lengths = a.task_rows;
        if (lengths.empty()) lengths.push_back(static_cast<std::uint32_t>(prompt.size()));
        int bad = 0;
        for (const auto n : lengths)
            bad += task_gate(a, pre, std::vector<std::uint32_t>(prompt.begin(), prompt.begin() + std::min<std::size_t>(n, prompt.size())), nullptr, 0U,
                             nullptr, true);
        return bad == 0 ? 0 : 1;
    }
    if (!a.reference) return 0;
    const auto ph = pre.state_hashes();
    strix::deepseek41::DeviceForwardOptions fo;
    fo.layers = a.layers;
    fo.max_positions = static_cast<std::uint32_t>(prompt.size() + 8U);
    fo.op_hashes = true;  // the per-layer stream hashes of the last step
    strix::deepseek41::DeviceForward fwd(a.model, fo);
    std::vector<float> logits;
    for (const auto t : prompt) logits = fwd.step(t);
    const auto fh = fwd.state_hashes();
    const auto& sh = fwd.last_layer_stream_hashes();
    int bad = 0;
    if (fnv(logits) != fnv(r.logits) || fwd.next_token() != r.next_token) {
        std::printf("MISMATCH last-row logits: decode %016llx token %u, prefill %016llx token %u\n", static_cast<unsigned long long>(fnv(logits)),
                    fwd.next_token(), static_cast<unsigned long long>(fnv(r.logits)), r.next_token);
        ++bad;
    }
    for (std::size_t l = 0; l < sh.size() && l < r.stream_hashes.size(); ++l)
        if (sh[l] != r.stream_hashes[l]) {
            std::printf("MISMATCH stream after layer %zu\n", l);
            ++bad;
        }
    if (fh.size() != ph.size()) {
        std::printf("MISMATCH state labels (%zu vs %zu)\n", fh.size(), ph.size());
        ++bad;
    }
    for (std::size_t i = 0; i < fh.size() && i < ph.size(); ++i)
        if (fh[i] != ph[i]) {
            std::printf("MISMATCH state %s (decode %016llx, prefill %s %016llx)\n", fh[i].first.c_str(), static_cast<unsigned long long>(fh[i].second),
                        ph[i].first.c_str(), static_cast<unsigned long long>(ph[i].second));
            ++bad;
        }
    std::printf("reference (decode forward, %zu steps): %s; %zu state hashes, %zu layer streams compared\n", prompt.size(),
                bad ? "DIFFERENT" : "bitwise EQUAL", fh.size(), sh.size());
    return bad ? 1 : 0;
}

std::atomic<bool> task_failed{false};

void run_rank(const Args& a, const std::vector<std::uint32_t>& prompt, std::uint32_t rank, std::unique_ptr<strix::dist::Transport> t,
              bool print, strix::deepseek41::PrefillResult* out, std::vector<std::pair<std::string, std::uint64_t>>* out_state = nullptr,
              std::unique_ptr<strix::dist::DeviceExchange> direct = nullptr) {
    strix::dist::Communicator comm(strix::dist::Group{rank, 2U, 1U, 0x64733450ULL}, std::move(t), std::move(direct), std::size_t{22} << 20U);
    strix::profile::check_peer(strix::deepseek41::pair_defaults::profile_resolution(), [&](std::span<const std::byte> profile_out, std::span<std::byte> profile_in) {
        comm.exchange_host(profile_out, profile_in, strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0x70726f66U, 0U});
    });  // both ranks must run the same product profile
    ttft_clock.line("communicator");
    auto o = options(a, prompt.size());
    o.rank = rank;
    o.world = 2;
    o.communicator = &comm;
    const bool decode_reference = a.reference && (a.transport != "emulate" || a.decode_reference);
    o.stream_hashes = decode_reference || out_state != nullptr;
    strix::deepseek41::PrefillResult r;
    std::vector<std::pair<std::string, std::uint64_t>> state;
    if (a.ttft && !decode_reference && out_state == nullptr) {
        strix::deepseek41::AttentionState st;
        {
            strix::deepseek41::DevicePrefill pre(a.model, o);
            ttft_clock.line("prefill weights load");
            if (print)
                std::printf("rank %u of 2 (%s): layers [0, %u), device weights %.2f GiB loaded in %.1f s\n", rank, comm.transport_name().c_str(),
                            a.layers, static_cast<double>(pre.weight_bytes()) / 1073741824.0, pre.load_seconds());
            if (const auto ov = pre.overlay_description(); print && !ov.empty()) std::printf("weights overlay: %s\n", ov.c_str());
            r = ttft_runs(a, pre, prompt, &comm, print, rank == 0U ? a.loads_out : std::string{});
            if (a.ttft_decode != 0U) {
                st = pre.export_state();
                ttft_clock.line("export state");
            }
        }
        if (out) *out = r;
        if (a.ttft_decode == 0U) return;
        ttft_clock.line("free prefill");
        ttft_decode(a, prompt, std::move(st), r.next_token, &comm, rank, print);
        return;
    }
    std::unique_ptr<strix::deepseek41::BatchForward> bfwd;  // --interleave: the decode of another request, weights shared as in the serve
    if (!a.task_levels.empty()) mem_line(rank, "start");
    if (!a.task_levels.empty() && a.interleave != 0U) {
        // as the serve: the decode forward keeps the expert-parallel expert tensors (expert TP is the P16 decode default),
        // so the prefill reads them in place; otherwise each rank would hold the experts twice (TP parts + EP half)
        strix::ds41::expert_tp::set_decode_default(false);
        if (strix::ds41::expert_tp::enabled_from_env())
            throw std::invalid_argument("--interleave: STRIX_DS41_EXPERT_TP=1 would hold the experts twice (the prefill reads the EP tensors in place)");
        if (a.transport != "emulate") mem_guard(a, 2U, "batch forward + prefill");
        strix::deepseek41::BatchForwardOptions bo;
        bo.layers = a.layers;
        bo.max_positions = a.bg_rows + a.interleave + 16U;
        bo.slots = 2U;
        bo.rank = rank;
        bo.world = 2U;
        bo.communicator = &comm;
        bo.tp_attention = true;
        bo.engram_direct_io = true;
        bfwd = std::make_unique<strix::deepseek41::BatchForward>(a.model, bo);
        o.shared_weights = bfwd->file_form_weights();
        mem_line(rank, "batch forward loaded");
    }
    {
        strix::deepseek41::DevicePrefill pre(a.model, o);
        if (print)
            std::printf("rank %u of 2 (%s): layers [0, %u), device weights %.2f GiB loaded in %.1f s (%.2f GiB read in place)\n", rank,
                        comm.transport_name().c_str(), a.layers, static_cast<double>(pre.weight_bytes()) / 1073741824.0, pre.load_seconds(),
                        static_cast<double>(pre.shared_weight_bytes()) / 1073741824.0);
        if (!a.task_levels.empty()) mem_line(rank, "prefill loaded");
        if (const auto ov = pre.overlay_description(); print && !ov.empty()) std::printf("weights overlay: %s\n", ov.c_str());
        for (std::uint32_t k = 0; k < a.repeat; ++k) {
            barrier(comm, k);  // the run's time does not include the peer still loading its weights
            const auto before = comm.stats();
            if (a.select_ab) {
                const bool on = a.select_pattern.empty() ? k % 2U == 1U : a.select_pattern[k % a.select_pattern.size()] == 'S';
                pre.set_dense_select(on);
                if (print) std::printf("arm %s (run %u)\n", on ? "SEL" : "OFF", k);
            }
            r = pre.run(prompt);
            if (print) {
                print_result(rank == 0U ? "rank 0 prefill" : "rank 1 prefill", r, prompt.size(), a.timing, rank == 0U ? a.loads_out : std::string{});
                const auto& s = comm.stats();
                std::printf("  exchanges %llu, %.1f MB sent, %.2f s in exchange (stage %.2f, wire %.2f, check %.2f)\n",
                            static_cast<unsigned long long>(s.exchanges - before.exchanges),
                            static_cast<double>(s.bytes_sent - before.bytes_sent) / 1e6, s.seconds - before.seconds, s.stage - before.stage,
                            s.wire - before.wire, s.check - before.check);
                std::fflush(stdout);
            }
        }
        write_receipts(a, pre, r, rank);
        if (a.state_timing && print) {  // the handoff to a decode forward: the attention state to the host
            const auto t0 = std::chrono::steady_clock::now();
            const auto exported = pre.export_state();
            std::size_t bytes = 0;
            for (const auto& l : exported.layers) bytes += l.ring.size() + l.kv.size() + l.idx_keys.size() + l.carry.size() * 4U + l.topk.size() * 4U;
            const double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
            std::uint64_t digest = 14695981039346656037ULL;  // FNV of the state's labelled hashes
            for (const auto& [name, h] : exported.hashes()) digest = (digest ^ h) * 1099511628211ULL;
            std::printf("  export_state: %.2f ms, %.2f MB, state digest %016llx\n", ms, static_cast<double>(bytes) / 1e6,
                        static_cast<unsigned long long>(digest));
        }
        if (a.extend_check != 0U) {
            // the prompt again as run() of its first piece + extend() of the rest: last-row logits and every state hash
            // must equal the last whole run's (with --select-ab / --select-pattern: the selector on)
            const auto whole_hashes = pre.state_hashes();
            const auto whole_logits = fnv(r.logits);
            if (a.select_ab) pre.set_dense_select(true);
            barrier(comm, a.repeat);
            strix::deepseek41::PrefillResult e;
            for (std::size_t p0 = 0; p0 < prompt.size(); p0 += a.extend_check) {
                const std::span<const std::uint32_t> part(prompt.data() + p0, std::min<std::size_t>(a.extend_check, prompt.size() - p0));
                e = p0 == 0U ? pre.run(part) : pre.extend(part);
            }
            const bool same = fnv(e.logits) == whole_logits && pre.state_hashes() == whole_hashes;
            if (print)
                std::printf("extend check (%u-row pieces): %s; logits %016llx vs whole %016llx\n", a.extend_check, same ? "bitwise EQUAL" : "MISMATCH",
                            static_cast<unsigned long long>(fnv(e.logits)), static_cast<unsigned long long>(whole_logits));
            if (!same) task_failed = true;
        }
        if (decode_reference || out_state != nullptr) state = pre.state_hashes();
        if (!a.task_levels.empty()) {
            auto lengths = a.task_rows;
            if (lengths.empty()) lengths.push_back(static_cast<std::uint32_t>(prompt.size()));
            for (const auto n : lengths) {
                const std::vector<std::uint32_t> part(prompt.begin(), prompt.begin() + std::min<std::size_t>(n, prompt.size()));
                if (task_gate(a, pre, part, &comm, rank, bfwd.get(), print) != 0) task_failed = true;
                mem_line(rank, "after task gate");
            }
        }
    }
    if (out) *out = r;
    if (out_state) *out_state = state;
    if (!decode_reference) return;
    // The two-rank decode forward fed the same tokens over the same communicator (the prefill's weights are freed).
    strix::deepseek41::DeviceForwardOptions fo;
    fo.layers = a.layers;
    fo.max_positions = static_cast<std::uint32_t>(prompt.size() + 8U);
    fo.rank = rank;
    fo.world = 2;
    fo.communicator = &comm;
    fo.op_hashes = true;
    strix::deepseek41::DeviceForward fwd(a.model, fo);
    std::vector<float> logits;
    for (const auto token : prompt) logits = fwd.step(token);
    const auto fh = fwd.state_hashes();
    const auto& sh = fwd.last_layer_stream_hashes();
    int bad = 0;
    if (fnv(logits) != fnv(r.logits) || fwd.next_token() != r.next_token) {
        std::printf("rank %u MISMATCH last-row logits: decode %016llx token %u, prefill %016llx token %u\n", rank,
                    static_cast<unsigned long long>(fnv(logits)), fwd.next_token(), static_cast<unsigned long long>(fnv(r.logits)), r.next_token);
        ++bad;
    }
    // row split: the last row's stream and top-k exist only on the rank that computed the row
    for (std::size_t l = 0; r.last_row_local && l < sh.size() && l < r.stream_hashes.size(); ++l)
        if (sh[l] != r.stream_hashes[l]) {
            if (bad < 20) std::printf("rank %u MISMATCH stream after layer %zu\n", rank, l);
            ++bad;
        }
    for (std::size_t i = 0; i < fh.size() && i < state.size(); ++i)
        if (fh[i] != state[i] && (r.last_row_local || fh[i].first.rfind("topk", 0) != 0)) {
            if (bad < 20) std::printf("rank %u MISMATCH state %s\n", rank, fh[i].first.c_str());
            ++bad;
        }
    std::printf("rank %u reference (two-rank decode forward, %zu steps): %s; %zu state hashes, %zu layer streams; logits %016llx token %u\n",
                rank, prompt.size(), bad == 0 && fh.size() == state.size() ? "bitwise EQUAL" : "DIFFERENT", fh.size(), r.last_row_local ? sh.size() : std::size_t{0},
                static_cast<unsigned long long>(fnv(r.logits)), r.next_token);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) try {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    const strix::deepseek41::pair_defaults::Effective production_defaults(strix::deepseek41::pair_defaults::App::prefill, argc, argv, strix::deepseek41::pair_defaults::this_build());  // before any option or env read
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
            return std::string(argv[++i]);
        };
        if (s == "--prompt") a.prompt_path = next();
        else if (s == "--rows") a.rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--layers") a.layers = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--world") a.world = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--rank") { a.rank = static_cast<std::uint32_t>(std::stoul(next())); a.world = 2; }
        else if (s == "--transport") { a.transport = next(); a.world = 2; }
        else if (s == "--chunk") a.chunk = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--window") a.window = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--lanes") a.lanes = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--moe-chunk") a.moe_chunk = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--no-tp") a.tp = false;
        else if (s == "--row-split") a.row_split = true;
        else if (s == "--no-row-split") a.row_split = false;  // tensor-parallel attention (every row on both ranks)
        else if (s == "--stream-experts") a.stream_experts = true;
        else if (s == "--repeat") a.repeat = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--timing") a.timing = true;
        else if (s == "--reference") a.reference = true;
        else if (s == "--embed-identity") {
            const std::string v = next();
            const auto colon = v.find(':');
            if (colon == std::string::npos) throw std::invalid_argument("--embed-identity A:B");
            a.embed_identity = true;
            a.embed_a = static_cast<std::uint32_t>(std::stoul(v.substr(0, colon)));
            a.embed_b = static_cast<std::uint32_t>(std::stoul(v.substr(colon + 1)));
        }
        else if (s == "--select-ab") a.select_ab = true;
        else if (s == "--extend-check") a.extend_check = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--select-pattern") a.select_ab = true, a.select_pattern = next();
        else if (s == "--decode-reference") a.reference = a.decode_reference = true;  // emulate: also the two-rank decode
        else if (s == "--no-cone") a.cone = false;
        else if (s == "--state-timing") a.state_timing = true;
        else if (s == "--draft-rows") a.draft_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--seed-out") a.seed_out = next();
        else if (s == "--warmup") a.warmup = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--split-min-rows") a.split_min_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--joint-max-rows") a.joint_max_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--loads") a.loads = true;
        else if (s == "--stream-hashes-out") a.hashes_out = next();
        else if (s == "--loads-out") {
            a.loads = true;
            a.loads_out = next();
        }
        else if (s == "--connect-timeout") a.connect_timeout = static_cast<unsigned>(std::stoul(next()));
        else if (s == "--ttft-timing") a.ttft = true;
        else if (s == "--task-levels") a.task_levels = next();
        else if (s == "--task-rows") {
            std::stringstream ss(next());
            for (std::string t; std::getline(ss, t, ',');) a.task_rows.push_back(static_cast<std::uint32_t>(std::stoul(t)));
        }
        else if (s == "--task-engram-rows") a.task_engram_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--mem-guard-gib") a.mem_guard_gib = std::stod(next());
        else if (s == "--task-units") a.task_units = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--task-budget") a.task_budget = std::stod(next());
        else if (s == "--task-moe-chunk") a.task_moe_chunk = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--task-table") a.task_table = true;
        else if (s == "--interleave") a.interleave = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--bg-rows") a.bg_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--ttft-warmup") a.ttft_warmup = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--ttft-decode") a.ttft_decode = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a.model.empty()) a.model = s;
        else throw std::invalid_argument("unknown argument " + s);
    }
    if (a.model.empty() || a.prompt_path.empty()) throw std::invalid_argument("usage: strix-deepseek41-prefill MODEL.gguf --prompt FILE ...");
    validate_embed_identity_args(a);
    if (!a.task_levels.empty() && a.mem_guard_gib > 0.0 && std::getenv("STRIX_DS41_PREFILL_MEM_GUARD_GIB") == nullptr)  // the prefill checks its exact bytes too
        ::setenv("STRIX_DS41_PREFILL_MEM_GUARD_GIB", std::to_string(a.mem_guard_gib).c_str(), 1);
    ttft_clock.on = a.ttft;
    ttft_clock.exec_to_main();
    ttft_clock.line("arguments");
    const auto prompt = read_prompt(a.prompt_path, a.rows);
    ttft_clock.line("prompt read");
    if ((a.ttft_warmup != 0U || a.ttft_decode != 0U) && !a.ttft) throw std::invalid_argument("--ttft-warmup and --ttft-decode need --ttft-timing");
    if (a.world == 1U) return run_world1(a, prompt);
    if (a.transport == "emulate") {  // both ranks in this process; rank 0 prints; --reference compares with world 1
        auto [t0, t1] = strix::dist::memory_pair();
        std::exception_ptr failure;
        strix::deepseek41::PrefillResult r0, r1;
        std::vector<std::pair<std::string, std::uint64_t>> s0, s1;
        std::thread peer([&, t = std::move(t1)]() mutable {
            try {
                run_rank(a, prompt, 1U, std::move(t), false, &r1, a.reference ? &s1 : nullptr);
            } catch (...) {
                failure = std::current_exception();
            }
        });
        try {
            run_rank(a, prompt, 0U, std::move(t0), true, &r0, a.reference ? &s0 : nullptr);
        } catch (...) {
            peer.join();
            throw;
        }
        peer.join();
        if (failure) std::rethrow_exception(failure);
        std::printf("ranks: next token %u / %u\n", r0.next_token, r1.next_token);
        if (!a.reference) return task_failed ? 1 : 0;
        auto o = options(a, prompt.size());
        o.stream_hashes = true;
        strix::deepseek41::DevicePrefill one(a.model, o);
        const auto r = one.run(prompt);
        // each rank's attention state and (on the last row's rank) the last row's stream after every layer = world 1
        const auto s = one.state_hashes();
        std::size_t state_bad = 0, stream_bad = 0;
        for (const auto* rank : {&r0, &r1}) {
            const auto& st = rank == &r0 ? s0 : s1;
            if (st.size() != s.size()) ++state_bad;
            for (std::size_t i = 0; i < s.size() && i < st.size(); ++i)
                if (st[i] != s[i] && (rank->last_row_local || s[i].first.rfind("topk", 0) != 0)) {
                    if (state_bad < 8) std::printf("rank %d MISMATCH state %s\n", rank == &r0 ? 0 : 1, s[i].first.c_str());
                    ++state_bad;
                }
            if (rank->last_row_local && rank->stream_hashes != r.stream_hashes) ++stream_bad;
        }
        std::printf("reference (world 1 prefill): attention state of both ranks %s (%zu hashes), last row's streams %s (%zu layers)\n",
                    state_bad == 0 ? "EQUAL" : "DIFFERENT", s.size(), stream_bad == 0 ? "EQUAL" : "DIFFERENT", r.stream_hashes.size());
        // each rank holds its vocabulary slice of the logits: compare slice by slice
        std::vector<float> both(r.logits.size(), 0.0F);
        const std::size_t half = r.logits.size() / 2U;
        std::copy(r0.logits.begin(), r0.logits.begin() + static_cast<std::ptrdiff_t>(half), both.begin());
        std::copy(r1.logits.begin() + static_cast<std::ptrdiff_t>(half), r1.logits.end(), both.begin() + static_cast<std::ptrdiff_t>(half));
        const bool equal = fnv(both) == fnv(r.logits) && r0.next_token == r.next_token && r1.next_token == r.next_token;
        std::printf("reference (world 1 prefill): two ranks %s (token %u, logits %016llx vs %016llx)\n", equal ? "bitwise EQUAL" : "DIFFERENT",
                    r.next_token, static_cast<unsigned long long>(fnv(both)), static_cast<unsigned long long>(fnv(r.logits)));
        return equal && state_bad == 0 && stream_bad == 0 ? 0 : 1;
    }
    ::alarm(a.connect_timeout + 30U);  // a stream on a dead USB4 link blocks open(): the alarm ends the process
    auto opened = strix::dist::open_pair(a.transport, a.rank, {std::chrono::seconds(a.connect_timeout)});
    auto transport = std::move(opened.host);
    const auto& receipt = opened.receipt;
    ::alarm(0);
    ttft_clock.line("transport open (peer handshake)");
    std::printf("connected (%s)%s%s\n", transport->name().c_str(), receipt.empty() ? "" : "\n", receipt.c_str());
    std::fflush(stdout);
    run_rank(a, prompt, a.rank, std::move(transport), true, nullptr, nullptr, std::move(opened.direct));
    return task_failed ? 1 : 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
