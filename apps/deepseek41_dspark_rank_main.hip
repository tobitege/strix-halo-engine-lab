#include "strix/device/hip_gate.hpp"
#include "strix/dist/open_transport.hpp"
// One rank of the two-node deepseek41 decode with DSpark speculative drafting and verify windows, or both ranks in
// one process over the in-memory transport (--transport emulate, trimmed models).
//   strix-deepseek41-dspark-rank MODEL.gguf DRAFT.gguf --rank R --transport usb4:DEV,DEV,... | usb4auto | usb4gpu | tcp-listen:PORT |
//       tcp-connect:HOST:PORT | emulate  --layers N --prompt-ids a,b,c --generate G
//       [--mode greedy | spec | bench] [--draft split | rank0] [--vocab SPEC] [--bench-rows 1,2,4,6,8] [--bench-reps N]
//       [--engram-direct] [--connect-timeout S] [--max-positions P] [--reference a,b,c --oracle-rows 2,4,8] [--gate]
// greedy: target-only, one row per step (the reference tokens). spec: the drafter proposes 5 tokens from the last
// committed token, the window policy (STRIX_DS41_DSPARK_WINDOW, default fixed:5) picks k, the target verifies
// [t, d_1..d_k] as one window of k + 1 rows, keeps the accepted prefix and appends the committed rows' inputs of the
// last three layers to the drafter; the tokens equal greedy's. bench: window costs per row count. oracle: windows of
// the cycled sizes whose drafts are the greedy reference continuation made wrong from a varying row on (every
// acceptance depth, every rollback); the tokens must equal the reference.
// --prompt-file F with greedy or spec: every prompt of F (one comma list per line) in turn on the loaded weights, one
// result block each (a prompt set for stable acceptance numbers).
// Draft placement: split = both ranks draft, the drafter's head split by vocabulary (8-byte candidate exchange per
// Markov step, the rest replicated); rank0 = rank 0 drafts alone and sends the window (k and the drafts) to rank 1.
#include "strix/deepseek41_device_forward.hpp"
#include "strix/hip_check.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/dist/gpu_direct.hpp"
#include "strix/ds41_dspark_driver.hpp"
#include "strix/ds41_dspark_files.hpp"
#include "strix/ds41_dspark_vocab.hpp"
#include "strix/hip_ds41_dspark.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace {

namespace ds = strix::ds41::dspark;
using clock_type = std::chrono::steady_clock;
double since(clock_type::time_point t) { return std::chrono::duration<double>(clock_type::now() - t).count(); }

void check(hipError_t e, const char* what) {
    ::strix::hip_check_prefixed("dspark rank: ", e, what);
}

struct Args {
    std::string model, draft, prompt, transport = "emulate", mode = "spec", placement = "split", vocab = "full", bench_rows = "1,2,4,6,8";
    std::string reference, oracle_rows = "2,4,8";  // oracle: the greedy tokens, and the window sizes cycled
    std::string prompt_file, dump;                 // dump: prompts (one comma list per line) and output prefix
    std::uint32_t fill_rows = 8;                   // dump: prompt rows per window (1 = single steps with top-32)
    std::uint32_t rank = 0, layers = 40, generate = 32, max_positions = 4096, bench_reps = 12;
    unsigned connect_timeout = 60;
    bool engram_direct = false, gate = false, tp_attention = false, stream_experts = false;
};

std::vector<std::uint32_t> parse_list(const std::string& s) {
    std::vector<std::uint32_t> out;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');) out.push_back(static_cast<std::uint32_t>(std::stoul(t)));
    return out;
}

struct Scratch {
    float* main_hidden = nullptr;  // [128][15360]
    std::uint32_t* packet = nullptr;  // [2][16]: own, peer
    Scratch() {
        check(hipMalloc(reinterpret_cast<void**>(&main_hidden), std::size_t{128} * ds::main_width * 4U), "scratch");
        check(hipMalloc(reinterpret_cast<void**>(&packet), 128U), "scratch");
    }
    ~Scratch() {
        (void)hipFree(main_hidden);
        (void)hipFree(packet);
    }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
};

struct Stats {
    std::uint32_t windows = 0, verified = 0, accepted = 0, generated = 0, plain_steps = 0;
    std::array<std::uint32_t, ds::block_size> reached{}, hit{};  // per depth: verified with all earlier correct, and correct
    std::map<std::uint32_t, std::pair<std::uint32_t, double>> verify_by_rows;  // R -> (count, seconds)
    double draft = 0, packet = 0, verify = 0, verify_exchange = 0, append = 0, total = 0;
};

// STRIX_DS41_DSPARK_FORCE=cycle:R1,R2,.. (spec): window i verifies R_(i mod n) rows (1..6; R1 = the drafter still
// runs on every step, its proposals are only recorded). Same on both ranks (window count). Unset = the policy decides.
std::vector<std::uint32_t> forced_windows_from_env() {
    const char* v = std::getenv("STRIX_DS41_DSPARK_FORCE");
    if (v == nullptr || *v == 0) return {};
    const std::string s(v);
    if (s.rfind("cycle:", 0) != 0) throw std::invalid_argument("STRIX_DS41_DSPARK_FORCE=cycle:R,R,..");
    auto rows = parse_list(s.substr(6));
    for (const auto R : rows)
        if (R < 1U || R > ds::block_size + 1U) throw std::invalid_argument("STRIX_DS41_DSPARK_FORCE rows 1..6");
    if (rows.empty()) throw std::invalid_argument("STRIX_DS41_DSPARK_FORCE: no rows");
    return rows;
}

// One comma list of ids per line (empty lines skipped).
std::vector<std::vector<std::uint32_t>> read_prompts(const std::string& path) {
    std::vector<std::vector<std::uint32_t>> prompts;
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) throw std::invalid_argument("cannot read " + path);
    std::string line;
    for (int c; (c = std::fgetc(f)) != EOF;) {
        if (c == '\n') { if (!line.empty()) prompts.push_back(parse_list(line)); line.clear(); }
        else if (c != '\r') line.push_back(static_cast<char>(c));
    }
    if (!line.empty()) prompts.push_back(parse_list(line));
    std::fclose(f);
    if (prompts.empty()) throw std::invalid_argument(path + ": no prompts");
    return prompts;
}

// Data for the drafter's offline replay and distillation: for every prompt of --prompt-file (one comma list per line),
// greedy decoding of --generate tokens, writing <dump>-<i>.tokens (prompt length, then every token: prompt and
// generated), <dump>-<i>.hidden (f32 [steps][15360]: the drafter's input row of each fed token, bitwise the
// runtime's) and <dump>-<i>.top (per step the target's 32 largest logits: u32 id, f32 value, descending). The state
// is reset between prompts.
void run_dump(const Args& a, strix::deepseek41::DeviceForward& fwd, bool write) {
    auto* st = static_cast<hipStream_t>(fwd.stream());
    const auto prompts = read_prompts(a.prompt_file);
    float *row = nullptr, *rows_dev = nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&row), std::size_t{ds::main_width} * 4U), "dump row");
    check(hipMalloc(reinterpret_cast<void**>(&rows_dev), std::size_t{ds::main_width} * 8U * 4U), "dump rows");
    std::vector<float> host(ds::main_width);
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        const auto t0 = clock_type::now();
        if (i > 0U) fwd.reset();
        const auto& prompt = prompts[i];
        std::FILE *fh = nullptr, *ft = nullptr;
        if (write) {
            fh = std::fopen((a.dump + "-" + std::to_string(i) + ".hidden").c_str(), "wb");
            ft = std::fopen((a.dump + "-" + std::to_string(i) + ".top").c_str(), "wb");
            if (fh == nullptr || ft == nullptr) throw std::runtime_error("cannot write the dump");
        }
        std::vector<std::uint32_t> tokens(prompt);
        std::uint32_t next = 0;
        const std::size_t steps = prompt.size() + a.generate - 1U;
        std::size_t s0 = 0;
        if (a.fill_rows > 1U && prompt.size() > 2U) {
            // prompt rows 1 .. P-2 in windows of fill_rows (row 0 alone: a window may not start before the first
            // compressed entry; the last prompt row as a single step for its logits). Their top-32 are not kept
            // (id 0xFFFFFFFF): only generated positions carry the target's distribution.
            const auto put_rows = [&](std::uint32_t rows) {
                if (!write) return;
                const auto caps = fwd.captured_streams();
                ds::enqueue_main_hidden(caps[0], caps[1], caps[2], rows, rows_dev, st);
                std::vector<float> h(std::size_t{rows} * ds::main_width);
                check(hipMemcpyAsync(h.data(), rows_dev, h.size() * 4U, hipMemcpyDeviceToHost, st), "dump rows");
                check(hipStreamSynchronize(st), "dump rows");
                std::fwrite(h.data(), 4, h.size(), fh);
                const std::uint32_t none = 0xFFFFFFFFU;
                const float zero = 0.0F;
                for (std::uint32_t r = 0; r < rows * 32U; ++r) {
                    std::fwrite(&none, 4, 1, ft);
                    std::fwrite(&zero, 4, 1, ft);
                }
            };
            (void)fwd.step(prompt[0]);
            put_rows(1U);
            std::size_t at = 1;
            while (at + 1U < prompt.size()) {
                const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(a.fill_rows, prompt.size() - 1U - at));
                (void)fwd.step_window(std::span<const std::uint32_t>(prompt).subspan(at, n));
                put_rows(n);
                at += n;
            }
            s0 = prompt.size() - 1U;
        }
        for (std::size_t s = s0; s < steps; ++s) {
            const std::uint32_t fed = s < prompt.size() ? prompt[s] : next;
            const auto& logits = fwd.step(fed);
            next = fwd.next_token();
            if (s + 1U >= prompt.size()) tokens.push_back(next);
            if (!write) continue;
            const auto caps = fwd.captured_streams();
            ds::enqueue_main_hidden(caps[0], caps[1], caps[2], 1U, row, st);
            check(hipMemcpyAsync(host.data(), row, host.size() * 4U, hipMemcpyDeviceToHost, st), "dump row");
            check(hipStreamSynchronize(st), "dump row");
            std::fwrite(host.data(), 4, host.size(), fh);
            if (logits.size() < 129280U) throw std::logic_error("dump needs the full logits");
            std::vector<std::uint32_t> idx(logits.size());
            for (std::uint32_t v = 0; v < idx.size(); ++v) idx[v] = v;
            std::partial_sort(idx.begin(), idx.begin() + 32, idx.end(), [&](std::uint32_t x, std::uint32_t y) {
                return logits[x] > logits[y] || (logits[x] == logits[y] && x < y);
            });
            for (std::uint32_t k = 0; k < 32U; ++k) {
                std::fwrite(&idx[k], 4, 1, ft);
                std::fwrite(&logits[idx[k]], 4, 1, ft);
            }
        }
        if (write) {
            std::fclose(fh);
            std::fclose(ft);
            std::FILE* fk = std::fopen((a.dump + "-" + std::to_string(i) + ".tokens").c_str(), "w");
            std::fprintf(fk, "%zu\n", prompt.size());
            for (std::size_t k = 0; k < tokens.size(); ++k) std::fprintf(fk, "%s%u", k ? "," : "", tokens[k]);
            std::fprintf(fk, "\n");
            std::fclose(fk);
            std::printf("dump %zu: prompt %zu, %zu steps in %.1f s\n", i, prompt.size(), steps, since(t0));
            std::fflush(stdout);
        }
    }
    (void)hipFree(row);
    (void)hipFree(rows_dev);
}

void run_rank(const Args& a, std::uint32_t rank, std::unique_ptr<strix::dist::Transport> transport, bool print,
              std::unique_ptr<strix::dist::DeviceExchange> direct = nullptr) {
    const auto t_start = clock_type::now();
    strix::dist::Communicator comm(strix::dist::Group{rank, 2U, 1U, 0x64733432ULL}, std::move(transport), std::move(direct), 512U << 10U);
    const bool spec = a.mode == "spec", oracle = a.mode == "oracle";
    const auto reference = a.reference.empty() ? std::vector<std::uint32_t>{} : parse_list(a.reference);
    const auto oracle_rows = parse_list(a.oracle_rows);
    if (oracle && (reference.size() < 2U || oracle_rows.empty())) throw std::invalid_argument("--mode oracle needs --reference");
    for (const auto R : oracle_rows)
        if (R < 1U || R > strix::deepseek41::DeviceForward::max_window_rows) throw std::invalid_argument("--oracle-rows 1..8");
    const bool split = a.placement == "split";
    const bool drafting = spec && (split || rank == 0U);
    strix::deepseek41::DeviceForwardOptions o;
    o.layers = a.layers;
    o.max_positions = a.max_positions;
    o.rank = rank;
    o.world = 2;
    o.communicator = &comm;
    o.op_hashes = false;
    o.gather_logits = a.mode == "dump";
    o.engram_direct_io = a.engram_direct;
    o.capture_main_hidden = drafting || a.mode == "dump";
    o.tp_attention = a.tp_attention;
    strix::deepseek41::DeviceForward fwd(a.model, o);
    auto* st = static_cast<hipStream_t>(fwd.stream());
    std::unique_ptr<ds::LoadedWeights> draft_weights;
    std::unique_ptr<ds::DeviceDrafter> drafter;
    std::uint64_t draft_request = 0;
    // --prompt-file with greedy / spec: every line in turn on the same weights (decode state, drafter and window
    // policy fresh per prompt), one result block per prompt
    const auto set = !a.prompt_file.empty() && (a.mode == "greedy" || spec) ? read_prompts(a.prompt_file)
                                                                          : std::vector<std::vector<std::uint32_t>>{parse_list(a.prompt)};
    const auto choice = ds::choose_vocabulary(a.vocab, set[0]);
    if (drafting) {
        draft_weights = ds::load_weights(a.model, a.draft);
        drafter = std::make_unique<ds::DeviceDrafter>(draft_weights->weights);
        drafter->set_vocabulary(choice.vocabulary);
        draft_weights.reset();  // the drafter keeps device copies; the host copies would only take memory
        if (split)
            drafter->set_rank(rank, 2U, [&](void* stream, const void* local, void* remote, std::size_t bytes) {
                comm.exchange(static_cast<hipStream_t>(stream), local, remote, bytes,
                              strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 1000U, draft_request++});
            });
    }
    if (print)
        std::printf("rank %u of 2 (%s): layers [0, %u), device weights %.2f GiB loaded in %.1f s; mode %s, draft %s, vocabulary %s\n", rank,
                    comm.transport_name().c_str(), a.layers, static_cast<double>(fwd.weight_bytes()) / 1073741824.0, fwd.load_seconds(),
                    a.mode.c_str(), spec ? a.placement.c_str() : "-", spec ? choice.label.c_str() : "-");
    if (a.mode == "dump") {
        run_dump(a, fwd, rank == 0U && print);
        return;
    }
    Scratch scratch;
    // STRIX_DS41_DSPARK_TRACE=FILE (spec, rank 0): one JSON line per window (input position, the 5 proposals and
    // confidences, verified rows, the target's tokens for them, accepted, draft / verify / append ms), and one line per
    // prompt with the prompt and the generated tokens. Output tokens are unchanged.
    const auto forced = spec ? forced_windows_from_env() : std::vector<std::uint32_t>{};
    std::FILE* trace = nullptr;
    if (const char* tv = std::getenv("STRIX_DS41_DSPARK_TRACE"); spec && rank == 0U && tv != nullptr && *tv != 0) {
        trace = std::fopen(tv, "a");
        if (trace == nullptr) throw std::runtime_error(std::string("cannot write ") + tv);
    }
    for (std::size_t si = 0; si < set.size(); ++si) {
        const auto& prompt = set[si];
        if (si > 0U) {
            fwd.reset();
            if (drafter) {
                drafter->reset();
                drafter->set_vocabulary(ds::choose_vocabulary(a.vocab, prompt).vocabulary);
            }
        }
        if (print && set.size() > 1U) std::printf("==== prompt %zu of %zu (%zu tokens)\n", si, set.size(), prompt.size());
        const auto policy = ds::window_policy_from_env();

        // ---- prompt: one row per step; the drafter gets every row's input ----
        const auto t_prompt = clock_type::now();
        std::uint32_t t = 0, staged = 0;
        const auto flush = [&] {
            if (staged == 0U) return;
            drafter->append_hidden_device(scratch.main_hidden, staged, st);
            staged = 0;
        };
        double prompt_step_sum = 0.0;
        std::uint32_t prompt_step_n = 0;
        for (std::size_t pi = 0; pi < prompt.size(); ++pi) {
            const auto ts = clock_type::now();
            (void)fwd.step(prompt[pi]);
            if (pi >= 4U) prompt_step_sum += since(ts), ++prompt_step_n;  // the first steps pay one-time costs
            t = fwd.next_token();
            if (drafting) {
                const auto caps = fwd.captured_streams();
                ds::enqueue_main_hidden(caps[0], caps[1], caps[2], 1U, scratch.main_hidden + std::size_t{staged} * ds::main_width, st);
                if (++staged == 128U) flush();
            }
        }
        if (drafting) {
            flush();
            drafter->add_vocabulary(prompt);
            const std::uint32_t first[1] = {t};
            drafter->add_vocabulary(first);
        }
        const double prompt_seconds = since(t_prompt);
        std::vector<std::uint32_t> out{t};
        Stats S;
        const auto t_gen = clock_type::now();

        if (a.mode == "greedy") {
            while (out.size() < a.generate) {
                const auto t0 = clock_type::now();
                const std::uint32_t one[1] = {t};
                t = fwd.step_window(one)[0];
                S.verify += since(t0);
                ++S.verify_by_rows[1].first;
                S.verify_by_rows[1].second += fwd.last_step_seconds();
                out.push_back(t);
            }
        } else if (a.mode == "bench") {
            // Window cost per row count: rows [t, t, ...] then keep one row (the state stays a valid prefix of t's).
            for (const auto R : parse_list(a.bench_rows)) {
                std::vector<std::uint32_t> w(R, t);
                double sum = 0, exch = 0;
                for (std::uint32_t i = 0; i < a.bench_reps + 1U; ++i) {
                    const double c0 = comm.stats().seconds;
                    const auto t0 = clock_type::now();
                    (void)fwd.step_window(w);
                    const double s = since(t0);
                    fwd.commit_window(1);
                    if (i > 0U) { sum += s; exch += comm.stats().seconds - c0; }  // the first pays one-time costs
                    t = fwd.last_tokens()[0];
                    std::fill(w.begin(), w.end(), t);
                }
                if (print)
                    std::printf("bench R%u: %.2f ms per window (exchange %.2f ms), %.2f ms per row\n", R, sum * 1e3 / a.bench_reps,
                                exch * 1e3 / a.bench_reps, sum * 1e3 / a.bench_reps / R);
            }
        } else {
            // Online window costs and the drafting gate, decided on rank 0 and sent to rank 1 after every window (32 bytes),
            // so both ranks choose the same windows: cost[k] = draft + verify of k + 1 rows (EMAs of rank 0's times, a
            // linear guess until a size is seen); the gate drafts while the measured rate of drafted windows (tokens per
            // second, draft and append included) beats single steps, and tries the other side every 16th decision.
            const auto ctl = spec ? policy.controller : nullptr;  // STRIX_DS41_DSPARK_WINDOW=controller
            const bool gated = spec && (a.gate || ctl);
            // controller costs, accumulated on rank 0 (sums and counts) and shared as means through the control packet
            std::array<double, ds::block_size + 2U> cost_sum{};
            std::array<std::uint32_t, ds::block_size + 2U> cost_n{};
            double draft_sum = 0.0, extra_sum = 0.0;
            std::uint32_t draft_n = 0;
            if (prompt_step_n != 0U) cost_sum[1] = prompt_step_sum, cost_n[1] = prompt_step_n;
            std::array<double, ds::block_size + 2U> verify_ema{};
            double draft_ema = 0.0, rate_spec = 0.0, rate_plain = 0.0;
            std::uint32_t since_switch = 0;
            bool use_spec = true;
            const auto ema = [](double& e, double v) { e = e == 0.0 ? v : 0.875 * e + 0.125 * v; };
            const auto push_costs = [&] {
                if (!policy.survival) return;
                const double v1 = verify_ema[1] != 0.0 ? verify_ema[1] : 0.075;
                for (std::uint32_t k = 0; k <= ds::block_size; ++k) {
                    const double v = verify_ema[k + 1U] != 0.0 ? verify_ema[k + 1U] : v1 * (1.0 + 0.3 * k);
                    policy.survival->set_cost(k, static_cast<float>((draft_ema + v) * 1e3));
                }
            };
            const auto control = [&](std::uint32_t seq) {  // rank 0's gate and costs to both ranks
                if (ctl) {
                    // [0..5] verify of 1..6 rows, [6] draft, [7] append + control, [8] rank 0's next decision (draft or not)
                    const auto t0 = clock_type::now();
                    float packet[16] = {};
                    for (std::uint32_t r = 1; r <= ds::block_size + 1U; ++r)
                        packet[r - 1U] = cost_n[r] != 0U ? static_cast<float>(cost_sum[r] / cost_n[r]) : 0.0F;
                    packet[6] = draft_n != 0U ? static_cast<float>(draft_sum / draft_n) : 0.0F;
                    packet[7] = draft_n != 0U ? static_cast<float>(extra_sum / draft_n) : 0.0F;
                    const auto apply = [&](const float* q) {
                        auto c = ctl->costs();  // sample counts are kept locally (both ranks run the same windows)
                        for (std::uint32_t r = 1; r <= ds::block_size + 1U; ++r) c.verify[r] = q[r - 1U];
                        c.draft = q[6];
                        c.extra = q[7];
                        ctl->set_costs(c);
                    };
                    if (rank == 0U) {
                        apply(packet);
                        packet[8] = ctl->draft_next() ? 1.0F : 0.0F;
                    }
                    check(hipMemcpyAsync(scratch.packet, packet, 64U, hipMemcpyHostToDevice, st), "control");
                    comm.exchange(st, scratch.packet, scratch.packet + 16, 64U,
                                  strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 1003U, seq});
                    if (rank == 1U) {
                        check(hipMemcpyAsync(packet, scratch.packet + 16, 64U, hipMemcpyDeviceToHost, st), "control");
                        check(hipStreamSynchronize(st), "control");
                        apply(packet);
                        // with the drafter on both ranks the controllers advance together: same decision or a bug
                        if (drafting && ctl->draft_next() != (packet[8] != 0.0F))
                            throw std::logic_error("DSpark controller: the ranks disagree on drafting");
                    }
                    use_spec = packet[8] != 0.0F;
                    S.packet += since(t0);
                    return;
                }
                if (!gated && !policy.survival) return;
                const auto t0 = clock_type::now();
                float packet[8] = {use_spec ? 1.0F : 0.0F};
                for (std::uint32_t k = 0; k <= ds::block_size; ++k) {
                    const double v = verify_ema[k + 1U] != 0.0 ? verify_ema[k + 1U] : 0.0;
                    packet[1U + k] = static_cast<float>(v);
                }
                packet[7] = static_cast<float>(draft_ema);
                check(hipMemcpyAsync(scratch.packet, packet, 32U, hipMemcpyHostToDevice, st), "control");
                comm.exchange(st, scratch.packet, scratch.packet + 8, 32U,
                              strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 1002U, seq});
                if (rank == 1U) {
                    check(hipMemcpyAsync(packet, scratch.packet + 8, 32U, hipMemcpyDeviceToHost, st), "control");
                    check(hipStreamSynchronize(st), "control");
                }
                use_spec = packet[0] != 0.0F;  // both ranks continue from the packet's (float) values
                for (std::uint32_t k = 0; k <= ds::block_size; ++k) verify_ema[k + 1U] = packet[1U + k];
                draft_ema = packet[7];
                push_costs();
                S.packet += since(t0);
            };
            std::uint32_t decisions = 0;
            if (ctl) control(decisions++);  // V1 from the prompt before the first decision
            while (out.size() < a.generate) {
                if (gated && !use_spec) {  // a single step; its row is staged for the drafter
                    const auto t0 = clock_type::now();
                    const std::uint32_t one[1] = {t};
                    t = fwd.step_window(one)[0];
                    const double vs = since(t0);
                    S.verify += vs;
                    ++S.verify_by_rows[1].first;
                    S.verify_by_rows[1].second += vs;
                    ++S.plain_steps;
                    out.push_back(t);
                    const auto t1 = clock_type::now();
                    if (drafting) {
                        const auto caps = fwd.captured_streams();
                        ds::enqueue_main_hidden(caps[0], caps[1], caps[2], 1U, scratch.main_hidden + std::size_t{staged} * ds::main_width, st);
                        if (++staged == 128U) flush();
                        const std::uint32_t one_out[1] = {t};
                        drafter->add_vocabulary(one_out);
                    }
                    S.append += since(t1);
                    if (ctl) {
                        auto c = ctl->costs();
                        ++c.samples[1];
                        ctl->set_costs(c);
                        if (rank == 0U) cost_sum[1] += vs, ++cost_n[1];
                    } else if (rank == 0U) {
                        ema(verify_ema[1], vs);
                        ema(rate_plain, 1.0 / since(t0));
                        if (++since_switch >= 16U || (rate_spec != 0.0 && rate_spec > rate_plain)) { use_spec = true; since_switch = 0; }
                    }
                    control(decisions++);
                    continue;
                }
                const auto t_window = clock_type::now();
                if (drafting) flush();  // rows of single steps since the last draft
                // draft (both ranks: split head; or rank 0 alone, then the window packet)
                auto t0 = clock_type::now();
                ds::Draft d{};
                std::uint32_t k = 0;
                std::vector<std::uint32_t> drafts;
                if (oracle) {  // the reference continuation, wrong from row `bad` on (none when bad == 0)
                    const std::uint32_t R = oracle_rows[S.windows % oracle_rows.size()];
                    k = R - 1U;
                    const std::uint32_t bad = (S.windows * 7U + S.windows / 3U) % R;
                    for (std::uint32_t j = 1; j <= k; ++j) {
                        const std::size_t at = out.size() - 1U + j;
                        std::uint32_t v = at < reference.size() ? reference[at] : 0U;
                        if (bad != 0U && j >= bad) v = (v + 1U) % 129280U;
                        drafts.push_back(v);
                    }
                } else if (drafter) {
                    d = drafter->draft(t, false);
                    k = policy.choose(d);
                }
                if (!forced.empty() && !oracle) k = forced[S.windows % forced.size()] - 1U;
                const double draft_s = since(t0);
                S.draft += draft_s;
                if (!split && !oracle) {
                    t0 = clock_type::now();
                    std::uint32_t packet[8] = {k, d.tokens[1], d.tokens[2], d.tokens[3], d.tokens[4], d.tokens[5], 0U, 0U};
                    check(hipMemcpyAsync(scratch.packet, packet, 32U, hipMemcpyHostToDevice, st), "packet");
                    comm.exchange(st, scratch.packet, scratch.packet + 8, 32U,
                                  strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 1001U, S.windows});
                    if (rank == 1U) {
                        check(hipMemcpyAsync(packet, scratch.packet + 8, 32U, hipMemcpyDeviceToHost, st), "packet");
                        check(hipStreamSynchronize(st), "packet");
                        k = packet[0];
                        for (std::uint32_t j = 0; j < ds::block_size; ++j) d.tokens[j + 1U] = packet[j + 1U];
                    }
                    S.packet += since(t0);
                }
                if (!oracle) drafts.assign(d.tokens.begin() + 1, d.tokens.begin() + 1 + k);
                k = std::min<std::uint32_t>(k, a.generate - static_cast<std::uint32_t>(out.size()) - 1U);  // no rows past the end
                drafts.resize(k);
                // verify window [t, d_1..d_k]
                std::vector<std::uint32_t> w{t};
                w.insert(w.end(), drafts.begin(), drafts.end());
                t0 = clock_type::now();
                const double c0 = comm.stats().seconds;
                const auto g = fwd.step_window(w);
                const double vs = since(t0);
                S.verify += vs;
                S.verify_exchange += comm.stats().seconds - c0;
                auto& vr = S.verify_by_rows[k + 1U];
                ++vr.first;
                vr.second += vs;
                std::uint32_t accepted = 0;
                while (accepted < k && drafts[accepted] == g[accepted]) ++accepted;
                const std::size_t trace_pos = out.size() - 1U;
                ++S.windows;
                S.verified += k;
                S.accepted += accepted;
                for (std::uint32_t j = 0; j < k && j <= accepted && j < ds::block_size; ++j) {
                    ++S.reached[j];
                    if (j < accepted) ++S.hit[j];
                }
                if (!oracle) policy.observe(d, k, accepted);
                for (std::uint32_t j = 0; j <= accepted; ++j) out.push_back(g[j]);
                t = g[accepted];
                // commit the accepted prefix and append its rows to the drafter
                t0 = clock_type::now();
                fwd.commit_window(accepted + 1U);
                if (drafting) {
                    const auto caps = fwd.captured_streams();
                    ds::enqueue_main_hidden(caps[0], caps[1], caps[2], accepted + 1U, scratch.main_hidden, st);
                    drafter->append_hidden_device(scratch.main_hidden, accepted + 1U, st);
                    drafter->add_vocabulary(std::span<const std::uint32_t>(out).last(accepted + 1U));
                }
                S.append += since(t0);
                if (trace != nullptr) {
                    std::fprintf(trace, "{\"p\":%zu,\"pos\":%zu,\"ctx\":%zu,\"t\":%u,\"R\":%u,\"d\":[%u,%u,%u,%u,%u],\"c\":[%.4f,%.4f,%.4f,%.4f,%.4f],\"g\":[",
                                 si, trace_pos, prompt.size() + trace_pos, w[0], k + 1U, d.tokens[1], d.tokens[2], d.tokens[3], d.tokens[4],
                                 d.tokens[5], d.confidence[0], d.confidence[1], d.confidence[2], d.confidence[3], d.confidence[4]);
                    for (std::uint32_t j = 0; j <= k; ++j) std::fprintf(trace, j ? ",%u" : "%u", g[j]);
                    std::fprintf(trace, "],\"a\":%u,\"D\":%.3f,\"V\":%.3f,\"A\":%.3f}\n", accepted, draft_s * 1e3, vs * 1e3, since(t0) * 1e3);
                }
                if (ctl) {
                    auto c = ctl->costs();
                    ++c.samples[k + 1U];
                    ctl->set_costs(c);
                    if (rank == 0U) {
                        cost_sum[k + 1U] += vs, ++cost_n[k + 1U];
                        draft_sum += draft_s, extra_sum += since(t0), ++draft_n;  // append (+ the control below, next time)
                    }
                } else if (gated && rank == 0U) {
                    ema(draft_ema, draft_s);
                    ema(verify_ema[k + 1U], vs);
                    ema(rate_spec, static_cast<double>(accepted + 1U) / since(t_window));
                    if (++since_switch >= 16U || (rate_plain != 0.0 && rate_plain > rate_spec)) { use_spec = false; since_switch = 0; }
                } else if (policy.survival && rank == 0U) {
                    ema(draft_ema, draft_s);
                    ema(verify_ema[k + 1U], vs);
                }
                if (!oracle) control(decisions++);
            }
            if (drafting) flush();
        }
        S.total = since(t_gen);
        out.resize(std::min<std::size_t>(out.size(), a.generate));
        if (trace != nullptr) {
            std::fprintf(trace, "{\"p\":%zu,\"done\":1,\"prompt\":[", si);
            for (std::size_t j = 0; j < prompt.size(); ++j) std::fprintf(trace, j ? ",%u" : "%u", prompt[j]);
            std::fprintf(trace, "],\"out\":[");
            for (std::size_t j = 0; j < out.size(); ++j) std::fprintf(trace, j ? ",%u" : "%u", out[j]);
            std::fprintf(trace, "],\"gen_s\":%.4f,\"prompt_s\":%.4f}\n", S.total, prompt_seconds);
            std::fflush(trace);
        }
        if (!print) continue;
        std::uint64_t h = 14695981039346656037ULL;
        for (const auto v : out) h = (h ^ v) * 1099511628211ULL;
        std::printf("tokens (%zu, fnv %016llx):", out.size(), static_cast<unsigned long long>(h));
        for (const auto v : out) std::printf(" %u", v);
        std::printf("\n");
        const auto& cs = comm.stats();
        if (a.mode != "bench") {
            const double n = static_cast<double>(out.size() - 1U);  // tokens produced by the generation loop
            std::printf("prompt %zu tokens in %.2f s; generation %.0f tokens in %.3f s = %.2f tok/s\n", prompt.size(), prompt_seconds, n, S.total,
                        n / S.total);
            std::printf("time (ms): draft %.1f, packet %.1f, verify %.1f (exchange %.1f), commit+append %.1f, other %.1f\n", S.draft * 1e3,
                        S.packet * 1e3, S.verify * 1e3, S.verify_exchange * 1e3, S.append * 1e3,
                        (S.total - S.draft - S.packet - S.verify - S.append) * 1e3);
            for (const auto& [R, v] : S.verify_by_rows)
                std::printf("  verify R%u: %u windows, %.2f ms each\n", R, v.first, v.second * 1e3 / std::max(1U, v.first));
        }
        if (spec || oracle) {
            std::printf("windows %u, drafts verified %u, accepted %u (%.3f per window, %.2f tokens per window)\n", S.windows, S.verified, S.accepted,
                        static_cast<double>(S.accepted) / std::max(1U, S.windows), 1.0 + static_cast<double>(S.accepted) / std::max(1U, S.windows));
            std::printf("survival P(d_j ok | d_<j ok):");
            for (std::uint32_t j = 0; j < ds::block_size; ++j)
                std::printf(" %u/%u=%.3f", S.hit[j], S.reached[j], S.reached[j] ? static_cast<double>(S.hit[j]) / S.reached[j] : 0.0);
            std::printf("\n");
            std::printf("draft per window %.2f ms; single steps (gate) %u\n", S.draft * 1e3 / std::max(1U, S.windows), S.plain_steps);
        }
        std::printf("exchanges %llu, %.1f MB sent, %.3f s in exchange (stage %.3f, wire %.3f, check %.3f); wall %.1f s\n",
                    static_cast<unsigned long long>(cs.exchanges), static_cast<double>(cs.bytes_sent) / 1e6, cs.seconds, cs.stage, cs.wire, cs.check,
                    since(t_start));
        std::fflush(stdout);
    }
}

// One node alone (world 1; --stream-experts: the routed experts read from the file per layer, decode R = 1 only):
// greedy, or spec verified row by row (the target is stepped with t, then with each draft only while the previous one
// matched; a rejected draft is never fed, so no rollback): the accepted prefixes, survival and tokens equal the
// window path's; times are not two-node times.
void run_single(const Args& a) {
    const bool spec = a.mode == "spec";
    strix::deepseek41::DeviceForwardOptions o;
    o.layers = a.layers;
    o.max_positions = a.max_positions;
    o.world = 1;
    o.op_hashes = false;
    o.gather_logits = a.mode == "dump";
    o.engram_direct_io = a.engram_direct;
    o.capture_main_hidden = spec || a.mode == "dump";
    o.stream_experts = a.stream_experts;
    strix::deepseek41::DeviceForward fwd(a.model, o);
    if (a.mode == "dump") {
        run_dump(a, fwd, true);
        return;
    }
    auto* st = static_cast<hipStream_t>(fwd.stream());
    const auto prompt = parse_list(a.prompt);
    const auto choice = ds::choose_vocabulary(a.vocab, prompt);
    std::unique_ptr<ds::DeviceDrafter> drafter;
    if (spec) {
        auto w = ds::load_weights(a.model, a.draft);
        drafter = std::make_unique<ds::DeviceDrafter>(w->weights);
        drafter->set_vocabulary(choice.vocabulary);
    }
    std::printf("one node: layers [0, %u)%s, device weights %.2f GiB loaded in %.1f s; mode %s, vocabulary %s\n", a.layers,
                a.stream_experts ? ", experts streamed" : "", static_cast<double>(fwd.weight_bytes()) / 1073741824.0, fwd.load_seconds(),
                a.mode.c_str(), spec ? choice.label.c_str() : "-");
    Scratch scratch;
    std::uint32_t staged = 0;
    const auto stage_row = [&] {
        const auto caps = fwd.captured_streams();
        ds::enqueue_main_hidden(caps[0], caps[1], caps[2], 1U, scratch.main_hidden + std::size_t{staged} * ds::main_width, st);
        ++staged;
    };
    const auto flush = [&] {
        if (staged != 0U) drafter->append_hidden_device(scratch.main_hidden, staged, st);
        staged = 0;
    };
    std::uint32_t t = 0;
    for (const auto token : prompt) {
        (void)fwd.step(token);
        t = fwd.next_token();
        if (spec) {
            stage_row();
            if (staged == 128U) flush();
        }
    }
    if (spec) {
        flush();
        drafter->add_vocabulary(prompt);
        const std::uint32_t first[1] = {t};
        drafter->add_vocabulary(first);
    }
    std::vector<std::uint32_t> out{t};
    Stats S;
    const auto t_gen = clock_type::now();
    while (out.size() < a.generate) {
        if (!spec) {
            (void)fwd.step(t);
            t = fwd.next_token();
            out.push_back(t);
            continue;
        }
        auto t0 = clock_type::now();
        const auto d = drafter->draft(t, false);
        S.draft += since(t0);
        const std::uint32_t k = std::min<std::uint32_t>(ds::block_size, a.generate - static_cast<std::uint32_t>(out.size()) - 1U);
        t0 = clock_type::now();
        (void)fwd.step(t);
        std::uint32_t g = fwd.next_token();
        stage_row();
        std::uint32_t accepted = 0;
        while (accepted < k && d.tokens[accepted + 1U] == g) {
            out.push_back(g);
            ++accepted;
            if (accepted == k) break;
            (void)fwd.step(d.tokens[accepted]);
            g = fwd.next_token();
            stage_row();
        }
        if (accepted == k && k > 0U) {  // every verified draft matched: one more row gives the next token
            (void)fwd.step(d.tokens[accepted]);
            g = fwd.next_token();
            stage_row();
        }
        S.verify += since(t0);
        ++S.windows;
        S.verified += k;
        S.accepted += accepted;
        for (std::uint32_t j = 0; j < k && j <= accepted; ++j) {
            ++S.reached[j];
            if (j < accepted) ++S.hit[j];
        }
        out.push_back(g);
        t = g;
        t0 = clock_type::now();
        flush();
        drafter->add_vocabulary(std::span<const std::uint32_t>(out).last(accepted + 1U));
        S.append += since(t0);
    }
    S.total = since(t_gen);
    out.resize(std::min<std::size_t>(out.size(), a.generate));
    std::uint64_t h = 14695981039346656037ULL;
    for (const auto v : out) h = (h ^ v) * 1099511628211ULL;
    std::printf("tokens (%zu, fnv %016llx):", out.size(), static_cast<unsigned long long>(h));
    for (const auto v : out) std::printf(" %u", v);
    std::printf("\ngeneration %zu tokens in %.1f s (one node%s; not a two-node time)\n", out.size() - 1U, S.total,
                a.stream_experts ? ", experts streamed" : "");
    if (spec) {
        std::printf("windows %u, drafts verified %u, accepted %u (%.3f per window, %.2f tokens per window)\n", S.windows, S.verified, S.accepted,
                    static_cast<double>(S.accepted) / std::max(1U, S.windows), 1.0 + static_cast<double>(S.accepted) / std::max(1U, S.windows));
        std::printf("survival P(d_j ok | d_<j ok):");
        for (std::uint32_t j = 0; j < ds::block_size; ++j)
            std::printf(" %u/%u=%.3f", S.hit[j], S.reached[j], S.reached[j] ? static_cast<double>(S.hit[j]) / S.reached[j] : 0.0);
        std::printf("\ndraft per window %.2f ms\n", S.draft * 1e3 / std::max(1U, S.windows));
    }
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) try {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
            return std::string(argv[++i]);
        };
        if (s == "--rank") a.rank = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--transport") a.transport = next();
        else if (s == "--layers") a.layers = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--prompt-ids") a.prompt = next();
        else if (s == "--generate") a.generate = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--max-positions") a.max_positions = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--mode") a.mode = next();
        else if (s == "--draft") a.placement = next();
        else if (s == "--vocab") a.vocab = next();
        else if (s == "--bench-rows") a.bench_rows = next();
        else if (s == "--bench-reps") a.bench_reps = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--reference") a.reference = next();
        else if (s == "--prompt-file") a.prompt_file = next();
        else if (s == "--dump") a.dump = next();
        else if (s == "--fill-rows") a.fill_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--oracle-rows") a.oracle_rows = next();
        else if (s == "--engram-direct") a.engram_direct = true;
        else if (s == "--gate") a.gate = true;
        else if (s == "--tp-attention") a.tp_attention = true;
        else if (s == "--stream-experts") a.stream_experts = true;
        else if (s == "--connect-timeout") a.connect_timeout = static_cast<unsigned>(std::stoul(next()));
        else if (a.model.empty()) a.model = s;
        else if (a.draft.empty()) a.draft = s;
        else throw std::invalid_argument("unknown argument " + s);
    }
    if (a.model.empty() || a.draft.empty() || (a.prompt.empty() && a.prompt_file.empty()) || a.generate < 2U ||
        (a.mode == "dump" && (a.prompt_file.empty() || a.dump.empty())))
        throw std::invalid_argument("usage: strix-deepseek41-dspark-rank MODEL DRAFT --rank R --transport T --prompt-ids a,b,c ...");
    if (a.mode != "greedy" && a.mode != "spec" && a.mode != "bench" && a.mode != "oracle" && a.mode != "dump")
        throw std::invalid_argument("--mode greedy | spec | bench | oracle | dump");
    if (a.placement != "split" && a.placement != "rank0") throw std::invalid_argument("--draft split | rank0");
    if (a.transport == "single") {
        if (a.mode != "greedy" && a.mode != "spec" && a.mode != "dump")
            throw std::invalid_argument("--transport single runs greedy, spec or dump");
        run_single(a);
        return 0;
    }
    if (a.transport == "emulate") {  // both ranks in this process over the in-memory pair; rank 0 prints
        auto [t0, t1] = strix::dist::memory_pair();
        std::exception_ptr failure;
        std::thread peer([&, t = std::move(t1)]() mutable {
            try {
                run_rank(a, 1U, std::move(t), false);
            } catch (...) {
                failure = std::current_exception();
            }
        });
        try {
            run_rank(a, 0U, std::move(t0), true);
        } catch (...) {
            peer.join();
            throw;
        }
        peer.join();
        if (failure) std::rethrow_exception(failure);
        return 0;
    }
    // A blocked open() of a stream on a dead link would hold this node (and the peer) indefinitely.
    ::alarm(a.connect_timeout);
    auto opened = strix::dist::open_pair(a.transport, a.rank, {std::chrono::seconds(a.connect_timeout), strix::dist::all_transport_capabilities, true});
    auto transport = std::move(opened.host);
    ::alarm(0);
    std::printf("connected (%s)\n", transport->name().c_str());
    std::fflush(stdout);
    run_rank(a, a.rank, std::move(transport), true, std::move(opened.direct));
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
