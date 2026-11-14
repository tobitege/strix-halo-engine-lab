#include "strix/device/hip_gate.hpp"
#include "strix/dist/open_transport.hpp"
#include "strix/deepseek41_batch_dspark.hpp"
#include "strix/deepseek41_batch_forward.hpp"
#include "strix/deepseek41_batch_schedule.hpp"
#include "strix/deepseek41_pair_defaults.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/dist/gpu_direct.hpp"
#include "strix/ds41_dspark_files.hpp"
#include "strix/ds41_engram_tp.hpp"
#include "strix/ds41_touch_batch.hpp"
#include "strix/ds41_exact_flags.hpp"
#include "strix/hip_ds41_attention.hpp"
#include "strix/hip_ds41_dspark.hpp"

#include "ds41_hetero_stage.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

// Several requests decoded together on the two-node deepseek41 runner (BatchForward), or both ranks in one process
// (--transport emulate). Gate and measurement tool:
//   strix-deepseek41-batch MODEL.gguf --rank R --transport usb4auto|usb4gpu|tcp-listen:P|tcp-connect:H:P|emulate
//       --prompts a.ids,b.ids,... --n 1,2,4,8 --generate G [--mode batch|alone|both] [--prompt-rows 8]
//       [--fill K --fill-ids FILE] [--layers L] [--max-positions P] [--tp-attention] [--engram-direct]
//       [--check] [--gather] [--print-rows] [--sync-start] [--pool N] [--context-state FILE]
//       [--mode spec --draft DSPARK.gguf [--vocab SPEC] [--window controller|fixed:R] [--spec-only]]
// Request i of a run reads prompt i (mod the list). --mode both runs every request alone first (one row per step in
// decode, its prompt in chunks of --prompt-rows), then all N together, and compares, per request and position, the
// fed token, the greedy token and (with --check) the FNV of this rank's logits slice (--gather: of the full logits):
// "GATE PASS" only if all are equal. --mode windows: verify windows of all N requests in one step (see windows_gate). --fill: a context of K tokens (cycled from FILE) read once into a template slot
// and copied into every request's slot before its prompt (decode measured after a long context). --pool N: the slots'
// KV caches come from a shared pool of N positions (BatchForwardOptions::pool_positions); every request reserves its
// fill + prompt + generated positions (the tokens must not change). --context-state FILE: every request starts from the
// attention state of FILE (a long prefilled context, strix-deepseek41-longctx --state-out) imported into its slot; the
// prompts must be the tokens that follow it. --mode spec: DSpark windows of every request inside the batch steps
// (BatchSpeculator: the drafter on rank 0, one drafting context per request, per-request windows from
// BatchWindowController or fixed:R); every request's tokens (and, with --check, every kept row's logits) must equal
// its run alone ("GATE spec"); prints the decode-only aggregate tok/s, tokens per window and the window histogram.
// --spec-only skips the alone runs (timing). Rank 0 plans; a 256-byte control packet per step (rows, the commits of
// the previous step) keeps rank 1 on the same rows.
//
// Split decode with the workstation GPU (ds41_hetero_stage.hpp, strix/ds41-pipeline-multi-v1), --mode batch / both / spec:
//   --hetero-listen HOST:PORT --hetero-token-file FILE [--hetero-cut 22] [--hetero-groups G] [--hetero-alone whole|split]
//   [--hetero-local]
// The batch run's layers [0, cut) of every request run on a FRONT of sequence slots (strix-cuda-ds41-multi-front, which
// connects to rank 0's HOST:PORT), the pair runs [cut, 40) and the head (BatchForward::step_boundary). The N requests
// are split into G groups of consecutive slots (default: 1 for N <= 2, else 2); each group's next rows go to the FRONT as
// soon as its step ends, so the FRONT computes one group while the pair steps another. --mode both: the alone runs are
// the pair alone (whole forward, the reference; --hetero-alone split runs them through the FRONT too), so "GATE N PASS"
// means every split, batched request equals the pair-only run of that request alone (tokens, and logits with --check).
// The final attention-state digests are not compared in a split run (the pair does not hold its layers [0, cut)).
// --mode spec: the DSpark windows of each group are planned (and drafted) as soon as its last step ends, submitted to the
// FRONT and parked in the speculator (BatchSpeculator::park / resume) while the other group steps; the controller costs
// each step with its whole wall (the FRONT's boundaries, the forward and the TAIL step).
// --hetero-local: no FRONT process; the FRONT half of every step runs on this forward (BatchForward::step_front) and its
// boundaries go through the wire's encode / decode (a gate of the split on the pair alone). Prints per N the decode
// window aggregate (every request decoding, until the first one ends), the FRONT wait, the forward to rank 1 and the
// TAIL step per step, and the bytes per row.
namespace {

using strix::deepseek41::BatchForward;
using strix::deepseek41::BatchRow;
using strix::deepseek41::BatchSequence;

struct Args {
    std::string model, transport = "emulate", prompts, fill_ids, mode = "batch", n_list = "1,2,4,8";
    std::uint32_t rank = 0, layers = 40, generate = 32, max_positions = 4096, fill = 0, prompt_rows = 8;
    unsigned connect_timeout = 60;
    bool tp = false, engram_direct = false, check = false, gather = false, print_rows = false;
    bool sync_start = false;  // timing: every prompt is read before any request decodes (N rows in every decode step)
    std::uint32_t pool = 0;   // BatchForwardOptions::pool_positions
    std::string context_state;
    std::string state_out;
    const strix::deepseek41::AttentionState* context = nullptr;  // loaded --context-state
    std::string draft, vocab = "full", window = "controller";      // --mode spec
    bool spec_only = false;
    std::uint32_t draft_ranks = 2;  // 2 (default since P17.2): drafts on both ranks (a request's context on rank slot % 2); 1: rank 0
    // --aba K [--aba-flags filter,direct,seghpw,fold,f16,live,engtp] (--mode batch): each N run 2K times from the same start, the listed
    // attention flags (STRIX_DS41_IDX_CAND_FILTER, STRIX_DS41_IDX_DIRECT, STRIX_DS41_ATTN_SEG_HPW,
    // STRIX_DS41_ATTN_WINDOW_FOLD, STRIX_DS41_ATTN_TILE_F16, STRIX_DS41_ATTN_LIVE_CHUNKS, STRIX_DS41_ENGRAM_TP; touch =
    // STRIX_DS41_TOUCH_BATCH, rowviews / normilp / routes / cfast / swz as in the rank app) off in even arms, on in
    // odd ones; one "aba arm" line each (decode ms/step, fingerprint of every row's fed / argmax / logits hash with
    // --check and the final attention states) and "aba exact IDENTICAL|DIFFER" (every arm == arm 0).
    std::uint32_t aba = 0;
    std::string aba_flags = "filter,direct,seghpw";
    // split decode (ds41_hetero_stage.hpp)
    std::string hetero_listen, hetero_token;
    std::uint32_t hetero_cut = 22, hetero_groups = 0;
    bool hetero_local = false, hetero_alone_split = false;
    [[nodiscard]] bool hetero() const { return hetero_local || !hetero_listen.empty(); }
};

std::uint64_t fnv(std::span<const float> v) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const float f : v) {
        std::uint32_t b;
        std::memcpy(&b, &f, 4);
        for (int k = 0; k < 4; ++k, b >>= 8U) h = (h ^ (b & 0xffU)) * 1099511628211ULL;
    }
    return h;
}

std::vector<std::uint32_t> read_ids(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::invalid_argument("cannot read " + path);
    std::vector<std::uint32_t> ids;
    for (std::string t; std::getline(f, t, ',');) {
        const auto b = t.find_first_not_of(" \r\n\t");
        if (b != std::string::npos) ids.push_back(static_cast<std::uint32_t>(std::stoul(t.substr(b))));
    }
    if (ids.empty()) throw std::invalid_argument(path + " holds no token ids");
    return ids;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, sep);)
        if (!t.empty()) out.push_back(t);
    return out;
}

// One row's record: the request's position, the token fed, the greedy token and the logits hash.
struct RowRecord {
    std::uint32_t fed = 0, argmax = 0;
    std::uint64_t hash = 0;
};
using Trace = std::map<std::uint32_t, RowRecord>;  // by position, per request

struct RunResult {
    std::vector<Trace> traces;
    std::vector<std::uint64_t> state_hashes;
    std::uint64_t steady_steps = 0, steady_rows = 0;
    double steady_seconds = 0.0, steady_exchange = 0.0, steady_engram = 0.0;
    double steady_wall = 0.0;  // the same steps end to end on the host (planning and bookkeeping included)
    double steady_union = 0.0;        // mean over steps of the mean union per layer (this rank)
    double steady_max_rows = 0.0;     // mean over steps of the mean max rows of one expert
    double steady_expert_bytes = 0.0; // per step
    std::uint64_t steps = 0;
    double seconds = 0.0;
    // split runs: the decode window (from the first step with every request decoding to the end of the first request)
    std::uint64_t window_rows = 0, window_steps = 0;
    double window_seconds = 0.0;
    strix::ds41::hetero::StageStats stage;
    std::uint32_t groups = 0;
};

// A fresh request in `slot` with room for `positions` (the KV pool's reservation, or a reset without a pool).
void claim(BatchForward& fwd, const Args& a, std::uint32_t slot, std::uint64_t positions) {
    const auto want = static_cast<std::uint32_t>(std::min<std::uint64_t>(positions, a.max_positions));
    if (!fwd.reserve(slot, want)) throw std::runtime_error("no room for slot " + std::to_string(slot) + " (" + std::to_string(want) + " positions)");
}

// Runs `requests` (prompt per request) together; slots 0..n-1, each starting from the template slot when filling.
RunResult run(BatchForward& fwd, const Args& a, const std::vector<std::vector<std::uint32_t>>& prompts, std::uint32_t n,
              std::uint32_t template_slot, bool filled, bool print) {
    std::vector<BatchSequence> seqs(n);
    RunResult res;
    res.traces.resize(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        claim(fwd, a, i, std::uint64_t{a.fill} + (a.context ? a.context->position : 0U) + prompts[i].size() + a.generate + 16U);
        if (a.context) fwd.import_state(i, *a.context);
        else if (filled) fwd.copy_slot(template_slot, i);
        seqs[i].slot = i;
        seqs[i].position = fwd.position(i);
        seqs[i].prompt = prompts[i];
    }
    std::vector<std::uint32_t> map_back(n);
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        std::vector<BatchSequence*> active;
        std::vector<std::uint32_t> index;
        for (std::uint32_t i = 0; i < n; ++i)
            if (seqs[i].generated.size() < a.generate) {
                active.push_back(&seqs[i]);
                index.push_back(i);
            }
        if (active.empty()) break;
        if (a.sync_start && std::any_of(active.begin(), active.end(), [](const BatchSequence* q) { return q->reading_prompt(); })) {
            std::vector<BatchSequence*> reading;
            std::vector<std::uint32_t> reading_index;
            for (std::size_t k = 0; k < active.size(); ++k)
                if (active[k]->reading_prompt()) reading.push_back(active[k]), reading_index.push_back(index[k]);
            active = std::move(reading);
            index = std::move(reading_index);
        }
        const auto t_step = std::chrono::steady_clock::now();
        const auto plan = strix::deepseek41::plan_step(active, a.prompt_rows, fwd.row_limit());
        std::vector<std::uint32_t> positions(plan.rows.size());
        for (std::size_t r = 0; r < plan.rows.size(); ++r) positions[r] = active[plan.owner[r]]->position;  // fixed up below
        {  // the position of every row: its sequence's position plus its rank inside the segment
            std::map<std::uint32_t, std::uint32_t> next;
            for (std::size_t r = 0; r < plan.rows.size(); ++r) {
                auto [it, fresh] = next.emplace(plan.owner[r], positions[r]);
                positions[r] = it->second++;
                (void)fresh;
            }
        }
        const auto& tokens = fwd.step(plan.rows);
        const auto& st = fwd.last_stats();
        bool decode_only = plan.rows.size() == n;
        for (std::size_t r = 0; r < plan.rows.size(); ++r) {
            const std::uint32_t req = index[plan.owner[r]];
            RowRecord rec{plan.rows[r].token, tokens[r], 0};
            if (a.check) rec.hash = fnv(fwd.row_logits(static_cast<std::uint32_t>(r)));
            res.traces[req][positions[r]] = rec;
            if (active[plan.owner[r]]->reading_prompt()) decode_only = false;
            if (print && a.print_rows)
                std::printf("req %u pos %u fed %u argmax %u logits_fnv %016llx\n", req, positions[r], rec.fed, rec.argmax,
                            static_cast<unsigned long long>(rec.hash));
        }
        strix::deepseek41::apply_step(active, plan, tokens);
        ++res.steps;
        if (decode_only && res.steps > 1U) {
            res.steady_wall += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_step).count();
            ++res.steady_steps;
            res.steady_rows += plan.rows.size();
            res.steady_seconds += st.seconds;
            res.steady_exchange += st.exchange_seconds;
            res.steady_engram += st.engram_seconds;
            double u = 0.0, mr = 0.0;
            for (std::size_t l = 0; l < st.union_experts.size(); ++l) {
                u += st.union_experts[l];
                mr += st.max_rows_per_expert[l];
            }
            res.steady_union += u / static_cast<double>(std::max<std::size_t>(st.union_experts.size(), 1U));
            res.steady_max_rows += mr / static_cast<double>(std::max<std::size_t>(st.union_experts.size(), 1U));
            res.steady_expert_bytes += static_cast<double>(st.expert_bytes);
        }
    }
    res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (a.check)
        for (std::uint32_t i = 0; i < n; ++i) {
            const auto hash = fwd.export_state(i).digest(true);
            res.state_hashes.push_back(hash);
            if (print && a.print_rows) std::printf("state req %u fnv %016llx\n", i, static_cast<unsigned long long>(hash));
        }
    return res;
}

// The batch run through a split decode (ds41_hetero_stage.hpp): the requests in `groups` groups of consecutive slots; every
// group's next rows are submitted to the FRONT as soon as its step ends, the steps go round robin over the groups.
RunResult run_hetero(BatchForward& fwd, strix::ds41::hetero::Stage& stage, const Args& a, const std::vector<std::vector<std::uint32_t>>& prompts,
                     std::uint32_t n, std::uint32_t groups, bool print) {
    std::vector<BatchSequence> seqs(n);
    RunResult res;
    res.traces.resize(n);
    res.groups = groups;
    const auto stage0 = stage.stats();
    for (std::uint32_t i = 0; i < n; ++i) {
        claim(fwd, a, i, (a.context ? a.context->position : 0U) + prompts[i].size() + a.generate + 16U);
        if (a.context) {
            fwd.import_state(i, *a.context);
            stage.import_state(i, *a.context);
        } else {
            stage.reset(i);
        }
        seqs[i].slot = i;
        seqs[i].position = fwd.position(i);
        seqs[i].prompt = prompts[i];
    }
    struct Group {
        std::vector<std::uint32_t> members;
        std::vector<BatchSequence*> active;
        std::vector<std::uint32_t> index;
        strix::deepseek41::PlannedStep plan;
        bool planned = false;
    };
    std::vector<Group> gs(groups);
    for (std::uint32_t i = 0; i < n; ++i) gs[std::min<std::uint32_t>(i * groups / n, groups - 1U)].members.push_back(i);
    const auto plan_group = [&](Group& g) {  // the group's next rows, submitted to the FRONT at once
        g.active.clear();
        g.index.clear();
        for (const auto i : g.members)
            if (seqs[i].generated.size() < a.generate) {
                g.active.push_back(&seqs[i]);
                g.index.push_back(i);
            }
        g.planned = !g.active.empty();
        if (!g.planned) return;
        g.plan = strix::deepseek41::plan_step(g.active, a.prompt_rows, fwd.row_limit());
        stage.submit(g.plan.rows);
    };
    const auto all_decoding = [&] {
        for (const auto& q : seqs)
            if (q.reading_prompt()) return false;
        return true;
    };
    for (auto& g : gs) plan_group(g);
    const auto t0 = std::chrono::steady_clock::now();
    bool window_open = false, window_closed = false;
    auto t_window = t0;
    for (bool any = true; any;) {
        any = false;
        for (auto& g : gs) {
            if (!g.planned) continue;
            any = true;
            const bool decoding_before = all_decoding();
            if (decoding_before && !window_open && !window_closed) {
                window_open = true;
                t_window = std::chrono::steady_clock::now();
            }
            const auto& plan = g.plan;
            std::vector<std::uint32_t> positions(plan.rows.size());
            {
                std::map<std::uint32_t, std::uint32_t> next;
                for (std::size_t r = 0; r < plan.rows.size(); ++r) {
                    auto [it, fresh] = next.emplace(plan.owner[r], g.active[plan.owner[r]]->position);
                    positions[r] = it->second++;
                    (void)fresh;
                }
            }
            const auto& tokens = stage.step(plan.rows);
            for (std::size_t r = 0; r < plan.rows.size(); ++r) {
                const std::uint32_t req = g.index[plan.owner[r]];
                RowRecord rec{plan.rows[r].token, tokens[r], 0};
                if (a.check) rec.hash = fnv(fwd.row_logits(static_cast<std::uint32_t>(r)));
                res.traces[req][positions[r]] = rec;
                if (print && a.print_rows)
                    std::printf("req %u pos %u fed %u argmax %u logits_fnv %016llx\n", req, positions[r], rec.fed, rec.argmax,
                                static_cast<unsigned long long>(rec.hash));
            }
            strix::deepseek41::apply_step(g.active, plan, tokens);
            ++res.steps;
            if (window_open && !window_closed) {
                res.window_rows += plan.rows.size();
                ++res.window_steps;
                bool ended = false;
                for (const auto i : g.index) ended = ended || seqs[i].generated.size() >= a.generate;
                if (ended) {  // the first request to end closes the window (later steps have fewer rows)
                    window_closed = true;
                    res.window_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_window).count();
                }
            }
            plan_group(g);
        }
    }
    res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (window_open && !window_closed) res.window_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_window).count();
    const auto& s1 = stage.stats();
    res.stage = s1;
    res.stage.steps -= stage0.steps;
    res.stage.segments -= stage0.segments;
    res.stage.rows -= stage0.rows;
    res.stage.bytes_in -= stage0.bytes_in;
    res.stage.bytes_forwarded -= stage0.bytes_forwarded;
    res.stage.wait_seconds -= stage0.wait_seconds;
    res.stage.forward_seconds -= stage0.forward_seconds;
    res.stage.tail_seconds -= stage0.tail_seconds;
    if (a.check && stage.local())  // only the local split holds every layer's state
        for (std::uint32_t i = 0; i < n; ++i) res.state_hashes.push_back(fwd.export_state(i).digest(true));
    return res;
}

// Verify windows of N requests in one step (the DSpark protocol with an oracle drafter): every decoding request
// proposes a window of w = 8 / N rows, its last token and the next w - 1 tokens of its run alone, with the draft at a
// random depth replaced by a wrong token; the step verifies every window greedily, commit() keeps the accepted rows
// and the correction. Every request must end with exactly its tokens alone (and, with --check, the same logits at
// every kept position).
bool windows_gate(BatchForward& fwd, const Args& a, const std::vector<std::vector<std::uint32_t>>& prompts, std::uint32_t n,
                  const std::vector<Trace>& alone, bool print) {
    const std::uint32_t w = std::min(BatchForward::max_rows, fwd.row_limit() / n);
    std::vector<BatchSequence> seqs(n);
    std::vector<std::vector<std::uint32_t>> truth(n);  // the tokens each request generates alone
    for (std::uint32_t i = 0; i < n; ++i) {
        claim(fwd, a, i, prompts[i].size() + a.generate + 16U);
        seqs[i].slot = i;
        seqs[i].prompt = prompts[i];
        for (const auto& [pos, rec] : alone[i])
            if (pos + 1U >= prompts[i].size()) truth[i].push_back(rec.argmax);
    }
    std::uint64_t state = 0x9E3779B97F4A7C15ULL;
    const auto rnd = [&] {
        state ^= state << 13U;
        state ^= state >> 7U;
        state ^= state << 17U;
        return state;
    };
    std::vector<Trace> traces(n);
    std::uint64_t steps = 0, windows = 0, rows_total = 0, rejected = 0;
    for (;;) {
        std::vector<BatchRow> rows;
        std::vector<std::uint32_t> owner, depth;
        std::vector<std::vector<std::uint32_t>> drafts(n);
        bool any = false;
        for (std::uint32_t i = 0; i < n; ++i) {
            auto& q = seqs[i];
            if (q.generated.size() >= a.generate) continue;
            any = true;
            if (q.reading_prompt()) {  // prompt chunks as in the batch runs
                std::size_t take = std::min<std::size_t>({a.prompt_rows, fwd.row_limit() - rows.size(), q.prompt.size() - q.fed});
                if (q.position == 0U) take = std::min<std::size_t>(take, 1U);
                if (take == 0U) continue;
                for (std::size_t k = 0; k < take; ++k) {
                    rows.push_back({q.slot, q.prompt[q.fed + k]});
                    owner.push_back(i);
                    depth.push_back(static_cast<std::uint32_t>(k));
                }
                continue;
            }
            // window: the last token, then drafts = the request's own next tokens, one of them wrong
            const std::size_t g = q.generated.size();
            const std::uint32_t room = static_cast<std::uint32_t>(std::min<std::size_t>({w, fwd.row_limit() - rows.size(), a.generate - g}));
            if (room == 0U) continue;
            const std::uint32_t wrong = static_cast<std::uint32_t>(rnd() % (room + 1U));  // == room: no wrong draft
            rows.push_back({q.slot, q.generated.back()});
            owner.push_back(i);
            depth.push_back(0U);
            for (std::uint32_t d = 1; d < room; ++d) {
                std::uint32_t t = g + d - 1U < truth[i].size() ? truth[i][g + d - 1U] : 0U;
                if (d == wrong) t = (t + 1U) % fwd.model().config.vocab_size;
                drafts[i].push_back(t);
                rows.push_back({q.slot, t});
                owner.push_back(i);
                depth.push_back(d);
            }
            ++windows;
        }
        if (!any) break;
        const auto& tokens = fwd.step(rows);
        ++steps;
        rows_total += rows.size();
        // per request: the accepted prefix, the commit, the emitted tokens
        for (std::uint32_t i = 0; i < n; ++i) {
            std::vector<std::size_t> mine;
            for (std::size_t r = 0; r < rows.size(); ++r)
                if (owner[r] == i) mine.push_back(r);
            if (mine.empty()) continue;
            auto& q = seqs[i];
            const std::uint32_t base = q.position;
            if (q.reading_prompt()) {
                for (std::size_t k = 0; k < mine.size(); ++k) {
                    const auto r = mine[k];
                    RowRecord rec{rows[r].token, tokens[r], a.check ? fnv(fwd.row_logits(static_cast<std::uint32_t>(r))) : 0U};
                    traces[i][base + static_cast<std::uint32_t>(k)] = rec;
                }
                q.position += static_cast<std::uint32_t>(mine.size());
                q.fed += mine.size();
                if (!q.reading_prompt()) q.generated.push_back(tokens[mine.back()]);
                continue;
            }
            std::size_t kept = 1;  // row 0 always; row k is kept when row k-1's greedy token equals draft k
            while (kept < mine.size() && tokens[mine[kept - 1U]] == drafts[i][kept - 1U]) ++kept;
            rejected += mine.size() - kept;
            fwd.commit(q.slot, static_cast<std::uint32_t>(kept));
            for (std::size_t k = 0; k < kept; ++k) {
                const auto r = mine[k];
                RowRecord rec{rows[r].token, tokens[r], a.check ? fnv(fwd.row_logits(static_cast<std::uint32_t>(r))) : 0U};
                traces[i][base + static_cast<std::uint32_t>(k)] = rec;
                if (q.generated.size() < a.generate) q.generated.push_back(tokens[r]);
            }
            q.position += static_cast<std::uint32_t>(kept);
        }
    }
    std::uint64_t compared = 0, differ = 0;
    for (std::uint32_t i = 0; i < n; ++i)
        for (const auto& [pos, rec] : traces[i]) {
            const auto it = alone[i].find(pos);
            if (it == alone[i].end()) continue;  // past the alone run's last fed row
            ++compared;
            if (it->second.fed != rec.fed || it->second.argmax != rec.argmax || it->second.hash != rec.hash) {
                if (print && differ < 10U)
                    std::printf("  DIFF windows req %u pos %u: alone fed %u argmax %u, windows fed %u argmax %u\n", i, pos, it->second.fed,
                                it->second.argmax, rec.fed, rec.argmax);
                ++differ;
            }
        }
    for (std::uint32_t i = 0; i < n; ++i) {
        const std::vector<std::uint32_t> head(truth[i].begin(), truth[i].begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(truth[i].size(), a.generate)));
        if (seqs[i].generated != head) ++differ;
    }
    if (print)
        std::printf("GATE windows N %u (w %u) %s: %llu steps, %llu windows, %llu rows (%llu rejected), %llu kept rows compared, %llu differ\n", n, w,
                    differ == 0U ? "PASS" : "FAIL", static_cast<unsigned long long>(steps), static_cast<unsigned long long>(windows),
                    static_cast<unsigned long long>(rows_total), static_cast<unsigned long long>(rejected), static_cast<unsigned long long>(compared),
                    static_cast<unsigned long long>(differ));
    return differ == 0U;
}

// One step's control packet from the planner (rank 0) to rank 1: the commits of the previous step, then the rows.
struct SpecPacket {
    std::uint32_t magic = 0x53504b31U, done = 0, rows = 0, commits = 0;
    std::uint32_t row[BatchForward::max_rows][2]{};     // slot, token
    std::uint32_t commit[BatchForward::max_rows][2]{};  // slot, kept
};
static_assert(sizeof(SpecPacket) <= strix::deepseek41::BatchSpeculator::round_bytes);

// The follower's side of the spec gate (rank 1): the rows of every step (fed token, argmax, this rank's logits slice) by
// position, the positions moved back by the commits (a commit (slot, kept) of the previous step keeps its first `kept`
// rows of that slot), then the kept rows against this rank's runs alone and each request's final attention state.
struct FollowerCheck {
    std::vector<std::uint32_t> pos, base;  // per slot: next position; the first position of its rows in the last step
    std::vector<Trace> rows;
    bool compare_states = true;  // false: a split decode with a FRONT process (the pair holds no layer before the cut)
    explicit FollowerCheck(std::uint32_t n) : pos(n, 0U), base(n, 0U), rows(n) {}
    void commit(std::uint32_t slot, std::uint32_t kept) { pos.at(slot) = base.at(slot) + kept; }
    void step(BatchForward& fwd, std::span<const BatchRow> step_rows, std::span<const std::uint32_t> tokens, bool hash) {
        std::vector<bool> seen(pos.size(), false);
        for (std::size_t r = 0; r < step_rows.size(); ++r) {
            const auto s = step_rows[r].slot;
            if (!seen.at(s)) seen[s] = true, base[s] = pos[s];
            rows[s][pos[s]++] = RowRecord{step_rows[r].token, tokens[r], hash ? fnv(fwd.row_logits(static_cast<std::uint32_t>(r))) : 0U};
        }
    }
    bool report(BatchForward& fwd, const Args& a, std::uint32_t n, const std::vector<Trace>& alone, const std::vector<std::uint64_t>& alone_state) {
        std::uint64_t compared = 0, differ = 0, states = 0;
        for (std::uint32_t i = 0; i < n; ++i) {
            for (const auto& [p, rec] : rows[i]) {
                if (p >= pos[i]) continue;  // dropped by a commit and not fed again
                const auto it = alone[i].find(p);
                if (it == alone[i].end()) continue;
                ++compared;
                if (it->second.fed != rec.fed || it->second.argmax != rec.argmax || it->second.hash != rec.hash) {
                    if (differ < 10U)
                        std::printf("  DIFF spec follower req %u pos %u: alone fed %u argmax %u, spec fed %u argmax %u\n", i, p, it->second.fed,
                                    it->second.argmax, rec.fed, rec.argmax);
                    ++differ;
                }
            }
            if (a.check && compare_states && fwd.export_state(i).digest(true) != alone_state[i]) ++states;
        }
        std::printf("GATE spec follower N %u %s: %llu kept rows compared (%s), %llu differ; final attention states %s\n", n,
                    differ == 0U && states == 0U ? "PASS" : "FAIL", static_cast<unsigned long long>(compared),
                    a.check ? (a.gather ? "tokens + full logits" : "tokens + this rank's logits slice") : "tokens",
                    static_cast<unsigned long long>(differ), !a.check ? "not compared (no --check)" : !compare_states ? "not compared (split)" : states == 0U ? "== alone" : "DIFFER");
        std::fflush(stdout);
        return differ == 0U && states == 0U;
    }
};

// DSpark windows inside the batch: N requests decoded with BatchSpeculator, compared with their runs alone (rank 0 its
// kept rows and final states below, rank 1 through FollowerCheck).
bool spec_gate(BatchForward& fwd, strix::ds41::dspark::DeviceDrafter* drafter, const Args& a, std::uint32_t rank, bool two_ranks,
               const std::vector<std::vector<std::uint32_t>>& prompts, std::uint32_t n, const std::vector<Trace>& alone,
               const std::vector<std::uint64_t>& alone_state, bool print, strix::ds41::hetero::Stage* stage = nullptr, std::uint32_t groups = 1) {
    std::vector<BatchSequence> seqs(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        claim(fwd, a, i, prompts[i].size() + a.generate + 16U);
        if (stage) stage->reset(i);
        seqs[i].slot = i;
        seqs[i].prompt = prompts[i];
    }
    // the TAIL step of `rows`: the whole forward, or a split decode's (its rows submitted to the FRONT first)
    const auto step_rows = [&](std::span<const BatchRow> rows) -> const std::vector<std::uint32_t>& {
        if (!stage) return fwd.step(rows);
        return stage->step(rows);
    };
    std::vector<Trace> traces(n);
    const auto send = [&](SpecPacket& pk) {  // one 512-byte message (the size of a draft round, which may come instead)
        if (!two_ranks) return;
        std::array<std::byte, strix::deepseek41::BatchSpeculator::round_bytes> o{}, i{};
        std::memcpy(o.data(), &pk, sizeof pk);
        fwd.exchange_host(o, i);
        if (rank == 1U) std::memcpy(&pk, i.data(), sizeof pk);
    };
    if (rank == 1U && a.draft_ranks == 2U) {  // the follower with its own drafter: the draft round, then the rows
        strix::deepseek41::BatchSpecOptions so;
        so.vocabulary = a.vocab;
        so.prompt_rows = a.prompt_rows;
        so.rank = 1U;
        so.world = 2U;
        so.exchange = [&](std::span<const std::byte> out, std::span<std::byte> in) { fwd.exchange_host(out, in); };
        strix::deepseek41::BatchSpeculator spec(fwd, *drafter, n, so);
        for (std::uint32_t i = 0; i < n; ++i) spec.start(i, prompts[i]);
        FollowerCheck check(n);
        check.compare_states = !stage || stage->local();
        std::vector<BatchRow> last;
        std::vector<std::uint32_t> last_tokens;
        std::uint64_t step = 0;
        for (;;) {  // a draft round (rank 1 drafts this step) or the step's packet
            std::array<std::byte, strix::deepseek41::BatchSpeculator::round_bytes> o{}, i{};
            fwd.exchange_host(o, i);
            std::uint32_t magic = 0;
            std::memcpy(&magic, i.data(), 4U);
            if (magic == strix::deepseek41::BatchSpeculator::round_magic) {
                std::vector<std::pair<std::uint32_t, std::uint32_t>> applied;
                const bool more = spec.follow_packet(i, last, last_tokens, step, &applied);
                for (const auto& [slot, kept] : applied) check.commit(slot, kept);
                if (!more) break;
                continue;
            }
            SpecPacket pk;
            std::memcpy(&pk, i.data(), sizeof pk);
            if (pk.magic != SpecPacket{}.magic) throw std::runtime_error("spec gate: bad control packet");
            for (std::uint32_t c = 0; c < pk.commits; ++c) {
                spec.follow_commit(pk.commit[c][0], pk.commit[c][1]);
                check.commit(pk.commit[c][0], pk.commit[c][1]);
            }
            spec.follow_stage(last, last_tokens, step);
            if (pk.done) break;
            last.assign(pk.rows, BatchRow{});
            for (std::uint32_t r = 0; r < pk.rows; ++r) last[r] = {pk.row[r][0], pk.row[r][1]};
            last_tokens = step_rows(last);
            check.step(fwd, last, last_tokens, a.check);
            ++step;
        }
        return a.spec_only || check.report(fwd, a, n, alone, alone_state);
    }
    if (rank == 1U) {  // the follower: commits, then the planner's rows
        FollowerCheck check(n);
        check.compare_states = !stage || stage->local();
        for (;;) {
            SpecPacket pk;
            send(pk);
            for (std::uint32_t c = 0; c < pk.commits; ++c) {
                fwd.commit(pk.commit[c][0], pk.commit[c][1]);
                check.commit(pk.commit[c][0], pk.commit[c][1]);
            }
            if (pk.done) break;
            std::vector<BatchRow> rows(pk.rows);
            for (std::uint32_t r = 0; r < pk.rows; ++r) rows[r] = {pk.row[r][0], pk.row[r][1]};
            const auto& tokens = step_rows(rows);
            check.step(fwd, rows, tokens, a.check);
        }
        return a.spec_only || check.report(fwd, a, n, alone, alone_state);
    }
    strix::deepseek41::BatchSpecOptions so;
    so.vocabulary = a.vocab;
    so.prompt_rows = a.prompt_rows;
    if (a.window.rfind("fixed:", 0) == 0) so.fixed_rows = static_cast<std::uint32_t>(std::stoul(a.window.substr(6)));
    else if (a.window != "controller") throw std::invalid_argument("--window is controller or fixed:R");
    const bool split = two_ranks && a.draft_ranks == 2U;
    if (split) {
        so.world = 2U;
        so.exchange = [&](std::span<const std::byte> out, std::span<std::byte> in) { fwd.exchange_host(out, in); };
    }
    double split_step = 0.0;  // a split decode: the wall of the last step (the FRONT's boundaries, the forward, the TAIL)
    if (stage) so.step_seconds = [&] { return split_step; };
    strix::deepseek41::BatchSpeculator spec(fwd, *drafter, n, so);
    for (std::uint32_t i = 0; i < n; ++i) spec.start(i, prompts[i]);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> commits;
    double wall_decode = 0.0;  // decode-only steps end to end on rank 0 (the control packet included)
    const auto t0 = std::chrono::steady_clock::now();
    // a split decode: `groups` groups of consecutive slots, each with its plan in flight (submitted to the FRONT and
    // parked in the speculator) while another group steps; the steps go round robin
    struct Group {
        std::vector<std::uint32_t> members;
        std::vector<BatchSequence*> active;
        std::vector<std::uint32_t> remaining;
        std::shared_ptr<strix::deepseek41::BatchSpeculator::ParkedPlan> parked;
        std::vector<BatchRow> rows;
        bool planned = false;
    };
    std::vector<Group> gs(stage ? groups : 0U);
    for (std::uint32_t i = 0; stage && i < n; ++i) gs[std::min<std::uint32_t>(i * groups / n, groups - 1U)].members.push_back(i);
    const auto plan_group = [&](Group& g) {
        g.active.clear();
        g.remaining.clear();
        for (const auto i : g.members)
            if (seqs[i].generated.size() < a.generate) {
                g.active.push_back(&seqs[i]);
                g.remaining.push_back(a.generate - static_cast<std::uint32_t>(seqs[i].generated.size()));
            }
        g.planned = !g.active.empty();
        if (!g.planned) return;
        const auto& plan = spec.plan(g.active, g.remaining);
        g.rows = plan.rows;
        stage->submit(g.rows);
        g.parked = spec.park();
    };
    if (stage) {
        for (auto& g : gs) plan_group(g);
        for (bool any = true; any;) {
            any = false;
            for (auto& g : gs) {
                if (!g.planned) continue;
                any = true;
                SpecPacket pk;  // rank 1: the commits not carried by a round yet, then this group's rows
                if (split) {
                    const auto c = spec.take_commits();
                    pk.commits = static_cast<std::uint32_t>(c.size());
                    for (std::size_t k = 0; k < c.size(); ++k) pk.commit[k][0] = c[k] >> 16U, pk.commit[k][1] = c[k] & 0xFFFFU;
                } else {
                    pk.commits = static_cast<std::uint32_t>(commits.size());
                    for (std::size_t c = 0; c < commits.size(); ++c) pk.commit[c][0] = commits[c].first, pk.commit[c][1] = commits[c].second;
                }
                pk.rows = static_cast<std::uint32_t>(g.rows.size());
                for (std::size_t r = 0; r < g.rows.size(); ++r) pk.row[r][0] = g.rows[r].slot, pk.row[r][1] = g.rows[r].token;
                send(pk);
                const auto t_step = std::chrono::steady_clock::now();
                const auto decode_before = spec.stats().decode_steps;
                spec.resume(g.parked);
                const auto& tokens = step_rows(g.rows);
                split_step = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_step).count();
                std::vector<std::pair<std::uint32_t, RowRecord>> recs(g.rows.size());
                std::map<std::uint32_t, std::uint32_t> at;
                for (std::size_t r = 0; r < g.rows.size(); ++r) {
                    const auto slot = g.rows[r].slot;
                    const auto [it, fresh] = at.emplace(slot, seqs[slot].position);
                    (void)fresh;
                    recs[r] = {it->second++, RowRecord{g.rows[r].token, tokens[r], a.check ? fnv(fwd.row_logits(static_cast<std::uint32_t>(r))) : 0U}};
                }
                commits = spec.finish(g.active, tokens);
                for (const auto& [slot, kept] : commits) stage->commit(slot, kept);
                if (spec.stats().decode_steps != decode_before) wall_decode += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_step).count();
                for (std::size_t r = 0; r < g.rows.size(); ++r) {
                    const auto slot = g.rows[r].slot;
                    if (recs[r].first < seqs[slot].position) traces[slot][recs[r].first] = recs[r].second;  // kept rows only
                }
                plan_group(g);
            }
        }
        SpecPacket pk;  // the end: the last commits, done
        if (split) {
            const auto c = spec.take_commits();
            pk.commits = static_cast<std::uint32_t>(c.size());
            for (std::size_t k = 0; k < c.size(); ++k) pk.commit[k][0] = c[k] >> 16U, pk.commit[k][1] = c[k] & 0xFFFFU;
        } else {
            pk.commits = static_cast<std::uint32_t>(commits.size());
            for (std::size_t c = 0; c < commits.size(); ++c) pk.commit[c][0] = commits[c].first, pk.commit[c][1] = commits[c].second;
        }
        pk.done = 1U;
        send(pk);
    }
    for (; !stage;) {
        std::vector<BatchSequence*> active;
        std::vector<std::uint32_t> remaining;
        for (auto& q : seqs)
            if (q.generated.size() < a.generate) active.push_back(&q);
        if (a.sync_start && std::any_of(active.begin(), active.end(), [](const BatchSequence* q) { return q->reading_prompt(); }))
            active.erase(std::remove_if(active.begin(), active.end(), [](const BatchSequence* q) { return !q->reading_prompt(); }), active.end());
        for (const auto* q : active) remaining.push_back(a.generate - static_cast<std::uint32_t>(q->generated.size()));
        SpecPacket pk;
        pk.commits = static_cast<std::uint32_t>(commits.size());
        for (std::size_t c = 0; c < commits.size(); ++c) pk.commit[c][0] = commits[c].first, pk.commit[c][1] = commits[c].second;
        const auto put_commits = [&] {  // split: the commits a round did not carry
            if (!split) return;
            const auto c = spec.take_commits();
            pk.commits = static_cast<std::uint32_t>(c.size());
            for (std::size_t k = 0; k < c.size(); ++k) pk.commit[k][0] = c[k] >> 16U, pk.commit[k][1] = c[k] & 0xFFFFU;
        };
        if (active.empty()) {
            put_commits();
            pk.done = 1U;
            send(pk);
            break;
        }
        const auto t_step = std::chrono::steady_clock::now();
        const auto decode_before = spec.stats().decode_steps;
        const auto& plan = spec.plan(active, remaining);
        put_commits();
        pk.rows = static_cast<std::uint32_t>(plan.rows.size());
        for (std::size_t r = 0; r < plan.rows.size(); ++r) pk.row[r][0] = plan.rows[r].slot, pk.row[r][1] = plan.rows[r].token;
        send(pk);
        const auto& tokens = fwd.step(plan.rows);
        // the rows' records before finish() moves the positions; a window keeps its first rows (filtered below)
        std::vector<std::uint32_t> base(n);
        for (std::uint32_t i = 0; i < n; ++i) base[i] = seqs[i].position;
        std::vector<std::pair<std::uint32_t, RowRecord>> recs(plan.rows.size());
        std::map<std::uint32_t, std::uint32_t> at;
        for (std::size_t r = 0; r < plan.rows.size(); ++r) {
            const auto slot = plan.rows[r].slot;
            const auto [it, fresh] = at.emplace(slot, base[slot]);
            (void)fresh;
            recs[r] = {it->second++, RowRecord{plan.rows[r].token, tokens[r], a.check ? fnv(fwd.row_logits(static_cast<std::uint32_t>(r))) : 0U}};
        }
        commits = spec.finish(active, tokens);
        if (spec.stats().decode_steps != decode_before)  // the whole decode step: plan, drafts, control packet, step, finish
            wall_decode += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_step).count();
        for (std::size_t r = 0; r < plan.rows.size(); ++r) {
            const auto slot = plan.rows[r].slot;
            if (recs[r].first < seqs[slot].position) traces[slot][recs[r].first] = recs[r].second;  // kept rows only
        }
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::uint64_t compared = 0, differ = 0;
    if (!a.spec_only)
        for (std::uint32_t i = 0; i < n; ++i) {
            for (const auto& [pos, rec] : traces[i]) {
                const auto it = alone[i].find(pos);
                if (it == alone[i].end()) continue;  // past the alone run's last fed row
                ++compared;
                if (it->second.fed != rec.fed || it->second.argmax != rec.argmax || it->second.hash != rec.hash) {
                    if (print && differ < 10U)
                        std::printf("  DIFF spec req %u pos %u: alone fed %u argmax %u, spec fed %u argmax %u\n", i, pos, it->second.fed,
                                    it->second.argmax, rec.fed, rec.argmax);
                    ++differ;
                }
            }
            std::vector<std::uint32_t> truth;
            for (const auto& [pos, rec] : alone[i])
                if (pos + 1U >= prompts[i].size()) truth.push_back(rec.argmax);
            truth.resize(std::min<std::size_t>(truth.size(), a.generate));
            std::vector<std::uint32_t> got(seqs[i].generated.begin(), seqs[i].generated.begin() + static_cast<std::ptrdiff_t>(std::min<std::size_t>(seqs[i].generated.size(), a.generate)));
            if (got != truth) {
                ++differ;
                if (print) std::printf("  DIFF spec req %u: %zu tokens differ from its run alone\n", i, got.size());
            }
        }
    std::uint64_t states = 0;  // each request's final attention state (this rank's) against its run alone
    if (!a.spec_only && a.check && (!stage || stage->local()))
        for (std::uint32_t i = 0; i < n; ++i)
            if (fwd.export_state(i).digest(true) != alone_state[i]) ++states;
    if (print) {
        const auto& st = spec.stats();
        const double agg = st.decode_seconds > 0.0 ? static_cast<double>(st.decode_tokens) / st.decode_seconds : 0.0;
        const double steps = static_cast<double>(std::max<std::uint64_t>(st.decode_steps, 1U));
        std::string hist;
        for (std::uint32_t r = 1; r < st.window_rows.size(); ++r)
            if (st.window_rows[r]) hist += " R" + std::to_string(r) + ":" + std::to_string(st.window_rows[r]);
        std::string surv;
        for (std::uint32_t j = 0; j < st.reached.size(); ++j)
            surv += (j ? " " : "") + (st.reached[j] ? std::to_string(static_cast<double>(st.hit[j]) / static_cast<double>(st.reached[j])).substr(0, 5) : std::string("-"));
        const auto& c = spec.controller().costs();
        std::printf("SPEC N %u (%s, vocab %s): decode %.2f tok/s aggregate (%.2f per request), %.2f tok/step, %.2f rows/step, %.2f ms/step "
                    "(draft %.2f ms per drafted step); %llu steps (%llu decode-only, %llu drafted), %.1f s; windows%s; survival %s; "
                    "costs ms plain[1..%u]",
                    n, a.window.c_str(), a.vocab.c_str(), agg, agg / n, static_cast<double>(st.decode_tokens) / steps,
                    static_cast<double>(st.rows) / static_cast<double>(std::max<std::uint64_t>(st.steps, 1U)), st.decode_seconds * 1e3 / steps,
                    st.drafted_steps ? st.draft_seconds * 1e3 / static_cast<double>(st.drafted_steps) : 0.0, static_cast<unsigned long long>(st.steps),
                    static_cast<unsigned long long>(st.decode_steps), static_cast<unsigned long long>(st.drafted_steps), wall, hist.c_str(), surv.c_str(), n);
        for (std::uint32_t k = 1; k <= n; ++k) std::printf(" %.2f", c.plain[k] * 1e3);
        std::printf(", row %.2f, draft %.2f; wall decode %.2f tok/s (%.2f ms/step incl. control)\n", c.row * 1e3, c.draft * 1e3,
                    wall_decode > 0.0 ? static_cast<double>(st.decode_tokens) / wall_decode : 0.0, wall_decode * 1e3 / steps);
        if (!a.spec_only)
            std::printf("GATE spec N %u %s: %llu kept rows compared (%s), %llu differ; final attention states %s\n", n,
                        differ == 0U && states == 0U ? "PASS" : "FAIL", static_cast<unsigned long long>(compared),
                        a.check ? (a.gather ? "tokens + full logits" : "tokens + this rank's logits slice") : "tokens",
                        static_cast<unsigned long long>(differ), !a.check ? "not compared (no --check)" : stage && !stage->local() ? "not compared (split)" : states == 0U ? "== alone" : "DIFFER");
        std::fflush(stdout);
    }
    return differ == 0U && states == 0U;
}

void run_rank(const Args& a, std::uint32_t rank, std::unique_ptr<strix::dist::Transport> transport, bool print,
              std::unique_ptr<strix::dist::DeviceExchange> direct = nullptr) {
    std::unique_ptr<strix::dist::Communicator> comm;
    if (transport)
        comm = std::make_unique<strix::dist::Communicator>(strix::dist::Group{rank, 2U, 1U, 0x64733462ULL}, std::move(transport), std::move(direct), 4U << 20U);
    if (comm) {  // both ranks must run the same product profile
        auto& c = *comm;
        strix::profile::check_peer(strix::deepseek41::pair_defaults::profile_resolution(), [&](std::span<const std::byte> profile_out, std::span<std::byte> profile_in) {
            c.exchange_host(profile_out, profile_in, strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0x70726f66U, 0U});
        });
    }
    std::vector<std::uint32_t> ns;
    for (const auto& s : split(a.n_list, ',')) ns.push_back(static_cast<std::uint32_t>(std::stoul(s)));
    const std::uint32_t n_max = *std::max_element(ns.begin(), ns.end());
    if (n_max == 0U || n_max > BatchForward::configured_row_limit()) throw std::invalid_argument("--n exceeds STRIX_DS41_BATCH_ROWS16 step limit (8, or 16 when enabled)");
    std::vector<std::vector<std::uint32_t>> prompt_list;
    for (const auto& f : split(a.prompts, ',')) prompt_list.push_back(read_ids(f));
    if (prompt_list.empty()) throw std::invalid_argument("--prompts needs at least one ids file");
    std::vector<std::vector<std::uint32_t>> prompts(n_max);
    for (std::uint32_t i = 0; i < n_max; ++i) prompts[i] = prompt_list[i % prompt_list.size()];
    strix::deepseek41::BatchForwardOptions o;
    o.layers = a.layers;
    o.max_positions = a.max_positions;
    o.slots = n_max + (a.fill != 0U ? 1U : 0U);
    o.pool_positions = a.pool;
    o.rank = rank;
    o.world = comm ? 2U : 1U;
    o.communicator = comm.get();
    o.tp_attention = a.tp;
    o.engram_direct_io = a.engram_direct;
    o.slice_logits = a.check && !a.gather;
    o.gather_logits = a.check && a.gather;
    const bool spec = a.mode == "spec";
    // DSpark windows (controller, spec packet, staging) are qualified at 8 rows: not combined with ROWS16
    if (spec && BatchForward::configured_row_limit() > BatchForward::max_rows)
        throw std::invalid_argument("--mode spec is not qualified with STRIX_DS41_BATCH_ROWS16=1 (DSpark windows stay at 8 rows)");
    o.capture_main_hidden = spec && (rank == 0U || a.draft_ranks == 2U);  // the drafters' input rows
    BatchForward fwd(a.model, o);
    std::unique_ptr<strix::ds41::hetero::Stage> stage;
    std::uint64_t hetero_exchanges = 0;
    if (a.hetero()) {
        if (a.mode == "windows" || a.mode == "alone") throw std::invalid_argument("--hetero-* runs --mode batch, both or spec");
        if (a.fill != 0U) throw std::invalid_argument("--hetero-* does not take --fill (use --context-state)");
        strix::ds41::hetero::StageOptions so;
        so.cut = a.hetero_cut;
        so.rank = rank;
        so.world = o.world;
        so.listen = a.hetero_listen;
        so.token_file = a.hetero_token;
        so.slots = n_max;
        so.local = a.hetero_local;
        so.connect_wait = std::chrono::seconds(std::max<unsigned>(a.connect_timeout, 600U));
        auto* c = comm.get();
        stage = std::make_unique<strix::ds41::hetero::Stage>(fwd, so, [c, &hetero_exchanges](std::span<const std::byte> out, std::span<std::byte> in) {
            c->exchange_host(out, in, strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0x4842U, hetero_exchanges++});
        });
        stage->connect();
    }
    std::unique_ptr<strix::ds41::dspark::DeviceDrafter> drafter;
    if (spec && (rank == 0U || (a.draft_ranks == 2U && comm))) {
        const auto w = strix::ds41::dspark::load_weights(a.model, a.draft);
        drafter = std::make_unique<strix::ds41::dspark::DeviceDrafter>(w->weights);
        if (print) std::printf("drafter %s on %s (one drafting context per request), window policy %s, vocabulary %s\n", a.draft.c_str(),
                               a.draft_ranks == 2U ? "both ranks (request i on rank i %% 2)" : "rank 0", a.window.c_str(), a.vocab.c_str());
    }
    const auto save_state = [&](const std::string& phase, std::uint32_t n, std::uint32_t request, std::uint32_t slot) {
        if (a.state_out.empty()) return;
        const auto path = a.state_out + "-" + phase + "-n" + std::to_string(n) + "-q" + std::to_string(request) +
                          "-r" + std::to_string(rank) + ".state";
        strix::deepseek41::save_attention_state(path, fwd.export_state(slot));
    };
    if (print) {
        std::printf("rank %u of %u (%s): layers [0, %u), weights %.2f GiB in %.1f s; per step: fixed %.3f GB + %.1f MB per routed expert per layer;"
                    " state %.1f MiB per request (%u positions), workspace %.1f MiB, %u slots\n",
                    rank, o.world, comm ? comm->transport_name().c_str() : "one node", a.layers, static_cast<double>(fwd.weight_bytes()) / 1073741824.0,
                    fwd.load_seconds(), static_cast<double>(fwd.fixed_bytes_per_step()) / 1e9, static_cast<double>(fwd.expert_bytes()) / 1e6,
                    static_cast<double>(fwd.state_bytes_per_slot()) / 1048576.0, a.max_positions, static_cast<double>(fwd.workspace_bytes()) / 1048576.0,
                    o.slots);
        if (const auto ov = fwd.overlay_description(); !ov.empty()) std::printf("weights overlay: %s\n", ov.c_str());
        if (const auto id = fwd.trellis_identity(); !id.empty())
            std::printf("trellis: %s, %.1f MiB (+ %.1f MiB library workspaces)\n", id.c_str(), static_cast<double>(fwd.trellis_bytes()) / 1048576.0,
                        static_cast<double>(fwd.trellis_library_bytes()) / 1048576.0);
        {  // the footprint the server sizes with, against the device memory in use after the load
            const auto f = BatchForward::footprint(a.model, o);
            const auto [free, total] = fwd.device_memory();
            const double mib = 1048576.0;
            std::printf("footprint: weights %.1f + state %.1f x %u + workspace %.1f + pool %.1f + trellis %.1f = %.1f MiB; "
                        "device in use %.1f MiB of %.1f (free %.1f)\n",
                        static_cast<double>(f.weights) / mib, static_cast<double>(f.state_per_slot) / mib, o.slots,
                        static_cast<double>(f.workspace) / mib, static_cast<double>(f.pool) / mib, static_cast<double>(f.trellis) / mib,
                        static_cast<double>(f.weights + f.state_per_slot * o.slots + f.workspace + f.pool + f.trellis) / mib,
                        static_cast<double>(total - free) / mib, static_cast<double>(total) / mib, static_cast<double>(free) / mib);
        }
        if (a.pool != 0U)
            std::printf("KV pool: %u positions over the slots, %.1f MiB of arenas (largest free reservation %llu)\n", a.pool,
                        static_cast<double>(fwd.pool_bytes()) / 1048576.0, static_cast<unsigned long long>(fwd.pool_free_positions()));
        std::fflush(stdout);
    }
    const std::uint32_t template_slot = n_max;
    if (a.fill != 0U) {
        const auto filler = read_ids(a.fill_ids);
        const auto f0 = std::chrono::steady_clock::now();
        claim(fwd, a, template_slot, std::uint64_t{a.fill} + 8U);
        std::vector<BatchRow> window{{template_slot, filler[0]}};
        fwd.step(window);  // the first row alone (no compressed entry before it)
        for (std::uint32_t i = 1; i < a.fill;) {
            window.clear();
            while (window.size() < BatchForward::max_rows && i < a.fill) window.push_back({template_slot, filler[i++ % filler.size()]});
            fwd.step(window);
        }
        if (print) {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - f0).count();
            std::printf("filled %u context tokens in %.1f s (%.1f tok/s in steps of 8 rows)\n", a.fill, s, a.fill / s);
        }
    }
    bool all_pass = true;
    for (const auto n : ns) {
        std::vector<Trace> alone(n);
        std::vector<std::uint64_t> alone_state(n);
        if (a.mode == "alone" || a.mode == "both" || a.mode == "windows" || (spec && !a.spec_only)) {
            for (std::uint32_t i = 0; i < n; ++i) {
                const std::vector<std::vector<std::uint32_t>> one{prompts[i]};
                auto r = stage && a.hetero_alone_split ? run_hetero(fwd, *stage, a, one, 1U, 1U, false) : run(fwd, a, one, 1U, template_slot, a.fill != 0U, false);
                if (a.check) alone_state[i] = r.state_hashes.front();
                save_state("alone", n, i, 0U);
                alone[i] = std::move(r.traces[0]);
                if (print && a.mode == "alone")
                    for (const auto& [pos, rec] : alone[i])
                        std::printf("alone req %u pos %u fed %u argmax %u logits_fnv %016llx\n", i, pos, rec.fed, rec.argmax,
                                    static_cast<unsigned long long>(rec.hash));
            }
        }
        if (a.mode == "alone") continue;
        if (spec) {
            if (a.fill != 0U || a.context) throw std::invalid_argument("--mode spec starts every request at position 0 (no --fill / --context-state)");
            const std::uint32_t spec_groups = a.hetero_groups != 0U ? std::min(a.hetero_groups, n) : n <= 2U ? 1U : 2U;
            all_pass = spec_gate(fwd, drafter.get(), a, rank, comm != nullptr, prompts, n, alone, alone_state, print, stage.get(), spec_groups) && all_pass;
            continue;
        }
        if (a.mode == "windows") {
            if (a.fill != 0U) throw std::invalid_argument("--mode windows starts every request at position 0 (no --fill)");
            all_pass = windows_gate(fwd, a, prompts, n, alone, print) && all_pass;
            continue;
        }
        if (a.aba != 0U) {
            namespace ah = strix::ds41::attn::hip;
            const bool f = a.aba_flags.find("filter") != std::string::npos, d = a.aba_flags.find("direct") != std::string::npos,
                       h = a.aba_flags.find("seghpw") != std::string::npos, fo = a.aba_flags.find("fold") != std::string::npos,
                       bf = a.aba_flags.find("f16") != std::string::npos, lv = a.aba_flags.find("live") != std::string::npos,
                       et = a.aba_flags.find("engtp") != std::string::npos, tb = a.aba_flags.find("touch") != std::string::npos,
                       rv = a.aba_flags.find("rowviews") != std::string::npos, ni = a.aba_flags.find("normilp") != std::string::npos,
                       ro = a.aba_flags.find("routes") != std::string::npos, cf = a.aba_flags.find("cfast") != std::string::npos,
                       sz = a.aba_flags.find("swz") != std::string::npos;
            std::vector<std::uint64_t> prints;
            for (std::uint32_t arm = 0; arm < 2U * a.aba; ++arm) {
                const int on = static_cast<int>(arm % 2U);
                ah::set_index_candidate_filter(f ? on : -1);
                ah::set_index_direct(d ? on : -1);
                ah::set_attention_segment_hpw(h ? on : -1);
                ah::set_attention_window_fold(fo ? on : -1);
                ah::set_attention_tile_f16(bf ? on : -1);
                ah::set_attention_live_chunks(lv ? on : -1);
                strix::ds41::set_engram_tp(et ? on : -1);  // STRIX_DS41_ENGRAM_TP
                strix::ds41::set_touch_batch(tb ? on : -1);
                strix::ds41::set_trellis_row_views(rv ? on : -1);
                strix::ds41::set_norm_ilp(ni ? on : -1);
                strix::ds41::set_r1fuse_routes(ro ? on : -1);
                ah::set_compress_fast(cf ? on : -1);
                ah::set_attention_lds_swizzle(sz ? on : -1);
                const auto rr = run(fwd, a, prompts, n, template_slot, a.fill != 0U, false);
                std::uint64_t fp = 14695981039346656037ULL;
                const auto mix = [&](std::uint64_t v) { fp = (fp ^ v) * 1099511628211ULL; };
                for (const auto& t : rr.traces)
                    for (const auto& [pos, rec] : t) { mix(pos); mix(rec.fed); mix(rec.argmax); mix(rec.hash); }
                for (const auto v : rr.state_hashes) mix(v);
                prints.push_back(fp);
                const double steps = static_cast<double>(std::max<std::uint64_t>(rr.steady_steps, 1U));
                if (print) {
                    std::printf("aba arm %u flags %s %s N %u decode %.3f ms/step wall %.3f ms/step steps %llu fp %016llx\n", arm, a.aba_flags.c_str(),
                                on ? "on" : "off", n, rr.steady_seconds * 1e3 / steps, rr.steady_wall * 1e3 / steps,
                                static_cast<unsigned long long>(rr.steady_steps), static_cast<unsigned long long>(fp));
                    std::fflush(stdout);
                }
            }
            ah::set_index_candidate_filter(-1);
            ah::set_index_direct(-1);
            ah::set_attention_segment_hpw(-1);
            ah::set_attention_window_fold(-1);
            ah::set_attention_tile_f16(-1);
            ah::set_attention_live_chunks(-1);
            strix::ds41::set_engram_tp(-1);
            strix::ds41::set_touch_batch(-1);
            strix::ds41::set_trellis_row_views(-1);
            strix::ds41::set_norm_ilp(-1);
            strix::ds41::set_r1fuse_routes(-1);
            ah::set_compress_fast(-1);
            ah::set_attention_lds_swizzle(-1);
            const bool same = std::all_of(prints.begin(), prints.end(), [&](std::uint64_t v) { return v == prints[0]; });
            if (print) std::printf("aba exact %s (N %u, %zu arms; %s, final attention states)\n", same ? "IDENTICAL" : "DIFFER", n, prints.size(),
                                   a.check ? "fed / argmax / logits hash per row" : "fed / argmax per row");
            all_pass = all_pass && same;
            continue;
        }
        const std::uint32_t groups = a.hetero_groups != 0U ? std::min(a.hetero_groups, n) : n <= 2U ? 1U : 2U;
        const auto r = stage ? run_hetero(fwd, *stage, a, prompts, n, groups, print) : run(fwd, a, prompts, n, template_slot, a.fill != 0U, print);
        for (std::uint32_t i = 0; i < n; ++i) save_state("batch", n, i, i);
        if (print && stage) {
            const auto& st = r.stage;
            const double steps = static_cast<double>(std::max<std::uint64_t>(st.steps, 1U));
            std::printf("N %u hetero cut %u (%s, %u groups): %llu steps, %.1f s; decode window %.2f tok/s (%llu rows in %llu steps, %.2f s); per step:"
                        " FRONT wait %.2f ms, forward to rank 1 %.2f ms, TAIL %.2f ms; %.1f KB per row from the FRONT, %.1f KB/step to rank 1\n",
                        n, stage->cut(), stage->local() ? "local FRONT" : "FRONT process", r.groups, static_cast<unsigned long long>(r.steps), r.seconds,
                        r.window_seconds > 0.0 ? static_cast<double>(r.window_rows) / r.window_seconds : 0.0,
                        static_cast<unsigned long long>(r.window_rows), static_cast<unsigned long long>(r.window_steps), r.window_seconds,
                        st.wait_seconds * 1e3 / steps, st.forward_seconds * 1e3 / steps, st.tail_seconds * 1e3 / steps,
                        st.rows ? static_cast<double>(st.bytes_in) / 1e3 / static_cast<double>(st.rows) : 0.0, static_cast<double>(st.bytes_forwarded) / 1e3 / steps);
            for (std::uint32_t i = 0; i < n; ++i) {
                std::string toks;
                std::uint32_t shown = 0;
                for (const auto& [pos, rec] : r.traces[i])
                    if (pos + 1U >= prompts[i].size() && shown++ < 12U) toks += (toks.empty() ? "" : ",") + std::to_string(rec.argmax);
                std::printf("  req %u: prompt %zu tokens, generated %s...\n", i, prompts[i].size(), toks.c_str());
            }
        } else if (print) {
            const double step_ms = r.steady_steps ? r.steady_seconds * 1e3 / static_cast<double>(r.steady_steps) : 0.0;
            const double agg = r.steady_seconds > 0.0 ? static_cast<double>(r.steady_rows) / r.steady_seconds : 0.0;
            const double steps = static_cast<double>(std::max<std::uint64_t>(r.steady_steps, 1U));
            const double fixed = static_cast<double>(fwd.fixed_bytes_per_step()), experts = r.steady_expert_bytes / steps;
            std::printf("N %u: %llu steps (%llu decode-only), %.1f s; decode %.2f ms/step, aggregate %.2f tok/s, per request %.2f tok/s;"
                        " exchange %.2f ms/step, engram %.2f ms/step; union %.2f experts/layer (this rank, max %.2f rows/expert);"
                        " bytes/step %.3f GB (fixed %.3f + experts %.3f), %.0f GB/s\n",
                        n, static_cast<unsigned long long>(r.steps), static_cast<unsigned long long>(r.steady_steps), r.seconds, step_ms, agg,
                        agg / n, r.steady_exchange * 1e3 / steps, r.steady_engram * 1e3 / steps, r.steady_union / steps, r.steady_max_rows / steps,
                        (fixed + experts) / 1e9, fixed / 1e9, experts / 1e9, step_ms > 0.0 ? (fixed + experts) / (step_ms * 1e-3) / 1e9 : 0.0);
            std::printf("N %u wall decode %.2f tok/s (%.2f ms/step end to end)\n", n,
                        r.steady_wall > 0.0 ? static_cast<double>(r.steady_rows) / r.steady_wall : 0.0, r.steady_wall * 1e3 / steps);
            for (std::uint32_t i = 0; i < n; ++i) {
                std::string toks;
                std::uint32_t shown = 0;
                for (const auto& [pos, rec] : r.traces[i])
                    if (pos + 1U >= prompts[i].size() + (a.fill != 0U ? a.fill : 0U) && shown++ < 12U) toks += (toks.empty() ? "" : ",") + std::to_string(rec.argmax);
                std::printf("  req %u: prompt %zu tokens, generated %s...\n", i, prompts[i].size(), toks.c_str());
            }
        }
        if (a.mode == "both") {
            std::uint64_t compared = 0, differ = 0;
            const bool states = !stage || stage->local();  // a split run with a FRONT process holds no layer before the cut
            if (a.check && states && alone_state != r.state_hashes) {
                ++differ;
                if (print) std::printf("  DIFF final attention state digests\n");
            }
            for (std::uint32_t i = 0; i < n; ++i) {
                if (alone[i].size() != r.traces[i].size()) {
                    ++differ;
                    if (print) std::printf("  req %u: %zu rows alone, %zu batched\n", i, alone[i].size(), r.traces[i].size());
                }
                for (const auto& [pos, rec] : r.traces[i]) {
                    const auto it = alone[i].find(pos);
                    ++compared;
                    if (it == alone[i].end() || it->second.fed != rec.fed || it->second.argmax != rec.argmax || it->second.hash != rec.hash) {
                        if (print && differ < 10U)
                            std::printf("  DIFF req %u pos %u: alone fed %u argmax %u fnv %016llx, batched fed %u argmax %u fnv %016llx\n", i, pos,
                                        it == alone[i].end() ? 0U : it->second.fed, it == alone[i].end() ? 0U : it->second.argmax,
                                        static_cast<unsigned long long>(it == alone[i].end() ? 0ULL : it->second.hash), rec.fed, rec.argmax,
                                        static_cast<unsigned long long>(rec.hash));
                        ++differ;
                    }
                }
            }
            if (print)
                std::printf("GATE N %u %s: %llu rows compared (%s), %llu differ\n", n, differ == 0U ? "PASS" : "FAIL",
                            static_cast<unsigned long long>(compared), a.check ? (a.gather ? "tokens + full logits" : "tokens + this rank's logits slice") : "tokens",
                            static_cast<unsigned long long>(differ));
            all_pass = all_pass && differ == 0U;
        }
        std::fflush(stdout);
    }
    if (print && (a.mode == "both" || a.mode == "windows" || (spec && !a.spec_only))) std::printf("GATE %s\n", all_pass ? "PASS" : "FAIL");
    if (comm && print) {
        const auto& s = comm->stats();
        std::printf("exchanges %llu, %.1f MB sent, %.2f s in exchange (stage %.2f, wire %.2f, check %.2f)\n", static_cast<unsigned long long>(s.exchanges),
                    static_cast<double>(s.bytes_sent) / 1e6, s.seconds, s.stage, s.wire, s.check);
    }
    if (stage) stage->close();
    if (!all_pass) throw std::runtime_error("gate failed");
}

}  // namespace

int main(int argc, char** argv) try {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    const strix::deepseek41::pair_defaults::Effective production_defaults(strix::deepseek41::pair_defaults::App::batch, argc, argv, strix::deepseek41::pair_defaults::this_build());  // before any option or env read
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
            return std::string(argv[++i]);
        };
        const auto u32 = [&] { return static_cast<std::uint32_t>(std::stoul(next())); };
        if (s == "--rank") a.rank = u32();
        else if (s == "--transport") a.transport = next();
        else if (s == "--layers") a.layers = u32();
        else if (s == "--prompts") a.prompts = next();
        else if (s == "--n") a.n_list = next();
        else if (s == "--generate") a.generate = u32();
        else if (s == "--mode") a.mode = next();
        else if (s == "--prompt-rows") a.prompt_rows = u32();
        else if (s == "--max-positions") a.max_positions = u32();
        else if (s == "--fill") a.fill = u32();
        else if (s == "--fill-ids") a.fill_ids = next();
        else if (s == "--tp-attention") a.tp = true;
        else if (s == "--no-tp-attention") a.tp = false;
        else if (s == "--engram-direct") a.engram_direct = true;
        else if (s == "--engram-mapped") a.engram_direct = false;
        else if (s == "--check") a.check = true;
        else if (s == "--aba") a.aba = u32();
        else if (s == "--aba-flags") a.aba_flags = next();
        else if (s == "--gather") a.gather = true;
        else if (s == "--print-rows") a.print_rows = true;
        else if (s == "--sync-start") a.sync_start = true;
        else if (s == "--pool") a.pool = u32();
        else if (s == "--context-state") a.context_state = next();
        else if (s == "--state-out") a.state_out = next();
        else if (s == "--connect-timeout") a.connect_timeout = static_cast<unsigned>(std::stoul(next()));
        else if (s == "--draft") a.draft = next();
        else if (s == "--vocab") a.vocab = next();
        else if (s == "--window") a.window = next();
        else if (s == "--spec-only") a.spec_only = true;
        else if (s == "--draft-ranks") a.draft_ranks = u32();
        else if (s == "--hetero-listen") a.hetero_listen = next();
        else if (s == "--hetero-token-file") a.hetero_token = next();
        else if (s == "--hetero-cut") a.hetero_cut = u32();
        else if (s == "--hetero-groups") a.hetero_groups = u32();
        else if (s == "--hetero-local") a.hetero_local = true;
        else if (s == "--hetero-alone") {
            const auto v = next();
            if (v != "whole" && v != "split") throw std::invalid_argument("--hetero-alone is whole or split");
            a.hetero_alone_split = v == "split";
        }
        else if (a.model.empty()) a.model = s;
        else throw std::invalid_argument("unknown argument " + s);
    }
    if (a.model.empty() || a.prompts.empty()) throw std::invalid_argument("usage: strix-deepseek41-batch MODEL.gguf --prompts a.ids,b.ids --n 1,2,4,8 ...");
    if (a.mode != "batch" && a.mode != "alone" && a.mode != "both" && a.mode != "windows" && a.mode != "spec")
        throw std::invalid_argument("--mode is batch, alone, both, windows or spec");
    if (a.mode == "spec" && a.draft.empty()) throw std::invalid_argument("--mode spec needs --draft DSPARK.gguf");
    if (a.prompt_rows == 0U || a.prompt_rows > BatchForward::max_rows) throw std::invalid_argument("--prompt-rows must be in 1..8");
    if (a.fill != 0U && a.fill_ids.empty()) throw std::invalid_argument("--fill needs --fill-ids");
    if (!a.hetero_listen.empty() && a.hetero_token.empty()) throw std::invalid_argument("--hetero-listen needs --hetero-token-file");
    if (!a.hetero_listen.empty() && a.hetero_local) throw std::invalid_argument("--hetero-listen and --hetero-local exclude each other");
    strix::deepseek41::AttentionState context;
    if (!a.context_state.empty()) {
        if (a.fill != 0U) throw std::invalid_argument("--context-state and --fill exclude each other");
        context = strix::deepseek41::load_attention_state(a.context_state);
        a.context = &context;
        std::printf("context state %s: %u positions, %.1f MB, digest %016llx\n", a.context_state.c_str(), context.position,
                    static_cast<double>(context.bytes()) / 1e6, static_cast<unsigned long long>(context.digest(true)));
    }
    if (a.transport == "single") {
        run_rank(a, 0U, nullptr, true);
        return 0;
    }
    if (a.transport == "emulate") {
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
