// One multi-session decode step on the CUDA Qwen engine (host code over the public Prefill API): MTP draft chains per
// session (depth capped by the session's window share), cohort packing (cuda_qwen_cohort_plan.hpp), then for every
// window verify_cohort + greedy acceptance + commit_cohort. Every committed token and logits row equals the session
// decoding alone (the engine's cohort windows are batch-invariant; drafts only choose what the verify checks).
// Used by strix-cuda-qwen-serve (slot scheduler), qwen_cohort_e2e (gate + bench).
#pragma once

#include "strix/env_flags.hpp"
#include "strix/cuda/core/decode_rows.hpp"
#include "strix/cuda/qwen_cohort_plan.hpp"
#include "strix/cuda/qwen_prefill.hpp"
#include "strix/mtp_context_draft.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::cuda::qwen_cohort {

namespace qp = strix::cuda::qwen_prefill;

// A decoding session: `pending` is the next input token (the last verified one), at position `position` (= the
// session's length).
// Context copy/n-gram drafter of one stream: the shared host drafter (strix/mtp_context_draft.hpp, the HIP engine's
// --mtp-ngram-draft=ctx|copy). copy-first: a gated context span supplies the window's rows with no MTP predictor
// forward (the commit writes the MTP teacher rows of every committed position from the target's hidden rows, so the
// predictor cache is the same as after a drafted window); otherwise the span extends or replaces the confident MTP
// prefix. Every row still passes the greedy verify: the committed tokens are the target's whatever the proposal.
enum class ContextMode { off, ctx, copy, first };
inline ContextMode context_mode_of(const char* v) {
    const std::string s = v ? v : "";
    if (s == "ctx") return ContextMode::ctx;
    if (s == "copy") return ContextMode::copy;
    if (s == "first" || s == "1" || s == "on") return ContextMode::first;
    return ContextMode::off;
}
inline const char* context_mode_name(ContextMode m) {
    return m == ContextMode::first ? "first" : m == ContextMode::copy ? "copy" : m == ContextMode::ctx ? "ctx" : "off";
}
// STRIX_QRTX_CONTEXT_DRAFT=off|first|ctx|copy, one resolution for every entry point. Default first (measured, 3 ABAB
// pairs, ids equal): edit C1 +13.6 %, C4 +3.5 %, C8 +2.1 %, json +-0.3 %; prose -0.5..-2.8 % = the drafter's host time
// (~50 us per stream per cycle) on requests that never copy. copy/ctx also verify extension rows: a rejected row costs
// ~1.2 ms of routed-expert bytes (json C1 -2.9 %), so they stay opt-in.
inline ContextMode context_mode_default() {
    static const ContextMode m = [] {
        const char* v = strix::envcfg::text("STRIX_QRTX_CONTEXT_DRAFT");
        return v ? context_mode_of(v) : ContextMode::first;
    }();
    return m;
}
struct ContextDraft {
    strix::MtpContextDraft drafter;
    double mtp_expected = -1.0;    // accepted drafts per MTP (not copy-first) step, EMA 0.75/0.25 (as the HIP runtime)
    bool copy_first = false;       // this step's rows came from the context, no predictor forward
    bool deliver_extension = true;  // first: context rows only as copy-first windows; extensions are scored, never verified
    bool copy_edit = false;         // copy / first: long-anchor spans (copy-first, wide copy windows)
    std::uint64_t copy_first_steps = 0, extended_steps = 0, context_rows = 0, context_accepted = 0;
    std::uint64_t wide_steps = 0, wide_rows = 0, wide_accepted = 0;  // copy windows of 9..16 rows (subset of copy_first_steps)
    double host_us = 0.0;           // host time in the drafter (proposals + learning)
    // live per-position draft acceptance of this session (both drafters): EMA (0.9) of accepted draft rows and of their
    // trials (accepted + 1 when a verified draft was rejected); p = accept_s / accept_t. Feeds the marginal rule.
    double accept_s = 0.0, accept_t = 0.0;
    static MtpContextDraftConfig config(ContextMode m) {
        MtpContextDraftConfig c;
        c.copy_edit = m == ContextMode::copy || m == ContextMode::first;
        return c;
    }
    ContextMode mode = ContextMode::off;
    explicit ContextDraft(ContextMode m) : drafter(config(m)), deliver_extension(m != ContextMode::first), copy_edit(config(m).copy_edit), mode(m) {}
};

struct Stream {
    qp::SessionId session = 0;
    std::uint32_t pending = 0;
    std::uint64_t position = 0;
    std::uint32_t depth_max = 0;  // per-request draft cap (0: the step's depth)
    ContextDraft* context = nullptr;  // borrowed; null = MTP drafts only
    std::uint32_t copy_rows = 0;      // per-request wide copy window rows (0: the step's; request field "copy_rows")
    std::uint32_t window_rows = 0;    // per-request window budget (0: the step's; request field "window_rows"; the
                                      // step takes the widest one of its streams, packing only)
};

inline std::uint32_t window_rows_default() {
    static const std::uint32_t rows = [] {
        const char* v = std::getenv("STRIX_QRTX_WINDOW_ROWS");
        // default 16 with the backend's 9..16-row decode forms (core::decode_rows_max() 16), else 8: above 8 rows the
        // marginal rule (wide_measured) widens on the live acceptance; STRIX_QRTX_WINDOW_ROWS=8 keeps 8-row windows
        // (request field "window_rows" per request)
        const long def = strix::cuda::core::decode_rows_max() >= 16U ? 16L : static_cast<long>(kWindowRows);
        const long n = v ? std::strtol(v, nullptr, 10) : def;
        return static_cast<std::uint32_t>(std::clamp<long>(n, 1, static_cast<long>(kMaxWindowRows)));
    }();
    return rows;
}

struct StepOptions {
    bool mtp = false;              // draft chains (needs the sidecar)
    std::uint32_t depth = 7;       // draft depth of a lone session
    float min_p = 0.7F;
    DepthPolicy policy = DepthPolicy::fair;
    // window rows: STRIX_QRTX_WINDOW_ROWS (default 8; up to 16, windows of 9..16 rows run the prefill-shaped families);
    // clamped to 1..kMaxWindowRows
    std::uint32_t budget = window_rows_default();
    std::uint64_t max_positions = 0;      // 0: the engine's Config::max_positions
    bool want_logits = false;      // gates: the FP32 rows of the kept tokens
    // Draft chains of several sessions batched by step (Prefill::draft_cohort, same tokens as per-session chains):
    // -1 = default on unless STRIX_QRTX_DRAFT_BATCH=0; 0 = per session; 1 = batched.
    int draft_batch = -1;
    // budget above 8: -1 = the marginal rule unless STRIX_QRTX_WIDE_RULE=0; 0 = the budget as given (gates); 1 = the rule
    int wide_rule = -1;
    // wide copy windows of a lone session: rows (0 = copy_rows_default(); <= 8 = off), minimum learned copy acceptance
    // (< 0 = copy_wide_accept_default())
    std::uint32_t copy_rows = 0;
    double copy_wide_accept = -1.0;
};
// The marginal rule's model (cuda_qwen_cohort_plan.hpp, choose_budget): STRIX_QRTX_WIDE_MS (cost of a 9..16-row window),
// STRIX_QRTX_WIDE_ACCEPT (acceptance per draft position), STRIX_QRTX_WIDE_A_MS / _B_MS (<= 8-row cost a + b * rows),
// STRIX_QRTX_WIDE_GAIN (minimum gain); STRIX_QRTX_WIDE_RULE=0 takes the budget as given (the c10-c12 behaviour).
inline const WideModel& wide_model() {
    static const WideModel m = [] {
        WideModel w = wide_measured(strix::cuda::core::decode_rows_max());
        const auto rd = [](const char* name, double& v) { if (const char* s = std::getenv(name); s && s[0]) v = std::strtod(s, nullptr); };
        rd("STRIX_QRTX_WIDE_MS", w.wide_ms);
        rd("STRIX_QRTX_WIDE_ACCEPT", w.accept);
        rd("STRIX_QRTX_WIDE_A_MS", w.a_ms);
        rd("STRIX_QRTX_WIDE_B_MS", w.b_ms);
        rd("STRIX_QRTX_WIDE_GAIN", w.min_gain);
        return w;
    }();
    return m;
}
// Wide copy windows: a session alone in its step whose context holds a long unanimous anchor (the shared drafter's
// propose_wide: suffix >= copy_anchor, every earlier occurrence continuing alike, no known edit inside, learned copy
// acceptance cells >= copy_wide_accept) verifies copy_rows - 1 context rows in ONE window of copy_rows (9..16) rows on
// the prefill-shaped families, and commits at the first rejected row (the engine's exact rollback). Wide serial
// equality gate: qwen_decode_e2e STRIX_QWEN_TMAX=16 (G >= 137) plus MTP cache validation. Otherwise the
// narrow copy-first / MTP window as before. STRIX_QRTX_COPY_ROWS (<= 8 = off), STRIX_QRTX_COPY_WIDE_ACCEPT.
inline std::uint32_t copy_rows_default() {
    static const std::uint32_t rows = [] {
        const char* v = strix::envcfg::text("STRIX_QRTX_COPY_ROWS");
        const long n = v ? std::strtol(v, nullptr, 10) : 16L;
        return static_cast<std::uint32_t>(std::clamp<long>(n, 1, static_cast<long>(kMaxWindowRows)));
    }();
    return rows;
}
inline double copy_wide_accept_default() {
    static const double p = [] {
        const char* v = strix::envcfg::text("STRIX_QRTX_COPY_WIDE_ACCEPT");
        const double x = v ? std::strtod(v, nullptr) : 0.9;
        return std::isfinite(x) ? std::clamp(x, 0.0, 1.0) : 0.9;
    }();
    return p;
}
// The wide copy window's decision (auto): it turns on only when a fully accepted wide window delivers more tokens per
// ms than a fully accepted 8-row window (the narrow copy-first window it replaces), from the measured window costs of
// one session. sm_120 RTX PRO 6000 (qwen_decode_bench): 9.65 ms at 1 row, +1.1 ms per row to 17.2 ms at 8
// rows, 62.5 ms flat for 9..16 rows (prefill-shaped families) -> 16 / 62.5 = 0.26 vs 8 / 17.2 = 0.46 tokens/ms: off
// (served edit C1 -29 %, 3 ABAB pairs). Costs require manual recalibration after kernel changes; other devices need
// separate qualification. A calibrated wide cost under ~34 ms enables auto (STRIX_QRTX_COPY_WIDE_MS). With the
// backend's 9..16-row decode pieces (core::decode_rows_max() 16, the default) the calibrated sm_120 wide cost is 29.2 ms
// -> 16 / 29.2 = 0.55 tokens/ms: on at 16 rows (copy_wide_measured, by cap).
// STRIX_QRTX_COPY_WIDE=1 forces it on, 0 off; the request field copy_rows
// forces the rows of one request (measurement).
inline const CopyWideModel& copy_wide_model() {
    static const CopyWideModel m = [] {
        CopyWideModel w = copy_wide_measured(strix::cuda::core::decode_rows_max());
        w.wide_ms = strix::envcfg::number<double>("STRIX_QRTX_COPY_WIDE_MS", w.wide_ms, std::numeric_limits<double>::min());
        return w;
    }();
    return m;
}
// -1 auto (copy_wide_pays), 0 off, 1 on
inline int copy_wide_switch() {
    static const int v = [] {
        const char* s = strix::envcfg::text("STRIX_QRTX_COPY_WIDE");
        if (!s || !s[0] || std::string(s) == "auto") return -1;
        return s[0] == '0' ? 0 : 1;
    }();
    return v;
}
inline bool wide_rule_default() {
    static const bool on = strix::envcfg::flag("STRIX_QRTX_WIDE_RULE", true);
    return on;
}
inline bool draft_batch_default() {
    static const bool on = strix::envcfg::flag("STRIX_QRTX_DRAFT_BATCH", true);
    return on;
}

// Mean live per-position acceptance of the step's sessions (ContextDraft accept_s / accept_t); -1 when a session has no
// context drafter or fewer than 4 trials of history, or when STRIX_QRTX_WIDE_ACCEPT fixes the model's value.
template <class Streams> double live_accept(const Streams& streams) {
    static const bool fixed = strix::envcfg::text("STRIX_QRTX_WIDE_ACCEPT") != nullptr;
    if (fixed || streams.empty()) return -1.0;
    double sum = 0.0;
    for (const auto& st : streams) {
        if (!st.context || st.context->accept_t < 4.0) return -1.0;
        sum += st.context->accept_s / st.context->accept_t;
    }
    return sum / static_cast<double>(streams.size());
}
struct StreamStep {
    std::vector<std::uint32_t> tokens;   // the verified tokens of this step (argmax of the kept rows), >= 1
    std::vector<float> logits;           // [tokens][vocab] when StepOptions::want_logits
    std::uint32_t rows = 0;              // window rows this session used (1 + drafts verified)
    std::uint32_t drafted = 0;           // predictor work, before context replacement and packing (depth controller)
    std::uint32_t mtp_proposed = 0, mtp_accepted = 0;          // predictor rows verified / accepted
    std::uint32_t context_proposed = 0, context_accepted = 0;  // context rows verified / accepted
    std::uint32_t window_sessions = 0;   // sessions sharing its window
    std::uint32_t window_rows = 0;       // rows of its window
    // The session's rows failed (non-finite logits, or a routed-MoE failure that it raised alone): no tokens; its window
    // was undone, so the session state is still consistent, but the request should end with `error`.
    bool failed = false;
    std::string error;
    bool retried = false;                // its shared window failed and it was re-run alone (isolation)
};

struct StepTiming {
    double draft_ms = 0, verify_ms = 0;
    std::uint32_t windows = 0, rows = 0;
};

// Every stream must have room for at least one row (position < min(max_positions, its session's capacity)) and its position must equal the
// engine session's length. Streams are advanced in place. Failure isolation: when a window with several sessions comes
// back with a failed entry (a routed-MoE failure word taints the whole window), every entry is undone (keep 0, exact) and
// each session is re-run alone in its own window; only a session that also fails alone is marked failed. So a healthy
// session's tokens never depend on another session's rows.
inline std::vector<StreamStep> step(qp::Prefill& p, std::vector<Stream>& streams, const StepOptions& opt, StepTiming* timing = nullptr) {
    using Clock = std::chrono::steady_clock;
    StepOptions o = opt;
    {
        std::uint32_t req = 0;
        for (const auto& s : streams) req = std::max(req, s.window_rows);
        if (req) o.budget = req;
    }
    o.budget = std::clamp<std::uint32_t>(o.budget, 1U, kMaxWindowRows);
    if (o.max_positions == 0) o.max_positions = p.max_positions();
    const auto n = static_cast<std::uint32_t>(streams.size());
    // above 8 rows only when the marginal rule says the wider windows pay (packing only: every session's tokens are
    // the same either way)
    if (o.budget > kWindowRows && (o.wide_rule < 0 ? wide_rule_default() : o.wide_rule != 0)) {
        WideModel m = wide_model();
        if (const double p = live_accept(streams); p >= 0.0) m.accept = p;
        o.budget = choose_budget(n, o.mtp ? o.depth : 0U, o.policy, o.budget, m);
    }
    std::vector<StreamStep> out(n);
    std::vector<std::vector<std::uint32_t>> rows(n);
    std::vector<std::uint32_t> want(n);
    std::vector<std::uint32_t> mtp_prefix(n, 0);
    for (const auto& s : streams)
        if (p.length(s.session) != s.position)
            throw std::logic_error("cohort step: stream position " + std::to_string(s.position) + " != session length " +
                                   std::to_string(p.length(s.session)));
    const auto t0 = Clock::now();
    const bool batch = o.draft_batch < 0 ? draft_batch_default() : o.draft_batch != 0;
    std::vector<qp::DraftRequest> requests(n);
    std::uint32_t chains = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        auto& s = streams[i];
        rows[i].push_back(s.pending);
        const auto m = window_of(i, n, o.budget);
        out[i].window_sessions = m;
        std::uint32_t depth = o.mtp ? draft_depth_cap(m, o.depth, o.policy, o.budget) : 0U;
        if (s.depth_max) depth = std::min(depth, s.depth_max);
        requests[i] = {s.session, s.pending, depth, o.min_p};
    }
    // copy-first: a gated context span replaces this stream's MTP chain (no predictor forward)
    std::vector<std::uint32_t> room(n, 0);
    std::vector<std::vector<std::uint32_t>> copy(n);
    const std::uint32_t copy_rows = std::min(o.copy_rows ? o.copy_rows : copy_rows_default(), kMaxWindowRows);
    const double wide_accept = o.copy_wide_accept < 0.0 ? copy_wide_accept_default() : std::min(o.copy_wide_accept, 1.0);
    std::uint32_t pack_budget = o.budget;  // a wide copy window widens the (single) window of this step
    for (std::uint32_t i = 0; i < n; ++i) {
        room[i] = requests[i].depth;
        auto* c = streams[i].context;
        if (!c) continue;
        c->copy_first = false;
        std::vector<double> acceptance;
        const auto h0 = Clock::now();
        if (room[i] == 0) {
            // Observe pending exactly once even before any draft capacity becomes available.
            // The shared zero-room cycle initializes history without building proposals.
            (void)c->drafter.begin_cycle(streams[i].pending, {}, 0, 0, acceptance, o.min_p);
        } else {
            bool wide = false;
            const std::uint32_t wide_rows = streams[i].copy_rows ? std::min(streams[i].copy_rows, kMaxWindowRows) : copy_rows;
            const int sw = copy_wide_switch();
            const bool wide_on = streams[i].copy_rows ? streams[i].copy_rows > kWindowRows
                : sw < 0 ? copy_wide_pays(wide_rows, copy_wide_model()) : sw == 1;
            if (n == 1 && wide_on && wide_rows > kWindowRows && c->copy_edit) {
                copy[i] = c->drafter.propose_wide(streams[i].pending, wide_rows - 1U, wide_accept);
                wide = !copy[i].empty();
            }
            if (wide) {
                pack_budget = std::max<std::uint32_t>(pack_budget, static_cast<std::uint32_t>(copy[i].size()) + 1U);
            } else {
                copy[i] = c->drafter.begin_copy_cycle(streams[i].pending, room[i], acceptance, o.min_p, c->mtp_expected);
                if (copy[i].size() > room[i]) copy[i].resize(room[i]);
            }
        }
        if (!copy[i].empty()) { c->copy_first = true; requests[i].depth = 0; }
        c->host_us += std::chrono::duration<double, std::micro>(Clock::now() - h0).count();
    }
    for (std::uint32_t i = 0; i < n; ++i) chains += requests[i].depth > 0;
    // drafts: batched by step across sessions (<= 8 chains per call), or one chain per session
    std::vector<qp::Draft> drafts(n);
    if (batch && chains > 1) {
        for (std::uint32_t first = 0; first < n; first += kWindowRows) {
            const auto last = std::min<std::uint32_t>(n, first + kWindowRows);
            const auto part = p.draft_cohort(std::vector<qp::DraftRequest>(requests.begin() + first, requests.begin() + last));
            for (std::uint32_t i = first; i < last; ++i) drafts[i] = part[i - first];
        }
    } else {
        for (std::uint32_t i = 0; i < n; ++i)
            if (requests[i].depth > 0) drafts[i] = p.draft(requests[i].session, requests[i].pending, requests[i].depth, o.min_p);
    }
    for (std::uint32_t i = 0; i < n; ++i) {
        auto& s = streams[i];
        auto& d = drafts[i];
        auto* c = s.context;
        out[i].drafted = static_cast<std::uint32_t>(d.tokens.size());
        mtp_prefix[i] = out[i].drafted;
        if (c && c->copy_first) {
            d.tokens = copy[i];
            ++c->copy_first_steps;
            c->context_rows += copy[i].size();
        } else if (c && room[i] > 0) {
            // context rows after any confident MTP prefix (also none), or a longer span replacing it
            std::size_t confident = 0;
            double kept_expected = 0.0, run = 1.0;
            while (confident < d.tokens.size() && confident < d.probability.size() && d.probability[confident] >= o.min_p) {
                run *= d.probability[confident];
                kept_expected += run;
                ++confident;
            }
            std::vector<double> acceptance;
            const auto h0 = Clock::now();
            auto ext = c->drafter.begin_cycle(s.pending, d.tokens, confident, room[i], acceptance, o.min_p, kept_expected);
            if (!c->deliver_extension) { c->drafter.discard_selection(); ext.clear(); }  // learning only
            c->host_us += std::chrono::duration<double, std::micro>(Clock::now() - h0).count();
            if (!ext.empty()) {
                const std::size_t kept = std::min<std::size_t>(c->drafter.kept_drafts(), d.tokens.size());
                mtp_prefix[i] = static_cast<std::uint32_t>(kept);
                d.tokens.resize(kept);
                for (std::size_t r = 0; r < ext.size() && d.tokens.size() < room[i]; ++r) d.tokens.push_back(ext[r]);
                ++c->extended_steps;
                c->context_rows += d.tokens.size() - kept;
            }
        }
        rows[i].insert(rows[i].end(), d.tokens.begin(), d.tokens.end());
        const std::uint64_t cap = std::min<std::uint64_t>(o.max_positions, p.capacity(s.session));  // the session's reservation
        const std::uint64_t position_room = cap > s.position ? cap - s.position : 0U;
        want[i] = static_cast<std::uint32_t>(std::min<std::uint64_t>(rows[i].size(), std::max<std::uint64_t>(position_room, 1U)));
    }
    const auto t1 = Clock::now();
    std::uint32_t windows_run = 0, rows_run = 0;
    // one window: verify, greedy acceptance, commit; false when some entry failed (then every entry was undone)
    const auto run_window = [&](const std::vector<Placement>& entries, bool isolate) {
        std::vector<qp::CohortRows> cohort;
        std::uint32_t total = 0;
        for (const auto& e : entries) {
            cohort.push_back({streams[e.index].session, rows[e.index].data(), e.rows, streams[e.index].position});
            total += e.rows;
        }
        ++windows_run;
        rows_run += total;
        const auto r = p.verify_cohort(cohort, o.want_logits);
        for (const auto& e : entries) {
            auto* c = streams[e.index].context;
            if (c && c->copy_first && e.rows > kWindowRows) {
                ++c->wide_steps;
                c->wide_rows += e.rows - 1U;
            }
        }
        bool all_ok = true;
        for (const auto& x : r) all_ok = all_ok && x.ok;
        if (!all_ok) {
            p.commit_cohort(std::vector<std::uint32_t>(entries.size(), 0U));  // undo every entry
            if (!isolate) {
                for (std::size_t j = 0; j < entries.size(); ++j) {
                    auto& st = out[entries[j].index];
                    st.failed = true;
                    st.error = r[j].ok ? "decode step failed" : r[j].error;
                }
            }
            return false;
        }
        std::vector<std::uint32_t> keep(entries.size());
        for (std::size_t j = 0; j < entries.size(); ++j) {
            const auto& e = entries[j];
            const auto& row = rows[e.index];
            std::uint32_t k = 1;
            while (k < e.rows && row[k] == r[j].argmax[k - 1]) ++k;
            keep[j] = k;
            auto& st = out[e.index];
            st.rows = e.rows;
            st.mtp_proposed = std::min(mtp_prefix[e.index], e.rows - 1U);
            st.context_proposed = e.rows - 1U - st.mtp_proposed;
            st.mtp_accepted = std::min(st.mtp_proposed, k - 1U);
            st.context_accepted = k - 1U - st.mtp_accepted;
            st.window_rows = total;
            st.tokens.assign(r[j].argmax.begin(), r[j].argmax.begin() + k);
            if (o.want_logits) st.logits.assign(r[j].logits.begin(), r[j].logits.begin() + static_cast<std::ptrdiff_t>(r[j].logits.size() / e.rows * k));
        }
        p.commit_cohort(keep);
        for (std::size_t j = 0; j < entries.size(); ++j) {
            auto& s = streams[entries[j].index];
            s.position += keep[j];
            s.pending = out[entries[j].index].tokens.back();
        }
        return true;
    };
    for (const auto& w : pack(want, pack_budget)) {
        if (run_window(w.entries, w.entries.size() > 1)) continue;
        if (w.entries.size() == 1) continue;  // already marked failed
        for (const auto& e : w.entries) {     // isolation: each session alone
            out[e.index].retried = true;
            (void)run_window({e}, false);
        }
    }
    // the context drafters learn from the verified tokens (accepted drafts + the target's next token)
    for (std::uint32_t i = 0; i < n; ++i) {
        auto* c = streams[i].context;
        if (!c || out[i].failed || out[i].tokens.empty()) continue;
        const std::size_t verified = out[i].rows > 0 ? out[i].rows - 1U : 0U, accepted = out[i].tokens.size() - 1U;
        const auto h0 = Clock::now();
        c->drafter.finish_cycle(out[i].tokens, verified, accepted);
        c->host_us += std::chrono::duration<double, std::micro>(Clock::now() - h0).count();
        if (verified > 0) {
            c->accept_s = 0.9 * c->accept_s + static_cast<double>(accepted);
            c->accept_t = 0.9 * c->accept_t + static_cast<double>(accepted + (accepted < verified ? 1U : 0U));
        }
        if (c->copy_first) {
            c->context_accepted += accepted;
            if (out[i].rows > kWindowRows) c->wide_accepted += accepted;
        }
        else if (room[i] > 0) c->mtp_expected = c->mtp_expected < 0.0 ? static_cast<double>(out[i].mtp_accepted) : 0.75 * c->mtp_expected + 0.25 * static_cast<double>(out[i].mtp_accepted);
    }
    if (timing) {
        timing->draft_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        timing->verify_ms += std::chrono::duration<double, std::milli>(Clock::now() - t1).count();
        timing->windows += windows_run;
        timing->rows += rows_run;
    }
    return out;
}

}  // namespace strix::cuda::qwen_cohort
