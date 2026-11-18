// Cohort window packing for multi-session decode on the CUDA Qwen engine (host only, no CUDA).
//
// Every decode step gives every live session at least one row (a session is never dropped from a step). The sessions are
// split, in order, into ceil(n / budget) windows of near-equal session counts (budget = the verify window's row cap, 8).
// Inside a window the spare rows (budget - sessions) go to the sessions' wanted extra rows (MTP drafts) round-robin, one
// row per session per round, lowest index first: max-min fair, decode first. A session's rows are a prefix of what it
// wanted (its pending token, then its first drafts), so shrinking only shortens a draft chain.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace strix::cuda::qwen_cohort {

inline constexpr std::uint32_t kWindowRows = 8;       // the default window budget (rows)
inline constexpr std::uint32_t kMaxWindowRows = 16;   // the engine's window cap (9..16: prefill-shaped kernels)
inline constexpr std::uint32_t kMaxWindowSessions = 8;  // sessions per window (engine segments)

struct Placement {
    std::uint32_t index = 0;  // the session's index in the input
    std::uint32_t rows = 0;   // 1 .. want[index]
};
struct Window {
    std::vector<Placement> entries;  // session-major row order
    std::uint32_t rows = 0;
};

// Session counts of the windows of one step: max(ceil(n / budget), ceil(n / 8)) windows (every session >= 1 row, at most 8
// sessions per window), sizes differing by at most one, larger first.
inline std::vector<std::uint32_t> window_sizes(std::uint32_t sessions, std::uint32_t budget = kWindowRows) {
    std::vector<std::uint32_t> sizes;
    if (sessions == 0 || budget == 0) return sizes;
    const std::uint32_t k = std::max((sessions + budget - 1U) / budget, (sessions + kMaxWindowSessions - 1U) / kMaxWindowSessions);
    for (std::uint32_t w = 0; w < k; ++w) sizes.push_back(sessions / k + (w < sessions % k ? 1U : 0U));
    return sizes;
}

// want[i] >= 1: rows session i would use (1 + its drafts); 0 is read as 1.
inline std::vector<Window> pack(const std::vector<std::uint32_t>& want, std::uint32_t budget = kWindowRows) {
    std::vector<Window> out;
    if (budget == 0) return out;
    std::uint32_t next = 0;
    for (const auto size : window_sizes(static_cast<std::uint32_t>(want.size()), budget)) {
        Window w;
        for (std::uint32_t j = 0; j < size; ++j) w.entries.push_back({next + j, 1U});
        std::uint32_t spare = budget - size;
        for (bool grew = true; spare > 0 && grew;) {
            grew = false;
            for (auto& e : w.entries) {
                if (spare == 0) break;
                if (e.rows < std::max<std::uint32_t>(want[e.index], 1U)) { ++e.rows; --spare; grew = true; }
            }
        }
        for (const auto& e : w.entries) w.rows += e.rows;
        next += size;
        out.push_back(std::move(w));
    }
    return out;
}

// Draft depth worth computing for a session whose window holds `window_sessions` sessions: "fair" = its round-robin
// share of the spare rows, ceil((budget - m) / m) (m = 1: budget - 1, the single-session depth); "max" = everything a
// lone drafter could use, budget - m. Never above `depth`.
enum class DepthPolicy { fair, max };
inline DepthPolicy depth_policy_of(const std::string& name) { return name == "max" ? DepthPolicy::max : DepthPolicy::fair; }
inline std::uint32_t draft_depth_cap(std::uint32_t window_sessions, std::uint32_t depth, DepthPolicy policy = DepthPolicy::fair,
                                     std::uint32_t budget = kWindowRows) {
    if (window_sessions == 0 || window_sessions >= budget) return 0;
    const std::uint32_t spare = budget - window_sessions;
    const std::uint32_t cap = policy == DepthPolicy::max ? spare : (spare + window_sessions - 1U) / window_sessions;
    return std::min(cap, depth);
}

// The window size (sessions) of the session at `index` among `sessions` (pack's split).
inline std::uint32_t window_of(std::uint32_t index, std::uint32_t sessions, std::uint32_t budget = kWindowRows) {
    std::uint32_t first = 0;
    for (const auto size : window_sizes(sessions, budget)) {
        if (index < first + size) return size;
        first += size;
    }
    return 0;
}

// Window budget of one step when a budget above 8 rows is allowed (STRIX_QRTX_WINDOW_ROWS > 8): the marginal rule. A
// window of 9..16 rows costs a different kernel form than one of <= 8 rows, so the wider budget is taken only when its
// expected tokens per millisecond beat the 8-row packing by `min_gain`. Expected tokens of a session with r rows:
// 1 + p + p^2 + ... + p^(r-1) (p: acceptance per draft position). Cost of a window of r rows: decode-shaped
// a_ms + b_ms * r up to 8 rows; wide_ms for 9..16 rows (padded to 16). Defaults from c12a (bench-w8 / bench-w16):
// a 10.5, b 2.81 (13.3 ms at 1 row .. 33.0 at 8), wide 62 (59-69 ms measured), p 0.65 (C8 w16: 0.56 accepted of
// 0.86 drafts per session). With these the rule never widens (C8: 8 / 33 = 0.24 vs 13.2 / 62 = 0.21 tokens/ms), and at
// C2 / C4 the 8-row windows already hold the drafts that pay.
struct WideModel {
    double a_ms = 10.5, b_ms = 2.81, wide_ms = 62.0, accept = 0.65, min_gain = 0.02;
};
// Cohort costs by the backend's decode row cap. Cap 16 (9..16-row decode forms, one-pass dense): a 16-row cohort
// window costs 50 ms against the model's 33 ms 8-row window, fitted from in-process forced budget 16 vs 8 receipts
// (2 ABAB pairs, ids equal, 90 % power cap): C8 edit +29.6 %, json +28.3 %, prose +3.4 / -0.1 % fit 46-48 ms; C4 edit
// +23.8 %, json +3.4 / +6.5 %, prose -26 / -31 % fit 48-58 ms. A wider budget must win by 5 %: with the live p, edit and
// json widen at C4 and C8, prose (p ~0.5) stays at 8; with no live p yet (the 0.65 prior) only C8 widens, which is
// what lets its sessions verify drafts and learn their p (at budget 8, C8 sessions run one row and never verify).
inline WideModel wide_measured(unsigned decode_rows_cap) {
    WideModel m;
    if (decode_rows_cap >= 16U) { m.wide_ms = 50.0; m.min_gain = 0.05; }
    return m;
}
inline double expected_tokens(std::uint32_t rows, double p) {
    double t = 0, q = 1;
    for (std::uint32_t k = 0; k < rows; ++k) { t += q; q *= p; }
    return t;
}
inline double window_ms(std::uint32_t rows, const WideModel& m) {
    return rows <= kWindowRows ? m.a_ms + m.b_ms * rows : m.wide_ms;
}
// tokens per ms of a step of `sessions` sessions (each wanting 1 + depth rows, depth capped as the step caps it)
inline double step_rate(std::uint32_t sessions, std::uint32_t depth, DepthPolicy policy, std::uint32_t budget, const WideModel& m) {
    std::vector<std::uint32_t> want(sessions);
    for (std::uint32_t i = 0; i < sessions; ++i) want[i] = 1U + (depth ? draft_depth_cap(window_of(i, sessions, budget), depth, policy, budget) : 0U);
    double tokens = 0, ms = 0;
    for (const auto& w : pack(want, budget)) {
        for (const auto& e : w.entries) tokens += expected_tokens(e.rows, m.accept);
        ms += window_ms(w.rows, m);
    }
    return ms > 0 ? tokens / ms : 0;
}
inline std::uint32_t choose_budget(std::uint32_t sessions, std::uint32_t depth, DepthPolicy policy, std::uint32_t budget,
                                   const WideModel& m = {}) {
    if (budget <= kWindowRows || sessions == 0 || depth == 0) return std::min(budget, kWindowRows);
    const double narrow = step_rate(sessions, depth, policy, kWindowRows, m);
    const double wide = step_rate(sessions, depth, policy, budget, m);
    return wide > narrow * (1.0 + m.min_gain) ? budget : kWindowRows;
}

// Wide copy window costs of one session (decision rule in qwen_cohort.hpp, copy_wide_switch).
// Fixed sm_120 calibration, not runtime measurements or device selection. Recalibrate manually
// after kernel changes; qualify other devices separately before using this model there.
struct CopyWideModel {
    double narrow_a_ms = 8.55, narrow_b_ms = 1.1, wide_ms = 62.5;
};
// Measured sm_120 decision table by the backend's decode row cap (strix/cuda/core/decode_rows.hpp), qwen_decode_bench on
// the RTX PRO 6000: cap 8 -> 9..16 rows on the tiled prefill forms, 62.5 ms flat; cap 16 -> 8-row decode
// pieces, 27.6 ms at 9 .. 29.2 ms at 16 (windows of 9..16 rows are padded to 16). Served edit C1
// copy_rows 16 vs 8: -29 % at cap 8, +12.7 % at cap 16 (3 ABAB pairs each, ids equal).
inline CopyWideModel copy_wide_measured(unsigned decode_rows_cap) {
    CopyWideModel m;
    m.wide_ms = decode_rows_cap >= 16U ? 29.2 : 62.5;
    return m;
}
inline bool copy_wide_pays(std::uint32_t rows, const CopyWideModel& m = {}) {
    if (rows <= kWindowRows || !(m.wide_ms > 0.0)) return false;
    const double narrow = static_cast<double>(kWindowRows) / (m.narrow_a_ms + m.narrow_b_ms * kWindowRows);
    return static_cast<double>(rows) / m.wide_ms > narrow;
}

}  // namespace strix::cuda::qwen_cohort
