#pragma once

#include "strix/env_flags.hpp"
#include "strix/execution_runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string>
#include <string_view>

namespace strix {

// A cohort window carries the rows of several independent sessions. Each session's rows are one
// segment: consecutive positions over that session's own state and scratch arena. While a
// KnownWindowCohortScope is alive on this thread, the stateless wide launchers (routed expert
// union, HC mixers, router, LM head) admit such rows: every row still passes its own complete R1
// admission against its own resources, and only the single-session cross-row checks (one state
// binding, one scratch arena with a fixed row stride, consecutive positions across all rows) are
// replaced by per-segment ones. Outside a scope every launcher keeps its single-session contract.
inline bool& known_window_cohort_flag() noexcept {
    thread_local bool active = false;
    return active;
}
[[nodiscard]] inline bool known_window_cohort_active() noexcept { return known_window_cohort_flag(); }

class KnownWindowCohortScope {
public:
    KnownWindowCohortScope() noexcept : previous_(known_window_cohort_flag()) { known_window_cohort_flag() = true; }
    ~KnownWindowCohortScope() { known_window_cohort_flag() = previous_; }
    KnownWindowCohortScope(const KnownWindowCohortScope&) = delete;
    KnownWindowCohortScope& operator=(const KnownWindowCohortScope&) = delete;
private:
    bool previous_;
};

// Row r starts a new session segment: inside a cohort scope, a row whose session state binding
// differs from the previous row's. Row 0 always starts one.
[[nodiscard]] inline bool known_window_segment_start(std::span<const ExecutionDispatchContext> rows, std::size_t r) noexcept {
    if (r == 0) return true;
    return known_window_cohort_active() && rows[r].session_state_memory != rows[r - 1U].session_state_memory;
}
// The single-session position rule (row r at row 0's position + r), per segment in a cohort.
[[nodiscard]] inline bool known_window_row_position_follows(std::span<const ExecutionDispatchContext> rows, std::size_t r) noexcept {
    if (known_window_segment_start(rows, r)) return true;
    return rows[r - 1U].position != UINT64_MAX && rows[r].position == rows[r - 1U].position + 1U;
}

// In-process A/B of cohort features (measurement only): with STRIX_COHORT_AB=1 every odd cohort cycle
// (known_window_cohort_cycle(), advanced once per cycle by the cohort scheduler) runs without the features
// listed in STRIX_COHORT_AB_OFF (comma list, e.g. "gdnseg,combine,draft"); timings are kept per parity.
inline std::uint64_t& known_window_cohort_cycle() noexcept {
    thread_local std::uint64_t cycle = 0;
    return cycle;
}
[[nodiscard]] inline bool known_window_cohort_ab() noexcept {
    static const bool on = strix::envcfg::flag("STRIX_COHORT_AB", false);
    return on;
}
[[nodiscard]] inline bool known_window_cohort_feature_off(std::string_view feature) {
    if (!known_window_cohort_ab() || !(known_window_cohort_cycle() & 1U)) return false;
    static const std::string list = [] { const char* v = std::getenv("STRIX_COHORT_AB_OFF"); return std::string(v ? v : ""); }();
    std::size_t at = 0;
    while (at <= list.size()) {
        const auto end = std::min(list.find(',', at), list.size());
        if (std::string_view(list).substr(at, end - at) == feature) return true;
        at = end + 1U;
    }
    return false;
}

}  // namespace strix
