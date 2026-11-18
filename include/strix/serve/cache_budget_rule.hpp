#pragma once
// Automatic prefix-state host budget, shared by every engine's late cache fit (Qwen fit_cache_late today;
// DS41 / GLM / MiniMax servers bind the same rule). Inputs are the post-residency available host bytes A and
// the checked reserve R the engine already computes (>= 4 GiB headroom + unpaid stream/request owners +
// staging + fixed margin). Neither rule ever exceeds A - R, so the reserves are respected by construction.
//
//   legacy (default): min(4 GiB, A/16, A - R)                        -- the incumbent rule, unchanged
//   share:            max(legacy, min(cap, share_pct% of (A - R)))    -- a share of what is left after the
//                     streams, never below the incumbent minimum; cap 0 = no ceiling beyond A - R.
//
// The budget only changes how many exact states are retained, never the arithmetic of a request (bitwise).
// Default OFF until a node receipt (16 chats x 17K, follow-ups at once) shows the share keeps the states.
#include "strix/env_flags.hpp"
#include "strix/lever.hpp"

#include <algorithm>
#include <cstdint>

namespace strix::serve {

inline constexpr lever::Decl cache_budget_share_lever{
    "prefix-cache-budget-share", "STRIX_PREFIX_CACHE_BUDGET_SHARE", lever::Exactness::bitwise, false, {}, {},
    "pending: node receipt 16x17K follow-ups, budget share vs min(4 GiB,A/16,A-R) (qcprot RESULT)"};

enum class CacheBudgetRule { legacy, share };
struct CacheBudgetPolicy {
    CacheBudgetRule rule = CacheBudgetRule::legacy;
    std::uint32_t share_pct = 50;  // 1..100 of (A - R)
    std::uint64_t cap_bytes = 0;   // caps the share component, not the legacy floor; 0: no cap
    bool operator==(const CacheBudgetPolicy&) const = default;
};
[[nodiscard]] constexpr const char* cache_budget_rule_name(CacheBudgetRule rule) {
    return rule == CacheBudgetRule::share ? "share" : "legacy";
}

// Callers have checked available >= reserve. The result is NOT aligned; engines align as before.
[[nodiscard]] constexpr std::uint64_t automatic_cache_budget(const CacheBudgetPolicy& p,
    std::uint64_t available, std::uint64_t reserve) {
    constexpr auto gib = std::uint64_t{1} << 30;
    if (available < reserve) return 0;
    const auto remainder = available - reserve;
    const auto legacy = std::min({4 * gib, available / 16, remainder});
    if (p.rule != CacheBudgetRule::share) return legacy;
    const auto pct = std::clamp<std::uint64_t>(p.share_pct, 1, 100);
    // remainder/100*pct + (remainder%100)*pct/100: exact floor without overflow.
    auto share = remainder / 100 * pct + remainder % 100 * pct / 100;
    if (p.cap_bytes) share = std::min(share, p.cap_bytes);
    return std::max(legacy, share);
}

// One resolution for CLI and server: STRIX_PREFIX_CACHE_BUDGET_SHARE (lever, "1" selects the share rule),
// STRIX_PREFIX_CACHE_BUDGET_SHARE_PCT (default 50), STRIX_PREFIX_CACHE_BUDGET_CAP_GIB (default 0 = none).
[[nodiscard]] inline CacheBudgetPolicy cache_budget_policy_from_env() {
    CacheBudgetPolicy p;
    if (!lever::select(cache_budget_share_lever, {}).on) return p;
    p.rule = CacheBudgetRule::share;
    p.share_pct = static_cast<std::uint32_t>(std::clamp(envcfg::integer("STRIX_PREFIX_CACHE_BUDGET_SHARE_PCT", 50), 1, 100));
    const auto cap_gib = std::max(envcfg::integer("STRIX_PREFIX_CACHE_BUDGET_CAP_GIB", 0), 0);
    p.cap_bytes = static_cast<std::uint64_t>(cap_gib) << 30;
    return p;
}

} // namespace strix::serve
