#pragma once
// CUDA Qwen serving: the session reservation both entry points resolve (ONE RESOLVER). The cohort server's admission and
// the serialized path (--http-slots 1, the CLI-shaped generate()) call request_positions() with the same rule, so a
// request gets the same session capacity either way. Host-only (tests/cuda_qwen_serve_policy_tests.cpp).
#include "strix/serve/session_reservation.hpp"

#include <cstdint>
#include <string>

namespace cuda_qwen_serve {

// Rows one session may write past its last committed position: one verify window of 1 + up to 7 drafts
// (qwen_prefill_driver.cu kMaxStepRows = 8), doubled so a draft chain never meets the capacity before max_tokens ends.
inline constexpr std::uint64_t kSessionSlack = 16;
// Capacities are multiples of 256 positions (8.1 MB of rows): requests of similar size reuse an idle session as is.
inline constexpr std::uint64_t kDefaultGranule = 256;

struct SessionPolicy {
    std::uint64_t max_positions = 0;
    std::uint64_t granule = kDefaultGranule;  // --session-granule / STRIX_QRTX_SESSION_GRANULE
    bool per_request = true;                  // --session-reserve request|max (STRIX_QRTX_SESSION_RESERVE): max = every session at max_positions
};

inline strix::serve::ReservationRule reservation_rule(const SessionPolicy& p) {
    strix::serve::ReservationRule r;
    r.max_positions = p.max_positions;
    r.granule = p.granule ? p.granule : 1U;
    r.slack = kSessionSlack;
    return r;
}

// The capacity of a request's session (prompt tokens, max_tokens). `max` mode: the context cap (the old fixed sizing).
inline std::uint64_t request_positions(const SessionPolicy& p, std::uint64_t prompt, std::uint64_t max_tokens) {
    if (!p.per_request) return p.max_positions;
    return strix::serve::reservation_positions(reservation_rule(p), prompt, max_tokens);
}

// Session 0's capacity at load (the engine builds it before any request): one granule in request mode.
inline std::uint64_t initial_positions(const SessionPolicy& p) {
    if (!p.per_request) return p.max_positions;
    return p.granule < p.max_positions ? (p.granule ? p.granule : 1U) : p.max_positions;
}

inline std::string describe(const SessionPolicy& p) {
    return p.per_request ? "per request (prompt + max_tokens + " + std::to_string(kSessionSlack) + ", granule " + std::to_string(p.granule) +
                               ", cap " + std::to_string(p.max_positions) + ")"
                         : "fixed at max_positions " + std::to_string(p.max_positions);
}

}  // namespace cuda_qwen_serve
