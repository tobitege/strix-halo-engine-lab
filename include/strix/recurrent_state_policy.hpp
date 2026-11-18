#pragma once
// Recurrent state storage policy: ONE resolver for the CLI, the server and every backend (HIP, CUDA, host).
// STRIX_RECURRENT_STATE = f32 (default, incumbent bits) | f16 | bf16 (FAST, bit-changing: see
// strix/recurrent_state_type.hpp). Experimental and opt-in; it is not part of any profile until its quality
// gate passes. Unknown values are rejected (no silent fallback). The resolved value is recorded for
// --print-effective-config through envcfg.
#include "strix/env_flags.hpp"
#include "strix/recurrent_state_type.hpp"

#include <stdexcept>
#include <string>

namespace strix {

inline constexpr const char* kRecurrentStateEnv = "STRIX_RECURRENT_STATE";

struct RecurrentStatePolicy {
    RecurrentStateType type = RecurrentStateType::f32;
    bool fast = false;              // bit-changing relative to the f32 contract
    const char* origin = "default"; // "default" | "explicit"
};

[[nodiscard]] inline RecurrentStatePolicy recurrent_state_policy_from(const char* raw) {
    RecurrentStatePolicy policy;
    if (raw == nullptr || raw[0] == '\0') return policy;
    const std::string value(raw);
    if (value == "f32") policy.type = RecurrentStateType::f32;
    else if (value == "f16") policy.type = RecurrentStateType::f16;
    else if (value == "bf16") policy.type = RecurrentStateType::bf16;
    else detail::env_reject(kRecurrentStateEnv, raw, "expected one of f32, f16, bf16");
    policy.fast = policy.type != RecurrentStateType::f32;
    policy.origin = "explicit";
    return policy;
}

// Process-wide resolution (read once, recorded).
[[nodiscard]] inline RecurrentStatePolicy recurrent_state_policy() {
    static const RecurrentStatePolicy policy = [] {
        const char* raw = envcfg::text(kRecurrentStateEnv);
        const auto resolved = recurrent_state_policy_from(raw);
        envcfg::record(kRecurrentStateEnv, raw, recurrent_state_type_name(resolved.type));
        return resolved;
    }();
    return policy;
}

// Admission of a route that has no typed form yet: a 16-bit state may only reach kernels that implement the
// codec (strix/attn_core/recurrent_state.hpp). Returns an empty string when admitted, else the observable reason.
[[nodiscard]] inline std::string recurrent_state_route_refusal(RecurrentStateType type, bool route_typed,
    const char* route) {
    if (type == RecurrentStateType::f32 || route_typed) return {};
    return std::string(kRecurrentStateEnv) + "=" + recurrent_state_type_name(type) + " refused: route " + route +
        " reads/writes the recurrent state as f32 only (no typed form yet)";
}

// Fail closed at application/protocol boundaries whose typed-state contract is
// not qualified. Storage support in a kernel alone does not admit composition.
inline void require_f32_recurrent_state(RecurrentStateType type, const char* route) {
    if (type != RecurrentStateType::f32)
        throw std::invalid_argument(std::string(kRecurrentStateEnv) + "=" + recurrent_state_type_name(type) +
            " refused: " + route + " requires f32 recurrent state");
}

}  // namespace strix
