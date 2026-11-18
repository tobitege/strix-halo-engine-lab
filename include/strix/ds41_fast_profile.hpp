#pragma once
// ONE resolver of the DeepSeek V4.1 FAST prefill levers, read by the device prefill (server and CLI alike) and by the
// prefix-cache identity, so both always agree.
//   STRIX_DS41_FAST_PREFILL=1 (default off) = profile ds41-fast-pf-v1:
//     attention  STRIX_DS41_FAST_ATTN=2   ds41-attn-bf16flash-v1 (BF16 one-pass sparse attention, shared flash core)
//     MoE        STRIX_DS41_FAST_MOE=1    ds41-moe-relaxed-v1
//     dense      STRIX_DS41_FAST_DENSE=1  ds41-dense-f16-v1 (shared policy dense-q8a8-f16-v1)
// A lever variable that is set (any value, "0" = off) wins over the profile. Changes logits; see docs/toggles.md.
// Bundle rules (explicit member wins, explicit OFF beats the bundle) are the shared ones: strix/fast_profile.hpp.
#include "strix/fast_profile.hpp"

#include <cstdlib>

namespace strix::ds41::fast_profile {
inline constexpr strix::fast_profile::Member kMembers[] = {
    {"STRIX_DS41_FAST_ATTN", "2", "ds41-attn-bf16flash-v1", "1 2"},
    {"STRIX_DS41_FAST_MOE", "1", "ds41-moe-relaxed-v1"},
    {"STRIX_DS41_FAST_DENSE", "1", "ds41-dense-f16-v1"},
};
inline constexpr strix::fast_profile::Bundle kBundle{"STRIX_DS41_FAST_PREFILL", "ds41-fast-pf-v1", kMembers};
inline const char* attention() { return strix::fast_profile::lever(kBundle, kMembers[0]); }  // "1" f16wmma-v1, "2" bf16flash-v1
inline bool attention_on() { const char* v = attention(); return (v[0] == '1' || v[0] == '2') && v[1] == 0; }
inline bool attention_flash() { const char* v = attention(); return v[0] == '2' && v[1] == 0; }
inline bool moe_on() { const char* v = strix::fast_profile::lever(kBundle, kMembers[1]); return v[0] == '1' && v[1] == 0; }
inline bool dense_on() { const char* v = strix::fast_profile::lever(kBundle, kMembers[2]); return v[0] == '1' && v[1] == 0; }
// Explicit FAST lever, not a bundle member until its qualification receipt: the Engram KV projection on the shared
// FP16 WMMA core (STRIX_DS41_FAST_ENGRAM=1, policy ds41-engram-f16wmma-v1; src/hip_ds41_layer.hip). Changes logits.
inline constexpr strix::fast_profile::Member kEngram{"STRIX_DS41_FAST_ENGRAM", "1", "ds41-engram-f16wmma-v1"};
inline bool engram_on() { return strix::fast_profile::switch_on(strix::envcfg::text(kEngram.env)); }
// Explicit FAST lever, not a bundle member until its qualification receipt: the prefill shared expert on the FAST dense
// core (STRIX_DS41_FAST_SHARED=1, policy ds41-shared-f16-v1; src/hip_deepseek41_prefill.hip Op::shared_expert). Changes logits.
inline constexpr strix::fast_profile::Member kShared{"STRIX_DS41_FAST_SHARED", "1", "ds41-shared-f16-v1"};
inline bool shared_on() { return strix::fast_profile::switch_on(strix::envcfg::text(kShared.env)); }
// Explicit FAST lever (with the FAST MoE lever): the relaxed routed-expert kernels in their FP16-operand form
// (STRIX_DS41_FAST_MOE_F16=1, policy ds41-moe-f16wmma-v1; src/hip_ds41_prefill.hip). Changes logits.
inline constexpr strix::fast_profile::Member kMoeF16{"STRIX_DS41_FAST_MOE_F16", "1", "ds41-moe-f16wmma-v1"};
inline bool moe_f16_on() { return strix::fast_profile::switch_on(strix::envcfg::text(kMoeF16.env)); }
// Preserve decimal-prefix parsing and unsigned wrap, including zero and negative values.
// Also accepts the canonical unsigned value exported by the pair resolver.
inline unsigned dense_min_rows_value(const char* raw) {
    return raw ? static_cast<unsigned>(std::strtoull(raw, nullptr, 10)) : 128U;
}
inline unsigned dense_min_rows() {
    return dense_min_rows_value(strix::envcfg::text("STRIX_DS41_FAST_DENSE_MIN_ROWS"));
}
}  // namespace strix::ds41::fast_profile
