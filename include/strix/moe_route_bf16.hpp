#pragma once
#include "strix/env_flags.hpp"
#include <cstdint>
#include <cstdlib>
#include <string_view>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

namespace strix {
// Integer RNE also preserves infinities and never turns a NaN into an infinity.
#if defined(__HIPCC__) || defined(__CUDACC__)
__host__ __device__
#endif
inline constexpr std::uint16_t moe_route_bf16_bits(std::uint32_t bits) {
    if ((bits & 0x7fffffffU) > 0x7f800000U)
        return static_cast<std::uint16_t>((bits >> 16U) | 0x40U);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}
inline bool moe_route_bf16_environment_requested(const char* request, const char* baseline, const char* defaults) {
    return request && request[0] && request[0] != '0' &&
        !(baseline && std::string_view(baseline) == "1") && !(defaults && defaults[0] == '0');
}
inline bool moe_route_i8_enabled() {
    static const bool enabled=strix::envcfg::flag("STRIX_PREFILL_MOE_ROUTE_I8", false, strix::EnvFlagMatch::exact);
    return enabled;
}
// Codes and exponents are hot; the cold, fixed BF16 fallback plane fits the reserved FP32 row.
struct MoeRouteI8Layout { std::uint64_t values, groups_per_slot, exponents, fallback, bytes; };
#if defined(__HIPCC__)
__host__ __device__
#endif
inline constexpr MoeRouteI8Layout moe_route_i8_layout(unsigned hidden,unsigned top_k) {
    const std::uint64_t n=static_cast<std::uint64_t>(hidden)*top_k,g=(static_cast<std::uint64_t>(hidden)+31U)/32U;
    const auto exponents=(n+1U)&~std::uint64_t{1},fallback=exponents+2U*g*top_k;
    return {n,g,exponents,fallback,fallback+2U*n};
}
inline constexpr std::int16_t moe_route_i8_fallback=-32768;
// RNE(127.5) is 128: the upper endpoint is strict. Zero groups canonically use E=0.
// Keeping 2^E and its reciprocal normal avoids overflow/denormal scale intermediates.
#if defined(__HIPCC__)
__host__ __device__
#endif
inline constexpr std::int16_t moe_route_i8_exponent(std::uint32_t magnitude) {
    if(!magnitude)return 0;
    if(magnitude>=0x7f800000U)return moe_route_i8_fallback;
    const int e=static_cast<int>(magnitude>>23U)-133+((magnitude&0x7fffffU)>=0x7f0000U?1:0);
    return e < -126 || e > 120 ? moe_route_i8_fallback : static_cast<std::int16_t>(e);
}
// Storage identity is separate from the prefill/decode profile-pair identity. The shared
// entry binds the env-only draft for receipts without a CLI/profile-table change.
inline constexpr std::string_view moe_prefill_storage_name(bool bf16,bool i8=moe_route_i8_enabled()) {
    return bf16 ? (i8 ? "fp16-v1+moeroute16+moeroutei8" : "fp16-v1+moeroute16") : "fp16-v1";
}
inline constexpr std::uint64_t moe_prefill_storage_identity(bool bf16,bool i8=moe_route_i8_enabled()) {
    std::uint64_t h=0xcbf29ce484222325ULL;
    for(const char c:moe_prefill_storage_name(bf16,i8)) {h^=static_cast<unsigned char>(c);h*=0x100000001b3ULL;}
    return h;
}
} // namespace strix
