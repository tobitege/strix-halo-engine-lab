#pragma once
// Exchange row fold: a row operation on the two partials of one small all-reduce exchange, computed by the channel's
// progress thread (CPU) between the peer payload's arrival and the publication of the RX flag. The consumer kernel then
// reads one folded slot instead of the two partial slots. Engines only declare the operation.
//
// Bitwise contract of sum_f32: out[i] = rank0[i] + rank1[i] in IEEE-754 binary32, round to nearest even, one addition
// per element (no contraction is possible: there is no multiply). This is the same value a device kernel computes
// with `p0[i] + p1[i]` for every pair of finite or infinite inputs, signed zeros included. NaN payloads are not part of
// the contract: both engines treat a non-finite row as an error.
#include <cstddef>
#include <cstdint>

namespace strix::xport {

enum class RowFold : std::uint32_t { none = 0, sum_f32 = 1 };

namespace row_fold_detail {
inline void sum_f32_generic(float* __restrict out, const float* __restrict a, const float* __restrict b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f"))) inline void sum_f32_avx512(float* __restrict out, const float* __restrict a,
                                                              const float* __restrict b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
inline bool has_avx512() {
    static const bool yes = __builtin_cpu_supports("avx512f") != 0;
    return yes;
}
#endif
}  // namespace row_fold_detail

// out = rank0 + rank1 over `bytes` (a multiple of 4) of binary32 values. `out` must not overlap the inputs.
inline void fold_rows(RowFold op, void* out, const void* rank0, const void* rank1, std::size_t bytes) {
    if (op != RowFold::sum_f32) return;
    auto* o = static_cast<float*>(out);
    const auto* a = static_cast<const float*>(rank0);
    const auto* b = static_cast<const float*>(rank1);
    const std::size_t n = bytes / sizeof(float);
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (row_fold_detail::has_avx512()) return row_fold_detail::sum_f32_avx512(o, a, b, n);
#endif
    row_fold_detail::sum_f32_generic(o, a, b, n);
}

}  // namespace strix::xport
