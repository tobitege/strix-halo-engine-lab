#pragma once
// Exact top-k selection through approximate index scores (used with index_score_wmma.hpp, mode "exact").
//
// Let a[b] be approximate scores with |a[b] - e[b]| <= delta for the exact scores e[b], and t the k-th largest a[b].
// Then, with band = { b : |a[b] - t| <= 2 delta }:
//   * a[b] > t + 2 delta  ->  b is in the exact top-k. Every block with a larger exact score has e > t + delta, so its
//     approximate score is above t, and fewer than k blocks are above t.
//   * a[b] < t - 2 delta  ->  b is not in the exact top-k. At least k blocks have a >= t, so e >= t - delta, and
//     e[b] < t - delta.
//   * Only the band is undecided. Replacing a[b] by e[b] for the band blocks and keeping every other value gives
//     scores whose top-k set, including ties at the cut-off taken in index order, is the exact one: the blocks above
//     the band stay above the cut-off (fewer than k values can exceed theirs), the blocks below it stay below
//     (at least k values are >= t - delta), and the band blocks compete with their exact values.
// The engine therefore rescores the band with its exact kernel arithmetic and leaves its selection kernel unchanged.
//
// delta for the split-query WMMA dot (BF16 keys exact, query = BF16 hi + BF16 lo of the FP32 residual):
//   operand error   <= 2^-16 sum |q_i k_i|   (two round-to-nearest steps of 8 significant bits)
//   accumulation    <= 2^-16 sum |q_i k_i|   (258 FP32 additions of exact 16-bit products, any order)
//   exact kernel    <= 2^-20 sum |q_i k_i|   (its own FP32 rounding against the real dot)
// and sum |q_i k_i| <= |q|_2 |k|_2. kSplitDotRelError leaves a factor of four over that sum.
#include <hip/hip_runtime.h>

namespace strix::attn_core::idx {

inline constexpr float kSplitDotRelError = 0x1p-13F;  // |wmma dot - exact dot| <= kSplitDotRelError * |q|_2 * |k|_2

__host__ __device__ constexpr bool in_band(float score, float threshold, float delta) {
    return score >= threshold - 2.0F * delta && score <= threshold + 2.0F * delta;
}

// delta for the FP16 WMMA dot with both FP32 operands rounded to FP16 (GLM), vectors of at most 256 elements, every
// |x| < kF16Safe (above it the conversion can overflow: the engine then rescores everything):
//   operand error   |fl16(x) - x| <= 2^-11 |x| + 2^-25 (the second term covers FP16 subnormals), so per vector
//                   |dq|_2 <= eq = 2^-11 |q|_2 + 2^-21, and |q~ . k~ - q . k| <= eq |k| + (|q| + eq) ek
//   accumulation    FP16 products are exact in FP32; the FP32 additions of the WMMA chain, the exact kernel's own FMA
//                   chain against the real dot and both head epilogues stay below 2^-16 |q~|_2 |k~|_2;
//                   kF16AccumRel = 2^-13 leaves a factor of eight over that.
// Streaming top-k (scores arrive in windows and only the retained top-k survives a window): rescore every score
// >= threshold - 2 delta instead of the band, so the retained values are exact ones (delta 0) in the next window.
inline constexpr float kF16Safe = 65504.0F, kF16RoundRel = 0x1p-11F, kF16RoundAbs = 0x1p-21F, kF16AccumRel = 0x1p-13F;
__host__ __device__ constexpr float f16_dot_error(float qn, float kn) {  // qn = |q|_2, kn = |k|_2 (or an upper bound)
    const float eq = kF16RoundRel * qn + kF16RoundAbs, ek = kF16RoundRel * kn + kF16RoundAbs;
    return eq * kn + (qn + eq) * ek + kF16AccumRel * (qn + eq) * (kn + ek);
}

// Work-group reduction of one value per thread (blockDim.x a multiple of 32, one slot per wave): wave shuffles, then one
// LDS slot per wave. Two barriers; every thread returns the same value.
template <class T, class Op> __device__ __forceinline__ T group_reduce(T x, T* slots, Op op) {
#pragma unroll
    for (unsigned d = 16; d >= 1; d >>= 1U) x = op(x, __shfl_xor(x, static_cast<int>(d), 32));
    if ((threadIdx.x & 31U) == 0) slots[threadIdx.x >> 5U] = x;
    __syncthreads();
    T m = slots[0];
    for (unsigned w = 1; w < blockDim.x / 32U; ++w) m = op(m, slots[w]);
    __syncthreads();  // the slots are free again
    return m;
}

}  // namespace strix::attn_core::idx
