#pragma once
// Flash arithmetic contracts: policies select score/max order, exponent and probability reduction.
// All schedules use tile::mma and this online state transition; policy calls are fully inlined.
#include "flash_tile.hpp"
// The 8-lane operand-rounded step (sparse::flash_rows_kernel) is tile::online_step in online_softmax.hpp;
// the policies here cover the other lane layouts and denominators.

namespace strix::attn_core {

using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Half2 = _Float16 __attribute__((ext_vector_type(2)));
using Word8 = unsigned __attribute__((ext_vector_type(8)));

template <unsigned Dim>
__device__ __forceinline__ unsigned pair_slot(unsigned p, unsigned d) {
    return p * Dim + (((d >> 2U) ^ (p & 15U)) << 2U) + (d & 3U);
}

// The policy retains the original reduction topology and operand rounding boundary.
// In particular, an FP32 denominator and an operand-rounded denominator are distinct policies.
template <class P, class Scores, class Probabilities>
__device__ __forceinline__ float online_step(const Scores& s, float& m, float& l, Probabilities& p, const P& policy = {}) {
    const float mn = policy.maximum(s, m);
    const float alpha = policy.rescale(m, mn);
    const float sum = policy.probabilities(s, mn, p);
    l = l * alpha + sum;
    m = mn;
    return alpha;
}

// Two adjacent keys per lane; independent 16-lane rows, FP32 probabilities and denominator.
// E supplies the model's exp implementation (including its range and NaN behavior).
template <class E>
struct PairSoftmax {
    __device__ __forceinline__ float maximum(const float (&s)[2], float m) const {
        float tm = fmaxf(s[0], s[1]);
#pragma unroll
        for (unsigned d = 1; d < 16; d *= 2) tm = fmaxf(tm, __shfl_xor(tm, static_cast<int>(d), 32));
        return fmaxf(m, tm);
    }
    __device__ __forceinline__ float rescale(float m, float mn) const { return E::exp(m - mn); }
    __device__ __forceinline__ float probabilities(const float (&s)[2], float mn, float (&p)[2]) const {
        p[0] = E::exp(s[0] - mn); p[1] = E::exp(s[1] - mn);
        float ps = p[0] + p[1];
#pragma unroll
        for (unsigned d = 1; d < 16; d *= 2) ps = ps + __shfl_xor(ps, static_cast<int>(d), 32);
        return ps;
    }
};

// Interleaved keys in opposite half-waves, FP32 denominator before probability rounding.
// S supplies the lane exchange and selection multiplicity; an empty row uses ref=0 while keeping m=-inf.
template <class S>
struct InterleavedSoftmax {
    const S& source;
    float lane_maximum;  // fused into the caller's score loop (keeps the score tile out of extra registers)
    __device__ __forceinline__ float maximum(const float (&)[8], float m) const {
        return fmaxf(m, fmaxf(lane_maximum, source.swap(lane_maximum)));
    }
    __device__ __forceinline__ float rescale(float m, float mn) const {
        const float ref = mn == -INFINITY ? 0.0F : mn;
        return __builtin_amdgcn_exp2f(m - ref);
    }
    __device__ __forceinline__ float probabilities(const float (&s)[8], float mn, float (&p)[8]) const {
        const float ref = mn == -INFINITY ? 0.0F : mn;
        float sum = 0.0F;
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) p[e] = __builtin_amdgcn_exp2f(s[e] - ref);
        source.weight(p);
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) sum += p[e];
        sum += source.swap(sum);
        return sum;
    }
};

}  // namespace strix::attn_core
