#pragma once
// attn_core online softmax: the one running-max / running-sum step for every flash core.
// State per row: m (running max, log2 units, starts -inf) and l (running sum, starts 0). A block's scores arrive
// already x scale*log2(e), masked keys = -inf, spread over `Lanes` consecutive lanes (K keys each).
// Contract (pinned once, review t2 section 4):
//   - p = exp2(s - m_new) rounded to the operand type; SumOf::rounded (default) adds the ROUNDED p to l, so P / l
//     normalises exactly what the PV WMMA accumulates; SumOf::fp32 adds the unrounded p (dense's current form, kept
//     selectable until it migrates).
//   - fully masked block (every s = -inf on all Lanes lanes): alpha = 1, p = 0, m and l unchanged.
//   - reductions: max then sum, each a shfl_xor butterfly over Lanes (1, 2, 4, ...) x Stride (lane group = lanes
//     Stride apart; default 1 = consecutive lanes), deterministic.
//   - finalize: out = acc * (1 / l) per row.
#include "attn_core/wmma_types.hpp"

namespace strix::attn_core::tile {

enum class SumOf : unsigned { rounded = 0, fp32 = 1 };

struct SoftmaxState {
    float m = -__builtin_huge_valf();
    float l = 0.0F;
};

template <Operand O, unsigned K, unsigned Lanes = 8, SumOf Sum = SumOf::rounded, unsigned Stride = 1>
__device__ __forceinline__ float online_step(const float (&s)[K], SoftmaxState& st, std::uint16_t (&p)[K]) {
    static_assert(Lanes >= 1 && Lanes <= 32 && (Lanes & (Lanes - 1)) == 0, "power-of-two lane group");
    float bm = -__builtin_huge_valf();
#pragma unroll
    for (unsigned i = 0; i < K; ++i) bm = s[i] > bm ? s[i] : bm;
#pragma unroll
    for (unsigned x = 1; x < Lanes; x <<= 1U) {
        const float o = __shfl_xor(bm, static_cast<int>(x * Stride));
        bm = o > bm ? o : bm;
    }
    if (bm == -__builtin_huge_valf()) {  // fully masked block (uniform over the lane group)
#pragma unroll
        for (unsigned i = 0; i < K; ++i) p[i] = 0;
        return 1.0F;
    }
    const float mn = bm > st.m ? bm : st.m;
    const float alpha = __builtin_amdgcn_exp2f(st.m - mn);
    float ps = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < K; ++i) {
        const float e = __builtin_amdgcn_exp2f(s[i] - mn);
        p[i] = to_operand<O>(e);
        ps += Sum == SumOf::rounded ? operand_value<O>(p[i]) : e;
    }
#pragma unroll
    for (unsigned x = 1; x < Lanes; x <<= 1U) ps += __shfl_xor(ps, static_cast<int>(x * Stride));
    st.l = st.l * alpha + ps;
    st.m = mn;
    return alpha;
}
// Two-term probabilities: p = hi + lo, both rounded to the operand type (lo = the rounding remainder of hi), for cores
// that run the PV product once per term. l accumulates hi + lo (what the two PV products accumulate). Same contract
// otherwise (fully masked group: alpha = 1, both terms 0).
template <Operand O, unsigned K, unsigned Lanes = 8, unsigned Stride = 1>
__device__ __forceinline__ float online_step_split(const float (&s)[K], SoftmaxState& st, std::uint16_t (&hi)[K], std::uint16_t (&lo)[K]) {
    static_assert(Lanes >= 1 && Lanes <= 32 && (Lanes & (Lanes - 1)) == 0, "power-of-two lane group");
    float bm = -__builtin_huge_valf();
#pragma unroll
    for (unsigned i = 0; i < K; ++i) bm = s[i] > bm ? s[i] : bm;
#pragma unroll
    for (unsigned x = 1; x < Lanes; x <<= 1U) {
        const float o = __shfl_xor(bm, static_cast<int>(x * Stride));
        bm = o > bm ? o : bm;
    }
    if (bm == -__builtin_huge_valf()) {
#pragma unroll
        for (unsigned i = 0; i < K; ++i) hi[i] = lo[i] = 0;
        return 1.0F;
    }
    const float mn = bm > st.m ? bm : st.m;
    const float alpha = __builtin_amdgcn_exp2f(st.m - mn);
    float ps = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < K; ++i) {
        const float e = __builtin_amdgcn_exp2f(s[i] - mn);
        hi[i] = to_operand<O>(e);
        const float h = operand_value<O>(hi[i]);
        lo[i] = to_operand<O>(e - h);
        ps += h + operand_value<O>(lo[i]);
    }
#pragma unroll
    for (unsigned x = 1; x < Lanes; x <<= 1U) ps += __shfl_xor(ps, static_cast<int>(x * Stride));
    st.l = st.l * alpha + ps;
    st.m = mn;
    return alpha;
}
__device__ __forceinline__ float finalize_scale(const SoftmaxState& st) { return 1.0F / st.l; }

}  // namespace strix::attn_core::tile
