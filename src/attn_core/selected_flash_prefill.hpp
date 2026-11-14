#pragma once
// Selected-key schedule: two feature-half waves exchange partial scores in a fixed operand order.
// Source policies own masks, score/value scales, multiplicity and the physical value-row permutation.
#include "flash_core.hpp"

namespace strix::attn_core {

template <class P>
__device__ __forceinline__ void selected_flash_step(
    const Half16 (&qf)[8], tile::Acc (&acc)[8],
    const _Float16 (&keys)[16][P::kPitch], const _Float16 (&values)[P::kDim][16],
    float (&exchange)[2][32][8], unsigned col, unsigned half, unsigned fh, unsigned lane,
    float& m, float& l, const P& source) {
    static_assert(P::kDim == 256U);
    tile::Acc part[2] = {};
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) {
        const tile::Frag a = *reinterpret_cast<const tile::Frag*>(&keys[col][fh * (P::kDim / 2U) + 16U * k]);
        part[k % 2U] = tile::mma<tile::Operand::f16>(a, __builtin_bit_cast(tile::Frag, qf[k]), part[k % 2U]);
    }
    const tile::Acc mine = part[0] + part[1];
    *reinterpret_cast<tile::Acc*>(&exchange[fh][lane][0]) = mine;
    source.barrier();
    const tile::Acc other = *reinterpret_cast<const tile::Acc*>(&exchange[fh ^ 1U][lane][0]);
    const tile::Acc scores = fh == 0U ? mine + other : other + mine;
    float p[8], top = -INFINITY;
#pragma unroll
    for (unsigned e = 0; e < 8U; ++e) {
        p[e] = source.score(scores[e], e);
        top = fmaxf(top, p[e]);
    }
    const float alpha = online_step<InterleavedSoftmax<P>>(p, m, l, p, InterleavedSoftmax<P>{source, top});
    if (__builtin_amdgcn_ballot_w32(alpha != 1.0F))
#pragma unroll
        for (unsigned n = 0; n < 8U; ++n) acc[n] *= alpha;
    Half16 pb;
#pragma unroll
    for (unsigned e = 0; e < 8U; ++e) {
        const float o = source.swap(p[e]);
        float lo = half ? o : p[e], hi = half ? p[e] : o;
        source.value_scale(lo, hi, e);
        pb[2U * e] = static_cast<_Float16>(lo);
        pb[2U * e + 1U] = static_cast<_Float16>(hi);
    }
#pragma unroll
    for (unsigned n = 0; n < 8U; ++n) {
        const tile::Frag a = *reinterpret_cast<const tile::Frag*>(&values[source.value_row(fh * (P::kDim / 2U) + 16U * n + col)][0]);
        acc[n] = tile::mma<tile::Operand::f16>(a, __builtin_bit_cast(tile::Frag, pb), acc[n]);
    }
    source.barrier();
}

}  // namespace strix::attn_core
