#pragma once
// attn_core LDS staging pieces shared by the flash cores.
//   V^T staging (dense and sparse): a thread holds 4 keys x 8 value dims (one 16-B piece per key, loaded early so the
//   loads land during QK) and stores them transposed, V^T[dim][key], as 8-B stores into rows of `Pitch` bytes
//   (Pitch = 2 x keys + 16: an odd multiple of 16 B, so 16 consecutive dim rows hit distinct bank groups).
//   sum_partials: the head-dim-split score partials, summed in the fixed order 0 .. Split-1.
#include "attn_core/wmma_types.hpp"

namespace strix::attn_core::tile {

// Element k (0..7) of an 8-element 16-bit piece.
__device__ __forceinline__ unsigned pick(U4 v, unsigned k) {
    const unsigned w = k < 2 ? v.x : k < 4 ? v.y : k < 6 ? v.z : v.w;
    return (k & 1U) ? w >> 16U : w & 0xffffU;
}
// Store 4 keys (keys 4 g .. 4 g + 3 of the block) x 8 dims (d0 .. d0 + 7) transposed into V^T rows.
template <unsigned Pitch> __device__ __forceinline__ void store_vt4x8(unsigned char* vt, const U4 (&v)[4], unsigned d0, unsigned g) {
#pragma unroll
    for (unsigned k = 0; k < 8; ++k) {
        const U2 x = U2{pick(v[0], k) | (pick(v[1], k) << 16U), pick(v[2], k) | (pick(v[3], k) << 16U)};
        *reinterpret_cast<U2*>(vt + (d0 + k) * Pitch + 8U * g) = x;
    }
}
// p[i] = part[0][i] + part[1][i] + ... (fixed order); part slices `stride` floats apart.
template <unsigned Split, unsigned K> __device__ __forceinline__ void sum_partials(const float* part, unsigned stride, float (&out)[K]) {
#pragma unroll
    for (unsigned i = 0; i < K; ++i) {
        float v = part[i];
#pragma unroll
        for (unsigned q = 1; q < Split; ++q) v = v + part[q * stride + i];
        out[i] = v;
    }
}

}  // namespace strix::attn_core::tile
