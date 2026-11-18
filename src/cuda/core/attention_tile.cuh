// Shared CUDA core: flash-attention tile pieces on mma.sync m16n8k16 BF16 -> FP32 (sm_80+; sm_89 / sm_120).
// FAST forms only: accumulation order inside the instruction is the hardware's, softmax is base-2 with ex2.approx.
//
// Tile shape: one warp, 16 query rows (heads, padded) x 16 keys per call, head_dim D (multiple of 16).
//   Q  smem [16][QS] bf16 row-major (A operand via ldmatrix),
//   K  smem [16][KS] bf16 key-major (B operand of S = Q K^T via ldmatrix, no transpose),
//   V  smem [16][VS] bf16 key-major (B operand of O += P V via ldmatrix.trans),
//   S  = two m16n8 accumulators (keys 0-7, 8-15), 4-way split over the K-loop for MMA ILP,
//   P  = S rounded to bf16, reused directly as the A fragment (accumulator layout == A layout),
//   O  = D/8 m16n8 accumulators; online softmax state m (running max, base 2) and l (per-thread partial sum) for the
//        two rows (g, g + 8) of each thread (g = lane / 4).
// Strides QS/KS/VS in bf16 elements; (stride * 2) mod 128 == 16 keeps ldmatrix conflict free.
#pragma once

#include "ptx.cuh"

#include <cstdint>

namespace strix::cuda::core::attn {
namespace {

using ptx::ldsm_x4;
using ptx::ldsm_x4_trans;
using ptx::mma_bf16;

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
// (lo, hi) -> bf16x2 with lo in the low half (lower k index), round to nearest even.
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
    unsigned r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

// s[j][..] (keys 8j .. 8j + 7) = sum over D of Q[row][d] K[key][d]; layout of an m16n8 accumulator.
template <int D, int QS, int KS>
__device__ __forceinline__ void qk_tile16(float (&s)[2][4], const std::uint16_t* q, const std::uint16_t* k, unsigned lane) {
    float acc[4][2][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[j][t][e] = 0.0F;
    const std::uint16_t* qp = q + (lane & 15U) * QS + (lane >> 4U) * 8U;
    const unsigned m = lane >> 3U, r = lane & 7U;
    const std::uint16_t* kp = k + ((m >> 1U) * 8U + r) * KS + (m & 1U) * 8U;
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
        unsigned a[4], b[4];
        ldsm_x4(a, qp + ks * 16);
        ldsm_x4(b, kp + ks * 16);
        mma_bf16(acc[ks & 3][0], a[0], a[1], a[2], a[3], b[0], b[1]);
        mma_bf16(acc[ks & 3][1], a[0], a[1], a[2], a[3], b[2], b[3]);
    }
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) s[t][e] = (acc[0][t][e] + acc[1][t][e]) + (acc[2][t][e] + acc[3][t][e]);
}

// Online softmax step on a 16-key tile: keys >= valid (0..16) are excluded exactly (p = 0). x = s * scale (scale carries
// log2 e). Updates m, l, rescales o, returns P as the bf16 A fragment of the PV product.
template <int NT>
__device__ __forceinline__ void online_softmax16(const float (&s)[2][4], unsigned valid, float scale, float (&m)[2], float (&l)[2],
                                                 float (&o)[NT][4], unsigned (&pa)[4], unsigned lane) {
    const float ninf = __uint_as_float(0xff800000U);
    const unsigned t = lane & 3U;
    float x[2][4];
    float mx0 = ninf, mx1 = ninf;
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const unsigned key = 8U * j + 2U * t + (e & 1U);
            x[j][e] = key < valid ? s[j][e] * scale : ninf;
            if (e < 2) mx0 = fmaxf(mx0, x[j][e]); else mx1 = fmaxf(mx1, x[j][e]);
        }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffU, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffU, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffU, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffU, mx1, 2));
    const float n0 = fmaxf(m[0], mx0), n1 = fmaxf(m[1], mx1);
    const float al0 = ex2(m[0] - n0), al1 = ex2(m[1] - n1);
    m[0] = n0; m[1] = n1;
    float p[2][4], ps0 = 0.0F, ps1 = 0.0F;
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            p[j][e] = ex2(x[j][e] - (e < 2 ? n0 : n1));
            if (e < 2) ps0 += p[j][e]; else ps1 += p[j][e];
        }
    l[0] = l[0] * al0 + ps0; l[1] = l[1] * al1 + ps1;
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        o[i][0] *= al0; o[i][1] *= al0; o[i][2] *= al1; o[i][3] *= al1;
    }
    pa[0] = pack_bf16(p[0][0], p[0][1]); pa[1] = pack_bf16(p[0][2], p[0][3]);
    pa[2] = pack_bf16(p[1][0], p[1][1]); pa[3] = pack_bf16(p[1][2], p[1][3]);
}

// o[nt] += P (16 rows x 16 keys) V (16 keys x 8 cols of tile nt), nt = 0 .. D / 8 - 1.
template <int D, int VS>
__device__ __forceinline__ void pv_tile16(float (&o)[D / 8][4], const unsigned (&pa)[4], const std::uint16_t* v, unsigned lane) {
    const unsigned m = lane >> 3U, r = lane & 7U;
    const std::uint16_t* vp = v + ((m & 1U) * 8U + r) * VS + (m >> 1U) * 8U;
#pragma unroll
    for (int i = 0; i < D / 16; ++i) {
        unsigned b[4];
        ldsm_x4_trans(b, vp + i * 16);
        mma_bf16(o[2 * i], pa[0], pa[1], pa[2], pa[3], b[0], b[1]);
        mma_bf16(o[2 * i + 1], pa[0], pa[1], pa[2], pa[3], b[2], b[3]);
    }
}

}  // namespace
}  // namespace strix::cuda::core::attn
