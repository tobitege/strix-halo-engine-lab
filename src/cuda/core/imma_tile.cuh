// Shared CUDA core: the int8 IMMA (m16n8k32) tile mechanics of a per-32-K-block scaled GEMM (A8 activations x Q8-like weights).
// One IMMA covers exactly one 32-K block, so its int32 result is the exact block dot; the FP32 chain
//   acc = fma(dw * dl, float(dot), acc)      (blocks ascending)
// is the numerical definition and does NOT depend on how rows/outputs are tiled over CTAs, warps or the pipeline.
// The int32 result is seeded with 0x4B400000 (= 1.5 * 2^23 as float bits) in C so that float(dot) is one exact subtraction.
// Per-format weight decode (where the codes/scales come from) stays with the model file; these helpers only take
// registers / shared-memory tiles.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "ptx.cuh"

namespace strix::cuda::core::imma {

// D = A(16x32 s8 row) x B(32x8 s8 col) + seed; d[i] bits = 0x4B400000 + dot.
__device__ __forceinline__ void mma_seeded(int (&d)[4], const std::uint32_t (&a)[4], const std::uint32_t (&b)[2]) {
    ptx::mma_s8_seeded(d, a[0], a[1], a[2], a[3], b[0], b[1], 0x4B400000);
}

// acc = fma(dw * dl, float(dot), acc): the HIP v_mul_f32 / v_fmac_f32 chain, unchanged by tiling.
__device__ __forceinline__ float block_acc(int d, float dw, float dl, float acc) {
    const float dot = __fsub_rn(__int_as_float(d), 12582912.0F);
    return __fmaf_rn(__fmul_rn(dw, dl), dot, acc);
}

// A fragment (16 rows x 32 K bytes, row-major s8 in shared memory, 16-byte aligned rows) with one ldmatrix.x4.
// `tile` = first byte of the tile's row 0 / K byte 0; rows are `pitch_bytes` apart. All 32 lanes call.
__device__ __forceinline__ void load_a_frag(std::uint32_t (&a)[4], const void* tile, std::uint32_t pitch_bytes, int lane) {
    const auto* p = static_cast<const unsigned char*>(tile) +
                    static_cast<std::uint32_t>((lane & 7) + 8 * ((lane >> 3) & 1)) * pitch_bytes + 16U * static_cast<unsigned>(lane >> 4);
    unsigned r[4];
    ptx::ldsm_x4(r, p);
    a[0] = r[0]; a[1] = r[1]; a[2] = r[2]; a[3] = r[3];
}

__device__ __forceinline__ float half_lo(std::uint32_t w) { return __half2float(__ushort_as_half(static_cast<unsigned short>(w & 0xffffU))); }
__device__ __forceinline__ float half_hi(std::uint32_t w) { return __half2float(__ushort_as_half(static_cast<unsigned short>(w >> 16U))); }

}  // namespace strix::cuda::core::imma
