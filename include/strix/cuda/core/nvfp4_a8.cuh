#pragma once
// Device side of nvfp4-a8-v1 (contract: strix/cuda/core/nvfp4_a8.hpp). Shared CUDA core: the converter kernels and
// the rows GEMV template. An engine binds the GEMV through a Policy that supplies the operand of output n and the
// epilogue (its own store forms); the arithmetic up to the scaled FP32 value is fixed here. Compile with --fmad=false
// (every product / sum below is an explicit _rn intrinsic anyway).
#include <cuda_runtime.h>

#include <cstdint>

#include "strix/cuda/core/nvfp4_a8.hpp"

namespace strix::cuda::core::nvfp4_a8 {

// Same value tables and nearest/tie rules as the #683 host functions (nvfp4_reference.cpp), loop for loop.
__host__ __device__ inline float dev_e2m1(std::uint32_t c) {
    const float v[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    return (c & 8U) ? -v[c & 7U] : v[c & 7U];
}
__host__ __device__ inline float dev_e4m3(std::uint32_t c) {
    const int e = static_cast<int>(c >> 3), m = static_cast<int>(c & 7U);
    return e ? ldexpf(1.0F + static_cast<float>(m) / 8.0F, e - 7) : ldexpf(static_cast<float>(m), -9);
}
__device__ inline std::uint32_t dev_encode_e2m1(float x) {
    const float a = fminf(fabsf(x), 6.0F);
    std::uint32_t best = 0;
    float d = a;
    for (std::uint32_t i = 1; i < 8; ++i) {
        const float t = fabsf(__fsub_rn(a, dev_e2m1(i)));
        if (t < d || (t == d && !(i & 1U))) { best = i; d = t; }
    }
    return best | (signbit(x) ? 8U : 0U);
}
__device__ inline std::uint32_t dev_encode_e4m3(float x) {
    std::uint32_t best = 0;
    float d = x;
    for (std::uint32_t i = 1; i < 127; ++i) {
        const float t = fabsf(__fsub_rn(x, dev_e4m3(i)));
        if (t < d || (t == d && !(i & 1U))) { best = i; d = t; }
    }
    return best;
}

// 4 E2M1 codes (one per byte, low nibble) -> 4 int8 values 2 * e2m1 (exact): magnitude by byte permute, sign by
// per-byte two's complement.
__device__ __forceinline__ std::uint32_t int8x4_of_codes(std::uint32_t c) {
    constexpr std::uint32_t lo = 0x03020100U, hi = 0x0C080604U;   // 2 * {0, .5, 1, 1.5} | 2 * {2, 3, 4, 6}
    const std::uint32_t m = c & 0x07070707U;
    const std::uint32_t packed = m | (m >> 4);                     // byte0 = m0 | m1 << 4, byte2 = m2 | m3 << 4
    const std::uint32_t sel = (packed & 0xFFU) | ((packed >> 8) & 0xFF00U);
    const std::uint32_t mag = __byte_perm(lo, hi, sel);
    const std::uint32_t neg = ((c >> 3) & 0x01010101U) * 0xFFU;   // 0xFF in negative bytes
    return __vsub4(mag ^ neg, neg);
}

// Weights of one 16-group (8 code bytes) -> 4 words of int8 in feature order.
__device__ __forceinline__ void decode_group(uint2 c, std::uint32_t (&w)[4]) {
    const std::uint32_t lo0 = c.x & 0x0F0F0F0FU, hi0 = (c.x >> 4) & 0x0F0F0F0FU;
    const std::uint32_t lo1 = c.y & 0x0F0F0F0FU, hi1 = (c.y >> 4) & 0x0F0F0F0FU;
    w[0] = int8x4_of_codes(__byte_perm(lo0, hi0, 0x5140U));
    w[1] = int8x4_of_codes(__byte_perm(lo0, hi0, 0x7362U));
    w[2] = int8x4_of_codes(__byte_perm(lo1, hi1, 0x5140U));
    w[3] = int8x4_of_codes(__byte_perm(lo1, hi1, 0x7362U));
}

// Rows GEMV. Policy: outputs() -> N; operand(n) -> Row{codes, scales, half_ts}; store(row, n, float y).
// One warp computes OUT consecutive outputs for the RMAX-row chunk blockIdx.y of rows [0, rows). Activation codes row r at
// a + r * a_stride (16-byte aligned, a_stride multiple of 16); deltas[b * pitch + r].
struct Row {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    float half_ts;   // 0.5 * tensor scale
};
template <class Policy, std::uint32_t OUT, std::uint32_t RMAX, std::uint32_t WARPS>
__global__ __launch_bounds__(WARPS * 32) void rows_gemv_kernel(Policy p, std::uint32_t rows, std::uint32_t k,
                                                              const std::int8_t* a, std::uint32_t a_stride,
                                                              const float* deltas, std::uint32_t pitch) {
    // blockIdx.y = row chunk of RMAX rows (same per-(n, r) chain: the split changes parallelism, never the bits).
    const std::uint32_t r0 = blockIdx.y * RMAX;
    if (r0 >= rows) return;
    rows = rows - r0 < RMAX ? rows - r0 : RMAX;
    a += static_cast<std::uint64_t>(r0) * a_stride;
    deltas += r0;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t n0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * OUT;
    const std::uint32_t outputs = p.outputs();
    if (n0 >= outputs) return;
    Row op[OUT];
#pragma unroll
    for (std::uint32_t o = 0; o < OUT; ++o) op[o] = p.operand(n0 + o < outputs ? n0 + o : n0);
    float acc[OUT][RMAX];
#pragma unroll
    for (std::uint32_t o = 0; o < OUT; ++o)
#pragma unroll
        for (std::uint32_t r = 0; r < RMAX; ++r) acc[o][r] = 0.0F;
    const std::uint32_t groups = k / kGroup;
#pragma unroll 2
    for (std::uint32_t g = lane; g < groups; g += kLanes) {
        std::uint32_t w[OUT][4];
        float s[OUT];
#pragma unroll
        for (std::uint32_t o = 0; o < OUT; ++o) {
            decode_group(__ldg(reinterpret_cast<const uint2*>(op[o].codes) + g), w[o]);
            s[o] = dev_e4m3(__ldg(op[o].scales + g));
        }
#pragma unroll
        for (std::uint32_t r = 0; r < RMAX; ++r) {
            if (r < rows) {
                const int4 x = *reinterpret_cast<const int4*>(a + static_cast<std::uint64_t>(r) * a_stride + g * kGroup);
                const float delta = deltas[static_cast<std::uint64_t>(g >> 1) * pitch + r];
#pragma unroll
                for (std::uint32_t o = 0; o < OUT; ++o) {
                    int dot = __dp4a(static_cast<int>(w[o][0]), x.x, 0);
                    dot = __dp4a(static_cast<int>(w[o][1]), x.y, dot);
                    dot = __dp4a(static_cast<int>(w[o][2]), x.z, dot);
                    dot = __dp4a(static_cast<int>(w[o][3]), x.w, dot);
                    acc[o][r] = __fmaf_rn(__fmul_rn(s[o], delta), static_cast<float>(dot), acc[o][r]);
                }
            }
        }
    }
#pragma unroll
    for (std::uint32_t o = 0; o < OUT; ++o)
#pragma unroll
        for (std::uint32_t r = 0; r < RMAX; ++r)
#pragma unroll
            for (std::uint32_t off = 16; off > 0; off >>= 1)
                acc[o][r] = __fadd_rn(acc[o][r], __shfl_xor_sync(0xFFFFFFFFU, acc[o][r], off));
#pragma unroll
    for (std::uint32_t r = 0; r < RMAX; ++r)
        if (r < rows && lane == r % 32U)
#pragma unroll
            for (std::uint32_t o = 0; o < OUT; ++o)
                if (n0 + o < outputs) p.store(r0 + r, n0 + o, __fmul_rn(acc[o][r], op[o].half_ts));
}

}  // namespace strix::cuda::core::nvfp4_a8
