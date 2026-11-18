#pragma once

// Device helpers shared by the deepseek41 CUDA expert kernels: exact FP16 decode, IQ2_XXS sign handling, the
// declared butterfly and the A8 epilogue. Compiled with --fmad=false (no contraction of the declared
// FP32 operations), IEEE division and denormals.

#include "../../core/reduce.cuh"
#include "strix/cuda/ds41_moe.hpp"
#include "strix/ds41_moe.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace strix::ds41::cuda::moe_detail {

constexpr unsigned kFull = 0xffffffffU;

// FP16 bits to FP32, exactly as the host's half_to_float (subnormals, infinities and NaN payloads). No branches: the
// hardware conversion is exact for every non-NaN encoding, one select restores a NaN's payload (the hardware returns
// a canonical NaN). The branchy form put a divergence region around every block scale of the decode loops.
__device__ __forceinline__ float half_bits(std::uint32_t bits) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(static_cast<unsigned short>(bits)));
    const bool nan = (bits & 0x7c00U) == 0x7c00U && (bits & 0x3ffU) != 0U;
    return nan ? __uint_as_float(((bits & 0x8000U) << 16U) | 0x7f800000U | ((bits & 0x3ffU) << 13U)) : f;
}

// IQ2_XXS: byte masks (0xff = negative) of four sign bits, and the signed codes of four grid magnitudes.
__device__ __forceinline__ std::uint32_t sign_mask4(std::uint32_t bits4) {
    return ((bits4 * 0x00204081U) & 0x01010101U) * 0xffU;
}
__device__ __forceinline__ std::uint32_t apply_signs(std::uint32_t magnitudes, std::uint32_t mask) {
    return (magnitudes ^ mask) + (mask & 0x01010101U);
}
// Sign byte of a 7-bit index: the index with its parity as bit 7 (iq2_xxs_sign_byte).
__device__ __forceinline__ std::uint32_t sign_byte(std::uint32_t index) {
    index &= 127U;
    return index | ((__popc(index) & 1U) << 7U);
}

// Butterfly of the declared product: v[l] = v[l] + v[l ^ d], d = 16 .. 1; lane 0 holds the reference result.
using ::strix::cuda::core::butterfly;

// quantize_a8 of one 32-value group held one value per lane: writes the lane's code, returns (delta, sum) in every
// lane. max ignores NaN as std::max(amax, |x|) does; NaN values give code -127 as the host's clamps.
struct A8Group {
    float delta;
    std::int32_t sum;
    std::int8_t code;
};
__device__ __forceinline__ A8Group a8_group(float value) {
    float amax = fabsf(value);
#pragma unroll
    for (int d = 16; d >= 1; d /= 2) amax = fmaxf(amax, __shfl_xor_sync(kFull, amax, d));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
    const float rounded = tiny ? 0.0F : rintf(value / delta);
    const float low = -127.0F < rounded ? rounded : -127.0F;  // std::max(-127, rounded)
    const float clamped = low < 127.0F ? low : 127.0F;        // std::min(127, low)
    const auto code = static_cast<std::int8_t>(clamped);
    std::int32_t sum = code;
#pragma unroll
    for (int d = 16; d >= 1; d /= 2) sum += __shfl_xor_sync(kFull, sum, d);
    return {delta, sum, code};
}

}  // namespace strix::ds41::cuda::moe_detail
