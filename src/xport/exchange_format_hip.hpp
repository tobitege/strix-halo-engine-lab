#pragma once
// Device side of the TP2 exchange format (xport/exchange_format.hpp), HIP backend core: the pack body a producer /
// copy kernel runs (binary32 -> binary16, round to nearest even, 16-byte stores) and the widening loads a consumer
// kernel uses (binary16 -> binary32, exact). Same values on every rank: the consumer of both ranks widens the same
// binary16 bits of both partials.
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::xport::hip {

// 8 binary32 -> 8 binary16 (RNE: v_cvt_f16_f32 in the default rounding mode). `overflow` is set when a finite input
// rounds to an infinity (|x| >= 65520): the exchange must then fail closed (the caller counts it).
__device__ inline uint4 pack_f16x8(const float4 a, const float4 b, bool& overflow) {
    const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    _Float16 h[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        h[i] = static_cast<_Float16>(v[i]);
        overflow |= __builtin_isinf(static_cast<float>(h[i])) && !__builtin_isinf(v[i]) && v[i] == v[i];
    }
    uint4 out;
    __builtin_memcpy(&out, h, sizeof(out));
    return out;
}

// 8 binary16 (one 16-byte load) -> 8 binary32, exact.
__device__ inline void widen_f16x8(const uint4 q, float* out) {
    _Float16 h[8];
    __builtin_memcpy(h, &q, sizeof(q));
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = static_cast<float>(h[i]);
}

__device__ inline float widen_f16(const void* base, std::size_t i) {
    return static_cast<float>(static_cast<const _Float16*>(base)[i]);
}

// Grid-stride pack of `elems` binary32 values (a multiple of 8, 16-byte aligned src / dst) into binary16; returns
// whether this thread saw an overflow (the caller reduces / counts).
__device__ inline bool pack_f16_grid(void* dst, const float* src, std::size_t elems) {
    const std::size_t n8 = elems / 8, stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const auto* s = reinterpret_cast<const float4*>(src);
    auto* d = static_cast<uint4*>(dst);
    bool overflow = false;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n8; i += stride)
        d[i] = pack_f16x8(s[2 * i], s[2 * i + 1], overflow);
    return overflow;
}

}  // namespace strix::xport::hip
