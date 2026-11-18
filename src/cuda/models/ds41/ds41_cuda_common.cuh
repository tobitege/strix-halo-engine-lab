#pragma once

// Device helpers of the deepseek41 CUDA kernels: the declared reductions of ds41-exact-v1 (ds41_layer.hpp) and the A8
// group quantizer, written so that the FP32 operations run in the reference order (the translation units are built
// without contraction).

#include "../../core/reduce.cuh"
#include "strix/cuda/ds41_common.hpp"
#include "strix/ds41_layer.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace strix::ds41::cuda::detail {

inline constexpr unsigned full_mask = 0xffffffffU;
inline constexpr std::uint32_t chunk = 640;  // dot_chunk of ds41_layer.hpp

inline void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("ds41 cuda: ") + what);
}

// The butterfly v[l] = v[l] + v[l ^ d], d = 16 .. 1 (every lane ends with lane 0's value).
using ::strix::cuda::core::butterfly;

// Lane dot over n elements (lane l: elements l, l + 32, ... ascending, acc = acc + a * b from 0), then the butterfly.
template <class A, class B>
__device__ __forceinline__ float lane_dot(A a, B b, std::uint32_t n, std::uint32_t lane) {
    float acc = 0.0F;
    std::uint32_t i = lane;
    // steps of 8 class elements: the step's 16 loads are issued before its products (a loop that loads and adds one
    // element at a time pays a full load latency per element); the products are added in the same ascending order
    for (; i + 224U < n; i += 256U) {
        float av[8], bv[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) { av[k] = a(i + 32U * k); bv[k] = b(i + 32U * k); }
#pragma unroll
        for (int k = 0; k < 8; ++k) acc = acc + av[k] * bv[k];
    }
    for (; i < n; i += 32U) acc = acc + a(i) * b(i);
    return butterfly(acc);
}

// A8 of one 32-value group held one value per lane (quantize_a8): returns the code; delta and sum are uniform.
// amax ignores NaN as std::max(amax, |v|) from 0 does; the clamp keeps std::min(127, std::max(-127, q))'s comparisons.
__device__ __forceinline__ std::int32_t a8_group(float v, float& delta, std::int32_t& sum) {
    const float a = fabsf(v);
    float amax = a > 0.0F ? a : 0.0F;
#pragma unroll
    for (unsigned d = 16U; d; d >>= 1U) {
        const float o = __shfl_xor_sync(full_mask, amax, d);
        amax = o > amax ? o : amax;
    }
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
    const float rounded = tiny ? 0.0F : rintf(v / delta);
    const float lo = -127.0F < rounded ? rounded : -127.0F;
    const auto code = static_cast<std::int32_t>(lo < 127.0F ? lo : 127.0F);
    std::int32_t s = code;
#pragma unroll
    for (unsigned d = 16U; d; d >>= 1U) s += __shfl_xor_sync(full_mask, s, d);
    sum = s;
    return code;
}

// A8 of the n values y[0..n) (n a multiple of 32) by one block: warp w takes groups w, w + warps, ...
__device__ __forceinline__ void a8_row(const float* y, std::uint32_t n, std::int8_t* codes, float* deltas, std::int32_t* sums) {
    const std::uint32_t lane = threadIdx.x & 31U, warps = blockDim.x / 32U;
    for (std::uint32_t g = threadIdx.x / 32U; g < n / 32U; g += warps) {
        float delta;
        std::int32_t sum;
        const auto code = a8_group(y[32U * g + lane], delta, sum);
        codes[32U * g + lane] = static_cast<std::int8_t>(code);
        if (lane == 0) { deltas[g] = delta; sums[g] = sum; }
    }
}

}  // namespace strix::ds41::cuda::detail
