// Shared CUDA core: warp / block reductions, the top-k order and the exact int -> float of the A8 dots. One
// definition each, the bodies the model files carried before the backend move; internal linkage as before.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace strix::cuda::core {
namespace {

// Warp sum by butterfly, fixed order: v[l] + v[l ^ d] for d = 16, 8, 4, 2, 1 (every lane ends with the same value).
__device__ __forceinline__ float butterfly(float v) {
#pragma unroll
    for (unsigned d = 16U; d; d >>= 1U) v = v + __shfl_xor_sync(0xffffffffU, v, d);
    return v;
}

// Selection order of the top-k: larger value first, ties to the lower index.
__device__ __forceinline__ bool better(float v, std::uint32_t i, float bv, std::uint32_t bi) {
    return v > bv || (v == bv && i < bi);
}

// Exclusive prefix sum over the block (blockDim.x a multiple of 32, <= 1024): value becomes the sum of the lower
// threads' values; returns the block total. warp_sums holds 33 words of shared memory.
__device__ std::uint32_t block_exclusive_scan(std::uint32_t& value, std::uint32_t* warp_sums) {
    constexpr unsigned kFull = 0xffffffffU;
    const std::uint32_t lane = threadIdx.x & 31U, warp = threadIdx.x / 32U, warps = blockDim.x / 32U;
    std::uint32_t inclusive = value;
    for (int d = 1; d < 32; d *= 2) {
        const std::uint32_t other = __shfl_up_sync(kFull, inclusive, d);
        if (lane >= static_cast<std::uint32_t>(d)) inclusive += other;
    }
    if (lane == 31U) warp_sums[warp] = inclusive;
    __syncthreads();
    if (warp == 0U) {
        std::uint32_t s = lane < warps ? warp_sums[lane] : 0U, inc = s;
        for (int d = 1; d < 32; d *= 2) {
            const std::uint32_t other = __shfl_up_sync(kFull, inc, d);
            if (lane >= static_cast<std::uint32_t>(d)) inc += other;
        }
        if (lane < warps) warp_sums[lane] = inc - s;
        if (lane == 31U) warp_sums[32] = inc;
    }
    __syncthreads();
    value = warp_sums[warp] + inclusive - value;
    const std::uint32_t total = warp_sums[32];
    __syncthreads();
    return total;
}

// |i| < 2^22 to float exactly (magic-number conversion, no cvt).
__device__ __forceinline__ float dot_float(int i) { return __int_as_float(0x4B400000 + i) - 12582912.0F; }

}  // namespace
}  // namespace strix::cuda::core
