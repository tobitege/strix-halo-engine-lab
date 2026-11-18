// nvfp4-a8-v1 converter kernels (strix/cuda/core/nvfp4_a8_cuda.hpp). Load-time only; same operations as the host
// oracle (reference_convert_q8_0 -> #683 nvfp4::quantize), operation for operation.
#include <cuda_fp16.h>
#include <stdexcept>

#include "strix/cuda/core/nvfp4_a8.cuh"
#include "strix/cuda/core/nvfp4_a8_cuda.hpp"

namespace strix::cuda::core::nvfp4_a8 {
namespace {

__device__ __forceinline__ float q8_value(const std::uint8_t* blk, std::uint32_t j) {
    const __half h = *reinterpret_cast<const __half*>(blk);   // 34-byte blocks: 2-byte aligned
    return __fmul_rn(__half2float(h), static_cast<float>(static_cast<std::int8_t>(blk[2 + j])));
}

__global__ void amax_kernel(const std::uint8_t* q8, std::uint64_t blocks, float* amax, int* nonfinite) {
    float m = 0.0F;
    for (std::uint64_t b = blockIdx.x * std::uint64_t{blockDim.x} + threadIdx.x; b < blocks;
         b += std::uint64_t{gridDim.x} * blockDim.x)
        for (std::uint32_t j = 0; j < kBlock; ++j) {
            const float v = q8_value(q8 + b * 34U, j);
            if (!isfinite(v)) atomicOr(nonfinite, 1);
            m = fmaxf(m, fabsf(v));
        }
    for (std::uint32_t off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFU, m, off));
    // Non-negative floats order as their bit patterns: max is order-independent (exact).
    if ((threadIdx.x & 31U) == 0) atomicMax(reinterpret_cast<unsigned int*>(amax), __float_as_uint(m));
}

__global__ void quantize_kernel(const std::uint8_t* q8, std::uint32_t n, std::uint32_t k, float ts, std::uint8_t* codes,
                                std::uint8_t* scales) {
    const std::uint64_t groups = std::uint64_t{n} * (k / kGroup);
    for (std::uint64_t i = blockIdx.x * std::uint64_t{blockDim.x} + threadIdx.x; i < groups;
         i += std::uint64_t{gridDim.x} * blockDim.x) {
        const std::uint64_t r = i / (k / kGroup);
        const std::uint32_t g = static_cast<std::uint32_t>(i % (k / kGroup));
        const std::uint8_t* blk = q8 + (r * (k / kBlock) + g / 2U) * 34U;
        const std::uint32_t j0 = (g & 1U) * kGroup;
        float x[kGroup];
        float amax = 0.0F;
        for (std::uint32_t j = 0; j < kGroup; ++j) {
            x[j] = q8_value(blk, j0 + j);
            amax = fmaxf(amax, fabsf(x[j]));
        }
        const float desired = fminf(448.0F, __fdiv_rn(__fdiv_rn(amax, 6.0F), ts));
        const std::uint32_t sc = dev_encode_e4m3(desired);
        scales[r * (k / kGroup) + g] = static_cast<std::uint8_t>(sc);
        const float scale = __fmul_rn(dev_e4m3(sc), ts);
        for (std::uint32_t j = 0; j < kGroup; j += 2) {
            float u = 0.0F, w = 0.0F;
            if (scale != 0.0F) {
                u = __fdiv_rn(x[j], scale);
                u = u < -6.0F ? -6.0F : (6.0F < u ? 6.0F : u);
                w = __fdiv_rn(x[j + 1], scale);
                w = w < -6.0F ? -6.0F : (6.0F < w ? 6.0F : w);
            }
            codes[r * (k / 2U) + g * 8U + j / 2U] =
                static_cast<std::uint8_t>(dev_encode_e2m1(u) | (dev_encode_e2m1(w) << 4));
        }
    }
}

}  // namespace

cudaError_t amax_q8_0_enqueue(const void* q8, std::uint32_t n, std::uint32_t k, float* amax, int* nonfinite,
                              cudaStream_t stream) {
    if (!q8 || !amax || !nonfinite || !n || !k || k % kBlock) return cudaErrorInvalidValue;
    const std::uint64_t blocks = std::uint64_t{n} * (k / kBlock);
    const unsigned grid = static_cast<unsigned>(blocks / 256U + 1U < 4096U ? blocks / 256U + 1U : 4096U);
    amax_kernel<<<grid, 256, 0, stream>>>(static_cast<const std::uint8_t*>(q8), blocks, amax, nonfinite);
    return cudaGetLastError();
}

cudaError_t quantize_q8_0_enqueue(const void* q8, std::uint32_t n, std::uint32_t k, float ts, std::uint8_t* codes,
                                  std::uint8_t* scales, cudaStream_t stream) {
    if (!q8 || !codes || !scales || !n || !k || k % kBlock || !(ts > 0.0F)) return cudaErrorInvalidValue;
    const std::uint64_t groups = std::uint64_t{n} * (k / kGroup);
    const unsigned grid = static_cast<unsigned>(groups / 256U + 1U < 65535U ? groups / 256U + 1U : 65535U);
    quantize_kernel<<<grid, 256, 0, stream>>>(static_cast<const std::uint8_t*>(q8), n, k, ts, codes, scales);
    return cudaGetLastError();
}

cudaError_t convert_q8_0(const void* q8, std::uint32_t n, std::uint32_t k, Image& image, float* scratch,
                         cudaStream_t stream) {
    if (!image.codes || !image.scales || !scratch) return cudaErrorInvalidValue;
    float amax = 0.0F;
    int nonfinite = 0;
    {   // The amax pass reads every value the quantize pass reads: one nonfinite check, before any image is written.
        struct Flag {
            int* device = nullptr;
            ~Flag() { if (device) cudaFree(device); }
        } flag;
        cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&flag.device), sizeof(int));
        if (e == cudaSuccess) e = cudaMemsetAsync(flag.device, 0, sizeof(int), stream);
        if (e == cudaSuccess) e = cudaMemsetAsync(scratch, 0, sizeof(float), stream);
        if (e == cudaSuccess) e = amax_q8_0_enqueue(q8, n, k, scratch, flag.device, stream);
        if (e == cudaSuccess) e = cudaMemcpyAsync(&amax, scratch, sizeof amax, cudaMemcpyDeviceToHost, stream);
        if (e == cudaSuccess) e = cudaMemcpyAsync(&nonfinite, flag.device, sizeof nonfinite, cudaMemcpyDeviceToHost, stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) return e;
    }
    if (nonfinite) throw std::invalid_argument("nonfinite q8_0 weight");
    image.n = n;
    image.k = k;
    image.tensor_scale = tensor_scale_for_amax(amax);
    return quantize_q8_0_enqueue(q8, n, k, image.tensor_scale, image.codes, image.scales, stream);
}

}  // namespace strix::cuda::core::nvfp4_a8
