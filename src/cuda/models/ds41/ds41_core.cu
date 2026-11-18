// deepseek41 CUDA layer core: A8 activations and the Q8_0 x A8 sites (ds41-exact-v1, bitwise the host reference).

#include "strix/cuda/ds41_core.hpp"

#include "strix/ds41_layer.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace strix::ds41::cuda {
namespace {

constexpr unsigned kFull = 0xffffffffU;

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("ds41 cuda: ") + what);
}

// ---- A8 ------------------------------------------------------------------------------------------------------------
// One warp per (row, group): lane j holds element j. amax ignores NaN as std::max(amax, |v|) does from 0; the clamp is
// std::min(127, std::max(-127, rint(v / delta))) with its comparison order (a NaN quotient gives -127).
__global__ void quantize_a8_kernel(const float* __restrict__ x, std::uint32_t rows, std::uint32_t groups, std::uint32_t x_stride,
                                   std::int8_t* __restrict__ codes, float* __restrict__ deltas, std::int32_t* __restrict__ sums,
                                   std::uint32_t out_stride) {
    // no early PDL trigger: a dependent GEMV launched at this kernel's entry sat resident beside it and slowed this
    // latency-bound kernel ~2x on the PRO; every CTA triggers implicitly when it exits
    const std::uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32U, lane = threadIdx.x & 31U;
    if (warp >= rows * groups) return;
    const std::uint32_t r = warp / groups, g = warp % groups;
    const float v = x[std::size_t{r} * x_stride + g * 32U + lane];
    const float a = fabsf(v);
    float amax = a > 0.0F ? a : 0.0F;  // NaN -> 0
    for (unsigned d = 16U; d; d >>= 1U) {
        const float o = __shfl_xor_sync(kFull, amax, d);
        amax = o > amax ? o : amax;
    }
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
    const float rounded = tiny ? 0.0F : rintf(v / delta);
    const float lo = -127.0F < rounded ? rounded : -127.0F;
    const float c = lo < 127.0F ? lo : 127.0F;
    const auto code = static_cast<std::int32_t>(c);
    std::int32_t sum = code;
    for (unsigned d = 16U; d; d >>= 1U) sum += __shfl_xor_sync(kFull, sum, d);
    codes[std::size_t{r} * out_stride + g * 32U + lane] = static_cast<std::int8_t>(code);
    if (lane == 0) {
        deltas[(std::size_t{r} * out_stride) / 32U + g] = delta;
        sums[(std::size_t{r} * out_stride) / 32U + g] = sum;
    }
}

// ---- Q8_0 x A8 decode product -------------------------------------------------------------------------------------
// One warp per output row. The weight row is read in chunks of 32 blocks (1088 bytes) staged in shared memory by
// 16-byte loads; lane l then owns block c0 + l, i.e. the lane class of product_q8_0: groups l, l + 32, ... in ascending
// order, acc = acc + float(dot) * (scale * delta) from 0, then the butterfly v += shfl_xor(v, d), d = 16 .. 1.
constexpr unsigned kQ8Warps = 8, kChunkBytes = 32U * 34U, kChunkPad = 16U;

__device__ __forceinline__ std::uint32_t lds_u32(const std::uint8_t* p) { return *reinterpret_cast<const std::uint32_t*>(p); }

template <int R>
__global__ void __launch_bounds__(kQ8Warps * 32)
q8_0_a8_kernel(const std::uint8_t* __restrict__ weight, std::size_t row_bytes, std::uint32_t first, std::uint32_t count,
               std::uint32_t per_group, std::uint32_t k_offset, std::uint32_t width, const std::int8_t* __restrict__ codes,
               const float* __restrict__ deltas, std::uint32_t x_stride, float* __restrict__ out, std::uint32_t out_stride,
               bool accumulate) {
    __shared__ __align__(16) std::uint8_t stage[kQ8Warps][kChunkBytes + kChunkPad];
    const std::uint32_t wib = threadIdx.x / 32U, lane = threadIdx.x & 31U;
    const std::uint32_t o = blockIdx.x * kQ8Warps + wib;
    if (o >= count) return;
    const std::uint32_t groups = width / 32U, xg = o / per_group;
    const std::uint8_t* wrow = weight + std::size_t{first + o} * row_bytes + std::size_t{k_offset / 32U} * 34U;
    const std::int8_t* xc[R];
    const float* xd[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        xc[r] = codes + std::size_t(r) * x_stride + std::size_t{xg} * width;
        xd[r] = deltas + (std::size_t(r) * x_stride + std::size_t{xg} * width) / 32U;
    }
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0F;
    std::uint8_t* s = stage[wib];
    // Chunk c + 1 is loaded into registers (at most 3 x 16 bytes per lane) while chunk c is computed from shared memory.
    uint4 pf[3];
    auto prefetch = [&](std::uint32_t c0) {
        const std::uint32_t n = groups - c0 < 32U ? groups - c0 : 32U, v16 = n * 34U / 16U;
        const uint4* src = reinterpret_cast<const uint4*>(wrow + std::size_t{c0} * 34U);
#pragma unroll
        for (int k = 0; k < 3; ++k)
            if (lane + 32U * k < v16) pf[k] = __ldg(src + lane + 32U * k);
    };
    prefetch(0);
    for (std::uint32_t c0 = 0; c0 < groups; c0 += 32U) {
        const std::uint32_t n = groups - c0 < 32U ? groups - c0 : 32U, bytes = n * 34U, v16 = bytes / 16U;
        __syncwarp();
#pragma unroll
        for (int k = 0; k < 3; ++k)
            if (lane + 32U * k < v16) reinterpret_cast<uint4*>(s)[lane + 32U * k] = pf[k];
        for (std::uint32_t i = v16 * 8U + lane; i < bytes / 2U; i += 32U)  // tail of a partial chunk (n % 8 != 0)
            reinterpret_cast<std::uint16_t*>(s)[i] = reinterpret_cast<const std::uint16_t*>(wrow + std::size_t{c0} * 34U)[i];
        __syncwarp();
        if (c0 + 32U < groups) prefetch(c0 + 32U);
        if (lane < n) {
            const std::uint32_t off = lane * 34U, cstart = off + 2U, base = cstart & ~3U, sh = (cstart & 3U) * 8U;
            const std::uint32_t dword = lds_u32(s + (off & ~3U));
            const auto dbits = static_cast<std::uint16_t>((off & 3U) ? dword >> 16U : dword & 0xffffU);
            const float scale = f16_to_float(dbits);
            int wc[8];
            std::uint32_t prev = lds_u32(s + base);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const std::uint32_t next = lds_u32(s + base + 4U * (j + 1));
                wc[j] = static_cast<int>(__funnelshift_r(prev, next, sh));
                prev = next;
            }
            const std::uint32_t g = c0 + lane;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int4 a0 = *reinterpret_cast<const int4*>(xc[r] + 32U * g);
                const int4 a1 = *reinterpret_cast<const int4*>(xc[r] + 32U * g + 16U);
                int dot = 0;
                dot = __dp4a(wc[0], a0.x, dot); dot = __dp4a(wc[1], a0.y, dot);
                dot = __dp4a(wc[2], a0.z, dot); dot = __dp4a(wc[3], a0.w, dot);
                dot = __dp4a(wc[4], a1.x, dot); dot = __dp4a(wc[5], a1.y, dot);
                dot = __dp4a(wc[6], a1.z, dot); dot = __dp4a(wc[7], a1.w, dot);
                const float term = static_cast<float>(dot) * (scale * xd[r][g]);
                acc[r] = acc[r] + term;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float v = acc[r];
        for (unsigned d = 16U; d; d >>= 1U) v = v + __shfl_xor_sync(kFull, v, d);
        if (lane == 0) {
            float* y = out + std::size_t(r) * out_stride + o;
            *y = accumulate ? *y + v : v;
        }
    }
}

template <int R>
void launch_q8(const Q8A8View& v, std::uint32_t width, std::uint32_t per_group, std::uint32_t out_stride, std::uint32_t row0,
               cudaStream_t stream) {
    const dim3 grid((v.count + kQ8Warps - 1U) / kQ8Warps), block(kQ8Warps * 32U);
    q8_0_a8_kernel<R><<<grid, block, 0, stream>>>(v.weight.data, v.weight.row_bytes(), v.first, v.count, per_group, v.k_offset, width,
                                                  v.x.codes + std::size_t{row0} * v.x.stride,
                                                  v.x.deltas + std::size_t{row0} * v.x.stride / 32U, v.x.stride,
                                                  v.out + std::size_t{row0} * out_stride, out_stride, v.accumulate);
}

}  // namespace

void enqueue_quantize_a8(const QuantizeA8View& v, cudaStream_t stream) {
    const std::uint32_t x_stride = v.x_stride ? v.x_stride : v.width, out_stride = v.out.stride ? v.out.stride : v.width;
    require(v.width % 32U == 0 && v.width > 0 && v.rows > 0 && out_stride % 32U == 0 && out_stride >= v.width, "A8 geometry");
    const std::uint32_t groups = v.width / 32U, warps = v.rows * groups;
    quantize_a8_kernel<<<(warps + 7U) / 8U, 256, 0, stream>>>(v.x, v.rows, groups, x_stride, v.out.codes, v.out.deltas, v.out.sums,
                                                              out_stride);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_q8_0_a8(const Q8A8View& v, cudaStream_t stream) {
    if (v.tiled.codes) {
        enqueue_q8_0_a8_tiled(v, stream);
        return;
    }
    const std::uint32_t width = v.width ? v.width : v.weight.width, per_group = v.per_group ? v.per_group : v.count;
    const std::uint32_t out_stride = v.out_stride ? v.out_stride : v.count;
    require(width % 32U == 0 && width > 0 && v.k_offset % 32U == 0 && v.k_offset + width <= v.weight.width && v.count > 0 &&
                v.first + v.count <= v.weight.rows && v.count % per_group == 0 && v.x.rows > 0 && v.x.stride % 32U == 0,
            "Q8_0 x A8 geometry");
    require((v.count / per_group - 1U) * width + width <= v.x.stride, "Q8_0 x A8 activation stride");
    require(reinterpret_cast<std::uintptr_t>(v.weight.data) % 16U == 0 && v.weight.row_bytes() % 16U == 0 &&
                (v.k_offset / 32U * 34U) % 16U == 0 && reinterpret_cast<std::uintptr_t>(v.x.codes) % 16U == 0,
            "Q8_0 x A8 alignment (16-byte weight rows and activation codes)");
    for (std::uint32_t r0 = 0; r0 < v.x.rows; r0 += 8U) {
        const std::uint32_t n = v.x.rows - r0 < 8U ? v.x.rows - r0 : 8U;
        switch (n) {
        case 1: launch_q8<1>(v, width, per_group, out_stride, r0, stream); break;
        case 2: launch_q8<2>(v, width, per_group, out_stride, r0, stream); break;
        case 3: launch_q8<3>(v, width, per_group, out_stride, r0, stream); break;
        case 4: launch_q8<4>(v, width, per_group, out_stride, r0, stream); break;
        case 5: launch_q8<5>(v, width, per_group, out_stride, r0, stream); break;
        case 6: launch_q8<6>(v, width, per_group, out_stride, r0, stream); break;
        case 7: launch_q8<7>(v, width, per_group, out_stride, r0, stream); break;
        default: launch_q8<8>(v, width, per_group, out_stride, r0, stream); break;
        }
        STRIX_CUDA_CHECK(cudaGetLastError());
    }
}

}  // namespace strix::ds41::cuda
