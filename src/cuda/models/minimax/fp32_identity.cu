// minimax-m3 FP32 prefill identity, RTX PRO (CUDA) instantiation of fp32_identity_device.inc.
// Build with --fmad=false (no contraction) and IEEE div/sqrt: the bits must equal the gfx1151 instantiation.
#include "minimax/rtxpf/fp32_identity.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <cstdlib>

namespace strix::minimax::rtxpf {
namespace dev {

#define MMX_DEV __device__ __forceinline__
#define MMX_SHFL_XOR(v, s) __shfl_xor_sync(0xffffffffU, (v), (s), 32)
#define MMX_FMA(a, b, c) __fmaf_rn((a), (b), (c))
#define mmx_sync() __syncthreads()
#define mmx_sync_wave() __syncwarp()
__device__ __forceinline__ float mmx_round_half(float v) { return __half2float(__float2half_rn(v)); }
__device__ __forceinline__ float mmx_half_to_float(std::uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ float mmx_rint(float v) { return rintf(v); }
__device__ __forceinline__ float mmx_u2f(std::uint32_t b) { return __uint_as_float(b); }
#include "minimax/rtxpf/fp32_identity_device.inc"
#undef MMX_DEV
#undef MMX_SHFL_XOR
#undef MMX_FMA
#undef mmx_sync
#undef mmx_sync_wave

// Last job whose begin (row_begin or tile_begin) is <= key.
__device__ __forceinline__ unsigned find_job(const F32Job* jobs, unsigned n, unsigned key, bool tiles) {
    unsigned lo = 0, hi = n;
    while (hi - lo > 1) {
        const unsigned mid = (lo + hi) / 2;
        if ((tiles ? jobs[mid].tile_begin : jobs[mid].row_begin) <= key) lo = mid;
        else hi = mid;
    }
    return lo;
}

__global__ __launch_bounds__(256) void transform_kernel(const F32Job* jobs, unsigned njobs, const float* x, std::uint32_t x_stride,
                                                        const std::uint32_t* xmap, std::uint32_t input, std::uint32_t entries,
                                                        float* xt) {
    const unsigned item = blockIdx.x * 8U + threadIdx.x / 32U, lane = threadIdx.x & 31U;
    if (item >= entries * (input / 128)) return;
    const unsigned j = find_job(jobs, njobs, item / (input / 128), false);
    f32_transform_body(x, x_stride, xmap, jobs[j].suh, input, entries, xt, item, lane);
}
__device__ __forceinline__ float mul1_v31(std::uint32_t state) {
    const unsigned p = state * 0x83dcd12dU;
    // Unsigned dot sums all four bytes exactly, including the 1024 bias.
    const unsigned sum = __dp4a(p, 0x01010101U, 1024U);
    return mmx_round_half(__fmaf_rn(static_cast<float>(sum), 0x1.bb8p-8F, -0x1.4c4p+3F));
}
template<unsigned Version>
struct CoreDecode {
    __device__ __forceinline__ float operator()(std::uint32_t state) const {
        if constexpr (Version == 31) return mul1_v31(state);
        else return f32_mul1(state);
    }
};
// v3.1 preserves v3 geometry, next-tile prefetch, accumulator order and grid.
template <unsigned TR, unsigned Version = 3>
// TR 128: no register cap (1 CTA per SM; capping at 128 registers spilled and was 7-9 % slower, bench/core_bench.cu).
// TR 64 fits 128 registers without spills: 2 CTAs per SM.
__global__ __launch_bounds__(256, TR == 128 ? 1 : 2) void core_kernel(const F32Job* jobs, unsigned njobs, const float* xt, float* raw) {
    __shared__ alignas(16) float as[16][TR + 4];
    __shared__ alignas(16) float ws[16][kF32TileCols + 4];
    __shared__ std::uint32_t wsrc[8][64];
    const unsigned j = find_job(jobs, njobs, blockIdx.x, true);
    const F32Job& jb = jobs[j];
    const unsigned local = blockIdx.x - jb.tile_begin, cols = jb.output / kF32TileCols;
    f32_core_body<TR, CoreDecode<Version>>(jb.trellis, jb.input, jb.output, static_cast<int>(jb.bits), xt + static_cast<std::size_t>(jb.row_begin) * jb.input,
                      jb.rows, raw + static_cast<std::size_t>(jb.row_begin) * jb.output, local / cols, local % cols, as, ws, wsrc, threadIdx.x);
}
__global__ __launch_bounds__(256) void output_kernel(const F32Job* jobs, unsigned njobs, const float* raw, std::uint32_t output,
                                                     std::uint32_t entries, float* y, std::uint32_t y_stride, const std::uint32_t* ymap) {
    const unsigned item = blockIdx.x * 8U + threadIdx.x / 32U, lane = threadIdx.x & 31U;
    if (item >= entries * (output / 128)) return;
    const unsigned j = find_job(jobs, njobs, item / (output / 128), false);
    f32_output_body(raw, jobs[j].svh, output, entries, y, y_stride, ymap, item, lane);
}
__global__ __launch_bounds__(256) void combine_kernel(const float* ye, const float* ys, const std::uint32_t* slot, const float* weight,
                                                      std::uint32_t rows, float* out) {
    f32_combine_body(ye, ys, slot, weight, rows, out, blockIdx.x * 256U + threadIdx.x);
}

__global__ __launch_bounds__(256) void swiglu_kernel(const float* g, const float* u, float* a, std::size_t n) {
    f32_swiglu_body(g, u, a, n, static_cast<std::size_t>(blockIdx.x) * 256U + threadIdx.x);
}

}  // namespace dev

static_assert(sizeof(F32Job) == sizeof(dev::F32Job), "F32Job layout");

Extent layout_jobs(std::vector<F32Job>& jobs, std::uint32_t tile_rows) {
    if (tile_rows != 64 && tile_rows != 128) throw std::invalid_argument("fp32 identity: tile rows 64 or 128");
    Extent e;
    for (auto& j : jobs) {
        if (j.input % 128 || j.output % 128 || (j.bits < 1 || j.bits > 8)) throw std::invalid_argument("fp32 identity: bad matrix");
        j.row_begin = e.entries;
        j.tile_begin = e.tiles;
        e.entries += j.rows;
        e.tiles += (j.rows + tile_rows - 1) / tile_rows * (j.output / dev::kF32TileCols);
    }
    return e;
}

// Shared by enqueue_grouped and the stage benchmark.
template<unsigned Version>
static void launch_core_version(const GroupedCall& c, float* xt, float* raw, cudaStream_t stream) {
    const auto* jobs = reinterpret_cast<const dev::F32Job*>(c.jobs_d);
    if (c.tile_rows == 128) {
        if (c.tiles) dev::core_kernel<128, Version><<<c.tiles, 256, 0, stream>>>(jobs, c.jobs, xt, raw);
        if (c.tail_tiles) dev::core_kernel<64, Version><<<c.tail_tiles, 256, 0, stream>>>(jobs + c.jobs, c.tail_jobs, xt, raw);
    } else if (c.tiles) dev::core_kernel<64, Version><<<c.tiles, 256, 0, stream>>>(jobs, c.jobs, xt, raw);
}
static void launch_grouped_core(const GroupedCall& c, float* xt, float* raw, cudaStream_t stream) {
    if (resolve_core_version(c.core_version, std::getenv("STRIX_MM_FP32_CORE")) == 3)
        launch_core_version<3>(c, xt, raw, stream);
    else launch_core_version<31>(c, xt, raw, stream);
}

void enqueue_grouped(const GroupedCall& c, float* xt, float* raw, void* stream_v) {
    const auto stream = static_cast<cudaStream_t>(stream_v);
    if (c.entries == 0) return;
    const unsigned ti = c.entries * (c.input / 128), to = c.entries * (c.output / 128);
    dev::transform_kernel<<<dim3((ti + 7) / 8), dim3(256), 0, stream>>>(reinterpret_cast<const dev::F32Job*>(c.jobs_d), c.jobs, c.x, c.x_stride, c.xmap, c.input, c.entries, xt);
    launch_grouped_core(c, xt, raw, stream);
    dev::output_kernel<<<dim3((to + 7) / 8), dim3(256), 0, stream>>>(reinterpret_cast<const dev::F32Job*>(c.jobs_d), c.jobs, raw, c.output, c.entries, c.y, c.y_stride, c.ymap);
}

void enqueue_combine(const float* ye, const float* shared, const std::uint32_t* slot, const float* weight, std::uint32_t rows, float* out,
                     void* stream_v) {
    const auto stream = static_cast<cudaStream_t>(stream_v);
    const unsigned n = rows * 6144U;
    dev::combine_kernel<<<dim3((n + 255) / 256), dim3(256), 0, stream>>>(ye, shared, slot, weight, rows, out);
}

void enqueue_swiglu(const float* g, const float* u, float* a, std::size_t n, void* stream_v) {
    if (n == 0) return;
    dev::swiglu_kernel<<<dim3(static_cast<unsigned>((n + 255) / 256)), dim3(256), 0, static_cast<cudaStream_t>(stream_v)>>>(g, u, a, n);
}

}  // namespace strix::minimax::rtxpf
