// deepseek41 CUDA fast dense prefill sites (cuda_ds41_dense_fast.hpp): relaxed FP32 arithmetic (FMA, any order) on
// the same weights and activations as the exact kernels. FMA is written with explicit intrinsics (the translation
// unit is built without contraction like the others).

#include "../../weight/arch/sm120_sources.cuh"
#include "../../core/ptx.cuh"
#include "../../core/reduce.cuh"
#include "ds41_cuda_common.cuh"

#include "strix/cuda/ds41_dense_fast.hpp"
#include "strix/cuda/ds41_prefill_q8.hpp"

#include <cuda_fp16.h>

#include <type_traits>

namespace strix::ds41::cuda::fast {
namespace {

using detail::full_mask;
using detail::require;

using ::strix::cuda::core::ptx::cp_async16;
using ::strix::cuda::core::ptx::cp_async4;
using ::strix::cuda::core::ptx::cp_async_commit;
using ::strix::cuda::core::ptx::cp_async_wait;
using ::strix::cuda::core::ptx::mma_s8;

// float(i), exact for |i| < 2^22 (|dot| <= 32 * 128 * 127).
using ::strix::cuda::core::dot_float;

// ================================================================================================ Q8_0 x A8 GEMM
struct Args {
    const uint4* codes;
    const std::uint16_t* scales;
    std::uint32_t groups_total, g_off, groups, tile_first, tiles, per_group_tiles, x_group_width;
    const std::int8_t* x;
    const float* deltas;
    std::uint32_t x_stride, rows;
    float* out;
    std::uint32_t out_stride;
    bool accumulate;
};

// Block: BT tiles of 16 outputs x BN rows, WM x WN warps (warp tile BT / WM tiles x BN / WN rows), KG groups of 32
// weights per pipeline stage, STAGES stages of cp.async.
template <int BT, int BN, int WM, int WN, int KG, int STAGES>
struct Shape {
    static constexpr int MT = BT / WM, NT = BN / (8 * WN), threads = 32 * WM * WN;
    static constexpr int w_bytes = KG * BT * 512, s_bytes = KG * BT * 32, x_bytes = KG * BN * 32, d_bytes = KG * BN * 4;
    static constexpr int stage = w_bytes + s_bytes + x_bytes + d_bytes;
    static constexpr std::size_t smem = std::size_t{STAGES} * stage;
    static_assert(BT % WM == 0 && BN % (8 * WN) == 0, "warp tiles");
    static_assert(s_bytes % 16 == 0 && x_bytes % 16 == 0 && stage % 16 == 0, "stage alignment");
};

template <int BT, int BN, int WM, int WN, int KG, int STAGES, class Source = ::strix::cuda::weight::Q8Ds41TileSource>
__global__ void __launch_bounds__(32 * WM * WN) q8_fast_kernel(const Args a) {
    using S = Shape<BT, BN, WM, WN, KG, STAGES>;
    constexpr int MT = S::MT, NT = S::NT, TPB = S::threads;
    extern __shared__ __align__(16) unsigned char smem[];
    const std::uint32_t tid = threadIdx.x, warp = tid / 32U, lane = tid & 31U, gid = lane / 4U, tig = lane % 4U;
    const std::uint32_t wm = warp % WM, wn = warp / WM;
    const std::uint32_t rb = blockIdx.x * BN, tb = blockIdx.y * BT;
    const std::size_t xoff = std::size_t{tb / a.per_group_tiles} * a.x_group_width;
    const std::uint32_t nst = (a.groups + KG - 1U) / KG;

    const auto load = [&](std::uint32_t slot, std::uint32_t g0) {
        unsigned char* st = smem + std::size_t{slot} * S::stage;
#pragma unroll
        for (int kk = 0; kk < KG; ++kk) {
            const std::uint32_t g = g0 + kk;
            if (g >= a.groups) break;
            uint4* ws = reinterpret_cast<uint4*>(st) + kk * BT * 32;
            std::uint16_t* ss = reinterpret_cast<std::uint16_t*>(st + S::w_bytes) + kk * BT * 16;
            std::int8_t* xs = reinterpret_cast<std::int8_t*>(st + S::w_bytes + S::s_bytes) + kk * BN * 32;
            float* ds = reinterpret_cast<float*>(st + S::w_bytes + S::s_bytes + S::x_bytes) + kk * BN;
            const std::uint32_t gg = a.g_off + g;
            for (std::uint32_t i = tid; i < BT * 32U; i += TPB) {
                const std::uint32_t t = min(tb + i / 32U, a.tiles - 1U);
                cp_async16(ws + i, a.codes + (std::size_t{a.tile_first + t} * a.groups_total + gg) * 32U + i % 32U);
            }
            for (std::uint32_t i = tid; i < BT * 2U; i += TPB) {
                const std::uint32_t t = min(tb + i / 2U, a.tiles - 1U);
                cp_async16(ss + 8U * i, a.scales + (std::size_t{a.tile_first + t} * a.groups_total + gg) * 16U + 8U * (i % 2U));
            }
            for (std::uint32_t i = tid; i < BN * 2U; i += TPB) {
                const std::uint32_t row = min(rb + i / 2U, a.rows - 1U);
                cp_async16(xs + 16U * i, a.x + std::size_t{row} * a.x_stride + xoff + 32U * g + 16U * (i % 2U));
            }
            for (std::uint32_t i = tid; i < BN; i += TPB) {
                const std::uint32_t row = min(rb + i, a.rows - 1U);
                cp_async4(ds + i, a.deltas + (std::size_t{row} * a.x_stride + xoff) / 32U + g);
            }
        }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (static_cast<std::uint32_t>(s) < nst) load(s, s * KG);
        cp_async_commit();
    }

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][n][e] = 0.0F;

#pragma unroll 1
    for (std::uint32_t s = 0; s < nst; ++s) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (s + STAGES - 1U < nst) load((s + STAGES - 1U) % STAGES, (s + STAGES - 1U) * KG);
        cp_async_commit();
        const unsigned char* st = smem + std::size_t{s % STAGES} * S::stage;
#pragma unroll
        for (int kk = 0; kk < KG; ++kk) {
            if (s * KG + kk >= a.groups) break;
            const uint4* ws = reinterpret_cast<const uint4*>(st) + kk * BT * 32;
            const __half* ss = reinterpret_cast<const __half*>(st + S::w_bytes) + kk * BT * 16;
            const std::int8_t* xs = reinterpret_cast<const std::int8_t*>(st + S::w_bytes + S::s_bytes) + kk * BN * 32;
            const float* ds = reinterpret_cast<const float*>(st + S::w_bytes + S::s_bytes + S::x_bytes) + kk * BN;
            int2 b[NT];
            float2 d[NT];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const std::uint32_t rl = (wn * NT + n) * 8U;
                b[n] = *reinterpret_cast<const int2*>(xs + (rl + gid) * 32U + 8U * tig);
                d[n] = *reinterpret_cast<const float2*>(ds + rl + 2U * tig);
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const std::uint32_t tl = wm * MT + i;
                const uint4 w = ws[tl * 32U + lane];
                const float s0 = Source::scale_bits(__half_as_ushort(ss[tl * 16U + gid])), s1 = Source::scale_bits(__half_as_ushort(ss[tl * 16U + gid + 8U]));
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    int c[4];
                    mma_s8(w, b[n].x, b[n].y, c);
                    acc[i][n][0] = __fmaf_rn(dot_float(c[0]), __fmul_rn(s0, d[n].x), acc[i][n][0]);
                    acc[i][n][1] = __fmaf_rn(dot_float(c[1]), __fmul_rn(s0, d[n].y), acc[i][n][1]);
                    acc[i][n][2] = __fmaf_rn(dot_float(c[2]), __fmul_rn(s1, d[n].x), acc[i][n][2]);
                    acc[i][n][3] = __fmaf_rn(dot_float(c[3]), __fmul_rn(s1, d[n].y), acc[i][n][3]);
                }
            }
        }
    }
    cp_async_wait<0>();

#pragma unroll
    for (int i = 0; i < MT; ++i) {
        const std::uint32_t t = tb + wm * MT + i;
        if (t >= a.tiles) continue;
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const std::uint32_t row = rb + (wn * NT + n) * 8U + 2U * tig + (e & 1U);
                if (row >= a.rows) continue;
                float* y = a.out + std::size_t{row} * a.out_stride + std::size_t{t} * 16U + gid + (e >= 2 ? 8U : 0U);
                *y = a.accumulate ? *y + acc[i][n][e] : acc[i][n][e];
            }
    }
}

template <int BT, int BN, int WM, int WN, int KG, int STAGES>
void launch(const Args& a, cudaStream_t stream) {
    using S = Shape<BT, BN, WM, WN, KG, STAGES>;
    const auto kernel = q8_fast_kernel<BT, BN, WM, WN, KG, STAGES>;
    static const bool configured = [&] {
        STRIX_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(S::smem)));
        return true;
    }();
    (void)configured;
    const dim3 grid((a.rows + BN - 1U) / BN, (a.tiles + BT - 1U) / BT);
    kernel<<<grid, S::threads, S::smem, stream>>>(a);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

int sm_count() {
    static const int n = [] {
        int dev = 0, v = 0;
        STRIX_CUDA_CHECK(cudaGetDevice(&dev));
        STRIX_CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev));
        return v;
    }();
    return n;
}

}  // namespace

bool q8_0_a8_supported(const Q8A8View& v) {
    if (!v.tiled.codes || v.count == 0 || v.x.rows == 0) return false;
    const std::uint32_t width = v.width ? v.width : v.tiled.width, per_group = v.per_group ? v.per_group : v.count;
    return v.first % 16U == 0 && v.count % 16U == 0 && per_group % 16U == 0 && v.count % per_group == 0 && v.first + v.count <= v.tiled.rows &&
           width % 32U == 0 && width > 0 && v.k_offset % 32U == 0 && v.k_offset + width <= v.tiled.width && v.x.stride % 32U == 0 &&
           (v.count / per_group - 1U) * width + width <= v.x.stride && reinterpret_cast<std::uintptr_t>(v.x.codes) % 16U == 0 &&
           reinterpret_cast<std::uintptr_t>(v.x.deltas) % 4U == 0 && (per_group == v.count || per_group % 128U == 0);
}

void enqueue_q8_0_a8(const Q8A8View& v, cudaStream_t stream) {
    if (!q8_0_a8_supported(v)) {
        pk::enqueue_q8_0_a8_rows(v, stream);
        return;
    }
    const std::uint32_t width = v.width ? v.width : v.tiled.width, per_group = v.per_group ? v.per_group : v.count;
    const Args a{reinterpret_cast<const uint4*>(v.tiled.codes), v.tiled.scales, v.tiled.width / 32U, v.k_offset / 32U, width / 32U,
                 v.first / 16U, v.count / 16U, per_group / 16U, width, v.x.codes, v.x.deltas, v.x.stride, v.x.rows, v.out,
                 v.out_stride ? v.out_stride : v.count, v.accumulate};
    // 128 outputs x 128 rows per block (8 warps of 64 x 32) when that fills the device; 64 x 64 otherwise
    const std::uint64_t big = std::uint64_t{(a.rows + 127U) / 128U} * ((a.tiles + 7U) / 8U);
    if (big >= static_cast<std::uint64_t>(sm_count())) launch<8, 128, 2, 4, 2, 4>(a, stream);
    else launch<4, 64, 2, 2, 4, 3>(a, stream);
}

namespace {
std::size_t align256(std::size_t n) { return (n + 255U) / 256U * 256U; }
}  // namespace

void enqueue_shared_expert(const SharedExpertView& v, cudaStream_t stream) {
    if (!v.gate_up_tiled.codes) {
        cuda::enqueue_shared_expert(v, stream);
        return;
    }
    const std::uint32_t rows = v.x.rows, inter = v.gate_up_tiled.rows / 2U, hidden = v.gate_up_tiled.width;
    require(rows > 0 && v.workspace && v.x.width == hidden && v.down_tiled.codes && v.down_tiled.rows == hidden && v.down_tiled.width == inter,
            "shared expert geometry");
    auto* p = static_cast<std::uint8_t*>(v.workspace);
    const std::size_t f = align256(2U * std::size_t{rows} * inter * 4U), c = align256(std::size_t{rows} * inter),
                      g = align256(std::size_t{rows} * inter / 32U * 4U);
    float* gate = reinterpret_cast<float*>(p);
    const A8Rows act{reinterpret_cast<std::int8_t*>(p + f), reinterpret_cast<float*>(p + f + c), reinterpret_cast<std::int32_t*>(p + f + c + g),
                     rows, inter, inter};
    Q8A8View gu;
    gu.tiled = v.gate_up_tiled;
    gu.count = 2U * inter;
    gu.x = v.x;
    gu.out = gate;
    fast::enqueue_q8_0_a8(gu, stream);
    enqueue_swiglu_a8({gate, gate + inter, rows, inter, 2U * inter, v.limit, nullptr, act}, stream);
    Q8A8View d;
    d.tiled = v.down_tiled;
    d.count = hidden;
    d.x = act;
    d.out = v.out;
    d.out_stride = v.out_stride ? v.out_stride : hidden;
    fast::enqueue_q8_0_a8(d, stream);
}

// ================================================================================================ A8 quantize
namespace {

// 8 lanes per 32-value group, 4 values per lane; code = rint(v * (1 / delta)) instead of rint(v / delta).
__global__ void quantize_a8_fast_kernel(const float* __restrict__ x, std::uint32_t rows, std::uint32_t groups, std::uint32_t x_stride,
                                        std::int8_t* __restrict__ codes, float* __restrict__ deltas, std::int32_t* __restrict__ sums,
                                        std::uint32_t out_stride) {
    const std::size_t t = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const std::size_t grp = t / 8U;
    const bool live = grp < std::size_t{rows} * groups;
    const std::uint32_t r = live ? static_cast<std::uint32_t>(grp / groups) : 0U, g = live ? static_cast<std::uint32_t>(grp % groups) : 0U;
    const std::uint32_t k = static_cast<std::uint32_t>(t % 8U);
    float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    if (live) v = *reinterpret_cast<const float4*>(x + std::size_t{r} * x_stride + g * 32U + 4U * k);
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));  // NaN ignored
#pragma unroll
    for (unsigned d = 4U; d; d >>= 1U) amax = fmaxf(amax, __shfl_xor_sync(full_mask, amax, d));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
    const float inv = tiny ? 0.0F : 1.0F / delta;
    const auto q = [&](float f) {
        const float c = fminf(fmaxf(rintf(f * inv), -127.0F), 127.0F);
        return static_cast<std::int32_t>(c == c ? c : 0.0F);
    };
    const std::int32_t c0 = q(v.x), c1 = q(v.y), c2 = q(v.z), c3 = q(v.w);
    std::int32_t sum = c0 + c1 + c2 + c3;
#pragma unroll
    for (unsigned d = 4U; d; d >>= 1U) sum += __shfl_xor_sync(full_mask, sum, d);
    if (!live) return;
    const std::size_t at = std::size_t{r} * out_stride + g * 32U;
    *reinterpret_cast<char4*>(codes + at + 4U * k) =
        make_char4(static_cast<signed char>(c0), static_cast<signed char>(c1), static_cast<signed char>(c2), static_cast<signed char>(c3));
    if (k == 0U) {
        deltas[at / 32U] = delta;
        sums[at / 32U] = sum;
    }
}

}  // namespace

void enqueue_quantize_a8(const QuantizeA8View& v, cudaStream_t stream) {
    const std::uint32_t x_stride = v.x_stride ? v.x_stride : v.width, out_stride = v.out.stride ? v.out.stride : v.width;
    if (x_stride % 4U != 0 || reinterpret_cast<std::uintptr_t>(v.x) % 16U != 0 || reinterpret_cast<std::uintptr_t>(v.out.codes) % 4U != 0) {
        cuda::enqueue_quantize_a8(v, stream);
        return;
    }
    require(v.width % 32U == 0 && v.width > 0 && v.rows > 0 && out_stride % 32U == 0 && out_stride >= v.width, "fast A8 geometry");
    const std::size_t threads = std::size_t{v.rows} * (v.width / 32U) * 8U;
    quantize_a8_fast_kernel<<<static_cast<unsigned>((threads + 255U) / 256U), 256, 0, stream>>>(v.x, v.rows, v.width / 32U, x_stride, v.out.codes,
                                                                                              v.out.deltas, v.out.sums, out_stride);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

// ================================================================================================ hc mixes
namespace {

constexpr int kHcRowsDefault = 32;  // rows per block
constexpr int kHcSpan = 64;         // hidden indices per chunk (16 threads x 4)
constexpr int kHcDots = 25;         // 24 fn rows + the sum of squares

struct HcArgs {
    const float* stream;  // [row][4][hidden] (the input; with expand: the residual, updated in place)
    std::uint32_t rows, hidden;
    const std::uint16_t* fn;
    const float *base, *scale;
    std::uint32_t iters;
    float hc_eps, mix_eps;
    float* mixes;
    // expand (emix null: none)
    const float* sub;
    std::uint32_t sub_stride;
    const float* emix;  // [row][24]
    // collapse + norm (weight null: none)
    const float* pre;   // [row][pre_stride] (null: identity)
    std::uint32_t pre_stride;
    const float* weight;
    float norm_eps;
    float* x;           // [row][hidden]
    std::int8_t* codes;
    float* deltas;
    std::int32_t* sums;
    std::uint32_t a8_stride;
};

// Block: ROWS rows, 8 ROWS threads. Thread t: rows 2 p, 2 p + 1 of the block (p = t / 16), hidden indices 4 j .. 4 j + 3
// of each 64-index chunk (j = t % 16), all 4 copies. fn of the chunk staged as FP32 in shared memory.
// Per chunk: [expand: y = post * sub + comb^T residual, written back], the 24 fn dots and the sum of squares of y,
// [norm: the collapse sum_k pre[k] y[k] written to x unscaled, and its sum of squares]. After the rows: the reduction
// over the 16 threads, the split per row (one warp per row), [norm: x = weight * (x * rsqrt(mean + eps)) and its A8].
template <int ROWS, bool EXPAND, bool NORM>
__global__ void __launch_bounds__(ROWS * 8) hc_fast_kernel(const HcArgs a) {
    constexpr int THREADS = ROWS * 8;
    __shared__ __align__(16) float fs[hc_mix_count][4][kHcSpan];
    __shared__ float red[ROWS][kHcDots + 2];
    __shared__ float em[EXPAND ? ROWS : 1][hc_mix_count];
    __shared__ float pr[NORM ? ROWS : 1][4];
    const std::uint32_t tid = threadIdx.x, p = tid / 16U, j = tid % 16U;
    const std::uint32_t r0 = blockIdx.x * ROWS;
    const std::uint32_t H = a.hidden, N = 4U * H;
    std::uint32_t row[2];
    bool valid[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const std::uint32_t r = r0 + 2U * p + q;
        valid[q] = r < a.rows;
        row[q] = valid[q] ? r : a.rows - 1U;
    }
    if constexpr (EXPAND) {
        for (std::uint32_t i = tid; i < ROWS * hc_mix_count; i += THREADS) {
            const std::uint32_t r = min(r0 + i / hc_mix_count, a.rows - 1U);
            em[i / hc_mix_count][i % hc_mix_count] = a.emix[std::size_t{r} * hc_mix_count + i % hc_mix_count];
        }
    }
    if constexpr (NORM) {
        for (std::uint32_t i = tid; i < ROWS * 4U; i += THREADS) {
            const std::uint32_t r = min(r0 + i / 4U, a.rows - 1U);
            pr[i / 4U][i % 4U] = a.pre ? a.pre[std::size_t{r} * a.pre_stride + i % 4U] : (i % 4U == 0U ? 1.0F : 0.0F);
        }
    }
    float acc[2][kHcDots], nsq[2] = {0.0F, 0.0F};
#pragma unroll
    for (int q = 0; q < 2; ++q)
#pragma unroll
        for (int m = 0; m < kHcDots; ++m) acc[q][m] = 0.0F;

#pragma unroll 1
    for (std::uint32_t i0 = 0; i0 < H; i0 += kHcSpan) {
        const std::uint32_t i = i0 + 4U * j;
        float4 x[2][4];
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const float* s = a.stream + std::size_t{row[q]} * N + i;
#pragma unroll
            for (int c = 0; c < 4; ++c) x[q][c] = *reinterpret_cast<const float4*>(s + std::size_t(c) * H);
        }
        float4 sv[2];
        if constexpr (EXPAND) {
#pragma unroll
            for (int q = 0; q < 2; ++q) sv[q] = *reinterpret_cast<const float4*>(a.sub + std::size_t{row[q]} * a.sub_stride + i);
        }
        __syncthreads();  // previous chunk's fn reads done (and em / pr written)
        for (std::uint32_t e = tid; e < hc_mix_count * 4U * kHcSpan / 2U; e += THREADS) {
            const std::uint32_t m = e / (2U * kHcSpan), c = e / (kHcSpan / 2U) % 4U, k = 2U * (e % (kHcSpan / 2U));
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(a.fn + std::size_t{m} * N + std::size_t{c} * H + i0 + k));
            fs[m][c][k] = f.x;
            fs[m][c][k + 1U] = f.y;
        }
        if constexpr (EXPAND) {
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const float* mx = em[2U * p + q];
                float4 y[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float t0 = mx[4 + k] * sv[q].x, t1 = mx[4 + k] * sv[q].y, t2 = mx[4 + k] * sv[q].z, t3 = mx[4 + k] * sv[q].w;
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const float cm = mx[8 + jj * 4 + k];
                        t0 = __fmaf_rn(cm, x[q][jj].x, t0);
                        t1 = __fmaf_rn(cm, x[q][jj].y, t1);
                        t2 = __fmaf_rn(cm, x[q][jj].z, t2);
                        t3 = __fmaf_rn(cm, x[q][jj].w, t3);
                    }
                    y[k] = make_float4(t0, t1, t2, t3);
                }
                if (valid[q]) {
                    float* o = const_cast<float*>(a.stream) + std::size_t{row[q]} * N + i;
#pragma unroll
                    for (int k = 0; k < 4; ++k) *reinterpret_cast<float4*>(o + std::size_t(k) * H) = y[k];
                }
#pragma unroll
                for (int k = 0; k < 4; ++k) x[q][k] = y[k];
            }
        }
        if constexpr (NORM) {
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const float* pc = pr[2U * p + q];
                float4 y;
                y.x = pc[0] * x[q][0].x;
                y.y = pc[0] * x[q][0].y;
                y.z = pc[0] * x[q][0].z;
                y.w = pc[0] * x[q][0].w;
#pragma unroll
                for (int c = 1; c < 4; ++c) {
                    y.x = __fmaf_rn(pc[c], x[q][c].x, y.x);
                    y.y = __fmaf_rn(pc[c], x[q][c].y, y.y);
                    y.z = __fmaf_rn(pc[c], x[q][c].z, y.z);
                    y.w = __fmaf_rn(pc[c], x[q][c].w, y.w);
                }
                float t = nsq[q];
                t = __fmaf_rn(y.x, y.x, t);
                t = __fmaf_rn(y.y, y.y, t);
                t = __fmaf_rn(y.z, y.z, t);
                nsq[q] = __fmaf_rn(y.w, y.w, t);
                if (valid[q]) *reinterpret_cast<float4*>(a.x + std::size_t{row[q]} * H + i) = y;
            }
        }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < 4; ++c) {
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const float4 v = x[q][c];
                float t = acc[q][hc_mix_count];
                t = __fmaf_rn(v.x, v.x, t);
                t = __fmaf_rn(v.y, v.y, t);
                t = __fmaf_rn(v.z, v.z, t);
                acc[q][hc_mix_count] = __fmaf_rn(v.w, v.w, t);
            }
#pragma unroll
            for (int m = 0; m < static_cast<int>(hc_mix_count); ++m) {
                const float4 f = *reinterpret_cast<const float4*>(&fs[m][c][4U * j]);
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const float4 v = x[q][c];
                    float t = acc[q][m];
                    t = __fmaf_rn(f.x, v.x, t);
                    t = __fmaf_rn(f.y, v.y, t);
                    t = __fmaf_rn(f.z, v.z, t);
                    acc[q][m] = __fmaf_rn(f.w, v.w, t);
                }
            }
        }
    }
    // reduce over the 16 threads of a row pair
#pragma unroll
    for (int q = 0; q < 2; ++q) {
#pragma unroll
        for (int m = 0; m < kHcDots; ++m) {
            float v = acc[q][m];
#pragma unroll
            for (unsigned d = 8U; d; d >>= 1U) v += __shfl_xor_sync(full_mask, v, d);
            acc[q][m] = v;
        }
#pragma unroll
        for (unsigned d = 8U; d; d >>= 1U) nsq[q] += __shfl_xor_sync(full_mask, nsq[q], d);
    }
    if (j == 0U)
#pragma unroll
        for (int q = 0; q < 2; ++q) {
#pragma unroll
            for (int m = 0; m < kHcDots; ++m) red[2U * p + q][m] = acc[q][m];
            red[2U * p + q][kHcDots] = nsq[q];
        }
    __syncthreads();  // also: this block's x rows are visible to every thread of the block
    const std::uint32_t warp = tid / 32U, lane = tid & 31U;
    for (std::uint32_t rr = warp; rr < ROWS; rr += THREADS / 32) {
        const std::uint32_t r = r0 + rr;
        if (r >= a.rows) break;
        // split (hc_split_sinkhorn): lane m < 25 a dot, lane t = 4 j + k < 16 comb[j][k]
        const float total = lane < kHcDots ? red[rr][lane] : 0.0F;
        const float sq = __shfl_sync(full_mask, total, hc_mix_count);
        const float rs = rsqrtf(sq / static_cast<float>(N) + a.mix_eps);
        const float m = total * rs;
        float* out = a.mixes + std::size_t{r} * hc_mix_count;
        if (lane < 4U) out[lane] = declared_sigmoid(m * a.scale[0] + a.base[lane]) + a.hc_eps;
        else if (lane < 8U) out[lane] = 2.0F * declared_sigmoid(m * a.scale[1] + a.base[lane]);
        const std::uint32_t t = lane & 15U, jr = t / 4U, k = t % 4U;
        float cv = __shfl_sync(full_mask, m, 8U + t);
        cv = cv * a.scale[2] + a.base[8U + t];
        float rowv[4];
#pragma unroll
        for (std::uint32_t q = 0; q < 4U; ++q) rowv[q] = __shfl_sync(full_mask, cv, 4U * jr + q);
        float mx = rowv[0];
#pragma unroll
        for (std::uint32_t q = 1; q < 4U; ++q) mx = rowv[q] > mx ? rowv[q] : mx;
        cv = __expf(cv - mx);
        float sum = 0.0F;
#pragma unroll
        for (std::uint32_t q = 0; q < 4U; ++q) sum += __shfl_sync(full_mask, cv, 4U * jr + q);
        cv = cv / sum + a.hc_eps;
        const auto columns = [&] {
            float s = 0.0F;
#pragma unroll
            for (std::uint32_t q = 0; q < 4U; ++q) s += __shfl_sync(full_mask, cv, 4U * q + k);
            cv = cv / (s + a.hc_eps);
        };
        const auto rows_ = [&] {
            float s = 0.0F;
#pragma unroll
            for (std::uint32_t q = 0; q < 4U; ++q) s += __shfl_sync(full_mask, cv, 4U * jr + q);
            cv = cv / (s + a.hc_eps);
        };
        columns();
        for (std::uint32_t it = 1; it < a.iters; ++it) { rows_(); columns(); }
        if (lane < 16U) out[8U + lane] = cv;
        if constexpr (NORM) {
            // x = weight * (x * rsqrt(mean of squares + eps)), then its A8 (one 32-value group per step)
            const float rv = rsqrtf(red[rr][kHcDots] / static_cast<float>(H) + a.norm_eps);
            float* xr = a.x + std::size_t{r} * H;
            for (std::uint32_t g = 0; g < H / 32U; ++g) {
                const std::uint32_t i = 32U * g + lane;
                const float y = a.weight[i] * (xr[i] * rv);
                xr[i] = y;
                if (a.codes) {
                    float delta;
                    std::int32_t s;
                    const auto code = detail::a8_group(y, delta, s);
                    const std::size_t at = std::size_t{r} * a.a8_stride + i;
                    a.codes[at] = static_cast<std::int8_t>(code);
                    if (lane == 0) {
                        a.deltas[at / 32U] = delta;
                        a.sums[at / 32U] = s;
                    }
                }
            }
        }
    }
}

template <int ROWS, bool EXPAND, bool NORM>
void hc_launch(const HcArgs& a, cudaStream_t stream) {
    hc_fast_kernel<ROWS, EXPAND, NORM><<<(a.rows + ROWS - 1U) / ROWS, ROWS * 8, 0, stream>>>(a);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void enqueue_hc_fused(const HcFusedView& f, cudaStream_t stream) {
    const HcMixView& v = f.mix;
    require(v.rows > 0 && v.hidden > 0 && v.hidden % kHcSpan == 0 && v.iters >= 1 && v.mixes && v.fn && v.base && v.scale, "fast hc geometry");
    HcArgs a{v.stream, v.rows, v.hidden, v.fn, v.base, v.scale, v.iters, v.hc_eps, v.norm_eps, v.mixes, nullptr, 0, nullptr,
             f.pre, f.pre_stride, f.weight, f.eps, f.x, f.a8.codes, f.a8.deltas, f.a8.sums, f.a8.stride ? f.a8.stride : v.hidden};
    const bool expand = f.emix != nullptr, norm = f.weight != nullptr;
    if (expand) {
        a.sub = f.sub;
        a.sub_stride = f.sub_stride ? f.sub_stride : v.hidden;
        a.emix = f.emix;
        require(f.sub && a.sub_stride % 4U == 0, "fast hc expand: sub rows (stride a multiple of 4)");
    }
    require(!norm || (f.x && f.pre_stride >= 4U), "fast hc norm: x rows and the pre stride");
    // 32 rows per block also when that leaves SMs idle (2K rows): 16 or 8 rows read fn more often and measured slower
    const int rows = f.rows_per_block ? static_cast<int>(f.rows_per_block) : kHcRowsDefault;
    require(rows == kHcRowsDefault || rows == 16 || rows == 8, "fast hc rows per block 8, 16 or 32");
    const auto go = [&](auto R) {
        constexpr int RR = decltype(R)::value;
        if (expand && norm) hc_launch<RR, true, true>(a, stream);
        else if (expand) hc_launch<RR, true, false>(a, stream);
        else if (norm) hc_launch<RR, false, true>(a, stream);
        else hc_launch<RR, false, false>(a, stream);
    };
    if (rows == kHcRowsDefault) go(std::integral_constant<int, kHcRowsDefault>{});
    else if (rows == 16) go(std::integral_constant<int, 16>{});
    else go(std::integral_constant<int, 8>{});
}

void enqueue_hc_mixes(const HcMixView& v, cudaStream_t stream, const HcExpandView* e) {
    HcFusedView f;
    f.mix = v;
    if (e) {
        require(e->rows == v.rows && e->hidden == v.hidden && e->residual == v.stream && e->out == v.stream && e->mixes,
                "fast hc expand + mix: in place on the mixed stream");
        f.sub = e->sub;
        f.sub_stride = e->sub_stride;
        f.emix = e->mixes;
    }
    enqueue_hc_fused(f, stream);
}

// ================================================================================================ norm
namespace {

constexpr int kNormThreads = 256, kNormPer = 32;  // n <= 8192

__global__ void __launch_bounds__(kNormThreads)
norm_fast_kernel(const float* __restrict__ x, std::uint32_t x_stride, const float* __restrict__ hc, const float* __restrict__ pre,
                 std::uint32_t pre_stride, const float* __restrict__ weight, std::uint32_t n, float eps, float* out, std::uint32_t out_stride,
                 std::int8_t* __restrict__ codes, float* __restrict__ deltas, std::int32_t* __restrict__ sums, std::uint32_t a8_stride) {
    __shared__ float part[kNormThreads / 32];
    const std::uint32_t r = blockIdx.x, tid = threadIdx.x, lane = tid & 31U, warp = tid / 32U;
    float p[4] = {1.0F, 0.0F, 0.0F, 0.0F};
    if (hc && pre)
#pragma unroll
        for (int c = 0; c < 4; ++c) p[c] = pre[std::size_t{r} * pre_stride + c];
    float v[kNormPer];
    float ss = 0.0F;
#pragma unroll
    for (int k = 0; k < kNormPer; ++k) {
        const std::uint32_t i = tid + k * kNormThreads;
        v[k] = 0.0F;
        if (i < n) {
            float y;
            if (hc) {
                const float* s = hc + std::size_t{r} * 4U * n + i;
                y = p[0] * s[0];
#pragma unroll
                for (int c = 1; c < 4; ++c) y = __fmaf_rn(p[c], s[std::size_t(c) * n], y);
            } else {
                y = x[std::size_t{r} * x_stride + i];
            }
            v[k] = y;
            ss = __fmaf_rn(y, y, ss);
        }
    }
#pragma unroll
    for (unsigned d = 16U; d; d >>= 1U) ss += __shfl_xor_sync(full_mask, ss, d);
    if (lane == 0) part[warp] = ss;
    __syncthreads();
    float total = 0.0F;
#pragma unroll
    for (int w = 0; w < kNormThreads / 32; ++w) total += part[w];
    const float rv = rsqrtf(total / static_cast<float>(n) + eps);
#pragma unroll
    for (int k = 0; k < kNormPer; ++k) {
        const std::uint32_t i = tid + k * kNormThreads;
        if (i >= n) break;  // n a multiple of 32: whole warps leave together
        const float y = weight[i] * (v[k] * rv);
        if (out) out[std::size_t{r} * out_stride + i] = y;
        if (codes) {
            float delta;
            std::int32_t sum;
            const auto code = detail::a8_group(y, delta, sum);
            const std::size_t at = std::size_t{r} * a8_stride + i;
            codes[at] = static_cast<std::int8_t>(code);
            if (lane == 0) {
                deltas[at / 32U] = delta;
                sums[at / 32U] = sum;
            }
        }
    }
}

}  // namespace

void enqueue_norm(const NormView& v, cudaStream_t stream) {
    const std::uint32_t x_stride = v.x_stride ? v.x_stride : v.n, out_stride = v.out_stride ? v.out_stride : v.n;
    const std::uint32_t a8_stride = v.a8.stride ? v.a8.stride : v.n;
    if (v.collapsed || !v.weight || v.n > kNormThreads * kNormPer || (v.x && !v.hc_stream && v.out == v.x)) {
        cuda::enqueue_norm(v, stream);
        return;
    }
    require(v.rows > 0 && v.n > 0 && v.n % 32U == 0 && (v.x || v.hc_stream), "fast norm geometry");
    norm_fast_kernel<<<v.rows, kNormThreads, 0, stream>>>(v.x, x_stride, v.hc_stream, v.pre, v.pre_stride, v.weight, v.n, v.eps, v.out,
                                                         out_stride, v.a8.codes, v.a8.deltas, v.a8.sums, a8_stride);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace strix::ds41::cuda::fast
