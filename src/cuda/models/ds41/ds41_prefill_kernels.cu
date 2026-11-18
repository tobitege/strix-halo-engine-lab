// deepseek41 CUDA prefill-shape lane-dot kernels (cuda_ds41_prefill_kernels.hpp): the F16 / F32 projections of the
// prefill (Engram kv, compressor kv / gate, indexer q_b / proj, router logits) over many rows per weight read, with
// the bits of the decode kernels. The Engram decode / injection and the router selection are those of
// ds41_core_io.cu / ds41_core_moe.cu (same code, same order).

#include "../../core/ptx.cuh"
#include "../../core/reduce.cuh"
#include "ds41_cuda_common.cuh"

#include "strix/cuda/ds41_prefill_kernels.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/ds41_scalar_codecs.hpp"

#include <cuda_fp16.h>

#include <cmath>

namespace strix::ds41::cuda::pk {
namespace {

using detail::chunk;
using detail::full_mask;
using detail::require;

// ---- lane-dot GEMM -------------------------------------------------------------------------------------------------
constexpr std::uint32_t kStage = 128;  // elements of a row per pipeline stage (4 lane-class groups)

struct Launch {
    LaneDotJob jobs[4];
    std::uint32_t tile_end[4];  // cumulative output tiles per job
    std::uint32_t job_count, rows;
};

using ::strix::cuda::core::ptx::cp_async16;
using ::strix::cuda::core::ptx::cp_async_commit;
using ::strix::cuda::core::ptx::cp_async_wait;

// Exact conversions of the weight to FP32 (F16 -> F32 is exact; arithmetic NaN results are canonical either way).
__device__ __forceinline__ float wvalue(std::uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ float wvalue(float f) { return f; }

// One step of the reduce-scatter butterfly over the LEN values a[0, LEN): lane l keeps the half selected by bit D of l
// and adds the partner's (l ^ D) copy of it into a[0, LEN / 2); then the steps D / 2 .. 1.
template <int N, int LEN, int D>
__device__ __forceinline__ void reduce_scatter(float (&a)[N], int lane) {
    if constexpr (D > 0) {
        constexpr int half = LEN / 2;
        const bool upper = (lane & D) != 0;
#pragma unroll
        for (int i = 0; i < half; ++i) {
            const float keep = upper ? a[i + half] : a[i];
            const float send = upper ? a[i] : a[i + half];
            a[i] = keep + __shfl_xor_sync(full_mask, send, D);
        }
        reduce_scatter<N, half, D / 2>(a, lane);
    }
}

template <class WT, int TM, int TN, int WM, int WN, int STAGES>
struct Shape {
    static constexpr int BM = TM * WM, BN = TN * WN, threads = WM * WN * 32, N = TM * TN;
    static constexpr std::size_t w_stage = std::size_t{BM} * kStage * sizeof(WT), x_stage = std::size_t{BN} * kStage * 4U;
    static constexpr std::size_t smem = STAGES * (w_stage + x_stage);
    static_assert(N >= 32 && (N & (N - 1)) == 0, "tile values: a power of two >= 32 (reduce-scatter)");
};

template <class WT, int TM, int TN, int WM, int WN, int STAGES, int EPI>
__global__ void __launch_bounds__(WM * WN * 32) lane_dot_kernel(const Launch L) {
    using S = Shape<WT, TM, TN, WM, WN, STAGES>;
    constexpr int BM = S::BM, BN = S::BN, NT = S::threads, N = S::N;
    extern __shared__ __align__(16) unsigned char smem[];
    WT* ws = reinterpret_cast<WT*>(smem);                              // [STAGES][BM][kStage]
    float* xs = reinterpret_cast<float*>(smem + STAGES * S::w_stage);  // [STAGES][BN][kStage]

    std::uint32_t tile = blockIdx.y, j = 0;
    while (tile >= L.tile_end[j]) ++j;
    const LaneDotJob& job = L.jobs[j];
    const std::uint32_t o0 = (tile - (j == 0 ? 0U : L.tile_end[j - 1])) * BM, r0 = blockIdx.x * BN;
    const std::uint32_t width = job.width, count = job.count, rows = L.rows;
    const std::uint32_t x_stride = job.x_stride ? job.x_stride : width;
    const WT* wbase = static_cast<const WT*>(job.weight);

    const auto load = [&](int stage, std::uint32_t k0) {
        constexpr int WCH = kStage * sizeof(WT) / 16;  // 16-byte chunks per weight row
        for (int i = threadIdx.x; i < BM * WCH; i += NT) {
            const int row = i / WCH, ch = i % WCH;
            const std::uint32_t o = min(o0 + row, count - 1U);  // rows past the end: read and discarded
            cp_async16(reinterpret_cast<unsigned char*>(ws + (static_cast<std::size_t>(stage) * BM + row) * kStage) + ch * 16,
                       reinterpret_cast<const unsigned char*>(wbase + std::size_t{o} * width + k0) + ch * 16);
        }
        constexpr int XCH = kStage * 4 / 16;
        for (int i = threadIdx.x; i < BN * XCH; i += NT) {
            const int row = i / XCH, ch = i % XCH;
            const std::uint32_t r = min(r0 + row, rows - 1U);
            cp_async16(xs + (static_cast<std::size_t>(stage) * BN + row) * kStage + ch * 4, job.x + std::size_t{r} * x_stride + k0 + ch * 4);
        }
    };

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wm = warp % WM, wn = warp / WM;
    float acc[N];
#pragma unroll
    for (int i = 0; i < N; ++i) acc[i] = 0.0F;

    const std::uint32_t nk = width / kStage;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (static_cast<std::uint32_t>(s) < nk) load(s, s * kStage);
        cp_async_commit();
    }
    for (std::uint32_t kt = 0; kt < nk; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        const int stage = static_cast<int>(kt % STAGES);
        const WT* wt = ws + (static_cast<std::size_t>(stage) * BM + wm * TM) * kStage + lane;
        const float* xt = xs + (static_cast<std::size_t>(stage) * BN + wn * TN) * kStage + lane;
#pragma unroll
        for (int g = 0; g < static_cast<int>(kStage / 32); ++g) {
            float wv[TM], xv[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) wv[i] = wvalue(wt[i * kStage + g * 32]);
#pragma unroll
            for (int r = 0; r < TN; ++r) xv[r] = xt[r * kStage + g * 32];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int r = 0; r < TN; ++r) acc[i * TN + r] = __fadd_rn(acc[i * TN + r], __fmul_rn(wv[i], xv[r]));
        }
        const std::uint32_t next = kt + STAGES - 1;
        if (next < nk) load(static_cast<int>(next % STAGES), next * kStage);
        cp_async_commit();
    }
    cp_async_wait<0>();

    // reduce-scatter butterfly: after the step of distance d, lane l holds the kept half (bit d of l) of the values,
    // each the butterfly's v[l] + v[l ^ d]; at the end lane l holds tile values [l * N / 32, (l + 1) * N / 32).
    reduce_scatter<N, N, 16>(acc, lane);
    constexpr int per_lane = N / 32;
#pragma unroll
    for (int q = 0; q < per_lane; ++q) {
        const int idx = lane * per_lane + q, i = idx / TN, r = idx % TN;
        const std::uint32_t o = o0 + wm * TM + i, row = r0 + wn * TN + r;
        if (o < count && row < rows) {
            float v = acc[q];
            if constexpr (EPI == static_cast<int>(Epilogue::router_score)) v = router_score(v);
            job.out[std::size_t{row} * (job.out_stride ? job.out_stride : count) + o] = v;
        }
    }
}

template <class WT, int TM, int TN, int WM, int WN, int STAGES, int EPI>
void launch_shape(const Launch& L, cudaStream_t stream) {
    using S = Shape<WT, TM, TN, WM, WN, STAGES>;
    auto* kernel = &lane_dot_kernel<WT, TM, TN, WM, WN, STAGES, EPI>;
    static bool configured = false;  // per instantiation (host side, before the first launch)
    if (!configured) {
        STRIX_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(S::smem)));
        configured = true;
    }
    const dim3 grid((L.rows + S::BN - 1U) / S::BN, L.tile_end[L.job_count - 1U]);
    kernel<<<grid, S::threads, S::smem, stream>>>(L);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

// Tile shapes: TM outputs x TN rows per warp, WM x WN warps per block, 3 stages.
template <class WT, int EPI>
void launch_lane_dot(const LaneDotView& v, cudaStream_t stream) {
    constexpr int TM = 8, TN = 16, WM = 4, WN = 2, STAGES = 3;
    Launch L{};
    std::uint32_t tiles = 0;
    for (std::uint32_t j = 0; j < v.job_count; ++j) {
        L.jobs[j] = v.jobs[j];
        tiles += (v.jobs[j].count + TM * WM - 1U) / (TM * WM);
        L.tile_end[j] = tiles;
    }
    require(tiles <= 65535U, "lane dot: output tiles");
    L.job_count = v.job_count;
    L.rows = v.rows;
    launch_shape<WT, TM, TN, WM, WN, STAGES, EPI>(L, stream);
}

// ---- Engram decode and injection (ds41_core_io.cu) ------------------------------------------------------------------
// emb[r][k * 256 + j] = round_to_bf16(e4m3(code j) * e8m0(scale j / 32)) of row (r, k).
__global__ void engram_decode_kernel(const std::uint8_t* __restrict__ rows, std::uint32_t n_rows, float* __restrict__ emb) {
    const std::size_t idx = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (idx >= std::size_t{n_rows} * engram_row_elements) return;
    const std::size_t row = idx / engram_row_elements, j = idx % engram_row_elements;
    const std::uint8_t* p = rows + row * engram_row_bytes;
    emb[idx] = round_to_bf16(codec::e4m3(p[j]) * codec::e8m0(p[256U + j / 32U]));
}

constexpr std::uint32_t kInjectThreads = 256, kMaxInjectChunks = 32;

// grid (rows, 4 copies): the three chunked dots of engram_gates for copy c (h.h, key.key, (h * (q * k)).key), the
// gate, then h = h + gate * value in place.
__global__ void __launch_bounds__(kInjectThreads)
engram_inject_kernel(float* stream, const float* __restrict__ kv, const float* __restrict__ qn, const float* __restrict__ kn,
                     std::uint32_t hidden, float eps, float inv_sqrt_hidden) {
    __shared__ float part[3][kMaxInjectChunks];
    __shared__ float gate_s;
    const std::uint32_t r = blockIdx.x, c = blockIdx.y, lane = threadIdx.x & 31U, warp = threadIdx.x / 32U, warps = blockDim.x / 32U;
    float* h = stream + (std::size_t{r} * 4U + c) * hidden;
    const float* key = kv + std::size_t{r} * 5U * hidden + std::size_t{c} * hidden;
    const float* value = kv + std::size_t{r} * 5U * hidden + 4U * std::size_t{hidden};
    const float* q = qn + std::size_t{c} * hidden;
    const float* k = kn + std::size_t{c} * hidden;
    const std::uint32_t chunks = (hidden + chunk - 1U) / chunk;
    for (std::uint32_t ch = warp; ch < chunks; ch += warps) {
        const std::uint32_t c0 = ch * chunk, len = hidden - c0 < chunk ? hidden - c0 : chunk;
        const float* hh = h + c0;
        const float* kk = key + c0;
        const float* qq = q + c0;
        const float* kq = k + c0;
        const float a = detail::lane_dot([&](std::uint32_t i) { return hh[i]; }, [&](std::uint32_t i) { return hh[i]; }, len, lane);
        const float b = detail::lane_dot([&](std::uint32_t i) { return kk[i]; }, [&](std::uint32_t i) { return kk[i]; }, len, lane);
        const float d = detail::lane_dot([&](std::uint32_t i) { return hh[i] * (qq[i] * kq[i]); }, [&](std::uint32_t i) { return kk[i]; },
                                         len, lane);
        if (lane == 0) { part[0][ch] = a; part[1][ch] = b; part[2][ch] = d; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float t[3] = {0.0F, 0.0F, 0.0F};
        for (int s = 0; s < 3; ++s)
            for (std::uint32_t ch = 0; ch < chunks; ++ch) t[s] = t[s] + part[s][ch];
        const float rh = declared_rsqrt(t[0] / static_cast<float>(hidden) + eps);
        const float rk = declared_rsqrt(t[1] / static_cast<float>(hidden) + eps);
        const float dot = (t[2] * (rh * rk)) * inv_sqrt_hidden;
        gate_s = engram_gate_of(dot);
    }
    __syncthreads();
    const float g = gate_s;
    for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) h[i] = h[i] + g * value[i];
}

std::size_t align256(std::size_t n) { return (n + 255U) / 256U * 256U; }

// ---- router selection (ds41_core_moe.cu) ----------------------------------------------------------------------------
constexpr std::uint32_t kMaxExpertsPerLane = 16;

using ::strix::cuda::core::better;

// One warp per row. Expert e = lane + 32 t. Step k: f = first untaken expert; if its key is NaN it is taken (nothing
// compares greater), else the largest non-NaN untaken key, lowest id on ties: the sequential reference's choice.
__global__ void router_select_kernel(const float* __restrict__ scores, std::uint32_t experts, const float* __restrict__ bias,
                                     const float* __restrict__ bias_image, const std::uint8_t* __restrict__ image_rows,
                                     std::uint32_t top_k, float route_scale, std::uint32_t* __restrict__ ids,
                                     float* __restrict__ weights) {
    const std::uint32_t r = blockIdx.x, lane = threadIdx.x;
    const float* s = scores + std::size_t{r} * experts;
    const float* b = image_rows && image_rows[r] ? bias_image : bias;
    float key[kMaxExpertsPerLane];
    std::uint32_t taken = 0;
#pragma unroll
    for (std::uint32_t t = 0; t < kMaxExpertsPerLane; ++t) {
        const std::uint32_t e = lane + 32U * t;
        key[t] = e < experts ? s[e] + b[e] : 0.0F;
    }
    std::uint32_t chosen[32];
    for (std::uint32_t k = 0; k < top_k; ++k) {
        std::uint32_t first = 0xffffffffU, best_i = 0xffffffffU;
        float first_v = 0.0F, best_v = -INFINITY;
        bool have = false;
#pragma unroll
        for (std::uint32_t t = 0; t < kMaxExpertsPerLane; ++t) {
            const std::uint32_t e = lane + 32U * t;
            if (e >= experts || (taken >> t) & 1U) continue;
            if (first == 0xffffffffU) { first = e; first_v = key[t]; }
            if (key[t] != key[t]) continue;
            if (!have || better(key[t], e, best_v, best_i)) { best_v = key[t]; best_i = e; have = true; }
        }
        for (unsigned d = 16U; d; d >>= 1U) {
            const std::uint32_t of = __shfl_xor_sync(full_mask, first, d);
            const float ofv = __shfl_xor_sync(full_mask, first_v, d);
            if (of < first) { first = of; first_v = ofv; }
            const std::uint32_t oi = __shfl_xor_sync(full_mask, best_i, d);
            const float ov = __shfl_xor_sync(full_mask, best_v, d);
            const bool oh = __shfl_xor_sync(full_mask, have ? 1 : 0, d) != 0;
            if (oh && (!have || better(ov, oi, best_v, best_i))) { best_v = ov; best_i = oi; have = true; }
        }
        const std::uint32_t pick = first_v != first_v || !have ? first : best_i;
        chosen[k] = pick;
        if (pick % 32U == lane) taken |= 1U << (pick / 32U);
    }
    if (lane == 0) {
        float sum = 0.0F;
        for (std::uint32_t k = 0; k < top_k; ++k) sum = sum + s[chosen[k]];
        const float denominator = sum + 1e-20F;
        for (std::uint32_t k = 0; k < top_k; ++k) {
            const float sc = s[chosen[k]];
            ids[std::size_t{r} * top_k + k] = chosen[k];
            weights[std::size_t{r} * top_k + k] = (top_k > 1U ? sc / denominator : sc) * route_scale;
        }
    }
}

}  // namespace

void enqueue_lane_dot(const LaneDotView& v, cudaStream_t stream) {
    require(v.job_count > 0 && v.job_count <= 4U && v.rows > 0, "lane dot: 1..4 jobs, rows > 0");
    for (std::uint32_t j = 0; j < v.job_count; ++j) {
        const LaneDotJob& job = v.jobs[j];
        const std::uint32_t xs = job.x_stride ? job.x_stride : job.width;
        require(job.weight && job.x && job.out && job.count > 0 && job.width > 0 && job.width % kStage == 0 && xs % 4U == 0 &&
                    (reinterpret_cast<std::uintptr_t>(job.weight) & 15U) == 0 && (reinterpret_cast<std::uintptr_t>(job.x) & 15U) == 0,
                "lane dot: width % 128, 16-byte aligned weight and x rows");
    }
    const bool rs = v.epilogue == Epilogue::router_score;
    if (v.type == WeightType::f16) {
        if (rs) launch_lane_dot<std::uint16_t, 1>(v, stream);
        else launch_lane_dot<std::uint16_t, 0>(v, stream);
    } else {
        if (rs) launch_lane_dot<float, 1>(v, stream);
        else launch_lane_dot<float, 0>(v, stream);
    }
}

void enqueue_engram(const EngramView& v, cudaStream_t stream) {
    const std::uint32_t width = v.cols * engram_row_elements;
    require(v.rows > 0 && v.cols > 0 && v.hidden > 0 && (v.hidden + chunk - 1U) / chunk <= kMaxInjectChunks && v.kv.width == width &&
                v.kv.rows == 5U * v.hidden && v.workspace,
            "Engram geometry");
    auto* base = static_cast<std::uint8_t*>(v.workspace);
    float* emb = reinterpret_cast<float*>(base);
    float* kv = v.kv_out ? v.kv_out : reinterpret_cast<float*>(base + align256(std::size_t{v.rows} * width * 4U));
    const std::size_t n = std::size_t{v.rows} * width;
    engram_decode_kernel<<<static_cast<unsigned>((n + 255U) / 256U), 256, 0, stream>>>(v.row_bytes, v.rows * v.cols, emb);
    STRIX_CUDA_CHECK(cudaGetLastError());
    LaneDotView f;
    f.jobs[0] = LaneDotJob{v.kv.data, width, v.kv.rows, emb, width, kv, v.kv.rows};
    f.job_count = 1;
    f.rows = v.rows;
    enqueue_lane_dot(f, stream);
    const float inv_sqrt_hidden = static_cast<float>(1.0 / std::sqrt(static_cast<double>(v.hidden)));
    engram_inject_kernel<<<dim3(v.rows, 4), kInjectThreads, 0, stream>>>(v.stream, kv, v.q_norm, v.k_norm, v.hidden, v.eps,
                                                                         inv_sqrt_hidden);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_router(const RouterView& v, cudaStream_t stream) {
    const std::uint32_t experts = v.weight.rows, hidden = v.weight.width;
    require(v.rows > 0 && experts > 0 && experts <= 32U * kMaxExpertsPerLane && hidden > 0 && v.top_k > 0 && v.top_k <= 32U &&
                v.top_k <= experts && v.scores && v.ids && v.weights && v.bias && (!v.image_rows || v.bias_image),
            "router geometry");
    LaneDotView f;
    f.jobs[0] = LaneDotJob{v.weight.data, hidden, experts, v.x, v.x_stride ? v.x_stride : hidden, v.scores, experts};
    f.job_count = 1;
    f.rows = v.rows;
    f.type = WeightType::f32;
    f.epilogue = Epilogue::router_score;
    enqueue_lane_dot(f, stream);
    router_select_kernel<<<v.rows, 32, 0, stream>>>(v.scores, experts, v.bias, v.bias_image, v.image_rows, v.top_k, v.route_scale,
                                                    v.ids, v.weights);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace strix::ds41::cuda::pk
