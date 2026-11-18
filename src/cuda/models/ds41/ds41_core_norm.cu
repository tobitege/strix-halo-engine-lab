// deepseek41 CUDA layer core: RMSNorm (with the hyper-connection collapse and A8 fused), hc mixes with the Sinkhorn
// split, and the hyper-connection expansion. Bitwise the host reference of ds41_layer.hpp.

#include "../../arch/capability.hpp"
#include "ds41_cuda_common.cuh"

#include "strix/cuda/ds41_core.hpp"

namespace strix::ds41::cuda {
namespace {

using detail::chunk;
using detail::full_mask;
using detail::require;

constexpr std::uint32_t kNormThreads = 256, kMaxNorm = 8192, kMaxChunks = kMaxNorm / chunk + 1;

// grid (parts, rows). Every block builds the whole input row in shared memory (the collapse is recomputed per block)
// and the chunk sums of squares; part b writes the outputs of [b * span, (b + 1) * span) (span a multiple of 32, so
// whole A8 groups). One part per row when the norm runs in place.
__device__ __forceinline__ void norm_body(std::uint32_t r, std::uint32_t part_index, const float* __restrict__ x,
            std::uint32_t x_stride, const float* __restrict__ hc, const float* __restrict__ pre,
            std::uint32_t pre_stride, const float* __restrict__ weight, std::uint32_t n, float eps, float* out, std::uint32_t out_stride,
            float* __restrict__ collapsed, std::int8_t* __restrict__ codes, float* __restrict__ deltas, std::int32_t* __restrict__ sums,
            std::uint32_t a8_stride, std::uint32_t span) {
    extern __shared__ float xs[];
    __shared__ float part[kMaxChunks];
    __shared__ float rr;
    const std::uint32_t tid = threadIdx.x, lane = tid & 31U, warp = tid / 32U, warps = blockDim.x / 32U;
    const std::uint32_t lo = part_index * span, hi = lo + span < n ? lo + span : n;
    float p[4] = {1.0F, 0.0F, 0.0F, 0.0F};
    if (hc && pre)
        for (int c = 0; c < 4; ++c) p[c] = pre[std::size_t{r} * pre_stride + c];
    // 4 elements per step with every load of the step issued first (the collapse per element is unchanged)
    std::uint32_t i0 = tid;
    for (; i0 + 3U * blockDim.x < n; i0 += 4U * blockDim.x) {
        float v[4][4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const std::uint32_t i = i0 + u * blockDim.x;
            if (hc) {
                const float* s = hc + std::size_t{r} * 4U * n + i;
#pragma unroll
                for (int c = 0; c < 4; ++c) v[u][c] = s[std::size_t(c) * n];
            } else {
                v[u][0] = x[std::size_t{r} * x_stride + i];
            }
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const std::uint32_t i = i0 + u * blockDim.x;
            float y;
            if (hc) {
                y = 0.0F;
                for (int c = 0; c < 4; ++c) y = y + p[c] * v[u][c];
                if (collapsed && i >= lo && i < hi) collapsed[std::size_t{r} * n + i] = y;
            } else {
                y = v[u][0];
            }
            xs[i] = y;
        }
    }
    for (std::uint32_t i = i0; i < n; i += blockDim.x) {
        float y;
        if (hc) {
            const float* s = hc + std::size_t{r} * 4U * n + i;
            y = 0.0F;
            for (int c = 0; c < 4; ++c) y = y + p[c] * s[std::size_t(c) * n];
            if (collapsed && i >= lo && i < hi) collapsed[std::size_t{r} * n + i] = y;
        } else {
            y = x[std::size_t{r} * x_stride + i];
        }
        xs[i] = y;
    }
    __syncthreads();
    if (!weight) return;
    const std::uint32_t chunks = (n + chunk - 1U) / chunk;
    for (std::uint32_t c = warp; c < chunks; c += warps) {
        const float* xc = xs + c * chunk;
        const std::uint32_t len = n - c * chunk < chunk ? n - c * chunk : chunk;
        const float s = detail::lane_dot([&](std::uint32_t i) { return xc[i]; }, [&](std::uint32_t i) { return xc[i]; }, len, lane);
        if (lane == 0) part[c] = s;
    }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0F;
        for (std::uint32_t c = 0; c < chunks; ++c) total = total + part[c];
        const float mean = total / static_cast<float>(n);
        rr = declared_rsqrt(mean + eps);
    }
    __syncthreads();
    const float rv = rr;
    for (std::uint32_t i = lo + tid; i < hi; i += blockDim.x) {
        const float y = weight[i] * (xs[i] * rv);
        xs[i] = y;
        if (out) out[std::size_t{r} * out_stride + i] = y;
    }
    if (!codes) return;
    __syncthreads();
    detail::a8_row(xs + lo, hi - lo, codes + std::size_t{r} * a8_stride + lo, deltas + (std::size_t{r} * a8_stride + lo) / 32U,
                   sums + (std::size_t{r} * a8_stride + lo) / 32U);
}

__global__ void __launch_bounds__(kNormThreads) norm_kernel(NormView n, std::uint32_t span) {
    // no early PDL trigger: a dependent GEMV launched at this kernel's entry sat resident beside it and slowed this
    // latency-bound kernel ~2x on the PRO; every CTA triggers implicitly when it exits
    norm_body(blockIdx.y, blockIdx.x, n.x, n.x_stride ? n.x_stride : n.n, n.hc_stream, n.pre, n.pre_stride,
              n.weight, n.n, n.eps, n.out, n.out_stride ? n.out_stride : n.n, n.collapsed,
              n.a8.codes, n.a8.deltas, n.a8.sums, n.a8.stride ? n.a8.stride : n.n, span);
}

// ---- hc mixes ------------------------------------------------------------------------------------------------------
constexpr std::uint32_t kDots = hc_mix_count + 1U;  // 24 fn rows + the sum of squares

// grid (chunks, rows, kDots / W), W warps: dot m = z * W + warp; m < 24 the chunk dot of fn row m with the stream,
// m = 24 the stream with itself (one warp per dot and chunk, whatever W). W = 25 is one block per chunk (32 blocks at
// decode, most SMs idle); W = 5 gives 5 x the blocks.
template <std::uint32_t W>
__global__ void __launch_bounds__(W * 32)
hc_partial_kernel(const float* __restrict__ stream, const std::uint16_t* __restrict__ fn, std::uint32_t total_n,
                  float* __restrict__ partial) {
    static_assert(kDots % W == 0, "hc dots per block");
    const std::uint32_t c = blockIdx.x, r = blockIdx.y, chunks = gridDim.x, m = blockIdx.z * W + threadIdx.x / 32U,
                        lane = threadIdx.x & 31U;
    const std::uint32_t c0 = c * chunk, len = total_n - c0 < chunk ? total_n - c0 : chunk;
    const float* s = stream + std::size_t{r} * total_n + c0;
    float v;
    if (len == chunk) {  // a whole chunk: the lane's 20 pairs loaded first, then the chain in ascending order
        constexpr std::uint32_t kPer = chunk / 32U;
        float av[kPer], bv[kPer];
        std::uint16_t wr[kPer];
        const std::uint16_t* w = fn + std::size_t{m < hc_mix_count ? m : 0U} * total_n + c0;
        // every load first (raw halves), then the conversions: a conversion next to its conditional load made each
        // load its own branch region on sm_120
#pragma unroll
        for (std::uint32_t k = 0; k < kPer; ++k) {
            bv[k] = s[lane + 32U * k];
            wr[k] = m < hc_mix_count ? w[lane + 32U * k] : std::uint16_t{0};
        }
#pragma unroll
        for (std::uint32_t k = 0; k < kPer; ++k) av[k] = m < hc_mix_count ? f16_to_float(wr[k]) : bv[k];
        float acc = 0.0F;
#pragma unroll
        for (std::uint32_t k = 0; k < kPer; ++k) acc = acc + av[k] * bv[k];
        v = detail::butterfly(acc);
    } else if (m < hc_mix_count) {
        const std::uint16_t* w = fn + std::size_t{m} * total_n + c0;
        v = detail::lane_dot([&](std::uint32_t i) { return f16_to_float(w[i]); }, [&](std::uint32_t i) { return s[i]; }, len, lane);
    } else {
        v = detail::lane_dot([&](std::uint32_t i) { return s[i]; }, [&](std::uint32_t i) { return s[i]; }, len, lane);
    }
    if (lane == 0) partial[(std::size_t{r} * kDots + m) * chunks + c] = v;
}

// One warp per row: lanes 0..24 sum their dot's chunks in ascending order; then the split (hc_split_sinkhorn), with
// lane t = 4 j + k (t < 16) holding comb[j][k]; row and column sums are taken in ascending index by every lane.
__device__ __forceinline__ void hc_split_body(std::uint32_t r, const float* __restrict__ partial, std::uint32_t chunks, std::uint32_t total_n,
                                const float* __restrict__ base, const float* __restrict__ scale, std::uint32_t iters, float hc_eps,
                                float norm_eps, float* __restrict__ mixes) {
    const std::uint32_t lane = threadIdx.x & 31U;
    float total = 0.0F;
    if (lane < kDots) {
        const float* p = partial + (std::size_t{r} * kDots + lane) * chunks;
        std::uint32_t c = 0;
        for (; c + 8U <= chunks; c += 8U) {  // 8 loads issued, then added in ascending order
            float v[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) v[k] = p[c + k];
#pragma unroll
            for (int k = 0; k < 8; ++k) total = total + v[k];
        }
        for (; c < chunks; ++c) total = total + p[c];
    }
    const float sq = __shfl_sync(full_mask, total, hc_mix_count);
    const float rs = declared_rsqrt(sq / static_cast<float>(total_n) + norm_eps);
    const float m = total * rs;  // lane m < 24: mixes[m]
    float* out = mixes + std::size_t{r} * hc_mix_count;
    if (lane < 4U) out[lane] = declared_sigmoid(m * scale[0] + base[lane]) + hc_eps;
    else if (lane < 8U) out[lane] = 2.0F * declared_sigmoid(m * scale[1] + base[lane]);
    // comb: lane t < 16 takes mixes[8 + t]
    const std::uint32_t t = lane & 15U, j = t / 4U, k = t % 4U;
    float cv = __shfl_sync(full_mask, m, 8U + t);
    cv = cv * scale[2] + base[8U + t];
    // row softmax: max over the row (first then ascending), exp(c - max), ascending sum, c / sum + eps
    float row[4];
    for (std::uint32_t q = 0; q < 4U; ++q) row[q] = __shfl_sync(full_mask, cv, 4U * j + q);
    float mx = row[0];
    for (std::uint32_t q = 1; q < 4U; ++q) mx = row[q] > mx ? row[q] : mx;
    cv = declared_exp(cv - mx);
    for (std::uint32_t q = 0; q < 4U; ++q) row[q] = __shfl_sync(full_mask, cv, 4U * j + q);
    float sum = 0.0F;
    for (std::uint32_t q = 0; q < 4U; ++q) sum = sum + row[q];
    cv = cv / sum + hc_eps;
    auto columns = [&] {
        float s = 0.0F;
        for (std::uint32_t q = 0; q < 4U; ++q) s = s + __shfl_sync(full_mask, cv, 4U * q + k);
        cv = cv / (s + hc_eps);
    };
    auto rows = [&] {
        float s = 0.0F;
        for (std::uint32_t q = 0; q < 4U; ++q) s = s + __shfl_sync(full_mask, cv, 4U * j + q);
        cv = cv / (s + hc_eps);
    };
    columns();
    for (std::uint32_t it = 1; it < iters; ++it) { rows(); columns(); }
    if (lane < 16U) out[8U + lane] = cv;
}

__global__ void hc_split_kernel(HcMixView m, std::uint32_t chunks) {
    hc_split_body(blockIdx.x, m.partial, chunks, 4U * m.hidden, m.base, m.scale, m.iters, m.hc_eps, m.norm_eps, m.mixes);
}

// Split and norm use independent pre coefficients. Block 0 does the Sinkhorn split; the other blocks
// do exactly the legacy norm parts. No block consumes another block's writes in this launch.
__global__ void __launch_bounds__(kNormThreads) hc_split_norm_kernel(HcMixView m, NormView n, std::uint32_t chunks) {
    // no early PDL trigger: a dependent GEMV launched at this kernel's entry sat resident beside it and slowed this
    // latency-bound kernel ~2x on the PRO; every CTA triggers implicitly when it exits
    if (blockIdx.x == 0U) {
        if (threadIdx.x < 32U)
            hc_split_body(blockIdx.y, m.partial, chunks, 4U * m.hidden, m.base, m.scale, m.iters, m.hc_eps, m.norm_eps, m.mixes);
    } else {
        norm_body(blockIdx.y, blockIdx.x - 1U, n.x, n.x_stride ? n.x_stride : n.n, n.hc_stream, n.pre, n.pre_stride,
                  n.weight, n.n, n.eps, n.out, n.out_stride ? n.out_stride : n.n, n.collapsed,
                  n.a8.codes, n.a8.deltas, n.a8.sums, n.a8.stride ? n.a8.stride : n.n, chunk);
    }
}

// sm_90+: the norm parts of a row as one thread block cluster (part b = chunk b of the row), plus a split warp per block
// (block 0's runs the Sinkhorn split as hc_split_norm_kernel's block 0 does). Part b collapses only its own chunk, takes
// that chunk's sum of squares (norm_body's lane_dot) and stores it into every part's shared memory; after one cluster
// barrier every thread adds the chunk sums in ascending order as norm_body's thread 0 does. Same bits as
// hc_split_norm_kernel, without every part re-reading the whole 4-stream row (9 blocks each reading 80 KB at decode).
// Cluster barrier phase 1 (arrive at entry, wait before the first remote store) establishes that every part has started;
// phase 2 (release / acquire) publishes the stores. The norm threads' CTA barrier is the non-aligned barrier.sync with a
// count of 256: the split warp never reaches it.
constexpr std::uint32_t kClusterParts = 8, kClusterPer = (chunk + kNormThreads - 1U) / kNormThreads;
__global__ void __launch_bounds__(kNormThreads + 32U) hc_split_norm_cluster_kernel(HcMixView m, NormView n, std::uint32_t chunks) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // no early PDL trigger (see norm_kernel): every CTA triggers implicitly when it exits
    __shared__ float xs[chunk];
    __shared__ float sq[kClusterParts];
    const std::uint32_t tid = threadIdx.x, b = blockIdx.x, r = blockIdx.y, parts = gridDim.x;
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");  // phase 1: this part has started
    if (tid >= kNormThreads) {  // the split warp: no shared memory of the cluster is touched
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
        if (b == 0U) hc_split_body(r, m.partial, chunks, 4U * m.hidden, m.base, m.scale, m.iters, m.hc_eps, m.norm_eps, m.mixes);
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        return;
    }
    const std::uint32_t lane = tid & 31U, warp = tid / 32U, nn = n.n, lo = b * chunk, hi = lo + chunk < nn ? lo + chunk : nn,
                        len = hi - lo;
    const std::uint32_t x8 = n.a8.stride ? n.a8.stride : nn, os = n.out_stride ? n.out_stride : nn;
    float p[4];
    for (int c = 0; c < 4; ++c) p[c] = n.pre[std::size_t{r} * n.pre_stride + c];
    const float* srow = n.hc_stream + std::size_t{r} * 4U * nn;
    float v[kClusterPer][4];
#pragma unroll
    for (std::uint32_t u = 0; u < kClusterPer; ++u) {
        const std::uint32_t i = lo + tid + u * kNormThreads;
#pragma unroll
        for (int c = 0; c < 4; ++c) v[u][c] = i < hi ? srow[std::size_t(c) * nn + i] : 0.0F;
    }
#pragma unroll
    for (std::uint32_t u = 0; u < kClusterPer; ++u) {
        const std::uint32_t i = lo + tid + u * kNormThreads;
        if (i >= hi) break;
        float y = 0.0F;
        for (int c = 0; c < 4; ++c) y = y + p[c] * v[u][c];
        if (n.collapsed) n.collapsed[std::size_t{r} * nn + i] = y;
        xs[i - lo] = y;
    }
    asm volatile("barrier.sync 1, %0;" ::"n"(kNormThreads) : "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");  // phase 1: every part of the cluster runs
    if (warp == 0U) {
        const float part = detail::lane_dot([&](std::uint32_t i) { return xs[i]; }, [&](std::uint32_t i) { return xs[i]; }, len, lane);
        if (lane < parts) {  // lane q stores this chunk's sum into part q's sq[b]
            const std::uint32_t local = static_cast<std::uint32_t>(__cvta_generic_to_shared(&sq[b]));
            std::uint32_t remote;
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(lane));
            asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(remote), "f"(part) : "memory");
        }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    float total = 0.0F;
    for (std::uint32_t c = 0; c < parts; ++c) total = total + sq[c];
    const float mean = total / static_cast<float>(nn);
    const float rv = declared_rsqrt(mean + n.eps);
    for (std::uint32_t i = lo + tid; i < hi; i += kNormThreads) {
        const float y = n.weight[i] * (xs[i - lo] * rv);
        xs[i - lo] = y;
        if (n.out) n.out[std::size_t{r} * os + i] = y;
    }
    if (!n.a8.codes) return;
    asm volatile("barrier.sync 1, %0;" ::"n"(kNormThreads) : "memory");
    for (std::uint32_t g = warp; g < len / 32U; g += kNormThreads / 32U) {
        float delta;
        std::int32_t sum;
        const auto code = detail::a8_group(xs[32U * g + lane], delta, sum);
        const std::size_t at = std::size_t{r} * x8 + lo + 32U * g;
        n.a8.codes[at + lane] = static_cast<std::int8_t>(code);
        if (lane == 0) { n.a8.deltas[at / 32U] = delta; n.a8.sums[at / 32U] = sum; }
    }
#endif
}

bool g_hc_cluster = false;
std::uint32_t g_hc_dot_warps = kDots;

void launch_hc_partial(const float* stream_in, const std::uint16_t* fn, std::uint32_t total_n, std::uint32_t chunks,
                       std::uint32_t rows, float* partial, cudaStream_t stream) {
    if (g_hc_dot_warps == 5U) hc_partial_kernel<5><<<dim3(chunks, rows, kDots / 5U), 5U * 32U, 0, stream>>>(stream_in, fn, total_n, partial);
    else hc_partial_kernel<kDots><<<dim3(chunks, rows, 1U), kDots * 32U, 0, stream>>>(stream_in, fn, total_n, partial);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

// ---- hc expand -----------------------------------------------------------------------------------------------------
__global__ void hc_expand_kernel(const float* __restrict__ sub, std::uint32_t sub_stride, const float* residual,
                                 const float* __restrict__ mixes, std::uint32_t rows, std::uint32_t hidden, float* out, float* pre_carry) {
    const std::size_t idx = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (idx >= std::size_t{rows} * hidden) return;
    const std::uint32_t r = static_cast<std::uint32_t>(idx / hidden), i = static_cast<std::uint32_t>(idx % hidden);
    const float* mix = mixes + std::size_t{r} * hc_mix_count;
    if (pre_carry && i == 0U)
        for (int c = 0; c < 4; ++c) pre_carry[std::size_t{r} * 4U + c] = mix[c];
    const float sv = sub[std::size_t{r} * sub_stride + i];
    float res[4];
    for (int j = 0; j < 4; ++j) res[j] = residual[(std::size_t{r} * 4U + j) * hidden + i];
    float y[4];
    for (int k = 0; k < 4; ++k) {
        float s = 0.0F;
        for (int j = 0; j < 4; ++j) s = s + mix[8 + j * 4 + k] * res[j];
        y[k] = mix[4 + k] * sv + s;
    }
    for (int k = 0; k < 4; ++k) out[(std::size_t{r} * 4U + k) * hidden + i] = y[k];
}

}  // namespace

void enqueue_norm(const NormView& v, cudaStream_t stream) {
    require(v.rows > 0 && v.n > 0 && v.n % 32U == 0 && v.n <= kMaxNorm && (v.x || v.hc_stream), "norm geometry");
    require(v.weight || (!v.out && !v.a8.codes), "norm outputs need a weight");
    require(!v.hc_stream || !v.pre || v.pre_stride >= 4U, "norm pre stride");
    // in place (out aliases x): one block per row; otherwise one block per 640-value part of the row
    const bool in_place = v.x && !v.hc_stream && v.out == v.x;
    const std::uint32_t span = in_place ? v.n : chunk, parts = (v.n + span - 1U) / span;
    norm_kernel<<<dim3(parts, v.rows), kNormThreads, v.n * sizeof(float), stream>>>(v, span);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

std::size_t hc_mix_workspace_bytes(std::uint32_t rows, std::uint32_t hidden) {
    return std::size_t{rows} * kDots * ((4U * hidden + chunk - 1U) / chunk) * sizeof(float);
}

void enqueue_hc_mixes(const HcMixView& v, cudaStream_t stream) {
    const std::uint32_t total_n = 4U * v.hidden, chunks = (total_n + chunk - 1U) / chunk;
    require(v.rows > 0 && v.hidden > 0 && v.iters >= 1 && v.partial && v.mixes, "hc mix geometry");
    launch_hc_partial(v.stream, v.fn, total_n, chunks, v.rows, v.partial, stream);
    hc_split_kernel<<<v.rows, 32, 0, stream>>>(v, chunks);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_hc_mixes_norm(const HcMixView& m, const NormView& n, cudaStream_t stream) {
    require(m.rows > 0 && m.hidden > 0 && m.iters >= 1 && m.partial && m.mixes && m.stream && m.fn && m.base && m.scale,
            "hc mix norm geometry");
    require(n.rows == m.rows && n.n == m.hidden && n.n % 32U == 0 && n.n <= kMaxNorm &&
                n.hc_stream == m.stream && n.pre && n.pre != m.mixes && n.pre_stride >= 4U && n.weight,
            "hc mix norm needs independent pre coefficients and matching stream");
    const std::uint32_t total_n = 4U * m.hidden, chunks = (total_n + chunk - 1U) / chunk;
    launch_hc_partial(m.stream, m.fn, total_n, chunks, m.rows, m.partial, stream);
    const std::uint32_t parts = (n.n + chunk - 1U) / chunk;
    // set_hc_norm_cluster(true) on compute capability 9.0+ (cached per device ordinal; API errors fail closed)
    const bool use = g_hc_cluster && parts <= kClusterParts && ::strix::cuda::arch::has_clusters();
    if (use) {
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(parts, n.rows);
        cfg.blockDim = dim3(kNormThreads + 32U);
        cfg.stream = stream;
        cudaLaunchAttribute attr{};
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim.x = parts;
        attr.val.clusterDim.y = 1;
        attr.val.clusterDim.z = 1;
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        STRIX_CUDA_CHECK(cudaLaunchKernelEx(&cfg, hc_split_norm_cluster_kernel, m, n, chunks));
        return;
    }
    hc_split_norm_kernel<<<dim3(1U + parts, n.rows), kNormThreads,
                           n.n * sizeof(float), stream>>>(m, n, chunks);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void set_hc_norm_cluster(bool on) { g_hc_cluster = on; }
void set_hc_partial_warps(unsigned warps) { g_hc_dot_warps = warps == 5U ? 5U : kDots; }

void enqueue_hc_expand(const HcExpandView& v, cudaStream_t stream) {
    require(v.rows > 0 && v.hidden > 0, "hc expand geometry");
    const std::size_t n = std::size_t{v.rows} * v.hidden;
    hc_expand_kernel<<<static_cast<unsigned>((n + 255U) / 256U), 256, 0, stream>>>(v.sub, v.sub_stride ? v.sub_stride : v.hidden,
                                                                                   v.residual, v.mixes, v.rows, v.hidden, v.out, v.pre_carry);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace strix::ds41::cuda
