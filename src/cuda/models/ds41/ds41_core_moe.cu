// deepseek41 CUDA layer core: router (scores, biased top-k, weights) and the Q8_0 shared expert. Bitwise the host
// reference of ds41_moe.hpp (reference_router, reference_window with one Q8_0 expert).

#include "../../core/reduce.cuh"
#include "ds41_cuda_common.cuh"

#include "strix/cuda/ds41_core.hpp"

#include <cmath>

namespace strix::ds41::cuda {
namespace {

using detail::full_mask;
using detail::require;

constexpr std::uint32_t kRouterWarps = 8, kMaxExpertsPerLane = 16;

// One warp per expert and up to 8 rows: the weight row is read once; per row the lane dot (lane l: elements l, l + 32,
// ... ascending, acc = acc + w * x from 0, then the butterfly), then router_score.
template <int RR>
__global__ void __launch_bounds__(kRouterWarps * 32)
router_scores_kernel(const float* __restrict__ x, std::uint32_t x_stride, const float* __restrict__ w, std::uint32_t experts,
                     std::uint32_t hidden, float* __restrict__ scores) {
    const std::uint32_t e = blockIdx.x * kRouterWarps + threadIdx.x / 32U, lane = threadIdx.x & 31U;
    if (e >= experts) return;
    const float* wr = w + std::size_t{e} * hidden;
    float acc[RR];
#pragma unroll
    for (int r = 0; r < RR; ++r) acc[r] = 0.0F;
#pragma unroll 8
    for (std::uint32_t i = lane; i < hidden; i += 32U) {
        const float wv = __ldg(wr + i);
#pragma unroll
        for (int r = 0; r < RR; ++r) acc[r] = acc[r] + wv * x[std::size_t(r) * x_stride + i];
    }
#pragma unroll
    for (int r = 0; r < RR; ++r) {
        const float z = detail::butterfly(acc[r]);
        if (lane == 0) scores[std::size_t(r) * experts + e] = router_score(z);
    }
}

template <int RR>
void launch_router_scores(const RouterView& v, std::uint32_t x_stride, std::uint32_t r0, cudaStream_t stream) {
    const std::uint32_t experts = v.weight.rows;
    router_scores_kernel<RR><<<(experts + kRouterWarps - 1U) / kRouterWarps, kRouterWarps * 32U, 0, stream>>>(
        v.x + std::size_t{r0} * x_stride, x_stride, v.weight.data, experts, v.weight.width, v.scores + std::size_t{r0} * experts);
}

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
        // warp: the lowest first index (with its key), and the best non-NaN candidate
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

constexpr std::uint32_t kSwigluThreads = 256, kMaxInter = 8192;

__global__ void __launch_bounds__(kSwigluThreads)
swiglu_a8_kernel(const float* __restrict__ gate, const float* __restrict__ up, std::uint32_t inter, std::uint32_t in_stride, float limit,
                 float* __restrict__ act,
                 std::int8_t* __restrict__ codes, float* __restrict__ deltas, std::int32_t* __restrict__ sums, std::uint32_t a8_stride) {
    // no early PDL trigger: a dependent GEMV launched at this kernel's entry sat resident beside it and slowed this
    // latency-bound kernel ~2x on the PRO; every CTA triggers implicitly when it exits
    extern __shared__ float ys[];
    const std::uint32_t r = blockIdx.x;
    for (std::uint32_t i = threadIdx.x; i < inter; i += blockDim.x) {
        const float y = swiglu_clamped(gate[std::size_t{r} * in_stride + i], up[std::size_t{r} * in_stride + i], limit);
        ys[i] = y;
        if (act) act[std::size_t{r} * inter + i] = y;
    }
    __syncthreads();
    detail::a8_row(ys, inter, codes + std::size_t{r} * a8_stride, deltas + std::size_t{r} * a8_stride / 32U,
                   sums + std::size_t{r} * a8_stride / 32U);
}

// One warp per 32-value group (block = 8 groups of a row): the same swiglu_clamped per value and a8_group per group as
// swiglu_a8_kernel's a8_row, spread over rows x inter / 256 blocks instead of one block per row.
__global__ void __launch_bounds__(256)
swiglu_a8_wide_kernel(const float* __restrict__ gate, const float* __restrict__ up, std::uint32_t inter, std::uint32_t in_stride, float limit,
                      float* __restrict__ act, std::int8_t* __restrict__ codes, float* __restrict__ deltas, std::int32_t* __restrict__ sums,
                      std::uint32_t a8_stride) {
    // no early PDL trigger: a dependent GEMV launched at this kernel's entry sat resident beside it and slowed this
    // latency-bound kernel ~2x on the PRO; every CTA triggers implicitly when it exits
    const std::uint32_t r = blockIdx.y, g = blockIdx.x * 8U + threadIdx.x / 32U, lane = threadIdx.x & 31U;
    if (g >= inter / 32U) return;
    const std::uint32_t i = 32U * g + lane;
    const float y = swiglu_clamped(gate[std::size_t{r} * in_stride + i], up[std::size_t{r} * in_stride + i], limit);
    if (act) act[std::size_t{r} * inter + i] = y;
    float delta;
    std::int32_t sum;
    const auto code = detail::a8_group(y, delta, sum);
    codes[std::size_t{r} * a8_stride + i] = static_cast<std::int8_t>(code);
    if (lane == 0) {
        deltas[std::size_t{r} * a8_stride / 32U + g] = delta;
        sums[std::size_t{r} * a8_stride / 32U + g] = sum;
    }
}

struct SharedScratch {
    float* gate;
    float* up;
    std::int8_t* codes;
    float* deltas;
    std::int32_t* sums;
};
std::size_t align256(std::size_t n) { return (n + 255U) / 256U * 256U; }
// [gate | up] floats (2 rows inter: [row][inter] each, or [row][2 inter] for the tiled form), then the A8 of the
// activation (codes, deltas, sums).
SharedScratch shared_scratch(void* base, std::uint32_t rows, std::uint32_t inter) {
    auto* p = static_cast<std::uint8_t*>(base);
    const std::size_t f = align256(2U * std::size_t{rows} * inter * 4U), c = align256(std::size_t{rows} * inter),
                      g = align256(std::size_t{rows} * inter / 32U * 4U);
    return {reinterpret_cast<float*>(p), reinterpret_cast<float*>(p) + std::size_t{rows} * inter, reinterpret_cast<std::int8_t*>(p + f),
            reinterpret_cast<float*>(p + f + c), reinterpret_cast<std::int32_t*>(p + f + c + g)};
}

}  // namespace

void enqueue_router(const RouterView& v, cudaStream_t stream) {
    const std::uint32_t experts = v.weight.rows, hidden = v.weight.width;
    require(v.rows > 0 && experts > 0 && experts <= 32U * kMaxExpertsPerLane && hidden > 0 && v.top_k > 0 && v.top_k <= 32U &&
                v.top_k <= experts && v.scores && v.ids && v.weights && v.bias && (!v.image_rows || v.bias_image),
            "router geometry");
    const std::uint32_t x_stride = v.x_stride ? v.x_stride : hidden;
    for (std::uint32_t r0 = 0; r0 < v.rows; r0 += 8U) {
        switch (v.rows - r0 < 8U ? v.rows - r0 : 8U) {
        case 1: launch_router_scores<1>(v, x_stride, r0, stream); break;
        case 2: launch_router_scores<2>(v, x_stride, r0, stream); break;
        case 3: launch_router_scores<3>(v, x_stride, r0, stream); break;
        case 4: launch_router_scores<4>(v, x_stride, r0, stream); break;
        case 5: launch_router_scores<5>(v, x_stride, r0, stream); break;
        case 6: launch_router_scores<6>(v, x_stride, r0, stream); break;
        case 7: launch_router_scores<7>(v, x_stride, r0, stream); break;
        default: launch_router_scores<8>(v, x_stride, r0, stream); break;
        }
        STRIX_CUDA_CHECK(cudaGetLastError());
    }
    router_select_kernel<<<v.rows, 32, 0, stream>>>(v.scores, experts, v.bias, v.bias_image, v.image_rows, v.top_k, v.route_scale,
                                                    v.ids, v.weights);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_swiglu_a8(const SwigluA8View& v, cudaStream_t stream) {
    require(v.rows > 0 && v.inter > 0 && v.inter % 32U == 0 && v.inter <= kMaxInter && v.a8.codes, "SwiGLU A8 geometry");
    if (v.a8.sums) {  // the per-group form (sums are always written there)
        swiglu_a8_wide_kernel<<<dim3((v.inter / 32U + 7U) / 8U, v.rows), 256, 0, stream>>>(
            v.gate, v.up, v.inter, v.in_stride ? v.in_stride : v.inter, v.limit, v.act, v.a8.codes, v.a8.deltas, v.a8.sums,
            v.a8.stride ? v.a8.stride : v.inter);
        STRIX_CUDA_CHECK(cudaGetLastError());
        return;
    }
    swiglu_a8_kernel<<<v.rows, kSwigluThreads, v.inter * sizeof(float), stream>>>(v.gate, v.up, v.inter,
                                                                                 v.in_stride ? v.in_stride : v.inter, v.limit, v.act, v.a8.codes,
                                                                                 v.a8.deltas, v.a8.sums, v.a8.stride ? v.a8.stride : v.inter);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

std::size_t shared_expert_workspace_bytes(std::uint32_t rows, std::uint32_t inter) {
    return align256(2U * std::size_t{rows} * inter * 4U) + align256(std::size_t{rows} * inter) +
           2U * align256(std::size_t{rows} * inter / 32U * 4U);
}

void enqueue_shared_expert(const SharedExpertView& v, cudaStream_t stream) {
    const bool tiled = v.gate_up_tiled.codes != nullptr;
    const std::uint32_t rows = v.x.rows, inter = tiled ? v.gate_up_tiled.rows / 2U : v.gate.rows,
                        hidden = tiled ? v.gate_up_tiled.width : v.gate.width;
    require(rows > 0 && v.workspace && v.x.width == hidden &&
                (tiled ? v.down_tiled.codes && v.down_tiled.rows == hidden && v.down_tiled.width == inter
                       : v.up.rows == inter && v.up.width == hidden && v.down.rows == hidden && v.down.width == inter),
            "shared expert geometry");
    const auto s = shared_scratch(v.workspace, rows, inter);
    const A8Rows act{s.codes, s.deltas, s.sums, rows, inter, inter};
    Q8A8View d;
    if (tiled) {
        // one site of 2 inter outputs: [row][gate | up]
        Q8A8View g;
        g.tiled = v.gate_up_tiled;
        g.count = 2U * inter;
        g.x = v.x;
        g.out = s.gate;
        g.pdl = v.pdl;
        enqueue_q8_0_a8(g, stream);
        enqueue_swiglu_a8({s.gate, s.gate + inter, rows, inter, 2U * inter, v.limit, nullptr, act}, stream);
        d.tiled = v.down_tiled;
    } else {
        Q8A8View g;
        g.weight = v.gate;
        g.count = inter;
        g.x = v.x;
        g.out = s.gate;
        enqueue_q8_0_a8(g, stream);
        g.weight = v.up;
        g.out = s.up;
        enqueue_q8_0_a8(g, stream);
        enqueue_swiglu_a8({s.gate, s.up, rows, inter, 0, v.limit, nullptr, act}, stream);
        d.weight = v.down;
    }
    d.count = hidden;
    d.x = act;
    d.out = v.out;
    d.out_stride = v.out_stride ? v.out_stride : hidden;
    d.pdl = v.pdl && tiled;
    enqueue_q8_0_a8(d, stream);
}

void enqueue_shared_expert_fused(const SharedExpertView& v, cudaStream_t stream) {
    const std::uint32_t inter = v.gate_up_tiled.rows / 2U, hidden = v.gate_up_tiled.width;
    require(v.workspace && v.gate_up_tiled.codes && v.down_tiled.codes && v.down_tiled.rows == hidden &&
                v.down_tiled.width == inter && inter > 0U && inter % 32U == 0U && inter <= 4096U &&
                v.gate_up_tiled.rows == 2U * inter && v.x.rows > 0U && v.x.width == hidden && v.out &&
                (!v.out_stride || v.out_stride >= hidden), "fused shared expert geometry");
    const auto s = shared_scratch(v.workspace, v.x.rows, inter);
    Q8A8View g;
    g.tiled = v.gate_up_tiled; g.count = 2U * inter; g.x = v.x; g.out = s.gate; g.pdl = v.pdl;
    enqueue_q8_0_a8(g, stream);
    enqueue_swiglu_a8_down(s.gate, v.x.rows, v.limit, v.down_tiled, v.out, v.out_stride, {}, stream);
}

}  // namespace strix::ds41::cuda
