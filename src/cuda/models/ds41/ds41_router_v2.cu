// deepseek41 CUDA router, decode v2 (cuda_ds41_router_v2.hpp): the bits of enqueue_router (ds41_core_moe.cu) with the
// weight rows streamed at full width. One warp per expert copies its F32 row into shared memory by 16-byte cp.async
// (every row of the router in flight at once instead of 48 blocks of one 4-byte load per lane), then each lane runs
// the declared chain over its class from shared memory (elements l, l + 32, ... ascending, acc = acc + w * x from 0),
// the butterfly and router_score. The block that finishes last (a self-resetting arrival counter, threadfence
// publication) runs the top-k selection of router_select_kernel unchanged, one warp per row.
// Form 2 (set_router_v2_form, rows <= 2): the same arithmetic with the loads moved earlier. The finisher's bias row
// (with the image-row flag) is read at the kernel's entry, while the weight rows are in flight, instead of after the
// arrival counter; and the x rows are prefetched into L1 (the weights bypass it: cp.async.cg) during the ~5 us weight
// load, so the score chain's round trips (nvcc unrolls 4 steps: 5 per row) hit L1 instead of L2.
// (Form 1, an x prefetch one step ahead, measured slower on the PRO (scores 14.2 vs 9.8 us): the
// loop-carried registers stopped nvcc's 4-step unroll, so every step waited for its own L2 round trip.)

#include "../../core/ptx.cuh"
#include "../../core/reduce.cuh"
#include "ds41_cuda_common.cuh"

#include "strix/cuda/ds41_router_v2.hpp"

#include <cmath>

namespace strix::ds41::cuda {
namespace {

using detail::full_mask;
using detail::require;

constexpr std::uint32_t kWarpsPerBlock = 2, kMaxExpertsPerLane = 16, kMaxHidden = 8192;

using ::strix::cuda::core::ptx_mem::cp_async16;

using ::strix::cuda::core::better;

// router_select_kernel's body for row r (one warp), scores read through L2 (written by other blocks). PRE: this
// lane's bias values are already in bpre (the values the load below would give)
template <bool PRE>
__device__ __forceinline__ void select_row(std::uint32_t r, const float* __restrict__ scores, std::uint32_t experts,
                                           const float* __restrict__ bias, const float* __restrict__ bias_image,
                                           const std::uint8_t* __restrict__ image_rows, std::uint32_t top_k, float route_scale,
                                           std::uint32_t* __restrict__ ids, float* __restrict__ weights,
                                           const float (&bpre)[kMaxExpertsPerLane]) {
    const std::uint32_t lane = threadIdx.x & 31U;
    const float* s = scores + std::size_t{r} * experts;
    const float* b = PRE ? nullptr : image_rows && image_rows[r] ? bias_image : bias;
    float key[kMaxExpertsPerLane], sc[kMaxExpertsPerLane];
    std::uint32_t taken = 0;
#pragma unroll
    for (std::uint32_t t = 0; t < kMaxExpertsPerLane; ++t) {
        const std::uint32_t e = lane + 32U * t;
        sc[t] = e < experts ? __ldcg(s + e) : 0.0F;
        key[t] = e < experts ? sc[t] + (PRE ? bpre[t] : b[e]) : 0.0F;
    }
    // lane k keeps pick k and its score (selection order), broadcast from the owner lane: no reloads
    std::uint32_t my_id = 0;
    float my_score = 0.0F;
    bool nan_here = false;
#pragma unroll
    for (std::uint32_t t = 0; t < kMaxExpertsPerLane; ++t) nan_here |= lane + 32U * t < experts && key[t] != key[t];
    if (!__any_sync(full_mask, nan_here)) {
        // no NaN key: the reference's pick is the largest untaken key, the lower id on ties (-0 == +0), so one warp
        // max over (order-preserving key bits, ~id) per pick
        std::uint64_t packed[kMaxExpertsPerLane];
#pragma unroll
        for (std::uint32_t t = 0; t < kMaxExpertsPerLane; ++t) {
            const std::uint32_t e = lane + 32U * t;
            const std::uint32_t bits = __float_as_uint(key[t] + 0.0F);  // -0 -> +0: equal keys tie by id
            const std::uint32_t ord = (bits & 0x80000000U) ? ~bits : (bits | 0x80000000U);
            packed[t] = e < experts ? (std::uint64_t{ord} << 32U) | (0xffffffffU - e) : 0ULL;
        }
        for (std::uint32_t k = 0; k < top_k; ++k) {
            std::uint64_t best = 0ULL;
#pragma unroll
            for (std::uint32_t t = 0; t < kMaxExpertsPerLane; ++t)
                if (!((taken >> t) & 1U) && packed[t] > best) best = packed[t];
#pragma unroll
            for (unsigned d = 16U; d; d >>= 1U) {
                const std::uint64_t o = __shfl_xor_sync(full_mask, best, d);
                best = o > best ? o : best;
            }
            const std::uint32_t pick = 0xffffffffU - static_cast<std::uint32_t>(best);
            float own = 0.0F;
#pragma unroll
            for (std::uint32_t t = 0; t < kMaxExpertsPerLane; ++t)
                if (t == pick / 32U) own = sc[t];
            const float picked = __shfl_sync(full_mask, own, pick % 32U);
            if (lane == k) { my_id = pick; my_score = picked; }
            if (pick % 32U == lane) taken |= 1U << (pick / 32U);
        }
    } else
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
        float own = 0.0F;
#pragma unroll
        for (std::uint32_t t = 0; t < kMaxExpertsPerLane; ++t)
            if (t == pick / 32U) own = sc[t];
        const float picked = __shfl_sync(full_mask, own, pick % 32U);
        if (lane == k) { my_id = pick; my_score = picked; }
        if (pick % 32U == lane) taken |= 1U << (pick / 32U);
    }
    float sum = 0.0F;
    for (std::uint32_t k = 0; k < top_k; ++k) sum = sum + __shfl_sync(full_mask, my_score, k);
    const float denominator = sum + 1e-20F;
    if (lane < top_k) {
        ids[std::size_t{r} * top_k + lane] = my_id;
        weights[std::size_t{r} * top_k + lane] = (top_k > 1U ? my_score / denominator : my_score) * route_scale;
    }
}

struct RouterArgs {
    const float* x;
    std::uint32_t x_stride, rows, experts, hidden;
    const float* w;
    float* scores;
    const float* bias;
    const float* bias_image;
    const std::uint8_t* image_rows;
    std::uint32_t top_k;
    float route_scale;
    std::uint32_t* ids;
    float* weights;
    std::uint32_t* counter;  // one word, zero between launches
    unsigned long long* stamps;  // diagnostics: [block][8] globaltimer at start / row loaded / scores / select done /
                                 // finisher past the arrival counter
};
__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

template <int RR, int FORM>
__global__ void __launch_bounds__(kWarpsPerBlock * 32) router_v2_kernel(RouterArgs a) {
    constexpr bool kPre = FORM == 2 && RR <= static_cast<int>(kWarpsPerBlock);
    extern __shared__ __align__(16) float rowbuf[];  // [kWarpsPerBlock][hidden]
    __shared__ bool last;
    const std::uint32_t warp = threadIdx.x / 32U, lane = threadIdx.x & 31U, e = blockIdx.x * kWarpsPerBlock + warp;
    // no early PDL trigger: a dependent GEMV launched at this kernel's entry sat resident beside it and slowed this
    // latency-bound kernel ~2x on the PRO; every CTA triggers implicitly when it exits
    float* row = rowbuf + std::size_t{warp} * a.hidden;
    if (a.stamps && threadIdx.x == 0) a.stamps[blockIdx.x * 8U] = gtime();
    float bpre[kMaxExpertsPerLane];
    if (e < a.experts) {
        const float* src = a.w + std::size_t{e} * a.hidden;
        for (std::uint32_t i = lane; i < a.hidden / 4U; i += 32U) cp_async16(row + 4U * i, src + 4U * i);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    if constexpr (kPre) {  // row `warp`'s bias for the finisher and the x rows into L1, while the weight rows are in flight
        for (std::uint32_t i = 32U * threadIdx.x; i < a.hidden; i += 32U * blockDim.x)  // one 128-byte line per thread
#pragma unroll
            for (int r = 0; r < RR; ++r) asm volatile("prefetch.global.L1 [%0];" ::"l"(a.x + std::size_t(r) * a.x_stride + i));
        const bool mine = static_cast<int>(warp) < RR;
        const float* b = mine && a.image_rows && a.image_rows[warp] ? a.bias_image : a.bias;
#pragma unroll
        for (std::uint32_t t = 0; t < kMaxExpertsPerLane; ++t) {
            const std::uint32_t x = lane + 32U * t;
            bpre[t] = mine && x < a.experts ? __ldg(b + x) : 0.0F;
        }
    }
    if (e < a.experts) {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncwarp();
        if (a.stamps && threadIdx.x == 0) a.stamps[blockIdx.x * 8U + 1U] = gtime();
        float acc[RR];
#pragma unroll
        for (int r = 0; r < RR; ++r) acc[r] = 0.0F;
        // steps of 8 elements of the lane's class: every load of a step issued before its products (hidden % 256 == 0),
        // the products then added in ascending element order
        for (std::uint32_t i0 = 0; i0 < a.hidden; i0 += 256U) {
            float wv[8], xv[RR][8];
#pragma unroll
            for (int k = 0; k < 8; ++k) wv[k] = row[i0 + 32U * k + lane];
#pragma unroll
            for (int r = 0; r < RR; ++r)
#pragma unroll
                for (int k = 0; k < 8; ++k) xv[r][k] = __ldg(a.x + std::size_t(r) * a.x_stride + i0 + 32U * k + lane);
#pragma unroll
            for (int k = 0; k < 8; ++k)
#pragma unroll
                for (int r = 0; r < RR; ++r) acc[r] = acc[r] + wv[k] * xv[r][k];
        }
#pragma unroll
        for (int r = 0; r < RR; ++r) {
            const float z = detail::butterfly(acc[r]);
            if (lane == 0) a.scores[std::size_t(r) * a.experts + e] = router_score(z);
        }
    }
    if (a.stamps && threadIdx.x == 0) a.stamps[blockIdx.x * 8U + 2U] = gtime();
    // publication: every score store, then the arrival; the last block selects
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const std::uint32_t old = atomicAdd(a.counter, 1U);
        last = old + 1U == gridDim.x;
        if (last) *a.counter = 0U;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (a.stamps && threadIdx.x == 0) a.stamps[blockIdx.x * 8U + 4U] = gtime();
    for (std::uint32_t r = warp; r < static_cast<std::uint32_t>(RR); r += kWarpsPerBlock)
        select_row<kPre>(r, a.scores, a.experts, a.bias, a.bias_image, a.image_rows, a.top_k, a.route_scale, a.ids, a.weights, bpre);
    __syncthreads();
    if (a.stamps && threadIdx.x == 0) a.stamps[blockIdx.x * 8U + 3U] = gtime();
}

}  // namespace

bool router_v2_supported(const RouterView& v) noexcept {
    return v.rows >= 1U && v.rows <= 8U && v.weight.rows > 0U && v.weight.rows <= 32U * kMaxExpertsPerLane && v.weight.width % 256U == 0U &&
           v.weight.width <= kMaxHidden && v.top_k > 0U && v.top_k <= 32U && v.top_k <= v.weight.rows &&
           reinterpret_cast<std::uintptr_t>(v.weight.data) % 16U == 0U;
}

unsigned long long* g_router_v2_stamps = nullptr;
void set_router_v2_stamps(unsigned long long* stamps) { g_router_v2_stamps = stamps; }
int g_router_v2_form = 0;
void set_router_v2_form(int form) { g_router_v2_form = form; }

void enqueue_router_v2(const RouterView& v, std::uint32_t* counter, cudaStream_t stream) {
    require(router_v2_supported(v) && counter && v.scores && v.ids && v.weights && v.bias && (!v.image_rows || v.bias_image),
            "router v2 geometry");
    const std::uint32_t experts = v.weight.rows, hidden = v.weight.width;
    const RouterArgs a{v.x, v.x_stride ? v.x_stride : hidden, v.rows, experts, hidden, v.weight.data, v.scores, v.bias, v.bias_image,
                       v.image_rows, v.top_k, v.route_scale, v.ids, v.weights, counter, g_router_v2_stamps};
    const std::size_t smem = std::size_t{kWarpsPerBlock} * hidden * sizeof(float);
    const unsigned blocks = (experts + kWarpsPerBlock - 1U) / kWarpsPerBlock;
    auto go = [&](auto kernel) {
        if (smem > 48U * 1024U) STRIX_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        kernel<<<blocks, kWarpsPerBlock * 32U, smem, stream>>>(a);
    };
    const bool f2 = g_router_v2_form == 2;
    switch (v.rows) {
    case 1: f2 ? go(router_v2_kernel<1, 2>) : go(router_v2_kernel<1, 0>); break;
    case 2: f2 ? go(router_v2_kernel<2, 2>) : go(router_v2_kernel<2, 0>); break;
    case 3: go(router_v2_kernel<3, 0>); break;
    case 4: go(router_v2_kernel<4, 0>); break;
    case 5: go(router_v2_kernel<5, 0>); break;
    case 6: go(router_v2_kernel<6, 0>); break;
    case 7: go(router_v2_kernel<7, 0>); break;
    default: go(router_v2_kernel<8, 0>); break;
    }
    STRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace strix::ds41::cuda
