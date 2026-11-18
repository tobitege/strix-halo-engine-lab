// HIP backend core for the shared logprob partial contract (strix/logprob_partials.hpp). Compiled with
// -fno-fast-math -ffp-contract=off (CMakeLists): the nonfinite test and the FP32 lane order must hold as written.
#include "strix/hip_logprob_partials.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace strix {
namespace {
constexpr unsigned kThreads = kLogprobPartialLanes;
static_assert(kThreads == 256U, "the kernel's fold assumes 256 lanes");

__device__ __forceinline__ bool ahead(float va, std::uint32_t ia, float vb, std::uint32_t ib) {
    return va > vb || (va == vb && ia < ib);
}
// kBatch strided entries of one lane loaded before any is used (several loads in flight); used in index order, so the
// lane's summation order is the contract's.
constexpr unsigned kBatch = 32;
__device__ __forceinline__ void load_batch(const float* __restrict__ row, std::uint32_t base, std::uint32_t count, float* xs) {
#pragma unroll
    for (unsigned u = 0; u < kBatch; ++u) {
        const std::uint32_t i = base + u * kThreads;
        xs[u] = i < count ? row[i] : 0.0F;
    }
}
__device__ __forceinline__ float infinity() { return __int_as_float(0x7F800000); }
__device__ __forceinline__ bool finite(float x) { return (__float_as_uint(x) & 0x7F800000U) != 0x7F800000U; }

// Two passes over the row (the second from cache):
// 1. maximum, nonfinite count and each lane's best entry (value descending, lowest ID first);
// 2. the contract's FP32 exp sum, the target counts, and the top candidates: every entry ranking at or ahead of tau,
//    the `top`-th best lane maximum (every finite entry when fewer lanes hold one). The `top` best lane maxima are `top` distinct entries, so the true top lies inside
//    the candidates; they are at most `top` lanes' entries (cap kCandidates). Ranks inside the candidates give the
//    sorted top (a total order: the result never depends on the candidates' arrival order). A row whose candidates
//    exceed the cap takes `top` exact selection passes instead (same result, slower).
constexpr unsigned kCandidates = 2048;
__global__ __launch_bounds__(kThreads) void logprob_partials_kernel(const float* __restrict__ logits, std::uint32_t count,
    std::uint32_t stride, std::uint32_t first, std::uint32_t top, const LogprobTarget* __restrict__ targets,
    LogprobPartial* __restrict__ out) {
    __shared__ float cand_v[kCandidates];
    __shared__ std::uint32_t cand_i[kCandidates];
    __shared__ float fold_f[kThreads], lane_v[kThreads];
    __shared__ std::uint32_t fold_u[kThreads], fold_w[kThreads], fold_x[kThreads], lane_i[kThreads];
    __shared__ std::uint32_t n_cand;
    __shared__ float tau_v;
    __shared__ std::uint32_t tau_i;
    const unsigned t = threadIdx.x;
    const float* row = logits + static_cast<std::size_t>(blockIdx.x) * stride;
    LogprobTarget target{};
    if (targets != nullptr) target = targets[blockIdx.x];
    const bool has_target = target.id != kLogprobNoTarget;
    const bool found = has_target && target.id >= first && target.id - first < count;
    if (found) { target.logit = row[target.id - first]; target.logit_known = true; }
    const bool counting = has_target && target.logit_known;
    // 1.
    float m = -infinity(), bv = -infinity();
    std::uint32_t bad = 0, bi = 0xFFFFFFFFU;
    for (std::uint32_t base = t; base < count; base += kThreads * kBatch) {
        float xs[kBatch];
        load_batch(row, base, count, xs);
#pragma unroll
        for (unsigned u = 0; u < kBatch; ++u) {
            const std::uint32_t i = base + u * kThreads;
            if (i >= count) break;
            const float x = xs[u];
            if (!finite(x)) { ++bad; continue; }
            m = fmaxf(m, x);
            if (bi == 0xFFFFFFFFU || x > bv) { bv = x; bi = first + i; }
        }
    }
    fold_f[t] = m;
    fold_u[t] = bad;
    lane_v[t] = bv;
    lane_i[t] = bi;
    if (t == 0U) { n_cand = 0U; tau_v = -infinity(); tau_i = 0xFFFFFFFFU; }
    __syncthreads();
    for (unsigned half = kThreads / 2U; half > 0U; half /= 2U) {
        if (t < half) { fold_f[t] = fmaxf(fold_f[t], fold_f[t + half]); fold_u[t] += fold_u[t + half]; }
        __syncthreads();
    }
    const float maximum = fold_f[0];
    const std::uint32_t nonfinite = fold_u[0];
    // tau: the lane maximum of rank min(top, live lanes) - 1 among the live lanes (ranks are unique: IDs differ)
    std::uint32_t live = 0, rank = 0;
    for (unsigned l = 0; l < kThreads; ++l) {
        if (lane_i[l] == 0xFFFFFFFFU) continue;
        ++live;
        if (bi != 0xFFFFFFFFU && l != t && ahead(lane_v[l], lane_i[l], bv, bi)) ++rank;
    }
    // fewer live lanes than `top`: every finite entry is a candidate (tau = the lowest possible key)
    const std::uint32_t finite_count = count - nonfinite;
    const std::uint32_t want = top < finite_count ? top : finite_count;
    if (t == 0U && live < top) { tau_v = -infinity(); tau_i = 0xFFFFFFFFU; }
    if (want != 0U && live >= top && bi != 0xFFFFFFFFU && rank == top - 1U) { tau_v = bv; tau_i = bi; }
    __syncthreads();
    // 2.
    float s = 0.0F;
    std::uint32_t greater = 0, ties = 0;
    const float tv = tau_v;
    const std::uint32_t ti = tau_i;
    for (std::uint32_t base = t; base < count; base += kThreads * kBatch) {
      float xs[kBatch];
      load_batch(row, base, count, xs);
#pragma unroll
      for (unsigned u = 0; u < kBatch; ++u) {
        const std::uint32_t i = base + u * kThreads;
        if (i >= count) break;
        const float x = xs[u];
        if (!finite(x)) continue;
        s += expf(x - maximum);
        if (counting) {
            if (x > target.logit) ++greater;
            else if (x == target.logit && first + i < target.id) ++ties;
        }
        if (want != 0U && (x > tv || (x == tv && first + i <= ti))) {
            const std::uint32_t at = atomicAdd(&n_cand, 1U);
            if (at < kCandidates) { cand_v[at] = x; cand_i[at] = first + i; }
        }
      }
    }
    fold_f[t] = s;
    fold_w[t] = greater;
    fold_x[t] = ties;
    __syncthreads();
    for (unsigned half = kThreads / 2U; half > 0U; half /= 2U) {
        if (t < half) { fold_f[t] += fold_f[t + half]; fold_w[t] += fold_w[t + half]; fold_x[t] += fold_x[t + half]; }
        __syncthreads();
    }
    LogprobPartial* p = out + blockIdx.x;
    const std::uint32_t c = n_cand;
    if (t == 0U) {
        p->max = maximum;
        p->sum = fold_f[0];
        p->target = found ? target.logit : 0.0F;
        p->flags = (found ? LogprobPartial::kFound : 0U) | (counting ? LogprobPartial::kCounted : 0U);
        p->greater = fold_w[0];
        p->ties_lower = fold_x[0];
        p->nonfinite = nonfinite;
        p->top_count = want;
    }
    if (want == 0U) return;
    if (c <= kCandidates) {
        for (std::uint32_t k = t; k < c; k += kThreads) {
            std::uint32_t r = 0;
            for (std::uint32_t j = 0; j < c; ++j) r += ahead(cand_v[j], cand_i[j], cand_v[k], cand_i[k]) ? 1U : 0U;
            if (r < want) { p->top_value[r] = cand_v[k]; p->top_id[r] = cand_i[k]; }
        }
        return;
    }
    // overflow: `want` exact passes, each the best entry strictly behind the previous one
    float pv = infinity();
    std::uint32_t pi = 0U;
    for (std::uint32_t k = 0; k < want; ++k) {
        float v = -infinity();
        std::uint32_t id = 0xFFFFFFFFU;
        for (std::uint32_t i = t; i < count; i += kThreads) {
            const float x = row[i];
            if (!finite(x) || (k != 0U && !ahead(pv, pi, x, first + i))) continue;
            if (id == 0xFFFFFFFFU || ahead(x, first + i, v, id)) { v = x; id = first + i; }
        }
        fold_f[t] = v;
        fold_u[t] = id;
        __syncthreads();
        for (unsigned half = kThreads / 2U; half > 0U; half /= 2U) {
            if (t < half && fold_u[t + half] != 0xFFFFFFFFU &&
                (fold_u[t] == 0xFFFFFFFFU || ahead(fold_f[t + half], fold_u[t + half], fold_f[t], fold_u[t]))) {
                fold_f[t] = fold_f[t + half];
                fold_u[t] = fold_u[t + half];
            }
            __syncthreads();
        }
        pv = fold_f[0];
        pi = fold_u[0];
        if (t == 0U) { p->top_value[k] = pv; p->top_id[k] = pi; }
        __syncthreads();
    }
}
} // namespace

void enqueue_hip_logprob_partials(const HipLogprobPartialsView& v, hipStream_t stream) {
    if (v.rows == 0U) return;
    if (v.logits == nullptr || v.out == nullptr || v.count == 0U || v.stride < v.count || v.top > kLogprobPartialTop)
        throw std::invalid_argument("hip logprob partials: bad view");
    hipLaunchKernelGGL(logprob_partials_kernel, dim3(v.rows), dim3(kThreads), 0, stream, v.logits, v.count, v.stride, v.first,
                       v.top, v.targets, v.out);
    if (const auto e = hipGetLastError(); e != hipSuccess)
        throw std::runtime_error(std::string("hip logprob partials: launch failed: ") + hipGetErrorString(e));
}

} // namespace strix
