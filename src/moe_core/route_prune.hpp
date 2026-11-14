#pragma once
// RoutePrune: FAST route policy shared by the routed-MoE engines. After the router has written a row's top-k expert ids
// and route weights, and before the route plan (CSR) is built, the assignments whose weight is negligible are removed:
// their weight becomes exactly 0, the planner leaves them out, and every kernel that walks the plan (gate/up, down,
// tiles) sees fewer assignments. The ordered merge of a pruned call skips zero-weight slots (their raw dots are stale).
// The policy is order-free (a slot's fate depends on the row's weights only, ties broken by the lower slot), so it is the
// same for routers that emit slots by rank and routers that emit them by expert id. mode off = no launch, no change.
#include "strix/moe_route_prune.hpp"

#include <hip/hip_runtime.h>

namespace strix::moe_core {

// RoutePrune{mode, value, renorm} (strix/moe_route_prune.hpp): relative keeps w >= value * row sum; top_p keeps the best
// slots until their mass reaches value * row sum; cap keeps the `value` best slots; the best slot is always kept;
// renorm scales the kept weights so the row keeps its original weight sum.

// Keep mask of one row (bit s = slot s stays). k <= 32. A row with a non-finite or non-positive weight sum is left whole
// (the validity checks of the planner decide). sum = ordered sum of the k weights, kept = ordered sum of the kept ones.
__device__ __forceinline__ unsigned route_prune_keep(const float* w, unsigned k, RoutePrune p, float& sum, float& kept) {
    const unsigned all = k >= 32U ? 0xffffffffU : (1U << k) - 1U;
    sum = 0.0F;
    for (unsigned s = 0; s < k; ++s) sum = sum + w[s];
    kept = sum;
    if (!p.active() || !(sum > 0.0F) || !isfinite(sum)) return all;
    const float bound = p.value * sum;
    unsigned mask = 0U;
    for (unsigned s = 0; s < k; ++s) {
        unsigned rank = 0U;
        float before = 0.0F;  // mass of the slots ranked ahead of s
        for (unsigned q = 0; q < k; ++q)
            if (w[q] > w[s] || (w[q] == w[s] && q < s)) { ++rank; before = before + w[q]; }
        const bool keep = rank == 0U || (p.mode == RoutePrune::relative ? w[s] >= bound
                                         : p.mode == RoutePrune::top_p ? before < bound
                                         : static_cast<float>(rank) < p.value);
        if (keep) mask |= 1U << s;
    }
    kept = 0.0F;
    for (unsigned s = 0; s < k; ++s) if (mask >> s & 1U) kept = kept + w[s];
    return mask;
}

// One thread per row over strided weight rows (row r at base + r * stride bytes, k floats).
// (A template so the header can be included by several translation units.)
template <unsigned Threads = 256U>
__global__ __launch_bounds__(Threads) void route_prune_kernel(float* base, unsigned long long stride, unsigned rows, unsigned k,
                                                               RoutePrune p) {
    const unsigned r = blockIdx.x * Threads + threadIdx.x;
    if (r >= rows) return;
    float* const w = reinterpret_cast<float*>(reinterpret_cast<char*>(base) + r * stride);
    float sum, kept;
    const unsigned mask = route_prune_keep(w, k, p, sum, kept);
    const float scale = p.renorm && kept > 0.0F ? sum / kept : 1.0F;
    for (unsigned s = 0; s < k; ++s) {
        if (!(mask >> s & 1U)) w[s] = 0.0F;
        else if (p.renorm) w[s] = w[s] * scale;
    }
}

// rows > 0, k <= 32, p.active(). Ordered on `stream` after the router and before the route plan.
inline hipError_t enqueue_route_prune(float* base, unsigned long long stride, unsigned rows, unsigned k, RoutePrune p,
                                      hipStream_t stream) {
    route_prune_kernel<256U><<<(rows + 255U) / 256U, 256, 0, stream>>>(base, stride, rows, k, p);
    return hipGetLastError();
}

}  // namespace strix::moe_core
