#pragma once

// deepseek41 CUDA: the resident-expert routes of a window of at most 32 slots computed by one warp (lane s owns slot s)
// with the arrays of resident_routes_kernel (ds41_parts_host.cu): union experts in ascending id (slot_of of each), the
// entries of an expert in ascending slot, begins / counts, and per row its slots in ascending id with the count below
// split. Ranks by comparison instead of a counting sort: the same integers.

#include "strix/cuda/ds41_parts_host.hpp"

#include <cuda_runtime.h>

namespace strix::ds41::cuda::routes_detail {

// the slot of a present expert e (guarded mode: abort -> guard, absent -> reported + guard)
__device__ __forceinline__ std::int32_t route_slot(const ResidentRoutesView& v, std::uint32_t e) {
    if (!v.abort) return v.slot_of[e];
    if (*reinterpret_cast<const volatile std::uint32_t*>(v.abort)) return v.guard_slot;
    const std::int32_t s = reinterpret_cast<const volatile std::int32_t*>(v.slot_of)[e];
    if (s >= 0) return s;
    if (v.absent_err) atomicOr(v.absent_err, 1U);
    return v.guard_slot;
}

__device__ __forceinline__ void resident_routes_warp(const ResidentRoutesView& v) {
    constexpr unsigned kAll = 0xffffffffU;
    const std::uint32_t lane = threadIdx.x & 31U, slots = v.rows * v.top_k;
    const std::uint32_t id = lane < slots ? v.ids[lane] : 0xffffffffU;
    const bool valid = lane < slots && id < v.experts && (v.abort || v.slot_of[id] >= 0);  // guarded: every routed id
    const unsigned valid_mask = __ballot_sync(kAll, valid);
    std::uint32_t lt = 0, eqb = 0, mult = 0;
    for (std::uint32_t t = 0; t < slots; ++t) {
        const std::uint32_t idt = __shfl_sync(kAll, id, t);
        if (valid && ((valid_mask >> t) & 1U)) {
            lt += idt < id ? 1U : 0U;
            if (idt == id) {
                ++mult;
                eqb += t < lane ? 1U : 0U;
            }
        }
    }
    const bool first = valid && eqb == 0U;
    const unsigned first_mask = __ballot_sync(kAll, first);
    std::uint32_t uidx = 0;
    for (std::uint32_t t = 0; t < slots; ++t) {
        const std::uint32_t idt = __shfl_sync(kAll, id, t);
        if (valid && ((first_mask >> t) & 1U) && idt < id) ++uidx;
    }
    std::uint32_t smax = valid ? mult : 0U;
    for (unsigned d = 16U; d; d >>= 1U) {
        const std::uint32_t o = __shfl_xor_sync(kAll, smax, d);
        smax = o > smax ? o : smax;
    }
    const std::uint32_t unions = __popc(first_mask), total = __popc(valid_mask);
    if (first) {
        v.out.experts[uidx] = static_cast<std::uint32_t>(route_slot(v, id));
        v.out.begins[uidx] = lt;
    }
    if (valid) {
        v.out.rows[lt + eqb] = lane / v.top_k;
        v.out.slots[lt + eqb] = lane;
    }
    if (lane == 0U) {
        v.out.begins[unions] = total;
        v.out.counts[0] = unions;
        v.out.counts[1] = smax;
    }
    // per row: slots in ascending id, count below split (every slot, resident or not)
    if (lane < v.rows) {
        const std::uint32_t r = lane;
        std::uint32_t ids[16], slot[16];
        for (std::uint32_t k = 0; k < v.top_k; ++k) {
            const std::uint32_t s = r * v.top_k + k, e = v.ids[s];
            std::uint32_t a = k;
            while (a > 0U && ids[a - 1U] > e) { ids[a] = ids[a - 1U]; slot[a] = slot[a - 1U]; --a; }
            ids[a] = e; slot[a] = s;
        }
        std::uint32_t low = 0;
        for (std::uint32_t k = 0; k < v.top_k; ++k) {
            v.out.order[r * v.top_k + k] = slot[k];
            low += ids[k] < v.split ? 1U : 0U;
        }
        v.out.low_counts[r] = low;
    }
}

}  // namespace strix::ds41::cuda::routes_detail
