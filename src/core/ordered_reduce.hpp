#pragma once
// Ordered reductions with fewer barriers, preserving the explicit-add LDS tree below.
// Broadcast qualification requires raw bit equality, including NaN payloads. Legacy += trees used by lane-0/pair
// callers have a weaker comparison contract: non-NaN bits and NaN positions match, but NaN payloads may differ
// because the compiler can select a different operand order for +=. This is not universal bitwise qualification.
//
// R2 tree. The engines' exact R2 reduction is the LDS halving tree
//     s[l] = part(l); sync; for (h = L/2; h; h >>= 1) { if (l < h) s[l] = s[l] + s[l + h]; sync; }
// Lane 0's result depends only on that pairing and operand order, not on where a level runs. Levels h >= 64 cross
// waves and stay in LDS with a barrier each; level 32 is read from LDS into wave 0's registers; levels 16..1 run inside
// wave 0 through qwen_dpp::down_tree32 (permlanex16 for 16, DPP row_xmask below, HIP shuffles off gfx1151/wave32).
// Lane l < h adds v(l) + v(l + h) = s[l] + s[l + h]: the same two operands in the same order, one IEEE add, no FMA and
// no reassociation, so signed zeros, cancellations, infinities and NaN results are those of the LDS loop. Lanes l >= h
// compute values nobody reads. This is the LDS tree's pairing, not a wave-first reduction (which would pair lanes l and
// l + 16 of every wave before crossing waves).
// Barriers per tree, L = 256: 9 -> 3 (tree_to_lane0), 10 -> 5 (tree_broadcast keeps the final reuse barrier).
//
// CSR wave scan. Unsigned 32-bit inclusive scan inside one wave (DPP row_shr within 16 lanes, readlane 15 across).
// Integer addition mod 2^32 is associative and commutative, so it equals any serial or Hillis-Steele prefix bit for bit.
//
// Preconditions: 32-lane waves of consecutive threads; every lane of a participating wave is active.
// The shuffle fallback does not establish wave64 support, including for CSR callers using scalarized wave_index.
#include "../../include/strix/hip_qwen_dpp_reduce.hpp"

#include <hip/hip_runtime.h>

namespace strix::core::ordered_reduce {

// One level: own + other, in that operand order. The empty asm keeps the exchange a separate v_mov_dpp: folded into
// v_add_f32_dpp the DPP operand must be src0, which computes other + own (same value for every non-NaN sum, but the
// other operand's payload when both are NaN), measured on gfx1151.
template <unsigned H> __device__ __forceinline__ float level(float v) {
    float other = ::strix::qwen_dpp::down_tree32<H>(v);
    asm volatile("" : "+v"(other));
    return v + other;
}

// Levels 16..1 of the tree in wave 0: on entry lane l < 32 holds the level-32 sum s[l] + s[l + 32].
__device__ __forceinline__ float wave0_levels(float v) {
    v = level<16U>(v);
    v = level<8U>(v);
    v = level<4U>(v);
    v = level<2U>(v);
    v = level<1U>(v);
    return v;
}

// N trees over L lanes that share each LDS level's barrier (blockDim.x == L, lane == threadIdx.x). v[i] holds the lane's
// partial on entry; on return lane 0's v[i] equals s_i[0] of the LDS loop (other lanes: unspecified). Wave 0 reads s
// last, so the caller must pass a barrier before any lane writes s again (the existing reuse barrier).
template <unsigned L, unsigned N>
__device__ __forceinline__ void tree_to_lane0(float* const (&s)[N], float (&v)[N], unsigned lane) {
    static_assert(L >= 64U && L <= 1024U && (L & (L - 1U)) == 0U, "tree_to_lane0: L must be a power of two in 64..1024");
#pragma unroll
    for (unsigned i = 0; i < N; ++i) s[i][lane] = v[i];
    __syncthreads();
#pragma unroll
    for (unsigned h = L / 2U; h >= 64U; h >>= 1U) {
        if (lane < h) {
#pragma unroll
            for (unsigned i = 0; i < N; ++i) s[i][lane] = s[i][lane] + s[i][lane + h];
        }
        __syncthreads();
    }
    if (lane < 32U) {
#pragma unroll
        for (unsigned i = 0; i < N; ++i) v[i] = wave0_levels(s[i][lane] + s[i][lane + 32U]);
    }
}

template <unsigned L> __device__ __forceinline__ float tree_to_lane0(float* s, float part, unsigned lane) {
    float* const a[1] = {s};
    float v[1] = {part};
    tree_to_lane0<L, 1U>(a, v, lane);
    return v[0];
}

// The lds_tree contract: every lane gets s[0]; the final barrier lets the caller reuse s at once.
template <unsigned L> __device__ __forceinline__ float tree_broadcast(float* s, float part, unsigned lane) {
    const float v = tree_to_lane0<L>(s, part, lane);
    if (lane == 0U) s[0] = v;
    __syncthreads();
    const float r = s[0];
    __syncthreads();
    return r;
}

// Inclusive unsigned scan over the 32 lanes of a fully active wave (lane = thread & 31).
__device__ __forceinline__ unsigned wave_inclusive_scan(unsigned x, unsigned lane) {
#if STRIX_DPP_REDUCE && defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1151__)
    if (__builtin_amdgcn_wavefrontsize() == 32) {
        // row_shr:D (0x110 + D): lane l reads l - D inside its 16-lane row; the first D lanes of a row read old = 0.
        x += static_cast<unsigned>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x111, 0xf, 0xf, false));
        x += static_cast<unsigned>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x112, 0xf, 0xf, false));
        x += static_cast<unsigned>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x114, 0xf, 0xf, false));
        x += static_cast<unsigned>(__builtin_amdgcn_update_dpp(0, static_cast<int>(x), 0x118, 0xf, 0xf, false));
        const unsigned row0 = static_cast<unsigned>(__builtin_amdgcn_readlane(static_cast<int>(x), 15));
        return (lane & 16U) ? x + row0 : x;
    }
#endif
#pragma unroll
    for (unsigned d = 1U; d < 32U; d <<= 1U) {
        const unsigned y = __shfl_up(x, d, 32);
        if (lane >= d) x += y;
    }
    return x;
}

}  // namespace strix::core::ordered_reduce
