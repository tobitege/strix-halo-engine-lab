#pragma once
// Paired GLU store policy (shared by every tiled gate|up projection).
//
// A paired site orders its 2 * I outputs as alternating 16-output blocks: block 2k holds gate columns 16k .. 16k + 15
// and block 2k + 1 the up columns of the same 16 features. In the WMMA accumulator layout (lane l holds column l % 16
// of a 16 x 16 sub-tile), a wave whose N sub-tiles start on an even block therefore holds gate and up of one
// (row, feature) in the same lane, so the epilogue writes act = silu(gate) * up directly as FP16 and the FP32 gate|up
// image (and its separate activation pass) is never materialized.
//
// Arithmetic: the same expression as the activation kernels it replaces, (g / (1 + expf(-g))) * u, with the final
// product as one v_mul_f32 (no contraction), the FP16 store an RNE cast, and the FP16 overflow counter incremented once
// per aligned group of eight features of one row that holds a finite value beyond the FP16 range (as the producers do).
#include <cstdint>

namespace strix::core::glu_pair {

constexpr unsigned kBlock = 16U;

// Output o of a paired site -> its segment (0 gate, 1 up) and feature column (the row of that segment's weights).
__host__ __device__ constexpr unsigned segment(std::uint32_t o) { return (o / kBlock) & 1U; }
__host__ __device__ constexpr std::uint32_t column(std::uint32_t o) { return o / (2U * kBlock) * kBlock + o % kBlock; }
// Paired geometry: two segments of `features` outputs each, features a multiple of 16.
__host__ __device__ constexpr bool geometry(std::uint32_t features, std::uint32_t outputs) {
    return features && features % kBlock == 0U && outputs == 2U * features;
}

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
__device__ __forceinline__ float act(float gate, float up) {
    const float s = gate / (1.0F + expf(-gate));
    float a;
    asm("v_mul_f32 %0, %1, %2" : "=v"(a) : "v"(s), "v"(up));
    return a;
}
__device__ __forceinline__ bool fp16_overflow(float a) { return isfinite(a) && !(fabsf(a) <= 65504.0F); }

// Wave-wide store of one accumulator step: every lane calls it (the overflow vote is a ballot). Lanes 8j .. 8j + 7 hold
// eight consecutive aligned features of one row (WMMA C layout), so a ballot byte is one aligned group of eight: the
// counter gains the number of non-zero bytes, the same total as one increment per bad group.
__device__ __forceinline__ void store(_Float16* dst, bool live, float gate, float up, unsigned* overflow) {
    const float a = act(gate, up);
    if (live) *dst = static_cast<_Float16>(a);
    unsigned bad = __builtin_amdgcn_ballot_w32(live && fp16_overflow(a));
    if (bad) {
        bad |= bad >> 4U; bad |= bad >> 2U; bad |= bad >> 1U;
        if (__lane_id() == 0U) atomicAdd(overflow, static_cast<unsigned>(__builtin_popcount(bad & 0x01010101U)));
    }
}
#endif

}  // namespace strix::core::glu_pair
