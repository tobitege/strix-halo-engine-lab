#pragma once
// attn_core WMMA operand types and the WMMA call (gfx11 wave32, v_wmma_f32_16x16x16_{bf16,f16}). Canonical: every
// attention core (index_score_wmma, dense_flash_prefill, sparse_flash_wmma) includes this instead of local copies.
// Rounding: RNE everywhere (bf16 keeps NaN quiet; f16 is the _Float16 cast, which saturates to inf).
#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::attn_core::tile {

typedef unsigned short Frag __attribute__((ext_vector_type(16)));
typedef float Acc __attribute__((ext_vector_type(8)));
typedef unsigned U4 __attribute__((ext_vector_type(4)));
typedef unsigned U2 __attribute__((ext_vector_type(2)));

enum class Operand : unsigned { bf16 = 1, f16 = 2 };

__device__ __forceinline__ std::uint16_t to_bf16(float x) {  // round to nearest even, NaN stays NaN
    const std::uint32_t u = __builtin_bit_cast(std::uint32_t, x);
    if ((u & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>((u >> 16U) | 0x40U);
    return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U);
}
__device__ __forceinline__ std::uint16_t to_f16(float x) { return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(x)); }
template <Operand O> __device__ __forceinline__ std::uint16_t to_operand(float x) {
    if constexpr (O == Operand::bf16) return to_bf16(x);
    else return to_f16(x);
}
template <Operand O> __device__ __forceinline__ float operand_value(std::uint16_t b) {
    if constexpr (O == Operand::bf16) return __builtin_bit_cast(float, static_cast<std::uint32_t>(b) << 16U);
    else return static_cast<float>(__builtin_bit_cast(_Float16, b));
}
template <Operand O> __device__ __forceinline__ unsigned pack2(float a, float b) {
    return static_cast<unsigned>(to_operand<O>(a)) | (static_cast<unsigned>(to_operand<O>(b)) << 16U);
}
// Eight FP32 values -> one 16-byte piece of operand elements (idx-wmma's pack8).
template <Operand O> __device__ __forceinline__ U4 pack8(float4 lo, float4 hi) {
    return U4{pack2<O>(lo.x, lo.y), pack2<O>(lo.z, lo.w), pack2<O>(hi.x, hi.y), pack2<O>(hi.z, hi.w)};
}
template <Operand O> __device__ __forceinline__ Acc mma(Frag a, Frag b, Acc c) {
#if defined(__gfx1151__) || defined(__gfx1150__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    if constexpr (O == Operand::bf16) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    } else {
        typedef _Float16 Half16 __attribute__((ext_vector_type(16)));
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(Half16, a), __builtin_bit_cast(Half16, b), c);
    }
#else
    (void)a;
    (void)b;
    return c;
#endif
}
__device__ __forceinline__ Acc acc_zero() { return Acc{0, 0, 0, 0, 0, 0, 0, 0}; }

}  // namespace strix::attn_core::tile
