#pragma once
// attn_core recurrent state codec: the ONE load / store rule for recurrent state elements, shared by every backend
// (HIP device, CUDA device, host / CPU). Contract (strix/recurrent_state_type.hpp):
//   - load widens the stored element to float exactly (f16 / bf16 -> f32 is exact);
//   - store narrows a float with IEEE round-to-nearest-even; f16 keeps subnormals and overflows to +-inf;
//   - every NaN is stored as the canonical quiet NaN of its sign (0x7e00 / 0x7fc0), so all backends write the
//     same bits for the same float;
//   - f32 is the identity (the incumbent bits).
// The host path is pure integer arithmetic; the device paths use the hardware RNE conversion and are pinned to
// the host path bit for bit by tests (hip_gdn_deferred_window_tests "state16 codec").
#include "strix/recurrent_state_type.hpp"

#include <cstdint>
#include <cstring>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_RS_FN __host__ __device__ __forceinline__
#define STRIX_RS_UNROLL _Pragma("unroll")
#else
#define STRIX_RS_FN inline
#define STRIX_RS_UNROLL
#endif
#if defined(__CUDACC__)
#include <cuda_fp16.h>
#endif

namespace strix::attn_core::rstate {

STRIX_RS_FN std::uint32_t f32_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
STRIX_RS_FN float f32_from_bits(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// ---- host reference (integer only) -------------------------------------------------------------------------
STRIX_RS_FN std::uint16_t bf16_from_f32(float value) {
    const auto bits = f32_bits(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>(((bits >> 16) & 0x8000U) | 0x7fc0U);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
STRIX_RS_FN float f32_from_bf16(std::uint16_t bits) { return f32_from_bits(static_cast<std::uint32_t>(bits) << 16); }

STRIX_RS_FN std::uint16_t f16_from_f32_host(float value) {
    const auto bits = f32_bits(value);
    const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000U);
    const auto abs = bits & 0x7fffffffU;
    if (abs > 0x7f800000U) return static_cast<std::uint16_t>(sign | 0x7e00U);
    if (abs >= 0x477ff000U) return static_cast<std::uint16_t>(sign | 0x7c00U);  // rounds to >= 65520 -> inf
    if (abs < 0x38800000U) {                                                   // f16 subnormal or zero
        if (abs < 0x33000000U) return sign;                                      // < 2^-25: rounds to zero
        const auto exponent = abs >> 23;                                         // 102 .. 112
        const auto mantissa = (abs & 0x7fffffU) | 0x800000U;
        const auto shift = 126U - exponent;                                      // 14 .. 24
        const auto kept = mantissa >> shift, rest = mantissa & ((1U << shift) - 1U), half = 1U << (shift - 1U);
        const auto rounded = kept + ((rest > half || (rest == half && (kept & 1U))) ? 1U : 0U);
        return static_cast<std::uint16_t>(sign | rounded);
    }
    const auto rebased = abs - 0x38000000U;  // exponent 127 -> 15
    const auto rounded = rebased + 0xfffU + ((rebased >> 13) & 1U);
    return static_cast<std::uint16_t>(sign | (rounded >> 13));
}
STRIX_RS_FN float f32_from_f16_host(std::uint16_t half) {
    const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000U) << 16;
    const std::uint32_t exponent = (half >> 10) & 0x1fU, mantissa = half & 0x3ffU;
    if (exponent == 0x1fU) return f32_from_bits(sign | 0x7f800000U | (mantissa << 13));
    if (exponent != 0U) return f32_from_bits(sign | ((exponent + 112U) << 23) | (mantissa << 13));
    if (mantissa == 0U) return f32_from_bits(sign);
    // subnormal: value = mantissa x 2^-24, exact in float
    float value = static_cast<float>(mantissa) * 5.9604644775390625e-8F;
    return sign ? -value : value;
}

// ---- device / host dispatch -------------------------------------------------------------------------------
STRIX_RS_FN std::uint16_t f16_from_f32(float value) {
#if defined(__HIP_DEVICE_COMPILE__)
    // The barrier keeps the operand a materialised f32: without it the compiler may fold fptrunc(fmul / fma) into
    // v_fma_mixlo_f16 (ONE rounding of the exact result to f16) instead of f32 rounding then f16 rounding, so a
    // fused and an unfused site would store different bits (hal2 RUN3 window != serial).
    __asm__ volatile("" : "+v"(value));
    if (__builtin_isnan(value)) return static_cast<std::uint16_t>((f32_bits(value) >> 16 & 0x8000U) | 0x7e00U);
    return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(value));
#elif defined(__CUDA_ARCH__)
    if (isnan(value)) return static_cast<std::uint16_t>((f32_bits(value) >> 16 & 0x8000U) | 0x7e00U);
    return __half_as_ushort(__float2half_rn(value));
#else
    return f16_from_f32_host(value);
#endif
}
STRIX_RS_FN float f32_from_f16(std::uint16_t half) {
#if defined(__HIP_DEVICE_COMPILE__)
    return static_cast<float>(__builtin_bit_cast(_Float16, half));
#elif defined(__CUDA_ARCH__)
    return __half2float(__ushort_as_half(half));
#else
    return f32_from_f16_host(half);
#endif
}

// Element storage of each contract: Elem is the in-memory type, load / store the one rule.
template <RecurrentStateType Type> struct Codec;
template <> struct Codec<RecurrentStateType::f32> {
    using Elem = float;
    STRIX_RS_FN static float load(Elem value) { return value; }
    STRIX_RS_FN static Elem store(float value) { return value; }
};
template <> struct Codec<RecurrentStateType::f16> {
    using Elem = std::uint16_t;
    STRIX_RS_FN static float load(Elem value) { return f32_from_f16(value); }
    STRIX_RS_FN static Elem store(float value) { return f16_from_f32(value); }
};
template <> struct Codec<RecurrentStateType::bf16> {
    using Elem = std::uint16_t;
    STRIX_RS_FN static float load(Elem value) { return f32_from_bf16(value); }
    STRIX_RS_FN static Elem store(float value) { return bf16_from_f32(value); }
};

// Typed view of an untyped state pointer: state<T>(p)[i] reads / writes through the codec.
template <RecurrentStateType Type> struct View {
    using Elem = typename Codec<Type>::Elem;
    Elem* base;
    STRIX_RS_FN float load(std::uint64_t index) const { return Codec<Type>::load(base[index]); }
    STRIX_RS_FN void store(std::uint64_t index, float value) const { base[index] = Codec<Type>::store(value); }
};
template <RecurrentStateType Type> STRIX_RS_FN View<Type> view(void* base) {
    return View<Type>{static_cast<typename Codec<Type>::Elem*>(base)};
}
// Contiguous run access for the 16-bit contracts: N consecutive elements (N = 2, 4 or a multiple of 8, base aligned to
// 2N bytes, at most 16) move as ONE 4 / 8 / 16-byte access per group instead of N two-byte accesses. Scalar 2-byte
// stores of lane-strided runs write partial sectors and ran the window commit at ~27 GB/s (hal2 RUN2); the packed form
// writes whole 16-byte segments like the f32 kernels' dwordx4. Same values and bits as N scalar load / store calls.
template <unsigned Words> struct alignas(Words * 4U) Pack { std::uint32_t w[Words]; };
template <RecurrentStateType Type, unsigned N>
STRIX_RS_FN void load_run(const std::uint16_t* src, float (&out)[N]) {
    static_assert(Type != RecurrentStateType::f32, "f32 kernels keep their incumbent scalar access");
    static_assert(N == 2U || N == 4U || N % 8U == 0U, "run of 2, 4 or 8k elements");
    constexpr unsigned group = N < 8U ? N : 8U, words = group / 2U;
STRIX_RS_UNROLL
    for (unsigned g = 0; g < N / group; ++g) {
        const auto p = reinterpret_cast<const Pack<words>*>(src)[g];
STRIX_RS_UNROLL
        for (unsigned i = 0; i < words; ++i) {
            out[g * group + 2U * i] = Codec<Type>::load(static_cast<std::uint16_t>(p.w[i] & 0xffffU));
            out[g * group + 2U * i + 1U] = Codec<Type>::load(static_cast<std::uint16_t>(p.w[i] >> 16));
        }
    }
}
template <RecurrentStateType Type, unsigned N>
STRIX_RS_FN void store_run(std::uint16_t* dst, const float (&in)[N]) {
    static_assert(Type != RecurrentStateType::f32, "f32 kernels keep their incumbent scalar access");
    static_assert(N == 2U || N == 4U || N % 8U == 0U, "run of 2, 4 or 8k elements");
    constexpr unsigned group = N < 8U ? N : 8U, words = group / 2U;
STRIX_RS_UNROLL
    for (unsigned g = 0; g < N / group; ++g) {
        Pack<words> p;
STRIX_RS_UNROLL
        for (unsigned i = 0; i < words; ++i)
            p.w[i] = static_cast<std::uint32_t>(Codec<Type>::store(in[g * group + 2U * i])) |
                (static_cast<std::uint32_t>(Codec<Type>::store(in[g * group + 2U * i + 1U])) << 16);
        reinterpret_cast<Pack<words>*>(dst)[g] = p;
    }
}
// In-register round trip (the value the next row would load after a store), without the NaN canonicalisation:
// only STORED bits are canonical; a NaN in registers stays a NaN and is canonicalised when it is stored.
template <RecurrentStateType Type> STRIX_RS_FN float round_in_register(float value) {
    if constexpr (Type == RecurrentStateType::bf16) return f32_from_bf16(bf16_from_f32(value));
    else if constexpr (Type == RecurrentStateType::f16) {
#if defined(__HIP_DEVICE_COMPILE__)
        __asm__ volatile("" : "+v"(value));  // no fptrunc(fmul / fma) -> v_fma_mixlo fold (see f16_from_f32)
        return static_cast<float>(static_cast<_Float16>(value));
#elif defined(__CUDA_ARCH__)
        return __half2float(__float2half_rn(value));
#else
        return f32_from_f16_host(f16_from_f32_host(value));
#endif
    } else return value;
}

// Host round trip (reference arms, quality oracles): the float the state holds after one store.
STRIX_RS_FN float round_trip(RecurrentStateType type, float value) {
    return type == RecurrentStateType::f16 ? f32_from_f16_host(f16_from_f32_host(value)) :
        type == RecurrentStateType::bf16 ? f32_from_bf16(bf16_from_f32(value)) : value;
}

}  // namespace strix::attn_core::rstate
