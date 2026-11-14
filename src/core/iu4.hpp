#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace strix::core {
// Lossy FAST v1. Inputs are already in the model's had128-rotated basis.
// Signed two's-complement nibbles, low nibble first; no zero point.
struct PackedIU4 { static constexpr int element_bits = 4, stage_tiles = 8; };
// Eight-byte fragment swizzle. All 16 distinct row fragments use disjoint
// dword banks even in a 32-bank set; lanes 16..31 broadcast lanes 0..15.
// A 16-byte staging chunk must be stored as two separately swizzled halves.
#if defined(__HIPCC__)
__host__ __device__
#endif
inline constexpr int iu4_lds_offset(int row, int byte) {
    return row * (PackedIU4::stage_tiles * 8) + (byte ^ (((row >> 1) & 7) * 8));
}
struct PackedF16 { static constexpr int element_bits = 16, stage_tiles = 2; };
// The one requantisation used by the converter at load time: inv = 1.0f / f32(s)
// (correctly rounded division, once per column x group); plain multiply (no FMA, the
// library builds with -ffp-contract=off), v_rndne, clamp. 
#if defined(__HIPCC__)
__host__ __device__
#endif
inline int iu4_requant(float w, float inv) {
#if defined(__HIP_DEVICE_COMPILE__)
    const int q = static_cast<int>(__builtin_rintf(w * inv));
#else
    const int q = static_cast<int>(std::nearbyint(w * inv));
#endif
    return q < -7 ? -7 : (q > 7 ? 7 : q);
}
// Scale rules of the load-time converter. Legacy: s = max|w| / 7, nibbles in [-7, 7]. Fast full range: one
// scale s = 0.9 max(max(w, 0) / 7, max(-w, 0) / 8) (0.9 of the largest scale at which [-8, 7] does not clip),
// nibbles RNE in [-8, 7]; no search, the cost of the legacy rule. Gaussian g64 groups, MSE / energy: legacy
// 0.0116, fast full range 0.0096.
inline constexpr int iu4_scale_rule_legacy = 0;
inline constexpr int iu4_scale_rule_fast_full = 2001;
inline constexpr bool iu4_scale_rule_valid(int rule) { return rule == iu4_scale_rule_legacy || rule == iu4_scale_rule_fast_full; }
#if defined(__HIPCC__)
__host__ __device__
#endif
inline float iu4_fast_full_scale(float positive, float negative) {
    const float a = positive / 7.0f, b = negative / 8.0f, base = a > b ? a : b, raw = 0.9f * base;
    return base == 0.0f ? 1.0f : (raw > 0x1p-24f ? raw : 0x1p-24f);
}
#if defined(__HIPCC__)
__host__ __device__
#endif
inline int iu4_requant_full(float w, float inv) {
    float x = w * inv;
    x = x < -8.0f ? -8.0f : (x > 7.0f ? 7.0f : x);
#if defined(__HIP_DEVICE_COMPILE__)
    return static_cast<int>(__builtin_rintf(x));
#else
    return static_cast<int>(std::nearbyint(x));
#endif
}
template<int Bits, int CB> struct TrellisF16 {
    static constexpr int element_bits = 16, stage_tiles = 2;
    static constexpr int bits = Bits, codebook = CB;
};
template<class OperandPolicy> struct TiledContract {
    using Operand = OperandPolicy;
    static constexpr unsigned version = 1;
};
// W tiles: [K/16][N/16][column within tile][K within tile / 2].
// This is the trellis K16,N16 tile traversal, with plain signed nibbles inside.
#if defined(__HIPCC__)
__host__ __device__
#endif
inline size_t iu4_weight_byte(int k, int n, int outputs) {
    return ((size_t(k / 16) * size_t(outputs / 16) + size_t(n / 16)) * 16U +
        size_t(n % 16)) * 8U + size_t((k % 16) / 2);
}
inline int iu4_unpack(uint8_t byte, int high) {
    const int nibble = (int(byte) >> (high * 4)) & 15;
    return nibble < 8 ? nibble : nibble - 16;
}
inline void iu4_put(uint8_t& byte, int high, int value) {
    const unsigned shift = unsigned(high) * 4U;
    byte = uint8_t((unsigned(byte) & ~(15U << shift)) | ((unsigned(value) & 15U) << shift));
}
// Independent RNE, including negative halfway cases; does not depend on fenv.
inline int iu4_rne(float value) {
    const float base = std::floor(value), fraction = value - base;
    const int lower = int(base);
    return std::clamp(lower + int(fraction > 0.5f || (fraction == 0.5f && lower % 2 != 0)), -7, 7);
}
template<class Half> struct HostIU4Weights {
    std::vector<uint8_t> packed;
    std::vector<Half> scales; // [N][K/64], FP16, stored scale also used for quantisation
};
// Half is the caller's IEEE FP16 scalar (_Float16 in HIP). W is row-major [K][N].
// All-zero groups use s=1; nonzero W scales are floored at FP16 min-subnormal.
template<class Half> HostIU4Weights<Half> quantise_iu4_weights(const Half* weights, int inputs, int outputs, int group = 64) {
    if (!weights || inputs <= 0 || (group != 64 && group != 128 && group != 256) || inputs % group || outputs <= 0 || outputs % 16)
        throw std::invalid_argument("IU4 weights require finite FP16 [K][N], g in {64,128,256}, K%g=N%16=0");
    HostIU4Weights<Half> result;
    result.packed.resize(size_t(inputs) * size_t(outputs) / 2U);
    result.scales.resize(size_t(outputs) * size_t(inputs / group));
    for (int n = 0; n < outputs; ++n) for (int g = 0; g < inputs / group; ++g) {
        float maximum = 0;
        for (int j = 0; j < group; ++j) {
            const float value = float(weights[size_t(g * group + j) * size_t(outputs) + size_t(n)]);
            if (!std::isfinite(value)) throw std::invalid_argument("non-finite IU4 weight");
            maximum = std::max(maximum, std::abs(value));
        }
        const Half scale = Half(maximum == 0 ? 1.0f : std::max(maximum / 7.0f, 0x1p-24f));
        result.scales[size_t(n) * size_t(inputs / group) + size_t(g)] = scale;
        for (int j = 0; j < group; ++j) {
            const int k = g * group + j;
            const float value = float(weights[size_t(k) * size_t(outputs) + size_t(n)]);
            iu4_put(result.packed[iu4_weight_byte(k, n, outputs)], k & 1, iu4_rne(value / float(scale)));
        }
    }
    return result;
}
} // namespace strix::core
