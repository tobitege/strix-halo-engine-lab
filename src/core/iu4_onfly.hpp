#pragma once
// Load-time Q8_0 -> IU4 g64 requantisation (TURBO+ onfly weights): one (output column, K64 group) at a time.
// The device converter (core/wmma_tiled.hip convert_q8_0_iu4) and the host reference below run this one function, so
// the host test checks the exact arithmetic the device executes. Output layout = core/iu4.hpp (tiles
// [K/16][N/16][column within tile][K within tile / 2], FP16 scale per column and K64 group at [n][K/64]); N is padded
// to a multiple of 64 (one wave of the W4A16 core covers 64 columns); pad columns hold zero nibbles and scale 0.
#include "core/iu4.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace strix::core::onfly {

#if defined(__HIPCC__)
#define STRIX_ONFLY_HD __host__ __device__
#else
#define STRIX_ONFLY_HD
#endif

inline constexpr unsigned kGroup = 64;
constexpr std::size_t padded_outputs(std::size_t n) { return (n + 63U) / 64U * 64U; }
constexpr std::size_t packed_bytes(std::size_t padded, std::size_t k) { return padded * k / 2U; }
constexpr std::size_t scale_bytes(std::size_t padded, std::size_t k) { return padded * (k / 64U) * 2U; }
constexpr std::size_t copy_bytes(std::size_t n, std::size_t k) { return packed_bytes(padded_outputs(n), k) + scale_bytes(padded_outputs(n), k); }
// Q8_0 source: rows of K/32 blocks of 34 bytes (f16 d, 32 int8).
constexpr std::size_t q8_row_bytes(std::size_t k) { return k / 32U * 34U; }

// Software IEEE binary16 (round to nearest even), identical on host and device.
STRIX_ONFLY_HD inline std::uint16_t f32_to_f16_bits(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, 4);
    const std::uint32_t sign = (x >> 16U) & 0x8000U, e = (x >> 23U) & 0xffU, m = x & 0x7fffffU;
    if (e == 0xffU) return static_cast<std::uint16_t>(sign | 0x7c00U | (m ? 0x200U : 0U));
    const int exp = static_cast<int>(e) - 112;
    if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);
        const std::uint32_t full = m | 0x800000U, shift = static_cast<std::uint32_t>(14 - exp);
        std::uint32_t half = full >> shift;
        const std::uint32_t rem = full & ((1U << shift) - 1U), mid = 1U << (shift - 1U);
        if (rem > mid || (rem == mid && (half & 1U))) ++half;
        return static_cast<std::uint16_t>(sign | half);
    }
    std::uint32_t half = (static_cast<std::uint32_t>(exp) << 10U) | (m >> 13U);
    const std::uint32_t rem = m & 0x1fffU;
    if (rem > 0x1000U || (rem == 0x1000U && (half & 1U))) ++half;
    return static_cast<std::uint16_t>(sign | half);
}
STRIX_ONFLY_HD inline float f16_bits_to_f32(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000U) << 16U, e = (h >> 10U) & 0x1fU, m = h & 0x3ffU;
    std::uint32_t x;
    if (e == 0U) {
        if (m == 0U) x = sign;
        else {  // subnormal: m * 2^-24, exact in F32
            float v = static_cast<float>(m) * 0x1p-24f;
            std::memcpy(&x, &v, 4);
            x |= sign;
        }
    } else if (e == 31U) x = sign | 0x7f800000U | (m << 13U);
    else x = sign | ((e + 112U) << 23U) | (m << 13U);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// 64 dequantised weights of one column / group -> nibbles q[0..63] (RNE, [-7,7] legacy or [-8,7] fast full range)
// and the FP16 scale bits; the stored (FP16) scale is the one used for quantisation.
STRIX_ONFLY_HD inline std::uint16_t quantise_group(const float* w, int rule, int* q) {
    float amax = 0.0f, pos = 0.0f, neg = 0.0f;
    for (int j = 0; j < 64; ++j) {
        const float a = w[j] < 0.0f ? -w[j] : w[j];
        amax = a > amax ? a : amax;
        pos = w[j] > pos ? w[j] : pos;
        neg = -w[j] > neg ? -w[j] : neg;
    }
    if (rule == iu4_scale_rule_mse) {
        const float a = pos / 7.0f, b = neg / 8.0f, base = a > b ? a : b;
        std::uint16_t best_bits = f32_to_f16_bits(1.0f);
        if (base != 0.0f) {
            float best = -1.0f;
            for (int i = 0; i < iu4_mse_steps; ++i) {
                const float raw = iu4_mse_clip(i) * base;
                const std::uint16_t cb = f32_to_f16_bits(raw > 0x1p-24f ? raw : 0x1p-24f);
                const float cs = f16_bits_to_f32(cb), inv = 1.0f / cs;
                float err = 0.0f;
                for (int j = 0; j < 64; ++j) {
                    const float d = w[j] - cs * static_cast<float>(iu4_requant_full(w[j], inv));
                    err += d * d;
                }
                if (best < 0.0f || err < best) { best = err; best_bits = cb; }
            }
        }
        const float inv = 1.0f / f16_bits_to_f32(best_bits);
        for (int j = 0; j < 64; ++j) q[j] = iu4_requant_full(w[j], inv);
        return best_bits;
    }
    float s;
    if (rule == iu4_scale_rule_fast_full) s = iu4_fast_full_scale(pos, neg);
    else { const float t = amax / 7.0f; s = amax == 0.0f ? 1.0f : (t > 0x1p-24f ? t : 0x1p-24f); }
    const std::uint16_t bits = f32_to_f16_bits(s);
    const float inv = 1.0f / f16_bits_to_f32(bits);
    for (int j = 0; j < 64; ++j) q[j] = rule == iu4_scale_rule_fast_full ? iu4_requant_full(w[j], inv) : iu4_requant(w[j], inv);
    return bits;
}
// Dequantised Q8_0 block pair (blocks 2g, 2g + 1 of a row) -> 64 floats; d * q is exact in F32.
STRIX_ONFLY_HD inline void dequant_q8_group(const std::uint8_t* row, unsigned g, float* w) {
    for (unsigned b = 0; b < 2U; ++b) {
        const std::uint8_t* blk = row + (2U * g + b) * 34U;
        const float d = f16_bits_to_f32(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8U)));
        for (unsigned j = 0; j < 32U; ++j) w[32U * b + j] = d * static_cast<float>(static_cast<std::int8_t>(blk[2U + j]));
    }
}

struct HostCopy {
    std::vector<std::uint8_t> packed;     // packed_bytes(padded, K)
    std::vector<std::uint16_t> scales;    // [padded][K/64] FP16 bits
};
// Scalar reference over a row-major Q8_0 tensor (N rows of K inputs).
inline HostCopy convert_q8_0_host(const std::uint8_t* q8, std::size_t n, std::size_t k, int rule) {
    if (!q8 || !n || !k || k % 64U || !iu4_scale_rule_valid(rule)) throw std::invalid_argument("onfly IU4: K % 64 == 0, valid rule");
    const std::size_t padded = padded_outputs(n), groups = k / 64U;
    HostCopy out;
    out.packed.assign(packed_bytes(padded, k), 0);
    out.scales.assign(padded * groups, 0);
    float w[64];
    int q[64];
    for (std::size_t c = 0; c < n; ++c)
        for (std::size_t g = 0; g < groups; ++g) {
            dequant_q8_group(q8 + c * q8_row_bytes(k), static_cast<unsigned>(g), w);
            out.scales[c * groups + g] = quantise_group(w, rule, q);
            for (std::size_t j = 0; j < 64; j += 2) {
                const std::size_t kk = g * 64U + j;
                out.packed[iu4_weight_byte(static_cast<int>(kk), static_cast<int>(c), static_cast<int>(padded))] =
                    static_cast<std::uint8_t>((static_cast<unsigned>(q[j]) & 15U) | ((static_cast<unsigned>(q[j + 1]) & 15U) << 4U));
            }
        }
    return out;
}
}  // namespace strix::core::onfly
