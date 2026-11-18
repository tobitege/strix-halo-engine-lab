#pragma once

// Device decode of one GSQ A8 group (see gsq_a8_groups.hpp): 8 words of 4 signed int8 weights in element order
// and the group scale(s). Used by the GSQ IQ gate/up and down kernels (R1, windows, prefill).

#include "strix/gsq_a8_groups.hpp"
#include "strix/hip_ggml_weight_formats.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::weight::hip::gsq_detail {
using namespace ::strix::gsq_a8;

struct IqGroup { int w[8]; float s0, s1; };

__device__ __forceinline__ int pack4(int a, int b, int c, int d) {
    return static_cast<int>(static_cast<std::uint32_t>(a & 255) | (static_cast<std::uint32_t>(b & 255) << 8U) |
                            (static_cast<std::uint32_t>(c & 255) << 16U) | (static_cast<std::uint32_t>(d & 255) << 24U));
}
__device__ __forceinline__ int sdot(int a, int b, int acc) { return __builtin_amdgcn_sudot4(true, a, true, b, acc, false); }

// Group g (32 weights) of a row starting at `row`.
template <std::uint32_t T>
__device__ __forceinline__ IqGroup iq_group(const unsigned char* row, std::uint32_t g) {
    constexpr std::uint32_t be = block_elements(T), bb = block_bytes(T);
    const unsigned char* x = row + static_cast<std::uint64_t>(g * 32U / be) * bb;
    const std::uint32_t ib = (g * 32U % be) / 32U;
    const float d = hip_ggml::half_bits(x);
    IqGroup r{};
    if constexpr (T == type_iq4_nl || T == type_iq4_xs) {
        const unsigned char* qs = T == type_iq4_nl ? x + 2U : x + 8U + ib * 16U;
        if constexpr (T == type_iq4_nl) {
            r.s0 = d;
        } else {
            const std::uint32_t scales_h = x[2] | (static_cast<std::uint32_t>(x[3]) << 8U);
            const int ls = static_cast<int>(((x[4U + ib / 2U] >> (4U * (ib % 2U))) & 0xFU) | (((scales_h >> (2U * ib)) & 3U) << 4U));
            r.s0 = d * static_cast<float>(ls - 32);
        }
#pragma unroll
        for (std::uint32_t k = 0; k < 4U; ++k) {
            const auto b0 = qs[4U * k], b1 = qs[4U * k + 1U], b2 = qs[4U * k + 2U], b3 = qs[4U * k + 3U];
            r.w[k] = pack4(hip_ggml::kvalues_iq4nl[b0 & 15U], hip_ggml::kvalues_iq4nl[b1 & 15U],
                           hip_ggml::kvalues_iq4nl[b2 & 15U], hip_ggml::kvalues_iq4nl[b3 & 15U]);
            r.w[k + 4U] = pack4(hip_ggml::kvalues_iq4nl[b0 >> 4U], hip_ggml::kvalues_iq4nl[b1 >> 4U],
                                hip_ggml::kvalues_iq4nl[b2 >> 4U], hip_ggml::kvalues_iq4nl[b3 >> 4U]);
        }
    } else if constexpr (T == type_iq3_s) {
        const std::uint32_t nib = (ib % 2U) == 0U ? (x[106U + ib / 2U] & 0x0FU) : (x[106U + ib / 2U] >> 4U);
        r.s0 = d * static_cast<float>(1 + 2 * static_cast<int>(nib));
        const std::uint32_t qh = x[66U + ib];
        const unsigned char* qs = x + 2U + ib * 8U;
        const unsigned char* signs = x + 74U + ib * 4U;
#pragma unroll
        for (std::uint32_t l = 0; l < 4U; ++l) {
            const std::uint32_t g1 = hip_ggml::iq3s_grid[qs[2U * l] | ((qh << (8U - 2U * l)) & 256U)];
            const std::uint32_t g2 = hip_ggml::iq3s_grid[qs[2U * l + 1U] | ((qh << (7U - 2U * l)) & 256U)];
            const std::uint32_t s = signs[l];
            int v[8];
#pragma unroll
            for (std::uint32_t j = 0; j < 4U; ++j) {
                v[j] = static_cast<int>((g1 >> (8U * j)) & 255U) * ((s >> j) & 1U ? -1 : 1);
                v[j + 4U] = static_cast<int>((g2 >> (8U * j)) & 255U) * ((s >> (j + 4U)) & 1U ? -1 : 1);
            }
            r.w[2U * l] = pack4(v[0], v[1], v[2], v[3]);
            r.w[2U * l + 1U] = pack4(v[4], v[5], v[6], v[7]);
        }
    } else if constexpr (T == type_iq3_xxs) {
        const std::uint32_t aux = hip_ggml::le32(x + 66U + 4U * ib);
        r.s0 = (d * (0.5F + static_cast<float>(aux >> 28U))) * 0.5F;
        const unsigned char* qs = x + 2U + ib * 8U;
#pragma unroll
        for (std::uint32_t l = 0; l < 4U; ++l) {
            const std::uint32_t s7 = (aux >> (7U * l)) & 127U;
            const std::uint32_t s = s7 | ((__popc(s7) & 1U) << 7U);
            const std::uint32_t g1 = hip_ggml::iq3xxs_grid[qs[2U * l]], g2 = hip_ggml::iq3xxs_grid[qs[2U * l + 1U]];
            int v[8];
#pragma unroll
            for (std::uint32_t j = 0; j < 4U; ++j) {
                v[j] = static_cast<int>((g1 >> (8U * j)) & 255U) * ((s >> j) & 1U ? -1 : 1);
                v[j + 4U] = static_cast<int>((g2 >> (8U * j)) & 255U) * ((s >> (j + 4U)) & 1U ? -1 : 1);
            }
            r.w[2U * l] = pack4(v[0], v[1], v[2], v[3]);
            r.w[2U * l + 1U] = pack4(v[4], v[5], v[6], v[7]);
        }
    } else {  // IQ2_S
        static_assert(T == type_iq2_s);
        const std::uint32_t sc = x[74U + ib], qh = x[66U + ib];
        r.s0 = (d * (0.5F + static_cast<float>(sc & 15U))) * 0.25F;
        r.s1 = (d * (0.5F + static_cast<float>(sc >> 4U))) * 0.25F;
        const unsigned char* qs = x + 2U + ib * 4U;
        const unsigned char* signs = x + 34U + ib * 4U;
#pragma unroll
        for (std::uint32_t l = 0; l < 4U; ++l) {
            const std::uint64_t grid = hip_ggml::iq2s_grid[qs[l] | ((qh << (8U - 2U * l)) & 0x300U)];
            const std::uint32_t s = signs[l];
            int v[8];
#pragma unroll
            for (std::uint32_t j = 0; j < 8U; ++j) v[j] = static_cast<int>((grid >> (8U * j)) & 255U) * ((s >> j) & 1U ? -1 : 1);
            r.w[2U * l] = pack4(v[0], v[1], v[2], v[3]);
            r.w[2U * l + 1U] = pack4(v[4], v[5], v[6], v[7]);
        }
    }
    return r;
}

// ---- dense decode (q8_rows dense A8, GSQ formats) ----
// Group kinds: one scale (v = s0 x int8), halves (s0 for weights 0..15, s1 for 16..31: Q3_K, Q6_K, IQ2_S),
// min (unsigned codes, value s0 * q - s1: Q4_K, Q5_K).
enum class DenseKind : unsigned { one, halves, min };
template <std::uint32_t T> constexpr DenseKind dense_kind() {
    return T == 11U || T == 14U || T == type_iq2_s ? DenseKind::halves : T == 12U || T == 13U ? DenseKind::min : DenseKind::one;
}
// Group g (32 weights) of a dense row of type T.
template <std::uint32_t T>
__device__ __forceinline__ IqGroup dense_group(const unsigned char* row, std::uint32_t g) {
    if constexpr (T == type_iq4_nl || T == type_iq4_xs) {
        return iq_group<T>(row, g);
    } else {
        constexpr std::uint32_t be = dense_block_elements(T), bb = dense_block_bytes(T);
        const unsigned char* x = row + static_cast<std::uint64_t>(g * 32U / be) * bb;
        const std::uint32_t ib = (g * 32U % be) / 32U;
        IqGroup r{};
        int v[32];
        if constexpr (T == 8U) {
            r.s0 = hip_ggml::half_bits(x);
#pragma unroll
            for (std::uint32_t j = 0; j < 32U; ++j) v[j] = static_cast<std::int8_t>(x[2U + j]);
        } else if constexpr (T == 2U || T == 6U) {  // Q4_0 / Q5_0
            r.s0 = hip_ggml::half_bits(x);
            const std::uint32_t qh = T == 6U ? hip_ggml::le32(x + 2U) : 0U;
            const unsigned char* qs = x + (T == 6U ? 6U : 2U);
#pragma unroll
            for (std::uint32_t j = 0; j < 16U; ++j) {
                if constexpr (T == 2U) {
                    v[j] = static_cast<int>(qs[j] & 15U) - 8;
                    v[j + 16U] = static_cast<int>(qs[j] >> 4U) - 8;
                } else {
                    v[j] = static_cast<int>((qs[j] & 15U) | (((qh >> j) & 1U) << 4U)) - 16;
                    v[j + 16U] = static_cast<int>((qs[j] >> 4U) | (((qh >> (j + 16U)) & 1U) << 4U)) - 16;
                }
            }
        } else if constexpr (T == type_q2_0) {
            r.s0 = hip_ggml::half_bits(x);
#pragma unroll
            for (std::uint32_t j = 0; j < 32U; ++j) {
                const std::uint32_t e = ib * 32U + j;
                v[j] = static_cast<int>((x[2U + e / 4U] >> (2U * (e % 4U))) & 3U) - 1;
            }
        } else if constexpr (T == 11U) {  // Q3_K: hmask[32], qs[64], scales[12], d; halves = the two 16-weight sub-blocks
            const std::uint32_t n = ib / 4U, j4 = ib % 4U;
            const float d_all = hip_ggml::half_bits(x + 108U);
#pragma unroll
            for (std::uint32_t sub = 0; sub < 2U; ++sub) {
                const std::uint32_t is = n * 8U + j4 * 2U + sub;
                const std::uint32_t low = is < 8U ? (x[96U + is] & 0x0FU) : (x[96U + is - 8U] >> 4U);
                const std::uint32_t high = (x[96U + 8U + is % 4U] >> (2U * (is / 4U))) & 3U;
                const float s = d_all * static_cast<float>(static_cast<int>(low | (high << 4U)) - 32);
                if (sub == 0U) r.s0 = s; else r.s1 = s;
#pragma unroll
                for (std::uint32_t l = 0; l < 16U; ++l) {
                    const auto q = (x[32U + n * 32U + sub * 16U + l] >> (2U * j4)) & 3U;
                    const bool set = (x[sub * 16U + l] & (1U << (n * 4U + j4))) != 0U;
                    v[sub * 16U + l] = static_cast<int>(q) - (set ? 0 : 4);
                }
            }
        } else if constexpr (T == 14U) {  // Q6_K: ql[128], qh[64], scales[16], d
            const std::uint32_t n = ib / 4U, quarter = ib % 4U;
            const float d = hip_ggml::half_bits(x + 208U);
#pragma unroll
            for (std::uint32_t half = 0; half < 2U; ++half) {
                const float s = d * static_cast<float>(static_cast<std::int8_t>(x[192U + n * 8U + half + 2U * quarter]));
                if (half == 0U) r.s0 = s; else r.s1 = s;
            }
#pragma unroll
            for (std::uint32_t l = 0; l < 32U; ++l) {
                const auto low = (x[n * 64U + l + (quarter & 1U) * 32U] >> (4U * (quarter >> 1U))) & 0x0FU;
                const auto high = (x[128U + n * 32U + l] >> (2U * quarter)) & 3U;
                v[l] = static_cast<int>(low | (high << 4U)) - 32;
            }
        } else {  // Q4_K / Q5_K: d, dmin, scales[12], [qh[32]], qs[128]; s0 = d * sc, s1 = dmin * m
            static_assert(T == 12U || T == 13U);
            std::uint32_t sc = 0, m = 0;
            hip_ggml::k_scale_min(x + 4U, ib, sc, m);
            r.s0 = hip_ggml::half_bits(x) * static_cast<float>(sc);
            r.s1 = hip_ggml::half_bits(x + 2U) * static_cast<float>(m);
            const unsigned char* qs = x + (T == 13U ? 48U : 16U) + (ib / 2U) * 32U;
#pragma unroll
            for (std::uint32_t l = 0; l < 32U; ++l) {
                std::uint32_t q = ib % 2U == 0U ? (qs[l] & 15U) : (qs[l] >> 4U);
                if constexpr (T == 13U) q += (x[16U + l] & (1U << ib)) != 0U ? 16U : 0U;
                v[l] = static_cast<int>(q);
            }
        }
#pragma unroll
        for (std::uint32_t k = 0; k < 8U; ++k) r.w[k] = pack4(v[4U * k], v[4U * k + 1U], v[4U * k + 2U], v[4U * k + 3U]);
        return r;
    }
}
// Dense A8 chain entry of one group for one activation row (codes a, block delta):
// one scale -> (s0 * delta, float(I)), the Q8_0 entry; halves -> ((s0 * delta) * float(I0) + (s1 * delta) * float(I1), 1);
// min -> ((s0 * delta) * float(I) - (s1 * delta) * float(S), 1), S = sum of the group's codes.
// Every operation rounds once, whatever the including TU's flags: the engine TUs (strix_hip_runtime) compile with the
// HIP default -ffp-contract=fast-honor-pragmas and the kernel's own pragma does not reach this inlined header code, so
// without this pragma u0 + u1 / u0 - u1 become an FMA whose fused operand the backend picks per instantiation (R1 and
// the 2..8-row windows then differ by one rounding, and neither is the contract).
template <std::uint32_t T>
__device__ __forceinline__ float2 dense_entry(const IqGroup& x, const int (&a)[8], float delta) {
#pragma clang fp contract(off)
    constexpr DenseKind kind = dense_kind<T>();
    if constexpr (kind == DenseKind::one) {
        int dot = 0;
#pragma unroll
        for (std::uint32_t k = 0; k < 8U; ++k) dot = sdot(a[k], x.w[k], dot);
        return make_float2(x.s0 * delta, static_cast<float>(dot));
    } else if constexpr (kind == DenseKind::halves) {
        int i0 = 0, i1 = 0;
#pragma unroll
        for (std::uint32_t k = 0; k < 4U; ++k) {
            i0 = sdot(a[k], x.w[k], i0);
            i1 = sdot(a[k + 4U], x.w[k + 4U], i1);
        }
        const float c0 = x.s0 * delta, c1 = x.s1 * delta;
        const float u0 = c0 * static_cast<float>(i0), u1 = c1 * static_cast<float>(i1);
        return make_float2(u0 + u1, 1.0F);
    } else {
        int dot = 0, sum = 0;
#pragma unroll
        for (std::uint32_t k = 0; k < 8U; ++k) {
            dot = sdot(a[k], x.w[k], dot);
            sum = sdot(a[k], 0x01010101, sum);
        }
        const float c0 = x.s0 * delta, c1 = x.s1 * delta;
        const float u0 = c0 * static_cast<float>(dot), u1 = c1 * static_cast<float>(sum);
        return make_float2(u0 - u1, 1.0F);
    }
}

// delta * (s * float(I)) of one group against the A8 image `a` (8 words); IQ2_S: two halves. Rounded per operation
// (as dense_entry; its users already compile with -ffp-contract=off, so their code is unchanged).
template <std::uint32_t T>
__device__ __forceinline__ float iq_apply(const IqGroup& x, const int* a, float delta) {
#pragma clang fp contract(off)
    if constexpr (T == type_iq2_s) {
        int i0 = sdot(a[0], x.w[0], 0); i0 = sdot(a[1], x.w[1], i0); i0 = sdot(a[2], x.w[2], i0); i0 = sdot(a[3], x.w[3], i0);
        int i1 = sdot(a[4], x.w[4], 0); i1 = sdot(a[5], x.w[5], i1); i1 = sdot(a[6], x.w[6], i1); i1 = sdot(a[7], x.w[7], i1);
        const float u0 = x.s0 * static_cast<float>(i0), u1 = x.s1 * static_cast<float>(i1);
        const float t = u0 + u1;
        return delta * t;
    } else {
        int integer = sdot(a[0], x.w[0], 0);
#pragma unroll
        for (std::uint32_t k = 1; k < 8U; ++k) integer = sdot(a[k], x.w[k], integer);
        const float scaled = x.s0 * static_cast<float>(integer);
        return delta * scaled;
    }
}

}  // namespace strix::weight::hip::gsq_detail
