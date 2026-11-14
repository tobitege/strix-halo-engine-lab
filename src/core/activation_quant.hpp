#pragma once
// IU4 g64 pack geometry and the W4A4 activation quantiser shared by every engine's FAST_I4 mode (lossy). The EXL3 ->
// IU4 weight converter itself is the strix-trellis library's convert_exl3_iu4 (wmma_tiled.hpp; one implementation,
// shared with the fused EXL3-resident IU4 loader), as are the IU4 GEMMs (grouped IU4, core/w4a16.hpp).
//   weights (library): the EXL3 tile decoded with mul1 (the FP16 values the exact WMMA cores multiply), then per output
//            column and K64 group s = FP16(max|w| / 7) (1 when all 0, floored at 2^-24), inv = 1 / f32(s),
//            q = clamp(rint(w * inv), -7, 7), nibbles in the trellis K16 x N16 tile order (core/iu4.hpp:
//            [K/16][N/16][column][K/2], low nibble = even k), all in the had128-rotated basis (the EXL3 suh / svh stay
//            with the caller: activations get suh + had128, the epilogue had128 * svh).
// Pack of one matrix: expert e's nibbles [K/16][N/16][16][8] bytes at wq + e * wq_expert, its FP16 scales [N][K/64] at
// ws + e * ws_expert (elements); the plain layout is wq_expert = iu4_weight_bytes, ws_expert = N * K / 64.
// tests/hip_iu4_convert_tests.hip checks the library converter byte for byte against a reference on exl3_trellis.hpp.
#include "../moe_core/exl3_trellis.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::core {

inline constexpr std::size_t iu4_weight_bytes(unsigned K, unsigned N) { return std::size_t{K} * N / 2U; }
inline constexpr std::size_t iu4_scale_bytes(unsigned K, unsigned N) { return std::size_t{N} * (K / 64U) * 2U; }

// Activation side (W4A4 prefill): v = this lane's FP16-exact values at channels begin + lane + 32 j of one 128-channel
// block (after the input transform). Two K64 groups (j 0-1, j 2-3): wave max, s = max / 7 (1 when 0), q = clamp(RNE(x /
// s), -7, 7); even lanes store the (even, odd) byte of row_q, lane 0 the two FP32 scales of row_s. The bits of the
// library quantiser (quantise_wmma_iu4_activations) on the same FP16 rows.
template<class Store> __device__ __forceinline__ void iu4_quant128_store(const float (&v)[4], unsigned begin, unsigned lane, Store store) {
#pragma unroll
    for (unsigned g = 0; g < 2U; ++g) {
        float amax = fmaxf(fabsf(v[2U * g]), fabsf(v[2U * g + 1U]));
#pragma unroll
        for (int delta = 16; delta > 0; delta /= 2) amax = fmaxf(amax, __shfl_xor(amax, delta));
        const float scale = amax == 0.0F ? 1.0F : amax / 7.0F;
#pragma unroll
        for (unsigned jj = 0; jj < 2U; ++jj) {
            const unsigned j = 2U * g + jj;
            const int qv = max(-7, min(7, static_cast<int>(nearbyintf(v[j] / scale))));
            const int odd = __shfl_xor(qv, 1);
            if (!(lane & 1U))
                store.byte((begin + lane + 32U * j) / 2U,
                    static_cast<std::uint8_t>((static_cast<unsigned>(qv) & 15U) | ((static_cast<unsigned>(odd) & 15U) << 4U)));
        }
        if (lane == 0U) store.scale(begin / 64U + g, scale);
    }
}

struct IU4RowStore {
    std::uint8_t* q; float* s;
    __device__ void byte(unsigned i, std::uint8_t v) const { q[i] = v; }
    __device__ void scale(unsigned i, float v) const { s[i] = v; }
};
__device__ __forceinline__ void iu4_quant128(const float (&v)[4], std::uint8_t* q, float* s, unsigned begin, unsigned lane) {
    iu4_quant128_store(v, begin, lane, IU4RowStore{q, s});
}

// Two-plane ("A8") twin of iu4_quant128: plane 1 is exactly iu4_quant128 (q, s); plane 2 holds the remainder
// r = v - q s (|r| <= s / 2) as q2 = clamp(RNE(r / s2), -7, 7) with s2 = s / 14, so v ~ q s + q2 s2 with a step of
// s / 14 (about 7.7 bits). Two IU4 GEMMs over the two planes add up to the W4A8 product; plane 1 alone is W4A4.
__device__ __forceinline__ void iu4_quant128_planes(const float (&v)[4], std::uint8_t* q, float* s, std::uint8_t* q2, float* s2,
                                                    unsigned begin, unsigned lane) {
#pragma unroll
    for (unsigned g = 0; g < 2U; ++g) {
        float amax = fmaxf(fabsf(v[2U * g]), fabsf(v[2U * g + 1U]));
#pragma unroll
        for (int delta = 16; delta > 0; delta /= 2) amax = fmaxf(amax, __shfl_xor(amax, delta));
        const float scale = amax == 0.0F ? 1.0F : amax / 7.0F;
        const float fine = scale / 14.0F;
#pragma unroll
        for (unsigned jj = 0; jj < 2U; ++jj) {
            const unsigned j = 2U * g + jj;
            const int qv = max(-7, min(7, static_cast<int>(nearbyintf(v[j] / scale))));
            const int rv = max(-7, min(7, static_cast<int>(nearbyintf((v[j] - static_cast<float>(qv) * scale) / fine))));
            const int odd = __shfl_xor(qv, 1), rodd = __shfl_xor(rv, 1);
            if (!(lane & 1U)) {
                const unsigned at = (begin + lane + 32U * j) / 2U;
                q[at] = static_cast<std::uint8_t>((static_cast<unsigned>(qv) & 15U) | ((static_cast<unsigned>(odd) & 15U) << 4U));
                q2[at] = static_cast<std::uint8_t>((static_cast<unsigned>(rv) & 15U) | ((static_cast<unsigned>(rodd) & 15U) << 4U));
            }
        }
        if (lane == 0U) {
            s[begin / 64U + g] = scale;
            s2[begin / 64U + g] = fine;
        }
    }
}

// Fake-quant twin of iu4_quant128 (quality studies): v <- q * s with the same s and q, in place.
__device__ __forceinline__ void iu4_fake128(float (&v)[4]) {
#pragma unroll
    for (unsigned g = 0; g < 2U; ++g) {
        float amax = fmaxf(fabsf(v[2U * g]), fabsf(v[2U * g + 1U]));
#pragma unroll
        for (int delta = 16; delta > 0; delta /= 2) amax = fmaxf(amax, __shfl_xor(amax, delta));
        const float scale = amax == 0.0F ? 1.0F : amax / 7.0F;
#pragma unroll
        for (unsigned jj = 0; jj < 2U; ++jj) {
            const unsigned j = 2U * g + jj;
            v[j] = static_cast<float>(max(-7, min(7, static_cast<int>(nearbyintf(v[j] / scale))))) * scale;
        }
    }
}

}  // namespace strix::core
