#pragma once

// GSQ-RCO A8 group contract for the lookup/scale formats of the IQ3_S build.
// One A8 activation group = 32 consecutive weights of a row. For ggml type T the group decodes to 32 signed
// int8 weights v_j (grid value x sign, or the IQ4 codebook value) and one float scale s (two, s0 for weights
// 0..15 and s1 for 16..31, for IQ2_S), computed with llama.cpp's float expressions:
//   IQ4_NL  (20): s = d                                   v = kvalues_iq4nl[code]
//   IQ4_XS  (23): s = d * float(ls - 32)                  v = kvalues_iq4nl[code]
//   IQ3_S   (21): s = d * float(1 + 2 * nibble)           v = +-iq3s_grid byte
//   IQ3_XXS (18): s = (d * (0.5f + float(aux >> 28))) * 0.5f      v = +-iq3xxs_grid byte
//   IQ2_S   (22): s0/s1 = (d * (0.5f + float(nibble))) * 0.25f    v = +-iq2s_grid byte
// The group value is delta * (s * float(I)) (IQ2_S: delta * ((s0 * float(I0)) + (s1 * float(I1)))), I = sum v_j c_j
// exact in int32 (s8 x s8 dot4). Lanes and trees are the Q2_0 ones (q20_numerical_identity).

#include <cstdint>
#include "strix/weight/formats/ggml.hpp"

namespace strix::gsq_a8 {

inline constexpr std::uint32_t type_iq3_xxs = 18, type_iq4_nl = 20, type_iq3_s = 21, type_iq2_s = 22, type_iq4_xs = 23,
                               type_q2_0 = 42;

// Gate/up and down weight types admitted by the GSQ A8 path (STRIX_GSQ_Q20_A8=1).
inline constexpr bool gate_type(std::uint32_t t) {
    return t == type_q2_0 || t == type_iq2_s || t == type_iq3_xxs || t == type_iq3_s || t == type_iq4_xs;
}
inline constexpr bool down_type(std::uint32_t t) { return t == type_q2_0 || t == type_iq4_nl; }
inline constexpr bool iq_type(std::uint32_t t) {
    return t == type_iq2_s || t == type_iq3_xxs || t == type_iq3_s || t == type_iq4_xs || t == type_iq4_nl;
}
// Block geometry: (elements, bytes).
inline constexpr std::uint32_t block_elements(std::uint32_t t) { return weight::formats::gsq_block_elements(t); }
inline constexpr std::uint32_t block_bytes(std::uint32_t t) {
    return weight::formats::gsq_block_bytes(t);
}
inline constexpr std::uint64_t row_bytes(std::uint32_t t, std::uint32_t width) {
    return static_cast<std::uint64_t>(width / block_elements(t)) * block_bytes(t);
}
// Dense A8 GSQ types (q8_rows typed segments): Q8_0, Q4_0, Q5_0, Q3_K, Q4_K, Q5_K, Q6_K, Q2_0, IQ4_NL, IQ4_XS.
inline constexpr std::uint32_t dense_block_elements(std::uint32_t t) {
    return t == 8U || t == 2U || t == 6U || t == type_iq4_nl ? 32U : t == type_q2_0 ? 64U : 256U;
}
inline constexpr std::uint32_t dense_block_bytes(std::uint32_t t) {
    return t == 8U ? 34U : t == 2U ? 18U : t == 6U ? 22U : t == 11U ? 110U : t == 12U ? 144U : t == 13U ? 176U :
           t == 14U ? 210U : block_bytes(t);
}
inline constexpr std::uint64_t dense_row_bytes(std::uint32_t t, std::uint32_t width) {
    return static_cast<std::uint64_t>(width / dense_block_elements(t)) * dense_block_bytes(t);
}
inline constexpr bool dense_type(std::uint32_t t) {
    return t == 8U || t == 2U || t == 6U || t == 11U || t == 12U || t == 13U || t == 14U || t == type_q2_0 ||
           t == type_iq4_nl || t == type_iq4_xs;
}


}  // namespace strix::gsq_a8
