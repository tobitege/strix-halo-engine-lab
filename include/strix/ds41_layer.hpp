#pragma once

// deepseek41 layer sites outside attention and the MoE: hyper-connections (mHC) with the Sinkhorn
// mix, RMSNorm, YaRN RoPE and its inverse, the Q8_0 / F16 projection sites, the token embedding,
// the output head with a vocabulary-split argmax and the Engram gated injection.
//
// Everything here is the declared FP32 arithmetic of identity ds41-exact-v1: the host functions are
// the reference that the device kernels (hip_ds41_layer.hpp) reproduce bitwise. Two rules fix every
// reduction order:
//   * lane dot: lane l (0..31) takes elements l, l + 32, ... in ascending order with acc = acc + a * b
//     (one rounding each, no fused multiply-add), then the butterfly v[l] += v[l ^ d] for
//     d = 16, 8, 4, 2, 1; the result is lane 0;
//   * small fixed sums (hyper-connection copies, Sinkhorn rows and columns) run in ascending index.
// Transcendentals use declared_exp / declared_sqrt (ds41_moe.hpp): basic IEEE operations only.
//
// The residual stream is kept in FP32 as hc_mult copies of hidden, copy-major: element (c, i) is at
// c * hidden + i. Weights keep their GGUF layout: a weight [in, out] is `out` contiguous rows of `in`.

#include "strix/ds41_moe.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace strix::ds41 {

inline constexpr std::uint32_t hc_copies = 4;                            // hc_mult of this model
inline constexpr std::uint32_t hc_mix_count = (2U + hc_copies) * hc_copies;  // pre 4, post 4, comb 16

STRIX_DS41_HD inline float declared_sigmoid(float x) { return 1.0F / (1.0F + declared_exp(-x)); }
STRIX_DS41_HD inline float declared_rsqrt(float x) { return 1.0F / declared_sqrt(x); }
// IEEE half to float (exact; NaN payloads kept). CUDA uses hardware conversion with payload restoration;
// the host uses selects, and HIP keeps the qualified branch-based formula.
STRIX_DS41_HD inline float f16_to_float(std::uint16_t h) {
#if defined(__CUDA_ARCH__)
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    const bool nan = (h & 0x7c00U) == 0x7c00U && (h & 0x3ffU) != 0U;
    return nan ? __uint_as_float((static_cast<std::uint32_t>(h & 0x8000U) << 16U) | 0x7f800000U | (static_cast<std::uint32_t>(h & 0x3ffU) << 13U)) : f;
#elif defined(__HIPCC__)
    // Keep the qualified HIP implementation; CUDA and host use their own forms.
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
    std::uint32_t exponent = (h >> 10U) & 0x1fU, mantissa = h & 0x3ffU;
    if (exponent == 0x1fU) return __builtin_bit_cast(float, sign | 0x7f800000U | (mantissa << 13U));
    if (exponent == 0) {
        if (mantissa == 0) return __builtin_bit_cast(float, sign);
        // subnormal half: mantissa * 2^-24, exact in float
        const float v = static_cast<float>(mantissa) * 5.9604644775390625e-8F;
        return sign ? -v : v;
    }
    return __builtin_bit_cast(float, sign | ((exponent + 112U) << 23U) | (mantissa << 13U));
#else
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
    const std::uint32_t exponent = (h >> 10U) & 0x1fU, mantissa = h & 0x3ffU;
    const std::uint32_t special = sign | 0x7f800000U | (mantissa << 13U);
    // zero or subnormal half: mantissa * 2^-24, exact in float (+0 for a zero mantissa), with the sign bit
    const std::uint32_t small = sign | __builtin_bit_cast(std::uint32_t, static_cast<float>(mantissa) * 5.9604644775390625e-8F);
    const std::uint32_t normal = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    return __builtin_bit_cast(float, exponent == 0x1fU ? special : exponent == 0U ? small : normal);
#endif
}
// f16_to_float by selects instead of branches (so that loads ahead of it are not held behind a branch per
// value): the same bits for every input, NaN payloads included (the host tests compare all 65536).
STRIX_DS41_HD inline float f16_to_float_select(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
    const std::uint32_t exponent = (h >> 10U) & 0x1fU, mantissa = h & 0x3ffU;
    const std::uint32_t special = sign | 0x7f800000U | (mantissa << 13U);
    const std::uint32_t normal = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    const std::uint32_t small = sign | __builtin_bit_cast(std::uint32_t, static_cast<float>(mantissa) * 5.9604644775390625e-8F);
    return __builtin_bit_cast(float, exponent == 0x1fU ? special : (exponent == 0 ? small : normal));
}
// Round a float to the nearest bfloat16 (ties to even), returned as float. NaN stays NaN.
STRIX_DS41_HD inline float round_to_bf16(float x) {
    const auto bits = __builtin_bit_cast(std::uint32_t, x);
    if ((bits & 0x7f800000U) == 0x7f800000U && (bits & 0x7fffffU)) return x;
    const std::uint32_t rounded = (bits + 0x7fffU + ((bits >> 16U) & 1U)) & 0xffff0000U;
    return __builtin_bit_cast(float, rounded);
}

// sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)): the sign bit of dot (also of -0) carries over and a
// NaN dot stays NaN.
STRIX_DS41_HD inline float engram_gate_of(float dot) {
    const auto bits = __builtin_bit_cast(std::uint32_t, dot);
    const float a = __builtin_bit_cast(float, bits & 0x7fffffffU);
    const float root = declared_sqrt(a < 1e-6F ? 1e-6F : a);
    return declared_sigmoid(__builtin_bit_cast(float, __builtin_bit_cast(std::uint32_t, root) | (bits & 0x80000000U)));
}

// ---- reductions ---------------------------------------------------------------------------------
// Lane dot of a[i] * b[i] over n elements (the rule above). b == nullptr gives the sum of a[i] * a[i].
[[nodiscard]] float lane_dot(const float* a, const float* b, std::size_t n);
// Lane dot where a is produced on the fly: a(i) for i in [0, n).
template <class A, class B>
[[nodiscard]] float lane_dot_of(A a, B b, std::size_t n) {
    std::array<float, 32> v{};
    for (std::size_t l = 0; l < 32; ++l)
        for (std::size_t i = l; i < n; i += 32) v[l] = v[l] + a(i) * b(i);
    for (std::size_t d = 16; d; d >>= 1U) {
        std::array<float, 32> next{};
        for (std::size_t l = 0; l < 32; ++l) next[l] = v[l] + v[l ^ d];
        v = next;
    }
    return v[0];
}

// Chunked dot (the long reductions: RMSNorm, hyper-connection mixes, Engram gate): the n elements are
// cut into chunks of dot_chunk; each chunk is a lane dot of its own elements (lane l takes chunk
// elements l, l + 32, ...), and total = total + chunk_sum in ascending chunk order from 0. Every chunk
// can run in its own wave.
inline constexpr std::size_t dot_chunk = 640;
template <class A, class B>
[[nodiscard]] float chunked_dot_of(A a, B b, std::size_t n, std::size_t chunk = dot_chunk) {
    float total = 0.0F;
    for (std::size_t c0 = 0; c0 < n; c0 += chunk) {
        const std::size_t len = n - c0 < chunk ? n - c0 : chunk;
        total = total + lane_dot_of([&](std::size_t i) { return a(c0 + i); }, [&](std::size_t i) { return b(c0 + i); }, len);
    }
    return total;
}
// chunked_dot_of of a[i] * b[i]; b == nullptr gives the sum of squares.
[[nodiscard]] float chunked_dot(const float* a, const float* b, std::size_t n);

// ---- RMSNorm -----------------------------------------------------------------------------------
// y[i] = w[i] * (x[i] * r), r = 1 / sqrt(chunked_dot(x, x) / n + eps).
void rms_norm(std::span<const float> x, std::span<const float> weight, float eps, std::span<float> out);

// ---- hyper-connections --------------------------------------------------------------------------
// The coefficients of one sublayer: pre (collapse weights), post (expansion weights) and comb[j][k]
// (residual copy j into copy k), doubly stochastic after Sinkhorn.
struct HcMix {
    std::array<float, hc_copies> pre{}, post{};
    std::array<float, hc_copies * hc_copies> comb{};
};
struct HcSiteWeights {
    std::span<const std::uint16_t> fn;  // F16 [hc_mix_count][hc_copies * hidden]
    std::span<const float> base;        // [hc_mix_count]
    std::span<const float> scale;       // [3]
};
// mixes[m] = chunked_dot(fn[m], stream) * r, r = 1 / sqrt(chunked_dot(stream, stream) / (hc * hidden) + norm_eps);
// then the split: pre = sigmoid(mixes * scale0 + base) + hc_eps, post = 2 * sigmoid(mixes * scale1 + base),
// comb = row softmax (max-subtracted, exp, ascending row sum, divide) + hc_eps, column normalise, and
// iters - 1 rounds of (row normalise, column normalise), each c / (ascending sum + hc_eps).
[[nodiscard]] std::array<float, hc_mix_count> hc_raw_mixes(std::span<const float> stream, const HcSiteWeights& site,
                                                           float norm_eps);
[[nodiscard]] HcMix hc_split_sinkhorn(const std::array<float, hc_mix_count>& mixes, std::span<const float> base,
                                      std::span<const float> scale, std::uint32_t iters, float hc_eps);
[[nodiscard]] HcMix hc_mixes(std::span<const float> stream, const HcSiteWeights& site, std::uint32_t iters, float hc_eps,
                             float norm_eps);
// The first layer's collapse uses copy 0 only.
[[nodiscard]] std::array<float, hc_copies> hc_identity_pre();
// out[i] = sum over c ascending of pre[c] * stream[c][i] (starting from 0).
void hc_collapse(std::span<const float> stream, std::span<const float, hc_copies> pre, std::span<float> out);
// out[k][i] = post[k] * sub[i] + s, s = sum over j ascending of comb[j][k] * residual[j][i] (from 0).
void hc_expand(std::span<const float> sub, std::span<const float> residual, const HcMix& mix, std::span<float> out);

// ---- RoPE (partial, adjacent pairs) ---------------------------------------------------------------
// Frequencies on the host in double: f_p = theta^(-2p / dims); with YaRN (original > 0) the
// reference's ramp between corrected dims of beta_fast and beta_slow blends f / factor and f. The
// angle position * f is evaluated in double and cos / sin are rounded once to FP32.
struct RopeConfig {
    std::uint32_t dims = 64;
    double theta = 10000.0;
    double original_positions = 0.0;  // 0 disables YaRN
    double factor = 1.0, beta_fast = 32.0, beta_slow = 1.0;
};
[[nodiscard]] std::vector<double> rope_frequencies(const RopeConfig& config);
// cos and sin for positions [first, first + count): [count][dims / 2] each.
struct RopeTable {
    std::uint32_t pairs = 0;
    std::uint64_t first = 0, count = 0;
    std::vector<float> cos, sin;
};
[[nodiscard]] RopeTable rope_table(const RopeConfig& config, std::uint64_t first, std::uint64_t count);
// Rotate pairs (2p, 2p + 1) of x (dims values): (a, b) -> (a c - b s, a s + b c); inverse uses -s.
// Each component is two products and one add or subtract, rounded once each.
void apply_rope(std::span<float> x, const RopeTable& table, std::uint64_t position, bool inverse);

// ---- projection sites ---------------------------------------------------------------------------
// Every Q8_0 linear (attn_q_a, attn_q_b, attn_kv, attn_output_a, attn_output_b, output) takes A8
// activations (quantize_a8) and is product_q8_0 (ds41_formats.hpp): the shared expert's contract.
// out[o] = product_q8_0(row first + o, x_g) for o in [0, count), x_g the g-th width-slice of x with
// g = o / per_group (per_group = count for a plain projection; the grouped attn_output_a uses 1024).
void q8_0_a8_rows(const std::uint8_t* weight, std::size_t width, std::size_t first, std::size_t count, std::size_t per_group,
                  const A8Activation& x, float* out);
// The A8 activation of a slice [first, first + count) of x (count a multiple of 32, first too).
[[nodiscard]] A8Activation a8_slice(const A8Activation& x, std::size_t first, std::size_t count);
// K-split Q8_0 site (attn_output_b with the attention split over two ranks): the width is cut into
// `parts` equal ranges (each a multiple of 32), each range is a product_q8_0 of its own columns and
// activation slice, and out[o] = h_0 + h_1 + ... in ascending part order (h_0 first, no leading zero).
void q8_0_a8_split_rows(const std::uint8_t* weight, std::size_t width, std::size_t count, std::size_t parts, const A8Activation& x,
                        float* out);
// F16 row (width values) times FP32 x by the lane dot.
[[nodiscard]] float f16_row_dot(const std::uint16_t* row, const float* x, std::size_t width);
void f16_rows(const std::uint16_t* weight, std::size_t width, std::size_t first, std::size_t count, const float* x,
              float* out);

// ---- fused pieces (each equal to the composition of the steps it fuses) ---------------------------
// Sublayer input: collapse with `pre`, RMSNorm, then A8 of the normalised row; `normed` gets the FP32 row.
[[nodiscard]] A8Activation collapse_norm_a8(std::span<const float> stream, std::span<const float, hc_copies> pre,
                                            std::span<const float> weight, float eps, std::span<float> normed);
// Attention output: inverse RoPE of every head's tail (heads x head_dim, tail = last 2 * pairs values),
// then A8 of the whole row.
[[nodiscard]] A8Activation inverse_rope_a8(std::span<float> o, std::uint32_t heads, std::uint32_t head_dim,
                                           const RopeTable& table, std::uint64_t position);

// ---- embedding and head ------------------------------------------------------------------------
// The F16 row of `token` as FP32, written to every hyper-connection copy.
void embed_token(const std::uint16_t* table, std::size_t hidden, std::uint32_t token, std::span<float> stream);
// Greedy selection over a vocabulary slice: the largest finite logit, ties to the lowest global
// index; finite is false (and the candidate unusable) when any logit of the slice is not finite. A
// slice without a finite logit gives value -inf and index 0xffffffff.
struct ArgmaxCandidate {
    float value = 0.0F;
    std::uint32_t index = 0;
    std::uint32_t finite = 1;
};
[[nodiscard]] ArgmaxCandidate argmax_slice(std::span<const float> logits, std::uint32_t first_index);
// Combine two ranks' candidates: larger value, equal values -> lower index; any non-finite slice
// makes the result non-finite. Equal to the single-node argmax over the concatenated slices.
[[nodiscard]] ArgmaxCandidate combine_argmax(const ArgmaxCandidate& a, const ArgmaxCandidate& b);
// Vocabulary split of the head across ranks: rank r owns rows [first(r), first(r) + count(r)).
struct VocabSlice { std::uint32_t first = 0, count = 0; };
[[nodiscard]] VocabSlice vocab_slice(std::uint32_t vocab, std::uint32_t ranks, std::uint32_t rank);

// ---- Engram injection ----------------------------------------------------------------------------
// Rows are decoded (decode_engram_row) and rounded to bfloat16 before the projection, as the model
// was trained. kv = engram_kv * rows (F16 [cols][hidden * (hc + 1)]): key c = kv[c * hidden ...],
// value = kv[hc * hidden ...]. Per copy c, with w = q_norm[c] * k_norm[c] (one product per element):
//   dot = (chunked_dot(h * w, key) * (rh * rk)) * inv_sqrt_hidden, element term (h[i] * w[i]) * key[i],
//   rh = 1 / sqrt(chunked_dot(h, h) / hidden + eps), rk likewise for key, inv_sqrt_hidden = float(1 / sqrt(hidden))
//   evaluated in double; gate = engram_gate_of(dot); h[i] = h[i] + gate * value[i].
void engram_rows_to_bf16(std::span<const std::uint8_t> rows, std::span<float> out);
// gate_off: an image position (upstream token_mask false): every gate is 0, so h[i] = h[i] + 0 * value[i].
void engram_inject(std::span<float> stream, std::span<const float> kv, std::span<const float> q_norm,
                   std::span<const float> k_norm, float eps, bool gate_off = false);
// The per-copy gates of engram_inject (for tests).
[[nodiscard]] std::array<float, hc_copies> engram_gates(std::span<const float> stream, std::span<const float> kv,
                                                        std::span<const float> q_norm, std::span<const float> k_norm,
                                                        float eps);

} // namespace strix::ds41
