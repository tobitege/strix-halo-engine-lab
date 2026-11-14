#pragma once

// g4-exl3-v1 is an opt-in arithmetic identity. It never claims BF16/Q4 equality.
// EXL3 mul1 codebook/ring semantics follow the independently implemented
// strix-trellis codec (Apache-2.0); see notices/ and README.md for provenance.
// Input: RN16(RN32(x*suh)), ascending-stride normalized H128 in FP32,
// then RN16. Core: decoded FP16 weights, g4-v1 separate multiply/add chunk16 dots.
// Finite FP16 products are exact FP32: a compiler's mixed-half FMA lowering is
// equivalent here. This does not permit contraction of general FP32 products.
// Output: ascending-stride normalized H128 in FP32, RN32(*svh), FP32 result.
// All other target operations, KV and selection retain g4-v1 arithmetic.
// Overflow/nonfinite at any conversion or transform is a hard failure.
// WMMA uses the same transforms and a separately named prefill identity.
#include "core/trellis_format.hpp"
#include "gemma4/contracts/g4_types.hpp"
#include "gemma4/prefill/prefill.hpp"
#include <atomic>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace strix::gemma4::exl3 {
inline constexpr char identity[] = "g4-exl3-v1";
// g4-exl3-v2 keeps v1's codec, scales, transforms and FP16 boundaries and
// replaces only the core dot order (cooperative tile layout):
// splits S = v2_splits(I, O) partition the I/16 chunks into equal contiguous
// ranges. In split s, for output o, lane a in 0..3 accumulates from +0, in
// increasing chunk k and then e in (2a, 2a+1, 2a+8, 2a+9), acc = acc + w*x
// (exact FP16 product, one FP32 rounding). The split partial is
// (acc0 + acc2) + (acc1 + acc3); the core value is p0, then + p1, + p2, ...
// in increasing s. Selected process-wide with STRIX_G4_EXL3_IDENTITY=v2.
inline constexpr char identity_v2[] = "g4-exl3-v2";
// g4-exl3-v3 is v2 with splits capped at 8 instead of 4 (same order otherwise).
inline constexpr char identity_v3[] = "g4-exl3-v3";
// Process-wide core identity: 1 (v1), 2 (v2) or 3 (v3), fixed once per process
// (one pack and one session), never per step.
// STRIX_G4_EXL3_IDENTITY=v1|v2|v3 forces it; unset or "auto" lets the app call
// configure_identity() with the pack's arm and the session's maximum concurrency,
// which applies auto_version(); without that call the identity is v1.
[[nodiscard]] inline unsigned forced_version() {
    static const unsigned forced = [] {
        const auto* p = std::getenv("STRIX_G4_EXL3_IDENTITY");
        if (!p || std::string_view(p) == "auto") return 0U;
        if (std::string_view(p) == "v1") return 1U;
        if (std::string_view(p) == "v2") return 2U;
        if (std::string_view(p) == "v3") return 3U;
        throw std::invalid_argument("STRIX_G4_EXL3_IDENTITY must be auto, v1, v2 or v3");
    }();
    return forced;
}
// Measured crossover (same-turn interleaved timing with grouped launches,
// measured per pack): the cooperative core is chosen only where it was not
// slower at the session's maximum concurrency and its quality is within noise of v1.
// M4: v3 at every concurrency (C1 +1.7 %, C4 +6.5 %, C8 +17 %).
// M5: v3 at >= 8 streams (C8 +4 %; C4 -4 %). M6: v3 at every concurrency
// (C1 +7 %, C4 +5 %, C8 +22 %). Unknown arms: v1.
[[nodiscard]] constexpr unsigned auto_version(std::string_view arm, unsigned max_batch) {
    if (arm == "M4" || arm == "M6") return 3U;
    if (arm == "M5" && max_batch >= 8) return 3U;
    return 1U;
}
inline std::atomic<unsigned>& identity_slot() {
    static std::atomic<unsigned> slot{0};
    return slot;
}
// Fixes the identity for this process; a later different request is refused.
inline unsigned configure_identity(std::string_view arm, unsigned max_batch) {
    const unsigned want = forced_version() ? forced_version() : auto_version(arm, max_batch);
    unsigned current = 0;
    if (!identity_slot().compare_exchange_strong(current, want) && current != want)
        throw std::logic_error("EXL3 identity is already fixed for this process");
    return want;
}
[[nodiscard]] inline unsigned core_version() {
    unsigned current = identity_slot().load();
    if (current) return current;
    const unsigned want = forced_version() ? forced_version() : 1U;
    identity_slot().compare_exchange_strong(current, want);
    return identity_slot().load();
}
// True for the cooperative-core identities (v2, v3).
[[nodiscard]] inline bool v2_active() { return core_version() >= 2; }
[[nodiscard]] inline const char* active_identity() {
    return core_version() == 3 ? identity_v3 : core_version() == 2 ? identity_v2 : identity;
}
// Splits of the active cooperative identity (cap 4 for v2, 8 for v3).
inline unsigned v2_splits(unsigned inputs, unsigned outputs) {
    const unsigned tiles = outputs / 16U, chunks = inputs / 16U, cap = core_version() == 3 ? 8U : 4U;
    unsigned s = 1;
    while (s < cap && s * 2U * tiles <= 1024U && chunks % (s * 2U) == 0 && chunks / (s * 2U) >= 16U) s *= 2U;
    return s;
}
inline constexpr char prefill_identity[] = "g4-exl3-wmma-pf-v1";
inline constexpr float hscale = 0.08838834613561630249F;

// One contiguous payload: suh[I] F16, svh[O] F16, then
// trellis[I/16,O/16,16*K] I16, in the source tile order, without padding.
// row_bytes records I*K/8, the nominal packed core bytes per logical output.
// It is not a row stride for this format. Use weight_storage_bytes to upload.
STRIX_G4_HD inline unsigned bits(LinearView w) { return unsigned(w.row_bytes * 8 / w.inputs); }
STRIX_G4_HD inline bool geometry(LinearView w) {
    return w.data && w.format == WeightFormat::exl3_mul1 && w.inputs && w.outputs &&
           w.inputs % 128 == 0 && w.outputs % 128 == 0 &&
           (bits(w) == 4 || bits(w) == 5 || bits(w) == 6 || bits(w) == 8) &&
           w.row_bytes == std::size_t(w.inputs) * bits(w) / 8;
}
STRIX_G4_HD inline std::uint32_t word(const std::uint8_t* p) {
    return std::uint32_t(load16(p)) | (std::uint32_t(load16(p + 2)) << 16U);
}
// Ring positions, state extraction and the mul1 constants are the shared format's (core/trellis_format.hpp).
STRIX_G4_HD inline unsigned stream_position(unsigned row, unsigned col) { return core::trellis::Format::position(row, col); }
STRIX_G4_HD inline unsigned state(const std::uint8_t* tile, unsigned k, unsigned position) {
    return core::trellis::Format::state_bytes(tile, int(k), position);
}
STRIX_G4_HD inline float codebook(unsigned s) {
    using Book = core::trellis::Mul1Constants;
    // Both operations are exact in FP32, so this also models the half FMA
    // with one final RNE conversion, without relying on host half support.
    const float value = float(1024U + Book::state_sum(s)) * fp16_value(Book::kScale) + fp16_value(Book::kBias);
    std::uint16_t half = 0;
    (void)prefill::half_bits(value, half);
    return fp16_value(half);
}
STRIX_G4_HD inline float rotated_at(LinearView w, unsigned input, unsigned output) {
    const auto k = bits(w);
    const auto* tile = w.data + 2ULL * (w.inputs + w.outputs) +
                       (std::size_t(input / 16U) * (w.outputs / 16U) + output / 16U) * 32U * k;
    return codebook(state(tile, k, stream_position(input % 16U, output % 16U)));
}
void validate(LinearView w);
void hadamard(float* block);
void input_transform(LinearView w, const float* x, unsigned rows, float* out);
void output_transform(LinearView w, float* core, unsigned rows);
void core_linear(LinearView w, const float* transformed, unsigned rows, float* out, unsigned begin, unsigned end);
void linear(LinearView w, const float* x, unsigned rows, float* out);
} // namespace strix::gemma4::exl3
