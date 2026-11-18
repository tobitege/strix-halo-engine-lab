#pragma once
// One host admission plan for GSQ (Q2_0 / IQ*) routed-expert and dense A8 operands.
// Composes the shared semantic definitions in strix/gsq_a8_groups.hpp (legal gate/down/dense
// types, block geometry); it never reads the environment: profile and switch values are inputs
// (STRIX_GSQ_Q20_A8 -> moe_q4k_a8::q20_a8_enabled(), STRIX_GSQ_WINDOW_EXPERTS/DENSE keep their
// meaning at the call site). Every predicate below is the exact pre-P07 predicate of its site.
#include "strix/gsq_a8_groups.hpp"
#include "registry.hpp"
#include <cstdint>
#include <stdexcept>

namespace strix::weight::hip {
struct GsqVerdict {
    const char* reason; // nullptr = admitted
    constexpr explicit operator bool() const noexcept { return reason == nullptr; }
};
inline constexpr GsqVerdict gsq_admit{nullptr};

// Block geometry of one operand type (routed experts or dense segments).
struct GsqGeometry {
    std::uint32_t elements, bytes;
    constexpr bool valid() const noexcept { return elements != 0U && bytes != 0U; }
};
constexpr GsqGeometry gsq_geometry(std::uint32_t type, bool dense = false) noexcept {
    namespace g = ::strix::gsq_a8;
    return dense ? GsqGeometry{g::dense_block_elements(type), g::dense_block_bytes(type)}
                 : GsqGeometry{g::block_elements(type), g::block_bytes(type)};
}

// Legal type roles and row capability of one routed-expert GSQ type.
struct GsqTypePlan {
    std::uint32_t type;
    bool gate, down, dense;
    GsqGeometry geometry;
};
constexpr GsqTypePlan gsq_type_plan(std::uint32_t type) noexcept {
    namespace g = ::strix::gsq_a8;
    return {type, g::gate_type(type), g::down_type(type), g::dense_type(type), gsq_geometry(type)};
}
// Rows of `width` elements are whole blocks, whole dwords, and 18-byte (Q2_0/IQ4_NL) blocks come
// in even/odd pairs: the capability of the GSQ F16 decoders (hip_moe_gsq_weights.inc;
// hip_moe_q4k_a8_known_rows.hip pairing gate). Rows of another geometry keep the A8 forms.
constexpr bool gsq_rows_whole_dword_paired(std::uint32_t type, std::uint32_t width) noexcept {
    namespace g = ::strix::gsq_a8;
    return width % g::block_elements(type) == 0U && g::row_bytes(type, width) % 4U == 0U &&
           (g::block_bytes(type) != 18U || (width / g::block_elements(type)) % 2U == 0U);
}
// Shared with the F16 registry and host tests. Whole blocks alone do not align
// a rebased IQ3_S (110-byte block), IQ2_S or IQ3_XXS source to a dword.
inline Status gsq_f16_launch(const WeightView& w, const LaunchShape& s, std::uint32_t type) {
    const auto geometry = gsq_geometry(type);
    if (!geometry.valid()) return {Error::unsupported_shape, "invalid GSQ geometry"};
    auto status = validate_launch(w, s, geometry.elements, geometry.bytes, geometry.bytes);
    if (!status) return status;
    if (!gsq_rows_whole_dword_paired(type, static_cast<std::uint32_t>(s.k)) ||
        s.row_stride_bytes % 4U || s.k_offset_bytes % 4U)
        return {Error::unsupported_shape, "HIP GSQ F16 source requires whole-dword rows and offsets and even 18-byte block pairs"};
    return {Error::ok, {}};
}
// GSQ gate/up + GSQ down prefill FP16 form capability (gate row = hidden, down row = rows).
constexpr bool gsq_f16_prefill_capable(std::uint32_t gate_type, std::uint32_t down_type,
                                       std::uint32_t hidden, std::uint32_t rows) noexcept {
    namespace g = ::strix::gsq_a8;
    return g::gate_type(gate_type) && g::down_type(down_type) &&
           gsq_rows_whole_dword_paired(gate_type, hidden) && gsq_rows_whole_dword_paired(down_type, rows);
}

// ---- routed-expert admission (hip_moe_experts_launcher.hip) ----
// The A8 integer down request: Q5_1 (a8_down_q51), Q8_0 (a8_down_q80) or a GSQ down type (q80 + STRIX_GSQ_Q20_A8).
constexpr bool gsq_integer_down(std::uint32_t down_type, bool a8_down_q51, bool a8_down_q80,
                                bool q20_enabled) noexcept {
    return (a8_down_q51 && down_type == 7U) || (a8_down_q80 && down_type == 8U) ||
           (a8_down_q80 && ::strix::gsq_a8::down_type(down_type) && q20_enabled);
}
constexpr GsqVerdict gsq_gate_needs_integer_down(std::uint32_t gate_type, bool integer_down) noexcept {
    return ::strix::gsq_a8::gate_type(gate_type) && !integer_down
               ? GsqVerdict{"Q2_0 A8 gate/up requires the A8 down profile"} : gsq_admit;
}
// Layer census exemption: separate GSQ gate/up of one type with a GSQ, Q5_1 or Q8_0 down.
constexpr bool gsq_q20_layer(bool q20_enabled, bool fused_gate_up, std::uint32_t gate_type,
                             std::uint32_t up_type, std::uint32_t down_type) noexcept {
    namespace g = ::strix::gsq_a8;
    return q20_enabled && !fused_gate_up && g::gate_type(gate_type) && up_type == gate_type &&
           (g::down_type(down_type) || down_type == 7U || down_type == 8U);
}
// Preparation owns the optional manifests: inspect the fused representation
// before evaluating any scalar arguments that require separate gate/up tensors.
template<class RoutedTensors>
void gsq_prepare_a8_down(const RoutedTensors& tensors, std::uint32_t layer, bool q20_enabled) {
    constexpr const char* refusal = "A8 down layer types differ from canonical fallback census";
    if (tensors.fused_gate_up) throw std::invalid_argument(refusal);
    const bool q80_layer = layer == 2U || layer == 4U || layer == 30U || layer == 46U || layer == 47U;
    const auto gate_type = layer == 2U ? 13U : 12U;
    if (!gsq_q20_layer(q20_enabled, false, tensors.gate.manifest->ggml_type,
                      tensors.up.manifest->ggml_type, tensors.down.manifest->ggml_type) &&
        (tensors.gate.manifest->ggml_type != gate_type || tensors.up.manifest->ggml_type != gate_type ||
         tensors.down.manifest->ggml_type != (q80_layer ? 8U : 7U)))
        throw std::invalid_argument(refusal);
}
// Known-window routed rows: serial Q4K / Q5K (q5k_a8g32_gate_up) / GSQ gate/up are the legal gates.
constexpr bool gsq_rows_gate_ok(std::uint32_t gate_type, bool q5k_gate_up, bool q20_enabled) noexcept {
    return gate_type == 12U || (gate_type == 13U && q5k_gate_up) ||
           (::strix::gsq_a8::gate_type(gate_type) && q20_enabled);
}
// Q4K A8 runtime gate/up: Q4K, Q5K or a GSQ gate under STRIX_GSQ_Q20_A8 (q20 returned for the caller).
constexpr bool gsq_q20_gate(std::uint32_t gate_type, bool q20_enabled) noexcept {
    return ::strix::gsq_a8::gate_type(gate_type) && q20_enabled;
}

// ---- compiled window admission (compiled_speculative_verify.cpp) ----
// q20 = profile.a8 && STRIX_GSQ_Q20_A8 && STRIX_GSQ_WINDOW_EXPERTS. `prior_gate` is the gate type
// already seen for this layer (0 when judging the gate itself).
struct GsqWindowProfile {
    bool q80, q5k, q20;
};
constexpr GsqVerdict gsq_window_expert(bool down, std::uint32_t type, std::uint32_t prior_gate,
                                       GsqWindowProfile p) noexcept {
    namespace g = ::strix::gsq_a8;
    if (down) {
        return type == 7U || (p.q80 && type == 8U) || (p.q20 && p.q80 && g::down_type(type))
                   ? gsq_admit : GsqVerdict{"routed down type is not a legal A8 down type for this profile"};
    }
    if (!(type == 12U || (p.q5k && type == 13U) || (p.q20 && g::gate_type(type))))
        return {"routed gate/up type is not a legal A8 gate type for this profile"};
    if (prior_gate && type != prior_gate) return {"routed up type differs from the gate type"};
    return gsq_admit;
}
// Dense typed segment: STRIX_GSQ_Q20_A8 and STRIX_GSQ_WINDOW_DENSE on, dense type, whole blocks.
constexpr GsqVerdict gsq_window_dense(std::uint32_t type, std::uint64_t width, bool q20_enabled,
                                      bool window_dense) noexcept {
    namespace g = ::strix::gsq_a8;
    if (!q20_enabled || !window_dense) return {"GSQ dense windows need STRIX_GSQ_Q20_A8 and STRIX_GSQ_WINDOW_DENSE"};
    if (!g::dense_type(type)) return {"dense type is not an A8 GSQ dense type"};
    if (width % g::dense_block_elements(type)) return {"dense width is not a whole number of blocks"};
    return gsq_admit;
}
// Legal gate/down pairs: any gate_type with any down_type admitted above; nothing narrower.
constexpr bool gsq_gate_down_legal(std::uint32_t gate_type, std::uint32_t down_type) noexcept {
    return ::strix::gsq_a8::gate_type(gate_type) &&
           (::strix::gsq_a8::down_type(down_type) || down_type == 7U || down_type == 8U);
}
} // namespace strix::weight::hip
