#pragma once

#include "strix/vision/encoder.hpp"

#include <array>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace strix::vision::gate {

inline unsigned parse_attention_terms(std::string_view value) {
    if (value == "1") return 1U;
    if (value == "2") return 2U;
    throw std::invalid_argument("--attn-terms takes exactly 1 or 2");
}

inline void admit_attention_terms(std::optional<unsigned> terms, bool device, bool diagnose, bool hip_build) {
    if (terms && (!hip_build || (!device && !diagnose)))
        throw std::invalid_argument("--attn-terms needs the device backend or --diagnose in a device build");
}

inline EncoderOptions resolve_attention(EncoderOptions options, std::optional<unsigned> terms) {
    if (terms) options.attention_terms = *terms;
    return options;
}

// Both diagnostic arms inherit the resolved arithmetic and differ only in linear splitting.
inline std::array<EncoderOptions, 2> diagnostic_options(EncoderOptions options) {
    std::array<EncoderOptions, 2> arms{options, options};
    arms[0].split_activations = false;
    arms[1].split_activations = true;
    return arms;
}

inline ReferenceRounding parse_rounding(std::string_view value) {
    ReferenceRounding rounding;
    bool recognized = value == "none" || value == "linear" || value == "attention" || value == "both" ||
                      value == "attention2" || value == "both2" || value == "order1" || value == "order2";
    rounding.linear_inputs_f16 = value == "linear" || value == "both" || value == "both2";
    if (value == "attention" || value == "both") rounding.attention(1U);
    if (value == "attention2" || value == "both2") rounding.attention(2U);
    // order1 / order2: the dense core's real products and their order (ReferenceRounding::attn_order).
    if (value == "order1" || value == "order2") rounding.attention_order(value == "order2" ? 2U : 1U);
    if (value.starts_with("attention:") && value.size() == 14U) {
        unsigned char* terms[] = {&rounding.attn_q, &rounding.attn_k, &rounding.attn_v, &rounding.attn_p};
        for (unsigned i = 0; i < 4U; ++i) {
            const char c = value[10U + i];
            if (c < '0' || c > '2') throw std::invalid_argument("--rounding attention:QKVP takes digits 0, 1, 2");
            *terms[i] = static_cast<unsigned char>(c - '0');
        }
        recognized = true;
    }
    if (!recognized)
        throw std::invalid_argument("--rounding none|linear|attention|both|attention2|both2|attention:QKVP|order1|order2");
    return rounding;
}

}  // namespace strix::vision::gate
