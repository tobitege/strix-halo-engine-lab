#pragma once

// Stored INT4 routed experts: which MoE layers and which matrices take the INT4 pack.
//
// A lossy INT4 pack is not equally cheap at every depth. Measured on GLM-5.3 (pooled KL against the FAST
// identity, three text domains): any single early MoE layer costs the same saturating step (about 0.007),
// late layers add about 0.0003 each. INT4 on the MoE ordinals >= B: B = 0: 0.0245, 10: 0.0184, 15: 0.0138,
// 20: 0.0088. The routed down projection is the largest single error source, so the default scope keeps it
// in the source format. Engines keep only the adapter: the environment names, the default spec and the load.
//
// Layer spec: "none" (no layer excluded) or a comma list of MoE ordinals and ranges a-b, 0 = the first MoE
// layer of the trunk. Ordinals are 0..63.

#include <cstdint>
#include <stdexcept>
#include <string>

namespace strix::moe_core {

struct Int4LayerPolicy {
    std::uint64_t source_mask = 0;  // bit o: MoE ordinal o keeps the source-format routed experts
    unsigned matrices = 2;          // 2 = gate, up; 3 = gate, up, down

    constexpr bool int4(unsigned ordinal) const { return ((source_mask >> (ordinal < 63U ? ordinal : 63U)) & 1U) == 0U; }
    constexpr bool down_int4() const { return matrices == 3U; }
    // INT4 layers among the first `layers` MoE ordinals.
    constexpr unsigned count(unsigned layers) const {
        unsigned n = 0;
        for (unsigned o = 0; o < layers; ++o) n += int4(o) ? 1U : 0U;
        return n;
    }
    // One word for a rank-agreement vote or a pack header: the mask folded to 24 bits, the scope and the scale rule.
    constexpr std::uint32_t tag(unsigned scale_rule) const {
        const std::uint64_t m = source_mask ^ (source_mask >> 24U) ^ (source_mask >> 48U);
        return (static_cast<std::uint32_t>(m & 0xffffffU) << 8U) ^ (matrices << 4U) ^ (scale_rule & 15U) ^ ((scale_rule >> 4U) << 28U);
    }
};

// `name` is the caller's option name, used in the error text.
inline std::uint64_t parse_int4_source_layers(const std::string& spec, const std::string& name) {
    if (spec == "none") return 0;
    const auto fail = [&name]() -> unsigned {
        throw std::invalid_argument(name + " must be none or a comma list of MoE ordinals / ranges a-b (0..63, a <= b)");
    };
    const auto ordinal = [&fail](const std::string& t) -> unsigned {
        if (t.empty() || t.size() > 2U || t.find_first_not_of("0123456789") != std::string::npos) return fail();
        const unsigned long v = std::stoul(t);
        return v > 63UL ? fail() : static_cast<unsigned>(v);
    };
    std::uint64_t mask = 0;
    for (std::size_t at = 0; at <= spec.size();) {
        const std::size_t comma = spec.find(',', at), end = comma == std::string::npos ? spec.size() : comma;
        const std::string item = spec.substr(at, end - at);
        const std::size_t dash = item.find('-');
        const unsigned lo = ordinal(item.substr(0, dash)), hi = dash == std::string::npos ? lo : ordinal(item.substr(dash + 1U));
        if (hi < lo) fail();
        for (unsigned o = lo; o <= hi; ++o) mask |= std::uint64_t{1} << o;
        at = end + 1U;
    }
    return mask;
}

inline unsigned parse_int4_scope(const std::string& scope, const std::string& name) {
    if (scope == "gate_up") return 2U;
    if (scope == "all") return 3U;
    throw std::invalid_argument(name + " must be gate_up or all");
}

}  // namespace strix::moe_core
