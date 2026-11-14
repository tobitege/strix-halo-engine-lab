#pragma once
// GLM prefill variant selectors (STRIX_GLM_PF_*): the single table of accepted values and defaults.
// The engine selectors (glm_prefill.hip) and the rank resolver / effective-config printer
// (apps/glm_rank_defaults.hpp) both use it, so an unknown value fails at startup instead of
// silently selecting the default, and the printer reports the variant that actually runs.
// Unset = default. Any set value (including empty) must be one of `allowed`.

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::glm::pf {

struct PfVariant {
    const char* env;
    const char* allowed;  // '|'-separated decimal versions, default first
    unsigned def;
};

inline constexpr PfVariant kPfVariants[] = {
    {"STRIX_GLM_PF_HC", "2|1", 2U},
    {"STRIX_GLM_PF_MLA", "2|1", 2U},
    {"STRIX_GLM_PF_IDX", "3|2|1", 3U},
    {"STRIX_GLM_PF_SELECT", "2|1", 2U},
    {"STRIX_GLM_PF_OBAR", "10|7|6|5|4|2", 10U},
    {"STRIX_GLM_PF_MLA_GEMM", "3|2|1", 3U},
    {"STRIX_GLM_PF_SCORES", "7|6|5|4|3|2", 7U},
    {"STRIX_GLM_PF_KDA_SCAN", "5|3|1", 5U},
    {"STRIX_GLM_PF_DENSE_ROWS", "4|3|2|1", 4U},
    // Physical form of dense rows version 4 (core::fma_rows::Form): 0 = automatic per shape, 1..7 force one form
    // where the shape admits it. A selector of the same table so startup rejects an unknown value.
    {"STRIX_GLM_PF_DENSE_FORM", "0|1|2|3|4|5|6|7", 0U},
};

inline const PfVariant* pf_variant_find(std::string_view env) {
    for (const auto& p : kPfVariants)
        if (env == p.env) return &p;
    return nullptr;
}

// Version selected by `value` (nullptr or empty = unset = default, matching the defaults contract where an
// explicit empty setting is not exported); throws std::invalid_argument when unknown.
inline unsigned pf_variant_parse(const PfVariant& p, const char* value) {
    if (value == nullptr || *value == '\0') return p.def;
    const std::string_view v(value);
    std::string_view rest(p.allowed);
    while (!rest.empty()) {
        const std::size_t bar = rest.find('|');
        const std::string_view tok = rest.substr(0, bar);
        if (tok == v) return static_cast<unsigned>(std::stoul(std::string(tok)));
        rest = bar == std::string_view::npos ? std::string_view() : rest.substr(bar + 1);
    }
    std::string allowed(p.allowed);
    for (char& c : allowed)
        if (c == '|') c = ',';
    throw std::invalid_argument(std::string(p.env) + "='" + std::string(v) + "' is not a known variant (allowed: " +
                                allowed + "; unset = " + std::to_string(p.def) + ")");
}

inline unsigned pf_variant_from_env(const PfVariant& p) { return pf_variant_parse(p, std::getenv(p.env)); }

}  // namespace strix::glm::pf
