#pragma once

#include "strix/engine_profile.hpp"

#include <array>

namespace strix::glm::rank_defaults {

// Default/quality export nothing. The FAST column selects the qualified,
// bit-changing prefill identity; the existing WMMA startup vote checks both ranks.
inline constexpr std::array<strix::profile::Lever, 1> product_profile_table{{
    {"pf_wmma", "STRIX_GLM_PF_WMMA", "", {"", "1", ""}, "0", strix::profile::Kind::bit_changing,
     {true, true, true}, {true, true, true}, "glm-pf-wmma-x", "0",
     "docs/optimizations.md#glm-pf-wmma-x: quality passed at 1K/8K/32K; OFF equals exact anchors"},
}};
inline const strix::profile::Engine product_profile_engine{"glm-pair", "", product_profile_table};

}  // namespace strix::glm::rank_defaults
