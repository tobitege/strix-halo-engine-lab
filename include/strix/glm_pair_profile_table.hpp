#pragma once

#include "strix/engine_profile.hpp"

#include <array>

namespace strix::glm::rank_defaults {

// Default/quality export nothing. The FAST column selects the qualified,
// bit-changing prefill identity; the existing WMMA startup vote checks both ranks.
inline constexpr std::array<strix::profile::Lever, 3> product_profile_table{{
    {"pf_wmma", "STRIX_GLM_PF_WMMA", "", {"", "1", ""}, "0", strix::profile::Kind::bit_changing,
     {true, true, true}, {true, true, true}, "glm-pf-wmma-x", "0",
     "docs/optimizations.md#glm-pf-wmma-x: quality passed at 1K/8K/32K; OFF equals exact anchors"},
    // PENDING turbo cells: not covered by pf_wmma (no code yet; the written analysis precedes any kernel).
    {"router_indexer_wmma", "STRIX_GLM_PF_ROUTER_INDEXER_WMMA", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {false, false, false, false}, "", "0",
     "pending: router and DSA indexer projections on WMMA (not covered by STRIX_GLM_PF_WMMA)"},
    {"kda_state16", "STRIX_GLM_PF_KDA_STATE16", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {false, false, false, false}, "", "0",
     "pending: KDA recurrent state in 16-bit (not covered by STRIX_GLM_PF_WMMA)"},
}};
inline const strix::profile::Engine product_profile_engine{"glm-pair", "", product_profile_table, false, "glm-turbo-v1"};

}  // namespace strix::glm::rank_defaults
