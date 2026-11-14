#pragma once
// FAST identity glm-pf-wmma-x, host side (no HIP): calls that actually ran on the WMMA linears, so a run's identity
// line reports executed arithmetic, not only the requested switch (STRIX_GLM_PF_WMMA=1 still leaves the dense calls
// below the X3 threshold and the MoE calls below V11 on the exact cores).
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strix::glm::pf::wmma_x {
// STRIX_GLM_PF_WMMA=1 selects this identity for the exact prefill X3 calls (dense and MoE); unset or 0 = exact.
// Part of the TP2 startup agreement (rank_defaults::kPfWmmaConfig): both ranks must resolve the same value.
inline bool enabled() {
    const char* p = std::getenv("STRIX_GLM_PF_WMMA");
    return p && std::strcmp(p, "1") == 0;
}
inline std::atomic<std::uint64_t> dense_calls{0}, moe_calls{0}, iu4_moe_calls{0};
}  // namespace strix::glm::pf::wmma_x

namespace strix::glm::pf::kda_wmma {
// FAST identity glm-pf-kda-wmma-v1 (glm_pf_kda_wmma.hpp): on by default when the FAST prefill route is selected
// (STRIX_GLM_PF_WMMA=1); STRIX_GLM_PF_KDA_WMMA=0 opts out, =1 forces it. The exact route (FAST not selected,
// switch unset) never takes it. Part of the TP2 startup agreement (rank_defaults::kPfKdaWmmaConfig).
inline bool enabled() {
    const char* p = std::getenv("STRIX_GLM_PF_KDA_WMMA");
    if (p == nullptr || *p == '\0') return wmma_x::enabled();
    if (std::strcmp(p, "0") == 0) return false;
    if (std::strcmp(p, "1") == 0) return true;
    throw std::invalid_argument("STRIX_GLM_PF_KDA_WMMA must be 0 or 1");
}
inline std::atomic<std::uint64_t> calls{0};  // launches that ran on the WMMA core (identity reporting)
}  // namespace strix::glm::pf::kda_wmma
