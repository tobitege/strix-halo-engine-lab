#pragma once
// FAST identity glm-pf-attn-wmma, host side (no HIP): the prefill MLA/DSA attention core (scores, softmax, o-bar) as
// one fused bf16 WMMA flash kernel (src/attn_core/sparse_flash_wmma.hpp) instead of the exact four-kernel form.
// STRIX_GLM_PF_ATTN_WMMA: unset = follow the FAST prefill switch (on with STRIX_GLM_PF_WMMA=1, off in exact mode, so
// exact bytes are unchanged); 0 = exact kernels (leave-one-out inside FAST); 1 = FAST core. Heads 32 per rank only
// (other head counts keep the exact kernels and are not counted). Other values fail closed.
// STRIX_GLM_PF_ATTN_WMMA_T (16 | 32, default 16): key block of the fused core (tuning; same identity family). Layer
// bench (C 2048, gfx1151): fused core 6.73 / 7.15 us/row at T 16 vs 7.47 / 8.86 at T 32 (positions 6144 / 30720),
// exact scores + softmax + o-bar 17.3 / 22.9 us/row; o-bar rel L2 vs exact 1.9e-3.
// Part of the TP2 startup agreement (rank_defaults::kPfAttnWmmaConfig): both ranks must resolve the same value.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "glm/prefill/glm_pf_wmma.hpp"

namespace strix::glm::pf::attn_wmma {
inline bool enabled() {
    const char* p = std::getenv("STRIX_GLM_PF_ATTN_WMMA");
    if (p == nullptr || *p == '\0') return wmma_x::enabled();
    if (std::strcmp(p, "0") == 0) return false;
    if (std::strcmp(p, "1") == 0) return true;
    throw std::invalid_argument("STRIX_GLM_PF_ATTN_WMMA must be 0 or 1");
}
inline unsigned block_keys() {
    const char* p = std::getenv("STRIX_GLM_PF_ATTN_WMMA_T");
    if (p == nullptr || *p == '\0' || std::strcmp(p, "16") == 0) return 16U;
    if (std::strcmp(p, "32") == 0) return 32U;
    throw std::invalid_argument("STRIX_GLM_PF_ATTN_WMMA_T must be 16 or 32");
}
inline std::atomic<std::uint64_t> calls{0};  // fused-core launches (identity witness)
}  // namespace strix::glm::pf::attn_wmma
