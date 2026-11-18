#pragma once
// FAST identity glm-pf-wmma-x, host side (no HIP): calls that actually ran on the WMMA linears, so a run's identity
// line reports executed arithmetic, not only the requested switch (STRIX_GLM_PF_WMMA=1 still leaves the dense calls
// below the X3 threshold and the MoE calls below V11 on the exact cores).
#include "strix/env_flags.hpp"

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
// x16 tile form of the WMMA groups of x (KDA f_a | g_a | b and the indexer group): same bits as the direct form.
// STRIX_GLM_PF_X16=0 restores the direct form (A/B and rollback).
inline bool x16_enabled() { return strix::envcfg::flag("STRIX_GLM_PF_X16", true, strix::EnvFlagMatch::exact); }  // recorded
inline std::atomic<std::uint64_t> x16_calls{0};  // of those, the x16 tile form (core f16_dense_tall X16Cols; same bits)
}  // namespace strix::glm::pf::kda_wmma

#include "strix/env_arms.hpp"
#include "strix/env_flags.hpp"

namespace strix::glm::pf::mla_wmma {
// FAST identity glm-pf-mla-wmma-v1 (glm_pf_mla_wmma.hpp): the MLA absorb / out GEMMs on the shared short-K FP16 WMMA
// tile GEMM (core/f16_tile_gemm.hpp). FAST gate PASS (vs FAST without it: KL 0.0053, top-1 98.2 %, dNLL -0.0002, 1,536
// rows prose/code/chat): on by default when the FAST prefill route is selected (STRIX_GLM_PF_WMMA=1);
// STRIX_GLM_PF_MLA_WMMA=0 opts out (exact FP32 GEMM), =1 forces it. The exact route (FAST not selected, switch unset)
// never takes it. Part of the TP2 startup agreement (rank_defaults::kPfMlaWmmaConfig).
inline bool enabled() {
    static env_arms::Resolved<bool> value;  // once per process (or per in-process arm, env_arms.hpp)
    return value.get([] { return envcfg::flag("STRIX_GLM_PF_MLA_WMMA", wmma_x::enabled()); });
}
// With it: the fused flash core stores o-bar as the FP16 image the out GEMM's staging would make (half the o-bar bytes
// written and read). Default OFF (opt-in STRIX_GLM_PF_MLA_OB16=1): measured NOT bitwise at C 4096 (layer bench: 3 % of
// the out elements differ, max |d| 4.6e-5, spread over every row; the C 129 unit test is equal) and no e2e gain (pair 8K
// FAST C vs OB16=0 8.438 vs 8.438 s warm, one ABAB bundle); its bits are not gated (OB16=0 = the gated 2124b550).
inline bool ob16() {
    static env_arms::Resolved<bool> value;  // once per process (or per in-process arm, env_arms.hpp)
    return value.get([] { return envcfg::flag("STRIX_GLM_PF_MLA_OB16", false); });
}
inline std::atomic<std::uint64_t> calls{0};  // launches that ran on the WMMA core (identity reporting)
}  // namespace strix::glm::pf::mla_wmma

namespace strix::glm::pf::mla_gemm {
// Exact MLA absorb / out GEMM placement (glm_prefill.hip mla_gemm_rows_v3_kernel): heads-mid CTA order, the same bits.
// Default on; STRIX_GLM_PF_MLA_GEMM_HEADS=0 restores the rows-then-head order.
inline bool heads_mid() {
    static env_arms::Resolved<bool> value;  // once per process (or per in-process arm, env_arms.hpp)
    return value.get([] { return envcfg::flag("STRIX_GLM_PF_MLA_GEMM_HEADS", true); });
}

// Explicit-version gates default to exact arithmetic, independent of the environment.
enum class Arithmetic { exact, wmma };
struct Dispatch {
    Arithmetic arithmetic = Arithmetic::exact;
    bool ob16 = true;
    bool heads_mid = true;
    constexpr bool use_wmma(bool fits) const { return arithmetic == Arithmetic::wmma && fits; }
};
// First use must follow application defaults. All readers retain these process-lifetime values;
// changing an environment control requires a restart or an in-process arm (strix/env_arms.hpp generation).
inline Dispatch automatic_dispatch() {
    return {mla_wmma::enabled() ? Arithmetic::wmma : Arithmetic::exact, mla_wmma::ob16(), heads_mid()};
}
}  // namespace strix::glm::pf::mla_gemm
namespace strix::glm::pf::idxproj_wmma {
// FAST identity glm-pf-idxproj-wmma-v1: the DSA indexer projections of x (wk | gate | weights_proj, one group) on the
// shared dense FP16 WMMA core (x16 form from core::f16_dense_tall::kX16Rows rows) instead of the exact dense v3
// chains: changes the indexer scores, so the selected tokens may differ on near ties. kWithFast = false: opt-in
// (STRIX_GLM_PF_IDXPROJ_WMMA=1) until its gate passes; true: on by default with the FAST route (STRIX_GLM_PF_WMMA=1).
// STRIX_GLM_PF_IDXPROJ_WMMA=0 opts out, =1 forces it. Part of the TP2 startup agreement (rank_defaults::kPfIdxProjWmmaConfig).
inline constexpr bool kWithFast = false;
inline bool enabled() {
    const char* v = strix::envcfg::text("STRIX_GLM_PF_IDXPROJ_WMMA");  // recorded for the effective config / receipt
    if (v == nullptr || *v == '\0') return kWithFast && wmma_x::enabled();
    if (std::strcmp(v, "0") == 0 || std::strcmp(v, "1") == 0) return v[0] == '1';
    throw std::invalid_argument("STRIX_GLM_PF_IDXPROJ_WMMA must be 0 or 1");
}
inline std::atomic<std::uint64_t> calls{0};  // indexer groups that ran on the WMMA core (identity reporting)
}  // namespace strix::glm::pf::idxproj_wmma
namespace strix::glm::pf::router_wmma {
// FAST identity glm-pf-router-wmma-v1 (glm_pf_router_wmma.hpp): the prefill router logits on the shared dense FP16
// WMMA core. Opt-in (STRIX_GLM_PF_ROUTER_WMMA=1), with or without the FAST route, until its quality gate passes;
// unset or 0 = the exact R1 chains. Part of the TP2 startup agreement (rank_defaults::kPfRouterWmmaConfig).
inline bool enabled() {
    static const bool on = strix::envcfg::flag("STRIX_GLM_PF_ROUTER_WMMA", false);
    return on;
}
inline std::atomic<std::uint64_t> calls{0};  // calls that ran on the WMMA core (identity reporting)
inline std::atomic<const char*> form_used{"none"};  // launch form of the last call (core::f16_dense_tall::matrix_form_name)
}  // namespace strix::glm::pf::router_wmma
