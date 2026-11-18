#pragma once
// Product-profile rows of the MiniMax M3.1 pair apps (strix-minimax-rank / strix-minimax-serve): the levers that are
// OFF by default (the defaults-ON items stay in mm_defaults.cpp kItems) and that both ranks must agree on. Resolved by
// the one MiniMax resolver (mm_defaults.cpp prefill_policy / profile_resolution): engine, session-cache identity,
// effective config and rank agreement read the same values. A row at its builtin value is left out of the profile
// fingerprint (omit_builtin_fingerprint), so with every row OFF these rows add no fingerprint inputs.
// Complete cache compatibility also depends on the executable and raw environment identity.
// FAST still uses its existing bundle switch. Product TURBO proposals stay pending until
// the adapter admits and exports them through prefill_policy before engine construction.
#include "strix/engine_profile.hpp"
#include "strix/fast_profile.hpp"
#include "xport/exchange_format.hpp"

#include <array>
#include <string_view>

namespace strix::minimax::defaults {

// STRIX_MM_FAST_PREFILL=1 (default off) = profile mm-fast-pf-v2:
//   MSA attention  STRIX_MM_PF_MSA_FLASH=1  msa-flash-bf16-l40-v1 (one-term BF16 operands on the shared key-split
//                  WMMA flash core, layers STRIX_MM_PF_MSA_FLASH_LAYERS = 40-59), on the exact EXL3 MoE base only:
//                  the bundle does not select it with STRIX_MMX_MODEL_FORMAT=fast_i4 (gate fails there, KL 0.079).
//   MoE router     STRIX_MM_PF_ROUTER_WMMA=1  mm-pf-router-wmma-v1 (router product of the MoE layers in
//                  STRIX_MM_PF_ROUTER_WMMA_LAYERS = 40-59 on the shared FP16 WMMA core; sigmoid, bias, selection
//                  unchanged), same EXL3-only admission. v2 = v1 + this member (gate 8K + 32K, pfe2o).
// A set member variable wins over the bundle ("0" = off beats it). Changes logits.
inline constexpr strix::fast_profile::Member kFastMembers[] = {
    {"STRIX_MM_PF_MSA_FLASH", "1", "msa-flash-bf16-l40-v1", "1 2 3"},
    {"STRIX_MM_PF_ROUTER_WMMA", "1", "mm-pf-router-wmma-v1"},
};
inline constexpr strix::fast_profile::Bundle kFastBundle{"STRIX_MM_FAST_PREFILL", "mm-fast-pf-v2", kFastMembers};
inline constexpr const char* kMsaFlashLayers = "40-59";  // passes the FAST gate on EXL3 at 8K and 32K; 30-59 8K only
// FAST mm-pf-router-wmma-v1 (STRIX_MM_PF_ROUTER_WMMA=1, or the bundle mm-fast-pf-v2; "0" opts out): the prefill
// router product of the MoE layers in STRIX_MM_PF_ROUTER_WMMA_LAYERS on the shared FP16 WMMA core. The default set is
// the flash set: a change in a shallower layer alters every later selection (the onset rule of the MSA gate).
inline constexpr const char* kRouterWmmaLayers = "40-59";
inline constexpr unsigned kMsaLayers = 60;
// Index projections (index q of the rank's heads + the shared index key, attn/index_proj.hpp) on the shared dense
// FP16 WMMA core (core/f16_dense_tall.hpp; x16 tile form from kX16Rows rows) instead of the exact slab chains:
// STRIX_MM_PF_IDX_WMMA=1 on the layer set STRIX_MM_PF_IDX_WMMA_LAYERS (builtin kIdxWmmaLayers = the bundle's flash
// onset, so the onset-matched floor nulls of the flash gate apply). Prefill rows > 8 only (decode keeps its exact
// kernels). Changes the index scores, hence near-tie block selections. Not a bundle member until its gate passes.
inline constexpr const char* kIdxWmmaLayers = "40-59";
inline constexpr const char* kIdxWmmaPolicy = "minimax-index-proj-wmma-v1";

// Named one-term policies (canonical layer ranges -> policy id), unchanged FAST gate (KL <= 0.02, top-1 >= 93 %,
// dNLL <= 0.01) on EXL3, 256 rows per prompt:
//   l40-v1  8K PASS (3 prompts, KL 0.0012); 32K PASS (b32k KL 0.0029, c32k KL 0.0029, top-1 99.6 / 100 %): the bundle's.
//   l30-v1  8K PASS (KL 0.0076, top-1 98.2 %, dNLL -0.0096); 32K FAIL (b32k KL 0.042, dNLL +0.012; c32k dNLL +0.016):
//           explicit STRIX_MM_PF_MSA_FLASH_LAYERS=30-59 only, for prompts of at most 8K tokens. The layer set is
//           resolved once per process, so the bundle cannot switch it by prompt length.
// Any other mask / the two-term mode is reported by its ranges. Mode 3 (explicit only, not gated yet) = one-term
// operands on the shared core's row-group form (attn_core/sparse_flash_rowgroup.hpp: 4 neighbouring rows per
// work-group share each K/V tile; one softmax stream per row): policy msa-flash-bf16-rg4-l<ranges>.
struct MsaFlashPolicy {
    const char* layers;
    const char* policy;
};
inline constexpr MsaFlashPolicy kMsaFlashPolicies[] = {
    {"40-59", "msa-flash-bf16-l40-v1"},
    {"30-59", "msa-flash-bf16-l30-v1"},
};
inline constexpr const char* msa_flash_policy(std::string_view layers) {
    for (const auto& p : kMsaFlashPolicies)
        if (layers == p.layers) return p.policy;
    return nullptr;
}
static_assert(msa_flash_policy(kMsaFlashLayers) != nullptr &&
                  std::string_view(msa_flash_policy(kMsaFlashLayers)) == std::string_view(kFastMembers[0].policy),
              "the bundle member's policy id names the bundle's layer set");

// Bundle selection is requested provenance; only effective members enter agreement.
inline constexpr std::array<strix::profile::Lever, 9> profile_table{{
    // Prefill only: MSA attention of windows > 8 rows; selection, block lists and K/V caches unchanged.
    {"msa_flash", "STRIX_MM_PF_MSA_FLASH", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, false}, "", "0",
     "msa-flash-bf16-l40-v1 (EXL3 gate PASS 8K KL 0.0012, 32K KL 0.0029); 2 = two-term operands; pending turbo adapter admission/export for mode 1 on layers 0-59",
     false, {}, true},
    {"msa_flash_layers", "STRIX_MM_PF_MSA_FLASH_LAYERS", "", {"", "", "", "0-59"}, kMsaFlashLayers, strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, false}, "", "", "FAST MSA flash layer set (read only while msa_flash is on); pending turbo adapter admission/export for 0-59, then per-prompt qualification with scripts/bench/mmflash",
     false, {}, true},
    // PENDING turbo cells: the lever exists only on an open PR / has no per-layer selector yet (reason in `source`).
    {"moe_invariant", "STRIX_MM_MOE_INVARIANT", "", {"", "", "", "1"}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {false, false, false, false}, "", "0",
     "pending: rows-free MoE order (open PR #662, chunkinv/linear-contract); cell flips with that merge", false, {}, true},
    {"iu4_layers", "STRIX_MM_IU4_LAYERS", "", {"", "", "", ""}, "", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {false, false, false, false}, "", "",
     "pending: transient IU4 per layer (fast_i4 is all-or-nothing today and fails the gate, KL 0.079); no per-layer selector yet; "
     "never imposed where it slows", false, {}, true},
    // Prefill only: index projections of rows > 8 on the shared FP16 WMMA core; K/V caches and MSA attention unchanged.
    {"idx_wmma", "STRIX_MM_PF_IDX_WMMA", "", {"", "", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, false}, "", "0", "minimax-index-proj-wmma-v1 (opt-in; FAST gate e3wire)", false, {}, true},
    {"idx_wmma_layers", "STRIX_MM_PF_IDX_WMMA_LAYERS", "", {"", "", "", ""}, kIdxWmmaLayers, strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, false}, "", "", "FAST index projection layer set (read only while idx_wmma is on)", false, {}, true},
    // Explicit only: 32K quality gate failed; no profile selects f16.
    strix::xport::exchange_format_lever(),
    // Prefill only: router product of the listed MoE layers on the FP16 WMMA core; sigmoid, bias and selection unchanged.
    {"router_wmma", "STRIX_MM_PF_ROUTER_WMMA", "", {"", "", "", ""}, "0", strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, false}, "", "0", "mm-pf-router-wmma-v1 (FAST bundle mm-fast-pf-v2; gate 8K KL 0.0020, 32K pfe2o)", false, {}, true},
    {"router_wmma_layers", "STRIX_MM_PF_ROUTER_WMMA_LAYERS", "", {"", "", "", ""}, kRouterWmmaLayers, strix::profile::Kind::bit_changing,
     {true, false, true, false}, {true, true, true, false}, "", "", "FAST router layer set (read only while router_wmma is on)", false, {}, true},
}};

}  // namespace strix::minimax::defaults
