#pragma once
// GLM-5.3-Flash chunked prefill seam (engine::PrefillPath, design notes).
// One layer over a chunk of R rows of ONE slot at consecutive positions:
//   Tier A: the row-parallel prefill kernels of src/glm/prefill (gated bitwise == the decode kernels row by row);
//   Tier B (EXL3 linears), by identity:
//     gemv (default, exact): the decode identity, so the chunked prefill equals the reference prefill
//       (decode row by row) bit for bit. STRIX_GLM_PF_EXACT_LIN uses the tiled DOT2 graph for supported linears;
//       STRIX_GLM_PF_EXACT_MOE uses grouped exact DOT2 for TP2 MoE. Both default ON (=0 off).
//       With the exact linears the KDA q/k/v projections write [R][q | k | v] directly (strided output, same bits);
//       STRIX_GLM_PF_KDA_PACK=1 restores contiguous projections + pack_qkv.
//       STRIX_GLM_PF_EXACT_TM=16|32|64, default 32; STRIX_GLM_PF_EXACTMOE_V=1|2|3, default 3.
//     wmma (STRIX_GLM_PF_TIERB=wmma, opt-in, glm-pf-wmma-v1): LinearRunner::run_pinned with the frozen split
//       table (every GLM shape: s = 1), routed experts through the grouped core
//       (bitwise == the per-expert pinned chain); deviation D10.
// The engine runs the mHC sites, norms, router and exchanges of the chunk itself (row-batched, bitwise == decode).
#include <strix/glm/engine.hpp>

#include <memory>
#include "glm/prefill/glm_pf_workspace.hpp"

namespace strix::glm::pf {

// Device bytes the chunked prefill adds per rank (memory budget):
//   path: every buffer of the PrefillPath, all allocated at construction (no device allocation during a prefill);
//         includes align256(max_rows * largest_linear_input * 2) exact-linear FP16 scratch,
//         reserved regardless of identity/switch so admission always includes it; host
//         side: per-chunk route vectors O(rows x 8) and a one-time per-layer table of expert views.
//   path: every buffer of the PrefillPath, all allocated at construction (no device allocation during a prefill; host
//         side: reusable route vectors O(rows x 8), an exact grouping plan, and expert views);
//         with STRIX_GLM_PF_EXACT_MOE=1, TP2 adds exact_moe::workspace_bytes(max_rows)
//         (GU FP32 + down-input FP16 + output FP32 + 2 H-wide FP16 input transforms
//         for 9 pairs per token, plus tables; 792849408 / 1585660416 bytes at R=2048 / 4096);
//         STRIX_GLM_PF_SCRATCH=auto (default) retains the original full arena when
//         admitted; routed reserves it only for TierB::wmma at TP2; full pins it.
//         GEMV (including FAST WMMA-X) and world-1 per-expert fallback never use it.
//   workspace: WMMA only, the largest strix_gemm_pinned workspace of the path's linears; zero for GEMV. Engine reserves the
//         larger of it and every GEMV call's workspace (gemv-mode Tier B, head);
//         DSA keys use 4 * max_rows * (4096 + 512) bytes, independent of capacity.
//         Sharded layouts also reserve kKvShardPrefillTemporaryBytes for bounded redistribution.
//         Attention qt/e/ob use attention_scratch_rows(max_rows) * rank_heads * (512 + 2304 + 512) floats;
//         STRIX_GLM_PF_ATTN_QWIN=auto (default) tries full chunk, then 2048 only
//         after routed scratch fails admission; 0 pins full chunk, 1..65536 pins a window.
//   pipeline: SeqparExchange channel rings, coherent full-row consumer and
//         runtime reserve. Pipeline 2 partitions engine_rows into disjoint halves
//         and serializes the path's column scratch, so it adds no second path.
//         Native install_prefill admits this separately; deployment includes it.
//   engine_rows: the engine's row buffers sized max(8, Options::max_rows) (allocated/admitted by the load plan).
//         Includes two H-wide final hidden buffers and, when mtp, nine H-wide MTP scratch equivalents.
//         pf_mem_budget mirrors must include (2 + (mtp ? 9 : 0)) * RC * H * 4 bytes: 352 MiB at RC=2048 with MTP.
//         Decode-only logits/argmax and other fixed-size scratch are outside this row-scaled subtotal.
struct PrefillBudget {
    std::uint64_t path = 0, workspace = 0, engine_rows = 0, pipeline = 0;
    PrefillMoeRoute moe_route;
    unsigned attention_rows = 0;
};
std::uint64_t admission_margin_bytes();
PrefillBudget prefill_budget(unsigned kda_heads, unsigned mla_heads, std::uint64_t capacity, unsigned max_rows, bool mtp = true,
                            TierB tierb = TierB::gemv, bool kv_shard = false);

// heads: rank KDA / MLA heads; capacity: DSA positions per slot; max_rows: largest chunk.
// Fails closed (std::runtime_error, nothing allocated) unless path + workspace + a margin fit both the free device
// memory (hipMemGetInfo) and the host's MemAvailable (/proc/meminfo; unified memory on gfx1151). Margin:
// STRIX_GLM_PF_MARGIN_GIB (default 1 GiB; see kPrefillSafetyBytes), unless the
// caller supplies its startup policy margin (rank CLI keeps the OS reserve).
// first_forward_reserve adds the identical policy headroom used by startup;
// kv_shard reserves bounded redistribution scratch, never a full cache copy.
std::shared_ptr<engine::PrefillPath> make_prefill_path(unsigned kda_heads, unsigned mla_heads, std::uint64_t capacity,
                                                       unsigned max_rows, TierB tierb, std::uint64_t margin = admission_margin_bytes(),
                                                       bool kv_shard = false, std::uint64_t first_forward_reserve = 0);
// TierB from STRIX_GLM_PF_TIERB (gemv | wmma; default gemv, exact).
TierB tierb_from_env();
// gemv identity: rows per strix_gemv / MoE-seam call over a chunk, STRIX_GLM_PF_GEMV_BLOCK in 1..8 (default 8). A row's
// bits do not depend on it (P1 batch invariance R = 1..8); 1 = the reference's one row per call.
unsigned gemv_block();

}  // namespace strix::glm::pf
