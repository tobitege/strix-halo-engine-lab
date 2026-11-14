#pragma once
// GLM-5.3-Flash chunked prefill, Tier A part (design notes section 2.1). One sequence slot; a chunk
// is C consecutive prompt rows at positions t0 .. t0 + C - 1. Every output and every state/cache byte equals the
// decode kernels (glm_tier_a.hip / glm_dsa.hip) applied row by row in position order, bit for bit: only the work
// placement differs (row-parallel kernels; the KDA recurrence stays token-sequential inside a register scan).
// Tier B (EXL3 linears) is not here; under identity glm-pf-wmma-v1 it is strix_gemm (deviation D10).
#include "glm/kernels/glm_dsa.hpp"
#include "glm/prefill/glm_pf_workspace.hpp"
#include "glm/kernels/glm_tier_a.hpp"
#include <functional>

namespace strix::glm::pf {

// ---- KDA. Shapes as dev::kda_step but for C rows of ONE slot:
//   qkv [C][3 * H * 128], f / gate [C][H * 128], b [C][H]; S [H][128][128] and conv_state [3 * H * 128][3] of the slot.
//   scratch [C][H][4][128] (q, k, v, g) + beta [C][H]; o [C][H][128]; y [C][H * 128].
// Updates S and conv_state to their values after row C - 1. C2 -> flag.
void kda_prefill(const dev::KdaDev& k, const float* qkv, const float* f, const float* gate, const float* b, float* S,
                 float* conv_state, float* scratch, float* beta, float* o, float* y, unsigned* flag, unsigned C,
                 hipStream_t s);
// Profiling hook (bench only): when set, kda_prefill records 5 events (before prep, conv commit, scan, out, after).
void set_kda_profile(hipEvent_t* ev5);
// Explicit KDA scan version for bitwise gates and paired timing (same bits): 1 = v1 (shuffle tree), 3 = v3 (quad-DPP
// tree), 5 = v5 (v3 with the batch fill off the serial path); 0 = the default (v5; STRIX_GLM_PF_KDA_SCAN=3 / 1).
void set_kda_scan_version(unsigned v);

// ---- DSA indexer + MLA. The slot holds positions [0, t0) on entry.
// Cache writes for rows t0 .. t0 + C - 1 (latent [C][512] normed, key / gate [C][128] = the R1 wk / gate outputs of
// the rows) and every pool completed by them (R12). C3/C4 -> flag.
// With latent_norm (BF16 [512]), latent is raw kv_a: the append consumes the
// original 256-lane RMSNorm graph before conversion, without a FP32 norm image.
void dsa_append_rows(const dev::DsaSlot& slot, unsigned t0, const float* latent, const float* key, const float* gate,
                     const dev::IndexerDev& w, unsigned* flag, unsigned C, hipStream_t s,
                     bool kv_shard = false, unsigned rank = 0, const std::uint16_t* latent_norm = nullptr);

// Canonical token lists of rows t0 .. t0 + C - 1 (section 1.4): tokens [C][2051], n [C].
//   qI [C][32 * 128] (wq_b outputs), wh [C][32] = R1 weights_proj outputs, unscaled on entry. Only when some row of the
//   chunk has P_t > 512 (scores needed) are they scaled in place by RN32(32^-1/2); otherwise wh is not read or written.
//   keys: workspace [C][kpool_cap] uint32 with kpool_cap >= (t0 + C) / 4.
void indexer_select_rows(const dev::DsaSlot& slot, unsigned t0, const float* qI, float* wh, std::uint32_t* keys,
                         unsigned kpool_cap, std::uint32_t* tokens, std::uint32_t* n, unsigned C, hipStream_t s);
// Capacity-independent workspace: keys [C][kIndexerKeyStride]. Same canonical
// list, with stable top-512 carried across fixed pool windows; no submit allocation.
void indexer_select_rows_bounded(const dev::DsaSlot& slot, unsigned t0, const float* qI, float* wh, std::uint32_t* keys,
                                 std::uint32_t* tokens, std::uint32_t* n, unsigned C, hipStream_t s, unsigned score_v = 0,
                                 const std::function<const float*(unsigned, unsigned)>& pool_window = {});
// Explicit versions for bitwise gates and paired timing: scores 1/2/3 (same keys), select 1/2 (same lists). The default
// entry point runs scores 3 + select 2; STRIX_GLM_PF_IDX=2 / STRIX_GLM_PF_SELECT=1 restore the previous ones.
void indexer_select_rows_versions(const dev::DsaSlot& slot, unsigned t0, const float* qI, float* wh, std::uint32_t* keys,
                                  unsigned kpool_cap, std::uint32_t* tokens, std::uint32_t* n, unsigned C, hipStream_t s,
                                  unsigned score_v, unsigned select_v);
// Profiling hook (bench only): when set, the indexer records 3 events (before scores, before select, after). nullptr disables.
void set_idx_profile(hipEvent_t* ev3);

// R13 for `heads` rank heads and C rows: q [C][heads][256]; kv_b rank rows [heads * 512][512] F16.
//   qt [Q][heads][512], e [Q][heads][kEStride] and ob [Q][heads][512] are workspace; o [C][heads][256].
//   The explicit query_window overload uses Q=min(C, query_window); zero/default keeps Q=C.
//   The owner resolves the environment at install; these launchers never read it.
//   Workspace contains only the final query window after return. e holds the
//   probabilities in token order, or lane-major (slot 9 * bitrev8(t % 256) + t / 256, padding +0) when
//   mla_e_lane_major(heads) (o-bar v5 / v7).
bool mla_e_lane_major(unsigned heads);
// Profiling hook (bench only): when set, mla_attention_rows records 6 events on its stream (before absorb, scores, softmax,
// o-bar, out, after). With query windows the events describe the final window. nullptr disables.
void set_mla_profile(hipEvent_t* ev6);
void mla_attention_rows(const dev::DsaSlot& slot, const std::uint32_t* tokens, const std::uint32_t* n, const float* q,
                        const std::uint16_t* kv_b, unsigned heads, float* qt, float* e, float* ob, float* o, unsigned C,
                        hipStream_t s);
void mla_attention_rows(const dev::DsaSlot& slot, const std::uint32_t* tokens, const std::uint32_t* n, const float* q,
                        const std::uint16_t* kv_b, unsigned heads, float* qt, float* e, float* ob, float* o, unsigned C,
                        hipStream_t s, unsigned query_window);

// Scores v7 (heads 32; heads 64 runs v5) and o-bar v10 are the defaults (STRIX_GLM_PF_SCORES=6 / STRIX_GLM_PF_OBAR=7
// restore the previous ones); STRIX_GLM_PF_OBAR=6 opts into o-bar v6 (exact, slower: not a default).
// Explicit selectors for bitwise gates and paired stage timing, independent of cached environment values.
// scores: 2/3/4/5/6/7; o-bar: 2/4/5/6/7/10 (5, 7 and 10 fall back to 2 for heads=64); legacy pair: 1/1.
// o-bar 5, 7 or 10 with heads=32 uses lane-major e; all other pairs use token-order e. gemm (absorb / out): 1 / 2 / 3
// (same per-output chain; the legacy pair has its own). The default entry point runs 3; STRIX_GLM_PF_MLA_GEMM=2
// restores v2, =1 the legacy tiled GEMM. V3 is bitwise equal to v2 on the pair at 8K and 32K.
// Explicit query_window is independent of the environment for bitwise/guard gates;
// zero retains C-row scratch and the full intermediate images used by existing gates.
void mla_attention_rows_versions(const dev::DsaSlot& slot, const std::uint32_t* tokens, const std::uint32_t* n, const float* q,
                                 const std::uint16_t* kv_b, unsigned heads, float* qt, float* e, float* ob, float* o, unsigned C,
                                 hipStream_t s, unsigned scores_v, unsigned obar_v, unsigned gemm_v = 1);
void mla_attention_rows_versions(const dev::DsaSlot& slot, const std::uint32_t* tokens, const std::uint32_t* n, const float* q,
                                 const std::uint16_t* kv_b, unsigned heads, float* qt, float* e, float* ob, float* o, unsigned C,
                                 hipStream_t s, unsigned scores_v, unsigned obar_v, unsigned gemm_v, unsigned query_window);

// ---- R1 plain-weight chains for many rows (router, KDA f_a/f_b/g_a/g_b/b_proj, indexer wk/gate/weights_proj):
// y[r][o] = chain over i ascending of fmaf(x[r][i], W[o0 + o][i], acc) from +0 = dev::dense bit for bit, as a tiled
// VALU GEMM (each weight tile read once per 64 rows instead of once per row). `in` must be a multiple of 32.
void dense_rows(const float* x, const void* w, dev::WType wt, unsigned in, unsigned o0, unsigned n, float* y, unsigned rows,
                hipStream_t s);
// Read-once STRIX_GLM_PF_DENSE_ROWS=4 (default) | 3 | 2 | 1; explicit selector bypasses the environment for bitwise gates/timing.
// v4 = the shared FMA-rows core (src/core/fma_rows.hpp) for chains of at most 4096 inputs, v3 route otherwise.
// v3 = v2's chains in a 128 x 64 tile for grouped projections (dense_rows_group); single calls keep the v2 tile.
// Same bits in every version.
void dense_rows_versioned(const float* x, const void* w, dev::WType wt, unsigned in, unsigned o0, unsigned n, float* y,
                          unsigned rows, hipStream_t s, unsigned version);
// Up to kDenseGroupMax projections of the same x (same in, rows and weight type) in one launch under v3; v1 / v2 run them
// one by one. Each output equals dense_rows of that projection bit for bit.
inline constexpr unsigned kDenseGroupMax = 3;
struct DenseProj {
    const void* w;
    unsigned o0, n;
    float* y;
};
void dense_rows_group(const float* x, dev::WType wt, unsigned in, const DenseProj* p, unsigned count, unsigned rows, hipStream_t s);
// Explicit version and v3 tile (rt 32 / 64 / 128 rows with c 32 / 64 columns; 0 / 0 = automatic) for gates and timing.
void dense_rows_group_versioned(const float* x, dev::WType wt, unsigned in, const DenseProj* p, unsigned count, unsigned rows,
                                hipStream_t s, unsigned version, unsigned rt = 0, unsigned c = 0);

// ---- mHC pre for many rows: = dev::hc_pre bit for bit (same outputs F, mix, x_collapsed), with the 24 R1 chains over
// the 16384 normalized values as a VALU GEMM. z: workspace [rows][24]. F remains materialized on both paths.
// STRIX_GLM_PF_HC is read once: 2 opts into v2; 1 / unset / other values select v1 pending native qualification.
// v2 chain shape (same bits): STRIX_GLM_PF_HC_TILE = 64 | 32 | 16 rows per CTA, STRIX_GLM_PF_HC_COLS = 24 | 8 outputs per CTA.
void hc_pre_rows(const float* X, const float* fn, const float* base, const float* scale, float* F, float* z, float* mix, float* xc,
                 unsigned rows, hipStream_t s);

// STRIX_GLM_PF_HC_FUSE (default 1, v2 sites only; =0 restores the unfused sites): hc_expand also computes the next site's norm factor rr
// per row (the hc_norm_rows expressions and order); that site runs hc_pre_rows_normed (chains on X * rr, + mix) and F
// is never materialised. Same bits. STRIX_GLM_PF_HC_FUSE=2 (split form, a profiling control): the unfused hc_expand, then
// hc_norm_factor_rows computes rr alone from the expanded X; the next site is the same hc_pre_rows_normed.
bool hc_expand_norm_enabled();
bool hc_expand_norm_split();
void hc_norm_factor_rows(const float* X, float* rr, unsigned rows, hipStream_t s);
void hc_expand_norm_rows(const float* X, const float* mix, const float* y, float* Xo, float* rr, unsigned* flag, unsigned rows,
                         hipStream_t s);
// Two-row expand CTA: strict cached STRIX_GLM_PF_HC_EXPAND_ROWS=1 (kernel default) / 2 (rank default).
// Default keeps the existing kernel. Explicit gate API bypasses the selector: each row
// uses 256 threads and its own ordered tree. y1 != nullptr forms rank-ordered y + y1
// before multiplication; hoist 0/1/2 are the existing load-placement controls.
// Hoist 2 requires disjoint X/Xo. No resource, payload or performance qualification implied.
unsigned hc_expand_cta_rows();
void hc_expand_norm_rows_cta(unsigned cta_rows, unsigned hoist, const float* X, const float* mix,
                             const float* y, const float* y1, float* Xo, float* rr, unsigned* flag,
                             unsigned rows, hipStream_t s);
// Gate/bench only: prints all variants' attributes and occupancy; false on spills or
// an unlaunchable block. Does device work, unlike reading hc_expand_cta_rows().
bool hc_expand_cta_resources();
void hc_pre_rows_normed(const float* X, const float* fn, const float* base, const float* scale, const float* rr, float* z, float* mix,
                        float* xc, unsigned rows, hipStream_t s);
// STRIX_GLM_PF_SUMEXP (default 1; =0 restores the exchange's sum2 then hc_expand_norm_rows): the pipelined TP2 seam hands
// both partials to the fused expand, which forms y = y0 + y1 (rank order, the sum2 expression) on load. Same bits.
// Enabled only with the fused (non-split) STRIX_GLM_PF_HC_FUSE form.
bool hc_sum_expand_enabled();
void hc_sum_expand_norm_rows(const float* X, const float* mix, const float* y0, const float* y1, float* Xo, float* rr, unsigned* flag,
                             unsigned rows, hipStream_t s);
// STRIX_GLM_PF_HC_HOIST (default 1; 0 = loads inside each step, 2 = 1 plus restrict): where the fused expand issues its loads (1: y / y0 + y1 into registers before the first
// step; 2: also restrict row pointers so later X loads issue early). Same expressions in the same order: same bits.
unsigned hc_expand_hoist();
void hc_expand_norm_rows_h(unsigned hoist, const float* X, const float* mix, const float* y, float* Xo, float* rr, unsigned* flag,
                           unsigned rows, hipStream_t s);
void hc_sum_expand_norm_rows_h(unsigned hoist, const float* X, const float* mix, const float* y0, const float* y1, float* Xo, float* rr,
                               unsigned* flag, unsigned rows, hipStream_t s);

// Explicit versions for bitwise gates and paired timing, independent of the cached environment selector.
void hc_pre_rows_v1(const float* X, const float* fn, const float* base, const float* scale, float* F, float* z, float* mix, float* xc,
                    unsigned rows, hipStream_t s);
void hc_pre_rows_v2(const float* X, const float* fn, const float* base, const float* scale, float* F, float* z, float* mix, float* xc,
                    unsigned rows, hipStream_t s);

// STRIX_GLM_PF_MIXNORM (cached, default 1; only 0/1 accepted): fuse the v2 collapse + BF16 RMS tail.
// The overloads write mix and xn, never xc; the original entry points remain explicit unfused controls.
bool hc_mix_norm_enabled();
void hc_pre_rows_v2(const float* X, const float* fn, const float* base, const float* scale, float* F, float* z, float* mix,
                    const std::uint16_t* norm, float* xn, unsigned rows, hipStream_t s);
void hc_pre_rows_normed(const float* X, const float* fn, const float* base, const float* scale, const float* rr, float* z, float* mix,
                        const std::uint16_t* norm, float* xn, unsigned rows, hipStream_t s);
// Explicit tail hooks for bitwise gates with independently supplied z (no chain or environment dispatch).
void hc_mix_rows_v2(const float* X, const float* z, const float* base, const float* scale, float* mix, float* xc,
                    unsigned rows, hipStream_t s);
void hc_mix_norm_rows(const float* X, const float* z, const float* base, const float* scale, float* mix,
                      const std::uint16_t* norm, float* xn, unsigned rows, hipStream_t s);

inline constexpr unsigned kMaxTokens = kIdxTopk + kKpool - 1;  // 2051
// Rounds of 256 lanes covering a list (9 x 256 >= 2051) and the stride of the per-(row, head) score / probability
// buffer e: kObRounds * 256 = 2304 >= kMaxTokens, so o-bar v5's lane-major p layout fits in place.
inline constexpr unsigned kObRounds = (kMaxTokens + 255) / 256;  // 9
inline constexpr unsigned kEStride = kObRounds * 256;            // 2304

}  // namespace strix::glm::pf
