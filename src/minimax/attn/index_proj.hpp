#pragma once
// minimax-m3 index projections (index q of the rank's 2 heads and the shared index key) in ONE launch, exact: every
// output is the G4 gate's dense_t chain, acc = 0; acc = fmaf(x[t][i], w[o][i], acc) for i ascending over the 6144
// inputs, with the F16 weights widened exactly. Bit-identical to dense_t (FP32 transposed weights) and to the engine's
// dense_t16 (src/minimax/engine/mm_kernels.hpp), only faster:
//   decode (rows <= 8): one thread per output keeps every row's chain; its weights stream from a packed chunk-major
//     layout through a register ring (loads issued kRing chunks ahead), the activations are wave-uniform operands;
//   prefill (any rows): blocks of 32 outputs x 64 tokens, the weight stage widened to F32 once in LDS for 8 waves,
//     each lane 8 token chains of one output, the wave's activations scalar operands.
#include <cstddef>
#include <cstdint>
#include <vector>

#include <hip/hip_runtime.h>

namespace strix::minimax::attn {

inline constexpr unsigned kProjIn = 6144;              // hidden size
inline constexpr unsigned kProjQ = 256, kProjK = 128;  // per rank: 2 index heads x 128, the shared index key
inline constexpr unsigned kProjOut = kProjQ + kProjK;

// Packed weights, chunk-major: chunk c (inputs 8c..8c+7) of output o at element (c * kProjOut + o) * 8. Outputs
// 0..255 = index_q_proj rows rank*256 .. rank*256+255, 256..383 = index_k_proj rows 0..127. The values must be the
// file's F16 values (exact). Host helper from the two row-major [out][6144] float slices.
std::vector<_Float16> pack_index_proj(const float* wq_rows, const float* wk_rows);

struct IndexProjView {
    const float* x;           // rows x 6144 (the attention input norm output)
    const _Float16* w;        // pack_index_proj layout, device
    float* iq;                // rows x 256
    float* ik;                // rows x 128
    std::uint32_t rows;
};
// Picks the decode kernel for rows <= 8 (ring 32 for one row, ring 16 + row pairs per wave above: measured) and the
// slab kernel above (all exact). STRIX_MMATTN_PROJ_RING / STRIX_MMATTN_PROJ_SPLIT=0 override for benches.
void enqueue_index_proj(const IndexProjView& v, hipStream_t stream);
// rows <= 8; rows_per_wave 0 = every row in each wave, 2 = row pairs over grid.y (more waves, weights re-read from L2)
void enqueue_index_proj_decode(const IndexProjView& v, unsigned ring, hipStream_t stream, unsigned rows_per_wave = 0);
void enqueue_index_proj_tiled(const IndexProjView& v, hipStream_t stream);                  // rows >= 8
// The tiled block with wave-uniform scalar activation loads and an F32 weight slab staged once per stage (double
// buffered LDS); rows >= 8. Exact, same bits as the tiled kernel, 0.67-0.70x its time: the prefill default.
void enqueue_index_proj_slab(const IndexProjView& v, hipStream_t stream);
// rows 2..8: 4-wave blocks sharing each weight chunk through an LDS ring. Exact but slower (measurement, cause in
// index_proj.hip); not used by enqueue_index_proj unless STRIX_MMATTN_PROJ_LDS=1.
void enqueue_index_proj_decode_lds(const IndexProjView& v, hipStream_t stream);
// FAST identity minimax-index-proj-wmma-v1 (not the exact chains): the same iq / ik on the shared dense FP16 WMMA
// core (core/f16_dense_tall.hpp): x rounded to FP16 (RNE), the packed FP16 weights as stored, per (16-row,
// 16-output) tile one WMMA chain from +0 over the 16-wide k units ascending. The weights are staged straight from
// the packed chunk-major layout above (no second copy). tc = column tiles per CTA (4, 2 or 1; 24 tiles in all):
// every tc gives the same bits. Any rows >= 1; rows * 6144 * 4 < 2^32. CTA order: column blocks fastest (the
// blocks of one row block share x in L2); rows_fast = true is the measured-slower order, kept for the bench.
void enqueue_index_proj_wmma(const IndexProjView& v, hipStream_t stream, unsigned tc = 4, bool rows_fast = false);
// The same bits with x first rounded to FP16 into k-tile-major tiles (xt: rows x 6144 FP16, device; the CTAs then
// read 16 KB contiguous per stage instead of 128 rows 24 KB apart). convert = false reuses xt as already written
// for the same x and row count, ordered on stream (bench: the cost of consuming producer-written tiles).
void enqueue_index_proj_wmma_x16(const IndexProjView& v, _Float16* xt, hipStream_t stream, bool convert = true);
// The FAST entry (minimax-index-proj-wmma-v1, both forms the same bits): the x16 tiles form from kFastTilesRows rows,
// the direct form below (historical ABAB x3, fixed first 12/30 of 60 weight sets, 11-15, gfxclk ~2.4 GHz, us/call:
//   rows   exact slab   direct   x16 (convert + wmma)   x16-only (withdrawn)
//   1024        686        203        274                   withdrawn
//   2048       1659        953        634                   withdrawn
//   4096       2758       1716       1234                   withdrawn
//   8192       5072       3058       2449                   withdrawn
// ; x16-only at 1024/2048/4096 used a stale stride; all x16-only figures await a gated rerun. The 1024-row direct figure keeps x (25 MB) in the 32 MB MALL across the bench's calls). xt: rows x 6144 FP16 when
// index_proj_fast_tiles(rows), else unused (may be null).
inline constexpr unsigned kFastTilesRows = 2048;
bool index_proj_fast_tiles(unsigned rows);
void enqueue_index_proj_fast(const IndexProjView& v, _Float16* xt, hipStream_t stream);
// Diagnostics of the tc 4 kernel (bench only): 1 = x rows taken modulo 128 (x stays in cache; wrong outputs),
// 2 = the same kernel capped at 192 VGPRs (2 CTAs per CU; same bits as enqueue_index_proj_wmma), 3 = only the first
// column block (outputs 0..63: x read exactly once, 1/6 of the products).
void enqueue_index_proj_wmma_diag(const IndexProjView& v, int mode, hipStream_t stream);
// Diagnostics (bench only, wrong outputs for modes 1-3): rows 1 (ring 32) or > 1 (row pairs, ring 16); mode 0 = the
// decode kernel, 1 = no weight loads, 2 = pure FMA chain, 3 = no FMAs (weight + activation streams).
void enqueue_index_proj_diag(const IndexProjView& v, int mode, hipStream_t stream);
// Concurrent form (exact: same kernels, same bits; measurement never slower): call index_proj_fork once hn is ready on `main`,
// enqueue the q/k/v linears on `main`, then index_proj_join before iq/ik are read. `side` is a non-blocking stream.
// The events are the caller's and must not be re-recorded while a previous record of them may still be pending (the
// installed HIP 10.1 runtime leaves that undefined): e.g. one pair per layer, reused only after the per-step sync
// when the caller synchronizes once per step, or a pool reset after a full synchronization.
// fast_xt != nullptr: the side stream runs enqueue_index_proj_fast (FAST identity) with that x16 workspace.
void index_proj_fork(const IndexProjView& v, hipStream_t main, hipStream_t side, hipEvent_t forked, _Float16* fast_xt = nullptr);
void index_proj_join(hipStream_t main, hipStream_t side, hipEvent_t joined);

}  // namespace strix::minimax::attn
