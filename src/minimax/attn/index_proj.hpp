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
// Diagnostics (bench only, wrong outputs for modes 1-3): rows 1 (ring 32) or > 1 (row pairs, ring 16); mode 0 = the
// decode kernel, 1 = no weight loads, 2 = pure FMA chain, 3 = no FMAs (weight + activation streams).
void enqueue_index_proj_diag(const IndexProjView& v, int mode, hipStream_t stream);
// Concurrent form (exact: same kernels, same bits; measurement never slower): call index_proj_fork once hn is ready on `main`,
// enqueue the q/k/v linears on `main`, then index_proj_join before iq/ik are read. `side` is a non-blocking stream.
// The events are the caller's and must not be re-recorded while a previous record of them may still be pending (the
// installed HIP 10.1 runtime leaves that undefined): e.g. one pair per layer, reused only after the per-step sync
// when the caller synchronizes once per step, or a pool reset after a full synchronization.
void index_proj_fork(const IndexProjView& v, hipStream_t main, hipStream_t side, hipEvent_t forked);
void index_proj_join(hipStream_t main, hipStream_t side, hipEvent_t joined);

}  // namespace strix::minimax::attn
