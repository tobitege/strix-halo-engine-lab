#pragma once
// MiniMax M3.1 Flash prefill on the RTX PRO: the non-linear kernels, each in the declared order
// of the engine's portable host references, so the results equal the nodes' bit for bit:
//   embed / input norm      engine mm_kernels.hpp embed_kernel / rmsnorm_gemma_kernel
//   post-exchange + router  strix/minimax_moe.hpp post_exchange_row / route_row
//   index projections       engine dense_t(16)_kernel (per output: fmaf chain over the 6144 inputs, ascending)
//   MSA                     strix/minimax_msa.hpp head_norm_rope / index_block_scores / select_blocks /
//                           sparse_attention (K/V/index-key caches BF16, RNE)
// Build with --fmad=false -prec-div=true -prec-sqrt=true (IEEE division and sqrt, no contraction).
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::minimax::rtxpf::cuda {

inline constexpr unsigned kHidden = 6144, kExperts = 128, kTopE = 4, kHeadDim = 128, kQHeads = 64, kKvHeads = 4;
inline constexpr unsigned kIdxHeads = 4, kBlock = 128, kTopK = 16, kLayers = 60;

struct RopeRow {
    float c[32], s[32];
};

// One prompt sequence: rows [row0, row0 + rows) have positions [pos0, pos0 + rows). Default pos0=0 preserves full prefill.
// Caches are this sequence's, every layer: k/v [layers][cap][4][128] BF16, ik [layers][cap][128] BF16.
struct SeqView {
    std::uint32_t row0, rows, cap;
    std::uint16_t *k, *v, *ik;
    std::uint32_t pos0 = 0;  // absolute position of row0; zero preserves whole-prompt calls
};

void embed(const std::uint16_t* table, const std::uint32_t* tokens, std::size_t rows, float* out, cudaStream_t s);
void rmsnorm(const float* x, const float* w, float* y, std::size_t rows, cudaStream_t s);
// h += p0 + p1; y = Gemma norm of h with w.
void post_exchange(float* h, const float* p0, const float* p1, const float* w, float* y, std::size_t rows, cudaStream_t s);
// Router of route_row: ids / weights [rows][4] (choice order), scratch [rows][128][2].
void route(const float* y, const float* router_w, const float* bias, std::size_t rows, float* scratch, std::uint32_t* ids,
           float* weights, cudaStream_t s);
// y[t][o] = chain over i ascending of fmaf(x[t][i], wt[i][o], acc); wt FP32 [6144][n] (transposed); y row stride ys.
void dense_t(const float* x, const float* wt, float* y, std::size_t rows, unsigned n, unsigned ys, cudaStream_t s);

// MSA of one layer for all rows of all sequences (rows of seq j are its positions 0..rows-1):
//  prep    q (64 heads) and iq (4 heads) normed + roped in place (FP32); k (4 heads) normed + roped, v, ik normed +
//          roped stored into the sequence caches (BF16) of layer L.
//  attend  per (row, KV head g): index scores of index head g over the visible blocks, top-16 selection (all blocks when
//          at most 16), sparse attention of query heads 16 g .. 16 g + 15 -> o [rows][64 * 128].
struct MsaLayer {
    unsigned layer;
    const float *q_w, *k_w, *iq_w, *ik_w;  // 128 Gemma weights each
};
void msa_prep(const SeqView* seqs_d, const std::uint32_t* row_seq, std::size_t rows, const MsaLayer& L, float* q, const float* k,
              const float* v, float* iq, const float* ik, const RopeRow* rope, float* scratch_k, cudaStream_t s);
// tiles: (first row, rows <= 32) of ONE sequence each, covering every row (the tiled index-score kernel); legacy = the
// first per-warp kernels (same bits; MM_RTX_ATTN_LEGACY=1).
void msa_attend(const SeqView* seqs_d, const std::uint32_t* row_seq, std::size_t rows, unsigned max_blocks, unsigned layer,
                const float* q, const float* iq, float* scores, std::uint32_t* sel, std::uint32_t* sel_count, float* o,
                cudaStream_t s, const uint2* tiles, unsigned n_tiles, bool legacy);

// This rank's state rows of one layer from a sequence's caches into a host-layout buffer (Engine::export_rows: per layer
// K [n][256], V [n][256], index key [n][128], rank r = KV heads 2r, 2r + 1). dst points at the layer's block.
extern double g_attn_prof[3];  // MM_RTX_PROFILE: msa_attend sub-phases (scores, select, attend), seconds
void export_layer(const SeqView& seq, unsigned layer, unsigned rank, std::uint64_t n, std::uint8_t* dst, cudaStream_t s);
void export_layer(const SeqView& seq, unsigned layer, unsigned rank, std::uint64_t r0, std::uint64_t n, std::uint8_t* dst, cudaStream_t s);

}  // namespace strix::minimax::rtxpf::cuda
