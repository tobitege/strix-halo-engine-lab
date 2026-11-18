#pragma once

// deepseek41 CUDA backend, layer core: the sites outside attention and the routed experts (ds41_layer.hpp,
// ds41_moe.hpp, ds41_formats.hpp), each reproducing its host reference bit for bit (docs/models/deepseek-v4.1/cuda-numeric-identity.md).
// Conventions: cuda_ds41_common.hpp. All pointers are device pointers; outputs must not overlap inputs unless a call
// says otherwise.

#include "strix/cuda/ds41_common.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::ds41::cuda {

// ---- A8 activations -----------------------------------------------------------------------------------------------
// out row r = quantize_a8(x[r * x_stride .. + width)) (codes, deltas, sums), bitwise. width a multiple of 32.
struct QuantizeA8View {
    const float* x = nullptr;
    std::uint32_t rows = 0, width = 0, x_stride = 0;  // x_stride 0 = width
    A8Rows out;                                       // out.stride 0 = width
};
void enqueue_quantize_a8(const QuantizeA8View& view, cudaStream_t stream);

// ---- Q8_0 x A8 sites ----------------------------------------------------------------------------------------------
// out[r * out_stride + o] = product_q8_0(weight row (first + o) columns [k_offset, k_offset + width), x_g of row r)
// for o in [0, count), with x_g the A8 slice at codes + r * x.stride + g * width (deltas / 32) and g = o / per_group
// (per_group 0 = count: a plain projection; 1024 for the grouped attn_output_a). With accumulate the result is
// out = out + product (the K-split sites, q8_0_a8_split_rows: one call per part in part order, the first without).
// Every row count gives the bits of the one-row product (decode kernel up to 8 rows per pass, more rows in passes).
// Tiled form of a Q8_0 tensor for the tensor-core kernel (src/cuda/core/dense/q8_a8_imma.cu): per 16-row tile and 32-weight
// group, 512 bytes of codes in mma fragment order and 16 FP16 scales. Same weights, same products (bitwise).
// rows a multiple of 16. Tensors that share an input can be concatenated row-wise before the repack (q_a + kv,
// shared gate + up) and run as one site.
using Q8Tiled = ::strix::cuda::core::Q8Tiled;
[[nodiscard]] std::size_t q8_tiled_code_bytes(std::uint32_t rows, std::uint32_t width);
[[nodiscard]] std::size_t q8_tiled_scale_count(std::uint32_t rows, std::uint32_t width);
struct Q8TiledBuffer {
    DeviceBuffer<std::uint8_t> codes;
    DeviceBuffer<std::uint16_t> scales;
    std::uint32_t rows = 0, width = 0;
    // From GGUF Q8_0 rows in device memory (rows x width values); enqueued on stream.
    void repack(const std::uint8_t* q8, std::uint32_t rows, std::uint32_t width, cudaStream_t stream);
    [[nodiscard]] Q8Tiled view() const noexcept { return {codes.get(), scales.get(), rows, width}; }
};

struct Q8A8View {
    Q8Weights weight;                      // the whole tensor (weight.width = the full row)
    Q8Tiled tiled;                         // when set (codes not null): the tensor-core kernel; first, count,
                                           // per_group multiples of 16
    std::uint32_t split = 0;               // tiled: lane classes over 1, 2 or 4 blocks (0 = by size; needs scratch)
    float* split_scratch = nullptr;        // tiled, split > 1: q8_0_a8_split_scratch_floats(count, rows, split)
    std::size_t split_scratch_floats = 0;
    std::uint32_t first = 0, count = 0, per_group = 0;
    std::uint32_t k_offset = 0, width = 0;  // columns read; width 0 = weight.width
    A8Rows x;                              // x.rows rows
    float* out = nullptr;
    std::uint32_t out_stride = 0;          // 0 = count
    bool accumulate = false;
    // tiled only: launched with programmatic stream serialization (sm_90+): the weight slab is prefetched into L2
    // before waiting for the previous kernel; same bits
    bool pdl = false;
    int occupancy = 0;  // 0: process default; 3: audit register allocation
    // tiled, split 1, not accumulate: 2 = also the next K-half (k_offset + width, the activation columns shifted by
    // width) in the same launch, out = y(half 0) + y(half 1): the bits of a second launch with accumulate
    std::uint32_t k_halves = 1;
    // tiled, split 1, not accumulate, first 0, count = hc_hidden: the epilogue also runs hc_expand with
    // sub = out for its outputs (hc_out[r][k][i] from hc_residual[r][j][i] and hc_mixes[r][24]; hc_out == hc_residual
    // allowed); the bits of enqueue_hc_expand after this launch
    const float* hc_residual = nullptr;
    const float* hc_mixes = nullptr;
    float* hc_out = nullptr;
    std::uint32_t hc_hidden = 0;
};
void enqueue_q8_0_a8(const Q8A8View& view, cudaStream_t stream);
// The tensor-core form (view.tiled set): the integer dot of each group comes from mma.m16n8k32.s8; the FP32 terms,
// the lane-class sums and the butterfly tree are those of product_q8_0. enqueue_q8_0_a8 dispatches here.
void enqueue_q8_0_a8_tiled(const Q8A8View& view, cudaStream_t stream);
[[nodiscard]] std::size_t q8_0_a8_split_scratch_floats(std::uint32_t count, std::uint32_t rows, std::uint32_t split);

// ---- RMSNorm and hyper-connections (ds41_layer.hpp) ----------------------------------------------------------------
// One block per row. Input: x[r * x_stride + i], or (hc_stream not null) the collapse of hc_stream[r] (4 x n values)
// with pre[r * pre_stride + c] (identity pre when pre is null), as hc_collapse. Then y = weight * (x * rsqrt(
// chunked_dot(x, x) / n + eps)) as rms_norm, written to out (optional; x == out allowed) and its A8 (optional,
// a8.codes not null). collapsed (optional) gets the collapsed row before the norm (the hc_pre op output); weight null
// skips the norm (collapse only). n a multiple of 32, at most 8192.
struct NormView {
    const float* x = nullptr;
    std::uint32_t x_stride = 0;          // 0 = n
    const float* hc_stream = nullptr;    // [row][4][n]
    const float* pre = nullptr;          // 4 floats per row (a pre_carry row, or the pre of an hc mix row)
    std::uint32_t pre_stride = 0;        // 4 for pre_carry rows, 24 for mix rows
    const float* weight = nullptr;
    std::uint32_t rows = 0, n = 0;
    float eps = 1e-20F;
    float* out = nullptr;
    std::uint32_t out_stride = 0;        // 0 = n
    float* collapsed = nullptr;          // [row][n], optional
    A8Rows a8;                           // a8.stride 0 = n
};
void enqueue_norm(const NormView& view, cudaStream_t stream);

// hc mixes of one site (hc_mixes): per row, 24 floats {pre 4, post 4, comb 16} into mixes[r * 24]: chunked dots of
// the 24 F16 fn rows and of the stream with itself over 4 * hidden values (one warp per (dot, chunk, row)), then per
// row the ascending chunk sums, the split and the Sinkhorn rounds (16 lanes, one per comb entry).
// 4 * hidden a multiple of 640. partial: hc_mix_workspace_bytes(rows, hidden) bytes of scratch.
struct HcMixView {
    const float* stream = nullptr;       // [row][4][hidden]
    std::uint32_t rows = 0, hidden = 0;
    const std::uint16_t* fn = nullptr;   // F16 [24][4 * hidden]
    const float* base = nullptr;         // [24]
    const float* scale = nullptr;        // [3]
    std::uint32_t iters = 20;
    float hc_eps = 1e-6F, norm_eps = 1e-20F;
    float* mixes = nullptr;              // [row][24]
    float* partial = nullptr;            // workspace
};
[[nodiscard]] std::size_t hc_mix_workspace_bytes(std::uint32_t rows, std::uint32_t hidden);
void enqueue_hc_mixes(const HcMixView& view, cudaStream_t stream);
// Two launches instead of three: chunk partials, then independent Sinkhorn and collapse/norm/A8 blocks.
// norm.pre must refer to the PREVIOUS site's coefficients (never mix.mixes); matching stream/rows/hidden.
void enqueue_hc_mixes_norm(const HcMixView& mix, const NormView& norm, cudaStream_t stream);
// hc_mixes_norm as one cluster of norm parts per row on compute capability 9.0+ (same bits; default off)
void set_hc_norm_cluster(bool on);
// the PDL prologue of the tiled Q8 GEMV prefetches its weights into L2 (default) or only waits
void set_q8_pdl_prefetch(bool on);
// PDL prologue mode: 0 wait at entry, 1 L2 slab prefetch (default), 2 the first step's weights into registers
void set_q8_pdl_mode(int mode);
// Compact exact single-row Q8 kernel; generic fallback for other shapes.
void set_q8_r1(bool on);
void set_q8_r1_pad(bool on);
void set_q8_occupancy(int blocks);
// hc chunk partials with 5 warps per block (5 x the blocks) or 25 (the default); same bits
void set_hc_partial_warps(unsigned warps);

// hc_expand: out[r][k][i] = post[k] * sub[r][i] + (sum over j ascending from 0 of comb[j][k] * residual[r][j][i]);
// out == residual allowed (the stream updated in place).
struct HcExpandView {
    const float* sub = nullptr;
    std::uint32_t sub_stride = 0;        // 0 = hidden
    const float* residual = nullptr;     // [row][4][hidden]
    const float* mixes = nullptr;        // [row][24]
    std::uint32_t rows = 0, hidden = 0;
    float* out = nullptr;                // [row][4][hidden]
    float* pre_carry = nullptr;          // optional [row][4]: copy mixes pre in this launch
};
void enqueue_hc_expand(const HcExpandView& view, cudaStream_t stream);

// ---- router (reference_router) -------------------------------------------------------------------------------------
// logits[r][e] = lane dot of the F32 weight row e with x[r]; scores = router_score; top_k by score + bias (bias_image
// for rows flagged in image_rows), larger first, lower id on ties (a NaN key is taken only when it is the first
// untaken expert, as in the sequential reference); weights = score / (ascending sum + 1e-20) * route_scale (score *
// route_scale when top_k == 1). top_k <= 32.
struct RouterView {
    const float* x = nullptr;
    std::uint32_t x_stride = 0;          // 0 = hidden
    std::uint32_t rows = 0;
    F32Weights weight;                   // [experts][hidden]
    const float* bias = nullptr;
    const float* bias_image = nullptr;
    const std::uint8_t* image_rows = nullptr;  // optional per-row flag
    std::uint32_t top_k = 6;
    float route_scale = 1.5F;
    float* scores = nullptr;             // [row][experts] (scratch or output)
    std::uint32_t* ids = nullptr;        // [row][top_k]
    float* weights = nullptr;            // [row][top_k]
};
void enqueue_router(const RouterView& view, cudaStream_t stream);

// ---- shared expert (Q8_0 SwiGLU, identity q8_0-a8-swiglu-clamp10-v1) -----------------------------------------------
// out[r] = product_q8_0(down, A8(swiglu_clamped(gate . x_r, up . x_r, limit))) per output, x the A8 MoE input.
struct SharedExpertView {
    Q8Weights gate, up, down;            // [inter][hidden], [inter][hidden], [hidden][inter]
    Q8Tiled gate_up_tiled;               // optional tensor-core form: gate rows then up rows ([2 inter][hidden])
    Q8Tiled down_tiled;                  // optional tensor-core form of down
    float limit = 10.0F;
    A8Rows x;                            // width hidden
    float* out = nullptr;                // [row][hidden]
    std::uint32_t out_stride = 0;        // 0 = hidden
    void* workspace = nullptr;           // shared_expert_workspace_bytes(rows, inter)
    bool pdl = false;                    // the tiled gate/up and down GEMVs launched as Q8A8View::pdl
};
[[nodiscard]] std::size_t shared_expert_workspace_bytes(std::uint32_t rows, std::uint32_t inter);
void enqueue_shared_expert(const SharedExpertView& view, cudaStream_t stream);
// Tiled gate/up, then SwiGLU + A8 + tiled down (two launches total). Same scratch allocation as above.
void enqueue_shared_expert_fused(const SharedExpertView& view, cudaStream_t stream);
// Second launch alone: each down tile computes identical A8 groups in CTA-local memory before the legacy
// product. inter = down.width, a multiple of 32, <=4096; gate_up is [row][2*inter]. Optional A8 tap (tile 0 only).
void enqueue_swiglu_a8_down(const float* gate_up, std::uint32_t rows, float limit, Q8Tiled down,
                           float* out, std::uint32_t out_stride, A8Rows tap, cudaStream_t stream);
// act[r][o] = swiglu_clamped(gate[r][o], up[r][o], limit) and its A8 (one block per row; inter a multiple of 32).
struct SwigluA8View {
    const float* gate = nullptr;
    const float* up = nullptr;
    std::uint32_t rows = 0, inter = 0;
    std::uint32_t in_stride = 0;         // row stride of gate and up (0 = inter)
    float limit = 10.0F;
    float* act = nullptr;                // optional [row][inter]
    A8Rows a8;
};
void enqueue_swiglu_a8(const SwigluA8View& view, cudaStream_t stream);

// ---- F16 projections (lane dot, f16_rows) --------------------------------------------------------------------------
// out[r * out_stride + o] = f16_row_dot(weight row first + o, x[r * x_stride ..]) for o < count.
struct F16RowsView {
    F16Weights weight;
    std::uint32_t first = 0, count = 0;
    const float* x = nullptr;
    std::uint32_t rows = 0, x_stride = 0;  // 0 = weight.width
    float* out = nullptr;
    std::uint32_t out_stride = 0;          // 0 = count
};
void enqueue_f16_rows(const F16RowsView& view, cudaStream_t stream);

// ---- embedding, Engram, head ---------------------------------------------------------------------------------------
// stream[r][c][i] = F16 row tokens[r] of the table for every copy c; pre_carry[r] = identity {1, 0, 0, 0} (optional).
struct EmbedView {
    F16Weights table;                    // [vocab][hidden]
    const std::uint32_t* tokens = nullptr;
    std::uint32_t rows = 0;
    float* stream = nullptr;
    float* pre_carry = nullptr;
};
void enqueue_embed(const EmbedView& view, cudaStream_t stream);

// Gather of 264-byte Engram rows: out[k] = table row ids[k] (table in device memory or mapped host memory).
void enqueue_engram_gather(const std::uint8_t* table, const std::uint64_t* ids, std::uint32_t count, std::uint8_t* out,
                           cudaStream_t stream);
// Engram injection of one layer (engram op): rows [r][cols][264] decoded (E4M3 x E8M0) and rounded to bfloat16,
// kv = engram_kv * rows (F16 lane dot: 5 * hidden outputs of cols * 256 values), then per copy the gate of
// engram_gates and stream = stream + gate * value, in place.
struct EngramView {
    const std::uint8_t* row_bytes = nullptr;  // [row][cols][264]
    std::uint32_t rows = 0, cols = 24, hidden = 0;
    F16Weights kv;                            // [5 * hidden][cols * 256]
    const float* q_norm = nullptr;            // [4 * hidden]
    const float* k_norm = nullptr;
    float eps = 1e-20F;
    float* stream = nullptr;                  // [row][4][hidden], in place
    float* kv_out = nullptr;                  // optional [row][5 * hidden] (else in the workspace)
    void* workspace = nullptr;                // engram_workspace_bytes(rows, cols, hidden)
};
[[nodiscard]] std::size_t engram_workspace_bytes(std::uint32_t rows, std::uint32_t cols, std::uint32_t hidden);
void enqueue_engram(const EngramView& view, cudaStream_t stream);

// Greedy candidate of each row's logits (indices first_index + i): the largest finite logit, lowest index on ties,
// finite = 0 if any logit is not finite (argmax_slice; equal to combine_argmax over any vocabulary split). out has
// the layout of ds41::ArgmaxCandidate {float value; uint32 index; uint32 finite}.
struct ArgmaxView {
    const float* logits = nullptr;
    std::uint32_t rows = 0, count = 0, stride = 0;  // stride 0 = count
    std::uint32_t first_index = 0;
    void* out = nullptr;                            // ArgmaxCandidate[rows]
};
void enqueue_argmax(const ArgmaxView& view, cudaStream_t stream);

}  // namespace strix::ds41::cuda
