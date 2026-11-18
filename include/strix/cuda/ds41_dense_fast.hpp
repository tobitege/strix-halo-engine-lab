#pragma once

// deepseek41 CUDA backend: fast (relaxed-arithmetic) forms of the dense prefill sites. NOT the bits of ds41-exact-v1:
// same weights, same A8 activations and the same per-group scaling, but FP32 sums in any order with FMA, so the
// results differ from the exact kernels by FP32 rounding only (quality gate: KL of the logits against the exact
// prefill). Selected by CudaPrefillOptions::fast_dense; the exact kernels stay the default.
//
//  - Q8_0 x A8 sites: an int8 tensor-core GEMM over the tiled weights (Q8Tiled, no extra copy) with large block
//    tiles; per 32-weight group term = float(dot) * (scale * delta) accumulated with FMA in group order.
//  - hc mixes: the 24 F16 projections and the sum of squares of many rows per block (fn staged once per block),
//    reduced and split (Sinkhorn) in the same kernel; optionally fused with the hc_expand that produces the stream.
//  - RMSNorm with the hc collapse and the A8 output: one block per row.

#include "strix/cuda/ds41_common.hpp"
#include "strix/cuda/ds41_core.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::ds41::cuda::fast {

// out as enqueue_q8_0_a8_tiled(view) up to FP32 rounding (split and split_scratch ignored). Needs the tiled form
// (view.tiled.codes) with first, count and per_group multiples of 16, 16-byte aligned activation codes; otherwise the
// exact prefill kernel runs.
[[nodiscard]] bool q8_0_a8_supported(const Q8A8View& view);
void enqueue_q8_0_a8(const Q8A8View& view, cudaStream_t stream);

// The shared expert (tiled form) on enqueue_q8_0_a8 above; same view and workspace as enqueue_shared_expert.
void enqueue_shared_expert(const SharedExpertView& view, cudaStream_t stream);

// quantize_a8 of rows (8 lanes per group; code = rint(v * (1 / delta)): a code may differ by one at a rounding tie).
void enqueue_quantize_a8(const QuantizeA8View& view, cudaStream_t stream);

// One pass over rows of the residual stream (many rows per block; view.partial unused):
//   [expand, emix set: stream = hc_expand(sub, stream, emix), in place]
//   [norm, weight set: x = rms_norm(collapse(stream, pre), weight) and its A8 (a8.codes set): the collapse of the
//    (expanded) stream with the pre of the previous site, so it needs nothing of the mixes computed here]
//   mixes = hc_mixes(stream) with the split (Sinkhorn) into view.mixes.
struct HcFusedView {
    HcMixView mix;
    const float* sub = nullptr;          // expand: [row][sub_stride]
    std::uint32_t sub_stride = 0;        // 0 = hidden (a multiple of 4)
    const float* emix = nullptr;         // expand: [row][24] (null: no expand)
    const float* pre = nullptr;          // norm: collapse pre [row][pre_stride] (null: identity)
    std::uint32_t pre_stride = 24;
    const float* weight = nullptr;       // norm weight (null: no norm)
    float eps = 1e-20F;
    float* x = nullptr;                  // norm output [row][hidden]
    A8Rows a8;                           // optional (codes null: none); stride 0 = hidden
    std::uint32_t rows_per_block = 0;    // 8, 16 or 32 (0: 32)
};
void enqueue_hc_fused(const HcFusedView& view, cudaStream_t stream);

// hc mixes of rows (view.partial unused). With expand set: first stream = hc_expand(expand) (written to expand.out,
// which must be view.stream), then the mixes of that stream; one pass over the rows.
void enqueue_hc_mixes(const HcMixView& view, cudaStream_t stream, const HcExpandView* expand = nullptr);

// RMSNorm (with the hc collapse / A8 as enqueue_norm; collapsed unsupported): one block per row.
void enqueue_norm(const NormView& view, cudaStream_t stream);

}  // namespace strix::ds41::cuda::fast
