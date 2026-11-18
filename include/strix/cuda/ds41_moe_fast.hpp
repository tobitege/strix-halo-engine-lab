#pragma once

// deepseek41 CUDA backend, routed experts for prefill, FAST arm (sm_120). NOT the host reference's bits: a quality
// arm checked by its error against the exact path (cuda_ds41_moe.hpp), selected by CudaPrefillOptions::fast_moe.
//
// Arithmetic (default, xa given and down_i8): the exact path's quantizations (A8 of the MoE input, IQ2_XXS / Q2_K
// integer codes, A8 of the clamped SwiGLU) on INT8 MMAs, each 32-weight group's int32 dot scaled once into an FP32
// accumulator by FMA in any order; the Q2_K dmin * m term as one FP16 MMA per 256-weight block. FP16 arm (no xa /
// down_i8 off): weight tiles decoded in shared memory to FP16 with their scales, the unquantized input and an FP16
// activation on FP16 MMAs with FP32 accumulation (warp-specialized: producer warps decode, MMA warps multiply).
// The per-slot down outputs are kept in FP16 (in the caller's raw buffer) and combined as y = shared + sum_k w_k *
// raw_k in FP32, slot order.

#include "strix/cuda/ds41_moe.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::ds41::cuda {

// The routed experts of one layer on rows of the MoE input x (FP32, [rows][hidden], the normed input the A8 of the
// exact path is taken from). Scratch: moe_fast_workspace_bytes(capacity, shape) bytes, rows <= capacity.
struct MoeFastLayerView {
    MoeShape shape;
    ExpertWeights weights;
    std::uint32_t rows = 0;
    std::uint32_t split = 192;             // the combine's low/high split (expert id)
    const float* x = nullptr;              // [rows][hidden] (FP16 gate/up; unused when xa is given)
    A8Rows xa;                             // optional A8 of x: given, gate/up run on INT8 MMAs (x.codes / x.deltas)
    bool down_i8 = true;                   // down on INT8 MMAs over the A8 of the activation (else FP16 activation)
    bool signed_lut = false;               // IQ2 shared sign-mask lookup (same codes), experimental/default off
    bool wide_tiles = false;               // 256-entry INT8 batches, legacy kernel on tails <=128 (default off)
    const std::uint32_t* ids = nullptr;    // [rows][top_k], router selection order
    const float* route_weights = nullptr;  // [rows][top_k]
    const float* shared = nullptr;         // [rows][hidden]
    float* raw = nullptr;                  // [rows * top_k][hidden] floats of scratch (holds the slot vectors in FP16)
    float* y = nullptr;                    // [rows][hidden]
    void* workspace = nullptr;
    std::uint32_t capacity = 0;
};
[[nodiscard]] std::size_t moe_fast_workspace_bytes(std::uint32_t capacity, const MoeShape& shape) noexcept;
void enqueue_moe_fast_layer(const MoeFastLayerView& view, cudaStream_t stream);
// The same layer with its experts in several places (enqueue_moe_layer_ranges): ids mapped through perm (null:
// identity, then mapped_ids is unused), routes / gate-up / down range by range (each range waits for its event), the
// combine in slot order as above. view.weights is unused.
void enqueue_moe_fast_layer_ranges(const MoeFastLayerView& view, const std::uint32_t* perm, std::uint32_t* mapped_ids, const MoeRange* ranges,
                                   std::uint32_t range_count, cudaStream_t stream);

}  // namespace strix::ds41::cuda
