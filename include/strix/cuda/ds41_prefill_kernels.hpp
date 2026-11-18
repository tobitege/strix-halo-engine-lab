#pragma once

// deepseek41 CUDA backend: prefill-shape forms of the lane-dot sites (many rows per call). Every output has the bits
// of the decode kernels (cuda_ds41_core.hpp enqueue_f16_rows / enqueue_router / enqueue_engram, cuda_ds41_attention.hpp
// attn::enqueue_f16_rows) and of their host references (f16_rows of ds41_layer.hpp, reference_router of
// ds41_moe.hpp), whatever the number of rows.
//
// Declared order (ds41-exact-v1): for an output (o, r), lane class c = element index mod 32 accumulates
// acc = acc + w * x (two roundings, no fused multiply-add) over its elements ascending from 0; the 32 class sums are
// combined by the xor butterfly v[l] = v[l] + v[l ^ d], d = 16 .. 1.
//
// Kernel shape: a block computes a tile of outputs x rows; each warp a register tile of TM outputs x TN rows with
// lane = lane class (every lane holds the class sums of the whole tile), weights and rows staged through shared
// memory by cp.async (each weight tile is read once per BN rows instead of once per 8 rows). The butterfly runs as a
// reduce-scatter over the tile: at step d lane l keeps the half of the tile selected by bit d of l and adds the
// partner's copy of it, so every kept value is the butterfly's v[l] + v[l ^ d] at every step (the same tree, the
// same operand pairs).

#include "strix/cuda/ds41_common.hpp"
#include "strix/cuda/ds41_core.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::ds41::cuda::pk {

// One projection: out[r * out_stride + o] = lane dot of weight row o (width values) with x[r * x_stride ..), o < count,
// r < rows (of the view). width a multiple of 128; weight rows, x rows and the bases 16-byte aligned.
struct LaneDotJob {
    const void* weight = nullptr;          // F16 (std::uint16_t) or F32 rows, count x width, contiguous
    std::uint32_t width = 0, count = 0;
    const float* x = nullptr;
    std::uint32_t x_stride = 0;            // 0 = width
    float* out = nullptr;
    std::uint32_t out_stride = 0;          // 0 = count
};
enum class WeightType : std::uint8_t { f16 = 0, f32 = 1 };
enum class Epilogue : std::uint8_t { none = 0, router_score = 1 };
// Up to four jobs of one weight type per launch (they may read different x and have different widths).
struct LaneDotView {
    LaneDotJob jobs[4]{};
    std::uint32_t job_count = 0, rows = 0;
    WeightType type = WeightType::f16;
    Epilogue epilogue = Epilogue::none;    // router_score: out = router_score(dot)
};
void enqueue_lane_dot(const LaneDotView& view, cudaStream_t stream);

// enqueue_engram with the projection on enqueue_lane_dot (same view, same workspace: engram_workspace_bytes).
void enqueue_engram(const EngramView& view, cudaStream_t stream);
// enqueue_router with the logits on enqueue_lane_dot (same view; scores sized rows x experts).
void enqueue_router(const RouterView& view, cudaStream_t stream);

}  // namespace strix::ds41::cuda::pk
