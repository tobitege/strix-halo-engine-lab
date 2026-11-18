#pragma once

// deepseek41 CUDA routed experts, decode v2: the bits of enqueue_routed_gate_up + enqueue_routed_down
// (cuda_ds41_moe.hpp), with whole weight slabs in flight per block and small work items. The SwiGLU output goes
// through y ([slots][inter] FP32); the down kernel quantizes it (the A8 of the v1 gate/up epilogue) and, when
// view.act.codes is set, also writes that A8 (for taps).

#include "strix/cuda/ds41_moe.hpp"

namespace strix::ds41::cuda {

// Diagnostics only (bench): 0 = the exact kernels, 1 = weight loads without the products, 2 = no IQ2 grid lookups.
void set_routed_v2_diagnostic(int mode);
// Diagnostics only: gate_up_v3 clock64 stamps [block][8][4] (top, slab ready, computed, end); nullptr = off.
void set_routed_v3_stamps(unsigned long long* stamps);
[[nodiscard]] std::size_t routed_v2_y_floats(std::uint32_t slots, const MoeShape& shape) noexcept;
// gate/up of every routed entry: y[slot][o] = swiglu_clamped(gate_o . x[row], up_o . x[row]) (view.act unused).
void enqueue_routed_gate_up_v2(const RoutedGateUpView& view, float* y, cudaStream_t stream);
// down of every routed entry: raw[slot][o] = down_o . quantize_a8(y[slot]) (no expert TP).
void enqueue_routed_down_v2(const RoutedDownView& view, const float* y, cudaStream_t stream);

// v3: persistent blocks, double-buffered weight slabs (hidden 5120 / inter 2304, 1..8 rows). gate/up writes the A8 of
// the SwiGLU outputs into view.act exactly as v1 (plus y, FP32 [slots][inter]); counters: routed_v3_counter_words(slots)
// words, zero before the first launch (every launch leaves them zero). down reads view.act as v1.
[[nodiscard]] bool routed_v3_supported(const MoeShape& shape, std::uint32_t max_rows) noexcept;
// Diagnostics: v3 gate/up with one item per block (8 outputs, ordinary scheduling) instead of persistent blocks.
void set_routed_v3_one_item(bool on);
[[nodiscard]] std::size_t routed_v3_counter_words(std::uint32_t slots) noexcept;
void enqueue_routed_gate_up_v3(const RoutedGateUpView& view, float* y, std::uint32_t* counters, cudaStream_t stream);
void enqueue_routed_down_v3(const RoutedDownView& view, cudaStream_t stream);
// v1b: v1's gate/up schedule and epilogue (view.act as v1) with v3's decode.
// dsrtxk diagnostics: the v1b R1 kernel form (0 = 3 blocks/SM bound + max-shared carveout, 1 = bound only, 2 = neither)
void set_v1b_r1_form(int form);
void enqueue_routed_gate_up_v1b(const RoutedGateUpView& view, cudaStream_t stream);

}  // namespace strix::ds41::cuda
