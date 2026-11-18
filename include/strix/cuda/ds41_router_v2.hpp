#pragma once

// deepseek41 CUDA router, decode v2: enqueue_router's bits (cuda_ds41_core.hpp) in one launch with the weight rows
// streamed at full width and the selection in the last finishing block. `counter`: one device word, zero before the
// first launch (every completed launch leaves it zero); a launch per counter at a time (the true router and the
// predictor each own one).

#include "strix/cuda/ds41_core.hpp"

namespace strix::ds41::cuda {

[[nodiscard]] bool router_v2_supported(const RouterView& view) noexcept;
// Diagnostics only: per-block globaltimer stamps [blocks][8] of the next launches (nullptr: off).
void set_router_v2_stamps(unsigned long long* stamps);
// 0: the loads as written; 2 (rows <= 2): the finisher's bias read at entry and the x rows prefetched into L1 during
// the weight load (same bits); any other value: 0
void set_router_v2_form(int form);
void enqueue_router_v2(const RouterView& view, std::uint32_t* counter, cudaStream_t stream);

}  // namespace strix::ds41::cuda
