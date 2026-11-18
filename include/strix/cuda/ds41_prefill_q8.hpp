#pragma once

// deepseek41 CUDA backend: prefill-shape form of the tiled Q8_0 x A8 sites (many rows per call). Same view as
// enqueue_q8_0_a8_tiled (cuda_ds41_core.hpp) and the same bits for every row count.
//
// Declared arithmetic (product_q8_0, ds41-exact-v1): per group g of 32 weights, term = float(dot) * (scale * delta);
// lane class c (groups c, c + 32, ... ascending) acc = acc + term from 0; the 32 class sums combined by the butterfly
// tree v[l] = v[l] + v[l + d], d = 16, 8, 4, 2, 1 (result v[0]).
//
// Kernel shape: a block computes a tile of outputs x rows, each warp a register tile over the full depth. The integer
// dot of a group comes from mma.m16n8k32.s8 as in the tiled kernel. The warp visits the lane classes in bit-reversed
// order (j = 0 .. 31, c = bitrev5(j)): the leaves of the butterfly tree then come left to right, so the tree is folded
// with a carry stack (as a binary counter: when class j ends, its sum is added to the right of the stack entries of
// the trailing one bits of j, then pushed), five saved sums instead of 32. Weight tiles, scales, activation codes and
// deltas are staged in shared memory by cp.async, each read once per block tile instead of once per 32 rows / 16
// outputs.

#include "strix/cuda/ds41_common.hpp"
#include "strix/cuda/ds41_core.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::ds41::cuda::pk {

// True when the many-row kernel takes the view (tiled weights; activation codes and stride 16-byte aligned; a grouped
// site with per_group a multiple of the block tile). Otherwise enqueue_q8_0_a8_rows runs enqueue_q8_0_a8_tiled.
[[nodiscard]] bool q8_0_a8_rows_supported(const Q8A8View& view);
// out as enqueue_q8_0_a8_tiled(view) (split and split_scratch are ignored), bitwise, for any number of rows.
void enqueue_q8_0_a8_rows(const Q8A8View& view, cudaStream_t stream);

// enqueue_shared_expert (tiled form: gate_up_tiled and down_tiled set) with both products on enqueue_q8_0_a8_rows;
// same view, same workspace (shared_expert_workspace_bytes), same bits.
void enqueue_shared_expert(const SharedExpertView& view, cudaStream_t stream);

}  // namespace strix::ds41::cuda::pk
