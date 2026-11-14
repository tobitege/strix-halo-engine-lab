#pragma once
// Host plan policy of the paired FAST_I4 chain: one decision for every engine (no device code, no environment).
#include <cstddef>

namespace strix::core::moe_i4 {
// requested: the plan selected for the call (0 BM32, 1 BM64, 2 BM128); the routed down always runs it.
// The paired gate/up GEMM keeps its pass-0 gate values either in registers or in a global FP32 scratch (GSave).
//   scratch admitted (GSave on and the caller's scratch holds one FP32 row of I values per assigned pair):
//     paired = requested, the two GEMMs share one tile list.
//   register save: BM128 spills, so paired runs at most plan 1 and a plan-2 down needs its own tile list.
// Every admitted form computes the same integers; only tile shape and the gate's residence change.
struct ChainPlan {
    int paired = 0, down = 0;
    bool gsave = false;                                            // pass-0 gate in the caller's scratch
    constexpr bool rebuild() const { return paired != down; }      // second tile build before the down
};
constexpr ChainPlan chain_plan(int requested, bool gsave_on, std::size_t scratch_floats, std::size_t needed_floats) {
    const bool admit = gsave_on && needed_floats != 0 && scratch_floats >= needed_floats;
    return {admit || requested < 1 ? requested : 1, requested, admit};
}
// FP32 values the scratch must hold: one row of I values for every pair index the binding can return.
constexpr std::size_t chain_gsave_floats(std::size_t pair_slots, std::size_t intermediate) { return pair_slots * intermediate; }
}  // namespace strix::core::moe_i4
