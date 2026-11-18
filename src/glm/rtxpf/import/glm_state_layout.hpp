#pragma once
#include <cstdint>
#include "glm_dims.hpp"
#include "strix/block_summary_state.hpp"

namespace strix::glm::state_layout {
// Serialized convolution blocks preserve q/k/v-major storage, not per-head interleaving.
inline constexpr std::uint64_t kConvFloatsPerHead = 3ULL * kKdaDim * (kConvK - 1);
constexpr std::uint64_t conv_floats(std::uint64_t heads) { return heads * kConvFloatsPerHead; }
constexpr std::uint64_t pooled_rows(std::uint64_t capacity) { return capacity / kKpool; }
// DSA index key + gate (BF16 [128] each per row) are read only while their pool block is open (glm_dsa.hip
// dsa_store_kernel, glm_prefill.hip, glm_spec_dev.hip, glm_attn_fast.hip: rows t0..t0+3 of the closing block); the
// score reads the pooled FP32 key only. So the closed blocks travel as pooled rows and only the open tail as raw rows.
inline constexpr strix::state::BlockSummaryGeometry kDsaIndex{kKpool, 2ULL * kIdxDim * 2ULL, kIdxDim * 4ULL,
                                                              strix::state::SummaryDtype::fp32};
constexpr strix::state::RowWindow index_rows(strix::state::RawRows form, std::uint64_t frontier) {
    return strix::state::raw_window(form, frontier, kKpool);
}
static_assert(kConvFloatsPerHead == 3 * 128 * 3);
static_assert(sizeof(float) == 4);
}  // namespace strix::glm::state_layout
