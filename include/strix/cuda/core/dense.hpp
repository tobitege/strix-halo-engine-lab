#pragma once

#include "strix/cuda/core/types.hpp"
#include "strix/cuda/core/plan.hpp"

namespace strix::cuda::core::dense {

// Tiled dense site; split scratch and stream are supplied by DensePlan.
// hc_expand and SwiGLU-down are optional epilogues/preparation of the same core tile,
// preserving the standalone operations' arithmetic and the existing tile instruction order.
struct Q8A8View {
    Q8Tiled tiled;                         // when set (codes not null): the tensor-core kernel; first, count,
                                           // per_group multiples of 16
    std::uint32_t split = 0;               // tiled: lane classes over 1, 2 or 4 blocks (0 = by size; needs scratch)
    std::uint32_t first = 0, count = 0, per_group = 0;
    std::uint32_t k_offset = 0, width = 0;  // columns read; width 0 = tiled.width
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

void enqueue_q8_0_a8_tiled(const Q8A8View& view, const DensePlan& plan);
void enqueue_swiglu_a8_down(const float* gate_up, float limit, Q8Tiled down,
                          float* out, std::uint32_t out_stride, A8Rows tap, const DensePlan& plan);
// Enqueue only: caller provides already allocated tiled storage with the stated geometry.
void enqueue_q8_tiled_repack(const std::uint8_t* q8, std::uint8_t* codes, std::uint16_t* scales,
                             std::uint32_t rows, std::uint32_t width, cudaStream_t stream);
[[nodiscard]] std::size_t q8_tiled_code_bytes(std::uint32_t rows, std::uint32_t width);
[[nodiscard]] std::size_t q8_tiled_scale_count(std::uint32_t rows, std::uint32_t width);
[[nodiscard]] std::size_t q8_0_a8_split_scratch_floats(std::uint32_t count, std::uint32_t rows, std::uint32_t split);
void set_q8_pdl_prefetch(bool on);
void set_q8_pdl_mode(int mode);
void set_q8_r1(bool on);
void set_q8_r1_pad(bool on);
void set_q8_occupancy(int blocks);

}  // namespace strix::cuda::core::dense
