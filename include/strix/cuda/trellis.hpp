#pragma once

// Experimental packed linear operators. No weight expansion or implicit enablement.
#include "strix/cuda/ds41_common.hpp"

namespace strix::ds41::cuda::trellis {

enum class Mode { disabled, gemv_quality, mma_prefill_quality, hip_bitwise_required };

// Q is [input/16, output/16, 16*bits] little-endian uint16 containers.
// Only the procedural mul1 codebook and complete signed FP16 scales are admitted.
struct Weights {
    const std::uint16_t *packed = nullptr, *suh = nullptr, *svh = nullptr;
    const float* bias = nullptr;
    std::size_t packed_bytes = 0, suh_bytes = 0, svh_bytes = 0, bias_bytes = 0;
    std::uint32_t input = 0, output = 0, bits = 0, marker = 0x83dcd12dU;
};

struct Linear {
    Weights weights;
    const float* x = nullptr;
    float* y = nullptr;
    std::size_t x_bytes = 0, y_bytes = 0, x_stride = 0, y_stride = 0;
    std::uint32_t rows = 0, splits = 4;
    Mode mode = Mode::disabled;
    void* workspace = nullptr;
    std::size_t workspace_bytes = 0;
};

// Queries validate shapes, exact payload lengths, mode and checked arithmetic;
// they neither inspect device memory nor initialize CUDA. Disabled and the
// unqualified HIP identity fail explicitly. GEMV accepts R<=8; MMA accepts any R.
[[nodiscard]] std::size_t packed_bytes(std::uint32_t input, std::uint32_t output, std::uint32_t bits);
[[nodiscard]] std::size_t workspace_bytes(const Linear&);
// Enqueue validates pointer spans/alignment/aliasing and declared capacities
// before the first launch. Caller owns device allocations and stream ordering.
// All work is async, allocation-free, capture-safe, overwrites Y, and ignores
// old Y. Zero rows is a validated no-op. No alpha/beta or implicit A8 conversion.
void enqueue_linear(const Linear&, cudaStream_t);

// Diagnostic exact rotated Q, row-major [input,output]. Does not apply scales/H.
void enqueue_decode(const Weights&, std::uint16_t* q, std::size_t q_bytes, cudaStream_t);
// Diagnostic exhaustive procedural-codebook probe, indexed by uint16 state.
void enqueue_codebook(std::uint16_t* values, std::size_t bytes, cudaStream_t);

}  // namespace strix::ds41::cuda::trellis
