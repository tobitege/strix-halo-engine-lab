#pragma once

// CUDA backend core: logprob partials of rows of logits held on the device (strix/logprob_partials.hpp contract).
// Same contract and order as strix/hip_logprob_partials.hpp. Model-agnostic: any engine whose head writes rows of a vocabulary slice binds it (DS41 per-rank slices, Qwen/GLM
// full rows). One 256-thread block per row; the FP32 exp sum uses the contract's lane order; only the 192-byte
// partials leave the kernel, so a scoring step moves rows x 192 bytes to the host or the peer, never the logits.

#include "strix/logprob_partials.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace strix {

struct CudaLogprobPartialsView {
    const float* logits = nullptr;  // row r, entry i (ID first + i) at logits[r * stride + i]
    std::uint32_t rows = 0, count = 0, stride = 0, first = 0;
    std::uint32_t top = 0;                   // <= kLogprobPartialTop
    const LogprobTarget* targets = nullptr;  // device, one per row; null: no target
    LogprobPartial* out = nullptr;           // device, `rows` partials
};

// Throws std::invalid_argument on a bad view (null pointers, top > kLogprobPartialTop, stride < count).
void enqueue_cuda_logprob_partials(const CudaLogprobPartialsView& view, cudaStream_t stream);

} // namespace strix
