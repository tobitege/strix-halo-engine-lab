#pragma once
// Internal: tensor-core dense path (see dense.hpp "mma arithmetic").
#include "strix/cuda_core/dense.hpp"

namespace strix::cuda_core {

constexpr int kMmaMaxMulti = 4;
struct MmaCall {
    const void* w[kMmaMaxMulti];
    float* y[kMmaMaxMulti];
    std::uint32_t out[kMmaMaxMulti];
    int n = 0;
    const void* w1 = nullptr;  // fused gate/up: the up weight (n == 1)
    WeightFormat format = WeightFormat::bf16;
    std::uint32_t in = 0, rows = 0;
    const float* x = nullptr;
    const float* residual = nullptr;
    int terms = 3;
};
// true when the mma path can serve the call (format, in % 32, out % 16); the caller falls back to the FP32 kernels otherwise.
bool dense_mma_supported(const MmaCall& c);
cudaError_t dense_mma_launch(const MmaCall& c, cudaStream_t stream);

}  // namespace strix::cuda_core
