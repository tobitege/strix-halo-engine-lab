#pragma once
#include "strix/cuda/core/nvfp4.hpp"
#include <cuda_runtime_api.h>
namespace strix::cuda::core::nvfp4 {
// Capability selection rejects non-SM120; never changes the current device.
// Workspace is caller-owned, 256-byte aligned, disjoint from operands/output.
// No allocation, copies, device synchronization or hidden stream state.
cudaError_t enqueue(const View&, const Plan&, void* workspace,
                    std::size_t workspace_bytes, cudaStream_t);
// Quantise finite FP32 activations to the same canonical operand layout as the
// host quantizer. K positive multiple of 64; strides are elements / bytes.
// The caller supplies the positive finite tensor scale and all output buffers.
cudaError_t quantize_enqueue(const float* input, std::size_t input_stride,
    std::uint32_t rows, std::uint32_t k, std::uint8_t* codes,
    std::size_t code_stride, std::uint8_t* scales, std::size_t scale_stride,
    float tensor_scale, cudaStream_t);
}
