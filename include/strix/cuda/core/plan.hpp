#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace strix::cuda::core {

enum class DenseWeightFormat { q8_0_tiled_16x32_a8 };
// Bitwise product_q8_0 group, lane-class and butterfly order; see docs/models/deepseek-v4.1/cuda-numeric-identity.md.
enum class DenseNumericIdentity { product_q8_0_bitwise };
enum class DenseMode { decode, prefill };  // decode: activation rows <= 8; prefill: more than 8

struct DensePlan {
    std::uint32_t count = 0, width = 0, rows = 0;  // output rows of weights, K width, activation rows
    DenseWeightFormat format = DenseWeightFormat::q8_0_tiled_16x32_a8;
    DenseNumericIdentity identity = DenseNumericIdentity::product_q8_0_bitwise;
    DenseMode mode = DenseMode::decode;
    int required_sm_major = 8, required_sm_minor = 6;
    // The kernel never allocates and never synchronises the host; workspace and stream belong to the caller.
    // Device barriers inside a tile preserve its declared reduction order.
    float* split_scratch = nullptr;
    std::size_t split_scratch_floats = 0;
    cudaStream_t stream = nullptr;
};

}  // namespace strix::cuda::core
