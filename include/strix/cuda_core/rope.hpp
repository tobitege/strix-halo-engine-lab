#pragma once
#include <cmath>
#include <vector>

#include "strix/cuda_core/types.hpp"

namespace strix::cuda_core {

// NeoX rotate-half over the first rotary_dim dims of each head (pairs i, i + rotary_dim/2). Table [pos][rotary_dim/2] of
// cos and sin; angle = pos * theta^(-2i/rotary_dim) evaluated in double on the host, rounded to float.
inline void rope_table_host(std::uint32_t positions, std::uint32_t rotary_dim, double theta, std::vector<float>& cos_t,
                            std::vector<float>& sin_t) {
    const std::uint32_t half = rotary_dim / 2;
    cos_t.assign(static_cast<std::size_t>(positions) * half, 0.0f);
    sin_t.assign(cos_t.size(), 0.0f);
    for (std::uint32_t i = 0; i < half; ++i) {
        const double inv = std::pow(theta, -2.0 * static_cast<double>(i) / static_cast<double>(rotary_dim));
        for (std::uint32_t p = 0; p < positions; ++p) {
            const double a = static_cast<double>(p) * inv;
            cos_t[static_cast<std::size_t>(p) * half + i] = static_cast<float>(std::cos(a));
            sin_t[static_cast<std::size_t>(p) * half + i] = static_cast<float>(std::sin(a));
        }
    }
}

// In place on x (view rows x heads(groups) x head_dim(n)); row r uses position *pos0_dev + r (device scalar: graph-capturable).
// out1 = x1*c - x2*s ; out2 = x2*c + x1*s  (each product rounded, then one add/sub).
cudaError_t rope_apply(float* x, const GroupedView& view, std::uint32_t rotary_dim, const float* cos_table,
                       const float* sin_table, const std::uint32_t* pos0_dev, cudaStream_t stream, const RowMap& rm = {});

}  // namespace strix::cuda_core
