#pragma once
#include "strix/cuda_core/types.hpp"

namespace strix::cuda_core {

enum class GateKind : std::uint8_t { swish, sigmoid };

// Every norm reduces the group's n squares with ONE order: thread t (of 256) adds x[t + 256 i] ascending, the 32-lane xor
// butterfly (16..1) reduces each warp, then the 8 warp sums are added in warp order 0..7. Independent of rows/groups.
// All views must share (rows, groups, n). In-place (y == x with equal views) is allowed.
// rmsnorm:  y = (x * (1 / sqrt(ss / n + eps))) * gain[i]     gain = stored effective gain (null -> 1)
cudaError_t rmsnorm(const float* x, const GroupedView& xv, const float* gain, float* y, const GroupedView& yv, float eps,
                    cudaStream_t stream);
// gated: y = ((x * inv) * gain[i]) * act(z), act = silu (swish) or sigmoid, silu(z) = z / (1 + expf(-z)).
cudaError_t gated_rmsnorm(const float* x, const GroupedView& xv, const float* gain, const float* z, const GroupedView& zv,
                          GateKind kind, float* y, const GroupedView& yv, float eps, cudaStream_t stream);
// l2norm: y = (x * (1 / sqrt(ss + eps))) * scale
cudaError_t l2norm(const float* x, const GroupedView& xv, float* y, const GroupedView& yv, float eps, float scale,
                   cudaStream_t stream);
// x[i] += d[i] for i < count.
cudaError_t add_inplace(float* x, const float* d, std::uint32_t count, cudaStream_t stream);

}  // namespace strix::cuda_core
