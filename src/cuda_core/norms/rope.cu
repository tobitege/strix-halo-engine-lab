// NeoX rotate-half RoPE over the first rotary_dim dims. See rope.hpp.
#include "strix/cuda_core/rope.hpp"

#include "../common.cuh"

namespace strix::cuda_core {
namespace {

__global__ void rope_kernel(float* x, GroupedView v, std::uint32_t half, const float* cos_t, const float* sin_t,
                            const std::uint32_t* pos0p, RowMap rm) {
    pdl_trigger();
    pdl_wait();
    const std::uint32_t r = blockIdx.y, h = blockIdx.x;
    const std::uint32_t pos = rm.pos ? rm.pos[r] : *pos0p + r;
    float* p = x + static_cast<std::size_t>(r) * v.rs() + static_cast<std::size_t>(h) * v.gs();
    const std::size_t tb = static_cast<std::size_t>(pos) * half;
    for (std::uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
        const float c = cos_t[tb + i], s = sin_t[tb + i];
        const float x1 = p[i], x2 = p[i + half];
        p[i] = __fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s));
        p[i + half] = __fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, s));
    }
}

}  // namespace

cudaError_t rope_apply(float* x, const GroupedView& view, std::uint32_t rotary_dim, const float* cos_table,
                       const float* sin_table, const std::uint32_t* pos0, cudaStream_t stream, const RowMap& rm) {
    if (rotary_dim == 0 || rotary_dim % 2 != 0 || rotary_dim > view.n) return cudaErrorInvalidValue;
    return launch_pdl(rope_kernel, dim3(view.groups, view.rows), dim3(32), stream, x, view, rotary_dim / 2, cos_table, sin_table, pos0, rm);
}

}  // namespace strix::cuda_core
