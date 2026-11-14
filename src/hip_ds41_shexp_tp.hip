// Shared expert TP on the trellis library (ds41_shexp_tp.hpp, identity ds41-trellis-k5-v1:shexp-tp).
#include "strix/ds41_shexp_tp.hpp"
#include "strix/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace strix::ds41 {
namespace {

void check(hipError_t e, const char* what) {
    ::strix::hip_check_prefixed("ds41 shexp tp: ", e, what);
}
unsigned blocks_for(std::size_t n, unsigned block) { return static_cast<unsigned>((n + block - 1U) / block); }

__global__ void add_inplace_kernel(float* __restrict__ y, const float* __restrict__ t, std::size_t n) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < n) y[i] = y[i] + t[i];
}
__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b, std::size_t n, float* __restrict__ y) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

}  // namespace

TrellisSlice shexp_tp_gate_up_slice(std::uint32_t inter, std::uint32_t rank) noexcept {
    return {TrellisSlice::Kind::rows, inter / shexp_tp_ranks * rank, inter / shexp_tp_ranks};
}
TrellisSlice shexp_tp_down_slice(std::uint32_t inter, std::uint32_t rank) noexcept {
    return {TrellisSlice::Kind::columns, inter / shexp_tp_ranks * rank, inter / shexp_tp_ranks};
}

void enqueue_shexp_tp_partial(const ShexpTpSlices& s, const float* x, std::uint32_t rows, float limit, float* gate, float* up,
                              float* act, float* partial, void* stream) {
    if (s.gate == nullptr || s.up == nullptr || s.down == nullptr) throw std::logic_error("ds41 shexp tp: slices not prepared");
    const std::uint32_t H = s.gate->columns, hI = s.gate->rows;
    if (s.up->rows != hI || s.up->columns != H || s.down->columns != hI || s.down->rows != H)
        throw std::logic_error("ds41 shexp tp: slice geometry");
    // contiguous rows of the exact widths: enqueue_trellis_linear submits them as they are (these shapes take any row
    // count 1..8 in one class: no padding, no staging)
    enqueue_trellis_linear(*s.gate, x, gate, rows, H, hI, false, stream);
    enqueue_trellis_linear(*s.up, x, up, rows, H, hI, false, stream);
    enqueue_swiglu_f32(gate, up, rows, hI, limit, act, stream);
    enqueue_trellis_linear(*s.down, act, partial, rows, hI, H, false, stream);
}

void enqueue_add_inplace_f32(float* y, const float* t, std::size_t n, void* stream) {
    if (n == 0U) return;
    hipLaunchKernelGGL(add_inplace_kernel, dim3(blocks_for(n, 256)), dim3(256), 0, static_cast<hipStream_t>(stream), y, t, n);
    check(hipGetLastError(), "add in place");
}
void enqueue_add_f32(const float* a, const float* b, std::size_t n, float* y, void* stream) {
    if (n == 0U) return;
    hipLaunchKernelGGL(add_kernel, dim3(blocks_for(n, 256)), dim3(256), 0, static_cast<hipStream_t>(stream), a, b, n, y);
    check(hipGetLastError(), "add");
}

}  // namespace strix::ds41
