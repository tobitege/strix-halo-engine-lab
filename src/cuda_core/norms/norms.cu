// RMSNorm (stored gain), gated RMSNorm, per-group l2norm, residual add. Reduction order documented in norms.hpp.
#include "strix/cuda_core/norms.hpp"

#include "../common.cuh"

namespace strix::cuda_core {
namespace {

enum class Mode : int { rms = 0, gated_swish = 1, gated_sigmoid = 2, l2 = 3 };

// Same order as before (thread t adds x[t + 256 i] ascending); the NREG values of a thread stay in registers between the sum and the
// write (one global read per element instead of two). Padding terms are 0 (p + 0*0 == p for p >= 0).
template <Mode M, int NREG>
__global__ void __launch_bounds__(256)
    norm_kernel(const float* x, GroupedView xv, const float* gain, const float* z, GroupedView zv, float* y, GroupedView yv,
                float eps, float scale) {
    __shared__ float red[8];
    pdl_trigger();
    pdl_wait();
    const std::uint32_t r = blockIdx.y, g = blockIdx.x, n = xv.n;
    const float* xp = x + static_cast<std::size_t>(r) * xv.rs() + static_cast<std::size_t>(g) * xv.gs();
    float* yp = y + static_cast<std::size_t>(r) * yv.rs() + static_cast<std::size_t>(g) * yv.gs();
    float xr[NREG];
    float p = 0.0f;
#pragma unroll
    for (int i = 0; i < NREG; ++i) {
        const std::uint32_t idx = threadIdx.x + 256u * i;
        xr[i] = idx < n ? xp[idx] : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < NREG; ++i) p = __fadd_rn(p, __fmul_rn(xr[i], xr[i]));
    const float ss = block_sum256(p, red);
    float inv;
    if constexpr (M == Mode::l2) {
        inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ss, eps)));
    } else {
        inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, static_cast<float>(n)), eps)));
    }
    const float* zp = nullptr;
    if constexpr (M == Mode::gated_swish || M == Mode::gated_sigmoid)
        zp = z + static_cast<std::size_t>(r) * zv.rs() + static_cast<std::size_t>(g) * zv.gs();
#pragma unroll
    for (int i = 0; i < NREG; ++i) {
        const std::uint32_t idx = threadIdx.x + 256u * i;
        if (idx < n) {
            float v = __fmul_rn(xr[i], inv);
            if constexpr (M == Mode::l2) {
                v = __fmul_rn(v, scale);
            } else {
                if (gain != nullptr) v = __fmul_rn(v, gain[idx]);
                if constexpr (M == Mode::gated_swish) v = __fmul_rn(v, cc_silu(zp[idx]));
                if constexpr (M == Mode::gated_sigmoid) v = __fmul_rn(v, cc_sigmoid(zp[idx]));
            }
            yp[idx] = v;
        }
    }
}

bool same_shape(const GroupedView& a, const GroupedView& b) { return a.rows == b.rows && a.groups == b.groups && a.n == b.n; }

template <Mode M>
cudaError_t run(const float* x, const GroupedView& xv, const float* gain, const float* z, const GroupedView& zv, float* y,
                const GroupedView& yv, float eps, float scale, cudaStream_t s) {
    if (!same_shape(xv, yv) || xv.n == 0 || xv.rows == 0 || xv.groups == 0) return cudaErrorInvalidValue;
    if (xv.groups > 65535u * 1u && false) return cudaErrorInvalidValue;
    const dim3 grid(xv.groups, xv.rows);
    const std::uint32_t nreg = (xv.n + 255) / 256;
    if (nreg <= 1) launch_pdl(norm_kernel<M, 1>, grid, dim3(256), s, x, xv, gain, z, zv, y, yv, eps, scale);
    else if (nreg <= 2) launch_pdl(norm_kernel<M, 2>, grid, dim3(256), s, x, xv, gain, z, zv, y, yv, eps, scale);
    else if (nreg <= 4) launch_pdl(norm_kernel<M, 4>, grid, dim3(256), s, x, xv, gain, z, zv, y, yv, eps, scale);
    else if (nreg <= 8) launch_pdl(norm_kernel<M, 8>, grid, dim3(256), s, x, xv, gain, z, zv, y, yv, eps, scale);
    else if (nreg <= 16) launch_pdl(norm_kernel<M, 16>, grid, dim3(256), s, x, xv, gain, z, zv, y, yv, eps, scale);
    else if (nreg <= 24) launch_pdl(norm_kernel<M, 24>, grid, dim3(256), s, x, xv, gain, z, zv, y, yv, eps, scale);
    else if (nreg <= 32) launch_pdl(norm_kernel<M, 32>, grid, dim3(256), s, x, xv, gain, z, zv, y, yv, eps, scale);
    else return cudaErrorInvalidValue;  // n <= 8192
    return cudaGetLastError();
}

__global__ void add_kernel(float* x, const float* d, std::uint32_t count) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = __fadd_rn(x[i], d[i]);
}

}  // namespace

cudaError_t rmsnorm(const float* x, const GroupedView& xv, const float* gain, float* y, const GroupedView& yv, float eps,
                    cudaStream_t stream) {
    return run<Mode::rms>(x, xv, gain, nullptr, GroupedView{}, y, yv, eps, 1.0f, stream);
}

cudaError_t gated_rmsnorm(const float* x, const GroupedView& xv, const float* gain, const float* z, const GroupedView& zv,
                          GateKind kind, float* y, const GroupedView& yv, float eps, cudaStream_t stream) {
    if (!same_shape(xv, zv)) return cudaErrorInvalidValue;
    return kind == GateKind::swish ? run<Mode::gated_swish>(x, xv, gain, z, zv, y, yv, eps, 1.0f, stream)
                                   : run<Mode::gated_sigmoid>(x, xv, gain, z, zv, y, yv, eps, 1.0f, stream);
}

cudaError_t l2norm(const float* x, const GroupedView& xv, float* y, const GroupedView& yv, float eps, float scale,
                   cudaStream_t stream) {
    return run<Mode::l2>(x, xv, nullptr, nullptr, GroupedView{}, y, yv, eps, scale, stream);
}

cudaError_t add_inplace(float* x, const float* d, std::uint32_t count, cudaStream_t stream) {
    add_kernel<<<(count + 255) / 256, 256, 0, stream>>>(x, d, count);
    return cudaGetLastError();
}

}  // namespace strix::cuda_core
