#pragma once
// GLM prefill FAST identity glm-pf-router-wmma-v1 (switch: glm_pf_wmma.hpp router_wmma::enabled, opt-in): the router
// logits of a prefill call (288 x 4096 F16 weights; the exact route runs the R1 chains of dense_rows) on the shared
// dense FP16 WMMA core (core/f16_dense_tall.hpp launch_matrix, Guarded store). Only the product changes: D::route
// (sigmoid, bias, ties, selection, weights) runs unchanged on the logits. A row whose product is not finite (FP16
// overflow of x makes every sum of the row inf / NaN) is recomputed with the exact R1 chain by exact_rows_kernel.
// Adapter only: shape constants, the fallback chain (the family's exact identity) and the launches.
#include "core/f16_dense_tall.hpp"
#include "glm/prefill/glm_pf_wmma.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace strix::glm::pf::router_wmma {
inline constexpr unsigned kIn = 4096, kOut = 288, kFallbackSpan = 1024;

// R1: acc = +0; acc = fmaf(x[i], float(w[e][i]), acc), i ascending (= dev::dense = dense_rows), x staged in LDS.
template <int = 0>  // a template: the header may be included by several translation units
__global__ __launch_bounds__(256) void exact_rows_kernel(const float* x, const std::uint16_t* w, float* y, unsigned rows, unsigned* flags) {
    __shared__ unsigned list[256];
    __shared__ unsigned count;
    __shared__ float xs[kIn];
    core::f16_dense_tall::exact_rows(flags, rows, kFallbackSpan, list, &count, [&](unsigned r) __attribute__((always_inline)) {
        for (unsigned i = threadIdx.x; i < kIn; i += 256U) xs[i] = x[static_cast<std::size_t>(r) * kIn + i];
        __syncthreads();
        for (unsigned e = threadIdx.x; e < kOut; e += 256U) {
            const auto* we = reinterpret_cast<const uint4*>(w + static_cast<std::size_t>(e) * kIn);
            float acc = 0.0F;
#pragma unroll 4  // the weight loads issue ahead; the chain order is unchanged
            for (unsigned q = 0; q < kIn / 8U; ++q) {
                const uint4 v = we[q];
                const auto* h = reinterpret_cast<const std::uint16_t*>(&v);
#pragma unroll
                for (unsigned k = 0; k < 8U; ++k)
                    acc = __builtin_fmaf(xs[8U * q + k], static_cast<float>(__builtin_bit_cast(_Float16, h[k])), acc);
            }
            y[static_cast<std::size_t>(r) * kOut + e] = acc;
        }
    });
}

// GLM router: 18 column tiles of 4,096-long chains. Default narrow4: one launch, five 4-tile column blocks that share
// each x row in L2 (two dead tiles) instead of the 16 + 2 plan whose second launch reads x again.
inline core::f16_dense_tall::MatrixForm form(unsigned forced = ~0U) {
    if (forced <= 2U) return static_cast<core::f16_dense_tall::MatrixForm>(forced);
    return core::f16_dense_tall::matrix_form(core::f16_dense_tall::MatrixForm::narrow4);
}
inline bool supported(unsigned rows) { return rows != 0 && std::uint64_t{rows} * kIn * 4U < (std::uint64_t{1} << 32U); }

// logits y[r][e] (stride 288) of x[r][4096]; flags: rows uint32 of zeroed device memory (left zero).
inline void logits(const float* x, const std::uint16_t* w, float* y, unsigned rows, unsigned* flags, hipStream_t s, unsigned forced = ~0U) {
    namespace tall = core::f16_dense_tall;
    if (!supported(rows) || flags == nullptr) throw std::invalid_argument("glm pf router wmma: rows 1..262143 and flags required");
    const tall::Guarded<tall::RawStore> store{tall::RawStore{}, flags};
    const tall::MatrixForm f = form(forced);
    tall::launch_matrix(f, x, rows, reinterpret_cast<const _Float16*>(w), kIn, kOut, y, kOut, store, s);
    exact_rows_kernel<0><<<(rows + kFallbackSpan - 1U) / kFallbackSpan, 256, 0, s>>>(x, w, y, rows, flags);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("glm pf router wmma: ") + hipGetErrorString(e));
    calls.fetch_add(1, std::memory_order_relaxed);
    form_used.store(tall::matrix_form_name(f), std::memory_order_relaxed);
}

}  // namespace strix::glm::pf::router_wmma
