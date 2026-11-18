#pragma once
// Large-M dense GEMM (prefill chunks of up to a few thousand rows) on tensor cores.
// Home: shared (backend core CUDA; arch specialisation: mma.sync m16n8k16 bf16/f16, sm_80+; nvfp4 expansion needs sm_120a).
//
// EXACT CLASS: for every supported weight format the output of dense_big_gemm is BITWISE equal to dense_rows / the nvfp4 decode
// GEMV for the same row (rows 1..32, see dense.hpp / nvfp4.hpp), because every output element keeps the documented reduction
// order of the decode kernels: the same K segments (mma3 bf16: the 8 warp partials of src/cuda_core/dense/dense_mma.cu; q8_0 the
// same with the per-32-block fold tot += d * part; nvfp4: the split-K ranges of nvfp4_split_k(out, in)), the same k permutation
// inside a k16 step, the same instruction shape and the same term order, then the segment partials added in ascending order.
// Only the tiling (rows ride the N dimension, 128 out x 64..128 rows per CTA) and the operand staging differ. The activation
// is split ONCE per layer input into bf16 planes (hi, mid, lo; `terms` 3 = exact, 2/1 = FAST mma2/mma1) or one f16 plane
// (nvfp4), shared by every GEMM that reads the same x. Weights other than bf16 are expanded per call into a bf16/f16 scratch
// (q8_0: int8 values exact in bf16 + FP32 scales; nvfp4: e2m1*e4m3 exact in f16).
#include "strix/cuda_core/types.hpp"

namespace strix::cuda_core {

// Scratch (planes, expanded weights) of one stream context. nullptr in every call = the module default context (single stream).
// Concurrent streams need one context each; a context binds to its first device.
// The implicit context is isolated by host thread/device. Drain its stream before release.
[[nodiscard]] bool dense_big_fast8_supported();
struct BigCtx;
[[nodiscard]] BigCtx* dense_big_ctx_create();
void dense_big_ctx_destroy(BigCtx* ctx);

// Activation planes made by dense_big_split (device memory owned by the module; valid until the next split).
struct BigAct {
    const void* planes = nullptr;  // [terms][rows_pad][in] 16-bit (bf16 hi, mid, lo | f16 permuted)
    std::uint32_t rows = 0, rows_pad = 0, in = 0;
    int terms = 3;
    bool f16 = false;  // f16 plane (nvfp4 weights) instead of bf16 planes
    bool f8 = false;   // FAST/TURBO: e4m3 plane + row scales (nvfp4 weights only; dense_big_gemm then runs the e4m3 x e4m3 mma kernel)
};

enum class BigEpi : std::uint8_t {
    none,      // y = v
    residual,  // y = aux + v (aux may alias y)
    silu_mul,  // y = silu(aux) * v   (aux = the gate GEMM output of the same shape; gate/up as two GEMMs)
};

// true when the large-M path serves this weight (bf16 / q8_0 with in % 32 == 0, nvfp4 with in % 128 == 0, out % 16 == 0).
[[nodiscard]] bool dense_big_supported(const DenseWeight& w);
// Splits x (FP32 [rows][in]) into the planes weights of format `fmt` need. terms: 1..3 (bf16 and q8_0 only; nvfp4 ignores it).
// rows <= max rows given to dense_big_reserve. `in` % 32 == 0.
cudaError_t dense_big_split(const float* x, std::uint32_t rows, std::uint32_t in, WeightFormat fmt, int terms, BigAct* out,
                            cudaStream_t stream, BigCtx* ctx = nullptr, bool f8 = false);
// y[r][o] = epilogue(W x_r), r < a.rows. The planes must come from dense_big_split with the same format class as w.
cudaError_t dense_big_gemm(const DenseWeight& w, const BigAct& a, float* y, const float* aux, BigEpi epi, cudaStream_t stream,
                           BigCtx* ctx = nullptr);
// Grows the module's scratch (planes, expanded weights) once, outside any measured region. max_in / max_out_in as elements.
cudaError_t dense_big_reserve(std::uint32_t max_rows, std::uint32_t max_in, std::uint64_t max_weight_elems, int max_terms,
                              BigCtx* ctx = nullptr);
// Release the scratch.
void dense_big_release(BigCtx* ctx = nullptr);
// Diagnostics: bytes currently reserved.
[[nodiscard]] std::uint64_t dense_big_reserved_bytes(const BigCtx* ctx = nullptr);

}  // namespace strix::cuda_core
