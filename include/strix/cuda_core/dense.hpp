#pragma once
#include "strix/cuda_core/types.hpp"

namespace strix::cuda_core {

// Dense arithmetic identity. fp32: the CUDA-core FP32 reference below (documented fixed order, rows 1..16 fast path).
// mma3 (default, "qwen35-cuda-exact-mma3-v1"): BF16 tensor-core mma.sync m16n8k16, FP32 accumulate, the FP32 activation split exactly
// into hi/mid/lo bf16 terms; FAST policies: mma2 (hi+mid), mma1 (hi only). One fixed K order for every row count 1..32 (see
// src/cuda_core/dense/dense_mma.cu); falls back to fp32 for shapes it does not cover (in % 32, out % 16, f32 weights).
enum class DenseArith : std::uint8_t { fp32, mma3, mma2, mma1 };
void set_dense_arith(DenseArith a);  // takes effect for launches / graph captures made afterwards
DenseArith dense_arith();

// y[r][o] = sum_k W[o][k] * x[r][k] (+ residual[r][o]) for r < rows (1..16). x is FP32 [rows][in], y FP32 [rows][out].
// FIXED ORDER (identical for every row count and launch shape): one warp per output o. bf16/f32: lane l owns chunks
// c = l + 32 i of 8 consecutive elements (i ascending) and adds the rounded products in ascending element order into one
// FP32 accumulator; q8_0: lane l owns blocks b = l + 32 i; per block s = sequential sum_j (float(q_j) * x_j), then
// acc = acc + fl(d * s). The 32 lane accumulators are combined by the xor butterfly (offsets 16,8,4,2,1).
// The optional residual is added after the dot is complete, so y = residual + W x is bitwise (W x) then add. y may alias
// residual. Requires in % 8 == 0 (bf16/f32) or in % 32 == 0 (q8_0).
cudaError_t dense_rows(const DenseWeight& w, const float* x, float* y, std::uint32_t rows, const float* residual,
                       cudaStream_t stream, DenseArith arithmetic = dense_arith());

// Up to 4 weights with the same x, format and in grouped by arithmetic route (ys[j] gets weights[j]'s output); bitwise equal to separate
// dense_rows calls (the per-output-row order does not depend on the launch composition). No residual.
cudaError_t dense_rows_multi(const DenseWeight* ws, float* const* ys, int n, const float* x, std::uint32_t rows,
                             const float* residual, cudaStream_t stream, DenseArith arithmetic = dense_arith());

// y[r][o] = silu(gate_dot) * up_dot with silu(g) = g / (1 + expf(-g)); bitwise equal to two dense_rows calls followed by the
// same epilogue. gate and up must share format/in/out.
cudaError_t dense_gate_up_silu_rows(const DenseWeight& gate, const DenseWeight& up, const float* x, float* y,
                                    std::uint32_t rows, cudaStream_t stream, DenseArith arithmetic = dense_arith());

// out[r][i] = dequant(row ids[r]) of a [vocab][in] table (embedding gather), FP32.
cudaError_t embedding_rows(const DenseWeight& table, const std::uint32_t* ids, float* out, std::uint32_t rows,
                           cudaStream_t stream);

// out[r] = argmax of logits row r (lowest index wins ties). workspace: kArgmaxWorkspaceBytes of device memory.
constexpr std::size_t kArgmaxWorkspaceBytes = 32 * 128 * 8;
cudaError_t argmax_rows(const float* logits, std::uint32_t vocab, std::uint32_t rows, std::uint32_t* out, void* workspace,
                        cudaStream_t stream);

}  // namespace strix::cuda_core
