#pragma once
#include <cstddef>
#include <cstdint>

// Vendor-free operand and arithmetic contract. CUDA enqueue declarations live
// separately in nvfp4_cuda.hpp. No model layout or implicit allocation.
namespace strix::cuda::core::nvfp4 {
inline constexpr const char* kArithmeticId = "nvfp4-mma-f32-v1";
struct Matrix {
    const std::uint8_t* codes = nullptr;  // [rows][K/2], low nibble first
    const std::uint8_t* scales = nullptr; // [rows][K/16], positive E4M3FN
    std::size_t code_stride = 0, scale_stride = 0; // bytes
    float tensor_scale = 1;
};
// C[M,N] = A[M,K] * transpose(B[N,K]); K positive multiple of 64.
// FP4 operands/scales are exact inputs, NOT an implicit float activation cast.
// Native FP32 tensor accumulation is tolerance-qualified against the scalar
// reference; repeated launches of one plan are deterministic. Tensor scales
// multiply once after the reduction, then optional residual adds once.
struct View {
    Matrix a, b;
    float* output = nullptr;
    const float* residual = nullptr;
    std::size_t output_stride = 0, residual_stride = 0; // float elements
    std::uint32_t m = 0, n = 0, k = 0;
};
enum class Route : std::uint32_t { gemm, decode };
struct Plan {
    std::uint32_t m = 0, n = 0, k = 0, splits = 0;
    Route route = Route::gemm;
    std::size_t workspace_bytes = 0;
};
bool valid(const View&);
Plan resolve(const View&); // decode for M<=8, fixed split count, no device work
float e2m1(std::uint8_t);
float e4m3(std::uint8_t);
// Finite inputs only. RNE with ties to even, saturates to finite range.
std::uint8_t encode_e2m1(float);
std::uint8_t encode_e4m3(float);
// Caller selects positive finite tensor scale; zero blocks use scale zero.
// Storage is canonical row-major; all strides may be padded.
void quantize(const float*, std::size_t input_stride, std::uint32_t rows,
              std::uint32_t k, std::uint8_t*, std::size_t code_stride,
              std::uint8_t*, std::size_t scale_stride, float tensor_scale);
void reference(const View&, const Plan&);
}
