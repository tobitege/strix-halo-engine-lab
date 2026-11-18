#pragma once
// NVFP4 weights x A8 activations (arithmetic nvfp4-a8-v1): the TURBO onfly decode route of the shared CUDA core.
// Home: shared (backend core CUDA). Weights use the canonical operand layout of strix/cuda/core/nvfp4.hpp (#683):
// codes [N][K/2] low nibble first (E2M1), block scales [N][K/16] (positive E4M3FN), one FP32 tensor scale per tensor.
// Activations are the A8 operand an engine already produces for its Q8_0 route: int8 codes [row][K] and one FP32
// delta per (32-feature block b, row r) at deltas[b * pitch + r]. Only the weights change form; activations are the
// same bytes as the exact route, so the TURBO error is the weight rounding alone.
//
// Arithmetic contract nvfp4-a8-v1 (output n, row r; K multiple of 32):
//   w_int[k] = 2 * e2m1(code[n][k])                       exact int8 in {-12..12}
//   dot(g)   = sum_{j<16} a[r][16g+j] * w_int[16g+j]      exact int32 (16-group g)
//   lane L of a 32-lane chain owns the groups g = L, L + 32, L + 64, ... in ascending order:
//     acc_L = +0; acc_L = fma(e4m3(scale[n][g]) * delta[g/2][r], float(dot(g)), acc_L)   (product rounded, then fma)
//   total = xor butterfly over the 32 lanes: for off = 16, 8, 4, 2, 1: v[i] = v[i] + v[i ^ off] (commutative: all
//           lanes hold the same value; lane 0 is the result)
//   y = total * (0.5 * tensor_scale)  (0.5 exact), NaN -> the engine's canonical NaN in its epilogue.
// Every (n, r) value depends only on that row's activations and that output's weights: rows 1..R are bitwise equal to
// rows 1 (batch-invariant), and repeated launches are deterministic. reference() below is the host oracle.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strix::cuda::core::nvfp4_a8 {

inline constexpr const char* kArithmeticId = "nvfp4-a8-v1";
// Converter identity: per 16-group RTN of the source values with the #683 quantizer, tensor scale amax / (6 * 448).
inline constexpr const char* kConverterId = "nvfp4-rtn-g16-ts-amax";
inline constexpr std::uint32_t kGroup = 16, kBlock = 32, kLanes = 32;

struct Image {                       // one converted [N][K] tensor (device or host memory)
    std::uint8_t* codes = nullptr;   // [N][K/2]
    std::uint8_t* scales = nullptr;  // [N][K/16]
    float tensor_scale = 1;
    std::uint32_t n = 0, k = 0;
    [[nodiscard]] std::size_t code_bytes() const { return std::size_t{n} * (k / 2); }
    [[nodiscard]] std::size_t scale_bytes() const { return std::size_t{n} * (k / 16); }
};
[[nodiscard]] inline std::size_t image_bytes(std::uint32_t n, std::uint32_t k) { return std::size_t{n} * (k / 2 + k / 16); }

// Tensor scale rule of the converter (finite positive; 1 for an all-zero tensor).
float tensor_scale_for_amax(float amax);

// Host oracle of the converter from GGUF Q8_0 rows (34-byte blocks: FP16 delta + 32 int8): dequantise (fp16(d) * q,
// exact in FP32), then strix::cuda::core::nvfp4::quantize with the tensor scale above.
void reference_convert_q8_0(const std::uint8_t* q8, std::uint32_t n, std::uint32_t k, std::uint8_t* codes,
                            std::uint8_t* scales, float* tensor_scale);
// Host oracle of the GEMV contract for one output and one row (codes8 = that row's int8 activation codes [K],
// deltas_row[b] = delta of block b of that row).
float reference_dot(const std::uint8_t* codes, const std::uint8_t* scales, float tensor_scale, const std::int8_t* a,
                    const float* deltas_row, std::uint32_t k);

}  // namespace strix::cuda::core::nvfp4_a8
