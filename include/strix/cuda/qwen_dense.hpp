// CUDA port of the Qwen dense A8 projections (numerical profile dense-a8-v1) and of the shared expert of the MLP half,
// bit-identical to the HIP tiled prefill (src/hip_q8_rows_gemv.hpp :1030-1640, src/hip_shared_expert_launcher.hip).
//
// Contract (HIP src/hip_q8_rows_gemv.hpp :1030-1046, notes/qwen-prefill-map-a8.md H1.1):
//   quantizer, row r, 32-feature block b: bad = any non-finite x; amax = max |x|; tiny = 0 < amax < 2^-119;
//     delta = bad ? NaN(0x7fc00000) : amax == 0 || tiny ? 1 : amax / 127 (correctly rounded);
//     code  = bad || tiny ? 0 : clamp(rint(x / delta), -127, 127) (half-even rint, correctly rounded division).
//   projection, row r, output n: acc = +0; for b ascending: acc = fma(float(d_w[n][b]) * delta[r][b], float(dot_b), acc);
//     y = isnan(acc) ? 0x7fc00000 : acc; then the site's store.
// Storage: codes int8 [row * width + feature]; deltas FP32 [block * pitch + row] (pitch >= rows).
// Weights: GGUF Q8_0 rows (blocks of 34 bytes: FP16 scale + 32 int8), row n of a segment at weight + n * width / 32 * 34.
// All pointers are device pointers; strides are in elements. codes must be 16-byte aligned, weights 4-byte aligned,
// width a multiple of 64. Every call is enqueued on `stream` (stream order, no atomics).
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "strix/cuda/amd_math.cuh"

namespace strix::cuda::qwen_dense {

constexpr std::uint32_t kMaxSegments = 5U;
constexpr std::uint32_t kCanonicalNan = 0x7fc00000U;

enum class Input : std::uint32_t { bf16 = 0, f32 = 1 };

// Site stores (notes/qwen-prefill-map-a8.md H1.3):
//   f32            shared gate/up (raw FP32, src/hip_shared_expert_launcher.hip:1503)
//   f32_round_bf16 GDN in (src/hip_gdn_launcher.hip:1063-1067), QSA in (src/hip_indexed_attention_launcher.hip:3494-3497)
//   bf16           GDN out (src/hip_gdn_launcher.hip:1081), QSA out (src/hip_indexed_attention_launcher.hip:3512)
//   bf16_gated     shared down: v_mul_f32(gate[r], y) then BF16 RNE (src/hip_shared_expert_launcher.hip:1519-1527, 177-189)
enum class Store : std::uint32_t { f32 = 0, f32_round_bf16 = 1, bf16 = 2, bf16_gated = 3 };

struct Segment {
    const void* weight = nullptr;   // Q8_0 [outputs][width / 32] blocks
    std::uint32_t outputs = 0;
    void* output = nullptr;         // float* (f32, f32_round_bf16) or std::uint16_t* (bf16, bf16_gated)
    std::uint64_t output_stride = 0;  // elements between rows
};

struct Site {
    Store store = Store::f32;
    std::uint32_t width = 0;
    std::uint32_t segments = 0;
    Segment segment[kMaxSegments]{};
    const float* gate = nullptr;    // bf16_gated: row r is multiplied by gate[r * gate_stride]
    std::uint64_t gate_stride = 0;
};

// Activation quantizer: rows x width (input row r at input + r * input_stride elements).
cudaError_t quantize(cudaStream_t stream, std::uint32_t rows, std::uint32_t width, Input type, const void* input,
                     std::uint64_t input_stride, std::int8_t* codes, float* deltas, std::uint32_t pitch);

// Projection of pre-quantized rows through the site's segments (outputs concatenated in segment order).
cudaError_t project(cudaStream_t stream, const Site& site, std::uint32_t rows, const std::int8_t* codes, const float* deltas,
                    std::uint32_t pitch);

#ifdef STRIX_WFMT_CAPTURE
// Diagnostic instantiation of the same arithmetic: [K/32][rows][sum outputs].
cudaError_t capture_project_partials(cudaStream_t stream, const Site& site, std::uint32_t rows,
                                    const std::int8_t* codes, const float* deltas, std::uint32_t pitch, float* partials);
#endif

// quantize + project; codes needs rows * site.width bytes, deltas (site.width / 32) * pitch floats.
cudaError_t quantize_project(cudaStream_t stream, const Site& site, std::uint32_t rows, Input type, const void* input,
                             std::uint64_t input_stride, std::int8_t* codes, float* deltas, std::uint32_t pitch);

// TURBO onfly weights (profile qwen-turbo-v1, --turbo-weights onfly): a Q8_0 weight converted once at load to the
// shared core's NVFP4 image (strix/cuda/core/nvfp4_a8.hpp, arithmetic nvfp4-a8-v1, same A8 activations). project()
// takes the NVFP4 route only inside an OnflyScope (the driver opens it for decode windows, every row count) and only
// when EVERY segment of the site is registered; otherwise the Q8_0 route runs unchanged. Register before the first
// decode, never concurrently with project(). Without registrations every route is the exact one, byte for byte.
struct OnflyImage {
    const std::uint8_t* codes = nullptr;   // [outputs][width / 2]
    const std::uint8_t* scales = nullptr;  // [outputs][width / 16]
    float tensor_scale = 1;
    std::uint32_t outputs = 0, width = 0;
};
void onfly_register(const void* q8_weight, const OnflyImage& image);
void onfly_clear();
std::size_t onfly_count();
// Tuning / test hook of the NVFP4 launch shape (rows per warp 1|2|4|8, outputs per warp 1|2; 0 = measured table).
// Shape changes parallelism only: every output is bitwise the same.
void onfly_set_launch(std::uint32_t rows_per_warp, std::uint32_t outputs_per_warp);
class OnflyScope {  // thread-local: the host thread that enqueues the decode window
public:
    explicit OnflyScope(bool on);
    ~OnflyScope();
    OnflyScope(const OnflyScope&) = delete;
    OnflyScope& operator=(const OnflyScope&) = delete;
private:
    bool previous_;
};

// Shared expert of the MLP half (HIP known_shared_prefill_enqueue, src/hip_shared_expert_launcher.hip:2071-2107).
struct SharedExpert {
    const void* gate_q8 = nullptr;   // ffn_gate_shexp  Q8_0 [intermediate][hidden]
    const void* up_q8 = nullptr;     // ffn_up_shexp    Q8_0 [intermediate][hidden]
    const void* down_q8 = nullptr;   // ffn_down_shexp  Q8_0 [hidden][intermediate]
    const float* scalar = nullptr;   // ffn_gate_inp_shexp F32 [hidden]
    std::uint32_t hidden = 2560U, intermediate = 640U;
};

// rows_shared_activation_scalar_kernel (src/hip_shared_expert_launcher.hip:535-562): per row, workspace[i] =
// (g / (1 + expf(-g))) * u for i < intermediate (g = workspace[i], u = workspace[intermediate + i]), then
// workspace[intermediate] = 1 / (1 + expf(-dot(input_bf16, scalar))) with the HIP 256-thread chain and LDS tree.
cudaError_t shared_activation(cudaStream_t stream, std::uint32_t rows, float* workspace, std::uint64_t workspace_stride,
                              const std::uint16_t* input, std::uint64_t input_stride, const float* scalar,
                              std::uint32_t intermediate, std::uint32_t hidden, const amdmath::Tables& tables);

// Whole shared expert: gate||up (A8, FP32 into workspace[0..2I)), activation + scalar gate, down (A8, gated BF16 into
// output). workspace_stride >= 2 * intermediate floats; codes rows * hidden bytes; deltas (hidden / 32) * pitch floats.
cudaError_t shared_expert(cudaStream_t stream, std::uint32_t rows, const SharedExpert& weights, const std::uint16_t* input,
                          std::uint64_t input_stride, float* workspace, std::uint64_t workspace_stride, std::uint16_t* output,
                          std::uint64_t output_stride, std::int8_t* codes, float* deltas, std::uint32_t pitch,
                          const amdmath::Tables& tables);

}  // namespace strix::cuda::qwen_dense
