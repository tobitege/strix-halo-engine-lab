#pragma once
// Shared vision encoder runtime: one data-driven ViT tower for every family (pre-norm blocks, LayerNorm or RMSNorm,
// GELU / GELU-tanh / SwiGLU MLP, 2D RoPE and/or interpolated learned positions, optional DeepStack taps).
//
// Two backends run the same graph from the same EncoderWeights:
//   ReferenceEncoder  host FP32, slow, the fixture oracle (src/vision/reference.cpp);
//   Encoder           HIP, arithmetic identity "vision-f16-v1" (src/vision/encoder.hip): weights -> FP16 panels with a
//                     power-of-two tensor scale, FP16 operands, FP32 accumulation and residual, FP32
//                     norm/activation/softmax; optional hi+lo split activations. Class FAST, judged on its gates.
// A family supplies an adapter (adapter.hpp): tensor names, resize rule, patch order, projector and span layout.
// Nothing here runs for text-only requests.
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace strix::vision {

struct EncoderConfig {  // filled by the adapter from the checkpoint metadata
    enum class Norm { layernorm, rmsnorm };
    enum class Mlp { gelu, gelu_tanh, swiglu };
    enum class Pos { rope2d, learned_interp, rope2d_plus_learned };
    enum class RopePairing { half, adjacent };  // rotated pairs (i, i + D/2) or (2 i, 2 i + 1)

    unsigned layers = 0, width = 0, heads = 0, mlp_width = 0;
    Norm norm = Norm::layernorm;  // layernorm: weight + bias; rmsnorm: weight only
    float eps = 1e-6F;
    Mlp mlp = Mlp::gelu;
    float swiglu_limit = 0.0F;      // 0 = no clamp; else gate <= limit and up in [-limit, limit] before the product
    bool swiglu_gate_first = true;  // first half of the 2 * mlp_width outputs is the SiLU gate
    bool qkv_bias = false, out_bias = false, mlp_bias = false, patch_bias = true;
    Pos pos = Pos::rope2d;
    float rope_theta = 10000.0F;
    RopePairing rope_pairing = RopePairing::half;
    unsigned pos_grid = 0;  // learned table is pos_grid x pos_grid rows
    unsigned patch = 0, temporal = 1, channels = 3;
    std::vector<unsigned> deepstack_taps;  // 0-based block indices; the hidden state after each is exported
    bool final_norm = false;

    unsigned head_dim() const { return heads ? width / heads : 0U; }
    unsigned patch_dim() const { return channels * temporal * patch * patch; }
    unsigned mlp_out() const { return mlp == Mlp::swiglu ? 2U * mlp_width : mlp_width; }  // outputs of the first MLP linear
    bool has_rope() const { return pos != Pos::learned_interp; }
    bool has_learned() const { return pos != Pos::rope2d; }
};

// Non-owning row-major matrix [rows][cols] (a vector is rows x 1). BF16/F16 are the IEEE bit patterns.
struct Tensor {
    enum class Type { f32, bf16, f16 };
    const void* data = nullptr;
    Type type = Type::f32;
    std::size_t rows = 0, cols = 0;

    explicit operator bool() const { return data != nullptr; }
    std::size_t size() const { return rows * cols; }
    float at(std::size_t i) const;  // element i of the flattened matrix as FP32
};

struct BlockWeights {
    Tensor norm1_w, norm1_b;
    Tensor qkv_w, qkv_b;    // [3 width][width]: q, k, v
    Tensor out_w, out_b;    // [width][width]
    Tensor norm2_w, norm2_b;
    Tensor mlp1_w, mlp1_b;  // [mlp_out][width]
    Tensor mlp2_w, mlp2_b;  // [width][mlp_width]
};
struct EncoderWeights {
    Tensor patch_w, patch_b;  // [width][patch_dim] over the flattened patch
    Tensor pos_embed;         // [pos_grid^2][width]
    Tensor final_norm_w, final_norm_b;
    std::vector<BlockWeights> blocks;
    std::deque<std::vector<float>> owned;  // storage for tensors an adapter had to build (stable addresses)
};

// One image (or the tiles of one any-res image) as flattened patches in the adapter's row order.
struct PatchGrid {
    unsigned rows = 0, patch_dim = 0;
    unsigned grid_h = 0, grid_w = 0;  // patch grid of the resized image (interpolation of the learned table)
    std::vector<float> data;          // [rows][patch_dim]
    std::vector<std::uint16_t> y, x;  // patch coordinates of each row
    std::vector<unsigned> tile_end;   // exclusive row ends of the attention tiles; empty = one tile
};
// Per-row position inputs. Adapters may replace default_positions (upstream arithmetic quirks).
struct PositionInputs {
    std::vector<float> addend;  // [rows][width] added after the patch embedding; empty = none
    std::vector<float> angles;  // [rows][head_dim / 2] rotary angles in pair order; empty = none
};
struct EncoderOutput {
    unsigned rows = 0, width = 0;
    std::vector<float> hidden;             // [rows][width], after the final norm when the config has one
    std::vector<std::vector<float>> taps;  // one [rows][width] per deepstack tap, in tap order
};
// Projected rows of one image as the cache and the splice store them.
struct ProjectedRows {
    unsigned rows = 0, width = 0;
    std::vector<std::uint16_t> bf16;               // [rows][width]
    std::vector<std::vector<std::uint16_t>> deep;  // DeepStack rows, same shape, tap order
    std::size_t bytes() const;
};

void validate(const EncoderConfig&);                         // throws std::invalid_argument
void validate(const EncoderConfig&, const EncoderWeights&);  // config + tensor shapes
void validate(const EncoderConfig&, const PatchGrid&, const PositionInputs&);
// Learned table: bilinear, align-corners, FP32. Rotary: head_dim / 4 frequencies per axis, [row | column] angles.
PositionInputs default_positions(const EncoderConfig&, const EncoderWeights&, const PatchGrid&);

std::uint16_t to_bf16(float);  // round to nearest even; NaN stays NaN
float from_bf16(std::uint16_t);
std::vector<std::uint16_t> to_bf16_rows(const float* rows, std::size_t count);

// y[r][o] = sum_k x[r][k] * w[o][k] + bias[o] for row-major x [rows][w.cols], y [rows][w.rows]. Adapters run their
// projector linears through this so the reference and the device share one graph.
class DenseOps {
public:
    virtual ~DenseOps() = default;
    virtual void linear(const Tensor& w, const Tensor& bias, const float* x, std::size_t rows, float* y) = 0;
};
class EncoderBackend : public DenseOps {
public:
    virtual EncoderOutput encode(const PatchGrid&, const PositionInputs&) = 0;
    virtual const char* arithmetic() const = 0;  // part of the adapter identity
};

class ReferenceEncoder final : public EncoderBackend {
public:
    ReferenceEncoder(EncoderConfig, const EncoderWeights&);  // the weights outlive the encoder
    EncoderOutput encode(const PatchGrid&, const PositionInputs&) override;
    void linear(const Tensor& w, const Tensor& bias, const float* x, std::size_t rows, float* y) override;
    const char* arithmetic() const override { return "vision-ref-f32"; }

private:
    EncoderConfig cfg_;
    const EncoderWeights* weights_;
};
EncoderOutput reference_encode(const EncoderConfig&, const EncoderWeights&, const PatchGrid&, const PositionInputs&);

// Host copy of rows into an embedding buffer: dst[dst_row[i]][0 .. width) = fp32(src row src_row[i]).
void splice_rows_host(float* dst, std::size_t dst_stride, const ProjectedRows& src, const std::uint32_t* dst_row,
                      const std::uint32_t* src_row, std::size_t count);

// ---- HIP runner ------------------------------------------------------------------------------------------------

struct EncoderOptions {
    bool split_activations = false;                   // hi + lo FP16 operands for every linear input (two GEMMs)
    std::size_t mlp_workspace_bytes = 96U << 20U;     // cap of the chunked MLP buffers
    void* stream = nullptr;                           // hipStream_t; null = default stream
};
// Device bytes of one tower, exact for the runner: what a family reports to its memory budget before admission.
struct EncoderPlan {
    unsigned head_pad = 0;        // head dim as the attention core sees it (64 or 80)
    unsigned mlp_chunk_rows = 0;  // rows per MLP chunk
    std::size_t weight_bytes = 0, fixed_bytes = 0, mlp_bytes = 0;
    std::size_t workspace_bytes() const { return fixed_bytes + mlp_bytes; }
};
bool hip_supported(const EncoderConfig&, std::string* why = nullptr);  // shapes the shared cores take
EncoderPlan plan(const EncoderConfig&, unsigned rows, unsigned tiles, const EncoderOptions& = {});
bool resident_from_env();  // STRIX_VISION_RESIDENT, default 1

struct EncodeStats {
    bool per_layer = false;  // in: synchronise after every block and fill layer_ms
    double total_ms = 0.0;
    std::vector<double> layer_ms;
};

// Defined in the HIP library only (src/vision/encoder.hip).
class Encoder final : public EncoderBackend {
public:
    static constexpr const char* kArithmetic = "vision-f16-v1";
    explicit Encoder(EncoderConfig, EncoderOptions = {});
    ~Encoder() override;
    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;

    void load(const EncoderWeights&);  // FP16 panels to the device; lazy callers call it on the first image
    void unload();                     // drops panels and workspace
    bool loaded() const;
    std::size_t device_bytes() const;  // panels + workspace currently held

    EncoderOutput encode(const PatchGrid&, const PositionInputs&) override;
    EncoderOutput encode(const PatchGrid&, const PositionInputs&, EncodeStats&);
    // Projector linear on the shared GEMM; the panel of w is built on first use and kept until unload().
    void linear(const Tensor& w, const Tensor& bias, const float* x, std::size_t rows, float* y) override;
    const char* arithmetic() const override { return kArithmetic; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Device gather for the embed stage of a family: dst[dst_row[i]][0 .. width) = fp32(src[src_row[i]][0 .. width)).
// dst FP32 rows of dst_stride floats, src BF16 rows of width; all pointers are device pointers; stream = hipStream_t.
void splice_rows(float* dst, std::size_t dst_stride, const std::uint16_t* src, unsigned width, const std::uint32_t* dst_row,
                 const std::uint32_t* src_row, unsigned count, void* stream);

}  // namespace strix::vision
