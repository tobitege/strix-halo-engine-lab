#pragma once

// deepseek41 CUDA layer core as the layer assembly uses it: the device weights of layers [first, end) (tensor-core
// forms of every Q8_0 site, q_a + kv and the shared gate + up concatenated), the per-step buffers of R rows, and the
// core part of each plan op between the attention and routed-expert pieces:
//
//   embed                                         (first stage only)
//   per layer:  attn_input   [engram] hc_mix/attn, hc_pre/attn + rms_norm/attn (x FP32 + A8)
//               attn_q       q_a + kv (one site), rms_norm/q (qr FP32 + A8), q_b            -> q, kv rows, qr, x
//               <attention piece: window_write, compress, index, attention>                 -> o
//               attn_output  A8(o) + o_a, A8 + o_b (K halves h0 + h1), hc_post/attn
//               ffn_input    hc_mix/ffn, hc_pre/ffn + rms_norm/ffn (x FP32 + A8)
//               ffn_core     router (ids, weights), shared expert                          -> route ids/w, shared
//               <routed-expert piece: experts + combine (shared + low) + high>             -> y
//               ffn_output   hc_post/ffn (the next layer collapses with this site's pre)
//   head                                          (last stage only)
//
// Every call only enqueues on the stream. With a tap set, every op output is handed to it under the host forward's
// label ("op@layer/site:buffer", rows x count values with a row stride) after the op, for op hashes; the tap may
// synchronise and download. route_ids are passed as their 32-bit words; "argmax:token" passes the candidates
// (ArgmaxCandidate words, count 3).

#include "strix/cuda/ds41_common.hpp"
#include "strix/cuda/ds41_core.hpp"
#include "strix/cuda/ds41_weights.hpp"
#include "strix/ds41_layer.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strix::ds41::cuda {

struct LayerCoreGeometry {
    std::uint32_t hidden = 0, q_lora = 0, q_out = 0, head_dim = 0, o_groups = 0, o_lora = 0;
    std::uint32_t experts = 0, top_k = 0, inter = 0, vocab = 0, hc_iters = 0, engram_cols = 24;
    float hc_eps = 0.0F, route_scale = 0.0F, swiglu_limit = 0.0F;
    [[nodiscard]] static LayerCoreGeometry from(const deepseek41::Config& c);
};

struct LayerCoreWeights {
    std::uint32_t layer = 0;
    bool engram = false;
    const float *attn_norm = nullptr, *q_a_norm = nullptr, *ffn_norm = nullptr;
    const std::uint16_t *hc_attn_fn = nullptr, *hc_ffn_fn = nullptr;
    const float *hc_attn_base = nullptr, *hc_attn_scale = nullptr, *hc_ffn_base = nullptr, *hc_ffn_scale = nullptr;
    const float *router = nullptr, *bias = nullptr, *bias_image = nullptr;
    const std::uint16_t* engram_kv = nullptr;
    const float *engram_q_norm = nullptr, *engram_k_norm = nullptr;
    Q8TiledBuffer qa_kv, q_b, o_a, o_b, gate_up, down;
};

// Per-step device buffers of up to `rows` rows (row-major, strides in elements).
struct LayerCoreBuffers {
    std::uint32_t rows = 0;
    DeviceBuffer<float> stream, pre_carry, mix_attn, mix_ffn, x, xcol, qkv, qr, q, o, o_low, y, shared, route_w, scores, split_scratch;
    DeviceBuffer<std::uint32_t> tokens, route_ids;
    DeviceBuffer<std::uint8_t> engram_rows, workspace;
    A8Buffer xa, qa, oa, la;
    DeviceBuffer<ArgmaxCandidate> candidates;
    DeviceBuffer<float> logits;
    std::uint32_t kv_offset = 0, qkv_stride = 0;  // the kv rows inside qkv: qkv + kv_offset, row stride qkv_stride
};

using LayerCoreTap = std::function<void(const std::string& label, const float* device, std::uint32_t rows, std::uint32_t count,
                                        std::uint32_t stride)>;

class LayerCore {
public:
    // Loads the core tensors of layers [first, end), plus the embedding when embed and the head when head.
    LayerCore(TensorFile& file, std::uint32_t first, std::uint32_t end, bool embed, bool head, std::uint32_t max_rows,
              cudaStream_t stream);
    [[nodiscard]] const LayerCoreGeometry& geometry() const noexcept { return g_; }
    [[nodiscard]] LayerCoreBuffers& buffers() noexcept { return b_; }
    [[nodiscard]] const LayerCoreWeights& weights(std::uint32_t layer) const { return *layers_.at(layer - first_); }
    void set_tap(LayerCoreTap tap) { tap_ = std::move(tap); }
    void set_rows(std::uint32_t rows);
    void set_decode_fusions(bool hc, bool shared) noexcept { fuse_hc_ = hc; fuse_shared_ = shared; }

    void embed(const std::uint32_t* host_tokens, cudaStream_t s);                          // tokens -> stream, pre_carry
    void attn_input(std::uint32_t layer, const std::uint8_t* host_engram_rows, cudaStream_t s);  // rows [r][cols][264] or null
    void attn_q(std::uint32_t layer, cudaStream_t s);
    void attn_output(std::uint32_t layer, cudaStream_t s);  // reads buffers().o
    void ffn_input(std::uint32_t layer, cudaStream_t s);
    void ffn_core(std::uint32_t layer, cudaStream_t s);     // router + shared expert
    void ffn_output(std::uint32_t layer, cudaStream_t s);   // reads buffers().y
    void head(cudaStream_t s);                             // logits + candidates
    [[nodiscard]] Q8Tiled head_tiled() const noexcept { return has_head_ ? output_.view() : Q8Tiled{}; }  // output.weight, tiled

private:
    void tap(const std::string& label, const float* p, std::uint32_t count, std::uint32_t stride = 0) const;
    std::string label(const char* op, std::uint32_t layer, const char* site, const char* buffer) const;
    LayerCoreGeometry g_;
    std::uint32_t first_ = 0, end_ = 0, rows_ = 0, max_rows_ = 0;
    std::vector<std::unique_ptr<LayerCoreWeights>> layers_;
    const std::uint16_t* embed_table_ = nullptr;
    const float* output_norm_ = nullptr;
    Q8TiledBuffer output_;
    bool has_head_ = false;
    bool fuse_hc_ = false, fuse_shared_ = false;
    LayerCoreBuffers b_;
    LayerCoreTap tap_;
};

}  // namespace strix::ds41::cuda
