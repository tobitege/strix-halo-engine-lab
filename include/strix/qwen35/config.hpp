#pragma once
// Qwen3.5-architecture dense text model (Qwen3.8-27B): the one host description every backend reads.
// Semantics and the GGUF contract ("qwen35") are documented in docs/models/qwen3.8-27b/README.md.
// Home: host shared (family layer). Backends (HIP src/qwen35/hip, CUDA src/cuda_core + apps) only consume it.

#include <cstdint>
#include <string>
#include <vector>

#include "strix/gguf_reader.hpp"

namespace strix::qwen35 {

inline constexpr const char* kArchitecture = "qwen35";

enum class LayerKind : std::uint8_t { gated_delta_net, full_attention };

struct Config {
    std::uint32_t layers = 64;
    std::uint32_t hidden = 5120;
    std::uint32_t intermediate = 17408;
    std::uint32_t vocab = 248320;
    std::uint32_t context = 262144;
    float rms_eps = 1e-6f;
    // full attention
    std::uint32_t heads = 24;
    std::uint32_t kv_heads = 4;
    std::uint32_t head_dim = 256;
    std::uint32_t rotary_dim = 64;      // NeoX rotate-half over the first rotary_dim dims of each head
    double rope_theta = 1.0e7;
    std::uint32_t full_attention_interval = 4;  // layer l is full attention iff (l + 1) % interval == 0
    // gated delta net
    std::uint32_t gdn_key_heads = 16;
    std::uint32_t gdn_key_dim = 128;
    std::uint32_t gdn_value_heads = 48;
    std::uint32_t gdn_value_dim = 128;
    std::uint32_t gdn_conv_kernel = 4;
    // multi-token prediction
    std::uint32_t mtp_layers = 1;

    [[nodiscard]] LayerKind kind(std::uint32_t layer) const noexcept {
        return (layer + 1) % full_attention_interval == 0 ? LayerKind::full_attention : LayerKind::gated_delta_net;
    }
    [[nodiscard]] std::uint32_t q_gate_width() const noexcept { return heads * head_dim * 2; }  // per head [q | gate]
    [[nodiscard]] std::uint32_t kv_width() const noexcept { return kv_heads * head_dim; }
    [[nodiscard]] std::uint32_t gdn_qk_width() const noexcept { return gdn_key_heads * gdn_key_dim; }
    [[nodiscard]] std::uint32_t gdn_v_width() const noexcept { return gdn_value_heads * gdn_value_dim; }
    [[nodiscard]] std::uint32_t gdn_conv_channels() const noexcept { return 2 * gdn_qk_width() + gdn_v_width(); }
};

// Reads and validates the qwen35.* metadata; throws std::invalid_argument naming the first bad key.
[[nodiscard]] Config config_from_gguf(const gguf::FileInfo& info);
// Full family admission, used by every backend before any device allocation.
[[nodiscard]] Config validate(const gguf::FileInfo& info, std::uint32_t max_positions = 0);
void validate_backend(const Config& config, std::uint32_t max_positions);

// One required tensor. dimensions are in GGUF (ggml) order: dimensions[0] is the fastest axis, i.e. the input
// features of a matrix. Matrices are BF16 or Q8_0 (the packs); everything else (norms, conv, ssm vectors) is F32.
struct TensorSpec {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    bool matrix = false;
};

// Every tensor the contract requires for `config` (trunk + MTP blk.<layers>), in plan order.
[[nodiscard]] std::vector<TensorSpec> expected_tensors(const Config& config);

// Names of expected_tensors, in the same order.
[[nodiscard]] std::vector<std::string> required_tensor_names(const Config& config);

// Registers the "qwen35" architecture with the model family registry (idempotent).
void register_family();

}  // namespace strix::qwen35
