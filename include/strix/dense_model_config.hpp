#pragma once
#include "strix/gguf_reader.hpp"
#include "strix/serve/json.hpp"
#include <cstdint>
#include <string>

namespace strix::model {
struct RopeConfig {
    std::string type = "default";
    double theta = 1000000, factor = 1, attention_factor = 1;
    double beta_fast = 32, beta_slow = 1;
    std::uint32_t dimensions = 0, original_context = 0;
    bool operator==(const RopeConfig&) const = default;
};
// Backend-neutral dense decoder geometry; no hard-coded parameter-count preset.
struct DenseConfig {
    std::string model_type;
    std::uint32_t hidden = 0, layers = 0, heads = 0, kv_heads = 0;
    std::uint32_t intermediate = 0, head_dim = 0, vocab = 0, context = 0, mtp_layers = 0;
    double rms_epsilon = 1e-6;
    bool tied_embeddings = false, attention_bias = false, mlp_bias = false, qk_norm = true;
    std::string activation = "silu";
    RopeConfig rope;
};
void validate(const DenseConfig& config);
[[nodiscard]] DenseConfig dense_config_from_hf(const serve::Json& config);
[[nodiscard]] DenseConfig dense_config_from_gguf(const gguf::FileInfo& info);
} // namespace strix::model
