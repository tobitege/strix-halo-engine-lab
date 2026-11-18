#pragma once
#include "strix/dense_model_config.hpp"
#include "strix/weight_archive.hpp"
#include <optional>

namespace strix::model_graph { struct ModelGraph; }
namespace strix::qwen27 {
using Config = model::DenseConfig;
struct Weight {
    weights::Tensor tensor;
    std::optional<weights::Nvfp4> nvfp4;
    // Optional input smoothing is exposed for the backend to execute, never silently discarded.
    std::optional<weights::Tensor> pre_quant_scale;
    std::optional<weights::Tensor> input_scale, input_scale_2;
    [[nodiscard]] std::vector<float> decode_row(std::uint64_t row) const;
};
struct Model {
    Config config;
    // Canonical HF names. All views own their mapped source, including tied head aliases.
    std::map<std::string, Weight> tensors;
    [[nodiscard]] const Weight& weight(const std::string& name) const;
};
[[nodiscard]] Config parse_hf_config(const std::filesystem::path& path);
[[nodiscard]] Config parse_gguf_config(const gguf::FileInfo& info);
[[nodiscard]] std::map<std::string,std::vector<std::uint64_t>> required_tensors(const Config& config);
[[nodiscard]] std::string canonical_gguf_name(const std::string& name, const Config& config);
[[nodiscard]] Model load_gguf(const std::filesystem::path& path);
[[nodiscard]] Model load_safetensors(const std::filesystem::path& directory);
[[nodiscard]] model_graph::ModelGraph graph(const Config& config);
// Explicit registration, as for other optional families. No production default or device initialization.
void register_family();
} // namespace strix::qwen27
