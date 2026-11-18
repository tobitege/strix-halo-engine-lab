#include "strix/gguf_reader.hpp"
#include "strix/model_family_registry.hpp"
#include "strix/qwen35/config.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace strix::qwen35 {
namespace {

constexpr std::uint32_t kF32 = 0, kQ8_0 = 8, kBF16 = 30;
}  // namespace

Config validate(const gguf::FileInfo& info, std::uint32_t max_positions) {
    const Config config = config_from_gguf(info);
    if (max_positions) validate_backend(config, max_positions);
    std::unordered_map<std::string, const gguf::TensorInfo*> by_name;
    for (const auto& t : info.tensors)
        if (!by_name.emplace(t.name, &t).second) throw std::invalid_argument("qwen35: duplicate tensor: " + t.name);
    for (const auto& spec : expected_tensors(config)) {
        const auto it = by_name.find(spec.name);
        if (it == by_name.end()) throw std::invalid_argument("qwen35: required tensor missing: " + spec.name);
        const auto& t = *it->second;
        if (t.dimensions != spec.dimensions) throw std::invalid_argument("qwen35: unexpected shape for " + spec.name);
        const bool type_ok = spec.matrix ? (t.ggml_type == kBF16 || t.ggml_type == kQ8_0 || t.ggml_type == gguf::kNvfp4TilePack ||
                                       t.ggml_type == gguf::type_strix_iu4_g64) : t.ggml_type == kF32;
        if (!type_ok) throw std::invalid_argument("qwen35: unexpected tensor type " + std::to_string(t.ggml_type) + " for " + spec.name);
    }
    return config;
}

void register_family() {
    if (model_family_registered("qwen35")) return;
    ModelFamily family;
    family.name = "qwen35";
    family.architectures = {kArchitecture};
    family.validate = [](const gguf::FileInfo& info) { (void)validate(info); };
    family.describe = [](const gguf::FileInfo& info) {
        const auto c = validate(info);
        return "qwen35: " + std::to_string(c.layers) + " layers, hidden " + std::to_string(c.hidden) + ", " +
               std::to_string(c.gdn_value_heads) + " GDN value heads, " + std::to_string(c.heads) + "/" + std::to_string(c.kv_heads) +
               " attention heads, mtp " + std::to_string(c.mtp_layers) + ", " + std::to_string(info.tensors.size()) + " tensors";
    };
    register_model_family(std::move(family));
}

}  // namespace strix::qwen35
