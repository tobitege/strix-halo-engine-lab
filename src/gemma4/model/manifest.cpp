#include "model.hpp"
#include "gemma4/exl3/format.hpp"
#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <stdexcept>

namespace strix::gemma4 {
std::string hash_bytes(const void* data, std::size_t size) {
    return strix::model_content_identity_hex(strix::sha256_bytes_fast({static_cast<const std::byte*>(data), size}));
}
ModelConfig parse_text_config(const std::filesystem::path& path) {
    using namespace model_detail;
    const auto root = read_json(path);
    const auto& text = require(root, "text_config");
    const std::set<std::string> supported = {"model_type", "hidden_size", "num_hidden_layers", "intermediate_size", "vocab_size",
        "num_attention_heads", "num_key_value_heads", "head_dim", "global_head_dim", "hidden_size_per_layer_input",
        "vocab_size_per_layer_input", "max_position_embeddings", "sliding_window", "num_kv_shared_layers", "rms_norm_eps",
        "final_logit_softcapping", "hidden_activation", "tie_word_embeddings", "attention_bias", "attention_k_eq_v",
        "enable_moe_block", "use_double_wide_mlp", "attention_dropout", "use_bidirectional_attention", "num_experts",
        "top_k_experts", "expert_intermediate_size", "num_global_key_value_heads", "rope_scaling", "attention_scale",
        "rope_parameters", "layer_types", "bos_token_id", "eos_token_id", "pad_token_id", "dtype", "initializer_range", "use_cache"};
    for (const auto& [key, value] : text.as_object()) {
        (void)value;
        if (!supported.contains(key)) throw std::runtime_error("unsupported text config field: " + key);
    }
    auto equal = [&](const char* key, const serve::Json& expected) {
        const auto& v = require(text, key);
        if (v.is_number() && expected.is_number() ? v.as_double() != expected.as_double() : v.dump() != expected.dump())
            throw std::runtime_error(std::string("unsupported text config: ") + key);
    };
    equal("model_type", "gemma4_text"); equal("hidden_size", 2560); equal("num_hidden_layers", 42);
    equal("intermediate_size", 10240); equal("vocab_size", 262144); equal("num_attention_heads", 8);
    equal("num_key_value_heads", 2); equal("head_dim", 256); equal("global_head_dim", 512);
    equal("hidden_size_per_layer_input", 256); equal("vocab_size_per_layer_input", 262144);
    equal("max_position_embeddings", 65536); equal("sliding_window", 512); equal("num_kv_shared_layers", 18);
    equal("rms_norm_eps", 1e-6); equal("final_logit_softcapping", 30.0);
    equal("hidden_activation", "gelu_pytorch_tanh"); equal("tie_word_embeddings", true);
    equal("attention_bias", false); equal("attention_k_eq_v", false); equal("enable_moe_block", false);
    equal("use_double_wide_mlp", false); equal("attention_dropout", 0.0); equal("use_bidirectional_attention", nullptr);
    for (const auto* key : {"num_experts", "top_k_experts", "expert_intermediate_size", "num_global_key_value_heads"})
        if (const auto* p = text.find(key); p && !p->is_null()) throw std::runtime_error(std::string("unsupported override: ") + key);
    if (const auto* p = text.find("rope_scaling"); p && !p->is_null()) throw std::runtime_error("rope_scaling override refused");
    if (const auto* p = text.find("attention_scale"); p && p->as_double() != 1.0) throw std::runtime_error("attention scale override refused");
    const auto& rope = require(text, "rope_parameters");
    if (rope.as_object().size() != 2) throw std::runtime_error("unexpected rotary type");
    for (const auto& [kind, global] : std::initializer_list<std::pair<const char*, bool>>{{"full_attention", true}, {"sliding_attention", false}}) {
        const auto& r = require(rope, kind);
        if (require(r, "rope_theta").as_double() != (global ? 1000000.0 : 10000.0) ||
            require(r, "rope_type").as_string() != (global ? "proportional" : "default")) throw std::runtime_error("unsupported rotary parameters");
        for (const auto& [key, value] : r.as_object()) {
            if (key == "rope_theta" || key == "rope_type") continue;
            if (key == "factor" && value.as_double() == 1.0) continue;
            if (key == "partial_rotary_factor" && value.as_double() == (global ? 0.25 : 1.0)) continue;
            throw std::runtime_error("unsupported rotary override: " + key);
        }
        if (global && require(r, "partial_rotary_factor").as_double() != 0.25) throw std::runtime_error("global rotary proportion refused");
    }
    ModelConfig c;
    const auto& types = require(text, "layer_types").as_array();
    if (types.size() != layers) throw std::runtime_error("invalid layer types count");
    for (std::uint32_t l = 0; l < layers; ++l) {
        const bool global = l % 6 == 5;
        if (types[l].as_string() != (global ? "full_attention" : "sliding_attention")) throw std::runtime_error("unsupported layer schedule");
        c.layer.push_back({global ? AttentionKind::global : AttentionKind::sliding, global ? 512U : 256U, l < 24 ? l : (global ? 23U : 22U)});
    }
    return c;
}
std::map<std::string, std::vector<std::uint64_t>> required_tensors(const ModelConfig& c) {
    std::map<std::string, std::vector<std::uint64_t>> out;
    const std::string p = "model.language_model.";
    out[p + "embed_tokens.weight"] = {c.vocab, c.hidden};
    out[p + "norm.weight"] = {c.hidden};
    out[p + "embed_tokens_per_layer.weight"] = {c.vocab, std::uint64_t(c.layers) * c.ple_dim};
    out[p + "per_layer_model_projection.weight"] = {std::uint64_t(c.layers) * c.ple_dim, c.hidden};
    out[p + "per_layer_projection_norm.weight"] = {c.ple_dim};
    if (c.layer.size() != c.layers) throw std::runtime_error("incomplete layer configuration");
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const auto s = p + "layers." + std::to_string(l) + ".";
        const auto d = c.layer[l].head_dim;
        for (const auto* n : {"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm"}) out[s + n + ".weight"] = {c.hidden};
        out[s + "layer_scalar"] = {1};
        out[s + "self_attn.q_proj.weight"] = {std::uint64_t(c.heads) * d, c.hidden};
        out[s + "self_attn.o_proj.weight"] = {c.hidden, std::uint64_t(c.heads) * d};
        out[s + "self_attn.q_norm.weight"] = {d};
        if (c.layer[l].producer == l) {
            out[s + "self_attn.k_proj.weight"] = {std::uint64_t(c.kv_heads) * d, c.hidden};
            out[s + "self_attn.v_proj.weight"] = {std::uint64_t(c.kv_heads) * d, c.hidden};
            out[s + "self_attn.k_norm.weight"] = {d};
        }
        out[s + "mlp.gate_proj.weight"] = {c.inter, c.hidden};
        out[s + "mlp.up_proj.weight"] = {c.inter, c.hidden};
        out[s + "mlp.down_proj.weight"] = {c.hidden, c.inter};
        out[s + "per_layer_input_gate.weight"] = {c.ple_dim, c.hidden};
        out[s + "per_layer_projection.weight"] = {c.hidden, c.ple_dim};
    }
    return out;
}
bool frozen_compatibility_tensor(const std::string& name) {
    for (unsigned l = 24; l < 42; ++l)
        for (const auto* tail : {"k_proj.weight", "v_proj.weight", "k_norm.weight"})
            if (name == "model.language_model.layers." + std::to_string(l) + ".self_attn." + tail) return true;
    return false;
}
std::size_t packed_row_bytes(WeightFormat f, std::uint32_t inputs) {
    if (f == WeightFormat::exl3_mul1) throw std::runtime_error("EXL3 requires an explicit bitrate and tiled view");
    const std::size_t block = f == WeightFormat::q8_0 ? 32 : (f == WeightFormat::q4_k || f == WeightFormat::q6_k ? 256 : 1);
    const std::size_t bytes = f == WeightFormat::q8_0 ? 34 : (f == WeightFormat::q4_k ? 144 : (f == WeightFormat::q6_k ? 210 : (f == WeightFormat::f32 ? 4 : 2)));
    if (inputs == 0 || inputs % block) throw std::runtime_error("partial packed weight block");
    return inputs / block * bytes;
}
float weight_at(const LinearView& v, std::uint32_t row, std::uint32_t col) {
    const auto* p = v.data + std::size_t(row) * v.row_bytes;
    switch (v.format) {
    case WeightFormat::bf16: return bf16_value(load16(p + 2ULL * col));
    case WeightFormat::f32: { float x; std::memcpy(&x, p + 4ULL * col, 4); return x; }
    case WeightFormat::q8_0: return q8_0_value(p + (col / 32U) * 34ULL, col % 32U);
    case WeightFormat::q4_k: return q4_k_value(p + (col / 256U) * 144ULL, col % 256U);
    case WeightFormat::q6_k: return q6_k_value(p + (col / 256U) * 210ULL, col % 256U);
    case WeightFormat::exl3_mul1: throw std::runtime_error("EXL3 coefficients require the transformed linear backend");
    }
    throw std::runtime_error("invalid weight format");
}
void validate_finite(const LinearView& v) {
    if (v.format == WeightFormat::exl3_mul1) { exl3::validate(v); return; }
    // Quantized integer products are bounded; checking decoded values also checks all scales/minima.
    for (std::uint32_t r = 0; r < v.outputs; ++r)
        for (std::uint32_t i = 0; i < v.inputs; ++i)
            if (!std::isfinite(weight_at(v, r, i))) throw std::runtime_error("nonfinite model tensor");
}
namespace model_detail {
LinearView make_view(const std::uint8_t* p, WeightFormat f, const std::vector<std::uint64_t>& shape) {
    if (shape.empty() || shape.size() > 2 || shape.back() == 0 || shape.back() > UINT32_MAX || shape.front() > UINT32_MAX) throw std::runtime_error("unsupported text tensor rank/extent");
    const auto inputs = std::uint32_t(shape.back()), outputs = shape.size() == 1 ? 1U : std::uint32_t(shape.front());
    return {p, outputs, inputs, packed_row_bytes(f, inputs), f};
}
void validate_manifest(ModelManifest& m) {
    auto expected = required_tensors(m.config);
    std::set<std::string> seen;
    for (const auto& t : m.tensors) {
        const auto it = expected.find(t.name);
        if (!seen.insert(t.name).second || it == expected.end() || it->second != t.shape) throw std::runtime_error("unexpected/duplicate text tensor or shape: " + t.name);
        expected.erase(it);
    }
    if (!expected.empty()) throw std::runtime_error("missing mandatory tensor: " + expected.begin()->first);
}
}
const TensorRecord& tensor(const ModelManifest& m, const std::string& name) {
    const auto it = std::find_if(m.tensors.begin(), m.tensors.end(), [&](const auto& t) { return t.name == name; });
    if (it == m.tensors.end()) throw std::runtime_error("missing tensor: " + name);
    return *it;
}
ModelManifest load_safetensors_model(const std::filesystem::path& directory) {
    using namespace model_detail;
    ModelManifest m; m.config = parse_text_config(directory / "config.json");
    const auto index = read_json(directory / "model.safetensors.index.json");
    std::map<std::string, std::string> names;
    std::set<std::string> shards;
    for (const auto& [name, file] : require(index, "weight_map").as_object()) {
        const auto s = file.as_string();
        const std::filesystem::path rel(s);
        if (rel.empty() || rel.is_absolute() || rel.has_parent_path() || s.find(':') != std::string::npos || s.find('\\') != std::string::npos) throw std::runtime_error("unsafe safetensors shard name");
        names.emplace(name, s); shards.insert(s);
    }
    std::set<std::string> found;
    strix::Sha256Fast model_hash;
    const auto config_json = read_json(directory / "config.json").dump();
    model_hash.update({reinterpret_cast<const std::byte*>(config_json.data()), config_json.size()});
    std::size_t frozen = 0;
    for (const auto& shard : shards) {
        auto file = read_safetensors(directory / shard);
        auto namespace_status = weight::namespace_index(file.index,shard);
        if (!namespace_status) throw std::runtime_error(namespace_status.detail);
        weight::WeightMap admitted;
        auto status = weight::bind_gemma4(file.index, admitted);
        if (!status) throw std::runtime_error("gemma4 admission: " + status.detail);
        m.weights.insert(m.weights.end(), admitted.begin(), admitted.end());
        const auto digest = hash_bytes(file.storage->data(), file.storage->size());
        model_hash.update({reinterpret_cast<const std::byte*>(digest.data()), digest.size()});
        for (const auto& t : file.tensors) {
            const auto it = names.find(t.name);
            if (it == names.end() || it->second != shard || !found.insert(t.name).second) throw std::runtime_error("safetensors index/header mismatch: " + t.name);
            const bool text = t.name.starts_with("model.language_model.");
            if (frozen_compatibility_tensor(t.name)) {
                const auto start = std::string("model.language_model.layers.").size();
                const auto l = std::stoul(t.name.substr(start, t.name.find('.', start) - start));
                const auto d = m.config.layer[l].head_dim;
                const auto shape = t.name.ends_with("k_norm.weight") ? std::vector<std::uint64_t>{d} : std::vector<std::uint64_t>{std::uint64_t(m.config.kv_heads) * d, m.config.hidden};
                if (t.shape != shape || (t.dtype != "BF16" && t.dtype != "F32")) throw std::runtime_error("invalid frozen shared-KV compatibility tensor");
                validate_finite(make_view(file.storage->data() + t.offset, t.dtype == "BF16" ? WeightFormat::bf16 : WeightFormat::f32,t.shape));
                ++frozen; m.ignored_tensors.push_back(t.name + " (frozen shared-KV compatibility)"); continue;
            }
            if (!text && t.name != "lm_head.weight") { m.ignored_tensors.push_back(t.name + " (non-text)"); continue; }
            if (t.dtype != "BF16" && t.dtype != "F32") throw std::runtime_error("unsupported text weight dtype: " + t.name);
            auto view = make_view(file.storage->data() + t.offset, t.dtype == "BF16" ? WeightFormat::bf16 : WeightFormat::f32, t.shape);
            validate_finite(view);
            m.tensors.push_back({t.name, t.shape, view, digest});
        }
        m.storage.push_back(file.storage);
    }
    if (found.size() != names.size() || frozen != 54) throw std::runtime_error("incomplete index or frozen compatibility tensor set");
    const auto head = std::find_if(m.tensors.begin(), m.tensors.end(), [](const auto& t) { return t.name == "lm_head.weight"; });
    if (head != m.tensors.end()) {
        const auto* embedding = &tensor(m, "model.language_model.embed_tokens.weight");
        if (head->shape != embedding->shape) throw std::runtime_error("untied head shape");
        for (std::uint32_t r = 0; r < head->view.outputs; ++r)
            for (std::uint32_t i = 0; i < head->view.inputs; ++i)
                if (bit_cast<std::uint32_t>(weight_at(head->view, r, i)) != bit_cast<std::uint32_t>(weight_at(embedding->view, r, i))) throw std::runtime_error("untied head values");
        m.ignored_tensors.push_back("lm_head.weight (verified tied alias)");
        m.tensors.erase(head);
    }
    validate_manifest(m);
    m.model_hash = strix::model_content_identity_hex(model_hash.finalize());
    return m;
}
} // namespace strix::gemma4
