#include "strix/qwen27_model.hpp"
#include "strix/ggml_dequant.hpp"
#include "strix/model_family_registry.hpp"
#include "model_graph/dense_adapter.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <set>
#include <stdexcept>

namespace strix::qwen27 {
namespace {
const std::vector<std::string> architectures = {"qwen3", "qwen38", "qwen3_8", "qwen3_8_dense"};
void validate_family(const Config& c) {
    model::validate(c);
    if (std::find(architectures.begin(), architectures.end(), c.model_type) == architectures.end())
        throw std::runtime_error("unsupported Qwen dense model_type/architecture: " + c.model_type);
}
std::string layer_prefix(std::uint32_t i) { return "model.layers." + std::to_string(i) + "."; }
void layer_requirements(std::map<std::string,std::vector<std::uint64_t>>& out, const Config& c, const std::string& p) {
    const std::uint64_t h = c.hidden, f = c.intermediate, q = std::uint64_t(c.heads) * c.head_dim, kv = std::uint64_t(c.kv_heads) * c.head_dim;
    out[p + "input_layernorm.weight"] = {h}; out[p + "post_attention_layernorm.weight"] = {h};
    for (const auto& [name, n] : std::vector<std::pair<std::string,std::uint64_t>>{{"q",q},{"k",kv},{"v",kv}}) {
        out[p + "self_attn." + name + "_proj.weight"] = {n,h};
        if (c.attention_bias) out[p + "self_attn." + name + "_proj.bias"] = {n};
    }
    out[p + "self_attn.o_proj.weight"] = {h,q};
    if (c.attention_bias) out[p + "self_attn.o_proj.bias"] = {h};
    if (c.qk_norm) {
        out[p + "self_attn.q_norm.weight"] = {c.head_dim};
        out[p + "self_attn.k_norm.weight"] = {c.head_dim};
    }
    out[p + "mlp.gate_proj.weight"] = {f,h}; out[p + "mlp.up_proj.weight"] = {f,h}; out[p + "mlp.down_proj.weight"] = {h,f};
    if (c.mlp_bias) { out[p + "mlp.gate_proj.bias"] = {f}; out[p + "mlp.up_proj.bias"] = {f}; out[p + "mlp.down_proj.bias"] = {h}; }
}
Model bind_model(Config c, const weights::Archive& archive) {
    Model model; model.config = std::move(c);
    std::set<std::string> used;
    for (const auto& [name, shape] : required_tensors(model.config)) {
        const auto it = archive.find(name);
        if (it == archive.end()) throw std::runtime_error("missing Qwen dense weight: " + name);
        Weight w; w.tensor = it->second; used.insert(name);
        if (w.tensor.dtype == "U8") {
            if (shape.size() != 2 || !name.ends_with(".weight")) throw std::runtime_error("packed NVFP4 requires a matrix weight: " + name);
            const auto base = name.substr(0,name.size()-7);
            w.nvfp4 = weights::nvfp4(archive,base,shape[0],shape[1]);
            used.insert(base + ".weight_scale"); used.insert(base + ".weight_scale_2");
            const auto aux = [&](const char* suffix, std::optional<weights::Tensor>& dst, bool vector) {
                const auto key = base + suffix;
                if (const auto p = archive.find(key); p != archive.end()) {
                    const auto& t = p->second;
                    if (t.dtype != "F32" && t.dtype != "F16" && t.dtype != "BF16") throw std::runtime_error("invalid NVFP4 input scale dtype: " + key);
                    const auto expected = vector ? std::vector<std::uint64_t>{shape[1]} : std::vector<std::uint64_t>{1};
                    if (t.shape != expected && !(t.shape.empty() && !vector)) throw std::runtime_error("invalid NVFP4 input scale shape: " + key);
                    auto scalar_view = t;
                    if (scalar_view.shape.empty()) scalar_view.shape = {1};
                    for (float f : weights::decode_row(scalar_view,0)) if (!std::isfinite(f) || f <= 0) throw std::runtime_error("invalid NVFP4 input scale");
                    dst = t; used.insert(key);
                }
            };
            aux(".pre_quant_scale", w.pre_quant_scale, true); aux(".input_scale",w.input_scale,false); aux(".input_scale_2",w.input_scale_2,false);
        } else {
            if (w.tensor.shape != shape) throw std::runtime_error("wrong Qwen dense weight shape: " + name);
            if (w.tensor.dtype == "GGUF") {
                if (!ggml_dequant::supported(w.tensor.ggml_type)) throw std::runtime_error("unsupported GGUF weight type: " + name);
                if (shape.back() % ggml_dequant::geometry(w.tensor.ggml_type)->block_elements)
                    throw std::runtime_error("GGUF row is not block-aligned: " + name);
            } else if (w.tensor.dtype != "F32" && w.tensor.dtype != "F16" && w.tensor.dtype != "BF16")
                throw std::runtime_error("unsupported scalar weight dtype: " + name);
        }
        model.tensors.emplace(name,std::move(w));
    }
    if (model.config.tied_embeddings) {
        if (const auto it = archive.find("lm_head.weight"); it != archive.end()) {
            const auto& embedding = model.tensors.at("model.embed_tokens.weight").tensor;
            if (it->second.shape != embedding.shape || it->second.dtype != embedding.dtype || it->second.ggml_type != embedding.ggml_type ||
                !std::ranges::equal(it->second.bytes(),embedding.bytes())) throw std::runtime_error("tied embedding/head payloads differ");
            used.insert("lm_head.weight");
        }
        model.tensors.emplace("lm_head.weight",model.tensors.at("model.embed_tokens.weight"));
    }
    for (const auto& [name, tensor] : archive) {
        (void)tensor;
        if (!used.contains(name)) throw std::runtime_error("unmapped Qwen dense tensor: " + name);
    }
    return model;
}
}
Config parse_hf_config(const std::filesystem::path& path) {
    const auto root = weights::read_json(path);
    const auto* nested = root.find("text_config"); const auto& j = nested ? *nested : root;
    // Dense adapter must not silently accept a MoE or hybrid attention config.
    for (const auto* key : {"num_experts","num_local_experts","n_routed_experts","num_experts_per_tok","linear_num_key_heads"})
        if (const auto* p = j.find(key); p && p->as_int() != 0) throw std::runtime_error("MoE config is not a dense decoder");
    if (const auto* p = j.find("layer_types")) for (const auto& type : p->as_array())
        if (type.as_string() != "full_attention") throw std::runtime_error("hybrid attention is not supported by dense family");
    const auto* use_sliding = j.find("use_sliding_window");
    if (const auto* p = j.find("sliding_window"); p && !p->is_null() && p->as_int() != 0 &&
        (!use_sliding || use_sliding->as_bool()))
        throw std::runtime_error("sliding attention requires a separate attention policy");
    auto c = model::dense_config_from_hf(root);
    if (const auto* p = j.find("layer_types"); p && p->size() != c.layers) throw std::runtime_error("layer_types length differs from trunk layer count");
    validate_family(c); return c;
}
Config parse_gguf_config(const gguf::FileInfo& info) {
    auto c = model::dense_config_from_gguf(info); validate_family(c);
    for (const auto* key : {"expert_count","attention.sliding_window"})
        if (const auto* p = info.find_metadata(c.model_type + "." + key); p &&
            (p->unsigned_value.value_or(0) || p->signed_value.value_or(0))) throw std::runtime_error("non-dense GGUF attention/FFN configuration");
    return c;
}
std::map<std::string,std::vector<std::uint64_t>> required_tensors(const Config& c) {
    validate_family(c);
    std::map<std::string,std::vector<std::uint64_t>> out = {{"model.embed_tokens.weight",{c.vocab,c.hidden}},{"model.norm.weight",{c.hidden}}};
    if (!c.tied_embeddings) out["lm_head.weight"] = {c.vocab,c.hidden};
    for (std::uint32_t i = 0; i < c.layers; ++i) layer_requirements(out,c,layer_prefix(i));
    for (std::uint32_t i = 0; i < c.mtp_layers; ++i) {
        const auto p = "mtp.layers." + std::to_string(i) + ".";
        layer_requirements(out,c,p); out[p + "eh_proj.weight"] = {c.hidden,std::uint64_t(c.hidden)*2};
        out[p + "enorm.weight"] = {c.hidden}; out[p + "hnorm.weight"] = {c.hidden};
    }
    return out;
}
std::string canonical_gguf_name(const std::string& name, const Config& c) {
    if (name == "token_embd.weight") return "model.embed_tokens.weight";
    if (name == "output_norm.weight") return "model.norm.weight";
    if (name == "output.weight") return "lm_head.weight";
    if (!name.starts_with("blk.")) throw std::runtime_error("unknown Qwen dense GGUF tensor: " + name);
    const auto dot = name.find('.',4);
    if (dot == std::string::npos) throw std::runtime_error("invalid GGUF layer name");
    std::uint32_t id = 0;
    const auto result = std::from_chars(name.data()+4,name.data()+dot,id);
    if (result.ec != std::errc{} || result.ptr != name.data()+dot || dot == 4 || id >= std::uint64_t(c.layers) + c.mtp_layers)
        throw std::runtime_error("GGUF layer index outside declared trunk/MTP");
    const auto tail = name.substr(dot+1);
    const auto suffix = tail.ends_with(".weight") ? ".weight" : tail.ends_with(".bias") ? ".bias" : "";
    if (!*suffix) throw std::runtime_error("unsupported GGUF tensor suffix");
    const auto stem = tail.substr(0,tail.size()-std::char_traits<char>::length(suffix));
    static const std::map<std::string,std::string> names = {{"attn_q","self_attn.q_proj"},{"attn_k","self_attn.k_proj"},
        {"attn_v","self_attn.v_proj"},{"attn_output","self_attn.o_proj"},{"attn_q_norm","self_attn.q_norm"},
        {"attn_k_norm","self_attn.k_norm"},{"attn_norm","input_layernorm"},{"ffn_norm","post_attention_layernorm"},
        {"ffn_gate","mlp.gate_proj"},{"ffn_up","mlp.up_proj"},{"ffn_down","mlp.down_proj"},
        {"nextn.eh_proj","eh_proj"},{"nextn.enorm","enorm"},{"nextn.hnorm","hnorm"}};
    const auto it = names.find(stem);
    if (it == names.end()) throw std::runtime_error("unmapped Qwen dense GGUF tensor: " + name);
    const auto p = id < c.layers ? layer_prefix(id) : "mtp.layers." + std::to_string(id-c.layers) + ".";
    return p + it->second + suffix;
}
Model load_gguf(const std::filesystem::path& path) {
    const auto info = gguf::read_file(path);
    auto c = parse_gguf_config(info);
    weights::Archive archive;
    for (const auto& [name, t] : weights::load_gguf(path,info))
        if (!archive.emplace(canonical_gguf_name(name,c),t).second) throw std::runtime_error("duplicate canonical GGUF weight");
    return bind_model(std::move(c),archive);
}
Model load_safetensors(const std::filesystem::path& directory) {
    auto c = parse_hf_config(directory / "config.json");
    return bind_model(std::move(c),weights::load_safetensors(directory));
}
std::vector<float> Weight::decode_row(std::uint64_t row) const { return nvfp4 ? nvfp4->decode_row(row) : weights::decode_row(tensor,row); }
const Weight& Model::weight(const std::string& name) const { return tensors.at(name); }
model_graph::ModelGraph graph(const Config& c) { validate_family(c); return model_graph::dense_decoder_graph(c,"qwen-dense","qwen-dense-v1"); }
void register_family() {
    ModelFamily family; family.name = "qwen27"; family.architectures = architectures;
    family.validate = [](const gguf::FileInfo& info) { (void)parse_gguf_config(info); };
    family.describe = [](const gguf::FileInfo& info) { const auto c = parse_gguf_config(info);
        return "qwen27: dense, " + std::to_string(c.layers) + " layers, hidden " + std::to_string(c.hidden) + ", host declaration"; };
    register_model_family(std::move(family));
}
} // namespace strix::qwen27
