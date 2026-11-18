#include "strix/dense_model_config.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strix::model {
namespace {
using serve::Json;
std::uint32_t integer(const Json& v, const std::string& key) {
    const auto n = v.as_int();
    if (n < 0 || static_cast<std::uint64_t>(n) > UINT32_MAX) throw std::runtime_error("invalid dimension: " + key);
    return static_cast<std::uint32_t>(n);
}
const Json& require(const Json& j, const char* key) {
    const auto* p = j.find(key);
    if (!p) throw std::runtime_error(std::string("missing config field: ") + key);
    return *p;
}
std::uint32_t number(const Json& j, const char* key, std::uint32_t fallback) {
    const auto* p = j.find(key); return p ? integer(*p, key) : fallback;
}
double real(const Json& j, const char* key, double fallback) {
    const auto* p = j.find(key); return p ? p->as_double() : fallback;
}
bool boolean(const Json& j, const char* key, bool fallback) {
    const auto* p = j.find(key); return p ? p->as_bool() : fallback;
}
std::uint32_t extent(const gguf::MetadataEntry* e, const std::string& key) {
    if (e && e->unsigned_value && *e->unsigned_value <= UINT32_MAX) return static_cast<std::uint32_t>(*e->unsigned_value);
    if (e && e->signed_value && *e->signed_value >= 0 && *e->signed_value <= UINT32_MAX)
        return static_cast<std::uint32_t>(*e->signed_value);
    throw std::runtime_error("missing/invalid GGUF dimension: " + key);
}
}
void validate(const DenseConfig& c) {
    if (!c.hidden || !c.layers || c.layers > 65536 || !c.heads || !c.kv_heads || c.heads % c.kv_heads ||
        !c.intermediate || !c.head_dim || !c.vocab || !c.context || c.mtp_layers > 64 ||
        !std::isfinite(c.rms_epsilon) || c.rms_epsilon <= 0 || c.activation != "silu")
        throw std::runtime_error("invalid/unsupported dense decoder geometry or activation");
    const auto& r = c.rope;
    if ((r.type != "default" && r.type != "linear" && r.type != "yarn") ||
        !r.dimensions || r.dimensions > c.head_dim || r.dimensions % 2 ||
        !std::isfinite(r.theta) || r.theta <= 0 || !std::isfinite(r.factor) || r.factor < 1 ||
        !std::isfinite(r.attention_factor) || r.attention_factor <= 0 ||
        !std::isfinite(r.beta_fast) || !std::isfinite(r.beta_slow) || r.beta_slow <= 0 || r.beta_fast < r.beta_slow ||
        (r.type == "yarn" && !r.original_context)) throw std::runtime_error("invalid/unsupported RoPE parameters");
}
DenseConfig dense_config_from_hf(const Json& root) {
    const auto* nested = root.find("text_config");
    const auto& j = nested ? *nested : root;
    DenseConfig c;
    c.model_type = require(j, "model_type").as_string();
    c.hidden = integer(require(j, "hidden_size"), "hidden_size");
    c.layers = integer(require(j, "num_hidden_layers"), "num_hidden_layers");
    c.heads = integer(require(j, "num_attention_heads"), "num_attention_heads");
    c.kv_heads = number(j, "num_key_value_heads", c.heads);
    c.intermediate = integer(require(j, "intermediate_size"), "intermediate_size");
    c.vocab = integer(require(j, "vocab_size"), "vocab_size");
    c.context = integer(require(j, "max_position_embeddings"), "max_position_embeddings");
    if (!j.find("head_dim") && (!c.heads || c.hidden % c.heads)) throw std::runtime_error("head_dim cannot be inferred");
    c.head_dim = number(j, "head_dim", c.heads ? c.hidden / c.heads : 0);
    c.rms_epsilon = real(j, "rms_norm_eps", c.rms_epsilon);
    c.tied_embeddings = boolean(j, "tie_word_embeddings", false);
    c.attention_bias = boolean(j, "attention_bias", false);
    c.mlp_bias = boolean(j, "mlp_bias", false);
    c.qk_norm = boolean(j, "qk_norm", true);
    if (const auto* p = j.find("hidden_act")) c.activation = p->as_string();
    c.mtp_layers = number(j, "num_nextn_predict_layers", number(j, "num_mtp_layers", 0));
    if (j.find("num_nextn_predict_layers") && j.find("num_mtp_layers") &&
        number(j, "num_mtp_layers", 0) != c.mtp_layers) throw std::runtime_error("conflicting MTP counts");
    c.rope.theta = real(j, "rope_theta", c.rope.theta);
    const auto* rope = j.find("rope_parameters");
    if (!rope || rope->is_null()) rope = j.find("rope_scaling");
    if (rope && !rope->is_null()) {
        if (j.find("rope_parameters") && !j.find("rope_parameters")->is_null() &&
            j.find("rope_scaling") && !j.find("rope_scaling")->is_null())
            throw std::runtime_error("ambiguous rope_parameters and rope_scaling");
        const auto* type = rope->find("rope_type");
        if (!type) type = rope->find("type");
        if (!type) throw std::runtime_error("missing rope_type");
        c.rope.type = type->as_string();
        if (const auto* old = rope->find("type"); old && old->as_string() != c.rope.type)
            throw std::runtime_error("conflicting RoPE type aliases");
        c.rope.theta = real(*rope, "rope_theta", c.rope.theta);
        c.rope.factor = real(*rope, "factor", 1);
        c.rope.original_context = number(*rope, "original_max_position_embeddings", 0);
        c.rope.attention_factor = real(*rope, "attention_factor", c.rope.type == "yarn" ? 1 + 0.1 * std::log(c.rope.factor) : 1);
        c.rope.beta_fast = real(*rope, "beta_fast", 32);
        c.rope.beta_slow = real(*rope, "beta_slow", 1);
    }
    const double partial = rope && !rope->is_null() ? real(*rope, "partial_rotary_factor", real(j, "partial_rotary_factor", 1)) : real(j, "partial_rotary_factor", 1);
    const double dims = partial * c.head_dim;
    if (!std::isfinite(dims) || dims < 0 || dims > UINT32_MAX || std::floor(dims) != dims) throw std::runtime_error("invalid rotary dimensions");
    c.rope.dimensions = static_cast<std::uint32_t>(dims);
    validate(c); return c;
}
DenseConfig dense_config_from_gguf(const gguf::FileInfo& info) {
    DenseConfig c;
    const auto* arch = info.find_metadata("general.architecture");
    if (!arch || !arch->string_value) throw std::runtime_error("missing GGUF architecture");
    c.model_type = *arch->string_value;
    const auto prefix = c.model_type + ".";
    const auto get = [&](const char* key) { return info.find_metadata(prefix + key); };
    const auto dim = [&](const char* key, std::uint32_t fallback) { const auto* p = get(key); return p ? extent(p, key) : fallback; };
    const auto scalar = [&](const char* key, double fallback) {
        const auto* p = get(key);
        if (!p) return fallback;
        if (p->floating_value) return *p->floating_value;
        if (p->unsigned_value) return static_cast<double>(*p->unsigned_value);
        if (p->signed_value) return static_cast<double>(*p->signed_value);
        throw std::runtime_error(std::string("invalid GGUF real: ") + key);
    };
    c.hidden = extent(get("embedding_length"), "embedding_length");
    c.layers = extent(get("block_count"), "block_count");
    c.heads = extent(get("attention.head_count"), "attention.head_count");
    c.kv_heads = dim("attention.head_count_kv", c.heads);
    c.intermediate = extent(get("feed_forward_length"), "feed_forward_length");
    c.context = extent(get("context_length"), "context_length");
    if (!get("attention.key_length") && (!c.heads || c.hidden % c.heads)) throw std::runtime_error("GGUF head_dim cannot be inferred");
    c.head_dim = dim("attention.key_length", c.heads ? c.hidden / c.heads : 0);
    if (dim("attention.value_length", c.head_dim) != c.head_dim) throw std::runtime_error("unequal key/value head dimensions");
    c.vocab = dim("vocab_size", 0);
    const auto* tokens = info.find_metadata("tokenizer.ggml.tokens");
    if (tokens && tokens->array_length) {
        if (*tokens->array_length > UINT32_MAX || (c.vocab && c.vocab != *tokens->array_length)) throw std::runtime_error("inconsistent GGUF vocabulary");
        c.vocab = static_cast<std::uint32_t>(*tokens->array_length);
    }
    bool output = false;
    for (const auto& t : info.tensors) {
        if (t.name == "output.weight") output = true;
        if (t.name == "token_embd.weight" && t.dimensions.size() == 2) {
            if (t.dimensions[1] > UINT32_MAX || (c.vocab && c.vocab != t.dimensions[1])) throw std::runtime_error("inconsistent embedding vocabulary");
            c.vocab = static_cast<std::uint32_t>(t.dimensions[1]);
        }
        if (t.name.ends_with(".bias")) {
            if (t.name.find(".attn_") != std::string::npos) c.attention_bias = true;
            if (t.name.find(".ffn_") != std::string::npos) c.mlp_bias = true;
        }
    }
    c.tied_embeddings = !output;
    c.rms_epsilon = scalar("attention.layer_norm_rms_epsilon", c.rms_epsilon);
    c.mtp_layers = dim("nextn_predict_layers", 0);
    c.rope.theta = scalar("rope.freq_base", c.rope.theta);
    c.rope.dimensions = dim("rope.dimension_count", c.head_dim);
    if (const auto* type = get("rope.scaling.type")) {
        if (!type->string_value) throw std::runtime_error("invalid GGUF rope type");
        c.rope.type = *type->string_value == "none" ? "default" : *type->string_value;
    }
    c.rope.factor = scalar("rope.scaling.factor", 1);
    c.rope.original_context = dim("rope.scaling.original_context_length", 0);
    c.rope.attention_factor = scalar("rope.scaling.attn_factor", c.rope.type == "yarn" ? 1 + 0.1 * std::log(c.rope.factor) : 1);
    c.rope.beta_fast = scalar("rope.scaling.beta_fast", 32);
    c.rope.beta_slow = scalar("rope.scaling.beta_slow", 1);
    validate(c); return c;
}
} // namespace strix::model
