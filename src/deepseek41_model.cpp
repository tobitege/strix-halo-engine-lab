#include "strix/deepseek41_model.hpp"

#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix::deepseek41 {
namespace {

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument("deepseek41: " + message); }
void require(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

// ---- strict JSON (objects, arrays, strings, numbers, true/false/null) ----

struct Json {
    enum class Kind : std::uint8_t { null, boolean, number, string, array, object } kind = Kind::null;
    bool boolean = false;
    double number = 0.0;
    bool integral = false;  // written without fraction or exponent
    std::int64_t integer = 0;
    std::string text;
    std::vector<Json> items;
    std::vector<std::pair<std::string, Json>> members;

    const Json* find(std::string_view key) const {
        for (const auto& [name, value] : members)
            if (name == key) return &value;
        return nullptr;
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string_view text) : text_(text) {}
    Json parse() {
        Json value = parse_value(0);
        skip_space();
        require(pos_ == text_.size(), "config JSON has trailing bytes");
        return value;
    }

private:
    void skip_space() {
        while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\n' || text_[pos_] == '\r' || text_[pos_] == '\t'))
            ++pos_;
    }
    char peek() {
        skip_space();
        require(pos_ < text_.size(), "config JSON ends early");
        return text_[pos_];
    }
    void expect(char c) {
        require(peek() == c, std::string("config JSON expected '") + c + "'");
        ++pos_;
    }
    bool literal(std::string_view word) {
        if (text_.substr(pos_, word.size()) != word) return false;
        pos_ += word.size();
        return true;
    }
    Json parse_value(int depth) {
        require(depth < 32, "config JSON nests too deeply");
        Json value;
        const char c = peek();
        if (c == '{') {
            value.kind = Json::Kind::object;
            ++pos_;
            std::set<std::string> seen;
            if (peek() == '}') { ++pos_; return value; }
            for (;;) {
                require(peek() == '"', "config JSON object key must be a string");
                std::string key = parse_string();
                require(seen.insert(key).second, "config JSON duplicate key " + key);
                expect(':');
                value.members.emplace_back(std::move(key), parse_value(depth + 1));
                if (peek() == ',') { ++pos_; continue; }
                expect('}');
                return value;
            }
        }
        if (c == '[') {
            value.kind = Json::Kind::array;
            ++pos_;
            if (peek() == ']') { ++pos_; return value; }
            for (;;) {
                value.items.push_back(parse_value(depth + 1));
                if (peek() == ',') { ++pos_; continue; }
                expect(']');
                return value;
            }
        }
        if (c == '"') {
            value.kind = Json::Kind::string;
            value.text = parse_string();
            return value;
        }
        if (literal("true")) { value.kind = Json::Kind::boolean; value.boolean = true; return value; }
        if (literal("false")) { value.kind = Json::Kind::boolean; return value; }
        if (literal("null")) return value;
        return parse_number();
    }
    std::string parse_string() {
        expect('"');
        std::string out;
        for (;;) {
            require(pos_ < text_.size(), "config JSON unterminated string");
            const char c = text_[pos_++];
            if (c == '"') return out;
            require(static_cast<unsigned char>(c) >= 0x20U, "config JSON control character in string");
            if (c != '\\') { out.push_back(c); continue; }
            require(pos_ < text_.size(), "config JSON bad escape");
            const char e = text_[pos_++];
            switch (e) {
            case '"': case '\\': case '/': out.push_back(e); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                require(pos_ + 4 <= text_.size(), "config JSON bad \\u escape");
                unsigned code = 0;
                const auto r = std::from_chars(text_.data() + pos_, text_.data() + pos_ + 4, code, 16);
                require(r.ec == std::errc{} && r.ptr == text_.data() + pos_ + 4, "config JSON bad \\u escape");
                pos_ += 4;
                // Keys and values used here are ASCII; keep other code points as UTF-8 (BMP only).
                if (code < 0x80U) {
                    out.push_back(static_cast<char>(code));
                } else if (code < 0x800U) {
                    out.push_back(static_cast<char>(0xC0U | (code >> 6)));
                    out.push_back(static_cast<char>(0x80U | (code & 0x3FU)));
                } else {
                    out.push_back(static_cast<char>(0xE0U | (code >> 12)));
                    out.push_back(static_cast<char>(0x80U | ((code >> 6) & 0x3FU)));
                    out.push_back(static_cast<char>(0x80U | (code & 0x3FU)));
                }
                break;
            }
            default: fail("config JSON bad escape");
            }
        }
    }
    Json parse_number() {
        const std::size_t begin = pos_;
        if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
        bool digits = false, integral = true;
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c >= '0' && c <= '9') { digits = true; ++pos_; continue; }
            if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') { integral = false; ++pos_; continue; }
            break;
        }
        require(digits, "config JSON invalid value");
        const char* first = text_.data() + begin;
        const char* last = text_.data() + pos_;
        Json value;
        value.kind = Json::Kind::number;
        const auto r = std::from_chars(first, last, value.number);
        require(r.ec == std::errc{} && r.ptr == last && std::isfinite(value.number), "config JSON invalid number");
        if (integral) {
            const auto ri = std::from_chars(first, last, value.integer);
            require(ri.ec == std::errc{} && ri.ptr == last, "config JSON integer out of range");
            value.integral = true;
        }
        return value;
    }

    std::string_view text_;
    std::size_t pos_ = 0;
};

const Json& member(const Json& object, std::string_view key) {
    require(object.kind == Json::Kind::object, "config JSON expected an object");
    const auto* value = object.find(key);
    require(value != nullptr, "config JSON missing key " + std::string(key));
    return *value;
}
std::uint64_t as_u64(const Json& value, std::string_view key) {
    require(value.kind == Json::Kind::number && value.integral && value.integer >= 0,
            "config " + std::string(key) + " must be a non-negative integer");
    return static_cast<std::uint64_t>(value.integer);
}
std::uint32_t as_u32(const Json& value, std::string_view key) {
    const auto v = as_u64(value, key);
    require(v <= std::numeric_limits<std::uint32_t>::max(), "config " + std::string(key) + " out of range");
    return static_cast<std::uint32_t>(v);
}
double as_f64(const Json& value, std::string_view key) {
    require(value.kind == Json::Kind::number, "config " + std::string(key) + " must be a number");
    return value.number;
}
std::string as_text(const Json& value, std::string_view key) {
    require(value.kind == Json::Kind::string, "config " + std::string(key) + " must be a string");
    return value.text;
}
bool as_bool(const Json& value, std::string_view key) {
    require(value.kind == Json::Kind::boolean, "config " + std::string(key) + " must be a boolean");
    return value.boolean;
}
template <typename T>
std::vector<T> as_vector(const Json& value, std::string_view key) {
    require(value.kind == Json::Kind::array, "config " + std::string(key) + " must be an array");
    std::vector<T> out;
    for (const auto& item : value.items) out.push_back(static_cast<T>(sizeof(T) == 4 ? as_u32(item, key) : as_u64(item, key)));
    return out;
}

// ---- metadata access ----

const gguf::MetadataEntry& meta(const gguf::FileInfo& info, const std::string& key) {
    const auto* entry = info.find_metadata(key);
    require(entry != nullptr, "missing metadata " + key);
    return *entry;
}
std::uint64_t meta_u64(const gguf::FileInfo& info, const std::string& key) {
    const auto& e = meta(info, key);
    require(e.unsigned_value.has_value(), "metadata " + key + " must be unsigned");
    return *e.unsigned_value;
}
double meta_f64(const gguf::FileInfo& info, const std::string& key) {
    const auto& e = meta(info, key);
    require(e.floating_value.has_value(), "metadata " + key + " must be a float");
    return *e.floating_value;
}
std::string meta_text(const gguf::FileInfo& info, const std::string& key) {
    const auto& e = meta(info, key);
    require(e.string_value.has_value(), "metadata " + key + " must be a string");
    return *e.string_value;
}
bool meta_bool(const gguf::FileInfo& info, const std::string& key) {
    const auto& e = meta(info, key);
    require(e.boolean_value.has_value(), "metadata " + key + " must be a boolean");
    return *e.boolean_value;
}
std::vector<std::uint64_t> meta_array(const gguf::FileInfo& info, const std::string& key) {
    const auto& e = meta(info, key);
    require(e.array_length.has_value() && e.signed_array.empty() && e.floating_array.empty() &&
                e.unsigned_array.size() == *e.array_length,
            "metadata " + key + " must be a retained unsigned array");
    return e.unsigned_array;
}
template <typename A, typename B>
void same_list(const std::vector<A>& a, const std::vector<B>& b, const std::string& what) {
    bool equal = a.size() == b.size();
    for (std::size_t i = 0; equal && i < a.size(); ++i) equal = static_cast<std::uint64_t>(a[i]) == static_cast<std::uint64_t>(b[i]);
    require(equal, what + " differs between metadata and config");
}
// GGUF stores float32; the config writes the decimal it was rounded from.
void same_float(double gguf_value, double config_value, const std::string& what) {
    require(static_cast<float>(gguf_value) == static_cast<float>(config_value), what + " differs between metadata and config");
}

bool contains(const std::vector<std::uint32_t>& list, std::uint32_t value) {
    return std::find(list.begin(), list.end(), value) != list.end();
}

// ---- Engram primes: the n-th unused prime above engram_vocab_size - 1, per (layer, n-gram size, head) ----

// Trial division: the bucket moduli are close to engram_vocab_size (about 1.6e7 here).
bool is_prime(std::uint64_t n) {
    require(n < (1ULL << 40U), "Engram bucket modulus too large to test");
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (std::uint64_t d = 3; d * d <= n; d += 2)
        if (n % d == 0) return false;
    return true;
}

EngramContract engram_contract(const gguf::FileInfo& info, const Config& c) {
    EngramContract e;
    e.encoding = meta_text(info, "deepseek41.engram.encoding");
    require(e.encoding == "e4m3_e8m0_32_row264", "unsupported Engram encoding " + e.encoding);
    require(c.engram_head_dim % 32U == 0, "Engram head_dim must be a multiple of 32");
    e.row_bytes = c.engram_head_dim + c.engram_head_dim / 32U;
    require(e.row_bytes == 264U, "Engram row bytes do not match the encoding");
    for (const auto v : meta_array(info, "deepseek41.engram.layer_ids")) e.layer_ids.push_back(static_cast<std::uint32_t>(v));
    e.rows = meta_array(info, "deepseek41.engram.rows");
    e.primes = meta_array(info, "deepseek41.engram.primes");
    e.multipliers = meta_array(info, "deepseek41.engram.multipliers");
    for (const auto v : meta_array(info, "deepseek41.engram.token_map")) {
        require(v <= std::numeric_limits<std::uint32_t>::max(), "Engram token map value out of range");
        e.token_map.push_back(static_cast<std::uint32_t>(v));
    }
    e.compressed_vocab_size = static_cast<std::uint32_t>(meta_u64(info, "deepseek41.engram.compressed_vocab_size"));
    e.pad_id = static_cast<std::uint32_t>(meta_u64(info, "deepseek41.engram.pad_id"));
    e.max_ngram_size = c.engram_max_ngram_size;
    e.n_heads = c.engram_n_heads;

    same_list(e.layer_ids, c.engram_layer_ids, "Engram layer ids");
    same_list(e.rows, c.engram_num_embeddings, "Engram table rows");
    require(e.compressed_vocab_size == c.engram_compressed_vocab_size, "Engram compressed vocabulary size differs");
    require(e.pad_id == c.engram_pad_token_id, "Engram pad id differs");
    require(e.max_ngram_size >= 2 && e.n_heads >= 1, "Engram n-gram contract is empty");
    const std::size_t layers = e.layer_ids.size(), per_layer = e.rows_per_token();
    require(e.primes.size() == layers * per_layer, "Engram prime count does not match layers x n-grams x heads");
    require(e.multipliers.size() == layers * e.max_ngram_size, "Engram multiplier count does not match layers x lookback");

    // Bucket moduli: drawn in order, never reused, each the next prime above engram_vocab_size - 1.
    std::set<std::uint64_t> seen;
    std::size_t index = 0;
    for (std::size_t layer = 0; layer < layers; ++layer) {
        std::uint64_t sum = 0;
        for (std::uint32_t n = 0; n + 1 < e.max_ngram_size; ++n) {
            std::uint64_t current = c.engram_vocab_size - 1U;
            for (std::uint32_t h = 0; h < e.n_heads; ++h, ++index) {
                std::uint64_t candidate = current + 1U;
                while (!is_prime(candidate) || seen.count(candidate) != 0) ++candidate;
                require(e.primes[index] == candidate, "Engram prime does not follow the bucket layout rule");
                seen.insert(candidate);
                current = candidate;
                sum += candidate;
            }
        }
        require(sum == e.rows[layer], "Engram table rows differ from the sum of the layer's bucket moduli");
    }
    // token * multiplier must not overflow int64 (the hash is a signed 64-bit XOR of products).
    const std::uint64_t bound = std::max<std::uint64_t>(
        1, (static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) / e.compressed_vocab_size) / 2U);
    for (const auto m : e.multipliers)
        require((m & 1U) == 1U && m < 2U * bound + 1U, "Engram multiplier must be odd and bounded");
    require(e.token_map.size() == c.vocab_size, "Engram token map must cover the vocabulary");
    std::vector<bool> used(e.compressed_vocab_size, false);
    for (const auto v : e.token_map) {
        require(v < e.compressed_vocab_size, "Engram token map value exceeds the compressed vocabulary");
        used[v] = true;
    }
    require(std::all_of(used.begin(), used.end(), [](bool u) { return u; }), "Engram compressed vocabulary has unused ids");
    require(e.pad_id < c.vocab_size, "Engram pad id outside the vocabulary");
    return e;
}

void cross_check(const gguf::FileInfo& info, const Config& c) {
    const auto u = [&](const char* key, std::uint64_t value) {
        require(meta_u64(info, std::string("deepseek41.") + key) == value, std::string(key) + " differs between metadata and config");
    };
    u("vocab_size", c.vocab_size);
    u("hidden_size", c.hidden_size);
    u("moe_intermediate_size", c.moe_intermediate_size);
    u("num_hidden_layers", c.num_hidden_layers);
    u("num_attention_heads", c.num_attention_heads);
    u("num_key_value_heads", c.num_key_value_heads);
    u("head_dim", c.head_dim);
    u("qk_rope_head_dim", c.qk_rope_head_dim);
    u("q_lora_rank", c.q_lora_rank);
    u("o_lora_rank", c.o_lora_rank);
    u("o_groups", c.o_groups);
    u("n_routed_experts", c.n_routed_experts);
    u("n_shared_experts", c.n_shared_experts);
    u("num_experts_per_tok", c.num_experts_per_tok);
    u("max_position_embeddings", c.max_position_embeddings);
    u("sliding_window", c.sliding_window);
    u("index_n_heads", c.index_n_heads);
    u("index_head_dim", c.index_head_dim);
    u("index_topk", c.index_topk);
    u("candidate_source_layer_id", c.candidate_source_layer_id);
    u("candidate_topk_blocks", c.candidate_topk_blocks);
    u("candidate_block_size", c.candidate_block_size);
    u("hc_mult", c.hc_mult);
    u("hc_sinkhorn_iters", c.hc_sinkhorn_iters);
    u("rope_theta", static_cast<std::uint64_t>(c.rope_theta));
    u("compress_rope_theta", static_cast<std::uint64_t>(c.compress_rope_theta));
    require(static_cast<double>(static_cast<std::uint64_t>(c.rope_theta)) == c.rope_theta &&
                static_cast<double>(static_cast<std::uint64_t>(c.compress_rope_theta)) == c.compress_rope_theta,
            "rope thetas must be integers");
    same_float(meta_f64(info, "deepseek41.rms_norm_eps"), c.rms_norm_eps, "rms_norm_eps");
    same_float(meta_f64(info, "deepseek41.hc_eps"), c.hc_eps, "hc_eps");
    same_float(meta_f64(info, "deepseek41.swiglu_limit"), c.swiglu_limit, "swiglu_limit");
    same_float(meta_f64(info, "deepseek41.routed_scaling_factor"), c.routed_scaling_factor, "routed_scaling_factor");
    same_float(meta_f64(info, "deepseek41.rope_scaling.factor"), c.rope_scaling.factor, "rope_scaling.factor");
    same_float(meta_f64(info, "deepseek41.rope_scaling.beta_fast"), c.rope_scaling.beta_fast, "rope_scaling.beta_fast");
    same_float(meta_f64(info, "deepseek41.rope_scaling.beta_slow"), c.rope_scaling.beta_slow, "rope_scaling.beta_slow");
    same_float(meta_f64(info, "deepseek41.rope_scaling.original_max_position_embeddings"),
               c.rope_scaling.original_max_position_embeddings, "rope_scaling.original_max_position_embeddings");
    require(meta_text(info, "deepseek41.scoring_func") == c.scoring_func, "scoring_func differs between metadata and config");
    require(meta_text(info, "deepseek41.hidden_act") == c.hidden_act, "hidden_act differs between metadata and config");
    require(meta_text(info, "deepseek41.topk_method") == c.topk_method, "topk_method differs between metadata and config");
    require(meta_bool(info, "deepseek41.norm_topk_prob") == c.norm_topk_prob, "norm_topk_prob differs between metadata and config");
    // The metadata list covers the target layers; the config appends one entry per draft stage.
    const auto ratios = meta_array(info, "deepseek41.compress_ratios");
    require(ratios.size() == c.num_hidden_layers, "compress_ratios metadata must have one entry per target layer");
    same_list(ratios, std::vector<std::uint32_t>(c.compress_ratios.begin(), c.compress_ratios.begin() + static_cast<std::ptrdiff_t>(c.num_hidden_layers)),
              "compress_ratios");
    same_list(meta_array(info, "deepseek41.kv_source_layer_ids"), c.kv_source_layer_ids, "kv_source_layer_ids");
    same_list(meta_array(info, "deepseek41.index_source_layer_ids"), c.index_source_layer_ids, "index_source_layer_ids");
    require(meta_u64(info, "tokenizer.ggml.bos_token_id") == c.bos_token_id, "bos token differs");
    require(meta_u64(info, "tokenizer.ggml.eos_token_id") == c.eos_token_id, "eos token differs");
    require(meta(info, "tokenizer.ggml.tokens").array_length.value_or(0) == c.vocab_size, "tokenizer size differs from vocab_size");
}

void add(std::vector<ExpectedTensor>& out, std::string name, std::vector<std::uint64_t> dims, std::uint32_t type, TensorRole role,
         std::int32_t layer, ResidencyPolicy residency = ResidencyPolicy::resident) {
    out.push_back({std::move(name), std::move(dims), type, role, residency, layer});
}

constexpr std::uint32_t F32 = 0, F16 = 1, Q8_0 = 8, Q2_K = 10, IQ2_XXS = 16, I8 = 24, MXFP4 = 39;

// The block layout shared by target layers and draft stages.
void add_block(std::vector<ExpectedTensor>& out, const std::string& p, std::int32_t layer, const Config& c,
               std::uint32_t experts, std::uint32_t expert_gate_up_type, std::uint32_t expert_down_type) {
    const std::uint64_t H = c.hidden_size, hc = c.hc_mult, mix = (2U + hc) * hc, I = c.moe_intermediate_size;
    for (const char* site : {"attn", "ffn"}) {
        add(out, p + "hc_" + site + "_fn.weight", {H * hc, mix}, F16, TensorRole::dense_projection, layer);
        add(out, p + "hc_" + site + "_base.weight", {mix}, F32, TensorRole::dense_projection, layer);
        add(out, p + "hc_" + site + "_scale.weight", {3}, F32, TensorRole::dense_projection, layer);
        add(out, p + site + "_norm.weight", {H}, F32, TensorRole::normalization, layer);
    }
    add(out, p + "attn_sinks.weight", {c.num_attention_heads}, F32, TensorRole::attention_projection, layer);
    add(out, p + "attn_q_a.weight", {H, c.q_lora_rank}, Q8_0, TensorRole::attention_projection, layer);
    add(out, p + "attn_q_a_norm.weight", {c.q_lora_rank}, F32, TensorRole::normalization, layer);
    add(out, p + "attn_q_b.weight", {c.q_lora_rank, std::uint64_t{c.num_attention_heads} * c.head_dim}, Q8_0,
        TensorRole::attention_projection, layer);
    add(out, p + "attn_kv.weight", {H, c.head_dim}, Q8_0, TensorRole::attention_projection, layer);
    add(out, p + "attn_kv_a_norm.weight", {c.head_dim}, F32, TensorRole::normalization, layer);
    add(out, p + "attn_output_a.weight",
        {std::uint64_t{c.num_attention_heads} * c.head_dim / c.o_groups, std::uint64_t{c.o_groups} * c.o_lora_rank}, Q8_0,
        TensorRole::attention_projection, layer);
    add(out, p + "attn_output_b.weight", {std::uint64_t{c.o_groups} * c.o_lora_rank, H}, Q8_0, TensorRole::attention_projection, layer);
    add(out, p + "ffn_gate_inp.weight", {H, experts}, F32, TensorRole::moe_router, layer);
    add(out, p + "exp_probs_b.bias", {experts}, F32, TensorRole::moe_router, layer);
    add(out, p + "exp_probs_b_vl.bias", {experts}, F32, TensorRole::moe_router, layer);
    add(out, p + "ffn_gate_shexp.weight", {H, I * c.n_shared_experts}, Q8_0, TensorRole::dense_projection, layer);
    add(out, p + "ffn_up_shexp.weight", {H, I * c.n_shared_experts}, Q8_0, TensorRole::dense_projection, layer);
    add(out, p + "ffn_down_shexp.weight", {I * c.n_shared_experts, H}, Q8_0, TensorRole::dense_projection, layer);
    add(out, p + "ffn_gate_exps.weight", {H, I, experts}, expert_gate_up_type, TensorRole::moe_expert, layer);
    add(out, p + "ffn_up_exps.weight", {H, I, experts}, expert_gate_up_type, TensorRole::moe_expert, layer);
    add(out, p + "ffn_down_exps.weight", {I, H, experts}, expert_down_type, TensorRole::moe_expert, layer);
}

ModelManifest manifest_of(const gguf::FileInfo& info, const std::vector<ExpectedTensor>& expected, const std::string& architecture) {
    std::map<std::string, const ExpectedTensor*> by_name;
    for (const auto& t : expected) require(by_name.emplace(t.name, &t).second, "duplicate expected tensor " + t.name);
    require(info.tensors.size() == expected.size(),
            "tensor count " + std::to_string(info.tensors.size()) + " differs from the expected " + std::to_string(expected.size()));
    ModelManifest manifest;
    manifest.architecture = architecture;
    for (const auto& t : info.tensors) {
        const auto it = by_name.find(t.name);
        require(it != by_name.end(), "unexpected tensor " + t.name);
        const auto& want = *it->second;
        require(t.dimensions == want.dimensions, "tensor " + t.name + " has unexpected dimensions");
        require(t.ggml_type == want.ggml_type, "tensor " + t.name + " has unexpected storage type " + std::to_string(t.ggml_type));
        ManifestTensor m;
        m.name = t.name;
        m.dimensions = t.dimensions;
        m.ggml_type = t.ggml_type;
        m.byte_size = t.byte_size;
        m.absolute_offset = t.absolute_offset;
        m.role = want.role;
        m.residency = want.residency;
        if (want.layer >= 0) m.layer = static_cast<std::uint32_t>(want.layer);
        manifest.total_tensor_bytes += m.byte_size;
        if (m.residency == ResidencyPolicy::file_backed_candidate) manifest.file_backed_candidate_bytes += m.byte_size;
        manifest.tensors.push_back(std::move(m));
        by_name.erase(it);
    }
    require(by_name.empty(), "missing tensor " + (by_name.empty() ? std::string() : by_name.begin()->first));
    manifest.layout_identity = model_manifest_layout_identity(manifest);
    return manifest;
}

}  // namespace

gguf::Limits reader_limits() noexcept {
    gguf::Limits limits;
    limits.max_retained_array_elements = 1U << 20U;
    return limits;
}

gguf::Limits vision_reader_limits() noexcept {
    gguf::Limits limits = reader_limits();
    limits.allow_hyphen_in_keys = true;
    return limits;
}

gguf::FileInfo read_file(const std::filesystem::path& path) { return gguf::read_file(path, reader_limits()); }

Config parse_config_json(std::string_view json) {
    const Json root = JsonParser(json).parse();
    const Json& t = member(root, "text_config");
    Config c;
    const auto u32 = [&](const char* key) { return as_u32(member(t, key), key); };
    const auto f64 = [&](const char* key) { return as_f64(member(t, key), key); };
    c.vocab_size = u32("vocab_size");
    c.hidden_size = u32("hidden_size");
    c.moe_intermediate_size = u32("moe_intermediate_size");
    c.num_hidden_layers = u32("num_hidden_layers");
    c.num_attention_heads = u32("num_attention_heads");
    c.num_key_value_heads = u32("num_key_value_heads");
    c.head_dim = u32("head_dim");
    c.qk_rope_head_dim = u32("qk_rope_head_dim");
    c.q_lora_rank = u32("q_lora_rank");
    c.o_lora_rank = u32("o_lora_rank");
    c.o_groups = u32("o_groups");
    c.n_routed_experts = u32("n_routed_experts");
    c.n_shared_experts = u32("n_shared_experts");
    c.num_experts_per_tok = u32("num_experts_per_tok");
    c.max_position_embeddings = as_u64(member(t, "max_position_embeddings"), "max_position_embeddings");
    c.sliding_window = u32("sliding_window");
    c.index_n_heads = u32("index_n_heads");
    c.index_head_dim = u32("index_head_dim");
    c.index_topk = u32("index_topk");
    c.candidate_source_layer_id = u32("candidate_source_layer_id");
    c.candidate_topk_blocks = u32("candidate_topk_blocks");
    c.candidate_block_size = u32("candidate_block_size");
    c.hc_mult = u32("hc_mult");
    c.hc_sinkhorn_iters = u32("hc_sinkhorn_iters");
    c.num_nextn_predict_layers = u32("num_nextn_predict_layers");
    c.rope_theta = f64("rope_theta");
    c.compress_rope_theta = f64("compress_rope_theta");
    c.rms_norm_eps = f64("rms_norm_eps");
    c.hc_eps = f64("hc_eps");
    c.swiglu_limit = f64("swiglu_limit");
    c.routed_scaling_factor = f64("routed_scaling_factor");
    c.scoring_func = as_text(member(t, "scoring_func"), "scoring_func");
    c.hidden_act = as_text(member(t, "hidden_act"), "hidden_act");
    c.topk_method = as_text(member(t, "topk_method"), "topk_method");
    c.norm_topk_prob = as_bool(member(t, "norm_topk_prob"), "norm_topk_prob");
    const Json& rope = member(t, "rope_scaling");
    c.rope_scaling.type = as_text(member(rope, "rope_type"), "rope_scaling.rope_type");
    c.rope_scaling.factor = as_f64(member(rope, "factor"), "rope_scaling.factor");
    c.rope_scaling.beta_fast = as_f64(member(rope, "beta_fast"), "rope_scaling.beta_fast");
    c.rope_scaling.beta_slow = as_f64(member(rope, "beta_slow"), "rope_scaling.beta_slow");
    c.rope_scaling.original_max_position_embeddings =
        as_f64(member(rope, "original_max_position_embeddings"), "rope_scaling.original_max_position_embeddings");
    c.compress_ratios = as_vector<std::uint32_t>(member(t, "compress_ratios"), "compress_ratios");
    c.kv_source_layer_ids = as_vector<std::uint32_t>(member(t, "kv_source_layer_ids"), "kv_source_layer_ids");
    c.index_source_layer_ids = as_vector<std::uint32_t>(member(t, "index_source_layer_ids"), "index_source_layer_ids");
    c.engram_layer_ids = as_vector<std::uint32_t>(member(t, "engram_layer_ids"), "engram_layer_ids");
    c.engram_num_embeddings = as_vector<std::uint64_t>(member(t, "engram_num_embeddings"), "engram_num_embeddings");
    c.engram_n_heads = u32("engram_n_heads");
    c.engram_head_dim = u32("engram_head_dim");
    c.engram_max_ngram_size = u32("engram_max_ngram_size");
    c.engram_vocab_size = as_u64(member(t, "engram_vocab_size"), "engram_vocab_size");
    c.engram_compressed_vocab_size = u32("engram_compressed_vocab_size");
    c.engram_pad_token_id = u32("engram_pad_token_id");
    c.dspark_block_size = u32("dspark_block_size");
    c.dspark_markov_rank = u32("dspark_markov_rank");
    c.dspark_n_routed_experts = u32("dspark_n_routed_experts");
    c.dspark_num_experts_per_tok = u32("dspark_num_experts_per_tok");
    c.dspark_noise_token_id = u32("dspark_noise_token_id");
    c.dspark_target_layer_ids = as_vector<std::uint32_t>(member(t, "dspark_target_layer_ids"), "dspark_target_layer_ids");
    c.bos_token_id = as_u32(member(root, "bos_token_id"), "bos_token_id");
    c.eos_token_id = as_u32(member(root, "eos_token_id"), "eos_token_id");
    c.pad_token_id = as_u32(member(root, "pad_token_id"), "pad_token_id");
    c.image_token_id = as_u32(member(root, "image_token_id"), "image_token_id");

    // Internal consistency of the geometry used to derive the inventory.
    require(c.num_hidden_layers > 0 && c.hidden_size > 0 && c.vocab_size > 0, "config geometry is empty");
    require(c.compress_ratios.size() == std::size_t{c.num_hidden_layers} + c.num_nextn_predict_layers,
            "compress_ratios must list every target layer and draft stage");
    for (std::size_t i = c.num_hidden_layers; i < c.compress_ratios.size(); ++i)
        require(c.compress_ratios[i] == 0, "draft stages must use sliding-window attention only");
    require(c.num_key_value_heads == 1 && c.qk_rope_head_dim <= c.head_dim && c.qk_rope_head_dim % 2U == 0,
            "attention geometry unsupported (one shared latent KV head, even RoPE part)");
    require(c.o_groups > 0 && (std::uint64_t{c.num_attention_heads} * c.head_dim) % c.o_groups == 0 &&
                c.num_attention_heads % c.o_groups == 0,
            "grouped output projection does not divide the heads");
    require(c.n_shared_experts == 1 && c.num_experts_per_tok <= c.n_routed_experts, "MoE geometry unsupported");
    require(c.hc_mult > 0 && c.rope_scaling.type == "yarn", "unsupported hyper-connection or RoPE scaling");
    require(!c.dspark_target_layer_ids.empty() &&
                std::all_of(c.dspark_target_layer_ids.begin(), c.dspark_target_layer_ids.end(),
                            [&](std::uint32_t l) { return l < c.num_hidden_layers; }),
            "draft target layers must be target layers");
    return c;
}

std::vector<LayerAttributes> layer_attributes(const Config& c) {
    std::vector<LayerAttributes> layers(c.num_hidden_layers);
    for (const auto l : c.kv_source_layer_ids) require(l < c.num_hidden_layers, "kv source layer out of range");
    for (const auto l : c.index_source_layer_ids) require(l < c.num_hidden_layers, "index source layer out of range");
    for (const auto l : c.engram_layer_ids) require(l < c.num_hidden_layers, "Engram layer out of range");
    std::int32_t kv_source = -1, index_source = -1;
    for (std::uint32_t l = 0; l < c.num_hidden_layers; ++l) {
        auto& a = layers[l];
        a.compress_ratio = c.compress_ratios[l];
        a.kv_source = contains(c.kv_source_layer_ids, l);
        a.index_source = contains(c.index_source_layer_ids, l);
        a.engram = contains(c.engram_layer_ids, l);
        if (a.kv_source) {
            require(a.compress_ratio > 0, "a KV source layer must compress");
            kv_source = static_cast<std::int32_t>(l);
        }
        if (a.index_source) {
            require(a.compress_ratio > 0, "an index source layer must attend to compressed KV");
            index_source = static_cast<std::int32_t>(l);
        }
        a.compressor_gate = a.kv_source && a.compress_ratio > 1;
        if (a.compress_ratio > 0) {
            require(kv_source >= 0 && layers[static_cast<std::size_t>(kv_source)].compress_ratio == a.compress_ratio,
                    "layer " + std::to_string(l) + " reads compressed KV of another ratio or of no source");
            require(index_source >= 0 && index_source >= kv_source,
                    "layer " + std::to_string(l) + " has no index source after its KV source");
            a.kv_source_layer = kv_source;
            a.index_source_layer = index_source;
        }
        a.candidate_source = l == c.candidate_source_layer_id;
        a.uses_candidates = a.index_source && l > c.candidate_source_layer_id;
    }
    require(c.candidate_source_layer_id < c.num_hidden_layers && layers[c.candidate_source_layer_id].index_source &&
                layers[c.candidate_source_layer_id].kv_source,
            "candidate source must be a KV and index source layer");
    return layers;
}

std::vector<ExpectedTensor> expected_target_tensors(const Config& c) {
    std::vector<ExpectedTensor> out;
    const auto layers = layer_attributes(c);
    const std::uint64_t H = c.hidden_size, V = c.vocab_size;
    add(out, "token_embd.weight", {H, V}, F16, TensorRole::embedding, -1);
    add(out, "output_norm.weight", {H}, F32, TensorRole::normalization, -1);
    add(out, "output.weight", {H, V}, Q8_0, TensorRole::output_head, -1);
    for (std::uint32_t l = 0; l < c.num_hidden_layers; ++l) {
        const auto layer = static_cast<std::int32_t>(l);
        const std::string p = "blk." + std::to_string(l) + ".";
        add_block(out, p, layer, c, c.n_routed_experts, IQ2_XXS, Q2_K);
        const auto& a = layers[l];
        if (a.engram) {
            const std::size_t e = static_cast<std::size_t>(
                std::find(c.engram_layer_ids.begin(), c.engram_layer_ids.end(), l) - c.engram_layer_ids.begin());
            require(e < c.engram_num_embeddings.size(), "Engram table rows missing for a layer");
            const std::uint64_t cols = std::uint64_t{c.engram_max_ngram_size - 1U} * c.engram_n_heads * c.engram_head_dim;
            add(out, p + "engram_q_norm.weight", {H, c.hc_mult}, F32, TensorRole::ngram_table, layer);
            add(out, p + "engram_k_norm.weight", {H, c.hc_mult}, F32, TensorRole::ngram_table, layer);
            add(out, p + "engram_kv.weight", {cols, H * (c.hc_mult + 1U)}, F16, TensorRole::ngram_table, layer);
            add(out, p + "engram_embd.weight", {c.engram_head_dim + c.engram_head_dim / 32U, c.engram_num_embeddings[e]}, I8,
                TensorRole::ngram_table, layer, ResidencyPolicy::file_backed_candidate);
        }
        if (a.kv_source) {
            add(out, p + "attn_compressor_kv.weight", {H, c.head_dim}, F16, TensorRole::attention_projection, layer);
            add(out, p + "attn_compressor_norm.weight", {c.head_dim}, F32, TensorRole::normalization, layer);
            add(out, p + "indexer.attn_k.weight", {c.head_dim, c.index_head_dim}, F16, TensorRole::sparse_index, layer);
            add(out, p + "indexer.k_norm.weight", {c.index_head_dim}, F32, TensorRole::sparse_index, layer);
            if (a.compressor_gate)
                add(out, p + "attn_compressor_gate.weight", {H, c.head_dim}, F16, TensorRole::attention_projection, layer);
        }
        if (a.index_source) {
            add(out, p + "indexer.attn_q_b.weight", {c.q_lora_rank, std::uint64_t{c.index_n_heads} * c.index_head_dim}, F16,
                TensorRole::sparse_index, layer);
            add(out, p + "indexer.proj.weight", {H, c.index_n_heads}, F16, TensorRole::sparse_index, layer);
        }
    }
    return out;
}

std::vector<ExpectedTensor> expected_draft_tensors(const Config& c) {
    std::vector<ExpectedTensor> out;
    const std::uint64_t H = c.hidden_size, V = c.vocab_size;
    const std::uint32_t stages = c.num_nextn_predict_layers;
    require(stages > 0, "config has no draft stages");
    for (std::uint32_t s = 0; s < stages; ++s) {
        const std::string p = "mtp." + std::to_string(s) + ".";
        const auto layer = static_cast<std::int32_t>(s);
        if (s == 0) {
            add(out, p + "main_proj.weight", {H * c.dspark_target_layer_ids.size(), H}, Q8_0, TensorRole::mtp, layer);
            add(out, p + "main_norm.weight", {H}, F32, TensorRole::mtp, layer);
        }
        std::vector<ExpectedTensor> block;
        add_block(block, p, layer, c, c.dspark_n_routed_experts, MXFP4, MXFP4);
        for (auto& t : block) {
            t.role = TensorRole::mtp;
            out.push_back(std::move(t));
        }
        if (s + 1 == stages) {
            add(out, p + "norm.weight", {H}, F32, TensorRole::mtp, layer);
            add(out, p + "markov_head.embed.weight", {c.dspark_markov_rank, V}, F16, TensorRole::mtp, layer);
            add(out, p + "markov_head.head.weight", {c.dspark_markov_rank, V}, F16, TensorRole::mtp, layer);
            add(out, p + "confidence_head.proj.weight", {H + c.dspark_markov_rank, 1}, F32, TensorRole::mtp, layer);
        }
    }
    return out;
}

TargetModel validate_target(const gguf::FileInfo& info) {
    require(meta_text(info, "general.architecture") == "deepseek41", "general.architecture must be deepseek41");
    TargetModel model;
    model.config = parse_config_json(meta_text(info, "deepseek41.config"));
    cross_check(info, model.config);
    model.layers = layer_attributes(model.config);
    model.engram = engram_contract(info, model.config);
    model.manifest = manifest_of(info, expected_target_tensors(model.config), "deepseek41");
    return model;
}

DraftModel validate_draft(const gguf::FileInfo& info, const Config& c) {
    require(meta_text(info, "general.architecture") == "deepseek41-dspark", "general.architecture must be deepseek41-dspark");
    DraftModel d;
    d.stages = c.num_nextn_predict_layers;
    d.block_size = static_cast<std::uint32_t>(meta_u64(info, "dspark.block_size"));
    d.markov_rank = static_cast<std::uint32_t>(meta_u64(info, "dspark.markov_rank"));
    d.noise_token_id = static_cast<std::uint32_t>(meta_u64(info, "dspark.noise_token_id"));
    d.n_routed_experts = static_cast<std::uint32_t>(meta_u64(info, "dspark.n_routed_experts"));
    d.num_experts_per_tok = static_cast<std::uint32_t>(meta_u64(info, "dspark.num_experts_per_tok"));
    for (const auto v : meta_array(info, "dspark.target_layer_ids")) d.target_layer_ids.push_back(static_cast<std::uint32_t>(v));
    require(d.block_size == c.dspark_block_size && d.markov_rank == c.dspark_markov_rank &&
                d.noise_token_id == c.dspark_noise_token_id && d.n_routed_experts == c.dspark_n_routed_experts &&
                d.num_experts_per_tok == c.dspark_num_experts_per_tok,
            "draft metadata differs from the target's config");
    same_list(d.target_layer_ids, c.dspark_target_layer_ids, "draft target layer ids");
    require(d.noise_token_id < c.vocab_size, "draft noise token outside the vocabulary");
    d.manifest = manifest_of(info, expected_draft_tensors(c), "deepseek41-dspark");
    return d;
}

std::vector<ExpectedTensor> expected_vision_tensors(const VisionConfig& c) {
    constexpr std::uint32_t BF16 = 30;
    const std::uint64_t W = c.hidden_size, M = c.intermediate_size, P = c.projection_size;
    std::vector<ExpectedTensor> out;
    add(out, "aligner.w1.bias", {P}, BF16, TensorRole::dense_projection, -1);
    add(out, "aligner.w1.weight", {c.aligner_inputs(), P}, BF16, TensorRole::dense_projection, -1);
    add(out, "aligner.w2.bias", {P}, BF16, TensorRole::dense_projection, -1);
    add(out, "aligner.w2.weight", {P, P}, BF16, TensorRole::dense_projection, -1);
    add(out, "image_end", {P}, BF16, TensorRole::embedding, -1);
    add(out, "image_newline", {P}, BF16, TensorRole::embedding, -1);
    add(out, "image_start", {P}, BF16, TensorRole::embedding, -1);
    for (std::uint32_t l = 0; l < c.num_hidden_layers; ++l) {
        const std::string p = "vision.blocks." + std::to_string(l) + ".";
        const auto layer = static_cast<std::int32_t>(l);
        add(out, p + "attn.wo.bias", {W}, BF16, TensorRole::attention_projection, layer);
        add(out, p + "attn.wo.weight", {W, W}, BF16, TensorRole::attention_projection, layer);
        add(out, p + "attn.wqkv.bias", {3U * W}, BF16, TensorRole::attention_projection, layer);
        add(out, p + "attn.wqkv.weight", {W, 3U * W}, BF16, TensorRole::attention_projection, layer);
        add(out, p + "mlp.w1.weight", {W, 2U * M}, BF16, TensorRole::dense_projection, layer);
        add(out, p + "mlp.w2.weight", {M, W}, BF16, TensorRole::dense_projection, layer);
        add(out, p + "norm1.weight", {W}, BF16, TensorRole::normalization, layer);
        add(out, p + "norm2.weight", {W}, BF16, TensorRole::normalization, layer);
    }
    add(out, "vision.norm.weight", {W}, BF16, TensorRole::normalization, -1);
    add(out, "vision.patch_embed.proj.bias", {W}, BF16, TensorRole::dense_projection, -1);
    add(out, "vision.patch_embed.proj.weight", {c.patch_features(), W}, BF16, TensorRole::dense_projection, -1);
    return out;
}

VisionModel validate_vision(const gguf::FileInfo& info) {
    const std::string a = "deepseek4-vision";
    require(meta_text(info, "general.architecture") == a, "general.architecture must be " + a);
    const auto u32 = [&](const std::string& key) {
        const std::uint64_t v = meta_u64(info, a + "." + key);
        require(v <= 0xffffffffULL, "metadata " + a + "." + key + " is out of range");
        return static_cast<std::uint32_t>(v);
    };
    VisionModel m;
    VisionConfig& c = m.config;
    c.checkpoint_variant = meta_text(info, a + ".checkpoint_variant");
    c.source_revision = meta_text(info, "general.source.revision");
    c.num_hidden_layers = u32("block_count");
    c.hidden_size = u32("embedding_length");
    c.intermediate_size = u32("feed_forward_length");
    c.num_attention_heads = u32("attention.head_count");
    c.projection_size = u32("projection_length");
    c.patch_size = u32("patch_size");
    c.downsample_ratio = u32("downsample_ratio");
    c.max_image_tokens = u32("image.max_tokens");
    c.min_pixels = u32("image.min_pixels");
    c.max_wh_ratio = u32("image.max_width_height_ratio");
    c.image_token_id = u32("image.token_id");
    c.language_layers = u32("language.block_count");
    c.language_experts = u32("language.expert_count");
    c.rope_theta = meta_f64(info, a + ".rope.freq_base");
    c.rms_norm_eps = meta_f64(info, a + ".attention.layer_norm_rms_epsilon");

    require(c.checkpoint_variant == "v4.1-flash", "vision checkpoint variant must be v4.1-flash");
    require(c.source_revision == pinned_source_revision, "vision file comes from another source revision");
    require(c.num_hidden_layers > 0 && c.num_hidden_layers <= 256, "vision block_count is out of range");
    require(c.num_attention_heads > 0 && c.hidden_size > 0 && c.hidden_size <= 16384 &&
                c.hidden_size % c.num_attention_heads == 0,
            "vision width must be a multiple of the head count");
    require(c.head_dim() % 4U == 0, "vision head dimension must be a multiple of 4 (two rotary axes)");
    require(c.intermediate_size > 0 && c.intermediate_size <= 65536, "vision feed_forward_length is out of range");
    require(c.projection_size > 0 && c.projection_size <= 65536, "vision projection_length is out of range");
    require(c.patch_size > 0 && c.patch_size <= 64, "vision patch_size is out of range");
    require(c.downsample_ratio > 0 && c.downsample_ratio <= 8, "vision downsample_ratio is out of range");
    require(c.max_image_tokens > 0 && c.max_image_tokens <= 16384, "vision image.max_tokens is out of range");
    const std::uint64_t cell = std::uint64_t{c.cell()} * c.cell();
    require(c.min_pixels >= cell && c.min_pixels <= cell * c.max_image_tokens,
            "vision image.min_pixels is outside the token range");
    require(std::isfinite(c.rope_theta) && c.rope_theta > 1.0, "vision rope.freq_base is out of range");
    require(std::isfinite(c.rms_norm_eps) && c.rms_norm_eps > 0.0 && c.rms_norm_eps < 1e-2,
            "vision norm epsilon is out of range");
    m.manifest = manifest_of(info, expected_vision_tensors(c), a);
    return m;
}

void require_vision_matches_target(const VisionConfig& v, const Config& t) {
    require(v.image_token_id == t.image_token_id, "vision image token id differs from the language model");
    require(v.image_token_id < t.vocab_size, "vision image token id is outside the vocabulary");
    require(v.projection_size == t.hidden_size, "vision projection width differs from the language hidden size");
    require(v.language_layers == t.num_hidden_layers, "vision file was released for another language depth");
    require(v.language_experts == t.n_routed_experts, "vision file was released for another expert count");
}

const PinnedFile& pinned_file(PinnedFileKind kind) noexcept {
    for (const auto& file : pinned_files)
        if (file.kind == kind) return file;
    return pinned_files[0];
}

void admit_file(PinnedFileKind kind, std::uint64_t file_bytes, const ModelContentIdentity& file_sha256) {
    const auto& pin = pinned_file(kind);
    require(file_bytes == pin.bytes, std::string(pin.file_name) + ": byte count differs from the admitted file");
    require(model_content_identity_hex(file_sha256) == pin.sha256_hex,
            std::string(pin.file_name) + ": SHA-256 differs from the admitted file");
}

}  // namespace strix::deepseek41
