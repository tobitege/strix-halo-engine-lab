#include "strix/qwen35/config.hpp"

#include <cmath>
#include <stdexcept>
#include <limits>
#include "../core/dense_linear.hpp"

namespace strix::qwen35 {
namespace {

[[noreturn]] void bad(const std::string& key, const std::string& why) {
    throw std::invalid_argument("qwen35: " + key + ": " + why);
}

std::uint32_t number(const gguf::FileInfo& info, const std::string& key) {
    const auto* e = info.find_metadata(key);
    if (e == nullptr) bad(key, "missing");
    if (!e->unsigned_value.has_value() || *e->unsigned_value == 0 || *e->unsigned_value > 0xFFFFFFFFULL)
        bad(key, "not a positive 32-bit integer");
    return static_cast<std::uint32_t>(*e->unsigned_value);
}

double real(const gguf::FileInfo& info, const std::string& key) {
    const auto* e = info.find_metadata(key);
    if (e == nullptr) bad(key, "missing");
    if (!e->floating_value.has_value() || !(*e->floating_value > 0.0) || !std::isfinite(*e->floating_value))
        bad(key, "not a positive finite float");
    return *e->floating_value;
}

}  // namespace

Config config_from_gguf(const gguf::FileInfo& info) {
    const auto* arch = info.find_metadata("general.architecture");
    if (arch == nullptr || !arch->string_value.has_value() || *arch->string_value != kArchitecture)
        bad("general.architecture", std::string("expected \"") + kArchitecture + "\"");
    Config c;
    const std::string p = std::string(kArchitecture) + ".";
    c.layers = number(info, p + "block_count");
    c.hidden = number(info, p + "embedding_length");
    c.intermediate = number(info, p + "feed_forward_length");
    c.context = number(info, p + "context_length");
    c.heads = number(info, p + "attention.head_count");
    c.kv_heads = number(info, p + "attention.head_count_kv");
    c.head_dim = number(info, p + "attention.key_length");
    if (number(info, p + "attention.value_length") != c.head_dim) bad(p + "attention.value_length", "must equal key_length");
    c.rms_eps = static_cast<float>(real(info, p + "attention.layer_norm_rms_epsilon"));
    c.rope_theta = real(info, p + "rope.freq_base");
    c.rotary_dim = number(info, p + "rope.dimension_count");
    c.gdn_conv_kernel = number(info, p + "ssm.conv_kernel");
    c.gdn_key_dim = number(info, p + "ssm.state_size");
    c.gdn_key_heads = number(info, p + "ssm.group_count");
    c.gdn_value_heads = number(info, p + "ssm.time_step_rank");
    const auto inner = number(info, p + "ssm.inner_size");
    c.full_attention_interval = number(info, p + "full_attention_interval");
    // nextn_predict_layers may legitimately be 0 (no MTP), so it is read without the positivity rule.
    const auto* nextn = info.find_metadata(p + "nextn_predict_layers");
    if (nextn == nullptr || !nextn->unsigned_value.has_value() || *nextn->unsigned_value > 1)
        bad(p + "nextn_predict_layers", "missing or not 0/1");
    c.mtp_layers = static_cast<std::uint32_t>(*nextn->unsigned_value);

    if (c.kv_heads > c.heads || c.heads % c.kv_heads != 0) bad(p + "attention.head_count_kv", "must divide head_count");
    if (c.rotary_dim > c.head_dim || c.rotary_dim % 2 != 0) bad(p + "rope.dimension_count", "even and <= key_length");
    if (c.gdn_value_heads % c.gdn_key_heads != 0) bad(p + "ssm.time_step_rank", "must be a multiple of group_count");
    if (inner % c.gdn_value_heads != 0) bad(p + "ssm.inner_size", "must be a multiple of time_step_rank");
    c.gdn_value_dim = inner / c.gdn_value_heads;
    if (c.gdn_value_dim != c.gdn_key_dim) bad(p + "ssm.inner_size", "value head dim must equal state_size");
    if (c.full_attention_interval > c.layers) bad(p + "full_attention_interval", "larger than block_count");

    // Vocabulary = rows of token_embd (tokenizer arrays are not retained by the default reader limits).
    for (const auto& t : info.tensors) {
        if (t.name == "token_embd.weight") {
            if (t.dimensions.size() != 2 || t.dimensions[0] != c.hidden) bad("token_embd.weight", "shape must be [hidden, vocab]");
            if (t.dimensions[1] == 0 || t.dimensions[1] > 0x7FFFFFFFULL)
                bad("token_embd.weight", "vocabulary must fit a positive signed index");
            c.vocab = static_cast<std::uint32_t>(t.dimensions[1]);
            for (const auto width : {2ULL * c.heads * c.head_dim, 1ULL * c.kv_heads * c.head_dim,
                                    (2ULL * c.gdn_key_heads + c.gdn_value_heads) * c.gdn_key_dim})
                if (width > std::numeric_limits<std::uint32_t>::max()) bad("dimensions", "width overflow");
            return c;
        }
    }
    bad("token_embd.weight", "missing");
}

void validate_backend(const Config& c, std::uint32_t max_positions) {
    if (!c.layers || c.layers > 64 || !c.full_attention_interval || c.full_attention_interval > c.layers ||
        !c.heads || !c.kv_heads || c.heads % c.kv_heads || c.heads / c.kv_heads > 8 ||
        c.head_dim != 256 || c.gdn_key_dim != 128 || c.gdn_value_dim != 128 || c.gdn_conv_kernel != 4 ||
        !c.gdn_key_heads || !c.gdn_value_heads || c.gdn_value_heads % c.gdn_key_heads ||
        !c.hidden || c.hidden % 32 || !c.intermediate || c.intermediate % 32 || !c.vocab || c.vocab > 0x7FFFFFFFU ||
        !c.rotary_dim || c.rotary_dim > c.head_dim || c.rotary_dim % 2 || !max_positions)
        bad("backend", "unsupported dimensions or capacity");
    // Launch row offsets and grids use 32-bit indices; allocation products use size_t.
    for (const auto width : {2ULL * c.hidden, 1ULL * c.intermediate, 1ULL * c.vocab,
                            2ULL * c.heads * c.head_dim, 1ULL * c.kv_heads * c.head_dim,
                            (2ULL * c.gdn_key_heads + c.gdn_value_heads) * c.gdn_key_dim})
        if (width > (std::numeric_limits<std::uint32_t>::max() - 255ULL) / 32)
            bad("backend", "row width overflow");
    (void)core::checked_product({c.kv_heads, max_positions, c.head_dim, 2});
    (void)core::checked_product({c.gdn_value_heads, c.gdn_key_dim, c.gdn_value_dim, 2, 4});
    for (const auto& spec : expected_tensors(c)) {
        std::size_t elements = 1;
        for (const auto d : spec.dimensions) elements = core::checked_product({elements, static_cast<std::size_t>(d)});
        (void)core::checked_product({elements, 4});
    }
}

std::vector<TensorSpec> expected_tensors(const Config& c) {
    std::vector<TensorSpec> out;
    const std::uint64_t H = c.hidden, I = c.intermediate, hd = c.head_dim;
    auto mat = [&](std::string n, std::uint64_t in, std::uint64_t rows) { out.push_back({std::move(n), {in, rows}, true}); };
    auto vec = [&](std::string n, std::vector<std::uint64_t> d) { out.push_back({std::move(n), std::move(d), false}); };
    mat("token_embd.weight", H, c.vocab);
    vec("output_norm.weight", {H});
    mat("output.weight", H, c.vocab);
    auto common = [&](const std::string& b) {
        vec(b + ".attn_norm.weight", {H});
        vec(b + ".post_attention_norm.weight", {H});
        mat(b + ".ffn_gate.weight", H, I);
        mat(b + ".ffn_up.weight", H, I);
        mat(b + ".ffn_down.weight", I, H);
    };
    auto attention = [&](const std::string& b) {
        mat(b + ".attn_q.weight", H, c.q_gate_width());
        mat(b + ".attn_k.weight", H, c.kv_width());
        mat(b + ".attn_v.weight", H, c.kv_width());
        mat(b + ".attn_output.weight", std::uint64_t{c.heads} * hd, H);
        vec(b + ".attn_q_norm.weight", {hd});
        vec(b + ".attn_k_norm.weight", {hd});
    };
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const std::string b = "blk." + std::to_string(l);
        common(b);
        if (c.kind(l) == LayerKind::full_attention) {
            attention(b);
            continue;
        }
        mat(b + ".attn_qkv.weight", H, c.gdn_conv_channels());
        mat(b + ".attn_gate.weight", H, c.gdn_v_width());
        vec(b + ".ssm_conv1d.weight", {c.gdn_conv_kernel, c.gdn_conv_channels()});
        mat(b + ".ssm_beta.weight", H, c.gdn_value_heads);
        mat(b + ".ssm_alpha.weight", H, c.gdn_value_heads);
        vec(b + ".ssm_dt.bias", {c.gdn_value_heads});
        vec(b + ".ssm_a", {c.gdn_value_heads});
        vec(b + ".ssm_norm.weight", {c.gdn_value_dim});
        mat(b + ".ssm_out.weight", c.gdn_v_width(), H);
    }
    if (c.mtp_layers != 0) {
        const std::string b = "blk." + std::to_string(c.layers);
        common(b);
        attention(b);
        vec(b + ".nextn.enorm.weight", {H});
        vec(b + ".nextn.hnorm.weight", {H});
        mat(b + ".nextn.eh_proj.weight", 2 * H, H);
        vec(b + ".nextn.shared_head_norm.weight", {H});
    }
    return out;
}

std::vector<std::string> required_tensor_names(const Config& config) {
    std::vector<std::string> names;
    for (auto& t : expected_tensors(config)) names.push_back(std::move(t.name));
    return names;
}

}  // namespace strix::qwen35
