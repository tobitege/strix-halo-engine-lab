// Host contract tests for the qwen35 family: a tiny GGUF is built in memory from the same expected_tensors()
// plan the family validates, then mutated (missing tensor, wrong head count, wrong arch, wrong shape/type).
#include "strix/model_family_registry.hpp"
#include "strix/qwen35/config.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

using namespace strix;

struct Writer {
    std::string bytes;
    template <class T> void put(T v) { bytes.append(reinterpret_cast<const char*>(&v), sizeof(T)); }
    void str(const std::string& s) { put<std::uint64_t>(s.size()); bytes += s; }
};

struct Meta {
    std::string arch = "qwen35";
    std::uint32_t heads = 2, kv_heads = 1, drop_key_index = 0;
    std::string drop_tensor;
    bool bad_shape = false, bad_type = false;
};

std::string tiny_gguf(const Meta& m) {
    // layers 4 (3 GDN + 1 attention) + 1 MTP; H 64, I 96, V 64, head 16, rotary 8, gdn 2 key / 4 value heads of 8.
    std::vector<std::pair<std::string, std::uint32_t>> u32 = {
        {"qwen35.block_count", 4}, {"qwen35.context_length", 4096}, {"qwen35.embedding_length", 64},
        {"qwen35.feed_forward_length", 96}, {"qwen35.attention.head_count", m.heads},
        {"qwen35.attention.head_count_kv", m.kv_heads}, {"qwen35.attention.key_length", 16},
        {"qwen35.attention.value_length", 16}, {"qwen35.rope.dimension_count", 8}, {"qwen35.ssm.conv_kernel", 4},
        {"qwen35.ssm.state_size", 8}, {"qwen35.ssm.group_count", 2}, {"qwen35.ssm.time_step_rank", 4},
        {"qwen35.ssm.inner_size", 32}, {"qwen35.full_attention_interval", 4}, {"qwen35.nextn_predict_layers", 1}};
    // Build the config the same way a consumer would, from a first pass without tensors is impossible (vocab comes
    // from token_embd), so derive the plan from a hand-set Config.
    qwen35::Config c;
    c.layers = 4; c.hidden = 64; c.intermediate = 96; c.vocab = 64; c.context = 4096; c.heads = 2; c.kv_heads = 1;
    c.head_dim = 16; c.rotary_dim = 8; c.gdn_key_heads = 2; c.gdn_key_dim = 8; c.gdn_value_heads = 4; c.gdn_value_dim = 8;
    c.gdn_conv_kernel = 4; c.mtp_layers = 1;
    auto plan = qwen35::expected_tensors(c);
    std::vector<qwen35::TensorSpec> tensors;
    for (auto& t : plan) {
        if (t.name == m.drop_tensor) continue;
        if (m.bad_shape && t.name == "blk.0.ffn_up.weight") t.dimensions[1] += 1;
        tensors.push_back(t);
    }
    Writer w;
    w.bytes = "GGUF";
    w.put<std::uint32_t>(3);
    w.put<std::uint64_t>(tensors.size());
    w.put<std::uint64_t>(u32.size() + 3);
    w.str("general.architecture"); w.put<std::uint32_t>(8); w.str(m.arch);
    for (const auto& [k, v] : u32) { w.str(k); w.put<std::uint32_t>(4); w.put<std::uint32_t>(v); }
    w.str("qwen35.attention.layer_norm_rms_epsilon"); w.put<std::uint32_t>(6); w.put<float>(1e-6f);
    w.str("qwen35.rope.freq_base"); w.put<std::uint32_t>(6); w.put<float>(1e7f);
    std::uint64_t offset = 0;
    std::vector<std::uint64_t> sizes;
    for (const auto& t : tensors) {
        std::uint64_t n = 1;
        for (auto d : t.dimensions) n *= d;
        const bool f32 = !t.matrix || (m.bad_type && t.name == "blk.0.ffn_up.weight");
        const std::uint64_t bytes = n * (f32 ? 4 : 2);
        w.str(t.name);
        w.put<std::uint32_t>(static_cast<std::uint32_t>(t.dimensions.size()));
        for (auto d : t.dimensions) w.put<std::uint64_t>(d);
        w.put<std::uint32_t>(f32 ? 0 : 30);
        w.put<std::uint64_t>(offset);
        sizes.push_back(bytes);
        offset += (bytes + 31) / 32 * 32;
    }
    w.bytes.append((32 - w.bytes.size() % 32) % 32, '\0');
    w.bytes.append(offset, '\0');
    return w.bytes;
}

gguf::FileInfo parse(const std::string& bytes) {
    std::istringstream in(bytes);
    return gguf::read(in, bytes.size());
}

template <class F> bool throws(F&& f) {
    try { f(); } catch (const std::invalid_argument&) { return true; }
    return false;
}

}  // namespace

int main(int argc, char** argv) try {
    const auto good = parse(tiny_gguf({}));
    const auto c = qwen35::config_from_gguf(good);
    expect(c.layers == 4 && c.hidden == 64 && c.vocab == 64 && c.mtp_layers == 1, "config values");
    expect(c.gdn_value_dim == 8 && c.gdn_conv_channels() == 2 * 16 + 32, "derived gdn geometry");
    expect(c.kind(0) == qwen35::LayerKind::gated_delta_net && c.kind(3) == qwen35::LayerKind::full_attention, "layer kinds");
    const auto names = qwen35::required_tensor_names(c);
    expect(names.size() == good.tensors.size(), "required names cover the file exactly");
    expect(names.front() == "token_embd.weight" && names.back() == "blk.4.nextn.shared_head_norm.weight", "plan order");

    qwen35::register_family();
    qwen35::register_family();  // idempotent
    const auto* family = find_model_family("qwen35");
    expect(family != nullptr && family->name == "qwen35", "family registered");
    if (family) {
        family->validate(good);
        expect(!family->describe(good).empty(), "describe");
    }

    Meta m;
    m.drop_tensor = "blk.2.ssm_a";
    expect(throws([&] { find_model_family("qwen35")->validate(parse(tiny_gguf(m))); }), "missing tensor rejected");
    m = {};
    m.heads = 3;  // 3 % 1 ok, but attn_q / attn_output shapes no longer match the plan built for 2 heads
    expect(throws([&] { find_model_family("qwen35")->validate(parse(tiny_gguf(m))); }), "wrong head count rejected");
    m = {};
    m.kv_heads = 3;
    expect(throws([&] { (void)qwen35::config_from_gguf(parse(tiny_gguf(m))); }), "kv heads not dividing heads rejected");
    m = {};
    m.arch = "llama";
    expect(throws([&] { (void)qwen35::config_from_gguf(parse(tiny_gguf(m))); }), "wrong arch rejected");
    m = {};
    m.bad_shape = true;
    expect(throws([&] { find_model_family("qwen35")->validate(parse(tiny_gguf(m))); }), "wrong shape rejected");
    m = {};
    m.bad_type = true;
    expect(throws([&] { find_model_family("qwen35")->validate(parse(tiny_gguf(m))); }), "wrong tensor type rejected");

    // Execution admission regressions (malformed files must fail before device work).
    auto malformed = good;
    for (auto& e : malformed.metadata)
        if (e.key == "qwen35.ssm.time_step_rank") e.unsigned_value = 0;
    expect(throws([&] { (void)qwen35::validate(malformed); }), "zero GDN rank rejected");
    malformed = good;
    for (auto& t : malformed.tensors)
        if (t.name == "blk.0.attn_qkv.weight") ++t.dimensions[1];
    expect(throws([&] { (void)qwen35::validate(malformed); }), "oversized qkv rejected");
    malformed = good;
    for (auto& t : malformed.tensors)
        if (t.name == "token_embd.weight") t.dimensions[1] = 0x100000000ULL;
    expect(throws([&] { (void)qwen35::validate(malformed); }), "vocabulary narrowing rejected");
    expect(throws([&] { qwen35::validate_backend(c, 16); }), "unsupported backend dimensions rejected");
    qwen35::Config native;
    qwen35::validate_backend(native, 16);
    native.heads = 0xFFFFFFFFU;
    expect(throws([&] { qwen35::validate_backend(native, 16); }), "backend width overflow rejected");

    // Optional: files written by tools/qwen35/convert.py must be admitted by the same family.
    for (int i = 1; i < argc; ++i) {
        const auto admitted = admit_model_file(argv[i]);
        expect(admitted.family != nullptr && admitted.family->name == "qwen35", std::string("admitted ") + argv[i]);
        if (admitted.family) std::cout << admitted.family->describe(admitted.info) << '\n';
    }

    if (failures == 0) std::cout << "qwen35 config tests passed\n";
    return failures == 0 ? 0 : 1;
} catch (const std::exception& e) {
    std::cerr << "FAIL: unexpected exception: " << e.what() << '\n';
    return 1;
}
