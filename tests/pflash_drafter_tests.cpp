// Host tests of the drafter CPU reference (blocks 0-14 and the block-15 QK mass).
//
// The goldens come from tools/pflash_ref.py, an independent numpy implementation:
//   - synthetic: weights from a small generator repeated here, so the check needs no file;
//   - public: the public drafter weights. That case runs only when STRIX_PFLASH_DRAFTER_GGUF names the
//     file; the weights are not part of the repository.
#include "strix/pflash/drafter.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

#include "pflash_drafter_public_golden.inc"
#include "pflash_drafter_synthetic_golden.inc"

namespace {

namespace dr = strix::pflash;
namespace sp = strix::serve::pflash;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

template <typename Fn>
void throws(Fn&& fn, const std::string& what) {
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    expect(false, what + ": no exception");
}

// Largest |got - want| / (|want| + floor) over the arrays.
template <typename Got, typename Want>
double worst(const Got& got, const Want& want, std::size_t count, double floor) {
    double out = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double g = static_cast<double>(got[i]), w = static_cast<double>(want[i]);
        const double error = std::fabs(g - w) / (std::fabs(w) + floor);
        if (!(error <= out)) out = error;  // NaN sticks
    }
    return out;
}

// ---- synthetic weights: the generator of tools/pflash_ref.py, same tensor order ------------------

struct Synthetic {
    dr::DrafterWeights weights;
    std::vector<float> table;  // [vocabulary][hidden]
};

class Filler {
public:
    std::vector<float> operator()(u64 count, double base, double span) {
        ++counter_;
        u32 state = 2654435769U * counter_;
        std::vector<float> out(static_cast<std::size_t>(count));
        for (float& value : out) {
            state = 1664525U * state + 1013904223U;
            value = static_cast<float>(base + (static_cast<double>(state >> 8) / 16777216.0 - 0.5) * span);
        }
        return out;
    }

private:
    u32 counter_ = 0;
};

Synthetic make_synthetic() {
    Synthetic s;
    dr::DrafterConfig& c = s.weights.config;
    c.hidden = 32;
    c.feed_forward = 64;
    c.vocabulary = 64;
    c.blocks = 15;
    c.full_attention_interval = 4;
    c.heads = 4;
    c.kv_heads = 2;
    c.head_dim = 8;
    c.rope_dims = 4;
    c.rope_base = 1.0e7f;
    c.delta_heads = 4;
    c.delta_head_dim = 8;
    c.conv_kernel = 4;
    c.rms_eps = 1.0e-6f;
    const u64 hidden = c.hidden, wide = c.feed_forward, width = u64{c.delta_heads} * c.delta_head_dim;
    const u64 attn = u64{c.heads} * c.head_dim, attn_kv = u64{c.kv_heads} * c.head_dim;
    Filler fill;
    s.table = fill(c.vocabulary * hidden, 0.0, 2.0);
    s.weights.blocks.resize(c.blocks);
    for (u32 b = 0; b < c.blocks; ++b) {
        dr::DrafterBlock& block = s.weights.blocks[b];
        block.full_attention = (b + 1) % c.full_attention_interval == 0;
        block.attn_norm = fill(hidden, 1.0, 0.5);
        if (block.full_attention) {
            block.attention.wq = fill(2 * attn * hidden, 0.0, 0.5);
            block.attention.wk = fill(attn_kv * hidden, 0.0, 0.5);
            block.attention.wv = fill(attn_kv * hidden, 0.0, 0.25);
            block.attention.wo = fill(hidden * attn, 0.0, 0.25);
            block.attention.q_norm = fill(c.head_dim, 1.0, 0.5);
            block.attention.k_norm = fill(c.head_dim, 1.0, 0.5);
        } else {
            block.delta.qkv = fill(3 * width * hidden, 0.0, 0.5);
            block.delta.gate = fill(width * hidden, 0.0, 0.5);
            block.delta.beta = fill(c.delta_heads * hidden, 0.0, 0.5);
            block.delta.alpha = fill(c.delta_heads * hidden, 0.0, 0.5);
            block.delta.conv = fill(3 * width * c.conv_kernel, 0.0, 1.0);
            block.delta.dt_bias = fill(c.delta_heads, 0.0, 2.0);
            block.delta.a = fill(c.delta_heads, -0.75, 1.0);
            block.delta.norm = fill(c.delta_head_dim, 1.0, 0.5);
            block.delta.out = fill(hidden * width, 0.0, 0.25);
        }
        block.ffn_norm = fill(hidden, 1.0, 0.5);
        block.ffn_gate = fill(wide * hidden, 0.0, 0.25);
        block.ffn_up = fill(wide * hidden, 0.0, 0.25);
        block.ffn_down = fill(hidden * wide, 0.0, 0.25);
    }
    sp::QkHead& q = s.weights.scorer;
    q.hidden = c.hidden;
    q.heads = c.heads;
    q.kv_heads = c.kv_heads;
    q.head_dim = c.head_dim;
    q.q_rows_per_head = 2 * c.head_dim;
    q.rms_eps = c.rms_eps;
    q.attn_norm = fill(hidden, 1.0, 0.5);
    q.wq = fill(2 * attn * hidden, 0.0, 0.5);
    q.wk = fill(attn_kv * hidden, 0.0, 0.5);
    q.q_norm = fill(c.head_dim, 1.0, 0.5);
    q.k_norm = fill(c.head_dim, 1.0, 0.5);
    return s;
}

std::vector<float> embed(const Synthetic& s, const std::vector<u32>& tokens) {
    const std::size_t hidden = s.weights.config.hidden;
    std::vector<float> out;
    for (const u32 token : tokens)
        out.insert(out.end(), s.table.begin() + static_cast<std::ptrdiff_t>(token * hidden),
                   s.table.begin() + static_cast<std::ptrdiff_t>((token + 1) * hidden));
    return out;
}

std::vector<u32> byte_begins(std::size_t tokens) {
    std::vector<u32> out(tokens);
    for (std::size_t t = 0; t < tokens; ++t) out[t] = static_cast<u32>(3 * t);
    return out;
}

// ---- golden comparison ---------------------------------------------------------------------------

template <std::size_t N, std::size_t M, std::size_t K>
void check_golden(const std::string& name, const dr::DrafterWeights& weights, const std::vector<float>& embedded,
                  u32 query_begin, const double (&mass)[N], const double (&norms)[M], const double (&samples)[K],
                  double tolerance) {
    const std::size_t hidden = weights.config.hidden, tokens = embedded.size() / hidden;
    expect(N == tokens && M == tokens && K == 24, name + ": golden sizes");
    const std::vector<float> residual = dr::forward(weights, embedded);
    std::vector<double> got_norm(tokens), got_sample;
    for (std::size_t t = 0; t < tokens; ++t) {
        double sum = 0.0;
        for (std::size_t d = 0; d < hidden; ++d) sum += static_cast<double>(residual[t * hidden + d]) * residual[t * hidden + d];
        got_norm[t] = std::sqrt(sum);
    }
    for (const std::size_t t : {std::size_t{0}, tokens / 2, tokens - 1})
        for (std::size_t d = 0; d < 8; ++d) got_sample.push_back(residual[t * hidden + d]);
    const std::vector<u32> bytes = byte_begins(tokens);
    const sp::ScoreBuffers scores =
        dr::score_residual(weights, residual, {query_begin, static_cast<u32>(tokens)}, {}, bytes);
    const double e_norm = worst(got_norm, norms, tokens, 1e-3);
    const double e_sample = worst(got_sample, samples, 24, 1e-2);
    const double e_mass = worst(scores.token_score, mass, tokens, 1e-4);
    std::cout << name << ": tokens " << tokens << ", worst relative error: residual norm " << e_norm
              << ", residual sample " << e_sample << ", QK mass " << e_mass << '\n';
    expect(e_norm <= tolerance, name + ": residual norm");
    expect(e_sample <= tolerance, name + ": residual sample");
    expect(e_mass <= tolerance, name + ": QK mass");
}

// ---- GGUF writer (version 3, alignment 32) -------------------------------------------------------

enum class Store { f32, f16, bf16 };

struct Fault {
    std::string drop_tensor;
    std::string drop_key;
    std::string reshape_tensor;  // written with its two dimensions swapped
    bool unknown_type = false;
};

class Writer {
public:
    explicit Writer(Fault fault) : fault_(std::move(fault)) {}

    void count(const std::string& key, u32 value) {
        if (key == fault_.drop_key) return;
        string(keys_, "qwen35." + key);
        pod(keys_, u32{4});
        pod(keys_, value);
        ++key_count_;
    }
    void real(const std::string& key, float value) {
        if (key == fault_.drop_key) return;
        string(keys_, "qwen35." + key);
        pod(keys_, u32{6});
        pod(keys_, value);
        ++key_count_;
    }
    void tensor(const std::string& name, const std::vector<float>& values, u64 out, u64 in, Store store) {
        if (name == fault_.drop_tensor) return;
        std::vector<u64> dims = out == 1 ? std::vector<u64>{in} : std::vector<u64>{in, out};
        if (name == fault_.reshape_tensor && dims.size() == 2) std::swap(dims[0], dims[1]);
        const bool wide = store == Store::f32 || out == 1;  // vectors stay F32, as in the public file
        string(infos_, name);
        pod(infos_, static_cast<u32>(dims.size()));
        for (const u64 d : dims) pod(infos_, d);
        pod(infos_, fault_.unknown_type && name == "blk.0.ffn_up.weight" ? u32{7} : wide ? u32{0} : store == Store::f16 ? u32{1} : u32{30});
        pod(infos_, static_cast<u64>(data_.size()));
        for (const float value : values) {
            if (wide) {
                pod(data_, value);
            } else if (store == Store::bf16) {
                u32 bits = 0;
                std::memcpy(&bits, &value, sizeof(bits));
                pod(data_, static_cast<std::uint16_t>(bits >> 16));
            } else {
                pod(data_, half(value));
            }
        }
        while (data_.size() % 32 != 0) data_.push_back('\0');
        ++tensor_count_;
    }
    void write(const std::filesystem::path& path) const {
        std::string head = "GGUF";
        pod(head, u32{3});
        pod(head, tensor_count_);
        pod(head, key_count_ + 1);
        string(head, "general.architecture");
        pod(head, u32{8});
        string(head, "qwen35");
        head += keys_;
        head += infos_;
        while (head.size() % 32 != 0) head.push_back('\0');
        std::ofstream out(path, std::ios::binary);
        out.write(head.data(), static_cast<std::streamsize>(head.size()));
        out.write(data_.data(), static_cast<std::streamsize>(data_.size()));
    }

    // Exact for multiples of 1/64 below 16 in magnitude, which is all the F16 case writes.
    static std::uint16_t half(float value) {
        if (value == 0.0f) return 0;
        int exponent = 0;
        const double fraction = std::frexp(std::fabs(static_cast<double>(value)), &exponent);  // [0.5, 1)
        const auto mantissa = static_cast<u32>(fraction * 2048.0) & 0x3FFU;                    // 11 bits, top one implied
        return static_cast<std::uint16_t>((value < 0.0f ? 0x8000U : 0U) | (static_cast<u32>(exponent + 14) << 10) | mantissa);
    }

private:
    template <typename T>
    static void pod(std::string& out, T value) {
        out.append(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    static void string(std::string& out, const std::string& text) {
        pod(out, static_cast<u64>(text.size()));
        out += text;
    }

    Fault fault_;
    std::string keys_, infos_, data_;
    u64 key_count_ = 0, tensor_count_ = 0;
};

// What a file stores of a value: itself, its upper 16 bits, or the nearest multiple of 1/64.
float stored(float value, Store store) {
    if (store == Store::f32) return value;
    if (store == Store::f16) return static_cast<float>(std::nearbyint(static_cast<double>(value) * 64.0) / 64.0);
    u32 bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits &= 0xFFFF0000U;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

// Rounds the matrices of `s` to what `store` keeps, so the file round trip is exact.
void quantize(Synthetic& s, Store store) {
    const auto apply = [store](std::vector<float>& values) {
        for (float& v : values) v = stored(v, store);
    };
    apply(s.table);
    for (dr::DrafterBlock& b : s.weights.blocks) {
        for (std::vector<float>* m : {&b.attention.wq, &b.attention.wk, &b.attention.wv, &b.attention.wo, &b.delta.qkv,
                                      &b.delta.gate, &b.delta.beta, &b.delta.alpha, &b.delta.out, &b.ffn_gate,
                                      &b.ffn_up, &b.ffn_down})
            apply(*m);
        // The convolution is a matrix too ([channels][taps]).
        apply(b.delta.conv);
    }
    apply(s.weights.scorer.wq);
    apply(s.weights.scorer.wk);
}

void write_file(const Synthetic& s, const std::filesystem::path& path, Store store, const Fault& fault = {}) {
    const dr::DrafterConfig& c = s.weights.config;
    const u64 hidden = c.hidden, wide = c.feed_forward, width = u64{c.delta_heads} * c.delta_head_dim;
    const u64 attn = u64{c.heads} * c.head_dim, attn_kv = u64{c.kv_heads} * c.head_dim;
    Writer w(fault);
    w.count("block_count", 24);
    w.count("embedding_length", c.hidden);
    w.count("feed_forward_length", c.feed_forward);
    w.count("full_attention_interval", c.full_attention_interval);
    w.count("attention.head_count", c.heads);
    w.count("attention.head_count_kv", c.kv_heads);
    w.count("attention.key_length", c.head_dim);
    w.count("rope.dimension_count", c.rope_dims);
    w.real("rope.freq_base", c.rope_base);
    w.count("ssm.group_count", c.delta_heads);
    w.count("ssm.state_size", c.delta_head_dim);
    w.count("ssm.conv_kernel", c.conv_kernel);
    w.real("attention.layer_norm_rms_epsilon", c.rms_eps);
    w.tensor("token_embd.weight", s.table, c.vocabulary, hidden, store);
    for (u32 i = 0; i < c.blocks; ++i) {
        const dr::DrafterBlock& b = s.weights.blocks[i];
        const std::string p = "blk." + std::to_string(i) + ".";
        w.tensor(p + "attn_norm.weight", b.attn_norm, 1, hidden, store);
        if (b.full_attention) {
            w.tensor(p + "attn_q.weight", b.attention.wq, 2 * attn, hidden, store);
            w.tensor(p + "attn_k.weight", b.attention.wk, attn_kv, hidden, store);
            w.tensor(p + "attn_v.weight", b.attention.wv, attn_kv, hidden, store);
            w.tensor(p + "attn_output.weight", b.attention.wo, hidden, attn, store);
            w.tensor(p + "attn_q_norm.weight", b.attention.q_norm, 1, c.head_dim, store);
            w.tensor(p + "attn_k_norm.weight", b.attention.k_norm, 1, c.head_dim, store);
        } else {
            w.tensor(p + "attn_qkv.weight", b.delta.qkv, 3 * width, hidden, store);
            w.tensor(p + "attn_gate.weight", b.delta.gate, width, hidden, store);
            w.tensor(p + "ssm_beta.weight", b.delta.beta, c.delta_heads, hidden, store);
            w.tensor(p + "ssm_alpha.weight", b.delta.alpha, c.delta_heads, hidden, store);
            w.tensor(p + "ssm_conv1d.weight", b.delta.conv, 3 * width, c.conv_kernel, store);
            w.tensor(p + "ssm_dt.bias", b.delta.dt_bias, 1, c.delta_heads, store);
            w.tensor(p + "ssm_a", b.delta.a, 1, c.delta_heads, store);
            w.tensor(p + "ssm_norm.weight", b.delta.norm, 1, c.delta_head_dim, store);
            w.tensor(p + "ssm_out.weight", b.delta.out, hidden, width, store);
        }
        w.tensor(p + "post_attention_norm.weight", b.ffn_norm, 1, hidden, store);
        w.tensor(p + "ffn_gate.weight", b.ffn_gate, wide, hidden, store);
        w.tensor(p + "ffn_up.weight", b.ffn_up, wide, hidden, store);
        w.tensor(p + "ffn_down.weight", b.ffn_down, hidden, wide, store);
    }
    const std::string p = "blk." + std::to_string(c.blocks) + ".";
    const sp::QkHead& q = s.weights.scorer;
    w.tensor(p + "attn_norm.weight", q.attn_norm, 1, hidden, store);
    w.tensor(p + "attn_q.weight", q.wq, 2 * attn, hidden, store);
    w.tensor(p + "attn_k.weight", q.wk, attn_kv, hidden, store);
    w.tensor(p + "attn_q_norm.weight", q.q_norm, 1, c.head_dim, store);
    w.tensor(p + "attn_k_norm.weight", q.k_norm, 1, c.head_dim, store);
    w.write(path);
}

// ---- tests ---------------------------------------------------------------------------------------

std::vector<u32> synthetic_tokens() {
    return {std::begin(golden_synthetic::tokens), std::end(golden_synthetic::tokens)};
}

void test_synthetic_golden(const Synthetic& s) {
    check_golden("synthetic", s.weights, embed(s, synthetic_tokens()), golden_synthetic::query_begin,
                 golden_synthetic::mass, golden_synthetic::residual_norm, golden_synthetic::residual_sample, 2e-4);
}

void test_forward_properties(const Synthetic& s) {
    const std::vector<u32> tokens = synthetic_tokens();
    const std::size_t hidden = s.weights.config.hidden;
    const std::vector<float> embedded = embed(s, tokens);
    const std::vector<float> full = dr::forward(s.weights, embedded);
    expect(full == dr::forward(s.weights, embedded), "forward: same input, same output");
    // Causal: a prefix of the prompt gives the same rows, bit for bit.
    const std::size_t cut = 17;
    const std::vector<float> prefix = dr::forward(s.weights, std::span<const float>(embedded.data(), cut * hidden));
    expect(prefix.size() == cut * hidden && std::equal(prefix.begin(), prefix.end(), full.begin()), "forward: causal");
    // A later token changes nothing before it and something after it.
    std::vector<float> changed = embedded;
    for (std::size_t d = 0; d < hidden; ++d) changed[20 * hidden + d] += 0.25f;
    const std::vector<float> other = dr::forward(s.weights, changed);
    expect(std::equal(other.begin(), other.begin() + static_cast<std::ptrdiff_t>(20 * hidden), full.begin()),
           "forward: earlier rows unchanged");
    expect(!std::equal(other.begin() + static_cast<std::ptrdiff_t>(30 * hidden), other.end(),
                       full.begin() + static_cast<std::ptrdiff_t>(30 * hidden)),
           "forward: later rows depend on the change");
    expect(dr::forward(s.weights, {}).empty(), "forward: empty prompt");
    throws([&] { (void)dr::forward(s.weights, std::span<const float>(embedded.data(), hidden + 1)); },
           "forward: ragged input");
    dr::DrafterWeights broken = s.weights;
    broken.blocks[0].delta.conv.pop_back();
    throws([&] { dr::validate(broken); }, "validate: short tensor");
    broken = s.weights;
    broken.blocks[3].full_attention = false;
    throws([&] { dr::validate(broken); }, "validate: block kind");
    broken = s.weights;
    broken.config.rope_dims = 3;
    throws([&] { dr::validate(broken); }, "validate: odd rotary width");
}

void test_scoring(const Synthetic& s) {
    const std::vector<u32> tokens = synthetic_tokens();
    const u32 count = static_cast<u32>(tokens.size());
    const std::vector<float> residual = dr::forward(s.weights, embed(s, tokens));
    const std::vector<u32> bytes = byte_begins(count);
    const dr::QueryWindow current{31, count};

    const sp::ScoreBuffers base = dr::score_residual(s.weights, residual, current, {}, bytes);
    double before = 0.0, after = 0.0;
    bool mask_ok = true, cut_zero = true;
    for (u32 t = 0; t < count; ++t) {
        (t < current.begin ? before : after) += static_cast<double>(base.token_score[t]);
        mask_ok = mask_ok && (base.request_mask[t] != 0) == (t >= current.begin);
        cut_zero = cut_zero && base.cut_prob[t] == 0.0f;
    }
    expect(std::fabs(before - 1.0) < 1e-5 && after == 0.0, "score: the mass lies on the keys before the request");
    expect(mask_ok, "score: request mask is the window");
    expect(cut_zero, "score: no probe, no cut probability");
    expect(base.token_byte_begin == bytes, "score: byte offsets pass through");

    // The window's rows are capped from the end; the window still never scores itself.
    dr::ScoreOptions capped;
    capped.request_cap = 4;
    const sp::ScoreBuffers last4 = dr::score_residual(s.weights, residual, current, {}, bytes, capped);
    const std::vector<u32> rows{count - 4, count - 3, count - 2, count - 1};
    const std::vector<float> keys = sp::qk_keys(s.weights.scorer, residual);
    const std::vector<float> want = sp::qk_mass(s.weights.scorer, residual, keys, rows,
                                                sp::window_key_mask(count, current.begin, current.end, false));
    expect(last4.token_score == want, "score: request cap keeps the last rows");
    u32 set = 0;
    for (const std::uint8_t m : last4.request_mask) set += m;
    expect(set == 4 && last4.request_mask[count - 1] == 1, "score: capped request mask");

    // History windows mix at 0.5^(i + 1).
    const std::vector<dr::QueryWindow> history{{20, 24}, {8, 11}};
    const sp::ScoreBuffers mixed = dr::score_residual(s.weights, residual, current, history, bytes);
    std::vector<std::vector<float>> parts{base.token_score};
    for (const dr::QueryWindow w : history) {
        std::vector<u32> r;
        for (u32 t = w.begin; t < w.end; ++t) r.push_back(t);
        parts.push_back(sp::qk_mass(s.weights.scorer, residual, keys, r, sp::window_key_mask(count, w.begin, w.end, false)));
    }
    const std::vector<sp::WeightedMass> weighted{{parts[0], 1.0f}, {parts[1], 0.5f}, {parts[2], 0.25f}};
    expect(mixed.token_score == sp::mix_masses(weighted), "score: history mixing");

    // Tokens after an earlier window count only when asked.
    dr::ScoreOptions suffix;
    suffix.suffix_candidates = true;
    const sp::ScoreBuffers both = dr::score_residual(s.weights, residual, {8, 11}, {}, bytes, suffix);
    expect(both.token_score[20] > 0.0f && both.token_score[9] == 0.0f, "score: suffix candidates");

    // Optional replacement projections: the drafter's own query rows, repacked, give the same scores.
    const sp::QkHead& own = s.weights.scorer;
    sp::ScoringHeadFile head;
    head.hidden = own.hidden;
    head.q_rows = own.heads * own.head_dim;
    head.k_rows = own.kv_heads * own.head_dim;
    head.wk = own.wk;
    for (u32 h = 0; h < own.heads; ++h) {
        const auto first = own.wq.begin() + static_cast<std::ptrdiff_t>(u64{h} * own.q_rows_per_head * own.hidden);
        head.wq.insert(head.wq.end(), first, first + static_cast<std::ptrdiff_t>(u64{own.head_dim} * own.hidden));
    }
    dr::ScoreOptions with_head;
    with_head.scoring_head = &head;
    expect(dr::score_residual(s.weights, residual, current, {}, bytes, with_head).token_score == base.token_score,
           "score: optional scoring head");
    head.k_rows += 1;
    throws([&] { (void)dr::score_residual(s.weights, residual, current, {}, bytes, with_head); },
           "score: scoring head shape");

    throws([&] { (void)dr::score_residual(s.weights, residual, {31, count + 1}, {}, bytes); }, "score: window past the end");
    throws([&] { (void)dr::score_residual(s.weights, residual, {5, 5}, {}, bytes); }, "score: empty window");
    const std::vector<dr::QueryWindow> bad_history{{9, 3}};
    throws([&] { (void)dr::score_residual(s.weights, residual, current, bad_history, bytes); }, "score: bad history");
    throws([&] { (void)dr::score_residual(s.weights, residual, current, {}, std::span<const u32>(bytes.data(), 3)); },
           "score: byte offsets size");

    // The PromptScorer wrapper is the same pipeline.
    dr::CpuDrafterScorer scorer(
        s.weights,
        [&](std::string_view) {
            dr::TokenizedPrompt prompt;
            prompt.tokens = tokens;
            prompt.token_byte_begin = bytes;
            prompt.current = current;
            prompt.history = history;
            return prompt;
        },
        [&](std::span<const u32> ids) { return embed(s, {ids.begin(), ids.end()}); });
    const sp::ScoreBuffers wrapped = scorer.score("unused", 0);
    expect(wrapped.token_score == mixed.token_score && wrapped.request_mask == mixed.request_mask,
           "scorer: wrapper equals the direct call");
    throws([&] { dr::CpuDrafterScorer(s.weights, {}, {}); }, "scorer: missing callbacks");
}

bool same_weights(const dr::DrafterWeights& a, const dr::DrafterWeights& b) {
    bool ok = a.blocks.size() == b.blocks.size() && a.config.hidden == b.config.hidden &&
              a.config.feed_forward == b.config.feed_forward && a.config.vocabulary == b.config.vocabulary &&
              a.config.heads == b.config.heads && a.config.kv_heads == b.config.kv_heads &&
              a.config.head_dim == b.config.head_dim && a.config.rope_dims == b.config.rope_dims &&
              a.config.rope_base == b.config.rope_base && a.config.delta_heads == b.config.delta_heads &&
              a.config.delta_head_dim == b.config.delta_head_dim && a.config.conv_kernel == b.config.conv_kernel &&
              a.config.rms_eps == b.config.rms_eps;
    for (std::size_t i = 0; ok && i < a.blocks.size(); ++i) {
        const dr::DrafterBlock& x = a.blocks[i];
        const dr::DrafterBlock& y = b.blocks[i];
        ok = x.full_attention == y.full_attention && x.attn_norm == y.attn_norm && x.ffn_norm == y.ffn_norm &&
             x.ffn_gate == y.ffn_gate && x.ffn_up == y.ffn_up && x.ffn_down == y.ffn_down &&
             x.attention.wq == y.attention.wq && x.attention.wk == y.attention.wk &&
             x.attention.wv == y.attention.wv && x.attention.wo == y.attention.wo &&
             x.attention.q_norm == y.attention.q_norm && x.attention.k_norm == y.attention.k_norm &&
             x.delta.qkv == y.delta.qkv && x.delta.gate == y.delta.gate && x.delta.beta == y.delta.beta &&
             x.delta.alpha == y.delta.alpha && x.delta.conv == y.delta.conv && x.delta.dt_bias == y.delta.dt_bias &&
             x.delta.a == y.delta.a && x.delta.norm == y.delta.norm && x.delta.out == y.delta.out;
    }
    return ok && a.scorer.attn_norm == b.scorer.attn_norm && a.scorer.wq == b.scorer.wq && a.scorer.wk == b.scorer.wk &&
           a.scorer.q_norm == b.scorer.q_norm && a.scorer.k_norm == b.scorer.k_norm &&
           a.scorer.q_rows_per_head == b.scorer.q_rows_per_head && a.scorer.rms_eps == b.scorer.rms_eps;
}

void test_files(const Synthetic& s) {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("strix-pflash-drafter-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    const std::filesystem::path path = dir / "drafter.gguf";
    const std::vector<u32> tokens = synthetic_tokens();

    const std::pair<Store, const char*> stores[] = {{Store::f32, "F32"}, {Store::bf16, "BF16"}, {Store::f16, "F16"}};
    for (const auto& [store, name] : stores) {
        Synthetic q = s;
        quantize(q, store);
        write_file(q, path, store);
        try {
            const dr::DrafterWeights loaded = dr::load_drafter(path);
            expect(same_weights(loaded, q.weights), std::string("file ") + name + ": weights round trip");
            expect(dr::load_embeddings(path, tokens) == embed(q, tokens), std::string("file ") + name + ": token rows");
        } catch (const std::exception& error) {
            expect(false, std::string("file ") + name + ": " + error.what());
        }
    }
    // The F32 file reproduces the golden through the loader.
    write_file(s, path, Store::f32);
    check_golden("synthetic file", dr::load_drafter(path), dr::load_embeddings(path, tokens),
                 golden_synthetic::query_begin, golden_synthetic::mass, golden_synthetic::residual_norm,
                 golden_synthetic::residual_sample, 2e-4);
    const std::vector<u32> outside{s.weights.config.vocabulary};
    throws([&] { (void)dr::load_embeddings(path, outside); }, "file: token outside the vocabulary");

    std::vector<std::pair<std::string, Fault>> faults;
    for (const char* tensor : {"token_embd.weight", "blk.0.ssm_a", "blk.0.attn_qkv.weight", "blk.3.attn_v.weight",
                               "blk.14.ffn_down.weight", "blk.15.attn_q.weight", "blk.15.attn_k_norm.weight"}) {
        Fault f;
        f.drop_tensor = tensor;
        faults.emplace_back(std::string("missing ") + tensor, f);
    }
    for (const char* key : {"embedding_length", "full_attention_interval", "rope.freq_base", "ssm.state_size",
                            "attention.layer_norm_rms_epsilon", "block_count"}) {
        Fault f;
        f.drop_key = key;
        faults.emplace_back(std::string("missing key ") + key, f);
    }
    for (const char* tensor : {"blk.0.attn_qkv.weight", "blk.15.attn_k.weight", "blk.2.ssm_conv1d.weight"}) {
        Fault f;
        f.reshape_tensor = tensor;
        faults.emplace_back(std::string("reshaped ") + tensor, f);
    }
    Fault type;
    type.unknown_type = true;
    faults.emplace_back("unsupported tensor type", type);
    for (const auto& [name, fault] : faults) {
        write_file(s, path, Store::f32, fault);
        throws([&] { (void)dr::load_drafter(path); }, "file: " + name);
    }
    throws([&] { (void)dr::load_drafter(dir / "absent.gguf"); }, "file: absent");
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
}

// The public drafter. Only run when the weights are present on this machine.
void test_public_golden() {
    const char* path = std::getenv("STRIX_PFLASH_DRAFTER_GGUF");
    if (path == nullptr || *path == '\0') {
        std::cout << "public: skipped (STRIX_PFLASH_DRAFTER_GGUF is not set)\n";
        return;
    }
    try {
        const std::vector<u32> tokens{std::begin(golden_public::tokens), std::end(golden_public::tokens)};
        const dr::DrafterWeights weights = dr::load_drafter(path);
        check_golden("public", weights, dr::load_embeddings(path, tokens), golden_public::query_begin,
                     golden_public::mass, golden_public::residual_norm, golden_public::residual_sample, 2e-3);
    } catch (const std::exception& error) {
        expect(false, std::string("public: ") + error.what());
    }
}

}  // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif
    try {
        const Synthetic synthetic = make_synthetic();
        test_synthetic_golden(synthetic);
        test_forward_properties(synthetic);
        test_scoring(synthetic);
        test_files(synthetic);
        test_public_golden();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "pflash drafter tests passed\n";
    return 0;
}
