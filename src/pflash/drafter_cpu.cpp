#include "strix/pflash/drafter.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "strix/gguf_reader.hpp"

namespace strix::pflash {
namespace {

using serve::pflash::ScoreBuffers;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// ---- arithmetic: fixed order, double accumulators ------------------------------------------------

// Four interleaved partial sums, combined as (s0 + s1) + (s2 + s3), then the tail in order.
double dot(const float* a, const float* b, std::size_t n) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        s1 += static_cast<double>(a[i + 1]) * static_cast<double>(b[i + 1]);
        s2 += static_cast<double>(a[i + 2]) * static_cast<double>(b[i + 2]);
        s3 += static_cast<double>(a[i + 3]) * static_cast<double>(b[i + 3]);
    }
    double sum = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    return sum;
}

// y[rows][out] = x[rows][in] * w[out][in]^T
std::vector<float> matmul(const std::vector<float>& w, const std::vector<float>& x, std::size_t in, std::size_t out) {
    const std::size_t rows = x.size() / in;
    std::vector<float> y(rows * out);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t o = 0; o < out; ++o) y[r * out + o] = static_cast<float>(dot(&w[o * in], &x[r * in], in));
    return y;
}

// In place over consecutive groups of `width`: x * weight / sqrt(mean(x^2) + eps).
void rms_norm(float* x, std::size_t groups, std::size_t width, const float* weight, double eps) {
    for (std::size_t g = 0; g < groups; ++g) {
        float* row = x + g * width;
        const double scale = 1.0 / std::sqrt(dot(row, row, width) / static_cast<double>(width) + eps);
        for (std::size_t i = 0; i < width; ++i)
            row[i] = static_cast<float>(static_cast<double>(row[i]) * scale * static_cast<double>(weight[i]));
    }
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double silu(double x) { return x * sigmoid(x); }
double softplus(double x) { return x > 30.0 ? x : std::log1p(std::exp(x)); }

void add_into(std::vector<float>& x, const std::vector<float>& delta) {
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += delta[i];
}

// ---- blocks --------------------------------------------------------------------------------------

std::vector<float> delta_block(const DrafterConfig& c, const DrafterBlock& block, const std::vector<float>& x) {
    const DeltaBlock& w = block.delta;
    const std::size_t hidden = c.hidden, heads = c.delta_heads, dim = c.delta_head_dim, taps = c.conv_kernel;
    const std::size_t width = heads * dim, channels = 3 * width, tokens = x.size() / hidden;
    const double eps = static_cast<double>(c.rms_eps);

    std::vector<float> h = x;
    rms_norm(h.data(), tokens, hidden, block.attn_norm.data(), eps);
    const std::vector<float> qkv = matmul(w.qkv, h, hidden, channels);
    const std::vector<float> gate = matmul(w.gate, h, hidden, width);
    const std::vector<float> beta = matmul(w.beta, h, hidden, heads);
    const std::vector<float> alpha = matmul(w.alpha, h, hidden, heads);

    // Causal depthwise convolution, oldest tap first, then SiLU.
    std::vector<float> mixed(tokens * channels);
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t ch = 0; ch < channels; ++ch) {
            double sum = 0.0;
            for (std::size_t i = 0; i < taps; ++i) {
                if (t + i + 1 < taps) continue;
                sum += static_cast<double>(w.conv[ch * taps + i]) *
                       static_cast<double>(qkv[(t + i + 1 - taps) * channels + ch]);
            }
            mixed[t * channels + ch] = static_cast<float>(silu(sum));
        }
    // Unit-length queries and keys per head; queries also carry 1 / sqrt(dim).
    const double q_scale = 1.0 / std::sqrt(static_cast<double>(dim));
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t g = 0; g < 2 * heads; ++g) {
            float* v = &mixed[t * channels + g * dim];
            const double scale = (g < heads ? q_scale : 1.0) / std::sqrt(dot(v, v, dim) + eps);
            for (std::size_t i = 0; i < dim; ++i) v[i] = static_cast<float>(static_cast<double>(v[i]) * scale);
        }

    std::vector<float> core(tokens * width);
    std::vector<double> state(dim * dim), delta(dim), read(dim);
    for (std::size_t head = 0; head < heads; ++head) {
        std::fill(state.begin(), state.end(), 0.0);  // [key][value]
        for (std::size_t t = 0; t < tokens; ++t) {
            const float* row = &mixed[t * channels];
            const float* query = row + head * dim;
            const float* key = row + width + head * dim;
            const float* value = row + 2 * width + head * dim;
            const double decay = std::exp(static_cast<double>(w.a[head]) *
                                          softplus(static_cast<double>(alpha[t * heads + head]) +
                                                   static_cast<double>(w.dt_bias[head])));
            const double strength = sigmoid(static_cast<double>(beta[t * heads + head]));
            std::fill(delta.begin(), delta.end(), 0.0);
            for (std::size_t k = 0; k < dim; ++k) {
                double* s = &state[k * dim];
                const double kk = static_cast<double>(key[k]);
                for (std::size_t v = 0; v < dim; ++v) {
                    s[v] *= decay;
                    delta[v] += s[v] * kk;
                }
            }
            for (std::size_t v = 0; v < dim; ++v) delta[v] = (static_cast<double>(value[v]) - delta[v]) * strength;
            std::fill(read.begin(), read.end(), 0.0);
            for (std::size_t k = 0; k < dim; ++k) {
                double* s = &state[k * dim];
                const double kk = static_cast<double>(key[k]), qq = static_cast<double>(query[k]);
                for (std::size_t v = 0; v < dim; ++v) {
                    s[v] += kk * delta[v];
                    read[v] += s[v] * qq;
                }
            }
            for (std::size_t v = 0; v < dim; ++v) core[t * width + head * dim + v] = static_cast<float>(read[v]);
        }
    }
    rms_norm(core.data(), tokens * heads, dim, w.norm.data(), eps);
    for (std::size_t i = 0; i < core.size(); ++i)
        core[i] = static_cast<float>(static_cast<double>(core[i]) * silu(static_cast<double>(gate[i])));
    return matmul(w.out, core, width, hidden);
}

// Rotates the leading rope_dims of each head: pair (i, i + rope_dims / 2), angle = position * base^(-i / half).
void rotate(const DrafterConfig& c, std::vector<float>& x, std::size_t tokens, std::size_t heads,
            std::size_t stride) {
    const std::size_t half = c.rope_dims / 2;
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t i = 0; i < half; ++i) {
            const double angle = static_cast<double>(t) * std::pow(static_cast<double>(c.rope_base),
                                                                   -static_cast<double>(i) / static_cast<double>(half));
            const double cs = std::cos(angle), sn = std::sin(angle);
            for (std::size_t head = 0; head < heads; ++head) {
                float* v = &x[(t * heads + head) * stride];
                const double a = static_cast<double>(v[i]), b = static_cast<double>(v[half + i]);
                v[i] = static_cast<float>(a * cs - b * sn);
                v[half + i] = static_cast<float>(b * cs + a * sn);
            }
        }
}

std::vector<float> attention_block(const DrafterConfig& c, const DrafterBlock& block, const std::vector<float>& x) {
    const AttentionBlock& w = block.attention;
    const std::size_t hidden = c.hidden, heads = c.heads, kv = c.kv_heads, dim = c.head_dim;
    const std::size_t tokens = x.size() / hidden, share = heads / kv;
    const double eps = static_cast<double>(c.rms_eps);

    std::vector<float> h = x;
    rms_norm(h.data(), tokens, hidden, block.attn_norm.data(), eps);
    std::vector<float> q = matmul(w.wq, h, hidden, heads * 2 * dim);  // per head: query, gate
    std::vector<float> k = matmul(w.wk, h, hidden, kv * dim);
    const std::vector<float> v = matmul(w.wv, h, hidden, kv * dim);
    for (std::size_t g = 0; g < tokens * heads; ++g) rms_norm(&q[g * 2 * dim], 1, dim, w.q_norm.data(), eps);
    rms_norm(k.data(), tokens * kv, dim, w.k_norm.data(), eps);
    rotate(c, q, tokens, heads, 2 * dim);
    rotate(c, k, tokens, kv, dim);

    const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
    std::vector<float> mixed(tokens * heads * dim);
    std::vector<double> p(tokens), acc(dim);
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t head = 0; head < heads; ++head) {
            const std::size_t group = head / share;
            const float* query = &q[(t * heads + head) * 2 * dim];
            double top = 0.0;
            for (std::size_t j = 0; j <= t; ++j) {
                p[j] = dot(query, &k[(j * kv + group) * dim], dim) * scale;
                if (j == 0 || p[j] > top) top = p[j];
            }
            double total = 0.0;
            for (std::size_t j = 0; j <= t; ++j) {
                p[j] = std::exp(p[j] - top);
                total += p[j];
            }
            std::fill(acc.begin(), acc.end(), 0.0);
            for (std::size_t j = 0; j <= t; ++j) {
                const float* value = &v[(j * kv + group) * dim];
                const double weight = p[j] / total;
                for (std::size_t d = 0; d < dim; ++d) acc[d] += weight * static_cast<double>(value[d]);
            }
            for (std::size_t d = 0; d < dim; ++d)
                mixed[(t * heads + head) * dim + d] =
                    static_cast<float>(acc[d] * sigmoid(static_cast<double>(query[dim + d])));
        }
    return matmul(w.wo, mixed, heads * dim, hidden);
}

std::vector<float> feed_forward(const DrafterConfig& c, const DrafterBlock& block, const std::vector<float>& x) {
    const std::size_t hidden = c.hidden, wide = c.feed_forward, tokens = x.size() / hidden;
    std::vector<float> h = x;
    rms_norm(h.data(), tokens, hidden, block.ffn_norm.data(), static_cast<double>(c.rms_eps));
    std::vector<float> gate = matmul(block.ffn_gate, h, hidden, wide);
    const std::vector<float> up = matmul(block.ffn_up, h, hidden, wide);
    for (std::size_t i = 0; i < gate.size(); ++i)
        gate[i] = static_cast<float>(silu(static_cast<double>(gate[i])) * static_cast<double>(up[i]));
    return matmul(block.ffn_down, gate, wide, hidden);
}

// ---- checks --------------------------------------------------------------------------------------

void expect(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("pflash drafter: ") + what);
}

void expect_size(const std::vector<float>& v, u64 rows, u64 columns, const char* what) {
    expect(static_cast<u64>(v.size()) == rows * columns, what);
}

// ---- GGUF ----------------------------------------------------------------------------------------

[[noreturn]] void fail(const std::filesystem::path& path, const std::string& what) {
    throw std::runtime_error("pflash drafter " + path.filename().string() + ": " + what);
}

constexpr u32 kTypeF32 = 0, kTypeF16 = 1, kTypeBf16 = 30;

float half_to_float(std::uint16_t bits) {
    const u32 sign = static_cast<u32>(bits & 0x8000U) << 16;
    u32 exponent = (bits >> 10) & 0x1FU, mantissa = bits & 0x3FFU, out = 0;
    if (exponent == 0x1FU) {
        out = sign | 0x7F800000U | (mantissa << 13);
    } else if (exponent != 0) {
        out = sign | ((exponent + 112U) << 23) | (mantissa << 13);
    } else if (mantissa != 0) {
        exponent = 113U;
        while ((mantissa & 0x400U) == 0) {
            mantissa <<= 1;
            --exponent;
        }
        out = sign | (exponent << 23) | ((mantissa & 0x3FFU) << 13);
    } else {
        out = sign;
    }
    float value = 0.0f;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

class Reader {
public:
    explicit Reader(const std::filesystem::path& path)
        : path_(path), info_(gguf::read_file(path)), input_(path, std::ios::binary) {
        if (!input_) fail(path_, "cannot open");
        const gguf::MetadataEntry* arch = info_.find_metadata("general.architecture");
        if (arch == nullptr || !arch->string_value) fail(path_, "general.architecture missing");
        prefix_ = *arch->string_value + ".";
    }

    [[nodiscard]] u32 number(const std::string& key) const {
        const gguf::MetadataEntry* entry = info_.find_metadata(prefix_ + key);
        if (entry == nullptr) fail(path_, "key " + key + " missing");
        u64 value = 0;
        if (entry->unsigned_value) {
            value = *entry->unsigned_value;
        } else if (entry->signed_value && *entry->signed_value >= 0) {
            value = static_cast<u64>(*entry->signed_value);
        } else {
            fail(path_, "key " + key + " is not a count");
        }
        if (value == 0 || value > 0x7FFFFFFFULL) fail(path_, "key " + key + " out of range");
        return static_cast<u32>(value);
    }

    [[nodiscard]] float real(const std::string& key) const {
        const gguf::MetadataEntry* entry = info_.find_metadata(prefix_ + key);
        if (entry == nullptr || !entry->floating_value) fail(path_, "key " + key + " missing or not a float");
        const double value = *entry->floating_value;
        if (!(value > 0.0) || !std::isfinite(value)) fail(path_, "key " + key + " out of range");
        return static_cast<float>(value);
    }

    [[nodiscard]] const gguf::TensorInfo& info(const std::string& name) const {
        for (const gguf::TensorInfo& tensor : info_.tensors)
            if (tensor.name == name) return tensor;
        fail(path_, "tensor " + name + " missing");
    }

    // Matrix [out][in] (GGUF dimensions {in, out}) or vector (out == 1, GGUF dimensions {in}).
    [[nodiscard]] std::vector<float> tensor(const std::string& name, u64 out, u64 in) {
        const gguf::TensorInfo& t = info(name);
        const bool shape_ok = out == 1 ? (t.dimensions.size() == 1 && t.dimensions[0] == in)
                                       : (t.dimensions.size() == 2 && t.dimensions[0] == in && t.dimensions[1] == out);
        if (!shape_ok) fail(path_, "tensor " + name + " has an unexpected shape");
        return values(t, 0, out * in);
    }

    // `count` values starting at element `first`.
    [[nodiscard]] std::vector<float> values(const gguf::TensorInfo& t, u64 first, u64 count) {
        const u64 width = t.ggml_type == kTypeF32 ? 4U : (t.ggml_type == kTypeF16 || t.ggml_type == kTypeBf16) ? 2U : 0U;
        if (width == 0) fail(path_, "tensor " + t.name + " has an unsupported type");
        if ((first + count) * width > t.byte_size) fail(path_, "tensor " + t.name + " is too short");
        std::vector<float> out(static_cast<std::size_t>(count));
        input_.clear();
        input_.seekg(static_cast<std::streamoff>(t.absolute_offset + first * width));
        if (t.ggml_type == kTypeF32) {
            input_.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * 4U));
        } else {
            std::vector<std::uint16_t> raw(static_cast<std::size_t>(count));
            input_.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(count * 2U));
            for (std::size_t i = 0; i < raw.size(); ++i) {
                if (t.ggml_type == kTypeF16) {
                    out[i] = half_to_float(raw[i]);
                } else {
                    const u32 bits = static_cast<u32>(raw[i]) << 16;
                    std::memcpy(&out[i], &bits, sizeof(bits));
                }
            }
        }
        if (!input_) fail(path_, "tensor " + t.name + " cannot be read");
        return out;
    }

private:
    std::filesystem::path path_;
    gguf::FileInfo info_;
    std::ifstream input_;
    std::string prefix_;
};

}  // namespace

void validate(const DrafterWeights& weights) {
    const DrafterConfig& c = weights.config;
    expect(c.hidden > 0 && c.feed_forward > 0 && c.blocks > 0 && c.full_attention_interval > 0, "empty config");
    expect(c.heads > 0 && c.kv_heads > 0 && c.heads % c.kv_heads == 0 && c.head_dim > 0, "bad attention heads");
    expect(c.rope_dims % 2 == 0 && c.rope_dims <= c.head_dim && (c.rope_dims == 0 || c.rope_base > 0.0f),
           "bad rotary config");
    expect(c.delta_heads > 0 && c.delta_head_dim > 0 && c.conv_kernel > 0, "bad delta-rule config");
    expect(c.rms_eps > 0.0f, "bad epsilon");
    expect(weights.blocks.size() == c.blocks, "block count");
    const u64 hidden = c.hidden, wide = c.feed_forward, width = static_cast<u64>(c.delta_heads) * c.delta_head_dim;
    const u64 attn = static_cast<u64>(c.heads) * c.head_dim, attn_kv = static_cast<u64>(c.kv_heads) * c.head_dim;
    for (std::size_t b = 0; b < weights.blocks.size(); ++b) {
        const DrafterBlock& block = weights.blocks[b];
        expect(block.full_attention == ((b + 1) % c.full_attention_interval == 0), "block kind");
        expect_size(block.attn_norm, 1, hidden, "attn_norm");
        if (block.full_attention) {
            expect_size(block.attention.wq, 2 * attn, hidden, "attention wq");
            expect_size(block.attention.wk, attn_kv, hidden, "attention wk");
            expect_size(block.attention.wv, attn_kv, hidden, "attention wv");
            expect_size(block.attention.wo, hidden, attn, "attention wo");
            expect_size(block.attention.q_norm, 1, c.head_dim, "attention q_norm");
            expect_size(block.attention.k_norm, 1, c.head_dim, "attention k_norm");
        } else {
            expect_size(block.delta.qkv, 3 * width, hidden, "delta qkv");
            expect_size(block.delta.gate, width, hidden, "delta gate");
            expect_size(block.delta.beta, c.delta_heads, hidden, "delta beta");
            expect_size(block.delta.alpha, c.delta_heads, hidden, "delta alpha");
            expect_size(block.delta.conv, 3 * width, c.conv_kernel, "delta conv");
            expect_size(block.delta.dt_bias, 1, c.delta_heads, "delta dt_bias");
            expect_size(block.delta.a, 1, c.delta_heads, "delta a");
            expect_size(block.delta.norm, 1, c.delta_head_dim, "delta norm");
            expect_size(block.delta.out, hidden, width, "delta out");
        }
        expect_size(block.ffn_norm, 1, hidden, "ffn_norm");
        expect_size(block.ffn_gate, wide, hidden, "ffn_gate");
        expect_size(block.ffn_up, wide, hidden, "ffn_up");
        expect_size(block.ffn_down, hidden, wide, "ffn_down");
    }
    const serve::pflash::QkHead& s = weights.scorer;
    expect(s.hidden == c.hidden && s.heads == c.heads && s.kv_heads == c.kv_heads && s.head_dim == c.head_dim,
           "scorer shape");
    serve::pflash::validate(s);
}

DrafterWeights load_drafter(const std::filesystem::path& gguf) {
    Reader reader(gguf);
    DrafterWeights weights;
    DrafterConfig& c = weights.config;
    c.hidden = reader.number("embedding_length");
    c.feed_forward = reader.number("feed_forward_length");
    c.full_attention_interval = reader.number("full_attention_interval");
    c.heads = reader.number("attention.head_count");
    c.kv_heads = reader.number("attention.head_count_kv");
    c.head_dim = reader.number("attention.key_length");
    c.rope_dims = reader.number("rope.dimension_count");
    c.rope_base = reader.real("rope.freq_base");
    c.delta_heads = reader.number("ssm.group_count");
    c.delta_head_dim = reader.number("ssm.state_size");
    c.conv_kernel = reader.number("ssm.conv_kernel");
    c.rms_eps = reader.real("attention.layer_norm_rms_epsilon");
    if (reader.number("block_count") <= c.blocks) fail(gguf, "too few blocks for the scoring block");
    if (c.heads % c.kv_heads != 0) fail(gguf, "head counts do not divide");
    if ((c.blocks + 1) % c.full_attention_interval != 0) fail(gguf, "the scoring block is not a full attention block");
    const gguf::TensorInfo& table = reader.info("token_embd.weight");
    if (table.dimensions.size() != 2 || table.dimensions[0] != c.hidden || table.dimensions[1] > 0x7FFFFFFFULL)
        fail(gguf, "token table has an unexpected shape");
    c.vocabulary = static_cast<u32>(table.dimensions[1]);

    const u64 hidden = c.hidden, wide = c.feed_forward, width = static_cast<u64>(c.delta_heads) * c.delta_head_dim;
    const u64 attn = static_cast<u64>(c.heads) * c.head_dim, attn_kv = static_cast<u64>(c.kv_heads) * c.head_dim;
    weights.blocks.resize(c.blocks);
    for (u32 b = 0; b < c.blocks; ++b) {
        const std::string p = "blk." + std::to_string(b) + ".";
        DrafterBlock& block = weights.blocks[b];
        block.full_attention = (b + 1) % c.full_attention_interval == 0;
        block.attn_norm = reader.tensor(p + "attn_norm.weight", 1, hidden);
        if (block.full_attention) {
            block.attention.wq = reader.tensor(p + "attn_q.weight", 2 * attn, hidden);
            block.attention.wk = reader.tensor(p + "attn_k.weight", attn_kv, hidden);
            block.attention.wv = reader.tensor(p + "attn_v.weight", attn_kv, hidden);
            block.attention.wo = reader.tensor(p + "attn_output.weight", hidden, attn);
            block.attention.q_norm = reader.tensor(p + "attn_q_norm.weight", 1, c.head_dim);
            block.attention.k_norm = reader.tensor(p + "attn_k_norm.weight", 1, c.head_dim);
        } else {
            block.delta.qkv = reader.tensor(p + "attn_qkv.weight", 3 * width, hidden);
            block.delta.gate = reader.tensor(p + "attn_gate.weight", width, hidden);
            block.delta.beta = reader.tensor(p + "ssm_beta.weight", c.delta_heads, hidden);
            block.delta.alpha = reader.tensor(p + "ssm_alpha.weight", c.delta_heads, hidden);
            block.delta.conv = reader.tensor(p + "ssm_conv1d.weight", 3 * width, c.conv_kernel);
            block.delta.dt_bias = reader.tensor(p + "ssm_dt.bias", 1, c.delta_heads);
            block.delta.a = reader.tensor(p + "ssm_a", 1, c.delta_heads);
            block.delta.norm = reader.tensor(p + "ssm_norm.weight", 1, c.delta_head_dim);
            block.delta.out = reader.tensor(p + "ssm_out.weight", hidden, width);
        }
        block.ffn_norm = reader.tensor(p + "post_attention_norm.weight", 1, hidden);
        block.ffn_gate = reader.tensor(p + "ffn_gate.weight", wide, hidden);
        block.ffn_up = reader.tensor(p + "ffn_up.weight", wide, hidden);
        block.ffn_down = reader.tensor(p + "ffn_down.weight", hidden, wide);
    }
    const std::string p = "blk." + std::to_string(c.blocks) + ".";
    serve::pflash::QkHead& s = weights.scorer;
    s.hidden = c.hidden;
    s.heads = c.heads;
    s.kv_heads = c.kv_heads;
    s.head_dim = c.head_dim;
    s.q_rows_per_head = 2 * c.head_dim;
    s.rms_eps = c.rms_eps;
    s.attn_norm = reader.tensor(p + "attn_norm.weight", 1, hidden);
    s.wq = reader.tensor(p + "attn_q.weight", 2 * attn, hidden);
    s.wk = reader.tensor(p + "attn_k.weight", attn_kv, hidden);
    s.q_norm = reader.tensor(p + "attn_q_norm.weight", 1, c.head_dim);
    s.k_norm = reader.tensor(p + "attn_k_norm.weight", 1, c.head_dim);
    try {
        validate(weights);
    } catch (const std::invalid_argument& error) {
        fail(gguf, error.what());
    }
    return weights;
}

std::vector<float> load_embeddings(const std::filesystem::path& gguf, std::span<const std::uint32_t> tokens) {
    Reader reader(gguf);
    const gguf::TensorInfo& table = reader.info("token_embd.weight");
    if (table.dimensions.size() != 2) fail(gguf, "token table has an unexpected shape");
    const u64 hidden = table.dimensions[0], vocabulary = table.dimensions[1];
    std::vector<float> out;
    out.reserve(tokens.size() * static_cast<std::size_t>(hidden));
    for (const std::uint32_t token : tokens) {
        if (token >= vocabulary) fail(gguf, "token outside the vocabulary");
        const std::vector<float> row = reader.values(table, static_cast<u64>(token) * hidden, hidden);
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}

std::vector<float> forward(const DrafterWeights& weights, std::span<const float> embedded) {
    validate(weights);
    const DrafterConfig& c = weights.config;
    expect(embedded.size() % c.hidden == 0, "embedded prompt size");
    std::vector<float> x(embedded.begin(), embedded.end());
    for (const DrafterBlock& block : weights.blocks) {
        add_into(x, block.full_attention ? attention_block(c, block, x) : delta_block(c, block, x));
        add_into(x, feed_forward(c, block, x));
    }
    return x;
}

ScoreBuffers score_residual(const DrafterWeights& weights, std::span<const float> residual, QueryWindow current,
                            std::span<const QueryWindow> history, std::span<const std::uint32_t> token_byte_begin,
                            const ScoreOptions& options) {
    namespace sp = serve::pflash;
    const DrafterConfig& c = weights.config;
    expect(c.hidden > 0 && residual.size() % c.hidden == 0, "residual size");
    const u64 count = residual.size() / c.hidden;
    expect(count > 0 && count <= 0xFFFFFFFFULL, "token count");
    const u32 tokens = static_cast<u32>(count);
    expect(token_byte_begin.size() == tokens, "token_byte_begin size");
    expect(options.request_cap > 0, "request_cap");
    const auto inside = [tokens](QueryWindow w) { return w.begin < w.end && w.end <= tokens; };
    expect(inside(current), "current window");
    for (const QueryWindow w : history) expect(inside(w), "history window");

    sp::QkHead head = weights.scorer;
    if (options.scoring_head != nullptr) {
        const sp::ScoringHeadFile& file = *options.scoring_head;
        expect(file.hidden == head.hidden && file.k_rows == head.kv_heads * head.head_dim &&
                   file.q_rows % head.heads == 0,
               "scoring head shape");
        head.q_rows_per_head = file.q_rows / head.heads;
        head.wq = file.wq;
        head.wk = file.wk;
    }
    sp::validate(head);
    const std::vector<float> keys = sp::qk_keys(head, residual);

    const auto window_rows = [&options](QueryWindow w) {
        std::vector<u32> rows;
        const u32 first = w.end - w.begin > options.request_cap ? w.end - options.request_cap : w.begin;
        for (u32 t = first; t < w.end; ++t) rows.push_back(t);
        return rows;
    };

    std::vector<u32> rows;
    if (options.request_probe != nullptr) {
        expect(options.message_of_token.size() == tokens, "message_of_token size");
        const std::vector<float> raw = sp::request_logits(*options.request_probe, residual);
        std::vector<std::uint8_t> runs = sp::request_runs(raw, options.request_probe->tau, options.message_of_token);
        for (u32 t = 0; t < tokens; ++t)
            if (t < current.begin || t >= current.end) runs[t] = 0;
        rows = sp::request_rows(runs, options.request_cap);
    }
    std::vector<std::uint8_t> allowed;
    if (rows.empty()) {  // no probe, or the probe found no request run in the window
        rows = window_rows(current);
        allowed = sp::window_key_mask(tokens, current.begin, current.end, options.suffix_candidates);
    } else {
        allowed = sp::request_key_mask(tokens, rows);
    }

    std::vector<std::vector<float>> masses;
    masses.push_back(sp::qk_mass(head, residual, keys, rows, allowed));
    for (const QueryWindow w : history) {
        const std::vector<u32> past = window_rows(w);
        masses.push_back(sp::qk_mass(head, residual, keys, past,
                                     sp::window_key_mask(tokens, w.begin, w.end, options.suffix_candidates)));
    }
    std::vector<sp::WeightedMass> parts;
    for (std::size_t i = 0; i < masses.size(); ++i)
        parts.push_back({masses[i], i == 0 ? 1.0f : sp::history_weight(static_cast<u32>(i - 1))});

    ScoreBuffers out;
    out.token_score = sp::mix_masses(parts);
    out.cut_prob = options.segment_probe != nullptr ? sp::segment_cut_prob(*options.segment_probe, residual)
                                                    : std::vector<float>(tokens, 0.0f);
    out.request_mask.assign(tokens, 0);
    for (const u32 row : rows) out.request_mask[row] = 1;
    out.token_byte_begin.assign(token_byte_begin.begin(), token_byte_begin.end());
    return out;
}

CpuDrafterScorer::CpuDrafterScorer(DrafterWeights weights, Tokenize tokenize, Embed embed, ScoreOptions options)
    : weights_(std::move(weights)), tokenize_(std::move(tokenize)), embed_(std::move(embed)), options_(options) {
    validate(weights_);
    expect(static_cast<bool>(tokenize_) && static_cast<bool>(embed_), "tokenizer or embedding source missing");
}

ScoreBuffers CpuDrafterScorer::score(std::string_view rendered, std::uint64_t /*session*/) {
    const TokenizedPrompt prompt = tokenize_(rendered);
    const std::vector<float> embedded = embed_(prompt.tokens);
    expect(embedded.size() == prompt.tokens.size() * static_cast<std::size_t>(weights_.config.hidden),
           "embedding size");
    const std::vector<float> residual = forward(weights_, embedded);
    return score_residual(weights_, residual, prompt.current, prompt.history, prompt.token_byte_begin, options_);
}

}  // namespace strix::pflash
