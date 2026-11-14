// Host tests of the prompt scorer reference. Every weight here is synthetic: closed-form cases, plus a
// seeded random case checked against an independent naive evaluation written in this file.
#include "strix/serve/prompt_scorer.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

namespace {

namespace pf = strix::serve::pflash;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

void near(double got, double want, double tolerance, const std::string& what) {
    if (!(std::fabs(got - want) <= tolerance)) {
        std::cerr << "FAIL: " << what << ": got " << got << ", want " << want << '\n';
        ++failures;
    }
}

template <typename Exception>
void throws(const std::function<void()>& body, const std::string& what) {
    try {
        body();
    } catch (const Exception&) {
        return;
    } catch (...) {
    }
    expect(false, what + " did not throw the expected exception");
}

// Deterministic values in [-1, 1).
struct Lcg {
    std::uint32_t state;
    float next() {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(state >> 8) / 8388608.0f - 1.0f;
    }
    std::vector<float> fill(std::size_t count) {
        std::vector<float> values(count);
        for (float& value : values) {
            value = next();
        }
        return values;
    }
};

pf::QkHead identity_head(std::uint32_t heads, std::uint32_t kv_heads) {
    pf::QkHead head;
    head.hidden = 2;
    head.heads = heads;
    head.kv_heads = kv_heads;
    head.head_dim = 2;
    head.q_rows_per_head = 2;
    head.rms_eps = 1e-12f;
    head.attn_norm = {1.0f, 1.0f};
    head.q_norm = {1.0f, 1.0f};
    head.k_norm = {1.0f, 1.0f};
    for (std::uint32_t h = 0; h < heads; ++h) {
        head.wq.insert(head.wq.end(), {1.0f, 0.0f, 0.0f, 1.0f});
    }
    for (std::uint32_t g = 0; g < kv_heads; ++g) {
        head.wk.insert(head.wk.end(), {1.0f, 0.0f, 0.0f, 1.0f});
    }
    return head;
}

void test_qk_closed_form() {
    // Tokens: a = (1,0), b = (0,1), query = (1,0). After both norms each vector has length sqrt(2), so the
    // scaled logits are sqrt(2) for a and 0 for b.
    const pf::QkHead head = identity_head(1, 1);
    const std::vector<float> residual{1.0f, 0.0f, 0.0f, 1.0f, 3.0f, 0.0f};
    const std::vector<float> keys = pf::qk_keys(head, residual);
    expect(keys.size() == 6, "key cache size");
    near(keys[0], std::sqrt(2.0), 1e-6, "key a.x");
    near(keys[3], std::sqrt(2.0), 1e-6, "key b.y");
    const std::vector<std::uint32_t> rows{2};
    const std::vector<std::uint8_t> allowed = pf::window_key_mask(3, 2, 3, false);
    const std::vector<float> mass = pf::qk_mass(head, residual, keys, rows, allowed);
    const double pa = 1.0 / (1.0 + std::exp(-std::sqrt(2.0)));
    near(mass[0], pa, 1e-6, "mass of the aligned key");
    near(mass[1], 1.0 - pa, 1e-6, "mass of the orthogonal key");
    expect(mass[2] == 0.0f, "the query window never scores itself");

    // Zero query projection: every allowed key gets the same mass.
    pf::QkHead flat = head;
    flat.wq.assign(flat.wq.size(), 0.0f);
    const std::vector<float> uniform = pf::qk_mass(flat, residual, keys, rows, allowed);
    near(uniform[0], 0.5, 1e-7, "uniform mass a");
    near(uniform[1], 0.5, 1e-7, "uniform mass b");

    // No query rows, or no allowed key: zeros.
    const std::vector<float> none = pf::qk_mass(head, residual, keys, {}, allowed);
    expect(none == std::vector<float>(3, 0.0f), "no query rows gives zeros");
    const std::vector<std::uint8_t> closed(3, 0);
    expect(pf::qk_mass(head, residual, keys, rows, closed) == std::vector<float>(3, 0.0f), "no allowed key gives zeros");
}

void test_qk_head_pairing_and_gate_half() {
    // 4 query heads over 2 key heads. Only query head 1 and key head 0 are non-zero, so the peaked
    // distribution appears only if head 1 reads key head 1 / (4 / 2) = 0.
    pf::QkHead head = identity_head(4, 2);
    for (std::size_t i = 0; i < head.wq.size(); ++i) {
        if (i / 4 != 1) {
            head.wq[i] = 0.0f;
        }
    }
    for (std::size_t i = 4; i < head.wk.size(); ++i) {
        head.wk[i] = 0.0f;
    }
    const std::vector<float> residual{1.0f, 0.0f, 0.0f, 1.0f, 3.0f, 0.0f};
    const std::vector<float> keys = pf::qk_keys(head, residual);
    const std::vector<std::uint32_t> rows{2};
    const std::vector<std::uint8_t> allowed = pf::request_key_mask(3, rows);
    const std::vector<float> mass = pf::qk_mass(head, residual, keys, rows, allowed);
    const double pa = 1.0 / (1.0 + std::exp(-std::sqrt(2.0)));
    near(mass[0], (3.0 * 0.5 + pa) / 4.0, 1e-6, "block pairing of query and key heads");
    near(static_cast<double>(mass[0]) + mass[1], 1.0, 1e-6, "mass sums to one");

    // Packed projection: a second half per head that must be ignored.
    pf::QkHead packed = head;
    packed.q_rows_per_head = 4;
    packed.wq.clear();
    for (std::uint32_t h = 0; h < 4; ++h) {
        for (std::size_t i = 0; i < 4; ++i) {
            packed.wq.push_back(head.wq[h * 4 + i]);
        }
        packed.wq.insert(packed.wq.end(), {7.0f, -3.0f, 5.0f, 11.0f});
    }
    expect(pf::qk_mass(packed, residual, keys, rows, allowed) == mass, "gate half of a packed projection is ignored");
}

// Independent evaluation of the QK mass definition: dense matrices first, then the softmax.
std::vector<double> naive_mass(const pf::QkHead& head, const std::vector<float>& residual,
                               const std::vector<std::uint32_t>& rows, const std::vector<std::uint8_t>& allowed) {
    const std::size_t hidden = head.hidden, dim = head.head_dim, heads = head.heads, kv = head.kv_heads;
    const std::size_t tokens = residual.size() / hidden;
    const auto rms = [](std::vector<double> v, const std::vector<float>& weight, double eps) {
        double square = 0.0;
        for (const double x : v) {
            square += x * x;
        }
        const double root = std::sqrt(square / static_cast<double>(v.size()) + eps);
        for (std::size_t i = 0; i < v.size(); ++i) {
            v[i] = static_cast<double>(static_cast<float>(v[i] / root * static_cast<double>(weight[i])));
        }
        return v;
    };
    const auto normed = [&](std::size_t t) {
        std::vector<double> x(residual.begin() + static_cast<std::ptrdiff_t>(t * hidden),
                              residual.begin() + static_cast<std::ptrdiff_t>((t + 1) * hidden));
        return rms(x, head.attn_norm, static_cast<double>(head.rms_eps));
    };
    const auto apply = [&](const std::vector<float>& matrix, std::size_t first_row, const std::vector<double>& in,
                           const std::vector<float>& weight) {
        std::vector<double> out(dim, 0.0);
        for (std::size_t d = 0; d < dim; ++d) {
            for (std::size_t i = 0; i < hidden; ++i) {
                out[d] += static_cast<double>(matrix[(first_row + d) * hidden + i]) * in[i];
            }
            out[d] = static_cast<double>(static_cast<float>(out[d]));
        }
        return rms(out, weight, static_cast<double>(head.rms_eps));
    };
    std::vector<double> mass(tokens, 0.0);
    for (const std::uint32_t row : rows) {
        for (std::size_t h = 0; h < heads; ++h) {
            const std::vector<double> q = apply(head.wq, h * head.q_rows_per_head, normed(row), head.q_norm);
            std::vector<double> weight(tokens, 0.0);
            double total = 0.0;
            for (std::size_t j = 0; j < tokens; ++j) {
                if (allowed[j] == 0) {
                    continue;
                }
                const std::vector<double> k = apply(head.wk, (h / (heads / kv)) * dim, normed(j), head.k_norm);
                double logit = 0.0;
                for (std::size_t d = 0; d < dim; ++d) {
                    logit += q[d] * k[d];
                }
                weight[j] = std::exp(logit / std::sqrt(static_cast<double>(dim)));
                total += weight[j];
            }
            for (std::size_t j = 0; j < tokens; ++j) {
                mass[j] += weight[j] / total / static_cast<double>(heads * rows.size());
            }
        }
    }
    return mass;
}

void test_qk_random_reference() {
    Lcg rng{20261112u};
    pf::QkHead head;
    head.hidden = 12;
    head.heads = 4;
    head.kv_heads = 2;
    head.head_dim = 6;
    head.q_rows_per_head = 12;
    head.rms_eps = 1e-6f;
    head.attn_norm = rng.fill(12);
    head.wq = rng.fill(4 * 12 * 12);
    head.wk = rng.fill(2 * 6 * 12);
    head.q_norm = rng.fill(6);
    head.k_norm = rng.fill(6);
    const std::uint32_t tokens = 37;
    const std::vector<float> residual = rng.fill(tokens * 12);
    const std::vector<std::uint32_t> rows{3, 4, 20, 21, 22, 36};
    const std::vector<std::uint8_t> allowed = pf::request_key_mask(tokens, rows);
    const std::vector<float> keys = pf::qk_keys(head, residual);
    const std::vector<float> mass = pf::qk_mass(head, residual, keys, rows, allowed);
    const std::vector<double> want = naive_mass(head, residual, rows, allowed);
    double total = 0.0;
    for (std::uint32_t j = 0; j < tokens; ++j) {
        near(mass[j], want[j], 2e-6, "random QK mass, token " + std::to_string(j));
        total += static_cast<double>(mass[j]);
    }
    near(total, 1.0, 1e-5, "random QK mass sums to one");
    for (const std::uint32_t row : rows) {
        expect(mass[row] == 0.0f, "request tokens are masked as keys");
    }
    expect(pf::qk_mass(head, residual, pf::qk_keys(head, residual), rows, allowed) == mass, "QK mass is repeatable");

    throws<std::invalid_argument>([&] { (void)pf::qk_mass(head, residual, keys, std::vector<std::uint32_t>{tokens}, allowed); },
                                  "query row out of range");
    pf::QkHead broken = head;
    broken.kv_heads = 3;
    throws<std::invalid_argument>([&] { pf::validate(broken); }, "kv_heads not dividing heads");
    broken = head;
    broken.wq.pop_back();
    throws<std::invalid_argument>([&] { pf::validate(broken); }, "short wq");
}

void test_masks_mix_pool() {
    using Mask = std::vector<std::uint8_t>;
    expect(pf::window_key_mask(6, 2, 4, false) == Mask({1, 1, 0, 0, 0, 0}), "window mask without suffix");
    expect(pf::window_key_mask(6, 2, 4, true) == Mask({1, 1, 0, 0, 1, 1}), "window mask with suffix");
    throws<std::invalid_argument>([] { (void)pf::window_key_mask(3, 2, 4, false); }, "window past the end");
    const std::vector<std::uint32_t> rows{1, 4};
    expect(pf::request_key_mask(5, rows) == Mask({1, 0, 1, 1, 0}), "request mask");

    expect(pf::history_weight(0) == 0.5f && pf::history_weight(1) == 0.25f && pf::history_weight(2) == 0.125f,
           "history weights halve");
    const std::vector<float> now{0.2f, 0.8f}, before{0.6f, 0.4f};
    const std::vector<pf::WeightedMass> parts{{now, 1.0f}, {before, pf::history_weight(0)}};
    const std::vector<float> mixed = pf::mix_masses(parts);
    near(mixed[0], (0.2 + 0.5 * 0.6) / 1.5, 1e-7, "mixed mass 0");
    near(mixed[1], (0.8 + 0.5 * 0.4) / 1.5, 1e-7, "mixed mass 1");
    expect(pf::mix_masses({}).empty(), "empty mix");

    const std::vector<float> mass{0.1f, 0.2f, 0.3f, 0.4f};
    near(pf::pool_span(mass, 1, 4, false), 0.9, 1e-7, "span sum");
    near(pf::pool_span(mass, 1, 4, true), 0.3, 1e-7, "span density");
    expect(pf::pool_span(mass, 2, 2, true) == 0.0f, "empty span");
    throws<std::invalid_argument>([&] { (void)pf::pool_span(mass, 3, 5, false); }, "span past the end");

    expect(pf::box_smooth(mass, 1) == mass, "box kernel 1 is the identity");
    const std::vector<float> smooth = pf::box_smooth(mass, 3);
    near(smooth[0], 0.15, 1e-7, "box left edge");
    near(smooth[1], 0.2, 1e-7, "box interior");
    near(smooth[3], 0.35, 1e-7, "box right edge");
    throws<std::invalid_argument>([&] { (void)pf::box_smooth(mass, 4); }, "even box kernel");

    const std::vector<float> even(8, 0.25f);
    near(pf::score_spread(even, 2), 4.0, 1e-5, "spread of flat scores");
    const std::vector<float> peak{1.0f, 0.0f, 0.0f, -2.0f};
    near(pf::score_spread(peak, 0), 1.0, 1e-6, "spread of one peak");
    expect(pf::score_spread(std::vector<float>(4, 0.0f), 3) == 0.0f, "spread without positive scores");
}

pf::ProbeMlp tiny_mlp() {
    pf::ProbeMlp mlp;
    mlp.hidden = 2;
    mlp.width = 1;
    mlp.norm_eps = 1e-12f;
    mlp.norm_weight = {1.0f, 1.0f};
    mlp.norm_bias = {0.0f, 0.0f};
    mlp.fc1 = {1.0f, 0.0f};
    mlp.fc1_bias = {0.0f};
    mlp.fc2 = {2.0f};
    mlp.fc2_bias = 0.5f;
    return mlp;
}

void test_probes_closed_form() {
    // x = (1,-1): layer norm leaves it unchanged, fc1 picks 1, so raw = 2 * gelu(1) + 0.5;
    // x = (-1,1) gives raw = 2 * gelu(-1) + 0.5.
    const pf::ProbeMlp mlp = tiny_mlp();
    const std::vector<float> residual{1.0f, -1.0f, -1.0f, 1.0f, 5.0f, -5.0f};
    const double erf_up = 0.5 * (1.0 + std::erf(1.0 / std::sqrt(2.0)));
    const double tanh_up = 0.5 * (1.0 + std::tanh(std::sqrt(2.0 / 3.14159265358979323846) * 1.044715));
    const std::vector<float> exact = pf::probe_logits(mlp, residual, pf::Gelu::erf);
    near(exact[0], 2.0 * erf_up + 0.5, 1e-6, "erf GELU logit");
    near(exact[1], -2.0 * (1.0 - erf_up) + 0.5, 1e-6, "erf GELU logit, negative side");
    near(exact[2], exact[0], 1e-6, "layer norm removes the scale");
    const std::vector<float> approx = pf::probe_logits(mlp, residual, pf::Gelu::tanh);
    near(approx[0], 2.0 * tanh_up + 0.5, 1e-6, "tanh GELU logit");
    expect(approx[0] != exact[0], "the two GELU forms differ");

    pf::ProbeMlp biased = mlp;
    biased.norm_weight = {2.0f, 1.0f};
    biased.norm_bias = {0.25f, 0.0f};
    biased.fc1_bias = {-0.25f};
    near(pf::probe_logits(biased, residual, pf::Gelu::erf)[0], 2.0 * 2.0 * (0.5 * (1.0 + std::erf(std::sqrt(2.0)))) + 0.5,
         1e-6, "norm affine and fc1 bias");

    pf::RequestProbe request;
    request.mlp = mlp;
    expect(pf::request_logits(request, residual) == exact, "request probe uses the erf GELU");

    // Smoothing: zero padding at the edges, and the raw logit is added on top of the taps.
    pf::SegmentProbe segment;
    segment.mlp = mlp;
    segment.conv_bias = -0.5f;
    const double r0 = static_cast<double>(approx[0]), r1 = static_cast<double>(approx[1]), r2 = static_cast<double>(approx[2]);
    const auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    segment.conv = {1.0f, 0.0f, 0.0f};  // adds the previous token
    std::vector<float> cut = pf::segment_cut_prob(segment, residual);
    near(cut[0], sigmoid(r0 - 0.5), 1e-6, "previous-token tap, left edge");
    near(cut[1], sigmoid(r1 - 0.5 + r0), 1e-6, "previous-token tap, interior");
    segment.conv = {0.0f, 0.0f, 0.25f};  // adds a quarter of the next token
    cut = pf::segment_cut_prob(segment, residual);
    near(cut[1], sigmoid(r1 - 0.5 + 0.25 * r2), 1e-6, "next-token tap, interior");
    near(cut[2], sigmoid(r2 - 0.5), 1e-6, "next-token tap, right edge");
    segment.conv = {0.0f, 1.0f, 0.0f};  // centre tap: the raw logit counts twice
    near(pf::segment_cut_prob(segment, residual)[0], sigmoid(2.0 * r0 - 0.5), 1e-6, "centre tap");
    segment.conv = {1.0f, 1.0f};
    throws<std::invalid_argument>([&] { (void)pf::segment_cut_prob(segment, residual); }, "even conv");
    throws<std::invalid_argument>([&] { (void)pf::probe_logits(mlp, std::vector<float>(3, 0.0f), pf::Gelu::erf); },
                                  "ragged residual");
}

void test_request_runs() {
    using Mask = std::vector<std::uint8_t>;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    //                             0     1     2     3     4     5     6     7     8     9
    const std::vector<float> raw{1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, nan, 0.5f, 0.5f, 0.5f};
    const std::vector<std::uint32_t> message{0, 0, 0, 0, 1, 2, 2, 2, 2, 2};
    // 0: single token (too short). 2-3: run. 4: cut off by the message change. 5: cut off by the NaN. 7-9: run.
    expect(pf::request_runs(raw, 0.5f, message) == Mask({0, 0, 1, 1, 0, 0, 0, 1, 1, 1}), "request runs");
    expect(pf::request_runs(raw, 0.5f, message, 1) == Mask({1, 0, 1, 1, 1, 1, 0, 1, 1, 1}), "request runs, min run 1");
    expect(pf::request_runs(raw, 0.75f, message) == Mask({0, 0, 1, 1, 0, 0, 0, 0, 0, 0}), "request runs, higher tau");
    throws<std::invalid_argument>([&] { (void)pf::request_runs(raw, 0.5f, std::vector<std::uint32_t>(3, 0)); },
                                  "short message index");
    const Mask mask{0, 0, 1, 1, 0, 0, 0, 1, 1, 1};
    expect(pf::request_rows(mask, 16) == std::vector<std::uint32_t>({2, 3, 7, 8, 9}), "request rows");
    expect(pf::request_rows(mask, 2) == std::vector<std::uint32_t>({8, 9}), "request rows keep the tail under a cap");
}

// ---- synthetic GGUF files --------------------------------------------------------------------------
struct GgufTensor {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::vector<float> values;
    std::uint32_t type = 0;  // F32
};

class GgufWriter {
public:
    void text(const std::string& key, const std::string& value) {
        begin(key, 8);
        str(meta_, value);
    }
    void f32(const std::string& key, float value) {
        begin(key, 6);
        raw(meta_, &value, sizeof value);
    }
    void u32(const std::string& key, std::uint32_t value) {
        begin(key, 4);
        raw(meta_, &value, sizeof value);
    }
    void tensor(GgufTensor t) { tensors_.push_back(std::move(t)); }

    void write(const std::filesystem::path& path, std::size_t truncate = 0) const {
        std::string out = "GGUF";
        const std::uint32_t version = 3;
        const std::uint64_t tensor_count = tensors_.size(), kv_count = kv_count_;
        raw(out, &version, sizeof version);
        raw(out, &tensor_count, sizeof tensor_count);
        raw(out, &kv_count, sizeof kv_count);
        out += meta_;
        std::uint64_t offset = 0;
        for (const GgufTensor& t : tensors_) {
            str(out, t.name);
            const std::uint32_t rank = static_cast<std::uint32_t>(t.dims.size());
            raw(out, &rank, sizeof rank);
            for (const std::uint64_t dim : t.dims) {
                raw(out, &dim, sizeof dim);
            }
            raw(out, &t.type, sizeof t.type);
            raw(out, &offset, sizeof offset);
            offset += (t.values.size() * sizeof(float) + 31) / 32 * 32;
        }
        out.resize((out.size() + 31) / 32 * 32, '\0');
        for (const GgufTensor& t : tensors_) {
            raw(out, t.values.data(), t.values.size() * sizeof(float));
            out.resize((out.size() + 31) / 32 * 32, '\0');
        }
        if (truncate != 0) {
            out.resize(out.size() - truncate);
        }
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file.write(out.data(), static_cast<std::streamsize>(out.size()));
    }

private:
    static void raw(std::string& out, const void* data, std::size_t size) {
        out.append(static_cast<const char*>(data), size);
    }
    static void str(std::string& out, const std::string& value) {
        const std::uint64_t size = value.size();
        raw(out, &size, sizeof size);
        out += value;
    }
    void begin(const std::string& key, std::uint32_t type) {
        ++kv_count_;
        str(meta_, key);
        raw(meta_, &type, sizeof type);
    }

    std::string meta_;
    std::uint64_t kv_count_ = 0;
    std::vector<GgufTensor> tensors_;
};

const std::string kSha = "5d41402abc4b2a76b9719d911017c5925d41402abc4b2a76b9719d911017c592";

// One knob per failure case; the default builds a valid file.
struct Fault {
    std::string skip_key;     // metadata key to leave out
    std::string skip_tensor;  // tensor to leave out
    std::string f16_tensor;   // tensor to mark as F16
    std::string schema;       // replacement schema
    std::string sha = kSha;
    std::string tap = std::string(pf::kFeatureTap);
    std::string base = "Qwen/Qwen3.5-0.8B";
    std::string architecture;
    std::size_t conv_taps = 5;
    float threshold = 0.4f;
    std::uint32_t min_segment = 8;
    std::size_t truncate = 0;
    bool u32_as_float = false;
};

void identity_keys(GgufWriter& w, const std::string& prefix, const std::string& architecture,
                   const std::string& schema, const Fault& fault) {
    const auto text = [&](const std::string& key, const std::string& value) {
        if (key != fault.skip_key) {
            w.text(key, value);
        }
    };
    text("general.architecture", fault.architecture.empty() ? architecture : fault.architecture);
    text(prefix + ".schema", fault.schema.empty() ? schema : fault.schema);
    text(prefix + ".base_model", fault.base);
    text(prefix + ".runtime_gguf_sha256", fault.sha);
    text(prefix + ".feature_tap", fault.tap);
}

void add_tensor(GgufWriter& w, const Fault& fault, const std::string& name, std::vector<std::uint64_t> dims,
                std::vector<float> values) {
    if (name == fault.skip_tensor) {
        return;
    }
    w.tensor({name, std::move(dims), std::move(values), name == fault.f16_tensor ? 1u : 0u});
}

constexpr std::uint32_t kHidden = 6, kWidth = 4;

pf::ProbeMlp synthetic_mlp(Lcg& rng) {
    pf::ProbeMlp mlp;
    mlp.hidden = kHidden;
    mlp.width = kWidth;
    mlp.norm_eps = 1e-5f;
    mlp.norm_weight = rng.fill(kHidden);
    mlp.norm_bias = rng.fill(kHidden);
    mlp.fc1 = rng.fill(kWidth * kHidden);
    mlp.fc1_bias = rng.fill(kWidth);
    mlp.fc2 = rng.fill(kWidth);
    mlp.fc2_bias = rng.next();
    return mlp;
}

void add_mlp(GgufWriter& w, const Fault& fault, const std::string& prefix, const pf::ProbeMlp& mlp) {
    add_tensor(w, fault, prefix + ".norm.weight", {kHidden}, mlp.norm_weight);
    add_tensor(w, fault, prefix + ".norm.bias", {kHidden}, mlp.norm_bias);
    add_tensor(w, fault, prefix + ".fc1.weight", {kHidden, kWidth}, mlp.fc1);
    add_tensor(w, fault, prefix + ".fc1.bias", {kWidth}, mlp.fc1_bias);
    add_tensor(w, fault, prefix + ".fc2.weight", {kWidth}, mlp.fc2);
    add_tensor(w, fault, prefix + ".fc2.bias", {1}, {mlp.fc2_bias});
    if (fault.skip_key != prefix + ".norm_eps") {
        w.f32(prefix + ".norm_eps", mlp.norm_eps);
    }
}

pf::SegmentProbe write_segment(const std::filesystem::path& path, const Fault& fault = {}) {
    Lcg rng{7u};
    pf::SegmentProbe probe;
    probe.mlp = synthetic_mlp(rng);
    probe.conv = rng.fill(fault.conv_taps);
    probe.conv_bias = rng.next();
    probe.threshold = fault.threshold;
    probe.min_segment = fault.min_segment;
    probe.max_segment = 64;
    GgufWriter w;
    identity_keys(w, "segmentprobe", "segmentprobe", std::string(pf::kSegmentProbeSchemaV2), fault);
    add_mlp(w, fault, "segmentprobe", probe.mlp);
    add_tensor(w, fault, "segmentprobe.conv.weight", {fault.conv_taps}, probe.conv);
    add_tensor(w, fault, "segmentprobe.conv.bias", {1}, {probe.conv_bias});
    w.f32("segmentprobe.threshold", probe.threshold);
    if (fault.u32_as_float) {
        w.f32("segmentprobe.min_segment", 8.0f);
    } else {
        w.u32("segmentprobe.min_segment", probe.min_segment);
    }
    w.u32("segmentprobe.max_segment", probe.max_segment);
    if (fault.skip_key != "segmentprobe.oversize_split") {
        w.text("segmentprobe.oversize_split", "line_aligned");
    }
    w.write(path, fault.truncate);
    return probe;
}

pf::RequestProbe write_request(const std::filesystem::path& path, const Fault& fault = {}) {
    Lcg rng{11u};
    pf::RequestProbe probe;
    probe.mlp = synthetic_mlp(rng);
    probe.tau = -0.125f;
    GgufWriter w;
    identity_keys(w, "requestprobe", "requestprobe", std::string(pf::kRequestProbeSchema), fault);
    if (fault.skip_key != "requestprobe.extraction_version") {
        w.text("requestprobe.extraction_version", std::string(pf::kRequestExtraction));
    }
    add_mlp(w, fault, "requestprobe", probe.mlp);
    if (fault.skip_key != "requestprobe.tau") {
        w.f32("requestprobe.tau", probe.tau);
    }
    w.text("requestprobe.candidate_rule", "cand_v5");
    w.write(path, fault.truncate);
    return probe;
}

struct HeadWeights {
    std::vector<float> wq, wk;
};

HeadWeights write_head(const std::filesystem::path& path, const Fault& fault = {}) {
    Lcg rng{13u};
    HeadWeights weights{rng.fill(8 * kHidden), rng.fill(4 * kHidden)};
    GgufWriter w;
    identity_keys(w, "scoringhead", "pflash_scoring_head", std::string(pf::kScoringHeadSchema), fault);
    add_tensor(w, fault, "scoringhead.attn_q.weight", {kHidden, 8}, weights.wq);
    add_tensor(w, fault, "scoringhead.attn_k.weight", {kHidden, fault.conv_taps == 5 ? 4u : 3u},
               fault.conv_taps == 5 ? weights.wk : std::vector<float>(3 * kHidden, 0.5f));
    w.write(path, fault.truncate);
    return weights;
}

void test_weight_files() {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("strix-prompt-scorer-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    const std::filesystem::path path = dir / "weights.gguf";
    Lcg rng{17u};
    const std::vector<float> residual = rng.fill(9 * kHidden);

    {
        const pf::SegmentProbe want = write_segment(path);
        const pf::SegmentProbe got = pf::load_segment_probe(path, kSha);
        expect(got.schema == pf::kSegmentProbeSchemaV2 && got.oversize_split == "line_aligned", "segment probe strings");
        expect(got.mlp.hidden == kHidden && got.mlp.width == kWidth, "segment probe shape");
        expect(got.mlp.fc1 == want.mlp.fc1 && got.mlp.fc2 == want.mlp.fc2 && got.mlp.norm_weight == want.mlp.norm_weight &&
                   got.mlp.norm_bias == want.mlp.norm_bias && got.mlp.fc1_bias == want.mlp.fc1_bias &&
                   got.mlp.fc2_bias == want.mlp.fc2_bias && got.conv == want.conv && got.conv_bias == want.conv_bias,
               "segment probe tensors");
        expect(got.threshold == 0.4f && got.min_segment == 8 && got.max_segment == 64 && got.mlp.norm_eps == 1e-5f,
               "segment probe scalars");
        expect(pf::segment_cut_prob(got, residual) == pf::segment_cut_prob(want, residual),
               "loaded segment probe scores like the source weights");
        Fault optional;
        optional.skip_key = "segmentprobe.oversize_split";
        write_segment(path, optional);
        expect(pf::load_segment_probe(path, kSha).oversize_split.empty(), "oversize_split is optional");
        Fault v1;
        v1.schema = std::string(pf::kSegmentProbeSchemaV1);
        write_segment(path, v1);
        expect(pf::load_segment_probe(path, kSha).schema == pf::kSegmentProbeSchemaV1, "segment probe v1 schema");
    }
    {
        const pf::RequestProbe want = write_request(path);
        const pf::RequestProbe got = pf::load_request_probe(path, kSha);
        expect(got.tau == -0.125f && got.candidate_rule == "cand_v5" && got.schema == pf::kRequestProbeSchema,
               "request probe scalars");
        expect(pf::request_logits(got, residual) == pf::request_logits(want, residual),
               "loaded request probe scores like the source weights");
    }
    {
        const HeadWeights want = write_head(path);
        const pf::ScoringHeadFile got = pf::load_scoring_head(path, kSha);
        expect(got.hidden == kHidden && got.q_rows == 8 && got.k_rows == 4, "scoring head shape");
        expect(got.wq == want.wq && got.wk == want.wk && got.schema == pf::kScoringHeadSchema, "scoring head tensors");
    }

    // Fail closed.
    const auto segment_fails = [&](const Fault& fault, const std::string& what, const std::string& sha = kSha) {
        write_segment(path, fault);
        throws<std::runtime_error>([&] { (void)pf::load_segment_probe(path, sha); }, "segment probe: " + what);
    };
    const auto with = [](const std::function<void(Fault&)>& edit) {
        Fault fault;
        edit(fault);
        return fault;
    };
    segment_fails({}, "different drafter hash", std::string(64, '0'));
    segment_fails({}, "empty drafter hash", "");
    segment_fails(with([](Fault& f) { f.schema = "qwen3_5_0_8b_segment_probe_v9"; }), "unknown schema");
    segment_fails(with([](Fault& f) { f.architecture = "requestprobe"; }), "wrong architecture");
    segment_fails(with([](Fault& f) { f.base = "other/model"; }), "wrong base model");
    segment_fails(with([](Fault& f) { f.tap = "post_block13"; }), "wrong feature tap");
    segment_fails(with([](Fault& f) { f.skip_key = "segmentprobe.runtime_gguf_sha256"; }), "missing hash key");
    segment_fails(with([](Fault& f) { f.skip_key = "segmentprobe.norm_eps"; }), "missing norm_eps");
    segment_fails(with([](Fault& f) { f.skip_tensor = "segmentprobe.fc2.bias"; }), "missing tensor");
    segment_fails(with([](Fault& f) { f.f16_tensor = "segmentprobe.fc1.weight"; }), "non-F32 tensor");
    segment_fails(with([](Fault& f) { f.conv_taps = 4; }), "even conv length");
    segment_fails(with([](Fault& f) { f.threshold = 1.0f; }), "threshold outside (0, 1)");
    segment_fails(with([](Fault& f) { f.min_segment = 0; }), "zero min_segment");
    segment_fails(with([](Fault& f) { f.min_segment = 65; }), "min_segment above max_segment");
    segment_fails(with([](Fault& f) { f.u32_as_float = true; }), "mistyped min_segment");
    segment_fails(with([](Fault& f) { f.truncate = 40; }), "truncated file");

    write_request(path, with([](Fault& f) { f.skip_key = "requestprobe.extraction_version"; }));
    throws<std::runtime_error>([&] { (void)pf::load_request_probe(path, kSha); }, "request probe: missing extraction");
    write_request(path, with([](Fault& f) { f.skip_key = "requestprobe.tau"; }));
    throws<std::runtime_error>([&] { (void)pf::load_request_probe(path, kSha); }, "request probe: missing tau");
    write_request(path);
    throws<std::runtime_error>([&] { (void)pf::load_segment_probe(path, kSha); }, "request file as segment probe");

    write_head(path, with([](Fault& f) { f.skip_tensor = "scoringhead.attn_k.weight"; }));
    throws<std::runtime_error>([&] { (void)pf::load_scoring_head(path, kSha); }, "scoring head: missing attn_k");
    write_head(path, with([](Fault& f) { f.conv_taps = 3; }));
    throws<std::runtime_error>([&] { (void)pf::load_scoring_head(path, kSha); }, "scoring head: q rows not a multiple");
    write_head(path);
    throws<std::runtime_error>([&] { (void)pf::load_scoring_head(path, "abc"); }, "scoring head: different hash");
    throws<std::exception>([&] { (void)pf::load_scoring_head(dir / "absent.gguf", kSha); }, "absent file");

    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
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
        test_qk_closed_form();
        test_qk_head_pairing_and_gate_half();
        test_qk_random_reference();
        test_masks_mix_pool();
        test_probes_closed_form();
        test_request_runs();
        test_weight_files();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return EXIT_FAILURE;
    }
    std::cout << "prompt scorer tests passed\n";
    return EXIT_SUCCESS;
}
