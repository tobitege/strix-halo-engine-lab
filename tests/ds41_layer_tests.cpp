// Host tests of the deepseek41 layer sites: declared FP32 arithmetic against double-precision
// oracles written from the model definition, structural properties (Sinkhorn, RoPE inverse, split
// argmax), and with STRIX_DEEPSEEK41_GGUF the real tensors, the Engram hash against an independent
// golden and the row reader against plain reads of the file.

#include "strix/deepseek41_model.hpp"
#include "strix/ds41_engram.hpp"
#include "strix/ds41_layer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#if defined(_MSC_VER) && !defined(__clang__)
#include <immintrin.h>  // _mm_cvtps_ph (F16C)
#endif

#include "ds41_engram_golden.inc"

namespace {

using namespace strix::ds41;
int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

#if defined(_MSC_VER) && !defined(__clang__)
// no _Float16 in MSVC: the F16C conversion, round to nearest even like the cast
std::uint16_t to_f16(float x) { return static_cast<std::uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), _MM_FROUND_TO_NEAREST_INT), 0)); }
#else
std::uint16_t to_f16(float x) { return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(x)); }
#endif

std::vector<float> random_vector(std::mt19937& rng, std::size_t n, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

double rel_err(double got, double want) { return std::fabs(got - want) / std::max(1e-30, std::fabs(want)); }

// ---- double oracles (from the model definition, not from the declared FP32 code) --------------------
double sigmoid_d(double x) { return 1.0 / (1.0 + std::exp(-x)); }

void oracle_mix(const std::vector<float>& stream, const std::vector<std::uint16_t>& fn, const std::vector<float>& base,
                const std::vector<float>& scale, std::uint32_t iters, double hc_eps, double norm_eps,
                std::array<double, 4>& pre, std::array<double, 4>& post, std::array<double, 16>& comb) {
    const std::size_t n = stream.size();
    double ss = 0.0;
    for (const float x : stream) ss += static_cast<double>(x) * x;
    const double r = 1.0 / std::sqrt(ss / static_cast<double>(n) + norm_eps);
    std::array<double, 24> m{};
    for (std::size_t j = 0; j < 24; ++j) {
        double d = 0.0;
        for (std::size_t i = 0; i < n; ++i) d += static_cast<double>(f16_to_float(fn[j * n + i])) * stream[i];
        m[j] = d * r;
    }
    for (int j = 0; j < 4; ++j) pre[j] = sigmoid_d(m[j] * scale[0] + base[j]) + hc_eps;
    for (int j = 0; j < 4; ++j) post[j] = 2.0 * sigmoid_d(m[j + 4] * scale[1] + base[j + 4]);
    for (int j = 0; j < 4; ++j) {
        double mx = -1e300, s = 0.0;
        for (int k = 0; k < 4; ++k) { comb[j * 4 + k] = m[8 + j * 4 + k] * scale[2] + base[8 + j * 4 + k]; mx = std::max(mx, comb[j * 4 + k]); }
        for (int k = 0; k < 4; ++k) { comb[j * 4 + k] = std::exp(comb[j * 4 + k] - mx); s += comb[j * 4 + k]; }
        for (int k = 0; k < 4; ++k) comb[j * 4 + k] = comb[j * 4 + k] / s + hc_eps;
    }
    auto cols = [&] {
        for (int k = 0; k < 4; ++k) {
            double s = 0.0;
            for (int j = 0; j < 4; ++j) s += comb[j * 4 + k];
            for (int j = 0; j < 4; ++j) comb[j * 4 + k] /= s + hc_eps;
        }
    };
    auto rows = [&] {
        for (int j = 0; j < 4; ++j) {
            double s = 0.0;
            for (int k = 0; k < 4; ++k) s += comb[j * 4 + k];
            for (int k = 0; k < 4; ++k) comb[j * 4 + k] /= s + hc_eps;
        }
    };
    cols();
    for (std::uint32_t it = 1; it < iters; ++it) { rows(); cols(); }
}

void test_rms_norm(std::mt19937& rng) {
    for (const std::size_t n : {512U, 1280U, 5120U}) {
        const auto x = random_vector(rng, n, 3.0F), w = random_vector(rng, n, 1.0F);
        std::vector<float> y(n);
        rms_norm(x, w, 1e-20F, y);
        double ss = 0.0;
        for (const float v : x) ss += static_cast<double>(v) * v;
        const double r = 1.0 / std::sqrt(ss / static_cast<double>(n) + 1e-20);
        double worst = 0.0;
        for (std::size_t i = 0; i < n; ++i) worst = std::max(worst, rel_err(y[i], static_cast<double>(w[i]) * x[i] * r));
        expect(worst < 4e-6, "RMSNorm within 4e-6 of the double oracle (n=" + std::to_string(n) + "): " + std::to_string(worst));
    }
}

void test_hc(std::mt19937& rng) {
    const std::size_t hidden = 5120, n = hidden * hc_copies;
    const auto stream = random_vector(rng, n, 1.0F);
    const auto fn_f = random_vector(rng, n * hc_mix_count, 0.02F);
    std::vector<std::uint16_t> fn(fn_f.size());
    for (std::size_t i = 0; i < fn.size(); ++i) fn[i] = to_f16(fn_f[i]);
    const auto base = random_vector(rng, hc_mix_count, 0.5F);
    const std::vector<float> scale{0.7F, 1.3F, 2.1F};
    const HcSiteWeights site{fn, base, scale};
    const auto mix = hc_mixes(stream, site, 20, 1e-6F, 1e-20F);
    std::array<double, 4> pre{}, post{};
    std::array<double, 16> comb{};
    oracle_mix(stream, fn, base, scale, 20, 1e-6, 1e-20, pre, post, comb);
    double worst = 0.0;
    for (int j = 0; j < 4; ++j) worst = std::max({worst, std::fabs(mix.pre[j] - pre[j]), std::fabs(mix.post[j] - post[j])});
    for (int j = 0; j < 16; ++j) worst = std::max(worst, std::fabs(mix.comb[j] - comb[j]));
    expect(worst < 2e-6, "hyper-connection mix within 2e-6 of the double oracle: " + std::to_string(worst));
    for (int j = 0; j < 4; ++j) {
        double rs = 0.0, cs = 0.0;
        for (int k = 0; k < 4; ++k) { rs += mix.comb[j * 4 + k]; cs += mix.comb[k * 4 + j]; }
        expect(std::fabs(cs - 1.0) < 1e-5 && std::fabs(rs - 1.0) < 1e-3, "Sinkhorn comb is doubly stochastic");
        expect(mix.pre[j] > 0.0F && mix.pre[j] < 1.0F + 2e-6F && mix.post[j] > 0.0F && mix.post[j] < 2.0F, "pre/post ranges");
    }
    // Collapse / expand against double.
    const auto sub = random_vector(rng, hidden, 1.0F);
    std::vector<float> collapsed(hidden), expanded(n);
    hc_collapse(stream, std::span<const float, hc_copies>(mix.pre.data(), hc_copies), collapsed);
    hc_expand(sub, stream, mix, expanded);
    double wc = 0.0, we = 0.0;
    for (std::size_t i = 0; i < hidden; ++i) {
        double y = 0.0;
        for (int c = 0; c < 4; ++c) y += static_cast<double>(mix.pre[c]) * stream[c * hidden + i];
        wc = std::max(wc, std::fabs(collapsed[i] - y));
        for (int k = 0; k < 4; ++k) {
            double e = static_cast<double>(mix.post[k]) * sub[i];
            for (int j = 0; j < 4; ++j) e += static_cast<double>(mix.comb[j * 4 + k]) * stream[j * hidden + i];
            we = std::max(we, std::fabs(expanded[k * hidden + i] - e));
        }
    }
    expect(wc < 2e-6 && we < 4e-6, "collapse/expand within FP32 rounding of double: " + std::to_string(wc) + " " + std::to_string(we));
    std::vector<float> first(hidden);
    const auto identity = hc_identity_pre();
    hc_collapse(stream, std::span<const float, hc_copies>(identity.data(), hc_copies), first);
    expect(std::equal(first.begin(), first.end(), stream.begin()), "identity pre collapses to copy 0 exactly");
}

void test_rope(std::mt19937& rng) {
    RopeConfig yarn{64, 160000.0, 65536.0, 16.0, 32.0, 1.0};
    const auto f = rope_frequencies(yarn);
    // Reference semantics: dims inside [low, high] blend; below low unchanged; above high divided by factor.
    const double pi = 3.14159265358979323846;
    auto corrected = [&](double rot) { return 64.0 * std::log(65536.0 / (rot * 2 * pi)) / (2 * std::log(160000.0)); };
    const double low = std::max(std::floor(corrected(32.0)), 0.0), high = std::min(std::ceil(corrected(1.0)), 63.0);
    expect(low >= 0 && high > low, "YaRN band");
    const double f0 = 1.0, flast = 1.0 / std::pow(160000.0, 62.0 / 64.0) / 16.0;
    expect(rel_err(f[0], f0) < 1e-15 && rel_err(f[31], flast) < 1e-12, "YaRN ends: unchanged / divided by factor");
    const auto table = rope_table(yarn, 0, 70000);
    auto x = random_vector(rng, 64, 1.0F);
    const auto original = x;
    apply_rope(x, table, 0, false);
    expect(x == original, "position 0 is the identity (cos 1, sin 0)");
    double worst = 0.0;
    for (const std::uint64_t pos : {1ULL, 127ULL, 4096ULL, 65535ULL, 69999ULL}) {
        auto y = original;
        apply_rope(y, table, pos, false);
        apply_rope(y, table, pos, true);
        for (std::size_t i = 0; i < 64; ++i) worst = std::max(worst, static_cast<double>(std::fabs(y[i] - original[i])));
    }
    expect(worst < 1e-6, "inverse RoPE undoes RoPE within FP32 rounding: " + std::to_string(worst));
    RopeConfig plain{64, 10000.0, 0.0, 16.0, 32.0, 1.0};
    const auto fp = rope_frequencies(plain);
    expect(rel_err(fp[5], 1.0 / std::pow(10000.0, 10.0 / 64.0)) < 1e-15, "window-only layers: plain theta");
}

void test_projections(std::mt19937& rng) {
    // Q8_0 sites take A8 activations: each output is product_q8_0 of its row and its group's slice.
    const std::size_t width = 1280, rows = 64, groups = 2;
    std::vector<std::uint8_t> w(rows * width / 32 * 34);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.001F, 0.05F);
    for (std::size_t b = 0; b < rows * width / 32; ++b) {
        const auto s16 = to_f16(sc(rng));
        w[b * 34] = static_cast<std::uint8_t>(s16 & 0xffU);
        w[b * 34 + 1] = static_cast<std::uint8_t>(s16 >> 8U);
        for (std::size_t k = 0; k < 32; ++k) w[b * 34 + 2 + k] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code(rng)));
    }
    const auto xf = random_vector(rng, groups * width, 1.0F);
    const auto x = quantize_a8(xf);
    std::vector<float> plain(rows), grouped(rows);
    q8_0_a8_rows(w.data(), width, 0, rows, rows, a8_slice(x, 0, width), plain.data());
    q8_0_a8_rows(w.data(), width, 0, rows, rows / groups, x, grouped.data());
    const std::size_t row_bytes = width / 32 * 34;
    bool same = true;
    for (std::size_t o = 0; o < rows; ++o) {
        const auto slice = a8_slice(x, (o / (rows / groups)) * width, width);
        same = same && grouped[o] == product_q8_0(std::span<const std::uint8_t>(w.data() + o * row_bytes, row_bytes), slice, width);
        same = same && plain[o] == product_q8_0(std::span<const std::uint8_t>(w.data() + o * row_bytes, row_bytes), a8_slice(x, 0, width), width);
    }
    expect(same, "Q8_0 sites (plain and grouped) = product_q8_0 of the group's A8 slice");
    // The A8 product against the double product of the unquantised activations: the activation
    // quantisation error, bounded by half a step per element (informational bound 2e-2 of sum |terms|).
    double worst = 0.0;
    for (std::size_t o = 0; o < rows; ++o) {
        double d = 0.0, mag = 0.0;
        for (std::size_t b = 0; b < width / 32; ++b) {
            const std::uint8_t* blk = w.data() + (o * width / 32 + b) * 34;
            const double s16 = f16_to_float(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8U)));
            for (std::size_t k = 0; k < 32; ++k) {
                const double t = s16 * static_cast<std::int8_t>(blk[2 + k]) * xf[b * 32 + k];
                d += t; mag += std::fabs(t);
            }
        }
        worst = std::max(worst, std::fabs(plain[o] - d) / mag);
    }
    expect(worst < 2e-2, "A8 product within the activation quantisation bound: " + std::to_string(worst));
    // F16 rows (FP32 activations).
    std::vector<std::uint16_t> wf(8 * 6144);
    for (auto& v : wf) v = to_f16(std::normal_distribution<float>(0.0F, 0.02F)(rng));
    const auto xff = random_vector(rng, 6144, 1.0F);
    std::vector<float> yf(8);
    f16_rows(wf.data(), 6144, 0, 8, xff.data(), yf.data());
    double wfe = 0.0;
    for (int o = 0; o < 8; ++o) {
        double d = 0.0, mag = 0.0;
        for (int i = 0; i < 6144; ++i) { const double t = static_cast<double>(f16_to_float(wf[o * 6144 + i])) * xff[i]; d += t; mag += std::fabs(t); }
        wfe = std::max(wfe, std::fabs(yf[o] - d) / mag);
    }
    expect(wfe < 1e-6, "F16 projection within 1e-6 of double: " + std::to_string(wfe));
    bool f16_same = true;
    for (std::uint32_t h = 0; h < 65536U; ++h) {
        const auto a = f16_to_float(static_cast<std::uint16_t>(h)), b = f16_to_float_select(static_cast<std::uint16_t>(h));
        f16_same = f16_same && __builtin_bit_cast(std::uint32_t, a) == __builtin_bit_cast(std::uint32_t, b);
    }
    expect(f16_same, "f16_to_float_select = f16_to_float bitwise for every half");
    // Fused pieces are their compositions.
    const std::size_t hidden = 5120;
    const auto stream = random_vector(rng, hidden * hc_copies, 1.0F), weight = random_vector(rng, hidden, 1.0F);
    const std::array<float, hc_copies> pre{0.3F, 0.2F, 0.4F, 0.1F};
    std::vector<float> normed(hidden), x1(hidden), x2(hidden);
    const auto fused = collapse_norm_a8(stream, std::span<const float, hc_copies>(pre.data(), hc_copies), weight, 1e-20F, normed);
    hc_collapse(stream, std::span<const float, hc_copies>(pre.data(), hc_copies), x1);
    rms_norm(x1, weight, 1e-20F, x2);
    const auto direct = quantize_a8(x2);
    expect(x2 == normed && fused.codes == direct.codes && fused.deltas == direct.deltas, "collapse + norm + A8 = composition");
    // K-split site: one part is the plain site; two parts are h0 + h1 of the halves' own products.
    {
        std::vector<float> one(rows), two(rows);
        const auto x1280 = a8_slice(x, 0, width);
        q8_0_a8_split_rows(w.data(), width, rows, 1, x1280, one.data());
        q8_0_a8_split_rows(w.data(), width, rows, 2, x1280, two.data());
        bool split_ok = one == plain;
        for (std::size_t o = 0; o < rows; ++o) {
            const std::uint8_t* r0 = w.data() + o * row_bytes;
            const float h0 = product_q8_0(std::span<const std::uint8_t>(r0, row_bytes / 2), a8_slice(x1280, 0, width / 2), width / 2);
            const float h1 = product_q8_0(std::span<const std::uint8_t>(r0 + row_bytes / 2, row_bytes / 2), a8_slice(x1280, width / 2, width / 2),
                                          width / 2);
            split_ok = split_ok && two[o] == h0 + h1;
        }
        expect(split_ok, "K-split Q8_0 site = plain (1 part) and h0 + h1 (2 parts)");
    }
}

void test_argmax(std::mt19937& rng) {
    const std::uint32_t vocab = 129280;
    auto logits = random_vector(rng, vocab, 3.0F);
    const auto a = vocab_slice(vocab, 2, 0), b = vocab_slice(vocab, 2, 1);
    expect(a.first == 0 && a.count == 64640 && b.first == 64640 && b.count == 64640, "two-rank vocabulary split");
    auto split = [&] {
        const auto ca = argmax_slice(std::span<const float>(logits.data() + a.first, a.count), a.first);
        const auto cb = argmax_slice(std::span<const float>(logits.data() + b.first, b.count), b.first);
        return std::pair{combine_argmax(ca, cb), combine_argmax(cb, ca)};
    };
    const auto single = argmax_slice(logits, 0);
    auto [ab, ba] = split();
    expect(ab.index == single.index && ba.index == single.index && ab.finite, "split argmax = single argmax");
    // A tie across the boundary goes to the lower global index in either combine order.
    logits[70000] = logits[5] = 100.0F;
    std::tie(ab, ba) = split();
    expect(ab.index == 5 && ba.index == 5 && argmax_slice(logits, 0).index == 5, "cross-rank tie -> lower index");
    logits[3] = 100.0F;
    expect(argmax_slice(logits, 0).index == 3, "in-slice tie -> lower index");
    logits[100000] = std::nanf("");
    std::tie(ab, ba) = split();
    expect(!ab.finite && !ba.finite, "a NaN in either slice makes the result unusable");
    expect(vocab_slice(10, 3, 0).count == 4 && vocab_slice(10, 3, 2).first == 7, "uneven split covers every index");
}

void test_engram_gate(std::mt19937& rng) {
    const std::size_t hidden = 5120, n = hidden * hc_copies;
    auto stream = random_vector(rng, n, 1.0F);
    const auto kv = random_vector(rng, hidden * (hc_copies + 1), 0.7F);
    const auto qn = random_vector(rng, n, 1.0F), kn = random_vector(rng, n, 1.0F);
    const auto gates = engram_gates(stream, kv, qn, kn, 1e-20F);
    double worst = 0.0;
    for (std::uint32_t c = 0; c < hc_copies; ++c) {
        double hh = 0, kk = 0, dk = 0;
        for (std::size_t i = 0; i < hidden; ++i) {
            const double h = stream[c * hidden + i], k = kv[c * hidden + i];
            hh += h * h; kk += k * k; dk += h * (static_cast<double>(qn[c * hidden + i]) * kn[c * hidden + i]) * k;
        }
        const double dot = dk / std::sqrt(hh / hidden + 1e-20) / std::sqrt(kk / hidden + 1e-20) / std::sqrt(static_cast<double>(hidden));
        const double g = sigmoid_d(std::copysign(std::sqrt(std::max(std::fabs(dot), 1e-6)), dot));
        worst = std::max(worst, std::fabs(gates[c] - g));
    }
    expect(worst < 1e-5, "Engram gate within 1e-5 of the double oracle: " + std::to_string(worst));
    expect(engram_gate_of(-0.0F) < 0.5F && engram_gate_of(0.0F) > 0.5F, "signed zero keeps its side (copysign)");
    expect(std::isnan(engram_gate_of(std::nanf(""))), "NaN dot stays NaN");
    auto before = stream;
    engram_inject(stream, kv, qn, kn, 1e-20F);
    for (std::uint32_t c = 0; c < hc_copies; ++c)
        expect(stream[c * hidden + 7] == before[c * hidden + 7] + gates[c] * kv[hc_copies * hidden + 7], "injection adds gate * value");
    // bf16 rounding of decoded rows: ties to even.
    expect(round_to_bf16(__builtin_bit_cast(float, 0x3f808000U)) == __builtin_bit_cast(float, 0x3f800000U) &&
               round_to_bf16(__builtin_bit_cast(float, 0x3f818000U)) == __builtin_bit_cast(float, 0x3f820000U),
           "bfloat16 rounding ties to even");
}

// ---- real file ---------------------------------------------------------------------------------
std::vector<std::uint8_t> read_bytes(const char* path, std::uint64_t offset, std::uint64_t bytes) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::uint8_t> out(bytes);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("short read from the model file");
    return out;
}

// Dead n-gram positions (rows whose embedding was provided, e.g. image rows): a synthetic contract.
void test_engram_dead() {
    strix::deepseek41::EngramContract c;
    c.layer_ids = {2, 5};
    c.max_ngram_size = 3;
    c.n_heads = 2;
    c.primes = {101, 103, 107, 109, 113, 127, 131, 137};  // [layer][(n - 1) * heads]
    c.rows = {101 + 103 + 107 + 109, 113 + 127 + 131 + 137};
    c.multipliers = {3, 5, 7, 11, 13, 17};                // [layer][n]
    c.token_map.resize(32);
    for (std::uint32_t t = 0; t < 32; ++t) c.token_map[t] = (t * 5U + 1U) % 29U;
    c.compressed_vocab_size = 29;
    c.pad_id = 31;
    const std::size_t per = 2U * c.rows_per_token();
    const auto row = [&](const std::vector<std::uint64_t>& all, std::size_t t) {
        return std::vector<std::uint64_t>(all.begin() + static_cast<std::ptrdiff_t>(t * per),
                                          all.begin() + static_cast<std::ptrdiff_t>((t + 1) * per));
    };
    const std::uint32_t dead = strix::ds41::engram_dead_token(9);
    EngramHasher h(c);
    const std::vector<std::uint32_t> text{4, 9, 9, 7, 8, 3}, mixed{4, dead, dead, 7, 8, 3};
    const auto plain = h.hash_sequence(text);
    expect(h.hash_sequence(text) == plain, "Engram: a text sequence hashes as before");
    const auto with = h.hash_sequence(mixed);
    expect(h.positions() == 6, "Engram: dead positions count as positions");
    expect(row(with, 0) == row(plain, 0), "Engram: rows before a dead position are unchanged");
    expect(row(with, 5) == row(plain, 5), "Engram: a row whose n-grams end after the dead span is unchanged");
    expect(row(with, 3) != row(plain, 3) && row(with, 4) != row(plain, 4), "Engram: n-grams do not read through a dead position");
    // a dead position hashes padding only (whatever its token), and the first row after it sees padding behind it:
    // the same rows as the start of a sequence
    const auto start = h.hash_sequence(std::vector<std::uint32_t>{7, 8});
    expect(row(with, 3) == row(start, 0), "Engram: the row after a dead span = a sequence start");
    expect(row(with, 4) == row(start, 1), "Engram: the rows after a dead span continue as after a sequence start");
    const auto other = h.hash_sequence(std::vector<std::uint32_t>{4, strix::ds41::engram_dead_token(1), dead, 7, 8, 3});
    expect(other == with, "Engram: a dead row does not depend on its token");
    expect(row(with, 1) == row(with, 2), "Engram: dead rows are all the padding row");
    // truncate + push replays, the bit stays in the history
    (void)h.hash_sequence(mixed);
    h.truncate(3);
    std::vector<std::uint64_t> again(per);
    h.push(7, again);
    expect(again == row(with, 3), "Engram: truncate keeps dead positions");
    bool threw = false;
    try {
        h.push(strix::ds41::engram_dead_token(32), again);
    } catch (const std::exception&) {
        threw = true;
    }
    expect(threw, "Engram: a dead token is still checked against the vocabulary");
}

void test_real(const char* path) {
    const auto model = strix::deepseek41::validate_target(strix::deepseek41::read_file(path));
    EngramHasher hasher(model.engram);
    const std::vector<std::uint32_t> tokens(std::begin(engram_golden_tokens), std::end(engram_golden_tokens));
    const auto rows = hasher.hash_sequence(tokens);
    expect(rows.size() == std::size(engram_golden_rows), "Engram golden geometry");
    expect(std::equal(rows.begin(), rows.end(), std::begin(engram_golden_rows)), "Engram hash rows = independent golden (reference algorithm)");
    // push after truncate replays the same rows.
    hasher.truncate(5);
    std::vector<std::uint64_t> again(hasher.layers() * hasher.columns());
    hasher.push(tokens[5], again);
    expect(std::equal(again.begin(), again.end(), rows.begin() + 5 * again.size()), "truncate + push = same rows");

    const auto tables = engram_tables(model);
    expect(tables.size() == 2, "two Engram tables");
    std::vector<EngramRowRequest> requests;
    for (std::size_t i = 0; i < rows.size(); ++i) requests.push_back({static_cast<std::uint32_t>((i / 24) % 2), rows[i]});
    std::mt19937_64 rng(7);
    for (int i = 0; i < 4000; ++i) {
        const std::uint32_t t = static_cast<std::uint32_t>(rng() % 2);
        requests.push_back({t, rng() % tables[t].rows});
    }
    requests.push_back({0, 0});
    requests.push_back({1, tables[1].rows - 1});
    for (const bool direct : {true, false}) {
        EngramRowReader reader(path, tables, 16, direct);
        std::vector<std::uint8_t> out(requests.size() * engram_row_bytes);
        const auto start = std::chrono::steady_clock::now();
        reader.read(requests, out);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        bool same = true;
        for (std::size_t i = 0; i < requests.size(); i += 97) {
            const auto ref = read_bytes(path, tables[requests[i].table].file_offset + requests[i].row * engram_row_bytes, engram_row_bytes);
            same = same && std::equal(ref.begin(), ref.end(), out.begin() + static_cast<std::ptrdiff_t>(i * engram_row_bytes));
        }
        const auto& last = requests.back();
        const auto ref = read_bytes(path, tables[last.table].file_offset + last.row * engram_row_bytes, engram_row_bytes);
        same = same && std::equal(ref.begin(), ref.end(), out.end() - static_cast<std::ptrdiff_t>(engram_row_bytes));
        expect(same, std::string("Engram reader rows = plain reads (") + (direct ? "direct" : "buffered") + ")");
        std::cout << "Engram reader (" << (reader.direct_io() ? "O_DIRECT" : "buffered") << ", 16 threads): " << requests.size()
                  << " rows in " << s * 1e3 << " ms (" << static_cast<double>(requests.size()) / s / 1e3 << " K rows/s)\n";
    }
    std::cout << "real-file Engram checks done\n";
}

} // namespace

int main() {
    std::mt19937 rng(20261019);
    test_rms_norm(rng);
    test_hc(rng);
    test_rope(rng);
    test_projections(rng);
    test_argmax(rng);
    test_engram_gate(rng);
    test_engram_dead();
    if (const char* path = std::getenv("STRIX_DEEPSEEK41_GGUF")) test_real(path);
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "ds41 layer tests passed\n";
    return 0;
}
