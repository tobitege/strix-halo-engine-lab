// Host tests of the shared vision encoder: FP32 reference tower on synthetic towers (hand-computed rows and graph
// invariants), default positions, the device plan, the embed cache and the row splice. No GPU, no checkpoint.
#include "strix/vision/embed_cache.hpp"

#include "../src/attn_core/two_term_model.hpp"
#include "../src/vision/encoder_layout.hpp"
#include "vision_encoder_toy.hpp"

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _MSC_VER
#include <crtdbg.h>
#endif

using namespace strix::vision;
using vision_toy::make_grid;
using vision_toy::make_tower;
using vision_toy::Tower;
using vision_toy::ToyAdapter;

namespace {

int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (false)

template <class F> bool throws(F&& f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return m;
}

float* mut(const Tensor& t) { return const_cast<float*>(static_cast<const float*>(t.data)); }

EncoderConfig base_config() {
    EncoderConfig c;
    c.layers = 2;
    c.width = 16;
    c.heads = 2;
    c.mlp_width = 24;
    c.patch = 2;
    c.channels = 3;
    c.norm = EncoderConfig::Norm::layernorm;
    c.mlp = EncoderConfig::Mlp::gelu;
    c.qkv_bias = c.out_bias = c.mlp_bias = true;
    c.pos = EncoderConfig::Pos::learned_interp;
    c.pos_grid = 4;
    return c;
}
EncoderConfig rms_config() {
    EncoderConfig c = base_config();
    c.norm = EncoderConfig::Norm::rmsnorm;
    c.mlp = EncoderConfig::Mlp::swiglu;
    c.swiglu_limit = 0.75F;
    c.mlp_bias = false;
    c.pos = EncoderConfig::Pos::rope2d;
    c.pos_grid = 0;
    return c;
}

// ---- independent single-row forward: with one row the attention output is the value projection itself ----------

std::vector<double> matvec(const Tensor& w, const Tensor& b, const std::vector<double>& x, std::size_t row0, std::size_t rows) {
    std::vector<double> y(rows);
    for (std::size_t o = 0; o < rows; ++o) {
        double acc = b ? static_cast<double>(b.at(row0 + o)) : 0.0;
        for (std::size_t k = 0; k < w.cols; ++k) acc += static_cast<double>(w.at((row0 + o) * w.cols + k)) * x[k];
        y[o] = acc;
    }
    return y;
}
std::vector<double> norm1(const EncoderConfig& c, const Tensor& w, const Tensor& b, const std::vector<double>& x) {
    const double n = static_cast<double>(x.size());
    double mean = 0.0, sq = 0.0;
    if (c.norm == EncoderConfig::Norm::layernorm) mean = std::accumulate(x.begin(), x.end(), 0.0) / n;
    for (const double v : x) sq += (v - mean) * (v - mean);
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        y[i] = (x[i] - mean) / std::sqrt(sq / n + static_cast<double>(c.eps)) * static_cast<double>(w.at(i)) + (b ? static_cast<double>(b.at(i)) : 0.0);
    return y;
}
std::vector<float> hand_single_row(const Tower& t, const PatchGrid& g, const PositionInputs& p) {
    const auto& c = t.cfg;
    const std::size_t W = c.width, M = c.mlp_width;
    std::vector<double> patch(g.data.begin(), g.data.end());
    std::vector<double> x = matvec(t.w.patch_w, t.w.patch_b, patch, 0, W);
    for (std::size_t i = 0; i < p.addend.size(); ++i) x[i] += static_cast<double>(p.addend[i]);
    for (const auto& b : t.w.blocks) {
        const auto n = norm1(c, b.norm1_w, b.norm1_b, x);
        const auto v = matvec(b.qkv_w, b.qkv_b, n, 2U * W, W);
        const auto o = matvec(b.out_w, b.out_b, v, 0, W);
        for (std::size_t i = 0; i < W; ++i) x[i] += o[i];
        const auto n2 = norm1(c, b.norm2_w, b.norm2_b, x);
        const auto m = matvec(b.mlp1_w, b.mlp1_b, n2, 0, c.mlp_out());
        std::vector<double> a(M);
        for (std::size_t j = 0; j < M; ++j) {
            if (c.mlp == EncoderConfig::Mlp::swiglu) {
                double gate = m[c.swiglu_gate_first ? j : M + j], up = m[c.swiglu_gate_first ? M + j : j];
                if (c.swiglu_limit > 0.0F) {
                    gate = std::min(gate, static_cast<double>(c.swiglu_limit));
                    up = std::clamp(up, -static_cast<double>(c.swiglu_limit), static_cast<double>(c.swiglu_limit));
                }
                a[j] = gate / (1.0 + std::exp(-gate)) * up;
            } else if (c.mlp == EncoderConfig::Mlp::gelu) {
                a[j] = 0.5 * m[j] * (1.0 + std::erf(m[j] / std::sqrt(2.0)));
            } else {
                a[j] = 0.5 * m[j] * (1.0 + std::tanh(std::sqrt(2.0 / 3.14159265358979323846) * (m[j] + 0.044715 * m[j] * m[j] * m[j])));
            }
        }
        const auto o2 = matvec(b.mlp2_w, b.mlp2_b, a, 0, W);
        for (std::size_t i = 0; i < W; ++i) x[i] += o2[i];
    }
    if (c.final_norm) x = norm1(c, t.w.final_norm_w, t.w.final_norm_b, x);
    std::vector<float> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = static_cast<float>(x[i]);
    return out;
}

void test_bf16() {
    CHECK(from_bf16(to_bf16(1.0F)) == 1.0F);
    CHECK(from_bf16(to_bf16(-0.375F)) == -0.375F);
    CHECK(to_bf16(1.0F) == 0x3F80U);
    // ties to even: 1 + 2^-8 is halfway between two BF16 values (mantissa lsb 2^-7)
    CHECK(to_bf16(1.0F + 1.0F / 256.0F) == 0x3F80U);
    CHECK(to_bf16(1.0F + 3.0F / 256.0F) == 0x3F82U);
    CHECK(std::isinf(from_bf16(to_bf16(INFINITY))));
    CHECK(std::isnan(from_bf16(to_bf16(NAN))));
    const float v[3] = {0.5F, -2.0F, 3.25F};
    const auto rows = to_bf16_rows(v, 3);
    CHECK(rows.size() == 3 && from_bf16(rows[2]) == 3.25F);
    const std::uint16_t half[4] = {0x3C00U, 0xC000U, 0x0001U, 0x7C00U};  // 1, -2, smallest subnormal, inf
    const Tensor t{half, Tensor::Type::f16, 4, 1};
    CHECK(t.at(0) == 1.0F && t.at(1) == -2.0F && t.at(2) == std::ldexp(1.0F, -24) && std::isinf(t.at(3)));
}

void test_validate() {
    validate(base_config());
    validate(rms_config());
    auto c = base_config();
    c.heads = 3;
    CHECK(throws([&] { validate(c); }));
    c = rms_config();
    c.width = 12;  // head dim 6: not a multiple of 4
    CHECK(throws([&] { validate(c); }));
    c = base_config();
    c.pos_grid = 0;
    CHECK(throws([&] { validate(c); }));
    c = base_config();
    c.deepstack_taps = {1, 1};
    CHECK(throws([&] { validate(c); }));
    c.deepstack_taps = {2};
    CHECK(throws([&] { validate(c); }));
    Tower t = make_tower(base_config(), 1);
    validate(t.cfg, t.w);
    t.w.blocks.pop_back();
    CHECK(throws([&] { validate(t.cfg, t.w); }));
    Tower u = make_tower(base_config(), 1);
    u.w.blocks[0].qkv_b = Tensor{};
    CHECK(throws([&] { validate(u.cfg, u.w); }));
    Tower v = make_tower(base_config(), 1);
    PatchGrid g = make_grid(v.cfg, 2, 3, 5);
    PositionInputs p = default_positions(v.cfg, v.w, g);
    validate(v.cfg, g, p);
    g.tile_end = {4};
    CHECK(throws([&] { validate(v.cfg, g, p); }));
    g.tile_end = {2, 6};
    validate(v.cfg, g, p);
    p.addend.clear();
    CHECK(throws([&] { validate(v.cfg, g, p); }));
}

void test_positions() {
    Tower t = make_tower(base_config(), 3);
    const unsigned G = t.cfg.pos_grid, W = t.cfg.width;
    // The grid of the table itself reproduces the table.
    PatchGrid g = make_grid(t.cfg, G, G, 4);
    PositionInputs p = default_positions(t.cfg, t.w, g);
    std::vector<float> table(t.w.pos_embed.size());
    for (std::size_t i = 0; i < table.size(); ++i) table[i] = t.w.pos_embed.at(i);
    CHECK(max_diff(p.addend, table) < 1e-6);
    CHECK(p.angles.empty());
    // A 2 x 2 grid samples the four corners.
    g = make_grid(t.cfg, 2, 2, 4);
    p = default_positions(t.cfg, t.w, g);
    for (unsigned k = 0; k < W; ++k) {
        CHECK(p.addend[k] == table[k]);
        CHECK(p.addend[W + k] == table[std::size_t{G - 1U} * W + k]);
        CHECK(p.addend[2U * W + k] == table[std::size_t{G} * (G - 1U) * W + k]);
        CHECK(p.addend[3U * W + k] == table[std::size_t{G * G - 1U} * W + k]);
    }
    // Rotary: head dim 8 -> 2 frequencies per axis, [row | column].
    Tower r = make_tower(rms_config(), 3);
    g = make_grid(r.cfg, 3, 5, 4);
    p = default_positions(r.cfg, r.w, g);
    CHECK(p.addend.empty() && p.angles.size() == std::size_t{15} * 4U);
    const std::size_t row = 2U * 5U + 3U;  // y = 2, x = 3
    CHECK(std::fabs(p.angles[row * 4U] - 2.0F) < 1e-6F);
    CHECK(std::fabs(p.angles[row * 4U + 1U] - 2.0F / 100.0F) < 1e-6F);
    CHECK(std::fabs(p.angles[row * 4U + 2U] - 3.0F) < 1e-6F);
    CHECK(std::fabs(p.angles[row * 4U + 3U] - 3.0F / 100.0F) < 1e-6F);
}

void test_single_row() {
    std::vector<EncoderConfig> configs{base_config(), rms_config()};
    EncoderConfig c = base_config();
    c.mlp = EncoderConfig::Mlp::gelu_tanh;
    c.pos = EncoderConfig::Pos::rope2d_plus_learned;
    c.final_norm = true;
    c.qkv_bias = false;
    configs.push_back(c);
    c = rms_config();
    c.swiglu_gate_first = false;
    c.swiglu_limit = 0.0F;
    c.final_norm = true;
    c.rope_pairing = EncoderConfig::RopePairing::adjacent;
    configs.push_back(c);
    std::uint64_t seed = 11;
    for (const auto& cfg : configs) {
        Tower t = make_tower(cfg, seed++);
        const PatchGrid g = make_grid(t.cfg, 1, 1, seed++);
        const PositionInputs p = default_positions(t.cfg, t.w, g);
        const EncoderOutput out = reference_encode(t.cfg, t.w, g, p);
        CHECK(out.rows == 1 && out.width == t.cfg.width);
        CHECK(max_diff(out.hidden, hand_single_row(t, g, p)) < 2e-5);
    }
}

// q and k weights zero: every score is 0, the attention output is the mean of the values over the tile.
void test_mean_attention() {
    EncoderConfig c = base_config();
    c.layers = 1;
    c.qkv_bias = false;
    Tower t = make_tower(c, 21);
    const std::size_t W = c.width;
    auto& b = t.w.blocks[0];
    std::fill(mut(b.qkv_w), mut(b.qkv_w) + 2U * W * W, 0.0F);
    std::fill(mut(b.mlp2_w), mut(b.mlp2_w) + W * c.mlp_width, 0.0F);
    std::fill(mut(b.mlp2_b), mut(b.mlp2_b) + W, 0.0F);
    const PatchGrid g = make_grid(c, 2, 3, 22, 2);  // two tiles of three rows
    const PositionInputs p = default_positions(c, t.w, g);
    const EncoderOutput out = reference_encode(c, t.w, g, p);
    std::vector<std::vector<double>> x0(g.rows), v(g.rows);
    for (unsigned r = 0; r < g.rows; ++r) {
        std::vector<double> patch(g.data.begin() + r * g.patch_dim, g.data.begin() + (r + 1U) * g.patch_dim);
        x0[r] = matvec(t.w.patch_w, t.w.patch_b, patch, 0, W);
        for (std::size_t i = 0; i < W; ++i) x0[r][i] += static_cast<double>(p.addend[r * W + i]);
        v[r] = matvec(b.qkv_w, Tensor{}, norm1(c, b.norm1_w, b.norm1_b, x0[r]), 2U * W, W);
    }
    std::vector<float> want(g.rows * W);
    for (unsigned r = 0; r < g.rows; ++r) {
        const unsigned t0 = r / 3U * 3U;
        std::vector<double> mean(W, 0.0);
        for (unsigned j = t0; j < t0 + 3U; ++j)
            for (std::size_t i = 0; i < W; ++i) mean[i] += v[j][i] / 3.0;
        const auto o = matvec(b.out_w, b.out_b, mean, 0, W);
        for (std::size_t i = 0; i < W; ++i) want[r * W + i] = static_cast<float>(x0[r][i] + o[i]);
    }
    CHECK(max_diff(out.hidden, want) < 2e-5);
}

PatchGrid permuted(const PatchGrid& g, const std::vector<unsigned>& order) {
    PatchGrid o = g;
    for (unsigned r = 0; r < g.rows; ++r) {
        std::copy_n(g.data.begin() + order[r] * g.patch_dim, g.patch_dim, o.data.begin() + r * g.patch_dim);
        o.y[r] = g.y[order[r]];
        o.x[r] = g.x[order[r]];
    }
    return o;
}

// Bidirectional attention: permuting the rows permutes the output (a causal tower would fail this).
void test_permutation() {
    for (const auto& cfg : {base_config(), rms_config()}) {
        Tower t = make_tower(cfg, 31);
        const PatchGrid g = make_grid(cfg, 3, 4, 32);
        std::vector<unsigned> order(g.rows);
        std::iota(order.begin(), order.end(), 0U);
        std::reverse(order.begin(), order.end());
        std::swap(order[1], order[7]);
        const PatchGrid h = permuted(g, order);
        const EncoderOutput a = reference_encode(cfg, t.w, g, default_positions(cfg, t.w, g));
        const EncoderOutput b = reference_encode(cfg, t.w, h, default_positions(cfg, t.w, h));
        std::vector<float> want(a.hidden.size());
        for (unsigned r = 0; r < g.rows; ++r) std::copy_n(a.hidden.begin() + order[r] * cfg.width, cfg.width, want.begin() + r * cfg.width);
        CHECK(max_diff(b.hidden, want) < 2e-5);
        // and the rows really interact: another row's patch changes row 0
        PatchGrid k = g;
        k.data[std::size_t{5} * g.patch_dim] += 1.0F;
        const EncoderOutput d = reference_encode(cfg, t.w, k, default_positions(cfg, t.w, k));
        double moved = 0.0;
        for (unsigned i = 0; i < cfg.width; ++i) moved = std::max(moved, std::fabs(static_cast<double>(d.hidden[i]) - static_cast<double>(a.hidden[i])));
        CHECK(moved > 1e-5);
    }
}

// Rotary positions are relative: a common shift of every coordinate changes nothing; a real move does.
void test_rope_translation() {
    for (const auto pairing : {EncoderConfig::RopePairing::half, EncoderConfig::RopePairing::adjacent}) {
        EncoderConfig c = rms_config();
        c.rope_pairing = pairing;
        Tower t = make_tower(c, 41);
        const PatchGrid g = make_grid(c, 3, 3, 42);
        PatchGrid s = g;
        for (unsigned r = 0; r < g.rows; ++r) {
            s.y[r] = static_cast<std::uint16_t>(g.y[r] + 3U);
            s.x[r] = static_cast<std::uint16_t>(g.x[r] + 5U);
        }
        const EncoderOutput a = reference_encode(c, t.w, g, default_positions(c, t.w, g));
        const EncoderOutput b = reference_encode(c, t.w, s, default_positions(c, t.w, s));
        CHECK(max_diff(a.hidden, b.hidden) < 5e-5);
        PatchGrid m = g;
        m.x[4] = 7;
        const EncoderOutput d = reference_encode(c, t.w, m, default_positions(c, t.w, m));
        CHECK(max_diff(a.hidden, d.hidden) > 1e-5);
    }
}

// Tiles do not see each other: two tiles equal two separate encodes.
void test_tiles() {
    const EncoderConfig c = rms_config();
    Tower t = make_tower(c, 51);
    const PatchGrid g = make_grid(c, 2, 4, 52, 2);
    const EncoderOutput both = reference_encode(c, t.w, g, default_positions(c, t.w, g));
    std::vector<float> want;
    for (unsigned tile = 0; tile < 2U; ++tile) {
        PatchGrid h;
        h.rows = 4;
        h.patch_dim = g.patch_dim;
        h.grid_h = g.grid_h;
        h.grid_w = g.grid_w;
        h.data.assign(g.data.begin() + tile * 4U * g.patch_dim, g.data.begin() + (tile + 1U) * 4U * g.patch_dim);
        h.y.assign(g.y.begin() + tile * 4U, g.y.begin() + (tile + 1U) * 4U);
        h.x.assign(g.x.begin() + tile * 4U, g.x.begin() + (tile + 1U) * 4U);
        const EncoderOutput o = reference_encode(c, t.w, h, default_positions(c, t.w, h));
        want.insert(want.end(), o.hidden.begin(), o.hidden.end());
    }
    CHECK(max_diff(both.hidden, want) < 1e-6);
    PatchGrid one = g;
    one.tile_end.clear();
    const EncoderOutput joined = reference_encode(c, t.w, one, default_positions(c, t.w, one));
    CHECK(max_diff(joined.hidden, want) > 1e-5);
}

void test_taps_and_types() {
    EncoderConfig c = base_config();
    c.layers = 3;
    c.deepstack_taps = {0, 2};
    c.final_norm = true;
    Tower t = make_tower(c, 61);
    const PatchGrid g = make_grid(c, 2, 2, 62);
    const PositionInputs p = default_positions(c, t.w, g);
    const EncoderOutput out = reference_encode(c, t.w, g, p);
    CHECK(out.taps.size() == 2);
    // tap 0 = a one-block tower without the final norm
    EncoderConfig c1 = c;
    c1.layers = 1;
    c1.deepstack_taps.clear();
    c1.final_norm = false;
    EncoderWeights w1 = t.w;
    w1.blocks.resize(1);
    const EncoderOutput first = reference_encode(c1, w1, g, p);
    CHECK(max_diff(out.taps[0], first.hidden) == 0.0);
    // tap 2 = the last hidden state before the final norm
    EncoderConfig c3 = c;
    c3.deepstack_taps.clear();
    c3.final_norm = false;
    const EncoderOutput last = reference_encode(c3, t.w, g, p);
    CHECK(max_diff(out.taps[1], last.hidden) == 0.0);
    CHECK(max_diff(out.hidden, last.hidden) > 1e-4);

    // BF16 weights: the same tower with every tensor rounded to BF16, stored as BF16 or as FP32, gives the same bits.
    Tower f = make_tower(c, 63), h = make_tower(c, 63);
    std::deque<std::vector<std::uint16_t>> store;
    const auto round = [&](Tensor& a, Tensor& b) {
        if (!a) return;
        auto& bits = store.emplace_back(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            bits[i] = to_bf16(a.at(i));
            mut(a)[i] = from_bf16(bits[i]);
        }
        b = Tensor{bits.data(), Tensor::Type::bf16, a.rows, a.cols};
    };
    round(f.w.patch_w, h.w.patch_w);
    round(f.w.patch_b, h.w.patch_b);
    round(f.w.pos_embed, h.w.pos_embed);
    round(f.w.final_norm_w, h.w.final_norm_w);
    round(f.w.final_norm_b, h.w.final_norm_b);
    for (std::size_t l = 0; l < f.w.blocks.size(); ++l) {
        auto& a = f.w.blocks[l];
        auto& b = h.w.blocks[l];
        round(a.norm1_w, b.norm1_w), round(a.norm1_b, b.norm1_b), round(a.qkv_w, b.qkv_w), round(a.qkv_b, b.qkv_b);
        round(a.out_w, b.out_w), round(a.out_b, b.out_b), round(a.norm2_w, b.norm2_w), round(a.norm2_b, b.norm2_b);
        round(a.mlp1_w, b.mlp1_w), round(a.mlp1_b, b.mlp1_b), round(a.mlp2_w, b.mlp2_w), round(a.mlp2_b, b.mlp2_b);
    }
    ReferenceEncoder ef(c, f.w), eh(c, h.w);
    const EncoderOutput of = ef.encode(g, default_positions(c, f.w, g)), oh = eh.encode(g, default_positions(c, h.w, g));
    CHECK(max_diff(of.hidden, oh.hidden) == 0.0);
    CHECK(std::string(ef.arithmetic()) == "vision-ref-f32");
}

void test_plan() {
    EncoderConfig c;  // the 32 x 1024 SwiGLU tower shape
    c.layers = 32;
    c.width = 1024;
    c.heads = 16;
    c.mlp_width = 2816;
    c.norm = EncoderConfig::Norm::rmsnorm;
    c.mlp = EncoderConfig::Mlp::swiglu;
    c.qkv_bias = c.out_bias = true;
    c.patch = 14;
    CHECK(hip_supported(c));
    const EncoderPlan p = plan(c, 9216, 1);
    CHECK(p.head_pad == 64);
    CHECK(p.mlp_chunk_rows % 128U == 0U && p.mlp_chunk_rows >= 128U && p.mlp_chunk_rows < 9216U);
    CHECK(p.mlp_bytes <= EncoderOptions{}.mlp_workspace_bytes + 3U * strix::vision::layout::kAlign);
    // Balanced chunks: 9216 rows under the default cap run as equal chunks (no small tail chunk).
    CHECK(9216U % p.mlp_chunk_rows == 0U);
    CHECK(plan(c, 1550, 1).mlp_chunk_rows == 1550U);  // under the cap: one chunk (was 1280 + a 270-row tail)
    {
        EncoderOptions small;
        small.mlp_workspace_bytes = 96U << 20U;  // 37,888 B per single-operand row: cap 2560 rows
        CHECK(plan(c, 3888, 1, small).mlp_chunk_rows == 2048U);  // 2048 + 1840, not 2560 + 1328
    }
    // FP16 panels: 32 x (3.15 M + 1.05 M + 5.77 M + 2.88 M) + patch, two bytes each = 785 MiB
    CHECK(p.weight_bytes > std::size_t{780} << 20U && p.weight_bytes < std::size_t{800} << 20U);
    const EncoderPlan small = plan(c, 1521, 1);
    CHECK(small.fixed_bytes < p.fixed_bytes && small.workspace_bytes() < p.workspace_bytes());
    EncoderOptions split;
    split.split_activations = true;
    CHECK(plan(c, 1521, 1, split).workspace_bytes() > small.workspace_bytes());
    CHECK(plan(c, 1521, 4).fixed_bytes >= small.fixed_bytes);
    {  // two-term attention (opt-in) holds the lo terms of K and V^T: 2 x slots x heads x dp x 2 B more
        EncoderOptions one, two;
        one.attention_terms = 1U;
        two.attention_terms = 2U;
        const std::size_t kv = strix::vision::layout::aligned(std::size_t{3072} * 16U * 64U * 2U);  // plan's slot bound: (2 x 1521 + 15 + 15) & ~15
        CHECK(plan(c, 1521, 1, two).fixed_bytes == plan(c, 1521, 1, one).fixed_bytes + 2U * kv);
    }
    CHECK(throws([&] { (void)plan(c, 0, 1); }));
    c.width = 1152;  // head dim 72 runs padded to 80
    c.mlp_width = 4304;
    c.mlp = EncoderConfig::Mlp::gelu_tanh;
    c.norm = EncoderConfig::Norm::layernorm;
    c.pos = EncoderConfig::Pos::rope2d_plus_learned;
    c.pos_grid = 48;
    CHECK(plan(c, 1024, 1).head_pad == 80);
    c.heads = 24;  // head dim 48
    std::string why;
    CHECK(!hip_supported(c, &why) && !why.empty());
    CHECK(throws([&] { (void)plan(c, 16, 1); }));
    using namespace strix::vision::layout;
    CHECK(key_slots(1) == 16 && key_slots(8) == 16 && key_slots(9) == 32 && key_slots(1521) == 3056);
    CHECK(head_pad(64) == 64 && head_pad(72) == 80 && head_pad(80) == 80 && head_pad(48) == 0);
    EncoderConfig odd = base_config();
    odd.width = 576;  // 8 x 72: heads * 80 == round_up(576, 128), still compacted
    odd.heads = 8;
    CHECK(geometry(odd, 16, 32, {}).compact);
}

// The shared two-term model (src/attn_core/two_term_model.hpp): its RNE agrees with the reference's own FP16
// rounding; split errors and product errors stay inside the format- and range-conditioned bounds (f16 and bf16,
// normal remainders, subnormal remainders, subnormal values, near overflow); below 2^-14 two terms gain nothing.
void test_two_term_large_head() {
    namespace tm = strix::attn_core::model;
    constexpr unsigned dp = 1026U;
    const std::vector<tm::Terms> q(dp, {0.0F, 0.0F});
    for (const auto f : {tm::Format::f16, tm::Format::bf16}) {
        for (unsigned terms : {1U, 2U}) {
            std::vector<tm::Terms> v(dp);
            std::vector<float> want(dp);
            for (unsigned d = 0; d < dp; ++d) {
                v[d] = tm::split(static_cast<float>(d % 8U + 1U) + 0x1p-12F, f, terms);
                want[d] = v[d].hi + v[d].lo;
            }
            for (unsigned nk : {1U, 17U}) {
                std::vector<float> out(dp + 2U, -123.0F);
                tm::dense_row(q.data(), nk, [&](unsigned) { return q.data(); },
                              [&](unsigned) { return v.data(); }, dp, 1.0F, terms, f, out.data() + 1U);
                CHECK(std::equal(want.begin(), want.end(), out.begin() + 1));
                CHECK(out.front() == -123.0F && out.back() == -123.0F);
            }
        }
    }
}

void test_f16_rounding_modes() {
    namespace tm = strix::attn_core::model;
    struct RestoreRounding {
        int mode = std::fegetround();
        ~RestoreRounding() { (void)std::fesetround(mode); }
    } restore;
    CHECK(restore.mode != -1);
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::bit_cast<float>(std::uint32_t{0x7FC12345U});
    const struct { float input, expected; } cases[] = {
        {0.0F, 0.0F}, {0x1p-149F, 0.0F}, {0x1p-25F, 0.0F},
        {0x1.8p-24F, 0x1p-23F}, {0x1p-24F, 0x1p-24F},
        {0x1.ffcp-15F, 0x1p-14F}, {0x1p-14F, 0x1p-14F},
        {0x1.002p0F, 1.0F}, {0x1.003p0F, 0x1.004p0F},
        {0x1.006p0F, 0x1.008p0F}, {65519.0F, 65504.0F},
        {65520.0F, inf}, {inf, inf}, {nan, nan}
    };
    for (int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        CHECK(std::fesetround(mode) == 0);
        for (const auto& c : cases) {
            for (bool negative : {false, true}) {
                const std::uint32_t sign = negative ? 0x80000000U : 0U;
                const float x = std::bit_cast<float>(std::bit_cast<std::uint32_t>(c.input) | sign);
                const auto expected = std::bit_cast<std::uint32_t>(c.expected) | sign;
                CHECK(std::bit_cast<std::uint32_t>(tm::round_operand(x, tm::Format::f16)) == expected);
                CHECK(std::bit_cast<std::uint32_t>(round_to_f16(x)) == expected);
            }
        }
        const float x = 0x1.003p0F;
        const tm::Terms q{0.0F, 0.0F};
        for (unsigned terms : {1U, 2U}) {
            const auto v = tm::split(x, tm::Format::f16, terms);
            CHECK(v.hi == 0x1.004p0F);
            CHECK(v.lo == (terms == 1U ? 0.0F : -0x1p-12F));
            CHECK(std::fabs(double(x) - v.hi - v.lo) <= tm::operand_bound(x, tm::Format::f16, terms));
            float out = -123.0F;
            tm::dense_row(&q, 1U, [&](unsigned) { return &q; }, [&](unsigned) { return &v; },
                          1U, 1.0F, terms, tm::Format::f16, &out);
            CHECK(out == (terms == 1U ? 0x1.004p0F : x));
        }
        CHECK(std::fegetround() == mode);
    }
}

void test_two_term_model() {
    namespace tm = strix::attn_core::model;
    std::uint64_t s = 0x9E3779B97F4A7C15ULL;
    const auto next = [&s] {
        s ^= s << 13U;
        s ^= s >> 7U;
        s ^= s << 17U;
        return s;
    };
    std::vector<float> xs = {0.0F, 1.0F, -1.0F, 0.1F, 1e-3F, 6.1e-5F, 0x1p-14F, 0x1.8p-15F, 3e-8F, 65504.0F, 65519.0F, 0.33333334F};
    for (int i = 0; i < 4000; ++i) {  // log-uniform magnitudes 2^-30 .. 2^15, both signs
        const double e = static_cast<double>(next() % 45000U) / 1000.0 - 30.0;
        const double m = 1.0 + static_cast<double>(next() % 1000000U) / 1e6;
        xs.push_back(static_cast<float>((next() & 1U ? -m : m) * std::exp2(e)));
    }
    unsigned gain_lost = 0;
    for (const float x : xs) {
        CHECK(tm::round_operand(x, tm::Format::f16) == round_to_f16(x));
        CHECK(round_f16_terms(x, 2U) == tm::split(x, tm::Format::f16, 2U).hi + tm::split(x, tm::Format::f16, 2U).lo);
        for (const tm::Format f : {tm::Format::f16, tm::Format::bf16}) {
            for (unsigned terms = 1; terms <= 2U; ++terms) {
                const tm::Terms t = tm::split(x, f, terms);
                const double err = std::fabs(static_cast<double>(x) - t.hi - t.lo);
                CHECK(err <= tm::operand_bound(x, f, terms));
                const float y = xs[next() % xs.size()];
                const tm::Terms u = tm::split(y, f, terms);
                if (std::isfinite(t.hi) && std::isfinite(u.hi))
                    CHECK(std::fabs(static_cast<double>(x) * y - tm::device_product(t, u)) <= tm::product_bound(x, y, f, terms));
            }
        }
        if (std::fabs(x) < 0x1p-14F) {  // hi subnormal: lo is 0, two terms are one term
            CHECK(tm::split(x, tm::Format::f16, 2U).lo == 0.0F);
            ++gain_lost;
        }
    }
    CHECK(gain_lost > 100U);
    CHECK(std::isinf(tm::split(65520.0F, tm::Format::f16, 2U).hi) && std::isinf(tm::operand_bound(65520.0, tm::Format::f16, 2U)));
    // The relative two-term figure depends on the format: f16 3 u^2 ~ 2^-20.4, bf16 3 u^2 ~ 2^-14.4 per product.
    CHECK(tm::product_bound(1.0, 1.0, tm::Format::f16, 2U) < 0x1p-20 && tm::product_bound(1.0, 1.0, tm::Format::bf16, 2U) > 0x1p-15);
    CHECK(tm::dot_roundings(64U, 1U) == 3U && tm::dot_roundings(64U, 2U) == 7U && tm::dot_roundings(80U, 2U) == 10U);
}

// The device-rounding models of the attention operands: one FP16 term carries a relative operand error up to 2^-11,
// hi + lo up to max(2^-22 |x|, 2^-25) (two_term_model.hpp); on a tower whose scores are large (ill-conditioned
// softmax), the two-term form tracks FP32 far closer than the one-term form (the device's opt-in vs its faster
// default), both in the operand model (attn_*) and in the model of the core's real products and order (attn_order).
void test_attention_terms() {
    for (const float x : {1.0F, 0.1F, -3.14159F, 1e-3F, 6.1e-5F, 1234.567F, 0.33333334F}) {
        CHECK(round_f16_terms(x, 0U) == x);
        CHECK(round_f16_terms(x, 1U) == round_to_f16(x));
        CHECK(std::fabs(round_to_f16(x) - x) <= std::fabs(x) * 0x1p-11F);
        CHECK(std::fabs(round_f16_terms(x, 2U) - x) <= std::max(std::fabs(x) * 0x1p-22F, 0x1p-25F));
    }
    EncoderConfig c = base_config();
    c.layers = 1;
    Tower t = make_tower(c, 31);
    for (std::size_t i = 0; i < 2U * c.width * c.width; ++i) mut(t.w.blocks[0].qkv_w)[i] *= 6.0F;  // sharp softmax
    const PatchGrid g = make_grid(c, 4, 4, 32, 1);
    const PositionInputs p = default_positions(c, t.w, g);
    const EncoderOutput exact = reference_encode(c, t.w, g, p);
    ReferenceRounding one, two, qk;
    one.attention(1U);
    two.attention(2U);
    qk.attn_q = qk.attn_k = 1U;  // one term on QK only
    const auto rel = [&](const ReferenceRounding& m) {
        const EncoderOutput o = reference_encode(c, t.w, g, p, m);
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < o.hidden.size(); ++i) {
            num += (static_cast<double>(o.hidden[i]) - exact.hidden[i]) * (static_cast<double>(o.hidden[i]) - exact.hidden[i]);
            den += static_cast<double>(exact.hidden[i]) * exact.hidden[i];
        }
        return std::sqrt(num / den);
    };
    ReferenceRounding order1, order2;
    order1.attention_order(1U);
    order2.attention_order(2U);
    const double e1 = rel(one), e2 = rel(two), eqk = rel(qk), o1 = rel(order1), o2 = rel(order2);
    std::printf("attention terms model: relL2 one %.3g two %.3g qk-only %.3g | order model one %.3g two %.3g\n", e1, e2, eqk, o1, o2);
    CHECK(e1 > 0.0 && eqk > 0.0 && e2 < e1 / 64.0);
    CHECK(o1 > e1 / 4.0 && o1 < e1 * 4.0 && o2 > 0.0 && o2 < o1 / 64.0);
    CHECK(ReferenceEncoder(c, t.w, two).arithmetic() == std::string("vision-ref-f32-rounded"));
}

ProjectedRows rows_of(unsigned rows, unsigned width, std::uint16_t fill) {
    ProjectedRows r;
    r.rows = rows;
    r.width = width;
    r.bf16.assign(std::size_t{rows} * width, fill);
    return r;
}

void test_cache() {
    Digest a{}, b{}, c{};
    a[0] = 1;
    b[0] = 2;
    c[31] = 3;
    const std::size_t one = rows_of(4, 64, 0).bytes();
    EmbedCache cache(3U * (one + 64U));
    CHECK(!cache.find("m", a));
    const auto ra = cache.insert("m", a, rows_of(4, 64, 1));
    const auto rb = cache.insert("m", b, rows_of(4, 64, 2));
    CHECK(cache.find("m", a) == ra && cache.find("m", b) == rb);
    CHECK(!cache.find("other", a));  // another adapter identity never shares rows
    const auto dup = cache.insert("m", a, rows_of(4, 64, 9));
    CHECK(dup == ra && dup->bf16[0] == 1);
    const auto rc = cache.insert("m", c, rows_of(4, 64, 3));
    CHECK(cache.stats().entries == 3 && cache.stats().evictions == 0);
    CHECK(cache.find("m", b) == rb);  // order now b, c, a: a is the oldest
    const auto rd = cache.insert("n", a, rows_of(4, 64, 4));
    CHECK(cache.stats().entries == 3 && cache.stats().evictions == 1);
    CHECK(!cache.find("m", a) && cache.find("m", b) && cache.find("m", c) && cache.find("n", a) == rd);
    CHECK(ra->bf16[0] == 1);  // evicted rows stay valid for their holders
    const auto big = cache.insert("m", a, rows_of(64, 64, 5));
    CHECK(big && big->rows == 64 && cache.stats().rejected == 1 && !cache.find("m", a));
    const auto s = cache.stats();
    CHECK(s.bytes <= s.budget && s.hits >= 6 && s.misses >= 4);
    cache.clear();
    CHECK(cache.stats().entries == 0 && cache.stats().bytes == 0);
    EmbedCache off(0);
    CHECK(off.insert("m", a, rows_of(1, 8, 1)) && !off.find("m", a));
    CHECK(EmbedCache::budget_from_env() == EmbedCache::kDefaultBudget || std::getenv("STRIX_VISION_CACHE_MIB"));
}

void test_encode_cached() {
    EncoderConfig cfg = base_config();
    cfg.deepstack_taps = {0};
    Tower t = make_tower(cfg, 71, 12);
    ToyAdapter adapter(t);
    ReferenceEncoder backend(cfg, t.w);
    EmbedCache cache;
    const PatchGrid g = make_grid(cfg, 2, 3, 72);
    Digest d{};
    d[5] = 9;
    const CachedEncode first = encode_cached(adapter, backend, t.w, &cache, d, g);
    CHECK(!first.hit && first.rows && first.rows->rows == 6 && first.rows->width == 12 && first.rows->deep.size() == 1);
    const CachedEncode again = encode_cached(adapter, backend, t.w, &cache, d, g);
    CHECK(again.hit && again.rows == first.rows && adapter.projections == 1);
    Digest e = d;
    e[5] = 10;  // a changed pixel is another digest
    PatchGrid g2 = g;
    g2.data[0] += 0.5F;
    const CachedEncode other = encode_cached(adapter, backend, t.w, &cache, e, g2);
    CHECK(!other.hit && other.rows != first.rows && other.rows->bf16 != first.rows->bf16 && adapter.projections == 2);
    const CachedEncode uncached = encode_cached(adapter, backend, t.w, nullptr, d, g);
    CHECK(!uncached.hit && uncached.rows->bf16 == first.rows->bf16 && adapter.projections == 3);
    // the projector went through DenseOps: row 0 equals the hand product
    const EncoderOutput h = backend.encode(g, default_positions(cfg, t.w, g));
    for (unsigned o = 0; o < 12U; ++o) {
        double acc = 0.0;
        for (unsigned k = 0; k < cfg.width; ++k) acc += static_cast<double>(t.projector[std::size_t{o} * cfg.width + k]) * static_cast<double>(h.hidden[k]);
        CHECK(first.rows->bf16[o] == to_bf16(static_cast<float>(acc)));
    }
    const SpanLayout layout = adapter.layout(g);
    CHECK(layout.rows() == 6 && layout.lm_pos[5][0] == 5 && layout.real_token_id == 7);

    // splice: rows 2 and 0 of the image into rows 1 and 3 of an embedding buffer
    std::vector<float> embed(std::size_t{4} * 16U, -1.0F);
    const std::uint32_t dst[2] = {1, 3}, src[2] = {2, 0};
    splice_rows_host(embed.data(), 16, *first.rows, dst, src, 2);
    for (unsigned k = 0; k < 12U; ++k) {
        CHECK(embed[16U + k] == from_bf16(first.rows->bf16[2U * 12U + k]));
        CHECK(embed[48U + k] == from_bf16(first.rows->bf16[k]));
    }
    CHECK(embed[0] == -1.0F && embed[16U + 12U] == -1.0F && embed[32U] == -1.0F);
    const std::uint32_t bad[1] = {6};
    CHECK(throws([&] { splice_rows_host(embed.data(), 16, *first.rows, dst, bad, 1); }));
}

// The projector as the DeepSeek adapter computed it before the shared op (host unshuffle, linear, FP64 activation,
// linear), kept verbatim as the oracle: DenseOps::merge_project must give the same bits.
std::vector<float> old_projector(const EncoderOutput& h, unsigned gh, unsigned gw, const MergeProjector& m, DenseOps& dense) {
    const unsigned R = m.ratio, W = h.width, mh = (gh + R - 1U) / R, mw = (gw + R - 1U) / R, tokens = mh * mw;
    const std::size_t in = std::size_t{W} * R * R, P = m.w1.rows;
    std::vector<float> merged(std::size_t{tokens} * in);
    for (unsigned my = 0; my < mh; ++my)
        for (unsigned mx = 0; mx < mw; ++mx) {
            float* dst = &merged[(std::size_t{my} * mw + mx) * in];
            for (unsigned dy = 0; dy < R; ++dy)
                for (unsigned dx = 0; dx < R; ++dx) {
                    if (my * R + dy >= gh || mx * R + dx >= gw) continue;
                    const float* src = &h.hidden[(std::size_t{my * R + dy} * gw + mx * R + dx) * W];
                    const unsigned slot = m.row_major ? dy * R + dx : dx * R + dy;
                    for (unsigned c = 0; c < W; ++c) dst[std::size_t{c} * R * R + slot] = src[c];
                }
        }
    std::vector<float> mid(std::size_t{tokens} * P), rows(std::size_t{tokens} * m.w2.rows);
    dense.linear(m.w1, m.b1, merged.data(), tokens, mid.data());
    for (float& v : mid) {
        const double x = static_cast<double>(v);
        switch (m.act) {
            case Activation::gelu: v = static_cast<float>(0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440))); break;
            case Activation::gelu_tanh:
                v = static_cast<float>(0.5 * x * (1.0 + std::tanh(0.79788456080286535588 * (x + 0.044715 * x * x * x))));
                break;
            case Activation::silu: v = static_cast<float>(x / (1.0 + std::exp(-x))); break;
        }
    }
    dense.linear(m.w2, m.b2, mid.data(), tokens, rows.data());
    return rows;
}

struct CountingDense final : DenseOps {
    ReferenceEncoder* ref = nullptr;
    int linears = 0;
    void linear(const Tensor& w, const Tensor& b, const float* x, std::size_t rows, float* y) override {
        ++linears;
        ref->linear(w, b, x, rows, y);
    }
};

void test_merge_project() {
    EncoderConfig cfg = base_config();
    Tower t = make_tower(cfg, 81, 12);
    ReferenceEncoder ref(cfg, t.w);
    vision_toy::Rng rng{82};
    const unsigned W = 4, R = 3, P = 7, O = 5;
    std::vector<float> w1(std::size_t{P} * W * R * R), b1(P), w2(std::size_t{O} * P), b2(O);
    for (auto* v : {&w1, &b1, &w2, &b2})
        for (float& x : *v) x = 2.0F * rng.next();
    for (const unsigned gh : {5U, 6U})
        for (const unsigned gw : {8U, 3U})
            for (const bool row_major : {true, false})
                for (const Activation act : {Activation::gelu, Activation::gelu_tanh, Activation::silu}) {
                    EncoderOutput h;
                    h.rows = gh * gw;
                    h.width = W;
                    h.hidden.resize(std::size_t{h.rows} * W);
                    for (float& x : h.hidden) x = 3.0F * rng.next();
                    MergeProjector m;
                    m.ratio = R;
                    m.row_major = row_major;
                    m.act = act;
                    m.w1 = Tensor{w1.data(), Tensor::Type::f32, P, std::size_t{W} * R * R};
                    m.b1 = Tensor{b1.data(), Tensor::Type::f32, P, 1};
                    m.w2 = Tensor{w2.data(), Tensor::Type::f32, O, P};
                    m.b2 = Tensor{b2.data(), Tensor::Type::f32, O, 1};
                    CountingDense dense;
                    dense.ref = &ref;
                    const std::vector<float> want = old_projector(h, gh, gw, m, dense);
                    dense.linears = 0;
                    std::vector<float> got(want.size(), -1.0F);
                    dense.merge_project(h, gh, gw, m, got.data());
                    CHECK(dense.linears == 2);
                    CHECK(std::size_t{m.tokens(gh, gw)} * O == want.size());
                    CHECK(std::memcmp(got.data(), want.data(), want.size() * 4U) == 0);  // bit for bit
                    CHECK(throws([&] { dense.merge_project(h, gh + 1U, gw, m, got.data()); }));
                }
    CHECK(activate(Activation::gelu, 0.0F) == 0.0F && activate(Activation::silu, 0.0F) == 0.0F);
    CHECK(throws([] {
        MergeProjector z;
        z.ratio = 0;
        (void)z.tokens(3, 3);
    }));
}

void test_merge_project_overflow() {
    MergeProjector m;
    // A failing old implementation stops here, before exercising its unsafe loops.
    const bool rejects_wrap = throws([&] { (void)m.tokens(65536U, 65537U); });
    CHECK(rejects_wrap);
    if (!rejects_wrap) return;
    EncoderOutput h;
    h.rows = 65536U;
    h.width = 1;
    h.hidden.resize(h.rows);
    float weight = 1.0F, output = 17.0F;
    m.w1 = m.w2 = Tensor{&weight, Tensor::Type::f32, 1, 1};
    CountingDense dense;
    CHECK(throws([&] { dense.merge_project(h, 65536U, 65537U, m, &output); }));
    // The resident HIP path calls the same validator without requiring host hidden storage.
    CHECK(throws([&] { (void)layout::projector_geometry(h, 65536U, 65537U, m, &output, true); }));
    CHECK(dense.linears == 0 && output == 17.0F);
    constexpr unsigned max = std::numeric_limits<unsigned>::max();
    m.ratio = 3;
    CHECK(m.tokens(max - 1U, 1U) == max / 3U);  // ceil addition used to wrap
    m.ratio = 2;
    CHECK(throws([&] { (void)m.tokens(max, 1U); }));  // padded coordinate would wrap
    m.ratio = max;
    CHECK(throws([&] { (void)m.tokens(1U, 1U); }));  // ratio squared would wrap
    m.ratio = 1;
    CHECK(throws([&] { (void)m.tokens(0U, 1U); }));
    h.rows = 1;
    h.width = max;
    m.ratio = 65535U;
    CHECK(throws([&] { (void)layout::projector_geometry(h, 1U, 1U, m, &output, true); }));
}

void test_resident_owner() {
    const std::shared_ptr<const void> a = std::make_shared<const char>(char{0}), b = std::make_shared<const char>(char{0});
    EncoderOutput h;
    h.device_epoch = 1;
    h.device_owner = a;
    CHECK(h.resident_for(a, 1));
    CHECK(!h.resident_for(b, 1));
    CHECK(!h.resident_for(a, 2));
    h.device_epoch = 0;
    CHECK(!h.resident_for(a, 0));
    h.device_owner.reset();
    CHECK(!h.resident_for({}, 0));
}

// prepare_backend: the projector linears and the workspace go through the backend hooks; without warm-up nothing is
// encoded, with it the adapter's image path runs once (on the path encode_cached picks) and nothing is cached.
struct RecordingBackend final : EncoderBackend {
    ReferenceEncoder* ref = nullptr;
    std::vector<const void*> prepared;
    unsigned reserved_rows = 0, reserved_tiles = 0;
    int encodes = 0, resident_encodes = 0;
    EncoderOutput encode(const PatchGrid& g, const PositionInputs& p) override {
        ++encodes;
        return ref->encode(g, p);
    }
    EncoderOutput encode_for_projection(const PatchGrid& g, const PositionInputs& p) override {
        ++resident_encodes;
        return ref->encode(g, p);
    }
    void linear(const Tensor& w, const Tensor& b, const float* x, std::size_t rows, float* y) override { ref->linear(w, b, x, rows, y); }
    void prepare_linear(const Tensor& w, const Tensor&) override { prepared.push_back(w.data); }
    void reserve(unsigned rows, unsigned tiles) override {
        reserved_rows = rows;
        reserved_tiles = tiles;
    }
    const char* arithmetic() const override { return "recording"; }
};

class PreparedAdapter final : public VisionAdapter {
public:
    PreparedAdapter(Tower& t, bool resident) : toy_(t), tower_(t), resident_(resident) {}
    EncoderConfig config() const override { return toy_.config(); }
    std::string identity() const override { return toy_.identity(); }
    ResizeRule resize() const override { return toy_.resize(); }
    PatchGrid patches(const Image&) const override { return make_grid(tower_.cfg, 2, 3, 91); }
    void bind(WeightSource& s, EncoderWeights& w) override { toy_.bind(s, w); }
    void project(EncoderOutput& h, const PatchGrid& g, DenseOps& d, ProjectedRows& o) override { toy_.project(h, g, d, o); }
    SpanLayout layout(const PatchGrid& g) const override { return toy_.layout(g); }
    std::vector<std::pair<Tensor, Tensor>> projector_linears() const override {
        return {{Tensor{tower_.projector.data(), Tensor::Type::f32, tower_.projector_out, tower_.cfg.width}, Tensor{}}};
    }
    unsigned max_encoder_rows() const override { return 77; }
    bool device_resident_hidden() const override { return resident_; }

private:
    ToyAdapter toy_;
    Tower& tower_;
    bool resident_;
};

void test_prepare_backend() {
    EncoderConfig cfg = base_config();
    Tower t = make_tower(cfg, 93, 12);
    ReferenceEncoder ref(cfg, t.w);
    for (const bool resident : {false, true}) {
        PreparedAdapter adapter(t, resident);
        RecordingBackend backend;
        backend.ref = &ref;
        PrepareReport r = prepare_backend(adapter, backend, t.w, false);
        CHECK(r.linears == 1 && backend.prepared.size() == 1 && backend.prepared[0] == t.projector.data());
        CHECK(r.reserved_rows == 77 && backend.reserved_rows == 77 && backend.reserved_tiles == 1);
        CHECK(backend.encodes == 0 && backend.resident_encodes == 0 && r.warmup_rows == 0);
        r = prepare_backend(adapter, backend, t.w, true);
        CHECK(r.warmup_rows == 6 && backend.encodes + backend.resident_encodes == 1);
        CHECK((resident ? backend.resident_encodes : backend.encodes) == 1);
    }
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
        if (argc == 3 && std::strcmp(argv[1], "--project-device") == 0) {
            const bool device = std::strcmp(argv[2], "on") == 0;
            CHECK(Encoder::project_device() == device);
            CHECK(std::strcmp(Encoder::effective_arithmetic(1U), device ? "vision-f16-v1-project-device-f64-v1" : "vision-f16-v1") == 0);
            CHECK(std::strcmp(Encoder::effective_arithmetic(2U), device ? "vision-f16-attn2-v1-project-device-f64-v1" : "vision-f16-attn2-v1") == 0);
            return failures ? 1 : 0;
        }
        test_bf16();
        test_validate();
        test_positions();
        test_single_row();
        test_mean_attention();
        test_permutation();
        test_rope_translation();
        test_tiles();
        test_taps_and_types();
        test_plan();
        test_two_term_model();
        test_two_term_large_head();
        test_f16_rounding_modes();
        test_attention_terms();
        test_cache();
        test_encode_cached();
        test_merge_project();
        test_merge_project_overflow();
        test_resident_owner();
        test_prepare_backend();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    if (failures) return 1;
    std::puts("vision-encoder: PASS");
    return 0;
}
