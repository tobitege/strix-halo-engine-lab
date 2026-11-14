// Host tests of the shared vision encoder: FP32 reference tower on synthetic towers (hand-computed rows and graph
// invariants), default positions, the device plan, the embed cache and the row splice. No GPU, no checkpoint.
#include "strix/vision/embed_cache.hpp"

#include "../src/vision/encoder_layout.hpp"
#include "vision_encoder_toy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    CHECK(p.mlp_bytes <= (std::size_t{96} << 20U) + 3U * strix::vision::layout::kAlign);
    // FP16 panels: 32 x (3.15 M + 1.05 M + 5.77 M + 2.88 M) + patch, two bytes each = 785 MiB
    CHECK(p.weight_bytes > std::size_t{780} << 20U && p.weight_bytes < std::size_t{800} << 20U);
    const EncoderPlan small = plan(c, 1521, 1);
    CHECK(small.fixed_bytes < p.fixed_bytes && small.workspace_bytes() < p.workspace_bytes());
    EncoderOptions split;
    split.split_activations = true;
    CHECK(plan(c, 1521, 1, split).workspace_bytes() > small.workspace_bytes());
    CHECK(plan(c, 1521, 4).fixed_bytes >= small.fixed_bytes);
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

}  // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
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
        test_cache();
        test_encode_cached();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    if (failures) return 1;
    std::puts("vision-encoder: PASS");
    return 0;
}
