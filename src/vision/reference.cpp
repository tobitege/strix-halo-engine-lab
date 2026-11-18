// Host side of the shared vision encoder: config/shape validation, default positions, the device plan and the FP32
// reference tower (the fixture oracle; slow by design: double accumulation, no approximation anywhere).
#include "strix/vision/encoder.hpp"

#include "../attn_core/two_term_model.hpp"
#include "encoder_layout.hpp"
#include "strix/env_flags.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace strix::vision {
namespace {

[[noreturn]] void bad(const std::string& what) { throw std::invalid_argument("vision encoder: " + what); }

using attn_core::model::half_to_float;

// fn(begin, end) over [0, n) on the host threads; small jobs run inline.
void parallel_for(std::size_t n, std::size_t work_per_item, const std::function<void(std::size_t, std::size_t)>& fn) {
    const std::size_t hw = std::max<std::size_t>(1U, std::thread::hardware_concurrency());
    const std::size_t workers = n * work_per_item < (std::size_t{1} << 18U) ? 1U : std::min(hw, n);
    if (workers <= 1U) {
        fn(0, n);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (std::size_t t = 0; t < workers; ++t) pool.emplace_back([&, t] { fn(n * t / workers, n * (t + 1U) / workers); });
    for (auto& th : pool) th.join();
}

void check(const Tensor& t, std::size_t rows, std::size_t cols, const char* name, bool required = true) {
    if (!t) {
        if (required) bad(std::string("missing tensor ") + name);
        return;
    }
    if (t.rows != rows || t.cols != cols) bad(std::string("shape of ") + name);
}

void linear_ref(const Tensor& w, const Tensor& bias, const float* x, std::size_t rows, float* y);

// linear_ref on FP16-rounded inputs when the rounding model asks for it (the default path is linear_ref unchanged).
void linear_model(const ReferenceRounding& m, const Tensor& w, const Tensor& bias, const float* x, std::size_t rows, float* y) {
    if (!m.linear_inputs_f16) return linear_ref(w, bias, x, rows, y);
    std::vector<float> xr(x, x + rows * w.cols);
    for (float& v : xr) v = round_to_f16(v);
    linear_ref(w, bias, xr.data(), rows, y);
}

void linear_ref(const Tensor& w, const Tensor& bias, const float* x, std::size_t rows, float* y) {
    const std::size_t out = w.rows, in = w.cols;
    if (!w || !x || !y) bad("linear operands");
    if (bias && bias.size() != out) bad("linear bias");
    parallel_for(out, rows * in, [&](std::size_t o0, std::size_t o1) {
        std::vector<float> wr(in);
        for (std::size_t o = o0; o < o1; ++o) {
            for (std::size_t k = 0; k < in; ++k) wr[k] = w.at(o * in + k);
            const double b = bias ? static_cast<double>(bias.at(o)) : 0.0;
            for (std::size_t r = 0; r < rows; ++r) {
                const float* xr = x + r * in;
                double acc = 0.0;
                for (std::size_t k = 0; k < in; ++k) acc += static_cast<double>(xr[k]) * static_cast<double>(wr[k]);
                y[r * out + o] = static_cast<float>(acc + b);
            }
        }
    });
}

void norm_ref(const EncoderConfig& c, const Tensor& w, const Tensor& b, const float* x, std::size_t rows, float* y) {
    const std::size_t n = c.width;
    std::vector<float> wv(n), bv(n, 0.0F);
    for (std::size_t i = 0; i < n; ++i) wv[i] = w.at(i);
    if (b)
        for (std::size_t i = 0; i < n; ++i) bv[i] = b.at(i);
    for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x + r * n;
        double mean = 0.0, sq = 0.0;
        if (c.norm == EncoderConfig::Norm::layernorm) {
            for (std::size_t i = 0; i < n; ++i) mean += static_cast<double>(xr[i]);
            mean /= static_cast<double>(n);
        }
        for (std::size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(xr[i]) - mean;
            sq += d * d;
        }
        const double inv = 1.0 / std::sqrt(sq / static_cast<double>(n) + static_cast<double>(c.eps));
        for (std::size_t i = 0; i < n; ++i)
            y[r * n + i] = static_cast<float>((static_cast<double>(xr[i]) - mean) * inv * static_cast<double>(wv[i]) + static_cast<double>(bv[i]));
    }
}

double activation(const EncoderConfig& c, double v) {
    if (c.mlp == EncoderConfig::Mlp::gelu) return 0.5 * v * (1.0 + std::erf(v * 0.70710678118654752440));
    return 0.5 * v * (1.0 + std::tanh(0.79788456080286535588 * (v + 0.044715 * v * v * v)));
}

// In-place rotation of one head vector [D] by its row's angles [D / 2].
void rotate(const EncoderConfig& c, float* v, const float* angle) {
    const unsigned d = c.head_dim(), half = d / 2U;
    for (unsigned i = 0; i < half; ++i) {
        const unsigned a = c.rope_pairing == EncoderConfig::RopePairing::half ? i : 2U * i;
        const unsigned b = c.rope_pairing == EncoderConfig::RopePairing::half ? i + half : 2U * i + 1U;
        const double cs = std::cos(static_cast<double>(angle[i])), sn = std::sin(static_cast<double>(angle[i]));
        const double va = static_cast<double>(v[a]), vb = static_cast<double>(v[b]);
        v[a] = static_cast<float>(va * cs - vb * sn);
        v[b] = static_cast<float>(vb * cs + va * sn);
    }
}

}  // namespace

float Tensor::at(std::size_t i) const {
    switch (type) {
    case Type::f32: return static_cast<const float*>(data)[i];
    case Type::bf16: return from_bf16(static_cast<const std::uint16_t*>(data)[i]);
    case Type::f16: return half_to_float(static_cast<const std::uint16_t*>(data)[i]);
    }
    return 0.0F;
}

std::size_t ProjectedRows::bytes() const {
    std::size_t b = bf16.size() * 2U;
    for (const auto& d : deep) b += d.size() * 2U;
    return b + sizeof(ProjectedRows);
}

std::uint16_t to_bf16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    if ((bits & 0x7F800000U) == 0x7F800000U) {  // inf / NaN: keep the class
        const bool nan = (bits & 0x007FFFFFU) != 0U;
        return static_cast<std::uint16_t>((bits >> 16U) | (nan ? 0x0040U : 0U));
    }
    bits += 0x7FFFU + ((bits >> 16U) & 1U);  // round to nearest, ties to even
    return static_cast<std::uint16_t>(bits >> 16U);
}
float from_bf16(std::uint16_t h) {
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16U;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
std::vector<std::uint16_t> to_bf16_rows(const float* rows, std::size_t count) {
    std::vector<std::uint16_t> out(count);
    for (std::size_t i = 0; i < count; ++i) out[i] = to_bf16(rows[i]);
    return out;
}

void validate(const EncoderConfig& c) {
    if (!c.layers || !c.width || !c.heads || !c.mlp_width) bad("zero dimension");
    if (c.width % c.heads) bad("width is not a multiple of heads");
    if (c.head_dim() % 2U) bad("odd head dimension");
    if (c.has_rope() && c.head_dim() % 4U) bad("2D rotary needs head_dim % 4 == 0");
    if (c.has_rope() && !(c.rope_theta > 1.0F)) bad("rope_theta");
    if (c.has_learned() && !c.pos_grid) bad("learned positions need pos_grid");
    if (!c.patch || !c.temporal || !c.channels) bad("patch geometry");
    if (!(c.eps > 0.0F)) bad("eps");
    if (c.swiglu_limit < 0.0F) bad("swiglu_limit");
    for (std::size_t i = 0; i < c.deepstack_taps.size(); ++i) {
        if (c.deepstack_taps[i] >= c.layers) bad("deepstack tap beyond the last block");
        if (i && c.deepstack_taps[i] <= c.deepstack_taps[i - 1U]) bad("deepstack taps must ascend");
    }
}

void validate(const EncoderConfig& c, const EncoderWeights& w) {
    validate(c);
    const std::size_t width = c.width;
    const bool ln = c.norm == EncoderConfig::Norm::layernorm;
    check(w.patch_w, width, c.patch_dim(), "patch_w");
    check(w.patch_b, width, 1, "patch_b", c.patch_bias);
    if (c.has_learned()) check(w.pos_embed, std::size_t{c.pos_grid} * c.pos_grid, width, "pos_embed");
    if (c.final_norm) {
        check(w.final_norm_w, width, 1, "final_norm_w");
        check(w.final_norm_b, width, 1, "final_norm_b", ln);
    }
    if (w.blocks.size() != c.layers) bad("block count");
    for (const auto& b : w.blocks) {
        check(b.norm1_w, width, 1, "norm1_w");
        check(b.norm1_b, width, 1, "norm1_b", ln);
        check(b.qkv_w, 3U * width, width, "qkv_w");
        check(b.qkv_b, 3U * width, 1, "qkv_b", c.qkv_bias);
        check(b.out_w, width, width, "out_w");
        check(b.out_b, width, 1, "out_b", c.out_bias);
        check(b.norm2_w, width, 1, "norm2_w");
        check(b.norm2_b, width, 1, "norm2_b", ln);
        check(b.mlp1_w, c.mlp_out(), width, "mlp1_w");
        check(b.mlp1_b, c.mlp_out(), 1, "mlp1_b", c.mlp_bias);
        check(b.mlp2_w, width, c.mlp_width, "mlp2_w");
        check(b.mlp2_b, width, 1, "mlp2_b", c.mlp_bias);
    }
}

void validate(const EncoderConfig& c, const PatchGrid& g, const PositionInputs& p) {
    if (!g.rows || g.patch_dim != c.patch_dim()) bad("patch grid shape");
    if (g.data.size() != std::size_t{g.rows} * g.patch_dim) bad("patch data size");
    if (g.y.size() != g.rows || g.x.size() != g.rows) bad("patch coordinates");
    unsigned prev = 0;
    for (const unsigned e : g.tile_end) {
        if (e <= prev || e > g.rows) bad("attention tiles");
        prev = e;
    }
    if (!g.tile_end.empty() && g.tile_end.back() != g.rows) bad("attention tiles do not cover the rows");
    if (!p.addend.empty() && p.addend.size() != std::size_t{g.rows} * c.width) bad("position addend size");
    if (c.has_learned() && p.addend.empty()) bad("learned positions without an addend");
    const std::size_t angles = c.has_rope() ? std::size_t{g.rows} * (c.head_dim() / 2U) : 0U;
    if (p.angles.size() != angles) bad("rotary angle size");
}

PositionInputs default_positions(const EncoderConfig& c, const EncoderWeights& w, const PatchGrid& g) {
    validate(c);
    PositionInputs p;
    if (c.has_learned()) {
        if (!w.pos_embed || w.pos_embed.rows != std::size_t{c.pos_grid} * c.pos_grid || w.pos_embed.cols != c.width) bad("pos_embed");
        if (!g.grid_h || !g.grid_w) bad("patch grid size");
        const unsigned G = c.pos_grid, W = c.width;
        p.addend.resize(std::size_t{g.rows} * W);
        const auto coord = [G](unsigned i, unsigned n) { return n > 1U ? static_cast<float>(i) * static_cast<float>(G - 1U) / static_cast<float>(n - 1U) : 0.0F; };
        for (unsigned r = 0; r < g.rows; ++r) {
            if (g.y[r] >= g.grid_h || g.x[r] >= g.grid_w) bad("patch coordinate outside the grid");
            const float fy = coord(g.y[r], g.grid_h), fx = coord(g.x[r], g.grid_w);
            const unsigned y0 = static_cast<unsigned>(fy), x0 = static_cast<unsigned>(fx);
            const unsigned y1 = std::min(y0 + 1U, G - 1U), x1 = std::min(x0 + 1U, G - 1U);
            const float dy = fy - static_cast<float>(y0), dx = fx - static_cast<float>(x0);
            const float wt[4] = {(1.0F - dy) * (1.0F - dx), (1.0F - dy) * dx, dy * (1.0F - dx), dy * dx};
            const std::size_t idx[4] = {std::size_t{y0} * G + x0, std::size_t{y0} * G + x1, std::size_t{y1} * G + x0, std::size_t{y1} * G + x1};
            for (unsigned k = 0; k < W; ++k) {
                float acc = 0.0F;
                for (unsigned j = 0; j < 4U; ++j) acc += w.pos_embed.at(idx[j] * W + k) * wt[j];
                p.addend[std::size_t{r} * W + k] = acc;
            }
        }
    }
    if (c.has_rope()) {
        const unsigned half = c.head_dim() / 2U, nf = half / 2U;
        std::vector<float> inv(nf);
        for (unsigned f = 0; f < nf; ++f)
            inv[f] = 1.0F / static_cast<float>(std::pow(static_cast<double>(c.rope_theta), static_cast<double>(static_cast<float>(2U * f) / static_cast<float>(half))));
        p.angles.resize(std::size_t{g.rows} * half);
        for (unsigned r = 0; r < g.rows; ++r)
            for (unsigned f = 0; f < nf; ++f) {
                p.angles[std::size_t{r} * half + f] = static_cast<float>(g.y[r]) * inv[f];
                p.angles[std::size_t{r} * half + nf + f] = static_cast<float>(g.x[r]) * inv[f];
            }
    }
    return p;
}

bool hip_supported(const EncoderConfig& c, std::string* why) {
    const auto no = [why](const char* text) {
        if (why) *why = text;
        return false;
    };
    try {
        validate(c);
    } catch (const std::invalid_argument& e) {
        return no(e.what());
    }
    if (!layout::head_pad(c.head_dim())) return no("head dimension must be 64, 72 or 80");
    if (c.heads > 64U) return no("more than 64 heads");
    return true;
}

EncoderPlan plan(const EncoderConfig& c, unsigned rows, unsigned tiles, const EncoderOptions& opt) {
    std::string why;
    if (!hip_supported(c, &why)) bad(why);
    if (!rows || !tiles || tiles > rows) bad("plan rows / tiles");
    // Upper bound of the key slots for `tiles` tiles covering `rows` rows: each tile rounds up by at most 15.
    const std::uint64_t slots = (2ULL * rows + 15ULL * tiles + 15ULL) & ~15ULL;
    if (slots > std::numeric_limits<unsigned>::max()) bad("too many rows");
    const auto g = layout::geometry(c, rows, static_cast<unsigned>(slots), opt);
    EncoderPlan p;
    p.head_pad = g.dp;
    p.mlp_chunk_rows = g.chunk;
    p.weight_bytes = layout::weight_bytes(c);
    p.fixed_bytes = g.fixed();
    p.mlp_bytes = g.mlp();
    return p;
}

unsigned attention_terms_from_env() {
    // Default 1: the main rows. Two terms change the tower bits and the served answer of one fixture, cost +14 % per
    // encode, and the device gate re-check still fails on preprocessing; 2 stays opt-in until that gate passes.
    static const unsigned terms = envcfg::number<unsigned>("STRIX_VISION_ATTN_TERMS", 1U, 1U, 2U);
    return terms;
}

bool resident_from_env() {
    const char* v = std::getenv("STRIX_VISION_RESIDENT");
    return !v || !*v || std::strcmp(v, "0") != 0;
}

void layout::parallel_ranges(std::size_t n, std::size_t work_per_item, const std::function<void(std::size_t, std::size_t)>& fn) {
    parallel_for(n, work_per_item, fn);
}

float round_to_f16(float f) { return attn_core::model::round_to_f16(f); }

float round_f16_terms(float f, unsigned terms) {
    if (terms == 0U) return f;
    const float hi = round_to_f16(f);
    if (terms == 1U || !std::isfinite(hi)) return hi;
    return hi + round_to_f16(f - hi);  // f - hi is exact in FP32 (Sterbenz); the sum is exact unless lo rounds
}

ReferenceEncoder::ReferenceEncoder(EncoderConfig c, const EncoderWeights& w, ReferenceRounding r)
    : cfg_(std::move(c)), weights_(&w), rounding_(r) {
    validate(cfg_, w);
}
EncoderOutput ReferenceEncoder::encode(const PatchGrid& g, const PositionInputs& p) { return reference_encode(cfg_, *weights_, g, p, rounding_); }
void ReferenceEncoder::linear(const Tensor& w, const Tensor& bias, const float* x, std::size_t rows, float* y) {
    linear_model(rounding_, w, bias, x, rows, y);
}

EncoderOutput reference_encode(const EncoderConfig& c, const EncoderWeights& w, const PatchGrid& g, const PositionInputs& p,
                               const ReferenceRounding& m) {
    validate(c, w);
    validate(c, g, p);
    const std::size_t N = g.rows, W = c.width, H = c.heads, D = c.head_dim(), M = c.mlp_width, M1 = c.mlp_out();
    std::vector<float> x(N * W), n(N * W), qkv(N * 3U * W), att(N * W), o(N * W), mid(N * M1), act(N * M);
    const auto lin = [&m](const Tensor& wt, const Tensor& bias, const float* in, std::size_t rows, float* y) {
        linear_model(m, wt, bias, in, rows, y);
    };
    lin(w.patch_w, w.patch_b, g.data.data(), N, x.data());
    for (std::size_t i = 0; i < p.addend.size(); ++i) x[i] += p.addend[i];
    std::vector<unsigned> tiles = g.tile_end;
    if (tiles.empty()) tiles.push_back(g.rows);
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    EncoderOutput out;
    out.rows = g.rows;
    out.width = c.width;
    std::size_t tap = 0;
    for (unsigned l = 0; l < c.layers; ++l) {
        const auto& b = w.blocks[l];
        norm_ref(c, b.norm1_w, b.norm1_b, x.data(), N, n.data());
        lin(b.qkv_w, b.qkv_b, n.data(), N, qkv.data());
        if (c.has_rope())
            for (std::size_t r = 0; r < N; ++r)
                for (std::size_t h = 0; h < H; ++h) {
                    rotate(c, &qkv[r * 3U * W + h * D], &p.angles[r * (D / 2U)]);
                    rotate(c, &qkv[r * 3U * W + W + h * D], &p.angles[r * (D / 2U)]);
                }
        if (m.any_attention() && !m.attn_order)  // the flash core's Q, K, V operands (qkv is read by the attention only)
            for (std::size_t r = 0; r < N; ++r)
                for (std::size_t i = 0; i < 3U * W; ++i) {
                    float& v = qkv[r * 3U * W + i];
                    v = round_f16_terms(v, i < W ? m.attn_q : i < 2U * W ? m.attn_k : m.attn_v);
                }
        // The core's constant: FP32 1/sqrt(D) times FP32 log2(e), rounded once more (the policy's kScaleLog2).
        const float scale_log2 = static_cast<float>(1.0 / std::sqrt(static_cast<double>(D))) * 1.44269504088896341F;
        std::vector<attn_core::model::Terms> qkv_terms(m.attn_order ? qkv.size() : 0U);
        for (std::size_t i = 0; i < qkv_terms.size(); ++i) qkv_terms[i] = attn_core::model::split(qkv[i], attn_core::model::Format::f16, m.attn_q);
        std::size_t begin = 0;
        for (const unsigned end : tiles) {
            const std::size_t nk = end - begin;
            parallel_for(nk, nk * W, [&](std::size_t r0, std::size_t r1) {
                std::vector<double> s(nk), acc(D);
                for (std::size_t r = begin + r0; r < begin + r1; ++r)
                    for (std::size_t h = 0; h < H; ++h) {
                        const float* q = &qkv[r * 3U * W + h * D];
                        if (m.attn_order) {  // the dense core's three products and their order (two_term_model.hpp)
                            const auto at = [&](std::size_t j, std::size_t part) { return &qkv_terms[(begin + j) * 3U * W + part * W + h * D]; };
                            attn_core::model::dense_row(
                                &qkv_terms[r * 3U * W + h * D], static_cast<unsigned>(nk), [&](unsigned j) { return at(j, 1U); }, [&](unsigned j) { return at(j, 2U); },
                                static_cast<unsigned>(D), scale_log2, m.attn_q, attn_core::model::Format::f16, &att[r * W + h * D]);
                            continue;
                        }
                        double top = -std::numeric_limits<double>::infinity();
                        for (std::size_t j = 0; j < nk; ++j) {
                            const float* k = &qkv[(begin + j) * 3U * W + W + h * D];
                            double dot = 0.0;
                            for (std::size_t d = 0; d < D; ++d) dot += static_cast<double>(q[d]) * static_cast<double>(k[d]);
                            s[j] = dot * scale;
                            top = std::max(top, s[j]);
                        }
                        double sum = 0.0;
                        std::fill(acc.begin(), acc.end(), 0.0);
                        const std::size_t step = m.attn_p ? 16U : nk;
                        if (m.attn_p) top = -std::numeric_limits<double>::infinity();
                        for (std::size_t j0 = 0; j0 < nk; j0 += step) {
                            const std::size_t j1 = std::min(j0 + step, nk);
                            if (m.attn_p) {
                                double next = top;
                                for (std::size_t j = j0; j < j1; ++j) next = std::max(next, s[j]);
                                const double alpha = std::exp(top - next);
                                sum *= alpha;
                                for (double& v : acc) v *= alpha;
                                top = next;
                            }
                            for (std::size_t j = j0; j < j1; ++j) {
                                // Round at this online step's maximum; earlier contributions are only rescaled.
                                const double e = std::exp(s[j] - top);
                                const double pj = m.attn_p ? static_cast<double>(round_f16_terms(static_cast<float>(e), m.attn_p)) : e;
                                sum += pj;
                                const float* v = &qkv[(begin + j) * 3U * W + 2U * W + h * D];
                                for (std::size_t d = 0; d < D; ++d) acc[d] += pj * static_cast<double>(v[d]);
                            }
                        }
                        for (std::size_t d = 0; d < D; ++d) att[r * W + h * D + d] = static_cast<float>(acc[d] / sum);
                    }
            });
            begin = end;
        }
        lin(b.out_w, b.out_b, att.data(), N, o.data());
        for (std::size_t i = 0; i < N * W; ++i) x[i] += o[i];
        norm_ref(c, b.norm2_w, b.norm2_b, x.data(), N, n.data());
        lin(b.mlp1_w, b.mlp1_b, n.data(), N, mid.data());
        if (c.mlp == EncoderConfig::Mlp::swiglu) {
            const double lim = static_cast<double>(c.swiglu_limit);
            for (std::size_t r = 0; r < N; ++r)
                for (std::size_t j = 0; j < M; ++j) {
                    double gate = static_cast<double>(mid[r * M1 + (c.swiglu_gate_first ? j : M + j)]);
                    double up = static_cast<double>(mid[r * M1 + (c.swiglu_gate_first ? M + j : j)]);
                    if (lim > 0.0) {
                        gate = std::min(gate, lim);
                        up = std::clamp(up, -lim, lim);
                    }
                    act[r * M + j] = static_cast<float>(gate / (1.0 + std::exp(-gate)) * up);
                }
        } else {
            for (std::size_t i = 0; i < N * M; ++i) act[i] = static_cast<float>(activation(c, static_cast<double>(mid[i])));
        }
        lin(b.mlp2_w, b.mlp2_b, act.data(), N, o.data());
        for (std::size_t i = 0; i < N * W; ++i) x[i] += o[i];
        if (tap < c.deepstack_taps.size() && c.deepstack_taps[tap] == l) {
            out.taps.push_back(x);
            ++tap;
        }
    }
    if (c.final_norm) {
        norm_ref(c, w.final_norm_w, w.final_norm_b, x.data(), N, n.data());
        out.hidden = std::move(n);
    } else {
        out.hidden = std::move(x);
    }
    return out;
}

void splice_rows_host(float* dst, std::size_t dst_stride, const ProjectedRows& src, const std::uint32_t* dst_row,
                      const std::uint32_t* src_row, std::size_t count) {
    if (!dst || dst_stride < src.width || src.bf16.size() != std::size_t{src.rows} * src.width) bad("splice shape");
    for (std::size_t i = 0; i < count; ++i) {
        if (src_row[i] >= src.rows) bad("splice source row");
        const std::uint16_t* s = &src.bf16[std::size_t{src_row[i]} * src.width];
        float* d = dst + std::size_t{dst_row[i]} * dst_stride;
        for (unsigned k = 0; k < src.width; ++k) d[k] = from_bf16(s[k]);
    }
}

}  // namespace strix::vision
