#include "strix/ds41_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strix::ds41 {
namespace {

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

} // namespace

float chunked_dot(const float* a, const float* b, std::size_t n) {
    if (b == nullptr) return chunked_dot_of([&](std::size_t i) { return a[i]; }, [&](std::size_t i) { return a[i]; }, n);
    return chunked_dot_of([&](std::size_t i) { return a[i]; }, [&](std::size_t i) { return b[i]; }, n);
}

float lane_dot(const float* a, const float* b, std::size_t n) {
    if (b == nullptr) return lane_dot_of([&](std::size_t i) { return a[i]; }, [&](std::size_t i) { return a[i]; }, n);
    return lane_dot_of([&](std::size_t i) { return a[i]; }, [&](std::size_t i) { return b[i]; }, n);
}

void rms_norm(std::span<const float> x, std::span<const float> weight, float eps, std::span<float> out) {
    require(x.size() == weight.size() && out.size() == x.size() && !x.empty(), "RMSNorm geometry");
    const float mean = chunked_dot(x.data(), nullptr, x.size()) / static_cast<float>(x.size());
    const float r = declared_rsqrt(mean + eps);
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = weight[i] * (x[i] * r);
}

std::array<float, hc_mix_count> hc_raw_mixes(std::span<const float> stream, const HcSiteWeights& site, float norm_eps) {
    const std::size_t n = stream.size();
    require(n % hc_copies == 0 && site.fn.size() == n * hc_mix_count, "hyper-connection mix geometry");
    const float r = declared_rsqrt(chunked_dot(stream.data(), nullptr, n) / static_cast<float>(n) + norm_eps);
    std::array<float, hc_mix_count> mixes{};
    for (std::uint32_t m = 0; m < hc_mix_count; ++m) {
        const std::uint16_t* row = site.fn.data() + static_cast<std::size_t>(m) * n;
        mixes[m] = chunked_dot_of([&](std::size_t i) { return f16_to_float(row[i]); },
                               [&](std::size_t i) { return stream[i]; }, n) * r;
    }
    return mixes;
}

HcMix hc_split_sinkhorn(const std::array<float, hc_mix_count>& m, std::span<const float> base, std::span<const float> scale,
                        std::uint32_t iters, float eps) {
    require(base.size() == hc_mix_count && scale.size() == 3 && iters >= 1, "hyper-connection split geometry");
    constexpr std::uint32_t h = hc_copies;
    HcMix out;
    for (std::uint32_t j = 0; j < h; ++j) out.pre[j] = declared_sigmoid(m[j] * scale[0] + base[j]) + eps;
    for (std::uint32_t j = 0; j < h; ++j) out.post[j] = 2.0F * declared_sigmoid(m[j + h] * scale[1] + base[j + h]);
    auto& c = out.comb;
    for (std::uint32_t j = 0; j < h; ++j)
        for (std::uint32_t k = 0; k < h; ++k) c[j * h + k] = m[2 * h + j * h + k] * scale[2] + base[2 * h + j * h + k];
    // row softmax + eps
    for (std::uint32_t j = 0; j < h; ++j) {
        float mx = c[j * h];
        for (std::uint32_t k = 1; k < h; ++k) mx = c[j * h + k] > mx ? c[j * h + k] : mx;
        float sum = 0.0F;
        for (std::uint32_t k = 0; k < h; ++k) { c[j * h + k] = declared_exp(c[j * h + k] - mx); sum = sum + c[j * h + k]; }
        for (std::uint32_t k = 0; k < h; ++k) c[j * h + k] = c[j * h + k] / sum + eps;
    }
    auto columns = [&] {
        std::array<float, h> sum{};
        for (std::uint32_t k = 0; k < h; ++k)
            for (std::uint32_t j = 0; j < h; ++j) sum[k] = sum[k] + c[j * h + k];
        for (std::uint32_t j = 0; j < h; ++j)
            for (std::uint32_t k = 0; k < h; ++k) c[j * h + k] = c[j * h + k] / (sum[k] + eps);
    };
    auto rows = [&] {
        std::array<float, h> sum{};
        for (std::uint32_t j = 0; j < h; ++j)
            for (std::uint32_t k = 0; k < h; ++k) sum[j] = sum[j] + c[j * h + k];
        for (std::uint32_t j = 0; j < h; ++j)
            for (std::uint32_t k = 0; k < h; ++k) c[j * h + k] = c[j * h + k] / (sum[j] + eps);
    };
    columns();
    for (std::uint32_t it = 1; it < iters; ++it) { rows(); columns(); }
    return out;
}

HcMix hc_mixes(std::span<const float> stream, const HcSiteWeights& site, std::uint32_t iters, float hc_eps, float norm_eps) {
    return hc_split_sinkhorn(hc_raw_mixes(stream, site, norm_eps), site.base, site.scale, iters, hc_eps);
}

std::array<float, hc_copies> hc_identity_pre() { return {1.0F, 0.0F, 0.0F, 0.0F}; }

void hc_collapse(std::span<const float> stream, std::span<const float, hc_copies> pre, std::span<float> out) {
    const std::size_t hidden = out.size();
    require(stream.size() == hidden * hc_copies, "hyper-connection collapse geometry");
    for (std::size_t i = 0; i < hidden; ++i) {
        float y = 0.0F;
        for (std::uint32_t c = 0; c < hc_copies; ++c) y = y + pre[c] * stream[c * hidden + i];
        out[i] = y;
    }
}

void hc_expand(std::span<const float> sub, std::span<const float> residual, const HcMix& mix, std::span<float> out) {
    const std::size_t hidden = sub.size();
    require(residual.size() == hidden * hc_copies && out.size() == residual.size(), "hyper-connection expand geometry");
    for (std::uint32_t k = 0; k < hc_copies; ++k)
        for (std::size_t i = 0; i < hidden; ++i) {
            float s = 0.0F;
            for (std::uint32_t j = 0; j < hc_copies; ++j) s = s + mix.comb[j * hc_copies + k] * residual[j * hidden + i];
            out[k * hidden + i] = mix.post[k] * sub[i] + s;
        }
}

std::vector<double> rope_frequencies(const RopeConfig& c) {
    require(c.dims >= 2 && c.dims % 2 == 0 && c.theta > 1.0, "RoPE configuration");
    const std::uint32_t pairs = c.dims / 2;
    std::vector<double> f(pairs);
    for (std::uint32_t p = 0; p < pairs; ++p) f[p] = 1.0 / std::pow(c.theta, static_cast<double>(2 * p) / c.dims);
    if (c.original_positions > 0.0) {
        const double pi = 3.14159265358979323846;
        auto corrected = [&](double rotations) {
            return c.dims * std::log(c.original_positions / (rotations * 2.0 * pi)) / (2.0 * std::log(c.theta));
        };
        const double low = std::max(std::floor(corrected(c.beta_fast)), 0.0);
        const double high = std::min(std::ceil(corrected(c.beta_slow)), static_cast<double>(c.dims - 1));
        for (std::uint32_t p = 0; p < pairs; ++p) {
            const double ramp = std::clamp((static_cast<double>(p) - low) / std::max(high - low, 1e-3), 0.0, 1.0);
            const double smooth = 1.0 - ramp;
            f[p] = f[p] / c.factor * (1.0 - smooth) + f[p] * smooth;
        }
    }
    return f;
}

RopeTable rope_table(const RopeConfig& config, std::uint64_t first, std::uint64_t count) {
    const auto f = rope_frequencies(config);
    RopeTable t;
    t.pairs = config.dims / 2;
    t.first = first;
    t.count = count;
    t.cos.resize(count * t.pairs);
    t.sin.resize(count * t.pairs);
    for (std::uint64_t q = 0; q < count; ++q)
        for (std::uint32_t p = 0; p < t.pairs; ++p) {
            const double angle = static_cast<double>(first + q) * f[p];
            t.cos[q * t.pairs + p] = static_cast<float>(std::cos(angle));
            t.sin[q * t.pairs + p] = static_cast<float>(std::sin(angle));
        }
    return t;
}

void apply_rope(std::span<float> x, const RopeTable& t, std::uint64_t position, bool inverse) {
    require(x.size() == 2U * t.pairs && position >= t.first && position < t.first + t.count, "RoPE application geometry");
    const std::uint64_t q = position - t.first;
    for (std::uint32_t p = 0; p < t.pairs; ++p) {
        const float c = t.cos[q * t.pairs + p], s0 = t.sin[q * t.pairs + p], s = inverse ? -s0 : s0;
        const float a = x[2 * p], b = x[2 * p + 1];
        x[2 * p] = a * c - b * s;
        x[2 * p + 1] = a * s + b * c;
    }
}

void q8_0_a8_rows(const std::uint8_t* weight, std::size_t width, std::size_t first, std::size_t count, std::size_t per_group,
                  const A8Activation& x, float* out) {
    require(width % 32 == 0 && per_group && count % per_group == 0, "Q8_0 site geometry");
    const std::size_t row_bytes = width / 32 * 34;
    for (std::size_t g = 0; g * per_group < count; ++g) {
        const auto slice = count == per_group ? x : a8_slice(x, g * width, width);
        for (std::size_t o = g * per_group; o < (g + 1) * per_group; ++o)
            out[o] = product_q8_0(std::span<const std::uint8_t>(weight + (first + o) * row_bytes, row_bytes), slice, width);
    }
}

void q8_0_a8_split_rows(const std::uint8_t* weight, std::size_t width, std::size_t count, std::size_t parts, const A8Activation& x,
                        float* out) {
    require(parts && width % (32 * parts) == 0, "K-split Q8_0 site geometry");
    const std::size_t row_bytes = width / 32 * 34, part = width / parts, part_bytes = part / 32 * 34;
    for (std::size_t p = 0; p < parts; ++p) {
        const auto slice = a8_slice(x, p * part, part);
        for (std::size_t o = 0; o < count; ++o) {
            const float h = product_q8_0(std::span<const std::uint8_t>(weight + o * row_bytes + p * part_bytes, part_bytes), slice, part);
            out[o] = p == 0 ? h : out[o] + h;
        }
    }
}

A8Activation a8_slice(const A8Activation& x, std::size_t first, std::size_t count) {
    require(first % 32 == 0 && count % 32 == 0 && first + count <= x.codes.size(), "A8 slice");
    A8Activation out;
    out.codes.assign(x.codes.begin() + static_cast<std::ptrdiff_t>(first), x.codes.begin() + static_cast<std::ptrdiff_t>(first + count));
    out.deltas.assign(x.deltas.begin() + static_cast<std::ptrdiff_t>(first / 32),
                      x.deltas.begin() + static_cast<std::ptrdiff_t>((first + count) / 32));
    out.sums.assign(x.sums.begin() + static_cast<std::ptrdiff_t>(first / 32), x.sums.begin() + static_cast<std::ptrdiff_t>((first + count) / 32));
    return out;
}

A8Activation collapse_norm_a8(std::span<const float> stream, std::span<const float, hc_copies> pre, std::span<const float> weight,
                              float eps, std::span<float> normed) {
    std::vector<float> x(normed.size());
    hc_collapse(stream, pre, x);
    rms_norm(x, weight, eps, normed);
    return quantize_a8(normed);
}

A8Activation inverse_rope_a8(std::span<float> o, std::uint32_t heads, std::uint32_t head_dim, const RopeTable& table,
                             std::uint64_t position) {
    require(o.size() == std::size_t{heads} * head_dim && head_dim >= 2 * table.pairs, "attention output geometry");
    for (std::uint32_t h = 0; h < heads; ++h)
        apply_rope(o.subspan(std::size_t{h} * head_dim + head_dim - 2 * table.pairs, 2 * table.pairs), table, position, true);
    return quantize_a8(o);
}

float f16_row_dot(const std::uint16_t* row, const float* x, std::size_t width) {
    return lane_dot_of([&](std::size_t i) { return f16_to_float(row[i]); }, [&](std::size_t i) { return x[i]; }, width);
}

void f16_rows(const std::uint16_t* weight, std::size_t width, std::size_t first, std::size_t count, const float* x, float* out) {
    for (std::size_t o = 0; o < count; ++o) out[o] = f16_row_dot(weight + (first + o) * width, x, width);
}

void embed_token(const std::uint16_t* table, std::size_t hidden, std::uint32_t token, std::span<float> stream) {
    require(stream.size() == hidden * hc_copies, "embedding geometry");
    const std::uint16_t* row = table + static_cast<std::size_t>(token) * hidden;
    for (std::uint32_t c = 0; c < hc_copies; ++c)
        for (std::size_t i = 0; i < hidden; ++i) stream[c * hidden + i] = f16_to_float(row[i]);
}

ArgmaxCandidate argmax_slice(std::span<const float> logits, std::uint32_t first_index) {
    require(!logits.empty(), "argmax over an empty slice");
    ArgmaxCandidate best{-__builtin_huge_valf(), 0xffffffffU, 1};  // index of a slice without a finite logit
    bool have = false;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const float v = logits[i];
        if (!std::isfinite(v)) { best.finite = 0; continue; }
        if (!have || v > best.value) { best.value = v; best.index = first_index + static_cast<std::uint32_t>(i); have = true; }
    }
    return best;
}

ArgmaxCandidate combine_argmax(const ArgmaxCandidate& a, const ArgmaxCandidate& b) {
    ArgmaxCandidate out = (b.value > a.value || (b.value == a.value && b.index < a.index)) ? b : a;
    out.finite = a.finite & b.finite;
    return out;
}

VocabSlice vocab_slice(std::uint32_t vocab, std::uint32_t ranks, std::uint32_t rank) {
    require(ranks >= 1 && rank < ranks && vocab >= ranks, "vocabulary split");
    const std::uint32_t base = vocab / ranks, extra = vocab % ranks;
    const std::uint32_t first = rank * base + (rank < extra ? rank : extra);
    return {first, base + (rank < extra ? 1U : 0U)};
}

void engram_rows_to_bf16(std::span<const std::uint8_t> rows, std::span<float> out) {
    require(rows.size() % engram_row_bytes == 0 && out.size() == rows.size() / engram_row_bytes * engram_row_elements,
            "Engram row geometry");
    for (std::size_t r = 0; r < rows.size() / engram_row_bytes; ++r) {
        decode_engram_row(std::span<const std::uint8_t, engram_row_bytes>(rows.data() + r * engram_row_bytes, engram_row_bytes),
                          std::span<float, engram_row_elements>(out.data() + r * engram_row_elements, engram_row_elements));
        for (std::size_t i = 0; i < engram_row_elements; ++i)
            out[r * engram_row_elements + i] = round_to_bf16(out[r * engram_row_elements + i]);
    }
}

std::array<float, hc_copies> engram_gates(std::span<const float> stream, std::span<const float> kv, std::span<const float> qn,
                                          std::span<const float> kn, float eps) {
    const std::size_t hidden = stream.size() / hc_copies;
    require(stream.size() == hidden * hc_copies && kv.size() == hidden * (hc_copies + 1) && qn.size() == stream.size() &&
                kn.size() == stream.size(), "Engram gate geometry");
    const float inv_sqrt_hidden = static_cast<float>(1.0 / std::sqrt(static_cast<double>(hidden)));
    std::array<float, hc_copies> gates{};
    for (std::uint32_t c = 0; c < hc_copies; ++c) {
        const float* h = stream.data() + c * hidden;
        const float* key = kv.data() + c * hidden;
        const float* q = qn.data() + c * hidden;
        const float* k = kn.data() + c * hidden;
        const float rh = declared_rsqrt(chunked_dot(h, nullptr, hidden) / static_cast<float>(hidden) + eps);
        const float rk = declared_rsqrt(chunked_dot(key, nullptr, hidden) / static_cast<float>(hidden) + eps);
        const float dot_sum = chunked_dot_of([&](std::size_t i) { return h[i] * (q[i] * k[i]); },
                                          [&](std::size_t i) { return key[i]; }, hidden);
        const float dot = (dot_sum * (rh * rk)) * inv_sqrt_hidden;
        gates[c] = engram_gate_of(dot);
    }
    return gates;
}

void engram_inject(std::span<float> stream, std::span<const float> kv, std::span<const float> qn, std::span<const float> kn,
                   float eps, bool gate_off) {
    auto gates = engram_gates(stream, kv, qn, kn, eps);
    if (gate_off) gates.fill(0.0F);
    const std::size_t hidden = stream.size() / hc_copies;
    const float* value = kv.data() + hc_copies * hidden;
    for (std::uint32_t c = 0; c < hc_copies; ++c)
        for (std::size_t i = 0; i < hidden; ++i) stream[c * hidden + i] = stream[c * hidden + i] + gates[c] * value[i];
}

} // namespace strix::ds41
