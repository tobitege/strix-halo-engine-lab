#include "strix/serve/prompt_scorer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace strix::serve::pflash {
namespace {

using std::size_t;

[[noreturn]] void bad(const char* what) {
    throw std::invalid_argument(std::string("pflash scorer: ") + what);
}

size_t token_count(std::span<const float> residual, std::uint32_t hidden) {
    if (hidden == 0 || residual.size() % hidden != 0) {
        bad("residual size is not a multiple of hidden");
    }
    return residual.size() / hidden;
}

double dot(const float* a, const float* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return sum;
}

// out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
void rms_norm(const float* x, const float* weight, float eps, size_t n, float* out) {
    const double scale = 1.0 / std::sqrt(dot(x, x, n) / static_cast<double>(n) + static_cast<double>(eps));
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(static_cast<double>(x[i]) * scale * static_cast<double>(weight[i]));
    }
}

// out[d] = rms_norm(rows[d] . in) * weight[d], for `dim` consecutive rows of a [..][hidden] matrix.
void project(const float* rows, const float* in, size_t hidden, const float* weight, float eps, size_t dim,
             std::vector<float>& scratch, float* out) {
    scratch.resize(dim);
    for (size_t d = 0; d < dim; ++d) {
        scratch[d] = static_cast<float>(dot(rows + d * hidden, in, hidden));
    }
    rms_norm(scratch.data(), weight, eps, dim, out);
}

double gelu(double x, Gelu kind) {
    if (kind == Gelu::erf) {
        return 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440));
    }
    return 0.5 * x * (1.0 + std::tanh(0.79788456080286535588 * (x + 0.044715 * x * x * x)));
}

}  // namespace

void validate(const QkHead& head) {
    if (head.hidden == 0 || head.heads == 0 || head.kv_heads == 0 || head.head_dim == 0) {
        bad("zero dimension in QK head");
    }
    if (head.heads % head.kv_heads != 0) {
        bad("kv_heads does not divide heads");
    }
    if (head.q_rows_per_head != head.head_dim && head.q_rows_per_head != 2 * head.head_dim) {
        bad("q_rows_per_head must be head_dim or 2 * head_dim");
    }
    if (!(head.rms_eps > 0.0f)) {
        bad("rms_eps must be positive");
    }
    const size_t hidden = head.hidden;
    if (head.attn_norm.size() != hidden || head.q_norm.size() != head.head_dim ||
        head.k_norm.size() != head.head_dim ||
        head.wq.size() != static_cast<size_t>(head.heads) * head.q_rows_per_head * hidden ||
        head.wk.size() != static_cast<size_t>(head.kv_heads) * head.head_dim * hidden) {
        bad("QK head tensor size mismatch");
    }
}

std::vector<float> qk_keys(const QkHead& head, std::span<const float> residual) {
    validate(head);
    const size_t hidden = head.hidden;
    const size_t dim = head.head_dim;
    const size_t kv = head.kv_heads;
    const size_t tokens = token_count(residual, head.hidden);
    std::vector<float> keys(tokens * kv * dim);
    std::vector<float> normed(hidden);
    std::vector<float> scratch;
    for (size_t t = 0; t < tokens; ++t) {
        rms_norm(residual.data() + t * hidden, head.attn_norm.data(), head.rms_eps, hidden, normed.data());
        for (size_t g = 0; g < kv; ++g) {
            project(head.wk.data() + g * dim * hidden, normed.data(), hidden, head.k_norm.data(), head.rms_eps,
                    dim, scratch, keys.data() + (t * kv + g) * dim);
        }
    }
    return keys;
}

std::vector<float> qk_mass(const QkHead& head, std::span<const float> residual, std::span<const float> keys,
                           std::span<const std::uint32_t> query_rows, std::span<const std::uint8_t> key_allowed) {
    validate(head);
    const size_t hidden = head.hidden;
    const size_t dim = head.head_dim;
    const size_t heads = head.heads;
    const size_t kv = head.kv_heads;
    const size_t group = heads / kv;
    const size_t tokens = token_count(residual, head.hidden);
    if (keys.size() != tokens * kv * dim) {
        bad("key cache size mismatch");
    }
    if (key_allowed.size() != tokens) {
        bad("key mask size mismatch");
    }
    std::vector<float> mass(tokens, 0.0f);
    if (query_rows.empty() || tokens == 0) {
        return mass;
    }
    const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
    std::vector<double> sum(tokens, 0.0);
    std::vector<double> logit(tokens);
    std::vector<float> normed(hidden);
    std::vector<float> query(dim);
    std::vector<float> scratch;
    for (const std::uint32_t row : query_rows) {
        if (row >= tokens) {
            bad("query row out of range");
        }
        rms_norm(residual.data() + static_cast<size_t>(row) * hidden, head.attn_norm.data(), head.rms_eps, hidden,
                 normed.data());
        for (size_t h = 0; h < heads; ++h) {
            project(head.wq.data() + h * head.q_rows_per_head * hidden, normed.data(), hidden, head.q_norm.data(),
                    head.rms_eps, dim, scratch, query.data());
            const size_t g = h / group;
            double top = -std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < tokens; ++j) {
                if (key_allowed[j] == 0) {
                    continue;
                }
                logit[j] = dot(query.data(), keys.data() + (j * kv + g) * dim, dim) * scale;
                top = std::max(top, logit[j]);
            }
            if (!(top > -std::numeric_limits<double>::infinity())) {
                continue;  // no allowed key (or only NaN logits): the row contributes nothing
            }
            double total = 0.0;
            for (size_t j = 0; j < tokens; ++j) {
                if (key_allowed[j] != 0) {
                    logit[j] = std::exp(logit[j] - top);
                    total += logit[j];
                }
            }
            for (size_t j = 0; j < tokens; ++j) {
                if (key_allowed[j] != 0) {
                    sum[j] += logit[j] / total;
                }
            }
        }
    }
    const double denominator = static_cast<double>(heads) * static_cast<double>(query_rows.size());
    for (size_t j = 0; j < tokens; ++j) {
        mass[j] = static_cast<float>(sum[j] / denominator);
    }
    return mass;
}

std::vector<std::uint8_t> window_key_mask(std::uint32_t tokens, std::uint32_t query_begin, std::uint32_t query_end,
                                          bool suffix_candidates) {
    if (query_begin > query_end || query_end > tokens) {
        bad("query window out of range");
    }
    std::vector<std::uint8_t> mask(tokens, 0);
    std::fill(mask.begin(), mask.begin() + query_begin, std::uint8_t{1});
    if (suffix_candidates) {
        std::fill(mask.begin() + query_end, mask.end(), std::uint8_t{1});
    }
    return mask;
}

std::vector<std::uint8_t> request_key_mask(std::uint32_t tokens, std::span<const std::uint32_t> query_rows) {
    std::vector<std::uint8_t> mask(tokens, 1);
    for (const std::uint32_t row : query_rows) {
        if (row >= tokens) {
            bad("query row out of range");
        }
        mask[row] = 0;
    }
    return mask;
}

std::vector<float> mix_masses(std::span<const WeightedMass> parts) {
    if (parts.empty()) {
        return {};
    }
    const size_t tokens = parts.front().mass.size();
    std::vector<double> sum(tokens, 0.0);
    double weights = 0.0;
    for (const WeightedMass& part : parts) {
        if (part.mass.size() != tokens) {
            bad("mixed masses differ in length");
        }
        if (!(part.weight > 0.0f)) {
            bad("mix weight must be positive");
        }
        weights += static_cast<double>(part.weight);
        for (size_t j = 0; j < tokens; ++j) {
            sum[j] += static_cast<double>(part.weight) * static_cast<double>(part.mass[j]);
        }
    }
    std::vector<float> mixed(tokens);
    for (size_t j = 0; j < tokens; ++j) {
        mixed[j] = static_cast<float>(sum[j] / weights);
    }
    return mixed;
}

float history_weight(std::uint32_t index) {
    return static_cast<float>(std::ldexp(1.0, -static_cast<int>(std::min<std::uint32_t>(index, 1000u)) - 1));
}

float pool_span(std::span<const float> mass, std::uint32_t begin, std::uint32_t end, bool density) {
    if (begin > end || end > mass.size()) {
        bad("span out of range");
    }
    double sum = 0.0;
    for (size_t j = begin; j < end; ++j) {
        sum += static_cast<double>(mass[j]);
    }
    if (density && end > begin) {
        sum /= static_cast<double>(end - begin);
    }
    return static_cast<float>(sum);
}

std::vector<float> box_smooth(std::span<const float> scores, std::uint32_t kernel) {
    if (kernel == 0 || kernel % 2 == 0) {
        bad("box kernel must be odd");
    }
    const size_t radius = kernel / 2;
    const size_t tokens = scores.size();
    std::vector<float> smooth(tokens);
    for (size_t j = 0; j < tokens; ++j) {
        const size_t first = j >= radius ? j - radius : 0;
        const size_t last = std::min(tokens, j + radius + 1);
        double sum = 0.0;
        for (size_t u = first; u < last; ++u) {
            sum += static_cast<double>(scores[u]);
        }
        smooth[j] = static_cast<float>(sum / static_cast<double>(last - first));
    }
    return smooth;
}

float score_spread(std::span<const float> scores, std::uint32_t selected) {
    double total = 0.0;
    for (const float score : scores) {
        if (score > 0.0f) {
            total += static_cast<double>(score);
        }
    }
    if (!(total > 0.0)) {
        return 0.0f;
    }
    double entropy = 0.0;
    for (const float score : scores) {
        if (score > 0.0f) {
            const double p = static_cast<double>(score) / total;
            entropy -= p * std::log(p);
        }
    }
    return static_cast<float>(std::exp(entropy) / static_cast<double>(std::max<std::uint32_t>(selected, 1u)));
}

void validate(const ProbeMlp& mlp) {
    if (mlp.hidden == 0 || mlp.width == 0) {
        bad("zero dimension in probe");
    }
    if (!(mlp.norm_eps > 0.0f)) {
        bad("probe norm_eps must be positive");
    }
    const size_t hidden = mlp.hidden;
    const size_t width = mlp.width;
    if (mlp.norm_weight.size() != hidden || mlp.norm_bias.size() != hidden || mlp.fc1.size() != width * hidden ||
        mlp.fc1_bias.size() != width || mlp.fc2.size() != width) {
        bad("probe tensor size mismatch");
    }
}

std::vector<float> probe_logits(const ProbeMlp& mlp, std::span<const float> residual, Gelu kind) {
    validate(mlp);
    const size_t hidden = mlp.hidden;
    const size_t width = mlp.width;
    const size_t tokens = token_count(residual, mlp.hidden);
    std::vector<float> raw(tokens);
    std::vector<float> normed(hidden);
    for (size_t t = 0; t < tokens; ++t) {
        const float* x = residual.data() + t * hidden;
        double mean = 0.0;
        for (size_t i = 0; i < hidden; ++i) {
            mean += static_cast<double>(x[i]);
        }
        mean /= static_cast<double>(hidden);
        double variance = 0.0;
        for (size_t i = 0; i < hidden; ++i) {
            const double centred = static_cast<double>(x[i]) - mean;
            variance += centred * centred;
        }
        variance /= static_cast<double>(hidden);
        const double scale = 1.0 / std::sqrt(variance + static_cast<double>(mlp.norm_eps));
        for (size_t i = 0; i < hidden; ++i) {
            normed[i] = static_cast<float>((static_cast<double>(x[i]) - mean) * scale *
                                               static_cast<double>(mlp.norm_weight[i]) +
                                           static_cast<double>(mlp.norm_bias[i]));
        }
        double out = static_cast<double>(mlp.fc2_bias);
        for (size_t w = 0; w < width; ++w) {
            const double pre = dot(mlp.fc1.data() + w * hidden, normed.data(), hidden) +
                               static_cast<double>(mlp.fc1_bias[w]);
            out += static_cast<double>(mlp.fc2[w]) * static_cast<double>(static_cast<float>(gelu(pre, kind)));
        }
        raw[t] = static_cast<float>(out);
    }
    return raw;
}

std::vector<float> segment_cut_prob(const SegmentProbe& probe, std::span<const float> residual) {
    if (probe.conv.empty() || probe.conv.size() % 2 == 0) {
        bad("segment probe conv length must be odd");
    }
    const std::vector<float> raw = probe_logits(probe.mlp, residual, Gelu::tanh);
    const size_t tokens = raw.size();
    const size_t taps = probe.conv.size();
    const size_t radius = taps / 2;
    std::vector<float> cut(tokens);
    for (size_t t = 0; t < tokens; ++t) {
        double acc = static_cast<double>(raw[t]) + static_cast<double>(probe.conv_bias);
        for (size_t k = 0; k < taps; ++k) {
            if (t + k < radius || t + k - radius >= tokens) {
                continue;
            }
            acc += static_cast<double>(probe.conv[k]) * static_cast<double>(raw[t + k - radius]);
        }
        cut[t] = static_cast<float>(1.0 / (1.0 + std::exp(-acc)));
    }
    return cut;
}

std::vector<float> request_logits(const RequestProbe& probe, std::span<const float> residual) {
    return probe_logits(probe.mlp, residual, Gelu::erf);
}

std::vector<std::uint8_t> request_runs(std::span<const float> raw, float tau,
                                       std::span<const std::uint32_t> message_of_token, std::uint32_t min_run) {
    if (message_of_token.size() != raw.size()) {
        bad("message index length mismatch");
    }
    const size_t tokens = raw.size();
    std::vector<std::uint8_t> mask(tokens, 0);
    size_t begin = 0;
    while (begin < tokens) {
        if (!(raw[begin] >= tau)) {  // also false for NaN
            ++begin;
            continue;
        }
        size_t end = begin + 1;
        while (end < tokens && raw[end] >= tau && message_of_token[end] == message_of_token[begin]) {
            ++end;
        }
        if (end - begin >= min_run) {
            std::fill(mask.begin() + static_cast<std::ptrdiff_t>(begin),
                      mask.begin() + static_cast<std::ptrdiff_t>(end), std::uint8_t{1});
        }
        begin = end;
    }
    return mask;
}

std::vector<std::uint32_t> request_rows(std::span<const std::uint8_t> mask, std::uint32_t cap) {
    std::vector<std::uint32_t> rows;
    for (size_t t = 0; t < mask.size(); ++t) {
        if (mask[t] != 0) {
            rows.push_back(static_cast<std::uint32_t>(t));
        }
    }
    if (rows.size() > cap) {
        rows.erase(rows.begin(), rows.end() - static_cast<std::ptrdiff_t>(cap));
    }
    return rows;
}

}  // namespace strix::serve::pflash
