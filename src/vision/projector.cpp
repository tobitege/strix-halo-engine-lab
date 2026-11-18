// Shared projector ops of the vision runtime (host side): the FP64 activation and the host graph of
// DenseOps::merge_project. Built without contraction into FMA (non-MSVC), as the family projectors were: the rows
// are cache content, and the device kernel evaluates the same formula in the same order.
#include "strix/vision/encoder.hpp"

#include "encoder_layout.hpp"
#include "strix/env_flags.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strix::vision {

bool Encoder::project_device() {
    static const bool on = envcfg::flag("STRIX_VISION_PROJECT_DEVICE", true);
    return on;
}

const char* Encoder::effective_arithmetic(unsigned attention_terms) {
    const bool two = attention_terms == 2U;
    if (project_device()) return two ? "vision-f16-attn2-v1-project-device-f64-v1" : "vision-f16-v1-project-device-f64-v1";
    return two ? kArithmeticAttn2 : kArithmetic;
}

layout::ProjectorGeometry layout::merge_geometry(unsigned grid_h, unsigned grid_w, unsigned ratio) {
    constexpr auto limit = std::numeric_limits<unsigned>::max();
    const std::uint64_t h = grid_h, w = grid_w, r = ratio;
    if (!h || !w || !r || h * w > limit || r * r > limit)
        throw std::invalid_argument("vision encoder: merge projector grid");
    const auto mh = (h + r - 1U) / r, mw = (w + r - 1U) / r;
    // The unchanged unshuffle kernel uses unsigned coordinates, including zero-filled edge cells.
    if (mh * mw > limit || mh * r > limit || mw * r > limit)
        throw std::invalid_argument("vision encoder: merge projector grid range");
    return {static_cast<unsigned>(h * w), static_cast<unsigned>(mh), static_cast<unsigned>(mw),
            static_cast<unsigned>(mh * mw), 0};
}

layout::ProjectorGeometry layout::projector_geometry(const EncoderOutput& hidden, unsigned grid_h, unsigned grid_w,
                                                     const MergeProjector& m, const float* rows, bool resident) {
    auto g = merge_geometry(grid_h, grid_w, m.ratio);
    const std::uint64_t in = std::uint64_t{hidden.width} * m.ratio * m.ratio;
    // Leave room for GEMM panel padding in unsigned dimensions on both backends.
    constexpr auto limit = std::numeric_limits<unsigned>::max() - 127U;
    if (!hidden.width || !rows || !m.w1 || !m.w2 || hidden.rows != g.rows || in > limit || m.w1.cols != in ||
        m.w1.rows > limit || m.w2.rows > limit || m.w2.cols != m.w1.rows)
        throw std::invalid_argument("vision encoder: merge projector operands");
    const auto elements = [](std::size_t n, std::size_t width) {
        if (!width || n > std::numeric_limits<std::size_t>::max() / sizeof(float) / width)
            throw std::invalid_argument("vision encoder: merge projector storage range");
        return n * width;
    };
    const auto hidden_size = elements(g.rows, hidden.width);
    if (!resident && hidden.hidden.size() != hidden_size)
        throw std::invalid_argument("vision encoder: merge projector hidden rows");
    g.in = static_cast<unsigned>(in);
    (void)elements(g.tokens, round_up(g.in, kGemmIn));
    (void)elements(g.tokens, round_up(static_cast<unsigned>(m.w1.rows), kGemmIn));
    (void)elements(g.tokens, round_up(static_cast<unsigned>(m.w2.rows), kGemmOut));
    return g;
}

float activate(Activation a, float v) {
    const double x = static_cast<double>(v);
    switch (a) {
        case Activation::gelu: return static_cast<float>(0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440)));
        case Activation::gelu_tanh: return static_cast<float>(0.5 * x * (1.0 + std::tanh(0.79788456080286535588 * (x + 0.044715 * x * x * x))));
        case Activation::silu: return static_cast<float>(x / (1.0 + std::exp(-x)));
    }
    return v;
}

unsigned MergeProjector::tokens(unsigned grid_h, unsigned grid_w) const {
    return layout::merge_geometry(grid_h, grid_w, ratio).tokens;
}

void DenseOps::merge_project(const EncoderOutput& hidden, unsigned grid_h, unsigned grid_w, const MergeProjector& m, float* rows) {
    const unsigned R = m.ratio, W = hidden.width;
    const auto g = layout::projector_geometry(hidden, grid_h, grid_w, m, rows);
    const unsigned mh = g.mh, mw = g.mw, tokens = g.tokens;
    const std::size_t in = g.in, P = m.w1.rows;
    // Unshuffle; the vector starts as zeros, so cells past the grid edge stay zero.
    std::vector<float> merged(std::size_t{tokens} * in);
    for (unsigned my = 0; my < mh; ++my)
        for (unsigned mx = 0; mx < mw; ++mx) {
            float* dst = &merged[(std::size_t{my} * mw + mx) * in];
            for (unsigned dy = 0; dy < R; ++dy)
                for (unsigned dx = 0; dx < R; ++dx) {
                    if (my * R + dy >= grid_h || mx * R + dx >= grid_w) continue;
                    const float* src = &hidden.hidden[(std::size_t{my * R + dy} * grid_w + mx * R + dx) * W];
                    const unsigned slot = m.row_major ? dy * R + dx : dx * R + dy;
                    for (unsigned c = 0; c < W; ++c) dst[std::size_t{c} * R * R + slot] = src[c];
                }
        }
    std::vector<float> mid(std::size_t{tokens} * P);
    linear(m.w1, m.b1, merged.data(), tokens, mid.data());
    merged = {};
    layout::parallel_ranges(mid.size(), 16U, [&](std::size_t i0, std::size_t i1) {  // elementwise: the split cannot change a value
        for (std::size_t i = i0; i < i1; ++i) mid[i] = activate(m.act, mid[i]);
    });
    linear(m.w2, m.b2, mid.data(), tokens, rows);
}

}  // namespace strix::vision
