#include "vision_adapter.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix::deepseek41 {
namespace {

[[noreturn]] void bad(const std::string& what) { throw std::invalid_argument("deepseek41 vision: " + what); }

std::int64_t checked_size(double value) {
    if (!std::isfinite(value) || value < 0.0 || value >= 0x1p63) bad("resize size is out of range");
    return static_cast<std::int64_t>(value);
}

const char* filter_name(vision::Filter f) {
    switch (f) {
        case vision::Filter::bilinear: return "bilinear";
        case vision::Filter::bicubic: return "bicubic";
        case vision::Filter::lanczos: return "lanczos";
    }
    return "?";
}

float activate(VisionRecipe::Activation a, float v) {
    const double x = static_cast<double>(v);
    switch (a) {
        case VisionRecipe::Activation::gelu: return static_cast<float>(0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440)));
        case VisionRecipe::Activation::gelu_tanh:
            return static_cast<float>(0.5 * x * (1.0 + std::tanh(0.79788456080286535588 * (x + 0.044715 * x * x * x))));
        case VisionRecipe::Activation::silu: return static_cast<float>(x / (1.0 + std::exp(-x)));
    }
    return v;
}

void copy_row(const vision::Tensor& row, std::uint16_t* dst, std::size_t width) {
    for (std::size_t i = 0; i < width; ++i) dst[i] = vision::to_bf16(row.at(i));
}

}  // namespace

std::string VisionRecipe::id() const {
    // Exact float bits: two recipes that differ in the last place normalise pixels differently, so they are two keys.
    const auto bits = [](float v) { return std::bit_cast<std::uint32_t>(v); };
    // "r2": the resize rule (patch-multiple canvas, contain + grey margin, partial 3x3 cells completed with zeros).
    char text[224];
    std::snprintf(text, sizeof text, "m%08x,%08x,%08x-s%08x,%08x,%08x-%s%s-g%d-p%d-a%d-u%d-n%d-r2-f%d-b%d", bits(mean[0]), bits(mean[1]),
                  bits(mean[2]), bits(stddev[0]), bits(stddev[1]), bits(stddev[2]), filter_name(resampler.filter),
                  resampler.antialias ? "-aa" : "", swiglu_gate_first ? 1 : 0,
                  rope_pairing == vision::EncoderConfig::RopePairing::half ? 0 : 1, static_cast<int>(aligner_activation),
                  unshuffle_row_major ? 1 : 0, newline_after_last_row ? 1 : 0, static_cast<int>(pad_fill),
                  pixels_bf16 ? 1 : 0);
    return text;
}

VisionAdapter::VisionAdapter(VisionConfig config, std::string weights_sha256_hex, VisionRecipe recipe)
    : config_(std::move(config)), sha_(std::move(weights_sha256_hex)), recipe_(recipe) {
    if (sha_.size() != 64) bad("the weights digest must be 64 hex characters");
    if (config_.num_hidden_layers == 0 || config_.num_attention_heads == 0 || config_.cell() == 0 ||
        config_.max_image_tokens == 0 || config_.projection_size == 0)
        bad("incomplete configuration");
    // The file states no rule for a bounded aspect ratio; refuse one instead of guessing it.
    if (config_.max_wh_ratio != 0) bad("a bounded width/height ratio is not supported");
    // The resize planner reserves a final newline even for recipes that omit it.
    if (config_.max_image_tokens < 4) bad("image token budget must fit at least four span rows");
    for (const float m : recipe_.mean)
        if (!std::isfinite(m)) bad("recipe mean");
    for (const float s : recipe_.stddev)
        if (!(s > 0.0F) || !std::isfinite(s)) bad("recipe std");
}

vision::EncoderConfig VisionAdapter::config() const {
    vision::EncoderConfig c;
    c.layers = config_.num_hidden_layers;
    c.width = config_.hidden_size;
    c.heads = config_.num_attention_heads;
    c.mlp_width = config_.intermediate_size;
    c.norm = vision::EncoderConfig::Norm::rmsnorm;
    c.eps = static_cast<float>(config_.rms_norm_eps);
    c.mlp = vision::EncoderConfig::Mlp::swiglu;
    c.swiglu_limit = 0.0F;
    c.swiglu_gate_first = recipe_.swiglu_gate_first;
    c.qkv_bias = true;
    c.out_bias = true;
    c.mlp_bias = false;
    c.patch_bias = true;
    c.pos = vision::EncoderConfig::Pos::rope2d;
    c.rope_theta = static_cast<float>(config_.rope_theta);
    c.rope_pairing = recipe_.rope_pairing;
    c.patch = config_.patch_size;
    c.temporal = 1;
    c.channels = 3;
    c.final_norm = true;
    return c;
}

std::string VisionAdapter::identity() const {
    return "deepseek41-vision/" + config_.checkpoint_variant + "/" + sha_ + "/" + recipe_.id();
}

unsigned VisionAdapter::span_rows(unsigned grid_h, unsigned grid_w) const noexcept {
    const unsigned newlines = grid_h == 0 ? 0U : recipe_.newline_after_last_row ? grid_h : grid_h - 1U;
    return grid_h * grid_w + newlines + 2U;
}

// The limit of the checkpoint counts the whole span: image rows, row ends, start and end.
unsigned VisionAdapter::max_span_rows() const noexcept { return config_.max_image_tokens; }

vision::MergedGrid VisionAdapter::checked_grid(const vision::PatchGrid& grid) const {
    const auto merged = vision::checked_merged_grid(grid, config_.downsample_ratio, max_span_rows(), true);
    const auto newlines = merged.grid_h - (recipe_.newline_after_last_row ? 0U : 1U);
    if (std::uint64_t{merged.tokens} + newlines + 2U > max_span_rows()) bad("the image span exceeds the row limit");
    return merged;
}

std::uint64_t VisionAdapter::projector_weight_elements() const noexcept {
    const std::uint64_t p = config_.projection_size;
    return p * config_.aligner_inputs() + p * p;
}

vision::ResizeRule VisionAdapter::resize() const {
    vision::ResizeRule r;
    r.multiple = config_.patch_size;
    r.min_pixels = config_.min_pixels;
    r.max_pixels = std::uint64_t{config_.max_image_tokens} * config_.cell() * config_.cell();
    r.max_rows = max_span_rows();
    r.max_aspect = 0.0F;
    return r;
}

vision::Size VisionAdapter::target(int width, int height) const {
    if (width <= 0 || height <= 0) bad("empty image");
    // The reference processor computes this plan in Python: integers where it has integers, IEEE doubles (true
    // division, pow(x, 0.5), sqrt, floor) where it has floats. The same operations in the same order here.
    const std::int64_t patch = config_.patch_size, cell = config_.cell(), limit = config_.max_image_tokens;
    const std::int64_t down = config_.downsample_ratio;
    std::int64_t w = width, h = height;
    if (static_cast<std::uint64_t>(w * h) < config_.min_pixels) {
        const double ratio = std::pow(static_cast<double>(config_.min_pixels) / static_cast<double>(w * h), 0.5);
        w = checked_size(static_cast<double>(w) * ratio);  // int(): truncation
        h = checked_size(static_cast<double>(h) * ratio);
    }
    const auto span = [&](std::int64_t canvas_h, std::int64_t canvas_w) {
        const std::int64_t mh = (canvas_h / patch + down - 1) / down, mw = (canvas_w / patch + down - 1) / down;
        return mh * (mw + 1) + 2;
    };
    std::int64_t best_w = (w + patch - 1) / patch * patch, best_h = (h + patch - 1) / patch * patch;
    if (span(best_h, best_w) > limit) {
        const double r = static_cast<double>(h) / static_cast<double>(w);
        const double max_w = std::sqrt(static_cast<double>(limit - 2) / r + 0.25) - 0.5;
        const double max_h = max_w * r;
        if (!std::isfinite(max_w) || !std::isfinite(max_h)) bad("resize size is out of range");
        if (max_w < 1.0) {  // very tall: one column
            best_h = (limit - 2) / 2 * cell;
            best_w = cell;
        } else if (max_h < 1.0) {  // very wide: one row
            best_h = cell;
            best_w = (limit - 3) * cell;
        } else {
            const double beta = std::min(std::floor(max_w) * static_cast<double>(cell) / static_cast<double>(w),
                                         std::floor(max_h) * static_cast<double>(cell) / static_cast<double>(h));
            const double scaled_h = static_cast<double>(h) * beta, scaled_w = static_cast<double>(w) * beta;
            best_h = checked_size(std::floor(scaled_h / static_cast<double>(patch))) * patch;
            best_w = checked_size(std::floor(scaled_w / static_cast<double>(patch))) * patch;
        }
    }
    if (best_w <= 0 || best_h <= 0 || best_w > std::numeric_limits<int>::max() ||
        best_h > std::numeric_limits<int>::max() || span(best_h, best_w) > limit)
        bad("the image does not fit " + std::to_string(config_.max_image_tokens) + " span rows");
    return {static_cast<int>(best_w), static_cast<int>(best_h)};
}

vision::PatchGrid VisionAdapter::patches(const vision::Image& image) const {
    if (image.width <= 0 || image.height <= 0 || image.rgb.size() != image.pixels() * 3U) bad("empty image");
    const vision::Size t = target(image.width, image.height);
    // Before any resample or allocation: the grid must fit the 16-bit patch coordinates of the shared representation.
    vision::check_grid_axes(static_cast<std::uint64_t>(t.height) / config_.patch_size,
                            static_cast<std::uint64_t>(t.width) / config_.patch_size);
    // The image keeps its aspect ratio: fitted inside the canvas, centred, the margin filled with grey.
    const vision::Image canvas = vision::resize_contain_pad(
        image, t.width, t.height, {recipe_.pad_fill, recipe_.pad_fill, recipe_.pad_fill}, recipe_.resampler);
    const std::vector<float> pixels =
        vision::normalize_div255(canvas, recipe_.mean, recipe_.stddev, recipe_.pixels_bf16);
    const float* frame = pixels.data();
    vision::PatchSpec spec;
    spec.patch = static_cast<int>(config_.patch_size);
    spec.temporal = 1;
    spec.merge = 1;  // raster order: the tower sees plain rows, the 3x3 grouping happens in project()
    vision::Patches p = vision::patchify(std::span<const float* const>(&frame, 1), t.height, t.width, spec);

    vision::PatchGrid g;
    g.grid_h = static_cast<unsigned>(p.grid_h);
    g.grid_w = static_cast<unsigned>(p.grid_w);
    g.rows = g.grid_h * g.grid_w;
    g.patch_dim = static_cast<unsigned>(p.feature);
    g.data = std::move(p.data);
    g.y.resize(g.rows);
    g.x.resize(g.rows);
    for (unsigned r = 0; r < g.rows; ++r) {
        g.y[r] = static_cast<std::uint16_t>(r / g.grid_w);
        g.x[r] = static_cast<std::uint16_t>(r % g.grid_w);
    }
    (void)checked_grid(g);
    return g;
}

void VisionAdapter::bind(vision::WeightSource& s, vision::EncoderWeights& w) {
    w = {};
    w.patch_w = s.tensor("vision.patch_embed.proj.weight");
    w.patch_b = s.tensor("vision.patch_embed.proj.bias");
    w.final_norm_w = s.tensor("vision.norm.weight");
    w.blocks.resize(config_.num_hidden_layers);
    for (std::uint32_t l = 0; l < config_.num_hidden_layers; ++l) {
        const std::string p = "vision.blocks." + std::to_string(l) + ".";
        vision::BlockWeights& b = w.blocks[l];
        b.norm1_w = s.tensor(p + "norm1.weight");
        b.qkv_w = s.tensor(p + "attn.wqkv.weight");
        b.qkv_b = s.tensor(p + "attn.wqkv.bias");
        b.out_w = s.tensor(p + "attn.wo.weight");
        b.out_b = s.tensor(p + "attn.wo.bias");
        b.norm2_w = s.tensor(p + "norm2.weight");
        b.mlp1_w = s.tensor(p + "mlp.w1.weight");
        b.mlp2_w = s.tensor(p + "mlp.w2.weight");
    }
    aligner_w1_ = s.tensor("aligner.w1.weight");
    aligner_b1_ = s.tensor("aligner.w1.bias");
    aligner_w2_ = s.tensor("aligner.w2.weight");
    aligner_b2_ = s.tensor("aligner.w2.bias");
    image_start_ = s.tensor("image_start");
    image_newline_ = s.tensor("image_newline");
    image_end_ = s.tensor("image_end");
    const std::size_t P = config_.projection_size;
    if (aligner_w1_.rows != P || aligner_w1_.cols != config_.aligner_inputs() || aligner_b1_.size() != P ||
        aligner_w2_.rows != P || aligner_w2_.cols != P || aligner_b2_.size() != P || image_start_.size() != P ||
        image_newline_.size() != P || image_end_.size() != P)
        bad("projector tensor shapes differ from the configuration");
}

void VisionAdapter::project(vision::EncoderOutput& hidden, const vision::PatchGrid& grid, vision::DenseOps& dense,
                            vision::ProjectedRows& out) {
    if (!aligner_w1_) bad("project() before bind()");
    const unsigned R = config_.downsample_ratio, W = config_.hidden_size, P = config_.projection_size;
    const vision::MergedGrid merged_grid = checked_grid(grid);
    if (hidden.width != W || hidden.rows != grid.rows || hidden.hidden.size() != std::size_t{hidden.rows} * W)
        bad("the encoder output does not match the patch grid");
    const unsigned mh = merged_grid.grid_h, mw = merged_grid.grid_w, tokens = merged_grid.tokens;
    const std::size_t in = std::size_t{W} * R * R;

    // Pixel unshuffle on the raster grid: merged row (my, mx), feature c * R * R + (dy, dx). A grid that is not a
    // multiple of R is completed with zero rows on the right and at the bottom (the vector starts as zeros).
    std::vector<float> merged(std::size_t{tokens} * in);
    for (unsigned my = 0; my < mh; ++my)
        for (unsigned mx = 0; mx < mw; ++mx) {
            float* dst = &merged[(std::size_t{my} * mw + mx) * in];
            for (unsigned dy = 0; dy < R; ++dy)
                for (unsigned dx = 0; dx < R; ++dx) {
                    if (my * R + dy >= grid.grid_h || mx * R + dx >= grid.grid_w) continue;
                    const float* src = &hidden.hidden[(std::size_t{my * R + dy} * grid.grid_w + mx * R + dx) * W];
                    const unsigned slot = recipe_.unshuffle_row_major ? dy * R + dx : dx * R + dy;
                    for (unsigned c = 0; c < W; ++c) dst[std::size_t{c} * R * R + slot] = src[c];
                }
        }
    std::vector<float> mid(std::size_t{tokens} * P), rows(std::size_t{tokens} * P);
    dense.linear(aligner_w1_, aligner_b1_, merged.data(), tokens, mid.data());
    merged = {};
    vision::parallel_for(tokens, 0, [&](std::size_t row) {  // elementwise: the thread count cannot change a value
        float* v = &mid[row * P];
        for (unsigned i = 0; i < P; ++i) v[i] = activate(recipe_.aligner_activation, v[i]);
    });
    dense.linear(aligner_w2_, aligner_b2_, mid.data(), tokens, rows.data());

    // Span order: start, each grid row followed by the row end, end.
    out = {};
    out.width = P;
    out.rows = span_rows(mh, mw);
    out.bf16.resize(std::size_t{out.rows} * P);
    std::uint16_t* dst = out.bf16.data();
    copy_row(image_start_, dst, P);
    dst += P;
    for (unsigned my = 0; my < mh; ++my) {
        for (unsigned mx = 0; mx < mw; ++mx, dst += P) {
            const float* src = &rows[(std::size_t{my} * mw + mx) * P];
            for (unsigned i = 0; i < P; ++i) dst[i] = vision::to_bf16(src[i]);
        }
        if (my + 1U < mh || recipe_.newline_after_last_row) {
            copy_row(image_newline_, dst, P);
            dst += P;
        }
    }
    copy_row(image_end_, dst, P);
}

vision::EncoderOptions VisionAdapter::device_options() const {
    vision::EncoderOptions options;
    options.split_activations = true;
    return options;
}

vision::SpanLayout VisionAdapter::layout(const vision::PatchGrid& grid) const {
    const vision::MergedGrid merged = checked_grid(grid);
    using Role = vision::SpanLayout::Role;
    vision::SpanLayout s;
    s.grid_h = merged.grid_h;
    s.grid_w = merged.grid_w;
    s.real_token_id = config_.image_token_id;  // every row of the span is the image token to id-keyed logic
    const unsigned rows = span_rows(s.grid_h, s.grid_w), stride = s.grid_w + 1U;  // a grid row and its row end
    s.role.assign(rows, Role::image);
    s.role[0] = Role::start;
    for (unsigned y = 0; y < s.grid_h; ++y)
        if (y + 1U < s.grid_h || recipe_.newline_after_last_row) s.role[1U + y * stride + s.grid_w] = Role::newline;
    s.role[rows - 1U] = Role::end;
    // The language model keeps 1D positions: one per row, in span order.
    s.lm_pos.resize(s.role.size());
    for (std::uint32_t i = 0; i < s.lm_pos.size(); ++i) s.lm_pos[i] = {i, 0U, 0U};
    return s;
}

}  // namespace strix::deepseek41
