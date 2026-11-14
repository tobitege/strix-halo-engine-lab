#pragma once
// Synthetic vision towers for the shared-encoder tests: deterministic weights, no checkpoint, no image decoder.
#include "strix/vision/adapter.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace vision_toy {

using namespace strix::vision;

struct Rng {
    std::uint64_t s;
    float next() {  // uniform in [-1, 1)
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>(static_cast<double>(s >> 40U) / 8388608.0 - 1.0);
    }
};

struct Tower {
    EncoderConfig cfg;
    EncoderWeights w;
    std::vector<float> projector;  // [out][width]
    unsigned projector_out = 0;

    Tensor fill(std::size_t rows, std::size_t cols, float scale, Rng& rng, float offset = 0.0F) {
        auto& v = w.owned.emplace_back(rows * cols);
        for (auto& e : v) e = offset + scale * rng.next();
        return Tensor{v.data(), Tensor::Type::f32, rows, cols};
    }
};

inline Tower make_tower(EncoderConfig cfg, std::uint64_t seed, unsigned projector_out = 0) {
    Tower t;
    t.cfg = std::move(cfg);
    const auto& c = t.cfg;
    Rng rng{seed};
    const std::size_t W = c.width;
    const bool ln = c.norm == EncoderConfig::Norm::layernorm;
    const float ws = 1.0F / static_cast<float>(c.heads * 4U);
    t.w.patch_w = t.fill(W, c.patch_dim(), 0.08F, rng);
    if (c.patch_bias) t.w.patch_b = t.fill(W, 1, 0.1F, rng);
    if (c.has_learned()) t.w.pos_embed = t.fill(std::size_t{c.pos_grid} * c.pos_grid, W, 0.3F, rng);
    for (unsigned l = 0; l < c.layers; ++l) {
        BlockWeights b;
        b.norm1_w = t.fill(W, 1, 0.2F, rng, 1.0F);
        if (ln) b.norm1_b = t.fill(W, 1, 0.1F, rng);
        b.qkv_w = t.fill(3U * W, W, ws, rng);
        if (c.qkv_bias) b.qkv_b = t.fill(3U * W, 1, 0.1F, rng);
        b.out_w = t.fill(W, W, ws, rng);
        if (c.out_bias) b.out_b = t.fill(W, 1, 0.1F, rng);
        b.norm2_w = t.fill(W, 1, 0.2F, rng, 1.0F);
        if (ln) b.norm2_b = t.fill(W, 1, 0.1F, rng);
        b.mlp1_w = t.fill(c.mlp_out(), W, ws, rng);
        if (c.mlp_bias) b.mlp1_b = t.fill(c.mlp_out(), 1, 0.1F, rng);
        b.mlp2_w = t.fill(W, c.mlp_width, ws, rng);
        if (c.mlp_bias) b.mlp2_b = t.fill(W, 1, 0.1F, rng);
        t.w.blocks.push_back(b);
    }
    if (c.final_norm) {
        t.w.final_norm_w = t.fill(W, 1, 0.2F, rng, 1.0F);
        if (ln) t.w.final_norm_b = t.fill(W, 1, 0.1F, rng);
    }
    if (projector_out) {
        t.projector_out = projector_out;
        t.projector.resize(std::size_t{projector_out} * W);
        for (auto& e : t.projector) e = 0.1F * rng.next();
    }
    return t;
}

// gh x gw patches in raster order; `tiles` equal attention tiles (rows must divide).
inline PatchGrid make_grid(const EncoderConfig& c, unsigned gh, unsigned gw, std::uint64_t seed, unsigned tiles = 1) {
    PatchGrid g;
    g.rows = gh * gw;
    g.patch_dim = c.patch_dim();
    g.grid_h = gh;
    g.grid_w = gw;
    g.data.resize(std::size_t{g.rows} * g.patch_dim);
    Rng rng{seed};
    for (auto& e : g.data) e = rng.next();
    for (unsigned y = 0; y < gh; ++y)
        for (unsigned x = 0; x < gw; ++x) {
            g.y.push_back(static_cast<std::uint16_t>(y));
            g.x.push_back(static_cast<std::uint16_t>(x));
        }
    if (tiles > 1U) {
        if (g.rows % tiles) throw std::invalid_argument("toy grid tiles");
        for (unsigned t = 1; t <= tiles; ++t) g.tile_end.push_back(g.rows / tiles * t);
    }
    return g;
}

// Adapter over a toy tower: the projector is one linear to projector_out, rows stored as BF16.
class ToyAdapter final : public VisionAdapter {
public:
    explicit ToyAdapter(Tower& t, std::string id = "toy-v1") : tower_(t), id_(std::move(id)) {}
    EncoderConfig config() const override { return tower_.cfg; }
    std::string identity() const override { return id_; }
    ResizeRule resize() const override { return {tower_.cfg.patch, 0, 0, 1024, 0.0F}; }
    PatchGrid patches(const Image&) const override { throw std::logic_error("toy adapter has no image path"); }
    void bind(WeightSource&, EncoderWeights&) override {}
    void project(EncoderOutput& h, const PatchGrid&, DenseOps& dense, ProjectedRows& out) override {
        ++projections;
        const Tensor w{tower_.projector.data(), Tensor::Type::f32, tower_.projector_out, tower_.cfg.width};
        std::vector<float> y(std::size_t{h.rows} * tower_.projector_out);
        dense.linear(w, Tensor{}, h.hidden.data(), h.rows, y.data());
        out.rows = h.rows;
        out.width = tower_.projector_out;
        out.bf16 = to_bf16_rows(y.data(), y.size());
        for (const auto& tap : h.taps) {
            dense.linear(w, Tensor{}, tap.data(), h.rows, y.data());
            out.deep.push_back(to_bf16_rows(y.data(), y.size()));
        }
    }
    SpanLayout layout(const PatchGrid& g) const override {
        SpanLayout s;
        s.role.assign(g.rows, SpanLayout::Role::image);
        for (std::uint32_t r = 0; r < g.rows; ++r) s.lm_pos.push_back({r, 0U, 0U});
        s.real_token_id = 7;
        s.grid_h = g.grid_h;
        s.grid_w = g.grid_w;
        return s;
    }
    unsigned projections = 0;

private:
    Tower& tower_;
    std::string id_;
};

}  // namespace vision_toy
