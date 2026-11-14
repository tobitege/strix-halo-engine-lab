#include "h3/te/te_vision_host.hpp"

#include "h3/te/te_tokenizer.hpp"

#include "strix/vision/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strix::h3::te {
namespace {

constexpr int kPatch = 16, kMerge = 2, kTemporal = 2, kFactor = kPatch * kMerge;

}  // namespace

// The preprocessing lives in the shared vision library; these keep the H3 signatures.
void smart_resize(int h, int w, int& h_bar, int& w_bar, double min_pixels, double max_pixels) {
    const vision::Size s = vision::smart_resize(h, w, kFactor, min_pixels, max_pixels);
    h_bar = s.height;
    w_bar = s.width;
}

std::vector<float> resize_bilinear(const float* rgb, int h, int w, int oh, int ow) {
    return vision::resize_bilinear_f32(rgb, h, w, oh, ow);
}

Patches make_patches(const Frames& f) {
    if ((f.t != 1 && f.t != 2) || f.h <= 0 || f.w <= 0 || f.rgb.size() != std::size_t(f.t) * f.h * f.w * 3)
        throw std::invalid_argument("make_patches: frames");
    int hb = 0, wb = 0;
    smart_resize(f.h, f.w, hb, wb, f.min_pixels, f.max_pixels);
    std::vector<std::vector<float>> frames;
    std::vector<const float*> views;
    for (int t = 0; t < f.t; ++t) {
        auto r = vision::resize_bilinear_f32(f.rgb.data() + std::size_t(t) * f.h * f.w * 3, f.h, f.w, hb, wb);
        vision::normalize_inplace(r, {0.5F, 0.5F, 0.5F}, {0.5F, 0.5F, 0.5F});
        frames.push_back(std::move(r));
    }
    for (const auto& fr : frames) views.push_back(fr.data());
    auto shared = vision::patchify(views, hb, wb, vision::PatchSpec{kPatch, kTemporal, kMerge});
    Patches p;
    p.grid_h = shared.grid_h;
    p.grid_w = shared.grid_w;
    p.data = std::move(shared.data);
    return p;
}

Layout expand_layout(const std::vector<std::int32_t>& presented, const std::vector<Patches>& blocks) {
    Layout l;
    std::size_t b = 0;
    for (const auto id : presented) {
        if (id != kVisionSlot) { l.ids.push_back(id); continue; }
        if (b >= blocks.size()) throw std::invalid_argument("expand_layout: more slots than blocks");
        const auto& p = blocks[b++];
        VisionSpan s;
        s.index = std::uint32_t(l.ids.size());
        s.size = std::uint32_t(p.merged_tokens());
        s.grid_h = p.grid_h;
        s.grid_w = p.grid_w;
        l.spans.push_back(s);
        l.ids.insert(l.ids.end(), s.size, kVisionSlot);
    }
    if (b != blocks.size()) throw std::invalid_argument("expand_layout: unused blocks");
    const std::size_t S = l.ids.size();
    l.tags.assign(S, 1);
    for (const auto& s : l.spans) {
        const std::size_t lo = s.index > 0 ? s.index - 1 : 0, hi = std::min<std::size_t>(S, s.index + s.size + 1);
        for (std::size_t i = lo; i < hi; ++i) l.tags[i] = 0;
    }
    if (l.spans.empty()) return l;
    // MRoPE positions, following the reference construction step by step (float32 values)
    l.positions.assign(3 * S, 0.0F);
    auto pos = [&](int axis, std::size_t i) -> float& { return l.positions[std::size_t(axis) * S + i]; };
    long long offset = 0;
    bool first = true;
    for (const auto& s : l.spans) {
        const std::size_t start = s.index, end = s.index + s.size;
        if (first) {
            for (std::size_t i = 0; i < start; ++i) for (int a = 0; a < 3; ++a) pos(a, i) = float(i);
            first = false;
        }
        const long long len_max = std::max({1, s.grid_h, s.grid_w}) / 2;
        const long long start_next = len_max + (long long)start;
        for (std::size_t i = end; i < S; ++i)
            for (int a = 0; a < 3; ++a) pos(a, i) = float(start_next + offset + (long long)(i - end));
        const long long mh = s.grid_h / 2, mw = s.grid_w / 2;
        const long long n_h = (s.size + mh - 1) / mh;  // ceil(size / max_d) repeats of each H value
        for (std::size_t j = 0; j < s.size; ++j) {
            pos(0, start + j) = float((long long)start + offset);
            pos(1, start + j) = float((long long)start + offset + (long long)j / n_h);
            pos(2, start + j) = float((long long)start + offset + (long long)j % mw);
        }
        offset += len_max - (long long)s.size;
    }
    return l;
}

}  // namespace strix::h3::te
