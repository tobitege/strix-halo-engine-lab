#include "strix/vision/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace strix::vision {
namespace {

[[noreturn]] void bad(const char* what) { throw ImageError(ImageErrorCode::invalid_argument, std::string("vision: ") + what); }

void check(const Image& im) {
    if (im.width <= 0 || im.height <= 0 || im.rgb.size() != im.pixels() * 3) bad("image size and buffer disagree");
}

// ---- fixed-point separable resampler (the published algorithm of the PIL resampler) ---------------------------------

constexpr int kPrecisionBits = 32 - 8 - 2;
constexpr double kPi = 3.14159265358979323846;

double filter_support(Filter f) { return f == Filter::bilinear ? 1.0 : f == Filter::bicubic ? 2.0 : 3.0; }

double sinc(double x) {
    if (x == 0.0) return 1.0;
    x *= kPi;
    return std::sin(x) / x;
}

double filter_weight(Filter f, double x) {
    switch (f) {
        case Filter::bilinear:
            if (x < 0.0) x = -x;
            return x < 1.0 ? 1.0 - x : 0.0;
        case Filter::bicubic: {
            constexpr double a = -0.5;
            if (x < 0.0) x = -x;
            if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
            if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
            return 0.0;
        }
        case Filter::lanczos: return -3.0 <= x && x < 3.0 ? sinc(x) * sinc(x / 3) : 0.0;
    }
    return 0.0;
}

struct Coeffs {
    int ksize = 0;
    std::vector<int> first, count;     // per output sample
    std::vector<std::int32_t> weight;  // [out][ksize], fixed point
};

Coeffs coefficients(int in, int out, Resampler r) {
    const double scale = double(in) / double(out);
    const double filterscale = r.antialias ? std::max(scale, 1.0) : 1.0;
    const double support = filter_support(r.filter) * filterscale;
    Coeffs c;
    c.ksize = int(std::ceil(support)) * 2 + 1;
    c.first.resize(std::size_t(out));
    c.count.resize(std::size_t(out));
    c.weight.assign(std::size_t(out) * std::size_t(c.ksize), 0);
    std::vector<double> k(std::size_t(c.ksize));
    const double ss = 1.0 / filterscale;
    for (int xx = 0; xx < out; ++xx) {
        const double center = (xx + 0.5) * scale;
        int xmin = int(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = int(center + support + 0.5);
        if (xmax > in) xmax = in;
        xmax -= xmin;
        double ww = 0.0;
        for (int x = 0; x < xmax; ++x) {
            const double w = filter_weight(r.filter, (x + xmin - center + 0.5) * ss);
            k[std::size_t(x)] = w;
            ww += w;
        }
        std::int32_t* dst = c.weight.data() + std::size_t(xx) * std::size_t(c.ksize);
        for (int x = 0; x < xmax; ++x) {
            double w = k[std::size_t(x)];
            if (ww != 0.0) w /= ww;
            dst[x] = w < 0 ? std::int32_t(-0.5 + w * (1 << kPrecisionBits)) : std::int32_t(0.5 + w * (1 << kPrecisionBits));
        }
        c.first[std::size_t(xx)] = xmin;
        c.count[std::size_t(xx)] = xmax;
    }
    return c;
}

std::uint8_t clip8(std::int32_t v) {
    v >>= kPrecisionBits;  // arithmetic shift: floor, as the reference table lookup
    return std::uint8_t(v < 0 ? 0 : v > 255 ? 255 : v);
}

}  // namespace

Size smart_resize(int height, int width, int factor, double min_pixels, double max_pixels) {
    if (height <= 0 || width <= 0 || factor <= 0 || min_pixels <= 0 || max_pixels < min_pixels || !std::isfinite(max_pixels))
        throw std::invalid_argument("vision: invalid image dimensions or processor bounds");
    // Python round(): half to even on the double quotient
    int h_bar = int(std::nearbyint(double(height) / factor)) * factor;
    int w_bar = int(std::nearbyint(double(width) / factor)) * factor;
    if (double(h_bar) * w_bar > max_pixels) {
        const double beta = std::sqrt(double(height) * width / max_pixels);
        h_bar = std::max(factor, int(std::floor(height / beta / factor)) * factor);
        w_bar = std::max(factor, int(std::floor(width / beta / factor)) * factor);
    } else if (double(h_bar) * w_bar < min_pixels) {
        const double beta = std::sqrt(min_pixels / (double(height) * width));
        h_bar = int(std::ceil(height * beta / factor)) * factor;
        w_bar = int(std::ceil(width * beta / factor)) * factor;
    }
    return Size{w_bar, h_bar};
}

Size select_best_resolution(Size original, std::span<const Size> pinpoints) {
    if (original.width <= 0 || original.height <= 0 || pinpoints.empty()) bad("any-res needs a size and pinpoints");
    Size best{};
    std::int64_t best_effective = 0, best_wasted = std::numeric_limits<std::int64_t>::max();
    for (const Size p : pinpoints) {
        if (p.width <= 0 || p.height <= 0) bad("any-res pinpoint");
        const double scale = std::min(double(p.width) / original.width, double(p.height) / original.height);
        const std::int64_t dw = std::int64_t(original.width * scale), dh = std::int64_t(original.height * scale);
        const std::int64_t effective = std::min(dw * dh, std::int64_t(original.width) * original.height);
        const std::int64_t wasted = std::int64_t(p.width) * p.height - effective;
        if (effective > best_effective || (effective == best_effective && wasted < best_wasted)) {
            best_effective = effective;
            best_wasted = wasted;
            best = p;
        }
    }
    return best;
}

Size ResizePolicy::target(int width, int height) const {
    if (width <= 0 || height <= 0) bad("image size");
    switch (kind) {
        case Kind::smart: return smart_resize(height, width, factor, min_pixels, max_pixels);
        case Kind::fixed:
            if (fixed.width <= 0 || fixed.height <= 0) bad("fixed target size");
            return fixed;
        case Kind::any_res: return select_best_resolution(Size{width, height}, pinpoints);
    }
    bad("resize policy");
}

Image resize(const Image& im, int ow, int oh, Resampler r) {
    check(im);
    if (ow <= 0 || oh <= 0) bad("resize target");
    const int iw = im.width, ih = im.height;
    // Horizontal pass (skipped when the width is unchanged, as the reference does).
    Image mid;
    const Image* src = &im;
    if (ow != iw) {
        const Coeffs c = coefficients(iw, ow, r);
        mid.width = ow;
        mid.height = ih;
        mid.rgb.resize(std::size_t(ow) * std::size_t(ih) * 3);
        for (int y = 0; y < ih; ++y) {
            const std::uint8_t* row = im.rgb.data() + std::size_t(y) * std::size_t(iw) * 3;
            std::uint8_t* out = mid.rgb.data() + std::size_t(y) * std::size_t(ow) * 3;
            for (int xx = 0; xx < ow; ++xx, out += 3) {
                const std::int32_t* k = c.weight.data() + std::size_t(xx) * std::size_t(c.ksize);
                const std::uint8_t* p = row + std::size_t(c.first[std::size_t(xx)]) * 3;
                std::int32_t s0 = 1 << (kPrecisionBits - 1), s1 = s0, s2 = s0;
                for (int x = 0, n = c.count[std::size_t(xx)]; x < n; ++x, p += 3) {
                    s0 += p[0] * k[x];
                    s1 += p[1] * k[x];
                    s2 += p[2] * k[x];
                }
                out[0] = clip8(s0);
                out[1] = clip8(s1);
                out[2] = clip8(s2);
            }
        }
        src = &mid;
    }
    if (oh == ih) return *src;
    const Coeffs c = coefficients(ih, oh, r);
    Image dst;
    dst.width = ow;
    dst.height = oh;
    dst.rgb.resize(std::size_t(ow) * std::size_t(oh) * 3);
    const std::size_t stride = std::size_t(ow) * 3;
    std::vector<std::int32_t> acc(stride);
    for (int yy = 0; yy < oh; ++yy) {
        const std::int32_t* k = c.weight.data() + std::size_t(yy) * std::size_t(c.ksize);
        std::fill(acc.begin(), acc.end(), 1 << (kPrecisionBits - 1));
        const std::uint8_t* p = src->rgb.data() + std::size_t(c.first[std::size_t(yy)]) * stride;
        for (int y = 0, n = c.count[std::size_t(yy)]; y < n; ++y, p += stride) {
            const std::int32_t w = k[y];
            for (std::size_t i = 0; i < stride; ++i) acc[i] += p[i] * w;
        }
        std::uint8_t* out = dst.rgb.data() + std::size_t(yy) * stride;
        for (std::size_t i = 0; i < stride; ++i) out[i] = clip8(acc[i]);
    }
    return dst;
}

namespace {

struct Taps { std::vector<int> i0, i1; std::vector<float> w0, w1; };

// Source indices/weights of one axis: half-pixel centers, clamped at 0, scale = in / out (fp32).
Taps taps(int in, int out) {
    Taps t;
    const float scale = float(in) / float(out);
    for (int o = 0; o < out; ++o) {
        float src = scale * (float(o) + 0.5F) - 0.5F;
        if (src < 0.0F) src = 0.0F;
        const int i0 = static_cast<int>(src);
        const float l1 = std::min(std::max(src - float(i0), 0.0F), 1.0F);
        t.i0.push_back(i0);
        t.i1.push_back(i0 < in - 1 ? i0 + 1 : i0);
        t.w0.push_back(1.0F - l1);
        t.w1.push_back(l1);
    }
    return t;
}

}  // namespace

std::vector<float> resize_bilinear_f32(const float* rgb, int h, int w, int oh, int ow) {
    if (rgb == nullptr || h <= 0 || w <= 0 || oh <= 0 || ow <= 0) bad("bilinear resize arguments");
    const auto th = taps(h, oh), tw = taps(w, ow);
    std::vector<float> out(std::size_t(oh) * std::size_t(ow) * 3);
    for (int y = 0; y < oh; ++y)
        for (int x = 0; x < ow; ++x)
            for (int c = 0; c < 3; ++c) {
                const auto at = [&](int yy, int xx) {
                    return rgb[(std::size_t(yy) * std::size_t(w) + std::size_t(xx)) * 3 + std::size_t(c)];
                };
                const std::size_t sy = std::size_t(y), sx = std::size_t(x);
                const float r0 = at(th.i0[sy], tw.i0[sx]) * tw.w0[sx] + at(th.i0[sy], tw.i1[sx]) * tw.w1[sx];
                const float r1 = at(th.i1[sy], tw.i0[sx]) * tw.w0[sx] + at(th.i1[sy], tw.i1[sx]) * tw.w1[sx];
                out[(sy * std::size_t(ow) + sx) * 3 + std::size_t(c)] = r0 * th.w0[sy] + r1 * th.w1[sy];
            }
    return out;
}

Image pad_center(const Image& im, int width, int height, std::array<std::uint8_t, 3> fill) {
    check(im);
    if (width < im.width || height < im.height) bad("pad canvas smaller than the image");
    Image out;
    out.width = width;
    out.height = height;
    out.rgb.resize(std::size_t(width) * std::size_t(height) * 3);
    for (std::size_t i = 0; i < out.rgb.size(); i += 3) {
        out.rgb[i] = fill[0];
        out.rgb[i + 1] = fill[1];
        out.rgb[i + 2] = fill[2];
    }
    const std::size_t x0 = std::size_t(width - im.width) / 2, y0 = std::size_t(height - im.height) / 2;
    for (int y = 0; y < im.height; ++y)
        std::copy_n(im.rgb.data() + std::size_t(y) * std::size_t(im.width) * 3, std::size_t(im.width) * 3,
                    out.rgb.data() + ((y0 + std::size_t(y)) * std::size_t(width) + x0) * 3);
    return out;
}

Size contain_size(int width, int height, int canvas_width, int canvas_height) {
    if (width <= 0 || height <= 0 || canvas_width <= 0 || canvas_height <= 0) bad("contain needs positive sizes");
    const double image_ratio = double(width) / double(height), canvas_ratio = double(canvas_width) / double(canvas_height);
    Size s{canvas_width, canvas_height};
    // nearbyint in the default rounding mode = round half to even, the rule of Python's round()
    if (image_ratio > canvas_ratio)
        s.height = int(std::nearbyint(double(height) / double(width) * double(canvas_width)));
    else if (image_ratio < canvas_ratio)
        s.width = int(std::nearbyint(double(width) / double(height) * double(canvas_height)));
    return s;
}

Image resize_contain_pad(const Image& im, int canvas_width, int canvas_height, std::array<std::uint8_t, 3> fill,
                         Resampler resampler) {
    check(im);
    const Size s = contain_size(im.width, im.height, canvas_width, canvas_height);
    const bool empty = s.width <= 0 || s.height <= 0;
    Image fitted;
    if (!empty) fitted = s.width == im.width && s.height == im.height ? im : resize(im, s.width, s.height, resampler);
    if (s.width == canvas_width && s.height == canvas_height) return fitted;
    Image out;
    out.width = canvas_width;
    out.height = canvas_height;
    out.rgb.resize(std::size_t(canvas_width) * std::size_t(canvas_height) * 3);
    for (std::size_t i = 0; i < out.rgb.size(); i += 3) {
        out.rgb[i] = fill[0];
        out.rgb[i + 1] = fill[1];
        out.rgb[i + 2] = fill[2];
    }
    // Only one axis has a margin (the width when both could: the order of the PIL function); half of it, to even.
    const auto half_even = [](int d) { return std::size_t(d / 2 + ((d & 1) != 0 && ((d / 2) & 1) != 0 ? 1 : 0)); };
    const bool margin_x = s.width != canvas_width;
    const std::size_t x0 = margin_x ? half_even(canvas_width - s.width) : 0;
    const std::size_t y0 = margin_x ? 0 : half_even(canvas_height - s.height);
    for (std::size_t y = 0; !empty && y < std::size_t(s.height); ++y)
        std::copy_n(fitted.rgb.data() + y * std::size_t(s.width) * 3, std::size_t(s.width) * 3,
                    out.rgb.data() + ((y0 + y) * std::size_t(canvas_width) + x0) * 3);
    return out;
}

std::vector<Image> tile(const Image& im, int tw, int th) {
    check(im);
    if (tw <= 0 || th <= 0 || im.width % tw != 0 || im.height % th != 0) bad("tile size does not divide the image");
    std::vector<Image> tiles;
    for (int ty = 0; ty < im.height / th; ++ty)
        for (int tx = 0; tx < im.width / tw; ++tx) {
            Image t;
            t.width = tw;
            t.height = th;
            t.rgb.resize(std::size_t(tw) * std::size_t(th) * 3);
            for (int y = 0; y < th; ++y)
                std::copy_n(im.rgb.data() + (std::size_t(ty * th + y) * std::size_t(im.width) + std::size_t(tx * tw)) * 3,
                            std::size_t(tw) * 3, t.rgb.data() + std::size_t(y) * std::size_t(tw) * 3);
            tiles.push_back(std::move(t));
        }
    return tiles;
}

}  // namespace strix::vision
