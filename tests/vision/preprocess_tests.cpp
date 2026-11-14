// Host tests of the shared vision preprocessing: decode (colour modes, EXIF, limits, refusals), the fixed-point
// resampler against the reference resampler's pixels, target-size rules, normalize/patchify order, digest, pool.
//   strix-vision-preprocess-tests FIXTURE_DIR

#include "strix/vision/preprocess.hpp"

#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <crtdbg.h>
#endif

#include "preprocess_reference.inc"

namespace {

using namespace strix::vision;

int g_checks = 0, g_failures = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

// The integer patterns of tools/vision/make_images.py.
int sample(int kind, int x, int y, int c, int w, int h) {
    static const int colours[4][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {250, 240, 10}};
    if (kind == 0) {
        if (c == 0) return x * 255 / (w > 1 ? w - 1 : 1);
        if (c == 1) return y * 255 / (h > 1 ? h - 1 : 1);
        return (x + y) * 255 / (w + h > 2 ? w + h - 2 : 1);
    }
    if (kind == 1) {
        std::uint32_t v = (std::uint32_t(x) * 73856093U) ^ (std::uint32_t(y) * 19349663U) ^ (std::uint32_t(c + 1) * 83492791U);
        v *= 2654435761U;
        return int(v >> 24);
    }
    return colours[((x / 5) + (y / 3) * 2) % 4][c];
}

Image pattern(int kind, int w, int h) {
    Image im;
    im.width = w;
    im.height = h;
    im.rgb.resize(im.pixels() * 3);
    std::size_t i = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) im.rgb[i++] = std::uint8_t(sample(kind, x, y, c, w, h));
    return im;
}

std::uint64_t fnv64(const std::uint8_t* p, std::size_t n) {
    std::uint64_t h = 0xCBF29CE484222325ULL;
    for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001B3ULL;
    return h;
}

std::uint64_t fnv64(const std::vector<float>& v) {
    return fnv64(reinterpret_cast<const std::uint8_t*>(v.data()), v.size() * sizeof(float));
}

std::string g_dir;

std::vector<std::uint8_t> load(const std::string& name) {
    std::ifstream in(g_dir + "/" + name, std::ios::binary);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    check(!bytes.empty(), "fixture " + name + " is readable");
    return bytes;
}

bool same(const Image& a, const Image& b) { return a.width == b.width && a.height == b.height && a.rgb == b.rgb; }

struct Diff { int max = 0; double mean = 0; };

Diff diff(const Image& a, const std::vector<std::uint8_t>& ref) {
    Diff d;
    if (a.rgb.size() != ref.size() || ref.empty()) return Diff{999, 999};
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const int e = std::abs(int(a.rgb[i]) - int(ref[i]));
        d.max = e > d.max ? e : d.max;
        d.mean += e;
    }
    d.mean /= double(ref.size());
    return d;
}

ImageErrorCode error_of(const std::vector<std::uint8_t>& bytes, const DecodeLimits& limits = {}) {
    try {
        (void)decode(bytes, limits);
    } catch (const ImageError& e) {
        return e.code();
    }
    return ImageErrorCode::invalid_argument;  // "no error": never expected by the callers below
}

void decode_tests() {
    const Image noise = pattern(1, 37, 23);
    check(same(decode(load("rgb_noise.png")), noise), "RGB PNG is exact");
    check(same(decode(load("rgba_noise.png")), noise), "RGBA PNG drops alpha (reference convert semantics)");
    check(same(decode(load("rgb16_noise.png")), noise), "16-bit PNG keeps the high byte");
    check(same(decode(load("palette_blocks.png")), pattern(2, 37, 23)), "palette PNG with transparency expands to RGB");
    Image grey = noise;
    for (std::size_t i = 0; i < grey.rgb.size(); i += 3) grey.rgb[i + 1] = grey.rgb[i + 2] = grey.rgb[i];
    check(same(decode(load("grey_noise.png")), grey), "grey PNG replicates");
    check(same(decode(load("grey_alpha_noise.png")), grey), "grey+alpha PNG replicates and drops alpha");

    ColorPolicy over;
    over.alpha = AlphaPolicy::composite;
    over.background = {10, 200, 255};
    Image expect = noise;
    for (int y = 0; y < 23; ++y)
        for (int x = 0; x < 37; ++x)
            for (std::size_t c = 0; c < 3; ++c) {
                const unsigned a = unsigned(x * 255 / 36);
                auto& v = expect.rgb[(std::size_t(y) * 37 + std::size_t(x)) * 3 + c];
                v = std::uint8_t((unsigned(v) * a + unsigned(over.background[c]) * (255U - a) + 127U) / 255U);
            }
    check(same(decode(load("rgba_noise.png"), {}, over), expect), "composite policy blends over the background");

    const ImageInfo info = probe(load("rgba_noise.png"));
    check(info.format == ImageFormat::png && info.width == 37 && info.height == 23 && info.channels == 4 &&
              info.orientation == 1, "probe reads the PNG header");

    // JPEG: compared with the reference loader's pixels (a different IDCT, so a small bounded difference).
    for (const char* name : {"rgb_grad_444", "rgb_grad_420_progressive", "grey_grad", "cmyk_grad", "exif6_grad"}) {
        const Image im = decode(load(std::string(name) + ".jpg"));
        const Diff d = diff(im, load(std::string(name) + ".rgb"));
        std::printf("jpeg %-26s vs reference loader: max %d mean %.3f\n", name, d.max, d.mean);
        check(im.width == 48 && im.height == 32, std::string(name) + " size (orientation applied)");
        check(d.max <= 6 && d.mean <= 1.0, std::string(name) + " within the JPEG decoder bound");
    }
    const ImageInfo jpeg = probe(load("exif6_grad.jpg"));
    check(jpeg.format == ImageFormat::jpeg && jpeg.width == 32 && jpeg.height == 48 && jpeg.orientation == 6,
          "probe reports stored size and orientation");
    DecodeLimits raw;
    raw.apply_exif_orientation = false;
    check(decode(load("exif6_grad.jpg"), raw).width == 32, "orientation can be left unapplied");

    const Image upright = pattern(1, 9, 6);
    for (int k = 1; k <= 8; ++k) {
        const auto bytes = load("exif" + std::to_string(k) + "_noise.png");
        check(probe(bytes).orientation == k, "PNG eXIf orientation " + std::to_string(k) + " is read");
        check(same(decode(bytes), upright), "orientation " + std::to_string(k) + " matches the reference transpose");
    }

    // Refusals.
    const auto png = load("rgb_noise.png");
    check(error_of({}) == ImageErrorCode::unknown_format, "empty input");
    check(error_of({'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!'}) == ImageErrorCode::unknown_format, "text input");
    check(error_of({'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'E', 'B', 'P', 'V', 'P', '8', ' '}) == ImageErrorCode::unsupported_format,
          "WebP is recognised and refused");
    check(error_of({'G', 'I', 'F', '8', '9', 'a', 1, 0, 1, 0}) == ImageErrorCode::unsupported_format, "GIF is refused");
    check(sniff_format(png) == ImageFormat::png && sniff_format(load("grey_grad.jpg")) == ImageFormat::jpeg, "magic bytes");
    std::vector<std::uint8_t> cut(png.begin(), png.begin() + std::ptrdiff_t(png.size() / 2));
    check(error_of(cut) == ImageErrorCode::corrupt, "truncated PNG");
    std::vector<std::uint8_t> head(png.begin(), png.begin() + 20);
    check(error_of(head) == ImageErrorCode::corrupt, "PNG cut inside the header");
    std::vector<std::uint8_t> jcut = load("rgb_grad_444.jpg");
    jcut.resize(40);
    check(error_of(jcut) == ImageErrorCode::corrupt, "truncated JPEG");
    DecodeLimits small;
    small.max_bytes = png.size() - 1;
    check(error_of(png, small) == ImageErrorCode::too_many_bytes, "byte limit");
    DecodeLimits few;
    few.max_pixels = 37 * 23 - 1;
    check(error_of(png, few) == ImageErrorCode::too_many_pixels, "pixel limit");
    // Decompression bombs: headers that declare huge images followed by an empty data chunk.
    const auto header = [](std::uint32_t side) {
        const auto b = [&](int shift) { return std::uint8_t((side >> shift) & 0xFF); };
        return std::vector<std::uint8_t>{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13, 'I', 'H', 'D', 'R',
                                         b(24), b(16), b(8), b(0), b(24), b(16), b(8), b(0), 8, 2, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 'I', 'D', 'A', 'T', 0, 0, 0, 0};
    };
    check(probe(header(16000)).width == 16000, "bomb header is readable");
    check(error_of(header(16000)) == ImageErrorCode::too_many_pixels, "bomb is refused on the header");
    check(error_of(header(60000)) == ImageErrorCode::too_many_pixels, "a header over the decoder limit is a size refusal");
    check(error_of(header(0)) == ImageErrorCode::corrupt, "zero-size header");
}

void resize_tests() {
    static const Filter filters[] = {Filter::bilinear, Filter::bicubic, Filter::lanczos};
    static const char* names[] = {"bilinear", "bicubic", "lanczos"};
    int mismatches = 0;
    for (const ResizeRef& r : kResizeRef) {
        const Image out = resize(pattern(r.pattern, r.w, r.h), r.ow, r.oh, Resampler{filters[r.filter], true});
        const bool ok = out.width == r.ow && out.height == r.oh && fnv64(out.rgb.data(), out.rgb.size()) == r.fnv;
        if (!ok) {
            ++mismatches;
            std::printf("resize mismatch: pattern %d %dx%d -> %dx%d %s\n", r.pattern, r.w, r.h, r.ow, r.oh, names[r.filter]);
        }
    }
    std::printf("resize: %zu reference cases, %d mismatches\n", std::size(kResizeRef), mismatches);
    check(mismatches == 0, "fixed-point resize is bit-equal to the reference resampler");

    const Image src = pattern(1, 40, 30);
    for (const Filter f : filters) {
        check(same(resize(src, 40, 30, Resampler{f, true}), src), "same-size resize is the identity");
        check(same(resize(src, 100, 75, Resampler{f, false}), resize(src, 100, 75, Resampler{f, true})),
              "antialias does not change an enlargement");
    }
    // Without antialias a 10x reduction samples a narrow window; with it the whole block is averaged.
    Image stripes = pattern(0, 100, 10);
    for (int y = 0; y < 10; ++y)
        for (int x = 0; x < 100; ++x)
            for (std::size_t c = 0; c < 3; ++c)
                stripes.rgb[(std::size_t(y) * 100 + std::size_t(x)) * 3 + c] = (x % 10) < 5 ? 0 : 255;
    const Image aa = resize(stripes, 10, 10, Resampler{Filter::bilinear, true});
    const Image raw = resize(stripes, 10, 10, Resampler{Filter::bilinear, false});
    check(std::abs(int(aa.rgb[5 * 3]) - 127) <= 8, "antialiased reduction averages the block");
    check(raw.rgb != aa.rgb, "non-antialiased reduction differs");

    // fp32 bilinear: identity at the same size, constant stays constant, corners clamp.
    std::vector<float> f(std::size_t(6) * 5 * 3);
    for (std::size_t i = 0; i < f.size(); ++i) f[i] = float(i % 17) / 16.0F;
    check(resize_bilinear_f32(f.data(), 6, 5, 6, 5) == f, "fp32 bilinear identity");
    const std::vector<float> flat(std::size_t(4) * 4 * 3, 0.25F);
    bool constant = true;
    for (const float v : resize_bilinear_f32(flat.data(), 4, 4, 9, 7)) constant = constant && v == 0.25F;
    check(constant, "fp32 bilinear keeps a constant image");

    const Image padded = pad_center(pattern(1, 4, 2), 8, 6, {1, 2, 3});
    check(padded.width == 8 && padded.height == 6 && padded.rgb[0] == 1 && padded.rgb[1] == 2 && padded.rgb[2] == 3 &&
              padded.rgb[(std::size_t(2) * 8 + 2) * 3] == std::uint8_t(sample(1, 0, 0, 0, 4, 2)), "pad_center");
    // contain + pad: sizes and margin origins as PIL `ImageOps.contain` / `ImageOps.pad` give them (the free side and
    // half of the margin are rounded half to even: 2.5 -> 2, 1.5 -> 2, 3.5 -> 4, 4.5 -> 4, 0.5 -> 0).
    check(contain_size(4, 2, 8, 6) == Size{8, 4} && contain_size(2, 1, 14, 14) == Size{14, 7} &&
              contain_size(1, 2, 6, 6) == Size{3, 6} && contain_size(1, 2, 5, 5) == Size{2, 5} &&
              contain_size(4, 3, 2, 2) == Size{2, 2} && contain_size(4, 1, 2, 2) == Size{2, 0} &&
              contain_size(517, 389, 630, 476) == Size{630, 474} && contain_size(1600, 200, 1610, 210) == Size{1610, 201} &&
              contain_size(5, 3, 5, 3) == Size{5, 3}, "contain_size");
    {
        struct Case {
            int w, h, cw, ch, x0, y0;
        };
        bool placed = true;
        for (const Case c : {Case{4, 2, 8, 6, 0, 1}, Case{2, 1, 6, 6, 0, 2}, Case{2, 1, 14, 14, 0, 4}, Case{1, 2, 6, 6, 2, 0},
                             Case{1, 2, 5, 5, 2, 0}, Case{16, 2, 161, 21, 0, 0}, Case{5, 3, 5, 3, 0, 0}}) {
            const Image source = pattern(3, c.w, c.h);
            const Size s = contain_size(c.w, c.h, c.cw, c.ch);
            const Image fitted = s.width == c.w && s.height == c.h ? source : resize(source, s.width, s.height);
            const Image out = resize_contain_pad(source, c.cw, c.ch, {127, 126, 125});
            placed = placed && out.width == c.cw && out.height == c.ch && out.rgb.size() == std::size_t(c.cw) * std::size_t(c.ch) * 3;
            for (int y = 0; placed && y < c.ch; ++y)
                for (int x = 0; placed && x < c.cw; ++x) {
                    const bool inside = x >= c.x0 && x < c.x0 + s.width && y >= c.y0 && y < c.y0 + s.height;
                    for (int ch = 0; ch < 3; ++ch) {
                        const std::uint8_t got = out.rgb[(std::size_t(y) * std::size_t(c.cw) + std::size_t(x)) * 3 + std::size_t(ch)];
                        const std::uint8_t want = inside ? fitted.rgb[(std::size_t(y - c.y0) * std::size_t(s.width) + std::size_t(x - c.x0)) * 3 + std::size_t(ch)]
                                                         : std::uint8_t(127 - ch);
                        placed = placed && got == want;
                    }
                }
        }
        check(placed, "resize_contain_pad places the fitted image");
        const Image gone = resize_contain_pad(pattern(3, 4, 1), 2, 2, {9, 8, 7});  // less than half a pixel high
        check(gone.width == 2 && gone.height == 2 && gone.rgb == std::vector<std::uint8_t>{9, 8, 7, 9, 8, 7, 9, 8, 7, 9, 8, 7},
              "resize_contain_pad: a vanished image leaves the fill");
    }
    {
        // (v / 255 - 0.5) / 0.5 with a true division, FP32 bits from the PyTorch expression; BF16 rounding to even.
        // Level 127 is where the reciprocal form of normalize() lands on another BF16 value.
        Image levels;
        levels.width = 6;
        levels.height = 1;
        levels.rgb = {0, 0, 0, 1, 1, 1, 127, 127, 127, 128, 128, 128, 200, 200, 200, 255, 255, 255};
        const std::array<float, 3> half{0.5F, 0.5F, 0.5F};
        const std::vector<float> exact = normalize_div255(levels, half, half, false);
        const std::vector<float> rounded = normalize_div255(levels, half, half, true);
        const std::uint32_t want_exact[6] = {0xbf800000U, 0xbf7dfdfeU, 0xbb808080U, 0x3b808100U, 0x3f119192U, 0x3f800000U};
        const std::uint32_t want_rounded[6] = {0xbf800000U, 0xbf7e0000U, 0xbb810000U, 0x3b810000U, 0x3f120000U, 0x3f800000U};
        bool bits = exact.size() == 18 && rounded.size() == 18;
        for (std::size_t i = 0; bits && i < 18; ++i)
            bits = std::bit_cast<std::uint32_t>(exact[i]) == want_exact[i / 3] &&
                   std::bit_cast<std::uint32_t>(rounded[i]) == want_rounded[i / 3];
        check(bits, "normalize_div255 bits");
    }
    const Image grid = pattern(1, 12, 8);
    const auto tiles = tile(grid, 4, 4);
    check(tiles.size() == 6 && tiles[4].rgb[0] == grid.rgb[(std::size_t(4) * 12 + 4) * 3] &&
              tiles[5].rgb[(std::size_t(3) * 4 + 3) * 3 + 2] == grid.rgb[(std::size_t(7) * 12 + 11) * 3 + 2], "tile order");
    bool threw = false;
    try { (void)tile(grid, 5, 4); } catch (const ImageError&) { threw = true; }
    check(threw, "tile rejects a size that does not divide");
}

void target_tests() {
    for (const SmartRef& r : kSmartRef) {
        const Size s = smart_resize(r.h, r.w, r.factor, r.min_pixels, r.max_pixels);
        check(s.height == r.oh && s.width == r.ow,
              "smart_resize " + std::to_string(r.h) + "x" + std::to_string(r.w) + " factor " + std::to_string(r.factor));
    }
    std::vector<Size> pins;
    for (const auto& p : kPinpoints) pins.push_back(Size{p[0], p[1]});
    for (const BestRef& r : kBestRef)
        check(select_best_resolution(Size{r.w, r.h}, pins) == Size{r.ow, r.oh},
              "any-res " + std::to_string(r.w) + "x" + std::to_string(r.h));
    ResizePolicy policy;
    policy.factor = 42;
    policy.min_pixels = 295936;
    policy.max_pixels = 1806336;
    check(policy.target(1008, 756) == Size{1008, 756}, "policy: smart");
    policy.kind = ResizePolicy::Kind::fixed;
    policy.fixed = Size{448, 448};
    check(policy.target(1008, 756) == Size{448, 448}, "policy: fixed");
    policy.kind = ResizePolicy::Kind::any_res;
    policy.pinpoints = pins;
    check(policy.target(640, 480) == select_best_resolution(Size{640, 480}, pins), "policy: any-res");
    bool threw = false;
    try { (void)smart_resize(0, 5, 32, 3136, 12845056); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "smart_resize rejects an empty image");
}

void patch_tests() {
    const Image im = pattern(1, 8, 4);
    const auto n = normalize(im, {0.5F, 0.25F, 0.0F}, {0.5F, 0.5F, 1.0F});
    check(n.size() == im.rgb.size() && n[0] == (float(im.rgb[0]) * (1.0F / 255.0F) - 0.5F) / 0.5F &&
              n[4] == (float(im.rgb[4]) * (1.0F / 255.0F) - 0.25F) / 0.5F && n[5] == float(im.rgb[5]) * (1.0F / 255.0F),
          "normalize is per channel");

    // Frame value encodes (frame, y, x, c) so every destination slot can be decoded back.
    const int h = 8, w = 12;
    const auto frame = [&](int t) {
        std::vector<float> f(std::size_t(h) * w * 3);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c) f[(std::size_t(y) * w + x) * 3 + c] = float(t * 1000 + y * 100 + x * 3 + c);
        return f;
    };
    const auto f0 = frame(0), f1 = frame(1);
    const float* two[] = {f0.data(), f1.data()};
    const float* one[] = {f0.data()};
    const PatchSpec spec{2, 2, 2};
    const Patches p = patchify(two, h, w, spec);
    check(p.grid_h == 4 && p.grid_w == 6 && p.feature == 24 && p.rows() == 24 && p.merged_tokens() == 6, "patch grid");
    bool order = true;
    std::size_t row = 0;
    for (int bh = 0; bh < 2; ++bh)
        for (int bw = 0; bw < 3; ++bw)
            for (int mh = 0; mh < 2; ++mh)
                for (int mw = 0; mw < 2; ++mw, ++row) {
                    const float* d = p.data.data() + row * 24;
                    for (int c = 0; c < 3; ++c)
                        for (int t = 0; t < 2; ++t)
                            for (int py = 0; py < 2; ++py)
                                for (int px = 0; px < 2; ++px) {
                                    const int y = (bh * 2 + mh) * 2 + py, x = (bw * 2 + mw) * 2 + px;
                                    order = order && *d++ == float(t * 1000 + y * 100 + x * 3 + c);
                                }
                }
    check(order, "patch rows follow merge-block order; features follow channel, frame, y, x");
    const Patches single = patchify(one, h, w, spec);
    check(single.data[4] == single.data[0] && single.data[0] == 0.0F && p.data[4] == 1000.0F,
          "a single frame fills both temporal slots");
    const Patches raster = patchify(one, h, w, PatchSpec{4, 1, 1});
    check(raster.grid_h == 2 && raster.grid_w == 3 && raster.data[std::size_t(raster.feature) * 4] == float(400 + 4 * 3),
          "merge 1 is raster order");
    bool threw = false;
    try { (void)patchify(one, h, w, PatchSpec{4, 1, 2}); } catch (const ImageError&) { threw = true; }
    check(threw, "patchify rejects a size that is not a multiple of patch x merge");

    // The H3 text-encoder recipe composed from the shared pieces (fp32 bilinear, (x - 0.5) / 0.5, patch 16 x
    // temporal 2, 2x2 merge): hashes recorded from the family-local code before it moved here.
    struct H3Ref { int w, h; std::uint64_t fnv; };
    static const H3Ref h3[] = {
#include "h3_patches_reference.inc"
    };
    for (const H3Ref& r : h3) {
        const Image src = pattern(1, r.w, r.h);
        std::vector<float> rgb(src.rgb.size());
        for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = float(src.rgb[i]) / 255.0F;
        const Size s = smart_resize(r.h, r.w, 32, 3136, 12845056);
        auto resized = resize_bilinear_f32(rgb.data(), r.h, r.w, s.height, s.width);
        normalize_inplace(resized, {0.5F, 0.5F, 0.5F}, {0.5F, 0.5F, 0.5F});
        const float* frames[] = {resized.data()};
        const std::uint64_t got = fnv64(patchify(frames, s.height, s.width, PatchSpec{16, 2, 2}).data);
        if (got != r.fnv) std::printf("h3 recipe %dx%d: got 0x%016llXULL\n", r.w, r.h, static_cast<unsigned long long>(got));
        check(got == r.fnv, "H3 recipe patches are bit-equal to the pre-move code");
    }
}

void identity_tests() {
    const Image a = pattern(1, 16, 8);
    Image b = a;
    const std::string d = pixel_digest("adapter-a", a);
    check(d.size() == 64 && d == pixel_digest("adapter-a", b), "digest is stable");
    b.rgb[5] ^= 1;
    check(d != pixel_digest("adapter-a", b), "a changed pixel changes the digest");
    check(d != pixel_digest("adapter-b", a), "the adapter identity is part of the digest");
    Image c = a;
    c.width = 8;
    c.height = 16;
    check(d != pixel_digest("adapter-a", c), "the size is part of the digest");
    // Two encodings of the same pixels share one digest.
    check(pixel_digest("x", decode(load("rgb_noise.png"))) == pixel_digest("x", decode(load("rgb16_noise.png"))),
          "files that decode to the same pixels share a digest");

    std::atomic<int> sum{0};
    parallel_for(100, 4, [&](std::size_t i) { sum += int(i); });
    check(sum == 4950, "parallel_for visits every index once");
    bool threw = false;
    try {
        parallel_for(8, 3, [](std::size_t i) { if (i == 5) throw ImageError(ImageErrorCode::corrupt, "x"); });
    } catch (const ImageError&) { threw = true; }
    check(threw, "parallel_for rethrows");
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_MSC_VER)
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    if (argc < 2) {
        std::printf("usage: strix-vision-preprocess-tests FIXTURE_DIR\n");
        return 2;
    }
    g_dir = argv[1];
    try {
        decode_tests();
        resize_tests();
        target_tests();
        patch_tests();
        identity_tests();
    } catch (const std::exception& e) {
        std::printf("FAIL: unexpected exception: %s\n", e.what());
        ++g_failures;
    }
    std::printf("vision-preprocess: %d checks, %s\n", g_checks, g_failures ? "FAILED" : "ok");
    return g_failures ? 1 : 0;
}
