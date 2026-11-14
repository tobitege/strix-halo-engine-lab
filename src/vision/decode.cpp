#include "strix/vision/image.hpp"

#include <climits>
#include <cstring>
#include <memory>
#include <utility>

// The vendored decoder is compiled into this translation unit only: static linkage, memory input only, PNG and JPEG
// only, and no assert on untrusted input (a decode failure is reported, never an abort).
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBI_NO_STDIO
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ASSERT(x) ((void)0)
#define STBI_MAX_DIMENSIONS (1 << 20)
#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include <stb/stb_image.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace strix::vision {
namespace {

using Bytes = std::span<const std::uint8_t>;

[[noreturn]] void fail(ImageErrorCode code, const std::string& what) { throw ImageError(code, "vision: " + what); }

std::uint32_t be32(const std::uint8_t* p) {
    return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) | (std::uint32_t(p[2]) << 8) | std::uint32_t(p[3]);
}

// Orientation tag (0x0112) of IFD0 in a TIFF block; 1 when absent or malformed.
int tiff_orientation(Bytes t) {
    if (t.size() < 8) return 1;
    const bool le = t[0] == 'I' && t[1] == 'I', be = t[0] == 'M' && t[1] == 'M';
    if (!le && !be) return 1;
    const auto u16 = [&](std::size_t o) {
        return le ? unsigned(t[o]) | (unsigned(t[o + 1]) << 8) : unsigned(t[o + 1]) | (unsigned(t[o]) << 8);
    };
    const auto u32 = [&](std::size_t o) {
        return le ? std::uint32_t(u16(o)) | (std::uint32_t(u16(o + 2)) << 16)
                  : std::uint32_t(u16(o + 2)) | (std::uint32_t(u16(o)) << 16);
    };
    if (u16(2) != 42) return 1;
    const std::size_t ifd = u32(4);
    if (ifd > t.size() || t.size() - ifd < 2) return 1;
    const std::size_t count = u16(ifd);
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t e = ifd + 2 + i * 12;
        if (e > t.size() || t.size() - e < 12) return 1;
        if (u16(e) != 0x0112) continue;
        if (u16(e + 2) != 3 || u32(e + 4) != 1) return 1;
        const unsigned v = u16(e + 8);
        return v >= 1 && v <= 8 ? int(v) : 1;
    }
    return 1;
}

int jpeg_orientation(Bytes b) {
    std::size_t p = 2;  // after SOI
    while (p + 4 <= b.size()) {
        if (b[p] != 0xFF) return 1;
        const unsigned marker = b[p + 1];
        if (marker == 0xFF) { ++p; continue; }                 // fill byte
        if (marker == 0xDA || marker == 0xD9) return 1;        // scan data or end: no EXIF before it
        if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) { p += 2; continue; }
        const std::size_t len = (std::size_t(b[p + 2]) << 8) | b[p + 3];
        if (len < 2 || len > b.size() - p - 2) return 1;
        if (marker == 0xE1 && len >= 8 && std::memcmp(b.data() + p + 4, "Exif\0\0", 6) == 0)
            return tiff_orientation(b.subspan(p + 10, len - 8));
        p += 2 + len;
    }
    return 1;
}

int png_orientation(Bytes b) {
    std::size_t p = 8;  // after the signature
    while (p + 12 <= b.size()) {
        const std::size_t len = be32(b.data() + p);
        if (len > b.size() - p - 12) return 1;
        if (std::memcmp(b.data() + p + 4, "eXIf", 4) == 0) return tiff_orientation(b.subspan(p + 8, len));
        if (std::memcmp(b.data() + p + 4, "IEND", 4) == 0) return 1;
        p += 12 + len;
    }
    return 1;
}

struct StbFree {
    void operator()(stbi_uc* p) const noexcept { stbi_image_free(p); }
};

}  // namespace

const char* format_name(ImageFormat format) noexcept {
    switch (format) {
        case ImageFormat::png: return "png";
        case ImageFormat::jpeg: return "jpeg";
        case ImageFormat::webp: return "webp";
        case ImageFormat::gif: return "gif";
        case ImageFormat::unknown: break;
    }
    return "unknown";
}

ImageFormat sniff_format(Bytes b) noexcept {
    static constexpr std::uint8_t png[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (b.size() >= 8 && std::memcmp(b.data(), png, 8) == 0) return ImageFormat::png;
    if (b.size() >= 3 && b[0] == 0xFF && b[1] == 0xD8 && b[2] == 0xFF) return ImageFormat::jpeg;
    if (b.size() >= 12 && std::memcmp(b.data(), "RIFF", 4) == 0 && std::memcmp(b.data() + 8, "WEBP", 4) == 0)
        return ImageFormat::webp;
    if (b.size() >= 6 && (std::memcmp(b.data(), "GIF87a", 6) == 0 || std::memcmp(b.data(), "GIF89a", 6) == 0))
        return ImageFormat::gif;
    return ImageFormat::unknown;
}

ImageInfo probe(Bytes bytes) {
    ImageInfo info;
    info.format = sniff_format(bytes);
    if (info.format == ImageFormat::unknown) fail(ImageErrorCode::unknown_format, "not a recognised image format");
    if (info.format != ImageFormat::png && info.format != ImageFormat::jpeg)
        fail(ImageErrorCode::unsupported_format, std::string(format_name(info.format)) + " images are not supported");
    if (bytes.size() > std::size_t(INT_MAX)) fail(ImageErrorCode::too_many_bytes, "encoded image too large");
    int w = 0, h = 0, n = 0;
    if (!stbi_info_from_memory(bytes.data(), int(bytes.size()), &w, &h, &n) || w <= 0 || h <= 0) {
        // The decoder refuses a PNG whose header declares more than 2^30 samples; report that as a size refusal.
        if (info.format == ImageFormat::png && bytes.size() >= 24 && std::memcmp(bytes.data() + 12, "IHDR", 4) == 0 &&
            std::uint64_t(be32(bytes.data() + 16)) * be32(bytes.data() + 20) > (std::uint64_t(1) << 28))
            fail(ImageErrorCode::too_many_pixels, "image over the decoder's pixel limit");
        fail(ImageErrorCode::corrupt, std::string("unreadable ") + format_name(info.format) + " header");
    }
    info.width = w;
    info.height = h;
    info.channels = n;
    info.orientation = info.format == ImageFormat::jpeg ? jpeg_orientation(bytes) : png_orientation(bytes);
    return info;
}

Image apply_orientation(Image src, int orientation) {
    if (orientation < 2 || orientation > 8) return src;
    const int w = src.width, h = src.height;
    const bool swap = orientation >= 5;
    Image dst;
    dst.width = swap ? h : w;
    dst.height = swap ? w : h;
    dst.rgb.resize(src.rgb.size());
    for (int y = 0; y < dst.height; ++y)
        for (int x = 0; x < dst.width; ++x) {
            int sx = x, sy = y;
            switch (orientation) {
                case 2: sx = w - 1 - x; break;                    // mirror
                case 3: sx = w - 1 - x; sy = h - 1 - y; break;    // rotate 180
                case 4: sy = h - 1 - y; break;                    // flip
                case 5: sx = y; sy = x; break;                    // transpose
                case 6: sx = y; sy = h - 1 - x; break;            // rotate 90 clockwise
                case 7: sx = w - 1 - y; sy = h - 1 - x; break;    // transverse
                default: sx = w - 1 - y; sy = x; break;           // 8: rotate 90 counter-clockwise
            }
            const std::uint8_t* s = src.rgb.data() + (std::size_t(sy) * std::size_t(w) + std::size_t(sx)) * 3;
            std::uint8_t* d = dst.rgb.data() + (std::size_t(y) * std::size_t(dst.width) + std::size_t(x)) * 3;
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
        }
    return dst;
}

Image decode(Bytes bytes, const DecodeLimits& limits, const ColorPolicy& color) {
    if (bytes.size() > limits.max_bytes) fail(ImageErrorCode::too_many_bytes, "encoded image over the byte limit");
    const ImageInfo info = probe(bytes);
    const std::uint64_t pixels = std::uint64_t(info.width) * std::uint64_t(info.height);
    if (pixels > limits.max_pixels) fail(ImageErrorCode::too_many_pixels, "image over the pixel limit");
    int w = 0, h = 0, n = 0;
    const std::unique_ptr<stbi_uc, StbFree> data(stbi_load_from_memory(bytes.data(), int(bytes.size()), &w, &h, &n, 0));
    if (!data || w != info.width || h != info.height || n < 1 || n > 4)
        fail(ImageErrorCode::corrupt, std::string("undecodable ") + format_name(info.format) + " data");
    Image out;
    out.width = w;
    out.height = h;
    out.rgb.resize(std::size_t(pixels) * 3);
    const stbi_uc* s = data.get();
    std::uint8_t* d = out.rgb.data();
    const bool has_alpha = n == 2 || n == 4, composite = has_alpha && color.alpha == AlphaPolicy::composite;
    for (std::uint64_t i = 0; i < pixels; ++i, s += n, d += 3) {
        if (n >= 3) { d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; }
        else d[0] = d[1] = d[2] = s[0];
        if (composite) {
            const unsigned a = s[n - 1];
            for (int c = 0; c < 3; ++c)
                d[c] = std::uint8_t((unsigned(d[c]) * a + unsigned(color.background[std::size_t(c)]) * (255U - a) + 127U) / 255U);
        }
    }
    return limits.apply_exif_orientation ? apply_orientation(std::move(out), info.orientation) : out;
}

}  // namespace strix::vision
