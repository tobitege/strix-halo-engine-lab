#pragma once

// Shared vision preprocessing, part 1: untrusted image bytes -> RGB8.
//
// Formats: PNG and JPEG (vendored single-header decoder, no system packages). WebP and GIF are recognised by their
// magic bytes and refused with ImageErrorCode::unsupported_format. The magic bytes decide the format; a declared
// media type is never consulted.
//
// Safety: probe() reads only the header; decode() checks the encoded size and the pixel count from the header before
// any pixel buffer is allocated (decompression-bomb guard). Multi-frame files yield their first frame.
//
// Colour: the default policy follows PIL `Image.convert("RGB")`: alpha is dropped (not composited), palette entries
// are expanded, grey is replicated, 16-bit samples keep their high byte. EXIF orientation (JPEG APP1, PNG eXIf) is
// applied the way `ImageOps.exif_transpose` does.
//
// Every function here is reentrant and uses no global state, so callers may run it on any host thread.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::vision {

enum class ImageFormat { unknown, png, jpeg, webp, gif };
[[nodiscard]] const char* format_name(ImageFormat format) noexcept;

enum class ImageErrorCode {
    unknown_format,      // magic bytes match no image format
    unsupported_format,  // a known format this build does not decode (WebP, GIF)
    too_many_bytes,      // encoded size over DecodeLimits::max_bytes
    too_many_pixels,     // header width x height over DecodeLimits::max_pixels
    corrupt,             // header or pixel data does not decode
    invalid_argument,    // a caller error (sizes, policy)
};

class ImageError : public std::runtime_error {
public:
    ImageError(ImageErrorCode code, const std::string& what) : std::runtime_error(what), code_(code) {}
    [[nodiscard]] ImageErrorCode code() const noexcept { return code_; }

private:
    ImageErrorCode code_;
};

struct ImageInfo {
    ImageFormat format = ImageFormat::unknown;
    int width = 0, height = 0;  // as stored, before the EXIF orientation is applied
    int channels = 0;           // 1 grey, 2 grey+alpha, 3 RGB, 4 RGBA (JPEG CMYK reports 3)
    int orientation = 1;        // EXIF orientation 1..8 (1 when absent or invalid)
};

struct Image {  // [height][width][3], row-major, 8 bits per sample
    int width = 0, height = 0;
    std::vector<std::uint8_t> rgb;
    [[nodiscard]] std::size_t pixels() const noexcept { return std::size_t(width) * std::size_t(height); }
};

enum class AlphaPolicy {
    drop,       // PIL convert("RGB"): the colour samples are kept, alpha is ignored
    composite,  // over ColorPolicy::background: (c * a + bg * (255 - a) + 127) / 255
};

struct ColorPolicy {
    AlphaPolicy alpha = AlphaPolicy::drop;
    std::array<std::uint8_t, 3> background{255, 255, 255};
};

struct DecodeLimits {
    std::size_t max_bytes = std::size_t(20) << 20;  // encoded bytes per image
    std::uint64_t max_pixels = 40'000'000;          // decoded pixels per image, checked on the header
    bool apply_exif_orientation = true;
};

// Format from the leading magic bytes only.
[[nodiscard]] ImageFormat sniff_format(std::span<const std::uint8_t> bytes) noexcept;
// Header only: format, stored size, channels, EXIF orientation. Throws ImageError.
[[nodiscard]] ImageInfo probe(std::span<const std::uint8_t> bytes);
// Full decode to RGB8 under the limits. Throws ImageError; never allocates pixels for a refused image.
[[nodiscard]] Image decode(std::span<const std::uint8_t> bytes, const DecodeLimits& limits = {},
                           const ColorPolicy& color = {});
// The transform of ImageOps.exif_transpose for orientation 1..8 (other values return the image unchanged).
[[nodiscard]] Image apply_orientation(Image image, int orientation);

}  // namespace strix::vision
