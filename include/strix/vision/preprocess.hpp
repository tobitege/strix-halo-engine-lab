#pragma once

// Shared vision preprocessing, part 2: RGB8 -> resized pixels -> normalized fp32 -> flattened patches.
// Host only (no accelerator code). A family adapter picks a ResizePolicy, a Resampler, mean/std and a PatchSpec;
// nothing here knows a model.

#include "strix/vision/image.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::vision {

struct Size {
    int width = 0, height = 0;
    friend bool operator==(const Size&, const Size&) = default;
};

// ---- target size ---------------------------------------------------------------------------------------------------

// The "smart resize" rule of the Qwen-VL class processors: both sides rounded (half to even) to a multiple of
// `factor`, then scaled down (floor) when over max_pixels or up (ceil) when under min_pixels, aspect kept.
[[nodiscard]] Size smart_resize(int height, int width, int factor, double min_pixels, double max_pixels);

// Any-res choice of the LLaVA-NeXT class processors: the pinpoint with the largest effective area
// min(scaled area, original area), ties broken by the least wasted area, first wins.
[[nodiscard]] Size select_best_resolution(Size original, std::span<const Size> pinpoints);

enum class Filter { bilinear, bicubic, lanczos };

struct Resampler {
    Filter filter = Filter::bicubic;
    // true: the filter support grows with the reduction factor (PIL `Image.resize`; uint8 results are bit-equal to
    // PIL). false: the support stays at the filter's own width (no low-pass on reduction).
    bool antialias = true;
};

struct ResizePolicy {
    enum class Kind { smart, fixed, any_res };
    Kind kind = Kind::smart;
    int factor = 32;                                      // smart: patch x merge
    double min_pixels = 3136, max_pixels = 12845056;      // smart
    Size fixed{};                                         // fixed
    std::vector<Size> pinpoints;                          // any_res
    Resampler resampler{};
    [[nodiscard]] Size target(int width, int height) const;
};

// ---- resampling ----------------------------------------------------------------------------------------------------

// Separable two-pass resize in 8-bit fixed point (22 fractional bits, horizontal pass first, each pass rounded and
// clamped to uint8). With antialias it reproduces PIL `Image.resize(size, resample)` bit for bit.
[[nodiscard]] Image resize(const Image& image, int out_width, int out_height, Resampler resampler = {});

// Bilinear resize of one [h][w][3] fp32 frame: half-pixel centres, source coordinate clamped at 0, no antialias.
[[nodiscard]] std::vector<float> resize_bilinear_f32(const float* rgb, int h, int w, int oh, int ow);

// Centre the image on a width x height canvas filled with `fill` (canvas must not be smaller than the image).
[[nodiscard]] Image pad_center(const Image& image, int width, int height, std::array<std::uint8_t, 3> fill);
// Largest size of the image's aspect ratio inside the canvas (PIL `ImageOps.contain`): one side is the canvas side,
// the other is rounded half to even and is 0 when an extreme aspect ratio leaves less than half a pixel.
// Requires FE_TONEAREST (the default floating-point rounding mode) for half-even sizing.
[[nodiscard]] Size contain_size(int width, int height, int canvas_width, int canvas_height);
// PIL `ImageOps.pad(image, (canvas_width, canvas_height), method, color=fill)`: contain-resize, then centre on the
// canvas; the margin offset is rounded half to even (pad_center rounds it down). Bit-equal to PIL with antialias.
// An image whose contained size has a zero side leaves the canvas all `fill`, as PIL does.
[[nodiscard]] Image resize_contain_pad(const Image& image, int canvas_width, int canvas_height,
                                       std::array<std::uint8_t, 3> fill, Resampler resampler = {});
// Raster-order tiles of tile_width x tile_height (the image size must be a multiple of the tile size).
[[nodiscard]] std::vector<Image> tile(const Image& image, int tile_width, int tile_height);

// ---- normalize and patchify ----------------------------------------------------------------------------------------

// [h][w][3] fp32: (v / 255 - mean[c]) / std[c].
[[nodiscard]] std::vector<float> normalize(const Image& image, std::array<float, 3> mean, std::array<float, 3> std);
// [h][w][3] fp32: (v / 255 - mean[c]) / std[c] with a true FP32 division by 255 (the order of a PyTorch
// `x / 255` processor; `normalize` multiplies by the reciprocal, which differs in the last place for some levels),
// then each value rounded to the nearest BF16 value (ties to even) when round_bf16 is set.
[[nodiscard]] std::vector<float> normalize_div255(const Image& image, std::array<float, 3> mean, std::array<float, 3> std,
                                                  bool round_bf16);
// In place on [n][3] fp32 samples: (v - mean[c]) / std[c].
void normalize_inplace(std::span<float> rgb, std::array<float, 3> mean, std::array<float, 3> std);

struct PatchSpec {
    int patch = 14;    // square spatial patch
    int temporal = 1;  // frames per patch; a single frame fills every temporal slot
    int merge = 1;     // rows are emitted in merge x merge block order (1 = raster order)
};

struct Patches {  // [grid_h * grid_w][3 * temporal * patch * patch]; feature order: channel, frame, y, x
    int grid_h = 0, grid_w = 0, feature = 0, merge = 1;
    std::vector<float> data;
    [[nodiscard]] std::size_t rows() const noexcept { return std::size_t(grid_h) * std::size_t(grid_w); }
    [[nodiscard]] std::size_t merged_tokens() const noexcept {
        return std::size_t(grid_h / merge) * std::size_t(grid_w / merge);
    }
};

// frames: 1 or spec.temporal pointers to [height][width][3] fp32 frames. height and width must be multiples of
// patch x merge.
[[nodiscard]] Patches patchify(std::span<const float* const> frames, int height, int width, const PatchSpec& spec);

// ---- identity and threading ----------------------------------------------------------------------------------------

// Lower-case hex SHA-256 over (adapter identity, width, height, RGB8). Two files that decode and resize to the same
// pixels under the same adapter share it.
[[nodiscard]] std::string pixel_digest(std::string_view adapter_identity, const Image& resized);

// Runs fn(0..count-1) on up to max_threads host threads (0 = hardware concurrency, capped at count). The first
// exception thrown by fn is rethrown after every worker has stopped.
void parallel_for(std::size_t count, unsigned max_threads, const std::function<void(std::size_t)>& fn);

}  // namespace strix::vision
