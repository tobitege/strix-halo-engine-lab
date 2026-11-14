#pragma once
// Per-family vision adapter: everything model-specific about an image tower. The shared runtime (encoder.hpp) owns
// the graph and the arithmetic; an adapter only names tensors, states its resize rule and patch order, runs its
// projector through DenseOps and describes the span the language model sees.
#include "strix/vision/encoder.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::vision {

struct Image;  // decoded pixels (preprocess library)

// Resize contract of a family, as data: sides are multiples of `multiple`, the area is kept inside
// [min_pixels, max_pixels], the row count after the projector never exceeds max_rows.
struct ResizeRule {
    unsigned multiple = 0;
    std::uint64_t min_pixels = 0, max_pixels = 0;
    unsigned max_rows = 0;
    float max_aspect = 0.0F;  // 0 = unlimited
};

// Checked geometry of a patch grid, for every adapter method that takes one (project(), layout(), a patch builder).
// A grid is supplied by a caller, so nothing about it is assumed: all products are formed in 64 bits before they are
// compared. Throws std::invalid_argument ("vision patch grid: ...").
inline constexpr std::uint64_t kMaxGridAxis = 65536;  // PatchGrid::y / ::x are 16-bit patch coordinates
struct MergedGrid {
    unsigned grid_h = 0, grid_w = 0, tokens = 0;  // after the merge x merge grouping
};
// Axis bounds alone: for a patch builder, before it allocates or narrows a coordinate.
inline void check_grid_axes(std::uint64_t grid_h, std::uint64_t grid_w) {
    if (grid_h == 0 || grid_w == 0) throw std::invalid_argument("vision patch grid: an empty axis");
    if (grid_h > kMaxGridAxis || grid_w > kMaxGridAxis)
        throw std::invalid_argument("vision patch grid: an axis of more than " + std::to_string(kMaxGridAxis) + " patches");
}
// The whole grid: positive axes inside the coordinate width, both multiples of `merge`, rows == grid_h * grid_w,
// coordinate vectors absent or one per row, and at most `max_tokens` merged image rows. With partial_cells an axis
// need not be a multiple of `merge`: the last cell of a row or column is partial and the merged grid rounds up (the
// adapter completes such a cell itself, e.g. with zero rows).
inline MergedGrid checked_merged_grid(const PatchGrid& grid, unsigned merge, std::uint64_t max_tokens,
                                      bool partial_cells = false) {
    if (merge == 0) throw std::invalid_argument("vision patch grid: a zero merge factor");
    check_grid_axes(grid.grid_h, grid.grid_w);
    if (!partial_cells && (grid.grid_h % merge != 0 || grid.grid_w % merge != 0))
        throw std::invalid_argument("vision patch grid: an axis is not a multiple of the merge factor");
    if (std::uint64_t{grid.grid_h} * grid.grid_w != grid.rows)
        throw std::invalid_argument("vision patch grid: rows differ from grid_h * grid_w");
    if (grid.y.size() != grid.x.size() || (!grid.y.empty() && grid.y.size() != grid.rows))
        throw std::invalid_argument("vision patch grid: coordinates differ from the row count");
    const std::uint64_t mh = (std::uint64_t{grid.grid_h} + merge - 1U) / merge;
    const std::uint64_t mw = (std::uint64_t{grid.grid_w} + merge - 1U) / merge;
    if (mh * mw > max_tokens)
        throw std::invalid_argument("vision patch grid: " + std::to_string(mh * mw) + " image rows, the limit is " +
                                    std::to_string(max_tokens));
    return {static_cast<unsigned>(mh), static_cast<unsigned>(mw), static_cast<unsigned>(mh * mw)};
}

// Tensor lookup of one checkpoint (GGUF, safetensors, pack). Views stay valid while the source lives.
class WeightSource {
public:
    virtual ~WeightSource() = default;
    virtual bool has(std::string_view name) const = 0;
    virtual Tensor tensor(std::string_view name) = 0;  // throws std::out_of_range when absent
};

// What the language model sees for one image: one entry per spliced row.
struct SpanLayout {
    enum class Role : std::uint8_t { image = 0, start = 1, newline = 2, end = 3 };
    std::vector<Role> role;
    std::vector<std::array<std::uint32_t, 3>> lm_pos;  // (t, h, w) relative to the span start; 1D families use [0]
    std::uint32_t real_token_id = 0;                   // id the family's per-row extras treat these rows as
    unsigned grid_h = 0, grid_w = 0;                   // merged grid
    unsigned rows() const { return static_cast<unsigned>(role.size()); }
};

class VisionAdapter {
public:
    virtual ~VisionAdapter() = default;
    virtual EncoderConfig config() const = 0;
    virtual std::string identity() const = 0;  // weights digest + arithmetic id: cache key and capability hash
    virtual ResizeRule resize() const = 0;
    virtual PatchGrid patches(const Image&) const = 0;
    virtual void bind(WeightSource&, EncoderWeights&) = 0;  // tensor-name mapping only
    virtual PositionInputs positions(const EncoderConfig& c, const EncoderWeights& w, const PatchGrid& g) const {
        return default_positions(c, w, g);
    }
    // Merger / unshuffle / pool + MLP and special rows; linears go through `dense`. The grid is the one the hidden
    // rows were encoded from (a projector that merges neighbours needs its shape and row order).
    virtual void project(EncoderOutput&, const PatchGrid&, DenseOps& dense, ProjectedRows& out) = 0;
    virtual SpanLayout layout(const PatchGrid&) const = 0;
    // Arithmetic policy the tower needs on the device (default: none). A family whose gate fails without a policy
    // bit states it here, so every service constructs the runner the same way.
    virtual EncoderOptions device_options() const { return {}; }
};

}  // namespace strix::vision
