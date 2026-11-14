#pragma once

// DeepSeek V4.1 image rows in the language model prefill: what the family adds on top of the shared pieces
// (session/embed_splice.hpp, vision/embed_span.hpp), and the memory image input costs.

#include "vision_adapter.hpp"

#include "strix/serve/slot_capacity.hpp"
#include "strix/session/embed_splice.hpp"
#include "strix/vision/embed_span.hpp"
#include "strix/vision/encoder.hpp"

#include <cstdint>

namespace strix::deepseek41 {

// The family's position of a row is its prompt index (layout() gives 0 .. rows - 1 on one axis), a span has no
// deep rows, and every row of a span (start, image, newline, end) is an image row for the routers and the Engram.
[[nodiscard]] inline session::EmbedResolveRules vision_resolve_rules(const VisionAdapter& adapter) {
    session::EmbedResolveRules r;
    r.width = adapter.model().projection_size;
    r.prompt_index_positions = true;
    r.allow_deep = false;
    return r;
}

// One image for the interner. axes = 1: the layout's positions are stored, so resolve_embed_run checks them.
[[nodiscard]] inline session::EmbedRows vision_embed_rows(const VisionAdapter& adapter, const vision::PatchGrid& grid,
                                                          const vision::ProjectedRows& rows, const session::EmbedDigest& digest) {
    return vision::span_embed_rows(adapter.layout(grid), rows, digest, 1);
}

struct VisionMemory {
    // device, once per rank that runs the tower and the embed stage
    std::uint64_t tower_weights = 0;     // vision::plan: FP16 panels of the tower
    std::uint64_t tower_workspace = 0;   // vision::plan: fixed + MLP chunk buffers at the largest image
    std::uint64_t projector_panels = 0;  // aligner panels, built on the first projection (not in vision::plan)
    std::uint64_t splice_staging = 0;    // DevicePrefill staging of provided rows, at its bound
    std::uint64_t splice_growth = 0;     // old allocation remains live while replacement is allocated
    // host, per image a lane references (EmbedTable rows; the EmbedCache has its own budget)
    std::uint64_t rows_per_image = 0;
    [[nodiscard]] std::uint64_t device_once() const noexcept {
        return tower_weights + tower_workspace + projector_panels + splice_staging + splice_growth;
    }
};

// max_run_rows: rows of the longest prefill run (PrefillOptions::max_rows); the staging never holds more.
[[nodiscard]] inline VisionMemory vision_memory(const VisionAdapter& adapter, std::uint32_t max_run_rows) {
    const VisionConfig& c = adapter.model();
    const unsigned patches = c.max_image_tokens * c.downsample_ratio * c.downsample_ratio;
    const vision::EncoderPlan p = vision::plan(adapter.config(), patches, 1, adapter.device_options());
    VisionMemory m;
    m.tower_weights = p.weight_bytes;
    m.tower_workspace = p.workspace_bytes();
    m.projector_panels = adapter.projector_weight_elements() * 2U;
    m.splice_staging = session::embed_splice_staging_bytes(max_run_rows, c.projection_size) + 256U;
    m.splice_growth = m.splice_staging;  // conservative peak: both allocations can approach the bound
    m.rows_per_image = std::uint64_t{adapter.max_span_rows()} * c.projection_size * 2U;
    return m;
}

// Adds image input to a slot capacity report: the one-time device bytes to the device pool's headroom, and the
// rows of `images_per_slot` largest images to what a slot retains on the host.
inline void add_vision_capacity(serve::CapacityPool& device, serve::CapacityPool& host, const VisionMemory& m,
                                std::uint32_t images_per_slot) {
    device.headroom_bytes += m.device_once();
    host.cost.retained_bytes += m.rows_per_image * images_per_slot;
}

}  // namespace strix::deepseek41
