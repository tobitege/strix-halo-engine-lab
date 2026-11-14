#pragma once

// Device half of the embed-rows splice (embed_splice.hpp), shared by every family's embed stage.

#include "strix/session/embed_splice.hpp"

#include <cstdint>

namespace strix::session {

// One segment: dst[(dst_row + i) * copies + c][0 .. width) = fp32(src[i][0 .. width)) for i < rows, c < copies.
// dst holds FP32 rows of `width` floats (a family with several residual streams per token has `copies` of them
// per row); src holds `rows` rows of `width` 16-bit values in `format`. All pointers are device pointers,
// stream is a hipStream_t. The conversion is exact in both formats. Nothing is launched for rows == 0.
void enqueue_embed_splice(float* dst, std::uint32_t copies, const std::uint16_t* src, std::uint32_t width,
                          EmbedRowFormat format, std::uint32_t dst_row, std::uint32_t rows, void* stream);

} // namespace strix::session
