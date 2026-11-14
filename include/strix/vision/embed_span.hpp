#pragma once

// One encoded image as the language model takes it: the projected rows of encode_cached (span order) and the
// adapter's layout() become one session::EmbedRows, ready for EmbedInterner::intern. Shared by every family.

#include "strix/session/embed_rows.hpp"
#include "strix/vision/adapter.hpp"
#include "strix/vision/encoder.hpp"

#include <cstdint>
#include <span>
#include <stdexcept>

namespace strix::vision {

// axes: position axes the family reads per row (0: its position is the prompt index and layout.lm_pos is not
// stored; 1..3: the first `axes` entries of every lm_pos). deep_layers: the LM layer of each DeepStack tap of
// `rows`, in tap order. Throws std::invalid_argument when the layout and the rows disagree.
[[nodiscard]] inline session::EmbedRows span_embed_rows(const SpanLayout& layout, const ProjectedRows& rows,
                                                        const session::EmbedDigest& digest, unsigned axes = 0,
                                                        std::span<const std::uint32_t> deep_layers = {}) {
    const std::size_t n = layout.role.size();
    if (n == 0 || n != rows.rows || rows.width == 0 || rows.bf16.size() != n * rows.width || axes > 3 ||
        (axes != 0 && layout.lm_pos.size() != n) || rows.deep.size() != deep_layers.size() || n > session::kEmbedMaxRows)
        throw std::invalid_argument("vision span: layout and rows disagree");
    session::EmbedRows out;
    out.digest = digest;
    out.rows = static_cast<std::uint32_t>(n);
    out.width = rows.width;
    out.axes = axes;
    out.real_token_id = layout.real_token_id;
    out.data = rows.bf16;
    out.role.resize(n);
    for (std::size_t r = 0; r < n; ++r) {
        switch (layout.role[r]) {
        case SpanLayout::Role::image: out.role[r] = session::EmbedRole::content; break;
        case SpanLayout::Role::start: out.role[r] = session::EmbedRole::begin; break;
        case SpanLayout::Role::newline: out.role[r] = session::EmbedRole::newline; break;
        case SpanLayout::Role::end: out.role[r] = session::EmbedRole::end; break;
        }
        for (unsigned a = 0; a < axes; ++a) out.lm_pos.push_back(static_cast<std::int32_t>(layout.lm_pos[r][a]));
    }
    for (std::size_t d = 0; d < rows.deep.size(); ++d) {
        if (rows.deep[d].size() != rows.bf16.size()) throw std::invalid_argument("vision span: deep rows shape");
        out.deep.push_back({deep_layers[d], rows.deep[d]});
    }
    return out;
}

}  // namespace strix::vision
