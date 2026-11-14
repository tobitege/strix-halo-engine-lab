#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix::qwen_pp {
// Wire v2 rows: BF16 HC streams, BF16 attention input (HC output), four FP32 coefficients.
// A folded next-layer HC is already executed: its output must travel with its skip marker.
struct Seam {
    static constexpr std::uint64_t magic = 0x5050325345414d02ULL;
    static constexpr std::size_t hc_bytes = 4U * 2560U * sizeof(std::uint16_t);
    static constexpr std::size_t attention_bytes = 2560U * sizeof(std::uint16_t);
    static constexpr std::size_t residual_bytes = attention_bytes; // legacy size alias, not hidden(1)
    static constexpr std::size_t coefficient_bytes = 4U * sizeof(float);
    static constexpr std::size_t row_bytes = hc_bytes + residual_bytes + coefficient_bytes;
    std::vector<std::byte> bytes;
    explicit Seam(std::uint32_t rows) : bytes(static_cast<std::size_t>(rows) * row_bytes) {}
    std::byte* row(std::uint32_t index) { return bytes.data() + static_cast<std::size_t>(index) * row_bytes; }
};
static_assert(Seam::row_bytes == 25616U);
static_assert(sizeof(float) == 4U);

// Layers [begin, end) of one stage. A half end ("0:22.5") also runs the attention sub-block of layer `end`;
// a half begin ("22.5:48") starts at the feed-forward half of layer `begin`.
struct LayerRange {
    std::uint32_t begin = 0;
    std::uint32_t end = 48;
    bool enabled = false;
    bool begin_half = false;
    bool end_half = false;
};
inline LayerRange parse_range(const char* value, std::uint32_t layers) {
    if (!value) return {0, layers, false};
    const std::string_view text(value);
    const auto colon = text.find(':');
    LayerRange range{0, 0, true};
    if (colon == std::string_view::npos || colon == 0 || colon + 1U == text.size())
        throw std::invalid_argument("STRIX_QWEN_PF_LAYER_RANGE requires begin:end");
    const auto half = [](std::string_view& part) {
        if (!part.ends_with(".5")) return false;
        part.remove_suffix(2U);
        return true;
    };
    auto begin_text = text.substr(0, colon), end_text = text.substr(colon + 1U);
    range.begin_half = half(begin_text);
    range.end_half = half(end_text);
    const auto first = std::from_chars(begin_text.data(), begin_text.data() + begin_text.size(), range.begin);
    const auto last = std::from_chars(end_text.data(), end_text.data() + end_text.size(), range.end);
    if (first.ec != std::errc{} || first.ptr != begin_text.data() + begin_text.size() || last.ec != std::errc{} ||
        last.ptr != end_text.data() + end_text.size() || (range.begin_half && range.begin == 0U) ||
        2ULL * range.begin + (range.begin_half ? 1U : 0U) >= 2ULL * range.end + (range.end_half ? 1U : 0U) ||
        range.end + (range.end_half ? 1U : 0U) > layers)
        throw std::invalid_argument("STRIX_QWEN_PF_LAYER_RANGE is outside the model");
    return range;
}
} // namespace strix::qwen_pp
