#pragma once
#include "types.hpp"
namespace strix::weight {
inline constexpr std::uint64_t plain_linear = 1ull << 0;
inline constexpr std::uint64_t quantized_linear = 1ull << 1;
inline constexpr std::uint64_t transformed_linear = 1ull << 2;
inline constexpr std::uint64_t row_lookup = 1ull << 3;
inline constexpr std::uint64_t typed_auxiliary = 1ull << 4;
// Empty name means the reader does not recognise the wire type.
FormatId gguf_format(std::uint32_t) noexcept;
std::uint32_t gguf_type(FormatId) noexcept; // UINT32_MAX if not a GGUF wire alias
Status format_geometry(FormatId, const Parameters&, Geometry&);
Status validate_weight(const WeightView&);
Status validate_metadata(const WeightView&);
Status validate_binding(const Registry&, const WeightMapEntry&);
}
