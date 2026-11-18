#pragma once
#include "strix/weight/reference.hpp"
#include "strix/weight/source.hpp"
#include <limits>
namespace strix::weight::detail {
// Explicit role masks: linear projections include router, lookup and auxiliary are separate.
inline constexpr std::uint64_t all_linear_roles = (1ull << (unsigned(Role::bias) + 1)) - 1;
inline constexpr std::uint64_t auxiliary_roles = (1ull << unsigned(Role::norm)) | (1ull << unsigned(Role::bias)) | (1ull << unsigned(Role::auxiliary));
Status validate(const WeightView&);
Status validate_metadata(const WeightView&);
Status decode(const WeightView&, const DecodeRequest&, DecodeBuffer&);
Status group(const WeightView&, std::uint64_t, SignedGroup32&);
Status unavailable(const WeightView&, const DecodeRequest&, DecodeBuffer&);
Status no_group(const WeightView&, std::uint64_t, SignedGroup32&);
CostDescriptor cost(const WeightView&, ImageId);
Status read(const Slice&, std::vector<Byte>&);
Status elements(const Shape&, std::uint64_t&);
const NamedSlice* sidecar(const WeightView&, std::string_view);
template <std::uint32_t Elements, std::uint32_t Bytes, std::uint32_t Group = 32, std::uint32_t Row = Elements>
Status fixed(const Parameters&, Geometry& out) {
    out = {Elements, Bytes, Group, 1, 1, Row, 1, true};
    return {Error::ok, {}};
}
inline FormatDescriptor descriptor(FormatId id, std::string_view wire,
        std::uint64_t category, bool semantic, Status (*geometry)(const Parameters&, Geometry&),
        bool signed_groups = false, std::uint64_t roles = all_linear_roles) {
    return {id, wire, category, roles, semantic, validate, geometry,
            semantic ? decode : unavailable, signed_groups ? group : no_group, cost};
}
std::span<const FormatDescriptor> plain_formats();
std::span<const FormatDescriptor> ggml_formats();
std::span<const FormatDescriptor> ds41_formats();
std::span<const FormatDescriptor> packed_formats();
std::span<const FormatDescriptor> fp8_formats();
std::span<const FormatDescriptor> exl3_formats();
std::span<const FormatDescriptor> channel_formats();
std::span<const FormatDescriptor> nvfp4_formats();
Status decode_plain(const WeightView&, const std::vector<Byte>&, std::vector<float>&);
Status decode_ggml(const WeightView&, const std::vector<Byte>&, std::vector<float>&);
Status decode_ds41(const WeightView&, const std::vector<Byte>&, std::vector<float>&);
Status decode_packed(const WeightView&, const std::vector<Byte>&, std::vector<float>&);
Status decode_fp8(const WeightView&, const std::vector<Byte>&, std::vector<float>&);
Status decode_exl3(const WeightView&, const std::vector<Byte>&, std::vector<float>&);
Status decode_channel(const WeightView&, const std::vector<Byte>&, std::vector<float>&);
Status decode_nvfp4(const WeightView&, const std::vector<Byte>&, std::vector<float>&);
}
