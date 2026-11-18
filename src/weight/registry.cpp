#include "formats/catalog.hpp"
#include <array>
#include <algorithm>
#include <exception>
namespace strix::weight {
namespace {
struct Wire { std::uint32_t type; FormatId id; };
constexpr Wire wires[] = {
 {0,{"plain.f32",1}}, {1,{"plain.f16",1}}, {2,{"gguf.q4_0",1}}, {3,{"gguf.q4_1",1}},
 {6,{"gguf.q5_0",1}}, {7,{"gguf.q5_1",1}}, {8,{"gguf.q8_0",1}}, {9,{"gguf.q8_1",1}},
 {10,{"gguf.q2_k",1}}, {11,{"gguf.q3_k",1}}, {12,{"gguf.q4_k",1}}, {13,{"gguf.q5_k",1}},
 {14,{"gguf.q6_k",1}}, {15,{"gguf.q8_k",1}}, {16,{"gguf.iq2_xxs",1}}, {17,{"gguf.iq2_xs",1}},
 {18,{"gguf.iq3_xxs",1}}, {19,{"gguf.iq1_s",1}}, {20,{"gguf.iq4_nl",1}}, {21,{"gguf.iq3_s",1}},
 {22,{"gguf.iq2_s",1}}, {23,{"gguf.iq4_xs",1}}, {24,{"typed.i8",1}}, {25,{"typed.i16",1}},
 {26,{"typed.i32",1}}, {27,{"typed.i64",1}}, {28,{"typed.f64",1}}, {29,{"gguf.iq1_m",1}},
 {30,{"plain.bf16",1}}, {34,{"gguf.tq1_0",1}}, {35,{"gguf.tq2_0",1}}, {39,{"gguf.mxfp4",1}},
 {42,{"gguf.q2_0.strix",1}}, {43,{"gguf.nvfp4_tile.strix",1}}, {200,{"gguf.iu4_g64.strix",1}}
};
}
FormatId gguf_format(std::uint32_t type) noexcept {
 for (const auto& wire : wires) if (wire.type == type) return wire.id;
 return {{}, 0};
}
std::uint32_t gguf_type(FormatId id) noexcept {
 for (const auto& wire : wires) if (wire.id == id) return wire.type;
 return UINT32_MAX;
}
const Registry& builtin_registry() {
 // Explicit construction, no dynamic registration or cross-TU initialisers.
 static const std::vector<FormatDescriptor> formats = [] {
   std::vector<FormatDescriptor> result;
   for (auto part : {detail::plain_formats(), detail::ggml_formats(), detail::ds41_formats(),
        detail::packed_formats(), detail::fp8_formats(), detail::exl3_formats(),
        detail::channel_formats(), detail::nvfp4_formats()})
     result.insert(result.end(), part.begin(), part.end());
   return result;
 }();
 static const Registry registry{formats, legacy_arithmetic()};
 return registry;
}
const FormatDescriptor* find_format(const Registry& registry, FormatId id) noexcept {
 for (const auto& format : registry.formats) if (format.id == id) return &format;
 return nullptr;
}
Status format_geometry(FormatId id, const Parameters& parameters, Geometry& geometry) {
 const auto* format = find_format(builtin_registry(), id);
 if (!format) return {Error::unknown_format, "unregistered storage identity"};
 return format->geometry(parameters, geometry);
}
Status validate_metadata(const WeightView& weight) {
 try{return detail::validate_metadata(weight);}catch(const std::exception& error){return {Error::malformed,error.what()};}
}

Status validate_weight(const WeightView& weight) {
 const auto* format = find_format(builtin_registry(), weight.storage);
 if (!format) return {Error::unknown_format, "unregistered storage identity"};
 try { return format->validate(weight); }
 catch(const std::exception& error) { return {Error::malformed,error.what()}; }
}
}
