#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::weight {
using Byte = std::byte;
using TensorId = std::uint64_t;
// All string_views and spans in this API are borrowed. Their backing storage must
// outlive every descriptor, registry, view and copy using it without relocation.
// String storage stays immutable; spans permit mutation only as their types allow.
// Dynamic registries must retain their own stable storage; IDs do not intern names.
struct Id { std::string_view name; std::uint32_t revision; };
using FormatId = Id;
using ArithmeticId = Id;
using ImageId = Id;
struct Digest256 { std::array<std::uint8_t, 32> bytes; };
enum class Backend : std::uint8_t { hip, cuda, cpu, intel };
enum class Exactness : std::uint8_t { legacy_bitwise, operand_exact, fast };
enum class Strategy : std::uint8_t { native, staged, converted, onfly };
enum class Scalar : std::uint8_t {
  f32, f16, bf16, s8, u8, i16, u16, i32, u32, i64, u64, bool8, f64,
  e4m3fn, e5m2, e8m0, e2m1
};
enum class Role : std::uint8_t {
  embedding, head, attention_q, attention_k, attention_v, attention_o,
  dense_gate, dense_up, dense_down, expert_gate, expert_up, expert_down,
  router, norm, bias, lookup_row, auxiliary
};
enum class Error : std::uint8_t {
  ok, unknown_format, malformed, unsupported_role, unsupported_shape,
  unsupported_backend, unsupported_arithmetic, unavailable_isa,
  memory_budget, unqualified, stale_source, device_failure
};
struct Status { Error code; std::string detail; explicit operator bool() const; };
struct Shape { std::array<std::uint64_t, 4> dims; std::uint8_t rank; };
struct Rational { std::uint64_t numerator, denominator; };
struct Geometry {
  std::uint32_t block_elements, block_bytes, group_elements;
  std::uint32_t tile_k, tile_n, row_alignment, payload_alignment;
  bool row_addressable; // false for ring/tile transports
};
struct ByteRange { std::uint64_t offset, bytes; };
struct Source; // read-only positioned IO/mapping; lifetime owned below
struct Slice { std::shared_ptr<const Source> source; ByteRange range; };
struct NamedSlice { std::string name; Scalar scalar; Shape shape; Slice data; };
struct Parameters {
  std::uint32_t bits, group_size;
  std::string scale_encoding, codebook_encoding, transform;
  Digest256 semantic_metadata;
};
struct WeightView {
  TensorId tensor; FormatId storage; Shape shape; Geometry geometry;
  Slice payload; std::vector<NamedSlice> sidecars; Parameters parameters;
  std::array<std::uint64_t, 4> byte_strides;
  bool has_byte_strides; // false for layouts requiring descriptor addressing
};
struct TensorBinding {
  TensorId tensor; Role role; Shape logical_shape;
  std::uint32_t expert_axis; // UINT32_MAX means not expert indexed
  std::vector<FormatId> admitted_formats;
  std::uint64_t admitted_categories; // zero = explicit allowlist only
};
struct WeightMapEntry { TensorBinding binding; WeightView weight; };
using WeightMap = std::vector<WeightMapEntry>;
struct DecodeOps {
  Rational bit_extracts, table_lookups, sign_applications, packs;
  Rational int_to_float, float_multiplies, rounds, integer_dots;
}; // per stored weight; integer_dots are PRODUCT work, never double-count
struct CostDescriptor {
  Rational packed_bytes_per_weight, image_bytes_per_weight;
  std::uint64_t fixed_metadata_bytes;
  DecodeOps decode;
  Scalar operand;
  std::uint32_t reuse_rows; // capability bound; effective reuse from launch
};
struct DecodeRequest {
  std::uint64_t first_element, count;
  ArithmeticId arithmetic; Scalar output;
};
struct DecodeBuffer { std::span<Byte> bytes; std::size_t written; };
struct SignedGroup32 {
  std::array<std::int8_t,32> codes;
  std::array<std::uint8_t,32> scale_index;
  std::array<float,32> scales;
  std::array<float,32> offsets;
  std::uint32_t scale_count;
}; // representation only; product ordering belongs to ArithmeticId
struct FormatDescriptor {
  FormatId id; std::string_view wire_schema;
  std::uint64_t category, applicable_roles;
  bool semantic_decode_supported;
  Status (*validate)(const WeightView&);
  Status (*geometry)(const Parameters&, Geometry&);
  Status (*decode_reference)(const WeightView&, const DecodeRequest&,
                             DecodeBuffer&);
  Status (*signed_group_reference)(const WeightView&, std::uint64_t,
                                   SignedGroup32&);
  CostDescriptor (*cost)(const WeightView&, ImageId);
};
struct ArithmeticDescriptor {
  ArithmeticId id; Exactness class_;
  std::string_view activation, weight_rounding, multiply, accumulation;
  std::string_view partition, reduction, input_transform, output_transform;
  bool permits_fma, flushes_subnormals;
};
struct Registry {
  std::span<const FormatDescriptor> formats;
  std::span<const ArithmeticDescriptor> arithmetic;
};
std::span<const ArithmeticDescriptor> legacy_arithmetic() noexcept;
std::string serialize_id(Id);
bool operator==(Id, Id) noexcept;
const Registry& builtin_registry();
const FormatDescriptor* find_format(const Registry&, FormatId) noexcept;
Status validate_map(const Registry&, const WeightMap&);
Status decode_reference(const Registry&, const WeightView&,
                        const DecodeRequest&, DecodeBuffer&);
} // namespace strix::weight
