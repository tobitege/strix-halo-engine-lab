#pragma once
#include "source.hpp"
#include "strix/gguf_reader.hpp"
namespace strix::weight {
enum class Container : std::uint8_t { gguf, safetensors, pack };
struct RawTensor {
  TensorId id{}; std::string name; std::string wire_dtype;
  std::uint32_t wire_type{}; Shape shape{}; Slice payload;
  // Full logical extents as stored by the container (any rank). A tensor whose rank exceeds the
  // WeightView rank (Shape::dims) is indexed opaquely: name, dtype, extents and byte range are
  // kept, shape stays empty, and it can never be admitted as a WeightView. Engines that need such
  // a tensor (for example a Conv3d patch embedding) read it by its byte range.
  std::vector<std::uint64_t> extents;
  bool opaque() const noexcept { return extents.size()>shape.dims.size(); }
};
struct TensorIndex {
  Container container{}; std::string schema;
  std::vector<RawTensor> tensors;
  std::vector<NamedSlice> metadata_blobs;
  std::shared_ptr<const Source> owner;
  Digest256 header_identity{};
  std::vector<std::pair<std::string,std::string>> metadata;
};
struct LegacyConversion { TensorId source; FormatId target; std::string recipe; };
struct AdmissionRequest {
  const TensorIndex* index{}; std::span<const TensorBinding> bindings;
  std::span<const Parameters> tensor_parameters;
};
Status index_gguf(std::shared_ptr<const Source>, TensorIndex&);
Status namespace_index(TensorIndex&, std::string shard);
Status index_gguf(const gguf::FileInfo&, std::shared_ptr<const Source>, TensorIndex&);
Status index_safetensors(std::shared_ptr<const Source>, TensorIndex&);
Status index_pack(std::shared_ptr<const Source>, TensorIndex&);
Status admit(const Registry&, const AdmissionRequest&, WeightMap&);
Status validate_binding(const Registry&, const WeightMapEntry&);
std::string describe_admission(const WeightMap&);
FormatId safetensors_format(std::string_view) noexcept;
Status bind_qwen(const TensorIndex&, WeightMap&);
Status bind_ds41(const TensorIndex&, WeightMap&);
Status bind_minimax(const TensorIndex&, WeightMap&);
Status bind_glm(const TensorIndex&, WeightMap&);
Status bind_gemma4(const TensorIndex&, WeightMap&);
Status bind_h3(const TensorIndex&, WeightMap&);
Status bind_vision(const TensorIndex&, WeightMap&);
}
