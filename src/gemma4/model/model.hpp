#pragma once
#include "strix/weight/admission.hpp"

#include "gemma4/contracts/g4_types.hpp"
#include "strix/serve/json.hpp"
#include "strix/weight_archive.hpp"
#include <filesystem>
#include <map>

namespace strix::gemma4 {

using MappedFile = weights::MappedFile;
using SafeTensor = weights::SafeTensor;
using SafeFile = weights::SafeFile;
[[nodiscard]] SafeFile read_safetensors(const std::filesystem::path& path);
[[nodiscard]] ModelConfig parse_text_config(const std::filesystem::path& path);
[[nodiscard]] std::map<std::string, std::vector<std::uint64_t>> required_tensors(const ModelConfig& config);
[[nodiscard]] bool frozen_compatibility_tensor(const std::string& name);
[[nodiscard]] ModelManifest load_safetensors_model(const std::filesystem::path& directory);
// GGUF norms are already direct multipliers; no offset adjustment or Q/K permutation.
[[nodiscard]] ModelManifest load_gguf_model(const std::filesystem::path& path);
[[nodiscard]] std::string gguf_tensor_name(const std::string& name);
[[nodiscard]] const TensorRecord& tensor(const ModelManifest& model, const std::string& name);
[[nodiscard]] float weight_at(const LinearView& view, std::uint32_t row, std::uint32_t column);
[[nodiscard]] std::size_t packed_row_bytes(WeightFormat format, std::uint32_t inputs);
void validate_finite(const LinearView& view);
[[nodiscard]] std::string hash_bytes(const void* bytes, std::size_t size);
[[nodiscard]] serve::Json read_json(const std::filesystem::path& path);

namespace model_detail {
[[nodiscard]] const serve::Json& require(const serve::Json& value, const char* key);
[[nodiscard]] std::uint64_t integer(const serve::Json& value);
[[nodiscard]] LinearView make_view(const std::uint8_t* data, WeightFormat format, const std::vector<std::uint64_t>& shape);
void validate_manifest(ModelManifest& model);
}
} // namespace strix::gemma4
