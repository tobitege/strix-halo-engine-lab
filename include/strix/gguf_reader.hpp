#pragma once

#include <cstdint>
#include <filesystem>
#include <istream>
#include <optional>
#include <string>
#include <vector>

namespace strix::gguf {

enum class ValueType : std::uint32_t {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

struct Limits {
    std::uint64_t max_metadata_entries = 1'000'000;
    std::uint64_t max_tensors = 1'000'000;
    std::uint64_t max_string_bytes = 64ULL * 1024ULL * 1024ULL;
    std::uint64_t max_total_array_elements = 100'000'000;
    std::uint32_t max_array_depth = 8;
    std::uint32_t max_tensor_dimensions = 4;
    // Top-level numeric arrays up to this length keep their values (0: summaries only, the default).
    std::uint64_t max_retained_array_elements = 0;
    // Metadata keys are lower-case snake case separated by dots. Some converted files prefix their keys with a
    // hyphenated architecture name; a caller that admits such a file opts in here.
    bool allow_hyphen_in_keys = false;
};

struct MetadataEntry {
    std::string key;
    ValueType type{};
    std::optional<std::uint64_t> unsigned_value;
    std::optional<std::int64_t> signed_value;
    std::optional<double> floating_value;
    std::optional<bool> boolean_value;
    std::optional<std::string> string_value;
    std::optional<ValueType> array_element_type;
    std::optional<std::uint64_t> array_length;
    // Retained top-level numeric array values (see Limits::max_retained_array_elements); exactly one of
    // these is filled, by element signedness: unsigned/boolean, signed, or float32/float64.
    std::vector<std::uint64_t> unsigned_array;
    std::vector<std::int64_t> signed_array;
    std::vector<double> floating_array;
};

struct TensorTypeGeometry {
    std::uint64_t block_elements = 0;
    std::uint64_t block_bytes = 0;
};

[[nodiscard]] std::optional<TensorTypeGeometry> tensor_type_geometry(
    std::uint32_t ggml_type) noexcept;

struct TensorInfo {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t ggml_type = 0;
    std::uint64_t relative_offset = 0;
    std::uint64_t absolute_offset = 0;
    std::uint64_t byte_size = 0;
    std::uint64_t absolute_end = 0;
};

struct FileInfo {
    std::uint32_t version = 0;
    std::uint64_t file_size = 0;
    std::uint64_t alignment = 32;
    std::uint64_t data_offset = 0;
    std::vector<MetadataEntry> metadata;
    std::vector<TensorInfo> tensors;

    [[nodiscard]] const MetadataEntry* find_metadata(const std::string& key) const;
};

[[nodiscard]] FileInfo read(std::istream& input,
                            std::uint64_t file_size,
                            const Limits& limits = {});
[[nodiscard]] FileInfo read_file(const std::filesystem::path& path,
                                 const Limits& limits = {});

}  // namespace strix::gguf
