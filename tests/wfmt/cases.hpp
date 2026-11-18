#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::weight::test {
struct Record {
    std::string arithmetic;
    std::vector<std::uint64_t> shape;
    std::vector<std::byte> bytes;
};
inline std::uint64_t fnv1a64(std::span<const std::byte> bytes) noexcept {
    std::uint64_t hash = 14695981039346656037ull;
    for (auto byte : bytes) {
        hash ^= std::to_integer<std::uint8_t>(byte);
        hash *= 1099511628211ull;
    }
    return hash;
}
inline bool compare_records(const Record& a, const Record& b) noexcept {
    return a.arithmetic == b.arithmetic && a.shape == b.shape &&
           a.bytes.size() == b.bytes.size() && a.bytes == b.bytes;
}
bool compare_case();
bool capture_case(std::string_view output_directory);
struct ByteSection { std::string_view name; std::span<const std::byte> bytes; };
bool write_capture_sections(std::string_view directory,std::string_view name,std::string_view format,
                            std::string_view arithmetic,std::span<const std::uint64_t> shape,
                            std::span<const ByteSection> sections);
bool write_capture_record(std::string_view directory, std::string_view name,
                          std::string_view format, std::span<const std::uint64_t> shape,
                          std::span<const std::byte> source, std::span<const std::byte> output,
                          std::span<const std::byte> sidecars = {});
}
