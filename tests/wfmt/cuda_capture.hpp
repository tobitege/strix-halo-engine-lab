#pragma once
// Test-only portable WFMT records. No production allocation, decode or launch.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <span>
#include <sstream>
#include <string>
#include <string_view>

namespace strix::weight::test {
struct CudaCaptureSection { const char* name; std::span<const std::byte> bytes; };
inline bool write_cuda_capture(std::string_view directory, std::string_view name,
                               std::string_view arithmetic, std::span<const std::uint64_t> shape,
                               std::initializer_list<CudaCaptureSection> sections) {
    std::error_code error;
    const std::filesystem::path path(directory);
    std::filesystem::create_directories(path, error);
    if (error) return false;
    std::ostringstream header;
    header << "{\"case\":\"" << name << "\",\"format\":\"gguf.q8_0@1\",\"arithmetic\":\""
           << arithmetic << "\",\"shape\":[";
    for (std::size_t i = 0; i < shape.size(); ++i) { if (i) header << ','; header << shape[i]; }
    header << "],\"sections\":[";
    bool first = true;
    for (const auto& section : sections) {
        if (!first) header << ',';
        first = false;
        std::uint64_t hash = 14695981039346656037ULL;
        for (const auto byte : section.bytes) hash = (hash ^ std::to_integer<std::uint8_t>(byte)) * 1099511628211ULL;
        header << "{\"name\":\"" << section.name << "\",\"bytes\":" << std::dec << section.bytes.size()
               << ",\"fnv1a64\":\"" << std::hex << std::setw(16) << std::setfill('0') << hash << "\"}";
    }
    header << "]}";
    const auto text = header.str();
    std::ofstream stream(path / (std::string(name) + ".wfmt"), std::ios::binary);
    stream.write("WFMT\0\1\r\n", 8);
    for (unsigned i = 0; i < 8; ++i) stream.put(static_cast<char>((std::uint64_t(text.size()) >> (8 * i)) & 255));
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    for (const auto& section : sections)
        stream.write(reinterpret_cast<const char*>(section.bytes.data()), static_cast<std::streamsize>(section.bytes.size()));
    return bool(stream);
}
}

