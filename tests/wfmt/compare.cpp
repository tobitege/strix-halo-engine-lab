#include "cases.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace strix::weight::test {
bool compare_case() {
    Record a{"legacy@1", {2, 2}, {std::byte{0}, std::byte{1}, std::byte{128}, std::byte{255}}};
    auto b = a;
    if (!compare_records(a, b)) return false;
    const auto receipt = fnv1a64(a.bytes);
    b.bytes[1] ^= std::byte{1};
    // A caller retaining the old hash/metadata must still fail the byte check.
    if (compare_records(a, b)) return false;
    b = a;
    b.bytes.push_back(std::byte{0});
    if (compare_records(a, b)) return false;
    b = a;
    b.arithmetic = "other@1";
    if (compare_records(a, b)) return false;
    return receipt == 5057505546192104323ull &&
           fnv1a64({}) == 14695981039346656037ull;
}
bool capture_case(std::string_view output_directory) {
    // Protocol fixture only: this never claims to capture a production operation.
    const std::array<std::array<std::byte, 4>, 3> data{{
        {std::byte{127}, std::byte{0}, std::byte{255}, std::byte{128}},
        {std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}},
        {std::byte{0}, std::byte{1}, std::byte{128}, std::byte{255}}}};
    const std::array<const char*, 3> names{"activation_codes", "output", "source"};
    std::ostringstream header;
    header << "{\"case\":\"protocol-fixture\",\"arithmetic\":\"legacy-fixture@1\",\"shape\":[4],\"sections\":[";
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (i) header << ',';
        header << "{\"name\":\"" << names[i] << "\",\"bytes\":4,\"fnv1a64\":\""
               << std::hex << std::setw(16) << std::setfill('0') << fnv1a64(data[i]) << "\"}";
    }
    header << "]}";
    const auto text = header.str();
    std::error_code error;
    const std::filesystem::path path(output_directory);
    std::filesystem::create_directories(path, error);
    if (error) return false;
    std::ofstream stream(path / "protocol.wfmt", std::ios::binary);
    stream.write("WFMT\0\1\r\n", 8);
    for (unsigned i = 0; i < 8; ++i) stream.put(static_cast<char>((std::uint64_t(text.size()) >> (8 * i)) & 255));
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    for (const auto& section : data)
        stream.write(reinterpret_cast<const char*>(section.data()), section.size());
    return bool(stream);
}
bool write_capture_record(std::string_view directory,std::string_view name,std::string_view format,
                          std::span<const std::uint64_t> shape,std::span<const std::byte> source,
                          std::span<const std::byte> output,std::span<const std::byte> sidecars) {
    const std::array<ByteSection,3> sections{{{"source",source},{"sidecars",sidecars},{"output",output}}};
    return write_capture_sections(directory,name,format,"legacy-host-decode@1",shape,sections);
}
bool write_capture_sections(std::string_view directory,std::string_view name,std::string_view format,
                            std::string_view arithmetic,std::span<const std::uint64_t> shape,
                            std::span<const ByteSection> sections) {
    std::error_code error;
    const std::filesystem::path path(directory);
    std::filesystem::create_directories(path,error); if(error) return false;
    std::ostringstream header;
    header << "{\"case\":\"" << name << "\",\"format\":\"" << format
           << "\",\"arithmetic\":\"" << arithmetic << "\",\"shape\":[";
    for(std::size_t i=0;i<shape.size();++i) { if(i) header << ','; header << shape[i]; }
    header << "],\"sections\":[";
    for(std::size_t i=0;i<sections.size();++i) {
        if(i) header << ',';
        header << "{\"name\":\"" << sections[i].name << "\",\"bytes\":" << std::dec << sections[i].bytes.size()
               << ",\"fnv1a64\":\"" << std::hex << std::setw(16) << std::setfill('0')
               << fnv1a64(sections[i].bytes) << "\"}";
    }
    header << "]}"; const auto text=header.str();
    std::ofstream stream(path/(std::string(name)+".wfmt"),std::ios::binary);
    stream.write("WFMT\0\1\r\n",8);
    for(unsigned i=0;i<8;++i) stream.put(static_cast<char>((std::uint64_t(text.size())>>(8*i))&255));
    stream.write(text.data(),static_cast<std::streamsize>(text.size()));
    for(const auto& section:sections) stream.write(reinterpret_cast<const char*>(section.bytes.data()),static_cast<std::streamsize>(section.bytes.size()));
    return bool(stream);
}
}
