#pragma once

// Test-only observations of existing transfers. Include after HIP/test headers.
// Production builds do not intercept any API and contain no capture state.
#if defined(STRIX_WFMT_CAPTURE) && STRIX_WFMT_CAPTURE
#include <hip/hip_runtime.h>
#include "strix/env_flags.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace strix::weight::capture {
inline const std::string& directory() {
    static const std::string value = [] {
        const char* path = strix::envcfg::text("STRIX_WFMT_CAPTURE_DIR");
        return path ? std::string(path) : std::string();
    }();
    return value;
}
[[noreturn]] inline void fail() {
    std::fputs("weight transfer capture failed\n", stderr);
    std::exit(EXIT_FAILURE);
}
struct Context {
    std::string case_name;
    std::string arithmetic = "legacy-hip-transfer@1";
    std::string format = "hip.transfer@1";
    std::vector<std::uint64_t> shape;
};
inline Context& context() { static Context value; return value; }
inline std::mutex& capture_mutex() { static std::mutex value; return value; }
inline void set_context(const char* case_name, const char* arithmetic,
                        std::initializer_list<std::uint64_t> shape,
                        const char* format = "hip.transfer@1") {
    const std::lock_guard<std::mutex> lock(capture_mutex());
    if (!case_name || !*case_name || !arithmetic || !*arithmetic || !format || !*format) fail();
    context() = {case_name, arithmetic, format, shape};
}
inline std::string json_string(const std::string& value) {
    std::ostringstream output;
    output << '"';
    for (const char raw : value) {
        const auto c = static_cast<unsigned char>(raw);
        if (c == '"' || c == '\\') output << '\\' << static_cast<char>(c);
        else if (c < 32) output << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << unsigned(c);
        else output << static_cast<char>(c);
    }
    output << '"';
    return output.str();
}
inline void emit(const char* section, const void* data, std::size_t bytes) {
    if (directory().empty() || bytes == 0) return;
    static std::uint64_t ordinal = 0;
    const std::lock_guard<std::mutex> lock(capture_mutex());
    try {
        std::error_code error;
        std::filesystem::create_directories(directory(), error);
        if (error || !data) fail();
        const auto* payload = static_cast<const unsigned char*>(data);
        std::uint64_t hash = UINT64_C(14695981039346656037);
        for (std::size_t i = 0; i < bytes; ++i)
            hash = (hash ^ payload[i]) * UINT64_C(1099511628211);
        std::ostringstream name;
        name << "transfer-" << std::setw(8) << std::setfill('0') << ordinal++;
        std::ostringstream header;
        const auto& current = context();
        header << "{\"case\":" << json_string(current.case_name.empty() ? name.str() : current.case_name + "." + name.str())
               << ",\"format\":" << json_string(current.format)
               << ",\"arithmetic\":" << json_string(current.arithmetic) << ",\"shape\":[";
        if (current.shape.empty()) header << bytes;
        else for (std::size_t i = 0; i < current.shape.size(); ++i) {
            if (i) header << ',';
            header << current.shape[i];
        }
        header << "],\"sections\":[{\"name\":\"" << section
               << "\",\"bytes\":" << bytes << ",\"fnv1a64\":\""
               << std::hex << std::setw(16) << std::setfill('0') << hash << "\"}]}";
        const std::string text = header.str();
        const auto path = std::filesystem::path(directory()) / (name.str() + ".wfmt");
        if (std::filesystem::exists(path)) fail();
        std::ofstream output(path, std::ios::binary);
        output.write("WFMT\0\1\r\n", 8);
        const auto size = static_cast<std::uint64_t>(text.size());
        for (unsigned i = 0; i < 8; ++i)
            output.put(static_cast<char>((size >> (i * 8)) & 255));
        output.write(text.data(), static_cast<std::streamsize>(text.size()));
        output.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        output.close();
        if (!output) fail();
    } catch (...) { fail(); }
}
inline hipError_t memcpy(void* destination, const void* source, std::size_t bytes,
                         hipMemcpyKind kind) {
    const hipError_t result = ::hipMemcpy(destination, source, bytes, kind);
    if (result == hipSuccess) {
        if (kind == hipMemcpyHostToDevice) emit("source", source, bytes);
        else if (kind == hipMemcpyDeviceToHost) emit("output", destination, bytes);
    }
    return result;
}
} // namespace strix::weight::capture

inline hipError_t strix_wfmt_capture_hipMemcpy(void* destination, const void* source,
                                             std::size_t bytes, hipMemcpyKind kind) {
    return ::strix::weight::capture::memcpy(destination, source, bytes, kind);
}
#define hipMemcpy strix_wfmt_capture_hipMemcpy
#endif
