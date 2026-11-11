// CRC-32C of the pair transport's payloads. Its own translation unit: it has no operating-system
// dependency, so every platform builds it, including those without the transport itself.
#include "strix/dist/communicator.hpp"
#include "dist_crc32c_parallel.hpp"
#include "dist_crc32c_vector.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strix::dist {
namespace {

const std::array<std::uint32_t, 256>& crc_table() {
    static const auto table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256U; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = c & 1U ? 0x82f63b78U ^ (c >> 1U) : c >> 1U;
            t[i] = c;
        }
        return t;
    }();
    return table;
}
#if defined(__x86_64__)
__attribute__((target("sse4.2"))) std::uint32_t crc32c_sse42(std::span<const std::byte> bytes) noexcept {
    std::uint64_t c = 0xffffffffU;
    std::size_t i = 0;
    for (; i + 8U <= bytes.size(); i += 8U) {
        std::uint64_t word;
        std::memcpy(&word, bytes.data() + i, 8U);
        c = __builtin_ia32_crc32di(c, word);
    }
    auto c32 = static_cast<std::uint32_t>(c);
    for (; i < bytes.size(); ++i) c32 = __builtin_ia32_crc32qi(c32, static_cast<unsigned char>(bytes[i]));
    return c32 ^ 0xffffffffU;
}
#endif

} // namespace

std::uint32_t crc32c(std::span<const std::byte> bytes) noexcept {
#if defined(__x86_64__)
    static const bool hardware = __builtin_cpu_supports("sse4.2");
    if (hardware) {
        static const bool parallel = [] { const char* v = std::getenv("STRIX_DIST_CRC_PARALLEL"); return !v || std::string(v) != "0"; }();  // default on
        if (parallel && bytes.size() >= 1024U && detail::crc32c_vector_available()) return detail::crc32c_vector(bytes);
        if (parallel && bytes.size() >= 12288U) return detail::crc32c_parallel(bytes);
        return crc32c_sse42(bytes);
    }
#endif
    const auto& table = crc_table();
    std::uint32_t c = 0xffffffffU;
    for (const auto b : bytes) c = table[(c ^ static_cast<std::uint8_t>(b)) & 0xffU] ^ (c >> 8U);
    return c ^ 0xffffffffU;
}

} // namespace strix::dist
