#include "strix/dist/communicator.hpp"
#include "../src/dist_crc32c_parallel.hpp"
#include "../src/dist_crc32c_vector.hpp"

#include <iostream>
#include <random>
#include <vector>

namespace {
std::uint32_t reference(std::span<const std::byte> bytes) {
    std::uint32_t c = 0xffffffffU;
    for (const auto b : bytes) {
        c ^= static_cast<unsigned char>(b);
        for (unsigned k = 0; k < 8U; ++k) c = (c >> 1U) ^ ((c & 1U) ? 0x82f63b78U : 0U);
    }
    return c ^ 0xffffffffU;
}
}
int main() {
#if defined(__x86_64__)
    if (!__builtin_cpu_supports("sse4.2")) return 0;
    std::mt19937 rng(1937);
    std::vector<std::byte> data((192U << 20U) + 128U); // reproduce the review corpus RNG state exactly
    for (auto& b : data) b = static_cast<std::byte>(rng());
    std::size_t checked = 0;
    const auto test = [&](std::size_t offset, std::size_t n) {
        const auto bytes = std::span<const std::byte>(data).subspan(offset, n);
        const auto expected = reference(bytes);
        if (strix::dist::crc32c(bytes) != expected || strix::dist::detail::crc32c_parallel(bytes) != expected ||
            (strix::dist::detail::crc32c_vector_available() && strix::dist::detail::crc32c_vector(bytes) != expected)) {
            std::cerr << "CRC differs at offset " << offset << " size " << n << '\n';
            return false;
        }
        ++checked;
        return true;
    };
    for (std::size_t o = 0; o < 32U; ++o) {
        for (std::size_t n = 0; n <= 128U; ++n) if (!test(o, n)) return 1;
        for (const std::size_t edge : {4096U, 12288U, 24576U, 122880U, 163840U})
            for (std::size_t delta = 0; delta < 17U; ++delta)
                if (!test(o, edge - 8U + delta)) return 1;
    }
    for (const std::size_t n : {1U << 20U, 13U << 20U, 21U << 20U})
        if (!test(1U, n)) return 1;
    const auto base_cases = checked;
    for (std::size_t o = 0; o < 32U; ++o)
        for (const std::size_t n : {0U, 1U, 7U, 8U, 9U, 15U, 16U, 127U, 255U, 256U, 257U, 511U, 512U, 513U,
                                  1024U, 12287U, 12288U, 12289U, 16383U, 16384U, 16385U, 20480U})
            if (!test(o, n)) return 1;
    for (unsigned i = 0; i < 100U; ++i) {
        const std::size_t n = rng() % 100000U, o = rng() % 64U;
        if (!test(o, n)) return 1;
    }
    for (const std::size_t n : {1U << 20U, 13U << 20U, 21U << 20U})
        if (!test(1U, n)) return 1;
    const char* known = "123456789";
    if (strix::dist::detail::crc32c_parallel({reinterpret_cast<const std::byte*>(known), 9U}) != 0xe3069283U) return 1;
    if (strix::dist::detail::crc32c_vector_available() &&
        strix::dist::detail::crc32c_vector({reinterpret_cast<const std::byte*>(known), 9U}) != 0xe3069283U) return 1;
    std::cout << "CRC32C exact: " << base_cases << " + " << checked - base_cases
              << " review cases; vector=" << strix::dist::detail::crc32c_vector_available() << '\n';
#endif
}
