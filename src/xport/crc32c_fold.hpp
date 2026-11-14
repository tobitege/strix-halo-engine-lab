#pragma once
// CRC-32C (Castagnoli, reflected 0x82F63B78, initial value and final XOR 0xFFFFFFFF: the seam and trailer CRC)
// computed in independent pieces and folded. The same functions run on the host and inside the device kernel
// (src/xport/device_crc32c.hip), so a device digest is the host CRC bit for bit.
//
//   crc(A || B) = crc(A) * x^(8 * |B|)  xor  crc(B)      (polynomial arithmetic modulo the generator)
//
// A buffer is cut into equal spans; each span into `crc32c_fold_lanes` equal lanes. One device thread computes the
// CRC of one lane, the block folds its lanes in a tree, and the host folds the span digests and the short tail.
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_CRC32C_FOLD_FN __host__ __device__ inline
#else
#define STRIX_CRC32C_FOLD_FN constexpr inline
#endif

namespace strix::xport {
inline constexpr std::uint32_t crc32c_polynomial = 0x82f63b78U;
inline constexpr unsigned crc32c_fold_lanes = 256U;          // lanes (device threads) per span
inline constexpr std::size_t crc32c_fold_lane_bytes = 256U;  // bytes per lane: whole 4-byte words
inline constexpr std::size_t crc32c_fold_span_bytes = crc32c_fold_lanes * crc32c_fold_lane_bytes; // 64 KiB
inline constexpr unsigned crc32c_fold_levels = 8U;           // log2(lanes)

// a * b modulo the generator; reflected representation (bit 31 is x^0).
STRIX_CRC32C_FOLD_FN std::uint32_t crc32c_multiply(std::uint32_t a, std::uint32_t b) noexcept {
    std::uint32_t product = 0;
    for (unsigned bit = 0; bit < 32U; ++bit) {
        product ^= b & (0U - (a >> 31U));
        a <<= 1U;
        b = (b >> 1U) ^ (crc32c_polynomial & (0U - (b & 1U)));
    }
    return product;
}
// x^(8 * bytes) modulo the generator: the factor that moves a CRC past `bytes` following bytes.
STRIX_CRC32C_FOLD_FN std::uint32_t crc32c_shift(std::uint64_t bytes) noexcept {
    std::uint32_t result = 0x80000000U, power = 0x00800000U; // x^0, x^8
    while (bytes != 0U) {
        if ((bytes & 1U) != 0U) result = crc32c_multiply(result, power);
        power = crc32c_multiply(power, power);
        bytes >>= 1U;
    }
    return result;
}
// crc(A || B) from crc(A), crc(B) and shift = crc32c_shift(|B|).
STRIX_CRC32C_FOLD_FN std::uint32_t crc32c_append(std::uint32_t head, std::uint32_t tail, std::uint32_t shift) noexcept {
    return crc32c_multiply(shift, head) ^ tail;
}
// One little-endian 32-bit word into the running (not finalized) register, four table lookups.
// table: 4 x 256 entries, crc32c_slice_tables().
template<class Table>
STRIX_CRC32C_FOLD_FN std::uint32_t crc32c_word(const Table& table, std::uint32_t state, std::uint32_t word) noexcept {
    state ^= word;
    return table[768U + (state & 255U)] ^ table[512U + ((state >> 8U) & 255U)] ^ table[256U + ((state >> 16U) & 255U)] ^
           table[state >> 24U];
}

#if !defined(__HIPCC__) && !defined(__CUDACC__)
#define STRIX_CRC32C_HOST_CONSTEXPR constexpr
#else
#define STRIX_CRC32C_HOST_CONSTEXPR
#endif
// Slicing-by-4 tables, [k * 256 + byte].
inline STRIX_CRC32C_HOST_CONSTEXPR std::array<std::uint32_t, 1024> crc32c_slice_tables() noexcept {
    std::array<std::uint32_t, 1024> table{};
    for (std::uint32_t index = 0; index < 256U; ++index) {
        std::uint32_t value = index;
        for (unsigned bit = 0; bit < 8U; ++bit) value = (value >> 1U) ^ (crc32c_polynomial & (0U - (value & 1U)));
        table[index] = value;
    }
    for (std::size_t k = 1; k < 4U; ++k)
        for (std::size_t index = 0; index < 256U; ++index) {
            const std::uint32_t previous = table[(k - 1U) * 256U + index];
            table[k * 256U + index] = (previous >> 8U) ^ table[previous & 255U];
        }
    return table;
}
// Tree factors of one span: level l folds two neighbours of (lane_bytes << l) bytes each.
inline STRIX_CRC32C_HOST_CONSTEXPR std::array<std::uint32_t, crc32c_fold_levels> crc32c_fold_factors() noexcept {
    std::array<std::uint32_t, crc32c_fold_levels> factors{};
    for (unsigned level = 0; level < crc32c_fold_levels; ++level)
        factors[level] = crc32c_shift(static_cast<std::uint64_t>(crc32c_fold_lane_bytes) << level);
    return factors;
}
// Host CRC-32C of a short run (the tail after the last whole span, and the reference of the tests).
inline std::uint32_t crc32c_bytes(std::span<const std::byte> bytes) noexcept {
    static const auto table = crc32c_slice_tables();
    std::uint32_t state = 0xffffffffU;
    for (const auto value : bytes) state = table[(state ^ std::to_integer<std::uint32_t>(value)) & 255U] ^ (state >> 8U);
    return ~state;
}
// The fold of the host side: `digests` are the finalized CRCs of the whole spans in order, `tail` the bytes after
// them (fewer than one span). Returns the CRC-32C of the whole run.
inline std::uint32_t crc32c_fold(std::span<const std::uint32_t> digests, std::span<const std::byte> tail) noexcept {
    static const std::uint32_t span_shift = crc32c_shift(crc32c_fold_span_bytes);
    std::uint32_t crc = 0; // CRC of the empty run
    for (const auto digest : digests) crc = crc32c_append(crc, digest, span_shift);
    if (!tail.empty()) crc = crc32c_append(crc, crc32c_bytes(tail), crc32c_shift(tail.size()));
    return crc;
}
// What the device kernel computes for one span, on the host (tests and the reference of a device self-check).
inline std::uint32_t crc32c_span_digest(std::span<const std::byte> span) noexcept {
    static const auto table = crc32c_slice_tables();
    static const auto factors = crc32c_fold_factors();
    std::array<std::uint32_t, crc32c_fold_lanes> lane{};
    for (unsigned t = 0; t < crc32c_fold_lanes; ++t) {
        std::uint32_t state = 0xffffffffU;
        for (std::size_t w = 0; w < crc32c_fold_lane_bytes / 4U; ++w) {
            const auto* p = span.data() + t * crc32c_fold_lane_bytes + w * 4U;
            const std::uint32_t word = std::to_integer<std::uint32_t>(p[0]) | (std::to_integer<std::uint32_t>(p[1]) << 8U) |
                (std::to_integer<std::uint32_t>(p[2]) << 16U) | (std::to_integer<std::uint32_t>(p[3]) << 24U);
            state = crc32c_word(table, state, word);
        }
        lane[t] = ~state;
    }
    for (unsigned level = 0; level < crc32c_fold_levels; ++level)
        for (unsigned t = 0; t < crc32c_fold_lanes; t += 2U << level)
            lane[t] = crc32c_append(lane[t], lane[t + (1U << level)], factors[level]);
    return lane[0];
}
} // namespace strix::xport
