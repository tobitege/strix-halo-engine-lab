// EXL3 trellis packed-weight format: the definition shared by every reader (host emulators, packers, device codec).
// Pure C++ (no HIP types), so host-only translation units can include it; device readers are in trellis_codec.hpp.
// Format and codebook semantics follow the adopted strix-trellis library (Apache License 2.0, see NOTICE.md); the
// EXL3 spelling identifies the exllamav3 format basis.
//
//   - a matrix of I inputs x O outputs is stored as (I / 16) x (O / 16) tiles, input-tile major: tile (tk, tn) starts
//     at word (tk * (O / 16) + tn) * 8 * Bits; a tile is a ring of 256 * Bits bits = 8 * Bits little-endian u32 words,
//     bits traversed most-significant first;
//   - state(p) = the 16 ring bits ending at bit (p + 1) * Bits (consecutive states overlap by 16 - Bits bits);
//   - coefficient (input k, output n) of a tile is state(position(k, n)); its value is the codebook value of that
//     state. Nothing is carried between tiles: random access is one tile (32 * Bits bytes), and inside a tile any
//     state is addressable from at most two words.
#pragma once
#include <cstddef>
#include <cstdint>

#if defined(__HIP__)
#include <hip/hip_runtime.h>
#define STRIX_TRELLIS_HD __host__ __device__
#else
#define STRIX_TRELLIS_HD
#endif

namespace strix::core::trellis {

struct Format {
    static constexpr int kTile = 16;             // inputs and outputs per tile
    static constexpr int kStates = 256;          // coefficients per tile
    static constexpr int kStateBits = 16;        // ring bits per state
    static constexpr int kDecoderStateBytes = 0; // nothing is carried between tiles
    static constexpr int words(int bits) { return 8 * bits; }
    static constexpr int tile_bytes(int bits) { return 32 * bits; }
    // Ring position of coefficient (input k, output n) inside its tile, k and n in 0..15.
    static constexpr unsigned position(unsigned k, unsigned n) {
        return ((n & 7U) << 5U) | ((k & 6U) << 2U) | (k & 1U) | ((k & 8U) >> 2U) | ((n & 8U) >> 1U);
    }
    // Word offset of tile (tile_k, tile_n) in the input-tile-major stream.
    static constexpr std::size_t tile(std::size_t tile_k, std::size_t tile_n, std::size_t tiles_n, int bits) {
        return (tile_k * tiles_n + tile_n) * static_cast<std::size_t>(words(bits));
    }
    // Reference state extraction: 16 ring bits ending at (position + 1) * bits. word(i) returns ring word i.
    template <class Word> STRIX_TRELLIS_HD static inline std::uint32_t state_of(const Word& word, int bits, unsigned position) {
        const unsigned count = static_cast<unsigned>(words(bits));
        const unsigned end = (position + 1U) * static_cast<unsigned>(bits) + 256U * static_cast<unsigned>(bits), start = end - 16U;
        const std::uint64_t pair = (static_cast<std::uint64_t>(word((start / 32U) % count)) << 32U) | word(((end - 1U) / 32U) % count);
        return static_cast<std::uint32_t>(pair >> ((32U - (end & 31U)) & 31U)) & 65535U;
    }
    STRIX_TRELLIS_HD static inline std::uint32_t state(const std::uint32_t* ring, int bits, unsigned position) {
        return state_of([ring](unsigned i) { return ring[i]; }, bits, position);
    }
    // Alignment- and byte-order-neutral form over the tile's bytes.
    STRIX_TRELLIS_HD static inline std::uint32_t state_bytes(const std::uint8_t* ring, int bits, unsigned position) {
        return state_of([ring](unsigned i) {
            const std::uint8_t* p = ring + 4U * i;
            return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8U) |
                   (static_cast<std::uint32_t>(p[2]) << 16U) | (static_cast<std::uint32_t>(p[3]) << 24U);
        }, bits, position);
    }
};

// mul1 codebook constants: value(s) = RN16((1024 + sum(s)) * F16(kScale) + F16(kBias)), state_sum = byte sum of s * M.
struct Mul1Constants {
    static constexpr int id = 2;
    static constexpr std::uint32_t kMultiplier = 0x83dcd12dU;
    static constexpr std::uint16_t kScale = 0x1eee, kBias = 0xc931;
    static constexpr unsigned state_sum(std::uint32_t state) {
        const std::uint32_t p = state * kMultiplier;
        return (p & 255U) + ((p >> 8U) & 255U) + ((p >> 16U) & 255U) + (p >> 24U);
    }
};

}  // namespace strix::core::trellis
