#pragma once
#include "../../xport/transport_backend.hpp"
// MiniMax wire ABI, schedule and existing policy defaults.
#include <cstddef>
#include <cstdint>
#include <string_view>
namespace strix::minimax::xfer {
inline constexpr std::size_t kHidden = 6144;
inline constexpr std::size_t kRowF32 = kHidden * sizeof(float);  // one hidden-state partial row, FP32: 24576 B

// The MiniMax decode schedule (design and proof in design notes): per layer exactly two reductions of FP32
// partial rows (after o_proj, after the FFN) and one 8-byte (max, id) exchange per step for the vocabulary-split
// head; N users ride in one message (N rows). Router, routing and indexer are replicated: no metadata is exchanged.
struct MmSchedule {
    static constexpr unsigned kLayers = 60;
    static constexpr unsigned kPerLayer = 2;
    static constexpr unsigned kPerStep = kLayers * kPerLayer + 1;  // 121
    [[nodiscard]] static constexpr std::size_t row_bytes(unsigned rows) { return std::size_t{rows} * kRowF32; }
    static constexpr std::size_t kArgmaxBytes = 8;
};

enum class Kind : std::uint32_t { attn_sum = 1, ffn_sum = 2, argmax = 3, control = 4, gather = 5, test = 15 };
struct Tag {
    Kind kind = Kind::test;
    std::uint32_t layer = 0;
    std::uint64_t step = 0;
};

inline constexpr std::uint64_t kTrailerMagic = 0x3146584d58494d4dULL;  // "MMIXMXF1"
struct Trailer {
    std::uint64_t magic = kTrailerMagic;
    std::uint64_t epoch = 0;
    std::uint64_t plan = 0;
    std::uint64_t sequence = 0;
    std::uint64_t step = 0;
    std::uint32_t bytes = 0;
    std::uint16_t kind = 0;
    std::uint16_t flags = 0;        // kTrailerDeferred, kTrailerPrev
    std::uint32_t layer = 0;
    std::uint32_t rank = 0;
    std::uint32_t crc32c = 0;       // this payload's CRC-32C (0 when deferred)
    std::uint32_t prev_crc32c = 0;  // the previous payload's CRC-32C when that one was deferred (kTrailerPrev)
};
static_assert(sizeof(Trailer) == 64);
inline constexpr std::uint16_t kTrailerDeferred = 1, kTrailerPrev = 2;
inline constexpr std::size_t kInlineCrcMax = 1024;
struct MinimaxTraits {
    using Payload = float;
    using Schedule = MmSchedule;
    using Tag = xfer::Tag;
    using Trailer = xfer::Trailer;
    using Kind = xfer::Kind;
    static constexpr std::size_t kWidth = 6144;
    static constexpr std::uint64_t kPlan = 0x46584d4dULL;
    static constexpr std::uint64_t kMagic = kTrailerMagic;
    static constexpr std::size_t kInlineCrc = kInlineCrcMax;
    static constexpr std::uint16_t kDeferred = kTrailerDeferred, kPrevious = kTrailerPrev;
    static constexpr std::uint32_t kHeadLayer = 60; // standalone argmax helper's historical tag
    static constexpr const char* kEnvironment = "STRIX_MMX_";
    static constexpr bool kCrc = true, kCrcDeferred = true;
    static constexpr unsigned kFoldRows = 0;  // xport row fold of the decode all-reduce sum (STRIX_MMX_FOLD=rows)
    [[nodiscard]] static constexpr int memory_kind(std::string_view value) {
        return value == "host" ? 0 : value == "device" ? 1 : value == "uncached" ? 2 : -1;
    }
    static constexpr const char* kMemoryChoices = "host or device";
    [[nodiscard]] static constexpr const char* crc_description(bool enabled, bool deferred) {
        return enabled ? (deferred ? "deferred" : "inline") : "off";
    }
    [[nodiscard]] static constexpr Tag tag_at(unsigned site, std::uint64_t step) {
        return site == Schedule::kPerStep - 1
            ? Tag{Kind::argmax, Schedule::kLayers, step}
            : Tag{site % 2 == 0 ? Kind::attn_sum : Kind::ffn_sum, site / 2, step};
    }
    [[nodiscard]] static constexpr std::size_t payload_bytes(unsigned site, unsigned rows) {
        return site == Schedule::kPerStep - 1
            ? std::size_t{rows} * Schedule::kArgmaxBytes : Schedule::row_bytes(rows);
    }
};
} // namespace strix::minimax::xfer
