#pragma once
// GLM wire ABI and schedule, extracted without changing their values.
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace strix::glm::xfer {
inline constexpr std::size_t kRowF32 = std::size_t{4096} * sizeof(float);  // one hidden-state partial row, FP32: 16384 B

// The GLM decode schedule (design notes section 1): every one of the 45 layers has two sublayers (attention: KDA or MLA/DSA; FFN:
// dense L0-2 or MoE + shared), each `X' = post * (p0 + p1) + C X` with an mHC pre-mix that is nonlinear in the whole
// 4 x 4096 stream state, so exactly one reduction of FP32 partial rows per sublayer (90), plus one 8-byte (max, id)
// exchange per step for the vocabulary-split head: 91. N users / verify rows ride in one message (N rows). Router,
// routing, mHC mixes and the indexer selection are replicated: no metadata is exchanged. The MTP draft layer adds 3
// (attention, FFN, its own head argmax) per draft step.
struct GlmSchedule {
    static constexpr unsigned kLayers = 45;
    static constexpr unsigned kPerLayer = 2;
    static constexpr unsigned kPerStep = kLayers * kPerLayer + 1;  // 91
    static constexpr unsigned kPerDraftStep = 3;                   // MTP layer: 2 reductions + head argmax
    [[nodiscard]] static constexpr std::size_t row_bytes(unsigned rows) { return std::size_t{rows} * kRowF32; }
    static constexpr std::size_t kArgmaxBytes = 8;
};

enum class Kind : std::uint32_t { attn_sum = 1, ffn_sum = 2, argmax = 3, control = 4, gather = 5,
                                shard_prepare = 6, shard_partial = 7, test = 15 };
struct Tag {
    Kind kind = Kind::test;
    std::uint32_t layer = 0;
    std::uint64_t step = 0;
};

inline constexpr std::uint64_t kTrailerMagic = 0x31524546584d4c47ULL;  // "GLMXFER1"
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
// Messages up to this size always carry their own CRC (the argmax / control messages); larger ones defer it when
// Options::crc_deferred: the CRC of message k travels in message k+1's trailer (or in a seal at drain()), computed by
// a worker thread while the wire moves the payload (TX) and after the stream was released (RX). A corrupted payload
// still fails the step (check() / drain()), one exchange later, before any result is trusted.
inline constexpr std::size_t kInlineCrcMax = 1024;


struct GlmTraits {
    static constexpr unsigned kStagingSlots = 2;
    using Payload = float;
    using Schedule = GlmSchedule;
    using Tag = xfer::Tag;
    using Trailer = xfer::Trailer;
    using Kind = xfer::Kind;
    static constexpr std::size_t kWidth = 4096;
    static constexpr std::uint64_t kPlan = 0x584d4c47ULL;
    static constexpr std::uint64_t kMagic = kTrailerMagic;
    static constexpr std::size_t kInlineCrc = kInlineCrcMax;
    static constexpr std::uint16_t kDeferred = kTrailerDeferred, kPrevious = kTrailerPrev;
    static constexpr std::uint32_t kHeadLayer = 60; // standalone argmax helper's historical tag
    static constexpr const char* kEnvironment = "STRIX_GLMX_";
    static constexpr bool kCrc = true, kCrcDeferred = true;
    static constexpr unsigned kFoldRows = 0;  // xport row fold of the sublayer boundary sum (STRIX_GLMX_FOLD=rows)
    [[nodiscard]] static constexpr int memory_kind(std::string_view value) {
        return value == "host" ? 0 : value == "device" ? 1 : value == "uncached" ? 2 : -1;
    }
    static constexpr const char* kMemoryChoices = "host or device";
    [[nodiscard]] static constexpr const char* crc_description(bool enabled, bool) {
        return enabled ? "on" : "off";
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
} // namespace strix::glm::xfer
