// Prefill group policy (shared): how many forward chunks of one prompt a pair-prefill interleaves as one
// layer-major group. A group runs all layers once over its chunks, so per-layer work that is independent of the
// rows (JIT weight conversion of the FAST_I4 slots, expert-table uploads, weight sweeps) is paid once per group
// instead of once per chunk window. Measured (MiniMax M3.1 FAST_I4, 32K prompt, TP2 USB4, ABAB both orders): group 4
// vs 2 = TTFT 55.0 / 54.5 -> 52.2 / 51.7 s (-5.1 %); group 8 vs 2 = 55.5 / 55.2 -> 52.5 / 51.4 s (-6.1 %, i.e. the
// same as 4 within run noise); identical state / rows / result hashes, decode unchanged. Default 4: the gain of 8 is
// not distinguishable from 4 and 8 costs 1.6 GiB more device memory per rank plus a deeper exchange (opt-in
// STRIX_MM_PF_GROUP=8 with STRIX_MM_PFX_DEPTH=8).
//
// The engine clamps the width to the prompt's chunk count at run time (an 8K prompt of 4096-row chunks is a group
// of 2 whatever the default), so a wider default only costs memory: (width - 1) x 4 row buffers of max_rows x
// hidden floats per rank (MiniMax: 0.8 GiB at width 4, 2.4 GiB at width 8), plus one exchange slot per chunk.
#pragma once

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::pair::prefill_group {

inline constexpr unsigned kMin = 2;
inline constexpr unsigned kMax = 8;      // one 32K prompt of 4096-row chunks as a single group
inline constexpr unsigned kDefault = 4;
// Prefix-cache reuse alignment, in chunks. Chunk bits do not depend on the group a chunk is scheduled in (group 2
// and group 4 produce identical state hashes, and a resumed prefill's tail group is already narrower), and a
// prefill restarted at a multiple of max_rows tiles its chunks exactly as the cold prefill does, so one chunk is the
// finest alignment that reproduces the cold chunk boundaries by construction.
inline constexpr unsigned kReuseChunks = 1;

// Width from an environment value (nullptr / empty = default). Throws on an out-of-range explicit value.
inline unsigned width_from_env(const char* value, const char* name) {
    if (value == nullptr || *value == '\0') return kDefault;
    const unsigned long w = std::strtoul(value, nullptr, 10);
    if (w < kMin || w > kMax)
        throw std::invalid_argument(std::string(name) + " must be " + std::to_string(kMin) + ".." + std::to_string(kMax));
    return static_cast<unsigned>(w);
}

// The exchange module needs one TX / RX slot per interleaved chunk (slot reuse of the schedule): the width
// a given exchange depth admits.
inline unsigned width_for_depth(unsigned width, unsigned depth) { return std::max(kMin, std::min(width, depth)); }

// MoE sub-group: how many chunks of the group one group-wide MoE call covers (an expert's weights are then decoded
// once for those chunks' rows). A group-wide call produces all its chunks' partials at once, so their exchanges start
// together; the pipeline stays free of idle time only while another sub-group's compute covers those transfers.
// Measured (MiniMax M3.1 FAST_I4 policy default, TP2 USB4, identical state / rows / result hashes in every arm):
//   8K, 2 chunks, one call for both:        12.28 / 12.96 -> 15.14 / 14.60 s (+13..23 %: the wire is serialised)
//   32K, group 4, calls of 2 chunks:        49.35 / 49.28 -> 49.36 / 49.15 s (neutral)
//   32K, group 8, calls of 4 chunks vs the default group 4 per chunk, both orders: 49.20 -> 48.99, 49.25 -> 48.75 s
// The automatic width is therefore half the chunks of a unit of at least kMoeAutoChunks chunks (two calls of four
// chunks in a 32K group of 8) and one call per chunk below that. 0 = automatic, 1 = one call per chunk, 2..8 = explicit.
inline constexpr unsigned kMoeAuto = 0;
inline constexpr unsigned kMoeAutoChunks = 8;
inline unsigned moe_width_from_env(const char* value, const char* name) {
    if (value == nullptr || *value == '\0') return kMoeAuto;
    const unsigned long w = std::strtoul(value, nullptr, 10);
    if (w > kMax) throw std::invalid_argument(std::string(name) + " must be 0.." + std::to_string(kMax));
    return static_cast<unsigned>(w);
}
// Sub-group width for a unit of `chunks` chunks.
inline unsigned moe_width(unsigned requested, unsigned chunks) {
    if (chunks == 0) return 1;
    const unsigned w = requested == kMoeAuto ? (chunks >= kMoeAutoChunks ? chunks / 2 : 1U) : std::min(requested, chunks);
    return std::max(1U, w);
}

}  // namespace strix::pair::prefill_group
