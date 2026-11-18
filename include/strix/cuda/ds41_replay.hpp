#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace strix::ds41::cuda {

// Configuration admission is separate from model quality qualification.
inline void validate_bounded_replay(std::uint32_t rows, bool cone, std::uint32_t first,
                                    std::uint32_t end, std::uint32_t head_rows,
                                    std::uint32_t draft_rows, bool fast_dense, bool fast_moe) {
    if (rows == 0U) return;
    if (rows != 256U || !cone || first != 0U || end != 40U ||
        head_rows > rows || draft_rows > rows || !fast_dense || !fast_moe)
        throw std::invalid_argument("bounded replay requires B=256, the full 40-layer cone, fast dense/MoE, and head/draft rows <=256");
}

// Clamp the already head-expanded exact cone, then round down to call boundaries.
// Encoder output remains available for older rows never updated by a decoder.
// The KV source still projects every input row (x0 is chosen by the driver).
inline void bound_decoder_replay(std::uint32_t total, std::uint32_t bound,
                                 std::uint32_t call_rows, std::uint32_t& out_begin,
                                 std::uint32_t& kv_begin) {
    if (bound == 0U) return;
    const auto b = std::min(total, bound);
    const auto kv = std::min<std::uint64_t>(total, std::uint64_t{b} + 127U);
    out_begin = std::max(out_begin, (total - b) / call_rows * call_rows);
    kv_begin = std::max(kv_begin, static_cast<std::uint32_t>(total - kv) / call_rows * call_rows);
}

} // namespace strix::ds41::cuda
