#pragma once
// Host-only switch of the FAST indexer score core (index_score_wmma.hpp), usable from plain C++ (rank apps, startup
// agreement) without HIP headers.
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strix::attn_core::idx {

// STRIX_IDX_WMMA = "0" -> 0 (off, exact/default kernels), "1"/"bf16" -> 1 (bf16), "f16" -> 2 (f16) where the engine
// accepts FP32 inputs for both operands (GLM), "bf16x2" -> 3 (the query as a BF16 hi + lo pair: two WMMA terms per
// dot, ~2^-16 relative operand error instead of 2^-8; engines with BF16 keys, MiniMax), "exact" -> 4 (WMMA scores,
// bf16x2 on MiniMax and f16 on GLM, then the blocks at the selection cut-off are rescored with the exact kernel
// arithmetic: the selected set is the exact one, see index_score_band.hpp); other values fail closed.
// Unset or empty -> the caller's default: off, or "exact" where the engine made it its default (MiniMax, GLM).
// Read on every call so a process can A/B the paths. Values 1 and 2 equal tile::Operand.
inline constexpr unsigned kModeOff = 0U, kModeBf16 = 1U, kModeF16 = 2U, kModeBf16x2 = 3U, kModeExact = 4U;
inline unsigned wmma_mode(unsigned unset_default = kModeOff) {
    const char* e = std::getenv("STRIX_IDX_WMMA");
    if (e == nullptr || *e == '\0') return unset_default;
    if (std::strcmp(e, "0") == 0) return kModeOff;
    if (std::strcmp(e, "1") == 0 || std::strcmp(e, "bf16") == 0) return kModeBf16;
    if (std::strcmp(e, "f16") == 0) return kModeF16;
    if (std::strcmp(e, "bf16x2") == 0) return kModeBf16x2;
    if (std::strcmp(e, "exact") == 0) return kModeExact;
    throw std::invalid_argument("STRIX_IDX_WMMA must be 0, 1, bf16, f16, bf16x2 or exact");
}
inline const char* wmma_mode_name(unsigned m) {
    return m == kModeBf16 ? "bf16" : m == kModeF16 ? "f16" : m == kModeBf16x2 ? "bf16x2" : m == kModeExact ? "exact" : "off";
}

}  // namespace strix::attn_core::idx
