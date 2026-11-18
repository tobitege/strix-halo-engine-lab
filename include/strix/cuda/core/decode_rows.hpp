#pragma once
// Decode-shaped row cap of the CUDA backend (shared by every model's row kernels, any arch).
// The decode-shaped (few-row) forms were written for <= 8 rows. A 9..16-row verify window (one chat's wide copy window,
// or two chats' 8-row windows in one pass) used to fall back to the tiled prefill forms, which cost a flat ~62 ms per
// Qwen step on sm_120 against 17 ms at 8 rows. Up to this cap, a family runs its decode form instead:
// either the form already loops over 8-row pieces, or the call is split into <= 8-row pieces. Either way every
// (row, output) chain is the one of the tiled form, so the bits do not change (rows never mix inside a chain).
// STRIX_CUDA_DECODE_ROWS_MAX = 8 restores the old switch point; values are clamped to 8..16.
#include "strix/env_flags.hpp"

#include <cstdlib>

namespace strix::cuda::core {

constexpr unsigned kDecodePieceRows = 8U;   // rows per decode-shaped piece (the forms' template ceiling)

inline unsigned decode_rows_max() {
    static const unsigned cap = [] {
        const char* e = strix::envcfg::text("STRIX_CUDA_DECODE_ROWS_MAX");
        const unsigned long n = e && *e ? std::strtoul(e, nullptr, 10) : 16UL;
        return n < 8UL ? 8U : n > 16UL ? 16U : static_cast<unsigned>(n);
    }();
    return cap;
}

}  // namespace strix::cuda::core
