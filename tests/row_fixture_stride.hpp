#pragma once
#include <algorithm>
#include <cstdint>

#include "../include/strix/env_flags.hpp"

// Pitch sweep (timing only): STRIX_ROW_FIXTURE_ROUND_BYTES=N rounds the row stride up to a multiple of N bytes and
// STRIX_ROW_FIXTURE_PAD_BYTES=M adds M more. Unset keeps the tight stride. The payload columns never move.
inline std::uint64_t fixture_row_stride_bytes(std::uint64_t tight) {
    const auto round = static_cast<std::uint64_t>(std::max(strix::envcfg::integer("STRIX_ROW_FIXTURE_ROUND_BYTES", 0), 0));
    const auto pad = static_cast<std::uint64_t>(std::max(strix::envcfg::integer("STRIX_ROW_FIXTURE_PAD_BYTES", 0), 0));
    return (round ? (tight + round - 1U) / round * round : tight) + pad;
}
