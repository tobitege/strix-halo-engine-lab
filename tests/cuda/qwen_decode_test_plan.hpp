#pragma once

#include "strix/env_flags.hpp"
#include <array>
#include <cstdint>

namespace strix::cuda::decode_test {
inline std::uint32_t max_width() {
    return envcfg::number<std::uint32_t>("STRIX_QWEN_TMAX", 8U, 2U, 16U);
}
inline bool benchmark_rows_sufficient(std::uint32_t rows, std::uint32_t width) {
    return rows >= width;
}
// Cycling keep=1..width consumes this many rows, then needs one continuation.
inline std::uint32_t rollback_rows_required(std::uint32_t width) {
    return width * (width + 1U) / 2U + 1U;
}
// Credit only actual full-width commits whose saved state was subsequently read.
struct Coverage {
    std::array<bool, 17> keeps{};
    std::uint32_t pending_keep = 0;
    bool full_window = false;
    void observe(std::uint32_t rows, std::uint32_t width) {
        if (pending_keep) keeps[pending_keep] = true;
        pending_keep = 0;
        full_window = full_window || rows == width;
    }
    void commit(std::uint32_t rows, std::uint32_t width, std::uint32_t keep) {
        pending_keep = rows == width ? keep : 0U;
    }
    bool complete(std::uint32_t width) const {
        for (std::uint32_t keep = 1; keep <= width; ++keep)
            if (!keeps[keep]) return false;
        return full_window;
    }
};
}  // namespace strix::cuda::decode_test
