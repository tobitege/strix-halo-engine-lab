#pragma once
// Fit a row capacity (prefill block / chunk rows) to the device memory that is left once everything else is resident.
// Shared host logic, model- and backend-agnostic: an engine allocates its per-row workspace LAST, tries its largest
// capacity first and steps down until the allocation succeeds with the caller's reserve still free. Block / chunk rows
// only change how many rows one launch covers, never the per-row arithmetic, so an engine whose kernels are
// row-independent keeps its bits (each binding proves that with logit dumps; the CUDA Qwen3.8-Flash-Next receipt is in
// docs/engine/serving/qwen-cuda-prefill.md).
// Bound: CUDA Qwen3.8-Flash-Next prefill driver (block rows, qwen_prefill_driver.cu allocate_fitted_block).
// Same pattern (follow-ups named in wave4/next1112/RTXLONG-RESULT.md): CUDA DS41 prefill driver chunk_rows, CUDA Qwen3.8-27B
// prefill chunk, the HIP engines' prefill blocks.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::rows_fit {

struct Result {
    std::uint32_t rows = 0;   // the capacity that fitted
    std::uint32_t want = 0;   // the capacity asked for
    std::uint32_t tries = 0;  // allocation attempts
    [[nodiscard]] bool reduced() const noexcept { return rows != want; }
};

// The next capacity below `rows`: the largest multiple of `step` strictly below it, never below `min_rows`;
// 0 when `rows` is already `min_rows` (nothing smaller to try).
[[nodiscard]] inline std::uint32_t next_lower(std::uint32_t rows, std::uint32_t min_rows, std::uint32_t step) {
    if (step == 0 || min_rows == 0) throw std::invalid_argument("rows_fit: step and min_rows must be > 0");
    if (rows <= min_rows) return 0;
    const std::uint32_t lower = (rows - 1U) / step * step;
    return lower < min_rows ? min_rows : lower;
}

// try_rows(rows) -> bool: allocate the workspace for `rows` rows and return true when it fits (keeping it), or release
// whatever it allocated and return false. Tries want, then next_lower(...) down to min_rows; throws when none fits.
template <class Try>
Result fit(std::uint32_t want, std::uint32_t min_rows, std::uint32_t step, Try&& try_rows) {
    if (want == 0 || min_rows == 0 || step == 0 || min_rows > want)
        throw std::invalid_argument("rows_fit: need 0 < min_rows <= want and step > 0");
    Result r;
    r.want = want;
    for (std::uint32_t rows = want; rows != 0; rows = next_lower(rows, min_rows, step)) {
        ++r.tries;
        if (try_rows(rows)) {
            r.rows = rows;
            return r;
        }
    }
    throw std::runtime_error("rows_fit: no capacity between " + std::to_string(min_rows) + " and " + std::to_string(want) +
                             " rows fits the free device memory");
}

// A block schedule clamped to a fitted capacity: entries above `cap` become `cap` (a ramp keeps its small first blocks).
[[nodiscard]] inline std::vector<std::uint32_t> clamp_schedule(std::vector<std::uint32_t> schedule, std::uint32_t cap) {
    for (auto& rows : schedule) rows = rows > cap ? cap : rows;
    return schedule;
}

}  // namespace strix::rows_fit
