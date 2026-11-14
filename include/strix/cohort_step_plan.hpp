#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace strix {
struct CohortRowLaunch {
    std::size_t offset = 0;
    std::size_t rows = 0;
};
// For 0..32 rows; zero-row entries are unused. Split only if >=2 pad rows are saved to amortize extra launches.
constexpr std::array<CohortRowLaunch, 3> cohort_row_split(std::size_t count) {
    if (count > 32U) throw std::invalid_argument("cohort row split requires 0..32 rows");
    std::array<CohortRowLaunch, 3> plan{};
    if (count == 15U || count == 31U) {
        plan[0] = {0U, count + 1U};
        return plan;
    }
    std::size_t offset = 0;
    std::size_t launch = 0;
    while (count != 0U) {
        const auto rows = count >= 32U ? 32U : count >= 16U ? 16U : count >= 8U ? 8U : count;
        plan[launch++] = {offset, rows};
        offset += rows;
        count -= rows;
    }
    return plan;
}
// Served MTP cohort policy: <=8 rows -> default path, 9-15 -> wide R16 MoE,
// exactly 16 -> wide with project G=4. STRIX_COHORT_AUTO=0 restores the old defaults; explicit knobs win.
inline bool cohort_auto() {
    static const bool enabled = [] {
#if defined(_MSC_VER)
        char setting[3]{};
        std::size_t required = 0;
        if (getenv_s(&required, setting, sizeof(setting), "STRIX_COHORT_AUTO") != 0 || required == 0) return true;
        const char* value = setting;
#else
        const char* value = std::getenv("STRIX_COHORT_AUTO");
        if (!value) return true;
#endif
        return !(value[0] == '0' && value[1] == '\0');
    }();
    return enabled;
}
inline std::uint32_t cohort_step_rows() {
    static const std::uint32_t rows = [] {
#if defined(_MSC_VER)
        char setting[3]{};
        std::size_t required = 0;
        if (getenv_s(&required, setting, sizeof(setting), "STRIX_COHORT_STEP_ROWS") != 0 || required == 0)
            return cohort_auto() ? 16U : 8U;
        const char* value = setting;
#else
        const char* value = std::getenv("STRIX_COHORT_STEP_ROWS");
        if (!value) return cohort_auto() ? 16U : 8U;
#endif
        return std::string_view(value) == "16" ? 16U : 8U;
    }();
    return rows;
}
// Input is already in stable descending window order; a session is never split.
inline std::vector<std::vector<std::size_t>> cohort_step_plan(
    std::span<const std::uint32_t> rows, std::uint32_t cap) {
    std::vector<std::vector<std::size_t>> bins;
    std::vector<std::uint32_t> sums;
    for (std::size_t index = 0; index < rows.size(); ++index) {
        bool placed = false;
        for (std::size_t bin = 0; bin < bins.size(); ++bin) {
            if (sums[bin] + rows[index] <= cap) {
                bins[bin].push_back(index); sums[bin] += rows[index]; placed = true; break;
            }
        }
        if (!placed) { bins.push_back({index}); sums.push_back(rows[index]); }
    }
    std::vector<std::vector<std::size_t>> result;
    for (std::size_t bin = 0; bin < bins.size(); ++bin) {
        auto& entries = bins[bin];
        std::vector<std::size_t> tail;
        while (entries.size() >= 2U && sums[bin] != 4U && sums[bin] != 8U && sums[bin] != 16U) {
            tail.push_back(entries.back()); sums[bin] -= rows[entries.back()]; entries.pop_back();
        }
        result.push_back(entries);
        for (auto index = tail.rbegin(); index != tail.rend(); ++index) result.push_back({*index});
    }
    return result;
}
}
