#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>

namespace strix {
// Indices into the live member list, never replacement slot IDs. Each member
// appears once. Bounded controller searches see at most eight requests; a
// fixed-row plan (above the row budget's lane bound) may take up to max_group
// (MtpBatchWindowController::max_group_lanes), so that one target forward reads
// the weights for all of its lanes instead of one forward per eight lanes.
inline std::vector<std::vector<std::size_t>> mtp_cohort_partition(std::size_t count, std::size_t start,
                                                                   std::size_t max_group = 8U) {
    std::vector<std::vector<std::size_t>> groups;
    if (!count) return groups;
    start %= count;
    if (!max_group) max_group = 8U;
    for (std::size_t offset = 0; offset < count;) {
        auto& group = groups.emplace_back();
        const auto size = std::min<std::size_t>(max_group, count - offset);
        for (std::size_t j = 0; j < size; ++j) group.push_back((start + offset + j) % count);
        offset += size;
    }
    return groups;
}
} // namespace strix
