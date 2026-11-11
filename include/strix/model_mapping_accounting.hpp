#pragma once

#include <cstdint>
#include <stdexcept>

namespace strix {
struct ModelMappingAccounting {
    std::uint64_t retained_bytes = 0;
    std::uint64_t reclaimable_bytes = 0;
};
// A read-only file mapping's clean, unregistered pages are reclaimable cache,
// already represented in MemAvailable. Charge every registered/pinned page,
// including coalescing and boundary pages, regardless of its current residency.
// Callers supply the page-rounded mapping and disjoint registration extents;
// this rule does not apply to writable mappings, copies or other owners.
inline ModelMappingAccounting read_only_registered_mapping_accounting(
    std::uint64_t mapped_bytes, std::uint64_t registered_bytes) {
    if (registered_bytes > mapped_bytes)
        throw std::invalid_argument("registered model backing exceeds read-only mapping");
    return {registered_bytes, mapped_bytes - registered_bytes};
}
} // namespace strix
