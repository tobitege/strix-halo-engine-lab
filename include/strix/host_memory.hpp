#pragma once
#include <cstdint>
#include <optional>

namespace strix {
// Whether system RAM is a pool of its own next to the GPU's memory. On Windows the firmware sets the GPU's
// dedicated memory aside, and Windows sees and allocates only the rest. On a Linux UMA node host and device
// allocations share one pool, which /proc/meminfo describes (strix/linux_uma_observation.hpp).
// Where the pools are separate, the model lives in the GPU pool (hip_model_copy_all_mode in
// strix/hip_model_residency.hpp; the options that would leave part of it registered are refused, see
// separate_pool_residency_refusal in apps/qwen_effective_config.hpp) and host-only items are priced against
// the host pool.
[[nodiscard]] constexpr bool host_memory_is_separate_pool() noexcept {
#if defined(_WIN32)
    return true;
#else
    return false;
#endif
}
// The RAM available to new allocations in a separate host pool (Windows: GlobalMemoryStatusEx, ullAvailPhys).
// Unknown (nullopt) when the query fails, and always where host and GPU share one pool.
[[nodiscard]] std::optional<std::uint64_t> separate_host_memory_available() noexcept;
} // namespace strix
