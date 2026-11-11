#pragma once
#include <cstdint>
#include <optional>

namespace strix {
// System RAM available to new allocations, where it is a pool of its own next to the GPU's memory: on Windows
// the firmware sets the GPU's dedicated memory aside, and Windows sees and allocates only the rest
// (GlobalMemoryStatusEx: ullAvailPhys). Unknown (nullopt) elsewhere: on a Linux UMA node host and device
// allocations share one pool, which /proc/meminfo describes (strix/linux_uma_observation.hpp).
[[nodiscard]] std::optional<std::uint64_t> separate_host_memory_available() noexcept;
} // namespace strix
