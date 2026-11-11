#include "strix/host_memory.hpp"

#if defined(_WIN32)
#include <windows.h>
#endif

namespace strix {
std::optional<std::uint64_t> separate_host_memory_available() noexcept {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) && status.ullTotalPhys) return status.ullAvailPhys;
#endif
    return std::nullopt;
}
} // namespace strix
