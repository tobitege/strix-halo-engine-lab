#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace strix::deploy {

// Device-usable bytes, not installed system RAM. Missing observations stay unknown.
struct DeviceFacts {
    std::string gfx_target;
    std::uint32_t cu_count = 0;
    std::uint64_t mem_total_bytes = 0, mem_free_bytes = 0;
    bool unified = false;
    std::string source = "unavailable";
    std::uint64_t gfx_target_version = 0;
    std::optional<std::uint64_t> gtt_total_bytes, gtt_used_bytes;
    std::optional<std::uint64_t> vram_total_bytes, vram_used_bytes;
    std::string drm_device; // Local observation path; deliberately omitted from plan JSON.
};

// root is a filesystem prefix containing sys/class/..., not the sys directory.
// A DRM device override preserves callers' explicit selection (including fixtures).
// Linux reads KFD/DRM sysfs and never initializes HIP. Windows has no such files: with root and override
// empty, a build with STRIX_DEPLOY_HIP_PROBE asks the HIP runtime for the device properties (no allocation and
// no kernel), which initializes it. The memory is the adapter's dedicated memory: its size from DXGI
// (DedicatedVideoMemory), its use from the adapter's PDH counter "Dedicated Usage" (not hipMemGetInfo or
// the DXGI segment budget, whose totals add the shared segment backed by system RAM); source is then
// "hip+dxgi+pdh". Without HIP the facts stay unknown.
[[nodiscard]] std::vector<DeviceFacts> probe_devices(const std::string& sysfs_root = "",
                                                     const std::string& drm_device = "");
// Refuses ambiguous selection by returning unknown facts. Never silently selects GPU 0.
[[nodiscard]] DeviceFacts probe_device(const std::string& sysfs_root = "",
                                      const std::string& drm_device = "");
// KFD's gfx_target_version of an architecture name: major * 10000 + minor * 100 + stepping, where the name
// spells the minor as one decimal digit and the stepping as one hex digit ("gfx1151" -> 110501,
// "gfx90a" -> 90010). A ":feature" suffix is ignored. 0 for a name that is not of this form.
[[nodiscard]] std::uint64_t gfx_target_version(std::string_view arch) noexcept;
// Existing GLM card-first selection, centralized here for its legacy telemetry contract.
[[nodiscard]] std::string first_drm_device(const std::string& sysfs_root = "");

} // namespace strix::deploy
