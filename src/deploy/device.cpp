#include "strix/deploy/device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

#if defined(STRIX_DEPLOY_HIP_PROBE)
#include <hip/hip_runtime_api.h>
#include <cstring>
#include <optional>
#include <utility>
#include <cwchar>
#include <dxgi1_4.h>
#include <pdh.h>
#include <pdhmsg.h>
#endif

namespace strix::deploy {
namespace {
namespace fs = std::filesystem;
fs::path rooted(const std::string& root, const std::string& path) {
    return root.empty() ? fs::path(path) : fs::path(root) / fs::path(path).relative_path();
}
std::optional<std::uint64_t> read_counter(const fs::path& path) {
    std::ifstream in(path);
    std::string value, tail;
    if (!(in >> value) || value.size() > 20 || (in >> tail) || value.empty() || value[0] == '-') return {};
    try {
        std::size_t end = 0;
        const auto n = std::stoull(value, &end);
        if (end == value.size()) return n;
    } catch (const std::exception&) {}
    return {};
}
std::vector<fs::path> directories(const fs::path& parent) {
    std::vector<fs::path> paths;
    std::error_code ec;
    for (fs::directory_iterator it(parent, ec), end; !ec && it != end; it.increment(ec)) {
        paths.push_back(it->path());
        if (paths.size() > 256) return {};
    }
    if (ec) return {};
    std::sort(paths.begin(), paths.end());
    return paths;
}
DeviceFacts drm(const fs::path& path, DeviceFacts d) {
    d.drm_device = path.string();
    d.gtt_total_bytes = read_counter(path / "mem_info_gtt_total");
    d.gtt_used_bytes = read_counter(path / "mem_info_gtt_used");
    d.vram_total_bytes = read_counter(path / "mem_info_vram_total");
    d.vram_used_bytes = read_counter(path / "mem_info_vram_used");
    const auto total = d.unified ? d.gtt_total_bytes : d.vram_total_bytes;
    const auto used = d.unified ? d.gtt_used_bytes : d.vram_used_bytes;
    if (total && used && *used <= *total) {
        d.mem_total_bytes = *total;
        d.mem_free_bytes = *total - *used;
    }
    if (d.gtt_total_bytes || d.vram_total_bytes) d.source = d.gfx_target.empty() ? "drm" : "kfd+drm";
    return d;
}
void fake_memory(DeviceFacts& d) {
    const char* value = std::getenv("STRIX_DEPLOY_FAKE_MEM_GIB");
    if (!value || !*value) return;
    char* end = nullptr;
    const long double gib = std::strtold(value, &end);
    const long double bytes = gib * 1073741824.0L;
    if (end == value || *end || !std::isfinite(gib) || gib <= 0 ||
        bytes >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max()))
        throw std::invalid_argument("STRIX_DEPLOY_FAKE_MEM_GIB must be a positive finite size");
    d.mem_total_bytes = d.mem_free_bytes = static_cast<std::uint64_t>(bytes);
    d.source = "test-memory-override";
    if (d.unified) { d.gtt_total_bytes = d.mem_total_bytes; d.gtt_used_bytes = 0; }
    else { d.vram_total_bytes = d.mem_total_bytes; d.vram_used_bytes = 0; }
}
#if defined(STRIX_DEPLOY_HIP_PROBE)
// Windows: no KFD topology or DRM counters exist; the HIP runtime is the only source of the architecture. A
// deliberate deviation from the Linux contract above (which never initializes HIP), limited to queries of the
// device properties. The memory comes from DXGI: the adapter's local segment, which on an APU is the memory
// the firmware set aside for the GPU, separate from the system RAM Windows uses. A failed HIP query leaves
// every fact unknown, a failed DXGI query the memory, as on a Linux host without the counters.

// The dedicated memory of the adapter with this LUID: {size, current use} in bytes. On an APU that is the memory
// the firmware set aside for the GPU. DXGI's segment budget does not separate it: it reports one local group
// that includes the shared segment Windows backs with system RAM. The size is the adapter's
// DedicatedVideoMemory, the use the "GPU Adapter Memory\Dedicated Usage" counter of the same adapter.
std::optional<std::pair<std::uint64_t, std::uint64_t>> dedicated_memory(const char (&luid)[8]) {
    LUID id{};
    std::memcpy(&id, luid, sizeof(id));
    std::uint64_t size = 0;
    IDXGIFactory4* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory)))) return {};
    IDXGIAdapter1* adapter = nullptr;
    if (SUCCEEDED(factory->EnumAdapterByLuid(id, __uuidof(IDXGIAdapter1), reinterpret_cast<void**>(&adapter)))) {
        DXGI_ADAPTER_DESC1 description{};
        if (SUCCEEDED(adapter->GetDesc1(&description))) size = description.DedicatedVideoMemory;
        adapter->Release();
    }
    factory->Release();
    if (!size) return {};

    wchar_t path[160];
    std::swprintf(path, std::size(path), L"\\GPU Adapter Memory(luid_0x%08lX_0x%08lX_phys_0)\\Dedicated Usage",
        static_cast<unsigned long>(id.HighPart), static_cast<unsigned long>(id.LowPart));
    PDH_HQUERY query = nullptr;
    if (PdhOpenQueryW(nullptr, 0, &query) != ERROR_SUCCESS) return {};
    std::optional<std::pair<std::uint64_t, std::uint64_t>> result;
    PDH_HCOUNTER counter = nullptr;
    PDH_FMT_COUNTERVALUE value{};
    if (PdhAddEnglishCounterW(query, path, 0, &counter) == ERROR_SUCCESS && PdhCollectQueryData(query) == ERROR_SUCCESS &&
        PdhGetFormattedCounterValue(counter, PDH_FMT_LARGE, nullptr, &value) == ERROR_SUCCESS &&
        value.CStatus == PDH_CSTATUS_VALID_DATA && value.largeValue >= 0)
        result = std::pair<std::uint64_t, std::uint64_t>{size, static_cast<std::uint64_t>(value.largeValue)};
    PdhCloseQuery(query);
    return result;
}
std::vector<DeviceFacts> probe_hip_devices() {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count <= 0) return {};
    std::vector<DeviceFacts> result;
    for (int index = 0; index < count; ++index) {
        hipDeviceProp_t properties{};
        if (hipGetDeviceProperties(&properties, index) != hipSuccess) return {};
        DeviceFacts d;
        const std::string_view arch(properties.gcnArchName);
        d.gfx_target = std::string(arch.substr(0, arch.find(':')));
        d.gfx_target_version = gfx_target_version(d.gfx_target);
        if (properties.multiProcessorCount > 0) d.cu_count = static_cast<std::uint32_t>(properties.multiProcessorCount);
        d.unified = properties.integrated != 0;
        // Memory: the adapter's dedicated memory (its size and current use), not hipMemGetInfo, which adds the
        // shared segment Windows backs with system RAM. Unknown when DXGI or the counter does not answer.
        if (const auto dedicated = dedicated_memory(properties.luid)) {
            const auto used = std::min(dedicated->second, dedicated->first);
            d.mem_total_bytes = dedicated->first;
            d.mem_free_bytes = dedicated->first - used;
            if (d.unified) { d.gtt_total_bytes = dedicated->first; d.gtt_used_bytes = used; }
            else { d.vram_total_bytes = dedicated->first; d.vram_used_bytes = used; }
        }
        d.source = "hip+dxgi";
        fake_memory(d);
        result.push_back(std::move(d));
    }
    return result;
}
#endif
} // namespace

std::uint64_t gfx_target_version(std::string_view arch) noexcept {
    arch = arch.substr(0, arch.find(':'));
    if (!arch.starts_with("gfx") || arch.size() < 6) return 0;
    const auto hex = [](char c) -> int {
        return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
    };
    const int stepping = hex(arch.back()), minor = hex(arch[arch.size() - 2]);
    const auto major_text = arch.substr(3, arch.size() - 5);
    std::uint64_t major = 0;
    for (const char c : major_text) {
        if (c < '0' || c > '9') return 0;
        major = major * 10U + static_cast<std::uint64_t>(c - '0');
    }
    if (stepping < 0 || minor < 0 || minor > 9 || major == 0 || major > 99) return 0;
    return major * 10000U + static_cast<std::uint64_t>(minor) * 100U + static_cast<std::uint64_t>(stepping);
}

std::string first_drm_device(const std::string& root) {
    for (const auto& p : directories(rooted(root, "/sys/class/drm"))) {
        const auto name = p.filename().string();
        if (name.starts_with("card") && name.find('-') == std::string::npos &&
            read_counter(p / "device/mem_info_gtt_total")) return (p / "device").string();
    }
    return {};
}

std::vector<DeviceFacts> probe_devices(const std::string& root, const std::string& override_device) {
#if defined(STRIX_DEPLOY_HIP_PROBE)
    if (root.empty() && override_device.empty()) return probe_hip_devices();
#endif
    std::vector<DeviceFacts> result;
    for (const auto& node : directories(rooted(root, "/sys/class/kfd/kfd/topology/nodes"))) {
        std::ifstream in(node / "properties", std::ios::binary);
        std::string line;
        std::map<std::string, std::uint64_t> properties;
        std::size_t bytes = 0;
        while (std::getline(in, line)) {
            bytes += line.size() + 1;
            if (bytes > 65536) break;
            std::istringstream fields(line);
            std::string key;
            std::uint64_t value = 0;
            if (fields >> key >> value) properties[key] = value;
        }
        if (bytes > 65536 || !properties["drm_render_minor"]) continue;
        const auto path = rooted(root, "/sys/class/drm/renderD" + std::to_string(properties["drm_render_minor"]) + "/device");
        if (!override_device.empty()) {
            std::error_code ec;
            if (fs::weakly_canonical(path, ec) != fs::weakly_canonical(override_device, ec)) continue;
        }
        DeviceFacts d;
        d.gfx_target_version = properties["gfx_target_version"];
        if (d.gfx_target_version) {
            // KFD encodes major*10000 + minor*100 + stepping; gfx spelling uses hex stepping.
            std::ostringstream gfx;
            gfx << "gfx" << d.gfx_target_version / 10000 << (d.gfx_target_version / 100) % 100
                << std::hex << d.gfx_target_version % 100;
            d.gfx_target = gfx.str();
        }
        const auto simd = properties["simd_count"], per_cu = properties["simd_per_cu"];
        if (per_cu && simd / per_cu <= std::numeric_limits<std::uint32_t>::max())
            d.cu_count = static_cast<std::uint32_t>(simd / per_cu);
        // KFD local_mem_size distinguishes dedicated memory from unified system memory.
        d.unified = properties.contains("local_mem_size") && properties["local_mem_size"] == 0;
        d = drm(path, std::move(d));
        if (!properties.contains("local_mem_size")) d.mem_total_bytes = d.mem_free_bytes = 0;
        fake_memory(d);
        result.push_back(std::move(d));
    }
    if (result.empty() && !override_device.empty()) {
        DeviceFacts d;
        // Legacy explicit DRM telemetry may have no KFD fixture. All raw counters remain available.
        d = drm(override_device, std::move(d));
        d.mem_total_bytes = d.mem_free_bytes = 0; // raw telemetry without a measured memory kind
        fake_memory(d);
        result.push_back(std::move(d));
    }
    return result;
}

DeviceFacts probe_device(const std::string& root, const std::string& override_device) {
    auto all = probe_devices(root, override_device);
    if (all.size() == 1) return all.front();
    DeviceFacts d;
    if (all.size() > 1) d.source = "ambiguous-device-selection";
    // The test override also permits host-only deploy-check without a physical GPU.
    fake_memory(d);
    return d;
}
} // namespace strix::deploy
