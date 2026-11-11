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
} // namespace

std::string first_drm_device(const std::string& root) {
    for (const auto& p : directories(rooted(root, "/sys/class/drm"))) {
        const auto name = p.filename().string();
        if (name.starts_with("card") && name.find('-') == std::string::npos &&
            read_counter(p / "device/mem_info_gtt_total")) return (p / "device").string();
    }
    return {};
}

std::vector<DeviceFacts> probe_devices(const std::string& root, const std::string& override_device) {
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
