#include "strix/device/intel_enumerator.hpp"

#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define STRIX_POPEN _popen
#define STRIX_PCLOSE _pclose
#define STRIX_NULL_DEVICE "NUL"
#else
#define STRIX_POPEN popen
#define STRIX_PCLOSE pclose
#define STRIX_NULL_DEVICE "/dev/null"
#endif

namespace strix::device {
namespace {
std::string trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t' || text.front() == '"')) text.remove_prefix(1);
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r' || text.back() == '"'))
        text.remove_suffix(1);
    return std::string(text);
}

std::vector<std::string> split(std::string_view line) {
    std::vector<std::string> out;
    std::string field;
    bool quoted = false;
    for (const char c : line) {
        if (c == '"') quoted = !quoted;
        else if (c == ',' && !quoted) {
            out.push_back(trim(field));
            field.clear();
        } else field.push_back(c);
    }
    out.push_back(trim(field));
    return out;
}

bool lower_contains(std::string_view text, std::string_view needle) {
    if (needle.size() > text.size()) return false;
    for (std::size_t i = 0; i + needle.size() <= text.size(); ++i) {
        bool ok = true;
        for (std::size_t j = 0; j < needle.size() && ok; ++j) {
            char a = text[i + j];
            if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
            ok = a == needle[j];
        }
        if (ok) return true;
    }
    return false;
}

std::uint64_t parse_memory(const std::string& text) {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (!std::isfinite(value) || value < 0.0) throw std::invalid_argument("memory");
    const auto unit = trim(std::string_view(text).substr(used));
    double scale = 1.0;
    if (unit.empty() || lower_contains(unit, "mib") || lower_contains(unit, "mb")) scale = 1024.0 * 1024.0;
    else if (lower_contains(unit, "gib") || lower_contains(unit, "gb")) scale = 1024.0 * 1024.0 * 1024.0;
    else if (lower_contains(unit, "kib") || lower_contains(unit, "kb")) scale = 1024.0;
    else throw std::invalid_argument("memory unit");
    const double rounded = std::floor(value * scale + 0.5);
    // 2^64 is an exact exclusive bound; double(UINT64_MAX) rounds up to it.
    if (!std::isfinite(rounded) || rounded >= 0x1p64) throw std::invalid_argument("memory range");
    return static_cast<std::uint64_t>(rounded);
}
} // namespace

std::string intel_arch_from_device_id(std::uint32_t id) {
    // Family by the high byte of the PCI device id (Linux i915/xe and Windows driver tables).
    switch (id & 0xff00U) {
        case 0xe200U: return "xe2-bmg"; // Arc B-series discrete (Battlemage)
        case 0x6400U: return "xe2-lnl"; // Lunar Lake integrated (Arc 130V/140V)
        case 0xb000U: return "xe3-ptl"; // Panther Lake integrated (Arc B370/B390)
        default: return {};
    }
}

bool intel_arch_integrated(std::string_view arch) noexcept { return arch == "xe2-lnl" || arch == "xe3-ptl"; }

std::vector<Visible> parse_xpu_smi(std::string_view csv) {
    std::vector<Visible> out;
    int c_index = -1, c_name = -1, c_uuid = -1, c_pci = -1, c_devid = -1, c_mem = -1;
    bool header = false;
    while (!csv.empty()) {
        const auto end = csv.find('\n');
        const auto line = csv.substr(0, end);
        csv.remove_prefix(end == std::string_view::npos ? csv.size() : end + 1);
        if (trim(line).empty()) continue;
        const auto fields = split(line);
        if (!header) {
            for (int i = 0; i < static_cast<int>(fields.size()); ++i) {
                const auto& f = fields[static_cast<std::size_t>(i)];
                if (lower_contains(f, "pci device id")) c_devid = i;
                else if (lower_contains(f, "device id")) c_index = i;
                else if (lower_contains(f, "device name")) c_name = i;
                else if (lower_contains(f, "uuid")) c_uuid = i;
                else if (lower_contains(f, "pci bdf")) c_pci = i;
                else if (lower_contains(f, "memory physical size")) c_mem = i;
            }
            if (c_index < 0 || c_name < 0) throw std::invalid_argument("xpu-smi: header without Device ID and Device Name");
            header = true;
            continue;
        }
        const auto at = [&](int c) -> std::string { return c >= 0 && c < static_cast<int>(fields.size()) ? fields[static_cast<std::size_t>(c)] : std::string{}; };
        Visible v;
        v.vendor = Vendor::intel;
        try {
            std::size_t used = 0;
            const auto index = at(c_index);
            v.ordinal = std::stoi(index, &used);
            if (used != index.size() || v.ordinal < 0) throw std::invalid_argument("index");
            if (!at(c_uuid).empty()) v.uuid = normalise_uuid(at(c_uuid));
            if (!at(c_pci).empty()) v.pci = normalise_pci(at(c_pci));
            if (!at(c_mem).empty()) v.memory_bytes = parse_memory(at(c_mem));
            if (!at(c_devid).empty()) v.arch = intel_arch_from_device_id(static_cast<std::uint32_t>(std::stoul(at(c_devid), nullptr, 16)));
        } catch (const std::exception& e) {
            throw std::invalid_argument("xpu-smi: unexpected line (" + std::string(e.what()) + ")");
        }
        v.name = at(c_name);
        v.integrated = intel_arch_integrated(v.arch);
        out.push_back(std::move(v));
    }
    return out;
}

std::optional<std::vector<Visible>> query_xpu_smi() {
    // 1 Device ID, 2 Device Name, 4 UUID, 11 PCI BDF Address, 18 Memory Physical Size, 9 PCI Device ID (xpu-smi ids).
    std::FILE* pipe = STRIX_POPEN("xpu-smi discovery --dump 1,2,4,9,11,18 2>" STRIX_NULL_DEVICE, "r");
    if (!pipe) return std::nullopt;
    std::string text;
    char buffer[512];
    while (std::fgets(buffer, sizeof buffer, pipe)) text += buffer;
    if (STRIX_PCLOSE(pipe) != 0) return std::nullopt;
    try {
        auto devices = parse_xpu_smi(text);
        if (devices.empty()) return std::nullopt;
        return devices;
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
}

std::string level_zero_selector(int ordinal) { return "level_zero:" + std::to_string(ordinal); }

} // namespace strix::device
