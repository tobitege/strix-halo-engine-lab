#include "strix/device/budget.hpp"

#include "strix/device/profiles.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace strix::device {
namespace {

std::uint64_t gib_bytes(double gib) { return static_cast<std::uint64_t>(std::llround(gib * 1073741824.0)); }

std::string gib_text(std::uint64_t bytes) {
    std::ostringstream os;
    os.precision(4);
    os << static_cast<double>(bytes) / 1073741824.0;
    return os.str();
}

std::string json_text(const std::string& s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out + "\"";
}

} // namespace

double parse_total_gib(const std::string& flag, const std::string& text) {
    if (text == "none") return -1.0;
    std::size_t used = 0;
    double v = 0;
    try {
        v = std::stod(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != text.size() || !std::isfinite(v) || v <= 0.0)
        throw std::invalid_argument(flag + ": expected a positive GiB value or none, got '" + text + "'");
    return v;
}

MemoryBudget resolve_budget(const Descriptor& gpu, const Descriptor* host, std::uint64_t physical,
                            const BudgetOverrides& overrides) {
    MemoryBudget b;
    b.device_id = gpu.id;
    b.profile = gpu.profile;
    b.physical_device_bytes = physical;
    const Profile* p = gpu.profile.empty() ? nullptr : find_profile(gpu.profile);
    if (p) {
        b.bandwidth_gbytes_ps = p->memory_gbytes_ps;
        b.units = p->units;
    }
    if (gpu.bandwidth_gbytes_ps > 0) b.bandwidth_gbytes_ps = gpu.bandwidth_gbytes_ps;
    if (gpu.units > 0) b.units = gpu.units;

    if (overrides.vram_total_gib) {
        b.device_bytes = *overrides.vram_total_gib > 0 ? gib_bytes(*overrides.vram_total_gib) : 0;
        b.device_source = *overrides.vram_total_gib > 0 ? overrides.vram_from : overrides.vram_from + " none";
    } else if (gpu.budget_bytes) {
        b.device_bytes = gpu.budget_bytes;
        b.device_source = "devices." + gpu.id + ".memory_gib";
    } else if (p && p->memory_bytes) {
        b.device_bytes = p->memory_bytes;
        b.device_source = "profile " + gpu.profile;
    }
    if (!overrides.vram_total_gib && b.device_bytes && physical && b.device_bytes > physical)
        throw std::invalid_argument("device " + gpu.id + ": " + b.device_source + " is " + gib_text(b.device_bytes) +
                                    " GiB but the card reports " + gib_text(physical) +
                                    " GiB; a declared budget cannot exceed the card");

    if (overrides.ram_total_gib) {
        b.host_bytes = *overrides.ram_total_gib > 0 ? gib_bytes(*overrides.ram_total_gib) : 0;
        b.host_source = *overrides.ram_total_gib > 0 ? overrides.ram_from : overrides.ram_from + " none";
    } else if (host && host->budget_bytes) {
        b.host_bytes = host->budget_bytes;
        b.host_source = "devices." + host->id + ".memory_gib";
    }
    return b;
}

std::string MemoryBudget::json() const {
    std::ostringstream os;
    os << "{\"schema\":\"strix-memory-budget-v1\",\"device\":" << json_text(device_id) << ",\"profile\":" << json_text(profile)
       << ",\"device_bytes\":" << device_bytes << ",\"device_source\":" << json_text(device_source)
       << ",\"host_bytes\":" << host_bytes << ",\"host_source\":" << json_text(host_source)
       << ",\"physical_device_bytes\":" << physical_device_bytes << ",\"emulated\":" << (emulated() ? "true" : "false")
       << ",\"bandwidth_gbytes_ps\":" << bandwidth_gbytes_ps << ",\"units\":" << units << "}";
    return os.str();
}

} // namespace strix::device
