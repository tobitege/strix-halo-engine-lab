#pragma once

#include "strix/device/declaration.hpp"

#include <cstdint>
#include <optional>
#include <string>

// strix-memory-budget-v1: the ONE place a route learns how much device memory and host RAM it may plan for.
// Every GPU binary binds its card through the start gate (gate.hpp); the bound entry's `profile` / `memory_gib` and
// the cpu unit's `memory_gib` resolve here, once, and the same record reaches the CLI and the server of a family.
//
// Precedence, per side: an operator override (the routes' --vram-total-gib / --ram-total-gib and their env twins) >
// the entry's memory_gib > the profile's memory > no cap (today's path: the route uses what the card has free).
// A declared budget above the card's own memory is refused (a profile cannot add memory); an override is passed
// through as before. A budget below the card is an emulation: the route behaves as on the smaller device.
namespace strix::device {

struct BudgetOverrides {
    // nullopt = not given; <= 0 = given as "none" (no cap); > 0 = the cap in GiB.
    std::optional<double> vram_total_gib, ram_total_gib;
    std::string vram_from = "--vram-total-gib", ram_from = "--ram-total-gib";
};

struct MemoryBudget {
    std::uint64_t device_bytes = 0; // 0 = no cap
    std::uint64_t host_bytes = 0;   // 0 = no cap
    std::uint64_t physical_device_bytes = 0;
    std::string device_source = "none", host_source = "none";
    std::string device_id, profile;
    double bandwidth_gbytes_ps = 0; // the profile's (or the entry's override); 0 = unknown
    std::uint32_t units = 0;

    [[nodiscard]] bool device_capped() const noexcept { return device_bytes != 0; }
    [[nodiscard]] bool host_capped() const noexcept { return host_bytes != 0; }
    [[nodiscard]] bool emulated() const noexcept {
        return device_capped() && physical_device_bytes != 0 && device_bytes < physical_device_bytes;
    }
    // GiB values in the routes' flag convention: < 0 = no cap.
    [[nodiscard]] double device_gib() const noexcept { return device_capped() ? static_cast<double>(device_bytes) / 1073741824.0 : -1.0; }
    [[nodiscard]] double host_gib() const noexcept { return host_capped() ? static_cast<double>(host_bytes) / 1073741824.0 : -1.0; }
    [[nodiscard]] std::string json() const; // {"schema":"strix-memory-budget-v1",...}
};

// gpu: the bound (or planned) GPU entry; host: the declared cpu unit or nullptr; physical: the card's reported memory
// (0 = unknown, e.g. planning). Throws std::invalid_argument when a declared budget exceeds `physical`.
[[nodiscard]] MemoryBudget resolve_budget(const Descriptor& gpu, const Descriptor* host, std::uint64_t physical,
                                          const BudgetOverrides& overrides = {});

// The flag form every route shares: "none" -> <= 0, a positive number -> GiB; throws std::invalid_argument otherwise.
[[nodiscard]] double parse_total_gib(const std::string& flag, const std::string& text);

} // namespace strix::device
