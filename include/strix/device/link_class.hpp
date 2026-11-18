#pragma once

#include <array>
#include <cstdint>
#include <string_view>

// Link classes of the device declaration (attach.link, links.<name>.class). A class is the one-line way to state what
// connects two units; it carries NOMINAL useful rates, used only until a measured rate is declared, and a wider
// uncertainty than a measurement. Rates are useful payload GB/s (decimal), not the signalling rate.
namespace strix::device {

struct LinkClass {
    std::string_view name;
    double gbytes_ps = 0.0;    // nominal useful payload rate, one direction
    double rtt_us = 0.0;       // nominal round trip of one small message
    double uncertainty = 0.0;  // relative half-width of the nominal rate
};

// 1gbe is the measured useful rate of the cluster's 1 GbE path; 2.5/10 GbE are estimates; usb4 is the measured pair
// cable; oculink/pcie are per-direction PCIe payload estimates. Each is replaced by attach.gbps_measured when probed.
inline constexpr std::array<LinkClass, 12> link_classes{{
    {"1gbe", 0.118, 200.0, 0.25},   {"2.5gbe", 0.29, 150.0, 0.30},  {"5gbe", 0.58, 120.0, 0.30},
    {"10gbe", 1.1, 100.0, 0.30},    {"25gbe", 2.8, 60.0, 0.35},     {"usb4", 4.58, 30.0, 0.30},
    {"usb4-gen2", 1.1, 40.0, 0.50}, {"pcie3x4", 3.2, 5.0, 0.30},    {"pcie4x4", 7.0, 5.0, 0.30},
    {"oculink4x4", 7.0, 5.0, 0.30}, {"pcie4x16", 25.0, 5.0, 0.30},  {"pcie5x16", 50.0, 5.0, 0.30},
}};

[[nodiscard]] constexpr const LinkClass* find_link_class(std::string_view name) noexcept {
    for (const auto& c : link_classes)
        if (c.name == name) return &c;
    return nullptr;
}
[[nodiscard]] constexpr bool known_link_class(std::string_view name) noexcept { return find_link_class(name) != nullptr; }

} // namespace strix::device
