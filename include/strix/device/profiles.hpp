#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// strix-device-profiles-v1: the catalogue of known device models. A device entry in the declaration names one with
// `profile = "rtx-5090-32"` and every route sizes itself from it (strix/device/budget.hpp); a cluster of identical
// devices is `count = N` plus an `interconnect` class. The table in src/device/profiles.cpp is the source of truth;
// data/devices/profiles.toml is its printed form (catalogue_toml(), checked byte-equal by the host tests).
//
// Values are vendor public specifications unless `measured` lists the field: peaks are dense (no 2:4 sparsity),
// in TFLOPS / TOPS, 0 = the format has no tensor path on the device or the vendor does not state it. A profile is a
// planning input, never a qualification: measured receipts (bandwidth, clocks) win where a route has them.
namespace strix::device {

struct Peaks {
    double fp32 = 0, fp16 = 0, bf16 = 0, fp8 = 0, fp4 = 0, int8 = 0; // dense TFLOPS / TOPS
};

struct Profile {
    std::string_view name;          // catalogue key: <model>-<memory GB>
    std::string_view display;       // marketing name
    std::string_view vendor;        // amd | nvidia
    std::string_view arch;          // sm_120 | sm_89 | sm_121 | gfx1151 | gfx1201
    std::uint64_t memory_bytes = 0; // device-visible memory the runtime reports (what a budget caps to)
    double memory_gbytes_ps = 0;    // memory bandwidth, GB/s
    Peaks peaks;
    std::uint32_t units = 0;        // SMs (nvidia) or CUs (amd)
    std::string_view unit_kind;     // "SM" | "CU"
    std::uint32_t pcie_gen = 0, pcie_width = 0; // 0 = integrated (no PCIe link to the host)
    std::string_view host_link;     // "pcie5x16" | "pcie4x16" | "integrated" | "nvlink-c2c"
    bool unified = false;           // device memory is the host memory (integrated APU / superchip)
    std::string_view measured;      // comma list of fields measured on our own device; "" = all public spec
    std::string_view note;
};

// The whole catalogue, in a fixed order. "custom" is the empty profile: every value comes from the entry.
[[nodiscard]] const std::vector<Profile>& catalogue();
// nullptr when unknown.
[[nodiscard]] const Profile* find_profile(std::string_view name) noexcept;
// Comma-separated catalogue names, for refusals ("unknown profile x; known: a, b, ...").
[[nodiscard]] std::string profile_names();
// The data file (data/devices/profiles.toml): one [profiles.<name>] table per entry.
[[nodiscard]] std::string catalogue_toml();

// Interconnect classes for N identical devices (`interconnect = "pcie5x16"`): nominal one-direction GB/s and round
// trip. Names follow the deploy link classes (pcieGxW, NgbE, usb4); "nvlink" is accepted only where the profile has
// it (no GeForce / RTX PRO Blackwell card does).
struct Interconnect {
    std::string_view name;
    double gbytes_ps = 0;
    double rtt_us = 0;
    std::string_view note;
};
[[nodiscard]] const std::vector<Interconnect>& interconnects();
[[nodiscard]] const Interconnect* find_interconnect(std::string_view name) noexcept;

} // namespace strix::device
