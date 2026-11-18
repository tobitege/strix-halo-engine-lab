#include "strix/device/profiles.hpp"

#include <iomanip>
#include <sstream>

namespace strix::device {
namespace {

constexpr std::uint64_t MiB = 1024ULL * 1024ULL;
constexpr std::uint64_t GiB = 1024ULL * MiB;

// NVIDIA tensor peaks: dense, derived from the stated FP4 (Blackwell) / FP8 (Ada) 2:4-sparse "AI TOPS" (halved for
// dense, halved again per wider format, FP16-accumulate rate); GeForce cards run BF16 / FP32-accumulate at half that
// rate, the RTX PRO and GB10 parts at full rate. memory_bytes for NVIDIA cards = the runtime's reported total
// (nvidia-smi memory.total), not the marketing GB.
const std::vector<Profile> kCatalogue = {
    {"rtx-pro-6000-blackwell-96", "NVIDIA RTX PRO 6000 Blackwell (96 GB)", "nvidia", "sm_120", 102641958912ULL, 1792,
     {125, 500, 500, 1000, 2000, 1000}, 188, "SM", 5, 16, "pcie5x16", false, "memory_bytes,units",
     "our card; memory_bytes and units read from the runtime, peaks public spec"},
    {"rtx-5090-32", "NVIDIA GeForce RTX 5090 (32 GB)", "nvidia", "sm_120", 32607 * MiB, 1792,
     {104.8, 419, 209.5, 838, 1676, 838}, 170, "SM", 5, 16, "pcie5x16", false, "",
     "same SM generation as the RTX PRO 6000 with 170 of its 188 SMs: about 10 % less compute at equal memory speed"},
    {"rtx-5080-16", "NVIDIA GeForce RTX 5080 (16 GB)", "nvidia", "sm_120", 16303 * MiB, 960,
     {56.3, 225.1, 112.6, 450.2, 900.5, 450.2}, 84, "SM", 5, 16, "pcie5x16", false, "", ""},
    {"rtx-4090-24", "NVIDIA GeForce RTX 4090 (24 GB)", "nvidia", "sm_89", 24564 * MiB, 1008,
     {82.6, 330.3, 165.2, 660.6, 0, 660.6}, 128, "SM", 4, 16, "pcie4x16", false, "",
     "profile only (planning); no FP4 tensor path on sm_89"},
    {"strix-halo-128", "AMD Ryzen AI Max+ 395 / Radeon 8060S (128 GB)", "amd", "gfx1151", 125829120000ULL, 256,
     {29.7, 59.4, 59.4, 0, 0, 59.4}, 40, "CU", 0, 0, "integrated", true, "memory_bytes",
     "our nodes; memory_bytes = GPU-visible memory as configured there (depends on the GTT setting); "
     "HIP reports 20 compute units (WGPs); measured stream read about 243 GB/s; WMMA FP16 = BF16 = IU8 rate"},
    {"dgx-spark-gb10-128", "NVIDIA DGX Spark GB10 (128 GB)", "nvidia", "sm_121", 128 * GiB, 273,
     {0, 125, 125, 250, 500, 250}, 48, "SM", 0, 0, "nvlink-c2c", true, "",
     "unified LPDDR5X shared with the CPU; usable GPU memory is lower than 128 GiB"},
    {"rx-9070-xt-16", "AMD Radeon RX 9070 XT (16 GB, RDNA4)", "amd", "gfx1201", 16 * GiB, 640,
     {48.7, 195, 195, 389, 0, 389}, 64, "CU", 5, 16, "pcie5x16", false, "", "RDNA4 discrete; WMMA FP8 and IU8 at 2x FP16"},
    {"custom", "custom device (every value from the declaration entry)", "", "", 0, 0, {}, 0, "", 0, 0, "", false, "",
     "set memory_gib (and optionally bandwidth_gbytes_ps, units) in the entry"},
};

const std::vector<Interconnect> kInterconnects = {
    {"pcie5x16", 50.0, 5.0, "peer copies through the host bridge, no NVLink"},
    {"pcie4x16", 25.0, 5.0, "peer copies through the host bridge"},
    {"pcie4x4", 7.0, 5.0, ""},
    {"oculink4x4", 7.0, 5.0, ""},
    {"pcie3x4", 3.2, 5.0, ""},
    {"usb4", 4.58, 30.0, "measured between our two nodes"},
    {"100gbe", 11.5, 20.0, ""},
    {"25gbe", 2.8, 60.0, ""},
    {"10gbe", 1.1, 100.0, ""},
    {"2.5gbe", 0.29, 150.0, ""},
    {"1gbe", 0.118, 200.0, ""},
};

std::string num(double v) {
    std::ostringstream os;
    os << std::setprecision(10) << v;
    return os.str();
}

std::string quoted(std::string_view s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out + "\"";
}

} // namespace

const std::vector<Profile>& catalogue() { return kCatalogue; }

const Profile* find_profile(std::string_view name) noexcept {
    for (const auto& p : kCatalogue)
        if (p.name == name) return &p;
    return nullptr;
}

std::string profile_names() {
    std::string out;
    for (const auto& p : kCatalogue) out += (out.empty() ? "" : ", ") + std::string(p.name);
    return out;
}

const std::vector<Interconnect>& interconnects() { return kInterconnects; }

const Interconnect* find_interconnect(std::string_view name) noexcept {
    for (const auto& i : kInterconnects)
        if (i.name == name) return &i;
    return nullptr;
}

std::string catalogue_toml() {
    std::ostringstream os;
    os << "# Generated from src/device/profiles.cpp (strix::device::catalogue_toml); do not edit by hand.\n"
          "# Peaks are dense TFLOPS / TOPS; 0 = no tensor path or not stated. `measured` lists the fields read on our\n"
          "# own devices; everything else is the vendor's public specification.\n"
          "schema = \"strix-device-profiles-v1\"\n";
    for (const auto& p : kCatalogue) {
        os << "\n[profiles." << p.name << "]\n"
           << "display = " << quoted(p.display) << "\n"
           << "vendor = " << quoted(p.vendor) << "\n"
           << "arch = " << quoted(p.arch) << "\n"
           << "memory_bytes = " << p.memory_bytes << "\n"
           << "memory_gbytes_ps = " << num(p.memory_gbytes_ps) << "\n"
           << "peaks = { fp32 = " << num(p.peaks.fp32) << ", fp16 = " << num(p.peaks.fp16) << ", bf16 = " << num(p.peaks.bf16)
           << ", fp8 = " << num(p.peaks.fp8) << ", fp4 = " << num(p.peaks.fp4) << ", int8 = " << num(p.peaks.int8) << " }\n"
           << "units = " << p.units << "\n"
           << "unit_kind = " << quoted(p.unit_kind) << "\n"
           << "pcie = { gen = " << p.pcie_gen << ", width = " << p.pcie_width << " }\n"
           << "host_link = " << quoted(p.host_link) << "\n"
           << "unified = " << (p.unified ? "true" : "false") << "\n"
           << "measured = " << quoted(p.measured) << "\n";
        if (!p.note.empty()) os << "note = " << quoted(p.note) << "\n";
    }
    os << "\n# Interconnect classes for `count = N` shapes: nominal one-direction GB/s and round trip (us).\n";
    for (const auto& i : kInterconnects)
        os << "[interconnects." << quoted(i.name) << "]\ngbytes_ps = " << num(i.gbytes_ps) << "\nrtt_us = " << num(i.rtt_us)
           << "\n";
    return os.str();
}

} // namespace strix::device
