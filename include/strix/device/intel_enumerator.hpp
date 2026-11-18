#pragma once
// Intel GPU side of the device gate (vendor intel): Arc B-series (Xe2, BMG), Lunar Lake (Xe2, LNL) and Panther Lake
// (Xe3, PTL) GPUs. Two sources, as for NVIDIA (nvidia_smi.cpp + cuda_enumerator.hpp):
//   1. xpu-smi (host only, src/device/xpu_smi.cpp): lists devices with UUID, PCI address and memory without loading the
//      SYCL runtime, so a gate can pin ONEAPI_DEVICE_SELECTOR before the runtime starts.
//   2. SyclEnumerator (STRIX_HAVE_SYCL only): what the Level Zero runtime exposes, used to confirm the pinned device.
// The arch string is derived from the PCI device id ("xe2-bmg", "xe2-lnl", "xe3-ptl"); the backend selects its
// specialisation by that capability, never by model.
#include "strix/device/gate.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace strix::device {

// Arch family for an Intel GPU PCI device id; empty when the id is not a known Xe2/Xe3 part.
[[nodiscard]] std::string intel_arch_from_device_id(std::uint32_t pci_device_id);
// True for the integrated families (memory is host LPDDR5X, as on Strix Halo).
[[nodiscard]] bool intel_arch_integrated(std::string_view arch) noexcept;

// Parses `xpu-smi discovery --dump ...` CSV. Header driven: the columns are found by name ("Device ID", "Device Name",
// "UUID", "PCI BDF Address", "PCI Device ID", "Memory Physical Size"), so the field order may change between releases.
// Throws std::invalid_argument on a malformed listing.
[[nodiscard]] std::vector<Visible> parse_xpu_smi(std::string_view csv);
[[nodiscard]] std::optional<std::vector<Visible>> query_xpu_smi();

// ONEAPI_DEVICE_SELECTOR value that exposes only one Level Zero device to the SYCL runtime.
[[nodiscard]] std::string level_zero_selector(int ordinal);

} // namespace strix::device

#if defined(STRIX_HAVE_SYCL) && STRIX_HAVE_SYCL
#include <sycl/sycl.hpp>

namespace strix::device {

class SyclEnumerator final : public Enumerator {
public:
    [[nodiscard]] Vendor vendor() const noexcept override { return Vendor::intel; }
    [[nodiscard]] std::vector<Visible> enumerate() override {
        std::vector<Visible> out;
        int ordinal = 0;
        for (const auto& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
            if (d.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
            Visible v;
            v.ordinal = ordinal++;
            v.vendor = Vendor::intel;
            v.name = d.get_info<sycl::info::device::name>();
            v.memory_bytes = d.get_info<sycl::info::device::global_mem_size>();
            if (d.has(sycl::aspect::ext_intel_device_info_uuid)) {
                const auto id = d.get_info<sycl::ext::intel::info::device::uuid>();
                v.uuid = uuid_from_bytes(id.data());
            }
            if (d.has(sycl::aspect::ext_intel_pci_address))
                v.pci = normalise_pci(d.get_info<sycl::ext::intel::info::device::pci_address>());
            if (d.has(sycl::aspect::ext_intel_device_id))
                v.arch = intel_arch_from_device_id(d.get_info<sycl::ext::intel::info::device::device_id>());
            v.integrated = intel_arch_integrated(v.arch);
            out.push_back(std::move(v));
        }
        return out;
    }
};

} // namespace strix::device
#endif
