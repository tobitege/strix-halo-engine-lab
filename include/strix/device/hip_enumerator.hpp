#pragma once

#include "strix/device/gate.hpp"

#include <hip/hip_runtime.h>

#include <string>

// What the HIP runtime exposes to this process. Properties only: no allocation is made on any device.
// AMD devices are identified by PCI address. The binding itself is in strix/device/hip_gate.hpp.
namespace strix::device {

class HipEnumerator final : public Enumerator {
public:
    [[nodiscard]] Vendor vendor() const noexcept override { return Vendor::amd; }
    [[nodiscard]] std::vector<Visible> enumerate() override {
        int count = 0;
        if (const auto status = hipGetDeviceCount(&count); status != hipSuccess)
            throw Refusal(std::string("device gate: refusing to start: cannot enumerate HIP devices: ") + hipGetErrorString(status));
        std::vector<Visible> out;
        for (int i = 0; i < count; ++i) {
            hipDeviceProp_t p{};
            if (const auto status = hipGetDeviceProperties(&p, i); status != hipSuccess)
                throw Refusal(std::string("device gate: refusing to start: cannot read HIP device properties: ") + hipGetErrorString(status));
            Visible v;
            v.ordinal = i;
            v.vendor = Vendor::amd;
            v.name = p.name;
            v.pci = pci_from_parts(static_cast<unsigned>(p.pciDomainID), static_cast<unsigned>(p.pciBusID), static_cast<unsigned>(p.pciDeviceID));
            v.arch = p.gcnArchName;
            if (const auto colon = v.arch.find(':'); colon != std::string::npos) v.arch.erase(colon);
            v.memory_bytes = p.totalGlobalMem;
            v.integrated = p.integrated != 0;
            out.push_back(std::move(v));
        }
        return out;
    }
};

} // namespace strix::device
