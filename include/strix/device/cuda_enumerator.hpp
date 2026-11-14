#pragma once

#include "strix/device/gate.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

// CUDA side of the start gate. bind_cuda() is the one call a CUDA binary makes before its first allocation.
//
// The current CUDA device is per host thread and a new thread starts on index 0, so selecting a device in one thread
// is not enough. The gate therefore decides before the CUDA runtime starts (it lists the cards with nvidia-smi),
// restricts the process to the chosen card by UUID (CUDA_VISIBLE_DEVICES), and then checks with the CUDA runtime that
// index 0 is that card. Every thread and every library in the process can then only reach the declared device.
namespace strix::device {

// What the CUDA runtime exposes to this process. Properties only: no context is created on any device.
class CudaEnumerator final : public Enumerator {
public:
    [[nodiscard]] Vendor vendor() const noexcept override { return Vendor::nvidia; }
    [[nodiscard]] std::vector<Visible> enumerate() override {
        int count = 0;
        if (const auto status = cudaGetDeviceCount(&count); status != cudaSuccess)
            throw Refusal(std::string("device gate: refusing to start: cannot enumerate CUDA devices: ") + cudaGetErrorString(status));
        std::vector<Visible> out;
        for (int i = 0; i < count; ++i) {
            cudaDeviceProp p{};
            if (const auto status = cudaGetDeviceProperties(&p, i); status != cudaSuccess)
                throw Refusal(std::string("device gate: refusing to start: cannot read CUDA device properties: ") + cudaGetErrorString(status));
            Visible v;
            v.ordinal = i;
            v.vendor = Vendor::nvidia;
            v.name = p.name;
            v.uuid = uuid_from_bytes(reinterpret_cast<const unsigned char*>(p.uuid.bytes));
            v.pci = pci_from_parts(static_cast<unsigned>(p.pciDomainID), static_cast<unsigned>(p.pciBusID), static_cast<unsigned>(p.pciDeviceID));
            v.arch = "sm_" + std::to_string(p.major) + std::to_string(p.minor);
            v.memory_bytes = p.totalGlobalMem;
            v.integrated = p.integrated != 0;
            out.push_back(std::move(v));
        }
        return out;
    }
};

namespace detail {
struct CudaProcess {
    std::mutex mutex;
    std::optional<Bound> bound;
};
inline CudaProcess& cuda_process() {
    static CudaProcess state;
    return state;
}
} // namespace detail

// Resolves the device once per process (throws Refusal), restricts the process to it, makes it current for the calling
// thread and prints the receipt line on `receipt` (null = silent). Later calls return the same binding and only make
// the device current for their thread; their arguments are ignored.
inline const Bound& bind_cuda(Request request = {}, const std::string& devices_path = {}, std::FILE* receipt = stderr) {
    auto& process = detail::cuda_process();
    const std::lock_guard<std::mutex> lock(process.mutex);
    const auto fail = [](const std::string& why) -> void { throw Refusal("device gate: refusing to start: " + why); };
    if (!process.bound) {
        request.vendor = Vendor::nvidia;
        Bound chosen;
        if (auto listed = query_nvidia_smi()) {
            FakeEnumerator cards(Vendor::nvidia, std::move(*listed));
            chosen = start_gate(cards, request, devices_path);
            set_process_environment("CUDA_VISIBLE_DEVICES", "GPU-" + chosen.visible.uuid);
        } else {
            CudaEnumerator cards;
            chosen = start_gate(cards, request, devices_path);
        }
        // The CUDA runtime is the authority: whatever the listing said, index 0 must be the declared card.
        CudaEnumerator runtime;
        const auto seen = runtime.enumerate();
        const Visible* match = nullptr;
        for (const auto& v : seen)
            if (!chosen.visible.uuid.empty() && v.uuid == chosen.visible.uuid) match = &v;
        const auto uuid = "GPU-" + chosen.visible.uuid;
        if (!match) fail("the CUDA runtime does not expose declared device " + chosen.device.id + " (" + uuid + ")");
        if (match->ordinal != 0)
            fail("declared device " + chosen.device.id + " is CUDA index " + std::to_string(match->ordinal) +
                 " and threads start on index 0: set CUDA_VISIBLE_DEVICES=" + uuid + " before starting");
        if (!chosen.device.arch.empty() && chosen.device.arch != match->arch)
            fail("device " + chosen.device.id + " is declared with arch " + chosen.device.arch + ", the device reports " + match->arch);
        chosen.visible = *match;
        if (const auto status = cudaSetDevice(0); status != cudaSuccess) fail(std::string("cudaSetDevice failed: ") + cudaGetErrorString(status));
        if (receipt) {
            std::fprintf(receipt, "%s\n", receipt_line(chosen).c_str());
            std::fflush(receipt);
        }
        process.bound = std::move(chosen);
    } else if (const auto status = cudaSetDevice(process.bound->visible.ordinal); status != cudaSuccess) {
        fail(std::string("cudaSetDevice failed: ") + cudaGetErrorString(status));
    }
    return *process.bound;
}

// For a main(): takes --devices FILE / --device-id ID out of the argument vector, binds, and returns the binding.
// On a refusal it prints the one-line reason and exits with status 2.
inline const Bound& bind_cuda_main(int& argc, char** argv, std::FILE* receipt = stderr) {
    try {
        const auto cli = take_command_line(argc, argv);
        Request request;
        request.select = cli.select;
        return bind_cuda(std::move(request), cli.devices_path, receipt);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        std::fflush(stderr);
        std::exit(2);
    }
}

} // namespace strix::device
