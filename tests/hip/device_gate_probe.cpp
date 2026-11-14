// Device probe for a HIP node: lists what the runtime reports, then runs the start gate and prints the receipt.
// Usage: strix-hip-device-gate-probe [--devices FILE] [--device-id ID]. Exit 0 = bound, 2 = refused.
#include "strix/device/hip_enumerator.hpp"
#include "strix/device/hip_gate.hpp"

#include <cstdio>
#include <exception>

int main(int argc, char** argv) {
    try {
        strix::device::HipEnumerator enumerator;
        for (const auto& v : enumerator.enumerate())
            std::printf("visible: ordinal=%d name=\"%s\" pci=%s arch=%s memory_bytes=%llu integrated=%d\n", v.ordinal, v.name.c_str(), v.pci.c_str(),
                        v.arch.c_str(), static_cast<unsigned long long>(v.memory_bytes), v.integrated ? 1 : 0);
        std::fflush(stdout);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
    strix::device::bind_hip_main(argc, argv, stdout);
    return 0;
}
