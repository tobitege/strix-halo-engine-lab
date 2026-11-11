// The deployment probe of a Windows HIP build (STRIX_DEPLOY_HIP_PROBE): facts from the HIP runtime of the GPU
// this runs on. Needs one visible AMD GPU.
#include "strix/deploy/device.hpp"

#include <hip/hip_runtime_api.h>

#include <iostream>

namespace {
int failures = 0;
void expect(bool condition, const char* what) {
    if (!condition) { std::cerr << "FAIL: " << what << '\n'; ++failures; }
}
}

int main() {
    const auto d = strix::deploy::probe_device();
    std::cout << "probe: source=" << d.source << " gfx_target=" << d.gfx_target << " version=" << d.gfx_target_version
              << " cu=" << d.cu_count << " unified=" << d.unified << " total=" << d.mem_total_bytes
              << " free=" << d.mem_free_bytes << '\n';
    expect(d.source == "hip+dxgi", "the facts come from the HIP runtime and DXGI");
    expect(!d.gfx_target.empty() && d.gfx_target_version == strix::deploy::gfx_target_version(d.gfx_target),
        "architecture name and KFD version agree");
    expect(d.cu_count > 0, "compute units are reported");
    expect(d.mem_total_bytes > 0 && d.mem_free_bytes <= d.mem_total_bytes, "memory is reported and consistent");
    const auto& total = d.unified ? d.gtt_total_bytes : d.vram_total_bytes;
    const auto& used = d.unified ? d.gtt_used_bytes : d.vram_used_bytes;
    expect(total && used && *total == d.mem_total_bytes && *used == d.mem_total_bytes - d.mem_free_bytes,
        "the memory kind's counters match the totals, as the sysfs probe fills them");
    // The local segment is the GPU's own memory: smaller than what hipMemGetInfo reports, which adds the
    // shared segment Windows backs with system RAM.
    std::size_t hip_free = 0, hip_total = 0;
    if (hipMemGetInfo(&hip_free, &hip_total) == hipSuccess)
        expect(d.mem_total_bytes < hip_total, "the memory excludes the shared segment");
    if (failures) return 1;
    std::cout << "deploy HIP probe tests passed\n";
}
