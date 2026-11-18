#pragma once
// deepseek41 CUDA prefill: what the run's sizes are decided from, queried or measured on the device at hand (no
// device is assumed): name, compute capability, SM count, total and free memory, PCIe link (NVML when the driver
// library is present) and the sustained pinned host -> device bandwidth of a short probe, cached per GPU in a small
// JSON file (strix-gpu-<uuid>.json under %LOCALAPPDATA%\strix, $XDG_CACHE_HOME/strix or ~/.cache/strix).
#include <cstddef>
#include <string>

namespace strix::ds41::cuda {

struct GpuProfile {
    int device = 0;
    std::string name, uuid, pci_bus_id;
    int cc_major = 0, cc_minor = 0, sms = 0;
    std::size_t total_bytes = 0, free_bytes = 0;
    int pcie_gen = -1, pcie_width = -1, pcie_max_gen = -1, pcie_max_width = -1;  // -1: unknown (no NVML)
    double h2d_gbps = 0.0;    // sustained pinned host -> device, 64 MiB pieces from a 256 MiB buffer
    bool h2d_cached = false;  // read from the cache file instead of measured now
    std::string cache_file;
};

// probe: measure the H2D bandwidth when the cache has no value, or it is older than a day (always with force_probe or
// STRIX_GPU_PROBE=force); seconds: probe length (the best of four windows).
GpuProfile query_gpu_profile(int device, bool probe = true, bool force_probe = false, double seconds = 1.0,
                             const std::string& cache_dir = {});
[[nodiscard]] std::string to_json(const GpuProfile& p);

}  // namespace strix::ds41::cuda
