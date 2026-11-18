// CUDA backend, architecture layer: what the current device can do, cached per device ordinal. Kernels choose
// an architecture path through these queries (capability), never by model. The specialisations themselves live in
// src/cuda/arch/<sm> (see src/cuda/arch/sm120/README.md).
#pragma once

#include "strix/device/runtime.hpp"

#include <cuda_runtime.h>
#include <atomic>

namespace strix::cuda::arch {

// True when the current device's compute capability major is >= Major. Cached per device ordinal (0..63); a failed
// API call or an ordinal outside the cache answers false (fail closed: the caller keeps its portable path).
template <int Major>
inline bool current_device_at_least() {
    static std::atomic<int> cap[64]{};  // 0 unknown, 1 yes, 2 no
    int dev = -1, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    // Only the capability value is published, not other memory. Concurrent first
    // queries may duplicate the API call; a failed query still returns false.
    int value = cap[dev].load(std::memory_order_relaxed);
    if (value == 0) {
        value = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && major >= Major ? 1 : 2;
        cap[dev].store(value, std::memory_order_relaxed);
    }
    return value == 1;
}

// Programmatic dependent launch (cudaLaunchAttributeProgrammaticStreamSerialization): compute capability 9.0+.
inline bool has_pdl() { return current_device_at_least<9>(); }
// Thread block clusters (cudaLaunchAttributeClusterDimension): compute capability 9.0+.
inline bool has_clusters() { return current_device_at_least<9>(); }

// Query the bound ordinal without changing the current device; errors fail closed.
inline ::strix::device::Capabilities capabilities(int ordinal) noexcept {
    ::strix::device::Capabilities result;
    if (cudaDeviceGetAttribute(&result.major, cudaDevAttrComputeCapabilityMajor, ordinal) != cudaSuccess ||
        cudaDeviceGetAttribute(&result.minor, cudaDevAttrComputeCapabilityMinor, ordinal) != cudaSuccess) return {};
    result.pdl = result.major >= 9;
    result.clusters = result.major >= 9;
    return result;
}

inline ::strix::device::Runtime runtime(const ::strix::device::Descriptor& descriptor, int ordinal) noexcept {
    return {descriptor, ordinal, capabilities};
}

}  // namespace strix::cuda::arch
