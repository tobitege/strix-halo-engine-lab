#pragma once
#include <cstddef>
#include <cstdlib>
#include <set>
#define __host__
#define __device__
using cudaStream_t = void*;
using cudaError_t = int;
inline constexpr int cudaSuccess = 0, cudaErrorInvalidValue = 1;
inline std::set<void*> live_allocations;
inline bool fail_allocation = false;
inline int current_device = 0;
inline int cudaMalloc(void** p, std::size_t n) {
    if (fail_allocation) return cudaErrorInvalidValue;
    *p = std::malloc(n);
    if (!*p) return cudaErrorInvalidValue;
    live_allocations.insert(*p);
    return cudaSuccess;
}
inline int cudaFree(void* p) {
    if (live_allocations.erase(p) != 1) std::abort();
    std::free(p);
    return cudaSuccess;
}
inline int cudaGetDevice(int* device) { *device = current_device; return cudaSuccess; }
