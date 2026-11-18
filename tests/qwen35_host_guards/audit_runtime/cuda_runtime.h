#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <mutex>
#include <vector>
#define __host__
#define __device__
using cudaStream_t = void*;
using cudaEvent_t = void*;
using cudaError_t = int;
inline constexpr int cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorNotSupported = 2;
inline constexpr int cudaStreamNonBlocking = 1, cudaMemcpyHostToDevice = 1;
inline thread_local int current_device = 0;
inline int fail_after = -1;
inline std::mutex runtime_mutex;
inline std::map<void*, int> live_allocations;
inline std::set<void*> streams, events;
inline std::vector<char> lifetime;
inline void runtime_require(bool ok) { if (!ok) { std::fputs("mock runtime contract failed\n", stderr); std::exit(2); } }
inline int cudaMalloc(void** p, std::size_t n) {
    std::lock_guard lock(runtime_mutex);
    if (fail_after == 0) return cudaErrorInvalidValue;
    if (fail_after > 0) --fail_after;
    *p = std::malloc(n);
    if (!*p) return cudaErrorInvalidValue;
    live_allocations.emplace(*p, current_device);
    return cudaSuccess;
}
inline int cudaFree(void* p) {
    std::lock_guard lock(runtime_mutex);
    auto it = live_allocations.find(p);
    runtime_require(it != live_allocations.end() && it->second == current_device);
    live_allocations.erase(it);
    lifetime.push_back('F');
    std::free(p);
    return cudaSuccess;
}
inline int cudaGetDevice(int* device) { *device = current_device; return cudaSuccess; }
inline int cudaSetDevice(int device) { current_device = device; return cudaSuccess; }
inline int cudaMemcpy(void* out, const void* in, std::size_t n, int) { std::memcpy(out, in, n); return cudaSuccess; }
inline int cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) {
    *s = std::malloc(1); streams.insert(*s); return cudaSuccess;
}
inline int cudaStreamSynchronize(cudaStream_t) { lifetime.push_back('S'); return cudaSuccess; }
inline int cudaStreamDestroy(cudaStream_t s) {
    runtime_require(streams.erase(s) == 1); lifetime.push_back('D'); std::free(s); return cudaSuccess;
}
inline int cudaEventDestroy(cudaEvent_t e) {
    runtime_require(events.erase(e) == 1); lifetime.push_back('E'); std::free(e); return cudaSuccess;
}
struct cudaDeviceProp { int major = 12, minor = 0; };
inline cudaDeviceProp device_prop;
inline int cudaGetDeviceProperties(cudaDeviceProp* p, int) { *p = device_prop; return cudaSuccess; }
