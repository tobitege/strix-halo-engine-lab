#pragma once
// Sequential visibility model, not native HIP evidence. Copy completion and
// CPU publication are separate so a device-only event cannot hide the defect.
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <random>
#include <vector>

inline constexpr unsigned hipEventDisableTiming = 2U;
inline constexpr unsigned hipEventReleaseToSystem = 0x80000000U;
inline constexpr unsigned hipHostMallocDefault = 0U;
inline constexpr unsigned hipStreamNonBlocking = 1U;
using hipError_t = int;
inline constexpr hipError_t hipSuccess = 0;
enum hipMemcpyKind { hipMemcpyDeviceToDevice, hipMemcpyDeviceToHost, hipMemcpyHostToDevice };
struct FakeSeamCopy {
    void* host;
    std::vector<std::byte> bytes;
    bool complete = false;
};
struct FakeSeamStream { std::vector<FakeSeamCopy> copies; };
using hipStream_t = FakeSeamStream*;
struct FakeSeamEvent { unsigned flags; hipStream_t stream = nullptr; std::size_t end = 0; };
using hipEvent_t = FakeSeamEvent*;
namespace seam_visibility {
inline bool device_only = false;
inline void publish(hipStream_t stream, std::size_t end, bool system) {
    if (!stream) return;
    for (std::size_t index = 0; index < end; ++index) {
        auto& copy = stream->copies.at(index);
        copy.complete = true;
        if (system) std::memcpy(copy.host, copy.bytes.data(), copy.bytes.size());
    }
}
inline void complete_shuffled(hipStream_t stream, std::mt19937& random) {
    std::vector<std::size_t> order;
    for (std::size_t index = 0; index < stream->copies.size(); ++index) order.push_back(index);
    std::shuffle(order.begin(), order.end(), random);
    for (const auto index : order) stream->copies[index].complete = true;
}
inline hipStream_t last_stream = nullptr;
}
inline const char* hipGetErrorString(hipError_t) { return "simulated HIP error"; }
inline hipError_t hipSetDevice(int) { return hipSuccess; }
inline hipError_t hipGetDevice(int* device) { *device = 0; return hipSuccess; }
inline hipError_t hipMalloc(void** address, std::size_t bytes) {
    *address = new std::byte[bytes]; return hipSuccess;
}
inline hipError_t hipHostMalloc(void** address, std::size_t bytes, unsigned) {
    *address = new std::byte[bytes]; std::memset(*address, 0xa5, bytes); return hipSuccess;
}
inline hipError_t hipFree(void* address) { delete[] static_cast<std::byte*>(address); return hipSuccess; }
inline hipError_t hipHostFree(void* address) { return hipFree(address); }
inline hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned) {
    *stream = new FakeSeamStream; seam_visibility::last_stream = *stream; return hipSuccess;
}
inline hipError_t hipStreamSynchronize(hipStream_t stream) {
    if (stream) seam_visibility::publish(stream, stream->copies.size(), true);
    return hipSuccess;
}
inline hipError_t hipStreamDestroy(hipStream_t stream) { delete stream; return hipSuccess; }
inline hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
    *event = new FakeSeamEvent{seam_visibility::device_only ? flags & ~hipEventReleaseToSystem : flags};
    return hipSuccess;
}
inline hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
    event->stream = stream; event->end = stream ? stream->copies.size() : 0U; return hipSuccess;
}
inline hipError_t hipEventSynchronize(hipEvent_t event) {
    seam_visibility::publish(event->stream, event->end, (event->flags & hipEventReleaseToSystem) != 0U);
    return hipSuccess;
}
inline hipError_t hipEventDestroy(hipEvent_t event) { delete event; return hipSuccess; }
inline hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned) { return hipSuccess; }
inline hipError_t hipMemcpyAsync(void* destination, const void* source, std::size_t bytes,
    hipMemcpyKind kind, hipStream_t stream) {
    if (kind == hipMemcpyDeviceToHost && stream) {
        const auto* begin = static_cast<const std::byte*>(source);
        stream->copies.push_back({destination, {begin, begin + bytes}, false});
    } else std::memcpy(destination, source, bytes);
    return hipSuccess;
}
// Row-by-row strided copy (device-to-device only in the staging under test).
inline hipError_t hipMemcpy2DAsync(void* destination, std::size_t destination_pitch, const void* source,
    std::size_t source_pitch, std::size_t width, std::size_t height, hipMemcpyKind kind, hipStream_t stream) {
    for (std::size_t row = 0; row < height; ++row) {
        const auto status = hipMemcpyAsync(static_cast<std::byte*>(destination) + row * destination_pitch,
            static_cast<const std::byte*>(source) + row * source_pitch, width, kind, stream);
        if (status != hipSuccess) return status;
    }
    return hipSuccess;
}
