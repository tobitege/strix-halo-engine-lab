#pragma once
// CRC-32C of device memory without bringing the bytes to the host: one kernel launch writes one digest per 64 KiB
// span, only the digests (4 bytes per span) and the short tail are copied to the host, and the host folds them
// (xport/crc32c_fold.hpp). The value is the host CRC-32C of the same bytes, bit for bit.
#include "xport/crc32c_fold.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace strix::xport {
class DeviceCrc32c {
public:
    DeviceCrc32c() = default;
    ~DeviceCrc32c();
    DeviceCrc32c(const DeviceCrc32c&) = delete;
    DeviceCrc32c& operator=(const DeviceCrc32c&) = delete;
    // Digest storage for runs inside a buffer of `capacity` bytes (one slot per span of the buffer). Current device.
    void allocate(std::size_t capacity);
    [[nodiscard]] bool allocated() const noexcept { return tables_ != nullptr; }
    // Enqueues on `stream` the digests of the whole spans of base[offset, offset + bytes) and their copy to pinned
    // host memory. `base` is 16-byte aligned device memory, `offset` a multiple of the span. Runs of different
    // offsets may be in flight together; the caller fences the stream (an event) before value().
    void enqueue(hipStream_t stream, const void* base, std::size_t offset, std::size_t bytes);
    // After the stream passed enqueue(): the CRC-32C of the run. `tail`: the host copy of the bytes after its last
    // whole span (bytes % span of them).
    [[nodiscard]] std::uint32_t value(std::size_t offset, std::size_t bytes, std::span<const std::byte> tail) const;
    [[nodiscard]] static std::size_t tail_bytes(std::size_t bytes) noexcept { return bytes % crc32c_fold_span_bytes; }
private:
    std::uint32_t* tables_ = nullptr;        // device: slicing tables
    std::uint32_t* digests_ = nullptr;       // device: one per span of the buffer
    std::uint32_t* host_digests_ = nullptr;  // pinned host copy
    std::size_t spans_ = 0;
};
} // namespace strix::xport
