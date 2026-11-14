#pragma once
// HIP staging of the shared two-stage seam (strix/pair/pipe.hpp): an immutable device snapshot of the
// producer's rows, pinned host bytes for the wire, piecewise D2H/H2D on a private copy stream.
#include "strix/pair/pipe.hpp"
#include "xport/device_crc32c.hpp"
#include <hip/hip_runtime.h>
#include <functional>
#include <stdexcept>
#include <string>

namespace strix::pair {
inline void pipe_hip_check(hipError_t status, const char* operation) {
    if (status != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}
class HipPipeStaging final : public PipeStaging {
public:
    ~HipPipeStaging() override {
        (void)hipSetDevice(device_);
        if (stream_) (void)hipStreamSynchronize(stream_);
        for (const auto event : pieces_) if (event) (void)hipEventDestroy(event);
        if (snapshot_) (void)hipEventDestroy(snapshot_);
        if (stream_) (void)hipStreamDestroy(stream_);
        if (device_bytes_) (void)hipFree(device_bytes_);
        if (host_bytes_) (void)hipHostFree(host_bytes_);
    }
    // compute: the queue whose work produces (save) or consumes (load) the seam rows.
    explicit HipPipeStaging(std::function<hipStream_t()> compute) : compute_(std::move(compute)) {}
    void allocate(std::size_t capacity) {
        capacity_ = capacity;
        pipe_hip_check(hipGetDevice(&device_), "pair prefill staging device");
        pipe_hip_check(hipHostMalloc(reinterpret_cast<void**>(&host_bytes_), capacity_, hipHostMallocDefault), "pair prefill pinned seam allocation");
        pipe_hip_check(hipMalloc(reinterpret_cast<void**>(&device_bytes_), capacity_), "pair prefill device seam snapshot allocation");
        pipe_hip_check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "pair prefill copy stream");
        pipe_hip_check(hipEventCreateWithFlags(&snapshot_, hipEventDisableTiming), "pair prefill snapshot event");
        pieces_.resize((capacity_ + seam_piece_bytes - 1U) / seam_piece_bytes);
        // CRC and the wire read pinned bytes on the CPU. Device completion
        // alone does not publish coarse-grained host memory to that agent.
        for (auto& event : pieces_) pipe_hip_check(hipEventCreateWithFlags(&event,
            hipEventDisableTiming | hipEventReleaseToSystem), "pair prefill piece event");
    }
    std::span<std::byte> resize(std::size_t bytes) override {
        if (bytes > capacity_) throw std::runtime_error("pair prefill seam exceeds reserved staging capacity");
        return {host_bytes_, bytes};
    }
    void copy(bool save, std::size_t offset, void* device, std::size_t bytes) override {
        auto* packed = device_bytes_ + offset;
        pipe_hip_check(hipMemcpyAsync(save ? packed : device, save ? device : packed, bytes,
            hipMemcpyDeviceToDevice, compute_()), "pair prefill snapshot/scatter");
    }
    void copy_rows(bool save, std::size_t offset, std::size_t pitch, void* device, std::size_t device_pitch,
        std::size_t bytes, std::size_t rows) override {
        auto* packed = device_bytes_ + offset;
        pipe_hip_check(hipMemcpy2DAsync(save ? static_cast<void*>(packed) : device, save ? pitch : device_pitch,
            save ? device : static_cast<const void*>(packed), save ? device_pitch : pitch, bytes, rows,
            hipMemcpyDeviceToDevice, compute_()), "pair prefill strided snapshot/scatter");
    }
    void seal(std::size_t bytes) override {
        pipe_hip_check(hipEventRecord(snapshot_, compute_()), "pair prefill snapshot fence");
        // Detached compute streams may reuse the source arena on return. Only
        // D2D packing must finish here; pinned D2H remains overlapped with compute.
        pipe_hip_check(hipEventSynchronize(snapshot_), "pair prefill immutable snapshot ready");
        pipe_hip_check(hipStreamWaitEvent(stream_, snapshot_, 0), "pair prefill copy dependency");
        for (std::size_t offset = 0; offset < bytes; offset += seam_piece_bytes) {
            const auto length = std::min(seam_piece_bytes, bytes - offset);
            // A device-checked piece stays on the device: digests and the sub-span tail are all the host reads.
            if (device_routed(offset, length)) enqueue_device_crc(offset, length);
            else pipe_hip_check(hipMemcpyAsync(host_bytes_ + offset, device_bytes_ + offset, length,
                hipMemcpyDeviceToHost, stream_), "pair prefill asynchronous D2H");
            pipe_hip_check(hipEventRecord(pieces_[offset / seam_piece_bytes], stream_), "pair prefill D2H piece fence");
        }
    }
    void route_device(std::function<bool(std::size_t)> routed) override {
        routed_ = std::move(routed);
        // DeviceLink's size rule is monotone: if the largest possible piece is
        // ineligible (host-only, CRC OFF, or a small arena), no piece needs CRC storage.
        if (routed_ && routed_(std::min(capacity_, seam_piece_bytes)) && !crc_.allocated()) {
            select_device(); crc_.allocate(capacity_);
        }
    }
    bool device_routed(std::size_t, std::size_t bytes) override { return routed_ && routed_(bytes); }
    std::uint32_t device_crc(std::size_t offset, std::size_t bytes, bool received) override {
        select_device();
        const auto piece = pieces_[offset / seam_piece_bytes];
        if (received) {
            enqueue_device_crc(offset, bytes);
            pipe_hip_check(hipEventRecord(piece, stream_), "pair prefill device CRC fence");
        }
        pipe_hip_check(hipEventSynchronize(piece), "pair prefill device CRC ready");
        const auto tail = xport::DeviceCrc32c::tail_bytes(bytes);
        return crc_.value(offset, bytes, {host_bytes_ + offset + bytes - tail, tail});
    }
    void await_piece(std::size_t offset) override {
        select_device();
        pipe_hip_check(hipEventSynchronize(pieces_[offset / seam_piece_bytes]), "pair prefill D2H piece ready");
    }
    void received_piece(std::size_t offset, std::size_t bytes) override {
        select_device();
        pipe_hip_check(hipMemcpyAsync(device_bytes_ + offset, host_bytes_ + offset, bytes,
            hipMemcpyHostToDevice, stream_), "pair prefill asynchronous H2D");
    }
    void* device_piece(std::size_t offset) override { return device_bytes_ + offset; }
    void read_device_piece(std::size_t offset, std::size_t bytes) override {
        select_device();
        pipe_hip_check(hipMemcpyAsync(host_bytes_ + offset, device_bytes_ + offset, bytes,
            hipMemcpyDeviceToHost, stream_), "pair prefill routed CRC readback");
        pipe_hip_check(hipEventRecord(pieces_[offset / seam_piece_bytes], stream_), "pair prefill routed publication");
        pipe_hip_check(hipEventSynchronize(pieces_[offset / seam_piece_bytes]), "pair prefill routed bytes ready");
    }
    void finish_receive() override {
        select_device();
        pipe_hip_check(hipStreamSynchronize(stream_), "pair prefill authenticated H2D complete");
    }
private:
    std::function<hipStream_t()> compute_;
    int device_ = 0;
    std::size_t capacity_ = 0;
    std::byte* host_bytes_ = nullptr;
    std::byte* device_bytes_ = nullptr;
    hipStream_t stream_ = nullptr;
    hipEvent_t snapshot_ = nullptr;
    std::vector<hipEvent_t> pieces_;
    std::function<bool(std::size_t)> routed_;
    xport::DeviceCrc32c crc_;
    void enqueue_device_crc(std::size_t offset, std::size_t bytes) {
        crc_.enqueue(stream_, device_bytes_, offset, bytes);
        if (const auto tail = xport::DeviceCrc32c::tail_bytes(bytes))
            pipe_hip_check(hipMemcpyAsync(host_bytes_ + offset + bytes - tail, device_bytes_ + offset + bytes - tail, tail,
                hipMemcpyDeviceToHost, stream_), "pair prefill device CRC tail");
    }
    void select_device() const { pipe_hip_check(hipSetDevice(device_), "pair prefill worker device"); }
};
inline std::shared_ptr<PipeStaging> make_hip_pipe_staging(std::size_t capacity, std::function<hipStream_t()> compute) {
    auto staging = std::make_shared<HipPipeStaging>(std::move(compute));
    staging->allocate(capacity);
    return staging;
}

} // namespace strix::pair
