#pragma once
// HIP part of core::WeightLoadQueue: pinned staging, an explicit non-blocking copy stream, one event per ticket.
// Nothing here waits unless asked to (wait / drain), and nothing uses the legacy stream, so weight copies never
// order themselves against the engine's kernels.
#include "weight_load_queue.hpp"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strix::core {

class HipWeightUploader final : public WeightUploader {
public:
    HipWeightUploader() {
        ok(hipGetDevice(&device_), "device");
        ok(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "copy stream");
    }
    ~HipWeightUploader() override {
        if (stream_ != nullptr) (void)hipStreamSynchronize(stream_);
        for (auto& [token, event] : live_) (void)hipEventDestroy(event);
        for (auto event : spare_) (void)hipEventDestroy(event);
        if (stream_ != nullptr) (void)hipStreamDestroy(stream_);
    }
    HipWeightUploader(const HipWeightUploader&) = delete;
    HipWeightUploader& operator=(const HipWeightUploader&) = delete;
    // The copy stream, for a consumer that orders a kernel after a ticket with hipStreamWaitEvent.
    [[nodiscard]] hipStream_t stream() const { return stream_; }

    std::uint8_t* allocate(std::uint64_t bytes) override {
        bind();
        void* p = nullptr;
        ok(hipHostMalloc(&p, static_cast<std::size_t>(bytes), hipHostMallocDefault), "pinned staging");
        std::lock_guard<std::mutex> lock(m_);
        pinned_ += bytes;
        peak_pinned_ = std::max(peak_pinned_, pinned_);
        return static_cast<std::uint8_t*>(p);
    }
    void release(std::uint8_t* staging, std::uint64_t bytes) noexcept override {
        (void)hipHostFree(staging);
        std::lock_guard<std::mutex> lock(m_);
        pinned_ -= bytes;
    }
    Token submit(const std::uint8_t* staging, const Copy* copies, std::size_t count) override {
        bind();
        std::lock_guard<std::mutex> lock(m_);  // the copies and their event are adjacent on the stream
        // All bookkeeping memory is taken before the first copy is enqueued: once a copy references the staging,
        // nothing here can fail without the stream being quiet first (and retire() never allocates).
        spare_.reserve(spare_.size() + live_.size() + 1);
        const auto slot = live_.emplace(next_, nullptr).first;
        hipEvent_t event = nullptr;
        if (!spare_.empty()) {
            event = spare_.back();
            spare_.pop_back();
        } else if (const hipError_t made = hipEventCreateWithFlags(&event, hipEventDisableTiming); made != hipSuccess) {
            live_.erase(slot);
            ok(made, "ready event");
        }
        hipError_t e = hipSuccess;
        for (std::size_t i = 0; i < count && e == hipSuccess; ++i)
            e = hipMemcpyAsync(copies[i].device, staging + copies[i].staging_offset, static_cast<std::size_t>(copies[i].bytes),
                               hipMemcpyHostToDevice, stream_);
        if (e == hipSuccess) e = hipEventRecord(event, stream_);
        if (e != hipSuccess) {
            // A copy may already reference the staging: the stream is quiet before the caller gets it back.
            (void)hipStreamSynchronize(stream_);
            spare_.push_back(event);
            live_.erase(slot);
            ok(e, "weight upload");
        }
        slot->second = event;
        return next_++;
    }
    bool done(Token token) override {
        std::lock_guard<std::mutex> lock(m_);
        const auto it = live_.find(token);
        if (it == live_.end()) return true;
        const hipError_t e = hipEventQuery(it->second);
        if (e == hipErrorNotReady) return false;
        ok(e, "ready event query");
        return true;
    }
    void wait(Token token) override {
        hipEvent_t event = nullptr;
        {
            std::lock_guard<std::mutex> lock(m_);
            const auto it = live_.find(token);
            if (it == live_.end()) return;
            event = it->second;
        }
        bind();
        ok(hipEventSynchronize(event), "ready event");
    }
    void retire(Token token) noexcept override {
        std::lock_guard<std::mutex> lock(m_);
        const auto it = live_.find(token);
        if (it == live_.end()) return;
        spare_.push_back(it->second);
        live_.erase(it);
    }
    void drain() override {
        bind();
        ok(hipStreamSynchronize(stream_), "weight upload");
    }
    [[nodiscard]] std::uint64_t peak_pinned_bytes() const {
        std::lock_guard<std::mutex> lock(m_);
        return peak_pinned_;
    }

private:
    static void ok(hipError_t e, const char* what) {
        if (e != hipSuccess) throw std::runtime_error(std::string("weight load: ") + what + ": " + hipGetErrorString(e));
    }
    // Worker threads start on device 0: every entry point runs on the device the uploader was created on.
    void bind() const {
        thread_local int bound = -1;
        if (bound != device_) {
            ok(hipSetDevice(device_), "device");
            bound = device_;
        }
    }
    int device_ = 0;
    hipStream_t stream_ = nullptr;
    mutable std::mutex m_;
    std::unordered_map<Token, hipEvent_t> live_;
    std::vector<hipEvent_t> spare_;
    Token next_ = 1;
    std::uint64_t pinned_ = 0, peak_pinned_ = 0;
};

// Content hash of device memory, read back in pieces through pageable host memory (identity gates only).
inline std::uint64_t device_weight_hash64(const void* device, std::uint64_t bytes) {
    constexpr std::uint64_t piece = std::uint64_t{64} << 20;
    std::vector<std::uint8_t> host(static_cast<std::size_t>(std::min(piece, std::max<std::uint64_t>(bytes, 1))));
    std::uint64_t h = 0;
    for (std::uint64_t at = 0; at < bytes; at += piece) {
        const std::uint64_t n = std::min(piece, bytes - at);
        if (hipMemcpy(host.data(), static_cast<const std::uint8_t*>(device) + at, static_cast<std::size_t>(n), hipMemcpyDeviceToHost) != hipSuccess)
            throw std::runtime_error("weight load: device read-back failed");
        h = weight_hash64(host.data(), n, h);
    }
    return h;
}

}  // namespace strix::core
