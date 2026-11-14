#pragma once
#include <hip/hip_runtime.h>
#include "strix/hip_check.hpp"
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strix::core {
// One-shot host-produced payload readiness. Events are RECORDED before any
// consumer can wait on them: waiting on an unrecorded HIP event is a no-op.
// A dedicated stream waits for coherent host publications, then signals each
// already-recorded event. The producer thread makes no HIP calls.
class HipHostReady {
    hipStream_t stream_ = nullptr, consumer_ = nullptr;
    std::uint64_t* host_ = nullptr;
    std::vector<hipEvent_t> events_;
    static void check(hipError_t e) {
        ::strix::hip_check_message(e, "host readiness: HIP fence failed");
    }
    void retire() noexcept {
        if (host_) for (std::size_t i = 0; i < events_.size(); ++i) publish(i);
        if (stream_) (void)hipStreamSynchronize(stream_);
        (void)hipStreamSynchronize(consumer_);
        for (auto event : events_) if (event) (void)hipEventDestroy(event);
        if (stream_) (void)hipStreamDestroy(stream_);
        if (host_) (void)hipHostFree(host_);
    }
public:
    HipHostReady(std::size_t count, hipStream_t consumer) : consumer_(consumer), events_(count) {
        try {
            check(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
            check(hipHostMalloc(reinterpret_cast<void**>(&host_), count * 64U, hipHostMallocMapped | hipHostMallocCoherent));
            auto* device = static_cast<std::uint64_t*>(nullptr);
            check(hipHostGetDevicePointer(reinterpret_cast<void**>(&device), host_, 0));
            for (std::size_t i = 0; i < count; ++i) host_[i * 8U] = 0;
            for (std::size_t i = 0; i < count; ++i) {
                check(hipEventCreateWithFlags(&events_[i], hipEventDisableTiming));
                check(hipStreamWaitValue64(stream_, device + i * 8U, 1, hipStreamWaitValueGte, ~std::uint64_t{0}));
                check(hipEventRecord(events_[i], stream_));
            }
        } catch (...) { retire(); throw; }
    }
    HipHostReady(const HipHostReady&) = delete;
    HipHostReady& operator=(const HipHostReady&) = delete;
    ~HipHostReady() { retire(); }
    void publish(std::size_t index) noexcept {
        std::atomic_ref<std::uint64_t>(host_[index * 8U]).store(1, std::memory_order_release);
    }
    void wait(std::size_t index, hipStream_t consumer) {
        check(hipStreamWaitEvent(consumer, events_.at(index), 0));
    }
};
} // namespace strix::core
