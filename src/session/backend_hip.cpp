#include "strix/session/backend_hip.hpp"

#include <hip/hip_runtime.h>
#ifdef STRIX_SESSION_DEVICE_GATE
#include "strix/device/hip_gate.hpp"
#endif
#ifdef STRIX_SESSION_HIP_ADD
#include "strix/ds41_shexp_tp.hpp"
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

namespace strix::session {
namespace {
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}
class DeviceScope {
public:
    explicit DeviceScope(int device) {
        check(hipGetDevice(&previous_), "HIP current device");
        check(hipSetDevice(device), "HIP select device");
    }
    ~DeviceScope() { (void)hipSetDevice(previous_); }
private:
    int previous_ = 0;
};
// Destruction cannot throw. If completion or device selection fails, native
// storage is deliberately quarantined (leaked) rather than freed while in use.
struct HipBuffer final : detail::BufferState {
    int device = 0;
    void* raw = nullptr;
    std::byte* data = nullptr;
    bool quarantined = false;
    std::function<void()> release;
    ~HipBuffer() override {
        try {
            DeviceScope scope(device);
            if (raw && !quarantined && hipFree(raw) == hipSuccess && release) release();
        } catch (...) {}
    }
};
struct Staging {
    int device = 0;
    void* data = nullptr;
    bool quarantined = false;
    ~Staging() {
        try { DeviceScope scope(device); if (data && !quarantined) (void)hipHostFree(data); }
        catch (...) {}
    }
};
struct HipQueue final : detail::QueueState {
    int device = 0;
    hipStream_t stream = nullptr;
    std::vector<std::shared_ptr<HipBuffer>> buffers;
    std::vector<std::shared_ptr<Staging>> staging;
    void quarantine() noexcept {
        for (auto& b : buffers) b->quarantined = true;
        for (auto& h : staging) h->quarantined = true;
    }
    ~HipQueue() override {
        try {
            DeviceScope scope(device);
            if (stream) {
                if (hipStreamSynchronize(stream) != hipSuccess) { quarantine(); return; }
                (void)hipStreamDestroy(stream);
            }
        } catch (...) { quarantine(); }
    }
};
struct HipEvent final : detail::EventState {
    int device = 0;
    hipEvent_t event = nullptr;
    ~HipEvent() override {
        try { DeviceScope scope(device); if (event) (void)hipEventDestroy(event); }
        catch (...) {}
    }
};
std::byte* address(const detail::BoundBuffer& bound) {
    return static_cast<HipBuffer&>(*bound.state).data + bound.offset;
}
void retain(HipQueue& queue, const detail::BoundBuffer& bound) {
    auto buffer = std::static_pointer_cast<HipBuffer>(bound.state);
    if (buffer->quarantined) throw std::runtime_error("HIP buffer completion is unknown");
    queue.buffers.push_back(std::move(buffer));
}
}

struct HipBackend::State {
    int device = 0;
    std::size_t limit = 0, reserved = 0;
    bool failed = false;
    void healthy() const { if (failed) throw std::runtime_error("HIP backend completion is unknown"); }
};
HipBackend::HipBackend(std::size_t budget_bytes) : state_(std::make_shared<State>()) {
#ifdef STRIX_SESSION_DEVICE_GATE
    // Start gate: the backend only ever runs on the declared device (a no-op when the main already bound it).
    (void)strix::device::bind_hip();
#endif
    check(hipGetDevice(&state_->device), "HIP bind device");
    state_->limit = budget_bytes;
#ifdef STRIX_SESSION_HIP_ADD
    // Reuse the existing launcher; no arithmetic is duplicated here. A bounded
    // flat batch avoids the launcher's unsigned block-count overflow.
    register_operation({"add", "f32", "none", "hip"}, {2, sizeof(float), alignof(float), 1024U * 1024U},
        [state = state_](detail::QueueState& base, const detail::BoundOperation& op) {
            state->healthy();
            DeviceScope scope(state->device);
            auto& queue = static_cast<HipQueue&>(base);
            retain(queue, op.output);
            for (const auto& input : op.inputs) retain(queue, input);
            strix::ds41::enqueue_add_f32(reinterpret_cast<const float*>(address(op.inputs[0])),
                reinterpret_cast<const float*>(address(op.inputs[1])), op.elements,
                reinterpret_cast<float*>(address(op.output)), queue.stream);
        });
#endif
}
HipBackend::~HipBackend() = default;
MemoryBudget HipBackend::memory_budget() const {
    MemoryBudget result{state_->limit, state_->reserved, 0, 0, 0, false};
    if (state_->failed) return result;
    DeviceScope scope(state_->device);
    std::size_t free_bytes = 0, total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
        result.observed_free = free_bytes;
        result.observed_total = total_bytes;
        result.probe_known = true;
        result.available = std::min(state_->limit - state_->reserved, free_bytes);
    }
    return result;
}
std::shared_ptr<detail::BufferState> HipBackend::allocate_buffer(std::size_t bytes, std::size_t alignment) {
    state_->healthy();
    const auto reservation = bytes + alignment - 1; // overflow checked by Backend
    if (reservation > memory_budget().available) throw std::length_error("HIP buffer budget/probe exhausted");
    DeviceScope scope(state_->device);
    auto buffer = std::make_shared<HipBuffer>();
    buffer->device = state_->device;
    buffer->release = [state = state_, reservation] { state->reserved -= reservation; };
    check(hipMalloc(&buffer->raw, reservation), "HIP allocate buffer");
    state_->reserved += reservation;
    const auto raw = reinterpret_cast<std::uintptr_t>(buffer->raw);
    if (raw > std::numeric_limits<std::uintptr_t>::max() - (alignment - 1))
        throw std::overflow_error("HIP aligned address overflow");
    buffer->data = reinterpret_cast<std::byte*>((raw + alignment - 1) & ~(alignment - 1));
    return buffer;
}
std::shared_ptr<detail::QueueState> HipBackend::make_queue() {
    state_->healthy();
    DeviceScope scope(state_->device);
    auto queue = std::make_shared<HipQueue>();
    queue->device = state_->device;
    check(hipStreamCreateWithFlags(&queue->stream, hipStreamNonBlocking), "HIP create queue");
    return queue;
}
std::shared_ptr<detail::EventState> HipBackend::record_event(detail::QueueState& base) {
    state_->healthy();
    DeviceScope scope(state_->device);
    auto event = std::make_shared<HipEvent>();
    event->device = state_->device;
    check(hipEventCreateWithFlags(&event->event, hipEventDisableTiming), "HIP create event");
    check(hipEventRecord(event->event, static_cast<HipQueue&>(base).stream), "HIP record event");
    return event;
}
void HipBackend::wait_event(detail::QueueState& base, detail::EventState& event) {
    state_->healthy();
    DeviceScope scope(state_->device);
    check(hipStreamWaitEvent(static_cast<HipQueue&>(base).stream, static_cast<HipEvent&>(event).event, 0), "HIP wait event");
}
void HipBackend::finish(detail::QueueState& base) {
    state_->healthy();
    DeviceScope scope(state_->device);
    auto& queue = static_cast<HipQueue&>(base);
    check(hipStreamSynchronize(queue.stream), "HIP complete queue");
    queue.buffers.clear();
    queue.staging.clear();
}
void HipBackend::drain(detail::QueueState& base) noexcept {
    auto& queue = static_cast<HipQueue&>(base);
    try {
        DeviceScope scope(state_->device);
        if (hipStreamSynchronize(queue.stream) == hipSuccess) {
            queue.buffers.clear();
            queue.staging.clear();
            return;
        }
    } catch (...) {}
    state_->failed = true;
    queue.quarantine();
}
void HipBackend::transfer(detail::QueueState& base, const detail::BoundBuffer* dst, const detail::BoundBuffer* src,
                          void* host_dst, const void* host_src, std::size_t bytes) {
    state_->healthy();
    DeviceScope scope(state_->device);
    auto& queue = static_cast<HipQueue&>(base);
    if (dst) retain(queue, *dst);
    if (src) retain(queue, *src);
    if (!bytes) return;
    if (dst && src) {
        check(hipMemcpyAsync(address(*dst), address(*src), bytes, hipMemcpyDeviceToDevice, queue.stream), "HIP device copy");
        return;
    }
    // GPU work never borrows caller host memory, even on failed drain.
    auto staging = std::make_shared<Staging>();
    staging->device = state_->device;
    check(hipHostMalloc(&staging->data, bytes, hipHostMallocDefault), "HIP copy staging");
    queue.staging.push_back(staging);
    if (dst) {
        std::memcpy(staging->data, host_src, bytes);
        check(hipMemcpyAsync(address(*dst), staging->data, bytes, hipMemcpyHostToDevice, queue.stream), "HIP upload");
    } else {
        check(hipMemcpyAsync(staging->data, address(*src), bytes, hipMemcpyDeviceToHost, queue.stream), "HIP download");
        finish(base);
        std::memcpy(host_dst, staging->data, bytes);
    }
}
void HipBackend::execute_plan(ExecutionPlan&) { throw UnsupportedOperation("HIP family plan unsupported"); }
}
