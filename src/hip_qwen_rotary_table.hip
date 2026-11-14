#include "strix/startup_markers.hpp"
#include "strix/hip_qwen_rotary_table.hpp"
#include "strix/hip_check.hpp"
#include "strix/model_content_hash.hpp"
#include <hip/hip_runtime.h>
#include <bit>
#include <stdexcept>
#include <string>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <functional>
#include <cstdio>

namespace strix {
namespace {
#if defined(STRIX_BUILD_QWEN_ROTARY_DIAGNOSTIC)
std::atomic<unsigned> retirement_failure{0};
bool fail_retirement(unsigned step) {
    auto expected=step; return retirement_failure.compare_exchange_strong(expected,0U);
}
#else
bool fail_retirement(unsigned) { return false; }
#endif
// HIP graph/user-object callbacks cannot call HIP. Final allocation release is
// handed to this CPU worker, which drains streams before freeing the allocation.
class RetirementQueue {
    std::mutex mutex;
    std::condition_variable changed;
    std::queue<std::function<void()>> pending;
    bool stop = false, active = false;
    std::thread worker{[this] {
        std::unique_lock lock(mutex);
        for (;;) {
            changed.wait(lock, [&] { return stop || !pending.empty(); });
            if (pending.empty() && stop) return;
            auto task = std::move(pending.front()); pending.pop();
            active = true;
            lock.unlock();
            task();
            task = {}; // lease release may enqueue allocation retirement
            lock.lock();
            active = false; changed.notify_all();
        }
    }};
public:
    ~RetirementQueue() {
        { const std::lock_guard lock(mutex); stop = true; }
        changed.notify_one(); worker.join();
    }
    void submit(std::function<void()> task) {
        { const std::lock_guard lock(mutex); pending.push(std::move(task)); }
        changed.notify_one();
    }
    void wait_idle() {
        std::unique_lock lock(mutex);
        changed.wait(lock,[&] { return pending.empty() && !active; });
    }
    void retire(const void* p, int device) {
        submit([p,device] {
            const bool retired = retire_qwen_rotary_allocation(
                [=] { return !fail_retirement(1U) && hipSetDevice(device) == hipSuccess; },
                [] { return !fail_retirement(2U) && hipDeviceSynchronize() == hipSuccess; },
                [=] { (void)hipFree(const_cast<void*>(p)); });
            // A failed drain is not proof of completion. The allocation stays
            // quarantined for process lifetime; a retry must never race readers.
            if (!retired) std::fputs("YaRN retirement quarantined after device/drain failure\n", stderr);
        });
    }
};
RetirementQueue& retirements() { static RetirementQueue queue; return queue; }
#if defined(STRIX_BUILD_QWEN_ROTARY_DIAGNOSTIC)
std::atomic<unsigned> reader_failure{0};
bool fail_reader(unsigned step) {
    auto expected=step; return reader_failure.compare_exchange_strong(expected,0U);
}
#else
bool fail_reader(unsigned) { return false; }
#endif
using Reader = std::shared_ptr<const HipQwenRotaryTable>;
void release_reader(void* value) { delete static_cast<Reader*>(value); }
void check(hipError_t error) {
    ::strix::hip_check(error, "YaRN upload");
}
}
HipQwenRotaryTable::HipQwenRotaryTable(const RotaryDescriptor& descriptor) : descriptor_(descriptor) {
    static_assert(std::endian::native == std::endian::little);
    static_assert(sizeof(YarnRow) == 256);
    if (descriptor.mode() != RotaryMode::yarn) throw std::invalid_argument("native rotary has no device table");
    check(hipGetDevice(&device_));
    (void)retirements();
    void* data = nullptr;
    { startup::Phase marker("rotary_malloc");
      check(hipMalloc(&data, descriptor.table_bytes())); }
    // Publish only after generation/upload/hash finishes. Arm allocation cleanup
    // first so a shared_ptr control-block failure cannot double-free the buffer.
    allocation_ = std::shared_ptr<const void>(data, [device = device_](const void* pointer) {
        retirements().retire(pointer, device);
    });
    Sha256Fast hash;
    generate_qwen_yarn_rows(descriptor, 0, extent(), [&](auto first, auto rows) {
        const auto bytes = std::as_bytes(rows);
        hash.update(bytes);
        startup::Phase marker("rotary_upload", first);
        check(hipMemcpy(static_cast<std::byte*>(data) + static_cast<std::size_t>(first) * sizeof(YarnRow),
            bytes.data(), bytes.size(), hipMemcpyHostToDevice));
    });
    identity_ = RotaryIdentity(descriptor, hash.finalize());
}

void validate_qwen_rotary_reader(const HipQwenRotaryTable& table, void* stream_ptr) {
    if (table.device() < 0) throw std::invalid_argument("CPU rotary adoption cannot enter HIP dispatch");
    int active = -1;
    check(hipGetDevice(&active));
    const int reader = qwen_rotary_reader_device(static_cast<hipStream_t>(stream_ptr),
        hipStreamLegacy, hipStreamPerThread, active, [](hipStream_t stream) {
            hipDevice_t device = -1;
            check(hipStreamGetDevice(stream, &device));
            return device;
        });
    require_qwen_rotary_reader_device(table.device(), active, reader);
    hipPointerAttribute_t attributes{};
    check(hipPointerGetAttributes(&attributes, table.data()));
    if (attributes.type != hipMemoryTypeDevice || attributes.isManaged || attributes.device != active)
        throw std::invalid_argument("rotary table is not a local device allocation");
    hipDeviceptr_t base{}; std::size_t bytes = 0;
    check(hipMemGetAddressRange(&base, &bytes, const_cast<void*>(table.data())));
    if (reinterpret_cast<const void*>(base) != table.data() || bytes < table.descriptor().table_bytes())
        throw std::invalid_argument("rotary table exceeds its actual device allocation");
}

void retain_qwen_rotary_reader(std::shared_ptr<const HipQwenRotaryTable> table, void* stream_ptr) {
    if (!table) return;
    validate_qwen_rotary_reader(*table, stream_ptr);
    auto stream = static_cast<hipStream_t>(stream_ptr);
    hipStreamCaptureStatus status{};
    hipGraph_t graph = nullptr;
    check(hipStreamGetCaptureInfo_v2(stream, &status, nullptr, &graph, nullptr, nullptr));
    auto reader = std::make_unique<Reader>(std::move(table));
    if (status == hipStreamCaptureStatusActive) {
        hipUserObject_t object = nullptr;
        check(hipUserObjectCreate(&object, reader.get(), release_reader, 1, hipUserObjectNoDestructorSync));
        reader.release(); // user object owns the reference even on retain failure
        const auto result = hipGraphRetainUserObject(graph, object, 1, hipGraphUserObjectMove);
        if (result != hipSuccess) { (void)hipUserObjectRelease(object, 1); check(result); }
    } else {
        // An event waiter also releases its lease on a device/stream failure;
        // host callbacks are not guaranteed to run after a device error.
        hipEvent_t done = nullptr;
        auto result = fail_reader(1U) ? hipErrorUnknown : hipEventCreateWithFlags(&done, hipEventDisableTiming);
        if (result == hipSuccess) result = fail_reader(2U) ? hipErrorUnknown : hipEventRecord(done, stream);
        if (result != hipSuccess) {
            (void)hipStreamSynchronize(stream);
            if (done) (void)hipEventDestroy(done);
            check(result);
        }
        try {
            if(fail_reader(3U)) throw std::runtime_error("induced rotary lease queue failure");
            retirements().submit([lease = *reader, done] {
                if (hipSetDevice(lease->device()) == hipSuccess) {
                    // Even failed event waits release only into allocation
                    // retirement, whose independent device drain must succeed.
                    (void)hipEventSynchronize(done);
                    (void)hipEventDestroy(done);
                }
            });
        } catch (...) {
            (void)hipStreamSynchronize(stream);
            (void)hipEventDestroy(done);
            throw;
        }
    }
}
#if defined(STRIX_BUILD_QWEN_ROTARY_DIAGNOSTIC)
namespace qwen_rotary_diagnostic {
void fail_next_retirement(unsigned step) { retirement_failure.store(step); }
void fail_next_reader(unsigned step) { reader_failure.store(step); }
void wait_retirements() { retirements().wait_idle(); }
}
#endif
} // namespace strix
