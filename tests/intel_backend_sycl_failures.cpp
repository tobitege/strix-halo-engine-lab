// Exercises the actual backend.cpp SYCL branch with a host fault-injection double.
#include "intel/core/backend.hpp"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#ifdef _MSC_VER
#include <crtdbg.h>
#endif

namespace {
using namespace strix::intel;
namespace inject = sycl::test;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

struct Owner {
    Queue queue;
    Owner() { require(open_queue(0, &queue, nullptr) == Status::ok, "open test queue"); }
    ~Owner() { close_queue(queue); }
};

void async_failures() {
    {
        Owner owner;
        require(synchronize(owner.queue) == Status::ok, "healthy wait");
        inject::async_wait = true;
        require(synchronize(owner.queue) == Status::device_error, "async error translated");
        require(inject::missing_handler == 0, "explicit async handler installed");
        require(synchronize(owner.queue) == Status::device_error, "async failure stays sticky");
        inject::async_destroy = true;
    }
    require(inject::live == 0 && inject::delivered == 2 && inject::missing_handler == 0, "handler survives teardown");
}

void sync_failures() {
    Owner owner;
    void* p = &owner;
    require(allocate({}, MemoryKind::shared, 1, &p) == Status::invalid && p == nullptr, "invalid queue clears output");
    const MemoryKind kinds[]{MemoryKind::device, MemoryKind::shared, MemoryKind::host};
    const inject::Fault faults[]{inject::Fault::device_alloc, inject::Fault::shared_alloc, inject::Fault::host_alloc};
    for (std::size_t i = 0; i < 3; ++i) {
        inject::fault = faults[i];
        p = &owner;
        require(allocate(owner.queue, kinds[i], 1, &p) == Status::device_error && p == nullptr, "allocation throw clears output");
    }
    inject::fault = inject::Fault::none;
    inject::null_alloc = true;
    require(allocate(owner.queue, MemoryKind::device, 1, &p) == Status::device_error && p == nullptr, "null allocation");
    inject::null_alloc = false;
    require(allocate(owner.queue, MemoryKind::device, 1, &p) == Status::ok && p, "successful allocation");
    inject::fault = inject::Fault::release;
    require(release(owner.queue, p) == Status::device_error, "release throw translated");
    inject::fault = inject::Fault::copy;
    require(copy(owner.queue, p, p, 1) == Status::device_error, "copy throw translated");
    inject::fault = inject::Fault::wait;
    require(synchronize(owner.queue) == Status::device_error, "wait throw translated");
    std::uint8_t weights[136]{};
    std::int8_t codes[128]{};
    float deltas[4]{}, out = 0;
    dense::Q8A8View v{weights, codes, deltas, &out, 136, 0, 128, 1, 1, 128, 1, 0};
    inject::fault = inject::Fault::submit;
    require(enqueue_dense_q8a8(owner.queue, v) == Status::device_error, "submission throw translated");
    inject::fault = inject::Fault::none;
    require(copy(owner.queue, p, p, 1) == Status::ok && enqueue_dense_q8a8(owner.queue, v) == Status::ok &&
            synchronize(owner.queue) == Status::ok && release(owner.queue, p) == Status::ok, "healthy wrappers");
}

void open_failures() {
    Queue queue;
    Capabilities caps;
    caps.arch = "unchanged";
    caps.compute_units = 123;
    caps.global_bytes = 456;
    for (const auto fault : {inject::Fault::construct, inject::Fault::info}) {
        inject::fault = fault;
        require(open_queue(0, &queue, &caps) == Status::device_error, "open failure status");
        require(!queue.impl && inject::live == 0, "failed open retains no ownership");
        require(caps.arch == "unchanged" && caps.compute_units == 123 && caps.global_bytes == 456, "caps published atomically");
    }
    inject::fault = inject::Fault::none;
    require(open_queue(0, &queue, &caps) == Status::ok && inject::live == 1, "retry succeeds");
    void* original = queue.impl;
    require(open_queue(0, &queue, &caps) == Status::invalid && queue.impl == original && inject::live == 1, "reopen rejected");
    close_queue(queue);
    require(!queue.impl && inject::live == 0, "owner closes");
    close_queue(queue);
}
} // namespace

void wfmt_intel_queue_test();
int main(int argc, char** argv) {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
        const std::string_view name = argc == 2 ? argv[1] : "";
        if (name == "async") async_failures();
        else if (name == "sync") sync_failures();
        else if (name == "open") open_failures();
        else if (name == "queues") wfmt_intel_queue_test();
        else throw std::runtime_error("expected async, sync or open");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "SYCL boundary host test: %s\n", e.what());
        return 1;
    }
    return 0;
}
