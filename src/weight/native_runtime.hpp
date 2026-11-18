#pragma once
#include <strix/weight/runtime.hpp>
#include <strix/weight/source.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <thread>
#include <mutex>
#include <vector>
namespace strix::weight::detail {
using namespace strix::weight;
template<class Api> struct NativeRuntime {
static Status check(typename Api::Error error) { return error==Api::Success ? Status{Error::ok,{}} : Status{error==Api::OutOfMemory ? Error::memory_budget : Error::device_failure,Api::GetErrorString(error)}; }
struct Pending { void* data; Retirement retirement; std::uint32_t device; bool quarantined=false; };
struct RetirementService {
    std::mutex mutex;
    std::condition_variable wake;
    std::vector<Pending> pending;
    bool stopping = false;
    std::thread worker;
    RetirementService() : worker([this] {
        std::unique_lock lock(mutex);
        while (!stopping) {
            collect_locked(false);
            if (pending.empty()) wake.wait(lock, [this] { return stopping || !pending.empty(); });
            else wake.wait_for(lock, std::chrono::milliseconds(2));
        }
    }) {}
    void collect_locked(bool drain) {
        for (auto it=pending.begin(); it!=pending.end();) {
            if (it->quarantined) { ++it; continue; }
            bool done=Api::SetDevice(static_cast<int>(it->device))==Api::Success;
            for (auto stream:it->retirement.unresolved_streams) {
                if (!done) break;
                done=(drain ? Api::StreamSynchronize(static_cast<typename Api::Stream>(stream)) :
                              Api::StreamQuery(static_cast<typename Api::Stream>(stream)))==Api::Success;
            }
            for (auto event:it->retirement.events) {
                if (!done) break;
                done=(drain ? Api::EventSynchronize(static_cast<typename Api::Event>(event)) :
                              Api::EventQuery(static_cast<typename Api::Event>(event)))==Api::Success;
            }
            if (!done) { ++it; continue; }
            // A failed free has uncertain ownership; never retry it blindly.
            if (Api::Free(it->data)!=Api::Success) { it->quarantined=true; ++it; continue; }
            for (auto event:it->retirement.events) Api::EventDestroy(static_cast<typename Api::Event>(event));
            it=pending.erase(it); // Allocation reclaimed: release its reservation.
        }
    }
    // Call only after all runtime users have stopped submitting retirements.
    void shutdown() {
        { std::lock_guard lock(mutex); stopping=true; }
        wake.notify_one();
        if (worker.joinable()) worker.join();
        std::lock_guard lock(mutex); collect_locked(true);
    }
    ~RetirementService() {
        shutdown();
        // Device errors cannot prove safety. Quarantine allocation, events and
        // reservation for process lifetime instead of reclaiming unsafely.
        if (!pending.empty()) (void)new std::vector<Pending>(std::move(pending));
    }
    void retire(Pending item) {
        { std::lock_guard lock(mutex); pending.push_back(std::move(item)); collect_locked(false); }
        wake.notify_one();
    }
};
static RetirementService& service() { static RetirementService owner; return owner; }
static void shutdown() { service().shutdown(); }

static strix::weight::Status bind(void* stream, std::uint32_t device, ExecutionContext& context, Hardware& hardware) {
    if (device>static_cast<std::uint32_t>(INT32_MAX)) return {Error::device_failure,"device index overflow"};
    auto status=check(Api::SetDevice(static_cast<int>(device))); if (!status) return status;
    typename Api::Properties properties{}; status=check(Api::GetDeviceProperties(&properties,static_cast<int>(device))); if (!status) return status;
    std::size_t free_bytes=0,total_bytes=0; status=check(Api::MemGetInfo(&free_bytes,&total_bytes)); if (!status) return status;
    Hardware h{}; h.backend=Api::backend; h.arch=Api::arch(properties);
    h.compiler_id=current_compiler_id(); h.device_class=h.arch; h.free_bytes=free_bytes;
    h.warp_or_wave=static_cast<std::uint32_t>(properties.warpSize);
    h.shared_bytes_per_sm=static_cast<std::uint32_t>(properties.sharedMemPerMultiprocessor);
    // Calibrated rates and image budget are supplied by deployment/receipts, never guessed here.
    auto callbacks=std::make_shared<RuntimeCallbacks>();
    callbacks->allocate=[device](std::uint64_t bytes,void*& data) -> Status {
        auto s=check(Api::SetDevice(static_cast<int>(device))); if (!s) return s;
        if (bytes>SIZE_MAX) return {Error::memory_budget,"allocation size overflow"};
        return check(Api::Malloc(&data,static_cast<std::size_t>(bytes ? bytes : 1)));
    };
    callbacks->free=[device](void* data) { if (Api::SetDevice(static_cast<int>(device))==Api::Success) Api::Free(data); };
    callbacks->record_event=[device](ExecutionContext& c,void*& value) -> Status {
        auto s=check(Api::SetDevice(static_cast<int>(device))); if (!s) return s;
        typename Api::Event event{}; s=check(Api::EventCreateWithFlags(&event,Api::EventDisableTiming)); if (!s) return s;
        s=check(Api::EventRecord(event,static_cast<typename Api::Stream>(c.stream)));
        if (!s) { Api::EventDestroy(event); return s; } value=event; return s;
    };
    callbacks->wait_event=[device](ExecutionContext& c,void* event) -> Status {
        auto s=check(Api::SetDevice(static_cast<int>(device))); if (!s) return s;
        return check(Api::StreamWaitEvent(static_cast<typename Api::Stream>(c.stream),static_cast<typename Api::Event>(event),0));
    };
    callbacks->event_complete=[device](void* event) { return Api::SetDevice(static_cast<int>(device))==Api::Success && Api::EventQuery(static_cast<typename Api::Event>(event))==Api::Success; };
    callbacks->destroy_event=[device](void* event) { if (Api::SetDevice(static_cast<int>(device))==Api::Success) Api::EventDestroy(static_cast<typename Api::Event>(event)); };
    callbacks->convert=[device](const WeightView& w,const Plan& p,void* data,ExecutionContext&) -> Status {
        if (p.selected.strategy!=Strategy::native || p.selected.resources.image_bytes!=w.payload.range.bytes)
            return {Error::unsupported_arithmetic,"image conversion source not registered"};
        auto s=check(Api::SetDevice(static_cast<int>(device))); if (!s) return s;
        std::array<Byte,65536> chunk{};
        for (std::uint64_t at=0; at<w.payload.range.bytes;) {
            auto bytes=static_cast<std::size_t>(std::min<std::uint64_t>(chunk.size(),w.payload.range.bytes-at));
            s=w.payload.source->read({w.payload.range.offset+at,bytes},{chunk.data(),bytes}); if (!s) return s;
            s=check(Api::Memcpy(static_cast<Byte*>(data)+at,chunk.data(),bytes,Api::MemcpyHostToDevice));
            // Pageable copies may only stage before returning. Drain the actual
            // default copy stream before publishing readiness on any consumer.
            auto copied=check(Api::StreamSynchronize(nullptr));
            if (!s) return s;
            if (!copied) return copied;
            at+=bytes;
        }
        return {Error::ok,{}};
    };
    callbacks->retire=[device](void* data,Retirement retirement,ExecutionContext) {
        // A failed conversion may have left work on the default upload stream.
        if (!retirement.unresolved_streams.empty()) retirement.unresolved_streams.push_back(nullptr);
        service().retire({data,std::move(retirement),device});
    };
    context={Api::backend,stream,device,std::move(callbacks)}; hardware=std::move(h);
    return {Error::ok,{}};
}
};

}
