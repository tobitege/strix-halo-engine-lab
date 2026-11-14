#pragma once
#include "deepseek41_prefill_xport_traits.hpp"
#include "xport/channel.hpp"
#include "xport/channel_communicator.hpp"
#include "strix/hip_check.hpp"

namespace strix::deepseek41::prefill_xport {
using Exchange = xport::Channel<Traits>;
struct Runtime {
    using Stream = hipStream_t;
    using Event = hipEvent_t;
    using InputTag = dist::Tag;
    Event bridge = nullptr;
    ~Runtime() { destroy(bridge); }
    static void check(hipError_t e) {
        ::strix::hip_check_message(e, "prefill xport: HIP fence/copy failed");
    }
    static Tag tag(const InputTag& value) {
        return {static_cast<Kind>(value.kind), value.layer, value.request};
    }
    static void record(Event& event, Stream stream) {
        if (!event) check(hipEventCreateWithFlags(&event, hipEventDisableTiming));
        check(hipEventRecord(event, stream));
    }
    static void wait(Event event, Stream stream) { check(hipStreamWaitEvent(stream, event, 0)); }
    void order(Stream producer, Stream posting) {
        if (producer == posting) return;
        record(bridge, producer);
        wait(bridge, posting);
    }
    static void acquire(const xport::RxView& view, Stream stream) {
        check(hipStreamWaitValue64(stream, const_cast<std::uint64_t*>(view.flag), view.sequence,
                                   hipStreamWaitValueGte, ~std::uint64_t{0}));
    }
    static void copy(void* dst, const void* src, std::size_t bytes, Stream stream) {
        check(hipMemcpyAsync(dst, src, bytes, hipMemcpyDefault, stream));
    }
    static void destroy(Event event) { if (event) (void)hipEventDestroy(event); }
    static void synchronize(Stream stream) { (void)hipStreamSynchronize(stream); }
    static void synchronize_event(Event event) { (void)hipEventSynchronize(event); }
};
using Communicator = xport::ChannelCommunicator<Exchange, dist::Communicator, Runtime>;
} // namespace strix::deepseek41::prefill_xport

namespace strix::xport {
extern template class Channel<deepseek41::prefill_xport::Traits>;
}
