#pragma once
#include "xport/transport_backend.hpp"

namespace strix::minimax::pfx {
// The prefill runtime keeps ownership of its paired TX/RX slots, workers and
// explicit retire-after-consumer contract. No Channel publication mode is added.
template<class ExchangerType, class TagType> class Route {
    enum class Publication { kernel };
    struct Runtime {
        using Tag = TagType;
        ExchangerType& exchanger;
        const auto& options() const { return exchanger.options(); }
        auto sent() const { return exchanger.sent(); }
        void send_copy(const void* source, std::size_t bytes, const Tag& tag) {
            exchanger.send_copy(source, bytes, tag);
        }
        void send(std::size_t bytes, const Tag& tag, Publication) { exchanger.send(bytes, tag); }
        auto recv_view() { return exchanger.recv(); }
        const void* recv() { return exchanger.recv().peer; }
        void drain() { exchanger.drain(); }
    };
    Runtime runtime_;
    using Core = strix::xport::TransportBackend<Runtime>;
    Core backend_;
public:
    using Ticket = typename Core::Ticket;
    explicit Route(ExchangerType& exchanger) : runtime_{exchanger}, backend_(runtime_) {}
    Ticket post(const void* source, std::size_t bytes, const TagType& tag) {
        return source == runtime_.exchanger.tx()
            ? backend_.post_publish(tag, bytes, Publication::kernel)
            : backend_.post_copy(tag, source, bytes);
    }
    // STRIX_TP2_EXCHANGE_FMT=f16: the copy route's pack form (FP32 device source rounded to binary16 in the TX slot).
    Ticket post_pack_f16(const float* source, std::size_t elems, const TagType& tag) {
        runtime_.exchanger.send_pack_f16(source, elems, tag);
        return Ticket{runtime_.exchanger.sent()};
    }
    auto wait(Ticket ticket) {
        return backend_.wait_view(ticket, runtime_.options().stream);
    }
    // retire() remains at the existing MiniMax consumer enqueue site. The core
    // release is deliberately fence-free, just as for the other Transport routes.
    void release(Ticket ticket) { backend_.release(ticket, runtime_.options().stream); }
    static_assert(strix::xport::Backend<Core>);
};
} // namespace strix::minimax::pfx
