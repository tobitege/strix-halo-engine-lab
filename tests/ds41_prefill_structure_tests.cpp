#include "xport/channel_communicator.hpp"
#include "xport/ds41_adapter.hpp"
#include "core/layer_read_ahead.hpp"
#include "deepseek41_prefill_xport_traits.hpp"
#include "xport/wire.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct Node {
    std::vector<std::shared_ptr<Node>> deps;
    std::function<void()> action;
    unsigned state = 0;
    void run() {
        if (state == 2) return;
        require(state == 0, "cross-stream dependency cycle");
        state = 1;
        for (const auto& dep : deps) if (dep) dep->run();
        if (action) action();
        state = 2;
    }
};
using Work = std::shared_ptr<Node>;
struct InputTag { unsigned kind = 2, layer = 0; std::uint64_t request = 0; };
struct Runtime {
    using Stream = unsigned;
    using Event = unsigned;
    using InputTag = ::InputTag;
    std::array<Work, 3> tails{};
    std::vector<Work> events{nullptr}, nodes;
    unsigned syncs = 0, reuses = 0, records = 0;
    Work submit(Stream stream, std::function<void()> action = {}, Work dependency = {}) {
        auto node = std::make_shared<Node>();
        node->deps = {tails.at(stream), dependency};
        node->action = std::move(action);
        nodes.push_back(node);
        tails.at(stream) = node;
        return node;
    }
    static InputTag tag(const InputTag& value) { return value; }
    void record(Event& event, Stream stream) {
        if (!event) { event = static_cast<unsigned>(events.size()); events.push_back(nullptr); }
        events.at(event) = submit(stream);
        ++records;
    }
    void wait(Event event, Stream stream) { submit(stream, {}, events.at(event)); ++reuses; }
    void order(Stream producer, Stream posting) {
        if (producer != posting) submit(posting, {}, tails.at(producer));
    }
    template<class View> void acquire(const View& view, Stream stream) { submit(stream, {}, view.ready); }
    void copy(void* dst, const void* src, std::size_t bytes, Stream stream) {
        submit(stream, [=] { std::memcpy(dst, src, bytes); });
    }
    void destroy(Event) {}
    void synchronize(Stream) { run(); }
    void synchronize_event(Event) { run(); }
    void run() {
        ++syncs;
        // Adversarial order: newer wire writes run before old consumers unless
        // the lease really carries the last-use dependency through slot reuse.
        for (auto i = nodes.rbegin(); i != nodes.rend(); ++i) (*i)->run();
    }
};
struct FakeChannel {
    struct Options { unsigned stream = 0, depth = 2; std::size_t capacity = 13; } opt;
    struct View { const void* payload; std::uint64_t sequence; Work ready; };
    Runtime& runtime;
    std::array<std::array<std::byte, 13>, 2> txs{}, rxs{};
    std::vector<Work> wire{nullptr};
    std::vector<InputTag> tags;
    Work previous;
    std::uint64_t sent_ = 0, received_ = 0;
    unsigned drains = 0;
    bool fail = false;
    bool idle = true;
    strix::xport::Stats totals{};
    explicit FakeChannel(Runtime& value) : runtime(value) {}
    const Options& options() const { return opt; }
    std::uint64_t sent() const { return sent_; }
    void* tx() { return txs[static_cast<std::size_t>(sent_ % opt.depth)].data(); }
    void send(std::size_t bytes, const InputTag& tag) {
        require(bytes && bytes <= opt.capacity, "invalid mock channel size");
        idle = false;
        const auto slot = static_cast<std::size_t>(sent_++ % opt.depth);
        auto job = std::make_shared<Node>();
        job->deps = {runtime.tails[opt.stream], previous};
        job->action = [this, slot, bytes] {
            std::memcpy(rxs[slot].data(), txs[slot].data(), bytes);
            ++totals.exchanges;
            totals.bytes += bytes;
            totals.producer_wait += 1.0;
            totals.wire += 2.0;
            totals.check += 3.0;
        };
        previous = job;
        runtime.nodes.push_back(job);
        wire.push_back(job);
        tags.push_back(tag);
    }
    void send_copy(const void* source, std::size_t bytes, const InputTag& tag) {
        runtime.copy(tx(), source, bytes, opt.stream);
        send(bytes, tag);
    }
    View recv_view() {
        require(received_ < sent_, "receive before send");
        const auto slot = static_cast<std::size_t>(received_++ % opt.depth);
        return {rxs[slot].data(), received_, wire.at(static_cast<std::size_t>(received_))};
    }
    void drain() { ++drains; runtime.run(); if (fail) throw std::runtime_error("CRC failure"); idle = true; }
    strix::xport::Stats stats() const {
        require(idle, "Channel statistics read before successful drain");
        return totals;
    }
};
struct Legacy {
    const void* source = nullptr;
    std::size_t bytes = 0;
    unsigned calls = 0;
    std::array<std::byte, 64> staging{};
    void* send_staging(std::size_t) { ++calls; return staging.data(); }
    void start(unsigned, const void* data, std::size_t n, const InputTag&) { source = data; bytes = n; ++calls; }
    void finish(unsigned, void* dst) { if (dst) std::memcpy(dst, source, bytes); ++calls; }
    void start_staged(unsigned s, std::size_t n, const InputTag& t) { start(s, staging.data(), n, t); }
    const void* finish_staged(unsigned) { ++calls; return source; }
    void release_staged(unsigned) { ++calls; }
    const void* finish_staged_ready() { return source; }
    void start_staged_ready(unsigned s, std::size_t n, const InputTag& t) { start_staged(s, n, t); }
    void release_staged_ready(unsigned s) { release_staged(s); }
};
using Facade = strix::xport::ChannelCommunicator<FakeChannel, Legacy, Runtime>;
void accounting_test() {
    Runtime runtime;
    FakeChannel channel{runtime};
    Legacy legacy;
    Facade facade(&legacy, &channel, runtime);
    std::array<std::byte, 31> source{}, output{};
    source.fill(std::byte{0x62});
    std::uint64_t exchanges = 10, bytes = 20;
    double seconds = 30.0;
    auto account = [&](std::uint64_t count, std::uint64_t size, double producer, double wire, double check) {
        require(channel.idle, "accounted before drain");
        exchanges += count;
        bytes += size;
        seconds += producer + wire + check;
    };
    for (std::uint64_t step = 1; step <= 2; ++step) {
        facade.exchange(0, source.data(), output.data(), source.size(), {});
        facade.drain_and_account(account);
        facade.drain_and_account(account);
        require(output == source && legacy.calls == 0, "gather bypassed Channel or changed bytes");
        require(exchanges == 10 + 3 * step && bytes == 20 + source.size() * step,
                "Channel deltas lost or counted twice across leases");
        require(seconds == 30.0 + 18.0 * static_cast<double>(step), "Channel elapsed components lost");
    }
    channel.fail = true;
    facade.exchange(0, source.data(), output.data(), source.size(), {});
    bool threw = false;
    try { facade.drain_and_account(account); } catch (const std::runtime_error&) { threw = true; }
    require(threw && exchanges == 16 && bytes == 82 && seconds == 66.0,
            "failed drain sampled or published mutable statistics");
}
void lease_test() {
    Runtime runtime;
    FakeChannel channel{runtime};
    Legacy legacy;
    Facade facade(&legacy, &channel, runtime);
    std::array<std::byte, 31> source{};
    std::array<std::array<std::byte, 31>, 12> output{};
    for (unsigned j = 0; j < output.size(); ++j) {
        const unsigned producer = j % 2U, consumer = (j + 1U) % 2U;
        runtime.submit(producer, [&, j] { source.fill(static_cast<std::byte>(j + 1U)); });
        // Caller orders an external consumer after the producer, as the engine
        // does with its gather-ready event.
        runtime.order(producer, consumer);
        facade.exchange(consumer, source.data(), output[j].data(), source.size(), InputTag{2, j, j + 17U});
        runtime.order(consumer, producer); // source lifetime before its next write
    }
    require(runtime.syncs == 0 && channel.drains == 0, "exchange blocked the host");
    facade.drain();
    for (unsigned j = 0; j < output.size(); ++j)
        for (auto byte : output[j]) require(byte == static_cast<std::byte>(j + 1U), "lease wrap changed bytes");
    require(legacy.calls == 0 && runtime.reuses > 20U, "missing device lease fences");
    require(channel.tags.size() == 36U, "large message was not bounded into pieces");
    for (unsigned j = 0; j < channel.tags.size(); ++j)
        require(channel.tags[j].request == j / 3U + 17U, "chunking changed collective identity");
}
void staged_and_seed_test() {
    Runtime runtime;
    FakeChannel channel{runtime};
    Legacy legacy;
    Facade facade(&legacy, &channel, runtime);
    std::array<std::array<std::byte, 13>, 11> output{};
    std::array<std::array<std::byte, 13>, 11> expected{};
    for (unsigned j = 0; j < output.size(); ++j) {
        auto* tx = static_cast<std::byte*>(facade.send_staging(13));
        expected[j].fill(static_cast<std::byte>(0x80U + j));
        runtime.copy(tx, expected[j].data(), 13, 0);
        facade.start_staged(0, 13, InputTag{2, j, j});
        const auto* rx = static_cast<const std::byte*>(facade.finish_staged(1));
        // A seed-style consumer uses a different stream, with reordered byte
        // ranges. Reuse must wait for all its reads, not only RX publication.
        runtime.copy(output[j].data(), rx + 5, 8, 1);
        runtime.copy(output[j].data() + 8, rx, 5, 1);
        facade.release_staged(1);
        runtime.submit(0); // subsequent layer work may be queued immediately
    }
    require(!runtime.syncs, "staged/seed exchange waited on host");
    require(facade.send_staging(14) == nullptr, "oversized staged view admitted");
    facade.drain();
    require(output == expected, "staged/seed consumer raced slot reuse");
}
void legacy_and_failure_test() {
    Runtime runtime;
    Legacy legacy;
    std::array<std::byte, 9> source{}, output{};
    source.fill(std::byte{0xA5});
    {
        Facade facade(&legacy, nullptr, runtime);
        facade.exchange(0, source.data(), output.data(), source.size(), {});
        auto* tx = facade.send_staging(9);
        std::memcpy(tx, source.data(), 9);
        facade.start_staged(0, 9, {});
        require(std::memcmp(facade.finish_staged(0), source.data(), 9) == 0, "fallback staged bytes");
        facade.release_staged(0);
        require(!facade.active() && output == source, "legacy fallback changed");
    }
    require(runtime.syncs == 0 && legacy.calls == 6U, "legacy route acquired new waits");
    FakeChannel channel{runtime};
    channel.fail = true;
    Facade facade(&legacy, &channel, runtime);
    facade.exchange(0, source.data(), output.data(), source.size(), {});
    bool threw = false;
    try { facade.drain(); } catch (const std::runtime_error&) { threw = true; }
    require(threw, "failed wire validation accepted");
}
void ds41_adapter_test() {
    Runtime runtime;
    FakeChannel channel{runtime};
    Legacy legacy;
    Facade facade(&legacy, &channel, runtime);
    std::array<std::byte, 13> source{}, output{}, staged_output{};
    source.fill(std::byte{0xCD});
    strix::xport::ds41_exchange_raw_slots(facade, 0U, source.data(), output.data(), source.size(), InputTag{}, [] {});
    auto* tx = facade.send_staging(13);
    runtime.copy(tx, source.data(), source.size(), 0U);
    strix::xport::Ds41RawSlotsStagedExchange<Facade, unsigned, InputTag> exchange(facade, 0U, tx, 13, {});
    runtime.copy(staged_output.data(), exchange.finish(0U), 13, 0U);
    exchange.release(0U);
    require(!runtime.syncs, "DS41 adapter blocked on host");
    facade.drain();
    require(output == source && staged_output == source, "DS41 adapter changed opaque bytes");
}
void read_ahead_test() {
    for (std::size_t fail = 0; fail <= 4; ++fail) {
        std::array<unsigned, 4> bytes{}, published{};
        unsigned reads = 0, cleared = 0;
        bool threw = false;
        try {
            strix::core::read_ahead_blocks(4,
                [&](std::size_t b) { ++reads; bytes[b] = 999; if (b == fail) throw std::runtime_error("I/O"); bytes[b] = static_cast<unsigned>(b + 1U); },
                [&](std::size_t b) { require(bytes[b] == b + 1U || bytes[b] == 0U, "published partial layer"); ++published[b]; },
                [&](std::size_t b) { bytes[b] = 0; ++cleared; });
        } catch (const std::runtime_error&) { threw = true; }
        require(threw == (fail < 4), "I/O exception lost");
        for (auto count : published) require(count == 1U, "layer gate not released exactly once");
        require(reads == std::min(static_cast<unsigned>(fail + 1U), 4U), "reader continued after failure");
        require(cleared == (fail < 4 ? 4U - fail : 0U), "unread payload not cleared");
    }
}
void wire_test() {
    using Traits = strix::deepseek41::prefill_xport::Traits;
    using Kind = Traits::Kind;
    const auto t = strix::xport::make_trailer<Traits>({Kind::payload, 39, 77}, 2, Traits::kPlan, 123, 131, 1, false, 0x1234, false, 0);
    require(t.magic == Traits::kMagic && t.plan == Traits::kPlan && t.sequence == 123 && t.step == 77 &&
            t.layer == 39 && t.bytes == 131 && t.rank == 1 && t.crc32c == 0x1234 && !t.flags, "wire identity lost");
}
}
int main() try {
    accounting_test(); lease_test(); staged_and_seed_test(); legacy_and_failure_test(); ds41_adapter_test(); read_ahead_test(); wire_test();
    std::cout << "DS41 structure host tests: PASS (leases, wrap, chunks, seeds, fallback, failures, layer readiness, wire)\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
