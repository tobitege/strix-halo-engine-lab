// strix::handoff::Lanes over loopback: an urgent piece submitted behind a deep bulk queue reaches its receiver early
// (FIFO would deliver it last), every piece arrives once with a valid manifest, and abort releases queued bulk once.
#include "strix/handoff_lanes.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace ho = strix::handoff;
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
constexpr std::uint32_t kUrgent = 1, kBulk = 2;
constexpr unsigned kBulkPieces = 24;
constexpr std::size_t kBulkBytes = 256 << 10, kUrgentBytes = 64 << 10;

struct Receiver {
    ho::Codec codec = ho::Codec::identity;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> order;  // (kind, index) in delivery order
    std::exception_ptr error;
    std::thread thread;
    void run(ho::Listener& l, std::atomic<bool>* go) {
        thread = std::thread([this, &l, go] {
            try {
                ho::ReceiverOptions o;
                o.max_buffered_wire_bytes = 512 << 10;  // a slow consumer: the transport stops reading ahead
                o.socket_buffer = 64 << 10;
                auto s = l.accept(o);
                codec = s->codec();
                while (go && !go->load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
                ho::Piece p;
                while (s->next(p)) {
                    order.emplace_back(p.header.kind, p.header.index);
                    for (std::size_t i = 0; i < p.plain.size(); i += 4099)
                        if (p.plain[i] != static_cast<std::uint8_t>(p.header.kind * 31 + p.header.index)) throw std::runtime_error("payload");
                }
                s->reply(0);
            } catch (...) { error = std::current_exception(); }
        });
    }
};

template<class F> void wait_for(F ready) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!ready()) {
        require(std::chrono::steady_clock::now() < end, "fixture readiness timeout");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void failed_write() {
    // Leave seq 0 buffered: the receiver then reads seq 1's header but cannot consume its payload.
    ho::Listener listener(0, "127.0.0.1");
    auto accepted = std::async(std::launch::async, [&] {
        ho::ReceiverOptions ro; ro.max_buffered_wire_bytes = 1; ro.socket_buffer = 4096;
        return listener.accept(ro);
    });
    std::vector<std::uint8_t> bytes(128U << 20, 42);
    std::atomic<unsigned> primed{0}, first{0}, second{0}, bulk{0};
    ho::SenderOptions so; so.codec = ho::Codec::identity; so.socket_buffer = 4096;
    so.send_timeout = std::chrono::seconds(3);
    ho::Sender sender({"127.0.0.1", listener.port(), ""}, {10, "lanes-failure", 0, 1, {}}, so);
    auto session = accepted.get();
    const strix::bp_codec::Region seed{bytes.data(), 1, 1}, big{bytes.data(), bytes.size(), 1};
    sender.submit({}, {&seed, 1}, [&] { ++primed; });
    wait_for([&] { return primed.load() == 1; });
    ho::Lanes lanes({&sender}, {});
    lanes.urgent(0, {}, {&big, 1}, [&] { ++first; });
    lanes.urgent(0, {}, {&seed, 1}, [&] { ++second; });
    wait_for([&] { return sender.stats().pieces == 3; });
    require(first == 0 && second == 0, "first urgent blocked, second encoded and queued");
    lanes.bulk(0, {}, {seed}, [&] { ++bulk; });
    bool failed = false;
    try { lanes.drain(); } catch (const std::runtime_error&) { failed = true; }
    require(failed, "asynchronous write failure reaches drain");
    require(first == 1 && second == 1 && bulk == 1, "failed drain joins sender and releases every piece once");
}

void rejected_urgent(bool stopped_only) {
    ho::Listener listener(0, "127.0.0.1");
    auto receiver = std::async(std::launch::async, [&] {
        auto session = listener.accept(); ho::Piece piece;
        require(!session->next(piece), "empty session"); session->reply(0);
    });
    ho::Sender sender({"127.0.0.1", listener.port(), ""}, {11, "lanes-reject", 0, 1, {}});
    sender.finish(); receiver.get();
    std::atomic<unsigned> released{0};
    if (stopped_only) {
        ho::Lanes stopped({&sender}, {}); stopped.abort();
        bool rejected = false;
        try { stopped.urgent(0, {}, {}, [&] { ++released; }); }
        catch (const std::runtime_error& e) { rejected = std::string(e.what()).find("urgent after abort") != std::string::npos; }
        require(rejected && released == 1, "urgent rejected by stopped lane releases once");
        rejected = false;
        try { stopped.urgent(1, {}, {}, [&] { ++released; }); } catch (const std::out_of_range&) { rejected = true; }
        require(rejected && released == 2, "invalid urgent destination releases once");
        return;
    }
    ho::Lanes lanes({&sender}, {});
    // A completed Sender rejects the feeder's submit synchronously, latching a lane error.
    lanes.bulk(0, {}, {}, [&] { ++released; });
    bool failed = false;
    try { lanes.drain(); } catch (const std::exception&) { failed = true; }
    require(failed && released == 1, "feeder error latched");
    failed = false;
    try { lanes.urgent(0, {}, {}, [&] { ++released; }); } catch (const std::exception&) { failed = true; }
    require(failed && released == 2, "urgent rejected after feeder error releases once");
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc > 1 && std::string(argv[1]) == "--failure") { failed_write(); return 0; }
    if (argc > 1 && std::string(argv[1]) == "--rejection") { rejected_urgent(false); return 0; }
    if (argc > 1 && std::string(argv[1]) == "--stopped") { rejected_urgent(true); return 0; }
    const bool mixed = argc > 1 && std::string(argv[1]) == "--mixed";
    ho::Listener la(0, "127.0.0.1"), lb(0, "127.0.0.1");
    std::atomic<bool> go{false};
    Receiver ra, rb;
    ra.run(la, &go);
    rb.run(lb, nullptr);
    ho::SenderOptions so;
    so.codec = mixed ? ho::Codec::bp_v1 : ho::Codec::identity;
    so.socket_buffer = 64 << 10;
    so.connect_timeout = std::chrono::seconds(20);
    auto a = std::make_unique<ho::Sender>(ho::Endpoint{"127.0.0.1", la.port(), ""}, ho::Hello{7, "lanes-test", 0, 1, {}}, so);
    auto b = std::make_unique<ho::Sender>(ho::Endpoint{"127.0.0.1", lb.port(), ""}, ho::Hello{8, "lanes-test", 0, 1, {}}, so);
    std::vector<std::vector<std::uint8_t>> data;
    for (unsigned i = 0; i < 2 * kBulkPieces; ++i)
        data.emplace_back(kBulkBytes, static_cast<std::uint8_t>(kBulk * 31 + i % kBulkPieces));
    std::vector<std::uint8_t> urgent(kUrgentBytes, static_cast<std::uint8_t>(kUrgent * 31 + 0));
    std::atomic<unsigned> released{0};
    {
        ho::Lanes lanes({a.get(), b.get()}, ho::Lanes::Options{kBulkBytes, true});
        for (unsigned i = 0; i < kBulkPieces; ++i)
            for (unsigned dst = 0; dst < 2; ++dst) {
                auto& d = data[dst * kBulkPieces + i];
                lanes.bulk(dst, {kBulk, i, 0, i, 1}, {strix::bp_codec::Region{d.data(), d.size(), 1}}, [&] { ++released; });
            }
        std::this_thread::sleep_for(std::chrono::milliseconds(300));  // the feeder fills what the slow receiver allows
        const strix::bp_codec::Region r{urgent.data(), urgent.size(), 1};
        lanes.urgent(0, {kUrgent, 0, 0, 0, 1}, {&r, 1}, [&] { ++released; });
        go = true;
        lanes.drain();
        require(a->finish().status == 0 && b->finish().status == 0, "replies");
        const auto st = lanes.stats();
        require(st.urgent_pieces == 1 && st.bulk_pieces == 2 * kBulkPieces && st.bulk_bytes == 2 * kBulkPieces * kBulkBytes, "stats");
    }
    ra.thread.join(); rb.thread.join();
    if (ra.error) std::rethrow_exception(ra.error);
    if (rb.error) std::rethrow_exception(rb.error);
    require(ra.codec == so.codec && rb.codec == so.codec, "negotiated codec");
    require(released.load() == 2 * kBulkPieces + 1, "every release ran once");
    require(ra.order.size() == kBulkPieces + 1 && rb.order.size() == kBulkPieces, "every piece arrived");
    std::size_t at = ra.order.size();
    for (std::size_t i = 0; i < ra.order.size(); ++i) if (ra.order[i].first == kUrgent) at = i;
    std::printf("urgent piece delivered at position %zu of %zu (FIFO: %u)\n", at, ra.order.size(), kBulkPieces);
    if (!mixed) require(at < kBulkPieces / 2, "urgent piece must overtake the queued bulk");
    for (unsigned i = 0, k = 0; i < ra.order.size(); ++i)
        if (ra.order[i].first == kBulk) require(ra.order[i].second == k++, "bulk keeps its order");

    // abort: queued bulk is released exactly once, later bulk is refused (and released)
    {
        ho::Listener lc(0, "127.0.0.1");
        Receiver rc;
        std::atomic<bool> never{false};
        rc.run(lc, &never);
        auto c = std::make_unique<ho::Sender>(ho::Endpoint{"127.0.0.1", lc.port(), ""}, ho::Hello{9, "lanes-test", 0, 1, {}}, so);
        std::atomic<unsigned> dropped{0};
        {
            ho::Lanes lanes({c.get()}, ho::Lanes::Options{kBulkBytes, true});
            for (unsigned i = 0; i < kBulkPieces; ++i)
                lanes.bulk(0, {kBulk, i, 0, i, 1}, {strix::bp_codec::Region{data[i].data(), data[i].size(), 1}}, [&] { ++dropped; });
            lanes.abort();
            bool refused = false;
            try { lanes.bulk(0, {kBulk, 0, 0, 0, 1}, {strix::bp_codec::Region{data[0].data(), 1, 1}}, [&] { ++dropped; }); }
            catch (const std::runtime_error&) { refused = true; }
            require(refused, "bulk after abort is refused");
        }
        c->abort("test");
        c.reset();  // the Sender releases what it was handed
        never = true;
        lc.shutdown();
        rc.thread.join();
        require(dropped.load() == kBulkPieces + 1, "abort releases every callback once");
    }
    std::puts("PASS: handoff lanes (urgent overtakes bulk, manifest, abort releases)");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
