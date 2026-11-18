#include "strix/pair/layercut.hpp"
#include <atomic>
#include <cstdio>
#include <future>

namespace {
namespace ho = strix::handoff;
namespace lc = strix::pair::layercut;
using namespace std::chrono_literals;
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// Real loopback transport: a producer has completed HELLO, but no relay connects.
void missing_relay(bool disconnect) {
    ho::Listener listener(0, "127.0.0.1"), relay(0, "127.0.0.1");
    auto connecting = std::async(std::launch::async, [&] {
        ho::SenderOptions options;
        options.codec = ho::Codec::identity;
        options.connect_timeout = 2s;
        options.reply_timeout = 2s;
        return std::make_unique<ho::Sender>(ho::Endpoint{"127.0.0.1", listener.port(), ""},
                                          ho::Hello{1, "startup-test", 0, 1, {}}, options);
    });
    auto session = listener.accept();
    auto producer = connecting.get();
    std::atomic<unsigned> cancelled{0};
    lc::StartupScope startup([&]() noexcept {
        ++cancelled;
        session->cancel("startup failed");
        relay.shutdown();
    }, disconnect ? 5s : 250ms);
    std::atomic<bool> reader_failed{false};
    std::jthread reader([&] {
        try { ho::Piece piece; while (session->next(piece)) {} }
        catch (...) { reader_failed = true; startup.fail(); }
    });
    const auto start = std::chrono::steady_clock::now();
    if (disconnect) producer->abort("producer disconnected before relay");
    bool refused = false;
    try { (void)relay.accept(); } catch (const std::exception&) { refused = true; }
    reader.join();
    require(refused && reader_failed, "startup must wake relay accept and retire producer session");
    require(std::chrono::steady_clock::now() - start < 2s, "producer failure must not wait for startup deadline");
    require(cancelled == 1, "startup cancellation must run exactly once");
    bool failed = false;
    try { startup.complete(); } catch (const std::runtime_error&) { failed = true; }
    require(failed, "expired/failed startup must not become ready");
}
}

int main() try {
    missing_relay(false);
    missing_relay(true);
    unsigned cancelled = 0;
    { lc::StartupScope scope([&]() noexcept { ++cancelled; }); scope.complete(); }
    require(cancelled == 0, "successful startup must disarm cancellation");
    try {
        lc::StartupScope scope([&]() noexcept { ++cancelled; });
        throw std::runtime_error("HELLO validation/allocation failed");
    } catch (const std::runtime_error&) {}
    require(cancelled == 1, "startup exception must retire accepted producer");
    std::puts("layercut startup tests passed");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
