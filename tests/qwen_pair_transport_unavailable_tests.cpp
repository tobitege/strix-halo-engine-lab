// apps/qwen_pair_transport.hpp built without the pair runtime (STRIX_DIST_PAIR_RUNTIME=0, the build of every
// platform but Linux): opening a pair link refuses, and the host path over a given link moves the bytes as is.
// The target defines the macro itself, so this runs on every platform.
#include "qwen_pair_transport.hpp"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int failures = 0;
void expect(bool condition, const char* what) {
    if (!condition) { std::cerr << "FAIL: " << what << '\n'; ++failures; }
}

// A loopback link: what is sent comes back on the next receive.
class Loopback final : public strix::dist::Transport {
public:
    std::vector<std::byte> queued;
    std::string aborted;
    void send(std::span<const std::byte> message) override { queued.assign(message.begin(), message.end()); }
    void receive(std::span<std::byte> message) override {
        if (message.size() != queued.size()) throw std::logic_error("loopback size");
        std::copy(queued.begin(), queued.end(), message.begin());
    }
    void exchange(std::span<const std::byte> out, std::span<std::byte> in) override { send(out); receive(in); }
    [[nodiscard]] std::string name() const override { return "loopback"; }
    void abort(const std::string& why) override { aborted = why; }
};
}

int main() {
    bool refused = false;
    try { strix::qwen_pp::PairTransport opened(0U); }
    catch (const std::invalid_argument& error) { refused = std::string(error.what()).find("Linux") != std::string::npos; }
    expect(refused, "opening a pair link refuses and names the platform requirement");

    bool empty_refused = false;
    try { strix::qwen_pp::PairTransport empty(std::unique_ptr<strix::dist::Transport>{}); }
    catch (const std::invalid_argument&) { empty_refused = true; }
    expect(empty_refused, "an empty deployment link is refused");

    auto link = std::make_unique<Loopback>();
    auto* raw = link.get();
    strix::qwen_pp::PairTransport pair(std::move(link));
    const std::vector<std::byte> sent{std::byte{1}, std::byte{2}, std::byte{3}};
    std::vector<std::byte> received(sent.size());
    pair.send(sent);
    pair.receive(received);
    expect(received == sent, "the host path moves the bytes unchanged");
    pair.abort("stop");
    expect(raw->aborted == "stop", "abort reaches the link");
    pair.finish();

    if (failures) return 1;
    std::cout << "qwen pair transport without the pair runtime: refused, host path intact\n";
}
