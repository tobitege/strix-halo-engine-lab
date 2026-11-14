#include "strix/direct_stream_control.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>


namespace {
using namespace strix;
int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
template<class Function> bool rejected(Function&& function) {
    try { function(); } catch (const std::exception&) { return true; }
    return false;
}
DirectStreamDeviceIdentity device() {
#if defined(_WIN32)
    return {"C:/synthetic/character-device", 2, 3, 4, true};
#else
    return {"/synthetic/character-device", 2, 3, 4, true};
#endif
}
DirectStreamControlPolicy policy(std::uint32_t calls = 64) { return {128, calls, std::chrono::milliseconds(20)}; }
std::vector<std::byte> framed(std::span<const std::byte> frame) {
    if (frame.size() > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("fixture frame too large");
    std::vector<std::byte> result(4 + frame.size());
    const auto bytes = static_cast<std::uint32_t>(frame.size());
    for (unsigned shift = 0; shift < 32; shift += 8) result[shift / 8U] = static_cast<std::byte>((bytes >> shift) & 0xffU);
    if (!frame.empty()) std::memcpy(result.data() + 4, frame.data(), frame.size());
    return result;
}
class Io : public DirectStreamControlIo {
public:
    DirectStreamFd open(const DirectStreamDeviceIdentity&) override { ++opened; return fd; }
    DirectStreamFdIdentity inspect(DirectStreamFd) override { ++inspections; return {current, generation}; }
    DirectStreamIoResult read(DirectStreamFd, std::span<std::byte> bytes) override {
        ++read_calls;
        if (!read_script.empty()) {
            const auto status = read_script.front(); read_script.pop_front();
            if (status != DirectStreamIoStatus::transferred) return {status, 0};
        }
        const auto count = std::min({bytes.size(), read_chunk, incoming.size() - offset});
        if (count == 0) return {DirectStreamIoStatus::end_of_stream, 0};
        std::memcpy(bytes.data(), incoming.data() + offset, count); offset += count;
        time += callback_delay;
        if (replace_during_read) ++generation;
        return {DirectStreamIoStatus::transferred, count};
    }
    DirectStreamIoResult write(DirectStreamFd, std::span<const std::byte> bytes) override {
        ++write_calls;
        if (!write_script.empty()) {
            const auto status = write_script.front(); write_script.pop_front();
            if (status != DirectStreamIoStatus::transferred) return {status, 0};
        }
        const auto count = std::min(bytes.size(), write_chunk);
        outgoing.insert(outgoing.end(), bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(count));
        time += callback_delay;
        return {DirectStreamIoStatus::transferred, count};
    }
    DirectStreamWaitStatus wait(DirectStreamFd, bool, DirectStreamClock::time_point) override {
        ++wait_calls; time += wait_delay;
        if (replace_during_wait) ++generation;
        return wait_result;
    }
    DirectStreamClock::time_point now() const override { return time; }
    void close(DirectStreamFd token) noexcept override {
        if (token.open_generation == generation) ++closed;
        else ++stale_close_refused;
    }
    DirectStreamFd fd{9, 1};
    DirectStreamDeviceIdentity current = device();
    std::uint64_t generation = 1;
    DirectStreamClock::time_point time{};
    std::chrono::milliseconds wait_delay{1}, callback_delay{0};
    DirectStreamWaitStatus wait_result = DirectStreamWaitStatus::ready;
    std::vector<std::byte> incoming, outgoing;
    std::deque<DirectStreamIoStatus> read_script, write_script;
    std::size_t offset = 0, read_chunk = 3, write_chunk = 3;
    int opened = 0, closed = 0, stale_close_refused = 0, inspections = 0, read_calls = 0, write_calls = 0, wait_calls = 0;
    bool replace_during_read = false, replace_during_wait = false;
};
}  // namespace

int main() {
    const std::array<std::byte, 5> message{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}};
    {
        Io io; io.incoming = framed(message);
        io.read_script = {DirectStreamIoStatus::interrupted, DirectStreamIoStatus::would_block};
        io.write_script = {DirectStreamIoStatus::would_block, DirectStreamIoStatus::interrupted};
        {
            auto stream = DirectStreamControlByteStream::open_with_io(io, device(), policy());
            stream.send(message);
            expect(io.outgoing == framed(message), "partial writes/EINTR/EAGAIN retain TCP-compatible U32 framing");
            expect(stream.receive() == std::vector<std::byte>(message.begin(), message.end()) && !stream.failed(),
                   "partial reads/EINTR/EAGAIN reconstruct one bounded frame");
            expect(io.wait_calls == 2, "would-block paths wait for explicit bounded readiness");
            auto moved = std::move(stream);
            expect(!stream.valid() && moved.valid(), "move transfers exclusive FD ownership");
        }
        expect(io.closed == 1, "normal destruction closes the owned FD exactly once");
    }
    for (const auto cutoff : {0U, 2U, 6U}) {
        Io io; io.incoming = framed(message); io.incoming.resize(cutoff);
        auto stream = DirectStreamControlByteStream::open_with_io(io, device(), policy());
        expect(rejected([&] { static_cast<void>(stream.receive()); }) && stream.failed() &&
               rejected([&] { stream.send(message); }) && io.write_calls == 0,
               "EOF before/inside prefix or payload poisons framing and forbids reuse");
    }
    for (const std::uint32_t declared : {0U, 129U, std::numeric_limits<std::uint32_t>::max()}) {
        Io io;
        for (unsigned shift = 0; shift < 32; shift += 8) io.incoming.push_back(static_cast<std::byte>((declared >> shift) & 0xffU));
        io.incoming.insert(io.incoming.end(), message.begin(), message.end());
        auto stream = DirectStreamControlByteStream::open_with_io(io, device(), policy());
        expect(rejected([&] { static_cast<void>(stream.receive()); }) && stream.failed() && io.offset == 4,
               "zero/oversized/malicious length fails before any payload read or allocation");
    }
    for (int failure = 0; failure < 4; ++failure) {
        Io io; io.incoming = framed(message); io.read_script.push_back(DirectStreamIoStatus::would_block);
        if (failure == 0) io.wait_delay = std::chrono::milliseconds(20);
        if (failure == 1) io.wait_result = DirectStreamWaitStatus::timed_out;
        if (failure == 2) io.wait_result = DirectStreamWaitStatus::end_of_stream;
        if (failure == 3) io.replace_during_wait = true;
        auto stream = DirectStreamControlByteStream::open_with_io(io, device(), policy());
        expect(rejected([&] { static_cast<void>(stream.receive()); }) && stream.failed() && io.read_calls == 1 && io.offset == 0,
               "late readiness/timeout/EOF/FD replacement cannot authorize another read");
    }
    {
        Io io; io.incoming = framed(message); io.callback_delay = std::chrono::milliseconds(20);
        auto stream = DirectStreamControlByteStream::open_with_io(io, device(), policy());
        expect(rejected([&] { static_cast<void>(stream.receive()); }) && stream.failed(),
               "late positive IO callback cannot publish a frame after its absolute deadline");
    }
    {
        Io io; io.incoming = framed(message); io.replace_during_read = true;
        auto stream = DirectStreamControlByteStream::open_with_io(io, device(), policy());
        expect(rejected([&] { static_cast<void>(stream.receive()); }) && stream.failed() && io.read_calls == 1,
               "same-path FD instance replacement during callback is rejected before frame exposure");
    }
    for (int drift = 0; drift < 4; ++drift) {
        Io io; auto stream = DirectStreamControlByteStream::open_with_io(io, device(), policy());
        if (drift == 0) ++io.current.inode;
        if (drift == 1) ++io.current.character_device_number;
        if (drift == 2) ++io.current.filesystem_device;
        if (drift == 3) ++io.generation;
        expect(rejected([&] { stream.send(message); }) && stream.failed() && io.write_calls == 0,
               "device inode/rdev/filesystem/open-generation drift is fenced before write");
    }
    {
        Io io; io.read_script.assign(16, DirectStreamIoStatus::interrupted);
        auto stream = DirectStreamControlByteStream::open_with_io(io, device(), policy(4));
        expect(rejected([&] { static_cast<void>(stream.receive()); }) && stream.failed() && io.read_calls == 4,
               "EINTR storm has a hard IO-call budget even with a stationary injected clock");
    }
    {
        Io io; io.write_chunk = 1;
        auto stream = DirectStreamControlByteStream::open_with_io(io, device(), policy(5));
        expect(rejected([&] { stream.send(message); }) && stream.failed() && io.outgoing.size() == 5 &&
               rejected([&] { stream.send(message); }), "partial frame budget exhaustion permanently fails stream");
    }
    {
        Io io; auto stream = DirectStreamControlByteStream::open_with_io(io, device(), policy());
        std::vector<std::byte> too_large(129);
        expect(rejected([&] { stream.send(too_large); }) && !stream.failed() && io.write_calls == 0,
               "caller length rejection before first byte leaves untouched stream healthy");
        auto invalid = device(); invalid.character_device = false;
        auto relative = device(); relative.canonical_path = "synthetic/device";
        expect(rejected([&] { static_cast<void>(DirectStreamControlByteStream::open_with_io(io, invalid, policy())); }) &&
               rejected([&] { static_cast<void>(DirectStreamControlByteStream::open_with_io(io, relative, policy())); }),
               "explicit non-character or relative routes are not opened");
        auto bad_policy = policy(); bad_policy.timeout = std::chrono::milliseconds(0);
        expect(rejected([&] { static_cast<void>(DirectStreamControlByteStream::open_with_io(io, device(), bad_policy)); }),
               "unbounded/zero deadline is rejected before FD acquisition");
    }
#if defined(__linux__) || defined(__APPLE__)
    {
        // /dev/null is a real character-device FD fixture, not a stream peer.
        const auto observed = inspect_direct_stream_device("/dev/null");
        auto stale = observed; ++stale.inode;
        expect(rejected([&] { static_cast<void>(DirectStreamControlByteStream::open_device(stale)); }),
               "concrete opener rejects observed path replaced by a different inode");
        auto stream = DirectStreamControlByteStream::open_device(observed, policy());
        stream.send(message);
        expect(rejected([&] { static_cast<void>(stream.receive()); }) && stream.failed(),
               "concrete POSIX owner reads/writes verified character FD and reports EOF cleanly");
        expect(rejected([&] { static_cast<void>(inspect_direct_stream_device("/etc/hosts")); }),
               "ordinary files cannot be promoted to character-stream routes");
    }
#else
    expect(rejected([] { static_cast<void>(inspect_direct_stream_device("/unsupported/device")); }),
           "non-POSIX character discovery is explicitly unsupported");
#endif
    if (failures == 0) std::cout << "direct stream control tests passed\n";
    return failures == 0 ? 0 : 1;
}
