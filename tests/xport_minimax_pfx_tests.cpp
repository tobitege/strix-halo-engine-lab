#include "minimax/pfx/pf_wire.hpp"
#include "minimax/pfx/pf_route.hpp"
#include "xport/framed_exchange.hpp"
#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
namespace px = strix::minimax::pfx;
using Message = std::vector<std::byte>;
void require(bool value) { if (!value) throw std::runtime_error("MiniMax pfx wire/route fixture"); }
struct Options {
    std::uint64_t epoch = 7, plan = 0x5846504dULL;
    std::size_t chunk = 65536;
    std::uint32_t rank = 0;
    bool crc = true;
    int stream = 3;
};
struct FakeTransport {
    std::vector<Message> messages;
    void exchange(std::span<const std::byte> source, std::span<std::byte> destination) {
        messages.emplace_back(source.begin(), source.end());
        std::copy(source.begin(), source.end(), destination.begin());
    }
};
// Frozen parent serializer: independent of PrefillTraits, including its native
// 64-byte identity, CRC bit, chunk boundaries and trailer CRC byte ordering.
std::vector<Message> parent_messages(const Options& options, std::uint64_t sequence,
                                    const px::Tag& tag, const Message& payload,
                                    const std::vector<std::uint32_t>& crcs) {
    struct OldHeader {
        std::uint64_t magic, epoch, plan, sequence, request, bytes;
        std::uint32_t kind, layer, chunk, rank;
    };
    static_assert(sizeof(OldHeader) == 64);
    OldHeader identity{0x3148584650584d4dULL, options.epoch, options.plan, sequence,
                       tag.request, payload.size(), tag.kind, tag.layer,
                       static_cast<std::uint32_t>(options.chunk) | (options.crc ? 0x80000000U : 0U), options.rank};
    std::vector<Message> result;
    result.emplace_back(sizeof(identity));
    std::memcpy(result.back().data(), &identity, sizeof(identity));
    for (std::size_t offset = 0; offset < payload.size(); offset += options.chunk) {
        const auto bytes = std::min(options.chunk, payload.size() - offset);
        result.emplace_back(payload.data() + offset, payload.data() + offset + bytes);
    }
    if (options.crc) {
        identity.magic = 0x3154584650584d4dULL;
        result.emplace_back(sizeof(identity) + crcs.size() * 4);
        std::memcpy(result.back().data(), &identity, sizeof(identity));
        std::memcpy(result.back().data() + sizeof(identity), crcs.data(), crcs.size() * 4);
    }
    return result;
}
struct FakeExchanger {
    struct View { const void* local; const void* peer; };
    Options settings;
    FakeTransport transport;
    Message tx_slot, rx_slot;
    std::vector<std::uint32_t> crcs;
    std::vector<char> operations;
    std::uint64_t sequence = 9;
    const Options& options() const { return settings; }
    std::uint64_t sent() const { return sequence; }
    void* tx() { return tx_slot.data(); }
    void send_copy(const void* source, std::size_t bytes, const px::Tag& tag) {
        operations.push_back('c');
        std::memcpy(tx_slot.data(), source, bytes);
        send_wire(bytes, tag);
    }
    void send(std::size_t bytes, const px::Tag& tag) {
        operations.push_back('p');
        send_wire(bytes, tag);
    }
    // The engine's own framing code (xport/framed_exchange.hpp), as Exchanger::process() calls it.
    strix::xport::FramedExchange<px::Header> frame{64};
    bool joined = false;
    strix::xport::FrameResult<px::Header> last;
    std::vector<std::size_t> landed_chunks;
    unsigned crc_waits = 0;
    void send_wire(std::size_t bytes, const px::Tag& tag) {
        const auto identity = px::PrefillTraits::header(settings, ++sequence, bytes, tag);
        const auto wire = [&](std::span<const std::byte> out, std::span<std::byte> in) { transport.exchange(out, in); };
        const auto landed = [&](std::size_t chunks) { landed_chunks.push_back(chunks); };
        const auto tx_crcs = [&] { ++crc_waits; return static_cast<const std::uint32_t*>(crcs.data()); };
        landed_chunks.clear(); crc_waits = 0;
        if (joined) last = frame.joined(wire, identity, px::kTrailerMagic, tx_slot.data(), rx_slot.data(), bytes, settings.chunk,
            settings.crc, landed, tx_crcs);
        else last = frame.rendezvous(wire, identity, px::kTrailerMagic, tx_slot.data(), rx_slot.data(), bytes, settings.chunk,
            settings.crc, [&](const px::Header& peer) { require(std::memcmp(&peer, &identity, sizeof(peer)) == 0); }, landed, tx_crcs);
        require(std::memcmp(&last.first, &identity, 8) != 0 || last.first_magic == px::kHeaderMagic);
        require(last.first.magic == last.first_magic && last.first.sequence == identity.sequence && last.first.bytes == bytes);
    }
    View recv() { operations.push_back('w'); return {tx_slot.data(), rx_slot.data()}; }
    void retire() { operations.push_back('r'); }
    void drain() { operations.push_back('d'); }
};
void check(std::size_t rows, std::size_t width, bool crc, std::uint32_t rank, bool inplace) {
    FakeExchanger runtime;
    runtime.settings.crc = crc;
    runtime.settings.rank = rank;
    const auto bytes = px::PrefillTraits::payload_bytes(rows, width);
    Message payload(bytes);
    for (std::size_t i = 0; i < bytes; ++i) payload[i] = static_cast<std::byte>(i % 251);
    runtime.tx_slot = payload;
    runtime.rx_slot.resize(bytes);
    require(!runtime.joined);
    const auto chunks = (bytes + runtime.settings.chunk - 1) / runtime.settings.chunk;
    for (std::size_t i = 0; i < chunks; ++i) runtime.crcs.push_back(static_cast<std::uint32_t>(i) ^ 0x92abcdefU);
    const px::Tag tag{rank + 1, 17, 23};
    for (unsigned iteration = 0; iteration < 2; ++iteration) {
        runtime.transport.messages.clear();
        runtime.operations.clear();
        const auto expected = parent_messages(runtime.settings, runtime.sent() + 1, tag, payload, runtime.crcs);
        px::Route<FakeExchanger, px::Tag> route(runtime);
        const auto ticket = route.post(inplace ? runtime.tx() : payload.data(), bytes, tag);
        const auto view = route.wait(ticket);
        require(view.local == runtime.tx() && view.peer == runtime.rx_slot.data());
        require(runtime.rx_slot == payload);
        require(runtime.transport.messages == expected);
        // The engine enqueues its rank-ordered consumer here, then retires once.
        runtime.operations.push_back('k');
        runtime.retire();
        route.release(ticket);
        require(runtime.operations == std::vector<char>({inplace ? 'p' : 'c', 'w', 'k', 'r'}));
    }
}
// The joined form against a loopback wire: fewer exchanges, the same payload bytes, the identity and (with CRC and
// two or more chunks) the CRC list behind the last chunk; with one chunk the CRC list keeps its own exchange.
void check_joined(std::size_t bytes, bool crc) {
    FakeExchanger runtime;
    runtime.joined = true;
    runtime.settings.crc = crc;
    const auto chunk = runtime.settings.chunk;
    const auto count = (bytes + chunk - 1) / chunk;
    const auto tail = strix::xport::FramedExchange<px::Header>::tail_bytes(count);
    Message payload(bytes);
    for (std::size_t i = 0; i < bytes; ++i) payload[i] = static_cast<std::byte>((i * 7U + 3U) % 253);
    runtime.tx_slot = payload; runtime.tx_slot.resize(bytes + tail);
    runtime.rx_slot.assign(bytes + tail, std::byte{0xee});
    for (std::size_t i = 0; i < count; ++i) runtime.crcs.push_back(static_cast<std::uint32_t>(i) * 0x01010101U ^ 0x5eed1234U);
    const px::Tag tag{2, 5, 41};
    runtime.send_wire(bytes, tag);
    const auto identity = px::PrefillTraits::header(runtime.settings, runtime.sent(), bytes, tag);
    const auto& messages = runtime.transport.messages;
    const bool with_list = crc && count >= 2;
    require(messages.size() == count + (crc && count == 1 ? 1U : 0U));
    // Payload: the chunks in order, nothing else in front of them.
    Message moved;
    for (std::size_t k = 0; k < count; ++k) {
        const auto n = std::min(chunk, bytes - k * chunk);
        require(messages[k].size() == n + (k + 1 == count ? sizeof(px::Header) + (with_list ? count * 4 : 0) : 0));
        moved.insert(moved.end(), messages[k].begin(), messages[k].begin() + static_cast<std::ptrdiff_t>(n));
    }
    require(moved == payload);
    require(std::equal(payload.begin(), payload.end(), runtime.rx_slot.begin()));
    // The record behind the last chunk.
    px::Header record;
    std::memcpy(&record, messages[count - 1].data() + (messages[count - 1].size() - sizeof(px::Header) - (with_list ? count * 4 : 0)), sizeof(record));
    auto expected = identity;
    if (with_list) expected.magic = px::kTrailerMagic;
    require(std::memcmp(&record, &expected, sizeof(record)) == 0);
    require(runtime.last.first_magic == expected.magic);
    if (crc) {
        require(runtime.last.trailer.magic == px::kTrailerMagic && runtime.last.trailer.sequence == identity.sequence);
        require(std::memcmp(runtime.last.crcs, runtime.crcs.data(), count * 4) == 0);
        require(runtime.crc_waits == 1);
        if (!with_list) { // one chunk: the original CRC-list record in its own exchange
            Message trailer(sizeof(px::Header) + 4);
            px::PrefillTraits::trailer(trailer.data(), identity, runtime.crcs.data(), 1);
            require(messages.back() == trailer);
        }
    } else require(runtime.crc_waits == 0 && runtime.last.crcs == nullptr);
    std::vector<std::size_t> order(count);
    for (std::size_t k = 0; k < count; ++k) order[k] = k + 1;
    require(runtime.landed_chunks == order);
}
void check_hello() {
    using namespace strix::xport;
    FakeTransport transport;
    const auto wire = [&](std::span<const std::byte> out, std::span<std::byte> in) { transport.exchange(out, in); };
    require(negotiate_frame_forms(wire, frame_rendezvous | frame_joined) == (frame_rendezvous | frame_joined));
    require(transport.messages.size() == 1 && transport.messages[0].size() == 64);
    // A peer that offers only the original form, and a peer that sends an identity record instead of a hello.
    const auto legacy = [&](std::span<const std::byte>, std::span<std::byte> in) {
        FrameHello peer; peer.forms = frame_rendezvous; std::memcpy(in.data(), &peer, sizeof(peer));
    };
    require(negotiate_frame_forms(legacy, frame_rendezvous | frame_joined) == frame_rendezvous);
    const auto old = [&](std::span<const std::byte>, std::span<std::byte> in) {
        px::Header header; std::memcpy(in.data(), &header, sizeof(header));
    };
    bool refused = false;
    try { static_cast<void>(negotiate_frame_forms(old, frame_rendezvous | frame_joined)); } catch (const std::runtime_error&) { refused = true; }
    require(refused);
    // Ranks with different chunk sizes / CRC modes are refused at the hello, before any payload moves.
    const auto other = [&](std::span<const std::byte>, std::span<std::byte> in) {
        FrameHello peer; peer.forms = frame_rendezvous | frame_joined; peer.layout = 4194304; std::memcpy(in.data(), &peer, sizeof(peer));
    };
    refused = false;
    try { static_cast<void>(negotiate_frame_forms(other, frame_rendezvous | frame_joined, 1048576)); } catch (const std::runtime_error&) { refused = true; }
    require(refused);
    require(negotiate_frame_forms(other, frame_rendezvous | frame_joined, 4194304) == (frame_rendezvous | frame_joined));
}
} // namespace
int main() {
    for (const auto rows : std::array<std::size_t, 7>{1, 8, 32, 1024, 4096, 8192, 32768})
        for (const bool crc : {false, true})
            for (const std::uint32_t rank : {0U, 1U})
                for (const bool inplace : {false, true}) check(rows, 8, crc, rank, inplace);
    check(3, 6144, true, 0, false); // full hidden rows and a partial final wire chunk
    for (const std::size_t bytes : {std::size_t{4}, std::size_t{4096}, std::size_t{65536}, std::size_t{65540},
             std::size_t{3 * 65536}, std::size_t{3 * 65536 + 100}, std::size_t{16 * 65536}})
        for (const bool crc : {false, true}) check_joined(bytes, crc);
    check_hello();
}
