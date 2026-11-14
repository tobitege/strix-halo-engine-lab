#pragma once
// Framing of one checked payload exchange between two ranks that already share its size (host only; the bytes are
// moved by the caller's `wire(out, in)`, a symmetric transport exchange). The engine supplies its 64-byte identity
// record (Header) and its per-chunk CRC lists; this file owns the order of the messages.
//
// rendezvous (the original form, byte for byte):
//     identity <-> identity | chunk 0 | ... | chunk N-1 | [identity' + N CRCs <-> identity' + N CRCs]
//   N + 2 exchanges with CRC, N + 1 without. The identities are compared before any payload moves.
// joined (when both ranks offer it in the hello):
//     chunk 0 | ... | chunk N-2 | chunk N-1 + record
//   The identity rides behind the last chunk in the same exchange, so the separate identity rendezvous is gone. With
//   CRC and N >= 2 the record is identity' + N CRCs (the sender's CRC lane has had N - 1 chunks of wire time): N
//   exchanges. With CRC and N == 1 the record is the identity only and the CRC list keeps its own exchange (waiting
//   for the CRC of the whole payload before sending it would cost more than the rendezvous): 2 exchanges. Without
//   CRC: N exchanges. The record needs `tail_bytes()` of room behind the payload in the TX and RX slots.
// Either way the caller compares identity and CRCs before it releases the payload to its consumer.
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::xport {
enum FrameForm : std::uint32_t { frame_rendezvous = 1U << 0U, frame_joined = 1U << 1U };
inline constexpr std::uint64_t kFrameHelloMagic = 0x314c4548454d5246ULL; // "FRMEHEL1"

// One 64-byte hello each way, once per connection: the forms both ranks offer.
struct FrameHello {
    std::uint64_t magic = kFrameHelloMagic;
    std::uint32_t forms = 0, reserved = 0;
    std::uint64_t layout = 0;  // the caller's framing invariants (chunk size, CRC mode): equal on both ranks
    std::uint64_t zero[5] = {};
};
static_assert(sizeof(FrameHello) == 64);
// `layout`: whatever fixes the part sizes of every later message. It is agreed here, before any payload moves: the
// joined form checks the identity only behind the payload, and ranks that split a message differently would stall.
template<class Wire>
[[nodiscard]] std::uint32_t negotiate_frame_forms(Wire&& wire, std::uint32_t offered, std::uint64_t layout = 0) {
    FrameHello mine, peer;
    mine.forms = offered;
    mine.layout = layout;
    wire(std::as_bytes(std::span(&mine, 1)), std::as_writable_bytes(std::span(&peer, 1)));
    if (peer.magic != kFrameHelloMagic || peer.reserved != 0U)
        throw std::runtime_error("framed exchange: the peer sent no frame hello (the ranks disagree on the framing switch)");
    if (peer.layout != layout)
        throw std::runtime_error("framed exchange: the ranks disagree on the frame layout (chunk size or CRC mode): mine " +
                                 std::to_string(layout) + " / peer " + std::to_string(peer.layout));
    return offered & peer.forms;
}

template<class Header>
struct FrameResult {
    Header first{};                      // the peer's identity record that travelled with or before the payload
    std::uint64_t first_magic = 0;       // the magic `first` must carry
    Header trailer{};                    // with CRC: the peer's identity' in front of its CRC list
    const std::byte* crcs = nullptr;     // with CRC: the peer's N CRCs (valid until the RX slot or scratch is reused)
    std::chrono::steady_clock::time_point landed{}; // the last payload byte is in the RX slot
};

template<class Header>
class FramedExchange {
    static_assert(sizeof(Header) == 64);
public:
    explicit FramedExchange(std::size_t max_chunks)
        : out_(sizeof(Header) + max_chunks * 4U), in_(out_.size()) {}
    // Room the joined form needs behind the payload in every TX and RX slot.
    [[nodiscard]] static constexpr std::size_t tail_bytes(std::size_t max_chunks) noexcept {
        return sizeof(Header) + max_chunks * 4U;
    }
    // wire(out, in): one symmetric exchange. agreed(peer): the peer's identity, before any payload moves.
    // landed(k): chunks [0, k) are in the RX slot. tx_crcs(): the N CRCs of the TX payload (may wait for them).
    template<class Wire, class Agreed, class Landed, class TxCrcs>
    [[nodiscard]] FrameResult<Header> rendezvous(Wire&& wire, const Header& mine, std::uint64_t trailer_magic,
        const std::byte* tx, std::byte* rx, std::size_t bytes, std::size_t chunk, bool crc,
        Agreed&& agreed, Landed&& landed, TxCrcs&& tx_crcs) {
        FrameResult<Header> result;
        result.first_magic = mine.magic;
        wire(std::as_bytes(std::span(&mine, 1)), std::as_writable_bytes(std::span(&result.first, 1)));
        agreed(result.first);
        const std::size_t count = (bytes + chunk - 1U) / chunk;
        for (std::size_t k = 0; k < count; ++k) {
            const std::size_t offset = k * chunk, n = bytes - offset < chunk ? bytes - offset : chunk;
            wire(std::span<const std::byte>(tx + offset, n), std::span<std::byte>(rx + offset, n));
            landed(k + 1U);
        }
        result.landed = std::chrono::steady_clock::now();
        if (crc) crc_list(wire, mine, trailer_magic, count, tx_crcs(), result);
        return result;
    }
    // tx / rx: the slots, with tail_bytes() of room behind `bytes`. The record is written behind the TX payload.
    template<class Wire, class Landed, class TxCrcs>
    [[nodiscard]] FrameResult<Header> joined(Wire&& wire, const Header& mine, std::uint64_t trailer_magic,
        std::byte* tx, std::byte* rx, std::size_t bytes, std::size_t chunk, bool crc, Landed&& landed, TxCrcs&& tx_crcs) {
        FrameResult<Header> result;
        const std::size_t count = (bytes + chunk - 1U) / chunk;
        for (std::size_t k = 0; k + 1U < count; ++k) {
            const std::size_t offset = k * chunk;
            wire(std::span<const std::byte>(tx + offset, chunk), std::span<std::byte>(rx + offset, chunk));
            landed(k + 1U);
        }
        const std::size_t offset = (count - 1U) * chunk, n = bytes - offset;
        const bool with_crcs = crc && count >= 2U;
        Header record = mine;
        if (with_crcs) record.magic = trailer_magic;
        std::size_t extra = sizeof(Header);
        std::memcpy(tx + bytes, &record, sizeof(record));
        if (with_crcs) {
            std::memcpy(tx + bytes + sizeof(Header), tx_crcs(), count * 4U);
            extra += count * 4U;
        }
        wire(std::span<const std::byte>(tx + offset, n + extra), std::span<std::byte>(rx + offset, n + extra));
        landed(count);
        result.landed = std::chrono::steady_clock::now();
        std::memcpy(&result.first, rx + bytes, sizeof(Header));
        result.first_magic = record.magic;
        if (with_crcs) {
            result.trailer = result.first;
            result.crcs = rx + bytes + sizeof(Header);
        } else if (crc) {
            crc_list(wire, mine, trailer_magic, count, tx_crcs(), result);
        }
        return result;
    }
private:
    std::vector<std::byte> out_, in_; // the identity' + CRC list record of the separate exchange, allocated once
    template<class Wire>
    void crc_list(Wire&& wire, Header identity, std::uint64_t trailer_magic, std::size_t count, const std::uint32_t* crcs,
        FrameResult<Header>& result) {
        const std::size_t size = sizeof(Header) + count * 4U;
        if (size > out_.size()) throw std::logic_error("framed exchange: more chunks than the exchange was sized for");
        identity.magic = trailer_magic;
        std::memcpy(out_.data(), &identity, sizeof(identity));
        std::memcpy(out_.data() + sizeof(Header), crcs, count * 4U);
        wire(std::span<const std::byte>(out_.data(), size), std::span<std::byte>(in_.data(), size));
        std::memcpy(&result.trailer, in_.data(), sizeof(Header));
        result.crcs = in_.data() + sizeof(Header);
    }
};
} // namespace strix::xport
