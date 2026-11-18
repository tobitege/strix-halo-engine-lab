#pragma once
// strix::pair::state::relay_piece: the relay leg of a shared piece (strix/pair/state_owners.hpp, Owner::shared). The
// prefill host sends the piece once, to the relay rank; this forwards it to the peer on the pair interconnect. Run by
// BOTH ranks at the same point (a collective). Model-agnostic: the family passes the bytes, the exchange and the hash.
//
//   header   {ok, bytes, hash} both ways: a rank that cannot take part (nothing staged, wrong size) makes both throw
//            together, before any payload moves.
//   payload  chunks of at most `chunk` bytes, relay rank -> peer. xchg(local, remote, seq) is one symmetric pair
//            exchange (remote empty = not wanted); the peer sends its destination block back, which the source ignores.
//   check    both ranks exchange verification status, then both throw if the received hash differs.
// Returns the next free seq (seq0 + 2 + chunks), so a caller can relay several pieces on one tag space. `hash` is
// state_hash below unless the family already links a faster one (bp_codec::xxh64).
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix::pair::state {

// Transport-defect check of a relayed piece (not an adversary check): four independent 64-bit multiply-rotate lanes
// over 32-byte strides, then the tail and the length, with a final avalanche. Header only, ~GB/s per core.
[[nodiscard]] inline std::uint64_t state_hash(const std::uint8_t* p, std::size_t n) noexcept {
    constexpr std::uint64_t k1 = 0x9E3779B185EBCA87ULL, k2 = 0xC2B2AE3D27D4EB4FULL;
    const auto rotl = [](std::uint64_t x, int r) { return (x << r) | (x >> (64 - r)); };
    std::uint64_t h[4] = {k1, k2, ~k1, ~k2};
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32)
        for (int l = 0; l < 4; ++l) {
            std::uint64_t w;
            std::memcpy(&w, p + i + 8 * l, 8);
            h[l] = rotl(h[l] ^ (w * k2), 31) * k1;
        }
    std::uint64_t t = rotl(h[0], 1) + rotl(h[1], 7) + rotl(h[2], 12) + rotl(h[3], 18);
    for (; i < n; ++i) t = rotl(t ^ (p[i] * k1), 11) * k2;
    t ^= static_cast<std::uint64_t>(n) * k1;
    t ^= t >> 33;
    t *= k2;
    t ^= t >> 29;
    return t;
}

struct RelayStats {
    std::uint64_t bytes = 0, chunks = 0;
    double ms = 0;
};

// Bind ordinal exchanges to the same session, prompt, piece and row range before moving payloads.
// Include end markers as pieces so an early end also makes both ranks reject together.
using RelayIdentity = std::array<std::uint64_t, 6>;  // session, prompt hash, piece, row0, rows, session-name hash
template <class Xchg>
void relay_agree(const RelayIdentity& identity, Xchg&& xchg, std::uint64_t seq) {
    RelayIdentity peer{};
    xchg(std::as_bytes(std::span(identity)), std::as_writable_bytes(std::span(peer)), seq);
    if (peer != identity) throw std::runtime_error("state relay: session / piece / row range mismatch");
}

template <class Xchg, class Hash>
std::uint64_t relay_piece(bool source, bool ok, std::span<std::uint8_t> bytes, std::size_t chunk, Xchg&& xchg, Hash&& hash,
                          std::uint64_t seq0 = 0, RelayStats* stats = nullptr) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint64_t mine = ok && source ? static_cast<std::uint64_t>(hash(bytes.data(), bytes.size())) : 0;
    const std::uint64_t hdr[3] = {ok && chunk > 0 ? 1U : 0U, bytes.size(), mine};
    std::uint64_t peer[3] = {};
    xchg(std::as_bytes(std::span(hdr)), std::as_writable_bytes(std::span(peer)), seq0);
    if (!ok) throw std::runtime_error("state relay: this rank has nothing usable to relay / receive");
    if (chunk == 0) throw std::runtime_error("state relay: chunk size 0");
    if (peer[0] != 1U) throw std::runtime_error("state relay: the peer cannot relay / receive");
    if (peer[1] != bytes.size())
        throw std::runtime_error("state relay: piece size " + std::to_string(bytes.size()) + " here, " + std::to_string(peer[1]) +
                                 " on the peer");
    std::uint64_t seq = seq0 + 1;
    for (std::size_t o = 0; o < bytes.size(); o += chunk, ++seq) {
        const std::size_t n = std::min(chunk, bytes.size() - o);
        const std::span<std::uint8_t> part(bytes.data() + o, n);
        if (source) xchg(std::as_bytes(part), std::span<std::byte>(), seq);
        else xchg(std::as_bytes(part), std::as_writable_bytes(part), seq);  // the block it sends back is copied out first
    }
    const std::uint64_t verified = source || static_cast<std::uint64_t>(hash(bytes.data(), bytes.size())) == peer[2] ? 1U : 0U;
    std::uint64_t peer_verified = 0;
    xchg(std::as_bytes(std::span(&verified, 1)), std::as_writable_bytes(std::span(&peer_verified, 1)), seq++);
    if (verified != 1U || peer_verified != 1U)
        throw std::runtime_error("state relay: relayed piece hash mismatch (" + std::to_string(bytes.size()) + " bytes)");
    if (stats) {
        stats->bytes += bytes.size();
        stats->chunks += seq - seq0 - 2;
        stats->ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    return seq;
}

}  // namespace strix::pair::state
