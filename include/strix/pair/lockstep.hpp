#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

// Two-rank lockstep control channel bookkeeping shared by the GLM, DeepSeek 4.1 and MiniMax serve pairs.
// Every control exchange carries the sender's rank, the round it is in and the last round it completed; both ranks
// must agree on all three before the round is accepted. This catches a duplicated or dropped control message (one rank
// a round ahead), a stale replay and a pair started with the same rank on both hosts, before either side acts on it.
namespace strix::pair {

inline constexpr std::uint64_t kNoRound = ~std::uint64_t{0};

class ProtocolError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct LockstepHeader {
    std::uint64_t round = 0;
    std::uint64_t ack_round = kNoRound;  // the last round this rank completed (kNoRound: none)
    std::uint32_t rank = 0;
};

// The round part alone, for a wire format with no room for ack_round / rank (DeepSeek 4.1's full 512-byte block).
inline void check_round(std::uint64_t mine, std::uint64_t peer) {
    if (peer != mine)
        throw ProtocolError("control channel: duplicate / stale round " + std::to_string(peer) + ", expected " +
                            std::to_string(mine));
}

// Validates the peer's header against the header this rank sent for the same exchange.
inline void check_pair(const LockstepHeader& mine, const LockstepHeader& peer) {
    check_round(mine.round, peer.round);
    if (peer.ack_round != mine.ack_round) throw ProtocolError("control channel: preceding round not acknowledged");
    if (peer.rank == mine.rank) throw ProtocolError("control channel: peer reports the same rank");
    if (peer.rank > 1U) throw ProtocolError("control channel: peer rank " + std::to_string(peer.rank) + " out of range");
}

class Lockstep {
public:
    explicit Lockstep(std::uint32_t rank, std::uint64_t first_round = 0) : rank_(rank), round_(first_round) {}

    // The header this rank sends for the current round.
    [[nodiscard]] LockstepHeader header() const noexcept {
        return {round_, round_ == 0 ? kNoRound : round_ - 1, rank_};
    }
    // Validates the peer's header for the current round and advances to the next one. Throws ProtocolError (and does
    // not advance) on any mismatch.
    void accept(const LockstepHeader& peer) {
        check(peer);
        ++round_;
    }
    void check(const LockstepHeader& peer) const { check_pair(header(), peer); }

    [[nodiscard]] std::uint64_t round() const noexcept { return round_; }
    [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }

private:
    std::uint32_t rank_ = 0;
    std::uint64_t round_ = 0;
};

}  // namespace strix::pair
