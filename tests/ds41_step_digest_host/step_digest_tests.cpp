#include "step_digest.hpp"

#include <array>
#include <barrier>
#include <cstdio>
#include <thread>

namespace {
struct Receipt { std::uint64_t hash, tokens; };

// Same threading arrangement as --transport emulate: rank 1 on a peer thread,
// rank 0 on the caller. Each invocation owns its checksum and publishes a copy.
Receipt run_rank(std::barrier<>& round, bool divergent) {
    StepTokenDigest step_tokens;
    for (unsigned i = 0; i < 1000U; ++i) {
        round.arrive_and_wait();
        if (divergent) step_tokens.add(std::array<std::uint32_t, 3>{7U, 8U, 9U});
        else step_tokens.add(std::array<std::uint32_t, 2>{1U, 2U});
    }
    return {step_tokens.hash, step_tokens.tokens};
}

bool pair(bool divergent) {
    std::barrier round(2);
    Receipt peer{};
    std::thread worker([&] { peer = run_rank(round, divergent); });
    const auto caller = run_rank(round, false);
    worker.join();
    StepTokenDigest expected0, expected1;
    for (unsigned i = 0; i < 1000U; ++i) {
        expected0.add(std::array<std::uint32_t, 2>{1U, 2U});
        if (divergent) expected1.add(std::array<std::uint32_t, 3>{7U, 8U, 9U});
        else expected1.add(std::array<std::uint32_t, 2>{1U, 2U});
    }
    return caller.tokens == 2000U && peer.tokens == (divergent ? 3000U : 2000U) &&
           caller.hash == expected0.hash && peer.hash == expected1.hash &&
           ((caller.hash != peer.hash) == divergent);
}
}  // namespace

int main() {
    StepTokenDigest empty, split, joined, changed;
    empty.add(std::array<std::uint32_t, 0>{});
    split.add(std::array<std::uint32_t, 1>{1U});
    split.add(std::array<std::uint32_t, 1>{2U});
    joined.add(std::array<std::uint32_t, 2>{1U, 2U});
    changed.add(std::array<std::uint32_t, 2>{1U, 3U});
    // A token-stream checksum deliberately omits step boundaries. Equal counts
    // must still distinguish these different streams; equality is not a proof.
    if (empty.tokens != 0U || empty.hash != 0xcbf29ce484222325ULL ||
        split.tokens != 2U || split.hash != joined.hash ||
        changed.tokens != joined.tokens || changed.hash == joined.hash ||
        !pair(false) || !pair(true) || !pair(false)) {
        std::fputs("rank-local token-stream checksum regression\n", stderr);
        return 1;
    }
    return 0;
}
