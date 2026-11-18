#include "step_digest.hpp"

#include <array>
#include <barrier>
#include <cstdio>
#include <thread>

namespace {
struct Receipt { std::uint64_t hash, tokens; StepProfile profile; };

// Same threading arrangement as --transport emulate: rank 1 on a peer thread,
// rank 0 on the caller. Each invocation owns its checksum and publishes a copy.
Receipt run_rank(std::barrier<>& round, bool divergent) {
    StepTokenDigest step_tokens;
    StepProfile step_profile;
    strix::deepseek41::BatchStepStats stats;
    stats.rows = divergent ? 16U : 7U;
    stats.seconds = divergent ? 2.0 : 1.0;
    stats.exchange_seconds = 0.5;
    stats.engram_seconds = 0.25;
    stats.engram_wait_seconds = 0.125;
    stats.union_experts = {2U, 3U};
    stats.expert_bytes = divergent ? 512U : 256U;
    for (unsigned i = 0; i < 1000U; ++i) {
        round.arrive_and_wait();
        if (divergent) step_tokens.add(std::array<std::uint32_t, 3>{7U, 8U, 9U});
        else step_tokens.add(std::array<std::uint32_t, 2>{1U, 2U});
        step_profile.add(stats);
    }
    stats.rows = 17U;  // outside the extended bin range
    step_profile.add(stats);
    return {step_tokens.hash, step_tokens.tokens, step_profile};
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
    const auto valid_profile = [](const StepProfile& profile, bool different) {
        for (std::size_t r = 0; r < profile.bins.size(); ++r) {
            const auto& b = profile.bins[r];
            if (r != (different ? 16U : 7U)) { if (b.steps != 0U) return false; continue; }
            if (b.steps != 1000U || b.seconds != (different ? 2000.0 : 1000.0) ||
                b.exchange != 500.0 || b.engram_read != 250.0 || b.engram_wait != 125.0 ||
                b.unions != 5000U || b.layers != 2000U || b.expert_bytes != (different ? 512000U : 256000U)) return false;
        }
        return true;
    };
    return valid_profile(caller.profile, false) && valid_profile(peer.profile, divergent) &&
           caller.tokens == 2000U && peer.tokens == (divergent ? 3000U : 2000U) &&
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
