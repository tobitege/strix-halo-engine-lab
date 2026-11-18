// Draft-chain segmentation for MTP / speculative drafters (shared host logic, any backend).
//
// A drafter that launches its whole chain of `depth` steps at once (one graph replay) wastes the steps after the first
// draft whose probability falls below the stop threshold: the verify discards them. Splitting the chain into two
// launches [0, a) and [a, depth), with the second issued only when the first did not stop, keeps every kept draft and
// every verified token identical (the steps that still run are the same launches on the same inputs) and saves
// depth - a steps whenever the chain stops inside the first segment, at the price of one host turn when it does not.
//
// DraftSplit keeps a histogram of how many drafts each chain kept and picks a to minimise the expected steps
//     a + P(kept > a) * (depth - a + host_turn)
// (host_turn = the extra sync + launch in draft-step units). STRIX_MTP_DRAFT_SPLIT (or the backend's own variable):
// unset / "auto" = that choice, 0 = one segment (whole chain, the old behaviour), N = fixed first segment N.
#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "strix/env_flags.hpp"

namespace strix::mtp {

class DraftSplit {
public:
    static constexpr std::uint32_t kBins = 16;

    // mode: -1 auto, 0 off, N > 0 fixed first segment. host_turn in draft-step units.
    explicit DraftSplit(int mode = -1, double host_turn = 0.25, std::uint64_t min_samples = 8)
        : mode_(mode), host_turn_(host_turn), min_samples_(min_samples) {}

    // -1 / 0 / N from an environment variable (unset, empty or "auto" -> -1), recorded for the effective config.
    static int mode_from_env(const char* name) {
        const char* e = strix::envcfg::text(name);
        if (!e || !*e || std::string(e) == "auto") return -1;
        return std::atoi(e);
    }

    // First segment length for a chain of `depth` steps (== depth: no split).
    std::uint32_t first_segment(std::uint32_t depth) const {
        if (mode_ == 0 || depth < 2) return depth;
        if (mode_ > 0) return static_cast<std::uint32_t>(mode_) < depth ? static_cast<std::uint32_t>(mode_) : depth;
        std::uint64_t total = 0;
        for (std::uint32_t n = 0; n <= depth && n < kBins; ++n) total += kept_[n];
        if (total < min_samples_) return depth < 2U ? depth : 2U;
        std::uint32_t best = depth;
        double best_cost = depth;
        for (std::uint32_t a = 1; a < depth; ++a) {
            std::uint64_t more = 0;
            for (std::uint32_t n = a + 1; n <= depth && n < kBins; ++n) more += kept_[n];
            const double cost = a + static_cast<double>(more) / static_cast<double>(total) * (depth - a + host_turn_);
            if (cost < best_cost) best_cost = cost, best = a;
        }
        return best;
    }

    // Record one finished chain that kept `kept` drafts (the stop rule keeps the low draft itself).
    void note(std::uint32_t kept) { ++kept_[kept < kBins ? kept : kBins - 1]; }

    int mode() const { return mode_; }
    std::uint64_t kept(std::uint32_t n) const { return n < kBins ? kept_[n] : 0; }

private:
    int mode_;
    double host_turn_;
    std::uint64_t min_samples_;
    std::array<std::uint64_t, kBins> kept_{};
};

}  // namespace strix::mtp
