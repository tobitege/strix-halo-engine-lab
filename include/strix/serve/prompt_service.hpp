#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace strix::serve {

// Admit-first prompt order with a regime selector (engine neutral, pure arithmetic).
//
// A decode step has a cost that does not depend on its rows, so a step taken while admitted prompts still wait for
// their prefill pays that cost for fewer rows. Admit-first runs the waiting prefill work before the next decode step,
// bounded by the longest gap the decoding requests may see.
//
// It pays only in the regime where a decode step is not filled otherwise. With few live requests a drafting engine
// fills the spare rows with speculative rows, so an early step is as productive as a late one and holding it gains
// nothing; with many live requests every early step runs short. The selector therefore engages by the live
// concurrency (requests admitted or queued), from `min_live` up.
//
// It only orders work. Which path a prompt takes (separate prefill or prompt rows inside the decode forward) must stay
// a pure function of the prompt: the two paths are different kernels, so a path chosen from the load changes the
// generated tokens.
//
// A step planner maps prompt_before_decode() onto its "prompt work before the decode forward" decision.
struct AdmitFirst {
    enum class Select { automatic, never, always };
    static constexpr std::size_t kDefaultMinLive = 5;
    static constexpr double kDefaultMaxDecodeGapMs = 1000.0;

    Select select = Select::automatic;
    std::size_t min_live = kDefaultMinLive;               // automatic: engage from this many live requests
    double max_decode_gap_ms = kDefaultMaxDecodeGapMs;    // 0 never holds a step

    // The regime switch: does admit-first apply at this live concurrency?
    [[nodiscard]] constexpr bool engaged(std::size_t live) const noexcept {
        return select == Select::always || (select == Select::automatic && live >= min_live);
    }
    // true: skip this decode step and keep serving waiting prompts.
    //   live      requests admitted or queued
    //   waiting   prompts whose prefill is not complete and can make progress now
    //   decoding  requests a decode step would advance now
    //   gap_ms    time since those requests last advanced
    [[nodiscard]] constexpr bool prompt_before_decode(std::size_t live, std::size_t waiting, std::size_t decoding,
                                                      double gap_ms) const noexcept {
        if (!engaged(live) || waiting == 0 || decoding == 0) return false;
        return gap_ms == gap_ms && gap_ms >= 0.0 && gap_ms < max_decode_gap_ms;
    }
    [[nodiscard]] constexpr const char* name() const noexcept {
        return select == Select::always ? "always" : select == Select::never ? "never" : "auto";
    }

    // STRIX_SERVE_ADMIT_FIRST = auto (default) | 0 | 1; STRIX_SERVE_ADMIT_FIRST_MIN_LIVE (auto: default 5);
    // STRIX_SERVE_ADMIT_FIRST_GAP_MS (default 1000). `env(name)` returns the value or an empty string.
    template <class Env>
    [[nodiscard]] static AdmitFirst from_env(Env env) {
        AdmitFirst a;
        if (const std::string s = env("STRIX_SERVE_ADMIT_FIRST"); !s.empty()) {
            if (s == "auto") a.select = Select::automatic;
            else if (s == "0" || s == "off") a.select = Select::never;
            else if (s == "1" || s == "on") a.select = Select::always;
            else throw std::invalid_argument("STRIX_SERVE_ADMIT_FIRST is auto, 0 or 1, not '" + s + "'");
        }
        if (const std::string s = env("STRIX_SERVE_ADMIT_FIRST_MIN_LIVE"); !s.empty()) {
            const long long v = std::stoll(s);
            if (v < 1) throw std::invalid_argument("STRIX_SERVE_ADMIT_FIRST_MIN_LIVE must be at least 1");
            a.min_live = static_cast<std::size_t>(v);
        }
        if (const std::string s = env("STRIX_SERVE_ADMIT_FIRST_GAP_MS"); !s.empty()) {
            a.max_decode_gap_ms = std::stod(s);
            if (!std::isfinite(a.max_decode_gap_ms) || a.max_decode_gap_ms < 0.0)
                throw std::invalid_argument("STRIX_SERVE_ADMIT_FIRST_GAP_MS must be finite and nonnegative");
        }
        return a;
    }
};

}  // namespace strix::serve
