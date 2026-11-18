// Server start: one timing line per process, and independent host work overlapped with the weight load.
//
// StartupPhases: mark("name") closes the interval since the previous mark; line() gives
//   "<who> startup: transport 1012 ms, weights 32810 ms, ..., total 44120 ms"
// so a slow start names its phase (weight load, waiting for the peer rank, auxiliary assets, session admission)
// without a profiler. Every family's server can print it before its READY line.
//
// startup_overlap(work): host-only start work that does not depend on the model weights (reading an auxiliary
// model file, a tokenizer, a table) starts on its own thread before the weight load and is collected where it was
// done before. Same bytes, same order of device work; only the wall clock overlaps. STRIX_SERVE_STARTUP_OVERLAP=0
// keeps the serial form (the work runs at the collection point).
#pragma once

#include "strix/lever.hpp"

#include <chrono>
#include <future>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::serve {

inline constexpr lever::Decl startup_overlap_lever{
    "serve-startup-overlap", "STRIX_SERVE_STARTUP_OVERLAP", lever::Exactness::bitwise, true, {}, {},
    "pair server start, auxiliary drafter read beside the weight load (docs/serving.md, server start)"};

[[nodiscard]] inline bool startup_overlap_enabled() { return lever::select(startup_overlap_lever, {}).on; }

// `work` must be host-only and independent of the caller's device state until get().
template <class Work>
[[nodiscard]] auto startup_overlap(Work&& work, bool enabled = startup_overlap_enabled()) {
    return std::async(enabled ? std::launch::async : std::launch::deferred, std::forward<Work>(work));
}

class StartupPhases {
public:
    using Clock = std::chrono::steady_clock;
    explicit StartupPhases(Clock::time_point origin = Clock::now()) : origin_(origin), last_(origin) {}
    void mark(std::string_view name, Clock::time_point now = Clock::now()) {
        phases_.emplace_back(std::string(name), ms(last_, now));
        last_ = now;
    }
    [[nodiscard]] std::string line(std::string_view who) const {
        std::string out(who);
        out += " startup:";
        for (const auto& [name, value] : phases_) out += " " + name + " " + std::to_string(value) + " ms,";
        out += " total " + std::to_string(ms(origin_, last_)) + " ms";
        return out;
    }
    [[nodiscard]] const std::vector<std::pair<std::string, long long>>& phases() const noexcept { return phases_; }

private:
    static long long ms(Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    }
    Clock::time_point origin_, last_;
    std::vector<std::pair<std::string, long long>> phases_;
};

}  // namespace strix::serve
