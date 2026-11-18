// Server start helpers (strix/serve/startup_phases.hpp): the phase line and the overlapped / serial start work.
// Host-only.
#include "strix/serve/startup_phases.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
void require(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
}  // namespace

int main() {
    using strix::serve::StartupPhases;
    using Clock = StartupPhases::Clock;
    using std::chrono::milliseconds;
    const auto t0 = Clock::now();
    StartupPhases phases(t0);
    phases.mark("transport", t0 + milliseconds(1012));
    phases.mark("weights", t0 + milliseconds(33822));
    phases.mark("session", t0 + milliseconds(33900));
    require(phases.line("rank 0") == "rank 0 startup: transport 1012 ms, weights 32810 ms, session 78 ms, total 33900 ms",
            "phase line text");
    require(phases.phases().size() == 3 && phases.phases()[1].second == 32810, "phase values");

    require(strix::serve::startup_overlap_lever.default_on, "start overlap is default ON (bitwise: same bytes)");
    require(strix::serve::startup_overlap_lever.exactness == strix::lever::Exactness::bitwise, "start overlap is bitwise");

    // Overlapped: the work runs on its own thread before get(). Serial: it runs at get(), on the caller's thread.
    std::atomic<bool> ran{false};
    const auto caller = std::this_thread::get_id();
    std::thread::id worker;
    auto on = strix::serve::startup_overlap([&] { worker = std::this_thread::get_id(); ran.store(true); return 7; }, true);
    for (int i = 0; i < 2000 && !ran.load(); ++i) std::this_thread::sleep_for(milliseconds(1));
    require(ran.load(), "overlapped work starts before it is collected");
    require(on.get() == 7 && worker != caller, "overlapped work ran on another thread with the same result");

    ran.store(false);
    auto off = strix::serve::startup_overlap([&] { worker = std::this_thread::get_id(); ran.store(true); return 7; }, false);
    std::this_thread::sleep_for(milliseconds(20));
    require(!ran.load(), "serial form does not start early");
    require(off.get() == 7 && worker == caller, "serial form runs at the collection point");

    // A failure of the work reaches the collector in both forms.
    for (const bool enabled : {true, false}) {
        auto bad = strix::serve::startup_overlap([]() -> int { throw std::runtime_error("short read"); }, enabled);
        bool thrown = false;
        try { (void)bad.get(); } catch (const std::runtime_error&) { thrown = true; }
        require(thrown, "work failure is rethrown at get()");
    }
    std::puts("serve startup phases: PASS");
    return 0;
}
