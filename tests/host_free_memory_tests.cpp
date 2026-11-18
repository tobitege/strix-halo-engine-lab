// Host test of the free-memory preparation policy (core/host_free_memory.hpp): the pure plan on synthetic meminfo
// values, the disabled switch, and one real pass that must be a no-op when nothing is asked.
#include "core/host_free_memory.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#if defined(_MSC_VER)
#include <crtdbg.h>
#endif

namespace {
using strix::core::HostMemory;
using strix::core::kFreePrepGiB;
using strix::core::plan_free_prepare;

int failures = 0;
void require(bool ok, const char* what) {
    if (!ok) { std::cerr << "FAIL: " << what << "\n"; ++failures; }
}
HostMemory mem(std::uint64_t free_gib, std::uint64_t available_gib) {
    HostMemory m;
    m.free = free_gib * kFreePrepGiB;
    m.available = available_gib * kFreePrepGiB;
    m.ok = true;
    return m;
}
}  // namespace

int main() {
#if defined(_MSC_VER)
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    const std::uint64_t G = kFreePrepGiB;
    // Free memory covers the load: nothing to do (the common case; rank 0 of the measured pair).
    require(plan_free_prepare(64 * G, mem(110, 125)).populate == 0, "free covers need");
    require(plan_free_prepare(64 * G, mem(64, 125)).populate == 0, "free equals need");
    // Page cache holds the memory (the measured peer: 25 GiB free, 125 available): populate up to the need.
    {
        const auto p = plan_free_prepare(64 * G, mem(25, 125));
        require(p.populate == 64 * G && p.target == 64 * G, "cache full: target is the need");
    }
    // Never beyond MemAvailable - reserve.
    {
        const auto p = plan_free_prepare(100 * G, mem(10, 60));
        require(p.populate == 56 * G, "capped at available - 4 GiB reserve");
        const auto q = plan_free_prepare(100 * G, mem(10, 60), 10 * G);
        require(q.populate == 50 * G, "explicit reserve");
    }
    // Nothing reclaimable: available is all free memory, or below the reserve.
    require(plan_free_prepare(100 * G, mem(50, 52)).populate == 0, "no cache beyond the reserve");
    require(plan_free_prepare(100 * G, mem(1, 3)).populate == 0, "available below the reserve");
    // A shortfall under the minimum pass is left to the kernel.
    require(plan_free_prepare(64 * G + (G / 2), mem(64, 125)).populate == 0, "shortfall below min_take");
    require(plan_free_prepare(66 * G, mem(64, 125)).populate == 66 * G, "shortfall above min_take");
    // Unreadable meminfo: no action.
    require(plan_free_prepare(64 * G, HostMemory{}).populate == 0, "meminfo unreadable");
    // The populate size is a multiple of 2 MiB.
    require(plan_free_prepare(64 * G + 12345, mem(2, 125)).populate % (2ULL << 20) == 0, "2 MiB granule");

    // Real entry point: off by default and a zero request never populates, on any host; the report says why.
    {
        const auto r = strix::core::prepare_free_memory(0);
        require(!r.ran && r.populated == 0, "zero request is a no-op");
        require(strix::core::describe(r).find("skipped") != std::string::npos, "describe names the skip");
    }
#if defined(__linux__)
    // A worker can die before reporting while its sibling holds memory. The
    // sibling must close done before waiting, allowing EOF and release/reap.
    for (const int failed_worker : {-1, 0, 1, 2}) {
        int done[2], hold[2];
        require(::pipe(done) == 0 && ::pipe(hold) == 0, "failure fixture pipes");
        pid_t kids[2];
        ::alarm(5);  // bound a regression to the original parent/child deadlock
        for (int i = 0; i < 2; ++i) {
            kids[i] = ::fork();
            if (kids[i] == 0) {
                (void)::close(done[0]);
                (void)::close(hold[1]);
                if (i == failed_worker) ::_exit(1);
                strix::core::free_prepare_detail::report_and_hold(done[1], hold[0], failed_worker == 2 && i == 0 ? 0 : 1);
                ::_exit(0);
            }
            require(kids[i] > 0, "failure fixture fork");
        }
        (void)::close(done[1]);
        (void)::close(hold[0]);
        const bool complete = strix::core::free_prepare_detail::collect_reports(done[0], 2);
        require(complete == (failed_worker == -1), "missing worker report fails closed");
        (void)::close(hold[1]);
        (void)::close(done[0]);
        for (const pid_t kid : kids) {
            if (kid <= 0) continue;
            int status = 0;
            pid_t reaped;
            do { reaped = ::waitpid(kid, &status, 0); } while (reaped < 0 && errno == EINTR);
            require(reaped == kid && WIFEXITED(status), "all fixture workers reaped");
        }
        ::alarm(0);
    }
#endif
    if (failures != 0) { std::cerr << failures << " failure(s)\n"; return 1; }
    std::cout << "host_free_memory: all cases pass\n";
    return 0;
}
