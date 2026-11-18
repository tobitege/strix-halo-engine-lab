#pragma once
// Free host memory before a large device load (shared by every engine's loader).
//
// On a unified-memory node the device allocations of a load come out of the same RAM as the page cache. When MemFree does
// not cover the load, the kernel reclaims and compacts WHILE the load and the first forward run: compaction migrates
// pages, a migrated page of a host buffer the GPU runtime has locked invalidates that mapping, and the driver then
// suspends every compute queue of the process until the mapping is restored (seconds, counted in
// /sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms). In a pair that node becomes the slow rank and the other one idles.
//
// prepare_free_memory() moves that work in front of the load, where no queue is live: it populates anonymous memory until
// the page cache has given back what the load needs and releases it at once, so the load finds the memory free.
// Opt-in. Nothing is done when MemFree already covers the request (the common case), and the step never takes more than
// MemAvailable minus a reserve: it cannot swap or OOM the node. It changes no system setting and needs no privilege.
#include "strix/env_flags.hpp"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fstream>
#endif

namespace strix::core {

inline constexpr std::uint64_t kFreePrepGiB = 1ULL << 30;

struct HostMemory {
    std::uint64_t free = 0;       // MemFree, bytes
    std::uint64_t available = 0;  // MemAvailable, bytes
    bool ok = false;
};

inline HostMemory read_host_memory() {
    HostMemory m;
#if defined(__linux__)
    std::ifstream in("/proc/meminfo");
    std::string key, unit;
    std::uint64_t kb = 0;
    bool f = false, a = false;
    while (in >> key >> kb) {
        std::getline(in, unit);
        if (key == "MemFree:") { m.free = kb * 1024U; f = true; }
        else if (key == "MemAvailable:") { m.available = kb * 1024U; a = true; }
        if (f && a) break;
    }
    m.ok = f && a;
#endif
    return m;
}

struct FreePrepPlan {
    std::uint64_t populate = 0;  // bytes of anonymous memory to populate and release (0 = nothing to do)
    std::uint64_t target = 0;    // MemFree expected afterwards
    const char* reason = "";
};

// Pure policy (host test): how much to populate so that `need` bytes are free afterwards.
//   - MemFree already covers `need`: nothing.
//   - Otherwise take the missing bytes from the page cache, never beyond MemAvailable - reserve; a shortfall below
//     `min_take` is left to the kernel (not worth a pass).
// Populating consumes the free memory first, so the populate size is the target itself.
inline FreePrepPlan plan_free_prepare(std::uint64_t need, const HostMemory& m, std::uint64_t reserve = 4 * kFreePrepGiB,
                                      std::uint64_t min_take = kFreePrepGiB) {
    FreePrepPlan p;
    if (!m.ok) { p.reason = "meminfo unreadable"; return p; }
    if (m.free >= need) { p.reason = "free memory covers the load"; return p; }
    const std::uint64_t ceiling = m.available > reserve ? m.available - reserve : 0;
    const std::uint64_t target = std::min(need, ceiling);
    if (target <= m.free || target - m.free < min_take) { p.reason = "nothing reclaimable beyond the reserve"; return p; }
    p.target = target & ~((2ULL << 20) - 1);
    p.populate = p.target;
    p.reason = "page cache holds memory the load needs";
    return p;
}

struct FreePrepReport {
    bool ran = false;
    std::uint64_t need = 0, free_before = 0, free_after = 0, available = 0, populated = 0;
    double seconds = 0.0;
    unsigned threads = 0;
    const char* reason = "";
};

inline std::string describe(const FreePrepReport& r) {
    char b[320];
    const auto g = [](std::uint64_t v) { return static_cast<double>(v) / static_cast<double>(kFreePrepGiB); };
    if (r.ran)
        std::snprintf(b, sizeof b, "host free-memory prepare: need %.1f GiB, MemFree %.1f -> %.1f GiB (MemAvailable %.1f), populated %.1f GiB in %.2f s on %u workers",
                      g(r.need), g(r.free_before), g(r.free_after), g(r.available), g(r.populated), r.seconds, r.threads);
    else
        std::snprintf(b, sizeof b, "host free-memory prepare: skipped (%s; need %.1f GiB, MemFree %.1f, MemAvailable %.1f GiB)", r.reason, g(r.need),
                      g(r.free_before), g(r.available));
    return b;
}

#if defined(__linux__)
namespace free_prepare_detail {
// Close the reporting end before holding memory: a dead sibling must produce EOF
// once all surviving workers have reported, rather than leaving the parent blocked.
inline void report_and_hold(int done, int hold, char result) {
    ssize_t n;
    do { n = ::write(done, &result, 1); } while (n < 0 && errno == EINTR);
    (void)::close(done);
    do { n = ::read(hold, &result, 1); } while (n < 0 && errno == EINTR);
}

inline bool collect_reports(int done, std::size_t workers) {
    bool success = true;
    for (std::size_t i = 0; i < workers; ++i) {
        char result = 0;
        ssize_t n;
        do { n = ::read(done, &result, 1); } while (n < 0 && errno == EINTR);
        if (n != 1) return false;
        if (result != 1) success = false;
    }
    return success;
}
}  // namespace free_prepare_detail
#endif

// Runs the plan. Call it once, before the first device allocation of a load, from the loading thread.
// Opt-in: STRIX_HOST_FREE_PREPARE=1 (the pass costs about 1 s per 5 GiB of page cache it has to take back, so it pays
// for a cold timed run, not for a server start); STRIX_HOST_FREE_PREPARE_THREADS sets the populate workers (default 16).
inline FreePrepReport prepare_free_memory(std::uint64_t need_bytes) {
    FreePrepReport r;
    r.need = need_bytes;
    const HostMemory m = read_host_memory();
    r.free_before = r.free_after = m.free;
    r.available = m.available;
    if (!envcfg::flag("STRIX_HOST_FREE_PREPARE", false)) { r.reason = "off: set STRIX_HOST_FREE_PREPARE=1"; return r; }
    const FreePrepPlan plan = plan_free_prepare(need_bytes, m);
    r.reason = plan.reason;
    if (plan.populate == 0) return r;
#if defined(__linux__)
    // Child processes populate and hold the memory (raw system calls only after fork), so the calling process keeps its
    // own resident size: an engine's RSS watchdog must not see this pass.
    const unsigned hw = std::max(1U, std::thread::hardware_concurrency());
    const unsigned workers = static_cast<unsigned>(std::clamp(envcfg::integer("STRIX_HOST_FREE_PREPARE_THREADS", 16), 1, static_cast<int>(std::min(hw, 64U))));
    const auto t0 = std::chrono::steady_clock::now();
    int done[2] = {-1, -1}, hold[2] = {-1, -1};
    if (::pipe(done) != 0) { r.reason = "pipe failed"; return r; }
    if (::pipe(hold) != 0) { (void)::close(done[0]); (void)::close(done[1]); r.reason = "pipe failed"; return r; }
    const std::uint64_t per = (plan.populate / workers) & ~((2ULL << 20) - 1);
    std::vector<pid_t> kids;
    kids.reserve(workers);
    for (unsigned i = 0; i < workers && per != 0; ++i) {
        const pid_t pid = ::fork();
        if (pid < 0) break;
        if (pid == 0) {
            (void)::close(done[0]);
            (void)::close(hold[1]);
            void* p = ::mmap(nullptr, per, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (p != MAP_FAILED) {
#if defined(MADV_POPULATE_WRITE)
                if (::madvise(p, per, MADV_POPULATE_WRITE) != 0)
#endif
                    for (std::uint64_t o = 0; o < per; o += 4096) static_cast<volatile unsigned char*>(p)[o] = 1;
            }
            char c = p != MAP_FAILED ? 1 : 0;
            free_prepare_detail::report_and_hold(done[1], hold[0], c);
            ::_exit(0);
        }
        kids.push_back(pid);
    }
    (void)::close(done[1]);
    (void)::close(hold[0]);
    const bool complete = free_prepare_detail::collect_reports(done[0], kids.size());
    const std::uint64_t populated = complete && kids.size() == workers ? per * kids.size() : 0;
    (void)::close(hold[1]);
    (void)::close(done[0]);
    for (const pid_t pid : kids) {
        int st = 0;
        while (::waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
    }
    r.ran = populated != 0;
    if (!r.ran) r.reason = "populate workers failed";
    r.populated = populated;
    r.threads = static_cast<unsigned>(kids.size());
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    r.free_after = read_host_memory().free;
#else
    r.reason = "not supported on this host";
#endif
    return r;
}

}  // namespace strix::core
