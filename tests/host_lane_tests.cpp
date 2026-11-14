#include "core/host_lane.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using strix::core::ByteBudget;
using strix::core::HostLane;
using strix::core::HostLaneConfig;
using strix::core::HostPriority;
using Clock = ByteBudget::Clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> bool throws(F f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

void budget_is_unlimited_without_decode() {
    ByteBudget budget(1000.0, 0.1); // 1000 B/s, burst 100 B
    const auto t0 = Clock::time_point{} + std::chrono::seconds(10);
    require(budget.reserve(1u << 30, t0) == Clock::duration::zero(), "idle budget made a job wait");
    require(!budget.capped(), "idle budget reports capped");
    const auto stats = budget.stats();
    require(stats.bytes_granted == (1u << 30) && stats.bytes_capped == 0 && stats.cap_hits == 0, "idle stats");
    ByteBudget never(0.0);
    never.decode_begin(t0);
    require(never.reserve(1u << 30, t0) == Clock::duration::zero() && !never.capped(), "cap 0 must never cap");
}

void budget_caps_during_decode() {
    ByteBudget budget(1000.0, 0.1, std::chrono::seconds(1));
    const auto t0 = Clock::time_point{} + std::chrono::seconds(10);
    budget.decode_begin(t0);
    require(budget.capped(), "decode did not arm the cap");
    require(budget.reserve(100, t0) == Clock::duration::zero(), "burst was not available");
    // Bucket empty: 500 bytes at 1000 B/s owe 0.5 s.
    require(duration_cast<milliseconds>(budget.reserve(500, t0)).count() == 500, "debt wait is not bytes / cap");
    // 0.25 s later half the debt is repaid; another 250 bytes owe 0.5 s again.
    require(duration_cast<milliseconds>(budget.reserve(250, t0 + milliseconds(250))).count() == 500, "refill");
    // After a long idle gap the bucket holds the burst only, never more.
    require(budget.reserve(100, t0 + std::chrono::seconds(100)) == Clock::duration::zero(), "burst after idle");
    require(budget.reserve(1, t0 + std::chrono::seconds(100)) > Clock::duration::zero(), "burst exceeded");
    // A clock reading older than the last one must not mint tokens.
    require(budget.reserve(1, t0) > Clock::duration::zero(), "stale clock minted tokens");
    const auto stats = budget.stats();
    require(stats.bytes_granted == 952 && stats.bytes_capped == 952 && stats.cap_hits == 4, "capped stats");
    require(stats.throttled_ns >= 1000000000ull, "throttle time not recorded");
    // The cap lingers after a decode step, and the next step of the same generation does not refill the bucket.
    const auto t1 = t0 + std::chrono::seconds(100);
    (void)budget.reserve(2000, t1); // two seconds of debt
    budget.decode_end(t1);
    require(budget.capped(t1 + milliseconds(500)), "cap dropped between two decode steps");
    budget.decode_begin(t1 + milliseconds(500));
    require(budget.reserve(1, t1 + milliseconds(500)) > Clock::duration::zero(), "a new step refilled the bucket");
    budget.decode_end(t1 + milliseconds(600));
    require(budget.capped(t1 + milliseconds(1500)) && !budget.capped(t1 + milliseconds(1700)), "linger length");
    require(budget.reserve(1u << 20, t1 + std::chrono::seconds(2)) == Clock::duration::zero(), "idle after decode");
    // A decode after a real idle gap starts with a full burst again.
    budget.decode_begin(t1 + std::chrono::seconds(3));
    require(budget.reserve(100, t1 + std::chrono::seconds(3)) == Clock::duration::zero(), "burst after an idle gap");
    budget.decode_end(t1 + std::chrono::seconds(3));
}

void budget_long_run_rate_stays_under_the_cap() {
    const double cap = 1e6, burst_seconds = 0.02;
    ByteBudget budget(cap, burst_seconds);
    auto now = Clock::time_point{} + std::chrono::seconds(1);
    const auto begin = now;
    budget.decode_begin(now);
    std::uint64_t bytes = 0;
    for (int i = 0; i < 2000; ++i) { // a caller that honours every wait
        const std::uint64_t slice = 1000 + static_cast<std::uint64_t>(i % 7) * 3000;
        now += budget.reserve(slice, now);
        bytes += slice;
        now += std::chrono::microseconds(i % 5);
    }
    const double elapsed = std::chrono::duration<double>(now - begin).count();
    require(static_cast<double>(bytes) <= cap * elapsed + cap * burst_seconds + 16.0, "rate exceeded cap + burst");
    require(static_cast<double>(bytes) >= 0.98 * cap * elapsed, "governor wasted more than 2 % of the cap");
    // Nested decodes: the cap holds until the last one ends.
    budget.decode_begin(now);
    budget.decode_end(now);
    require(budget.capped(now + std::chrono::seconds(60)), "cap dropped while a decode was still active");
    budget.decode_end(now);
    require(budget.capped(now + milliseconds(100)) && !budget.capped(now + std::chrono::seconds(1)), "cap after the last decode");
}

void budget_wait_ends_with_the_decode() {
    ByteBudget budget(1000.0, 0.0);
    budget.decode_begin();
    std::thread releaser([&] {
        std::this_thread::sleep_for(milliseconds(30));
        budget.decode_end();
    });
    const auto begin = Clock::now();
    budget.acquire(3600 * 1000); // an hour of debt at the cap
    releaser.join();
    require(Clock::now() - begin < std::chrono::seconds(20), "acquire did not wake when decode ended");
    {
        ByteBudget::DecodeScope decode(budget);
        require(budget.capped(), "scope did not arm the cap");
    }
    require(!budget.capped(Clock::now() + std::chrono::seconds(1)), "scope did not release the cap");
    std::uint64_t seen = 0, calls = 0;
    strix::core::governed_slices(budget, 10, 4, [&](std::uint64_t offset, std::uint64_t length) {
        require(offset == seen, "slices out of order");
        seen += length;
        ++calls;
    });
    require(seen == 10 && calls == 3, "slices do not cover the range");
}

void core_lists() {
    using strix::core::parse_core_list;
    require(parse_core_list("2-5") == std::vector<int>({2, 3, 4, 5}), "range");
    require(parse_core_list("0,2,4-6") == std::vector<int>({0, 2, 4, 5, 6}), "mixed list");
    require(parse_core_list("").empty(), "empty list");
    for (const char* bad : {"a", "1-", "3-2", "1,", "1;2", "-1", "99999"})
        require(throws([&] { (void)parse_core_list(bad); }), "malformed core list was accepted");
}

void lane_runs_jobs_and_reports() {
    HostLaneConfig config;
    config.critical_threads = 7; // clamped to two
    config.background_threads = 2;
    config.byte_cap_gbytes_ps = 1.0;
    std::atomic<int> sum{0};
    {
        HostLane lane(config);
        require(lane.config().critical_threads == 2, "critical threads were not clamped");
        const auto caller = std::this_thread::get_id();
        auto id = lane.submit(HostPriority::critical, [] { return std::this_thread::get_id(); });
        require(id.get() != caller, "critical job ran on the caller");
        auto value = lane.submit(HostPriority::background, [] { return std::string("done"); });
        require(value.get() == "done", "result lost");
        auto failing = lane.submit(HostPriority::background, [](int fail = 1) -> int {
            if (fail) throw std::runtime_error("job failed");
            return 0;
        });
        require(throws([&] { (void)failing.get(); }), "job exception was swallowed");
        // Both background threads blocked: a critical job still completes.
        std::promise<void> gate;
        const std::shared_future<void> opened = gate.get_future().share();
        auto blocked_a = lane.submit(HostPriority::background, [opened] { opened.wait(); });
        auto blocked_b = lane.submit(HostPriority::background, [opened] { opened.wait(); });
        auto urgent = lane.submit(HostPriority::critical, [] { return 7; });
        require(urgent.wait_for(std::chrono::seconds(20)) == std::future_status::ready && urgent.get() == 7,
                "critical job waited behind background work");
        gate.set_value();
        blocked_a.get();
        blocked_b.get();
        // Background jobs draw from the lane's budget; decode arms the cap.
        {
            ByteBudget::DecodeScope decode(lane.budget());
            lane.submit(HostPriority::background, [&lane] { (void)lane.budget().reserve(4096, Clock::now()); }).get();
        }
        const auto stats = lane.stats();
        require(stats.critical_jobs == 2 && stats.background_jobs == 5 && stats.inline_jobs == 0, "job counts");
        require(stats.budget.bytes_granted == 4096 && stats.budget.bytes_capped == 4096, "budget stats");
        const auto receipt = lane.receipt_json();
        require(receipt.find("\"host_lane_bytes_granted\":4096") != std::string::npos &&
                    receipt.find("\"host_lane_busy_ms\":") != std::string::npos &&
                    receipt.find("\"host_lane_cap_hits\":0") != std::string::npos,
                "receipt fields");
        for (int i = 1; i <= 200; ++i)
            (void)lane.submit(i % 2 ? HostPriority::critical : HostPriority::background, [&sum, i] { sum += i; });
    } // destruction completes everything that was queued
    require(sum.load() == 200 * 201 / 2, "queued jobs were dropped at shutdown");
}

// Latency class: independent jobs overlap up to the worker count, a chain keeps its order, the class narrows
// while a decode is active, and with no latency workers the jobs use the critical class.
void latency_class_runs_independent_jobs_in_parallel() {
    HostLaneConfig config;
    config.latency_threads = 3;
    config.latency_decode_threads = 1;
    config.byte_cap_gbytes_ps = 1.0;
    {
        HostLane lane(config);
        require(lane.config().latency_threads == 3 && lane.config().latency_decode_threads == 1, "latency config");
        // Three jobs that only finish once all three are running: they must be on three threads at once.
        std::atomic<int> running{0};
        std::vector<std::future<bool>> met;
        for (int i = 0; i < 3; ++i)
            met.push_back(lane.submit(HostPriority::latency, [&running] {
                ++running;
                const auto deadline = Clock::now() + std::chrono::seconds(20);
                while (running.load() < 3 && Clock::now() < deadline) std::this_thread::yield();
                return running.load() == 3;
            }));
        for (auto& f : met) require(f.get(), "independent latency jobs did not overlap");
        require(lane.stats().latency_peak_running == 3, "latency peak");

        // Chains: finish protected work before queuing the successor. That work stays ordered;
        // return/cleanup may overlap the successor, while different chains interleave.
        constexpr int chains = 8, links = 200;
        struct Chain {
            std::vector<int> seen;
            std::atomic<int> inside{0};
            std::atomic<bool> overlap{false};
            std::promise<void> done;
        };
        std::vector<Chain> state(chains);
        std::function<void(int, int)> link = [&](int chain, int index) {
            (void)lane.submit(HostPriority::latency, [&, chain, index] {
                Chain& c = state[static_cast<std::size_t>(chain)];
                if (c.inside.fetch_add(1) != 0) c.overlap = true;
                c.seen.push_back(index);
                c.inside.fetch_sub(1);
                if (index + 1 == links) c.done.set_value();
                else link(chain, index + 1);
            });
        };
        for (int chain = 0; chain < chains; ++chain) link(chain, 0);
        for (auto& c : state) {
            c.done.get_future().get();
            require(!c.overlap.load() && static_cast<int>(c.seen.size()) == links, "chain overlapped or lost a link");
            for (int i = 0; i < links; ++i) require(c.seen[static_cast<std::size_t>(i)] == i, "chain order");
        }

        // While a decode is active only latency_decode_threads jobs run at once; everything still completes.
        std::atomic<int> now{0}, peak{0};
        std::vector<std::future<void>> capped;
        {
            ByteBudget::DecodeScope decode(lane.budget());
            for (int i = 0; i < 12; ++i)
                capped.push_back(lane.submit(HostPriority::latency, [&now, &peak] {
                    const int mine = ++now;
                    for (int seen = peak.load(); mine > seen && !peak.compare_exchange_weak(seen, mine);) {}
                    std::this_thread::sleep_for(milliseconds(2));
                    --now;
                }));
            for (auto& f : capped) f.get();
        }
        require(peak.load() == 1, "latency class did not narrow during decode");
        const auto stats = lane.stats();
        require(stats.latency_jobs == 3U + chains * links + 12U && stats.critical_jobs == 0, "latency job counts");
        require(stats.latency_wait_max_ns <= stats.latency_wait_ns && stats.latency_busy_ns > 0, "latency wait stats");
        require(lane.receipt_json().find("\"host_lane_latency_jobs\":") != std::string::npos, "latency receipt field");
        // A critical job does not wait behind latency work that fills every latency worker.
        std::promise<void> gate;
        const std::shared_future<void> opened = gate.get_future().share();
        std::vector<std::future<void>> blocked;
        for (int i = 0; i < 6; ++i) blocked.push_back(lane.submit(HostPriority::latency, [opened] { opened.wait(); }));
        auto urgent = lane.submit(HostPriority::critical, [] { return 7; });
        require(urgent.wait_for(std::chrono::seconds(20)) == std::future_status::ready && urgent.get() == 7,
                "critical job waited behind latency work");
        gate.set_value();
        for (auto& f : blocked) f.get();
    }
    {
        // No latency workers: the previous arrangement, latency jobs on the critical thread.
        HostLaneConfig previous;
        previous.latency_threads = 0;
        HostLane lane(previous);
        const auto caller = std::this_thread::get_id();
        require(lane.submit(HostPriority::latency, [] { return std::this_thread::get_id(); }).get() != caller, "latency fallback thread");
        const auto stats = lane.stats();
        require(stats.critical_jobs == 1 && stats.latency_jobs == 0 && stats.inline_jobs == 0, "latency fallback counts");
    }
    {
        HostLaneConfig automatic, wide, off;
        wide.latency_threads = 99;
        wide.latency_decode_threads = 99;
        off.enabled = false;
        HostLane a(automatic), w(wide), o(off);
        require(a.config().latency_threads >= 2 && a.config().latency_threads <= 4, "automatic latency worker count");
        require(w.config().latency_threads == 8 && w.config().latency_decode_threads == 8, "latency clamps");
        const auto caller = std::this_thread::get_id();
        require(o.submit(HostPriority::latency, [] { return std::this_thread::get_id(); }).get() == caller, "inline latency");
    }
    // Jobs still queued at destruction complete, also past the worker count.
    std::atomic<int> sum{0};
    {
        HostLaneConfig small;
        small.latency_threads = 2;
        HostLane lane(small);
        for (int i = 1; i <= 300; ++i) (void)lane.submit(HostPriority::latency, [&sum, i] { sum += i; });
    }
    require(sum.load() == 300 * 301 / 2, "queued latency jobs were dropped at shutdown");
}

void lane_disabled_runs_inline() {
    HostLaneConfig config;
    config.enabled = false;
    HostLane lane(config);
    const auto caller = std::this_thread::get_id();
    require(lane.submit(HostPriority::critical, [] { return std::this_thread::get_id(); }).get() == caller, "inline critical");
    require(lane.submit(HostPriority::background, [] { return std::this_thread::get_id(); }).get() == caller, "inline background");
    auto failing = lane.submit(HostPriority::critical, [](int fail = 1) -> int {
        if (fail) throw std::runtime_error("job failed");
        return 0;
    });
    require(throws([&] { (void)failing.get(); }), "inline exception must travel through the future");
    require(lane.stats().inline_jobs == 3, "inline count");
}

void lane_pins_to_declared_cores() {
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
    std::vector<int> cores;
    for (int core = 0; core < CPU_SETSIZE && cores.size() < 3; ++core)
        if (CPU_ISSET(core, &allowed)) cores.push_back(core);
    if (cores.size() < 2) return;
    HostLaneConfig config;
    config.cores = cores;
    config.critical_threads = 1;
    config.background_threads = 1;
    HostLane lane(config);
    const auto mask = [] {
        cpu_set_t set;
        CPU_ZERO(&set);
        std::vector<int> out;
        if (pthread_getaffinity_np(pthread_self(), sizeof(set), &set) == 0)
            for (int core = 0; core < CPU_SETSIZE; ++core)
                if (CPU_ISSET(core, &set)) out.push_back(core);
        return out;
    };
    require(lane.submit(HostPriority::critical, mask).get() == std::vector<int>({cores[0]}), "critical thread pin");
    require(lane.submit(HostPriority::background, mask).get() == std::vector<int>(cores.begin() + 1, cores.end()),
            "background threads must stay off the critical core");
#endif
}
volatile std::sig_atomic_t load_stop = 0;
extern "C" void stop_load(int) { load_stop = 1; }

// --load SECONDS THREADS CAP_GBPS [BUFFER_MIB]: background lane jobs that read memory under the byte budget with
// decode marked active (CAP_GBPS 0: uncapped). Stops after SECONDS or on SIGTERM/SIGINT and prints the achieved rate;
// a load generator, not a test.
int load(double seconds, unsigned threads, double cap_gbps, std::size_t buffer_mib) {
    std::signal(SIGTERM, stop_load);
    std::signal(SIGINT, stop_load);
    HostLaneConfig config;
    config.critical_threads = 0;
    config.background_threads = threads;
    config.byte_cap_gbytes_ps = cap_gbps;
    HostLane lane(config);
    lane.budget().decode_begin();
    std::atomic<std::uint64_t> bytes{0}, checksum{0};
    std::vector<std::future<void>> jobs;
    const auto begin = Clock::now();
    const auto until = begin + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    for (unsigned t = 0; t < threads; ++t)
        jobs.push_back(lane.submit(HostPriority::background, [&, t] {
            std::vector<std::uint64_t> buffer(buffer_mib * (std::size_t{1} << 17U), 0x9E3779B97F4A7C15ULL * (t + 1U));
            const std::uint64_t total = buffer.size() * sizeof(std::uint64_t);
            while (Clock::now() < until && !load_stop)
                strix::core::governed_slices(lane.budget(), total, std::uint64_t{4} << 20U, [&](std::uint64_t offset, std::uint64_t length) {
                    if (Clock::now() >= until || load_stop) return;
                    std::uint64_t x = 0;
                    const std::uint64_t* p = buffer.data() + offset / sizeof(std::uint64_t);
                    for (std::uint64_t i = 0; i < length / sizeof(std::uint64_t); ++i) x ^= p[i];
                    checksum.fetch_xor(x, std::memory_order_relaxed);
                    bytes.fetch_add(length, std::memory_order_relaxed);
                });
        }));
    for (auto& job : jobs) job.get();
    lane.budget().decode_end();
    const double elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
    std::cout << "load threads " << threads << " cap_gbps " << cap_gbps << " seconds " << elapsed << " read_gb "
              << static_cast<double>(bytes.load()) / 1e9 << " rate_gbps " << static_cast<double>(bytes.load()) / 1e9 / elapsed
              << " checksum " << checksum.load() << ' ' << lane.receipt_json() << std::endl;
    return 0;
}
} // namespace

// No submission or job completion may be needed to wake the workers after linger expires.
void latency_resumes_after_decode_linger() {
    HostLaneConfig config;
    config.latency_threads = 4;
    config.latency_decode_threads = 1;
    std::promise<void> started, release;
    auto released = release.get_future().share();
    HostLane lane(config);
    lane.budget().decode_begin();
    auto long_job = lane.submit(HostPriority::latency, [&] {
        started.set_value();
        (void)released.wait_for(std::chrono::seconds(5));
    });
    started.get_future().wait();
    std::vector<std::future<void>> short_jobs;
    for (int i = 0; i < 3; ++i) short_jobs.push_back(lane.submit(HostPriority::latency, [] {}));
    const bool held = short_jobs.front().wait_for(milliseconds(100)) == std::future_status::timeout;
    lane.budget().decode_end();
    const auto deadline = Clock::now() + std::chrono::seconds(2);
    bool resumed = true;
    for (auto& job : short_jobs) resumed &= job.wait_until(deadline) == std::future_status::ready;
    const bool still_blocked = long_job.wait_for(milliseconds(0)) == std::future_status::timeout;
    release.set_value(); // also let a failing implementation shut down without hanging
    long_job.get();
    for (auto& job : short_jobs) job.get();
    require(held, "latency jobs escaped the active decode cap");
    require(resumed && still_blocked, "queued latency jobs did not resume after decode linger");
}

int main(int argc, char** argv) {
    try {
        if (argc >= 5 && std::string(argv[1]) == "--load")
            return load(std::stod(argv[2]), static_cast<unsigned>(std::stoul(argv[3])), std::stod(argv[4]),
                        argc >= 6 ? static_cast<std::size_t>(std::stoull(argv[5])) : 256U);
        budget_is_unlimited_without_decode();
        budget_caps_during_decode();
        budget_long_run_rate_stays_under_the_cap();
        budget_wait_ends_with_the_decode();
        core_lists();
        lane_runs_jobs_and_reports();
        latency_class_runs_independent_jobs_in_parallel();
        latency_resumes_after_decode_linger();
        lane_disabled_runs_inline();
        lane_pins_to_declared_cores();
    } catch (const std::exception& error) {
        std::cerr << "host lane tests failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "host lane tests passed\n";
    return 0;
}
