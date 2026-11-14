#pragma once
#include "byte_budget.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace strix::core {
// The one host-side pool. Engines do not own threads for request preprocessing or background work; they
// submit jobs here.
//   critical   : short latency-bound jobs (tokenizer, row folds, masks). At most two threads, which keep
//                polling for a short window after a job so back-to-back jobs do not pay a wake-up.
//   background : byte-heavy jobs (snapshot packing, checkpoint CRC, cache hashing). Sleeping threads; each
//                job draws the bytes it reads from budget(), which is capped while any session decodes.
//   latency    : request preparation that is independent between requests (prompt tokenisation). A bounded
//                set of workers takes jobs in arrival order, so independent requests run on different cores.
//                While the byte budget's decode cap is in force the class narrows to
//                latency_decode_threads admissions; already-running jobs drain without preemption.
//                A caller that needs order inside one request finishes all shared-state accesses before
//                queuing the next job (a chain); job return/cleanup may overlap its successor.
enum class HostPriority { critical, background, latency };

struct HostLaneConfig {
    bool enabled = true;                 // false: every job runs inline in submit()
    std::vector<int> cores;              // cores the lane may pin; empty: no pinning
    unsigned critical_threads = 1;       // clamped to [0, 2]
    unsigned background_threads = 2;
    int latency_threads = -1;            // -1: from the core count (default_latency_threads()); 0: latency
                                         // jobs use the critical class (previous behaviour); clamped to [0, 8]
    unsigned latency_decode_threads = 2; // admission width during decode; running jobs drain; [1, latency_threads]
    double byte_cap_gbytes_ps = 10.0;    // background ceiling while any decode is active; <= 0: never capped
    std::chrono::microseconds critical_spin{200};
};

struct HostLaneStats {
    std::uint64_t critical_jobs = 0, background_jobs = 0, latency_jobs = 0, inline_jobs = 0;
    std::uint64_t critical_busy_ns = 0, background_busy_ns = 0, latency_busy_ns = 0;
    // Time between submit() and the start of the job, summed per class, and the longest single wait.
    std::uint64_t critical_wait_ns = 0, critical_wait_max_ns = 0, latency_wait_ns = 0, latency_wait_max_ns = 0;
    std::uint64_t latency_peak_running = 0; // most latency jobs that ran at the same time
    ByteBudget::Stats budget;
    double busy_ms() const { return static_cast<double>(critical_busy_ns + background_busy_ns + latency_busy_ns) / 1e6; }
};

// A quarter of the hardware threads, between two and four: enough for independent prompts to overlap,
// small enough to leave the engine's own host threads alone.
inline unsigned default_latency_threads() {
    return std::clamp<unsigned>(std::thread::hardware_concurrency() / 4U, 2U, 4U);
}

// "2-15", "0,2,4-6". Throws std::invalid_argument on anything else.
inline std::vector<int> parse_core_list(const std::string& text) {
    std::vector<int> cores;
    std::size_t at = 0;
    const auto number = [&]() {
        if (at >= text.size() || text[at] < '0' || text[at] > '9') throw std::invalid_argument("core list: expected a number");
        long value = 0;
        while (at < text.size() && text[at] >= '0' && text[at] <= '9') {
            value = value * 10 + (text[at++] - '0');
            if (value > 4095) throw std::invalid_argument("core list: core index too large");
        }
        return static_cast<int>(value);
    };
    while (at < text.size()) {
        const int first = number();
        int last = first;
        if (at < text.size() && text[at] == '-') {
            ++at;
            last = number();
            if (last < first) throw std::invalid_argument("core list: descending range");
        }
        for (int core = first; core <= last; ++core) cores.push_back(core);
        if (at == text.size()) break;
        if (text[at] != ',' || at + 1 == text.size()) throw std::invalid_argument("core list: expected ','");
        ++at;
    }
    return cores;
}

// Best effort; false when the platform has no affinity call or the kernel refuses the set.
inline bool pin_current_thread(const std::vector<int>& cores) {
#if defined(__linux__)
    if (cores.empty()) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (const int core : cores)
        if (core >= 0 && core < CPU_SETSIZE) CPU_SET(core, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)cores;
    return false;
#endif
}

class HostLane {
public:
    // budget: null gives the lane its own budget from config.byte_cap_gbytes_ps; the process-wide lane shares
    // the process-wide budget.
    explicit HostLane(HostLaneConfig config = {}, ByteBudget* budget = nullptr)
        : config_(std::move(config)), own_budget_(budget ? 0.0 : config_.byte_cap_gbytes_ps * 1e9),
          budget_(budget ? budget : &own_budget_) {
        if (config_.critical_threads > 2) config_.critical_threads = 2;
        if (config_.latency_threads < 0) config_.latency_threads = static_cast<int>(default_latency_threads());
        if (config_.latency_threads > 8) config_.latency_threads = 8;
        if (!config_.enabled) config_.critical_threads = config_.background_threads = 0, config_.latency_threads = 0;
        config_.latency_decode_threads = std::clamp<unsigned>(
            config_.latency_decode_threads, 1U, std::max<unsigned>(1U, static_cast<unsigned>(config_.latency_threads)));
        const std::size_t reserved = std::min<std::size_t>(config_.critical_threads, config_.cores.size());
        std::vector<int> rest(config_.cores.begin() + static_cast<std::ptrdiff_t>(reserved), config_.cores.end());
        if (rest.empty()) rest = config_.cores;
        for (unsigned i = 0; i < config_.critical_threads; ++i) {
            std::vector<int> own = i < config_.cores.size() ? std::vector<int>{config_.cores[i]} : config_.cores;
            threads_.emplace_back([this, own = std::move(own)] { pin_current_thread(own); critical_loop(); });
        }
        for (unsigned i = 0; i < config_.background_threads; ++i)
            threads_.emplace_back([this, rest] { pin_current_thread(rest); background_loop(); });
        for (int i = 0; i < config_.latency_threads; ++i)
            threads_.emplace_back([this, rest] { pin_current_thread(rest); latency_loop(); });
    }
    HostLane(const HostLane&) = delete;
    HostLane& operator=(const HostLane&) = delete;
    // Queued jobs are completed before the threads leave, so no future is abandoned.
    ~HostLane() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        critical_wake_.notify_all();
        background_wake_.notify_all();
        latency_wake_.notify_all();
        for (auto& thread : threads_) thread.join();
    }

    template<class F>
    auto submit(HostPriority priority, F&& job) -> std::future<std::invoke_result_t<std::decay_t<F>&>> {
        using Result = std::invoke_result_t<std::decay_t<F>&>;
        auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<F>(job));
        auto result = task->get_future();
        if (priority == HostPriority::latency && config_.latency_threads == 0) priority = HostPriority::critical;
        const bool critical = priority == HostPriority::critical;
        const bool latency = priority == HostPriority::latency;
        if ((critical ? config_.critical_threads : latency ? 1U : config_.background_threads) == 0) {
            inline_jobs_.fetch_add(1, std::memory_order_relaxed);
            (*task)();
            return result;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) throw std::logic_error("host lane: submit after shutdown");
            (critical ? critical_ : latency ? latency_ : background_).push_back({[task] { (*task)(); }, Clock::now()});
            if (critical) critical_pending_.fetch_add(1, std::memory_order_release);
        }
        // A latency worker that queues the next job of its chain takes a job itself as soon as it returns, so
        // it wakes nobody: the chain stays on a warm core instead of moving to a sleeping one.
        if (!latency) (critical ? critical_wake_ : background_wake_).notify_one();
        else if (latency_worker() != this) latency_wake_.notify_one();
        return result;
    }

    ByteBudget& budget() { return *budget_; }
    const HostLaneConfig& config() const { return config_; }
    HostLaneStats stats() const {
        HostLaneStats out;
        out.critical_jobs = critical_jobs_.load(std::memory_order_relaxed);
        out.background_jobs = background_jobs_.load(std::memory_order_relaxed);
        out.inline_jobs = inline_jobs_.load(std::memory_order_relaxed);
        out.critical_busy_ns = critical_busy_ns_.load(std::memory_order_relaxed);
        out.background_busy_ns = background_busy_ns_.load(std::memory_order_relaxed);
        out.latency_jobs = latency_jobs_.load(std::memory_order_relaxed);
        out.latency_busy_ns = latency_busy_ns_.load(std::memory_order_relaxed);
        out.critical_wait_ns = critical_wait_ns_.load(std::memory_order_relaxed);
        out.critical_wait_max_ns = critical_wait_max_ns_.load(std::memory_order_relaxed);
        out.latency_wait_ns = latency_wait_ns_.load(std::memory_order_relaxed);
        out.latency_wait_max_ns = latency_wait_max_ns_.load(std::memory_order_relaxed);
        out.latency_peak_running = latency_peak_.load(std::memory_order_relaxed);
        out.budget = budget_->stats();
        return out;
    }
    // Receipt fields: busy time, bytes granted to background work, and how often the cap made a job wait.
    std::string receipt_json() const {
        const auto s = stats();
        return "{\"host_lane_busy_ms\":" + std::to_string(s.busy_ms()) +
               ",\"host_lane_jobs\":" + std::to_string(s.critical_jobs + s.background_jobs + s.latency_jobs + s.inline_jobs) +
               ",\"host_lane_latency_jobs\":" + std::to_string(s.latency_jobs) +
               ",\"host_lane_latency_wait_ms\":" + std::to_string(static_cast<double>(s.latency_wait_ns) / 1e6) +
               ",\"host_lane_latency_wait_max_ms\":" + std::to_string(static_cast<double>(s.latency_wait_max_ns) / 1e6) +
               ",\"host_lane_latency_peak\":" + std::to_string(s.latency_peak_running) +
               ",\"host_lane_bytes_granted\":" + std::to_string(s.budget.bytes_granted) +
               ",\"host_lane_cap_hits\":" + std::to_string(s.budget.cap_hits) +
               ",\"host_lane_throttled_ms\":" + std::to_string(static_cast<double>(s.budget.throttled_ns) / 1e6) + "}";
    }

private:
    using Clock = std::chrono::steady_clock;
    struct Job {
        std::function<void()> work;
        Clock::time_point queued{};
        explicit operator bool() const { return static_cast<bool>(work); }
    };

    // The lane whose latency worker the calling thread is; null on every other thread.
    static const HostLane*& latency_worker() {
        thread_local const HostLane* lane = nullptr;
        return lane;
    }
    static std::uint64_t ns(Clock::duration d) {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
    }
    static void run(Job& job, std::atomic<std::uint64_t>& jobs, std::atomic<std::uint64_t>& busy_ns,
                    std::atomic<std::uint64_t>* wait_ns = nullptr, std::atomic<std::uint64_t>* wait_max_ns = nullptr) {
        jobs.fetch_add(1, std::memory_order_relaxed); // before the future becomes ready
        const auto begin = Clock::now();
        if (wait_ns) {
            const std::uint64_t waited = begin > job.queued ? ns(begin - job.queued) : 0;
            wait_ns->fetch_add(waited, std::memory_order_relaxed);
            if (wait_max_ns)
                for (std::uint64_t seen = wait_max_ns->load(std::memory_order_relaxed);
                     waited > seen && !wait_max_ns->compare_exchange_weak(seen, waited, std::memory_order_relaxed);) {}
        }
        job.work(); // a packaged task: exceptions travel through the future
        busy_ns.fetch_add(static_cast<std::uint64_t>(
                              std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count()),
                          std::memory_order_relaxed);
    }
    void critical_loop() {
        auto spin_until = Clock::now();
        for (;;) {
            Job job;
            if (critical_pending_.load(std::memory_order_acquire) != 0) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!critical_.empty()) {
                    job = std::move(critical_.front());
                    critical_.pop_front();
                    critical_pending_.fetch_sub(1, std::memory_order_release);
                }
            }
            if (job) {
                run(job, critical_jobs_, critical_busy_ns_, &critical_wait_ns_, &critical_wait_max_ns_);
                spin_until = Clock::now() + config_.critical_spin;
                continue;
            }
            if (Clock::now() < spin_until) {
                std::this_thread::yield();
                continue;
            }
            std::unique_lock<std::mutex> lock(mutex_);
            critical_wake_.wait(lock, [&] { return stop_ || !critical_.empty(); });
            if (critical_.empty()) return; // stop requested and nothing queued
        }
    }
    void background_loop() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                background_wake_.wait(lock, [&] { return stop_ || !background_.empty(); });
                if (background_.empty()) return;
                job = std::move(background_.front());
                background_.pop_front();
            }
            run(job, background_jobs_, background_busy_ns_);
        }
    }
    // Admission width: running jobs are not preempted when the decode cap starts.
    unsigned latency_limit() const {
        return budget_->capped() ? config_.latency_decode_threads : static_cast<unsigned>(config_.latency_threads);
    }
    void latency_loop() {
        latency_worker() = this;
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            // At shutdown the limit no longer applies: everything queued is completed.
            while (!stop_ && (latency_.empty() || latency_running_ >= latency_limit())) {
                if (latency_.empty()) latency_wake_.wait(lock);
                else {
                    if (latency_held_ < 8U) ++latency_held_;
                    // Decode end and linger expiry do not signal this CV. Recheck only while queued
                    // work is held by the cap; an empty queue still sleeps without polling.
                    latency_wake_.wait_for(lock, std::chrono::milliseconds(10));
                }
            }
            if (latency_.empty()) return; // stop requested and nothing queued
            Job job = std::move(latency_.front());
            latency_.pop_front();
            const unsigned running = ++latency_running_;
            if (running > latency_peak_.load(std::memory_order_relaxed)) latency_peak_.store(running, std::memory_order_relaxed);
            if (latency_held_ != 0 && !latency_.empty() && running < latency_limit()) {
                --latency_held_;
                latency_wake_.notify_one();
            }
            lock.unlock();
            run(job, latency_jobs_, latency_busy_ns_, &latency_wait_ns_, &latency_wait_max_ns_);
            job = Job{}; // release the task before the lock
            lock.lock();
            --latency_running_;
        }
    }

    HostLaneConfig config_;
    ByteBudget own_budget_;
    ByteBudget* budget_;
    std::mutex mutex_;
    std::condition_variable critical_wake_, background_wake_, latency_wake_;
    std::deque<Job> critical_, background_, latency_;
    std::atomic<std::uint64_t> critical_pending_{0};
    unsigned latency_running_ = 0; // under mutex_
    unsigned latency_held_ = 0;    // under mutex_
    bool stop_ = false;
    std::atomic<std::uint64_t> critical_jobs_{0}, background_jobs_{0}, latency_jobs_{0}, inline_jobs_{0};
    std::atomic<std::uint64_t> critical_busy_ns_{0}, background_busy_ns_{0}, latency_busy_ns_{0};
    std::atomic<std::uint64_t> critical_wait_ns_{0}, critical_wait_max_ns_{0}, latency_wait_ns_{0}, latency_wait_max_ns_{0}, latency_peak_{0};
    std::vector<std::thread> threads_;
};

// Configuration of the process-wide lane from the environment (the off switch and the declared cores):
//   STRIX_HOST_LANE=0                  run every job inline (previous behaviour)
//   STRIX_HOST_LANE_CORES=2-15         cores the lane may pin
//   STRIX_HOST_LANE_BACKGROUND=N       background threads (default 2)
//   STRIX_HOST_LANE_LATENCY=N          latency workers (default: a quarter of the cores, 2 to 4); 0 = latency
//                                      jobs share the critical thread (previous behaviour)
//   STRIX_HOST_LANE_LATENCY_DECODE=N   admission width during decode; running jobs drain (default 2)
//   STRIX_HOST_BYTE_CAP_GBPS=10        background ceiling during decode; 0 = never capped
inline HostLaneConfig host_lane_config_from_environment() {
    HostLaneConfig config;
    if (const char* value = host_environment("STRIX_HOST_LANE")) config.enabled = std::string(value) != "0";
    if (const char* value = host_environment("STRIX_HOST_LANE_CORES")) config.cores = parse_core_list(value);
    if (const char* value = host_environment("STRIX_HOST_LANE_BACKGROUND")) {
        const long threads = std::strtol(value, nullptr, 10);
        if (threads < 0 || threads > 64) throw std::invalid_argument("STRIX_HOST_LANE_BACKGROUND out of range");
        config.background_threads = static_cast<unsigned>(threads);
    }
    if (const char* value = host_environment("STRIX_HOST_LANE_LATENCY")) {
        const long threads = std::strtol(value, nullptr, 10);
        if (threads < 0 || threads > 8) throw std::invalid_argument("STRIX_HOST_LANE_LATENCY out of range");
        config.latency_threads = static_cast<int>(threads);
    }
    if (const char* value = host_environment("STRIX_HOST_LANE_LATENCY_DECODE")) {
        const long threads = std::strtol(value, nullptr, 10);
        if (threads < 1 || threads > 8) throw std::invalid_argument("STRIX_HOST_LANE_LATENCY_DECODE out of range");
        config.latency_decode_threads = static_cast<unsigned>(threads);
    }
    config.byte_cap_gbytes_ps = byte_cap_gbytes_ps_from_environment(config.byte_cap_gbytes_ps);
    return config;
}

// The process-wide lane. Created on first use; every engine and the serving layer share it.
inline HostLane& shared_host_lane() {
    static HostLane lane(host_lane_config_from_environment(), &shared_byte_budget());
    return lane;
}
} // namespace strix::core
