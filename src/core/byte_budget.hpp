#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace strix::core {
// Governor for background host work that reads memory (snapshot packing, checkpoint I/O and CRC, cache hashing).
// Memory bandwidth is shared with the decode weight stream, so while any session decodes the bytes read by
// background jobs are limited to `cap` bytes per second; with no decode active the budget is unlimited.
//
// Token bucket with debt: a reservation is always granted, and the returned duration is how long the caller
// must wait before touching the bytes. Requests larger than the burst are therefore legal (they simply wait
// longer) and the long-run rate under the cap never exceeds cap + burst / elapsed.
//
// Decode is marked per step, so the cap stays in force for `linger` after the last decode ends: the gaps
// between the steps of one generation neither lift the cap nor refill the bucket.
class ByteBudget {
public:
    using Clock = std::chrono::steady_clock;
    struct Stats {
        std::uint64_t bytes_granted = 0; // every reservation
        std::uint64_t bytes_capped = 0;  // reservations made while the cap was in force
        std::uint64_t cap_hits = 0;      // reservations that had to wait
        std::uint64_t throttled_ns = 0;  // total wait handed out
    };

    // cap_bytes_per_second <= 0 means never capped.
    explicit ByteBudget(double cap_bytes_per_second = 0.0, double burst_seconds = 0.02,
                        Clock::duration linger = std::chrono::milliseconds(250))
        : cap_(cap_bytes_per_second), burst_(std::max(0.0, cap_bytes_per_second) * std::max(0.0, burst_seconds)),
          linger_(linger), tokens_(burst_) {}

    void decode_begin(Clock::time_point now = Clock::now()) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_++ == 0 && !capped_locked(now)) { // a new decode period, not the next step of the current one
            tokens_ = burst_;
            last_ = now;
        }
    }
    void decode_end(Clock::time_point now = Clock::now()) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_ == 0 || --active_ != 0) return;
        ended_ = now;
        decoded_ = true;
    }
    // One decode call; background work is capped for its lifetime and for the linger after it.
    class DecodeScope {
    public:
        explicit DecodeScope(ByteBudget& budget) : budget_(&budget) { budget_->decode_begin(); }
        DecodeScope(const DecodeScope&) = delete;
        DecodeScope& operator=(const DecodeScope&) = delete;
        ~DecodeScope() { budget_->decode_end(); }

    private:
        ByteBudget* budget_;
    };

    bool capped(Clock::time_point now = Clock::now()) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return capped_locked(now);
    }
    double cap_bytes_per_second() const { return cap_; }

    // Deterministic core: charge `bytes` at time `now`; returns the wait owed before the bytes may be read.
    Clock::duration reserve(std::uint64_t bytes, Clock::time_point now) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.bytes_granted += bytes;
        if (!capped_locked(now)) return Clock::duration::zero();
        stats_.bytes_capped += bytes;
        if (now > last_) {
            tokens_ = std::min(burst_, tokens_ + std::chrono::duration<double>(now - last_).count() * cap_);
            last_ = now;
        }
        tokens_ -= static_cast<double>(bytes);
        if (tokens_ >= 0.0) return Clock::duration::zero();
        const auto wait = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(-tokens_ / cap_));
        ++stats_.cap_hits;
        stats_.throttled_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(wait).count());
        return wait;
    }
    // Blocking form: returns when the bytes may be read. The wait ends early once the cap is lifted.
    void acquire(std::uint64_t bytes) {
        const auto wait = reserve(bytes, Clock::now());
        if (wait <= Clock::duration::zero()) return;
        const auto deadline = Clock::now() + wait;
        const auto poll = std::max<Clock::duration>(linger_ / 4, std::chrono::milliseconds(1));
        for (auto now = Clock::now(); now < deadline && capped(now); now = Clock::now())
            std::this_thread::sleep_for(std::min<Clock::duration>(deadline - now, poll));
    }
    Stats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

private:
    bool capped_locked(Clock::time_point now) const {
        return cap_ > 0.0 && (active_ != 0 || (decoded_ && now - ended_ < linger_));
    }

    const double cap_;
    const double burst_;
    const Clock::duration linger_;
    mutable std::mutex mutex_;
    double tokens_;
    Clock::time_point last_{}, ended_{};
    unsigned active_ = 0;
    bool decoded_ = false;
    Stats stats_;
};

// Runs `visit(offset, length)` over [0, total) in slices, drawing each slice from the budget first, so a
// long background pass reacts to a decode that starts in the middle of it.
template<class Visit>
void governed_slices(ByteBudget& budget, std::uint64_t total, std::uint64_t slice, Visit visit) {
    if (slice == 0) slice = total;
    for (std::uint64_t offset = 0; offset < total;) {
        const std::uint64_t length = std::min(slice, total - offset);
        budget.acquire(length);
        visit(offset, length);
        offset += length;
    }
}

inline const char* host_environment(const char* name) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996) // read-only lookup
#endif
    return std::getenv(name);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
}

// STRIX_HOST_BYTE_CAP_GBPS: background ceiling in GB/s while a session decodes (default 10; 0 = never capped).
inline double byte_cap_gbytes_ps_from_environment(double fallback = 10.0) {
    const char* value = host_environment("STRIX_HOST_BYTE_CAP_GBPS");
    if (!value) return fallback;
    char* end = nullptr;
    const double cap = std::strtod(value, &end);
    if (end == value || *end != '\0' || !(cap >= 0.0))
        throw std::invalid_argument("STRIX_HOST_BYTE_CAP_GBPS is not a non-negative number");
    return cap;
}

// The process-wide budget: decode loops mark it, shared background paths (checkpoint I/O, cache hashing) and
// the process-wide host lane draw from it.
inline ByteBudget& shared_byte_budget() {
    static ByteBudget budget(byte_cap_gbytes_ps_from_environment() * 1e9);
    return budget;
}
} // namespace strix::core
