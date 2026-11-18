#pragma once
// Protection of the prefix states that requests still WAITING for a slot will restore.
//
// Without it the restore lookup runs only after admission, so a capture made while a follow-up waits can
// evict the one state that follow-up needs. Here the owner thread scans its queue at each admission turn
// (bounded per scan), prepares each waiting prompt once (block digests only, no cache state), takes one cache
// reference on the longest compatible state, and hands that reference to the admitted job, which holds it
// until its restore lookup has run (first output, or its end). A job that leaves the queue any other way
// (cancelled, timed out, failed, refused) releases at the next scan. The cache honours the reference in
// every eviction path; a byte cap bounds what can be held (see PrefixStateCache::protect_longest).
//
// Engine-neutral: the cache binding is two callbacks, so Qwen, DS41, GLM and MiniMax owners bind the same
// object. Retention-only (no request arithmetic changes): bitwise lever, default OFF until measured.
#include "strix/lever.hpp"
#include "strix/serve/engine.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace strix::serve {

inline constexpr lever::Decl queued_prefix_protection_lever{
    "prefix-queue-protect", "STRIX_PREFIX_QUEUE_PROTECT", lever::Exactness::bitwise, false, {}, {},
    "pending: node receipt 16 follow-ups x 17K, 8 slots, capture pressure (qcprot RESULT)"};

struct QueuedPrefixProtectionPolicy {
    bool enabled = false;
    std::size_t max_waiting = 64;  // waiting jobs examined per scan, in queue order
    std::size_t max_prepare = 4;   // new prompts prepared (copied + hashed) per scan
    std::size_t max_protect = 16;  // protection attempts per scan (round-robin retries)
};
// One resolution for every server owner: STRIX_PREFIX_QUEUE_PROTECT ("1" on; unset = default OFF).
[[nodiscard]] inline QueuedPrefixProtectionPolicy queued_prefix_protection_policy_from_env() {
    QueuedPrefixProtectionPolicy p;
    p.enabled = lever::select(queued_prefix_protection_lever, {}).on;
    return p;
}

class QueuedPrefixProtection {
public:
    using Prepared = std::shared_ptr<const void>;
    using Handle = std::shared_ptr<void>;  // opaque; destruction releases the cache reference
    struct Binding {
        std::function<Prepared(std::span<const std::uint32_t>)> prepare;  // null result: not cacheable
        std::function<Handle(const Prepared&)> protect;                    // null result: miss or cap refusal
        explicit operator bool() const noexcept { return prepare && protect; }
    };
    struct Stats {
        std::uint64_t scans = 0, prepared = 0, attempts = 0, protected_jobs = 0;
        std::uint64_t released_left = 0, handed_over = 0;
        std::size_t tracked = 0, held = 0;
    };

    QueuedPrefixProtection() = default;
    QueuedPrefixProtection(QueuedPrefixProtectionPolicy policy, Binding binding)
        : policy_(policy), binding_(std::move(binding)) {}
    [[nodiscard]] bool enabled() const noexcept { return policy_.enabled && static_cast<bool>(binding_); }

    // Owner thread, before a reservation. Cost: one queue-lock visit of at most max_waiting entries (prompt
    // copies only for at most max_prepare new jobs), then at most max_prepare preparations and
    // max_protect protection attempts outside the queue lock.
    void scan(const GenerationQueue& queue) {
        if (!enabled()) return;
        ++epoch_; ++stats_.scans;
        struct Visit { const GenerationTicket* key; std::weak_ptr<GenerationTicket> ticket; std::vector<std::uint32_t> prompt; bool fresh; };
        std::vector<Visit> visits;
        auto prepare_budget = policy_.max_prepare;
        queue.visit_waiting(policy_.max_waiting, [&](const GenerationJob& job, const std::shared_ptr<GenerationTicket>& ticket) {
            const auto it = jobs_.find(ticket.get());
            if (it != jobs_.end() && same(it->second.ticket, ticket)) { visits.push_back({ticket.get(), ticket, {}, false}); return true; }
            if (!prepare_budget) return true;  // tracked at a later scan
            --prepare_budget;
            visits.push_back({ticket.get(), ticket, job.prompt, true});
            return true;
        });
        for (auto& v : visits) {
            auto& job = jobs_[v.key];
            if (v.fresh) {
                job = Job{};  // a stale entry at a reused address belongs to an ended ticket
                job.ticket = v.ticket;
                if (!v.prompt.empty()) job.prepared = binding_.prepare(v.prompt);
                ++stats_.prepared;
            }
            job.seen = epoch_;
        }
        for (auto it = jobs_.begin(); it != jobs_.end();) {
            if (it->second.seen == epoch_) { ++it; continue; }
            if (it->second.handle) ++stats_.released_left;
            it = jobs_.erase(it);  // left the queue without admission: cancel, timeout, failure, refusal
        }
        auto attempts = policy_.max_protect;
        const auto start = visits.empty() ? 0 : next_attempt_ % visits.size();
        for (std::size_t i = 0; i < visits.size() && attempts; ++i) {
            const auto index = (start + i) % visits.size();
            const auto& v = visits[index];
            next_attempt_ = index + 1;
            auto& job = jobs_.at(v.key);
            if (job.handle || !job.prepared) continue;
            --attempts; ++stats_.attempts;
            job.handle = binding_.protect(job.prepared);
            if (job.handle) ++stats_.protected_jobs;
        }
        refresh();
    }
    // The admitted job's reference (may be null); the caller keeps it until the restore lookup has run.
    [[nodiscard]] Handle take(const std::shared_ptr<GenerationTicket>& ticket) {
        const auto it = jobs_.find(ticket.get());
        if (it == jobs_.end() || !same(it->second.ticket, ticket)) return {};
        auto handle = std::move(it->second.handle);
        jobs_.erase(it);
        if (handle) ++stats_.handed_over;
        refresh();
        return handle;
    }
    void clear() noexcept { jobs_.clear(); refresh(); }
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
    struct Job { std::weak_ptr<GenerationTicket> ticket; Prepared prepared; Handle handle; std::uint64_t seen = 0; };
    static bool same(const std::weak_ptr<GenerationTicket>& a, const std::shared_ptr<GenerationTicket>& b) noexcept {
        return !a.owner_before(b) && !b.owner_before(a);
    }
    void refresh() noexcept {
        stats_.tracked = jobs_.size(); stats_.held = 0;
        for (const auto& [key, job] : jobs_) stats_.held += job.handle ? 1U : 0U;
    }
    QueuedPrefixProtectionPolicy policy_{};
    Binding binding_{};
    std::map<const GenerationTicket*, Job> jobs_;
    std::uint64_t epoch_ = 0;
    std::size_t next_attempt_ = 0;
    Stats stats_{};
};

// A held handle released once: at the admitted job's first output or its end, whichever comes first.
class RestoreHold {
public:
    explicit RestoreHold(QueuedPrefixProtection::Handle h) : handle_(std::move(h)) {}
    void release() noexcept { std::lock_guard lock(mutex_); handle_.reset(); }
    [[nodiscard]] bool held() const noexcept { std::lock_guard lock(mutex_); return static_cast<bool>(handle_); }
private:
    mutable std::mutex mutex_;
    QueuedPrefixProtection::Handle handle_;
};

} // namespace strix::serve
