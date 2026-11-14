#pragma once

#include "strix/serve/api.hpp"
#include "strix/session/engine.hpp"
#include "strix/session/kv.hpp"
#include "strix/session/prefix_cache.hpp"

namespace strix::serve {

// Borrowed only during a hook. The owner retains tickets, handles and accounting.
struct SessionView {
    const GenerationJob& job;
    session::KvHandle state;
    const GenerationStats& stats;
    std::size_t prompt_offset;
    std::uint64_t rng_draws;
    std::uint32_t phase_remaining;
};
enum class SessionPhase { prefill, decode, mixed };
struct SessionWork {
    SessionPhase phase;
    // Indices into this tick's views, in caller order. No lane twice per tick.
    std::vector<std::size_t> lanes;
    // One positive suffix length per prefill lane; empty for decode.
    std::vector<std::size_t> chunks;
};
struct SessionContinuation {
    // Append new input to the existing frontier, including its pending anchor.
    std::vector<std::uint32_t> suffix, stop_tokens;
    std::uint32_t allowance = 0;
    enum class Mode { append, restart } mode = Mode::append;
};
enum class SessionCheckpoint { progress, terminal };
class SessionRuntime;
// Borrowed native-planner binding. The runtime alone owns the TakenJob, stats
// and lease. References expire at complete(); a planner keeps only metadata.
struct SessionLease {
    TakenJob& taken;
    GenerationStats& stats;
    session::KvHandle state;
    bool resident_reply = false;  // the resident hit reused consumed reply rows (reply-extended record)
};
// Taken request held by this owner without a logical lease or physical slot.
// Generations are never reused within an owner. Copies convey no ownership.
struct PendingSession {
    std::uint64_t generation = 0, owner = 0;
    [[nodiscard]] bool valid() const noexcept { return generation && owner; }
    bool operator==(const PendingSession&) const = default;
};
struct SessionCacheStats {
    std::uint64_t entries = 0, bytes = 0, hits = 0, misses = 0;
    std::uint64_t turn_admitted = 0, turn_refused = 0, turn_failed = 0, restore_failed = 0;
    // Resident-slot reuse (activate with a resident alignment). Every lookup is a hit or a miss; hit_rows sums the
    // rows a hit skipped; reply_hits counts hits whose reused rows reach into consumed reply rows. kept/reply_kept
    // count records certified at completion, claim_refused the reply claims the runtime declined to certify.
    std::uint64_t resident_hits = 0, resident_misses = 0, resident_hit_rows = 0, resident_reply_hits = 0;
    std::uint64_t resident_kept = 0, resident_reply_kept = 0, resident_claim_refused = 0;
};
// A family's explicit claim that a lane ended on a committed frontier whose first `rows` rows are canonical model
// state for tokens prompt + reply: rows == prompt.size() + reply_tokens == the store's consumed frontier. Only the
// family knows whether decode-produced rows may serve a later prefill; without a claim the shared default (prefill
// rows only, decode rows throw) is unchanged.
struct ResidentClaim {
    std::uint64_t rows = 0;
    std::uint64_t reply_tokens = 0;
};
struct SessionTurnCache {
    session::TurnMetadata metadata;
    // Upper bound on ALL retained rank payloads, including recurrent/conv,
    // draft and history, supplied by the producer before sealing work. Zero
    // means unknown and refuses retention before any commit/capture work.
    std::uint64_t max_state_bytes = 0;
};
struct SessionHooks {
    // Required stable model/profile/physical-format/rank identity for caching.
    session::Identity identity;
    // Certifies the arithmetic selected by the original full prompt, including reply retention.
    std::function<session::Identity(const GenerationJob&)> resident_identity;
    // Additional family admission checks, always before acquire/restore/prefill.
    std::function<void(const GenerationJob&)> validate;
    // Typed family context must be checked by validate. opened binds a freshly
    // acquired/reset lease before lookup, including full-prompt phase restart.
    std::function<void(const SessionView&)> opened;
    std::function<void(const SessionView&)> ready;
    std::function<std::uint64_t(const SessionView&)> initial_cursor;
    std::function<bool(const SessionView&)> cancelled;
    bool prompt_restart_cache = false;
    // Optional first phase budget, bounded by the request's total allowance.
    // A terminal phase may continue through continuation below.
    std::function<std::uint32_t(const GenerationJob&)> initial_allowance;
    // Empty uses basic_host_schedule. A family chooses chunks/groups and can
    // retain its own planner state; this is not an alternate owner loop.
    std::function<std::vector<SessionWork>(std::span<const SessionView>)> schedule;
    QueueSelection selection = QueueSelection::fifo;
    std::function<bool(const GenerationJob&)> eligible;
    std::function<bool(std::chrono::steady_clock::time_point)> reorder;
    std::function<bool(const GenerationJob&, std::chrono::steady_clock::time_point)> timed_eligible;
    std::function<void()> overtake_blocked;
    // Called only at a successful prefill boundary with no pending anchor.
    // Default refuses retention: arithmetic/boundary safety is family-owned.
    std::function<bool(const SessionView&)> checkpoint;
    std::function<bool(const SessionView&, SessionCheckpoint)> prompt_checkpoint;
    // Called at terminal decode boundaries. New phases retain RNG and the
    // request's total output allowance. Cancellation never continues.
    std::function<std::optional<SessionContinuation>(const SessionView&, session::FinishReason)> continuation;
    // Receipt formatting is family-owned. Called before release, even on error.
    std::function<void(const SessionView&, GenerationStats&)> receipt;
    // Explicit opt-in for family-owned named-session protocols. This validator
    // must check BOTH public identifier forms and compatibility before acquire.
    // Without it, session/session_id retain their blanket refusal.
    std::function<void(const GenerationJob&)> validate_session;
    // Re-evaluated before each take, without removing a blocked FIFO head.
    // Views are borrowed for the call only; this gate runs outside the queue lock.
    std::function<bool(std::span<const SessionView>)> can_admit;
    // Serialized work before admission with no live lanes (bounded admin/session,
    // heartbeat, transport quiesce). A hook uses a timed queue wait, never a
    // blocked take or a second owner. Producers wake it with wake_control().
    std::function<void()> idle;
    std::chrono::milliseconds idle_interval{2};
    // Called once by run(), after shutdown and lane cleanup, including errors.
    std::function<void()> stopped;
    // Bounded native planning turn for families whose layer/control boundaries
    // precede canonical token progress. Queue ownership stays in this runtime:
    // reserve -> inspect/defer -> activate/refuse -> publish -> complete.
    // This turn must return after bounded native work; it must never run a
    // request loop or call the queue/output callbacks itself.
    std::function<void(SessionRuntime&)> control_turn;
    // Native protocols may validate their richer request ABI instead of the
    // scalar SamplerState admission contract. Never used by generic execution.
    std::function<void(const GenerationJob&)> native_validate;
    // Some native runners must drain accepted/reserved work after queue stop.
    // Default preserves the immediate-close MiniMax control protocol.
    bool drain_native_on_shutdown = false;
    // Required for draining protocols: true only AFTER their final bounded
    // native shutdown turn, including native growth/admission boundaries.
    std::function<bool()> native_drained;
    // A request-only adopter has no shared retaining-cache facade at all.
    // Requires owner_control and a disabled retention policy.
    bool disable_native_cache = false;
    // Opt-in generate() protocol: cancelled FIFO entries reach the observer;
    // TTFT is entry-relative, and escaping idle/observer callbacks do not close
    // the queue. Existing native/batched owners retain their prior protocol.
    bool serial_queue_protocol = false;
    std::function<void(const GenerationStats&)> serial_observe;
    std::function<void()> closing;
    // Notify a native protocol of an escaping owner failure before cleanup.
    std::function<void()> failed;
    std::function<bool()> stop_requested;
    std::function<std::chrono::steady_clock::time_point()> now;
    std::function<void(std::chrono::milliseconds)> wait;
    // Default absent. Frozen after opened; no wire/session identifier inference.
    std::function<std::optional<SessionTurnCache>(const SessionView&)> turn_cache;
    // Explicit family final-phase admission, after continuation has declined.
    // Refuse incomplete thinking, failed/cancelled/in-flight state and unsupported
    // complete-state producers. One bounded sealing/capture on this same owner.
    std::function<bool(const SessionView&, session::FinishReason)> turn_checkpoint;
    // Optional candidate preparation after successful commit/frontier validation,
    // before capture_prompt. False refuses caching; throw fails caching only.
    // Absent preserves producers that prepare their candidate during commit.
    std::function<bool(const SessionView&)> turn_snapshot;
};

// Fixture policy: whole-suffix prefill, then ordered native decode batches.
// It is not a qualified production scheduling choice for any model family.
std::vector<SessionWork> basic_host_schedule(std::span<const SessionView>, const session::Capabilities&);

class SessionRuntime {
public:
    SessionRuntime(GenerationQueue&, session::Engine&, session::KvStore&,
                   session::Policy cache_policy = {}, SessionHooks hooks = {},
                   std::shared_ptr<PrefixStateCache> cache_owner = {});
    ~SessionRuntime();
    SessionRuntime(const SessionRuntime&) = delete;
    SessionRuntime& operator=(const SessionRuntime&) = delete;
    // Synchronous owner-thread entry. step(false) makes deterministic host
    // replay possible without a thread. shutdown wakes a blocked run().
    void run();
    bool step(bool wait = false);
    void shutdown() noexcept;
    [[nodiscard]] std::size_t live() const noexcept;
    [[nodiscard]] std::size_t cache_entries() const noexcept;
    [[nodiscard]] const std::shared_ptr<PrefixStateCache>& cache_owner() const noexcept;
    [[nodiscard]] SessionCacheStats cache_stats() const noexcept;
    // Library control only, serialized on the same owner as step/control_turn.
    // Remove named protection without releasing or mutating a live lane.
    bool unpin_turn_cache(const std::string& keep);
    [[nodiscard]] session::Capabilities capabilities() const;
    // Honest ApiService wiring: reject sampling downgrades, expose per-field
    // support in capabilities; runtime admission still validates every field.
    void configure_api(ApiConfig&) const;
    // Native planner operations, serialized on the owner thread. Reservation
    // retains the selected candidate's original submission/reorder metadata.
    const TakenJob* reservation() const noexcept;
    bool reserve(QueueSelection,
        const std::function<bool(std::chrono::steady_clock::time_point)>& reorder = {},
        const std::function<bool(const GenerationJob&, std::chrono::steady_clock::time_point)>& eligible = {},
        const std::function<void()>& overtake_blocked = {});
    // Native legacy FIFO uses the queue's original cancellation sweep (the
    // bool take API), including arrivals during cancelled-entry cleanup.
    // wait=true preserves the legacy blocking FIFO take, including its repeated
    // cancellation sweep and control wake. step(false) remains nonblocking.
    bool reserve_fifo(bool wait = false, bool short_first = false);
    bool reserve_serial(bool wait = false);
    void refuse(GenerationStats = {}, std::string error = {});
    SessionLease activate(std::uint64_t resident_alignment = 0, bool preserve_take_time = false);
    // Request-only native protocols may hold multiple taken candidates while
    // their native planner defers admission. No queue/accounting operation or
    // KvStore acquire/reset occurs here. Inspection is borrowed and stable
    // across other pending operations AND promotion, until terminal completion.
    // Callbacks must be invoked through the owner operations below.
    PendingSession hold_reservation();
    [[nodiscard]] const TakenJob& inspect(PendingSession) const;
    [[nodiscard]] const TakenJob& inspect(session::KvHandle) const;
    [[nodiscard]] bool pending_cancelled(PendingSession) const noexcept;
    [[nodiscard]] std::size_t pending_count() const noexcept;
    SessionLease promote(PendingSession);
    void refuse_pending(PendingSession, GenerationStats = {}, std::string error = {});
    // Native planner only. Invalidate before session/admin work outside leases.
    void invalidate_residents() noexcept;
    void abandon_resident(session::KvHandle);
    bool has_waiting(const std::function<bool(const GenerationJob&, std::chrono::steady_clock::time_point)>&) const;
    // Accounts before callback, catches callback errors per lane, and preserves
    // already-published peer commits. Family supplies stop/commit semantics.
    bool publish(session::KvHandle, std::span<const std::uint32_t>, bool deliver = true,
                 bool propagate_callback_error = false);
    // Owner-thread native callback. Retired/foreign generations are cancelled;
    // never borrow a raw ticket beyond the activation call or observe its reuse.
    [[nodiscard]] bool ticket_cancelled(session::KvHandle) const noexcept;
    // Native protocol events at their existing admission/progress boundaries.
    // The runtime retains the ticket callbacks; native metadata keeps handles.
    void notify_admitted(session::KvHandle);
    void publish_progress(session::KvHandle, GenerationProgress);
    void complete(session::KvHandle, std::string error = {}, std::uint64_t safe_prefill_rows = 0);
    // Reply-extended resident record. Retained only when the lane is not cancelled/failed, the store sits exactly at
    // claim.rows with the lane's prompt as the head of its tokens; otherwise nothing is retained (never throws for
    // an uncertifiable claim). Costs no memory beyond the free slot's own KV and is dropped when the slot is taken.
    void complete(session::KvHandle, std::string error, const ResidentClaim& claim);
    // Native protocol already formed its complete terminal receipt. Replace,
    // never merge/add its output accounting; completion still owns the lease.
    void complete_native(session::KvHandle, GenerationStats, std::string error = {});
    void wait_for(std::chrono::milliseconds, bool generation_ready = false);
    bool stopping() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strix::serve
