#include "http_session_owner.hpp"

#include <algorithm>

namespace strix::qwen {
namespace {
[[noreturn]] void unsupported() { throw std::invalid_argument("Qwen native-control owner has no generic state execution"); }
session::Policy no_retention() {
    session::Policy p; p.enabled = false; p.store = false; p.completed_turns = false;
    return p;
}
std::string error_text() {
    try { throw; }
    catch (const std::exception& e) { return e.what()[0] ? e.what() : "engine failure"; }
    catch (...) { return "engine failure"; }
}
}

HttpSessionOwner::ControlEngine::ControlEngine(std::uint64_t context, std::uint32_t capacity) {
    caps_.max_context = context; caps_.max_slots = capacity; caps_.max_batch = 1;
    caps_.owner_control = true;
    // Native ABI execution is supplied by Native, not scalar Engine methods.
    // No generic batching, snapshots, cache, fanout or arithmetic is advertised.
}
void HttpSessionOwner::ControlEngine::prefill(std::span<const std::uint32_t>, session::KvHandle&) { unsupported(); }
session::StepResult HttpSessionOwner::ControlEngine::decode(session::KvHandle&, const session::SamplerState&) { unsupported(); }
session::Capabilities HttpSessionOwner::ControlEngine::caps() const { return caps_; }
::strix::profile::ExecutionReceipt HttpSessionOwner::ControlEngine::receipt() const { return {}; }

session::Hooks HttpSessionOwner::logical_hooks() {
    session::Hooks h;
    h.initialize = [](session::KvHandle) {};
    h.reset = [](session::KvHandle) { unsupported(); };
    h.release = [](session::KvHandle) {};
    h.view = [](session::KvHandle, std::uint32_t) -> session::RankState { unsupported(); };
    h.capture = h.view;
    h.restore = [](session::KvHandle, std::uint32_t, const session::RankState&) { unsupported(); };
    h.truncate = [](session::KvHandle, std::uint32_t, std::uint64_t) { unsupported(); };
    return h;
}
serve::SessionHooks HttpSessionOwner::owner_hooks() {
    serve::SessionHooks h;
    h.control_turn = [this](serve::SessionRuntime&) {
        try { turn(); } catch (...) { failure_ = error_text(); throw; }
    };
    // Request translation retains the executing protocol's admission semantics,
    // including its richer drafting ABI and request-local sampling refusal.
    h.native_validate = [](const serve::GenerationJob& j) {
        if (j.family_context) throw std::invalid_argument("Qwen native family context unsupported");
    };
    h.drain_native_on_shutdown = true;
    h.native_drained = [this] { return drained_; };
    h.disable_native_cache = true;
    h.failed = [this] {
        if (native_.failed) native_.failed();
    };
    h.closing = [this] {
        if (!failure_.empty()) {
            // Preserve native failure receipts before common fallback cleanup
            // of an admission which never reached a native slot.
            native_.fail_requests(failure_);
        }
    };
    h.stopped = [this] {
        // Shared fallback tickets/reservation finish BEFORE physical abandon,
        // matching the native protocol even when admission never set active.
        if (!failure_.empty()) native_.abandon();
        if (batch_lock_.owns_lock()) batch_lock_.unlock();
    };
    return h;
}
HttpSessionOwner::HttpSessionOwner(serve::GenerationQueue& q, std::uint64_t context,
    std::uint32_t capacity, Native native)
    : queue_(q), context_(context), native_(std::move(native)), engine_(context, capacity),
      leases_(capacity, 1, logical_hooks()), runtime_(q, engine_, leases_, no_retention(), owner_hooks()) {
    if (!native_.add_slot || !native_.has_free_slot || !native_.active || !native_.submit ||
        !native_.service_admission || !native_.step || !native_.fail_requests || !native_.abandon)
        throw std::invalid_argument("Qwen owner requires the native runner binding");
    if (native_.session_mutex) batch_lock_ = std::unique_lock<std::mutex>(*native_.session_mutex, std::defer_lock);
    native_.service_admission([this] {
        return runtime_.reservation() || !queue_.has_queued_work() ? std::size_t{0} :
            std::max<std::size_t>(1, static_cast<std::size_t>(queue_.snapshot().queued));
    }, [this](const Submit& prepare, const std::function<bool()>& can_prepare) { pump(prepare, can_prepare); });
}
HttpSessionOwner::~HttpSessionOwner() {
    if (runtime_.live() || runtime_.reservation()) {
        try { native_.fail_requests("server shutting down"); } catch (...) {}
        native_.abandon();
    }
    // Clear borrowed native callbacks while runtime/leases are still alive.
    native_.service_admission({}, {});
}
void HttpSessionOwner::run() {
    try { runtime_.run(); }
    catch (...) {
        if (batch_lock_.owns_lock()) batch_lock_.unlock();
        throw;
    }
}
void HttpSessionOwner::named(serve::SessionLease lease) {
    if (!native_.named || native_.active()) throw std::logic_error("named session requires an idle session runner");
    if (batch_lock_.owns_lock()) batch_lock_.unlock();
    // Keep the original TakenJob callback untouched. A temporary job translates
    // the fixed protocol's output back to the same generation-checked owner.
    auto job = lease.taken.job;
    job.on_tokens = [this, h=lease.state](auto ids) { return runtime_.publish(h, ids, true, true); };
    job.on_admitted = [this, h=lease.state] { runtime_.notify_admitted(h); };
    job.on_progress = [this, h=lease.state](auto p) { runtime_.publish_progress(h, p); };
    serve::GenerationStats stats;
    try { stats = native_.named(job); }
    catch (const serve::GenerationRequestError& e) {
        runtime_.complete_native(lease.state, e.stats, e.what()[0] ? e.what() : "engine failure");
        if (native_.named_failed && native_.named_failed()) throw;
        return;
    } catch (const std::exception& e) {
        stats.finish = "execution_failure";
        runtime_.complete_native(lease.state, std::move(stats), e.what()[0] ? e.what() : "engine failure");
        throw;
    } catch (...) {
        runtime_.complete_native(lease.state, {}, "engine failure");
        throw;
    }
    runtime_.complete_native(lease.state, std::move(stats));
}
void HttpSessionOwner::admit(const Submit& submit) {
    const auto* reserved = runtime_.reservation();
    if (!reserved) throw std::logic_error("Qwen admission without owner reservation");
    if (reserved->job.prompt.empty() || reserved->job.prompt.size() >= context_) {
        runtime_.refuse({}, "prompt is empty or fills the context"); return;
    }
    auto lease = runtime_.activate(0, true);
    if (!lease.taken.job.session_id.empty()) { named(lease); return; }
    Callbacks callbacks;
    callbacks.tokens = [this, h=lease.state](auto ids) { return runtime_.publish(h, ids, true, true); };
    callbacks.progress = [this, h=lease.state](auto p) { runtime_.publish_progress(h, p); };
    callbacks.cancelled = [this, h=lease.state] {
        return runtime_.ticket_cancelled(h) || (native_.stop_requested && native_.stop_requested());
    };
    callbacks.finish = [this, h=lease.state](auto stats, auto error) {
        runtime_.complete_native(h, std::move(stats), std::move(error));
    };
    callbacks.before_submit = [this] {
        if (native_.session_mutex && !batch_lock_.owns_lock()) batch_lock_.lock();
    };
    // Translation may refuse sampling before taking the native session mutex.
    submit(lease.taken.job, std::move(callbacks));
}
void HttpSessionOwner::pump(const Submit& prepare, const std::function<bool()>& can_prepare) {
    // A native boundary reenters this SAME owner synchronously. No queue take,
    // lease allocation in the runner, native growth, or wait is added here.
    // Native preparation does not execute prompts or complete successful jobs:
    // successful arrivals are bounded by free physical slots. Bound refusing
    // candidates too, so a producer continuously replacing refused requests
    // cannot monopolize a parked native service boundary.
    auto remaining = queue_.snapshot().queued;
    const auto slots = engine_.caps().max_slots;
    remaining = remaining > UINT64_MAX - slots ? UINT64_MAX : remaining + slots;
    while (!runtime_.reservation() && can_prepare()) {
        if (!remaining) break;
        --remaining;
        if (!runtime_.reserve_fifo()) break;
        if (!runtime_.reservation()->job.session_id.empty()) break;
        admit(prepare);
    }
}
void HttpSessionOwner::turn() {
    // One candidate per common-owner turn. Preserve the two native growth
    // points across these bounded turns; yielding never adds native calls.
    if (turn_ == Turn::growth) {
        native_.add_slot(queue_.snapshot().queued != 0);
        turn_ = Turn::admission;
    }
    if (turn_ == Turn::admission) {
        if (native_.has_free_slot() && !(runtime_.reservation() && native_.active())) {
            if (!runtime_.reserve_fifo() && !native_.active() && !queue_.stopping()) {
                runtime_.wait_for(std::chrono::seconds(1), true);
                (void)runtime_.reserve_fifo();
            }
            if (runtime_.reservation() && (runtime_.reservation()->job.session_id.empty() || !native_.active())) {
                admit(native_.submit);
                return;
            }
        }
        turn_ = Turn::work;
    }
    native_.add_slot(queue_.snapshot().queued != 0);
    turn_ = Turn::growth;
    if (!native_.active()) {
        drained_ = queue_.stopping();
        return;
    }
    native_.step();
    if (!native_.active() && batch_lock_.owns_lock()) batch_lock_.unlock();
}

} // namespace strix::qwen
