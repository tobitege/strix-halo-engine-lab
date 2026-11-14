#include "strix/serve/session_runtime.hpp"
#include "../core/byte_budget.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <stop_token>

namespace strix::serve {
namespace {
using session::FinishReason;
std::atomic<std::uint64_t> pending_owner_sequence{1};
std::uint64_t logical(const session::Frontier& f) {
    if (f.pending && f.consumed_tokens == UINT64_MAX) throw std::runtime_error("session frontier overflow");
    return f.consumed_tokens + f.pending.has_value();
}
double elapsed(std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}
const char* name(FinishReason r) {
    switch (r) {
    case FinishReason::stop: return "stop";
    case FinishReason::context: return "context";
    case FinishReason::cancelled: return "cancelled";
    default: return "length";
    }
}
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
// Embedding rows (session/embed_rows.hpp) pass through untouched: the runtime only checks, before a lane exists,
// that the engine takes them and that the job's table resolves every virtual id of its prompt. Text jobs on an
// engine without the capability pay one scan and change nothing.
void embed_admission(const GenerationJob& j, const session::Capabilities& c) {
    if (!c.embed_rows) {
        if (j.embeds) throw std::invalid_argument("unsupported or invalid request field: embeds");
        session::reject_embed_ids(j.prompt);
    } else if (j.embeds) j.embeds->validate(j.prompt);
    else session::reject_embed_ids(j.prompt);
}
// Gives back a lease that never ran: nothing was written, so a plain release is right; a store that refuses
// it quarantines the slot instead of leaking it.
void drop_unused(session::KvStore& store, session::KvHandle handle) noexcept {
    try { store.release(handle); } catch (...) { store.invalidate(handle); }
}
void admission(const GenerationJob& j, const session::Capabilities& c, const SessionHooks& hooks) {
    const auto& s = j.sampling;
    const auto reject = [](bool bad, const char* field) {
        if (bad) throw std::invalid_argument(std::string("unsupported or invalid request field: ") + field);
    };
    reject(j.prompt.empty() || j.prompt.size() > c.max_context, "prompt");
    embed_admission(j, c);
    reject(!std::isfinite(s.temperature) || s.temperature < 0 || (s.temperature != 0 && !c.temperature), "temperature");
    reject(!std::isfinite(s.top_p) || s.top_p <= 0 || s.top_p > 1 || (s.top_p != 1 && !c.top_p), "top_p");
    reject(s.top_k != 0 && !c.top_k, "top_k");
    reject(!std::isfinite(s.min_p) || s.min_p < 0 || s.min_p >= 1 || (s.min_p != 0 && !c.min_p), "min_p");
    reject(s.seeded && !c.seed, "seed");
    if (j.constraint) {
        reject(!j.constraint->program && !j.constraint->thinking && !j.constraint->transform.active() &&
               !j.constraint->logprobs, "constraint");
        reject(j.constraint->program && !c.constraints, "response_format");
        reject(j.constraint->thinking.has_value() && !c.thinking_budget, "thinking_budget");
        reject(j.constraint->transform.active() && !c.logit_transforms, "logit_transforms");
        reject(j.constraint->logprobs && !c.logprobs, "logprobs");
        reject(j.constraint->top_logprobs != 0 && !j.constraint->logprobs, "top_logprobs");
    }
    // These fields are not represented in session::SamplerState yet.
    reject(j.ngram_draft.has_value(), "ngram_draft");
    reject(j.draft_depth_max != 0, "draft_depth_max");
    reject(j.tool_draft.has_value(), "tool_draft");
    reject(j.ngram_peers.has_value(), "ngram_peers");
    if (!j.session.empty() || !j.session_id.empty()) {
        reject(!hooks.validate_session, "session");
        hooks.validate_session(j);
    }
}
}

std::vector<SessionWork> basic_host_schedule(std::span<const SessionView> views, const session::Capabilities& caps) {
    std::vector<SessionWork> out;
    SessionWork decode{SessionPhase::decode, {}, {}};
    for (std::size_t i = 0; i < views.size(); ++i) {
        if (views[i].prompt_offset < views[i].job.prompt.size())
            out.push_back({SessionPhase::prefill, {i}, {views[i].job.prompt.size() - views[i].prompt_offset}});
        else {
            decode.lanes.push_back(i);
            if (!caps.batch_decode || decode.lanes.size() == caps.max_batch) {
                out.push_back(std::move(decode)); decode = {SessionPhase::decode, {}, {}};
            }
        }
    }
    if (!decode.lanes.empty()) out.push_back(std::move(decode));
    return out;
}

struct SessionRuntime::Impl {
    struct Lane {
        TakenJob taken;
        session::KvHandle handle;
        GenerationStats stats;
        std::size_t offset = 0;
        std::uint64_t rng = 0;
        std::uint32_t phase_remaining = 0;
        std::stop_source cancel;
        bool done = false, receipt_done = false, cache_cleanup = false;
        std::string publication_error;
        session::Identity resident_identity;
        bool resident_reply_hit = false;  // the resident hit reused consumed reply rows
        std::optional<SessionTurnCache> turn;
        SessionView view() const { return {taken.job, handle, stats, offset, rng, phase_remaining}; }
    };
    GenerationQueue& queue;
    session::Engine& engine;
    session::KvStore& store;
    session::Policy policy;
    SessionHooks hooks;
    std::optional<session::PrefixCache> cache;
    session::Capabilities caps;
    std::vector<std::unique_ptr<Lane>> lanes;
    std::optional<TakenJob> reserved;
    // Heap Lane storage keeps borrowed request metadata stable on promotion.
    // Pending lanes have no KvHandle and consume no logical/native capacity.
    std::map<std::uint64_t, std::unique_ptr<Lane>> pending;
    const std::uint64_t pending_owner = pending_owner_sequence.fetch_add(1, std::memory_order_relaxed);
    std::uint64_t pending_generation = 0;
    std::string owner_error;
    bool closed = false;
    bool control_wait = false;
    SessionRuntime* owner = nullptr;
    std::uint64_t hits = 0, misses = 0;
    std::uint64_t resident_hits = 0, resident_misses = 0, resident_hit_rows = 0, resident_reply_hits = 0;
    std::uint64_t resident_kept = 0, resident_reply_kept = 0, resident_claim_refused = 0;
    bool stop_notified = false;
    std::uint64_t turn_admitted = 0, turn_refused = 0, turn_failed = 0, restore_failed = 0;
    void close() {
        if (closed) return;
        closed = true;
        std::exception_ptr failure;
        try { if (hooks.closing) hooks.closing(); } catch (...) { failure = std::current_exception(); }
        for (auto& l : lanes) if (!l->done) {
            if (!hooks.control_turn && caps.mixed_advance && caps.lane_cancellation) {
                try { engine.cancel_lane(l->handle); } catch (...) { store.invalidate(l->handle); }
            }
            if (hooks.drain_native_on_shutdown && !owner_error.empty()) l->stats = {};
            else l->stats.finish = "cancelled";
            finish(*l, failure ? "session owner failed" : owner_error);
        }
        if (reserved) { GenerationStats stats; if (!hooks.drain_native_on_shutdown || owner_error.empty()) stats.finish = "cancelled"; queue.finish(*reserved, stats, owner_error); reserved.reset(); }
        while (!pending.empty()) {
            auto l = std::move(pending.begin()->second);
            pending.erase(pending.begin()); // retire before terminal callbacks
            GenerationStats stats;
            stats.prompt_tokens = l->taken.job.prompt.size();
            if (owner_error.empty()) stats.finish = "cancelled";
            queue.finish(l->taken, std::move(stats), owner_error);
        }
        if (failure) std::rethrow_exception(failure);
    }
    void notify_stopped() {
        if (stop_notified) return;
        stop_notified = true;
        if (hooks.stopped) hooks.stopped();
    }

    Impl(GenerationQueue& q, session::Engine& e, session::KvStore& s, session::Policy p, SessionHooks h, std::shared_ptr<PrefixStateCache> pooled)
        : queue(q), engine(e), store(s), policy(p), hooks(std::move(h)), caps(e.caps()) {
        if ((hooks.native_validate || hooks.drain_native_on_shutdown || hooks.native_drained || hooks.disable_native_cache || hooks.serial_queue_protocol || hooks.serial_observe) && !caps.owner_control)
            throw std::invalid_argument("native protocol requires owner_control");
        if (hooks.drain_native_on_shutdown && !hooks.native_drained)
            throw std::invalid_argument("native draining requires an explicit terminal boundary");
        if (hooks.disable_native_cache && (policy.enabled || policy.store))
            throw std::invalid_argument("request-only native owner requires retention disabled");
        if (hooks.serial_queue_protocol && (!hooks.disable_native_cache || hooks.drain_native_on_shutdown))
            throw std::invalid_argument("serial queue protocol requires request-only ownership");
        if (hooks.serial_observe && !hooks.serial_queue_protocol)
            throw std::invalid_argument("serial observer requires serial queue protocol");
        if (!hooks.disable_native_cache) cache.emplace(p, std::move(pooled));
        if ((hooks.idle || hooks.can_admit) && hooks.idle_interval <= std::chrono::milliseconds::zero())
            throw std::invalid_argument("session idle interval must be positive");
        if (!caps.max_slots || !caps.max_batch || !caps.max_context)
            throw std::invalid_argument("session runtime requires nonzero engine limits");
        if (caps.owner_control != static_cast<bool>(hooks.control_turn))
            throw std::invalid_argument("native session engine requires its bounded control planner");
        if (policy.enabled && caps.prefix_cache && !policy.completed_turns &&
            (!hooks.identity.known() || hooks.identity.ranks != store.ranks()))
            throw std::invalid_argument("cache requires a complete matching identity");
    }
    session::Frontier frontier(const Lane& l) const { return {store.consumed(l.handle), store.pending(l.handle)}; }
    bool caching() const {
        // A completed-turn miss must never fall back to a resume checkpoint's
        // old request RNG/policy. Ordinary prompt-cache defaults stay separate.
        return cache && policy.enabled && !policy.completed_turns && caps.prefix_cache &&
            hooks.identity.known() && hooks.identity.ranks == store.ranks();
    }
    bool turn_caching() const { return cache && cache->turns_enabled() && caps.turn_commit && !caps.owner_control; }
    // This operation is called ONLY after successful final-phase completion.
    // Every cache failure stays outside reply error/finish/receipt accounting.
    void completed_turn(Lane& l, FinishReason reason) noexcept {
        if (!turn_caching()) return;
        try {
            if (reason == FinishReason::none || reason == FinishReason::cancelled || cancelled(l) ||
                !l.publication_error.empty() || !l.turn || !policy.store ||
                l.taken.job.cache_retention == CacheRetention::no_store || !hooks.identity.known() ||
                hooks.identity.ranks != store.ranks() || !l.turn->metadata.known() ||
                !l.turn->max_state_bytes ||
                !cache->can_admit_turn(hooks.identity, l.turn->metadata, logical(frontier(l)), l.turn->max_state_bytes) ||
                !hooks.turn_checkpoint || !hooks.turn_checkpoint(l.view(), reason)) {
                ++turn_refused; return;
            }
            if (cancelled(l)) { ++turn_refused; return; }
            auto credit = cache->reserve_turn(hooks.identity, l.turn->metadata, logical(frontier(l)), l.turn->max_state_bytes);
            if (!credit) { ++turn_refused; return; }
            const auto before = frontier(l);
            l.cache_cleanup = true;
            engine.commit_pending(l.handle);
            require(frontier(l) == session::Frontier{logical(before), {}}, "turn seal frontier mismatch");
            if (cancelled(l)) { ++turn_refused; return; }
            if (hooks.turn_snapshot && !hooks.turn_snapshot(l.view())) { ++turn_refused; return; }
            if (cancelled(l)) { ++turn_refused; return; }
            auto snap = std::make_shared<session::PrefixSnapshot>();
            snap->identity = hooks.identity; snap->turn = l.turn->metadata;
            snap->state = store.capture_prompt(l.handle);
            require(snap->state.bytes <= l.turn->max_state_bytes, "turn payload exceeds declared bound");
            snap->attention_frontier = snap->recurrent_frontier = snap->state.consumed;
            if (cancelled(l)) { ++turn_refused; return; }
            if (cache->admit(std::move(snap), &credit)) ++turn_admitted; else ++turn_refused;
        } catch (...) { ++turn_failed; }
    }
    void receipt(Lane& l, std::string& error) {
        if (l.receipt_done) return;
        l.receipt_done = true;
        if (hooks.receipt && l.handle.valid()) {
            try { hooks.receipt(l.view(), l.stats); }
            catch (...) { if (error.empty()) error = "session receipt callback failed"; }
        }
    }
    // Rows a reply claim may certify, 0 = refuse. The lane must have ended cleanly on the exact committed frontier
    // with its own prompt as the head of the store's token history.
    std::uint64_t claim_rows(const Lane& l, const ResidentClaim& k, const std::string& error) const {
        const auto prompt = l.taken.job.prompt.size();
        if (!error.empty() || cancelled(l) || !k.rows || k.rows != prompt + k.reply_tokens ||
            k.rows != store.consumed(l.handle)) return 0;
        const auto history = store.tokens(l.handle);
        if (history.size() < k.rows || !std::equal(l.taken.job.prompt.begin(), l.taken.job.prompt.end(), history.begin())) return 0;
        return k.rows;
    }
    void finish(Lane& l, std::string error = {}, std::uint64_t resident_rows = 0, const ResidentClaim* claim = nullptr) {
        if (l.done) return;
        if (error.empty()) error = l.publication_error;
        receipt(l, error);
        if (l.handle.valid()) {
            try {
                std::uint64_t reply_rows = 0;
                if (claim) {
                    resident_rows = claim_rows(l, *claim, error);
                    reply_rows = resident_rows ? claim->reply_tokens : 0;
                    if (!resident_rows) ++resident_claim_refused;
                }
                const bool retain = resident_rows && error.empty() && policy.store && hooks.identity.known() &&
                    l.taken.job.cache_retention != CacheRetention::no_store;
                if (cache) cache->invalidate_resident(l.handle);
                if (!retain) store.release(l.handle);
                else {
                    if (resident_rows > l.taken.job.prompt.size() + reply_rows)
                        throw std::logic_error("decode rows cannot certify a resident prefill prefix");
                    const auto certificate = store.release_resident(l.handle, resident_rows);
                    try {
                        if (cache->admit_resident(store, l.handle, certificate->tokens(), l.resident_identity, reply_rows)) {
                            ++resident_kept; if (reply_rows) ++resident_reply_kept;
                        }
                    }
                    catch (...) { /* Optional index allocation cannot fail a delivered reply. */ }
                }
            }
            catch (...) {
                // A throwing release leaves physical ownership uncertain. The
                // store quarantines this generation; never reuse it silently.
                store.invalidate(l.handle);
                if (l.cache_cleanup) ++turn_failed;
                else if (error.empty()) error = "session cleanup failed; slot quarantined";
            }
        }
        l.done = true;
        queue.finish(l.taken, std::move(l.stats), std::move(error), hooks.serial_observe,
            hooks.serial_queue_protocol ? QueueCompletion::serial : QueueCompletion::taken);
    }
    void fail(Lane& l, const std::string& error) {
        // Exceptions from adapters may leave invalid physical state. Release is
        // still attempted, with quarantine on failure. Already published IDs
        // and their accounting survive all subsequent errors.
        finish(l, error);
    }
    // Open prompts (GenerationJob::prompt_tail). All of this runs on the runtime thread.
    static void settle_prompt(GenerationJob& job) {
        if (!job.prompt_tail) return;
        while (!job.prompt_tail->take(job.prompt, true)) {}
        job.prompt_tail.reset();
    }
    void grow_prompt(Lane& l, bool wait) {
        try {
            if (l.taken.job.prompt_tail->take(l.taken.job.prompt, wait)) l.taken.job.prompt_tail.reset();
            l.stats.prompt_tokens = l.taken.job.prompt.size();
        } catch (const std::exception& e) {
            l.taken.job.prompt_tail.reset();
            fail(l, e.what());
        }
    }
    void admit(TakenJob taken) {
        auto l = std::make_unique<Lane>();
        l->taken = std::move(taken);
        std::string tail_error;
        if (l->taken.job.prompt_tail) {
            // A cache lookup or an owner-controlled turn decides on the whole prompt: wait for it. Otherwise the
            // lane is admitted on the ids that are final now and grows before each round.
            try {
                if (caching() || turn_caching() || caps.owner_control) settle_prompt(l->taken.job);
                else if (l->taken.job.prompt_tail->take(l->taken.job.prompt, l->taken.job.prompt.empty())) l->taken.job.prompt_tail.reset();
            } catch (const std::exception& e) {
                tail_error = e.what();
                l->taken.job.prompt_tail.reset();
            }
        }
        l->stats.prompt_tokens = l->taken.job.prompt.size();
        l->stats.sampling = l->taken.job.sampling.temperature > 0 ? "sample" : "greedy";
        l->phase_remaining = l->taken.job.max_tokens;
        bool preflight = true;
        try {
            if (!tail_error.empty()) throw std::runtime_error(tail_error);
            admission(l->taken.job, caps, hooks);
            if(l->taken.job.family_context && !hooks.validate) throw std::invalid_argument("unvalidated family context");
            if (hooks.validate) hooks.validate(l->taken.job);
            if (hooks.initial_allowance) {
                l->phase_remaining = hooks.initial_allowance(l->taken.job);
                if (l->phase_remaining > l->taken.job.max_tokens ||
                    (!l->phase_remaining && l->taken.job.max_tokens))
                    throw std::invalid_argument("invalid first phase allowance");
            }
            preflight = false;
            l->handle = store.acquire();
            if (l->taken.job.embeds) engine.bind_embeds(l->handle, l->taken.job.embeds);
            open(*l);
            if (l->taken.job.on_admitted) l->taken.job.on_admitted();
            progress(*l);
            lanes.push_back(std::move(l));
        } catch (const std::invalid_argument& e) {
            if (preflight) { l->stats.error_status = 400; l->stats.error_code = "invalid_request"; }
            fail(*l, e.what());
        } catch (const std::exception& e) { fail(*l, e.what()); }
        catch (...) { fail(*l, "session admission failed"); }
    }
    bool cancelled(const Lane& l) const {
        return queue.stopping() || l.taken.ticket->cancelled() ||
            (hooks.cancelled && hooks.cancelled(l.view()));
    }
    void open(Lane& l) {
        if (hooks.opened) hooks.opened(l.view());
        if (hooks.initial_cursor) l.rng = hooks.initial_cursor(l.view());
        l.turn.reset();
        if (turn_caching() && hooks.turn_cache) {
            try { l.turn = hooks.turn_cache(l.view()); } catch (...) { ++turn_failed; }
        }
        if (l.turn && hooks.identity.ranks == store.ranks()) {
            auto hit = cache->lookup_turn(l.taken.job.prompt, hooks.identity, l.turn->metadata);
            if (!hit) ++misses;
            else {
                bool restored = false;
                try { restored = store.restore_prompt(l.handle, hit->state); }
                catch (...) { ++restore_failed; throw; }
                if (restored) {
                    require(frontier(l) == session::Frontier{hit->state.consumed, {}}, "turn restore frontier mismatch");
                    l.offset = hit->state.tokens.size(); l.stats.cached_tokens += l.offset; ++hits;
                } else ++misses;
            }
        }
        if (caching() && !l.offset) {
            auto hit = cache->lookup(l.taken.job.prompt, hooks.identity);
            if (!hit) ++misses;
            else {
                ++hits;
                const bool restored = hooks.prompt_restart_cache ? store.restore_prompt(l.handle,hit->state) :
                    (store.restore(l.handle,hit->state),true);
                if (restored) {
                    require(frontier(l) == session::Frontier{hit->state.consumed, {}}, "cache restore frontier mismatch");
                    l.offset = hit->state.tokens.size(); l.stats.cached_tokens += l.offset;
                }
            }
        }
        if(hooks.ready) hooks.ready(l.view());
    }

    void prompt_checkpoint(Lane& l, SessionCheckpoint boundary) {
        if (!caching() || !hooks.prompt_restart_cache || !policy.store || !policy.max_entries ||
            (!policy.max_bytes && !policy.retain_one_oversize) ||
            l.taken.job.cache_retention == CacheRetention::no_store || !hooks.prompt_checkpoint ||
            !hooks.prompt_checkpoint(l.view(), boundary)) return;
        auto snap = std::make_shared<session::PrefixSnapshot>();
        snap->identity = hooks.identity;
        snap->state = store.capture_prompt(l.handle);
        snap->attention_frontier = snap->recurrent_frontier = snap->state.consumed;
        (void)cache->admit(snap);
    }
    void progress(Lane& l) {
        if (l.taken.job.on_progress) l.taken.job.on_progress({
            l.offset == l.taken.job.prompt.size() ? GenerationPhase::decode : GenerationPhase::prefill,
            l.offset - std::min<std::uint64_t>(l.offset, l.stats.cached_tokens), l.stats.cached_tokens, l.taken.job.prompt.size()});
    }
    void checkpoint(Lane& l) {
        if (!caching() || !policy.store || !policy.max_entries || !policy.max_bytes ||
            l.taken.job.cache_retention == CacheRetention::no_store || !hooks.checkpoint ||
            l.stats.generated_tokens != 0 ||
            store.pending(l.handle) || !store.consumed(l.handle) || !hooks.checkpoint(l.view())) return;
        auto snap = std::make_shared<session::PrefixSnapshot>();
        snap->identity = hooks.identity;
        snap->state = store.capture(l.handle);
        snap->attention_frontier = snap->recurrent_frontier = snap->state.consumed;
        (void)cache->admit(snap); // policy may decline size/duplicates without failing generation
    }
    void terminal(Lane& l, FinishReason reason) {
        if(caps.mixed_advance && caps.lane_cancellation && reason==FinishReason::cancelled) engine.cancel_lane(l.handle);
        l.stats.finish = name(reason);
        prompt_checkpoint(l, SessionCheckpoint::terminal);
        if (reason != FinishReason::cancelled && !cancelled(l) && hooks.continuation && l.stats.generated_tokens < l.taken.job.max_tokens) {
            if (auto next = hooks.continuation(l.view(), reason)) {
                const auto remaining = l.taken.job.max_tokens - l.stats.generated_tokens;
                const bool restart = next->mode == SessionContinuation::Mode::restart;
                if (next->suffix.empty() || !next->allowance || next->allowance > remaining ||
                    next->suffix.size() > caps.max_context - (restart ? 0 : logical(frontier(l))))
                    throw std::invalid_argument("invalid session continuation");
                l.taken.job.stop_tokens = std::move(next->stop_tokens);
                l.phase_remaining = next->allowance;
                l.stats.prompt_tokens += next->suffix.size();
                if (restart) {
                    store.reset(l.handle);
                    l.taken.job.prompt = std::move(next->suffix); l.offset = 0;
                    open(l); progress(l);
                } else {
                    l.offset = l.taken.job.prompt.size();
                    l.taken.job.prompt.insert(l.taken.job.prompt.end(), next->suffix.begin(), next->suffix.end());
                }
                return;
            }
        }
        // Freeze the delivered reply receipt before optional physical sealing.
        std::string error = l.publication_error;
        receipt(l, error);
        if (error.empty()) completed_turn(l, reason);
        finish(l, std::move(error));
    }
    void validate_result(Lane& l, const session::SamplerState& s, const session::Frontier& before,
                         const session::StepResult& r) {
        require(r.before == before && r.after == frontier(l), "decode frontier mismatch");
        require(r.finish == FinishReason::none || r.finish == FinishReason::stop || r.finish == FinishReason::length ||
                r.finish == FinishReason::context || r.finish == FinishReason::cancelled, "invalid finish reason");
        require(r.rng_draws >= l.rng && r.tokens.size() <= s.max_tokens &&
                logical(r.after) <= caps.max_context, "decode allowance/cursor mismatch");
        require(r.accepted_drafts <= r.proposed_drafts && r.accepted_drafts <= r.tokens.size(), "invalid draft counts");
        require(!r.tokens.empty() || r.finish != FinishReason::none, "empty nonterminal decode");
        if (r.tokens.empty()) require(r.after == before && r.rng_draws == l.rng, "empty decode changed state");
        else require(logical(r.after) == logical(before) + r.tokens.size() &&
                     r.after.pending == r.tokens.back(), "decode commit mismatch");
        const auto stop = std::find_first_of(r.tokens.begin(), r.tokens.end(), s.stop_tokens.begin(), s.stop_tokens.end());
        require(stop == r.tokens.end() || (stop + 1 == r.tokens.end() && r.finish == FinishReason::stop), "post-stop decode output");
        require(r.finish != FinishReason::stop || stop != r.tokens.end(), "stop without stop ID");
        require(r.finish != FinishReason::length || r.tokens.size() == s.max_tokens, "premature length result");
        require(r.finish != FinishReason::context || logical(r.after) == caps.max_context, "premature context result");
        require(r.finish != FinishReason::cancelled || l.cancel.stop_requested(), "unsignalled cancellation result");
        require(r.finish != FinishReason::none ||
                (r.tokens.size() < s.max_tokens && logical(r.after) < caps.max_context), "missing terminal decode result");
    }
    void execute(const SessionWork& w) {
        const auto started = std::chrono::steady_clock::now();
        if (w.phase == SessionPhase::mixed) { execute_mixed(w); return; }
        if (w.phase == SessionPhase::prefill) {
            std::vector<session::PrefillItem> items;
            std::vector<session::Frontier> before;
            for (std::size_t n = 0; n < w.lanes.size(); ++n) {
                auto& l = *lanes[w.lanes[n]];
                before.push_back(frontier(l));
                require(w.chunks[n] <= caps.max_context - logical(before.back()), "prefill context overflow");
                items.push_back({std::span(l.taken.job.prompt).subspan(l.offset, w.chunks[n]), l.handle});
            }
            if (items.size() > 1) engine.prefill_batch(items);
            else engine.prefill(items[0].tokens, items[0].state);
            const auto ms = elapsed(started);
            for (std::size_t n = 0; n < w.lanes.size(); ++n) {
                auto& l = *lanes[w.lanes[n]];
                const auto after = frontier(l);
                require(logical(after) == logical(before[n]) + w.chunks[n] &&
                        (!after.pending || after.pending == items[n].tokens.back()), "prefill frontier mismatch");
                l.stats.prefill_ms += ms;
                l.offset += w.chunks[n];
                checkpoint(l);
                progress(l);
            }
        } else {
            std::vector<session::SamplerState> samplers;
            std::vector<session::Frontier> before;
            samplers.reserve(w.lanes.size());
            // Validate every lane before the one native batch call.
            for (auto i : w.lanes) {
                auto& l = *lanes[i];
                admission(l.taken.job, caps, hooks);
                before.push_back(frontier(l));
                const auto remaining = std::min<std::uint64_t>(l.phase_remaining,
                    l.taken.job.max_tokens - l.stats.generated_tokens);
                samplers.push_back({l.taken.job.sampling, l.rng,
                    static_cast<std::uint32_t>(std::min(remaining, caps.max_context - logical(before.back()))),
                    l.taken.job.stop_tokens, l.cancel.get_token(), l.taken.job.constraint});
            }
            std::vector<session::DecodeItem> items;
            for (std::size_t n = 0; n < w.lanes.size(); ++n) items.push_back({lanes[w.lanes[n]]->handle, samplers[n]});
            std::vector<session::StepResult> results;
            if (items.size() > 1) results = engine.decode_batch(items);
            else results.push_back(engine.decode(items[0].state, items[0].sampler));
            const auto ms = elapsed(started);
            require(results.size() == items.size(), "decode batch result count mismatch");
            // Validate every result before publication. A malformed adapter
            // result in one lane must not discard another lane's valid commit.
            for (std::size_t n = 0; n < items.size(); ++n) {
                auto& l = *lanes[w.lanes[n]];
                try { validate_result(l, samplers[n], before[n], results[n]); }
                catch (const std::exception& e) { fail(l, e.what()); }
                catch (...) { fail(l, "invalid session result"); }
            }
            for (std::size_t n = 0; n < items.size(); ++n) {
                auto& l = *lanes[w.lanes[n]];
                if (l.done) continue;
                const auto& r = results[n];
                l.stats.decode_ms += ms;
                if (!r.tokens.empty() && !l.stats.generated_tokens && l.taken.job.observation_complete())
                    l.stats.first_token_ms = elapsed(l.taken.submitted);
                l.rng = r.rng_draws; l.phase_remaining -= static_cast<std::uint32_t>(r.tokens.size());
                l.stats.generated_tokens += r.tokens.size();
                ++l.stats.cycles; l.stats.proposed += r.proposed_drafts; l.stats.accepted += r.accepted_drafts;
                // Account committed output before any user callback can throw.
                try {
                    const bool keep = r.tokens.empty() || !l.taken.job.on_tokens || l.taken.job.on_tokens(r.tokens);
                    if (!keep) { l.cancel.request_stop(); terminal(l, FinishReason::cancelled); }
                    else if (r.finish != FinishReason::none) terminal(l, r.finish);
                } catch (const std::exception& e) { fail(l, e.what()); }
                catch (...) { fail(l, "session output callback failed"); }
            }
        }
    }
    void execute_mixed(const SessionWork& w) {
        std::vector<session::SamplerState> samplers;
        std::vector<session::Frontier> before;
        samplers.reserve(w.lanes.size());
        for (auto i : w.lanes) {
            auto& l = *lanes[i]; admission(l.taken.job, caps, hooks);
            before.push_back(frontier(l));
            const auto remaining = std::min<std::uint64_t>(l.phase_remaining, l.taken.job.max_tokens-l.stats.generated_tokens);
            samplers.push_back({l.taken.job.sampling,l.rng,static_cast<std::uint32_t>(remaining),
                l.taken.job.stop_tokens,l.cancel.get_token(),l.taken.job.constraint});
        }
        std::vector<session::AdvanceItem> items;
        for (std::size_t n=0;n<w.lanes.size();++n) {
            auto& l=*lanes[w.lanes[n]];
            items.push_back({l.handle,std::span(l.taken.job.prompt).subspan(l.offset),samplers[n],
                [this, lane=&l] { const bool stop=cancelled(*lane); if(stop) lane->cancel.request_stop(); return stop; }});
        }
        auto results=engine.advance(items);
        require(results.size()==items.size(),"mixed result count mismatch");
        require(std::any_of(results.begin(),results.end(),[](const auto& r){return r.input_tokens || !r.output.tokens.empty() ||
            r.output.finish!=FinishReason::none || !r.error.empty();}),"mixed plan made no progress");
        for (std::size_t n=0;n<results.size();++n) {
            auto& l=*lanes[w.lanes[n]]; const auto& r=results[n]; const auto& o=r.output;
            try {
                require(r.input_tokens<=items[n].prompt.size() && o.before==before[n] &&
                    o.after==frontier(l) && o.tokens.size()<=samplers[n].max_tokens && o.rng_draws>=l.rng,
                    "mixed progress/frontier mismatch");
                if(r.error.empty()) {
                    require(logical(o.after)==logical(o.before)+r.input_tokens+o.tokens.size() &&
                        logical(o.after)<=caps.max_context && (o.tokens.empty() || o.after.pending==o.tokens.back()),
                        "mixed commit mismatch");
                    const auto stop=std::find_first_of(o.tokens.begin(),o.tokens.end(),samplers[n].stop_tokens.begin(),samplers[n].stop_tokens.end());
                    require(stop==o.tokens.end() || (stop+1==o.tokens.end() && o.finish==FinishReason::stop),"mixed post-stop output");
                    require(o.finish==FinishReason::none || o.finish==FinishReason::stop || o.finish==FinishReason::length ||
                        o.finish==FinishReason::context || o.finish==FinishReason::cancelled,"invalid mixed finish reason");
                    require(o.finish!=FinishReason::stop || stop!=o.tokens.end(),"mixed stop without stop ID");
                    require(o.finish!=FinishReason::length || o.tokens.size()==samplers[n].max_tokens,"premature mixed length");
                    require(o.finish!=FinishReason::context || logical(o.after)==caps.max_context,"premature mixed context");
                    require(o.finish!=FinishReason::cancelled || l.cancel.stop_requested(),"unsignalled mixed cancellation");
                    require(o.accepted_drafts<=o.proposed_drafts && o.accepted_drafts<=o.tokens.size(),"mixed draft counts");
                }
                l.offset+=r.input_tokens; l.rng=o.rng_draws;
                l.phase_remaining-=static_cast<std::uint32_t>(o.tokens.size());
                l.stats.generated_tokens+=o.tokens.size();
                if(!o.tokens.empty()) {++l.stats.cycles;l.stats.proposed+=o.proposed_drafts;l.stats.accepted+=o.accepted_drafts;}
                // Commit/account first. Even a lane error must not erase output
                // already published by the family before another lane failed.
                const bool keep=o.tokens.empty() || !l.taken.job.on_tokens || l.taken.job.on_tokens(o.tokens);
                progress(l);
                if(!r.error.empty()) {store.invalidate(l.handle);fail(l,r.error);continue;}
                prompt_checkpoint(l,SessionCheckpoint::progress);
                if(!keep) {l.cancel.request_stop();terminal(l,FinishReason::cancelled);}
                else if(o.finish!=FinishReason::none) terminal(l,o.finish);
            } catch(const std::exception& e) {fail(l,e.what());}
            catch(...) {fail(l,"mixed publication failed");}
        }
    }
    bool step(bool wait) {
        if (hooks.control_turn) {
            std::erase_if(lanes, [](const auto& l) { return l->done; });
            // An external owner stop closes native work/protocol first. The
            // shell retains responsibility for shutting down unreserved HTTP
            // work, preserving the queue's existing shutdown ordering.
            if ((queue.stopping() || (hooks.stop_requested && hooks.stop_requested())) &&
                (!hooks.drain_native_on_shutdown || hooks.native_drained())) { close(); return false; }
            control_wait = wait;
            hooks.control_turn(*owner);
            return true;
        }
        if (hooks.stop_requested && hooks.stop_requested()) queue.shutdown();
        std::erase_if(lanes, [](const auto& l) { return l->done; });
        if (lanes.empty() && !queue.stopping() && hooks.idle) hooks.idle();
        bool blocked = false;
        while (lanes.size() < caps.max_slots && !queue.stopping()) {
            if (hooks.can_admit) {
                std::vector<SessionView> admission_views;
                for (const auto& l : lanes) admission_views.push_back(l->view());
                if (!hooks.can_admit(admission_views)) { blocked = true; break; }
            }
            auto taken = queue.take(wait && lanes.empty() && !hooks.idle && !hooks.can_admit,
                hooks.selection, hooks.eligible,
                hooks.reorder, hooks.timed_eligible, hooks.overtake_blocked, &blocked);
            if (!taken) {
                break;
            }
            admit(std::move(*taken));
        }
        for (auto& l : lanes) if (!caps.mixed_advance && !l->done && cancelled(*l)) {
            l->cancel.request_stop(); terminal(*l, FinishReason::cancelled);
        }
        std::erase_if(lanes, [](const auto& l) { return l->done; });
        if (lanes.empty()) {
            if (wait && !queue.stopping() && (hooks.idle || hooks.can_admit))
                queue.wait_for_work_until(std::chrono::steady_clock::now() + hooks.idle_interval, !blocked);
            return !queue.stopping();
        }
        // Open prompts take the ids that became final; a lane that has used all of its ids waits for more, so no
        // round ever sees an open prompt as complete.
        bool grew = false;
        for (auto& l : lanes)
            if (!l->done && l->taken.job.prompt_tail) {
                grow_prompt(*l, l->offset == l->taken.job.prompt.size());
                grew = true;
            }
        if (grew) {
            std::erase_if(lanes, [](const auto& l) { return l->done; });
            if (lanes.empty()) return !queue.stopping();
        }
        std::vector<SessionView> views;
        for (auto& l : lanes) views.push_back(l->view());
        try {
            const auto work = hooks.schedule ? hooks.schedule(views) : basic_host_schedule(views, caps);
            require(!work.empty(), "session schedule made no progress");
            // A plan that would use every id an open prompt holds (or leave prefill) was made on a prompt that is
            // still growing: wait for more ids and plan again. A chunk strictly inside the held ids is final.
            bool replan = false;
            for (const auto& w : work)
                for (std::size_t n = 0; n < w.lanes.size(); ++n) {
                    if (w.lanes[n] >= lanes.size()) continue;
                    auto& l = *lanes[w.lanes[n]];
                    if (l.done || !l.taken.job.prompt_tail) continue;
                    if (w.phase == SessionPhase::prefill && n < w.chunks.size() &&
                        w.chunks[n] < l.taken.job.prompt.size() - l.offset) continue;
                    grow_prompt(l, true);
                    replan = true;
                }
            if (replan) return true;
            std::set<std::size_t> used;
            for (const auto& w : work) {
                require(w.phase == SessionPhase::prefill || w.phase == SessionPhase::decode || w.phase == SessionPhase::mixed, "invalid session phase");
                require(!w.lanes.empty() && w.lanes.size() <= caps.max_batch, "invalid session group size");
                require(w.phase == SessionPhase::mixed ? caps.mixed_advance : (w.lanes.size() == 1 || (w.phase == SessionPhase::prefill ? caps.batch_prefill : caps.batch_decode)), "unsupported native batch phase");
                require(w.phase == SessionPhase::prefill ? w.chunks.size() == w.lanes.size() : w.chunks.empty(), "invalid session chunks");
                for (std::size_t n = 0; n < w.lanes.size(); ++n) {
                    const auto i = w.lanes[n];
                    require(i < lanes.size() && used.insert(i).second, "invalid or duplicate session lane");
                    const auto& l = *lanes[i];
                    require(w.phase == SessionPhase::mixed || (w.phase == SessionPhase::prefill ? w.chunks[n] && w.chunks[n] <= l.taken.job.prompt.size() - l.offset :
                            l.offset == l.taken.job.prompt.size()), "invalid session phase frontier");
                }
            }
            for (const auto& w : work) {
                // Decode marks the byte budget: background host work is capped while sessions decode.
                std::optional<::strix::core::ByteBudget::DecodeScope> decoding;
                if (w.phase != SessionPhase::prefill) decoding.emplace(::strix::core::shared_byte_budget());
                try { execute(w); }
                catch (const std::exception& e) { for (auto i : w.lanes) if (!lanes[i]->done) fail(*lanes[i], e.what()); }
                catch (...) { for (auto i : w.lanes) if (!lanes[i]->done) fail(*lanes[i], "session phase failed"); }
            }
        } catch (const std::exception& e) { for (auto& l : lanes) if (!l->done) fail(*l, e.what()); }
        catch (...) { for (auto& l : lanes) if (!l->done) fail(*l, "session schedule failed"); }
        std::erase_if(lanes, [](const auto& l) { return l->done; });
        return !queue.stopping() || !lanes.empty();
    }
};

SessionRuntime::SessionRuntime(GenerationQueue& q, session::Engine& e, session::KvStore& s, session::Policy p, SessionHooks h, std::shared_ptr<PrefixStateCache> pooled)
    : impl_(std::make_unique<Impl>(q, e, s, p, std::move(h), std::move(pooled))) { impl_->owner = this; }
SessionRuntime::~SessionRuntime() {
    if (!impl_->closed) shutdown();
    try { impl_->close(); } catch (...) {}
    for (auto& l : impl_->lanes) if (!l->done) {
        if(impl_->caps.mixed_advance && impl_->caps.lane_cancellation) {
            try {impl_->engine.cancel_lane(l->handle);} catch(...) {}
        }
        l->stats.finish = "cancelled"; impl_->finish(*l, "server shutting down");
    }
}
void SessionRuntime::run() {
    try { while (step(true)) {} impl_->close(); }
    catch (...) {
        if (impl_->hooks.serial_queue_protocol) {
            // generate() failures are request-local. Only idle/observer or owner
            // failures escape; preserve run()'s caller-owned queue lifetime.
            impl_->close();
            throw;
        }
        try { throw; }
        catch (const std::exception& e) {
            impl_->owner_error = e.what();
            if (impl_->hooks.drain_native_on_shutdown && impl_->owner_error.empty()) impl_->owner_error = "engine failure";
        }
        catch (...) { impl_->owner_error = impl_->hooks.drain_native_on_shutdown ? "engine failure" : "session owner failed"; }
        for (auto& l : impl_->lanes) if (!l->done && l->publication_error.empty()) l->publication_error = impl_->owner_error;
        if (!impl_->hooks.drain_native_on_shutdown) shutdown();
        try { if (impl_->hooks.failed) impl_->hooks.failed(); } catch (...) {}
        try { impl_->close(); } catch (...) {}
        for (auto& l : impl_->lanes) if (!l->done) impl_->fail(*l, "session owner failed");
        try { impl_->notify_stopped(); } catch (...) {}
        if (impl_->hooks.drain_native_on_shutdown) shutdown();
        throw;
    }
    impl_->notify_stopped();
}
bool SessionRuntime::step(bool wait) try { return impl_->step(wait); }
catch (...) { invalidate_residents(); throw; }
void SessionRuntime::shutdown() noexcept { impl_->queue.shutdown(); }
std::size_t SessionRuntime::live() const noexcept {
    return static_cast<std::size_t>(std::count_if(impl_->lanes.begin(), impl_->lanes.end(), [](const auto& l) { return !l->done; }));
}
const TakenJob* SessionRuntime::reservation() const noexcept { return impl_->reserved ? &*impl_->reserved : nullptr; }
bool SessionRuntime::reserve(QueueSelection selection,
    const std::function<bool(std::chrono::steady_clock::time_point)>& reorder,
    const std::function<bool(const GenerationJob&, std::chrono::steady_clock::time_point)>& eligible,
    const std::function<void()>& blocked) {
    if (!impl_->hooks.control_turn) throw std::logic_error("native reservation requires a control planner");
    if (!impl_->reserved && !stopping())
        impl_->reserved = impl_->queue.take(false, selection, {}, reorder, eligible, blocked);
    if (impl_->reserved) Impl::settle_prompt(impl_->reserved->job); // native reservations plan on whole prompts
    return impl_->reserved.has_value();
}
bool SessionRuntime::reserve_fifo(bool wait, bool short_first) {
    if (!impl_->hooks.control_turn) throw std::logic_error("native reservation requires a control planner");
    if (!impl_->reserved && !stopping()) impl_->reserved = impl_->queue.take(wait && impl_->control_wait, short_first);
    if (impl_->reserved) Impl::settle_prompt(impl_->reserved->job); // native reservations plan on whole prompts
    return impl_->reserved.has_value();
}
bool SessionRuntime::reserve_serial(bool wait) {
    if (!impl_->hooks.control_turn || !impl_->hooks.serial_queue_protocol)
        throw std::logic_error("serial reservation requires the serial queue protocol");
    if (!impl_->reserved) impl_->reserved = impl_->queue.take_serial(wait && impl_->control_wait);
    if (impl_->reserved) Impl::settle_prompt(impl_->reserved->job); // native reservations plan on whole prompts
    return impl_->reserved.has_value();
}
void SessionRuntime::refuse(GenerationStats stats, std::string error) {
    if (!impl_->reserved) throw std::logic_error("no reserved session");
    impl_->queue.finish(*impl_->reserved, std::move(stats), std::move(error));
    impl_->reserved.reset();
}
SessionLease SessionRuntime::activate(std::uint64_t resident_alignment, bool preserve_take_time) {
    if (!impl_->reserved) throw std::logic_error("no reserved session");
    if (live() >= impl_->caps.max_slots) throw std::runtime_error("session slots exhausted");
    auto l = std::make_unique<Impl::Lane>();
    // Allocate owner metadata before acquiring a physical lease or moving work.
    impl_->lanes.reserve(impl_->lanes.size() + 1);
    if (impl_->hooks.native_validate) impl_->hooks.native_validate(impl_->reserved->job);
    else admission(impl_->reserved->job, impl_->caps, impl_->hooks);
    if (impl_->reserved->job.family_context && !impl_->hooks.validate)
        throw std::invalid_argument("unvalidated family context");
    if (impl_->hooks.validate) impl_->hooks.validate(impl_->reserved->job);
    if (impl_->hooks.native_validate) embed_admission(impl_->reserved->job, impl_->caps); // admission() ran it otherwise
    const auto started = impl_->hooks.now ? impl_->hooks.now() : std::chrono::steady_clock::now();
    if (resident_alignment && !impl_->cache) throw std::invalid_argument("native retention disabled");
    l->resident_identity = impl_->hooks.resident_identity ?
        impl_->hooks.resident_identity(impl_->reserved->job) : impl_->hooks.identity;
    const auto hit = resident_alignment ? impl_->cache->lookup_resident(impl_->store,
        impl_->reserved->job.prompt, l->resident_identity, resident_alignment) : session::ResidentHit{};
    if (hit.entry) {
        // Remove the descriptor BEFORE a new lease or native mutation exists.
        impl_->cache->invalidate_resident(hit.entry->source);
        l->handle = impl_->store.acquire_resident(hit.entry->source, hit.entry->tokens().first(static_cast<std::size_t>(hit.rows)));
        l->stats.cached_tokens = hit.rows; ++impl_->hits;
        ++impl_->resident_hits; impl_->resident_hit_rows += hit.rows;
        l->resident_reply_hit = hit.rows > hit.entry->tokens().size() - hit.entry->reply_rows;
        if (l->resident_reply_hit) ++impl_->resident_reply_hits;
    } else {
        l->handle = impl_->store.acquire();
        if (impl_->cache) impl_->cache->invalidate_resident(l->handle);
        if (resident_alignment) { ++impl_->misses; ++impl_->resident_misses; }
    }
    // The reservation stays whole until the lane is certain: a refused table gives the lease back and the
    // caller still owns a reservation it can refuse().
    if (impl_->reserved->job.embeds) {
        try { impl_->engine.bind_embeds(l->handle, impl_->reserved->job.embeds); }
        catch (...) { drop_unused(impl_->store, l->handle); throw; }
    }
    l->taken = std::move(*impl_->reserved); impl_->reserved.reset();
    if (!preserve_take_time) l->taken.started = started;
    l->stats.prompt_tokens = l->taken.job.prompt.size();
    SessionLease result{l->taken, l->stats, l->handle, l->resident_reply_hit};
    impl_->lanes.push_back(std::move(l));
    return result;
}
PendingSession SessionRuntime::hold_reservation() {
    if (!impl_->hooks.control_turn || !impl_->hooks.disable_native_cache || impl_->hooks.serial_queue_protocol)
        throw std::logic_error("pending requests require request-only native ownership");
    if (!impl_->reserved) throw std::logic_error("no reserved session");
    if (!impl_->pending_owner || impl_->pending_generation == UINT64_MAX)
        throw std::overflow_error("pending session generation exhausted");
    auto l = std::make_unique<Impl::Lane>();
    const auto generation = impl_->pending_generation + 1;
    auto [it, inserted] = impl_->pending.emplace(generation, std::move(l));
    if (!inserted) throw std::logic_error("pending session generation collision");
    it->second->taken = std::move(*impl_->reserved);
    impl_->reserved.reset();
    impl_->pending_generation = generation;
    return {generation, impl_->pending_owner};
}
const TakenJob& SessionRuntime::inspect(PendingSession h) const {
    if (h.owner != impl_->pending_owner || !h.valid()) throw std::invalid_argument("foreign pending session");
    const auto it = impl_->pending.find(h.generation);
    if (it == impl_->pending.end()) throw std::invalid_argument("stale pending session");
    return it->second->taken;
}
const TakenJob& SessionRuntime::inspect(session::KvHandle h) const {
    for (const auto& l : impl_->lanes) if (!l->done && l->handle == h) {
        (void)impl_->store.consumed(h);
        return l->taken;
    }
    throw std::invalid_argument("stale native request inspection");
}
bool SessionRuntime::pending_cancelled(PendingSession h) const noexcept {
    try { return inspect(h).ticket->cancelled(); } catch (...) { return true; }
}
std::size_t SessionRuntime::pending_count() const noexcept { return impl_->pending.size(); }
SessionLease SessionRuntime::promote(PendingSession h) {
    const auto& taken = inspect(h); // validate provenance before any hooks
    if (live() >= impl_->caps.max_slots) throw std::runtime_error("session slots exhausted");
    impl_->lanes.reserve(impl_->lanes.size() + 1);
    if (impl_->hooks.native_validate) impl_->hooks.native_validate(taken.job);
    else admission(taken.job, impl_->caps, impl_->hooks);
    if (taken.job.family_context && !impl_->hooks.validate) throw std::invalid_argument("unvalidated family context");
    if (impl_->hooks.validate) impl_->hooks.validate(taken.job);
    if (impl_->hooks.native_validate) embed_admission(taken.job, impl_->caps); // admission() ran it otherwise
    auto& l = *impl_->pending.at(h.generation);
    l.handle = impl_->store.acquire();
    if (taken.job.embeds) {
        // A refused table gives the lease back; the entry stays pending and can be promoted again or refused.
        try { impl_->engine.bind_embeds(l.handle, taken.job.embeds); }
        catch (...) { drop_unused(impl_->store, l.handle); l.handle = {}; throw; }
    }
    l.stats.prompt_tokens = taken.job.prompt.size();
    // Preserve original queue submission/take clocks, reorder and observation.
    SessionLease result{l.taken, l.stats, l.handle};
    impl_->lanes.push_back(std::move(impl_->pending.at(h.generation)));
    impl_->pending.erase(h.generation);
    return result;
}
void SessionRuntime::refuse_pending(PendingSession h, GenerationStats stats, std::string error) {
    (void)inspect(h); // reject stale/foreign before accounting or callbacks
    auto l = std::move(impl_->pending.at(h.generation));
    impl_->pending.erase(h.generation);
    impl_->queue.finish(l->taken, std::move(stats), std::move(error));
}
bool SessionRuntime::has_waiting(const std::function<bool(const GenerationJob&, std::chrono::steady_clock::time_point)>& f) const {
    return impl_->queue.has_waiting(f);
}
bool SessionRuntime::publish(session::KvHandle h, std::span<const std::uint32_t> ids, bool deliver, bool propagate_callback_error) {
    for (auto& l : impl_->lanes) if (!l->done && l->handle == h) {
        (void)impl_->store.consumed(h); // validate generation even at publication
        l->stats.generated_tokens += ids.size();
        if (!deliver) return false;
        if (propagate_callback_error) return !l->taken.job.on_tokens || l->taken.job.on_tokens(ids);
        try { return !l->taken.job.on_tokens || l->taken.job.on_tokens(ids); }
        catch (const std::exception& e) { l->publication_error = e.what(); }
        catch (...) { l->publication_error = "session output callback failed"; }
        return false;
    }
    throw std::invalid_argument("stale session publication");
}
void SessionRuntime::invalidate_residents() noexcept {
    if (impl_->cache) impl_->cache->invalidate_residents();
    impl_->store.invalidate_residents();
}
bool SessionRuntime::ticket_cancelled(session::KvHandle h) const noexcept {
    for (const auto& l : impl_->lanes) if (!l->done && l->handle == h) {
        try { (void)impl_->store.consumed(h); }
        catch (...) { return true; }
        return l->taken.ticket->cancelled();
    }
    return true;
}
void SessionRuntime::notify_admitted(session::KvHandle h) {
    if (!impl_->hooks.control_turn) throw std::logic_error("native admission event requires a control planner");
    for (const auto& l : impl_->lanes) if (!l->done && l->handle == h) {
        (void)impl_->store.consumed(h);
        if (l->taken.job.on_admitted) l->taken.job.on_admitted();
        return;
    }
    throw std::invalid_argument("stale native admission event");
}
void SessionRuntime::publish_progress(session::KvHandle h, GenerationProgress progress) {
    if (!impl_->hooks.control_turn) throw std::logic_error("native progress requires a control planner");
    for (const auto& l : impl_->lanes) if (!l->done && l->handle == h) {
        (void)impl_->store.consumed(h);
        if (l->taken.job.on_progress) l->taken.job.on_progress(progress);
        return;
    }
    throw std::invalid_argument("stale native progress");
}
void SessionRuntime::abandon_resident(session::KvHandle h) {
    for (auto& l : impl_->lanes) if (!l->done && l->handle == h) {
        impl_->store.reset(h);
        if (impl_->hits) { --impl_->hits; ++impl_->misses; }
        if (impl_->resident_hits) {
            --impl_->resident_hits; ++impl_->resident_misses; impl_->resident_hit_rows -= l->stats.cached_tokens;
            if (l->resident_reply_hit) --impl_->resident_reply_hits;
        }
        l->stats.cached_tokens = 0; l->resident_reply_hit = false;
        return;
    }
    throw std::invalid_argument("stale resident abandonment");
}
void SessionRuntime::complete(session::KvHandle h, std::string error, const ResidentClaim& claim) {
    for (auto& l : impl_->lanes) if (!l->done && l->handle == h) { impl_->finish(*l, std::move(error), 0, &claim); return; }
    throw std::invalid_argument("stale session completion");
}
void SessionRuntime::complete(session::KvHandle h, std::string error, std::uint64_t rows) {
    for (auto& l : impl_->lanes) if (!l->done && l->handle == h) { impl_->finish(*l, std::move(error), rows); return; }
    throw std::invalid_argument("stale session completion");
}
bool SessionRuntime::stopping() const { return impl_->queue.stopping() || (impl_->hooks.stop_requested && impl_->hooks.stop_requested()); }
void SessionRuntime::complete_native(session::KvHandle h, GenerationStats stats, std::string error) {
    if (!impl_->hooks.control_turn) throw std::logic_error("native completion requires a control planner");
    for (auto& l : impl_->lanes) if (!l->done && l->handle == h) {
        (void)impl_->store.consumed(h);
        l->stats = std::move(stats);
        impl_->finish(*l, std::move(error));
        return;
    }
    throw std::invalid_argument("stale native completion");
}
void SessionRuntime::wait_for(std::chrono::milliseconds duration, bool generation_ready) {
    if (impl_->hooks.wait) impl_->hooks.wait(duration);
    else impl_->queue.wait_for_work_until(std::chrono::steady_clock::now() + duration, generation_ready);
}
const std::shared_ptr<PrefixStateCache>& SessionRuntime::cache_owner() const noexcept { static const std::shared_ptr<PrefixStateCache> none; return impl_->cache ? impl_->cache->owner() : none; }
std::size_t SessionRuntime::cache_entries() const noexcept { return impl_->cache ? impl_->cache->size() : 0; }
SessionCacheStats SessionRuntime::cache_stats() const noexcept { return {impl_->cache ? impl_->cache->size() : 0,impl_->cache ? impl_->cache->bytes() : 0,impl_->hits,impl_->misses,
    impl_->turn_admitted,impl_->turn_refused,impl_->turn_failed,impl_->restore_failed,
    impl_->resident_hits,impl_->resident_misses,impl_->resident_hit_rows,impl_->resident_reply_hits,
    impl_->resident_kept,impl_->resident_reply_kept,impl_->resident_claim_refused}; }
bool SessionRuntime::unpin_turn_cache(const std::string& keep) { return impl_->cache && impl_->cache->unpin(keep); }
session::Capabilities SessionRuntime::capabilities() const { return impl_->caps; }
void SessionRuntime::configure_api(ApiConfig& c) const {
    const auto caps = capabilities();
    c.sampling_policy = "reject"; c.engine_progress = true;
    c.engine_constraints = caps.constraints; c.engine_thinking_budget = caps.thinking_budget;
    c.engine_logprobs = caps.logprobs; c.engine_logit_transforms = caps.logit_transforms;
    const auto previous = c.capabilities_extra;
    c.capabilities_extra = [caps, previous](Json& j) {
        if (previous) previous(j);
        j.set("session_sampling", Json(Json::Object{{"temperature", caps.temperature}, {"top_p", caps.top_p},
            {"top_k", caps.top_k}, {"min_p", caps.min_p}, {"seed", caps.seed}}));
    };
}
} // namespace strix::serve
