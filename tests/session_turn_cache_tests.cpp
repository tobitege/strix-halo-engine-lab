#include "session_replay/legacy_reference.hpp"

#include <array>
#include <thread>
#include <limits>

using namespace session_fixture;
namespace {
struct Rank {
    std::vector<std::uint32_t> attention, recurrent, conv, history, draft;
    std::uint64_t seed = 0, draws = 0, outputs = 0, selection = 0;
    bool terminal = false, cancelled = false, failed = false, flight = false;
    bool operator==(const Rank&) const = default;
    std::uint64_t bytes() const {
        return sizeof(Rank) + sizeof(std::uint32_t) * (attention.size() + recurrent.size() +
            conv.size() + history.size() + draft.size());
    }
};
// Real mutable slot/rank storage. This fixture executes causal state transitions;
// it is not a model producer or an accelerator/distributed qualification.
struct TurnEngine final : ss::Engine {
    std::array<std::array<Rank, 2>, 3> slots;
    ss::Capabilities limits;
    std::vector<std::string> events;
    std::vector<std::vector<std::uint32_t>> suffixes;
    unsigned seals = 0, consumed_anchors = 0, captures = 0, restores = 0, rollbacks = 0;
    bool fail_consume = false, fail_capture = false, fail_restore = false;
    bool fail_rollback = false, fail_release = false, decline_restore = false;
    bool require_prepared = false;
    struct Candidate { ss::KvHandle handle; std::vector<std::uint32_t> tokens; };
    std::array<std::optional<Candidate>, 3> candidates;
    unsigned preparations = 0;
    std::array<Rank, 2> restored_state;
    std::vector<std::uint32_t> sealed_tokens, selected_ids;
    ss::KvStore store;
    static void row(Rank& p, std::uint32_t id, unsigned rank) {
        p.attention.push_back(id);
        const auto last = p.recurrent.empty() ? rank : p.recurrent.back();
        p.recurrent.push_back(last * 3U + id);
        p.conv.push_back(id ^ last); p.history.push_back(id);
        p.draft.push_back(id + last);
    }
    ss::Hooks hooks() {
        ss::Hooks h;
        h.initialize = [this](ss::KvHandle k) { slots.at(k.slot) = {}; candidates.at(k.slot).reset(); };
        h.reset = h.initialize;
        h.release = [this](ss::KvHandle k) {
            events.push_back("release");
            if (fail_release) throw std::runtime_error("release fault");
            slots.at(k.slot) = {};
        };
        h.capture = [this](ss::KvHandle k, std::uint32_t r) {
            const auto& p = slots.at(k.slot).at(r);
            return ss::RankState{std::make_shared<const Rank>(p), p.bytes()};
        };
        h.view = h.capture;
        h.restore = [this](ss::KvHandle k, std::uint32_t r, const ss::RankState& state) {
            slots.at(k.slot).at(r) = *std::static_pointer_cast<const Rank>(state.owner);
        };
        h.truncate = [](ss::KvHandle, std::uint32_t, std::uint64_t) {
            throw std::invalid_argument("hybrid truncation unsupported");
        };
        h.capture_prompt = [this](ss::KvHandle k, std::uint32_t r) {
            events.push_back("capture"); ++captures;
            CHECK(!store.pending(k));
            CHECK(store.tokens(k) == slots.at(k.slot).at(r).attention);
            if (require_prepared && !candidate_valid(k)) throw std::invalid_argument("unprepared or stale candidate");
            if (fail_capture && r == 1) throw std::runtime_error("capture fault");
            auto p = slots.at(k.slot).at(r);
            p.seed = p.draws = p.outputs = p.selection = 0;
            p.terminal = p.cancelled = p.failed = p.flight = false;
            return ss::RankState{std::make_shared<const Rank>(p), p.bytes()};
        };
        h.restore_prompt = [this](ss::KvHandle k, std::span<const ss::RankState> states) {
            ++restores; events.push_back("restore");
            if (decline_restore) return false;
            const auto before = slots.at(k.slot);
            try {
                for (unsigned r = 0; r < 2; ++r) {
                    auto restored = *std::static_pointer_cast<const Rank>(states[r].owner);
                    // NEW job seed/policy/cursors, not the captured request RNG.
                    restored.seed = before[r].seed;
                    slots.at(k.slot).at(r) = std::move(restored);
                    if (fail_restore && r == 1) throw std::runtime_error("restore rank fault");
                }
            } catch (...) {
                // Includes the throwing rank. KvStore additionally poisons this
                // failed prompt destination even after successful rollback.
                if (!fail_rollback) { slots.at(k.slot) = before; rollbacks += 2; }
                throw;
            }
            restored_state = slots.at(k.slot);
            return true;
        };
        return h;
    }
    TurnEngine() : store(3, 2, hooks()) {
        limits.max_context = 512; limits.max_slots = 3; limits.max_batch = 3;
        limits.turn_commit = true; limits.temperature = true; limits.seed = true;
        // Intentionally prefix_cache=false: independent complete-turn route.
    }
    ss::Capabilities caps() const override { return limits; }
    strix::profile::ExecutionReceipt receipt() const override { return {}; }
    void validate(ss::KvHandle k) {
        (void)store.consumed(k);
        for (const auto& p : slots.at(k.slot))
            if (p.cancelled || p.failed || p.flight) throw std::invalid_argument("unsealable lane");
    }
    void consume(ss::KvHandle k, std::span<const std::uint32_t> ids, bool sealing = false) {
        store.reserve_consumed(k, ids.size());
        const auto before = slots.at(k.slot);
        try {
            for (unsigned r = 0; r < 2; ++r) {
                for (auto id : ids) row(slots.at(k.slot).at(r), id, r);
                if (sealing && fail_consume && r == 1) throw std::runtime_error("consume rank fault");
            }
            store.append_consumed(k, ids);
        } catch (...) {
            if (fail_rollback) store.invalidate(k);
            else { slots.at(k.slot) = before; rollbacks += 2; }
            throw;
        }
    }
    void anchor(ss::KvHandle k, bool sealing = false) {
        if (const auto p = store.pending(k)) {
            const std::array<std::uint32_t, 1> ids{*p}; consume(k, ids, sealing);
            if (sealing) ++consumed_anchors;
        }
    }
    void commit_pending(ss::KvHandle& k) override {
        if (!limits.turn_commit) return ss::Engine::commit_pending(k);
        validate(k);
        for (const auto& rank : slots.at(k.slot))
            if (!rank.terminal) throw std::invalid_argument("turn is not complete");
        ++seals; events.push_back("seal");
        candidates.at(k.slot).reset();
        anchor(k, true); sealed_tokens = store.tokens(k);
    }
    bool candidate_valid(ss::KvHandle k) const {
        const auto& c = candidates.at(k.slot);
        return c && c->handle == k && c->tokens == store.tokens(k) && !store.pending(k);
    }
    bool prepare(ss::KvHandle k) {
        validate(k); CHECK(!store.pending(k));
        for (const auto& p : slots.at(k.slot)) CHECK(p.terminal);
        ++preparations; events.push_back("turn_snapshot");
        candidates.at(k.slot) = Candidate{k, store.tokens(k)};
        return true;
    }
    void prefill(std::span<const std::uint32_t> ids, ss::KvHandle& k) override {
        if (ids.empty()) return;
        validate(k); for (const auto& p : slots.at(k.slot)) CHECK(!p.terminal);
        suffixes.emplace_back(ids.begin(), ids.end()); anchor(k); consume(k, ids);
    }
    ss::StepResult decode(ss::KvHandle& k, const ss::SamplerState& sampler) override {
        validate(k);
        auto& all = slots.at(k.slot); auto& p = all[0];
        CHECK(!p.terminal && p.draws == sampler.rng_draws);
        ss::StepResult out; out.before = {store.consumed(k), store.pending(k)};
        if (!sampler.max_tokens) out.finish = ss::FinishReason::length;
        else if (store.consumed(k) + store.pending(k).has_value() == limits.max_context) out.finish = ss::FinishReason::context;
        else {
            anchor(k);
            // Canonical causal state + NEW request seed/cursor determine output.
            const auto id = static_cast<std::uint32_t>(32 + ((p.recurrent.back() + p.seed + p.draws) % 80));
            for (auto& rank : all) {
                ++rank.outputs; ++rank.selection;
                if (sampler.parameters.temperature > 0) ++rank.draws;
            }
            store.set_pending(k, id); out.tokens.push_back(id); selected_ids.push_back(id);
            if (std::find(sampler.stop_tokens.begin(), sampler.stop_tokens.end(), id) != sampler.stop_tokens.end()) out.finish = ss::FinishReason::stop;
            else if (store.consumed(k) + 1 == limits.max_context) out.finish = ss::FinishReason::context;
            else if (sampler.max_tokens == 1) out.finish = ss::FinishReason::length;
        }
        if (out.finish != ss::FinishReason::none) for (auto& r : all) r.terminal = true;
        out.after = {store.consumed(k), store.pending(k)}; out.rng_draws = p.draws;
        return out;
    }
};
ss::Policy turn_policy(std::uint64_t bytes = 100000) {
    ss::Policy p; p.completed_turns = true; p.max_bytes = bytes; return p;
}
sv::SessionHooks turn_hooks(TurnEngine& e, std::string conversation = "conversation") {
    sv::SessionHooks h;
    h.identity = {"asset-digest", "arithmetic-digest", "schema-v1", 2};
    h.opened = [&e](const sv::SessionView& v) {
        for (auto& rank : e.slots.at(v.state.slot)) rank.seed = v.job.sampling.seed;
    };
    h.turn_cache = [conversation = std::move(conversation)](const sv::SessionView&) {
        return std::optional<sv::SessionTurnCache>{{{conversation, "render-v1", "ordered-pair", ""}, 20000}};
    };
    h.turn_checkpoint = [&e](const sv::SessionView&, ss::FinishReason) { e.events.push_back("admit-final"); return true; };
    h.receipt = [&e](const sv::SessionView& v, sv::GenerationStats& s) {
        e.events.push_back("receipt");
        s.engine_extra.emplace_back("frontier", std::to_string(e.store.consumed(v.state)));
    };
    return h;
}
sv::GenerationJob request(std::vector<std::uint32_t> prompt = {10, 11}, unsigned count = 2, unsigned seed = 3) {
    auto j = job(std::move(prompt), count); j.stop_tokens.clear();
    j.sampling.seeded = true; j.sampling.seed = seed; j.sampling.temperature = 1; return j;
}
template<class F> void rejects(F f) {
    bool threw = false; try { f(); } catch (const std::exception&) { threw = true; } CHECK(threw);
}
std::shared_ptr<ss::PrefixSnapshot> snapshot(TurnEngine& e, std::vector<std::uint32_t> ids,
    std::string conversation, std::string keep = "") {
    auto k = e.store.acquire(); e.prefill(ids, k);
    ss::SamplerState zero; zero.max_tokens = 0;
    CHECK(e.decode(k, zero).finish == ss::FinishReason::length); e.commit_pending(k);
    auto s = std::make_shared<ss::PrefixSnapshot>();
    s->identity = turn_hooks(e).identity; s->turn = {conversation, "render-v1", "ordered-pair", keep};
    s->state = e.store.capture_prompt(k);
    s->attention_frontier = s->recurrent_frontier = s->state.consumed;
    e.store.release(k); return s;
}
void direct_contract() {
    TurnEngine e; auto k = e.store.acquire();
    rejects([&] { e.ss::Engine::commit_pending(k); });
    e.limits.turn_commit = false; rejects([&] { e.commit_pending(k); });
    e.limits.turn_commit = true; e.prefill(std::array{10U, 11U}, k);
    rejects([&] { e.commit_pending(k); });
    ss::SamplerState sampler; sampler.max_tokens = 1; sampler.parameters.temperature = 1;
    const auto result = e.decode(k, sampler); const auto before = e.slots[k.slot];
    e.prefill({}, k); CHECK(e.store.pending(k) == result.tokens.back());
    e.commit_pending(k); CHECK(!e.store.pending(k)); CHECK(e.store.consumed(k) == 3);
    CHECK(e.consumed_anchors == 1 && e.selected_ids == result.tokens);
    for (unsigned r = 0; r < 2; ++r) {
        CHECK(e.slots[k.slot][r].attention.back() == result.tokens.back());
        CHECK(e.slots[k.slot][r].draws == before[r].draws);
        CHECK(e.slots[k.slot][r].selection == before[r].selection);
        CHECK(e.slots[k.slot][r].outputs == before[r].outputs && e.slots[k.slot][r].terminal);
    }
    const auto sealed = e.slots[k.slot]; e.commit_pending(k);
    CHECK(e.slots[k.slot] == sealed && e.consumed_anchors == 1);
    for (auto flag : {&Rank::cancelled, &Rank::failed, &Rank::flight}) {
        e.slots[k.slot][0].*flag = true; rejects([&] { e.commit_pending(k); });
        e.slots[k.slot][0].*flag = false;
    }
    auto foreign = k; ++foreign.store; rejects([&] { e.commit_pending(foreign); });
    e.store.release(k); rejects([&] { e.commit_pending(k); });
    k = e.store.acquire(); e.store.invalidate(k); rejects([&] { e.commit_pending(k); }); e.store.release(k);

    TurnEngine broken; k = broken.store.acquire(); broken.prefill(std::array{1U}, k);
    auto output = broken.decode(k, sampler); const auto original = broken.slots[k.slot];
    broken.fail_consume = true; rejects([&] { broken.commit_pending(k); });
    CHECK(broken.slots[k.slot] == original && broken.rollbacks == 2);
    CHECK(broken.store.pending(k) == output.tokens.back() && broken.store.consumed(k) == 1);
    broken.fail_rollback = true; broken.fail_release = true;
    rejects([&] { broken.commit_pending(k); }); rejects([&] { (void)broken.store.tokens(k); });
    rejects([&] { broken.store.release(k); });
    auto other = broken.store.acquire(); CHECK(other.slot != k.slot);
    broken.fail_release = false; broken.store.release(other);
}
void cache_policy() {
    TurnEngine e; auto a = snapshot(e, {1, 2}, "a");
    auto b = snapshot(e, {3, 4}, "b"); auto c = snapshot(e, {5, 6}, "c");
    ss::PrefixCache measure(turn_policy()); CHECK(measure.admit(a)); const auto cost = measure.bytes();
    CHECK(cost > a->state.bytes + a->state.tokens.size() * 4);
    auto p = turn_policy(cost * 2); p.max_entries = 2;
    // This fixture intentionally holds the source captures across several
    // independent caches. Give those outstanding physical leases service credit
    // while preserving the exact two-entry/byte retention quota under test.
    ss::PrefixCache cache(p, ss::make_cache_owner(cost * 10)); CHECK(cache.admit(a) && cache.admit(b));
    CHECK(cache.lookup_turn(std::array{1U, 2U, 9U}, a->identity, *a->turn));
    CHECK(cache.admit(c)); CHECK(!cache.lookup_turn(b->state.tokens, b->identity, *b->turn));
    CHECK(cache.lookup_turn(a->state.tokens, a->identity, *a->turn));
    CHECK(!cache.lookup_turn(std::array{1U, 9U}, a->identity, *a->turn)); // no hybrid truncation
    CHECK(!cache.lookup(a->state.tokens, a->identity)); // cannot bypass template checks
    for (unsigned field = 0; field < 4; ++field) {
        auto id = a->identity;
        if (field == 0) id.model = "other"; if (field == 1) id.profile = "other";
        if (field == 2) id.state_format = "other"; if (field == 3) id.ranks = 1;
        CHECK(!cache.lookup_turn(a->state.tokens, id, *a->turn));
    }
    for (unsigned field = 0; field < 3; ++field) {
        auto meta = *a->turn;
        if (field == 0) meta.template_id = "other-render"; if (field == 1) meta.topology = "other";
        if (field == 2) meta.conversation = "other";
        CHECK(!cache.lookup_turn(a->state.tokens, a->identity, meta));
    }
    auto replacement = snapshot(e, {1, 2, 7}, "a"); CHECK(cache.admit(replacement));
    CHECK(cache.size() <= 2 && !cache.lookup_turn(a->state.tokens, a->identity, *a->turn));
    auto bad = std::make_shared<ss::PrefixSnapshot>(*replacement); bad->state.tokens[0] = 99;
    const auto old_bytes = cache.bytes(); CHECK(!cache.admit(bad) && cache.bytes() == old_bytes);
    auto kept = snapshot(e, {1, 2}, "a", "bookmark");
    ss::PrefixCache sizing(turn_policy()); CHECK(sizing.admit(kept));
    auto strict = turn_policy(sizing.bytes()); strict.retain_one_oversize = true; strict.charge_token_bytes = false;
    ss::PrefixCache protected_cache(strict, ss::make_cache_owner(cost * 10)); CHECK(protected_cache.admit(kept));
    CHECK(!protected_cache.admit(replacement)); CHECK(!protected_cache.admit(b));
    CHECK(protected_cache.size() == 1 && protected_cache.lookup_turn(kept->state.tokens, kept->identity, *kept->turn));
    CHECK(!protected_cache.can_admit_turn(a->identity, *a->turn, UINT64_MAX, 0));
    CHECK(!protected_cache.can_admit_turn(a->identity, *a->turn, 2, UINT64_MAX));
    CHECK(protected_cache.unpin("bookmark")); CHECK(!protected_cache.unpin("bookmark"));
    CHECK(protected_cache.admit(b)); CHECK(protected_cache.bytes() <= strict.max_bytes);
    ss::PrefixCache unlimited(turn_policy(UINT64_MAX)); CHECK(!unlimited.admit(a));
    ss::PrefixCache zero(turn_policy(0)); CHECK(!zero.admit(a));
    auto under = turn_policy(cost - 1); under.retain_one_oversize = true; under.charge_token_bytes = false;
    ss::PrefixCache too_small(under); CHECK(!too_small.admit(a) && too_small.size() == 0);
    auto ordinary = std::make_shared<ss::PrefixSnapshot>(*b); ordinary->turn.reset();
    ss::PrefixCache shared_budget(strict); CHECK(shared_budget.admit(kept));
    CHECK(!shared_budget.admit(ordinary) && shared_budget.size() == 1);
    auto renamed = std::make_shared<ss::PrefixSnapshot>(*b); renamed->turn->keep = "bookmark";
    CHECK(!shared_budget.admit(renamed));
    ss::PrefixCache frozen(turn_policy()); auto indexed = snapshot(e, {8,9}, "freeze");
    const auto metadata = *indexed->turn; CHECK(frozen.admit(indexed));
    indexed->turn->template_id = "edited";
    CHECK(frozen.lookup_turn(indexed->state.tokens, indexed->identity, metadata));
    ss::Policy defaults; ss::PrefixCache off(defaults); CHECK(!off.admit(a));
    // Store provenance prevalidation must not invoke any erased payload hook.
    auto k = e.store.acquire(); e.prefill(std::array{1U,2U},k);
    auto resume = std::make_shared<ss::PrefixSnapshot>(*a); resume->state = e.store.capture(k);
    CHECK(resume->state.sealed() && !cache.admit(resume)); // correct seal, wrong purpose
    e.store.reset(k); const auto calls = e.restores;
    rejects([&] { (void)e.store.restore_prompt(k, bad->state); }); CHECK(e.restores == calls);
    TurnEngine other;
    auto other_k = other.store.acquire(); rejects([&] { (void)other.store.restore_prompt(other_k, a->state); });
    CHECK(other.restores == 0); other.store.release(other_k); e.store.release(k);
}
void runtime_reuse() {
    TurnEngine e; e.limits.prefix_cache = true; e.require_prepared = true;
    sv::GenerationQueue q; auto h = turn_hooks(e);
    h.turn_snapshot = [&e](const auto& v) { return e.prepare(v.state); };
    h.checkpoint = [](const auto&) { return true; };
    sv::SessionRuntime r(q, e, e.store, turn_policy(), h);
    const auto first = execute(r, q, request()); CHECK(first.error.empty() && first.ids.size() == 2);
    CHECK(e.seals == 1 && e.consumed_anchors == 1 && r.cache_stats().turn_admitted == 1);
    CHECK(r.cache_entries() == 1); // no legacy resume checkpoint fallback
    auto history = e.sealed_tokens; history.push_back(71);
    const auto warm = execute(r, q, request(history, 2, 19));
    CHECK(warm.error.empty() && warm.stats.cached_tokens == 4 && e.restores == 1);
    CHECK(e.preparations == 2 && e.selected_ids.size() == 4);
    CHECK(e.suffixes.back() == std::vector<std::uint32_t>{71});
    for (unsigned rank = 0; rank < 2; ++rank) {
        Rank reference; for (auto id : std::vector<std::uint32_t>(history.begin(), history.end() - 1)) TurnEngine::row(reference, id, rank);
        reference.seed = 19;
        CHECK(e.restored_state[rank] == reference);
    }
    TurnEngine cold; sv::GenerationQueue cq; auto off = turn_policy(); off.completed_turns = false;
    sv::SessionRuntime cr(cq, cold, cold.store, off, turn_hooks(cold));
    const auto reference = execute(cr, cq, request(history, 2, 19));
    CHECK(warm.ids == reference.ids && warm.stats.finish == reference.stats.finish);
    CHECK(warm.stats.engine_extra == reference.stats.engine_extra);
    auto edited = history; edited[1] ^= 1;
    CHECK(execute(r, q, request(edited)).stats.cached_tokens == 0);
    CHECK(r.cache_stats().hits == 1);
    CHECK(e.events[e.events.size() - 1] == "release");
}
void runtime_boundaries() {
    for (unsigned mode = 0; mode < 28; ++mode) {
        TurnEngine e; sv::GenerationQueue q; auto p = turn_policy(); auto h = turn_hooks(e); auto j = request();
        if (mode == 0) p.completed_turns = false;
        if (mode == 1) j.cache_retention = sv::CacheRetention::no_store;
        if (mode == 2) h.identity.model.clear();
        if (mode == 3) h.identity.profile.clear();
        if (mode == 4) h.identity.ranks = 1;
        if (mode == 5) h.identity.state_format.clear();
        if (mode == 6) h.turn_cache = [](const sv::SessionView&) { return std::optional<sv::SessionTurnCache>{{{"a", "", "pair", ""}, 20000}}; };
        if (mode == 7) e.limits.turn_commit = false;
        if (mode == 8) h.turn_checkpoint = [](const sv::SessionView&, ss::FinishReason) { return false; };
        if (mode == 9) h.cancelled = [](const sv::SessionView& v) { return v.stats.generated_tokens != 0; };
        if (mode == 10) p.max_bytes = 1;
        if (mode == 11) e.fail_consume = true;
        if (mode == 12) e.fail_capture = true;
        if (mode == 13) j.max_tokens = 0;
        if (mode == 14) { e.limits.max_context = 2; j.max_tokens = 0; }
        if (mode == 15) { e.fail_consume = e.fail_rollback = e.fail_release = true; }
        if (mode == 16) h.turn_cache = [](const sv::SessionView&) {
            return std::optional<sv::SessionTurnCache>{{{"a", "render-v1", "pair", ""}, 0}};
        };
        if (mode == 20) { e.limits.prefix_cache = true; h.identity.model.clear(); }
        if (mode == 21) { e.limits.prefix_cache = true; h.identity.ranks = 1; }
        if (mode == 22) h.turn_cache = [](const sv::SessionView&) {
            return std::optional<sv::SessionTurnCache>{{{"a", "render-v1", "pair", ""}, 1}};
        };
        if (mode == 23) p.enabled = false;
        if (mode == 24) p.store = false;
        if (mode == 25) h.turn_cache = {};
        if (mode == 26) h.turn_checkpoint = {};
        if (mode == 27) h.receipt = [](const auto&, auto&) { throw std::runtime_error("receipt fault"); };
        bool hook_cancelled = false;
        unsigned snapshot_calls = 0;
        h.turn_snapshot = [&snapshot_calls](const auto&) { ++snapshot_calls; return true; };
        if (mode == 17) h.cancelled = [&e](const auto&) { return e.seals != 0; };
        if (mode == 18) {
            h.turn_checkpoint = [&hook_cancelled](const auto&, auto) { hook_cancelled = true; return true; };
            h.cancelled = [&hook_cancelled](const auto&) { return hook_cancelled; };
        }
        if (mode == 19) h.cancelled = [&e](const auto&) { return e.captures != 0; };
        sv::SessionRuntime r(q, e, e.store, p, h); const auto result = execute(r, q, j);
        CHECK(result.error.empty() == (mode != 27));
        if (mode <= 10) CHECK(e.seals == 0 && r.cache_entries() == 0);
        if (mode == 11 || mode == 12 || mode == 22) {
            CHECK(result.ids.size() == 2 && result.stats.finish == "length");
            CHECK(r.cache_stats().turn_failed == 1 && r.cache_entries() == 0);
            CHECK(result.stats.engine_extra.back().second == "3"); // pre-seal receipt
        }
        if (mode == 22) CHECK(e.seals == 1 && e.captures == 2); // Nonzero underestimated bound.
        if (mode == 16) {
            CHECK(e.seals == 0 && e.captures == 0 && r.cache_stats().turn_refused == 1);
            CHECK(std::find(e.events.begin(), e.events.end(), "admit-final") == e.events.end());
        }
        if (e.seals == 0 || mode == 11 || mode == 15 || mode == 17) CHECK(snapshot_calls == 0);
        if (mode == 13 || mode == 14) CHECK(e.seals == 1 && e.consumed_anchors == 0 && result.ids.empty());
        if (mode >= 17 && mode <= 19) {
            CHECK(r.cache_entries() == 0 && r.cache_stats().turn_refused == 1);
            CHECK(e.seals == (mode == 18 ? 0U : 1U));
        }
        if (mode == 20 || mode == 21) CHECK(e.seals == 0 && r.cache_entries() == 0);
        if (mode >= 23) CHECK(e.seals == 0 && e.captures == 0 && r.cache_entries() == 0);
        if (mode == 15) {
            CHECK(result.ids.size() == 2 && result.stats.finish == "length" && r.cache_entries() == 0);
            CHECK(r.cache_stats().turn_failed == 2); // consume plus quarantined cleanup
            auto k = e.store.acquire(); CHECK(k.slot != 0); e.fail_release = false; e.store.release(k);
        }
    }
    // EOS, length and full-context each seal the last published anchor once.
    for (unsigned mode = 0; mode < 3; ++mode) {
        TurnEngine e; sv::GenerationQueue q; auto j = request({10,11}, 1);
        Rank reference; TurnEngine::row(reference, 10, 0); TurnEngine::row(reference, 11, 0);
        if (mode == 0) j.stop_tokens = {static_cast<std::uint32_t>(32 + (reference.recurrent.back() + 3) % 80)};
        if (mode == 2) e.limits.max_context = 3;
        sv::SessionRuntime r(q,e,e.store,turn_policy(),turn_hooks(e)); auto out = execute(r,q,j);
        CHECK(out.error.empty() && e.consumed_anchors == 1 && e.sealed_tokens.back() == out.ids.back());
        CHECK(out.stats.finish == (mode == 0 ? "stop" : mode == 1 ? "length" : "context"));
    }
    // A thinking boundary continues before any sealing; evaluated delimiters
    // and both phases' generated IDs become the canonical store-issued key.
    TurnEngine e; sv::GenerationQueue q; auto h = turn_hooks(e);
    e.require_prepared = true;
    h.turn_snapshot = [&e](const auto& v) { return e.prepare(v.state); };
    h.initial_allowance = [](const auto&) { return 1U; };
    h.continuation = [&e](const sv::SessionView& v, ss::FinishReason) -> std::optional<sv::SessionContinuation> {
        e.events.push_back("continue"); CHECK(e.seals == 0);
        if (v.stats.generated_tokens != 1) return {};
        for (auto& rank : e.slots[v.state.slot]) rank.terminal = false;
        return sv::SessionContinuation{{91,92}, {}, 1};
    };
    sv::SessionRuntime r(q,e,e.store,turn_policy(),h); auto out = execute(r,q,request({10,11},3));
    CHECK(out.error.empty() && out.ids.size() == 2 && e.seals == 1 && e.preparations == 1);
    CHECK(e.sealed_tokens == (std::vector<std::uint32_t>{10,11,out.ids[0],91,92,out.ids[1]}));
    const auto receipt = std::find(e.events.begin(),e.events.end(),"receipt");
    const auto seal = std::find(e.events.begin(),e.events.end(),"seal");
    CHECK(receipt < seal && std::find(e.events.begin(),seal,"continue") != seal);
}
void runtime_snapshot_handshake() {
    {
        TurnEngine e; e.limits.owner_control = true;
        sv::GenerationQueue q; auto h = turn_hooks(e); unsigned calls = 0;
        h.turn_snapshot = [&](const auto&) { ++calls; return true; };
        h.control_turn = [&](sv::SessionRuntime& owner) {
            if (!owner.reserve(sv::QueueSelection::fifo)) return;
            const auto lease = owner.activate(); owner.complete(lease.state);
        };
        sv::SessionRuntime r(q, e, e.store, turn_policy(), h);
        CHECK(execute(r, q, request({10,11}, 0)).error.empty());
        CHECK(calls == 0 && e.seals == 0 && e.captures == 0 && r.cache_entries() == 0);
    }
    // The producer really refuses prompt capture until explicitly prepared.
    {
        TurnEngine e; e.require_prepared = true;
        auto k = e.store.acquire(); e.prefill(std::array{1U, 2U}, k);
        ss::SamplerState s; s.max_tokens = 1; e.decode(k, s); e.commit_pending(k);
        rejects([&] { (void)e.store.capture_prompt(k); });
        CHECK(e.prepare(k)); CHECK(e.store.capture_prompt(k).sealed()); e.store.release(k);
    }
    TurnEngine cold; sv::GenerationQueue cq; auto off = turn_policy(); off.completed_turns = false;
    auto ch = turn_hooks(cold); ch.initial_allowance = [](const auto&) { return 2U; };
    ch.continuation = [](const auto&, auto) -> std::optional<sv::SessionContinuation> { return {}; };
    sv::SessionRuntime cr(cq, cold, cold.store, off, ch);
    const auto reference = execute(cr, cq, request({10,11}, 3));
    for (unsigned mode = 0; mode < 7; ++mode) {
        TurnEngine e; e.require_prepared = true;
        // Live source and bystander state must survive every finalization path.
        auto source = e.store.acquire(), bystander = e.store.acquire();
        e.prefill(std::array{7U,8U}, source); e.prefill(std::array{9U}, bystander);
        const auto source_before = e.slots[source.slot], bystander_before = e.slots[bystander.slot];
        sv::GenerationQueue q; auto h = turn_hooks(e);
        h.initial_allowance = [](const auto&) { return 2U; };
        h.continuation = [&e](const auto&, auto) -> std::optional<sv::SessionContinuation> {
            e.events.push_back("continuation"); return {};
        };
        std::shared_ptr<sv::GenerationTicket> ticket;
        bool hook_cancelled = false;
        h.cancelled = [&](const auto&) { return hook_cancelled; };
        ss::KvHandle sealed_handle;
        h.turn_snapshot = [&](const auto& v) {
            sealed_handle = v.state;
            const auto before = e.slots[v.state.slot];
            CHECK(e.prepare(v.state)); CHECK(e.slots[v.state.slot] == before);
            CHECK(v.rng_draws == 2 && v.stats.generated_tokens == 2);
            CHECK(e.selected_ids == reference.ids);
            if (mode == 1) return false;
            if (mode == 2) throw std::runtime_error("candidate fault");
            if (mode == 3) hook_cancelled = true; // Reentrant family cancellation.
            if (mode == 4) {
                ++e.candidates[v.state.slot]->handle.generation;
                return e.candidate_valid(v.state); // Stale candidates refuse.
            }
            if (mode == 5) e.fail_release = true;
            return true;
        };
        if (mode == 6) h.turn_snapshot = {}; // Old producer cannot satisfy this fixture.
        sv::SessionRuntime r(q, e, e.store, turn_policy(), h);
        e.events.clear(); ticket = q.submit(request({10,11}, 3));
        for (unsigned tick = 0; tick < 20 && q.snapshot().completed == 0; ++tick) r.step(false);
        const auto out = drain(ticket);
        if (!out.error.empty() || out.ids != reference.ids || out.stats.finish != reference.stats.finish)
            throw std::runtime_error("snapshot mode " + std::to_string(mode) + ": " + out.error + "; finish " + out.stats.finish);
        CHECK(out.error.empty() && out.ids == reference.ids && out.stats.finish == reference.stats.finish);
        CHECK(out.stats.engine_extra == reference.stats.engine_extra);
        CHECK(out.stats.generated_tokens == reference.stats.generated_tokens);
        CHECK(e.slots[source.slot] == source_before && e.slots[bystander.slot] == bystander_before);
        CHECK(e.seals == 1 && e.consumed_anchors == 1 && e.selected_ids == reference.ids);
        CHECK(e.preparations == (mode == 6 ? 0U : 1U));
        if (mode == 0 || mode == 5) {
            CHECK(r.cache_stats().turn_admitted == 1 && e.captures == 2);
            CHECK(e.events == (std::vector<std::string>{"continuation", "receipt", "admit-final", "seal",
                "turn_snapshot", "capture", "capture", "release"}));
            CHECK(r.cache_stats().turn_failed == (mode == 5 ? 1U : 0U));
            if (mode == 0) {
                std::cout << "finalization events:";
                for (const auto& event : e.events) std::cout << ' ' << event;
                std::cout << '\n';
            }
        } else {
            CHECK(r.cache_entries() == 0 && r.cache_stats().turn_admitted == 0);
            CHECK(r.cache_stats().turn_failed == (mode == 2 || mode == 6 ? 1U : 0U));
            CHECK(r.cache_stats().turn_refused == (mode == 1 || mode == 3 || mode == 4 ? 1U : 0U));
            CHECK(e.captures == (mode == 6 ? 1U : 0U));
            CHECK(e.events.back() == "release");
        }
        if (mode == 5) {
            rejects([&] { (void)e.store.tokens(sealed_handle); });
            rejects([&] { (void)e.store.acquire(); }); // Quarantined, other slots still live.
        }
        e.fail_release = false; e.store.release(source); e.store.release(bystander);
    }
}
void runtime_keeps() {
    TurnEngine e; sv::GenerationQueue q; auto h = turn_hooks(e);
    h.turn_cache = [](const sv::SessionView&) {
        return std::optional<sv::SessionTurnCache>{{{"conversation", "render-v1", "ordered-pair", "named"}, 20000}};
    };
    sv::SessionRuntime r(q,e,e.store,turn_policy(),h);
    CHECK(execute(r,q,request()).error.empty());
    const auto original = e.sealed_tokens;
    auto next = original; next.push_back(87);
    auto no_store = request(next); no_store.cache_retention = sv::CacheRetention::no_store;
    CHECK(execute(r,q,no_store).stats.cached_tokens == original.size());
    CHECK(e.seals == 1 && r.cache_entries() == 1);
    CHECK(execute(r,q,request(next)).stats.cached_tokens == original.size());
    CHECK(e.seals == 1 && r.cache_stats().turn_refused == 2);
    CHECK(r.unpin_turn_cache("named") && !r.unpin_turn_cache("named"));
    CHECK(execute(r,q,request(next)).stats.cached_tokens == original.size());
    CHECK(e.seals == 2 && r.cache_entries() == 1);
}
void runtime_restore_faults() {
    for (unsigned mode = 0; mode < 3; ++mode) {
        TurnEngine e; sv::GenerationQueue q; sv::SessionRuntime r(q,e,e.store,turn_policy(),turn_hooks(e));
        CHECK(execute(r,q,request()).error.empty()); auto ids = e.sealed_tokens; ids.push_back(77);
        e.decline_restore = mode == 0; e.fail_restore = mode != 0;
        e.fail_rollback = mode == 2; e.fail_release = mode == 2;
        const auto out = execute(r,q,request(ids));
        CHECK(out.stats.cached_tokens == 0 && r.cache_stats().hits == 0);
        if (mode == 0) CHECK(out.error.empty() && r.cache_stats().misses == 2);
        else {
            CHECK(!out.error.empty() && out.ids.empty() && r.cache_stats().restore_failed == 1);
            if (mode == 1) CHECK(e.rollbacks == 2);
            if (mode == 2) { auto k = e.store.acquire(); CHECK(k.slot != 0); e.fail_release = false; e.store.release(k); }
        }
    }
    // Capture failure after an earlier admission cannot evict that checkpoint.
    {
        TurnEngine e; sv::GenerationQueue q; sv::SessionRuntime r(q,e,e.store,turn_policy(),turn_hooks(e));
        CHECK(execute(r,q,request()).error.empty()); auto original = e.sealed_tokens; original.push_back(77);
        const auto bytes = r.cache_stats().bytes;
        e.fail_capture = true; CHECK(execute(r,q,request({4,5})).error.empty());
        CHECK(r.cache_entries() == 1 && r.cache_stats().bytes == bytes);
        e.fail_capture = false; CHECK(execute(r,q,request(original)).stats.cached_tokens == 4);
    }
    // Two simultaneously live slots with independent physical ranks.
    TurnEngine e; sv::GenerationQueue q; sv::SessionRuntime r(q,e,e.store,turn_policy(),turn_hooks(e));
    auto a = q.submit(request({1,2},1)), b = q.submit(request({3,4},1));
    r.step(); CHECK(r.live() == 2); CHECK(e.slots[0][0].attention != e.slots[1][0].attention);
    r.step(); CHECK(drain(a).error.empty() && drain(b).error.empty() && e.seals == 2);
}
void http_two_turns() {
    auto tokens = tokenizer();
    auto replay = [&](bool enabled) {
        TurnEngine e; sv::GenerationQueue q; auto p = turn_policy(); p.completed_turns = enabled;
        sv::SessionRuntime r(q,e,e.store,p,turn_hooks(e));
        sv::ApiConfig cfg; cfg.model_id = "turn-fixture"; cfg.sampling_policy = "reject"; r.configure_api(cfg);
        sv::ApiService api(cfg,tokens,q,512,false);
        std::exception_ptr failure; std::thread owner([&] { try { r.run(); } catch (...) { failure = std::current_exception(); } });
        std::array<std::string,2> texts;
        try {
            for (unsigned turn=0;turn<2;++turn) {
                sv::HttpRequest req; req.method="POST"; req.path="/v1/completions";
                // Exact byte-token IDs, including the prior answer; no text
                // reconstruction is used by the cache or its index.
                sv::Json::Array ids{10,11};
                if (turn) {
                    CHECK(e.selected_ids.size() == 2);
                    for (auto id : e.selected_ids) ids.emplace_back(std::uint64_t(id));
                    ids.emplace_back(71);
                }
                req.body=sv::Json(sv::Json::Object{{"prompt",sv::Json(std::move(ids))},{"max_tokens",2},{"temperature",0}}).dump();
                Writer w; api.handle(req,w); CHECK(w.status==200); texts[turn]=w.body;
            }
        } catch (...) { r.shutdown(); owner.join(); throw; }
        r.shutdown(); owner.join(); if(failure) std::rethrow_exception(failure);
        if(enabled) CHECK(e.restores==1 && q.snapshot().cached_tokens==4);
        else CHECK(e.restores==0 && q.snapshot().cached_tokens==0);
        std::array<std::string,2> outputs;
        for(unsigned i=0;i<2;++i) outputs[i]=sv::parse_json(texts[i]).find("choices")->dump();
        return outputs;
    };
    CHECK(replay(true)==replay(false));
}
}
int main() {
    try {
        direct_contract(); cache_policy(); runtime_reuse(); runtime_boundaries(); runtime_snapshot_handshake();
        runtime_keeps(); runtime_restore_faults(); http_two_turns();
        std::cout << "completed-turn physical state, policy, lifecycle and HTTP checks passed\n"; return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
