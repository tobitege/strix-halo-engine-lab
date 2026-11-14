#include "strix/serve/constraint_selection.hpp"
#include "strix/env_flags.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <chrono>
#include <cmath>
#include <type_traits>

namespace strix::serve {

// One plan = the masks one step can need. Slots are claimed with phase 0 -> 1 by whoever gets there first
// (the worker in slot order, or the selecting thread for the row it is deciding) and published with phase 2.
struct ConstraintMaskAhead {
    struct Slot {
        explicit Slot(ConstraintCursor c) : cursor(std::move(c)) {}
        std::atomic<int> phase{0};
        ConstraintCursor cursor;
        AllowedTokens mask;
        std::exception_ptr error;
    };
    std::shared_ptr<const ConstraintSpec> spec;
    std::shared_ptr<const TokenByteTrie> trie;
    std::shared_ptr<ConstraintAheadTotals> totals;
    std::vector<std::unique_ptr<Slot>> slots;   // null: that state needs no mask (reasoning)
    std::atomic<bool> retired{false};

    // Phase 3: the worker abandoned the walk because the plan was retired (a retired plan is never taken;
    // take() still handles it by walking itself).
    void build(Slot& s, bool stoppable) const noexcept {
        int done = 2;
        try {
            if (stoppable) {
                if (auto mask = allowed_until(*spec->program, s.cursor, *trie, spec->eos_ids, retired)) s.mask = std::move(*mask);
                else done = 3;
            } else s.mask = allowed(*spec->program, s.cursor, *trie, spec->eos_ids);
        } catch (...) { s.error = std::current_exception(); }
        s.phase.store(done, std::memory_order_release);
        s.phase.notify_all();
    }
    void run() const noexcept {
        for (const auto& s : slots) {
            if (retired.load(std::memory_order_acquire)) return;
            int idle = 0;
            if (!s || !s->phase.compare_exchange_strong(idle, 1, std::memory_order_acq_rel)) continue;
            if (totals->observer) totals->observer(1);
            const auto t0 = std::chrono::steady_clock::now();
            build(*s, true);
            totals->ns.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count()), std::memory_order_relaxed);
            totals->masks.fetch_add(1, std::memory_order_relaxed);
            if (totals->observer) totals->observer(s->phase.load(std::memory_order_acquire));
        }
    }
    // The mask of the slot, or the error of its walk rethrown here (where prepare() would have raised it).
    const AllowedTokens& take(Slot& s, bool& from_worker) const {
        int phase = 0;
        if (s.phase.compare_exchange_strong(phase, 1, std::memory_order_acq_rel)) build(s, false);
        else {
            if (phase == 1 && totals->observer) totals->observer(4);
            for (; phase == 1; phase = s.phase.load(std::memory_order_acquire)) s.phase.wait(phase, std::memory_order_acquire);
            if (phase == 3) build(s, false); else from_worker = true;
        }
        if (s.error) std::rethrow_exception(s.error);
        return s.mask;
    }
};

ConstraintSelection::ConstraintSelection(std::shared_ptr<const ConstraintSpec> spec,
    std::shared_ptr<const TokenByteTrie> trie, std::shared_ptr<const BpeTokenizer> tokenizer)
    : spec_(std::move(spec)), trie_(std::move(trie)), tokenizer_(std::move(tokenizer)),
      state_{{}, spec_->starts_in_reasoning, ConstraintCursor(spec_->program ? *spec_->program : compile_json_object())} {}

ConstraintSelectionPolicy constraint_selection_policy_from_environment() {
    ConstraintSelectionPolicy policy;
    policy.mask_ahead = envcfg::flag("STRIX_CONSTRAINT_MASK_AHEAD", true, EnvFlagMatch::exact);
    policy.ranked = envcfg::number<unsigned>("STRIX_CONSTRAINT_RANKED", 0U, 0U, 4096U);
    return policy;
}
ConstraintSelection::Prepared ConstraintSelection::prepare() const { return prepare(state_); }
ConstraintSelection::Prepared ConstraintSelection::prepare(const State& state) const { return prepare(state, nullptr, 0); }
void ConstraintSelection::retire_ahead(std::shared_ptr<ConstraintMaskAhead>& plan) noexcept {
    if (plan) plan->retired.store(true, std::memory_order_release);
    plan.reset();
}
std::shared_ptr<ConstraintMaskAhead> ConstraintSelection::plan_ahead(const State& base, std::span<const std::uint32_t> path) const {
    if (!ahead_submit_ || !spec_->program || base.terminal) return {};
    // Per regime: with argmax-first and no scores or transforms a row walks a mask only when its ranked
    // candidates are all refused, and walks nobody takes then only load the host; every other request walks
    // one mask per answer row.
    if (argmax_first_ && !spec_->logprobs && !spec_->transform.active()) return {};
    auto plan = std::make_shared<ConstraintMaskAhead>();
    plan->spec = spec_; plan->trie = trie_; plan->totals = ahead_totals_;
    // The unforced part of advance(): the chain stops where a row could not select its draft (a stop ID, a
    // token the cursor refuses) or where the reasoning closure is forced.
    bool reasoning = base.reasoning, any = false;
    auto reasoning_tokens = base.reasoning_tokens;
    auto cursor = base.cursor;
    for (std::size_t r = 0;; ++r) {
        plan->slots.push_back(reasoning ? nullptr : std::make_unique<ConstraintMaskAhead::Slot>(cursor));
        any = any || !reasoning;
        if (r == path.size()) break;
        const auto id = path[r];
        if (std::find(spec_->eos_ids.begin(), spec_->eos_ids.end(), id) != spec_->eos_ids.end()) break;
        if (reasoning) {
            if (spec_->thinking && reasoning_tokens >= spec_->thinking->reasoning_limit) break;
            if (spec_->end_thinking_id == id) reasoning = false; else ++reasoning_tokens;
        } else {
            if (spec_->end_thinking_id == id || !trie_->contains(id)) break;
            // A transition that exhausts its evaluation budget ends the chain like a refusal: the slots before
            // it are safe, and its error belongs to the row that really takes this draft (its mask walk or advance() raises it
            // there), never to a window that only planned it.
            bool admitted = true;
            try {
                for (const char b : tokenizer_->token_bytes(id)) if (!cursor.advance(static_cast<unsigned char>(b))) { admitted = false; break; }
            } catch (const ConstraintError&) { admitted = false; }
            if (!admitted) break;
        }
    }
    if (!any) return {};
    // A refused submission (worker shut down) leaves the plain path: the hint never decides a token.
    try {
        ahead_submit_([plan] { plan->run(); });
        if (plan->slots.size() > 1) ahead_submit_([plan] { plan->run(); });
    } catch (...) { ConstraintSelection::retire_ahead(plan); return {}; }
    return plan;
}
ConstraintSelection::Prepared ConstraintSelection::prepare(const State& state, ConstraintMaskAhead* ahead, std::size_t slot) const {
    if (state.terminal) throw ConstraintError("selection after constraint termination");
    Prepared p{state, {}};
    if (!state.reasoning && spec_->program) {
        const auto t0 = std::chrono::steady_clock::now();
        struct Timed { const std::chrono::steady_clock::time_point& t0; std::uint64_t& ns;
            ~Timed() { ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()); } }
            timed{t0, profile_.mask_ns};
        if (auto* s = ahead && slot < ahead->slots.size() ? ahead->slots[slot].get() : nullptr) {
            bool from_worker = false;
            p.mask = ahead->take(*s, from_worker);
            if (from_worker) ++profile_.ahead_rows;
        } else p.mask = allowed(*spec_->program, state.cursor, *trie_, spec_->eos_ids);
        if (spec_->end_thinking_id) p.mask.set(*spec_->end_thinking_id, false);
        if (std::none_of(p.mask.words.begin(), p.mask.words.end(), [](auto word) { return word != 0; }))
            throw ConstraintError("empty constraint mask");
    }
    return p;
}
ConstraintSelection::Staged ConstraintSelection::select(std::span<const float> logits, const Prepared& p) const {
    const auto t0 = std::chrono::steady_clock::now();
    struct Timed { const std::chrono::steady_clock::time_point& t0; Profile& p;
        ~Timed() { ++p.rows; p.select_ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()); } }
        timed{t0, profile_};
    if (logits.size() != trie_->vocab_size()) throw ConstraintError("constraint logits vocabulary mismatch");
    // Preserve the target's all-vocabulary finite-row contract, including masked entries.
    for (const auto value : logits) if (!std::isfinite(value)) throw ConstraintError("nonfinite target logits");
    std::vector<float> transformed;
    if (spec_->transform.active()) {
        transformed.assign(logits.begin(), logits.end());
        const auto& t = spec_->transform;
        // logit-transform-v1: sign-sensitive repetition on the raw logit of every token seen in the prompt or the
        // reply, then presence and frequency over reply counts, then bias.
        if (t.repetition != 1) {
            std::vector<bool> seen(logits.size(), false);
            for (auto token : spec_->prompt_ids) seen.at(token) = true;
            for (const auto& [token, count] : p.state.reply_counts) { (void)count; seen.at(token) = true; }
            for (std::size_t i = 0; i < transformed.size(); ++i) if (seen[i])
                transformed[i] = static_cast<float>(transformed[i] > 0 ? transformed[i] / t.repetition : transformed[i] * t.repetition);
        }
        for (const auto& [token, count] : p.state.reply_counts) {
            transformed.at(token) -= static_cast<float>(t.presence);
            transformed.at(token) -= static_cast<float>(t.frequency * static_cast<double>(count));
        }
        for (const auto& [token, bias] : t.bias) transformed.at(token) += static_cast<float>(bias);
        for (auto value : transformed) if (!std::isfinite(value)) throw ConstraintError("nonfinite transformed logits");
        logits = transformed;
    }
    std::uint32_t id = 0;
    const bool force = p.state.reasoning && spec_->thinking &&
        p.state.reasoning_tokens >= spec_->thinking->reasoning_limit;
    if (force) id = spec_->thinking->closure_ids.at(p.state.forced_ids.size());
    else if (p.state.reasoning || !spec_->program) {
        if (logits.empty()) throw ConstraintError("empty target logits");
        for (std::size_t i = 1; i < logits.size(); ++i)
            if (logits[i] > logits[id]) id = static_cast<std::uint32_t>(i);
    } else id = constrained_greedy(logits, p.mask);
    Staged staged{id, p.state, force, {id, {}, {}, force}};
    if (spec_->logprobs && !force) {
        std::vector<std::uint32_t> order;
        for (std::uint32_t i = 0; i < logits.size(); ++i)
            if (p.mask.words.empty() || p.mask.contains(i)) order.push_back(i);
        double maximum = logits[id], sum = 0;
        for (auto token : order) maximum = std::max(maximum, static_cast<double>(logits[token]));
        for (auto token : order) sum += std::exp(static_cast<double>(logits[token]) - maximum);
        const double normalizer = std::log(sum);
        const auto score = [&](std::uint32_t token) { return (static_cast<double>(logits[token]) - maximum) - normalizer; };
        staged.score.logprob = score(id);
        const auto n = std::min<std::size_t>(spec_->top_logprobs, order.size());
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(n), order.end(), [&](auto a, auto b) {
            const auto pa = score(a), pb = score(b);
            return pa != pb ? pa > pb : a < b;
        });
        for (std::size_t i = 0; i < n; ++i) staged.score.top.emplace_back(order[i], score(order[i]));
    }
    advance(staged, p.state, force);
    return staged;
}
void ConstraintSelection::advance(Staged& staged, const State& prior, bool force) const {
    const auto id = staged.token;
    if (spec_->transform.active()) ++staged.state.reply_counts[id];
    staged.state.frontier = std::make_shared<Frontier>(Frontier{prior.frontier});
    const bool eos = std::find(spec_->eos_ids.begin(), spec_->eos_ids.end(), id) != spec_->eos_ids.end();
    if (eos) staged.state.terminal = true;
    else if (prior.reasoning) {
        if (force) {
            staged.state.forced_ids.push_back(id);
            if (staged.state.forced_ids.size() == spec_->thinking->closure_ids.size()) staged.state.reasoning = false;
        } else if (spec_->end_thinking_id == id) staged.state.reasoning = false;
        else ++staged.state.reasoning_tokens;
    } else if (spec_->program) staged.state.cursor = advance_token(prior.cursor, tokenizer_->token_bytes(id));
}
bool ConstraintSelection::admits(const State& state, std::uint32_t id) const {
    // prepare(): the walk admits trie tokens whose bytes all advance the cursor, EOS IDs are then set to
    // cursor.complete(), and the end-of-thinking ID is cleared last.
    if (spec_->end_thinking_id && *spec_->end_thinking_id == id) return false;
    if (std::find(spec_->eos_ids.begin(), spec_->eos_ids.end(), id) != spec_->eos_ids.end()) return state.cursor.complete();
    if (!trie_->contains(id)) return false;
    auto cursor = state.cursor;
    for (const char b_char : tokenizer_->token_bytes(id)) { unsigned char b = static_cast<unsigned char>(b_char); if (!cursor.advance(b)) return false; }
    return true;
}
std::optional<ConstraintSelection::Staged> ConstraintSelection::select_ranked(std::span<const float> row, const State& state) const {
    // Only where select() is constrained_greedy() and nothing else: a finite full row (the caller scanned it),
    // an answer-phase program state, no scores, no transforms.
    if (!ranked_ || state.terminal || state.reasoning || !spec_->program || spec_->logprobs || spec_->transform.active()) return std::nullopt;
    const auto t0 = std::chrono::steady_clock::now();
    struct Timed { const std::chrono::steady_clock::time_point& t0; std::uint64_t& ns;
        ~Timed() { ns += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()); } }
        timed{t0, profile_.select_ns};
    // The best ranked_ + 1 logits in masked-greedy order: value descending, lowest ID first among equals (both
    // zeros equal, as in constrained_greedy). IDs ascend, so an equal value goes behind those already kept.
    struct Candidate { float value; std::uint32_t id; };
    const std::size_t keep = static_cast<std::size_t>(ranked_) + 1;
    std::vector<Candidate> top; top.reserve(keep + 1);
    for (std::uint32_t i = 0; i < row.size(); ++i) {
        const float value = row[i];
        if (top.size() == keep && !(value > top.back().value)) continue;
        auto at = top.end();
        while (at != top.begin() && value > (at - 1)->value) --at;
        top.insert(at, Candidate{value, i});
        if (top.size() > keep) top.pop_back();
    }
    // top[0] is the row's lowest-ID maximum, already refused by select_known().
    for (std::size_t k = 1; k < top.size(); ++k) {
        if (!admits(state, top[k].id)) continue;
        ++profile_.ranked_rows;
        Staged staged{top[k].id, state, false, {top[k].id, {}, {}, false}};
        advance(staged, state, false);
        return staged;
    }
    return std::nullopt;
}
std::optional<ConstraintSelection::Staged> ConstraintSelection::select_known(const State& state, std::uint32_t token, std::uint32_t status) const {
    // A nonfinite row, a terminal state and score or transform requests take the full-row path (its errors and
    // their order stay those of prepare()/select()).
    if (!argmax_first_ || status != 0 || token >= trie_->vocab_size() || state.terminal || spec_->logprobs || spec_->transform.active())
        return std::nullopt;
    const bool force = state.reasoning && spec_->thinking && state.reasoning_tokens >= spec_->thinking->reasoning_limit;
    std::uint32_t id = token;
    if (force) id = spec_->thinking->closure_ids.at(state.forced_ids.size());
    else if (!state.reasoning && spec_->program && !admits(state, token)) return std::nullopt;
    ++profile_.known_rows;
    Staged staged{id, state, force, {id, {}, {}, force}};
    advance(staged, state, force);
    return staged;
}
void ConstraintSelection::commit(const Staged& staged) {
    auto ancestor = staged.state.frontier->parent;
    while (ancestor && ancestor != state_.frontier) ancestor = ancestor->parent;
    if (!ancestor) throw ConstraintError("stale or already committed constraint branch");
    if (spec_->logprobs) {
        std::lock_guard lock(spec_->scores->mutex);
        spec_->scores->tokens.push_back(staged.score);
    }
    state_ = staged.state;
    // Descendants still identify this frontier, but serial requests do not retain
    // a token-count-sized ancestry chain (or recursively destroy one at finish).
    state_.frontier->parent.reset();
}

void ConstraintSelection::publish_prefix(const std::shared_ptr<Frontier>& base, const std::vector<Staged>& records) {
    if (base != state_.frontier) throw ConstraintError("stale or empty constraint window");
    if(records.empty())return;
    auto parent = base;
    for (const auto& record : records) {
        if (!record.state.frontier || record.state.frontier->parent != parent)
            throw ConstraintError("noncontiguous constraint window");
        parent = record.state.frontier;
    }
    // All copying/allocation and lock acquisition precede publication. Each score
    // move and the final state installation are nonthrowing after reserve.
    auto final_state = records.back().state;
    std::vector<TokenScore> scores;
    if (spec_->logprobs) {
        scores.reserve(records.size());
        for (const auto& record : records) scores.push_back(record.score);
    }
    static_assert(std::is_nothrow_move_constructible_v<TokenScore>);
    static_assert(std::is_nothrow_move_assignable_v<State>);
    const auto install = [&]() noexcept {
        for (auto& score : scores) spec_->scores->tokens.push_back(std::move(score));
        state_ = std::move(final_state);
        state_.frontier->parent.reset();
    };
    if (spec_->logprobs) {
        std::lock_guard lock(spec_->scores->mutex);
        auto& tokens = spec_->scores->tokens;
        if (scores.size() > tokens.max_size() - tokens.size()) throw std::length_error("constraint score storage overflow");
        tokens.reserve(tokens.size() + scores.size());
        install();
    } else install();
}

class ConstraintWindowReceipt final : public MtpSelectionReceipt {
public:
    explicit ConstraintWindowReceipt(std::shared_ptr<ConstraintSelection> selection,
        std::shared_ptr<ConstraintSelection::Frontier> base)
        : selection_(std::move(selection)), base_(std::move(base)) {}
    void publish() override {
        if (published_) throw ConstraintError("already published constraint window");
        selection_->publish_prefix(base_, records_);
        published_ = true;
    }
    std::size_t size() const noexcept override { return records_.size(); }
private:
    friend class ConstraintWindowSelection;
    // Binding the request owner, not a slot address, is the request-generation
    // identity: a receipt can never publish into a replacement slot's selector.
    std::shared_ptr<ConstraintSelection> selection_;
    std::shared_ptr<ConstraintSelection::Frontier> base_;
    std::vector<ConstraintSelection::Staged> records_;
    bool published_ = false;
};

ConstraintSelection::Staged ConstraintSelection::select_row(std::span<const float> row, const State& state) const {
    return select_row(row, state, nullptr, 0);
}
ConstraintSelection::Staged ConstraintSelection::select_row(std::span<const float> row, const State& state,
        ConstraintMaskAhead* ahead, std::size_t slot) const {
    if (argmax_first_ && row.size() == trie_->vocab_size() && !row.empty()) {
        // The host form of the device reduction: lowest-ID maximum (both zeros equal), status 1 on any nonfinite.
        std::uint32_t best = 0, status = 0;
        for (std::uint32_t i = 0; i < row.size(); ++i) {
            if (!std::isfinite(row[i])) { status = 1; break; }
            if (row[i] > row[best]) best = i;
        }
        if (auto known = select_known(state, best, status)) return std::move(*known);
        if (status == 0) if (auto ranked = select_ranked(row, state)) return std::move(*ranked);
    }
    return select(row, prepare(state, ahead, slot));
}
std::unique_ptr<MtpSelectionTransaction> ConstraintSelection::window_transaction() {
    return std::make_unique<ConstraintWindowSelection>(shared_from_this());
}
ConstraintWindowSelection::ConstraintWindowSelection(std::shared_ptr<ConstraintSelection> selection)
    : selection_(std::move(selection)) {
    if (!selection_) throw std::invalid_argument("null constraint selection");
    reset();
}
void ConstraintWindowSelection::reset() {
    auto state = selection_->state();
    discard();
    base_ = state.frontier;
    branch_ = std::move(state);
}
void ConstraintWindowSelection::discard() noexcept {
    records_.clear(); branch_.reset(); base_.reset();
    ConstraintSelection::retire_ahead(ahead_); expected_.clear(); on_path_ = false;
}
ConstraintWindowSelection::~ConstraintWindowSelection() { ConstraintSelection::retire_ahead(ahead_); }
void ConstraintWindowSelection::expect(std::span<const std::uint32_t> drafts) {
    if (!branch_ || !records_.empty()) return;
    ConstraintSelection::retire_ahead(ahead_);
    expected_.assign(drafts.begin(), drafts.end());
    ahead_ = selection_->plan_ahead(*branch_, drafts);
    on_path_ = static_cast<bool>(ahead_);
}
void ConstraintWindowSelection::note_selected(std::uint32_t id) noexcept {
    // records_ already holds this row: the next row is on the planned chain only if every row took its draft.
    const auto row = records_.size() - 1;
    on_path_ = on_path_ && row < expected_.size() && expected_[row] == id;
}
std::uint32_t ConstraintWindowSelection::select(std::span<const float> row) try {
    if (!branch_) throw ConstraintError("inactive constraint window");
    auto staged = selection_->select_row(row, *branch_, on_path_ ? ahead_.get() : nullptr, records_.size());
    auto next = staged.state;
    const auto id = staged.token;
    records_.push_back(std::move(staged));
    *branch_ = std::move(next);
    note_selected(id);
    return id;
} catch(const ConstraintError& e) {throw MtpSelectionError(e.what());}
std::optional<std::uint32_t> ConstraintWindowSelection::select_known(std::uint32_t token, std::uint32_t status) try {
    if (!branch_) throw ConstraintError("inactive constraint window");
    auto staged = selection_->select_known(*branch_, token, status);
    if (!staged) return std::nullopt;
    auto next = staged->state;
    const auto id = staged->token;
    records_.push_back(std::move(*staged));
    *branch_ = std::move(next);
    note_selected(id);
    return id;
} catch(const ConstraintError& e) {throw MtpSelectionError(e.what());}
std::unique_ptr<MtpSelectionReceipt> ConstraintWindowSelection::seal() {
    if (!branch_) throw ConstraintError("inactive or empty constraint window");
    auto receipt = std::make_unique<ConstraintWindowReceipt>(selection_, base_);
    receipt->records_.swap(records_);
    discard();
    return receipt;
}

ConstraintVerdict ConstraintSelection::verdict(std::string finish, std::string execution) const {
    ConstraintVerdict v;
    v.profile_rows = profile_.rows; v.profile_window_rows = profile_.window_rows; v.profile_known_rows = profile_.known_rows;
    v.profile_mask_ms = static_cast<double>(profile_.mask_ns) / 1e6; v.profile_select_ms = static_cast<double>(profile_.select_ns) / 1e6;
    v.profile_window_row_ms = static_cast<double>(profile_.window_row_ns) / 1e6;
    v.profile_ranked_rows = profile_.ranked_rows;
    v.profile_ahead_rows = profile_.ahead_rows; v.profile_ahead_masks = ahead_totals_->masks.load(std::memory_order_relaxed);
    v.profile_ahead_ms = static_cast<double>(ahead_totals_->ns.load(std::memory_order_relaxed)) / 1e6;
    v.reasoning_tokens = state_.reasoning_tokens;
    v.forced_ids = state_.forced_ids;
    v.closure = state_.forced_ids.empty() ? "natural" : "forced";
    v.execution = std::move(execution);
    v.phase = state_.reasoning ? "reasoning" : "answer";
    v.accepting = !state_.reasoning && (!spec_->program || state_.cursor.complete());
    v.status = "incomplete";
    if (finish == "constraint_failure" || finish == "execution_failure") { v.status = "failed"; v.reason = finish; }
    else if (finish == "cancelled" || finish == "context") v.reason = finish;
    else if (state_.terminal && state_.reasoning) v.reason = "reasoning_eos";
    else if (state_.terminal && v.accepting) v.status = "complete";
    else v.reason = "max_output_tokens";
    return v;
}

}  // namespace strix::serve
