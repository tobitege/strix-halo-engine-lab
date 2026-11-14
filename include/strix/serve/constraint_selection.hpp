#pragma once

#include <atomic>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include "strix/mtp_selection_transaction.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/json_constraint.hpp"

namespace strix::serve {

class ConstraintWindowSelection;
class ConstraintWindowReceipt;
// Masks one step ahead. A mask is a pure function of the program, the cursor, the trie and the stop IDs, and
// the cursors a step can reach are known before its logits are: the committed state for a serial row, the
// committed state advanced over drafts 0..r-1 for window row r. plan_ahead() hands those mask walks to a host
// worker (the submit callback: the host lane) while the device runs the step; the row then
// takes the finished mask, waits for the one being built, or builds an unstarted one itself as before. The
// mask words, the errors and the point where they surface are those of prepare(); only who walks changes.
using ConstraintAheadSubmit = std::function<void(std::function<void()>)>;
struct ConstraintMaskAhead;
// Worker-side totals (every walk it ran, used or not); shared with the plans, which may outlive a row.
// `observer` (tracing and tests; set before the first plan, must not throw): on the walking worker with 1 when it
// has claimed a slot, then with the phase it published for it (2 built, 3 abandoned); on the selecting thread
// with 4 when a row finds its walk in flight and is about to wait for it. Calls can run concurrently (several
// workers and the selecting thread) and after the plan was retired or the selection destroyed, for as long as
// submitted jobs are outstanding: whatever the callback captures must be thread-safe and outlive those jobs.
struct ConstraintAheadTotals { std::atomic<std::uint64_t> masks{0}, ns{0}; std::function<void(int)> observer; };
// The serving policy of the two selection levers, read from the environment through the shared registry
// (strict, recorded): STRIX_CONSTRAINT_MASK_AHEAD (on unless exactly "0") and STRIX_CONSTRAINT_RANKED (decimal
// 0..4096, default 0 = off). A value that does not parse throws std::invalid_argument naming the variable.
struct ConstraintSelectionPolicy { bool mask_ahead = true; unsigned ranked = 0; };
[[nodiscard]] ConstraintSelectionPolicy constraint_selection_policy_from_environment();
// A branch is a copy of the committed output frontier. Selection advances only
// that branch; publish it with commit only after the model prefix commits.
class ConstraintSelection : public std::enable_shared_from_this<ConstraintSelection> {
public:
    struct Frontier {
        std::shared_ptr<Frontier> parent;
    };
    struct State {
        std::unordered_map<std::uint32_t, std::uint64_t> reply_counts;
        bool reasoning;
        ConstraintCursor cursor;
        bool terminal = false;
        std::uint32_t reasoning_tokens = 0;
        std::vector<std::uint32_t> forced_ids{};
        std::shared_ptr<Frontier> frontier = std::make_shared<Frontier>();
    };
    struct Prepared {
        State state;
        AllowedTokens mask;
    };
    struct Staged {
        std::uint32_t token;
        State state;
        bool forced_control = false;
        TokenScore score;
    };
    ConstraintSelection(std::shared_ptr<const ConstraintSpec>, std::shared_ptr<const TokenByteTrie>,
        std::shared_ptr<const BpeTokenizer>);
    [[nodiscard]] Prepared prepare() const;
    [[nodiscard]] Prepared prepare(const State&) const;
    [[nodiscard]] Staged select(std::span<const float>, const Prepared&) const;
    // Argmax-first (set_argmax_first(true)): `token` / `status` are the row's unmasked lowest-ID maximum and
    // its finite status (0: every logit finite). When the result provably equals select() on the full row -
    // finite row, no scores or transforms, and the token is the forced/unmasked choice or admitted by the
    // cursor - this stages it without the row; otherwise nullopt and the caller selects on the full row.
    // Every route selects through it (select_row), so serial and window rows behave the same, including
    // mask-walk evaluation limits that only a full mask walk reaches.
    [[nodiscard]] std::optional<Staged> select_known(const State&, std::uint32_t token, std::uint32_t status) const;
    // One row on any route: argmax-first when enabled, else (or on decline) prepare() + select().
    [[nodiscard]] Staged select_row(std::span<const float>, const State&) const;
    // Mask-ahead forms: `slot` of `ahead` holds (or will hold) the mask of this state; a null plan or a slot
    // outside it is the plain call.
    [[nodiscard]] Prepared prepare(const State&, ConstraintMaskAhead* ahead, std::size_t slot) const;
    [[nodiscard]] Staged select_row(std::span<const float>, const State&, ConstraintMaskAhead* ahead, std::size_t slot) const;
    // Slot r = the mask of `base` advanced over path[0..r) (slot 0 = base), for the answer-phase states of
    // that chain; submitted at once (twice for a chain, so two lane threads may share its slots). Null when mask-ahead is off, no slot can need a mask, or the request is
    // in the regime where rows rarely walk one (argmax-first without scores or transforms). A plan is
    // request-local and single-use; retire it (retire_ahead) when its step ends so unstarted walks are skipped.
    [[nodiscard]] std::shared_ptr<ConstraintMaskAhead> plan_ahead(const State& base, std::span<const std::uint32_t> path) const;
    static void retire_ahead(std::shared_ptr<ConstraintMaskAhead>&) noexcept;
    // An empty callback turns mask-ahead off (the default).
    void set_mask_ahead(ConstraintAheadSubmit submit) { ahead_submit_ = std::move(submit); }
    void set_ahead_observer(std::function<void(int)> observer) { ahead_totals_->observer = std::move(observer); }
    // prepare(state).mask.contains(id) for an answer-phase program state (not reasoning, not terminal) and an ID
    // inside the vocabulary, without building the mask; a full walk may additionally stop on evaluation limits.
    [[nodiscard]] bool admits(const State&, std::uint32_t id) const;
    void set_argmax_first(bool on) noexcept { argmax_first_ = on; }
    // Ranked membership (part of argmax-first): when the row's maximum is refused, the next `count` logits in
    // masked-greedy order (value descending, lowest ID first among equals) are checked with admits(); the
    // first admitted one is the masked result, so the row needs no mask walk. 0 (the default) turns it off; a
    // row whose `count` best tokens are all refused takes the full mask. Opt-in because it is not refusal
    // equivalent: a row whose full walk would stop on an evaluation limit reached by a token outside the
    // checked ones selects a token here, where argmax-first alone refuses the request.
    void set_ranked(unsigned count) noexcept { ranked_ = count; }
    void commit(const Staged&);
    [[nodiscard]] std::unique_ptr<MtpSelectionTransaction> window_transaction();
    [[nodiscard]] const State& state() const { return state_; }
    [[nodiscard]] ConstraintVerdict verdict(std::string finish, std::string execution) const;
private:
    friend class ConstraintWindowReceipt;
    friend class ConstraintWindowSelection;
    struct Profile { std::uint64_t rows = 0, window_rows = 0, known_rows = 0, mask_ns = 0, select_ns = 0, window_row_ns = 0, ahead_rows = 0, ranked_rows = 0; };
    std::shared_ptr<ConstraintAheadTotals> ahead_totals_ = std::make_shared<ConstraintAheadTotals>();
    ConstraintAheadSubmit ahead_submit_;
    void advance(Staged&, const State& prior, bool force) const;
    mutable Profile profile_;
    bool argmax_first_ = false;
    unsigned ranked_ = 0;
    [[nodiscard]] std::optional<Staged> select_ranked(std::span<const float>, const State&) const;
    void publish_prefix(const std::shared_ptr<Frontier>&, const std::vector<Staged>&);
    std::shared_ptr<const ConstraintSpec> spec_;
    std::shared_ptr<const TokenByteTrie> trie_;
    std::shared_ptr<const BpeTokenizer> tokenizer_;
    State state_;
};

// One request-owned attempt. Sealing transfers all records to a receipt; reset
// starts a new attempt at the current committed frontier, including after replay.
class ConstraintWindowSelection final : public MtpSelectionTransaction {
public:
    explicit ConstraintWindowSelection(std::shared_ptr<ConstraintSelection>);
    void reset() override;
    void expect(std::span<const std::uint32_t> drafts) override;
    void discard() noexcept override;
    ~ConstraintWindowSelection() override;
    std::uint32_t select(std::span<const float>) override;
    std::optional<std::uint32_t> select_known(std::uint32_t token, std::uint32_t status) override;
    std::unique_ptr<MtpSelectionReceipt> seal() override;
    void note_row_ns(std::uint64_t ns) noexcept override { ++selection_->profile_.window_rows; selection_->profile_.window_row_ns += ns; }
private:
    std::shared_ptr<ConstraintSelection> selection_;
    std::optional<ConstraintSelection::State> branch_;
    std::shared_ptr<ConstraintSelection::Frontier> base_;
    std::vector<ConstraintSelection::Staged> records_;
    // Mask-ahead plan of this attempt: valid for row r while rows 0..r-1 selected expected_[0..r).
    std::shared_ptr<ConstraintMaskAhead> ahead_;
    std::vector<std::uint32_t> expected_;
    bool on_path_ = false;
    void note_selected(std::uint32_t id) noexcept;
};

// One serial output transaction. A per-call target callback stages the token;
// the runner commits it only after receiving the actual model ticket. Destruction
// drops any uncommitted selection (cancellation, failure or slot reuse).
class ConstraintRowSelection {
public:
    explicit ConstraintRowSelection(std::shared_ptr<ConstraintSelection> selection) : selection_(std::move(selection)) {}
    // Starts a row; the mask (when needed) is built with the row, after target execution.
    // With mask-ahead the committed state's mask is walked on the host worker during target execution.
    void prepare() { staged_.reset(); ConstraintSelection::retire_ahead(ahead_); ahead_ = selection_->plan_ahead(selection_->state(), {}); }
    std::uint32_t select(std::span<const float> row) {
        staged_ = std::make_unique<ConstraintSelection::Staged>(selection_->select_row(row, selection_->state(), ahead_.get(), 0));
        ConstraintSelection::retire_ahead(ahead_);
        return staged_->token;
    }
    void commit() {
        if (!staged_) throw std::logic_error("constraint row did not select a token");
        selection_->commit(*staged_); staged_.reset();
    }
    ~ConstraintRowSelection() { ConstraintSelection::retire_ahead(ahead_); }
    [[nodiscard]] std::unique_ptr<MtpSelectionTransaction> window_transaction() {return selection_->window_transaction();}
    void set_execution(std::string route,std::string fallback={}) {execution_=std::move(route);fallback_=std::move(fallback);}
    ConstraintVerdict verdict(const std::string& finish) const {auto v=selection_->verdict(finish,execution_);v.execution_fallback=fallback_;return v;}
private:
    std::shared_ptr<ConstraintSelection> selection_;
    std::string execution_="serial-host",fallback_;
    // Heap-held: GCC 13 -O2 flags an optional<Staged> member as maybe-uninitialized.
    std::unique_ptr<ConstraintSelection::Staged> staged_;
    std::shared_ptr<ConstraintMaskAhead> ahead_;
};

}  // namespace strix::serve
