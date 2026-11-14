#include "strix/mtp_selection_transaction.hpp"
#include "strix/compiled_speculative_verify.hpp"
#include "strix/serve/constraint_selection.hpp"
#include "strix/env_flags.hpp"

#include <functional>
#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>

// Fail individual allocation sites without introducing production test hooks.
thread_local std::ptrdiff_t allocation_countdown = -1;
#if defined(_MSC_VER)
#define TEST_NOINLINE __declspec(noinline)
#else
#define TEST_NOINLINE __attribute__((noinline))
#endif
TEST_NOINLINE void free_test_allocation(void* memory) noexcept { std::free(memory); }
void* operator new(std::size_t size) {
    if (allocation_countdown >= 0 && allocation_countdown-- == 0) { allocation_countdown = -1; throw std::bad_alloc(); }
    if (auto* memory = std::malloc(size ? size : 1)) return memory;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* memory) noexcept { free_test_allocation(memory); }
void operator delete[](void* memory) noexcept { ::operator delete(memory); }
void operator delete(void* memory, std::size_t) noexcept { ::operator delete(memory); }
void operator delete[](void* memory, std::size_t) noexcept { ::operator delete(memory); }
#undef TEST_NOINLINE

namespace {
using namespace strix;
using namespace strix::serve;
std::size_t checks = 0, runs = 0, replays = 0;
std::array<bool, 4> rejection_rows{}, eos_rows{}, closure_rows{};
bool crossing_closure = false, truncated_closure = false, bonus_seen = false;
bool natural_transition = false, reasoning_eos = false, divergent_suffix = false;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) throw std::runtime_error(what);
}
template<class F> void refuses(F&& f) {
    bool failed = false;
    try { f(); } catch (const std::exception&) { failed = true; }
    check(failed, "expected refusal");
}
bool same_double(double a, double b) { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); }
void scores_equal(const std::vector<TokenScore>& a, const std::vector<TokenScore>& b) {
    check(a.size() == b.size(), "score count");
    for (std::size_t i = 0; i < a.size(); ++i) {
        check(a[i].id == b[i].id && a[i].forced_control == b[i].forced_control &&
            a[i].logprob.has_value() == b[i].logprob.has_value() && a[i].top.size() == b[i].top.size(), "score record");
        if (a[i].logprob) check(same_double(*a[i].logprob, *b[i].logprob), "chosen score bits");
        for (std::size_t j = 0; j < a[i].top.size(); ++j)
            check(a[i].top[j].first == b[i].top[j].first && same_double(a[i].top[j].second, b[i].top[j].second), "top score bits");
    }
}
struct Fixture {
    std::shared_ptr<const BpeTokenizer> tokenizer;
    std::shared_ptr<const TokenByteTrie> trie;
    explicit Fixture(std::vector<std::string> tokens) {
        for (auto& token : tokens) if (token == " ") token = "\xC4\xA0";
        tokenizer = BpeTokenizer::from_parts(tokens, std::vector<std::int32_t>(tokens.size(), 1), {}, "qwen35");
        trie = std::make_shared<TokenByteTrie>(*tokenizer);
    }
    std::shared_ptr<ConstraintSelection> selection(const std::shared_ptr<ConstraintSpec>& spec) const {
        return std::make_shared<ConstraintSelection>(spec, trie, tokenizer);
    }
};
std::shared_ptr<ConstraintSpec> copy_spec(const ConstraintSpec& spec) {
    auto result = std::make_shared<ConstraintSpec>(spec);
    result->scores = std::make_shared<SelectionScores>();
    return result;
}
void state_equal(const ConstraintSelection& a, const ConstraintSelection& b, const Fixture& f, const ConstraintSpec& spec) {
    const auto& x = a.state(); const auto& y = b.state();
    check(x.reply_counts == y.reply_counts && x.reasoning == y.reasoning && x.terminal == y.terminal &&
        x.reasoning_tokens == y.reasoning_tokens && x.forced_ids == y.forced_ids, "selection state");
    check(x.cursor.complete() == y.cursor.complete() && x.cursor.dead() == y.cursor.dead(), "cursor status");
    if (spec.program) {
        const auto mx = allowed(*spec.program, x.cursor, *f.trie, spec.eos_ids);
        const auto my = allowed(*spec.program, y.cursor, *f.trie, spec.eos_ids);
        check(mx.words == my.words, "cursor vocabulary behavior");
    }
    for (std::uint32_t id = 0; id < f.tokenizer->vocab_size(); ++id) {
        auto cx = x.cursor, cy = y.cursor;
        bool ax = true, ay = true;
        try { cx = advance_token(cx, f.tokenizer->token_bytes(id)); } catch (const ConstraintError&) { ax = false; }
        try { cy = advance_token(cy, f.tokenizer->token_bytes(id)); } catch (const ConstraintError&) { ay = false; }
        check(ax == ay && cx.complete() == cy.complete() && cx.dead() == cy.dead(), "cursor transitions");
    }
    for (const auto* finish : {"length", "stop", "cancelled", "execution_failure"}) {
        const auto vx = a.verdict(finish, "cpu"), vy = b.verdict(finish, "cpu");
        check(vx.status == vy.status && vx.accepting == vy.accepting && vx.phase == vy.phase && vx.reason == vy.reason &&
            vx.closure == vy.closure && vx.forced_ids == vy.forced_ids && vx.reasoning_tokens == vy.reasoning_tokens, "verdict");
    }
}

// Every consumed input contributes to the hash and to a common exactly representable
// offset. Even when an ordering collides, different consumed prefixes change row bytes.
struct Target {
    std::vector<float> ordering;
    std::optional<std::size_t> eos_at, natural_at;
    std::uint32_t eos = 0, natural = 0;
    std::optional<std::uint32_t> preferred = {};
    std::vector<float> row(const std::vector<std::uint32_t>& consumed) const {
        std::uint64_t hash = 0, code = 0;
        for (auto id : consumed) { hash = hash * 17 + id + 1; code = code * (ordering.size() + 1) + id + 1; }
        std::vector<float> logits(ordering.size());
        for (std::size_t i = 0; i < logits.size(); ++i)
            logits[i] = ordering[(i + hash % logits.size()) % logits.size()] + static_cast<float>(code % 65536) * 0.0000152587890625F;
        if (preferred) logits[*preferred] = 20;
        if (eos_at && consumed.size() == *eos_at) logits[eos] = 100;
        if (natural_at && consumed.size() == *natural_at) logits[natural] = 101;
        return logits;
    }
};
struct Model {
    std::vector<std::uint32_t> consumed{1};
    std::uint32_t pending = 0;
    void consume_pending() { consumed.push_back(pending); }
};
std::vector<std::uint32_t> serial_ids(const Fixture& f, const ConstraintSpec& spec, const Target& target, std::size_t budget) {
    auto s = f.selection(copy_spec(spec)); Model m;
    const auto first = s->select(target.row(m.consumed), s->prepare());
    s->commit(first); m.pending = first.token;
    std::vector<std::uint32_t> ids;
    while (ids.size() + 1 < budget && !s->state().terminal) {
        m.consume_pending();
        auto staged = s->select(target.row(m.consumed), s->prepare());
        m.pending = staged.token; ids.push_back(staged.token); s->commit(staged);
    }
    if (ids.empty()) ids.push_back(first.token); // Only used to fill unreachable proposals.
    return ids;
}

// The serial oracle never uses the acceptance walker, staged window states or
// provisional IDs. Its own consumed prefix drives each fresh target invocation.
void compare_run(const Fixture& f, const ConstraintSpec& spec, const Target& target, std::size_t rows,
    std::size_t budget, const std::vector<std::uint32_t>& proposals, bool checkpoint) {
    ++runs;
    auto ws = copy_spec(spec), ss = copy_spec(spec);
    auto window = f.selection(ws), serial = f.selection(ss);
    Model model, reference;
    // The terminal-prompt selection is already emitted but has not been consumed
    // by target execution. Its counts, score and closure/cursor state are live.
    const auto first = window->select(target.row(model.consumed), window->prepare());
    window->commit(first); model.pending = first.token;
    const auto serial_first = serial->select(target.row(reference.consumed), serial->prepare());
    serial->commit(serial_first); reference.pending = serial_first.token;
    std::vector<std::uint32_t> outputs{first.token}, expected{serial_first.token};
    check(outputs == expected, "terminal-prompt IDs");
    state_equal(*window, *serial, f, spec);
    scores_equal(ws->scores->tokens, ss->scores->tokens);
    while (outputs.size() < budget && !window->state().terminal) {
        auto attempt = window->window_transaction();
        const auto base_frontier = window->state().frontier;
        const auto old_scores = ws->scores->tokens;
        const auto snapshot = model;
        // Capture physical rows on the proposed model branch, including rejected
        // suffixes. The reader must never inspect that suffix.
        std::vector<std::vector<float>> physical;
        std::vector<Model> physical_states;
        auto speculative = model;
        for (std::size_t i = 0; i < rows; ++i) {
            speculative.consume_pending(); physical.push_back(target.row(speculative.consumed));
            physical_states.push_back(speculative);
            if (i < proposals.size()) speculative.pending = proposals[i];
        }
        std::size_t reads = 0;
        std::uint32_t previous = 0;
        const auto remaining = budget - outputs.size();
        const auto plan = walk_mtp_selection(proposals, rows, remaining, spec.eos_ids, [&](std::size_t row) {
            check(row == reads, "ordered reader");
            if (row) check(proposals[row - 1] == previous, "reader reached only accepted inputs");
            ++reads;
            previous = attempt->select(physical.at(row));
            return previous;
        });
        check(reads == plan.decided_rows && reads == plan.selected_ids.size() && reads > 0, "explicit decided rows");
        check(window->state().frontier == base_frontier, "selection remains private");
        scores_equal(old_scores, ws->scores->tokens);
        check(plan.accepted_draft_inputs == reads - 1, "accepted consumed draft inputs");
        const auto last = plan.selected_ids.back();
        const bool eos = std::find(spec.eos_ids.begin(), spec.eos_ids.end(), last) != spec.eos_ids.end();
        const auto stop = eos ? MtpSelectionStop::eos : reads == remaining ? MtpSelectionStop::output_limit :
            reads == rows ? MtpSelectionStop::row_limit : reads > proposals.size() ? MtpSelectionStop::no_draft : MtpSelectionStop::rejection;
        check(plan.stop == stop, "stop reason precedence");
        if (plan.stop == MtpSelectionStop::rejection) {
            check(proposals[reads - 1] != last, "correction retained");
            rejection_rows[reads - 1] = true;
        }
        if (eos) eos_rows[reads - 1] = true;
        if (reads == rows && reads == proposals.size() + 1) bonus_seen = true;
        if (window->state().reasoning && !window->state().forced_ids.empty()) crossing_closure = true;
        if (checkpoint) model = speculative;
        const bool replay = checkpoint && (reads != rows || plan.accepted_draft_inputs + 1 != rows);
        std::unique_ptr<MtpSelectionReceipt> receipt;
        if (replay) {
            ++replays;
            model = snapshot;
            attempt->discard();
            scores_equal(old_scores, ws->scores->tokens);
            check(window->state().frontier == base_frontier, "restore before replay");
            attempt->reset();
            for (std::size_t i = 0; i < reads; ++i) {
                model.consume_pending();
                model.pending = attempt->select(target.row(model.consumed));
                check(model.pending == plan.selected_ids[i], "corrected serial replay");
            }
            if (plan.stop == MtpSelectionStop::rejection && reads < rows) {
                auto corrected = model; corrected.consume_pending();
                check(target.row(corrected.consumed) != physical[reads], "rejected suffix differs from corrected continuation");
                divergent_suffix = true;
            }
            receipt = attempt->seal();
        } else {
            if (checkpoint) check(model.consumed == physical_states.back().consumed, "complete checkpoint chunk retained");
            else model = physical_states[reads - 1]; // Commit the captured model input prefix.
            model.pending = plan.selected_ids.back();
            receipt = attempt->seal();
        }
        check(receipt->size() == reads, "receipt length");
        scores_equal(old_scores, ws->scores->tokens);
        for (std::size_t i = 0; i < reads; ++i) {
            reference.consume_pending();
            const auto staged = serial->select(target.row(reference.consumed), serial->prepare());
            if (staged.forced_control && serial->state().forced_ids.empty()) closure_rows[i] = true;
            if (!staged.forced_control && serial->state().reasoning && !staged.state.reasoning) natural_transition = true;
            if (staged.state.terminal && staged.state.reasoning) reasoning_eos = true;
            reference.pending = staged.token; expected.push_back(staged.token); serial->commit(staged);
        }
        receipt->publish();
        outputs.insert(outputs.end(), plan.selected_ids.begin(), plan.selected_ids.end());
        check(outputs == expected && model.pending == reference.pending && model.consumed == reference.consumed, "complete output and model frontier");
        state_equal(*window, *serial, f, spec);
        if (outputs.size() == budget && window->state().reasoning && !window->state().forced_ids.empty()) truncated_closure = true;
        scores_equal(ws->scores->tokens, ss->scores->tokens);
        refuses([&] { receipt->publish(); });
        scores_equal(ws->scores->tokens, ss->scores->tokens);
    }
}

void enumeration() {
    const Fixture f({"a", "b", "c"});
    constexpr std::array<float, 4> values{-1, -0.0F, 0.0F, 1};
    for (unsigned ordering = 0; ordering < 64; ++ordering) {
        Target target{{values[ordering % 4], values[(ordering / 4) % 4], values[ordering / 16]}, {}, {}, 0, 0};
        for (unsigned policy = 0; policy < 4; ++policy) {
            ConstraintSpec spec;
            spec.logprobs = policy != 1; spec.top_logprobs = 3;
            if (policy & 1U) { spec.transform.frequency = .75; spec.transform.presence = .25;
                spec.transform.repetition = 2; spec.transform.bias = {{1, .5}}; spec.prompt_ids = {2}; }
            if (policy & 2U) spec.eos_ids = {2};
            for (std::size_t rows = 1; rows <= 4; ++rows) {
                std::size_t count = 1;
                for (std::size_t i = 1; i < rows; ++i) count *= 3;
                for (std::size_t code = 0; code < count; ++code) {
                    auto digits = code;
                    std::vector<std::uint32_t> proposals(rows - 1);
                    for (auto& id : proposals) { id = static_cast<std::uint32_t>(digits % 3); digits /= 3; }
                    for (bool checkpoint : {false, true}) compare_run(f, spec, target, rows, 6, proposals, checkpoint);
                }
                // Enumerate all match/reject bit patterns independently, including
                // deliberately unreachable suffix bits after the first rejection.
                const auto ids = serial_ids(f, spec, target, 6);
                for (unsigned bits = 0; bits < (1U << (rows - 1)); ++bits) {
                    std::vector<std::uint32_t> proposals;
                    for (std::size_t i = 0; i + 1 < rows; ++i) {
                        auto id = ids[i % ids.size()];
                        proposals.push_back((bits & (1U << i)) ? id : (id + 1) % 3);
                    }
                    for (bool checkpoint : {false, true}) compare_run(f, spec, target, rows, 6, proposals, checkpoint);
                }
            }
        }
    }
}

void grammar_and_thinking() {
    const Fixture f({"a", "b", "{", "}", "x", "y", "z", " "});
    for (unsigned mask = 0; mask < 3; ++mask) for (unsigned policy = 0; policy < 3; ++policy) {
        ConstraintSpec spec; spec.eos_ids = {6}; spec.end_thinking_id = 4;
        spec.logprobs = policy != 0; spec.top_logprobs = 8;
        if (policy != 1) { spec.transform.presence = .5; spec.transform.frequency = 1; spec.transform.repetition = 2; spec.prompt_ids = {0}; }
        if (mask == 1) spec.program = std::make_shared<ConstraintProgram>(compile_json_object());
        if (mask == 2) spec.program = std::make_shared<ConstraintProgram>(compile_json_schema(parse_json("{\"const\":{}}")));
        for (std::size_t rows = 1; rows <= 4; ++rows) for (std::size_t start = 0; start <= 4; ++start)
            for (std::size_t budget = 1; budget <= 8; ++budget) for (unsigned event = 0; event < 4; ++event) {
                spec.starts_in_reasoning = true;
                spec.thinking = ThinkingPolicy{0, 0, static_cast<std::uint32_t>(start), {4, 5, 5}};
                Target target{{2, 1, 0, -1, -2, -3, -4, -5}, {}, {}, 6, 4};
                target.preferred = 0;
                if (event == 1) target.eos_at = std::max<std::size_t>(1, start);
                if (event == 2) target.natural_at = std::max<std::size_t>(1, start);
                if (event == 3) { target.eos_at = start + 2; target.natural_at = 2; }
                const auto ids = serial_ids(f, spec, target, budget);
                for (std::size_t reject = 0; reject <= rows; ++reject) {
                    std::vector<std::uint32_t> proposals;
                    for (std::size_t i = 0; i + 1 < rows; ++i) {
                        const auto id = ids[i % ids.size()];
                        proposals.push_back(i == reject ? (id + 1) % 8 : id);
                    }
                    for (bool checkpoint : {false, true}) compare_run(f, spec, target, rows, budget, proposals, checkpoint);
                }
            }
    }
}

void masked_enumeration() {
    const Fixture f({"{", "}", " ", "z"});
    constexpr std::array<float, 4> values{-1, -0.0F, 0.0F, 1};
    for (unsigned mask = 0; mask < 2; ++mask) for (unsigned ordering = 0; ordering < 64; ++ordering) {
        ConstraintSpec spec; spec.eos_ids = {3}; spec.logprobs = true; spec.top_logprobs = 4;
        spec.program = std::make_shared<ConstraintProgram>(mask ? compile_json_object() : compile_json_schema(parse_json("{\"const\":{}}")));
        spec.transform.bias = {{3, 5}};
        Target target{{values[ordering % 4], values[(ordering / 4) % 4], values[ordering / 16], 10}, {}, {}, 3, 0};
        for (std::size_t rows = 1, count = 1; rows <= 4; ++rows, count *= 4) {
            for (std::size_t code = 0; code < count; ++code) {
                std::vector<std::uint32_t> proposals(rows - 1);
                auto digits = code;
                for (auto& id : proposals) { id = static_cast<std::uint32_t>(digits % 4); digits /= 4; }
                for (bool checkpoint : {false, true}) compare_run(f, spec, target, rows, 6, proposals, checkpoint);
            }
        }
    }
}

void allocation_failures() {
    const Fixture f({"a", "b", "c"});
    const std::array<float, 3> row{2, 1, 0};
    for (bool scoring : {false, true}) {
        auto spec = std::make_shared<ConstraintSpec>(); spec->logprobs = scoring; spec->top_logprobs = 3; spec->transform.frequency = 1;
        auto s = f.selection(spec);
        s->commit(s->select(row, s->prepare()));
        auto attempt = s->window_transaction();
        for (unsigned i = 0; i < 4; ++i) attempt->select(row);
        auto receipt = attempt->seal();
        const auto frontier = s->state().frontier;
        const auto before = spec->scores->tokens;
        bool published = false;
        std::size_t failures = 0;
        for (std::ptrdiff_t allocation = 0; allocation < 100 && !published; ++allocation) {
            allocation_countdown = allocation;
            try { receipt->publish(); published = true; } catch (const std::bad_alloc&) { ++failures; }
            allocation_countdown = -1;
            if (!published) {
                check(s->state().frontier == frontier && s->state().reply_counts.size() == 1, "publication allocation failure preserves state");
                scores_equal(before, spec->scores->tokens);
            }
        }
        check(published && failures > 0 && spec->scores->tokens.size() == (scoring ? 5U : 0U), "allocation failures then atomic publication");
        // Reset, row staging and sealing also leave the live request untouched on
        // setup failure; the caller discards any surviving private attempt.
        for (std::ptrdiff_t allocation = 0; allocation < 30; ++allocation) {
            auto tx = s->window_transaction();
            const auto base = s->state().frontier;
            const auto scores = spec->scores->tokens;
            allocation_countdown = allocation;
            try { tx->reset(); tx->select(row); auto unused = tx->seal(); } catch (const std::bad_alloc&) {}
            allocation_countdown = -1;
            check(s->state().frontier == base, "setup failure preserves state");
            scores_equal(scores, spec->scores->tokens);
            tx->discard();
        }
    }
}

void boundaries() {
    const Fixture f({"a", "b", "c"});
    auto spec = std::make_shared<ConstraintSpec>(); spec->logprobs = true; spec->top_logprobs = 3;
    spec->transform.frequency = 1;
    auto selection = f.selection(spec);
    for (const auto& zeros : {std::array<float, 3>{-0.0F, 0.0F, -1}, std::array<float, 3>{0.0F, -0.0F, -1}}) {
        auto signed_spec = copy_spec(*spec); signed_spec->transform = {};
        auto signed_selection = f.selection(signed_spec); auto t = signed_selection->window_transaction();
        check(t->select(zeros) == 0, "signed zero tie lowest ID"); auto r = t->seal(); r->publish();
        auto serial = f.selection(copy_spec(*signed_spec)); serial->commit(serial->select(zeros, serial->prepare()));
        check(signed_spec->scores->tokens[0].top[0].first == 0 && signed_spec->scores->tokens[0].top[1].first == 1, "signed zero top ordering");
    }
    const std::vector<float> row{2, 1.5F, 0};
    auto a = selection->window_transaction(), b = selection->window_transaction();
    check(a->select(row) == 0 && a->select(row) == 1, "penalty counts selected outputs only");
    check(spec->scores->tokens.empty(), "private multirow scores");
    auto receipt = a->seal();
    check(b->select(row) == 0, "independent attempt starts at live frontier");
    auto stale = b->seal(); receipt->publish();
    check(spec->scores->tokens.size() == 2 && spec->scores->tokens[0].id == 0 && spec->scores->tokens[1].id == 1, "every score exactly once");
    auto scores = spec->scores->tokens;
    refuses([&] { stale->publish(); }); refuses([&] { receipt->publish(); });
    scores_equal(scores, spec->scores->tokens);
    refuses([&] { a->select(row); }); refuses([&] { a->seal(); });
    b->reset(); b->select(row); b->discard();
    scores_equal(scores, spec->scores->tokens);
    b->reset();
    for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
        auto invalid = row; invalid[2] = bad;
        refuses([&] { b->select(invalid); });
        scores_equal(scores, spec->scores->tokens);
    }
    refuses([&] { b->select(std::span<const float>(row.data(), 2)); });
    b->select(row); auto valid = b->seal(); valid->publish();
    check(spec->scores->tokens.size() == 3, "failed rows stage no scores");
    // Setup failure and cancellation drop attempts/receipts without publication.
    scores = spec->scores->tokens;
    try { auto setup = selection->window_transaction(); setup->select(row); throw std::runtime_error("setup"); } catch (const std::runtime_error&) {}
    { auto cancel = selection->window_transaction(); cancel->select(row); auto cancelled = cancel->seal(); }
    scores_equal(scores, spec->scores->tokens);
    auto reused = f.selection(copy_spec(*spec));
    auto old = selection->window_transaction(); old->select(row); auto old_receipt = old->seal();
    selection.reset(); old_receipt->publish();
    check(reused->state().reply_counts.empty(), "slot reuse binds receipt to original request");

    const Target target{{2, 1, 0}, {}, {}, 2, 0};
    ConstraintSpec short_spec; short_spec.logprobs = true; short_spec.top_logprobs = 3;
    for (bool checkpoint : {false, true}) compare_run(f, short_spec, target, 4, 5, {}, checkpoint);
    auto interrupted = reused->window_transaction();
    const auto live = reused->state().frontier;
    refuses([&] { (void)walk_mtp_selection(std::array<std::uint32_t, 2>{0, 0}, 3, 3, {}, [&](std::size_t row_index) {
        if (row_index == 1) throw std::runtime_error("row read failure");
        return interrupted->select(std::array<float, 3>{10, 0, -1});
    }); });
    interrupted->discard();
    check(reused->state().frontier == live && reused->state().reply_counts.empty(), "interrupted walker discards private branch");

    std::size_t reads = 0;
    for (std::size_t rows = 0; rows <= 4; ++rows) {
        auto plan = walk_mtp_selection({}, rows, 0, {}, [&](auto) { ++reads; return 0U; });
        check(plan.decided_rows == 0 && plan.stop == MtpSelectionStop::output_limit, "zero budget");
    }
    auto zero = walk_mtp_selection({}, 0, 5, {}, [&](auto) { ++reads; return 0U; });
    check(zero.decided_rows == 0 && reads == 0, "zero rows never read");
    auto bonus = walk_mtp_selection({}, 4, 5, {}, [&](auto) { ++reads; return 0U; });
    check(bonus.decided_rows == 1 && bonus.stop == MtpSelectionStop::no_draft && reads == 1, "no proposals bonus");
    // Poison every unreachable row, for each termination boundary.
    for (std::size_t stop = 0; stop < 4; ++stop) for (unsigned reason = 0; reason < 4; ++reason) {
        const std::array<std::uint32_t, 3> proposals{0, 0, 0};
        const std::array<std::uint32_t, 1> eos{1};
        reads = 0;
        auto plan = walk_mtp_selection(proposals, reason == 2 ? stop + 1 : 4, reason == 1 ? stop + 1 : 8,
            reason == 0 ? std::span<const std::uint32_t>(eos) : std::span<const std::uint32_t>{}, [&](std::size_t i) {
                check(i <= stop, "unreachable poison row reader"); ++reads;
                return i == stop && (reason == 0 || reason == 3) ? 1U : 0U;
            });
        check(plan.decided_rows == stop + 1 && reads == stop + 1, "reachability boundary");
    }
}

void routes_and_reuse() {
    CompiledKnownPrefixWindow owner;
    for(const auto rows:{2U,4U,8U}) {
        owner.logical_rows=rows;
        for(unsigned variant=0;variant<=33U;++variant) {
            owner.r2_variant={static_cast<KnownWindowR2Variant>(variant)};
            check(known_window_policy_serial_rows(owner)==(variant!=2U && variant<=32U),"closed arithmetic recipe admission");
        }
    }
    owner.r2_variant={};owner.logical_rows=16;
    check(!known_window_policy_serial_rows(owner),"tiled owner is not a decode policy owner");
    owner.logical_rows=2;owner.output_policy=KnownWindowOutputPolicy::state_only;
    check(!known_window_policy_serial_rows(owner),"policy needs every reachable full-logit row");
    for(bool enabled:{false,true})for(bool window:{false,true})for(bool serial:{false,true}) {
        const auto route=resolve_mtp_selection_route(enabled,window,serial);
        check(route.mtp==(enabled && window && serial),"route admission");
        const std::string expected=!enabled?"selection-mtp-off":!window?"no-window-owner":!serial?"window-arithmetic-not-serial":"";
        check(std::string(route.fallback?route.fallback:"")==expected,"route fallback precedence");
    }
    const Fixture f({"a","b","c"});
    for(unsigned policy=0;policy<3;++policy) {
        auto spec=std::make_shared<ConstraintSpec>();spec->logprobs=policy!=1;
        if(policy)spec->transform.presence=1;
        auto old=f.selection(spec);ConstraintRowSelection adapter(old);
        adapter.prepare();check(adapter.select(std::array<float,3>{3,2,1})==0,"terminal prompt selection");adapter.commit();
        const auto committed=old->state().frontier;
        adapter.set_execution("mtp-host");
        auto attempt=adapter.window_transaction();attempt->select(std::array<float,3>{1,3,2});
        auto cancelled=attempt->seal();cancelled.reset();
        check(old->state().frontier==committed,"cancelled receipt leaves committed prompt only");
        auto replacement_spec=copy_spec(*spec);auto replacement=f.selection(replacement_spec);
        ConstraintRowSelection reused(replacement);reused.set_execution("serial-host","selection-mtp-off");
        auto stale=adapter.window_transaction();stale->select(std::array<float,3>{1,2,3});auto receipt=stale->seal();
        const auto frontier=replacement->state().frontier;receipt->publish();
        check(replacement->state().frontier==frontier && replacement_spec->scores->tokens.empty(),"old receipt cannot target reused request");
        refuses([&]{receipt->publish();});
        check(adapter.verdict("length").execution=="mtp-host" && reused.verdict("length").execution_fallback=="selection-mtp-off","request route lifecycle");
        auto next=reused.window_transaction();next->select(std::array<float,3>{1,3,2});next->seal()->publish();
        check(replacement->state().frontier!=frontier && (!policy || replacement->state().reply_counts.at(1)==1),"reuse starts from its own frontier");
    }
}

void selection_failures() {
    const Fixture f({"a","b","c"});
    for(std::size_t fail=0;fail<4;++fail)for(unsigned kind=0;kind<4;++kind) {
        auto spec=std::make_shared<ConstraintSpec>();spec->logprobs=true;spec->top_logprobs=3;
        if(kind==2)spec->transform.bias.emplace_back(0U,std::numeric_limits<float>::max());
        auto serial_spec=copy_spec(*spec);auto serial=f.selection(serial_spec);
        auto selection=f.selection(spec);auto tx=selection->window_transaction();
        const auto base=selection->state().frontier;
        const std::array<float,3> good{3,2,1};
        std::size_t reads=0;
        const auto plan=walk_mtp_selection(std::array<std::uint32_t,3>{0,0,0},4,4,{},[&](std::size_t row) {
            check(row<=fail,"failure suffix unreachable");++reads;
            if(row==fail) {
                if(kind==0) {auto bad=good;bad[1]=std::numeric_limits<float>::infinity();return tx->select(bad);}
                if(kind==1) {
                    auto empty_spec=std::make_shared<ConstraintSpec>();
                    empty_spec->program=std::make_shared<ConstraintProgram>(compile_json_object());
                    return f.selection(empty_spec)->window_transaction()->select(good);
                }
                if(kind==2) {auto bad=good;bad[0]=std::numeric_limits<float>::max();return tx->select(bad);}
                throw MtpSelectionError("");
            }
            return tx->select(good);
        });
        check(plan.stop==MtpSelectionStop::selection_failure && plan.selection_failure.has_value(),"explicit selection failure");
        if(kind==3)check(plan.selection_failure->empty(),"empty error text still carries failure");
        check(reads==fail+1 && plan.decided_rows==fail && plan.selected_ids==std::vector<std::uint32_t>(fail,0),"failure retains only decided prefix");
        check(spec->scores->tokens.empty() && selection->state().frontier==base,"failure never partially publishes");
        auto receipt=tx->seal();check(receipt->size()==fail,"failure receipt prefix length");
        for(std::size_t i=0;i<fail;++i)serial->commit(serial->select(good,serial->prepare()));
        receipt->publish();scores_equal(spec->scores->tokens,serial_spec->scores->tokens);
        state_equal(*selection,*serial,f,*spec);
        tx->reset();check(tx->select(good)==0,"failed transaction reusable after reset");
        tx->seal()->publish();serial->commit(serial->select(good,serial->prepare()));
        scores_equal(spec->scores->tokens,serial_spec->scores->tokens);
        state_equal(*selection,*serial,f,*spec);
    }
    // A failed member neither throws out of the shared dispatch nor changes peers.
    for(std::size_t member=0;member<3;++member) {
        const auto plan=walk_mtp_selection(std::array<std::uint32_t,3>{0,0,0},4,4,{},[&](std::size_t row) {
            if(member==1 && row==2)throw MtpSelectionError("member failure");
            return 0U;
        });
        check(plan.decided_rows==(member==1?2U:4U),"cohort failure isolation");
    }
    refuses([&]{walk_mtp_selection({},1,1,{},[](auto)->std::uint32_t{throw std::logic_error("execution invariant");});});
}

void masks_and_cohorts() {
    const Fixture f({"a", "{", "}", "z"});
    auto spec = std::make_shared<ConstraintSpec>(); spec->program = std::make_shared<ConstraintProgram>(compile_json_object());
    spec->logprobs = true; spec->eos_ids = {3};
    auto s = f.selection(spec); auto tx = s->window_transaction();
    std::vector<float> row{100, 2, 1, 90};
    check(tx->select(row) == 1, "masked maximum");
    auto bad = row; bad[0] = std::numeric_limits<float>::quiet_NaN();
    refuses([&] { tx->select(bad); });
    auto receipt = tx->seal(); receipt->publish();
    check(spec->scores->tokens.size() == 1, "reachable nonfinite masked logit stages nothing");
    const Fixture empty({"a", "b"});
    auto es = std::make_shared<ConstraintSpec>(); es->program = spec->program;
    auto empty_selection = empty.selection(es); auto et = empty_selection->window_transaction();
    const auto frontier = empty_selection->state().frontier;
    refuses([&] { et->select(std::array<float, 2>{1, 0}); });
    check(empty_selection->state().frontier == frontier && es->scores->tokens.empty(), "empty mask atomic refusal");

    std::array<std::shared_ptr<ConstraintSelection>, 4> members;
    std::array<std::unique_ptr<MtpSelectionTransaction>, 4> attempts;
    std::array<std::unique_ptr<MtpSelectionReceipt>, 4> receipts;
    for (std::size_t i = 0; i < 3; ++i) { members[i] = f.selection(copy_spec(*spec)); attempts[i] = members[i]->window_transaction(); }
    check(!attempts[3], "ordinary cohort member stays null");
    for (std::size_t i = 0; i < 3; ++i) {
        const auto plan = walk_mtp_selection({}, 1, 1, spec->eos_ids, [&](auto) { return attempts[i]->select(row); });
        check(plan.selected_ids == std::vector<std::uint32_t>{1}, "independent cohort plan");
        receipts[i] = attempts[i]->seal();
    }
    for (std::size_t i : {2U, 0U, 1U}) {
        receipts[i]->publish();
        for (std::size_t j = 0; j < 3; ++j) if (j != i && attempts[j])
            check(!members[j]->state().cursor.complete(), "cohort cursor isolated");
        attempts[i].reset();
    }
    row = {100, 1, 2, 90};
    for (auto& member : members) if (member) {
        auto t = member->window_transaction(); check(t->select(row) == 2, "independent advancing cursor"); auto r = t->seal(); r->publish();
        check(member->state().cursor.complete(), "cohort completed object");
    }
}
}  // namespace

// Argmax-first selection: admits() equals the prepared mask for every ID, select_row() (argmax-first on)
// stages exactly what prepare() + select() stage on the full row, and a window mixing known and
// full-row selections publishes the same frontier and scores as serial selection.
std::size_t known_answer = 0, known_forced = 0, known_reasoning = 0, known_declined = 0;
bool same_state(const ConstraintSelection::State& x, const ConstraintSelection::State& y, const ConstraintSpec& spec, const TokenByteTrie& trie) {
    if (!(x.reply_counts == y.reply_counts && x.reasoning == y.reasoning && x.terminal == y.terminal &&
          x.reasoning_tokens == y.reasoning_tokens && x.forced_ids == y.forced_ids &&
          x.cursor.complete() == y.cursor.complete() && x.cursor.dead() == y.cursor.dead())) return false;
    if (spec.program && !x.terminal)
        return allowed(*spec.program, x.cursor, trie, spec.eos_ids).words == allowed(*spec.program, y.cursor, trie, spec.eos_ids).words;
    return true;
}
void argmax_first() {
    // Multi-byte tokens; 9 = EOS (excluded from the trie, as in serving), 10 = end of thinking.
    const std::vector<std::string> vocab{"{", "}", "\"a\"", ":", "12", ",", "\"", "{\"", "\":", "z", "x", "1}", "a"};
    const std::uint32_t eos = 9, end_thinking = 10, n = static_cast<std::uint32_t>(vocab.size());
    Fixture f(vocab);
    const std::array<std::uint32_t, 1> excluded{eos};
    f.trie = std::make_shared<TokenByteTrie>(*f.tokenizer, excluded);
    for (std::uint32_t id = 0; id < n; ++id) check(f.trie->contains(id) == (id != eos), "trie membership");
    check(!f.trie->contains(n) && !f.trie->contains(UINT32_MAX), "trie membership outside vocabulary");
    const std::array<const char*, 3> schemas{"", "{\"const\":{}}", "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}}}"};
    constexpr std::array<float, 7> values{-1, -0.0F, 0.0F, 1, 2, 2, 3};
    std::uint64_t seed = 12345;
    const auto next = [&] { seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<std::uint32_t>(seed >> 33); };
    const auto random_row = [&] { std::vector<float> row(n); for (auto& v : row) v = values[next() % values.size()]; return row; };
    const auto lowest_max = [](const std::vector<float>& row) {
        std::uint32_t g = 0;
        for (std::uint32_t i = 1; i < row.size(); ++i) if (row[i] > row[g]) g = i;
        return g;
    };
    for (unsigned program = 0; program < 3; ++program) for (unsigned policy = 0; policy < 4; ++policy) for (unsigned thinking = 0; thinking < 2; ++thinking) {
        ConstraintSpec spec; spec.eos_ids = {eos}; spec.end_thinking_id = end_thinking;
        spec.program = std::make_shared<ConstraintProgram>(program ? compile_json_schema(parse_json(schemas[program])) : compile_json_object());
        spec.logprobs = policy == 1; spec.top_logprobs = 3;
        if (policy == 2) spec.transform.bias = {{1, 0.5}};
        if (policy == 3) spec.program.reset();
        if (thinking) { spec.starts_in_reasoning = true; spec.thinking = ThinkingPolicy{0, 0, 3, {end_thinking, 5}}; }
        for (unsigned walk = 0; walk < 16; ++walk) {
            auto fast = f.selection(copy_spec(spec)), full = f.selection(copy_spec(spec));
            fast->set_argmax_first(true);
            for (unsigned depth = 0; depth < 16 && !fast->state().terminal; ++depth) {
                const auto& state = fast->state();
                check(same_state(state, full->state(), spec, *f.trie), "fast and full frontiers agree");
                std::optional<ConstraintSelection::Prepared> prepared;
                try { prepared = fast->prepare(state); } catch (const ConstraintError&) {}
                if (spec.program && !state.reasoning)
                    for (std::uint32_t id = 0; id < n; ++id)
                        check(fast->admits(state, id) == (prepared && prepared->mask.contains(id)), "admits equals the prepared mask");
                // the off switch never takes the known path
                check(!full->select_known(full->state(), lowest_max(random_row()), 0), "argmax-first off declines");
                std::vector<float> chosen;
                for (unsigned trial = 0; trial < 24; ++trial) {
                    auto row = random_row();
                    const auto g = lowest_max(row);
                    check(!fast->select_known(state, g, 1), "nonfinite status declines");
                    const auto known = fast->select_known(state, g, 0);
                    if (spec.logprobs || spec.transform.active()) check(!known, "scores and transforms decline");
                    if (known) {
                        const bool force = state.reasoning && spec.thinking && state.reasoning_tokens >= spec.thinking->reasoning_limit;
                        if (force) ++known_forced; else if (state.reasoning || !spec.program) ++known_reasoning; else ++known_answer;
                    } else ++known_declined;
                    // select_row (any route) equals prepare + select on the full row, including refusals
                    std::optional<ConstraintSelection::Staged> a, b;
                    bool a_failed = false, b_failed = false;
                    try { a = fast->select_row(row, state); } catch (const ConstraintError&) { a_failed = true; }
                    try { b = full->select(row, full->prepare(full->state())); } catch (const ConstraintError&) { b_failed = true; }
                    check(a_failed == b_failed, "select_row refuses exactly when the full row refuses");
                    if (a && b) {
                        check(a->token == b->token && a->forced_control == b->forced_control && same_state(a->state, b->state, spec, *f.trie),
                            "select_row equals full-row selection");
                        if (known) check(known->token == a->token, "known token equals select_row");
                        chosen = row;
                    }
                    auto bad = row; bad[next() % n] = std::numeric_limits<float>::quiet_NaN();
                    bool fast_nan = false, full_nan = false;
                    try { (void)fast->select_row(bad, state); } catch (const ConstraintError&) { fast_nan = true; }
                    try { (void)full->select(bad, full->prepare(full->state())); } catch (const ConstraintError&) { full_nan = true; }
                    check(fast_nan && full_nan, "nonfinite rows refuse on both paths");
                }
                if (chosen.empty()) break;
                try {
                    fast->commit(fast->select_row(chosen, fast->state()));
                    full->commit(full->select(chosen, full->prepare(full->state())));
                } catch (const ConstraintError&) { break; }
            }
            // A window mixing known and full-row selections publishes what serial selection commits.
            const auto serial_spec = copy_spec(spec), window_spec = copy_spec(spec);
            auto serial = f.selection(serial_spec), windowed = f.selection(window_spec);
            serial->set_argmax_first(true); windowed->set_argmax_first(true);
            auto t = windowed->window_transaction();
            std::size_t staged = 0;
            for (unsigned row_index = 0; row_index < 6 && !serial->state().terminal; ++row_index) {
                const auto row = random_row();
                std::optional<ConstraintSelection::Staged> s;
                try { s = serial->select_row(row, serial->state()); } catch (const ConstraintError&) { break; }
                std::uint32_t id = 0;
                if (row_index % 2 == 0) {
                    auto k = t->select_known(lowest_max(row), 0);
                    id = k ? *k : t->select(row);
                } else id = t->select(row);
                check(id == s->token, "window row equals serial row");
                serial->commit(*s); ++staged;
            }
            if (staged) {
                auto receipt = t->seal();
                check(receipt->size() == staged, "window receipt size");
                receipt->publish();
                check(same_state(windowed->state(), serial->state(), spec, *f.trie), "published window equals serial frontier");
                scores_equal(window_spec->scores->tokens, serial_spec->scores->tokens);
            }
        }
    }
    check(known_answer > 100 && known_forced > 10 && known_reasoning > 100 && known_declined > 1000, "argmax-first coverage");
}

// Mask-ahead: the worker only changes who walks a mask. Three executors cover the three ways a row meets its
// slot: built before the row (inline executor), never started (dropped job: the row builds it), and a real
// thread racing the rows.
// A schema that compiles within its evaluation budget but whose state after `[{` exhausts it on a quote
// (the "candidate traversal" fixture of the constraint tests): every transition up to `[{` is cheap.
std::shared_ptr<const ConstraintProgram> budgeted_program() {
    Json properties = Json::object(), required = Json::array();
    for (unsigned i = 0; i < 128; ++i) {
        char index[4]; std::snprintf(index, sizeof(index), "%03u", i);
        const auto name = std::string(50, 'a') + index;
        properties.set(name, true); required.push(name);
    }
    Json object = Json::object(); object.set("type", "object"); object.set("properties", properties);
    object.set("required", required); object.set("additionalProperties", false);
    Json choices = Json::array(), arrays = Json::array();
    for (int i = 0; i < 8; ++i) {
        Json choice = Json::object(); choice.set("$ref", "#/$defs/Required"); choice.set("minItems", i);
        choices.push(choice);
    }
    for (int i = 0; i < 512; ++i) {
        Json array = Json::object(); array.set("type", "array"); array.set("minItems", i);
        array.set("items", parse_json(R"({"$ref":"#/$defs/Choices"})")); arrays.push(array);
    }
    Json choice_schema = Json::object(); choice_schema.set("anyOf", choices);
    Json defs = Json::object(); defs.set("Required", object); defs.set("Choices", choice_schema);
    Json branched = Json::object(); branched.set("$defs", defs); branched.set("anyOf", arrays);
    ConstraintLimits limits; limits.compilation_steps = 400000; limits.max_output_bytes = 16 * 1048576;
    return std::make_shared<ConstraintProgram>(compile_json_schema(branched, limits));
}
// What a selection did with a row: the token, or the text of its refusal.
template<class F> std::string outcome(F&& f) {
    try { return "token " + std::to_string(f()); } catch (const std::exception& e) { return std::string("error ") + e.what(); }
}
void speculation_and_limits() {
    const std::vector<std::string> vocab{"[", "{", "\"", "]", " ", "z", "b"};
    const std::uint32_t open = 0, brace = 1, quote = 2, close = 3, space = 4, eos = 5, letter = 6;
    Fixture f(vocab);
    const std::array<std::uint32_t, 1> excluded{eos};
    f.trie = std::make_shared<TokenByteTrie>(*f.tokenizer, excluded);
    ConstraintSpec spec; spec.eos_ids = {eos}; spec.program = budgeted_program();
    const auto favor = [&](std::uint32_t first, std::uint32_t second) {
        std::vector<float> row(vocab.size(), -1.0F); row[second] = 1.0F; row[first] = 2.0F; return row;
    };
    const std::string limit = "error constraint key candidate resource limit";
    // The state after `[{`: a quote exhausts the evaluation budget, a full mask walk therefore refuses the row.
    {
        auto plain = f.selection(copy_spec(spec)); plain->set_argmax_first(false);
        plain->commit(plain->select_row(favor(open, open), plain->state()));
        check(outcome([&] { return plain->select_row(favor(close, close), plain->state()).token; }) == "token " + std::to_string(close), "plain row closes the array");
        plain->commit(plain->select_row(favor(brace, brace), plain->state()));
        check(outcome([&] { return plain->select_row(favor(quote, quote), plain->state()).token; }) == limit, "plain row after [{ exhausts the budget");
    }
    // A draft chain that would run into the exhausting transition: planning it must not fail the window, whose
    // rows never get there (the target closes the array instead of following the drafts).
    for (unsigned executor = 0; executor < 2; ++executor) for (unsigned follow = 0; follow < 2; ++follow) {
        auto plain = f.selection(copy_spec(spec)), windowed = f.selection(copy_spec(spec));
        plain->set_argmax_first(false); windowed->set_argmax_first(false);
        std::uint64_t jobs = 0;
        windowed->set_mask_ahead([&, executor](std::function<void()> job) { ++jobs; if (executor == 0) job(); });
        auto t = windowed->window_transaction();
        t->reset();
        check(t->select(favor(open, open)) == open, "window opens the array");
        t->seal()->publish();
        plain->commit(plain->select_row(favor(open, open), plain->state()));
        const std::array<std::uint32_t, 3> drafts{brace, quote, quote};
        t->reset();
        check(outcome([&] { t->expect(drafts); return 0; }) == "token 0", "planning a draft chain past an exhausting transition does not fail");
        check(jobs == 2, "the safe part of the chain is still planned");
        if (!follow) {
            const auto row = favor(close, space);
            const auto want = outcome([&] { return plain->select_row(row, plain->state()).token; });
            check(want == "token " + std::to_string(close) && outcome([&] { return t->select(row); }) == want, "a rejected draft leaves the row as the plain row");
            t->seal()->publish();
        } else {
            // The drafts are taken: the row that reaches the exhausting walk refuses exactly as the plain row.
            const auto row = favor(brace, space);
            const auto staged = plain->select_row(row, plain->state());
            check(t->select(row) == staged.token && staged.token == brace, "a taken draft selects as the plain row");
            plain->commit(staged);
            const auto next = favor(quote, space);
            const auto want = outcome([&] { return plain->select_row(next, plain->state()).token; });
            check(want == limit && outcome([&] { return t->select(next); }) == want, "a reached exhausting walk refuses as the plain row");
        }
    }
    // Argmax-first with the maximum refused (`b`): the row walks the full mask, whose quote exhausts the budget.
    // The default must keep that refusal; ranked membership (opt-in) checks the next-best logit instead.
    {
        const auto at_object = [&](const std::shared_ptr<ConstraintSelection>& s) {
            s->commit(s->select_row(favor(open, open), s->state())); s->commit(s->select_row(favor(brace, brace), s->state()));
        };
        const auto row = favor(letter, space);
        auto before = f.selection(copy_spec(spec)), standard = f.selection(copy_spec(spec)), ranked = f.selection(copy_spec(spec));
        before->set_argmax_first(true); before->set_ranked(0); standard->set_argmax_first(true);
        ranked->set_argmax_first(true); ranked->set_ranked(32);
        at_object(before); at_object(standard); at_object(ranked);
        const auto want = outcome([&] { return before->select_row(row, before->state()).token; });
        check(want == limit, "argmax-first refuses a row whose mask walk exhausts the budget");
        check(outcome([&] { return standard->select_row(row, standard->state()).token; }) == want, "default selection keeps the mask-walk refusal");
        check(outcome([&] { return ranked->select_row(row, ranked->state()).token; }) == "token " + std::to_string(space), "opt-in ranked membership takes the next-best admitted logit");
        check(standard->verdict("stop", "test").profile_ranked_rows == 0 && ranked->verdict("stop", "test").profile_ranked_rows == 1, "ranked rows are counted only when opted in");
    }
}
void mask_ahead() {
    const std::vector<std::string> vocab{"{", "}", "\"a\"", ":", "12", ",", "\"", "{\"", "\":", "z", "x", "1}", "a"};
    const std::uint32_t eos = 9, end_thinking = 10, n = static_cast<std::uint32_t>(vocab.size());
    Fixture f(vocab);
    const std::array<std::uint32_t, 1> excluded{eos};
    f.trie = std::make_shared<TokenByteTrie>(*f.tokenizer, excluded);
    const std::array<const char*, 3> schemas{"", "{\"const\":{}}", "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}}}"};
    constexpr std::array<float, 7> values{-1, -0.0F, 0.0F, 1, 2, 2, 3};
    std::uint64_t seed = 777;
    const auto next = [&] { seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<std::uint32_t>(seed >> 33); };
    const auto random_row = [&] { std::vector<float> row(n); for (auto& v : row) v = values[next() % values.size()]; return row; };
    const auto lowest_max = [](const std::vector<float>& row) {
        std::uint32_t g = 0;
        for (std::uint32_t i = 1; i < row.size(); ++i) if (row[i] > row[g]) g = i;
        return g;
    };
    struct Worker {
        std::mutex m; std::condition_variable cv; std::deque<std::function<void()>> q; bool stop = false; std::thread t;
        Worker() : t([this] {
            for (;;) {
                std::function<void()> job;
                { std::unique_lock lock(m); cv.wait(lock, [this] { return stop || !q.empty(); }); if (q.empty()) return; job = std::move(q.front()); q.pop_front(); }
                job();
            }
        }) {}
        ~Worker() { { std::lock_guard lock(m); stop = true; } cv.notify_all(); t.join(); }
        void submit(std::function<void()> job) { { std::lock_guard lock(m); q.push_back(std::move(job)); } cv.notify_one(); }
    } worker;
    std::uint64_t jobs = 0, windows = 0, rows_checked = 0, serial_rows = 0, ranked_rows = 0;
    std::array<std::uint64_t, 3> ahead_rows{}, ahead_masks{};
    for (unsigned executor = 0; executor < 3; ++executor) for (unsigned program = 0; program < 3; ++program)
    for (unsigned policy = 0; policy < 3; ++policy) for (unsigned thinking = 0; thinking < 2; ++thinking) for (unsigned argmax = 0; argmax < 3; ++argmax) {
        // argmax 0: every answer row walks a mask; 1: argmax-first with two ranked candidates (the rest falls to
        // the mask); 2: argmax-first, ranked membership off.
        const unsigned ranked = argmax == 1 ? 2U : 0U;
        ConstraintSpec spec; spec.eos_ids = {eos}; spec.end_thinking_id = end_thinking;
        spec.program = std::make_shared<ConstraintProgram>(program ? compile_json_schema(parse_json(schemas[program])) : compile_json_object());
        spec.logprobs = policy == 1; spec.top_logprobs = 3;
        if (policy == 2) spec.transform.bias = {{1, 0.5}};
        if (thinking) { spec.starts_in_reasoning = true; spec.thinking = ThinkingPolicy{0, 0, 3, {end_thinking, 5}}; }
        const ConstraintAheadSubmit submit = [&, executor](std::function<void()> job) {
            ++jobs;
            if (executor == 0) job(); else if (executor == 2) worker.submit(std::move(job));
        };
        for (unsigned walk = 0; walk < 24; ++walk) {
            // Windows: plain serial selection against a window whose drafts are sometimes wrong.
            const auto serial_spec = copy_spec(spec), window_spec = copy_spec(spec);
            auto serial = f.selection(serial_spec), windowed = f.selection(window_spec);
            serial->set_argmax_first(false); windowed->set_argmax_first(argmax != 0); windowed->set_ranked(ranked);
            windowed->set_mask_ahead(submit);
            auto t = windowed->window_transaction();
            for (unsigned window = 0; window < 6 && !serial->state().terminal; ++window) {
                std::vector<std::vector<float>> rows; std::vector<ConstraintSelection::Staged> want; std::vector<std::uint32_t> drafts;
                auto state = serial->state();
                bool refused = false;
                for (unsigned r = 0; r < 5 && !state.terminal; ++r) {
                    rows.push_back(random_row());
                    try { want.push_back(serial->select_row(rows.back(), state)); } catch (const ConstraintError&) { rows.pop_back(); refused = true; break; }
                    state = want.back().state;
                    drafts.push_back(next() % 4 ? want.back().token : next() % n);
                }
                if (want.empty()) { check(refused, "an empty window is a refusal"); break; }
                if (next() % 3 == 0 && drafts.size() > 1) drafts.pop_back();   // the last row may have no draft
                t->reset(); t->expect(drafts); ++windows;
                std::size_t taken = 0;
                for (std::size_t r = 0; r < want.size(); ++r) {
                    std::uint32_t id = 0;
                    if (next() % 2) { auto k = t->select_known(lowest_max(rows[r]), 0); id = k ? *k : t->select(rows[r]); }
                    else id = t->select(rows[r]);
                    check(id == want[r].token, "mask-ahead window row equals serial row");
                    ++taken; ++rows_checked;
                    if (r >= drafts.size() || drafts[r] != id) break;
                }
                auto receipt = t->seal();
                check(receipt->size() == taken, "mask-ahead window receipt size");
                receipt->publish();
                for (std::size_t r = 0; r < taken; ++r) serial->commit(want[r]);
                check(same_state(windowed->state(), serial->state(), spec, *f.trie), "mask-ahead window equals serial frontier");
            }
            scores_equal(window_spec->scores->tokens, serial_spec->scores->tokens);
            const auto v = windowed->verdict("stop", "test");
            ahead_rows[executor] += v.profile_ahead_rows;
            // Serial rows: the committed state is planned at prepare(), taken at select().
            const auto plain_spec = copy_spec(spec), row_spec = copy_spec(spec);
            auto plain = f.selection(plain_spec), ahead = f.selection(row_spec);
            plain->set_argmax_first(false); ahead->set_argmax_first(argmax != 0); ahead->set_ranked(ranked); ahead->set_mask_ahead(submit);
            ranked_rows += v.profile_ranked_rows;
            ConstraintRowSelection route(ahead);
            for (unsigned r = 0; r < 12 && !plain->state().terminal; ++r) {
                const auto row = random_row();
                std::optional<ConstraintSelection::Staged> want;
                try { want = plain->select_row(row, plain->state()); } catch (const ConstraintError&) {}
                route.prepare();
                if (next() % 5 == 0) route.prepare();   // a replanned row drops its first plan
                std::optional<std::uint32_t> got;
                try { got = route.select(row); } catch (const ConstraintError&) {}
                check(want.has_value() == got.has_value(), "mask-ahead serial row refuses exactly when the plain row does");
                if (!want) break;
                check(*got == want->token, "mask-ahead serial row equals the plain row");
                plain->commit(*want); route.commit(); ++serial_rows;
                check(same_state(ahead->state(), plain->state(), spec, *f.trie), "mask-ahead serial frontier");
            }
            scores_equal(row_spec->scores->tokens, plain_spec->scores->tokens);
            const auto rv = ahead->verdict("stop", "test");
            ahead_rows[executor] += rv.profile_ahead_rows;
            if (executor != 2) ahead_masks[executor] += v.profile_ahead_masks + rv.profile_ahead_masks;
        }
    }
    check(jobs > 1000 && windows > 1000 && rows_checked > 3000 && serial_rows > 1000 && ranked_rows > 200, "mask-ahead and ranked coverage");
    check(ahead_rows[0] > 500 && ahead_masks[0] >= ahead_rows[0], "inline executor: rows take worker-built masks");
    check(ahead_rows[1] == 0 && ahead_masks[1] == 0, "dropped jobs: every row builds its own mask");
    // No submit callback: no plan.
    ConstraintSpec spec; spec.eos_ids = {eos}; spec.program = std::make_shared<ConstraintProgram>(compile_json_object());
    auto off = f.selection(copy_spec(spec));
    check(!off->plan_ahead(off->state(), {}), "mask-ahead is off without a submit callback");
}

// Worker threads for the overlap cases: jobs run in submission order on `threads` threads; drain() returns
// when every submitted job has finished.
struct WalkPool {
    std::mutex m; std::condition_variable cv, idle; std::deque<std::function<void()>> q; bool stop = false; unsigned busy = 0;
    std::vector<std::thread> threads;
    explicit WalkPool(unsigned count) {
        for (unsigned i = 0; i < count; ++i) threads.emplace_back([this] {
            for (;;) {
                std::function<void()> job;
                { std::unique_lock lock(m); cv.wait(lock, [this] { return stop || !q.empty(); }); if (q.empty()) return; job = std::move(q.front()); q.pop_front(); ++busy; }
                job();
                { std::lock_guard lock(m); --busy; }
                idle.notify_all();
            }
        });
    }
    ~WalkPool() { { std::lock_guard lock(m); stop = true; } cv.notify_all(); for (auto& t : threads) t.join(); }
    void submit(std::function<void()> job) { { std::lock_guard lock(m); q.push_back(std::move(job)); } cv.notify_one(); }
    // Bounded: a lost job or notification fails the test instead of hanging it.
    void drain() {
        std::unique_lock lock(m);
        if (!idle.wait_for(lock, std::chrono::seconds(120), [this] { return q.empty() && busy == 0; })) throw std::runtime_error("walk pool did not drain in time");
    }
};
void wait_for(const std::atomic<int>& value, int target) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (value.load() < target) {
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("walk gate not reached in time");
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}
// Opens a gate when the scope ends, so held workers are released (and their pool can join) on a failed check too.
struct OpenGate {
    std::atomic<int>& gate;
    ~OpenGate() { gate.store(1); gate.notify_all(); }
};
// The worker / selector interleavings of mask-ahead, each forced with gates instead of left to the scheduler:
// two workers walking at once, a row arriving while its walk is in flight, a plan retired under its walkers, and
// a walk error produced on the worker and raised by the row.
void mask_ahead_overlap() {
    const std::vector<std::string> vocab{"{", "}", "\"a\"", ":", "12", ",", "\"", "{\"", "\":", "z", "x", "1}", "a"};
    const std::uint32_t eos = 9, n = static_cast<std::uint32_t>(vocab.size());
    Fixture f(vocab);
    const std::array<std::uint32_t, 1> excluded{eos};
    f.trie = std::make_shared<TokenByteTrie>(*f.tokenizer, excluded);
    ConstraintSpec spec; spec.eos_ids = {eos};
    spec.program = std::make_shared<ConstraintProgram>(compile_json_schema(parse_json("{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}}}")));
    const auto favor = [&](std::uint32_t id) { std::vector<float> row(n, -1.0F); row[id] = 2.0F; return row; };
    // claimed: walks a worker has started; gate 0 holds every started walk before its first step. waited: rows
    // that found their walk in flight; the first of them opens the gate, from inside the wait branch.
    std::atomic<int> claimed{0}, built{0}, abandoned{0}, gate{0}, waited{0};
    {
        WalkPool pool(2);
        OpenGate open_gate{gate};
        auto serial = f.selection(copy_spec(spec)), windowed = f.selection(copy_spec(spec));
        serial->set_argmax_first(false); windowed->set_argmax_first(false);
        windowed->set_mask_ahead([&](std::function<void()> job) { pool.submit(std::move(job)); });
        windowed->set_ahead_observer([&](int phase) {
            if (phase == 1) { claimed.fetch_add(1); gate.wait(0); }
            else if (phase == 4) { waited.fetch_add(1); gate.store(1); gate.notify_all(); }
            else if (phase == 2) built.fetch_add(1);
            else abandoned.fetch_add(1);
        });
        auto t = windowed->window_transaction();
        const auto window = [&](std::initializer_list<std::uint32_t> ids) {
            std::vector<std::vector<float>> rows; std::vector<ConstraintSelection::Staged> want;
            auto state = serial->state();
            for (const auto id : ids) { rows.push_back(favor(id)); want.push_back(serial->select_row(rows.back(), state)); state = want.back().state; }
            return std::pair{std::move(rows), std::move(want)};
        };
        // Two walks in flight at once, and the row of each arriving while its walk is held.
        {
            auto [rows, want] = window({0, 2, 3});
            const std::array<std::uint32_t, 2> drafts{want[0].token, want[1].token};
            t->reset(); t->expect(drafts);
            wait_for(claimed, 2);
            check(claimed.load() == 2 && built.load() == 0 && abandoned.load() == 0, "two workers hold a walk each at the same time");
            // Slot 0 is worker-owned and held at the gate: only the row's own wait branch opens it, so the row
            // cannot return without having waited for the in-flight walk.
            check(t->select(rows[0]) == want[0].token, "a row that waited for an in-flight walk equals the serial row");
            check(waited.load() == 1, "the row entered the wait for its in-flight walk");
            check(t->select(rows[1]) == want[1].token && t->select(rows[2]) == want[2].token, "rows after concurrent walks equal the serial rows");
            auto receipt = t->seal(); check(receipt->size() == 3, "overlap window receipt size"); receipt->publish();
            for (const auto& staged : want) serial->commit(staged);
            check(same_state(windowed->state(), serial->state(), spec, *f.trie), "overlap window equals the serial frontier");
            pool.drain();
            const auto v = windowed->verdict("stop", "test");
            check(v.profile_ahead_rows >= 2 && v.profile_ahead_rows <= 3, "the gated rows took worker-built masks");
            check(built.load() == claimed.load() && abandoned.load() == 0 && v.profile_ahead_masks == static_cast<std::uint64_t>(built.load()), "every started walk completed");
        }
        // A plan retired while both workers hold a walk: the walks are abandoned and the next attempt, which
        // has no plan, still equals the serial rows.
        {
            auto [rows, want] = window({4, 1});
            const std::array<std::uint32_t, 2> drafts{want[0].token, want[1].token};
            const int before = claimed.load(), finished = built.load(), waits = waited.load();
            gate.store(0);
            t->reset(); t->expect(drafts);
            wait_for(claimed, before + 2);
            t->reset();
            gate.store(1); gate.notify_all();
            pool.drain();
            check(claimed.load() == before + 2 && abandoned.load() == 2 && built.load() == finished && waited.load() == waits, "walks of a retired plan are abandoned");
            check(t->select(rows[0]) == want[0].token && t->select(rows[1]) == want[1].token, "rows after a retired plan equal the serial rows");
            t->seal()->publish();
            for (const auto& staged : want) serial->commit(staged);
            check(same_state(windowed->state(), serial->state(), spec, *f.trie), "retired plan leaves the serial frontier");
        }
    }
    // A walk that fails on the worker: the error is stored with the slot and raised by the row that takes it.
    {
        const std::vector<std::string> budget_vocab{"[", "{", "\"", "]", " ", "z", "b"};
        Fixture b(budget_vocab);
        const std::array<std::uint32_t, 1> budget_excluded{5};
        b.trie = std::make_shared<TokenByteTrie>(*b.tokenizer, budget_excluded);
        ConstraintSpec budget; budget.eos_ids = {5}; budget.program = budgeted_program();
        const auto row = [&](std::uint32_t id) { std::vector<float> r(budget_vocab.size(), -1.0F); r[id] = 2.0F; return r; };
        WalkPool pool(1);
        std::atomic<int> walked{0};
        auto plain = b.selection(copy_spec(budget)), windowed = b.selection(copy_spec(budget));
        plain->set_argmax_first(false); windowed->set_argmax_first(false);
        windowed->set_mask_ahead([&](std::function<void()> job) { pool.submit(std::move(job)); });
        windowed->set_ahead_observer([&](int phase) { if (phase == 2) walked.fetch_add(1); });
        auto t = windowed->window_transaction();
        t->reset(); check(t->select(row(0)) == 0, "budgeted window opens the array"); t->seal()->publish();
        plain->commit(plain->select_row(row(0), plain->state()));
        const std::array<std::uint32_t, 1> drafts{1};
        t->reset(); t->expect(drafts);
        pool.drain();
        check(walked.load() == 2, "the worker finished both walks, one of them with an error");
        const auto staged = plain->select_row(row(1), plain->state());
        check(t->select(row(1)) == staged.token, "row before the failed walk equals the plain row");
        plain->commit(staged);
        const auto want = outcome([&] { return plain->select_row(row(2), plain->state()).token; });
        check(want == "error constraint key candidate resource limit" && outcome([&] { return t->select(row(2)); }) == want,
              "a walk error from the worker is raised by the row that takes it");
        check(windowed->verdict("stop", "test").profile_ahead_rows == 1, "only the completed row counts as served by the worker");
    }
}
// The two selection levers resolve through the shared registry: strict numbers, recorded values.
void selection_policy() {
    const auto put = [](const char* key, const char* value) {
#ifdef _WIN32
        check(_putenv_s(key, value ? value : "") == 0, "set test environment");
#else
        check((value ? ::setenv(key, value, 1) : ::unsetenv(key)) == 0, "set test environment");
#endif
    };
    constexpr const char* ahead = "STRIX_CONSTRAINT_MASK_AHEAD";
    constexpr const char* ranked = "STRIX_CONSTRAINT_RANKED";
    put(ahead, nullptr); put(ranked, nullptr);
    auto policy = constraint_selection_policy_from_environment();
    check(policy.mask_ahead && policy.ranked == 0, "selection policy defaults: mask-ahead on, ranked membership off");
    const auto recorded = [&](const char* name, const char* value, bool set) {
        for (const auto& entry : envcfg::registered()) if (entry.name == name) return entry.value == value && entry.set == set;
        return false;
    };
    check(recorded(ahead, "1", false) && recorded(ranked, "0", false), "selection policy defaults are recorded");
    put(ahead, "0"); put(ranked, "32");
    policy = constraint_selection_policy_from_environment();
    check(!policy.mask_ahead && policy.ranked == 32, "selection policy reads both settings");
    check(recorded(ahead, "0", true) && recorded(ranked, "32", true), "selection policy settings are recorded");
    put(ahead, "1"); put(ranked, "4096");
    policy = constraint_selection_policy_from_environment();
    check(policy.mask_ahead && policy.ranked == 4096, "selection policy upper bound");
    for (const char* bad : {"12x", "-1", "4097", " 5", "0x10", "1e2", "many"}) {
        put(ranked, bad);
        bool rejected = false;
        try { (void)constraint_selection_policy_from_environment(); }
        catch (const std::invalid_argument& e) { rejected = std::string(e.what()).find(ranked) != std::string::npos; }
        check(rejected, "a ranked setting that is not a number in range is rejected by name");
    }
    put(ahead, nullptr); put(ranked, nullptr);
}

int main() try {
    argmax_first();
    speculation_and_limits();
    selection_policy();
    mask_ahead();
    mask_ahead_overlap();
    selection_failures(); routes_and_reuse(); boundaries(); allocation_failures(); masks_and_cohorts(); enumeration(); masked_enumeration(); grammar_and_thinking();
    // A four-row window can reject comparisons at rows 0..2; row 3 is bonus.
    check(rejection_rows[0] && rejection_rows[1] && rejection_rows[2], "rejection coverage");
    check(std::all_of(eos_rows.begin(), eos_rows.end(), [](bool x) { return x; }), "EOS row coverage");
    check(std::all_of(closure_rows.begin(), closure_rows.end(), [](bool x) { return x; }), "closure start row coverage");
    check(crossing_closure, "closure crosses windows");
    check(truncated_closure, "closure truncated by budget");
    check(bonus_seen && replays && divergent_suffix, "bonus and divergent replay coverage");
    check(natural_transition, "natural end-thinking coverage");
    check(reasoning_eos, "early reasoning EOS coverage");
    std::cout << "mtp selection transaction: " << runs << " complete runs, " << replays << " checkpoint replays, " << checks << " checks passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL after " << runs << " runs / " << checks << " checks: " << e.what() << '\n';
    return 1;
}
