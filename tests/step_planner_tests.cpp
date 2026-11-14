// Host-only tests for the shared step planner.
#include "strix/serve/step_planner.hpp"

#include "strix/serve/short_first.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace strix::serve;

namespace {
int failures = 0;
void check(bool value, const char* label) {
    if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}

struct Rng {
    std::uint64_t state;
    std::uint32_t next(std::uint32_t bound) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>((state >> 33) % bound);
    }
};

// ---- Policy 1: time slicing with a debt ledger (prompt and decode never share a forward) ----------------------
// Reference: the per-step decision of a time-sliced serve loop. The prompt with at most short_rows rows left
// goes first (oldest inside each class); it runs when nobody decodes or the debt is paid; prompt time is charged
// with the share, decode time pays it back; the debt is cleared when no prompt is left.
struct SliceSlot {
    std::uint32_t id;
    bool prefilling;
    std::uint64_t remaining;
    std::uint64_t order;
};
struct SliceReference {
    double debt = 0.0;
    // Returns the slot index that runs a prompt unit, or -1.
    int step(const std::vector<SliceSlot>& slots, std::uint64_t short_rows) {
        const SliceSlot* prompt = nullptr;
        bool decoding = false;
        const auto ahead = [&](const SliceSlot& a, const SliceSlot& b) {
            const bool sa = short_rows && a.remaining <= short_rows, sb = short_rows && b.remaining <= short_rows;
            return sa != sb ? sa : a.order < b.order;
        };
        for (const auto& s : slots) {
            if (s.prefilling && (!prompt || ahead(s, *prompt))) prompt = &s;
            if (!s.prefilling) decoding = true;
        }
        if (!prompt) debt = 0.0;
        if (prompt && (!decoding || debt <= 0.0)) return static_cast<int>(prompt - slots.data());
        return -1;
    }
};

void time_slice_equivalence_tests() {
    for (const double share : {0.0, 0.5, 1.0, 3.0}) {
        for (const std::uint64_t short_rows : {std::uint64_t{2048}, std::uint64_t{0}, std::uint64_t{512}}) {
            Rng rng{0x5eed0000ULL + static_cast<std::uint64_t>(share * 16) + short_rows};
            StepPlanner planner;
            SliceReference reference;
            std::vector<SliceSlot> slots;
            std::uint64_t arrivals = 0;
            bool same = true, shape = true;
            for (int step = 0; step < 4000; ++step) {
                if (slots.size() < 8 && rng.next(4) == 0)
                    slots.push_back({static_cast<std::uint32_t>(arrivals), true, 1 + rng.next(6000), arrivals}), ++arrivals;
                // Shuffle the slot order now and then: the decision must not depend on it.
                if (!slots.empty() && rng.next(8) == 0) std::swap(slots[0], slots[rng.next(static_cast<std::uint32_t>(slots.size()))]);

                std::vector<StepLane> lanes;
                for (const auto& s : slots) lanes.push_back({s.id, s.prefilling, s.remaining, s.order});
                StepLimits limits;
                limits.prompt_share = share;
                limits.short_first_rows = short_rows;
                const auto plan = planner.plan(lanes, limits);
                const int expected = reference.step(slots, short_rows);

                const bool ran = !plan.prompt.empty();
                same &= ran == (expected >= 0) && (!ran || plan.prompt[0].lane == slots[static_cast<std::size_t>(expected)].id);
                shape &= plan.prompt.size() <= 1 && plan.prompt_before_decode == ran && (!ran || plan.prompt[0].rows == 0);
                std::size_t decoders = 0;
                for (const auto& s : slots) decoders += !s.prefilling;
                shape &= plan.decode_lanes.size() == decoders;

                double prompt_ms = 0.0, decode_ms = 0.0, suspended_ms = 0.0;
                if (expected >= 0) {
                    auto& s = slots[static_cast<std::size_t>(expected)];
                    const std::uint64_t rows = std::min<std::uint64_t>(s.remaining, 1024);
                    prompt_ms = 1.9 * static_cast<double>(rows) + rng.next(5);
                    suspended_ms = rng.next(3) == 0 ? rng.next(40) : 0;
                    s.remaining -= rows;
                    if (s.remaining == 0) s.prefilling = false;
                    reference.debt += share * std::max(0.0, prompt_ms - suspended_ms);
                }
                if (decoders) {
                    decode_ms = 48.0 + 7.5 * static_cast<double>(decoders) + rng.next(7);
                    reference.debt -= decode_ms;
                }
                planner.account(prompt_ms, decode_ms, suspended_ms);
                same &= planner.debt_ms() == reference.debt;
                // A decoder finishes now and then.
                for (std::size_t i = 0; i < slots.size(); ++i)
                    if (!slots[i].prefilling && rng.next(24) == 0) { slots.erase(slots.begin() + static_cast<std::ptrdiff_t>(i)); break; }
            }
            check(same, "time slicing: same prompt choice and debt as the reference on every step");
            check(shape, "time slicing: one head, prompt before decode, every decoder kept");
        }
    }
}

// ---- Policy 2: shortest-first admission with aging (short_first.hpp) -------------------------------------------
struct Waiting {
    std::uint32_t id;
    std::uint64_t rows;
    unsigned skips;
    bool eligible;
};

void short_first_equivalence_tests() {
    for (const bool short_first : {true, false}) {
        Rng rng{short_first ? 0xa9e1ULL : 0x0ddULL};
        std::vector<Waiting> reference, shared;
        std::uint32_t arrivals = 0;
        bool same = true;
        for (int step = 0; step < 6000; ++step) {
            for (std::uint32_t n = rng.next(3); n != 0 && reference.size() < 12; --n) {
                const Waiting w{arrivals++, 1 + rng.next(rng.next(4) == 0 ? 9000 : 300), 0, true};
                reference.push_back(w);
                shared.push_back(w);
            }
            for (std::size_t i = 0; i < reference.size(); ++i) reference[i].eligible = shared[i].eligible = rng.next(5) != 0;

            const std::size_t expected = select_waiting(
                reference, short_first, [](const Waiting& w) { return w.rows; }, [](Waiting& w) -> unsigned& { return w.skips; },
                static_cast<std::size_t>(-1), [](const Waiting& w) { return w.eligible; });

            std::vector<StepLane> lanes;
            for (const auto& w : shared) {
                StepLane lane;
                lane.lane = w.id; lane.prompt = true; lane.prompt_remaining = w.rows; lane.order = w.id;
                lane.overtakes = w.skips; lane.eligible = w.eligible;
                lanes.push_back(lane);
            }
            StepLimits limits;
            limits.prompt_order = PromptOrder::shortest;
            limits.short_first_rows = short_first ? 1 : 0;
            StepPlanner planner;
            const auto plan = planner.plan(lanes, limits);
            for (const auto id : plan.overtaken)
                for (auto& w : shared) if (w.id == id) ++w.skips;

            const bool none = expected == reference.size();
            same &= plan.prompt.empty() == none;
            if (!none && !plan.prompt.empty()) {
                same &= plan.prompt[0].lane == reference[expected].id;
                reference.erase(reference.begin() + static_cast<std::ptrdiff_t>(expected));
                shared.erase(std::find_if(shared.begin(), shared.end(), [&](const Waiting& w) { return w.id == plan.prompt[0].lane; }));
            }
            same &= reference.size() == shared.size();
            for (std::size_t i = 0; same && i < reference.size(); ++i)
                same &= reference[i].id == shared[i].id && reference[i].skips == shared[i].skips;
        }
        check(same, short_first ? "short first: same owner and overtake counts as select_waiting"
                                : "arrival order: same owner as select_waiting, an ineligible oldest owner blocks");
    }
}

// A long prompt that keeps being overtaken by younger short prompts runs after kMaxQueueOvertakes overtakes.
void starvation_tests() {
    for (const auto order : {PromptOrder::short_class, PromptOrder::shortest}) {
        StepPlanner planner;
        StepLimits limits;
        limits.prompt_order = order;
        unsigned long_overtakes = 0;
        std::uint64_t arrivals = 1;
        int ran_at = -1;
        for (int step = 0; step < 64 && ran_at < 0; ++step) {
            std::vector<StepLane> lanes;
            StepLane big;
            big.lane = 0; big.prompt = true; big.prompt_remaining = 100000; big.order = 0; big.overtakes = long_overtakes;
            lanes.push_back(big);
            for (int k = 0; k < 2; ++k) {  // two fresh short prompts every step
                StepLane small;
                small.lane = static_cast<std::uint32_t>(arrivals); small.prompt = true; small.prompt_remaining = 40; small.order = arrivals++;
                lanes.push_back(small);
            }
            const auto plan = planner.plan(lanes, limits);
            check(plan.prompt.size() == 1, "starvation: one prompt unit per step");
            if (plan.prompt[0].lane == 0) ran_at = step;
            for (const auto id : plan.overtaken) if (id == 0) ++long_overtakes;
        }
        check(ran_at == static_cast<int>(kMaxQueueOvertakes), "starvation: the long prompt runs after exactly the overtake bound");
        check(long_overtakes == kMaxQueueOvertakes, "starvation: overtakes never pass the bound");
    }
}

// ---- Policy 3: row fill of one forward (decode rows first, prompt rows in the rows left) ----------------------
struct FillSequence {
    std::uint32_t position;
    std::uint64_t prompt_left;
    bool has_token;  // a decoding sequence with a token to feed
};
// Reference: rows per sequence of a co-forward step (one row per decoding sequence, then prompt chunks oldest first;
// a sequence at position 0 reads its first token alone).
std::vector<std::uint32_t> fill_reference(const std::vector<FillSequence>& sequences, std::uint32_t prompt_rows,
                                          std::uint32_t max_rows, std::uint32_t segment_rows) {
    std::vector<std::uint32_t> rows(sequences.size(), 0);
    std::uint32_t used = 0;
    for (std::size_t i = 0; i < sequences.size() && used < max_rows; ++i) {
        if (sequences[i].prompt_left != 0 || !sequences[i].has_token) continue;
        rows[i] = 1; ++used;
    }
    for (std::size_t i = 0; i < sequences.size() && used < max_rows; ++i) {
        if (sequences[i].prompt_left == 0) continue;
        std::uint64_t take = std::min<std::uint64_t>({prompt_rows, segment_rows, max_rows - used, sequences[i].prompt_left});
        if (sequences[i].position == 0) take = std::min<std::uint64_t>(take, 1);
        rows[i] = static_cast<std::uint32_t>(take); used += rows[i];
    }
    return rows;
}

void co_forward_equivalence_tests() {
    Rng rng{0xd5ULL};
    bool same = true, order_kept = true;
    for (int trial = 0; trial < 20000; ++trial) {
        const std::uint32_t segment_rows = 8;
        const std::uint32_t max_rows = rng.next(2) ? 8 : 16;
        const std::uint32_t prompt_rows = rng.next(12);  // 0 = no prompt rows
        std::vector<FillSequence> sequences(rng.next(13));
        for (auto& s : sequences) {
            const bool reading = rng.next(2) != 0;
            s = {reading && rng.next(3) == 0 ? 0U : 1U + rng.next(500), reading ? 1U + rng.next(40) : 0U, rng.next(8) != 0};
        }
        const auto expected = fill_reference(sequences, prompt_rows, max_rows, segment_rows);

        // Adapter: decoding sequences with no token are not lanes; a first token runs alone.
        std::vector<StepLane> lanes;
        for (std::uint32_t i = 0; i < sequences.size(); ++i) {
            const auto& s = sequences[i];
            if (s.prompt_left == 0 && !s.has_token) continue;
            StepLane lane;
            lane.lane = i; lane.prompt = s.prompt_left != 0; lane.prompt_remaining = s.prompt_left; lane.order = i;
            lane.max_prompt_rows = s.position == 0 ? 1 : 0;
            lanes.push_back(lane);
        }
        const std::array<std::uint32_t, 1> units{std::min(prompt_rows, segment_rows)};
        StepLimits limits;
        limits.co_forward = true;
        limits.rows_per_forward = max_rows;
        limits.prompt_units = units;
        limits.short_first_rows = 0;
        StepPlanner planner;
        const auto plan = planner.plan(lanes, limits);

        std::vector<std::uint32_t> got(sequences.size(), 0);
        for (const auto lane : plan.decode_lanes) got[lane] += 1;
        for (const auto& p : plan.prompt) got[p.lane] += p.rows;
        same &= got == expected && !plan.prompt_before_decode && plan.overtaken.empty();
        order_kept &= std::is_sorted(plan.decode_lanes.begin(), plan.decode_lanes.end()) &&
                      std::is_sorted(plan.prompt.begin(), plan.prompt.end(), [](const auto& a, const auto& b) { return a.lane < b.lane; });
    }
    check(same, "co-forward: same rows per sequence as the row-fill reference");
    check(order_kept, "co-forward: decode rows first in sequence order, prompt rows oldest first");

    // No certified unit: prompts fill the rows the decoders leave.
    const std::vector<StepLane> lanes{{0, false, 0, 0}, {1, true, 100, 1}, {2, true, 100, 2}};
    StepLimits limits;
    limits.co_forward = true;
    limits.rows_per_forward = 8;
    StepPlanner planner;
    const auto plan = planner.plan(lanes, limits);
    check(plan.decode_lanes.size() == 1 && plan.prompt.size() == 1 && plan.prompt[0].lane == 1 && plan.prompt[0].rows == 7,
          "co-forward: no unit fills the forward");
}

// ---- Bounded prompt slices ----------------------------------------------------------------------------------
void decode_gap_tests() {
    const std::array<std::uint32_t, 4> units{64, 256, 1024, 4096};
    const std::vector<StepLane> busy{{0, false, 0, 0}, {1, true, 2853, 1}};
    const std::vector<StepLane> idle{{1, true, 2853, 1}};
    StepLimits limits;
    limits.prompt_units = units;
    limits.prompt_row_ms = 1.9;

    {
        StepPlanner planner;
        auto plan = planner.plan(busy, limits);
        check(plan.prompt.size() == 1 && plan.prompt[0].rows == 2853, "gap: no bound runs the largest unit, capped to the prompt");
        limits.max_decode_gap_ms = 500.0;
        plan = planner.plan(busy, limits);
        check(plan.prompt.size() == 1 && plan.prompt[0].rows == 256, "gap: 500 ms at 1.9 ms per row is the 256-row unit");
        plan = planner.plan(idle, limits);
        check(plan.prompt[0].rows == 2853, "gap: nobody decodes, the largest unit runs");
    }
    // The gap never exceeds the bound when a unit of that size exists, for any bound and row cost.
    bool bounded = true, largest = true;
    for (const double row_ms : {0.5, 1.9, 7.5})
        for (double gap = 10.0; gap < 20000.0; gap *= 1.37) {
            StepPlanner planner;
            limits.prompt_row_ms = row_ms;
            limits.max_decode_gap_ms = gap;
            const auto plan = planner.plan(std::vector<StepLane>{{0, false, 0, 0}, {1, true, 1000000, 1}}, limits);
            const std::uint32_t rows = plan.prompt[0].rows;
            const bool some_unit_fits = units[0] * row_ms <= gap;
            bounded &= !some_unit_fits || rows * row_ms <= gap;
            bounded &= some_unit_fits || rows == units[0];
            for (const auto unit : units) largest &= !(unit * row_ms <= gap) || rows >= unit;
        }
    check(bounded, "gap: the unit fits the bound, or is the smallest when none fits");
    check(largest, "gap: the largest fitting unit is taken");

    {
        // The observed cost replaces the prior: 4 ms per row measured, so 500 ms now holds 64 rows only.
        StepPlanner planner;
        limits.prompt_row_ms = 1.9;
        limits.max_decode_gap_ms = 500.0;
        auto plan = planner.plan(busy, limits);
        planner.account(4.0 * plan.prompt[0].rows, 100.0, 0.0);
        check(std::fabs(planner.prompt_row_ms() - 4.0) < 1e-9, "gap: observed cost per row");
        plan = planner.plan(busy, limits);
        check(plan.prompt[0].rows == 64, "gap: observed cost replaces the prior");
        // No prior and nothing observed: probe with the smallest unit.
        StepPlanner fresh;
        limits.prompt_row_ms = 0.0;
        check(fresh.plan(busy, limits).prompt[0].rows == 64, "gap: unknown cost probes with the smallest unit");
        // No certified units: the bound gives a row count once a cost is known, the family unit before.
        limits.prompt_units = {};
        check(fresh.plan(busy, limits).prompt[0].rows == 0, "gap: no units and no cost keeps the family unit");
        limits.prompt_row_ms = 2.0;
        check(fresh.plan(busy, limits).prompt[0].rows == 250, "gap: no units, rows from the bound");
    }
}

// Worst decode gap of a late long prompt, simulated: bounded slices against whole-block units.
void gap_simulation_tests() {
    const std::array<std::uint32_t, 3> units{256, 1024, 4096};
    const auto worst_gap = [&](double bound, double& total_ms) {
        StepPlanner planner;
        StepLimits limits;
        limits.prompt_units = units;
        limits.prompt_row_ms = 1.9;
        limits.max_decode_gap_ms = bound;
        std::uint64_t remaining = 2853;
        double gap = 0.0, worst = 0.0;
        total_ms = 0.0;
        for (int step = 0; step < 200 && remaining; ++step) {
            std::vector<StepLane> lanes{{0, false, 0, 0}, {1, false, 0, 1}, {2, false, 0, 2}, {3, false, 0, 3}, {4, true, remaining, 4}};
            const auto plan = planner.plan(lanes, limits);
            double prompt_ms = 0.0;
            if (!plan.prompt.empty()) { prompt_ms = 1.9 * plan.prompt[0].rows; remaining -= plan.prompt[0].rows; }
            gap += prompt_ms;
            worst = std::max(worst, gap);
            gap = 0.0;  // the decode round emits tokens
            planner.account(prompt_ms, 78.0, 0.0);
            total_ms += prompt_ms + 78.0;
        }
        check(remaining == 0, "simulation: the prompt finishes");
        return worst;
    };
    double whole_ms = 0.0, sliced_ms = 0.0;
    const double whole = worst_gap(0.0, whole_ms), sliced = worst_gap(500.0, sliced_ms);
    check(whole > 5000.0 && sliced <= 500.0, "simulation: a 500 ms bound cuts the worst gap of a 2.8K prompt from 5.4 s to under 0.5 s");
}

// ---- Debt ledger and cancellation ---------------------------------------------------------------------------
void debt_tests() {
    StepPlanner planner;
    StepLimits limits;
    limits.prompt_share = 1.0;
    const std::vector<StepLane> lanes{{0, false, 0, 0}, {1, true, 5000, 1}};
    auto plan = planner.plan(lanes, limits);
    check(plan.prompt.size() == 1 && plan.prompt_before_decode, "debt: no debt, the prompt runs first");
    planner.account(100.0, 40.0, 10.0);
    check(std::fabs(planner.debt_ms() - 50.0) < 1e-9, "debt: prompt time less suspension, less decode time");
    plan = planner.plan(lanes, limits);
    check(plan.prompt.empty() && !plan.prompt_before_decode && plan.decode_lanes.size() == 1, "debt: decoders are owed time, decode only");
    planner.account(0.0, 40.0, 0.0);
    plan = planner.plan(lanes, limits);
    check(plan.prompt.empty(), "debt: still owed");
    planner.account(0.0, 40.0, 0.0);
    check(planner.plan(lanes, limits).prompt.size() == 1, "debt: paid, the prompt runs again");

    // Nobody decodes: the prompt runs whatever the debt.
    StepPlanner alone;
    const std::vector<StepLane> only_prompt{{1, true, 5000, 1}};
    (void)alone.plan(only_prompt, limits);
    alone.account(100.0, 0.0, 0.0);
    check(alone.debt_ms() > 0.0 && alone.plan(only_prompt, limits).prompt.size() == 1, "debt: no decoder, the prompt is not held");

    // Cancel during a prompt unit: the lane is gone at the next plan; the debt of its unit is dropped with it.
    StepPlanner cancelled;
    (void)cancelled.plan(lanes, limits);
    cancelled.account(300.0, 40.0, 0.0);
    const std::vector<StepLane> decoders{{0, false, 0, 0}};
    plan = cancelled.plan(decoders, limits);
    check(plan.prompt.empty() && plan.overtaken.empty() && plan.decode_lanes.size() == 1 && cancelled.debt_ms() == 0.0,
          "cancel: no prompt left clears the debt and keeps the decoders");
    // A new prompt after the cancel is not charged for the cancelled one.
    check(cancelled.plan(lanes, limits).prompt.size() == 1, "cancel: the next prompt starts with no debt");
    // Cancel of the head while another prompt waits: the other one is planned, the debt stays owed to the decoders.
    StepPlanner two;
    const std::vector<StepLane> both{{0, false, 0, 0}, {1, true, 5000, 1}, {2, true, 300, 2}};
    plan = two.plan(both, limits);
    check(plan.prompt[0].lane == 2 && plan.overtaken.size() == 1 && plan.overtaken[0] == 1, "cancel: the short prompt is the head");
    two.account(20.0, 0.0, 0.0);  // cancelled inside its unit, no decode round yet
    const std::vector<StepLane> rest{{0, false, 0, 0}, {1, true, 5000, 1}};
    check(two.plan(rest, limits).prompt.empty() && two.debt_ms() == 20.0, "cancel: the time spent is still owed");
    two.account(0.0, 40.0, 0.0);
    check(two.plan(rest, limits).prompt[0].lane == 1, "cancel: the waiting prompt runs once the debt is paid");

    two.reset();
    check(two.debt_ms() == 0.0 && two.prompt_row_ms() == 0.0, "reset clears the ledger");
    check(planner.plan({}, limits).decode_lanes.empty(), "no lanes, empty plan");
}

}  // namespace

int main() {
    time_slice_equivalence_tests(); short_first_equivalence_tests(); starvation_tests(); co_forward_equivalence_tests();
    decode_gap_tests(); gap_simulation_tests(); debt_tests();
    std::cout << "step planner failures: " << failures << '\n';
    return failures ? 1 : 0;
}
