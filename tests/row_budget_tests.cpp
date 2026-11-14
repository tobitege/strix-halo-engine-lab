// Host-only tests for the shared row budget planner.
#include "strix/serve/row_budget.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

using namespace strix::serve;

namespace {
int failures = 0;
void check(bool value, const char* label) {
    if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

std::uint32_t total(const RowPlan& plan) { return std::accumulate(plan.rows.begin(), plan.rows.end(), 0u); }

std::vector<LaneDemand> lanes(std::uint32_t count, std::uint32_t min_rows, std::uint32_t max_rows,
                              std::uint32_t remaining = 1000) {
    std::vector<LaneDemand> out(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        out[i].lane = i; out[i].remaining_tokens = remaining; out[i].min_rows = min_rows; out[i].max_rows = max_rows;
    }
    return out;
}

// Expected tokens of a lane with r rows at a flat acceptance a: 1 + a + ... + a^(r-1).
double useful(std::uint32_t rows, double a) {
    double sum = 0, value = 1;
    for (std::uint32_t i = 0; i < rows; ++i) { sum += value; value *= a; }
    return sum;
}

void cost_model_tests() {
    // The priors with a = 0.9: four lanes hold four rows each and eight lanes fill 32 rows.
    const std::array<std::uint32_t, 4> totals{4, 8, 16, 32};
    RowLimits limits; limits.rows_per_forward = 32; limits.totals = totals;
    auto four = lanes(4, 2, 8);
    RowPlan plan = plan_rows(four, CycleCost{}, limits, 0.9);
    check(plan.rows == std::vector<std::uint32_t>{4, 4, 4, 4}, "C=4 priors give four rows per lane");
    check(near(plan.expected_ms, 48 + 7.5 * 16), "C=4 cycle time");
    check(near(plan.expected_tokens, 4 * useful(4, 0.9)), "C=4 expected tokens");
    auto eight = lanes(8, 2, 8);
    plan = plan_rows(eight, CycleCost{}, limits, 0.9);
    check(plan.rows == std::vector<std::uint32_t>(8, 4u), "C=8 priors give four rows per lane");

    // Lane and draft terms: a lane with r rows drafts r - 1 of them.
    CycleCost cost; cost.fixed_ms = 10; cost.row_ms = 2; cost.lane_ms = 3; cost.draft_row_ms = 1;
    RowLimits fixed; fixed.rows_per_forward = 6; const std::array<std::uint32_t, 1> six{6}; fixed.totals = six;
    auto two = lanes(2, 1, 3);
    plan = plan_rows(two, cost, fixed);
    check(plan.rows == std::vector<std::uint32_t>{3, 3}, "single qualified total is used");
    check(near(plan.expected_ms, 10 + 2 * 6 + 3 * 2 + 1 * 4), "lane and draft cost terms");

    // Nothing to run.
    plan = plan_rows({}, CycleCost{}, RowLimits{});
    check(plan.rows.empty() && plan.expected_ms == 0 && plan.expected_tokens == 0, "empty demand");
    auto done = lanes(2, 2, 8, 0);
    plan = plan_rows(done, CycleCost{}, RowLimits{});
    check(total(plan) == 0 && plan.expected_ms == 0, "finished lanes get no rows");
}

void monotone_tests() {
    // A higher flat acceptance never buys fewer rows.
    RowLimits limits; limits.rows_per_forward = 32;
    for (std::uint32_t count : {1u, 2u, 4u, 8u}) {
        auto demand = lanes(count, 1, 8);
        std::uint32_t previous = 0;
        double previous_tokens = 0;
        for (int step = 0; step <= 100; ++step) {
            const RowPlan plan = plan_rows(demand, CycleCost{}, limits, step / 100.0);
            check(total(plan) >= previous, "total rows monotone in the prior");
            check(plan.expected_tokens + 1e-12 >= previous_tokens, "expected tokens monotone in the prior");
            previous = total(plan); previous_tokens = plan.expected_tokens;
        }
        check(previous == std::min(32u, count * 8), "certain rows fill the limit");
    }
    // Raising one lane's history never lowers that lane's rows, at any level of the others.
    for (int others = 1; others <= 9; ++others) {
        std::uint32_t previous = 0;
        for (int step = 0; step <= 50; ++step) {
            auto demand = lanes(4, 1, 8);
            const std::vector<float> mine(8, static_cast<float>(step) / 50.0f);
            std::vector<float> first = mine; first[0] = 1.0f;
            demand[2].accept = first;
            const RowPlan plan = plan_rows(demand, CycleCost{}, limits, others / 10.0);
            check(plan.rows[2] >= previous, "lane rows monotone in its own acceptance");
            previous = plan.rows[2];
        }
    }
    // A lane with a better history gets at least the rows of a worse one.
    auto demand = lanes(3, 2, 8);
    const std::array<float, 8> good{1.0f, 0.95f, 0.95f, 0.95f, 0.95f, 0.95f, 0.95f, 0.95f};
    const std::array<float, 8> bad{1.0f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f};
    demand[0].accept = bad; demand[2].accept = good;
    const RowPlan plan = plan_rows(demand, CycleCost{}, limits);
    check(plan.rows[2] >= plan.rows[1] && plan.rows[1] >= plan.rows[0], "rows follow acceptance");
    check(plan.rows[0] == 2, "poor lane keeps its minimum");
    // Short histories fall back to the prior; out-of-range values are clamped.
    auto pair = lanes(2, 1, 8);
    const std::array<float, 1> only_first{1.0f};
    pair[0].accept = only_first;
    RowPlan same = plan_rows(pair, CycleCost{}, limits, 0.8);
    check(same.rows[0] == same.rows[1], "missing history equals the prior");
    const std::array<float, 3> wild{7.0f, -1.0f, 0.9f};
    pair[0].accept = wild;
    same = plan_rows(pair, CycleCost{}, limits, 0.8);
    check(same.rows[0] == 1, "clamped zero acceptance stops the lane");
}

void limit_tests() {
    // Sweep: the plan never exceeds a limit, a lane's bounds or its remaining tokens.
    const std::array<std::uint32_t, 4> totals{4, 8, 16, 32};
    std::uint32_t seed = 12345;
    auto next = [&seed](std::uint32_t mod) { seed = seed * 1664525u + 1013904223u; return (seed >> 8) % mod; };
    for (int trial = 0; trial < 4000; ++trial) {
        const std::uint32_t count = 1 + next(12);
        std::vector<LaneDemand> demand(count);
        std::vector<std::vector<float>> history(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            demand[i].lane = i;
            demand[i].remaining_tokens = next(12);
            demand[i].min_rows = 1 + next(2);
            demand[i].max_rows = 1 + next(8);
            demand[i].waited_ms = next(5) * 50.0;
            history[i].resize(next(9));
            for (float& value : history[i]) value = static_cast<float>(next(101)) / 100.0f;
            demand[i].accept = history[i];
        }
        RowLimits limits;
        limits.rows_per_forward = (trial % 3 == 0) ? 8u : 32u;
        if (trial % 2 == 0) limits.totals = totals;
        limits.max_lanes = (trial % 5 == 0) ? 0u : 8u;
        limits.max_cycle_ms = (trial % 4 == 0) ? 150.0 : 0.0;
        const RowPlan plan = plan_rows(demand, CycleCost{}, limits);
        check(plan.rows.size() == count, "one entry per demand");
        std::uint32_t running = 0;
        for (std::uint32_t i = 0; i < count; ++i) {
            const std::uint32_t rows = plan.rows[i];
            if (rows == 0) continue;
            ++running;
            const std::uint32_t floor = demand[i].min_rows;
            check(demand[i].remaining_tokens != 0, "finished lane not planned");
            check(rows >= floor, "running lane holds its minimum");
            check(rows <= std::max(floor, demand[i].max_rows), "lane max respected");
            check(rows <= std::max(floor, demand[i].remaining_tokens), "remaining tokens respected");
        }
        const std::uint32_t sum = total(plan);
        check(sum <= limits.rows_per_forward, "rows per forward respected");
        check(limits.max_lanes == 0 || running <= limits.max_lanes, "lane limit respected");
        check(plan.expected_tokens <= sum + 1e-9, "at most one token per row");
        const bool any = std::any_of(demand.begin(), demand.end(), [](const LaneDemand& d) { return d.remaining_tokens != 0; });
        check(any == (running != 0), "work is never left idle");
        if (running > 1) {
            check(limits.totals.empty() || std::find(totals.begin(), totals.end(), sum) != totals.end(), "total is qualified");
            check(limits.max_cycle_ms <= 0 || plan.expected_ms <= limits.max_cycle_ms, "latency guard respected");
        }
        const RowPlan again = plan_rows(demand, CycleCost{}, limits);
        check(again.rows == plan.rows && again.expected_tokens == plan.expected_tokens, "deterministic");
    }
    // Latency guard trims rows first, then lanes.
    RowLimits guard; guard.rows_per_forward = 32; guard.max_cycle_ms = 48 + 7.5 * 10;
    auto demand = lanes(4, 2, 8);
    RowPlan plan = plan_rows(demand, CycleCost{}, guard, 0.95);
    check(total(plan) == 10 && std::count(plan.rows.begin(), plan.rows.end(), 0u) == 0, "guard trims rows before lanes");
    guard.max_cycle_ms = 48 + 7.5 * 5;
    plan = plan_rows(demand, CycleCost{}, guard, 0.95);
    check(std::count(plan.rows.begin(), plan.rows.end(), 0u) == 2 && total(plan) == 5, "guard parks lanes when rows are not enough");
    // A one-row tail still holds the engine's physical minimum and counts one token.
    auto tail = lanes(1, 2, 8, 1);
    plan = plan_rows(tail, CycleCost{}, RowLimits{});
    check(plan.rows[0] == 2 && near(plan.expected_tokens, 1.0), "physical minimum on the last token");
}

void table_tests() {
    // The fixed table (8 rows for one lane, 4 for two, 2 for three or more) is what the planner
    // returns for a per-row cost that grows with the number of lanes, at the prior 0.8.
    RowLimits limits; limits.rows_per_forward = 16;
    CycleCost one; one.row_ms = 2.69;
    auto demand = lanes(1, 2, 8);
    check(plan_rows(demand, one, limits).rows == std::vector<std::uint32_t>{8}, "one lane: 8 rows");
    CycleCost two; two.row_ms = 10;
    demand = lanes(2, 2, 8);
    check(plan_rows(demand, two, limits).rows == std::vector<std::uint32_t>{4, 4}, "two lanes: 4 rows");
    CycleCost many; many.row_ms = 24;
    for (std::uint32_t count = 3; count <= 8; ++count) {
        demand = lanes(count, 2, 8);
        check(plan_rows(demand, many, limits).rows == std::vector<std::uint32_t>(count, 2u), "three or more lanes: 2 rows");
    }
    // One-row engines with eight rows per forward: one row per lane up to eight lanes.
    RowLimits small; small.rows_per_forward = 8; small.max_lanes = 8;
    for (std::uint32_t count = 1; count <= 12; ++count) {
        demand = lanes(count, 1, 1);
        const RowPlan plan = plan_rows(demand, CycleCost{}, small);
        check(total(plan) == std::min(count, 8u), "one row per lane up to eight");
        check(*std::max_element(plan.rows.begin(), plan.rows.end()) == 1, "never more than one row");
    }
}

void starvation_tests() {
    // 12 lanes, 8 per cycle, 1,000 cycles: the caller ages parked lanes and resets served ones.
    const std::array<std::uint32_t, 4> totals{4, 8, 16, 32};
    RowLimits limits; limits.rows_per_forward = 32; limits.totals = totals; limits.max_lanes = 8;
    auto demand = lanes(12, 2, 8, 1u << 30);
    std::vector<int> served(12, 0), gap(12, 0), worst(12, 0);
    for (int cycle = 0; cycle < 1000; ++cycle) {
        const RowPlan plan = plan_rows(demand, CycleCost{}, limits);
        check(std::count_if(plan.rows.begin(), plan.rows.end(), [](std::uint32_t r) { return r != 0; }) == 8, "eight lanes run");
        for (std::size_t i = 0; i < demand.size(); ++i) {
            if (plan.rows[i] != 0) { ++served[i]; gap[i] = 0; demand[i].waited_ms = 0; }
            else { worst[i] = std::max(worst[i], ++gap[i]); demand[i].waited_ms += plan.expected_ms; }
        }
    }
    check(*std::max_element(worst.begin(), worst.end()) <= 1, "no lane waits more than one cycle");
    check(*std::min_element(served.begin(), served.end()) >= 500, "every lane runs at least every other cycle");
    // Rows too: with a row limit below the lanes' minimum sum, parked lanes rotate in.
    RowLimits tight; tight.rows_per_forward = 8;
    auto wide = lanes(12, 2, 8, 1u << 30);
    std::vector<int> runs(12, 0);
    for (int cycle = 0; cycle < 1000; ++cycle) {
        const RowPlan plan = plan_rows(wide, CycleCost{}, tight);
        for (std::size_t i = 0; i < wide.size(); ++i) {
            if (plan.rows[i] != 0) { ++runs[i]; wide[i].waited_ms = 0; }
            else wide[i].waited_ms += plan.expected_ms;
        }
    }
    check(*std::min_element(runs.begin(), runs.end()) >= 300, "row-limited lanes rotate");
    // Longest-waiting first, lane id on ties, whatever the input order.
    auto order = lanes(3, 1, 1);
    order[0].lane = 7; order[1].lane = 3; order[2].lane = 5; order[2].waited_ms = 10;
    RowLimits single; single.max_lanes = 1;
    check(plan_rows(order, CycleCost{}, single).rows == std::vector<std::uint32_t>{0, 0, 1}, "longest wait first");
    order[2].waited_ms = 0;
    check(plan_rows(order, CycleCost{}, single).rows == std::vector<std::uint32_t>{0, 1, 0}, "ties break by lane id");
}

void snap_tests() {
    // Three lanes, each with its own history, snapped to a total of 8: the two rows beyond the
    // minimum go to the highest-value candidates.
    const std::array<std::uint32_t, 1> eight{8};
    RowLimits limits; limits.rows_per_forward = 8; limits.totals = eight;
    auto demand = lanes(3, 2, 8);
    const std::array<float, 4> low{1.0f, 0.9f, 0.2f, 0.2f};
    const std::array<float, 4> mid{1.0f, 0.9f, 0.6f, 0.1f};
    const std::array<float, 4> high{1.0f, 0.9f, 0.9f, 0.8f};
    demand[0].accept = low; demand[1].accept = mid; demand[2].accept = high;
    RowPlan plan = plan_rows(demand, CycleCost{}, limits);
    // Candidates: high row 3 = 0.81, high row 4 = 0.648, mid row 3 = 0.54, low row 3 = 0.18.
    check(plan.rows == std::vector<std::uint32_t>{2, 2, 4}, "snap keeps the highest-value rows");
    check(std::fabs(plan.expected_tokens - (3 * 1.9 + 0.81 + 0.648)) < 1e-5, "snap expected tokens");
    // Unsnapped, the same lanes buy more; the snapped plan is a prefix of the same ranking.
    RowLimits open; open.rows_per_forward = 32;
    const RowPlan free_plan = plan_rows(demand, CycleCost{}, open);
    for (std::size_t i = 0; i < 3; ++i) check(free_plan.rows[i] >= plan.rows[i], "snap only drops rows");
    // Equal values: earlier rows first, then lane id.
    auto flat = lanes(3, 1, 4);
    const std::array<std::uint32_t, 1> five{5};
    RowLimits tie; tie.rows_per_forward = 8; tie.totals = five;
    check(plan_rows(flat, CycleCost{}, tie).rows == std::vector<std::uint32_t>{2, 2, 1}, "ties break by lane");
    // Minimum rows that miss every total: pad up when the lanes allow it, else park.
    const std::array<std::uint32_t, 2> four_eight{4, 8};
    RowLimits pad; pad.rows_per_forward = 8; pad.totals = four_eight;
    auto three = lanes(3, 2, 8);
    CycleCost steep; steep.row_ms = 100;
    plan = plan_rows(three, steep, pad);
    check(total(plan) == 8 && std::count(plan.rows.begin(), plan.rows.end(), 0u) == 0, "pads up to a qualified total");
    auto capped = lanes(3, 2, 2);
    plan = plan_rows(capped, steep, pad);
    check(plan.rows == std::vector<std::uint32_t>{2, 2, 0}, "parks a lane when no total is reachable");
    capped[2].waited_ms = 5;
    plan = plan_rows(capped, steep, pad);
    check(plan.rows == std::vector<std::uint32_t>{2, 0, 2}, "the parked lane is the shortest-waiting one");
    // A lone lane that cannot reach a total still runs with its minimum.
    auto lone = lanes(1, 2, 3);
    plan = plan_rows(lone, CycleCost{}, pad);
    check(plan.rows == std::vector<std::uint32_t>{2}, "lone lane runs with its minimum");
    lone = lanes(1, 2, 6);
    RowLimits slow = pad; slow.max_cycle_ms = 1;
    plan = plan_rows(lone, CycleCost{}, slow);
    check(plan.rows == std::vector<std::uint32_t>{4}, "lone lane pads to the smallest total past the guard");
}
}  // namespace

int main() {
    cost_model_tests(); monotone_tests(); limit_tests(); table_tests(); starvation_tests(); snap_tests();
    std::cout << "row budget failures: " << failures << '\n';
    return failures ? 1 : 0;
}
