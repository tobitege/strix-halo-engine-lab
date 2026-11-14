#include "strix/serve/row_budget.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace strix::serve {
namespace {

struct Lane {
    std::size_t index = 0;  // position in the demand span
    std::uint32_t id = 0;
    std::uint32_t min_rows = 1;
    std::uint32_t max_rows = 1;   // >= min_rows
    std::uint32_t useful_cap = 0; // rows that can still yield a token
    double waited_ms = 0;
};

struct Growth {
    double value = 0;  // P(this row yields a token)
    std::uint32_t row = 0;
    std::size_t lane = 0;  // position in the admitted list
    std::uint32_t id = 0;
};

double clean(double value) { return std::isfinite(value) && value > 0 ? value : 0.0; }

double clamp01(double value, double fallback) {
    if (!std::isfinite(value)) return fallback;
    return std::clamp(value, 0.0, 1.0);
}

// Conditional acceptance of row `row` (0-based) of one lane.
double accept_at(const LaneDemand& demand, std::uint32_t row, double prior) {
    if (row < demand.accept.size()) return clamp01(static_cast<double>(demand.accept[row]), prior);
    return row == 0 ? 1.0 : prior;
}

bool qualified(std::uint32_t total, const RowLimits& limits) {
    if (limits.rows_per_forward != 0 && total > limits.rows_per_forward) return false;
    if (limits.totals.empty()) return true;
    return std::find(limits.totals.begin(), limits.totals.end(), total) != limits.totals.end();
}

struct Solver {
    std::span<const LaneDemand> demands;
    CycleCost cost;
    const RowLimits& limits;
    double prior;

    double cycle_ms(std::size_t lanes, std::uint32_t rows) const {
        // Every running lane holds at least one row, so rows >= lanes.
        return cost.fixed_ms + cost.row_ms * rows + cost.lane_ms * static_cast<double>(lanes) +
               cost.draft_row_ms * static_cast<double>(rows - lanes);
    }

    // Plans the admitted lanes. `relaxed` ignores the latency guard and accepts an unqualified
    // minimum; it is used only for a single lane that cannot meet the limits.
    bool solve(const std::vector<Lane>& lanes, bool relaxed, RowPlan& plan) const {
        std::uint32_t min_total = 0;
        double min_tokens = 0;
        std::vector<Growth> growth;
        for (std::size_t i = 0; i < lanes.size(); ++i) {
            const Lane& lane = lanes[i];
            const LaneDemand& demand = demands[lane.index];
            min_total += lane.min_rows;
            const std::uint32_t top = std::min(lane.max_rows, std::max(lane.min_rows, lane.useful_cap));
            double value = 1.0;
            for (std::uint32_t row = 0; row < top; ++row) {
                value *= accept_at(demand, row, prior);
                if (row < lane.min_rows) {
                    if (row < lane.useful_cap) min_tokens += value;
                } else {
                    growth.push_back({value, row, i, lane.id});
                }
            }
        }
        // Values never rise along a lane, so this order keeps each lane's rows in sequence.
        std::sort(growth.begin(), growth.end(), [](const Growth& a, const Growth& b) {
            if (a.value != b.value) return a.value > b.value;
            if (a.row != b.row) return a.row < b.row;
            if (a.id != b.id) return a.id < b.id;
            return a.lane < b.lane;
        });

        const double floor_ms = 1e-9;
        bool found = false;
        std::size_t best = 0;
        double best_tokens = 0, best_ms = 0, tokens = min_tokens;
        for (std::size_t extra = 0; extra <= growth.size(); ++extra) {
            if (extra != 0) tokens += growth[extra - 1].value;
            const std::uint32_t total = min_total + static_cast<std::uint32_t>(extra);
            if (limits.rows_per_forward != 0 && total > limits.rows_per_forward) break;
            const double ms = std::max(cycle_ms(lanes.size(), total), floor_ms);
            if (relaxed) {
                // Smallest qualified total wins; the bare minimum stands when none is reachable.
                if (qualified(total, limits)) { found = true; best = extra; best_tokens = tokens; best_ms = ms; break; }
                if (extra == 0) { found = true; best = 0; best_tokens = tokens; best_ms = ms; }
                continue;
            }
            if (!qualified(total, limits)) continue;
            if (limits.max_cycle_ms > 0 && ms > limits.max_cycle_ms) continue;
            if (!found || tokens * best_ms > best_tokens * ms) {
                found = true; best = extra; best_tokens = tokens; best_ms = ms;
            }
        }
        if (!found) return false;

        std::fill(plan.rows.begin(), plan.rows.end(), 0u);
        for (const Lane& lane : lanes) plan.rows[lane.index] = lane.min_rows;
        for (std::size_t extra = 0; extra < best; ++extra) ++plan.rows[lanes[growth[extra].lane].index];
        plan.expected_tokens = best_tokens;
        plan.expected_ms = cycle_ms(lanes.size(), min_total + static_cast<std::uint32_t>(best));
        return true;
    }
};

}  // namespace

RowPlan plan_rows(std::span<const LaneDemand> demands, const CycleCost& cost, const RowLimits& limits,
                  double accept_prior) {
    RowPlan plan;
    plan.rows.assign(demands.size(), 0u);

    std::vector<Lane> waiting;
    waiting.reserve(demands.size());
    for (std::size_t i = 0; i < demands.size(); ++i) {
        const LaneDemand& demand = demands[i];
        if (demand.remaining_tokens == 0) continue;
        Lane lane;
        lane.index = i;
        lane.id = demand.lane;
        lane.min_rows = std::max<std::uint32_t>(demand.min_rows, 1);
        lane.max_rows = std::max(demand.max_rows, lane.min_rows);
        lane.useful_cap = demand.remaining_tokens;
        lane.waited_ms = clean(demand.waited_ms);
        // A lane whose minimum can never fit one forward is left parked.
        if (limits.rows_per_forward != 0 && lane.min_rows > limits.rows_per_forward) continue;
        waiting.push_back(lane);
    }
    std::sort(waiting.begin(), waiting.end(), [](const Lane& a, const Lane& b) {
        if (a.waited_ms != b.waited_ms) return a.waited_ms > b.waited_ms;
        if (a.id != b.id) return a.id < b.id;
        return a.index < b.index;
    });

    std::vector<Lane> admitted;
    std::uint32_t min_total = 0;
    for (const Lane& lane : waiting) {
        if (limits.max_lanes != 0 && admitted.size() >= limits.max_lanes) break;
        if (limits.rows_per_forward != 0 && lane.min_rows > limits.rows_per_forward - min_total) continue;
        min_total += lane.min_rows;
        admitted.push_back(lane);
    }
    if (admitted.empty()) return plan;

    const Solver solver{demands,
                        {clean(cost.fixed_ms), clean(cost.row_ms), clean(cost.lane_ms), clean(cost.draft_row_ms)},
                        limits, clamp01(accept_prior, 0.8)};
    while (!solver.solve(admitted, false, plan)) {
        if (admitted.size() == 1) {
            solver.solve(admitted, true, plan);
            break;
        }
        admitted.pop_back();  // shortest-waiting lane; it ages and returns first
    }
    return plan;
}

}  // namespace strix::serve
