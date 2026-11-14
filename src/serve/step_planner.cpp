#include "strix/serve/step_planner.hpp"

#include "strix/serve/short_first.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace strix::serve {
namespace {

constexpr std::size_t kNone = static_cast<std::size_t>(-1);

// Prompt lanes in arrival order (stable for equal orders).
std::vector<std::size_t> prompt_lanes_by_arrival(std::span<const StepLane> lanes) {
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < lanes.size(); ++i)
        if (lanes[i].prompt) out.push_back(i);
    std::stable_sort(out.begin(), out.end(), [&](std::size_t a, std::size_t b) { return lanes[a].order < lanes[b].order; });
    return out;
}

// True when a goes before b among lanes that are not aged. Arrival order breaks every tie.
bool ahead(const StepLane& a, const StepLane& b, const StepLimits& limits) {
    if (limits.prompt_order == PromptOrder::shortest) return a.prompt_remaining < b.prompt_remaining;
    const bool sa = a.prompt_remaining <= limits.short_first_rows, sb = b.prompt_remaining <= limits.short_first_rows;
    return sa && !sb;
}

// Position in `arrival` of the lane that runs first, or kNone. The first aged lane wins; if it cannot
// run, nobody does, so its starvation bound holds.
std::size_t head_position(std::span<const StepLane> lanes, const std::vector<std::size_t>& arrival, const StepLimits& limits) {
    if (arrival.empty()) return kNone;
    if (limits.short_first_rows == 0) return lanes[arrival[0]].eligible ? 0 : kNone;  // oldest first
    std::size_t best = kNone;
    for (std::size_t p = 0; p < arrival.size(); ++p) {
        const auto& lane = lanes[arrival[p]];
        if (lane.overtakes >= kMaxQueueOvertakes) return lane.eligible ? p : kNone;
        if (!lane.eligible) continue;
        if (best == kNone || ahead(lane, lanes[arrival[best]], limits)) best = p;
    }
    return best;
}

// Rows of one prompt unit. 0 = the family's own unit.
std::uint32_t unit_rows(const StepLimits& limits, bool decoding, double row_ms) {
    const auto units = limits.prompt_units;
    const bool bounded = limits.max_decode_gap_ms > 0.0 && decoding;
    if (!bounded) return units.empty() ? 0U : units.back();
    if (units.empty()) {
        if (!(row_ms > 0.0)) return 0U;
        const double rows = std::floor(limits.max_decode_gap_ms / row_ms);
        return static_cast<std::uint32_t>(std::clamp(rows, 1.0, static_cast<double>(std::numeric_limits<std::uint32_t>::max())));
    }
    // No cost known yet: probe with the smallest unit. No unit fits: the smallest is the closest.
    std::uint32_t pick = units.front();
    if (row_ms > 0.0)
        for (const auto unit : units)
            if (static_cast<double>(unit) * row_ms <= limits.max_decode_gap_ms) pick = std::max(pick, unit);
    return pick;
}

std::uint32_t clamp_rows(std::uint64_t rows) {
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(rows, std::numeric_limits<std::uint32_t>::max()));
}

}  // namespace

StepPlan StepPlanner::plan(std::span<const StepLane> lanes, const StepLimits& limits) {
    StepPlan out;
    share_ = limits.prompt_share > 0.0 ? limits.prompt_share : 0.0;
    planned_rows_ = 0;

    const std::size_t decode_cap = limits.co_forward ? limits.rows_per_forward : lanes.size();
    for (const auto& lane : lanes)
        if (!lane.prompt && out.decode_lanes.size() < decode_cap) out.decode_lanes.push_back(lane.lane);
    const bool decoding = !out.decode_lanes.empty();

    const auto arrival = prompt_lanes_by_arrival(lanes);
    if (arrival.empty()) {
        debt_ms_ = 0.0;
        return out;
    }
    if (decoding && debt_ms_ > 0.0) return out;  // the decoders are owed time

    const std::size_t head = head_position(lanes, arrival, limits);
    if (head == kNone) return out;

    const double row_ms = observed_rows_ ? observed_ms_ / static_cast<double>(observed_rows_) : limits.prompt_row_ms;
    const std::uint32_t unit = unit_rows(limits, decoding, row_ms);
    const auto lane_rows = [&](const StepLane& lane, std::uint64_t room) {
        std::uint64_t rows = std::min<std::uint64_t>(lane.prompt_remaining, room);
        if (lane.max_prompt_rows != 0) rows = std::min<std::uint64_t>(rows, lane.max_prompt_rows);
        return clamp_rows(rows);
    };

    if (!limits.co_forward) {
        const auto& lane = lanes[arrival[head]];
        const std::uint32_t rows = unit == 0 ? 0U : lane_rows(lane, unit);
        out.prompt.push_back({lane.lane, rows});
        out.prompt_before_decode = true;
        planned_rows_ = rows;
    } else {
        // The head first, then the other prompts in arrival order, until the forward is full.
        const bool whole_forward = limits.prompt_units.empty() && unit == 0;
        std::uint64_t room = limits.rows_per_forward > out.decode_lanes.size() ? limits.rows_per_forward - out.decode_lanes.size() : 0;
        const auto give = [&](const StepLane& lane) {
            if (room == 0 || !lane.eligible) return;
            const std::uint32_t rows = lane_rows(lane, whole_forward ? room : std::min<std::uint64_t>(room, unit));
            if (rows == 0) return;
            out.prompt.push_back({lane.lane, rows});
            room -= rows;
            planned_rows_ += rows;
        };
        give(lanes[arrival[head]]);
        for (std::size_t p = 0; p < arrival.size(); ++p)
            if (p != head) give(lanes[arrival[p]]);
    }
    if (!out.prompt.empty())
        for (std::size_t p = 0; p < head; ++p) out.overtaken.push_back(lanes[arrival[p]].lane);
    return out;
}

void StepPlanner::account(double prompt_ms, double decode_ms, double suspended_ms, std::uint64_t prompt_rows) {
    const double own_ms = std::max(0.0, prompt_ms - suspended_ms);
    debt_ms_ += share_ * own_ms;
    debt_ms_ -= decode_ms;
    const std::uint64_t rows = prompt_rows ? prompt_rows : planned_rows_;
    if (prompt_ms > 0.0 && rows != 0) {
        observed_ms_ += own_ms;
        observed_rows_ += rows;
    }
    if (prompt_ms > 0.0) planned_rows_ = 0;
}

void StepPlanner::reset() { *this = StepPlanner{}; }

}  // namespace strix::serve
