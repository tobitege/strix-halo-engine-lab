#pragma once

#include "strix/device/budget.hpp"
#include "strix/device/declaration.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// strix-memory-plan-v1: from a resolved budget (strix/device/budget.hpp) and what a route needs, where the weights
// live (VRAM / pinned host / the model file), how many positions and session slots fit, and, when nothing fits, the
// minimum budget that would. Pure arithmetic shared by every route and backend; a route that measures its own parts
// at load (DS41 / GLM through tierstream's ss_budget_plan) feeds the same caps and reports its measured plan beside
// this one. Also sizes planned shapes (`count = N` identical devices) with a bandwidth-model prediction.
//
// Order: the device's fixed parts and the resident weights first, then the sessions (positions before slots: a
// request's context is kept and slots drop first, then positions halve down to min_positions), then the tiered
// weights fill what is left in VRAM, then pinned host up to the host budget, then the file.
namespace strix::deploy {

struct RouteNeeds {
    std::string route;                    // "ds41-replica", "qwen-serve", "glm-rtxpf", ...
    std::uint64_t device_fixed = 0;       // context, workspaces, graph pools, drafter: always on the device
    std::uint64_t weights_resident = 0;   // weights that never leave the device (trunk, attention, dense)
    std::uint64_t tier_units = 0;         // weights that may live in host RAM or the file, in units
    std::uint64_t tier_unit_bytes = 0;    // one unit (a routed expert)
    std::uint64_t min_device_units = 0;   // the device arena's minimum
    bool host_tier = false, file_tier = false;
    std::uint64_t host_fixed = 0;         // process commit before the tiers (incl. the engine's pinned buffers)
    std::uint64_t session_fixed = 0, session_per_position = 0;
    std::uint32_t positions = 0, min_positions = 0, slots = 1, min_slots = 1;
    std::uint64_t margin_bytes = 0;
    // Prediction inputs (optional): bytes read per decoded token at C1, the tiered share of them, activation bytes
    // handed between pipeline stages per token, and the fraction of memory bandwidth the route reached on our card.
    std::uint64_t bytes_per_token = 0, tiered_bytes_per_token = 0, hop_bytes_per_token = 0;
    double decode_efficiency = 0.0;
};

// strix-route-needs-v1 TOML: `schema`, `route`, then any RouteNeeds field by name (integers in bytes / counts;
// host_tier / file_tier booleans; decode_efficiency a number). Unknown keys are errors (std::invalid_argument).
[[nodiscard]] RouteNeeds parse_route_needs(std::string_view toml);

struct MemoryPlan {
    device::MemoryBudget budget;
    std::string route;
    std::uint64_t device_fixed = 0, weights_resident = 0, sessions_bytes = 0, margin_bytes = 0;
    std::uint64_t units_device = 0, units_host = 0, units_file = 0, unit_bytes = 0;
    std::uint32_t positions = 0, slots = 0;
    std::uint64_t device_used = 0, host_used = 0;
    bool refused = false;
    std::string refusal;
    std::uint64_t min_device_bytes = 0, min_host_bytes = 0; // the smallest budget that fits the minimum request
    std::vector<std::string> notes;                         // what moved and why, in plain words
    [[nodiscard]] std::string json() const;
};

[[nodiscard]] MemoryPlan plan_memory(const device::MemoryBudget& budget, const RouteNeeds& needs);

// N identical devices of one planned entry (`profile`, `count`, `interconnect`): layers split evenly, one plan per
// device, and the predicted C1 decode from the bandwidth model (each stage reads its share at the profile's
// bandwidth x decode_efficiency, spilled tiered bytes cross the profile's host link, plus N-1 hops on the
// interconnect). Prediction only: no device of the shape is measured.
struct ShapePlan {
    std::string shape;          // "2x rtx-5090-32 over pcie5x16"
    std::uint32_t count = 1;
    MemoryPlan per_device;
    double predicted_decode_tok_s = 0; // 0 = not predicted (missing inputs)
    double predicted_ms_per_token = 0, hop_ms = 0, spill_ms = 0;
    [[nodiscard]] std::string json() const;
};
[[nodiscard]] ShapePlan plan_shape(const device::Descriptor& shape, const device::Descriptor* host, const RouteNeeds& needs);

} // namespace strix::deploy
