#pragma once

#include <functional>

#include "strix/deploy/footprint.hpp"

namespace strix::deploy {

struct RankBudget {
    std::uint32_t rank = 0, node = 0;
    std::uint64_t total_bytes = 0, free_bytes = 0, margin_bytes = 0, budget_bytes = 0;
    std::uint64_t kv_bytes = 0, after_kv_bytes = 0, after_workspace_bytes = 0;
    std::uint64_t required_bytes = 0, free_after_bytes = 0;
    Footprint footprint;
    bool fits = false;
    std::string reason;
};
struct Plan {
    Spec requested;
    std::vector<DeviceFacts> devices;
    Candidate chosen;
    std::vector<RankBudget> ranks;
    std::vector<std::string> reasons;
    bool refused = true;
    std::string refusal;
    [[nodiscard]] std::string to_json() const; // strix-deployment-v1, includes deployment_id
    [[nodiscard]] std::string hash() const; // canonical JSON, FNV-1a-64, 16 lowercase hex digits
};

// Per-rank admission shared by resolve() and the route planner (deploy/route.hpp): KV first, then workspace, weights,
// staging and extras; the margin is preserved and elastic extras are fitted last. `footprint` may throw; its message
// becomes the refusal reason.
[[nodiscard]] RankBudget fit_rank(std::uint32_t rank, std::uint32_t node, const DeviceFacts& device, std::uint64_t margin,
                                  std::uint32_t context, std::uint32_t slots, const std::function<Footprint()>& footprint);
[[nodiscard]] Plan resolve(const Spec&, const std::vector<DeviceFacts>&, const FootprintProvider&);

} // namespace strix::deploy
