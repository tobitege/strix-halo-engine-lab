#pragma once
#include "strix/mtp_window_controller.hpp"
#include <array>
#include <cstdint>
#include <map>
#include <span>
#include <string_view>
#include <vector>

namespace strix {
// Plan rule of the joint controller (measurement; production = 0.8 prior, owners up to R8). pooled: at plan
// time a depth's acceptance prior is the request's own pooled acceptance over all depths instead of 0.8, so thin
// deep history does not inherit an optimistic prior. cap: the largest planned owner (2, 4 or 8).
// calibrated: after drafting, a predictor draft's acceptance is its probability calibrated by observed outcome
// per probability bin (the slot's bins, shrunk to the controller-wide bins), instead of the probability blended
// with the slot's per-depth history (which, once seen, barely depends on the draft's own confidence).
// calibrated_min_members gates calibration by the joint group's member count at plan time.
// budget: plan() and choose() take each request's rows from the shared row budget (strix/serve/row_budget.hpp):
// a cycle costs fixed + per-row + per-request milliseconds, so draft rows follow acceptance and shrink as the lanes
// of a cycle grow. The serving default (STRIX_SERVE_ROW_BUDGET=0 turns it off); also the rule name "budget".
// Scheduling only, as every other rule.
// measured ("budget-measured"): through eight lanes, plan() and choose() price each owner topology (#R2,#R4,#R8)
// by its own measured cycle ms (>= 3 samples) instead of the affine fit, and maximise expected tokens per ms
// exactly over every owner assignment; above the lane bound the bounded probes run too, so 7-8 lanes can leave
// the fixed two rows once a deeper topology is measured faster per token. No measured topology fits: the budget
// rule's plan. Scheduling only. measured_lanes is a bit mask by live lane count (bit L = L lanes): "budget-measured" sets
// 1..8, "budget-fit" none, plain "budget" (the serving default) mtp_serve_row_measured_lanes().
struct MtpBatchPlanRule {bool pooled=false; bool calibrated=false; std::uint32_t cap=8; std::size_t calibrated_min_members=0;
    bool budget=false; std::uint32_t measured_lanes=0;};
// Lane counts where the budget rule prices by measured topologies (STRIX_SERVE_ROW_MEASURED: auto = the measured
// decision table kMtpRowMeasuredAutoLanes, 1 = every count 1..8, 0 = none, the affine fit everywhere).
[[nodiscard]] std::uint32_t mtp_serve_row_measured_lanes() noexcept;
[[nodiscard]] const char* mtp_serve_row_measured_choice() noexcept; // "auto", "1", "0" or "list" (e.g. "7,8")
[[nodiscard]] std::uint32_t mtp_serve_row_measured_lane_list(std::string_view text) noexcept;
// Decision table (receipts 16-nov, one node, served Qwen, strict ABAB in one process, ids identical): lane counts where
// the measured rule's decode-window tok/s beat the affine fit. 7 lanes: 102.4 -> 107.9 (+5.4 %, R4 on every lane at
// 197 ms against R2 at 129 ms). Not set: 2 (+6.9 % then -0.7 %: flat), 4 (-2.8 %, -4.1 %), 6 (+1.0 %, -2.7 %): the
// draft-confidence yield over-promises the marginal R2 -> R4 lane there (also with the calibrated acceptance bins:
// -2.7 %, -3.2 % on clean pairs); 8 (+1.1 %, then +5.5 % with yield calibration, ids identical, but small kfd
// evictions in every baseline arm: provisional until a clean receipt; STRIX_SERVE_ROW_MEASURED=7,8 turns it on).
// Qualification boundary: this controller's only consumer is the HIP MTP greedy runtime (served Qwen joint cycle,
// hip_mtp_greedy_runtime.hip), and the table was measured on that device class only. Another model, backend or device
// binding this controller starts from 0 (the affine fit) until its own ABAB receipt qualifies a lane count.
inline constexpr std::uint32_t kMtpRowMeasuredAutoLanes=(1U<<7U);
// Yield calibration memory: per observed cycle the topology's delivered/predicted totals keep this share (about the
// last eight cycles, like the topology timing), so a workload change recalibrates instead of being outvoted.
inline constexpr double kMtpYieldKeep=0.875;
// The batched decode runs its joint cycle with the budget rule. On by default; STRIX_SERVE_ROW_BUDGET=0 (or the
// baseline mode) restores the fixed rows-per-request table.
[[nodiscard]] bool mtp_serve_row_budget_enabled() noexcept;
// Most decoding requests the budget plans for (STRIX_SERVE_ROW_BUDGET_MAX_LANES, default 6, 0..8). Above it every
// request gets the fixed table's two rows: measured, the budget wins at 2 to 6 requests and loses at 8.
[[nodiscard]] std::uint32_t mtp_serve_row_budget_max_lanes() noexcept;
// Most decoding requests one joint group (one target forward) takes above the lane bound, where every request has
// the fixed two rows: STRIX_SERVE_JOINT_GROUP_LANES, 8..16 (16 = 32 rows, the widest shared step); default 16 (one
// weight read for up to 16 requests, same ids as 8; 8 = old form, one forward per eight). "ab" alternates 8 and 16 by joint cycle (measurement).
// Scheduling only: every row keeps its arithmetic in the 9..32-row wide steps.
inline constexpr std::uint32_t kMtpJointGroupLanesMax=16U;
[[nodiscard]] std::uint32_t mtp_serve_joint_group_lanes(std::uint64_t cycle=0) noexcept;
// Joint scheduling only: all selections still come from each request's target.
// A non-drafting request normally uses R2, commits one row and discards padding.
class MtpBatchWindowController {
public:
    struct Request {
        std::uint32_t slot=0, limit=8;
        std::span<const float> probabilities;
        // Candidates from this index on come from another source (context rows): never calibrated as predictor.
        std::size_t external_first=SIZE_MAX;
    };
    using Rule=MtpBatchPlanRule;
    // Unset: calibrated-c3, or production under baseline. Explicit rules win over that default.
    [[nodiscard]] static Rule rule_from_env(bool baseline=false);
    // "calibrated-c3", "calibrated", "pooled", "cap4", "cap2", "budget", "budget-measured"; production/0/unknown = production.
    [[nodiscard]] static Rule parse_rule(std::string_view text);
    // Measurement: the rule may change between admissions (in-process A/B); histories and costs are kept.
    void set_rule(Rule rule) noexcept {rule_=rule;}
    [[nodiscard]] const Rule& rule() const noexcept {return rule_;}
    // Measurement (controlled in-process A/B): the state shared across requests (topology costs, draft cost,
    // exploration counters, controller-wide calibration bins) saved once and restored before each measured wave,
    // so every arm starts from the same warmed controller. Per-slot histories are reset by admission as always.
    void save_shared() {saved_={costs_,draft_,plans_,tested_,accepted_,yields_,true};}
    [[nodiscard]] bool restore_shared() {
        if(!saved_.valid) return false;
        costs_=saved_.costs;draft_=saved_.draft;plans_=saved_.plans;tested_=saved_.tested;accepted_=saved_.accepted;yields_=saved_.yields;
        probe_.clear();
        return true;
    }
    explicit MtpBatchWindowController(std::size_t slots,Rule rule={}):history_(slots),rule_(rule) {}
    // The last plan(): its expected tokens and ms (target estimate + draft estimate), and whether it was a probe.
    struct PlanInfo {double expected=0.0, ms=0.0; bool probe=false;};
    [[nodiscard]] const PlanInfo& last_plan() const noexcept {return last_;}
    // Requests per joint group for `count` live requests: above the budget rule's lane bound the plan is the fixed
    // two rows per request (no search), so a group holds up to mtp_serve_joint_group_lanes(cycle); otherwise 8.
    [[nodiscard]] std::size_t max_group_lanes(std::size_t count,std::uint64_t cycle=0) const noexcept {
        return rule_.budget && count>mtp_serve_row_budget_max_lanes()?mtp_serve_joint_group_lanes(cycle):8U;
    }
    // Budget rule cost model: a cycle costs fixed + row x physical rows + lane x lanes ms. Priors 28 + 6 x rows
    // + 6 x lanes (one lane: 34 + 6 x rows; STRIX_SERVE_ROW_COST=fixed,row,lane pins them); once the measured
    // topologies identify the three terms, their least-squares fit.
    struct BudgetCost {double fixed_ms=28.0,row_ms=6.0,lane_ms=6.0; bool fitted=false;};
    [[nodiscard]] BudgetCost budget_cost() const;
    void resize_slots(std::size_t slots) {history_.resize(slots);}
    void reset(std::uint32_t slot) {history_.at(slot)={};}
    // Before paying for drafts: rows 0 means no draft; otherwise draft rows-1.
    [[nodiscard]] std::vector<std::uint32_t> plan(std::span<const Request>,std::uint32_t budget);
    // After drafting: choose all owners together, with the draft cost already sunk.
    [[nodiscard]] std::vector<std::uint32_t> choose(std::span<const Request>,std::uint32_t budget) const;
    void observe(std::span<const Request>,std::span<const std::uint32_t> rows,
        std::span<const std::uint32_t> accepted,double draft_ms,double target_ms,std::uint32_t draft_steps);
private:
    struct History {std::array<double,7> seen{},hits{};std::array<double,10> tested{},accepted{};};
    std::vector<History> history_;
    std::array<double,10> tested_{},accepted_{}; // controller-wide predictor calibration bins
    // Costs belong to a whole topology, never to an individual session owner.
    std::map<std::array<std::uint32_t,3>,MtpWindowCostEstimate> costs_;
    MtpWindowCostEstimate draft_;
    // Measured rule: per topology, the tokens the plan model predicted and the tokens the cycles emitted (both summed
    // over executed cycles), so a topology's expected yield is calibrated by what it delivered.
    std::map<std::array<std::uint32_t,3>,std::array<double,2>> yields_;
    std::array<std::uint64_t,9> plans_{};
    std::vector<std::uint32_t> probe_;
    struct Shared {
        std::map<std::array<std::uint32_t,3>,MtpWindowCostEstimate> costs;MtpWindowCostEstimate draft;
        std::array<std::uint64_t,9> plans{};std::array<double,10> tested{},accepted{};
        std::map<std::array<std::uint32_t,3>,std::array<double,2>> yields;bool valid=false;
    } saved_;
    Rule rule_;
    std::size_t plan_members_=0; // latched before drafting; not server capacity or verify row count
    PlanInfo last_;
    [[nodiscard]] double expected(const Request&,std::uint32_t rows,bool prior) const;
    [[nodiscard]] double cost(std::span<const std::uint32_t>) const;
    [[nodiscard]] static std::size_t calibration_bin(double p);
    [[nodiscard]] PlanInfo estimate(std::span<const Request>,std::span<const std::uint32_t> rows,bool prior) const;
    [[nodiscard]] std::vector<std::uint32_t> search(std::span<const Request>,std::uint32_t,bool prior,
        PlanInfo* info=nullptr) const;
    [[nodiscard]] std::vector<std::uint32_t> budget_rows(std::span<const Request>,std::uint32_t,bool prior,
        PlanInfo* info=nullptr) const;
    // The measured rule's plan, empty when no measured topology fits (the caller keeps the budget plan).
    [[nodiscard]] std::vector<std::uint32_t> measured_rows(std::span<const Request>,std::uint32_t,bool prior,
        PlanInfo* info=nullptr) const;
    // The measured rule plans and probes up to eight lanes, above the budget's lane bound too.
    [[nodiscard]] bool measured_plans(std::size_t lanes) const noexcept {return lanes<=8U && ((rule_.measured_lanes>>lanes)&1U);}
};
} // namespace strix
