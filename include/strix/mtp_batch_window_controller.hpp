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
struct MtpBatchPlanRule {bool pooled=false; bool calibrated=false; std::uint32_t cap=8; std::size_t calibrated_min_members=0;
    bool budget=false;};
// The batched decode runs its joint cycle with the budget rule. On by default; STRIX_SERVE_ROW_BUDGET=0 (or the
// baseline mode) restores the fixed rows-per-request table.
[[nodiscard]] bool mtp_serve_row_budget_enabled() noexcept;
// Most decoding requests the budget plans for (STRIX_SERVE_ROW_BUDGET_MAX_LANES, default 6, 0..8). Above it every
// request gets the fixed table's two rows: measured, the budget wins at 2 to 6 requests and loses at 8.
[[nodiscard]] std::uint32_t mtp_serve_row_budget_max_lanes() noexcept;
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
    // "calibrated-c3", "calibrated", "pooled", "cap4", "cap2", "budget"; production/0/unknown = production.
    [[nodiscard]] static Rule parse_rule(std::string_view text);
    // Measurement: the rule may change between admissions (in-process A/B); histories and costs are kept.
    void set_rule(Rule rule) noexcept {rule_=rule;}
    [[nodiscard]] const Rule& rule() const noexcept {return rule_;}
    // Measurement (controlled in-process A/B): the state shared across requests (topology costs, draft cost,
    // exploration counters, controller-wide calibration bins) saved once and restored before each measured wave,
    // so every arm starts from the same warmed controller. Per-slot histories are reset by admission as always.
    void save_shared() {saved_={costs_,draft_,plans_,tested_,accepted_,true};}
    [[nodiscard]] bool restore_shared() {
        if(!saved_.valid) return false;
        costs_=saved_.costs;draft_=saved_.draft;plans_=saved_.plans;tested_=saved_.tested;accepted_=saved_.accepted;probe_.clear();
        return true;
    }
    explicit MtpBatchWindowController(std::size_t slots,Rule rule={}):history_(slots),rule_(rule) {}
    // The last plan(): its expected tokens and ms (target estimate + draft estimate), and whether it was a probe.
    struct PlanInfo {double expected=0.0, ms=0.0; bool probe=false;};
    [[nodiscard]] const PlanInfo& last_plan() const noexcept {return last_;}
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
    std::array<std::uint64_t,9> plans_{};
    std::vector<std::uint32_t> probe_;
    struct Shared {
        std::map<std::array<std::uint32_t,3>,MtpWindowCostEstimate> costs;MtpWindowCostEstimate draft;
        std::array<std::uint64_t,9> plans{};std::array<double,10> tested{},accepted{};bool valid=false;
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
};
} // namespace strix
