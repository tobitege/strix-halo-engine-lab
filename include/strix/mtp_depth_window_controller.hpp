#pragma once
#include "strix/mtp_window_controller.hpp"
#include <array>
#include <span>
#include <string>
#include <vector>

namespace strix {
// Candidate host policy. The target owns every token/state decision.
class MtpDepthWindowController {
public:
    static constexpr std::size_t serial=MtpMinPWindowController::serial;
    using Prior=std::array<double,7>;
    struct CostStats {std::uint32_t rows=0,filled=0;std::uint64_t samples=0;double ms=0;std::uint64_t probes=0;};
    MtpDepthWindowController(std::vector<std::uint32_t>,float,Prior={.5,.5,.5,.5,.5,.5,.5});
    void note_draft(std::uint32_t steps,double ms);
    [[nodiscard]] bool worth_drafting(std::span<const float>) const;
    [[nodiscard]] std::size_t choose(std::span<const float>,std::size_t,std::span<const std::uint8_t>);
    void calibrate(std::span<const float>,std::size_t accepted,std::size_t verified);
    void observe(std::size_t owner,std::size_t emitted,double ms);
    void set_external_rows(std::size_t first,std::span<const double>);
    void clear_external_rows() noexcept {external_first_=serial;external_.clear();}
    // Draft depth to launch BEFORE the chain exists (hosts whose chain is one fixed-depth launch, e.g. the CUDA Qwen graph:
    // a min-p stop saves nothing there). Maximises expected tokens / (measured window cost of owner depth+1 + measured
    // draft cost of that depth) with the per-position acceptance calibrated so far (prior until 8 tests). A cell with no
    // measurement is explored first (at most 4 plans each); every 64th plan launches max_depth to refresh the dear cells.
    [[nodiscard]] std::uint32_t plan_depth(std::uint32_t max_depth);
    [[nodiscard]] std::vector<CostStats> cost_stats() const;
    [[nodiscard]] std::vector<CostStats> draft_stats() const;
private:
    struct Owner {
        std::uint32_t rows=0;bool warmed=false;
        std::array<MtpWindowCostEstimate,8> residual;
        std::array<std::uint64_t,8> last_used{},interval{},probes{};
    };
    [[nodiscard]] double acceptance(std::size_t,float) const;
    [[nodiscard]] double expected(std::uint32_t,std::span<const float>,bool lookahead=false) const;
    [[nodiscard]] double cost(const Owner&,std::size_t filled) const;
    [[nodiscard]] bool uncertain(const Owner&,std::size_t filled) const;
    static void validate(std::span<const float>);
    std::vector<Owner> owners_;
    std::array<MtpWindowCostEstimate,8> draft_;
    MtpWindowCostEstimate serial_;
    std::array<std::array<double,10>,7> tested_{},accepted_{};
    // Future priors come from complete-position data, never selected R8 outcomes.
    Prior prior_;
    float min_probability_;
    std::size_t filled_=0,external_first_=serial;
    std::uint32_t drafted_=0;
    double current_draft_ms_=0;
    bool noted_=false;
    std::uint64_t choices_=0,plans_=0;
    std::array<std::uint32_t,8> explored_{};
    std::vector<double> external_;
};
[[nodiscard]] bool mtp_depth_controller_enabled() noexcept;
[[nodiscard]] float mtp_depth_domain_min_probability(float fallback);
[[nodiscard]] MtpDepthWindowController::Prior mtp_depth_domain_prior();
[[nodiscard]] const char* mtp_depth_domain();
// The one resolver line every front end prints: controller on/off, why, domain, effective min-p.
[[nodiscard]] std::string mtp_depth_state_line(float cli_min_p);
}
