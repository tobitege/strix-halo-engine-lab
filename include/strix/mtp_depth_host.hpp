#pragma once
// Host adapter binding MtpDepthWindowController to an engine whose draft chain is one fixed-depth launch and whose
// verifier takes any row count 2..8 (CUDA Qwen serve). Same owners/mask/choose/observe sequence as the HIP window
// verifier (hip_mtp_window_verifier.hip), so equal inputs pick equal rows. Never alters committed tokens: it only
// picks the draft depth and how many drafts are verified (greedy verify is row-count invariant).
#include "strix/mtp_depth_window_controller.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <span>
#include <vector>

namespace strix {
class MtpDepthHost {
public:
    static constexpr std::uint32_t warm_uses=3; // graph keys (draft depth / verify rows) are eager, capture, replay
    // warm: windows per draft depth / verify rows discarded as graph eager/capture runs (0 = every window is a sample, as HIP).
    MtpDepthHost(std::uint32_t max_depth,float cli_min_p,bool enabled,std::uint32_t warm=warm_uses)
        :max_depth_(std::min<std::uint32_t>(max_depth,7U)),on_(enabled && max_depth>=1),warm_(warm) {
        if(!on_)return;
        min_p_=mtp_depth_domain_min_probability(cli_min_p);
        std::vector<std::uint32_t> rows;for(std::uint32_t r=2;r<=max_depth_+1U;++r)rows.push_back(r);
        ctrl_=std::make_unique<MtpDepthWindowController>(std::move(rows),min_p_,mtp_depth_domain_prior());
    }
    [[nodiscard]] bool on() const noexcept {return on_;}
    [[nodiscard]] float min_p() const noexcept {return min_p_;}
    // Draft depth to launch for the next window (cap = request/CLI limit).
    [[nodiscard]] std::uint32_t plan(std::uint32_t cap) {
        const auto m=std::min(cap,max_depth_);
        return on_?ctrl_->plan_depth(m):m;
    }
    // Rows to verify (incl. the pending token; 1 = serial step) for a chain with confidences `p`.
    [[nodiscard]] std::uint32_t select(std::span<const float> p,bool exact=false) {
        // HIP fits(): an owner needs its rows filled (candidates+1 >= rows)
        std::array<std::uint8_t,8> mask{};for(std::size_t n=0;n<owners();++n)mask[n]=(exact?n+2U==p.size()+1U:n+2U<=p.size()+1U)?1U:0U;
        drafted_=static_cast<std::uint32_t>(p.size());
        probs_.assign(p.begin(),p.end());
        const auto c=ctrl_->choose(p,p.size(),std::span<const std::uint8_t>(mask.data(),owners()));
        owner_=c;
        return c==MtpDepthWindowController::serial?1U:static_cast<std::uint32_t>(c)+2U;
    }
    void note_draft(std::uint32_t steps,double ms) {
        draft_steps_=steps;draft_ms_=ms;
        if(steps>0 && steps<=7U && draft_uses_[steps]++<warm_)cold_=true;
        ctrl_->note_draft(steps,ms);
    }
    // After the window: rows verified, tokens committed (accepted+1), wall ms of draft+verify+commit.
    void finish(std::uint32_t rows,std::size_t accepted,double wall_ms) {
        ctrl_->calibrate(probs_,accepted,rows-1U);
        bool cold=cold_;cold_=false;
        if(rows<=8U && verify_uses_[rows]++<warm_)cold=true;
        ctrl_->observe(owner_,cold?0:accepted+1U,wall_ms);
    }
    // Whole-step hosts (cohort step, draft+verify not separable): a lone stream's window launched `depth` drafts and
    // verified depth+1 rows; cost cell (depth+1, depth) then includes the draft chain. `wall_ms` covers the whole step.
    void finish_fixed(std::uint32_t depth,std::size_t accepted,double wall_ms,bool full) {
        if(!on_ || depth==0 || depth>max_depth_)return;
        const std::vector<float> ones(depth,1.0F);
        const auto rows=select(ones,true);
        ctrl_->calibrate(ones,accepted,depth);
        bool cold=!full || rows==1U;
        if(draft_uses_[depth]++<warm_ || verify_uses_[rows<=8U?rows:8U]++<warm_)cold=true;
        ctrl_->observe(owner_,cold?0:accepted+1U,wall_ms);
    }
    void skip() {ctrl_->observe(owner_,0,0);cold_=false;}
    [[nodiscard]] const MtpDepthWindowController& controller() const {return *ctrl_;}
private:
    [[nodiscard]] std::size_t owners() const {return max_depth_;}
    std::uint32_t max_depth_;bool on_;std::uint32_t warm_;float min_p_=0.7F;
    std::unique_ptr<MtpDepthWindowController> ctrl_;
    std::vector<float> probs_;
    std::size_t owner_=MtpDepthWindowController::serial;
    std::uint32_t drafted_=0,draft_steps_=0;double draft_ms_=0;bool cold_=false;
    std::array<std::uint32_t,9> draft_uses_{},verify_uses_{};
};
}
