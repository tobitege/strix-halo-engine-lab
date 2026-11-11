#pragma once
#include "strix/deploy/binding.hpp"

namespace strix::qwen::deployment {
// A single resolved compiler-owner inventory. No independent model-size table.
class Provider final : public deploy::FootprintProvider {
public:
    Provider(deploy::Footprint owners, std::uint32_t context, std::uint32_t slots,
             std::string topology, std::string profile, std::string refusal = {},
             std::uint64_t slot_retained = 0, std::uint64_t slot_peak = 0, std::uint64_t slot_state = 0)
        : owners_(std::move(owners)), context_(context), slots_(slots), topology_(std::move(topology)),
          profile_(std::move(profile)), refusal_(std::move(refusal)),
          slot_retained_(slot_retained), slot_peak_(slot_peak), slot_state_(slot_state) {}
    std::string model() const override { return "qwen3.8"; }
    std::vector<std::string> topologies() const override { return {topology_}; }
    std::vector<deploy::Format> formats(const std::string&) const override { return {{"native", "native"}}; }
    std::uint32_t max_slots() const override { return slots_; }
    deploy::Footprint footprint(const deploy::Spec& s, const deploy::Candidate& c, std::uint32_t) const override {
        if (!refusal_.empty()) throw std::invalid_argument(refusal_);
        if (s.profile != "auto" && s.profile != profile_) throw std::invalid_argument("profile pin differs from active profile");
        if (c.context != context_ || !c.slots || c.slots > slots_)
            throw std::invalid_argument("Qwen candidate differs from resolved owner inventory");
        if ((c.topology == "replica" || c.topology == "pipeline") && c.ranks != 2)
            throw std::invalid_argument("Qwen replica/pipeline binding requires two measured nodes");
        auto f = owners_;
        // The primary stack is in owners_. Extra stacks retain independent
        // state/scratch/metadata; bridge peaks are paid for every runnable slot.
        if (slot_state_ > slot_retained_ || (c.slots > 1 && (!slot_retained_ || !slot_state_)))
            throw std::invalid_argument("Qwen extra slot owner inventory unavailable or inconsistent");
        f.kv_total = deploy::checked_add(f.kv_total, deploy::checked_mul(c.slots-1U, slot_state_));
        f.extra.emplace_back("http_extra_stack_scratch_metadata_external",
            deploy::checked_mul(c.slots-1U, slot_retained_-slot_state_));
        f.extra.emplace_back("http_bridge_slots", deploy::checked_mul(c.slots, slot_peak_));
        // Every pipeline rank retains the full model for rank-0 continuation.
        if (c.topology == "pipeline") f.staging = deploy::checked_add(f.staging, 32ULL << 20);
        return f;
    }
private:
    deploy::Footprint owners_;
    std::uint32_t context_, slots_;
    std::string topology_, profile_, refusal_;
    std::uint64_t slot_retained_, slot_peak_, slot_state_;
};
} // namespace strix::qwen::deployment
