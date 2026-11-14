#pragma once
#include "strix/distributed_runtime.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace strix {
class HipDistributedTokenParticipant;
class NativeDistributedBackend;
enum class NativeDistributedPositionDependencyPolicy {
    reject = 0,
    materialized_input_position_v1 = 1,
};
// Classifies metadata from an already admitted canonical placement. This pure
// predicate is not a native execution or ownership proof.
[[nodiscard]] inline bool is_materialized_input_position_dependency(
    const DistributedPlacementPlan& plan,const DistributedExecutionBoundary& boundary) noexcept {
    return boundary.payload_bytes==0 && boundary.payloads.empty() &&
        boundary.source_node==0U && boundary.destination_node==1U &&
        boundary.source_step<boundary.destination_step && boundary.destination_step<plan.steps.size() &&
        plan.steps[boundary.source_step].step_id==boundary.source_step &&
        plan.steps[boundary.source_step].op==SerialOp::position_prepare &&
        plan.steps[boundary.source_step].node==boundary.source_node &&
        plan.steps[boundary.destination_step].step_id==boundary.destination_step &&
        plan.steps[boundary.destination_step].op==SerialOp::indexed_attention &&
        plan.steps[boundary.destination_step].node==boundary.destination_node;
}
// Logical completion ONLY: a replicated host position word, not DMA or a
// transferred payload. Only the actual native participant can mint it after
// local completion, exact position validation and the real two-ACK commit.
// Native data operations still require their independent live-lease retirement.
class NativeDistributedPositionCompletion final {
public:
    static constexpr std::size_t maximum_dependencies = 64;
    NativeDistributedPositionCompletion() = default;
    NativeDistributedPositionCompletion(const NativeDistributedPositionCompletion&) = delete;
    NativeDistributedPositionCompletion& operator=(const NativeDistributedPositionCompletion&) = delete;
    NativeDistributedPositionCompletion(NativeDistributedPositionCompletion&& other) noexcept
        : request_(other.request_), binding_(other.binding_), placement_(other.placement_),
          decision_(other.decision_), position_(other.position_), node_(other.node_),
          boundaries_(other.boundaries_), count_(other.count_), owner_(std::move(other.owner_)),
          valid_(std::exchange(other.valid_,false)) {}
    NativeDistributedPositionCompletion& operator=(NativeDistributedPositionCompletion&& other) noexcept {
        if(this!=&other) {
            request_=other.request_;binding_=other.binding_;placement_=other.placement_;
            decision_=other.decision_;position_=other.position_;node_=other.node_;
            boundaries_=other.boundaries_;count_=other.count_;owner_=std::move(other.owner_);
            valid_=std::exchange(other.valid_,false);
        }
        return *this;
    }
private:
    friend class HipDistributedTokenParticipant;
    friend class NativeDistributedBackend;
    DistributedRuntimeRequest request_;
    std::uint64_t binding_=0,placement_=0,decision_=0,position_=0;
    std::uint32_t node_=0;
    std::array<std::uint64_t,maximum_dependencies> boundaries_{};
    std::size_t count_=0;
    std::shared_ptr<const void> owner_;
    bool valid_=false;
};
} // namespace strix
