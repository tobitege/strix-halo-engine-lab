#pragma once

#include "strix/direct_transport_native_session.hpp"
#include "strix/distributed_runtime.hpp"
#include "strix/distributed_position_completion.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include <utility>

namespace strix {

class DistributedNativeControlPlane {
public:
    virtual ~DistributedNativeControlPlane() = default;

    virtual void execute_remote_step(
        const DistributedStepPlacement& step) = 0;

    virtual void publish_boundary_transfer(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer) = 0;

    // Implementations must reconcile the logical request/plan, epoch, dtype,
    // boundary and message sequence before exposing native receive metadata.
    [[nodiscard]] virtual DistributedBoundaryTransfer
    await_boundary_transfer(
        const DistributedExecutionBoundary& boundary) = 0;
};

class DistributedNativeLocalExecutor {
public:
    virtual ~DistributedNativeLocalExecutor() = default;

    virtual void execute_local_step(
        const DistributedStepPlacement& step) = 0;
};

enum class DistributedInboundCompletionStatus {
    pending,
    complete,
    failed,
};

class DistributedInboundCompletion {
public:
    virtual ~DistributedInboundCompletion() = default;
    // Complete proves every consumer/reassembly operation has stopped
    // accessing every staged RX slice. A CPU callback return is insufficient.
    // An implementation must drain or retain outstanding consumer resources
    // safely on destruction; native GPU event/visibility proof is separate
    // qualification work.
    [[nodiscard]] virtual DistributedInboundCompletionStatus poll() = 0;
};

class DistributedBoundaryStagingProvider {
public:
    virtual ~DistributedBoundaryStagingProvider() = default;

    // Fixed C1 staging slots may remain owned until peer ACK and consumer
    // credit retire the transport operation, even after local DMA completion.
    [[nodiscard]] virtual bool requires_full_send_retirement() const noexcept { return false; }

    [[nodiscard]] virtual transport::TransportBufferView
    stage_outbound(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer) = 0;

    [[nodiscard]] virtual transport::TransportBufferView
    stage_inbound(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer) = 0;

    virtual void consume_inbound(
        const DistributedExecutionBoundary& boundary,
        const transport::TransportBufferView& staged) = 0;

    // Optional explicit error-only quiescence proof. Default never infers
    // release from a callback exception. Cleanup must not publish success.
    [[nodiscard]] virtual std::unique_ptr<DistributedInboundCompletion> cleanup_inbound(
        const DistributedExecutionBoundary&, const DistributedBoundaryTransfer&) { return {}; }

    // Providers retain any launched consumers safely across callback errors.
    // All sessions and pool allocations must outlive their completion proof.

    [[nodiscard]] virtual transport::TransportBufferView
    stage_outbound_slice(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer,
        const transport::LinkSlice& slice,
        std::uint16_t stripe_index,
        std::uint16_t stripe_count);

    [[nodiscard]] virtual transport::TransportBufferView
    stage_inbound_slice(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer,
        const transport::LinkSlice& slice,
        std::uint16_t stripe_index,
        std::uint16_t stripe_count);

    virtual void consume_inbound_slice(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer,
        const transport::LinkSlice& slice,
        std::uint16_t stripe_index,
        std::uint16_t stripe_count,
        const transport::TransportBufferView& staged);

    [[nodiscard]] virtual std::unique_ptr<DistributedInboundCompletion>
    finish_inbound(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer) = 0;
};

class HipDistributedTokenParticipant;
class NativeDistributedBackend;
struct NativeDistributedTokenLedger;
enum class NativeDistributedRetirementOutcome { completed_boundaries, aborted_attempt };
// Move-only, privately minted evidence of actual token operation retirement.
// NOT a token-commit/decision permit or physical transport qualification.
// The supplied composite owner retains the actual sessions, native provider,
// imported pools, staging and GPU source/consumer owners until this proof dies.
class NativeDistributedTokenRetirement final {
public:
    NativeDistributedTokenRetirement() = default;
    NativeDistributedTokenRetirement(const NativeDistributedTokenRetirement&) = delete;
    NativeDistributedTokenRetirement& operator=(const NativeDistributedTokenRetirement&) = delete;
    NativeDistributedTokenRetirement(NativeDistributedTokenRetirement&&) noexcept = default;
    NativeDistributedTokenRetirement& operator=(NativeDistributedTokenRetirement&&) noexcept = default;
    ~NativeDistributedTokenRetirement() = default;
    // Read-only preflight; does not consume or authorize model publication.
    void validate_matches(std::uint64_t binding_identity,std::uint64_t placement_identity,
        const DistributedRuntimeRequest&,std::uint32_t local_node,NativeDistributedRetirementOutcome) const;
private:
    friend class NativeDistributedBackend;
    friend class HipDistributedTokenParticipant;
    explicit NativeDistributedTokenRetirement(std::shared_ptr<NativeDistributedTokenLedger> ledger)
        : ledger_(std::move(ledger)) {}
    // Check first, consume last. No own-container allocation; native provider
    // readiness queries may themselves throw. Caller publishes no frontier yet.
    void consume_matches(std::uint64_t binding_identity, std::uint64_t placement_identity,
        const DistributedRuntimeRequest&, std::uint32_t local_node,
        NativeDistributedRetirementOutcome);
    std::shared_ptr<NativeDistributedTokenLedger> ledger_;
};

struct DistributedNativeBackendPolicy {
    std::uint64_t maximum_event_polls = 4096;
    std::uint64_t maximum_consumer_polls = 4096;
};

// A backend and all adapters sharing its sessions require a single serialized
// progress owner. Different requests can remain pending remotely, but calling
// these APIs concurrently from multiple threads is unsupported.
class NativeDistributedBackend final
    : public DistributedExecutionBackend {
public:
    NativeDistributedBackend(
        std::span<transport::DirectTransportNativeLinkSession*> sessions,
        DistributedNativeControlPlane& control,
        DistributedNativeLocalExecutor& local,
        DistributedBoundaryStagingProvider& staging,
        DistributedNativeBackendPolicy policy = {});

    // Optional one-token mode, configured before ANY dispatch. Fresh backend
    // adapters per token; persistent native sessions/lease sequences are reused.
    // Single serialized progress owner, including proof consumption. Limits are
    // closed to1024 local boundaries and64 links (<=65536 actual operations).
    void begin_token_retirement(const DistributedPlacementPlan&, const DistributedRuntimeRequest&,
        std::uint32_t local_node, std::uint64_t binding_identity,
        std::shared_ptr<const void> actual_transport_owner,
        NativeDistributedPositionDependencyPolicy = NativeDistributedPositionDependencyPolicy::reject);
    // Logical position completion does not retire any native data operation.
    void complete_position_dependencies(NativeDistributedPositionCompletion&&);
    [[nodiscard]] NativeDistributedTokenRetirement retire_token();
    // Missing boundaries are explicitly aborted, not completed. Every ACTUAL
    // operation must still be fully retired; a failed backend cannot mint it.
    [[nodiscard]] NativeDistributedTokenRetirement retire_aborted_token();

    void execute_local_step(
        const DistributedStepPlacement& step) override;

    void execute_remote_step(
        const DistributedStepPlacement& step) override;

    void send_boundary(
        const DistributedExecutionBoundary& boundary,
        const DistributedBoundaryTransfer& transfer) override;

    [[nodiscard]] DistributedBoundaryTransfer
    receive_boundary(
        const DistributedExecutionBoundary& boundary) override;

private:
    [[nodiscard]] transport::DirectTransportNativeLinkSession&
    session_for_link(
        std::size_t link_index) const;

    void wait_send_reusable(
        transport::DirectTransportNativeLinkSession& session,
        transport::TransferHandle handle) const;

    void wait_receive_complete(
        transport::DirectTransportNativeLinkSession& session,
        transport::TransferHandle handle) const;

    void wait_receive_reusable(
        transport::DirectTransportNativeLinkSession& session,
        transport::TransferHandle handle) const;

    void require_healthy() const;
    void begin_boundary(const DistributedExecutionBoundary&, const transport::TransferDescriptor&, bool inbound);
    void record_operation(transport::DirectTransportNativeLinkSession&, transport::TransferHandle,
        const transport::TransferDescriptor&, bool inbound);
    void finish_boundary(const DistributedExecutionBoundary&, bool inbound);
    [[nodiscard]] NativeDistributedTokenRetirement retire(NativeDistributedRetirementOutcome);

    std::span<transport::DirectTransportNativeLinkSession*> sessions_;
    DistributedNativeControlPlane* control_ = nullptr;
    DistributedNativeLocalExecutor* local_ = nullptr;
    DistributedBoundaryStagingProvider* staging_ = nullptr;
    DistributedNativeBackendPolicy policy_{};
    bool failed_ = false;
    bool dispatched_ = false;
    std::shared_ptr<NativeDistributedTokenLedger> token_;
    // Retain failed/pending proofs on error, together with consumer-owned
    // session slots. No credit/repost or further work is permitted afterward.
    std::unique_ptr<DistributedInboundCompletion> inbound_completion_;
};

using SingleLinkNativeDistributedBackend =
    NativeDistributedBackend;

}  // namespace strix
