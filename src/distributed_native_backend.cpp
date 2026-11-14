#include "strix/distributed_native_backend.hpp"

#include <limits>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strix {
struct NativeDistributedTokenLedger {
    struct Link {
        transport::DirectTransportNativeLinkSession* session=nullptr;
        transport::DirectTransportLease lease;
    };
    struct Boundary { DistributedExecutionBoundary value; bool inbound=false, started=false, complete=false, logical_position=false; };
    struct Operation {
        std::size_t link=0;
        transport::TransferHandle handle=0;
        transport::TransferDescriptor descriptor;
        bool inbound=false;
    };
    DistributedRuntimeRequest request;
    std::uint64_t placement=0,binding=0;
    std::uint32_t node=0;
    NativeDistributedPositionDependencyPolicy position_policy=NativeDistributedPositionDependencyPolicy::reject;
    std::shared_ptr<const void> actual_owner;
    DistributedBoundaryStagingProvider* staging=nullptr;
    std::vector<Link> links;
    std::vector<Boundary> boundaries;
    std::vector<Operation> operations;
    std::vector<std::unique_ptr<DistributedInboundCompletion>> consumers;
    NativeDistributedRetirementOutcome outcome=NativeDistributedRetirementOutcome::completed_boundaries;
    bool sealed=false,consumed=false,failed=false;
};
namespace {

// Token-only fail-closed latch also covers metadata/staging allocation errors
// before the legacy prepared-operation try/catch. Legacy mode remains unchanged.
class NativeTokenFailureLatch final {
public:
    NativeTokenFailureLatch(std::shared_ptr<NativeDistributedTokenLedger> token,bool& failed) noexcept
        :token_(std::move(token)),failed_(failed),exceptions_(std::uncaught_exceptions()) {}
    ~NativeTokenFailureLatch() noexcept {
        if(token_ && std::uncaught_exceptions()>exceptions_) {failed_=true;token_->failed=true;}
    }
private:
    std::shared_ptr<NativeDistributedTokenLedger> token_;
    bool& failed_;
    int exceptions_;
};

struct NativePreparedTransfer {
    transport::DirectTransportNativeLinkSession* session = nullptr;
    transport::TransferHandle handle = 0;
    transport::TransferDescriptor descriptor;
    transport::TransportBufferView staged;
    transport::LinkSlice slice;
    std::uint16_t stripe_index = 0;
};

void require_default_single_slice(
    const DistributedBoundaryTransfer& transfer,
    const transport::LinkSlice& slice,
    std::uint16_t stripe_index,
    std::uint16_t stripe_count) {
    if (stripe_count != 1 ||
        stripe_index != 0 ||
        transfer.link_plan.kind !=
            transport::PlanKind::single_link ||
        slice.byte_offset != 0 ||
        slice.byte_length !=
            transfer.descriptor.byte_length) {
        throw std::logic_error(
            "staging provider does not implement striped boundary access");
    }
}

std::uint16_t validate_slices(
    const DistributedBoundaryTransfer& transfer,
    std::size_t session_count) {
    const auto count =
        transfer.link_plan.slices.size();
    if (count == 0 ||
        count >
            static_cast<std::size_t>(
                std::numeric_limits<std::uint16_t>::max()) ||
        transfer.descriptor.byte_length == 0) {
        throw std::invalid_argument(
            "native distributed transfer has invalid stripe count/payload");
    }

    if (transfer.link_plan.kind ==
            transport::PlanKind::single_link &&
        count != 1) {
        throw std::invalid_argument(
            "single-link transfer plan must contain exactly one slice");
    }
    if (transfer.link_plan.kind ==
            transport::PlanKind::striped &&
        count < 2) {
        throw std::invalid_argument(
            "striped transfer plan requires at least two slices");
    }
    if (transfer.link_plan.kind != transport::PlanKind::single_link &&
        transfer.link_plan.kind != transport::PlanKind::striped) {
        throw std::invalid_argument(
            "unavailable transfer plan cannot reach native backend");
    }

    std::vector<bool> used_links(
        session_count,
        false);
    std::uint64_t expected_offset = 0;
    for (const auto& slice :
         transfer.link_plan.slices) {
        if (slice.link_index >=
                session_count ||
            slice.byte_length == 0 ||
            slice.byte_length > std::numeric_limits<std::uint32_t>::max() ||
            slice.byte_offset !=
                expected_offset ||
            slice.byte_length >
                transfer.descriptor.byte_length -
                    expected_offset) {
            throw std::invalid_argument(
                "native distributed transfer slices are not an exact contiguous cover");
        }
        if (transfer.link_plan.kind ==
                transport::PlanKind::striped &&
            used_links[slice.link_index]) {
            throw std::invalid_argument(
                "striped native transfer requires distinct physical links");
        }
        used_links[slice.link_index] = true;
        expected_offset +=
            slice.byte_length;
    }

    if (expected_offset !=
        transfer.descriptor.byte_length) {
        throw std::invalid_argument(
            "native distributed transfer slices do not cover logical payload");
    }

    return static_cast<std::uint16_t>(
        count);
}

transport::TransferDescriptor slice_descriptor(
    const transport::TransferDescriptor& logical,
    const transport::LinkSlice& slice,
    const DistributedNativeSliceIdentity& identity) {
    auto descriptor =
        logical;
    descriptor.byte_length =
        slice.byte_length;
    descriptor.connection_epoch = identity.connection_epoch;
    descriptor.sequence = identity.sequence;
    return descriptor;
}

transport::GpuCommitRecord make_commit(
    const transport::TransferDescriptor& physical,
    const transport::LinkSlice& slice,
    std::uint16_t stripe_index,
    std::uint16_t stripe_count) {
    if (slice.byte_length >
        std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "native direct stripe payload exceeds 32-bit commit record");
    }

    transport::GpuCommitRecord record;
    record.protocol_version =
        physical.protocol_version;
    record.commit_word =
        transport::gpu_commit_word;
    record.connection_epoch =
        physical.connection_epoch;
    record.sequence =
        physical.sequence;
    record.payload_bytes =
        static_cast<std::uint32_t>(
            slice.byte_length);
    record.stripe_index =
        stripe_index;
    record.stripe_count =
        stripe_count;
    return record;
}

bool retirement_request_equal(const DistributedRuntimeRequest& a,const DistributedRuntimeRequest& b) noexcept {
    return a.connection_epoch==b.connection_epoch && a.request_id==b.request_id &&
        a.generation==b.generation && a.dtype_id==b.dtype_id;
}
bool retirement_boundary_equal(const DistributedExecutionBoundary& a,const DistributedExecutionBoundary& b) {
    return a.boundary_id==b.boundary_id && a.source_step==b.source_step && a.destination_step==b.destination_step &&
        a.source_node==b.source_node && a.destination_node==b.destination_node &&
        a.payload_bytes==b.payload_bytes && a.payloads==b.payloads;
}
void require_retirement_live(const NativeDistributedTokenLedger& ledger) {
    if(!ledger.actual_owner || ledger.failed || ledger.consumed)
        throw std::logic_error("native token retirement lost its owner or was already consumed");
    if(!ledger.staging || !ledger.staging->requires_full_send_retirement())
        throw std::logic_error("native token staging no longer requires full send retirement");
    for(const auto& link:ledger.links) {
        if(!link.session) throw std::logic_error("native token transport session disappeared");
        link.session->require_current_retirement_lease(link.lease);
    }
    for(const auto& operation:ledger.operations) {
        if(operation.link>=ledger.links.size()) throw std::logic_error("native token ledger link is invalid");
        const auto& link=ledger.links[operation.link];
        if(!link.session) throw std::logic_error("native token retirement session disappeared");
        if(operation.inbound) link.session->require_retired_receive(operation.handle,link.lease,operation.descriptor);
        else link.session->require_retired_send(operation.handle,link.lease,operation.descriptor.byte_length);
    }
    // poll()==complete was observed by the actual receive path BEFORE credit
    // and repost. Retain that proof owner, but do not poll it after releasing
    // the RX slice: some providers' poll contract ends at consumer retirement.
    for(const auto& consumer:ledger.consumers)
        if(!consumer) throw std::logic_error("native token lost its completed GPU receive proof owner");
}

}  // namespace

transport::TransportBufferView
DistributedBoundaryStagingProvider::stage_outbound_slice(
    const DistributedExecutionBoundary& boundary,
    const DistributedBoundaryTransfer& transfer,
    const transport::LinkSlice& slice,
    std::uint16_t stripe_index,
    std::uint16_t stripe_count) {
    require_default_single_slice(
        transfer,
        slice,
        stripe_index,
        stripe_count);
    return stage_outbound(
        boundary,
        transfer);
}

transport::TransportBufferView
DistributedBoundaryStagingProvider::stage_inbound_slice(
    const DistributedExecutionBoundary& boundary,
    const DistributedBoundaryTransfer& transfer,
    const transport::LinkSlice& slice,
    std::uint16_t stripe_index,
    std::uint16_t stripe_count) {
    require_default_single_slice(
        transfer,
        slice,
        stripe_index,
        stripe_count);
    return stage_inbound(
        boundary,
        transfer);
}

void DistributedBoundaryStagingProvider::consume_inbound_slice(
    const DistributedExecutionBoundary& boundary,
    const DistributedBoundaryTransfer& transfer,
    const transport::LinkSlice& slice,
    std::uint16_t stripe_index,
    std::uint16_t stripe_count,
    const transport::TransportBufferView& staged) {
    require_default_single_slice(
        transfer,
        slice,
        stripe_index,
        stripe_count);
    consume_inbound(
        boundary,
        staged);
}

NativeDistributedBackend::NativeDistributedBackend(
    std::span<transport::DirectTransportNativeLinkSession*> sessions,
    DistributedNativeControlPlane& control,
    DistributedNativeLocalExecutor& local,
    DistributedBoundaryStagingProvider& staging,
    DistributedNativeBackendPolicy policy)
    : sessions_(sessions),
      control_(&control),
      local_(&local),
      staging_(&staging),
      policy_(policy) {
    if (sessions_.empty() ||
        policy_.maximum_event_polls == 0 ||
        policy_.maximum_consumer_polls == 0) {
        throw std::invalid_argument(
            "native distributed backend requires sessions and non-zero poll budget");
    }
}

void NativeDistributedBackend::begin_token_retirement(const DistributedPlacementPlan& plan,
    const DistributedRuntimeRequest& request,std::uint32_t node,std::uint64_t binding,
    std::shared_ptr<const void> actual_owner,NativeDistributedPositionDependencyPolicy position_policy) {
    require_healthy();
    if(position_policy!=NativeDistributedPositionDependencyPolicy::reject &&
        position_policy!=NativeDistributedPositionDependencyPolicy::materialized_input_position_v1)
        throw std::invalid_argument("unknown logical position dependency policy");
    if(token_ || dispatched_ || !actual_owner || !binding || !plan.plan_identity || !plan.executable ||
        !plan.has_remote_work || node>1U || !request.connection_epoch || !request.request_id ||
        !request.generation || !request.dtype_id || sessions_.size()>64U || plan.boundaries.size()>1024U ||
        !staging_->requires_full_send_retirement())
        throw std::invalid_argument("native token mode requires bounded exact placement and full-retirement actual transport owners before dispatch");
    auto ledger=std::make_shared<NativeDistributedTokenLedger>();
    ledger->request=request;ledger->placement=plan.plan_identity;ledger->binding=binding;ledger->node=node;
    ledger->position_policy=position_policy;
    ledger->actual_owner=std::move(actual_owner);ledger->staging=staging_;
    ledger->links.reserve(sessions_.size());
    for(std::size_t index=0;index<sessions_.size();++index) {
        auto& session=session_for_link(index);
        ledger->links.push_back({&session,session.lease()});
    }
    ledger->boundaries.reserve(plan.boundaries.size());
    std::size_t position_dependencies=0,data_boundaries=0;
    for(const auto& boundary:plan.boundaries) {
        const bool logical_position=
            position_policy==NativeDistributedPositionDependencyPolicy::materialized_input_position_v1 &&
            is_materialized_input_position_dependency(plan,boundary);
        if(!boundary.boundary_id || (!logical_position && (!boundary.payload_bytes || boundary.payloads.empty())) ||
            boundary.payloads.size()>1024U || boundary.source_node==boundary.destination_node ||
            boundary.source_node>1U || boundary.destination_node>1U)
            throw std::invalid_argument("native token boundary metadata is invalid or unbounded");
        if(logical_position) {
            if(++position_dependencies>NativeDistributedPositionCompletion::maximum_dependencies)
                throw std::invalid_argument("too many logical position dependencies");
        } else ++data_boundaries;
        for(const auto& existing:ledger->boundaries)
            if(existing.value.boundary_id==boundary.boundary_id)
                throw std::invalid_argument("native token boundary identity is duplicated");
        if(boundary.source_node==node || boundary.destination_node==node)
            ledger->boundaries.push_back({boundary,boundary.destination_node==node,false,false,logical_position});
    }
    if(ledger->boundaries.empty() || !data_boundaries)
        throw std::invalid_argument("native token mode has no actual data boundary");
    ledger->operations.reserve(ledger->boundaries.size()*ledger->links.size());
    ledger->consumers.reserve(ledger->boundaries.size());
    token_=std::move(ledger);
}
void NativeDistributedBackend::begin_boundary(const DistributedExecutionBoundary& boundary,
    const transport::TransferDescriptor& descriptor,bool inbound) {
    dispatched_=true;
    if(!token_) return;
    auto& ledger=*token_;
    try {
        if(descriptor.model_plan_id!=ledger.placement || descriptor.request_id!=ledger.request.request_id ||
            descriptor.connection_epoch!=ledger.request.connection_epoch || descriptor.generation!=ledger.request.generation ||
            descriptor.dtype_id!=ledger.request.dtype_id ||
            descriptor.shape_fingerprint!=(boundary.boundary_id^boundary.payload_bytes))
            throw std::invalid_argument("native token transfer is foreign or stale");
        for(auto& expected:ledger.boundaries) if(expected.value.boundary_id==boundary.boundary_id) {
            if(expected.logical_position || expected.started || expected.inbound!=inbound || !retirement_boundary_equal(expected.value,boundary))
                throw std::invalid_argument("native token boundary is duplicated, wrong-direction or mutated");
            expected.started=true;return;
        }
        throw std::invalid_argument("native token transfer is outside its closed boundary inventory");
    } catch(...) {failed_=true;ledger.failed=true;throw;}
}
void NativeDistributedBackend::record_operation(transport::DirectTransportNativeLinkSession& session,
    transport::TransferHandle handle,const transport::TransferDescriptor& descriptor,bool inbound) {
    if(!token_) return;
    auto& ledger=*token_;
    if(!handle || ledger.operations.size()==ledger.operations.capacity())
        throw std::logic_error("native token operation ledger capacity exhausted");
    for(std::size_t i=0;i<ledger.links.size();++i) if(ledger.links[i].session==&session) {
        for(const auto& old:ledger.operations) if(old.link==i && old.handle==handle)
            throw std::logic_error("native token backend reused an operation handle");
        ledger.operations.push_back({i,handle,descriptor,inbound});return;
    }
    throw std::logic_error("native token operation used an unretained session");
}
void NativeDistributedBackend::finish_boundary(const DistributedExecutionBoundary& boundary,bool inbound) {
    if(!token_) return;
    for(auto& expected:token_->boundaries) if(expected.value.boundary_id==boundary.boundary_id && expected.inbound==inbound) {
        if(!expected.started || expected.complete) throw std::logic_error("native token boundary completion is duplicated");
        expected.complete=true;return;
    }
    throw std::logic_error("native token completion has no admitted boundary");
}
void NativeDistributedBackend::complete_position_dependencies(NativeDistributedPositionCompletion&& proof) {
    require_healthy();
    NativeTokenFailureLatch failure(token_,failed_);
    if(!token_ || token_->position_policy!=NativeDistributedPositionDependencyPolicy::materialized_input_position_v1 ||
        !proof.valid_ || !proof.owner_ || !proof.decision_ || !proof.count_ ||
        proof.count_>proof.boundaries_.size() || proof.binding_!=token_->binding ||
        proof.placement_!=token_->placement || proof.node_!=token_->node ||
        !retirement_request_equal(proof.request_,token_->request))
        throw std::invalid_argument("logical position proof is empty, foreign or unavailable");
    std::size_t expected_count=0;
    for(const auto& boundary:token_->boundaries) if(boundary.logical_position) {
        if(boundary.started || boundary.complete)
            throw std::logic_error("logical position completion is duplicated");
        ++expected_count;
        std::size_t matches=0;
        for(std::size_t i=0;i<proof.count_;++i) matches+=proof.boundaries_[i]==boundary.value.boundary_id;
        if(matches!=1U) throw std::invalid_argument("logical position proof omits or duplicates a dependency");
    }
    if(expected_count!=proof.count_)
        throw std::invalid_argument("logical position proof has extra dependencies");
    require_retirement_live(*token_);
    // No allocation or fallible operation follows this logical publication.
    // DMA handle completion remains independently checked by retire_token.
    for(auto& boundary:token_->boundaries) if(boundary.logical_position) {
        boundary.started=true;boundary.complete=true;
    }
    proof.valid_=false;
}
NativeDistributedTokenRetirement NativeDistributedBackend::retire(NativeDistributedRetirementOutcome outcome) {
    require_healthy();
    if(!token_) throw std::logic_error("native backend has no explicit token retirement mode");
    try {
        if(inbound_completion_) throw std::logic_error("native token retains a pending receive consumer");
        for(const auto& expected:token_->boundaries) {
            if((expected.started && !expected.complete) ||
                (outcome==NativeDistributedRetirementOutcome::completed_boundaries && !expected.complete))
                throw std::logic_error("native token retirement boundary coverage is incomplete");
        }
        require_retirement_live(*token_);
        token_->outcome=outcome;token_->sealed=true;
        return NativeDistributedTokenRetirement(token_);
    } catch(...) {failed_=true;token_->failed=true;throw;}
}
NativeDistributedTokenRetirement NativeDistributedBackend::retire_token() {
    return retire(NativeDistributedRetirementOutcome::completed_boundaries);
}
NativeDistributedTokenRetirement NativeDistributedBackend::retire_aborted_token() {
    return retire(NativeDistributedRetirementOutcome::aborted_attempt);
}
void NativeDistributedTokenRetirement::validate_matches(std::uint64_t binding,std::uint64_t placement,
    const DistributedRuntimeRequest& request,std::uint32_t node,NativeDistributedRetirementOutcome outcome) const {
    if(!ledger_ || !ledger_->sealed || ledger_->binding!=binding || ledger_->placement!=placement ||
        ledger_->node!=node || ledger_->outcome!=outcome || !retirement_request_equal(ledger_->request,request))
        throw std::invalid_argument("native token retirement proof is empty, foreign or has the wrong outcome");
    require_retirement_live(*ledger_);
}
void NativeDistributedTokenRetirement::consume_matches(std::uint64_t binding,std::uint64_t placement,
    const DistributedRuntimeRequest& request,std::uint32_t node,NativeDistributedRetirementOutcome outcome) {
    validate_matches(binding,placement,request,node,outcome);
    ledger_->consumed=true;
}

void NativeDistributedBackend::require_healthy() const {
    if (failed_) {
        throw std::logic_error("native distributed backend is poisoned");
    }
    if(token_ && token_->sealed) throw std::logic_error("native token backend was permanently sealed");
}

transport::DirectTransportNativeLinkSession&
NativeDistributedBackend::session_for_link(
    std::size_t link_index) const {
    if (link_index >=
            sessions_.size() ||
        sessions_[link_index] == nullptr ||
        !sessions_[link_index]->active() ||
        sessions_[link_index]->lease().link_index != link_index) {
        throw std::invalid_argument(
            "distributed transfer references unavailable native link session");
    }
    return *sessions_[link_index];
}

void NativeDistributedBackend::execute_local_step(
    const DistributedStepPlacement& step) {
    require_healthy();
    dispatched_=true;
    if(!token_) {local_->execute_local_step(step);return;}
    try {local_->execute_local_step(step);} catch(...) {failed_=true;token_->failed=true;throw;}
}

void NativeDistributedBackend::execute_remote_step(
    const DistributedStepPlacement& step) {
    require_healthy();
    dispatched_=true;
    if(!token_) {control_->execute_remote_step(step);return;}
    try {control_->execute_remote_step(step);} catch(...) {failed_=true;token_->failed=true;throw;}
}

void NativeDistributedBackend::wait_send_reusable(
    transport::DirectTransportNativeLinkSession& session,
    transport::TransferHandle handle) const {
    for (std::uint64_t poll = 0;
         poll < policy_.maximum_event_polls;
         ++poll) {
        const auto& state =
            session.send_state(handle);
        if (state.failed) {
            throw std::runtime_error(
                "native distributed send failed");
        }
        if (state.local_buffer_reusable &&
            (!staging_->requires_full_send_retirement() ||
             (state.remote_transport_acked && state.remote_consumer_complete))) {
            return;
        }
        static_cast<void>(
            session.poll_event());
    }
    const auto& final = session.send_state(handle);
    if (!final.failed && final.local_buffer_reusable &&
        (!staging_->requires_full_send_retirement() ||
         (final.remote_transport_acked && final.remote_consumer_complete))) {
        return;
    }
    throw std::runtime_error(
        "native distributed send completion poll budget exhausted");
}

void NativeDistributedBackend::wait_receive_complete(
    transport::DirectTransportNativeLinkSession& session,
    transport::TransferHandle handle) const {
    for (std::uint64_t poll = 0;
         poll < policy_.maximum_event_polls;
         ++poll) {
        const auto& state =
            session.receive_state(handle);
        if (state.failed) {
            throw std::runtime_error(
                "native distributed receive failed");
        }
        if (state.ownership ==
            transport::BufferOwnership::receive_complete) {
            return;
        }
        static_cast<void>(
            session.poll_event());
    }
    const auto& final = session.receive_state(handle);
    if (!final.failed && final.ownership == transport::BufferOwnership::receive_complete) {
        return;
    }
    throw std::runtime_error(
        "native distributed receive completion poll budget exhausted");
}

void NativeDistributedBackend::wait_receive_reusable(
    transport::DirectTransportNativeLinkSession& session,
    transport::TransferHandle handle) const {
    for (std::uint64_t poll = 0;
         poll < policy_.maximum_event_polls;
         ++poll) {
        const auto& state =
            session.receive_state(handle);
        if (state.failed) {
            throw std::runtime_error(
                "native distributed receive failed during repost");
        }
        if (state.ownership ==
            transport::BufferOwnership::reusable) {
            return;
        }
        static_cast<void>(
            session.poll_event());
    }
    const auto& final = session.receive_state(handle);
    if (!final.failed && final.ownership == transport::BufferOwnership::reusable) {
        return;
    }
    throw std::runtime_error(
        "native distributed receive repost poll budget exhausted");
}

void NativeDistributedBackend::send_boundary(
    const DistributedExecutionBoundary& boundary,
    const DistributedBoundaryTransfer& transfer) {
    require_healthy();
    dispatched_=true;
    const NativeTokenFailureLatch token_failure(token_,failed_);
    const auto stripe_count =
        validate_slices(
            transfer,
            sessions_.size());
    if (!transfer.native_slices.empty()) {
        throw std::invalid_argument("native send requires an unbound logical transfer");
    }
    if (transfer.descriptor.boundary_id != boundary.boundary_id ||
        transfer.descriptor.byte_length != boundary.payload_bytes ||
        transfer.flow.direction != transport::LinkDirection::outbound ||
        transfer.descriptor.sequence == 0 || transfer.descriptor.generation == 0 ||
        !transport::validate_descriptor(transfer.descriptor,
            {1, transfer.descriptor.connection_epoch, boundary.payload_bytes})) {
        throw std::invalid_argument("native send metadata does not match boundary");
    }

    begin_boundary(boundary,transfer.descriptor,false);

    auto physical_transfer = transfer;
    physical_transfer.native_slices.reserve(stripe_count);

    std::vector<NativePreparedTransfer>
        prepared;
    prepared.reserve(
        transfer.link_plan.slices.size());

    try {
        for (const auto& slice : transfer.link_plan.slices) {
            auto& session = session_for_link(slice.link_index);
            physical_transfer.native_slices.push_back({
                session.lease().connection_epoch, session.reserve_send_sequence(),
            });
        }
        for (std::size_t index = 0;
             index <
                transfer.link_plan.slices.size();
             ++index) {
            const auto& slice =
                transfer.link_plan.slices[index];
            auto& session =
                session_for_link(
                    slice.link_index);
            const auto stripe_index =
                static_cast<std::uint16_t>(
                    index);
            const auto& identity = physical_transfer.native_slices[index];
            const auto staged =
                staging_->stage_outbound_slice(
                    boundary,
                    physical_transfer,
                    slice,
                    stripe_index,
                    stripe_count);
            const auto descriptor =
                slice_descriptor(
                    transfer.descriptor,
                    slice,
                    identity);
            const auto handle =
                session.prepare_send(
                    descriptor,
                    staged);
            prepared.push_back({
                &session,
                handle,
                descriptor,
                staged,
                slice,
                stripe_index,
            });
            record_operation(session,handle,descriptor,false);
        }

        control_->publish_boundary_transfer(
            boundary,
            physical_transfer);

        for (auto& item :
             prepared) {
            const auto commit =
                make_commit(
                    item.descriptor,
                    item.slice,
                    item.stripe_index,
                    stripe_count);
            item.session->publish_commit(
                item.handle,
                item.descriptor,
                commit);
            item.session->submit(
                item.handle);
        }

        for (auto& item :
             prepared) {
            wait_send_reusable(
                *item.session,
                item.handle);
        }
        finish_boundary(boundary,false);
    } catch (...) {
        failed_ = true;
        for (auto& item :
             prepared) {
            try {
                if (!item.session->
                        send_state(
                            item.handle).
                        local_buffer_reusable) {
                    item.session->cancel(
                        item.handle);
                }
            } catch (...) {
            }
        }
        throw;
    }
}

DistributedBoundaryTransfer
NativeDistributedBackend::receive_boundary(
    const DistributedExecutionBoundary& boundary) {
    require_healthy();
    dispatched_=true;
    const NativeTokenFailureLatch token_failure(token_,failed_);
    auto transfer =
        control_->await_boundary_transfer(
            boundary);
    const auto stripe_count =
        validate_slices(
            transfer,
            sessions_.size());
    if (transfer.native_slices.size() != stripe_count ||
        transfer.descriptor.boundary_id != boundary.boundary_id ||
        transfer.descriptor.byte_length != boundary.payload_bytes ||
        transfer.flow.direction != transport::LinkDirection::inbound ||
        transfer.descriptor.sequence == 0 || transfer.descriptor.generation == 0 ||
        !transport::validate_descriptor(transfer.descriptor,
            {1, transfer.descriptor.connection_epoch, boundary.payload_bytes})) {
        throw std::invalid_argument("native receive lacks exact physical/logical boundary identities");
    }

    // Validate every current epoch before reserving sequences or staging any
    // payload. Logical request/plan reconciliation belongs to the control
    // plane; physical descriptors retain those identities unchanged.
    for (std::size_t index = 0; index < stripe_count; ++index) {
        auto& session = session_for_link(transfer.link_plan.slices[index].link_index);
        const auto& identity = transfer.native_slices[index];
        if (identity.connection_epoch != session.lease().connection_epoch ||
            identity.sequence == 0) {
            throw std::invalid_argument("native receive slice epoch does not match leased link");
        }
    }

    begin_boundary(boundary,transfer.descriptor,true);

    std::vector<NativePreparedTransfer>
        prepared;
    prepared.reserve(
        transfer.link_plan.slices.size());

    try {
        for (std::size_t index = 0;
             index < transfer.link_plan.slices.size();
             ++index) {
            const auto& slice = transfer.link_plan.slices[index];
            auto& session = session_for_link(slice.link_index);
            const auto stripe_index = static_cast<std::uint16_t>(index);
            const auto& identity = transfer.native_slices[index];
            session.reserve_receive_sequence(identity.sequence);
            const auto staged = staging_->stage_inbound_slice(
                boundary, transfer, slice, stripe_index, stripe_count);
            const auto descriptor = slice_descriptor(
                transfer.descriptor, slice, identity);
            const auto handle = session.prepare_receive(descriptor, staged);
            prepared.push_back({
                &session, handle, descriptor, staged, slice, stripe_index,
            });
            record_operation(session,handle,descriptor,true);
        }

        for (auto& item :
             prepared) {
            item.session->submit(
                item.handle);
        }

        std::vector<transport::TransferDescriptor> physical_descriptors;
        physical_descriptors.reserve(prepared.size());
        for (const auto& item : prepared) {
            physical_descriptors.push_back(item.descriptor);
        }
        transport::StripedMessageTracker
            tracker(
                transfer.descriptor,
                physical_descriptors);

        for (auto& item :
             prepared) {
            wait_receive_complete(
                *item.session,
                item.handle);
            const auto commit =
                item.session->read_commit(
                    item.handle,
                    item.descriptor);
            if (commit.stripe_index != item.stripe_index) {
                throw std::runtime_error("native receive commit belongs to a different physical stripe");
            }
            if (!tracker.observe_commit(
                    commit)) {
                throw std::runtime_error(
                    "native striped receive commit validation failed");
            }
        }

        if (!tracker.complete() ||
            tracker.invalid() ||
            tracker.committed_bytes() !=
                transfer.descriptor.byte_length) {
            throw std::runtime_error(
                "native striped receive did not publish complete logical payload");
        }

        for (auto& item :
             prepared) {
            item.session->acquire_receive(
                item.handle);
            staging_->consume_inbound_slice(
                boundary,
                transfer,
                item.slice,
                item.stripe_index,
                stripe_count,
                item.staged);
        }
        inbound_completion_ = staging_->finish_inbound(
            boundary,
            transfer);
        if (!inbound_completion_) {
            throw std::runtime_error("native receive consumer supplied no completion proof");
        }
        bool consumer_complete = false;
        for (std::uint64_t poll = 0; poll < policy_.maximum_consumer_polls; ++poll) {
            const auto status = inbound_completion_->poll();
            if (status == DistributedInboundCompletionStatus::complete) {
                consumer_complete = true;
                break;
            }
            if (status == DistributedInboundCompletionStatus::failed) {
                throw std::runtime_error("native receive consumer completion failed");
            }
        }
        if (!consumer_complete) {
            throw std::runtime_error("native receive consumer completion poll budget exhausted");
        }

        for (auto& item :
             prepared) {
            item.session->release_receive(
                item.handle);
            item.session->
                publish_receive_credit(
                    item.handle);
            item.session->repost_receive(
                item.handle);
        }
        for (auto& item :
             prepared) {
            wait_receive_reusable(
                *item.session,
                item.handle);
        }
        if(token_) token_->consumers.push_back(std::move(inbound_completion_));
        else inbound_completion_.reset();
        finish_boundary(boundary,true);
    } catch (...) {
        failed_ = true;
        for (auto& item :
             prepared) {
            try {
                const auto& state =
                    item.session->
                        receive_state(
                            item.handle);
                if (state.ownership ==
                        transport::BufferOwnership::transport_owned) {
                    try { item.session->cancel(item.handle); } catch (...) {}
                }
                if (state.ownership != transport::BufferOwnership::reusable) {
                    item.session->quarantine_receive(item.handle);
                }
            } catch (...) {
            }
        }
        throw;
    }

    return transfer;
}

}  // namespace strix
