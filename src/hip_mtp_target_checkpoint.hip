#include "strix/hip_mtp_target_checkpoint.hpp"
#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_session_snapshot_transfer.hpp"
#include "strix/hip_mtp_target_window.hpp"
#include "strix/indexed_attention_state.hpp"
#include "strix/hip_check.hpp"
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace strix {
namespace {
constexpr std::uint64_t alignment=256;
void check(hipError_t result) {
    ::strix::hip_check_message(result, "MTP target checkpoint HIP operation failed");
}
bool valid_recorded_lineage(const SerialSessionContext& session) noexcept {
    if(session.next_position==0 && session.committed_tokens==0 &&
       session.arithmetic_lineage.intervals.empty() && !session.arithmetic_lineage.history_identity &&
       !session.arithmetic_lineage.tainted) return true;
    return arithmetic_lineage_valid(session.arithmetic_lineage,session.next_position);
}
bool same(const SessionStateMemoryBinding& a,const SessionStateMemoryBinding& b) {
    return a.layout_identity==b.layout_identity && a.device_base==b.device_base &&
        a.device_bytes==b.device_bytes && a.device_allocation_generation==b.device_allocation_generation &&
        a.host_base==b.host_base && a.host_bytes==b.host_bytes &&
        a.host_allocation_generation==b.host_allocation_generation && a.stage_mapping==b.stage_mapping;
}
bool same(const ExecutionMemoryBinding& a,const ExecutionMemoryBinding& b) {
    return a.execution_identity==b.execution_identity && a.arena_base==b.arena_base &&
        a.arena_bytes==b.arena_bytes && a.allocation_generation==b.allocation_generation;
}
bool same(const ModelRuntimeBinding& a,const ModelRuntimeBinding& b) {
    if(a.manifest_layout_identity!=b.manifest_layout_identity ||
       a.model_content_identity!=b.model_content_identity || a.tensors.size()!=b.tensors.size()) return false;
    for(std::size_t i=0;i<a.tensors.size();++i) {
        const auto& x=a.tensors[i]; const auto& y=b.tensors[i];
        if(x.tensor_index!=y.tensor_index || x.storage!=y.storage || x.address!=y.address ||
           x.bytes!=y.bytes || x.allocation_generation!=y.allocation_generation || x.owner_node!=y.owner_node)
            return false;
    }
    return true;
}
} // namespace

struct HipMtpTargetCheckpoint::Impl {
    ModelContract model;
    CompiledSerialExecution compiled;
    SerialSessionStateLayout layout;
    std::shared_ptr<const RuntimeExecutionMemoryOwner> memory;
    std::shared_ptr<const HipModelResidency> weights;
    std::shared_ptr<HipRuntimeAllocator> allocator;
    hipStream_t stream=nullptr;
    std::shared_ptr<const void> stream_owner;
    std::shared_ptr<const HipSetupQuiescenceGuard> guard;
    SessionSnapshotPolicy policy=SessionSnapshotPolicy::full_components;
    HipMtpTargetCheckpointReservation reservation;
    SessionStateMemoryBinding original_state;
    ExecutionMemoryBinding original_scratch;
    ModelRuntimeBinding original_weights;
    std::uint64_t original_weights_generation=0,device_lease=0,host_lease=0;
    SessionSnapshotMemoryBinding pools;
    std::optional<SerialSessionContext> saved;
    SerialSessionSnapshotPlan snapshot;
    const SerialSessionContext* session_address=nullptr;
    std::uint64_t compatibility=0;
    bool broken=false;
    std::shared_ptr<const HipMtpTargetWindowCapture> producer;
    std::shared_ptr<HipQsaCommittedPool> qsa_pool;
    std::optional<HipMtpTargetRollbackPlan> rollback;
    ExecutionMemoryBinding deferred;
    void require_producer() const {
        if(!producer || !producer->window_execution_identity() || !producer->window_kernel_plan_identity() ||
           producer->source_serial_execution_identity()!=compiled.execution_identity ||
           !same(producer->source_state_memory_binding(),original_state) ||
           !same(producer->source_scratch_binding(),original_scratch) ||
           !same(producer->source_weights_binding(),original_weights) ||
           !same(producer->deferred_status_binding(),deferred) || producer->source_stream()!=stream)
            throw std::invalid_argument("MTP bounded rollback actual producer/binding changed");
    }
    std::vector<SessionSnapshotTransfer> rollback_transfers(const HipMtpTargetRollbackPlan& plan,bool restoring) const {
        std::vector<SessionSnapshotTransfer> result;result.reserve(plan.ranges.size());
        for(const auto& range:plan.ranges) {
            const auto source_base=range.deferred_status?deferred.arena_base:
                range.storage==SessionStateStorage::device?original_state.device_base:original_state.host_base;
            const auto source_bytes=range.deferred_status?deferred.arena_bytes:
                range.storage==SessionStateStorage::device?original_state.device_bytes:original_state.host_bytes;
            const auto destination_base=range.storage==SessionStateStorage::device?pools.device_section_base:pools.host_section_base;
            const auto destination_bytes=range.storage==SessionStateStorage::device?reservation.device_payload_bytes:reservation.host_payload_bytes;
            if(range.source_offset>source_bytes || range.bytes>source_bytes-range.source_offset ||
               range.snapshot_offset>destination_bytes || range.bytes>destination_bytes-range.snapshot_offset ||
               !source_base || !destination_base || source_bytes>UINTPTR_MAX-source_base || destination_bytes>UINTPTR_MAX-destination_base)
                throw std::invalid_argument("MTP bounded rollback transfer escapes retained allocation");
            const auto source=source_base+range.source_offset,destination=destination_base+range.snapshot_offset;
            result.push_back({range.storage,restoring?destination:source,restoring?source:destination,range.bytes});
        }
        return result;
    }

    ~Impl() {
        // Retain every owner through this stream only on normal teardown.
        // Unknown completion/device failure still requires fail-stop recovery.
        if(guard && (pools.device_section_base || pools.host_section_base)) {
            try { complete_owned_stream(); }
            catch(...) { guard->recover_or_fail_stop(); }
        }
        if(allocator) {
            if(pools.device_section_base) allocator->free_device(pools.device_section_base,reservation.device_payload_bytes);
            if(pools.host_section_base) allocator->free_host(pools.host_section_base,reservation.host_payload_bytes);
        }
    }
    void complete_owned_stream() const {
        guard->require_original();
        check(hipStreamSynchronize(stream));
        guard->require_original();
    }
    void require_bindings() const {
        if(broken) throw std::logic_error("MTP target checkpoint is permanently failed");
        guard->require_original();
        if(current_hip_launch_context()) throw std::invalid_argument("MTP checkpoint requires no active native launch scope");
        hipStreamCaptureStatus capture{};
        check(hipStreamIsCapturing(stream,&capture));
        if(capture!=hipStreamCaptureStatusNone) throw std::invalid_argument("MTP checkpoint rejects stream capture");
        memory->validate_inventory_ownership();
        if(!same(memory->state(),original_state) || !same(memory->scratch(),original_scratch) ||
           weights->generation()!=original_weights_generation || !same(weights->binding(),original_weights))
            throw std::invalid_argument("MTP target checkpoint physical target binding changed");
        // Strong-create ownership is checked by the residency itself, not by a
        // caller supplied non-null token or an inferred allocation generation.
        (void)weights->owner_inventory(weights);
        if(pools.device_section_base && allocator->allocation_lease_identity(
                pools.device_section_base,reservation.device_payload_bytes,false)!=device_lease)
            throw std::invalid_argument("MTP checkpoint device lease changed");
        if(pools.host_section_base && allocator->allocation_lease_identity(
                pools.host_section_base,reservation.host_payload_bytes,true)!=host_lease)
            throw std::invalid_argument("MTP checkpoint host lease changed");
    }
    void admit_session(const SerialSessionContext& session,bool require_ready) const {
        if(session.execution_identity!=compiled.execution_identity || session.state_layout_identity!=layout.layout_identity ||
           !session.runtime_compatibility_identity || (compatibility && session.runtime_compatibility_identity!=compatibility) ||
           (session_address && session_address!=&session) || session.next_position>layout.context_tokens ||
           session.next_position>compiled.max_position_embeddings ||
           (require_ready && (session.health!=SerialSessionHealth::ready ||
                !valid_recorded_lineage(session))))
            throw std::invalid_argument("MTP checkpoint target session or recorded lineage is incompatible");
    }
};

HipMtpTargetCheckpoint::HipMtpTargetCheckpoint(const ModelContract& model,
    const CompiledSerialExecution& compiled,const SerialSessionStateLayout& layout,
    std::shared_ptr<const RuntimeExecutionMemoryOwner> memory,
    std::shared_ptr<const HipModelResidency> weights,std::shared_ptr<HipRuntimeAllocator> allocator,
    hipStream_t stream,std::shared_ptr<const void> stream_owner,
    std::shared_ptr<const HipSetupQuiescenceGuard> guard,SessionSnapshotPolicy policy,
    std::uint64_t budget,AllocationGenerationSource& generations,
    std::shared_ptr<const HipMtpTargetWindowCapture> producer,std::shared_ptr<HipQsaCommittedPool> qsa_pool):impl_(std::make_unique<Impl>()) {
    if(!memory || !weights || !allocator || !stream_owner || !guard || !stream ||
       stream==hipStreamLegacy || stream==hipStreamPerThread || !budget || !compiled.execution_identity ||
       compiled.logical_rows!=1 || compiled.request_concurrency!=1 || memory->state().stage_mapping ||
       memory->scratch().execution_identity!=compiled.execution_identity ||
       memory->retained_allocator().get()!=allocator.get() ||
       memory->retained_allocator().owner_before(allocator) || allocator.owner_before(memory->retained_allocator()))
        throw std::invalid_argument("MTP checkpoint requires actual retained C1 target owners and an exclusive eager stream");
    auto& p=*impl_;
    p.model=model; p.compiled=compiled; p.layout=layout;
    p.memory=std::move(memory); p.weights=std::move(weights); p.allocator=std::move(allocator);
    p.stream=stream; p.stream_owner=std::move(stream_owner); p.guard=std::move(guard); p.policy=policy;
    validate_session_snapshot_policy(policy);
    p.producer=std::move(producer);p.qsa_pool=std::move(qsa_pool);
    if(p.producer) p.deferred=p.producer->deferred_status_binding();
    const bool status=p.deferred.arena_base || p.deferred.arena_bytes || p.deferred.execution_identity || p.deferred.allocation_generation;
    if(status && (!p.producer || p.deferred.execution_identity!=p.producer->window_execution_identity() ||
       !p.deferred.arena_base || p.deferred.arena_base%4 || p.deferred.arena_bytes!=8 || !p.deferred.allocation_generation))
        throw std::invalid_argument("MTP bounded rollback D1 record is not the exact producer record");
    p.reservation=p.producer?plan_hip_mtp_target_window_rollback(model,layout,0,p.producer->logical_rows(),status).reservation:
        plan_hip_mtp_target_checkpoint(model,layout,policy);
    if(p.reservation.retained_bytes>budget) throw std::length_error("MTP checkpoint retained byte budget exceeded");
    if(p.reservation.device_payload_bytes>std::numeric_limits<std::size_t>::max() ||
       p.reservation.host_payload_bytes>std::numeric_limits<std::size_t>::max())
        throw std::length_error("MTP checkpoint pools exceed addressable allocation size");
    validate_session_state_memory_binding(layout,p.memory->state());
    p.original_state=p.memory->state(); p.original_scratch=p.memory->scratch();
    p.original_weights=p.weights->binding(); p.original_weights_generation=p.weights->generation();
    if(!p.original_weights.model_content_identity.valid() || !p.original_weights_generation)
        throw std::invalid_argument("MTP checkpoint requires actual model content and residency generation");
    p.require_bindings();
    if(p.producer) p.require_producer();
    p.pools.allocation_generation=generations.next();
    p.guard->run([&] {
        if(p.reservation.device_payload_bytes) {
            p.pools.device_section_base=p.allocator->allocate_device(p.reservation.device_payload_bytes,alignment);
            p.device_lease=p.allocator->allocation_lease_identity(p.pools.device_section_base,p.reservation.device_payload_bytes,false);
        }
        if(p.reservation.host_payload_bytes) {
            p.pools.host_section_base=p.allocator->allocate_host(p.reservation.host_payload_bytes,alignment);
            p.host_lease=p.allocator->allocation_lease_identity(p.pools.host_section_base,p.reservation.host_payload_bytes,true);
        }
    });
    if((p.pools.device_section_base && !p.device_lease) || (p.pools.host_section_base && !p.host_lease))
        throw std::invalid_argument("MTP checkpoint snapshot pool leases are opaque");
}
HipMtpTargetCheckpoint::~HipMtpTargetCheckpoint()=default;
void HipMtpTargetCheckpoint::capture(const SerialSessionContext& session) {
    auto& p=*impl_; p.require_bindings(); p.admit_session(session,true);
    if(p.producer) throw std::logic_error("bounded target checkpoint requires capture_window");
    if(p.saved) throw std::logic_error("MTP checkpoint must be restored or explicitly discarded before recapture");
    // All metadata allocations and canonical range validation precede copying.
    auto recorded=session;
    SerialSessionSnapshotRequest request;
    request.compiled_execution_identity=session.execution_identity;
    request.runtime_compatibility_identity=session.runtime_compatibility_identity;
    request.next_position=session.next_position; request.committed_tokens=session.committed_tokens;
    request.state=session.state; request.alignment=alignment; request.policy=p.policy;
    auto snapshot=build_serial_session_snapshot_plan(p.model,p.layout,request);
    validate_serial_session_snapshot_plan(p.model,p.layout,snapshot);
    if(snapshot.device_bytes>p.reservation.device_payload_bytes || snapshot.host_bytes>p.reservation.host_payload_bytes)
        throw std::length_error("MTP checkpoint actual frontier exceeds reserved geometry");
    auto view=p.pools;
    view.snapshot_plan_identity=snapshot.plan_identity;
    view.device_section_bytes=snapshot.device_bytes; view.host_section_bytes=snapshot.host_bytes;
    HipSessionSnapshotTransferBackend transfer(p.stream,p.guard);
    try {
        p.complete_owned_stream(); // Complete only the actual target producer before host/device capture.
        capture_serial_session_snapshot(snapshot,p.layout,p.memory->state(),view,transfer,&p.model);
        p.guard->require_original();
    } catch(...) { p.broken=true; p.guard->recover_or_fail_stop(); throw; }
    p.snapshot=std::move(snapshot); p.pools.snapshot_plan_identity=view.snapshot_plan_identity;
    p.session_address=&session; p.compatibility=session.runtime_compatibility_identity;
    p.saved.emplace(std::move(recorded));
}
void HipMtpTargetCheckpoint::restore(SerialSessionContext& session) {
    auto& p=*impl_; p.require_bindings(); p.admit_session(session,false);
    if(p.producer) throw std::logic_error("bounded target checkpoint requires actual completed restore_window proof");
    if(!p.saved) throw std::logic_error("MTP checkpoint has no captured frontier");
    if(!valid_recorded_lineage(*p.saved) ||
       p.saved->health!=SerialSessionHealth::ready)
        throw std::logic_error("MTP checkpoint recorded lineage is invalid");
    auto recorded=*p.saved; // Allocate lineage before touching target bytes.
    const auto restore=build_serial_session_restore_plan(p.model,p.layout,p.compatibility,p.snapshot);
    auto view=p.pools; view.device_section_bytes=p.snapshot.device_bytes; view.host_section_bytes=p.snapshot.host_bytes;
    HipSessionSnapshotTransferBackend transfer(p.stream,p.guard);
    try {
        p.complete_owned_stream();
        if(p.qsa_pool) p.qsa_pool->invalidate(false,0);
        restore_serial_session_snapshot_payload(restore,p.layout,p.memory->state(),view,transfer,&p.model);
        p.guard->require_original();
    } catch(...) {
        p.broken=true; session.health=SerialSessionHealth::poisoned;
        session.arithmetic_lineage={}; session.arithmetic_lineage.tainted=true;
        p.guard->recover_or_fail_stop(); throw;
    }
    session=std::move(recorded); // Only completed payload copies authorize this exact recorded metadata.
    p.saved.reset(); p.snapshot={};
}
void HipMtpTargetCheckpoint::capture_window(const SerialSessionContext& session) {
    auto& p=*impl_;p.require_bindings();p.require_producer();p.admit_session(session,true);
    if(p.saved) throw std::logic_error("MTP bounded checkpoint already has a captured frontier");
    auto recorded=session;
    auto plan=plan_hip_mtp_target_window_rollback(p.model,p.layout,session.next_position,p.producer->logical_rows(),p.deferred.arena_bytes!=0);
    const auto transfers=p.rollback_transfers(plan,false);
    HipSessionSnapshotTransferBackend transfer(p.stream,p.guard);
    try {p.complete_owned_stream();transfer.capture_batch(transfers);p.guard->require_original();}
    catch(...) {p.broken=true;p.guard->recover_or_fail_stop();throw;}
    p.session_address=&session;p.compatibility=session.runtime_compatibility_identity;
    p.rollback.emplace(std::move(plan));p.saved.emplace(std::move(recorded));
}
namespace {
bool same_record(const SerialSessionContext& a,const SerialSessionContext& b) {
    return a.execution_identity==b.execution_identity && a.runtime_compatibility_identity==b.runtime_compatibility_identity &&
        a.state_layout_identity==b.state_layout_identity && a.next_position==b.next_position && a.committed_tokens==b.committed_tokens &&
        a.health==b.health && a.state.valid==b.state.valid && a.state.generations==b.state.generations &&
        a.arithmetic_lineage.history_identity==b.arithmetic_lineage.history_identity && a.arithmetic_lineage.tainted==b.arithmetic_lineage.tainted &&
        a.arithmetic_lineage.intervals==b.arithmetic_lineage.intervals;
}
}
void HipMtpTargetCheckpoint::restore_window(SerialSessionContext& session,const HipMtpTargetWindowBatch& batch) {
    auto& p=*impl_;p.require_bindings();p.require_producer();p.admit_session(session,false);
    if(!p.saved || !p.rollback || !batch.valid() || !valid_recorded_lineage(*p.saved) ||
       batch.window_execution_identity()!=p.producer->window_execution_identity() ||
       batch.window_kernel_plan_identity()!=p.producer->window_kernel_plan_identity() ||
       batch.logical_rows()!=p.rollback->logical_rows || batch.chunk_begin()!=p.saved->next_position ||
       batch.chunk_end()-batch.chunk_begin()!=p.rollback->logical_rows ||
       !same_record(batch.base_session(),*p.saved) || !same_record(batch.committed_session(),session) ||
       !same(batch.source_state_memory_binding(),p.original_state) || !same(batch.source_scratch_binding(),p.original_scratch) ||
       !same(batch.source_weights_binding(),p.original_weights) ||
       !same(batch.deferred_status_binding(),p.deferred) || batch.source_stream()!=p.stream)
        throw std::invalid_argument("MTP bounded rollback lacks the exact actual completed producer footprint");
    auto recorded=*p.saved;
    const auto transfers=p.rollback_transfers(*p.rollback,true);
    HipSessionSnapshotTransferBackend transfer(p.stream,p.guard);
    try {p.complete_owned_stream();if(p.qsa_pool) p.qsa_pool->invalidate(false,p.saved->next_position>3?p.saved->next_position-3:0);
        transfer.restore_batch(transfers);p.guard->require_original();}
    catch(...) {
        p.broken=true;session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage={};session.arithmetic_lineage.tainted=true;
        p.guard->recover_or_fail_stop();throw;
    }
    session=std::move(recorded);p.saved.reset();p.rollback.reset();
}

void HipMtpTargetCheckpoint::discard() {
    auto& p=*impl_; p.require_bindings();
    try { p.complete_owned_stream(); }
    catch(...) { p.broken=true; p.guard->recover_or_fail_stop(); throw; }
    p.saved.reset(); p.snapshot={};p.rollback.reset();
}
bool HipMtpTargetCheckpoint::has_checkpoint() const noexcept { return impl_->saved.has_value(); }
bool HipMtpTargetCheckpoint::failed() const noexcept { return impl_->broken; }
std::uint64_t HipMtpTargetCheckpoint::reserved_bytes() const noexcept { return impl_->reservation.retained_bytes; }
} // namespace strix



