#include "strix/hip_check.hpp"
#include "strix/hip_mtp_target_window.hpp"
#include "strix/mtp_payload_geometry.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/hip_token_sampling.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/hip_runtime_allocator.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/prompt_terminal_diagnostics.hpp"
#include "strix/hip_token_sampling.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/mtp_step_logits_digest.hpp"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <optional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>
namespace strix {
namespace {
void require(bool v,const char* m){if(!v)throw std::invalid_argument(m);}
using strix::hip_check;
std::uint64_t mul(std::uint64_t a,std::uint64_t b){if(b && a>UINT64_MAX/b)throw std::overflow_error("MTP target window payload overflow");return a*b;}
bool same(ExecutionMemoryBinding a,ExecutionMemoryBinding b){return a.execution_identity==b.execution_identity && a.arena_base==b.arena_base && a.arena_bytes==b.arena_bytes && a.allocation_generation==b.allocation_generation;}
void copy_session(SerialSessionContext& to,const SerialSessionContext& from){
    require(to.arithmetic_lineage.intervals.capacity()>=from.arithmetic_lineage.intervals.size(),"MTP capture history exceeds preallocated metadata");
    to.execution_identity=from.execution_identity;to.state_layout_identity=from.state_layout_identity;to.runtime_compatibility_identity=from.runtime_compatibility_identity;
    to.next_position=from.next_position;to.committed_tokens=from.committed_tokens;to.health=from.health;to.state=from.state;
    to.arithmetic_lineage.intervals.resize(from.arithmetic_lineage.intervals.size());
    std::copy(from.arithmetic_lineage.intervals.begin(),from.arithmetic_lineage.intervals.end(),to.arithmetic_lineage.intervals.begin());
    to.arithmetic_lineage.history_identity=from.arithmetic_lineage.history_identity;to.arithmetic_lineage.tainted=from.arithmetic_lineage.tainted;
}
}
struct HipMtpTargetWindowCapture::Impl {
    std::shared_ptr<const void> actual_owner;
    std::shared_ptr<HipRuntimeAllocator> allocator;
    std::shared_ptr<HipSetupQuiescenceGuard> guard;
    const void* source_owner=nullptr;
    ModelContentIdentity content;
    ExecutionRuntimeResources resources;
    SessionStateMemoryBinding state_snapshot;
    ModelRuntimeBinding weights_snapshot;
    ExecutionMemoryBinding source_scratch,window_scratch;
    HipLaunchContext scope;
    RuntimeExecutionIdentities source_identities;
    HipMtpTargetWindowCapturePolicy policy;
    std::vector<ExecutionBufferAddress> source_views,logit_views;
    std::vector<std::unique_ptr<HipGreedyOutput>> selectors;
    // Host-mapped per-row argmax written by the window stream after its head rows
    // (same finite first-max rule as the selectors); null keeps the selectors.
    const HipGreedyRowsReply* device_greedy=nullptr;
    HipMtpTargetWindowCapture::RowSelection row_selection;
    HipMtpTargetWindowCapture::TransactionSelection transaction_selection;
    std::vector<MtpDeviceView> snapshots;
    std::shared_ptr<HipMtpTargetWindowBatch::Lease> active;
    std::weak_ptr<const HipMtpTargetWindowBatch::Lease> previous;
    std::uintptr_t pool=0;
    std::uint64_t pool_bytes=0,pool_lease=0,payload=0,window_id=0,kernel_id=0,serial_id=0,next_epoch=0,row_bytes=0,capacity=0;bool sources_contiguous=false;
    std::uint32_t rows=0,step=0,layer=0,vocab=0;
    ExecutionBufferUse use;
    bool failed=false;
    ~Impl(){if(guard){guard->recover_or_fail_stop();selectors.clear();if(pool)allocator->free_device(pool,pool_bytes);}}
    void require_pool()const{
        guard->require_original();require(pool && allocator->allocation_lease_identity(pool,pool_bytes,false)==pool_lease,
            "MTP target capture actual allocation lease changed");
    }
};
HipMtpTargetWindowCapture::HipMtpTargetWindowCapture():impl_(std::make_unique<Impl>()){}
HipMtpTargetWindowCapture::~HipMtpTargetWindowCapture()=default;
std::uint64_t HipMtpTargetWindowCapture::retained_payload_bytes()const noexcept{return impl_->payload;}
std::uint32_t HipMtpTargetWindowCapture::tap_step()const noexcept{return impl_->step;}
bool HipMtpTargetWindowCapture::ready_for_reuse()const noexcept{return !impl_->failed && !impl_->active && impl_->previous.expired() && impl_->next_epoch<UINT64_MAX;}
const SessionStateMemoryBinding& HipMtpTargetWindowCapture::source_state_memory_binding()const noexcept{return impl_->state_snapshot;}
const ExecutionMemoryBinding& HipMtpTargetWindowCapture::source_scratch_binding()const noexcept{return impl_->source_scratch;}
const ModelRuntimeBinding& HipMtpTargetWindowCapture::source_weights_binding()const noexcept{return impl_->weights_snapshot;}
std::uint64_t HipMtpTargetWindowCapture::source_serial_execution_identity()const noexcept{return impl_->serial_id;}
std::uint32_t HipMtpTargetWindowCapture::logical_rows()const noexcept{return impl_->rows;}
std::uint64_t HipMtpTargetWindowCapture::window_execution_identity()const noexcept{return impl_->window_id;}
std::uint64_t HipMtpTargetWindowCapture::window_kernel_plan_identity()const noexcept{return impl_->kernel_id;}
const ExecutionMemoryBinding& HipMtpTargetWindowCapture::deferred_status_binding()const noexcept{return impl_->scope.deferred_moe_status;}
void* HipMtpTargetWindowCapture::source_stream()const noexcept{return impl_->scope.stream;}
std::shared_ptr<HipMtpTargetWindowCapture> HipMtpTargetWindowCapture::create(const void* source_owner,const ModelContract& model,
    const CompiledSerialExecution& serial,const CompiledKnownPrefixWindow& window,const ExecutionRuntimeResources& resources,
    ExecutionMemoryBinding scratch,HipLaunchContext scope,std::shared_ptr<const void> actual_owner,HipMtpTargetWindowCapturePolicy policy){
    require(!current_hip_launch_context() && source_owner && actual_owner && policy.retained_byte_budget && policy.maximum_history_intervals &&
        policy.maximum_history_intervals<=model.effective_max_positions() && window.output_policy==KnownWindowOutputPolicy::all_rows &&
        resources.scratch && resources.model_weights && resources.session_state_memory && resources.session_state_layout &&
        !resources.session_state_memory->stage_mapping && resources.model_weights->model_content_identity.valid() &&
        model.vocab_size && model.vocab_size<=UINT32_MAX,"MTP window capture needs bounded actual all-head closed-window resources");
    validate_compiled_known_prefix_window(window,serial);validate_execution_memory_binding(window.window_execution_identity,window.arena.physical,scratch);
    require(scope.stream && !scope.graph_capture,"MTP window capture needs an owned eager stream");
    hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;hip_check(hipStreamIsCapturing(scope.stream,&capture));
    require(capture==hipStreamCaptureStatusNone,"MTP capture preparation cannot enter active capture");
    const auto head=std::find_if(serial.execution.steps.begin(),serial.execution.steps.end(),[](const auto& s){return s.op==SerialOp::final_hyper_mix;});
    require(head!=serial.execution.steps.end() && head!=serial.execution.steps.begin(),"MTP window capture has no final wide producer");
    const auto& tap=*std::prev(head);const auto& tapio=serial.io.steps[tap.id];const auto& headio=serial.io.steps[head->id];
    require(tap.op==SerialOp::mlp_inject && tap.layer==model.hidden_layers-1U && tapio.step_id==tap.id && tapio.writes.size()==1U &&
        tapio.writes[0].buffer.buffer_class==ExecutionBufferClass::expanded_hidden &&
        std::find(headio.reads.begin(),headio.reads.end(),tapio.writes[0])!=headio.reads.end(),"MTP window tap is not the actual pre-final-mix wide residual");
    const auto bytes=mul(mul(model.hidden_size,model.hc_count),2U);const auto rows=window.logical_rows;
    const auto pool=mul(bytes,rows);
    static_assert(sizeof(MtpDeviceView)==24U);
    const auto payload=mtp_payload::window_capture(model,rows,policy.maximum_history_intervals,resources.model_weights->tensors.size());
    require(payload<=policy.retained_byte_budget,"MTP window slots/selectors/history exceed upfront retained payload budget");
    auto out=std::shared_ptr<HipMtpTargetWindowCapture>(new HipMtpTargetWindowCapture);auto& i=*out->impl_;
    i.actual_owner=std::move(actual_owner);i.allocator=std::make_shared<HipRuntimeAllocator>();i.guard=std::make_shared<HipSetupQuiescenceGuard>();
    i.source_owner=source_owner;i.resources=resources;i.state_snapshot=*resources.session_state_memory;i.weights_snapshot=*resources.model_weights;i.source_scratch=*resources.scratch;i.window_scratch=scratch;i.scope=scope;i.policy=policy;
    i.source_identities=build_runtime_execution_identities(serial.execution_identity,serial.arena,resources);
    i.content=resources.model_weights->model_content_identity;i.payload=payload;i.row_bytes=bytes;i.pool_bytes=pool;i.rows=rows;
    i.window_id=window.window_execution_identity;i.kernel_id=window.window_kernel_plan_identity;i.serial_id=serial.execution_identity;
    i.step=tap.id;i.layer=*tap.layer;i.use=tapio.writes[0];i.vocab=static_cast<std::uint32_t>(model.vocab_size);
    i.capacity=std::min(resources.session_state_layout->context_tokens,model.effective_max_positions());
    i.source_views.reserve(rows);i.logit_views.reserve(rows);i.snapshots.resize(rows);i.selectors.reserve(rows);
    for(std::uint32_t row=0;row<rows;++row){
        auto view=resolve_execution_buffer(i.window_id,window.arena.row_view[row],scratch,i.use);require(view.bytes==bytes,"MTP capture wide row is not exact BF16");i.source_views.push_back(view);
        const auto& io=window.row_plan[row].io.steps.back();require(io.op==SerialOp::lm_head && io.writes.size()==1U,"MTP window capture requires every full target head");
        auto logits=resolve_execution_buffer(i.window_id,window.arena.row_view[row],scratch,io.writes[0]);require(logits.bytes==mul(model.vocab_size,4U),"MTP captured head is not full FP32 vocabulary");i.logit_views.push_back(logits);
    }
    // R2-11: rows that are adjacent in the arena are captured with one copy after the last row.
    i.sources_contiguous=rows>1U;for(std::uint32_t row=0;row<rows;++row)i.sources_contiguous=i.sources_contiguous&&i.source_views[row].address==i.source_views[0].address+row*bytes;
    i.pool=i.allocator->allocate_device(pool,256U);i.pool_lease=i.allocator->allocation_lease_identity(i.pool,pool,false);
    require(i.pool_lease,"MTP snapshot allocation lacks actual lease");
    require(i.pool_bytes<=std::numeric_limits<std::uintptr_t>::max()-i.pool,"MTP capture allocation range overflows");
    const auto disjoint=[&](std::uintptr_t address,std::uint64_t n){if(!n)return;require(address && n<=std::numeric_limits<std::uintptr_t>::max()-address &&
        !(i.pool<address+n && address<i.pool+i.pool_bytes),"MTP capture slots alias a retained producer owner");};
    disjoint(scratch.arena_base,scratch.arena_bytes);disjoint(i.source_scratch.arena_base,i.source_scratch.arena_bytes);
    disjoint(resources.session_state_memory->device_base,resources.session_state_memory->device_bytes);
    disjoint(resources.session_state_memory->host_base,resources.session_state_memory->host_bytes);
    disjoint(scope.deferred_moe_status.arena_base,scope.deferred_moe_status.arena_bytes);
    for(const auto& weight:resources.model_weights->tensors)if(runtime_tensor_gpu_visible(weight.storage))disjoint(weight.address,weight.bytes);
    for(std::uint32_t row=0;row<rows;++row){i.snapshots[row]={i.pool+row*bytes,bytes,i.pool_lease};
        i.selectors.push_back(std::make_unique<HipGreedyOutput>(i.vocab,scratch,i.actual_owner,i.guard));}
    return out;
}
void HipMtpTargetWindowCapture::begin(const void* source,const SerialSessionContext& session,std::span<const std::uint32_t> tokens){
    auto& i=*impl_;i.require_pool();require(!i.failed && !i.active && i.previous.expired(),"MTP capture cannot reuse retained or failed batch slots");
    require(source==i.source_owner && session.execution_identity==i.serial_id && session.runtime_compatibility_identity==i.source_identities.compatibility_identity &&
        session.state_layout_identity==i.resources.session_state_layout->layout_identity && session.health==SerialSessionHealth::ready &&
        arithmetic_lineage_valid(session.arithmetic_lineage,session.next_position) && tokens.size()==i.rows &&
        session.next_position<=i.capacity && i.rows<=i.capacity-session.next_position && session.committed_tokens<=UINT64_MAX-i.rows &&
        session.arithmetic_lineage.intervals.size()<i.policy.maximum_history_intervals && i.next_epoch<UINT64_MAX,
        "MTP target capture base session, history, input count or capacity differs");
    for(auto token:tokens)require(token<i.vocab,"MTP window input token is outside vocabulary");
    auto lease=std::make_shared<HipMtpTargetWindowBatch::Lease>();lease->tokens.assign(tokens.begin(),tokens.end());lease->predictions.resize(i.rows);lease->recorded.resize(i.rows);
    lease->base.arithmetic_lineage.intervals.reserve(static_cast<std::size_t>(i.policy.maximum_history_intervals));
    lease->committed.arithmetic_lineage.intervals.reserve(static_cast<std::size_t>(i.policy.maximum_history_intervals));
    copy_session(lease->base,session);lease->epoch=++i.next_epoch;i.active=std::move(lease);
}
void HipMtpTargetWindowCapture::preflight_row(const ExecutionDispatchContext& c)const{
    const auto& i=*impl_;require(i.active && !i.failed && c.position>=i.active->base.next_position,"MTP captured row has no active base");
    const auto row=c.position-i.active->base.next_position;
    require(row<i.rows && c.execution_identity==i.window_id && c.step_id==i.step && c.op==SerialOp::mlp_inject && c.layer==i.layer &&
        c.logical_rows==1U && c.request_concurrency==1U && c.input_token_id==i.active->tokens[static_cast<std::size_t>(row)] &&
        c.resources && c.resources->scratch && same(*c.resources->scratch,i.window_scratch) && c.writes.size()==1U && c.writes[0]==i.use,
        "MTP capture observed a foreign row/step/input/scratch descriptor");
    const auto actual=resolve_dispatch_write(c,0);const auto& expected=i.source_views[static_cast<std::size_t>(row)];
    require(actual.address==expected.address && actual.bytes==expected.bytes && actual.allocation_generation==expected.allocation_generation,
        "MTP captured wide source view changed");
}
void HipMtpTargetWindowCapture::record_row(const ExecutionDispatchContext& c){
    preflight_row(c);auto& i=*impl_;const auto row=static_cast<std::size_t>(c.position-i.active->base.next_position);
    require(!i.active->recorded[row],"MTP wide row was captured twice");i.require_pool();
    i.active->recorded[row]=1U;
    if(i.sources_contiguous){
        if(std::all_of(i.active->recorded.begin(),i.active->recorded.end(),[](auto x){return x==1U;}))
            hip_check(hipMemcpyAsync(reinterpret_cast<void*>(i.snapshots[0].address),reinterpret_cast<const void*>(i.source_views[0].address),
                static_cast<std::size_t>(i.row_bytes)*i.rows,hipMemcpyDeviceToDevice,i.scope.stream));
        return;
    }
    hip_check(hipMemcpyAsync(reinterpret_cast<void*>(i.snapshots[row].address),reinterpret_cast<const void*>(i.source_views[row].address),
        static_cast<std::size_t>(i.row_bytes),hipMemcpyDeviceToDevice,i.scope.stream));
}
void HipMtpTargetWindowCapture::use_device_greedy(const HipGreedyRowsReply* replies) noexcept {impl_->device_greedy=replies;}
void HipMtpTargetWindowCapture::completed_before_publish(){
    auto& i=*impl_;i.require_pool();require(i.active && std::all_of(i.active->recorded.begin(),i.active->recorded.end(),[](auto x){return x==1U;}),
        "MTP window completed without every wide snapshot");
    // Snapshot every head before selection/promotion and any subsequent scratch reuse.
    if(prefix_batch_certification_enabled() && prompt_terminal_digest_enabled()) {
        std::vector<float> row;
        for(const auto& view:i.logit_views) {
            read_hip_logits_row(view,i.vocab,i.scope.stream,row);
            i.active->row_digests.push_back(raw_fp32_logits_digest(row));
        }
    }
    if(mtp_logits_digest_enabled() || mtp_step_logits_digest_enabled()) {
        i.active->logit_digests.reserve(i.rows);
        for(std::uint32_t row=0;row<i.rows;++row) {
            std::vector<float> logits;
            read_hip_logits_row(i.logit_views[row],i.vocab,i.scope.stream,logits);
            const auto bytes=std::as_bytes(std::span<const float>(logits));
            const auto sha=sha256_bytes(bytes);i.active->logit_digests.push_back(sha);
            if(mtp_step_logits_digest_enabled())
                std::fprintf(stderr,"mtp-step-logits physical capture=%p position=%llu epoch=%llu row=%u rows=%u vocab=%u sha256=%s\n",
                    static_cast<void*>(this),static_cast<unsigned long long>(i.active->base.next_position+row),
                    static_cast<unsigned long long>(i.active->epoch),row,i.rows,i.vocab,model_content_identity_hex(sha).c_str());
            if(mtp_logits_digest_enabled()) std::fprintf(stderr,"mtp-logits physical position=%llu epoch=%llu row=%u sha256=%s\n",
                static_cast<unsigned long long>(i.active->base.next_position+row),
                static_cast<unsigned long long>(i.active->epoch),row,model_content_identity_hex(sha).c_str());
        }
    }
    if(i.transaction_selection) {
        require(!i.row_selection,"MTP selection callbacks are mutually exclusive");
        // The window stream already completed: the device replies are final (unvalidated here; a nonzero
        // status sends that row to the full-row selector, which reports it exactly as serial selection does).
        auto plan=i.transaction_selection(i.logit_views,i.device_greedy?std::span<const HipGreedyRowsReply>(i.device_greedy,i.rows):
            std::span<const HipGreedyRowsReply>{});
        require((plan.decided_rows || plan.stop==MtpSelectionStop::selection_failure) && plan.decided_rows<=i.rows && plan.selected_ids.size()==plan.decided_rows,
            "MTP transaction returned an invalid decided prefix");
        std::fill(i.active->predictions.begin(),i.active->predictions.end(),UINT32_MAX);
        for(std::size_t r=0;r<plan.decided_rows;++r) {
            require(plan.selected_ids[r]<i.vocab,"MTP transaction selected outside vocabulary");
            i.active->predictions[r]=plan.selected_ids[r];
        }
        i.active->selection=std::move(plan);i.active->completed=true;return;
    }
    // The actual producer has completed. Queue independent reductions/readbacks
    // before waiting, then validate every result before the batch can publish.
    if(i.device_greedy) {
        // The window stream already completed (its synchronize precedes this call).
        for(std::uint32_t row=0;row<i.rows;++row) {
            const auto reply=i.device_greedy[row];
            require(reply.status==0U && reply.token<i.vocab,"GPU greedy rows logits must all be finite and selection valid");
            i.active->predictions[row]=reply.token;
        }
    } else {
        for(std::uint32_t row=0;row<i.rows;++row)
            i.selectors[row]->submit(i.logit_views[row],i.vocab,i.active->base.next_position+row,i.active->epoch);
        for(std::uint32_t row=0;row<i.rows;++row)i.active->predictions[row]=i.selectors[row]->complete();
    }
    // Sampled selection replaces the finite first-max tokens before any row is published.
    if(i.row_selection) {
        i.row_selection(i.logit_views,i.active->predictions);
        for(const auto token:i.active->predictions) require(token<i.vocab,"MTP sampled window selection is outside the vocabulary");
    }
    if(!prefix_batch_certification_enabled() && prompt_terminal_digest_enabled()) {
        std::vector<float> row;
        read_hip_logits_row(i.logit_views.back(),i.vocab,i.scope.stream,row);
        i.active->terminal_digest=raw_fp32_logits_digest(row);
    }
    i.active->completed=true;
}
void HipMtpTargetWindowCapture::set_row_selection(RowSelection selection) {impl_->row_selection=std::move(selection);}
void HipMtpTargetWindowCapture::set_transaction_selection(TransactionSelection selection) {impl_->transaction_selection=std::move(selection);}
const MtpSelectionPrefix* HipMtpTargetWindowBatch::selection_prefix()const{require(valid(),"MTP target batch is invalid");return lease_->selection?&*lease_->selection:nullptr;}
HipMtpTargetWindowBatch HipMtpTargetWindowCapture::seal(const SerialSessionContext& session){
    auto& i=*impl_;require(i.active && i.active->completed && session.health==SerialSessionHealth::ready &&
        session.execution_identity==i.serial_id && session.runtime_compatibility_identity==i.active->base.runtime_compatibility_identity &&
        session.state_layout_identity==i.active->base.state_layout_identity && session.next_position==i.active->base.next_position+i.rows &&
        session.committed_tokens==i.active->base.committed_tokens+i.rows && arithmetic_lineage_valid(session.arithmetic_lineage,session.next_position) &&
        session.arithmetic_lineage.intervals.size()==i.active->base.arithmetic_lineage.intervals.size()+1U,
        "MTP captured batch is not the actual committed whole-window frontier");
    const auto& interval=session.arithmetic_lineage.intervals.back();
    require(interval.producer==ArithmeticProducer::known_window_chunk && interval.begin==i.active->base.next_position && interval.end==session.next_position,
        "MTP target batch lacks its actual known-window producer receipt");
    copy_session(i.active->committed,session);
    auto lease=std::move(i.active);lease->sealed=true;lease->capture=shared_from_this();i.previous=lease;
    HipMtpTargetWindowBatch out;out.lease_=std::move(lease);return out;
}
void HipMtpTargetWindowCapture::abort()noexcept{auto& i=*impl_;if(i.active){i.failed=true;i.active.reset();}}
std::optional<ModelContentIdentity> HipMtpTargetWindowBatch::logits_digest(std::uint32_t row) const {
    require(row<logical_rows(),"MTP logit digest row is out of bounds");
    return lease_->logit_digests.empty()?std::nullopt:std::optional<ModelContentIdentity>{lease_->logit_digests[row]};
}
bool HipMtpTargetWindowBatch::valid()const noexcept{return lease_ && lease_->sealed && lease_->completed && lease_->capture;}
std::uint32_t HipMtpTargetWindowBatch::logical_rows()const{require(valid(),"MTP target batch is invalid");return lease_->capture->impl_->rows;}
std::uint64_t HipMtpTargetWindowBatch::chunk_begin()const{require(valid(),"MTP target batch is invalid");return lease_->base.next_position;}
std::uint64_t HipMtpTargetWindowBatch::chunk_end()const{require(valid(),"MTP target batch is invalid");return lease_->committed.next_position;}
std::uint64_t HipMtpTargetWindowBatch::source_position(std::uint32_t row)const{require(row<logical_rows(),"MTP batch row is out of bounds");return chunk_begin()+row;}
std::uint32_t HipMtpTargetWindowBatch::input_token(std::uint32_t row)const{require(row<logical_rows(),"MTP batch row is out of bounds");return lease_->tokens[row];}
std::uint32_t HipMtpTargetWindowBatch::predicted_token(std::uint32_t row)const{require(row<logical_rows() && (!lease_->selection || row<lease_->selection->decided_rows),"MTP batch row is undecided or out of bounds");return lease_->predictions[row];}
std::optional<ModelContentIdentity> HipMtpTargetWindowBatch::terminal_logits_digest(std::uint32_t row)const {
    require(row<logical_rows(),"MTP batch row is out of bounds");
    return lease_->terminal_logits_digest(row);
}
MtpDeviceView HipMtpTargetWindowBatch::view(std::uint32_t row)const{require(row<logical_rows(),"MTP batch row is out of bounds");lease_->capture->impl_->require_pool();return lease_->capture->impl_->snapshots[row];}
const SerialSessionContext& HipMtpTargetWindowBatch::base_session()const{require(valid(),"MTP target batch is invalid");return lease_->base;}
const SerialSessionContext& HipMtpTargetWindowBatch::committed_session()const{require(valid(),"MTP target batch is invalid");return lease_->committed;}
ModelContentIdentity HipMtpTargetWindowBatch::target_content_identity()const{require(valid(),"MTP target batch is invalid");return lease_->capture->impl_->content;}
std::uint64_t HipMtpTargetWindowBatch::window_execution_identity()const{require(valid(),"MTP target batch is invalid");return lease_->capture->impl_->window_id;}
std::uint64_t HipMtpTargetWindowBatch::window_kernel_plan_identity()const{require(valid(),"MTP target batch is invalid");return lease_->capture->impl_->kernel_id;}
std::uint64_t HipMtpTargetWindowBatch::epoch()const{require(valid(),"MTP target batch is invalid");return lease_->epoch;}
const SessionStateMemoryBinding& HipMtpTargetWindowBatch::source_state_memory_binding()const{require(valid(),"MTP target batch is invalid");return lease_->capture->impl_->state_snapshot;}
const ExecutionMemoryBinding& HipMtpTargetWindowBatch::source_scratch_binding()const{require(valid(),"MTP target batch is invalid");return lease_->capture->impl_->source_scratch;}
const ModelRuntimeBinding& HipMtpTargetWindowBatch::source_weights_binding()const{require(valid(),"MTP target batch is invalid");return lease_->capture->impl_->weights_snapshot;}
const ExecutionMemoryBinding& HipMtpTargetWindowBatch::deferred_status_binding()const{require(valid(),"MTP target batch is invalid");return lease_->capture->impl_->scope.deferred_moe_status;}
void* HipMtpTargetWindowBatch::source_stream()const{require(valid(),"MTP target batch is invalid");return lease_->capture->impl_->scope.stream;}
} // namespace strix
