#include "strix/env_flags.hpp"
#include "strix/hip_check.hpp"
#include "strix/hip_mtp_target_hidden.hpp"
#include "strix/mtp_payload_geometry.hpp"
#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/hip_qwen_rotary_table.hpp"
#include "strix/hip_mtp_target_window.hpp"
#include "strix/mtp_target_window_policy.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/serial_host_state_shadow.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/hip_mtp_native_validation.hpp"
#include "strix/session_executor_handoff.hpp"
#include "strix/hip_token_sampling.hpp"
#include "strix/hip_token_sampling_device.hpp"
#include <chrono>
#include "strix/serial_request.hpp"
#include "strix/arithmetic_lineage.hpp"
#include "strix/prompt_terminal_diagnostics.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/mtp_step_logits_digest.hpp"
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <vector>
namespace strix {
namespace {
using strix::hip_check;
}
namespace {
bool same_target_state(const SessionStateMemoryBinding& a,const SessionStateMemoryBinding& b) {
    return a.layout_identity==b.layout_identity && a.device_base==b.device_base && a.device_bytes==b.device_bytes &&
        a.device_allocation_generation==b.device_allocation_generation && a.host_base==b.host_base &&
        a.host_bytes==b.host_bytes && a.host_allocation_generation==b.host_allocation_generation &&
        !a.stage_mapping && !b.stage_mapping;
}
bool same_target_weights(const ModelRuntimeBinding& a,const ModelRuntimeBinding& b) {
    if(a.manifest_layout_identity!=b.manifest_layout_identity || a.model_content_identity!=b.model_content_identity ||
       a.tensors.size()!=b.tensors.size())return false;
    for(std::size_t index=0;index<a.tensors.size();++index) {
        const auto& x=a.tensors[index];const auto& y=b.tensors[index];
        if(x.tensor_index!=y.tensor_index || x.storage!=y.storage || x.address!=y.address || x.bytes!=y.bytes ||
           x.allocation_generation!=y.allocation_generation || x.owner_node!=y.owner_node)return false;
    }
    return true;
}
bool same_target_session(const SerialSessionContext& a,const SerialSessionContext& b) {
    return a.execution_identity==b.execution_identity && a.state_layout_identity==b.state_layout_identity &&
        a.runtime_compatibility_identity==b.runtime_compatibility_identity && a.next_position==b.next_position &&
        a.committed_tokens==b.committed_tokens && a.health==b.health && a.state.valid==b.state.valid &&
        a.state.generations==b.state.generations && a.arithmetic_lineage.history_identity==b.arithmetic_lineage.history_identity &&
        a.arithmetic_lineage.tainted==b.arithmetic_lineage.tainted && a.arithmetic_lineage.intervals==b.arithmetic_lineage.intervals;
}
}
struct HipMtpTargetHiddenTicket::Lease {
    std::shared_ptr<HipMtpTargetHidden> owner;
    MtpDeviceView snapshot;
    std::uint64_t allocation_lease=0,position=0;
    std::uint32_t token=0,predicted=0;
    SerialSessionContext committed;
    std::shared_ptr<const HipMtpTargetWindowBatch> batch;
    std::uint32_t batch_row=0;
    ModelContentIdentity content;
    std::uint64_t producer=0; // execution identity that produced a serial row
    // Set for rows promoted from a partially accepted window: the batch's committed session
    // covers every executed row, this one ends at the accepted prefix.
    std::optional<SerialSessionContext> prefix_committed;
    // Set for a row adopted from a tiled prefill block: the block's first position.
    std::optional<std::uint64_t> block_begin;
    // Retained actual FP32 row hash for serial, MTP and committed-step diagnostics.
    std::optional<ModelContentIdentity> logits_digest;
    OutputLogitsSnapshot terminal;
};
struct HipMtpTargetHidden::Impl {
    struct Slot {
        MtpDeviceView view;
        std::uint64_t lease=0;
        std::weak_ptr<const HipMtpTargetHiddenTicket::Lease> reader;
    };
    std::unique_ptr<const ModelContract> model;
    std::unique_ptr<const CompiledSerialExecution> compiled;
    std::shared_ptr<const RuntimeExecutionMemoryOwner> memory;
    std::shared_ptr<const HipModelResidency> weights;
    std::shared_ptr<HipRuntimeAllocator> allocator;
    std::shared_ptr<HipSetupQuiescenceGuard> guard;
    hipStream_t stream=nullptr;
    std::vector<Slot> slots;
    std::uint32_t tap_step=0;
    ExecutionBufferUse tap_use;
    ExecutionBufferAddress logits;
    std::unique_ptr<HipGreedyOutput> selector;
    std::uint64_t epoch=0;
    std::unique_ptr<SerialHostStateShadow> shadow;
    // Sampled selection (null: finite first-max) and its host copy of the logits row.
    std::shared_ptr<TokenSampler> sampler;
    std::vector<float> sampled_logits;
    std::function<void(std::uint64_t, std::span<const float>)> session_logits;
    std::unique_ptr<HipTokenSamplerDevice> device_sampler; // sampling-v1 on the GPU (STRIX_SAMPLE_HOST=1: host)
    ExecutionMemoryBinding deferred_status;
    std::uint64_t status_lease=0;
    bool busy=false,failed=false;
    // Split prefill: sibling prompt execution on the same scratch arena, and the admitted handoff.
    std::unique_ptr<const CompiledSerialExecution> prompt;
    ExecutionMemoryBinding prompt_scratch;
    struct Handoff {
        std::uint64_t position=0,prompt_execution=0,prompt_compatibility=0,decode_execution=0,decode_compatibility=0;
    };
    std::optional<Handoff> handoff;
    bool started=false;
    std::unique_ptr<MtpStepLogitsDigest> step_digest;
    void record_step(std::uint64_t position,const std::optional<ModelContentIdentity>& digest) {
        if(!step_digest)return;
        step_digest->commit(position,digest);
        std::fprintf(stderr,"mtp-step-logits committed stream=%p position=%llu vocab=%llu sha256=%s\n",
            static_cast<void*>(this),static_cast<unsigned long long>(position),
            static_cast<unsigned long long>(model->vocab_size),digest?model_content_identity_hex(*digest).c_str():"NO-REFERENCE");
    }
    ~Impl() {
        if(guard) guard->recover_or_fail_stop();
        if(stream) (void)hipStreamDestroy(stream);
        if(allocator && deferred_status.arena_base)
            allocator->free_device(deferred_status.arena_base,deferred_status.arena_bytes);
        if(allocator) for(auto& s:slots)
            if(s.view.address) allocator->free_device(s.view.address,s.view.bytes);
    }
};
HipMtpTargetHidden::HipMtpTargetHidden():impl_(std::make_unique<Impl>()){}
HipMtpTargetHidden::~HipMtpTargetHidden()=default;
void HipMtpTargetHiddenTicket::begin_step_logits_digest() const {
    if(!mtp_step_logits_digest_enabled())return;
    if(!valid())throw std::invalid_argument("MTP step digest needs actual pending target");
    auto& i=*lease_->owner->impl_;
    i.step_digest=std::make_unique<MtpStepLogitsDigest>(source_position());
    std::fprintf(stderr,"mtp-step-logits begin stream=%p begin=%llu vocab=%llu\n",static_cast<void*>(&i),
        static_cast<unsigned long long>(source_position()),static_cast<unsigned long long>(i.model->vocab_size));
    i.record_step(source_position(),lease_->logits_digest);
}
void HipMtpTargetHiddenTicket::print_step_logits_summary(std::uint64_t end) const {
    if(!mtp_step_logits_digest_enabled())return;
    if(!valid())throw std::invalid_argument("MTP step summary needs actual pending target");
    auto& i=*lease_->owner->impl_;
    if(!i.step_digest)return;
    const auto& d=*i.step_digest;
    std::fprintf(stderr,"mtp-step-logits summary stream=%p begin=%llu end=%llu count=%llu vocab=%llu coverage=%s sha256=%s\n",
        static_cast<void*>(&i),static_cast<unsigned long long>(d.begin()),static_cast<unsigned long long>(end),
        static_cast<unsigned long long>(d.count()),static_cast<unsigned long long>(i.model->vocab_size),
        d.complete(end)?"COMPLETE":"NO-REFERENCE",model_content_identity_hex(d.chain()).c_str());
    i.step_digest.reset();
}
std::uint32_t HipMtpTargetHidden::retained_rows() const noexcept {return static_cast<std::uint32_t>(impl_->slots.size());}
const ModelContract& HipMtpTargetHidden::model() const noexcept {return *impl_->model;}
const CompiledSerialExecution& HipMtpTargetHidden::compiled() const noexcept {return *impl_->compiled;}
std::shared_ptr<const RuntimeExecutionMemoryOwner> HipMtpTargetHidden::memory_owner() const noexcept {return impl_->memory;}
std::shared_ptr<const HipModelResidency> HipMtpTargetHidden::weights_owner() const noexcept {return impl_->weights;}
std::shared_ptr<HipRuntimeAllocator> HipMtpTargetHidden::allocator() const noexcept {return impl_->allocator;}
void* HipMtpTargetHidden::stream() const noexcept {return impl_->stream;}
std::shared_ptr<const HipSetupQuiescenceGuard> HipMtpTargetHidden::guard() const noexcept {return impl_->guard;}

bool HipMtpTargetHiddenTicket::valid() const noexcept {return !!lease_;}
bool HipMtpTargetHiddenTicket::same_producer(const HipMtpTargetHiddenTicket& b) const noexcept {
    return lease_ && b.lease_ && lease_->owner==b.lease_->owner;
}
bool HipMtpTargetHiddenTicket::same_commit(const HipMtpTargetHiddenTicket& b) const noexcept {
    return lease_ && b.lease_ && (lease_->batch ? lease_->batch==b.lease_->batch : lease_==b.lease_);
}
bool HipMtpTargetHiddenTicket::is_window() const noexcept {return lease_ && !!lease_->batch;}
bool HipMtpTargetHiddenTicket::is_prefill_block() const noexcept {return lease_ && !!lease_->block_begin;}
std::uint64_t HipMtpTargetHiddenTicket::chunk_begin() const {
    if(!lease_)throw std::invalid_argument("MTP target hidden ticket is empty");
    return lease_->batch?lease_->batch->chunk_begin():lease_->block_begin?*lease_->block_begin:lease_->position;
}
std::uint64_t HipMtpTargetHiddenTicket::chunk_end() const {
    if(!lease_)throw std::invalid_argument("MTP target hidden ticket is empty");
    if(lease_->prefix_committed) return lease_->prefix_committed->next_position;
    return lease_->batch?lease_->batch->chunk_end():lease_->position+1;
}
MtpDeviceView HipMtpTargetHiddenTicket::view() const {
    if(!lease_ || !lease_->owner) throw std::invalid_argument("MTP target hidden ticket is empty");
    const auto& i=*lease_->owner->impl_;
    i.guard->require_original();
    i.memory->validate_inventory_ownership();
    if(lease_->batch) return lease_->batch->view(lease_->batch_row);
    const auto v=lease_->snapshot;
    if(i.allocator->allocation_lease_identity(v.address,v.bytes,false)!=lease_->allocation_lease)
        throw std::invalid_argument("MTP target hidden allocation lease changed");
    return v;
}
std::uint64_t HipMtpTargetHiddenTicket::source_position() const {if(!lease_)throw std::invalid_argument("MTP target hidden ticket is empty");return lease_->position;}
std::uint32_t HipMtpTargetHiddenTicket::input_token() const {if(!lease_)throw std::invalid_argument("MTP target hidden ticket is empty");return lease_->token;}
std::uint32_t HipMtpTargetHiddenTicket::predicted_token() const {if(!lease_)throw std::invalid_argument("MTP target hidden ticket is empty");return lease_->predicted;}
const SerialSessionContext& HipMtpTargetHiddenTicket::committed_session() const {if(!lease_)throw std::invalid_argument("MTP target hidden ticket is empty");return lease_->prefix_committed?*lease_->prefix_committed:lease_->batch?lease_->batch->committed_session():lease_->committed;}
std::optional<ModelContentIdentity> HipMtpTargetHiddenTicket::serial_logits_digest() const {return lease_ ? lease_->logits_digest : std::nullopt;}
std::optional<ModelContentIdentity> HipMtpTargetHiddenTicket::terminal_logits_digest() const {
    if(!lease_) return std::nullopt;
    return lease_->terminal.digest;
}
ModelContentIdentity HipMtpTargetHiddenTicket::target_content_identity() const {if(!lease_)throw std::invalid_argument("MTP target hidden ticket is empty");return lease_->content;}
RotaryIdentity HipMtpTargetHiddenTicket::producer_rotary_identity() const {
    if(!lease_ || !lease_->owner) throw std::invalid_argument("MTP target hidden ticket is empty");
    return lease_->owner->model().rotary_identity();
}
void HipMtpTargetHiddenTicket::admit_rotary_reader(const RotaryIdentity& reader) const {
    require_rotary_producer_identity(producer_rotary_identity(),reader);
}
std::uint64_t HipMtpTargetHiddenTicket::producer_execution_identity() const {
    if(!lease_)throw std::invalid_argument("MTP target hidden ticket is empty");
    return lease_->batch || lease_->prefix_committed ? committed_session().execution_identity : lease_->producer;
}
bool HipMtpTargetHiddenTicket::admits_executor_change(const HipMtpTargetHiddenTicket& next) const noexcept {
    if(!lease_ || !next.lease_ || lease_->owner!=next.lease_->owner || !lease_->owner || lease_->batch || next.lease_->batch ||
       lease_->prefix_committed || next.lease_->prefix_committed || lease_->block_begin || next.lease_->block_begin) return false;
    const auto& h=lease_->owner->impl_->handoff;
    if(!h) return false;
    const auto& a=lease_->committed;const auto& b=next.lease_->committed;
    return a.execution_identity==h->prompt_execution && a.runtime_compatibility_identity==h->prompt_compatibility &&
        b.execution_identity==h->decode_execution && b.runtime_compatibility_identity==h->decode_compatibility &&
        lease_->producer==h->prompt_execution && next.lease_->producer==h->prompt_execution &&
        next.lease_->position+1U==h->position && b.next_position==h->position;
}
std::shared_ptr<HipMtpTargetHidden> HipMtpTargetHidden::create(
    const ModelContract& m,const CompiledSerialExecution& c,
    std::shared_ptr<const RuntimeExecutionMemoryOwner> memory,
    std::shared_ptr<const HipModelResidency> weights,
    std::shared_ptr<HipRuntimeAllocator> allocator,std::uint32_t rows,std::uint64_t budget,HipMtpTargetDeferredConfig deferred) {
    require_rotary_producer_identity(c.rotary_table?c.rotary_table->identity():RotaryIdentity{},m.rotary_identity());
    if(m.qwen_rotary_table) validate_qwen_rotary_reader(*m.qwen_rotary_table,nullptr);
    if(!m.validate().empty() || !memory || !weights || !allocator || !rows ||
       !c.execution_identity || m.vocab_size>std::numeric_limits<std::uint32_t>::max() || c.logical_rows!=1 || c.request_concurrency!=1 ||
       !weights->binding().model_content_identity.valid())
        throw std::invalid_argument("MTP target tap requires actual single-row retained target resources");
    if(weights.owner_before(weights->weak_from_this().lock()) ||
       weights->weak_from_this().lock().owner_before(weights))
        throw std::invalid_argument("MTP target capture requires actual shared residency control block");
    (void)weights->owner_inventory(weights);
    memory->validate_inventory_ownership();
    if(!!deferred.state_layout!=!!deferred.generations)
        throw std::invalid_argument("MTP target D1 requires actual state layout and generation issuer together");
    if(deferred.state_layout) {
        validate_session_state_memory_binding(*deferred.state_layout,memory->state());
        if(!deferred.state_layout->host_total_bytes || !deferred.state_layout->host_alignment)
            throw std::invalid_argument("MTP target D1 requires bounded transactional host history");
    }
    const auto bytes=mtp_native::product(mtp_native::product(m.hidden_size,m.hc_count),2);
    const auto required=mtp_payload::target_capture(m,rows,deferred.state_layout);
    if(required>budget)
        throw std::invalid_argument("MTP target capture exceeds upfront retained budget");
    auto result=std::shared_ptr<HipMtpTargetHidden>(new HipMtpTargetHidden);
    auto& i=*result->impl_;
    i.model=std::make_unique<const ModelContract>(m);i.compiled=std::make_unique<const CompiledSerialExecution>(c);i.memory=std::move(memory);i.weights=std::move(weights);i.allocator=std::move(allocator);
    i.guard=std::make_shared<HipSetupQuiescenceGuard>();
    auto head=std::find_if(c.execution.steps.begin(),c.execution.steps.end(),[](const auto& s){return s.op==SerialOp::final_hyper_mix;});
    if(head==c.execution.steps.end() || head==c.execution.steps.begin())
        throw std::invalid_argument("MTP target tap requires final wide hidden producer");
    const auto& producer=*std::prev(head);
    if(producer.op!=SerialOp::mlp_inject || producer.layer!=m.hidden_layers-1)
        throw std::invalid_argument("MTP target hidden must be last MLP injection before final mix");
    auto io=std::find_if(c.io.steps.begin(),c.io.steps.end(),[&](const auto& s){return s.step_id==producer.id;});
    auto headio=std::find_if(c.io.steps.begin(),c.io.steps.end(),[&](const auto& s){return s.step_id==head->id;});
    if(io==c.io.steps.end() || headio==c.io.steps.end() || io->writes.size()!=1 ||
       io->writes[0].buffer.buffer_class!=ExecutionBufferClass::expanded_hidden ||
       std::find(headio->reads.begin(),headio->reads.end(),io->writes[0])==headio->reads.end())
        throw std::invalid_argument("MTP target tap wide producer/consumer IO differs");
    const auto* allocation=find_arena_allocation(c.arena,io->writes[0]);
    if(!allocation || allocation->bytes!=bytes)
        throw std::invalid_argument("MTP target capture requires exact BF16 wide activation layout");
    i.tap_step=producer.id;i.tap_use=io->writes[0];
    if(c.execution.steps.back().op!=SerialOp::lm_head || c.io.steps.empty() ||
       c.io.steps.back().op!=SerialOp::lm_head || c.io.steps.back().writes.size()!=1)
        throw std::invalid_argument("MTP target producer requires actual terminal target logits");
    i.logits=resolve_execution_buffer(c.execution_identity,c.arena,i.memory->scratch(),c.io.steps.back().writes[0]);
    if(i.logits.bytes!=mtp_native::product(m.vocab_size,4))
        throw std::invalid_argument("MTP target producer requires full FP32 target vocabulary");
    i.selector=std::make_unique<HipGreedyOutput>(static_cast<std::uint32_t>(m.vocab_size),
        i.memory->scratch(),i.memory,i.guard);
    hip_check(hipStreamCreateWithFlags(&i.stream,hipStreamNonBlocking),"MTP target capture stream");
    if(deferred.state_layout) {
        i.shadow=std::make_unique<SerialHostStateShadow>(*deferred.state_layout,i.memory->state(),deferred.generations->next());
        i.deferred_status={c.execution_identity,i.allocator->allocate_device(8,256),8,deferred.generations->next()};
        i.status_lease=i.allocator->allocation_lease_identity(i.deferred_status.arena_base,8,false);
        if(!i.status_lease)throw std::runtime_error("MTP target D1 lacks actual allocator lease");
        hip_check(hipMemsetAsync(reinterpret_cast<void*>(i.deferred_status.arena_base),0,8,i.stream),"MTP target D1 initialization");
        hip_check(hipStreamSynchronize(i.stream),"MTP target D1 setup completion");
        i.guard->require_original();
    }
    i.slots.resize(rows);
    for(auto& s:i.slots) {
        s.view.address=i.allocator->allocate_device(bytes,256);s.view.bytes=bytes;
        s.lease=i.allocator->allocation_lease_identity(s.view.address,bytes,false);
        if(!s.lease) throw std::runtime_error("MTP target capture lacks allocation lease");
        s.view.allocation_generation=s.lease;
    }
    return result;
}
// Host selection runs after the execution transaction: a selector refusal leaves a
// healthy committed model row. The caller discards this request, never retries unmasked.
HipMtpTargetHiddenTicket HipMtpTargetHidden::execute_serial(
    SerialSessionContext& session,const ExecutionRuntimeResources& resources,
    DirectLaunchDispatcher& dispatcher,std::uint32_t token,const TokenDraft* draft,
    const HipMtpHostRowSelector& host_selector) {
    auto ticket=execute_serial_unselected(session,resources,dispatcher,token,draft,!!host_selector);
    if(host_selector) {
        auto& i=*impl_;
        try {
            read_hip_logits_row(i.logits,static_cast<std::uint32_t>(i.model->vocab_size),i.stream,i.sampled_logits);
        } catch(...) {
            i.failed=true;i.guard->recover_or_fail_stop();
            session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true;
            throw;
        }
        const auto selected=host_selector(i.sampled_logits);
        if(selected>=i.model->vocab_size) throw std::invalid_argument("host row selector token outside vocabulary");
        std::const_pointer_cast<HipMtpTargetHiddenTicket::Lease>(ticket.lease_)->predicted=selected;
    }
    return ticket;
}
HipMtpTargetHiddenTicket HipMtpTargetHidden::execute_serial_unselected(
    SerialSessionContext& session,const ExecutionRuntimeResources& resources,
    DirectLaunchDispatcher& dispatcher,std::uint32_t token,const TokenDraft* draft,bool host_selection) {
    auto& i=*impl_;
    if(draft && !i.sampler) throw std::invalid_argument("MTP target draft decision requires a token sampler");
    // The prompt lane runs while the session is owned by the sibling prompt execution.
    const bool prompt_lane=i.prompt && !i.handoff && session.execution_identity==i.prompt->execution_identity;
    const auto& lane=prompt_lane?*i.prompt:*i.compiled;
    const auto same_scratch=[&](const ExecutionMemoryBinding* b) {
        if(!prompt_lane) return b==&i.memory->scratch();
        return b && b->execution_identity==i.prompt_scratch.execution_identity && b->arena_base==i.prompt_scratch.arena_base &&
            b->arena_bytes==i.prompt_scratch.arena_bytes && b->allocation_generation==i.prompt_scratch.allocation_generation;
    };
    if(i.busy || i.failed || session.health!=SerialSessionHealth::ready ||
       session.execution_identity!=lane.execution_identity ||
       !same_scratch(resources.scratch) || resources.session_state_memory!=&i.memory->state() ||
       resources.model_weights!=&i.weights->binding() || !resources.model_manifest ||
       resources.model_manifest->layout_identity!=i.weights->binding().manifest_layout_identity ||
       token>=i.model->vocab_size || current_hip_launch_context())
        throw std::invalid_argument("MTP target capture requires idle exact retained eager target binding");
    i.guard->require_original(); i.memory->validate_inventory_ownership();
    if(i.epoch==std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("MTP target output epoch exhausted");
    auto slot=std::find_if(i.slots.begin(),i.slots.end(),[](auto& s){return s.reader.expired();});
    if(slot==i.slots.end()) throw std::runtime_error("MTP target capture slots retained by active readers");
    // Allocate host publication metadata before starting native work.
    auto lease=std::make_shared<HipMtpTargetHiddenTicket::Lease>();
    lease->owner=shared_from_this();lease->snapshot=slot->view;lease->allocation_lease=slot->lease;
    lease->position=session.next_position;lease->token=token;
    lease->content=i.weights->binding().model_content_identity;
    lease->producer=lane.execution_identity;
    struct Tap final:EagerExecutionDispatcher {
        Impl& owner;DirectLaunchDispatcher& next;MtpDeviceView destination;
        std::uint64_t position,identity;const ExecutionMemoryBinding* memory;bool captured=false;
        Tap(Impl& o,DirectLaunchDispatcher& n,MtpDeviceView d,std::uint64_t p,std::uint64_t e,const ExecutionMemoryBinding* m)
            :owner(o),next(n),destination(d),position(p),identity(e),memory(m){}
        void after(const ExecutionDispatchContext& c) {
            if(c.step_id!=owner.tap_step) return;
            if(captured || c.execution_identity!=identity ||
               c.position!=position || c.logical_rows!=1 || c.op!=SerialOp::mlp_inject ||
               c.writes.size()!=1 || c.writes[0]!=owner.tap_use ||
               c.memory!=memory)
                throw std::invalid_argument("MTP target capture dispatch producer differs");
            auto source=resolve_dispatch_write(c,0);
            if(source.bytes!=destination.bytes || !source.allocation_generation)
                throw std::invalid_argument("MTP target capture source is undersized");
            hip_check(hipMemcpyAsync(reinterpret_cast<void*>(destination.address),
                reinterpret_cast<const void*>(source.address),destination.bytes,hipMemcpyDeviceToDevice,owner.stream),"MTP target wide capture");
            captured=true;
        }
        void dispatch_kernel(const ExecutionDispatchContext& c) override {next.dispatch_kernel(c);after(c);}
        void dispatch_runtime_step(const ExecutionDispatchContext& c) override {next.dispatch_runtime_step(c);after(c);}
    } tap(i,dispatcher,slot->view,session.next_position,lane.execution_identity,resources.scratch);
    if(i.deferred_status.arena_base && i.allocator->allocation_lease_identity(i.deferred_status.arena_base,8,false)!=i.status_lease)
        throw std::invalid_argument("MTP target D1 backing lease changed");
    i.busy=true;i.started=true;
    try {
        auto launch_resources=resources;
        if(i.shadow){i.shadow->begin();launch_resources.session_state_memory=&i.shadow->binding();}
        HipLaunchContext context;context.stream=i.stream;context.deferred_moe_status=i.deferred_status;
        if(context.deferred_moe_status.arena_base) context.deferred_moe_status.execution_identity=lane.execution_identity;
        HipLaunchScope scope(context);
        (void)execute_serial_token(lane,session,launch_resources,token,tap,[&] {
            if(!tap.captured) throw std::runtime_error("MTP target hidden producer was not executed");
            DeferredMoeFirstError status{};
            if(i.deferred_status.arena_base)
                hip_check(hipMemcpyAsync(&status,reinterpret_cast<const void*>(i.deferred_status.arena_base),sizeof(status),
                    hipMemcpyDeviceToHost,i.stream),"MTP target D1 readback");
            // The copy and D1 readback precede this completion on the same stream.
            hip_check(hipStreamSynchronize(i.stream),"MTP target execution completion");
            i.guard->require_original();
            if(i.deferred_status.arena_base)check_deferred_moe_status(status);
            // Host policy validation belongs after the model execution commits.
            // Its nonfinite-row refusal must not poison this execution transaction.
            if(!host_selection)lease->predicted=i.selector->select(i.logits,static_cast<std::uint32_t>(i.model->vocab_size),
                lease->position,++i.epoch);
            if(i.sampler) {
                const auto t0=std::chrono::steady_clock::now();
                const auto vocabulary=static_cast<std::uint32_t>(i.model->vocab_size);
                if(device_token_sampling_enabled(i.sampler->policy())) {
                    if(!i.device_sampler) i.device_sampler=std::make_unique<HipTokenSamplerDevice>(vocabulary,1U);
                    const DeviceSampleRow row{i.logits,lease->position+1U,draft};
                    const auto decision=i.device_sampler->decide(*i.sampler,std::span<const DeviceSampleRow>(&row,1U),i.stream).front();
                    i.sampler->count(decision);
                    lease->predicted=decision.token;
                } else
                    read_hip_logits_row(i.logits,vocabulary,i.stream,i.sampled_logits);
                const auto t1=std::chrono::steady_clock::now();
                if(!device_token_sampling_enabled(i.sampler->policy()))
                    lease->predicted=i.sampler->decide(i.sampled_logits,lease->position+1U,draft).token;
                if(sample_timing_enabled()) record_sample_timing(SampleTimingKind::serial,1U,
                    std::chrono::duration<double>(t1-t0).count(),
                    std::chrono::duration<double>(std::chrono::steady_clock::now()-t1).count(),1U);
            }
            // STRIX_SERIAL_LOGITS_DIGEST=1 (diagnostic, gates): the row's FP32 logits digest (prompt receipts report the terminal one).
            static const bool logits_digest=strix::envcfg::flag("STRIX_SERIAL_LOGITS_DIGEST", false);
            const MtpSerialLogitsDigestPolicy digest_policy{
                logits_digest,mtp_logits_digest_enabled(),mtp_step_logits_digest_enabled()};
            // One read of the logits row serves the digests and the session callback (R2-18).
            const bool digest=digest_policy.enabled() || prompt_terminal_digest_enabled();
            if(digest || i.session_logits) {
                std::vector<float> row;
                read_hip_logits_row(i.logits,static_cast<std::uint32_t>(i.model->vocab_size),i.stream,row);
                if(digest) {
                    lease->logits_digest=digest_policy.capture(row);
                    if(prompt_terminal_digest_enabled()) lease->terminal.capture(raw_fp32_logits_digest(row));
                }
                if(i.session_logits) i.session_logits(lease->position,row);
            }
            if(i.shadow)i.shadow->publish();
        });
        if(resources.qsa_committed_pool) resources.qsa_committed_pool->commit(false,session.next_position);
        lease->committed=session;
        i.record_step(lease->position,lease->logits_digest);
        if(mtp_logits_digest_enabled() && lease->logits_digest)
            std::fprintf(stderr,"mtp-logits committed position=%llu sha256=%s\n",static_cast<unsigned long long>(lease->position),
                model_content_identity_hex(*lease->logits_digest).c_str());
        if(session.next_position!=lease->position+1 || session.health!=SerialSessionHealth::ready)
            throw std::runtime_error("MTP target capture lacks committed target row");
        slot->reader=lease; i.busy=false;
        HipMtpTargetHiddenTicket ticket;ticket.lease_=std::move(lease);return ticket;
    } catch(...) {
        if(resources.qsa_committed_pool) resources.qsa_committed_pool->cancel();
        i.failed=true;i.guard->recover_or_fail_stop();if(i.shadow)i.shadow->cancel();i.busy=false;
        session.health=SerialSessionHealth::poisoned;session.arithmetic_lineage.tainted=true;
        throw;
    }
}
void HipMtpTargetHidden::set_session_logits_observer(std::function<void(std::uint64_t, std::span<const float>)> observer) {
    // Clearing is also allowed during failure unwind, before captured callback locals expire.
    if(observer && (impl_->busy || impl_->failed)) throw std::invalid_argument("session logits observer needs idle target");
    impl_->session_logits=std::move(observer);
}
void HipMtpTargetHidden::set_token_sampler(std::shared_ptr<TokenSampler> sampler) {
    auto& i=*impl_;
    if(i.busy || i.failed || i.sampler) throw std::invalid_argument("MTP target token sampler is set once on an idle producer");
    i.sampler=std::move(sampler);
}
void HipMtpTargetHidden::replace_token_sampler(std::shared_ptr<TokenSampler> sampler) {
    auto& i=*impl_;
    if(i.busy || i.failed) throw std::invalid_argument("MTP target token sampler changes only on an idle healthy producer");
    i.sampler=std::move(sampler);
}
std::shared_ptr<TokenSampler> HipMtpTargetHidden::token_sampler() const noexcept {return impl_->sampler;}
HipMtpTargetHiddenTicket HipMtpTargetHidden::adopt_prefill_row(const SerialSessionContext& session,
    std::uint32_t token,MtpDeviceView hidden,std::span<const float> logits,const RotaryIdentity& producer_rotary,
    const HipMtpHostRowSelector& host_selector) {
    auto& i=*impl_;
    require_rotary_producer_identity(producer_rotary,i.model->rotary_identity());
    const auto& lineage=session.arithmetic_lineage;
    const auto* last=lineage.intervals.empty()?nullptr:&lineage.intervals.back();
    if(i.busy || i.failed || i.prompt || session.health!=SerialSessionHealth::ready ||
       session.execution_identity!=i.compiled->execution_identity || !session.next_position ||
       session.committed_tokens!=session.next_position || lineage.tainted || !last ||
       (last->producer!=ArithmeticProducer::prefill_chunk && last->producer!=ArithmeticProducer::known_window_chunk) ||
       last->end!=session.next_position || last->begin>=last->end ||
       !arithmetic_lineage_valid(lineage,session.next_position) ||
       token>=i.model->vocab_size || logits.size()!=i.model->vocab_size || !hidden.address ||
       current_hip_launch_context())
        throw std::invalid_argument("MTP prefill row adoption requires the idle target and a just-committed prefill block");
    i.guard->require_original(); i.memory->validate_inventory_ownership();
    auto slot=std::find_if(i.slots.begin(),i.slots.end(),[](auto& s){return s.reader.expired();});
    if(slot==i.slots.end()) throw std::runtime_error("MTP target capture slots retained by active readers");
    if(hidden.bytes!=slot->view.bytes) throw std::invalid_argument("MTP prefill row hidden width differs from the target tap");
    auto lease=std::make_shared<HipMtpTargetHiddenTicket::Lease>();
    lease->owner=shared_from_this();lease->snapshot=slot->view;lease->allocation_lease=slot->lease;
    lease->position=session.next_position-1U;lease->token=token;
    lease->content=i.weights->binding().model_content_identity;
    lease->producer=session.execution_identity;lease->block_begin=last->begin;
    if(prefix_batch_certification_enabled() && prompt_terminal_digest_enabled()) lease->terminal.capture(raw_fp32_logits_digest(logits));
    if(mtp_step_logits_digest_enabled())lease->logits_digest=sha256_bytes(std::as_bytes(logits));
    lease->predicted=select_greedy_token(logits,i.model->vocab_size);
    // Sampled decoding: the block's last row selects through the sampler, as a final serial row would.
    if(i.sampler) lease->predicted=i.sampler->decide(logits,session.next_position,nullptr).token;
    if(host_selector) {
        lease->predicted=host_selector(logits);
        if(lease->predicted>=i.model->vocab_size) throw std::invalid_argument("host row selector token outside vocabulary");
    }
    i.busy=true;
    try {
        hip_check(hipMemcpyAsync(reinterpret_cast<void*>(slot->view.address),reinterpret_cast<const void*>(hidden.address),
            static_cast<std::size_t>(hidden.bytes),hipMemcpyDeviceToDevice,i.stream),"MTP prefill row capture");
        hip_check(hipStreamSynchronize(i.stream),"MTP prefill row capture completion");
        i.guard->require_original();
    } catch(...) {i.failed=true;i.busy=false;i.guard->recover_or_fail_stop();throw;}
    if(i.session_logits) i.session_logits(lease->position,logits);
    lease->committed=session;
    slot->reader=lease;i.busy=false;i.started=true;
    HipMtpTargetHiddenTicket ticket;ticket.lease_=std::move(lease);return ticket;
}
void HipMtpTargetHidden::set_prompt_execution(const CompiledSerialExecution& prompt) {
    auto& i=*impl_;
    require_rotary_producer_identity(prompt.rotary_table?prompt.rotary_table->identity():RotaryIdentity{},i.model->rotary_identity());
    i.guard->require_original();i.memory->validate_inventory_ownership();
    if(i.prompt || i.started || i.busy || i.failed)
        throw std::invalid_argument("MTP prompt execution is set once, before any target row");
    if(prompt.execution.steps.size()!=i.compiled->execution.steps.size() ||
       prompt.execution.steps[i.tap_step].op!=SerialOp::mlp_inject)
        throw std::invalid_argument("MTP prompt execution differs from the target tap geometry");
    i.prompt_scratch=sibling_scratch_binding(prompt,*i.compiled,i.memory->scratch());
    i.prompt=std::make_unique<const CompiledSerialExecution>(prompt);
}
HipMtpTargetHiddenTicket HipMtpTargetHidden::hand_off_prompt(SerialSessionContext& session,
    const HipMtpTargetHiddenTicket& pending,const ExecutionRuntimeResources& prompt_resources,
    const ExecutionRuntimeResources& decode_resources) {
    auto& i=*impl_;i.guard->require_original();i.memory->validate_inventory_ownership();
    if(!i.prompt || i.handoff || i.busy || i.failed || !pending.lease_ || pending.lease_->owner.get()!=this ||
       pending.lease_->batch || pending.lease_->prefix_committed || pending.lease_->producer!=i.prompt->execution_identity ||
       !same_target_session(pending.lease_->committed,session) || decode_resources.scratch!=&i.memory->scratch())
        throw std::invalid_argument("MTP prompt handoff requires the actual pending prompt row of this producer");
    auto slot=std::find_if(i.slots.begin(),i.slots.end(),[&](const auto& s){return s.view.address==pending.lease_->snapshot.address;});
    if(slot==i.slots.end() || slot->reader.lock()!=pending.lease_)
        throw std::invalid_argument("MTP prompt handoff pending row does not own its capture slot");
    const auto prompt_compatibility=session.runtime_compatibility_identity;
    hand_off_serial_session(session,{i.prompt.get(),&prompt_resources},{i.compiled.get(),&decode_resources});
    auto lease=std::make_shared<HipMtpTargetHiddenTicket::Lease>(*pending.lease_);
    lease->committed=session;
    i.handoff=Impl::Handoff{session.next_position,i.prompt->execution_identity,prompt_compatibility,
        session.execution_identity,session.runtime_compatibility_identity};
    slot->reader=lease;
    HipMtpTargetHiddenTicket ticket;ticket.lease_=std::move(lease);return ticket;
}
void HipMtpTargetHidden::validate_window_capture(const HipMtpTargetWindowCapture& capture) const {
    const auto& i=*impl_;i.guard->require_original();i.memory->validate_inventory_ownership();
    if(i.busy || i.failed || capture.source_serial_execution_identity()!=i.compiled->execution_identity ||
       !same_target_state(capture.source_state_memory_binding(),i.memory->state()) ||
       !same_target_weights(capture.source_weights_binding(),i.weights->binding()))
        throw std::invalid_argument("MTP window capture does not bind the actual retained C1 target");
}
std::vector<HipMtpTargetHiddenTicket> HipMtpTargetHidden::accept_verified_window(
    std::shared_ptr<const HipMtpTargetWindowBatch> batch,const HipMtpTargetHiddenTicket& base,
    const SerialSessionContext& target,std::span<const std::uint32_t> eos,std::uint32_t budget) {
    auto& i=*impl_;i.guard->require_original();i.memory->validate_inventory_ownership();
    if(!batch || !batch->valid() || !base.lease_ || base.lease_->owner.get()!=this || i.busy || i.failed ||
       !same_target_state(batch->source_state_memory_binding(),i.memory->state()) ||
       !same_target_weights(batch->source_weights_binding(),i.weights->binding()) ||
       batch->target_content_identity()!=i.weights->binding().model_content_identity ||
       !same_target_session(batch->base_session(),base.committed_session()) ||
       !same_target_session(batch->committed_session(),target) || target.health!=SerialSessionHealth::ready ||
       batch->chunk_begin()!=base.source_position()+1 || batch->chunk_end()!=target.next_position ||
       batch->chunk_end()<=batch->chunk_begin() ||
       batch->chunk_end()-batch->chunk_begin()!=batch->logical_rows() ||
       !arithmetic_lineage_valid(target.arithmetic_lineage,target.next_position))
        throw std::invalid_argument("MTP window promotion lacks exact actual target chunk binding");
    const auto rows=batch->logical_rows();
    const auto& previous=base.committed_session();
    if(target.committed_tokens<previous.committed_tokens || target.committed_tokens-previous.committed_tokens!=rows ||
       target.arithmetic_lineage.intervals.empty())
        throw std::invalid_argument("MTP window promotion target counters are not the actual chunk advance");
    const auto& interval=target.arithmetic_lineage.intervals.back();
    if(interval.producer!=ArithmeticProducer::known_window_chunk || interval.begin!=batch->chunk_begin() ||
       interval.end!=batch->chunk_end() || interval.numerical_plan_identity!=batch->window_execution_identity())
        throw std::invalid_argument("MTP window promotion lacks its actual producer recipe");
    const auto expected=prepare_known_window_chunk_lineage(previous.arithmetic_lineage,interval.begin,interval.end,
        interval.numerical_plan_identity,interval.reduction_plan_identity,interval.state_update_identity);
    if(!arithmetic_lineage_matches(expected,target.arithmetic_lineage,target.next_position))
        throw std::invalid_argument("MTP window promotion changed preceding target producer history");
    std::vector<std::uint32_t> inputs,predictions;
    inputs.reserve(rows);predictions.reserve(rows);
    for(std::uint32_t row=0;row<rows;++row) {
        if(batch->source_position(row)!=batch->chunk_begin()+row)
            throw std::invalid_argument("MTP window source positions are not the actual contiguous chunk");
        const auto v=batch->view(row);
        if(v.bytes!=mtp_native::product(mtp_native::product(i.model->hidden_size,i.model->hc_count),2))
            throw std::invalid_argument("MTP window wide snapshot is not the actual target geometry");
        inputs.push_back(batch->input_token(row));predictions.push_back(batch->predicted_token(row));
    }
    if(!assess_mtp_target_window(inputs,predictions,base.predicted_token(),budget,eos,i.model->vocab_size).accepts_complete_chunk())
        throw std::invalid_argument("MTP provisional window is not a complete accepted greedy target chunk");
    std::vector<HipMtpTargetHiddenTicket> result;result.reserve(rows);
    for(std::uint32_t row=0;row<rows;++row) {
        auto lease=std::make_shared<HipMtpTargetHiddenTicket::Lease>();
        lease->owner=shared_from_this();lease->batch=batch;lease->batch_row=row;
        if(prompt_terminal_digest_enabled()) lease->terminal.capture(batch->terminal_logits_digest(row));
        lease->logits_digest=batch->logits_digest(row);
        i.record_step(batch->source_position(row),lease->logits_digest);
        if(lease->logits_digest) std::fprintf(stderr,"mtp-logits committed position=%llu sha256=%s\n",
            static_cast<unsigned long long>(batch->source_position(row)),model_content_identity_hex(*lease->logits_digest).c_str());
        lease->position=batch->source_position(row);lease->token=inputs[row];lease->predicted=predictions[row];
        lease->content=batch->target_content_identity();
        HipMtpTargetHiddenTicket ticket;ticket.lease_=std::move(lease);result.push_back(std::move(ticket));
    }
    return result;
}
std::vector<HipMtpTargetHiddenTicket> HipMtpTargetHidden::accept_verified_prefix(
    std::shared_ptr<const HipMtpTargetWindowBatch> batch,const HipMtpTargetHiddenTicket& base,
    const SerialSessionContext& target,std::uint32_t n_keep) {
    auto& i=*impl_;i.guard->require_original();i.memory->validate_inventory_ownership();
    if(!batch || !batch->valid() || !base.lease_ || base.lease_->owner.get()!=this || i.busy || i.failed ||
       !same_target_state(batch->source_state_memory_binding(),i.memory->state()) ||
       !same_target_weights(batch->source_weights_binding(),i.weights->binding()) ||
       batch->target_content_identity()!=i.weights->binding().model_content_identity ||
       !same_target_session(batch->base_session(),base.committed_session()) ||
       target.health!=SerialSessionHealth::ready || !n_keep || n_keep>batch->logical_rows() ||
       batch->chunk_begin()!=base.source_position()+1 || batch->chunk_begin()+n_keep!=target.next_position ||
       target.committed_tokens!=base.committed_session().committed_tokens+n_keep ||
       !arithmetic_lineage_valid(target.arithmetic_lineage,target.next_position))
        throw std::invalid_argument("MTP prefix promotion lacks exact actual target chunk binding");
    std::vector<HipMtpTargetHiddenTicket> result;result.reserve(n_keep);
    for(std::uint32_t row=0;row<n_keep;++row) {
        if(batch->source_position(row)!=batch->chunk_begin()+row)
            throw std::invalid_argument("MTP window source positions are not the actual contiguous chunk");
        if(row && batch->input_token(row)!=batch->predicted_token(row-1U))
            throw std::invalid_argument("MTP prefix row was not an accepted draft");
        auto lease=std::make_shared<HipMtpTargetHiddenTicket::Lease>();
        lease->owner=shared_from_this();lease->batch=batch;lease->batch_row=row;
        if(prompt_terminal_digest_enabled()) lease->terminal.capture(batch->terminal_logits_digest(row));
        lease->logits_digest=batch->logits_digest(row);
        i.record_step(batch->source_position(row),lease->logits_digest);
        if(lease->logits_digest) std::fprintf(stderr,"mtp-logits committed position=%llu sha256=%s\n",
            static_cast<unsigned long long>(batch->source_position(row)),model_content_identity_hex(*lease->logits_digest).c_str());
        lease->position=batch->source_position(row);lease->token=batch->input_token(row);
        lease->predicted=batch->predicted_token(row);lease->content=batch->target_content_identity();
        lease->prefix_committed=target; // every prefix row's chunk ends at the accepted frontier
        HipMtpTargetHiddenTicket ticket;ticket.lease_=std::move(lease);result.push_back(std::move(ticket));
    }
    return result;
}
std::vector<HipMtpTargetHiddenTicket> HipMtpTargetHidden::accept_known_rows(
    std::shared_ptr<const HipMtpTargetWindowBatch> batch,const SerialSessionContext& target) {
    auto& i=*impl_;i.guard->require_original();i.memory->validate_inventory_ownership();
    if(!batch || !batch->valid() || i.busy || i.failed ||
       !same_target_state(batch->source_state_memory_binding(),i.memory->state()) ||
       !same_target_weights(batch->source_weights_binding(),i.weights->binding()) ||
       batch->target_content_identity()!=i.weights->binding().model_content_identity ||
       target.health!=SerialSessionHealth::ready || batch->chunk_begin()!=batch->base_session().next_position ||
       batch->chunk_begin()+batch->logical_rows()!=target.next_position ||
       target.committed_tokens!=batch->base_session().committed_tokens+batch->logical_rows() ||
       !arithmetic_lineage_valid(target.arithmetic_lineage,target.next_position))
        throw std::invalid_argument("MTP known prompt rows lack exact actual target chunk binding");
    std::vector<HipMtpTargetHiddenTicket> result;result.reserve(batch->logical_rows());
    for(std::uint32_t row=0;row<batch->logical_rows();++row) {
        if(batch->source_position(row)!=batch->chunk_begin()+row)
            throw std::invalid_argument("MTP window source positions are not the actual contiguous chunk");
        auto lease=std::make_shared<HipMtpTargetHiddenTicket::Lease>();
        lease->owner=shared_from_this();lease->batch=batch;lease->batch_row=row;
        if(prompt_terminal_digest_enabled()) lease->terminal.capture(batch->terminal_logits_digest(row));
        lease->logits_digest=batch->logits_digest(row);
        i.record_step(batch->source_position(row),lease->logits_digest);
        if(lease->logits_digest) std::fprintf(stderr,"mtp-logits committed position=%llu sha256=%s\n",
            static_cast<unsigned long long>(batch->source_position(row)),model_content_identity_hex(*lease->logits_digest).c_str());
        lease->position=batch->source_position(row);lease->token=batch->input_token(row);
        lease->predicted=batch->predicted_token(row);lease->content=batch->target_content_identity();
        lease->prefix_committed=target;
        HipMtpTargetHiddenTicket ticket;ticket.lease_=std::move(lease);result.push_back(std::move(ticket));
    }
    return result;
}
} // namespace strix
