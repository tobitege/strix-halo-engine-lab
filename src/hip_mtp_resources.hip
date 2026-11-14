#include "strix/hip_check.hpp"
#include "strix/hip_mtp_resources.hpp"
#include "strix/mtp_payload_geometry.hpp"
#include "strix/hip_indexed_attention_launcher.hpp"
#include "strix/hip_qwen_rotary_table.hpp"
#include "strix/mtp_sidecar_binding.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/execution_buffers.hpp"
#include "strix/indexed_attention_state.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace strix {
namespace {
constexpr std::size_t buffer_count=static_cast<std::size_t>(HipMtpBuffer::count);
void require(bool ok,const char* message) {if(!ok) throw std::invalid_argument(message);}
using strix::hip_check_message;
std::uint64_t mul(std::uint64_t a,std::uint64_t b) {
    if(a && b>UINT64_MAX/a) throw std::overflow_error("MTP allocation product overflows");return a*b;
}
bool same(const RuntimeTensorBinding& a,const RuntimeTensorBinding& b) {
    return a.tensor_index==b.tensor_index && a.storage==b.storage && a.address==b.address &&
        a.bytes==b.bytes && a.allocation_generation==b.allocation_generation && a.owner_node==b.owner_node;
}
template<class A,class B> bool same_owner(const std::shared_ptr<A>& a,const std::shared_ptr<B>& b) {
    return a && b && !a.owner_before(b) && !b.owner_before(a);
}
void require_residency(const std::shared_ptr<const HipModelResidency>& owner) {
    require(owner && same_owner(owner,owner->weak_from_this().lock()),"MTP requires its actual shared residency owner");
    // Strong-create ownership is checked by the existing residency issuer.
    (void)owner->owner_inventory(owner);
}
}
struct HipMtpResources::Impl {
    std::shared_ptr<HipRuntimeAllocator> allocator;
    std::shared_ptr<const HipModelResidency> sidecar,target;
    std::shared_ptr<const HipSetupQuiescenceGuard> guard;
    ModelManifest sidecar_manifest,target_manifest;
    ModelRuntimeBinding sidecar_runtime,target_runtime;
    std::optional<const ModelContract> model;
    std::optional<const CompiledMtpDraftPlan> plan;
    MtpDraftRuntimeBinding binding;
    MtpSharedTargetBinding shared;
    std::optional<const MtpNativeTensorViews> tensors;
    std::optional<const HipMtpBudget> budget;
    HipMtpReservation reservation;
    std::shared_ptr<HipQsaCommittedPool> qsa_pool;
    IndexedAttentionWorkspaceLayout attention;
    std::array<MtpDeviceView,buffer_count> views{};
    MtpDraftStateMemoryBinding state;
    std::uintptr_t pool=0;
    std::uint64_t pool_generation=0,identity=0;
    HipRuntimeAllocationDescription pool_lease;
    hipStream_t stream=nullptr;
    bool pending=false,failed=false,deferred=false;
    std::array<std::uint64_t,3> deferred_counts{};
    mutable bool validation_entry=false;
    const RuntimeTensorBinding* sidecar_data=nullptr;
    const RuntimeTensorBinding* target_data=nullptr;
    std::size_t sidecar_size=0,target_size=0;
    std::uint64_t sidecar_generation=0,target_generation=0;
    std::array<std::pair<const RuntimeTensorBinding*,RuntimeTensorBinding>,34> live_roles{};
    ~Impl() {
        if(!guard) return;
        guard->recover_or_fail_stop();
        if(stream && hipStreamDestroy(stream)!=hipSuccess) std::terminate();
        if(pool) allocator->free_device(pool,reservation.device_pool_bytes);
    }
    MtpTensorView tensor(std::string_view suffix) const {
        std::optional<std::uint32_t> selected;
        for(std::uint32_t i=0;i<sidecar_manifest.tensors.size();++i) {
            const auto& t=sidecar_manifest.tensors[i];
            if(std::string_view(t.name).ends_with(suffix)) {
                require(!selected,"MTP native tensor role is ambiguous");selected=i;
            }
        }
        require(selected.has_value(),"MTP native tensor role is missing");
        const auto* runtime=find_runtime_tensor_binding(sidecar_runtime,*selected);
        require(runtime!=nullptr,"MTP native tensor has no actual runtime binding");
        return {&sidecar_manifest.tensors[*selected],runtime};
    }
    MtpNativeTensorViews bind_roles() const {
        return {
            {&target_manifest.tensors[shared.target_embedding_tensor_index],find_runtime_tensor_binding(target_runtime,shared.target_embedding_tensor_index)},
            tensor("nextn.enorm.weight"),tensor("nextn.hnorm.weight"),tensor("nextn.eh_proj.weight"),
            tensor("hc_attn_norm.weight"),tensor("hc_attn_down.weight"),tensor("hc_attn_up.weight"),tensor("hc_attn_inject.weight"),
            tensor("attn_q.weight"),tensor("attn_k.weight"),tensor("attn_v.weight"),tensor("attn_output.weight"),tensor("attn_q_norm.weight"),tensor("attn_k_norm.weight"),
            tensor("indexer.q_proj.weight"),tensor("indexer.k_proj.weight"),tensor("indexer.q_norm.weight"),tensor("indexer.k_norm.weight"),
            tensor("hc_ffn_norm.weight"),tensor("hc_ffn_down.weight"),tensor("hc_ffn_up.weight"),tensor("hc_ffn_inject.weight"),
            tensor("ffn_gate_inp.weight"),tensor("ffn_gate_exps.weight"),tensor("ffn_up_exps.weight"),tensor("ffn_down_exps.weight"),
            tensor("ffn_gate_shexp.weight"),tensor("ffn_up_shexp.weight"),tensor("ffn_down_shexp.weight"),tensor("ffn_gate_inp_shexp.weight"),
            tensor("nextn.hc_head_norm.weight"),tensor("nextn.hc_head_down.weight"),tensor("nextn.hc_head_up.weight"),
            {&target_manifest.tensors[shared.target_output_tensor_index],find_runtime_tensor_binding(target_runtime,shared.target_output_tensor_index)}};
    }
};
HipMtpResources::HipMtpResources():impl_(std::make_unique<Impl>()) {}
HipMtpResources::~HipMtpResources()=default;
void HipMtpResources::attach_qsa_pool(std::shared_ptr<HipQsaCommittedPool> pool) {
    require_idle();require(!impl_->qsa_pool,"MTP QSA pool attaches once before execution");
    if(pool) pool->bind_domain(true,impl_->shared.sidecar_model_content_identity,impl_->plan->plan_identity);
    impl_->qsa_pool=std::move(pool);
}
const std::shared_ptr<HipQsaCommittedPool>& HipMtpResources::qsa_pool() const noexcept {return impl_->qsa_pool;}
std::shared_ptr<HipMtpResources> HipMtpResources::create(const ModelContract& model,const CompiledMtpDraftPlan& plan,
    const ModelManifest& sidecar_manifest,std::shared_ptr<const HipModelResidency> sidecar,
    const ModelManifest& target_manifest,std::shared_ptr<const HipModelResidency> target,
    const MtpSharedTargetBinding& shared,HipMtpBudget budget,std::shared_ptr<HipRuntimeAllocator> allocator,
    const RotaryIdentity& target_rotary) {
    require_rotary_producer_identity(target_rotary,model.rotary_identity());
    require_qwen_rotary_device_payload(model.rotary_identity(),model.qwen_rotary_table.get());
    if(model.qwen_rotary_table) validate_qwen_rotary_reader(*model.qwen_rotary_table,nullptr);
    require(!current_hip_launch_context(),"MTP setup cannot nest an active HIP scope");
    require(model.validate().empty() && model.mtp_full_attention_layers==1U,"MTP native owner requires one explicit sidecar attention layer");
    require(model.hidden_size<=4096U && model.hidden_size%32U==0U && (model.hc_count==2U || model.hc_count==4U) &&
        model.vocab_size<=UINT32_MAX,"MTP native owner geometry exceeds its explicit native profile");
    require(budget.retained_bytes && budget.context_capacity && budget.context_capacity<=model.effective_max_positions() &&
        budget.context_capacity<=UINT32_MAX && budget.maximum_candidate_depth && budget.maximum_candidate_depth<=8U,"MTP native owner requires explicit bounded capacity/depth/budget");
    const auto rounding=[](HipMtpInputRounding r) {return r==HipMtpInputRounding::fp32 || r==HipMtpInputRounding::bf16_rne;};
    require(rounding(budget.normalized_rounding) && rounding(budget.projection_rounding),"MTP native owner requires explicit cast boundaries");
    require(allocator && same_owner(allocator,allocator->weak_from_this().lock()),"MTP requires its actual shared HIP allocator");
    require_residency(sidecar);require_residency(target);
    require(shared.actual_sidecar_owner.get()==sidecar.get() && same_owner(shared.actual_sidecar_owner,sidecar) &&
        shared.actual_target_owner.get()==target.get() && same_owner(shared.actual_target_owner,target),"MTP shared tensor pins are not the actual residency owners");
    validate_mtp_shared_target_binding(model,sidecar_manifest,sidecar->binding(),target_manifest,target->binding(),shared.target_model_content_identity,shared);
    require(shared.sidecar_manifest==&sidecar_manifest && shared.sidecar_runtime==&sidecar->binding() &&
        shared.target_manifest==&target_manifest && shared.target_runtime==&target->binding(),"MTP source descriptor handles must name their actual retained bindings");
    validate_mtp_forward_semantic_contract(model,plan.forward_semantics);
    const auto binding=build_mtp_draft_runtime_binding(sidecar_manifest,sidecar->binding());
    const auto expected_plan=compile_mtp_draft_plan(model,sidecar_manifest,binding,&shared);
    require(plan.plan_identity && plan.plan_identity==recompute_compiled_mtp_draft_plan_identity(plan) &&
        plan.plan_identity==expected_plan.plan_identity &&
        plan.proposal_identity==binding.compatibility_identity && plan.state.state_width==mul(model.hidden_size,model.hc_count),
        "MTP native owner requires its unchanged compiled sidecar plan");
    for(const auto index:binding.tensor_indices) {
        const auto* tensor=find_runtime_tensor_binding(sidecar->binding(),index);
        require(tensor && runtime_tensor_gpu_visible(tensor->storage) && sidecar->covers_device_range(tensor->address,tensor->bytes),
            "MTP sidecar weight is outside its actual retained GPU backing");
    }
    for(const auto index:{shared.target_embedding_tensor_index,shared.target_output_tensor_index}) {
        const auto* tensor=find_runtime_tensor_binding(target->binding(),index);
        require(tensor && runtime_tensor_gpu_visible(tensor->storage) && target->covers_device_range(tensor->address,tensor->bytes),
            "MTP shared target tensor is outside its actual retained GPU backing");
    }
    const auto attention=build_indexed_attention_workspace_layout(model);
    const auto payload=mtp_payload::runtime(model,budget.context_capacity);
    static_assert(buffer_count==std::tuple_size_v<decltype(payload.sizes)>);
    const auto& sizes=payload.sizes;const auto& offsets=payload.offsets;
    const auto bytes=payload.pool,selector=payload.reader,retained=payload.retained;
    require(retained<=budget.retained_bytes,"MTP actual pool plus finite-reader payload exceeds explicit budget");
    auto result=std::shared_ptr<HipMtpResources>(new HipMtpResources);auto& p=*result->impl_;
    p.guard=std::make_shared<HipSetupQuiescenceGuard>();p.allocator=std::move(allocator);p.sidecar=std::move(sidecar);p.target=std::move(target);
    p.sidecar_manifest=sidecar_manifest;p.target_manifest=target_manifest;p.sidecar_runtime=p.sidecar->binding();p.target_runtime=p.target->binding();
    p.model.emplace(model);p.plan.emplace(plan);p.binding=binding;p.shared=shared;p.budget.emplace(budget);p.attention=attention;
    // Deep owned metadata replaces borrowed descriptor handles, without any
    // change to source tensor indices, roles, content or physical identities.
    p.shared.sidecar_manifest=&p.sidecar_manifest;p.shared.target_manifest=&p.target_manifest;
    p.reservation={bytes,selector,retained};p.tensors.emplace(p.bind_roles());
    p.identity=next_uma_owner_view_identity(); // Independent physical organization token, never a target execution ID.
    p.pool=p.allocator->allocate_device(bytes,256U);
    p.pool_lease=p.allocator->describe_allocation(p.pool,bytes,HipRuntimeAllocationSpace::device);
    p.pool_generation=p.pool_lease.lease_identity;
    for(std::size_t i=0;i<buffer_count;++i) if(sizes[i])p.views[i]={p.pool+offsets[i],sizes[i],p.pool_generation};
    const auto state_plan=plan_mtp_draft_state_memory(plan.proposal_identity,plan.state,1U,2U,256U);
    p.state={state_plan.plan_identity,state_plan.proposal_identity,state_plan.state_width,1U,2U,256U,
        p.views[static_cast<std::size_t>(HipMtpBuffer::wide_current)].address,state_plan.total_bytes,p.pool_generation};
    p.guard->run([&] {hip_check_message(hipStreamCreateWithFlags(&p.stream,hipStreamNonBlocking),"MTP stream creation failed");
        hip_check_message(hipMemsetAsync(reinterpret_cast<void*>(p.pool),0,static_cast<std::size_t>(bytes),p.stream),"MTP pool initialization failed");});
    // Save only actual stable retained table addresses after the factory has
    // closed all 34 roles. A later reallocation is rejected before dereference.
    p.sidecar_data=p.sidecar->binding().tensors.data(); p.sidecar_size=p.sidecar->binding().tensors.size();
    p.target_data=p.target->binding().tensors.data(); p.target_size=p.target->binding().tensors.size();
    p.sidecar_generation=p.sidecar->generation(); p.target_generation=p.target->generation();
    const auto& roles=*p.tensors;
    const std::array<MtpTensorView,34> selected{roles.embedding,roles.enorm,roles.hnorm,roles.eh_projection,
        roles.attention_hc_norm,roles.attention_hc_down,roles.attention_hc_up,roles.attention_hc_inject,
        roles.query,roles.key,roles.value,roles.attention_output,roles.query_norm,roles.key_norm,
        roles.index_query,roles.index_key,roles.index_query_norm,roles.index_key_norm,
        roles.mlp_hc_norm,roles.mlp_hc_down,roles.mlp_hc_up,roles.mlp_hc_inject,
        roles.router,roles.expert_gate,roles.expert_up,roles.expert_down,
        roles.shared_gate,roles.shared_up,roles.shared_down,roles.shared_scalar_gate,
        roles.head_norm,roles.head_down,roles.head_up,roles.lm_head};
    for(std::size_t i=0;i<selected.size();++i) {
        const auto& actual=(i==0 || i==selected.size()-1)?p.target->binding():p.sidecar->binding();
        const auto* live=find_runtime_tensor_binding(actual,selected[i].runtime->tensor_index);
        require(live && same(*live,*selected[i].runtime),"MTP admitted live role differs from its immutable descriptor");
        p.live_roles[i]={live,*selected[i].runtime};
    }
    result->validate();return result;
}
MtpDeviceView HipMtpResources::view(HipMtpBuffer kind) const {
    const auto i=static_cast<std::size_t>(kind);require(i<buffer_count,"MTP buffer role is invalid");
    require(kind!=HipMtpBuffer::wide_checkpoint && kind!=HipMtpBuffer::kv_checkpoint &&
        kind!=HipMtpBuffer::raw_index_checkpoint,"MTP draft checkpoint buffer role is retired");
    return impl_->views[i];
}
const ModelContract& HipMtpResources::model() const noexcept {return *impl_->model;}
const CompiledMtpDraftPlan& HipMtpResources::plan() const noexcept {return *impl_->plan;}
const MtpDraftRuntimeBinding& HipMtpResources::binding() const noexcept {return impl_->binding;}
const ModelManifest& HipMtpResources::manifest() const noexcept {return impl_->sidecar_manifest;}
const ModelRuntimeBinding& HipMtpResources::runtime_binding() const noexcept {return impl_->sidecar_runtime;}
const MtpNativeTensorViews& HipMtpResources::tensors() const noexcept {return *impl_->tensors;}
const IndexedAttentionWorkspaceLayout& HipMtpResources::attention_workspace() const noexcept {return impl_->attention;}
std::uint64_t HipMtpResources::context_capacity() const noexcept {return impl_->budget->context_capacity;}
std::uint64_t HipMtpResources::physical_identity() const noexcept {return impl_->identity;}
std::uint64_t HipMtpResources::retained_payload_bytes() const noexcept {return impl_->reservation.device_pool_bytes;}
const HipMtpReservation& HipMtpResources::reservation() const noexcept {return impl_->reservation;}
const HipMtpBudget& HipMtpResources::budget() const noexcept {return *impl_->budget;}
const MtpSharedTargetBinding& HipMtpResources::shared_target() const noexcept {return impl_->shared;}
const MtpDraftStateMemoryBinding& HipMtpResources::state_memory() const noexcept {return impl_->state;}
std::shared_ptr<const HipSetupQuiescenceGuard> HipMtpResources::guard() const noexcept {return impl_->guard;}
void* HipMtpResources::stream() const noexcept {return impl_->stream;}
void HipMtpResources::validate() const {
    const auto& p=*impl_;require(!p.failed && p.guard && p.pool && p.stream,"MTP owner is failed or uninitialized");p.guard->require_original();
    validate_mtp_shared_target_binding(*p.model,p.sidecar_manifest,p.sidecar->binding(),p.target_manifest,p.target->binding(),p.shared.target_model_content_identity,p.shared);
    for(const auto& pair: {std::pair{&p.sidecar_runtime,&p.sidecar->binding()},std::pair{&p.target_runtime,&p.target->binding()}}) {
        require(pair.first->tensors.size()==pair.second->tensors.size(),"MTP retained weight table changed");
        for(std::size_t i=0;i<pair.first->tensors.size();++i) require(same(pair.first->tensors[i],pair.second->tensors[i]),"MTP retained weight allocation changed");
    }
    require(p.allocator->describe_allocation(p.pool,p.reservation.device_pool_bytes,HipRuntimeAllocationSpace::device)==p.pool_lease,
        "MTP original allocator lease changed");
}
HipMtpResources::NativeValidationScope::~NativeValidationScope() {
    if(owner_) owner_->impl_->validation_entry=false;
}
HipMtpResources::NativeValidationScope::NativeValidationScope(NativeValidationScope&& other) noexcept
    :owner_(std::exchange(other.owner_,nullptr)) {}
HipMtpResources::NativeValidationScope HipMtpResources::admit_native_entry(bool carry_deferred) const {
    if(impl_->deferred && !carry_deferred) settle_deferred();
    require(!impl_->validation_entry && (!impl_->pending || (carry_deferred && impl_->deferred)),"MTP native admission cannot nest or borrow queued work");
    validate();
    impl_->validation_entry=true;
    return NativeValidationScope(shared_from_this());
}
void HipMtpResources::validate_stage() const {
    const auto& p=*impl_;
    if(!p.validation_entry) {validate();return;}
    require(!p.failed && p.guard && p.pool && p.stream,"MTP owner is failed or uninitialized");
    int device=-1;
    hip_check_message(hipGetDevice(&device),"MTP current device query failed");
    require(device==p.guard->device_index(),"MTP native stage changed its admitted device");
    const auto& sidecar=p.sidecar->binding(); const auto& target=p.target->binding();
    require(p.sidecar->generation()==p.sidecar_generation && p.target->generation()==p.target_generation &&
        sidecar.tensors.data()==p.sidecar_data && sidecar.tensors.size()==p.sidecar_size &&
        target.tensors.data()==p.target_data && target.tensors.size()==p.target_size &&
        sidecar.manifest_layout_identity==p.sidecar_runtime.manifest_layout_identity &&
        target.manifest_layout_identity==p.target_runtime.manifest_layout_identity &&
        sidecar.model_content_identity==p.sidecar_runtime.model_content_identity &&
        target.model_content_identity==p.target_runtime.model_content_identity,
        "MTP native stage retained weight owner/table changed");
    for(const auto& role:p.live_roles)
        require(role.first && same(*role.first,role.second),"MTP native stage weight descriptor/generation changed");
    require(p.allocator->describe_allocation(p.pool,p.reservation.device_pool_bytes,HipRuntimeAllocationSpace::device)==p.pool_lease,
        "MTP original allocator lease changed");
}
void HipMtpResources::require_idle(bool carry_deferred) const {
    if(impl_->deferred && !carry_deferred) settle_deferred();
    validate_stage();require(!impl_->pending || (carry_deferred && impl_->deferred),"MTP native queue is already pending");
}
void HipMtpResources::mark_submitted() {
    validate_stage();
    if(impl_->deferred) {impl_->deferred=false;++impl_->deferred_counts[1];}
    impl_->pending=true;
}
void HipMtpResources::defer_completion() {
    validate_stage();require(impl_->pending && !impl_->deferred,"MTP deferred completion requires submitted native work");
    impl_->deferred=true;++impl_->deferred_counts[0];
}
bool HipMtpResources::completion_deferred() const noexcept {return impl_->deferred;}
// Every HipMtpResources is a heap object owned through a non-const shared_ptr (create()).
void HipMtpResources::settle_deferred() const {
    if(!impl_->deferred) return;
    ++impl_->deferred_counts[2];const_cast<HipMtpResources&>(*this).complete();
}
std::array<std::uint64_t,3> HipMtpResources::deferred_counts() const noexcept {return impl_->deferred_counts;}
void HipMtpResources::complete() {
    auto& p=*impl_;validate_stage();require(p.pending,"MTP completion requires submitted native work");
    try {
        std::array<std::uint32_t,2> status{};const auto source=view(HipMtpBuffer::status);
        hip_check_message(hipMemcpyAsync(status.data(),reinterpret_cast<const void*>(source.address),8U,hipMemcpyDeviceToHost,p.stream),"MTP status transfer failed");
        hip_check_message(hipStreamSynchronize(p.stream),"MTP native completion failed");p.guard->require_original();
        require(!status[0] && !status[1],"MTP native first-error status is nonzero");p.pending=false;p.deferred=false;
    } catch(...) {fail();throw;}
}
void HipMtpResources::fail() noexcept {impl_->failed=true;if(impl_->guard) impl_->guard->recover_or_fail_stop();impl_->pending=false;impl_->deferred=false;}
bool HipMtpResources::failed() const noexcept {return impl_->failed;}
} // namespace strix
