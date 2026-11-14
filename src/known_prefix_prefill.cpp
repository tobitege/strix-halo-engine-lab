#include "strix/known_prefix_prefill.hpp"
#include "strix/compact_stage_images.hpp"
#include "strix/known_prefill_limits.hpp"
#include "strix/ctx_exact_memory.hpp"
#include "strix/indexed_attention_workspace.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace strix {
namespace {
void require(bool condition,const char* message) {if(!condition) throw std::invalid_argument(message);}
std::uint64_t add(std::uint64_t a,std::uint64_t b) {
    if(b>UINT64_MAX-a) throw std::overflow_error("known prefill storage overflow");
    return a+b;
}
std::uint64_t multiply(std::uint64_t a,std::uint64_t b) {
    if(b&&a>UINT64_MAX/b) throw std::overflow_error("known prefill storage overflow");
    return a*b;
}
void hash(std::uint64_t& h,std::uint64_t value) {
    for(unsigned shift=0;shift<64;shift+=8) {h^=(value>>shift)&255U;h*=1099511628211ULL;}
}
void hash_arena(std::uint64_t& h,const ExecutionArenaPlan& arena) {
    hash(h,arena.alignment);hash(h,arena.total_bytes);hash(h,arena.naive_nonaliased_bytes);hash(h,arena.allocations.size());
    for(const auto& a:arena.allocations) {
        hash(h,static_cast<std::uint64_t>(a.use.buffer.buffer_class));hash(h,a.use.buffer.slot);hash(h,a.use.generation);
        hash(h,a.first_step);hash(h,a.last_step);hash(h,a.offset);hash(h,a.bytes);
    }
}
std::uint64_t metadata_payload(std::uint32_t rows,std::size_t allocations) {
    // Declared arena-vector payload, not allocator overhead/RSS. The closed
    // selection plan remains bounded at eight rows independently of T.
    return add(multiply(rows,sizeof(ExecutionArenaPlan)),add(multiply(allocations,sizeof(std::uint64_t)),
        multiply(multiply(static_cast<std::uint64_t>(rows)+1U,allocations),sizeof(ExecutionArenaAllocation))));
}
std::pair<std::uint64_t,std::uint64_t> identities(const CompiledKnownPrefixPrefill& plan) {
    std::uint64_t numerical=14695981039346656037ULL;
    for(char c:std::string_view("known-prefix-prefill-step-major-tile8-v1")) {numerical^=static_cast<unsigned char>(c);numerical*=1099511628211ULL;}
    hash(numerical,plan.tile8.window_kernel_plan_identity);hash(numerical,plan.logical_rows);hash(numerical,8U);
    hash(numerical,static_cast<std::uint64_t>(plan.output_policy));
    for(auto row:plan.logit_row_indices) hash(numerical,row);
    auto execution=numerical;hash(execution,plan.tile8.window_execution_identity);hash_arena(execution,plan.arena.physical);
    if (plan.qsa_score_ring_rows) {
        hash(execution,plan.qsa_score_ring_rows);hash(execution,plan.qsa_score_ring_workspace_bytes);
    }
    for(auto stride:plan.arena.row_stride) hash(execution,stride);
    // Row r's view is row 0's plus r strides (validate_compiled_known_prefix_prefill checks every row before any
    // block runs), so physical arena, strides, T and row 0 determine all T views. Hashing each of the T views
    // byte-serially cost about 40 ms per evaluation at T = 1024 (three evaluations per prompt).
    hash(execution,plan.arena.row_view.size());
    if(!plan.arena.row_view.empty()) hash_arena(execution,plan.arena.row_view.front());
    require(execution&&numerical&&execution!=plan.tile8.window_execution_identity&&numerical!=plan.tile8.window_kernel_plan_identity,
        "known prefill identity is zero or collides with the closed window");return {execution,numerical};
}
}
CompiledKnownPrefixPrefill compile_known_prefix_prefill(const ModelContract& model,const CompiledSerialExecution& serial,
    const ModelManifest& manifest,const ExecutionTensorAccessPlan& access,const KnownPrefillRequest& request) {
    require(request.logical_rows>=16U&&request.logical_rows<=known_prefill_max_rows,("known prefill rows must be within "+known_prefill_rows_range()).c_str());
    require(request.workspace_budget_bytes&&request.host_metadata_payload_budget_bytes,"known prefill requires explicit workspace and host metadata budgets");
    validate_known_window_output_policy(request.output_policy);
    const auto payload=metadata_payload(request.logical_rows,serial.arena.allocations.size());
    if(payload>request.host_metadata_payload_budget_bytes)
        throw KnownPrefillBudgetExceeded("known prefill host metadata payload exceeds its budget: T = "+std::to_string(request.logical_rows)+
            " needs "+std::to_string(payload)+" bytes (--known-prefix-host-metadata-budget-bytes "+
            std::to_string(request.host_metadata_payload_budget_bytes)+")");
    CompiledKnownPrefixPrefill plan;plan.logical_rows=request.logical_rows;plan.output_policy=request.output_policy;
    plan.workspace_budget_bytes=request.workspace_budget_bytes;plan.host_metadata_payload_budget_bytes=request.host_metadata_payload_budget_bytes;
    plan.host_metadata_payload_bytes=payload;
    KnownWindowRequest tile;tile.logical_rows=8;tile.r2_shared_expert=request.r2_shared_expert;
    tile.r2_gdn=request.r2_gdn;tile.r2_gdn_f32_gates=request.r2_gdn_f32_gates;tile.routed_expert_union=request.routed_expert_union;
    tile.router_rows=request.router_rows;tile.combine_inject_rows=request.combine_inject_rows;
    tile.rows_hyper_mix=request.rows_hyper_mix;tile.qsa_rows=request.qsa_rows;tile.rows_ple=request.rows_ple;
    require(!request.lm_head_rows||request.output_policy==KnownWindowOutputPolicy::all_rows,
        "known prefill multi-row LM head requires the all-rows output policy");
    tile.lm_head_rows=request.lm_head_rows;
    // Preserve complete tile admission even when this T request omits heads.
    plan.tile8=compile_known_prefix_window(model,serial,manifest,access,tile);
    require(known_window_has_promoted_steps(plan.tile8),"known prefill requires an actual admitted multi-row projection entry");
    if(request.exact_memory && request.full_qsa_score_ring_entry && request.qsa_rows && plan.tile8.qsa_rows_steps==model.full_attention_layers && request.logical_rows%8U==0 && !serial.device_graph_inputs &&
        model.indexer_compress_ratio==4 && request.output_policy!=KnownWindowOutputPolicy::all_rows && qsa_score_ring_enabled()) {
        plan.external_qsa_score_bytes=build_indexed_attention_workspace_layout(model).block_scores_bytes;
        // Effective positions (YaRN extends them), as the block-score layout and the serial validation use.
        const QsaScoreRingGeometry geometry(model.effective_max_positions(),model.indexer_compress_ratio,
            request.qsa_score_ring_rows ? request.qsa_score_ring_rows :
                qsa_score_ring_rows_override(std::getenv("STRIX_QSA_SCORE_RING_ROWS")));
        plan.qsa_score_ring_rows=geometry.rows;
        plan.qsa_score_ring_workspace_bytes=geometry.total_bytes();
    }
    // Only the full-T QSA owner may bind the smaller image. Closed windows,
    // decode, MTP, capture and optional-entry callers retain their FP32 ABI.
    if (request.compact_stage_images && model.bytes_fp32==sizeof(float) && plan.external_qsa_score_bytes && serial.arena.alignment==256U && compact_stage_images_enabled()) {
        const auto fp32 = build_indexed_attention_workspace_layout(model,256,false,1,true);
        const auto compact = build_indexed_attention_workspace_layout(model,256,false,1,true,true);
        plan.compact_qsa_output_savings = fp32.total_bytes - compact.total_bytes;
    }
    static const char* const dump=std::getenv("STRIX_QRTX_STEP_DUMP"); // R2-13: read once
    plan.terminal_logits_arena=request.exact_memory && !serial.device_graph_inputs &&
        request.output_policy!=KnownWindowOutputPolicy::all_rows && (!dump || !*dump) &&
        ctx_exact_memory_enabled("STRIX_TERMINAL_LOGITS_ARENA");
    // Every stripe/interval scales uniformly from the closed eight-row arena.
    // Its offsets are multiples of eight alignments, including odd T shapes.
    require(plan.tile8.arena.physical.total_bytes%8U==0U,"known prefill tile arena cannot be uniformly scaled");
    if(!plan.external_qsa_score_bytes && !plan.terminal_logits_arena && multiply(plan.tile8.arena.physical.total_bytes/8U,request.logical_rows)>request.workspace_budget_bytes)
        throw KnownPrefillBudgetExceeded("known prefill workspace exceeds its explicit arena budget");
    plan.arena=build_known_prefill_arena(serial.arena,serial.arena.alignment,request.logical_rows,plan.external_qsa_score_bytes,plan.terminal_logits_arena,true,plan.compact_qsa_output_savings);
    if(plan.arena.physical.total_bytes>request.workspace_budget_bytes)
        throw KnownPrefillBudgetExceeded("known prefill workspace exceeds its explicit arena budget");
    for(std::uint32_t row=0;row<request.logical_rows;++row)
        if(known_window_produces_logits(request.output_policy,row,request.logical_rows)) plan.logit_row_indices.push_back(row);
    const auto [execution,numerical]=identities(plan);plan.prefill_execution_identity=execution;plan.prefill_kernel_plan_identity=numerical;
    return plan;
}
void validate_compiled_known_prefix_prefill(const CompiledKnownPrefixPrefill& plan,const CompiledSerialExecution& serial) {
    require(plan.logical_rows>=16U&&plan.logical_rows<=known_prefill_max_rows,("known prefill rows must be within "+known_prefill_rows_range()).c_str());
    validate_compiled_known_prefix_window(plan.tile8,serial);
    require(plan.tile8.logical_rows==8U&&plan.tile8.output_policy==KnownWindowOutputPolicy::all_rows&&
        known_window_has_promoted_steps(plan.tile8),"known prefill selection must retain its complete closed tile-eight contract");
    validate_known_window_output_policy(plan.output_policy);
    require(plan.workspace_budget_bytes&&plan.arena.physical.total_bytes<=plan.workspace_budget_bytes,
        "known prefill workspace exceeds its explicit arena budget");
    require(plan.host_metadata_payload_bytes==metadata_payload(plan.logical_rows,serial.arena.allocations.size())&&
        plan.host_metadata_payload_budget_bytes&&plan.host_metadata_payload_bytes<=plan.host_metadata_payload_budget_bytes,
        "known prefill host metadata payload budget differs");
    if(plan.external_qsa_score_bytes) {
        require(qsa_score_ring_rows_valid(plan.qsa_score_ring_rows),"known prefill score ring rows differ");
        const QsaScoreRingGeometry geometry(serial.max_position_embeddings,4U,plan.qsa_score_ring_rows);
        require(plan.logical_rows%8U==0 && plan.output_policy!=KnownWindowOutputPolicy::all_rows && !serial.device_graph_inputs &&
            plan.external_qsa_score_bytes==geometry.blocks*4 && plan.qsa_score_ring_workspace_bytes==geometry.total_bytes(),
            "known prefill score ring geometry differs");
    } else require(!plan.qsa_score_ring_workspace_bytes && !plan.qsa_score_ring_rows,"known prefill unexpected score ring storage");
    require(!plan.terminal_logits_arena || (plan.output_policy!=KnownWindowOutputPolicy::all_rows && !serial.device_graph_inputs),
        "known prefill terminal arena policy differs");
    require(!plan.compact_qsa_output_savings || (plan.external_qsa_score_bytes &&
        plan.compact_qsa_output_savings % serial.arena.alignment == 0), "compact QSA output admission differs");
    const auto expected=build_known_prefill_arena(serial.arena,serial.arena.alignment,plan.logical_rows,
        plan.external_qsa_score_bytes,plan.terminal_logits_arena,false,plan.compact_qsa_output_savings);
    const auto same=[](const ExecutionArenaPlan& a,const ExecutionArenaPlan& b) {
        if(a.alignment!=b.alignment || a.total_bytes!=b.total_bytes || a.naive_nonaliased_bytes!=b.naive_nonaliased_bytes ||
            a.allocations.size()!=b.allocations.size()) return false;
        for(std::size_t i=0;i<a.allocations.size();++i) {
            const auto& x=a.allocations[i];const auto& y=b.allocations[i];
            if(x.use!=y.use || x.first_step!=y.first_step || x.last_step!=y.last_step || x.offset!=y.offset || x.bytes!=y.bytes) return false;
        }
        return true;
    };
    require(same(plan.arena.physical,expected.physical) && plan.arena.row_stride==expected.row_stride &&
        plan.arena.row_view.size()==plan.logical_rows,"known prefill physical arena or stripe differs");
    auto view=expected.physical;
    std::vector<std::uint64_t> row_bytes;
    for(const auto& a:expected.physical.allocations) {
        const auto* original=find_arena_allocation(serial.arena,a.use);
        require(original!=nullptr,"known prefill lost source allocation");
        auto bytes=original->bytes;
        if(plan.external_qsa_score_bytes && a.use.buffer.buffer_class==ExecutionBufferClass::indexed_attention_workspace)
            bytes-=(plan.external_qsa_score_bytes+serial.arena.alignment-1)/serial.arena.alignment*serial.arena.alignment + plan.compact_qsa_output_savings;
        row_bytes.push_back(bytes);
    }
    for(std::uint32_t r=0;r<plan.logical_rows;++r) {
        for(std::size_t i=0;i<view.allocations.size();++i) {
            view.allocations[i].offset=add(expected.physical.allocations[i].offset,multiply(r,expected.row_stride[i]));
            view.allocations[i].bytes=plan.terminal_logits_arena && execution_use_is_logits(view.allocations[i].use) && r+1U!=plan.logical_rows?0U:row_bytes[i];
        }
        require(same(plan.arena.row_view[r],view),"known prefill row arena differs");
    }
    std::vector<std::uint32_t> logits;
    for(std::uint32_t row=0;row<plan.logical_rows;++row)
        if(known_window_produces_logits(plan.output_policy,row,plan.logical_rows)) logits.push_back(row);
    require(logits==plan.logit_row_indices,"known prefill output row mapping differs");
    const auto [execution,numerical]=identities(plan);
    require(execution==plan.prefill_execution_identity&&numerical==plan.prefill_kernel_plan_identity&&
        execution!=serial.execution_identity,"known prefill execution or numerical identity differs");
}
} // namespace strix
