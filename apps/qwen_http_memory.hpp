#pragma once
#include "http_slot_capacity.hpp"
#include <cstdlib>
#include "qwen_prompt_organization.hpp"
#include "strix/indexed_attention_workspace.hpp"
#include "strix/indexed_attention_state.hpp"
#include "strix/q51_known_rows.hpp"
#include "strix/ple_lookup.hpp"

namespace strix::decode_cli {
// Tap bytes of one tiled MTP prompt: T rows per buffer plus the held row. The prompt allocates two buffers only
// when STRIX_PREFILL_PIPE bit 1 is effective; the default bound charges two for every slot.
inline std::uint64_t http_mtp_prompt_tap_bytes(std::uint64_t rows, std::uint64_t width,
    bool exact, bool double_buffered) {
    const auto buffers = exact && !double_buffered ? UINT64_C(1) : UINT64_C(2);
    return prefill_checked_mul(prefill_checked_add(prefill_checked_mul(buffers,rows),1U),width);
}
inline bool http_mtp_prompt_tap_double(bool gsq) {
    const char* pipe=std::getenv("STRIX_PREFILL_PIPE");
    return pipe && (std::strtoul(pipe,nullptr,0)&1U) && !qpf2k::suppressed(qpf2k::Item::pipe,gsq);
}
// Geometry bounds for owners built by the MTP bridge/window factory. The primary
// stack and shared tiled arenas are excluded. No option budget is an allocation.
inline HttpServingMemory http_mtp_memory(const Options& o, const ModelContract& model,
    const CompiledSerialExecution& compiled, const ModelManifest& manifest,
    const ExecutionTensorAccessPlan& access, std::uint64_t sidecar_host,
    std::uint64_t sidecar_device, std::uint64_t golden_bytes) {
    if (!o.mtp.enabled) return {};
    const auto add = prefill_checked_add;
    const auto mul = prefill_checked_mul;
    const auto width = mul(mul(model.hidden_size,model.hc_count),2U);
    const auto selector = add(mul((model.vocab_size+255U)/256U,16U),64U);
    const auto buffers = build_serial_execution_buffer_plan(model,{});
    const auto attention = build_indexed_attention_workspace_layout(model);
    auto mtp_model = model; mtp_model.attention_kv_storage = AttentionKvStorage::bf16;
    const auto predictor = build_indexed_attention_state_layout(mtp_model);
    const auto input = mul(mul(model.hidden_size,model.hc_count+1U),4U);
    std::uint64_t pool = 0;
    for (const auto bytes : {width,width,mul(model.hidden_size,2U),input,input,width,width,
            buffers.hidden_bytes,buffers.hidden_bytes,buffers.hidden_bytes,buffers.injection_weight_bytes,
            buffers.route_index_bytes,buffers.route_weight_bytes,buffers.hyper_workspace_bytes,
            attention.total_bytes,buffers.mlp_intermediate_bytes,mul(model.num_experts,4U),buffers.logits_bytes,
            mul(o.context_tokens,predictor.kv_row_bytes),mul(o.context_tokens,predictor.raw_index_row_bytes),UINT64_C(8)})
        if (bytes) pool = add(add(pool,255U)/256U*256U,bytes);
    pool = add(add(pool,255U),selector);
    if (pool > o.mtp.runtime_bytes) throw std::runtime_error("one-slot:native pool exceeds runtime budget");
    SerialSessionStateLayoutRequest request; request.context_tokens=o.context_tokens;
    request.recurrent_storage=o.gdn_recurrent_storage;
    const auto state=build_serial_session_state_layout(model,request);
    const auto tail=o.known_prefix_tail_capture_rows ? o.known_prefix_tail_capture_rows :
        o.known_prefix_tail_rows.empty() ? o.known_prefix_prefill_rows : o.known_prefix_tail_rows.back();
    const auto rows=std::max({static_cast<std::uint64_t>(mtp_calibration_rows(o))+1U,
        static_cast<std::uint64_t>(tail)+1U,static_cast<std::uint64_t>(o.mtp.candidate_depth)+3U});
    auto capture=add(add(mul(rows,add(width,255U)),selector),
        add(263U,add(state.host_total_bytes,state.host_alignment-1U)));
    if (capture > o.mtp.target_capture_bytes) throw std::runtime_error("one-slot:target capture exceeds budget");
    auto slot=add(pool,capture);
    slot=add(slot,qsa_mtp_pool_bytes(o,model,qpf2k::has_gsq_gate(manifest)));
    // Prompt tap may be double buffered. Charge both independently of the
    // environment so inspection and startup have the same safe bound.
    // STRIX_SERVE_SLOT_EXACT=1: charge the tap buffers the prompt really allocates.
    slot=add(slot,http_mtp_prompt_tap_bytes(o.known_prefix_prefill_rows,width,strix::serve::slot_exact(),
        http_mtp_prompt_tap_double(qpf2k::has_gsq_gate(manifest))));
    const bool subset=!o.mtp.draft_vocab.empty() || o.mtp.draft_context_vocab;
    const auto subset_row=subset ? mul(model.hidden_size/32U,o.mtp.draft_q4?18U:34U) : 0U;
    slot=add(slot,mul(model.vocab_size,add(subset_row,77U)));
    slot=add(slot,add(selector,4U*(11U*640U+1U)+2U*4U*10240U*4U));
    if (o.mtp.window_rows) {
        const auto sizes=o.mtp.window_rows_set.empty()?std::vector<std::uint32_t>{o.mtp.window_rows}:o.mtp.window_rows_set;
        for (const auto n : sizes) {
            const auto plan=compile_known_prefix_window(model,compiled,manifest,access,
                {true,true,true,n,o.mtp.window_expert_union,KnownWindowOutputPolicy::all_rows,
                 o.mtp.window_hyper_mix,o.mtp.window_qsa_rows,o.mtp.window_prefix,o.mtp.window_prefix,
                 o.mtp.window_prefix,o.mtp.window_prefix || o.known_window_router_rows,
                 o.mtp.window_prefix || o.known_window_router_rows});
            slot=add(slot,add(plan.arena.physical.total_bytes,8U));
            if (o.mtp.window_expert_union) slot=add(slot,q51_known_rows::required_scratch_bytes(n,
                static_cast<std::uint32_t>(model.num_experts),static_cast<std::uint32_t>(model.num_experts_per_token)));
            // Four capture pools: hidden rows, selectors, two lineage histories,
            // binding snapshots and bounded per-row/lease metadata (rounded up).
            auto payload=add(add(mul(n,width),255U),mul(n,selector));
            payload=add(payload,mul(o.mtp.window_history_intervals,2U*sizeof(ArithmeticRowInterval)));
            payload=add(payload,add(mul(n,256U),add(4096U,mul(manifest.tensors.size(),sizeof(RuntimeTensorBinding)))));
            if (payload>o.mtp.window_capture_bytes) throw std::runtime_error("one-slot:window capture exceeds budget");
            slot=add(slot,mul(4U,payload));
            slot=add(slot,mul(n,add(selector,64U))); // greedy rows
            // Each owner creates its banks before sharing the primary's banks.
            const auto align = [&](std::uint64_t bytes) { return mul(add(bytes,255U)/256U,256U); };
            const auto keys=mul(model.gdn_key_heads,model.gdn_key_head_dim);
            const auto values=mul(model.gdn_value_heads,model.gdn_key_head_dim);
            const auto saved=align(add(align(mul(add(mul(2U,keys),values),2U)),
                add(mul(add(keys,values),4U),mul(model.gdn_value_heads,8U))));
            if (o.mtp.window_prefix) slot=add(slot,add(mul(mul(16U,saved),model.gdn_layers),8U));
            const auto conv=mul(mul(width,model.ple_conv_kernel_size-1U),model.ngram_size);
            slot=add(slot,add(mul(n,add(conv,mul(mul(model.hidden_size,model.hc_count+1U),4U))),
                mul(mul(2U,n),PleHashContract::maximum_heads*512U)));
            slot=add(slot,8U<<20U); // known-window shared-expert workspace
            if (!o.mtp.window_prefix) // full checkpoint bounds include alignment
                slot=add(slot,add(add(state.device_total_bytes,state.host_total_bytes),65536U));
        }
    }
    // Reader holds encoded + decoded calibration data; replay retains its own
    // rows/receipts. Four file payloads plus bounded row metadata cover the peak.
    const auto calibration=add(mul(4U,golden_bytes),mul(mtp_calibration_rows(o)+1U,4096U));
    return {add(sidecar_host,calibration),sidecar_device,slot,slot};
}
} // namespace strix::decode_cli
