#pragma once
#include "qwen_effective_config.hpp"
#include "strix/hip_launcher_catalog.hpp"
#include "strix/known_prefix_prefill.hpp"
#include "strix/session_executor_handoff.hpp"
#include "strix/model_tensor_index.hpp"
#include "strix/execution_tensor_access.hpp"
#include "strix/qsa_input_prefetch.hpp"
#include "strix/shared_expert_prefetch.hpp"
#include "strix/lane_chain_hyper_output.hpp"

namespace strix::decode_cli {
// Shared CLI/admission tensor checks, before any device allocation.
inline void validate_prompt_tensor_contract(const Options& options, const ModelManifest& manifest,
    const CompiledSerialExecution& compiled, const ExecutionTensorAccessPlan& access_plan) {
    if (numerical_profile_hc_splitk(options.numerical_profile)) {
        const auto require_hc_tensor = [&](std::uint32_t step, std::uint32_t slot,
            std::uint32_t type, std::vector<std::uint64_t> dimensions, std::uint64_t bytes) {
            const auto* access = strix::find_execution_tensor_access(access_plan, step, slot);
            if (!access || access->tensor_indices.size() != 1U || access->tensor_indices[0] >= manifest.tensors.size())
                throw std::invalid_argument("HC numerical profile requires complete canonical tensor access before allocation");
            const auto& tensor = manifest.tensors[access->tensor_indices[0]];
            if (tensor.ggml_type != type || tensor.dimensions != dimensions || tensor.byte_size != bytes)
                throw std::invalid_argument("HC numerical profile tensor format, shape or extent differs before allocation");
        };
        for (const auto& step : compiled.execution.steps) {
            if (step.op != strix::SerialOp::attn_hyper_mix && step.op != strix::SerialOp::mlp_hyper_mix &&
                step.op != strix::SerialOp::final_hyper_mix) continue;
            require_hc_tensor(step.id, 0U, 0U, {10240U}, 40960U);
            require_hc_tensor(step.id, 1U, 8U, {10240U,320U}, 3481600U);
            require_hc_tensor(step.id, 2U, 8U, {320U,10240U}, 3481600U);
            if (step.op != strix::SerialOp::final_hyper_mix)
                require_hc_tensor(step.id, 3U, 0U, {10240U,4U}, 163840U);
        }
    }
}
struct PromptCompilerRequests {
    ExecutionBindingRequest binding;
    ExecutionBufferRequest buffers;
    HipLauncherCatalogConfig catalog;
};
// Host plan construction only: no launcher objects, registration or HIP calls.
inline PromptCompilerRequests prompt_compiler_requests(const Options& options,
    const ModelContract& model, const ModelManifest& manifest) {
    const auto hc_splitk_f32 = numerical_profile_hc_splitk(options.numerical_profile);
    const auto hc_deferred_norm = numerical_profile_hc_deferred_norm(options.numerical_profile);
    const auto a8_down_q51 = numerical_profile_q51_a8_down(options.numerical_profile);
    const auto a8_down_q80 = numerical_profile_q80_a8_down(options.numerical_profile);
    const auto q5k_a8_gateup = numerical_profile_q5k_a8_gateup(options.numerical_profile);
    const auto dense_a8 = numerical_profile_dense_a8(options.numerical_profile);
    // The same plans as the decode builder (hip_decode_token_main.cpp): layer/step lists only for an enabled
    // organization. A base without --prefetch-q8-qsa-input, --prefetch-q8-shared-expert or --lane-chain-hyper-output
    // (an explicit organization) otherwise failed "... requires opt-in" in the serving memory check (issue 758).
    const auto qsa_prefetch_plan = options.prefetch_q8_qsa_input ?
        qsa_input_prefetch::build_plan(model, manifest) : qsa_input_prefetch::Plan{};
    const auto shared_prefetch_plan = options.prefetch_q8_shared_expert ?
        shared_expert_prefetch::build_plan(model, manifest) : shared_expert_prefetch::Plan{};
    const auto wave_router_admission = WaveRouterManifestAdmission::create(manifest);
    HipModelResidencyPolicy policy;
    policy.mode = options.residency;
    const auto lane_chain_steps = options.lane_chain_hyper_output ?
        lane_chain_hyper_output_steps(model, manifest, policy) : std::vector<std::uint32_t>{};
    strix::ExecutionBindingRequest
        binding_request;
    binding_request.logical_rows = 1;
    binding_request.request_concurrency = 1;
    binding_request.minimum_qualification =
        strix::kernels::
            QualificationStatus::
                experimental;

    constexpr bool graph_mode = false;
    binding_request.require_graph_safe = graph_mode;
    binding_request.device_graph_inputs = graph_mode;
    strix::HipLauncherCatalogConfig catalog_config;
    catalog_config.activation = strix::HipActivationStorage::bf16;
    catalog_config.enable_device_graph_inputs = graph_mode;
    catalog_config.parallel_hyper_mix = options.parallel_hyper_mix;
    catalog_config.staged_hyper_projections = options.staged_hyper_projections;
    catalog_config.fused_staged_hyper_mix = options.fused_staged_hyper_mix;
    catalog_config.typed_hyper_projection = options.typed_hyper_projection;
    catalog_config.prefetch_typed_hyper_rows = options.prefetch_typed_hyper_rows;
    catalog_config.prefetch_hc_projection_next_tile = options.prefetch_hc_projection_next_tile;
    catalog_config.numerical_profile = options.numerical_profile;
    catalog_config.splitk_hyper_projection = hc_splitk_f32;
    catalog_config.splitk_deferred_norm = hc_deferred_norm;
    catalog_config.splitk_hyper_projection_fp64 = false;
    catalog_config.arithmetic_stats = nullptr;
    binding_request.prefetch_hc_projection_next_tile = options.prefetch_hc_projection_next_tile;
    catalog_config.prefetch_q8_hyper_output = options.prefetch_q8_hyper_output;
    catalog_config.lane_chain_hyper_output=options.lane_chain_hyper_output;
    binding_request.lane_chain_hyper_output=options.lane_chain_hyper_output;
    binding_request.lane_chain_hyper_output_eligible_steps=lane_chain_steps;
    catalog_config.virtual_q8_lm_head = options.virtual_q8_lm_head;
    catalog_config.prefetch_q8_lm_head = options.prefetch_q8_lm_head;
    catalog_config.wave8_q8_lm_head = options.wave8_q8_lm_head;
    catalog_config.wave8_q8_gdn_input = options.wave8_q8_gdn_input;
    catalog_config.gdn_recurrent_layout = binding_request.gdn_recurrent_layout = options.gdn_recurrent_layout;
    catalog_config.gdn_recurrent_storage = binding_request.gdn_recurrent_storage = options.gdn_recurrent_storage;
    catalog_config.wave8_q8_gdn_output = options.wave8_q8_gdn_output;
    catalog_config.wave8_q8_qsa_output = options.wave8_q8_qsa_output;
    binding_request.prefetch_q8_lm_head = options.prefetch_q8_lm_head;
    binding_request.wave8_q8_lm_head = options.wave8_q8_lm_head;
    binding_request.wave8_q8_gdn_input = options.wave8_q8_gdn_input;
    binding_request.wave8_q8_gdn_output = options.wave8_q8_gdn_output;
    binding_request.wave8_q8_qsa_output = options.wave8_q8_qsa_output;
    catalog_config.batched_moe_activation = options.batched_moe_activation;
    binding_request.batched_moe_activation = options.batched_moe_activation;
    catalog_config.q4k_a8g32_gate_up = options.q4k_a8g32_gate_up;
    catalog_config.grouped_exact_q51_down = options.grouped_exact_q51_down;
    catalog_config.q51_slot_parallel_down = options.q51_slot_parallel_down;
    catalog_config.qsa_hierarchical_selection = options.qsa_hierarchical_selection;
    catalog_config.qsa_selection_stats = nullptr;
    catalog_config.qsa_gqa_heads_per_block = binding_request.qsa_gqa_heads_per_block = options.qsa_gqa_heads_per_block;
    catalog_config.qsa_gqa_stats = nullptr;
    binding_request.qsa_hierarchical_selection = options.qsa_hierarchical_selection;
    catalog_config.q4k_a8_stats = nullptr;
    binding_request.q4k_a8g32_gate_up = options.q4k_a8g32_gate_up;
    binding_request.grouped_exact_q51_down = options.grouped_exact_q51_down;
    binding_request.q51_slot_parallel_down = options.q51_slot_parallel_down;
    catalog_config.a8_down_q51 = binding_request.a8_down_q51 = a8_down_q51;
    catalog_config.a8_down_q80 = binding_request.a8_down_q80 = a8_down_q80;
    catalog_config.q5k_a8g32_gate_up = binding_request.q5k_a8g32_gate_up = q5k_a8_gateup;
    catalog_config.dense_a8 = binding_request.dense_a8 = dense_a8;
    catalog_config.deferred_moe_status = options.deferred_moe_status;
    binding_request.deferred_moe_status = options.deferred_moe_status;
    catalog_config.ranked_moe_router_topk = options.ranked_moe_router_topk;
    binding_request.ranked_moe_router_topk = options.ranked_moe_router_topk;
    catalog_config.wave_moe_router_topk = options.wave_moe_router_topk;
    catalog_config.wave_router_manifest_admission = wave_router_admission;
    binding_request.wave_moe_router_topk = options.wave_moe_router_topk;
    catalog_config.staged_moe_router_logits = options.staged_moe_router_logits;
    binding_request.staged_moe_router_logits = options.staged_moe_router_logits;
    catalog_config.parallel_moe_router = options.parallel_moe_router;
    binding_request.parallel_moe_router = options.parallel_moe_router;
    catalog_config.virtual_q8_gdn_input = options.virtual_q8_gdn_input;
    catalog_config.prefetch_q8_gdn_input = options.prefetch_q8_gdn_input;
    catalog_config.prefetch_q8_qsa_input=options.prefetch_q8_qsa_input;
    catalog_config.qsa_input_prefetch_layers=qsa_prefetch_plan.layers;
    catalog_config.qsa_input_prefetch_stats=nullptr;
    binding_request.prefetch_q8_qsa_input=options.prefetch_q8_qsa_input;
    binding_request.qsa_input_prefetch_layers=qsa_prefetch_plan.layers;
    catalog_config.prefetch_q8_shared_expert=options.prefetch_q8_shared_expert;
    catalog_config.shared_expert_prefetch_layers=shared_prefetch_plan.layers;
    catalog_config.shared_expert_prefetch_stats=nullptr;
    binding_request.prefetch_q8_shared_expert=options.prefetch_q8_shared_expert;
    binding_request.shared_expert_prefetch_layers=shared_prefetch_plan.layers;
    catalog_config.virtual_q8_gdn_output = options.virtual_q8_gdn_output;
    strix::ExecutionBufferRequest
        buffer_request;
    buffer_request.logical_rows = 1;
    buffer_request.parallel_moe_router = options.parallel_moe_router;
    buffer_request.q4k_a8g32_gate_up = options.q4k_a8g32_gate_up;
    buffer_request.grouped_exact_q51_down = options.grouped_exact_q51_down;
    buffer_request.q51_slot_parallel_down = options.q51_slot_parallel_down;
    buffer_request.a8_down_q51 = a8_down_q51;
    buffer_request.a8_down_q80 = a8_down_q80;
    buffer_request.q5k_a8g32_gate_up = q5k_a8_gateup;
    buffer_request.dense_a8 = dense_a8;
    buffer_request.qsa_hierarchical_selection = options.qsa_hierarchical_selection;
    buffer_request.qsa_gqa_heads_per_block = options.qsa_gqa_heads_per_block;
    buffer_request.activation_element_bytes =
        2;
    buffer_request.control_element_bytes =
        4;
    buffer_request.index_element_bytes =
        4;
    buffer_request.logit_element_bytes =
        4;
    buffer_request.mlp_intermediate_element_bytes =
        4;

    return {std::move(binding_request), std::move(buffer_request), std::move(catalog_config)};
}
} // namespace strix::decode_cli
