#include "strix/env_flags.hpp"
#include "strix/lane_chain_hyper_output.hpp"
#include "strix/compiled_speculative_verify.hpp"
#include <sstream>
#include "strix/qsa_input_prefetch.hpp"
#include "strix/q8_projection_organization.hpp"
#include "strix/shared_expert_prefetch.hpp"
#include "strix/moe_q4k_a8_runtime.hpp"
#include "strix/moe_q4k_a8_contract.hpp"
#include "strix/gsq_a8_groups.hpp"
#include <cstdlib>
#include "strix/gguf_reader.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <span>
#include <stdexcept>
#include <vector>

namespace strix {

namespace {
GraphExecutionContract verify_graph_contract(const CompiledSpeculativeVerifyExecution& compiled) {
    bool safe = compiled.whole_plan_graph_safe && compiled.dynamic_steps == 0 &&
        !compiled.verify.steps.empty() && compiled.bindings.missing_steps == 0 &&
        compiled.bindings.bindings.size() == compiled.verify.steps.size();
    for (const auto& step : compiled.verify.steps) {
        safe = safe && !step.dynamic_shape && step.graph_safe_candidate &&
            step.mode != SpeculativeVerifyMode::dynamic_row_parallel &&
            step.mode != SpeculativeVerifyMode::runtime_position_window;
    }
    for (const auto& binding : compiled.bindings.bindings) {
        safe = safe && !binding.host_or_runtime_step && binding.kernel_id.has_value();
    }
    return {compiled.execution_identity, compiled.kernel_plan_identity, compiled.candidate_depth, safe, compiled.rotary_table};
}
} // namespace

GraphCapturePlan prepare_compiled_verify_graph_capture(
    const CompiledSpeculativeVerifyExecution& compiled, const ExecutionRuntimeResources& resources,
    const GraphDynamicArguments& arguments, const GraphMemoryRequest& memory) {
    return prepare_graph_capture_plan(
        verify_graph_contract(compiled),
        compiled.arena, resources, arguments, memory);
}

GraphEligibility assess_compiled_verify_graph_replay(
    const CompiledSpeculativeVerifyExecution& compiled, const GraphCapturePlan& capture,
    const ExecutionRuntimeResources& resources, const GraphDynamicArguments& arguments,
    std::uint64_t position) {
    return assess_graph_replay_eligibility(capture,
        verify_graph_contract(compiled),
        compiled.arena, resources, arguments, position);
}

namespace {

constexpr std::uint64_t fnv_offset =
    14695981039346656037ULL;
constexpr std::uint64_t fnv_prime =
    1099511628211ULL;

void hash_u64(
    std::uint64_t& hash,
    std::uint64_t value) noexcept {
    for (unsigned shift = 0;
         shift < 64;
         shift += 8) {
        hash ^= (value >> shift) & 0xffULL;
        hash *= fnv_prime;
    }
}

std::uint64_t build_identity(
    const ModelContract& model,
    const CompiledSpeculativeVerifyExecution& compiled,
    const ExecutionBufferRequest& buffer_request) noexcept {
    std::uint64_t hash = fnv_offset;

    hash = append_rotary_identity(hash, model.rotary_identity());
    hash_u64(hash, model.vocab_size);
    hash_u64(hash, model.hidden_size);
    hash_u64(hash, model.max_position_embeddings);
    hash_u64(hash, model.hidden_layers);
    hash_u64(hash, model.full_attention_interval);
    hash_u64(hash, model.ple_layer_1based);
    hash_u64(hash, model.hc_count);
    hash_u64(hash, model.gdn_layers);
    hash_u64(hash, model.num_experts);
    hash_u64(hash, model.num_experts_per_token);

    hash_u64(hash, compiled.candidate_depth);
    hash_u64(hash, compiled.request_concurrency);
    hash_u64(hash, compiled.kernel_plan_identity);

    hash_u64(hash, buffer_request.activation_element_bytes);
    hash_u64(hash, buffer_request.control_element_bytes);
    hash_u64(hash, buffer_request.index_element_bytes);
    hash_u64(hash, buffer_request.logit_element_bytes);
    hash_u64(hash, buffer_request.mlp_intermediate_element_bytes);

    if (buffer_request.parallel_moe_router) hash_u64(hash, 0x524f555445525057ULL);

    for (const auto& step :
         compiled.verify.steps) {
        hash_u64(hash, step.id);
        hash_u64(
            hash,
            static_cast<std::uint64_t>(step.op));
        hash_u64(
            hash,
            step.layer.has_value()
                ? static_cast<std::uint64_t>(*step.layer) + 1ULL
                : 0ULL);
        hash_u64(
            hash,
            static_cast<std::uint64_t>(step.mode));
        hash_u64(
            hash,
            static_cast<std::uint64_t>(step.reads_state));
        hash_u64(
            hash,
            static_cast<std::uint64_t>(step.writes_state));
        hash_u64(hash, step.dynamic_shape ? 1ULL : 0ULL);
        hash_u64(hash, step.graph_safe_candidate ? 1ULL : 0ULL);
        for (const auto dependency : step.dependencies) {
            hash_u64(hash, dependency);
        }
        hash_u64(hash, 0xffffffffffffffffULL);
    }

    hash_u64(hash, compiled.arena.alignment);
    hash_u64(hash, compiled.arena.total_bytes);
    for (const auto& allocation :
         compiled.arena.allocations) {
        hash_u64(
            hash,
            static_cast<std::uint64_t>(
                allocation.use.buffer.buffer_class));
        hash_u64(hash, allocation.use.buffer.slot);
        hash_u64(hash, allocation.use.generation);
        hash_u64(hash, allocation.first_step);
        hash_u64(hash, allocation.last_step);
        hash_u64(hash, allocation.offset);
        hash_u64(hash, allocation.bytes);
    }

    return hash;
}

void validate_topology_match(
    const SerialExecutionPlan& topology,
    const SpeculativeVerifyPlan& verify) {
    if (topology.steps.size() !=
        verify.steps.size()) {
        throw std::logic_error(
            "speculative verify topology size differs from serial model topology");
    }

    for (std::size_t index = 0;
         index < topology.steps.size();
         ++index) {
        const auto& serial =
            topology.steps[index];
        const auto& step =
            verify.steps[index];
        if (serial.id != step.id ||
            serial.op != step.op ||
            serial.layer != step.layer ||
            serial.dependencies != step.dependencies ||
            serial.reads_state != step.reads_state ||
            serial.writes_state != step.writes_state) {
            throw std::logic_error(
                "speculative verify topology no longer matches serial operator graph");
        }
    }
}

}  // namespace

CompiledSpeculativeVerifyExecution
compile_speculative_verify_execution(
    const ModelContract& model,
    std::uint32_t candidate_depth,
    const kernels::KernelRegistry& registry,
    const SpeculativeVerifyBindingRequest& binding_request,
    const ExecutionBufferRequest& buffer_request,
    std::uint64_t arena_alignment) {
    if(binding_request.prefetch_q8_qsa_input)throw std::invalid_argument("QSA input prefetch runtime does not support speculative verify");
    if(binding_request.prefetch_q8_shared_expert)throw std::invalid_argument("shared prefetch runtime does not support speculative verify");
    if (buffer_request.q4k_a8g32_gate_up || buffer_request.grouped_exact_q51_down)
        throw std::invalid_argument("Q4K A8 runtime does not support speculative verify");
    if (candidate_depth == 0) {
        throw std::invalid_argument(
            "compiled speculative verify requires positive depth");
    }
    if (binding_request.request_concurrency != 1) {
        throw std::invalid_argument(
            "initial speculative verify compilation supports one request only");
    }
    if (buffer_request.logical_rows != candidate_depth) {
        throw std::invalid_argument(
            "verify buffer logical rows must equal candidate depth");
    }

    if (binding_request.wave_moe_router_topk &&
        (!binding_request.parallel_moe_router || binding_request.ranked_moe_router_topk))
        throw std::invalid_argument("wave router requires parallel topology and excludes ranked mode");
    if (binding_request.deferred_moe_status)
        throw std::invalid_argument("deferred MoE status is not a speculative verification contract");
    if (binding_request.ranked_moe_router_topk && !binding_request.parallel_moe_router)
        throw std::invalid_argument("ranked router requires parallel router topology");
    if (binding_request.parallel_moe_router != buffer_request.parallel_moe_router)
        throw std::invalid_argument("router kernel request differs from buffer admission");

    CompiledSpeculativeVerifyExecution compiled;
    compiled.rotary_table = model.qwen_rotary_table;
    compiled.vocab_size =
        model.vocab_size;
    compiled.candidate_depth =
        candidate_depth;
    compiled.request_concurrency =
        binding_request.request_concurrency;
    compiled.topology =
        build_serial_decode_execution_plan(model);
    compiled.verify =
        build_speculative_verify_plan(
            model,
            candidate_depth);

    validate_topology_match(
        compiled.topology,
        compiled.verify);

    compiled.bindings =
        bind_speculative_verify_plan(
            model,
            compiled.verify,
            registry,
            binding_request);
    for(const auto& binding:compiled.bindings.bindings)if(binding.kernel_id && binding.kernel_id->ends_with(lane_chain_hyper_output_suffix))
        throw std::invalid_argument("lane-chain HC output requires serial decode binding");
    for (const auto& binding : compiled.bindings.bindings)
        if (binding.op == SerialOp::moe_experts && binding.kernel_id &&
            binding.kernel_id->ends_with(moe_q4k_a8::runtime_kernel_suffix))
            throw std::invalid_argument("Q4K A8 runtime does not support speculative verify");
    if (compiled.bindings.missing_steps != 0) {
        throw std::invalid_argument(
            "compiled speculative verify requires all kernel steps bound");
    }

    for (const auto& binding : compiled.bindings.bindings) {
        if (binding.op == SerialOp::moe_route && binding.kernel_id &&
            binding.kernel_id->find(".staged_logits") != std::string::npos)
            throw std::invalid_argument("staged router logits does not support speculative verify");
        // Defense for externally assembled bindings; current request admission rejects first.
        if(binding.op==SerialOp::indexed_attention && binding.kernel_id && binding.kernel_id->ends_with(qsa_input_prefetch::kernel_suffix))throw std::invalid_argument("QSA input prefetch runtime does not support speculative verify");
        if(binding.op==SerialOp::shared_expert && binding.kernel_id && binding.kernel_id->ends_with(shared_expert_prefetch::kernel_suffix))throw std::invalid_argument("shared prefetch runtime does not support speculative verify");
        if (binding.op == SerialOp::moe_route && binding.kernel_id.has_value() &&
            (binding.kernel_id->ends_with(".parallel") != buffer_request.parallel_moe_router ||
             binding.kernel_id->ends_with(".ranked.parallel") != binding_request.ranked_moe_router_topk ||
             binding.kernel_id->ends_with(".wave.parallel") != binding_request.wave_moe_router_topk))
            throw std::invalid_argument("bound router kernel differs from admitted topology");
    }

    compiled.io =
        build_serial_execution_io_plan(
            compiled.topology, buffer_request.parallel_moe_router);
    compiled.arena =
        build_serial_execution_arena_plan(
            model,
            buffer_request,
            compiled.topology,
            compiled.io,
            arena_alignment);

    std::vector<const kernels::KernelVariant*>
        selected;
    selected.reserve(
        compiled.bindings.bound_steps);

    bool graph_safe = true;

    for (std::size_t index = 0;
         index < compiled.verify.steps.size();
         ++index) {
        const auto& step =
            compiled.verify.steps[index];
        const auto& binding =
            compiled.bindings.bindings[index];

        switch (step.mode) {
        case SpeculativeVerifyMode::sequential_window:
            ++compiled.sequential_window_steps;
            break;
        case SpeculativeVerifyMode::causal_window:
            ++compiled.causal_window_steps;
            break;
        case SpeculativeVerifyMode::dynamic_row_parallel:
            ++compiled.dynamic_steps;
            graph_safe = false;
            break;
        case SpeculativeVerifyMode::runtime_position_window:
            graph_safe = false;
            break;
        case SpeculativeVerifyMode::row_parallel:
            break;
        }

        if (step.dynamic_shape &&
            step.mode !=
                SpeculativeVerifyMode::dynamic_row_parallel) {
            ++compiled.dynamic_steps;
            graph_safe = false;
        }
        if (!step.graph_safe_candidate) {
            graph_safe = false;
        }

        if (binding.host_or_runtime_step) {
            graph_safe = false;
            continue;
        }
        if (!binding.kernel_id.has_value()) {
            throw std::logic_error(
                "fully bound verify step lacks kernel id");
        }

        const auto* variant =
            registry.find(
                *binding.kernel_id);
        if (variant == nullptr ||
            variant->phase !=
                kernels::ExecutionPhase::
                    speculative_verify) {
            throw std::logic_error(
                "verify binding refers to missing or wrong-phase kernel");
        }
        selected.push_back(variant);
        if (!variant->graph_safe) {
            graph_safe = false;
        }
    }

    compiled.kernel_plan_identity =
        kernels::plan_identity(
            std::span<
                const kernels::KernelVariant* const>(
                selected.data(),
                selected.size()));
    if (compiled.kernel_plan_identity == 0) {
        throw std::logic_error(
            "verify kernel plan identity must be non-zero");
    }

    compiled.whole_plan_graph_safe =
        graph_safe;
    compiled.execution_identity =
        build_identity(
            model,
            compiled,
            buffer_request);
    if (compiled.execution_identity == 0) {
        throw std::logic_error(
            "verify execution identity must be non-zero");
    }

    return compiled;
}


namespace {
void window_require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
void window_hash_text(std::uint64_t& hash, std::string_view text) noexcept {
    hash_u64(hash, text.size());
    for (char value : text) hash_u64(hash, static_cast<unsigned char>(value));
}
void window_hash_use(std::uint64_t& hash, ExecutionBufferUse use) noexcept {
    hash_u64(hash, static_cast<std::uint64_t>(use.buffer.buffer_class));
    hash_u64(hash, use.buffer.slot); hash_u64(hash, use.generation);
}
void window_hash_arena(std::uint64_t& hash, const ExecutionArenaPlan& arena) noexcept {
    hash_u64(hash, arena.alignment); hash_u64(hash, arena.total_bytes);
    hash_u64(hash, arena.naive_nonaliased_bytes); hash_u64(hash, arena.allocations.size());
    for (const auto& allocation : arena.allocations) {
        window_hash_use(hash, allocation.use); hash_u64(hash, allocation.first_step);
        hash_u64(hash, allocation.last_step); hash_u64(hash, allocation.offset);
        hash_u64(hash, allocation.bytes);
    }
}
std::uint64_t window_serial_fingerprint(const CompiledSerialExecution& serial) noexcept {
    auto hash = fnv_offset;
    window_hash_text(hash, "known-window-serial-source-v1");
    for (auto value : {serial.execution_identity, serial.kernel_plan_identity, serial.vocab_size,
                       serial.max_position_embeddings}) hash_u64(hash, value);
    for (auto value : {serial.logical_rows, serial.request_concurrency, serial.dynamic_steps,
                       serial.non_graph_safe_steps}) hash_u64(hash, value);
    hash_u64(hash, serial.whole_plan_graph_safe); hash_u64(hash, serial.device_graph_inputs);
    hash_u64(hash, serial.execution.steps.size());
    for (const auto& step : serial.execution.steps) {
        hash_u64(hash, step.id); hash_u64(hash, static_cast<std::uint64_t>(step.op));
        hash_u64(hash, step.layer ? static_cast<std::uint64_t>(*step.layer) + 1U : 0U);
        hash_u64(hash, static_cast<std::uint64_t>(step.reads_state));
        hash_u64(hash, static_cast<std::uint64_t>(step.writes_state));
        hash_u64(hash, step.dynamic_shape); hash_u64(hash, step.graph_safe_candidate);
        hash_u64(hash, step.dependencies.size());
        for (auto dependency : step.dependencies) hash_u64(hash, dependency);
    }
    for (auto value : {serial.execution.linear_attention_steps, serial.execution.indexed_attention_steps,
        serial.execution.ple_steps, serial.execution.moe_route_steps, serial.execution.moe_expert_steps,
        serial.bindings.bound_steps, serial.bindings.missing_steps, serial.bindings.host_or_runtime_steps})
        hash_u64(hash, value);
    hash_u64(hash, serial.bindings.bindings.size());
    for (const auto& binding : serial.bindings.bindings) {
        hash_u64(hash, binding.step_id); hash_u64(hash, static_cast<std::uint64_t>(binding.op));
        hash_u64(hash, binding.family ? static_cast<std::uint64_t>(*binding.family) + 1U : 0U);
        hash_u64(hash, binding.host_or_runtime_step); hash_u64(hash, binding.kernel_id.has_value());
        if (binding.kernel_id) window_hash_text(hash, *binding.kernel_id);
    }
    for (auto value : {serial.io.expanded_hidden_slots, serial.io.hidden_slots, serial.io.injection_slots,
        serial.io.route_index_slots, serial.io.route_weight_slots, serial.io.logits_slots,
        serial.io.mlp_intermediate_slots, serial.io.gdn_workspace_slots, serial.io.ple_workspace_slots,
        serial.io.indexed_attention_workspace_slots, serial.io.hyper_workspace_slots, serial.io.router_workspace_slots})
        hash_u64(hash, value);
    hash_u64(hash, serial.io.steps.size());
    for (const auto& io : serial.io.steps) {
        hash_u64(hash, io.step_id); hash_u64(hash, static_cast<std::uint64_t>(io.op));
        for (const auto* uses : {&io.reads, &io.writes, &io.temporaries}) {
            hash_u64(hash, uses->size());
            for (auto use : *uses) window_hash_use(hash, use);
        }
    }
    window_hash_arena(hash, serial.arena);
    return hash;
}
bool window_same_arena(const ExecutionArenaPlan& left, const ExecutionArenaPlan& right) noexcept {
    if (left.alignment != right.alignment || left.total_bytes != right.total_bytes ||
        left.naive_nonaliased_bytes != right.naive_nonaliased_bytes ||
        left.allocations.size() != right.allocations.size()) return false;
    for (std::size_t i = 0; i < left.allocations.size(); ++i) {
        const auto& a = left.allocations[i]; const auto& b = right.allocations[i];
        if (a.use != b.use || a.first_step != b.first_step || a.last_step != b.last_step ||
            a.offset != b.offset || a.bytes != b.bytes) return false;
    }
    return true;
}
constexpr std::string_view known_shared_base = "hip.shared_expert.bf16.h2560.m640.r1.c1";
constexpr std::string_view known_gdn_wave8_head = "hip.gdn.bf16.h2560.kh16.vh48.d128.r1.c1.scalar.wave8-input-v1.wave8-q8-output-v1";
constexpr std::string_view known_gdn_wave8_tail = ".prefetch8.virtual256.q8-input.virtual256.q8-output";
constexpr std::string_view known_trellis_marker = ".trellis-g1-fp32-s4-v1";
// The serial plan's dense A8 marker (one identity for the whole plan, checked by the
// serial compiler) carries over to every window/prefill row of the step.
bool known_shared_id(std::string_view id) {
    for (const bool dense : {false, true})
        for (const bool prefetch : {false, true})
            if (id == std::string(known_shared_base) + std::string(dense ? q8_projection_organization::dense_a8_marker : "") +
                    std::string(prefetch ? shared_expert_prefetch::kernel_suffix : ""))
                return true;
    return false;
}
bool known_gdn_id(std::string_view id) {
    for (const bool dense : {false, true})
      for (const bool trellis : {false, true}) {
        if (trellis && !dense) continue;
        if (id == std::string(known_gdn_wave8_head) + std::string(dense ? q8_projection_organization::dense_a8_marker : "") +
                std::string(trellis ? known_trellis_marker : "") +
                std::string(known_gdn_wave8_tail))
            return true;
      }
    return false;
}
// Serial routed-expert arithmetic the union reproduces: the float profile
// (slot-parallel Q5_1 down) or the A8 profile (Q5_1 A8 down, optional Q8_0
// A8 down and Q5_K A8 gate/up layers), read from the serial kernel id.
struct RoutedUnionProfile { bool a8 = false, q5k = false, q80 = false; };
std::optional<RoutedUnionProfile> routed_union_profile(std::string_view id) {
    constexpr std::string_view base="hip.moe_experts.bf16.h2560.m640.e512.k10.r1.c1";
    constexpr std::string_view suffix=".deferred-status-v1.q51-slot-parallel-v1"
        ".grouped-exact-q51-down-v1.q4k-a8g32-gateup-v2";
    if(id==std::string(base)+std::string(suffix) ||
       id==std::string(base)+".batched-slot-activation-v1"+std::string(suffix)) return RoutedUnionProfile{};
    const auto prefix=std::string(base)+".batched-slot-activation-v1.deferred-status-v1";
    for(const bool q5k:{false,true}) for(const bool q80:{false,true}) {
        if(q5k && !q80) continue; // serial config: Q5_K requires the complete Q5_1/Q8_0 A8 down profile
        if(id==prefix+std::string(q5k?moe_q4k_a8::q5k_gate_up_kernel_marker:"")+
            std::string(moe_q4k_a8::q51_a8_down_kernel_marker)+std::string(q80?moe_q4k_a8::q80_a8_down_kernel_marker:"")+
            std::string(moe_q4k_a8::a8_down_kernel_ending)) return RoutedUnionProfile{true,q5k,q80};
    }
    return std::nullopt;
}
bool known_qsa_id(std::string_view id) {
    constexpr std::string_view base="hip.indexed_attention.bf16.h2560.q24x256.kv2.idx4x128.b2048.r1.c1.scalar";
    for(const bool wave8:{false,true}) for(const bool dense:{false,true}) for(const bool prefetch:{false,true})
      for(const bool trellis:{false,true}) {
        if (trellis && !dense) continue;
        if(id==std::string(base)+std::string(wave8?q8_projection_organization::qsa_output_wave8_marker:"")+
            std::string(dense?q8_projection_organization::dense_a8_marker:"")+
            std::string(trellis?known_trellis_marker:"")+
            std::string(prefetch?qsa_input_prefetch::kernel_suffix:"")) return true;
      }
    return false;
}
// Source manifests retain Q8 tensors for the wide prompt. Window accounting
// describes the actual selected decode operands, including signed scale vectors.
std::pair<std::uint64_t, std::uint32_t> trellis_window_adjustment(
    std::string_view id, KnownWindowStepMode mode, std::uint32_t rows) {
    if (id.find(known_trellis_marker) == std::string_view::npos) return {};
    std::uint64_t outputs = 0;
    std::uint32_t saved = rows - 1U;
    if (mode == KnownWindowStepMode::qsa_rows) outputs = 12288U;
    else if (mode == KnownWindowStepMode::gdn_rows || mode == KnownWindowStepMode::gdn_deferred_rows) outputs = 6144U;
    else if (mode == KnownWindowStepMode::r2_gdn_staged) { outputs = 6144U; saved = rows / 2U; }
    else return {};
    const auto q8 = 2560ULL / 32U * 34U * outputs;
    const auto packed = 2560ULL * outputs * 5U / 8U + (2560U + outputs) * 2U;
    // Compact input/partial/output launches replace one input projection;
    // the remaining source projections retain their existing merged launch.
    return {(q8 - packed) * saved, 3U * saved};
}
const ManifestTensor& window_tensor(const ModelManifest& manifest,
    const ExecutionTensorAccessPlan& access, std::uint32_t step, SerialOp op,
    std::uint32_t layer, std::string_view suffix) {
    const auto name = "blk." + std::to_string(layer) + "." + std::string(suffix);
    const ManifestTensor* selected = nullptr; std::uint32_t index = 0;
    for (std::size_t i = 0; i < manifest.tensors.size(); ++i) if (manifest.tensors[i].name == name) {
        window_require(selected == nullptr, "known window tensor family is duplicated");
        window_require(i <= std::numeric_limits<std::uint32_t>::max(), "known window tensor index overflows");
        selected = &manifest.tensors[i]; index = static_cast<std::uint32_t>(i);
    }
    window_require(selected != nullptr && selected->layer == layer,
        "known window promoted tensor family is missing or has the wrong layer");
    const auto role = suffix.ends_with("_exps.weight") ? TensorRole::moe_expert :
        suffix.starts_with("attn_") ? TensorRole::attention_projection :
        suffix.starts_with("ssm_") ? TensorRole::recurrent_state :
        suffix.starts_with("indexer.") ? TensorRole::sparse_index : TensorRole::dense_projection;
    window_require(selected->role == role, "known window promoted tensor role is invalid");
    unsigned matches = 0;
    for (const auto& item : access.accesses) if (item.step_id == step) {
        window_require(item.op == op, "known window tensor access operator mismatch");
        for (auto candidate : item.tensor_indices) {
            window_require(candidate < manifest.tensors.size(), "known window tensor access index is invalid");
            if (candidate == index) ++matches;
        }
    }
    window_require(matches == 1, "known window promoted tensor is not selected exactly once");
    return *selected;
}
bool window_routed_union_types(const ModelManifest& manifest, const ExecutionTensorAccessPlan& access,
    const ExecutionStep& step, RoutedUnionProfile profile) {
    bool eligible=true;
    std::uint32_t gate_type=0;
    for(const auto suffix:{"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"}) {
        const bool down=std::string_view(suffix)=="ffn_down_exps.weight";
        const auto width=down?640ULL:2560ULL, outputs=down?2560ULL:640ULL;
        const auto& tensor=window_tensor(manifest,access,step.id,step.op,*step.layer,suffix);
        window_require(tensor.dimensions==std::vector<std::uint64_t>{width,outputs,512ULL},
            "known window routed expert shape differs from canonical operands");
        const auto type=tensor.ggml_type;
        // STRIX_GSQ_WINDOW_EXPERTS=0 (diagnostic): GSQ routed experts stay out of windows (per-op serial fallback).
        static const bool gsq_window_experts=strix::envcfg::flag("STRIX_GSQ_WINDOW_EXPERTS", true);
        const bool q20=profile.a8 && moe_q4k_a8::q20_a8_enabled() && gsq_window_experts;
        const bool admitted=down ? type==7U || (profile.q80 && type==8U) || (q20 && profile.q80 && gsq_a8::down_type(type)) :
            (type==12U || (profile.q5k && type==13U) || (q20 && gsq_a8::gate_type(type))) && (!gate_type || type==gate_type);
        if(!down) gate_type=type;
        if(!admitted) {eligible=false;continue;}
        const auto geometry=gguf::tensor_type_geometry(tensor.ggml_type);
        window_require(geometry && geometry->block_elements && width%geometry->block_elements==0,
            "known window routed expert block geometry is invalid");
        const auto bytes=width/geometry->block_elements*geometry->block_bytes*outputs*512ULL;
        window_require(tensor.byte_size==bytes && tensor.absolute_offset%16U==0 &&
            bytes<=std::numeric_limits<std::uint64_t>::max()-tensor.absolute_offset,
            "known window routed expert extent/alignment is invalid");
    }
    return eligible;
}
// Typed prefetched Q8 HC selection, optional split-K FP32 projection and lane-chain output.
bool known_hyper_rows_id(std::string_view id) {
    constexpr std::string_view base="hip.hyper_mix.bf16.h2560.hc4.lr320.r1.c1.parallel.staged.fused-injection-v5"
        ".typed-q8-f32-v1.prefetch-rows8-v2.prefetch-q8-output-v1";
    if(!id.starts_with(base)) return false;
    id.remove_prefix(base.size());
    if(id.starts_with(".splitk-hc-proj-fp32-v1")) id.remove_prefix(std::string_view(".splitk-hc-proj-fp32-v1").size());
    return id.empty() || id==lane_chain_hyper_output_suffix;
}
bool hyper_mix_op(SerialOp op) {
    return op==SerialOp::attn_hyper_mix || op==SerialOp::mlp_hyper_mix || op==SerialOp::final_hyper_mix;
}
KnownWindowR2Variant hyper_rows_variant(std::uint32_t rows) {
    return rows==2U?KnownWindowR2Variant::hyper_mix_rows2_v1:
        rows==4U?KnownWindowR2Variant::hyper_mix_rows4_v1:KnownWindowR2Variant::hyper_mix_rows8_v1;
}
// Q8 down + Q8 up (+ F32 injection) per mixer, read once per window instead of per row.
std::uint64_t known_hyper_projection_bytes(SerialOp op) {
    return 2ULL*(10240ULL/32ULL*34ULL*320ULL) + (op==SerialOp::final_hyper_mix ? 0ULL : 10240ULL*4ULL*4ULL);
}
bool known_router_rows_id(std::string_view id) {
    return id.starts_with("hip.moe_route.") && id.ends_with(".parallel");
}
KnownWindowStepMode known_elementwise_rows_mode(const ExecutionStep& step, std::string_view id,
    std::uint64_t hidden_layers) {
    if (step.op == SerialOp::moe_combine && id.starts_with("hip.moe_combine.")) return KnownWindowStepMode::moe_combine_rows;
    if ((step.op == SerialOp::attn_inject ||
         (step.op == SerialOp::mlp_inject && step.layer && *step.layer + 1U != hidden_layers)) &&
        id.starts_with("hip.inject.")) return KnownWindowStepMode::inject_rows;
    return KnownWindowStepMode::r1_row_loop;
}
KnownWindowR2Variant routed_union_variant(std::uint32_t rows, RoutedUnionProfile profile) {
    if(profile.a8) return rows==2U?KnownWindowR2Variant::routed_a8_union_rows2_v1:
        rows==4U?KnownWindowR2Variant::routed_a8_union_rows4_v1:KnownWindowR2Variant::routed_a8_union_rows8_v1;
    return rows==2U?KnownWindowR2Variant::routed_q51_union_rows2_v1:
        rows==4U?KnownWindowR2Variant::routed_q51_union_rows4_v1:KnownWindowR2Variant::routed_q51_union_rows8_v1;
}
bool window_matrix(const ManifestTensor& tensor, std::uint64_t width, std::uint64_t rows) {
    window_require(tensor.dimensions == std::vector<std::uint64_t>{width, rows},
        "known window promoted matrix shape differs from its canonical operand");
    if (tensor.ggml_type != 8U) {
        // GSQ-RCO files (STRIX_GSQ_Q20_A8=1): the dense A8 typed segments run these matrices at R rows.
        // STRIX_GSQ_WINDOW_DENSE=0 (diagnostic): GSQ dense matrices stay out of windows (per-op serial fallback).
        static const bool gsq_window_dense=strix::envcfg::flag("STRIX_GSQ_WINDOW_DENSE", true);
        if (!moe_q4k_a8::q20_a8_enabled() || !gsq_window_dense || !gsq_a8::dense_type(tensor.ggml_type) ||
            width % gsq_a8::dense_block_elements(tensor.ggml_type))
            return false; // Explicit per-op serial fallback.
        window_require(tensor.byte_size == gsq_a8::dense_row_bytes(tensor.ggml_type, static_cast<std::uint32_t>(width)) * rows &&
            tensor.absolute_offset % 4U == 0, "known window GSQ weight extent/alignment is invalid");
        return true;
    }
    const auto bytes = (width / 32U) * 34U * rows;
    window_require(width % 32U == 0 && tensor.byte_size == bytes && tensor.absolute_offset % 16U == 0 &&
        tensor.byte_size <= std::numeric_limits<std::uint64_t>::max() - tensor.absolute_offset,
        "known window Q8 weight extent/alignment is invalid");
    return true;
}
bool window_shared_types(const ModelManifest& manifest, const ExecutionTensorAccessPlan& access,
    const ExecutionStep& step) {
    const auto layer = *step.layer;
    const auto& gate = window_tensor(manifest, access, step.id, step.op, layer, "ffn_gate_shexp.weight");
    const auto& up = window_tensor(manifest, access, step.id, step.op, layer, "ffn_up_shexp.weight");
    const auto& down = window_tensor(manifest, access, step.id, step.op, layer, "ffn_down_shexp.weight");
    const auto& scalar = window_tensor(manifest, access, step.id, step.op, layer, "ffn_gate_inp_shexp.weight");
    const bool g = window_matrix(gate, 2560, 640), u = window_matrix(up, 2560, 640),
               d = window_matrix(down, 640, 2560);
    window_require(scalar.dimensions == std::vector<std::uint64_t>{2560}, "known window scalar gate shape mismatch");
    if (scalar.ggml_type != 0U) return false;
    window_require(scalar.byte_size == 10240 && scalar.absolute_offset % 16U == 0 &&
        scalar.byte_size <= std::numeric_limits<std::uint64_t>::max() - scalar.absolute_offset,
        "known window scalar gate extent/alignment mismatch");
    return g && u && d;
}
constexpr std::array<std::pair<std::string_view, std::array<std::uint64_t, 2>>, 6> known_qsa_matrices{{
    {"attn_q.weight", {2560, 12288}}, {"attn_k.weight", {2560, 512}}, {"attn_v.weight", {2560, 512}},
    {"attn_output.weight", {6144, 2560}}, {"indexer.q_proj.weight", {2560, 512}}, {"indexer.k_proj.weight", {2560, 128}}}};
enum class WindowQsaFormat { unsupported, all_q8, q8_bf16_indexer };
WindowQsaFormat window_qsa_format(const ModelManifest& manifest, const ExecutionTensorAccessPlan& access, const ExecutionStep& step) {
    bool attention_q8 = true, indexer_q8 = true, indexer_bf16 = true;
    // Always resolve and shape-check all six operands, even for fallback.
    for (std::size_t i = 0; i < known_qsa_matrices.size(); ++i) {
        const auto& item = known_qsa_matrices[i];
        const auto& tensor = window_tensor(manifest, access, step.id, step.op, *step.layer, item.first);
        const bool q8 = window_matrix(tensor, item.second[0], item.second[1]);
        if (i < 4U) { attention_q8 = q8 && attention_q8; continue; }
        indexer_q8 = q8 && indexer_q8;
        const bool bf16 = tensor.ggml_type == 30U;
        if (bf16) window_require(tensor.byte_size == item.second[0] * item.second[1] * 2U && tensor.absolute_offset % 16U == 0 &&
            tensor.byte_size <= std::numeric_limits<std::uint64_t>::max() - tensor.absolute_offset,
            "known window BF16 indexer extent/alignment is invalid");
        indexer_bf16 = bf16 && indexer_bf16;
    }
    if (!attention_q8) return WindowQsaFormat::unsupported;
    return indexer_q8 ? WindowQsaFormat::all_q8 : indexer_bf16 ? WindowQsaFormat::q8_bf16_indexer : WindowQsaFormat::unsupported;
}
KnownWindowR2Variant qsa_rows_variant(std::uint32_t rows, WindowQsaFormat format) {
    if (format == WindowQsaFormat::q8_bf16_indexer)
        return rows==2U?KnownWindowR2Variant::qsa_q8_bf16idx_rows2_r1_v1:
            rows==4U?KnownWindowR2Variant::qsa_q8_bf16idx_rows4_r1_v1:KnownWindowR2Variant::qsa_q8_bf16idx_rows8_r1_v1;
    return rows==2U?KnownWindowR2Variant::qsa_q8_rows2_r1_v1:
        rows==4U?KnownWindowR2Variant::qsa_q8_rows4_r1_v1:KnownWindowR2Variant::qsa_q8_rows8_r1_v1;
}
KnownWindowR2Variant gdn_deferred_variant(std::uint32_t rows) {
    return rows==2U ? KnownWindowR2Variant::gdn_deferred_rows2_v1 :
        rows==4U ? KnownWindowR2Variant::gdn_deferred_rows4_v1 : KnownWindowR2Variant::gdn_deferred_rows8_v1;
}
enum class WindowGdnFormat { unsupported, all_q8, q8_large_f32_gates };
WindowGdnFormat window_gdn_format(const ModelManifest& manifest, const ExecutionTensorAccessPlan& access,
    const ExecutionStep& step, bool admit_f32_gates) {
    bool all_q8 = true, large_q8 = true, f32_gates = true;
    std::size_t index = 0;
    for (const auto& item : std::array<std::pair<std::string_view, std::array<std::uint64_t, 2>>, 5>{{
        {"attn_qkv.weight", {2560, 10240}}, {"attn_gate.weight", {2560, 6144}},
        {"ssm_beta.weight", {2560, 48}}, {"ssm_alpha.weight", {2560, 48}},
        {"ssm_out.weight", {6144, 2560}}}}) {
        const auto& tensor = window_tensor(manifest, access, step.id, step.op, *step.layer, item.first);
        // Always resolve and shape-check all five operands, even for fallback.
        const bool q8 = window_matrix(tensor, item.second[0], item.second[1]);
        all_q8 = q8 && all_q8;
        if (index == 2U || index == 3U) {
            f32_gates = (tensor.ggml_type == 0U || (tensor.ggml_type == 30U && moe_q4k_a8::q20_a8_enabled())) && f32_gates;
            if (admit_f32_gates && tensor.ggml_type == 30U && moe_q4k_a8::q20_a8_enabled()) {
                window_require(tensor.byte_size == 245760ULL && tensor.absolute_offset % 16U == 0,
                    "known window BF16 gate extent/alignment is invalid");
            } else if (admit_f32_gates && tensor.ggml_type == 0U) {
                window_require(tensor.byte_size == 491520ULL && tensor.absolute_offset % 16U == 0 &&
                    tensor.byte_size <= std::numeric_limits<std::uint64_t>::max() - tensor.absolute_offset,
                    "known window F32 gate extent/alignment is invalid");
            }
        } else large_q8 = q8 && large_q8;
        ++index;
    }
    if (all_q8) return WindowGdnFormat::all_q8;
    if (admit_f32_gates && large_q8 && f32_gates) return WindowGdnFormat::q8_large_f32_gates;
    return WindowGdnFormat::unsupported;
}
bool known_head_id(std::string_view id, std::uint64_t vocab) {
    const auto base = "hip.lm_head.bf16.h2560.v" + std::to_string(vocab) + ".r1.c1";
    return id == base + ".prefetch8-wave8-v1.virtual256.q8" || id == base + ".prefetch8-load-v1.virtual256.q8";
}
KnownWindowR2Variant head_rows_variant(std::uint32_t rows) {
    return rows==2U?KnownWindowR2Variant::lm_head_q8_prefetch8_rows2_v1:
        rows==4U?KnownWindowR2Variant::lm_head_q8_prefetch8_rows4_v1:KnownWindowR2Variant::lm_head_q8_prefetch8_rows8_v1;
}
std::uint64_t known_head_bytes(std::uint64_t vocab) { return 2560ULL / 32ULL * 34ULL * vocab; }
bool window_head_types(const ModelManifest& manifest, const ExecutionTensorAccessPlan& access,
    const ExecutionStep& step, std::uint64_t vocab) {
    const ManifestTensor* selected = nullptr;
    for (const auto& item : access.accesses) if (item.step_id == step.id) {
        window_require(item.op == step.op && item.tensor_indices.size() == 1U && !selected &&
            item.tensor_indices[0] < manifest.tensors.size(), "known window head tensor access is not one tensor");
        selected = &manifest.tensors[item.tensor_indices[0]];
    }
    window_require(selected && selected->role == TensorRole::output_head, "known window head tensor is missing");
    return window_matrix(*selected, 2560, vocab);
}
constexpr std::uint64_t known_shared_projection_bytes = 3ULL * (2560ULL / 32ULL * 34ULL * 640ULL);
constexpr std::uint64_t known_gdn_projection_bytes = 2560ULL / 32ULL * 34ULL *
    (10240ULL + 6144ULL + 48ULL + 48ULL) + 6144ULL / 32ULL * 34ULL * 2560ULL;
constexpr std::uint64_t known_qsa_attention_bytes = 2560ULL / 32ULL * 34ULL * (12288ULL + 512ULL + 512ULL) +
    6144ULL / 32ULL * 34ULL * 2560ULL;
constexpr std::uint64_t known_qsa_projection_bytes = known_qsa_attention_bytes + 2560ULL / 32ULL * 34ULL * (512ULL + 128ULL);
constexpr std::uint64_t known_qsa_bf16_indexer_projection_bytes = known_qsa_attention_bytes + 2560ULL * (512ULL + 128ULL) * 2ULL;
// Serial: five input and one output projection launch per row; rows: one each.
constexpr std::uint32_t known_qsa_launches_saved(std::uint32_t rows) { return 6U * rows - 2U; }
constexpr std::uint64_t known_gdn_q8_large_projection_bytes = 2560ULL / 32ULL * 34ULL *
    (10240ULL + 6144ULL) + 6144ULL / 32ULL * 34ULL * 2560ULL;
std::pair<std::uint64_t, std::uint64_t> known_window_identities(
    const CompiledKnownPrefixWindow& window, const CompiledSerialExecution& serial) {
    window_require(window.mode.size() == serial.execution.steps.size() &&
        window.r2_variant.size() == window.mode.size(), "known window mode table is incomplete");
    window_require(window.logical_rows==2U || window.logical_rows==4U || window.logical_rows==8U,
        "known window rows are outside the closed table");
    validate_known_window_output_policy(window.output_policy);
    window_require(!serial.execution.steps.empty() && serial.io.steps.size()==serial.execution.steps.size() &&
        serial.execution.steps.back().id==serial.execution.steps.size()-1U &&
        serial.execution.steps.back().op==SerialOp::lm_head &&
        serial.execution.steps.back().reads_state==PersistentState::none &&
        serial.execution.steps.back().writes_state==PersistentState::none &&
        serial.io.steps.back().writes.size()==1U &&
        serial.io.steps.back().writes[0].buffer.buffer_class==ExecutionBufferClass::logits &&
        (window.mode.back()==KnownWindowStepMode::r1_row_loop ||
         (window.mode.back()==KnownWindowStepMode::lm_head_rows && window.output_policy==KnownWindowOutputPolicy::all_rows)) &&
        !serial.bindings.bindings.back().host_or_runtime_step,
        "known-window output policy requires a terminal state-free LM head");
    std::vector<std::uint32_t> expected_logits;
    for(std::uint32_t row=0;row<window.logical_rows;++row)
        if(known_window_produces_logits(window.output_policy,row,window.logical_rows))expected_logits.push_back(row);
    window_require(window.logit_row_indices==expected_logits,"known-window output row mapping differs from its policy");
    auto kernel = fnv_offset; window_hash_text(kernel, window.logical_rows==2U ? "known-window-r2-kernels-v1" : "known-window-rows-kernels-v1");
    if(window.logical_rows!=2U) hash_u64(kernel,window.logical_rows);
    hash_u64(kernel, serial.kernel_plan_identity); hash_u64(kernel, window.mode.size());
    std::uint32_t r2 = 0, loops = 0, runtime = 0, shared_rows = 0, gdn_rows = 0, routed_union=0, hyper_rows=0, qsa_rows=0, gdn_deferred=0, ple_rows=0, head_rows=0, router_rows=0, elementwise_rows=0;
    std::uint64_t hidden_layers = 0;
    for (const auto& step : serial.execution.steps) if (step.layer) hidden_layers = std::max<std::uint64_t>(hidden_layers, *step.layer + 1U);
    std::uint64_t avoided = 0;
    std::uint32_t launches = 0;
    for (std::size_t i = 0; i < window.mode.size(); ++i) {
        const auto& step = serial.execution.steps[i]; const auto& binding = serial.bindings.bindings[i];
        const auto mode = window.mode[i]; const auto variant = window.r2_variant[i];
        if (mode == KnownWindowStepMode::r1_row_loop) {
            window_require(variant == KnownWindowR2Variant::none, "row-loop step has a promoted variant");
            ++loops; if (binding.host_or_runtime_step) ++runtime;
        } else if (mode == KnownWindowStepMode::r2_shared_q8) {
            window_require(step.op == SerialOp::shared_expert && variant == KnownWindowR2Variant::shared_q8_virtual256_r2_v1 &&
                binding.kernel_id && known_shared_id(*binding.kernel_id), "known window shared variant/source ID mismatch"); ++r2;
            window_require(window.logical_rows==2U,"R2 shared mode requires a two-row window");
            avoided += known_shared_projection_bytes; launches += 3;
        } else if(mode==KnownWindowStepMode::shared_rows) {
            window_require(step.op==SerialOp::shared_expert && binding.kernel_id && known_shared_id(*binding.kernel_id) &&
                ((window.logical_rows==4U && variant==KnownWindowR2Variant::shared_q8_virtual256_r4_v1) ||
                 (window.logical_rows==8U && variant==KnownWindowR2Variant::shared_q8_virtual256_r8_v1)),
                "known window shared rows variant/source ID mismatch");
            ++shared_rows;avoided+=known_shared_projection_bytes*(window.logical_rows-1U);
            launches+=3U*(window.logical_rows-1U);
        } else if (mode == KnownWindowStepMode::gdn_rows) {
            window_require(step.op==SerialOp::gated_delta_net && binding.kernel_id && known_gdn_id(*binding.kernel_id) &&
                ((window.logical_rows==4U && variant==KnownWindowR2Variant::gdn_q8_rows4_f32_gates_r1_v1) ||
                 (window.logical_rows==8U && variant==KnownWindowR2Variant::gdn_q8_rows8_f32_gates_r1_v1)),
                "known window GDN rows variant/source ID mismatch");
            ++gdn_rows; avoided+=known_gdn_q8_large_projection_bytes*(window.logical_rows-1U);
            launches+=3U*(window.logical_rows-1U);
        } else if (mode==KnownWindowStepMode::routed_expert_union) {
            const auto profile=binding.kernel_id?routed_union_profile(*binding.kernel_id):std::nullopt;
            window_require(step.op==SerialOp::moe_experts && profile &&
                variant==routed_union_variant(window.logical_rows,*profile),
                "known window routed expert union variant/source ID mismatch");
            ++routed_union;
            // Routes are device results. No static weight-byte or launch-saving
            // claim is inferred before the actual union has been constructed.
        } else if (mode==KnownWindowStepMode::hyper_mix_rows) {
            window_require(hyper_mix_op(step.op) && binding.kernel_id && known_hyper_rows_id(*binding.kernel_id) &&
                variant==hyper_rows_variant(window.logical_rows), "known window HC rows variant/source ID mismatch");
            ++hyper_rows; avoided+=known_hyper_projection_bytes(step.op)*(window.logical_rows-1U);
            launches+=2U*(window.logical_rows-1U);
        } else if(mode==KnownWindowStepMode::qsa_rows) {
            const bool bf16=variant==qsa_rows_variant(window.logical_rows,WindowQsaFormat::q8_bf16_indexer);
            window_require(step.op==SerialOp::indexed_attention && binding.kernel_id && known_qsa_id(*binding.kernel_id) &&
                (bf16 || variant==qsa_rows_variant(window.logical_rows,WindowQsaFormat::all_q8)),
                "known window QSA rows variant/source ID mismatch");
            ++qsa_rows; avoided+=(bf16?known_qsa_bf16_indexer_projection_bytes:known_qsa_projection_bytes)*(window.logical_rows-1U);
            launches+=known_qsa_launches_saved(window.logical_rows);
        } else if(mode==KnownWindowStepMode::gdn_deferred_rows) {
            window_require(step.op==SerialOp::gated_delta_net && binding.kernel_id && known_gdn_id(*binding.kernel_id) &&
                variant==gdn_deferred_variant(window.logical_rows), "known window deferred GDN variant/source ID mismatch");
            ++gdn_deferred; avoided+=known_gdn_q8_large_projection_bytes*(window.logical_rows-1U);
            launches+=3U*(window.logical_rows-1U);
        } else if(mode==KnownWindowStepMode::ple_rows) {
            window_require(step.op==SerialOp::ple && variant==KnownWindowR2Variant::ple_rows_host_staged_v1,
                "known window PLE rows variant/source mismatch");
            ++ple_rows;
        } else if (mode==KnownWindowStepMode::lm_head_rows) {
            window_require(step.op==SerialOp::lm_head && i+1U==window.mode.size() && binding.kernel_id &&
                known_head_id(*binding.kernel_id,serial.vocab_size) && variant==head_rows_variant(window.logical_rows),
                "known window LM head rows variant/source ID mismatch");
            ++head_rows; avoided+=known_head_bytes(serial.vocab_size)*(window.logical_rows-1U);
            launches+=window.logical_rows-1U;
        } else if (mode==KnownWindowStepMode::moe_route_rows) {
            window_require(step.op==SerialOp::moe_route && binding.kernel_id && known_router_rows_id(*binding.kernel_id) &&
                variant==KnownWindowR2Variant::moe_route_rows_v1, "known window router rows variant/source ID mismatch");
            ++router_rows;
        } else if (mode==KnownWindowStepMode::moe_combine_rows || mode==KnownWindowStepMode::inject_rows) {
            window_require(binding.kernel_id && known_elementwise_rows_mode(step,*binding.kernel_id,hidden_layers)==mode && variant==(mode==KnownWindowStepMode::inject_rows ? KnownWindowR2Variant::inject_rows_v1 :
                KnownWindowR2Variant::moe_combine_rows_v1), "known window elementwise rows variant/source ID mismatch");
            ++elementwise_rows;
        } else if (mode == KnownWindowStepMode::r2_gdn_staged) {
            window_require(step.op == SerialOp::gated_delta_net &&
                (variant == KnownWindowR2Variant::gdn_q8_virtual256_r2_v1 ||
                 variant == KnownWindowR2Variant::gdn_q8_rows2_f32_gates_r1_v1) &&
                binding.kernel_id && known_gdn_id(*binding.kernel_id), "known window GDN variant/source ID mismatch"); ++r2;
            if (variant == KnownWindowR2Variant::gdn_q8_virtual256_r2_v1) {
                avoided += known_gdn_projection_bytes*(window.logical_rows/2U); launches += 5U*(window.logical_rows/2U);
            } else {
                avoided += known_gdn_q8_large_projection_bytes*(window.logical_rows/2U); launches += 3U*(window.logical_rows/2U);
            }
        } else throw std::invalid_argument("known window mode is not in the closed table");
        if (binding.kernel_id) {
            const auto adjustment = trellis_window_adjustment(*binding.kernel_id, mode, window.logical_rows);
            avoided -= adjustment.first; launches += adjustment.second;
        }
        hash_u64(kernel, step.id); hash_u64(kernel, static_cast<std::uint64_t>(mode));
        hash_u64(kernel, static_cast<std::uint64_t>(variant));
        hash_u64(kernel, binding.host_or_runtime_step);
        if (binding.kernel_id) window_hash_text(kernel, *binding.kernel_id);
    }
    window_require(r2 == window.r2_steps && loops == window.row_loop_steps && runtime == window.runtime_row_steps &&
        shared_rows==window.shared_rows_steps && gdn_rows==window.gdn_rows_steps &&
        routed_union==window.routed_expert_union_steps && hyper_rows==window.hyper_mix_rows_steps &&
        qsa_rows==window.qsa_rows_steps && gdn_deferred==window.gdn_deferred_rows_steps &&
        ple_rows==window.ple_rows_steps && head_rows==window.lm_head_rows_steps &&
        router_rows==window.router_rows_steps && elementwise_rows==window.combine_inject_rows_steps,
        "known window counters do not match its actual mode table");
    // GSQ-RCO files: the planner counts the actual (non-Q8) operand bytes; the launch count stays closed.
    window_require((avoided == window.logical_projection_weight_bytes_avoided || moe_q4k_a8::q20_a8_enabled()) &&
        launches == window.projection_launches_saved,
        "known window logical projection counters differ from the closed operand shapes");
    hash_u64(kernel, avoided); hash_u64(kernel, launches);
    // Preserve every byte of the default identity recipe. Nondefault output
    // intent receives its own organization/lineage domain, not a serial ID.
    if(window.output_policy!=KnownWindowOutputPolicy::all_rows) {
        window_hash_text(kernel,"known-window-output-policy-v1");
        hash_u64(kernel,static_cast<std::uint64_t>(window.output_policy));
        hash_u64(kernel,window.logit_row_indices.size());
        for(auto row:window.logit_row_indices)hash_u64(kernel,row);
    }
    auto execution = fnv_offset; window_hash_text(execution, window.logical_rows==2U ? "known-window-r2-v1" : "known-window-rows-v1");
    if(window.logical_rows!=2U) hash_u64(execution,window.logical_rows);
    hash_u64(execution, window_serial_fingerprint(serial)); hash_u64(execution, kernel);
    hash_u64(execution, window.manifest_layout_identity); hash_u64(execution, window.tensor_access_plan_identity);
    window_hash_arena(execution, window.arena.physical);
    for (auto stride : window.arena.row_stride) hash_u64(execution, stride);
    for (const auto& row : window.arena.row_view) window_hash_arena(execution, row);
    window_require(kernel != 0 && execution != 0 && execution != serial.execution_identity,
        "known window identities are empty or collide with the serial identity");
    return {execution, kernel};
}
} // namespace

CompiledKnownPrefixWindow compile_known_prefix_window(
    const ModelContract& model, const CompiledSerialExecution& serial,
    const ModelManifest& manifest, const ExecutionTensorAccessPlan& tensor_access,
    const KnownWindowRequest& request) {
    if (const auto error = model.validate(); !error.empty()) throw std::invalid_argument(error);
    window_require(model.hidden_layers == 48 && model.hidden_size == 2560 && model.hc_count == 4 &&
        model.hc_lowrank == 320 && model.moe_intermediate_size == 640 && model.shared_expert_intermediate_size == 640 &&
        model.gdn_key_heads == 16 && model.gdn_key_head_dim == 128 && model.gdn_value_heads == 48 &&
        model.gdn_value_head_dim == 128 && model.full_attention_layers == 12 && model.gdn_layers == 36 &&
        model.attention_heads == 24 && model.kv_heads == 2 && model.head_dim == 256 &&
        model.num_experts == 512 && model.num_experts_per_token == 10 &&
        model.bytes_bf16 == 2 && model.bytes_fp32 == 4,
        "known window currently supports the canonical R2 model only");
    // The serial limit follows the resolved rotary identity, independently of
    // the checkpoint's native maximum and the allocated session context.
    window_require(serial.execution_identity && serial.kernel_plan_identity && serial.logical_rows == 1 &&
        serial.request_concurrency == 1 && !serial.device_graph_inputs &&
        serial.vocab_size == model.vocab_size && serial.max_position_embeddings == model.effective_max_positions() &&
        serial.bindings.missing_steps == 0 && serial.bindings.bindings.size() == serial.execution.steps.size(),
        "known window requires a complete eager C1 serial source");
    window_require(manifest.layout_identity && tensor_access.plan_identity &&
        tensor_access.manifest_layout_identity == manifest.layout_identity,
        "known window manifest/tensor access identities are missing or inconsistent");
    const auto expected = build_serial_decode_execution_plan(model);
    window_require(expected.steps.size() == serial.execution.steps.size(), "known window source topology is incomplete");
    validate_serial_execution_io_plan(serial.execution, serial.io);
    for (std::size_t i = 0; i < expected.steps.size(); ++i) {
        const auto& a = expected.steps[i]; const auto& b = serial.execution.steps[i];
        const auto& binding = serial.bindings.bindings[i];
        window_require(a.id == b.id && a.op == b.op && a.layer == b.layer && a.dependencies == b.dependencies &&
            a.reads_state == b.reads_state && a.writes_state == b.writes_state &&
            a.dynamic_shape == b.dynamic_shape && a.graph_safe_candidate == b.graph_safe_candidate &&
            binding.step_id == b.id && binding.op == b.op,
            "known window source step/binding does not match the canonical topology");
        window_require(binding.host_or_runtime_step == (b.op == SerialOp::position_prepare) &&
            (binding.host_or_runtime_step ? !binding.kernel_id : binding.kernel_id && !binding.kernel_id->empty()),
            "known window source runtime/kernel binding is invalid");
        for (const auto* uses : {&serial.io.steps[i].reads, &serial.io.steps[i].writes, &serial.io.steps[i].temporaries})
            for (auto use : *uses) window_require(find_arena_allocation(serial.arena, use) != nullptr,
                "known window serial IO references an unallocated use");
    }
    window_require(request.logical_rows==2U || request.logical_rows==4U || request.logical_rows==8U,
        "known window rows must be two, four or eight");
    CompiledKnownPrefixWindow output;
    output.logical_rows=request.logical_rows;
    validate_known_window_output_policy(request.output_policy);
    output.output_policy=request.output_policy;
    for(std::uint32_t row=0;row<output.logical_rows;++row)
        if(known_window_produces_logits(output.output_policy,row,output.logical_rows))output.logit_row_indices.push_back(row);
    output.serial_execution_identity = serial.execution_identity;
    output.serial_kernel_plan_identity = serial.kernel_plan_identity;
    output.manifest_layout_identity = manifest.layout_identity;
    output.tensor_access_plan_identity = tensor_access.plan_identity;
    output.arena = build_known_window_arena(serial.arena, serial.arena.alignment,output.logical_rows);
    output.mode.assign(serial.execution.steps.size(), KnownWindowStepMode::r1_row_loop);
    output.r2_variant.assign(serial.execution.steps.size(), KnownWindowR2Variant::none);
    for (std::size_t i = 0; i < output.mode.size(); ++i) {
        const auto& step = serial.execution.steps[i]; const auto& binding = serial.bindings.bindings[i];
        const auto avoided_before=output.logical_projection_weight_bytes_avoided;
        const auto launches_before=output.projection_launches_saved;
        if (request.r2_shared_expert && step.op == SerialOp::shared_expert && binding.kernel_id &&
            known_shared_id(*binding.kernel_id) && window_shared_types(manifest, tensor_access, step)) {
            output.mode[i] = output.logical_rows==2U ? KnownWindowStepMode::r2_shared_q8 : KnownWindowStepMode::shared_rows;
            output.r2_variant[i] = output.logical_rows==2U ? KnownWindowR2Variant::shared_q8_virtual256_r2_v1 :
                output.logical_rows==4U ? KnownWindowR2Variant::shared_q8_virtual256_r4_v1 : KnownWindowR2Variant::shared_q8_virtual256_r8_v1;
            for (const auto suffix : {"ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"})
                output.logical_projection_weight_bytes_avoided +=
                    window_tensor(manifest, tensor_access, step.id, step.op, *step.layer, suffix).byte_size*(output.logical_rows-1U);
            output.projection_launches_saved += 3U*(output.logical_rows-1U);
        } else if (request.r2_gdn && step.op == SerialOp::gated_delta_net && binding.kernel_id &&
            known_gdn_id(*binding.kernel_id)) {
            const auto format = window_gdn_format(manifest, tensor_access, step, request.r2_gdn_f32_gates);
            if (format != WindowGdnFormat::unsupported) {
                const bool mixed = format == WindowGdnFormat::q8_large_f32_gates;
                const bool wide = mixed && output.logical_rows!=2U;
                output.mode[i] = wide ? KnownWindowStepMode::gdn_rows : KnownWindowStepMode::r2_gdn_staged;
                output.r2_variant[i] = wide ? (output.logical_rows==4U ?
                    KnownWindowR2Variant::gdn_q8_rows4_f32_gates_r1_v1 : KnownWindowR2Variant::gdn_q8_rows8_f32_gates_r1_v1) :
                    mixed ? KnownWindowR2Variant::gdn_q8_rows2_f32_gates_r1_v1 : KnownWindowR2Variant::gdn_q8_virtual256_r2_v1;
                const auto saved_copies = wide ? output.logical_rows-1U : output.logical_rows/2U;
                for (const auto suffix : {"attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight", "ssm_out.weight"}) {
                    if (mixed && (std::string_view(suffix) == "ssm_beta.weight" ||
                                  std::string_view(suffix) == "ssm_alpha.weight")) continue;
                    output.logical_projection_weight_bytes_avoided +=
                        window_tensor(manifest, tensor_access, step.id, step.op, *step.layer, suffix).byte_size*saved_copies;
                }
                output.projection_launches_saved += (mixed ? 3U : 5U)*saved_copies;
            }
        }
        const auto qsa_format=request.qsa_rows && step.op==SerialOp::indexed_attention && binding.kernel_id &&
            known_qsa_id(*binding.kernel_id) ? window_qsa_format(manifest,tensor_access,step) : WindowQsaFormat::unsupported;
        if(qsa_format!=WindowQsaFormat::unsupported) {
            output.mode[i]=KnownWindowStepMode::qsa_rows;
            output.r2_variant[i]=qsa_rows_variant(output.logical_rows,qsa_format);
            for(const auto& item:known_qsa_matrices)
                output.logical_projection_weight_bytes_avoided+=
                    window_tensor(manifest,tensor_access,step.id,step.op,*step.layer,item.first).byte_size*(output.logical_rows-1U);
            output.projection_launches_saved+=known_qsa_launches_saved(output.logical_rows);
        }
        if(request.deferred_gdn && step.op==SerialOp::gated_delta_net && binding.kernel_id &&
            known_gdn_id(*binding.kernel_id) && window_gdn_format(manifest,tensor_access,step,true)!=WindowGdnFormat::unsupported) {
            if(output.mode[i]==KnownWindowStepMode::r2_gdn_staged || output.mode[i]==KnownWindowStepMode::gdn_rows) {
                // Replace the checkpointed wide GDN entry and its claimed savings.
                const bool mixed=window_gdn_format(manifest,tensor_access,step,request.r2_gdn_f32_gates)==WindowGdnFormat::q8_large_f32_gates;
                const bool wide=mixed && output.logical_rows!=2U;
                const auto saved=wide ? output.logical_rows-1U : output.logical_rows/2U;
                for(const auto suffix:{"attn_qkv.weight","attn_gate.weight","ssm_beta.weight","ssm_alpha.weight","ssm_out.weight"}) {
                    if(mixed && (std::string_view(suffix)=="ssm_beta.weight" || std::string_view(suffix)=="ssm_alpha.weight")) continue;
                    output.logical_projection_weight_bytes_avoided-=
                        window_tensor(manifest,tensor_access,step.id,step.op,*step.layer,suffix).byte_size*saved;
                }
                output.projection_launches_saved-=(mixed ? 3U : 5U)*saved;
            }
            output.mode[i]=KnownWindowStepMode::gdn_deferred_rows;
            output.r2_variant[i]=gdn_deferred_variant(output.logical_rows);
            output.logical_projection_weight_bytes_avoided+=known_gdn_q8_large_projection_bytes*(output.logical_rows-1U);
            output.projection_launches_saved+=3U*(output.logical_rows-1U);
        }
        if(request.rows_ple && step.op==SerialOp::ple) {
            output.mode[i]=KnownWindowStepMode::ple_rows;
            output.r2_variant[i]=KnownWindowR2Variant::ple_rows_host_staged_v1;
        }
        if(const auto profile=request.routed_expert_union && step.op==SerialOp::moe_experts && binding.kernel_id ?
               routed_union_profile(*binding.kernel_id) : std::nullopt;
           profile && window_routed_union_types(manifest,tensor_access,step,*profile)) {
            output.mode[i]=KnownWindowStepMode::routed_expert_union;
            output.r2_variant[i]=routed_union_variant(output.logical_rows,*profile);
        }
        if(request.rows_hyper_mix && hyper_mix_op(step.op) && binding.kernel_id && known_hyper_rows_id(*binding.kernel_id)) {
            output.mode[i]=KnownWindowStepMode::hyper_mix_rows;
            output.r2_variant[i]=hyper_rows_variant(output.logical_rows);
            output.logical_projection_weight_bytes_avoided+=known_hyper_projection_bytes(step.op)*(output.logical_rows-1U);
            output.projection_launches_saved+=2U*(output.logical_rows-1U);
        }
        if(request.lm_head_rows && output.output_policy==KnownWindowOutputPolicy::all_rows &&
            i+1U==output.mode.size() && step.op==SerialOp::lm_head && binding.kernel_id &&
            known_head_id(*binding.kernel_id,serial.vocab_size) &&
            window_head_types(manifest,tensor_access,step,serial.vocab_size)) {
            output.mode[i]=KnownWindowStepMode::lm_head_rows;
            output.r2_variant[i]=head_rows_variant(output.logical_rows);
            output.logical_projection_weight_bytes_avoided+=known_head_bytes(serial.vocab_size)*(output.logical_rows-1U);
            output.projection_launches_saved+=output.logical_rows-1U;
        }
        if(request.router_rows && step.op==SerialOp::moe_route && binding.kernel_id && known_router_rows_id(*binding.kernel_id)) {
            output.mode[i]=KnownWindowStepMode::moe_route_rows;
            output.r2_variant[i]=KnownWindowR2Variant::moe_route_rows_v1;
        }
        if(request.combine_inject_rows && binding.kernel_id) {
            const auto mode=known_elementwise_rows_mode(step,*binding.kernel_id,model.hidden_layers);
            if(mode!=KnownWindowStepMode::r1_row_loop) {
                output.mode[i]=mode;
                output.r2_variant[i]=mode==KnownWindowStepMode::inject_rows ? KnownWindowR2Variant::inject_rows_v1 :
                    KnownWindowR2Variant::moe_combine_rows_v1;
            }
        }
        // Select before counters, identities, row clones or any native owner setup.
        // In particular the five-Q8 paired GDN gate projection is not R1 math;
        // mixed gates and deferred prefix rows already have admitted entries.
        const auto selected=known_window_select_serial_rows(request.require_serial_rows,output.mode[i],output.r2_variant[i]);
        if(selected.mode!=output.mode[i] || selected.variant!=output.r2_variant[i]) {
            output.mode[i]=selected.mode; output.r2_variant[i]=selected.variant;
            output.logical_projection_weight_bytes_avoided=avoided_before;
            output.projection_launches_saved=launches_before;
        }
        if (binding.kernel_id) {
            const auto adjustment = trellis_window_adjustment(*binding.kernel_id, output.mode[i], output.logical_rows);
            output.logical_projection_weight_bytes_avoided -= adjustment.first;
            output.projection_launches_saved += adjustment.second;
        }
        if (output.mode[i] == KnownWindowStepMode::r1_row_loop) {
            ++output.row_loop_steps; if (binding.host_or_runtime_step) ++output.runtime_row_steps;
        } else if(output.mode[i]==KnownWindowStepMode::shared_rows) ++output.shared_rows_steps;
        else if(output.mode[i]==KnownWindowStepMode::gdn_rows) ++output.gdn_rows_steps;
        else if(output.mode[i]==KnownWindowStepMode::routed_expert_union) ++output.routed_expert_union_steps;
        else if(output.mode[i]==KnownWindowStepMode::hyper_mix_rows) ++output.hyper_mix_rows_steps;
        else if(output.mode[i]==KnownWindowStepMode::qsa_rows) ++output.qsa_rows_steps;
        else if(output.mode[i]==KnownWindowStepMode::gdn_deferred_rows) ++output.gdn_deferred_rows_steps;
        else if(output.mode[i]==KnownWindowStepMode::ple_rows) ++output.ple_rows_steps;
        else if(output.mode[i]==KnownWindowStepMode::lm_head_rows) ++output.lm_head_rows_steps;
        else if(output.mode[i]==KnownWindowStepMode::moe_route_rows) ++output.router_rows_steps;
        else if(output.mode[i]==KnownWindowStepMode::moe_combine_rows || output.mode[i]==KnownWindowStepMode::inject_rows)
            ++output.combine_inject_rows_steps;
        else ++output.r2_steps;
    }
    window_require(!request.require_serial_rows || known_window_policy_serial_rows(output),
        "known window has no serial-row recipe for this size");
    const auto identities = known_window_identities(output, serial);
    output.window_execution_identity = identities.first; output.window_kernel_plan_identity = identities.second;
    output.row_plan.assign(output.logical_rows,serial);
    for (std::size_t row = 0; row < output.row_plan.size(); ++row) {
        output.row_plan[row].execution_identity = output.window_execution_identity;
        output.row_plan[row].arena = output.arena.row_view[row];
        output.row_plan[row].whole_plan_graph_safe = false;
    }
    validate_compiled_known_prefix_window(output, serial);
    return output;
}

void validate_compiled_known_prefix_window(
    const CompiledKnownPrefixWindow& window, const CompiledSerialExecution& serial) {
    window_require(serial.logical_rows == 1 && serial.request_concurrency == 1 && !serial.device_graph_inputs &&
        serial.execution_identity != 0 && serial.kernel_plan_identity != 0 &&
        serial.bindings.bindings.size() == serial.execution.steps.size(), "known window source is not a complete serial plan");
    window_require(window.serial_execution_identity == serial.execution_identity &&
        window.serial_kernel_plan_identity == serial.kernel_plan_identity && window.manifest_layout_identity &&
        window.tensor_access_plan_identity, "known window serial/provenance identities differ");
    window_require((window.logical_rows==2U || window.logical_rows==4U || window.logical_rows==8U) &&
        window.row_plan.size()==window.logical_rows && window.arena.row_view.size()==window.logical_rows,
        "known window row clone count differs from its closed geometry");
    const auto arena = build_known_window_arena(serial.arena, serial.arena.alignment,window.logical_rows);
    window_require(window_same_arena(arena.physical, window.arena.physical) &&
        arena.row_stride == window.arena.row_stride, "known window physical allocation/stride mutated");
    for (std::size_t row = 0; row < window.row_plan.size(); ++row) {
        window_require(window_same_arena(arena.row_view[row], window.arena.row_view[row]) &&
            window_same_arena(arena.row_view[row], window.row_plan[row].arena), "known window row views differ from the serial-derived layout");
        auto restored = window.row_plan[row]; restored.execution_identity = serial.execution_identity;
        restored.arena = serial.arena; restored.whole_plan_graph_safe = serial.whole_plan_graph_safe;
        window_require(!window.row_plan[row].whole_plan_graph_safe &&
            window.row_plan[row].execution_identity == window.window_execution_identity &&
            window_serial_fingerprint(restored) == window_serial_fingerprint(serial),
            "known window row clone changed serial topology, kernel IDs, IO or metadata");
    }
    const auto identities = known_window_identities(window, serial);
    window_require(window.window_execution_identity == identities.first &&
        window.window_kernel_plan_identity == identities.second, "known window identities no longer match its content");
}

void validate_known_prefix_window_formats(
    const CompiledKnownPrefixWindow& window, const CompiledSerialExecution& serial,
    const ModelManifest& manifest, const ExecutionTensorAccessPlan& tensor_access) {
    validate_compiled_known_prefix_window(window, serial);
    window_require(manifest.layout_identity == window.manifest_layout_identity &&
        tensor_access.plan_identity == window.tensor_access_plan_identity &&
        tensor_access.manifest_layout_identity == manifest.layout_identity,
        "known window format provenance differs from its compiled manifest/access plan");
    for (std::size_t i = 0; i < window.mode.size(); ++i) {
        const auto& step = serial.execution.steps[i];
        if (window.mode[i] == KnownWindowStepMode::r2_shared_q8 || window.mode[i] == KnownWindowStepMode::shared_rows) {
            window_require(window_shared_types(manifest, tensor_access, step),
                "known window shared variant differs from the selected tensor formats");
        } else if(window.mode[i]==KnownWindowStepMode::qsa_rows) {
            const auto format=window_qsa_format(manifest,tensor_access,step);
            window_require(format!=WindowQsaFormat::unsupported && window.r2_variant[i]==qsa_rows_variant(window.logical_rows,format),
                "known window QSA rows differ from the selected tensor formats");
        } else if(window.mode[i]==KnownWindowStepMode::lm_head_rows) {
            window_require(window_head_types(manifest,tensor_access,step,serial.vocab_size),
                "known window LM head rows differ from the selected head format");
        } else if(window.mode[i]==KnownWindowStepMode::routed_expert_union) {
            const auto& id=serial.bindings.bindings[i].kernel_id;
            const auto profile=id?routed_union_profile(*id):std::nullopt;
            window_require(profile && window_routed_union_types(manifest,tensor_access,step,*profile) &&
                window.r2_variant[i]==routed_union_variant(window.logical_rows,*profile),
                "known window routed union differs from the selected tensor formats");
        } else if(window.mode[i]==KnownWindowStepMode::gdn_deferred_rows) {
            window_require(window_gdn_format(manifest,tensor_access,step,true)!=WindowGdnFormat::unsupported,
                "known window deferred GDN differs from the selected tensor formats");
        } else if (window.mode[i] == KnownWindowStepMode::r2_gdn_staged || window.mode[i] == KnownWindowStepMode::gdn_rows) {
            const auto expected = window.r2_variant[i] == KnownWindowR2Variant::gdn_q8_virtual256_r2_v1 ?
                WindowGdnFormat::all_q8 : WindowGdnFormat::q8_large_f32_gates;
            window_require(window_gdn_format(manifest, tensor_access, step, true) == expected,
                "known window GDN variant differs from the selected tensor formats");
        }
    }
}

std::string known_window_arithmetic_witness(const CompiledKnownPrefixWindow& window,
    bool serial_rows_required, bool prefill, bool gsq, KnownWindowFolds folds) {
    static constexpr std::array<const char*, known_window_r2_variant_slots> names{
        "none",
        "shared_q8_virtual256_r2_v1",
        "gdn_q8_virtual256_r2_v1",
        "gdn_q8_rows2_f32_gates_r1_v1",
        "shared_q8_virtual256_r4_v1",
        "shared_q8_virtual256_r8_v1",
        "gdn_q8_rows4_f32_gates_r1_v1",
        "gdn_q8_rows8_f32_gates_r1_v1",
        "routed_q51_union_rows2_v1",
        "routed_q51_union_rows4_v1",
        "routed_q51_union_rows8_v1",
        "hyper_mix_rows2_v1",
        "hyper_mix_rows4_v1",
        "hyper_mix_rows8_v1",
        "qsa_q8_rows2_r1_v1",
        "qsa_q8_rows4_r1_v1",
        "qsa_q8_rows8_r1_v1",
        "gdn_deferred_rows2_v1",
        "gdn_deferred_rows4_v1",
        "gdn_deferred_rows8_v1",
        "ple_rows_host_staged_v1",
        "qsa_q8_bf16idx_rows2_r1_v1",
        "qsa_q8_bf16idx_rows4_r1_v1",
        "qsa_q8_bf16idx_rows8_r1_v1",
        "lm_head_q8_prefetch8_rows2_v1",
        "lm_head_q8_prefetch8_rows4_v1",
        "lm_head_q8_prefetch8_rows8_v1",
        "moe_route_rows_v1",
        "moe_combine_rows_v1",
        "inject_rows_v1",
        "routed_a8_union_rows2_v1",
        "routed_a8_union_rows4_v1",
        "routed_a8_union_rows8_v1",
    };
    std::array<std::uint32_t, known_window_r2_variant_slots> counts{};
    std::uint32_t unknown=0;
    for(const auto v:window.r2_variant) {
        const auto index=static_cast<std::uint32_t>(v);
        if(index<counts.size()) ++counts[index]; else ++unknown;
    }
    std::ostringstream out;
    out << "{\"rows\":" << window.logical_rows
        << ",\"serial_rows_required\":" << (serial_rows_required?"true":"false")
        << ",\"policy_selection_admitted\":" << (known_window_policy_selection_admitted(window,prefill,gsq,folds)?"true":"false")
        << ",\"execution_identity\":" << window.window_execution_identity
        << ",\"kernel_plan_identity\":" << window.window_kernel_plan_identity
        << ",\"folds\":{\"inject_norm\":" << (folds.inject_norm?"true":"false")
        << ",\"combine_inject\":" << (folds.combine_inject?"true":"false")
        << ",\"commit_combine\":" << (folds.commit_combine?"true":"false") << "},\"variants\":{";
    bool comma=false;
    for(std::size_t i=0;i<counts.size();++i) if(counts[i]) {
        out << (comma?",":"") << '"' << names[i] << "\":" << counts[i]; comma=true;
    }
    out << "},\"unknown_variants\":" << unknown << '}';
    return out.str();
}

}  // namespace strix
