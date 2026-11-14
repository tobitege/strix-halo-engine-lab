#include "strix/compact_stage_images.hpp"
#include "strix/qpf2k_defaults.hpp"
#include "strix/compiled_speculative_verify.hpp"
#include "strix/known_prefill_limits.hpp"
#include "../src/hip_gdn_identity.hpp"

#include "strix/execution_binding.hpp"
#include "strix/arithmetic_lineage.hpp"
#include "strix/shared_expert_prefetch.hpp"
#include "strix/dispatch_resources.hpp"
#include "strix/runtime_launchers.hpp"
#include "strix/speculative_verify_runtime.hpp"
#include "strix/known_prefix_prefill.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <optional>
#include <functional>
#include <limits>

#include <set>
#include <string>
#include <string_view>
#include <span>
#include <vector>
#include <utility>

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::kernels::KernelVariant make_variant(
    std::string id,
    strix::kernels::OperatorFamily family,
    const strix::kernels::ShapeClass& shape,
    double measured_seconds = 0.0) {
    strix::kernels::KernelVariant variant;
    variant.id = std::move(id);
    variant.family = family;
    variant.phase =
        strix::kernels::ExecutionPhase::
            speculative_verify;
    variant.shape = shape;
    variant.wave =
        strix::kernels::WaveSize::wave32;
    variant.resources.workgroup_threads = 256;
    variant.numerics.input_precision = "bf16";
    variant.numerics.accumulator_precision = "fp32";
    variant.numerics.output_precision = "bf16";
    variant.numerics.row_invariant = true;
    variant.numerics.deterministic = true;
    variant.qualification =
        strix::kernels::QualificationStatus::
            oracle_matched;
    variant.graph_safe = true;
    variant.input_layout = "verify";
    variant.output_layout = "verify";
    variant.compiler_identity = "fixture";
    variant.measured_seconds =
        measured_seconds;
    return variant;
}

strix::kernels::KernelRegistry make_registry(
    const strix::ModelContract& model,
    std::uint32_t depth,
    std::string suffix = "", bool parallel_router = false, bool ranked_router = false, bool wave_router = false) {
    strix::kernels::KernelRegistry registry;
    const auto verify =
        strix::build_speculative_verify_plan(
            model,
            depth);

    strix::ExecutionBindingRequest shape_request;
    shape_request.logical_rows = depth;
    shape_request.request_concurrency = 1;

    std::set<std::string> unique;
    for (const auto& step :
         verify.steps) {
        if (step.op ==
            strix::SerialOp::position_prepare) {
            continue;
        }
        const auto family =
            strix::operator_family_for(
                step.op);
        if (!family.has_value()) {
            continue;
        }
        const auto shape =
            strix::execution_shape_for(
                model,
                step.op,
                shape_request);
        const auto key =
            std::to_string(
                static_cast<unsigned>(*family)) +
            "." +
            std::to_string(
                shape.auxiliary_size);
        if (!unique.insert(key).second) {
            continue;
        }
        registry.add(
            make_variant(
                "verify." + key + suffix +
                    (step.op == strix::SerialOp::moe_route && wave_router ? ".wave.parallel" :
                     step.op == strix::SerialOp::moe_route && ranked_router ? ".ranked.parallel" :
                     step.op == strix::SerialOp::moe_route && parallel_router ? ".parallel" : ""),
                *family,
                shape));
    }
    return registry;
}


struct KnownFixture {
    strix::CompiledSerialExecution serial;
    strix::ModelManifest manifest;
    strix::ExecutionTensorAccessPlan access;
};
constexpr std::string_view known_head_wave8 = "hip.lm_head.bf16.h2560.v248320.r1.c1.prefetch8-wave8-v1.virtual256.q8";
constexpr std::string_view known_hyper_base="hip.hyper_mix.bf16.h2560.hc4.lr320.r1.c1.parallel.staged"
    ".fused-injection-v5.typed-q8-f32-v1.prefetch-rows8-v2.prefetch-q8-output-v1";
bool hyper_op(strix::SerialOp op) {
    return op==strix::SerialOp::attn_hyper_mix || op==strix::SerialOp::mlp_hyper_mix || op==strix::SerialOp::final_hyper_mix;
}
// Routed experts of the union fixture: 0 = float profile with Q4_K/Q5_1 on every
// layer, 1 = float profile id over the canonical census formats, 2 = A8 profile.
unsigned routed_fixture_profile = 0U;
KnownFixture make_known_fixture(const strix::ModelContract& model, std::uint32_t gate_type = 8U,
    bool routed_union = false, bool hyper = false, bool qsa = false, std::uint32_t indexer_type = 8U,
    std::uint32_t head_type = 0xffffffffU) {
    KnownFixture fixture;
    fixture.serial.execution = strix::build_serial_decode_execution_plan(model);
    fixture.serial.io = strix::build_serial_execution_io_plan(fixture.serial.execution);
    strix::ExecutionBufferRequest buffers;
    buffers.q4k_a8g32_gate_up = true;
    buffers.grouped_exact_q51_down = true;
    buffers.q51_slot_parallel_down = true;
    fixture.serial.arena = strix::build_serial_execution_arena_plan(model, buffers,
        fixture.serial.execution, fixture.serial.io);
    fixture.serial.execution_identity = 0x51515231;
    fixture.serial.kernel_plan_identity = 0x51514b31;
    fixture.serial.vocab_size = model.vocab_size;
    fixture.serial.max_position_embeddings = model.max_position_embeddings;
    fixture.manifest.layout_identity = 17;
    fixture.access.manifest_layout_identity = 17;
    fixture.access.plan_identity = 29;
    std::uint64_t offset = 256;
    for (const auto& step : fixture.serial.execution.steps) {
        strix::ExecutionKernelBinding binding;
        binding.step_id = step.id; binding.op = step.op;
        binding.family = strix::operator_family_for(step.op);
        if (step.op == strix::SerialOp::position_prepare) {
            binding.host_or_runtime_step = true;
            ++fixture.serial.bindings.host_or_runtime_steps;
        } else {
            binding.kernel_id = "fixture.serial." + std::to_string(step.id);
            ++fixture.serial.bindings.bound_steps;
        }
        strix::ExecutionTensorAccess tensors;
        tensors.step_id = step.id; tensors.op = step.op;
        const auto tensor = [&](std::string suffix, std::vector<std::uint64_t> dimensions, std::uint32_t type) {
            strix::ManifestTensor value;
            value.name = "blk." + std::to_string(*step.layer) + "." + suffix;
            value.role = suffix.ends_with("_exps.weight") ? strix::TensorRole::moe_expert :
                suffix.starts_with("attn_") ? strix::TensorRole::attention_projection :
                suffix.starts_with("ssm_") ? strix::TensorRole::recurrent_state :
                suffix.starts_with("indexer.") ? strix::TensorRole::sparse_index : strix::TensorRole::dense_projection;
            value.layer = step.layer; value.dimensions = std::move(dimensions); value.ggml_type = type;
            value.absolute_offset = offset;
            std::uint64_t elements = 1;
            for (const auto extent : value.dimensions) elements *= extent;
            value.byte_size = type == 8 ? elements / 32U * 34U :
                type==12U ? elements/256U*144U : type==13U ? elements/256U*176U : type==7U ? elements/32U*24U : type==30U ? elements * 2U : elements * 4U;
            offset += (value.byte_size + 255U) / 256U * 256U;
            tensors.tensor_indices.push_back(static_cast<std::uint32_t>(fixture.manifest.tensors.size()));
            fixture.manifest.tensors.push_back(std::move(value));
        };
        if (step.op == strix::SerialOp::shared_expert) {
            binding.kernel_id = "hip.shared_expert.bf16.h2560.m640.r1.c1";
            tensor("ffn_gate_shexp.weight", {2560,640}, 8);
            tensor("ffn_up_shexp.weight", {2560,640}, 8);
            tensor("ffn_down_shexp.weight", {640,2560}, 8);
            tensor("ffn_gate_inp_shexp.weight", {2560}, 0);
        } else if(step.op==strix::SerialOp::moe_experts && routed_union) {
            binding.kernel_id=routed_fixture_profile==2U ?
                "hip.moe_experts.bf16.h2560.m640.e512.k10.r1.c1.batched-slot-activation-v1.deferred-status-v1"
                ".q5k-a8g32-gateup-v1.q51-a8g32-down-v1.q80-a8g32-down-v1.mixed-a8-down-v1.q4k-a8g32-gateup-v2" :
                "hip.moe_experts.bf16.h2560.m640.e512.k10.r1.c1.deferred-status-v1"
                ".q51-slot-parallel-v1.grouped-exact-q51-down-v1.q4k-a8g32-gateup-v2";
            const auto layer=step.layer.value_or(0U);
            const bool census=routed_fixture_profile!=0U;
            const bool q80=census && (layer==2U || layer==4U || layer==30U || layer==46U || layer==47U);
            const std::uint32_t gate=census && layer==2U ? 13U : 12U;
            tensor("ffn_gate_exps.weight",{2560,640,512},gate);
            tensor("ffn_up_exps.weight",{2560,640,512},gate);
            tensor("ffn_down_exps.weight",{640,2560,512},q80 ? 8U : 7U);
        } else if (step.op == strix::SerialOp::gated_delta_net) {
            strix::HipGdnLauncherConfig gdn;
            gdn.model = model;
            gdn.virtual_q8_input_projections = gdn.prefetch_q8_input_projections = true;
            gdn.wave8_q8_input_projections = gdn.wave8_q8_output_projection = true;
            gdn.virtual_q8_output_projection = true;
            binding.kernel_id = strix::hip_gdn_identity::kernel_id({}, gdn);
            tensor("attn_qkv.weight", {2560,10240}, 8);
            tensor("attn_gate.weight", {2560,6144}, 8);
            tensor("ssm_beta.weight", {2560,48}, gate_type);
            tensor("ssm_alpha.weight", {2560,48}, gate_type);
            tensor("ssm_out.weight", {6144,2560}, 8);
        } else if(step.op==strix::SerialOp::indexed_attention && qsa) {
            binding.kernel_id="hip.indexed_attention.bf16.h2560.q24x256.kv2.idx4x128.b2048.r1.c1.scalar"
                ".wave8-q8-qsa-output-v1.prefetch8.virtual256.q8-qsa-in";
            tensor("attn_q.weight",{2560,12288},8); tensor("attn_k.weight",{2560,512},8);
            tensor("attn_v.weight",{2560,512},8); tensor("attn_output.weight",{6144,2560},8);
            tensor("attn_q_norm.weight",{256},0); tensor("attn_k_norm.weight",{256},0);
            tensor("indexer.q_proj.weight",{2560,512},indexer_type); tensor("indexer.k_proj.weight",{2560,128},indexer_type);
            tensor("indexer.q_norm.weight",{128},0); tensor("indexer.k_norm.weight",{128},0);
        } else if (step.op == strix::SerialOp::lm_head && head_type != 0xffffffffU) {
            binding.kernel_id = std::string(known_head_wave8);
            strix::ManifestTensor head;
            head.name = "output.weight"; head.role = strix::TensorRole::output_head;
            head.dimensions = {2560, model.vocab_size}; head.ggml_type = head_type; head.absolute_offset = offset;
            head.byte_size = head_type == 8U ? 2560U / 32U * 34U * model.vocab_size : 2560U * 2U * model.vocab_size;
            offset += (head.byte_size + 255U) / 256U * 256U;
            tensors.tensor_indices.push_back(static_cast<std::uint32_t>(fixture.manifest.tensors.size()));
            fixture.manifest.tensors.push_back(std::move(head));
        } else if (step.op == strix::SerialOp::ple) {
            // Only metadata is retained. A start-fault backend below never
            // accesses this table or claims a real PLE numerical baseline.
            const auto hash = strix::build_ple_hash_contract(model);
            strix::ManifestTensor table;
            table.name = "fixture.ngram_table.weight";
            table.role = strix::TensorRole::ngram_table;
            table.dimensions = {hash.head_dim, hash.minimum_table_rows};
            table.ggml_type = 1; table.absolute_offset = offset;
            table.byte_size = hash.row_bytes * hash.minimum_table_rows;
            tensors.tensor_indices.push_back(static_cast<std::uint32_t>(fixture.manifest.tensors.size()));
            fixture.manifest.tensors.push_back(std::move(table));
        }
        // Mixed split-K / lane-chain selections; every typed prefetched Q8 mixer is eligible.
        if (hyper && hyper_op(step.op))
            binding.kernel_id = std::string(known_hyper_base) + (step.id % 3U ? ".splitk-hc-proj-fp32-v1" : "") +
                (step.id % 2U ? ".lane-chain-v1" : "");
        fixture.serial.bindings.bindings.push_back(std::move(binding));
        if (!tensors.tensor_indices.empty()) fixture.access.accesses.push_back(std::move(tensors));
    }
    return fixture;
}
// Host flow witnesses, not numerical/GPU qualification. Canonical topology and
// real dispatch/resource resolution surround deliberately tiny state recurrences.
constexpr std::uint64_t recording_qsa_identity = 0x5445535451534131ULL;
constexpr std::array<strix::PersistentState, 7> mock_state_flags{
    strix::PersistentState::attention_kv, strix::PersistentState::recurrent_matrix,
    strix::PersistentState::convolution_history, strix::PersistentState::convolution_history,
    strix::PersistentState::sparse_index, strix::PersistentState::ngram_history,
    strix::PersistentState::position_history};
constexpr std::array<strix::SessionStateComponent, 7> mock_components{
    strix::SessionStateComponent::attention_kv, strix::SessionStateComponent::recurrent_matrix,
    strix::SessionStateComponent::gdn_convolution, strix::SessionStateComponent::ple_convolution,
    strix::SessionStateComponent::sparse_index, strix::SessionStateComponent::ngram_history,
    strix::SessionStateComponent::position_history};

std::uint64_t read_word(std::uintptr_t address) {
    std::uint64_t word = 0;
    std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
    return word;
}
void write_word(std::uintptr_t address, std::uint64_t word) {
    std::memcpy(reinterpret_cast<void*>(address), &word, sizeof(word));
}
bool mock_has(strix::PersistentState set, strix::PersistentState flag) {
    return (static_cast<std::uint32_t>(set) & static_cast<std::uint32_t>(flag)) != 0;
}
struct WindowTrace {
    std::uint32_t calls = 0;
    std::uint32_t admissions = 0;
    bool fail_kernel = false;
    bool fail_qsa = false;
    std::uint64_t fail_position = std::numeric_limits<std::uint64_t>::max();
    strix::SessionStateMemoryBinding* rebind_after_qsa = nullptr;
    std::uint32_t rebindings = 0;
    std::vector<std::uint64_t> qsa_positions;
    std::vector<std::uint64_t> ple_previous;
    std::vector<std::uint64_t> head_positions, head_admissions;
};
void recording_window_preflight(const strix::ExecutionDispatchContext& context, void* opaque) {
    ++static_cast<WindowTrace*>(opaque)->admissions;
    if(context.op==strix::SerialOp::lm_head)static_cast<WindowTrace*>(opaque)->head_admissions.push_back(context.position);
    if (context.logical_rows != 1 || context.request_concurrency != 1 || !context.resources)
        throw std::invalid_argument("mock R1 preflight requires a complete one-row binding");
    for (std::size_t i = 0; i < context.reads.size(); ++i) (void)strix::resolve_dispatch_read(context, i);
    for (std::size_t i = 0; i < context.writes.size(); ++i) (void)strix::resolve_dispatch_write(context, i);
    for (std::size_t i = 0; i < context.temporaries.size(); ++i) (void)strix::resolve_dispatch_temporary(context, i);
    for (std::size_t i = 0; i < mock_components.size(); ++i)
        if (mock_has(context.reads_state | context.writes_state, mock_state_flags[i]))
            (void)strix::resolve_dispatch_state(context, mock_components[i]);
}
void throwing_window_preflight(const strix::ExecutionDispatchContext&, void* opaque) {
    ++static_cast<WindowTrace*>(opaque)->admissions;
    throw std::runtime_error("late row1 terminal metadata admission failure");
}
class StartFaultLookup final : public strix::PleLookupBackend {
public:
    explicit StartFaultLookup(bool zero) : zero_(zero) {}
    std::uint32_t starts = 0, cancels = 0;
    std::uint64_t implementation_identity() const noexcept override { return 0x54455354504c4531ULL; }
    strix::PleLookupHandle start(const strix::PleLookupRequest&) override {
        ++starts;
        if (zero_) return 0;
        throw std::runtime_error("lookup start failure before mutation");
    }
    void finish(strix::PleLookupHandle, std::span<std::byte>) override {
        throw std::logic_error("a failed start has no finishable handle");
    }
    void cancel(strix::PleLookupHandle) noexcept override { ++cancels; }
private:
    bool zero_;
};
void recording_window_kernel(const strix::ExecutionDispatchContext& context, void* opaque) {
    auto& trace = *static_cast<WindowTrace*>(opaque);
    ++trace.calls;
    if(context.op==strix::SerialOp::lm_head)trace.head_positions.push_back(context.position);
    if (context.logical_rows != 1 || context.request_concurrency != 1)
        throw std::logic_error("mock requires the incumbent R1 context");
    if (mock_has(context.reads_state, strix::PersistentState::position_history)) {
        const auto position = strix::resolve_dispatch_state(context,
            strix::SessionStateComponent::position_history);
        const auto word = read_word(position.address);
        if (word != context.position) throw std::logic_error("row reader observed future position");
        trace.qsa_positions.push_back(word);
        if (context.op == strix::SerialOp::indexed_attention && trace.rebind_after_qsa &&
            trace.rebindings == 0) {
            ++trace.rebind_after_qsa->host_allocation_generation;
            ++trace.rebindings;
        }
    }
    if (((trace.fail_kernel && context.position == 1) || context.position==trace.fail_position) &&
        context.op == strix::SerialOp::gated_delta_net)
        throw std::runtime_error("injected row1 kernel failure");
    if (trace.fail_qsa && context.position == 1 && context.op == strix::SerialOp::indexed_attention)
        throw std::runtime_error("injected row1 QSA failure");
    std::uint64_t value = 101U + context.input_token_id + 17U * context.position + context.step_id;
    // Snapshot all input words before writing any potentially reused arena slot.
    for (std::size_t i = 0; i < context.reads.size(); ++i)
        value = value * 1099511628211ULL + read_word(strix::resolve_dispatch_read(context, i).address);
    for (std::size_t i = 0; i < mock_components.size() - 1; ++i) {
        if (!mock_has(context.writes_state, mock_state_flags[i])) continue;
        if (mock_components[i] == strix::SessionStateComponent::gdn_convolution &&
            context.op != strix::SerialOp::gated_delta_net) continue;
        if (mock_components[i] == strix::SessionStateComponent::ple_convolution &&
            context.op != strix::SerialOp::ple) continue;
        const auto state = strix::resolve_dispatch_state(context, mock_components[i]);
        const auto lane = static_cast<std::uint64_t>(context.layer.value_or(0));
        const auto address = state.address + lane * sizeof(std::uint64_t);
        const auto previous = read_word(address);
        if (mock_components[i] == strix::SessionStateComponent::ngram_history)
            trace.ple_previous.push_back(previous);
        value = value * 1099511628211ULL + previous;
        write_word(address, previous * 33U + context.input_token_id + context.position + 1U);
    }
    for (std::size_t i = 0; i < context.writes.size(); ++i)
        write_word(strix::resolve_dispatch_write(context, i).address, value + i);
}
void recording_r2_preflight(std::span<const strix::ExecutionDispatchContext, 2> rows, void* opaque) {
    ++static_cast<WindowTrace*>(opaque)->admissions;
    if (rows[0].step_id != rows[1].step_id || rows[1].position != rows[0].position + 1 ||
        rows[0].execution_identity != rows[1].execution_identity)
        throw std::invalid_argument("mock R2 admission needs adjacent rows of one step");
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < row.reads.size(); ++i) (void)strix::resolve_dispatch_read(row, i);
        for (std::size_t i = 0; i < row.writes.size(); ++i) (void)strix::resolve_dispatch_write(row, i);
    }
}
void recording_r2_launch(std::span<const strix::ExecutionDispatchContext, 2> rows, void* opaque) {
    recording_window_kernel(rows[0], opaque);
    recording_window_kernel(rows[1], opaque);
}

void recording_rows_preflight(std::span<const strix::ExecutionDispatchContext> rows,void* opaque) {
    if(rows.size()!=2U && rows.size()!=4U && rows.size()!=8U) throw std::invalid_argument("mock joint rows require2/4/8");
    for(std::size_t r=0;r<rows.size();r+=2U)
        recording_r2_preflight(std::span<const strix::ExecutionDispatchContext,2>(rows.data()+r,2U),opaque);
    for(std::size_t r=1;r<rows.size();++r)
        if(rows[r].position!=rows[0].position+r || rows[r].execution_identity!=rows[0].execution_identity)
            throw std::invalid_argument("mock wide rows are not one adjacent window");
}
void recording_rows_launch(std::span<const strix::ExecutionDispatchContext> rows,void* opaque) {
    for(const auto& row:rows) recording_window_kernel(row,opaque);
}
// Native QSA rows consume each row's own context position, never the host word.
void recording_qsa_rows_launch(std::span<const strix::ExecutionDispatchContext> rows,void* opaque) {
    for(auto row:rows) {
        row.reads_state=static_cast<strix::PersistentState>(static_cast<std::uint32_t>(row.reads_state)&
            ~static_cast<std::uint32_t>(strix::PersistentState::position_history));
        recording_window_kernel(row,opaque);
    }
}

struct KnownRuntimeFixture {
    KnownFixture source;
    strix::CompiledKnownPrefixWindow window;
    std::vector<std::uint64_t> serial_pool, window_pool, state_words;
    strix::ExecutionMemoryBinding serial_memory, window_memory;
    strix::SerialSessionStateLayout layout;
    strix::SessionStateMemoryBinding state_memory;
    strix::ModelRuntimeBinding weights;
    strix::ExecutionRuntimeResources resources;
    strix::SerialSessionContext session;
    WindowTrace trace;
    strix::ExecutionLauncherRegistry registry;
    strix::CompiledLaunchPlan serial_launch;
    std::vector<strix::CompiledLaunchPlan> row_launch;
    std::array<std::uint64_t, 1> allowed_readers{recording_qsa_identity};
    std::vector<std::uint32_t> tokens{7,11};
    std::uint32_t quiesces = 0;

    explicit KnownRuntimeFixture(const strix::ModelContract& model, bool promote = false, bool mixed = false,
        std::uint32_t rows = 2, bool routed_union = false,
        strix::KnownWindowOutputPolicy output_policy = strix::KnownWindowOutputPolicy::all_rows, bool hyper = false, bool qsa = false, bool head_rows = false)
        : source(make_known_fixture(model, mixed ? 0U : 8U,routed_union,hyper,qsa,8U,head_rows ? 8U : 0xffffffffU)),
          window(strix::compile_known_prefix_window(model, source.serial, source.manifest,
              source.access, {promote, promote, mixed, rows,routed_union,output_policy,hyper,qsa,false,false,head_rows})),
          serial_pool(static_cast<std::size_t>((source.serial.arena.total_bytes + 7U) / 8U)),
          window_pool(static_cast<std::size_t>((window.arena.physical.total_bytes + 7U) / 8U)),
          state_words(7U * 48U, 0) {
        serial_memory = {source.serial.execution_identity, reinterpret_cast<std::uintptr_t>(serial_pool.data()),
            source.serial.arena.total_bytes, 1};
        window_memory = {window.window_execution_identity, reinterpret_cast<std::uintptr_t>(window_pool.data()),
            window.arena.physical.total_bytes, 2};
        layout.context_tokens = 64; layout.layout_identity = 0x535441544531;
        layout.host_total_bytes = state_words.size() * sizeof(std::uint64_t);
        for (std::size_t i = 0; i < mock_components.size(); ++i)
            layout.allocations.push_back({mock_components[i], mock_state_flags[i],
                strix::SessionStateStorage::host, i * 48U * sizeof(std::uint64_t),
                48U * sizeof(std::uint64_t), false});
        state_memory.layout_identity = layout.layout_identity;
        state_memory.host_base = reinterpret_cast<std::uintptr_t>(state_words.data());
        state_memory.host_bytes = layout.host_total_bytes; state_memory.host_allocation_generation = 3;
        weights.manifest_layout_identity = source.manifest.layout_identity;
        weights.model_content_identity.bytes[0] = 0x42;
        for (std::size_t i = 0; i < source.manifest.tensors.size(); ++i)
            weights.tensors.push_back({static_cast<std::uint32_t>(i), strix::RuntimeTensorStorage::resident,
                0x10000000U + source.manifest.tensors[i].absolute_offset,
                source.manifest.tensors[i].byte_size, 4, 0});
        resources = {&serial_memory, &source.manifest, &weights, &source.access, &layout, &state_memory};
        strix::register_default_runtime_launchers(registry);
        std::set<std::string> registered;
        for (const auto& binding : source.serial.bindings.bindings) {
            if (binding.host_or_runtime_step || !registered.insert(*binding.kernel_id).second) continue;
            registry.add_kernel_launcher({*binding.kernel_id,
                binding.op == strix::SerialOp::indexed_attention ? recording_qsa_identity : 0x544553544b45524eULL,
                &recording_window_kernel, &trace, &recording_window_preflight});
        }
        serial_launch = strix::compile_launch_plan(source.serial, registry);
        row_launch.resize(rows);
        tokens.resize(rows);
        for(std::uint32_t r=2;r<rows;++r) tokens[r]=13U+r;
        for (std::size_t r = 0; r < rows; ++r)
            row_launch[r] = strix::compile_launch_plan(window.row_plan[r], registry);
        reset();
    }
    void reset() {
        std::fill(serial_pool.begin(), serial_pool.end(), 0);
        std::fill(window_pool.begin(), window_pool.end(), 0);
        std::fill(state_words.begin(), state_words.end(), 0);
        trace = {}; quiesces = 0; session = {};
        session.execution_identity = source.serial.execution_identity;
        session.state_layout_identity = layout.layout_identity;
        session.runtime_compatibility_identity = strix::build_runtime_execution_identities(
            source.serial.execution_identity, source.serial.arena, resources).compatibility_identity;
        for (const auto flag : mock_state_flags) session.state.valid = session.state.valid | flag;
    }
    strix::KnownWindowExecutionHooks hooks() {
        strix::KnownWindowExecutionHooks h;
        h.preflight_plan = [] {}; h.preflight_scope = [] {}; h.complete = [] {};
        h.quiesce_or_fail_stop = [this] { ++quiesces; }; h.position_reader_implementations = allowed_readers;
        return h;
    }
    strix::KnownWindowResult run(strix::KnownWindowExecutionHooks callbacks) {
        const std::array<strix::KnownWindowR2Entry, 2> entries{{
            {strix::KnownWindowR2Variant::shared_q8_virtual256_r2_v1, &trace,
                &recording_r2_preflight, &recording_r2_launch},
            {strix::KnownWindowR2Variant::gdn_q8_virtual256_r2_v1, &trace,
                &recording_r2_preflight, &recording_r2_launch}}};
        return run_entries(callbacks, entries);
    }
    strix::KnownWindowResult run_entries(strix::KnownWindowExecutionHooks callbacks,
        std::span<const strix::KnownWindowR2Entry> entries) {
        std::vector<const strix::CompiledLaunchPlan*> plans;
        for(const auto& plan:row_launch) plans.push_back(&plan);
        return strix::execute_known_prefix_window_eager(window, source.serial, resources,
            window_memory, session, tokens, plans, entries, {}, callbacks);
    }
    std::uint64_t serial_token(std::uint32_t token) {
        strix::DirectLaunchDispatcher dispatcher(serial_launch);
        (void)strix::execute_serial_token(source.serial, session, resources, token, dispatcher);
        const auto use = source.serial.io.steps.back().writes.front();
        return read_word(strix::resolve_execution_buffer(source.serial.execution_identity,
            source.serial.arena, serial_memory, use).address);
    }
    std::uintptr_t position_address() const {
        return strix::resolve_session_state(layout, state_memory,
            strix::SessionStateComponent::position_history).address;
    }
};

void yarn_serial_window_selection_tests(const strix::ModelContract& native_model) {
    using namespace strix;
    using V=KnownWindowR2Variant; using M=KnownWindowStepMode;
    // Captured before this change, from the unchanged native compiler.
    constexpr std::array<std::array<std::uint64_t,5>,12> native_pins{{
        {13083362910635310691ULL,4863534667393275108ULL,1996800ULL,3831541760ULL,638ULL},
        {15918103719367961873ULL,14460702830281900315ULL,3993600ULL,9279283200ULL,1638ULL},
        {15789861501899281442ULL,15169130105291753025ULL,7987200ULL,20174766080ULL,3638ULL},
        {15654963642845729118ULL,1949998428958718756ULL,1996800ULL,4497571840ULL,567ULL},
        {1569738126787255186ULL,11974845279599050728ULL,3993600ULL,13492715520ULL,1605ULL},
        {16108088439835032299ULL,8544296000465250752ULL,7987200ULL,31483002880ULL,3681ULL},
        {11923134548858325583ULL,8305500008284324988ULL,1996800ULL,3822141440ULL,566ULL},
        {16888714372931873527ULL,14975567280879251392ULL,3993600ULL,11466424320ULL,1602ULL},
        {2018404911890620875ULL,6135181892703517445ULL,7987200ULL,26754990080ULL,3674ULL},
        {15654963642845729118ULL,1949998428958718756ULL,1996800ULL,4497571840ULL,567ULL},
        {1569738126787255186ULL,11974845279599050728ULL,3993600ULL,13492715520ULL,1605ULL},
        {16108088439835032299ULL,8544296000465250752ULL,7987200ULL,31483002880ULL,3681ULL},
    }};
    for(auto variant:{V::gdn_q8_virtual256_r2_v1,static_cast<V>(known_window_r2_variant_slots),static_cast<V>(UINT32_MAX)}) {
        const auto repaired=known_window_select_serial_rows(true,M::r2_gdn_staged,variant);
        expect(repaired.mode==M::r1_row_loop && repaired.variant==V::none,
            "nonserial and future unknown window arms fall back to the complete serial source");
        const auto native=known_window_select_serial_rows(false,M::r2_gdn_staged,variant);
        expect(native.mode==M::r2_gdn_staged && native.variant==variant,"native arm selection is unchanged");
    }
    std::size_t pin=0;
    for(auto gate:{8U,0U}) for(bool prefix:{false,true}) for(auto rows:{2U,4U,8U}) {
        const auto f=make_known_fixture(native_model,gate,true,true,true,30U,8U);
        KnownWindowRequest req{true,true,true,rows,true,KnownWindowOutputPolicy::all_rows,
            true,true,prefix,prefix,prefix,true,true};
        const auto w=compile_known_prefix_window(native_model,f.serial,f.manifest,f.access,req);
        const std::array<std::uint64_t,5> actual{w.window_execution_identity,w.window_kernel_plan_identity,
            w.arena.physical.total_bytes,w.logical_projection_weight_bytes_avoided,w.projection_launches_saved};
        expect(actual==native_pins[pin++],"native R2/R4/R8 plan content hashes, arena and savings stay unchanged");
    }
    for(auto factor:{2U,4U}) for(auto gate:{8U,0U}) for(bool prefix:{false,true}) for(auto rows:{2U,4U,8U}) for(auto profile:{0U,2U}) {
        auto model=native_model;ModelContentIdentity hash;hash.bytes[0]=1;
        model.resolved_rotary=RotaryIdentity(RotaryDescriptor::yarn(factor),hash);
        routed_fixture_profile=profile;
        auto f=make_known_fixture(model,gate,true,true,true,30U,8U);
        f.serial.max_position_embeddings=model.effective_max_positions();
        for(std::size_t i=0;i<f.serial.execution.steps.size();++i) {
            const auto op=f.serial.execution.steps[i].op;
            auto& id=f.serial.bindings.bindings[i].kernel_id;
            if(op==SerialOp::moe_route) id="hip.moe_route.bf16.h2560.e512.k10.r1.c1.staged_logits.wave.parallel";
            else if(op==SerialOp::attn_inject || op==SerialOp::mlp_inject) id="hip.inject.bf16.h2560.hc4.r1.c1";
            else if(op==SerialOp::moe_combine) id="hip.moe_combine.bf16.h2560.hc4.r1.c1";
        }
        KnownWindowRequest req{true,true,true,rows,true,KnownWindowOutputPolicy::all_rows,
            true,true,prefix,prefix,prefix,true,true,true};
        const auto w=compile_known_prefix_window(model,f.serial,f.manifest,f.access,req);
        validate_compiled_known_prefix_window(w,f.serial);
        validate_known_prefix_window_formats(w,f.serial,f.manifest,f.access);
        expect(known_window_policy_serial_rows(w),"every YaRN selected variant is a serial-row arm");
        expect(w.routed_expert_union_steps==48 && w.hyper_mix_rows_steps==97 && w.qsa_rows_steps==12 &&
            w.shared_rows_steps+(rows==2?w.r2_steps:0)>=48 && w.router_rows_steps==48 &&
            w.combine_inject_rows_steps>0,"YaRN retains serial-row union, HC, QSA, shared and elementwise promotions");
        if(prefix) expect(w.gdn_deferred_rows_steps==36 && w.ple_rows_steps>0,"YaRN retains deferred GDN and PLE prefix acceptance");
        for(std::size_t i=0;i<w.mode.size();++i) if(f.serial.execution.steps[i].op==SerialOp::gated_delta_net) {
            if(!prefix && gate==8U) expect(w.mode[i]==M::r1_row_loop && w.r2_variant[i]==V::none,
                "all-Q8 nonserial paired GDN falls back to its original serial row launcher");
            if(!prefix && gate!=8U) expect(w.r2_variant[i]==(rows==2U?V::gdn_q8_rows2_f32_gates_r1_v1:
                rows==4U?V::gdn_q8_rows4_f32_gates_r1_v1:V::gdn_q8_rows8_f32_gates_r1_v1),
                "mixed gates retain the serial-equivalent GDN rows variant");
        }
        for(unsigned bits=0;bits<8;++bits) {
            const KnownWindowFolds requested{(bits&1)!=0,(bits&2)!=0,(bits&4)!=0};
            const auto folds=known_window_folds_for_route(true,requested);
            expect(known_window_policy_selection_admitted(w,false,false,folds),"YaRN owner admits R2/R4/R8 for every process fold combination");
            const auto native=known_window_folds_for_route(false,requested);
            expect(native.inject_norm==requested.inject_norm && native.combine_inject==requested.combine_inject &&
                native.commit_combine==requested.commit_combine,"native owner preserves all process fold choices");
        }
        expect(!known_window_policy_selection_admitted(w,true,false,{}) &&
            !known_window_policy_selection_admitted(w,false,true,{}),"prefill and GSQ remain excluded from policy admission");
        auto bad=w;bad.r2_variant[0]=V::gdn_q8_virtual256_r2_v1;
        expect(!known_window_policy_selection_admitted(bad,false,false,{}),"nonserial GDN remains excluded after repair");
        bad.r2_variant[0]=static_cast<V>(known_window_r2_variant_slots);
        expect(!known_window_policy_selection_admitted(bad,false,false,{}),"unclassified future variant remains excluded");
        const auto witness=known_window_arithmetic_witness(w,true,false,false,{});
        expect(witness.find("\"policy_selection_admitted\":true")!=std::string::npos &&
            witness.find("\"folds\":{\"inject_norm\":false,\"combine_inject\":false,\"commit_combine\":false}")!=std::string::npos &&
            witness.find("gdn_q8_virtual256_r2_v1")==std::string::npos &&
            witness.find(profile?"routed_a8_union_rows":"routed_q51_union_rows")!=std::string::npos,
            "witness records selected variants and disabled folds");
    }
    routed_fixture_profile=0U;
}

void known_prefix_prefill_tests(const strix::ModelContract& model) {
    using P=strix::KnownWindowOutputPolicy;using V=strix::KnownWindowR2Variant;
    // Host flow/state oracle only. Native projection arithmetic is exercised
    // separately; these recording kernels deliberately use tiny state words.
    for(bool mixed:{false,true}) for(auto rows:{16U,17U,32U}) for(auto policy:{P::all_rows,P::terminal_row,P::state_only})
        for(bool partial:{false,true}) for(bool early:{false,true}) {
        if(partial && (rows%8U || policy==P::all_rows)) continue;
        KnownRuntimeFixture f(model,true,mixed);
        strix::KnownPrefillRequest request;request.logical_rows=partial?rows+16U:rows;request.r2_shared_expert=true;request.r2_gdn=true;
        request.r2_gdn_f32_gates=mixed;request.output_policy=policy;
        request.workspace_budget_bytes=UINT64_MAX;request.host_metadata_payload_budget_bytes=UINT64_MAX;
        const auto prefill=strix::compile_known_prefix_prefill(model,f.source.serial,f.source.manifest,f.source.access,request);
        strix::validate_compiled_known_prefix_prefill(prefill,f.source.serial);
        std::vector<std::uint32_t> tokens(rows);for(std::uint32_t r=0;r<rows;++r) tokens[r]=7U+r*3U;
        std::vector<std::uint64_t> expected_heads;
        for(auto token:tokens) expected_heads.push_back(f.serial_token(token));
        const auto expected_state=f.state_words;const auto expected_generations=f.session.state.generations;
        const auto continuation=f.serial_token(91U);const auto continued_state=f.state_words;
        f.window_pool.resize(static_cast<std::size_t>((prefill.arena.physical.total_bytes+7U)/8U));
        f.window_memory={prefill.prefill_execution_identity,reinterpret_cast<std::uintptr_t>(f.window_pool.data()),
            prefill.arena.physical.total_bytes,11U};
        f.reset();
        auto source=f.source.serial;source.execution_identity=prefill.prefill_execution_identity;source.arena=prefill.arena.row_view[0];
        const auto launch=strix::compile_launch_plan(source,f.registry);
        const std::array<strix::KnownWindowR2Entry,1> pairs{{{V::gdn_q8_virtual256_r2_v1,&f.trace,&recording_r2_preflight,&recording_r2_launch}}};
        const std::array<strix::KnownWindowRowsEntry,2> wide{{
            {V::shared_q8_virtual256_r8_v1,&f.trace,&recording_rows_preflight,&recording_rows_launch},
            {V::gdn_q8_rows8_f32_gates_r1_v1,&f.trace,&recording_rows_preflight,&recording_rows_launch}}};
        auto prefill_hooks=f.hooks();prefill_hooks.partial_rows=partial;
        std::vector<const void*> static_memo, live_memo;
        prefill_hooks.static_preflight_memo=&static_memo;
        prefill_hooks.preflight_memo=&live_memo;
        if(early) {
            auto setup=prefill_hooks;setup.partial_rows=false;
            setup.preflight_only=true;setup.static_preflight_only=true;
            setup.complete=[] {throw std::logic_error("setup must never complete");};
            const auto initial_state=f.state_words, initial_arena=f.window_pool;
            const auto initial_session=f.session;
            (void)strix::execute_known_prefix_prefill_eager(prefill,f.source.serial,f.resources,f.window_memory,
                f.session,{},launch,pairs,wide,setup);
            expect(f.trace.calls==0 && f.state_words==initial_state && f.window_pool==initial_arena &&
                f.session.next_position==initial_session.next_position &&
                f.session.committed_tokens==initial_session.committed_tokens &&
                f.session.state.generations==initial_session.state.generations &&
                f.session.arithmetic_lineage.history_identity==initial_session.arithmetic_lineage.history_identity,
                "static setup without a prompt never launches, completes or mutates state/session/arena");
            expect(static_memo.size()==1 && live_memo.empty(),"setup seals only static admission");
            auto invalid=tokens;invalid.back()=static_cast<std::uint32_t>(model.vocab_size);
            bool refused=false;
            try {(void)strix::execute_known_prefix_prefill_eager(prefill,f.source.serial,f.resources,f.window_memory,
                f.session,invalid,launch,pairs,wide,prefill_hooks);} catch(const std::invalid_argument&) {refused=true;}
            expect(refused && f.trace.calls==0 && live_memo.empty() && f.state_words==initial_state,
                "static admission never admits a bad live prompt or publishes a live memo");
            // Clear recording-only head admissions: partial blocks still admit their own live geometry.
            if(partial) f.trace.head_admissions.clear();
        }
        const auto result=strix::execute_known_prefix_prefill_eager(prefill,f.source.serial,f.resources,f.window_memory,
            f.session,tokens,launch,pairs,wide,prefill_hooks);
        expect(f.session.next_position==rows&&f.session.committed_tokens==rows&&f.session.state.generations==expected_generations,
            "T prefill commits real T inputs and exactly serial state generations once");
        expect(f.state_words==expected_state,"T step-major prefill preserves all seven logical host state payloads");
        // The final-block cone runs the trailing stateless steps (here: the last layer's shared step) from the tile
        // of the terminal row only.
        expect(policy!=P::terminal_row||rows<=8U||(rows%8U!=0U||(result.final_cone_first_row==rows-8U&&result.final_cone_steps>0U)),
            "a terminal-row T block cones its trailing stateless steps to the terminal tile");
        expect(result.shared_rows_dispatches==48U*(rows/8U)-result.final_cone_skipped_tiles&&
            (mixed?result.gdn_rows_dispatches==36U*(rows/8U):result.r2_dispatches==36U*(rows/2U)),
            "T prefill uses real closed tiles at every source step and explicit row tails");
        expect(f.trace.head_admissions.size()==(prefill.terminal_logits_arena?prefill.logit_row_indices.size():rows),
            "only logits-capable terminal rows receive head preflight in compact arenas");
        const auto outputs=policy==P::all_rows?rows:(policy==P::terminal_row?1U:0U);
        expect(result.logits.size()==outputs&&f.trace.head_positions.size()==outputs&&result.omitted_lm_heads==rows-outputs,
            "T output policy maps only actual selected terminal heads");
        for(std::size_t i=0;i<result.logits.size();++i)
            expect(read_word(result.logits[i].address)==expected_heads[result.logit_row_indices[i]],"T selected head payload equals its serial row");
        expect(f.trace.ple_previous.size()==rows,"PLE retains one ordered history transition per T row");
        expect(f.trace.qsa_positions.size()==12U*rows,"every QSA layer reads every absolute causal row position");
        for(std::size_t i=0;i<f.trace.qsa_positions.size();++i)
            expect(f.trace.qsa_positions[i]==i%rows,"QSA observes no future host position across T tiles");
        expect(!f.session.arithmetic_lineage.tainted&&strix::arithmetic_lineage_valid(f.session.arithmetic_lineage,rows)&&
            f.session.arithmetic_lineage.intervals.back().numerical_plan_identity==prefill.prefill_execution_identity,
            "T producer lineage uses the explicit new organization identity");
        expect(f.serial_token(91U)==continuation&&f.state_words==continued_state,"serial continuation after T prefill retains every causal state");
    }
    {
        KnownRuntimeFixture f(model,true,true);
        strix::KnownPrefillRequest request;request.logical_rows=16;request.r2_shared_expert=true;request.r2_gdn=true;request.r2_gdn_f32_gates=true;
        request.workspace_budget_bytes=UINT64_MAX;request.host_metadata_payload_budget_bytes=UINT64_MAX;
        const auto good=strix::compile_known_prefix_prefill(model,f.source.serial,f.source.manifest,f.source.access,request);
        for(unsigned fault=0;fault<6;++fault) {
            auto bad=good;
            if(fault==0) bad.logical_rows=8;
            if(fault==1) ++bad.arena.row_view.back().allocations.back().offset;
            if(fault==2) ++bad.arena.physical.allocations.back().bytes;
            if(fault==3) bad.logit_row_indices.pop_back();
            if(fault==4) bad.workspace_budget_bytes=bad.arena.physical.total_bytes-1U;
            if(fault==5) bad.host_metadata_payload_budget_bytes=bad.host_metadata_payload_bytes-1U;
            bool rejected=false;try {strix::validate_compiled_known_prefix_prefill(bad,f.source.serial);} catch(const std::exception&) {rejected=true;}
            expect(rejected,"T prefill forged geometry/arena/map/budget rejected with unchanged identities");
        }
        for(auto rows:{15U,strix::known_prefill_max_rows+1U,32768U}) {
            auto bad=request;bad.logical_rows=rows;bool rejected=false;
            try {(void)strix::compile_known_prefix_prefill(model,f.source.serial,f.source.manifest,f.source.access,bad);} catch(const std::invalid_argument&) {rejected=true;}
            expect(rejected,"T prefill closed range rejects out-of-bounds geometry");
        }
        for(unsigned fault=0;fault<2;++fault) {
            auto bad=request;if(fault==0) bad.workspace_budget_bytes=1;else bad.host_metadata_payload_budget_bytes=1;
            bool rejected=false;try {(void)strix::compile_known_prefix_prefill(model,f.source.serial,f.source.manifest,f.source.access,bad);} catch(const std::invalid_argument&) {rejected=true;}
            expect(rejected,"T arena and host payload budgets reject before native allocation or state mutation");
        }
        // Maximum T is a planner-only fixture: never allocate its model-sized
        // scratch merely to inspect the bounded row-view/identity contract.
        auto maximum=request;maximum.logical_rows=strix::known_prefill_max_rows;
        const auto maximum_plan=strix::compile_known_prefix_prefill(model,f.source.serial,f.source.manifest,f.source.access,maximum);
        strix::validate_compiled_known_prefix_prefill(maximum_plan,f.source.serial);
        expect(maximum_plan.arena.row_view.size()==strix::known_prefill_max_rows,"maximum T owns bounded row metadata without T topology clones");
        expect(maximum.logical_rows==16384U&&maximum_plan.logit_row_indices.size()==16384U,
            "T16384 compiles the complete output map without narrowing its row count");
        for(unsigned fault=0;fault<4;++fault) {
            f.reset();std::vector<std::uint32_t> tokens(16U,7U);
            f.window_pool.resize(static_cast<std::size_t>((good.arena.physical.total_bytes+7U)/8U));
            f.window_memory={good.prefill_execution_identity,reinterpret_cast<std::uintptr_t>(f.window_pool.data()),good.arena.physical.total_bytes,11U};
            auto source=f.source.serial;source.execution_identity=good.prefill_execution_identity;source.arena=good.arena.row_view[0];
            auto launch=strix::compile_launch_plan(source,f.registry);
            const auto state_before=f.state_words;const auto position_before=read_word(f.position_address());
            if(fault==0) launch.steps.back().preflight=[](const strix::ExecutionDispatchContext& c,void* p) {
                if(c.position==15U) throw std::invalid_argument("late T row terminal admission failure");
                recording_window_preflight(c,p);};
            if(fault==1) tokens.back()=static_cast<std::uint32_t>(model.vocab_size);
            if(fault==2) f.trace.fail_position=15U;
            const std::array<strix::KnownWindowRowsEntry,2> entries{{
                {V::shared_q8_virtual256_r8_v1,&f.trace,&recording_rows_preflight,&recording_rows_launch},
                {V::gdn_q8_rows8_f32_gates_r1_v1,&f.trace,&recording_rows_preflight,&recording_rows_launch}}};
            bool rejected=false;
            try {(void)strix::execute_known_prefix_prefill_eager(good,f.source.serial,f.resources,f.window_memory,f.session,tokens,
                launch,{},fault==3?std::span<const strix::KnownWindowRowsEntry>{}:std::span<const strix::KnownWindowRowsEntry>(entries),f.hooks());}
            catch(const std::exception&) {rejected=true;}
            expect(rejected,"late T admission/callback fault is observable");
            if(fault==2) expect(f.session.health==strix::SerialSessionHealth::poisoned&&f.session.arithmetic_lineage.tainted&&
                f.quiesces==1&&f.session.committed_tokens==0,"T dispatch failure drains once and publishes no partial session metadata");
            else expect(f.trace.calls==0&&f.session.health==strix::SerialSessionHealth::ready&&f.state_words==state_before&&
                read_word(f.position_address())==position_before,"all T rows admit before any target mutation");
        }
    }
}

// Full-T mock entry: every row as its R1 row; rows of several sessions (segments) each keep consecutive positions.
void recording_full_preflight(std::span<const strix::ExecutionDispatchContext> rows,void* opaque) {
    if(rows.size()<8U || rows.size()%8U) throw std::invalid_argument("mock full rows require 8-row tiles");
    for(std::size_t r=0;r<rows.size();++r) {
        recording_window_preflight(rows[r],opaque);
        const bool start=r==0U || rows[r].session_state_memory!=rows[r-1U].session_state_memory;
        if(!start && rows[r].position!=rows[r-1U].position+1U) throw std::invalid_argument("mock full rows break a segment's positions");
        if(!start && rows[r].resources!=rows[r-1U].resources) throw std::invalid_argument("mock full rows mix resources inside a segment");
    }
}

// Segmented prefill (several sessions' blocks in one run) against each block alone: per-session state words,
// generations, positions, terminal head, lineage and observed rows, with shared full-T steps and per-segment steps.
void known_prefix_prefill_segment_tests(const strix::ModelContract& model) {
    using P=strix::KnownWindowOutputPolicy;using V=strix::KnownWindowR2Variant;
    const std::vector<std::uint32_t> sizes{16U,32U,24U};
    const std::vector<std::uint32_t> lead_in{0U,3U,5U};
    const auto request_for=[&](std::uint32_t rows,P policy) {
        strix::KnownPrefillRequest request;request.logical_rows=rows;request.r2_shared_expert=true;request.r2_gdn=true;
        request.r2_gdn_f32_gates=true;request.output_policy=policy;request.exact_memory=false;
        request.workspace_budget_bytes=UINT64_MAX;request.host_metadata_payload_budget_bytes=UINT64_MAX;
        return request;
    };
    // An observed step with an output, past the middle of the forward.
    const auto tap_step=[](const strix::CompiledSerialExecution& serial) {
        auto i=serial.execution.steps.size()/2U;
        while(i+1U<serial.execution.steps.size() && serial.io.steps[i].writes.empty()) ++i;
        return i;
    };
    const auto block_tokens=[](std::size_t s,std::uint32_t rows) {
        std::vector<std::uint32_t> tokens(rows);for(std::uint32_t r=0;r<rows;++r) tokens[r]=5U+static_cast<std::uint32_t>(s)*29U+r*7U;
        return tokens;
    };
    for(auto policy:{P::terminal_row,P::state_only}) for(bool shared:{false,true}) {
        // Reference: each block alone after its own serial lead-in.
        struct Reference {std::vector<std::uint64_t> state;std::array<std::uint64_t,6> generations{};std::uint64_t position=0,head=0,lineage=0;
            std::vector<std::uint64_t> observed;};
        std::vector<Reference> reference(sizes.size());
        for(std::size_t s=0;s<sizes.size();++s) {
            KnownRuntimeFixture f(model,true,true);
            for(std::uint32_t k=0;k<lead_in[s];++k) (void)f.serial_token(3U+k*5U+static_cast<std::uint32_t>(s));
            const auto plan=strix::compile_known_prefix_prefill(model,f.source.serial,f.source.manifest,f.source.access,request_for(sizes[s],policy));
            f.window_pool.assign(static_cast<std::size_t>((plan.arena.physical.total_bytes+7U)/8U),0U);
            f.window_memory={plan.prefill_execution_identity,reinterpret_cast<std::uintptr_t>(f.window_pool.data()),plan.arena.physical.total_bytes,11U};
            auto source=f.source.serial;source.execution_identity=plan.prefill_execution_identity;source.arena=plan.arena.row_view[0];
            const auto launch=strix::compile_launch_plan(source,f.registry);
            const std::array<strix::KnownWindowRowsEntry,2> wide{{{V::shared_q8_virtual256_r8_v1,&f.trace,&recording_rows_preflight,&recording_rows_launch},
                {V::gdn_q8_rows8_f32_gates_r1_v1,&f.trace,&recording_rows_preflight,&recording_rows_launch}}};
            const std::array<strix::KnownWindowRowsEntry,1> full{{{V::shared_q8_virtual256_r8_v1,&f.trace,&recording_full_preflight,&recording_rows_launch}}};
            auto hooks=f.hooks();
            const auto tap=tap_step(f.source.serial);
            hooks.after_step=[&](std::size_t step,const std::function<strix::ExecutionDispatchContext(std::uint32_t)>& row) {
                if(step!=tap) return;
                for(std::uint32_t r=0;r<sizes[s];++r) reference[s].observed.push_back(read_word(strix::resolve_dispatch_write(row(r),0).address));
            };
            const auto tokens=block_tokens(s,sizes[s]);
            const auto result=strix::execute_known_prefix_prefill_eager(plan,f.source.serial,f.resources,f.window_memory,f.session,tokens,
                launch,{},wide,hooks,full);
            reference[s].state=f.state_words;reference[s].generations=f.session.state.generations;reference[s].position=f.session.next_position;
            reference[s].lineage=f.session.arithmetic_lineage.history_identity;
            if(policy==P::terminal_row) reference[s].head=read_word(result.logits.at(0).address);
            expect(result.logits.size()==(policy==P::terminal_row?1U:0U),"solo block maps its terminal head by policy");
        }
        // Segmented: one fixture's model and serial plan; sessions 1 and 2 bring their own state memory.
        KnownRuntimeFixture f(model,true,true);
        // Each further session also holds its own copy of the state layout (as a cohort's further requests do).
        struct Extra {std::vector<std::uint64_t> words;strix::SessionStateMemoryBinding memory;strix::ExecutionRuntimeResources resources;
            strix::SerialSessionContext session;strix::SerialSessionStateLayout layout;};
        std::vector<Extra> extra(sizes.size()-1U);
        for(std::size_t e=0;e<extra.size();++e) {
            auto& x=extra[e];
            x.words.assign(f.state_words.size(),0U);
            x.memory=f.state_memory;x.memory.host_base=reinterpret_cast<std::uintptr_t>(x.words.data());x.memory.host_allocation_generation=40U+e;
            x.resources=f.resources;x.resources.session_state_memory=&x.memory;
            x.layout=*f.resources.session_state_layout;x.resources.session_state_layout=&x.layout;
            x.session=f.session;
            x.session.runtime_compatibility_identity=strix::build_runtime_execution_identities(f.source.serial.execution_identity,
                f.source.serial.arena,x.resources).compatibility_identity;
        }
        const auto session_of=[&](std::size_t s)->strix::SerialSessionContext& {return s?extra[s-1U].session:f.session;};
        const auto resources_of=[&](std::size_t s)->const strix::ExecutionRuntimeResources& {return s?extra[s-1U].resources:f.resources;};
        for(std::size_t s=0;s<sizes.size();++s) {
            strix::DirectLaunchDispatcher dispatcher(f.serial_launch);
            for(std::uint32_t k=0;k<lead_in[s];++k)
                (void)strix::execute_serial_token(f.source.serial,session_of(s),resources_of(s),3U+k*5U+static_cast<std::uint32_t>(s),dispatcher);
        }
        std::uint32_t total=0;for(auto rows:sizes) total+=rows;
        const auto plan=strix::compile_known_prefix_prefill(model,f.source.serial,f.source.manifest,f.source.access,request_for(total+8U,policy));
        f.window_pool.assign(static_cast<std::size_t>((plan.arena.physical.total_bytes+7U)/8U),0U);
        f.window_memory={plan.prefill_execution_identity,reinterpret_cast<std::uintptr_t>(f.window_pool.data()),plan.arena.physical.total_bytes,12U};
        auto source=f.source.serial;source.execution_identity=plan.prefill_execution_identity;source.arena=plan.arena.row_view[0];
        const auto launch=strix::compile_launch_plan(source,f.registry);
        const std::array<strix::KnownWindowRowsEntry,2> wide{{{V::shared_q8_virtual256_r8_v1,&f.trace,&recording_rows_preflight,&recording_rows_launch},
                {V::gdn_q8_rows8_f32_gates_r1_v1,&f.trace,&recording_rows_preflight,&recording_rows_launch}}};
        const std::array<strix::KnownWindowRowsEntry,1> full{{{V::shared_q8_virtual256_r8_v1,&f.trace,&recording_full_preflight,&recording_rows_launch}}};
        std::vector<std::vector<std::uint32_t>> tokens(sizes.size());
        std::vector<strix::KnownPrefillSegment> segments(sizes.size());
        std::vector<strix::CompiledKnownPrefixPrefill> own;
        for(std::size_t s=0;s<sizes.size();++s)
            own.push_back(strix::compile_known_prefix_prefill(model,f.source.serial,f.source.manifest,f.source.access,request_for(sizes[s],policy)));
        for(std::size_t s=0;s<sizes.size();++s) {
            tokens[s]=block_tokens(s,sizes[s]);
            segments[s]={&session_of(s),&resources_of(s),tokens[s],policy==P::terminal_row,own[s].prefill_execution_identity,own[s].prefill_kernel_plan_identity};
        }
        std::vector<std::vector<std::uint64_t>> observed(sizes.size());
        const auto tap=tap_step(f.source.serial);
        strix::KnownPrefillSegmentHooks segment_hooks;
        std::uint32_t shared_launches=0;
        if(shared) segment_hooks.shared=[&](std::size_t step) {
            const bool on=plan.tile8.mode[step]==strix::KnownWindowStepMode::shared_rows;shared_launches+=on;return on;};
        bool one_layout=true;
        segment_hooks.after_step=[&](std::size_t step,std::size_t s,const std::function<strix::ExecutionDispatchContext(std::uint32_t)>& row) {
            one_layout=one_layout&&row(0).session_state_layout==f.resources.session_state_layout;
            if(step!=tap) return;
            for(std::uint32_t r=0;r<sizes[s];++r) observed[s].push_back(read_word(strix::resolve_dispatch_write(row(r),0).address));
        };
        const auto results=strix::execute_known_prefix_prefill_segments_eager(plan,f.source.serial,f.window_memory,segments,launch,{},wide,
            f.hooks(),full,segment_hooks);
        expect(results.size()==sizes.size()&&(!shared||shared_launches>0U),"segmented prefill returns one result per segment");
        expect(one_layout,"every segment's rows carry the lead's state layout object");
        for(std::size_t s=0;s<sizes.size();++s) {
            const auto& session=session_of(s);
            const auto& words=s?extra[s-1U].words:f.state_words;
            expect(words==reference[s].state,"segment state words equal its block alone");
            expect(session.state.generations==reference[s].generations&&session.next_position==reference[s].position&&
                session.committed_tokens==reference[s].position&&session.health==strix::SerialSessionHealth::ready,
                "segment publishes exactly its block's positions and state generations");
            expect(session.arithmetic_lineage.history_identity==reference[s].lineage&&!session.arithmetic_lineage.tainted,
                "segment lineage records its own block's producer");
            expect(observed[s]==reference[s].observed,"segment observer sees its own rows as alone");
            if(policy==P::terminal_row)
                expect(results[s].logits.size()==1U&&results[s].logit_row_indices[0]==sizes[s]-1U&&read_word(results[s].logits[0].address)==reference[s].head,
                    "segment terminal head equals its block alone");
            else expect(results[s].logits.empty(),"state-only segments compute no head");
        }
    }
    {
        // Admission failures leave every session untouched; a dispatch failure poisons every segment once.
        for(unsigned fault=0;fault<4;++fault) {
            KnownRuntimeFixture f(model,true,true);
            std::vector<std::uint64_t> words(f.state_words.size(),0U);
            auto memory=f.state_memory;memory.host_base=reinterpret_cast<std::uintptr_t>(words.data());memory.host_allocation_generation=50U;
            auto resources=f.resources;resources.session_state_memory=&memory;
            auto other=f.session;
            other.runtime_compatibility_identity=strix::build_runtime_execution_identities(f.source.serial.execution_identity,
                f.source.serial.arena,resources).compatibility_identity;
            const auto plan=strix::compile_known_prefix_prefill(model,f.source.serial,f.source.manifest,f.source.access,request_for(32U,P::terminal_row));
            f.window_pool.assign(static_cast<std::size_t>((plan.arena.physical.total_bytes+7U)/8U),0U);
            f.window_memory={plan.prefill_execution_identity,reinterpret_cast<std::uintptr_t>(f.window_pool.data()),plan.arena.physical.total_bytes,13U};
            auto source=f.source.serial;source.execution_identity=plan.prefill_execution_identity;source.arena=plan.arena.row_view[0];
            const auto launch=strix::compile_launch_plan(source,f.registry);
            const std::array<strix::KnownWindowRowsEntry,2> wide{{{V::shared_q8_virtual256_r8_v1,&f.trace,&recording_rows_preflight,&recording_rows_launch},
                {V::gdn_q8_rows8_f32_gates_r1_v1,&f.trace,&recording_rows_preflight,&recording_rows_launch}}};
            std::vector<std::uint32_t> a(16U,7U),b(fault==0?24U:16U,9U);
            if(fault==1) b.back()=static_cast<std::uint32_t>(model.vocab_size);
            if(fault==3) f.trace.fail_position=3U;
            std::vector<strix::KnownPrefillSegment> segments{{&f.session,&f.resources,a,true},{fault==2?&f.session:&other,&resources,b,true}};
            const auto before=f.state_words;
            bool rejected=false;
            try {(void)strix::execute_known_prefix_prefill_segments_eager(plan,f.source.serial,f.window_memory,segments,launch,{},wide,f.hooks(),{},{});}
            catch(const std::exception&) {rejected=true;}
            expect(rejected,"segmented prefill rejects oversized, out-of-vocabulary, duplicate-session or failing blocks");
            if(fault==3) expect(f.session.health==strix::SerialSessionHealth::poisoned&&other.health==strix::SerialSessionHealth::poisoned&&
                f.quiesces==1U&&!f.session.committed_tokens&&!other.committed_tokens,"a dispatch failure poisons every segment and drains once");
            else expect(f.trace.calls==0U&&f.state_words==before&&f.session.health==strix::SerialSessionHealth::ready&&
                other.health==strix::SerialSessionHealth::ready,"segmented admission completes before any mutation");
        }
    }
}

void known_window_output_policy_tests(const strix::ModelContract& model) {
    using P=strix::KnownWindowOutputPolicy;
    for(const std::uint32_t rows:{2U,4U,8U}) {
        KnownRuntimeFixture baseline(model,false,false,rows);
        const auto all=baseline.run(baseline.hooks());
        const auto state=baseline.state_words;const auto generations=baseline.session.state.generations;
        const auto ple=baseline.trace.ple_previous;std::vector<std::uint64_t> expected;
        for(const auto& logit:all.logits)expected.push_back(read_word(logit.address));
        const auto continuation=baseline.serial_token(31U);const auto continued=baseline.state_words;
        const auto explicit_all=strix::compile_known_prefix_window(model,baseline.source.serial,baseline.source.manifest,
            baseline.source.access,{false,false,false,rows,false,P::all_rows});
        expect(explicit_all.window_execution_identity==baseline.window.window_execution_identity&&
            explicit_all.window_kernel_plan_identity==baseline.window.window_kernel_plan_identity,
            "explicit all-rows leaves the default known-window hash recipe and arena intact");
        std::set<std::uint64_t> ids{explicit_all.window_execution_identity},kernels{explicit_all.window_kernel_plan_identity};
        for(const auto policy:{P::terminal_row,P::state_only}) {
            KnownRuntimeFixture f(model,false,false,rows,false,policy);const auto result=f.run(f.hooks());
            const auto heads=policy==P::terminal_row?1U:0U;
            const auto omitted=rows-heads;
            expect(ids.insert(f.window.window_execution_identity).second&&kernels.insert(f.window.window_kernel_plan_identity).second,
                "nondefault head policies receive distinct execution and organization identities");
            expect(result.logits.size()==heads&&result.logit_row_indices==f.window.logit_row_indices&&result.omitted_lm_heads==omitted&&
                f.trace.head_positions==(heads?std::vector<std::uint64_t>{rows-1U}:std::vector<std::uint64_t>{}),
                "only explicitly mapped terminal heads dispatch, and absent outputs have no fake addresses");
            if(heads)expect(result.logit_row_indices.front()==rows-1U&&read_word(result.logits.front().address)==expected.back(),
                "compact terminal output maps to its real original row and complete R1 value");
            std::vector<std::uint64_t> admitted;for(std::uint32_t r=0;r<rows;++r)admitted.push_back(r);
            expect(f.trace.head_admissions==admitted,"every omitted head still receives full launcher metadata admission before mutation");
            expect(f.state_words==state&&f.session.state.generations==generations&&f.trace.ple_previous==ple&&
                f.session.next_position==rows&&f.session.committed_tokens==rows&&f.quiesces==0U&&
                result.row_loop_dispatches==rows*f.source.serial.execution.steps.size()-omitted&&
                result.logical_row_steps==rows*f.source.serial.execution.steps.size()-omitted,
                "head omission preserves all seven state payloads, generations, PLE order and complete row commit");
            expect(f.serial_token(31U)==continuation&&f.state_words==continued,
                "serial continuation after either output policy observes the identical causal state");
            for(unsigned fault=0;fault<2U;++fault) {
                KnownRuntimeFixture bad(model,false,false,rows,false,policy);
                // Row0 is omitted by BOTH policies, but its preflight cannot be bypassed.
                if(fault==0U)bad.row_launch[0].steps.back().preflight=nullptr;
                else bad.row_launch[0].steps.back().preflight=&throwing_window_preflight;
                const auto before=bad.state_words;bool rejected=false;
                try{(void)bad.run(bad.hooks());}catch(const std::exception&){rejected=true;}
                expect(rejected&&bad.trace.calls==0U&&bad.state_words==before&&bad.session.health==strix::SerialSessionHealth::ready&&
                    bad.session.committed_tokens==0U&&bad.session.next_position==0U&&bad.quiesces==0U,
                    "omitted-head metadata failure rejects before any PLE/position/dispatch or poison");
            }
            KnownRuntimeFixture failing(model,false,false,rows,false,policy);failing.trace.fail_kernel=true;
            const auto before=failing.session.state.generations;bool rejected=false;
            try{(void)failing.run(failing.hooks());}catch(const std::runtime_error&){rejected=true;}
            expect(rejected&&failing.session.health==strix::SerialSessionHealth::poisoned&&failing.quiesces==1U&&
                failing.session.arithmetic_lineage.tainted&&failing.session.state.generations==before&&failing.session.committed_tokens==0U,
                "post-submit head-policy failure retains drain/poison and no partial metadata commit");
        }
        for(unsigned fault=0;fault<5U;++fault) {
            auto bad=explicit_all;
            if(fault==0U)bad.output_policy=P::terminal_row;
            if(fault==1U)bad.logit_row_indices={rows-1U};
            if(fault==2U){bad.output_policy=P::terminal_row;bad.logit_row_indices={rows-1U};}
            if(fault==3U)bad.output_policy=static_cast<P>(255U);
            if(fault==4U)bad.logit_row_indices.push_back(rows-1U);
            bool rejected=false;try{strix::validate_compiled_known_prefix_window(bad,baseline.source.serial);}catch(const std::invalid_argument&){rejected=true;}
            expect(rejected,"forged policy, compact row map and unchanged identity cannot bypass window admission");
        }
        auto request=strix::KnownWindowRequest{};request.logical_rows=rows;request.output_policy=static_cast<P>(255U);
        bool rejected=false;try{(void)strix::compile_known_prefix_window(model,baseline.source.serial,baseline.source.manifest,baseline.source.access,request);}
        catch(const std::invalid_argument&){rejected=true;}
        expect(rejected,"unknown requested output policy rejects during compilation");
    }
    for(const auto policy:{P::terminal_row,P::state_only}) {
        auto source=make_known_fixture(model);source.serial.execution.steps.back().writes_state=strix::PersistentState::ngram_history;
        strix::KnownWindowRequest request;request.output_policy=policy;bool rejected=false;
        try{(void)strix::compile_known_prefix_window(model,source.serial,source.manifest,source.access,request);}
        catch(const std::invalid_argument&){rejected=true;}
        expect(rejected,"an output policy cannot admit a stateful or noncanonical terminal head");
    }
}

void actual_known_window_lineage_tests(const strix::ModelContract& model) {
    using namespace strix;
    static_assert(static_cast<unsigned>(ArithmeticProducer::serial_row)==0U);
    static_assert(static_cast<unsigned>(ArithmeticProducer::prefill_chunk)==1U);
    auto chunks=prepare_known_window_chunk_lineage({},0,2,11,22,33);
    chunks=prepare_known_window_chunk_lineage(chunks,2,4,11,22,33);
    expect(arithmetic_lineage_valid(chunks,4) && chunks.intervals.size()==2U &&
        chunks.intervals[0].producer==ArithmeticProducer::known_window_chunk && chunks.intervals[0].begin==0U &&
        chunks.intervals[0].end==2U && chunks.intervals[1].begin==2U && chunks.intervals[1].end==4U,
        "actual closed window lineage preserves whole chunk cuts without coalescing");
    const auto prefill=prepare_prefill_chunk_lineage(prepare_prefill_chunk_lineage({},0,2,11,22,33),2,4,11,22,33);
    const auto unsplit=prepare_known_window_chunk_lineage({},0,4,11,22,33);
    expect(!arithmetic_lineage_matches(chunks,prefill,4) && !arithmetic_lineage_matches(chunks,unsplit,4) &&
        !arithmetic_lineage_serial_only(chunks,11,4),"known window cannot relabel prefill, serial or an intermediate chunk prefix");
    expect(prepare_known_window_chunk_lineage({},7,9,11,22,33).intervals.empty(),
        "unknown prior history remains unknown for actual closed window lineage");
    for(unsigned fault=0;fault<7U;++fault) {
        bool rejected=false;
        try {
            if(fault==0U) (void)prepare_known_window_chunk_lineage({},2,2,11,22,33);
            else if(fault==1U) (void)prepare_known_window_chunk_lineage({},0,2,0,22,33);
            else if(fault==2U) (void)prepare_known_window_chunk_lineage({},0,2,11,0,33);
            else if(fault==3U) (void)prepare_known_window_chunk_lineage({},0,2,11,22,0);
            else if(fault==4U) {auto bad=chunks;bad.tainted=true;(void)prepare_known_window_chunk_lineage(bad,4,6,11,22,33);}
            else if(fault==5U) (void)prepare_known_window_chunk_lineage(chunks,3,5,11,22,33);
            else {auto bad=chunks;++bad.history_identity;(void)prepare_known_window_chunk_lineage(bad,4,6,11,22,33);}
        } catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected,"actual closed window helper rejects invalid bounds, zero recipes and corrupt prior history");
    }
    KnownRuntimeFixture legacy(model);
    (void)legacy.run(legacy.hooks());
    const auto expected_legacy=prepare_prefill_chunk_lineage({},0,2,legacy.window.window_execution_identity,
        legacy.window.window_kernel_plan_identity,legacy.window.window_execution_identity);
    expect(arithmetic_lineage_matches(legacy.session.arithmetic_lineage,expected_legacy,2),
        "disabled recorder preserves the existing complete default prefill history recipe");

    KnownRuntimeFixture base(model);
    auto recorded=base.hooks();recorded.record_actual_window_lineage=true;
    recorded.complete=[&] {expect(base.session.next_position==0U && base.session.arithmetic_lineage.intervals.empty(),
        "known window lineage is not published before actual completion succeeds");};
    (void)base.run(recorded);
    const auto first=base.session.arithmetic_lineage;
    expect(arithmetic_lineage_valid(first,2) && first.intervals.size()==1U &&
        first.intervals.front().producer==ArithmeticProducer::known_window_chunk &&
        first.intervals.front().numerical_plan_identity==base.window.window_execution_identity &&
        !arithmetic_lineage_matches(first,expected_legacy,2),"captured row-loop records its actual distinct window recipe");
    recorded.complete=[&] {expect(base.session.next_position==2U && base.session.arithmetic_lineage.history_identity==first.history_identity,
        "later chunk keeps the previously committed producer until completion");};
    (void)base.run(recorded);
    expect(base.session.arithmetic_lineage.intervals.size()==2U && base.session.arithmetic_lineage.intervals.back().begin==2U &&
        base.session.arithmetic_lineage.intervals.back().end==4U &&
        base.session.arithmetic_lineage.intervals.front().reduction_plan_identity==base.session.arithmetic_lineage.intervals.back().reduction_plan_identity,
        "repeated identical implementations keep separate actual chunks with one stable recipe identity");
    (void)base.serial_token(13U);
    expect(arithmetic_lineage_valid(base.session.arithmetic_lineage,5) && base.session.arithmetic_lineage.intervals.size()==3U &&
        base.session.arithmetic_lineage.intervals.back().producer==ArithmeticProducer::serial_row,
        "real serial continuation appends to the complete actual window history");
    for(unsigned mutation=0;mutation<2U;++mutation) {
        KnownRuntimeFixture changed(model);
        if(mutation==0U) changed.row_launch[1].steps[0].implementation_identity^=0x400U;
        else changed.row_launch[1].plan_identity^=0x400U;
        auto hooks=changed.hooks();hooks.record_actual_window_lineage=true;
        (void)changed.run(hooks);
        expect(!arithmetic_lineage_matches(first,changed.session.arithmetic_lineage,2),
            "actual row implementation or launch-plan changes alter closed window provenance");
    }
    const auto entries=[](KnownRuntimeFixture& fixture,std::uint64_t shared_recipe) {
        return std::array<KnownWindowR2Entry,2>{{
            {KnownWindowR2Variant::shared_q8_virtual256_r2_v1,&fixture.trace,&recording_r2_preflight,&recording_r2_launch,shared_recipe},
            {KnownWindowR2Variant::gdn_q8_virtual256_r2_v1,&fixture.trace,&recording_r2_preflight,&recording_r2_launch,0x5445535447444e31ULL}}};
    };
    KnownRuntimeFixture promoted(model,true),other_recipe(model,true),missing(model,true);
    auto promoted_hooks=promoted.hooks();promoted_hooks.record_actual_window_lineage=true;
    (void)promoted.run_entries(promoted_hooks,entries(promoted,0x5445535453485231ULL));
    auto other_hooks=other_recipe.hooks();other_hooks.record_actual_window_lineage=true;
    (void)other_recipe.run_entries(other_hooks,entries(other_recipe,0x5445535453485232ULL));
    expect(!arithmetic_lineage_matches(promoted.session.arithmetic_lineage,other_recipe.session.arithmetic_lineage,2),
        "actual promoted implementation recipe is bound independently of the unchanged variant enum");
    const auto missing_words=missing.state_words;auto missing_hooks=missing.hooks();missing_hooks.record_actual_window_lineage=true;
    bool rejected=false;
    try {(void)missing.run_entries(missing_hooks,entries(missing,0U));} catch(const std::invalid_argument&) {rejected=true;}
    expect(rejected && missing.trace.calls==0U && missing.state_words==missing_words && missing.quiesces==0U &&
        missing.session.health==SerialSessionHealth::ready,"missing promoted recipe rejects before any dispatch or target mutation");
    KnownRuntimeFixture unknown(model);unknown.session.next_position=2U;unknown.session.committed_tokens=2U;
    auto unknown_hooks=unknown.hooks();unknown_hooks.record_actual_window_lineage=true;rejected=false;
    try {(void)unknown.run(unknown_hooks);} catch(const std::invalid_argument&) {rejected=true;}
    expect(rejected && unknown.trace.calls==0U && unknown.quiesces==0U && unknown.session.next_position==2U &&
        unknown.session.arithmetic_lineage.intervals.empty(),"captured window refuses a fabricated missing prior producer frontier");
    KnownRuntimeFixture failing(model);auto failure_hooks=failing.hooks();failure_hooks.record_actual_window_lineage=true;
    failure_hooks.complete=[] {throw std::runtime_error("injected actual completion failure");};bool failed=false;
    try {(void)failing.run(failure_hooks);} catch(const std::runtime_error&) {failed=true;}
    expect(failed && failing.trace.calls>0U && failing.quiesces==1U && failing.session.health==SerialSessionHealth::poisoned &&
        failing.session.next_position==0U && failing.session.committed_tokens==0U && failing.session.arithmetic_lineage.intervals.empty() &&
        failing.session.arithmetic_lineage.tainted,"failed native completion drains and publishes neither chunk history nor target frontier");
}

void known_window_run_cache_tests(const strix::ModelContract& model) {
    using namespace strix;
    using V = KnownWindowR2Variant;
    for (const auto rows : {2U, 8U}) for (const bool promote : {false, true}) {
        KnownRuntimeFixture reference(model, promote, false, rows), cached(model, promote, false, rows);
        KnownWindowRunCache cache;
        std::vector<const void*> memo;
        std::vector<std::array<std::uintptr_t, 2>> slots;
        auto hooks = cached.hooks();
        hooks.run_cache = &cache; hooks.preflight_memo = &memo; hooks.preflight_slots = &slots;
        std::array<KnownWindowR2Entry, 2> pairs{{
            {V::shared_q8_virtual256_r2_v1, &cached.trace, &recording_r2_preflight, &recording_r2_launch},
            {V::gdn_q8_virtual256_r2_v1, &cached.trace, &recording_r2_preflight, &recording_r2_launch}}};
        std::array<KnownWindowRowsEntry, 1> wide{{
            {V::shared_q8_virtual256_r8_v1, &cached.trace, &recording_rows_preflight, &recording_rows_launch}}};
        auto reference_pairs = pairs;
        for (auto& entry : reference_pairs) entry.config = &reference.trace;
        auto reference_wide = wide; reference_wide[0].config = &reference.trace;
        std::vector<const CompiledLaunchPlan*> plans, reference_plans;
        for (const auto& plan : cached.row_launch) plans.push_back(&plan);
        for (const auto& plan : reference.row_launch) reference_plans.push_back(&plan);
        auto alternate_plans = cached.row_launch;
        auto alternate_memory = cached.state_memory;
        std::array<SerialSessionContext, 2> sessions{cached.session, cached.session};
        const ExecutionDispatchContext* prior_context = nullptr;
        const ExecutionRuntimeResources* prior_resource = nullptr;
        for (unsigned cycle = 0; cycle < 7U; ++cycle) {
            // Cold, warm, owner invalidation, plan-key change, resource rebinding,
            // slot invalidation, then warm again. The session object alternates too.
            if (cycle == 2U) { memo.clear(); slots.clear(); }
            if (cycle == 3U) for (std::size_t r = 0; r < plans.size(); ++r) plans[r] = &alternate_plans[r];
            if (cycle == 4U) cached.resources.session_state_memory = &alternate_memory;
            if (cycle == 5U) slots.clear();
            auto& session = sessions[cycle % 2U];
            session = cached.session;
            for (std::uint32_t r = 0; r < rows; ++r)
                cached.tokens[r] = reference.tokens[r] = 7U + cycle + r;
            hooks.before_step = [&](std::size_t i, const auto& contexts) {
                if (i == 0U) {
                    if (cycle == 1U || cycle == 6U)
                        expect(prior_context == contexts[0].data() && prior_resource == contexts[0][0].resources,
                            "warm window retains contexts and their owned resource copies");
                    prior_context = contexts[0].data(); prior_resource = contexts[0][0].resources;
                }
                return false;
            };
            const auto expected = execute_known_prefix_window_eager(reference.window, reference.source.serial,
                reference.resources, reference.window_memory, reference.session, reference.tokens,
                reference_plans, reference_pairs, reference_wide, reference.hooks());
            const auto actual = execute_known_prefix_window_eager(cached.window, cached.source.serial,
                cached.resources, cached.window_memory, session, cached.tokens, plans, pairs, wide, hooks);
            cached.session = session;
            for (std::size_t r = 0; r < actual.logits.size(); ++r)
                expect(actual.logits[r].bytes == expected.logits[r].bytes &&
                    std::memcmp(reinterpret_cast<const void*>(actual.logits[r].address),
                        reinterpret_cast<const void*>(expected.logits[r].address), actual.logits[r].bytes) == 0,
                    "cached and uncached window logits match byte for byte across cycles");
            expect(cached.state_words == reference.state_words && session.state.valid == reference.session.state.valid &&
                session.state.generations == reference.session.state.generations &&
                session.next_position == reference.session.next_position && session.committed_tokens == reference.session.committed_tokens,
                "cached window preserves all state payloads, generations and session frontiers");
            expect(session.arithmetic_lineage.intervals == reference.session.arithmetic_lineage.intervals &&
                session.arithmetic_lineage.history_identity == reference.session.arithmetic_lineage.history_identity &&
                session.arithmetic_lineage.tainted == reference.session.arithmetic_lineage.tainted,
                "cached window prepares and publishes the same complete arithmetic lineage");
            expect(actual.execution.completion_order == expected.execution.completion_order &&
                actual.execution.completed_steps == expected.execution.completed_steps &&
                actual.r2_dispatches == expected.r2_dispatches && actual.shared_rows_dispatches == expected.shared_rows_dispatches,
                "cached window refreshes per-cycle completion records and dispatch counters");
        }
    }
}

void known_window_runtime_tests(const strix::ModelContract& model) {
    KnownRuntimeFixture fixture(model);
    const std::array<std::uint64_t, 2> expected{fixture.serial_token(7), fixture.serial_token(11)};
    const auto expected_state = fixture.state_words;
    const auto expected_generations = fixture.session.state.generations;
    const auto expected_ple = fixture.trace.ple_previous;
    const auto third = fixture.serial_token(13);
    const auto third_state = fixture.state_words;
    for (const bool promote : {false, true}) {
        KnownRuntimeFixture candidate(model, promote);
        const auto result = candidate.run(candidate.hooks());
        expect(read_word(result.logits[0].address) == expected[0] &&
            read_word(result.logits[1].address) == expected[1] &&
            result.logits[0].address != result.logits[1].address,
            "known R2/all-off host flow preserves both serial row outputs in distinct storage");
        expect(candidate.state_words == expected_state &&
            candidate.session.state.generations == expected_generations &&
            candidate.trace.ple_previous == expected_ple,
            "known layer-major flow preserves seven state recurrences, six generations and PLE row order");
        expect(candidate.session.next_position == 2 && candidate.session.committed_tokens == 2 &&
            read_word(candidate.position_address()) == 1 && candidate.quiesces == 0 &&
            candidate.session.arithmetic_lineage.intervals.size() == 1 &&
            (promote ? result.r2_dispatches == 84 : result.r2_dispatches == 0),
            "known window commits only after completion and records a genuine chunk lineage");
        expect(candidate.serial_token(13) == third && candidate.state_words == third_state &&
            candidate.session.arithmetic_lineage.intervals.size() == 2,
            "serial continuation after a known chunk preserves state and appends serial lineage");
    }
    // A direct mutant witness: absent host materialization makes row0 observe
    // row1's position even though all ordinary kernel arguments still say row0.
    fixture.reset(); write_word(fixture.position_address(), 1);
    strix::ExecutionDispatchContext reader;
    reader.resources = &fixture.resources; reader.position = 0;
    reader.reads_state = strix::PersistentState::position_history;
    bool caught = false;
    try { recording_window_kernel(reader, &fixture.trace); }
    catch (const std::logic_error&) { caught = true; }
    expect(caught, "position reader witness detects omitted row0 materialization");

    // Call the actual fixed-position launcher, independently of the recorder.
    fixture.reset();
    auto position_context = reader;
    position_context.op = strix::SerialOp::position_prepare;
    position_context.step_id = 1;
    const auto position_launcher = strix::make_position_prepare_launcher();
    auto position_words = fixture.state_words;
    for (const std::uint64_t position : {7ULL, 8ULL}) {
        position_context.position = position;
        position_launcher.preflight(position_context, position_launcher.user_data);
        position_launcher.function(position_context, position_launcher.user_data);
        const auto position_index = static_cast<std::size_t>(
            (fixture.position_address() - fixture.state_memory.host_base) / sizeof(std::uint64_t));
        position_words[position_index] = position;
        expect(read_word(fixture.position_address()) == position && fixture.state_words == position_words &&
            fixture.trace.calls == 0,
            "actual position_prepare writes p then p+1 and preserves every surrounding state word");
    }

    const auto reject = [&](const std::function<void(KnownRuntimeFixture&, strix::KnownWindowExecutionHooks&)>& mutate,
                            const char* message, const char* expected_error = nullptr) {
        KnownRuntimeFixture candidate(model);
        auto callbacks = candidate.hooks(); mutate(candidate, callbacks);
        const auto words = candidate.state_words;
        const auto health = candidate.session.health;
        const auto position = candidate.session.next_position;
        const auto committed = candidate.session.committed_tokens;
        bool rejected = false;
        std::string error;
        try { (void)candidate.run(callbacks); }
        catch (const std::exception& exception) { rejected = true; error = exception.what(); }
        expect(rejected && candidate.trace.calls == 0 && candidate.state_words == words &&
            candidate.session.health == health && candidate.session.next_position == position &&
            candidate.session.committed_tokens == committed && candidate.quiesces == 0 &&
            (!expected_error || error == expected_error), message);
    };
    reject([](auto& f, auto&) { ++f.window.arena.row_stride.front(); }, "G0 rejects changed row stride before mutation");
    reject([](auto& f, auto&) { f.window.row_plan[1].arena = f.window.row_plan[0].arena; }, "G0 rejects aliased row views");
    reject([](auto& f, auto&) { f.window_memory.allocation_generation = 0; }, "G0 rejects stale scratch generation");
    reject([](auto& f, auto&) { f.window_memory.arena_bytes = std::numeric_limits<std::uint64_t>::max(); },
        "G0 rejects an overflowing full scratch owner extent");
    reject([](auto& f, auto&) { f.serial_memory.arena_bytes = std::numeric_limits<std::uint64_t>::max(); },
        "G0 rejects an overflowing original serial owner extent");
    reject([](auto& f, auto&) { ++f.window.window_execution_identity; }, "G0 rejects changed window identity");
    reject([](auto& f, auto&) { ++f.session.execution_identity; },
        "G0 rejects a session belonging to a different serial execution before dispatch",
        "known window serial session execution identity mismatch");
    reject([](auto& f, auto&) { f.layout.context_tokens = 1; }, "G0 rejects insufficient two-row capacity");
    reject([](auto& f, auto&) { f.tokens[1] = static_cast<std::uint32_t>(f.source.serial.vocab_size); },
        "G0 rejects an out-of-vocabulary row1 token");
    reject([](auto& f, auto&) { f.session.committed_tokens = std::numeric_limits<std::uint64_t>::max(); },
        "G0 rejects committed-token overflow");
    reject([](auto& f, auto&) { f.session.health = strix::SerialSessionHealth::poisoned; }, "G0 refuses a poisoned session");
    reject([](auto& f, auto&) { f.session.arithmetic_lineage.tainted = true; }, "G0 refuses tainted lineage");
    reject([](auto& f, auto&) {
        f.session.arithmetic_lineage = strix::prepare_serial_row_lineage({}, 0,
            f.source.serial.execution_identity, f.source.serial.kernel_plan_identity);
    }, "G0 rejects producer history whose endpoint differs from the session position");
    reject([](auto& f, auto&) { f.session.state.generations[0] = std::numeric_limits<std::uint64_t>::max(); },
        "G0 rejects persistent generation overflow");
    reject([](auto&, auto& h) { h.preflight_scope = [] { throw std::runtime_error("scope fault"); }; },
        "owned-scope admission failure dispatches nothing and preserves ready state");
    reject([](auto&, auto& h) { h.preflight_plan = [] { throw std::runtime_error("deep plan fault"); }; },
        "deep R1 admission failure dispatches nothing and preserves ready state");
    reject([](auto& f, auto&) { ++f.row_launch[1].execution_identity; }, "G0 rejects an unrelated row launch plan");
    reject([](auto& f, auto&) { f.row_launch[1].steps.back().preflight = nullptr; },
        "G0 refuses a row-loop launcher without a pure metadata preflight");
    reject([](auto& f, auto&) { f.row_launch[1].steps.back().preflight = &throwing_window_preflight; },
        "late row1 head preflight rejects before any row0 launch or position write");
    reject([](auto&, auto& h) { h.position_reader_implementations = {}; }, "G0 refuses unadmitted position readers");
    reject([](auto& f, auto&) { f.row_launch[0].steps[1].function = &recording_window_kernel; },
        "G0 rejects a fake position handler even with the incumbent implementation ID");
    reject([](auto& f, auto&) {
        const auto verify = strix::make_position_verify_window_launcher();
        auto& position = f.row_launch[1].steps[1];
        position.function = verify.function;
        position.implementation_identity = verify.implementation_identity;
        // Retain the pure metadata callback so C2, rather than a missing
        // preflight, must reject the actual verify-window implementation.
    }, "G0 rejects the actual verify-window position launcher at step1 before either row runs",
        "known window position-history reader implementation is not admitted");
    reject([](auto& f, auto&) {
        const auto qsa = std::find_if(f.source.serial.execution.steps.begin(), f.source.serial.execution.steps.end(),
            [](const auto& step) { return step.op == strix::SerialOp::indexed_attention; });
        if (qsa == f.source.serial.execution.steps.end()) throw std::logic_error("fixture requires a QSA step");
        f.window.mode[qsa->id] = strix::KnownWindowStepMode::r2_gdn_staged;
        f.window.r2_variant[qsa->id] = strix::KnownWindowR2Variant::gdn_q8_virtual256_r2_v1;
    }, "G0 window validation rejects a forged QSA position-reader R2 entry before callbacks",
        "known window GDN variant/source ID mismatch");
    reject([](auto& f, auto&) {
        write_word(f.position_address(), 0x123456789abcdef0ULL);
        auto& position = f.layout.allocations.back();
        f.state_memory.device_base = f.position_address();
        f.state_memory.device_bytes = position.bytes;
        f.state_memory.device_allocation_generation = 9;
        f.layout.device_total_bytes = position.bytes;
        position.storage = strix::SessionStateStorage::device;
        position.offset = 0;
    }, "G0 rejects device position storage with zero dispatches, ready session and unchanged sentinel",
        "known window requires writable bounded host position history");
    reject([](auto& f, auto&) {
        write_word(f.position_address(), 0x123456789abcdef0ULL);
        f.layout.allocations.back().bytes = sizeof(std::uint64_t) - 1U;
    }, "G0 rejects a short host position view with zero dispatches, ready session and unchanged sentinel",
        "known window requires writable bounded host position history");
    reject([](auto& f, auto&) {
        write_word(f.position_address(), 0x123456789abcdef0ULL);
        f.state_memory.host_base = 0;
    }, "G0 rejects a null host position pool with zero dispatches, ready session and unchanged sentinel",
        "host session-state pool has a null base");

    {
        KnownRuntimeFixture candidate(model);
        const auto generations = candidate.session.state.generations;
        const auto valid = candidate.session.state.valid;
        const auto generation = candidate.state_memory.host_allocation_generation;
        const auto compatibility = candidate.session.runtime_compatibility_identity;
        candidate.trace.rebind_after_qsa = &candidate.state_memory;
        std::string error;
        try { (void)candidate.run(candidate.hooks()); }
        catch (const std::invalid_argument& exception) { error = exception.what(); }
        expect(error == "known window host position binding changed after admission" &&
            candidate.trace.admissions > 0 && candidate.trace.calls > 0 && candidate.trace.rebindings == 1 &&
            candidate.trace.qsa_positions == std::vector<std::uint64_t>{0} &&
            candidate.state_memory.host_allocation_generation == generation + 1 &&
            candidate.session.health == strix::SerialSessionHealth::poisoned && candidate.quiesces == 1 &&
            candidate.session.arithmetic_lineage.tainted && candidate.session.arithmetic_lineage.intervals.empty() &&
            candidate.session.state.generations == generations && candidate.session.state.valid == valid &&
            candidate.session.next_position == 0 && candidate.session.committed_tokens == 0 &&
            candidate.session.runtime_compatibility_identity == compatibility && read_word(candidate.position_address()) == 0,
            "a generation swap inside QSA is re-resolved before row1, poisons and drains once without metadata publication");
    }
    for (const bool zero : {false, true}) {
        KnownRuntimeFixture candidate(model);
        StartFaultLookup lookup(zero);
        const auto hash = strix::build_ple_hash_contract(model);
        strix::PleLookupPrefetchState prefetch;
        candidate.resources.ple_lookup = &lookup;
        candidate.resources.ple_hash_contract = &hash;
        candidate.resources.ple_prefetch = &prefetch;
        candidate.session.runtime_compatibility_identity = strix::build_runtime_execution_identities(
            candidate.source.serial.execution_identity, candidate.source.serial.arena,
            candidate.resources).compatibility_identity;
        const auto words = candidate.state_words;
        caught = false;
        try { (void)candidate.run(candidate.hooks()); } catch (const std::runtime_error&) { caught = true; }
        expect(caught && lookup.starts == 1 && lookup.cancels == 0 && !prefetch.active &&
            candidate.trace.calls == 0 && candidate.state_words == words && candidate.quiesces == 0 &&
            candidate.session.health == strix::SerialSessionHealth::ready &&
            !candidate.session.arithmetic_lineage.tainted && candidate.session.committed_tokens == 0,
            "PLE start throw/zero handle preserves ready state without cancelling an unknown handle");
    }
    for (const bool completion_fault : {false, true}) {
        KnownRuntimeFixture candidate(model);
        auto callbacks = candidate.hooks();
        if (completion_fault) callbacks.complete = [&candidate] {
            expect(candidate.session.committed_tokens == 0 && candidate.session.next_position == 0,
                "completion observes uncommitted window metadata");
            throw std::runtime_error("completion fault");
        };
        else candidate.trace.fail_kernel = true;
        caught = false;
        try { (void)candidate.run(callbacks); } catch (const std::runtime_error&) { caught = true; }
        expect(caught && candidate.trace.calls > 0 && candidate.session.health == strix::SerialSessionHealth::poisoned &&
            candidate.session.arithmetic_lineage.tainted && candidate.session.committed_tokens == 0 &&
            candidate.session.next_position == 0 && candidate.quiesces == 1,
            "mid-dispatch/completion failure poisons, taints and drains exactly once without metadata commit");
    }
    {
        KnownRuntimeFixture candidate(model);
        candidate.trace.fail_qsa = true;
        const auto generations = candidate.session.state.generations;
        caught = false;
        try { (void)candidate.run(candidate.hooks()); }
        catch (const std::runtime_error& exception) {
            caught = std::string(exception.what()) == "injected row1 QSA failure";
        }
        expect(caught && candidate.trace.qsa_positions == std::vector<std::uint64_t>{0, 1} &&
            candidate.session.health == strix::SerialSessionHealth::poisoned &&
            candidate.session.arithmetic_lineage.tainted && candidate.session.arithmetic_lineage.intervals.empty() &&
            candidate.session.state.generations == generations && candidate.session.committed_tokens == 0 &&
            candidate.session.next_position == 0 && candidate.quiesces == 1,
            "an actual row1 QSA dispatch failure poisons and drains once without publishing generations or chunk metadata");
    }
}

// Full causal core coverage with honest host-recording launchers. Native
// projection math/owner integration is separately covered by the opt-in driver.
void known_window_wide_runtime_tests(const strix::ModelContract& model) {
    using V=strix::KnownWindowR2Variant;
    for(const std::uint32_t rows:{4U,8U}) for(const bool promote:{false,true}) for(const bool mixed:{false,true}) {
        KnownRuntimeFixture baseline(model,false,mixed,rows);
        std::vector<std::uint64_t> expected;
        for(const auto token:baseline.tokens) expected.push_back(baseline.serial_token(token));
        const auto expected_words=baseline.state_words;
        const auto expected_generations=baseline.session.state.generations;
        const auto expected_ple=baseline.trace.ple_previous;
        const auto continuation=baseline.serial_token(29);
        const auto continuation_state=baseline.state_words;
        KnownRuntimeFixture candidate(model,promote,mixed,rows);
        const std::array<strix::KnownWindowR2Entry,2> pairs{{
            {V::gdn_q8_virtual256_r2_v1,&candidate.trace,&recording_r2_preflight,&recording_r2_launch},
            {V::gdn_q8_rows2_f32_gates_r1_v1,&candidate.trace,&recording_r2_preflight,&recording_r2_launch}}};
        const std::array<strix::KnownWindowRowsEntry,2> wide{{
            {rows==4U?V::shared_q8_virtual256_r4_v1:V::shared_q8_virtual256_r8_v1,
                &candidate.trace,&recording_rows_preflight,&recording_rows_launch},
            {rows==4U?V::gdn_q8_rows4_f32_gates_r1_v1:V::gdn_q8_rows8_f32_gates_r1_v1,
                &candidate.trace,&recording_rows_preflight,&recording_rows_launch}}};
        std::vector<const strix::CompiledLaunchPlan*> plans;
        for(const auto& plan:candidate.row_launch) plans.push_back(&plan);
        const auto result=strix::execute_known_prefix_window_eager(candidate.window,candidate.source.serial,
            candidate.resources,candidate.window_memory,candidate.session,candidate.tokens,plans,pairs,wide,candidate.hooks());
        std::set<std::uintptr_t> output_addresses;
        expect(result.logits.size()==rows,"wide core returns every full-head row");
        for(std::uint32_t row=0;row<rows;++row) {
            expect(read_word(result.logits[row].address)==expected[row],"wide core retains every serial row output");
            output_addresses.insert(result.logits[row].address);
        }
        expect(output_addresses.size()==rows && candidate.state_words==expected_words &&
            candidate.session.state.generations==expected_generations && candidate.trace.ple_previous==expected_ple,
            "wide row arenas, seven recurrences, six generations and causal PLE match serial");
        expect(candidate.session.next_position==rows && candidate.session.committed_tokens==rows &&
            read_word(candidate.position_address())==rows-1U && candidate.quiesces==0 &&
            candidate.session.arithmetic_lineage.intervals.size()==1U,
            "wide core commits the whole known frontier exactly once after completion");
        expect(result.shared_rows_dispatches==(promote?48U:0U) &&
            result.gdn_rows_dispatches==(promote&&mixed?36U:0U) &&
            result.r2_dispatches==(promote&&!mixed?36U*(rows/2U):0U),
            "actual wide shared/GDN and contiguous R2 tile dispatches remain distinct");
        expect(candidate.serial_token(29)==continuation && candidate.state_words==continuation_state,
            "serial continuation after wide commit preserves all recurrent state");
    }
    for(const std::uint32_t rows:{4U,8U}) {
        for(const unsigned fault:{0U,1U,2U,3U,4U}) {
            KnownRuntimeFixture candidate(model,false,false,rows);
            auto hooks=candidate.hooks();
            if(fault==0U) candidate.window.arena.row_view.back().allocations[0].offset=candidate.window.arena.row_view[0].allocations[0].offset;
            if(fault==1U) candidate.tokens.back()=static_cast<std::uint32_t>(model.vocab_size);
            if(fault==2U) candidate.row_launch.back().steps.back().preflight=&throwing_window_preflight;
            if(fault==3U) candidate.session.next_position=64U-rows+1U;
            if(fault==4U) --candidate.window.logical_rows;
            const auto original=candidate.state_words;
            bool rejected=false;
            try {(void)candidate.run(hooks);} catch(const std::exception&) {rejected=true;}
            expect(rejected && candidate.trace.calls==0U && candidate.state_words==original &&
                candidate.session.health==strix::SerialSessionHealth::ready && candidate.quiesces==0U,
                "late wide row/token/arena/geometry/capacity fault rejects before state or dispatch");
        }
        for(const bool completion_failure:{false,true}) {
            KnownRuntimeFixture failing(model,false,false,rows);
            auto hooks=failing.hooks();
            if(completion_failure) hooks.complete=[] {throw std::runtime_error("wide completion failure");};
            else failing.trace.fail_position=rows-1U;
            bool threw=false;try {(void)failing.run(hooks);} catch(const std::runtime_error&) {threw=true;}
            expect(threw && failing.trace.calls>0U && failing.quiesces==1U &&
                failing.session.health==strix::SerialSessionHealth::poisoned && failing.session.arithmetic_lineage.tainted &&
                failing.session.next_position==0U && failing.session.committed_tokens==0U,
                "late wide dispatch/completion failure drains once and poisons without publishing a partial commit");
        }
        KnownRuntimeFixture candidate(model,true,true,rows);
        std::vector<const strix::CompiledLaunchPlan*> plans;
        for(const auto& plan:candidate.row_launch) plans.push_back(&plan);
        bool rejected=false;
        try {(void)strix::execute_known_prefix_window_eager(candidate.window,candidate.source.serial,candidate.resources,
            candidate.window_memory,candidate.session,candidate.tokens,plans,{}, {},candidate.hooks());}
        catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected && !candidate.trace.calls && !candidate.quiesces,
            "missing wide typed entry never falls through to a pair or an arbitrary launcher");
        const auto shared=rows==4U?V::shared_q8_virtual256_r4_v1:V::shared_q8_virtual256_r8_v1;
        const std::array<strix::KnownWindowRowsEntry,2> duplicated{{
            {shared,&candidate.trace,&recording_rows_preflight,&recording_rows_launch},
            {shared,&candidate.trace,&recording_rows_preflight,&recording_rows_launch}}};
        rejected=false;
        try {(void)strix::execute_known_prefix_window_eager(candidate.window,candidate.source.serial,candidate.resources,
            candidate.window_memory,candidate.session,candidate.tokens,plans,{},duplicated,candidate.hooks());}
        catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected && !candidate.trace.calls && !candidate.quiesces,
            "ambiguous wide typed entries reject before the first dispatch");
    }
}

void known_window_routed_union_runtime_tests(const strix::ModelContract& model) {
    using V=strix::KnownWindowR2Variant;
    for(const std::uint32_t rows:{2U,4U,8U}) {
        KnownRuntimeFixture baseline(model,false,false,rows,true);
        std::vector<std::uint64_t> expected;
        for(const auto token:baseline.tokens) expected.push_back(baseline.serial_token(token));
        const auto state=baseline.state_words;
        const auto generations=baseline.session.state.generations;
        KnownRuntimeFixture grouped(model,false,false,rows,true);
        const auto variant=rows==2U?V::routed_q51_union_rows2_v1:
            rows==4U?V::routed_q51_union_rows4_v1:V::routed_q51_union_rows8_v1;
        const std::array<strix::KnownWindowRowsEntry,1> entries{{
            {variant,&grouped.trace,&recording_rows_preflight,&recording_rows_launch}}};
        std::vector<const strix::CompiledLaunchPlan*> plans;
        for(const auto& plan:grouped.row_launch) plans.push_back(&plan);
        const auto result=strix::execute_known_prefix_window_eager(grouped.window,grouped.source.serial,
            grouped.resources,grouped.window_memory,grouped.session,grouped.tokens,plans,{},entries,grouped.hooks());
        expect(grouped.state_words==state && grouped.session.state.generations==generations &&
            grouped.session.next_position==rows && grouped.session.committed_tokens==rows,
            "typed expert union scheduling retains all seven states and exactly one full-window frontier");
        for(std::uint32_t row=0;row<rows;++row)
            expect(read_word(result.logits[row].address)==expected[row],"expert union preserves every row's complete output");
        expect(result.routed_expert_union_dispatches==48U && result.gdn_rows_dispatches==0U &&
            result.shared_rows_dispatches==0U && result.r2_dispatches==0U &&
            result.r2_variant_dispatches[static_cast<std::size_t>(variant)]==48U &&
            result.logical_projection_weight_bytes_avoided==0U && result.projection_launches_saved==0U,
            "expert union dispatches are explicit and do not masquerade as GDN or static savings");
        KnownRuntimeFixture missing(model,false,false,rows,true);
        plans.clear();for(const auto& plan:missing.row_launch) plans.push_back(&plan);
        bool rejected=false;
        try{(void)strix::execute_known_prefix_window_eager(missing.window,missing.source.serial,
            missing.resources,missing.window_memory,missing.session,missing.tokens,plans,{}, {},missing.hooks());}
        catch(const std::invalid_argument&){rejected=true;}
        expect(rejected && missing.trace.calls==0U && missing.quiesces==0U && missing.session.next_position==0U,
            "missing grouped expert callback rejects before dispatch or state mutation");
    }
}

void known_window_qsa_bf16_indexer_tests(const strix::ModelContract& model) {
    using V=strix::KnownWindowR2Variant;
    const auto fixture=make_known_fixture(model,0U,false,false,true,30U);
    constexpr std::uint64_t layer_bytes=80ULL*34ULL*(12288ULL+512ULL+512ULL)+192ULL*34ULL*2560ULL+2560ULL*640ULL*2ULL;
    for(const std::uint32_t rows:{2U,4U,8U}) {
        strix::KnownWindowRequest request{true,true,true,rows};
        const auto off=strix::compile_known_prefix_window(model,fixture.serial,fixture.manifest,fixture.access,request);
        request.qsa_rows=true;
        const auto on=strix::compile_known_prefix_window(model,fixture.serial,fixture.manifest,fixture.access,request);
        const auto expected=rows==2U?V::qsa_q8_bf16idx_rows2_r1_v1:rows==4U?V::qsa_q8_bf16idx_rows4_r1_v1:V::qsa_q8_bf16idx_rows8_r1_v1;
        expect(on.qsa_rows_steps==12U && std::count(on.r2_variant.begin(),on.r2_variant.end(),expected)==12 &&
            on.logical_projection_weight_bytes_avoided==off.logical_projection_weight_bytes_avoided+12U*layer_bytes*(rows-1U),
            "real-checkpoint BF16 indexer projections promote every QSA step to its own variant");
        strix::validate_known_prefix_window_formats(on,fixture.serial,fixture.manifest,fixture.access);
        auto mixed=fixture; // A Q8/BF16 indexer pair matches neither closed variant.
        for(auto& t:mixed.manifest.tensors) if(t.name=="blk.3.indexer.k_proj.weight") {t.ggml_type=8U;t.byte_size=80U*34U*128U;}
        const auto fallback=strix::compile_known_prefix_window(model,mixed.serial,mixed.manifest,mixed.access,request);
        expect(fallback.qsa_rows_steps==11U,"mixed Q8/BF16 indexer pair keeps its serial row loop");
        bool rejected=false;
        try{strix::validate_known_prefix_window_formats(on,mixed.serial,mixed.manifest,mixed.access);}catch(const std::invalid_argument&){rejected=true;}
        expect(rejected,"a compiled BF16-indexer variant cannot inherit a changed indexer format");
        auto short_bf16=fixture;
        for(auto& t:short_bf16.manifest.tensors) if(t.name=="blk.7.indexer.q_proj.weight") --t.byte_size;
        rejected=false;
        try{(void)strix::compile_known_prefix_window(model,short_bf16.serial,short_bf16.manifest,short_bf16.access,request);}
        catch(const std::invalid_argument&){rejected=true;}
        expect(rejected,"short BF16 indexer weight rejects before callbacks");
    }
}

void known_window_trellis_profile_tests(const strix::ModelContract& model) {
    auto parent = make_known_fixture(model, 0U, false, false, true);
    const auto add_marker = [](auto& fixture, std::string_view marker) {
        for (auto& binding : fixture.serial.bindings.bindings) {
            if ((binding.op != strix::SerialOp::gated_delta_net && binding.op != strix::SerialOp::indexed_attention) || !binding.kernel_id) continue;
            const auto at = binding.kernel_id->find(".prefetch8");
            expect(at != std::string::npos, "dense fixture has the exact prefetch suffix");
            binding.kernel_id->insert(at, marker);
        }
    };
    add_marker(parent, ".dense-a8-v1");
    auto compact = parent;
    add_marker(compact, ".trellis-g1-fp32-s4-v1");
    for (const auto rows : {2U, 4U, 8U}) for (const bool deferred : {false, true}) {
        strix::KnownWindowRequest request{false, true, true, rows};
        request.qsa_rows = true; request.deferred_gdn = deferred;
        const auto a = strix::compile_known_prefix_window(model, parent.serial, parent.manifest, parent.access, request);
        const auto b = strix::compile_known_prefix_window(model, compact.serial, compact.manifest, compact.access, request);
        strix::validate_compiled_known_prefix_window(b, compact.serial);
        expect(b.qsa_rows_steps == 12U && (deferred ? b.gdn_deferred_rows_steps == 36U : b.r2_steps + b.gdn_rows_steps == 36U),
            "compact G1 promotes every selected GDN and QSA with its separate identity");
        expect(a.window_kernel_plan_identity != b.window_kernel_plan_identity && a.window_execution_identity != b.window_execution_identity,
            "compact G1 window cannot alias the parent arithmetic identity");
        expect(a.logical_projection_weight_bytes_avoided - b.logical_projection_weight_bytes_avoided == 411893760ULL * (rows - 1U),
            "compact window counts K5 payload and signed scales rather than source Q8 bytes");
        expect(b.projection_launches_saved - a.projection_launches_saved == 144ULL * (rows - 1U),
            "compact window accounts for three additional projection launches per selected serial row");
    }
    for (const auto marker : {".trellis-g1-fp32-s4-v1", ".dense-a8-v1.trellis-g1-fp32-s4-v2"}) {
        auto invalid = make_known_fixture(model, 0U, false, false, true);
        add_marker(invalid, marker);
        strix::KnownWindowRequest request{false, true, true, 8U};
        request.qsa_rows = true; request.deferred_gdn = true;
        const auto plan = strix::compile_known_prefix_window(model, invalid.serial, invalid.manifest, invalid.access, request);
        expect(plan.qsa_rows_steps == 0U && plan.gdn_deferred_rows_steps == 0U,
            "compact promotion excludes a missing parent marker and unknown arithmetic versions");
    }
}

void known_window_qsa_tests(const strix::ModelContract& model) {
    using V=strix::KnownWindowR2Variant;
    using M=strix::KnownWindowStepMode;
    const auto fixture=make_known_fixture(model,0U,false,false,true);
    const auto reject=[&](auto&& operation,const char* message) {
        bool rejected=false;try{operation();}catch(const std::invalid_argument&){rejected=true;}expect(rejected,message);
    };
    {
        strix::KnownPrefillRequest request;request.logical_rows=64;request.r2_shared_expert=true;request.r2_gdn=true;
        request.qsa_rows=true;request.full_qsa_score_ring_entry=true;request.output_policy=strix::KnownWindowOutputPolicy::terminal_row;
        request.workspace_budget_bytes=UINT64_MAX;request.host_metadata_payload_budget_bytes=UINT64_MAX;
        const auto plan=strix::compile_known_prefix_prefill(model,fixture.serial,fixture.manifest,fixture.access,request);
        strix::validate_compiled_known_prefix_prefill(plan,fixture.serial);
        expect(plan.external_qsa_score_bytes==262144U && plan.qsa_score_ring_rows==512U && plan.qsa_score_ring_workspace_bytes==335544320U,
            "eligible prefill owns two bounded external score/maxima slots");
        expect(plan.terminal_logits_arena,"terminal arena defaults on for eligible output policy");
        request.compact_stage_images=false;
        const auto fp32_images=strix::compile_known_prefix_prefill(model,fixture.serial,fixture.manifest,fixture.access,request);
        request.compact_stage_images=true;
        expect(plan.compact_qsa_output_savings==(strix::compact_stage_images_enabled()?12288U:0U),"compact output follows its startup switch");
        expect(plan.prefill_kernel_plan_identity==fp32_images.prefill_kernel_plan_identity,
            "compact storage does not change numerical identity");
        expect((plan.prefill_execution_identity!=fp32_images.prefill_execution_identity)==(plan.compact_qsa_output_savings!=0U),
            "only the physical execution identity changes with compact storage");
        const auto qsa_extent=[](const auto& p) {
            for(const auto& a:p.arena.row_view[0].allocations)
                if(a.use.buffer.buffer_class==strix::ExecutionBufferClass::indexed_attention_workspace) return a.bytes;
            throw std::runtime_error("missing QSA allocation");
        };
        expect(qsa_extent(fp32_images)-qsa_extent(plan)==plan.compact_qsa_output_savings,
            "compiled row extents remove exactly the compact output savings");
        auto bad_image=plan;bad_image.compact_qsa_output_savings+=256U;
        reject([&]{strix::validate_compiled_known_prefix_prefill(bad_image,fixture.serial);},"forged compact image geometry rejects");

        for(unsigned fault=0;fault<4;++fault) {
            auto malformed=plan;
            if(fault==0) malformed.arena.row_view.front().allocations.back().bytes=993280U;
            if(fault==1) ++malformed.arena.physical.allocations.back().offset;
            if(fault==2) malformed.arena.row_stride.back()=993280U;
            if(fault==3) malformed.arena.row_view.back().allocations.back().bytes=0;
            reject([&]{strix::validate_compiled_known_prefix_prefill(malformed,fixture.serial);},"malformed mixed logits extent rejects");
        }
        auto bad=plan;++bad.external_qsa_score_bytes;
        reject([&]{strix::validate_compiled_known_prefix_prefill(bad,fixture.serial);},"forged score-free geometry rejects");
        request.exact_memory=false;
        const auto off=strix::compile_known_prefix_prefill(model,fixture.serial,fixture.manifest,fixture.access,request);
        expect(!off.external_qsa_score_bytes && !off.qsa_score_ring_workspace_bytes && off.arena.physical.total_bytes==998400ULL*64,
            "score ring disabled retains the original arena byte receipt");
        const auto set=[](const char* name,const char* value) {
#ifdef _WIN32
            _putenv_s(name,value?value:"");
#else
            if(value) setenv(name,value,1);else unsetenv(name);
#endif
        };
        std::vector<std::pair<std::string,std::optional<std::string>>> saved;
        for(const char* name:{"STRIX_CTX_EXACT_MEMORY","STRIX_QSA_SCORE_RING","STRIX_TERMINAL_LOGITS_ARENA","STRIX_PREFILL_FULL_QSA","STRIX_QSA_SCORE_RING_ROWS",
                "STRIX_QPF2K_DEFAULTS","STRIX_BASELINE"}) {
            const char* value=std::getenv(name);saved.emplace_back(name,value?std::optional<std::string>(value):std::nullopt);
        }
        request.exact_memory=true;
        for(const char* rows : {"64","128","256","512","1024"}) {
            set("STRIX_QSA_SCORE_RING_ROWS",rows);
            const auto configured=strix::compile_known_prefix_prefill(model,fixture.serial,fixture.manifest,fixture.access,request);
            strix::validate_compiled_known_prefix_prefill(configured,fixture.serial);
            expect(configured.qsa_score_ring_rows==std::stoul(rows) &&
                configured.qsa_score_ring_workspace_bytes==static_cast<std::uint64_t>(configured.qsa_score_ring_rows)*655360U,
                "compiled owner receipt prices explicit score/maxima rows");
            expect(configured.prefill_kernel_plan_identity==plan.prefill_kernel_plan_identity &&
                ((configured.prefill_execution_identity==plan.prefill_execution_identity)==(configured.qsa_score_ring_rows==plan.qsa_score_ring_rows)),
                "slot geometry changes execution identity and retains numerical identity");
            auto forged=configured;forged.qsa_score_ring_rows=192U;
            reject([&]{strix::validate_compiled_known_prefix_prefill(forged,fixture.serial);},"forged ring row geometry rejects");
            forged=configured;++forged.qsa_score_ring_workspace_bytes;
            reject([&]{strix::validate_compiled_known_prefix_prefill(forged,fixture.serial);},"forged ring byte charge rejects");
        }
        set("STRIX_QSA_SCORE_RING_ROWS","192");
        reject([&]{(void)strix::compile_known_prefix_prefill(model,fixture.serial,fixture.manifest,fixture.access,request);},"bad ring row environment rejects at admission");
        // Validation uses the admitted geometry, never the current environment.
        strix::validate_compiled_known_prefix_prefill(plan,fixture.serial);
        set("STRIX_QSA_SCORE_RING_ROWS",nullptr);
        request.exact_memory=true;
        set("STRIX_CTX_EXACT_MEMORY","1");set("STRIX_QSA_SCORE_RING","0");set("STRIX_TERMINAL_LOGITS_ARENA","0");
        const auto switches_off=strix::compile_known_prefix_prefill(model,fixture.serial,fixture.manifest,fixture.access,request);
        expect(switches_off.prefill_execution_identity==off.prefill_execution_identity &&
            switches_off.arena.physical.total_bytes==off.arena.physical.total_bytes && switches_off.arena.row_stride==off.arena.row_stride,
            "both off switches preserve byte-identical legacy compiled plans");
        set("STRIX_CTX_EXACT_MEMORY","0");set("STRIX_QSA_SCORE_RING","1");set("STRIX_TERMINAL_LOGITS_ARENA","1");
        const auto master_off=strix::compile_known_prefix_prefill(model,fixture.serial,fixture.manifest,fixture.access,request);
        expect(master_off.prefill_execution_identity==off.prefill_execution_identity &&
            master_off.prefill_kernel_plan_identity==plan.prefill_kernel_plan_identity,
            "master off wins over individual on switches and preserves numerical identity");
        set("STRIX_CTX_EXACT_MEMORY","1");set("STRIX_QSA_SCORE_RING","0");
        const auto terminal_only=strix::compile_known_prefix_prefill(model,fixture.serial,fixture.manifest,fixture.access,request);
        expect(terminal_only.terminal_logits_arena && !terminal_only.external_qsa_score_bytes,"terminal stripe can be enabled independently");
        set("STRIX_QSA_SCORE_RING","1");set("STRIX_TERMINAL_LOGITS_ARENA","0");
        const auto ring_only=strix::compile_known_prefix_prefill(model,fixture.serial,fixture.manifest,fixture.access,request);
        expect(!ring_only.terminal_logits_arena && ring_only.external_qsa_score_bytes,"score ring can be enabled independently");
        set("STRIX_PREFILL_FULL_QSA","0");
        const auto full_off=strix::compile_known_prefix_prefill(model,fixture.serial,fixture.manifest,fixture.access,request);
        expect(!full_off.external_qsa_score_bytes && !full_off.qsa_score_ring_workspace_bytes,
            "FULL_QSA=0 selects embedded scores at compilation");
        set("STRIX_PREFILL_FULL_QSA","1");
        // The defaults policy (--baseline, STRIX_BASELINE=1, STRIX_QPF2K_DEFAULTS=0, GSQ) compiles the legacy layout.
        // Both policies are on first, so each suppression below removes the score ring and the terminal stripe.
        set("STRIX_TERMINAL_LOGITS_ARENA",nullptr);
        const auto both_on=strix::compile_known_prefix_prefill(model,fixture.serial,fixture.manifest,fixture.access,request);
        expect(both_on.terminal_logits_arena && both_on.external_qsa_score_bytes && both_on.qsa_score_ring_workspace_bytes,
            "suppression matrix starts with both compact-memory policies enabled");
        const auto legacy=[&](const char* what) {
            const auto compiled=strix::compile_known_prefix_prefill(model,fixture.serial,fixture.manifest,fixture.access,request);
            expect(!compiled.external_qsa_score_bytes && !compiled.qsa_score_ring_workspace_bytes && !compiled.terminal_logits_arena &&
                compiled.prefill_execution_identity==off.prefill_execution_identity &&
                compiled.arena.physical.total_bytes==off.arena.physical.total_bytes,what);
        };
        set("STRIX_QPF2K_DEFAULTS","0"); legacy("STRIX_QPF2K_DEFAULTS=0 compiles the legacy layout"); set("STRIX_QPF2K_DEFAULTS",nullptr);
        set("STRIX_QPF2K_DEFAULTS","00"); legacy("STRIX_QPF2K_DEFAULTS=00 compiles the legacy layout"); set("STRIX_QPF2K_DEFAULTS",nullptr);
        set("STRIX_BASELINE","1"); legacy("STRIX_BASELINE=1 compiles the legacy layout"); set("STRIX_BASELINE",nullptr);
        strix::qpf2k::baseline=true; legacy("--baseline compiles the legacy layout"); strix::qpf2k::baseline=false;
        { const strix::qpf2k::ModelScope gsq(true); legacy("GSQ checkpoints compile the legacy layout"); }
        set("STRIX_TERMINAL_LOGITS_ARENA","0");
        request.full_qsa_score_ring_entry=false;
        const auto optional_entry=strix::compile_known_prefix_prefill(model,fixture.serial,fixture.manifest,fixture.access,request);
        expect(!optional_entry.external_qsa_score_bytes && optional_entry.prefill_execution_identity==full_off.prefill_execution_identity,
            "optional or missing full entry retains the same embedded byte layout and identity");
        for(const auto& [name,value]:saved) set(name.c_str(),value?value->c_str():nullptr);
    }
    {
        KnownRuntimeFixture f(model,true,true,8U,false,strix::KnownWindowOutputPolicy::terminal_row,false,true);
        strix::KnownPrefillRequest request;request.logical_rows=16;request.r2_shared_expert=true;request.r2_gdn=true;
        request.r2_gdn_f32_gates=true;request.qsa_rows=true;request.full_qsa_score_ring_entry=true;
        request.output_policy=strix::KnownWindowOutputPolicy::terminal_row;
        request.workspace_budget_bytes=UINT64_MAX;request.host_metadata_payload_budget_bytes=UINT64_MAX;
        const auto plan=strix::compile_known_prefix_prefill(model,f.source.serial,f.source.manifest,f.source.access,request);
        expect(plan.external_qsa_score_bytes!=0,"missing entry rejection fixture uses compact storage");
        f.window_pool.resize(static_cast<std::size_t>((plan.arena.physical.total_bytes+7U)/8U));
        f.window_memory={plan.prefill_execution_identity,reinterpret_cast<std::uintptr_t>(f.window_pool.data()),plan.arena.physical.total_bytes,11U};
        auto source=f.source.serial;source.execution_identity=plan.prefill_execution_identity;source.arena=plan.arena.row_view.back();
        const auto launch=strix::compile_launch_plan(source,f.registry);
        const std::vector<std::uint32_t> tokens(16U,7U);
        for(unsigned entry=0;entry<2;++entry) {
            const std::array<strix::KnownWindowRowsEntry,1> full{{
                {V::qsa_q8_rows8_r1_v1,&f.trace,&recording_rows_preflight,&recording_rows_launch}}};
            const auto before=f.state_words;
            bool binding_refusal=false;
            try {(void)strix::execute_known_prefix_prefill_eager(plan,f.source.serial,f.resources,f.window_memory,
                f.session,tokens,launch,{}, {},f.hooks(),entry?std::span<const strix::KnownWindowRowsEntry>(full):
                std::span<const strix::KnownWindowRowsEntry>{});}
            catch(const std::invalid_argument& e) {binding_refusal=std::string(e.what()).find("score ring binding")!=std::string::npos;}
            expect(binding_refusal,"compact runtime refuses absent or unbound full entry with a clear binding error");
            expect(f.state_words==before && f.session.next_position==0 && !f.trace.calls && !f.trace.admissions && !f.quiesces,
                "compact entry refusal precedes callbacks, state mutation and device work");
        }
    }
    constexpr std::uint64_t layer_bytes=80ULL*34ULL*(12288ULL+512ULL+512ULL+512ULL+128ULL)+192ULL*34ULL*2560ULL;
    for(const std::uint32_t rows:{2U,4U,8U}) {
        strix::KnownWindowRequest request{true,true,true,rows};
        const auto off=strix::compile_known_prefix_window(model,fixture.serial,fixture.manifest,fixture.access,request);
        request.qsa_rows=true;
        const auto on=strix::compile_known_prefix_window(model,fixture.serial,fixture.manifest,fixture.access,request);
        expect(off.qsa_rows_steps==0U && on.qsa_rows_steps==12U && on.row_loop_steps+12U==off.row_loop_steps &&
            on.logical_projection_weight_bytes_avoided==off.logical_projection_weight_bytes_avoided+12U*layer_bytes*(rows-1U) &&
            on.projection_launches_saved==off.projection_launches_saved+12U*(6U*rows-2U) &&
            on.window_execution_identity!=off.window_execution_identity &&
            on.window_kernel_plan_identity!=off.window_kernel_plan_identity,
            "QSA rows are explicitly selected with whole-window identities and static projection savings");
        const auto expected=rows==2U?V::qsa_q8_rows2_r1_v1:rows==4U?V::qsa_q8_rows4_r1_v1:V::qsa_q8_rows8_r1_v1;
        std::size_t first=0;
        for(std::size_t i=0;i<on.mode.size();++i) if(on.mode[i]==M::qsa_rows) {
            first=i;
            expect(on.r2_variant[i]==expected && fixture.serial.execution.steps[i].op==strix::SerialOp::indexed_attention &&
                on.row_plan.back().bindings.bindings[i].kernel_id==fixture.serial.bindings.bindings[i].kernel_id,
                "QSA rows retain every incumbent R1 source ID");
        }
        strix::validate_known_prefix_window_formats(on,fixture.serial,fixture.manifest,fixture.access);
        auto changed=on;++changed.qsa_rows_steps;
        reject([&]{strix::validate_compiled_known_prefix_window(changed,fixture.serial);},"forged QSA rows counter rejects");
        changed=on;changed.r2_variant[first]=V::gdn_q8_rows8_f32_gates_r1_v1;
        reject([&]{strix::validate_compiled_known_prefix_window(changed,fixture.serial);},"wrong QSA rows variant rejects");
        auto malformed=fixture;
        auto tensor=std::find_if(malformed.manifest.tensors.begin(),malformed.manifest.tensors.end(),
            [](const auto& t){return t.name=="blk.3.indexer.k_proj.weight";});
        expect(tensor!=malformed.manifest.tensors.end(),"QSA fixture supplies the indexer key projection");
        if(tensor==malformed.manifest.tensors.end()) continue;
        --tensor->byte_size;
        reject([&]{(void)strix::compile_known_prefix_window(model,malformed.serial,malformed.manifest,malformed.access,request);},
            "short QSA projection weight rejects before callbacks");
        ++tensor->byte_size;
        tensor->ggml_type=13U; // An unsupported indexer format keeps that step on its serial row loop.
        const auto fallback=strix::compile_known_prefix_window(model,malformed.serial,malformed.manifest,malformed.access,request);
        const auto layer3=std::find_if(fixture.serial.execution.steps.begin(),fixture.serial.execution.steps.end(),
            [](const auto& step){return step.op==strix::SerialOp::indexed_attention && step.layer==3U;})-fixture.serial.execution.steps.begin();
        expect(fallback.qsa_rows_steps==11U && fallback.mode[static_cast<std::size_t>(layer3)]==M::r1_row_loop,
            "unmatched QSA formats retain their row loop");
        reject([&]{strix::validate_known_prefix_window_formats(on,malformed.serial,malformed.manifest,malformed.access);},
            "an already compiled QSA window cannot inherit a changed projection format");

        KnownRuntimeFixture baseline(model,false,false,rows);
        std::vector<std::uint64_t> serial;
        for(const auto token:baseline.tokens) serial.push_back(baseline.serial_token(token));
        KnownRuntimeFixture wide(model,false,false,rows,false,strix::KnownWindowOutputPolicy::all_rows,false,true);
        const std::array<strix::KnownWindowRowsEntry,1> entries{{
            {expected,&wide.trace,&recording_rows_preflight,&recording_qsa_rows_launch}}};
        std::vector<const strix::CompiledLaunchPlan*> plans;
        for(const auto& plan:wide.row_launch) plans.push_back(&plan);
        const auto result=strix::execute_known_prefix_window_eager(wide.window,wide.source.serial,wide.resources,
            wide.window_memory,wide.session,wide.tokens,plans,{},entries,wide.hooks());
        expect(wide.state_words==baseline.state_words && wide.session.state.generations==baseline.session.state.generations &&
            wide.session.next_position==rows,"QSA rows keep every state word and one full-window frontier");
        for(std::uint32_t row=0;row<rows;++row)
            expect(read_word(result.logits[row].address)==serial[row],"QSA rows preserve every row's complete output");
        expect(result.qsa_rows_dispatches==12U && result.r2_variant_dispatches[static_cast<std::size_t>(expected)]==12U,
            "QSA rows dispatch once per QSA layer");
        KnownRuntimeFixture missing(model,false,false,rows,false,strix::KnownWindowOutputPolicy::all_rows,false,true);
        plans.clear();for(const auto& plan:missing.row_launch) plans.push_back(&plan);
        reject([&]{(void)strix::execute_known_prefix_window_eager(missing.window,missing.source.serial,missing.resources,
            missing.window_memory,missing.session,missing.tokens,plans,{},{},missing.hooks());},
            "missing QSA rows callback rejects before dispatch");
        expect(missing.trace.calls==0U && missing.session.next_position==0U,"rejected QSA window mutates nothing");
    }
}

void known_window_lm_head_rows_tests(const strix::ModelContract& model) {
    using V=strix::KnownWindowR2Variant;
    using M=strix::KnownWindowStepMode;
    using P=strix::KnownWindowOutputPolicy;
    const auto head_bytes=2560ULL/32ULL*34ULL*model.vocab_size;
    const auto reject=[](auto&& operation,const char* message) {
        bool rejected=false;try{operation();}catch(const std::invalid_argument&){rejected=true;}expect(rejected,message);
    };
    const auto fixture=make_known_fixture(model,0U,false,false,false,8U,8U);
    for(const std::uint32_t rows:{2U,4U,8U}) {
        const auto variant=rows==2U?V::lm_head_q8_prefetch8_rows2_v1:rows==4U?V::lm_head_q8_prefetch8_rows4_v1:V::lm_head_q8_prefetch8_rows8_v1;
        const auto off=strix::compile_known_prefix_window(model,fixture.serial,fixture.manifest,fixture.access,
            {true,true,true,rows,false,P::all_rows,false,false,false,false,false});
        const auto on=strix::compile_known_prefix_window(model,fixture.serial,fixture.manifest,fixture.access,
            {true,true,true,rows,false,P::all_rows,false,false,false,false,true});
        strix::validate_known_prefix_window_formats(on,fixture.serial,fixture.manifest,fixture.access);
        expect(off.lm_head_rows_steps==0U && off.mode.back()==M::r1_row_loop && on.lm_head_rows_steps==1U &&
            on.mode.back()==M::lm_head_rows && on.r2_variant.back()==variant && on.row_loop_steps+1U==off.row_loop_steps &&
            on.logical_projection_weight_bytes_avoided==off.logical_projection_weight_bytes_avoided+head_bytes*(rows-1U) &&
            on.projection_launches_saved==off.projection_launches_saved+rows-1U &&
            on.window_execution_identity!=off.window_execution_identity &&
            on.window_kernel_plan_identity!=off.window_kernel_plan_identity && on.logit_row_indices.size()==rows &&
            on.row_plan.back().bindings.bindings.back().kernel_id==fixture.serial.bindings.bindings.back().kernel_id,
            "multi-row head is an explicit terminal mode with distinct identities and its R1 source ID");
        for(const auto policy:{P::terminal_row,P::state_only}) {
            const auto kept=strix::compile_known_prefix_window(model,fixture.serial,fixture.manifest,fixture.access,
                {true,true,true,rows,false,policy,false,false,false,false,true});
            expect(kept.lm_head_rows_steps==0U && kept.mode.back()==M::r1_row_loop,
                "omitted-head policies keep the per-row terminal loop");
        }
        auto changed=on;++changed.lm_head_rows_steps;
        reject([&]{strix::validate_compiled_known_prefix_window(changed,fixture.serial);},"forged head rows counter rejects");
        changed=on;changed.r2_variant.back()=rows==2U?V::lm_head_q8_prefetch8_rows4_v1:V::lm_head_q8_prefetch8_rows2_v1;
        reject([&]{strix::validate_compiled_known_prefix_window(changed,fixture.serial);},"wrong-width head variant rejects");
        changed=on;changed.output_policy=P::terminal_row;changed.logit_row_indices={rows-1U};
        reject([&]{strix::validate_compiled_known_prefix_window(changed,fixture.serial);},"head rows cannot claim an omitted-head policy");
        auto bf16=make_known_fixture(model,0U,false,false,false,8U,30U);
        const auto fallback=strix::compile_known_prefix_window(model,bf16.serial,bf16.manifest,bf16.access,
            {true,true,true,rows,false,P::all_rows,false,false,false,false,true});
        expect(fallback.lm_head_rows_steps==0U,"non-Q8 head retains its R1 row loop");
        reject([&]{strix::validate_known_prefix_window_formats(on,bf16.serial,bf16.manifest,bf16.access);},
            "a compiled head rows window cannot inherit a changed head format");
        const auto plain=make_known_fixture(model,0U);
        expect(strix::compile_known_prefix_window(model,plain.serial,plain.manifest,plain.access,
            {true,true,true,rows,false,P::all_rows,false,false,false,false,true}).lm_head_rows_steps==0U,"other head kernel IDs stay R1");

        KnownRuntimeFixture baseline(model,false,true,rows,false,P::all_rows,false,false,true);
        std::vector<std::uint64_t> expected;
        for(const auto token:baseline.tokens) expected.push_back(baseline.serial_token(token));
        KnownRuntimeFixture candidate(model,false,true,rows,false,P::all_rows,false,false,true);
        const std::array<strix::KnownWindowRowsEntry,1> entries{{
            {variant,&candidate.trace,&recording_rows_preflight,&recording_rows_launch}}};
        std::vector<const strix::CompiledLaunchPlan*> plans;
        for(const auto& plan:candidate.row_launch) plans.push_back(&plan);
        const auto result=strix::execute_known_prefix_window_eager(candidate.window,candidate.source.serial,
            candidate.resources,candidate.window_memory,candidate.session,candidate.tokens,plans,{},entries,candidate.hooks());
        expect(result.logits.size()==rows && result.lm_head_rows_dispatches==1U &&
            result.r2_variant_dispatches[static_cast<std::size_t>(variant)]==1U &&
            candidate.trace.head_positions.size()==rows && candidate.session.next_position==rows,
            "terminal head runs once for all rows and still materialises every row logits");
        for(std::uint32_t row=0;row<rows;++row)
            expect(read_word(result.logits[row].address)==expected[row],"head rows keep every serial row output");
        KnownRuntimeFixture missing(model,false,true,rows,false,P::all_rows,false,false,true);
        plans.clear();for(const auto& plan:missing.row_launch) plans.push_back(&plan);
        reject([&]{(void)strix::execute_known_prefix_window_eager(missing.window,missing.source.serial,missing.resources,
            missing.window_memory,missing.session,missing.tokens,plans,{},{},missing.hooks());},
            "missing head rows entry rejects before any dispatch");
        expect(missing.trace.calls==0U && missing.session.next_position==0U,"rejected head rows window mutates nothing");
    }
}

// A8 profile: every census format (Q5_K gate/up, Q8_0 down) has an A8 union,
// so all 48 MoE layers promote; the float profile keeps those 5 as row loops.
void known_window_routed_a8_union_compiler_tests(const strix::ModelContract& model) {
    using V=strix::KnownWindowR2Variant;
    using M=strix::KnownWindowStepMode;
    for(const unsigned profile:{1U,2U}) {
        routed_fixture_profile=profile;
        const auto fixture=make_known_fixture(model,0U,true);
        routed_fixture_profile=0U;
        for(const std::uint32_t rows:{2U,4U,8U}) {
            const auto on=strix::compile_known_prefix_window(model,fixture.serial,fixture.manifest,fixture.access,
                {true,true,true,rows,true});
            const auto expected=profile==2U ? (rows==2U?V::routed_a8_union_rows2_v1:rows==4U?V::routed_a8_union_rows4_v1:V::routed_a8_union_rows8_v1) :
                (rows==2U?V::routed_q51_union_rows2_v1:rows==4U?V::routed_q51_union_rows4_v1:V::routed_q51_union_rows8_v1);
            expect(on.routed_expert_union_steps==(profile==2U?48U:43U),"A8 profile promotes all 48 MoE layers, float profile 43");
            for(std::size_t i=0;i<on.mode.size();++i) if(on.mode[i]==M::routed_expert_union)
                expect(on.r2_variant[i]==expected,"routed union variant follows the serial numerical profile");
            strix::validate_known_prefix_window_formats(on,fixture.serial,fixture.manifest,fixture.access);
            if(profile==2U) {
                auto forged=on;
                for(std::size_t i=0;i<forged.mode.size();++i) if(forged.mode[i]==M::routed_expert_union)
                    {forged.r2_variant[i]=rows==2U?V::routed_q51_union_rows2_v1:rows==4U?V::routed_q51_union_rows4_v1:V::routed_q51_union_rows8_v1;break;}
                bool rejected=false;
                try{strix::validate_known_prefix_window_formats(forged,fixture.serial,fixture.manifest,fixture.access);}
                catch(const std::invalid_argument&){rejected=true;}
                expect(rejected,"float union variant cannot stand in for the A8 profile");
            }
        }
    }
}

void known_window_routed_union_compiler_tests(const strix::ModelContract& model) {
    using V=strix::KnownWindowR2Variant;
    using M=strix::KnownWindowStepMode;
    const auto fixture=make_known_fixture(model,0U,true);
    for(const std::uint32_t rows:{2U,4U,8U}) {
        const auto off=strix::compile_known_prefix_window(model,fixture.serial,fixture.manifest,fixture.access,
            {true,true,true,rows,false});
        const auto on=strix::compile_known_prefix_window(model,fixture.serial,fixture.manifest,fixture.access,
            {true,true,true,rows,true});
        expect(off.routed_expert_union_steps==0U && on.routed_expert_union_steps==48U &&
            on.row_loop_steps+48U==off.row_loop_steps && on.window_execution_identity!=off.window_execution_identity &&
            on.window_kernel_plan_identity!=off.window_kernel_plan_identity,
            "expert union is explicitly selected with distinct whole-window identities");
        expect(strix::known_window_has_promoted_steps(off) && strix::known_window_has_promoted_steps(on),
            "CLI admission accepts wide complete operators even when there are no paired steps");
        expect(on.logical_projection_weight_bytes_avoided==off.logical_projection_weight_bytes_avoided &&
            on.projection_launches_saved==off.projection_launches_saved,
            "dynamic routed unions never manufacture static weight or launch savings");
        const auto expected=rows==2U?V::routed_q51_union_rows2_v1:
            rows==4U?V::routed_q51_union_rows4_v1:V::routed_q51_union_rows8_v1;
        std::size_t first=0;
        for(std::size_t i=0;i<on.mode.size();++i) if(on.mode[i]==M::routed_expert_union) {
            first=i;
            expect(on.r2_variant[i]==expected && on.row_plan.back().bindings.bindings[i].kernel_id==
                fixture.serial.bindings.bindings[i].kernel_id,"union callback retains every incumbent R1 source ID");
        }
        strix::validate_known_prefix_window_formats(on,fixture.serial,fixture.manifest,fixture.access);
        const auto reject=[&](auto&& operation,const char* message) {
            bool rejected=false;try{operation();}catch(const std::invalid_argument&){rejected=true;}expect(rejected,message);
        };
        auto short_down=fixture;
        auto down=std::find_if(short_down.manifest.tensors.begin(),short_down.manifest.tensors.end(),
            [](const auto& t){return t.name=="blk.0.ffn_down_exps.weight";});
        expect(down!=short_down.manifest.tensors.end() && down->ggml_type==7U && down->byte_size==629145600ULL,
            "Q5_1 fixture has 24-byte blocks for the actual 640x2560x512 tensor geometry");
        if(down!=short_down.manifest.tensors.end()) {
            down->byte_size=576716800ULL; // Q5_0's 22-byte blocks must never admit a Q5_1 tensor.
            reject([&]{(void)strix::compile_known_prefix_window(model,short_down.serial,short_down.manifest,
                short_down.access,{false,false,false,rows,true});},
                "Q5_1 union rejects the formerly accepted Q5_0-sized extent before dispatch");
        }
        auto changed=on;++changed.routed_expert_union_steps;
        reject([&]{strix::validate_compiled_known_prefix_window(changed,fixture.serial);},"forged union counter rejects");
        changed=on;changed.r2_variant[first]=V::shared_q8_virtual256_r2_v1;
        reject([&]{strix::validate_compiled_known_prefix_window(changed,fixture.serial);},"wrong union variant rejects");
        auto malformed=fixture;
        auto tensor=std::find_if(malformed.manifest.tensors.begin(),malformed.manifest.tensors.end(),
            [](const auto& t){return t.name=="blk.0.ffn_gate_exps.weight";});
        expect(tensor!=malformed.manifest.tensors.end(),"union fixture supplies actual routed tensor family");
        if(tensor==malformed.manifest.tensors.end()) continue;
        --tensor->byte_size;
        reject([&]{(void)strix::compile_known_prefix_window(model,malformed.serial,malformed.manifest,
            malformed.access,{false,false,false,rows,true});},"short routed union weight rejects before callbacks");
        tensor->ggml_type=13U; // Different gate format remains the real serial fallback.
        const auto fallback=strix::compile_known_prefix_window(model,malformed.serial,malformed.manifest,
            malformed.access,{false,false,false,rows,true});
        expect(fallback.routed_expert_union_steps==47U,"unmatched routed formats retain their original row loop");
        reject([&]{strix::validate_known_prefix_window_formats(on,malformed.serial,malformed.manifest,malformed.access);},
            "an already compiled union cannot inherit a changed gate format");
    }
}

void known_window_hyper_rows_tests(const strix::ModelContract& model) {
    using V=strix::KnownWindowR2Variant;
    using M=strix::KnownWindowStepMode;
    const auto reject=[](auto&& operation,const char* message) {
        bool rejected=false;try{operation();}catch(const std::invalid_argument&){rejected=true;}expect(rejected,message);
    };
    auto fixture=make_known_fixture(model,8U,false,true);
    // One unadmitted selection (deferred RMS) stays on its serial row loop.
    const auto deferred=std::find_if(fixture.serial.execution.steps.begin(),fixture.serial.execution.steps.end(),
        [](const auto& step){return step.op==strix::SerialOp::final_hyper_mix;})->id;
    fixture.serial.bindings.bindings[deferred].kernel_id=std::string(known_hyper_base)+
        ".splitk-hc-proj-fp32-v1.hc-deferred-norm-v1.lane-chain-v1";
    for(const std::uint32_t rows:{2U,4U,8U}) {
        const auto off=strix::compile_known_prefix_window(model,fixture.serial,fixture.manifest,fixture.access,
            {true,true,true,rows,false});
        const auto on=strix::compile_known_prefix_window(model,fixture.serial,fixture.manifest,fixture.access,
            {true,true,true,rows,false,strix::KnownWindowOutputPolicy::all_rows,true});
        expect(off.hyper_mix_rows_steps==0U && on.hyper_mix_rows_steps==96U && on.row_loop_steps+96U==off.row_loop_steps &&
            on.window_execution_identity!=off.window_execution_identity &&
            on.window_kernel_plan_identity!=off.window_kernel_plan_identity,
            "HC rows are explicitly selected with distinct whole-window identities");
        const auto variant=rows==2U?V::hyper_mix_rows2_v1:rows==4U?V::hyper_mix_rows4_v1:V::hyper_mix_rows8_v1;
        std::size_t first=0;std::uint64_t bytes=0;
        for(std::size_t i=0;i<on.mode.size();++i) {
            const auto& step=fixture.serial.execution.steps[i];
            if(on.mode[i]!=M::hyper_mix_rows) {
                expect(!hyper_op(step.op) || i==deferred,"every admitted HC mixer is promoted");
                continue;
            }
            if(!first) first=i;
            expect(hyper_op(step.op) && on.r2_variant[i]==variant,"HC rows variant follows the window row count");
            bytes+=(2ULL*3481600ULL+(step.op==strix::SerialOp::final_hyper_mix?0U:163840ULL))*(rows-1U);
        }
        expect(on.logical_projection_weight_bytes_avoided==off.logical_projection_weight_bytes_avoided+bytes &&
            on.projection_launches_saved==off.projection_launches_saved+2U*96U*(rows-1U),
            "HC rows count Q8 down/up and F32 injection once per window");
        strix::validate_known_prefix_window_formats(on,fixture.serial,fixture.manifest,fixture.access);
        auto changed=on;++changed.hyper_mix_rows_steps;
        reject([&]{strix::validate_compiled_known_prefix_window(changed,fixture.serial);},"forged HC rows counter rejects");
        changed=on;changed.r2_variant[first]=rows==2U?V::hyper_mix_rows4_v1:V::hyper_mix_rows2_v1;
        reject([&]{strix::validate_compiled_known_prefix_window(changed,fixture.serial);},"wrong HC rows variant rejects");
        changed=on;changed.mode[deferred]=M::hyper_mix_rows;changed.r2_variant[deferred]=variant;
        --changed.row_loop_steps;++changed.hyper_mix_rows_steps;
        reject([&]{strix::validate_compiled_known_prefix_window(changed,fixture.serial);},"unadmitted HC selection rejects");
    }
    for(const std::uint32_t rows:{2U,4U,8U}) {
        KnownRuntimeFixture baseline(model,false,false,rows,false,strix::KnownWindowOutputPolicy::all_rows,true);
        std::vector<std::uint64_t> expected;
        for(const auto token:baseline.tokens) expected.push_back(baseline.serial_token(token));
        KnownRuntimeFixture hc(model,false,false,rows,false,strix::KnownWindowOutputPolicy::all_rows,true);
        const auto variant=rows==2U?V::hyper_mix_rows2_v1:rows==4U?V::hyper_mix_rows4_v1:V::hyper_mix_rows8_v1;
        const std::array<strix::KnownWindowRowsEntry,1> entries{{{variant,&hc.trace,&recording_rows_preflight,&recording_rows_launch}}};
        std::vector<const strix::CompiledLaunchPlan*> plans;
        for(const auto& plan:hc.row_launch) plans.push_back(&plan);
        const auto result=strix::execute_known_prefix_window_eager(hc.window,hc.source.serial,hc.resources,
            hc.window_memory,hc.session,hc.tokens,plans,{},entries,hc.hooks());
        expect(hc.state_words==baseline.state_words && hc.session.state.generations==baseline.session.state.generations &&
            hc.session.next_position==rows,"HC rows keep every state and the full-window frontier");
        for(std::uint32_t row=0;row<rows;++row)
            expect(read_word(result.logits[row].address)==expected[row],"HC rows preserve the output of every row");
        expect(result.hyper_mix_rows_dispatches==97U && result.r2_variant_dispatches[static_cast<std::size_t>(variant)]==97U &&
            result.routed_expert_union_dispatches==0U && result.shared_rows_dispatches==0U,
            "HC rows dispatch once per mixer for the whole window");
        KnownRuntimeFixture missing(model,false,false,rows,false,strix::KnownWindowOutputPolicy::all_rows,true);
        plans.clear();for(const auto& plan:missing.row_launch) plans.push_back(&plan);
        reject([&]{(void)strix::execute_known_prefix_window_eager(missing.window,missing.source.serial,missing.resources,
            missing.window_memory,missing.session,missing.tokens,plans,{},{},missing.hooks());},
            "missing HC rows entry rejects before dispatch");
        expect(!missing.trace.calls && !missing.quiesces,"missing HC rows entry never dispatches");
    }
}

void known_window_router_rows_compiler_tests(const strix::ModelContract& model) {
    using V=strix::KnownWindowR2Variant;
    using M=strix::KnownWindowStepMode;
    auto fixture=make_known_fixture(model);
    for(std::size_t i=0;i<fixture.serial.execution.steps.size();++i) {
        const auto op=fixture.serial.execution.steps[i].op;
        auto& id=fixture.serial.bindings.bindings[i].kernel_id;
        if(op==strix::SerialOp::moe_route) id="hip.moe_route.bf16.h2560.e512.k10.r1.c1.staged_logits.wave.parallel";
        else if(op==strix::SerialOp::attn_inject || op==strix::SerialOp::mlp_inject) id="hip.inject.bf16.h2560.hc4.r1.c1";
        else if(op==strix::SerialOp::moe_combine) id="hip.moe_combine.bf16.h2560.hc4.r1.c1";
    }
    const auto reject=[&](auto&& operation,const char* message) {
        bool rejected=false;try{operation();}catch(const std::invalid_argument&){rejected=true;}expect(rejected,message);
    };
    for(const std::uint32_t rows:{2U,4U,8U}) {
        strix::KnownWindowRequest request{true,true,true,rows,false};
        const auto off=strix::compile_known_prefix_window(model,fixture.serial,fixture.manifest,fixture.access,request);
        request.router_rows=true;request.combine_inject_rows=true;
        const auto on=strix::compile_known_prefix_window(model,fixture.serial,fixture.manifest,fixture.access,request);
        expect(off.router_rows_steps==0U && off.combine_inject_rows_steps==0U && on.router_rows_steps==48U &&
            on.combine_inject_rows_steps==48U+48U+47U && on.row_loop_steps+48U+143U==off.row_loop_steps &&
            on.window_execution_identity!=off.window_execution_identity &&
            on.window_kernel_plan_identity!=off.window_kernel_plan_identity,
            "router/combine/inject rows are explicit and carry distinct whole-window identities");
        std::size_t route=0,inject=0;
        for(std::size_t i=0;i<on.mode.size();++i) {
            const auto& step=fixture.serial.execution.steps[i];
            if(step.op==strix::SerialOp::moe_route) {
                route=i;expect(on.mode[i]==M::moe_route_rows && on.r2_variant[i]==V::moe_route_rows_v1,"router rows variant");
            } else if(step.op==strix::SerialOp::moe_combine)
                expect(on.mode[i]==M::moe_combine_rows && on.r2_variant[i]==V::moe_combine_rows_v1,"combine rows variant");
            else if(step.op==strix::SerialOp::attn_inject || step.op==strix::SerialOp::mlp_inject) {
                const bool tap=step.op==strix::SerialOp::mlp_inject && *step.layer+1U==model.hidden_layers;
                expect(on.mode[i]==(tap?M::r1_row_loop:M::inject_rows) &&
                    on.r2_variant[i]==(tap?V::none:V::inject_rows_v1),"inject rows keep the MTP-observed last MLP inject");
                if(!tap) inject=i;
            }
        }
        strix::validate_known_prefix_window_formats(on,fixture.serial,fixture.manifest,fixture.access);
        auto changed=on;++changed.router_rows_steps;
        reject([&]{strix::validate_compiled_known_prefix_window(changed,fixture.serial);},"forged router rows counter rejects");
        changed=on;changed.r2_variant[route]=V::inject_rows_v1;
        reject([&]{strix::validate_compiled_known_prefix_window(changed,fixture.serial);},"wrong router rows variant rejects");
        changed=on;changed.mode[inject]=M::moe_combine_rows;changed.r2_variant[inject]=V::moe_combine_rows_v1;
        reject([&]{strix::validate_compiled_known_prefix_window(changed,fixture.serial);},"inject cannot pose as combine");
    }
    auto serial_router=fixture;
    for(std::size_t i=0;i<serial_router.serial.execution.steps.size();++i)
        if(serial_router.serial.execution.steps[i].op==strix::SerialOp::moe_route)
            serial_router.serial.bindings.bindings[i].kernel_id="hip.moe_route.bf16.h2560.e512.k10.r1.c1";
    strix::KnownWindowRequest request{true,true,true,4U,false};request.router_rows=true;
    expect(strix::compile_known_prefix_window(model,serial_router.serial,serial_router.manifest,serial_router.access,
        request).router_rows_steps==0U,"non-parallel router binding keeps its row loop");
}

void known_window_tests(const strix::ModelContract& model) {
    const auto fixture = make_known_fixture(model);
    const auto off = strix::compile_known_prefix_window(model, fixture.serial, fixture.manifest, fixture.access);
    expect(off.r2_steps == 0 && off.row_loop_steps == fixture.serial.execution.steps.size(),
        "known window default never promotes a serial kernel");
    expect(!strix::known_window_has_promoted_steps(off),"an all-row-loop plan has no promoted operators");
    expect(off.logical_projection_weight_bytes_avoided == 0 && off.projection_launches_saved == 0,
        "all-off row loops claim no logical projection savings");
    const auto on = strix::compile_known_prefix_window(model, fixture.serial, fixture.manifest, fixture.access, {true,true});
    expect(on.r2_steps == 84 && on.runtime_row_steps == 1 &&
        on.window_execution_identity != fixture.serial.execution_identity &&
        on.window_execution_identity != off.window_execution_identity,
        "known window closed table promotes 48 shared and 36 GDN steps with a distinct identity");
    expect(on.projection_launches_saved == 48U * 3U + 36U * 5U &&
        on.logical_projection_weight_bytes_avoided ==
            48ULL * (3ULL * (2560ULL / 32ULL * 34ULL * 640ULL)) +
            36ULL * (2560ULL / 32ULL * 34ULL * (10240ULL + 6144ULL + 48ULL + 48ULL) +
                6144ULL / 32ULL * 34ULL * 2560ULL),
        "logical savings derive only from the verified three/five selected Q8 projections");
    for (std::size_t row = 0; row < 2; ++row) {
        expect(on.row_plan[row].execution_identity == on.window_execution_identity &&
            on.row_plan[row].kernel_plan_identity == fixture.serial.kernel_plan_identity &&
            on.row_plan[row].logical_rows == 1 && !on.row_plan[row].whole_plan_graph_safe,
            "known window row clones retain serial kernel contract with window memory identity");
        for (std::size_t i = 0; i < fixture.serial.bindings.bindings.size(); ++i)
            expect(on.row_plan[row].bindings.bindings[i].kernel_id == fixture.serial.bindings.bindings[i].kernel_id,
                "known window never retags a serial KernelVariant ID");
    }
    for (std::size_t i = 0; i < on.arena.physical.allocations.size(); ++i) {
        const auto* original = strix::find_arena_allocation(fixture.serial.arena, on.arena.physical.allocations[i].use);
        expect(original && on.arena.row_view[0].allocations[i].bytes == original->bytes &&
            on.arena.row_view[1].allocations[i].offset - on.arena.row_view[0].allocations[i].offset == on.arena.row_stride[i],
            "R2 derives row strides from actual R1 payloads including the A8/Q51 workspace");
    }
    const auto reject = [&](auto changed, const char* message) {
        bool caught = false;
        try { strix::validate_compiled_known_prefix_window(changed, fixture.serial); }
        catch (const std::invalid_argument&) { caught = true; }
        expect(caught, message);
    };
    auto changed = on; changed.arena.row_stride[0] += 256;
    reject(changed, "known window rejects a mutated public stride");
    changed = on; changed.arena.row_view[1].allocations[0].offset = changed.arena.row_view[0].allocations[0].offset;
    reject(changed, "known window rejects row1/row0 physical alias");
    changed = on; changed.row_plan[1].bindings.bindings.back().kernel_id = "mutated.head";
    reject(changed, "known window rejects a mutated row kernel ID");
    changed = on; ++changed.row_plan[1].io.hidden_slots;
    reject(changed, "known window rejects mutated row IO metadata");
    changed = on; ++changed.r2_steps;
    reject(changed, "known window rejects fabricated promotion counters");
    changed = on; ++changed.logical_projection_weight_bytes_avoided;
    reject(changed, "known window rejects fabricated logical weight savings");
    changed = on; ++changed.projection_launches_saved;
    reject(changed, "known window rejects fabricated projection launch savings");
    changed = on; changed.mode[0] = strix::KnownWindowStepMode::r2_shared_q8;
    reject(changed, "known window rejects an R2 mode on a runtime step");
    auto unsupported = fixture;
    for (auto& binding : unsupported.serial.bindings.bindings)
        if (binding.op == strix::SerialOp::gated_delta_net) *binding.kernel_id += ".unknown";
    const auto fallback = strix::compile_known_prefix_window(model, unsupported.serial, unsupported.manifest, unsupported.access, {true,true});
    expect(fallback.r2_steps == 48, "unknown GDN serial IDs remain explicit R1 row loops");
    unsupported = fixture; unsupported.manifest.tensors[0].ggml_type = 0;
    const auto typed_fallback = strix::compile_known_prefix_window(model, unsupported.serial, unsupported.manifest, unsupported.access, {true,true});
    expect(typed_fallback.r2_steps == 83, "a non-Q8 matrix cannot promote the corresponding R2 step");
    unsupported = fixture; unsupported.manifest.tensors.back().byte_size -= 1;
    bool caught = false;
    try { (void)strix::compile_known_prefix_window(model, unsupported.serial, unsupported.manifest, unsupported.access, {true,true}); }
    catch (const std::invalid_argument&) { caught = true; }
    expect(caught, "a late malformed Q8 tensor rejects before runtime dispatch");
    unsupported = fixture; unsupported.serial.device_graph_inputs = true; caught = false;
    try { (void)strix::compile_known_prefix_window(model, unsupported.serial, unsupported.manifest, unsupported.access); }
    catch (const std::invalid_argument&) { caught = true; }
    expect(caught, "known window rejects a graph source");
    strix::ArithmeticLineage lineage;
    lineage = strix::prepare_prefill_chunk_lineage(lineage, 0, 2, on.window_execution_identity, on.window_kernel_plan_identity, on.window_execution_identity);
    lineage = strix::prepare_prefill_chunk_lineage(lineage, 2, 4, on.window_execution_identity, on.window_kernel_plan_identity, on.window_execution_identity);
    expect(lineage.intervals.size() == 2 && strix::arithmetic_lineage_valid(lineage,4) &&
        !strix::arithmetic_lineage_serial_only(lineage, fixture.serial.execution_identity,4),
        "prefill chunks retain boundaries and cannot forge a serial producer history");
    lineage = strix::prepare_serial_row_lineage(lineage,4,fixture.serial.execution_identity,fixture.serial.kernel_plan_identity);
    expect(lineage.intervals.size() == 3 && strix::arithmetic_lineage_valid(lineage,5),
        "a serial continuation may append to a complete known-prefix chunk history");
    expect(strix::prepare_prefill_chunk_lineage({}, 7,9,1,2,3).intervals.empty(),
        "prefill lineage never invents an unknown prior history");
    lineage.tainted = true; caught = false;
    try { (void)strix::prepare_prefill_chunk_lineage(lineage,5,7,1,2,3); }
    catch (const std::invalid_argument&) { caught = true; }
    expect(caught, "tainted lineage cannot become a clean chunk producer");
}


// Format/dispatch source witnesses only. Actual historical golden IDs must be
// captured from the frozen pre-change build during the final validation phase.
void known_window_wide_compiler_tests(const strix::ModelContract& model) {
    using V=strix::KnownWindowR2Variant;
    const auto source=make_known_fixture(model,0U);
    const auto base=strix::compile_known_prefix_window(model,source.serial,source.manifest,source.access,{true,true,true});
    const auto explicit_two=strix::compile_known_prefix_window(model,source.serial,source.manifest,source.access,{true,true,true,2});
    expect(base.window_execution_identity==explicit_two.window_execution_identity &&
        base.window_kernel_plan_identity==explicit_two.window_kernel_plan_identity,
        "default R2 uses the unchanged identity domain and physical geometry");
    std::set<std::uint64_t> identities{base.window_execution_identity};
    for(const std::uint32_t rows:{4U,8U}) {
        const auto wide=strix::compile_known_prefix_window(model,source.serial,source.manifest,source.access,{true,true,true,rows});
        strix::validate_known_prefix_window_formats(wide,source.serial,source.manifest,source.access);
        expect(wide.logical_rows==rows && wide.row_plan.size()==rows && !wide.r2_steps &&
            wide.shared_rows_steps==48U && wide.gdn_rows_steps==36U &&
            wide.logical_projection_weight_bytes_avoided==2456616960ULL*(rows-1U) &&
            wide.projection_launches_saved==252U*(rows-1U),
            "wide compiler counts actual wide projections and logical reuse, not fictitious R2 tiles");
        expect(identities.insert(wide.window_execution_identity).second,
            "R2/R4/R8 windows retain distinct complete execution identities");
        for(const auto variant:wide.r2_variant)
            expect(variant==V::none || variant==(rows==4U?V::shared_q8_virtual256_r4_v1:V::shared_q8_virtual256_r8_v1) ||
                variant==(rows==4U?V::gdn_q8_rows4_f32_gates_r1_v1:V::gdn_q8_rows8_f32_gates_r1_v1),
                "mixed wide table never silently aliases a two-row numerical variant");
        for(std::uint32_t row=0;row<rows;++row) {
            expect(wide.row_plan[row].kernel_plan_identity==source.serial.kernel_plan_identity &&
                wide.row_plan[row].logical_rows==1U,"every wide row remains an incumbent R1 plan");
            for(std::size_t i=0;i<wide.arena.physical.allocations.size();++i) {
                const auto& physical=wide.arena.physical.allocations[i];
                const auto* original=strix::find_arena_allocation(source.serial.arena,physical.use);
                expect(original && wide.arena.row_view[row].allocations[i].bytes==original->bytes &&
                    wide.arena.row_view[row].allocations[i].offset==physical.offset+row*wide.arena.row_stride[i],
                    "wide row payloads are bounded stripes of their actual serial allocations");
            }
        }
        for(const unsigned fault:{0U,1U,2U}) {
            auto changed=wide;
            if(fault==0U) ++changed.gdn_rows_steps;
            if(fault==1U) changed.r2_variant[static_cast<std::size_t>(std::find(changed.mode.begin(),changed.mode.end(),
                strix::KnownWindowStepMode::gdn_rows)-changed.mode.begin())]=V::gdn_q8_rows2_f32_gates_r1_v1;
            if(fault==2U) changed.row_plan.pop_back();
            bool rejected=false;try {strix::validate_compiled_known_prefix_window(changed,source.serial);}
            catch(const std::invalid_argument&) {rejected=true;}
            expect(rejected,"wide table rejects forged counts, wrong-width variants and missing row plans");
        }
    }
    for(const std::uint32_t rows:{0U,1U,3U,16U}) {
        bool rejected=false;
        try {(void)strix::compile_known_prefix_window(model,source.serial,source.manifest,source.access,{true,true,true,rows});}
        catch(const std::invalid_argument&) {rejected=true;}
        expect(rejected,"known windows admit only the closed R2/R4/R8 row table");
    }
}

void known_window_mixed_format_tests(const strix::ModelContract& model) {
    using Variant = strix::KnownWindowR2Variant;
    const auto q8 = make_known_fixture(model);
    const auto old_request = strix::compile_known_prefix_window(model, q8.serial, q8.manifest, q8.access, {true,true});
    const auto extra_request = strix::compile_known_prefix_window(model, q8.serial, q8.manifest, q8.access, {true,true,true});
    expect(old_request.window_execution_identity == extra_request.window_execution_identity &&
        old_request.window_kernel_plan_identity == extra_request.window_kernel_plan_identity &&
        old_request.r2_variant == extra_request.r2_variant && old_request.projection_launches_saved == 324U &&
        old_request.logical_projection_weight_bytes_avoided == 2466017280ULL &&
        std::count(old_request.r2_variant.begin(), old_request.r2_variant.end(), Variant::gdn_q8_virtual256_r2_v1) == 36 &&
        std::count(old_request.r2_variant.begin(), old_request.r2_variant.end(), Variant::shared_q8_virtual256_r2_v1) == 48,
        "unused F32-gate opt-in leaves all-Q8 identities, variant table and ledgers unchanged");
    const auto mixed = make_known_fixture(model, 0U);
    const auto fallback = strix::compile_known_prefix_window(model, mixed.serial, mixed.manifest, mixed.access, {true,true});
    const auto shared_only = strix::compile_known_prefix_window(model, mixed.serial, mixed.manifest, mixed.access, {true,false});
    expect(fallback.r2_steps == 48U && fallback.projection_launches_saved == 144U &&
        fallback.logical_projection_weight_bytes_avoided == 250675200ULL &&
        fallback.window_execution_identity == shared_only.window_execution_identity &&
        fallback.window_kernel_plan_identity == shared_only.window_kernel_plan_identity,
        "old two-boolean request keeps all mixed GDN steps as explicit R1 fallback");
    const auto inert_flag = strix::compile_known_prefix_window(model, mixed.serial, mixed.manifest, mixed.access, {true,false,true});
    expect(inert_flag.window_execution_identity == shared_only.window_execution_identity &&
        inert_flag.r2_variant == shared_only.r2_variant,
        "the F32-gate flag cannot enable GDN promotion without the GDN opt-in");
    const auto promoted = strix::compile_known_prefix_window(model, mixed.serial, mixed.manifest, mixed.access, {true,true,true});
    std::array<std::uint32_t, strix::known_window_r2_variant_slots> counts{};
    for (const auto variant : promoted.r2_variant) ++counts[static_cast<std::size_t>(variant)];
    expect(promoted.r2_steps == 84U && counts[1] == 48U && counts[2] == 0U && counts[3] == 36U &&
        promoted.projection_launches_saved == 252U &&
        promoted.logical_projection_weight_bytes_avoided == 2456616960ULL &&
        promoted.window_kernel_plan_identity != old_request.window_kernel_plan_identity &&
        promoted.window_execution_identity != fallback.window_execution_identity,
        "mixed promotion counts only three reused Q8 projections and identifies variant 3 separately");
    strix::validate_known_prefix_window_formats(promoted, mixed.serial, mixed.manifest, mixed.access);
    const auto tensor = [](auto& source, std::string_view name) -> strix::ManifestTensor& {
        const auto found = std::find_if(source.manifest.tensors.begin(), source.manifest.tensors.end(),
            [&](const auto& value) { return value.name == name; });
        if (found == source.manifest.tensors.end()) throw std::logic_error("mixed format fixture tensor missing");
        return *found;
    };
    const auto reject = [&](auto&& action, const char* message) {
        bool caught = false;
        try { action(); } catch (const std::invalid_argument&) { caught = true; }
        expect(caught, message);
    };
    for (const auto name : {"blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight"}) {
        auto asymmetric = mixed;
        auto& gate = tensor(asymmetric, name);
        gate.ggml_type = 8U; gate.byte_size = 2560ULL / 32U * 34U * 48U;
        const auto result = strix::compile_known_prefix_window(model, asymmetric.serial,
            asymmetric.manifest, asymmetric.access, {true,true,true});
        expect(result.r2_steps == 83U, "asymmetric F32/Q8 gates fall back only their own GDN layer");
    }
    for (const auto type : {1U, 30U}) {
        auto unsupported = mixed;
        tensor(unsupported, "blk.0.ssm_beta.weight").ggml_type = type;
        const auto result = strix::compile_known_prefix_window(model, unsupported.serial,
            unsupported.manifest, unsupported.access, {true,true,true});
        expect(result.r2_steps == 83U, "F16/BF16 gates do not enter the closed F32 gate variant");
    }
    {
        auto unsupported = mixed;
        auto& qkv = tensor(unsupported, "blk.0.attn_qkv.weight");
        qkv.ggml_type = 0U; qkv.byte_size = 2560ULL * 10240U * 4U;
        expect(strix::compile_known_prefix_window(model, unsupported.serial, unsupported.manifest,
            unsupported.access, {true,true,true}).r2_steps == 83U, "F32 large projection remains explicit R1 fallback");
    }
    for (unsigned fault = 0; fault < 4U; ++fault) {
        auto malformed = mixed;
        auto& gate = tensor(malformed, "blk.0.ssm_beta.weight");
        if (fault == 0U) --gate.byte_size;
        if (fault == 1U) ++gate.absolute_offset;
        if (fault == 2U) gate.absolute_offset = std::numeric_limits<std::uint64_t>::max() - 15U;
        if (fault == 3U) gate.dimensions[1] = 47U;
        reject([&] { (void)strix::compile_known_prefix_window(model, malformed.serial,
            malformed.manifest, malformed.access, {true,true,true}); }, "recognised malformed F32 gate rejects before runtime");
        if (fault < 3U)
            expect(strix::compile_known_prefix_window(model, malformed.serial, malformed.manifest,
                malformed.access, {true,true}).r2_steps == 48U,
                "disabled F32 format preserves old unsupported-type extent fallback");
    }
    for (unsigned fault = 0; fault < 4U; ++fault) {
        auto malformed = mixed;
        auto& gate = tensor(malformed, "blk.0.ssm_beta.weight");
        const auto index = static_cast<std::uint32_t>(&gate - malformed.manifest.tensors.data());
        if (fault == 0U) gate.name += ".missing";
        if (fault == 1U) malformed.manifest.tensors.push_back(gate);
        if (fault == 2U) {
            const auto selected = std::find_if(malformed.access.accesses.begin(), malformed.access.accesses.end(),
                [&](const auto& access) { return std::find(access.tensor_indices.begin(), access.tensor_indices.end(), index) != access.tensor_indices.end(); });
            if (selected == malformed.access.accesses.end()) throw std::logic_error("mixed selected gate missing");
            selected->tensor_indices.push_back(index);
        }
        if (fault == 3U) gate.layer = 1U;
        reject([&] { (void)strix::compile_known_prefix_window(model, malformed.serial,
            malformed.manifest, malformed.access, {true,true,true}); },
            "missing/duplicate/duplicate-selected/wrong-layer F32 gate cannot bypass exact tensor selection");
    }
    // This is a consistent, compiler-produced variant-3 table/identity/counter
    // combination supplied with all-Q8 metadata: structural validation alone
    // cannot establish the format, but the mandatory manifest check must.
    strix::validate_compiled_known_prefix_window(promoted, q8.serial);
    reject([&] { strix::validate_known_prefix_window_formats(promoted, q8.serial, q8.manifest, q8.access); },
        "consistent variant-3 forgery over all-Q8 metadata fails manifest revalidation");
    reject([&] { strix::validate_known_prefix_window_formats(old_request, mixed.serial, mixed.manifest, mixed.access); },
        "variant-2 table cannot admit actual F32 gates");
    auto changed = promoted;
    const auto qsa = std::find_if(mixed.serial.execution.steps.begin(), mixed.serial.execution.steps.end(),
        [](const auto& step) { return step.op == strix::SerialOp::indexed_attention; });
    if (qsa == mixed.serial.execution.steps.end()) throw std::logic_error("mixed fixture lacks QSA");
    changed.mode[qsa->id] = strix::KnownWindowStepMode::r2_gdn_staged;
    changed.r2_variant[qsa->id] = Variant::gdn_q8_rows2_f32_gates_r1_v1;
    reject([&] { strix::validate_compiled_known_prefix_window(changed, mixed.serial); }, "variant 3 cannot be forged onto QSA");
    changed = promoted; ++changed.projection_launches_saved;
    reject([&] { strix::validate_compiled_known_prefix_window(changed, mixed.serial); }, "mixed launch counter forgery rejects");
    changed = promoted; ++changed.logical_projection_weight_bytes_avoided;
    reject([&] { strix::validate_compiled_known_prefix_window(changed, mixed.serial); }, "mixed payload counter forgery rejects");

    KnownRuntimeFixture runtime(model, true, true);
    const std::array<strix::KnownWindowR2Entry, 3> entries{{
        {Variant::shared_q8_virtual256_r2_v1, &runtime.trace, &recording_r2_preflight, &recording_r2_launch},
        {Variant::gdn_q8_virtual256_r2_v1, &runtime.trace, &recording_r2_preflight, &recording_r2_launch},
        {Variant::gdn_q8_rows2_f32_gates_r1_v1, &runtime.trace, &recording_r2_preflight, &recording_r2_launch}}};
    const auto result = runtime.run_entries(runtime.hooks(), entries);
    expect(result.r2_variant_dispatches == std::array<std::uint32_t,strix::known_window_r2_variant_slots>{0,48,0,36} &&
        result.r2_dispatches == 84U && result.row_loop_dispatches == 708U &&
        result.projection_launches_saved == 252U && result.logical_projection_weight_bytes_avoided == 2456616960ULL,
        "runtime reports actual closed variant-3 dispatches independently of logical launches");
    for (const bool duplicate : {false, true}) {
        runtime.reset();
        auto bad_entries = std::vector<strix::KnownWindowR2Entry>(entries.begin(), entries.end());
        if (duplicate) bad_entries.push_back(entries.back()); else bad_entries.pop_back();
        std::uint32_t scopes = 0;
        auto hooks = runtime.hooks(); hooks.preflight_scope = [&] { ++scopes; };
        const auto state = runtime.state_words;
        reject([&] { (void)runtime.run_entries(hooks, bad_entries); },
            "missing/duplicate variant-3 implementation fails closed");
        expect(runtime.trace.calls == 0U && runtime.trace.admissions == 0U && scopes == 0U &&
            runtime.state_words == state && runtime.quiesces == 0U &&
            runtime.session.health == strix::SerialSessionHealth::ready && runtime.session.committed_tokens == 0U,
            "entry-table failure occurs before any admission, dispatch, state write or poison scope");
    }
}

}  // namespace

int main() {
    // The final-block cone keeps 128 trailing rows by default; the fixtures here are smaller, so cone to one tile, and cone
    // every trailing stateless step (the fixture has no step observer).
#ifdef _WIN32
    _putenv_s("STRIX_PREFILL_FINAL_CONE_ROWS", "8");
    _putenv_s("STRIX_PREFILL_FINAL_CONE", "2");
#else
    setenv("STRIX_PREFILL_FINAL_CONE_ROWS", "8", 1);
    setenv("STRIX_PREFILL_FINAL_CONE", "2", 1);
#endif
    const auto model =
        strix::ModelContract::qwen38_flash_next();

    strix::SpeculativeVerifyBindingRequest binding;
    binding.request_concurrency = 1;

    strix::ExecutionBufferRequest buffers;
    buffers.logical_rows = 4;

    const auto registry =
        make_registry(model, 4);
    const auto compiled =
        strix::compile_speculative_verify_execution(
            model,
            4,
            registry,
            binding,
            buffers,
            256);

    expect(compiled.candidate_depth == 4 &&
               compiled.bindings.missing_steps == 0 &&
               compiled.execution_identity != 0 &&
               compiled.kernel_plan_identity != 0,
           "compiled verify freezes depth, bindings and identities");
    // Captured from the pinned, pre-router implementation with this fixture.
    expect(compiled.execution_identity == 2522750290423016878ULL &&
               compiled.kernel_plan_identity == 2260442504783589592ULL &&
               compiled.arena.total_bytes == 3993600ULL,
           "default speculative identities and arena retain the pre-router receipt");
    expect(compiled.arena.total_bytes > 0 &&
               compiled.sequential_window_steps == 37 &&
               compiled.causal_window_steps == 12,
           "compiled verify includes D-row arena and stateful window counts");
    expect(!compiled.whole_plan_graph_safe,
           "whole verify plan remains non-graph-safe while dynamic/runtime steps exist");

    const auto compiled_again =
        strix::compile_speculative_verify_execution(
            model,
            4,
            registry,
            binding,
            buffers,
            256);
    expect(compiled_again.execution_identity ==
               compiled.execution_identity,
           "verify execution identity is reproducible");

    const auto different_registry =
        make_registry(
            model,
            4,
            ".alternate");
    const auto different =
        strix::compile_speculative_verify_execution(
            model,
            4,
            different_registry,
            binding,
            buffers,
            256);
    expect(different.execution_identity !=
               compiled.execution_identity,
           "verify kernel identity change invalidates execution identity");

    auto wrong_rows = buffers;
    wrong_rows.logical_rows = 2;
    bool rows_rejected = false;
    try {
        static_cast<void>(
            strix::compile_speculative_verify_execution(
                model,
                4,
                registry,
                binding,
                wrong_rows,
                256));
    } catch (const std::invalid_argument&) {
        rows_rejected = true;
    }
    expect(rows_rejected,
           "verify compile rejects buffer rows that differ from depth");

    auto wrong_concurrency = binding;
    wrong_concurrency.request_concurrency = 2;
    bool concurrency_rejected = false;
    try {
        static_cast<void>(
            strix::compile_speculative_verify_execution(
                model,
                4,
                registry,
                wrong_concurrency,
                buffers,
                256));
    } catch (const std::invalid_argument&) {
        concurrency_rejected = true;
    }
    expect(concurrency_rejected,
           "initial verify compile keeps request concurrency at one");

    auto router_binding = binding;
    auto router_buffers = buffers;
    router_binding.parallel_moe_router = true;
    router_buffers.parallel_moe_router = true;
    const auto router_registry = make_registry(model, 4, "", true);
    const auto router_compiled = strix::compile_speculative_verify_execution(
        model, 4, router_registry, router_binding, router_buffers);
    expect(router_compiled.execution_identity != compiled.execution_identity &&
               router_compiled.kernel_plan_identity != compiled.kernel_plan_identity &&
               router_compiled.io.router_workspace_slots == 1,
           "parallel router has admitted workspace and distinct speculative identities");
    bool router_mismatch = false;
    try { static_cast<void>(strix::compile_speculative_verify_execution(model, 4,
                                      router_registry, router_binding, buffers)); }
    catch (const std::invalid_argument&) { router_mismatch = true; }
    expect(router_mismatch, "verify compilation rejects router request/admission mismatch");
    router_mismatch = false;
    try { static_cast<void>(strix::compile_speculative_verify_execution(model, 4,
                                      registry, router_binding, router_buffers)); }
    catch (const std::invalid_argument&) { router_mismatch = true; }
    expect(router_mismatch, "verify compilation rejects legacy router under parallel admission");
    router_mismatch = false;
    try { static_cast<void>(strix::compile_speculative_verify_execution(model, 4,
                                      router_registry, binding, buffers)); }
    catch (const std::invalid_argument&) { router_mismatch = true; }
    expect(router_mismatch, "verify compilation rejects parallel router under default admission");
    const auto staged_registry = make_registry(model,4,".staged_logits",true);
    bool staged_rejected = false;
    try { (void)strix::compile_speculative_verify_execution(model,4,staged_registry,router_binding,router_buffers); }
    catch (const std::invalid_argument& error) {
        staged_rejected = std::string(error.what()) == "staged router logits does not support speculative verify";
    }
    expect(staged_rejected,"speculative verify rejects a selected staged router even with parallel workspace");
    auto ranked_binding = router_binding; ranked_binding.ranked_moe_router_topk = true;
    const auto ranked_registry = make_registry(model, 4, "", true, true);
    const auto ranked_compiled = strix::compile_speculative_verify_execution(model, 4, ranked_registry, ranked_binding, router_buffers);
    expect(ranked_compiled.execution_identity != router_compiled.execution_identity &&
               ranked_compiled.kernel_plan_identity != router_compiled.kernel_plan_identity &&
               ranked_compiled.arena.total_bytes == router_compiled.arena.total_bytes,
           "ranked speculative binding changes identities with unchanged arena");
    bool rejected_ranked = false;
    try { (void)strix::compile_speculative_verify_execution(model, 4, router_registry, ranked_binding, router_buffers); }
    catch (const std::invalid_argument&) { rejected_ranked = true; }
    expect(rejected_ranked, "ranked speculative admission rejects parallel-only kernel");
    rejected_ranked = false;
    try { (void)strix::compile_speculative_verify_execution(model, 4, ranked_registry, router_binding, router_buffers); }
    catch (const std::invalid_argument&) { rejected_ranked = true; }
    expect(rejected_ranked, "parallel speculative admission rejects ranked kernel");
    auto wave_binding = router_binding; wave_binding.wave_moe_router_topk = true;
    const auto wave_registry = make_registry(model, 4, "", true, false, true);
    const auto wave_compiled = strix::compile_speculative_verify_execution(model, 4, wave_registry, wave_binding, router_buffers);
    expect(wave_compiled.execution_identity != ranked_compiled.execution_identity &&
               wave_compiled.kernel_plan_identity != ranked_compiled.kernel_plan_identity &&
               wave_compiled.arena.total_bytes == router_compiled.arena.total_bytes,
           "wave speculative identities change without changing arena");
    bool rejected_wave = false;
    try { (void)strix::compile_speculative_verify_execution(model, 4, router_registry, wave_binding, router_buffers); }
    catch (const std::invalid_argument&) { rejected_wave = true; }
    expect(rejected_wave, "wave speculative admission rejects parallel-only binding");
    rejected_wave = false;
    try { (void)strix::compile_speculative_verify_execution(model, 4, wave_registry, router_binding, router_buffers); }
    catch (const std::invalid_argument&) { rejected_wave = true; }
    expect(rejected_wave, "parallel speculative admission rejects wave binding");
    auto invalid_wave = wave_binding; invalid_wave.ranked_moe_router_topk = true;
    rejected_wave = false;
    try { (void)strix::compile_speculative_verify_execution(model, 4, wave_registry, invalid_wave, router_buffers); }
    catch (const std::invalid_argument&) { rejected_wave = true; }
    expect(rejected_wave, "wave and ranked speculative modes mutually exclude");
    invalid_wave = wave_binding; invalid_wave.parallel_moe_router = false;
    rejected_wave = false;
    try { (void)strix::compile_speculative_verify_execution(model, 4, wave_registry, invalid_wave, router_buffers); }
    catch (const std::invalid_argument&) { rejected_wave = true; }
    expect(rejected_wave, "wave speculative mode requires parallel topology");
    auto invalid_ranked = binding; invalid_ranked.ranked_moe_router_topk = true;
    rejected_ranked = false;
    try { (void)strix::compile_speculative_verify_execution(model, 4, registry, invalid_ranked, buffers); }
    catch (const std::invalid_argument&) { rejected_ranked = true; }
    expect(rejected_ranked, "ranked speculative mode requires parallel topology");
    router_binding.parallel_moe_router = false;
    router_buffers.parallel_moe_router = false;
    const auto explicit_default = strix::compile_speculative_verify_execution(model, 4,
                                      registry, router_binding, router_buffers);
    expect(explicit_default.execution_identity == compiled.execution_identity &&
               explicit_default.kernel_plan_identity == compiled.kernel_plan_identity &&
               explicit_default.arena.total_bytes == compiled.arena.total_bytes,
           "explicit disabled router preserves speculative default identities and arena");

    auto q4_buffers=buffers; q4_buffers.logical_rows=1; q4_buffers.q4k_a8g32_gate_up=true;
    bool q4_rejected=false;
    try { (void)strix::compile_speculative_verify_execution(model,1,make_registry(model,1),binding,q4_buffers); }
    catch(const std::invalid_argument& error) {
        q4_rejected=std::string(error.what()).find("Q4K A8 runtime does not support speculative verify") != std::string::npos;
    }
    expect(q4_rejected,"Q4K A8 explicitly rejects speculative depth1 before alternate admission");
    q4_buffers.q4k_a8g32_gate_up=false; q4_rejected=false;
    try { (void)strix::compile_speculative_verify_execution(model,1,
        make_registry(model,1,".q4k-a8g32-gateup-v2"),binding,q4_buffers); }
    catch(const std::invalid_argument& error) {
        q4_rejected=std::string(error.what()).find("Q4K A8 runtime does not support speculative verify") != std::string::npos;
    }
    expect(q4_rejected,"speculative depth1 rejects quantized kernel even without mode admission");


    auto hc_binding=binding;hc_binding.lane_chain_hyper_output=true;bool hc_rejected=false;
    try{(void)strix::compile_speculative_verify_execution(model,4,registry,hc_binding,buffers);}catch(const std::invalid_argument&){hc_rejected=true;}
    expect(hc_rejected,"lane-chain HC rejects compiled speculative execution");
    auto deferred=binding;deferred.deferred_moe_status=true;
    bool deferred_rejected=false;
    try{(void)strix::compile_speculative_verify_execution(model,4,registry,deferred,buffers);}
    catch(const std::invalid_argument&){deferred_rejected=true;}
    expect(deferred_rejected,"deferred MoE explicitly rejects speculative verification");

    yarn_serial_window_selection_tests(model);
    known_window_routed_union_compiler_tests(model);
    known_window_routed_a8_union_compiler_tests(model);
    known_window_router_rows_compiler_tests(model);
    known_window_lm_head_rows_tests(model);
    known_window_routed_union_runtime_tests(model);
    known_window_hyper_rows_tests(model);
    known_window_qsa_tests(model);
    known_window_trellis_profile_tests(model);
    known_window_qsa_bf16_indexer_tests(model);
    known_window_tests(model);
    known_window_run_cache_tests(model);
    known_window_runtime_tests(model);
    actual_known_window_lineage_tests(model);
    known_window_output_policy_tests(model);
    known_window_wide_runtime_tests(model);
    known_window_mixed_format_tests(model);
    known_window_wide_compiler_tests(model);
    known_prefix_prefill_tests(model);
    known_prefix_prefill_segment_tests(model);

    if (failures == 0) {
        std::cout << "compiled speculative verify tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
