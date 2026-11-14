#include "strix/hip_check.hpp"
#include "strix/hip_serial_graph.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_qwen_rotary_table.hpp"
#include "strix/runtime_launchers.hpp"
#include "strix/dispatch_resources.hpp"
#include <hip/hip_runtime.h>
#include <array>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace strix {
namespace {
using strix::hip_check;
void graph_position_prepare(const ExecutionDispatchContext& context, void*) {
    if (context.op != SerialOp::position_prepare || !hip_graph_position())
        throw std::invalid_argument("graph position bookkeeping requires device-input scope");
    // Host position history is published by the replay owner after completion.
}
constexpr std::uint64_t graph_position_identity = 0x47525048504f5331ULL;
std::atomic<std::uint64_t> next_graph_generation{1};
std::uint64_t reserve_generation() {
    auto generation = next_graph_generation.load();
    for (;;) {
        if (!generation || generation == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("graph allocation generation exhausted");
        if (next_graph_generation.compare_exchange_weak(generation, generation + 1)) return generation;
    }
}
__global__ void validate_graph_logits(const float* values, std::uint32_t count, std::uint32_t* status) {
    const auto index = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && !isfinite(values[index])) atomicOr(status, 32U);
}
class CaptureDispatcher final : public EagerExecutionDispatcher {
public:
    CaptureDispatcher(const CompiledLaunchPlan& plan, std::uint32_t vocabulary) : direct_(plan), vocabulary_(vocabulary) {}
    void dispatch_kernel(const ExecutionDispatchContext& context) override {
        direct_.dispatch_kernel(context);
        if (context.op == SerialOp::lm_head) {
            const auto output = resolve_dispatch_write(context, 0);
            if (output.bytes < static_cast<std::uint64_t>(vocabulary_) * 4)
                throw std::invalid_argument("graph logits output is undersized");
            hipLaunchKernelGGL(validate_graph_logits, dim3((vocabulary_ - 1U) / 256U + 1U), dim3(256),
                0, hip_execution_stream(), reinterpret_cast<const float*>(output.address), vocabulary_, hip_graph_status());
            hip_check(hipGetLastError(), "graph logits finite validation");
        }
    }
    void dispatch_runtime_step(const ExecutionDispatchContext& context) override { direct_.dispatch_runtime_step(context); }
private:
    DirectLaunchDispatcher direct_;
    std::uint32_t vocabulary_;
};
}
struct HipSerialGraph::Impl {
    CompiledSerialExecution compiled;
    std::shared_ptr<HipLauncherCatalog> catalog;
    std::shared_ptr<const HipSerialGraphResources> owner;
    ExecutionLauncherRegistry launchers;
    CompiledLaunchPlan launches;
    GraphCapturePlan capture;
    HipLaunchContext scope;
    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr, abandoned_graph = nullptr;
    hipGraphExec_t executable = nullptr;
    void* auxiliary = nullptr;
    int device = -1;
    bool failed = false, pending = false, capturing = false;
    std::array<std::byte, 16> input{};
    std::array<std::uint32_t, 7> previous{}, next{};
    std::uint32_t status = 0, history_count = 0;
    std::vector<std::uint32_t> order;
    const ExecutionRuntimeResources& current() const {
        int active = -1; hip_check(hipGetDevice(&active), "graph active device");
        if (active != device) throw std::invalid_argument("graph device changed");
        const auto& resources = owner->current();
        const auto assessment = assess_compiled_serial_graph_replay(compiled, capture, resources, scope.arguments, 0);
        if (!assessment.candidate_eligible) throw std::invalid_argument("graph resources/bindings changed");
        return resources;
    }
    bool drain() noexcept {
        if (!stream && !auxiliary && !graph && !executable && !abandoned_graph) return true;
        try { int active = -1; if (hipGetDevice(&active) != hipSuccess || active != device) return false;
            if (stream && hipStreamQuery(stream) != hipSuccess) return false;
            pending = false; return true;
        } catch (...) { return false; }
    }
    bool destroy() noexcept {
        if (abandoned_graph) { if (hipGraphDestroy(abandoned_graph) != hipSuccess) return false; abandoned_graph = nullptr; }
        if (executable) { if (hipGraphExecDestroy(executable) != hipSuccess) return false; executable = nullptr; }
        if (graph) { if (hipGraphDestroy(graph) != hipSuccess) return false; graph = nullptr; }
        if (auxiliary) { if (hipFree(auxiliary) != hipSuccess) return false; auxiliary = nullptr; }
        if (stream) { if (hipStreamDestroy(stream) != hipSuccess) return false; stream = nullptr; }
        return true;
    }
    ~Impl() { (void)destroy(); }
    void replay(std::uint32_t token, std::uint64_t position) {
        const auto& resources = current();
        const auto packed = pack_graph_inputs(graph_input_layout(1), std::span<const std::uint32_t>(&token, 1),
            std::span<const std::uint64_t>(&position, 1), compiled.vocab_size,
            resources.session_state_layout->context_tokens);
        std::copy(packed.begin(), packed.end(), input.begin());
        const auto ngram = resolve_session_state(*resources.session_state_layout, *resources.session_state_memory,
            SessionStateComponent::ngram_history);
        const auto position_state = resolve_session_state(*resources.session_state_layout, *resources.session_state_memory,
            SessionStateComponent::position_history);
        if (ngram.storage != SessionStateStorage::host || ngram.bytes < history_count * 4ULL ||
            position_state.storage != SessionStateStorage::host || position_state.bytes < 8)
            throw std::invalid_argument("graph host bookkeeping bindings changed");
        if (history_count) std::memcpy(previous.data(), reinterpret_cast<const void*>(ngram.address), history_count * 4ULL);
        pending = true;
        hip_check(hipMemcpyAsync(reinterpret_cast<void*>(scope.arguments.memory.arena_base), input.data(), input.size(),
            hipMemcpyHostToDevice, stream), "graph input upload");
        if (history_count) hip_check(hipMemcpyAsync(reinterpret_cast<void*>(scope.ngram_input.arena_base), previous.data(),
            history_count * 4ULL, hipMemcpyHostToDevice, stream), "graph history upload");
        hip_check(hipGraphLaunch(executable, stream), "whole serial graph replay");
        hip_check(hipMemcpyAsync(&status, reinterpret_cast<void*>(scope.status.arena_base), 4, hipMemcpyDeviceToHost, stream),
            "graph completion status");
        if (history_count) hip_check(hipMemcpyAsync(next.data(), reinterpret_cast<void*>(scope.ngram_output.arena_base),
            history_count * 4ULL, hipMemcpyDeviceToHost, stream), "graph next history");
        hip_check(hipStreamSynchronize(stream), "graph replay completion");
        pending = false;
        if (status != 0) throw std::runtime_error("whole graph device validation failed");
        (void)current();
        if (history_count) std::memcpy(reinterpret_cast<void*>(ngram.address), next.data(), history_count * 4ULL);
        std::memcpy(reinterpret_cast<void*>(position_state.address), &position, 8);
    }
};
HipSerialGraph::HipSerialGraph(CompiledSerialExecution compiled, std::shared_ptr<HipLauncherCatalog> catalog,
    std::shared_ptr<const HipSerialGraphResources> owner, GraphMemoryRequest memory) : impl_(std::make_unique<Impl>()) {
    auto& p = *impl_;
    try {
    p.compiled = std::move(compiled); p.catalog = std::move(catalog); p.owner = std::move(owner);
    if (!p.catalog || !p.owner || !p.compiled.device_graph_inputs || !p.compiled.whole_plan_graph_safe ||
        memory.sessions != 1 || memory.shapes_per_session != 1 || memory.eligibility_classes_per_shape != 1 ||
        memory.workspace_bytes_per_instance < 112)
        throw std::invalid_argument("whole graph requires explicit C1 graph catalog, owner and auxiliary admission");
    const auto& model=p.catalog->gdn_config().model;
    require_qwen_rotary_device_payload(model.rotary_identity(),model.qwen_rotary_table.get());
    if(model.qwen_rotary_table != p.compiled.rotary_table)
        throw std::invalid_argument("whole graph catalog and compiled rotary allocation differ");
    const auto& resources = p.owner->current();
    if (!resources.session_state_layout || !resources.session_state_memory || resources.ple_lookup ||
        resources.ple_prefetch || resources.ple_hash_contract)
        throw std::invalid_argument("whole graph resources must omit eager host lookup/preprocessing");
    p.catalog->register_for_compiled_execution(p.compiled, p.launchers, &resources);
    // Catalog must register the dedicated no-host-write position implementation.
    const auto* position = p.launchers.find_runtime_launcher(SerialOp::position_prepare);
    if (!position || position->implementation_identity != graph_position_identity || position->function != &graph_position_prepare)
        throw std::invalid_argument("whole graph catalog has a host position consumer");
    p.launches = compile_launch_plan(p.compiled, p.launchers);
    hip_check(hipGetDevice(&p.device), "graph device");
    struct Range { std::uintptr_t base; std::uint64_t bytes; };
    const auto valid_range = [](Range range) {
        if (!range.base || !range.bytes || range.bytes > std::numeric_limits<std::uintptr_t>::max() - range.base)
            throw std::invalid_argument("graph resource address range is invalid");
    };
    const auto overlaps = [](Range a, Range b) {
        return a.base < b.base + b.bytes && b.base < a.base + a.bytes;
    };
    const auto device_binding = [&](Range range) {
        valid_range(range);
        hipPointerAttribute_t attributes{};
        hip_check(hipPointerGetAttributes(&attributes, reinterpret_cast<const void*>(range.base)), "graph resource device identity");
        if (attributes.type != hipMemoryTypeDevice || attributes.isManaged || attributes.device != p.device)
            throw std::invalid_argument("whole graph requires local owned device allocations");
        hipDeviceptr_t allocation_base{}; std::size_t allocation_bytes = 0;
        hip_check(hipMemGetAddressRange(&allocation_base, &allocation_bytes,
            reinterpret_cast<hipDeviceptr_t>(range.base)), "graph resource native extent");
        const auto base = reinterpret_cast<std::uintptr_t>(allocation_base);
        if (range.base < base || range.base - base > allocation_bytes || range.bytes > allocation_bytes - (range.base - base))
            throw std::invalid_argument("graph resource view exceeds native allocation");
    };
    if (!resources.scratch || !resources.model_weights) throw std::invalid_argument("graph lacks model/scratch bindings");
    const std::array<Range, 3> writable{{{resources.scratch->arena_base, p.compiled.arena.total_bytes},
        {resources.session_state_memory->device_base, resources.session_state_layout->device_total_bytes},
        {resources.session_state_memory->host_base, resources.session_state_layout->host_total_bytes}}};
    device_binding({resources.scratch->arena_base, resources.scratch->arena_bytes});
    device_binding({resources.session_state_memory->device_base, resources.session_state_memory->device_bytes});
    for (const auto range : writable) valid_range(range);
    for (std::size_t i = 0; i < writable.size(); ++i)
        for (std::size_t j = i + 1; j < writable.size(); ++j)
            if (overlaps(writable[i], writable[j])) throw std::invalid_argument("graph scratch and session state alias");
    if (const auto& table = p.compiled.rotary_table) {
        validate_qwen_rotary_reader(*table, nullptr);
        const Range read_only{reinterpret_cast<std::uintptr_t>(table->data()), table->descriptor().table_bytes()};
        device_binding(read_only);
        for (const auto range : writable)
            if (overlaps(read_only, range)) throw std::invalid_argument("graph rotary table aliases writable resources");
    }
    for (const auto& tensor : resources.model_weights->tensors) {
        const Range read_only{tensor.address, tensor.bytes};
        device_binding(read_only);
        for (const auto range : writable)
            if (overlaps(read_only, range)) throw std::invalid_argument("graph weights alias writable resources");
    }
    hipDeviceProp_t device_properties{}; hip_check(hipGetDeviceProperties(&device_properties, p.device), "graph device properties");
    const auto* override_arch = std::getenv("HSA_OVERRIDE_GFX_VERSION");
    if (std::string(device_properties.gcnArchName).substr(0, 7) != "gfx1151" || (override_arch && *override_arch))
        throw std::invalid_argument("whole graph requires actual gfx1151 without architecture override");
    hip_check(hipStreamCreateWithFlags(&p.stream, hipStreamNonBlocking), "graph stream create");
    hip_check(hipMalloc(&p.auxiliary, 128), "graph auxiliary allocation");
    hipDeviceptr_t actual_base{}; std::size_t actual_bytes = 0;
    hip_check(hipMemGetAddressRange(&actual_base, &actual_bytes, reinterpret_cast<hipDeviceptr_t>(p.auxiliary)), "graph auxiliary extent");
    hipDeviceptr_t scratch_base{}; std::size_t scratch_bytes = 0;
    hip_check(hipMemGetAddressRange(&scratch_base, &scratch_bytes,
        reinterpret_cast<hipDeviceptr_t>(resources.scratch->arena_base)), "graph retained scratch extent");
    const auto scratch_start = reinterpret_cast<std::uintptr_t>(scratch_base);
    if (resources.scratch->arena_base < scratch_start || resources.scratch->arena_base - scratch_start > scratch_bytes ||
        resources.scratch->arena_bytes > scratch_bytes - (resources.scratch->arena_base - scratch_start))
        throw std::invalid_argument("graph scratch view does not fit native allocation");
    if (reinterpret_cast<void*>(actual_base) != p.auxiliary || actual_bytes < 128 ||
        memory.workspace_bytes_per_instance < scratch_bytes ||
        actual_bytes - 16 > memory.workspace_bytes_per_instance - scratch_bytes)
        throw std::invalid_argument("graph retained scratch/auxiliary capacity exceeds admission");
    const auto generation = reserve_generation();
    const auto base = reinterpret_cast<std::uintptr_t>(p.auxiliary);
    p.scope.stream = p.stream; p.scope.graph_capture = true;
    p.scope.context_tokens = resources.session_state_layout->context_tokens;
    p.scope.vocabulary_size = static_cast<std::uint32_t>(p.compiled.vocab_size);
    p.scope.arguments = bind_graph_inputs(graph_input_layout(1), {p.compiled.execution_identity, base, 16, generation});
    p.scope.status = {p.compiled.execution_identity, base + 16, 4, generation};
    p.scope.ngram_input = {p.compiled.execution_identity, base + 32, 28, generation};
    p.scope.ngram_output = {p.compiled.execution_identity, base + 64, 28, generation};
    const auto ngram = resolve_session_state(*resources.session_state_layout, *resources.session_state_memory,
        SessionStateComponent::ngram_history);
    if (ngram.bytes > 28 || ngram.bytes % 4 || !ngram.bytes) throw std::invalid_argument("graph n-gram state exceeds fixed C1 bound");
    p.history_count = static_cast<std::uint32_t>(ngram.bytes / 4);
    p.capture = prepare_compiled_serial_graph_capture(p.compiled, resources, p.scope.arguments, memory);
    EagerExecutionRequest request;
    for (const auto& step : p.compiled.execution.steps) request.state.valid = request.state.valid | step.reads_state | step.writes_state;
    request.state.generations.fill(1);
    CaptureDispatcher dispatcher(p.launches, p.scope.vocabulary_size);
    try {
        hip_check(hipStreamBeginCapture(p.stream, hipStreamCaptureModeThreadLocal), "serial graph capture begin"); p.capturing = true;
        { HipLaunchScope scope(p.scope);
          p.order = execute_compiled_serial_eager(p.compiled, resources, request, dispatcher).completion_order; }
        hip_check(hipStreamEndCapture(p.stream, &p.graph), "serial graph capture end"); p.capturing = false;
        std::size_t nodes = 0; hip_check(hipGraphGetNodes(p.graph, nullptr, &nodes), "serial graph node inventory");
        if (!nodes || p.order.size() != p.compiled.execution.steps.size()) throw std::runtime_error("whole graph capture has missing work");
        hip_check(hipGraphInstantiate(&p.executable, p.graph, nullptr, nullptr, 0), "serial graph instantiate");
    } catch (...) {
        p.failed = true;
        if (p.capturing) { hipGraph_t invalid = nullptr; const auto result = hipStreamEndCapture(p.stream, &invalid);
            if (invalid && invalid != p.graph && hipGraphDestroy(invalid) != hipSuccess) p.abandoned_graph = invalid;
            p.capturing = result != hipSuccess && result != hipErrorStreamCaptureInvalidated; }
        if (p.capturing || !p.drain() || !p.destroy()) (void)impl_.release();
        throw;
    }
    } catch (...) {
        if (impl_ && (!p.drain() || !p.destroy())) (void)impl_.release();
        throw;
    }
}
HipSerialGraph::~HipSerialGraph() {
    if (impl_ && (!impl_->drain() || !impl_->destroy())) (void)impl_.release();
}
bool HipSerialGraph::failed() const noexcept { return impl_->failed; }
bool HipSerialGraph::drain() noexcept { return impl_->drain(); }
const GraphCapturePlan& HipSerialGraph::capture_plan() const noexcept { return impl_->capture; }
EagerExecutionResult HipSerialGraph::execute(SerialSessionContext& session, std::uint32_t token) {
    if (impl_->failed || impl_->pending) throw std::logic_error("whole graph is failed or retains a pending replay");
    const auto& resources = impl_->current();
    const auto eligibility = assess_compiled_serial_graph_replay(impl_->compiled, impl_->capture, resources,
        impl_->scope.arguments, session.next_position);
    if (!eligibility.candidate_eligible) throw std::invalid_argument("whole graph replay preflight failed");
    class ReplayDispatcher final : public EagerExecutionDispatcher {
    public:
        ReplayDispatcher(Impl& impl, std::uint32_t token, std::uint64_t position) : p(impl), t(token), pos(position) {}
        void dispatch_kernel(const ExecutionDispatchContext& context) override { dispatch(context); }
        void dispatch_runtime_step(const ExecutionDispatchContext& context) override { dispatch(context); }
        void dispatch(const ExecutionDispatchContext& context) {
            if (context.execution_identity != p.compiled.execution_identity || index >= p.order.size() ||
                context.step_id != p.order[index]) throw std::logic_error("graph dispatch order differs from captured DAG");
            if (!index) p.replay(t, pos);
            ++index;
        }
        Impl& p; std::uint32_t t; std::uint64_t pos; std::size_t index = 0;
    } dispatcher(*impl_, token, session.next_position);
    try { return execute_serial_token(impl_->compiled, session, resources, token, dispatcher); }
    catch (...) { if (session.health == SerialSessionHealth::poisoned) impl_->failed = true; throw; }
}
// Used only by the explicit device-input catalog.
void register_hip_graph_position_launcher(ExecutionLauncherRegistry& registry) {
    registry.add_runtime_launcher(SerialOp::position_prepare,
        {"runtime.device_graph_position", graph_position_identity, &graph_position_prepare, nullptr});
}
} // namespace strix
