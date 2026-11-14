#include "strix/hip_check.hpp"
#include "strix/hip_moe_overlap_dispatcher.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <vector>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix {
namespace {
std::atomic<std::uint64_t> next_overlap_owner{1};
std::uint64_t overlap_owner_epoch() {
    auto value = next_overlap_owner.load(std::memory_order_relaxed);
    for (;;) {
        if (!value || value == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("MoE overlap owner epoch exhausted");
        if (next_overlap_owner.compare_exchange_weak(value,value + 1U,std::memory_order_relaxed)) return value;
    }
}
void require(bool ok, const char* text) { if (!ok) throw std::logic_error(text); }
using strix::hip_check;
bool same_memory(const ExecutionMemoryBinding& a, const ExecutionMemoryBinding& b) {
    return a.execution_identity == b.execution_identity && a.arena_base == b.arena_base &&
           a.arena_bytes == b.arena_bytes && a.allocation_generation == b.allocation_generation;
}
bool same_state_memory(const SessionStateMemoryBinding& a, const SessionStateMemoryBinding& b) {
    return a.layout_identity == b.layout_identity && a.device_base == b.device_base && a.device_bytes == b.device_bytes &&
        a.device_allocation_generation == b.device_allocation_generation && a.host_base == b.host_base &&
        a.host_bytes == b.host_bytes && a.host_allocation_generation == b.host_allocation_generation;
}
bool same_tensor(const RuntimeTensorBinding& a, const RuntimeTensorBinding& b) {
    return a.tensor_index == b.tensor_index && a.storage == b.storage && a.address == b.address &&
        a.bytes == b.bytes && a.allocation_generation == b.allocation_generation && a.owner_node == b.owner_node;
}
bool same_layout(const SerialSessionStateLayout& a, const SerialSessionStateLayout& b) {
    return a.context_tokens == b.context_tokens && a.device_alignment == b.device_alignment &&
        a.host_alignment == b.host_alignment && a.device_total_bytes == b.device_total_bytes &&
        a.host_total_bytes == b.host_total_bytes && a.layout_identity == b.layout_identity &&
        a.recurrent_storage == b.recurrent_storage &&
        a.allocations.size() == b.allocations.size() && std::equal(a.allocations.begin(),a.allocations.end(),b.allocations.begin(),
            [](const auto& x,const auto& y) { return x.component == y.component && x.state_flag == y.state_flag &&
                x.storage == y.storage && x.offset == y.offset && x.bytes == y.bytes && x.provisional == y.provisional; });
}
constexpr std::array state_flags{PersistentState::attention_kv, PersistentState::recurrent_matrix,
    PersistentState::convolution_history, PersistentState::sparse_index, PersistentState::ngram_history,
    PersistentState::position_history};
bool branch_op(SerialOp op) {
    return op == SerialOp::moe_route || op == SerialOp::shared_expert ||
           op == SerialOp::moe_experts || op == SerialOp::moe_combine;
}
void trace_dependency(QueueDependencyKind kind,hipStream_t stream,hipEvent_t event,std::uint32_t layer) noexcept {
    if(auto* observer=current_execution_telemetry()) {
        auto correlation=current_execution_correlation();correlation.layer=layer;
        ExecutionCorrelationScope scoped(correlation);
        observer->queue_dependency(kind,reinterpret_cast<std::uintptr_t>(stream),reinterpret_cast<std::uintptr_t>(event));
    }
}
struct OverlapRange { std::uintptr_t address = 0; std::uint64_t bytes = 0; };
class OverlapAllocationLease;
struct OverlapLeaseRegistry { std::mutex mutex; std::vector<OverlapAllocationLease*> owners; };
OverlapLeaseRegistry& overlap_leases() { static OverlapLeaseRegistry registry; return registry; }
// Registry storage grows only when an adapter is constructed. The token path
// claims preallocated slots, never allocates tickets or infers completion from
// host return. It guards adapters using these virtual allocation views; external
// writes/aliases still obey the exclusive actual-owner contract.
class OverlapAllocationLease {
public:
    OverlapAllocationLease() {
        auto& registry = overlap_leases(); const std::lock_guard lock(registry.mutex);
        registry.owners.push_back(this);
    }
    ~OverlapAllocationLease() {
        // Declared before streams in Impl: their destructor drains first.
        auto& registry = overlap_leases(); const std::lock_guard lock(registry.mutex);
        registry.owners.erase(std::remove(registry.owners.begin(),registry.owners.end(),this),registry.owners.end());
    }
    OverlapAllocationLease(const OverlapAllocationLease&) = delete;
    OverlapAllocationLease& operator=(const OverlapAllocationLease&) = delete;
    void claim(std::array<OverlapRange,4> ranges) {
        auto& registry = overlap_leases(); const std::lock_guard lock(registry.mutex);
        require(!active_, "MoE overlap allocation ticket is already pending");
        for (const auto range : ranges)
            require(!range.bytes || (range.address && range.bytes <= std::numeric_limits<std::uintptr_t>::max() - range.address),
                "MoE overlap physical owner range overflows");
        for (std::size_t i = 0; i < ranges.size(); ++i) for (std::size_t j = i + 1U; j < ranges.size(); ++j) {
            const auto a = ranges[i], b = ranges[j];
            require(!a.bytes || !b.bytes || a.address >= b.address + b.bytes || b.address >= a.address + a.bytes,
                "MoE overlap writable owner ranges alias");
        }
        for (const auto* owner : registry.owners) if (owner != this && owner->active_)
            for (const auto a : ranges) for (const auto b : owner->ranges_)
                require(!a.bytes || !b.bytes || a.address >= b.address + b.bytes || b.address >= a.address + a.bytes,
                    "MoE overlap allocation is retained by another pending owner");
        ranges_ = ranges; active_ = true;
    }
    void release() noexcept {
        auto& registry = overlap_leases(); const std::lock_guard lock(registry.mutex);
        active_ = false;
    }
private:
    std::array<OverlapRange,4> ranges_{};
    bool active_ = false;
};
} // namespace
struct HipMoeSiblingStreams::Impl {
    enum class Phase { idle, forking, shared_recorded, routed };
    hipStream_t main = nullptr, side = nullptr;
    hipEvent_t fork = nullptr, join = nullptr;
    std::shared_ptr<const void> resources_owner;
    std::shared_ptr<const HipSetupQuiescenceGuard> guard;
    HipMoeOverlapStats counters;
    Phase phase = Phase::idle;
    std::uint64_t position = 0, committed_before = 0;
    std::uint32_t expected = 0, forks = 0, joins = 0, layer = 0;
    bool active = false, verified = false, broken = false;
    HipMoeWorkTicket work;
    Impl(hipStream_t borrowed, std::shared_ptr<const void> owner,
         std::shared_ptr<const HipSetupQuiescenceGuard> setup)
        : main(borrowed), resources_owner(std::move(owner)), guard(std::move(setup)) {
        work.owner_epoch = overlap_owner_epoch();
        require(main && main != hipStreamLegacy && main != hipStreamPerThread && resources_owner && guard,
                "MoE sibling streams require actual main/resources/guard owners");
        guard->require_original();
        unsigned flags = 0;
        hip_check(hipStreamGetFlags(main, &flags), "validate borrowed MoE main stream");
        try {
            hip_check(hipStreamCreateWithFlags(&side, hipStreamNonBlocking), "create MoE sibling stream");
            hip_check(hipEventCreateWithFlags(&fork, hipEventDisableTiming), "create MoE fork event");
            hip_check(hipEventCreateWithFlags(&join, hipEventDisableTiming), "create MoE join event");
        } catch (...) { cleanup(); throw; }
    }
    void cleanup() noexcept {
        if (!guard) return;
        guard->recover_or_fail_stop();
        const auto release = [](hipError_t e) { if (e != hipSuccess) std::_Exit(3); };
        if (join) { release(hipEventDestroy(join)); join = nullptr; }
        if (fork) { release(hipEventDestroy(fork)); fork = nullptr; }
        if (side) { release(hipStreamDestroy(side)); side = nullptr; }
    }
    ~Impl() { cleanup(); }
    void abort() noexcept {
        const bool was_active = active;
        if (active) guard->recover_or_fail_stop();
        if (active && counters.aborted_tokens != std::numeric_limits<std::uint64_t>::max()) ++counters.aborted_tokens;
        active = false; verified = false; broken = true; phase = Phase::idle;
        work.state = HipMoeWorkState::failed_drained;
        if (was_active) { work.validated = false; work.committed = false; }
    }
    template<class Work> void protected_call(Work&& call) {
        try { call(); } catch (...) { abort(); throw; }
    }
    void begin(std::uint64_t p, std::uint64_t committed, std::uint32_t layers, MoeTokenSchedule schedule) {
        require(schedule == MoeTokenSchedule::serial_expanded || schedule == MoeTokenSchedule::fork_join,
                "invalid MoE token schedule");
        require(!active && !broken && phase == Phase::idle && layers &&
                work.token_epoch < std::numeric_limits<std::uint64_t>::max() &&
                p < std::numeric_limits<std::uint64_t>::max() && committed < std::numeric_limits<std::uint64_t>::max(),
                "MoE sibling token is invalid, already active or failed");
        require(layers <= std::numeric_limits<std::uint64_t>::max() - counters.fork_records &&
                layers <= std::numeric_limits<std::uint64_t>::max() - counters.join_waits &&
                layers <= std::numeric_limits<std::uint64_t>::max() - counters.committed_layers &&
                counters.verified_tokens < std::numeric_limits<std::uint64_t>::max() &&
                counters.committed_tokens < std::numeric_limits<std::uint64_t>::max() &&
                counters.serial_tokens < std::numeric_limits<std::uint64_t>::max() &&
                counters.fork_join_tokens < std::numeric_limits<std::uint64_t>::max(), "MoE sibling counters exhausted");
        guard->require_original();
        hipStreamCaptureStatus main_capture{}, side_capture{};
        hip_check(hipStreamIsCapturing(main, &main_capture), "query MoE main capture");
        hip_check(hipStreamIsCapturing(side, &side_capture), "query MoE side capture");
        require(main_capture == hipStreamCaptureStatusNone && side_capture == hipStreamCaptureStatusNone,
                "MoE sibling phase one rejects graph capture");
        position = p; committed_before = committed; expected = schedule == MoeTokenSchedule::fork_join ? layers : 0U; forks = joins = 0;
        work.schedule = schedule;
        active = true; verified = false;
        ++work.token_epoch; work.input_position = p; work.committed_before = committed;
        work.state = HipMoeWorkState::prepared; work.gpu_completed = work.validated = work.committed = false;
        work.completion_queried = false;
        work.main_completion = work.side_completion = hipSuccess;
    }
    void expected_phase(Phase wanted, std::uint32_t current_layer) {
        require(active && !verified && !broken && phase == wanted &&
                (wanted == Phase::idle || layer == current_layer), "MoE sibling branch phase/layer differs");
    }
};
HipMoeSiblingStreams::HipMoeSiblingStreams(hipStream_t main, std::shared_ptr<const void> owner,
    std::shared_ptr<const HipSetupQuiescenceGuard> guard)
    : impl_(std::make_unique<Impl>(main, std::move(owner), std::move(guard))) {}
HipMoeSiblingStreams::~HipMoeSiblingStreams() = default;
void HipMoeSiblingStreams::begin_token(std::uint64_t position, std::uint64_t committed, std::uint32_t layers, MoeTokenSchedule schedule) {
    impl_->protected_call([&] { impl_->begin(position, committed, layers, schedule); });
}
void HipMoeSiblingStreams::fork_shared(std::uint32_t layer) {
    auto& s = *impl_; s.protected_call([&] { s.expected_phase(Impl::Phase::idle, layer);
    require(s.forks < s.expected, "too many MoE sibling forks");
    s.layer = layer; s.phase = Impl::Phase::forking;
    try {
        hip_check(hipEventRecord(s.fork, s.main), "record shared producer-ready");
        trace_dependency(QueueDependencyKind::event_record,s.main,s.fork,layer);
        ++s.counters.fork_records; // Actual successful fork-event recording, even if a later operation fails.
        s.work.state = HipMoeWorkState::submitted;
        hip_check(hipStreamWaitEvent(s.side, s.fork, 0), "side waits for shared input producer");
        trace_dependency(QueueDependencyKind::event_wait,s.side,s.fork,layer);
    } catch (...) { s.abort(); throw; }
    });
}
void HipMoeSiblingStreams::record_shared_done(std::uint32_t layer) {
    auto& s = *impl_; s.protected_call([&] { s.expected_phase(Impl::Phase::forking, layer);
    try { hip_check(hipEventRecord(s.join, s.side), "record shared branch done");
        trace_dependency(QueueDependencyKind::event_record,s.side,s.join,layer); }
    catch (...) { s.abort(); throw; }
    ++s.forks; s.phase = Impl::Phase::shared_recorded;
    });
}
void HipMoeSiblingStreams::begin_experts(std::uint32_t layer) {
    auto& s = *impl_; s.protected_call([&] { s.expected_phase(Impl::Phase::shared_recorded, layer);
    s.phase = Impl::Phase::routed;
    });
}
void HipMoeSiblingStreams::join_before_combine(std::uint32_t layer) {
    auto& s = *impl_; s.protected_call([&] { s.expected_phase(Impl::Phase::routed, layer);
    try { hip_check(hipStreamWaitEvent(s.main, s.join, 0), "main joins shared before combine");
        trace_dependency(QueueDependencyKind::event_wait,s.main,s.join,layer); }
    catch (...) { s.abort(); throw; }
    ++s.joins; ++s.counters.join_waits; s.phase = Impl::Phase::idle;
    });
}
void HipMoeSiblingStreams::verify_before_commit() {
    auto& s = *impl_; s.protected_call([&] {
    require(s.active && !s.broken && !s.verified && s.phase == Impl::Phase::idle &&
            s.forks == s.expected && s.joins == s.expected, "incomplete MoE sibling token cannot commit");
    try {
        // Collect both results before choosing a deterministic main-then-side
        // completion error. The side branch never races on D1's logical record.
        s.work.main_completion = hipStreamSynchronize(s.main);
        s.work.side_completion = hipStreamQuery(s.side);
        s.work.completion_queried = true;
        hip_check(s.work.main_completion, "drain main including shared joins");
        hip_check(s.work.side_completion, "prove shared stream complete before publication");
        s.work.gpu_completed = true; s.work.state = HipMoeWorkState::gpu_completed;
        s.guard->require_original();
    } catch (...) { s.abort(); throw; }
    s.verified = true; ++s.counters.verified_tokens;
    s.work.validated = true; s.work.state = HipMoeWorkState::validated;
    });
}
void HipMoeSiblingStreams::mark_committed(std::uint64_t next, std::uint64_t committed) {
    auto& s = *impl_; s.protected_call([&] {
    require(s.active && s.verified && !s.broken && next == s.position + 1U && committed == s.committed_before + 1U,
            "MoE sibling completion requires actual serial-token commit");
    ++s.counters.committed_tokens; s.counters.committed_layers += s.expected;
    if (s.work.schedule == MoeTokenSchedule::serial_expanded) ++s.counters.serial_tokens;
    else ++s.counters.fork_join_tokens;
    s.work.committed = true; s.work.state = HipMoeWorkState::committed;
    s.active = false; s.verified = false;
    // Completion and validation preceded commit, so this owner may now reuse
    // its persistent events and arena on the next monotonically new ticket.
    s.work.state = HipMoeWorkState::recyclable;
    });
}
void HipMoeSiblingStreams::abort() noexcept { impl_->abort(); }
hipStream_t HipMoeSiblingStreams::main_stream() const noexcept { return impl_->main; }
hipStream_t HipMoeSiblingStreams::side_stream() const noexcept { return impl_->side; }
HipMoeOverlapStats HipMoeSiblingStreams::stats() const noexcept { return impl_->counters; }
bool HipMoeSiblingStreams::failed() const noexcept { return impl_->broken; }
HipMoeWorkTicket HipMoeSiblingStreams::ticket() const noexcept { return impl_->work; }

struct HipMoeOverlapDispatcher::Impl {
    const CompiledSerialExecution compiled; // Deep immutable admission snapshot.
    std::vector<std::vector<std::size_t>> accessed_allocations;
    std::vector<std::vector<std::size_t>> accessed_weights;
    const ExecutionRuntimeResources& resources;
    const ExecutionRuntimeResources resource_snapshot;
    MoeForkJoinArenaPlan plan;
    CompiledLaunchPlan launches;
    DirectLaunchDispatcher direct;
    ExecutionMemoryBinding scratch;
    RuntimeExecutionIdentities runtime;
    SessionStateMemoryBinding state_memory;
    SerialSessionStateLayout state_layout;
    ModelRuntimeBinding weights;
    std::uint64_t manifest_identity = 0, access_identity = 0;
    std::array<std::uint64_t,6> writes{};
    const SerialSessionContext* session_owner = nullptr;
    std::uint64_t session_epoch = 0;
    PersistentStateGenerations entry_state, committed_state;
    std::uint64_t committed_before = 0, entry_history = 0;
    std::size_t entry_intervals = 0;
    ExecutionMemoryBinding d1_record;
    bool d1_bound = false, verified = false;
    std::shared_ptr<const void> launcher_owner;
    OverlapAllocationLease allocation_lease;
    HipMoeSiblingStreams streams;
    std::size_t cursor = 0, layer_cursor = 0;
    std::uint64_t position = 0;
    bool active = false;
    MoeTokenSchedule schedule = MoeTokenSchedule::fork_join;
    Impl(const CompiledSerialExecution& execution, const MoeForkJoinArenaPlan& arena,
         const CompiledLaunchPlan& launch, const ExecutionRuntimeResources& bound, hipStream_t main,
         std::shared_ptr<const void> owner, std::shared_ptr<const void> configs,
         std::shared_ptr<const HipSetupQuiescenceGuard> guard, ExecutionTelemetry* telemetry)
        : compiled(execution), resources(bound), resource_snapshot(bound),
          plan(arena), launches(launch), direct(launches, telemetry),
          scratch(bound.scratch ? *bound.scratch : ExecutionMemoryBinding{}), launcher_owner(std::move(configs)),
          streams(main, std::move(owner), std::move(guard)) {
        plan.validate_execution(compiled);
        plan.validate_binding(scratch);
        require(launcher_owner && !compiled.device_graph_inputs && compiled.logical_rows == 1 &&
                compiled.request_concurrency == 1 && launches.execution_identity == compiled.execution_identity &&
                launches.plan_identity && launches.steps.size() == compiled.execution.steps.size() &&
                compiled.io.steps.size() == compiled.execution.steps.size() &&
                compiled.bindings.bindings.size() == compiled.execution.steps.size(), "MoE overlap requires fresh expanded C1 launch plan/resources");
        runtime = build_runtime_execution_identities(compiled.execution_identity, compiled.arena, resources);
        state_memory = *resources.session_state_memory; state_layout = *resources.session_state_layout;
        weights = *resources.model_weights;
        manifest_identity = resources.model_manifest->layout_identity;
        access_identity = resources.tensor_access->plan_identity;
        for (std::size_t i = 0; i < launches.steps.size(); ++i) {
            const auto& step = compiled.execution.steps[i];
            const auto& launcher = launches.steps[i];
            require(step.id == i && launcher.step_id == step.id && launcher.op == step.op && launcher.function &&
                    launcher.implementation_identity && launcher.runtime_step == compiled.bindings.bindings[i].host_or_runtime_step,
                    "MoE overlap compiled launcher differs");
            for (std::size_t flag = 0; flag < state_flags.size(); ++flag)
                if (has_state(step.writes_state,state_flags[flag])) ++writes[flag];
        }
        accessed_allocations.resize(compiled.io.steps.size());
        accessed_weights.resize(compiled.io.steps.size());
        for (std::size_t step = 0; step < compiled.io.steps.size(); ++step) {
            const auto cache = [&](const auto& uses) {
                for (const auto use : uses) {
                    const auto found = std::find_if(compiled.arena.allocations.begin(), compiled.arena.allocations.end(),
                        [&](const auto& allocation) { return allocation.use == use; });
                    require(found != compiled.arena.allocations.end(), "MoE overlap IO has no arena allocation");
                    accessed_allocations[step].push_back(static_cast<std::size_t>(found - compiled.arena.allocations.begin()));
                }
            };
            cache(compiled.io.steps[step].reads); cache(compiled.io.steps[step].writes); cache(compiled.io.steps[step].temporaries);
        }
        for (const auto& access : resources.tensor_access->accesses) {
            require(access.step_id < accessed_weights.size(), "MoE overlap tensor access step is invalid");
            for (const auto index : access.tensor_indices) {
                const auto found = std::find_if(weights.tensors.begin(),weights.tensors.end(),
                    [&](const auto& tensor) { return tensor.tensor_index == index; });
                require(found != weights.tensors.end(), "MoE overlap tensor access lacks its actual weight owner");
                accessed_weights[access.step_id].push_back(static_cast<std::size_t>(found - weights.tensors.begin()));
            }
        }
        for (const auto& layer : plan.layers()) {
            require(layer.route_step + 1U == layer.shared_step && layer.shared_step + 1U == layer.experts_step &&
                    layer.experts_step + 1U == layer.combine_step && !launches.steps[layer.route_step].runtime_step &&
                    !launches.steps[layer.shared_step].runtime_step &&
                    !launches.steps[layer.experts_step].runtime_step && !launches.steps[layer.combine_step].runtime_step,
                    "MoE overlap requires unchanged route/shared/experts/combine host order");
        }
    }
    void validate_owner_epoch(bool all_weights) const {
        require(resources.scratch == resource_snapshot.scratch && resources.scratch && same_memory(*resources.scratch,scratch) &&
            resources.model_manifest == resource_snapshot.model_manifest && resources.model_weights == resource_snapshot.model_weights &&
            resources.tensor_access == resource_snapshot.tensor_access && resources.session_state_layout == resource_snapshot.session_state_layout &&
            resources.session_state_memory == resource_snapshot.session_state_memory && resources.ple_lookup == resource_snapshot.ple_lookup &&
            resources.ple_hash_contract == resource_snapshot.ple_hash_contract && resources.ple_prefetch == resource_snapshot.ple_prefetch &&
            resources.model_manifest->layout_identity == manifest_identity && resources.tensor_access->plan_identity == access_identity &&
            same_state_memory(*resources.session_state_memory,state_memory) && same_layout(*resources.session_state_layout,state_layout) &&
            resources.model_weights->manifest_layout_identity == weights.manifest_layout_identity &&
            resources.model_weights->model_content_identity == weights.model_content_identity &&
            resources.model_weights->tensors.size() == weights.tensors.size(),
            "MoE overlap retained allocation owner or generation changed");
        if (all_weights)
            require(std::equal(weights.tensors.begin(),weights.tensors.end(),resources.model_weights->tensors.begin(),same_tensor),
                "MoE overlap retained weight allocation or generation changed");
    }
    void validate_d1() const {
        const auto* scope = current_hip_launch_context();
        require(scope && !scope->graph_capture && scope->stream == streams.main_stream() &&
            scope->deferred_moe_status.execution_identity == compiled.execution_identity &&
            scope->deferred_moe_status.arena_bytes == 8 && scope->deferred_moe_status.allocation_generation &&
            scope->deferred_moe_status.arena_base && scope->deferred_moe_status.arena_base % alignof(std::uint32_t) == 0 &&
            scope->deferred_moe_status.arena_base <= std::numeric_limits<std::uintptr_t>::max() - 8U &&
            (!d1_bound || same_memory(scope->deferred_moe_status,d1_record)),
            "MoE overlap D1 record owner or allocation epoch changed");
    }
    void validate_unpublished_session() const {
        require(session_owner && session_owner->execution_identity == compiled.execution_identity &&
            session_owner->runtime_compatibility_identity == runtime.compatibility_identity &&
            session_owner->state_layout_identity == state_layout.layout_identity && session_owner->health == SerialSessionHealth::ready &&
            session_owner->next_position == position && session_owner->committed_tokens == committed_before &&
            session_owner->state.valid == entry_state.valid && session_owner->state.generations == entry_state.generations &&
            !session_owner->arithmetic_lineage.tainted && session_owner->arithmetic_lineage.history_identity == entry_history &&
            session_owner->arithmetic_lineage.intervals.size() == entry_intervals,
            "MoE overlap session was moved, modified or published before completion");
    }
    void dispatch(const ExecutionDispatchContext& c, bool runtime_step) {
        try {
            require(active, "MoE overlap dispatch has no active token");
            validate_owner_epoch(false); validate_d1(); validate_unpublished_session();
            require(active && cursor < compiled.execution.steps.size() && c.step_id == cursor &&
                    c.execution_identity == compiled.execution_identity && c.position == position &&
                    c.logical_rows == 1 && c.request_concurrency == 1 && c.arena &&
                    c.arena->alignment == compiled.arena.alignment && c.arena->total_bytes == compiled.arena.total_bytes &&
                    c.arena->allocations.size() == compiled.arena.allocations.size() &&
                    c.resources == &resources && c.memory && resources.scratch &&
                    same_memory(*c.memory, scratch) && same_memory(*resources.scratch, scratch) &&
                    c.session_state_layout == resources.session_state_layout && c.session_state_memory == resources.session_state_memory,
                    "MoE overlap live context differs from retained execution/resources");
            for (const auto index : accessed_allocations[cursor]) {
                // Prepared eager owns an equivalent immutable arena copy. Validate
                // every accessed mapping, without rejecting its distinct address.
                const auto& actual = c.arena->allocations[index];
                const auto& expected = compiled.arena.allocations[index];
                require(actual.use == expected.use && actual.first_step == expected.first_step && actual.last_step == expected.last_step &&
                        actual.offset == expected.offset && actual.bytes == expected.bytes, "MoE overlap arena mapping changed");
            }
            for (const auto slot : accessed_weights[cursor])
                require(same_tensor(resources.model_weights->tensors[slot],weights.tensors[slot]),
                    "MoE overlap step weight allocation or generation changed");
            const auto& step = compiled.execution.steps[cursor];
            const auto& io = compiled.io.steps[cursor];
            const auto& binding = compiled.bindings.bindings[cursor];
            require(c.op == step.op && c.layer == step.layer && c.reads_state == step.reads_state &&
                    c.writes_state == step.writes_state && binding.host_or_runtime_step == runtime_step &&
                    std::equal(c.reads.begin(),c.reads.end(),io.reads.begin(),io.reads.end()) &&
                    std::equal(c.writes.begin(),c.writes.end(),io.writes.begin(),io.writes.end()) &&
                    std::equal(c.temporaries.begin(),c.temporaries.end(),io.temporaries.begin(),io.temporaries.end()) &&
                    (binding.kernel_id ? (c.kernel_id && *c.kernel_id == *binding.kernel_id) : !c.kernel_id),
                    "MoE overlap step/IO/binding changed");
            require(!runtime_step || !branch_op(c.op), "MoE overlap branches cannot be runtime dispatches");
            if (c.op == SerialOp::shared_expert || c.op == SerialOp::moe_experts || c.op == SerialOp::moe_combine) {
                require(layer_cursor < plan.layers().size(), "unexpected MoE overlap layer");
                const auto& layer = plan.layers()[layer_cursor];
                require(c.layer && *c.layer == layer.layer, "MoE overlap layer identity differs");
                if (c.op == SerialOp::shared_expert) {
                    require(c.step_id == layer.shared_step && c.writes_state == PersistentState::none,
                            "unexpected shared branch or persistent state writer");
                    if (schedule == MoeTokenSchedule::serial_expanded) {
                        direct.dispatch_kernel(c);
                    } else {
                        streams.fork_shared(layer.layer);
                        {
                            HipSharedExpertSideScope side(streams.main_stream(), streams.side_stream());
                            direct.dispatch_kernel(c);
                        }
                        streams.record_shared_done(layer.layer);
                    }
                } else if (c.op == SerialOp::moe_experts) {
                    require(c.step_id == layer.experts_step, "unexpected routed branch");
                    if (schedule == MoeTokenSchedule::fork_join) streams.begin_experts(layer.layer);
                    direct.dispatch_kernel(c);
                } else {
                    require(c.step_id == layer.combine_step, "unexpected MoE combine");
                    if (schedule == MoeTokenSchedule::fork_join) streams.join_before_combine(layer.layer);
                    direct.dispatch_kernel(c); ++layer_cursor;
                }
            } else {
                if (runtime_step) direct.dispatch_runtime_step(c); else direct.dispatch_kernel(c);
            }
            ++cursor;
        } catch (...) { streams.abort(); allocation_lease.release(); active = false; throw; }
    }
};
HipMoeOverlapDispatcher::HipMoeOverlapDispatcher(const CompiledSerialExecution& compiled,
    const MoeForkJoinArenaPlan& plan, const CompiledLaunchPlan& launches, const ExecutionRuntimeResources& resources,
    hipStream_t main, std::shared_ptr<const void> owner, std::shared_ptr<const void> launchers,
    std::shared_ptr<const HipSetupQuiescenceGuard> guard, ExecutionTelemetry* telemetry)
    : impl_(std::make_unique<Impl>(compiled,plan,launches,resources,main,std::move(owner),std::move(launchers),std::move(guard),telemetry)) {}
HipMoeOverlapDispatcher::~HipMoeOverlapDispatcher() = default;
void HipMoeOverlapDispatcher::begin_token(const SerialSessionContext& session, MoeTokenSchedule schedule) {
    auto& s = *impl_;
    try {
    require(!s.active && !s.streams.failed(), "MoE overlap token is already active or failed");
    require(schedule == MoeTokenSchedule::serial_expanded || schedule == MoeTokenSchedule::fork_join,
            "invalid MoE token schedule");
    s.validate_owner_epoch(true); s.validate_d1();
    require((!s.session_owner || s.session_owner == &session) &&
            session.health == SerialSessionHealth::ready && !session.arithmetic_lineage.tainted &&
            session.execution_identity == s.compiled.execution_identity &&
            session.state_layout_identity == s.state_layout.layout_identity &&
            session.runtime_compatibility_identity == s.runtime.compatibility_identity && s.resources.scratch &&
            same_memory(*s.resources.scratch,s.scratch), "MoE overlap token session/binding differs");
    if (session.arithmetic_lineage.history_identity || !session.arithmetic_lineage.intervals.empty())
        require(arithmetic_lineage_valid(session.arithmetic_lineage,session.next_position), "MoE overlap session lineage is invalid");
    if (!s.session_owner) s.session_epoch = overlap_owner_epoch();
    s.committed_state = session.state;
    for (std::size_t i = 0; i < s.writes.size(); ++i) {
        require(s.writes[i] <= std::numeric_limits<std::uint64_t>::max() - session.state.generations[i],
            "MoE overlap state generation overflow");
        s.committed_state.generations[i] += s.writes[i];
        if (s.writes[i]) s.committed_state.valid = s.committed_state.valid | state_flags[i];
    }
    const auto& record = current_hip_launch_context()->deferred_moe_status;
    s.allocation_lease.claim({{{s.scratch.arena_base,s.scratch.arena_bytes},
        {s.state_memory.device_base,s.state_memory.device_bytes}, {s.state_memory.host_base,s.state_memory.host_bytes},
        {record.arena_base,record.arena_bytes}}});
    s.streams.begin_token(session.next_position,session.committed_tokens,static_cast<std::uint32_t>(s.plan.layers().size()),schedule);
    s.schedule = schedule;
    s.session_owner = &session; s.entry_state = session.state; s.committed_before = session.committed_tokens;
    s.entry_history = session.arithmetic_lineage.history_identity; s.entry_intervals = session.arithmetic_lineage.intervals.size();
    s.d1_record = current_hip_launch_context()->deferred_moe_status; s.d1_bound = true;
    s.position = session.next_position; s.cursor = 0; s.layer_cursor = 0; s.active = true; s.verified = false;
    } catch (...) { s.streams.abort(); s.allocation_lease.release(); s.active = false; throw; }
}
void HipMoeOverlapDispatcher::dispatch_kernel(const ExecutionDispatchContext& c) { impl_->dispatch(c,false); }
void HipMoeOverlapDispatcher::dispatch_runtime_step(const ExecutionDispatchContext& c) { impl_->dispatch(c,true); }
void HipMoeOverlapDispatcher::verify_before_commit(bool terminal_head_omitted) {
    auto& s = *impl_;
    try {
    require(s.active, "MoE overlap verification has no active token");
    s.validate_owner_epoch(true); s.validate_d1(); s.validate_unpublished_session();
    require(s.active && s.layer_cursor == s.plan.layers().size(),
            "MoE overlap missing compiled layers cannot publish");
    if (terminal_head_omitted) {
        require(!s.compiled.execution.steps.empty() && s.cursor == s.compiled.execution.steps.size() - 1U,
                "MoE overlap prefix may omit exactly one terminal head");
        const auto& head = s.compiled.execution.steps.back();
        const auto& binding = s.compiled.bindings.bindings.back();
        require(head.id == s.cursor && head.op == SerialOp::lm_head && !head.layer &&
                head.reads_state == PersistentState::none && head.writes_state == PersistentState::none &&
                !binding.host_or_runtime_step && !s.launches.steps.back().runtime_step,
                "MoE overlap omitted head must be a terminal stateless kernel");
        // Logical traversal only: the exterior output dispatcher admitted the
        // omission. No launcher, telemetry or device-work counter is advanced.
        ++s.cursor;
    }
    require(s.cursor == s.compiled.execution.steps.size(),
            "MoE overlap missing compiled steps cannot publish");
    s.streams.verify_before_commit();
    s.verified = true;
    } catch (...) { s.streams.abort(); s.allocation_lease.release(); s.active = false; s.verified = false; throw; }
}
void HipMoeOverlapDispatcher::mark_committed(const SerialSessionContext& session) {
    auto& s = *impl_;
    try {
    s.validate_owner_epoch(false); s.validate_d1();
    require(s.active && s.verified && s.session_owner == &session && session.execution_identity == s.compiled.execution_identity &&
            session.state_layout_identity == s.state_layout.layout_identity &&
            session.runtime_compatibility_identity == s.runtime.compatibility_identity && !session.arithmetic_lineage.tainted &&
            session.state.valid == s.committed_state.valid && session.state.generations == s.committed_state.generations &&
            session.health == SerialSessionHealth::ready,
            "MoE overlap commit session differs");
    if (s.position == 0 || s.entry_history) {
        require(arithmetic_lineage_valid(session.arithmetic_lineage,session.next_position), "MoE overlap committed lineage is invalid");
        const auto& last = session.arithmetic_lineage.intervals.back();
        require(last.producer == ArithmeticProducer::serial_row && last.end == session.next_position &&
            last.numerical_plan_identity == s.compiled.execution_identity && last.state_update_identity == s.compiled.execution_identity &&
            last.reduction_plan_identity == s.compiled.kernel_plan_identity, "MoE overlap committed numerical lineage differs");
    } else require(!session.arithmetic_lineage.history_identity && session.arithmetic_lineage.intervals.empty(),
        "MoE overlap invented previously unknown lineage");
    s.streams.mark_committed(session.next_position,session.committed_tokens); s.active = false;
    s.allocation_lease.release();
    s.verified = false;
    } catch (...) { s.streams.abort(); s.allocation_lease.release(); s.active = false; s.verified = false; throw; }
}
void HipMoeOverlapDispatcher::abort() noexcept {
    impl_->streams.abort(); impl_->allocation_lease.release(); impl_->active = false; impl_->verified = false;
}
HipMoeOverlapStats HipMoeOverlapDispatcher::stats() const noexcept { return impl_->streams.stats(); }
HipMoeWorkTicket HipMoeOverlapDispatcher::ticket() const noexcept {
    auto result = impl_->streams.ticket();
    result.binding_identity = impl_->runtime.binding_identity;
    result.scratch_generation = impl_->scratch.allocation_generation;
    result.session_epoch = impl_->session_epoch;
    result.state_device_generation = impl_->state_memory.device_allocation_generation;
    result.state_host_generation = impl_->state_memory.host_allocation_generation;
    result.status_generation = impl_->d1_record.allocation_generation;
    return result;
}
} // namespace strix
