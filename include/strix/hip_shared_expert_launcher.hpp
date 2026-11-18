#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/shared_expert_prefetch.hpp"

#include <cstdint>
#include <string>
#include <memory>
#include <span>
#include <vector>

namespace strix {

struct HipSharedExpertLauncherConfig {
    std::uint32_t hidden_size = 0;
    std::uint32_t intermediate_size = 0;
    HipActivationStorage activation =
        HipActivationStorage::bf16;
    bool prefetch_q8 = false;
    std::vector<std::uint32_t> prefetch_eligible_layers;
    std::shared_ptr<shared_expert_prefetch::RuntimeDispatchStats> prefetch_stats;
    // Dense A8 numerical profile (dense-a8-q80-v1): every Q8_0 projection of the
    // step (R1, windows and prompt rows) runs Q8_0 x per-32 int8 activations
    // through q8_rows::launch_dense_a8. Different numerics from the exact path.
    bool dense_a8_projections = false;
    // fast-prefill-f16wmma-v1: prompt launches of 32..256 rows run q8_rows::launch_dense_f16 (prefill only).
    bool dense_f16_prefill = false;
    // fast-prefill component dense8: prompt launches of 64..256 rows run q8_rows::launch_dense_g64 (prefill only).
    bool dense_g64_prefill = false;
};

enum class HipSharedQ8ProjectionMode {legacy256,virtual256,prefetch8};
enum class HipSharedQ8ProjectionRole {gate_up,down};
struct HipSharedQ8ProjectionView {std::uintptr_t address=0;std::uint64_t bytes=0;};
// Diagnostic uses the exact native kernels, no allocation/sync. Caller owns views
// through stream drain, including enqueue failure. Gate/up: storage is input
// dtype, FP32 output and empty gate view. Down: FP32 input, FP32 shared_gate,
// storage selects the output dtype. Complete exact views, output disjoint.
void launch_hip_shared_q8_projection_probe(HipSharedQ8ProjectionRole,HipSharedQ8ProjectionMode,
    HipActivationStorage storage,HipSharedQ8ProjectionView input,HipSharedQ8ProjectionView weights,
    HipSharedQ8ProjectionView shared_gate,HipSharedQ8ProjectionView output,
    std::uint32_t width,std::uint32_t outputs,void* stream);
#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
// Test-only T = 1/2/4/8/16/32/64 rows of the prefetch8 BF16 gate/up (FP32 out) or FP32 down
// (per-row FP32 gate, BF16 out) projection: the q8_rows engine (exact or A8) or
// the R1 kernel once per row. Exact 16-byte aligned views; gate empty for gate/up.
enum class HipSharedRowsProbeMode {
    rows, rows_a8, row_loop,
    // Explicit A8 organizations for bitwise qualification independent of the environment.
    rows_a8_reference, rows_a8_narrow
};
void launch_hip_shared_rows_projection_probe(HipSharedQ8ProjectionRole, HipSharedRowsProbeMode,
    std::span<const HipSharedQ8ProjectionView> input, HipSharedQ8ProjectionView weights,
    std::span<const HipSharedQ8ProjectionView> gate, std::span<const HipSharedQ8ProjectionView> output,
    std::uint32_t width, std::uint32_t outputs, void* stream);
// T = 1/2/4/8 rows of the dense A8 sites with GSQ-RCO typed segments (ggml types, 8 = Q8_0) through the engine's own
// site types: gate/up (1..2 segments, output[r] holds every segment's outputs) or down (one segment, per-row FP32
// gate, BF16 out; narrow = the R8/M640 SharedDownR8NarrowSite organization).
void launch_hip_shared_gsq_rows_probe(HipSharedQ8ProjectionRole, bool narrow, std::span<const std::uint32_t> types,
    std::span<const HipSharedQ8ProjectionView> weights, std::span<const std::uint32_t> counts,
    std::span<const HipSharedQ8ProjectionView> input, std::span<const HipSharedQ8ProjectionView> gate,
    std::span<const HipSharedQ8ProjectionView> output, std::uint32_t width, void* stream);
#endif

void register_hip_shared_expert_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipSharedExpertLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_shared_expert_launcher(
    std::string kernel_id,
    HipSharedExpertLauncherConfig* config);

// Known-prefix window only: both contexts are C1 row views of one retained
// window allocation and the same serial shared-expert step. Gate/up/down are
// Q8_0, the scalar gate is F32, and H2560/M640 activations are BF16. Prepared
// caches are absent; row.memory is null or the resources->scratch object itself.
// Only the exact base
// and prefetch serial IDs are admitted. Preflight performs no HIP enqueue.
// Launch repeats the complete two-row preflight and requires an explicit eager
// HipLaunchScope. The caller retains both contexts, config, all resource/weight
// owners and the stream through quiescence, including after a launch exception.
void shared_expert_known_window_r2_preflight(
    std::span<const ExecutionDispatchContext, 2> rows,
    const HipSharedExpertLauncherConfig& config);
void shared_expert_known_window_r2_launch(
    std::span<const ExecutionDispatchContext, 2> rows,
    const HipSharedExpertLauncherConfig& config);

// Closed retained-window R2/R4/R8 entry. R2 delegates to the unchanged typed
// entry above. R4/R8 share each Q8 decode across independent row accumulators;
// activation and F32 scalar gate retain the incumbent row-local kernels/order.
// Row strides are align_up(serial allocation bytes,256) in one physical scratch
// binding. All rows/weights/IO/positions and cross-row aliases are admitted
// before enqueue. Both callbacks require an owned eager scope for R4/R8.
// The window owner admits the actual session/allocation epoch; contexts expose
// its execution identity and allocation generation, not a separate epoch token.
// No allocation, host payload reads, capture, sync or publication. Retain every
// borrowed descriptor/config/owner and stream through quiescence after errors.
void shared_expert_known_window_rows_preflight(
    std::span<const ExecutionDispatchContext> rows,
    const HipSharedExpertLauncherConfig& config);
void shared_expert_known_window_rows_launch(
    std::span<const ExecutionDispatchContext> rows,
    const HipSharedExpertLauncherConfig& config);
// The same rows (R2/R4/R8, same admission, arithmetic and kernels), enqueued on `stream` (a hipStream_t) with the
// workspace rows at the arena's row stride in [workspace, workspace + workspace_bytes) instead of the
// arena. The caller orders `stream` after the step's input and before its readers, and owns the
// workspace and stream through quiescence (also after an exception).
// Cohort of several sessions' windows of one shared-expert step (2/4/8 rows each, four or eight in
// total). Every segment passes its own window admission; the gate/up and down projections run once
// for all rows (each row's arithmetic as in its own window), the activation/scalar kernel per
// segment. A nonzero workspace holds every segment's workspace rows in turn (detached use).
void shared_expert_known_window_cohort_launch(std::span<const std::span<const ExecutionDispatchContext>> segments,
    const HipSharedExpertLauncherConfig& config, void* stream, std::uintptr_t workspace, std::uint64_t workspace_bytes);
void shared_expert_known_window_rows_launch_detached(std::span<const ExecutionDispatchContext> rows,
    const HipSharedExpertLauncherConfig& config, void* stream, std::uintptr_t workspace, std::uint64_t workspace_bytes);
// Byte-identity fixture seam: force either producer, preserving all production admission checks. Optional device
// gate/up rows (2 * intermediate floats, compact) replace the projection result before activation on the owned stream.
// glu_pair: -1 = production STRIX_QPF2K_GLU_PAIR, 0 = activation pre-kernel, 1 / 2 = paired gate|up store (2: form 0).
// paired_form overrides only the admitted paired launch; route witnesses the producer, effective form and fallback.
enum class SharedPreProducer { complete_activation, pre_kernel, paired };
enum class SharedPreFallback { none, disabled, suppressed, short_rows, preimage_unavailable,
    pair_disabled, pair_site_disabled, pair_short_rows, pair_unsupported };
struct SharedPreFixtureRoute {
    SharedPreProducer producer = SharedPreProducer::complete_activation;
    SharedPreFallback fallback = SharedPreFallback::none;
    unsigned form = ~0U;  // Effective paired projection form; sentinel for other producers.
};
unsigned shared_expert_shared_pre_fixture_overflow_count();
bool shared_expert_shared_pre_fixture_launch(std::span<const ExecutionDispatchContext> rows,
    const HipSharedExpertLauncherConfig& config, bool shared_pre, const float* injected_gate_up = nullptr, int glu_pair = -1,
    unsigned paired_form = ~0U, SharedPreFixtureRoute* route = nullptr);
// The FP16 preimage rows the shared-pre producer left for `stream` (features halves per row, the pitch tail dropped).
std::vector<std::uint16_t> shared_expert_shared_pre_fixture_a16(void* stream, std::uint32_t rows, std::uint32_t features);
// Prefill: all T rows of a shared-expert step (8-row tiles of one strided view).
void shared_expert_known_prefill_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipSharedExpertLauncherConfig& config);
void shared_expert_known_prefill_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipSharedExpertLauncherConfig& config);
// Prefill, detached (STRIX_QPF2K_SCHED): the same T-row step on `stream`, its workspace rows at the arena's row stride in
// the caller's [workspace, workspace + workspace_bytes) (the arena may give the step's own workspace to the routed
// kernels that now run alongside it). Same kernels, launches and per-row arithmetic as the launch above; the first
// kernel keeps its barrier (it follows a cross-stream wait). The caller orders `stream` after the step's input and
// before its readers and owns stream and workspace through quiescence.
void shared_expert_known_prefill_rows_launch_detached(std::span<const ExecutionDispatchContext> rows,
    const HipSharedExpertLauncherConfig& config, void* stream, std::uintptr_t workspace, std::uint64_t workspace_bytes);

class HipSharedExpertLauncherBundle {
public:
    explicit HipSharedExpertLauncherBundle(
        HipSharedExpertLauncherConfig config)
        : config_(config), fallback_config_(config) {fallback_config_.prefetch_q8=false;fallback_config_.prefetch_eligible_layers.clear();}

    HipSharedExpertLauncherBundle(
        const HipSharedExpertLauncherBundle&) = delete;
    HipSharedExpertLauncherBundle& operator=(
        const HipSharedExpertLauncherBundle&) = delete;
    HipSharedExpertLauncherBundle(
        HipSharedExpertLauncherBundle&&) = delete;
    HipSharedExpertLauncherBundle& operator=(
        HipSharedExpertLauncherBundle&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

private:
    HipSharedExpertLauncherConfig config_;
    HipSharedExpertLauncherConfig fallback_config_;
};

}  // namespace strix
