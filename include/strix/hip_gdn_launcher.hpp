#pragma once

#include "strix/execution_binding.hpp"
#include "strix/gdn_recurrent_layout.hpp"
#include "strix/gdn_recurrent_storage.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/model_contract.hpp"
#include "strix/recurrent_state_type.hpp"

#include <cstdint>
#include <string>
#include <span>

namespace strix {

class HipGdnIslands;

struct HipGdnLauncherConfig {
    ModelContract model =
        ModelContract::qwen38_flash_next();
    HipActivationStorage activation =
        HipActivationStorage::bf16;
    bool enable_device_graph_inputs = false;
    bool virtual_q8_input_projections = false;
    bool prefetch_q8_input_projections = false;
    // Explicit eager geometry overlay; requires the two input options above.
    bool wave8_q8_input_projections = false;
    bool virtual_q8_output_projection = false;
    // Explicit eager FP32 output organization; requires virtual Q8 output.
    bool wave8_q8_output_projection = false;
    // Eager experimental LDS organization; persistent state layout is unchanged.
    HipGdnRecurrentLayout recurrent_layout = HipGdnRecurrentLayout::baseline;
    // Chosen with the session allocation/import; never converted per token.
    GdnRecurrentStorage recurrent_storage = GdnRecurrentStorage::key_value;
    // model.recurrent_state owns allocation, dispatch and identity together.
    // Dense A8 numerical profile (dense-a8-q80-v1): every Q8_0 projection of the
    // step (R1, windows and prompt rows) runs Q8_0 x per-32 int8 activations
    // through q8_rows::launch_dense_a8. Different numerics from the exact path.
    bool dense_a8_projections = false;
    // fast-prefill-f16wmma-v1: prompt launches of 32..256 rows run q8_rows::launch_dense_f16 (prefill only).
    bool dense_f16_prefill = false;
    // fast-prefill component dense8: prompt launches of 64..256 rows run q8_rows::launch_dense_g64 (prefill only).
    bool dense_g64_prefill = false;
    bool dense_trellis_projections = false;
};

// Closed eager R2 window helper. Each context remains an R1 row view. Both
// rows are deeply validated before any enqueue; recurrence updates row0 then
// row1 on the same state. No allocation, synchronization or session publication.
// Caller retains contexts, config, resources and owners through stream quiescence.
void gdn_known_window_r2_preflight(std::span<const ExecutionDispatchContext, 2> rows,
    const HipGdnLauncherConfig& config);
void gdn_known_window_r2_launch(std::span<const ExecutionDispatchContext, 2> rows,
    const HipGdnLauncherConfig& config);
// Closed mixed variant: paired Q8 qkv/z/out and incumbent F32 beta/alpha
// projections for each row. The same deep admission and lifetime rules apply.
void gdn_known_window_r2_f32_gates_preflight(std::span<const ExecutionDispatchContext, 2> rows,
    const HipGdnLauncherConfig& config);
void gdn_known_window_r2_f32_gates_launch(std::span<const ExecutionDispatchContext, 2> rows,
    const HipGdnLauncherConfig& config);

// Closed R4/R8 mixed-format windows. Three large Q8 projections share weights
// across every row; the small F32 gates and causal state transitions retain
// ascending R1 order. No padding rows, capture, storage conversion or commit.
// Known-prompt prefill: T rows (multiple of 8, >= 16) of one GDN layer at consecutive positions, committed in
// order (no deferral). Same per-row arithmetic as the window kernels; conv history and state written once.
void gdn_known_prefill_rows_preflight(std::span<const ExecutionDispatchContext> rows, const HipGdnLauncherConfig& config);
void gdn_known_prefill_rows_launch(std::span<const ExecutionDispatchContext> rows, const HipGdnLauncherConfig& config);
void gdn_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipGdnLauncherConfig& config);
void gdn_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipGdnLauncherConfig& config);

// Deferred-commit GDN window: one GDN layer, T = 1..8 consecutive rows.
// A window never commits its own rows. It first applies the `pending_rows`
// accepted rows that the previous window saved in slot `pending_slot` to the
// recurrent matrix and conv history (each read once, written once if any are
// pending), then runs its T rows from that state in registers, producing each
// row's output exactly as the serial R1 path would, and saves every row's
// commit inputs (pre-conv BF16 qkv, normalized k, v, beta, gate) in the other
// slot. The control word is device memory, so enqueues are graph-capturable.
// After acceptance the caller stores {n_keep, pending_slot ^ 1}; the standalone
// commit applies the pending rows and the caller then stores pending_rows = 0
// before any serial GDN token. Every GDN layer of a round shares one control.
inline constexpr std::uint32_t gdn_window_max_rows = 8;
// Sessions of one GDN cohort call (kCohortWideMaxSessions). The fused part 2 table holds gdn_window_max_rows segments
// (kernel argument budget), so a wider cohort launches it once per eight sessions; each block owns one session's state.
inline constexpr std::uint32_t gdn_window_cohort_max_segments = 16;
// Rows of one GDN cohort launch (the sessions of several 8-row chunks: 16 or 32).
inline constexpr std::uint32_t gdn_window_cohort_max_rows = 32;
struct GdnWindowControl { std::uint32_t pending_rows = 0, pending_slot = 0; };
struct GdnWindowLayer {
    // Q8_0 qkv/z/out; beta/alpha both F32 or both Q8_0; conv/dt/a/norm F32/F16/BF16 (ggml types).
    const void *qkv = nullptr, *z = nullptr, *beta = nullptr, *alpha = nullptr, *out = nullptr;
    const void *conv = nullptr, *dt = nullptr, *a = nullptr, *norm = nullptr;
    std::uint32_t gate_type = 0, conv_type = 0, dt_type = 0, a_type = 0, norm_type = 0;
    // qkv/z/out ggml types: Q8_0, or GSQ dense types under dense A8 (STRIX_GSQ_Q20_A8=1); gates may then be BF16.
    std::uint32_t qkv_type = 8, z_type = 8, out_type = 8;
    float* matrix = nullptr;             // config.recurrent_storage order
    std::uint16_t* conv_state = nullptr; // [conv channels][kernel] BF16 history
    void* retained = nullptr;            // gdn_window_retained_bytes(): [2 slots][8 rows]
};
// `workspace` uses the R1 temporary layout: qkv | z | beta | gate | core (FP32).
struct GdnWindowRow { const std::uint16_t* input = nullptr; std::uint16_t* output = nullptr; float* workspace = nullptr; };
[[nodiscard]] std::uint64_t gdn_window_retained_row_bytes(const ModelContract& model);
[[nodiscard]] std::uint64_t gdn_window_retained_bytes(const ModelContract& model);
// Requires wave8 Q8 input and output projections, BF16 activations, kernel 4.
// Any recurrent LDS layout (all share one reduction tree) and either storage.
void gdn_window_enqueue(const GdnWindowLayer& layer, std::span<const GdnWindowRow> rows,
    const GdnWindowControl* control, const HipGdnLauncherConfig& config, void* stream);
// Applies the pending rows to matrix + conv history; a no-op when none are pending.
void gdn_window_commit_enqueue(const GdnWindowLayer& layer, const GdnWindowControl* control,
    const HipGdnLauncherConfig& config, void* stream);
// Mode-table form: each context is one R1 GDN row of the same layer at
// consecutive positions (R1 admission per row). Layer L uses the retained
// block at GDN ordinal L - (L + 1) / full_attention_interval.
struct HipGdnWindowConfig {
    HipGdnLauncherConfig gdn;
    void* retained = nullptr; // gdn_layers x gdn_window_retained_bytes(model)
    std::uint64_t retained_bytes = 0;
    const GdnWindowControl* control = nullptr;
};
void gdn_window_rows_preflight(std::span<const ExecutionDispatchContext> rows, const HipGdnWindowConfig& config);
void gdn_window_rows_launch(std::span<const ExecutionDispatchContext> rows, const HipGdnWindowConfig& config);
// Commit for the layer of `row` (any R1 GDN context of that layer) on the current stream.
void gdn_window_commit_launch(const ExecutionDispatchContext& row, const HipGdnWindowConfig& config);
// Cohort of several sessions' windows of one layer (at most eight rows in total): each segment
// is one session's layer state, consecutive rows and control, admitted as by gdn_window_enqueue.
// The input projections, gates, gated norm and output projection run once for all rows (each
// row's arithmetic exactly as in its own window); prepare and recurrence run per segment.
struct GdnWindowCohortSegment {
    GdnWindowLayer layer;
    std::span<const GdnWindowRow> rows;
    const GdnWindowControl* control = nullptr;
};
void gdn_window_cohort_enqueue(std::span<const GdnWindowCohortSegment> segments, const HipGdnLauncherConfig& config,
    void* stream);
// Cohort form: the rows of several sessions (at most eight in total), each segment one session's
// consecutive rows with that session's own window configuration (retained rows and control). Every
// segment passes the complete single-session binding. The input projections, gates, gated norm and
// output projection run once for all rows (each row's arithmetic exactly as in its own window);
// prepare and recurrence run per segment from that session's state. Same layer and weights.
struct HipGdnWindowCohortSegment {
    std::span<const ExecutionDispatchContext> rows;
    const HipGdnWindowConfig* config = nullptr;
};
void gdn_window_cohort_rows_preflight(std::span<const HipGdnWindowCohortSegment> segments);
void gdn_window_cohort_rows_launch(std::span<const HipGdnWindowCohortSegment> segments);

#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
// Test-only incumbent R1 sequence (same kernels and dispatch as the serial
// launcher, without graph validation) on the window's pointer views.
void gdn_window_serial_row_probe(const GdnWindowLayer& layer, const GdnWindowRow& row,
    const HipGdnLauncherConfig& config, void* stream);
enum class GdnKnownWindowProjectionDirection { bf16_to_fp32, fp32_to_bf16 };
enum class GdnKnownWindowProjectionMode { incumbent_row_loop, rows2 };
enum class GdnKnownWindowFormat : std::uint32_t { q8_all5 = 2, q8_rows2_f32_gates = 3 };
// Test-only full input-stage payloads. Both modes perform complete typed row
// admission; neither updates the causal state, core workspace or final output.
void gdn_known_window_r2_input_stage_probe(std::span<const ExecutionDispatchContext, 2> rows,
    const HipGdnLauncherConfig& config, GdnKnownWindowFormat format,
    GdnKnownWindowProjectionMode mode);
struct GdnKnownWindowProjectionView { std::uintptr_t address = 0; std::uint64_t bytes = 0; };
enum class GdnRetainedWindowMode { serial_rows, retained_rows };
struct GdnRecurrentWindowRowView {
    GdnKnownWindowProjectionView qkv, beta, gate, core;
};
// Standalone recurrence experiment on preprocessed FP32 rows, not a decoder
// modality. Canonical key/value geometry 16/48/128, R2/R4, key-value matrix.
// Caller retains exclusive state/output owners and immutable input rows until
// completion of the explicit eager stream. No allocation, reset or commit.
void gdn_recurrent_window_probe(std::span<const GdnRecurrentWindowRowView> rows,
    GdnKnownWindowProjectionView state, std::uint32_t key_heads,
    std::uint32_t value_heads, std::uint32_t head_dim, GdnRetainedWindowMode, void* stream);
// Test-only projection payloads on independent row addresses (contiguous
// or strided). Same incumbent epilogues; no state mutation, allocation or sync.
void gdn_known_window_r2_projection_probe(GdnKnownWindowProjectionDirection,
    GdnKnownWindowProjectionMode, std::span<const GdnKnownWindowProjectionView, 2> input,
    GdnKnownWindowProjectionView weights, std::span<const GdnKnownWindowProjectionView, 2> output,
    std::uint32_t width, std::uint32_t outputs, void* stream);
// T = 1..8, 16, 32 or 64 rows: the q8_rows engine (exact or A8) or the R1 kernel per row.
enum class GdnKnownWindowRowsProbeMode { rows, rows_a8, row_loop };
void gdn_known_window_rows_projection_probe(GdnKnownWindowProjectionDirection,
    std::span<const GdnKnownWindowProjectionView> input, GdnKnownWindowProjectionView weights,
    std::span<const GdnKnownWindowProjectionView> output,
    std::uint32_t width, std::uint32_t outputs, void* stream,
    GdnKnownWindowRowsProbeMode mode = GdnKnownWindowRowsProbeMode::rows);
// T = 1..8 rows of the dense A8 sites with GSQ-RCO typed segments (ggml types, 8 = Q8_0) through the engine's own
// site types and q8_rows::launch: bf16_to_fp32 = the input site (1..4 segments, output[m * rows + r]),
// fp32_to_bf16 = the output site (one segment, output[r]).
void gdn_gsq_rows_projection_probe(GdnKnownWindowProjectionDirection, std::span<const std::uint32_t> types,
    std::span<const GdnKnownWindowProjectionView> weights, std::span<const std::uint32_t> counts,
    std::span<const GdnKnownWindowProjectionView> input, std::span<const GdnKnownWindowProjectionView> output,
    std::uint32_t width, void* stream);
#endif

void register_hip_gdn_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipGdnLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_gdn_launcher(
    std::string kernel_id,
    HipGdnLauncherConfig* config,
    bool reserve_device_scratch = true);

class HipGdnLauncherBundle {
public:
    explicit HipGdnLauncherBundle(
        HipGdnLauncherConfig config)
        : config_(config) {}

    HipGdnLauncherBundle(
        const HipGdnLauncherBundle&) = delete;
    HipGdnLauncherBundle& operator=(
        const HipGdnLauncherBundle&) = delete;
    HipGdnLauncherBundle(
        HipGdnLauncherBundle&&) = delete;
    HipGdnLauncherBundle& operator=(
        HipGdnLauncherBundle&&) = delete;

    // Caller retains islands until all borrowing plans/dispatchers and their work are quiescent.
    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry, HipGdnIslands* islands = nullptr,
        bool reserve_device_scratch = true);

private:
    HipGdnLauncherConfig config_;
};

}  // namespace strix
