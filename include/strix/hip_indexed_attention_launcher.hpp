#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"
#include "strix/model_contract.hpp"
#include "strix/graph_attention_state.hpp"
#include "strix/qsa_input_prefetch.hpp"
#include "strix/qsa_selection_plan.hpp"
#include "strix/session_state_layout.hpp"
#include "strix/session_state_memory.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strix {

// STRIX_BATCH_SERVICE_MS: diagnostic closure certificate after a complete launcher returns.
[[nodiscard]] bool qsa_prefill_bundle_closed() noexcept;
// Reserve the stage ring for the process maximum T/context before any submission.
void reserve_indexed_attention_prefill_pools(std::uint32_t rows, std::uint64_t context_tokens,
    std::uint32_t ratio, bool compact_scores = false);

class HipQsaWindowPool;
class HipQsaExactScoreOwner;
// Allocate the complete admitted ring during setup, before prompt execution.
[[nodiscard]] std::shared_ptr<HipQsaExactScoreOwner> make_hip_qsa_exact_score_owner(
    std::uint64_t maximum, std::uint64_t admitted_bytes, void* stream, std::uint64_t generation, std::uint32_t rows);
struct MtpQsaPoolPlan;
class HipQsaCommittedPool;
struct HipQsaKeyDescriptor {
    const std::uint16_t* packed=nullptr;
    std::uint64_t epoch=0;
    std::uint32_t role=0, dim=128, complete_capacity=0;
    void* producer_event=nullptr;
};
// K1 consumes these views while the access lease is retained. Selection stays legacy in M1.
struct HipQsaPoolRowDescriptor {
    const float* query=nullptr; float* scores=nullptr; std::uint32_t* selected=nullptr;
    std::uint32_t visible=0, complete_blocks=0, topk=0, maxima_stride=0;
    std::uint32_t* status=nullptr;
};
class HipQsaCommittedAccess {
public:
    struct Impl;
    explicit HipQsaCommittedAccess(std::unique_ptr<Impl>);
    ~HipQsaCommittedAccess();
    HipQsaCommittedAccess(const HipQsaCommittedAccess&)=delete;
    HipQsaCommittedAccess& operator=(const HipQsaCommittedAccess&)=delete;
    [[nodiscard]] void* storage() const noexcept;
    // Records the current producer fence. K1 queues consumers on this lease's stream and retains the lease through enqueue.
    [[nodiscard]] HipQsaKeyDescriptor key() const;
    [[nodiscard]] float* scratch() const noexcept;
private:
    std::unique_ptr<Impl> impl_;
};
class HipQsaCommittedPool : public std::enable_shared_from_this<HipQsaCommittedPool> {
public:
    struct Impl;
    static std::shared_ptr<HipQsaCommittedPool> create(const MtpQsaPoolPlan&);
    ~HipQsaCommittedPool();
    [[nodiscard]] std::unique_ptr<HipQsaCommittedAccess> before_rewrite(bool predictor,
        std::uintptr_t raw,std::uint64_t raw_bytes,std::uint64_t raw_generation,
        const RuntimeTensorBinding& norm,std::uint32_t norm_type,const ModelContract&,
        std::uint64_t first,std::uint32_t rows,void* stream,bool certified_prefix);
    // scratch lease serializes K1 only, not unrelated key entries; no K1 dispatch here.
    [[nodiscard]] std::unique_ptr<HipQsaCommittedAccess> selector_lease(void* stream);
    void bind_domain(bool predictor,const ModelContentIdentity& content,std::uint64_t profile);
    void commit(bool predictor,std::uint64_t end);
    void invalidate(bool predictor,std::uint64_t first);
    void begin_epoch();
    void cancel() noexcept;
    void checked(std::uint64_t scores,std::uint64_t differing,std::uint32_t role=13U);
    void print_stats() const;
    [[nodiscard]] std::uint64_t reserved_bytes() const noexcept;
private:
    HipQsaCommittedPool();
    std::unique_ptr<Impl> impl_;
};
// One eager window owner's cache. The owner must invalidate it before replacing
// its session's committed prefix (discard_deferred does this), and retain it
// through stream drain. Rejected/window rows are invalidated from the next frontier.
[[nodiscard]] std::shared_ptr<HipQsaWindowPool> make_hip_qsa_window_pool();
// Serial decode pool (STRIX_QSA_DECODE_POOL): one entry per QSA layer, allocated per entry; a failed allocation leaves
// that layer on block_score_kernel. bf16_keys selects packed-only upper halves; the CLI supplies the admitted policy.
[[nodiscard]] std::shared_ptr<HipQsaWindowPool> make_hip_qsa_decode_pool(bool bf16_keys = false);
void invalidate_hip_qsa_window_pool(const std::shared_ptr<HipQsaWindowPool>&);

struct HipIndexedAttentionLauncherConfig {
    ModelContract model =
        ModelContract::qwen38_flash_next();
    HipActivationStorage activation =
        HipActivationStorage::bf16;
    bool enable_device_graph_inputs = false;
    bool prefetch_q8_input = false;
    bool wave8_q8_output = false;
    std::vector<std::uint32_t> prefetch_eligible_layers;
    std::shared_ptr<qsa_input_prefetch::RuntimeDispatchStats> prefetch_stats;
    bool qsa_hierarchical_selection = false;
    std::uint32_t qsa_gqa_heads_per_block = 1;
    std::shared_ptr<QsaHierarchicalSelectionStats> selection_stats;
    std::shared_ptr<QsaHierarchicalSelectionStats> gqa_stats;
    // Dense A8 numerical profile (dense-a8-q80-v1): every Q8_0 projection of the
    // step (R1, windows and prompt rows) runs Q8_0 x per-32 int8 activations
    // through q8_rows::launch_dense_a8. Different numerics from the exact path.
    bool dense_a8_projections = false;
    // fast-prefill-f16wmma-v1: prompt launches of 32..256 rows run q8_rows::launch_dense_f16 (prefill only).
    bool dense_f16_prefill = false;
    // fast-prefill component dense8: prompt launches of 64..256 rows run q8_rows::launch_dense_g64 (prefill only).
    bool dense_g64_prefill = false;
    bool dense_trellis_projections = false;
    // fast-prefill-f16wmma-v1, qsa component: prefill groups run the fused union attention (hip_qsa_fast_prefill.inc).
    bool qsa_fast_prefill = false;
    std::shared_ptr<HipQsaWindowPool> window_pool;
    std::shared_ptr<HipQsaCommittedPool> committed_pool;
    // Eager serial decode of an append-only session (the plain CLI route; set through HipLauncherCatalogConfig::
    // qsa_decode_pool): complete blocks are pooled once and scored by block_score_pooled_rows_kernel, bitwise the
    // scores of block_score_kernel. The owner must not let another writer rewrite raw index keys below the position of a
    // later serial step (no windows, rollback, restore or slot reuse on this config).
    std::shared_ptr<HipQsaWindowPool> decode_pool;
    std::shared_ptr<HipQsaExactScoreOwner> exact_score_owner;
};

enum class HipQsaInputProjectionMode {legacy256,virtual256,prefetch8};
struct HipQsaInputProjectionView {std::uintptr_t address=0;std::uint64_t bytes=0;};
// Standalone diagnostic, no allocation/sync. Caller retains all owners through
// stream drain (also on enqueue failure). Exact disjoint output views, BF16 input,
// original Q8_0 weights and BF16-rounded values represented in FP32 output.
void launch_hip_qsa_input_projection_probe(HipQsaInputProjectionMode,
    HipQsaInputProjectionView input,HipQsaInputProjectionView weights,
    HipQsaInputProjectionView output,std::uint32_t width,std::uint32_t outputs,void* stream);

#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
// Test-only QSA top-k block selection over caller-owned device views; no
// allocation/sync. reference = the original single-thread loop, one launch per
// row; otherwise the production kernels: one row = the C1 kernel (with a
// device position: its graph-input form, sizes read on the device), 2/4/8 rows
// = the known-window rows kernel. Without a device position a row with
// visible/ratio <= block_topk takes the full-visible fill. Scores are consumed.
void launch_hip_qsa_committed_pool_probe(bool pooled,bool predictor,const std::shared_ptr<HipQsaCommittedPool>&,
    const ModelContract&,std::uintptr_t raw,std::uint64_t capacity,const RuntimeTensorBinding& norm,
    std::uint64_t first,std::uint32_t writer_rows,std::span<const HipQsaPoolRowDescriptor>,void* stream);
struct HipQsaBlockSelectRow { float* scores = nullptr; std::uint32_t* selected = nullptr; std::uint32_t visible = 0; };
void launch_hip_qsa_block_select_probe(bool reference, std::span<const HipQsaBlockSelectRow> rows, std::uint32_t ratio,
    std::uint32_t block_topk, const std::uint64_t* device_position, std::uint64_t context_tokens, std::uint32_t* device_status,
    void* stream);
#endif

#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
// Test-only kv-int8-tok-v1 writers and readers over caller-owned device memory; no
// allocation/sync. Each of the 8 rows appends k[r] / v[r] (kv_heads x head_dim FP32) at
// position[r]: the decode writer into decode_state, the 8-row window writer into rows_state
// (int8 rows of 2 kv_width + 8 kv_heads bytes). The FP32 query q_gate (attention_heads x 2
// head_dim) then attends the `count` tokens of `selected` in decode_state through the decode
// score/softmax/value kernels (scores_decode, out_decode), every row of the 8-row window
// kernels (scores_rows, out_rows: 8 x) and of the 16-row prefill group kernels (scores_group,
// out_group: 16 x). Scores are heads x maximum_selected, outputs attention_heads x head_dim.
struct HipKvInt8Probe {
    std::uint32_t attention_heads = 24, kv_heads = 2, head_dim = 256, count = 0, maximum_selected = 0;
    const float* k[8]{};
    const float* v[8]{};
    std::uint64_t position[8]{};
    void* decode_state = nullptr;
    void* rows_state = nullptr;
    const float* q_gate = nullptr;
    const std::uint32_t* selected = nullptr;
    float* scores_decode = nullptr;
    float* out_decode = nullptr;
    float* scores_rows = nullptr;
    float* out_rows = nullptr;
    float* scores_group = nullptr;
    float* out_group = nullptr;
};
void launch_hip_attention_kv_int8_probe(const HipKvInt8Probe& probe, void* stream);
#endif

#ifdef STRIX_BUILD_QSA_OUTPUT_PROJECTION_DIAGNOSTIC
enum class HipQsaOutputProjectionMode { legacy256, wave8 };
// Test-only explicit-stream entry. FP32 input and original Q8_0 weights produce
// BF16 output. No allocation/synchronization; retain owners through completion.
// Diagnostic capture is supported to inspect geometry, independently of the
// eager-only production launcher's pre-enqueue capture rejection.
void launch_hip_qsa_output_projection_probe(HipQsaOutputProjectionMode,
    HipQsaInputProjectionView input, HipQsaInputProjectionView weights,
    HipQsaInputProjectionView output, std::uint32_t width, std::uint32_t outputs, void* stream);
#endif
#ifdef STRIX_BUILD_KNOWN_WINDOW_DIAGNOSTIC
// Test-only T = 1/2/4/8/16/32/64 rows of the prefetch8 input (BF16 -> BF16-rounded FP32)
// or wave8 output (FP32 -> BF16) Q8_0 projection: the q8_rows engine (exact or
// A8) or the C1 kernel once per row. Same ownership rules as the probes above.
enum class HipQsaRowsProjection { input, output };
enum class HipQsaRowsProbeMode { rows, rows_a8, row_loop };
void launch_hip_qsa_rows_projection_probe(HipQsaRowsProjection, HipQsaRowsProbeMode,
    std::span<const HipQsaInputProjectionView> input, HipQsaInputProjectionView weights,
    std::span<const HipQsaInputProjectionView> output, std::uint32_t width, std::uint32_t outputs, void* stream);
// T = 1/2/4/8 rows of the dense A8 sites with GSQ-RCO typed segments (ggml types, 8 = Q8_0) through the engine's own
// site types and q8_rows::launch: input = BF16 rows -> BF16-rounded FP32 (1..5 segments, output[m * rows + r]),
// output = FP32 rows -> BF16 (one segment, output[r]). Input side, 16 / 32 rows of one typed (non-Q8) kind: the cohort
// window dispatcher (launch_qsa_rows_input with typed_rows8 = true).
void launch_hip_qsa_gsq_rows_probe(HipQsaRowsProjection, std::span<const std::uint32_t> types,
    std::span<const HipQsaInputProjectionView> weights, std::span<const std::uint32_t> counts,
    std::span<const HipQsaInputProjectionView> input, std::span<const HipQsaInputProjectionView> output,
    std::uint32_t width, void* stream);
#endif

// Experimental post-projection C1 norm/RoPE/KV stage. Async on caller's
// explicitly owned stream; retain all resource owners through completion.
// Read status only after completion: 0=stage valid, 2=position out of bounds,
// 4=nonfinite source/norm, 8=nonfinite arithmetic. A nonzero status invalidates
// the stage. This does not publish a serial token/state commit.
void launch_hip_graph_attention_state(
    const GraphAttentionStatePlan& plan, std::uint64_t scalar_control_position,
    void* stream);

void register_hip_indexed_attention_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipIndexedAttentionLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_indexed_attention_launcher(
    std::string kernel_id,
    HipIndexedAttentionLauncherConfig* config);

// Optional device journal of raw indexer keys: per absolute layer,
// (compress_ratio-1) partial-block tail cells, then up to eight window row keys.
struct HipQsaIndexJournal { std::uintptr_t address = 0; std::uint64_t bytes = 0; };
[[nodiscard]] std::uint64_t qsa_index_journal_bytes(const ModelContract&);

// Known-prefix window: T = 2/4/8 rows of one QSA step at consecutive positions
// p..p+T-1, eager only. Q/K/V/indexer and output weights are read once for all
// rows; every row's raw-index key and K/V are appended before any row attends,
// and row t sees exactly cells <= p+t. Each row's output, workspace and state
// bytes equal the serial launcher's for this config (the row's serial kernel id
// is checked). A bound journal receives the T raw keys for the tail commit.
void indexed_attention_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipIndexedAttentionLauncherConfig&, HipQsaIndexJournal journal = {});
void indexed_attention_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipIndexedAttentionLauncherConfig&, HipQsaIndexJournal journal = {});
// Cohort form: the rows of several sessions (four or eight in total), each segment 2/4/8 consecutive
// rows of one session with its own KV and raw-index state. Every segment passes the complete
// single-session admission; the input and output projections run once for all rows (each row's
// arithmetic exactly as in its own window), store/selection/attention per segment. No journal.
struct HipQsaCohortSegment {
    std::span<const ExecutionDispatchContext> rows;
    const HipIndexedAttentionLauncherConfig* config = nullptr;
};
void indexed_attention_known_window_cohort_preflight(std::span<const HipQsaCohortSegment> segments);
void indexed_attention_known_window_cohort_launch(std::span<const HipQsaCohortSegment> segments);
// Prefill: all T rows of a QSA step (8-row tiles; projections 64 rows per launch).
void indexed_attention_known_prefill_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipIndexedAttentionLauncherConfig&, HipQsaIndexJournal journal = {});
void indexed_attention_known_prefill_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipIndexedAttentionLauncherConfig&, HipQsaIndexJournal journal = {});

// Indexer partial-block tail around a window. Snapshot (before the window)
// copies each QSA layer's raw cells [p - p%ratio, p) into the journal; commit
// restores them and re-appends the journal keys of the first n_keep rows at
// p..p+n_keep-1. p and n_keep are device words (graph-capturable, no sync).
// Cells past the accepted prefix stay stale: they are rewritten before any read.
struct HipQsaIndexTailPlan {
    HipQsaIndexJournal journal;
    std::vector<std::uintptr_t> raw_index;
    std::vector<std::uint32_t> layers;
    std::uint64_t context_tokens = 0;
    std::uint32_t index_dim = 0, ratio = 0;
};
[[nodiscard]] HipQsaIndexTailPlan prepare_qsa_index_tail(const ModelContract&, const SerialSessionStateLayout&,
    const SessionStateMemoryBinding&, HipQsaIndexJournal);
void enqueue_qsa_index_tail_snapshot(const HipQsaIndexTailPlan&, const std::uint64_t* device_position, void* stream);
void enqueue_qsa_index_tail_commit(const HipQsaIndexTailPlan&, const std::uint64_t* device_position,
    const std::uint32_t* device_n_keep, void* stream);

class HipIndexedAttentionLauncherBundle {
public:
    explicit HipIndexedAttentionLauncherBundle(
        HipIndexedAttentionLauncherConfig config)
        : config_(config), fallback_config_(config) {fallback_config_.prefetch_q8_input=false;fallback_config_.prefetch_eligible_layers.clear();}

    HipIndexedAttentionLauncherBundle(
        const HipIndexedAttentionLauncherBundle&) = delete;
    HipIndexedAttentionLauncherBundle& operator=(
        const HipIndexedAttentionLauncherBundle&) = delete;
    HipIndexedAttentionLauncherBundle(
        HipIndexedAttentionLauncherBundle&&) = delete;
    HipIndexedAttentionLauncherBundle& operator=(
        HipIndexedAttentionLauncherBundle&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

    // Admit only this bundle's exact current eager QSA launchers. The bundle
    // and its configurations must outlive use of the returned identities.
    [[nodiscard]] std::vector<std::uint64_t> known_window_position_readers(
        const CompiledSerialExecution& compiled,
        const ExecutionLauncherRegistry& registry) const;

private:
    HipIndexedAttentionLauncherConfig config_;
    HipIndexedAttentionLauncherConfig fallback_config_;
};

// Requires an exclusive, drained single-request prompt handoff. Shared decode
// pools, state, rotary tables and diagnostics are retained.
void release_hip_qsa_prefill_storage();

// Call before destroying a stream that may have carried QSA prefill submissions. The shared handoff event of
// the last submission is recorded on its stream; this completes it while the stream exists, so that neither a
// later submission on another stream nor the end of the process waits on an event whose stream is gone.
// No effect for any other stream.
void complete_hip_qsa_prefill_handoff(void* stream);

}  // namespace strix
