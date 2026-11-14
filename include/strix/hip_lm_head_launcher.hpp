#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"

#include <cstdint>
#include <span>
#include <string>

namespace strix {
struct HipGreedyRowsReply;

struct HipLmHeadLauncherConfig {
    std::uint32_t hidden_size = 0;
    std::uint32_t vocab_size = 0;
    HipActivationStorage input =
        HipActivationStorage::bf16;
    bool virtual_q8_wave = false;
    // Explicit eager BF16/Q8_0 candidates; wave8 also requires prefetch.
    bool prefetch_q8_wave = false;
    bool wave8_q8_wave = false;
    // Known-window rows kernel: 0 = staged rows kernel, 1/2 = lean rows kernel with 1/2 logits
    // per wave (same per-logit arithmetic and trees; STRIX_LM_HEAD_ROWS_LEAN).
    std::uint32_t rows_lean = 0;
};

void register_hip_lm_head_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipLmHeadLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_lm_head_launcher(
    std::string kernel_id,
    HipLmHeadLauncherConfig* config);

class HipLmHeadLauncherBundle {
public:
    explicit HipLmHeadLauncherBundle(
        HipLmHeadLauncherConfig config)
        : config_(config) {}

    HipLmHeadLauncherBundle(
        const HipLmHeadLauncherBundle&) = delete;
    HipLmHeadLauncherBundle& operator=(
        const HipLmHeadLauncherBundle&) = delete;
    HipLmHeadLauncherBundle(
        HipLmHeadLauncherBundle&&) = delete;
    HipLmHeadLauncherBundle& operator=(
        HipLmHeadLauncherBundle&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry);

private:
    HipLmHeadLauncherConfig config_;
};

// Known-window terminal head for T=2..8 rows. Each context is the row's own R1
// head dispatch; the configured eager prefetch BF16/Q8_0 head reads every weight
// block once and keeps each row's R1 leaf/tree arithmetic, so every row's logits
// equal its R1 launch bitwise. All rows share one head tensor; outputs are disjoint.
inline constexpr std::uint32_t lm_head_window_max_rows = 8;
// Cohort windows of several sessions: 16 or 32 rows (lm_head_known_window_rows_launch only).
inline constexpr std::uint32_t lm_head_cohort_max_rows = 32;
void lm_head_known_window_rows_preflight(std::span<const ExecutionDispatchContext> rows,
    const HipLmHeadLauncherConfig& config);
void lm_head_known_window_rows_launch(std::span<const ExecutionDispatchContext> rows,
    const HipLmHeadLauncherConfig& config);
struct HipLmHeadGreedyRows {
    std::span<const ExecutionDispatchContext> rows;
    ExecutionBufferAddress partials;
    HipGreedyRowsReply* replies;
};
// Launch the head once, then each greedy segment in order (including null replies).
// The fallback's window-size check deliberately follows the head launch.
void lm_head_rows_with_greedy(std::span<const ExecutionDispatchContext> rows,
    const HipLmHeadLauncherConfig& config, std::span<const HipLmHeadGreedyRows> greedy,
    bool window_rows_only = false);
// The same launch followed by the window greedy selection of every row: replies as from
// enqueue_hip_greedy_rows over the written logits. When the lean rows kernel applies
// (config.rows_lean, 5-8 rows) its greedy partials are computed in the head kernel and only the
// finish launch follows; otherwise this is the rows launch plus enqueue_hip_greedy_rows.
void lm_head_known_window_rows_launch_greedy(std::span<const ExecutionDispatchContext> rows,
    const HipLmHeadLauncherConfig& config, ExecutionBufferAddress partials, HipGreedyRowsReply* replies);

// Source-only native MTP entry points; qualification is separate.
struct MtpNativeContext;
struct MtpTensorView;
enum class HipMtpLmHeadRecipe : std::uint32_t {
    generic256=0,
    prefetch8_wave8_q8=1,
};
// Pure host recipe admission: no HIP calls or qualification. Unsupported fast
// geometry/format uses the original generic native MTP head. Actual launches
// still require the full retained-owner context and dynamic buffer checks.
[[nodiscard]] HipMtpLmHeadRecipe select_hip_mtp_lm_head_recipe(const ModelContract&,const MtpTensorView&);
[[nodiscard]] std::uint64_t hip_mtp_lm_head_profile_identity(const ModelContract&,const MtpTensorView&);
void preflight_hip_mtp_lm_head(const MtpNativeContext&);
void launch_hip_mtp_lm_head(const MtpNativeContext&,void* stream);
// Raw MTP draft head over Q8_0 rows [rows][hidden]: the prefetch8 wave8 recipe
// above, so each row's logit is bit-identical to the same row of the full head.
void launch_hip_q8_head_rows(const std::uint16_t* hidden,const void* q8_rows,std::uint32_t hidden_size,
    std::uint32_t rows,float* logits,void* stream);

}  // namespace strix
