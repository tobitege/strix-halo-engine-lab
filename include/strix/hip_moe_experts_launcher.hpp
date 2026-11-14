#pragma once

#include "strix/execution_binding.hpp"
#include "strix/hip_runtime_launchers.hpp"
#include "strix/launch_dispatch.hpp"

#include <cstdint>
#include <string>
#include <memory>
#include "strix/moe_q4k_a8_runtime.hpp"
#include "strix/moe_route_prune.hpp"
#include "strix/q51_known_rows.hpp"
#include <functional>
#include <span>

namespace strix {
class HipSetupQuiescenceGuard;

struct HipMoeExpertsLauncherConfig {
    std::uint32_t hidden_size = 0;
    std::uint32_t intermediate_size = 0;
    std::uint32_t expert_count = 0;
    std::uint32_t top_k = 0;
    HipActivationStorage activation =
        HipActivationStorage::bf16;
    bool q4k_a8g32_gate_up = false;
    bool batched_slot_activation = false;
    bool grouped_exact_q51_down = false;
    bool q51_slot_parallel_down = false;
    bool deferred_moe_status = false;
    std::shared_ptr<const HipSetupQuiescenceGuard> q4k_a8_quiescence;
    std::shared_ptr<moe_q4k_a8::RuntimeDispatchStats> q4k_a8_stats;
    // Explicit experimental arithmetic; defaults retain every original path.
    bool a8_down_q51 = false;
    bool a8_down_q80 = false;
    bool q5k_a8g32_gate_up = false;
    // fast-prefill-f16wmma-v1: known-prompt prefill blocks (T rows) run the routed MoE as FP16
    // WMMA (known_rows::enqueue_prefill form 3). Serial and window rows are unaffected.
    bool prefill_f16wmma = false;
    bool prefill_moe_route_bf16 = false;
    // RoutePrune (FAST, default off): the lowest-weight route assignments leave the prefill route plan (moe_route_prune.hpp).
    moe_core::RoutePrune prefill_moe_route_prune{};
};

void register_hip_moe_experts_kernel_variant(
    const ModelContract& model,
    const ExecutionBindingRequest& request,
    const HipMoeExpertsLauncherConfig& config,
    kernels::KernelRegistry& registry,
    kernels::QualificationStatus qualification =
        kernels::QualificationStatus::experimental);

[[nodiscard]] ExecutionLauncher
make_hip_moe_experts_launcher(
    std::string kernel_id,
    HipMoeExpertsLauncherConfig* config,
    bool perform_device_setup = true);

// Test-only (Q4K A8 runtime builds): the serial A8-down glue tail on caller buffers, either fused
// (STRIX_MOE_R1_GLUE_FUSE's one launch) or as the incumbent merge, finite scan, BF16 commit and deferred
// first-error publication. record points at the 8-byte DeferredMoeFirstError; stream is a hipStream_t.
void enqueue_q4k_a8_serial_tail_for_test(bool fused, const float* rawdots, const std::uint32_t* routes,
    const float* route_weights, std::uint32_t hidden_size, std::uint32_t experts, std::uint32_t top_k,
    float* accumulation, std::uint16_t* output, std::uint32_t* status, void* record, void* stream);

// Explicit retained known-prefix window R2/R4/R8 organization, not an R1
// variant or automatic strategy. Requires the selected Q4K A8 / Q5_1
// slot-parallel layer, BF16 C1 rows and owned eager D1 scope. Quantization,
// gate/up and SiLU retain their incumbent per-row kernels. Device CSR groups
// only selected assignments; down reuses each decoded coefficient across the
// expert's rows and scatters raw dots before the original ordered slot FMA.
// Every row and the exclusive CSR owner are admitted before any enqueue.
// The caller admits session/epoch, holds exclusive allocation leases, retains
// all descriptors/owners and drains the stream before reuse or destruction.
// No host payload reads, allocation, synchronization, capture or publication.
void moe_experts_known_window_rows_preflight(
    std::span<const ExecutionDispatchContext> rows,
    const HipMoeExpertsLauncherConfig& config,
    const q51_known_rows::Workspace& workspace);
// active_rows (0 = every row): rows from active_rows on are left out of the expert union and get a zero
// routed output (their MoE output is the shared expert only); the caller must never accept them.
void moe_experts_known_window_rows_launch(
    std::span<const ExecutionDispatchContext> rows,
    const HipMoeExpertsLauncherConfig& config,
    const q51_known_rows::Workspace& workspace,
    std::uint32_t active_rows = 0);
// The routed rows step with the MoE-combine rows step that follows it folded into the commit
// (STRIX_WINDOW_COMMIT_COMBINE=1): `combine_*` are the combine step's admitted views (shared, routed, output),
// `before_commit` runs on the host just before the commit is enqueued (the window joins the shared-expert
// stream there). Returns false, with nothing enqueued, when the views do not chain or alias.
bool moe_experts_known_window_rows_launch_combined(std::span<const ExecutionDispatchContext> rows,
    const HipMoeExpertsLauncherConfig& config, const q51_known_rows::Workspace& workspace,
    std::span<const void* const> combine_shared, std::span<const void* const> combine_routed,
    std::span<void* const> combine_output, const std::function<void()>& before_commit, std::uint32_t active_rows = 0);
// STRIX_WINDOW_COMMIT_COMBINE against the separate launches it replaces (known_moe_rows_commit_kernel, then the window
// combine_rows_kernel) on generated rows (2560 wide) of any bits: flavor 0 finite, 1 Inf / denormal / -0 / NaN payloads
// (quiet and signalling, both signs) in the accumulation, shared and earlier routed rows, 2 BF16 rounding ties, carries
// and overflow, 3 a row quantizer status, 4 a header route failure, 5 a row with status 8, 6 / 7 the combined row in place of
// the shared / routed row, 8 a NaN in every accumulation and shared element. Compares every byte of every row view,
// the CSR header and the D1 record. Returns {words, words differing}.
std::array<double, 2> known_moe_commit_combine_selftest(std::uint32_t rows, std::uint32_t flavor, std::uint32_t seed);
// Known-prompt prefill, A8 profile: all T rows (multiple of 8) of one routed step in one
// call. Every 8-row chunk passes the window admission above (with the window's own
// 8-row workspace); each per-row view must then be row 0 plus r times one stride.
// One CSR over the T rows in `prefill` (q51_known_rows::prefill_scratch_bytes(T)), so
// each touched expert's weights are read once per batch of its assignments. Each
// (row, slot) and each row's slot merge keep the window arithmetic and order.
void moe_experts_known_prefill_rows_preflight(
    std::span<const ExecutionDispatchContext> rows,
    const HipMoeExpertsLauncherConfig& config,
    const q51_known_rows::Workspace& window, const q51_known_rows::Workspace& prefill);
void moe_experts_known_prefill_rows_launch(
    std::span<const ExecutionDispatchContext> rows,
    const HipMoeExpertsLauncherConfig& config,
    const q51_known_rows::Workspace& window, const q51_known_rows::Workspace& prefill);

class HipMoeExpertsLauncherBundle {
public:
    explicit HipMoeExpertsLauncherBundle(
        HipMoeExpertsLauncherConfig config)
        : config_(config) {}

    HipMoeExpertsLauncherBundle(
        const HipMoeExpertsLauncherBundle&) = delete;
    HipMoeExpertsLauncherBundle& operator=(
        const HipMoeExpertsLauncherBundle&) = delete;
    HipMoeExpertsLauncherBundle(
        HipMoeExpertsLauncherBundle&&) = delete;
    HipMoeExpertsLauncherBundle& operator=(
        HipMoeExpertsLauncherBundle&&) = delete;

    void register_for_compiled_execution(
        const CompiledSerialExecution& compiled,
        ExecutionLauncherRegistry& registry,
        bool perform_device_setup = true);

private:
    HipMoeExpertsLauncherConfig config_;
};

}  // namespace strix
