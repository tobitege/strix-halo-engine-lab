#pragma once

// deepseek41 CUDA parts-host kernels (sm_120): the routed experts of a layer when only a subset of them is resident
// on this device and the others are computed by expert servers (strix/ds41-expert-rpc-v1, ds41_expert_rpc.hpp).
// Conventions: cuda_ds41_common.hpp (enqueue_* only enqueue).
//
//  - Resident routes: the window routes of enqueue_moe_routes restricted to the resident experts, with experts[] the
//    expert's physical slot in a compact resident buffer (slot_of[id]), so the routed gate/up and down GEMVs compute
//    any resident subset with ExpertWeights pointing at the compact buffer. order / low_counts cover every slot of the
//    window (canonical ids). The raw vector of a resident slot has the bits of the all-resident path.
//  - Seeded combine: y = (shared + low) + high where per (row, half) the fold starts from a seed vector (a FOLD item
//    of an answer: the fold of the half's leading remote slots) or 0, skips the first `start` slots of the half, and
//    continues over the later slots in ascending id with the local raw vector or an answer's RAW vector; each step
//    acc = acc + w * v (one FP32 product, one FP32 add). This is xrpc::combine_host bit for bit, and
//    enqueue_moe_combine when every slot is local (no seed, start 0, every source local).

#include "strix/cuda/ds41_moe.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::ds41::cuda {

// Routes of ids[row * top_k + k] (distinct within a row, < experts <= 1024, top_k <= 16) over the experts with
// slot_of[id] >= 0 (device table [experts]); experts[u] = slot_of[union expert u] (unions in ascending id); entries
// in ascending row; order / low_counts for every slot. One block (any window size; meant for decode windows).
// Sizes of `out`: as MoeRoutes.
struct ResidentRoutesView {
    const std::uint32_t* ids = nullptr;
    const std::int32_t* slot_of = nullptr;
    std::uint32_t rows = 0, top_k = 0, experts = 0, split = 0;
    MoeRoutes out;
    // Guarded mode (an external slot table, e.g. tierstream's device-gated view; abort != nullptr): every routed id gets a
    // slot, the guard slot when *abort is set or its entry is < 0 (then *absent_err |= 1). Off: entries < 0 are skipped.
    const std::uint32_t* abort = nullptr;
    std::int32_t guard_slot = -1;
    std::uint32_t* absent_err = nullptr;
};
void enqueue_resident_routes(const ResidentRoutesView& view, cudaStream_t stream);
// The 1024-thread block form only (windows of more than 32 slots always take it; the reference of the one-warp form).
void enqueue_resident_routes_block(const ResidentRoutesView& view, cudaStream_t stream);

// Combine tables of one window, packed as int32 (one upload per layer, or read in place from mapped host memory):
// seed[rows * 2] (-1 or the seed vector: answer << 24 | item), start[rows * 2] (leading slots of the half covered by the
// seed), source[rows * top_k] (-1 the local raw vector, else the slot's RAW vector: answer << 24 | item). Index
// (row * 2 + half), half 0 = low. Vector (a, i) is remote[a] + i * hidden (xrpc::CombinePlan's encoding).
inline constexpr std::uint32_t kMaxAnswers = 8;
[[nodiscard]] constexpr std::size_t seeded_combine_table_ints(std::uint32_t rows, std::uint32_t top_k) noexcept {
    return std::size_t{rows} * (4U + top_k);
}
struct SeededCombineView {
    const float* raw = nullptr;      // [rows * top_k][hidden], local slots' raw vectors (window slot index)
    const float* remote[kMaxAnswers] = {};  // per answer its vectors [items][hidden] (device or mapped host memory)
    const float* shared = nullptr;   // [rows][hidden]
    const float* weights = nullptr;  // [rows * top_k] router weights
    const std::uint32_t* order = nullptr;       // [rows * top_k] slots in ascending id
    const std::uint32_t* low_counts = nullptr;  // [rows]
    const std::int32_t* tables = nullptr;       // packed: seeded_combine_table_ints(rows, top_k)
    std::uint32_t rows = 0, top_k = 0, hidden = 0;
    float* out = nullptr;            // [rows][hidden]
};
void enqueue_seeded_combine(const SeededCombineView& view, cudaStream_t stream);

// Copies a window's router output and MoE input to mapped host memory (device-accessible host pointers), then
// publishes `sequence` in *flag (system-scope fence first): the host spins on the flag instead of an event and a
// copy. ids / weights [rows * top_k], codes [rows * hidden], deltas [rows * hidden / 32]. One block.
struct ExportWindowView {
    const std::uint32_t* ids = nullptr;
    const float* weights = nullptr;
    const std::int8_t* codes = nullptr;
    const float* deltas = nullptr;
    std::uint32_t rows = 0, top_k = 0, hidden = 0;
    std::uint32_t* host_ids = nullptr;
    float* host_weights = nullptr;
    std::int8_t* host_codes = nullptr;
    float* host_deltas = nullptr;
    volatile std::uint32_t* flag = nullptr;
    std::uint32_t sequence = 0;
};
void enqueue_export_window(const ExportWindowView& view, cudaStream_t stream);

}  // namespace strix::ds41::cuda
