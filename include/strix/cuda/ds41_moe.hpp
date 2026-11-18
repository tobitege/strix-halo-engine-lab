#pragma once

// deepseek41 CUDA backend, routed experts (sm_120): window routes built on the device, the fused gate/up SwiGLU
// (IQ2_XXS), the down projection (Q2_K) and the combine, as decode GEMVs (1 to 8 entries of an expert per pass) and
// as a grouped GEMM over many rows (prefill, integer MMA). Conventions: cuda_ds41_common.hpp.
//
// Every result is the host reference's bit for bit (ds41_moe.hpp / ds41_formats.hpp, docs/models/deepseek-v4.1/cuda-numeric-identity.md):
//  - each 32-weight group's integer dot of every (entry, output) is exact in int32 (DP4A or IMMA);
//  - its FP32 term is the declared one: IQ2_XXS float(dot) * (scale * delta), scale = (d * (0.5 + s)) * 0.25;
//    Q2_K (d * float(A) - dmin * float(B)) * delta;
//  - lane class l (groups l, l + 32, ...) is accumulated in ascending order from 0, and the 32 class sums are
//    combined in the butterfly's tree v[l] += v[l ^ d], d = 16 .. 1;
//  - the gate/up epilogue is swiglu_clamped (declared_exp) then quantize_a8 per 32 outputs;
//  - the combine is y = (shared + low) + high over the row's slots in ascending expert id, low = ids < split.

#include "strix/cuda/ds41_common.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::ds41::cuda {

// Geometry of the routed experts of one layer. hidden and inter are multiples of 256.
struct MoeShape {
    std::uint32_t hidden = 5120;
    std::uint32_t inter = 2304;
    std::uint32_t experts = 384;
    std::uint32_t top_k = 6;
    float limit = 10.0F;  // swiglu_limit
    [[nodiscard]] std::uint64_t gate_row_bytes() const noexcept { return std::uint64_t{hidden} / 256U * 66U; }  // IQ2_XXS
    [[nodiscard]] std::uint64_t down_row_bytes() const noexcept { return std::uint64_t{inter} / 256U * 84U; }   // Q2_K
    [[nodiscard]] std::uint64_t gate_expert_bytes() const noexcept { return gate_row_bytes() * inter; }
    [[nodiscard]] std::uint64_t down_expert_bytes() const noexcept { return down_row_bytes() * hidden; }
    [[nodiscard]] std::uint64_t expert_bytes() const noexcept { return 2U * gate_expert_bytes() + down_expert_bytes(); }
};

// The three [experts] tensors of a layer with their GGUF bytes (gate and up [experts][inter][hidden] IQ2_XXS, down
// [experts][hidden][inter] Q2_K). Union expert index e reads the tensors at e * expert bytes; a device that keeps
// only a slice of the experts passes the slice's base pointers and builds routes with base = the slice's first id.
struct ExpertWeights {
    const std::uint8_t* gate = nullptr;
    const std::uint8_t* up = nullptr;
    const std::uint8_t* down = nullptr;
};

// Window routes in device memory (build_window_routes / restrict_routes of ds41_moe.hpp): the union experts in
// ascending id; entry i of union slot j (begins[j] <= i < begins[j + 1]) reads activation row rows[i] and writes slot
// slots[i] = row * top_k + k; the entries of an expert in ascending row. order[row * top_k + k] are the row's slots
// in ascending expert id, low_counts[row] how many have id < split. counts[0] = union experts, counts[1] = most
// entries of one expert. Sizes: rows * top_k for experts / rows / slots / order, + 1 for begins; rows for
// low_counts; 2 for counts.
struct MoeRoutes {
    std::uint32_t* experts = nullptr;
    std::uint32_t* begins = nullptr;
    std::uint32_t* rows = nullptr;
    std::uint32_t* slots = nullptr;
    std::uint32_t* order = nullptr;
    std::uint32_t* low_counts = nullptr;
    std::uint32_t* counts = nullptr;
};
// Routes of ids[row * top_k + k] (distinct within a row, < experts <= 1024, top_k <= 16). Only experts in
// [first, end) get entries (end 0: all), experts[] holds id - base; order / low_counts cover every slot (the whole
// window's combine). rows * top_k <= 4096 builds in one block (decode windows), more rows in three launches with
// `scratch` ([experts] values). Both forms give the same arrays.
struct MoeRoutesView {
    const std::uint32_t* ids = nullptr;
    std::uint32_t rows = 0, top_k = 0, experts = 0, split = 0;
    std::uint32_t first = 0, end = 0, base = 0;
    std::uint32_t* scratch = nullptr;
    MoeRoutes out;
};
void enqueue_moe_routes(const MoeRoutesView& view, cudaStream_t stream);

// Gate/up of every routed entry: act[slot] = quantize_a8(swiglu_clamped(gate . x[row], up . x[row])); act holds
// [slot][inter] codes and [slot][inter / 32] deltas and sums (sums optional). gate_out / up_out (optional, both or
// neither) receive the FP32 products [slot][inter]. The launch covers max_union union slots (a bound of counts[0]:
// min(rows * top_k, experts)); max_rows (1..8) is the entries staged per pass (the window's rows).
struct RoutedGateUpView {
    MoeShape shape;
    ExpertWeights weights;
    MoeRoutes routes;
    std::uint32_t max_union = 0, max_rows = 8;
    A8Rows x;     // the MoE input, x.stride = hidden (0 = hidden)
    A8Rows act;   // [slots][inter]
    float* gate_out = nullptr;
    float* up_out = nullptr;
};
// Down of every routed entry: raw[slot][o] = down row o . act[slot], unweighted (the raw slot vector).
struct RoutedDownView {
    MoeShape shape;
    ExpertWeights weights;
    MoeRoutes routes;
    std::uint32_t max_union = 0, max_rows = 8;
    A8Rows act;          // [slots][inter]
    float* raw = nullptr;  // [slots][hidden]
    // Expert TP (ds41-expert-tp2-i1152-v1): when set, raw receives rank 0's K-half part of every output and raw1 rank
    // 1's (inter 2304 only; decode GEMVs only).
    float* raw1 = nullptr;
};
// Decode GEMVs (any entry count; best for windows of 1..8 rows).
void enqueue_routed_gate_up(const RoutedGateUpView& view, cudaStream_t stream);
void enqueue_routed_down(const RoutedDownView& view, cudaStream_t stream);
// Prefill grouped GEMM (integer MMA; same bits). Entries of each expert in tiles; `rows` = the window's rows.
// Scratch of routed_grouped_workspace_bytes(rows, shape) bytes.
[[nodiscard]] std::size_t routed_grouped_workspace_bytes(std::uint32_t rows, const MoeShape& shape) noexcept;
void enqueue_routed_gate_up_grouped(const RoutedGateUpView& view, std::uint32_t rows, void* scratch, cudaStream_t stream);
void enqueue_routed_down_grouped(const RoutedDownView& view, std::uint32_t rows, void* scratch, cudaStream_t stream);

// y[row] = (shared + low) + high: low sums w * raw over the row's slots with id < split in ascending id from 0, high
// the others likewise; each step one FP32 product then one FP32 add (reference_window). With half = 0 / 1 only the
// low / high sum is written (reference_half_sum: what a device owning one half sends); shared is not read then.
struct MoeCombineView {
    const float* raw = nullptr;      // [slots][hidden]
    const float* shared = nullptr;   // [rows][hidden], the shared expert's output
    const float* weights = nullptr;  // [slots], the router weights
    const std::uint32_t* order = nullptr;
    const std::uint32_t* low_counts = nullptr;
    std::uint32_t rows = 0, top_k = 0, hidden = 0;
    int half = -1;                   // -1: the whole combine
    float* out = nullptr;            // [rows][hidden]
    // Expert TP: raw / raw1 are the two K-half parts; out = (shared + S0) + S1, S_p the fold of w * part p over the
    // row's slots in ascending expert id from 0 (low_counts unused).
    const float* raw1 = nullptr;
    // the whole combine (half -1, no raw1): hc_expand with sub = out in the same launch (hc_out[r][k][i] from
    // hc_residual[r][j][i] and hc_mixes[r][24], in place allowed; hc_pre_carry optional): enqueue_hc_expand's bits
    const float* hc_residual = nullptr;
    const float* hc_mixes = nullptr;
    float* hc_out = nullptr;
    float* hc_pre_carry = nullptr;
};
void enqueue_moe_combine(const MoeCombineView& view, cudaStream_t stream);
// out = (shared + low) + high elementwise (the combine from two half sums).
void enqueue_moe_canonical_sum(const float* shared, const float* low, const float* high, std::size_t count, float* out,
                               cudaStream_t stream);

// The routed experts of one layer on a window: routes from the router's ids, gate/up, down and the combine with the
// shared expert's output. Decode GEMVs below grouped_min_rows rows, the grouped GEMM from there (0: never).
// Scratch: moe_layer_workspace_bytes(capacity, shape) bytes, rows <= capacity.
[[nodiscard]] std::size_t moe_layer_workspace_bytes(std::uint32_t capacity, const MoeShape& shape) noexcept;
struct MoeLayerView {
    MoeShape shape;
    ExpertWeights weights;
    std::uint32_t rows = 0;
    std::uint32_t split = 192;             // the combine's low/high split (expert id)
    A8Rows x;                              // A8 of the MoE input [rows][hidden]
    const std::uint32_t* ids = nullptr;    // [rows][top_k], router selection order
    const float* route_weights = nullptr;  // [rows][top_k]
    const float* shared = nullptr;         // [rows][hidden]
    float* raw = nullptr;                  // [rows * top_k][hidden] (kept: the routed_experts op hash)
    float* y = nullptr;                    // [rows][hidden]
    void* workspace = nullptr;
    std::uint32_t capacity = 0;
    std::uint32_t grouped_min_rows = 64;
    float* raw1 = nullptr;                 // expert TP: rank 1's K-half parts [rows * top_k][hidden] (decode GEMVs only)
};
void enqueue_moe_layer(const MoeLayerView& view, cudaStream_t stream);
// The parts of a layer workspace (routes, the A8 activations of every slot, scratch), for taps and for running the
// sites one by one.
struct MoeLayerScratch {
    MoeRoutes routes;
    A8Rows act;                        // [capacity * top_k][inter]
    std::uint32_t* routes_scratch = nullptr;
    void* grouped = nullptr;
};
[[nodiscard]] MoeLayerScratch moe_layer_workspace_views(void* workspace, std::uint32_t capacity, const MoeShape& shape);

// The routed experts of one layer whose weights live in several places (resident experts, streamed groups landing
// in a slot): the layer's experts are numbered by a virtual position perm[id] (null: the id itself), each range of
// positions [first, end) reads its own tensors at (position - base) * expert bytes and may wait for an event first
// (its copy). Routes, gate/up and down run range by range on the mapped ids; the combine is the layer's (routes of
// the original ids: ascending expert id, the same split), so y is bitwise enqueue_moe_layer's.
struct MoeRange {
    std::uint32_t first = 0, end = 0, base = 0;
    ExpertWeights weights;
    cudaEvent_t ready = nullptr;  // the stream waits for it before the range (null: none)
    // timing (optional, with ready): recorded on the stream just before / after the wait for ready, so their elapsed
    // time is the time the stream sat idle waiting for the range's copy (null: not recorded)
    cudaEvent_t wait_begin = nullptr, wait_end = nullptr;
};
struct MoeRangesView {
    MoeLayerView layer;                      // weights unused
    const std::uint32_t* perm = nullptr;     // [experts] virtual position of each id (null: identity)
    std::uint32_t* mapped_ids = nullptr;     // [capacity][top_k] scratch (with perm)
    const MoeRange* ranges = nullptr;
    std::uint32_t range_count = 0;
};
void enqueue_moe_layer_ranges(const MoeRangesView& view, cudaStream_t stream);

}  // namespace strix::ds41::cuda
