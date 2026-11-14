#pragma once

// Device side of the deepseek41 prefill experts (gfx1151, wave32, v_wmma_i32_16x16x16_iu8).
//
// Routes are built on the device from the router's ids (ds41_prefill.hpp layout, bit-identical to
// build_prefill_routes). The gate/up and down kernels take one batch of up to 16 or 32 entries of one
// expert per workgroup: batches of 16 run the class-split kernels (each wave owns four lane classes,
// so weight rows are read block by block in address order), batches of 32 the stage kernels (two
// 16-row tiles per wave); STRIX_DS41_PREFILL_STAGE=1 sends batches of 16 to the stage kernels too
// (comparison only). Both reproduce the decode products bitwise:
//  - each 32-weight group's integer dot of every (entry, output) comes from two 16x16x16 iu8 WMMAs
//    (K = 16 + 16) on an accumulator of 1.5 * 2^23, so one FP32 subtraction gives float(dot);
//    Q2_K weights enter the WMMA as code * (scale & 15) (<= 45), so the dot is A of the declared
//    Q2_K term, and B = sum_h (scale_h >> 4) * sum x_h is formed in integers from staged half sums;
//  - the group's FP32 term is the declared one (IQ2_XXS / MXFP4 / Q8_0: float(dot) * (scale *
//    delta); Q2_K: (d * float(A) - dmin * float(B)) * delta);
//  - lane class l of the declared product (groups l, l + 32, ...) is accumulated in ascending order
//    from 0, and the 32 class sums are combined in the butterfly's tree, the decode's v[l] += v[l ^ d],
//    d = 16..1 (stage kernels: classes in bit-reversed order folded by a five-level carry stack;
//    class-split kernels: levels 1-3 between waves through LDS, levels 4-5 in registers).
// The gate/up epilogue is the decode's: clamped SwiGLU (declared exp) and A8 per 32 outputs.

#include "strix/ds41_formats.hpp"
#include "strix/hip_ds41_formats.hpp"

#include <cstdint>

namespace strix::ds41 {

// Routes on the device, sized by the bounds of prefill_max_batches.
struct PrefillRoutesView {
    const std::uint32_t* begins = nullptr;        // [experts + 1]
    const std::uint32_t* entry_rows = nullptr;    // [rows * top_k]
    const std::uint32_t* entry_slots = nullptr;   // [rows * top_k]
    const std::uint32_t* batch_expert = nullptr;  // [max_batches]
    const std::uint32_t* batch_begin = nullptr;
    const std::uint32_t* batch_count = nullptr;
    const std::uint32_t* batch_total = nullptr;   // one value: the number of batches
    std::uint32_t max_batches = 0;
    std::uint32_t batch_rows = 32;                // 16 or 32
};
struct PrefillRoutesBuffers {
    std::uint32_t* counts = nullptr;              // [experts] scratch
    // batch_begin holds max_batches + 1 entries: batch_begin[batch_total] = the entry count, so the
    // batches are also window routes of at most batch_rows entries each.
    std::uint32_t* begins = nullptr;              // [experts + 1]
    std::uint32_t* entry_rows = nullptr;
    std::uint32_t* entry_slots = nullptr;
    std::uint32_t* batch_expert = nullptr;
    std::uint32_t* batch_begin = nullptr;
    std::uint32_t* batch_count = nullptr;
    std::uint32_t* batch_total = nullptr;
    std::uint32_t* order = nullptr;               // [rows * top_k]
    std::uint32_t* low_counts = nullptr;          // [rows]
    // Histogram split (optional, sparse_max > 0): every owned expert with 1..sparse_max entries gets batches of 8 in the
    // sparse_* arrays (batch j = entries sparse_begin[j] .. sparse_begin[j + 1], so window routes as they are) and its
    // entries are placed first, in ascending expert; the other experts get the batch_* batches above. sparse_begin holds
    // max_batches + 1 entries (the batch bound of 8-entry batches).
    std::uint32_t sparse_max = 0;
    std::uint32_t* sparse_expert = nullptr;
    std::uint32_t* sparse_begin = nullptr;
    std::uint32_t* sparse_total = nullptr;
};
// ids[row * top_k + k]; experts <= 1024, top_k <= 16. With an expert range [first, end) (end 0: all)
// only those experts get entries and batches, numbered from first (a rank that keeps only its slice
// of the expert tensors indexes them from its own base); order and low_counts cover every slot.
void enqueue_build_prefill_routes(const std::uint32_t* ids, std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts,
                                  std::uint32_t split, std::uint32_t batch_rows, const PrefillRoutesBuffers& out, void* stream,
                                  std::uint32_t first = 0, std::uint32_t end = 0);

// Router of T rows (any count): the decode router's declared arithmetic per row (RouterView of
// hip_ds41_formats.hpp: scores, ids and weights of every row equal enqueue_router's bitwise).
void enqueue_prefill_router(const RouterView& view, void* stream);
[[nodiscard]] PrefillRoutesView prefill_routes_view(const PrefillRoutesBuffers& buffers, std::uint32_t rows, std::uint32_t top_k,
                                                    std::uint32_t experts, std::uint32_t batch_rows);

// Gate/up of every entry: activation row entry_rows[i] (codes[row * hidden], deltas[row * hidden /
// 32]) times the gate and up rows of the batch's expert; writes the A8 activation of slot
// entry_slots[i] (act_codes[slot * inter], act_deltas / act_sums[slot * inter / 32]) and, if given,
// the FP32 gate / up values (gate_out[slot * inter + o]). Formats IQ2_XXS, MXFP4, Q8_0;
// hidden <= 5120, a multiple of 256 (IQ2_XXS) or 32; inter a multiple of 64.
struct PrefillGateUpView {
    const void* gate = nullptr;
    const void* up = nullptr;
    std::uint64_t expert_stride_bytes = 0;
    std::uint64_t row_bytes = 0;
    std::uint32_t hidden = 0;
    std::uint32_t inter = 0;
    PrefillRoutesView routes;
    const std::int8_t* codes = nullptr;
    const float* deltas = nullptr;
    float limit = 10.0F;
    float* gate_out = nullptr;
    float* up_out = nullptr;
    std::int8_t* act_codes = nullptr;
    float* act_deltas = nullptr;
    std::int32_t* act_sums = nullptr;
};
void enqueue_prefill_gate_up(WeightFormat format, const PrefillGateUpView& view, void* stream);
// FAST lever ds41-moe-relaxed-v1 (NOT bit-exact; STRIX_DS41_FAST_MOE=1, default off): IQ2_XXS gate/up and Q2_K down of the
// batch path (enqueue_prefill_gate_up / _down) add each output's group terms in ascending group order with FMAs, 64 rows of
// one expert per block; same A8 activations, weight integers and group dots. Test override: 1 on, 0 off, -1 environment.
void set_fast_moe(int mode);
[[nodiscard]] bool fast_moe();
// Draft FAST_I4: original GGUF weights decoded in-kernel to IU4, A8 input
// requantized to A4 per K32. Default OFF. STRIX_DS41_FAST_I4=1; scope
// STRIX_DS41_FAST_I4_SCOPE=all (default) or gu. Separate numerical identity
// gguf-iu4-direct-v1; no pack or converter. Runs on the shared tiled core.
// Overrides are for test harnesses.
void set_fast_i4(int mode); // 1 on, 0 off, -1 environment
[[nodiscard]] bool fast_i4();
// STRIX_DS41_PF_CORE: which prefill products run on the shared tiled core (src/core, policy interface v1) instead of
// the kernels described above: 0 none, moe (expert gate/up and down), dense (dense Q8_0 x A8), 1 both. Unset: per form,
// the shared core where it is measured faster (today the Q8_0 gate/up and down of the shared expert, the routed IQ2_XXS gate/up at
// hidden widths that are a multiple of 1024 and the routed Q2_K down, in 16-row batches,
// FAST_MOE off, STRIX_DS41_PREFILL_STAGE not 1). The same bits. Resolved once per run; any other value (the empty one
// included) throws std::invalid_argument at the first product.
// Harness override, may change between calls: set_pf_core(1 every product / 0 none / -1 back to the environment), or
// by policy name: "0", "1", "moe", "dense", nullptr = the per-form default (as if unset). pf_core / pf_core_dense: the
// product is named on.
void set_pf_core(int mode);
void set_pf_core(const char* policy);
[[nodiscard]] bool pf_core();       // expert products (gate/up, down)
[[nodiscard]] bool pf_core_dense(); // dense Q8_0 x A8 products
// The route the next expert product of this shape takes under the current policy (true: the shared core). The per-form
// default excludes IQ2_XXS gate/up widths that are not a multiple of 1024 (the core loads whole chunks). For harnesses
// that compare the two routes.
[[nodiscard]] bool prefill_gate_up_on_core(WeightFormat format, std::uint32_t hidden, std::uint32_t batch_rows);
[[nodiscard]] bool prefill_down_on_core(WeightFormat format, std::uint32_t width, std::uint32_t batch_rows);

// Down of every entry: A8 activation of slot entry_slots[i] (codes[slot * width], deltas) times the
// batch expert's rows; out[slot * outputs + o] is the unweighted raw slot vector. Formats Q2_K,
// MXFP4, Q8_0; width <= 5120; outputs a multiple of 128.
struct PrefillDownView {
    const void* weights = nullptr;
    std::uint64_t expert_stride_bytes = 0;
    std::uint64_t row_bytes = 0;
    std::uint32_t width = 0;
    std::uint32_t outputs = 0;
    PrefillRoutesView routes;
    const std::int8_t* codes = nullptr;
    const float* deltas = nullptr;
    float* out = nullptr;
};
void enqueue_prefill_down(WeightFormat format, const PrefillDownView& view, void* stream);

// Routed experts of T rows in chunks of at most chunk_rows rows (the raw slot vectors of a whole 8K
// prompt would take ~1 GB). Rows are independent at every step (routes, products, SwiGLU, A8,
// combine), so the results are those of one pass over all rows, whatever the chunk size. A rank that
// keeps only experts [first, end) of the tensors (weights pointers at expert `first`) produces the
// half sums of its half instead of the combine: half 0 = ids below split, half 1 = the rest
// (enqueue_half_sum's sums), and the canonical y = (shared + low) + high is formed after the exchange.
struct PrefillMoeWeights {
    WeightFormat gate_up_format = WeightFormat::iq2_xxs, down_format = WeightFormat::q2_k;
    const void* gate = nullptr;
    const void* up = nullptr;
    const void* down = nullptr;
    std::uint64_t gate_row_bytes = 0, down_row_bytes = 0; // one output row
    std::uint32_t hidden = 0, inter = 0;
    // Down width (0: inter). Larger than inter (expert TP, ds41-expert-tp2-i1152-v1 in its 420-byte "dup" rows: inter
    // 1152, down rows of 5 whole Q2_K blocks = 1280): the A8 activation of every slot is copied to a down_width stride
    // with the groups past inter zero (codes 0, delta 0), so each of those groups adds +0 to its lane class and the
    // down product is the declared 36-group product bit for bit.
    std::uint32_t down_width = 0;
    float limit = 10.0F;
};
// Per-call inputs over all T rows; chunk c covers rows [c * chunk_rows, min(T, (c + 1) * chunk_rows)).
struct PrefillMoeInputs {
    std::uint32_t rows = 0;
    const std::uint32_t* ids = nullptr;     // [rows * top_k]
    const float* weights = nullptr;         // [rows * top_k] routing weights
    const std::int8_t* codes = nullptr;     // [rows * hidden] A8 MoE input
    const float* deltas = nullptr;          // [rows * hidden / 32]
    const float* shared = nullptr;          // [rows * hidden] shared-expert output (combine only)
};
class PrefillMoe {
public:
    PrefillMoe(const PrefillMoeWeights& weights, std::uint32_t top_k, std::uint32_t experts, std::uint32_t split,
               std::uint32_t chunk_rows, std::uint32_t first = 0, std::uint32_t end = 0);
    ~PrefillMoe();
    PrefillMoe(const PrefillMoe&) = delete;
    PrefillMoe& operator=(const PrefillMoe&) = delete;
    [[nodiscard]] std::uint32_t chunks(std::uint32_t rows) const noexcept { return (rows + chunk_rows_ - 1U) / chunk_rows_; }
    [[nodiscard]] std::uint32_t chunk_rows() const noexcept { return chunk_rows_; }
    // Routes, gate/up and down of chunk c (the owned experts), then y = (shared + low) + high of its
    // rows (requires every expert) ...
    void enqueue_combine(const PrefillMoeInputs& in, std::uint32_t chunk, float* y, void* stream);
    // ... or the half sum of its rows (requires the owned range to be exactly that half).
    void enqueue_half(const PrefillMoeInputs& in, std::uint32_t chunk, std::uint32_t half, float* out, void* stream);
    // Joint form for the two-rank row split: the rows of `peer` (the other rank's output rows) and of `own` as one row
    // set [peer; own], in chunks of chunk_rows rows: one route build and one expert pass per chunk, then the half sums of
    // the chunk's peer rows (into out_peer[row]) BEFORE those of its own rows (into out_own[row]); peer_done (optional)
    // is recorded on the stream right after the last peer row's half sum, so their send can start there. Every row
    // gets the same bits as enqueue_half on its own set. Inputs are copied into scratch kept by this object.
    void enqueue_half_joint(const PrefillMoeInputs& peer, const PrefillMoeInputs& own, std::uint32_t half, float* out_peer,
                            float* out_own, void* stream, void* peer_done = nullptr);
    // Selector between the decode window kernels (batches of 8 entries) and the WMMA kernels (batches of 16), the same
    // declared products, so the same bits. Default (STRIX_DS41_PREFILL_SELECT=hist): per owned expert, from its real
    // entry count in the chunk: at most window_mean entries -> window kernels, else WMMA; both run in the chunk. "mean":
    // the whole chunk by its estimated mean rows * top_k / owned (the previous selector; it counts every slot of a row,
    // also those of the other rank's experts). window_mean default 5, STRIX_DS41_PREFILL_WINDOW_MEAN overrides it (0:
    // always WMMA).
    void set_window_mean(double mean) noexcept { window_mean_ = mean; }
    void set_select_hist(bool hist) noexcept { select_hist_ = hist; }
    // Weights of another layer (same geometry), so one PrefillMoe and its scratch serve every layer.
    void set_weights(const PrefillMoeWeights& weights);
    // Device scratch held (bytes).
    [[nodiscard]] std::uint64_t scratch_bytes() const noexcept { return scratch_bytes_; }

private:
    void enqueue_experts(const PrefillMoeInputs& in, std::uint32_t chunk, std::uint32_t& row0, std::uint32_t& rows, void* stream);
    void enqueue_experts_split(const PrefillMoeInputs& in, std::uint32_t row0, std::uint32_t rows, void* stream);
    [[nodiscard]] unsigned selected_batch_rows(unsigned rows) const;
    void experts_rows(const PrefillMoeInputs& in, std::uint32_t row0, std::uint32_t rows, void* stream);
    void half_rows(const float* weights, std::uint32_t first_row, std::uint32_t rows, std::uint32_t half, float* out, void* stream);
    void pad_activations(std::uint32_t rows, void* stream);  // down_width > inter only
    [[nodiscard]] const std::int8_t* down_codes() const noexcept { return pad_codes_ ? pad_codes_ : act_codes_; }
    [[nodiscard]] const float* down_deltas() const noexcept { return pad_deltas_ ? pad_deltas_ : act_deltas_; }
    PrefillMoeWeights w_;
    std::uint32_t top_k_, experts_, split_, chunk_rows_, first_, end_;
    std::uint64_t scratch_bytes_ = 0;
    double window_mean_ = 5.0;
    bool select_hist_ = true;
    void* memory_ = nullptr;
    PrefillRoutesBuffers routes_{};
    std::int8_t* act_codes_ = nullptr;
    float* act_deltas_ = nullptr;
    std::int32_t* act_sums_ = nullptr;
    std::int8_t* pad_codes_ = nullptr;  // down_width > inter: the activations at the down's stride
    float* pad_deltas_ = nullptr;
    float* raw_ = nullptr;
    void* joint_ = nullptr;  // enqueue_half_joint's input copies (allocated on first use)
};

} // namespace strix::ds41
