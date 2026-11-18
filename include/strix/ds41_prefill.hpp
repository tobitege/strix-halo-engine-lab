#pragma once

#include <memory>
#include <exception>

// deepseek41 prefill: the routed experts of T prompt rows at once, and the exact suffix replay plan.
//
// Routes. Every (row, k) assignment is one slot = row * top_k + k, as in the decode window. The
// prefill CSR is dense over expert ids: expert e owns entries begins[e] .. begins[e + 1] in
// ascending slot (= ascending row, an expert appears at most once per row); entry i reads activation
// row rows[i] and writes slot slots[i]. The entries of each expert are cut into batches of at most
// batch_rows consecutive entries, in ascending expert id; batch b is (batch_expert[b],
// batch_begin[b], batch_count[b]). order / low_counts are the combine's (ds41_moe.hpp): per row its
// top_k slots in ascending expert id and how many of them have id < split.
//
// Arithmetic. The prefill expert kernels produce, for every (entry, output), the same bits as the
// decode products of ds41_formats.hpp (same integer group dots, same per-group FP32 expressions,
// same lane classes and the same butterfly tree), so a prefill row equals the row computed alone by
// the decode window, whatever T, the batch or the other rows. No new numerical identity.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace strix::ds41 {

struct PrefillPass {
    std::uint32_t begin = 0, rows = 0;
};
// Consecutive global positions, bounded row buffers and aligned starts. Rebalance a short final pass so that a
// decoder suffix can fit entirely in it. The final row count need not be aligned; all preceding counts are.
[[nodiscard]] std::vector<PrefillPass> plan_prefill_passes(std::uint32_t total, std::uint32_t pass_rows,
                                                        std::uint32_t alignment, std::uint32_t min_final_rows = 0);

struct PrefillRoutes {
    std::uint32_t rows = 0, top_k = 0, experts = 0, batch_rows = 0;
    std::vector<std::uint32_t> begins;                                 // [experts + 1]
    std::vector<std::uint32_t> entry_rows, entry_slots;                // [rows * top_k]
    std::vector<std::uint32_t> batch_expert, batch_begin, batch_count; // [batches]
    std::vector<std::uint32_t> order, low_counts;                      // [rows * top_k], [rows]
};
// Upper bound of the batch count for any routing of rows * top_k slots over `experts` experts.
[[nodiscard]] constexpr std::uint32_t prefill_max_batches(std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts,
                                                          std::uint32_t batch_rows) noexcept {
    return (rows * top_k + batch_rows - 1U) / batch_rows + experts;
}
// ids[row * top_k + k] < experts, distinct within a row. batch_rows is 16 or 32. With an expert range
// [first, end) (end 0: all) only those experts own entries; begins and batch_expert are relative to
// first; order / low_counts still cover every slot.
[[nodiscard]] PrefillRoutes build_prefill_routes(std::span<const std::uint32_t> ids, std::uint32_t rows, std::uint32_t top_k,
                                                 std::uint32_t experts, std::uint32_t split, std::uint32_t batch_rows,
                                                 std::uint32_t first = 0, std::uint32_t end = 0);

// ---- Exact suffix replay (CED prefill) ----
//
// Each layer keeps its own sliding window of `window` K/V rows (the decode ring), computed from that
// layer's own input. A layer's output for row s reads the window rows s - window + 1 .. s of the same
// layer. The compressed KV and index keys every decoder layer reads come from the first decoder layer
// (the KV source), whose input is the encoder output, so they exist for every row once the encoder
// has run. Hence, after the encoder layers ran on every new row, the decoder layers need only a cone
// of trailing rows to leave the decode state (every layer's window ring, the KV-source caches) and
// the last row's logits exactly as a full prefill would.
struct SuffixReplayGeometry {
    std::uint32_t layers = 40;          // decoder layers are first_decoder .. layers - 1
    std::uint32_t first_decoder = 20;   // the KV source of every decoder layer
    std::uint32_t window = 128;         // per-layer sliding window (rows attended, current row included)
    // Rows whose inputs of the last draft_layers layers the drafter needs (window seed), 0 for none.
    std::uint32_t draft_rows = 128;
    std::uint32_t draft_layers = 3;     // inputs of layers layers - draft_layers .. layers - 1
};
struct SuffixReplayLayer {
    std::uint32_t layer = 0;
    // Rows [out_begin, total) run the whole layer (attention output, MoE, ...); rows
    // [kv_begin, total) need this layer's window K/V (projection of the layer input). Encoder layers
    // run every new row: out_begin = kv_begin = cached.
    std::uint64_t out_begin = 0, kv_begin = 0;
};
struct SuffixReplayPlan {
    std::uint64_t cached = 0, total = 0;
    std::vector<SuffixReplayLayer> layers;
    // Rows of the first decoder layer's input (= encoder output) that must be available, including
    // rows before `cached` that only a ring entry can provide (see ring_rows).
    std::uint64_t encoder_output_begin = 0;
    // Window-ring rows of the cached prefix each layer reads (cached - window + 1 .. cached - 1 at
    // most): they come from the ring saved with the prefix, never recomputed.
    std::uint64_t ring_rows = 0;
    std::uint64_t full_row_layers = 0;    // rows x layers of a full prefill of the new rows
    std::uint64_t replay_row_layers = 0;  // rows x layers this plan runs (out rows)
};
// cached = rows already in the state (0 for a fresh prompt), total = rows after this prefill.
[[nodiscard]] SuffixReplayPlan plan_suffix_replay(const SuffixReplayGeometry& geometry, std::uint64_t cached, std::uint64_t total);
// Brute-force check of a plan: the backward dependency closure of the decode state, logits and
// drafter seed over the per-layer windows, row by row. Returns an empty string when the plan
// computes exactly the needed rows (nothing missing, nothing extra), else a description.
[[nodiscard]] std::string verify_suffix_replay(const SuffixReplayGeometry& geometry, const SuffixReplayPlan& plan);

struct PrefillMoeLayout {
    std::uint64_t o_counts, o_begins, o_rows, o_slots, o_bexp, o_bbeg, o_bcnt, o_btot, o_order, o_low, o_sexp, o_sbeg, o_stot, o_codes, o_deltas, o_sums, o_raw, o_pcodes, o_pdeltas, bytes;
};
inline PrefillMoeLayout prefill_moe_layout(std::uint32_t chunk_rows, std::uint32_t top_k,
    std::uint32_t owned, std::uint32_t hidden, std::uint32_t inter, std::uint32_t down_width) {
    if (down_width == 0U) down_width = inter;
    const std::uint64_t slots = static_cast<std::uint64_t>(chunk_rows) * top_k;
    const std::uint64_t batches = prefill_max_batches(chunk_rows, top_k, owned, 8U);
    // One allocation, 256-byte aligned parts.
    std::uint64_t offset = 0;
    const auto take = [&](std::uint64_t bytes) { const auto at = offset; offset += (bytes + 255U) / 256U * 256U; return at; };
    const auto o_counts = take(owned * 4U), o_begins = take((owned + 1U) * 4U), o_rows = take(slots * 4U), o_slots = take(slots * 4U);
    const auto o_bexp = take(batches * 4U), o_bbeg = take((batches + 1U) * 4U), o_bcnt = take(batches * 4U), o_btot = take(4U);
    const auto o_order = take(slots * 4U), o_low = take(static_cast<std::uint64_t>(chunk_rows) * 4U);
    const auto o_sexp = take(batches * 4U), o_sbeg = take((batches + 1U) * 4U), o_stot = take(4U);
    const auto o_codes = take(slots * inter), o_deltas = take(slots * inter / 32U * 4U), o_sums = take(slots * inter / 32U * 4U);
    const auto o_raw = take(slots * hidden * 4U);
    const bool pad = down_width != inter;
    const auto o_pcodes = pad ? take(slots * down_width) : 0U, o_pdeltas = pad ? take(slots * down_width / 32U * 4U) : 0U;
    return {o_counts, o_begins, o_rows, o_slots, o_bexp, o_bbeg, o_bcnt, o_btot, o_order, o_low, o_sexp, o_sbeg, o_stot, o_codes, o_deltas, o_sums, o_raw, o_pcodes, o_pdeltas, offset};
}

// Budget is free memory while the old allocation remains live. Commit only after success.
template<class Scratch, class Allocate>
bool try_grow_moe_scratch(std::unique_ptr<Scratch>& current, std::uint64_t bytes,
                          std::uint64_t free_bytes, Allocate allocate) {
    if (bytes > free_bytes) return false;
    try {
        auto replacement = allocate();
        current.swap(replacement);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}
// ---- Bounded decoder replay (FAST policy ds41-decoder-replay-bounded-v1; NOT bit-equal to a full prefill) ----
//
// The encoder layers run every new row as above; every decoder layer runs only its last `bound` rows:
// out_begin = kv_begin = max(cached, floor), and a row s of a decoder layer reads the window keys
// max(kv_begin, s - window + 1) .. s (the key floor; no window key of that layer before the cut). The first decoder
// layer still projects every row's compressed KV and index keys (the engine's x0 = 0): those keys stay readable.
// Policy identity: floor = round_down(max(T - R, 0), 256), independent of window_rows.
// 256 is the largest legal scheduling window (attn::kMaxCallRows); v1 scheduling windows must divide it.
// draft_rows > R is refused.
// `bound` is a positive multiple of the attention window;
// the ring after the run then holds rows total - window .. total - 1 of every decoder layer, as a full prefill's.
// The smallest bound the DS41 FAST quality gate qualifies (20 prompts x (the prefill's last row + 64 teacher-forced rows
// from the bounded state), USER rule KL <= 0.02, top-1 >= 93 %, dNLL <= null mean + 2 sd): R = 256 passes (dNLL +0.00632
// against +0.00991), R = 128 fails (+0.01406). Engines refuse smaller bounds; the planner itself takes any.
inline constexpr std::uint32_t bounded_replay_floor_unit = 256;
[[nodiscard]] constexpr std::uint64_t bounded_replay_key_floor(std::uint64_t total, std::uint32_t bound) {
    return total > bound ? (total - bound) / bounded_replay_floor_unit * bounded_replay_floor_unit : 0U;
}
inline constexpr std::uint32_t bounded_replay_min_rows = 256;
[[nodiscard]] SuffixReplayPlan plan_bounded_replay(const SuffixReplayGeometry& geometry, std::uint64_t cached, std::uint64_t total,
                                                   std::uint32_t bound);

// Routed-MoE schedule of a row-split pair prefill. Bitwise: the MoE's rows are independent, so neither the window size
// nor the joint / two-pass form changes a row's bits; only the number of expert reads and launches changes.
// moe_window_rows: one MoE window per pass. The pass rows (the larger of a layer's peer and own rows, the same value on
// both ranks) rounded up to whole `chunk` rows, at least `base` (the configured window) and at most max(cap, base).
[[nodiscard]] std::uint32_t moe_window_rows(std::uint32_t base, std::uint32_t chunk, std::uint32_t pass_rows, std::uint32_t cap);
// Explicit windows and scheduled slices bound automatic growth.
inline std::uint32_t bounded_moe_window_rows(std::uint32_t base, std::uint32_t chunk,
    std::uint32_t pass_rows, std::uint32_t cap, bool automatic, std::uint32_t task_cap) {
    const auto rows = automatic ? moe_window_rows(base, chunk, pass_rows, cap) : base;
    return task_cap != 0U && task_cap < rows ? task_cap : rows;
}
// moe_joint_layer: one expert pass over [peer rows; own rows] when the LAYER's rows (peer + own, the same on both ranks)
// are at most max_rows (0: never). The two-pass form overlaps the peer rows' half sums with the own rows but reads every
// expert twice; below max_rows the saved reads outweigh the exposed half-sum wire (rows x hidden x 4 B over the link).
[[nodiscard]] bool moe_joint_layer(std::uint32_t peer_rows, std::uint32_t own_rows, std::uint32_t max_rows);

} // namespace strix::ds41
