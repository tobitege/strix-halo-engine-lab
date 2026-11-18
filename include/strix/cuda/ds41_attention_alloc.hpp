#pragma once
// CPU allocation plan shared by the CUDA allocator and requested-byte receipts.
#include "strix/ds41_attention.hpp"

#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <string_view>
#include <vector>

namespace strix::ds41::cuda::attn {

using namespace ::strix::ds41::attn;

// Rows above kSelDirect use two selection levels over kSelSlice entries.
inline constexpr unsigned kSelDirect = 8192, kSelSlice = 4096, kSelMaxSlices = 512;
[[nodiscard]] constexpr std::size_t select_survivor_words(std::uint32_t rows, std::uint32_t max_count, std::uint32_t topk) {
    const std::size_t slices = (std::size_t{max_count} + kSelSlice - 1U) / kSelSlice;
    return max_count > kSelDirect ? std::size_t{rows} * slices * (2U * std::size_t{topk} + 1U) : 0U;
}

struct LayerAttentionOptions {
    std::uint32_t first_layer = 0, end_layer = 40;  // layers [first, end) run here
    std::uint32_t max_positions = 0;                // context length (positions 0 .. max_positions - 1)
    std::uint32_t max_rows = kMaxRows;              // rows of one call (decode window 8; a prefill chunk more)
    std::uint32_t window_rows = 64;                 // prefill: indexer / selection / attention launches of this many rows
    std::uint32_t hidden = 0, q_lora_rank = 0;     // the model's hidden_size and q_lora_rank (required)
    // External drivers can supply their own workspace and compact the decoder's per-row selection/candidate data.
    // Defaults retain the original enqueue_layer allocation and layout.
    bool allocate_workspace = true;
    std::uint32_t decoder_rows = 0;                // 0: max_rows; layers >= 20 only
    std::uint32_t candidate_rows = 0;              // 0: max_rows
    std::uint32_t ring_rows = 0;                   // 0: kWindow + max_rows; explicit physical ring capacity
};

inline LayerAttentionOptions decode_attention_options(LayerAttentionOptions o, bool decode_only, bool lean_state,
                                                     std::uint32_t decode_rows) {
    if (decode_only && lean_state) {
        o.max_rows = decode_rows;
        o.window_rows = std::min(o.window_rows, decode_rows);
        o.ring_rows = 136U + decode_rows;
        o.decoder_rows = o.candidate_rows = 0;
        o.allocate_workspace = true;
    }
    return o;
}

inline LayerAttentionOptions prefill_attention_options(LayerAttentionOptions o, std::uint32_t pass_rows,
                                                      std::uint32_t decode_rows) {
    if (pass_rows) {
        o.max_rows = std::min(o.max_rows, pass_rows);
        o.ring_rows = 136U + std::max(o.max_rows, decode_rows);
        o.allocate_workspace = false;
    }
    return o;
}

enum class AllocationClass { persistent_history, bounded_window, working };
struct Allocation {
    std::string_view name;
    AllocationClass kind;
    int layer;
    std::size_t bytes;
};
struct AllocationLedger {
    std::vector<Allocation> allocations;
    std::uint32_t ring_rows = 0, cand_stride = 0;
    void add(std::string_view name, AllocationClass kind, int layer, std::size_t count, std::size_t element = 1) {
        allocations.push_back({name, kind, layer, std::max<std::size_t>(count, 1) * element});
    }
    [[nodiscard]] std::size_t bytes(std::string_view name, int layer = -1) const {
        for (const auto& a : allocations) if (a.name == name && a.layer == layer) return a.bytes;
        return 0;
    }
    [[nodiscard]] std::size_t total() const {
        std::size_t n = 0;
        for (const auto& a : allocations) n += a.bytes;
        return n;
    }
    [[nodiscard]] std::size_t total(AllocationClass kind) const {
        std::size_t n = 0;
        for (const auto& a : allocations) if (a.kind == kind) n += a.bytes;
        return n;
    }
};

[[nodiscard]] constexpr std::size_t rope_table_bytes(std::uint32_t positions) {
    return std::max<std::size_t>(positions, 1) * sizeof(RopeRow);
}

inline AllocationLedger attention_allocations(std::span<const ds41::attn::LayerAttention> plan, const LayerAttentionOptions& o) {
    AllocationLedger a;
    if (o.first_layer >= o.end_layer || o.end_layer > plan.size() || !o.max_positions || !o.max_rows || !o.window_rows ||
        o.decoder_rows > o.max_rows || o.candidate_rows > o.max_rows)
        throw std::invalid_argument("ds41 attention allocation options");
    a.ring_rows = o.ring_rows ? o.ring_rows : kWindow + o.max_rows;
    if (a.ring_rows < kWindow + o.max_rows) throw std::invalid_argument("ds41 attention ring rows");
    a.cand_stride = static_cast<std::uint32_t>((std::size_t{o.max_positions} + kCandBlockSize - 1U) / kCandBlockSize + 1U);
    const auto P = AllocationClass::persistent_history, B = AllocationClass::bounded_window, W = AllocationClass::working;
    std::vector<bool> need(plan.size(), false);
    for (std::uint32_t l = o.first_layer; l < o.end_layer; ++l) {
        need[l] = true;
        if (plan[l].ratio) {
            need.at(static_cast<std::size_t>(plan[l].kv_source)) = true;
            need.at(static_cast<std::size_t>(plan[l].index_source)) = true;
        }
    }
    bool candidates = false;
    for (std::size_t l = 0; l < plan.size(); ++l) {
        if (!need[l]) continue;
        const auto& r = plan[l];
        const int layer = static_cast<int>(l);
        const bool local = l >= o.first_layer && l < o.end_layer;
        if (local) a.add("ring", B, layer, std::size_t{a.ring_rows} * kWindowEntryBytes);
        if (r.owns_kv) {
            const std::size_t entries = o.max_positions / r.ratio + 1U;
            a.add("kv_cache", P, layer, entries * kKvEntryBytes);
            a.add("idx_cache", P, layer, entries * kIdxEntryBytes);
            if (r.ratio == 2U) {
                a.add("carry", P, layer, 2U * kHeadDim, 4);
                a.add("carry_valid", P, layer, 1, 4);
                if (local && o.allocate_workspace) a.add("carry_rows", W, layer, std::size_t{o.max_rows} * 2U * kHeadDim, 4);
            }
        }
        if (r.owns_index) {
            const auto rows = l >= 20U && o.decoder_rows ? o.decoder_rows : o.max_rows;
            a.add("idx", W, layer, std::size_t{rows} * kTopK, 4);
            a.add("idx_count", W, layer, rows, 4);
        }
        candidates = candidates || r.candidate_source || r.candidate_user;
    }
    if (candidates) a.add("candidates", W, -1, std::size_t{o.candidate_rows ? o.candidate_rows : o.max_rows} * a.cand_stride);
    if (o.allocate_workspace) {
        const std::size_t rows = o.max_rows, w = std::min(o.window_rows, o.max_rows);
        a.add("ckv", W, -1, rows * kHeadDim, 4);
        a.add("cscore", W, -1, rows * kHeadDim, 4);
        a.add("iq", W, -1, rows * kIdxHeads * kIdxDim, 4);
        a.add("iw", W, -1, rows * kIdxHeads, 4);
        a.add("qprep", W, -1, w * (kIdxHeads * kIdxDim + kIdxHeads * 4U * 4U + kIdxHeads * 4U));
        a.add("scores", W, -1, w * o.max_positions, 4);
        a.add("block_keys", W, -1, w * a.cand_stride, 4);
        a.add("survivors", W, -1, select_survivor_words(static_cast<std::uint32_t>(w), o.max_positions, kTopK), 4);
        a.add("partial", W, -1, w * kHeads * kMaxChunks * (4U + kHeadDim), 4);
    }
    a.add("rope_window", W, -1, rope_table_bytes(o.max_positions));
    a.add("rope_compressed", W, -1, rope_table_bytes(o.max_positions));
    return a;
}

inline std::string allocation_json(const AllocationLedger& a) {
    std::ostringstream s;
    s << "{\"persistent_history\":" << a.total(AllocationClass::persistent_history)
      << ",\"bounded_window\":" << a.total(AllocationClass::bounded_window)
      << ",\"working\":" << a.total(AllocationClass::working) << ",\"total\":" << a.total() << "}";
    return s.str();
}

inline AllocationLedger prefill_sequence_allocations(std::uint32_t rows, std::uint32_t hidden,
                                                     std::uint32_t q_lora_rank, std::uint32_t top_k) {
    AllocationLedger a;
    const auto W = AllocationClass::working;
    const std::size_t t = rows;
    a.add("stream", W, -1, t * 4U * hidden, 4);
    for (auto name : {"mix_attn", "mix_ffn"}) a.add(name, W, -1, t * 24U, 4);
    for (auto name : {"x", "y", "shared"}) a.add(name, W, -1, t * hidden, 4);
    a.add("qkv", W, -1, t * (q_lora_rank + kHeadDim), 4);
    a.add("kvrows", W, -1, t * kHeadDim, 4);
    a.add("qr", W, -1, t * q_lora_rank, 4);
    for (auto name : {"route_w", "ids"}) a.add(name, W, -1, t * top_k, 4);
    a.add("xa", W, -1, t * hidden * 5U / 4U);
    a.add("qa", W, -1, t * q_lora_rank * 5U / 4U);
    return a;
}

struct DecodeAllocationOptions {
    std::uint32_t rows = 8, hidden = 5120, top_k = 6, cand_stride = 0, inter = 2304;
    std::uint32_t raw_rows = 0;                    // 0: rows; combined objects retain moe_chunk_rows
    std::size_t moe_workspace = 0, trellis_workspace = 0;
    bool expert_tp = false, trellis = false;
};
inline AllocationLedger decode_allocations(const DecodeAllocationOptions& o) {
    AllocationLedger a;
    const auto W = AllocationClass::working;
    a.add("raw", W, -1, std::size_t{o.raw_rows ? o.raw_rows : o.rows} * o.top_k * o.hidden, 4);
    if (o.expert_tp) a.add("raw1", W, -1, std::size_t{o.rows} * o.top_k * o.hidden, 4);
    // DeviceBuffer's empty buffers do not allocate.
    if (o.moe_workspace) a.add("moe_ws", W, -1, o.moe_workspace);
    a.add("pos", W, -1, 8, 8);
    a.add("kvrows", W, -1, 8U * kHeadDim, 4);
    a.add("router_counters", W, -1, 2, 4);
    a.add("sec_dev", W, -1, std::size_t{o.rows} * (4U * (kKvEntryBytes + kIdxEntryBytes) +
          40U * (4U + kTopK * 4U) + o.cand_stride) + 4096U);
    if (o.trellis) {
        for (auto name : {"tr_gate", "tr_up", "tr_act"}) a.add(name, W, -1, std::size_t{o.rows} * o.inter, 4);
        if (o.trellis_workspace) a.add("tr_ws", W, -1, o.trellis_workspace);
    }
    return a;
}

}  // namespace strix::ds41::cuda::attn
