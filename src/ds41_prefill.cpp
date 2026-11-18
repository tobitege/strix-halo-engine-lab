#include "strix/ds41_prefill.hpp"

#include <algorithm>
#include <stdexcept>

namespace strix::ds41 {

std::vector<PrefillPass> plan_prefill_passes(std::uint32_t total, std::uint32_t pass_rows,
                                          std::uint32_t alignment, std::uint32_t min_final_rows) {
    if (total == 0U || alignment == 0U || pass_rows == 0U || pass_rows % alignment != 0U || min_final_rows > pass_rows)
        throw std::invalid_argument("ds41 prefill passes: nonempty rows, aligned capacity and final suffix within capacity");
    std::vector<PrefillPass> passes;
    std::uint32_t begin = 0;
    while (total - begin > pass_rows) {
        const auto left = total - begin;
        auto rows = pass_rows;
        if (left - rows < min_final_rows) rows = (left - min_final_rows) / alignment * alignment;
        if (rows == 0U) throw std::invalid_argument("ds41 prefill passes: no aligned split preserves the final suffix");
        passes.push_back({begin, rows});
        begin += rows;
    }
    passes.push_back({begin, total - begin});
    return passes;
}

PrefillRoutes build_prefill_routes(std::span<const std::uint32_t> ids, std::uint32_t rows, std::uint32_t top_k,
                                   std::uint32_t experts, std::uint32_t split, std::uint32_t batch_rows, std::uint32_t first,
                                   std::uint32_t end) {
    if (end == 0U) end = experts;
    if (first >= end || end > experts) throw std::invalid_argument("ds41 prefill routes: expert range");
    if (rows == 0U || top_k == 0U || experts == 0U || top_k > experts || (batch_rows != 8U && batch_rows != 16U && batch_rows != 32U) ||
        ids.size() != static_cast<std::size_t>(rows) * top_k)
        throw std::invalid_argument("ds41 prefill routes need rows * top_k expert ids and a batch of 8, 16 or 32 rows");
    PrefillRoutes r;
    r.rows = rows; r.top_k = top_k; r.experts = experts; r.batch_rows = batch_rows;
    std::vector<std::uint32_t> counts(experts, 0U);
    r.order.resize(ids.size());
    r.low_counts.resize(rows);
    for (std::uint32_t row = 0; row < rows; ++row) {
        std::vector<std::pair<std::uint32_t, std::uint32_t>> sorted; // (expert, slot)
        for (std::uint32_t k = 0; k < top_k; ++k) {
            const auto slot = row * top_k + k, expert = ids[slot];
            if (expert >= experts) throw std::invalid_argument("ds41 prefill routes: expert id out of range");
            for (const auto& [seen, unused] : sorted)
                if (seen == expert) throw std::invalid_argument("ds41 prefill routes: an expert repeats within a row");
            sorted.emplace_back(expert, slot);
            ++counts[expert];
        }
        std::sort(sorted.begin(), sorted.end());
        std::uint32_t low = 0;
        for (std::uint32_t k = 0; k < top_k; ++k) {
            r.order[row * top_k + k] = sorted[k].second;
            low += sorted[k].first < split ? 1U : 0U;
        }
        r.low_counts[row] = low;
    }
    const std::uint32_t owned = end - first;
    r.begins.assign(owned + 1U, 0U);
    for (std::uint32_t e = 0; e < owned; ++e) r.begins[e + 1U] = r.begins[e] + counts[first + e];
    r.entry_rows.resize(r.begins[owned]);
    r.entry_slots.resize(r.begins[owned]);
    std::vector<std::uint32_t> fill(r.begins.begin(), r.begins.end() - 1);
    for (std::uint32_t slot = 0; slot < ids.size(); ++slot) {
        if (ids[slot] < first || ids[slot] >= end) continue;
        const auto at = fill[ids[slot] - first]++;
        r.entry_rows[at] = slot / top_k;
        r.entry_slots[at] = slot;
    }
    for (std::uint32_t e = 0; e < owned; ++e)
        for (auto b = r.begins[e]; b < r.begins[e + 1U]; b += batch_rows) {
            r.batch_expert.push_back(e);
            r.batch_begin.push_back(b);
            r.batch_count.push_back(std::min(batch_rows, r.begins[e + 1U] - b));
        }
    return r;
}

namespace {
void require_geometry(const SuffixReplayGeometry& g) {
    if (g.layers == 0U || g.first_decoder >= g.layers || g.window == 0U || g.draft_layers > g.layers - g.first_decoder)
        throw std::invalid_argument("ds41 suffix replay geometry is inconsistent");
}
std::uint64_t clamp_down(std::int64_t value, std::uint64_t floor) {
    return value < static_cast<std::int64_t>(floor) ? floor : static_cast<std::uint64_t>(value);
}
} // namespace

SuffixReplayPlan plan_suffix_replay(const SuffixReplayGeometry& g, std::uint64_t cached, std::uint64_t total) {
    require_geometry(g);
    if (total <= cached) throw std::invalid_argument("ds41 suffix replay needs at least one new row");
    SuffixReplayPlan plan;
    plan.cached = cached; plan.total = total;
    plan.layers.resize(g.layers);
    const auto t = static_cast<std::int64_t>(total), w = static_cast<std::int64_t>(g.window);
    const auto draft_begin = g.draft_rows ? t - static_cast<std::int64_t>(g.draft_rows) : t;
    // Rows of each decoder layer's input needed, from the top: the last layer outputs the last row.
    std::int64_t out_begin = t - 1;
    std::int64_t earliest_kv = t;
    for (std::uint32_t layer = g.layers; layer-- > g.first_decoder;) {
        // Window K/V: the ring keeps the last `window` rows; every output row reads its window.
        const std::int64_t kv_begin = std::min(out_begin - (w - 1), t - w);
        earliest_kv = std::min(earliest_kv, kv_begin);
        auto& l = plan.layers[layer];
        l.layer = layer;
        l.out_begin = clamp_down(out_begin, cached);
        l.kv_begin = clamp_down(kv_begin, cached);
        // The layer input is needed on its output rows, its K/V rows and the drafter's seed rows.
        std::int64_t input_begin = std::min(out_begin, kv_begin);
        if (layer >= g.layers - g.draft_layers) input_begin = std::min(input_begin, draft_begin);
        out_begin = input_begin; // rows the layer below must output
    }
    plan.encoder_output_begin = clamp_down(out_begin, cached);
    for (std::uint32_t layer = 0; layer < g.first_decoder; ++layer) plan.layers[layer] = {layer, cached, cached};
    // Every layer that computes row `cached` reads the window - 1 rows before it from its ring.
    if (g.first_decoder > 0U) earliest_kv = std::min(earliest_kv, static_cast<std::int64_t>(cached) - (w - 1));
    const auto ring_begin = clamp_down(earliest_kv, 0U);
    plan.ring_rows = ring_begin < cached ? std::min<std::uint64_t>(cached - ring_begin, g.window) : 0U;
    plan.full_row_layers = (total - cached) * g.layers;
    for (const auto& l : plan.layers) plan.replay_row_layers += total - l.out_begin;
    return plan;
}

std::string verify_suffix_replay(const SuffixReplayGeometry& g, const SuffixReplayPlan& plan) {
    require_geometry(g);
    const auto total = plan.total, cached = plan.cached;
    if (plan.layers.size() != g.layers || total <= cached) return "plan shape";
    // needed_out[L][s]: layer L must produce its output for row s; needed_kv[L][s]: its window K/V.
    std::vector<std::vector<char>> out(g.layers, std::vector<char>(total, 0)), kv(g.layers, std::vector<char>(total, 0));
    const auto window = static_cast<std::uint64_t>(g.window);
    // Requirements of the final state: logits of the last row, every layer's ring (last `window`
    // rows), the KV-source caches (compressed KV and index keys of every new row are computed from
    // the first decoder layer's input, i.e. every encoder output), the drafter's seed rows.
    out[g.layers - 1U][total - 1U] = 1;
    for (std::uint32_t L = 0; L < g.layers; ++L)
        for (auto s = total > window ? total - window : 0U; s < total; ++s) kv[L][s] = 1;
    // Encoder layers feed caches of their own (compressed KV of their KV sources) for every row.
    for (std::uint32_t L = 0; L < g.first_decoder; ++L)
        for (auto s = cached; s < total; ++s) out[L][s] = 1;
    std::string problem;
    for (std::uint32_t L = g.layers; L-- > 0U;) {
        for (std::uint64_t s = 0; s < total; ++s)
            if (out[L][s])
                for (auto j = s + 1U > window ? s + 1U - window : 0U; j <= s; ++j) kv[L][j] = 1;
        if (L == 0U) break;
        for (std::uint64_t s = 0; s < total; ++s) {
            const bool draft = g.draft_rows && L >= g.layers - g.draft_layers && s + g.draft_rows >= total;
            if (!(out[L][s] || kv[L][s] || draft)) continue;
            if (s < cached) {
                // Before the new rows: only ring entries of the saved state can provide this.
                if (kv[L][s] && s + window < cached && problem.empty())
                    problem = "layer " + std::to_string(L) + " reads row " + std::to_string(s) + " outside the cached ring";
                continue;
            }
            out[L - 1U][s] = 1;
        }
    }
    for (std::uint32_t L = 0; L < g.layers; ++L) {
        const auto& l = plan.layers[L];
        for (std::uint64_t s = cached; s < total; ++s) {
            const bool plan_out = s >= l.out_begin, plan_kv = s >= l.kv_begin;
            if (plan_out != static_cast<bool>(out[L][s]) && problem.empty())
                problem = "layer " + std::to_string(L) + " row " + std::to_string(s) + (plan_out ? " computed but not needed" : " needed but not computed");
            if (plan_kv != static_cast<bool>(kv[L][s] || out[L][s]) && problem.empty())
                problem = "layer " + std::to_string(L) + " window K/V of row " + std::to_string(s) + (plan_kv ? " computed but not needed" : " needed but not computed");
        }
    }
    return problem;
}

SuffixReplayPlan plan_bounded_replay(const SuffixReplayGeometry& g, std::uint64_t cached, std::uint64_t total, std::uint32_t bound) {
    require_geometry(g);
    if (total <= cached) throw std::invalid_argument("ds41 bounded replay needs at least one new row");
    if (bound == 0U || bound % g.window != 0U) throw std::invalid_argument("ds41 bounded replay: the bound must be a positive multiple of the window");
    SuffixReplayPlan plan;
    plan.cached = cached; plan.total = total;
    plan.layers.resize(g.layers);
    if (g.draft_rows > bound)
        throw std::invalid_argument("ds41-decoder-replay-bounded-v1: draft_rows = " + std::to_string(g.draft_rows) +
                                    " exceeds R = " + std::to_string(bound));
    const std::uint64_t begin = std::max(cached, bounded_replay_key_floor(total, bound));
    for (std::uint32_t layer = 0; layer < g.layers; ++layer) {
        auto& l = plan.layers[layer];
        l.layer = layer;
        l.out_begin = l.kv_begin = layer < g.first_decoder ? cached : begin;
    }
    // the first decoder layer's input: every new row (its compressed KV and index keys; the engine keeps x0 = 0)
    plan.encoder_output_begin = cached;
    // a layer that computes row `cached` reads its ring only above its key floor: no ring row of the prefix when cut > cached
    plan.ring_rows = cached > 0U && begin == cached ? std::min<std::uint64_t>(cached, g.window - 1U) : 0U;
    plan.full_row_layers = (total - cached) * g.layers;
    for (const auto& l : plan.layers) plan.replay_row_layers += total - l.out_begin;
    return plan;
}

std::uint32_t moe_window_rows(std::uint32_t base, std::uint32_t chunk, std::uint32_t pass_rows, std::uint32_t cap) {
    if (base == 0U || chunk == 0U) throw std::invalid_argument("ds41 moe window: base and chunk rows must be positive");
    const std::uint64_t whole = (std::uint64_t{pass_rows} + chunk - 1U) / chunk * chunk;
    const std::uint64_t top = std::max<std::uint64_t>(cap / chunk * chunk, base);
    return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(whole, base, top));
}

bool moe_joint_layer(std::uint32_t peer_rows, std::uint32_t own_rows, std::uint32_t max_rows) {
    return max_rows != 0U && std::uint64_t{peer_rows} + own_rows <= max_rows;
}

} // namespace strix::ds41
