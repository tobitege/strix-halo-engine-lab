#pragma once
// DeepSeek V4.1 prefill: the dense route of every projection site (q_a, kv, q_b, o_a, o_b) of a FRESH solo run, per rank
// and layer, from the run's own geometry (rows, window ownership of the row split, the cone of the decoder layers, the
// attention chunk) and the resolved selectors. Pure (no device): the group pass (DevicePrefill::run_group) and the
// bounded decoder replay share it, so neither re-selects a route from rows it reshaped (summed members / compacted rows).
// See strix/prefill_dense_plan.hpp for the arithmetic argument.
#include "strix/prefill_dense_plan.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace strix::ds41 {

enum DenseSiteKind : std::uint32_t { kDenseQa = 0, kDenseKv, kDenseQb, kDenseOa, kDenseOb, kDenseSites };

inline constexpr std::uint32_t kGroupContractVersion = 3;  // 3: canonical dense-plan fingerprint in the rank handshake
// The group FAST identity fixture (each member of a FAST group == its solo FAST run, logits and state, both ranks) passed
// on both nodes: FAST serving groups by default (serve::qualify_fast). false: FAST groups only with STRIX_SERVE_PREFILL_GROUP=auto.
inline constexpr bool kGroupFastQualified = true;  // 11-15: identity fixture PASS (6 arms, both ranks), C8 +2.8 %

struct DenseConeRows {
    std::uint32_t k0 = 0, o0 = 0;  // first global row of the layer's window K/V and of its outputs (multiples of W)
};

struct DenseRunGeometry {
    std::uint32_t rows = 0;          // rows of the fresh run (positions 0 .. rows - 1)
    std::uint32_t window_rows = 8;   // W: attention window rows; the row split gives window j to rank j % 2
    std::uint32_t chunk_rows = 0;    // C: rows per attention chunk (q_b / o_a / o_b calls); 0 = unbounded
    std::uint32_t world = 1;
    bool split = false;              // world 2: the row split (else the tensor-parallel form: every rank all rows)
    std::uint32_t layers = 0;
    std::span<const DenseConeRows> cone;  // per layer (empty: no cone, every row in every layer)
};

// Rows [r0, rows) that rank `rank` runs (r0 a multiple of W under the split).
[[nodiscard]] inline std::uint32_t dense_local_rows(const DenseRunGeometry& g, std::uint32_t rank, std::uint32_t r0) noexcept {
    if (r0 >= g.rows) return 0U;
    if (g.world < 2U || !g.split) return rank < std::max(g.world, 1U) ? g.rows - r0 : 0U;
    const std::uint32_t W = std::max(g.window_rows, 1U);
    std::uint32_t n = 0;
    for (std::uint32_t j = r0 / W; std::uint64_t{j} * W < g.rows; ++j)
        if (j % 2U == rank) n += std::min(W, g.rows - j * W);
    return n;
}

[[nodiscard]] inline bool dense_multi_chunk(const DenseRunGeometry& g) noexcept {
    return g.chunk_rows != 0U && g.rows > g.chunk_rows;
}

// rules: [rank][site] (world x kDenseSites). Only single-chunk runs are represented; reject larger runs rather than
// returning an incomplete descriptor. This conservative bound also applies to split/cone runs.
[[nodiscard]] inline prefill_plan::DensePlan plan_dense(const DenseRunGeometry& g, std::span<const prefill_plan::DenseRule> rules) {
    if (dense_multi_chunk(g)) throw std::invalid_argument("plan_dense: multi-chunk runs are not represented");
    const std::uint32_t ranks = std::max(g.world, 1U);
    if (rules.size() != std::size_t{ranks} * kDenseSites) throw std::invalid_argument("plan_dense: rules are [rank][site]");
    if (!g.cone.empty() && g.cone.size() < g.layers) throw std::invalid_argument("plan_dense: cone rows per layer");
    prefill_plan::DensePlan p;
    p.ranks = ranks, p.layers = g.layers, p.sites = kDenseSites;
    p.site.resize(std::size_t{ranks} * g.layers * kDenseSites);
    for (std::uint32_t r = 0; r < ranks; ++r)
        for (std::uint32_t l = 0; l < g.layers; ++l) {
            const DenseConeRows c = g.cone.empty() ? DenseConeRows{} : g.cone[l];
            const std::uint32_t out_rows = dense_local_rows(g, r, c.o0), kv_rows = dense_local_rows(g, r, c.k0);
            const std::uint32_t chunk = g.chunk_rows != 0U ? std::min(out_rows, g.chunk_rows) : out_rows;
            const std::array<std::uint32_t, kDenseSites> rows{out_rows, kv_rows, chunk, chunk, chunk};
            for (std::uint32_t s = 0; s < kDenseSites; ++s) p.at(r, l, s) = prefill_plan::make_site(rules[std::size_t{r} * kDenseSites + s], rows[s]);
        }
    return p;
}

}  // namespace strix::ds41
