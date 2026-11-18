#pragma once
// The arithmetic route of every dense projection call of a prefill run, decided from the run's OWN solo geometry
// (engine-neutral, pure, host-testable).
//
// A dense call's bits depend on which arithmetic it runs (the batch kernel, the exact dense kernels, or a FAST form such
// as one FP16 WMMA chain) and on its K split (K width, K offset, independent K parts, accumulate into the previous
// part). They do NOT depend on the rows of the call, the CTA shape, the waves or the tail form: every output row is its
// own chain of products over K (the kernels document it). The only row-count dependence is the SELECTION: a selector
// that takes a form from N rows on (FAST dense from 128 rows, the dense entry from 256) gives different bits to a call
// of N - 1 rows. Work that re-shapes the rows of a call must therefore keep the route each row's solo run selected:
//   - a group pass (several prompts' rows concatenated) runs a site with the members' common route, never re-selected
//     from the summed rows; members whose solo routes differ at a site that both run cannot share the pass;
//   - a bounded replay (fewer rows than the prompt) pins the route the full prompt selected.
// DensePlan is that decision per (rank, layer, site); DenseRule is the selector a site applies to its rows.
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace strix::prefill_plan {

enum class DenseRoute : std::uint8_t {
    none = 0,            // the site launches no call (no rows on this rank / layer)
    batch_exact = 1,     // the exact batch kernel (the site's non-dense entry)
    dense_exact = 2,     // the exact dense kernels (their tall / CTA forms: the same bits)
    dense_fast = 3,      // the FAST dense form (e.g. ds41-dense-f16-v1: one FP16 WMMA chain per output)
};

[[nodiscard]] constexpr const char* route_name(DenseRoute r) noexcept {
    switch (r) {
    case DenseRoute::none: return "none";
    case DenseRoute::batch_exact: return "batch_exact";
    case DenseRoute::dense_exact: return "dense_exact";
    case DenseRoute::dense_fast: return "dense_q8a8_f16";
    }
    return "?";
}

// The selector of one call site and the K split of its calls (constant for a site; the rows vary).
struct DenseRule {
    bool dense = true;                 // the site has a dense entry (off: every call takes the batch kernel)
    std::uint32_t dense_min_rows = 0;  // calls of fewer rows take the batch kernel
    bool fast = false;                 // the FAST dense form is on for this site
    std::uint32_t fast_min_rows = 0;   // calls of fewer rows keep the exact dense kernels
    bool fast_supported = true;        // the site's shape is one the FAST form takes (row-independent)
    std::uint32_t k = 0, k_offset = 0; // K width and K offset of the call in the weight row
    std::uint8_t k_parts = 1;          // independent K parts (each its own chain, summed in order)
    bool accumulate = false;           // the call adds to the previous part's output

    [[nodiscard]] constexpr DenseRoute route(std::uint32_t rows) const noexcept {
        if (rows == 0U) return DenseRoute::none;
        if (!dense || rows < dense_min_rows) return DenseRoute::batch_exact;
        return fast && fast_supported && rows >= fast_min_rows ? DenseRoute::dense_fast : DenseRoute::dense_exact;
    }
};

// One site of a solo run: its route, its K split, and its solo geometry (the rows the solo call launches).
struct DenseSite {
    DenseRoute route = DenseRoute::none;
    std::uint32_t k = 0, k_offset = 0;
    std::uint8_t k_parts = 1;
    bool accumulate = false;
    std::uint32_t rows = 0;
    // the route an empty site is compared with: its rule's route of one row (an empty site never constrains a group)
    DenseRoute empty_as = DenseRoute::none;
};

struct DensePlan {
    std::uint32_t ranks = 0, layers = 0, sites = 0;
    std::vector<DenseSite> site;  // [rank][layer][site]
    [[nodiscard]] std::size_t index(std::uint32_t rank, std::uint32_t layer, std::uint32_t s) const noexcept {
        return (std::size_t{rank} * layers + layer) * sites + s;
    }
    [[nodiscard]] const DenseSite& at(std::uint32_t rank, std::uint32_t layer, std::uint32_t s) const { return site.at(index(rank, layer, s)); }
    DenseSite& at(std::uint32_t rank, std::uint32_t layer, std::uint32_t s) { return site.at(index(rank, layer, s)); }
};

// The site of a call of `rows` rows under `rule`.
[[nodiscard]] constexpr DenseSite make_site(const DenseRule& rule, std::uint32_t rows) noexcept {
    DenseSite s;
    s.route = rule.route(rows);
    s.k = rule.k, s.k_offset = rule.k_offset, s.k_parts = rule.k_parts, s.accumulate = rule.accumulate;
    s.rows = rows;
    s.empty_as = rule.route(1U);
    return s;
}

// The arithmetic signature of a plan: per site its route (an empty site as empty_as) and K split; never the rows,
// CTA shape or waves. Two runs whose signatures are equal take the same arithmetic at every site either of them runs.
[[nodiscard]] inline std::vector<std::uint32_t> signature(const DensePlan& p) {
    std::vector<std::uint32_t> out;
    out.reserve(4U + p.site.size() * 3U);
    out.push_back(p.ranks), out.push_back(p.layers), out.push_back(p.sites);
    for (const auto& s : p.site) {
        const auto r = s.route == DenseRoute::none ? s.empty_as : s.route;
        out.push_back(static_cast<std::uint32_t>(r) | (std::uint32_t{s.k_parts} << 8) | (s.accumulate ? 1U << 16 : 0U));
        out.push_back(s.k);
        out.push_back(s.k_offset);
    }
    return out;
}

// Canonical cross-owner/rank identity of ordered member plans, independent of local class-table ordinals.
// FNV-1a over explicitly ordered bytes of the arithmetic signatures (not native object representations).
[[nodiscard]] inline std::uint64_t fingerprint(std::span<const DensePlan> plans) {
    std::uint64_t h = 14695981039346656037ULL;
    const auto word = [&h](std::uint32_t v) {
        for (unsigned shift = 0; shift < 32U; shift += 8U) h = (h ^ ((v >> shift) & 0xFFU)) * 1099511628211ULL;
    };
    const auto count = static_cast<std::uint64_t>(plans.size());
    word(static_cast<std::uint32_t>(count));
    word(static_cast<std::uint32_t>(count >> 32));
    for (const auto& p : plans)
        for (const auto v : signature(p)) word(v);
    return h;
}

// The common route of the members at (rank, layer, site): `none` when no member runs it. `mismatch` is set when two
// members that both run the site took different routes or K splits (they cannot share one call of that site).
[[nodiscard]] inline DenseRoute common_route(std::span<const DensePlan* const> members, std::uint32_t rank, std::uint32_t layer,
                                             std::uint32_t s, bool& mismatch) {
    const DenseSite* first = nullptr;
    for (const auto* m : members) {
        const auto& x = m->at(rank, layer, s);
        if (x.route == DenseRoute::none) continue;
        if (first == nullptr) { first = &x; continue; }
        if (x.route != first->route || x.k != first->k || x.k_offset != first->k_offset || x.k_parts != first->k_parts ||
            x.accumulate != first->accumulate)
            mismatch = true;
    }
    return first != nullptr ? first->route : DenseRoute::none;
}

// Interns signatures: equal signatures get the same small class number (exact comparison, no hash).
struct DenseClassTable {
    std::vector<std::vector<std::uint32_t>> classes;
    [[nodiscard]] std::uint32_t intern(std::vector<std::uint32_t> sig) {
        for (std::size_t i = 0; i < classes.size(); ++i)
            if (classes[i] == sig) return static_cast<std::uint32_t>(i);
        classes.push_back(std::move(sig));
        return static_cast<std::uint32_t>(classes.size() - 1U);
    }
};

}  // namespace strix::prefill_plan
