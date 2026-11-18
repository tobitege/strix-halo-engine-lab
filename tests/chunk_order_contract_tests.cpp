// Host reference for the chunk-invariant numeric order contract (moe_core::GroupedSplitContract,
// core::fixed_linear_contract). It emulates the declared K chain (S partitions, K16 boundaries floor(k16*s/S), each
// from +0 in ascending K, partials added in ascending s) for routed experts, runs the same rows in chunks of
// 256/1024/4096/whole and with 1..17 row tails, and requires identical bits under the bound contract. The adaptive
// policy (S from the expert's rows inside a chunk, route::gemm_splits) is the negative control: it must differ.
// The host checks the contract and its order; it does not emulate WMMA (the device fixture compares GPU bytes).
#include "core/linear_plan.hpp"
#include "moe_core/grouped_policy.hpp"
#include "minimax/engine/route_identity.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
namespace mc = strix::moe_core;
namespace route = strix::minimax::route;

constexpr unsigned kHidden = 6144, kInter = 1536, kBits = 4;  // MiniMax policy dims (per rank) (the split is chosen from them)
constexpr unsigned kK = 512, kOut = 4, kExperts = 32, kTop = 2; // emulated product size (order only)

void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}

std::uint32_t lcg(std::uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<std::uint32_t>(s >> 33);
}

// One output under the contract order.
float ordered_dot(const float* x, const float* w, unsigned k16, unsigned splits) {
    float total = 0.0F;
    for (unsigned s = 0; s < splits; ++s) {
        const unsigned b0 = static_cast<unsigned>(std::uint64_t{k16} * s / splits);
        const unsigned b1 = static_cast<unsigned>(std::uint64_t{k16} * (s + 1) / splits);
        volatile float part = 0.0F;  // +0 start, ascending K16 then ascending k inside the block
        for (unsigned k = b0 * 16; k < b1 * 16; ++k) part = part + x[k] * w[k];
        total = total + part;  // ascending partitions
    }
    return total;
}

struct Case {
    std::vector<float> x, w;
    std::vector<std::uint32_t> ids;
    unsigned rows = 0;
};

Case make_case(unsigned rows, std::uint64_t seed) {
    Case c;
    c.rows = rows;
    c.x.resize(std::size_t{rows} * kK);
    c.w.resize(std::size_t{kExperts} * kOut * kK);
    c.ids.resize(std::size_t{rows} * kTop);
    for (auto& v : c.x) v = static_cast<float>(static_cast<int>(lcg(seed) % 2001) - 1000) * 1.37e-3F;
    for (auto& v : c.w) v = static_cast<float>(static_cast<int>(lcg(seed) % 2001) - 1000) * 7.9e-4F;
    // Skewed routing: a few hot experts, a long tail of small ones (1..16-row experts appear in every chunk size).
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned k = 0; k < kTop; ++k) {
            const std::uint32_t u = lcg(seed) % 100;
            c.ids[std::size_t{r} * kTop + k] = u < 60 ? (u % 4) * kTop + k : 8 + (lcg(seed) % (kExperts - 8));
        }
    return c;
}

// Runs every chunk; returns [rows][top][out] bits.
std::vector<std::uint32_t> run(const Case& c, unsigned chunk, const mc::GroupedSplitContract& order) {
    std::vector<std::uint32_t> out(std::size_t{c.rows} * kTop * kOut);
    for (unsigned c0 = 0; c0 < c.rows; c0 += chunk) {
        const unsigned n = std::min(chunk, c.rows - c0);
        std::vector<unsigned> count(kExperts, 0);
        for (unsigned r = c0; r < c0 + n; ++r)
            for (unsigned k = 0; k < kTop; ++k) ++count[c.ids[std::size_t{r} * kTop + k]];
        for (unsigned r = c0; r < c0 + n; ++r)
            for (unsigned k = 0; k < kTop; ++k) {
                const unsigned e = c.ids[std::size_t{r} * kTop + k];
                const unsigned adaptive = route::gemm_splits(kHidden, kInter, kBits, count[e]);
                const unsigned s = std::min(order.gate.resolve(adaptive), kK / 16);
                for (unsigned o = 0; o < kOut; ++o) {
                    const float v = ordered_dot(&c.x[std::size_t{r} * kK], &c.w[(std::size_t{e} * kOut + o) * kK],
                                                kK / 16, s);
                    std::memcpy(&out[(std::size_t{r} * kTop + k) * kOut + o], &v, 4);
                }
            }
    }
    return out;
}

void contract_shape() {
    static_assert(mc::kGroupedSingleChainContract.invariant() && !mc::kGroupedAdaptiveContract.invariant());
    static_assert(mc::kGroupedAdaptiveContract.gate.resolve(8) == 8 && mc::kGroupedSingleChainContract.down.resolve(4) == 1);
    // S = 1 is today's product for large experts: gate/up >= 97 rows, down >= 17 rows, every MiniMax bit width.
    for (unsigned bits : {3U, 4U, 5U, 6U, 8U})
        for (unsigned n = 1; n <= 9000; ++n) {
            if (n >= 97) require(route::gemm_splits(kHidden, kInter, bits, n) == 1, "gate/up S=1 at >=97 rows");
            if (n >= 17) require(route::gemm_splits(kInter, kHidden, bits, n) == 1, "down S=1 at >=17 rows");
        }
    require(route::gemm_splits(kHidden, kInter, 4, 8) > 1, "adaptive splits small experts (control)");
    const auto lc = strix::core::fixed_linear_contract(3072, 1);
    require(lc.k16 == 192 && lc.splits == 1 && lc.boundary(0) == 0 && lc.boundary(1) == 192, "fixed linear contract");
    require(strix::core::fixed_linear_contract(64, 99).splits == 4, "fixed contract clamps to k16");
}

void chunk_invariance() {
    const Case c = make_case(9000, 7);
    const auto ref = run(c, c.rows, mc::kGroupedSingleChainContract);
    bool adaptive_differs = false;
    const auto adaptive_ref = run(c, c.rows, mc::kGroupedAdaptiveContract);
    for (unsigned chunk : {256U, 1024U, 4096U}) {
        require(run(c, chunk, mc::kGroupedSingleChainContract) == ref, "invariant contract changed bits with chunk");
        adaptive_differs |= run(c, chunk, mc::kGroupedAdaptiveContract) != adaptive_ref;
    }
    require(adaptive_differs, "negative control: adaptive split should depend on chunking");
    // Tails: 4096 + t rows, chunk 4096 (a t-row tail chunk) vs whole, and 1..17-row calls vs one row at a time.
    for (unsigned t = 1; t <= 17; ++t) {
        const Case d = make_case(4096 + t, 100 + t);
        require(run(d, 4096, mc::kGroupedSingleChainContract) == run(d, d.rows, mc::kGroupedSingleChainContract),
                "tail chunk changed bits");
        const Case e = make_case(t, 300 + t);
        require(run(e, 1, mc::kGroupedSingleChainContract) == run(e, t, mc::kGroupedSingleChainContract),
                "1..17-row call changed bits");
    }
}
// Dense q/k/v/o (per rank: 6144 -> 4096 / 256 / 256, 4096 -> 6144): the fixed contract takes no rows; the adaptive
// library split (route::gemm_splits) moves with the rows of a chunk (control). The engine keeps strix_gemm only where
// its split equals the fixed one (> 8 rows, S = 1), so the effective K chain is the same for every row count.
void dense_contract() {
    struct M { unsigned in, out; };
    const M mats[] = {{kHidden, 4096}, {kHidden, 256}, {4096, kHidden}};
    const unsigned rows[] = {1, 8, 9, 16, 17, 48, 49, 96, 97, 255, 256, 257, 1024, 4095, 4096, 4097, 8192};
    bool adaptive_moves = false;
    for (const M& m : mats)
        for (unsigned bits : {3U, 4U, 5U, 6U}) {
            const auto fixed = strix::core::fixed_linear_contract(static_cast<int>(m.in), 1);
            require(fixed.splits == 1 && fixed.k16 == static_cast<int>(m.in / 16), "dense fixed contract shape");
            const unsigned whole = route::gemm_splits(m.in, m.out, bits, 4096);
            for (unsigned r : rows) {
                const unsigned lib = route::gemm_splits(m.in, m.out, bits, r);
                adaptive_moves |= lib != whole;
                // Actual engine admission/fallback is exercised by tests/minimax_route_host, not inferred from S here.
            }
        }
    require(adaptive_moves, "negative control: adaptive dense split should depend on the rows of a chunk");
}
} // namespace

int main() {
    try {
        require(!mc::resolve_grouped_order(nullptr).invariant(), "unset mode must stay adaptive");
        require(!mc::resolve_grouped_order("").invariant(), "empty mode must stay adaptive");
        require(!mc::resolve_grouped_order("adaptive").invariant(), "adaptive mode must stay adaptive");
        require(mc::resolve_grouped_order("invariant").invariant(), "invariant mode must bind every matrix");
        bool invalid = false;
        try { (void)mc::resolve_grouped_order("typo"); }
        catch (const std::invalid_argument&) { invalid = true; }
        require(invalid, "invalid mode must be rejected");
        contract_shape();
        chunk_invariance();
        dense_contract();
        std::cout << "chunk order contract: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
