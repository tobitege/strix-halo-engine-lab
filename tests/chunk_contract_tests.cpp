// Host reference for src/core/chunk_contract.hpp: segmented attention and an open-block recurrence give identical bits
// for any chunk size, row grouping and prefix-cache split under the contract; the old forms are negative controls.
#include "core/chunk_contract.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace cc = strix::chunk_contract;
namespace {
int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

float score(unsigned q, unsigned k) { return std::sin(0.37F * float(q) + 1.13F * float(k)) * 4.0F; }
float value(unsigned k) { return std::cos(0.71F * float(k)) + 0.001F * float(k); }

struct Part { float m, l, o; };
// One segment of one query, 16-key tiles ascending, keys > q masked (fully masked tile = no-op).
Part run_segment(unsigned q, cc::Segment s) {
    Part p{-INFINITY, 0.0F, 0.0F};
    for (unsigned t = s.tile_begin; t < s.tile_end; ++t) {
        float bm = -INFINITY;
        for (unsigned k = t * 16U; k < t * 16U + 16U; ++k) if (k <= q) bm = std::fmax(bm, score(q, k));
        if (bm == -INFINITY) continue;
        const float mn = std::fmax(bm, p.m), a = std::exp2(p.m - mn);
        float ps = 0.0F, pv = 0.0F;
        for (unsigned k = t * 16U; k < t * 16U + 16U; ++k)
            if (k <= q) { const float e = std::exp2(score(q, k) - mn); ps += e; pv += e * value(k); }
        p = {mn, p.l * a + ps, p.o * a + pv};
    }
    return p;
}
float combine(const std::vector<Part>& parts) {  // ascending z, as dense_flash_prefill.hpp:279-284
    float m = -INFINITY, l = 0.0F, o = 0.0F;
    for (const Part& p : parts) m = std::fmax(m, p.m);
    for (const Part& p : parts) if (p.m != -INFINITY) { const float w = std::exp2(p.m - m); l += p.l * w; o += p.o * w; }
    return o / l;
}
// Rows [0, n) in calls of `chunk` rows, each call in row blocks of `block_rows`; returns per-row output bits.
std::vector<std::uint32_t> attention(unsigned n, unsigned chunk, unsigned block_rows, unsigned splits, bool absolute) {
    std::vector<std::uint32_t> out(n);
    for (unsigned c0 = 0; c0 < n; c0 += chunk)
        for (unsigned b0 = c0; b0 < std::min(n, c0 + chunk); b0 += block_rows) {
            const unsigned b1 = std::min({n, c0 + chunk, b0 + block_rows});
            for (unsigned q = b0; q < b1; ++q) {
                std::vector<Part> parts;
                if (absolute) {
                    const unsigned seg = 4U;  // 64 keys per segment
                    for (unsigned s = 0; s < cc::absolute_segment_count(q / 16U, seg); ++s)
                        parts.push_back(run_segment(q, cc::absolute_segment(0, q / 16U, seg, s)));
                } else {
                    for (unsigned z = 0; z < splits; ++z) parts.push_back(run_segment(q, cc::block_share_segment(0, (b1 - 1U) / 16U, splits, z)));
                }
                const float r = combine(parts);
                std::memcpy(&out[q], &r, 4);
            }
        }
    return out;
}

// Block recurrence (stand-in for GDN WY C32 / KDA B32): per block, a pairwise tree sum of the block's tokens, then
// state = state * 0.75 + tree; the output of row r = state at block start + ascending prefix inside the block.
// A partial block is closed with the same formula over its m rows, which is why closing per call changes bits.
float tok(std::uint64_t i) { return std::sin(0.013F * float(i)) * 1.7F + 0.3F; }
float tree(const float* x, unsigned m) {
    std::vector<float> v(x, x + m);
    while (v.size() > 1) { std::vector<float> w; for (std::size_t i = 0; i + 1 < v.size(); i += 2) w.push_back(v[i] + v[i + 1]); if (v.size() & 1U) w.push_back(v.back()); v = w; }
    return v.empty() ? 0.0F : v[0];
}
struct Rec { float state = 0.0F; std::vector<std::uint32_t> rows; };
// Process tokens [b, e) from `state`, blocks of B counted from b (as the kernels count from the call start).
float run_rows(float state, std::uint64_t b, std::uint64_t e, unsigned B, std::vector<std::uint32_t>& rows) {
    for (std::uint64_t k0 = b; k0 < e; k0 += B) {
        const unsigned m = unsigned(std::min<std::uint64_t>(B, e - k0));
        std::vector<float> x(m);
        float acc = state;
        for (unsigned i = 0; i < m; ++i) { x[i] = tok(k0 + i); acc += x[i]; std::memcpy(&rows[k0 + i], &acc, 4); }
        state = state * 0.75F + tree(x.data(), m);
    }
    return state;
}
// cuts: absolute call ends (last = n). planned: chunk_contract plan_call; else every call closes its tail.
Rec recurrence(std::uint64_t n, const std::vector<std::uint64_t>& cuts, unsigned B, bool planned) {
    Rec r; r.rows.assign(n, 0);
    std::uint64_t committed = 0;
    for (std::size_t i = 0; i < cuts.size(); ++i) {
        const bool final = i + 1 == cuts.size();
        if (!planned) { r.state = run_rows(r.state, committed, cuts[i], B, r.rows); committed = cuts[i]; continue; }
        const cc::CallPlan p = cc::plan_call(committed, cuts[i], B, final);
        r.state = run_rows(r.state, p.run_begin, p.commit_end, B, r.rows);  // committed state
        if (p.carry_tokens) { float scratch = r.state; (void)run_rows(scratch, p.commit_end, cuts[i], B, r.rows); }  // tail rows, state not kept
        CHECK(final || cc::cacheable_boundary(p.commit_end, B));
        committed = p.commit_end;
    }
    return r;
}
std::uint32_t bits(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }
}  // namespace

int main() {
    // Attention: chunks 256/1024/whole and row blocks 64/128 against whole, 64-row blocks.
    const unsigned n = 1100;
    const auto ref = attention(n, n, 64, 4, true);
    for (unsigned chunk : {256U, 1024U, 33U, 15U, n})
        for (unsigned br : {16U, 64U, 128U}) CHECK(attention(n, chunk, br, 4, true) == ref);
    bool block_share_moves = false;
    const auto old_ref = attention(n, n, 128, 4, false);
    for (unsigned chunk : {256U, 1024U}) for (unsigned br : {64U, 128U}) block_share_moves |= attention(n, chunk, br, 4, false) != old_ref;
    CHECK(block_share_moves);  // negative control: block-relative split boundaries change bits
    CHECK(attention(n, 256, 64, 1, false) == attention(n, n, 128, 1, false));  // splits == 1: already invariant

    // Recurrence: chunks 256/1024/4096/whole, interior cuts, prefix-cache splits 1/15/31/33.
    const std::uint64_t L = 5000; const unsigned B = 32;
    const Rec whole = recurrence(L, {L}, B, true);
    std::vector<std::vector<std::uint64_t>> plans;
    for (std::uint64_t c : {256ULL, 1024ULL, 4096ULL}) { std::vector<std::uint64_t> v; for (std::uint64_t e = c; e < L; e += c) v.push_back(e); v.push_back(L); plans.push_back(v); }
    plans.push_back({17, 100, 1001, 3333, L});
    for (std::uint64_t pre : {1ULL, 15ULL, 31ULL, 33ULL}) plans.push_back({pre, L});
    bool naive_moves = false;
    for (const auto& cuts : plans) {
        const Rec r = recurrence(L, cuts, B, true);
        CHECK(bits(r.state) == bits(whole.state)); CHECK(r.rows == whole.rows);
        const Rec w = recurrence(L, cuts, B, false);
        naive_moves |= bits(w.state) != bits(whole.state) || w.rows != whole.rows;
    }
    CHECK(naive_moves);  // negative control: closing the tail per call changes bits
    // plan arithmetic and identity
    CHECK(cc::plan_call(0, 33, 32, false).commit_end == 32 && cc::plan_call(0, 33, 32, false).carry_tokens == 1);
    CHECK(cc::plan_call(32, 40, 32, false).commit_end == 32);
    CHECK(cc::plan_call(32, 40, 32, true).commit_end == 40);
    // Section 4: the WMMA attention open row (odd call end from an even start) is carried; even ends commit all.
    CHECK(cc::plan_call(0, 15, cc::kWmmaKPairRows, false).commit_end == 14 && cc::plan_call(0, 15, cc::kWmmaKPairRows, false).carry_tokens == 1);
    CHECK(cc::plan_call(0, 1, cc::kWmmaKPairRows, false).commit_end == 0 && cc::plan_call(14, 526, cc::kWmmaKPairRows, false).carry_tokens == 0);
    static_assert(32 % cc::kWmmaKPairRows == 0 && 16 % cc::kWmmaKPairRows == 0);
    CHECK(cc::chunk_contract_tag("gdn-c32-r3", 32, 4) == "gdn-c32-r3+chunkinv-attn-state-v1/b32/s4");
    CHECK(cc::absolute_segment(5, 9, 4, 1).tile_begin == 5 && cc::absolute_segment(5, 9, 4, 1).tile_end == 8);
    {   // A final tail may not be split into independently closed partial blocks.
        const std::vector<std::uint64_t> ok{4096, 8192, 8256, 8272}, bad{4096, 8192, 8208, 8224, 8229}, tail{8192, 8224, 8240, 8245};
        CHECK(cc::block_plan_canonical(ok, 0, 32)); CHECK(!cc::block_plan_canonical(bad, 0, 32)); CHECK(!cc::block_plan_canonical(tail, 0, 32));
        CHECK(!cc::block_plan_canonical(std::vector<std::uint64_t>{1, 2}, 0, 32));
        CHECK(cc::block_plan_canonical(std::vector<std::uint64_t>{33}, 0, 32));
        CHECK(!cc::block_plan_canonical(std::vector<std::uint64_t>{32}, 1, 32));
        CHECK(!cc::block_plan_canonical(std::vector<std::uint64_t>{32, 32}, 0, 32));
        CHECK(!cc::block_plan_canonical(std::vector<std::uint64_t>{64, 32}, 0, 32));
        CHECK(!cc::block_plan_canonical(std::vector<std::uint64_t>{1}, 0, 0));
        CHECK(cc::block_plan_canonical(std::vector<std::uint64_t>{1, 2}, 0, 1));
        const auto split = recurrence(2, {1, 2}, 32, false), full = recurrence(2, {2}, 32, false);
        CHECK(bits(split.state) != bits(full.state)); CHECK(split.rows == full.rows); // rows alone miss the state mismatch
        for (const auto& cuts : {ok, std::vector<std::uint64_t>{33}}) {
            const auto r = recurrence(cuts.back(), cuts, 32, false);
            const auto reference = recurrence(cuts.back(), {cuts.back()}, 32, false);
            CHECK(r.rows == reference.rows); CHECK(bits(r.state) == bits(reference.state));
        }
    }
    // Exhaust every partition of short prompts, including a nonzero aligned start and initial state.
    for (std::uint64_t begin : {0ULL, 4ULL}) for (unsigned count = 1; count <= 12; ++count) {
        const auto end = begin + count;
        std::vector<std::uint32_t> reference(static_cast<std::size_t>(end));
        const float state = run_rows(0.25F, begin, end, 4, reference);
        for (unsigned mask = 0; mask < (1U << (count - 1U)); ++mask) {
            std::vector<std::uint64_t> cuts;
            for (unsigned i = 1; i < count; ++i) if (mask & (1U << (i - 1U))) cuts.push_back(begin + i);
            cuts.push_back(end);
            if (!cc::block_plan_canonical(cuts, begin, 4)) continue;
            std::vector<std::uint32_t> rows(static_cast<std::size_t>(end));
            float actual = 0.25F;
            auto last = begin;
            for (const auto e : cuts) { actual = run_rows(actual, last, e, 4, rows); last = e; }
            CHECK(rows == reference); CHECK(bits(actual) == bits(state));
        }
    }
    std::printf(g_fail ? "chunk_contract_tests: %d failures\n" : "chunk_contract_tests: ok\n", g_fail);
    return g_fail ? 1 : 0;
}
