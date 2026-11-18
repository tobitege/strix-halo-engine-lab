// Host test of the cohort window packing (include/strix/cuda/qwen_cohort_plan.hpp); no GPU.
#include "strix/cuda/qwen_cohort_plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace qc = strix::cuda::qwen_cohort;

static int failures = 0;
#define EXPECT(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

static std::vector<std::uint32_t> rows_of(const std::vector<qc::Window>& ws, std::size_t n) {
    std::vector<std::uint32_t> r(n, 0);
    for (const auto& w : ws) for (const auto& e : w.entries) r[e.index] += e.rows;
    return r;
}

// Invariants of any packing: every session exactly once, 1 <= rows <= max(want, 1), window rows <= budget, windows in input
// order, session-major, and no spare row left while some session in that window still wants more.
static void check_invariants(const std::vector<std::uint32_t>& want, std::uint32_t budget) {
    const auto ws = qc::pack(want, budget);
    std::vector<int> seen(want.size(), 0);
    std::uint32_t expect_index = 0;
    for (const auto& w : ws) {
        std::uint32_t sum = 0;
        bool wants_more = false;
        for (const auto& e : w.entries) {
            EXPECT(e.index == expect_index);
            ++expect_index;
            ++seen[e.index];
            EXPECT(e.rows >= 1);
            EXPECT(e.rows <= std::max<std::uint32_t>(want[e.index], 1U));
            wants_more = wants_more || e.rows < std::max<std::uint32_t>(want[e.index], 1U);
            sum += e.rows;
        }
        EXPECT(sum == w.rows);
        EXPECT(w.rows <= budget);
        EXPECT(w.entries.size() <= qc::kMaxWindowSessions);
        if (wants_more) EXPECT(w.rows == budget);
    }
    for (const int s : seen) EXPECT(s == 1);
    // fairness: inside a window, two sessions that both want more than they got differ by at most one row
    for (const auto& w : ws)
        for (const auto& a : w.entries)
            for (const auto& b : w.entries)
                if (a.rows < want[a.index] && b.rows < want[b.index]) EXPECT(a.rows + 1 >= b.rows && b.rows + 1 >= a.rows);
}

int main() {
    // single session: the whole window
    {
        const auto ws = qc::pack({8});
        EXPECT(ws.size() == 1 && ws[0].rows == 8 && ws[0].entries[0].rows == 8);
        const auto w3 = qc::pack({3});
        EXPECT(w3.size() == 1 && w3[0].rows == 3);
    }
    // two sessions both drafting 7: 4 + 4
    {
        const auto r = rows_of(qc::pack({8, 8}), 2);
        EXPECT(r[0] == 4 && r[1] == 4);
    }
    // three sessions: 3 + 3 + 2 (round-robin, lowest index first)
    {
        const auto r = rows_of(qc::pack({8, 8, 8}), 3);
        EXPECT(r[0] == 3 && r[1] == 3 && r[2] == 2);
    }
    // a short chain frees rows for the others
    {
        const auto r = rows_of(qc::pack({1, 8, 8}), 3);
        EXPECT(r[0] == 1 && r[1] == 4 && r[2] == 3);
        const auto r2 = rows_of(qc::pack({2, 2, 8, 1}), 4);
        EXPECT(r2[0] == 2 && r2[1] == 2 && r2[2] == 3 && r2[3] == 1);
    }
    // eight sessions: one row each, one window
    {
        const auto ws = qc::pack(std::vector<std::uint32_t>(8, 8));
        EXPECT(ws.size() == 1 && ws[0].rows == 8);
        for (const auto& e : ws[0].entries) EXPECT(e.rows == 1);
    }
    // nine sessions: two windows of 5 + 4 sessions, nobody dropped
    {
        const auto ws = qc::pack(std::vector<std::uint32_t>(9, 8));
        EXPECT(ws.size() == 2 && ws[0].entries.size() == 5 && ws[1].entries.size() == 4);
        EXPECT(ws[0].rows == 8 && ws[1].rows == 8);
        const auto sizes = qc::window_sizes(17);
        EXPECT(sizes.size() == 3 && sizes[0] == 6 && sizes[1] == 6 && sizes[2] == 5);
        EXPECT(qc::window_of(0, 9) == 5 && qc::window_of(5, 9) == 4 && qc::window_of(8, 9) == 4);
    }
    // want 0 is one row; empty input is no window
    {
        const auto r = rows_of(qc::pack({0, 0}), 2);
        EXPECT(r[0] == 1 && r[1] == 1);
        EXPECT(qc::pack({}).empty());
    }
    // draft depth caps
    {
        using qc::DepthPolicy;
        EXPECT(qc::draft_depth_cap(1, 7) == 7);
        EXPECT(qc::draft_depth_cap(2, 7) == 3);
        EXPECT(qc::draft_depth_cap(3, 7) == 2);
        EXPECT(qc::draft_depth_cap(4, 7) == 1);
        EXPECT(qc::draft_depth_cap(5, 7) == 1);
        EXPECT(qc::draft_depth_cap(8, 7) == 0);
        EXPECT(qc::draft_depth_cap(2, 7, DepthPolicy::max) == 6);
        EXPECT(qc::draft_depth_cap(1, 3) == 3);
        EXPECT(qc::depth_policy_of("max") == DepthPolicy::max && qc::depth_policy_of("fair") == DepthPolicy::fair);
    }
    // 16-row windows (STRIX_QRTX_WINDOW_ROWS=16): C8 with one draft each in one window; at most 8 sessions per window
    {
        const auto ws = qc::pack(std::vector<std::uint32_t>(8, 2), 16);
        EXPECT(ws.size() == 1 && ws[0].rows == 16);
        for (const auto& e : ws[0].entries) EXPECT(e.rows == 2);
        const auto w10 = qc::pack(std::vector<std::uint32_t>(10, 1), 16);
        EXPECT(w10.size() == 2 && w10[0].entries.size() == 5 && w10[1].entries.size() == 5);
        const auto r4 = rows_of(qc::pack({8, 8, 8, 8}, 16), 4);
        EXPECT(r4[0] == 4 && r4[1] == 4 && r4[2] == 4 && r4[3] == 4);
        EXPECT(qc::draft_depth_cap(8, 7, qc::DepthPolicy::fair, 16) == 1);
        EXPECT(qc::draft_depth_cap(4, 7, qc::DepthPolicy::fair, 16) == 3);
        EXPECT(qc::window_of(9, 12, 16) == 6);
    }
    // marginal rule (choose_budget): with the c12a model a 16-row budget never pays; with a hypothetical cheap wide
    // window (40 ms) it pays at C8 (no spare row at 8) but not at C2 / C4 (the 8-row windows already hold the drafts)
    {
        for (std::uint32_t c = 1; c <= 16; ++c) EXPECT(qc::choose_budget(c, 7, qc::DepthPolicy::fair, 16) == 8);
        qc::WideModel cheap;
        cheap.wide_ms = 40.0;
        EXPECT(qc::choose_budget(8, 7, qc::DepthPolicy::fair, 16, cheap) == 16);
        EXPECT(qc::choose_budget(2, 7, qc::DepthPolicy::fair, 16, cheap) == 8);
        // an ideal real 16-row decode form at the 8-row slope (a + b * 16 = 55.5 ms) pays nowhere: C2 / C4 / C8 stay at 8
        qc::WideModel linear;
        linear.wide_ms = linear.a_ms + linear.b_ms * 16.0;
        for (std::uint32_t c : {2U, 4U, 6U, 8U}) EXPECT(qc::choose_budget(c, 7, qc::DepthPolicy::fair, 16, linear) == 8);
        EXPECT(qc::choose_budget(8, 0, qc::DepthPolicy::fair, 16, cheap) == 8);   // no drafts: nothing to widen for
        EXPECT(qc::choose_budget(8, 7, qc::DepthPolicy::fair, 8, cheap) == 8);
        EXPECT(qc::expected_tokens(1, 0.65) == 1.0);
        // break-even of a real 16-row form at C8 (p 0.65): wide_ms must be below ~(8 * 1.65) / (8 / 33) / 1.02 = 53.4
        qc::WideModel edge;
        edge.wide_ms = 52.0;
        EXPECT(qc::choose_budget(8, 7, qc::DepthPolicy::fair, 16, edge) == 16);
        edge.wide_ms = 55.0;
        EXPECT(qc::choose_budget(8, 7, qc::DepthPolicy::fair, 16, edge) == 8);
    }
    // cap-16 cohort table (wide_measured, 50 ms) with the live acceptance: high-acceptance classes widen at C4 / C8 (edit
    // p ~0.95, json p ~0.87), low acceptance (prose p ~0.5) stays at 8; the 0.65 prior (no live p) widens at C8 only
    {
        auto at = [](double p) { qc::WideModel m = qc::wide_measured(16); m.accept = p; return m; };
        EXPECT(qc::choose_budget(8, 7, qc::DepthPolicy::fair, 16, at(0.95)) == 16);
        EXPECT(qc::choose_budget(4, 7, qc::DepthPolicy::fair, 16, at(0.95)) == 16);
        EXPECT(qc::choose_budget(8, 7, qc::DepthPolicy::fair, 16, at(0.87)) == 16);
        EXPECT(qc::choose_budget(4, 7, qc::DepthPolicy::fair, 16, at(0.87)) == 16);
        EXPECT(qc::choose_budget(8, 7, qc::DepthPolicy::fair, 16, at(0.65)) == 16);
        EXPECT(qc::choose_budget(4, 7, qc::DepthPolicy::fair, 16, at(0.65)) == 8);
        EXPECT(qc::choose_budget(8, 7, qc::DepthPolicy::fair, 16, at(0.5)) == 8);
        EXPECT(qc::choose_budget(4, 7, qc::DepthPolicy::fair, 16, at(0.5)) == 8);
        EXPECT(qc::choose_budget(1, 7, qc::DepthPolicy::fair, 16, at(0.95)) == 8);  // one session: copy windows decide
        EXPECT(qc::choose_budget(8, 7, qc::DepthPolicy::fair, 16, qc::wide_measured(8)) == 8);
    }
    // randomized invariants
    std::mt19937 rng(12345);
    for (int t = 0; t < 20000; ++t) {
        const std::uint32_t n = rng() % 20;
        const std::uint32_t budget = 1 + rng() % 16;
        std::vector<std::uint32_t> want(n);
        for (auto& w : want) w = rng() % 10;
        check_invariants(want, budget);
    }
    // wide copy window rule: measured sm_120 costs (wide 62.5 ms) -> off; a decode-shaped wide form (26 ms) -> on at 16
    // rows, still off at 9 rows (9 / 26 < 8 / 17.35); <= 8 rows never
    {
        const qc::CopyWideModel measured{}, decode16{8.55, 1.1, 26.0};
        EXPECT(!qc::copy_wide_pays(16, measured));
        EXPECT(!qc::copy_wide_pays(8, decode16));
        EXPECT(qc::copy_wide_pays(16, decode16));
        EXPECT(!qc::copy_wide_pays(9, decode16));
        // the measured table by decode row cap: cap 8 -> 62.5 off, cap 16 -> 29.2 on at 16 only
        EXPECT(!qc::copy_wide_pays(16, qc::copy_wide_measured(8)));
        EXPECT(qc::copy_wide_pays(16, qc::copy_wide_measured(16)));
        EXPECT(!qc::copy_wide_pays(12, qc::copy_wide_measured(16)));
    }
    std::printf("%s (%d failures)\n", failures == 0 ? "COHORT PLAN PASS" : "COHORT PLAN FAIL", failures);
    return failures == 0 ? 0 : 1;
}
