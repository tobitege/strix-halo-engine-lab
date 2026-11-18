// Shared logprob partial contract: FP32 slice partials merged in FP64 vs the FP64 reference
// (strix::compute_requested_logprobs), for 1, 2 and 7 slices of a DS41-sized vocabulary, ties, rank and refusals.
#include "strix/generation_output.hpp"
#include "strix/logprob_partials.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
int failures = 0;
void expect(bool condition, const char* text) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", text); ++failures; }
}
std::uint64_t rng = 0x9E3779B97F4A7C15ULL;
float next_logit(float scale) {
    rng ^= rng << 13U; rng ^= rng >> 7U; rng ^= rng << 17U;
    return static_cast<float>(static_cast<double>(rng >> 40U) / static_cast<double>(1ULL << 24U) - 0.5) * scale;
}
std::vector<strix::LogprobPartial> split(const std::vector<float>& row, std::uint32_t slices, std::uint32_t top,
                                         strix::LogprobTarget target) {
    std::vector<strix::LogprobPartial> out;
    const auto n = static_cast<std::uint32_t>(row.size());
    for (std::uint32_t s = 0; s < slices; ++s) {
        const std::uint32_t a = static_cast<std::uint32_t>(std::uint64_t{n} * s / slices);
        const std::uint32_t b = static_cast<std::uint32_t>(std::uint64_t{n} * (s + 1U) / slices);
        out.push_back(strix::logprob_partial_host(std::span<const float>(row.data() + a, b - a), a, top, target));
    }
    return out;
}
double worst = 0.0;
void check_row(const std::vector<float>& row, std::uint32_t slices, std::uint32_t top, std::uint32_t target_id, bool known) {
    strix::LogprobTarget t{target_id, target_id < row.size() ? row[target_id] : 0.0F, known};
    const auto parts = split(row, slices, top, t);
    const auto score = strix::combine_logprob_partials(parts, top, target_id);
    // reference order: value descending, lowest ID first
    std::vector<std::uint32_t> order(row.size());
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return row[a] > row[b]; });
    std::vector<std::uint32_t> ids(order.begin(), order.begin() + top);
    ids.push_back(target_id);
    std::vector<strix::RequestedLogprob> ref(ids.size());
    strix::compute_requested_logprobs(row, ids, ref);
    expect(score.top.size() == top, "top count");
    for (std::uint32_t k = 0; k < top && k < score.top.size(); ++k) {
        expect(score.top[k].id == order[k], "top order equals the reference order");
        worst = std::max(worst, std::abs(score.top[k].logprob - ref[k].log_probability));
    }
    expect(score.has_target, "target scored");
    worst = std::max(worst, std::abs(score.target_logprob - ref.back().log_probability));
    const auto true_rank = static_cast<std::uint64_t>(std::find(order.begin(), order.end(), target_id) - order.begin()) + 1U;
    if (known || slices == 1U) expect(score.target_rank == true_rank, "rank with counts in every slice");
    else if (true_rank <= top) expect(score.target_rank == true_rank, "rank inside the merged top");
    else expect(score.target_rank == 0U, "rank unknown outside the top without counts");
}
} // namespace

int main() {
    for (const float offset : {1e20F, -1e20F}) {
        const std::vector<float> large(2, offset);
        for (const std::uint32_t slices : {1U, 2U}) {
            for (const std::uint32_t top : {0U, 2U}) check_row(large, slices, top, 0U, true);
            const auto score = strix::combine_logprob_partials(split(large, slices, 2U, {0U, offset, true}), 2U, 0U);
            expect(std::abs(score.target_logprob + std::log(2.0)) < 1e-12, "large-offset target stays centered");
            for (const auto& entry : score.top)
                expect(std::abs(entry.logprob + std::log(2.0)) < 1e-12, "large-offset top stays centered");
            expect(std::abs((static_cast<double>(offset) - score.maximum) - score.centered_log_normalizer + std::log(2.0)) < 1e-12,
                   "retained centered normalizer supports further scores");
        }
    }
    constexpr std::uint32_t V = 129280;  // DeepSeek V4 vocabulary (two ranks: 64,640 per slice)
    std::vector<float> row(V);
    for (const float scale : {4.0F, 40.0F}) {
        for (auto& x : row) x = next_logit(scale);
        for (const std::uint32_t slices : {1U, 2U, 7U})
            for (const std::uint32_t top : {0U, 1U, 5U, 20U}) {
                check_row(row, slices, top, 12345U, true);
                check_row(row, slices, top, 100000U, false);
            }
    }
    {  // a peaked row (the usual decode case): one dominant logit
        for (auto& x : row) x = next_logit(8.0F);
        row[777] = 60.0F;
        check_row(row, 2U, 20U, 777U, false);
        check_row(row, 2U, 20U, 3U, true);
    }
    {  // ties: equal values order by lowest ID across slices; rank counts the lower-ID ties
        std::vector<float> tie(1000, 1.0F);
        tie[999] = 2.0F;
        check_row(tie, 2U, 20U, 600U, true);
        const auto parts = split(tie, 2U, 3U, {});
        const auto s = strix::combine_logprob_partials(parts, 3U);
        expect(s.top.size() == 3 && s.top[0].id == 999U && s.top[1].id == 0U && s.top[2].id == 1U, "tie order across slices");
    }
    bool threw = false;
    {  // a nonfinite entry anywhere refuses the row (the finite-row contract of every scoring route)
        std::vector<float> bad(100, 0.0F);
        bad[70] = std::numeric_limits<float>::infinity();
        try { (void)strix::combine_logprob_partials(split(bad, 2U, 5U, {}), 5U); } catch (const std::invalid_argument&) { threw = true; }
        expect(threw, "nonfinite refused");
    }
    threw = false;
    try { (void)strix::combine_logprob_partials({}, 1U); } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "no slice refused");
    threw = false;
    {  // a target found in no slice: an incomplete normalizer is never returned
        std::vector<float> r(100, 0.5F);
        auto parts = split(r, 2U, 1U, {});
        try { (void)strix::combine_logprob_partials(parts, 1U, 7U); } catch (const std::invalid_argument&) { threw = true; }
        expect(threw, "target in no slice refused");
    }
    expect(worst <= strix::kLogprobPartialTolerance, "FP32 partials within the documented tolerance of the FP64 reference");
    std::printf("logprob partials: worst |FP32 partial - FP64 reference| = %.3e nats (tolerance %.1e)\n", worst,
                strix::kLogprobPartialTolerance);
    if (failures != 0) return 1;
    std::puts("logprob partial tests passed");
    return 0;
}
