// HIP logprob partials vs the host lane-order oracle and the FP64 reference; --bench R prints the kernel cost per
// step of R rows of a DS41 rank slice (64,640 entries) for the activated-cost budget (scripts/bench/logprobs).
#include "strix/generation_output.hpp"
#include "strix/hip_logprob_partials.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace {
int failures = 0;
void expect(bool c, const char* text) { if (!c) { std::fprintf(stderr, "FAIL: %s\n", text); ++failures; } }
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "HIP %s: %s\n", what, hipGetErrorString(e)); std::exit(2); }
}
std::uint64_t rng = 0x2545F4914F6CDD1DULL;
float next_logit(float scale) {
    rng ^= rng << 13U; rng ^= rng >> 7U; rng ^= rng << 17U;
    return static_cast<float>(static_cast<double>(rng >> 40U) / static_cast<double>(1ULL << 24U) - 0.5) * scale;
}
struct Device {
    float* logits = nullptr;
    strix::LogprobTarget* targets = nullptr;
    strix::LogprobPartial* out = nullptr;
    Device(std::size_t floats, std::size_t rows) {
        check(hipMalloc(&logits, floats * 4U), "malloc");
        check(hipMalloc(&targets, rows * sizeof(strix::LogprobTarget)), "malloc");
        check(hipMalloc(&out, rows * 2U * sizeof(strix::LogprobPartial)), "malloc");
    }
    ~Device() { (void)hipFree(logits); (void)hipFree(targets); (void)hipFree(out); }
};
int bench(std::uint32_t rows, std::uint32_t top) {
    constexpr std::uint32_t count = 64640;
    std::vector<float> host(std::size_t{rows} * count);
    for (auto& x : host) x = next_logit(16.0F);
    Device d(host.size(), rows);
    check(hipMemcpy(d.logits, host.data(), host.size() * 4U, hipMemcpyHostToDevice), "copy");
    hipStream_t st{};
    check(hipStreamCreate(&st), "stream");
    hipEvent_t a{}, b{};
    check(hipEventCreate(&a), "event");
    check(hipEventCreate(&b), "event");
    const strix::HipLogprobPartialsView v{d.logits, rows, count, count, 0U, top, nullptr, d.out};
    for (int i = 0; i < 20; ++i) strix::enqueue_hip_logprob_partials(v, st);
    check(hipStreamSynchronize(st), "warm");
    constexpr int iters = 200;
    check(hipEventRecord(a, st), "record");
    for (int i = 0; i < iters; ++i) strix::enqueue_hip_logprob_partials(v, st);
    check(hipEventRecord(b, st), "record");
    check(hipEventSynchronize(b), "sync");
    float ms = 0.0F;
    check(hipEventElapsedTime(&ms, a, b), "elapsed");
    const double us = 1e3 * ms / iters;
    std::printf("bench rows %u top %u count %u: %.2f us per step (%.2f us/row), %.1f GB/s logits read, %zu B partials out\n", rows, top, count, us,
                us / rows, (double(rows) * count * 4.0) / (us * 1e3), std::size_t{rows} * sizeof(strix::LogprobPartial));
    (void)hipEventDestroy(a); (void)hipEventDestroy(b); (void)hipStreamDestroy(st);
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--bench") == 0)
        return bench(static_cast<std::uint32_t>(std::stoul(argv[2])), argc >= 4 ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 20U);
    constexpr std::uint32_t V = 129280, rows = 6;
    std::vector<float> host(std::size_t{rows} * V);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t i = 0; i < V; ++i) host[std::size_t{r} * V + i] = next_logit(r % 2U ? 40.0F : 4.0F);
    host[2U * V + 500] = 70.0F;                                        // a peaked row
    for (std::uint32_t i = 0; i < V; ++i) host[3U * V + i] = 1.25F;   // all ties
    for (std::uint32_t i = 0; i < V; ++i)  // 20 lanes of rising maxima: > 2048 top candidates (the exact-pass path)
        if (i % 256U < 20U) host[4U * V + i] = 10.0F + static_cast<float>(i % 256U) * 100.0F + static_cast<float>(i) * 1e-3F;
    std::vector<strix::LogprobTarget> targets(rows);
    for (std::uint32_t r = 0; r < rows; ++r) targets[r] = {r * 20011U % V, host[std::size_t{r} * V + r * 20011U % V], true};
    targets[4].id = strix::kLogprobNoTarget;
    targets[5].logit_known = false;  // owner slice only: rank known only inside the top
    Device d(host.size(), rows);
    check(hipMemcpy(d.logits, host.data(), host.size() * 4U, hipMemcpyHostToDevice), "copy");
    check(hipMemcpy(d.targets, targets.data(), rows * sizeof(strix::LogprobTarget), hipMemcpyHostToDevice), "copy");
    const std::uint32_t half = V / 2U;
    double worst_ref = 0.0, worst_host = 0.0;
    for (const std::uint32_t top : {0U, 20U}) {  // target-only and top-list scoring
        // two slices of every row, as two ranks: [0, half) and [half, V), partials [slice][row]
        strix::enqueue_hip_logprob_partials({d.logits, rows, half, V, 0U, top, d.targets, d.out}, nullptr);
        strix::enqueue_hip_logprob_partials({d.logits + half, rows, V - half, V, half, top, d.targets, d.out + rows}, nullptr);
        std::vector<strix::LogprobPartial> got(2U * rows);
        check(hipMemcpy(got.data(), d.out, got.size() * sizeof(strix::LogprobPartial), hipMemcpyDeviceToHost), "copy back");
        for (std::uint32_t r = 0; r < rows; ++r) {
            const std::span<const float> row(host.data() + std::size_t{r} * V, V);
            const strix::LogprobPartial host_parts[2] = {strix::logprob_partial_host(row.subspan(0, half), 0U, top, targets[r]),
                                                         strix::logprob_partial_host(row.subspan(half), half, top, targets[r])};
            const strix::LogprobPartial dev_parts[2] = {got[r], got[rows + r]};
            for (int s = 0; s < 2; ++s) {
                const auto& h = host_parts[s];
                const auto& g = dev_parts[s];
                expect(h.max == g.max && h.flags == g.flags && h.greater == g.greater && h.ties_lower == g.ties_lower &&
                           h.nonfinite == g.nonfinite && h.top_count == g.top_count && h.target == g.target,
                       "device partial fields equal the host oracle");
                for (std::uint32_t k = 0; k < h.top_count && k < 20U; ++k)
                    expect(h.top_id[k] == g.top_id[k] && h.top_value[k] == g.top_value[k], "device top equals the host oracle");
                worst_host = std::max(worst_host, std::abs(double(h.sum) - double(g.sum)) / double(h.sum));
            }
            const auto score = strix::combine_logprob_partials(dev_parts, top, targets[r].id);
            std::vector<std::uint32_t> ids;
            for (const auto& e : score.top) ids.push_back(e.id);
            if (score.has_target) ids.push_back(targets[r].id);
            std::vector<strix::RequestedLogprob> ref(ids.size());
            strix::compute_requested_logprobs(row, ids, ref);
            for (std::size_t k = 0; k < score.top.size(); ++k) worst_ref = std::max(worst_ref, std::abs(score.top[k].logprob - ref[k].log_probability));
            if (score.has_target) worst_ref = std::max(worst_ref, std::abs(score.target_logprob - ref.back().log_probability));
            if (top != 0U && r == 3U) expect(score.top.size() == 20U && score.top[0].id == 0U && score.top[19].id == 19U, "ties order by ID");
            if (top != 0U && r == 2U) expect(score.top[0].id == 500U, "peaked row");
            if (top != 0U && r == 4U) expect(score.top.size() == 20U && score.top[0].id == 256U * 504U + 19U, "candidate overflow row");
        }
    }
    expect(worst_ref <= strix::kLogprobPartialTolerance, "device partials within the documented tolerance");
    std::printf("hip logprob partials: worst vs FP64 %.3e nats, worst sum vs host lane order %.3e relative\n", worst_ref, worst_host);
    if (failures != 0) return 1;
    std::puts("hip logprob partial tests passed");
    return 0;
}
