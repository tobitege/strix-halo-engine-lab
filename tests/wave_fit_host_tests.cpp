// Host-only: shared wave-aligned step rows (strix/wave_fit.hpp).
#include "strix/wave_fit.hpp"

#include <cmath>
#include <cstdio>

namespace {
int failures = 0;
void check(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); ++failures; }
}
}  // namespace

int main() {
    namespace wf = strix::wave_fit;
    const wf::Grid q27{128, 24, 188, 1};  // Qwen3.8-27B flash on the RTX PRO 6000: 128 q rows x 1 q head per CTA, 1 CTA/SM
    check(std::fabs(wf::efficiency(q27, 4096) - 768.0 / 940.0) < 1e-12, "4096 rows: 768 CTAs in 5 waves of 188");
    check(std::fabs(wf::efficiency(q27, 6016) - 1.0) < 1e-12, "6016 rows: 1128 CTAs = 6 full waves");
    check(wf::rows(q27, 4096, 1024, 128) == 3968, "4096 -> 3968 (744 CTAs, 98.9 %)");
    check(wf::rows(q27, 8192, 1024, 128) == 7936,
          "8192 -> 7936 (1488 CTAs, 8 waves, 98.9 %)");
    check(wf::rows(q27, 6016, 1024, 128) == 6016, "an already full-wave step is kept");
    const wf::Grid tiny{128, 1, 188, 1};  // fewer CTAs than SMs: no step reaches the floor -> the most efficient (largest)
    check(wf::rows(tiny, 2048, 1024, 128) == 2048, "single partial wave: keep the largest");
    bool threw = false;
    try { (void)wf::rows(q27, 1024, 2048, 128); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "min_rows > max_rows refused");
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::puts("wave_fit_host_tests: ok");
    return 0;
}
