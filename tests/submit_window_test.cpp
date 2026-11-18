// Host test: cuda_core/submit_window.hpp chunk planner (ramp + full chunks + tail), every row covered exactly once.
#include "strix/cuda_core/submit_window.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}
std::vector<std::uint32_t> rows(const std::vector<strix::cuda_core::ChunkSpan>& p) {
    std::vector<std::uint32_t> r;
    for (const auto& c : p) r.push_back(c.rows);
    return r;
}
bool covers(const std::vector<strix::cuda_core::ChunkSpan>& p, std::uint32_t n) {
    std::uint32_t at = 0;
    for (const auto& c : p) {
        if (c.pos != at || c.rows == 0) return false;
        at += c.rows;
    }
    return at == n;
}
}  // namespace

int main() {
    using strix::cuda_core::plan_chunks;
    using V = std::vector<std::uint32_t>;
    check(plan_chunks(0, 4096).empty(), "n = 0 gives an empty plan");
    check(rows(plan_chunks(8192, 4096)) == V{4096, 4096}, "no ramp: full chunks");
    check(rows(plan_chunks(10000, 4096)) == V{4096, 4096, 1808}, "no ramp: tail");
    check(rows(plan_chunks(32768, 4096, 1024)) == V{1024, 2048, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 1024},
          "ramp 1024 -> 4096, tail");
    check(rows(plan_chunks(1500, 4096, 1024)) == V{1024, 476}, "ramp longer than the prompt");
    check(rows(plan_chunks(8192, 4096, 4096)) == V{4096, 4096}, "ramp >= C is off");
    check(rows(plan_chunks(8192, 6016, 1000)) == V{896, 1792, 3584, 1920}, "ramp rounds down to align 128");
    check(rows(plan_chunks(300, 4096, 64)) == V{128, 172}, "ramp below align uses align");
    for (std::uint32_t n : {1u, 127u, 128u, 4095u, 4097u, 65536u, 131072u, 262144u})
        for (std::uint32_t C : {1024u, 4096u, 6016u, 8192u})
            for (std::uint32_t r : {0u, 512u, 1024u, 2048u})
                check(covers(plan_chunks(n, C, r), n), "every row exactly once");
    if (fails) return EXIT_FAILURE;
    std::printf("submit_window_test: PASS\n");
    return EXIT_SUCCESS;
}
