// Host-only tests for the shared prompt join rule (nest at a service point or join the next group forward).
#include "strix/serve/prompt_join.hpp"

#include <array>
#include <cstdint>
#include <iostream>

using namespace strix::serve;

namespace {
int failures = 0;
void check(bool value, const char* label) {
    if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
}  // namespace

int main() {
    const PromptJoinLimits def;
    // A burst of equal prompts behind a tail block: every newcomer joins the next forward.
    check(prompt_join(1024, 1383, def) == PromptJoin::group, "equal prompt behind a 1024-row unit groups");
    check(prompt_join(1024, 257, def) == PromptJoin::group, "just above a quarter of the unit groups");
    check(prompt_join(1024, 256, def) == PromptJoin::nest, "a quarter of the unit nests");
    check(prompt_join(1024, 7, def) == PromptJoin::nest, "a tiny prompt nests");
    // A short prompt behind a whole T block keeps nesting (the wait for the boundary would be the long one).
    check(prompt_join(8192, 2048, def) == PromptJoin::nest, "2048 rows behind an 8192-row block nests");
    check(prompt_join(8192, 2049, def) == PromptJoin::group, "more than a quarter of a T block groups");
    check(prompt_join(16, 1, def) == PromptJoin::nest, "one row behind a 16-row block nests");
    check(prompt_join(16, 5, def) == PromptJoin::group, "five rows behind a 16-row block wait for it");
    // Switches: no group forward in the family, or ratio 0, keep the previous behaviour.
    check(prompt_join(1024, 1383, PromptJoinLimits{false, 4}) == PromptJoin::nest, "no groups: nest");
    check(prompt_join(1024, 1383, PromptJoinLimits{true, 0}) == PromptJoin::nest, "ratio 0: nest");
    check(prompt_join(1024, 1024, PromptJoinLimits{true, 1}) == PromptJoin::nest, "ratio 1: up to the unit nests");
    check(prompt_join(1024, 1025, PromptJoinLimits{true, 1}) == PromptJoin::group, "ratio 1: larger groups");
    // Monotone: growing the prompt never turns group back into nest; growing the unit never turns nest into group.
    const std::array<std::uint64_t, 6> units{16, 64, 256, 1024, 8192, UINT64_MAX};
    for (const auto unit : units) {
        bool grouped = false;
        for (std::uint64_t rows = 1; rows <= 20000; rows += 37) {
            const bool g = prompt_join(unit, rows, def) == PromptJoin::group;
            check(!grouped || g, "monotone in prompt rows");
            grouped = g;
        }
    }
    check(prompt_join(UINT64_MAX, UINT64_MAX, def) == PromptJoin::group, "no overflow at the limits");
    static_assert(prompt_join(1024, 1383, PromptJoinLimits{}) == PromptJoin::group);
    if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
    std::cout << "prompt join tests passed\n";
    return 0;
}
