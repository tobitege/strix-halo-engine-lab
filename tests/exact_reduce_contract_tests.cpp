#include "exact_reduce_contract.hpp"

#include <cstdio>

int main() {
    namespace contract = exact_reduce_contract;
    unsigned failures = 0U;
    const auto expect = [&failures](bool ok, const char* what) {
        if (!ok) { ++failures; if (failures <= 5U) std::fprintf(stderr, "FAIL: %s\n", what); }
    };
    struct Words { std::uint32_t a, b; bool both_nan; };
    constexpr Words cases[] = {
        {0x7fc00001U, 0x7fc00002U, true},  // reported false pass
        {0x7f800001U, 0xffc00002U, true},  // signaling/quiet and sign changes
        {0x7fc00001U, 0x7f800000U, false}, // NaN/infinity
        {0xff800000U, 0xff800001U, false}, // infinity/NaN
        {0U, 0x80000000U, false},         // signed zero
        {0x3f800000U, 0x3f800001U, false}, // finite one-bit change
    };
    constexpr unsigned sizes[] = {64U, 128U, 256U, 512U, 1024U};
    for (const unsigned lanes : sizes) {
        const std::size_t per = 2U * lanes + 3U;
        for (std::size_t cta = 0; cta < 2U; ++cta) {
            for (std::size_t k = 0; k < per; ++k) {
                const unsigned slot = contract::output_slot(cta * per + k, lanes);
                const bool legacy = k >= 2U * lanes;
                expect(slot == (legacy ? 2U + static_cast<unsigned>(k - 2U * lanes)
                                       : static_cast<unsigned>(k / lanes)), "slot classification");
                for (const auto& words : cases) {
                    expect(contract::mismatch(words.a, words.b, slot) == !(legacy && words.both_nan),
                           "broadcast raw equality / legacy NaN-only exception");
                    expect(!contract::mismatch(words.a, words.a, slot), "identical words pass");
                }
            }
        }
        for (unsigned lane = 0U; lane < lanes; ++lane) {
            expect(contract::collision_leaf(lane, false) ==
                       (lane == 0U ? 0x7fc00001U : lane == 1U ? 0x7fc00002U : 0U), "collision leaves");
            expect(contract::collision_leaf(lane, true) ==
                       (lane == 0U ? 0x7fc00002U : lane == 1U ? 0x7fc00001U : 0U), "reversed collision leaves");
        }
    }
    return failures == 0U ? 0 : 1;
}
