#pragma once

#include <cstddef>
#include <cstdint>

namespace exact_reduce_contract {

// Per CTA: two L-word broadcasts, then the three legacy += results (r1, pa, pb).
constexpr unsigned output_slot(std::size_t index, unsigned lanes) {
    const std::size_t k = index % (2U * lanes + 3U);
    return k < lanes ? 0U : k < 2U * lanes ? 1U : static_cast<unsigned>(k - 2U * lanes) + 2U;
}

constexpr bool nan_word(std::uint32_t word) {
    return (word & 0x7fffffffU) > 0x7f800000U;
}

constexpr bool mismatch(std::uint32_t old_word, std::uint32_t new_word, unsigned slot) {
    const bool legacy_nan = slot >= 2U && slot <= 4U && nan_word(old_word) && nan_word(new_word);
    return old_word != new_word && !legacy_nan;
}

// Distinct quiet NaNs survive separate branches and meet at the final h=1 addition.
// Reverse their order for the second broadcast; all other leaves are positive zero.
constexpr std::uint32_t collision_leaf(unsigned lane, bool reverse) {
    return lane < 2U ? 0x7fc00001U + (lane ^ static_cast<unsigned>(reverse)) : 0U;
}

}  // namespace exact_reduce_contract
