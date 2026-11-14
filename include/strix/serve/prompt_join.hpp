#pragma once

#include <cstdint>

// How a prompt that becomes ready while another prompt's unit is on the device gets its rows run.
//
// A serving loop that pauses a running prompt unit at service points has two exact ways to serve the
// newcomer; both give every prompt the units, rows and bits of a solo run, so the choice is scheduling only:
//   nest  - at the next service point the newcomer runs whole and alone while the running unit waits.
//           Right for a short prompt behind a long unit; a burst served this way is a serial stack
//           (the last prompt of N waits for N - 1 whole prompts, and the running prompt for all of them).
//   group - the newcomer waits for the unit boundary; its blocks then join the next blocks of the other
//           prompts in one forward (cross-request prefill). The wait is bounded by the running unit.
// The rule compares the two waits: a newcomer of at most 1/ratio of the running unit's rows nests,
// a larger one groups. ratio 0 keeps the previous behaviour (always nest).
namespace strix::serve {

enum class PromptJoin : std::uint8_t { nest, group };

struct PromptJoinLimits {
    bool groups = true;        // the family runs several prompts' blocks in one forward
    std::uint32_t ratio = 4;   // nest while prompt_rows * ratio <= running_unit_rows; 0 = always nest
};

[[nodiscard]] constexpr PromptJoin prompt_join(std::uint64_t running_unit_rows, std::uint64_t prompt_rows,
                                               const PromptJoinLimits& limits) noexcept {
    if (!limits.groups || limits.ratio == 0U) return PromptJoin::nest;
    // prompt_rows <= running_unit_rows / ratio, without overflow
    return prompt_rows <= running_unit_rows / limits.ratio ? PromptJoin::nest : PromptJoin::group;
}

}  // namespace strix::serve
