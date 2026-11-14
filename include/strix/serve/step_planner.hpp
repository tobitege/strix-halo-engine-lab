#pragma once

#include <cstdint>
#include <span>
#include <vector>

// Shared step planner: what runs in one serving step.
//
// Decode progress is reserved, prompt work is cut to a bounded slice, and short prompts may overtake
// long ones under a starvation bound. StepPlanner is a pure host class: lane descriptors in, plan out.
// It decides scheduling only; a prompt's rows and bits do not depend on the order or the size of its
// units, and every decode lane's tokens come from its own rows, so any plan gives the ids of a solo run.
//
// One planner covers three policies that already exist in the families:
//   - time slicing with a debt ledger: a prompt unit runs before the decode round only while the
//     decoders are not owed time (co_forward = false);
//   - shortest-first admission with aging (short_first.hpp: kMaxQueueOvertakes);
//   - row fill of one forward: one row per decoding lane, the rows left to prompts (co_forward = true).
namespace strix::serve {

struct StepLane {
    std::uint32_t lane = 0;               // caller's identifier, returned in the plan
    bool prompt = false;                  // true: still reading its prompt; false: decoding
    std::uint64_t prompt_remaining = 0;   // prompt rows left (prompt lanes)
    std::uint64_t order = 0;              // arrival order, smaller = older
    unsigned overtakes = 0;               // times a younger prompt ran first (caller keeps the count)
    std::uint32_t max_prompt_rows = 0;    // per-lane cap for this step, 0 = none (a first token that must run alone = 1)
    bool eligible = true;                 // false: cannot run this step (an aged ineligible lane blocks the others)
};

enum class PromptOrder : std::uint8_t {
    short_class,  // prompts with at most short_first_rows left go first, oldest first inside each class
    shortest,     // strictly fewest rows left first, arrival order on ties (requires short_first_rows != 0)
};

struct StepLimits {
    bool co_forward = false;                        // prompt rows may share the decode forward
    std::uint32_t rows_per_forward = 0;             // co_forward: physical rows of one forward
    std::span<const std::uint32_t> prompt_units;    // family-certified slice sizes, ascending; empty = the family's own unit
    double prompt_share = 0.0;                      // ms of decode owed per ms of prompt; 0 = prompts never wait for decode
    double max_decode_gap_ms = 0.0;                 // 0 = no bound; else the prompt unit is capped to fit while lanes decode
    std::uint64_t short_first_rows = 2048;          // 0 = oldest prompt first, no overtaking (an ineligible oldest lane blocks)
    PromptOrder prompt_order = PromptOrder::short_class;
    double prompt_row_ms = 0.0;                     // prior cost of one prompt row; the observed cost replaces it
};

struct StepPlan {
    std::vector<std::uint32_t> decode_lanes;        // lanes that decode this step, in the order given
    struct Prompt {
        std::uint32_t lane;
        std::uint32_t rows;                         // 0 = the family's own unit (time slicing with no certified sizes)
    };
    std::vector<Prompt> prompt;                     // time slicing: at most one entry (the head of the unit or group)
    std::vector<std::uint32_t> overtaken;           // older prompt lanes passed over by the head: their overtakes grow by one
    bool prompt_before_decode = false;              // time slicing: run the prompt unit, then the decode round
};

class StepPlanner {
public:
    // Plans one step. Clears the debt when no prompt lane is present.
    StepPlan plan(std::span<const StepLane> lanes, const StepLimits& limits);

    // After the step: prompt time (less the time it was suspended for other service) is charged with the
    // share of the last plan, decode time pays it back. prompt_rows = rows the prompt unit really ran
    // (0 = the rows of the last plan); it feeds the per-row cost used by max_decode_gap_ms.
    void account(double prompt_ms, double decode_ms, double suspended_ms, std::uint64_t prompt_rows = 0);

    void reset();
    [[nodiscard]] double debt_ms() const noexcept { return debt_ms_; }
    [[nodiscard]] double prompt_row_ms() const noexcept { return observed_rows_ ? observed_ms_ / static_cast<double>(observed_rows_) : 0.0; }

private:
    double debt_ms_ = 0.0;
    double share_ = 0.0;
    std::uint64_t planned_rows_ = 0;
    double observed_ms_ = 0.0;
    std::uint64_t observed_rows_ = 0;
};

}  // namespace strix::serve
