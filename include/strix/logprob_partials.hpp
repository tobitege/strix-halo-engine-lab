#pragma once

// Logprob partials: the shared contract for target-row log-softmax scores computed where the logits live.
//
// A backend that holds a row's logits split in vocabulary slices (one per rank, or one per device block) reduces each
// slice to a LogprobPartial: its FP32 maximum, its FP32 sum of exp(x - max), its best `top` entries (value descending,
// lowest ID first among equals), and for an optional target ID its logit and the number of slice entries that rank
// ahead of it. Only these fixed-size records move (192 bytes per row and slice, whatever the vocabulary), never the
// row. combine_logprob_partials() merges the slices of one row in FP64 on the host.
//
// Values are RAW: the model's target-row distribution, before any logit transform or constraint mask. A route that
// scores transformed logits names that basis itself (serve "logit-transform-v1"); this contract never does.
//
// Precision (documented, not equality): the FP64 reference is strix::compute_requested_logprobs (FP64 maximum,
// ascending-ID FP64 exp sum). The partial sum is FP32 in ONE order shared by every backend and the host: lane l of
// kLogprobPartialLanes sums slice entries l, l + L, l + 2L, ... in turn, then the lanes fold pairwise (l += l + L/2,
// halving). A logprob differs from the reference by |log(S32 / S64)|. Bound: (terms per lane + log2 L + 2) * 2^-24
// relative on S plus the exp implementation's ulps; for a 129,280-entry row on 256 lanes that is < 3e-5 nats
// (kLogprobPartialTolerance); the host tests observe ~1e-6. Host and device bits may still differ by the ulps of
// their exp. The selected IDs, the top order and the target rank come from the FP32 logits
// themselves (exact comparisons), so they equal the reference whenever the reference sorts the same logits.
//
// The arithmetic changes no generated token: a scoring route reads the logits after the head, the greedy argmax is
// computed exactly as before.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace strix {

inline constexpr std::uint32_t kLogprobPartialTop = 20;          // OpenAI top_logprobs maximum
inline constexpr std::uint32_t kLogprobNoTarget = 0xFFFFFFFFU;
inline constexpr std::uint32_t kLogprobPartialLanes = 256;       // the FP32 summation order (one 256-thread block)
inline constexpr double kLogprobPartialTolerance = 3e-5;         // nats, FP32 partials vs the FP64 reference

// Bit layout shared by the host and the device kernels (HIP, CUDA): keep it a POD of 192 bytes.
struct alignas(8) LogprobPartial {
    float max = 0.0F;            // slice maximum (finite entries)
    float sum = 0.0F;            // sum of exp(x - max) over the slice, FP32
    float target = 0.0F;         // the target's logit when flags & kFound
    std::uint32_t flags = 0;     // kFound: the target lies in this slice; kCounted: greater/ties_lower are valid
    std::uint32_t greater = 0;   // slice entries with x > target logit (kCounted)
    std::uint32_t ties_lower = 0;  // slice entries with x == target logit and ID < target
    std::uint32_t nonfinite = 0; // slice entries that are NaN or infinite (a row with any is refused)
    std::uint32_t top_count = 0; // valid entries of top_*
    float top_value[kLogprobPartialTop]{};
    std::uint32_t top_id[kLogprobPartialTop]{};
    static constexpr std::uint32_t kFound = 1U, kCounted = 2U;
};
static_assert(sizeof(LogprobPartial) == 192);

// A target ID (a prompt token, or the emitted token). The slice holding it reads its logit itself and counts the
// entries ahead of it; another slice counts only when the caller knows the logit (logit_known: a single-device row,
// or a second pass after the owner slice published it). Without counts in every slice the rank is exact only when the
// target is inside the merged top list; otherwise it is reported unknown (0), never guessed.
struct LogprobTarget {
    std::uint32_t id = kLogprobNoTarget;
    float logit = 0.0F;
    bool logit_known = false;
};

struct LogprobEntry {
    std::uint32_t id = 0;
    double logprob = 0.0;
};
struct LogprobRowScore {
    double log_normalizer = 0.0;  // max + log(sum), informational: may lose the centered term at large offsets
    double maximum = 0.0;
    double centered_log_normalizer = 0.0;  // logprob(x) = (double(x) - maximum) - centered_log_normalizer
    bool has_target = false;
    double target_logprob = 0.0;
    std::uint64_t target_rank = 0;  // 1 = best (value descending, lowest ID first among equals); 0 = unknown
    std::vector<LogprobEntry> top;
};

// Host reduction of one slice (logits of IDs [first, first + slice.size())), the same record a device kernel
// writes, in the same lane order. Used where the slice is already on the host (a prefill's last row) and as the
// device tests' companion.
[[nodiscard]] LogprobPartial logprob_partial_host(std::span<const float> slice, std::uint32_t first, std::uint32_t top,
                                                  LogprobTarget target = {});

// FP64 merge of the slices of ONE row. `top` <= kLogprobPartialTop entries are returned. Throws std::invalid_argument
// on a nonfinite entry, an empty row, a partial with more than kLogprobPartialTop entries or a target found in no or
// in two slices (an incomplete normalizer is never returned).
[[nodiscard]] LogprobRowScore combine_logprob_partials(std::span<const LogprobPartial> slices, std::uint32_t top,
                                                       std::uint32_t target_id = kLogprobNoTarget);

// The bytes a scoring route moves per row: one partial per slice. Reported in capabilities and receipts.
[[nodiscard]] constexpr std::uint64_t logprob_partial_bytes(std::uint64_t rows, std::uint32_t slices) noexcept {
    return rows * slices * sizeof(LogprobPartial);
}

} // namespace strix
