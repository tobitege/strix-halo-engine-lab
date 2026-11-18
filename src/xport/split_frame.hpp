#pragma once
// Split-producer chunk frames (Channel::split_head), shared by every pair engine: the payload layout and the rows lever.
//
// A TP partial of `rows` x `width` values is produced in two column parts: columns [0, head_cols) first, then the rest
// (a GEMV / MoE down launched twice on its output columns: the weights are read once, as before). Each part is stored
// row-major on its own ("chunk-major"), so the head is the first contiguous rows * head_cols values of the message and
// can go on the wire while the tail is computed. Every element keeps its value and its sum order, so consumers that
// index through SplitLayout give the unsplit result bit for bit. head_cols == width is the plain row-major layout.
//
// Cost model (LNKS2): one more transport round per message and one more producer launch; hidden: the head round
// while the tail computes. Measured on usb4 (tests/hip_xport_split_frame_bench.hip, 1-8 rows of 4096/6144 FP32): with a
// tail of >= ~15 us the split saves 5-10 us per message at 1 row and 10-22 us at 2-8 rows; with no tail compute it
// loses one round (2-17 us). Hence a rows regime per family (its producers' tail time and launch cost), decided per
// step identically on both ranks (both ranks run the same rows).
#include "strix/env_flags.hpp"
#include "strix/lever.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

#if defined(__HIPCC__)
#define STRIX_XPORT_HD __host__ __device__
#else
#define STRIX_XPORT_HD
#endif

namespace strix::xport {

struct SplitLayout {
    unsigned rows = 0, width = 0, head_cols = 0;  // head_cols == width: row-major, no split
    [[nodiscard]] STRIX_XPORT_HD constexpr bool split() const { return head_cols < width; }
    [[nodiscard]] STRIX_XPORT_HD constexpr std::size_t head_elems() const { return std::size_t(rows) * head_cols; }
    // Leading dimension and first element of the part that holds column c.
    [[nodiscard]] STRIX_XPORT_HD constexpr unsigned ld(unsigned c) const { return c < head_cols ? head_cols : width - head_cols; }
    [[nodiscard]] STRIX_XPORT_HD constexpr std::size_t base(unsigned c) const { return c < head_cols ? 0 : head_elems(); }
    [[nodiscard]] STRIX_XPORT_HD constexpr std::size_t index(unsigned r, unsigned c) const {
        return c < head_cols ? std::size_t(r) * head_cols + c
                             : head_elems() + std::size_t(r) * (width - head_cols) + (c - head_cols);
    }
};

// The plain layout, or the split one at `head_cols` (a multiple of the producers' column tile, chosen by the engine).
[[nodiscard]] constexpr SplitLayout split_layout(unsigned rows, unsigned width, bool split, unsigned head_cols) {
    return SplitLayout{rows, width, split ? head_cols : width};
}

// Per-family lever row (same switch for every family; the crossover is each family's ABAB receipt).
[[nodiscard]] constexpr lever::Decl split_frame_decl(std::string_view name, std::uint64_t crossover_rows, std::string_view source) {
    return lever::Decl{name, "STRIX_XPORT_SPLIT_ROWS", lever::Exactness::bitwise, false,
                       lever::Regime{lever::Axis::rows, crossover_rows, false, true}, {}, source};
}

// Resolved once (family defaults resolver, the same call for CLI and serve); on(rows) per step.
struct SplitFramePolicy {
    enum class Mode : std::uint8_t { automatic = 0, off = 1, on = 2 };
    Mode mode = Mode::off;
    std::uint64_t crossover = 0;
    // No family has a one-row split producer form (the MiniMax and GLM MoE down launches split row groups), so even
    // an explicit "1" never splits below two rows; engines and their bench overrides decide through on().
    static constexpr unsigned kMinRows = 2;
    [[nodiscard]] constexpr bool on(unsigned rows) const {
        if (mode == Mode::off || rows < kMinRows) return false;
        return mode == Mode::on || rows >= crossover;
    }
    [[nodiscard]] std::string describe() const {
        return mode == Mode::off ? "0" : mode == Mode::on ? "1" : "auto(rows>=" + std::to_string(crossover) + ")";
    }
};

// No production regime is qualified yet: unset/empty stays OFF. Explicit 0/1 retains the diagnostic controls.
// The crossover is a pending hypothesis, not admission evidence; activate automatic mode only after real route qualification.
[[nodiscard]] inline SplitFramePolicy split_frame_policy(const lever::Decl& row, const char* raw) {
    const lever::Choice c = lever::select(row, lever::Inputs{1}, raw);  // validates the text
    SplitFramePolicy p;
    p.crossover = row.regime.crossover;
    if (c.origin == lever::Origin::explicit_switch) p.mode = c.on ? SplitFramePolicy::Mode::on : SplitFramePolicy::Mode::off;
    else p.mode = SplitFramePolicy::Mode::off;
    return p;
}

// Reads the process environment once and records the decision for --print-effective-config.
[[nodiscard]] inline SplitFramePolicy split_frame_policy(const lever::Decl& row) {
    const std::string env(row.env);
    const char* raw = strix::envcfg::text(env.c_str());
    const SplitFramePolicy p = split_frame_policy(row, raw);
    envcfg::record(env.c_str(), raw != nullptr && *raw != '\0' ? raw : nullptr, p.describe());
    return p;
}

}  // namespace strix::xport
