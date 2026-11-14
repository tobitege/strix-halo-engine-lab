#pragma once
// Host-only model scheduling glue. All conversion and arithmetic stay in the library.
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace strix::minimax::engine::fast_i4 {
enum class Scope { all, gate_up };
inline Scope scope(const char* value) {
    const std::string_view s = value == nullptr ? "" : value;
    if (s.empty() || s == "all") return Scope::all;
    if (s == "gate_up") return Scope::gate_up;
    throw std::invalid_argument("STRIX_MMX_FAST_I4_SCOPE must be all or gate_up");
}

// Default quality policy of FAST_I4 (wave4/deep1111/MMI4-FABLE.md): the MoE layers that keep the exact EXL3 operator
// when STRIX_MMX_FAST_I4_EXL3_LAYERS is unset. INT4 noise in an early layer is amplified by every later router and
// sparse-attention selection, so the early layers stay exact and INT4 runs where it is cheap in quality. "none"
// restores INT4 in every MoE layer (the pre-policy identity); the grammar is core/layer_set.hpp.
inline constexpr const char* kQualityExl3Layers = "3-37";

// Regime rule: a prompt shorter than this many rows runs the exact EXL3 prefill in every layer (bit-identical to the
// exact format). The per-layer JIT conversion is a fixed cost, so INT4 only pays above about 3K rows; the default is
// one full prefill chunk. STRIX_MMX_FAST_I4_MIN_ROWS overrides it (0 = INT4 at every length).
inline constexpr std::size_t kMinRows = 4096;
inline std::size_t min_rows(const char* value) {
    const std::string_view s = value == nullptr ? "" : value;
    if (s.empty()) return kMinRows;
    std::size_t v = 0;
    for (const char c : s) {
        if (c < '0' || c > '9' || v > 100000000U) throw std::invalid_argument("STRIX_MMX_FAST_I4_MIN_ROWS must be a row count");
        v = v * 10U + static_cast<std::size_t>(c - '0');
    }
    return v;
}
// The row gate of ONE prefill task. It is task state: a nested child begins its own task (short, so OFF) while the
// parent is parked, and the resumed parent must keep the decision it began with (arithmetic and speed).
struct RowGate {
    bool on = true;
    void begin(std::size_t rows, std::size_t minimum) { on = rows >= minimum; }
};

// STATS of ONE prefill task and layer. Task state like the row gate: launches and timing samples are charged to the
// task that issued them, so a nested child's report neither prints nor clears the parked parent's. Timing is kept per
// arithmetic (the route of the sampled group), not per layer; launch counts do not depend on a timing sample.
struct LayerStats {
    std::uint64_t i4_calls = 0, exl3_calls = 0;                    // MoE launches (one per chunk or group)
    std::uint64_t i4_samples = 0, exl3_samples = 0, converts = 0;  // timed groups / conversions
    double i4_ms = 0, exl3_ms = 0, convert_ms = 0;
    void count(bool i4, std::size_t calls) { (i4 ? i4_calls : exl3_calls) += calls; }
    void sample(bool i4, double ms) { (i4 ? i4_ms : exl3_ms) += ms; ++(i4 ? i4_samples : exl3_samples); }
    void converted(double ms) { convert_ms += ms; ++converts; }
    bool any() const { return (i4_calls | exl3_calls | i4_samples | exl3_samples | converts) != 0; }
    static double mean(double ms, std::uint64_t n) { return n != 0 ? ms / static_cast<double>(n) : 0.0; }
};
template<std::size_t Layers>
struct TaskStats {
    std::array<LayerStats, Layers> layer{};
};

// One acquire/release surrounds ALL chunks of a layer. A new group/range must
// begin afresh; a pack is never treated as a resident layer across groups.
struct Schedule {
    static constexpr unsigned slots = 2;
    std::array<int, slots> layer{{-1, -1}};
    void begin() { layer.fill(-1); }

    // convert(slot, layer, side) must wait for the slot's last release before
    // side-stream writes, then publish ready. wait(slot) is an engine-stream wait.
    template<class Convert, class Wait>
    unsigned acquire(unsigned L, unsigned end, bool prefetch, Convert convert, Wait wait) {
        if (L >= end) throw std::invalid_argument("fast_i4: layer outside prefill range");
        const unsigned s = L & 1U;
        if (layer[s] != static_cast<int>(L)) {
            convert(s, L, false);  // serial on the engine stream, including the first layer
            layer[s] = static_cast<int>(L);
        } else {
            wait(s);
        }
        if (prefetch && L + 1U < end) {
            const unsigned next = L + 1U, other = next & 1U;
            convert(other, next, true);
            layer[other] = static_cast<int>(next);
        }
        return s;
    }
};
}  // namespace strix::minimax::engine::fast_i4
