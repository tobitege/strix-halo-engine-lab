#pragma once

#include <algorithm>
#include <cstdint>

namespace strix::session {

// A request's prompt prefill as a sequence of bounded units (engine neutral; both ranks of a pair hold one, created by
// the same command and stepped by the same commands, so its collectives run in the same order on both ranks).
// One unit is what the engine can stop after: a run of rows (DeepSeek V4.1's DevicePrefill pieces) or one layer of the
// current piece (GLM: the prompt's pass over the layers paused at a layer boundary, see LayerUnits).
// Contract for an implementation:
//  - step() returns with no collective of the prefill in flight (a decode step's collectives follow it);
//  - the prefill's state, scratch and row buffers are its own, never a decode slot's or the decode workspace
//    (an engine whose row buffers are shared with decode keeps the decode rows in a reserved window while paused);
//  - the bits of every row and of the final state do not depend on how the prompt was cut into units.
class PrefillSession {
public:
    virtual ~PrefillSession() = default;
    // Advances the prefill to row `end` (a unit; resumable implementations may stop earlier inside the piece within
    // `budget` and return false with rows() unchanged); true when rows() == the prompt's length.
    virtual bool step(std::uint32_t end, std::uint32_t budget) = 0;
    [[nodiscard]] virtual std::uint32_t rows() const noexcept = 0;   // rows of the prompt done (at)
    [[nodiscard]] virtual std::uint32_t total() const noexcept = 0;  // the prompt's length
    [[nodiscard]] virtual bool done() const noexcept { return rows() == total(); }
    [[nodiscard]] virtual bool mid_piece() const noexcept { return false; }  // a resumable unit stopped inside a piece
    [[nodiscard]] virtual std::uint32_t piece_end() const noexcept { return rows(); }  // mid_piece: the end the piece runs to
    // Layer-unit engines: units of the whole prompt done / in total (0 / 0: the units are rows, as rows() / total()).
    [[nodiscard]] virtual std::uint32_t units_done() const noexcept { return 0; }
    [[nodiscard]] virtual std::uint32_t units_total() const noexcept { return 0; }
};

// Unit arithmetic of a prompt prefilled in pieces of at most `piece_rows` rows, each piece one pass over `layers`
// layers that may pause at any layer boundary (pure; both ranks and the scheduler derive the same numbers).
// Unit u (0-based) is layer u % layers of piece u / layers.
struct LayerUnits {
    std::uint64_t rows = 0;        // prompt rows of the session
    std::uint32_t piece_rows = 0;  // rows per piece (the engine's max_rows)
    std::uint32_t layers = 0;      // layers per pass

    [[nodiscard]] constexpr bool valid() const noexcept { return rows != 0 && piece_rows != 0 && layers != 0; }
    [[nodiscard]] constexpr std::uint64_t pieces() const noexcept {
        return valid() ? (rows + piece_rows - 1U) / piece_rows : 0U;
    }
    [[nodiscard]] constexpr std::uint64_t total() const noexcept { return pieces() * layers; }
    // Admission must cover every piece, including the short tail, before a session starts.
    // Full pieces share one geometry; checking it and the tail is sufficient.
    template<class Pausable>
    [[nodiscard]] constexpr bool all_pieces_pausable(Pausable pausable) const {
        if (!valid()) return false;
        if (!pausable(static_cast<std::uint32_t>(std::min<std::uint64_t>(rows, piece_rows)))) return false;
        const auto tail = static_cast<std::uint32_t>(rows % piece_rows);
        return rows <= piece_rows || tail == 0 || pausable(tail);
    }
    // Rows of the piece unit `u` belongs to (the last piece may be short).
    [[nodiscard]] constexpr std::uint64_t piece_rows_at(std::uint64_t u) const noexcept {
        if (!valid() || u >= total()) return 0;
        const std::uint64_t first = (u / layers) * piece_rows;
        return std::min<std::uint64_t>(piece_rows, rows - first);
    }
    // The scheduler's base chunk of `base_tokens` prompt rows expressed in layer units of the piece at unit `u`:
    // the same work per decode round as a base-token chunk, without its per-call pass over all the layers
    // (rounded up, at least one unit, at most the units left).
    [[nodiscard]] constexpr std::uint64_t base_units(std::uint64_t base_tokens, std::uint64_t u) const noexcept {
        const std::uint64_t pr = piece_rows_at(u);
        if (pr == 0) return 0;
        const std::uint64_t left = total() - u;
        if (base_tokens == 0) return left;
        const std::uint64_t n = (base_tokens * layers + pr - 1U) / pr;
        return std::clamp<std::uint64_t>(n, 1U, left);
    }
};

}  // namespace strix::session
