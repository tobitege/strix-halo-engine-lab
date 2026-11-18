#pragma once
// Closed-block summaries + open tail: the minimal exact form of a per-row state piece that its consumer only reads to
// build a fixed-size summary when a block of `block` rows closes (host shared; every engine / backend binds it).
//
// A family declares, per such piece, its geometry (BlockSummaryGeometry); this header owns the arithmetic every side
// must agree on: which raw rows a state at frontier P carries (RowWindow), the bytes of each form, and the hole rule.
//
// Contract of RawRows::open_tail (checked by each adapter's gate, never assumed by name):
//  1. The consumer reads raw rows of block b only while b is open (to build its summary when it closes); afterwards
//     only the summary. So a state at frontier P needs the summaries of the closed blocks [0, P / block) and the raw
//     rows of the open block [P - P % block, P), never the raw rows of a closed block.
//  2. The raw rows [0, P - P % block) are holes: absent from the wire, their device contents unspecified (a restore
//     leaves the slot-reset value). No restore, diagnostic, rollback or recompute path may read them; a route that
//     would must decline the form with its reason and fall back to RawRows::full.
//  3. Two states are compared on their canonical rows only (canonical_raw_window): holes are not state.
//  4. Rows at or after the frontier are appended by the consumer exactly as after a full restore.
// The same row-window arithmetic covers a fixed-lifetime ring (last_rows): only the last `window` rows are ever read.
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strix::state {

// Which raw rows of a summarised piece a state carries.
enum class RawRows : std::uint8_t {
    full = 0,       // every row [0, P): today's hand-off
    open_tail = 1,  // only the open block's rows [P - P % block, P); the closed blocks travel as summaries
};

inline const char* raw_rows_name(RawRows r) { return r == RawRows::open_tail ? "open_tail" : "full"; }

// Effective element type of one summary row (bytes carried vs container): the import widens exactly, never rounds.
enum class SummaryDtype : std::uint8_t {
    fp32 = 0,          // FP32 values computed in FP32 (GLM pooled index keys)
    bf16_in_fp32 = 1,  // BF16-rounded values held in an FP32 container: 2 bytes carry them bit for bit (Qwen QSA pool)
    bf16 = 2,
};

struct RowWindow {
    std::uint64_t first = 0, rows = 0;
    [[nodiscard]] constexpr std::uint64_t end() const { return first + rows; }
    constexpr bool operator==(const RowWindow&) const = default;
};

// One summarised piece of one layer.
struct BlockSummaryGeometry {
    std::uint32_t block = 0;              // raw rows per summary row (a block closes at rows multiple of `block`)
    std::uint64_t raw_row_bytes = 0;      // bytes of the raw rows per position (all raw pieces of the summary together)
    std::uint64_t summary_row_bytes = 0;  // bytes of one closed block's summary as carried
    SummaryDtype summary_dtype = SummaryDtype::fp32;
};

[[nodiscard]] constexpr std::uint64_t closed_blocks(std::uint64_t rows, std::uint32_t block) { return block ? rows / block : 0; }
[[nodiscard]] constexpr RowWindow open_tail(std::uint64_t rows, std::uint32_t block) {
    const std::uint64_t tail = block ? rows % block : rows;
    return {rows - tail, tail};
}
[[nodiscard]] constexpr RowWindow raw_window(RawRows form, std::uint64_t rows, std::uint32_t block) {
    return form == RawRows::open_tail ? open_tail(rows, block) : RowWindow{0, rows};
}
// The raw rows that are state at frontier `rows` (what an identity / canonical digest compares): the open block only.
[[nodiscard]] constexpr RowWindow canonical_raw_window(std::uint64_t rows, std::uint32_t block) { return open_tail(rows, block); }
[[nodiscard]] constexpr bool is_hole(RawRows form, std::uint64_t row, std::uint64_t rows, std::uint32_t block) {
    return form == RawRows::open_tail && row < open_tail(rows, block).first;
}
// A ring of fixed lifetime: only the last `window` rows of [0, rows) are ever read again.
[[nodiscard]] constexpr RowWindow last_rows(std::uint64_t rows, std::uint64_t window) {
    return rows > window ? RowWindow{rows - window, window} : RowWindow{0, rows};
}

[[nodiscard]] constexpr std::uint64_t raw_bytes(const BlockSummaryGeometry& g, RawRows form, std::uint64_t rows) {
    return raw_window(form, rows, g.block).rows * g.raw_row_bytes;
}
[[nodiscard]] constexpr std::uint64_t summary_bytes(const BlockSummaryGeometry& g, std::uint64_t rows) {
    return closed_blocks(rows, g.block) * g.summary_row_bytes;
}
[[nodiscard]] constexpr std::uint64_t state_bytes(const BlockSummaryGeometry& g, RawRows form, std::uint64_t rows) {
    return raw_bytes(g, form, rows) + summary_bytes(g, rows);
}
// Bytes the open-tail form saves against the full form at frontier `rows` (per layer, per copy).
[[nodiscard]] constexpr std::uint64_t open_tail_saving(const BlockSummaryGeometry& g, std::uint64_t rows) {
    return raw_bytes(g, RawRows::full, rows) - raw_bytes(g, RawRows::open_tail, rows);
}

// Which form a route resolved, and why (an observable reason string when the open tail is declined).
struct RawRowsChoice {
    RawRows form = RawRows::full;
    std::string reason;  // "" when the requested form was granted
};
// requested open_tail + a consumer that cannot honour the hole rule (decline != "") -> full, with the reason.
[[nodiscard]] inline RawRowsChoice resolve_raw_rows(RawRows requested, const std::string& decline) {
    if (requested == RawRows::open_tail && !decline.empty()) return {RawRows::full, "open_tail declined: " + decline};
    return {requested, {}};
}

inline void require_window_inside(const RowWindow& w, std::uint64_t capacity_rows, const char* what) {
    if (w.end() > capacity_rows || w.end() < w.first)
        throw std::out_of_range(std::string(what) + ": raw row window beyond the slot capacity");
}

}  // namespace strix::state
