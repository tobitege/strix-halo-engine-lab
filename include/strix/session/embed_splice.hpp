#pragma once

// Embed-stage input of one prefill run: which rows of the run take their layer-0 input from provided
// embedding rows instead of the family's embedding table.
//
// A family whose prefill takes token ids resolves a run in two steps:
//   1. resolve_embed_run(ids, table, rules): the ids with every virtual id (embed_rows.hpp) replaced by the
//      vocabulary token of its span, plus an EmbedSplice naming the rows that come from the table;
//   2. its embed stage looks the tokens up as before and overwrites the spliced rows
//      (device: enqueue_embed_splice, embed_splice_hip.hpp), then applies its per-row extras to the
//      segments marked `extras`.
// A run without a virtual id resolves to the same ids and an empty splice: nothing changes for text.
//
// Header only: an engine library that only consumes an EmbedSplice needs no session library.

#include "strix/session/embed_rows.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::session {

enum class EmbedRowFormat : std::uint8_t {
    bf16 = 0, // EmbedRows::data: the upper 16 bits of the FP32 value
    f16 = 1,  // IEEE binary16 (an embedding table's own rows: identity gates)
};

// Consecutive run rows that read consecutive source rows.
struct EmbedSegment {
    std::uint32_t run_row = 0;           // first row, index in the run
    std::uint32_t rows = 0;              // > 0
    const std::uint16_t* data = nullptr; // [rows][width]; alive while EmbedSplice::keep is
    // The family applies its per-row extras for provided rows (e.g. image router bias, dead n-gram
    // positions). false: the rows are treated as the token rows they replace.
    bool extras = true;
};

struct EmbedSplice {
    std::uint32_t width = 0;
    EmbedRowFormat format = EmbedRowFormat::bf16;
    std::vector<EmbedSegment> segments; // ascending run_row, no overlap
    std::shared_ptr<const void> keep;   // owner of the segment data (e.g. the EmbedTable)

    [[nodiscard]] bool empty() const noexcept { return segments.empty(); }
    [[nodiscard]] std::uint64_t rows() const noexcept {
        std::uint64_t n = 0;
        for (const auto& s : segments) n += s.rows;
        return n;
    }
    // Prevalidation against the run it belongs to. Throws std::invalid_argument.
    void validate(std::uint64_t run_rows, std::uint32_t hidden) const {
        if (segments.empty()) return;
        if (width != hidden || hidden == 0U) throw std::invalid_argument("embed splice: row width differs from the model");
        std::uint64_t next = 0;
        for (const auto& s : segments) {
            if (s.rows == 0U || s.data == nullptr) throw std::invalid_argument("embed splice: empty segment");
            if (s.run_row < next || std::uint64_t{s.run_row} + s.rows > run_rows)
                throw std::invalid_argument("embed splice: segment outside the run or out of order");
            next = std::uint64_t{s.run_row} + s.rows;
        }
    }
    // extras flag of every run row (0 for token rows), run_rows entries.
    [[nodiscard]] std::vector<std::uint8_t> extras_rows(std::uint64_t run_rows) const {
        std::vector<std::uint8_t> out(run_rows, 0);
        for (const auto& s : segments)
            if (s.extras)
                for (std::uint32_t i = 0; i < s.rows; ++i) out[std::size_t{s.run_row} + i] = 1;
        return out;
    }
};

// Device bytes an embed stage stages for at most `rows` spliced rows of one run (the rows themselves; the
// shared kernel needs no index arrays).
[[nodiscard]] constexpr std::uint64_t embed_splice_staging_bytes(std::uint64_t rows, std::uint64_t width) noexcept {
    return rows * width * 2U;
}

struct EmbedResolveRules {
    std::uint32_t width = 0; // LM hidden width
    // The family's position of a row is its prompt index: a span must appear in row order without gaps
    // (a run may start inside a span), and its lm_pos, when given, must be 0, 1, 2, ... on axis 0.
    bool prompt_index_positions = true;
    bool allow_deep = false; // rows added after later layers (EmbedRows::deep)
};

struct ResolvedEmbedRun {
    std::vector<std::uint32_t> tokens; // vocabulary ids of every run row
    EmbedSplice splice;
};

// Throws std::invalid_argument: a virtual id without a table, an unknown handle or row, a span of another
// width, or a span the rules refuse.
[[nodiscard]] inline ResolvedEmbedRun resolve_embed_run(std::span<const std::uint32_t> ids,
                                                        const std::shared_ptr<const EmbedTable>& table,
                                                        const EmbedResolveRules& rules) {
    ResolvedEmbedRun out;
    out.tokens.assign(ids.begin(), ids.end());
    out.splice.width = rules.width;
    const auto fail = [](const char* what) { throw std::invalid_argument(std::string("embed rows: ") + what); };
    std::uint32_t last_handle = 0, last_row = 0;
    bool last_virtual = false;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const std::uint32_t id = ids[i];
        if (!is_embed_id(id)) {
            last_virtual = false;
            continue;
        }
        if (!table) fail("a virtual id without a bound table");
        const std::uint32_t handle = embed_handle(id), row = embed_row(id);
        const EmbedRows* span = table->find(handle);
        if (span == nullptr || row >= span->rows) fail("a virtual id outside the bound table");
        if (span->width != rules.width || rules.width == 0U) fail("span width differs from the model");
        if (!rules.allow_deep && !span->deep.empty()) fail("the family takes no deep rows");
        const bool follows = last_virtual && last_handle == handle && last_row + 1U == row;
        if (rules.prompt_index_positions) {
            if (row != 0U && i != 0U && !follows) fail("a span is not in row order");
            if (span->axes != 0U && span->lm_pos.at(std::size_t{row} * span->axes) != static_cast<std::int32_t>(row))
                fail("span positions are not the prompt index");
        }
        out.tokens[i] = span->real_token_id;
        if (follows) {
            ++out.splice.segments.back().rows;
        } else {
            EmbedSegment s;
            s.run_row = static_cast<std::uint32_t>(i);
            s.rows = 1;
            s.data = span->row(row).data();
            out.splice.segments.push_back(s);
        }
        last_virtual = true;
        last_handle = handle;
        last_row = row;
    }
    if (!out.splice.segments.empty()) out.splice.keep = table;
    return out;
}

} // namespace strix::session
