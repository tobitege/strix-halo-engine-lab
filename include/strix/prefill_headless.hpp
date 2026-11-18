#pragma once

#include <cstddef>
#include <cstdint>

// Headless chunks of a chunked prefill (shared host policy, every engine).
//
// A chunked prefill runs a prompt as consecutive chunks, layer after layer. After the LAST layer only the last chunk's
// rows are read (the head takes the prompt's last row; logits / output digests read the last chunk). Every other chunk
// is "headless" at that layer: nothing reads its attention output, its output projection, its FFN or its exchanges.
// What the layer must still write for such a chunk is its persistent state (the K / V cache rows and, with sparse
// attention, the index-key rows), because the last chunk's attention reads them.
//
// So a headless chunk's last layer runs cache-only: the input projections and the cache writes unchanged (same
// launches, same bits: the last chunk's attention sees the same caches), nothing after them. The rows that are read
// keep their bits; the work removed is (chunks - 1) / chunks of one layer's attention output + FFN (+ the pair's two
// exchanges of that layer for those chunks). The Qwen launcher's own form of this rule is strix/qsa_cache_only.hpp.
//
// The engine says when its rows are observed beyond the last chunk (layer taps for a draft model, per-row output
// digests, a stack that does not end at the last layer); then no chunk is headless. Both ranks of a pair evaluate the
// same inputs, so they skip the same messages.
namespace strix::prefill_headless {

struct Query {
    unsigned layer = 0;      // the layer about to run
    unsigned layers = 0;     // the model's layer count
    unsigned chunk = 0;      // chunk index in the prompt's unit
    unsigned chunks = 0;     // chunks in the unit
    bool stack_end = false;  // this unit reaches the model's last layer; it need not end the prompt
    bool observed = false;   // rows of non-last chunks are read after the layer (taps, digests, explicit row outputs)
};

// True when `q.chunk` may run `q.layer` cache-only.
constexpr bool cache_only(const Query& q) noexcept {
    return q.stack_end && !q.observed && q.layers != 0 && q.layer + 1 == q.layers && q.chunk + 1 < q.chunks;
}

// Number of leading headless chunks of the unit at `layer` (all of them or none: the rule does not depend on the
// chunk's rows, only on its position).
constexpr unsigned headless_chunks(unsigned layer, unsigned layers, unsigned chunks, bool stack_end, bool observed) noexcept {
    return cache_only({layer, layers, 0, chunks, stack_end, observed}) ? chunks - 1 : 0;
}

// Qualification target for a complete grouped prefill: every unit keeps its last
// chunk full, including a partial final unit. Independent of runtime admission so
// disabled/unsupported routes cannot qualify merely by reporting zero work.
constexpr std::size_t expected_chunks(std::size_t rows, std::size_t chunk_rows, std::size_t unit_chunks) noexcept {
    if (rows == 0 || chunk_rows == 0 || unit_chunks < 2) return 0;
    const auto chunks = rows / chunk_rows + (rows % chunk_rows != 0 ? 1U : 0U);
    const auto units = chunks / unit_chunks + (chunks % unit_chunks != 0 ? 1U : 0U);
    return chunks - units;
}

// A zero target is an inactive/unqualified experiment, even if hashes match.
constexpr bool qualified_arm(bool enabled, std::uint64_t before, std::uint64_t after, std::size_t expected) noexcept {
    return expected != 0 && after >= before && after - before == (enabled ? expected : 0);
}

}  // namespace strix::prefill_headless
