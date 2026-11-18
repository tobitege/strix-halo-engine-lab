#pragma once
// Per-request two-node mode selection (regime auto-select: a lever faster only in some regimes ships with
// per-regime selection). Pure policy, shared by every engine's pair route and server scheduler; engines
// supply measured thresholds. Modes:
//   single    - the request runs on one node (the other node serves its own queue);
//   pipeline  - both nodes prefill the prompt as a two-stage layer pipeline, one node then decodes;
//   disaggregate - the idle node prefills, the K/V state moves to the node already decoding, which
//               batches the new stream with its active ones (decode never waits for prompt chunks).
//   tensor    - the resident tensor-parallel route of a split-weight pair (both ranks run every layer on half
//               the weights).
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix::pair {
enum class Mode { single, pipeline, disaggregate, tensor };
[[nodiscard]] inline const char* mode_name(Mode mode) {
    switch (mode) {
    case Mode::single: return "single";
    case Mode::pipeline: return "pipeline";
    case Mode::disaggregate: return "disaggregate";
    case Mode::tensor: return "tensor";
    }
    return "?";
}
struct Load {
    std::size_t prompt = 0;          // prompt tokens of the request being placed
    std::size_t chunks = 0;          // its pipeline chunks under the engine's chunk schedule
    std::size_t queued = 0;          // requests waiting behind it (both nodes)
    std::size_t active_decodes = 0;  // streams decoding on the decode node
    bool peer_idle = true;           // the other node has no prefill or decode of its own
};
struct Policy {
    std::size_t min_pipeline_chunks = 2;   // one chunk has nothing to overlap: the seam only adds latency
    std::size_t min_disaggregate_prompt = 512; // below this the prompt chunk stalls decode less than the move costs
};
[[nodiscard]] inline Mode choose(const Load& load, const Policy& policy) {
    // A busy peer or a queue: two independent nodes give more throughput than one shared prompt.
    if (!load.peer_idle || load.queued > 0) return Mode::single;
    if (load.active_decodes > 0)
        return load.prompt >= policy.min_disaggregate_prompt ? Mode::disaggregate : Mode::single;
    return load.chunks >= policy.min_pipeline_chunks ? Mode::pipeline : Mode::single;
}

// Length-aware two-stage pipeline rule for an idle pair (engines supply their measured costs).
// Per chunk of c rows both stages together cost P(c) = fixed_ms + ms_per_row x c. The critical path of
// N balanced chunks is stage A of chunk 0, then every chunk of stage B:
//   wall(N) = (N + 1) / 2 x P(n / N) + seam_ms + handoff_fixed_ms + handoff_ms_per_row x n.
// The chunk minimising wall wins; the pipeline is chosen only if that beats single_ms_per_row x n.
// This estimate averages rows across chunks and assumes balanced stages. Execution uses full chunks
// and a shorter tail: qualify uneven tails and unbalanced cuts before treating estimates as timings.
struct PipelineCost {
    double fixed_ms = 0.0;           // per chunk, both stages
    double ms_per_row = 0.0;         // per chunk row, both stages
    double seam_ms = 0.0;            // first-frame seam latency (reaches the critical path once)
    double handoff_fixed_ms = 0.0;   // state hand-off to the decode rank after the prefill
    double handoff_ms_per_row = 0.0;
    double single_ms_per_row = 0.0;  // one node with its own tiles, all in
    // Interleaved virtual stages (every stage boundary is one hop between the ranks):
    double hop_latency_ms = 0.0;     // per frame, on top of the transfer (seam_ms is the one-off first frame)
    double hop_ms_per_row = 0.0;     // per frame row (activation bytes / link rate)
    double tp_ms_per_row = 0.0;      // resident tensor-parallel route of a split-weight pair (0: none)
    double min_gain = 0.03;          // run noise: the layout-aware rule pipelines only if it wins by this fraction
};
struct PipelineChoice {
    Mode mode = Mode::single;
    std::uint32_t chunk = 0;         // rows per chunk of the best pipeline (0: no legal chunk)
    std::uint64_t chunks = 0;
    double pipeline_ms = 0.0;        // predicted wall of that pipeline
    double single_ms = 0.0;          // predicted one-node wall
    std::uint32_t virtual_stages = 1; // stages per rank (v; 2v stages alternate between the ranks)
    double alt_ms = 0.0;             // predicted wall of the alternative (tensor route or one node)
    std::string reason;              // printed in the effective config
};
[[nodiscard]] inline double pipeline_wall_ms(std::uint64_t rows, std::uint32_t chunk, const PipelineCost& cost) {
    if (rows == 0U || chunk == 0U) return 0.0;
    const auto chunks = static_cast<double>((rows + chunk - 1U) / chunk);
    const auto n = static_cast<double>(rows);
    return (chunks + 1.0) / 2.0 * (cost.fixed_ms + cost.ms_per_row * n / chunks) + cost.seam_ms +
        cost.handoff_fixed_ms + cost.handoff_ms_per_row * n;
}
// `candidates` in ascending order (ties keep the smaller chunk); `legal(chunk)` is the engine's tile rule.
// A chunk covering the whole prompt has nothing to overlap and is never a pipeline.
template <class Legal>
[[nodiscard]] PipelineChoice choose_pipeline(std::uint64_t rows, const PipelineCost& cost,
    std::span<const std::uint32_t> candidates, Legal&& legal) {
    PipelineChoice best;
    best.single_ms = cost.single_ms_per_row * static_cast<double>(rows);
    for (const auto chunk : candidates) {
        if (chunk == 0U || chunk >= rows || !legal(chunk)) continue;
        const auto wall = pipeline_wall_ms(rows, chunk, cost);
        if (best.chunk == 0U || wall < best.pipeline_ms) {
            best.chunk = chunk;
            best.chunks = (rows + chunk - 1U) / chunk;
            best.pipeline_ms = wall;
        }
    }
    if (best.chunk != 0U && best.pipeline_ms < best.single_ms) best.mode = Mode::pipeline;
    return best;
}

// Weight residency of a pair layout. A split-weight (tensor_split) pair can pipeline only if each node can
// also hold its whole stage layers next to the decode layout.
enum class Residency : std::uint8_t { whole_model, stage_layers, tensor_split };
struct LayoutDecl {
    Residency weights = Residency::whole_model;
    double node_free_bytes = 0.0;          // after load + workspace (engine-measured MemAvailable)
    double whole_stage_extra_bytes = 0.0;  // to hold whole stage layers next to the decode layout
    const char* reason = "";               // file:line of the split dependency (printed)
};

// Interleaved virtual stages: C chunks x 2v stages; stage s runs on rank s % 2 and its output (boundary s)
// crosses to the other rank, forward (rank 0 -> 1, even boundaries, v per chunk) or reverse (odd, v - 1).
// Unbound host planning API: no engine stage-pipe execution or layout adapter is bound here yet.
// Proposed execution order per rank: chunks in groups of at most
// `group`; within a group depth-first (round k: every chunk of the group through stage 2k + rank). A fully
// depth-first order over C chunks would buffer a whole round of reverse frames; with the two-slot pipe the
// producer stalls for good once a group exceeds 2 x 2 + 1 = 5 chunks, so groups are capped at 4.
inline constexpr std::uint32_t stage_group_chunks = 4U;
// Bound host simulation storage and work to one million stage/chunk events (tens of MiB).
inline constexpr std::uint64_t max_stage_events = 1U << 20U;
[[nodiscard]] inline bool valid_stage_shape(std::uint64_t chunks, std::uint32_t v, std::uint32_t group) {
    return chunks != 0U && v != 0U && v <= max_stage_events / 2U && group != 0U &&
        chunks <= max_stage_events / (2U * v);
}
// Position of (chunk, boundary) in its direction's frame order (the pipe sequence).
[[nodiscard]] inline std::uint64_t stage_sequence(std::uint64_t chunk, std::uint32_t boundary, std::uint64_t chunks,
    std::uint32_t v, std::uint32_t group = stage_group_chunks) {
    if (!valid_stage_shape(chunks, v, group) || chunk >= chunks || boundary >= 2U * v - 1U)
        throw std::invalid_argument("pair stage sequence: invalid or oversized shape/index");
    const std::uint64_t rounds = boundary % 2U == 0U ? v : v - 1U;
    const std::uint64_t base = chunk - chunk % group;
    const std::uint64_t size = std::min<std::uint64_t>(group, chunks - base);
    return base * rounds + (boundary / 2U) * size + (chunk - base);
}
// Deterministic event simulation of that order: stage time P(n / C) / 2v (balanced chunks and stages); hop =
// hop_latency_ms + hop_ms_per_row x rows on one FIFO link per direction; the producer of frame q waits until
// frame q - 2 of its direction is consumed (two slots); plus seam_ms and the hand-off terms. Returns +inf for
// an invalid/oversized shape or an order that cannot complete. Empty rows cost zero.
// At v = 1 with no hop cost this equals the averaged-tail analytical pipeline_wall_ms estimate.
[[nodiscard]] inline double interleaved_wall_ms(std::uint64_t rows, std::uint32_t chunk, std::uint32_t v,
    const PipelineCost& cost, std::uint32_t group = stage_group_chunks) {
    if (chunk == 0U || v == 0U || group == 0U) return std::numeric_limits<double>::infinity();
    if (rows == 0U) return 0.0;
    const std::uint64_t chunks = rows / chunk + (rows % chunk != 0U);
    if (!valid_stage_shape(chunks, v, group)) return std::numeric_limits<double>::infinity();
    const auto n = static_cast<double>(rows);
    const double u = n / static_cast<double>(chunks);
    const std::uint32_t stages = 2U * v;
    const double t = (cost.fixed_ms + cost.ms_per_row * u) / static_cast<double>(stages);
    const double hop = cost.hop_latency_ms + cost.hop_ms_per_row * u;
    constexpr double unset = -1.0;
    std::vector<double> arrival(chunks * stages, unset);       // input of (stage s + 1, chunk) on the other rank
    std::vector<double> consumed[2] = {std::vector<double>(chunks * v, unset), std::vector<double>(chunks * v, unset)};
    std::vector<std::pair<std::uint32_t, std::uint64_t>> order[2];  // (stage, chunk) per rank
    for (std::uint64_t base = 0; base < chunks; base += group)
        for (std::uint32_t s = 0; s < stages; ++s)
            for (std::uint64_t c = base; c < std::min<std::uint64_t>(chunks, base + group); ++c) order[s % 2U].emplace_back(s, c);
    std::size_t next[2] = {0, 0};
    double rank_free[2] = {0.0, 0.0}, link_free[2] = {0.0, 0.0}, last = 0.0;
    double computed[2] = {unset, unset};  // an op that began and waits for its output slot
    bool moved = true;
    while (moved) {
        moved = false;
        for (unsigned r = 0; r < 2U; ++r) {
            bool blocked = false;
            while (!blocked && next[r] < order[r].size()) {
                const std::uint32_t s = order[r][next[r]].first;
                const std::uint64_t c = order[r][next[r]].second;
                if (computed[r] == unset) {  // begin: the input frame is consumed (copied in, slot released)
                    double begin = rank_free[r];
                    if (s > 0U) {
                        const double in = arrival[(s - 1U) * chunks + c];
                        if (in == unset) { blocked = true; continue; }
                        begin = std::max(begin, in);
                        consumed[(s - 1U) % 2U][stage_sequence(c, s - 1U, chunks, v, group)] = begin;
                    }
                    computed[r] = begin + t;
                    moved = true;
                }
                double done = computed[r];
                if (s + 1U < stages) {  // the output needs its slot: frame q - 2 of this direction consumed
                    const std::uint64_t q = stage_sequence(c, s, chunks, v, group);
                    if (q >= 2U) {
                        const double free_at = consumed[s % 2U][q - 2U];
                        if (free_at == unset) { blocked = true; continue; }
                        done = std::max(done, free_at);
                    }
                    const double start = std::max(done, link_free[r]);
                    link_free[r] = start + hop;
                    arrival[s * chunks + c] = start + hop;
                }
                computed[r] = unset;
                rank_free[r] = done;
                last = std::max(last, done);
                ++next[r];
                moved = true;
            }
        }
    }
    if (next[0] < order[0].size() || next[1] < order[1].size()) return std::numeric_limits<double>::infinity();
    return last + cost.seam_ms + cost.handoff_fixed_ms + cost.handoff_ms_per_row * n;
}

// Layout-aware pipeline rule. `chunks` and `virtual_stages` ascending (ties keep the smaller chunk, then the
// smaller v); `legal(chunk)` is the engine's tile rule, `legal_v(v)` its stage-map rule (2v stages fit the
// layers). The alternative is the resident tensor route when tp_ms_per_row > 0, else one node (0: none). A
// tensor_split layout whose whole stage layers do not fit next to the decode layout stays on the tensor route.
template <class Legal, class LegalV>
[[nodiscard]] PipelineChoice choose_pipeline(std::uint64_t rows, const PipelineCost& cost,
    std::span<const std::uint32_t> chunks, Legal&& legal, std::span<const std::uint32_t> virtual_stages,
    LegalV&& legal_v, const LayoutDecl& layout) {
    PipelineChoice best;
    const auto n = static_cast<double>(rows);
    best.single_ms = cost.single_ms_per_row * n;
    const bool tensor = cost.tp_ms_per_row > 0.0;
    const Mode alt_mode = tensor ? Mode::tensor : Mode::single;
    best.alt_ms = tensor ? cost.tp_ms_per_row * n : best.single_ms;
    best.mode = alt_mode;
    for (const auto chunk : chunks) {
        if (chunk == 0U || chunk >= rows || !legal(chunk)) continue;
        for (const auto v : virtual_stages) {
            if (v == 0U || !legal_v(v)) continue;
            const auto wall = interleaved_wall_ms(rows, chunk, v, cost);
            if (wall == std::numeric_limits<double>::infinity()) continue;
            if (best.chunk == 0U || wall < best.pipeline_ms) {
                best.chunk = chunk;
                best.chunks = rows / chunk + (rows % chunk != 0U);
                best.virtual_stages = v;
                best.pipeline_ms = wall;
            }
        }
    }
    char text[256];
    const char* alt_name = mode_name(alt_mode);
    if (layout.weights == Residency::tensor_split && layout.whole_stage_extra_bytes > layout.node_free_bytes) {
        std::snprintf(text, sizeof text, "layout: %s needs +%.1f GB/node, free %.1f GB", layout.reason,
            layout.whole_stage_extra_bytes / 1e9, layout.node_free_bytes / 1e9);
        best.mode = Mode::tensor;
    } else if (best.chunk == 0U) {
        std::snprintf(text, sizeof text, "no legal chunk and virtual stage: %s", alt_name);
    } else if (best.alt_ms <= 0.0 || best.pipeline_ms < best.alt_ms * (1.0 - cost.min_gain)) {
        std::snprintf(text, sizeof text, "length: C=%llu v=%u %.0f ms beats %s %.0f ms",
            static_cast<unsigned long long>(best.chunks), best.virtual_stages, best.pipeline_ms, alt_name, best.alt_ms);
        best.mode = Mode::pipeline;
    } else {
        std::snprintf(text, sizeof text, "min_gain: best pipeline %.0f ms not %.0f %% under %s %.0f ms",
            best.pipeline_ms, cost.min_gain * 100.0, alt_name, best.alt_ms);
    }
    best.reason = text;
    return best;
}

// Host planning diagnostic; candidate_pipeline_ms can describe a rejected candidate.
[[nodiscard]] inline std::string effective_line(const PipelineChoice& choice) {
    char text[160];
    std::snprintf(text, sizeof text, "pair.pipeline mode=%s chunk=%u virtual=%u candidate_pipeline_ms=%.0f alt_ms=%.0f reason=",
        mode_name(choice.mode), choice.chunk, choice.virtual_stages, choice.pipeline_ms, choice.alt_ms);
    return text + choice.reason;
}
} // namespace strix::pair
