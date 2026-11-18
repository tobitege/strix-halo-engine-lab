#pragma once
// Qwen pair interleaved virtual stages (MMPIPE-PLAN PR-3; design option C from main, 16-11 11:30).
// One session and one chunk cursor per rank stay. Each advance runs one (stage, layer range, chunk) step; the
// layer range is passed in the step (qwen_pp::pipe_context.step), not read from a process static. The frontier is
// a per-stage position vector inside the same session (StageFrontier, swapped in around each step), the lineage
// is chained per stage, and the session lineage is the fold of the stage lineages in stage order, so v = 1 (one
// stage per rank) is exactly today's chain. Home: specific (Qwen adapter); the shared layer (pair/stage_map.hpp,
// pair/regime.hpp) stays generic.
#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/arithmetic_lineage.hpp"
#include "strix/pair/stage_map.hpp"
#include "strix/qwen_prefill_pipeline.hpp"

namespace strix::qwen_pp {
// Layers of stage `stage`. A map with a single stage pair covering the model end to end is never "split".
inline LayerRange stage_range(const pair::StageMap& map, std::uint32_t stage) {
    if (stage + 1U >= map.cuts.size()) throw std::out_of_range("qwen stage range: no such stage");
    const auto text = map.cuts[stage].text() + ":" + map.cuts[stage + 1U].text();
    return parse_range(text.c_str(), map.layers());
}

// One stage's session frontier: what one session frontier was before stages, per stage.
struct StageFrontier {
    std::uint64_t next_position = 0;
    std::uint64_t committed_tokens = 0;
    ArithmeticLineage lineage;
};

// Every stage runs the same canonical chunks with the same plan identities, so each stage chain equals the
// one-stage chain; the fold takes the first stage's chain and taints the result if any later stage differs.
inline ArithmeticLineage fold_stage_lineages(std::span<const StageFrontier> stages) {
    if (stages.empty()) throw std::invalid_argument("qwen stage lineage fold needs a stage");
    ArithmeticLineage out = stages.front().lineage;
    for (const auto& stage : stages.subspan(1U))
        if (stage.lineage.tainted || stage.lineage.intervals != out.intervals ||
            stage.lineage.history_identity != out.history_identity || stage.next_position != stages.front().next_position)
            out.tainted = true;
    return out;
}

// One-node order of every stage (the identity emulation): groups of stage_group_chunks chunks, stage-major inside a
// group. It is the merge of both ranks' StagePipes orders and respects (s-1, c) -> (s, c) and (s, c-1) -> (s, c).
inline std::vector<pair::StageStep> local_order(std::uint32_t stages, std::uint64_t chunks) {
    std::vector<pair::StageStep> order;
    for (std::uint64_t base = 0; base < chunks; base += pair::stage_group_chunks)
        for (std::uint32_t s = 0; s < stages; ++s)
            for (std::uint64_t c = base; c < std::min<std::uint64_t>(chunks, base + pair::stage_group_chunks); ++c)
                order.push_back({s, c});
    return order;
}

// Even map of 2v stages over `layers` whole layers (identity emulation and the v > 1 default before a measured table).
inline pair::StageMap even_map(std::uint32_t v, std::uint32_t layers) {
    if (v == 0U || v > layers / 2U) throw std::invalid_argument("qwen stage map: bad virtual stage count");  // no 2v wrap
    pair::StageMap map;
    const std::uint64_t stages = 2ULL * v;
    for (std::uint64_t s = 0; s <= stages; ++s) map.cuts.push_back(pair::StageCut(static_cast<std::uint32_t>(s * layers / stages)));
    map.validate(layers);
    return map;
}

// A serial row has no seam (the pair path refuses such a plan too): with more than one stage, every stage step would
// run its whole model for that row on the one emulated session. One stage runs the plan as the unsplit step loop.
inline void require_stage_seams(std::uint32_t stages, std::uint64_t serial_rows) {
    if (stages > 1U && serial_rows)
        throw std::invalid_argument("stage emulation plan has " + std::to_string(serial_rows) +
            " serial rows (no seam for a serial row; the tail owners must cover the prompt on the 16-row grain)");
}
} // namespace strix::qwen_pp
