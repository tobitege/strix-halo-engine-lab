#pragma once
#include "strix/serve/json.hpp"
#include "strix/pair/balance.hpp"
#include "strix/pair/pipe.hpp"
#include "strix/pair/regime.hpp"
#include "strix/lever.hpp"
#include <array>
#include <algorithm>
#include <vector>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <span>
#include <string>

namespace strix::qwen_pp {
using LegalCuts = std::array<bool, 49>;
inline std::uint32_t validate_cut(std::uint32_t cut, const LegalCuts& legal) {
    if (cut == 0U || cut >= 48U || !legal[cut])
        throw std::invalid_argument("pair prefill cut requires an interior folded attention-HC boundary of the reference kind");
    return cut;
}
// Unset STRIX_QWEN_PF_PIPE_CUT: stage A ends after layer 22 (cut 23), measured fastest with 4,096-row chunks
// (32K 12.08 s vs 12.44-12.51 s at 24; 128K 47.53-47.55 s vs 49.36-50.15 s; 20-22 slower), because stage A also
// runs the embedding/PLE prelude and the seam pack. Falls back to 24 when 23 is not a legal boundary.
// STRIX_QWEN_PF_PIPE_CUT=24 restores the even split. With stage B timed honestly (it drains the device before
// waiting for a frame) the stages at 23 are 10.0-10.1 s and 10.2 s at 32K; half-layer cuts 22.5 and 23.5 measured
// 11.82-11.85 s against 11.61-11.69 s at 23, so 23 stays. Since then (length chunk policy, faster QSA) the
// half cut 23.5 wins below 64K context and loses at 128K: qwen.pair_half_cut (below) picks it per prompt.
inline constexpr std::uint32_t measured_default_cut = 23U;
inline std::uint32_t parse_cut(const char* value, const LegalCuts& legal) {
    std::uint32_t cut = legal[measured_default_cut] ? measured_default_cut : 24U;
    if (value) {
        const std::string_view text(value);
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), cut);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
            throw std::invalid_argument("STRIX_QWEN_PF_PIPE_CUT requires a layer number");
    }
    return validate_cut(cut, legal);
}
// STRIX_QWEN_PF_PIPE_CUT=N.5 cuts inside layer N, after its attention sub-block (stage A runs the attention of
// layer N, stage B its expert half). `legal_half[N]` marks layers whose feed-forward mix can carry the seam.
inline pair::StageCut parse_stage_cut(const char* value, const LegalCuts& legal, const LegalCuts& legal_half) {
    if (!value || std::string_view(value).find('.') == std::string_view::npos) return parse_cut(value, legal);
    pair::StageCut cut;
    try { cut = pair::StageCut::parse(value); }
    catch (const std::invalid_argument&) {
        throw std::invalid_argument("STRIX_QWEN_PF_PIPE_CUT requires a layer number or a half layer (N.5)");
    }
    if (cut.layer == 0U || cut.layer >= 48U || !legal_half[cut.layer])
        throw std::invalid_argument("pair prefill half-layer cut requires an interior feed-forward HC boundary");
    return cut;
}
// Whole cuts (first = true): the layer opens with `mix` and has `inject`. Half cuts: it has both, in that order.
template<class Steps, class Op> LegalCuts legal_cuts(const Steps& steps, Op mix, Op inject, bool first) {
    LegalCuts legal{};
    for (std::uint32_t layer = 1; layer < 48U; ++layer) {
        bool opened = false, mixed = false, injected = false, valid = true;
        for (const auto& step : steps) {
            if (step.layer != layer) continue;
            if (step.op == mix) { if (first && opened) valid = false; mixed = true; }
            else if (step.op == inject) { if (!mixed) valid = false; injected = true; }
            opened = true;
        }
        legal[layer] = valid && mixed && injected;
    }
    return legal;
}
struct LayerCosts {
    std::array<double, 48> layers{};
    // Optional "attention": per-layer cost of the attention sub-block (part of `layers`); enables half-layer cuts
    // in the cost-model choice. Without it an explicit half cut is predicted as half of its layer.
    std::array<double, 48> attention{};
    bool has_attention = false;
    double lm_head = 0;
    std::uint64_t context = 0;
    // Measured sender staging/CRC/wire cost; optional for old compute-only files.
    double transfer_ms_per_row = 0;
    double fixed_ms = 0;
    bool has_transfer = false;
};
inline const serve::Json& cost_field(const serve::Json& object, std::string_view key) {
    const auto* value = object.find(key);
    if (!value) throw std::invalid_argument("pair prefill costs missing field: " + std::string(key));
    return *value;
}
inline double cost_number(const serve::Json& value) {
    if (!value.is_number()) throw std::invalid_argument("pair prefill costs must be numbers");
    const double number = value.as_double();
    if (!std::isfinite(number) || number < 0) throw std::invalid_argument("pair prefill costs must be finite and nonnegative");
    return number;
}
// Schema: {"buckets":[{"context":32768,"layers":{"0":ms_per_row,...,"47":ms_per_row},"lm_head":ms_per_row}]}.
// Use the smallest covering context bucket, or the largest bucket beyond calibration.
inline LayerCosts parse_costs(std::string_view text, std::uint64_t context) {
    const auto root = serve::parse_json(text);
    const auto& buckets = cost_field(root, "buckets").as_array();
    if (buckets.empty()) throw std::invalid_argument("pair prefill costs require at least one context bucket");
    LayerCosts selected;
    bool found = false;
    std::vector<std::uint64_t> seen;
    for (const auto& bucket : buckets) {
        const auto bucket_context = cost_field(bucket, "context").as_int();
        if (bucket_context <= 0) throw std::invalid_argument("pair prefill cost context must be positive");
        LayerCosts candidate;
        candidate.context = static_cast<std::uint64_t>(bucket_context);
        const auto* transfer = bucket.find("transfer_ms_per_row");
        const auto* fixed = bucket.find("fixed_ms");
        if ((transfer == nullptr) != (fixed == nullptr))
            throw std::invalid_argument("pair prefill costs require both transfer_ms_per_row and fixed_ms");
        if (transfer) {
            candidate.transfer_ms_per_row = cost_number(*transfer);
            candidate.fixed_ms = cost_number(*fixed);
            candidate.has_transfer = true;
        }
        if (std::find(seen.begin(), seen.end(), candidate.context) != seen.end())
            throw std::invalid_argument("pair prefill costs contain duplicate context buckets");
        seen.push_back(candidate.context);
        const auto& layers = cost_field(bucket, "layers");
        if (!layers.is_object() || layers.as_object().size() != candidate.layers.size())
            throw std::invalid_argument("pair prefill costs require exactly 48 layer costs");
        double total = 0;
        for (std::size_t layer = 0; layer < candidate.layers.size(); ++layer) {
            candidate.layers[layer] = cost_number(cost_field(layers, std::to_string(layer)));
            total += candidate.layers[layer];
        }
        if (const auto* attention = bucket.find("attention")) {
            if (!attention->is_object() || attention->as_object().size() != candidate.attention.size())
                throw std::invalid_argument("pair prefill costs require exactly 48 attention costs");
            for (std::size_t layer = 0; layer < candidate.attention.size(); ++layer) {
                candidate.attention[layer] = cost_number(cost_field(*attention, std::to_string(layer)));
                if (candidate.attention[layer] > candidate.layers[layer])
                    throw std::invalid_argument("pair prefill attention cost exceeds its layer cost");
            }
            candidate.has_attention = true;
        }
        candidate.lm_head = cost_number(cost_field(bucket, "lm_head"));
        if (!std::isfinite(total + candidate.lm_head) || total <= 0)
            throw std::invalid_argument("pair prefill costs require positive finite layer work");
        if (!found || (candidate.context >= context && (selected.context < context || candidate.context < selected.context)) ||
            (candidate.context < context && selected.context < context && candidate.context > selected.context)) {
            selected = candidate; found = true;
        }
    }
    return selected;
}
inline LayerCosts load_costs(const char* path, std::uint64_t context) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::invalid_argument("cannot open STRIX_QWEN_PF_PIPE_COSTS file");
    const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    return parse_costs(text, context);
}
struct CutPrediction {
    std::uint32_t cut = 24; double a = 0; double b = 0; bool half = false;
    pair::StageCut stage() const noexcept { return {cut, half}; }
};
inline std::array<double, 48> attention_costs(const LayerCosts& costs) {
    if (costs.has_attention) return costs.attention;
    auto front = costs.layers;
    for (auto& value : front) value *= 0.5;
    return front;
}
inline CutPrediction predict_cut(const LayerCosts& costs, pair::StageCut cut) {
    const auto split = pair::predict_split(costs.layers, attention_costs(costs), costs.lm_head, cut.layer, cut.half);
    return {split.cut, split.a, split.b, split.half};
}
// Half-layer cuts compete only when the costs carry the attention sub-block times.
inline CutPrediction choose_cut(const LayerCosts& costs, const LegalCuts& legal, const LegalCuts& legal_half = {}) {
    try {
        const auto split = pair::balance_split(costs.layers, attention_costs(costs), costs.lm_head,
            [&](std::uint32_t cut) { return legal[cut]; },
            [&](std::uint32_t cut) { return costs.has_attention && legal_half[cut]; });
        return {split.cut, split.a, split.b, split.half};
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("pair prefill model has no legal folded attention-HC cut");
    }
}
// Largest prefix the pipeline carries: full tiles, then a remainder on the 16-row tail grain. The cursor runs a
// remainder as a merged partial block and 64 / 16-row tail blocks, and only its last rem % 16 rows serially; a
// serial row has no seam (each rank would run its own whole model), so those rows go to the hand-off suffix.
inline constexpr std::uint32_t pipe_tail_grain = 16U;
inline std::uint64_t legal_prefix(std::uint64_t prompt, std::uint32_t rows) {
    if (rows < 16U || rows % 8U != 0U) throw std::invalid_argument("pair handoff invalid tile rows");
    return prompt - prompt % pipe_tail_grain;
}
struct PromptRoute { std::uint64_t prefix = 0; const char* reason = "single-short"; };
inline PromptRoute choose_prompt_route(std::uint64_t prompt, std::uint32_t rows,
    const LayerCosts* costs, const CutPrediction& prediction) {
    const auto prefix = legal_prefix(prompt, rows);
    if (prefix <= rows) return {};
    // Conservative fallback from the supplied 32K/128K end-to-end receipts:
    // savings = 1.84s + (N-32768)*23.01s/98304, crossing at 24907 rows.
    // Round up to a 4096-row boundary (the measured default organization).
    // This is a routing estimate, not qualification of short-context speed.
    if (!costs || !costs->has_transfer)
        return prefix >= 28672U ? PromptRoute{prefix, "pipeline-measured-estimate"} : PromptRoute{};
    const auto n = static_cast<double>(prefix);
    const auto tile = static_cast<double>(rows);
    const auto chunks = static_cast<double>((prefix + rows - 1U) / rows);
    const auto transfer = costs->transfer_ms_per_row * tile + costs->fixed_ms;
    const auto stage = std::max({prediction.a * tile, prediction.b * tile, transfer});
    // Keep the old, slower handoff fit until the overlapped path is measured.
    const auto handoff_ms = 266.6666667 + n * (5200.0 / 98304.0);
    const auto pipe_ms = (prediction.a + prediction.b) * tile + transfer +
        (chunks - 1.0) * stage + handoff_ms;
    return pipe_ms < (prediction.a + prediction.b) * n ?
        PromptRoute{prefix, "pipeline-cost-model"} : PromptRoute{0U, "single-cost-model"};
}
struct ChunkChoice {
    std::uint32_t rows = 0;
    const char* reason = "legacy";
};
inline bool valid_chunk(std::uint32_t rows, std::uint64_t prompt) {
    return rows >= 16U && rows <= 16384U && rows % 8U == 0U &&
        prompt != 0U && prompt % 8U == 0U && (prompt % rows == 0U || prompt % rows >= 16U);
}
inline ChunkChoice choose_chunk(const char* value, std::uint32_t legacy, std::uint64_t prompt,
    const LayerCosts* costs, const CutPrediction& prediction) {
    prompt = legal_prefix(prompt, legacy);
    if (value && std::string_view(value) == "legacy") return {legacy, "legacy"};
    // Unset: 4096-row chunks, which were faster than legacy, 2048, 8192 and the
    // cost-model choice at 32K and 128K, all exact. Prompts where 4096 is not a legal chunk keep legacy.
    if (!value) {
        constexpr std::uint32_t measured_best = 4096U;
        if (measured_best <= legacy && measured_best < prompt && valid_chunk(measured_best, prompt))
            return {measured_best, "measured-default"};
        return {legacy, "legacy"};
    }
    if (value && std::string_view(value) != "auto") {
        std::uint32_t rows = 0;
        const std::string_view text(value);
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), rows);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
            !valid_chunk(rows, prompt) || rows > legacy)
            throw std::invalid_argument("STRIX_QWEN_PF_PIPE_CHUNK requires legacy, auto, or 16..legacy rows in multiples of 8 with a valid tail");
        return {rows, "explicit"};
    }
    if (!costs || !costs->has_transfer) return {legacy, "auto-no-transfer-costs"};
    // Both stages must cover the seam, including an unbalanced cut's faster stage.
    for (const auto rows : {1024U, 2048U, 4096U, 8192U, 16384U}) {
        if (rows > legacy || rows > prompt || !valid_chunk(rows, prompt)) continue;
        const auto b = prediction.b - (prompt > rows ? costs->lm_head : 0.0);
        const auto compute = std::min(prediction.a, b);
        if (compute * rows > costs->transfer_ms_per_row * rows + costs->fixed_ms)
            return {rows, "auto-cost-model"};
    }
    return {legacy, "auto-transfer-bound"};
}
// Measured two-stage costs (14-nov pair receipts: exact, host USB4 route, cut 23): per chunk, both stages,
// P(c) = 168 + 0.5474 c ms (2,415 at 4,096 rows, 1,288 at 2,048, 752 at 1,024); seam latency 35 ms once;
// hand-off 28 + 0.0028 n ms (51 at 8K, 120 at 32K); one node 0.536 ms/row with its own T8192/T16384 tiles.
inline constexpr pair::PipelineCost pipeline_cost{168.0, 0.5474, 35.0, 28.0, 0.0028, 0.536};
// Alternate cuts and layer-only cost files still use this balanced cut-23 calibration, not their
// supplied stage imbalance. Their crossover and native exactness remain unqualified.
inline constexpr std::array<std::uint32_t, 5> pipeline_chunk_candidates{1024U, 2048U, 4096U, 8192U, 16384U};
// qwen.pair_chunk_policy: ON = length (pair::choose_pipeline picks route and chunk per prompt length),
// OFF = fixed (4,096-row chunks, pipeline from 28,672 rows without a cost file). Both arms are bitwise.
// ON from 8,192 context tokens (measured, exact, logits equal to one node): 8K 4 x 2,048 3,302-3,309 ms
// (2,476-2,481 tok/s) vs fixed 4,358-4,365 ms; 32K identical plan (2,932-2,968 both); 128K 16 x 8,192
// 41,668 ms (3,146 tok/s) vs 32 x 4,096 42,778 ms. Below 8,192 (4K: 4 x 1,024 predicted) not measured: fixed.
inline constexpr lever::Decl chunk_policy_lever{
    "qwen.pair_chunk_policy", "STRIX_QWEN_PAIR_CHUNK_POLICY", lever::Exactness::bitwise, false,
    {lever::Axis::context, 8192U, false, true}, {},
    "pair receipts 14/15-nov: 8K 2,476-2,481 vs 1,877-1,880 tok/s; 32K equal; 128K 3,146 vs 3,064"};
// qwen.pair_single_tiles: when the pair declines a prompt (single route), rank 0 keeps the one-node tile
// instead of the pipeline chunk (8K: 4,096-row tiles 1,752 tok/s vs one node's 8,192-row tiles 1,860).
inline constexpr lever::Decl single_tiles_lever{
    "qwen.pair_single_tiles", "STRIX_QWEN_PAIR_SINGLE_TILES", lever::Exactness::bitwise, true, {}, {},
    "pair receipts 14-nov: 8K single route 1,752 tok/s at 4,096-row tiles, one node 1,853-1,867"};
// qwen.pair_half_cut: without an explicit cut or a cost file, seam inside layer 23 (23.5: stage A also runs the
// QSA attention of layer 23) instead of after it. Stage B holds 7 of the 12 QSA layers at cut 23 and is the
// critical stage; the attention block it hands over (~0.3 s at 32K, ~1.65 s at 128K) is smaller than the stage
// gap below 64K and larger at 128K (gap ~1.3 s: stage A becomes critical). ON below 65,537 context tokens
// (largest measured win), off from there; "0" keeps 23 everywhere, "1" forces 23.5. Bitwise: placement only.
inline constexpr lever::Decl half_cut_lever{
    "qwen.pair_half_cut", "STRIX_QWEN_PAIR_HALF_CUT", lever::Exactness::bitwise, false,
    {lever::Axis::context, 65537U, true, false}, {},
    "pair receipts 17-nov (exact / FAST ms, 23 vs 23.5): 8K 3,320 / 3,051 vs 3,285 / 3,003; 32K 11,176 / 10,111 vs "
    "11,105 / 10,023 (5 / 4 pairs); 64K 21,604 / 19,634 vs 21,512 / 19,369; 128K 42,489 / 38,123 vs 42,714 / 38,404"};
enum class PairExecution { handoff, prefill_only };
struct PairPolicy {
    lever::Choice chunk_policy;
    lever::Choice single_tiles;
    PairExecution execution = PairExecution::handoff;
    bool baseline = false;
    lever::Choice half_cut;
};
// Select once before read-only inspection and retain these choices for execution.
inline PairPolicy resolve_pair_policy(std::uint64_t prompt, PairExecution execution, bool baseline) {
    const lever::Inputs inputs{0U, prompt, 0U};
    const auto select = [&](lever::Decl row) {
        // Baseline disables automatic defaults; explicit per-lever switches retain precedence.
        if (baseline) { row.default_on = false; row.regime = {}; }
        return lever::select(row, inputs);
    };
    return {select(chunk_policy_lever), select(single_tiles_lever), execution, baseline, select(half_cut_lever)};
}
// The stage cut without a cost file: an explicit STRIX_QWEN_PF_PIPE_CUT wins, else the measured default 23, or
// 23.5 where qwen.pair_half_cut is on and layer 23 can carry a feed-forward seam.
inline pair::StageCut resolve_stage_cut(const char* value, const LegalCuts& legal, const LegalCuts& legal_half,
                                        bool half_cut) {
    const auto cut = parse_stage_cut(value, legal, legal_half);
    if (value || !half_cut || cut.half || cut.layer != measured_default_cut || !legal_half[cut.layer]) return cut;
    return pair::StageCut(cut.layer, true);
}
inline std::string inspect_pair_policy(std::string_view config, const PairPolicy& policy) {
    using serve::Json;
    const auto control = [&](lever::Choice choice) {
        return Json(Json::Object{
            {"requested", choice.origin == lever::Origin::explicit_switch ? (choice.on ? "1" : "0") : "auto"},
            {"effective", choice.on}, {"source", policy.baseline && choice.origin != lever::Origin::explicit_switch
                ? "baseline" : lever::origin_name(choice.origin)}});
    };
    auto out = serve::parse_json(config);
    out.as_object().emplace_back("qwen_pair", Json::Object{
        {std::string(chunk_policy_lever.env), control(policy.chunk_policy)},
        {std::string(single_tiles_lever.env), control(policy.single_tiles)},
        {std::string(half_cut_lever.env), control(policy.half_cut)},
        {"execution", policy.execution == PairExecution::handoff ? "handoff" : "prefill-only"},
        {"route", "pending-device-admission"}, {"tile_rows", nullptr}});
    return out.dump();
}
struct PairRoute {
    ChunkChoice chunk;               // pipeline chunk rows (the pipe's tile)
    PromptRoute route;               // prefix > 0: the pipeline prefills that prefix
    std::uint32_t tile_rows = 0;     // known-prefix prefill tile this process runs
    std::uint64_t chunks = 0;        // actual pipeline chunks (not the model's candidate)
    const char* policy = "fixed";
    pair::PipelineChoice model{};    // length policy only: the predicted walls
};
// The one resolver of the pair route (CLI and any server front end call this, never a second branch).
// Transfer-bearing costs retain the old cost-model route. Layer-only costs still select the cut,
// but use the calibrated length route when enabled. An explicit chunk is its only candidate.
inline PairRoute resolve_pair_route(std::uint64_t prompt, std::uint32_t legacy, const char* chunk_value,
    const LayerCosts* costs, const CutPrediction& prediction, bool length_policy, bool single_tiles,
    PairExecution execution = PairExecution::handoff) {
    PairRoute out;
    out.chunk = choose_chunk(chunk_value, legacy, prompt, costs, prediction);
    if (length_policy && !(costs && costs->has_transfer)) {
        out.policy = "length";
        const std::array<std::uint32_t, 1> explicit_rows{out.chunk.rows};
        const bool fixed_rows = chunk_value != nullptr;
        const auto candidates = fixed_rows ? std::span<const std::uint32_t>(explicit_rows)
                                           : std::span<const std::uint32_t>(pipeline_chunk_candidates);
        // Prefill-only runs the entire prompt (below), so its candidates must hold for the full prompt;
        // trimming to a legal prefix is only sound when the suffix is handed off.
        const bool full_prompt = execution == PairExecution::prefill_only;
        out.model = pair::choose_pipeline(prompt - prompt % 8U, pipeline_cost, candidates, [&](std::uint32_t rows) {
            if (full_prompt) return rows <= legacy && rows < prompt && valid_chunk(rows, prompt);
            const auto prefix = legal_prefix(prompt, rows);
            return rows <= legacy && rows < prefix && valid_chunk(rows, prefix);
        });
        if (out.model.mode == pair::Mode::pipeline) {
            if (!fixed_rows) out.chunk = {out.model.chunk, "length-model"};
            out.route = {legal_prefix(prompt, out.chunk.rows), "pipeline-length-model"};
        } else {
            out.route = {0U, "single-length-model"};
        }
    } else {
        out.route = choose_prompt_route(prompt, out.chunk.rows, costs, prediction);
    }
    // Prefill-only has no single-node fallback or suffix handoff; it always runs the entire prompt.
    // Execution still validates full-prompt tails before launching the pipe.
    if (execution == PairExecution::prefill_only) out.route = {prompt, "pipeline-prefill-only"};
    out.tile_rows = out.route.prefix == 0U && single_tiles ? legacy : out.chunk.rows;
    if (out.route.prefix && out.tile_rows)
        out.chunks = (out.route.prefix + out.tile_rows - 1U) / out.tile_rows;
    return out;
}
} // namespace strix::qwen_pp
