#pragma once
#include "strix/serve/json.hpp"
#include "strix/pair/balance.hpp"
#include "strix/pair/pipe.hpp"
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
// 11.82-11.85 s against 11.61-11.69 s at 23, so 23 stays.
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
// Largest prefix accepted by a full tile or a >=16 row, multiple-of-eight tail.
inline std::uint64_t legal_prefix(std::uint64_t prompt, std::uint32_t rows) {
    if (rows < 16U || rows % 8U != 0U) throw std::invalid_argument("pair handoff invalid tile rows");
    auto prefix = prompt - prompt % 8U;
    if (prefix % rows == 8U) prefix -= 8U;
    return prefix;
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
} // namespace strix::qwen_pp
