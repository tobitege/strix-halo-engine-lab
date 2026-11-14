#pragma once
// Stage balance of a two-stage layer pipeline, shared by every engine's pair route: stage A runs layers
// [0, cut), stage B runs [cut, layers) plus its tail (head, final norm). The best cut minimises the slower
// stage; ties keep the smallest cut. Costs are per-row (or per-chunk) times measured by the engine.
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace strix::pair {
struct StageSplit { std::uint32_t cut = 0; double a = 0, b = 0; bool half = false; };
inline StageSplit predict_split(std::span<const double> layers, double tail, std::uint32_t cut) {
    StageSplit split{cut, 0, tail};
    for (std::size_t layer = 0; layer < layers.size(); ++layer) (layer < cut ? split.a : split.b) += layers[layer];
    return split;
}
template<class Legal> StageSplit balance_split(std::span<const double> layers, double tail, Legal legal) {
    StageSplit best{};
    double score = std::numeric_limits<double>::infinity();
    for (std::uint32_t cut = 1; cut < layers.size(); ++cut) if (legal(cut)) {
        const auto candidate = predict_split(layers, tail, cut);
        const auto work = candidate.a > candidate.b ? candidate.a : candidate.b;
        if (work < score) { best = candidate; score = work; }
    }
    if (!(score < std::numeric_limits<double>::infinity())) throw std::invalid_argument("pair pipeline has no legal cut");
    return best;
}
// Half-layer cuts: `front[layer]` is the cost of the first sub-block (attention) of a layer, part of layers[layer].
// A half cut at `cut` gives stage A layers [0, cut) plus front[cut].
inline StageSplit predict_split(std::span<const double> layers, std::span<const double> front, double tail,
    std::uint32_t cut, bool half) {
    auto split = predict_split(layers, tail, cut);
    if (!half) return split;
    if (cut >= layers.size() || front.size() != layers.size() || !(front[cut] <= layers[cut]))
        throw std::invalid_argument("pair pipeline half-layer cut is outside the layer costs");
    split.a += front[cut]; split.b -= front[cut]; split.half = true;
    return split;
}
template<class Legal, class LegalHalf> StageSplit balance_split(std::span<const double> layers,
    std::span<const double> front, double tail, Legal legal, LegalHalf legal_half) {
    StageSplit best{};
    double score = std::numeric_limits<double>::infinity();
    const auto offer = [&](const StageSplit& candidate) {
        const auto work = candidate.a > candidate.b ? candidate.a : candidate.b;
        if (work < score) { best = candidate; score = work; }
    };
    for (std::uint32_t cut = 1; cut < layers.size(); ++cut) {
        if (legal(cut)) offer(predict_split(layers, tail, cut));
        if (legal_half(cut)) offer(predict_split(layers, front, tail, cut, true));
    }
    if (!(score < std::numeric_limits<double>::infinity())) throw std::invalid_argument("pair pipeline has no legal cut");
    return best;
}
} // namespace strix::pair
