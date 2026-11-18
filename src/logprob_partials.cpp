#include "strix/logprob_partials.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace strix {
namespace {
// value descending, lowest ID first among equals (the greedy order of every selection route)
[[nodiscard]] bool ahead(float va, std::uint32_t ia, float vb, std::uint32_t ib) noexcept {
    return va > vb || (va == vb && ia < ib);
}
} // namespace

LogprobPartial logprob_partial_host(std::span<const float> slice, std::uint32_t first, std::uint32_t top, LogprobTarget target) {
    if (top > kLogprobPartialTop) throw std::invalid_argument("logprob partial: top exceeds kLogprobPartialTop");
    LogprobPartial p;
    float maximum = -std::numeric_limits<float>::infinity();
    for (const float x : slice) {
        if (!std::isfinite(x)) { ++p.nonfinite; continue; }
        maximum = std::max(maximum, x);
    }
    p.max = maximum;
    // the device order: lane l sums entries l, l + L, l + 2L, ... in turn, then lanes fold pairwise (l += l + L/2, ...)
    float lanes[kLogprobPartialLanes]{};
    for (std::size_t i = 0; i < slice.size(); ++i)
        if (std::isfinite(slice[i])) lanes[i % kLogprobPartialLanes] += std::exp(slice[i] - maximum);
    for (std::uint32_t half = kLogprobPartialLanes / 2U; half > 0U; half /= 2U)
        for (std::uint32_t l = 0; l < half; ++l) lanes[l] += lanes[l + half];
    p.sum = lanes[0];
    const bool has_target = target.id != kLogprobNoTarget;
    if (has_target && target.id >= first && target.id - first < slice.size()) {
        p.flags |= LogprobPartial::kFound;
        target.logit = slice[target.id - first];
        target.logit_known = true;
        p.target = target.logit;
    }
    if (has_target && target.logit_known) {
        p.flags |= LogprobPartial::kCounted;
        for (std::size_t i = 0; i < slice.size(); ++i) {
            const auto id = first + static_cast<std::uint32_t>(i);
            if (!std::isfinite(slice[i])) continue;
            if (slice[i] > target.logit) ++p.greater;
            else if (slice[i] == target.logit && id < target.id) ++p.ties_lower;
        }
    }
    for (std::size_t i = 0; i < slice.size(); ++i) {  // insertion into the sorted top list (rare after the first entries)
        const float x = slice[i];
        if (!std::isfinite(x) || top == 0U) continue;
        const auto id = first + static_cast<std::uint32_t>(i);
        if (p.top_count == top && !ahead(x, id, p.top_value[top - 1U], p.top_id[top - 1U])) continue;
        std::uint32_t at = std::min(p.top_count, top - 1U);
        while (at > 0U && ahead(x, id, p.top_value[at - 1U], p.top_id[at - 1U])) {
            p.top_value[at] = p.top_value[at - 1U];
            p.top_id[at] = p.top_id[at - 1U];
            --at;
        }
        p.top_value[at] = x;
        p.top_id[at] = id;
        p.top_count = std::min(p.top_count + 1U, top);
    }
    return p;
}

LogprobRowScore combine_logprob_partials(std::span<const LogprobPartial> slices, std::uint32_t top, std::uint32_t target_id) {
    if (slices.empty()) throw std::invalid_argument("logprob partials: no slice");
    if (top > kLogprobPartialTop) throw std::invalid_argument("logprob partials: top exceeds kLogprobPartialTop");
    double maximum = -std::numeric_limits<double>::infinity();
    for (const auto& p : slices) {
        if (p.nonfinite != 0U) throw std::invalid_argument("logprob partials: nonfinite target logits");
        if (p.top_count > kLogprobPartialTop) throw std::invalid_argument("logprob partials: corrupt top count");
        if (p.sum > 0.0F) maximum = std::max(maximum, static_cast<double>(p.max));
    }
    if (!std::isfinite(maximum)) throw std::invalid_argument("logprob partials: empty row");
    double sum = 0.0;  // slice order: a fixed order per route (slices are given in vocabulary order)
    for (const auto& p : slices)
        if (p.sum > 0.0F) sum += static_cast<double>(p.sum) * std::exp(static_cast<double>(p.max) - maximum);
    LogprobRowScore out;
    out.maximum = maximum;
    out.centered_log_normalizer = std::log(sum);
    out.log_normalizer = maximum + out.centered_log_normalizer;
    // merged top: each slice list is sorted; take the best head `top` times
    std::vector<std::uint32_t> head(slices.size(), 0U);
    for (std::uint32_t k = 0; k < top; ++k) {
        std::size_t best = slices.size();
        for (std::size_t s = 0; s < slices.size(); ++s) {
            if (head[s] >= slices[s].top_count) continue;
            if (best == slices.size() || ahead(slices[s].top_value[head[s]], slices[s].top_id[head[s]],
                                               slices[best].top_value[head[best]], slices[best].top_id[head[best]]))
                best = s;
        }
        if (best == slices.size()) break;
        const auto& p = slices[best];
        out.top.push_back({p.top_id[head[best]], (static_cast<double>(p.top_value[head[best]]) - maximum) - out.centered_log_normalizer});
        ++head[best];
    }
    if (target_id != kLogprobNoTarget) {
        std::size_t found = 0, counted = 0;
        std::uint64_t rank = 1;
        float logit = 0.0F;
        for (const auto& p : slices) {
            if ((p.flags & LogprobPartial::kFound) != 0U) { ++found; logit = p.target; }
            if ((p.flags & LogprobPartial::kCounted) != 0U) { ++counted; rank += std::uint64_t{p.greater} + p.ties_lower; }
        }
        if (found != 1U) throw std::invalid_argument("logprob partials: target found in " + std::to_string(found) + " slices");
        out.has_target = true;
        out.target_logprob = (static_cast<double>(logit) - maximum) - out.centered_log_normalizer;
        if (counted == slices.size()) out.target_rank = rank;
        else
            for (std::size_t k = 0; k < out.top.size(); ++k)
                if (out.top[k].id == target_id) { out.target_rank = k + 1U; break; }
    }
    return out;
}

} // namespace strix
