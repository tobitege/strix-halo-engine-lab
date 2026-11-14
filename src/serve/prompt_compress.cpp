#include "strix/serve/prompt_compress.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>

namespace strix::serve::pflash {

namespace {

constexpr std::size_t npos = static_cast<std::size_t>(-1);

std::uint32_t saturate(std::uint64_t value) {
    return value > std::numeric_limits<std::uint32_t>::max() ? std::numeric_limits<std::uint32_t>::max()
                                                             : static_cast<std::uint32_t>(value);
}

double mass(const Unit& unit) {
    return unit.density > 0 ? static_cast<double>(unit.density) * unit.tokens : 0.0;
}

bool ordered(std::span<const Unit> units) {
    for (std::size_t i = 0; i < units.size(); ++i) {
        if (units[i].begin > units[i].end) return false;
        if (i > 0 && units[i].begin < units[i - 1].end) return false;
    }
    return true;
}

}  // namespace

float keep_ratio(const Policy& policy, std::uint32_t tokens) {
    const auto& curve = policy.curve;
    if (curve.empty()) return 1.0f;
    if (tokens <= curve.front().tokens) return curve.front().keep;
    for (std::size_t i = 1; i < curve.size(); ++i) {
        if (tokens > curve[i].tokens) continue;
        const CurvePoint& a = curve[i - 1];
        const CurvePoint& b = curve[i];
        if (b.tokens <= a.tokens) return b.keep;
        const double t = static_cast<double>(tokens - a.tokens) / static_cast<double>(b.tokens - a.tokens);
        return static_cast<float>(a.keep + t * (static_cast<double>(b.keep) - a.keep));
    }
    return curve.back().keep;
}

std::vector<Unit> segment(std::span<const Region> regions, const Scores& scores, const Policy& policy,
                          std::string_view content) {
    const auto offsets = scores.token_byte_begin;
    const std::size_t count = offsets.size();
    if (count == 0 || regions.empty()) return {};
    if (scores.token_score.size() != count || scores.cut_prob.size() != count ||
        scores.request_mask.size() != count) {
        return {};
    }
    for (std::size_t i = 1; i < count; ++i) {
        if (offsets[i] < offsets[i - 1]) return {};
    }
    if (!content.empty() && offsets[count - 1] > content.size()) return {};
    for (std::size_t i = 0; i < regions.size(); ++i) {
        if (regions[i].begin >= regions[i].end) return {};
        if (i > 0 && regions[i].begin < regions[i - 1].end) return {};
        if (!content.empty() && regions[i].end > content.size()) return {};
    }

    // Token range [first, last) of every region, and the token total of every message.
    struct Range { std::size_t first, last; };
    std::vector<Range> ranges(regions.size());
    std::map<std::uint32_t, std::uint64_t> message_tokens;
    for (std::size_t i = 0; i < regions.size(); ++i) {
        const auto first = std::lower_bound(offsets.begin(), offsets.end(), regions[i].begin);
        const auto last = std::lower_bound(first, offsets.end(), regions[i].end);
        ranges[i] = {static_cast<std::size_t>(first - offsets.begin()), static_cast<std::size_t>(last - offsets.begin())};
        message_tokens[regions[i].message] += ranges[i].last - ranges[i].first;
    }

    const std::size_t min_unit = std::max<std::size_t>(1, policy.min_unit_tokens);
    const std::size_t max_unit = std::max<std::size_t>(min_unit, policy.max_unit_tokens);
    const auto line_start = [&](std::size_t token) {
        if (content.empty()) return true;
        const std::uint32_t byte = offsets[token];
        return byte == 0 || content[byte - 1] == '\n';
    };

    std::vector<Unit> units;
    std::vector<std::uint32_t> request_tokens;  // per unit
    const auto emit = [&](std::uint32_t begin, std::uint32_t end, std::size_t first, std::size_t last, bool mandatory) {
        Unit unit;
        unit.begin = begin;
        unit.end = end;
        unit.tokens = saturate(last - first);
        unit.mandatory = mandatory;
        double sum = 0;
        std::uint32_t request = 0;
        for (std::size_t t = first; t < last; ++t) {
            sum += scores.token_score[t];
            request += scores.request_mask[t] != 0 ? 1u : 0u;
        }
        unit.density = last > first ? static_cast<float>(sum / static_cast<double>(last - first)) : 0.0f;
        units.push_back(unit);
        request_tokens.push_back(request);
    };

    std::vector<std::size_t> cuts;
    for (std::size_t i = 0; i < regions.size(); ++i) {
        const Region& region = regions[i];
        const std::size_t first = ranges[i].first;
        const std::size_t last = ranges[i].last;
        if (!region.droppable || message_tokens[region.message] <= policy.skeleton_tokens || last == first) {
            emit(region.begin, region.end, first, last, true);
            continue;
        }
        cuts.clear();
        std::size_t start = first;
        std::uint32_t start_byte = region.begin;
        while (true) {
            std::size_t cut = npos;
            const std::size_t reach = std::min(start + max_unit, last - 1);  // last token a cut may sit on
            for (std::size_t t = start + min_unit; t <= reach; ++t) {
                if (scores.cut_prob[t] >= policy.cut_threshold && offsets[t] > start_byte && line_start(t)) {
                    cut = t;
                    break;
                }
            }
            if (cut == npos && last - start > max_unit) {
                // No probable start in range: take the most probable line start, the latest on ties.
                for (std::size_t t = start + min_unit; t <= start + max_unit; ++t) {
                    if (offsets[t] <= start_byte || !line_start(t)) continue;
                    if (cut == npos || !(scores.cut_prob[t] < scores.cut_prob[cut])) cut = t;
                }
                if (cut == npos && offsets[start + max_unit] > start_byte) cut = start + max_unit;
            }
            if (cut == npos) break;
            cuts.push_back(cut);
            start = cut;
            start_byte = offsets[cut];
        }
        if (!cuts.empty() && last - cuts.back() < min_unit) cuts.pop_back();  // a short tail joins its neighbour
        std::size_t token = first;
        std::uint32_t byte = region.begin;
        for (const std::size_t cut : cuts) {
            emit(byte, offsets[cut], token, cut, false);
            token = cut;
            byte = offsets[cut];
        }
        emit(byte, region.end, token, last, false);
    }

    // Pin the units that carry the request, most request tokens first, while they fit.
    std::vector<std::size_t> carriers;
    for (std::size_t i = 0; i < units.size(); ++i) {
        if (!units[i].mandatory && request_tokens[i] > 0) carriers.push_back(i);
    }
    std::stable_sort(carriers.begin(), carriers.end(),
                     [&](std::size_t a, std::size_t b) { return request_tokens[a] > request_tokens[b]; });
    std::uint64_t pinned = 0;
    for (const std::size_t i : carriers) {
        if (pinned + units[i].tokens > policy.query_pin) continue;
        units[i].mandatory = true;
        pinned += units[i].tokens;
    }
    return units;
}

Plan bypass(std::uint32_t raw_tokens, Reason reason) {
    Plan plan;
    plan.raw_tokens = raw_tokens;
    plan.kept_tokens = raw_tokens;
    plan.reason = reason;
    return plan;
}

Plan select(std::span<const Unit> units, std::uint32_t droppable_tokens, const Policy& policy) {
    std::uint64_t raw = 0;
    for (const Unit& unit : units) raw += unit.tokens;
    const std::uint32_t raw_tokens = saturate(raw);
    if (units.empty() || !ordered(units)) return bypass(raw_tokens, Reason::scorer_failed);
    for (const Unit& unit : units) {
        if (!std::isfinite(unit.density)) return bypass(raw_tokens, Reason::scorer_failed);
    }
    if (raw_tokens < policy.threshold) return bypass(raw_tokens, Reason::below_threshold);

    std::vector<std::size_t> ranked;
    float low = std::numeric_limits<float>::infinity();
    float high = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < units.size(); ++i) {
        if (units[i].mandatory) continue;
        ranked.push_back(i);
        low = std::min(low, units[i].density);
        high = std::max(high, units[i].density);
    }
    if (ranked.size() < 2 || !(high - low > policy.min_spread)) return bypass(raw_tokens, Reason::flat_scores);

    const double ratio = std::clamp(static_cast<double>(keep_ratio(policy, raw_tokens)), 0.0, 1.0);
    const auto budget = static_cast<std::uint64_t>(ratio * static_cast<double>(std::min(droppable_tokens, raw_tokens)));
    std::stable_sort(ranked.begin(), ranked.end(),
                     [&](std::size_t a, std::size_t b) { return units[a].density > units[b].density; });
    std::vector<char> keep(units.size(), 0);
    for (std::size_t i = 0; i < units.size(); ++i) keep[i] = units[i].mandatory ? 1 : 0;
    std::uint64_t used = 0;
    for (const std::size_t i : ranked) {
        if (used + units[i].tokens > budget) break;
        keep[i] = 1;
        used += units[i].tokens;
    }

    Plan plan;
    plan.raw_tokens = raw_tokens;
    std::uint64_t kept_tokens = 0;
    for (std::size_t i = 0; i < units.size(); ++i) {
        if (keep[i]) {
            plan.kept.push_back(units[i]);
            kept_tokens += units[i].tokens;
        } else {
            plan.dropped.push_back(units[i]);
        }
    }
    if (plan.dropped.empty()) return bypass(raw_tokens, Reason::breakeven);
    plan.kept_tokens = saturate(kept_tokens);
    plan.reason = Reason::compressed;
    return plan;
}

std::string gap_marker(std::uint64_t tokens) {
    return "[... about " + std::to_string(tokens) + " tokens omitted ...]";
}

std::string assemble(std::string_view content, const Plan& plan) {
    if (plan.reason != Reason::compressed || plan.dropped.empty()) return std::string(content);
    if (!ordered(plan.dropped) || plan.dropped.back().end > content.size()) {
        throw std::invalid_argument("pflash assemble: dropped units do not fit the content");
    }
    std::string out;
    out.reserve(content.size());
    std::size_t position = 0;
    bool first = true;
    for (std::size_t i = 0; i < plan.dropped.size();) {
        const std::size_t begin = plan.dropped[i].begin;
        std::size_t end = plan.dropped[i].end;
        std::uint64_t tokens = plan.dropped[i].tokens;
        for (++i; i < plan.dropped.size() && plan.dropped[i].begin == end; ++i) {
            end = plan.dropped[i].end;
            tokens += plan.dropped[i].tokens;
        }
        if (end == begin) continue;
        out.append(content.substr(position, begin - position));
        if (!out.empty() && out.back() != '\n') out += '\n';
        if (first) {
            out += "[context compressed: " + std::to_string(plan.kept_tokens) + " of about " +
                   std::to_string(plan.raw_tokens) + " tokens kept; lines like the next one stand for omitted text]\n";
            first = false;
        }
        out += gap_marker(tokens);
        out += '\n';
        position = end;
    }
    out.append(content.substr(position));
    return out;
}

Plan recall(const Plan& held, std::span<const Unit> all, const Policy& policy) {
    Plan plan = bypass(held.raw_tokens, held.reason);
    plan.kept_tokens = held.kept_tokens;
    if (held.reason != Reason::compressed || held.dropped.empty()) return plan;
    plan.dropped = held.dropped;

    double total = 0;
    for (const Unit& unit : all) {
        if (!std::isfinite(unit.density)) return plan;
        total += mass(unit);
    }
    if (!(total > 0)) return plan;

    // The new score of every unit still omitted, matched by byte range.
    std::vector<const Unit*> sorted;
    sorted.reserve(all.size());
    for (const Unit& unit : all) sorted.push_back(&unit);
    std::stable_sort(sorted.begin(), sorted.end(), [](const Unit* a, const Unit* b) { return a->begin < b->begin; });
    struct Omitted { std::size_t index; double mass; float density; };
    std::vector<Omitted> omitted;
    double omitted_mass = 0;
    for (std::size_t i = 0; i < held.dropped.size(); ++i) {
        const Unit& unit = held.dropped[i];
        auto it = std::lower_bound(sorted.begin(), sorted.end(), unit.begin,
                                   [](const Unit* a, std::uint32_t begin) { return a->begin < begin; });
        for (; it != sorted.end() && (*it)->begin == unit.begin; ++it) {
            if ((*it)->end != unit.end) continue;
            omitted.push_back({i, mass(**it), (*it)->density});
            omitted_mass += omitted.back().mass;
            break;
        }
    }
    if (omitted_mass / total < policy.recall_gate) return plan;

    std::stable_sort(omitted.begin(), omitted.end(), [](const Omitted& a, const Omitted& b) { return a.mass > b.mass; });
    std::vector<char> recalled(held.dropped.size(), 0);
    std::vector<float> density(held.dropped.size(), 0.0f);
    double remaining = omitted_mass;
    for (const Omitted& item : omitted) {
        if (remaining / total < policy.recall_gate || !(item.mass > 0)) break;
        recalled[item.index] = 1;
        density[item.index] = item.density;
        remaining -= item.mass;
    }
    plan.dropped.clear();
    std::uint64_t kept_tokens = held.kept_tokens;
    for (std::size_t i = 0; i < held.dropped.size(); ++i) {
        if (!recalled[i]) {
            plan.dropped.push_back(held.dropped[i]);
            continue;
        }
        Unit unit = held.dropped[i];
        unit.density = density[i];
        plan.kept.push_back(unit);
        kept_tokens += unit.tokens;
    }
    plan.kept_tokens = saturate(kept_tokens);
    return plan;
}

std::string recalled_text(std::string_view content, const Plan& recalled) {
    if (recalled.kept.empty()) return {};
    std::uint64_t tokens = 0;
    for (const Unit& unit : recalled.kept) {
        if (unit.begin > unit.end || unit.end > content.size()) {
            throw std::invalid_argument("pflash recalled_text: a unit does not fit the content");
        }
        tokens += unit.tokens;
    }
    std::string out = "[recalled context: about " + std::to_string(tokens) + " tokens omitted earlier]\n";
    for (const Unit& unit : recalled.kept) {
        out.append(content.substr(unit.begin, unit.end - unit.begin));
        if (out.back() != '\n') out += '\n';
    }
    return out;
}

bool worth_it(std::uint32_t raw, std::uint32_t kept, const CostModel& cost, const Policy& policy) {
    if (!std::isfinite(cost.target_tok_s) || !(cost.target_tok_s > 0) || kept >= raw) return false;
    const double size = raw;
    const double scorer = cost.drafter_a_s_per_tok * size + cost.drafter_b_s_per_tok2 * size * size;
    if (!std::isfinite(scorer) || scorer < 0) return false;
    const double saving = (size - static_cast<double>(kept)) / cost.target_tok_s - scorer;
    return saving >= policy.min_saving_s;
}

const char* reason_name(Reason reason) {
    switch (reason) {
        case Reason::compressed: return "compressed";
        case Reason::below_threshold: return "below_threshold";
        case Reason::flat_scores: return "flat_scores";
        case Reason::scorer_failed: return "scorer_failed";
        case Reason::breakeven: return "breakeven";
        case Reason::tools: return "tools";
        case Reason::off: return "off";
    }
    return "unknown";
}

std::string format_receipt(const Plan& plan) {
    return "pflash raw=" + std::to_string(plan.raw_tokens) + " kept=" + std::to_string(plan.kept_tokens) +
           " reason=" + reason_name(plan.reason);
}

}  // namespace strix::serve::pflash
