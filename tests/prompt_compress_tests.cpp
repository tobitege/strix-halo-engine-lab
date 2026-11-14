// Host-only tests for the prompt compression selection layer.
#include "strix/serve/prompt_compress.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _MSC_VER
#include <crtdbg.h>
#include <cstdlib>
#endif

using namespace strix::serve::pflash;

namespace {
int failures = 0;
void check(bool value, const char* label) {
    if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
bool near(double a, double b) { return std::fabs(a - b) < 1e-6; }

// A stand-in tokenizer: one token per word, the following whitespace included.
std::vector<std::string> words(std::string_view text) {
    std::vector<std::string> out;
    std::size_t begin = 0;
    while (begin < text.size()) {
        std::size_t end = begin;
        while (end < text.size() && text[end] != ' ' && text[end] != '\n') ++end;
        while (end < text.size() && (text[end] == ' ' || text[end] == '\n')) ++end;
        out.emplace_back(text.substr(begin, end - begin));
        begin = end;
    }
    return out;
}

// A rendered prompt with its regions and per-token scorer output.
struct Prompt {
    std::string text;
    std::vector<Region> regions;
    std::vector<float> score, cut;
    std::vector<std::uint8_t> request;
    std::vector<std::uint32_t> offsets;

    Scores scores() const { return {score, cut, request, offsets}; }

    void envelope(std::string_view tag) { text += tag; }

    // Appends `lines` lines of `words_per_line` words as one region. Every line start gets `cut_prob`.
    void message(std::uint32_t index, bool droppable, int lines, int words_per_line, float token_score,
                 float cut_prob = 0.9f, bool carries_request = false) {
        const auto begin = static_cast<std::uint32_t>(text.size());
        for (int line = 0; line < lines; ++line) {
            for (int word = 0; word < words_per_line; ++word) {
                offsets.push_back(static_cast<std::uint32_t>(text.size()));
                score.push_back(token_score);
                cut.push_back(word == 0 ? cut_prob : 0.0f);
                request.push_back(carries_request ? 1 : 0);
                text += "w" + std::to_string(index) + "_" + std::to_string(line);
                text += word + 1 == words_per_line ? '\n' : ' ';
            }
        }
        regions.push_back({index, begin, static_cast<std::uint32_t>(text.size()), droppable});
    }
};

std::uint32_t droppable_tokens(const Prompt& prompt, std::span<const Unit> units) {
    std::uint32_t total = 0;
    for (const Unit& unit : units) {
        for (const Region& region : prompt.regions) {
            if (region.droppable && unit.begin >= region.begin && unit.end <= region.end) total += unit.tokens;
        }
    }
    return total;
}

bool same(const Plan& a, const Plan& b) {
    if (a.reason != b.reason || a.raw_tokens != b.raw_tokens || a.kept_tokens != b.kept_tokens) return false;
    if (a.kept.size() != b.kept.size() || a.dropped.size() != b.dropped.size()) return false;
    for (std::size_t i = 0; i < a.kept.size(); ++i) {
        if (a.kept[i].begin != b.kept[i].begin || a.kept[i].end != b.kept[i].end) return false;
    }
    return true;
}

void curve_tests() {
    const Policy policy;
    check(near(keep_ratio(policy, 1000), 0.315), "curve: flat below the first point");
    check(near(keep_ratio(policy, 16384), 0.315), "curve: first point");
    check(near(keep_ratio(policy, 24576), 0.2925), "curve: linear between 16K and 32K");
    check(near(keep_ratio(policy, 32768), 0.27), "curve: 32K");
    check(near(keep_ratio(policy, 50000), 0.27), "curve: flat between 32K and 64K");
    check(near(keep_ratio(policy, 92768), 0.2475), "curve: linear between 64K and 120K");
    check(near(keep_ratio(policy, 120000), 0.225), "curve: last point");
    check(near(keep_ratio(policy, 500000), 0.225), "curve: flat above the last point");
    Policy none;
    none.curve.clear();
    check(near(keep_ratio(none, 40000), 1.0), "curve: empty curve keeps everything");
}

void segment_tests() {
    Policy policy;
    policy.min_unit_tokens = 8;
    policy.max_unit_tokens = 40;
    policy.skeleton_tokens = 16;
    policy.query_pin = 30;

    Prompt prompt;
    prompt.envelope("<sys>");
    prompt.message(0, false, 10, 6, 0.1f);            // system text: mandatory
    prompt.envelope("<user>");
    prompt.message(1, true, 60, 4, 0.2f);             // long turn: cut every 8 tokens (two lines)
    prompt.envelope("<user>");
    prompt.message(2, true, 3, 4, 0.9f);              // 12 tokens: short turn, kept whole
    prompt.envelope("<user>");
    prompt.message(3, true, 30, 4, 0.3f, 0.0f);       // no probable start: forced cuts at max_unit
    prompt.envelope("<user>");
    prompt.message(4, true, 40, 1, 0.3f);             // one-word lines, then a 3-token tail
    prompt.message(4, true, 5, 4, 0.4f, 0.9f, true);  // same turn, carries the request

    const auto units = segment(prompt.regions, prompt.scores(), policy, prompt.text);
    check(!units.empty(), "segment: units produced");

    // Units tile the regions exactly, in order, and count every token once.
    std::size_t region = 0;
    std::uint32_t at = prompt.regions.front().begin;
    std::uint64_t tokens = 0;
    bool tiled = true, line_starts = true, sized = true;
    for (const Unit& unit : units) {
        if (unit.begin != at) tiled = false;
        if (unit.begin != 0 && unit.begin != prompt.regions[region].begin && prompt.text[unit.begin - 1] != '\n') {
            line_starts = false;
        }
        if (!unit.mandatory && (unit.tokens < policy.min_unit_tokens || unit.tokens >= 2 * policy.max_unit_tokens)) {
            sized = false;
        }
        tokens += unit.tokens;
        at = unit.end;
        if (at == prompt.regions[region].end && region + 1 < prompt.regions.size()) at = prompt.regions[++region].begin;
    }
    check(tiled && at == prompt.regions.back().end, "segment: units tile the regions");
    check(tokens == prompt.offsets.size(), "segment: every token counted once");
    check(line_starts, "segment: cuts fall on line starts");
    check(sized, "segment: unit sizes within min / max");

    const auto in_region = [&](std::size_t index) {
        std::vector<Unit> out;
        for (const Unit& unit : units) {
            if (unit.begin >= prompt.regions[index].begin && unit.end <= prompt.regions[index].end) out.push_back(unit);
        }
        return out;
    };
    const auto system = in_region(0);
    check(system.size() == 1 && system[0].mandatory && system[0].tokens == 60, "segment: non-droppable region is one mandatory unit");
    const auto turn = in_region(1);
    check(turn.size() == 30 && turn[0].tokens == 8 && !turn[0].mandatory, "segment: cut at the first start past min_unit");
    check(near(turn[0].density, 0.2), "segment: density is the mean token score");
    const auto brief = in_region(2);
    check(brief.size() == 1 && brief[0].mandatory, "segment: short turn kept whole");
    const auto forced = in_region(3);
    check(forced.size() == 3 && forced[0].tokens == 40 && forced[1].tokens == 40 && forced[2].tokens == 40,
          "segment: forced cut at max_unit on a line start");
    const auto lines = in_region(4);
    check(lines.size() == 5 && lines[0].tokens == 8 && lines[4].tokens == 8, "segment: tail shorter than min_unit joins");

    // Request carriers: 20 tokens in units of 8, 8 (short tail joined: 12); the pin holds 30.
    const auto carriers = in_region(5);
    std::uint32_t pinned = 0;
    for (const Unit& unit : carriers) pinned += unit.mandatory ? unit.tokens : 0;
    check(carriers.size() == 2 && pinned == 20, "segment: request carriers pinned within query_pin");
    policy.query_pin = 10;
    const auto capped = segment(prompt.regions, prompt.scores(), policy, prompt.text);
    pinned = 0;
    for (const Unit& unit : capped) {
        if (unit.begin >= prompt.regions[5].begin) pinned += unit.mandatory ? unit.tokens : 0;
    }
    check(pinned == 8, "segment: pin stops at query_pin");

    // Without the content the cut probabilities are trusted as they are.
    const auto blind = segment(prompt.regions, prompt.scores(), policy);
    check(blind.size() == units.size(), "segment: same cuts without content when cut_prob sits on line starts");

    // Inconsistent inputs give no units.
    Prompt bad = prompt;
    bad.score.pop_back();
    check(segment(bad.regions, bad.scores(), policy, bad.text).empty(), "segment: span length mismatch");
    bad = prompt;
    bad.offsets[5] = bad.offsets[3];
    check(segment(bad.regions, bad.scores(), policy, bad.text).empty(), "segment: offsets going back");
    bad = prompt;
    bad.regions[1].begin = bad.regions[0].begin;
    check(segment(bad.regions, bad.scores(), policy, bad.text).empty(), "segment: overlapping regions");
    bad = prompt;
    bad.regions.back().end += 10;
    check(segment(bad.regions, bad.scores(), policy, bad.text).empty(), "segment: region past the content");
    check(segment({}, prompt.scores(), policy, prompt.text).empty(), "segment: no regions");
}

// 40 units of 500 tokens after a mandatory head of 600; density grows with a fixed permutation.
std::vector<Unit> ranked_units() {
    std::vector<Unit> units;
    units.push_back({0, 600, 600, 0.0f, true});
    for (std::uint32_t i = 0; i < 40; ++i) {
        const std::uint32_t begin = 600 + i * 500;
        units.push_back({begin, begin + 500, 500, static_cast<float>((i * 7) % 40) / 40.0f, false});
    }
    return units;
}

void select_tests() {
    const Policy policy;
    auto units = ranked_units();
    const Plan plan = select(units, 20000, policy);
    // raw 20,600: ratio 0.315 - 0.045 * 4216 / 16384 = 0.30342; budget 6,068 tokens = 12 units.
    check(plan.reason == Reason::compressed && plan.raw_tokens == 20600, "select: compressed");
    check(plan.kept.size() == 13 && plan.kept_tokens == 6600, "select: budget from the curve, mandatory on top");
    check(plan.dropped.size() == 28, "select: the rest is dropped");
    check(plan.kept.front().mandatory, "select: mandatory unit kept");
    bool order = true, dense = true;
    for (std::size_t i = 1; i < plan.kept.size(); ++i) order = order && plan.kept[i - 1].begin < plan.kept[i].begin;
    for (const Unit& unit : plan.kept) dense = dense && (unit.mandatory || unit.density >= 28.0f / 40.0f);
    for (const Unit& unit : plan.dropped) dense = dense && unit.density < 28.0f / 40.0f;
    check(order, "select: kept units in original order");
    check(dense, "select: the densest units are kept");
    check(same(plan, select(units, 20000, policy)), "select: deterministic");
    check(format_receipt(plan) == "pflash raw=20600 kept=6600 reason=compressed", "receipt: compressed");

    // Stop at the first unit that does not fit: a big dense unit blocks the smaller ones behind it.
    auto blocked = units;
    blocked[3].tokens = 7200;
    blocked[3].end = blocked[3].begin + 500;
    blocked[3].density = 0.99f;
    blocked[5].density = 0.999f;
    const Plan stop = select(blocked, 26700, policy);
    check(stop.reason == Reason::compressed && stop.kept.size() == 2, "select: stops at the first unit that does not fit");

    // A mandatory unit larger than the whole budget is still kept.
    auto heavy = units;
    heavy[0].tokens = 30000;
    const Plan big = select(heavy, 20000, policy);
    check(big.reason == Reason::compressed && big.kept.front().tokens == 30000, "select: mandatory beyond the budget");

    // Fallbacks.
    Policy high = policy;
    high.threshold = 20601;
    const Plan below = select(units, 20000, high);
    check(below.reason == Reason::below_threshold && below.kept.empty() && below.dropped.empty() &&
              below.kept_tokens == below.raw_tokens,
          "select: below threshold");
    auto flat = units;
    for (Unit& unit : flat) unit.density = 0.5f;
    check(select(flat, 20000, policy).reason == Reason::flat_scores, "select: flat scores");
    Policy spread = policy;
    spread.min_spread = 2.0f;
    check(select(units, 20000, spread).reason == Reason::flat_scores, "select: spread under min_spread");
    auto all_mandatory = units;
    for (Unit& unit : all_mandatory) unit.mandatory = true;
    check(select(all_mandatory, 0, policy).reason == Reason::flat_scores, "select: nothing to rank");
    auto broken = units;
    broken[7].density = std::numeric_limits<float>::quiet_NaN();
    check(select(broken, 20000, policy).reason == Reason::scorer_failed, "select: non-finite density");
    broken = units;
    broken[4].begin = broken[3].begin;
    check(select(broken, 20000, policy).reason == Reason::scorer_failed, "select: overlapping units");
    check(select({}, 0, policy).reason == Reason::scorer_failed, "select: no units");
    Policy everything = policy;
    everything.curve = {{0, 1.0f}};
    check(select(units, 20000, everything).reason == Reason::breakeven, "select: budget keeps everything");
    check(format_receipt(below) == "pflash raw=20600 kept=20600 reason=below_threshold", "receipt: fallback");
}

void assemble_tests() {
    Policy policy;
    policy.threshold = 100;
    policy.min_unit_tokens = 8;
    policy.max_unit_tokens = 40;
    policy.skeleton_tokens = 16;
    policy.curve = {{0, 0.25f}};

    Prompt prompt;
    prompt.envelope("<sys>");
    prompt.message(0, false, 4, 5, 0.0f);
    prompt.envelope("<end>\n<user>");
    prompt.message(1, true, 40, 4, 0.1f);
    prompt.envelope("<end>\n<user>");
    prompt.message(2, false, 2, 5, 0.0f);
    prompt.envelope("<end>\n<assistant>");
    // Lines 10-11 and 30-31 of the long turn are the relevant ones; 12-13 a little.
    for (std::size_t token = 0; token < prompt.offsets.size(); ++token) {
        const std::string_view rest = std::string_view(prompt.text).substr(prompt.offsets[token]);
        if (rest.starts_with("w1_10 ") || rest.starts_with("w1_11 ") || rest.starts_with("w1_30 ") || rest.starts_with("w1_31 ")) {
            prompt.score[token] = 0.9f;
        }
        if (rest.starts_with("w1_12 ") || rest.starts_with("w1_13 ")) prompt.score[token] = 0.5f;
    }
    // The scorer's last tokens of lines 10-13 end in "\n", not " ": score the whole lines.
    for (std::size_t token = 0; token + 1 < prompt.offsets.size(); ++token) {
        if (prompt.score[token] > 0.1f && prompt.score[token + 1] <= 0.1f && prompt.cut[token + 1] == 0.0f) {
            prompt.score[token + 1] = prompt.score[token];
        }
    }

    const auto units = segment(prompt.regions, prompt.scores(), policy, prompt.text);
    const Plan plan = select(units, droppable_tokens(prompt, units), policy);
    // 160 droppable tokens at 0.25 = 40: the two 0.9 units and the 0.5 unit (8 tokens each) fit, then 0.1 units.
    check(plan.reason == Reason::compressed, "assemble: plan compressed");
    const std::string out = assemble(prompt.text, plan);

    check(out.size() < prompt.text.size(), "assemble: shorter than the input");
    check(out.starts_with(prompt.text.substr(0, prompt.regions[1].begin)), "assemble: head kept byte for byte");
    const std::string tail = prompt.text.substr(prompt.regions[1].end);
    check(out.ends_with(tail), "assemble: query turn and generation prompt kept byte for byte");
    const auto count = [&](std::string_view needle) {
        std::size_t hits = 0;
        for (std::size_t at = out.find(needle); at != std::string::npos; at = out.find(needle, at + 1)) ++hits;
        return hits;
    };
    check(count("[context compressed: ") == 1, "assemble: one note");
    check(out.find("[context compressed: ") < out.find("[... about "), "assemble: note before the first marker");
    check(count("w1_10 ") == 3 && count("w1_30 ") == 3 && count("w1_12 ") == 3, "assemble: relevant lines kept");
    check(out.find("w1_10 ") < out.find("w1_12 ") && out.find("w1_12 ") < out.find("w1_30 "), "assemble: original order");

    // Markers: one per run of adjacent dropped units, and their counts add up to the dropped tokens.
    std::size_t runs = 0;
    std::uint64_t dropped_tokens = 0;
    for (std::size_t i = 0; i < plan.dropped.size(); ++i) {
        if (i == 0 || plan.dropped[i].begin != plan.dropped[i - 1].end) ++runs;
        dropped_tokens += plan.dropped[i].tokens;
    }
    std::uint64_t marked = 0;
    for (std::size_t at = out.find("[... about "); at != std::string::npos; at = out.find("[... about ", at + 1)) {
        marked += std::stoull(out.substr(at + 11));
        check(at == 0 || out[at - 1] == '\n', "assemble: marker starts a line");
    }
    check(count("[... about ") == runs && runs >= 2, "assemble: one marker per gap");
    check(marked == dropped_tokens && dropped_tokens == plan.raw_tokens - plan.kept_tokens, "assemble: markers carry the omitted count");
    check(gap_marker(8) == "[... about 8 tokens omitted ...]", "assemble: marker text");
    check(assemble(prompt.text, plan) == out, "assemble: deterministic");

    // Fallback identity: every plan that is not `compressed` gives today's token vector.
    const auto reference = words(prompt.text);
    for (const Reason reason : {Reason::below_threshold, Reason::flat_scores, Reason::scorer_failed, Reason::breakeven,
                                Reason::tools, Reason::off}) {
        Plan fallback = bypass(plan.raw_tokens, reason);
        fallback.dropped = plan.dropped;  // even a stale unit list must not change the text
        check(assemble(prompt.text, fallback) == prompt.text, reason_name(reason));
        check(words(assemble(prompt.text, fallback)) == reference, "fallback: token vector identical");
    }
    Policy high = policy;
    high.threshold = 100000;
    check(words(assemble(prompt.text, select(units, 160, high))) == reference, "fallback: below threshold, tokens identical");
    auto flat = units;
    for (Unit& unit : flat) unit.density = 0.3f;
    check(words(assemble(prompt.text, select(flat, 160, policy))) == reference, "fallback: flat scores, tokens identical");
    check(words(assemble(prompt.text, select({}, 0, policy))) == reference, "fallback: scorer failed, tokens identical");
    check(words(out) != reference, "compressed: token vector differs");

    // A plan that does not fit the content is refused.
    Plan wrong = plan;
    wrong.dropped.back().end = static_cast<std::uint32_t>(prompt.text.size() + 1);
    bool thrown = false;
    try { (void)assemble(prompt.text, wrong); } catch (const std::invalid_argument&) { thrown = true; }
    check(thrown, "assemble: dropped unit past the content throws");
}

void recall_tests() {
    const Policy policy;
    const auto units = ranked_units();
    const Plan held = select(units, 20000, policy);

    // The same scores again: the omitted units hold far more than 7.5 % of the mass, so the heaviest return.
    const Plan again = recall(held, units, policy);
    double total = 0, left = 0, least_recalled = 1e9, most_left = 0;
    for (const Unit& unit : units) total += static_cast<double>(unit.density) * unit.tokens;
    for (const Unit& unit : again.dropped) {
        left += static_cast<double>(unit.density) * unit.tokens;
        most_left = std::max(most_left, static_cast<double>(unit.density));
    }
    for (const Unit& unit : again.kept) least_recalled = std::min(least_recalled, static_cast<double>(unit.density));
    check(!again.kept.empty() && again.kept.size() + again.dropped.size() == held.dropped.size(), "recall: units recalled");
    check(left / total < policy.recall_gate, "recall: the rest is under the gate");
    check(least_recalled >= most_left, "recall: heaviest units first");
    check(again.kept_tokens == held.kept_tokens + 500 * again.kept.size() && again.raw_tokens == held.raw_tokens,
          "recall: token counts");
    bool order = true;
    for (std::size_t i = 1; i < again.kept.size(); ++i) order = order && again.kept[i - 1].begin < again.kept[i].begin;
    check(order, "recall: original order");

    // A follow-up about the kept units: nothing comes back.
    auto focused = units;
    for (Unit& unit : focused) unit.density = 0.001f;
    for (const Unit& kept : held.kept) {
        for (Unit& unit : focused) {
            if (unit.begin == kept.begin) unit.density = 1.0f;
        }
    }
    const Plan none = recall(held, focused, policy);
    check(none.kept.empty() && none.dropped.size() == held.dropped.size() && none.kept_tokens == held.kept_tokens,
          "recall: under the gate nothing is recalled");
    check(recalled_text("abc", none).empty(), "recall: no text when nothing is recalled");

    // One omitted unit becomes the answer: only that one is recalled, with its new density.
    auto shifted = focused;
    const Unit target = held.dropped[5];
    for (Unit& unit : shifted) {
        if (unit.begin == target.begin) unit.density = 2.0f;
    }
    const Plan one = recall(held, shifted, policy);
    check(one.kept.size() == 1 && one.kept[0].begin == target.begin && near(one.kept[0].density, 2.0),
          "recall: the unit that matters now");
    check(one.dropped.size() == held.dropped.size() - 1, "recall: the others stay omitted");

    std::string content(20600, 'x');
    content[target.begin] = 'A';
    content[target.end - 1] = '\n';
    const std::string text = recalled_text(content, one);
    check(text.starts_with("[recalled context: about 500 tokens omitted earlier]\nAxx") && text.size() == 53 + 500,
          "recall: text of the recalled unit");

    // Plans that were not compressed, or scores that are not usable, recall nothing.
    check(recall(bypass(900, Reason::below_threshold), units, policy).kept.empty(), "recall: fallback plan");
    auto broken = units;
    broken[2].density = std::numeric_limits<float>::infinity();
    check(recall(held, broken, policy).kept.empty(), "recall: non-finite scores");
    check(recall(held, {}, policy).kept.empty(), "recall: no scores");
}

void worth_it_tests() {
    const Policy policy;
    // 32K prompt, 9.4K kept, target at 2,812 tok/s: 8.3 s saved before the scorer.
    const CostModel three{2812.0, 3.0 / 32768.0, 0.0};
    const CostModel nine{2812.0, 9.0 / 32768.0, 0.0};
    check(worth_it(32768, 9400, three, policy), "worth_it: 3 s scorer pays at 32K");
    check(!worth_it(32768, 9400, nine, policy), "worth_it: 9 s scorer does not");
    // Quadratic scorer cost: fine at 32K, a loss at 128K on a fast target.
    const CostModel quadratic{2922.0, 4.0e-5, 1.8e-9};
    check(worth_it(32768, 9400, quadratic, policy), "worth_it: quadratic model at 32K");
    check(!worth_it(131072, 30100, quadratic, policy), "worth_it: quadratic model at 128K");
    const CostModel slow{623.0, 4.0e-5, 1.8e-9};
    check(worth_it(131072, 30100, slow, policy), "worth_it: slow target still gains at 128K");
    // The margin: a saving under min_saving_s is not taken.
    const CostModel free_scorer{1000.0, 0.0, 0.0};
    check(!worth_it(1000, 600, free_scorer, policy), "worth_it: 0.4 s saved is under the margin");
    check(worth_it(1000, 500, free_scorer, policy), "worth_it: 0.5 s saved meets the margin");
    check(!worth_it(1000, 1000, free_scorer, policy) && !worth_it(1000, 1200, free_scorer, policy), "worth_it: nothing dropped");
    check(!worth_it(32768, 9400, CostModel{}, policy), "worth_it: no rate");
    check(!worth_it(32768, 9400, {std::numeric_limits<double>::quiet_NaN(), 0, 0}, policy), "worth_it: NaN rate");
    check(!worth_it(32768, 9400, {2812.0, std::numeric_limits<double>::infinity(), 0}, policy), "worth_it: non-finite scorer cost");
}

void reason_tests() {
    check(std::string(reason_name(Reason::tools)) == "tools" && std::string(reason_name(Reason::off)) == "off" &&
              std::string(reason_name(Reason::breakeven)) == "breakeven" &&
              std::string(reason_name(Reason::scorer_failed)) == "scorer_failed",
          "reason names");
    const Plan off = bypass(40000, Reason::off);
    check(format_receipt(off) == "pflash raw=40000 kept=40000 reason=off", "receipt: off");
    check(Plan{}.reason == Reason::off, "plan: default is off");
}
}  // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif
    try {
        curve_tests();
        segment_tests();
        select_tests();
        assemble_tests();
        recall_tests();
        worth_it_tests();
        reason_tests();
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "FAIL: exception: " << error.what() << '\n';
    }
    if (failures != 0) {
        std::cerr << failures << " prompt compression test(s) failed\n";
        return 1;
    }
    std::cout << "prompt compression tests passed\n";
    return 0;
}
