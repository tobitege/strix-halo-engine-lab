#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// Prompt compression before prefill: the host selection layer.
//
// A small scorer reads the whole rendered prompt and marks which parts matter for the request; the
// target model then prefills only the kept parts, with one marker line per omitted gap. This layer
// is the pure host half: it cuts the prompt into units, picks the units that fit a length-dependent
// budget, rebuilds the text, and decides whether the saving pays for the scorer.
//
// Exactness class: LOSSY. The target sees a different prompt, so answers can change. Nothing here
// runs unless a caller asks for it per request; a Plan whose reason is not Reason::compressed makes
// assemble() return the input bytes unchanged, so every fallback tokenises to today's prompt.
//
// All offsets are bytes of the rendered prompt, not target tokens: scorer and target tokenizers may
// differ. Token counts are scorer tokens and serve as the length estimate for threshold and budget.
// Every function is deterministic: the same inputs give the same plan.
namespace strix::serve::pflash {

// A byte range of the rendered prompt that holds message content. Bytes outside every region (role
// tags, turn envelopes, the generation prompt) are always kept. Regions are sorted and disjoint.
struct Region {
    std::uint32_t message = 0;  // message index; a message of at most Policy::skeleton_tokens is kept whole
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
    bool droppable = false;     // false: system / developer text, tool definitions, the query turn
};

struct Unit {
    std::uint32_t begin = 0;    // bytes
    std::uint32_t end = 0;
    std::uint32_t tokens = 0;   // scorer tokens that start inside the unit
    float density = 0;          // mean token score
    bool mandatory = false;     // kept outside the budget
};

struct CurvePoint {
    std::uint32_t tokens = 0;
    float keep = 1;
};

struct Policy {
    std::uint32_t threshold = 16384;      // prompts shorter than this are not compressed
    std::uint32_t skeleton_tokens = 256;  // messages up to this length are kept whole
    std::uint32_t query_pin = 512;        // token budget for units pinned because they carry the request
    // Keep ratio of the droppable tokens by prompt length; linear between points, flat outside.
    std::vector<CurvePoint> curve{{16384, .315f}, {32768, .27f}, {65536, .27f}, {120000, .225f}};
    float recall_gate = 0.075f;           // follow-up: re-insert when omitted units hold this share of the score
    float min_spread = 0;                 // density spread at or below this counts as flat relevance
    double min_saving_s = 0.5;            // worth_it(): seconds the compressed path must save
    std::uint32_t min_unit_tokens = 32;   // a cut is taken only this far from the unit start
    std::uint32_t max_unit_tokens = 1024; // a unit this long is cut at its best line start
    float cut_threshold = 0.5f;           // cut_prob at or above this starts a unit
};

// Scorer output, one entry per scorer token. The spans must have equal length.
struct Scores {
    std::span<const float> token_score;                 // relevance to the request, >= 0
    std::span<const float> cut_prob;                    // probability that a unit starts at this token
    std::span<const std::uint8_t> request_mask;         // non-zero: the token carries the request
    std::span<const std::uint32_t> token_byte_begin;    // first byte of the token, non-decreasing
};

enum class Reason : std::uint8_t { compressed, below_threshold, flat_scores, scorer_failed, breakeven, tools, off };

struct Plan {
    std::vector<Unit> kept;     // original order; filled only when reason == compressed
    std::vector<Unit> dropped;  // original order; what assemble() replaces by markers
    std::uint32_t raw_tokens = 0;
    std::uint32_t kept_tokens = 0;
    Reason reason = Reason::off;
};

// Keep ratio for a prompt of `tokens` scorer tokens; 1 when the curve is empty.
float keep_ratio(const Policy& policy, std::uint32_t tokens);

// Cuts the regions into units that tile them exactly. A region that is not droppable, or belongs to
// a message of at most skeleton_tokens, is one mandatory unit. Other regions are cut where cut_prob
// reaches cut_threshold, at least min_unit_tokens apart; a unit that would pass max_unit_tokens is
// cut at the most probable start in range. With `content` given, cuts fall on line starts only.
// Units that carry request tokens are then pinned (mandatory), most request tokens first, while
// they fit in query_pin. Returns an empty vector when the inputs are inconsistent (span lengths,
// offsets, region order); select() turns that into Reason::scorer_failed.
std::vector<Unit> segment(std::span<const Region> regions, const Scores& scores, const Policy& policy,
                          std::string_view content = {});

// Picks the units to keep: every mandatory unit, then the others by density (ties: earlier first)
// until the first one that does not fit in keep_ratio(raw) * droppable_tokens. `units` are sorted
// and disjoint, as segment() returns them; raw is the sum of their tokens. Fallbacks, all with
// empty kept / dropped and kept_tokens == raw_tokens: below_threshold; flat_scores (fewer than two
// units to rank, or density spread <= min_spread); scorer_failed (no units, bad order, non-finite
// density); breakeven (the budget keeps everything).
Plan select(std::span<const Unit> units, std::uint32_t droppable_tokens, const Policy& policy);

// Rebuilds the prompt. Kept bytes stay in their original order; each run of adjacent dropped units
// becomes one marker line, and the first marker is preceded by one note line. A plan that is not
// `compressed` returns `content` byte for byte. Throws std::invalid_argument when the dropped units
// are unordered, overlap or pass the end of `content`.
std::string assemble(std::string_view content, const Plan& plan);

// Follow-up turn. `held` is the plan the conversation was prefilled with; `all` are the same units
// scored against the new request. When the units still omitted hold at least recall_gate of the
// score mass (density * tokens), the heaviest ones are recalled until the rest falls below the
// gate. Result: kept = the recalled units only (original order, new densities), dropped = what
// stays omitted, kept_tokens = held.kept_tokens + recalled tokens. The cached prefix is not rebuilt:
// the caller adds recalled_text() to the new turn.
Plan recall(const Plan& held, std::span<const Unit> all, const Policy& policy);

// Text of the units recalled by recall(): one header line, then the units in original order. Empty
// when nothing was recalled. Throws std::invalid_argument when a unit passes the end of `content`.
std::string recalled_text(std::string_view content, const Plan& recalled);

// Time model of one request: the target prefills at target_tok_s; scoring a prompt of S tokens
// costs drafter_a_s_per_tok * S + drafter_b_s_per_tok2 * S * S seconds.
struct CostModel {
    double target_tok_s = 0;
    double drafter_a_s_per_tok = 0;
    double drafter_b_s_per_tok2 = 0;
};

// True when prefilling `kept` tokens plus the scorer saves at least min_saving_s against prefilling
// `raw` tokens. False for a model without a positive finite rate. A caller that gets false reports
// Reason::breakeven and prefills the full prompt.
bool worth_it(std::uint32_t raw, std::uint32_t kept, const CostModel& cost, const Policy& policy);

// A plan that leaves the prompt alone (flag off, tools present, break-even not met, ...).
Plan bypass(std::uint32_t raw_tokens, Reason reason);

const char* reason_name(Reason reason);

// "pflash raw=N kept=M reason=R": the line every request that reaches this layer prints.
std::string format_receipt(const Plan& plan);

// The marker line (without the newline) that stands for `tokens` omitted tokens.
std::string gap_marker(std::uint64_t tokens);

}  // namespace strix::serve::pflash
