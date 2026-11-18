#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::serve {

// Prefill of a newly admitted prompt between the decode rounds of the running requests (engine neutral, pure
// arithmetic; a scheduler that chunks a prompt's prefill calls plan() before every chunk).
//
// A prefill call has a cost per call (plan, launches of short row counts, one exchange per layer on a pair) on top of
// its cost per token, so many small chunks cost more per token than few large ones. A chunk also stalls every decoding
// request for its whole duration, so grouping is admitted by an estimate of the stall the decode cohort may see.
// Each stalled request pays that cost: a small cohort affords a large chunk, a large cohort a small one.
//
// plan() keeps the scheduler's base ratio (one decode round per `base` prompt tokens) and changes only the grouping:
// a chunk of k x base tokens is followed by k decode rounds, so the decoding requests receive the same rounds per
// prompt token and the prompt reaches its first token sooner. The first chunk of a prompt stays at `base`: it measures
// the cost per token on this machine, used to estimate later chunks against `max_stall_ms` (not a latency ceiling).
//
// Engines with shape-dependent arithmetic must supply route_identity to preserve the base calls' routes, including
// tails. Route preservation alone does not qualify native numerical equality for a new grouping.
struct PrefillInterleave {
    enum class Select { automatic, off };
    static constexpr unsigned kDefaultScale = 4;
    static constexpr double kDefaultMaxStallMs = 3000.0;
    static constexpr std::size_t kDefaultSmallCohort = 2;

    Select select = Select::automatic;
    unsigned max_scale = kDefaultScale;                 // largest chunk, in base chunks (power of two)
    double max_stall_ms = kDefaultMaxStallMs;           // grouping admission estimate from the latest call
    std::size_t small_cohort = kDefaultSmallCohort;     // decoding requests that still take max_scale
    bool decode_ratio = true;                           // false: one decode round per chunk, whatever its size
    // Stable identity for a uniform call; nullopt for mixed/unknown routes (decline grouping).
    // Capture engine policy at startup, never read environment/configuration in this callback.
    std::function<std::optional<unsigned>(std::size_t)> route_identity;

    struct Plan {
        std::size_t tokens = 0;      // tokens of the next prefill call
        unsigned decode_rounds = 1;  // decode rounds owed to the cohort before the call after this one
        unsigned scale = 1;          // base chunks grouped in this call
    };

    // The largest grouping the cohort affords: max_scale up to small_cohort decoding requests, halved for every
    // doubling of the cohort beyond it.
    [[nodiscard]] constexpr unsigned cohort_scale(std::size_t decoding) const noexcept {
        unsigned s = max_scale == 0 ? 1U : max_scale;
        for (std::size_t lanes = small_cohort == 0 ? 1 : small_cohort; lanes < decoding && s > 1U; lanes *= 2U) s /= 2U;
        return s;
    }
    //   base          the scheduler's chunk (tokens per decode round); 0 = no chunking
    //   remaining     prompt tokens not yet prefilled
    //   decoding      requests a decode round would advance now
    //   cap           grouping cap (0 = no bound); base, unchunked and no-cohort calls may exceed it.
    //                 The engine must split those calls safely or validate their sizes separately.
    //   ms_per_token  measured cost of this prompt's earlier chunks (0 = none yet: the call stays at base)
    [[nodiscard]] constexpr Plan plan(std::size_t base, std::size_t remaining, std::size_t decoding, std::size_t cap,
                                      double ms_per_token) const noexcept {
        Plan p;
        if (base == 0 || decoding == 0) {  // nothing to protect: the rest of the prompt in one call
            p.tokens = remaining;
            p.decode_rounds = 0;
            return p;
        }
        unsigned s = 1;
        if (select == Select::automatic && ms_per_token == ms_per_token && ms_per_token > 0.0) {
            s = cohort_scale(decoding);
            while (s > 1U && (static_cast<double>(s * base) * ms_per_token > max_stall_ms || (cap != 0 && s * base > cap) ||
                             !keeps_routes(base, std::min<std::size_t>(remaining, s * base)))) s /= 2U;
        }
        p.tokens = std::min<std::size_t>(remaining, s * base);
        p.scale = static_cast<unsigned>((p.tokens + base - 1U) / base);
        p.decode_rounds = decode_ratio ? std::max(1U, p.scale) : 1U;
        return p;
    }
    [[nodiscard]] std::string describe(std::size_t base) const {
        if (select == Select::off || max_scale <= 1U)
            return "prefill-interleave=off (one chunk of " + std::to_string(base) + " tokens per decode round)";
        char stall[32];
        std::snprintf(stall, sizeof(stall), "%.0f", max_stall_ms);
        return "prefill-interleave=auto (serving only, the CLI has no decode cohort: after a first chunk of " + std::to_string(base) +
               " tokens, chunks of up to " + std::to_string(max_scale) + " x " + std::to_string(base) + " tokens while at most " +
               std::to_string(small_cohort) + " requests decode, halved per doubling of the cohort, grouping admission estimate " +
               stall + " ms from the latest call (not a latency ceiling); " +
               (route_identity ? "grouping preserves base and tail arithmetic routes; " : "") +
               (decode_ratio ? "one decode round per " + std::to_string(base) + " prompt tokens is kept across prompt completion"
                                               : std::string("one decode round per chunk")) +
               "; STRIX_SERVE_PREFILL_INTERLEAVE=0 restores one base chunk per round)";
    }

    // STRIX_SERVE_PREFILL_INTERLEAVE = auto (default) | 0; _SCALE (power of two, default 4); _STALL_MS (default 3000);
    // _SMALL_COHORT (default 2); _ROUNDS = ratio (default) | 1. `env(name)` returns the value, or null / empty.
    template <class Env>
    [[nodiscard]] static PrefillInterleave from_env(Env&& env) {
        const auto get = [&](const char* name) { return text(env(name)); };
        PrefillInterleave a;
        if (const std::string s = get("STRIX_SERVE_PREFILL_INTERLEAVE"); !s.empty()) {
            if (s == "auto" || s == "1" || s == "on") a.select = Select::automatic;
            else if (s == "0" || s == "off") a.select = Select::off;
            else throw std::invalid_argument("STRIX_SERVE_PREFILL_INTERLEAVE is auto or 0, not '" + s + "'");
        }
        if (const std::string s = get("STRIX_SERVE_PREFILL_INTERLEAVE_SCALE"); !s.empty()) {
            const long long v = std::stoll(s);
            if (v < 1 || v > 64 || (v & (v - 1)) != 0)
                throw std::invalid_argument("STRIX_SERVE_PREFILL_INTERLEAVE_SCALE must be a power of two in 1..64");
            a.max_scale = static_cast<unsigned>(v);
        }
        if (const std::string s = get("STRIX_SERVE_PREFILL_INTERLEAVE_STALL_MS"); !s.empty()) {
            a.max_stall_ms = std::stod(s);
            if (!std::isfinite(a.max_stall_ms) || a.max_stall_ms <= 0.0)
                throw std::invalid_argument("STRIX_SERVE_PREFILL_INTERLEAVE_STALL_MS must be finite and positive");
        }
        if (const std::string s = get("STRIX_SERVE_PREFILL_INTERLEAVE_SMALL_COHORT"); !s.empty()) {
            const long long v = std::stoll(s);
            if (v < 1) throw std::invalid_argument("STRIX_SERVE_PREFILL_INTERLEAVE_SMALL_COHORT must be at least 1");
            a.small_cohort = static_cast<std::size_t>(v);
        }
        if (const std::string s = get("STRIX_SERVE_PREFILL_INTERLEAVE_ROUNDS"); !s.empty()) {
            if (s == "ratio") a.decode_ratio = true;
            else if (s == "1") a.decode_ratio = false;
            else throw std::invalid_argument("STRIX_SERVE_PREFILL_INTERLEAVE_ROUNDS is ratio or 1, not '" + s + "'");
        }
        return a;
    }

private:
    [[nodiscard]] bool keeps_routes(std::size_t base, std::size_t tokens) const {
        if (!route_identity || tokens <= base) return true;
        const auto grouped = route_identity(tokens);
        if (!grouped || grouped != route_identity(base)) return false;
        const auto tail = tokens % base;
        return tail == 0 || grouped == route_identity(tail);
    }
    static std::string text(const char* v) { return v ? std::string(v) : std::string(); }
    static std::string text(std::string v) { return v; }
};

// Scheduler-owned: completion/cancellation of a prompt must not erase service owed to existing decoders.
struct PrefillDecodeDebt {
    unsigned rounds = 0;
    void charge(const PrefillInterleave::Plan& plan) noexcept { rounds = plan.decode_rounds; }
    void decoded() noexcept { if (rounds != 0) --rounds; }
    void clear() noexcept { rounds = 0; }
    [[nodiscard]] bool ready() const noexcept { return rounds == 0; }
};

// Timeline of one prompt prefilled in chunks between decode rounds: where its admission-to-first-token window went.
// The scheduler adds one entry per chunk; line() is the receipt (one line, printed by the server when asked).
struct PrefillTimeline {
    struct Chunk {
        std::uint32_t tokens = 0;
        std::uint32_t decoding = 0;       // requests decoding when the chunk started
        std::uint32_t decode_rounds = 0;  // decode rounds since the previous chunk (or the admission)
        double chunk_ms = 0;              // the prefill call, control exchange included
        double control_ms = 0;            // of chunk_ms: the control exchange with the other rank
        double decode_ms = 0;             // decode rounds since the previous chunk: the wait for the decode cohort
        double device_ms = 0;             // of chunk_ms: device time of the engine's components (profiled runs only)
        double exchange_ms = 0;           // of chunk_ms: activation exchange incl. the wait for the other rank (profiled)
    };
    std::vector<Chunk> chunks;
    double window_ms = 0;  // admission -> last chunk done
    bool profiled = false;

    void add(const Chunk& c) { chunks.push_back(c); }
    [[nodiscard]] std::string line(std::size_t per_chunk_limit = 20) const {
        Chunk sum;
        for (const Chunk& c : chunks) {
            sum.tokens += c.tokens;
            sum.decode_rounds += c.decode_rounds;
            sum.chunk_ms += c.chunk_ms;
            sum.control_ms += c.control_ms;
            sum.decode_ms += c.decode_ms;
            sum.device_ms += c.device_ms;
            sum.exchange_ms += c.exchange_ms;
        }
        const double tokens = sum.tokens ? static_cast<double>(sum.tokens) : 1.0;
        char b[512];
        std::snprintf(b, sizeof(b),
                      "prefill-timeline tokens=%u chunks=%zu window_ms=%.1f (%.3f ms/token) chunk_ms=%.1f (%.3f) control_ms=%.1f "
                      "decode_wait_ms=%.1f (%.3f) decode_rounds=%u host_other_ms=%.1f (%.3f)",
                      sum.tokens, chunks.size(), window_ms, window_ms / tokens, sum.chunk_ms, sum.chunk_ms / tokens, sum.control_ms,
                      sum.decode_ms, sum.decode_ms / tokens, sum.decode_rounds, window_ms - sum.chunk_ms - sum.decode_ms,
                      (window_ms - sum.chunk_ms - sum.decode_ms) / tokens);
        std::string out = b;
        if (profiled) {
            std::snprintf(b, sizeof(b), " device_ms=%.1f (%.3f) exchange_ms=%.1f (%.3f) chunk_host_ms=%.1f", sum.device_ms,
                          sum.device_ms / tokens, sum.exchange_ms, sum.exchange_ms / tokens,
                          sum.chunk_ms - sum.device_ms - sum.exchange_ms - sum.control_ms);
            out += b;
        }
        out += " | tokens:lanes:chunk_ms:rounds:decode_ms";
        if (profiled) out += ":device_ms:exchange_ms";
        for (std::size_t i = 0; i < chunks.size() && i < per_chunk_limit; ++i) {
            const Chunk& c = chunks[i];
            std::snprintf(b, sizeof(b), " %u:%u:%.0f:%u:%.0f", c.tokens, c.decoding, c.chunk_ms, c.decode_rounds, c.decode_ms);
            out += b;
            if (profiled) {
                std::snprintf(b, sizeof(b), ":%.0f:%.0f", c.device_ms, c.exchange_ms);
                out += b;
            }
        }
        if (chunks.size() > per_chunk_limit) out += " ...";
        return out;
    }
};

}  // namespace strix::serve
