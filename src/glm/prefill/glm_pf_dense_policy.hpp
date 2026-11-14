#pragma once
// Dense prefill linears: which core a call of `rows` rows takes, and what a serving prefill chunk does to that choice.
// Host only (no HIP). One definition for the kernel launcher (glm_pf_exact.hip), the route levers of the rank CLI and
// the server (apps/glm_route_levers.hpp) and the host policy test, so they cannot disagree.
//
// Measurement (whole call, four dense shapes, rows per call): the exact X3 core against the TM16 core is 0.99-1.19 at
// 32 rows, 0.52-0.80 at 64, 0.48-0.62 at 256, 0.44-0.58 at 512 and 0.49-0.55 at 2048 rows; the FP16 WMMA core against
// TM16 is 0.54-0.75 at 32 rows, 0.34-0.49 at 256 and 0.33-0.38 at 2048. So auto takes the large-call core from 64 rows
// (exact, same bits as TM16) and from 32 rows when the FAST WMMA identity is selected.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix::glm::pf::dense_policy {
inline constexpr unsigned kExactMinRows = 64, kWmmaMinRows = 32;
// The values of exact::Route (glm_pf_exact.hpp), checked there.
inline constexpr unsigned kTm = 0, kX = 1, kX2 = 2, kX3 = 3, kX3r8 = 4;

template<class Env> bool wmma_selected(Env&& env) {
    const char* p = env("STRIX_GLM_PF_WMMA");
    return p != nullptr && std::strcmp(p, "1") == 0;
}
// STRIX_GLM_PF_EXACT_X_MIN_ROWS overrides the auto threshold (rows per linear call); =2048 restores the former one.
template<class Env> unsigned auto_min_rows(Env&& env) {
    const char* p = env("STRIX_GLM_PF_EXACT_X_MIN_ROWS");
    if (!p) return wmma_selected(env) ? kWmmaMinRows : kExactMinRows;
    char* end = nullptr;
    const unsigned long v = std::strtoul(p, &end, 10);
    if (!*p || *end || v == 0 || v > 65536) throw std::invalid_argument("STRIX_GLM_PF_EXACT_X_MIN_ROWS must be 1..65536");
    return static_cast<unsigned>(v);
}
struct Choice {
    unsigned route = kTm;     // an exact::Route value
    bool automatic = false;   // STRIX_GLM_PF_EXACT_X=auto decided by rows
    unsigned min_rows = 0;    // the auto threshold (0 when the route is forced)
};
template<class Env> Choice choose(unsigned rows, Env&& env) {
    const char* p = env("STRIX_GLM_PF_EXACT_X");
    if (!p || std::strcmp(p, "0") == 0) return {kTm, false, 0};
    if (std::strcmp(p, "1") == 0) return {kX, false, 0};
    if (std::strcmp(p, "2") == 0) return {kX2, false, 0};
    if (std::strcmp(p, "3") == 0) return {kX3, false, 0};
    if (std::strcmp(p, "4") == 0) return {kX3r8, false, 0};
    if (std::strcmp(p, "auto") == 0) {
        const unsigned floor = auto_min_rows(env);
        return {rows >= floor ? kX3r8 : kTm, true, floor};
    }
    throw std::invalid_argument("STRIX_GLM_PF_EXACT_X must be 0, 1, 2, 3, 4 or auto");
}

// Extrema over all nonempty dense calls in a full serving chunk, including the engine's max_rows remainder.
// Pipeline 2 splits even SP rows into R/2 column calls (also with one SP tile); the legacy/TP1 path splits
// ceil/floor halves. Thus the same bounds apply for every world/seqpar policy. Route selection is monotone.
struct CallRows { unsigned min, max; };
inline CallRows call_rows(unsigned tokens, unsigned max_rows, unsigned pipeline) {
    const unsigned capacity = std::max(1U, max_rows);
    const auto bounds = [pipeline](unsigned rows) -> CallRows {
        return pipeline == 2 ? CallRows{std::max(1U, rows / 2U), rows / 2U + rows % 2U} : CallRows{rows, rows};
    };
    auto calls = bounds(std::min(tokens, capacity));
    if (tokens > capacity && tokens % capacity != 0) {
        const auto tail = bounds(tokens % capacity);
        calls.min = std::min(calls.min, tail.min);
        calls.max = std::max(calls.max, tail.max);
    }
    return calls;
}

// The serving prefill interleave chunk: tokens of a new prompt per decode step while another request decodes (the
// whole prompt when nothing decodes, and always in the rank CLI). 0 = whole prompt (the off switch).
inline constexpr unsigned kServeChunk = 512, kServeChunkCeiling = 1024;
struct Chunk {
    bool explicit_request = false;
    unsigned requested = 0;   // the value of the flag when explicit
    unsigned tokens = 0;      // effective chunk
    unsigned rows = 0;        // minimum rows per dense call of one full chunk (0: unknown/inapplicable)
    unsigned max_call_rows = 0;
    unsigned route = kTm;     // core of the smallest calls; mixed says larger calls select another core
    bool mixed = false;
    bool applicable = true;
    bool wmma = false;        // FAST WMMA identity selected for those calls (a matrix still needs input % 64 == 0)
    bool declined = false;    // chunked calls do not take the large-call core
    std::string reason;
};
template<class Env> Chunk resolve_chunk(bool explicit_request, unsigned requested, unsigned /*world*/, unsigned max_rows,
                                       std::string_view prefill, Env&& env) {
    Chunk c;
    c.explicit_request = explicit_request;
    c.requested = requested;
    c.tokens = explicit_request ? requested : kServeChunk;
    if (prefill != "gemv") {
        c.applicable = false;
        c.reason = "dense X3 route inapplicable to " + std::string(prefill) + " prefill; chunk unchanged";
        return c;
    }
    if (c.tokens == 0) {
        c.reason = "chunking off: every prompt is prefilled whole, as in the CLI; dense route depends on prompt length";
        return c;
    }
    const char* pl = env("STRIX_GLM_PF_PIPELINE");
    const unsigned pipeline = pl != nullptr && std::strcmp(pl, "2") == 0 ? 2U : 1U;
    const auto rows_of = [&](unsigned tokens) {
        return call_rows(tokens, max_rows, pipeline);
    };
    c.rows = rows_of(c.tokens).min;
    Choice pick = choose(c.rows, env);
    if (pick.automatic && pick.route == kTm && !explicit_request) {
        // Default chunk: grow it until its calls reach the auto threshold, within the decode-stall ceiling.
        for (unsigned t = c.tokens * 2U; t <= kServeChunkCeiling; t *= 2U)
            if (choose(rows_of(t).min, env).route != kTm) {
                c.tokens = t;
                c.rows = rows_of(t).min;
                pick = choose(c.rows, env);
                break;
            }
    }
    c.route = pick.route;
    c.max_call_rows = rows_of(c.tokens).max;
    c.mixed = choose(c.max_call_rows, env).route != c.route;
    const bool fast = wmma_selected(env);
    c.wmma = pick.route == kX3r8 && fast;
    const std::string rows = std::to_string(c.rows) + ".." + std::to_string(c.max_call_rows) + "-row dense calls";
    if (!pick.automatic) {
        c.declined = pick.route != kX3r8 && fast;
        c.reason = rows + ", route forced by STRIX_GLM_PF_EXACT_X" + (c.declined ? " (FAST WMMA core declined: wrong_route)" : "");
    } else if (pick.route == kTm) {
        c.declined = true;
        c.reason = rows + ", minimum below the auto threshold of " + std::to_string(pick.min_rows) + " rows: large-call core" +
                   (fast ? " and FAST WMMA identity" : "") + " declined (short_call), " +
                   (c.mixed ? "mixed TM/X3 cores; " : "TM core; ") +
                   (explicit_request ? std::string("raise --prefill-chunk")
                                     : "the default chunk is not grown past " + std::to_string(kServeChunkCeiling) + " tokens") +
                   " or lower STRIX_GLM_PF_EXACT_X_MIN_ROWS";
    } else {
        c.reason = rows + " at or above the auto threshold of " + std::to_string(pick.min_rows) + " rows: " +
                   (c.wmma ? "X3 route with the FAST WMMA core" : "exact X3 core") +
                   (!explicit_request && c.tokens != kServeChunk ? " (default chunk grown from " + std::to_string(kServeChunk) + ")"
                                                                 : std::string());
    }
    return c;
}
inline std::string chunk_line(const Chunk& c) {
    return "effective serve-prefill-chunk=" + std::to_string(c.tokens) + " (" + (c.explicit_request ? "explicit" : "default") +
           "; serving only: one full chunk of a new prompt per decode step while another request decodes; " + c.reason +
           "; prediction only: prompt tails and whole-prompt execution may select other routes)\n";
}
}  // namespace strix::glm::pf::dense_policy
