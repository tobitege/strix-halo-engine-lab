#pragma once

// R2-16: one explicit policy table for the routed-MoE prefill family selection
// (enqueue_prefill in src/hip_moe_q4k_a8_known_rows.hip and the prefill .inc
// files it includes). Every environment switch that used to be read ad hoc at
// its call site is parsed here, once, with the exact same rule as before, so
// the default choices are unchanged. Test overrides (prefill_*_override) and
// the per-call thread_local flags stay at their sites: they are runtime state,
// not configuration.
//
// Deliberately NOT in the table: STRIX_MOE_R16_PROJECT_G. Its default depends
// on the cohort of the call and an invalid value throws at the call site, so
// it is not a process-wide constant.
//
// resolve() is a pure function of a lookup so --print-effective-config can show
// the table for the environment the run WILL have (config file environment
// overlaid on the process environment) before that environment is injected.

#include <cstdlib>
#include <string>

#include "strix/env_flags.hpp"

namespace strix {

struct PrefillMoePolicy {
    bool route_ctrl = false;          // STRIX_QPF2K_MOE_ROUTE_CTRL == "1"
    int tiers = 0;                    // STRIX_QPF2K_MOE_TIERS: "1" or "2", else 0
    bool down_lds = false;            // STRIX_QPF2K_MOE_DOWN_LDS_EPI == "1"
    bool resources = false;           // STRIX_INTPREFILL_RESOURCES starts with '1'
    unsigned route = 0;               // STRIX_QPF2K_MOE_ROUTE: 0..15, else 0
    unsigned small = 0;               // STRIX_QPF2K_MOE_SMALL: 1..128, else 0
    bool tile64 = false;              // STRIX_QPF2K_MOE_TILE64 == "1"
    unsigned p1_lag = 6;              // STRIX_QPF2K_MOE_LAG: 1..65535, else 6
    bool p1_stats = false;            // STRIX_QPF2K_MOE_P1STATS starts with '1'
    bool hdr = false;                 // STRIX_QPF2K_MOE_HDR == "1"
    bool xor64 = false;               // STRIX_QPF2K_MOE_XOR64 == "1"
    bool da16 = false;                // STRIX_QPF2K_MOE_DA16 == "1"
    int core2_raw = 0;                // STRIX_QPF2K_MOE (atoi); validity checked at the core2 site
    bool direct_b = false;            // STRIX_QPF2K_MOE_DIRECT_B == "1"
    bool gateup_half_chunk = false;   // STRIX_MOE_GATEUP_CHUNK starts with '5'
    int wmma = -1;                    // STRIX_PREFILL_MOE_WMMA (atoi), -1 when unset
    bool gsq_f16 = true;              // STRIX_GSQ_PREFILL_F16 unset or not starting with '0'
    unsigned tune = 82;               // STRIX_INTPREFILL_MOE_TUNE: 0..65535, else 0; 82 when unset
    bool tune_automatic = true;       // STRIX_INTPREFILL_MOE_TUNE unset
    unsigned defaulted = 0;           // STRIX_QPF2K_MOE_DEFAULTED: atoi & 3
    unsigned batch = 8;               // STRIX_PREFILL_MOE_BATCH: 1..64, else 8

    bool csr_dump = false;            // STRIX_PREFILL_MOE_CSR_DUMP starts with '1'

    template <class Lookup>
    static PrefillMoePolicy resolve(Lookup&& lookup) {
        const auto exact1 = [&](const char* n) { return detail::env_flag_matches(lookup(n), "1", EnvFlagMatch::exact); };
        const auto first1 = [&](const char* n) { return detail::env_flag_matches(lookup(n), "1", EnvFlagMatch::first_character); };
        const auto number = [&](const char* n, int fallback) { const char* v = lookup(n); return v ? std::atoi(v) : fallback; };
        const auto range = [](int n, int lo, int hi, unsigned bad) { return n >= lo && n <= hi ? static_cast<unsigned>(n) : bad; };
        PrefillMoePolicy p;
        p.route_ctrl = exact1("STRIX_QPF2K_MOE_ROUTE_CTRL");
        if (const char* v = lookup("STRIX_QPF2K_MOE_TIERS"))
            p.tiers = (v[0] == '1' || v[0] == '2') && v[1] == '\0' ? v[0] - '0' : 0;
        p.down_lds = exact1("STRIX_QPF2K_MOE_DOWN_LDS_EPI");
        p.resources = first1("STRIX_INTPREFILL_RESOURCES");
        p.route = range(number("STRIX_QPF2K_MOE_ROUTE", 0), 0, 15, 0U);
        p.small = range(number("STRIX_QPF2K_MOE_SMALL", 0), 1, 128, 0U);
        p.tile64 = exact1("STRIX_QPF2K_MOE_TILE64");
        p.p1_lag = range(number("STRIX_QPF2K_MOE_LAG", 6), 1, 65535, 6U);
        p.p1_stats = first1("STRIX_QPF2K_MOE_P1STATS");
        p.hdr = exact1("STRIX_QPF2K_MOE_HDR");
        p.xor64 = exact1("STRIX_QPF2K_MOE_XOR64");
        p.da16 = exact1("STRIX_QPF2K_MOE_DA16");
        p.core2_raw = number("STRIX_QPF2K_MOE", 0);
        p.direct_b = exact1("STRIX_QPF2K_MOE_DIRECT_B");
        { const char* v = lookup("STRIX_MOE_GATEUP_CHUNK"); p.gateup_half_chunk = v && v[0] == '5'; }
        p.wmma = number("STRIX_PREFILL_MOE_WMMA", -1);
        { const char* v = lookup("STRIX_GSQ_PREFILL_F16"); p.gsq_f16 = v == nullptr || v[0] != '0'; }
        p.tune = range(number("STRIX_INTPREFILL_MOE_TUNE", 82), 0, 65535, 0U);
        p.tune_automatic = lookup("STRIX_INTPREFILL_MOE_TUNE") == nullptr;
        { const char* v = lookup("STRIX_QPF2K_MOE_DEFAULTED"); p.defaulted = v ? static_cast<unsigned>(std::atoi(v)) & 3U : 0U; }
        p.batch = range(number("STRIX_PREFILL_MOE_BATCH", 8), 1, 64, 8U);
        p.csr_dump = first1("STRIX_PREFILL_MOE_CSR_DUMP");
        return p;
    }

    // Visit every field as (env name, resolved value as text).
    template <class Fn>
    void for_each(Fn&& fn) const {
        const auto b = [](bool v) { return std::string(v ? "1" : "0"); };
        const auto n = [](long long v) { return std::to_string(v); };
        fn("STRIX_QPF2K_MOE_ROUTE_CTRL", b(route_ctrl));
        fn("STRIX_QPF2K_MOE_TIERS", n(tiers));
        fn("STRIX_QPF2K_MOE_DOWN_LDS_EPI", b(down_lds));
        fn("STRIX_INTPREFILL_RESOURCES", b(resources));
        fn("STRIX_QPF2K_MOE_ROUTE", n(route));
        fn("STRIX_QPF2K_MOE_SMALL", n(small));
        fn("STRIX_QPF2K_MOE_TILE64", b(tile64));
        fn("STRIX_QPF2K_MOE_LAG", n(p1_lag));
        fn("STRIX_QPF2K_MOE_P1STATS", b(p1_stats));
        fn("STRIX_QPF2K_MOE_HDR", b(hdr));
        fn("STRIX_QPF2K_MOE_XOR64", b(xor64));
        fn("STRIX_QPF2K_MOE_DA16", b(da16));
        fn("STRIX_QPF2K_MOE", n(core2_raw));
        fn("STRIX_QPF2K_MOE_DIRECT_B", b(direct_b));
        fn("STRIX_MOE_GATEUP_CHUNK", b(gateup_half_chunk));
        fn("STRIX_PREFILL_MOE_WMMA", n(wmma));
        fn("STRIX_GSQ_PREFILL_F16", b(gsq_f16));
        fn("STRIX_INTPREFILL_MOE_TUNE", n(tune));
        fn("STRIX_QPF2K_MOE_DEFAULTED", n(defaulted));
        fn("STRIX_PREFILL_MOE_BATCH", n(batch));
        fn("STRIX_PREFILL_MOE_CSR_DUMP", b(csr_dump));
    }

    // JSON object body for --print-effective-config: {"NAME":"value",...}.
    std::string json() const {
        std::string out = "{";
        for_each([&](const char* name, const std::string& value) {
            if (out.size() > 1) out += ',';
            out += '"'; out += name; out += "\":\""; out += value; out += '"';
        });
        return out + "}";
    }
};

// Process-wide policy, resolved once from the process environment on first use
// (no runtime code sets these variables), and recorded in the envcfg registry.
inline const PrefillMoePolicy& prefill_moe_policy() {
    static const PrefillMoePolicy policy = [] {
        const PrefillMoePolicy p = PrefillMoePolicy::resolve([](const char* n) { return std::getenv(n); });
        p.for_each([](const char* name, const std::string& value) { envcfg::record(name, std::getenv(name), value); });
        return p;
    }();
    return policy;
}

}  // namespace strix
