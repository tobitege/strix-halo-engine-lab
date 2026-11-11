#pragma once

// STRIX_QWEN_ALLON: one opt-in switch that turns on every Qwen lever that is implemented in this tree but not part of
// the shipped defaults, for composition testing (how far does the engine get with everything on).
//   unset | 0          nothing changes (the shipped defaults, bitwise)
//   1 | exact          every EXACT lever below (bitwise identical to the defaults)
//   fast               the exact levers plus the FAST levers (bit-changing; qualification in progress)
// An explicit per-lever value always wins (the profile only fills variables that are unset), and STRIX_BASELINE=1 vetoes
// the profile. The profile exports plain environment variables before main, exactly as a user would, so every lever keeps
// its own admission, decline notice and off switch. Build-time levers (STRIX_DPP_REDUCE) and levers that replace one
// another (TILE64 instead of the tier split, TIERS=2 instead of 1) are not part of the profile; they are separate arms.
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

namespace strix::qwen_allon {

inline constexpr const char* master_env = "STRIX_QWEN_ALLON";

struct Lever {
    const char* name;     // short lever name (receipts, table)
    const char* env;      // environment variable the profile sets
    const char* on;       // value the profile gives it
    bool fast;            // FAST (bit-changing, qualification in progress) instead of exact
    const char* receipt;  // where the lever comes from / its last evidence
};

inline constexpr std::array levers{
    // Exact levers.
    Lever{"prompt_group_by_load", "STRIX_PROMPT_BATCH_AUTO", "1", false,
        "serve scheduling; 8x1024 burst first-token p50 -21 %, tokens equal"},
    Lever{"ple_panel_reuse", "STRIX_QPF2K_PLE_TILE", "1", false,
        "PLE operator exact on both nodes; engine gain not shown"},
    Lever{"hc_recompute", "STRIX_QPF2K_HC_RECOMPUTE_NORM", "1", false,
        "HC normalized rows recomputed in consumers; OUT16 17.006 -> 14.991 ms per call; engine bytes pending"},
    Lever{"hc_inverse_reuse", "STRIX_QPF2K_HC_OUT16_INVERSE_REUSE", "1", false,
        "needs hc_recompute; 1,327 host cases, native ISA/byte gates pending"},
    Lever{"moe_tiers", "STRIX_QPF2K_MOE_TIERS", "1", false,
        "narrow routed MoE tier in a 3-CTA LDS budget; byte gate and timing pending"},
    Lever{"moe_down_lds_epilogue", "STRIX_QPF2K_MOE_DOWN_LDS_EPI", "1", false,
        "down LDS-backed vector stores; byte gate and timing pending"},
    Lever{"moe_direct_b", "STRIX_QPF2K_MOE_DIRECT_B", "1", false,
        "Q4_K gate/up direct-B fragments (plain CLI prefill only); native gate pending"},
    Lever{"qsa_kv_pair", "STRIX_QSA_FAST_KV_PAIR", "1", false,
        "single-image paired QSA attend loader; fixture byte-equal, native pending"},
    Lever{"mtp_device_handoff", "STRIX_MTP_DEVICE_HANDOFF", "1", false,
        "MTP cycle without a host fence; same launches and stream order"},
    Lever{"decode_fuse", "STRIX_QWEN_DECODE_FUSE", "1", false,
        "single-row GDN preparation fusion; host and fixture checks"},
    Lever{"router_bf16", "STRIX_QWEN_ROUTER_BF16", "1", false,
        "exact BF16 router weight copy (120 MiB); device gate pending"},
    Lever{"sampled_coupled_accept", "STRIX_MTP_SAMPLED_ACCEPT", "coupled", false,
        "sampled MTP: drafted output equals plain sampled output per seed (3,456 runs)"},
    // FAST levers (bit-changing; named identities, qualification in progress).
    Lever{"gdn_fp16_chunk", "STRIX_QPF2K_GDN_FP16_CHUNK", "1", true,
        "262,144-token prefill 177.9 -> 170.0 s; relL2 3.3-4.4e-4 against FP64"},
    Lever{"moe_route_bf16", "STRIX_QPF2K_MOE_ROUTE_BF16", "1", true,
        "BF16 routed output storage"},
    Lever{"moe_route_i8", "STRIX_PREFILL_MOE_ROUTE_I8", "1", true,
        "one-byte route dot storage with per-group exponent; needs moe_route_bf16"},
    Lever{"index_proj_wmma", "STRIX_PREFILL_INDEX_PROJ_WMMA", "1", true,
        "BF16 indexer projections on WMMA"},
    Lever{"index_score_wmma", "STRIX_PREFILL_INDEX_SCORE_WMMA", "1", true,
        "BF16 index score on WMMA; P16384 unpacked regression recorded, repair pending"},
    Lever{"qsa_budget_1536", "STRIX_QPF2K_QSA_BUDGET", "1536", true,
        "attention budget 1536 (kept at 1536 by decision)"},
};

enum class Mode { off, exact, fast, invalid };

template<class Get>
Mode mode(Get get) {
    const char* value = get(master_env);
    if (!value || !*value) return Mode::off;
    const std::string_view text(value);
    if (text == "0") return Mode::off;
    if (text == "1" || text == "exact") return Mode::exact;
    if (text == "fast") return Mode::fast;
    return Mode::invalid;
}
inline Mode mode() { return mode([](const char* name) { return std::getenv(name); }); }

inline const char* mode_name(Mode value) noexcept {
    return value == Mode::exact ? "exact" : value == Mode::fast ? "fast" : value == Mode::invalid ? "invalid" : "off";
}
inline bool selected(const Lever& lever, Mode value) noexcept {
    return value == Mode::fast || (value == Mode::exact && !lever.fast);
}

// Startup provenance, like qpf2k::pre_injected: never infer automatic ownership
// merely from ALLON being present. Explicit per-lever values retain their contract.
inline std::array<bool, levers.size()> injected{};
inline bool injected_by_profile(std::string_view name) {
    for (std::size_t i = 0; i < levers.size(); ++i)
        if (name == levers[i].env) return injected[i];
    return false;
}
// Conservative exclusion at the measured 262K prompt boundary,
// not a measured crossover. Smaller prompts retain their previous selection.
inline constexpr std::size_t long_prompt_rows = 262143U;
inline bool long_prompt_declined(std::string_view name, std::size_t rows) {
    return rows >= long_prompt_rows &&
        (name == "STRIX_QPF2K_MOE_DIRECT_B" ||
         name == "STRIX_QPF2K_HC_RECOMPUTE_NORM" ||
         name == "STRIX_QWEN_ROUTER_BF16");
}
template<class Get, class Put>
unsigned gate_long_prompt(std::size_t rows, Get get, Put put) {
    unsigned gated = 0;
    for (const auto& lever : levers) {
        const char* value = get(lever.env);
        if (injected_by_profile(lever.env) && long_prompt_declined(lever.env, rows) &&
            value && std::string_view(value) == lever.on) {
            put(lever.env, "0");
            ++gated;
        }
    }
    return gated;
}

// Fills every selected lever whose variable is unset; returns the number of variables it set. STRIX_BASELINE=1 vetoes.
template<class Get, class Put>
unsigned apply(Get get, Put put) {
    const Mode value = mode(get);
    if (value == Mode::off || value == Mode::invalid) return 0U;
    if (const char* baseline = get("STRIX_BASELINE"); baseline && std::string_view(baseline) == "1") return 0U;
    unsigned set = 0U;
    for (const auto& lever : levers) {
        if (!selected(lever, value)) continue;
        if (get(lever.env)) continue;
        if (put(lever.env, lever.on)) {
            injected[static_cast<std::size_t>(&lever - levers.data())] = true;
            ++set;
        }
    }
    return set;
}

// One JSON object for --print-effective-config; empty when the profile is off.
template<class Get>
std::string json(Get get, std::size_t rows = 0) {
    const Mode value = mode(get);
    if (value == Mode::off) return {};
    std::string out = std::string("{\"mode\":\"") + mode_name(value) + "\"";
    const char* baseline = get("STRIX_BASELINE");
    out += baseline && std::string_view(baseline) == "1" ? ",\"vetoed\":\"STRIX_BASELINE\"" : "";
    out += ",\"levers\":[";
    bool first = true;
    for (const auto& lever : levers) {
        if (!selected(lever, value)) continue;
        const char* current = get(lever.env);
        out += first ? "" : ",";
        first = false;
        out += std::string("{\"name\":\"") + lever.name + "\",\"env\":\"" + lever.env + "\",\"value\":\"" +
            (current ? current : "unset") + "\",\"long_prompt_gate\":" +
            (injected_by_profile(lever.env) && long_prompt_declined(lever.env, rows) ? "true" : "false") +
            ",\"class\":\"" + (lever.fast ? "fast" : "exact") + "\"}";
    }
    return out + "]}";
}
inline std::string json() { return json([](const char* name) { return std::getenv(name); }); }

}  // namespace strix::qwen_allon
