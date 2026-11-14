#pragma once

// Prefill arithmetic options that change bits relative to the FP16 prefill identity (fast-prefill-f16wmma-v1 with
// the exact HC output chains, the token-serial GDN recurrence and the ordered FP32 indexer chains). Every READY option is ON by
// default and individually switchable off; one master switch restores the plain FP16 identity at once.
//   environment: STRIX_PREFILL_EXACT_FP16=1 (all off), STRIX_PREFILL_<NAME>=0 (one off), e.g. STRIX_PREFILL_HC_OUTPUT_WMMA=0
//   command line: --prefill-exact-fp16, --prefill-disable <name>[,<name>...] (names as below, lower case)
// Environment and startup restrictions combine monotonically. The active set names the prefill identity (prefill_identity_name/hash),
// which receipts and the serving capability endpoint report.

#include "strix/env_flags.hpp"
#include "strix/decode_numerical_profile.hpp"
#include "strix/moe_q4k_a8_contract.hpp"
#include "strix/qsa_prefill_budget.hpp"
#include "strix/moe_route_prune.hpp"
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strix {

enum class PrefillOption : unsigned {
    hc_output_wmma,     // HC output: FP16 low values and coefficients, WMMA accumulation (was ordered FP32 chains)
    index_proj_wmma,    // Candidate BF16 indexer projections; explicit opt-in until qualified
    gdn_fp16_chunk,     // Candidate chunked GDN prefill on FP16 WMMA (qwen-gdn-fp16-chunk rev 2/3); explicit opt-in until qualified
    index_score_wmma,   // Candidate FAST BF16 index scores; explicit opt-in until qualified
    count
};

struct PrefillOptionInfo {
    std::string_view name;  // command line / config name
    std::string_view env;   // STRIX_PREFILL_<NAME>
    std::string_view tag;   // identity name component
};

inline constexpr std::array<PrefillOptionInfo, static_cast<unsigned>(PrefillOption::count)> kPrefillOptions{{
    {"hc_output_wmma", "STRIX_PREFILL_HC_OUTPUT_WMMA", "hcout16"},
    {"index_proj_wmma", "STRIX_PREFILL_INDEX_PROJ_WMMA", "idxproj16"},
    {"gdn_fp16_chunk", "STRIX_QPF2K_GDN_FP16_CHUNK", "gdnchunk16"},
    {"index_score_wmma", "STRIX_PREFILL_INDEX_SCORE_WMMA", "idxscore16"},
}};

// Options whose implementation is built in and enabled by default. An option joins this mask only when it is READY;
// Unqualified candidates stay default OFF and require an explicit environment opt-in.
inline constexpr unsigned kPrefillOptionsReady = 1U << static_cast<unsigned>(PrefillOption::hc_output_wmma);

// Revision discriminator, not another user option: admission/disable/accounting still use gdn_fp16_chunk.
inline constexpr unsigned kPrefillGdnChunkRev3 = 1U << static_cast<unsigned>(PrefillOption::count);
// Implemented candidates may be exercised explicitly without promotion to READY.
inline constexpr unsigned kPrefillOptionsCandidates = (1U << static_cast<unsigned>(PrefillOption::index_proj_wmma)) |
    (1U << static_cast<unsigned>(PrefillOption::gdn_fp16_chunk)) |
    (1U << static_cast<unsigned>(PrefillOption::index_score_wmma));

// Candidate index arithmetic is not qualified for MTP or nested scheduling.
inline constexpr bool index_wmma_unqualified_route(unsigned mask, unsigned fast_components) {
    return (fast_components & kFastPrefillQsa) &&
        (mask & ((1U << static_cast<unsigned>(PrefillOption::index_proj_wmma)) |
                 (1U << static_cast<unsigned>(PrefillOption::index_score_wmma))));
}

namespace detail {
struct PrefillIdentityOverride {
    bool set = false;
    bool exact = false;
    unsigned disabled = 0U;
};
inline PrefillIdentityOverride& prefill_identity_override() {
    static PrefillIdentityOverride o;
    return o;
}
}  // namespace detail

inline unsigned prefill_option_bit(PrefillOption option) { return 1U << static_cast<unsigned>(option); }

// Command line: --prefill-exact-fp16 and --prefill-disable a,b. Call before the first prefill (the mask is read per call).
inline void set_prefill_identity_override(bool exact, unsigned disabled) {
    auto& o = detail::prefill_identity_override();
    o.set = true;
    o.exact = o.exact || exact;
    o.disabled |= disabled;
}
inline unsigned parse_prefill_option_list(std::string_view list) {
    if (list.empty() || list.back() == ',')
        throw std::invalid_argument("prefill option list must contain names without a trailing comma");
    unsigned mask = 0U;
    while (!list.empty()) {
        const auto comma = list.find(',');
        const auto item = list.substr(0, comma);
        bool found = false;
        for (unsigned i = 0; i < kPrefillOptions.size(); ++i)
            if (item == kPrefillOptions[i].name) { mask |= 1U << i; found = true; }
        if (!found) throw std::invalid_argument("unknown prefill option '" + std::string(item) + "'");
        list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
    }
    return mask;
}

// Resolve only the supplied environment and restrictions; no process-global state.
template<class Get>
inline unsigned resolve_prefill_options(Get get, bool exact = false, unsigned disabled = 0U) {
    const char* master = get("STRIX_PREFILL_EXACT_FP16");
    if (exact || (master && master[0] == '1')) return 0U;
    unsigned mask = kPrefillOptionsReady & ~disabled;
    for (unsigned i = 0; i < kPrefillOptions.size(); ++i) {
        const char* value = get(kPrefillOptions[i].env.data());
        const auto bit = 1U << i;
        const bool rev3 = i == static_cast<unsigned>(PrefillOption::gdn_fp16_chunk) && value && std::string_view(value) == "3";
        if (value && (std::string_view(value) == "1" || rev3) && (kPrefillOptionsCandidates & bit) && !(disabled & bit)) {
            mask |= bit;
            if (rev3) mask |= kPrefillGdnChunkRev3;
        }
        if (value && value[0] == '0') mask &= ~bit;
    }
    return mask;
}

// The active set: environment, then the command line override.
inline unsigned prefill_options_active() {
    unsigned mask = resolve_prefill_options([](const char* name) { return std::getenv(name); });
    const auto& o = detail::prefill_identity_override();
    if (o.set) mask = o.exact ? 0U : mask & ~o.disabled;
    if (!(mask & prefill_option_bit(PrefillOption::gdn_fp16_chunk))) mask &= ~kPrefillGdnChunkRev3;
    return mask & (kPrefillOptionsReady | kPrefillOptionsCandidates | kPrefillGdnChunkRev3);
}
inline bool prefill_option_enabled(PrefillOption option) { return (prefill_options_active() & prefill_option_bit(option)) != 0U; }

// Explicit draft outside READY; restart to change it. Zero keeps the old identity bytes.
inline std::uint32_t parse_qsa_union_cap(std::string_view text) {
    if (text.empty()) return 0U;
    std::uint32_t value = 0U;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::invalid_argument("STRIX_QPF2K_QSA_UNION_CAP requires a nonnegative decimal uint32");
    return value;
}
inline std::uint32_t qsa_union_cap_active() {
    static const auto cap = [] {
        const char* value = std::getenv("STRIX_QPF2K_QSA_UNION_CAP");
        return parse_qsa_union_cap(value ? value : "");
    }();
    static const bool exact = strix::envcfg::flag("STRIX_PREFILL_EXACT_FP16", false);
    return exact || detail::prefill_identity_override().exact ? 0U : cap;
}

inline std::uint32_t qsa_prefill_budget_active() {
    const char* master = std::getenv("STRIX_PREFILL_EXACT_FP16");
    if (detail::prefill_identity_override().exact || (master && master[0] == '1')) return 0U;
    static const auto budget = [] {
        const char* value = std::getenv("STRIX_QPF2K_QSA_BUDGET");
        return parse_qsa_prefill_budget(value ? value : "");
    }();
    return budget;
}

// One effective draft policy for both arithmetic binding and native submission.
inline moe_core::RoutePrune prefill_route_prune_active() {
    static const bool exact = envcfg::flag("STRIX_PREFILL_EXACT_FP16", false);
    return exact || detail::prefill_identity_override().exact ? moe_core::RoutePrune{} : moe_core::route_prune_active();
}

// Budget and union cap are explicit drafts outside READY. Bind their values after the governed tags; zero preserves old bytes.
inline std::string prefill_identity_name(unsigned mask, std::uint32_t qsa_budget = qsa_prefill_budget_active(),
    std::uint32_t union_cap = qsa_union_cap_active()) {
    std::string name = "fp16-v1";
    for (unsigned i = 0; i < kPrefillOptions.size(); ++i)
        if (mask & (1U << i)) {
            name += '+';
            name += i == static_cast<unsigned>(PrefillOption::gdn_fp16_chunk) && (mask & kPrefillGdnChunkRev3) ?
                "gdnchunk16r3" : kPrefillOptions[i].tag;
        }
    if (qsa_budget) name += "+qsab" + std::to_string(qsa_budget);
    if (union_cap) name += "+qsauc" + std::to_string(union_cap);
    // RoutePrune draft: its spec is part of the identity; off preserves the old bytes.
    if (const auto prune = moe_core::route_prune_tag(prefill_route_prune_active()); !prune.empty())
        name += "+rp" + prune;
    return name;
}
inline std::string prefill_identity_name() { return prefill_identity_name(prefill_options_active()); }
// FNV-1a 64 of the identity name.
inline std::uint64_t prefill_identity_hash(unsigned mask, std::uint32_t qsa_budget = qsa_prefill_budget_active(),
    std::uint32_t union_cap = qsa_union_cap_active()) {
    return host_fnv::text(prefill_identity_name(mask, qsa_budget, union_cap), 14695981039346656037ULL);
}
inline std::uint64_t prefill_identity_hash() { return prefill_identity_hash(prefill_options_active()); }

// Bind the enabled set into compiled execution, then inherited tiled/lineage identities.
inline std::uint64_t bind_prefill_option_identity(std::uint64_t arithmetic, unsigned mask,
    std::uint32_t qsa_budget = qsa_prefill_budget_active(), std::uint32_t union_cap = qsa_union_cap_active()) {
    for (const std::uint64_t word : {UINT64_C(0x5052464f50545331),
             static_cast<std::uint64_t>(mask), prefill_identity_hash(mask, qsa_budget, union_cap)})
        for (unsigned i = 0; i < 8; ++i) {
            arithmetic ^= (word >> (8U * i)) & 0xffU;
            arithmetic *= UINT64_C(1099511628211);
        }
    return arithmetic;
}
// One arithmetic builder for host compilation and native execution. GSQ is
// supplied from the model contract, never inferred from unrelated process state.
inline std::uint64_t prefill_arithmetic_identity(const NumericalProfilePair& pair, unsigned mask, bool gsq,
    std::uint32_t qsa_budget = qsa_prefill_budget_active(), std::uint32_t union_cap = qsa_union_cap_active()) {
    if (!numerical_profile_fast_prefill(pair.prefill) && !qsa_budget && !union_cap) return numerical_profile_pair_identity(pair);
    auto identity = bind_prefill_option_identity(numerical_profile_pair_identity(pair), mask, qsa_budget, union_cap);
    if (gsq && (pair.fast_components & kFastPrefillMoe))
        for (const std::uint64_t word : {UINT64_C(0x4753514d4f455046), moe_q4k_a8::gsq_fast_prefill_moe_identity()})
            for (unsigned i = 0; i < 8; ++i) {
                identity ^= (word >> (8U * i)) & 0xffU;
                identity *= UINT64_C(1099511628211);
            }
    return identity;
}
// Legacy selector shares the public arithmetic controls. Explicit fixture kernel IDs are separate.
inline unsigned governed_hc_output_kernel(unsigned requested, unsigned mask) {
    return requested == 2U && !(mask & prefill_option_bit(PrefillOption::hc_output_wmma)) ? 1U : requested;
}

}  // namespace strix
