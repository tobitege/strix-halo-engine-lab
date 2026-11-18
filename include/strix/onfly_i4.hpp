#pragma once
// TURBO+ onfly INT4 weights for Qwen (decode dense projections): the request, its tensor-group switches, the named
// identity that enters every cache key, and the tensor classifier. One resolver (resolve_onfly_i4) serves the CLI and
// the HTTP server (same binary, same RequestedOptions); the effective values are printed by --print-effective-config.
//
// The weight source itself (--turbo-weights native|onfly|converted, --turbo-weight-bits) is owned by
// strix/turbo.hpp and resolved by strix::qwen::profile; this header adds only the Qwen tensor-group sub-switches:
//   --turbo-onfly-groups LIST                STRIX_TURBO_ONFLY_GROUPS    gdn_in,gdn_out,qsa_io,shared_expert (default: all)
//   --turbo-onfly-scale-rule legacy|full|mse STRIX_TURBO_ONFLY_SCALE    default full (core/iu4.hpp fast full range;
//                                            mse = per-group clip search at load, same nibble grid and kernel)
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include "core/iu4_rows_plan.hpp"

namespace strix::onfly_i4 {

enum Group : unsigned { gdn_in = 1U, gdn_out = 2U, qsa_io = 4U, shared_expert = 8U, all_groups = 15U };
inline constexpr std::string_view kIdentity = "qwen-turbo-plus-onfly-i4g64-v2";
inline constexpr std::string_view kScopeNote = "decode/window dense projections i4g64; cohort/batching unqualified; prefill (rows >= 16) q8_0";
inline constexpr int kScaleLegacy = 0, kScaleFullRange = 2001, kScaleMse = 2002;
inline constexpr bool dispatch_width(std::uint64_t k) {
    return k == 2048U || k == 2560U || k == 4096U || k == 6144U;
}
// Cohort/batch execution remains unqualified. IU4 never consumes A8 slots.
// Refuse these routes rather than silently falling back to different weights.
inline void require_serial(bool active, bool window_route) {
    if (active && window_route)
        throw std::invalid_argument("onfly INT4 cohort/batching or projection source is not serial-equivalence qualified");
}
inline void require_serial_projection(unsigned rows, bool from_a8_slot) {
    require_serial(true, rows < 1U || rows > 8U || from_a8_slot);
}
inline std::string_view scale_rule_name(int rule) { return rule == kScaleLegacy ? "legacy" : rule == kScaleMse ? "mse" : "full"; }

enum class Source : unsigned { native = 0, onfly = 1, converted = 2 };

struct Request {  // as typed (empty = not given)
    std::string groups, scale_rule;
};
struct Resolved {
    Source source = Source::native;
    unsigned bits = 4, groups = all_groups;
    int scale_rule = kScaleFullRange;
    core::iu4_rows::SingleDecodeResolution row_decode;
    core::iu4_rows::RowReductionResolution row_reduction;
    [[nodiscard]] bool active() const { return source == Source::onfly; }
    // Identity for cache keys / receipts; empty for native (byte-identical to the unmodified engine).
    [[nodiscard]] std::string identity() const {
        if (!active()) return {};
        auto value = std::string(kIdentity) + "-g" + std::to_string(groups) + "-" + std::string(scale_rule_name(scale_rule));
        if (row_reduction.effective == core::iu4_rows::RowReduction::canonical)
            value += "-" + std::string(core::iu4_rows::row_reduction_contract(row_reduction.effective));
        return value;
    }
    [[nodiscard]] std::string json() const {
        return "{\"identity\":\"" + identity() + "\",\"group_mask\":" + std::to_string(groups) +
            ",\"scale_rule\":\"" + std::string(scale_rule_name(scale_rule)) + "\"}";
    }
};

inline std::string_view group_name(unsigned bit) {
    switch (bit) {
    case gdn_in: return "gdn_in";
    case gdn_out: return "gdn_out";
    case qsa_io: return "qsa_io";
    case shared_expert: return "shared_expert";
    default: return "?";
    }
}
inline unsigned parse_groups(std::string_view list) {
    if (list.empty() || list == "all") return all_groups;
    unsigned mask = 0;
    while (!list.empty()) {
        const auto comma = list.find(',');
        const auto item = list.substr(0, comma);
        unsigned bit = 0;
        for (unsigned b : {gdn_in, gdn_out, qsa_io, shared_expert}) if (group_name(b) == item) bit = b;
        if (!bit) throw std::invalid_argument("--turbo-onfly-groups: unknown group '" + std::string(item) + "'");
        mask |= bit;
        list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1U);
    }
    return mask;
}
inline std::string groups_text(unsigned mask) {
    std::string out;
    for (unsigned b : {gdn_in, gdn_out, qsa_io, shared_expert})
        if (mask & b) out += (out.empty() ? "" : ",") + std::string(group_name(b));
    return out.empty() ? "none" : out;
}

// CLI > environment > default; `env(name)` returns nullptr when unset.
// `onfly`: the profile resolution's effective weight source is onfly at 4 bits (turbo::Weights).
inline Resolved resolve(const Request& cli, const std::function<const char*(const char*)>& env, bool onfly) {
    const auto pick = [&](const std::string& typed, const char* name) {
        if (!typed.empty()) return typed;
        const char* v = env(name);
        return v ? std::string(v) : std::string{};
    };
    Resolved r;
    r.source = onfly ? Source::onfly : Source::native;
    r.groups = parse_groups(pick(cli.groups, "STRIX_TURBO_ONFLY_GROUPS"));
    const auto rule = pick(cli.scale_rule, "STRIX_TURBO_ONFLY_SCALE");
    if (rule.empty() || rule == "full") r.scale_rule = kScaleFullRange;
    else if (rule == "legacy") r.scale_rule = kScaleLegacy;
    else if (rule == "mse") r.scale_rule = kScaleMse;
    else throw std::invalid_argument("--turbo-onfly-scale-rule: expected legacy, full or mse");
    r.row_decode = core::iu4_rows::resolve_single_decode_policy(r.active(), pick({}, "STRIX_IU4_R8_SINGLE_DECODE"));
    r.row_reduction = core::iu4_rows::resolve_row_reduction(r.active(), pick({}, "STRIX_IU4_ROW_REDUCTION"));
    return r;
}

// Group of a Q8_0 dense projection by tensor name (0 = not an onfly candidate).
inline unsigned classify(std::string_view name) {
    const auto ends = [&](std::string_view s) { return name.size() >= s.size() && name.substr(name.size() - s.size()) == s; };
    if (ends("attn_qkv.weight") || ends("attn_gate.weight")) return gdn_in;
    if (ends("ssm_out.weight")) return gdn_out;
    if (ends("attn_q.weight") || ends("attn_k.weight") || ends("attn_v.weight") || ends("attn_output.weight")) return qsa_io;
    if (ends("ffn_gate_shexp.weight") || ends("ffn_up_shexp.weight") || ends("ffn_down_shexp.weight")) return shared_expert;
    return 0;
}

}  // namespace strix::onfly_i4
