#pragma once
// Shared FAST prefill bundle of an engine: one switch (e.g. STRIX_DS41_FAST_PREFILL=1, STRIX_MM_FAST_PREFILL=1) selects
// named member levers. Precedence per member: a set member variable (any non-empty value, "0" = off) wins over the
// bundle; an explicit OFF therefore beats it. Two readers share these rules:
//   lever()   the device side (process environment through envcfg, recorded for the effective config);
//   expand()  the host resolver: canonical member values that feed the engine's profile table resolution
//             (strix/engine_profile.hpp), its rank agreement, its cache identity and its effective-config printing.
// Header-only and host-only. The bundle switch is on only for exactly "1".
#include "strix/env_flags.hpp"

#include <functional>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::fast_profile {

struct Member {
    const char* env;            // the lever variable
    const char* value;          // value the bundle selects
    const char* policy;         // named policy id of that value (receipts, cache identity)
    const char* accept = "1";   // space-separated values that are "on" (canonicalized as given); anything else is "0"
};

struct Bundle {
    const char* env;            // the one switch
    const char* profile;        // named profile id, e.g. "ds41-fast-pf-v1"
    std::span<const Member> members;
};

using Getenv = std::function<const char*(const char*)>;
using Admit = std::function<bool(const Member&)>;     // bundle member admitted in this configuration (empty: all)
using Overrides = std::map<std::string, std::string>;  // canonical values the resolver hands to the profile table

inline bool switch_on(const char* v) { return v != nullptr && v[0] == '1' && v[1] == 0; }
inline bool set(const char* v) { return v != nullptr && *v != 0; }

// Canonical ranges for a parsed policy mask; syntax/provenance must not affect agreement.
template <class Contains>
inline std::string layer_ranges(unsigned end, const Contains& contains) {
    std::string out;
    for (unsigned l = 0; l < end; ++l) {
        if (!contains(l)) continue;
        const unsigned first = l;
        while (l + 1 < end && contains(l + 1)) ++l;
        if (!out.empty()) out += ',';
        out += std::to_string(first);
        if (l != first) out += '-' + std::to_string(l);
    }
    return out.empty() ? "none" : out;
}

inline bool accepted(const Member& m, std::string_view raw) {
    std::string_view list(m.accept);
    while (!list.empty()) {
        const auto sp = list.find(' ');
        if (list.substr(0, sp) == raw) return !raw.empty();
        if (sp == std::string_view::npos) break;
        list.remove_prefix(sp + 1);
    }
    return false;
}
inline std::string canonical(const Member& m, std::string_view raw) { return accepted(m, raw) ? std::string(raw) : "0"; }

// Device-side reader: the explicit member value if set, else the bundle's value when the bundle is on, else "".
inline const char* lever(const Bundle& b, const Member& m) {
    if (const char* e = strix::envcfg::text(m.env); set(e)) return e;
    return switch_on(strix::envcfg::text(b.env)) ? m.value : "";
}

// Host resolver: canonical bundle switch and member values. A set member is canonicalized; an unset member takes
// the bundle's value when the bundle is on and `admit` (if any) admits it; unset and not selected: absent.
inline Overrides expand(const Bundle& b, const Getenv& raw, const Admit& admit = {}) {
    Overrides out;
    const char* sw = raw(b.env);
    const bool on = switch_on(sw);
    if (set(sw)) out.emplace(b.env, on ? "1" : "0");
    for (const auto& m : b.members) {
        if (const char* v = raw(m.env); set(v)) out.emplace(m.env, canonical(m, v));
        else if (on && (!admit || admit(m))) out.emplace(m.env, m.value);
    }
    return out;
}

// The lookup the profile resolution reads: overrides first, then the raw environment (empty = unset).
inline Getenv lookup(const Overrides& overrides, Getenv raw) {
    return [&overrides, raw = std::move(raw)](const char* key) -> const char* {
        if (const auto it = overrides.find(key); it != overrides.end()) return it->second.c_str();
        const char* v = raw(key);
        return set(v) ? v : nullptr;
    };
}

}  // namespace strix::fast_profile
