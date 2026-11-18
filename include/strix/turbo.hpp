#pragma once

#include "strix/fnv.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// TURBO product mechanics shared by every engine (host only, header only; strix/engine_profile.hpp includes it).
// TURBO is a product profile selected by name ('--profile turbo'), never a default: it turns on every speed lever
// whose measured loss stays inside the relative bar (tools/strix-gate/fast_logit_gate.py compare --mode turbo).
// Every family carries a versioned name (qwen-turbo-v1, ds41-turbo-v1, glm-turbo-v1, minimax-turbo-v1) that enters
// the rank agreement and the cache identity even when every lever of the profile is switched off.
//
// Weight source (owner requirement, a toggle like any other lever; CLI > environment > profile > builtin):
//   --turbo-weights native|onfly|converted   STRIX_TURBO_WEIGHTS       native: our packs as they are
//   --turbo-weight-bits 4|3|2                STRIX_TURBO_WEIGHT_BITS   onfly / converted only (4 first)
//   --turbo-weights-pack PATH                STRIX_TURBO_WEIGHTS_PACK  the pre-converted file of 'converted'
// onfly rounds our values to the selected width at load (per-group RTN with scales, optional Hadamard) and keeps them
// in memory only; converted reads a pre-converted low-bit file. A source or width the build cannot run is never
// replaced silently: the resolution records requested and effective values with the reason, prints them in
// --print-effective-config, and require_honoured() refuses the run.
namespace strix::profile::turbo {

using Env = std::function<const char*(const char*)>;

enum class Source : unsigned { native = 0, onfly = 1, converted = 2 };
inline constexpr std::array<std::string_view, 3> kSourceNames{"native", "onfly", "converted"};
inline constexpr std::string_view kWeightsEnv = "STRIX_TURBO_WEIGHTS";
inline constexpr std::string_view kBitsEnv = "STRIX_TURBO_WEIGHT_BITS";
inline constexpr std::string_view kPackEnv = "STRIX_TURBO_WEIGHTS_PACK";
inline constexpr std::string_view kWeightsOption = "--turbo-weights";
inline constexpr std::string_view kBitsOption = "--turbo-weight-bits";
inline constexpr std::string_view kPackOption = "--turbo-weights-pack";
// Cache namespace of the resolved TURBO run, exported by the profile so every cache-key builder of the process
// (prefix, session, drafter) reads one value; empty or unset outside TURBO (legacy namespaces unchanged).
inline constexpr std::string_view kCacheNamespaceEnv = "STRIX_TURBO_CACHE_NS";

inline const char* source_name(Source s) { return kSourceNames[static_cast<unsigned>(s)].data(); }
inline Source parse_source(std::string_view value) {
    for (unsigned i = 0; i < kSourceNames.size(); ++i)
        if (value == kSourceNames[i]) return static_cast<Source>(i);
    throw std::invalid_argument("--turbo-weights must be native, onfly or converted (got '" + std::string(value) + "')");
}
inline unsigned parse_bits(std::string_view value) {
    if (value.size() == 1 && value[0] >= '2' && value[0] <= '8') return static_cast<unsigned>(value[0] - '0');
    throw std::invalid_argument("--turbo-weight-bits must be a width 2..8 (got '" + std::string(value) + "')");
}

// What this build of an engine can load. Adapters declare it; a source flips to runnable with its loader.
struct WeightCaps {
    std::array<bool, 3> source{true, false, false};  // native always
    std::array<bool, 9> bits{};                      // bits[b]: onfly / converted at b bits
    std::string_view onfly_method = "rtn-g32";       // identity of the onfly rounding (group size, scales, Hadamard)
    std::string_view onfly_element = "int";          // element family of the onfly image: int (INT4 ...) | nvfp (E2M1)
    std::string_view onfly_pending = "onfly low-bit load is not implemented in this build";
    std::string_view converted_pending = "converted low-bit pack loader is not implemented in this build";
};

struct WeightsRequest {  // explicit CLI spellings (take_options fills it)
    std::optional<std::string> source{}, bits{}, pack{};
};

inline bool has_weight_request(const WeightsRequest& request, const Env& env) {
    if (request.source || request.bits || request.pack) return true;
    for (const auto key : {kWeightsEnv, kBitsEnv, kPackEnv}) {
        const char* value = env ? env(key.data()) : nullptr;
        if (value != nullptr && *value != 0) return true;
    }
    return false;
}

struct Weights {
    Source requested = Source::native, effective = Source::native;
    unsigned requested_bits = 0, effective_bits = 0;  // 0: the pack's own widths (native)
    std::string pack{}, method{};
    std::string element = "int";                      // onfly element family (WeightCaps::onfly_element)
    std::string origin = "builtin";                   // builtin | profile | env | cli
    std::string reason{};                             // why requested != effective; empty: honoured

    [[nodiscard]] bool honoured() const { return reason.empty(); }
    // Named identity per source, element and width: "native", "onfly-int4-rtn-g32", "onfly-nvfp4-rtn-g16",
    // "converted-int4".
    [[nodiscard]] std::string identity() const {
        if (effective == Source::native) return "native";
        const std::string& e = effective == Source::onfly ? element : std::string("int");
        std::string out = std::string(source_name(effective)) + "-" + e + std::to_string(effective_bits);
        if (effective == Source::onfly && !method.empty()) out += "-" + method;
        return out;
    }
    [[nodiscard]] std::string requested_text() const {
        return std::string(source_name(requested)) + (requested_bits != 0 ? ":" + std::to_string(requested_bits) : "");
    }
    [[nodiscard]] std::string describe() const {
        std::string out = "turbo weights requested " + requested_text() + " (" + origin + ") effective " + identity();
        if (!pack.empty()) out += " pack " + pack;
        if (!reason.empty()) out += " refused: " + reason;
        return out + "\n";
    }
    void require_honoured() const {
        if (!honoured()) throw std::invalid_argument("turbo weights " + requested_text() + ": " + reason);
    }
};

// One resolution of the weight toggle for both entry points (CLI and serve). `turbo`: the product profile is turbo.
inline Weights resolve_weights(bool turbo, const WeightsRequest& cli, const Env& env, const WeightCaps& caps) {
    const auto pick = [&](const std::optional<std::string>& c, std::string_view key, std::string& origin) -> std::optional<std::string> {
        if (c) { origin = "cli"; return c; }
        const char* v = env ? env(std::string(key).c_str()) : nullptr;
        if (v != nullptr && *v != '\0') { origin = "env"; return std::string(v); }
        return std::nullopt;
    };
    Weights w;
    std::string bits_origin = "builtin", pack_origin = "builtin";
    if (const auto s = pick(cli.source, kWeightsEnv, w.origin)) w.requested = parse_source(*s);
    else if (turbo) w.origin = "profile";  // the turbo column of the toggle: native
    const auto bits = pick(cli.bits, kBitsEnv, bits_origin);
    if (const auto p = pick(cli.pack, kPackEnv, pack_origin)) w.pack = *p;
    if (w.requested != Source::native) w.requested_bits = bits ? parse_bits(*bits) : 4U;
    else if (bits) w.reason = "a weight width applies to onfly or converted only";
    if (w.requested == Source::native) {
        if (!w.pack.empty() && w.reason.empty()) w.reason = "a converted pack applies to --turbo-weights converted only";
        return w;
    }
    const auto i = static_cast<unsigned>(w.requested);
    if (!turbo) w.reason = "--turbo-weights " + std::string(source_name(w.requested)) + " requires --profile turbo";
    else if (!caps.source[i])
        w.reason = "pending: " + std::string(w.requested == Source::onfly ? caps.onfly_pending : caps.converted_pending);
    else if (w.requested_bits >= caps.bits.size() || !caps.bits[w.requested_bits]) {
        std::string list;
        for (unsigned b = 0; b < caps.bits.size(); ++b) if (caps.bits[b]) list += (list.empty() ? "" : ",") + std::to_string(b);
        w.reason = "incompatible: " + std::to_string(w.requested_bits) + "-bit weights are not supported (supported: " +
            (list.empty() ? std::string("none") : list) + ")";
    } else if (w.requested == Source::converted && w.pack.empty())
        w.reason = "incompatible: converted needs --turbo-weights-pack PATH";
    if (!w.reason.empty()) { w.pack.clear(); return w; }  // effective stays native; require_honoured() refuses
    w.effective = w.requested;
    w.effective_bits = w.requested_bits;
    if (w.effective == Source::onfly) {
        w.method = std::string(caps.onfly_method);
        w.element = std::string(caps.onfly_element);
    }
    return w;
}

// Effective facts an engine reports beside its levers in --print-effective-config under turbo. A fact the adapter
// does not report prints "unreported", so a gap is visible rather than assumed.
inline constexpr std::array<std::string_view, 7> kFactKeys{"precision", "layers", "k_parts", "chunks", "acceptance",
                                                           "hardware", "backend"};
using Facts = std::vector<std::pair<std::string, std::string>>;
inline std::string describe_facts(const Facts& facts) {
    std::string out;
    for (const auto key : kFactKeys) {
        std::string value = "unreported";
        for (const auto& [k, v] : facts) if (k == key) value = v;
        out += "turbo fact " + std::string(key) + " " + value + "\n";
    }
    return out;
}

// Prefix / session cache identity. A state is reusable only under the same key; any differing field is a miss.
// The drafter's caches use the same key with role=drafter, so they never mix with the target's.
enum class CacheRole : unsigned { target = 0, drafter = 1 };
struct CacheKey {
    std::string model{};         // model id
    std::string weights_sha{};   // content digest of the weights the run reads (the converted pack's own digest)
    std::string profile{};       // "default" | "fast" | "quality" | the versioned turbo name ("ds41-turbo-v1")
    std::string levers{};        // effective lever values (hex of the profile fingerprint)
    std::string weights{};       // Weights::identity()
    std::string order{};         // reduction order and chunk plan
    std::string state{};         // KV / recurrent state format
    std::string rotary{};        // rotary descriptor
    std::string backend{};       // backend and lineage
    CacheRole role = CacheRole::target;

    [[nodiscard]] std::string text() const {
        std::string out = "strix-cache-key/v2;";
        const std::pair<const char*, const std::string*> fields[] = {
            {"model", &model}, {"weights_sha", &weights_sha}, {"profile", &profile}, {"levers", &levers},
            {"weights", &weights}, {"order", &order}, {"state", &state}, {"rotary", &rotary}, {"backend", &backend}};
        for (const auto& [k, v] : fields)
            out += std::string(k) + "=" + std::to_string(v->size()) + ":" + *v + ";";
        return out + "role=" + (role == CacheRole::target ? "target" : "drafter") + ";";
    }
    [[nodiscard]] std::uint64_t id() const { return host_fnv::field(host_fnv::offset64, text()); }
    // Unknown fields are never a proof of compatibility: an incomplete key must not admit reuse.
    [[nodiscard]] bool complete() const {
        for (const auto* v : {&model, &weights_sha, &profile, &levers, &weights, &order, &state, &rotary, &backend})
            if (v->empty()) return false;
        return true;
    }
    [[nodiscard]] CacheKey drafter() const { CacheKey k = *this; k.role = CacheRole::drafter; return k; }
    bool operator==(const CacheKey&) const = default;
};
inline bool may_reuse(const CacheKey& stored, const CacheKey& wanted) {
    return stored.complete() && wanted.complete() && stored == wanted;
}

// Namespace text of a TURBO run that the engines' existing cache-key builders append (prefix, session): versioned
// name, effective levers and weight identity. Always non-empty under turbo, even with every lever off, so a TURBO
// state never aliases the default, fast or quality namespaces.
inline std::string cache_namespace(std::string_view versioned, std::uint64_t levers, std::string_view weights) {
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(levers));
    return "turbo=" + std::string(versioned) + ",levers=" + hex + ",weights=" + std::string(weights);
}
// The namespace the profile exported in this process (empty outside turbo). Cache-key builders call this one accessor
// instead of reading the variable themselves. Read at startup, after the profile's apply(), before any thread.
inline std::string exported_namespace(const Env& env = {}) {
    const std::string key(kCacheNamespaceEnv);
    if (env) { const char* v = env(key.c_str()); return v != nullptr ? v : ""; }
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    const char* v = std::getenv(key.c_str());
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    return v != nullptr ? v : "";
}
// The drafter's namespace of the same run (DSpark / MTP / prompt-lookup states).
inline std::string drafter_namespace(std::string_view target_namespace) {
    return target_namespace.empty() ? std::string() : std::string(target_namespace) + ",role=drafter";
}

}  // namespace strix::profile::turbo
