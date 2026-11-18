#pragma once
#include "strix/engine_profile.hpp"
#include "strix/onfly_i4.hpp"
#include "strix/prefill_identity.hpp"
#include <array>
#include <cstdlib>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "strix/serve/json.hpp"

namespace strix::qwen::profile {
using ::strix::profile::Name;
using ::strix::profile::Kind;
using ::strix::profile::Status;
using ::strix::profile::Lever;
using ::strix::profile::Entry;
using ::strix::profile::name;
using ::strix::profile::kind_name;
using ::strix::profile::status_name;
#include "strix/qwen_profile_table.hpp"
inline std::span<const Lever> levers() { return table; }
inline Name parse(std::string_view value) {
    if (value == "exact") return Name::standard;
    return ::strix::profile::parse(value);
}  // default|fast|quality|turbo
inline constexpr std::string_view kTurboVersion = "qwen-turbo-v1";
// Weight sources this build loads under turbo (onfly / converted flip to runnable with their loaders).
inline constexpr ::strix::profile::turbo::WeightCaps kWeightCaps{
    {true, true, false}, {false, false, false, false, true, false, false, false, false}, "rtn-g64",
    "onfly INT4 g64 is built for decode dense projections only (qwen)", "converted low-bit pack loader is not implemented in this build"};
struct Resolution {
    Name profile = Name::standard;
    std::vector<Entry> entries;
    ::strix::profile::turbo::Weights weights{};  // weight toggle (native outside turbo)
    ::strix::onfly_i4::Resolved onfly{};
    bool is_turbo() const { return profile == Name::turbo; }
    std::string versioned() const { return is_turbo() ? std::string(kTurboVersion) : std::string(name(profile)); }
    // Effective lever values (not how they were set), as in the shared strix::profile::Resolution::fingerprint().
    std::uint64_t fingerprint() const {
        auto h = ::strix::host_fnv::field(::strix::host_fnv::offset64, versioned());
        h = ::strix::host_fnv::field(h, qualification());
        h = ::strix::host_fnv::field(h, weights.identity());
        for (const auto& entry : entries) {
            h = ::strix::host_fnv::field(h, entry.lever->name);
            h = ::strix::host_fnv::field(h, entry.value);
        }
        return h;
    }
    // TURBO cache namespace: the profile, not only its switches, names the cache space (every lever off included).
    std::string cache_namespace() const {
        return is_turbo() ? ::strix::profile::turbo::cache_namespace(kTurboVersion, fingerprint(), weights.identity()) : std::string();
    }
    const Entry* find(std::string_view key) const {
        for (const auto& entry : entries) if (entry.lever->name == key) return &entry;
        return nullptr;
    }
    std::string qualification() const {
        std::string provisional, unqualified;
        for (const auto& entry : entries) {
            auto* rows = entry.status == Status::provisional ? &provisional :
                entry.status == Status::unqualified ? &unqualified : nullptr;
            if (rows) *rows += (rows->empty() ? "" : ",") + std::string(entry.lever->name);
        }
        std::string out;
        if (!provisional.empty()) out = "qualification in progress: " + provisional;
        if (!unqualified.empty()) out += (out.empty() ? "" : "; ") + std::string("unqualified: ") + unqualified;
        return out.empty() ? "qualified" : out;
    }
    std::vector<std::string> notices() const {
        std::vector<std::string> out;
        if (profile == Name::quality) out.emplace_back("quality: no quality rows for this model, using default");
        if (!weights.honoured()) out.push_back("profile turbo: " + weights.describe().substr(0, weights.describe().size() - 1U));
        for (const auto& entry : entries) if (entry.status == Status::provisional)
            out.push_back(std::string("profile ") + name(profile) + ": " + std::string(entry.lever->name) +
                " is bit-changing; qualification in progress");
        for (const auto& entry : entries) if (entry.status == Status::pending)
            out.push_back(std::string("profile ") + name(profile) + ": " + std::string(entry.lever->name) +
                " pending " + entry.target + " (" + entry.value + " runs)" +
                (entry.reason.empty() ? "" : ": " + entry.reason));
        return out;
    }
    std::string json() const {
        auto out = serve::Json::object(), rows = serve::Json::object();
        auto pending = serve::Json::array(), provisional = serve::Json::array(), unqualified = serve::Json::array();
        out.set("profile", serve::Json(name(profile)));
        out.set("qualification", serve::Json(qualification()));
        if (profile == Name::quality) out.set("note", serve::Json("quality: no quality rows for this model, using default"));
        for (const auto& entry : entries) {
            const auto& lever = *entry.lever;
            auto row = serve::Json::object();
            row.set("value", serve::Json(entry.value));
            row.set("status", serve::Json(status_name(entry.status)));
            row.set("switch", serve::Json(std::string(lever.env.empty() ? lever.option : lever.env)));
            row.set("kind", serve::Json(kind_name(lever.kind)));
            row.set("source", serve::Json(std::string(lever.source)));
            if (!entry.reason.empty()) row.set("reason", serve::Json(entry.reason));
            rows.set(std::string(lever.name), std::move(row));
            if (entry.status == Status::pending) {
                auto item = serve::Json::object();
                item.set("lever", serve::Json(std::string(lever.name)));
                item.set("target", serve::Json(entry.target));
                item.set("runs", serve::Json(entry.value));
                if (!entry.reason.empty()) item.set("reason", serve::Json(entry.reason));
                pending.push(std::move(item));
            }
            if (entry.status == Status::provisional) provisional.push(serve::Json(std::string(lever.name)));
            if (entry.status == Status::unqualified) unqualified.push(serve::Json(std::string(lever.name)));
        }
        out.set("levers", std::move(rows));
        if (is_turbo()) {
            auto t = serve::Json::object();
            t.set("version", serve::Json(std::string(kTurboVersion)));
            t.set("weights_requested", serve::Json(weights.requested_text()));
            t.set("weights", serve::Json(onfly.active() ? onfly.identity() : weights.identity()));
            if (onfly.active()) t.set("onfly", serve::parse_json(onfly.json()));
            t.set("weights_origin", serve::Json(weights.origin));
            t.set("weights_reason", serve::Json(weights.reason));
            t.set("cache_namespace", serve::Json(cache_namespace()));
            out.set("turbo", std::move(t));
        }
        out.set("pending", std::move(pending));
        out.set("provisional", std::move(provisional));
        out.set("unqualified", std::move(unqualified));
        return out.dump();
    }
};
using ::strix::profile::Lookup;
using ::strix::profile::Pairs;
struct Inputs {
    Pairs cli;
    Lookup env;
    bool allow_pending = false;
    ::strix::profile::turbo::WeightsRequest weights{};  // --turbo-weights / --turbo-weight-bits / --turbo-weights-pack
    // What the calling backend loads (nullptr: kWeightCaps of the HIP engine). The CUDA engine passes kCudaWeightCaps.
    const ::strix::profile::turbo::WeightCaps* caps = nullptr;
};
// CUDA engine (sm_120): onfly 4-bit = NVFP4 images of the dense Q8_0 roles (src/cuda/qwen_prefill_driver.cu
// convert_onfly, shared core strix/cuda/core/nvfp4_a8.hpp); converted packs need the owner's OK (pending).
inline constexpr ::strix::profile::turbo::WeightCaps kCudaWeightCaps{
    {true, true, false},
    {false, false, false, false, true, false, false, false, false},
    "rtn-g16",
    "nvfp",
    "onfly low-bit load is not implemented in this build",
    "converted low-bit pack loader is not implemented in this build (needs the owner's OK)"};
using ::strix::profile::environment;
// Export provenance survives a future re-exec. A changed inherited value
// is explicit again; matching profile exports retain their original status.
inline std::string marker_key(const Lever& lever) { return "STRIX_QWEN_PROFILE_OWNED_" + std::string(lever.name); }
inline std::string marker(Name profile, Status status, std::string_view value) {
    return std::string(name(profile)) + ":" + status_name(status) + ":" + std::string(value);
}
inline Resolution resolve(Name profile, const Inputs& inputs = {}, std::span<const Lever> rows = levers()) {
    Resolution out{profile, {}, {}, {}};
    const char* allow = environment(inputs.env, "STRIX_QWEN_PROFILE_ALLOW_PENDING");
    const bool allow_pending = inputs.allow_pending || (allow && std::string_view(allow) == "1");
    out.weights = ::strix::profile::turbo::resolve_weights(out.is_turbo(), inputs.weights,
        [&inputs](const char* key) { return environment(inputs.env, key); }, inputs.caps ? *inputs.caps : kWeightCaps);
    for (const auto& lever : rows) {
        const auto column = ::strix::profile::value_column(lever, profile);  // empty turbo cell: the fast cell
        const std::string* cli = nullptr;
        for (const auto& [key, value] : inputs.cli) if (key == lever.name) cli = &value;
        const char* env = lever.env.empty() ? nullptr : environment(inputs.env, std::string(lever.env));
        const auto wanted = lever.value[column];
        bool inherited = false;
        if (env && !cli && !wanted.empty() && std::string_view(env) == wanted) {
            const char* owned = environment(inputs.env, marker_key(lever));
            const auto status = ::strix::profile::admit_column(lever.ready[column], lever.runnable[column], allow_pending,
                lever.provisional[column]);
            inherited = owned && std::string_view(owned) == marker(profile, status, env) &&
                (status != Status::pending);
        }
        Entry entry{&lever, std::string(lever.builtin), Status::builtin, {}, false, {}, std::string(wanted)};
        if (cli || (env && !inherited)) {
            entry.cli_override = cli != nullptr;
            entry.value = cli ? *cli : env;
            entry.requested = entry.value;
            entry.status = cli ? Status::explicit_cli : Status::explicit_env;
            if ((!lever.off.empty() && entry.value == lever.off) || entry.value == "off") entry.status = Status::off;
            bool runnable = false;
            for (bool available : lever.runnable) runnable |= available;
            if (!runnable && entry.status != Status::off && entry.value != lever.builtin)
                throw std::invalid_argument(std::string(lever.env.empty() ? lever.option : lever.env) + "=" + entry.value +
                    " is not in this build (" + std::string(lever.source) + ")");
        } else if (wanted.empty()) continue;
        else if (::strix::profile::admit_column(lever.ready[column], lever.runnable[column], allow_pending,
                 lever.provisional[column]) == Status::pending) {
            entry.status = Status::pending;
            entry.target = std::string(lever.target.empty() ? wanted : lever.target);
        } else {
            entry.value = std::string(wanted);
            entry.status = ::strix::profile::admit_column(lever.ready[column], lever.runnable[column], allow_pending,
                lever.provisional[column]);
            if (entry.status == Status::set && wanted == lever.builtin) entry.status = Status::builtin;
        }
        out.entries.push_back(std::move(entry));
    }
    return out;
}
using ::strix::profile::Export;
using ::strix::profile::export_environment;
using Erase = std::function<bool(const char*)>;
inline bool erase_environment(const char* key) {
    return process_environment::unset(key);
}
// A re-exec can inherit a proposal that its new route declines. Remove only
// switches whose matching marker proves profile ownership; explicit env wins.
inline void withdraw_declined(const Resolution& resolution, const Lookup& get = {}, const Erase& erase = erase_environment) {
    for (const auto& entry : resolution.entries) {
        if (entry.status != Status::pending || entry.reason.empty() || entry.lever->env.empty()) continue;
        const auto key = std::string(entry.lever->env), owned_key = marker_key(*entry.lever);
        const char* value = environment(get, key);
        const char* owned = environment(get, owned_key);
        if (!value || !owned) continue;
        bool inherited = false;
        for (const auto status : {Status::set, Status::unqualified, Status::provisional})
            inherited |= std::string_view(owned) == marker(resolution.profile, status, value);
        if (inherited && (!erase(key.c_str()) || !erase(owned_key.c_str())))
            throw std::runtime_error("cannot withdraw profile switch " + key);
    }
}
// Pure proposal for a local map, including pending weights during inspection.
inline void propose_exports(const Resolution& resolution, const Export& put) {
    if (resolution.is_turbo() && !put(std::string(::strix::profile::turbo::kCacheNamespaceEnv).c_str(),
                                      resolution.cache_namespace().c_str()))
        throw std::runtime_error("cannot export " + std::string(::strix::profile::turbo::kCacheNamespaceEnv));
    for (const auto& entry : resolution.entries) {
        const auto& lever = *entry.lever;
        // Native CLI overrides already live in Options; exporting them would
        // change the historical environment receipt even with profile=default.
        if (lever.env.empty() || (entry.status != Status::set && entry.status != Status::unqualified && entry.status != Status::provisional &&
            !(entry.cli_override && !lever.native_option))) continue;
        if (!put(std::string(lever.env).c_str(), entry.value.c_str()) ||
            !put(marker_key(lever).c_str(), marker(resolution.profile, entry.status, entry.value).c_str()))
            throw std::runtime_error("cannot export profile switch " + std::string(lever.env));
    }
}
// Admit execution after route resolution and before device switch readers.
inline void apply(const Resolution& resolution, const Export& put = export_environment) {
    resolution.weights.require_honoured();
    propose_exports(resolution, put);
}
// Profile policy has no numerical identity bytes of its own. Only applied
// arithmetic switches belong in session/prefix identities.
inline bool metadata_environment(std::string_view key) {
    return key == "STRIX_QWEN_PROFILE_ALLOW_PENDING" || key.starts_with("STRIX_QWEN_PROFILE_OWNED_");
}
inline bool policy_option(std::string_view key) {
    return key == "--profile" || key == "--allow-pending";
}
inline bool take_option(int argc, char** argv, int& i, std::optional<Name>& profile, Inputs& inputs,
        std::span<const Lever> rows = levers()) {
    const std::string_view arg = argv[i];
    const auto equal = arg.find('=');
    const auto key = arg.substr(0, equal);
    if (arg == "--allow-pending") { inputs.allow_pending = true; return true; }
    namespace tb = ::strix::profile::turbo;
    if (key == tb::kWeightsOption || key == tb::kBitsOption || key == tb::kPackOption) {
        const auto value = equal != std::string_view::npos ? arg.substr(equal + 1U) :
            (i + 1 < argc ? std::string_view(argv[++i]) : std::string_view{});
        if (value.empty() || value.starts_with("--")) throw std::invalid_argument(std::string(key) + " requires a value");
        (key == tb::kWeightsOption ? inputs.weights.source : key == tb::kBitsOption ? inputs.weights.bits :
            inputs.weights.pack) = std::string(value);
        return true;
    }
    const Lever* lever = nullptr;
    for (const auto& row : rows) if (!row.native_option && key == row.option) lever = &row;
    if (key != "--profile" && !lever) return false;
    const auto value = equal != std::string_view::npos ? arg.substr(equal + 1U) :
        (i + 1 < argc ? std::string_view(argv[++i]) : std::string_view{});
    if (value.empty() || value.starts_with("--")) throw std::invalid_argument(std::string(key) + " requires a value");
    if (key == "--profile") {
        if (profile) throw std::invalid_argument("duplicate --profile");
        profile = parse(value);
    } else inputs.cli.emplace_back(lever->name, value);
    return true;
}
// Host metadata only: callers attach verified assets, execution provenance and
// observed digests later. A compiled/native lineage must be supplied by its owner;
// the product profile and the prefill option name are not a compiled lineage.
inline ::strix::profile::ExecutionReceipt host_receipt(const NumericalProfilePair& pair,
        unsigned prefill_mask, bool gsq, std::string_view prefill_lineage = {}) {
    ::strix::profile::ExecutionReceipt receipt;
    receipt.engine = "qwen";
    receipt.identity.arithmetic_contract = "qwen;decode=" + std::string(numerical_profile_name(pair.decode)) +
        ";prefill=" + std::string(numerical_profile_name(pair.prefill)) +
        ";fast_components=" + std::to_string(pair.fast_components) +
        ";prefill_options=" + prefill_identity_name(prefill_mask) +
        ";arithmetic_identity=" + ::strix::profile::hex_identity(prefill_arithmetic_identity(pair, prefill_mask, gsq));
    receipt.identity.prefill_lineage = prefill_lineage;
    return receipt;
}
} // namespace strix::qwen::profile
