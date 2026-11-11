#pragma once

#include "strix/fnv.hpp"
#include "strix/process_environment.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Product profiles of an engine: one switch '--profile default|fast|quality' selects every lever's value.
//   default  today's exact defaults (exports nothing: the plain invocation, byte for byte)
//   fast     default + every QUALIFIED bit-changing lever of the engine (and any marked provisional, stamped so)
//   quality  a quality pack where the engine has one, else default plus one notice
// Each lever is one table row (the engine's *_profile_table.hpp). A profile column that is not qualified yet
// (`ready = false`) is pending: it is never applied and never makes the profile refuse; the builtin keeps
// running, the receipt lists the row with its declared target and one stderr line says so. '--allow-pending'
// (or the engine's ..._PROFILE_ALLOW_PENDING=1) applies a pending column whose implementation exists in this
// build (`runnable`), for benchmarks only, and stamps the receipt "unqualified: <rows>". A `provisional` column
// runs its runnable code ahead of qualification without --allow-pending, stamped "qualification in progress: <rows>"
// with one stderr line per row; it is never "qualified". Precedence per lever:
// CLI option > environment variable > profile > builtin; an explicit setting is recorded as explicit (or `off`).
// Header-only and host-only (no HIP/CUDA, no JSON library): shared by the two-node engines' apps.
namespace strix::profile {

enum class Name : unsigned { standard = 0, fast = 1, quality = 2 };  // `standard` prints as "default"
enum class Kind { exact, bit_changing, pack };

struct Lever {
    std::string_view name;                  // receipt key
    std::string_view env;                   // switch the profile exports (empty: CLI-only lever)
    std::string_view option;                // CLI option that overrides the lever (empty: none)
    std::array<std::string_view, 3> value;  // per profile (default, fast, quality); empty: left alone
    std::string_view builtin;               // behaviour of this build with the switch unset
    Kind kind = Kind::exact;
    std::array<bool, 3> ready{};            // qualified (qualification receipt) and present; false: pending
    std::array<bool, 3> runnable{};         // a pending column's implementation exists in this build
    std::string_view target;                // declared target of a pending column; empty: the value
    std::string_view off;                   // explicit value that switches the lever off
    std::string_view source;                // qualification receipt, or what a pending column waits for
    bool native_option = false;            // an adapter's native CLI parser owns validation/export
    std::array<bool, 3> provisional{};      // selected ahead of qualification; still requires `runnable`; wins over `ready` until the row is flipped
};

// One engine: its receipt tag, the allow-pending variable and the table.
struct Engine {
    std::string_view tag;                   // "deepseek41-pair", "glm-pair", "minimax-pair"
    std::string_view allow_env;             // e.g. STRIX_DS41_PROFILE_ALLOW_PENDING
    std::span<const Lever> table;
    bool quality_rows = false;              // false: 'quality' runs default with one notice
};

inline Name parse(std::string_view value) {
    if (value == "default") return Name::standard;
    if (value == "fast") return Name::fast;
    if (value == "quality") return Name::quality;
    throw std::invalid_argument("--profile must be default, fast or quality (got '" + std::string(value) + "')");
}
inline bool is_name(std::string_view value) { return value == "default" || value == "fast" || value == "quality"; }
inline const char* name(Name value) {
    switch (value) {
    case Name::standard: return "default";
    case Name::fast: return "fast";
    case Name::quality: return "quality";
    }
    return "unknown";
}

enum class Status { set, explicit_env, explicit_cli, builtin, pending, unqualified, off, provisional };
// Shared column admission; adapters retain asset, route and baseline policies.
inline Status admit_column(bool ready, bool runnable, bool allow_pending, bool provisional = false) {
    if (!ready && !((provisional || allow_pending) && runnable)) return Status::pending;
    return provisional ? Status::provisional : ready ? Status::set : Status::unqualified;
}
inline const char* status_name(Status value) {
    switch (value) {
    case Status::set: return "profile";
    case Status::explicit_env: return "explicit-env";
    case Status::explicit_cli: return "explicit-cli";
    case Status::builtin: return "builtin";
    case Status::pending: return "pending";
    case Status::unqualified: return "unqualified";
    case Status::off: return "off";
    case Status::provisional: return "provisional";
    }
    return "unknown";
}
inline const char* kind_name(Kind value) {
    switch (value) {
    case Kind::exact: return "exact";
    case Kind::bit_changing: return "bit-changing";
    case Kind::pack: return "pack";
    }
    return "unknown";
}

inline std::string json_string(std::string_view s) {
    std::string out = "\"";
    for (const char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20U) {
                char buf[8];
                std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
                out += buf;
            } else out += c;
        }
    }
    return out + "\"";
}

struct Entry {
    const Lever* lever = nullptr;
    std::string value;   // effective value (explicit, profile or builtin)
    Status status = Status::builtin;
    std::string target;  // pending: the declared target that is not applied
    bool cli_override = false;  // adapter provenance; does not enter the rank fingerprint
    std::string reason;         // adapter admission explanation
};

struct Resolution {
    const Engine* engine = nullptr;
    Name profile = Name::standard;
    bool allow_pending = false;
    std::vector<Entry> entries;

    [[nodiscard]] const Entry* find(std::string_view key) const {
        for (const auto& entry : entries) if (entry.lever->name == key) return &entry;
        return nullptr;
    }
    [[nodiscard]] bool quality_fallback() const { return profile == Name::quality && !engine->quality_rows; }
    // "qualified"; else "qualification in progress: <rows>" for provisional columns and/or "unqualified: <rows>"
    // when --allow-pending ran pending columns, separated by "; ".
    [[nodiscard]] std::string qualification() const {
        std::string provisional, unqualified;
        for (const auto& entry : entries) {
            auto* rows = entry.status == Status::provisional ? &provisional :
                entry.status == Status::unqualified ? &unqualified : nullptr;
            if (rows != nullptr) *rows += (rows->empty() ? "" : ",") + std::string(entry.lever->name);
        }
        std::string out;
        if (!provisional.empty()) out = "qualification in progress: " + provisional;
        if (!unqualified.empty()) out += (out.empty() ? "" : "; ") + std::string("unqualified: ") + unqualified;
        return out.empty() ? "qualified" : out;
    }
    // One stderr line per pending, provisional or unqualified row (and the quality fallback note).
    [[nodiscard]] std::vector<std::string> notices() const {
        std::vector<std::string> out;
        const std::string head = std::string(engine->tag) + " profile " + name(profile) + ": ";
        if (quality_fallback()) out.push_back(head + "no quality rows for this model, using default");
        for (const auto& entry : entries) if (entry.status == Status::provisional)
            out.push_back(head + std::string(entry.lever->name) + "=" + entry.value + " is bit-changing; qualification in progress");
        for (const auto& entry : entries) if (entry.status == Status::pending)
            out.push_back(head + std::string(entry.lever->name) + " pending " + entry.target + " (" + entry.value + " runs)");
        for (const auto& entry : entries) if (entry.status == Status::unqualified)
            out.push_back(head + std::string(entry.lever->name) + "=" + entry.value + " UNQUALIFIED (--allow-pending, benchmarks only)");
        return out;
    }
    // {"profile":...,"engine":...,"qualification":...,"levers":{...},"pending":[...],"unqualified":[...],"provisional":[...]}
    [[nodiscard]] std::string json() const {
        std::string levers, pending, unqualified, provisional;
        for (const auto& entry : entries) {
            const auto& lever = *entry.lever;
            if (!levers.empty()) levers += ',';
            levers += json_string(lever.name) + ":{\"value\":" + json_string(entry.value) +
                ",\"status\":" + json_string(status_name(entry.status)) +
                ",\"switch\":" + json_string(lever.env.empty() ? lever.option : lever.env) +
                ",\"kind\":" + json_string(kind_name(lever.kind)) +
                ",\"source\":" + json_string(lever.source) + "}";
            if (entry.status == Status::pending) {
                if (!pending.empty()) pending += ',';
                pending += "{\"lever\":" + json_string(lever.name) + ",\"target\":" + json_string(entry.target) +
                    ",\"runs\":" + json_string(entry.value) + "}";
            }
            if (entry.status == Status::unqualified) {
                if (!unqualified.empty()) unqualified += ',';
                unqualified += json_string(lever.name);
            }
            if (entry.status == Status::provisional) {
                if (!provisional.empty()) provisional += ',';
                provisional += json_string(lever.name);
            }
        }
        std::string out = "{\"profile\":" + json_string(name(profile)) + ",\"engine\":" + json_string(engine->tag) +
            ",\"qualification\":" + json_string(qualification());
        if (quality_fallback()) out += ",\"note\":\"quality: no quality rows for this model, using default\"";
        return out + ",\"levers\":{" + levers + "},\"pending\":[" + pending + "],\"unqualified\":[" + unqualified +
            "],\"provisional\":[" + provisional + "]}";
    }
    // Line form for line-oriented --print-effective-config outputs (appended after the engine's own lines).
    [[nodiscard]] std::string describe() const {
        std::string out = std::string("profile ") + name(profile) + "\nprofile qualification " + qualification() + "\n";
        if (quality_fallback()) out += "profile note quality: no quality rows for this model, using default\n";
        for (const auto& entry : entries) {
            const auto& lever = *entry.lever;
            out += "profile lever " + std::string(lever.name) + " " + entry.value + " (" + status_name(entry.status) + ", " +
                kind_name(lever.kind) + ", " + std::string(lever.env.empty() ? lever.option : lever.env) + ")";
            if (entry.status == Status::pending) out += " pending " + entry.target;
            out += "\n";
        }
        return out;
    }
    // FNV-1a 64 of what the ranks of a pair must share: the profile name, the qualification and every lever's
    // effective value (not how it was set: an explicit value equal to the profile's is the same run).
    [[nodiscard]] std::uint64_t fingerprint() const {
        std::uint64_t h = 1469598103934665603ULL;
        const auto mix = [&h](std::string_view s) {
            h = host_fnv::byte(host_fnv::text(s, h), 0xFFU);  // preserve legacy seed and separator
        };
        mix(name(profile));
        mix(qualification());
        for (const auto& entry : entries) { mix(entry.lever->name); mix(entry.value); }
        return h;
    }
};

// Rank agreement: both ranks of a pair must run the same resolved profile (a bit-changing row on one rank only
// desynchronizes the exchanges). `exchange` swaps 32 host bytes with the peer (the app's control channel, both ranks
// at the same point of the startup); a mismatch throws, naming both profiles.
using HostExchange = std::function<void(std::span<const std::byte>, std::span<std::byte>)>;
inline void check_peer(const Resolution& resolution, const HostExchange& exchange) {
    struct Hello {
        std::uint32_t magic = 0x31465250U;  // "PRF1"
        std::uint32_t reserved = 0;
        std::uint64_t fingerprint = 0;
        char profile[16] = {};
    };
    static_assert(sizeof(Hello) == 32U);
    Hello mine, peer;
    mine.fingerprint = resolution.fingerprint();
    std::snprintf(mine.profile, sizeof mine.profile, "%s", name(resolution.profile));
    std::array<std::byte, sizeof(Hello)> out{}, in{};
    std::memcpy(out.data(), &mine, sizeof mine);
    exchange(out, in);
    std::memcpy(&peer, in.data(), sizeof peer);
    peer.profile[sizeof peer.profile - 1U] = '\0';
    if (peer.magic != mine.magic)
        throw std::runtime_error("product profile check: the peer did not send its profile (different engine build?)");
    if (peer.fingerprint != mine.fingerprint) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%016llx vs %016llx", static_cast<unsigned long long>(mine.fingerprint),
                      static_cast<unsigned long long>(peer.fingerprint));
        throw std::runtime_error(std::string("product profile differs between the ranks: this rank '") + name(resolution.profile) +
            "' (" + resolution.qualification() + "), peer '" + peer.profile + "' (fingerprint " + buf +
            "); run both ranks with the same --profile, --allow-pending and lever switches");
    }
}

using Lookup = std::function<const char*(const char*)>;
inline const char* environment(const Lookup& env, std::string_view key);

// Shared defaults row; adapters retain application-specific admission and exports.
struct DefaultItem {
    const char* name;
    const char* env;
    const char* on;
    const char* off;
    const char* receipt;
    bool app_sets = true;
    bool in_baseline = true;
};
enum class DefaultSource { on, baseline, env, cli };
struct DefaultValue {
    std::string value;
    DefaultSource source = DefaultSource::on;
};
inline DefaultValue resolve_default(const DefaultItem& row, bool baseline, const Lookup& env,
                                    std::optional<std::string_view> cli = std::nullopt) {
    if (cli) return {std::string(*cli), DefaultSource::cli};
    if (const char* value = environment(env, row.env))
        return {value, DefaultSource::env};
    if (baseline && row.in_baseline) return {row.off, DefaultSource::baseline};
    return {row.on, DefaultSource::on};
}
using Pairs = std::vector<std::pair<std::string, std::string>>;
struct Inputs {
    std::optional<Name> profile;  // --profile; none: default
    Pairs cli;                    // levers set by their own CLI option: {lever name, value}
    Lookup env;                   // environment lookup; empty: the process environment
    bool allow_pending = false;   // --allow-pending
};

inline const char* environment(const Lookup& env, std::string_view key) {
    const std::string k(key);
    if (env) return env(k.c_str());
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv: read once at startup, before any thread
#endif
    return std::getenv(k.c_str());
#ifdef _MSC_VER
#pragma warning(pop)
#endif
}

inline Resolution resolve(const Engine& engine, const Inputs& inputs) {
    Resolution out;
    out.engine = &engine;
    out.profile = inputs.profile.value_or(Name::standard);
    const char* allow = engine.allow_env.empty() ? nullptr : environment(inputs.env, engine.allow_env);
    out.allow_pending = inputs.allow_pending || (allow != nullptr && std::string_view(allow) == "1");
    // 'quality' without quality rows is the default column.
    const auto column = static_cast<unsigned>(out.quality_fallback() ? Name::standard : out.profile);
    for (const auto& lever : engine.table) {
        const std::string* cli = nullptr;
        for (const auto& [key, value] : inputs.cli) if (key == lever.name) cli = &value;
        const char* env = lever.env.empty() ? nullptr : environment(inputs.env, lever.env);
        const auto wanted = lever.value[column];
        Entry entry{&lever, std::string(lever.builtin), Status::builtin, {}, false, {}};
        if (cli != nullptr || env != nullptr) {  // a set variable is explicit even when empty (the app judges it)
            entry.value = cli != nullptr ? *cli : std::string(env);
            entry.status = cli != nullptr ? Status::explicit_cli : Status::explicit_env;
            if ((!lever.off.empty() && entry.value == lever.off) || entry.value == "off") entry.status = Status::off;
        } else if (wanted.empty() || wanted == lever.builtin) {
            // builtin (the default column always is: default exports nothing)
        } else if (admit_column(lever.ready[column], lever.runnable[column], out.allow_pending,
                                lever.provisional[column]) == Status::pending) {
            entry.status = Status::pending;
            entry.target = std::string(lever.target.empty() ? wanted : lever.target);
        } else {
            entry.value = std::string(wanted);
            entry.status = admit_column(lever.ready[column], lever.runnable[column], out.allow_pending,
                                         lever.provisional[column]);
        }
        out.entries.push_back(std::move(entry));
    }
    return out;
}

// The statuses the profile itself applies (an explicit setting is the app's own).
inline bool applied(Status status) {
    return status == Status::set || status == Status::unqualified || status == Status::provisional;
}

using Export = std::function<bool(const char*, const char*)>;
inline bool export_environment(const char* key, const char* value) {
    return process_environment::set(key, value, true);
}
// Exports the switches the profile sets (status set / unqualified / provisional, env levers only; CLI-only levers:
// profile_args).
// Call right after argument parsing, before anything reads (and caches) those switches. Default exports nothing.
inline void apply(const Resolution& resolution, const Export& put = export_environment) {
    for (const auto& entry : resolution.entries) {
        const auto& lever = *entry.lever;
        if (lever.env.empty() || !applied(entry.status)) continue;
        if (!put(std::string(lever.env).c_str(), entry.value.c_str()))
            throw std::runtime_error("cannot export profile switch " + std::string(lever.env));
    }
}

// Options the profile adds to the app's arguments for CLI-only levers it sets ({option, value} each; put them
// in front of the user's arguments). Default adds none.
inline std::vector<std::string> profile_args(const Resolution& resolution) {
    std::vector<std::string> out;
    for (const auto& entry : resolution.entries) {
        const auto& lever = *entry.lever;
        if (!lever.env.empty() || lever.option.empty() || !applied(entry.status)) continue;
        out.emplace_back(lever.option);
        out.push_back(entry.value);
    }
    return out;
}

// Removes '--profile default|fast|quality' ('--profile=NAME') and '--allow-pending' from `args` into `inputs` and records
// (without removing) the table's lever options as explicit CLI values. Any other '--profile' token ('--profile' followed
// by something that is not a profile name, '--profile=OTHER') stays in `args` untouched, for the app's own parser: an
// older meaning (GLM kernel timing, MiniMax timed session) or a value of another option keeps working. Only a second
// profile name is an error.
inline void take_options(const Engine& engine, std::vector<std::string>& args, Inputs& inputs) {
    std::vector<std::string> kept;
    kept.reserve(args.size());
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        const auto equal = a.find('=');
        const std::string key = a.substr(0, equal);
        if (a == "--allow-pending") { inputs.allow_pending = true; continue; }
        if (key == "--profile") {
            std::string value;
            if (equal != std::string::npos) value = a.substr(equal + 1U);
            else if (i + 1 < args.size()) value = args[i + 1];
            // '--profile WORD' with a plain word that is not a profile name is a typo ('--profile fastt'); a path, a
            // NAME:... session, an option ('--profile --x') or a '--profile=...' token belongs to the app.
            if (equal == std::string::npos && !value.empty() && !is_name(value) && value.find_first_of("-:/.=") == std::string::npos)
                throw std::invalid_argument("--profile must be default, fast or quality (got '" + value + "')");
            if (is_name(value)) {
                if (inputs.profile) throw std::invalid_argument("duplicate --profile");
                inputs.profile = parse(value);
                if (equal == std::string::npos) ++i;
                continue;
            }
        }
        kept.push_back(a);
        const Lever* lever = nullptr;
        for (const auto& row : engine.table) if (!row.option.empty() && key == row.option) lever = &row;
        if (lever == nullptr) continue;
        // The app parses its own option too (it stays in args); the last spelling wins, as in the app's parser.
        std::string value;
        if (equal != std::string::npos) value = a.substr(equal + 1U);
        else if (i + 1 < args.size()) kept.push_back(value = args[++i]);
        inputs.cli.emplace_back(std::string(lever->name), value);
    }
    args = std::move(kept);
}

inline std::string hex_identity(std::uint64_t value) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(value));
    return buf;
}

// Numerical compatibility and execution provenance are deliberately independent.
// Empty fields mean unknown, never a proof that two executions are compatible.
struct NumericalIdentity {
    std::string arithmetic_contract;
    std::string model_sha256;
    std::string pack_hash;
    std::string table_hash;
    std::string state_format;
    std::string prefill_lineage;
    std::string reduction_plan;

    [[nodiscard]] std::uint64_t id() const {
        auto h = host_fnv::field(host_fnv::offset64, "strix-numerical-identity/v1");
        for (const auto* value : {&arithmetic_contract, &model_sha256, &pack_hash, &table_hash,
                                  &state_format, &prefill_lineage, &reduction_plan})
            h = host_fnv::field(h, *value);
        return h;
    }
    [[nodiscard]] std::string json() const {
        return "{\"id\":" + json_string(hex_identity(id())) +
            ",\"arithmetic_contract\":" + json_string(arithmetic_contract) +
            ",\"model_sha256\":" + json_string(model_sha256) +
            ",\"pack_hash\":" + json_string(pack_hash) +
            ",\"table_hash\":" + json_string(table_hash) +
            ",\"state_format\":" + json_string(state_format) +
            ",\"prefill_lineage\":" + json_string(prefill_lineage) +
            ",\"reduction_plan\":" + json_string(reduction_plan) + "}";
    }
};

struct ExecutionReceipt {
    std::string engine;
    NumericalIdentity identity;
    std::string implementation;
    std::string build;
    std::string kernel;
    std::string device;
    std::string topology;
    std::string command;  // canonical invocation when supplied by the harness
    // Ordered named digests; duplicate names are rejected rather than ambiguous JSON.
    std::vector<std::pair<std::string, std::string>> digests;

    [[nodiscard]] std::string json() const {
        std::string hashes;
        for (std::size_t i = 0; i < digests.size(); ++i) {
            for (std::size_t j = 0; j < i; ++j)
                if (digests[j].first == digests[i].first)
                    throw std::invalid_argument("duplicate receipt digest: " + digests[i].first);
            if (!hashes.empty()) hashes += ',';
            hashes += json_string(digests[i].first) + ':' + json_string(digests[i].second);
        }
        return "{\"schema\":\"strix-engine-receipt/v1\",\"engine\":" + json_string(engine) +
            ",\"identity\":" + identity.json() +
            ",\"implementation\":" + json_string(implementation) +
            ",\"build\":" + json_string(build) + ",\"kernel\":" + json_string(kernel) +
            ",\"device\":" + json_string(device) + ",\"topology\":" + json_string(topology) +
            ",\"command\":" + json_string(command) +
            ",\"digests\":{" + hashes + "}}";
    }
    [[nodiscard]] std::string witness() const { return "strix-engine-witness " + json() + "\n"; }
};

}  // namespace strix::profile
