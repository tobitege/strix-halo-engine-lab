#pragma once
// GLM RTX product profiles: one switch, --profile default|fast|quality (STRIX_GLM_RTX_PROFILE), selects every lever
// of the GLM RTX decode service (glm_native_service --decode-commands / --enable-generate) and its harness. Structure
// and naming mirror the DS RTX resolver (strix/ds41_rtx_profile.hpp); this table holds GLM RTX rows only.
//   default  every exact lever ON: the tierstream expert cache, DS-style placement (residency seed / routes census /
//            round-robin), SCORED_AUTO eviction, 5% arena spares, the planner's row tile, CUDA graphs. No --profile =
//            default = today's configuration, byte for byte. Predicted prefetch is exact by construction but is listed
//            as pending until a PRO window proves it (the untimed q1024 compare with --tier-predict 1).
//   fast     default + every bit-changing lever that has passed qualification. Today none has: the fast-only rows (a larger
//            prefill tile, the DSA MMA index variants) are pending (listed in the receipt and on stderr, never used).
//   quality  = default for GLM on the RTX (no higher-precision expert pack exists), stated in the receipt.
// One table row per lever (profile.cpp): a value per profile, ready (qualified and present) and runnable (the
// implementation is in this build) per profile, kind exact | bit-changing, the CLI options that override it, and the
// source (owner commit, or what a pending column waits for). Precedence: explicit CLI > explicit environment (the
// lever's own variable) > profile > builtin. A pending column never refuses the run: the builtin form runs and the row is
// listed as pending; --allow-pending (STRIX_GLM_RTX_PROFILE_ALLOW_PENDING=1) runs a pending column whose implementation
// is in this build, for benchmarks only, and stamps the receipt "unqualified: <rows>". GLM RTX has no --baseline switch,
// so the DS baseline statuses do not occur here.
#include <array>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>
namespace strix::glm::rtxpf::profile {

enum class Name : unsigned { standard = 0, fast = 1, quality = 2 };  // `standard` prints as "default"
enum class App : unsigned { decode_service = 1 };
enum class Kind { exact, bit_changing };

struct Option {
    std::string_view flag;   // e.g. "--tier-policy"
    std::string_view value;  // what the flag sets the lever to; "*" = the option's argument
};

struct Lever {
    std::string_view name;                  // receipt key
    std::string_view env;                   // the lever's own environment switch (empty: none)
    std::array<Option, 4> options{};        // CLI options that set the lever explicitly (unused entries empty)
    std::array<std::string_view, 3> value;  // per profile (default, fast, quality); empty: the profile leaves it alone
    std::string_view builtin;               // this build with no profile, option or variable
    std::array<bool, 3> ready{};            // qualified and present; false: pending
    std::array<bool, 3> runnable{};         // a pending column's implementation exists in this build
    Kind kind = Kind::exact;
    unsigned apps = 0;                      // mask of App values the lever concerns
    std::string_view source;                // owner commit; for a pending column, what it waits for
};
[[nodiscard]] std::span<const Lever> levers();

enum class Status { profile, explicit_cli, explicit_env, pending, unqualified };
[[nodiscard]] const char* status_name(Status status);
[[nodiscard]] const char* kind_name(Kind kind);

struct Entry {
    const Lever* lever = nullptr;
    std::string value;   // the value the resolution asks for ("on", "off", an option's argument, ...)
    Status status = Status::profile;
    std::string wanted;  // pending / unqualified: the profile's value
};

struct Resolution {
    Name name = Name::standard;
    App app = App::decode_service;
    bool allow_pending = false;
    std::vector<Entry> entries;
    [[nodiscard]] const Entry* find(std::string_view lever) const;
    // "qualified", or "unqualified: <rows>" when --allow-pending ran pending columns
    [[nodiscard]] std::string qualification() const;
    [[nodiscard]] std::vector<std::string> pending() const;
    // The stderr lines of a run: one per pending row, one for an unqualified run, one for quality = default.
    [[nodiscard]] std::vector<std::string> notices() const;
    // {"profile","app","qualification","allow_pending","note","levers":{name:{value,status,switch,kind,source[,wanted]
    //  [,effective]}},"pending":[...],"unqualified":[...]}; `effective` (optional) gives the service's resolved value of
    // a lever ("" = omit). The same object is --print-effective-config's output and the first receipt event's "profile".
    [[nodiscard]] std::string json(const std::function<std::string(std::string_view)>& effective = {}) const;
};

[[nodiscard]] Name parse(std::string_view value);  // default|fast|quality, else throws std::invalid_argument
[[nodiscard]] const char* name(Name profile);
[[nodiscard]] const char* app_name(App app);

using Lookup = std::function<const char*(const char*)>;
struct Inputs {
    std::vector<std::string> args;  // the service options (argv[1..])
    Lookup env;                     // environment lookup; empty: the process environment
};
// --profile NAME (else STRIX_GLM_RTX_PROFILE, else default) and --allow-pending (or STRIX_GLM_RTX_PROFILE_ALLOW_PENDING=1)
// out of the options; throws std::invalid_argument on a bad name or a missing value.
struct Selection {
    Name name = Name::standard;
    bool allow_pending = false;
};
[[nodiscard]] Selection select(const Inputs& inputs);
// Throws only through select(); a pending column never refuses the run (it is refused as a lever and listed).
[[nodiscard]] Resolution resolve(App app, const Inputs& inputs, std::span<const Lever> table = levers());
[[nodiscard]] Resolution resolve(Name profile, bool allow_pending, App app, const Inputs& inputs,
                                 std::span<const Lever> table = levers());

}  // namespace strix::glm::rtxpf::profile
