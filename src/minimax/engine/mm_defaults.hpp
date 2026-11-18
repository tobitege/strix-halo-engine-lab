#pragma once
#include <strix/weight/family.hpp>
// MiniMax M3.1 defaults-on resolver (optimizations are ON by default in code and are
// switched OFF, never the other way round). Host only, no HIP: the apps and the engine call apply() before they read
// any switch, and the CPU tests diff the explicit host configuration against the measured production receipt,
// with the new routed serve transport default checked separately.
//
// Optimizations and admission policies resolve ON unless their env names the off value.
// Unqualified admission cells select plain; the item receipt distinguishes policy from measured kernels.
// STRIX_MM_BASELINE=1 is the master switch: every item resolves to its off value (the path before the item); an
// explicit per-item value still wins in both modes. apply() exports each unset item's resolved value into the process
// environment, so the readers in the engine, the exchange modules and the dist layer see exactly what
// --print-effective-config prints.
#include "strix/pair/regime.hpp"
#include <cstdint>
#include <cstddef>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <vector>
#include "strix/engine_profile.hpp"
#include "xport/split_frame.hpp"
#include "xport/exchange_format.hpp"


namespace strix::minimax::defaults {

using Item = profile::DefaultItem;

inline constexpr const char* kBaselineEnv = "STRIX_MM_BASELINE";

std::span<const Item> items();

// Split-producer chunk frames (xport/split_frame.hpp): the lever row and its policy, resolved ONCE per process from
// STRIX_XPORT_SPLIT_ROWS (CLI and serve call it from print_or_apply, the engine reads the same cached value).
// Invalid text throws std::invalid_argument naming the variable. Not an Item: it has no on/off default value.
const lever::Decl& split_frame_row();
xport::SplitFramePolicy split_frame_policy();

// Startup parameter, independent of the baseline switch (async start itself is
// switched off by baseline). Zero restores unconditional grouped async start.
std::size_t pf_async_min_rows(const char* value);
inline bool pf_async_group(bool enabled, std::span<const std::size_t> rows, std::size_t minimum) {
    if (!enabled || rows.empty()) return false;
    for (const auto n : rows) if (n < minimum) return false;
    return true;
}

using Source = profile::DefaultSource;
struct Resolved {
    const Item* item = nullptr;
    std::string value;
    Source source = Source::on;
};

using Lookup = profile::Lookup;

bool baseline(const Lookup& env);                 // STRIX_MM_BASELINE set and not 0
std::vector<Resolved> resolve(const Lookup& env);  // pure: same order as items()

// FAST prefill profile (default-OFF governed rows, mm_profile_table.hpp). The one resolver of the bundle
// STRIX_MM_FAST_PREFILL and its members: the engine (prefill MSA attention), the session-cache identity, the effective
// config and the rank agreement all call it. Invalid member values throw std::invalid_argument naming the variable.
struct PrefillPolicy {
    std::map<std::string, std::string> canonical;  // expanded bundle switch and members (strix/fast_profile.hpp)
    unsigned msa_flash = 0;                        // 0 exact in-block kernel, 1 one-term BF16, 2 two-term operands
    unsigned msa_flash_requested = 0;              // selected before static execution guards
    std::string msa_flash_layers;                  // layer set text the engine parses (builtin kMsaFlashLayers)
    std::string msa_flash_layers_requested;        // original layer syntax, for provenance only
    std::string msa_flash_asked;                   // "default", the explicit value, or "fast_prefill" (bundle)
    bool idx_wmma = false;                         // index projections on the shared FP16 WMMA core (prefill rows > 8)
    std::string idx_wmma_layers;                   // canonical ranges of its layer set (builtin kIdxWmmaLayers)
    std::string idx_wmma_asked;                    // "default" or the explicit value
    bool router_wmma = false;                      // STRIX_MM_PF_ROUTER_WMMA=1: FAST router product (mm-pf-router-wmma-v1)
    std::string router_wmma_layers;                // its layer set (builtin kRouterWmmaLayers), canonical ranges
    std::string policies;                          // named FAST policies that run, '+'-joined; empty: none
    std::string note;                              // why a bundle member does not run; empty: none
    // STRIX_TP2_EXCHANGE_FMT (shared trait, xport/exchange_format.hpp): an explicit f16 = named FAST policy
    // tp2-xchg-f16-v0 (throws under STRIX_MM_BASELINE=1); default f32 in every profile (the FAST bundle does not select it).
    strix::xport::ExchangeFormat exchange_fmt = strix::xport::ExchangeFormat::f32;
    std::string exchange_fmt_reason;
};
// Static execution guards, also used by the engine receipt after runtime control changes.
const char* msa_flash_decline_reason(bool rework, bool fused, unsigned layers);
PrefillPolicy prefill_policy(const Lookup& env);
// Reference estimate using the shared pair rule and 8K FAST costs, extrapolated to rows. Assumes +41.6 GB/node
// for whole stage layers and 31 GB free; neither is a runtime measurement or pipeline qualification.
// Execution stays TP2: slices are fixed at load (engine.hip:89-91,1120-1131), with no pipeline implementation.
// The returned choice is diagnostic only and must never select the execution route.
strix::pair::PipelineChoice pair_pipeline_choice(std::uint64_t rows);
const profile::Engine& profile_engine();                 // tag "minimax-pair", table profile_table
profile::Resolution profile_resolution(const Lookup& env);
// Product profile (default|fast|quality|turbo) and TURBO weight toggle of strix-minimax-rank and strix-minimax-serve:
// print_or_apply() takes '--profile NAME' (a product name) and --turbo-weights / --turbo-weight-bits /
// --turbo-weights-pack from argv and exports them (STRIX_MM_PRODUCT_PROFILE, STRIX_TURBO_WEIGHTS*) before anything
// resolves, so both apps feed profile_resolution() the same values. The per-phase timing session is
// '--timing-profile NAME:PROMPT.u32:STEPS'; the old '--profile NAME:PROMPT.u32:STEPS' stays accepted with one notice.
inline constexpr const char* kProductProfileEnv = "STRIX_MM_PRODUCT_PROFILE";
// The apps' own option loops call this: true when `a` is one of the options above (its value skipped). `timing`
// (rank only; nullptr in serve) receives a timing session spec ('--timing-profile', or the old '--profile' spelling).
bool product_option(const std::string& a, const std::function<std::string()>& next, std::string* timing);
// Cache / receipt identity of the active prefill policies; empty with every governed row at its builtin and the EXL3
// MoE format. This field's omission alone does not establish complete cache compatibility.
std::string prefill_identity(const Lookup& env);
// Effective STRIX_MMX_PF_ORDER (table default or explicit): "adaptive" or "invariant" (identity mmx-pf-linear-s1-v1).
std::string linear_order(const Lookup& env);
// Policy id of the index projection member for a canonical layer set ("minimax-index-proj-wmma-v1-l40-59").
std::string idx_wmma_policy(const std::string& layers);
// Rank agreement of the profile rows (strix::profile::check_peer over the process environment): both ranks call it at
// the same startup point with the control-channel byte exchange; a mismatch throws naming both fingerprints.
void check_peer(const std::function<void(const void*, void*, std::size_t)>& exchange);
void check_peer(const profile::Resolution& mine, const std::function<void(const void*, void*, std::size_t)>& exchange);

// Partial host receipt, not evidence of execution or asset compatibility. Records
// the engine's declared arithmetic and selected prefill route, without probing a
// device or assets. Runtime callers must supply measured identities/digests before
// emitting witness(); startup and effective_config() do not emit a witness.
profile::ExecutionReceipt host_receipt(const Lookup& env);

// Process environment, once per process: exports every unset item's resolved value (no overwrite) and prints one line
// when the baseline switch is on.
void apply();
// True when apply() exported `env` (the user did not set it): a default that cannot hold falls back with a notice
// instead of failing.
bool from_defaults(const char* env);
// A defaulted item that cannot hold on this load: re-export `value` (normally the item's off form) so receipts and
// cache identities name what runs, and print the reason. Throws if `env` was not exported by apply().
void decline_default(const char* env, const char* value, const std::string& reason);

// The USB4 transport names in session routes, whichever accepted spelling was requested
// (strix::dist::negotiated_transport(requested, direct_live, kRouteHost, kRouteGpu)).
inline constexpr const char* kRouteHost = "usb4-auto";
inline constexpr const char* kRouteGpu = "usb4gpu";

// The engine-shape options of the plain invocation (other options do not change the configuration).
struct Cli {
    std::string app = "rank";  // rank | serve
    std::string dspark_prior;  // explicit --dspark-prior, both apps
    std::string dspark_mode;  // explicit serve CLI, empty = resolver
    std::string transport = "usb4-auto";
    std::uint32_t slots = 8;
    std::uint64_t capacity = 8256;
    std::uint32_t max_rows = 4096;
    std::shared_ptr<const strix::weight::FamilyPlans> weight_plans{};
};
struct SessionConfig {
    bool enabled = true, disk = true;
    std::uint64_t ram_bytes = 0, disk_bytes = 0, max_parked = 8;
    std::string directory, reason, disk_reason, route;
};
// Resolve the startup vote without measuring resources or touching storage.
// Runtime callers must set Cli::transport from the negotiated device exchange.
SessionConfig session_policy(const Cli&, const Lookup&, bool explicit_cli = false);
// Runtime admission measures memory/storage and may probe the cache directory.
// Effective-config inspection uses pure policy resolution instead.
// explicit_cli bypasses automatic qualification/switches (except unqualified direct serving), retaining resource checks
// and validating explicit_directory rather than the automatic cache path.
SessionConfig session_config(const Cli&, const Lookup&, std::uint64_t available = UINT64_MAX,
                             bool explicit_cli = false, const std::string& explicit_directory = "");
Cli parse_cli(const std::string& app, const std::vector<std::string>& args);  // args without argv[0]

// Canonical text, one key=value per line: CLI shape, route/build switches, every item and resolved serve policy.
// Serve configuration validates scheduler and DSpark policy, independently of process environment export.
std::string effective_config(const Cli& cli, const Lookup& env);
inline constexpr const char* kPrintFlag = "--print-effective-config";
// Defaults of both entry points that are not items (transport timeouts, prefill table ring): ONE table.
// `serve == rank` unless serving needs another value, and then `serving_reason` says why; effective_config() prints
// every row (host.NAME=...) and the reason, so the server/CLI parity gate sees the same list on both sides.
struct HostDefault {
    const char* name;
    const char* rank;            // nullptr = left unset
    const char* serve;           // nullptr = left unset
    const char* serving_reason;  // nullptr when rank == serve
    bool optimization;           // true: not applied under STRIX_MM_BASELINE=1
};
std::span<const HostDefault> host_defaults();
// The value `app` ends with for one row: the explicit environment value, else the row default, else nullptr.
const char* host_default_value(const HostDefault& row, const std::string& app, const Lookup& env);
// setenv-if-unset of every row for `app` (rank | serve). An explicit value always wins.
void apply_host_defaults(const std::string& app);
// App entry: when argv holds kPrintFlag, apply() and print effective_config() of the invocation (no model, no GPU)
// and return true (the app exits 0); otherwise apply() and return false. Serve policy is validated
// on both paths before any transport/device/model work; rank invocations do not validate serve policy.
bool print_or_apply(const std::string& app, int argc, char** argv);

}  // namespace strix::minimax::defaults
