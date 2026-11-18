#pragma once
#include <strix/weight/family.hpp>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "strix/engine_profile.hpp"

// Two-node (TP2 pair) defaults of the deepseek41 apps: the plain invocation resolves to the measured production
// command of the 2026-10-26 qualification run, so nobody has to
// pass the production environment or flags by hand.
//
//   strix-deepseek41-{rank,batch,serve} MODEL.gguf --rank R --transport usb4auto     (prefill: --transport usb4-auto)
//
// resolves, before any argument or environment variable is read, to:
//   environment (every app but prefill): STRIX_DS41_OVERLAY=<packs>/c-u6.gguf STRIX_DS41_CB4=1
//     STRIX_DS41_TRELLIS=<packs>/ours-K4 STRIX_DS41_TRELLIS_MIX=head=<packs>/ours-K5 STRIX_DS41_TRELLIS_SITES=all
//     STRIX_DS41_TRELLIS_DENSE_FULL=1 STRIX_DS41_TRELLIS_HEADOUT=1
//   prefill: STRIX_DS41_OVERLAY + STRIX_DS41_CB4 as above and STRIX_DS41_EXPERT_TP=0 (the EP prefill, no trellis sites)
//   rank / batch on two ranks: --tp-attention --engram-direct
//   serve: --peer-timeout 120 --prefill-path device --slots 8 --prefill-rows auto (largest piece both ranks admit)
//     --dspark <model dir>/*-DSpark-MXFP4.gguf --dspark-vocab auto:<packs>/vocab
// <packs> is STRIX_DS41_PACKS when set, else <model dir>/packs, else the model's directory.
//
// The defaults go in front of the user's arguments, so any flag given on the command line wins (the parsers keep
// the last value); an environment variable that is already set is never changed. Off switches, one per item:
//   STRIX_DS41_OVERLAY=0 (overlay + CB4), STRIX_DS41_TRELLIS=0 (every trellis variable), STRIX_DS41_PACKS=0 (both),
//   --no-tp-attention, --engram-mapped, serve --no-dspark / --dspark-vocab full / explicit --slots, --prefill-rows,
//   --peer-timeout, --prefill-path; and the code defaults of the same gates, each with its own =0 (see
//   baseline_off_items()).
// Master switch STRIX_DS41_BASELINE=1: nothing above is added and every item adopted on 2026-10-25/26 is switched off
// (=0, unless set explicitly): the plain invocation runs as it did before, without those items.
// A precondition that does not hold (a pack, the draft or the vocabulary not found, a build without the dense_r1
// trellis library, one rank) drops that item with one notice line on stderr; it never fails the run.
// Automatic serving sessions require a GPU-qualified route and memory/disk admission. The qualification
// allowlist is empty. STRIX_SESSION_CACHE=0, --session-cache off and STRIX_DS41_BASELINE disable sessions.
// One-shot rank commands remain explicit.
// --print-effective-config prints the resolved environment and arguments and exits before any device work.
//
// Product profile (every app): '--profile default|fast|quality' (strix/engine_profile.hpp, rows in
// strix/deepseek41_pair_profile_table.hpp) resolves first; default sets nothing, so the resolution above is
// unchanged. A fast/quality row that is not qualified is pending: listed in the receipt ("profile {...}") and in
// one stderr line, never applied, never a refusal. '--allow-pending' (STRIX_DS41_PROFILE_ALLOW_PENDING=1) runs a
// pending row whose code is in this build, stamped "unqualified". Both ranks must run the same profile.
namespace strix::deepseek41::pair_defaults {

enum class App { rank, batch, prefill, serve };

struct DsparkSettings {
    bool copy_rows8 = false;
    bool arbiter = false;
};
// Device-independent switch registration, also used before inspection exits.
// Lookup admission still controls whether BatchSpeculator applies these settings.
DsparkSettings resolve_dspark_settings();

struct Build {
    bool trellis_dense_r1 = false;  // linked against a trellis library with the dense-r1 API (STRIX_DS41_TRELLIS_API)
};
// The app's own build (STRIX_DS41_PAIR_DENSE_R1 comes with strix_ds41_trellis_hip when it links dense_r1).
inline Build this_build() {
#if defined(STRIX_DS41_PAIR_DENSE_R1)
    return Build{true};
#else
    return Build{false};
#endif
}

// Filesystem questions of the resolver (a fake in the unit test).
struct Probe {
    std::function<bool(const std::string&)> file, dir;
    std::function<std::uint64_t()> available_memory;
    // Creates/probes a private cache directory; zero means unavailable.
    std::function<std::uint64_t(const std::string&)> writable_space;
    std::function<std::vector<std::string>(const std::string&)> list;  // file names in a directory
};

struct SessionDefaults {
    bool enabled = false, disk = false;
    std::uint64_t ram_bytes = 0, disk_bytes = 0, max_parked = 8;
    std::string directory, reason;
};

struct Resolved {
    SessionDefaults session;
    std::vector<std::string> args;                            // argv[1..] after the defaults
    std::vector<std::pair<std::string, std::string>> set_env;  // set (they were unset)
    std::vector<std::string> unset_env;                        // a path variable given as =0: removed
    std::vector<std::string> notices;                          // one line each
    bool print_only = false;                                   // --print-effective-config (removed from args)
    std::optional<strix::profile::Resolution> profile;         // --profile (removed from args); always resolved
    // STRIX_DS41_DECODE_XPORT as BatchForward resolves it (strix/ds41_decode_xport.hpp): "communicator" / "channel",
    // then the origin ("default" / "explicit"). One rule for CLI, serve and engine; both ranks also agree at load.
    std::string decode_xport;
    std::shared_ptr<const strix::weight::FamilyPlans> weight_plans{};
};

using Getenv = std::function<const char*(const char*)>;

const char* app_name(App app);
// The product-profile table of the pair apps (deepseek41_pair_profile_table.hpp).
const strix::profile::Engine& profile_engine();
// JSON of this process's resolved profile ("" before Effective ran): the serve's capabilities and receipts.
const std::string& profile_json();
// This process's resolved profile (Effective ran first): two-rank apps check it against the peer
// (strix::profile::check_peer) on their control channel right after the communicator exists.
const strix::profile::Resolution& profile_resolution();
// The code defaults adopted on 2026-10-25/26 that STRIX_DS41_BASELINE=1 switches off (variable, value).
const std::vector<std::pair<std::string, std::string>>& baseline_off_items();
// Empty ON means leave the app/kernel builtin untouched; OFF is baseline export.
std::span<const strix::profile::DefaultItem> default_items();
// Host provenance only, never a complete startup execution witness. The execution
// owner supplies the admitted numerical contract (e.g. ds41-exact-v1 versus
// ds41-expert-tp2-i1152-v1), assets and lineage; unresolved fields stay empty.
strix::profile::ExecutionReceipt host_receipt(App app, const Resolved& resolved,
                                             strix::profile::NumericalIdentity identity = {});
Resolved resolve(App app, const std::vector<std::string>& args, const Getenv& env, const Probe& probe, const Build& build);
// The printed form of --print-effective-config: "app", then every set/unset variable and argument, one per line.
std::string describe(App app, const Resolved& r);

// Process side: resolve against the real environment and filesystem, apply the environment, print the notices
// (--print-effective-config: print and exit 0) and replace argc / argv (storage owned by the returned object).
class Effective {
public:
    Effective(App app, int& argc, char**& argv, const Build& build);

private:
    std::vector<std::string> storage_;
    std::vector<char*> ptrs_;
};

}  // namespace strix::deepseek41::pair_defaults
