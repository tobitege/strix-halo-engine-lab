#pragma once
// Shared default resolution for exact speculative decoding (MTP heads, draft
// sidecars, self-speculation). An engine only declares what it has; this file
// decides whether the production command uses it and words the one fallback
// line. One opt-out for every engine: `--no-mtp` or STRIX_MTP=0.
#include <string>
#include <string_view>

namespace strix::session {

// What an engine's production route can do. A path that is not token-identical
// to plain greedy decode, or that is not faster, is never a default.
struct SpeculationCapability {
    std::string_view engine;
    bool available = false;      // a speculative path exists in this build
    bool exact = false;          // output equals plain greedy decode
    bool faster = false;         // measured faster than plain decode on the production route
    bool needs_assets = false;   // a drafter asset must be found and verified
};

struct SpeculationRequest {
    bool cli_opt_out = false;        // --no-mtp
    const char* environment = nullptr; // value of STRIX_MTP, or null
    bool baseline = false;           // the engine's baseline switch
    bool explicit_route = false;     // the caller spelled a route of its own
    bool explicit_speculation = false; // the caller spelled the speculative options
    bool assets_found = true;        // meaningful only with needs_assets
};

struct SpeculationDecision {
    bool enabled = false;
    // "default", "explicit", "cli-opt-out", "env-opt-out", "baseline",
    // "explicit-route", "assets-missing", "not-exact", "not-faster", "unavailable"
    std::string_view source = "unavailable";
    bool report_fallback = false;    // print fallback_line() once
};

inline constexpr std::string_view speculation_opt_out_flag = "--no-mtp";
inline constexpr const char* speculation_environment = "STRIX_MTP";

inline bool speculation_opt_out_argument(std::string_view argument) {
    return argument == speculation_opt_out_flag;
}
inline bool speculation_environment_off(const char* value) {
    const std::string_view text = value ? value : "";
    return text == "0" || text == "off";
}

inline SpeculationDecision resolve_speculation_default(const SpeculationCapability& capability,
    const SpeculationRequest& request) {
    if (!capability.available) return {false, "unavailable", false};
    if (request.cli_opt_out) return {false, "cli-opt-out", false};
    if (speculation_environment_off(request.environment)) return {false, "env-opt-out", false};
    if (request.baseline) return {false, "baseline", false};
    if (request.explicit_speculation) return {true, "explicit", false};
    if (request.explicit_route) return {false, "explicit-route", false};
    if (!capability.exact) return {false, "not-exact", false};
    if (!capability.faster) return {false, "not-faster", false};
    if (capability.needs_assets && !request.assets_found) return {false, "assets-missing", true};
    return {true, "default", false};
}

// The single line an engine prints when its default speculative path cannot run.
inline std::string speculation_fallback_line(const SpeculationCapability& capability, std::string_view detail) {
    std::string line = "MTP default (";
    line += capability.engine;
    line += ") unavailable; using plain decode: ";
    line += detail;
    line += " (silence with --no-mtp or STRIX_MTP=0)";
    return line;
}

} // namespace strix::session
