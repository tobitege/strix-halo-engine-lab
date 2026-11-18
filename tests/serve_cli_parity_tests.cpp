// Server / CLI configuration parity gate (strix/serve_cli_parity.hpp): for every family built here, the same plain
// request is resolved through the CLI entry point and through the HTTP server entry point of the family's ONE resolver;
// the test fails on any difference that is not listed below with the reason serving requires it, and on any listed
// difference that no longer exists. `--print` prints both resolved configurations side by side (the report tools use it).
#include "strix/serve_cli_parity.hpp"

#include <span>

#include <cstring>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#if defined(STRIX_PARITY_GLM)
#include "glm_rank_defaults.hpp"
#endif
#if defined(STRIX_PARITY_MINIMAX)
#include "minimax/engine/mm_defaults.hpp"
#include "minimax/engine/mm_profile_table.hpp"
#endif
#if defined(STRIX_PARITY_DS41)
#include "strix/deepseek41_pair_defaults.hpp"
#endif
#if defined(STRIX_PARITY_GEMMA4)
#include "gemma4/engine/runtime.hpp"
#endif

namespace {

namespace sp = strix::parity;
int g_failures = 0;
bool g_print = false;

void check(bool ok, const std::string& what) {
    if (ok) return;
    ++g_failures;
    std::cerr << "FAIL: " << what << "\n";
}

void gate(const std::string& family, const sp::Config& cli, const sp::Config& server, std::span<const sp::Allowed> allowed) {
    const auto report = sp::compare(cli, server, allowed);
    if (g_print) {
        std::cout << "== " << family << ": " << cli.size() << " cli keys, " << server.size() << " server keys, "
                  << report.explained.size() << " serving-only, " << report.unexplained.size() << " unexplained\n";
        for (const auto& [k, v] : cli) {
            const auto it = server.find(k);
            std::cout << family << " " << k << " cli=" << v << " server=" << (it == server.end() ? std::string(sp::kAbsent) : it->second) << "\n";
        }
        for (const auto& [k, v] : server)
            if (!cli.contains(k)) std::cout << family << " " << k << " cli=" << sp::kAbsent << " server=" << v << "\n";
    }
    if (!report.ok() || g_print) std::cerr << sp::text(family, report);
    check(report.ok(), family + ": server and CLI resolve the same request to different values");
}

void helper_tests() {
    const sp::Config a{{"x", "1"}, {"y", "2"}, {"only_cli", "c"}}, b{{"x", "1"}, {"y", "3"}, {"only_server", "s"}};
    const sp::Allowed none[] = {{"unused", "never differs"}};
    auto r = sp::compare(a, a, {});
    check(r.ok() && r.explained.empty(), "identical configurations pass");
    r = sp::compare(a, b, {});
    check(!r.ok() && r.unexplained.size() == 3U, "three unexplained differences");
    const sp::Allowed all[] = {{"y", "serving reason"}, {"only_*", "entry point only"}};
    r = sp::compare(a, b, all);
    check(r.ok() && r.explained.size() == 3U && r.explained[0].reason == "entry point only", "allowances with reasons");
    r = sp::compare(a, a, none);
    check(!r.ok() && r.stale.size() == 1U, "an allowance that no longer differs fails");
    const sp::Allowed no_reason[] = {{"y", ""}};
    r = sp::compare(a, b, no_reason);
    check(!r.ok(), "an allowance without a reason does not count");
    const auto l = sp::from_lines("a=1\nb=2 (auto)\r\nnote\n\n");
    check(l.size() == 3U && l.at("a") == "1" && l.at("b") == "2 (auto)" && l.at("line.0") == "note", "from_lines");
    const std::vector<std::string> args{"--rank", "0", "--tp", "--rows", "8", "16"};
    const auto g = sp::from_args(args);
    check(g.size() == 3U && g.at("arg:--rank") == "0" && g.at("arg:--tp").empty() && g.at("arg:--rows") == "8 16", "from_args");
    check(sp::text("f", sp::compare(a, b, {})).find("f UNEXPLAINED y cli=2 server=3") != std::string::npos, "report text");
}

#if defined(STRIX_PARITY_GLM)
// GLM: strix-glm5-rank and strix-glm5-serve call rd::scan / rd::resolve; the server forwards its rank options through
// rd::resolve_serve, including normal decode and startup adjustment (apps/glm_serve_main.cpp).
void glm() {
    namespace rd = strix::glm::rank_defaults;
    const rd::EnvGet none = [](const char*) -> const char* { return nullptr; };
    const std::vector<std::string> request{"--world", "2"};
    // Nothing is allowed to differ: max-rows auto is admitted at load by the same function on both entry points
    // (apps/glm_auto_rows.hpp), so the server resolves the CLI's 8192-row candidate and its admission line.
    const std::span<const sp::Allowed> allowed{};
    // "value (source)": the gate compares the value; the source label names the entry point's path to it.
    const auto values = [](sp::Config c) {
        for (auto& [k, v] : c)
            if (const auto at = v.rfind(" ("); at != std::string::npos && !v.empty() && v.back() == ')') v.erase(at);
        return c;
    };
    for (const char* profile : {"default", "fast", "quality"})
    for (const bool complete : {false, true}) for (const bool opt_out : {false, true}) {
        auto profile_request = request;
        profile_request.insert(profile_request.end(), {"--profile", profile});
        rd::ModelFacts facts;
        facts.mtp_present = facts.mtp_complete = complete;
        facts.reason = complete ? "complete" : "missing mtp.safetensors";
        auto cli_args = profile_request;
        cli_args.insert(cli_args.end(), {"--decode", "64"});
        if (opt_out) cli_args.push_back("--no-mtp");
        const auto cli = rd::resolve(rd::scan(cli_args), none, facts);
        const auto server = rd::resolve_serve(profile_request, !opt_out, none, facts);
        const std::string expected = complete && !opt_out ? "1" : "0";
        check(cli.mtp_spec.value == expected && server.mtp_spec.value == expected &&
              cli.mtp.value == expected && server.mtp.value == expected && server.prefill == "gemv",
              "GLM complete/missing assets and opt-out select intended MTP/prefill route");
        gate("glm", values(sp::from_lines(rd::print(cli))), values(sp::from_lines(rd::print(server))), allowed);
    }
}
#endif

#if defined(STRIX_PARITY_MINIMAX)
// MiniMax: strix-minimax-rank and strix-minimax-serve call defaults::print_or_apply(app, ...) = parse_cli + effective_config.
void minimax() {
    namespace md = strix::minimax::defaults;
    const md::Lookup none = [](const char*) -> const char* { return nullptr; };
    const auto cli = sp::from_lines(md::effective_config(md::parse_cli("rank", {}), none));
    const auto server = sp::from_lines(md::effective_config(md::parse_cli("serve", {}), none));
    static constexpr sp::Allowed allowed[] = {
        {"app", "the entry point's own name"},
        {"session_reason", "text that quotes the app and transport names"},
        // EXCEPTION for the user (not measured server vs CLI on the same transport): the server defaults to the device
        // exchange, the CLI to the host exchange.
        {"transport", "serve default usb4gpu (device exchange), rank default usb4-auto (host exchange)"},
        {"serve.*", "scheduler / speculation policy between concurrent requests; no CLI counterpart"},
        // One host-path table (mm_defaults host_defaults()), two columns:
        {"host.STRIX_DIST_USB4_TIMEOUT_MS", "transport only: a request fails within a minute when the peer is gone"},
        {"host.STRIX_DIST_WAIT_TIMEOUT_MS", "transport only: a request fails within a minute when the peer is gone"},
        {"host.STRIX_DIST_USB4_TIMEOUT_MS.serving_reason", "the reason line exists on the server only"},
        {"host.STRIX_DIST_WAIT_TIMEOUT_MS.serving_reason", "the reason line exists on the server only"},
        // EXCEPTION for the user: exact prefill MoE table ring (1K -0.5 %), qualified with one engine owner only.
        {"host.STRIX_MM_PF_TABLE_RING", "not qualified with several sessions sharing the engine"},
        {"host.STRIX_MM_PF_TABLE_RING.serving_reason", "the reason line exists on the server only"},
    };
    gate("minimax", cli, server, allowed);
    // FAST prefill profile rows (mm_profile_table.hpp): every row is printed by both entry points and resolves the
    // same with the bundle on, with an explicit OFF beating it, and on the FAST_I4 format (member not selected).
    for (const auto& keys : {std::map<std::string, std::string>{{"STRIX_MM_FAST_PREFILL", "1"}},
                             std::map<std::string, std::string>{{"STRIX_MM_FAST_PREFILL", "1"}, {"STRIX_MM_PF_MSA_FLASH", "0"}},
                             std::map<std::string, std::string>{{"STRIX_MM_FAST_PREFILL", "1"}, {"STRIX_MMX_MODEL_FORMAT", "fast_i4"}},
                             std::map<std::string, std::string>{{"STRIX_MM_PF_MSA_FLASH", "2"}, {"STRIX_MM_PF_MSA_FLASH_LAYERS", "50-59"}}}) {
        const md::Lookup env = [&keys](const char* k) -> const char* {
            const auto it = keys.find(k);
            return it == keys.end() ? nullptr : it->second.c_str();
        };
        const auto c = sp::from_lines(md::effective_config(md::parse_cli("rank", {}), env));
        const auto s = sp::from_lines(md::effective_config(md::parse_cli("serve", {}), env));
        for (const auto& row : md::profile_table)
            check(c.count(std::string(row.env)) == 1 && s.count(std::string(row.env)) == 1,
                  "minimax: profile row " + std::string(row.env) + " printed by both entry points");
        for (const char* k : {"prefill.fast_profile", "prefill.fast_policies", "prefill.identity", "profile_fingerprint"})
            check(c.count(k) == 1 && s.count(k) == 1, std::string("minimax: ") + k + " printed by both entry points");
        gate("minimax+fast", c, s, allowed);
    }
}
#endif

#if defined(STRIX_PARITY_DS41)
// DeepSeek41 pair: every app calls pair_defaults::resolve(App, ...) (Effective in the process).
void ds41() {
    namespace pd = strix::deepseek41::pair_defaults;
    const std::string model = "/md/ds41.gguf";
    const std::set<std::string> files{model}, dirs{"/md"};
    pd::Probe probe;
    probe.available_memory = [] { return std::uint64_t{64} << 30U; };
    probe.writable_space = [](const std::string&) { return std::uint64_t{512} << 30U; };
    probe.file = [&](const std::string& f) { return files.count(f) != 0; };
    probe.dir = [&](const std::string& d) { return dirs.count(d) != 0; };
    probe.list = [](const std::string&) { return std::vector<std::string>{}; };
    const auto config = [&](pd::App app, const char* profile) {
        const auto r = pd::resolve(app, {model, "--rank", "0", "--transport", "usb4auto", "--profile", profile},
                                   [](const char*) -> const char* { return nullptr; }, probe, pd::Build{true});
        auto c = sp::from_args(std::span<const std::string>(r.args).subspan(1));
        for (const auto& [k, v] : r.set_env) c["env:" + k] = v;
        for (const auto& k : r.unset_env) c["env:" + k] = "<unset>";
        c["session.enabled"] = r.session.enabled ? "1" : "0";
        c["session.disk"] = r.session.disk ? "1" : "0";
        if (r.profile) {
            c["profile"] = std::string(strix::profile::name(r.profile->profile));
            c["profile.fingerprint"] = strix::profile::hex_identity(r.profile->fingerprint());
            for (const auto& entry : r.profile->entries)
                if (entry.lever) c["profile." + std::string(entry.lever->name)] = entry.value;
        }
        return c;
    };
    static constexpr sp::Allowed allowed[] = {
        {"arg:--peer-timeout", "serving waits for the peer rank; the CLI starts both ranks together"},
        {"arg:--slots", "request slots exist on the server only"},
        {"arg:--prefill-path", "the server prefills prompts on the device batch path; the CLI's prompt goes through its decode windows"},
        {"arg:--prefill-rows", "row budget of the server's device prefill path (auto)"},
    };
    for (const char* profile : {"default", "fast", "quality"})
        gate(std::string("ds41+") + profile, config(pd::App::rank, profile), config(pd::App::serve, profile), allowed);
}
#endif

#if defined(STRIX_PARITY_GEMMA4)
// Gemma: strix-gemma4-decode calls parse_options, strix-gemma4-serve calls prepare_serve = parse_options + the serve
// additions (src/gemma4/engine/runtime.cpp). Same request words through both; every option and profile entry compared.
void gemma() {
    namespace e = strix::gemma4::engine;
    const auto config = [](const e::Options& o) {
        sp::Config c;
        c["model"] = o.model.generic_string();
        c["rope"] = o.rope.generic_string();
        c["backend"] = o.backend;
        c["threads"] = std::to_string(o.threads);
        c["streams"] = std::to_string(o.streams);
        c["request.max_tokens"] = std::to_string(o.request.max_tokens);
        std::string stops;
        for (const auto id : o.request.stop_ids) stops += std::to_string(id) + ",";
        c["request.stop_ids"] = stops;
        c["request.temperature"] = std::to_string(o.request.sampling.temperature);
        c["request.top_p"] = std::to_string(o.request.sampling.top_p);
        c["request.top_k"] = std::to_string(o.request.sampling.top_k);
        c["request.seed"] = std::to_string(o.request.sampling.seed);
        c["request.dflash"] = std::to_string(static_cast<int>(o.request.dflash));
        c["request.draft_k"] = std::to_string(o.request.draft_k);
        c["draft"] = o.draft.generic_string();
        c["draft_mtp"] = o.draft_mtp.generic_string();
        c["profile"] = std::to_string(static_cast<int>(o.profile.name));
        c["profile.app"] = std::to_string(static_cast<unsigned>(o.profile.app));
        for (const auto& entry : o.profile.entries)
            if (entry.lever) c["profile." + std::string(entry.lever->name)] = entry.value;
        return c;
    };
    std::vector<std::string> words{"x", "--model", "/m/model.gguf", "--rope-tables", "/m/rope"};
    std::vector<char*> argv;
    for (auto& w : words) argv.push_back(w.data());
    const auto cli = config(e::parse_options(static_cast<int>(argv.size()), argv.data()));
    const auto served = e::prepare_serve(static_cast<int>(argv.size()), argv.data());
    check(!served.exit_code, "gemma: prepare_serve accepts the CLI's request words");
    static constexpr sp::Allowed allowed[] = {
        // EXCEPTION for the user: the server sizes session-fixed choices (the EXL3 core identity) for the requests that
        // can decode together (STRIX_G4_SERVE_SLOTS, default 8); the CLI runs one stream unless --streams N.
        {"streams", "the server decodes up to STRIX_G4_SERVE_SLOTS requests together"},
    };
    gate("gemma", cli, config(served.options), allowed);
}
#endif

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) g_print |= std::strcmp(argv[i], "--print") == 0;
    try {
        helper_tests();
#if defined(STRIX_PARITY_GLM)
        std::cout << "included: glm (default/fast/quality; complete/missing MTP assets, default/opt-out)\n";
        glm();
#else
        std::cout << "skipped: glm (resolver not linked)\n";
#endif
#if defined(STRIX_PARITY_MINIMAX)
        std::cout << "included: minimax\n";
        minimax();
#else
        std::cout << "skipped: minimax (resolver not linked)\n";
#endif
#if defined(STRIX_PARITY_DS41)
        std::cout << "included: ds41\n";
        ds41();
#else
        std::cout << "skipped: ds41 (resolver not linked)\n";
#endif
#if defined(STRIX_PARITY_GEMMA4)
        std::cout << "included: gemma\n";
        gemma();
#else
        std::cout << "skipped: gemma (strix_gemma4_engine not built)\n";
#endif
        std::cout << "qwen: strix-qwen-effective-config-tests (serve / CLI parity case)\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: exception: " << e.what() << "\n";
        return 1;
    }
    if (g_failures) std::cerr << g_failures << " failure(s)\n";
    else std::cout << "serve/cli parity: ok\n";
    return g_failures ? 1 : 0;
}
