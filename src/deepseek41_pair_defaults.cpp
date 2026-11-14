#include "strix/env_flags.hpp"
#include "strix/deepseek41_pair_defaults.hpp"
#include "strix/deepseek41_pair_profile_table.hpp"
#include "strix/deepseek41_serve_cache_policy.hpp"
#include "strix/session/speculation_default.hpp"

#include <algorithm>
#include <array>
#include <string_view>
#include <chrono>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <limits>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <sstream>
#include <system_error>

namespace strix::deepseek41::pair_defaults {
namespace {

bool set(const char* v) { return v != nullptr && *v != 0; }
bool zero(const char* v) { return v != nullptr && v[0] == '0' && v[1] == 0; }
bool has(const std::vector<std::string>& args, const char* flag) { return std::find(args.begin(), args.end(), flag) != args.end(); }

// The value of the last `flag VALUE` pair, or "".
std::string last_value(const std::vector<std::string>& args, const char* flag) {
    std::string v;
    for (std::size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == flag) v = args[i + 1];
    return v;
}

// MODEL.gguf: the first argument (the documented form), else the first .gguf that is not a flag's value.
std::string model_path(const std::vector<std::string>& args) {
    if (!args.empty() && !args[0].empty() && args[0][0] != '-') return args[0];
    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto& s = args[i];
        if (s.size() > 5 && s.ends_with(".gguf") && s[0] != '-' && (i == 0 || args[i - 1].rfind("--", 0) != 0)) return s;
    }
    return {};
}

std::string parent_dir(const std::string& path) {
    const auto p = std::filesystem::path(path).parent_path().generic_string();
    return p.empty() ? std::string(".") : p;
}

// Sole admission allowlist: add a route only after its complete native GPU gate passes.
// Empty deliberately: CPU evidence does not qualify any TP2 session route.
constexpr std::array<std::string_view, 0> gpu_qualified_session_routes{};

// Resolve serving optimizations after the production route is known. One-shot rank
// save/load remains explicit; enabling this cache never selects a different producer.
void resolve_sessions(App app, Resolved& r, const Getenv& env, const Probe& probe, bool baseline) {
    auto& c = r.session;
    const auto value = [&](const char* flag, const char* variable, const std::string& fallback) {
        const auto cli = last_value(r.args, flag);
        if (!cli.empty()) return cli;
        const char* e = env(variable);
        return set(e) ? std::string(e) : fallback;
    };
    bool invalid_budget = false;
    const auto number = [&](const std::string& v, std::uint64_t fallback = 0) -> std::uint64_t {
        try {
            if (!v.empty() && v.find_first_not_of("0123456789") == std::string::npos)
                return std::stoull(v);
        } catch (const std::exception&) {}
        invalid_budget = true;
        return fallback;
    };
    const auto toggle = [](const std::string& v) {
        return v != "0" && v != "off" && v != "false";
    };
    const bool requested = toggle(value("--session-cache", "STRIX_SESSION_CACHE", "on"));
    const bool want_disk = toggle(value("--session-disk", "STRIX_SESSION_DISK", "on"));
    const auto available = [&]() -> std::uint64_t {
        try { return probe.available_memory ? probe.available_memory() : 0; }
        catch (const std::exception&) { return 0; }
    }();
    const auto default_ram = std::min<std::uint64_t>(8ULL << 30U, available / 10U);
    c.ram_bytes = number(value("--session-ram-bytes", "STRIX_SESSION_RAM_BYTES",
        std::to_string(default_ram)), default_ram);
    const auto legacy_mb = last_value(r.args, "--session-cache-mb");
    if (!legacy_mb.empty() && last_value(r.args, "--session-ram-bytes").empty()) {
        const auto mb = number(legacy_mb);
        if (mb > (1ULL << 20U)) invalid_budget = true;
        else c.ram_bytes = mb << 20U;
    }
    c.max_parked = number(value("--session-max-parked", "STRIX_SESSION_MAX_PARKED", "8"), 8);
    std::string base;
    if (set(env("XDG_CACHE_HOME"))) base = env("XDG_CACHE_HOME");
    else if (set(env("HOME"))) base = std::string(env("HOME")) + "/.cache";
    else if (set(env("USERPROFILE"))) base = std::string(env("USERPROFILE")) + "/.cache";
    c.directory = value("--session-dir", "STRIX_SESSION_DIR", base.empty() ? "" : base + "/strix-engine/sessions");
    if (!last_value(r.args, "--session-cache-dir").empty() && last_value(r.args, "--session-dir").empty())
        c.directory = last_value(r.args, "--session-cache-dir");
    const auto disk_budget = value("--session-disk-bytes", "STRIX_SESSION_DISK_BYTES", "");
    c.disk_bytes = disk_budget.empty() ? 0 : number(disk_budget);
    std::string effective_draft;
    for (std::size_t i = 0; i < r.args.size(); ++i) {
        if (r.args[i] == "--no-dspark" || strix::session::speculation_opt_out_argument(r.args[i])) effective_draft.clear();
        else if (r.args[i] == "--dspark" && i + 1 < r.args.size()) effective_draft = r.args[++i];
    }
    const std::string route = !effective_draft.empty() ? "tp2-http-batch-dspark" : "tp2-http-target";
    const bool qualified = std::find(gpu_qualified_session_routes.begin(), gpu_qualified_session_routes.end(), route) != gpu_qualified_session_routes.end();
    if (baseline) c.reason = "baseline switch";
    else if (!requested) c.reason = "off switch";
    else if (app != App::serve) c.reason = "one-shot commands remain explicit";
    else if (!qualified) c.reason = "route not yet GPU-qualified: " + route;
    else if (!effective_draft.empty())
        c.reason = "batch DSpark has no qualified session adapter; sequential R1 rank sessions only";
    else if (!last_value(r.args, "--hetero-listen").empty() || last_value(r.args, "--transport") == "single" ||
             (!last_value(r.args, "--layers").empty() && last_value(r.args, "--layers") != "40"))
        c.reason = "requires whole TP2 ranks";
    else if (invalid_budget) c.reason = "invalid session budget";
    else {
        const auto context = last_value(r.args, "--context");
        const auto n = context.empty() ? 4096ULL : number(context);
        // Conservative per-rank envelope plus export/validation and slot-state headroom.
        const auto one = n > (1ULL << 32U) ? std::numeric_limits<std::uint64_t>::max() : (16ULL << 20U) + n * 2048ULL;
        if (invalid_budget || !c.max_parked || c.max_parked > std::numeric_limits<std::uint32_t>::max() || c.ram_bytes < one || available < one || c.ram_bytes > available / 2U)
            c.reason = "insufficient available memory or parked-session budget";
        else {
            c.enabled = true;
            if (want_disk) {
                const auto free = [&]() -> std::uint64_t {
                    try { return !c.directory.empty() && probe.writable_space ? probe.writable_space(c.directory) : 0; }
                    catch (const std::exception&) { return 0; }
                }();
                if (disk_budget.empty()) c.disk_bytes = std::min<std::uint64_t>(64ULL << 30U, free / 4U);
                c.disk = free >= 2U * one && c.disk_bytes >= 2U * one;
                if (!c.disk) r.notices.emplace_back("session disk: off (directory unavailable or insufficient space)");
            }
        }
    }
    // Consume common controls here so every app accepts the off switches.
    std::vector<std::string> kept;
    for (std::size_t i = 0; i < r.args.size(); ++i) {
        const auto& a = r.args[i];
        const bool common = a == "--session-cache" || a == "--session-disk" || (a == "--session-ram-bytes" && app != App::rank) ||
            a == "--session-max-parked" || a == "--session-dir" || a == "--session-disk-bytes";
        const bool legacy = app == App::serve && (a == "--session-cache-mb" || a == "--session-cache-dir");
        const bool rank_off = app == App::rank && (baseline || !requested) && a.starts_with("--session-");
        if (rank_off && (a == "--session-dspark" || a == "--session-validate-only" || a == "--session-hash-only")) continue;
        if (common || legacy || rank_off) {
            if (++i >= r.args.size()) throw std::invalid_argument(a + " needs a value");
        } else kept.push_back(a);
    }
    r.args = std::move(kept);
    if (app == App::serve) {
        if (c.enabled) {
            r.args.insert(r.args.end(), {"--session-ram-bytes", std::to_string(c.ram_bytes),
                "--session-max-parked", std::to_string(c.max_parked),
                "--session-disk-bytes", std::to_string(c.disk ? c.disk_bytes : 0)});
            if (c.disk) r.args.insert(r.args.end(), {"--session-cache-dir", c.directory});
        } else r.notices.push_back("session cache: off (" + c.reason + ")");
    }
}

}  // namespace

const char* app_name(App app) {
    switch (app) {
    case App::rank: return "strix-deepseek41-rank";
    case App::batch: return "strix-deepseek41-batch";
    case App::prefill: return "strix-deepseek41-prefill";
    case App::serve: return "strix-deepseek41-serve";
    }
    return "?";
}

const strix::profile::Engine& profile_engine() {
    static const strix::profile::Engine engine{"deepseek41-pair", "STRIX_DS41_PROFILE_ALLOW_PENDING", profile_table, false};
    return engine;
}

namespace {
std::string& profile_json_storage() {
    static std::string json;
    return json;
}
std::optional<strix::profile::Resolution>& profile_storage() {
    static std::optional<strix::profile::Resolution> resolution;
    return resolution;
}
}  // namespace

const std::string& profile_json() { return profile_json_storage(); }

const strix::profile::Resolution& profile_resolution() {
    auto& r = profile_storage();
    if (!r) r = strix::profile::resolve(profile_engine(), {});  // no Effective (tests): the default resolution
    return *r;
}

const std::vector<std::pair<std::string, std::string>>& baseline_off_items() {
    // 2026-10-25 release: D1 shared expert as grouped SwiGLU pairs. 2026-10-26 release: D1b fused shared
    // down, prefill MoE form 7, exact dense selector, attention window fold / live chunks / f16 tile, serve warm-up
    // and the DSpark cost calibration that follows it.
    static const auto items = [] {
        std::vector<std::pair<std::string, std::string>> out;
        for (const auto& row : default_items()) out.emplace_back(row.env, row.off);
        return out;
    }();
    return items;
}

std::span<const strix::profile::DefaultItem> default_items() {
    static constexpr strix::profile::DefaultItem items[] = {
        {"shared_grouped", "STRIX_DS41_SHARED_GROUPED", "", "0", ""},
        {"shared_down_fused", "STRIX_DS41_SHARED_DOWN_FUSED", "", "0", ""},
        {"moe_prefill", "STRIX_DS41_MOE_PREFILL", "", "0", ""},
        {"dense_select", "STRIX_DS41_DENSE_SELECT", "", "0", ""},
        {"attn_window_fold", "STRIX_DS41_ATTN_WINDOW_FOLD", "", "0", ""},
        {"attn_live_chunks", "STRIX_DS41_ATTN_LIVE_CHUNKS", "", "0", ""},
        {"attn_tile_f16", "STRIX_DS41_ATTN_TILE_F16", "", "0", ""},
        {"serve_warm", "STRIX_DS41_SERVE_WARM", "", "0", ""},
        {"dspark_calib", "STRIX_DS41_DSPARK_CALIB", "", "0", ""},
    };
    return items;
}

strix::profile::ExecutionReceipt host_receipt(App app, const Resolved& resolved,
                                             strix::profile::NumericalIdentity identity) {
    strix::profile::ExecutionReceipt out;
    out.engine = "deepseek41-pair";
    out.implementation = app_name(app);
    out.identity = std::move(identity);
    if (resolved.profile)
        out.digests.emplace_back("profile_fingerprint", strix::profile::hex_identity(resolved.profile->fingerprint()));
    return out;
}

Resolved resolve(App app, const std::vector<std::string>& in, const Getenv& raw_env, const Probe& probe, const Build& build) {
    Resolved r;
    std::vector<std::string> user;
    for (const auto& a : in) {
        if (a == "--print-effective-config") r.print_only = true;
        else user.push_back(a);
    }
    // Product profile first: its switches are environment for everything below (an explicit one wins).
    strix::profile::Inputs profile_inputs;
    profile_inputs.env = raw_env;
    strix::profile::take_options(profile_engine(), user, profile_inputs);
    r.profile = strix::profile::resolve(profile_engine(), profile_inputs);
    std::vector<std::pair<std::string, std::string>> profile_env;
    for (const auto& e : r.profile->entries)
        if (!e.lever->env.empty() && strix::profile::applied(e.status))
            profile_env.emplace_back(std::string(e.lever->env), e.value);
    r.set_env = profile_env;
    if (auto extra = strix::profile::profile_args(*r.profile); !extra.empty())
        user.insert(user.begin() + (!user.empty() && !user[0].empty() && user[0][0] != '-' ? 1 : 0), extra.begin(), extra.end());
    for (auto& n : r.profile->notices()) r.notices.push_back(std::move(n));
    const Getenv env = [&](const char* name) -> const char* {
        for (const auto& [k, v] : profile_env) if (k == name) return v.c_str();
        return raw_env(name);
    };
    const auto put = [&](const char* name, std::string value) {
        if (env(name) == nullptr) r.set_env.emplace_back(name, std::move(value));
    };
    const auto notice = [&](const std::string& s) { r.notices.push_back(std::string("deepseek41 defaults: ") + s); };

    const bool baseline = set(env("STRIX_DS41_BASELINE")) && !zero(env("STRIX_DS41_BASELINE"));
    if (baseline) {
        for (const auto& row : default_items()) {
            const auto value = strix::profile::resolve_default(row, true, env);
            if (value.source == strix::profile::DefaultSource::baseline) put(row.env, value.value);
        }
        r.args = std::move(user);
        resolve_sessions(app, r, env, probe, true);
        return r;
    }

    // Production identity: c-u6 dense overlay (CB4) and, outside the EP prefill, the trellis mix-B packs at every
    // dense site. All or nothing: a partial identity was never measured.
    const auto model = model_path(user);
    const char* packs_env = env("STRIX_DS41_PACKS");
    std::vector<std::string> roots;
    if (set(packs_env) && !zero(packs_env)) roots.emplace_back(packs_env);
    else if (!model.empty()) roots = {parent_dir(model) + "/packs", parent_dir(model)};
    const auto find = [&](const std::string& name, bool directory) -> std::string {
        for (const auto& root : roots) {
            const auto p = root + "/" + name;
            if (directory ? probe.dir(p) : probe.file(p)) return p;
        }
        return {};
    };
    const char* overlay_env = env("STRIX_DS41_OVERLAY");
    const char* trellis_env = env("STRIX_DS41_TRELLIS");
    if (zero(overlay_env)) r.unset_env.emplace_back("STRIX_DS41_OVERLAY");
    if (zero(trellis_env)) r.unset_env.emplace_back("STRIX_DS41_TRELLIS");
    if (zero(env("STRIX_DS41_TRELLIS_MIX"))) r.unset_env.emplace_back("STRIX_DS41_TRELLIS_MIX");
    const bool want_trellis = app != App::prefill;
    if (!zero(packs_env)) {
        const bool overlay_auto = !set(overlay_env);
        const bool trellis_auto = want_trellis && !set(trellis_env) && build.trellis_dense_r1;
        const auto overlay = overlay_auto ? find("c-u6.gguf", false) : std::string();
        const auto k4 = trellis_auto ? find("ours-K4", true) : std::string();
        const auto k5 = trellis_auto ? find("ours-K5", true) : std::string();
        std::string missing;
        if (overlay_auto && overlay.empty()) missing += " c-u6.gguf";
        if (trellis_auto && k4.empty()) missing += " ours-K4/";
        if (trellis_auto && k5.empty()) missing += " ours-K5/";
        if (want_trellis && !set(trellis_env) && !zero(trellis_env) && !build.trellis_dense_r1)
            notice("trellis packs off (this build has no dense_r1 trellis library: CMake STRIX_DS41_TRELLIS_API=dense_r1)");
        if (!missing.empty()) {
            if (roots.empty()) notice("production packs off (no MODEL.gguf in the arguments)");
            else notice("production packs off (not found:" + missing + " in " + roots.front() + (roots.size() > 1 ? " or " + roots[1] : "") +
                        "; set STRIX_DS41_PACKS=DIR)");
        } else {
            if (!overlay.empty()) {
                put("STRIX_DS41_OVERLAY", overlay);
                put("STRIX_DS41_CB4", "1");
            }
            if (!k4.empty()) {
                put("STRIX_DS41_TRELLIS", k4);
                put("STRIX_DS41_TRELLIS_MIX", "head=" + k5);
                put("STRIX_DS41_TRELLIS_SITES", "all");
                put("STRIX_DS41_TRELLIS_DENSE_FULL", "1");
                put("STRIX_DS41_TRELLIS_HEADOUT", "1");
            }
        }
    }
    if (app == App::prefill) put("STRIX_DS41_EXPERT_TP", "0");

    std::vector<std::string> def;
    const auto transport = last_value(user, "--transport");
    const bool two_ranks = !transport.empty() && transport != "single";
    if (app == App::rank || app == App::batch) {
        if (two_ranks && !has(user, "--no-tp-attention")) def.emplace_back("--tp-attention");
        if (!has(user, "--engram-mapped")) def.emplace_back("--engram-direct");
    }
    if (app == App::serve) {
        def.insert(def.end(), {"--peer-timeout", "120", "--prefill-path", "device", "--slots", "8", "--prefill-rows", "auto"});
        const char* rows16 = env("STRIX_DS41_BATCH_ROWS16");
        if (has(user, "--no-dspark") || has(user, strix::session::speculation_opt_out_flag.data()) || has(user, "--dspark") ||
            model.empty()) {
        } else if (strix::session::speculation_environment_off(env(strix::session::speculation_environment))) {
            notice("serve DSpark off (STRIX_MTP=0)");
        } else if (rows16 != nullptr && rows16[0] == '1') {
            notice("serve DSpark off (not qualified with STRIX_DS41_BATCH_ROWS16=1)");
        } else if ([&] {
            ServeCachePolicy cache;
            for (std::size_t i = 0; i < user.size(); ++i) {
                const auto& key = user[i];
                std::string_view value;
                if (ServeCachePolicy::takes_value(key)) {
                    if (++i >= user.size()) throw std::invalid_argument(key + " needs a value");
                    value = user[i];
                }
                (void)cache.parse(key, value);
            }
            return cache.prefix_session && !cache.no_session && !cache.no_cache;
        }()) {
            notice("serve DSpark off (--prefix-session-snapshots is not exact with --dspark)");
        } else {
            std::string draft;
            for (const auto& root : std::vector<std::string>{parent_dir(model), roots.empty() ? std::string() : roots.front()}) {
                if (root.empty() || !draft.empty()) continue;
                auto names = probe.list(root);
                std::sort(names.begin(), names.end());
                for (const auto& n : names)
                    if (n.ends_with("-DSpark-MXFP4.gguf")) { draft = root + "/" + n; break; }
            }
            if (draft.empty()) {
                notice("serve DSpark off (no *-DSpark-MXFP4.gguf next to the model)");
            } else {
                def.insert(def.end(), {"--dspark", draft});
                const auto vocab = find("vocab", true);
                if (!has(user, "--dspark-vocab")) {
                    if (!vocab.empty() && probe.file(vocab + "/core48k.bin") && probe.file(vocab + "/core32k.bin"))
                        def.insert(def.end(), {"--dspark-vocab", "auto:" + vocab});
                    else notice("serve DSpark vocabulary full (no vocab/core48k.bin + core32k.bin in the packs)");
                }
            }
        }
    }
    // MODEL first (the documented form), the defaults, then the rest of the user's arguments (theirs win).
    if (!user.empty() && !user[0].empty() && user[0][0] != '-') {
        r.args.push_back(user[0]);
        r.args.insert(r.args.end(), def.begin(), def.end());
        r.args.insert(r.args.end(), user.begin() + 1, user.end());
    } else {
        r.args = def;
        r.args.insert(r.args.end(), user.begin(), user.end());
    }
    resolve_sessions(app, r, env, probe, false);
    return r;
}

std::string describe(App app, const Resolved& r) {
    std::ostringstream o;
    o << "app " << app_name(app) << '\n';
    if (r.profile) o << r.profile->describe();
    for (const auto& [k, v] : r.set_env) o << "env " << k << '=' << v << '\n';
    for (const auto& k : r.unset_env) o << "unset " << k << '\n';
    for (const auto& a : r.args) o << "arg " << a << '\n';
    o << "session cache " << (r.session.enabled ? "on" : "off") << "\n"
      << "session ram_bytes " << r.session.ram_bytes << "\n"
      << "session max_parked " << r.session.max_parked << "\n"
      << "session disk " << (r.session.disk ? "on" : "off") << "\n"
      << "session disk_bytes " << r.session.disk_bytes << "\n"
      << "session directory " << r.session.directory << "\n"
      << "session reason " << r.session.reason << "\n";
    for (const auto& n : r.notices) o << "notice " << n << '\n';
    return o.str();
}

Effective::Effective(App app, int& argc, char**& argv, const Build& build) {
    std::vector<std::string> in(argv + 1, argv + argc);
    Probe probe;
    probe.available_memory = []() -> std::uint64_t {
#ifdef _WIN32
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        return GlobalMemoryStatusEx(&status) ? status.ullAvailPhys : 0;
#else
        std::ifstream meminfo("/proc/meminfo");
        for (std::string key, unit; meminfo >> key;) {
            std::uint64_t kb = 0;
            meminfo >> kb; std::getline(meminfo, unit);
            if (key == "MemAvailable:") return kb * 1024U;
        }
        return 0;
#endif
    };
    probe.writable_space = [](const std::string& p) -> std::uint64_t {
        std::error_code ec;
        std::filesystem::create_directories(p, ec);
        if (ec) return 0;
        // Unique probe: never truncate an existing cache or another process's probe.
        const auto test = std::filesystem::path(p) / (".probe-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
        if (!std::filesystem::create_directory(test, ec) || ec) return 0;
        std::ofstream out(test / "write", std::ios::binary);
        out.put('x'); out.close();
        const bool ok = static_cast<bool>(out);
        std::filesystem::remove(test / "write", ec);
        std::filesystem::remove(test, ec);
        if (!ok || ec) return 0;
        const auto space = std::filesystem::space(p, ec);
        return ec ? 0 : space.available;
    };
    probe.file = [](const std::string& p) { std::error_code ec; return std::filesystem::is_regular_file(p, ec); };
    probe.dir = [](const std::string& p) { std::error_code ec; return std::filesystem::is_directory(p, ec); };
    probe.list = [](const std::string& d) {
        std::vector<std::string> out;
        std::error_code ec;
        for (std::filesystem::directory_iterator it(d, ec), end; !ec && it != end; it.increment(ec)) out.push_back(it->path().filename().string());
        return out;
    };
    auto r = resolve(app, in, [](const char* n) { return static_cast<const char*>(std::getenv(n)); }, probe, build);
    for (const auto& [k, v] : r.set_env) {
#ifdef _WIN32
        _putenv_s(k.c_str(), v.c_str());
#else
        ::setenv(k.c_str(), v.c_str(), 0);
#endif
    }
    for (const auto& k : r.unset_env) {
#ifdef _WIN32
        _putenv_s(k.c_str(), "");
#else
        ::unsetenv(k.c_str());
#endif
    }
    profile_json_storage() = r.profile ? r.profile->json() : std::string();
    profile_storage() = r.profile;
    if (r.print_only) {
        std::cout << describe(app, r) << ::strix::envcfg::describe();
        std::cout.flush();
        std::exit(0);
    }
    for (const auto& n : r.notices) std::cerr << n << '\n';
    // Receipt line on stderr (stdout stays the app's own, e.g. --describe-json); the serve also reports it in its
    // capabilities.
    std::cerr << "profile " << profile_json_storage() << '\n';
    storage_.reserve(r.args.size() + 1U);
    storage_.emplace_back(argv[0]);
    storage_.insert(storage_.end(), r.args.begin(), r.args.end());
    for (auto& s : storage_) ptrs_.push_back(s.data());
    ptrs_.push_back(nullptr);
    argc = static_cast<int>(storage_.size());
    argv = ptrs_.data();
}

}  // namespace strix::deepseek41::pair_defaults
