#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif
// MiniMax M3.1 defaults-on resolver, CPU only: the host invocation matches the measured production
// receipt (argv + env, tests/fixtures/minimax_production_receipt.txt), serve's routed default is checked separately, every off switch restores its item's former
// value, STRIX_MM_BASELINE=1 restores all of them, and apply() exports what --print-effective-config prints.
#include "minimax/engine/mm_defaults.hpp"
#include "strix/env_flags.hpp"
#include "strix/dist/transport_spec.hpp"
#include "strix/minimax/session_startup.hpp"

#include <algorithm>
#include <cstdlib>
#include <array>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>

#ifdef _WIN32
int setenv(const char* key, const char* value, int overwrite) { if (!overwrite && std::getenv(key)) return 0; return ::_putenv_s(key, value); }
int unsetenv(const char* key) { return ::_putenv_s(key, ""); }
#endif
namespace {

namespace md = strix::minimax::defaults;

int g_failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++g_failures;
        std::cerr << "FAIL: " << what << "\n";
    }
}

using EnvMap = std::map<std::string, std::string>;
md::Lookup lookup(const EnvMap& m) {
    return [&m](const char* name) -> const char* {
        const auto it = m.find(name);
        return it == m.end() ? nullptr : it->second.c_str();
    };
}

std::vector<std::string> words(const std::string& s) {
    std::istringstream in(s);
    std::vector<std::string> out;
    for (std::string w; in >> w;) out.push_back(w);
    return out;
}

struct Receipt {
    std::string app;
    std::vector<std::string> argv, plain;
    EnvMap env;
};

std::vector<Receipt> load(const char* path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    std::vector<Receipt> out;
    Receipt cur;
    bool open = false;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        const auto sp = line.find(' ');
        const std::string key = line.substr(0, sp), rest = sp == std::string::npos ? "" : line.substr(sp + 1);
        if (key == "receipt") {
            cur = Receipt{};
            cur.app = rest;
            open = true;
        } else if (key == "argv") cur.argv = words(rest);
        else if (key == "plain") cur.plain = words(rest);
        else if (key == "env") {
            const auto eq = rest.find('=');
            if (eq == std::string::npos) throw std::runtime_error("bad env line: " + line);
            cur.env[rest.substr(0, eq)] = rest.substr(eq + 1);
        } else if (key == "end" && open) {
            out.push_back(cur);
            open = false;
        } else throw std::runtime_error("bad fixture line: " + line);
    }
    return out;
}

std::set<std::string> lines(const std::string& text) {
    std::set<std::string> s;
    std::istringstream in(text);
    for (std::string l; std::getline(in, l);) s.insert(l);
    return s;
}

// The value each item replaced (measurement evidence), written out independently of the table.
const std::map<std::string, std::string> kFormer = {
    {"STRIX_MM_SERVE_DSPARK", "0"}, {"STRIX_MM_LOOKUP", "0"},
    {"STRIX_MM_SERVE_NEST", "0"}, {"STRIX_MM_SERVE_SCHED", "off"}, {"STRIX_MM_SERVE_SERVICE_MS", "0"}, {"STRIX_MM_SERVE_SHORT_FIRST", "0"},
    {"STRIX_MMX_MOE_WIDE", "0"},           {"STRIX_MM_INDEX_CONC", "0"},       {"STRIX_MMX_CRC", "1"},
    {"STRIX_MM_XB_WAIT", "0"},             {"STRIX_MM_XB_INPLACE", "0"},       {"STRIX_MM_XB_PUBLISH", "0"},
    {"STRIX_MM_XB_POSTILP", "0"},          {"STRIX_MM_XB_WARM", "0"},          {"STRIX_MM_DSPARK_ATTN", "0"},
    {"STRIX_MM_DRAFT_LIN", "lib"},         {"STRIX_MM_PF_MSA_BATCH", "0"},
    {"STRIX_MMX_LINEAR_V11", "0"},
    {"STRIX_MM_MARKOV", "full"},           {"STRIX_MM_MARKOV_C", "1024"},      {"STRIX_MM_MARKOV_H", "2048"},
    {"STRIX_MM_MARKOV_TOPC", "v1"},
    {"STRIX_MM_PF_ATTEND4", "0"},          {"STRIX_MM_PF_ROUTER_BATCH", "0"},  {"STRIX_MMX_PREFILL_MTFIT", "0"},
    {"STRIX_MM_PF_MSA_Q16", "0"},
    {"STRIX_MM_PF_MSA_REWORK", "0"},       {"STRIX_MMX_PREFILL_PIPE", "0"},    {"STRIX_MM_PF_KV_TILE", "off"},
    {"STRIX_MM_PF_ASYNC_START", "0"}, {"STRIX_MM_PF_LATE_SRC_FENCE", "0"},
    {"STRIX_MM_PF_TX_UNCACHED", "0"},      {"STRIX_DIST_CRC_PARALLEL", "0"},   {"STRIX_DIST_HOST_CRC_OVERLAP", "0"},
    {"STRIX_DIST_HOST_CHUNK_BYTES", "0"},  {"STRIX_MM_PFX_JOINED", "0"},       {"STRIX_MM_PF_IDX_TILE", "0"},
    {"STRIX_MM_PF_IDX_ROWS", "32"},        {"STRIX_MMX_MOE_GU_INTERLEAVE", "0"}, {"STRIX_MMX_MOE_SINGLE", "0"},
    {"STRIX_MMX_MOE_DOWN_SPLIT", "0"},    {"STRIX_MM_ATTEND_DIET", "0"},
    {"STRIX_MM_ATTEND4_DECODE", "0"},      {"STRIX_MM_SELECT_ONCE", "0"},
    {"STRIX_MMX_PREFILL_GATHER_TOK", "0"}, {"STRIX_MM_PF_DENSE_QKV", "0"},
    {"STRIX_MMX_PREFILL_TAIL_STREAM", "0"}, {"STRIX_MMX_PREFILL_PIPE_PAIR", "0"}, {"STRIX_MMX_PREFILL_PIPE_TAIL", "0"},
    {"STRIX_MM_PF_ROUTER_ROWS", "8"}, {"STRIX_MM_PF_MSA_SCRATCH_MB", "512"}, {"STRIX_MM_PF_DENSE_O", "0"},
    {"STRIX_MM_XB_WARM_O", "0"},           {"STRIX_MMX_MOE_TIGHT_GRID", "0"},
    {"STRIX_MMX_MOE_PIPE", "0"},
    {"STRIX_MM_PF_HEADLESS", "0"},
    {"STRIX_MM_PF_DENSE_MLP", "0"},         {"STRIX_MMX_PF_ORDER", "adaptive"},
    {"STRIX_MM_PF_FLASH_KEYS", "0"},         {"STRIX_MM_PF_MSA_RG_ROWS", "4"},
};
// Cache identity field of the default chunk-invariant linear order (table item pf_order).
const std::string kOrderId = "prefill_order=mmx-pf-linear-s1-v1";

void test_table() {
    static_assert(std::is_same_v<md::Item, strix::profile::DefaultItem>);
    std::set<std::string> envs, names;
    for (const md::Item& it : md::items()) {
        check(envs.insert(it.env).second, std::string("unique env ") + it.env);
        check(names.insert(it.name).second, std::string("unique name ") + it.name);
        check(std::strcmp(it.on, it.off) != 0, std::string("on != off for ") + it.env);
        check(std::strlen(it.receipt) > 10, std::string("receipt for ") + it.env);
        check(it.app_sets && it.in_baseline, std::string("export and baseline membership for ") + it.env);
        const auto f = kFormer.find(it.env);
        check(f != kFormer.end() && f->second == it.off, std::string("off value is the former path for ") + it.env);
    }
    check(envs.size() == kFormer.size(), "every former value has an item");
    check(envs.count("STRIX_DIST_ASYNC_START") == 0, "MiniMax must not export a global async default");
}

void test_receipts(const char* path) {
    const auto receipts = load(path);
    check(receipts.size() == 2, "fixture has the rank and serve receipts");
    const EnvMap none;
    for (const Receipt& r : receipts) {
        const std::string want = md::effective_config(md::parse_cli(r.app, r.argv), lookup(r.env));
        auto plain = r.plain;
        // Preserve the measured host receipt verbatim. The new serve default
        // is a transport policy, with native qualification still pending;
        // compare its explicit host restoration to the historical receipt.
        if (r.app == "serve") { plain.push_back("--transport"); plain.push_back("usb4-auto"); }
        const std::string got = md::effective_config(md::parse_cli(r.app, plain), lookup(none));
        auto numerical = [](const std::string& text) {
            std::istringstream input(text); std::string result, line;
            // The order's request line shows how it was asked (receipt: explicit; plain: "default"); the effective line
            // and the item value are compared.
            while (std::getline(input,line))
                if (!line.starts_with("session_") && !line.starts_with("STRIX_MMX_PF_ORDER.requested=")) result += line + "\n";
            return result;
        };
        check(numerical(got) == numerical(want), r.app + ": host configuration == receipt\n--- receipt\n" + want + "--- host\n" + got);
        // Every receipt env value is one of the resolved items (no receipt switch is left unmanaged).
        for (const auto& [k, v] : r.env) {
            bool found = false;
            for (const md::Item& it : md::items()) found = found || (k == it.env && v == it.on);
            check(found, r.app + ": receipt env " + k + "=" + v + " is an item's default");
        }
    }
}

// Execution residency is independent of the hypothetical 8K FAST capacity/timing estimate.
void test_pair_pipeline_declaration() {
    const std::string actual = "prefill.pair_pipeline=off mode=tensor residency=tensor_split free_memory=pending "
        "reason=layout: tensor-split weights; pipeline execution unimplemented";
    const std::string reference = "prefill.pair_pipeline_reference=estimate profile=FAST rows=8192 "
        "assumed_free_gb=31.0 assumed_extra_gb=41.6 pair.pipeline mode=tensor chunk=";
    const std::string reason = "reason=layout: src/minimax/engine/engine.hip:89-91,1120-1131 needs +41.6 GB/node, free 31.0 GB";
    const char* apps[2] = {"rank", "serve"};
    for (const char* fast : {"0", "1"}) {
        const EnvMap env{{"STRIX_MM_FAST_PREFILL", fast}};
        std::string estimates[2];
        for (int i = 0; i < 2; ++i) {
            const std::vector<std::string> args = i == 0 ? std::vector<std::string>{"model", "0", "tcp", "out"} : std::vector<std::string>{};
            std::string row;
            for (const auto& line : lines(md::effective_config(md::parse_cli(apps[i], args), lookup(env)))) {
                if (line.rfind("prefill.pair_pipeline=", 0) == 0) row = line;
                if (line.rfind("prefill.pair_pipeline_reference=", 0) == 0) estimates[i] = line;
            }
            check(row == actual, std::string(apps[i]) + ": execution stays tensor with memory pending, FAST=" + fast);
            check(estimates[i].rfind(reference, 0) == 0 && estimates[i].find(reason) != std::string::npos,
                std::string(apps[i]) + ": reference labels its profile, rows and assumed memory, FAST=" + fast);
        }
        check(estimates[0] == estimates[1], "rank and serve format the same reference estimate");
    }
    const auto at8k = md::pair_pipeline_choice(8192U);
    check(at8k.pipeline_ms > 9500.0 && at8k.pipeline_ms < 9700.0 && at8k.alt_ms > 9809.0 && at8k.alt_ms < 9811.0,
        "MiniMax 8K reference estimate: pipeline ~9.6 s vs TP2 9.81 s");
    const auto at32k = md::pair_pipeline_choice(32768U);
    check(at32k.mode == strix::pair::Mode::tensor && "reason=" + at32k.reason == reason,
        "32K reference stays TP2 because the assumed layout does not fit");
}

void test_off_switches() {
    const EnvMap none;
    const auto on_lines = lines(md::effective_config(md::Cli{}, lookup(none)));
    for (const md::Item& it : md::items()) {
        const EnvMap off = {{it.env, it.off}};
        const auto r = md::resolve(lookup(off));
        for (const md::Resolved& x : r) {
            if (x.item == &it) check(x.value == it.off && x.source == md::Source::env, std::string(it.env) + " off");
            else check(x.value == x.item->on, std::string(it.env) + " off leaves " + x.item->env + " on");
        }
        for (const char* app : {"rank", "serve"}) {
            md::Cli cli; cli.app = app;
            const auto off_lines = lines(md::effective_config(cli, lookup(off)));
            check(off_lines.count(std::string(it.env) + "=" + it.off) == 1, std::string(app) + ": " + it.env + " off printed");
            if (cli.app == "rank") {
                std::size_t diff = 0;
                for (const auto& l : off_lines) if (!l.starts_with("session_")) diff += on_lines.count(l) == 0 ? 1 : 0;
                // The linear order also prints its request / effective inspection and names its cache identity, so its
                // off form changes those three lines too.
                const std::size_t want = std::strcmp(it.env, "STRIX_MMX_PF_ORDER") == 0 ? 4 : 1;
                check(diff == want, std::string(it.env) + " off changes one rank line (plus inspection + identity for the order)");
            }
        }
    }
}

void test_resolution_sources() {
    const EnvMap none;
    for (const auto& x : md::resolve(lookup(none)))
        check(x.value == x.item->on && x.source == md::Source::on, std::string("default source: ") + x.item->env);
    // Every row independently: explicit ON, OFF, empty and arbitrary strings
    // override baseline; validation/admission belongs to the existing readers.
    for (const auto& it : md::items()) {
        for (const char* value : {it.on, it.off, "", "explicit-host-value"}) {
            for (const bool base : {false, true}) {
                EnvMap env{{it.env, value}};
                if (base) env[md::kBaselineEnv] = "1";
                for (const auto& x : md::resolve(lookup(env))) {
                    if (x.item == &it)
                        check(x.value == value && x.source == md::Source::env, std::string("explicit source: ") + it.env);
                    else
                        check(x.value == (base ? x.item->off : x.item->on) &&
                              x.source == (base ? md::Source::baseline : md::Source::on), std::string("isolated source: ") + x.item->env);
                }
            }
        }
    }
}

void test_host_receipt() {
    const EnvMap none, base{{md::kBaselineEnv, "1"}}, fp32{{"STRIX_MM_PREFILL_IDENTITY", "fp32"}},
        other{{"STRIX_MM_PREFILL_IDENTITY", "FP32"}};
    const auto receipt = md::host_receipt(lookup(none));
    check(receipt.engine == "minimax-m3.1" && receipt.implementation == "minimax-pair", "host receipt engine route");
    check(receipt.build == "gu4_scalar=on" || receipt.build == "gu4_scalar=off", "host receipt compiled addressing switch");
    check(receipt.identity.arithmetic_contract == "MiniMax M3.1 FP32 partials with fixed rank-order sum", "separate MiniMax arithmetic contract");
    check(receipt.identity.prefill_lineage == "production", "production prefill lineage");
    check(receipt.identity.model_sha256.empty() && receipt.identity.pack_hash.empty() && receipt.identity.table_hash.empty() &&
          receipt.identity.state_format.empty() && receipt.identity.reduction_plan.empty(), "unknown numerical identities stay empty");
    check(receipt.kernel.empty() && receipt.device.empty() && receipt.topology.empty() && receipt.digests.empty(), "no host execution/device/digest claims");
    check(receipt.json() == md::host_receipt(lookup(base)).json(), "baseline does not invent another arithmetic contract");
    check(receipt.identity.id() != md::host_receipt(lookup(fp32)).identity.id(), "FP32 prefill has a separate lineage");
    check(md::host_receipt(lookup(fp32)).identity.prefill_lineage == "fp32", "explicit FP32 lineage");
    check(receipt.json() == md::host_receipt(lookup(other)).json(), "prefill selector remains case sensitive");
    const auto json = receipt.json();
    check(json.find("\"model_sha256\":\"\"") != std::string::npos && json.find("\"device\":\"\"") != std::string::npos,
          "unknown identities serialize as empty");
    check(receipt.witness() == "strix-engine-witness " + json + "\n", "canonical witness prefix and payload");
}

void test_baseline() {
    for (const char* v : {"1", "yes"}) {
        const EnvMap base = {{md::kBaselineEnv, v}};
        check(md::baseline(lookup(base)), std::string("baseline on for ") + v);
        for (const md::Resolved& x : md::resolve(lookup(base)))
            check(x.value == x.item->off && x.source == md::Source::baseline, std::string("baseline: ") + x.item->env + " off");
    }
    for (const char* v : {"0", ""}) {
        const EnvMap base = {{md::kBaselineEnv, v}};
        check(!md::baseline(lookup(base)), std::string("baseline off for '") + v + "'");
        for (const md::Resolved& x : md::resolve(lookup(base))) check(x.value == x.item->on, "baseline=0 keeps defaults");
    }
    // An explicit item value wins over the master switch.
    const md::Item& first = md::items().front();
    const EnvMap mixed = {{md::kBaselineEnv, "1"}, {first.env, first.on}};
    for (const md::Resolved& x : md::resolve(lookup(mixed)))
        check(x.item == &first ? x.value == first.on : x.value == x.item->off, std::string("baseline + explicit ") + x.item->env);
}

void test_async_group() {
    const auto minimum = md::pf_async_min_rows(nullptr);
    check(minimum == 2048, "async minimum defaults to 2048");
    for (const auto& rows : {std::array<std::size_t, 2>{4096, 3844}, {2048, 2048}, {4096, 2048}}) {
        check(md::pf_async_group(true, rows, minimum), "every member meets inclusive threshold");
        check(!md::pf_async_group(false, rows, minimum), "global off overrides qualifying group");
    }
    for (const auto& rows : {std::array<std::size_t, 2>{4096, 443}, {443, 4096}, {4096, 2047}, {2047, 4096}}) {
        check(!md::pf_async_group(true, rows, minimum), "any small member rejects entire group");
        check(md::pf_async_group(true, rows, md::pf_async_min_rows("0")), "zero restores unconditional async");
        check(!md::pf_async_group(false, rows, 0), "global off overrides zero threshold");
    }
    const std::array<std::size_t, 4> four{4096, 3844, 4096, 443};
    check(!md::pf_async_group(true, four, minimum), "four-member group checks final member");
    check(md::pf_async_group(true, std::span(four).first(3), minimum), "only active members count");
    check(!md::pf_async_group(true, {}, 0), "empty group is not async work");
    check(!md::pf_async_group(true, std::span(four).first(1), md::pf_async_min_rows("4097")), "custom threshold");
    for (const char* value : {"", "-1", "+1", "1x", " 2048", "184467440737095516160"}) {
        bool rejected = false;
        try { (void)md::pf_async_min_rows(value); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, std::string("invalid async minimum: ") + value);
    }
    const EnvMap none, custom{{"STRIX_MM_PF_ASYNC_MIN_ROWS", "0"}};
    check(lines(md::effective_config(md::Cli{}, lookup(none))).count("STRIX_MM_PF_ASYNC_MIN_ROWS=2048") == 1,
          "inspection reports default threshold");
    check(lines(md::effective_config(md::Cli{}, lookup(custom))).count("STRIX_MM_PF_ASYNC_MIN_ROWS=0") == 1,
          "inspection reports custom threshold");
}
void test_session_inspection() {
    const auto directory = std::filesystem::current_path() / "minimax-inspection-fixture";
    check(!std::filesystem::exists(directory), "inspection fixture absent");
    for (const char* app : {"rank", "serve"}) {
        md::Cli cli; cli.app = app;
        for (const char* mode : {"auto", "off", "0", "on", "1"}) {
            EnvMap env{{"STRIX_SESSION_CACHE", mode}, {"STRIX_SESSION_DIR", directory.string()}};
            const auto clean = md::effective_config(cli, lookup(env));
            env["HOME"] = "alternate-home"; env["XDG_CACHE_HOME"] = "alternate-cache";
            env["USERPROFILE"] = "alternate-profile";
            const auto guarded = [&](const char* key) -> const char* {
                check(std::string(key) != "HOME" && std::string(key) != "XDG_CACHE_HOME" &&
                      std::string(key) != "USERPROFILE", "inspection never resolves a host home");
                return lookup(env)(key);
            };
            check(md::effective_config(cli, guarded) == clean, "inspection independent of host paths");
            check(!std::filesystem::exists(directory), "inspection creates no files or directories, including ON");
            std::istringstream input(clean); std::string line; unsigned count = 0;
            while (std::getline(input, line)) if (line.starts_with("session_")) {
                check(line.starts_with("session_requested=") || line.starts_with("session_cache=") ||
                      line.starts_with("session_reason="), "inspection contains only host-independent session policy");
                ++count;
            }
            check(count == 3, "three documented session policy lines");
            env[md::kBaselineEnv] = "1";
            const auto base = md::effective_config(cli, guarded);
            check(base.find("session_cache=off\nsession_reason=baseline\n") != std::string::npos,
                  "baseline inspection is off");
        }
    }
}
void test_session_defaults() {
    // Exercise the same shared negotiated-name helper as the serving app.
    for (const bool direct_live : {false, true}) {
        auto negotiated = md::parse_cli("serve", {});
        check(negotiated.transport == "usb4gpu", "serve requests GPU transport by default");
        negotiated.transport = strix::dist::negotiated_transport(negotiated.transport,
            direct_live, md::kRouteHost, md::kRouteGpu);
        for (const EnvMap& request : {EnvMap{}, EnvMap{{"STRIX_SESSION_CACHE", "1"}}}) {
            for (const bool explicit_cli : {false, true}) {
                const auto policy = md::session_policy(negotiated, lookup(request), explicit_cli);
                check(policy.route == (direct_live ? "serve/usb4gpu/fixed-width-one-session-v1"
                                                  : "serve/usb4-auto/fixed-width-one-session-v1"),
                      "session route follows negotiated transport");
                check(policy.enabled == !direct_live, "only negotiated host serving is cache eligible");
                check(policy.reason == (direct_live ? "route not yet GPU-qualified: " + policy.route : ""),
                      "unqualified direct route supplies one clear OFF reason");
                if (!explicit_cli) {
                    const auto runtime = md::session_config(negotiated, lookup(request), 100ULL << 30);
                    check(runtime.enabled == !direct_live && runtime.route == policy.route,
                          "runtime admission retains negotiated eligibility");
                }
            }
        }
    }
    md::Cli cli; cli.app = "serve";
    for (const EnvMap& off : {EnvMap{{"STRIX_SESSION_CACHE", "0"}},
                             EnvMap{{"STRIX_SESSION_CACHE", "off"}}, EnvMap{{"STRIX_MM_BASELINE", "1"}},
                             EnvMap{{"STRIX_BASELINE", "1"}, {"STRIX_SESSION_CACHE", "1"}}}) {
        const auto policy = md::session_policy(cli, [&](const char* key) -> const char* {
            const std::string name(key);
            check(name != "HOME" && name != "USERPROFILE" && name != "XDG_CACHE_HOME" &&
                  name != "STRIX_SESSION_DIR" && name != "STRIX_SESSION_RAM_BYTES" &&
                  name != "STRIX_SESSION_DISK_BYTES", "OFF startup does not resolve resources");
            return lookup(off)(key);
        });
        check(!policy.enabled && policy.ram_bytes == 0 && policy.directory.empty(), "OFF startup uses pure policy");
    }
    {
        md::Cli rank; rank.app = "rank";
        const auto policy = md::session_policy(rank, lookup(EnvMap{}));
        check(!policy.enabled && policy.reason == "route not yet GPU-qualified: rank/usb4-auto/fixed-width-one-session-v1" &&
              policy.ram_bytes == 0 && policy.directory.empty(), "unqualified rank route stays off with pure policy");
    }
    EnvMap env{{"STRIX_SESSION_DISK","0"},{"HOME","/test-home"}};
    auto c = md::session_config(cli, lookup(env), 100ULL << 30);
    check(c.enabled && c.reason.empty() && c.route == "serve/usb4-auto/fixed-width-one-session-v1" &&
          c.ram_bytes == (8ULL << 30) && c.max_parked == 8 && !c.disk, "qualified serve route admits by default");
    {
        md::Cli tcp = cli; tcp.transport = "tcp-listen:9000";
        const auto other = md::session_config(tcp, lookup(env), 100ULL << 30);
        check(!other.enabled && other.reason == "route not yet GPU-qualified: serve/tcp-listen:9000/fixed-width-one-session-v1",
              "unqualified transport declines");
    }
    {
        EnvMap no_disk_switch{{"HOME", "/test-home"}};
        const auto automatic = md::session_config(cli, lookup(no_disk_switch), 100ULL << 30);
        check(automatic.enabled && !automatic.disk, "automatic serving parks in RAM only unless the disk tier is requested");
    }
    env["STRIX_SESSION_CACHE"] = "1";
    check(md::session_config(cli,lookup(env),100ULL<<30).enabled, "explicit qualification gate opt-in");
    env.erase("STRIX_SESSION_CACHE");
    check(c.directory == "/test-home/.cache/strix-engine/sessions", "default disk path");
    env["XDG_CACHE_HOME"] = "/test-xdg";
    check(md::session_config(cli,lookup(env),100ULL<<30).directory == "/test-xdg/strix-engine/sessions", "XDG path");
    for (const char* key : {"STRIX_SESSION_CACHE","STRIX_BASELINE", md::kBaselineEnv}) {
        env[key] = std::string(key) == "STRIX_SESSION_CACHE" ? "0" : "1";
        check(!md::session_config(cli,lookup(env),100ULL<<30).enabled, std::string("session off switch ") + key);
        env.erase(key);
    }
    cli.capacity = 122880;
    c = md::session_config(cli,lookup(env),100ULL<<30);
    check(!c.enabled && !c.reason.empty(), "120K refuses default budget cleanly");
    cli.capacity = 8256;
    check(!md::session_config(cli,lookup(env),2ULL<<30).enabled, "MemAvailable guard");
    {
        // Two-node serve node after the model load: ~31 GB available admits the default 8256-token context.
        md::Cli production; production.app = "serve";
        const auto node = md::session_config(production, lookup(env), 31ULL << 30);
        check(node.enabled && node.ram_bytes == (31ULL << 30) / 4, "default context fits a quarter of post-load MemAvailable");
    }
    env["STRIX_SESSION_RAM_BYTES"] = "7000000000"; env["STRIX_SESSION_MAX_PARKED"] = "3";
    c = md::session_config(cli,lookup(env),100ULL<<30);
    check(c.enabled && c.ram_bytes == 7000000000 && c.max_parked == 3, "budget overrides");
    const auto directory = std::filesystem::current_path() / "minimax-explicit-defaults-fixture";
    check(!std::filesystem::exists(directory), "explicit test directory fresh");
    env["STRIX_SESSION_CACHE"] = "0";
    env[md::kBaselineEnv] = "1";
    env["STRIX_SESSION_DIR"] = "invalid-default-cache";
    c = md::session_config(cli,lookup(env),100ULL<<30,true,directory.string());
    check(c.enabled && c.disk && c.directory == directory.string(), "explicit disk independent of automatic switches and path");
    env["STRIX_SESSION_DISK_BYTES"] = "1";
    check(!md::session_config(cli,lookup(env),100ULL<<30,true,directory.string()).disk, "explicit disk cap retained");
    env.erase("STRIX_SESSION_DISK_BYTES");
    check(!md::session_config(cli,lookup(env),2ULL<<30,true,directory.string()).enabled, "explicit memory guard retained");
    std::filesystem::remove_all(directory);
    cli.slots = 1;
    check(!md::session_config(cli,lookup(env),100ULL<<30).enabled, "transaction slot precondition");
}
// Execute the same serving admission path with a CPU factory. A baseline must
// reach ordinary serving without constructing a runtime; explicit ON failures
// must reach the server-style nonzero exit handler.
void test_session_startup() {
    md::Cli cli; cli.app = "serve"; cli.capacity = 32;
    for (const char* baseline : {"STRIX_MM_BASELINE", "STRIX_BASELINE"}) {
        for (const char* on : {"1", "on"}) {
            EnvMap env{{baseline, "1"}, {"STRIX_SESSION_CACHE", on}, {"STRIX_SESSION_DISK", "0"}};
            const auto config = md::session_config(cli, lookup(env), 100ULL << 30);
            bool attempted = false, served = false;
            std::string log;
            auto serve = [&] {
                try {
                    auto runtime = strix::minimax::session::admit_runtime(config.enabled, true, config.reason,
                        [&]() -> std::unique_ptr<int> { attempted = true; throw std::runtime_error("injected failure"); },
                        [&](const std::string& line) { log = line; });
                    check(!runtime, "baseline runtime is absent");
                    served = true;
                    return 0;
                } catch (const std::exception&) { return 2; }
            };
            check(!config.enabled && config.reason == "baseline", "baseline resolves sessions OFF over explicit ON");
            check(serve() == 0 && served && !attempted, "baseline ordinary serving continues and exits zero");
            check(log == "session cache: off (baseline)", "baseline diagnostic");
        }
    }
    EnvMap env{{"STRIX_SESSION_CACHE", "1"}, {"STRIX_SESSION_DISK", "0"}};
    auto config = md::session_config(cli, lookup(env), 100ULL << 30);
    bool served = false;
    std::string log;
    int exit = 0;
    try {
        (void)strix::minimax::session::admit_runtime(config.enabled, true, config.reason,
            []() -> std::unique_ptr<int> { throw std::runtime_error("injected failure"); },
            [&](const std::string& line) { log = line; });
        served = true;
    } catch (const std::exception&) { exit = 2; }
    check(config.enabled && exit == 2 && !served, "explicit ON initialization failure exits nonzero");
    check(log == "session cache: initialization failed (injected failure)", "explicit ON failure diagnostic");
}

void test_cli() {
    const auto r = md::parse_cli("rank", {"M", "1", "tcp-listen:9000", "OUT", "--slots", "2", "--capacity", "100", "--max-rows", "16"});
    check(r.transport == "tcp-listen:9000" && r.slots == 2 && r.capacity == 100 && r.max_rows == 16, "rank cli");
    const auto s = md::parse_cli("serve", {"M", "--transport", "usb4gpu", "--slots", "4"});
    check(s.app == "serve" && s.transport == "usb4gpu" && s.slots == 4 && s.capacity == 8256 && s.max_rows == 4096, "serve cli");
    const auto override_cli = md::parse_cli("serve", {"M", "--dspark-mode", "on"});
    const EnvMap bad_mode = {{"STRIX_MM_SERVE_DSPARK", "invalid"}, {md::kBaselineEnv, "1"}};
    check(lines(md::effective_config(override_cli, lookup(bad_mode))).count("serve.dspark_mode=on") == 1,
          "CLI overrides env and baseline before validation");
    const auto d = md::parse_cli("serve", {"M"});
    check(d.transport == "usb4gpu" && d.slots == 8, "serve routed transport default");
    check(md::parse_cli("serve", {"M", "--transport", "usb4-auto"}).transport == "usb4-auto", "serve host override");
    const EnvMap none;
    for (const char* session : {"named:prompt.u32:16", "./prompt.u32", "fast", "default"}) {
        // Defaults inspection never interprets --profile, including strings
        // understood as product profiles by other engines. The app owns it.
        const auto timed = md::parse_cli("rank", {"M", "0", "usb4-auto", "OUT", "--profile", session});
        check(md::effective_config(timed, lookup(none)) == md::effective_config(md::Cli{}, lookup(none)),
              std::string("legacy timed-session --profile left to app: ") + session);
    }
}

void test_serve_preflight() {
    const auto cli = md::parse_cli("serve", {"M"});
    const EnvMap none;
    const auto defaults = lines(md::effective_config(cli, lookup(none)));
    for (const char* line : {"serve.sched=interleave", "serve.service_ms=250", "serve.gap_ms=1500",
                             "serve.short_first=1", "serve.nest=1", "serve.dspark_mode=auto", "serve.dspark_effective=per_request_if_assets", "serve.dspark_prior=structural-v1", "dspark.prior=structural-v1", "serve.dspark_inflight=1"})
        check(defaults.count(line) == 1, std::string("resolved policy: ") + line);
    const EnvMap prior_env = {{"STRIX_MM_DSPARK_PRIOR", "unqualified-v1"}};
    check(lines(md::effective_config(cli, lookup(prior_env))).count("serve.dspark_reason=plain:prior_unqualified") == 1, "prior env fallback");
    for (const char* app : {"serve", "rank"}) {
        auto c = md::Cli{}; c.app = app; c.dspark_prior = "structural-v1";
        const EnvMap invalid = {{"STRIX_MM_DSPARK_PRIOR", "invalid"}};
        check(lines(md::effective_config(c, lookup(invalid))).count("dspark.prior=structural-v1") == 1, "prior CLI overrides invalid env");
        c.dspark_prior.clear(); bool rejected = false;
        try { (void)md::effective_config(c, lookup(invalid)); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "both apps reject invalid prior");
    }
    check(md::parse_cli("rank", {"M", "0", "usb4-auto", "O", "--dspark-prior", "unqualified-v1"}).dspark_prior == "unqualified-v1", "rank prior CLI");
    const EnvMap custom = {{"STRIX_MM_SERVE_SCHED", "0"}, {"STRIX_MM_SERVE_SERVICE_MS", "0"},
                          {"STRIX_MM_SERVE_GAP_MS", "100"}, {"STRIX_MM_SERVE_SHORT_FIRST", "0"},
                          {"STRIX_MM_SERVE_NEST", "0"}};
    const auto printed = lines(md::effective_config(cli, lookup(custom)));
    for (const char* line : {"serve.sched=off", "serve.service_ms=0", "serve.gap_ms=100",
                             "serve.short_first=0", "serve.nest=0"})
        check(printed.count(line) == 1, std::string("custom policy: ") + line);
    check(printed.count("serve.gap_ms=1500") == 0, "different deadlines have different config identity");
    const EnvMap base = {{md::kBaselineEnv, "1"}};
    const auto baseline = lines(md::effective_config(cli, lookup(base)));
    check(baseline.count("serve.sched=off") == 1 && baseline.count("serve.service_ms=0") == 1 &&
          baseline.count("serve.short_first=0") == 1 && baseline.count("serve.nest=0") == 1 &&
          baseline.count("serve.gap_ms=1500") == 1, "baseline policy resolves before export");
    for (const char* name : {"STRIX_MM_SERVE_SCHED", "STRIX_MM_SERVE_SERVICE_MS", "STRIX_MM_SERVE_GAP_MS",
                             "STRIX_MM_SERVE_SHORT_FIRST", "STRIX_MM_SERVE_NEST", "STRIX_MM_SERVE_DSPARK", "STRIX_MM_SERVE_DSPARK_INFLIGHT"}) {
        std::vector<std::string> invalid = {"nan", "inf", "1e999", "4294967296", "-1", "invalid"};
        if (std::strcmp(name, "STRIX_MM_SERVE_GAP_MS") == 0) invalid.push_back("0");
        for (const auto& value : invalid) {
            const EnvMap env = {{name, value}};
            bool rejected = false;
            try { (void)md::effective_config(cli, lookup(env)); }
            catch (const std::invalid_argument& e) { rejected = std::string(e.what()).find(name) != std::string::npos; }
            check(rejected, std::string("serve preflight rejects ") + name + "=" + value);
            // Serving-only policy must not reject a standalone rank invocation.
            (void)md::effective_config(md::Cli{}, lookup(env));
        }
    }
}

// Last: changes the process environment.
void test_apply() {
    (void)::setenv("STRIX_SESSION_DISK", "0", 1); // CPU checks never create user cache directories.
    for (const md::Item& it : md::items()) (void)::unsetenv(it.env);
    (void)::unsetenv(md::kBaselineEnv);
    (void)::setenv("STRIX_MM_XB_PUBLISH", "0", 1);  // a user value
    char flag[] = "--print-effective-config", app[] = "strix-minimax-rank", m[] = "M", r0[] = "0", tr[] = "usb4-auto", out[] = "OUT";
    char* argv[] = {app, m, r0, tr, out, flag};
    check(md::print_or_apply("rank", 6, argv), "print_or_apply with the flag");
    for (const md::Item& it : md::items()) {
        const char* v = std::getenv(it.env);
        if (std::strcmp(it.env, "STRIX_MM_XB_PUBLISH") == 0) {
            check(v != nullptr && std::strcmp(v, "0") == 0, "user value kept");
            check(!md::from_defaults(it.env), "user value is not from the defaults");
        } else {
            check(v != nullptr && std::strcmp(v, it.on) == 0, std::string("apply exported ") + it.env);
            check(md::from_defaults(it.env), std::string("from_defaults ") + it.env);
        }
    }
    // Shared recorder: a read of an exported default records as default; a user value stays "set" (receipt switches).
    for (const char* name : {"STRIX_MM_PF_MSA_REWORK", "STRIX_MM_XB_PUBLISH"}) {
        (void)strix::envcfg::text(name);
        bool found = false, set = false;
        for (const auto& e : strix::envcfg::registered())
            if (e.name == name) { found = true; set = e.set; }
        const bool user = std::strcmp(name, "STRIX_MM_XB_PUBLISH") == 0;
        check(found && set == user, std::string("envcfg records exported default as default, user value as set: ") + name);
    }
    // Exercise the actual app entry with and without inspection, for both rank arguments.
    for (const char* name : {"STRIX_MM_SERVE_SCHED", "STRIX_MM_SERVE_SERVICE_MS", "STRIX_MM_SERVE_GAP_MS",
                             "STRIX_MM_SERVE_SHORT_FIRST", "STRIX_MM_SERVE_NEST", "STRIX_MM_SERVE_DSPARK", "STRIX_MM_SERVE_DSPARK_INFLIGHT"}) {
        const char* old = std::getenv(name);
        const std::string saved = old == nullptr ? "" : old;
        const bool had = old != nullptr;
        (void)::setenv(name, "nan", 1);
        for (char rank : {'0', '1'}) {
            char rank_value[] = {rank, 0};
            char rank_flag[] = "--rank";
            char* serve_argv[] = {app, m, rank_flag, rank_value, flag};
            for (int count : {4, 5}) {
                bool rejected = false;
                try { (void)md::print_or_apply("serve", count, serve_argv); }
                catch (const std::invalid_argument&) { rejected = true; }
                check(rejected, std::string("app preflight rejects ") + name + " on rank " + rank);
            }
        }
        if (had) (void)::setenv(name, saved.c_str(), 1);
        else (void)::unsetenv(name);
    }
    char* argv2[] = {app, m, r0, tr, out};
    check(!md::print_or_apply("rank", 5, argv2), "print_or_apply without the flag");
    // Both argv orders resolve OFF. --session-cache is the CLI form of ON.
    char baseline[] = "--baseline", cache[] = "--session-cache", on[] = "on";
    char* baseline_first[] = {app, m, baseline, cache, on};
    char* baseline_last[] = {app, m, cache, on, baseline};
    for (auto args : {baseline_first, baseline_last}) {
        (void)::unsetenv(md::kBaselineEnv);
        check(!md::print_or_apply("serve", 5, args), "baseline ordinary startup");
        auto config = md::session_config(md::Cli{}, [](const char* key) { return std::getenv(key); }, 100ULL << 30);
        check(!config.enabled && config.reason == "baseline", "--baseline wins in either argv order");
        std::string log;
        auto runtime = strix::minimax::session::admit_runtime(config.enabled, true, config.reason,
            []() -> std::unique_ptr<int> { throw std::runtime_error("must not initialize"); },
            [&](const std::string& line) { log = line; });
        check(!runtime && log == "session cache: off (baseline)", "--baseline continues ordinary serving");
    }
}

// TURBO product profile: '--profile turbo' of rank and serve (exported as STRIX_MM_PRODUCT_PROFILE by print_or_apply)
// resolves once; its cache identity is minimax-turbo-v1 even with every row off; the timing session moved to
// --timing-profile with the old spelling kept; an unhonoured weight request is visible and refused.
void test_turbo_product_profile() {
    const EnvMap plain, turbo{{md::kProductProfileEnv, "turbo"}};
    const auto pr = md::profile_resolution(lookup(turbo));
    check(pr.is_turbo() && pr.versioned() == "minimax-turbo-v1", "turbo: versioned name");
    check(md::prefill_identity(lookup(plain)) == kOrderId, "turbo: default cache identity unchanged");
    check(md::prefill_identity(lookup(turbo)).find("turbo=minimax-turbo-v1") != std::string::npos,
          "turbo: cache identity carries the turbo namespace with every row off");
    check(pr.fingerprint() != md::profile_resolution(lookup(plain)).fingerprint(), "turbo: ranks never agree with default");
    const md::Cli rank = md::parse_cli("rank", {"/m", "0", "usb4gpu", "/o"}), serve = md::parse_cli("serve", {"/m"});
    const auto rank_text = md::effective_config(rank, lookup(turbo)), serve_text = md::effective_config(serve, lookup(turbo));
    const auto profile_lines = [](const std::string& text) {
        std::string out;
        std::size_t at = 0;
        while ((at = text.find("profile.", at)) != std::string::npos) {
            const auto end = text.find('\n', at);
            out += text.substr(at, end - at) + "|";
            at = end;
        }
        return out;
    };
    check(!profile_lines(rank_text).empty() && profile_lines(rank_text) == profile_lines(serve_text),
          "turbo: rank and serve print the same profile resolution");
    std::string timing;
    std::vector<std::string> values{"turbo"};
    auto next = [&]() { auto v = values.back(); values.pop_back(); return v; };
    check(md::product_option("--profile", next, &timing) && timing.empty(), "turbo: --profile NAME is the product profile");
    values = {"s:prompt.u32:8"};
    check(md::product_option("--profile", next, &timing) && timing == "s:prompt.u32:8", "turbo: old timing spelling kept");
    values = {"t:prompt.u32:4"};
    timing.clear();
    check(md::product_option("--timing-profile", next, &timing) && timing == "t:prompt.u32:4", "turbo: --timing-profile");
    values = {"s:prompt.u32:8"};
    bool refused = false;
    try { (void)md::product_option("--profile", next, nullptr); } catch (const std::invalid_argument&) { refused = true; }
    check(refused, "turbo: serve refuses a timing spec as a product profile");
    const EnvMap onfly{{md::kProductProfileEnv, "turbo"}, {"STRIX_TURBO_WEIGHTS", "onfly"}};
    const auto w = md::profile_resolution(lookup(onfly)).weights;
    check(!w.honoured() && md::effective_config(serve, lookup(onfly)).find("profile.weights_reason=pending") != std::string::npos,
          "turbo: unbuilt onfly is printed as pending, never run silently");
}

// Split-producer chunk frames: one lever row, one policy for the rank CLI and serve entry points.
void test_split_frame() {
    namespace sx = strix::xport;
    const auto& row = md::split_frame_row();
    check(std::string(row.name) == "mm_split_frame" && row.regime.crossover == 4, "split frame row");
    const auto unset = sx::split_frame_policy(row, nullptr);
    check(!unset.on(1) && !unset.on(4) && !unset.on(8) && !unset.on(1024), "unqualified split frame stays off by default");
    check(!sx::split_frame_policy(row, "").on(8), "empty split frame stays off");
    check(!sx::split_frame_policy(row, "0").on(8), "split frame explicit 0");
    check(sx::split_frame_policy(row, "1").on(2) && !sx::split_frame_policy(row, "1").on(1), "split frame explicit 1, never 1 row");
    bool rejected = false;
    try { (void)sx::split_frame_policy(row, "maybe"); } catch (const std::exception&) { rejected = true; }
    check(rejected, "split frame invalid text rejected");
    // Both entry points end with the same effective value (one cached resolution; the env is read once).
    (void)::unsetenv("STRIX_XPORT_SPLIT_ROWS");
    char flag[] = "--print-effective-config", app[] = "strix-minimax-rank", m[] = "M", r0[] = "0", tr[] = "usb4-auto", out[] = "OUT";
    char rank_flag[] = "--rank";
    char* rank_argv[] = {app, m, r0, tr, out, flag};
    char* serve_argv[] = {app, m, rank_flag, r0, flag};
    (void)md::print_or_apply("rank", 6, rank_argv);
    const auto a = md::split_frame_policy();
    (void)md::print_or_apply("serve", 5, serve_argv);
    const auto b = md::split_frame_policy();
    check(a.mode == b.mode && a.crossover == b.crossover && a.mode == unset.mode, "rank and serve resolve the same split policy");
}

// FAST prefill profile (mm_profile_table.hpp): bundle expansion, explicit-OFF precedence, FAST_I4 exclusion, legacy
// field omission with every row OFF, rank agreement.
void test_fast_prefill_profile() {
    const EnvMap off;
    const auto none_fp = strix::profile::resolve(strix::profile::Engine{"minimax-pair", "", {}, false}, {}).fingerprint();
    const auto fp = [](const EnvMap& m) { return md::profile_resolution(lookup(m)).fingerprint(); };
    {
        const auto p = md::prefill_policy(lookup(off));
        check(p.msa_flash == 0 && p.msa_flash_asked == "default" && p.policies.empty() && p.note.empty() && p.canonical.empty(),
              "fast profile: plain invocation runs nothing");
        // The chunk-invariant linear order is the table default (pf_order): a plain invocation names it; the
        // baseline switch and the explicit off form keep the legacy (empty) identity.
        check(md::prefill_identity(lookup(off)) == "prefill_order=mmx-pf-linear-s1-v1" && md::linear_order(lookup(off)) == "invariant",
              "fast profile: plain cache identity names the default invariant order");
        const EnvMap base{{md::kBaselineEnv, "1"}};
        check(md::prefill_identity(lookup(base)).empty() && md::linear_order(lookup(base)) == "adaptive",
              "linear order: baseline restores the adaptive order and the legacy cache identity");
        EnvMap inv{{"STRIX_MMX_PF_ORDER", "invariant"}}, ada{{"STRIX_MMX_PF_ORDER", "adaptive"}};
        check(md::prefill_identity(lookup(inv)) == "prefill_order=mmx-pf-linear-s1-v1" && md::linear_order(lookup(inv)) == "invariant",
              "linear order: invariant names its cache identity");
        check(md::prefill_identity(lookup(ada)).empty() && md::linear_order(lookup(ada)) == "adaptive",
              "linear order: adaptive keeps the legacy cache identity");
        check(fp(off) == none_fp, "fast profile: rows at builtin leave the legacy fingerprint unchanged");
        for (const auto& e : md::profile_resolution(lookup(off)).entries)
            check(e.status == strix::profile::Status::builtin && e.value == e.lever->builtin, "fast profile: builtin row " + std::string(e.lever->name));
        const EnvMap unused{{"STRIX_MM_PF_MSA_FLASH_LAYERS", "10-59"}, {"STRIX_MM_FAST_PREFILL", "01"}};
        check(fp(unused) == none_fp && md::prefill_policy(lookup(unused)).msa_flash == 0,
              "fast profile: an unused layer set and a non-'1' switch keep the legacy fingerprint");
        const EnvMap zero{{"STRIX_MM_PF_MSA_FLASH", "0"}};
        check(md::prefill_identity(lookup(zero)) == kOrderId && fp(zero) == none_fp,
              "fast profile: explicit OFF alone is the plain identity (only the default order field)");
    }
    const EnvMap bundle{{"STRIX_MM_FAST_PREFILL", "1"}};
    for (const auto& guard : {std::pair{"STRIX_MM_PF_MSA_REWORK", "0"},
                              std::pair{"STRIX_MM_PF_MSA_FLASH_LAYERS", "none"},
                              std::pair{"STRIX_MM_PF_MSA_FLASH_LAYERS", "!0-59"},
                              std::pair{"STRIX_MM_ATTN_PREFILL", "off"},
                              std::pair{"STRIX_MM_BASELINE", "1"}}) {
        EnvMap declined = bundle;
        declined[guard.first] = guard.second;
        const auto p = md::prefill_policy(lookup(declined));
        check(p.msa_flash == 0 && p.msa_flash_requested == 1 && p.policies.empty() && !p.note.empty() && p.msa_flash_asked == "fast_prefill",
              "fast admission: static guard declines flash and preserves request: " + std::string(guard.first));
        const std::string order_id = md::linear_order(lookup(declined)) == "invariant" ? kOrderId : std::string();
        // The router member does not depend on the flash guards: it still runs and is the only FAST identity left
        // beside the default linear-order field (none under the baseline switch, which restores adaptive).
        const std::string router_id = "prefill_router=mm-pf-router-wmma-v1-l40-59";
        const std::string want_id = order_id.empty() ? router_id : router_id + ";" + order_id;
        check(md::prefill_identity(lookup(declined)) == want_id && p.router_wmma &&
                  fp(declined) == fp(EnvMap{{"STRIX_MM_PF_ROUTER_WMMA", "1"}}),
              "fast admission: declined flash leaves only the router member in the cache field and agreement input");
        const auto cfg = md::effective_config(md::parse_cli("rank", {}), lookup(declined));
        check(cfg.find("\nSTRIX_MM_PF_MSA_FLASH=0\n") != std::string::npos &&
              cfg.find("\nprefill.fast_policies=none\n") != std::string::npos &&
              cfg.find("\nprefill.fast_profile=mm-fast-pf-v2\n") != std::string::npos &&
              cfg.find("\nprefill.fast_profile_requested=mm-fast-pf-v2\n") != std::string::npos &&
              cfg.find("\nprefill.msa_flash_requested=1\n") != std::string::npos &&
              cfg.find("\nprefill.msa_flash_layers_requested=" + p.msa_flash_layers_requested + "\n") != std::string::npos &&
              cfg.find("\nprefill.identity=" + want_id + "\n") != std::string::npos,
              "fast admission: effective receipt matches disabled execution route");
    }
    check(md::msa_flash_decline_reason(false, true, 20) != nullptr &&
          md::msa_flash_decline_reason(true, false, 20) != nullptr &&
          md::msa_flash_decline_reason(true, true, 0) != nullptr &&
          md::msa_flash_decline_reason(true, true, 20) == nullptr,
          "fast admission: live engine receipt guards match static admission");
    {
        const auto p = md::prefill_policy(lookup(bundle));
        check(p.msa_flash == 1 && p.msa_flash_layers == "40-59" && p.msa_flash_asked == "fast_prefill" &&
              p.policies == "msa-flash-bf16-l40-v1", "fast profile: bundle selects msa-flash-bf16-l40-v1");
        check(p.router_wmma && p.router_wmma_layers == "40-59", "fast profile: bundle selects mm-pf-router-wmma-v1 on 40-59");
        check(md::prefill_identity(lookup(bundle)) ==
                  "prefill_msa=msa-flash-bf16-l40-v1;prefill_router=mm-pf-router-wmma-v1-l40-59;" + kOrderId,
              "fast profile: cache identity names both policies");
        check(fp(bundle) != none_fp, "fast profile: bundle changes the profile fingerprint");
        const auto r = md::profile_resolution(lookup(bundle));
        check(r.find("msa_flash") != nullptr && r.find("msa_flash")->value == "1", "fast profile: governed row carries the member");
        const auto cfg = md::effective_config(md::parse_cli("rank", {}), lookup(bundle));
        check(cfg.find("\nSTRIX_MM_PF_MSA_FLASH=1\n") != std::string::npos &&
              cfg.find("\nprefill.fast_policies=msa-flash-bf16-l40-v1\n") != std::string::npos &&
              cfg.find("\nprefill.fast_profile=mm-fast-pf-v2\n") != std::string::npos, "fast profile: effective config prints the rows");
    }
    {   // An explicit member wins: OFF beats the bundle, 2 replaces its value.
        const EnvMap explicit_off{{"STRIX_MM_FAST_PREFILL", "1"}, {"STRIX_MM_PF_MSA_FLASH", "0"}};
        const auto p = md::prefill_policy(lookup(explicit_off));
        check(p.msa_flash == 0 && p.policies.empty() &&
                  md::prefill_identity(lookup(explicit_off)) == "prefill_router=mm-pf-router-wmma-v1-l40-59;" + kOrderId,
              "fast profile: explicit OFF beats the bundle (the other member stays)");
        const EnvMap router_off{{"STRIX_MM_FAST_PREFILL", "1"}, {"STRIX_MM_PF_ROUTER_WMMA", "0"}};
        check(!md::prefill_policy(lookup(router_off)).router_wmma &&
                  md::prefill_identity(lookup(router_off)) == "prefill_msa=msa-flash-bf16-l40-v1;" + kOrderId,
              "fast profile: explicit router OFF beats the bundle");
        const auto r = md::profile_resolution(lookup(explicit_off));
        const auto* row = r.find("msa_flash");
        check(row != nullptr && row->status == strix::profile::Status::off, "fast profile: explicit OFF recorded as off");
        const EnvMap two_env{{"STRIX_MM_FAST_PREFILL", "1"}, {"STRIX_MM_PF_MSA_FLASH", "2"}};
        const auto two = md::prefill_policy(lookup(two_env));
        check(two.msa_flash == 2 && two.msa_flash_asked == "2" && two.policies == "msa-flash-bf16-2term-l40-59", "fast profile: explicit 2 wins");
        const EnvMap rg_env{{"STRIX_MM_FAST_PREFILL", "1"}, {"STRIX_MM_PF_MSA_FLASH", "3"}};
        const auto rg = md::prefill_policy(lookup(rg_env));
        check(rg.msa_flash == 3 && rg.policies == "msa-flash-bf16-rg4-l40-59", "fast profile: explicit 3 = row-group form");
        const EnvMap legacy_env{{"STRIX_MM_PF_MSA_FLASH", "1"}, {"STRIX_MM_PF_MSA_FLASH_LAYERS", "50-59"}};
        const auto legacy = md::prefill_policy(lookup(legacy_env));
        check(legacy.msa_flash == 1 && legacy.policies == "msa-flash-bf16-l50-59", "fast profile: the legacy opt-in still runs, named by its layers");
        // A named layer set is reported by its policy id, whatever the mask syntax.
        const EnvMap l30_env{{"STRIX_MM_FAST_PREFILL", "1"}, {"STRIX_MM_PF_MSA_FLASH_LAYERS", "30-39,40-59"}};
        const auto l30 = md::prefill_policy(lookup(l30_env));
        check(l30.msa_flash == 1 && l30.msa_flash_layers == "30-59" && l30.policies == "msa-flash-bf16-l30-v1" &&
                  md::prefill_identity(lookup(l30_env)) ==
                      "prefill_msa=msa-flash-bf16-l30-v1;prefill_router=mm-pf-router-wmma-v1-l40-59;" + kOrderId,
              "fast profile: layers 30-59 = msa-flash-bf16-l30-v1");
    }
    {   // EXL3 base only: the bundle does not select it on FAST_I4; an explicit value still does.
        const EnvMap i4{{"STRIX_MM_FAST_PREFILL", "1"}, {"STRIX_MMX_MODEL_FORMAT", "fast_i4"}};
        const auto p = md::prefill_policy(lookup(i4));
        check(p.msa_flash == 0 && !p.note.empty(), "fast profile: bundle excludes the flash member on fast_i4");
        check(md::prefill_identity(lookup(i4)) == "moe_prefill_format=fast_i4;" + kOrderId, "fast profile: fast_i4 is part of the cache identity");
        EnvMap i4_explicit = i4;
        i4_explicit["STRIX_MM_PF_MSA_FLASH"] = "1";
        check(md::prefill_policy(lookup(i4_explicit)).msa_flash == 1, "fast profile: explicit flash on fast_i4 wins");
    }
    {
        bool threw = false;
        const EnvMap bad_mode{{"STRIX_MM_PF_MSA_FLASH", "4"}}, bad_layers{{"STRIX_MM_PF_MSA_FLASH_LAYERS", "70"}};
        try { (void)md::prefill_policy(lookup(bad_mode)); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "fast profile: invalid flash mode rejected");
        threw = false;
        try { (void)md::prefill_policy(lookup(bad_layers)); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "fast profile: invalid layer set rejected");
    }
    {   // Rank agreement over the control channel: same resolution passes, asymmetric ranks are rejected.
        const auto hello = [](const strix::profile::Resolution& r) {
            std::vector<unsigned char> bytes;
            md::check_peer(r, [&](const void* o, void* i, std::size_t n) {
                bytes.assign(static_cast<const unsigned char*>(o), static_cast<const unsigned char*>(o) + n);
                std::memcpy(i, o, n);
            });
            return bytes;
        };
        const auto agree = [&](const EnvMap& mine, const EnvMap& peer) {
            const auto peer_bytes = hello(md::profile_resolution(lookup(peer)));
            try {
                md::check_peer(md::profile_resolution(lookup(mine)), [&](const void*, void* i, std::size_t n) {
                    std::memcpy(i, peer_bytes.data(), n < peer_bytes.size() ? n : peer_bytes.size());
                });
            } catch (const std::runtime_error&) { return false; }
            return true;
        };
        const EnvMap one{{"STRIX_MM_PF_MSA_FLASH", "1"}}, two{{"STRIX_MM_PF_MSA_FLASH", "2"}};
        const EnvMap same{{"STRIX_MM_FAST_PREFILL", "1"}, {"STRIX_MM_PF_MSA_FLASH", "1"}};
        check(agree(off, off) && agree(bundle, bundle), "fast profile: symmetric ranks agree");
        check(!agree(bundle, off) && !agree(off, bundle), "fast profile: bundle on one rank only is rejected");
        check(!agree(two, one), "fast profile: member value mismatch rejected");
        check(agree(same, bundle), "fast profile: an explicit value equal to the bundle's is the same run");
        const EnvMap members{{"STRIX_MM_PF_MSA_FLASH", "1"}, {"STRIX_MM_PF_ROUTER_WMMA", "1"}};
        check(agree(members, bundle) && agree(bundle, members), "fast agreement: bundle-only and explicit-only agree");
        check(!agree(one, bundle) && !agree(bundle, one), "fast agreement: a missing bundle member is a different run");
        for (const char* layers : {"40-49,50-59", "!0-39", "50-59,40-55", "040-059"}) {
            const EnvMap equivalent{{"STRIX_MM_PF_MSA_FLASH", "1"}, {"STRIX_MM_PF_MSA_FLASH_LAYERS", layers}};
            check(agree(one, equivalent) && agree(equivalent, one), "fast agreement: equivalent parsed layer masks agree");
            check(md::prefill_identity(lookup(equivalent)) == md::prefill_identity(lookup(one)),
                  "fast agreement: equivalent masks name the same cache policy");
        }
        const EnvMap different{{"STRIX_MM_PF_MSA_FLASH", "1"}, {"STRIX_MM_PF_MSA_FLASH_LAYERS", "41-59"}};
        check(!agree(one, different), "fast agreement: different masks remain rejected");
        // FAST index projections (minimax-index-proj-wmma-v1): opt-in member, its own layer set (builtin 40-59).
        EnvMap idx = bundle;
        idx["STRIX_MM_PF_IDX_WMMA"] = "1";
        const auto pi = md::prefill_policy(lookup(idx));
        check(pi.idx_wmma && pi.idx_wmma_layers == "40-59" && pi.msa_flash == 1 &&
                  pi.router_wmma && pi.router_wmma_layers == "40-59" &&
                  pi.policies == "msa-flash-bf16-l40-v1+minimax-index-proj-wmma-v1-l40-59" &&
                  md::prefill_identity(lookup(idx)) ==
                      "prefill_msa=msa-flash-bf16-l40-v1;prefill_idxproj=minimax-index-proj-wmma-v1-l40-59;prefill_router=mm-pf-router-wmma-v1-l40-59;" + kOrderId,
              "fast idx: explicit member runs beside the bundle, own cache field");
        auto idx_router_off = idx;
        idx_router_off["STRIX_MM_PF_ROUTER_WMMA"] = "0";
        const auto pi_router_off = md::prefill_policy(lookup(idx_router_off));
        check(pi_router_off.idx_wmma && pi_router_off.idx_wmma_layers == "40-59" && pi_router_off.msa_flash == 1 &&
                  !pi_router_off.router_wmma && pi_router_off.policies == pi.policies &&
                  md::prefill_identity(lookup(idx_router_off)) == "prefill_msa=msa-flash-bf16-l40-v1;prefill_idxproj=minimax-index-proj-wmma-v1-l40-59;" + kOrderId &&
                  !agree(idx, idx_router_off) && !agree(idx_router_off, idx),
              "fast idx: router OFF preserves index policy and separates rank identities");
        check(!md::prefill_policy(lookup(bundle)).idx_wmma, "fast idx: not a bundle member before its gate");
        for (const char* layout : {"tiled", "off"}) {
            auto declined = idx;
            declined["STRIX_MM_INDEX_PROJ"] = layout;
            auto disabled = declined;
            disabled["STRIX_MM_PF_IDX_WMMA"] = "0";
            const auto policy = md::prefill_policy(lookup(declined));
            check(!policy.idx_wmma && !policy.note.empty() && policy.idx_wmma_asked == "1" &&
                      policy.policies == md::prefill_policy(lookup(disabled)).policies &&
                      md::prefill_identity(lookup(declined)) == md::prefill_identity(lookup(disabled)) && agree(declined, disabled),
                  "fast idx: incompatible layout declines in policy, identity and rank agreement");
            for (const char* app : {"rank", "serve"}) {
                const auto cfg = md::effective_config(md::parse_cli(app, {}), lookup(declined));
                check(cfg.find("\nprefill.idx_wmma=0\n") != std::string::npos &&
                          cfg.find("\nprefill.idx_wmma_layers=none\n") != std::string::npos &&
                          cfg.find("minimax-index-proj-wmma") == std::string::npos,
                      "fast idx: effective config declines incompatible layout");
            }
        }
        auto idx_fp32 = idx;
        idx_fp32["STRIX_MM_PREFILL_IDENTITY"] = "fp32";
        check(!md::prefill_policy(lookup(idx_fp32)).idx_wmma, "fast idx: FP32 prefill declines");
        check(agree(idx, idx) && !agree(idx, bundle) && !agree(bundle, idx), "fast idx: one rank only is rejected");
        EnvMap idx_zero = bundle;
        idx_zero["STRIX_MM_PF_IDX_WMMA"] = "0";
        idx_zero["STRIX_MM_PF_IDX_WMMA_LAYERS"] = "0-59";
        check(agree(idx_zero, bundle) && md::prefill_identity(lookup(idx_zero)) == md::prefill_identity(lookup(bundle)),
              "fast idx: explicit OFF with an unused layer set is the bundle's run");
        EnvMap idx_other = idx;
        idx_other["STRIX_MM_PF_IDX_WMMA_LAYERS"] = "30-59";
        check(!agree(idx, idx_other), "fast idx: different layer sets rejected");
        const EnvMap idx_only{{"STRIX_MM_PF_IDX_WMMA", "1"}, {"STRIX_MM_PF_IDX_WMMA_LAYERS", "40-49,50-59"}};
        check(md::prefill_identity(lookup(idx_only)) == "prefill_idxproj=minimax-index-proj-wmma-v1-l40-59;" + kOrderId,
              "fast idx: alone, only the idxproj field");
        EnvMap idx_none = idx;
        idx_none["STRIX_MM_PF_IDX_WMMA_LAYERS"] = "none";
        check(!md::prefill_policy(lookup(idx_none)).idx_wmma && !md::prefill_policy(lookup(idx_none)).note.empty(),
              "fast idx: empty layer set declines with a note");
        bool threw = false;
        try { (void)md::prefill_policy(lookup(EnvMap{{"STRIX_MM_PF_IDX_WMMA", "2"}})); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "fast idx: invalid value rejected");
        {   // The FAST resolution consults only its own switches (the engine records every read into the receipt).
            const auto base = lookup(idx);
            std::set<std::string> seen;
            (void)md::prefill_policy([&](const char* k) -> const char* { seen.insert(k); return base(k); });
            check(seen.count("STRIX_DIST_CRC_PARALLEL") == 0 && seen.count("STRIX_MM_PF_MSA_REWORK") == 1,
                  "fast receipt: the flash guard reads the rework item, not the whole defaults set");
        }
        // ONE RESOLVER: rank and serve resolve the same request to the same prefill lines.
        const auto prefill_lines = [](const std::string& cfg) {
            std::string out;
            for (std::size_t a = 0; a < cfg.size();) {
                const std::size_t b = std::min(cfg.find('\n', a), cfg.size());
                if (cfg.compare(a, 8, "prefill.") == 0 || cfg.compare(a, 19, "STRIX_MM_PF_IDX_WMM") == 0) out += cfg.substr(a, b - a + 1);
                a = b + 1;
            }
            return out;
        };
        const auto rank_cfg = md::effective_config(md::parse_cli("rank", {}), lookup(idx));
        const auto serve_cfg = md::effective_config(md::parse_cli("serve", {}), lookup(idx));
        check(prefill_lines(rank_cfg) == prefill_lines(serve_cfg) && rank_cfg.find("\nprefill.idx_wmma=1\n") != std::string::npos &&
                  rank_cfg.find("\nprefill.idx_wmma_layers=40-59\n") != std::string::npos,
              "fast idx: rank and serve resolve the same prefill policy");
    }
}

// FAST mm-pf-router-wmma-v1: one resolver (prefill_policy) for the engine, the cache identity, the effective config and
// the rank agreement; opt-in, its layer set read only while it runs.
void test_router_wmma() {
    const auto none_fp = strix::profile::resolve(strix::profile::Engine{"minimax-pair", "", {}, false}, {}).fingerprint();
    const auto fp = [](const EnvMap& m) { return md::profile_resolution(lookup(m)).fingerprint(); };
    const EnvMap off, zero{{"STRIX_MM_PF_ROUTER_WMMA", "0"}}, unused{{"STRIX_MM_PF_ROUTER_WMMA_LAYERS", "3-59"}};
    for (const EnvMap* m : {&off, &zero, &unused}) {
        const auto p = md::prefill_policy(lookup(*m));
        check(!p.router_wmma && p.policies.empty() && md::prefill_identity(lookup(*m)) == kOrderId && fp(*m) == none_fp,
              "router wmma: off / unused layer set keep the legacy identity");
    }
    const EnvMap on{{"STRIX_MM_PF_ROUTER_WMMA", "1"}}, all{{"STRIX_MM_PF_ROUTER_WMMA", "1"}, {"STRIX_MM_PF_ROUTER_WMMA_LAYERS", "3-59"}};
    const auto p = md::prefill_policy(lookup(on));
    check(p.router_wmma && p.router_wmma_layers == "40-59" && p.msa_flash == 0 && p.policies.empty(), "router wmma: default layer set 40-59");
    check(md::prefill_identity(lookup(on)) == "prefill_router=mm-pf-router-wmma-v1-l40-59;" + kOrderId,
          "router wmma: cache identity names policy and layers");
    check(md::prefill_identity(lookup(all)) == "prefill_router=mm-pf-router-wmma-v1-l3-59;" + kOrderId && fp(all) != fp(on) && fp(on) != none_fp,
          "router wmma: the layer set is part of the identity while on");
    const auto cfg = md::effective_config(md::Cli{}, lookup(on));
    check(cfg.find("STRIX_MM_PF_ROUTER_WMMA=1\n") != std::string::npos && cfg.find("STRIX_MM_PF_ROUTER_WMMA_LAYERS=40-59\n") != std::string::npos &&
              cfg.find("prefill.identity=prefill_router=mm-pf-router-wmma-v1-l40-59;" + kOrderId + "\n") != std::string::npos,
          "router wmma: effective config shows the resolved values");
    const EnvMap both{{"STRIX_MM_PF_ROUTER_WMMA", "1"}, {"STRIX_MM_PF_MSA_FLASH", "1"}};
    check(md::prefill_identity(lookup(both)) == "prefill_msa=msa-flash-bf16-l40-v1;prefill_router=mm-pf-router-wmma-v1-l40-59;" + kOrderId,
          "router wmma: composes with the MSA flash member");
    bool threw = false;
    try { (void)md::prefill_policy(lookup(EnvMap{{"STRIX_MM_PF_ROUTER_WMMA", "2"}})); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "router wmma: values other than 0 / 1 are refused");
    const auto hello = [](const strix::profile::Resolution& r) {
        std::vector<unsigned char> bytes;
        md::check_peer(r, [&](const void* o, void* i, std::size_t n) {
            bytes.assign(static_cast<const unsigned char*>(o), static_cast<const unsigned char*>(o) + n);
            std::memcpy(i, o, n);
        });
        return bytes;
    };
    const auto agree = [&](const EnvMap& mine, const EnvMap& peer) {
        const auto peer_bytes = hello(md::profile_resolution(lookup(peer)));
        try {
            md::check_peer(md::profile_resolution(lookup(mine)), [&](const void*, void* i, std::size_t n) {
                std::memcpy(i, peer_bytes.data(), n < peer_bytes.size() ? n : peer_bytes.size());
            });
        } catch (const std::runtime_error&) { return false; }
        return true;
    };
    check(agree(on, on) && !agree(on, off) && !agree(off, on) && !agree(on, all), "router wmma: ranks must resolve the same switch and layer set");
    check(agree(off, zero) && agree(off, unused), "router wmma: off forms agree with the plain invocation");
}

}  // namespace

// ONE RESOLVER: the same request through both entry points. Every host-path default is one row of one table; the
// server may end with another value only where the row names the reason, and effective_config() prints both.
void test_entry_points_host_defaults() {
    const EnvMap none;
    md::Cli rank, serve; rank.app = "rank"; serve.app = "serve";
    const auto rank_lines = lines(md::effective_config(rank, lookup(none)));
    const auto serve_lines = lines(md::effective_config(serve, lookup(none)));
    check(!md::host_defaults().empty(), "host defaults table is not empty");
    std::size_t differing = 0;
    for (const md::HostDefault& row : md::host_defaults()) {
        const char* a = md::host_default_value(row, "rank", lookup(none));
        const char* b = md::host_default_value(row, "serve", lookup(none));
        const std::string av = a ? a : "unset", bv = b ? b : "unset", key = std::string("host.") + row.name;
        check(rank_lines.count(key + "=" + av) == 1, key + " printed by the CLI");
        check(serve_lines.count(key + "=" + bv) == 1, key + " printed by the server");
        if (av != bv) {
            ++differing;
            check(row.serving_reason != nullptr && std::string(row.serving_reason).size() > 20, key + " differs with a named reason");
            check(serve_lines.count(key + ".serving_reason=" + row.serving_reason) == 1, key + " reason printed by the server");
        } else {
            check(row.serving_reason == nullptr, key + " equal on both entry points carries no reason");
        }
        // An explicit value wins on both entry points and removes the difference (and the reason line).
        const EnvMap set = {{row.name, "7"}};
        check(std::string(md::host_default_value(row, "rank", lookup(set))) == "7" &&
              std::string(md::host_default_value(row, "serve", lookup(set))) == "7", key + " explicit value wins");
        check(lines(md::effective_config(serve, lookup(set))).count(key + "=7") == 1, key + " explicit value printed");
    }
    check(differing == 3, "exactly the two transport timeouts and the prefill table ring differ between the entry points");
    const EnvMap base = {{"STRIX_MM_BASELINE", "1"}};
    for (const md::HostDefault& row : md::host_defaults())
        if (row.optimization) check(md::host_default_value(row, "rank", lookup(base)) == nullptr, std::string(row.name) + " off under the baseline");
}

// Shared TP2 exchange format (xport/exchange_format.hpp) through the one MiniMax resolver: default f32 everywhere with
// the legacy identity and fingerprint; an explicit f16 is the named FAST policy tp2-xchg-f16-v0 (identity field,
// fingerprint input, effective-config line); the exact baseline refuses it; bf16 / junk are refused.
void test_tp2_exchange_format() {
    const auto fp = [](const EnvMap& m) { return md::profile_resolution(lookup(m)).fingerprint(); };
    const EnvMap off;
    const auto p0 = md::prefill_policy(lookup(off));
    check(p0.exchange_fmt == strix::xport::ExchangeFormat::f32 && p0.policies.empty(), "tp2 exchange: default f32, no policy");
    const EnvMap f32{{"STRIX_TP2_EXCHANGE_FMT", "f32"}};
    check(md::prefill_identity(lookup(f32)) == kOrderId && fp(f32) == fp(off), "tp2 exchange: explicit f32 keeps the legacy identity");
    const EnvMap f16{{"STRIX_TP2_EXCHANGE_FMT", "f16"}};
    const auto p = md::prefill_policy(lookup(f16));
    check(p.exchange_fmt == strix::xport::ExchangeFormat::f16 && p.policies == "tp2-xchg-f16-v0" && p.msa_flash == 0,
          "tp2 exchange: explicit f16 = named FAST policy");
    check(md::prefill_identity(lookup(f16)) == "tp2_exchange_fmt=f16;" + kOrderId, "tp2 exchange: f16 enters the cache identity");
    check(fp(f16) != fp(off), "tp2 exchange: f16 enters the rank-agreement fingerprint");
    const EnvMap stack{{"STRIX_MM_FAST_PREFILL", "1"}, {"STRIX_TP2_EXCHANGE_FMT", "f16"}};
    const auto ps = md::prefill_policy(lookup(stack));
    check(ps.policies == "msa-flash-bf16-l40-v1+tp2-xchg-f16-v0", "tp2 exchange: joins the bundle's policy list: " + ps.policies);
    check(md::prefill_identity(lookup(stack)) == "prefill_msa=msa-flash-bf16-l40-v1;tp2_exchange_fmt=f16;prefill_router=mm-pf-router-wmma-v1-l40-59;" + kOrderId,
          "tp2 exchange: bundle identity keeps its MSA field: " + md::prefill_identity(lookup(stack)));
    for (const auto& bad : {EnvMap{{"STRIX_TP2_EXCHANGE_FMT", "f16"}, {"STRIX_MM_BASELINE", "1"}},
                            EnvMap{{"STRIX_TP2_EXCHANGE_FMT", "bf16"}}, EnvMap{{"STRIX_TP2_EXCHANGE_FMT", "F16"}}}) {
        bool threw = false;
        try { (void)md::prefill_policy(lookup(bad)); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "tp2 exchange: refused (baseline f16 / unknown format)");
    }
    check(md::effective_config(md::Cli{}, lookup(f16)).find("prefill.tp2_exchange_fmt=f16 (explicit f16 under tp2-xchg-f16-v0") != std::string::npos,
          "tp2 exchange: effective-config line");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " minimax_production_receipt.txt\n";
        return 2;
    }
    try {
        test_table();
        test_receipts(argv[1]);
        test_off_switches();
        test_pair_pipeline_declaration();
        test_resolution_sources();
        test_baseline();
        test_host_receipt();
        test_cli();
        test_async_group();
        test_serve_preflight();
        test_session_inspection();
        test_session_defaults();
        test_session_startup();
        test_apply();
        test_entry_points_host_defaults();
        test_split_frame();
        test_fast_prefill_profile();
        test_tp2_exchange_format();
        test_router_wmma();
        test_turbo_product_profile();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "minimax defaults tests: ALL PASS (" << md::items().size() << " items)\n";
    return 0;
}
