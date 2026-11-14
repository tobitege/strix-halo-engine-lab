#include "../apps/qwen_effective_config.hpp"
#include "strix/session_state_cache.hpp"
#include <iostream>
#include <map>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace pr = strix::qwen::profile;
namespace dc = strix::decode_cli;
namespace {
unsigned checks = 0;
void check(bool ok, const char* message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
template<class F> void refuses(F operation, std::string_view needle) {
    try { operation(); }
    catch (const std::invalid_argument& error) {
        check(std::string_view(error.what()).find(needle) != std::string_view::npos, "refusal explains the cause");
        return;
    }
    check(false, "expected refusal");
}
using Env = std::map<std::string, std::string>;
pr::Lookup lookup(const Env& env) {
    return [&env](const char* key) -> const char* {
        const auto it = env.find(key);
        return it == env.end() ? nullptr : it->second.c_str();
    };
}
constexpr pr::Lever synthetic[] = {
    {.name="ready", .env="STRIX_TEST_PROFILE_READY", .option="--ready", .value={"", "1", ""}, .builtin="0",
     .kind=pr::Kind::exact, .ready={true,true,true}, .runnable={true,true,true}, .target="1", .off="0", .source="host fixture"},
    {.name="pending", .env="STRIX_TEST_PROFILE_PENDING", .option="--pending", .value={"", "1", ""}, .builtin="0",
     .kind=pr::Kind::bit_changing, .ready={true,false,true}, .runnable={false,true,false}, .target="candidate", .off="0", .source="host fixture"},
    {.name="absent", .env="STRIX_TEST_PROFILE_ABSENT", .option="--absent", .value={"", "1", ""}, .builtin="0",
     .kind=pr::Kind::bit_changing, .ready={true,false,true}, .runnable={false,false,false}, .target="future", .off="0", .source="host fixture"},
};
dc::Options parse(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    return dc::parse_options(static_cast<int>(argv.size()), argv.data(), true);
}
void product() {
    const Env empty;
    pr::Inputs inputs{{}, lookup(empty), false};
    check(pr::parse("default") == pr::Name::standard && pr::parse("fast") == pr::Name::fast &&
        pr::parse("quality") == pr::Name::quality, "three profile names");
    refuses([] { (void)pr::parse("turbo"); }, "default, fast or quality");
    for (const auto profile : {pr::Name::standard, pr::Name::fast, pr::Name::quality}) {
        for (const bool allow : {false, true}) {
            inputs.allow_pending = allow;
            const auto r = pr::resolve(profile, inputs);
            Env exported;
            pr::apply(r, [&](const char* key, const char* value) { exported[key] = value; return true; });
            const bool provisional = profile == pr::Name::fast;
            // Fast applies the runnable GDN chunk row as provisional, with or without permission.
            const bool gdn = provisional;
            check(exported.size() == (provisional ? (gdn ? 6U : 4U) : 0U), "provisional runnable rows and provenance export without permission");
            if (gdn) check(exported.at("STRIX_QPF2K_GDN_FP16_CHUNK") == "1" &&
                exported.at("STRIX_QWEN_PROFILE_OWNED_gdn_fp16_chunk") == "fast:provisional:1", "fast applies the GDN chunk row without permission");
            if (provisional) {
                check(exported.at("STRIX_QPF2K_MOE_ROUTE_BF16") == "1", "fast exports BF16 storage without permission");
                check(exported.at("STRIX_QWEN_PROFILE_OWNED_moe_route_bf16") == "fast:provisional:1",
                    "fast exports distinct provisional provenance without permission");
            }
            check(r.qualification() == (provisional ? "qualification in progress: qsa_budget,moe_route_bf16,gdn_fp16_chunk" : "qualified"), "qualification names applied provisional rows");
            check(r.entries.size() == (profile == pr::Name::fast ? 4U : 0U), "only fast has product rows");
            const auto json = strix::serve::parse_json_unique(r.json());
            check(json.find("profile")->as_string() == pr::name(profile), "JSON profile name");
            check(json.find("unqualified")->size() == 0U, "provisional is distinct from unqualified rows");
            check(json.find("provisional")->size() == (provisional ? 3U : 0U) &&
                (!provisional || json.find("provisional")->as_array()[0].as_string() == "qsa_budget"), "JSON provisional list");
            check(json.find("pending")->size() == r.entries.size() - (provisional ? 2U : 0U) - (gdn ? 1U : 0U), "JSON pending rows");
            for (const auto& entry : r.entries) {
                if (provisional && entry.lever->name == "moe_route_bf16") {
                    check(entry.status == pr::Status::provisional && entry.value == "1", "provisional row applies");
                    const auto* row = json.find("levers")->find("moe_route_bf16");
                    check(row->find("status")->as_string() == "provisional", "JSON provisional row status");
                    check(row->find("source")->as_string() ==
                        "selected for fast ahead of its qualification receipt; qualification in progress; changes logits",
                        "JSON retains honest source text");
                } else if (provisional && entry.lever->name == "qsa_budget") {
                    check(entry.status == pr::Status::provisional && entry.value == "1536", "provisional QSA budget applies");
                } else if (gdn && entry.lever->name == "gdn_fp16_chunk") {
                    check(entry.status == pr::Status::provisional && entry.value == "1", "provisional GDN chunk row applies");
                } else {
                    check(entry.status == pr::Status::pending && entry.value == entry.lever->builtin, "pending keeps builtin");
                    check(!entry.target.empty(), "pending records declared target");
                }
            }
            check(r.notices().size() == (profile == pr::Name::quality ? 1U : r.entries.size()),
                "one notice per provisional or pending row");
            unsigned provisional_notices = 0;
            for (const auto& notice : r.notices()) if (notice.find("moe_route_bf16") != std::string::npos) {
                ++provisional_notices;
                check(notice == "profile fast: moe_route_bf16 is bit-changing; qualification in progress" &&
                    notice.find('\n') == std::string::npos, "single-line provisional notice discloses bit change and qualification");
            }
            check(provisional_notices == (provisional ? 1U : 0U), "exactly one provisional notice");
        }
    }
    check(pr::resolve(pr::Name::quality, inputs).notices()[0] == "quality: no quality rows for this model, using default", "quality notice");
    check(pr::resolve(pr::Name::quality, inputs).json().find("\"note\":\"quality: no quality rows for this model, using default\"") != std::string::npos &&
        pr::resolve(pr::Name::standard, inputs).json().find("\"note\"") == std::string::npos, "quality receipt names the default it runs");
    for (const auto& lever : pr::levers()) {
        check(!lever.name.empty() && !lever.option.empty() && !lever.source.empty(), "editable table has provenance and option");
        check(lever.value[0].empty() && lever.value[2].empty(), "default and quality leave every lever alone");
        check(!lever.ready[1] && lever.runnable[1] == (lever.name == "qsa_budget" || lever.name == "moe_route_bf16" || lever.name == "gdn_fp16_chunk" ||
                lever.name == "moe_route_prune"),
            "QSA budget, MoE route storage, the GDN chunk route and the optional route prune are runnable");
        check(!lever.provisional[0] && !lever.provisional[2] &&
            lever.provisional[1] == (lever.name == "qsa_budget" || lever.name == "moe_route_bf16" || lever.name == "gdn_fp16_chunk"),
            "fast QSA budget, MoE route storage and the GDN chunk route are provisional");
        if (lever.env.empty() || lever.value[1].empty()) continue;  // an optional lever in no profile has no fast value to export
        Env env{{std::string(lever.env), std::string(lever.value[1])}};
        inputs = {{}, lookup(env), true};
        if (lever.runnable[1])
            check(pr::resolve(pr::Name::standard, inputs).find(lever.name)->status == pr::Status::explicit_env, "existing env stays explicit");
        else refuses([&] { (void)pr::resolve(pr::Name::standard, inputs); }, "not in this build");
        env[std::string(lever.env)] = std::string(lever.off);
        const auto off = pr::resolve(pr::Name::fast, inputs);
        check(off.find(lever.name)->status == pr::Status::off, "explicit env off wins");
        if (lever.name == "moe_route_bf16") {
            // The provisional GDN chunk row stays applied.
            check(off.qualification() == "qualification in progress: qsa_budget,gdn_fp16_chunk", "explicit off removes MoE provisional qualification");
            unsigned writes = 0;
            pr::apply(off, [&](const char* key, const char*) {
                writes += std::string_view(key).find("MOE_ROUTE_BF16") != std::string_view::npos ||
                    std::string_view(key).ends_with("moe_route_bf16");
                return true;
            });
            check(writes == 0, "explicit env off is never overwritten");
        }
    }
}
void resolver() {
    Env env;
    pr::Inputs inputs{{}, lookup(env), false};
    auto r = pr::resolve(pr::Name::fast, inputs, synthetic);
    check(r.find("ready")->status == pr::Status::set, "ready row applies");
    check(r.find("pending")->status == pr::Status::pending && r.find("absent")->status == pr::Status::pending, "pending never refuses");
    inputs.allow_pending = true;
    r = pr::resolve(pr::Name::fast, inputs, synthetic);
    check(r.find("pending")->status == pr::Status::unqualified && r.find("absent")->status == pr::Status::pending, "only runnable pending applies");
    check(r.qualification() == "unqualified: pending", "unqualified stamp names executed rows");
    auto json = strix::serve::parse_json_unique(r.json());
    check(json.find("unqualified")->as_array()[0].as_string() == "pending", "JSON unqualified list");
    check(json.find("pending")->as_array()[0].find("target")->as_string() == "future", "JSON declared target");
    pr::apply(r, [&](const char* key, const char* value) { env[key] = value; return true; });
    const auto restarted = pr::resolve(pr::Name::fast, inputs, synthetic);
    check(restarted.json() == r.json(), "ready and unqualified export provenance survives re-resolution");
    env["STRIX_TEST_PROFILE_PENDING"] = "0";
    check(pr::resolve(pr::Name::fast, inputs, synthetic).find("pending")->status == pr::Status::off, "changed inherited env is explicit off");
    env["STRIX_TEST_PROFILE_READY"] = "2";
    check(pr::resolve(pr::Name::fast, inputs, synthetic).find("ready")->status == pr::Status::explicit_env, "explicit env beats profile");
    inputs.cli = {{"ready","0"}};
    r = pr::resolve(pr::Name::fast, inputs, synthetic);
    check(r.find("ready")->status == pr::Status::off && r.find("ready")->value == "0", "CLI off beats env");
    pr::apply(r, [&](const char* key, const char* value) { env[key] = value; return true; });
    check(env.at("STRIX_TEST_PROFILE_READY") == "0", "CLI off exports over env");
    inputs.cli = {{"ready","3"}};
    r = pr::resolve(pr::Name::standard, inputs, synthetic);
    check(r.find("ready")->status == pr::Status::explicit_cli, "CLI recorded outside profile");
    pr::apply(r, [&](const char* key, const char* value) { env[key] = value; return true; });
    check(pr::resolve(pr::Name::standard, inputs, synthetic).json() == r.json(), "explicit CLI provenance survives re-resolution");
    inputs.cli.clear(); inputs.allow_pending = false; env.clear();
    env["STRIX_QWEN_PROFILE_ALLOW_PENDING"] = "1";
    check(pr::resolve(pr::Name::fast, inputs, synthetic).qualification() == "unqualified: pending", "allow-pending env alias");
    env["STRIX_TEST_PROFILE_ABSENT"] = "1";
    refuses([&] { (void)pr::resolve(pr::Name::fast, inputs, synthetic); }, "not in this build");
    env.clear(); inputs.cli = {{"absent","1"}};
    refuses([&] { (void)pr::resolve(pr::Name::fast, inputs, synthetic); }, "not in this build");
}
void provisional_resolution() {
    constexpr pr::Lever provisional{
        .name="provisional", .env="STRIX_TEST_PROFILE_PROVISIONAL", .option="--provisional",
        .value={"", "1", ""}, .builtin="0", .kind=pr::Kind::bit_changing,
        .ready={true,false,true}, .runnable={false,true,false}, .target="candidate", .off="0",
        .source="host fixture", .provisional={false,true,false}};
    std::vector<pr::Lever> rows(std::begin(synthetic), std::end(synthetic));
    rows.push_back(provisional);
    Env env;
    pr::Inputs inputs{{}, lookup(env), true};
    const auto mixed = pr::resolve(pr::Name::fast, inputs, rows);
    check(mixed.qualification() == "qualification in progress: provisional; unqualified: pending",
        "mixed qualification deterministically reports provisional then unqualified rows");
    pr::apply(mixed, [&](const char* key, const char* value) { env[key] = value; return true; });
    check(pr::resolve(pr::Name::fast, inputs, rows).json() == mixed.json(), "mixed provenance survives resolution");
    env.clear(); inputs.allow_pending = false;
    const auto automatic = pr::resolve(pr::Name::fast, inputs, rows);
    check(automatic.find("provisional")->status == pr::Status::provisional &&
        automatic.find("pending")->status == pr::Status::pending, "provisional does not enable pending rows");
    rows.back().runnable[1] = false;
    check(pr::resolve(pr::Name::fast, inputs, rows).find("provisional")->status == pr::Status::pending,
        "provisional still requires runnable code");
}
void cli_and_receipts() {
    const auto plain = parse({"decoder","model.gguf","1"});
    const auto explicit_default = parse({"decoder","model.gguf","1","--profile","default"});
    const auto effective_bytes = [](const dc::Options& o) {
        return dc::effective_config_json(dc::resolve_effective_config({o,{}},
            strix::ModelContract::qwen38_flash_next(), {}, {}, "", {}, {}));
    };
    check(effective_bytes(plain) == effective_bytes(explicit_default), "omitted and explicit default effective bytes are identical");
    for (const auto value : {"default","fast","quality"}) {
        const auto o = parse({"decoder","model.gguf","1","--profile",value,"--allow-pending"});
        check(o.product_profile.profile == pr::parse(value), "CLI profile parsed");
        check(!o.route_owned && !o.any_low_level_flag && o.optimization_profile == plain.optimization_profile,
            "profile selector does not claim explicit route");
        check(o.numerical_profile == plain.numerical_profile && o.prefill_numerical_profile == plain.prefill_numerical_profile &&
            o.prefill_moe_route_bf16_requested == plain.prefill_moe_route_bf16_requested,
            "product profiles preserve existing arithmetic and exact bundle");
        check(dc::effective_prefill_argv(o) == dc::effective_prefill_argv(plain), "prefix effective arguments exclude product metadata");
        const auto effective = dc::resolve_effective_config({o,{}}, strix::ModelContract::qwen38_flash_next(), {}, {}, "", {}, {});
        const auto json = strix::serve::parse_json_unique(dc::effective_config_json(effective));
        check(json.find("profile")->dump() == effective.options.product_profile.json(), "effective config profile receipt");
        // fast with --allow-pending runs the GDN chunk row: the same as plain with its switch set.
        const bool fast = std::string_view(value) == "fast";
        dc::Environment gdn_env;
        if (fast) {
            gdn_env["STRIX_QPF2K_GDN_FP16_CHUNK"] = "1";
            gdn_env["STRIX_QPF2K_QSA_BUDGET"] = "1536";
        }
        auto expected = strix::serve::parse_json_unique(dc::effective_config_json(dc::resolve_effective_config({plain,gdn_env}, strix::ModelContract::qwen38_flash_next(), {}, {}, "", {}, {})));
        expected.set("profile", *json.find("profile"));
        if (fast) {
            const auto* row = effective.options.product_profile.find("moe_route_bf16");
            check(row->status == pr::Status::pending && row->value == "0" &&
                row->reason == effective.options.prefill_moe_route_bf16_reason,
                "unavailable backend keeps profile storage pending");
            expected.set("prefill_moe_route_bf16_reason", *json.find("prefill_moe_route_bf16_reason"));
        }
        check(expected.dump() == json.dump(), "effective config differs only by profile receipt");
    }
    check(parse({"decoder","model.gguf","1","--profile=fast"}).product_profile.profile == pr::Name::fast, "equals syntax");
    refuses([] { (void)parse({"decoder","model.gguf","1","--profile"}); }, "requires a value");
    refuses([] { (void)parse({"decoder","model.gguf","1","--profile","turbo"}); }, "default, fast or quality");
    refuses([] { (void)parse({"decoder","model.gguf","1","--profile","fast","--profile=default"}); }, "duplicate");
    check(parse({"decoder","model.gguf","1","--qsa-budget","1536"}).product_profile.find("qsa_budget")->value == "1536",
        "implemented QSA budget CLI accepted");
    check(parse({"decoder","model.gguf","1","--profile","fast","--gdn-fp16-chunk","1"}).product_profile.find("gdn_fp16_chunk")->status ==
        pr::Status::explicit_cli, "GDN chunk CLI recorded");
    check(parse({"decoder","model.gguf","1","--profile","fast","--gdn-fp16-chunk","0"}).product_profile.find("gdn_fp16_chunk")->status ==
        pr::Status::off, "GDN chunk CLI off recorded");
    check(parse({"decoder","model.gguf","1","--profile","fast","--prefill-moe-route-bf16","off"}).product_profile.find("moe_route_bf16")->status == pr::Status::off,
        "explicit CLI off recorded");
    check(pr::policy_option("--profile") && pr::policy_option("--allow-pending") && !pr::policy_option("--prefill-disable"), "only policy options excluded from identity");
}
void native_switch_and_yarn() {
    for (const auto* value : {"on", "off", "1", "0"}) for (const bool allow : {false, true}) {
        const bool on = std::string_view(value) == "on" || std::string_view(value) == "1";
        std::vector<std::string> args{"decoder","model.gguf","1","--profile","fast","--prefill-moe-route-bf16",value};
        if (allow) args.push_back("--allow-pending");
        auto o = parse(std::move(args));
        const auto* row = o.product_profile.find("moe_route_bf16");
        check(o.prefill_moe_route_bf16_requested == on, "native option retains parsing");
        check(row->status == (on ? pr::Status::explicit_cli : pr::Status::off) && row->value == (on ? "1" : "0"), "native CLI wins over profile");
        unsigned writes = 0;
        pr::apply(o.product_profile, [&](const char*, const char*) { ++writes; return true; });
        // Fast also exports the provisional GDN chunk row (switch + provenance).
        check(writes == 4U, "native CLI retains optional field while other profile rows export");
        check(o.product_profile.qualification() == "qualification in progress: qsa_budget,gdn_fp16_chunk",
            "explicit native override is not applied by provisional profile");
    }
    refuses([] { (void)parse({"decoder","model.gguf","1","--prefill-moe-route-bf16","maybe"}); }, "requires on or off");
    refuses([] { (void)parse({"decoder","model.gguf","1","--prefill-moe-route-bf16","on","--prefill-moe-route-bf16","off"}); }, "once");
    for (const auto profile : {"default", "fast", "quality"}) for (const bool allow : {false, true}) {
        std::vector<std::string> args{"decoder","model.gguf","1","--profile",profile,"--rope-yarn","2","--context","32768"};
        if (allow) args.push_back("--allow-pending");
        auto o = parse(std::move(args));
        dc::admit_product_profile(o);
        Env exported;
        pr::apply(o.product_profile, [&](const char* key, const char* value) { exported[key] = value; return true; });
        // The YaRN sweep skips profile metadata, as the decoder does before its admission check.
        Env execution;
        for (const auto& [key, value] : exported) if (!pr::metadata_environment(key)) execution.emplace(key, value);
        check(dc::yarn_environment_refusal(execution, 2).empty(), "profile cannot introduce YaRN refusal");
        const bool fast = std::string_view(profile) == "fast";
        // YaRN admits BF16 route storage and the GDN chunk route (no rotary sites), so fast applies them there as on native.
        const Env fast_env{{"STRIX_QPF2K_MOE_ROUTE_BF16", "1"}, {"STRIX_QPF2K_GDN_FP16_CHUNK", "1"}};
        check(fast ? execution == fast_env : execution.empty(), "YaRN exports only admitted profile switches");
        check(!fast || exported.at("STRIX_QWEN_PROFILE_OWNED_moe_route_bf16") == "fast:provisional:1",
            "YaRN exports provisional provenance");
        check(o.product_profile.qualification() == (fast ? "qualification in progress: moe_route_bf16,gdn_fp16_chunk" : "qualified"), "YaRN stamps the applied provisional row");
        if (!fast) continue;
        const auto* row = o.product_profile.find("moe_route_bf16");
        check(row && row->status == pr::Status::provisional && row->value == "1" && row->reason.empty(),
            "YaRN applies the admitted provisional row");
    }
    check(pr::metadata_environment("STRIX_QWEN_PROFILE_ALLOW_PENDING") &&
        pr::metadata_environment("STRIX_QWEN_PROFILE_OWNED_moe_route_bf16") &&
        !pr::metadata_environment("STRIX_QPF2K_MOE_ROUTE_BF16"), "YaRN sweep excludes only profile metadata");
    Env env;
    pr::Inputs inputs{{}, lookup(env), false};
    env["STRIX_QWEN_PROFILE_ALLOW_PENDING"] = "1";
    auto r = pr::resolve(pr::Name::fast, inputs);
    check(r.qualification() == "qualification in progress: qsa_budget,moe_route_bf16,gdn_fp16_chunk",
        "allow-pending alias preserves provisional status");
    pr::apply(r, [&](const char* key, const char* value) { env[key] = value; return true; });
    check(env.at("STRIX_QPF2K_GDN_FP16_CHUNK") == "1", "fast applies the GDN chunk row");
    // Drop the exported GDN chunk switch so the re-resolution below starts from the table again.
    env.erase("STRIX_QPF2K_GDN_FP16_CHUNK");
    env.erase("STRIX_QWEN_PROFILE_OWNED_gdn_fp16_chunk");
    check(env.at("STRIX_QPF2K_MOE_ROUTE_BF16") == "1", "MoE route BF16 environment exported");
    check(env.at("STRIX_QWEN_PROFILE_OWNED_moe_route_bf16") == "fast:provisional:1", "provisional provenance exported");
    env.erase("STRIX_QWEN_PROFILE_ALLOW_PENDING");
    {
        const auto again = pr::resolve(pr::Name::fast, inputs);
        const auto *a = again.find("moe_route_bf16"), *b = r.find("moe_route_bf16");
        check(a->status == b->status && a->value == b->value && again.qualification() == "qualification in progress: qsa_budget,moe_route_bf16,gdn_fp16_chunk",
            "MoE route BF16 provenance re-resolves identically");
    }
    env["STRIX_QPF2K_MOE_ROUTE_BF16"] = "0";
    check(pr::resolve(pr::Name::fast, inputs).find("moe_route_bf16")->status == pr::Status::off, "explicit off beats env allow-pending");
    inputs.cli = {{"moe_route_bf16","1"}};
    check(pr::resolve(pr::Name::fast, inputs).find("moe_route_bf16")->status == pr::Status::explicit_cli, "explicit CLI beats env off");
}
void declined_provenance() {
    Env env{{"STRIX_QPF2K_MOE_ROUTE_BF16", "1"}, {"STRIX_QWEN_PROFILE_OWNED_moe_route_bf16", "fast:provisional:1"}};
    auto options = parse({"decoder", "model.gguf", "1"});
    options.product_profile = pr::resolve(pr::Name::fast, {{}, lookup(env)});
    const auto got = dc::resolve_effective_config({options, env},
        strix::ModelContract::qwen38_flash_next(), {}, {}, "", {}, {});
    check(got.error.empty() && got.options.product_profile.find("moe_route_bf16")->status == pr::Status::pending,
        "inherited profile ownership still permits soft decline");
    const auto erase = [&](const char* key) { env.erase(key); return true; };
    pr::withdraw_declined(got.options.product_profile, lookup(env), erase);
    pr::apply(got.options.product_profile, [&](const char* key, const char* value) { env[key] = value; return true; });
    check(env.size() == 4 && env.at("STRIX_QPF2K_QSA_BUDGET") == "1536" &&
            env.count("STRIX_QWEN_PROFILE_OWNED_qsa_budget") == 1 && env.at("STRIX_QPF2K_GDN_FP16_CHUNK") == "1" &&
            env.count("STRIX_QWEN_PROFILE_OWNED_gdn_fp16_chunk") == 1,
        "declined inherited proposal and provenance are removed before runtime");
    env.erase("STRIX_QPF2K_QSA_BUDGET");
    env.erase("STRIX_QWEN_PROFILE_OWNED_qsa_budget");
    env["STRIX_QPF2K_MOE_ROUTE_BF16"] = "1";
    pr::withdraw_declined(got.options.product_profile, lookup(env), erase);
    check(env.at("STRIX_QPF2K_MOE_ROUTE_BF16") == "1", "withdrawal preserves unowned explicit environment");
    options.product_profile = pr::resolve(pr::Name::fast, {{}, lookup(env)});
    const auto explicit_request = dc::resolve_effective_config({options, env},
        strix::ModelContract::qwen38_flash_next(), {}, {}, "", {}, {});
    check(!explicit_request.error.empty(), "unowned inherited environment keeps hard refusal");
}
void identity() {
    const auto before = strix::session_numerical_config("fixture");
    check(pr::export_environment("STRIX_QWEN_PROFILE_ALLOW_PENDING", "1"), "set policy metadata");
    check(pr::export_environment("STRIX_QWEN_PROFILE_OWNED_ready", "fast:profile:1"), "set provenance metadata");
    check(strix::session_numerical_config("fixture") == before, "profile metadata has zero session identity bytes");
    check(pr::export_environment("STRIX_TEST_PROFILE_READY", "1"), "set arithmetic fixture");
    check(strix::session_numerical_config("fixture") != before, "applied arithmetic switches remain session identity-bound");
}
#ifndef _WIN32
int restart_child() {
    pr::Inputs inputs; inputs.allow_pending = true;
    const auto r = pr::resolve(pr::Name::fast, inputs, synthetic);
    pr::apply(r);
    if (!std::getenv("STRIX_TEST_PROFILE_REEXEC")) {
        if (!pr::export_environment("STRIX_TEST_PROFILE_REEXEC", "1")) return 4;
        ::execl("/proc/self/exe", "qwen-product-profile-tests", "--restart-child", nullptr);
        return 2;
    }
    check(r.find("ready")->status == pr::Status::set && r.find("pending")->status == pr::Status::unqualified,
        "actual exec retains statuses and unqualified stamp");
    return r.qualification() == "unqualified: pending" ? 0 : 3;
}
void provenance_restart() {
    const auto pid = ::fork();
    check(pid >= 0, "fork restart fixture");
    if (pid == 0) {
        ::unsetenv("STRIX_TEST_PROFILE_REEXEC");
        const int result = restart_child();
        ::_exit(result);
    }
    int status = 0;
    check(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0, "real re-exec is idempotent");
}
#endif
}
int main(int argc, char** argv) try {
#ifndef _WIN32
    if (argc == 2 && std::string_view(argv[1]) == "--restart-child") return restart_child();
    provenance_restart();
#else
    (void)argc; (void)argv;
#endif
    product(); resolver(); provisional_resolution(); cli_and_receipts(); native_switch_and_yarn(); declined_provenance(); identity();
    std::cout << checks << " profile checks passed\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
