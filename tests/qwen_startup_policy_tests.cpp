#include "../apps/qwen_effective_config.hpp"
#include <filesystem>
#include <iostream>

using namespace strix;
using namespace strix::decode_cli;
namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
Options parse(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    return parse_options(static_cast<int>(argv.size()), argv.data(), true);
}
}
int main() try {
    const auto plain = parse({"decoder", "model.gguf", "1"});
    const auto resolve = [&](Options options, Environment env = {}) {
        return initialize_prefill_config({options, env});
    };
    {
        namespace pr = qwen::profile;
        const auto lookup = [](const char*) -> const char* { return nullptr; };
        const auto fast = pr::resolve(pr::Name::fast, {{},lookup,false});
        const auto* row = fast.find("qsa_budget");
        check(row && row->status == pr::Status::provisional && row->value == "1536" &&
            row->lever->runnable[1] && !row->lever->ready[1] && row->lever->provisional[1], "fast QSA budget runnable but not READY");
        for (const auto profile : {pr::Name::standard, pr::Name::quality})
            check(!pr::resolve(profile, {{},lookup,false}).find("qsa_budget"), "other profiles leave budget unset");
        Environment exported;
        pr::apply(fast, [&](const char* key, const char* value) { exported[key] = value; return true; });
        const auto budgeted = resolve(plain, exported);
        check(budgeted.options.resolved_qsa_prefill_budget == 1536U &&
            exported.at("STRIX_QWEN_PROFILE_OWNED_qsa_budget") == "fast:provisional:1536", "fast exports budget and provenance");
        const auto fields = serve::parse_json(effective_config_json(budgeted));
        check(fields.find("prefill_identity")->as_string() == "fp16-v1+hcout16+gdnchunk16+qsab1536" &&
            fields.find("qsa_prefill_budget")->as_int() == 1536, "inspection binds actual budget into identity and witness");
        check(effective_config_notice(budgeted).find("ranked_slots=1536 local_tail=unchanged decode_verify=unchanged") != std::string::npos,
            "startup witness states scope");
        check(prefill_identity_hash(1U,1536U) != prefill_identity_hash(1U,0U) &&
            prefill_identity_hash(1U,1536U) != prefill_identity_hash(1U,512U), "different budgets have different identity hashes");
        check(prefill_identity_hash(1U,1536U) == UINT64_C(0xac0991573b76e126), "qualification identity hash pinned");
        check(bind_prefill_option_identity(42U,1U,1536U) != bind_prefill_option_identity(42U,1U,0U), "compiled arithmetic binds budget");
        check(effective_config_json(resolve(plain, {{"STRIX_QPF2K_QSA_BUDGET","0"}})).find("qsab") == std::string::npos &&
            !resolve(plain).options.resolved_qsa_prefill_budget.value(), "zero and unset retain incumbent identity");
        for (const auto* master : {"--prefill-exact-fp16"})
            check(resolve(parse({"decoder","model.gguf","1",master}), exported).options.resolved_qsa_prefill_budget == 0U,
                "exact master disables budget");
        check(resolve(plain, {{"STRIX_QPF2K_QSA_BUDGET","1536"},{"STRIX_PREFILL_EXACT_FP16","1"}}).options.resolved_qsa_prefill_budget == 0U,
            "environment exact master disables budget");
        for (const auto* text : {"-4", "+4", "1x", "4294967296", " 4"}) {
            bool refused = false;
            try { (void)resolve(plain, {{"STRIX_QPF2K_QSA_BUDGET",text}}); }
            catch (const std::invalid_argument&) { refused = true; }
            check(refused, "malformed budget refused before device work");
        }
        check(qsa_prefill_block_topk(512U,4U,0U) == 512U && qsa_prefill_block_topk(512U,4U,1536U) == 384U &&
            qsa_prefill_block_topk(512U,4U,4096U) == 512U && qsa_prefill_block_topk(256U,8U,1536U) == 192U,
            "budget caps rather than enlarges selection and follows the model ratio");
        bool refused = false;
        try { (void)qsa_prefill_block_topk(512U,4U,1535U); }
        catch (const std::invalid_argument&) { refused = true; }
        check(refused, "budget must divide into complete blocks");
        check(yarn_environment_refusal({{"STRIX_QPF2K_QSA_BUDGET","0"}},2U).empty() &&
            !yarn_environment_refusal({{"STRIX_QPF2K_QSA_BUDGET","1536"}},2U).empty(), "YaRN admits only budget restoration");
        check(parse({"decoder","model.gguf","1","--qsa-budget","1536"}).product_profile.find("qsa_budget")->status == pr::Status::explicit_cli,
            "CLI budget uses existing profile wiring");
        const auto off = pr::resolve(pr::Name::fast, {{{"qsa_budget","0"}},lookup,false});
        check(off.find("qsa_budget")->status == pr::Status::off, "explicit CLI off overrides fast");
        const auto env_off = pr::resolve(pr::Name::fast, {{},[](const char* key) -> const char* {
            return std::string_view(key) == "STRIX_QPF2K_QSA_BUDGET" ? "0" : nullptr;
        },false});
        check(env_off.find("qsa_budget")->status == pr::Status::off, "explicit environment off overrides fast");
    }
    for (const auto value : {"8200", "8193", "16383"}) {
        bool refused = false;
        try { (void)parse({"decoder", "model.gguf", "1", "--known-prefix-prefill-rows", value}); }
        catch (const std::invalid_argument& e) { refused = std::string_view(e.what()).find("multiple of 32") != std::string_view::npos; }
        check(refused, "wide unaligned T fails early with actionable message");
    }
    check(known_prefill_pool_rows(8224U) == 8704U, "32-aligned non-512 T has padded pool capacity");
    for (const auto& off : {resolve(parse({"decoder", "model.gguf", "1", "--baseline"})),
            resolve(plain, {{"STRIX_BASELINE","1"}}), resolve(plain, {{"STRIX_QPF2K_DEFAULTS","0"}})}) {
        check(!off.options.early_setup && !off.options.eager_code_load &&
            !off.environment.contains("HIP_ENABLE_DEFERRED_LOADING"), "master disables startup policies");
        check(off.prefill_option_mask == kPrefillOptionsReady && off.options.prefill_disabled == plain.prefill_disabled &&
            !off.environment.contains("STRIX_QPF2K_HC_LOWBAND"),
            "bundle masters preserve governed arithmetic and defer low-band injection");
        check(!serve::parse_json(effective_config_json(off)).find("compact_stage_images_requested")->as_bool(),
            "master disables compact images");
    }
    for (const auto* value : {"0", "00", "0x0"}) {
        const auto off = resolve(plain, {{"STRIX_QPF2K_EARLY_SETUP",value},{"STRIX_EAGER_CODE_LOAD",value}});
        check(!off.options.early_setup && !off.options.eager_code_load, "startup zero-leading switches consistent");
    }
    for (const auto* name : {"STRIX_QPF2K_T16K"}) for (const auto* value : {"0", "1"})
        check(yarn_environment_refusal({{name,value}}, 2).empty(), "YaRN accepts nonrotary T policy");
    for (const auto& item : Environment{{"STRIX_QSA_INDEX_ROWS","0"},{"STRIX_QPF2K_INDEX_ROWS","0"},
            {"STRIX_QPF2K_QSA_INDEX_SHAPE","2"},{"STRIX_QPF2K_QSA_MASK","0x180"}}) {
        const Environment env{item};
        check(!qpf2k::index_batch_enabled([&](const char* name)->const char* {
            const auto it = env.find(name); return it == env.end() ? nullptr : it->second.c_str();
        }, false, false), "automatic batching respects explicit operator path");
    }
    for (const bool bundled : {false,true}) {
        auto mtp = plain;
        mtp.rotary = RotaryDescriptor::yarn(2U);
        mtp.mtp.enabled = true;
        if (bundled) mtp.mtp_bundle_path = "model.mtp-bundle.json";
        const auto automatic = resolve(mtp, {{"STRIX_PREFILL_INDEX_PROJ_WMMA","1"}});
        check(automatic.prefill_option_mask == 0U && automatic.options.resolved_prefill_option_mask == 0U &&
            automatic.options.prefill_disabled == (kPrefillOptionsReady | kPrefillOptionsCandidates),
            "automatic YaRN MTP retains upstream arithmetic even with candidate opt-in");
    }
    for (const auto& item : qpf2k::defaults)
        check(yarn_environment_path(item.item) != YarnEnvironmentPath::uncovered_rotary,
            "every upstream and bundle default has a rotary classification");
    const auto candidate = 1U << static_cast<unsigned>(PrefillOption::index_proj_wmma);
    check(index_wmma_unqualified_route(candidate,kFastPrefillQsa) &&
        !index_wmma_unqualified_route(0U,kFastPrefillQsa) && !index_wmma_unqualified_route(candidate,kFastPrefillHc),
        "MTP and nested shared candidate exclusion follows active component");
    const Environment http_env{{"STRIX_QPF2K_HC_LOWBAND","3"},{"HIP_ENABLE_DEFERRED_LOADING","0"}};
    const auto http = serve::parse_json("{" + http_exact_prefill_fields([&](const char* name)->const char* {
        const auto it = http_env.find(name); return it == http_env.end() ? nullptr : it->second.c_str();
    }, false, false) + "}");
    const auto* items = http.find("prefill_exact_items");
    check(items && !items->find("early_setup")->as_bool() && !items->find("automatic_t16k")->as_bool() &&
        !items->find("compact_stage_images")->as_bool() && items->find("index_batch")->as_bool() &&
        items->find("ple_staged_acceptance_fix")->as_bool() && items->find("hip_deferred_loading_disabled")->as_bool(),
        "capabilities and HTTP report exact route policy separately from numerical identity");
    const auto on = resolve(plain);
    check(on.options.early_setup && on.options.eager_code_load &&
        on.environment.at("HIP_ENABLE_DEFERRED_LOADING") == "0", "startup defaults");
    for (const bool early : {false,true}) {
        const char* flag = early ? "--no-early-setup" : "--no-eager-code-load";
        const char* env = early ? "STRIX_QPF2K_EARLY_SETUP" : "STRIX_EAGER_CODE_LOAD";
        const char* key = early ? "early_setup" : "eager_code_load";
        for (const auto& off : {resolve(parse({"decoder", "model.gguf", "1", flag})), resolve(plain, {{env,"0"}})}) {
            const auto json = serve::parse_json(effective_config_json(off));
            check(!json.find(key)->as_bool(), "CLI/env off reflected in effective config");
            check(off.prefill_option_mask == on.prefill_option_mask, "exact startup policy preserves arithmetic mask");
        }
        const auto cli_off = parse({"decoder", "model.gguf", "1", flag});
        const auto restricted = resolve(cli_off, {{env,"1"}});
        check(!(early ? restricted.options.early_setup : restricted.options.eager_code_load), "CLI off dominates env on");
        const auto path = std::filesystem::temp_directory_path() / (early ? "qwen-early-config-test.toml" : "qwen-eager-config-test.toml");
        const char* config_key = early ? "qpf2k_early_setup" : "eager_code_load";
        for (const bool config_on : {false,true}) {
            { std::ofstream file(path); file << config_key << " = " << (config_on ? "true" : "false") << '\n'; }
            for (const bool config_first : {false,true}) {
                auto args = std::vector<std::string>{"decoder", "model.gguf", "1"};
                if (!config_first) args.push_back(flag);
                args.insert(args.end(), {"--config",path.string()});
                if (config_first) args.push_back(flag);
                const auto options = resolve(parse(args), {{env,"1"}}).options;
                check(!(early ? options.early_setup : options.eager_code_load), "config cannot undo CLI off in either order");
            }
            const auto options = resolve(parse({"decoder", "model.gguf", "1", "--config=" + path.string()}), {{env,"0"}}).options;
            check(!(early ? options.early_setup : options.eager_code_load), "config on cannot undo env off");
        }
        { std::ofstream file(path); file << config_key << " = 1\n"; }
        bool rejected = false;
        try { (void)parse({"decoder", "model.gguf", "1", "--config",path.string()}); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "startup config requires boolean");
        std::filesystem::remove(path);
    }
    for (const auto* value : {"0", "1", "", "custom\"value"}) {
        const auto config = resolve(plain, {{"HIP_ENABLE_DEFERRED_LOADING",value}});
        check(config.environment.at("HIP_ENABLE_DEFERRED_LOADING") == value, "user HIP setting preserved");
        check(serve::parse_json(effective_config_json(config)).find("hip_enable_deferred_loading")->as_string() == value,
            "effective HIP setting is escaped and reported");
    }
    check(!resolve(plain, {{"STRIX_EAGER_CODE_LOAD","0"}}).environment.contains("HIP_ENABLE_DEFERRED_LOADING"), "eager off leaves HIP default untouched");
    for (Environment env : {Environment{}, {{"STRIX_EAGER_CODE_LOAD","0"}}, {{"HIP_ENABLE_DEFERRED_LOADING","1"}}, {{"HIP_ENABLE_DEFERRED_LOADING",""}}}) {
        const auto before = env;
        unsigned writes = 0;
        const bool applied = apply_eager_code_loading(true, [&](const char* name) -> const char* {
            const auto it = env.find(name); return it == env.end() ? nullptr : it->second.c_str();
        }, [&](const char* name, const char* value) { ++writes; return env.emplace(name,value).second; });
        check(writes == (before.empty() ? 1U : 0U), "eager application never overwrites explicit settings");
        check(applied == before.empty(), "a restart is requested only when the value was set here");
        using namespace std::string_view_literals;
        check(split_process_arguments("runner\0model\0--qualification-prefix-gate\0out\0"sv) ==
            std::vector<std::string>{"runner", "model", "--qualification-prefix-gate", "out"},
            "a restart repeats every original argument, including a wrapping runner's own options");
        check(split_process_arguments("a\0\0b\0"sv) == std::vector<std::string>{"a", "", "b"}, "empty arguments are kept");
        check(split_process_arguments("truncated"sv).empty() && split_process_arguments(""sv).empty(),
            "an incomplete command line falls back to the given arguments");
        if (applied)
            check(!apply_eager_code_loading(true, [&](const char* name) -> const char* {
                const auto it = env.find(name); return it == env.end() ? nullptr : it->second.c_str();
            }, [&](const char*, const char*) { return false; }), "the restarted image does not restart again");
    }
    for (Environment env : {Environment{}, {{"STRIX_QPF2K_INDEX_BATCH","0"}}, {{"STRIX_QPF2K_DEFAULTS","0"}}, {{"STRIX_BASELINE","1"}}}) {
        const auto config = resolve(plain,env);
        check(serve::parse_json(effective_config_json(config)).find("index_batch")->as_bool() == env.empty(), "exact batching default and off policy");
    }
    check(!on.environment.contains("STRIX_QPF2K_HC_LOWBAND"), "lowband injection waits for decoder admission");
    const auto low_off = resolve(plain, {{"STRIX_QPF2K_HC_LOWBAND","0"}});
    check(low_off.environment.at("STRIX_QPF2K_HC_LOWBAND") == "0" &&
        low_off.prefill_option_mask == on.prefill_option_mask, "lowband OFF preserves arithmetic identity");
    for (const auto* value : {"0", "1"}) {
        const auto compact = resolve(plain, {{"STRIX_QPF2K_COMPACT_IMAGES",value}});
        check(serve::parse_json(effective_config_json(compact)).find("compact_stage_images_requested")->as_bool() == (value[0] != '0'),
            "compact storage requested policy reported before device admission");
        check(compact.prefill_option_mask == on.prefill_option_mask, "compact storage preserves arithmetic identity");
    }
    const auto getter = [](const char*) -> const char* { return nullptr; };
    check(!qpf2k::index_batch_enabled(getter,true,false) && !qpf2k::index_batch_enabled(getter,false,true), "batch baseline and GSQ suppression");
    const auto idx = prefill_option_bit(PrefillOption::index_proj_wmma);
    check(parse_prefill_option_list("index_proj_wmma") == idx && !(kPrefillOptionsReady & idx), "index option recognized but not READY");
    const Environment optin{{"STRIX_PREFILL_INDEX_PROJ_WMMA","1"}};
    const auto index_on = resolve(plain,optin);
    check(index_on.prefill_option_mask == (1U | idx), "index WMMA explicit opt-in");
    check(prefill_identity_name(index_on.prefill_option_mask) == "fp16-v1+hcout16+idxproj16", "index identity tag");
    for (const auto* flag : {"--prefill-disable=index_proj_wmma", "--prefill-exact-fp16"}) {
        const auto off = resolve(parse({"decoder", "model.gguf", "1",flag}),optin);
        check(!(off.prefill_option_mask & idx), "index off and master dominate opt-in");
    }
    check(resolve(plain, {{"STRIX_PREFILL_INDEX_PROJ_WMMA","0"}}).prefill_option_mask == 1U, "index env off preserves base");
    const auto score = prefill_option_bit(PrefillOption::index_score_wmma);
    const Environment score_optin{{"STRIX_PREFILL_INDEX_SCORE_WMMA","1"}};
    check(resolve(plain, score_optin).prefill_option_mask == (1U | score) && !(kPrefillOptionsReady & score),
        "score WMMA explicit candidate opt-in");
    check(prefill_identity_fields(1U | score).find("index_score_wmma") != std::string::npos,
        "effective identity reports score candidate");
    for (const auto* flag : {"--prefill-disable=index_score_wmma", "--prefill-exact-fp16"})
        check(!(resolve(parse({"decoder", "model.gguf", "1",flag}),score_optin).prefill_option_mask & score),
            "score off and master dominate opt-in");
    check(resolve(plain, {{"STRIX_PREFILL_INDEX_SCORE_WMMA","0"}}).prefill_option_mask == 1U,
        "score env off preserves base");
    check(resolve(plain, {{"STRIX_PREFILL_INDEX_SCORE_WMMA","1"},{"STRIX_PREFILL_EXACT_FP16","1"}}).prefill_option_mask == 0U,
        "score environment master dominates opt-in");
    check(index_wmma_unqualified_route(score,kFastPrefillQsa), "score MTP/nested exclusion");
    const auto gdn = prefill_option_bit(PrefillOption::gdn_fp16_chunk);
    check(parse_prefill_option_list("gdn_fp16_chunk") == gdn && !(kPrefillOptionsReady & gdn) && (kPrefillOptionsCandidates & gdn),
        "GDN chunk option recognized as candidate, not READY");
    const Environment gdn_optin{{"STRIX_QPF2K_GDN_FP16_CHUNK","1"}};
    const auto gdn_on = resolve(plain,gdn_optin);
    check(gdn_on.prefill_option_mask == (1U | gdn) && prefill_identity_name(gdn_on.prefill_option_mask) == "fp16-v1+hcout16+gdnchunk16",
        "GDN chunk explicit opt-in names its identity");
    for (const auto* flag : {"--prefill-disable=gdn_fp16_chunk", "--prefill-exact-fp16"})
        check(!(resolve(parse({"decoder", "model.gguf", "1",flag}),gdn_optin).prefill_option_mask & gdn), "GDN chunk off and master dominate opt-in");
    const Environment gdn_v3{{"STRIX_QPF2K_GDN_FP16_CHUNK","3"}};
    const auto v3 = resolve(plain,gdn_v3);
    check(v3.prefill_option_mask == (1U | gdn | kPrefillGdnChunkRev3) &&
        prefill_identity_name(v3.prefill_option_mask) == "fp16-v1+hcout16+gdnchunk16r3" &&
        prefill_identity_hash(v3.prefill_option_mask) == UINT64_C(0x7df952f79f083ab6) &&
        prefill_identity_hash(v3.prefill_option_mask) != prefill_identity_hash(gdn_on.prefill_option_mask),
        "GDN rev 3 has a distinct arithmetic/cache identity with the same resource owner");
    for (const auto* flag : {"--prefill-disable=gdn_fp16_chunk", "--prefill-exact-fp16"}) {
        const auto off = resolve(parse({"decoder", "model.gguf", "1",flag}),gdn_v3);
        check(!(off.prefill_option_mask & (gdn | kPrefillGdnChunkRev3)), "rev 3 off removes route and revision");
    }
    check(resolve(plain, {{"STRIX_QPF2K_GDN_FP16_CHUNK","3"},{"STRIX_PREFILL_EXACT_FP16","1"}}).prefill_option_mask == 0U,
        "environment master removes rev 3 and its discriminator");
    for (const auto* value : {"0", "2", "30", "invalid"})
        check(resolve(plain, {{"STRIX_QPF2K_GDN_FP16_CHUNK",value}}).prefill_option_mask == 1U,
            "GDN unsupported values stay off");
    check(resolve(plain, {{"STRIX_PREFILL_INDEX_PROJ_WMMA","1"},{"STRIX_PREFILL_EXACT_FP16","1"}}).prefill_option_mask == 0U, "env master dominates opt-in");
    check(prefill_identity_fields(index_on.prefill_option_mask).find("index_proj_wmma") != std::string::npos, "effective identity reports candidate");
    check(prefill_identity_hash(1U) == UINT64_C(0xee2cd64684149069) &&
        prefill_identity_hash(0U) == UINT64_C(0xa0094634843e16ee) &&
        prefill_identity_hash(3U) == UINT64_C(0xde0052de9ed6494d) &&
        prefill_identity_hash(2U) == UINT64_C(0x3569d521db7757d6), "governed identity hashes remain pinned");
    const auto config_path = std::filesystem::temp_directory_path() / "qwen-index-off-config-test.toml";
    { std::ofstream file(config_path); file << "prefill_disable = [\"index_proj_wmma\"]\n"; }
    check(resolve(parse({"decoder", "model.gguf", "1", "--config", config_path.string()}),optin).prefill_option_mask == 1U,
        "index config off dominates environment opt-in");
    { std::ofstream file(config_path); file << "prefill_disable = [\"index_score_wmma\"]\n"; }
    check(resolve(parse({"decoder", "model.gguf", "1", "--config", config_path.string()}),score_optin).prefill_option_mask == 1U,
        "score config off dominates environment opt-in");
    std::filesystem::remove(config_path);
    check(t16k_prompt_threshold == 32768U, "automatic T16384 threshold");
    std::cout << "startup policy tests passed\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
