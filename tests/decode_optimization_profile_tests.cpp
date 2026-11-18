#include "../apps/hip_decode_options.hpp"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <array>
#include "teacher_rows_test_directory.hpp"
#include <string>
#include <vector>
using namespace strix;
using namespace strix::decode_cli;
namespace {
int failures = 0;
void expect(bool good, const char* message) {
    if (!good) { ++failures; std::cerr << message << '\n'; }
}
Options parse(std::initializer_list<const char*> args) {
    std::vector<std::string> storage;
    for (const auto* arg : args) storage.emplace_back(arg);
    std::vector<char*> argv;
    for (auto& arg : storage) argv.push_back(arg.data());
    return parse_options(static_cast<int>(argv.size()), argv.data());
}
template<class Function> void rejects(Function operation, std::string_view message) {
    try { operation(); expect(false, "expected CLI rejection"); }
    catch (const std::invalid_argument& error) {
        if (std::string_view(error.what()) != message) {
            ++failures; std::cerr << "expected CLI rejection: " << message << "\nactual CLI rejection: " << error.what() << '\n';
        }
    }
}
}
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME

void transposed_island_cli_contract(){
    using namespace strix;
    using namespace strix::decode_cli;
    const auto expect_case=[](bool ok,const char* message){expect(ok,message);};
    const auto parse_args=[](std::vector<std::string> args){
        std::vector<char*> argv;for(auto& value:args)argv.push_back(value.data());
        return parse_options(static_cast<int>(argv.size()),argv.data());
    };
    const auto reject_args=[&](const std::vector<std::string>& args,const char* message){
        bool failed=false;try{(void)parse_args(args);}catch(const std::invalid_argument&){failed=true;}
        expect_case(failed,message);
    };
    const auto set_arg=[](std::vector<std::string>& args,const std::string& key,const std::string& value){
        for(std::size_t i=0;i+1U<args.size();++i)if(args[i]==key){args[i+1U]=value;return;}
        throw std::logic_error("transposed fixture key absent");
    };
    const auto remove_arg=[](std::vector<std::string>& args,const std::string& key){
        for(auto it=args.begin();it!=args.end();++it)if(*it==key){
            const auto end=it+1!=args.end() && !(it+1)->starts_with("--") ? it+2 : it+1;
            args.erase(it,end);return;
        }
        throw std::logic_error("transposed fixture option absent");
    };

    const auto flags=[] {
        return std::vector<std::string>{"decoder","target.gguf","1","--context","8192","--generate","20",
            "--optimization-profile","manual",
            "--gdn-recurrent-layout","transposed","--gdn-islands",
            "--parallel-hyper-mix","--staged-hyper-projections","--fused-staged-hyper-mix",
            "--typed-fused-hyper-projections","--prefetch-typed-hyper-rows","--prefetch-q8-hyper-output",
            "--lane-chain-hyper-output","--virtual-q8-lm-head","--prefetch-q8-lm-head","--wave8-q8-lm-head",
            "--parallel-moe-router","--wave-moe-router-topk","--q4k-a8g32-gate-up","--grouped-exact-q51-down",
            "--q51-slot-parallel-down","--batched-moe-activation","--deferred-moe-status",
            "--virtual-q8-gdn-input","--prefetch-q8-gdn-input","--virtual-q8-gdn-output",
            "--wave8-q8-gdn-input","--wave8-q8-gdn-output","--prefetch-q8-qsa-input","--wave8-q8-qsa-output",
            "--prefetch-q8-shared-expert","--prepared-eager","--moe-shared-overlap"};
    };
    const auto model=ModelContract::qwen38_flash_next(); gguf::FileInfo info;
    ModelManifest manifest;manifest.architecture="qwen3next";
    manifest.layout_identity=model_manifest_layout_identity(manifest);
    const auto resolve=[&](Options& o){
        validate_resolved_gdn_recurrent_layout(o);validate_resolved_gdn_islands(o);
        auto result=resolve_optimization_profile(o,model,info,manifest,validated_model_content_identity,true);
        apply_numerical_profile(o,model,manifest,validated_model_content_identity,true,result);
        return result;
    };
    for(const auto* profile:{"hc-splitk-f32+q51-q80-a8-down-v1",
            "hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1"}){
        auto args=flags();args.insert(args.end(),{"--numerical-profile",profile});
        auto o=parse_args(args);const auto result=resolve(o);
        expect_case(!result.automatic_applied && result.requested=="manual" &&
            o.gdn_recurrent_layout==HipGdnRecurrentLayout::transposed && o.gdn_islands &&
            result.selected.find(profile)!=std::string::npos,
            "explicit manual transposed islands compose only through the complete selected numerical recipe");
        auto missing=args;remove_arg(missing,"--wave8-q8-lm-head");
        bool rejected=false;try{auto incomplete=parse_args(missing);(void)resolve(incomplete);}
        catch(const std::invalid_argument&){rejected=true;}
        expect_case(rejected,"numerical composition cannot invent absent manual full-organization flags");
        auto automatic=args;set_arg(automatic,"--optimization-profile","auto");
        reject_args(automatic,"numerical transposed islands cannot enter automatic organization");
        auto wrong_layout=args;set_arg(wrong_layout,"--gdn-recurrent-layout","stride33");
        reject_args(wrong_layout,"numerical composition does not admit stride33");
        auto no_islands=args;remove_arg(no_islands,"--gdn-islands");
        reject_args(no_islands,"nonbaseline arithmetic composition requires actual islands");
        auto implicit=args;remove_arg(implicit,"--optimization-profile");
        reject_args(implicit,"numerical composition requires an explicitly selected manual organization");
        auto unmarked=o;unmarked.numerical_profile_explicit=false;
        bool implicit_recipe_rejected=false;
        try{validate_numerical_profile_cli(unmarked);}catch(const std::invalid_argument&){implicit_recipe_rejected=true;}
        expect_case(implicit_recipe_rejected,"layout composition does not infer an arithmetic recipe");
    }
    for(const auto* profile:{"q51-a8-down-v1","hc-deferred-norm-f32+q51-q80-a8-down-v1",
            "hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup-v1"}){
        auto args=flags();args.insert(args.end(),{"--numerical-profile",profile});
        reject_args(args,"transposed islands exclude every unadmitted arithmetic recipe");
    }
    auto exact=parse_args(flags());const auto exact_result=resolve(exact);
    expect_case(exact.numerical_profile==DecodeNumericalProfile::exact_v5 && !exact_result.automatic_applied,
        "manual exact-v5 retains its arithmetic identity with explicit transposed islands");
    for(const auto layout:{HipGdnRecurrentLayout::stride32,HipGdnRecurrentLayout::stride33,
            static_cast<HipGdnRecurrentLayout>(99)}){
        expect_case(!gdn_island_recurrent_layout_admitted(layout),"shared island admission excludes every other layout");
    }
    expect_case(gdn_island_recurrent_layout_admitted(HipGdnRecurrentLayout::baseline) &&
        gdn_island_recurrent_layout_admitted(HipGdnRecurrentLayout::transposed),
        "shared island predicate admits exactly incumbent and transposed layouts");
    for(unsigned fault=0;fault<8U;++fault){
        auto bad=exact;
        if(fault==0U)bad.optimization_profile=OptimizationProfile::automatic;
        if(fault==1U)bad.gdn_recurrent_storage=GdnRecurrentStorage::value_key;
        if(fault==2U)bad.known_prefix_window=true;
        if(fault==3U)bad.prompt_lookup=true;
        if(fault==4U)bad.serving_concurrency_explicit=true;
        if(fault==5U)bad.mtp.supplied=true;
        if(fault==6U)bad.execution.mode=DecodeExecutionMode::graph;
        if(fault==7U)bad.gdn_recurrent_layout=HipGdnRecurrentLayout::stride32;
        bool rejected=false;try{validate_resolved_gdn_islands(bad);}catch(const std::invalid_argument&){rejected=true;}
        expect_case(rejected,"transposed island controller/layout/storage admission rejects before native setup");
    }
    auto automatic=parse_args({"decoder","target.gguf","1","--generate","20"});
    const auto automatic_result=resolve_optimization_profile(automatic,model,info,manifest,
        validated_model_content_identity,true);
    expect_case(automatic_result.automatic_applied && automatic.gdn_recurrent_layout==HipGdnRecurrentLayout::baseline,
        "automatic generation preserves the incumbent recurrent layout");
}

#endif

// --prefill-numerical-profile: parsing, the resolved pair, admission and identities.
void prefill_profile_contract() {
    constexpr std::string_view fp32 = "hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1";
    constexpr std::string_view a8 = "hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1";
    constexpr std::string_view deferred = "hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup-v1";
    constexpr std::string_view deferred_a8 = "hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1";
    const auto profile = [](std::string_view name) { return *parse_decode_numerical_profile(name); };
    const auto tiled = [](std::string decode, std::string prefill) {
        std::vector<std::string> args{"decoder", "m", "1", "2", "--generate", "2", "--known-prefix-window",
            "--known-prefix-full-rows", "--known-prefix-prefill-rows", "256", "--known-prefix-workspace-budget-bytes", "1",
            "--known-prefix-host-metadata-budget-bytes", "2", "--known-prefix-uma-budget-bytes", "1",
            "--known-prefix-host-headroom-bytes", "0", "--numerical-profile", decode};
        if (!prefill.empty()) { args.push_back("--prefill-numerical-profile"); args.push_back(prefill); }
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        return parse_options(static_cast<int>(argv.size()), argv.data());
    };

    // Absent flag: the prefill runs the decode profile; the pair keeps that profile's identity.
    const auto absent = tiled(std::string(a8), "");
    expect(!absent.prefill_numerical_profile && !numerical_profile_pair(absent).split() &&
        numerical_profile_pair(absent).prefill == profile(a8) &&
        numerical_profile_pair_identity(numerical_profile_pair(absent)) == numerical_profile_identity(profile(a8)),
        "absent prefill profile resolves to the decode profile and its identity");
    // Explicit and equal: the same pure pair.
    const auto equal = tiled(std::string(a8), std::string(a8));
    expect(equal.prefill_numerical_profile && numerical_profile_pair(equal) == numerical_profile_pair(absent),
        "an explicit equal prefill profile is the pure pair");
    // Both directions of the dense-A8 sibling, for both HC recipes.
    for (const auto& [decode, prefill] : std::array<std::pair<std::string_view, std::string_view>, 4>{
             {{fp32, a8}, {a8, fp32}, {deferred, deferred_a8}, {deferred_a8, deferred}}}) {
        const auto split = tiled(std::string(decode), std::string(prefill));
        const auto pair = numerical_profile_pair(split);
        expect(pair.split() && pair.prefill == profile(prefill) && pair.decode == profile(decode) &&
            split.numerical_profile == profile(decode) && numerical_profile_pair_admitted(pair),
            "dense-A8 sibling pairs are admitted on the tiled prefill route");
        const auto id = numerical_profile_pair_identity(pair);
        expect(id != numerical_profile_identity(pair.prefill) && id != numerical_profile_identity(pair.decode) &&
            id != numerical_profile_pair_identity({pair.decode, pair.prefill}),
            "a split pair identity differs from both profiles and from the reversed pair");
    }
    expect(*parse({"decoder", "m", "1", "--numerical-profile", std::string(fp32).c_str(),
               ("--prefill-numerical-profile=" + std::string(a8)).c_str(), "--known-prefix-window", "--known-prefix-full-rows",
               "--known-prefix-prefill-rows", "256", "--known-prefix-workspace-budget-bytes", "1",
               "--known-prefix-host-metadata-budget-bytes", "2", "--known-prefix-uma-budget-bytes", "1",
               "--known-prefix-host-headroom-bytes", "0", "--generate", "2"}).prefill_numerical_profile == profile(a8),
        "--prefill-numerical-profile=NAME form parses");
    // Profile identities are distinct across every profile.
    for (const auto x : {DecodeNumericalProfile::exact_v5, DecodeNumericalProfile::hc_splitk_f32_v1,
             DecodeNumericalProfile::q51_a8_down_v1, DecodeNumericalProfile::hc_splitk_f32_q51_a8_down_v1,
             DecodeNumericalProfile::q51_q80_a8_down_v1, DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_v1,
             DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_v1,
             DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_v1,
             DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_v1,
             DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1,
             DecodeNumericalProfile::hc_deferred_norm_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1}) {
        unsigned same = 0;
        for (const auto name : {"exact-v5", "hc-splitk-f32-v1", "q51-a8-down-v1", "hc-splitk-f32+q51-a8-down-v1",
                 "q51-q80-a8-down-v1", "hc-splitk-f32+q51-q80-a8-down-v1", fp32.data(), "hc-deferred-norm-f32+q51-q80-a8-down-v1",
                 deferred.data(), a8.data(), deferred_a8.data()})
            same += numerical_profile_identity(profile(name)) == numerical_profile_identity(x);
        expect(same == 1U && numerical_profile_identity(x), "numerical profile identities are non-zero and distinct");
        const auto sibling = numerical_profile_dense_a8_sibling(x);
        expect(!sibling || (numerical_profile_dense_a8(*sibling) != numerical_profile_dense_a8(x) &&
            numerical_profile_dense_a8_sibling(*sibling) == x && numerical_profile_hc_deferred_norm(*sibling) == numerical_profile_hc_deferred_norm(x)),
            "the dense-A8 sibling relation is symmetric and changes only dense A8");
    }

    // Admission.
    rejects([&] { (void)tiled(std::string(fp32), "unknown"); },
        "--prefill-numerical-profile requires a supported numerical recipe");
    rejects([] { (void)parse({"decoder", "m", "1", "--prefill-numerical-profile"}); },
        "--prefill-numerical-profile requires a supported numerical recipe");
    rejects([] { (void)parse({"decoder", "m", "1", "--prefill-numerical-profile=exact-v5", "--prefill-numerical-profile", "exact-v5"}); },
        "duplicate prefill numerical profile option");
    const std::string_view pair_message = "prefill numerical profile may differ from an explicit --numerical-profile only in "
        "the dense A8 projections (q5k-a8-gateup profiles with and without +dense-a8), or be "
        "fast-prefill-f16wmma-v1 paired with hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1";
    rejects([&] { (void)tiled(std::string(fp32), std::string(deferred_a8)); }, pair_message);
    rejects([&] { (void)tiled(std::string(fp32), "hc-splitk-f32+q51-q80-a8-down-v1"); }, pair_message);
    rejects([&] { (void)tiled(std::string(a8), std::string(deferred)); }, pair_message);
    rejects([] { (void)parse({"decoder", "m", "1", "--prefill-numerical-profile", "hc-splitk-f32-v1"}); }, pair_message);
    const std::string_view route_message = "prefill numerical profile requires a prefill route: the tiled known-prefix "
        "prefill (--known-prefix-prefill-rows with --known-prefix-full-rows), the MTP prompt, or both (tiled MTP prompt)";
    auto plain = absent; plain.prefill_numerical_profile = profile(fp32);
    validate_prefill_numerical_profile_cli(plain); // tiled route admitted
    auto serial = plain; serial.known_prefix_window = false; serial.known_prefix_prefill_rows = 0; serial.known_prefix_full_rows = false;
    rejects([&] { validate_prefill_numerical_profile_cli(serial); }, route_message);
    auto window_only = plain; window_only.known_prefix_prefill_rows = 0;
    rejects([&] { validate_prefill_numerical_profile_cli(window_only); }, route_message);
    auto partial_rows = plain; partial_rows.known_prefix_full_rows = false;
    rejects([&] { validate_prefill_numerical_profile_cli(partial_rows); }, route_message);
    auto mtp = serial; mtp.mtp.enabled = true;
    validate_prefill_numerical_profile_cli(mtp); // MTP prompt route admitted
    auto both = plain; both.mtp.enabled = true;
    validate_prefill_numerical_profile_cli(both); // tiled MTP prompt (serving route) admitted
    auto mtp_window_only = window_only; mtp_window_only.mtp.enabled = true;
    validate_prefill_numerical_profile_cli(mtp_window_only); // serial MTP prompt (no tiled blocks) admitted
    const std::string_view exclusion_message = "prefill numerical profile excludes model copy tiers, GDN islands, "
        "prepared/overlapped serial execution and MTP metadata diagnostics";
    for (int which = 0; which < 12; ++which) {
        // The same exclusions on the tiled route (and the MTP prompt), then on the tiled MTP prompt.
        auto excluded = which < 5 ? plain : which == 5 ? mtp : both;
        if (which == 0) excluded.model_copy_tier = HipModelCopyTier::experts;
        if (which == 1) excluded.validation_final_state_path = "state";
        if (which == 2) excluded.gdn_islands = true;
        if (which == 3) excluded.prepared_eager = true;
        if (which == 4) excluded.moe_shared_overlap = true;
        if (which == 5 || which == 11) excluded.mtp.metadata_only = true;
        if (which == 6) excluded.model_copy_tier = HipModelCopyTier::experts;
        if (which == 7) excluded.validation_final_state_path = "state";
        if (which == 8) excluded.gdn_islands = true;
        if (which == 9) excluded.prepared_eager = true;
        if (which == 10) excluded.moe_shared_overlap = true;
        if (which == 1 || which == 7) validate_prefill_numerical_profile_cli(excluded);
        else rejects([&] { validate_prefill_numerical_profile_cli(excluded); }, exclusion_message);
    }
    auto implicit = plain; implicit.numerical_profile_explicit = false;
    rejects([&] { validate_prefill_numerical_profile_cli(implicit); }, pair_message);

    // Fast prefill profile: prefill-only, paired with the dense-A8 split-K profile, components hc/dense/moe.
    constexpr std::string_view fast = "fast-prefill-f16wmma-v1";
    const auto with_components = [&](std::string decode, std::string components) {
        std::vector<std::string> args{"decoder", "m", "1", "2", "--generate", "2", "--known-prefix-window",
            "--known-prefix-full-rows", "--known-prefix-prefill-rows", "256", "--known-prefix-workspace-budget-bytes", "1",
            "--known-prefix-host-metadata-budget-bytes", "2", "--known-prefix-uma-budget-bytes", "1",
            "--known-prefix-host-headroom-bytes", "0", "--numerical-profile", decode, "--prefill-numerical-profile", std::string(fast)};
        if (!components.empty()) { args.push_back("--prefill-fast-components"); args.push_back(components); }
        std::vector<char*> argv;
        for (auto& arg : args) argv.push_back(arg.data());
        return parse_options(static_cast<int>(argv.size()), argv.data());
    };
    const auto all = with_components(std::string(a8), "");
    const auto all_pair = numerical_profile_pair(all);
    expect(all_pair.split() && all_pair.prefill == profile(fast) && all_pair.decode == profile(a8) &&
        all_pair.fast_components == kFastPrefillAll && numerical_profile_pair_admitted(all_pair) &&
        numerical_profile_fast_prefill(profile(fast)) && numerical_profile_dense_a8(profile(fast)) &&
        numerical_profile_hc_splitk(profile(fast)) && !numerical_profile_hc_deferred_norm(profile(fast)),
        "the fast prefill profile pairs with the dense-A8 split-K profile and mirrors its recipe predicates");
    auto fast_mtp = all;
    fast_mtp.mtp.enabled = true;
    fast_mtp.resolved_prefill_option_mask = 1U;
    rejects([&] { validate_prefill_numerical_profile_cli(fast_mtp); },
        "MTP fast HC output WMMA is unqualified; use --prefill-disable hc_output_wmma");
    fast_mtp.resolved_prefill_option_mask = 0U;
    validate_prefill_numerical_profile_cli(fast_mtp);
    const auto hc_only = numerical_profile_pair(with_components(std::string(a8), "hc"));
    const auto dense_moe = numerical_profile_pair(with_components(std::string(a8), "moe,dense"));
    const auto spelled_all = numerical_profile_pair(with_components(std::string(a8), "dense,router,hc,qsa,moe"));
    expect(hc_only.fast_components == kFastPrefillHc && dense_moe.fast_components == (kFastPrefillDense | kFastPrefillMoe) &&
        spelled_all == all_pair, "--prefill-fast-components parses any order of hc, dense, moe, qsa, router");
    const auto fast_id = numerical_profile_pair_identity(all_pair);
    expect(fast_id != numerical_profile_pair_identity(hc_only) && numerical_profile_pair_identity(hc_only) != numerical_profile_pair_identity(dense_moe) &&
        fast_id != numerical_profile_pair_identity({profile(fp32), profile(a8)}) && fast_id != numerical_profile_identity(profile(a8)) &&
        numerical_profile_pair_identity(spelled_all) == fast_id,
        "every fast component subset has its own pair identity");
    const auto dense8 = numerical_profile_pair(with_components(std::string(a8), "hc,dense8,moe"));
    expect(dense8.fast_components == (kFastPrefillHc | kFastPrefillDense8 | kFastPrefillMoe) && numerical_profile_pair_admitted(dense8) &&
        numerical_profile_pair_identity(dense8) != fast_id, "dense8 is an admitted opt-in component with its own identity");
    rejects([&] { (void)with_components(std::string(a8), "dense,dense8"); }, pair_message);
    rejects([&] { (void)with_components(std::string(fp32), ""); }, pair_message);
    rejects([&] { (void)with_components(std::string(deferred_a8), ""); }, pair_message);
    rejects([&] { (void)with_components(std::string(a8), "hc,hc"); }, "--prefill-fast-components accepts hc, dense, moe, qsa, router, dense8 once each");
    rejects([&] { (void)with_components(std::string(a8), "attention"); }, "--prefill-fast-components accepts hc, dense, moe, qsa, router, dense8 once each");
    rejects([&] { (void)tiled(std::string(fast), ""); }, "fast prefill profiles are prefill-only (--prefill-numerical-profile)");
    auto stray = plain; stray.prefill_fast_components = kFastPrefillHc;
    rejects([&] { validate_prefill_numerical_profile_cli(stray); }, "--prefill-fast-components requires a fast prefill numerical profile");
}

int main() {
    const auto dry = parse({"decoder","m","1","--generate","1","--print-effective-config"});
    expect(dry.print_effective_config && !dry.any_low_level_flag && !dry.route_owned,
        "effective-config inspection does not take route ownership");
    const auto baseline_cli = parse({"decoder","m","1","--generate","1","--baseline"});
    expect(baseline_cli.baseline && baseline_cli.numerical_profile == DecodeNumericalProfile::exact_v5,
        "syntax parsing records baseline without confusing historical exact-v5 defaults with resolution");
    const auto baseline_prepared = parse({"decoder","m","1","--generate","1","--baseline","--prepared-eager"});
    expect(baseline_prepared.baseline && baseline_prepared.prepared_eager,
        "baseline retains explicit historical organization");
    rejects([] { (void)parse({"decoder","m","1","--baseline","--known-prefix-window",
        "--prefill-numerical-profile","fast-prefill-f16wmma-v1"}); },
        "prefill numerical profile may differ from an explicit --numerical-profile only in the dense A8 projections (q5k-a8-gateup profiles with and without +dense-a8), or be fast-prefill-f16wmma-v1 paired with hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1");
    const auto dense_prefill = parse({"decoder","m","1","--generate","1","--prefill-numerical-profile","dense-a8"});
    expect(dense_prefill.prefill_profile_deferred && dense_prefill.prefill_numerical_profile ==
        DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1,
        "dense-a8 prefill alias records complete identity for host resolution");
    const auto compact = DecodeNumericalProfile::dense_trellis_g1_fp32_v1;
    const auto compact_name = numerical_profile_name(compact);
    expect(parse_decode_numerical_profile(compact_name) == compact && numerical_profile_dense_a8(compact) &&
        numerical_profile_hc_splitk(compact) && !numerical_profile_hc_deferred_norm(compact), "compact profile retains its specified parent recipe");
    const auto parent = DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_q5k_a8_gateup_dense_a8_v1;
    const NumericalProfilePair original_pair{DecodeNumericalProfile::fast_prefill_f16wmma_v1, parent, kFastPrefillAll};
    const NumericalProfilePair compact_pair{original_pair.prefill, compact, kFastPrefillAll};
    expect(numerical_profile_pair_admitted(compact_pair) && numerical_profile_pair_identity(compact_pair) != numerical_profile_pair_identity(original_pair) &&
        numerical_profile_identity(compact) != numerical_profile_identity(parent), "compact recipe cannot alias original profile or cache identities");
    rejects([] { (void)parse({"decoder", "m", "1", "--dense-trellis-pack=pack.gguf"}); },
        "the explicit dense trellis numerical profile and its pinned pack are required together");
    rejects([&] { (void)parse({"decoder", "m", "1", "--numerical-profile", compact_name.data()}); },
        "the explicit dense trellis numerical profile and its pinned pack are required together");
    prefill_profile_contract();
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    transposed_island_cli_contract();
#endif

    const auto model = ModelContract::qwen38_flash_next();
    gguf::FileInfo info;
    ModelManifest manifest;
    manifest.architecture = "qwen3next";
    manifest.layout_identity = model_manifest_layout_identity(manifest);
    auto defaults = parse({"decoder", "fixture.gguf", "1"});
    expect(defaults.optimization_profile == OptimizationProfile::automatic &&
           defaults.context_tokens == 128 && !defaults.q4k_a8g32_gate_up &&
           !defaults.profile_explicit && !defaults.any_low_level_flag &&
           !defaults.prefetch_q8_qsa_input && !defaults.prefetch_q8_shared_expert &&
           !defaults.deferred_moe_status && !defaults.gpu_greedy_token && !defaults.prepared_eager && !defaults.moe_shared_overlap && !defaults.batched_moe_activation && !defaults.gdn_islands,
           "parser defers auto until content identification");
    auto on = defaults;
    auto inventory_only = parse({"decoder", "fixture.gguf", "1", "--execution-inventory"});
    expect(inventory_only.execution_inventory && !inventory_only.any_low_level_flag,
           "inventory export is independent of kernel selection");
    auto scalar = parse({"decoder", "fixture.gguf", "1", "--generate", "2", "--gpu-greedy-token"});
    const auto scalar_profile = resolve_optimization_profile(scalar, model, info, manifest,
        validated_model_content_identity, true);
    expect(scalar.gpu_greedy_token && scalar_profile.automatic_applied,
           "explicit scalar output preserves the selected automatic kernel bundle");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--gpu-greedy-token"})); },
        "GPU greedy token output requires eager generation without RAW, reference, roundtrip or integer head");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--generate", "2", "--prefix-cache-roundtrip",
        "--gpu-greedy-token"})); },
        "GPU greedy token output requires eager generation without RAW, reference, roundtrip or integer head");
    auto prepared = parse({"decoder", "fixture.gguf", "1", "--prepared-eager"});
    const auto prepared_profile = resolve_optimization_profile(prepared, model, info, manifest,
        validated_model_content_identity, true);
    expect(prepared.prepared_eager && !prepared.any_low_level_flag && prepared_profile.automatic_applied,
           "prepared host submission preserves automatic kernel selection");
    auto composed = parse({"decoder", "fixture.gguf", "1", "--generate", "2",
        "--prepared-eager", "--gpu-greedy-token"});
    expect(composed.prepared_eager && composed.gpu_greedy_token,
           "prepared submission and scalar output compose explicitly");
    expect(parse({"decoder", "m", "1", "--reference", "reference.json", "--prepared-eager"}).prepared_eager,
           "prepared diagnostic replay retains full arrays");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--prepared-eager", "--prepared-eager"})); },
        "duplicate prepared eager option");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--prepared-eager", "--prefix-cache-roundtrip", "--generate", "2"})); },
        "prepared eager execution excludes graph, integer head and prefix roundtrip");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--generate", "2", "--gpu-greedy-token", "--reference", "r"})); },
        "--generate and --reference are separate execution modes");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--generate", "2", "--gpu-greedy-token", "--gpu-greedy-token"})); },
        "duplicate GPU greedy token option");
    auto overlap = parse({"decoder", "fixture.gguf", "1", "--moe-shared-overlap"});
    const auto overlap_profile = resolve_optimization_profile(overlap, model, info, manifest,
        validated_model_content_identity, true);
    expect(overlap.moe_shared_overlap && !overlap.any_low_level_flag && overlap_profile.automatic_applied &&
        overlap.deferred_moe_status && overlap.moe_overlap_arena_budget_bytes == 64ULL * 1024ULL * 1024ULL,
        "explicit overlap retains the automatic arithmetic bundle and a bounded arena budget");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--moe-overlap-arena-budget-bytes", "64"})); },
        "MoE overlap arena budget requires enabled overlap and nonzero bytes");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--moe-shared-overlap", "--moe-overlap-arena-budget-bytes", "0"})); },
        "MoE overlap arena budget requires enabled overlap and nonzero bytes");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--moe-shared-overlap", "--moe-shared-overlap"})); },
        "duplicate MoE shared overlap option");
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    auto batched = parse({"decoder", "fixture.gguf", "1", "--batched-moe-activation"});
    const auto batched_profile = resolve_optimization_profile(batched, model, info, manifest,
        validated_model_content_identity, true);
    expect(batched.batched_moe_activation && !batched.any_low_level_flag && batched_profile.automatic_applied &&
        batched.q4k_a8g32_gate_up && batched.deferred_moe_status,
        "batched activation overlay preserves the existing automatic optimization bundle");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--batched-moe-activation", "--batched-moe-activation"})); },
        "duplicate batched MoE activation option");
#else
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--batched-moe-activation"})); },
        "--batched-moe-activation requires the experimental runtime build capability");
#endif
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    auto islands = parse({"decoder", "fixture.gguf", "1", "--generate", "2", "--gdn-islands",
        "--prepared-eager", "--gpu-greedy-token", "--moe-shared-overlap"});
    const auto islands_profile = resolve_optimization_profile(islands, model, info, manifest,
        validated_model_content_identity, true);
    validate_resolved_gdn_islands(islands);
    expect(islands.gdn_islands && !islands.any_low_level_flag && islands_profile.automatic_applied &&
        islands.prepared_eager && islands.gpu_greedy_token && islands.moe_shared_overlap &&
        islands.deferred_moe_status && islands.virtual_q8_gdn_input && islands.prefetch_q8_gdn_input &&
        islands.virtual_q8_gdn_output && !islands.wave8_q8_gdn_input && !islands.wave8_q8_gdn_output,
        "GDN islands preserve automatic arithmetic and compose with eager submission overlays");
    auto wave_islands = parse({"decoder", "fixture.gguf", "1", "--gdn-islands",
        "--wave8-q8-gdn-input", "--wave8-q8-gdn-output"});
    const auto wave_islands_profile = resolve_optimization_profile(wave_islands, model, info, manifest,
        validated_model_content_identity, true);
    validate_resolved_gdn_islands(wave_islands);
    expect(wave_islands_profile.automatic_applied && wave_islands.wave8_q8_gdn_input && wave_islands.wave8_q8_gdn_output,
        "paired wave8 island organization preserves explicitly requested geometry without altering defaults");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--gdn-islands", "--gdn-islands"})); },
        "duplicate GDN islands option");
    for (const auto* wave : {"--wave8-q8-gdn-input", "--wave8-q8-gdn-output"})
        rejects([&] { static_cast<void>(parse({"decoder", "m", "1", "--gdn-islands", wave})); },
            "GDN islands require eager floating execution without prefix roundtrip and paired wave8 GDN flags");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--gdn-islands", "--prefix-cache-roundtrip", "--generate", "2"})); },
        "GDN islands require eager floating execution without prefix roundtrip and paired wave8 GDN flags");
    auto unsupported_islands = islands;
    unsupported_islands.execution.mode = DecodeExecutionMode::graph;
    rejects([&] { validate_resolved_gdn_islands(unsupported_islands); },
        "GDN islands require eager floating execution without prefix roundtrip and paired wave8 GDN flags");
    unsupported_islands = islands;
    unsupported_islands.execution.integer_head = DecodeIntegerHead::a8_iu8;
    rejects([&] { validate_resolved_gdn_islands(unsupported_islands); },
        "GDN islands require eager floating execution without prefix roundtrip and paired wave8 GDN flags");
    for (unsigned missing = 0; missing < 4; ++missing) {
        auto missing_dependency = islands;
        if (missing == 0) missing_dependency.deferred_moe_status = false;
        if (missing == 1) missing_dependency.virtual_q8_gdn_input = false;
        if (missing == 2) missing_dependency.prefetch_q8_gdn_input = false;
        if (missing == 3) missing_dependency.virtual_q8_gdn_output = false;
        rejects([&] { validate_resolved_gdn_islands(missing_dependency); },
            "GDN islands require resolved D1 and virtual/prefetch input with virtual output");
    }
    auto manual_islands = parse({"decoder", "m", "1", "--optimization-profile", "manual", "--gdn-islands"});
    rejects([&] { validate_resolved_gdn_islands(manual_islands); },
        "GDN islands require resolved D1 and virtual/prefetch input with virtual output");
#else
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--gdn-islands"})); },
        "--gdn-islands requires the experimental runtime build capability");
#endif
    validate_resolved_gdn_islands(defaults);
    expect(!defaults.qsa_hierarchical_selection, "hierarchical QSA stays opt-in");
    validate_resolved_qsa_hierarchical_selection(defaults);
    expect(!defaults.known_prefix_window, "known-prefix route stays opt-in");
    validate_resolved_known_prefix_window(defaults);
    expect(defaults.offline_tuning_cache_path.empty(), "offline persisted tuning stays opt-in");
    const auto provenance = parse({"decoder", "m", "1", "--offline-tuning-cache", "cache.bin", "--offline-compile-provenance", "context.bin"});
    expect(provenance.offline_compile_provenance_path == "context.bin" && !provenance.any_low_level_flag,
        "explicit evaluated compile context is a prepare-only consumer input");
    rejects([] { (void)parse({"decoder", "m", "1", "--offline-compile-provenance", "context.bin"}); },
        "offline compile provenance requires --offline-tuning-cache");
    rejects([] { (void)parse({"decoder", "m", "1", "--offline-compile-provenance"}); },
        "offline compile provenance requires a file");
    rejects([] { (void)parse({"decoder", "m", "1", "--offline-tuning-cache", "cache.bin", "--offline-compile-provenance", "a", "--offline-compile-provenance", "b"}); },
        "duplicate offline compile provenance option");
    const auto persisted = parse({"decoder", "m", "1", "--offline-tuning-cache", "source-only-cache.bin"});
    expect(persisted.offline_tuning_cache_path == "source-only-cache.bin" && !persisted.any_low_level_flag,
        "persisted tuning does not silently alter base optimization profile");
    rejects([] { (void)parse({"decoder", "m", "1", "--offline-tuning-cache"}); }, "offline tuning cache requires a file");
    rejects([] { (void)parse({"decoder", "m", "1", "--offline-tuning-cache", "a", "--offline-tuning-cache", "b"}); },
        "duplicate offline tuning cache option");
    expect(defaults.logprob_token_ids.empty(), "requested logprobs are opt-in");
    auto logprobs = parse({"decoder", "m", "1", "--generate", "3", "--logprob-token", "0", "--logprob-token", "15", "--logprob-token", "0"});
    expect(logprobs.logprob_token_ids == std::vector<std::uint32_t>({0,15,0}) && !logprobs.any_low_level_flag,
        "requested ID order/duplicates survive without disabling automatic profile");
    validate_resolved_requested_logprobs(logprobs,16);
    for (unsigned mutation = 0; mutation < 9U; ++mutation) {
        auto bad = logprobs;
        if (mutation == 0) bad.generate_tokens = 0;
        if (mutation == 1) bad.gpu_greedy_token = true;
        if (mutation == 2) bad.known_prefix_window = true;
        if (mutation == 3) bad.prompt_lookup = true;
        if (mutation == 4) bad.serving_prefix_files.push_back("source-only-prefix");
        if (mutation == 5) bad.prefix_cache_roundtrip = true;
        if (mutation == 6) bad.raw_recording_path = "source-only-raw";
        if (mutation == 7) bad.reference_path = "source-only-reference";
        if (mutation == 8) bad.generate_tokens = 0;
        rejects([&] { validate_resolved_requested_logprobs(bad,16); },
            "requested logprobs require serial full-output generation without scalar, window, lookup, serving, cache, RAW or reference modes");
    }
    { auto bad = logprobs; bad.logprob_token_ids.push_back(16);
      rejects([&] { validate_resolved_requested_logprobs(bad,16); }, "requested logprob token exceeds model vocabulary"); }
    { auto bad = logprobs; bad.logprob_token_ids.resize(65U);
      rejects([&] { validate_resolved_requested_logprobs(bad,16); }, "requested logprob receipt exceeds bounded row/value capacity"); }
    { auto bad = logprobs; bad.context_tokens = 262145;
      rejects([&] { validate_resolved_requested_logprobs(bad,16); }, "requested logprob receipt exceeds bounded row/value capacity"); }
    { auto accepted = logprobs; accepted.logprob_token_ids.assign(64U,0);accepted.context_tokens=16384;
      accepted.prepared_eager=true;accepted.moe_shared_overlap=true;
      validate_resolved_requested_logprobs(accepted,16);
      ++accepted.context_tokens;
      rejects([&] { validate_resolved_requested_logprobs(accepted,16); }, "requested logprob receipt exceeds bounded row/value capacity"); }
    rejects([] { (void)parse({"decoder", "m", "1", "--logprob-token", "4294967296"}); }, "logprob token exceeds uint32");
    expect(defaults.ple_cache_rows == 0 && defaults.ple_cache_budget_bytes == 0,
        "PLE row cache remains absent by default");
    validate_resolved_ple_cache(defaults);
    auto cache_ple = parse({"decoder", "m", "1", "--ple-cache-rows", "8", "--ple-cache-budget-bytes", "4096"});
    expect(cache_ple.ple_cache_rows == 8 && cache_ple.ple_cache_budget_bytes == 4096 && !cache_ple.any_low_level_flag,
        "PLE cache limits do not silently select a kernel profile");
    rejects([&] { validate_resolved_ple_cache(cache_ple); }, "PLE row cache requires resolved planned PLE I/O");
    cache_ple.planned_ple_io = true;
    validate_resolved_ple_cache(cache_ple);
    rejects([] { (void)parse({"decoder", "m", "1", "--ple-cache-rows", "8"}); },
        "PLE cache requires both positive explicit row and byte limits");
    rejects([] { (void)parse({"decoder", "m", "1", "--ple-cache-rows", "0"}); }, "PLE cache limits must be positive");
    rejects([] { (void)parse({"decoder", "m", "1", "--ple-cache-rows"}); }, "PLE cache option requires a value");
    rejects([] { (void)parse({"decoder", "m", "1", "--ple-cache-rows", "2", "--ple-cache-rows", "4"}); },
        "duplicate PLE cache limit");
    for (unsigned mutation = 0; mutation < 4; ++mutation) {
        auto bad = cache_ple;
        if (mutation == 0) bad.ple_cache_rows_explicit = false;
        if (mutation == 1) bad.ple_cache_budget_explicit = false;
        if (mutation == 2) bad.ple_cache_rows = 0;
        if (mutation == 3) bad.ple_cache_budget_bytes = 0;
        rejects([&] { validate_resolved_ple_cache(bad); }, "PLE cache requires both positive explicit row and byte limits");
    }
    expect(defaults.ple_cpu_helper_threads == 0 && !defaults.ple_cpu_helper_threads_explicit,
        "PLE helper budget default retains legacy backend behavior");
    validate_resolved_ple_cpu_helpers(defaults);
    for (const auto* value : {"1", "2", "8"}) {
        auto explicit_helpers = parse({"decoder", "m", "1", "--ple-cpu-helper-threads", value});
        expect(explicit_helpers.ple_cpu_helper_threads == parse_u64(value, "fixture") &&
            !explicit_helpers.any_low_level_flag && !explicit_helpers.planned_ple_io,
            "helper budget neither changes automatic profile nor implicitly enables planned PLE");
        rejects([&] { validate_resolved_ple_cpu_helpers(explicit_helpers); },
            "PLE CPU helper threads require resolved planned PLE I/O");
        explicit_helpers.planned_ple_io = true;
        validate_resolved_ple_cpu_helpers(explicit_helpers);
    }
    expect(parse({"decoder", "m", "1", "--planned-ple-io", "--ple-cpu-helper-threads=8"}).ple_cpu_helper_threads == 8,
        "explicit planned backend accepts helper cap equals form");
    for (const auto* value : {"0", "9"})
        rejects([&] { (void)parse({"decoder", "m", "1", "--ple-cpu-helper-threads", value}); },
            "PLE CPU helper threads must be 1 through 8 when explicit");
    rejects([] { (void)parse({"decoder", "m", "1", "--ple-cpu-helper-threads", "2", "--ple-cpu-helper-threads=4"}); },
        "duplicate PLE CPU helper threads option");
    { auto invalid = defaults; invalid.ple_cpu_helper_threads = 9;
      rejects([&] { validate_resolved_ple_cpu_helpers(invalid); }, "PLE CPU helper threads must be 1 through 8 when explicit"); }

    auto paired_prompt = defaults;
    paired_prompt.known_prefix_window = true; paired_prompt.generate_tokens = 3;
    paired_prompt.deferred_moe_status = true;
    paired_prompt.virtual_q8_gdn_input = paired_prompt.prefetch_q8_gdn_input = true;
    paired_prompt.virtual_q8_gdn_output = paired_prompt.wave8_q8_gdn_input = paired_prompt.wave8_q8_gdn_output = true;
    validate_resolved_known_prefix_window(paired_prompt);
    expect(parse({"decoder", "m", "1", "--known-prefix-window"}).known_prefix_window,
        "parser records opt-in before resolved profile admission");
    expect(!defaults.known_prefix_expert_union, "known-prefix routed expert union stays opt-in");
    expect(parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-expert-union"}).known_prefix_expert_union &&
        !parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-expert-union"}).any_low_level_flag,
        "union overlay preserves automatic profile selection");
    rejects([] { (void)parse({"decoder", "m", "1", "--known-prefix-expert-union"}); },
        "known-prefix expert union requires --known-prefix-window");
    rejects([] { (void)parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-expert-union", "--known-prefix-expert-union"}); },
        "duplicate known-prefix expert union option");
    { auto selected = paired_prompt; selected.known_prefix_expert_union = true;
      selected.q4k_a8g32_gate_up = selected.grouped_exact_q51_down = selected.q51_slot_parallel_down = true;
      validate_resolved_known_prefix_window(selected);
      for (unsigned missing = 0; missing < 4U; ++missing) {
          auto bad = selected;
          if (missing == 0U) bad.q4k_a8g32_gate_up = false;
          if (missing == 1U) bad.grouped_exact_q51_down = false;
          if (missing == 2U) bad.q51_slot_parallel_down = false;
          if (missing == 3U) bad.deferred_moe_status = false;
          rejects([&] { validate_resolved_known_prefix_window(bad); },
              "known-prefix expert union requires resolved A8/grouped Q5_1/slot-parallel/D1");
      }
    }
    expect(!needs_generation_output_dispatcher(defaults), "ordinary CPU output does not construct an output wrapper");
    { auto state = paired_prompt; state.known_prefix_output_policy = KnownWindowOutputPolicy::state_only;
      expect(needs_generation_output_dispatcher(state) && !state.gpu_greedy_token,
          "state-only known prefix uses head omission without a vocabulary selector");
      state.known_prefix_output_policy = KnownWindowOutputPolicy::terminal_row;
      expect(!needs_generation_output_dispatcher(state), "terminal-row windows retain ordinary serial output");
      state.known_prefix_output_policy = KnownWindowOutputPolicy::all_rows;
      expect(!needs_generation_output_dispatcher(state), "all-row windows preserve serial default output");
      state.known_prefix_window = false; state.gpu_greedy_token = true;
      expect(needs_generation_output_dispatcher(state), "existing GPU scalar output retains its wrapper"); }
    expect(defaults.known_prefix_output_policy == KnownWindowOutputPolicy::all_rows &&
        !defaults.known_prefix_output_policy_explicit, "known-window heads remain all rows by default");
    for (const auto* policy : {"all-rows", "terminal-row", "state-only"}) {
        const auto selected = parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-output-policy", policy});
        expect(known_window_output_policy_name(selected.known_prefix_output_policy) == policy &&
            !selected.any_low_level_flag, "known-window output policy does not change optimization profile");
        auto admitted = paired_prompt; admitted.known_prefix_output_policy = selected.known_prefix_output_policy;
        validate_resolved_known_prefix_window(admitted);
    }
    expect(parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-output-policy=terminal-row"}).known_prefix_output_policy ==
        KnownWindowOutputPolicy::terminal_row, "output policy supports equals form");
    rejects([] { (void)parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-output-policy=other"}); },
        "known-prefix output policy must be all-rows, terminal-row or state-only");
    rejects([] { (void)parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-output-policy"}); },
        "known-prefix output policy requires a value");
    rejects([] { (void)parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-output-policy=all-rows", "--known-prefix-output-policy=state-only"}); },
        "duplicate known-prefix output policy");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup", "--known-prefix-output-policy=terminal-row"}); },
        "known-prefix output policy requires --known-prefix-window");
    { auto bad = paired_prompt; bad.known_prefix_output_policy = static_cast<KnownWindowOutputPolicy>(255);
      rejects([&] { validate_resolved_known_prefix_window(bad); }, "unknown known-window output policy"); }
    expect(defaults.known_prefix_rows == 2 && !defaults.known_prefix_rows_explicit,
        "known-prefix logical rows default to two without changing automatic profile");
    for (const auto* text : {"2", "4", "8"}) {
        const auto selected = parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-rows", text});
        expect(selected.known_prefix_rows == parse_u64(text, "fixture") && !selected.any_low_level_flag,
            "known-prefix row selection preserves automatic profile resolution");
        auto admitted = paired_prompt; admitted.known_prefix_rows = selected.known_prefix_rows;
        validate_resolved_known_prefix_window(admitted);
    }
    expect(parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-rows=8"}).known_prefix_rows == 8,
        "known-prefix rows supports explicit equals form");
    for (const auto* text : {"0", "1", "3", "16"})
        rejects([&] { (void)parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-rows", text}); },
            "known-prefix rows must be 2, 4 or 8");
    rejects([] { (void)parse({"decoder", "m", "1", "--known-prefix-rows", "2"}); },
        "known-prefix rows require --known-prefix-window");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup", "--known-prefix-rows", "4"}); },
        "known-prefix rows require --known-prefix-window");
    rejects([] { (void)parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-rows", "4", "--known-prefix-rows=8"}); },
        "duplicate known-prefix rows option");
    { auto bad = paired_prompt; bad.known_prefix_rows = 3;
      rejects([&] { validate_resolved_known_prefix_window(bad); }, "known-prefix rows must be 2, 4 or 8"); }
    rejects([] { (void)parse({"decoder", "m", "1", "--known-prefix-window", "--known-prefix-window"}); },
        "duplicate known-prefix window option");
    for (unsigned mutation = 0; mutation < 10; ++mutation) {
        auto bad = paired_prompt;
        if (mutation == 0) bad.gpu_greedy_token = true;
        if (mutation == 1) bad.prepared_eager = true;
        if (mutation == 2) bad.moe_shared_overlap = true;
        if (mutation == 3) bad.gdn_islands = true;
        if (mutation == 4) bad.qsa_hierarchical_selection = true;
        if (mutation == 5) bad.qsa_gqa_heads_per_block = 2;
        if (mutation == 6) bad.gdn_recurrent_layout = HipGdnRecurrentLayout::stride33;
        if (mutation == 7) bad.execution.mode = DecodeExecutionMode::graph;
        if (mutation == 8) bad.execution.integer_head = DecodeIntegerHead::a8_iu8;
        if (mutation == 9) bad.raw_recording_path = "unused-RAW-path";
        rejects([&] { validate_resolved_known_prefix_window(bad); },
            "known-prefix window requires eager full-head generation without RAW/reference, alternate submission, cache, islands or candidate layouts");
    }
    expect(!defaults.device_queue_trace, "device event profiling is off by default");
    validate_resolved_device_queue_trace(defaults);
    auto trace = defaults; trace.device_queue_trace = true; trace.host_dispatch_telemetry = true; trace.deferred_moe_status = true;
    validate_resolved_device_queue_trace(trace);
    for(unsigned mutation=0;mutation<8;++mutation) {
        auto bad=trace;
        if(mutation==0)bad.host_dispatch_telemetry=false;
        if(mutation==1)bad.deferred_moe_status=false;
        if(mutation==2)bad.execution.mode=DecodeExecutionMode::graph;
        if(mutation==3)bad.serving_prefix_files={"not-opened"};
        if(mutation==4)bad.known_prefix_window=true;
        if(mutation==5)bad.prompt_lookup=true;
        if(mutation==6)bad.prefix_cache_roundtrip=true;
        if(mutation==7)bad.execution.integer_head=DecodeIntegerHead::a8_iu8;
        rejects([&]{validate_resolved_device_queue_trace(bad);},
            "device queue trace requires host telemetry and eager D1 C1 without graph, integer, serving, known-prefix, lookup or cache roundtrip");
    }
    expect(parse({"decoder","m","1","--host-dispatch-telemetry","--device-queue-trace"}).device_queue_trace,
        "explicit device trace does not open a GPU or enable D1 implicitly");
    rejects([]{(void)parse({"decoder","m","1","--device-queue-trace"});},
        "device queue trace requires --host-dispatch-telemetry");
    rejects([]{(void)parse({"decoder","m","1","--host-dispatch-telemetry","--device-queue-trace","--device-queue-trace"});},
        "duplicate device queue trace option");
    expect(!defaults.prompt_lookup_adaptive, "adaptive lookup is default off");
    const auto adaptive_lookup = parse({"decoder", "m", "1", "--prompt-lookup", "--prompt-lookup-adaptive"});
    expect(adaptive_lookup.prompt_lookup_adaptive && !adaptive_lookup.prompt_lookup_depth_explicit &&
        !adaptive_lookup.any_low_level_flag, "adaptive lookup does not replace automatic kernel profile");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup-adaptive"}); },
        "adaptive prompt lookup requires --prompt-lookup without an explicit fixed depth");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup", "--prompt-lookup-adaptive", "--prompt-lookup-depth", "2"}); },
        "adaptive prompt lookup requires --prompt-lookup without an explicit fixed depth");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup", "--prompt-lookup-adaptive", "--prompt-lookup-adaptive"}); },
        "duplicate prompt lookup adaptive option");
    expect(!defaults.prompt_lookup, "prompt lookup remains opt-in");
    validate_resolved_prompt_lookup(defaults);
    auto lookup = paired_prompt;
    lookup.known_prefix_window = false; lookup.prompt_lookup = true;
    for (std::uint32_t depth = 0; depth <= 2; ++depth) {
        lookup.prompt_lookup_depth = depth;
        validate_resolved_prompt_lookup(lookup);
    }
    expect(parse({"decoder", "m", "1", "--prompt-lookup", "--prompt-lookup-depth", "0"}).prompt_lookup_depth == 0,
        "explicit depth zero preserves the device frontier route");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup-depth", "1"}); },
        "prompt lookup depth requires --prompt-lookup");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup", "--prompt-lookup-depth", "3"}); },
        "prompt lookup depth must be zero, one or two");
    rejects([] { (void)parse({"decoder", "m", "1", "--prompt-lookup", "--prompt-lookup"}); },
        "duplicate prompt lookup option");
    for (unsigned mutation = 0; mutation < 14; ++mutation) {
        auto bad = lookup;
        if (mutation == 0) bad.known_prefix_window = true;
        if (mutation == 1) bad.prepared_eager = true;
        if (mutation == 2) bad.moe_shared_overlap = true;
        if (mutation == 3) bad.gdn_islands = true;
        if (mutation == 4) bad.gpu_greedy_token = true;
        if (mutation == 5) bad.execution.mode = DecodeExecutionMode::graph;
        if (mutation == 6) bad.execution.integer_head = DecodeIntegerHead::a8_iu8;
        if (mutation == 7) bad.prefix_cache_roundtrip = true;
        if (mutation == 8) bad.reference_path = "not-opened";
        if (mutation == 9) bad.raw_recording_path = "not-opened";
        if (mutation == 10) bad.qsa_hierarchical_selection = true;
        if (mutation == 11) bad.qsa_gqa_heads_per_block = 4;
        if (mutation == 12) bad.gdn_recurrent_layout = HipGdnRecurrentLayout::stride32;
        if (mutation == 13) bad.generate_tokens = 0;
        rejects([&] { validate_resolved_prompt_lookup(bad); }, "prompt lookup requires eager generation with its own device selectors, without alternate submission/cache/RAW/reference/candidate layouts");
    }
    auto missing_lookup_status = lookup; missing_lookup_status.deferred_moe_status = false;
    rejects([&] { validate_resolved_prompt_lookup(missing_lookup_status); }, "prompt lookup requires resolved D1 and paired wave8 GDN input/output");
    auto missing_lookup_wave = lookup; missing_lookup_wave.wave8_q8_gdn_input = false;
    rejects([&] { validate_resolved_prompt_lookup(missing_lookup_wave); }, "prompt lookup requires resolved D1 and paired wave8 GDN input/output");
    paired_prompt.wave8_q8_gdn_output = false;
    rejects([&] { validate_resolved_known_prefix_window(paired_prompt); },
        "known-prefix window requires resolved D1 and explicit paired wave8 GDN input/output");
    expect(defaults.qsa_gqa_heads_per_block == 1 && defaults.gdn_recurrent_layout == HipGdnRecurrentLayout::baseline,
        "existing QSA and GDN organizations remain default");
    for (const auto* value : {"1", "2", "4"}) {
        auto grouped = parse({"decoder", "fixture.gguf", "1", "--qsa-gqa-heads-per-block", value});
        const auto selected_grouped = resolve_optimization_profile(grouped, model, info, manifest,
            validated_model_content_identity, true);
        validate_resolved_qsa_gqa(grouped);
        expect(selected_grouped.automatic_applied && !grouped.any_low_level_flag,
            "explicit GQA geometry preserves the automatic arithmetic profile");
    }
    rejects([] { (void)parse({"decoder", "m", "1", "--qsa-gqa-heads-per-block", "3"}); },
        "QSA heads per block must be 1, 2 or 4");
    rejects([] { (void)parse({"decoder", "m", "1", "--qsa-gqa-heads-per-block", "2", "--qsa-gqa-heads-per-block", "4"}); },
        "QSA heads per block requires one value");
    for (const auto* layout : {"baseline", "stride32", "stride33", "transposed"}) {
        auto chosen_layout = parse({"decoder", "fixture.gguf", "1", "--gdn-recurrent-layout", layout});
        const auto layout_profile = resolve_optimization_profile(chosen_layout, model, info, manifest,
            validated_model_content_identity, true);
        validate_resolved_gdn_recurrent_layout(chosen_layout);
        expect(chosen_layout.gdn_recurrent_layout_explicit && !chosen_layout.any_low_level_flag && layout_profile.automatic_applied,
            "explicit GDN LDS geometry preserves automatic selection");
    }
    expect(parse({"decoder", "m", "1", "--gdn-recurrent-layout=stride33"}).gdn_recurrent_layout == HipGdnRecurrentLayout::stride33,
        "GDN recurrent layout also accepts the explicit equals form");
    rejects([] { (void)parse({"decoder", "m", "1", "--gdn-recurrent-layout", "other"}); },
        "GDN recurrent layout must be baseline, stride32, stride33 or transposed");
    auto bad_layout = defaults; bad_layout.gdn_recurrent_layout = HipGdnRecurrentLayout::stride32; bad_layout.gdn_islands = true;
    rejects([&] { validate_resolved_gdn_recurrent_layout(bad_layout); },
        "GDN recurrent LDS layout requires eager floating execution with an admitted island layout");
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    auto hierarchical = parse({"decoder", "fixture.gguf", "1", "--qsa-hierarchical-selection"});
    const auto hierarchical_profile = resolve_optimization_profile(hierarchical, model, info, manifest,
        validated_model_content_identity, true);
    validate_resolved_qsa_hierarchical_selection(hierarchical);
    expect(hierarchical.qsa_hierarchical_selection && !hierarchical.any_low_level_flag && hierarchical_profile.automatic_applied,
        "hierarchical selection preserves automatic kernel profile selection");
    rejects([] { (void)parse({"decoder", "m", "1", "--qsa-hierarchical-selection", "--qsa-hierarchical-selection"}); },
        "duplicate hierarchical QSA option");
    for (unsigned mutation = 0; mutation < 5; ++mutation) {
        auto bad = hierarchical;
        if (mutation == 0) bad.deferred_moe_status = false;
        if (mutation == 1) bad.execution.mode = DecodeExecutionMode::graph;
        if (mutation == 2) bad.execution.integer_head = DecodeIntegerHead::a8_iu8;
        if (mutation == 3) bad.prefix_cache_roundtrip = true;
        if (mutation == 4) bad.context_tokens = 131073;
        rejects([&] { validate_resolved_qsa_hierarchical_selection(bad); },
            "hierarchical QSA requires eager D1 without integer head or roundtrip and context<=131072");
    }
#else
    rejects([] { (void)parse({"decoder", "m", "1", "--qsa-hierarchical-selection"}); },
        "--qsa-hierarchical-selection requires the experimental runtime build capability");
#endif
    auto serving = defaults;
    serving.tokens.clear(); serving.serving_prefix_files = {"not-opened.u32le"};
    serving.generate_tokens = 4; serving.deferred_moe_status = true;
    serving.serving_memory_budget = 100; serving.serving_request_budget = 10; serving.serving_headroom = 1;
    serving.serving_memory_explicit = serving.serving_request_explicit = serving.serving_headroom_explicit = true;
    for (const auto concurrency : {1U, 2U, 4U}) {
        serving.serving_concurrency = concurrency;
        validate_resolved_serving(serving);
    }
    const auto file_request = parse({"decoder", "m", "--serve-prefix-u32le", "not-opened.u32le", "--generate", "4",
        "--serve-concurrency", "2", "--serve-memory-budget-bytes", "100", "--serve-request-budget-bytes", "10", "--serve-headroom-bytes", "1"});
    expect(file_request.tokens.empty() && file_request.serving_prefix_files.size() == 1 &&
        !file_request.any_low_level_flag, "serving parser records file requests without changing optimization profile");
    rejects([] { (void)parse({"decoder", "m", "1", "--serve-concurrency", "2"}); },
        "serving controls require --serve-prefix-u32le");
    rejects([] { (void)parse({"decoder", "m", "1", "--serve-concurrency", "3"}); },
        "serving concurrency must be1,2or4");
    const auto cached_serving = parse({"decoder", "m", "--serve-prefix-u32le", "not-opened.u32le", "--generate", "4",
        "--serve-prefix-cache-entries", "2", "--serve-prefix-cache-snapshot-bytes", "4096",
        "--serve-prefix-cache-capture-peak-bytes", "8192", "--serve-prefix-cache-max-tokens", "16"});
    expect(cached_serving.serving_cache_entries == 2 && cached_serving.serving_cache_snapshot_bytes == 4096 &&
        cached_serving.serving_cache_capture_peak_bytes == 8192 && cached_serving.serving_cache_max_prefix_tokens == 16 &&
        !cached_serving.any_low_level_flag, "serving cache stores explicit independent budgets without opening files");
    expect(!defaults.serving_cache_entries_explicit && defaults.serving_cache_entries == 0,
        "serving cache is absent by default");
    rejects([] { (void)parse({"decoder", "m", "1", "--serve-prefix-cache-entries", "1"}); },
        "serving prefix cache requires serving files and all four positive explicit bounded limits");
    rejects([] { (void)parse({"decoder", "m", "--serve-prefix-u32le", "x", "--generate", "4",
        "--serve-prefix-cache-entries", "1"}); },
        "serving prefix cache requires serving files and all four positive explicit bounded limits");
    rejects([] { (void)parse({"decoder", "m", "1", "--serve-prefix-cache-entries", "0"}); },
        "serving prefix cache limits must be positive");
    rejects([] { (void)parse({"decoder", "m", "1", "--serve-prefix-cache-entries"}); },
        "serving prefix cache option requires a value");
    rejects([] { (void)parse({"decoder", "m", "1", "--serve-prefix-cache-entries", "1", "--serve-prefix-cache-entries", "2"}); },
        "duplicate serving prefix cache option");
    rejects([] { (void)parse({"decoder", "m", "--serve-prefix-u32le", "x", "--generate", "4", "--context", "8",
        "--serve-prefix-cache-entries", "1", "--serve-prefix-cache-snapshot-bytes", "4096",
        "--serve-prefix-cache-capture-peak-bytes", "8192", "--serve-prefix-cache-max-tokens", "16"}); },
        "serving prefix cache requires serving files and all four positive explicit bounded limits");
    for (unsigned mutation = 0; mutation < 6; ++mutation) {
        auto bad = cached_serving;
        if (mutation == 0) bad.serving_cache_entries_explicit = false;
        if (mutation == 1) bad.serving_cache_snapshot_explicit = false;
        if (mutation == 2) bad.serving_cache_peak_explicit = false;
        if (mutation == 3) bad.serving_cache_tokens_explicit = false;
        if (mutation == 4) bad.serving_cache_max_prefix_tokens = bad.context_tokens + 1U;
        if (mutation == 5) bad.serving_prefix_files.clear();
        rejects([&] { validate_serving_prefix_cache_limits(bad); },
            "serving prefix cache requires serving files and all four positive explicit bounded limits");
    }
    for (unsigned mutation = 0; mutation < 13; ++mutation) {
        auto bad = serving;
        if (mutation == 0) bad.serving_headroom_explicit = false;
        if (mutation == 1) bad.serving_request_budget = 0;
        if (mutation == 2) bad.serving_headroom = bad.serving_memory_budget;
        if (mutation == 3) bad.moe_shared_overlap = true;
        if (mutation == 4) bad.gdn_islands = true;
        if (mutation == 5) bad.known_prefix_window = true;
        if (mutation == 6) bad.prompt_lookup = true;
        if (mutation == 7) bad.prefix_cache_roundtrip = true;
        if (mutation == 8) bad.execution.mode = DecodeExecutionMode::graph;
        if (mutation == 9) bad.execution.integer_head = DecodeIntegerHead::a8_iu8;
        if (mutation == 10) bad.tokens = {1};
        if (mutation == 11) bad.execution_inventory = true;
        if (mutation == 12) bad.deferred_moe_status = false;
        rejects([&] { validate_resolved_serving(bad); }, mutation < 3 ?
            "serving requires concurrency 1/2/4, at most64 requests and explicit total/request/headroom reservation bytes" :
            "serving requires eager D1 generation from prefix files, without positional tokens, graph/integer, overlap/islands, cache, paired/prompt routes or single-session validation outputs");
    }
    auto stored = defaults;
    stored.gdn_recurrent_storage = GdnRecurrentStorage::value_key;
    validate_resolved_gdn_recurrent_storage(stored);
    expect(parse({"decoder", "m", "1", "--gdn-recurrent-storage=value-key"}).gdn_recurrent_storage == GdnRecurrentStorage::value_key,
        "physical storage has an explicit independent selector");
    for (unsigned mutation = 0; mutation < 5; ++mutation) {
        auto bad = stored;
        if (mutation == 0) bad.gdn_islands = true;
        if (mutation == 1) bad.known_prefix_window = true;
        if (mutation == 2) bad.prompt_lookup = true;
        if (mutation == 3) bad.execution.mode = DecodeExecutionMode::graph;
        if (mutation == 4) bad.execution.integer_head = DecodeIntegerHead::a8_iu8;
        rejects([&] { validate_resolved_gdn_recurrent_storage(bad); }, "GDN value-key storage requires eager C1 without graph/integer, islands or R2 routes");
    }
    rejects([] { (void)parse({"decoder", "m", "1", "--gdn-recurrent-storage", "unknown"}); }, "GDN recurrent storage must be key-value or value-key");
    expect(defaults.attention_kv_storage == AttentionKvStorage::bf16 && !defaults.attention_kv_storage_explicit,
        "the BF16 KV cache is the default");
    auto int8_kv = parse({"decoder", "m", "1", "--attention-kv-storage", "kv-int8-tok-v1"});
    expect(int8_kv.attention_kv_storage == AttentionKvStorage::int8_token && int8_kv.attention_kv_storage_explicit,
        "kv-int8-tok-v1 has an explicit selector");
    expect(parse({"decoder", "m", "1", "--attention-kv-storage=bf16"}).attention_kv_storage == AttentionKvStorage::bf16,
        "explicit BF16 KV storage");
    validate_resolved_attention_kv_storage(int8_kv);
    for (unsigned mutation = 0; mutation < 4; ++mutation) {
        auto admitted = int8_kv;
        if (mutation == 0) admitted.known_prefix_window = true;
        if (mutation == 1) admitted.gdn_islands = true;
        if (mutation == 2) admitted.prepared_eager = true;
        if (mutation == 3) admitted.moe_shared_overlap = true;
        validate_resolved_attention_kv_storage(admitted);
    }
    for (unsigned mutation = 0; mutation < 8; ++mutation) {
        auto bad = int8_kv;
        if (mutation == 0) bad.execution.mode = DecodeExecutionMode::graph;
        if (mutation == 1) bad.execution.integer_head = DecodeIntegerHead::a8_iu8;
        if (mutation == 2) bad.prompt_lookup = true;
        if (mutation == 3) bad.prefix_cache_roundtrip = true;
        if (mutation == 4) bad.qsa_gqa_heads_per_block = 2;
        if (mutation == 5) bad.qsa_hierarchical_selection = true;
        if (mutation == 6) bad.serving_concurrency = 2;
        if (mutation == 7) bad.gpu_greedy_token = true;
        rejects([&] { validate_resolved_attention_kv_storage(bad); }, "kv-int8-tok-v1 requires eager C1 decode, known-prefix windows/prefill "
            "or MTP windows without graph/integer head, GPU greedy, prefix cache, prompt lookup, serving, QSA GQA/hierarchical, "
            "RAW/reference or offline tuning");
    }
    rejects([] { (void)parse({"decoder", "m", "1", "--attention-kv-storage", "int8"}); }, "attention KV storage must be bf16 or kv-int8-tok-v1");
    rejects([] { (void)parse({"decoder", "m", "1", "--attention-kv-storage=bf16", "--attention-kv-storage=kv-int8-tok-v1"}); },
        "duplicate attention KV storage option");
    auto instrumented = parse({"decoder", "fixture.gguf", "1", "--host-dispatch-telemetry"});
    const auto instrumented_profile = resolve_optimization_profile(instrumented, model, info, manifest,
        validated_model_content_identity, true);
    expect(instrumented.host_dispatch_telemetry && !instrumented.any_low_level_flag &&
           instrumented_profile.automatic_applied,
           "host observation does not silently disable the qualified automatic profile");
    rejects([] { static_cast<void>(parse({"decoder", "m", "1", "--host-dispatch-telemetry",
        "--host-dispatch-telemetry"})); }, "duplicate host dispatch telemetry option");
    // Policy-only metadata fixture: the content pin is a trusted resolver
    // input here, not evidence that these synthetic metadata own actual assets.
    const auto ordinary_request = parse({"decoder", "fixture.gguf", "7734", "264", "4434", "--generate", "14"});
    auto concurrent = ordinary_request;
    const auto concurrent_profile = resolve_optimization_profile(concurrent, model, info, manifest,
        validated_model_content_identity, true);
    expect(concurrent_profile.automatic_applied &&
        concurrent_profile.selected == "canonical-a8-wave-qsa-hc-shared-output-v5" &&
        concurrent_profile.reason == "qualified-ordinary-cpu-generation" &&
        concurrent.q51_slot_parallel_down && concurrent.batched_moe_activation &&
        concurrent.wave8_q8_gdn_input && concurrent.wave8_q8_gdn_output &&
        concurrent.gdn_islands && concurrent.prepared_eager && concurrent.moe_shared_overlap &&
        concurrent.prefetch_q8_lm_head && concurrent.wave8_q8_lm_head && concurrent.wave8_q8_qsa_output &&
        !concurrent.gpu_greedy_token && concurrent.context_tokens == 8192 &&
        concurrent.moe_overlap_arena_budget_bytes == 64ULL * 1024ULL * 1024ULL,
        "ordinary CPU generation selects the literal complete concurrency profile without scalar output or budget growth");
    expect(concurrent.parallel_hyper_mix && concurrent.staged_hyper_projections && concurrent.fused_staged_hyper_mix &&
        concurrent.typed_hyper_projection && concurrent.prefetch_typed_hyper_rows && concurrent.prefetch_q8_hyper_output &&
        concurrent.lane_chain_hyper_output && concurrent.virtual_q8_lm_head && concurrent.parallel_moe_router &&
        concurrent.wave_moe_router_topk && concurrent.q4k_a8g32_gate_up && concurrent.grouped_exact_q51_down &&
        concurrent.virtual_q8_gdn_input && concurrent.prefetch_q8_gdn_input && concurrent.virtual_q8_gdn_output &&
        concurrent.prefetch_q8_qsa_input && concurrent.prefetch_q8_shared_expert && concurrent.deferred_moe_status,
        "concurrent generation retains every literal baseline prerequisite rather than replacing the original arithmetic bundle");
    // Use the real production resolved validators, not a test copy of the
    // concurrency predicate. These callbacks are metadata-only on this path.
    validate_resolved_mtp_cli(concurrent);
    validate_resolved_q8_projection_organization(concurrent);
    validate_resolved_q51_slot_parallel_down(concurrent);
    validate_resolved_gdn_islands(concurrent);
    validate_resolved_gdn_recurrent_layout(concurrent);
    validate_resolved_gdn_recurrent_storage(concurrent);
    validate_resolved_qsa_hierarchical_selection(concurrent);
    validate_resolved_qsa_gqa(concurrent);
    validate_resolved_known_prefix_window(concurrent);
    validate_resolved_prompt_lookup(concurrent);
    validate_resolved_serving(concurrent);
    const auto submitted_flags = [](const Options& o) {
        return std::array{o.q51_slot_parallel_down,o.batched_moe_activation,o.wave8_q8_gdn_input,
            o.wave8_q8_gdn_output,o.gdn_islands,o.prepared_eager,o.moe_shared_overlap,
            o.prefetch_q8_lm_head,o.wave8_q8_lm_head,o.wave8_q8_qsa_output};
    };
    for (unsigned mutation = 0; mutation < 24; ++mutation) {
        auto ineligible = ordinary_request;
        if (mutation == 0) ineligible.generate_tokens = 0;
        if (mutation == 1) ineligible.tokens.clear();
        if (mutation == 2) ineligible.mtp.enabled = true;
        if (mutation == 3) ineligible.serving_prefix_files = {"prefix.u32le"};
        if (mutation == 4) ineligible.prompt_lookup = true;
        if (mutation == 5) ineligible.known_prefix_window = true;
        if (mutation == 6) ineligible.prefix_cache_roundtrip = true;
        if (mutation == 7) ineligible.raw_recording_path = "fresh-raw";
        if (mutation == 8) ineligible.gpu_greedy_token = true;
        if (mutation == 9) ineligible.logprob_token_ids = {0};
        if (mutation == 10) ineligible.moe_token_cost_policy = true;
        if (mutation == 11) ineligible.ranked_moe_router_topk = true;
        if (mutation == 12) ineligible.offline_tuning_cache_path = "cache.bin";
        if (mutation == 13) ineligible.offline_compile_provenance_path = "provenance.bin";
        if (mutation == 14) ineligible.gdn_recurrent_layout = HipGdnRecurrentLayout::stride32;
        if (mutation == 15) ineligible.gdn_recurrent_storage = GdnRecurrentStorage::value_key;
        if (mutation == 16) ineligible.qsa_hierarchical_selection = true;
        if (mutation == 17) ineligible.qsa_gqa_heads_per_block = 2;
        if (mutation == 18) ineligible.prefetch_hc_projection_next_tile = true;
        if (mutation == 19) ineligible.prefetch_q8_lm_head = true;
        if (mutation == 20) ineligible.wave8_q8_lm_head = true;
        if (mutation == 21) ineligible.wave8_q8_qsa_output = true;
        if (mutation == 23) ineligible.device_queue_trace = ineligible.host_dispatch_telemetry = true;
        // Target-input export is deliberately a separate caller-supplied route
        // bit because its private fixture options are not decoder Options.
        const auto before = ineligible;
        const auto retained = resolve_optimization_profile(ineligible, model, info, manifest,
            validated_model_content_identity, true, mutation != 22);
        expect(retained.automatic_applied && retained.selected == (mutation == 11 ?
            "canonical-a8-ranked-qsa-hc-shared-deferred-v3" : "canonical-a8-wave-qsa-hc-shared-deferred-v3") &&
            submitted_flags(ineligible) == submitted_flags(before),
            "ineligible concurrent request retains v3 and all ten explicit/default submission and projection flags");
    }
    auto preserved_overlay = ordinary_request;
    preserved_overlay.prompt_lookup = true; // An alternate route keeps its explicit compatible arithmetic overlays.
    preserved_overlay.q51_slot_parallel_down = preserved_overlay.wave8_q8_gdn_input = true;
    const auto retained_overlay = resolve_optimization_profile(preserved_overlay, model, info, manifest,
        validated_model_content_identity, true);
    expect(retained_overlay.selected == "canonical-a8-wave-qsa-hc-shared-deferred-v3" &&
        preserved_overlay.q51_slot_parallel_down && preserved_overlay.wave8_q8_gdn_input &&
        !preserved_overlay.batched_moe_activation && !preserved_overlay.wave8_q8_gdn_output &&
        !preserved_overlay.prepared_eager && !preserved_overlay.moe_shared_overlap && !preserved_overlay.gdn_islands,
        "v3 preservation does not clear explicit overlays or complete them into concurrent generation on an excluded route");
    auto manual_generation = ordinary_request;
    manual_generation.optimization_profile = OptimizationProfile::manual;
    manual_generation.prepared_eager = true;
    const auto manual_generation_before = submitted_flags(manual_generation);
    const auto manual_generation_profile = resolve_optimization_profile(manual_generation, model, info, manifest,
        validated_model_content_identity, true);
    expect(!manual_generation_profile.automatic_applied && manual_generation_profile.selected == "manual" &&
        manual_generation_profile.reason == "explicit-manual" &&
        submitted_flags(manual_generation) == manual_generation_before && manual_generation.context_tokens == 128,
        "manual generation preserves its exact explicit submission flags and context");
    auto observational = ordinary_request;
    observational.validation_final_state_path = "fresh-final-state";
    observational.execution_inventory = observational.host_dispatch_telemetry = true;
    expect(resolve_optimization_profile(observational, model, info, manifest,
        validated_model_content_identity, true).selected == "canonical-a8-wave-qsa-hc-shared-output-v5" &&
        observational.validation_final_state_path == "fresh-final-state" &&
        observational.execution_inventory && observational.host_dispatch_telemetry,
        "post-request state validation and host observation do not silently switch arithmetic profiles");
    auto bounded_overlap = parse({"decoder", "fixture.gguf", "7734", "264", "4434", "--generate", "14",
        "--moe-shared-overlap", "--moe-overlap-arena-budget-bytes", "16777216"});
    expect(resolve_optimization_profile(bounded_overlap, model, info, manifest,
        validated_model_content_identity, true).selected == "canonical-a8-wave-qsa-hc-shared-output-v5" &&
        bounded_overlap.moe_overlap_budget_explicit && bounded_overlap.moe_overlap_arena_budget_bytes == 16777216ULL,
        "automatic generation never grows an explicit overlap reservation budget");
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    auto explicit_d = parse({"decoder", "fixture.gguf", "7734", "264", "4434", "--generate", "14", "--optimization-profile", "auto",
        "--q51-slot-parallel-down", "--batched-moe-activation", "--wave8-q8-gdn-input", "--wave8-q8-gdn-output",
        "--gdn-islands", "--prepared-eager", "--moe-shared-overlap",
        "--prefetch-q8-lm-head", "--wave8-q8-lm-head", "--wave8-q8-qsa-output"});
    (void)resolve_optimization_profile(explicit_d, model, info, manifest, validated_model_content_identity, true);
    expect(submitted_flags(explicit_d) == submitted_flags(concurrent) && explicit_d.context_tokens == concurrent.context_tokens &&
        explicit_d.grouped_exact_q51_down && explicit_d.q4k_a8g32_gate_up && explicit_d.deferred_moe_status,
        "actual explicit output parser route agrees with all ten literal automatic flags and original prerequisites");
#endif
    const auto selected = resolve_optimization_profile(on, model, info, manifest,
        validated_model_content_identity, true);
    expect(selected.automatic_applied && selected.selected == "canonical-a8-wave-qsa-hc-shared-deferred-v3" &&
           selected.reason == "qualified-checkpoint", "qualified default selects fast profile");
    expect(on.parallel_hyper_mix && on.staged_hyper_projections && on.fused_staged_hyper_mix &&
           on.typed_hyper_projection && on.prefetch_typed_hyper_rows &&
           on.virtual_q8_lm_head && on.parallel_moe_router && on.wave_moe_router_topk &&
           !on.ranked_moe_router_topk && on.virtual_q8_gdn_input && on.prefetch_q8_gdn_input &&
           on.q4k_a8g32_gate_up && on.grouped_exact_q51_down && on.prefetch_q8_hyper_output &&
           on.virtual_q8_gdn_output && on.lane_chain_hyper_output && on.context_tokens == 8192,
           "exact validated flag bundle and qualified implicit context");
    expect(on.prefetch_q8_qsa_input && on.prefetch_q8_shared_expert &&
           on.deferred_moe_status,
           "automatic candidate includes QSA, shared-expert and deferred status");
    const auto fallback = [&](Options input, const char* reason, bool capability = true,
                              std::string_view identity = validated_model_content_identity) {
        const auto before = input;
        const auto result = resolve_optimization_profile(input, model, info, manifest, identity, capability);
        expect(!result.automatic_applied && result.reason == reason,
               "specific automatic fallback reason");
        expect(input.context_tokens == before.context_tokens &&
               input.virtual_q8_lm_head == before.virtual_q8_lm_head &&
               input.q4k_a8g32_gate_up == before.q4k_a8g32_gate_up &&
               input.lane_chain_hyper_output == before.lane_chain_hyper_output &&
               input.wave_moe_router_topk == before.wave_moe_router_topk &&
               input.ranked_moe_router_topk == before.ranked_moe_router_topk &&
               input.prefetch_q8_qsa_input == before.prefetch_q8_qsa_input &&
               input.prefetch_q8_shared_expert == before.prefetch_q8_shared_expert &&
               input.deferred_moe_status == before.deferred_moe_status &&
               input.q51_slot_parallel_down == before.q51_slot_parallel_down &&
               input.batched_moe_activation == before.batched_moe_activation &&
               input.wave8_q8_gdn_input == before.wave8_q8_gdn_input &&
               input.wave8_q8_gdn_output == before.wave8_q8_gdn_output &&
               input.gdn_islands == before.gdn_islands && input.prepared_eager == before.prepared_eager &&
               input.moe_shared_overlap == before.moe_shared_overlap &&
               input.prefetch_q8_lm_head == before.prefetch_q8_lm_head &&
               input.wave8_q8_lm_head == before.wave8_q8_lm_head &&
               input.wave8_q8_qsa_output == before.wave8_q8_qsa_output,
               "fallback retains explicit and previous defaults");
    };
    fallback(defaults, "checkpoint-content-not-qualified", true, "another-checkpoint");
    fallback(defaults, "a8-runtime-not-built", false);
    fallback(parse({"decoder","m","1","--optimization-profile","manual"}), "explicit-manual");
    fallback(parse({"decoder","m","1","--context","4096"}), "context-not-qualified");
    fallback(parse({"decoder","m","1","--residency","copy"}), "residency-not-qualified");
    fallback(parse({"decoder","m","1","--reference","reference.json"}), "explicit-replay-reference");
    fallback(parse({"decoder","m","1","--planned-ple-io"}), "explicit-low-level-options");
    fallback(parse({"decoder","m","1","--optimization-profile","auto","--planned-ple-io"}),
             "planned-ple-io-explicit");
    fallback(parse({"decoder","m","1","--parallel-hyper-mix"}), "explicit-low-level-options");
    fallback(parse({"decoder","m","1","--coalesce-gap","64"}), "explicit-low-level-options");
    auto explicit_increments = parse({"decoder","m","1","--prefetch-q8-qsa-input",
        "--prefetch-q8-shared-expert","--deferred-moe-status"});
    fallback(explicit_increments, "explicit-low-level-options");
    auto explicit_manual = explicit_increments;
    explicit_manual.optimization_profile = OptimizationProfile::manual;
    fallback(explicit_manual, "explicit-manual");
    auto unqualified_explicit = explicit_increments;
    unqualified_explicit.profile_explicit = true;
    fallback(unqualified_explicit, "checkpoint-content-not-qualified", true, "another-checkpoint");
    for (const auto* flag : {"--parallel-hyper-mix","--prefetch-q8-qsa-input",
        "--prefetch-q8-shared-expert","--deferred-moe-status","--lane-chain-hyper-output",
        "--wave-moe-router-topk","--q4k-a8g32-gate-up","--grouped-exact-q51-down"})
        expect(is_low_level_optimization_option(flag), "all experimental optimization flags tracked");
    expect(!is_low_level_optimization_option("--context") &&
           !is_low_level_optimization_option("--optimization-profile"),
           "ordinary context and explicit profile are distinct from low-level choices");
    auto graph = parse({"decoder","m","1","--execution-mode","graph","--residency","copy",
        "--graph-workspace-bytes","1","--graph-executable-bytes","1",
        "--graph-capture-peak-bytes","1","--graph-budget-bytes","3"});
    fallback(graph, "execution-mode-not-qualified");
    auto integer = parse({"decoder","m","1","--residency","copy","--integer-lm-head","a8-iu8",
        "--integer-internal-bytes","1","--integer-retained-bytes","1"});
    fallback(integer, "integer-head-explicit");
    auto ranked = parse({"decoder","m","1","--parallel-moe-router","--ranked-moe-router-topk"});
    const auto ranked_result = resolve_optimization_profile(ranked, model, info, manifest,
        validated_model_content_identity, true);
    expect(ranked.profile_explicit == false && ranked.any_low_level_flag &&
           ranked.ranked_moe_router_topk && !ranked.wave_moe_router_topk &&
           !ranked.lane_chain_hyper_output && !ranked.q4k_a8g32_gate_up &&
           !ranked.prefetch_q8_qsa_input && !ranked.prefetch_q8_shared_expert &&
           !ranked.deferred_moe_status &&
           ranked_result.selected == "manual" && !ranked_result.automatic_applied &&
           ranked_result.reason == "explicit-low-level-options" && ranked.context_tokens == 128,
           "legacy explicit ranked command is never expanded by implicit auto");
    auto ranked_auto = parse({"decoder","m","1","--parallel-moe-router",
        "--ranked-moe-router-topk","--optimization-profile","auto"});
    const auto ranked_auto_result = resolve_optimization_profile(ranked_auto, model, info, manifest,
        validated_model_content_identity, true);
    expect(ranked_auto.profile_explicit && ranked_auto.ranked_moe_router_topk &&
           !ranked_auto.wave_moe_router_topk && ranked_auto.lane_chain_hyper_output &&
           ranked_auto.prefetch_q8_qsa_input && ranked_auto.prefetch_q8_shared_expert &&
           ranked_auto.deferred_moe_status &&
           ranked_auto_result.selected == "canonical-a8-ranked-qsa-hc-shared-deferred-v3" &&
           ranked_auto_result.reason == "explicit-ranked-router",
           "explicit auto retains ranked routing and adds the candidate bundle");
    auto explicit_context = parse({"decoder","m","1","--context","8192","--optimization-profile","auto"});
    expect(resolve_optimization_profile(explicit_context, model, info, manifest,
        validated_model_content_identity, true).automatic_applied, "explicit qualified context eligible");
    auto roundtrip = parse({"decoder","m","1","--generate","511","--prefix-cache-roundtrip"});
    roundtrip.tokens.assign(43,1);
    expect(resolve_optimization_profile(roundtrip, model, info, manifest,
        validated_model_content_identity, true).automatic_applied && roundtrip.context_tokens == 8192,
        "implicit auto resolves roundtrip context before capacity validation");
    validate_resolved_prefix_roundtrip(roundtrip);
    auto manual_roundtrip = parse({"decoder","m","1","--generate","511",
        "--prefix-cache-roundtrip","--optimization-profile","manual"});
    manual_roundtrip.tokens.assign(43,1);
    (void)resolve_optimization_profile(manual_roundtrip, model, info, manifest,
                                     validated_model_content_identity, true);
    validate_resolved_prefix_roundtrip(manual_roundtrip);
    auto explicit_small = parse({"decoder","m","1","--generate","511",
        "--prefix-cache-roundtrip","--context","128"});
    explicit_small.tokens.assign(43,1);
    expect(resolve_optimization_profile(explicit_small, model, info, manifest,
        validated_model_content_identity, true).reason == "context-not-qualified",
        "explicit small context is never expanded");
    validate_resolved_prefix_roundtrip(explicit_small);
    auto unknown_roundtrip = parse({"decoder","m","1","--generate","511","--prefix-cache-roundtrip"});
    unknown_roundtrip.tokens.assign(43,1);
    (void)resolve_optimization_profile(unknown_roundtrip, model, info, manifest,"unknown",true);
    validate_resolved_prefix_roundtrip(unknown_roundtrip);
    auto long_roundtrip = parse({"decoder","m","1","--generate","511","--prefix-cache-roundtrip"});
    long_roundtrip.tokens.assign(200,1);
    expect(resolve_optimization_profile(long_roundtrip, model, info, manifest,
        validated_model_content_identity, true).automatic_applied && long_roundtrip.context_tokens == 8192,
        "prefix beyond implicit legacy context accepted after automatic expansion");
    validate_resolved_prefix_roundtrip(long_roundtrip);
    manual_roundtrip.tokens.assign(200,1);
    rejects([&] { validate_resolved_prefix_roundtrip(manual_roundtrip); },
        "prefix-cache roundtrip requires at least two generated tokens and remaining context");
    manual_roundtrip.tokens.assign(43,1);
    manual_roundtrip.generate_tokens = 86; //43prompt+85generated-inputs exactly128steps.
    validate_resolved_prefix_roundtrip(manual_roundtrip);
    auto altered_model = model;
    altered_model.hidden_size /= 2;
    auto bad_contract = defaults;
    expect(resolve_optimization_profile(bad_contract, altered_model, info, manifest,
        validated_model_content_identity, true).reason == "model-contract-not-qualified",
        "same claimed content cannot bypass incompatible model geometry");
    auto malformed = manifest;
    ++malformed.layout_identity;
    auto bad_manifest = defaults;
    expect(resolve_optimization_profile(bad_manifest, model, info, malformed,
        validated_model_content_identity, true).reason == "manifest-layout-not-qualified",
        "recomputed manifest identity required");
    const auto numerical_router_info = info; // Preserve canonical metadata before the grouped-router rejection fixture.
    gguf::MetadataEntry group;
    group.key = "qwen3next.expert_group_count";
    info.metadata.push_back(group);
    auto unsupported = defaults;
    expect(resolve_optimization_profile(unsupported, model, info, manifest,
        validated_model_content_identity, true).reason == "router-contract-not-qualified" &&
        !unsupported.wave_moe_router_topk, "unsupported grouped router falls back without opting in");
    rejects([] { (void)parse({"decoder","m","1","--optimization-profile","fast"}); },
        "--optimization-profile requires auto or manual");
    rejects([] { (void)parse({"decoder","m","1","--optimization-profile"}); },
        "--optimization-profile requires auto or manual");
    rejects([] { (void)parse({"decoder","m","1","--optimization-profile","auto",
        "--optimization-profile","manual"}); }, "duplicate optimization profile option");
#ifdef STRIX_HAS_EXPERIMENTAL_Q4K_A8_RUNTIME
    auto legacy_full = parse({"decoder","m","1","--parallel-hyper-mix",
        "--staged-hyper-projections","--fused-staged-hyper-mix","--virtual-q8-lm-head",
        "--parallel-moe-router","--ranked-moe-router-topk","--virtual-q8-gdn-input",
        "--prefetch-q8-gdn-input","--typed-fused-hyper-projections","--prefetch-typed-hyper-rows",
        "--q4k-a8g32-gate-up","--grouped-exact-q51-down","--prefetch-q8-hyper-output",
        "--virtual-q8-gdn-output"});
    const auto legacy_result = resolve_optimization_profile(legacy_full, model, info, manifest,
        validated_model_content_identity, true);
    expect(legacy_result.reason == "explicit-low-level-options" &&
           !legacy_result.automatic_applied && legacy_full.q4k_a8g32_gate_up &&
           legacy_full.grouped_exact_q51_down && legacy_full.virtual_q8_gdn_output &&
           legacy_full.prefetch_q8_hyper_output && legacy_full.ranked_moe_router_topk &&
           !legacy_full.wave_moe_router_topk && !legacy_full.lane_chain_hyper_output &&
           !legacy_full.prefetch_q8_qsa_input && !legacy_full.prefetch_q8_shared_expert &&
           !legacy_full.deferred_moe_status &&
           legacy_full.context_tokens == 128,
           "full fourteen-option legacy baseline remains its explicitly requested configuration");
    auto a8_manual = parse({"decoder","m","1","--optimization-profile","manual",
        "--q4k-a8g32-gate-up","--grouped-exact-q51-down"});
    expect(a8_manual.q4k_a8g32_gate_up && a8_manual.grouped_exact_q51_down,
           "actual A8-enabled CLI retains explicitly requested manual kernels");
#else
    rejects([] { (void)parse({"decoder","m","1","--optimization-profile","manual",
        "--q4k-a8g32-gate-up"}); },
        "--q4k-a8g32-gate-up requires the experimental runtime build capability");
#endif
    // Manual is order-independent and never activates a flag the caller omitted.
    auto manual = parse({"decoder","m","1","--parallel-moe-router",
        "--ranked-moe-router-topk","--optimization-profile","manual"});
    const auto manual_result = resolve_optimization_profile(manual, model, info, manifest,
        validated_model_content_identity, true);
    expect(manual_result.requested == "manual" && manual_result.selected == "manual" &&
           manual_result.reason == "explicit-manual" && !manual_result.automatic_applied &&
           manual.ranked_moe_router_topk &&
           !manual.virtual_q8_lm_head && !manual.q4k_a8g32_gate_up &&
           !manual.lane_chain_hyper_output && !manual.prefetch_q8_qsa_input &&
           !manual.prefetch_q8_shared_expert && !manual.deferred_moe_status &&
           manual.context_tokens == 128,
           "actual application CLI preserves manual baseline without auto dependencies");
    auto manual_hc = parse({"decoder","m","1","--optimization-profile","manual",
        "--parallel-hyper-mix","--staged-hyper-projections","--fused-staged-hyper-mix",
        "--typed-fused-hyper-projections","--prefetch-q8-hyper-output","--lane-chain-hyper-output"});
    const auto explicit_hc = resolve_optimization_profile(manual_hc, model, info, manifest,
        validated_model_content_identity, true);
    expect(explicit_hc.requested == "manual" && explicit_hc.selected == "manual" &&
           explicit_hc.reason == "explicit-manual" && !explicit_hc.automatic_applied &&
           manual_hc.lane_chain_hyper_output &&
           !manual_hc.wave_moe_router_topk && !manual_hc.q4k_a8g32_gate_up &&
           !manual_hc.prefetch_q8_qsa_input && !manual_hc.prefetch_q8_shared_expert &&
           !manual_hc.deferred_moe_status,
           "manual HC selection retains its own dependencies without auto wave or A8");
    auto manual_increments = parse({"decoder","m","1","--optimization-profile","manual",
        "--prefetch-q8-qsa-input","--prefetch-q8-shared-expert","--deferred-moe-status"});
    const auto explicit_increments_result = resolve_optimization_profile(manual_increments,
        model, info, manifest, validated_model_content_identity, true);
    expect(explicit_increments_result.requested == "manual" &&
           explicit_increments_result.selected == "manual" &&
           explicit_increments_result.reason == "explicit-manual" &&
           !explicit_increments_result.automatic_applied &&
           manual_increments.prefetch_q8_qsa_input && manual_increments.prefetch_q8_shared_expert &&
           manual_increments.deferred_moe_status && !manual_increments.wave_moe_router_topk &&
           !manual_increments.lane_chain_hyper_output && !manual_increments.q4k_a8g32_gate_up &&
           manual_increments.context_tokens == 128,
           "manual increments preserve their flags without adding the automatic bundle");
    const std::array<std::string_view, 11> numerical_names{
        "exact-v5", "hc-splitk-f32-v1", "q51-a8-down-v1", "hc-splitk-f32+q51-a8-down-v1",
        "q51-q80-a8-down-v1", "hc-splitk-f32+q51-q80-a8-down-v1",
        "hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup-v1",
        "hc-deferred-norm-f32+q51-q80-a8-down-v1",
        "hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup-v1",
        "hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1",
        "hc-deferred-norm-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1"};
    for (const auto name : numerical_names) {
        const std::string value(name);
        auto candidate = parse({"decoder", "fixture.gguf", "1", "--generate", "2", "--numerical-profile", value.c_str()});
        expect(candidate.numerical_profile_explicit && !candidate.any_low_level_flag &&
            numerical_profile_name(candidate.numerical_profile) == name,
            "typed numerical option is independent of low-level fallback selection");
        const bool dense = name.ends_with("+dense-a8-v1");
        expect(numerical_profile_q5k_a8_gateup(candidate.numerical_profile) ==
                (name.find("+q5k-a8-gateup") != std::string_view::npos) &&
            numerical_profile_dense_a8(candidate.numerical_profile) == dense &&
            numerical_profile_hc_deferred_norm(candidate.numerical_profile) == name.starts_with("hc-deferred-norm-f32") &&
            numerical_profile_hc_splitk(candidate.numerical_profile) == name.starts_with("hc-") &&
            numerical_profile_q51_a8_down(candidate.numerical_profile) == (name.find("q51") != std::string_view::npos) &&
            numerical_profile_q80_a8_down(candidate.numerical_profile) == (name.find("q80") != std::string_view::npos),
            "typed numerical atoms select exact static host recipes for all old and new profiles");
        auto base = resolve_optimization_profile(candidate, model, numerical_router_info, manifest, validated_model_content_identity, true);
        const auto numerical_base_selected = base.selected;
        expect(numerical_base_selected == "canonical-a8-wave-qsa-hc-shared-output-v5" && candidate.prepared_eager &&
            candidate.moe_shared_overlap && candidate.gdn_islands == !dense && candidate.q51_slot_parallel_down &&
            candidate.batched_moe_activation && candidate.wave8_q8_gdn_input && candidate.wave8_q8_gdn_output &&
            candidate.prefetch_q8_lm_head && candidate.wave8_q8_lm_head && candidate.wave8_q8_qsa_output,
            "numerical overlay retains every v5 organization prerequisite");
        apply_numerical_profile(candidate, model, manifest, validated_model_content_identity, true, base);
        expect(base.selected == numerical_base_selected + (name == "exact-v5" ? "" : "+" + value) &&
            base.reason == "qualified-ordinary-cpu-generation" && candidate.q51_slot_parallel_down,
            "numerical recipe adds its identity without changing organization or reason");
        if (name != "exact-v5") {
            if (numerical_profile_hc_deferred_norm(candidate.numerical_profile)) {
                auto partial = candidate; partial.lane_chain_hyper_output = false;
                rejects([&] { auto result = base; apply_numerical_profile(partial, model, manifest,
                    validated_model_content_identity, true, result); },
                    "numerical profile requires the resolved full v5 organization without implicit flag changes");
                expect(!partial.lane_chain_hyper_output,"deferred normalization cannot silently restore a missing lane-chain path");
            }
            auto missing = candidate; missing.q51_slot_parallel_down = false;
            rejects([&] { auto result = base; apply_numerical_profile(missing, model, manifest,
                validated_model_content_identity, true, result); },
                "numerical profile requires the resolved full v5 organization without implicit flag changes");
            expect(!missing.q51_slot_parallel_down, "missing prerequisites are not silently enabled");
            rejects([&] { auto result = base; apply_numerical_profile(candidate, model, manifest, "other-content", true, result); },
                "numerical profile requires the admitted canonical content, model and context");
            rejects([&] { auto result = base; apply_numerical_profile(candidate, model, manifest,
                validated_model_content_identity, false, result); },
                "numerical profile requires the admitted canonical content, model and context");
            auto stale_manifest = manifest; ++stale_manifest.layout_identity;
            rejects([&] { auto result = base; apply_numerical_profile(candidate, model, stale_manifest,
                validated_model_content_identity, true, result); },
                "numerical profile requires the admitted canonical content, model and context");
            auto bad_mode = candidate; bad_mode.execution.mode = DecodeExecutionMode::graph;
            rejects([&] { validate_numerical_profile_cli(bad_mode); },
                "numerical profile requires ordinary eager floating C1 execution without alternate controllers or layouts");
            bad_mode = candidate; bad_mode.known_prefix_window = true;
            rejects([&] { validate_numerical_profile_cli(bad_mode); },
                "numerical profile requires ordinary eager floating C1 execution without alternate controllers or layouts");
            bad_mode = candidate; bad_mode.serving_concurrency = 2;
            rejects([&] { validate_numerical_profile_cli(bad_mode); },
                "numerical profile requires ordinary eager floating C1 execution without alternate controllers or layouts");
            if (dense) {
                auto with_islands = candidate; with_islands.gdn_islands = true;
                rejects([&] { validate_numerical_profile_cli(with_islands); }, "dense A8 numerical profile excludes GDN islands");
            }
        }
    }
    expect(static_cast<int>(DecodeNumericalProfile::exact_v5) == 0 &&
        static_cast<int>(DecodeNumericalProfile::hc_splitk_f32_q51_q80_a8_down_v1) == 5,
        "new arithmetic atoms append enum values without changing the six existing profile identities");
    rejects([] { (void)parse({"decoder", "m", "1", "--numerical-profile", "hc-deferred-norm-fp64-v1"}); },
        "--numerical-profile requires a supported numerical recipe");
    rejects([] { (void)parse({"decoder", "m", "1", "--numerical-profile", "q5k-a8-gateup-v1"}); },
        "--numerical-profile requires a supported numerical recipe");
    rejects([] { (void)parse({"decoder", "m", "1", "--numerical-profile", "hc-splitk-fp64-v1"}); },
        "--numerical-profile requires a supported numerical recipe");
    rejects([] { (void)parse({"decoder", "m", "1", "--numerical-profile"}); },
        "--numerical-profile requires a supported numerical recipe");
    rejects([] { (void)parse({"decoder", "m", "1", "--numerical-profile=unknown"}); },
        "--numerical-profile requires a supported numerical recipe");
    rejects([] { (void)parse({"decoder", "m", "1", "--numerical-profile=exact-v5", "--numerical-profile", "exact-v5"}); },
        "duplicate numerical profile option");
    auto teacher = parse({"decoder", "m", "1", "2", "3", "--generate", "1", "--teacher-forced-score"});
    expect(teacher.teacher_forced_score && !teacher.any_low_level_flag && teacher.logprob_token_ids.empty(),
        "teacher forcing has a separate bounded dynamic-label diagnostic intent");
    auto spool_teacher = parse({"decoder", "m", "1", "2", "--generate", "1", "--teacher-forced-score", "--teacher-logits-file", "fresh-logits.bin"});
    expect(spool_teacher.teacher_logits_path == "fresh-logits.bin" && !spool_teacher.any_low_level_flag,
        "teacher full-logit spool preserves diagnostic intent without selecting a numerical recipe");
    auto equals_spool = parse({"decoder", "m", "1", "2", "--generate", "1", "--teacher-forced-score", "--teacher-logits-file=fresh-logits.bin"});
    expect(equals_spool.teacher_logits_path == spool_teacher.teacher_logits_path,"teacher spool accepts equal syntax");
    rejects([] { (void)parse({"decoder", "m", "1", "2", "--generate", "1", "--teacher-logits-file", "unused.bin"}); },
        "teacher logits file requires teacher-forced score");
    rejects([] { (void)parse({"decoder", "m", "1", "2", "--generate", "1", "--teacher-forced-score", "--teacher-logits-file="}); },
        "teacher logits file requires a nonempty path");
    rejects([] { (void)parse({"decoder", "m", "1", "2", "--generate", "1", "--teacher-forced-score", "--teacher-logits-file"}); },
        "teacher logits file requires a nonempty path");
    rejects([] { (void)parse({"decoder", "m", "1", "2", "--generate", "1", "--teacher-forced-score", "--teacher-logits-file=a", "--teacher-logits-file=b"}); },
        "duplicate teacher logits file option");
    {
        const TeacherRowsDirectory rows_dir, other_dir;
        expect(rows_dir.path != other_dir.path, "teacher row fixtures own distinct directories");
        const auto rows_path = (rows_dir.path / "rows.txt").string();
        { std::ofstream rows_file(rows_path); rows_file << "0 1 "; }
        auto rows_teacher = parse({"decoder", "m", "1", "2", "3", "--generate", "1", "--teacher-forced-score",
            "--teacher-logits-file", "fresh-logits.bin", "--teacher-logits-rows", rows_path.c_str()});
        expect(rows_teacher.teacher_logits_rows == std::vector<std::uint64_t>{0U, 1U} && !rows_teacher.any_low_level_flag,
            "teacher logits rows keep the listed positions without selecting a numerical recipe");
        rejects([&] { (void)parse({"decoder", "m", "1", "2", "--generate", "1", "--teacher-forced-score",
            "--teacher-logits-rows", rows_path.c_str()}); }, "teacher logits rows require a teacher logits file");
        { std::ofstream rows_file(rows_path); rows_file << "1 1 "; }
        rejects([&] { (void)parse({"decoder", "m", "1", "2", "--generate", "1", "--teacher-forced-score",
            "--teacher-logits-file", "a.bin", "--teacher-logits-rows", rows_path.c_str()}); },
            "teacher logits rows must be ascending decimal positions");
        std::istringstream long_word(std::string(100000U, '0'));
        rejects([&] { (void)read_teacher_logits_rows(long_word); },
            "teacher logits rows must be ascending decimal positions");
        expect(long_word.tellg() <= std::streampos(20), "oversized row token rejected without reading its tail");
        std::ostringstream many_rows;
        for (unsigned row = 0; row < 65536U; ++row) many_rows << row << ' ';
        const auto limit_bytes = many_rows.str().size();
        std::istringstream too_many(many_rows.str() + "65536 invalid-tail");
        rejects([&] { (void)read_teacher_logits_rows(too_many); },
            "teacher logits rows file must list 1..65536 positions");
        expect(too_many.tellg() == static_cast<std::streamoff>(limit_bytes), "row limit checked before reading position 65537");
        std::istringstream at_limit(many_rows.str());
        expect(read_teacher_logits_rows(at_limit).size() == 65536U, "row limit remains inclusive");
        std::istringstream maximum("18446744073709551615");
        expect(read_teacher_logits_rows(maximum).back() == UINT64_MAX, "maximum uint64 row is accepted");
    }
    validate_resolved_teacher_forced_score(teacher, 4);
    rejects([&] { validate_resolved_teacher_forced_score(teacher, 3); }, "teacher input exceeds model vocabulary");
    auto short_context = teacher; short_context.context_tokens = 2;
    rejects([&] { validate_resolved_teacher_forced_score(short_context, 4); }, "teacher-forced score vocabulary or context is invalid");
    for (const std::uint64_t generation : {0ULL, 2ULL}) {
        auto invalid = teacher; invalid.generate_tokens = generation;
        rejects([&] { validate_teacher_forced_score(invalid); },
            "teacher-forced score requires at least two teacher inputs and generate 1 in serial eager full-output diagnostic mode");
    }
    rejects([] { (void)parse({"decoder", "m", "1", "--generate", "1", "--teacher-forced-score"}); },
        "teacher-forced score requires at least two teacher inputs and generate 1 in serial eager full-output diagnostic mode");
    rejects([] { (void)parse({"decoder", "m", "1", "2", "--generate", "1", "--teacher-forced-score", "--teacher-forced-score"}); },
        "duplicate teacher-forced score option");
    auto incompatible_teacher = teacher; incompatible_teacher.logprob_token_ids = {1};
    rejects([&] { validate_teacher_forced_score(incompatible_teacher); },
        "teacher-forced score requires at least two teacher inputs and generate 1 in serial eager full-output diagnostic mode");
    incompatible_teacher = teacher; incompatible_teacher.mtp.enabled = true;
    rejects([&] { validate_teacher_forced_score(incompatible_teacher); },
        "teacher-forced score requires at least two teacher inputs and generate 1 in serial eager full-output diagnostic mode");
    // Teacher forcing over the tiled known-prefix prefill: full rows, T rows and every row's head.
    auto tiled_teacher = teacher;
    tiled_teacher.known_prefix_window = true; tiled_teacher.known_prefix_full_rows = true;
    tiled_teacher.known_prefix_prefill_rows = 256; tiled_teacher.known_prefix_expert_union = true;
    expect(teacher_forced_tiled(tiled_teacher) && !teacher_forced_tiled(teacher), "tiled teacher route is selected by the tiled prefill flags");
    validate_teacher_forced_score(tiled_teacher);
    validate_resolved_teacher_forced_score(tiled_teacher, 4);
    const auto tiled_message = "teacher-forced score over a known prefix requires the tiled prefill "
        "(--known-prefix-window --known-prefix-full-rows --known-prefix-prefill-rows T) with the all-rows output policy";
    for (const auto policy : {KnownWindowOutputPolicy::terminal_row, KnownWindowOutputPolicy::state_only}) {
        auto invalid = tiled_teacher; invalid.known_prefix_output_policy = policy;
        rejects([&] { validate_teacher_forced_score(invalid); }, tiled_message);
    }
    auto untiled = tiled_teacher; untiled.known_prefix_prefill_rows = 0;
    rejects([&] { validate_teacher_forced_score(untiled); }, tiled_message);
    untiled = tiled_teacher; untiled.known_prefix_full_rows = false;
    rejects([&] { validate_teacher_forced_score(untiled); }, tiled_message);
    untiled = tiled_teacher; untiled.known_prefix_window = false;
    rejects([&] { validate_teacher_forced_score(untiled); }, tiled_message);
    untiled = teacher; untiled.known_prefix_expert_union = true;
    rejects([&] { validate_teacher_forced_score(untiled); }, tiled_message);
    auto tiled_invalid = tiled_teacher; tiled_invalid.generate_tokens = 2;
    rejects([&] { validate_teacher_forced_score(tiled_invalid); },
        "teacher-forced score requires at least two teacher inputs and generate 1 in serial eager full-output diagnostic mode");
    tiled_invalid = tiled_teacher; tiled_invalid.prompt_lookup = true;
    rejects([&] { validate_teacher_forced_score(tiled_invalid); },
        "teacher-forced score requires at least two teacher inputs and generate 1 in serial eager full-output diagnostic mode");
    tiled_invalid = tiled_teacher; tiled_invalid.mtp.enabled = true;
    rejects([&] { validate_teacher_forced_score(tiled_invalid); },
        "teacher-forced score requires at least two teacher inputs and generate 1 in serial eager full-output diagnostic mode");
    auto tiled_parsed = parse({"decoder", "m", "1", "2", "3", "--generate", "1", "--teacher-forced-score",
        "--known-prefix-window", "--known-prefix-full-rows", "--known-prefix-prefill-rows", "16",
        "--known-prefix-workspace-budget-bytes", "8589934592", "--known-prefix-host-metadata-budget-bytes", "67108864",
        "--known-prefix-uma-budget-bytes", "128849018880", "--known-prefix-host-headroom-bytes", "4294967296"});
    expect(teacher_forced_tiled(tiled_parsed) && tiled_parsed.known_prefix_output_policy == KnownWindowOutputPolicy::all_rows,
        "tiled teacher forcing parses with the default all-rows policy");
    rejects([] { (void)parse({"decoder", "m", "1", "2", "3", "--generate", "1", "--teacher-forced-score",
        "--known-prefix-window", "--known-prefix-full-rows", "--known-prefix-prefill-rows", "16",
        "--known-prefix-output-policy", "terminal-row",
        "--known-prefix-workspace-budget-bytes", "8589934592", "--known-prefix-host-metadata-budget-bytes", "67108864",
        "--known-prefix-uma-budget-bytes", "128849018880", "--known-prefix-host-headroom-bytes", "4294967296"}); },
        tiled_message);
    // Stored-byte relocation has a separate closed admission contract and never
    // acts as a low-level organization flag.
    auto copy_options = parse({"decoder", "fixture.gguf", "1", "--generate", "2",
        "--model-copy-tier", "hot_nonexpert", "--model-copy-payload-bytes", "4830154240",
        "--model-retained-budget", "120000000000", "--model-setup-peak-budget", "120000000000"});
    expect(copy_options.model_copy_tier == HipModelCopyTier::hot_nonexpert &&
        copy_options.model_copy_payload_bytes == 4830154240ULL && !copy_options.any_low_level_flag,
        "selective model copy records exact payload without disabling automatic organization");
    auto copy_resolution = resolve_optimization_profile(copy_options, model, numerical_router_info, manifest,
        validated_model_content_identity, true);
    const auto copy_base_selected = copy_resolution.selected;
    const auto copy_base_reason = copy_resolution.reason;
    apply_numerical_profile(copy_options, model, manifest, validated_model_content_identity, true, copy_resolution);
    apply_model_copy_profile(copy_options, copy_resolution);
    expect(copy_base_selected == "canonical-a8-wave-qsa-hc-shared-output-v5" &&
        copy_resolution.selected == copy_base_selected + "+model-copy-tier:hot_nonexpert" &&
        copy_resolution.reason == copy_base_reason && copy_resolution.automatic_applied &&
        copy_options.prepared_eager && copy_options.moe_shared_overlap && copy_options.gdn_islands &&
        copy_options.wave8_q8_lm_head && copy_options.wave8_q8_qsa_output,
        "selective relocation appends only its scope and retains the resolved v5 organization");
    auto copy_none = parse({"decoder", "m", "1", "--model-copy-tier=none",
        "--model-retained-budget=120000000000", "--model-setup-peak-budget=120000000000"});
    OptimizationProfileResolution copy_none_resolution;
    copy_none_resolution.selected = "unchanged";
    apply_model_copy_profile(copy_none, copy_none_resolution);
    expect(copy_none_resolution.selected == "unchanged" && !copy_none.any_low_level_flag &&
        !copy_none.model_copy_payload_bytes, "none preserves the default profile and permits independent reservation limits");
    auto copy_norm = parse({"decoder", "m", "1", "2", "--generate", "1", "--teacher-forced-score",
        "--teacher-logits-file", "fresh-selective-logits.bin", "--execution-inventory",
        "--model-copy-tier=normalization", "--model-copy-payload-bytes=3997696",
        "--model-retained-budget=120000000000", "--model-setup-peak-budget=120000000000"});
    expect(copy_norm.teacher_forced_score && copy_norm.execution_inventory &&
        copy_norm.model_copy_tier == HipModelCopyTier::normalization && !copy_norm.any_low_level_flag,
        "selective copy admits full-logit teacher diagnostics and inventory");
    auto copy_allowed = copy_options;
    copy_allowed.raw_recording_path = "future-raw-directory";
    copy_allowed.validation_final_state_path = "future-state-directory";
    copy_allowed.logprob_token_ids = {1};
    validate_model_copy_cli(copy_allowed);
    const auto reject_copy = [&](auto mutate, std::string_view message) {
        auto invalid_copy = copy_options;
        mutate(invalid_copy);
        rejects([&] { validate_model_copy_cli(invalid_copy); }, message);
    };
    reject_copy([](auto& o) { o.model_copy_payload_bytes.reset(); },
        "selective model copy requires positive exact payload, retained and setup-peak budgets");
    reject_copy([](auto& o) { o.model_retained_budget.reset(); },
        "selective model copy requires positive exact payload, retained and setup-peak budgets");
    reject_copy([](auto& o) { o.model_setup_peak_budget.reset(); },
        "selective model copy requires positive exact payload, retained and setup-peak budgets");
    reject_copy([](auto& o) { o.residency = HipModelResidencyMode::resident_copy; },
        "selective model copy requires register residency");
    reject_copy([](auto& o) { o.execution.mode = DecodeExecutionMode::graph; },
        "selective model copy requires eager execution");
    reject_copy([](auto& o) { o.mtp.enabled = true; }, "selective model copy excludes MTP options");
    reject_copy([](auto& o) { o.mtp.supplied = true; }, "selective model copy excludes MTP options");
    reject_copy([](auto& o) { o.serving_memory_explicit = true; }, "selective model copy excludes serving options");
    reject_copy([](auto& o) { o.serving_prefill_explicit = true; }, "selective model copy excludes serving options");
    reject_copy([](auto& o) { o.execution.integer_head = DecodeIntegerHead::a8_iu8; },
        "selective model copy excludes integer head");
    reject_copy([](auto& o) { o.offline_tuning_cache_path = "cache"; },
        "selective model copy excludes offline tuning and compile provenance");
    reject_copy([](auto& o) { o.offline_compile_provenance_path = "provenance"; },
        "selective model copy excludes offline tuning and compile provenance");
    reject_copy([](auto& o) { o.known_prefix_window = true; },
        "selective model copy excludes known-prefix window and prefill");
    reject_copy([](auto& o) { o.known_prefix_prefill_rows = 16; },
        "selective model copy excludes known-prefix window and prefill");
    reject_copy([](auto& o) { o.prompt_lookup = true; }, "selective model copy excludes prompt lookup");
    reject_copy([](auto& o) { o.prefix_cache_roundtrip = true; }, "selective model copy excludes prefix cache roundtrip");
    reject_copy([](auto& o) { o.gpu_greedy_token = true; }, "selective model copy excludes GPU greedy token output");
    reject_copy([](auto& o) { o.planned_ple_io = true; }, "selective model copy excludes planned PLE IO");
    reject_copy([](auto& o) { o.numerical_profile = DecodeNumericalProfile::hc_splitk_f32_v1; },
        "selective model copy requires exact-v5 or an admitted explicit expert recipe");
    rejects([] { (void)parse({"decoder", "m", "1", "--model-copy-tier=unknown"}); },
        "model copy tier must be none, normalization, hot_nonexpert or experts");
    rejects([] { (void)parse({"decoder", "m", "1", "--model-copy-tier"}); },
        "model copy option requires a value");
    rejects([] { (void)parse({"decoder", "m", "1", "--model-copy-tier=none", "--model-copy-tier=none"}); },
        "duplicate model copy tier option");
    rejects([] { (void)parse({"decoder", "m", "1", "--model-retained-budget=1", "--model-retained-budget=2"}); },
        "duplicate model copy budget option");
    rejects([] { (void)parse({"decoder", "m", "1", "--model-copy-payload-bytes=0"}); },
        "model copy budgets must be positive");
    rejects([] { (void)parse({"decoder", "m", "1", "--model-copy-payload-bytes=1"}); },
        "model copy payload requires a selective model copy tier");
    std::cout << "Decode optimization profile failures=" << failures << '\n';
    return failures ? 1 : 0;
}

