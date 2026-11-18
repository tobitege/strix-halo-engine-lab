#include "../apps/qwen_effective_config.hpp"
#include <iostream>
#include <map>
#include <type_traits>

namespace pr = strix::qwen::profile;
namespace dc = strix::decode_cli;
namespace {
unsigned checks = 0, fixtures = 0;
bool emit = false;
using Env = std::map<std::string, std::string>;
void check(bool ok, const char* message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
pr::Lookup lookup(const Env& env) {
    return [&env](const char* key) -> const char* {
        const auto it = env.find(key);
        return it == env.end() ? nullptr : it->second.c_str();
    };
}
void fixture(std::string_view label, const std::string& value) {
    ++fixtures;
    if (emit) std::cout << fixtures << ' ' << label << ' ' << value << '\n';
}
std::string pairs(const Env& env) {
    auto json = strix::serve::Json::object();
    for (const auto& [key, value] : env) json.set(key, strix::serve::Json(value));
    return json.dump();
}
void profile_fixture(pr::Name profile, const pr::Inputs& input, std::span<const pr::Lever> rows) {
    try {
        const auto resolved = pr::resolve(profile, input, rows);
        auto json = strix::serve::Json::object();
        json.set("resolution", strix::serve::parse_json(resolved.json()));
        auto notices = strix::serve::Json::array();
        for (const auto& notice : resolved.notices()) notices.push(strix::serve::Json(notice));
        json.set("notices", std::move(notices));
        Env exported;
        pr::apply(resolved, [&](const char* key, const char* value) { exported[key] = value; return true; });
        json.set("exports", strix::serve::parse_json(pairs(exported)));
        fixture("profile", json.dump());
        Env inherited;
        for (const auto& row : rows) if (!row.env.empty()) {
            if (const auto* value = pr::environment(input.env, std::string(row.env))) inherited[std::string(row.env)] = value;
            if (const auto* value = pr::environment(input.env, pr::marker_key(row))) inherited[pr::marker_key(row)] = value;
        }
        for (const auto& [key, value] : exported) inherited[key] = value;
        auto restarted_input = input;
        restarted_input.env = lookup(inherited);
        fixture("inherited", pr::resolve(profile, restarted_input, rows).json());
    } catch (const std::exception& error) { fixture("refusal", error.what()); }
}
void product_matrix() {
    for (auto profile : {pr::Name::standard, pr::Name::fast, pr::Name::quality}) for (bool allow : {false, true}) {
        Env env;
        pr::Inputs input{{}, lookup(env), allow};
        profile_fixture(profile, input, pr::levers());
        for (const auto& row : pr::levers()) {
            for (auto value : {row.off, row.builtin, std::string_view{}, row.value[1]}) {
                env.clear(); input.cli.clear();
                if (!row.env.empty()) {
                    env[std::string(row.env)] = value;
                    profile_fixture(profile, input, pr::levers());
                }
                input.cli = {{std::string(row.name), std::string(value)}};
                profile_fixture(profile, input, pr::levers());
            }
        }
        env.clear(); input.cli.clear(); env["STRIX_QWEN_PROFILE_ALLOW_PENDING"] = "1";
        profile_fixture(profile, input, pr::levers());
    }
}
void admission_matrix() {
    // Exercise the shared decision through Qwen, including its skipped empty
    // columns, builtin classification, native exports and inherited markers.
    for (bool ready : {false,true}) for (bool runnable : {false,true})
    for (bool provisional : {false,true}) for (bool allow : {false,true})
    for (bool native : {false,true}) for (auto wanted : {"", "0", "1"}) {
        pr::Lever row{.name="candidate", .env="STRIX_TEST_CANDIDATE", .option="--candidate",
            .value={wanted,wanted,wanted}, .builtin="0", .kind=pr::Kind::bit_changing,
            .ready={ready,ready,ready}, .runnable={runnable,runnable,runnable},
            .target="declared target", .off="0", .source="host fixture", .native_option=native,
            .provisional={provisional,provisional,provisional}};
        const auto rows = std::span<const pr::Lever>(&row, 1);
        for (auto profile : {pr::Name::standard,pr::Name::fast,pr::Name::quality}) {
            Env env;
            pr::Inputs input{{},lookup(env),allow};
            const auto r = pr::resolve(profile,input,rows);
            if (!*wanted) check(r.entries.empty(), "empty columns are skipped");
            else {
                const auto status = !ready && !((provisional || allow) && runnable) ? pr::Status::pending :
                    provisional ? pr::Status::provisional : !ready ? pr::Status::unqualified :
                    std::string_view(wanted)=="0" ? pr::Status::builtin : pr::Status::set;
                check(r.entries.size()==1 && r.entries[0].status==status, "Qwen column admission unchanged");
                check(r.entries[0].value==(status==pr::Status::pending ? "0" : wanted), "pending retains builtin");
            }
            profile_fixture(profile,input,rows);
            env[std::string(row.env)]="0";
            check(pr::resolve(profile,input,rows).entries[0].status==pr::Status::off, "each OFF beats profile");
            profile_fixture(profile,input,rows);
            env[std::string(row.env)]="1";
            input.cli={{"candidate","0"}};
            const auto cli=pr::resolve(profile,input,rows);
            check(cli.entries[0].status==pr::Status::off && cli.entries[0].cli_override, "CLI OFF beats environment");
            profile_fixture(profile,input,rows);
            input.cli.clear(); env.clear();
            env[std::string(row.env)]=wanted;
            for (auto status : {pr::Status::set,pr::Status::unqualified,pr::Status::provisional}) {
                env[pr::marker_key(row)]=pr::marker(profile,status,wanted);
                profile_fixture(profile,input,rows);
            }
            // Route admission may decline a proposal after resolution. Withdrawal
            // must remove only a matching owned export, with unchanged reason JSON.
            auto declined=r;
            if (!declined.entries.empty()) {
                auto& entry=declined.entries[0]; entry.status=pr::Status::pending;
                entry.value="0"; entry.target="declared target"; entry.reason="route fixture refusal";
                env[std::string(row.env)]="1";
                env[pr::marker_key(row)]=pr::marker(profile,pr::Status::provisional,"1");
                pr::withdraw_declined(declined,lookup(env),[&](const char* key) { env.erase(key); return true; });
                check(env.empty(), "declined owned proposal withdrawn");
                fixture("declined",declined.json()+" "+pairs(env));
            }
        }
    }
}
dc::Options options(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    return dc::parse_options(static_cast<int>(argv.size()),argv.data(),true);
}
void defaults_matrix() {
    const auto saved=strix::qpf2k::baseline;
    for (bool base : {false,true}) for (auto allowed : {~0U,strix::qpf2k::startup_items}) {
        strix::qpf2k::baseline=base;
        const auto capture=[&](Env env) {
            const auto original=env;
            const auto injected=strix::qpf2k::initialize(lookup(env),[&](const char* key,const char* value) {
                env[key]=value; return true;
            },allowed);
            if (base || original.contains("STRIX_BASELINE") || original.contains("STRIX_QPF2K_DEFAULTS"))
                check(injected==0 && env==original,"baseline/master exports nothing");
            for (const auto& row : strix::qpf2k::defaults) if (original.contains(row.name))
                check(env.at(row.name)==original.at(row.name) && !(injected & strix::qpf2k::bit(row.item)),
                    "explicit OFF/empty/default retains value and provenance");
            fixture("defaults",std::to_string(injected)+" "+pairs(env));
        };
        capture({}); capture({{"STRIX_BASELINE","1"}}); capture({{"STRIX_QPF2K_DEFAULTS","0"}});
        for (const auto& row : strix::qpf2k::defaults) for (auto value : {"0",""}) capture({{row.name,value}});
    }
    strix::qpf2k::baseline=saved;
}
void pending_weights_inspection() {
    const auto o = options({"decoder", "model.gguf", "1", "--profile", "turbo", "--turbo-weights", "converted",
                            "--print-effective-config"});
    const auto c = dc::resolve_effective_config({o, {}}, strix::ModelContract::qwen38_flash_next(), {}, {}, "", {}, {});
    const auto text = dc::effective_config_json(c);
    check(text.find("converted") != std::string::npos && text.find("pending") != std::string::npos,
          "inspection must serialize the refused weight request");
    bool refused = false;
    bool exported = false;
    try { pr::apply(c.options.product_profile, [&](const char*, const char*) { exported = true; return true; }); }
    catch (const std::invalid_argument&) { refused = true; }
    check(refused && !exported, "execution must refuse pending weights before exporting");
}

void config_matrix() {
    for (auto profile : {"default","fast","quality"}) for (bool allow : {false,true}) for (bool base : {false,true}) {
        std::vector<std::string> args{"decoder","model.gguf","1","--profile",profile};
        if (allow) args.push_back("--allow-pending");
        if (base) args.push_back("--baseline");
        const auto o=options(args);
        const auto capture=[&](const Env& env) {
            const auto c=dc::resolve_effective_config({o,env},strix::ModelContract::qwen38_flash_next(),{},{},"",{},{});
            fixture("config",dc::effective_config_json(c));
            fixture("effective-argv",[&] {
                auto a=strix::serve::Json::array();
                for (const auto& value : dc::effective_prefill_argv(c.options)) a.push(strix::serve::Json(value));
                return a.dump();
            }());
        };
        capture({});
        capture({{"STRIX_BASELINE","1"}});
        capture({{"STRIX_QPF2K_DEFAULTS","0"}});
        // The actual defaults initializer has parent-mask and baseline admission;
        // compare its exports/provenance independently for every OFF row.
        for (const auto& row : strix::qpf2k::defaults) {
            Env env{{row.name,"0"}};
            capture(env);
            const auto injected=strix::qpf2k::initialize(lookup(env),[&](const char* key,const char* value) {
                env[key]=value; return true;
            });
            fixture("qpf2k",std::to_string(injected)+" "+pairs(env));
        }
    }
    for (const auto& args : std::vector<std::vector<std::string>>{
            {"--profile"},{"--profile","turbbo"},{"--profile","fast","--profile=default"},
            {"--output-head-profile","fp16-output-head"}}) {
        auto argv=std::vector<std::string>{"decoder","model.gguf","1"}; argv.insert(argv.end(),args.begin(),args.end());
        try { (void)options(argv); check(false,"invalid CLI must refuse"); }
        catch (const std::invalid_argument& e) { fixture("parser-refusal",e.what()); }
    }
}
std::uint64_t legacy_hash(std::string_view s) {
    std::uint64_t h=14695981039346656037ULL;
    for (char c : s) { h^=static_cast<unsigned char>(c); h*=1099511628211ULL; }
    return h;
}
void identities() {
    for (unsigned i=0;i<=static_cast<unsigned>(strix::DecodeNumericalProfile::dense_trellis_g1_fp32_v1);++i) {
        const auto p=static_cast<strix::DecodeNumericalProfile>(i);
        check(strix::numerical_profile_identity(p)==legacy_hash(strix::numerical_profile_name(p)),"historical profile name hash");
        for (unsigned j=0;j<=static_cast<unsigned>(strix::DecodeNumericalProfile::dense_trellis_g1_fp32_v1);++j)
        for (unsigned components=0;components<=strix::kFastPrefillKnown;++components) {
            const strix::NumericalProfilePair pair{p,static_cast<strix::DecodeNumericalProfile>(j),components};
            fixture("pair-hash",std::to_string(strix::numerical_profile_pair_identity(pair)));
        }
    }
    for (unsigned mask=0;mask<(1U<<strix::kPrefillOptions.size());++mask) {
        check(strix::prefill_identity_hash(mask)==legacy_hash(strix::prefill_identity_name(mask)),"historical prefill name hash");
        for (std::uint64_t seed : {UINT64_C(0),UINT64_C(1),UINT64_C(0xcbf29ce484222325),UINT64_C(0xffffffffffffffff)})
            fixture("bind-hash",std::to_string(strix::bind_prefill_option_identity(seed,mask)));
    }
}
#ifndef STRIX_STEP1_ORIGINAL
void common_adapters() {
    static_assert(std::is_same_v<pr::Name,strix::profile::Name> && std::is_same_v<pr::Lever,strix::profile::Lever> &&
        std::is_same_v<pr::Entry,strix::profile::Entry> && std::is_base_of_v<strix::profile::DefaultItem,strix::qpf2k::Default>);
    for (const auto& row : strix::qpf2k::defaults) for (bool base : {false,true}) {
        Env env; const auto get=lookup(env);
        const auto on=strix::profile::resolve_default(row,base,get);
        check(on.value==(base ? row.off : row.on),"common default/baseline for every row");
        env[row.env]="";
        check(strix::profile::resolve_default(row,base,get).source==strix::profile::DefaultSource::env &&
            strix::profile::resolve_default(row,base,get).value.empty(),"explicit empty environment retained");
        check(strix::profile::resolve_default(row,base,get,"cli").value=="cli","common CLI beats empty env");
    }
    const auto o=options({"decoder","model.gguf","1","--profile","fast"});
    const auto c=dc::initialize_prefill_config({o,{}});
    const auto r=dc::host_receipt(c);
    check(r.engine=="qwen" && r.identity.arithmetic_contract.find("decode=exact-v5")!=std::string::npos,
        "host receipt records resolved numerical contract");
    check(r.identity.model_sha256.empty() && r.identity.pack_hash.empty() && r.identity.table_hash.empty() &&
        r.identity.state_format.empty() && r.identity.prefill_lineage.empty() && r.identity.reduction_plan.empty() &&
        r.implementation.empty() && r.build.empty() && r.kernel.empty() && r.device.empty() && r.topology.empty() && r.digests.empty(),
        "unknown execution and asset identities remain empty");
    const auto lineage=dc::host_receipt(c,"observed-lineage");
    check(lineage.identity.prefill_lineage=="observed-lineage" && lineage.identity.arithmetic_contract==r.identity.arithmetic_contract &&
        lineage.identity.id()!=r.identity.id(),"lineage is independent of arithmetic contract");
    check(r.witness()=="strix-engine-witness "+r.json()+"\n","canonical witness prefix");
    check(c.options.product_profile.json().find("\"engine\"")==std::string::npos,"legacy JSON has no engine tag");
}
#endif
}
int main(int argc,char** argv) try {
    emit=argc==2 && std::string_view(argv[1])=="--fixtures";
    pending_weights_inspection();
    product_matrix(); admission_matrix(); defaults_matrix(); config_matrix(); identities();
#ifndef STRIX_STEP1_ORIGINAL
    common_adapters();
#endif
    if (!emit) std::cout << checks << " checks passed; " << fixtures << " deterministic fixtures\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
