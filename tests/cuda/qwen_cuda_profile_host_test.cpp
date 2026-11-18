// Host-only: the CUDA Qwen apps resolve profiles through the shared resolver. The same request through the serve entry
// and the prefill-service / CLI entry must give the same Resolution and the same engine values, except the named role
// differences. No CUDA, no GPU, no process environment (the resolver reads the lookup it is given).
#include "strix/cuda/qwen_options.hpp"
#include "strix/cuda/qwen_prefix_identity.hpp"
#include "strix/rows_fit.hpp"

#include <cstdio>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace qo = strix::cuda::qwen_options;
namespace cp = strix::qwen::cuda_profile;
namespace pr = strix::qwen::profile;
namespace px = strix::cuda::qwen_prefix;
namespace qp = strix::cuda::qwen_prefill;
namespace fs = std::filesystem;
using Args = std::vector<std::string>;
using Env = std::map<std::string, std::string>;

namespace {
int failures = 0;
void check(bool ok, const std::string& why) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why.c_str()); ++failures; }
}
// `env` must outlive the returned request's use (the lookup captures it).
qo::Request parse(const Args& args, const Env& env) {
    std::vector<std::string> storage{"app"};
    for (const auto& a : args) storage.push_back(a);
    std::vector<char*> argv;
    for (auto& s : storage) argv.push_back(s.data());
    qo::Request r;
    r.inputs.env = [&env](const char* key) -> const char* {
        const auto it = env.find(key);
        return it == env.end() ? nullptr : it->second.c_str();
    };
    const int argc = static_cast<int>(argv.size());
    for (int i = 1; i < argc; ++i)
        if (!qo::take(argc, argv.data(), i, r)) throw std::invalid_argument("unknown " + storage[static_cast<std::size_t>(i)]);
    return r;
}
const Args kPaths{"--model", "m.gguf", "--tables", "t", "--rope-table", "r.bin"};
Args with_paths(Args a) { a.insert(a.end(), kPaths.begin(), kPaths.end()); return a; }
qo::Resolved one(const Args& a, qo::Entry e, const Env& env = {}) { return qo::resolve(parse(with_paths(a), env), e); }

// Every Config field the entries share, by name, that differs between two configs.
std::set<std::string> diff(const qp::Config& a, const qp::Config& b) {
    std::set<std::string> d;
    const auto ne = [&](const char* name, bool differs) { if (differs) d.insert(name); };
    ne("model_gguf", a.model_gguf != b.model_gguf);
    ne("expert_arithmetic",a.expert_arithmetic!=b.expert_arithmetic);
    ne("mtp_sidecar_gguf", a.mtp_sidecar_gguf != b.mtp_sidecar_gguf);
    ne("tables_dir", a.tables_dir != b.tables_dir);
    ne("rope_table", a.rope_table != b.rope_table);
    ne("max_positions", a.max_positions != b.max_positions);
    ne("block_schedule", a.block_schedule != b.block_schedule);
    ne("tail_large", a.tail_large != b.tail_large);
    ne("final_block_rows", a.final_block_rows != b.final_block_rows);
    ne("block_fit", a.block_fit != b.block_fit);
    ne("block_fit_reserve_mib", a.block_fit_reserve_mib != b.block_fit_reserve_mib);
    ne("device", a.device != b.device);
    ne("ple_host_pinned", a.ple_host_pinned != b.ple_host_pinned);
    ne("release_by_layer", a.release_by_layer != b.release_by_layer);
    ne("decode", a.decode != b.decode);
    ne("draft_vocab", a.draft_vocab != b.draft_vocab);
    ne("export_state", a.export_state != b.export_state);
    ne("moe_route_bf16", a.moe_route_bf16 != b.moe_route_bf16);
    ne("diagnostic_bf16_mtp", a.diagnostic_bf16_mtp != b.diagnostic_bf16_mtp);
    ne("forms.hc_mma", a.forms.hc_mma != b.forms.hc_mma);
    ne("forms.mtp_mma", a.forms.mtp_mma != b.forms.mtp_mma);
    ne("forms.qsa_mma", a.forms.qsa_mma != b.forms.qsa_mma);
    ne("forms.moe_mma", a.forms.moe_mma != b.forms.moe_mma);
    return d;
}
// serve vs cli: the same request. Fails on a Resolution, contract or Config difference that is not named.
void same_request(const std::string& label, const Args& a, const Env& env = {}) {
    const auto serve = one(a, qo::Entry::serve, env), service = one(a, qo::Entry::prefill_service, env),
               cli = one(a, qo::Entry::cli, env);
    check(serve.resolution.json() == cli.resolution.json() && cli.resolution.json() == service.resolution.json(), label + ": resolution differs");
    check(serve.contract == cli.contract && serve.contract == service.contract, label + ": contract differs");
    check(serve.resolution.notices() == cli.resolution.notices(), label + ": notices differ");
    check(diff(service.config, cli.config).empty(), label + ": prefill-service vs cli engine values differ");
    std::set<std::string> named;
    for (const auto& d : serve.differences) {
        named.insert(d.name);
        check(!d.reason.empty() && !d.value.empty(), label + ": difference " + d.name + " has no reason/value");
    }
    for (const auto& name : diff(serve.config, cli.config))
        check(named.count(name) == 1, label + ": unexplained serve/cli difference in " + name);
    check(named == std::set<std::string>({"decode", "export_state", "release_by_layer", "block_schedule"}),
          label + ": serve differences are the three role levers and the flat block schedule");
    check(serve.config.block_schedule == std::vector<std::uint32_t>({8192}) &&
          cli.config.block_schedule == std::vector<std::uint32_t>({1024, 2048, 4096, 8192}) && cli.config.block_schedule == service.config.block_schedule,
          label + ": serve runs flat blocks of the schedule maximum, the hand-off entries keep the ramp");
    check(cli.differences.empty() && service.differences.empty(), label + ": cli/service name no differences when --model is given");
    check(serve.config.decode && !serve.config.export_state && !serve.config.release_by_layer, label + ": serve role values");
    check(!cli.config.decode && cli.config.export_state && cli.config.release_by_layer, label + ": cli keeps the Config defaults");
}
bool refuses(const Args& a, const Env& env = {}, qo::Entry e = qo::Entry::cli) {
    try { (void)one(a, e, env); } catch (const std::invalid_argument&) { return true; }
    return false;
}
std::string status(const qo::Resolved& r, const char* lever) {
    const auto* e = r.resolution.find(lever);
    return e ? std::string(pr::status_name(e->status)) : "absent";
}

void profiles() {
    for (const auto& a : {Args{}, Args{"--profile", "default"}, Args{"--profile", "exact"}}) {
        same_request("exact", a);
        const auto r = one(a, qo::Entry::cli);
        check(!r.choice.fast() && r.contract == "qwen-cuda;prefill=exact", "default/exact is the exact CUDA path");
        check(!r.config.forms.hc_mma && !r.config.forms.mtp_mma && !r.config.forms.qsa_mma && !r.config.forms.moe_mma && r.config.moe_route_bf16 == false, "exact: all forms off");
        check(r.resolution.entries.empty(), "default lists no rows");
    }
    check(one({"--profile", "exact"}, qo::Entry::cli).resolution.json() == one({"--profile", "default"}, qo::Entry::cli).resolution.json(), "exact is an alias of default");
    same_request("fast", {"--profile", "fast"});
    const auto fast = one({"--profile", "fast"}, qo::Entry::serve);
    check(fast.config.forms.hc_mma && fast.config.forms.mtp_mma && fast.config.forms.qsa_mma && fast.config.forms.moe_mma, "fast: the four forms on");
    check(fast.contract == "qwen-cuda;prefill=fast;forms=hc_mma,mtp_mma,qsa_mma,moe_mma;moe_route_bf16=0", "fast contract: " + fast.contract);
    for (const char* f : {"hc_mma", "mtp_mma", "qsa_mma", "moe_mma"}) check(status(fast, f) == "provisional", std::string("fast: ") + f + " provisional");
    for (const char* f : {"moe_route_bf16", "qsa_budget", "gdn_fp16_chunk", "output_head"})
        check(status(fast, f) == "pending", std::string("fast: ") + f + " pending (listed, not used)");
    check(fast.config.moe_route_bf16 == false, "fast: pending route storage is not applied");
    const auto json = fast.resolution.json();
    for (const char* f : {"moe_route_bf16", "qsa_budget", "gdn_fp16_chunk", "output_head"})
        check(json.find(std::string("\"lever\":\"") + f + "\"") != std::string::npos, std::string("pending list names ") + f);
    check(fast.resolution.qualification().find("hc_mma") != std::string::npos, "qualification names the provisional forms");
    // quality = default rows on this engine.
    check(!one({"--profile", "quality"}, qo::Entry::cli).choice.fast(), "quality runs exact");
}

void overrides() {
    same_request("fast-hc-off", {"--profile", "fast", "--hc-mma", "off"});
    auto r = one({"--profile", "fast", "--hc-mma", "off"}, qo::Entry::cli);
    check(!r.config.forms.hc_mma && r.config.forms.mtp_mma && r.config.forms.qsa_mma && r.config.forms.moe_mma, "--hc-mma off wins over the profile");
    check(status(r, "hc_mma") == "off", "hc_mma status off");
    check(r.contract == "qwen-cuda;prefill=fast;forms=mtp_mma,qsa_mma,moe_mma;moe_route_bf16=0", r.contract);
    // moe_mma: option and env of the table row, CLI beats env, off wins over the fast profile.
    same_request("fast-moe-off", {"--profile", "fast", "--moe-mma", "off"});
    r = one({"--profile", "fast", "--moe-mma", "off"}, qo::Entry::cli);
    check(!r.config.forms.moe_mma && r.config.forms.hc_mma && status(r, "moe_mma") == "off", "--moe-mma off wins over the profile");
    check(r.contract == "qwen-cuda;prefill=fast;forms=hc_mma,mtp_mma,qsa_mma;moe_route_bf16=0", r.contract);
    Env moe_env{{"STRIX_QRTX_MOE_MMA", "0"}};
    same_request("fast-env-moe-off", {"--profile", "fast"}, moe_env);
    r = one({"--profile", "fast"}, qo::Entry::serve, moe_env);
    check(!r.config.forms.moe_mma && status(r, "moe_mma") == "off", "STRIX_QRTX_MOE_MMA=0 reaches only through the lookup");
    moe_env = {{"STRIX_QRTX_MOE_MMA", "1"}};
    r = one({}, qo::Entry::prefill_service, moe_env);
    check(r.config.forms.moe_mma && !r.config.forms.hc_mma && status(r, "moe_mma") == "explicit-env" && r.contract == "qwen-cuda;prefill=fast;forms=moe_mma;moe_route_bf16=0",
          "env turns the MoE form on in the default profile: " + r.contract);
    r = one({"--moe-mma", "off"}, qo::Entry::cli, moe_env);
    check(!r.config.forms.moe_mma, "CLI beats env (moe_mma)");
    r = one({"--moe-mma", "1"}, qo::Entry::cli);
    check(r.config.forms.moe_mma && !r.config.forms.hc_mma && cp::form_identities(r.choice) == "qwen-cuda-moe-i8fma-v1", "--moe-mma 1 alone: " + cp::form_identities(r.choice));
    Env env{{"STRIX_QRTX_QSA_MMA", "0"}};
    same_request("fast-env-qsa-off", {"--profile", "fast"}, env);
    r = one({"--profile", "fast"}, qo::Entry::serve, env);
    check(!r.config.forms.qsa_mma && status(r, "qsa_mma") == "off", "STRIX_QRTX_QSA_MMA=0 reaches only through the lookup");
    env = {{"STRIX_QRTX_MTP_MMA", "1"}};
    r = one({}, qo::Entry::prefill_service, env);
    check(r.config.forms.mtp_mma && status(r, "mtp_mma") == "explicit-env" && r.contract == "qwen-cuda;prefill=fast;forms=mtp_mma;moe_route_bf16=0",
          "env turns one form on in the default profile: " + r.contract);
    r = one({"--mtp-mma", "off"}, qo::Entry::cli, env);
    check(!r.config.forms.mtp_mma, "CLI beats env");
    r = one({"--hc-mma", "1", "--qsa-mma", "on"}, qo::Entry::cli);
    check(r.config.forms.hc_mma && r.config.forms.qsa_mma && !r.config.forms.mtp_mma, "default + explicit forms");
    same_request("default-explicit-forms", {"--hc-mma", "1", "--qsa-mma", "on"});
    // The route storage: explicit option, env, or --allow-pending (unqualified).
    r = one({"--prefill-moe-route-bf16", "on"}, qo::Entry::cli);
    check(r.config.moe_route_bf16 == true && r.contract == "qwen-cuda;prefill=fast;forms=none;moe_route_bf16=1", "route storage on: " + r.contract);
    r = one({"--profile", "fast", "--allow-pending"}, qo::Entry::cli);
    check(r.config.moe_route_bf16 == true && status(r, "moe_route_bf16") == "unqualified", "--allow-pending runs the route storage as unqualified");
    check(status(r, "qsa_budget") == "pending", "not-runnable rows stay pending under --allow-pending");
    env = {{"STRIX_QPF2K_MOE_ROUTE_BF16", "1"}};
    check(one({}, qo::Entry::cli, env).config.moe_route_bf16 == true, "route storage env through the lookup");
    // Refusals: never silently used.
    check(refuses({"--qsa-budget", "1536"}), "HIP-only lever refused on CUDA");
    check(refuses({"--gdn-fp16-chunk", "1"}), "gdn_fp16_chunk refused on CUDA");
    check(refuses({"--hc-mma", "maybe"}), "malformed boolean refused");
    check(refuses({"--profile", "fast", "--profile", "default"}), "duplicate --profile");
    check(refuses({"--profile", "turbo-x"}), "unknown profile");  // turbo itself resolves (turbo_binding)
    check(refuses({"--profile"}), "missing profile value");
    check(refuses({}, Env{{"STRIX_QPF2K_GDN_FP16_CHUNK", "1"}}), "HIP-only env refused on CUDA");
    check(refuses({"--release-by-layer"}, {}, qo::Entry::serve), "serve refuses an unexplained release-by-layer");
    r = one({"--release-by-layer"}, qo::Entry::prefill_service);
    check(r.config.release_by_layer, "service takes --release-by-layer");
    r = one({"--no-release-by-layer"}, qo::Entry::prefill_service);
    check(!r.config.release_by_layer, "service takes --no-release-by-layer");
}

// Flat blocks: a bitwise scheduling lever of the serve entry; a typed schedule or the switch keeps the caller's blocks.
void flat_blocks() {
    const std::vector<std::uint32_t> ramp{1024, 2048, 4096, 8192};
    const Env off{{"STRIX_QRTX_FLAT_BLOCKS", "0"}}, on{{"STRIX_QRTX_FLAT_BLOCKS", "1"}}, bad{{"STRIX_QRTX_FLAT_BLOCKS", "yes"}};
    auto r = one({}, qo::Entry::serve, off);
    check(r.config.block_schedule == ramp && r.differences.size() == 3, "flat blocks off by its switch: ramp, no named schedule difference");
    r = one({}, qo::Entry::serve, on);
    check(r.config.block_schedule == std::vector<std::uint32_t>({8192}), "flat blocks on by its switch");
    check(one({"--profile", "fast"}, qo::Entry::serve).config.block_schedule == one({}, qo::Entry::serve).config.block_schedule,
          "the schedule does not depend on the profile");
    r = one({"--blocks", "1024,2048,4096,8192"}, qo::Entry::serve);
    check(r.config.block_schedule == ramp && r.differences.size() == 3, "a typed --blocks is the caller's schedule");
    r = one({"--block-rows", "4096"}, qo::Entry::serve);
    check(r.config.block_schedule == std::vector<std::uint32_t>({4096}) && r.differences.size() == 3, "a typed --block-rows is kept");
    check(one({}, qo::Entry::cli, on).config.block_schedule == ramp && one({}, qo::Entry::prefill_service, on).config.block_schedule == ramp,
          "the hand-off entries keep the ramp");
    check(refuses({}, bad, qo::Entry::serve), "STRIX_QRTX_FLAT_BLOCKS accepts 0 or 1 only");
    bool listed = false;
    for (const auto& d : one({}, qo::Entry::serve).differences) listed = listed || (d.name == "block_schedule" && d.value == "8192" && !d.reason.empty());
    check(listed, "the flat schedule is named with its value and reason in the effective config");
}

void common_options() {
    const Args a{"--model", "m", "--mtp-sidecar", "s", "--draft-vocab", "d", "--tables", "t", "--rope-table", "r", "--max-positions", "4096",
                 "--blocks", "1024,2048", "--block-schedule", "default", "--final-block-rows", "64"};
    const Env none;
    const auto request = parse(a, none);
    const auto serve = qo::resolve(request, qo::Entry::serve), cli = qo::resolve(request, qo::Entry::cli);
    check(cli.config.max_positions == 4096 && cli.config.block_schedule == std::vector<std::uint32_t>({1024, 2048}) && !cli.config.tail_large &&
          cli.config.final_block_rows == 64 && cli.config.mtp_sidecar_gguf == "s" && cli.config.draft_vocab == "d", "common options parsed");
    check(diff(serve.config, cli.config) == std::set<std::string>({"decode", "export_state", "release_by_layer"}), "same values in both entries");
    check(parse({"--block-rows", "2048"}, none).engine.block_schedule == std::vector<std::uint32_t>({2048}), "--block-rows");
    bool threw = false;
    try { (void)parse({"--block-rows", "100"}, none); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "--block-rows 100 refused");
    threw = false;
    try { qo::resolve(parse({"--model", "m"}, none), qo::Entry::cli); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "tables and rope required");
    // The prefill service names its default model.
    const auto service = qo::resolve(parse({"--tables", "t", "--rope-table", "r"}, none), qo::Entry::prefill_service, "default.gguf");
    check(service.config.model_gguf == "default.gguf" && service.differences.size() == 1 && service.differences[0].name == "model_gguf", "default model is a named difference");
    check(refuses({}, {}, qo::Entry::serve) == false, "serve with paths resolves");
}

void identities() {
    std::set<std::string> contracts;
    std::set<std::uint64_t> ids;
    const char* base = "strix-cuda-qwen-prefill:a8-single-profile:hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1";
    for (unsigned mask = 0; mask < 16; ++mask) {
        cp::Choice c{(mask & 1U) != 0, (mask & 2U) != 0, (mask & 4U) != 0, (mask & 8U) != 0};
        contracts.insert(cp::arithmetic_contract(c));
        ids.insert(qo::handoff_numerical_id(base, c));
    }
    check(contracts.size() == 16 || contracts.size() == 16U, "16 distinct arithmetic contracts: " + std::to_string(contracts.size()));
    check(ids.size() == 16, "16 distinct hand-off producer ids");
    std::uint64_t legacy = 14695981039346656037ULL;
    for (const char ch : std::string(base)) { legacy ^= static_cast<unsigned char>(ch); legacy *= 1099511628211ULL; }
    check(qo::handoff_numerical_id(base, {}) == legacy, "exact hand-off id keeps its historical value");
    check(cp::form_identities({false, true, true, true}) == "qwen-cuda-hc-f16mma-v1,qwen-cuda-mtp-f16mma-v1,qwen-cuda-qsa-f16mma-v1", "form identities");
    // Prefix-cache identity: forms enter only when on.
    const auto dir = fs::temp_directory_path() / "strix_qwen_cuda_profile_host_test";
    fs::create_directories(dir / "tables");
    { std::ofstream(dir / "tables" / "a.bin") << "x"; std::ofstream(dir / "rope.bin") << "y"; std::ofstream(dir / "m.gguf") << "z"; }
    qp::Config c;
    c.model_gguf = (dir / "m.gguf").string();
    c.tables_dir = (dir / "tables").string();
    c.rope_table = (dir / "rope.bin").string();
    const auto fields = [&](const qp::Config& cfg) { return px::identity_fields(cfg, 1024); };
    const auto off = fields(c);
    for (const auto& [k, v] : off) check(k.rfind("form.", 0) != 0, "OFF keys carry no form field: " + k);
    std::set<std::string> keys{px::canonical_identity(off)};
    for (int which = 0; which < 4; ++which) {
        auto f = c;
        (which == 0 ? f.forms.hc_mma : which == 1 ? f.forms.mtp_mma : which == 2 ? f.forms.qsa_mma : f.forms.moe_mma) = true;
        const auto on = fields(f);
        check(on.size() == off.size() + 1, "one form adds exactly one identity field");
        check(keys.insert(px::canonical_identity(on)).second, "each form changes the prefix identity");
        bool found = false;
        for (const auto& [k, v] : on) found |= k.rfind("form.", 0) == 0 && v == cp::form_identities(cp::Choice{false, which == 0, which == 1, which == 2, which == 3});
        check(found, "prefix identity field carries the form identity string");
    }
    fs::remove_all(dir);
    // Receipt.
    const auto receipt = cp::receipt({false, true, false, true}).json();
    check(receipt.find("qwen-cuda;prefill=fast;forms=hc_mma,qsa_mma;moe_route_bf16=0") != std::string::npos, "receipt carries the contract: " + receipt);
}

void effective_config() {
    const auto serve = one({"--profile", "fast"}, qo::Entry::serve);
    const auto text = qo::effective_config_json(serve);
    const auto j = strix::serve::parse_json(text);
    const auto at = [](const strix::serve::Json& o, const char* key) -> const strix::serve::Json& { return *o.find(key); };
    check(at(j, "entry").as_string() == "serve" && at(j, "profile").as_string() == "fast", "effective config: entry/profile");
    check(at(j, "arithmetic_contract").as_string() == serve.contract, "effective config: contract");
    check(at(at(j, "forms"), "hc_mma").as_bool() && !at(at(j, "forms"), "moe_route_bf16").as_bool(), "effective config: forms");
    check(at(j, "serve_differences").as_array().size() == 4, "effective config names the three role differences and the flat block schedule");
    check(text.find("\"reason\"") != std::string::npos && text.find("pending") != std::string::npos, "effective config carries reasons and pending rows");
    const auto cli = qo::effective_config_json(one({}, qo::Entry::cli));
    check(at(strix::serve::parse_json(cli), "serve_differences").as_array().empty(), "cli names no serve differences");
}

void route_governors() {
    const auto off = one({}, qo::Entry::serve);
    for (unsigned mask = 0; mask < 8; ++mask) {
        const auto r = one({"--prefill-moe-route-bf16", "on"}, qo::Entry::cli,
            {{"STRIX_BASELINE", (mask & 1U) ? "1" : "0"},
             {"STRIX_QPF2K_DEFAULTS", (mask & 2U) ? "0" : "1"},
             {"STRIX_PREFILL_EXACT_FP16", (mask & 4U) ? "1" : "0"}});
        check(r.choice.moe_route_bf16 == (mask == 0), "all governor combinations");
    }
    for (const auto& [key, value, reason] : {
        std::array<const char*, 3>{"STRIX_BASELINE", "1", "baseline"},
        std::array<const char*, 3>{"STRIX_QPF2K_DEFAULTS", "0", "qpf2k-defaults-off"},
        std::array<const char*, 3>{"STRIX_PREFILL_EXACT_FP16", "1", "exact-master"}}) {
        for (const auto entry : {qo::Entry::serve, qo::Entry::prefill_service, qo::Entry::cli}) {
            for (const bool cli : {false, true}) {
                const Env env{{key, value}, {"STRIX_QPF2K_MOE_ROUTE_BF16", "1"}};
                const auto r = one(cli ? Args{"--prefill-moe-route-bf16", "on"} : Args{}, entry, env);
                check(r.config.route_policy->requested && !r.config.route_policy->enabled, "governor suppresses requested storage");
                check(r.config.route_policy->reason == reason, "governor reason");
                check(!r.choice.moe_route_bf16 && r.contract == off.contract, "effective contract is exact");
                check(!cp::choose(r.resolution).moe_route_bf16, "lever resolution is effective");
                check(qo::handoff_numerical_id("old", r.choice) == qo::handoff_numerical_id("old", off.choice), "exact handoff restored");
                const auto frozen = qo::resolve_route(r.config, [](const char*) -> const char* {
                    throw std::runtime_error("resolved engine must not re-read environment");
                });
                check(!frozen.moe_route_bf16.value(), "execution retains resolved OFF");
                const auto json = strix::serve::parse_json(qo::effective_config_json(r));
                const auto* policy = json.find("route_storage");
                check(policy->find("requested")->as_bool() && !policy->find("effective")->as_bool(), "requested and effective reported separately");
            }
        }
    }
    const auto on = one({"--prefill-moe-route-bf16", "on"}, qo::Entry::cli);
    const auto frozen = qo::resolve_route(on.config, [](const char*) { return "1"; });
    check(frozen.moe_route_bf16.value(), "execution retains resolved ON despite changed process environment");
    check(!one({"--prefill-moe-route-bf16", "off"}, qo::Entry::cli,
               {{"STRIX_QPF2K_MOE_ROUTE_BF16", "1"}}).config.route_policy->requested, "CLI OFF wins");
}

void snapshot_forms() {
    std::set<std::uint64_t> hashes;
    for (unsigned mask = 0; mask < 32; ++mask) {
        qp::Config c;
        c.moe_route_bf16 = (mask & 16U) != 0;
        c.forms = {(mask & 1U) != 0, (mask & 2U) != 0, (mask & 4U) != 0, (mask & 8U) != 0};
        const auto id = qo::numerical_identity(c);
        check(hashes.insert(id.hash).second, "every form set has a distinct snapshot identity");
        if ((mask & 15U) == 0) {
            const auto legacy = strix::cuda::qwen_route::identity(mask != 0, mask != 0);
            check(id.hash == legacy.hash && id.prefill == legacy.prefill, "no forms preserves historical identity");
        }
    }
}
}  // namespace

// TURBO (qwen-turbo-v1) through the one resolver: serve == CLI == strix/cuda/qwen_turbo.hpp, weight toggle only (the
// exact CUDA prefill forms), lever options refused, and no turbo field outside turbo.
void turbo_binding() {
    const Env none;
    const auto request = parse(with_paths({"--profile", "turbo", "--turbo-weights", "onfly", "--blocks", "1024,2048"}), none);
    const auto serve = qo::resolve(request, qo::Entry::serve), cli = qo::resolve(request, qo::Entry::cli);
    const auto direct = strix::cuda::qwen_turbo::resolve({request.profile, request.inputs, false});
    const auto plain = one({}, qo::Entry::serve);
    check(serve.resolution.is_turbo() && serve.config.turbo_onfly_nvfp4 && cli.config.turbo_onfly_nvfp4 && direct.onfly_nvfp4,
          "turbo: onfly effective in both entries");
    check(!serve.config.turbo_namespace.empty() && serve.config.turbo_namespace == cli.config.turbo_namespace &&
              serve.config.turbo_namespace == direct.cache_namespace, "turbo: namespace of the qwen_turbo resolver");
    check(!serve.choice.fast() && !cli.choice.fast() && serve.contract == plain.contract, "turbo: exact CUDA prefill forms");
    check(diff(serve.config, cli.config) == std::set<std::string>({"decode", "export_state", "release_by_layer"}),
          "turbo: same values in both entries");
    check(!plain.config.turbo_onfly_nvfp4 && plain.config.turbo_namespace.empty() &&
              qo::effective_config_json(plain).find("turbo_weights") == std::string::npos &&
              qo::effective_config_json(serve).find("\"turbo_weights\"") != std::string::npos,
          "turbo: effective config field only under turbo");
    bool threw = false;
    try {
        (void)qo::resolve(parse(with_paths({"--profile", "turbo", "--prefill-moe-route-bf16", "1"}), none), qo::Entry::serve);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "turbo: CUDA lever option refused");
}

// Block fit: shared rows_fit planner (fake allocator) and the resolver binding (every entry, typed schedule strict).
void block_fit() {
    namespace rf = strix::rows_fit;
    check(rf::next_lower(8192, 1024, 1024) == 7168 && rf::next_lower(8200, 1024, 1024) == 8192 &&
              rf::next_lower(1500, 1024, 1024) == 1024 && rf::next_lower(1024, 1024, 1024) == 0,
          "next_lower steps to the multiple below, floors at min_rows, stops at min_rows");
    // per-row bytes 700, budget 5,000,000: 7168 rows fit (5,017,600 does not; 6144 -> 4,300,800 does)
    std::vector<std::uint32_t> tried;
    const auto fits = [&](std::uint32_t rows) { tried.push_back(rows); return std::uint64_t{rows} * 700U <= 5'000'000U; };
    auto r = rf::fit(8192, 1024, 1024, fits);
    check(r.rows == 6144 && r.reduced() && r.tries == 3 && tried == std::vector<std::uint32_t>({8192, 7168, 6144}),
          "fit steps 8192 -> 7168 -> 6144 and keeps the first that fits");
    r = rf::fit(4096, 1024, 1024, fits);
    check(r.rows == 4096 && !r.reduced() && r.tries == 1, "a capacity that fits is kept on the first try");
    bool threw = false;
    try { (void)rf::fit(8192, 1024, 1024, [](std::uint32_t) { return false; }); } catch (const std::runtime_error&) { threw = true; }
    check(threw, "nothing fits: the load is refused");
    check(rf::clamp_schedule({1024, 2048, 4096, 8192}, 6144) == std::vector<std::uint32_t>({1024, 2048, 4096, 6144}) &&
              rf::clamp_schedule({8192}, 4096) == std::vector<std::uint32_t>({4096}),
          "the schedule is clamped to the fitted capacity, small ramp blocks kept");
    const Env off{{"STRIX_QRTX_BLOCK_FIT", "0"}}, bad{{"STRIX_QRTX_BLOCK_FIT", "yes"}};
    for (const auto e : {qo::Entry::serve, qo::Entry::cli, qo::Entry::prefill_service}) {
        check(one({}, e).config.block_fit, "block fit on by default on every entry");
        check(!one({}, e, off).config.block_fit, "STRIX_QRTX_BLOCK_FIT=0 turns it off");
        check(!one({"--block-rows", "8192"}, e).config.block_fit && !one({"--blocks", "1024,2048,4096,8192"}, e).config.block_fit,
              "a typed schedule is the caller's: no fit");
    }
    check(refuses({}, bad, qo::Entry::serve), "STRIX_QRTX_BLOCK_FIT accepts 0 or 1 only");
}

int main() {
    try {
        const Args expert={"--expert-arithmetic","gsq-iq3s-a8-dot4-cuda-v1"};
        same_request("explicit GSQ arithmetic",expert);
        const auto native=one({},qo::Entry::cli);
        const auto gsq=one(expert,qo::Entry::cli);
        check(native.config.expert_arithmetic==strix::qwen_packs::ExpertArithmetic::incumbent,"omitted expert arithmetic preserves incumbent");
        check(gsq.config.expert_arithmetic==strix::qwen_packs::ExpertArithmetic::gsq_a8_dot4,"explicit GSQ parsed once");
        check(gsq.contract.find("gsq-iq3s-a8-dot4-cuda-v1")!=std::string::npos,"GSQ contract named");
        check(qo::effective_config_json(gsq).find("gsq-iq3s-a8-dot4-cuda-v1")!=std::string::npos,"GSQ effective config names arithmetic");
        check(qo::numerical_identity(gsq.config).hash!=qo::numerical_identity(native.config).hash,"GSQ snapshots have independent numerical identity");
        check(qo::handoff_numerical_id_config("old",gsq.config)!=qo::handoff_numerical_id_config("old",native.config),"GSQ handoffs cannot masquerade as native");
        check(qo::handoff_numerical_id_config("old",native.config)==qo::handoff_numerical_id("old",qo::choice_of(native.config)),"incumbent handoff bytes preserved");
        check(refuses({"--expert-arithmetic","unknown"}),"unknown expert arithmetic rejected");
        check(refuses({"--expert-arithmetic","gsq-iq3s-a8-dot4-cuda-v1","--moe-mma","on"}),"GSQ single-chain composition rejects explicitly");
        profiles();
        overrides();
        common_options();
        turbo_binding();
        flat_blocks();
        block_fit();
        identities();
        effective_config();
        route_governors();
        snapshot_forms();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 2;
    }
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::puts("qwen_cuda_profile_host_test: ok");
    return 0;
}
