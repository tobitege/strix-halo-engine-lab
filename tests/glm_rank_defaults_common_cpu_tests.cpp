// Host-only defaults and product-profile checks. The bare --profile retains kernel timing.
#include "glm_rank_receipt.hpp"
#include "glm/engine/glm_kvshard_exchange.hpp"
#include "glm/engine/glm_mem.hpp"

#include <cstdio>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

namespace rd = strix::glm::rank_defaults;
namespace profile = strix::profile;

namespace {
int checks = 0, failures = 0;
#define CHECK(c) do { ++checks; if (!(c)) { ++failures; \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

using Env = std::map<std::string, std::string>;
profile::Lookup lookup(const Env& env) {
    return [&env](const char* key) -> const char* {
        const auto it = env.find(key);
        return it == env.end() ? nullptr : it->second.c_str();
    };
}

void test_common_default_rows() {
    static_assert(std::is_same_v<rd::EnvItem, profile::DefaultItem>);
    const Env absent;
    const auto empty = lookup(absent);
    CHECK(rd::env_items().size() == 42);
    const auto normal = rd::resolve(rd::scan({}), empty);
    const auto baseline_config = rd::resolve(rd::scan({"--baseline"}), empty);
    CHECK(rd::print(normal).find("effective pf-seqpar=on") != std::string::npos);
    CHECK(rd::print(baseline_config).find("effective pf-seqpar=off") != std::string::npos);
    CHECK(rd::print(baseline_config).find("capacity-layout=replicated/full/qwin=0") != std::string::npos);
    for (const auto& args : {std::vector<std::string>{"--prefill", "wmma"},
                             std::vector<std::string>{"--max-rows", "1"}, std::vector<std::string>{"--world", "1"}})
        CHECK(rd::print(rd::resolve(rd::scan(args), empty)).find("effective pf-seqpar=declined") != std::string::npos);
    const Env routed{{"STRIX_GLM_PF_SCRATCH", "routed"}, {"STRIX_GLM_PF_ATTN_QWIN", "2048"}};
    const auto config = rd::print(rd::resolve(rd::scan({}), lookup(routed)));
    CHECK(config.find("capacity-layout=replicated/routed/qwin=2048") != std::string::npos);
    CHECK(config.find("capacity_layout_reason=first-admitted-query-window") != std::string::npos);
    CHECK(config.find("admission and rank vote pending") != std::string::npos);
    for (const auto& row : rd::env_items()) {
        using Source = profile::DefaultSource;
        auto value = profile::resolve_default(row, false, empty);
        CHECK(value.value == row.on && value.source == Source::on);
        value = profile::resolve_default(row, true, empty);
        CHECK(value.value == (row.in_baseline ? row.off : row.on));
        CHECK(value.source == (row.in_baseline ? Source::baseline : Source::on));
        for (bool baseline : {false, true}) {
            for (const std::string& setting : {std::string(row.off), std::string(row.on), std::string{}}) {
                const Env env{{row.env, setting}};
                value = profile::resolve_default(row, baseline, lookup(env));
                CHECK(value.value == setting && value.source == Source::env);
                value = profile::resolve_default(row, baseline, lookup(env), "cli-value");
                CHECK(value.value == "cli-value" && value.source == Source::cli);
                value = profile::resolve_default(row, baseline, lookup(env), "");
                CHECK(value.value.empty() && value.source == Source::cli);
            }
        }
    }
}

void test_glm_exports_and_precedence() {
    const auto request = rd::scan({"--decode", "64"});
    const Env absent;
    const auto plain = rd::resolve(request, lookup(absent));
    for (const auto& row : rd::env_items()) for (bool baseline : {false, true}) {
        auto r = request;
        r.baseline_flag = baseline;
        for (const std::string& setting : {std::string(row.off), std::string{}}) {
            const Env env{{row.env, setting}};
            if (setting.empty() && (std::string(row.env) == "STRIX_GLM_PF_SEQPAR" ||
                                    std::string(row.env) == "STRIX_GLM_PF_SCRATCH" ||
                                    std::string(row.env) == "STRIX_GLM_PF_ATTN_QWIN")) {
                bool refused = false;
                try { (void)rd::resolve(r, lookup(env)); }
                catch (const std::invalid_argument&) { refused = true; }
                CHECK(refused);
                continue;
            }
            const auto e = rd::resolve(r, lookup(env));
            CHECK(e.env.at(row.env).value == setting && e.env.at(row.env).source == "explicit");
            Env exported;
            rd::apply_env(e, [&](const char* key, const char* value) { exported[key] = value; });
            CHECK(exported.count(row.env) == 0);
            for (const auto& other : rd::env_items()) {
                if (std::string(row.env) == other.env) continue;
                const bool off = baseline && other.in_baseline;
                CHECK(e.env.at(other.env).value == (off ? other.off : other.on));
                CHECK(e.env.at(other.env).source == (off ? "baseline" : "default"));
                CHECK(exported.count(other.env) == (off || other.app_sets ? 1U : 0U));
                if (exported.count(other.env)) CHECK(exported.at(other.env) == (off ? other.off : other.on));
            }
        }
    }
    const auto baseline = rd::resolve(rd::scan({"--baseline"}), lookup(absent));
    CHECK(baseline.env.at("STRIX_GLM_MOE").value == "fast");
    CHECK(baseline.env.at("STRIX_GLM_MOE").source == "default");
    const Env overrides{{"STRIX_GLM_IDENTITY", "r1"}, {"STRIX_GLM_DECODE_XCH", "comm"},
                        {"STRIX_GLM_PREFILL", "ref"}, {"STRIX_GLM_MTP_SPEC", "0"}};
    rd::ModelFacts facts;
    facts.mtp_complete = true;
    const auto e = rd::resolve(rd::scan({"--decode", "64", "--identity", "r1b", "--decode-xch", "xfer",
                                       "--prefill", "gemv", "--mtp-spec", "1"}), lookup(overrides), facts);
    CHECK(e.identity.value == "r1");              // legacy env > CLI
    CHECK(e.decode_xch.value == "comm");         // legacy env > CLI
    CHECK(e.prefill_requested.value == "gemv"); // CLI > env
    CHECK(e.mtp_spec.value == "1");             // CLI > env
    CHECK(rd::is_value_free("--profile"));
    const auto timing = rd::scan({"--profile", "--decode", "64"});
    CHECK(timing.decode && !timing.baseline_flag);
    CHECK(rd::print(rd::resolve(timing, lookup(absent))) == rd::print(plain));
}

void test_glm_product_profiles() {
    const Env absent;
    const auto empty = lookup(absent);
    const auto exports = [&](const rd::Effective& e) {
        Env result;
        rd::apply_env(e, [&](const char* key, const char* value) { result[key] = value; });
        return result;
    };
    const auto plain = rd::resolve(rd::scan({"--decode", "64"}), empty);
    CHECK(!plain.product);
    const auto normal_exports = exports(plain);
    for (const auto* name : {"default", "quality"}) {
        const auto e = rd::resolve(rd::scan({"--profile", name, "--decode", "64"}), empty);
        CHECK(e.product && e.product->find("pf_wmma")->value == "0");
        CHECK(exports(e) == normal_exports);
        // Exact defaults and inspection lines are unchanged; only the requested profile receipt is appended.
        CHECK(rd::print(e) == rd::print(plain) + e.product->describe());
        CHECK(e.product->quality_fallback() == (std::string(name) == "quality"));
    }
    auto fast = rd::resolve(rd::scan({"--profile", "fast", "--decode", "64"}), empty);
    CHECK(fast.product && fast.product->find("pf_wmma")->status == profile::Status::set);
    CHECK(fast.product->find("pf_wmma")->value == "1");
    auto wanted = normal_exports;
    wanted["STRIX_GLM_PF_WMMA"] = "1";
    CHECK(exports(fast) == wanted); // FAST adds exactly one switch.
    CHECK(fast.product->qualification() == "qualified");
    CHECK(rd::print(fast).find("profile lever pf_wmma 1 (profile, bit-changing, STRIX_GLM_PF_WMMA)") != std::string::npos);
    CHECK(rd::print(fast) == rd::print(plain) + fast.product->describe());
    for (const char* value : {"0", "1", "", "1tail"}) {
        const Env env{{"STRIX_GLM_PF_WMMA", value}};
        const auto e = rd::resolve(rd::scan({"--profile=fast", "--decode", "64"}), lookup(env));
        CHECK(e.product->find("pf_wmma")->value == value);
        CHECK(e.product->find("pf_wmma")->status == (std::string(value) == "0" ? profile::Status::off : profile::Status::explicit_env));
        CHECK(exports(e).count("STRIX_GLM_PF_WMMA") == 0); // explicit env is never rewritten
    }
    for (const auto& args : {std::vector<std::string>{"--profile", "fast", "--baseline"},
                             std::vector<std::string>{"--baseline", "--profile=fast"}}) {
        const auto e = rd::resolve(rd::scan(args), empty);
        CHECK(e.product->find("pf_wmma")->value == "0");
        CHECK(exports(e) == exports(rd::resolve(rd::scan({"--baseline"}), empty)));
        const Env explicit_on{{"STRIX_GLM_PF_WMMA", "1"}};
        CHECK(rd::resolve(rd::scan(args), lookup(explicit_on)).product->find("pf_wmma")->value == "1");
    }
    const Env baseline_env{{"STRIX_GLM_BASELINE", "1"}};
    CHECK(rd::resolve(rd::scan({"--profile", "fast"}), lookup(baseline_env)).product->find("pf_wmma")->value == "0");
    std::vector<std::string> args{"--profile", "--profile=fast", "--decode", "64"};
    profile::Inputs input;
    profile::take_options(rd::product_profile_engine, args, input);
    CHECK(input.profile == profile::Name::fast);
    CHECK((args == std::vector<std::string>{"--profile", "--decode", "64"}));
    CHECK(rd::scan({"--profile", "--profile", "fast", "--decode", "64"}).decode);
    for (const auto& invalid : {std::vector<std::string>{"--profile", "fastt"},
                                std::vector<std::string>{"--profile", "fast", "--profile", "default"}}) {
        bool refused = false;
        try { (void)rd::scan(invalid); } catch (const std::invalid_argument&) { refused = true; }
        CHECK(refused);
    }
}

void test_common_profile_admission() {
    // Synthetic levers, never applied to the GLM process environment or rank CLI.
    const auto levers = [] {
        std::vector<profile::Lever> rows;
        for (const char* key : {"qualified", "pending", "unavailable", "provisional", "provisional-unavailable"}) {
            profile::Lever row;
            row.name = key;
            row.env = key;
            row.option = "--lever";
            row.builtin = "0";
            row.off = "0";
            row.value = {"0", "1", ""};
            row.kind = profile::Kind::bit_changing;
            row.target = "candidate";
            rows.push_back(row);
        }
        rows[0].ready[1] = true;
        rows[1].runnable[1] = true;
        rows[3].runnable[1] = true;
        rows[3].provisional[1] = true;
        rows[4].provisional[1] = true;
        return rows;
    }();
    const profile::Engine engine{"glm-host-test", "ALLOW_PENDING", levers};
    const Env absent;
    profile::Inputs input;
    input.env = lookup(absent);
    const auto standard = profile::resolve(engine, input);
    CHECK(standard.profile == profile::Name::standard);
    CHECK(standard.qualification() == "qualified");
    CHECK(standard.notices().empty());
    unsigned exports = 0;
    profile::apply(standard, [&](const char*, const char*) { ++exports; return true; });
    CHECK(exports == 0);
    for (const auto& entry : standard.entries) {
        CHECK(entry.value == "0" && entry.status == profile::Status::builtin);
    }
    input.profile = profile::Name::fast;
    auto fast = profile::resolve(engine, input);
    CHECK(fast.find("qualified")->status == profile::Status::set);
    CHECK(fast.find("pending")->status == profile::Status::pending);
    CHECK(fast.find("pending")->value == "0" && fast.find("pending")->target == "candidate");
    CHECK(fast.find("unavailable")->status == profile::Status::pending);
    CHECK(fast.find("provisional")->status == profile::Status::provisional);
    CHECK(fast.find("provisional-unavailable")->status == profile::Status::pending);
    CHECK(fast.qualification() == "qualification in progress: provisional");
    Env applied;
    profile::apply(fast, [&](const char* key, const char* value) { applied[key] = value; return true; });
    CHECK((applied == Env{{"qualified", "1"}, {"provisional", "1"}}));
    for (bool env_allow : {false, true}) {
        const Env env = env_allow ? Env{{"ALLOW_PENDING", "1"}} : Env{};
        input.env = lookup(env);
        input.allow_pending = !env_allow;
        const auto allowed = profile::resolve(engine, input);
        CHECK(allowed.allow_pending);
        CHECK(allowed.find("pending")->value == "1");
        CHECK(allowed.find("pending")->status == profile::Status::unqualified);
        CHECK(allowed.find("unavailable")->status == profile::Status::pending);
        CHECK(allowed.find("provisional")->status == profile::Status::provisional);
        CHECK(allowed.find("provisional-unavailable")->status == profile::Status::pending);
        CHECK(allowed.qualification() == "qualification in progress: provisional; unqualified: pending");
    }
    const Env explicit_env{{"qualified", "env-value"}, {"pending", ""}, {"unavailable", "0"}};
    input.env = lookup(explicit_env);
    input.allow_pending = false;
    fast = profile::resolve(engine, input);
    CHECK(fast.find("qualified")->value == "env-value");
    CHECK(fast.find("qualified")->status == profile::Status::explicit_env);
    CHECK(fast.find("pending")->value.empty() && fast.find("pending")->status == profile::Status::explicit_env);
    CHECK(fast.find("unavailable")->status == profile::Status::off);
    input.cli = {{"qualified", "cli-value"}};
    fast = profile::resolve(engine, input);
    CHECK(fast.find("qualified")->value == "cli-value");
    CHECK(fast.find("qualified")->status == profile::Status::explicit_cli);
    input.cli = {{"qualified", "0"}};
    CHECK(profile::resolve(engine, input).find("qualified")->status == profile::Status::off);
    input.cli.clear();
    input.env = lookup(absent);
    input.profile = profile::Name::quality;
    const auto quality = profile::resolve(engine, input);
    CHECK(quality.quality_fallback() && quality.notices().size() == 1);
    for (const auto& entry : quality.entries) CHECK(entry.value == "0");
    bool refused = false;
    try { (void)profile::parse("unsupported"); } catch (const std::invalid_argument&) { refused = true; }
    CHECK(refused);
}

void test_partial_receipt() {
    const Env absent;
    const auto e = rd::resolve(rd::scan({"--decode", "64"}), lookup(absent));
    const auto receipt = rd::host_receipt(e);
    CHECK(receipt.engine == "glm" && receipt.implementation == "strix-glm5-rank");
    CHECK(receipt.identity.arithmetic_contract.find("glm_math.hpp") != std::string::npos);
    CHECK(receipt.identity.arithmetic_contract.find("ascending fmaf chain") != std::string::npos);
    CHECK(receipt.identity.reduction_plan == "r1b" && receipt.identity.prefill_lineage == "gemv");
    CHECK(receipt.identity.model_sha256.empty() && receipt.identity.pack_hash.empty());
    CHECK(receipt.identity.table_hash.empty() && receipt.identity.state_format.empty());
    CHECK(receipt.build.empty() && receipt.kernel.empty() && receipt.device.empty() && receipt.topology.empty());
    CHECK(receipt.digests.empty());
    CHECK(receipt.witness() == "strix-engine-witness " + receipt.json() + "\n");
    CHECK(receipt.json().find("\"model_sha256\":\"\"") != std::string::npos);
    CHECK(receipt.json().find("\"device\":\"\"") != std::string::npos);
    const auto r1 = rd::host_receipt(rd::resolve(rd::scan({"--identity", "r1"}), lookup(absent)));
    const auto all = rd::host_receipt(rd::resolve(rd::scan({"--identity", "r1b-all"}), lookup(absent)));
    CHECK(r1.identity.arithmetic_contract == receipt.identity.arithmetic_contract);
    CHECK(r1.identity.reduction_plan == "r1");
    CHECK(all.identity.reduction_plan == "r1b-all" && all.identity.prefill_lineage == "ref");
    CHECK(r1.identity.id() != receipt.identity.id() && all.identity.id() != receipt.identity.id());
    const Env explicit_identity{{"STRIX_GLM_IDENTITY", "r1"}};
    const auto explicit_r1 = rd::host_receipt(rd::resolve(rd::scan({"--identity", "r1b"}), lookup(explicit_identity)));
    CHECK(explicit_r1.identity.id() == r1.identity.id());
    const auto ref = rd::host_receipt(rd::resolve(rd::scan({"--prefill", "ref"}), lookup(absent)));
    CHECK(ref.identity.id() != receipt.identity.id());
    CHECK(rd::print(e).find("strix-engine-witness") == std::string::npos);
}
}  // namespace

void test_kv_shard_draft() {
    rd::Request request{};
    request.decode = true;
    request.mtp = false;
    request.mtp_spec = false;
    request.prefill = "gemv";
    const Env absent, off{{"STRIX_GLM_KV_SHARD", "0"}}, on{{"STRIX_GLM_KV_SHARD", "1"}};
    const auto original = rd::resolve(request, lookup(absent));
    const auto disabled = rd::resolve(request, lookup(off));
    const auto sharded = rd::resolve(request, lookup(on));
    CHECK(rd::print(original) == rd::print(disabled));
    CHECK(rd::host_receipt(original).json() == rd::host_receipt(disabled).json());
    CHECK(!original.kv_shard && sharded.kv_shard);
    CHECK(sharded.decode_xch.value == "xfer" && sharded.decode_xch.source == "kv-shard-draft");
    CHECK(rd::print(sharded).find(strix::glm::kvshard::numerical_identity) != std::string::npos);
    const auto shard_receipt = rd::host_receipt(sharded);
    CHECK(shard_receipt.identity.reduction_plan ==
          sharded.identity.value + ";" + strix::glm::kvshard::numerical_identity);
    CHECK(shard_receipt.identity.state_format.find("layout=474c4d4b56533033") != std::string::npos);
    CHECK(shard_receipt.identity.prefill_lineage.find("layout 474c4d4b56533033") != std::string::npos);
    auto legacy_identity = shard_receipt.identity;
    legacy_identity.reduction_plan = sharded.identity.value + ";glm-dsa-kvshard-rank01-fp32-v2-declaredexp-FAST";
    CHECK(legacy_identity.id() != shard_receipt.identity.id());
    for (const std::string route : {"comm", "xfer"}) {
        auto explicit_request = request;
        explicit_request.decode_xch = route;
        const auto explicit_route = rd::resolve(explicit_request, lookup(on));
        CHECK(explicit_route.decode_xch.value == route && explicit_route.decode_xch.source == "explicit");
    }
    const Env explicit_xfer{{"STRIX_GLM_KV_SHARD", "1"}, {"STRIX_GLM_DECODE_XCH", "xfer"}};
    CHECK(rd::resolve(request, lookup(explicit_xfer)).decode_xch.value == "xfer");
    CHECK(rd::host_receipt(sharded).identity.id() != rd::host_receipt(original).identity.id());
    CHECK(!rd::host_receipt(sharded).identity.state_format.empty());
    request.mtp = true;
    bool rejected = false;
    try { (void)rd::resolve(request, lookup(on)); } catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
}

void test_kv_shard_speed_contract() {
    namespace k = strix::glm::kvshard;
    namespace mem = strix::glm::engine::mem;
    static_assert(k::prepare_bytes(0) == 65592);
    static_assert(k::prepare_bytes(8192 / 4) == 73784);
    CHECK(k::decode_capacity(65536) == 131128); // larger than eight TP rows
    static_assert(k::partial_bytes == 65848);
    CHECK(k::decode_capacity(0) >= k::partial_bytes);
    const auto header = k::decode_header(9, 4, 65535, false, 16384);
    CHECK(header.epoch == 9 && header.position == 65535 && header.pools == 16384 && header.phase == 2);

    // A synthetic memory-plan boundary, not a device capacity claim: the
    // replicated cache crosses RAM headroom; its owned-pool plan still fits.
    mem::Snapshot s;
    s.meminfo_ok = s.cma_free_ok = s.zoneinfo_ok = true;
    s.mem_available = 10 * mem::kGiB;
    const mem::Margins margins{mem::kGiB, mem::kGiB};
    auto fits = [&](std::uint64_t context, bool shard) {
        const auto tokens = shard ? k::storage_tokens(context) : context;
        return mem::fits({{"resident weights and work", 8 * mem::kGiB, 0},
                          {"DSA history", tokens * 4096, 0}}, s, margins);
    };
    CHECK(!k::auto_shard(fits(8192, false), fits(8192, false)));
    CHECK(!k::auto_shard(fits(65536, false), fits(65536, false)));
    CHECK(fits(262144, false)); // exactly at the budget is admitted
    CHECK(k::auto_shard(fits(262152, false), fits(262152, false)));
    CHECK(fits(262152, true));
    CHECK(k::auto_shard(true, false) && k::auto_shard(false, true));
    s.meminfo_ok = false;
    CHECK(!fits(8192, false)); // unavailable telemetry cannot establish a fit

    rd::Request r;
    r.decode = true; r.mtp = false; r.mtp_spec = false; r.prefill = "gemv";
    const Env automatic{{"STRIX_GLM_KV_SHARD", "auto"}};
    const auto e = rd::resolve(r, lookup(automatic));
    CHECK(e.kv_shard_auto && !e.kv_shard);
    CHECK(rd::print(e).find("pending calibrated load + unified prefill admission") != std::string::npos);
    const Env invalid{{"STRIX_GLM_KV_SHARD", "typo"}};
    bool rejected = false;
    try { (void)rd::resolve(r, lookup(invalid)); } catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
}

int main() {
    test_kv_shard_draft();
    test_kv_shard_speed_contract();
    test_common_default_rows();
    test_glm_exports_and_precedence();
    test_common_profile_admission();
    test_glm_product_profiles();
    test_partial_receipt();
    std::printf("glm defaults common cpu tests: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
