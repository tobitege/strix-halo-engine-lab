// TURBO bundles (turbo column of each family's profile table): what every family's turbo resolution selects, what it
// holds pending, and that default / fast stay untouched. The Qwen rows go through both entry points (CLI option reader
// and the serve-style Inputs) and must resolve to one fingerprint.
#include "strix/deepseek41_pair_profile_table.hpp"
#include "strix/engine_profile.hpp"
#include "strix/glm_pair_profile_table.hpp"
#include "strix/profiles/qwen.hpp"
#include "minimax/engine/mm_profile_table.hpp"
#include "minimax/engine/mm_defaults.hpp"
#include "glm/prefill/glm_pf_attn_wmma.hpp"

#include <cstdio>
#include <map>
#include <string>

namespace {
namespace p = strix::profile;
int failures = 0;
void require(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
p::Lookup env_of(const std::map<std::string, std::string>& vars) {
    return [&vars](const char* key) -> const char* {
        const auto it = vars.find(key);
        return it == vars.end() ? nullptr : it->second.c_str();
    };
}
template <class R>
const p::Entry* find(const R& r, const char* name) { return r.find(name); }

void generic(const p::Engine& engine, p::Name profile, const std::map<std::string, std::string>& vars, p::Resolution& out) {
    p::Inputs in;
    in.env = env_of(vars);
    in.profile = profile;
    out = p::resolve(engine, in);
}

void qwen_rows() {
    namespace q = strix::qwen::profile;
    const std::map<std::string, std::string> vars;
    q::Inputs in;
    in.env = env_of(vars);
    const auto turbo = q::resolve(p::Name::turbo, in);
    const auto def = q::resolve(p::Name::standard, in);
    const auto fast = q::resolve(p::Name::fast, in);
    for (const char* lever : {"index_proj_wmma", "index_score_wmma", "ple_rows_f16", "gdn_gates_f16"}) {
        const auto* e = find(turbo, lever);
        require(e != nullptr && e->value == "1" && e->status == p::Status::provisional, lever);
        require(find(def, lever) == nullptr && find(fast, lever) == nullptr, "turbo-only row is absent outside turbo");
    }
    for (const char* lever : {"qsa_budget", "moe_route_bf16", "gdn_fp16_chunk"}) {
        require(find(turbo, lever) != nullptr && find(turbo, lever)->value == find(fast, lever)->value, "turbo inherits the FAST cell");
        require(find(def, lever) == nullptr || find(def, lever)->status == p::Status::builtin, "default keeps the lever off");
    }
    const auto* prune = find(turbo, "moe_route_prune");
    require(prune != nullptr && prune->status == p::Status::pending && prune->target == "rel<f>[n] | topp<f>[n] | cap<k>[n]",
            "calibrated route pruning is a pending turbo cell");
    // One resolver: CLI spelling and serve-style inputs give the same fingerprint.
    const char* argv[] = {"app", "--profile", "turbo"};
    std::optional<p::Name> name;
    q::Inputs cli;
    int i = 1;
    require(q::take_option(3, const_cast<char**>(argv), i, name, cli), "CLI takes --profile turbo");
    const auto via_cli = q::resolve(name.value_or(p::Name::standard), cli);
    q::Inputs serve;
    serve.env = env_of(vars);
    require(via_cli.fingerprint() == q::resolve(p::Name::turbo, serve).fingerprint(), "CLI and serve share one qwen turbo fingerprint");
    // Opting a member out: the env value 0 wins and drops it from the effective set.
    const std::map<std::string, std::string> leave{{"STRIX_PREFILL_PLE_ROWS_F16", "0"}};
    q::Inputs one_out;
    one_out.env = env_of(leave);
    const auto without = q::resolve(p::Name::turbo, one_out);
    require(find(without, "ple_rows_f16")->status == p::Status::off && without.fingerprint() != turbo.fingerprint(),
            "a single lever can be left out");
}

void families() {
    using strix::deepseek41::pair_defaults::profile_table;
    const p::Engine ds{"deepseek41-pair", "", profile_table, false, "ds41-turbo-v1"};
    const std::map<std::string, std::string> none;
    p::Resolution r;
    generic(ds, p::Name::turbo, none, r);
    require(find(r, "attention_f16wmma") != nullptr && find(r, "attention_f16wmma")->value == "2", "ds41 turbo attention = bf16flash");
    require(find(r, "dense_f16") != nullptr && find(r, "dense_f16")->value == "1", "ds41 turbo dense f16");
    require(profile_table.size() == 9, "ds41 keeps nine rows (pending Engram replaced, opt-in shared_f16 and moe_f16wmma added)");
    require(find(r, "shared_f16") == nullptr || find(r, "shared_f16")->value != "1", "ds41 turbo does not turn on shared_f16");
    require(find(r, "moe_f16wmma") == nullptr || find(r, "moe_f16wmma")->value != "1", "ds41 turbo does not turn on moe_f16wmma");
    std::size_t engram_rows = 0;
    for (const auto& row : profile_table)
        if (std::string_view(row.env) == "STRIX_DS41_FAST_ENGRAM") ++engram_rows;
    require(engram_rows == 1 && find(r, "engram_wmma") == nullptr, "one real Engram row per env");
    const auto* engram = find(r, "engram_f16wmma");
    require(engram != nullptr && engram->status == p::Status::builtin && engram->value == "0" &&
                std::string_view(engram->lever->value[3]).empty() && !engram->lever->runnable[3],
            "ds41 real Engram stays explicit-only outside turbo");
    require(find(r, "decoder_replay") != nullptr && find(r, "decoder_replay")->status == p::Status::pending, "ds41 replay pending");
    generic(ds, p::Name::fast, none, r);
    require(find(r, "attention_f16wmma")->requested == "1", "ds41 fast unchanged (f16wmma)");
    generic(ds, p::Name::standard, none, r);
    require(find(r, "engram_f16wmma") != nullptr && find(r, "engram_f16wmma")->status == p::Status::builtin, "ds41 default unchanged");

    const p::Engine mm{"minimax-pair", "", strix::minimax::defaults::profile_table, false, "minimax-turbo-v1"};
    generic(mm, p::Name::turbo, none, r);
    require(strix::minimax::defaults::profile_table.size() == 9, "minimax retains six base rows, the FAST router rows and explicit-only TP2 exchange row");
    for (const char* lever : {"idx_wmma", "idx_wmma_layers", "router_wmma", "router_wmma_layers"}) {
        const auto* e = find(r, lever);
        require(e != nullptr && e->status == p::Status::builtin && e->value == e->lever->builtin &&
                    std::string_view(e->lever->value[3]).empty() && !e->lever->runnable[3],
                "minimax index rows stay explicit-only outside turbo");
    }
    require(find(r, "msa_flash")->status == p::Status::pending && find(r, "msa_flash_layers")->status == p::Status::pending,
            "minimax turbo flash waits for adapter admission/export");
    require(find(r, "moe_invariant")->status == p::Status::pending, "minimax moe invariant pending");
    generic(mm, p::Name::fast, none, r);
    require(find(r, "msa_flash") == nullptr || find(r, "msa_flash")->status == p::Status::builtin, "minimax fast unchanged");

    namespace md = strix::minimax::defaults;
    const std::map<std::string, std::string> mm_env{{md::kProductProfileEnv, "turbo"}};
    const auto admitted = md::profile_resolution(env_of(mm_env));
    const auto execution = md::prefill_policy(env_of(mm_env));
    require(find(admitted, "msa_flash")->value == std::to_string(execution.msa_flash),
            "MiniMax actual adapter receipt matches engine flash mode");
    require(find(admitted, "msa_flash_layers")->value == execution.msa_flash_layers,
            "MiniMax actual adapter receipt matches engine layer set");
    std::map<std::string, std::string> exported;
    p::apply(admitted, [&](const char* key, const char* value) { exported[key] = value; return true; });
    require(!exported.contains("STRIX_MM_PF_MSA_FLASH") && !exported.contains("STRIX_MM_PF_MSA_FLASH_LAYERS"),
            "pending MiniMax proposals never escape through the shared exporter");
    p::Inputs allow;
    allow.profile = p::Name::turbo;
    allow.allow_pending = true;
    const auto pending = p::resolve(mm, allow);
    require(find(pending, "msa_flash")->status == p::Status::pending, "allow-pending cannot bypass missing MiniMax adapter");

    generic(strix::glm::rank_defaults::product_profile_engine, p::Name::turbo, none, r);
    require(find(r, "pf_wmma")->value == "1", "glm turbo inherits pf_wmma");
    require(find(r, "kda_state16")->status == p::Status::pending && find(r, "router_indexer_wmma")->status == p::Status::pending,
            "glm router/indexer and KDA16 pending");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 2 && std::string_view(argv[1]) == "--minimax-cli")
            return strix::minimax::defaults::print_or_apply("rank", argc - 1, argv + 1) ? 0 : 1;
        if (argc >= 3 && std::string_view(argv[1]) == "--glm-arm") {
            for (const char* key : {"STRIX_GLM_PF_WMMA", "STRIX_GLM_PF_ATTN_WMMA", "STRIX_GLM_PF_KDA_WMMA"})
                require(p::export_environment(key, ""), "clear inherited GLM switch");
            for (int i = 3; i < argc; ++i) {
                const std::string assignment = argv[i];
                const auto eq = assignment.find('=');
                require(eq != std::string::npos && p::export_environment(assignment.substr(0, eq).c_str(), assignment.substr(eq + 1).c_str()), "set arm switch");
            }
            const std::string_view arm = argv[2];
            require(strix::glm::pf::wmma_x::enabled() == (arm != "default" && arm != "no-pf_wmma"), "GLM dense execution switch");
            require(strix::glm::pf::attn_wmma::enabled() == (arm != "default" && arm != "no-attn_wmma"), "GLM attention execution switch");
            require(strix::glm::pf::kda_wmma::enabled() == (arm != "default" && arm != "no-kda_wmma"), "GLM KDA execution switch");
            return failures == 0 ? 0 : 1;
        }
        qwen_rows();
        families();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: unexpected exception: %s\n", e.what());
        return 1;
    }
    if (failures != 0) return 1;
    std::puts("turbo bundle tests: ok");
    return 0;
}
