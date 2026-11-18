#include "../apps/hip_decode_options.hpp"
#include "strix/qwen_prefill_balance.hpp"
#include <cstdio>

// This CLI-only fixture must not load assets, deployment files or export state. Fail closed if
// parsing reaches those unrelated runtime branches; keep their engine dependencies out of this build.
namespace strix {
namespace deploy { Spec load_spec(const std::string&) { throw std::logic_error("unexpected deployment load"); } }
ModelContentIdentity sha256_file_content(const std::filesystem::path&, FileHashPolicy) {
    throw std::logic_error("unexpected asset hash");
}
ModelContentIdentity sha256_bytes(std::span<const std::byte>) { throw std::logic_error("unexpected content hash"); }
std::string model_content_identity_hex(const ModelContentIdentity&) { throw std::logic_error("unexpected asset identity"); }
void validate_session_snapshot_policy(SessionSnapshotPolicy policy) {
    if (policy != SessionSnapshotPolicy::full_components) throw std::logic_error("unexpected snapshot policy");
}
void validate_raw_baseline_output_directory(const std::filesystem::path&) { throw std::logic_error("unexpected raw export"); }
void validate_final_generation_state_destination(const std::filesystem::path&) { throw std::logic_error("unexpected state export"); }
}

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void set_env(const char* key, const char* value) {
#ifdef _WIN32
    require(_putenv_s(key, value ? value : "") == 0, "set environment");
#else
    require((value ? ::setenv(key, value, 1) : ::unsetenv(key)) == 0, "set environment");
#endif
}
}
int main(int argc, char** argv) try {
    using namespace strix::qwen_pp;
    const bool cli = argc == 2 && std::string_view(argv[1]) == "cli";
    const bool baseline = argc == 2 && std::string_view(argv[1]) != "default";
    set_env("STRIX_BASELINE", baseline && !cli ? "1" : nullptr);
    set_env("STRIX_QWEN_PAIR_CHUNK_POLICY", nullptr);
    set_env("STRIX_QWEN_PAIR_SINGLE_TILES", nullptr);
    set_env("STRIX_QWEN_PAIR_HALF_CUT", nullptr);
    set_env("STRIX_QWEN_PF_PIPE_CHUNK", nullptr);
    std::vector<std::string> args{"decoder", "model.gguf", "--no-mtp",
        "--known-prefix-window", "--known-prefix-full-rows", "--known-prefix-prefill-rows", "8192",
        "--known-prefix-workspace-budget-bytes", "1", "--known-prefix-host-metadata-budget-bytes", "2",
        "--known-prefix-uma-budget-bytes", "1", "--known-prefix-host-headroom-bytes", "0"};
    if (cli) args.emplace_back("--baseline");
    args.insert(args.end(), 8192, "1");
    std::vector<char*> pointers;
    for (auto& arg : args) pointers.push_back(arg.data());
    const auto options = strix::decode_cli::parse_options(static_cast<int>(pointers.size()), pointers.data(), true);
    require(options.baseline == baseline && options.tokens.size() == 8192U &&
        options.known_prefix_window && options.known_prefix_full_rows && options.known_prefix_prefill_rows == 8192U,
        "CLI/env baseline preserves explicit eligible tiled settings");
    for (const auto execution : {PairExecution::handoff, PairExecution::prefill_only}) {
        const auto policy = resolve_pair_policy(options.tokens.size(), execution, options.baseline);
        const auto route = resolve_pair_route(options.tokens.size(), options.known_prefix_prefill_rows,
            nullptr, nullptr, {}, policy.chunk_policy.on, policy.single_tiles.on, policy.execution);
        require(route.route.prefix == (baseline && execution == PairExecution::handoff ? 0U : 8192U) &&
            route.tile_rows == (baseline ? 4096U : 2048U), "baseline restores historical route and tile size");
        require(policy.chunk_policy.on == !baseline && policy.single_tiles.on == !baseline, "automatic choices obey baseline");
        require(policy.half_cut.on == !baseline, "half cut: auto on at 8K, off under baseline");
        const auto config = strix::serve::parse_json(inspect_pair_policy("{}", policy));
        const auto* control = config.find("qwen_pair")->find("STRIX_QWEN_PAIR_CHUNK_POLICY");
        require(control->find("requested")->as_string() == "auto" && control->find("effective")->as_bool() == !baseline,
            "inspection reports saved baseline choice");
        if (baseline) require(control->find("source")->as_string() == "baseline", "baseline provenance");
    }
    set_env("STRIX_QWEN_PAIR_CHUNK_POLICY", "1");
    set_env("STRIX_QWEN_PAIR_SINGLE_TILES", "1");
    const auto explicit_policy = resolve_pair_policy(8192, PairExecution::handoff, baseline);
    require(explicit_policy.chunk_policy.on && explicit_policy.single_tiles.on, "explicit switches retain precedence");
    set_env("STRIX_QWEN_PAIR_CHUNK_POLICY", "invalid");
    bool rejected = false;
    try { (void)resolve_pair_policy(8192, PairExecution::handoff, baseline); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "baseline does not bypass boolean validation");
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
