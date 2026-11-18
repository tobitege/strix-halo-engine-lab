#include "strix/minimax_layercut.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace lc = strix::pair::layercut;
namespace mlc = strix::minimax::layercut;
// Stand-ins only for engine defaults, policy suffix and prompt I/O; the CLI
// parsing, identity construction and resolver calls below come from production.
namespace mm { struct Options { std::uint32_t max_rows = mlc::pair_chunk_rows; }; }
namespace strix::minimax::defaults {
inline std::string prefill_identity(const lc::Lookup&) { return "test-policy"; }
}
std::vector<std::uint32_t> read_u32(const std::string&) { return std::vector<std::uint32_t>(16); }
#include "rank_arguments.inc"

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void set_env(const char* key, const char* value) {
#ifdef _WIN32
    require(_putenv_s(key, value) == 0, "set env");
#else
    require(setenv(key, value, 1) == 0, "set env");
#endif
}
CutArguments rank_args(std::vector<std::string> flags) {
    std::vector<std::string> values{"rank", "model", "0", "usb4-auto", "out"};
    values.insert(values.end(), flags.begin(), flags.end());
    std::vector<char*> argv;
    for (auto& value : values) argv.push_back(value.data());
    int argc = static_cast<int>(argv.size());
    argv.push_back(nullptr);
    return cut_arguments(argc, argv.data());
}
int main(int argc, char** argv) try {
    for (const char* key : {"STRIX_SPLIT", "STRIX_SPLIT_CUT", "STRIX_SPLIT_BLOCK", "STRIX_SPLIT_ACT",
                           "STRIX_MM_PREFILL_IDENTITY", "STRIX_SPLIT_RTX_ROWS_S", "STRIX_SPLIT_PAIR_ROWS_S",
                           "STRIX_SPLIT_LINK_MBPS", "STRIX_HANDOFF_LINK"}) set_env(key, "");
    if (argc == 2 && std::string_view(argv[1]) == "rank") {
        for (const char* rows : {"8", "24", "4097"}) {
            const auto normal = rank_args({"--max-rows", rows});
            require(normal.config.split == lc::chain::Split::device_only &&
                normal.chunk_rows == lc::number(rows, "rows"), "ordinary max-rows preserved");
        }
        const auto listener = rank_args({"--layercut-listen", "case:prompt:46000:0:1", "--cut", "24"});
        require(listener.config.grain == 4096 && listener.config.block == 4096, "listener default grain");
        require(cut_pair_identity(4096) == "default/c4096;test-policy", "policy suffix preserved");
        set_env("STRIX_SPLIT", "layer-cut");
        require(rank_args({"--split", "device-only", "--max-rows", "8"}).config.split ==
            lc::chain::Split::device_only, "CLI device-only overrides environment");
        for (const char* route : {"layer-cut", "auto"}) {
            set_env("STRIX_SPLIT", route);
            bool rejected = false;
            try { (void)rank_args({"--max-rows", "8", "--cut", "24"}); }
            catch (const std::invalid_argument& e) {
                rejected = std::string_view(e.what()).find("multiple of 16") != std::string_view::npos;
            }
            require(rejected, "layer-cut and auto still enforce grain");
        }
        return 0;
    }
    require(argc >= 5, "expected actual producer arguments");
    lc::Cli cut_cli;
    std::string cut_pair_identity = mlc::default_pair_identity();
    bool print_effective = false;
    struct Prompt { std::vector<std::uint32_t> tokens; };
    std::vector<Prompt> prompts;
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        auto cut_value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
#include "producer_arguments.inc"
        if (a == "--resident" || a == "--repeat" || a == "--send" || a == "--codec") {
            (void)cut_value(); continue;
        }
        require(a.starts_with("cut:"), "unexpected wrapper argument");
        prompts.push_back({read_u32(a.substr(4))});
    }
#include "producer_resolution.inc"
    (void)print_effective;
    std::printf("%s", cut_cfg.lines.c_str());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
}
