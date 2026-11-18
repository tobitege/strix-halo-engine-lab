#include "strix/minimax_layercut.hpp"
#include <cstdio>
#include <stdexcept>

namespace lc = strix::pair::layercut;
namespace mlc = strix::minimax::layercut;
// Only the prompt reader, the engine chunk default and the policy suffix are substituted; all argument parsing and
// the pair identity below come from the production adapter.
namespace mm { struct Options { std::uint32_t max_rows = mlc::pair_chunk_rows; }; }
namespace strix::minimax::defaults {
inline std::string prefill_identity(const lc::Lookup&) { return "test-policy"; }
}
std::vector<std::uint32_t> read_u32(const std::string&) { return std::vector<std::uint32_t>(4096, 1); }
#include "cut_arguments_under_test.inc"

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

int main() try {
    std::vector<std::string> args{"rank", "MODEL_DIR", "0", "usb4-auto", "OUT_DIR",
        "--cut-xport", "fifo", "--layercut-listen", "test:prompt.u32:1234:1:1",
        "--cut", "1", "--cut-kv-frag", "256", "--cut-block", "4096", "--cut-act", "link",
        "--split", "layer-cut", "--session", "keep-this-value"};
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    argv.push_back(nullptr);
    int argc = static_cast<int>(args.size());
    const auto cut = cut_arguments(argc, argv.data());
    require(cut.config.xport == lc::Xport::fifo && cut.config.kv_frag == 256, "transport values resolved");
    require(cut.config.cut == 1 && cut.config.block == 4096 && cut.listens.size() == 1, "other cut values retained");
    // Execution continues over the remaining argv, without taking the inspection early exit.
    unsigned consumed = 0;
    for (int i = 5; i < argc; ++i) {
        const std::string flag = argv[static_cast<std::size_t>(i)];
        require(flag == "--layercut-listen" || flag == "--session", "execution received an unconsumed cut option");
        require(++i < argc, "execution option value retained");
        ++consumed;
    }
    require(argc == 9 && consumed == 2 && argv[static_cast<std::size_t>(argc)] == nullptr, "remaining argv is intact");
    for (const char* flag : {"--cut-xport", "--cut-kv-frag"}) {
        std::vector<std::string> missing{"rank", "MODEL_DIR", "0", "usb4-auto", "OUT_DIR", flag};
        std::vector<char*> av;
        for (auto& arg : missing) av.push_back(arg.data());
        av.push_back(nullptr);
        int ac = static_cast<int>(missing.size());
        bool rejected = false;
        try { (void)cut_arguments(ac, av.data()); }
        catch (const std::invalid_argument& e) { rejected = std::string(e.what()).find(flag) != std::string::npos; }
        require(rejected, "missing transport value names flag");
    }
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
