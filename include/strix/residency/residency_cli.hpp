#pragma once
// The residency command-line surface shared by every entry point (the Qwen CUDA serve, qwen_decode_bench): one consumer
// for `--expert-residency auto|full|tiered`, `--vram-budget-gib N` and `--print-effective-config`, and one finisher that
// folds in the environment (flags win). Header only and CUDA-free, so a host test can drive it.
#include "strix/residency/expert_residency.hpp"

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix::residency {

struct CliOptions {
    Options options;
    bool print_effective_config = false;
};

// Consumes one argument (and its value through `next`) when it is a residency flag. Returns false for any other argument.
// A malformed value throws std::invalid_argument (parse_flag's rule).
inline bool consume_cli_arg(std::string_view arg, const std::function<std::string()>& next, CliOptions& out) {
    if (arg == "--print-effective-config") { out.print_effective_config = true; return true; }
    return parse_flag(arg, next, out.options);
}

// A tail of arguments made only of residency flags (the bench's optional arguments after its positionals).
// Returns the first unrecognised argument index, or args.size() when all were consumed.
inline std::size_t consume_cli_tail(const std::vector<std::string>& args, std::size_t first, CliOptions& out) {
    std::size_t i = first;
    while (i < args.size()) {
        const auto next = [&]() -> std::string {
            if (i + 1 >= args.size()) throw std::invalid_argument(args[i] + " needs a value");
            return args[++i];
        };
        if (!consume_cli_arg(args[i], next, out)) return i;
        ++i;
    }
    return i;
}

// After every flag was seen: the environment fills what the flags left unset.
inline Options finish_cli(const CliOptions& in) { return with_environment(in.options); }

}  // namespace strix::residency
