#pragma once

#include <charconv>
#include <string_view>
#include <system_error>

namespace strix::test {
struct DenseFormsArgs {
    unsigned reps = 10U;
    unsigned rounds = 7U;
    unsigned rows = 1024U;
};

// Bound benchmark work and allocations; reject signs, suffixes and overflow.
inline bool parse_dense_forms_args(int argc, const char* const* argv, DenseFormsArgs& args) {
    if (argc < 2 || argc > 5 || std::string_view(argv[1]) != "--forms") return false;
    unsigned* values[] = {&args.reps, &args.rounds, &args.rows};
    constexpr unsigned minimum[] = {1U, 2U, 1U};
    constexpr unsigned maximum[] = {1000U, 101U, 16384U};
    for (int i = 2; i < argc; ++i) {
        const std::string_view text(argv[i]);
        unsigned value = 0;
        const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
        if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
            value < minimum[i - 2] || value > maximum[i - 2]) return false;
        *values[i - 2] = value;
    }
    return true;
}
}  // namespace strix::test
