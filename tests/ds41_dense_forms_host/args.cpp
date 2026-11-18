#include "../hip_ds41_fast_dense_args.hpp"

#include <cstdio>
#include <string>

int main() {
    using strix::test::DenseFormsArgs;
    using strix::test::parse_dense_forms_args;
    const char* argv[] = {"test", "--forms", "20", "11", "680", "extra"};
    DenseFormsArgs args;
    if (!parse_dense_forms_args(2, argv, args) || args.reps != 10U || args.rounds != 7U || args.rows != 1024U) return 1;
    if (!parse_dense_forms_args(5, argv, args) || args.reps != 20U || args.rounds != 11U || args.rows != 680U) return 1;
    if (parse_dense_forms_args(6, argv, args)) return 1;
    const unsigned low[] = {1U, 2U, 1U}, high[] = {1000U, 101U, 16384U};
    for (int i = 2; i < 5; ++i) {
        const char* saved = argv[i];
        for (const char* bad : {"", "0", "-1", "+1", "abc", "1x", " 1", "1 ", "4294967296"}) {
            argv[i] = bad;
            if (parse_dense_forms_args(5, argv, args)) {
                std::fprintf(stderr, "accepted invalid argument %d: %s\n", i, bad);
                return 1;
            }
        }
        for (unsigned value : {low[i - 2] - 1U, low[i - 2], high[i - 2], high[i - 2] + 1U}) {
            const std::string text = std::to_string(value);
            argv[i] = text.c_str();
            const bool expected = value >= low[i - 2] && value <= high[i - 2];
            if (parse_dense_forms_args(5, argv, args) != expected) return 1;
        }
        argv[i] = saved;
    }
    return 0;
}
