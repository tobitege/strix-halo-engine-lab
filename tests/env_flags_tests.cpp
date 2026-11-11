#include "strix/env_flags.hpp"

#include <cstdio>
#include <string>

int main() {
    constexpr const char* key = "STRIX_ENV_FLAGS_TEST_VALUE";
    const char* old = std::getenv(key);
    const bool had_old = old != nullptr;
    const std::string saved = old ? old : "";
    const auto put = [&](const char* value) {
#ifdef _WIN32
        return _putenv_s(key, value ? value : "");
#else
        return value ? ::setenv(key, value, 1) : ::unsetenv(key);
#endif
    };
    int checks = 0, failures = 0;
    const auto check = [&](bool ok) { ++checks; if (!ok) ++failures; };
    const char* cases[] = {nullptr, "", "0", "1", "00", "01", "10", "11", "0tail", "1tail",
                              "true", "false", " 0", "0 ", " 1", "1 ", "2", "-1"};
    for (const char* value : cases) {
        if (put(value) != 0) { std::fprintf(stderr, "cannot set test environment\n"); return 2; }
        const bool exact_zero = value && std::string(value) == "0";
        const bool exact_one = value && std::string(value) == "1";
        const bool first_zero = value && value[0] == '0';
        const bool first_one = value && value[0] == '1';
        // Original Gemma coordinator/device helpers: entire string, unset/empty ON.
        check(strix::env_flag_default_on(key) == !exact_zero);
        // Original DS41 batch helper: first byte, including "0tail" as OFF.
        check(strix::env_flag_default_on(key, strix::EnvFlagMatch::first_character) == !first_zero);
        // Original prefix-window and DS41 layer helpers: first byte '1', not exact "1".
        check(strix::env_flag_default_off(key, strix::EnvFlagMatch::first_character) == first_one);
        check(strix::env_flag_default_off(key) == exact_one);
        // Windows removes empty environment values; test the empty literal directly too.
        check(strix::detail::env_flag_matches(value, "0", strix::EnvFlagMatch::exact) == exact_zero);
        check(strix::detail::env_flag_matches(value, "1", strix::EnvFlagMatch::first_character) == first_one);
    }
    if (put(had_old ? saved.c_str() : nullptr) != 0) return 2;
    std::printf("env flags: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
