#include "strix/env_flags.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
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
    // Numbers, byte counts and choices: unset/empty is the fallback, junk throws, the whole value must parse.
    const auto rejects = [&](auto&& read) {
        try { (void)read(); } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    const auto with = [&](const char* value, auto&& body) {
        if (put(value) != 0) { std::fprintf(stderr, "cannot set test environment\n"); std::exit(2); }
        body();
    };
    with(nullptr, [&] {
        check(strix::env_number<int>(key, -7, 0, 10) == -7);
        check(strix::env_bytes(key, 5) == 5U);
        check(strix::env_choice(key, 1, {"a", "b"}) == 1U);
    });
    with("12", [&] {
        check(strix::env_number<unsigned>(key, 0U) == 12U);
        check(strix::env_number<double>(key, 0.0) == 12.0);
        check(rejects([&] { return strix::env_number<int>(key, 0, 0, 10); }));
        check(strix::env_bytes(key, 0) == 12U);
    });
    with("010", [&] { check(strix::env_number<std::uint64_t>(key, 0) == 10U); });  // decimal, not octal
    with("0x10", [&] { check(rejects([&] { return strix::env_number<unsigned>(key, 0U); })); });
    with("-3", [&] {
        check(strix::env_number<int>(key, 0) == -3);
        check(rejects([&] { return strix::env_number<unsigned>(key, 0U); }));
        check(rejects([&] { return strix::env_bytes(key, 0); }));
    });
    with("0.25", [&] {
        check(strix::env_number<float>(key, 1.0f, 0.0f, 1.0f) == 0.25f);
        check(rejects([&] { return strix::env_number<int>(key, 0); }));
    });
    for (const char* junk : {"12x", " 1", "1 ", "nan", "inf", "false"})
        with(junk, [&] {
            check(rejects([&] { return strix::env_number<double>(key, 0.0); }));
            check(rejects([&] { return strix::env_number<long>(key, 0L); }));
        });
    with("99999999999999999999999", [&] { check(rejects([&] { return strix::env_number<std::uint64_t>(key, 0); })); });
    with("3G", [&] { check(strix::env_bytes(key, 0) == (std::uint64_t{3} << 30U)); });
    with("512KiB", [&] { check(strix::env_bytes(key, 0) == (std::uint64_t{512} << 10U)); });
    with("2MB", [&] { check(strix::env_bytes(key, 0) == (std::uint64_t{2} << 20U)); });
    with("2Q", [&] { check(rejects([&] { return strix::env_bytes(key, 0); })); });
    with("20000000T", [&] { check(rejects([&] { return strix::env_bytes(key, 0); })); });
    with("4G", [&] { check(rejects([&] { return strix::env_bytes(key, 0, 0, std::uint64_t{1} << 30U); })); });
    with("b", [&] { check(strix::env_choice(key, 0, {"a", "b", "c"}) == 1U); });
    with("B", [&] { check(rejects([&] { return strix::env_choice(key, 0, {"a", "b"}); })); });
    // envcfg:: registry helpers (R2-15).
    if (put(nullptr) != 0) return 2;
    check(strix::envcfg::flag(key, true) == true);
    check(strix::envcfg::flag(key, false) == false);
    check(strix::envcfg::integer(key, 7) == 7);
    check(strix::envcfg::text(key) == nullptr);
    check(strix::envcfg::choice(key, {"a", "b"}, 5) == 5);
    if (put("b") != 0) return 2;
    check(strix::envcfg::choice(key, {"a", "b"}, 5) == 1);
    check(strix::envcfg::flag(key, true) == true);
    check(strix::envcfg::flag(key, false) == false);
    if (put("0") != 0) return 2;
    check(strix::envcfg::flag(key, true) == false);
    check(strix::envcfg::integer(key, 7) == 0);
    if (put("12") != 0) return 2;
    check(strix::envcfg::integer(key, 7) == 12);
    // Strict recorded number: the value, its range and the registry entry; a bad value throws and records nothing new.
    check(strix::envcfg::number<unsigned>(key, 3U, 0U, 64U) == 12U);
    check(rejects([&] { return strix::envcfg::number<unsigned>(key, 3U, 0U, 8U); }));
    if (put("12x") != 0) return 2;
    check(rejects([&] { return strix::envcfg::number<unsigned>(key, 3U, 0U, 64U); }));
    {
        bool found = false;
        for (const auto& entry : strix::envcfg::registered()) if (entry.name == key) found = entry.value == "12" && entry.set;
        check(found);
    }
    if (put("12") != 0) return 2;
    check(strix::envcfg::flag(key, false) == true);  // first-character '1'
    check(std::string(strix::envcfg::text(key)) == "12");
    {
        int hits = 0;
        for (const auto& e : strix::envcfg::registered())
            if (e.name == key) { ++hits; check(e.set && e.value == "12"); }
        check(hits == 1);  // one entry per name, last value wins
        check(strix::envcfg::describe().find(std::string("env ") + key + " 12 (set)\n") != std::string::npos);
    }
    if (put(had_old ? saved.c_str() : nullptr) != 0) return 2;
    std::printf("env flags: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
