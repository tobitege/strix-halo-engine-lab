#include "strix/env_arms.hpp"
#include "strix/env_flags.hpp"
#include <cstdint>
#include <cstdio>
#include <string_view>

using hipStream_t = void*;
static unsigned diagnostic_calls = 0;
#include "overlap_guard.inc"

static int failures = 0;
static void check(bool value, const char* message) {
    if (!value) { ++failures; std::fprintf(stderr, "%s\n", message); }
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    namespace A = strix::env_arms;
    namespace E = strix::envcfg;
    const std::string_view test(argv[1]);
    if (test == "no-arms") {
        const auto arms = A::parse("-:-");
        A::Resolved<bool> cached;
        int resolves = 0;
        const auto resolve = [&] { ++resolves; return false; };
        (void)cached.get(resolve);
        const auto generation = A::generation();
        for (int rep = 0; rep < 4; ++rep) {
            A::apply(arms, arms.arms[0]);
            (void)cached.get(resolve);
        }
        A::restore(arms);
        (void)cached.get(resolve);
        check(A::generation() == generation && resolves == 1, "no-arm runs must preserve cached resolution");
        const auto real = A::parse("B:-;C:STRIX_ENV_ARMS_TEST_REAL=1");
        A::apply(real, real.arms[1]);
        (void)cached.get(resolve);
        A::apply(real, real.arms[0]);
        (void)cached.get(resolve);
        A::restore(real);
        (void)cached.get(resolve);
        check(resolves == 4 && A::generation() == generation + 3, "real overrides and restorations must invalidate");
    } else if (test == "governed-snapshot") {
        A::set_env("STRIX_ENV_ARMS_TEST_SET", std::string("launch value"));
        A::set_env("STRIX_ENV_ARMS_TEST_UNSET", std::nullopt);
        const auto arms = A::parse("B:-;C:STRIX_ENV_ARMS_TEST_SET=changed+STRIX_ENV_ARMS_TEST_UNSET=1");
        const auto entries = E::registered();
        check(entries.size() == 2, "arm parsing must register both dynamic keys");
        if (entries.size() == 2) {
            check(entries[0].set && entries[0].value == "launch value", "present launch value must be recorded");
            check(!entries[1].set && entries[1].value.empty(), "unset launch value must be recorded");
        }
        A::apply(arms, arms.arms[1]);
        check(arms.launch[0].second == "launch value" && !arms.launch[1].second, "launch snapshot must own its text");
        A::restore(arms);
        check(std::string(E::text("STRIX_ENV_ARMS_TEST_SET")) == "launch value" && E::text("STRIX_ENV_ARMS_TEST_UNSET") == nullptr,
              "launch values must be restored");
#ifndef _WIN32
        A::set_env("STRIX_ENV_ARMS_TEST_EMPTY", std::string());
        const auto empty = A::parse("C:STRIX_ENV_ARMS_TEST_EMPTY=1");
        check(empty.launch[0].second && empty.launch[0].second->empty(), "present empty must differ from unset");
#endif
    } else if (test == "overlap-cache") {
        constexpr const char* key = "STRIX_GLM_PF_TOPK_OVERLAP";
        A::set_env(key, std::nullopt);
        topk_overlap_diag(nullptr, nullptr, 0, nullptr);
        check(diagnostic_calls == 0 && E::registered().size() == 1, "first disabled call must resolve through envcfg");
        A::set_env(key, std::string("1"));
        {
            // CTest timeout detects any steady-state registry lock acquisition.
            std::lock_guard<std::mutex> lock(E::registry_mutex());
            for (int i = 0; i < 100; ++i) topk_overlap_diag(nullptr, nullptr, 0, nullptr);
        }
        check(diagnostic_calls == 0, "steady-state guard must not re-read the environment");
        A::set_env(key, std::nullopt);
        const auto arms = A::parse("B:-;C:STRIX_GLM_PF_TOPK_OVERLAP=1");
        A::apply(arms, arms.arms[1]);
        topk_overlap_diag(nullptr, nullptr, 0, nullptr);
        check(diagnostic_calls == 1, "real arm transition must re-resolve enabled diagnostic");
        A::restore(arms);
        topk_overlap_diag(nullptr, nullptr, 0, nullptr);
        check(diagnostic_calls == 1, "restoration must disable the diagnostic again");
    } else return 2;
    return failures ? 1 : 0;
}
