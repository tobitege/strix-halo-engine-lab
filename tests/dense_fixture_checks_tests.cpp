#include "dense_fixture_checks.hpp"

#include <cstdio>
#include <limits>
#include <sstream>

namespace {
unsigned failures = 0;

void check(bool pass, const char* name) {
    if (!pass) {
        std::fprintf(stderr, "FAIL %s\n", name);
        ++failures;
    }
}

template <class Call> void rejects(Call call, const char* name) {
    try {
        call();
        check(false, name);
    } catch (const std::runtime_error&) {
    }
}

void core_verdicts() {
    constexpr float guard = -1234.5F;
    // Independent hand-computed dot products, including row/column indexing.
    const std::vector<float> x{0.5F, -1.0F, 2.0F, 0.25F}, w{0.25F, 0.5F, -0.5F, 0.25F};
    const auto want = dense_fixture::exact_reference(x, w, 2, 2, 2, 1, guard);
    const std::vector<float> known{-0.375F, -0.5F, guard, 0.625F, -0.9375F, guard, guard, guard, guard};
    check(dense_fixture::same_bytes(want, known), "independent scalar oracle");
    dense_fixture::check_dense(known, want, 2, 2, 1, guard);
    rejects([&] { dense_fixture::check_dense(std::vector<float>(want.size(), guard), want, 2, 2, 1, guard); }, "no stores");
    for (unsigned rest = 1; rest < 16; ++rest) {
        const unsigned n = rest * 16U;
        std::vector<float> reference(std::size_t{3} * (n + 1U), guard);
        for (unsigned r = 0; r < 2; ++r)
            for (unsigned c = 0; c < n; ++c) reference[std::size_t{r} * (n + 1U) + c] = static_cast<float>(r + c);
        dense_fixture::check_dense(reference, reference, 2, n, 1, guard);
        for (const std::size_t index : {std::size_t{n - 1U}, std::size_t{n}, std::size_t{2U * (n + 1U)}}) {
            auto corrupt = reference;
            corrupt[index] += 1.0F;
            rejects([&] { dense_fixture::check_dense(corrupt, reference, 2, n, 1, guard); }, "remainder arithmetic/padding/guard corruption");
        }
        auto unwritten = reference;
        unwritten[n - 1U] = guard;
        rejects([&] { dense_fixture::check_dense(unwritten, reference, 2, n, 1, guard); }, "partial missing stores");
    }
}

void numeric_verdicts() {
    for (const char* message : {"vision encoder: numeric failure bits 1", "vision encoder: numeric failure bits 8",
                                "vision encoder: numeric failure bits 9"})
        dense_fixture::require_numeric_failure([&] { throw std::runtime_error(message); });
    rejects([] { dense_fixture::require_numeric_failure([] {}); }, "missing numeric report");
    for (const char* message : {"allocation failed", "device lost", "vision encoder: numeric failure bits 0",
                                "vision encoder: numeric failure bits 2", "vision encoder: numeric failure bits 8junk"})
        rejects([&] { dense_fixture::require_numeric_failure([&] { throw std::runtime_error(message); }); }, "unrelated error accepted");
    bool propagated = false;
    try {
        dense_fixture::require_numeric_failure([] { throw std::bad_alloc(); });
    } catch (const std::bad_alloc&) {
        propagated = true;
    }
    check(propagated, "allocation exception propagated");
}

void repeat_verdicts() {
    struct Output { std::vector<float> hidden; std::vector<std::vector<float>> taps; };
    const Output first{{0.0F, 2.0F}, {{3.0F}}};
    dense_fixture::require_repeatable(first, first);
    auto changed = first;
    changed.hidden[0] = -0.0F;
    rejects([&] { dense_fixture::require_repeatable(changed, first); }, "forward byte change");
    changed = first;
    changed.taps[0][0] += 1.0F;
    rejects([&] { dense_fixture::require_repeatable(changed, first); }, "tap byte change");
    changed.taps.clear();
    rejects([&] { dense_fixture::require_repeatable(changed, first); }, "missing tap");
}

void linear_verdicts() {
    const std::vector<float> values{1.0F, 2.0F};
    std::ostringstream saved(std::ios::binary);
    dense_fixture::linear_reference(nullptr, &saved, values, 1, 16, 2);
    const auto compare = [&](const std::string& bytes, const std::vector<float>& got, unsigned in = 16) {
        std::istringstream input(bytes, std::ios::binary);
        dense_fixture::linear_reference(&input, nullptr, got, 1, in, 2);
        dense_fixture::finish_reference(input);
    };
    compare(saved.str(), values);
    rejects([&] { compare(saved.str(), {1.0F, 3.0F}); }, "OFF byte mismatch");
    rejects([&] { compare(saved.str(), values, 32); }, "OFF shape mismatch");
    rejects([&] { compare(saved.str().substr(0, saved.str().size() - 1U), values); }, "truncated OFF reference");
    rejects([&] { compare(saved.str() + 'x', values); }, "trailing OFF record");
    rejects([&] { compare("", values); }, "missing OFF reference");
    rejects([&] { compare(saved.str(), {1.0F, std::numeric_limits<float>::infinity()}); }, "nonfinite linear");
}
}  // namespace

int main() {
    try {
        core_verdicts();
        numeric_verdicts();
        repeat_verdicts();
        linear_verdicts();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL unexpected exception: %s\n", e.what());
        return 1;
    }
    std::printf("dense fixture host verdicts: %u failures\n", failures);
    return failures ? 1 : 0;
}
