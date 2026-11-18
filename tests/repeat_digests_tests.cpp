// DAYAUDIT 7 (P2): the --repeat loop must compare the digests of every repetition and fail on a difference.
#include "strix/repeat_digests.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
void expect(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
}  // namespace

int main() try {
    strix::RepeatDigests insufficient;
    expect(insufficient.equal() && insufficient.report("p").find("0 comparisons (insufficient repetitions)") != std::string::npos,
           "zero runs do not establish repeatability");
    insufficient.add({0, "A", 1, 2, 3});
    insufficient.add({1, "B", 4, 5, 6});
    for (const auto* arm : {"A", "B"})
        expect(insufficient.report("p").find(std::string("arm ") + arm + ": 1 runs, 0 comparisons (insufficient repetitions)") != std::string::npos,
               "singleton arms do not establish repeatability");
    strix::RepeatDigests same;
    for (std::uint32_t k = 0; k < 3; ++k) same.add({k, "", 0x10257ebdULL, 0x55ULL, 42U});
    expect(same.equal(), "equal repetitions");
    expect(same.report("prefill").find("prefill repeat digests: 3 runs, equal") != std::string::npos, "equal verdict line");
    expect(same.report("prefill").find("prefill repeat 2: next token 42, logits_fnv 0000000010257ebd") != std::string::npos,
           "one line per repetition");

    // the incident's shape: the second run of one process changes its logits hash (10257ebd then 2e268be6)
    strix::RepeatDigests drift;
    drift.add({0, "", 0x10257ebdULL, 0, 7U});
    drift.add({1, "", 0x2e268be6ULL, 0, 7U});
    expect(!drift.equal(), "a different logits hash is a mismatch");
    const auto text = drift.report("rank 0");
    expect(text.find("rank 0 REPEAT DIGEST MISMATCH: run 1 (logits_fnv 000000002e268be6") != std::string::npos &&
               text.find("!= run 0 (logits_fnv 0000000010257ebd") != std::string::npos,
           "the mismatch line names both runs");

    strix::RepeatDigests streams;  // equal logits, a layer stream differs (state drift before the head)
    streams.add({0, "", 1U, 0xAAU, 3U});
    streams.add({1, "", 1U, 0xABU, 3U});
    expect(!streams.equal(), "a different stream digest is a mismatch");

    strix::RepeatDigests tokens;
    tokens.add({0, "", 1U, 2U, 3U});
    tokens.add({1, "", 1U, 2U, 4U});
    tokens.add({2, "", 1U, 2U, 3U});
    expect(!tokens.equal(), "next-token-only drift is retained even after recovery");
    expect(tokens.report("p").find("3 runs, 2 comparisons") != std::string::npos, "comparison count excludes reference");

    strix::RepeatDigests ab;  // A/B arms compare only within their own arm
    ab.add({0, "OFF", 1U, 0, 3U});
    ab.add({1, "SEL", 2U, 0, 3U});
    ab.add({2, "OFF", 1U, 0, 3U});
    ab.add({3, "SEL", 2U, 0, 3U});
    expect(ab.equal(), "arms differ from each other, each arm is stable");
    ab.add({4, "SEL", 9U, 0, 3U});
    const strix::RepeatDigests::Run* ref = nullptr;
    const auto* bad = ab.first_mismatch(&ref);
    expect(bad != nullptr && bad->index == 4U && ref != nullptr && ref->index == 1U, "compared with the first run of its arm");
    expect(ab.report("p").find("arm OFF: 2 runs, 1 comparisons") != std::string::npos &&
               ab.report("p").find("arm SEL: 3 runs, 2 comparisons") != std::string::npos, "counts stay within each arm");
    std::puts("repeat digests: ok");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}
