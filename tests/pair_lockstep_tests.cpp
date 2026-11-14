#include "strix/pair/lockstep.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

template <class F>
std::string thrown(F&& f) {
    try {
        f();
    } catch (const strix::pair::ProtocolError& e) {
        return e.what();
    }
    return {};
}

void test_rounds_advance_in_step() {
    strix::pair::Lockstep a(0), b(1);
    for (int k = 0; k < 4; ++k) {
        const auto ha = a.header(), hb = b.header();
        a.accept(hb);
        b.accept(ha);
    }
    expect(a.round() == 4U && b.round() == 4U, "both ranks advance one round per exchange");
    expect(a.header().ack_round == 3U, "ack_round is the last completed round");
    expect(strix::pair::Lockstep(0).header().ack_round == strix::pair::kNoRound, "first round acknowledges none");
}

void test_duplicate_or_stale_round() {
    strix::pair::Lockstep a(0), b(1);
    a.accept(b.header());  // b missed this round (dropped message): a is one ahead
    const auto e = thrown([&] { a.accept(b.header()); });
    expect(e.find("stale round 0, expected 1") != std::string::npos, "stale round names both rounds");
    expect(a.round() == 1U, "a rejected round does not advance");
    expect(!thrown([] { strix::pair::check_round(7, 8); }).empty(), "check_round rejects a mismatch");
    expect(thrown([] { strix::pair::check_round(7, 7); }).empty(), "check_round accepts an equal round");
}

void test_ack_mismatch() {
    strix::pair::LockstepHeader mine{3, 2, 0}, peer{3, 1, 1};
    expect(thrown([&] { strix::pair::check_pair(mine, peer); }).find("not acknowledged") != std::string::npos,
           "ack mismatch rejected");
}

void test_same_rank_and_range() {
    strix::pair::Lockstep a(0), twin(0);
    expect(thrown([&] { a.check(twin.header()); }).find("same rank") != std::string::npos, "same rank rejected");
    strix::pair::LockstepHeader mine{0, strix::pair::kNoRound, 0}, peer{0, strix::pair::kNoRound, 2};
    expect(thrown([&] { strix::pair::check_pair(mine, peer); }).find("out of range") != std::string::npos,
           "rank > 1 rejected");
}

}  // namespace

int main() {
    test_rounds_advance_in_step();
    test_duplicate_or_stale_round();
    test_ack_mismatch();
    test_same_rank_and_range();
    if (failures) return EXIT_FAILURE;
    std::printf("PASS pair lockstep\n");
    return EXIT_SUCCESS;
}
