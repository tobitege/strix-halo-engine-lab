// Host tests of the shared one-completion-per-cycle pack (include/strix/session/cycle_pack.hpp): record layout,
// limits, settlement/proof lifetime, and the experimental default-OFF lever.
#include "strix/session/cycle_pack.hpp"
#include "strix/deferred_moe_status.hpp"
#include "strix/q51_known_rows.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

#if defined(_MSC_VER)
#include <crtdbg.h>
#endif

using namespace strix;

namespace {
int failures = 0;
void check(bool value, const char* label) {
    if (!value) {
        ++failures;
        std::cerr << "FAIL: " << label << '\n';
    }
}
template <class E, class F> bool throws(F&& f) {
    try { f(); } catch (const E&) { return true; } catch (...) { return false; }
    return false;
}

void layout_places_records_back_to_back() {
    session::CyclePackLayout l;
    check(l.add(8) == 0, "first record at word 0");
    check(l.add(128) == 2, "second record after the 8-byte status");
    check(l.add(8) == 34, "third record after the 128-byte summary");
    check(l.records == 3 && l.words == 36, "three records, 36 words");
    check(l.size[1] == 32 && l.offset[2] == 34, "sizes and offsets kept per record");
    l.reset();
    check(l.records == 0 && l.words == 0 && l.add(4) == 0, "reset starts a new cycle at word 0");
}

void layout_rejects_bad_records() {
    session::CyclePackLayout l;
    check(throws<std::invalid_argument>([&] { l.add(0); }), "empty record rejected");
    check(throws<std::invalid_argument>([&] { l.add(6); }), "non 4-byte multiple rejected");
    check(throws<std::length_error>([&] { l.add(4U * session::CyclePackLayout::max_words + 4U); }), "oversized record rejected");
    for (std::uint32_t i = 0; i < session::CyclePackLayout::max_records; ++i) l.add(4);
    check(throws<std::length_error>([&] { l.add(4); }), "65th record rejected");
    session::CyclePackLayout full;
    full.add(4U * (session::CyclePackLayout::max_words - 1U));
    check(full.add(4) == session::CyclePackLayout::max_words - 1U, "last word usable");
    check(throws<std::length_error>([&] { full.add(4); }), "words beyond the block rejected");
}

void layout_rejects_truncated_sizes() {
    if constexpr (sizeof(std::size_t) > 4) {
        session::CyclePackLayout l;
        l.add(8);
        constexpr std::uint64_t sixteen_gib = std::uint64_t{1} << 34;
        for (const auto bytes : {sixteen_gib, sixteen_gib + 8U,
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() & ~std::size_t{3})}) {
            check(throws<std::length_error>([&] { l.add(static_cast<std::size_t>(bytes)); }), "wrapped word count rejected");
            check(l.records == 1 && l.words == 2, "rejected record leaves layout intact");
        }
    }
}

void capacity_preflight_declines_oversized_cohorts() {
    session::CyclePackLayout predictors;
    check(predictors.try_add(8, 64), "64 predictor owners fit");
    predictors.reset();
    check(!predictors.try_add(8, 65), "65 owners decline packing for individual completion");
    check(predictors.records == 0 && predictors.words == 0, "decline reserves no partial pack");
    check(!predictors.try_add(8, std::numeric_limits<std::size_t>::max()), "owner count cannot overflow");

    session::CyclePackLayout target;
    for (unsigned k = 0; k < 31; ++k) {
        check(target.try_add(sizeof(q51_known_rows::Summary)), "member summary fits");
        check(target.try_add(sizeof(DeferredMoeFirstError)), "member first-error fits");
    }
    check(target.try_add(sizeof(q51_known_rows::Summary)) && target.try_add(sizeof(DeferredMoeFirstError)),
        "primary summary and first-error fill 64-record target pack");
    check(!target.try_add(sizeof(q51_known_rows::Summary)), "wide summary overhead declines target pack");
    target.reset();
    check(target.try_add(sizeof(q51_known_rows::Summary), 63), "63 summaries fit");
    check(!target.try_add(sizeof(q51_known_rows::Summary) + 4U), "byte capacity checked with a record slot remaining");
    check(target.records == 63 && target.try_add(sizeof(q51_known_rows::Summary)), "exact byte boundary remains usable");
}

// Host fake of the owner boundary: no claim about GPU gathering or stream joins.
struct CompletionOwner {
    session::CyclePackProof proof;
    std::array<std::uint32_t, 2> device_status{};
    bool failed = false, deferred = true, pending = true;
    unsigned fences = 0;
    void fail() noexcept { failed = true; deferred = pending = false; proof.invalidate(); }
    void submit() { proof.invalidate(); deferred = false; pending = true; }
    void complete() {
        std::array<std::uint32_t, 2> status{};
        if (!proof.consume(status)) { ++fences; status = device_status; }
        if (status[0] || status[1]) throw std::runtime_error("injected first-error");
        deferred = pending = false;
    }
};

void first_error_settles_survivor_before_reuse() {
    std::array<CompletionOwner, 2> owners;
    owners[0].proof.publish({1, 0});
    owners[1].proof.publish({0, 0});
    auto fail = [&](std::size_t k) noexcept { owners[k].fail(); };
    {
        session::CyclePackSettlement settlement(owners.size(), fail);
        settlement.take(0); settlement.take(1);
        check(throws<std::runtime_error>([&] { settlement.settle([&](std::size_t k) { owners[k].complete(); }); }),
            "first member error propagated after all settlements");
        check(owners[0].failed && !owners[1].failed && !owners[1].deferred && !owners[1].proof.ready(),
            "survivor healthy with old completion consumed");
    }
    check(!owners[1].failed, "settled survivor released from cleanup ownership");
    owners[1].submit(); owners[1].device_status = {0, 1};
    check(throws<std::runtime_error>([&] { owners[1].complete(); }), "survivor reuse checks new first-error");
    check(owners[1].fences == 1, "survivor reuse requires fresh fence");
}

void unfinished_settlement_fails_all_taken_members() {
    std::array<CompletionOwner, 3> owners;
    auto fail = [&](std::size_t k) noexcept { owners[k].fail(); };
    check(throws<std::runtime_error>([&] {
        session::CyclePackSettlement settlement(owners.size(), fail);
        settlement.take(0); settlement.take(1);
        owners[0].proof.publish({0, 0});
        throw std::runtime_error("injected finish/pack failure");
    }), "finish/pack exception preserved");
    check(owners[0].failed && owners[1].failed && !owners[2].failed && !owners[0].proof.ready(),
        "unwinding fails every taken owner, including closed cycles");
}

void new_submission_invalidates_carried_proof() {
    CompletionOwner owner;
    owner.proof.publish({0, 0});
    owner.submit(); owner.device_status = {1, 0};
    check(throws<std::runtime_error>([&] { owner.complete(); }), "carried proof cannot hide new error");
    check(owner.fences == 1 && !owner.proof.ready(), "new submission requires a new fence");
}

void lever_row_is_bitwise_and_off_by_default() {
    const auto& d = session::serve_cycle_pack_lever;
    check(d.env == "STRIX_SERVE_CYCLE_PACK", "lever switch name");
    check(d.exactness == lever::Exactness::bitwise, "lever is bitwise (scheduling only)");
    check(!d.default_on && d.regime.axis == lever::Axis::none, "lever default OFF until measured");
    check(!lever::select(d, lever::Inputs{0, 0, 4}, nullptr).on, "unset switch: OFF");
    check(lever::select(d, lever::Inputs{0, 0, 4}, "1").on, "STRIX_SERVE_CYCLE_PACK=1: ON");
    check(!lever::select(d, lever::Inputs{0, 0, 4}, "0").on, "STRIX_SERVE_CYCLE_PACK=0: OFF");
    const std::array table{d};
    check(!lever::valid(table), "experimental declaration must not enter a production lever table");
}

void switch_is_resolved_once() {
#if defined(_WIN32)
    check(_putenv_s("STRIX_SERVE_CYCLE_PACK", "1") == 0, "set pack switch");
#else
    check(setenv("STRIX_SERVE_CYCLE_PACK", "1", 1) == 0, "set pack switch");
#endif
    check(session::serve_cycle_pack_choice(2).on, "initial governed switch resolution");
#if defined(_WIN32)
    check(_putenv_s("STRIX_SERVE_CYCLE_PACK", "0") == 0, "change pack switch");
#else
    check(setenv("STRIX_SERVE_CYCLE_PACK", "0", 1) == 0, "change pack switch");
#endif
    check(session::serve_cycle_pack_choice(65).on, "hot path retains startup switch until restart");
}
}  // namespace

int main() {
#if defined(_MSC_VER)
    // A failed assertion or abort must fail the test, never open a dialog.
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif
    try {
        layout_places_records_back_to_back();
        layout_rejects_bad_records();
        layout_rejects_truncated_sizes();
        capacity_preflight_declines_oversized_cohorts();
        first_error_settles_survivor_before_reuse();
        unfinished_settlement_fails_all_taken_members();
        new_submission_invalidates_carried_proof();
        lever_row_is_bitwise_and_off_by_default();
        switch_is_resolved_once();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: unexpected exception: " << e.what() << '\n';
        return 1;
    }
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "cycle pack tests passed\n";
    return 0;
}
