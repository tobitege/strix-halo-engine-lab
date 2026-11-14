// Typed lever declaration (include/strix/lever.hpp): table validation, the decision rule and conflicts.
#include "strix/lever.hpp"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

#if defined(_MSC_VER)
#include <crtdbg.h>
#include <cstdlib>
#endif

namespace {
using namespace strix::lever;
using std::string_view;

void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }

constexpr std::array<string_view, 1> kWideConflicts{"tall_rows"};
constexpr Decl kTable[] = {
    {.name = "ring", .env = "STRIX_TEST_LEVER_RING", .exactness = Exactness::bitwise, .default_on = true,
     .source = "exact; faster in every regime"},
    {.name = "tall_rows", .env = "STRIX_TEST_LEVER_TALL", .exactness = Exactness::bitwise,
     .regime = {.axis = Axis::rows, .crossover = 512, .on_from = true}, .source = "exact; wins from 512 rows"},
    {.name = "wide_rows", .env = "STRIX_TEST_LEVER_WIDE", .exactness = Exactness::bitwise,
     .regime = {.axis = Axis::concurrency, .crossover = 4, .on_below = true}, .conflicts = kWideConflicts,
     .source = "exact; wins below 4 streams"},
    {.name = "half_route", .env = "STRIX_TEST_LEVER_HALF", .exactness = Exactness::bit_changing, .default_on = true,
     .source = "changes bits; behind its gate"},
};
static_assert(valid(kTable));
static_assert(automatic(kTable[0], {}, false).on);
static_assert(!automatic(kTable[1], {.rows = 511}, false).on && automatic(kTable[1], {.rows = 512}, false).on);
static_assert(automatic(kTable[3], {}, false).origin == Origin::gate_closed && automatic(kTable[3], {}, true).on);

constexpr bool rejects(const Decl& row) {
    const Decl one[] = {row};
    return !valid(one);
}
constexpr std::array<string_view, 1> kUnknown{"missing"};
static_assert(rejects({.name = "", .env = "STRIX_X", .default_on = true}));
static_assert(rejects({.name = "x", .env = "OTHER_X", .default_on = true}));
static_assert(rejects({.name = "x", .env = "STRIX_", .default_on = true}));
static_assert(rejects({.name = "x", .env = "STRIX_X"}));  // bitwise and off everywhere
static_assert(rejects({.name = "x", .env = "STRIX_X", .regime = {.axis = Axis::rows}}));
static_assert(rejects({.name = "x", .env = "STRIX_X", .regime = {.axis = Axis::rows, .crossover = 8}}));
static_assert(rejects({.name = "x", .env = "STRIX_X",
                       .regime = {.axis = Axis::rows, .crossover = 8, .on_below = true, .on_from = true}}));
static_assert(rejects({.name = "x", .env = "STRIX_X", .default_on = true, .regime = {.crossover = 8}}));
static_assert(rejects({.name = "x", .env = "STRIX_X", .default_on = true, .conflicts = kUnknown}));
static_assert(!rejects({.name = "x", .env = "STRIX_X", .exactness = Exactness::bit_changing}));

constexpr Decl kDuplicate[] = {
    {.name = "a", .env = "STRIX_A", .default_on = true},
    {.name = "b", .env = "STRIX_A", .default_on = true},
};
static_assert(!valid(kDuplicate));

bool throws(const Decl& row, const char* value) {
    try {
        (void)select(row, {}, value);
    } catch (const std::invalid_argument& e) {
        return std::string(e.what()).find(std::string(row.env)) != std::string::npos;
    }
    return false;
}
}  // namespace

int main() try {
#if defined(_MSC_VER)
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    const Decl& ring = kTable[0];
    const Decl& tall = kTable[1];
    const Decl& wide = kTable[2];
    const Decl& half = kTable[3];

    // Unset and empty are automatic; an explicit switch wins over the default, the regime and the gate.
    require(select(ring, {}, nullptr).on && select(ring, {}, "").on, "default-on lever is on when unset");
    require(select(ring, {}, nullptr).origin == Origin::builtin, "default origin");
    const Choice off = select(ring, {}, "0");
    require(!off.on && off.origin == Origin::explicit_switch, "explicit 0 switches a default-on lever off");
    require(select(tall, {.rows = 16}, "1").on, "explicit 1 overrides the regime rule");
    require(!select(tall, {.rows = 4096}, "0").on, "explicit 0 overrides the regime rule");
    require(select(half, {}, "1").on, "explicit 1 selects a gated lever for a measurement");
    require(throws(ring, "on") && throws(ring, "2") && throws(ring, "01"), "malformed value names the switch");

    // Regime rule: one crossover on one axis; an unknown axis value keeps the reference arm.
    require(!select(tall, {.rows = 511}, nullptr).on && select(tall, {.rows = 512}, nullptr).on, "rows crossover");
    require(select(tall, {.rows = 512}, nullptr).origin == Origin::regime, "regime origin");
    require(!select(tall, {}, nullptr).on, "unknown rows keeps the reference arm");
    require(select(wide, {.concurrency = 3}, nullptr).on && !select(wide, {.concurrency = 4}, nullptr).on,
            "concurrency crossover");
    require(!select(wide, {.rows = 9999}, nullptr).on, "a rule reads only its own axis");

    // Bit-changing levers stay off until their gate opens.
    require(!select(half, {}, nullptr).on && select(half, {}, nullptr).origin == Origin::gate_closed, "gate closed");
    require(select(half, {}, nullptr, true).on, "gate open applies the declared default");
    require(!select(half, {}, "0", true).on, "explicit 0 wins over an open gate");

    // Declared conflicts are reported, in either declaration direction.
    const Inputs both{.rows = 1024, .concurrency = 2};
    const auto on = [&](const Decl& row) { return select(row, both, nullptr).on; };
    const Conflict c = conflict(kTable, on);
    require(static_cast<bool>(c) && c.first == &wide && c.second == &tall, "conflict found");
    bool refused = false;
    try {
        require_consistent(kTable, on);
    } catch (const std::invalid_argument& e) {
        const std::string what = e.what();
        refused = what.find("STRIX_TEST_LEVER_WIDE") != std::string::npos &&
                  what.find("STRIX_TEST_LEVER_TALL") != std::string::npos;
    }
    require(refused, "conflict names both switches");
    const Inputs apart{.rows = 1024, .concurrency = 8};
    require(!conflict(kTable, [&](const Decl& row) { return select(row, apart, nullptr).on; }), "no conflict");
    require_consistent(kTable, [&](const Decl& row) { return select(row, apart, nullptr).on; });

    // The environment reader records the decision for --print-effective-config.
    const Choice recorded = select(tall, {.rows = 2048});
    require(recorded.on, "environment reader applies the rule when the switch is unset");
    bool listed = false;
    for (const auto& entry : strix::envcfg::registered())
        listed = listed || (entry.name == "STRIX_TEST_LEVER_TALL" && entry.value == "1" && !entry.set);
    require(listed, "decision is recorded in the switch registry");

    require(find(kTable, "half_route") == &half && find(kTable, "nope") == nullptr, "find");
    require(string_view(origin_name(Origin::gate_closed)) == "gate-closed", "origin name");
    std::puts("lever tests: ok");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "lever tests: FAIL: %s\n", e.what());
    return 1;
}
