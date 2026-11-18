// Host-only tests for the shared prefill grouping rule (several short prompts in one device prefill pass).
#include "strix/serve/prefill_group.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix::serve;

namespace {
int failures = 0;
void check(bool value, const char* label) {
    if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
constexpr double kInf = std::numeric_limits<double>::infinity();
const PrefillGroupCost kDs41{170.0, 1.69};  // DS41 pair T_dev(P) = 0.17 s + 1.69 ms/token

std::vector<GroupCandidate> burst(const std::vector<std::uint32_t>& rows, std::uint32_t route = 0) {
    std::vector<GroupCandidate> c;
    for (std::size_t i = 0; i < rows.size(); ++i) c.push_back({100U + i, rows[i], 10U + 2U * i, route});
    return c;
}
std::vector<std::uint64_t> ids(std::initializer_list<std::uint64_t> v) { return v; }
}  // namespace

int main() {
    const PrefillGroupPolicy def;
    // The C8 burst of the DS41 receipt: eight prompts of 17-46 tokens, 191 rows, nothing decoding yet.
    const auto c8 = burst({20, 17, 23, 21, 46, 22, 19, 23});
    check(plan_prefill_group(c8, def, kDs41, kInf) == ids({100, 101, 102, 103, 104, 105, 106, 107}),
          "eight short prompts of different lengths form one group");
    // A single request: unchanged path (itself, alone).
    check(plan_prefill_group(burst({20}), def, kDs41, kInf) == ids({100}), "a single request runs alone");
    check(plan_prefill_group(std::vector<GroupCandidate>{}, def, kDs41, kInf).empty(), "no candidate: no pass");
    // The oldest prompt is always in, even when it alone exceeds the budget; it never waits for company.
    check(plan_prefill_group(c8, def, kDs41, 10.0) == ids({100}), "a tight gap budget leaves the oldest alone");
    // Gap budget: 170 + 1.69 x rows <= budget. 20+17 = 37 rows -> 232.5 ms; +23 = 60 -> 271.4 ms.
    check(plan_prefill_group(c8, def, kDs41, 240.0) == ids({100, 101}), "budget bounds the company");
    check(plan_prefill_group(c8, def, kDs41, std::numeric_limits<double>::quiet_NaN()) == ids({100}), "NaN budget: no company");
    {   // DAYAUDIT 3: the group transport has no embedding rows: an image prompt runs alone and ends the company
        auto img = burst({20, 17, 23, 21});
        img[2].image_rows = true;
        check(plan_prefill_group(img, def, kDs41, kInf) == ids({100, 101}), "an image candidate ends the company");
        img[0].image_rows = true;
        check(plan_prefill_group(img, def, kDs41, kInf) == ids({100}), "an image head runs alone");
        check(std::string(kGroupImageDecline) == "image rows not supported in groups", "image decline reason");
    }
    {   // DAYAUDIT 4: asymmetric ranks. Rank 0 holds groups of 8; rank 1 has the group off, or 2 members: rank 0 must
        // plan with the pair's minimum, never a group of 8 that rank 1 cannot hold.
        PrefillGroupCapacity cap0;
        cap0.members = 8; cap0.rows = 512; cap0.member_rows = 63; cap0.windows = 64; cap0.local_rows = 512; cap0.window_rows = 8;
        cap0.split_class_mask = 1U; cap0.rejected_class_mask = 4U;
        check(plan_prefill_group(c8, def, kDs41, kInf, cap0).size() == 8U, "rank 0 alone: a group of eight");
        PrefillGroupPolicy off = def; off.mode = PrefillGroupPolicy::Mode::off;
        PrefillGroupCapacity none = cap0; none.members = 0;
        const auto with_off = agree_prefill_group(def, cap0, prefill_group_words(off, none));
        check(!with_off.policy.groups() && with_off.capacity.members == 0U && std::string(with_off.capacity.reason) == "peer-capacity",
              "peer group off: no group");
        check(plan_prefill_group(c8, with_off.policy, kDs41, kInf, with_off.capacity) == ids({100}), "peer group off: rank 0 runs alone");
        PrefillGroupPolicy two = def; two.max_members = 2;
        PrefillGroupCapacity cap1 = cap0; cap1.members = 2; cap1.rows = 128;
        const auto with_two = agree_prefill_group(def, cap0, prefill_group_words(two, cap1));
        check(with_two.policy.max_members == 2U && with_two.capacity.members == 2U && with_two.capacity.rows == 128U &&
                  std::string(with_two.capacity.reason) == "pair-minimum", "peer of two members: the minimum");
        check(plan_prefill_group(c8, with_two.policy, kDs41, kInf, with_two.capacity) == ids({100, 101}), "peer of two: groups of two");
        // the agreement is symmetric in its result
        const auto from1 = agree_prefill_group(two, cap1, prefill_group_words(def, cap0));
        check(prefill_group_words(from1.policy, from1.capacity) == prefill_group_words(with_two.policy, with_two.capacity),
              "both ranks reach the same limits");
        PrefillGroupCapacity w16 = cap0; w16.window_rows = 16;
        check(agree_prefill_group(def, cap0, prefill_group_words(def, w16)).capacity.members == 0U, "window rows differ: no group");
        const auto equal = agree_prefill_group(def, cap0, prefill_group_words(def, cap0));
        check(prefill_group_words(equal.policy, equal.capacity) == prefill_group_words(def, cap0),
              "equal ranks: unchanged");
        check(plan_prefill_group(c8, equal.policy, kDs41, kInf, equal.capacity) == ids({100, 101, 102, 103, 104, 105, 106, 107}),
              "equal ranks: all eight candidates still group");
    }
    // Members cap and rows cap.
    PrefillGroupPolicy p3 = def; p3.max_members = 3;
    check(plan_prefill_group(c8, p3, kDs41, kInf) == ids({100, 101, 102}), "members cap");
    PrefillGroupPolicy r60 = def; r60.max_rows = 60;
    check(plan_prefill_group(c8, r60, kDs41, kInf) == ids({100, 101, 102}), "rows cap 60 = 20+17+23");
    PrefillGroupPolicy r59 = def; r59.max_rows = 59;
    check(plan_prefill_group(c8, r59, kDs41, kInf) == ids({100, 101}), "rows cap 59 stops before 23");
    // Only short prompts group: a long first prompt runs alone; a long later one ends the group (arrival order kept).
    check(plan_prefill_group(burst({64, 20, 17}), def, kDs41, kInf) == ids({100}), "a 64-row first prompt runs alone");
    check(plan_prefill_group(burst({20, 17, 300, 21}), def, kDs41, kInf) == ids({100, 101}), "a long prompt ends the group");
    check(plan_prefill_group(burst({63, 1}), def, kDs41, kInf) == ids({100, 101}), "63 rows is short");
    // R1: a different route class ends the group (never re-selected from the group's total).
    auto mixed = burst({20, 17, 23});
    mixed[2].route_class = 1;
    check(plan_prefill_group(mixed, def, kDs41, kInf) == ids({100, 101}), "route class differs: not in the group");
    mixed[0].route_class = 1; mixed[1].route_class = 1;
    check(plan_prefill_group(mixed, def, kDs41, kInf) == ids({100, 101, 102}), "same non-default class groups");
    // A failed / cancelled request is not offered again: the rest still group in arrival order.
    auto without = burst({20, 17, 23, 21});
    without.erase(without.begin() + 1);
    check(plan_prefill_group(without, def, kDs41, kInf) == ids({100, 102, 103}), "a removed member leaves the rest grouped");
    // Zero-row candidates never group.
    check(plan_prefill_group(burst({20, 0, 17}), def, kDs41, kInf) == ids({100}), "zero rows end the group");
    // Off: every prompt alone (the previous path).
    PrefillGroupPolicy off = def; off.mode = PrefillGroupPolicy::Mode::off;
    check(plan_prefill_group(c8, off, kDs41, kInf) == ids({100}), "off: oldest alone");
    // Order is enforced.
    bool threw = false;
    try { auto bad = burst({20, 17}); bad[1].arrival_seq = 1; (void)plan_prefill_group(bad, def, kDs41, kInf); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "candidates out of arrival order are refused");

    // Environment switch.
    std::map<std::string, std::string> e;
    auto env = [&](const char* k) { const auto it = e.find(k); return it == e.end() ? std::string{} : it->second; };
    check(PrefillGroupPolicy::from_env(env).mode == PrefillGroupPolicy::Mode::automatic, "unset: default auto");
    check(PrefillGroupPolicy::from_env(env, PrefillGroupPolicy::Mode::off).mode == PrefillGroupPolicy::Mode::off,
          "unset: family default off is kept");
    check(PrefillGroupPolicy::from_env(env, PrefillGroupPolicy::Mode::off, 48).short_rows == 48, "family short limit");
    e["STRIX_SERVE_PREFILL_GROUP"] = "auto";
    check(PrefillGroupPolicy::from_env(env, PrefillGroupPolicy::Mode::off).groups(), "auto overrides family off");
    e["STRIX_SERVE_PREFILL_GROUP"] = "0";
    check(!PrefillGroupPolicy::from_env(env).groups(), "0 = off");
    e["STRIX_SERVE_PREFILL_GROUP"] = "auto";
    e["STRIX_SERVE_PREFILL_GROUP_MEMBERS"] = "4";
    e["STRIX_SERVE_PREFILL_GROUP_ROWS"] = "128";
    const auto p = PrefillGroupPolicy::from_env(env);
    check(p.max_members == 4 && p.max_rows == 128, "members and rows from the environment");
    for (const char* bad : {"maybe", "2"}) {
        e["STRIX_SERVE_PREFILL_GROUP"] = bad;
        threw = false;
        try { (void)PrefillGroupPolicy::from_env(env); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "invalid switch value refused");
    }
    e["STRIX_SERVE_PREFILL_GROUP"] = "auto";
    e["STRIX_SERVE_PREFILL_GROUP_MEMBERS"] = "1";
    threw = false;
    try { (void)PrefillGroupPolicy::from_env(env); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "one member is refused");

    if (failures != 0) { std::cerr << failures << " prefill group test(s) failed\n"; return 1; }
    std::cout << "prefill group tests passed\n";
    return 0;
}
