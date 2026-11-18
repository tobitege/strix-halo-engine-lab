// Host tests for the resident test plan library: parser, arm grouping, run order, early stop, receipts.
#include "strix/serve/resident_plan.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

using namespace strix::serve::resident;
using strix::serve::Json;
using strix::serve::parse_json;

namespace {
int checks = 0, failures = 0;
}  // namespace

#define CHECK(c)                                                              \
    do {                                                                      \
        ++checks;                                                             \
        if (!(c)) {                                                           \
            ++failures;                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
        }                                                                     \
    } while (0)

namespace {

bool rejects(std::string_view text) {
    try {
        (void)parse_plan(text);
    } catch (const PlanError&) {
        return true;
    }
    return false;
}

std::string error_of(std::string_view text) {
    try {
        (void)parse_plan(text);
    } catch (const PlanError& e) {
        return e.what();
    }
    return {};
}

constexpr std::string_view kTwoArms = R"({
  "engine": "glm",
  "order": "abab",
  "repeats": 2,
  "warmup": 1,
  "steps": 64,
  "prompts": [{"name": "p1", "path": "prompts/p1.txt"}],
  "arms": [
    {"name": "A", "env": {}},
    {"name": "B", "env": {"STRIX_X": true}}
  ]
})";

// Same plan, keys reordered and whitespace changed.
constexpr std::string_view kTwoArmsReordered =
    R"({"arms":[{"env":{},"name":"A"},{"env":{"STRIX_X":true},"name":"B"}],"steps":64,"warmup":1,)"
    R"("prompts":[{"path":"prompts/p1.txt","name":"p1"}],"repeats":2,"order":"abab","engine":"glm"})";

std::string plan_text(std::string_view order, int repeats, int n_arms, bool cold, bool early) {
    std::string s = "{\"engine\":\"glm\",\"order\":\"" + std::string(order) + "\",\"repeats\":" +
                    std::to_string(repeats) + ",\"warmup\":1,\"steps\":32,";
    s += "\"prompts\":[{\"name\":\"p1\",\"path\":\"p1.txt\"}],";
    if (cold) s += "\"cold_control\":true,";
    if (early) s += "\"early_stop\":{\"enabled\":true,\"min_pairs\":3,\"band_pct\":1.0},";
    s += "\"arms\":[";
    for (int i = 0; i < n_arms; ++i) {
        if (i) s += ",";
        s += "{\"name\":\"" + std::string(1, static_cast<char>('A' + i)) + "\",\"env\":{\"STRIX_X\":" +
             std::to_string(i) + "}}";
    }
    return s + "]}";
}

// Records for the measured schedule: value(arm, pass) as decode_tok_s.
template <class F>
std::vector<RunRecord> fill(const Plan& plan, const std::vector<ArmGroup>& groups, F value) {
    std::vector<RunRecord> out;
    for (const RunSlot& s : schedule(plan, groups)) {
        RunRecord r{s, {}};
        r.result.decode_tok_s = value(s);
        out.push_back(r);
    }
    return out;
}

const LeverTable kRuntimeX = {{"STRIX_X", LeverClass::runtime}};

void test_parse() {
    const Plan p = parse_plan(kTwoArms);
    CHECK(p.engine == "glm");
    CHECK(p.order == Order::abab);
    CHECK(p.repeats == 2 && p.warmup == 1 && p.steps == 64);
    CHECK(p.prompts.size() == 1 && p.prompts[0].name == "p1");
    CHECK(p.arms.size() == 2);
    CHECK(p.arms[0].env.empty());
    CHECK(p.arms[1].env.size() == 1 && p.arms[1].env[0].first == "STRIX_X" && p.arms[1].env[0].second == "1");
    CHECK(p.arms[0].prompts.size() == 1 && p.arms[0].prompts[0] == 0);
    CHECK(p.arms[1].steps == 64);
    CHECK(!p.early_stop.enabled && p.early_stop.metric == "decode_tok_s");

    CHECK(rejects(R"({"engine":"glm","steps":8,"bogus":1,"prompts":[{"name":"p","path":"x"}],"arms":[{"name":"A"}]})"));
    CHECK(rejects(R"({"engine":"glm","steps":8,"order":"abc","prompts":[{"name":"p","path":"x"}],"arms":[{"name":"A"}]})"));
    CHECK(rejects(R"({"engine":"glm","steps":8,"prompts":[{"name":"p","path":"x"}],"arms":[{"name":"A"},{"name":"A"}]})"));
    CHECK(rejects(R"({"engine":"glm","steps":8,"prompts":[{"name":"p","path":"x"}],"arms":[{"name":"A","prompts":["q"]}]})"));
    CHECK(rejects(R"({"engine":"glm","prompts":[{"name":"p","path":"x"}],"arms":[{"name":"A"}]})"));
    CHECK(rejects(
        R"({"engine":"glm","steps":8,"prompts":[{"name":"p","path":"x"}],"arms":[{"name":"A","capture":["logits"]}]})"));
    CHECK(rejects("{"));
    CHECK(error_of("{").find("invalid JSON") != std::string::npos);
}

void test_batch_plan() {
    const auto text = [](std::string_view fields) {
        return R"({"engine":"fake","steps":4,"prompts":[{"name":"p","path":"p.u32"},
                   {"name":"q","path":"q.u32"}],"arms":[{"name":"a"}],)" + std::string(fields) + "}";
    };
    const Plan defaults = parse_plan(text(R"("concurrency":3)"));
    CHECK(defaults.concurrency == 3U);
    CHECK(!defaults.requests && !defaults.batch_mode && !defaults.mtp && !defaults.solo_control);
    const auto normalized = parse_json(defaults.canonical);
    CHECK(normalized.find("concurrency")->as_int() == 3);
    CHECK(!normalized.find("requests") && !normalized.find("batch_mode") && !normalized.find("mtp") &&
          !normalized.find("solo_control"));
    const Plan full = parse_plan(text(R"("concurrency":2,"requests":[{},
        {"prompt":0,"max_tokens":2,"seed":18446744073709551615}],"batch_mode":"continuous",
        "mtp":{"depth":8,"rows":64},"solo_control":false)"));
    CHECK(full.requests->at(0).prompt == 0 && full.requests->at(0).max_tokens == 4);
    CHECK(full.requests->at(1).seed == std::numeric_limits<std::uint64_t>::max());
    CHECK(full.batch_mode == "continuous" && full.mtp->depth == 8 && full.mtp->rows == 64);
    CHECK(full.solo_control && !*full.solo_control);
    CHECK(parse_plan(full.canonical).canonical == full.canonical);
    CHECK(parse_plan(full.canonical).hash == full.hash);
    const Plan cycling = parse_plan(text(R"("concurrency":3,"requests":[{},{},{}],"mtp":{})"));
    CHECK(cycling.requests->at(1).prompt == 1 && cycling.requests->at(2).prompt == 0);
    CHECK(cycling.mtp->depth == 0 && cycling.mtp->rows == 0);
    CHECK(parse_plan(text(R"("concurrency":1,"batch_mode":"lockstep","solo_control":true)" )).concurrency == 1U);
    CHECK(parse_plan(text(R"("concurrency":32)" )).concurrency == 32U);
    for (const auto bad : {
             R"("concurrency":0)", R"("concurrency":33)", R"("concurrency":1.5)", R"("concurrency":true)",
             R"("concurrency":2,"requests":[{}])", R"("concurrency":1,"requests":[{"prompt":2}])",
             R"("concurrency":1,"requests":[{"prompt":-1}])",
             R"("concurrency":1,"requests":[{"max_tokens":5}])",
             R"("concurrency":1,"requests":[{"max_tokens":0}])",
             R"("concurrency":1,"requests":[{"typo":1}])",
             R"("concurrency":1,"requests":[{"seed":18446744073709551616}])",
             R"("concurrency":1,"requests":[{"seed":-1}])",
             R"("concurrency":1,"requests":[{"seed":1.5}])",
             R"("concurrency":1,"mtp":{"typo":1})", R"("concurrency":1,"mtp":{"depth":9})",
             R"("concurrency":1,"mtp":{"rows":65})", R"("concurrency":1,"mtp":{"depth":-1})",
             R"("concurrency":1,"batch_mode":"other")", R"("concurrency":1,"solo_control":0)",
             R"("concurrency":1,"requests":[false])", R"("concurrency":1,"mtp":false)",
             R"("requests":[{}])"})
        CHECK(rejects(text(bad)));
    const Plan legacy = parse_plan(kTwoArms);
    const Json legacy_json = parse_json(legacy.canonical);
    for (const auto key : {"concurrency", "requests", "batch_mode", "mtp", "solo_control"})
        CHECK(!legacy_json.find(key));
}

void test_hash() {
    const Plan a = parse_plan(kTwoArms);
    const Plan b = parse_plan(kTwoArmsReordered);
    CHECK(a.canonical == b.canonical);
    CHECK(a.hash == b.hash);
    std::string changed(kTwoArms);
    changed.replace(changed.find("\"steps\": 64"), 11, "\"steps\": 65");
    CHECK(parse_plan(changed).hash != a.hash);
}

void test_grouping() {
    const Plan p = parse_plan(R"({"engine":"glm","steps":8,"prompts":[{"name":"p","path":"x"}],"arms":[
        {"name":"A"},{"name":"B","env":{"STRIX_X":1}},{"name":"C","env":{"STRIX_Y":1}}]})");
    const std::vector<ArmGroup> g = group_arms(p, kRuntimeX);
    CHECK(g.size() == 2);
    CHECK(g[0].arms == (std::vector<std::size_t>{0, 1}));
    CHECK(g[1].arms == (std::vector<std::size_t>{2}));
    CHECK(group_of(g, 1) == 0 && group_of(g, 2) == 1);
    CHECK(group_of(g, 7) == g.size());
    CHECK(lever_class(kRuntimeX, "STRIX_UNKNOWN") == LeverClass::reload);
}

void test_schedule() {
    {
        const Plan p = parse_plan(kTwoArms);
        const std::vector<RunSlot> s = schedule(p, group_arms(p, kRuntimeX));
        CHECK(s.size() == 8);
        const std::size_t arms[] = {0, 0, 1, 1, 0, 0, 1, 1};
        const bool warm[] = {true, false, true, false, true, false, true, false};
        for (std::size_t i = 0; i < s.size() && i < 8; ++i) {
            CHECK(s[i].seq == i);
            CHECK(s[i].arm == arms[i]);
            CHECK(s[i].warmup == warm[i]);
            CHECK(!s[i].cold);
        }
        CHECK(s[4].pass == 1 && s[7].pass == 1);
    }
    {
        const Plan p = parse_plan(plan_text("baba", 2, 2, false, false));
        const std::vector<RunSlot> s = schedule(p, group_arms(p, kRuntimeX));
        CHECK(!s.empty() && s[0].arm == 1);
    }
    {
        const Plan p = parse_plan(plan_text("abab", 3, 3, false, false));
        CHECK(pass_order({0, 1, 2}, Order::abab, 1) == (std::vector<std::size_t>{1, 2, 0}));
        const std::vector<RunSlot> s = schedule(p, group_arms(p, kRuntimeX));
        std::size_t first_pass1 = s.size();
        for (std::size_t i = 0; i < s.size(); ++i)
            if (s[i].pass == 1) {
                first_pass1 = i;
                break;
            }
        CHECK(first_pass1 < s.size() && s[first_pass1].arm == 1);
    }
    {
        const Plan p = parse_plan(plan_text("sequential", 2, 2, false, false));
        const std::vector<RunSlot> s = schedule(p, group_arms(p, kRuntimeX));
        CHECK(s.size() == 6);
        if (s.size() == 6) {
            CHECK(s[0].warmup && s[0].arm == 0);
            CHECK(!s[1].warmup && !s[2].warmup && s[2].arm == 0 && s[2].pass == 1);
            CHECK(s[3].warmup && s[3].arm == 1);
            CHECK(!s[5].warmup && s[5].arm == 1);
        }
    }
    {
        const Plan p = parse_plan(plan_text("abab", 2, 2, true, false));
        const std::vector<RunSlot> s = schedule(p, group_arms(p, kRuntimeX));
        CHECK(s.size() == 10);
        if (s.size() == 10) {
            CHECK(s[8].cold && s[8].warmup && s[8].arm == 0);
            CHECK(s[9].cold && !s[9].warmup && s[9].arm == 0);
            CHECK(!s[7].cold);
        }
    }
}

void test_compare() {
    const Plan p = parse_plan(plan_text("abab", 4, 2, false, true));
    const std::vector<ArmGroup> g = group_arms(p, kRuntimeX);
    const auto verdict = [&](const double* rel) {
        const std::vector<RunRecord> r =
            fill(p, g, [&](const RunSlot& s) { return s.arm == 0 ? 100.0 : 100.0 * (1.0 + rel[s.pass]); });
        return compare_pair(p, g, 0, 1, r);
    };
    const double sep[] = {0.1, 0.1, 0.1, 0.1};
    const double eq[] = {0.001, -0.001, 0.001, -0.001};
    const double noisy[] = {0.05, -0.04, 0.06, -0.05};
    const Comparison cs = verdict(sep);
    CHECK(cs.n_pairs == 4);
    CHECK(std::fabs(cs.mean_rel - 0.1) < 1e-9);
    CHECK(cs.verdict == "separated");
    CHECK(!cs.cross_reload);
    CHECK(verdict(eq).verdict == "equivalent");
    CHECK(verdict(noisy).verdict == "open");

    // Two passes only: fewer pairs than min_pairs.
    const Plan p2 = parse_plan(plan_text("abab", 2, 2, false, true));
    const std::vector<ArmGroup> g2 = group_arms(p2, kRuntimeX);
    const Comparison ci = compare_pair(p2, g2, 0, 1, fill(p2, g2, [](const RunSlot& s) {
        return s.arm == 0 ? 100.0 : 110.0;
    }));
    CHECK(ci.n_pairs == 2 && ci.verdict == "insufficient");
    const Comparison none = compare_pair(p2, g2, 0, 1, {});
    CHECK(none.n_pairs == 0 && none.verdict == "insufficient" && std::isnan(none.mean_rel));

    // Errors and warmups are excluded from pairs.
    std::vector<RunRecord> r = fill(p, g, [](const RunSlot& s) { return s.arm == 0 ? 100.0 : 110.0; });
    for (RunRecord& rec : r)
        if (rec.slot.arm == 1 && rec.slot.pass == 0 && !rec.slot.warmup) rec.result.error = "oom";
    CHECK(compare_pair(p, g, 0, 1, r).n_pairs == 3);

    CHECK(std::isinf(t95(0)) && std::fabs(t95(1) - 12.706) < 1e-12 && std::fabs(t95(3) - 3.182) < 1e-12);
}

Comparison with_verdict(const char* v) {
    Comparison c;
    c.verdict = v;
    return c;
}

void test_early_stop() {
    const Plan off = parse_plan(plan_text("abab", 4, 3, false, false));
    CHECK(!decide_early_stop(off, {with_verdict("separated")}).stop);
    const Plan on = parse_plan(plan_text("abab", 4, 3, false, true));
    CHECK(!decide_early_stop(on, {}).stop);
    const StopDecision sep = decide_early_stop(on, {with_verdict("separated"), with_verdict("separated")});
    CHECK(sep.stop && sep.reason == "separated");
    const StopDecision eq = decide_early_stop(on, {with_verdict("equivalent")});
    CHECK(eq.stop && eq.reason == "equivalent");
    const StopDecision mixed = decide_early_stop(on, {with_verdict("separated"), with_verdict("equivalent")});
    CHECK(mixed.stop && mixed.reason == "mixed");
    const StopDecision open = decide_early_stop(on, {with_verdict("separated"), with_verdict("open")});
    CHECK(!open.stop && open.reason.empty());
    CHECK(!decide_early_stop(on, {with_verdict("insufficient")}).stop);
}

void test_receipts() {
    const Plan p = parse_plan(R"({"engine":"glm","steps":8,"prompts":[{"name":"p1","path":"x"}],"arms":[
        {"name":"A","capture":["timing","tokens"]},{"name":"B"}]})");
    RunSlot slot;
    slot.seq = 7;
    slot.arm = 0;
    slot.prompt = 0;
    CHECK(receipt_name(p, slot) == "run-00007-A-p1.json");
    slot.seq = 123456;
    CHECK(receipt_name(p, slot) == "run-123456-A-p1.json");

    RunRecord rec{slot, {}};
    rec.result.tokens = {1, 2, 3};
    rec.result.decode_tok_s = 42.5;
    const Json a = receipt_json(p, rec);
    CHECK(a.find("tokens") != nullptr && a.find("tokens")->size() == 3);
    CHECK(a.find("n_tokens") != nullptr && a.find("n_tokens")->as_int() == 3);
    CHECK(a.find("decode_tok_s") != nullptr && a.find("decode_tok_s")->as_double() == 42.5);
    CHECK(a.find("ttft_ms") != nullptr && a.find("ttft_ms")->is_null());
    CHECK(a.find("error") == nullptr);
    CHECK(a.find("plan_hash") != nullptr && a.find("plan_hash")->as_string() == hex64(p.hash));

    rec.slot.arm = 1;
    rec.result.error = "timeout";
    const Json b = receipt_json(p, rec);
    CHECK(b.find("tokens") == nullptr);
    CHECK(b.find("tokens_fnv") != nullptr && b.find("tokens_fnv")->as_string() == a.find("tokens_fnv")->as_string());
    CHECK(b.find("error") != nullptr && b.find("error")->as_string() == "timeout");
    CHECK(b.find("arm_digest")->as_string() != a.find("arm_digest")->as_string());
}

void test_summary() {
    {
        const Plan p = parse_plan(R"({"engine":"glm","steps":8,"repeats":4,"prompts":[{"name":"p","path":"x"}],
            "arms":[{"name":"A"},{"name":"B","env":{"STRIX_X":1}},{"name":"C","env":{"STRIX_Y":1}}]})");
        const std::vector<ArmGroup> g = group_arms(p, kRuntimeX);
        const std::vector<RunRecord> r = fill(p, g, [](const RunSlot& s) { return 100.0 + 10.0 * static_cast<double>(s.arm); });
        const std::vector<Comparison> c = compare_all(p, g, r);
        CHECK(c.size() == 2);
        // Arms in different reload groups are never paired across the load boundary silently.
        const Json sum = summary_json(p, g, r, c, decide_early_stop(p, c));
        const Json* cmp = sum.find("comparisons");
        CHECK(cmp != nullptr && cmp->size() == 2);
        if (cmp != nullptr && cmp->size() == 2) {
            CHECK(!cmp->as_array()[0].find("cross_reload")->as_bool());
            CHECK(cmp->as_array()[1].find("cross_reload")->as_bool());
        }
        CHECK(sum.find("groups")->as_int() == 2);
        CHECK(sum.find("errors")->as_int() == 0);
        CHECK(sum.find("cold_control") == nullptr);
        const Json& arm_c = sum.find("arms")->as_array()[2];
        CHECK(arm_c.find("group")->as_int() == 1);
        CHECK(arm_c.find("n_runs")->as_int() == 4);
        CHECK(arm_c.find("mean")->find("decode_tok_s")->as_double() == 120.0);
    }
    {
        const Plan p = parse_plan(plan_text("abab", 4, 2, true, false));
        const std::vector<ArmGroup> g = group_arms(p, kRuntimeX);
        const double resident[] = {100.0, 101.0, 99.0, 100.0};
        const auto run = [&](double cold_value) {
            const std::vector<RunRecord> r = fill(
                p, g, [&](const RunSlot& s) { return s.cold ? cold_value : (s.arm == 0 ? resident[s.pass] : 90.0); });
            const std::vector<Comparison> c = compare_all(p, g, r);
            return summary_json(p, g, r, c, decide_early_stop(p, c));
        };
        const Json far = run(150.0);
        const Json* cc = far.find("cold_control");
        CHECK(cc != nullptr);
        if (cc != nullptr) {
            CHECK(cc->find("arm")->as_string() == "A");
            CHECK(cc->find("cold_mean")->as_double() == 150.0);
            CHECK(cc->find("resident_mean")->as_double() == 100.0);
            CHECK(cc->find("resident_drift")->as_bool());
        }
        const Json near = run(100.0);
        CHECK(near.find("cold_control") != nullptr && !near.find("cold_control")->find("resident_drift")->as_bool());
        // Cold runs never enter the resident mean of arm 0.
        CHECK(far.find("arms")->as_array()[0].find("mean")->find("decode_tok_s")->as_double() == 100.0);
    }
}

void test_schedule_bound() {
    auto p = parse_plan(plan_text("abab", 10000, 20, false, false));
    p.warmup = 1000;
    bool refused = false;
    try { (void)schedule(p, group_arms(p, kRuntimeX)); }
    catch (const PlanError&) { refused = true; }
    CHECK(refused);
    p.repeats = std::numeric_limits<std::uint32_t>::max();
    refused = false;
    try { (void)schedule(p, group_arms(p, kRuntimeX)); }
    catch (const PlanError&) { refused = true; }
    CHECK(refused);
    CHECK(rejects(std::string(kMaxPlanBytes + 1, ' ')));
}

void test_cold_warm_statistics() {
    auto p = parse_plan(plan_text("sequential", 3, 2, false, true));
    p.warmup = 0;
    p.early_stop.metric = "ttft_ms";
    const auto groups = group_arms(p, kRuntimeX);
    std::vector<RunRecord> records;
    for (const auto& s : schedule(p, groups)) {
        RunRecord r{s, {}};
        r.result.cold_run = s.pass == 0;
        r.result.ttft_ms = s.pass == 0 ? (s.arm == 0 ? 10.0 : 100.0) : 1.0;
        records.push_back(r);
    }
    const auto warm = compare_all(p, groups, records);
    const auto cold = compare_all(p, groups, records, true);
    CHECK(warm[0].n_pairs == 2 && warm[0].mean_rel == 0);
    CHECK(cold[0].n_pairs == 1 && cold[0].mean_rel == 9);
    CHECK(!decide_early_stop(p, warm).stop); // two warm pairs, min_pairs is three
    p.early_stop.min_pairs = 2;
    CHECK(decide_early_stop(p, compare_all(p, groups, records)).stop);
    auto cold_decided = cold;
    cold_decided[0].verdict = "separated";
    CHECK(!decide_early_stop(p, cold_decided).stop);
    const auto summary = summary_json(p, groups, records, warm, {});
    const auto& a = summary.find("arms")->as_array()[0];
    CHECK(a.find("n_runs")->as_int() == 2);
    CHECK(a.find("mean")->find("ttft_ms")->as_double() == 1);
    CHECK(a.find("cold_n_runs")->as_int() == 1);
    CHECK(a.find("cold_mean")->find("ttft_ms")->as_double() == 10);
    CHECK(summary.find("cold_comparisons")->as_array()[0].find("n_pairs")->as_int() == 1);
    records.back().result.cold_run = true; // never pair a warm baseline with a cold arm
    CHECK(compare_all(p, groups, records)[0].n_pairs == 1);
}

}  // namespace

int main() {
    test_schedule_bound();
    test_cold_warm_statistics();
    test_parse();
    test_batch_plan();
    test_hash();
    test_grouping();
    test_schedule();
    test_compare();
    test_early_stop();
    test_receipts();
    test_summary();
    std::printf("resident_plan_cpu_tests: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
