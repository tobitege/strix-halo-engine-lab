// Host regressions for group admission; no device, model or HIP runtime required.
#include "strix/deepseek41_prefill.hpp"
#include "strix/serve/prefill_group.hpp"
#include "strix/ds41_dense_plan.hpp"
#include <string>
#include <iostream>
#include <limits>
#include <vector>

using namespace strix::serve;
namespace {
int failures = 0;
void check(bool ok, const char* name) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << name << '\n'; }
}
std::vector<GroupCandidate> prompts(std::uint32_t rows, std::uint32_t route = 0U) {
    return {{0U, rows, 0U, route}, {1U, rows, 1U, route}, {2U, rows, 2U, route}};
}
std::size_t planned(const std::vector<GroupCandidate>& c, PrefillGroupCapacity cap) {
    return plan_prefill_group(c, {}, {}, std::numeric_limits<double>::infinity(), cap).size();
}

void put_env(const char* name, const char* value) {  // value nullptr: unset
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
    strix::lever_environment_changed();
}
// The serve's group admission (apps/deepseek41_serve_main.cpp slot_prefill_options) asks the device prefill's own
// option resolver: every environment lever that group_decline_reason refuses must reach that decision, or the serve
// reserves group states (group_state_bytes in its memory room) that group_capacity() then never uses.
void resolved_option_tests() {
    using strix::deepseek41::DevicePrefill;
    const char* names[] = {"STRIX_DS41_FAST_PREFILL", "STRIX_DS41_FAST_ATTN", "STRIX_DS41_FAST_MOE", "STRIX_DS41_FAST_DENSE",
                           "STRIX_DS41_FAST_ENGRAM", "STRIX_DS41_FAST_SHARED", "STRIX_DS41_FAST_MOE_F16",
                           "STRIX_DS41_PREFILL_FP16_DENSE", "STRIX_DS41_PREFILL_BF16_ATTN", "STRIX_DS41_PREFILL_FP16_PROJ"};
    const auto clear = [&] { for (const char* n : names) put_env(n, nullptr); };
    const strix::deepseek41::PrefillOptions plain;
    const auto reason = [](const strix::deepseek41::PrefillOptions& o) {
        const char* r = DevicePrefill::group_decline_reason(o);
        return std::string(r ? r : "");
    };
    clear();
    check(DevicePrefill::group_options_supported(DevicePrefill::resolved_options(plain)), "resolved exact options group");
    put_env("STRIX_DS41_FAST_PREFILL", "1");
    {
        const auto r = DevicePrefill::resolved_options(plain);
        check(r.fast_attention && r.fast_attention_flash && r.fast_dense && r.fast_moe, "FAST bundle resolved");
        check(DevicePrefill::group_options_supported(r), "FAST bundle still groups");
    }
    clear();
    const struct { const char* env; const char* value; const char* reason; } declined[] = {
        {"STRIX_DS41_FAST_SHARED", "1", "shared-fast-group-unqualified"},
        {"STRIX_DS41_FAST_ENGRAM", "1", "engram-fast-flag-per-view"},
        {"STRIX_DS41_PREFILL_FP16_DENSE", "1", "fp16-solver-geometry-unpinned"},
    };
    for (const auto& d : declined) {
        put_env(d.env, d.value);
        // the request carries no lever: only the environment resolution sees it (the old serve copy did not)
        check(DevicePrefill::group_options_supported(plain), "request alone has no declining lever");
        check(reason(DevicePrefill::resolved_options(plain)) == d.reason, d.env);
        clear();
    }
    put_env("STRIX_DS41_FAST_MOE", "1");
    put_env("STRIX_DS41_FAST_MOE_F16", "1");
    check(reason(DevicePrefill::resolved_options(plain)) == "moe-f16-fast-group-unqualified", "FAST MoE FP16 from env declined");
    clear();
    strix::deepseek41::PrefillOptions req;
    req.fast_dense = true;
    req.fp16_min_rows = 77U;
    {
        const auto r = DevicePrefill::resolved_options(req);
        check(r.fast_dense && r.fp16_min_rows == 77U && !r.fast_moe, "request values kept");
    }
    put_env("STRIX_DS41_PREFILL_FP16_PROJ", "bogus");
    bool threw = false;
    try { (void)DevicePrefill::resolved_options(plain); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "malformed FP16_PROJ refused before construction");
    clear();
}

namespace pp = strix::prefill_plan;
namespace d4 = strix::ds41;
// rules of every site: dense entry everywhere, q_b / o_a from 256 rows, FAST dense from 128 rows when `fast`
std::vector<pp::DenseRule> rules(std::uint32_t ranks, bool fast) {
    std::vector<pp::DenseRule> r(std::size_t{ranks} * d4::kDenseSites);
    for (std::uint32_t k = 0; k < ranks; ++k)
        for (std::uint32_t s = 0; s < d4::kDenseSites; ++s) {
            auto& u = r[std::size_t{k} * d4::kDenseSites + s];
            u.fast = fast, u.fast_min_rows = 128U, u.k = 4096U;
            if (s == d4::kDenseQb || s == d4::kDenseOa) u.dense_min_rows = 256U;
            if (s == d4::kDenseOb) u.k_parts = 2U;
        }
    return r;
}
pp::DensePlan plan(std::uint32_t rows, bool fast, bool split = false, std::span<const d4::DenseConeRows> cone = {}, std::uint32_t layers = 2U) {
    d4::DenseRunGeometry g;
    g.rows = rows, g.window_rows = 8U, g.chunk_rows = 4096U, g.world = 2U, g.split = split, g.layers = layers, g.cone = cone;
    const auto r = rules(2U, fast);
    return d4::plan_dense(g, r);
}
bool mismatch_of(std::initializer_list<const pp::DensePlan*> members, std::uint32_t rank, std::uint32_t layer, std::uint32_t site,
                 pp::DenseRoute* route = nullptr) {
    bool mismatch = false;
    const std::vector<const pp::DensePlan*> v(members);
    const auto r = pp::common_route(v, rank, layer, site, mismatch);
    if (route) *route = r;
    return mismatch;
}
void dense_plan_tests() {
    bool multi_chunk_rejected = false;
    try { (void)plan(4097U, true); }
    catch (const std::invalid_argument&) { multi_chunk_rejected = true; }
    check(multi_chunk_rejected, "multi-chunk plan must not expose only its first chunk");
    check(plan(4096U, true).at(0, 0, d4::kDenseQb).rows == 4096U, "single chunk boundary admitted");
    // the spec's counterexample: two 80-row members (TP form) keep the exact dense kernels although they sum to 160 rows
    const auto a80 = plan(80U, true), b80 = plan(80U, true), s160 = plan(160U, true);
    check(a80.at(0, 0, d4::kDenseQa).route == pp::DenseRoute::dense_exact, "80 rows: q_a exact under FAST");
    check(s160.at(0, 0, d4::kDenseQa).route == pp::DenseRoute::dense_fast, "160 rows alone: q_a FAST");
    pp::DenseRoute common{};
    check(!mismatch_of({&a80, &b80}, 0, 0, d4::kDenseQa, &common) && common == pp::DenseRoute::dense_exact,
          "group of 80 + 80: the members' route (exact), not the summed rows' (FAST)");
    check(pp::signature(a80) == pp::signature(plan(40U, true)), "40 and 80 rows share the dense class under FAST");
    check(pp::signature(a80) != pp::signature(plan(200U, true)), "80 and 200 rows: different classes under FAST");
    check(mismatch_of({&a80, &s160}, 0, 0, d4::kDenseQa), "80 + 160: mismatch (exact vs FAST)");
    const auto c200 = plan(200U, true), d300 = plan(300U, true);
    check(!mismatch_of({&c200, &d300}, 0, 0, d4::kDenseKv, &common) && common == pp::DenseRoute::dense_fast, "200 + 300: FAST (sum also above)");
    check(mismatch_of({&c200, &d300}, 0, 0, d4::kDenseQb), "q_b: 200 batch vs 300 dense differ");
    check(d300.at(1, 1, d4::kDenseQb).route == pp::DenseRoute::dense_fast && c200.at(1, 1, d4::kDenseQb).route == pp::DenseRoute::batch_exact,
          "q_b routes from dense_min_rows then the FAST threshold");
    // exact profile: the dense route does not depend on rows, every short / long member shares the class
    check(pp::signature(plan(80U, false)) == pp::signature(plan(200U, false)), "exact: 80 and 200 rows same q_a/kv class");
    // row split: window j on rank j % 2; a one-window member has no rows on rank 1 (empty site never constrains)
    std::vector<d4::DenseConeRows> none;
    d4::DenseRunGeometry g;
    g.rows = 20U, g.window_rows = 8U, g.world = 2U, g.split = true, g.layers = 1U;
    check(d4::dense_local_rows(g, 0U, 0U) == 12U && d4::dense_local_rows(g, 1U, 0U) == 8U, "split local rows 12 / 8");
    check(d4::dense_local_rows(g, 0U, 8U) == 4U && d4::dense_local_rows(g, 1U, 8U) == 8U && d4::dense_local_rows(g, 1U, 24U) == 0U,
          "split local rows from r0");
    const auto w1 = plan(8U, true, true), w2 = plan(16U, true, true);
    check(w1.at(1, 0, d4::kDenseQa).route == pp::DenseRoute::none && w2.at(1, 0, d4::kDenseQa).route == pp::DenseRoute::dense_exact,
          "one window: no rows on rank 1");
    check(pp::signature(w1) == pp::signature(w2), "8 and 16 rows (split) share the class");
    // the cone: solo rows of a decoder layer come from its own cone rows, not from the prompt's rows
    const std::vector<d4::DenseConeRows> cone{{0U, 0U}, {64U, 96U}};
    const auto k200 = plan(200U, true, false, cone);
    check(k200.at(0, 0, d4::kDenseQa).route == pp::DenseRoute::dense_fast, "encoder layer: 200 rows FAST");
    check(k200.at(0, 1, d4::kDenseQa).rows == 104U && k200.at(0, 1, d4::kDenseQa).route == pp::DenseRoute::dense_exact,
          "cone layer: q_a from o0 = 96 -> 104 rows, exact");
    check(k200.at(0, 1, d4::kDenseKv).rows == 136U && k200.at(0, 1, d4::kDenseKv).route == pp::DenseRoute::dense_fast,
          "cone layer: kv from k0 = 64 -> 136 rows, FAST");
    // K split is part of the arithmetic: a different part count never shares a call
    auto e80 = a80;
    e80.at(0, 0, d4::kDenseOb).k_parts = 1U;
    check(mismatch_of({&a80, &e80}, 0, 0, d4::kDenseOb), "o_b part count differs: mismatch");
    check(pp::signature(a80) != pp::signature(e80), "o_b part count is in the signature");
    // the 128 / 256 / 512 / 1024 sums of short members never change a member's route
    for (const std::uint32_t rows : {16U, 32U, 63U, 127U}) {
        const auto m = plan(rows, true);
        for (std::uint32_t s = 0; s < d4::kDenseSites; ++s) {
            const auto r = m.at(0, 0, s).route;
            check(r == pp::DenseRoute::dense_exact || r == pp::DenseRoute::batch_exact, "short member: exact route at every site");
        }
    }
    pp::DenseClassTable t;
    check(t.intern(pp::signature(a80)) == 0U && t.intern(pp::signature(c200)) == 1U && t.intern(pp::signature(plan(64U, true))) == 0U,
          "class table interns exact signatures");
    pp::DenseClassTable other_table;
    check(other_table.intern(pp::signature(c200)) == 0U, "independent tables alias ordinal zero for different plans");
    const auto hash = [](const pp::DensePlan& p) { return pp::fingerprint(std::span(&p, 1U)); };
    check(hash(a80) != hash(c200), "canonical fingerprint detects aliased ordinals with different effective routes");
    check(hash(a80) == hash(plan(64U, true)), "same arithmetic fingerprint despite different row counts");
    check(hash(a80) != hash(e80), "fingerprint includes K parts");
    auto changed = a80;
    changed.at(0, 0, d4::kDenseQa).k_offset += 32U;
    check(hash(a80) != hash(changed), "fingerprint includes K offset");
    changed = a80;
    changed.at(0, 0, d4::kDenseQa).k += 32U;
    check(hash(a80) != hash(changed), "fingerprint includes K width");
    changed = a80;
    changed.at(0, 0, d4::kDenseQa).accumulate = true;
    check(hash(a80) != hash(changed), "fingerprint includes accumulation policy");
    check(pp::fingerprint(std::vector{a80, c200}) != pp::fingerprint(std::vector{c200, a80}),
          "fingerprint includes every member in order");
    (void)none;
}
void qualification_tests() {
    PrefillGroupCapacity cap;
    cap.members = 8U;
    cap.fast = true;
    PrefillGroupPolicy policy;
    auto q = qualify_fast(cap, policy, false);
    check(q.members == 0U && std::string(q.reason) == "pending-group-fast-identity", "FAST unqualified: solo with its reason");
    check(qualify_fast(cap, policy, true).members == 8U, "FAST qualified: groups");
    const auto env_auto = [](const char* k) { return std::string(k) == "STRIX_SERVE_PREFILL_GROUP" ? std::string("auto") : std::string(); };
    const auto p_auto = PrefillGroupPolicy::from_env(env_auto);
    check(p_auto.explicit_mode && qualify_fast(cap, p_auto, false).members == 8U, "FAST with explicit auto: groups");
    cap.fast = false;
    check(qualify_fast(cap, policy, false).members == 8U, "exact capacity unchanged");
    // dense route classes in the planner: a different class (bits 8+) ends the group, the solo class never groups
    PrefillGroupCapacity c2;
    c2.rejected_class_mask = 0x80000000U;
    std::vector<GroupCandidate> v{{0U, 20U, 0U, 1U << 8}, {1U, 20U, 1U, 1U << 8}, {2U, 20U, 2U, 2U << 8}, {3U, 20U, 3U, 1U << 8}};
    check(planned(v, c2) == 2U, "dense class change ends the group (no skipping)");
    std::vector<GroupCandidate> solo{{0U, 20U, 0U, 0x80000000U}, {1U, 20U, 1U, 0x80000000U}};
    check(planned(solo, c2) == 1U, "unrepresented dense class stays solo");
}
}
int main() {
    PrefillGroupCapacity cap;
    cap.rows = 32U;
    check(planned(prompts(20U), cap) == 1U, "32-row device: two 20-row prompts stay solo");
    cap.rows = 40U;
    check(planned(prompts(20U), cap) == 2U, "total row boundary admits two");
    cap = {};
    cap.members = 2U;
    check(planned(prompts(20U), cap) == 2U, "allocated member count");
    cap = {};
    cap.member_rows = 19U;
    check(planned(prompts(20U), cap) == 1U, "member storage bound");
    cap = {};
    cap.window_rows = 8U;
    cap.windows = 5U;
    check(planned(prompts(20U), cap) == 1U, "partial windows count per member");
    cap.windows = 6U;
    check(planned(prompts(20U), cap) == 2U, "six windows admit two partial members");
    cap = {};
    cap.window_rows = 8U;
    cap.split_class_mask = 1U;
    cap.local_rows = 23U;
    check(planned(prompts(20U, 1U), cap) == 1U, "owner restarts: 12+12 local rows exceed 23");
    cap.local_rows = 24U;
    check(planned(prompts(20U, 1U), cap) == 2U, "both rank bounds: 24 local rows fit");
    check(planned(prompts(20U), cap) == 1U, "unsplit members each consume all local rows");
    check(cap.local_count(17U, 1U) == 9U && cap.window_count(17U) == 3U,
          "partial terminal window retains rank zero ownership");
    cap = {};
    cap.rejected_class_mask = 4U;
    check(planned(prompts(20U, 4U), cap) == 1U, "matching dense classes still stay solo");
    cap.members = 0U;
    check(planned(prompts(20U), cap) == 1U, "unsupported options retain solo");

    using strix::deepseek41::DevicePrefill;
    strix::deepseek41::PrefillOptions options;
    check(DevicePrefill::group_options_supported(options), "exact options supported");
    // GRPFAST: the FAST levers and the BF16 / f16wmma attention group (per-member routes and mirrors); FP16 hipBLASLt stays solo
    options.bf16_attention = true;
    check(DevicePrefill::group_options_supported(options), "BF16 attention groups");
    options = {}; options.fp16_dense = true;
    check(!DevicePrefill::group_options_supported(options), "FP16 group declined");
    check(std::string(DevicePrefill::group_decline_reason(options)) == "fp16-solver-geometry-unpinned", "FP16 decline names its cause");
    options = {}; options.fast_dense = true;
    check(DevicePrefill::group_options_supported(options), "FAST dense groups");
    options = {}; options.fast_attention = true; options.fast_attention_flash = true;
    check(DevicePrefill::group_options_supported(options), "flash groups");
    options = {}; options.fast_attention = true;
    check(DevicePrefill::group_options_supported(options), "f16wmma attention groups");
    options = {}; options.fast_moe = true;
    check(DevicePrefill::group_options_supported(options), "FAST MoE groups");
    options = {}; options.fast_dense = true; options.fast_moe = true; options.fast_attention = true; options.fast_attention_flash = true;
    check(DevicePrefill::group_options_supported(options), "FAST profile groups");
    options = {}; options.trace = true;
    check(!DevicePrefill::group_options_supported(options), "trace group declined");
    options = {}; options.quality_head_rows = 2U;
    check(!DevicePrefill::group_options_supported(options), "multirow quality head declined");
    check(std::string(DevicePrefill::group_decline_reason(options)) == "diagnostic-output-not-segmented", "diagnostic decline names its cause");
    // FAST Engram: one overflow flag per view, a group view spans members (PROVISIONAL decline until a per-row flag)
    options = {}; options.fast_dense = true; options.fast_moe = true; options.fast_engram = true;
    check(!DevicePrefill::group_options_supported(options), "FAST Engram group declined");
    check(std::string(DevicePrefill::group_decline_reason(options)) == "engram-fast-flag-per-view", "FAST Engram decline names its cause");
    options = {}; options.fast_shared = true;
    check(!DevicePrefill::group_options_supported(options), "FAST shared expert group declined");
    check(std::string(DevicePrefill::group_decline_reason(options)) == "shared-fast-group-unqualified", "FAST shared decline names its cause");
    options = {}; options.fast_moe = true; options.fast_moe_f16 = true;
    check(!DevicePrefill::group_options_supported(options), "FAST MoE FP16 operands group declined");
    check(std::string(DevicePrefill::group_decline_reason(options)) == "moe-f16-fast-group-unqualified", "FAST MoE FP16 decline names its cause");
    resolved_option_tests();
    dense_plan_tests();
    qualification_tests();

    PrefillGroupPolicy policy;
    policy.short_rows = 0U;
    check(!policy.groups() && policy.storage_rows(4096U) == 0U, "zero threshold disables allocation");
    policy.short_rows = 1U;
    check(!policy.groups() && policy.storage_rows(4096U) == 0U, "exclusive threshold one has no members");
    policy.short_rows = 64U;
    check(policy.storage_rows(32U) == 32U, "member allocation bounded by context");
    check(policy.storage_rows(4096U) == 63U, "exclusive threshold is distinct from storage");
    check(policy.describe().find("prefill_group: auto") != std::string::npos &&
          policy.describe().find("device eligibility/capacity pending") != std::string::npos,
          "inspection reports requested policy without claiming device admission");
    bool invalid = false;
    try {
        (void)PrefillGroupPolicy::from_env([](const char* key) {
            return std::string(key) == "STRIX_SERVE_PREFILL_GROUP" ? std::string("bad") : std::string();
        });
    } catch (const std::invalid_argument&) { invalid = true; }
    check(invalid, "bad grouping policy rejected before inspection exits");

    struct Waiting { unsigned rows; std::uint32_t overtakes; };
    std::vector<Waiting> waiting{{100U, 0U}};
    for (unsigned i = 0; i < 7U; ++i) waiting.push_back({20U, 0U});
    const auto size = [](const Waiting& q) { return q.rows; };
    const auto skips = [](Waiting& q) -> std::uint32_t& { return q.overtakes; };
    const auto first = select_waiting(waiting, true, size, skips);
    check(first == 1U, "short-first head overtakes the long owner");
    std::vector<std::uint64_t> selected{1U, 2U, 3U, 4U, 5U, 6U, 7U};
    limit_prefill_overtakes(selected, waiting, skips);
    check(selected.size() == 7U && waiting[0].overtakes == 7U, "all seven real overtakes charged");
    for (std::size_t i = 1; i < waiting.size(); ++i) waiting[i].overtakes = 0U;
    check(select_waiting(waiting, true, size, skips) == 1U, "eighth short selection admitted");
    selected = {1U, 2U, 3U, 4U, 5U, 6U, 7U};
    limit_prefill_overtakes(selected, waiting, skips);
    check(selected.size() == 1U && waiting[0].overtakes == 8U, "companions cannot cross starvation bound");
    check(select_waiting(waiting, true, size, skips) == 0U, "protected long owner selected next");
    check(waiting[1].overtakes == 0U, "selected group owners are not charged");
    if (failures != 0) return 1;
    std::cout << "prefill group regressions passed\n";
}
