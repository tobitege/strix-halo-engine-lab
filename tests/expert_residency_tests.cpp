#include "strix/residency/expert_residency.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace strix::residency;

namespace {
int failures = 0;
void require(bool ok, const char* what, int line) {
    if (!ok) {
        std::cerr << "FAIL line " << line << ": " << what << "\n";
        ++failures;
    }
}
#define REQUIRE(c) require((c), #c, __LINE__)

constexpr std::uint64_t GiB = 1ull << 30;
constexpr std::uint64_t kExpert = 3'070'000;

ModelBytes qwen_like() {
    ModelBytes m;
    m.core = 4'800'000'000ull;
    m.layers = 48;
    m.experts = 512;
    m.top_k = 10;
    m.expert_bytes.assign(48, kExpert);
    m.host_tables = 0;
    return m;
}
SessionBytes sessions(std::uint32_t pos = 32768) {
    SessionBytes s;
    s.per_position = 24576;
    s.fixed = 64ull << 20;
    s.scratch = 1 * GiB;
    s.margin = 1 * GiB;
    s.sessions = 1;
    s.positions = pos;
    return s;
}
HostBytes big_host() { return {128 * GiB}; }

void set_env(const char* k, const char* v) {
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}

bool has(const std::string& s, const char* k) { return s.find(k) != std::string::npos; }

void test_resolve() {
    const ModelBytes m = qwen_like();
    const std::uint64_t per_layer_set = 48 * kExpert;
    {
        Decision d = resolve({}, m, {96 * GiB, 96 * GiB}, sessions(), big_host());
        REQUIRE(d.ok && d.mode == Mode::full);
        REQUIRE(d.slots_per_layer == 512 && d.host_expert_bytes == 0);
        REQUIRE(has(d.reason, "fits:"));
    }
    for (std::uint64_t g : {8ull, 12ull, 24ull}) {
        Options o;
        o.vram_budget = g * GiB;
        Decision d = resolve(o, m, {96 * GiB, 96 * GiB}, sessions(), big_host());
        const std::uint64_t sess = (24576ull * 32768 + (64ull << 20));
        const std::uint64_t used = m.core + 2 * GiB + sess + per_layer_set;
        REQUIRE(d.ok && d.mode == Mode::tiered && d.positions == 32768);
        REQUIRE(d.budget == g * GiB);
        const std::uint64_t pool = g * GiB - used;
        REQUIRE(d.expert_pool == pool);
        REQUIRE(d.slots_per_layer == pool / per_layer_set);
        REQUIRE(d.host_expert_bytes == per_layer_set * 512);
        REQUIRE(d.slots_per_layer > 0 && d.slots_per_layer < 512);
    }
    {  // free < total wins over the budget
        Options o;
        o.vram_budget = 24 * GiB;
        Decision d = resolve(o, m, {96 * GiB, 10 * GiB}, sessions(), big_host());
        REQUIRE(d.budget == 10 * GiB);
    }
    {  // explicit full on 8 GB
        Options o;
        o.mode = Mode::full;
        o.vram_budget = 8 * GiB;
        Decision d = resolve(o, m, {96 * GiB, 96 * GiB}, sessions(), big_host());
        REQUIRE(!d.ok && d.mode == Mode::full && has(d.error, "short"));
    }
    {  // explicit tiered where full would fit
        Options o;
        o.mode = Mode::tiered;
        Decision d = resolve(o, m, {96 * GiB, 96 * GiB}, sessions(), big_host());
        REQUIRE(d.ok && d.mode == Mode::tiered && d.slots_per_layer == 512);
    }
    {  // context reduction: 262144 positions at 8 GiB cannot fit, reduces in multiples of 1024
        Options o;
        o.vram_budget = 8 * GiB;
        Decision d = resolve(o, m, {96 * GiB, 96 * GiB}, sessions(262144), big_host());
        REQUIRE(d.ok && d.mode == Mode::tiered);
        REQUIRE(d.positions_requested == 262144 && d.positions < 262144 && d.positions % 1024 == 0 && d.positions >= 1024);
        REQUIRE(has(d.reason, "positions reduced"));
        REQUIRE(d.slots_per_layer >= 1);
        // the next 1024 step up would not have left one resident slot per layer
        const std::uint64_t used = m.core + 2 * GiB + (24576ull * (d.positions + 1024) + (64ull << 20)) + 2 * per_layer_set;
        REQUIRE(used > 8 * GiB);
    }
    {  // not even 1024 positions fit
        Options o;
        o.vram_budget = 6 * GiB;
        Decision d = resolve(o, m, {96 * GiB, 96 * GiB}, sessions(), big_host());
        REQUIRE(!d.ok && d.mode == Mode::tiered && !d.error.empty() && d.positions == 1024);
    }
    {  // host RAM too small
        Options o;
        o.vram_budget = 12 * GiB;
        Decision d = resolve(o, m, {96 * GiB, 96 * GiB}, sessions(), {60 * GiB});
        REQUIRE(!d.ok && has(d.error, "host RAM"));
        ModelBytes m2 = m;
        m2.host_tables = 108 * GiB;  // gate = non-resident experts + host tables + 4 GiB > 110 GiB available
        Decision d2 = resolve(o, m2, {96 * GiB, 96 * GiB}, sessions(), {110 * GiB});
        REQUIRE(!d2.ok);
        Decision d3 = resolve(o, m, {96 * GiB, 96 * GiB}, sessions(), {110 * GiB});
        REQUIRE(d3.ok);
    }
    {  // json carries every field
        Options o;
        o.vram_budget = 12 * GiB;
        Decision d = resolve(o, m, {96 * GiB, 96 * GiB}, sessions(), big_host());
        const std::string j = decision_json(d);
        for (const char* k : {"\"expert_residency\":\"tiered\"", "\"reason\":", "\"vram_budget_bytes\":", "\"expert_pool_bytes\":",
                              "\"slots_per_layer\":", "\"positions\":", "\"positions_requested\":", "\"host_expert_bytes\":"})
            REQUIRE(has(j, k));
        REQUIRE(has(j, std::to_string(d.slots_per_layer).c_str()));
        REQUIRE(j.front() == '{' && j.back() == '}');
    }
    {  // device profile cap (route_budget) is a residency input: same plan as the equal explicit budget, smaller cap wins
        const CardBytes card{96 * GiB, 96 * GiB};
        auto with = [&](std::uint64_t budget, std::uint64_t cap, Mode mode = Mode::automatic) {
            Options o;
            o.mode = mode;
            o.vram_budget = budget;
            o.device_cap = cap;
            return resolve(o, m, card, sessions(), big_host());
        };
        auto same_plan = [](const Decision& a, const Decision& b) {
            return a.ok == b.ok && a.mode == b.mode && a.budget == b.budget && a.expert_pool == b.expert_pool &&
                   a.slots_per_layer == b.slots_per_layer && a.positions == b.positions && a.cap == b.cap;
        };
        const Decision none = with(0, 0), flag12 = with(12 * GiB, 0), prof12 = with(0, 12 * GiB);
        REQUIRE(none.cap == card.total && none.cap_source == "card" && none.mode == Mode::full);
        REQUIRE(flag12.cap_source == "residency" && prof12.cap_source == "device_profile");
        REQUIRE(same_plan(flag12, prof12) && prof12.mode == Mode::tiered && prof12.budget == 12 * GiB);
        const Decision tighter = with(8 * GiB, 12 * GiB), looser = with(24 * GiB, 12 * GiB), tie = with(12 * GiB, 12 * GiB);
        REQUIRE(tighter.cap_source == "residency" && same_plan(tighter, with(8 * GiB, 0)));
        REQUIRE(looser.cap_source == "device_profile" && same_plan(looser, prof12));
        REQUIRE(tie.cap_source == "residency" && same_plan(tie, prof12));
        // the forced tiered re-resolve of a route sees the same cap as its automatic decision
        REQUIRE(with(0, 12 * GiB, Mode::tiered).budget == prof12.budget);
        // the live free bytes still bound the budget below the cap; the cap is reported as applied
        Options o;
        o.device_cap = 12 * GiB;
        const Decision busy = resolve(o, m, {96 * GiB, 10 * GiB}, sessions(), big_host());
        REQUIRE(busy.budget == 0 && !busy.ok && busy.cap == 12 * GiB && busy.cap_source == "device_profile");
        o.device_cap = 48 * GiB;
        const CardBytes occupied{96 * GiB, 80 * GiB};
        REQUIRE(resolve(o, m, occupied, sessions(), big_host()).budget == 32 * GiB);
        o.vram_budget = 40 * GiB;
        REQUIRE(resolve(o, m, occupied, sessions(), big_host()).budget == 32 * GiB);
        o.vram_budget = 24 * GiB;
        const auto tighter_busy = resolve(o, m, occupied, sessions(), big_host());
        REQUIRE(tighter_busy.budget == 24 * GiB && tighter_busy.cap_source == "residency");
        o.device_cap = 0;
        REQUIRE(resolve(o, m, occupied, sessions(), big_host()).budget == 24 * GiB);
        o.vram_budget = 0;
        REQUIRE(resolve(o, m, occupied, sessions(), big_host()).budget == 80 * GiB);
        // External allocations may grow after startup: intersect the live total-card headroom first.
        const auto live = live_budget_view({96 * GiB, 56 * GiB}, 48 * GiB, 32 * GiB, 20 * GiB);
        REQUIRE(live.free == 8 * GiB && live.total == 32 * GiB);
        REQUIRE(live_budget_view({96 * GiB, 64 * GiB}, 48 * GiB, 32 * GiB, 20 * GiB).free == 12 * GiB);
        REQUIRE(live_budget_view({96 * GiB, 40 * GiB}, 48 * GiB, 32 * GiB, 20 * GiB).free == 0);
        REQUIRE(live_budget_view({96 * GiB, 56 * GiB}, 0, 32 * GiB, 20 * GiB).free == 12 * GiB);
        REQUIRE(live_budget_view({96 * GiB, 56 * GiB}, 48 * GiB).free == 8 * GiB);
        const std::string j = decision_json(prof12);
        REQUIRE(has(j, "\"vram_cap_source\":\"device_profile\"") && has(j, ("\"vram_cap_bytes\":" + std::to_string(12 * GiB)).c_str()));
        REQUIRE(has(decision_json(none), "\"vram_cap_source\":\"card\""));
    }
}

Options parse(std::vector<std::string> args) {
    Options o;
    for (std::size_t i = 0; i < args.size(); ++i) {
        auto next = [&]() -> std::string {
            if (i + 1 >= args.size()) throw std::invalid_argument("missing value");
            return args[++i];
        };
        if (!parse_flag(args[i], next, o)) throw std::invalid_argument("unknown " + args[i]);
    }
    return o;
}
bool rejects(std::vector<std::string> args) {
    try { parse(args); } catch (const std::invalid_argument&) { return true; }
    return false;
}

void test_flags() {
    Options o = parse({"--expert-residency", "tiered", "--vram-budget-gib", "7.5"});
    REQUIRE(o.mode == Mode::tiered && o.vram_budget == 15 * GiB / 2);
    REQUIRE(parse({"--expert-residency", "auto"}).mode == Mode::automatic);
    REQUIRE(parse({"--expert-residency", "full"}).mode == Mode::full);
    REQUIRE(rejects({"--vram-budget-gib", "0"}));
    REQUIRE(rejects({"--vram-budget-gib", "-3"}));
    REQUIRE(rejects({"--vram-budget-gib", "abc"}));
    REQUIRE(rejects({"--vram-budget-gib", "8x"}));
    REQUIRE(rejects({"--expert-residency", "some"}));
    Options unused;
    REQUIRE(!parse_flag("--other", [] { return std::string(); }, unused));

    set_env("STRIX_EXPERT_RESIDENCY", "full");
    set_env("STRIX_VRAM_BUDGET_GIB", "12");
    for (const char* mode : {"full", "tiered"}) {
        set_env("STRIX_EXPERT_RESIDENCY", mode);
        REQUIRE(with_environment(parse({"--expert-residency", "auto"})).mode == Mode::automatic);
    }
    set_env("STRIX_EXPERT_RESIDENCY", "full");
    Options e = with_environment({});
    REQUIRE(e.mode == Mode::full && e.vram_budget == 12 * GiB);
    Options f = with_environment(parse({"--expert-residency", "tiered", "--vram-budget-gib", "8"}));
    REQUIRE(f.mode == Mode::tiered && f.vram_budget == 8 * GiB);  // flags win
    set_env("STRIX_VRAM_BUDGET_GIB", "junk");
    bool threw = false;
    try { with_environment({}); } catch (const std::invalid_argument&) { threw = true; }
    REQUIRE(threw);
    set_env("STRIX_EXPERT_RESIDENCY", "");
    set_env("STRIX_VRAM_BUDGET_GIB", "");
    Options n = with_environment({});
    REQUIRE(n.mode == Mode::automatic && n.vram_budget == 0);

    // CLI == serve parity: the same words through the same two calls give the same Options.
    const std::vector<std::string> words{"--expert-residency", "tiered", "--vram-budget-gib", "12"};
    Options a = with_environment(parse(words)), b = with_environment(parse(words));
    REQUIRE(a.mode == b.mode && a.vram_budget == b.vram_budget);
}

ModelBytes small_model(std::uint32_t layers = 2, std::uint32_t experts = 16) {
    ModelBytes m;
    m.layers = layers;
    m.experts = experts;
    m.top_k = 4;
    m.expert_bytes.assign(layers, 1ull << 20);
    return m;
}

// One step on every layer: ids as given (same on all layers), then end_step.
void step(Planner& p, std::uint32_t layers, const std::vector<std::uint32_t>& ids) {
    for (std::uint32_t l = 0; l < layers; ++l) p.observe(l, ids.data(), std::uint32_t(ids.size()));
    p.end_step();
}
std::size_t apply(Planner& p) {
    auto mv = p.plan();
    for (const Move& m : mv) p.commit(m);
    return mv.size();
}

void test_planner() {
    PlannerConfig cfg;
    cfg.upload_bytes_per_step = 1ull << 30;
    {  // seeded fill picks the top counts
        Planner p(small_model(), 4, cfg);
        std::uint32_t c[16] = {};
        c[3] = 50; c[7] = 40; c[9] = 30; c[12] = 20; c[1] = 5; c[2] = 1;
        for (std::uint32_t l = 0; l < 2; ++l) p.seed(l, c);
        p.fill_from_seed();
        for (std::uint32_t l = 0; l < 2; ++l) {
            REQUIRE(p.table(l)[3] >= 0 && p.table(l)[7] >= 0 && p.table(l)[9] >= 0 && p.table(l)[12] >= 0);
            REQUIRE(p.table(l)[1] < 0 && p.table(l)[2] < 0);
        }
        REQUIRE(p.receipt().resident_bytes == 8ull << 20);
    }
    {  // stable hot set: hit rate ~1, no thrash
        Planner p(small_model(), 4, cfg);
        const std::vector<std::uint32_t> hot{2, 5, 8, 11};
        std::size_t moves_late = 0;
        for (int s = 0; s < 200; ++s) {
            step(p, 2, hot);
            const std::size_t n = apply(p);
            if (s >= 20) moves_late += n;
        }
        const Receipt r = p.receipt();
        REQUIRE(moves_late == 0);
        REQUIRE(r.byte_hit_rate() > 0.9);
        REQUIRE(r.evictions == 0 && r.promotions == 8);
        REQUIRE(has(receipt_json(r), "\"byte_hit_rate\":"));
    }
    {  // alternating two sets that each fit half: no ping-pong
        Planner p(small_model(), 4, cfg);
        const std::vector<std::uint32_t> a{0, 1, 2, 3}, b{4, 5, 6, 7};
        // 4 slots, union is 8 experts: only half can be resident.
        std::size_t moves = 0;
        for (int s = 0; s < 400; ++s) {
            step(p, 2, (s % 2) ? b : a);
            const std::size_t n = apply(p);
            if (s >= 20) moves += n;
        }
        REQUIRE(moves == 0);
    }
    {  // upload budget respected
        PlannerConfig c2 = cfg;
        c2.upload_bytes_per_step = 3ull << 20;  // 3 experts of 1 MiB
        Planner p(small_model(), 4, c2);
        step(p, 2, {0, 1, 2, 3, 4, 5});
        auto mv = p.plan();
        REQUIRE(mv.size() == 3);
        for (const Move& m : mv) p.commit(m);
        std::uint64_t total = 0;
        for (int s = 0; s < 10; ++s) {
            step(p, 2, {0, 1, 2, 3, 4, 5});
            auto v = p.plan();
            REQUIRE(v.size() * (1ull << 20) <= c2.upload_bytes_per_step);
            total += v.size();
            for (const Move& m : v) p.commit(m);
        }
        REQUIRE(total > 0);
    }
    {  // a real shift of the hot set is followed, never evicting this step's experts
        Planner p(small_model(1, 16), 4, cfg);
        for (int s = 0; s < 50; ++s) { step(p, 1, {0, 1, 2, 3}); apply(p); }
        bool protect_ok = true;
        for (int s = 0; s < 200; ++s) {
            step(p, 1, {8, 9, 10, 0});
            for (const Move& m : p.plan())
                if (m.evict == 0) protect_ok = false;
            apply(p);
        }
        REQUIRE(protect_ok);
        REQUIRE(p.table(0)[8] >= 0 && p.table(0)[9] >= 0 && p.table(0)[10] >= 0 && p.table(0)[0] >= 0);
    }
    {  // determinism
        auto run = [&]() {
            Planner p(small_model(), 4, cfg);
            std::vector<Move> all;
            for (int s = 0; s < 60; ++s) {
                step(p, 2, {std::uint32_t(s % 7), std::uint32_t((s * 3) % 11), std::uint32_t(s % 5), 12u});
                for (const Move& m : p.plan()) { all.push_back(m); p.commit(m); }
            }
            return all;
        };
        const auto x = run(), y = run();
        REQUIRE(!x.empty() && x.size() == y.size());
        bool same = true;
        for (std::size_t i = 0; i < x.size(); ++i)
            same = same && x[i].layer == y[i].layer && x[i].slot == y[i].slot && x[i].evict == y[i].evict && x[i].expert == y[i].expert;
        REQUIRE(same);
    }
    {  // receipt json: every field
        Receipt r;
        r.cpu_ms = 1.5; r.context_filled = 7;
        const std::string j = receipt_json(r);
        for (const char* k : {"active_bytes", "hit_bytes", "promoted_bytes", "promotions", "evictions", "steps", "cpu_ms",
                              "upload_ms", "exchange_ms", "resident_bytes", "context_filled"})
            REQUIRE(has(j, k));
    }
}
}  // namespace

int main() try {
    test_resolve();
    test_flags();
    test_planner();
    if (failures) {
        std::cerr << failures << " residency check(s) failed\n";
        return 1;
    }
    std::cout << "strix-expert-residency-tests: all passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "exception: " << e.what() << "\n";
    return 1;
}
