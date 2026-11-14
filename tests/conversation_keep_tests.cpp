#include "strix/serve/conversation_keep.hpp"
#include "strix/serve/slot_capacity.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

namespace {

using namespace strix::serve;
using Tokens = std::vector<std::uint32_t>;

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) { std::cerr << "FAIL: " << what << '\n'; ++failures; }
}

Tokens seq(std::uint32_t base, std::size_t n) {
    Tokens t;
    for (std::size_t i = 0; i < n; ++i) t.push_back(base + static_cast<std::uint32_t>(i));
    return t;
}
Tokens cat(Tokens a, const Tokens& b) { a.insert(a.end(), b.begin(), b.end()); return a; }

KeepCandidate cand(std::uint64_t id, Tokens ids, std::uint64_t frontier, std::uint64_t payload = 1000,
                   KeepKind kind = KeepKind::prompt_end, KeepIdentity who = {1, 0}) {
    KeepCandidate c;
    c.id = id; c.identity = who; c.kind = kind; c.ids = std::move(ids); c.frontier = frontier; c.payload_bytes = payload;
    return c;
}
std::uint64_t one(std::uint64_t payload, std::size_t key) { return ConversationKeep::charged_bytes(payload, key); }
std::uint64_t weak_hash(std::uint64_t, std::span<const std::uint32_t>) { return 7; }

void test_find_longest_strict() {
    ConversationKeep k(1 << 20);
    const auto p = seq(100, 10);
    expect(k.store(cand(1, p, 10)), "prompt-end store");
    const auto reply = cat(p, seq(500, 4));
    expect(k.store(cand(2, reply, 14, 1000, KeepKind::reply_end)), "reply-end store");
    const KeepIdentity id{1, 0};
    expect(k.find(id, cat(reply, seq(900, 3))) && k.find(id, cat(reply, seq(900, 3)))->id == 2, "longest key wins");
    expect(k.find(id, cat(p, seq(900, 3))) && k.find(id, cat(p, seq(900, 3)))->id == 1, "falls back to the shorter key");
    expect(k.find(id, reply) && k.find(id, reply)->id == 1, "key equal to the prompt is not a strict prefix: the shorter key serves");
    expect(!k.find(id, p), "prompt equal to key refused");
    expect(!k.find(id, seq(700, 20)), "unrelated prompt misses");
    expect(!k.find({2, 0}, cat(p, seq(900, 3))), "model mismatch");
    expect(!k.find({1, 1}, cat(p, seq(900, 3))), "variant mismatch");
    // frontier below the prompt but key longer than frontier (MTP style: key = rows + pending token)
    ConversationKeep m(1 << 20);
    expect(m.store(cand(1, cat(p, seq(5, 1)), 10)), "key longer than frontier");
    expect(!m.find(id, p), "prompt as long as frontier refused");
    expect(!m.find(id, cat(p, seq(900, 1))), "pending token differs");
    expect(m.find(id, cat(cat(p, seq(5, 1)), seq(900, 1))) != nullptr, "pending token equal");
    const auto s = k.stats();
    expect(s.stores == 2 && s.stores_by_kind[1] == 1 && s.entries == 2, "stats stores");
    const auto* e = k.lookup(id, cat(reply, seq(900, 3)));
    expect(e && k.stats().hits == 1 && k.stats().hit_tokens[1] == 14 && k.stats().hits_by_kind[1] == 1, "lookup counts hit by kind");
    (void)k.lookup(id, seq(700, 20));
    expect(k.stats().misses == 1, "lookup counts miss");
}

void test_hash_collision() {
    ConversationKeep k(1 << 20, &weak_hash);
    const auto a = seq(10, 6), b = seq(50, 6);
    expect(k.store(cand(1, a, 6)), "store a");
    expect(k.store(cand(2, b, 6)), "colliding hashes are different keys");
    expect(k.find({1, 0}, cat(a, seq(90, 2))) && k.find({1, 0}, cat(a, seq(90, 2)))->id == 1, "collision resolved by token compare (a)");
    expect(k.find({1, 0}, cat(b, seq(90, 2))) && k.find({1, 0}, cat(b, seq(90, 2)))->id == 2, "collision resolved by token compare (b)");
    expect(!k.find({1, 0}, cat(seq(10, 5), seq(77, 4))), "same hash, different tokens: no hit");
}

void test_replacement_lru_budget() {
    const auto n = one(1000, 4);
    ConversationKeep k(n * 3);
    expect(k.store(cand(1, seq(1, 4), 4)) && k.store(cand(2, seq(10, 4), 4)) && k.store(cand(3, seq(20, 4), 4)), "three fit");
    k.touch(1);  // LRU order is now 2, 3, 1
    expect(k.prepare(cand(4, seq(30, 4), 4)), "prepare 4");
    std::vector<std::uint64_t> v;
    expect(k.victims(v) && v == std::vector<std::uint64_t>{2}, "LRU victim is 2");
    expect(!k.pending_fits(), "does not fit before the victim is erased");
    expect(!k.commit(true), "commit(true) without room is refused");
    expect(k.stats().refused == 1, "refused counted");
    expect(k.prepare(cand(4, seq(30, 4), 4)), "prepare again");
    expect(k.victims(v) && v.size() == 1 && k.erase(v[0]), "erase victim");
    expect(k.pending_fits() && k.commit(true) && k.get(4) && !k.get(2), "commit after erase");
    // equal-key replacement goes first and wins over LRU order
    expect(k.prepare(cand(5, seq(20, 4), 4)), "prepare equal key");
    expect(k.replacement_id() == 3, "replacement id");
    expect(k.victims(v) && v == std::vector<std::uint64_t>{3}, "replacement is the only victim");
    expect(k.erase(3, KeepEvict::replaced) && k.commit(true) && k.get(5) && !k.get(3), "replaced");
    expect(k.stats().bytes == n * 3 && k.stats().entries == 3, "byte ledger");
    expect(!k.prepare(cand(5, seq(40, 4), 4)), "duplicate id refused");
    expect(!k.prepare(cand(6, seq(40, 4), 4, 0)), "empty payload refused");
    expect(!k.prepare(cand(7, seq(40, 4), 4, n * 4)), "larger than the budget refused");
    ConversationKeep off(0);
    expect(!off.store(cand(1, seq(1, 4), 4)), "budget 0 stores nothing");
    k.clear();
    expect(k.bytes() == 0 && k.size() == 0, "clear");
}

void test_leases() {
    const auto n = one(1000, 4);
    ConversationKeep k(n * 2);
    expect(k.store(cand(1, seq(1, 4), 4)) && k.store(cand(2, seq(10, 4), 4)), "two fit");
    expect(k.acquire(1), "lease 1");
    expect(k.oldest() == 2, "leased entry is not the oldest candidate");
    expect(k.prepare(cand(3, seq(20, 4), 4)), "prepare 3");
    std::vector<std::uint64_t> v;
    expect(k.victims(v) && v == std::vector<std::uint64_t>{2}, "victim skips the leased entry");
    expect(k.acquire(2), "lease 2");
    expect(!k.victims(v), "everything leased: no plan");
    expect(!k.store(cand(4, seq(30, 4), 4)), "store refused while all leased");
    k.release(1);
    k.release(2);
    expect(k.store(cand(5, seq(40, 4), 4)) && !k.get(1), "after release the LRU goes");
    // a leased equal key cannot be replaced
    ConversationKeep r(n * 4);
    expect(r.store(cand(1, seq(1, 4), 4)) && r.acquire(1), "leased");
    expect(r.prepare(cand(2, seq(1, 4), 4)) && !r.victims(v), "leased equal key blocks replacement");
    expect(!r.commit(true) && r.size() == 1, "commit refused, kept entry intact");
}

void test_pressure() {
    const auto n = one(1000, 4);
    ConversationKeep k(n * 4);
    for (std::uint64_t i = 1; i <= 4; ++i) expect(k.store(cand(i, seq(static_cast<std::uint32_t>(i * 20), 4), 4)), "fill");
    expect(k.acquire(1), "lease oldest");
    expect(k.reclaim_plan(1) == std::vector<std::uint64_t>{2}, "pressure plan skips the leased oldest");
    expect(k.reclaim_plan(n + 1) == (std::vector<std::uint64_t>{2, 3}), "plan covers the target in LRU order");
    expect(k.reclaim_plan(n * 100).size() == 3, "plan never includes leased entries");
    expect(k.reclaim_plan(0).empty(), "nothing to free");
    expect(k.shrink_plan().empty(), "at budget: no shrink");
    k.set_budget(n * 2);
    expect(k.shrink_plan() == (std::vector<std::uint64_t>{2, 3}), "lowered budget: shrink plan");
    for (const auto id : k.shrink_plan()) expect(k.erase(id, KeepEvict::pressure), "pressure erase");
    const auto s = k.stats();
    expect(s.pressure_evictions == 2 && s.evictions == 2 && s.bytes == n * 2 && s.budget == n * 2, "pressure stats");
    expect(!k.prepare(cand(9, seq(300, 4), 4, 1000 * 4)), "lowered budget refuses a big entry");
    k.release(1);
    expect(k.store(cand(9, seq(300, 4), 4)) && k.size() == 2, "store under the lowered budget evicts LRU");
}

void test_two_ranks_agree() {
    const auto n = one(500, 6);
    ConversationKeep a(n * 3), b(n * 3);
    auto both = [&](auto&& f) { f(a); f(b); };
    std::uint64_t id = 1;
    std::vector<std::uint64_t> va, vb;
    for (int step = 0; step < 40; ++step) {
        auto key = seq(static_cast<std::uint32_t>(10 * (step % 7)), 6);
        auto c = cand(id++, key, 6, 500, step % 3 ? KeepKind::prompt_end : KeepKind::reply_end);
        // rank 0 decides; rank 1 replays the lead's victims
        const bool pa = a.prepare(c), pb = b.prepare(c);
        expect(pa == pb, "prepare agrees");
        if (pa) {
            const bool ok = a.victims(va);
            expect(ok, "plan");
            for (const auto v : va) { expect(a.erase(v), "lead erase"); expect(b.erase(v), "peer erase"); }
            expect(a.commit(ok) == b.commit(ok), "commit agrees");
        }
        const auto p = cat(key, seq(900, 2));
        const auto* ea = a.lookup({1, 0}, p);
        const auto* eb = b.lookup({1, 0}, p);
        expect((ea == nullptr) == (eb == nullptr) && (!ea || ea->id == eb->id), "lookup agrees");
        if (ea) both([&](ConversationKeep& k) { k.touch(ea->id); });
    }
    const auto sa = a.stats(), sb = b.stats();
    expect(sa.bytes == sb.bytes && sa.entries == sb.entries && sa.stores == sb.stores && sa.evictions == sb.evictions &&
               sa.hits == sb.hits && sa.hit_tokens[0] == sb.hit_tokens[0] && sa.hit_tokens[1] == sb.hit_tokens[1],
           "two ranks fed the same calls agree");
    expect(sa.bytes <= n * 3, "budget respected");
}

void test_capacity() {
    CapacityPool pool{"MemAvailable", 100ULL << 20, 10ULL << 20, {20ULL << 20, 5ULL << 20}};
    const auto base = report_slot_capacity({pool});
    const auto zero = report_slot_capacity({with_keep_budget(pool, 0)});
    expect(base.slots == zero.slots && base.terms[0].keep_bytes == 0, "no keep budget: report unchanged");
    expect(format_capacity_report(base).find("keep=") == std::string::npos, "no keep term in the text when unused");
    expect(capacity_report_json(base).find("keep_bytes") == std::string::npos, "no keep term in the json when unused");
    // (100 - 10 headroom - 5 first slot) / 25 = 3 extra: 4 slots
    expect(base.slots == 4, "baseline slots");
    // an explicit charge is paid before slots are counted: 30 more MiB of headroom -> 2 extra, 3 slots
    auto charged = pool;
    charged.keep_bytes = 30ULL << 20;
    const auto r = report_slot_capacity({charged});
    expect(r.slots == 3 && r.terms[0].keep_bytes == (30ULL << 20) && r.terms[0].headroom_bytes == (10ULL << 20), "charged before slots are counted");
    expect(format_capacity_report(r).find("keep=31457280") != std::string::npos, "text shows the keep budget");
    expect(capacity_report_json(r).find("\"keep_bytes\":31457280") != std::string::npos, "json shows the keep budget");
    // live slots win: the grant keeps min_slots (4) slots and shrinks the cache instead
    const auto small = with_keep_budget(pool, 6ULL << 20, 4);
    expect(small.keep_bytes == (6ULL << 20), "grant within the free memory");
    const auto big = with_keep_budget(pool, 1ULL << 40, 4);
    expect(big.keep_bytes == (100ULL << 20) - (10ULL << 20) - (5ULL << 20) - 3 * (25ULL << 20), "grant leaves min_slots slots");
    expect(report_slot_capacity({big}).slots == 4 && report_slot_capacity({small}).slots == 4, "slots do not shrink for the cache");
    expect(keep_budget_grant(pool, 1 << 20, 100) == 0, "no room for that many slots: no cache");
    CapacityPool unknown{"X", std::nullopt, 0, {1, 1}};
    expect(keep_budget_grant(unknown, 100) == 0, "unobserved pool grants nothing");
}

}  // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, 0);
    _CrtSetReportMode(_CRT_ERROR, 0);
#endif
    test_find_longest_strict();
    test_hash_collision();
    test_replacement_lru_budget();
    test_leases();
    test_pressure();
    test_two_ranks_agree();
    test_capacity();
    if (failures) { std::cerr << failures << " conversation keep check(s) failed" << std::endl; return 1; }
    std::cout << "conversation keep tests passed" << std::endl;
    return 0;
}
