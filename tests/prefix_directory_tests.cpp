#include "strix/serve/prefix_directory.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

namespace {

using strix::serve::PrefixDirectory;
using strix::serve::PrefixKey;
using strix::serve::PrefixMatch;
using strix::serve::PrefixState;
using Tokens = std::vector<std::uint32_t>;

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

// prefix: `shared` tokens common to a family, then a tail owned by `salt`.
Tokens prompt(std::uint32_t family, std::size_t shared, std::uint32_t salt, std::size_t tail) {
    Tokens t;
    t.reserve(shared + tail);
    for (std::size_t i = 0; i < shared; ++i) t.push_back(family * 100000U + static_cast<std::uint32_t>(i));
    for (std::size_t i = 0; i < tail; ++i) t.push_back(salt * 1000000U + static_cast<std::uint32_t>(i) + 7U);
    return t;
}

bool is(const PrefixMatch& m, std::uint64_t entry, std::uint64_t tokens, bool inflight) {
    return m.entry == entry && m.tokens == tokens && m.inflight == inflight;
}

std::uint64_t weak_hash(std::uint64_t, std::span<const std::uint32_t>) { return 42; }

const PrefixKey kKey{1, 2, 3};

void test_lookup_announce_publish() {
    PrefixDirectory dir(64, 256);
    const Tokens a = prompt(1, 2853, 1, 40);
    expect(is(dir.lookup(kKey, a), 0, 0, false), "empty directory matches nothing");
    const std::uint64_t e = dir.announce(kKey, a, 5);
    expect(e != 0 && dir.size() == 1, "announce returns an entry");
    expect(dir.status(e).state == PrefixState::inflight && dir.status(e).lane == 5 &&
           dir.status(e).tokens == a.size(), "in-flight status");

    // Frontier inside a block: 2853 = 44 * 64 + 37.
    const Tokens b = prompt(1, 2853, 2, 90);
    expect(is(dir.lookup(kKey, b), e, 2853, true), "in-flight match at the exact common length");
    // A shorter divergence point, also off the block grid.
    Tokens c = b;
    c[1001] ^= 1U;
    expect(is(dir.lookup(kKey, c), e, 1001, true), "in-flight match stops at the first differing token");
    // The same prompt again keeps one suffix token.
    expect(is(dir.lookup(kKey, a), e, a.size() - 1, true), "identical prompt leaves one suffix token");
    // A prompt that is a strict prefix of the entry.
    const Tokens head(a.begin(), a.begin() + 700);
    expect(is(dir.lookup(kKey, head), e, 699, true), "prompt shorter than the entry leaves one suffix token");

    dir.publish(e, 2853);
    expect(dir.status(e).state == PrefixState::sealed && dir.status(e).tokens == 2853, "sealed status");
    expect(is(dir.lookup(kKey, b), e, 2853, false), "sealed match at the frontier");
    expect(is(dir.lookup(kKey, c), 0, 0, false), "sealed state is not usable below its frontier");
    const Tokens exact(a.begin(), a.begin() + 2853);
    expect(is(dir.lookup(kKey, exact), 0, 0, false), "sealed frontier equal to the prompt has no suffix");
    dir.publish(e, 2853);
    expect(dir.status(e).state == PrefixState::sealed, "repeated publish at the same frontier is a no-op");
    dir.publish(e, 2000);
    expect(dir.status(e).state == PrefixState::absent && dir.size() == 0, "re-seal at another length withdraws");
    expect(is(dir.lookup(kKey, b), 0, 0, false), "withdrawn entry does not match");
}

void test_min_tokens_and_bad_publish() {
    PrefixDirectory dir(64, 256);
    expect(dir.announce(kKey, prompt(1, 255, 1, 0), 0) == 0, "prompt below min_tokens is not announced");
    const Tokens a = prompt(1, 300, 1, 300);
    const std::uint64_t e = dir.announce(kKey, a, 0);
    expect(is(dir.lookup(kKey, prompt(1, 255, 2, 400)), 0, 0, false), "common length below min_tokens");
    expect(is(dir.lookup(kKey, prompt(1, 256, 2, 400)), e, 256, true), "common length equal to min_tokens");
    expect(is(dir.lookup(kKey, prompt(1, 256, 2, 0)), 0, 0, false), "no suffix left above min_tokens");
    dir.publish(e, a.size() + 1);
    expect(dir.status(e).state == PrefixState::absent, "frontier beyond the announced tokens withdraws");
    const std::uint64_t f = dir.announce(kKey, a, 0);
    expect(f > e, "entry numbers are not reused");
    dir.publish(f, 100);
    expect(dir.status(f).state == PrefixState::absent, "frontier below min_tokens withdraws");
    dir.publish(12345, 300);
    dir.withdraw(12345);
    dir.release(12345);
    expect(!dir.acquire(12345) && dir.size() == 0, "unknown entries are ignored");

    // Prompts shorter than one block still match through the identity bucket.
    PrefixDirectory small(64, 4);
    const std::uint64_t s = small.announce(kKey, prompt(3, 20, 1, 5), 0);
    expect(is(small.lookup(kKey, prompt(3, 20, 2, 5)), s, 20, true), "match below one block");
}

void test_identity_mismatch() {
    PrefixDirectory dir(64, 256);
    const Tokens a = prompt(1, 1000, 1, 10);
    const Tokens b = prompt(1, 1000, 2, 10);
    const std::uint64_t e = dir.announce(kKey, a, 0);
    expect(is(dir.lookup(PrefixKey{9, 2, 3}, b), 0, 0, false), "model mismatch");
    expect(is(dir.lookup(PrefixKey{1, 9, 3}, b), 0, 0, false), "template mismatch");
    expect(is(dir.lookup(PrefixKey{1, 2, 9}, b), 0, 0, false), "format mismatch");
    const std::uint64_t other = dir.announce(PrefixKey{1, 2, 9}, a, 1);
    expect(is(dir.lookup(kKey, b), e, 1000, true), "identity selects its own entry");
    expect(is(dir.lookup(PrefixKey{1, 2, 9}, b), other, 1000, true), "second identity selects its own entry");
}

void test_hash_collision() {
    // Every chain value collides, across identities too: only the token compare can decide.
    PrefixDirectory dir(64, 256, &weak_hash);
    const Tokens a = prompt(1, 1500, 1, 20);
    const Tokens unrelated = prompt(2, 1500, 1, 20);
    const std::uint64_t e = dir.announce(kKey, a, 0);
    expect(is(dir.lookup(kKey, unrelated), 0, 0, false), "colliding hashes with different tokens do not match");
    const std::uint64_t u = dir.announce(kKey, unrelated, 1);
    expect(is(dir.lookup(kKey, prompt(1, 1500, 3, 9)), e, 1500, true), "collision falls back to the token compare");
    expect(is(dir.lookup(kKey, prompt(2, 1500, 3, 9)), u, 1500, true), "collision picks the entry whose tokens agree");
    Tokens partial = prompt(1, 1500, 3, 9);
    partial[777] ^= 1U;
    expect(is(dir.lookup(kKey, partial), e, 777, true), "collision keeps the exact common length");
    // Weak identity seed as well: a colliding bucket holding another identity.
    const std::uint64_t alien = dir.announce(PrefixKey{7, 7, 7}, a, 2);
    expect(alien != 0 && is(dir.lookup(kKey, prompt(1, 1500, 3, 9)), e, 1500, true),
           "another identity in a colliding bucket is skipped");
}

void test_longest_and_ties() {
    PrefixDirectory dir(64, 256);
    const Tokens deep = prompt(1, 2000, 1, 50);
    Tokens shallow = deep;
    shallow[600] ^= 1U; // shares 600 tokens with deep
    const std::uint64_t es = dir.announce(kKey, shallow, 0);
    const std::uint64_t ed = dir.announce(kKey, deep, 1);
    expect(is(dir.lookup(kKey, prompt(1, 2000, 2, 9)), ed, 2000, true), "longest entry wins");
    Tokens early = prompt(1, 2000, 2, 9);
    early[400] ^= 2U;
    expect(is(dir.lookup(kKey, early), es, 400, true), "equal length prefers the older entry");
    dir.publish(ed, 1024);
    // deep is sealed at 1024; the in-flight shallow entry offers only 600.
    expect(is(dir.lookup(kKey, prompt(1, 2000, 2, 9)), ed, 1024, false), "sealed frontier beats a shorter in-flight one");
    // A request that leaves the sealed prefix before 1024 falls to the in-flight entry.
    Tokens mid = prompt(1, 2000, 2, 9);
    mid[500] ^= 4U;
    expect(is(dir.lookup(kKey, mid), es, 500, true), "unusable sealed entry does not hide an in-flight one");
    // Sealed and in-flight at the same length: sealed wins although it is newer.
    PrefixDirectory tie(64, 256);
    const std::uint64_t live = tie.announce(kKey, prompt(4, 512, 1, 0), 0);
    const std::uint64_t done = tie.announce(kKey, prompt(4, 512, 2, 30), 1);
    tie.publish(done, 512);
    expect(live < done && is(tie.lookup(kKey, prompt(4, 512, 3, 30)), done, 512, false),
           "sealed entry preferred at equal length");
}

void test_producer_cancel() {
    PrefixDirectory dir(64, 256);
    const Tokens a = prompt(1, 1200, 1, 30);
    const Tokens b = prompt(1, 1200, 2, 30);
    const Tokens c = prompt(1, 1200, 3, 30);
    const std::uint64_t e = dir.announce(kKey, a, 0);
    const PrefixMatch mb = dir.lookup(kKey, b);
    const PrefixMatch mc = dir.lookup(kKey, c);
    expect(mb.entry == e && mc.entry == e && dir.acquire(e) && dir.acquire(e), "two waiters lease the producer");
    expect(dir.status(e).leases == 2, "lease count");
    dir.withdraw(e); // producer cancelled mid-prefill
    expect(dir.status(e).state == PrefixState::withdrawn && dir.status(e).tokens == 0,
           "waiters see the withdrawal");
    expect(!dir.acquire(e), "no new lease on a withdrawn entry");
    expect(is(dir.lookup(kKey, b), 0, 0, false), "withdrawn producer stops matching");
    dir.publish(e, 1200);
    expect(dir.status(e).state == PrefixState::withdrawn, "a late publish does not revive the entry");
    // First waiter falls back to its own prefill and becomes the producer for the other one.
    const std::uint64_t e2 = dir.announce(kKey, b, 1);
    dir.release(e);
    expect(dir.status(e).state == PrefixState::withdrawn, "entry kept until the last release");
    expect(is(dir.lookup(kKey, c), e2, 1200, true), "second waiter joins the new producer");
    dir.release(e);
    expect(dir.status(e).state == PrefixState::absent && dir.size() == 1, "dropped at the last release");
    dir.release(e);
    dir.withdraw(e);
    expect(dir.size() == 1 && dir.status(e2).state == PrefixState::inflight, "stale calls touch nothing else");
}

void test_eviction_while_leased() {
    PrefixDirectory dir(64, 256);
    const Tokens a = prompt(1, 900, 1, 30);
    const std::uint64_t e = dir.announce(kKey, a, 3);
    dir.publish(e, 900);
    expect(dir.acquire(e) && dir.status(e).leases == 1, "fork leases a sealed entry");
    dir.withdraw(e); // owner evicts the state
    expect(dir.status(e).state == PrefixState::withdrawn && dir.status(e).leases == 1 && dir.size() == 1,
           "evicted entry stays visible to its lease holder");
    expect(is(dir.lookup(kKey, prompt(1, 900, 2, 30)), 0, 0, false), "evicted entry stops matching");
    dir.withdraw(e);
    expect(dir.status(e).leases == 1, "repeated withdraw keeps the lease");
    dir.release(e);
    expect(dir.status(e).state == PrefixState::absent && dir.size() == 0, "released after eviction");
    // Without leases an eviction drops the entry at once.
    const std::uint64_t f = dir.announce(kKey, a, 3);
    dir.acquire(f);
    dir.release(f);
    expect(dir.status(f).state == PrefixState::inflight, "release alone does not drop a live entry");
    dir.withdraw(f);
    expect(dir.size() == 0, "unleased eviction drops at once");
}

std::vector<PrefixDirectory::Group> plan(const PrefixDirectory& dir, const std::vector<Tokens>& prompts) {
    std::vector<std::span<const std::uint32_t>> views;
    for (const Tokens& t : prompts) views.emplace_back(t);
    return dir.plan_groups(views);
}

bool group_is(const PrefixDirectory::Group& g, std::uint64_t tokens, std::vector<std::uint32_t> members) {
    return g.tokens == tokens && g.members == members && g.producer == members.front();
}

void test_plan_groups() {
    PrefixDirectory dir(64, 256);
    expect(plan(dir, {}).empty() && plan(dir, {prompt(1, 500, 1, 5)}).empty(), "fewer than two prompts");
    expect(plan(dir, {prompt(1, 500, 1, 5), prompt(2, 500, 1, 5)}).empty(), "unrelated prompts");
    expect(plan(dir, {prompt(1, 255, 1, 50), prompt(1, 255, 2, 50)}).empty(), "common prefix below min_tokens");

    // Four divergent suffixes on one prefix.
    auto g = plan(dir, {prompt(1, 2853, 1, 9), prompt(1, 2853, 2, 9), prompt(1, 2853, 3, 9), prompt(1, 2853, 4, 9)});
    expect(g.size() == 1 && group_is(g[0], 2853, {0, 1, 2, 3}), "one group at the exact frontier");

    // Nested: X = 300 tokens, XYZ = 2000 tokens. A, B share XYZ; C, D share only X.
    const Tokens xyz = prompt(1, 2000, 0, 0);
    auto with_tail = [](Tokens base, std::size_t keep, std::uint32_t salt) {
        base.resize(keep);
        for (std::uint32_t i = 0; i < 40; ++i) base.push_back(salt * 1000000U + i);
        return base;
    };
    const Tokens A = with_tail(xyz, 2000, 11), B = with_tail(xyz, 2000, 12);
    const Tokens C = with_tail(xyz, 300, 13), D = with_tail(xyz, 300, 14);
    g = plan(dir, {C, A, D, B});
    expect(g.size() == 2 && group_is(g[0], 300, {0, 2}) && group_is(g[1], 2000, {1, 3}),
           "nested prefixes: deep group and shallow group");
    // With one shallow prompt only, it stays alone and the deep pair still groups.
    g = plan(dir, {A, C, B});
    expect(g.size() == 1 && group_is(g[0], 2000, {0, 2}), "single shallow prompt is left to its own prefill");
    // A wide shallow group beats a short deep one when it saves more tokens.
    const Tokens E = with_tail(xyz, 400, 15), F = with_tail(xyz, 400, 16);
    g = plan(dir, {C, D, E, F, with_tail(xyz, 300, 17)});
    expect(g.size() == 1 && group_is(g[0], 300, {0, 1, 2, 3, 4}), "group chosen by tokens saved");

    // Identical prompts and a prompt that is a prefix of another keep one suffix token.
    g = plan(dir, {A, A});
    expect(g.size() == 1 && group_is(g[0], A.size() - 1, {0, 1}), "identical prompts");
    const Tokens head(A.begin(), A.begin() + 1000);
    g = plan(dir, {A, head});
    expect(g.size() == 1 && group_is(g[0], 999, {0, 1}), "prefix prompt keeps one suffix token");
    // Two families at once, producers are the lowest indices.
    g = plan(dir, {prompt(5, 600, 1, 9), prompt(6, 700, 1, 9), prompt(6, 700, 2, 9), prompt(5, 600, 2, 9)});
    expect(g.size() == 2 && group_is(g[0], 600, {0, 3}) && group_is(g[1], 700, {1, 2}), "two families");
}

} // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif
    test_lookup_announce_publish();
    test_min_tokens_and_bad_publish();
    test_identity_mismatch();
    test_hash_collision();
    test_longest_and_ties();
    test_producer_cancel();
    test_eviction_while_leased();
    test_plan_groups();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "prefix_directory_tests: ok\n";
    return 0;
}
