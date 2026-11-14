#include "glm/serve/glm_serve_prefix_cache.hpp"
#include <cstdio>
#include <cstdlib>
#include <new>
#include <utility>

// Allocation faults are local to this single-threaded host executable.
static bool fail_alloc = false;
static void* watched = nullptr;
static bool freed = false;
void* operator new(std::size_t n) {
    if (fail_alloc) throw std::bad_alloc();
    if (auto* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept {
    if (p && p == watched) freed = true;
    std::free(p);
}
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }

using strix::serve::ConversationKeep;
using strix::serve::KeepCandidate;
using strix::glm::serve::PrefixCache;
static int failures = 0;
static void check(bool b, const char* why) {
    if (!b) { ++failures; std::fprintf(stderr, "FAIL: %s\n", why); }
}
static KeepCandidate candidate(std::uint64_t id) {
    KeepCandidate c;
    c.id = id; c.identity = {1, 0}; c.ids = {1, 2}; c.frontier = 2; c.payload_bytes = 300;
    return c;
}
static PrefixCache::Entry entry(std::uint64_t id) {
    PrefixCache::Entry e;
    e.id = id; e.identity = 1; e.ids = {1, 2}; e.frontier = 2; e.state.resize(300);
    return e;
}
int main() {
    check(strix::glm::serve::prefix_snapshot_bound(UINT64_MAX) == UINT64_MAX, "snapshot bound saturates safely");
    check(strix::glm::serve::prefix_snapshot_bound(6000) > strix::glm::serve::prefix_snapshot_bound(0),
          "snapshot bound includes prompt state before copying");
    ConversationKeep k(4096);
    check(k.store(candidate(1)), "initial store");
    auto c = candidate(2);
    fail_alloc = true;
    bool failed = false;
    try { (void)k.prepare(std::move(c)); } catch (const std::bad_alloc&) { failed = true; }
    fail_alloc = false;
    check(failed && k.get(1) && !k.has_pending() && k.stats().stores == 1,
          "index allocation fails during preparation, preserving the old entry and ledger");
    check(k.prepare(candidate(2)), "prepare replacement");
    check(k.erase(1), "paired victim removal");
    fail_alloc = true;
    bool committed = false;
    try { committed = k.commit(true); } catch (const std::bad_alloc&) { /* regression */ }
    fail_alloc = false;
    check(committed && k.get(2) && k.stats().stores == 2, "shared commit allocates nothing");

    PrefixCache cache(4096);
    check(cache.prepare(entry(1)), "adapter preparation");
    fail_alloc = true;
    try { cache.commit(true); } catch (const std::bad_alloc&) { ++failures; }
    fail_alloc = false;
    check(cache.get(1) && cache.state(1), "index and payload commit together without allocations");
    auto e = entry(2);
    fail_alloc = true;
    bool prepared = false;
    try { prepared = cache.prepare(std::move(e)); } catch (const std::bad_alloc&) { /* refusal */ }
    fail_alloc = false;
    check(!prepared && cache.get(1) && cache.state(1) && cache.pending_bytes() == 0,
          "payload-node failure preserves old entry and resets staging");
    for (const bool clear : {false, true}) {
        auto pending = entry(3);
        watched = pending.state.data(); freed = false;
        check(cache.prepare(std::move(pending)), "prepare pending owner");
        if (clear) cache.clear(); else cache.commit(false);
        check(freed && cache.pending_bytes() == 0, "abort and clear release pending vector capacity");
        watched = nullptr;
    }
    PrefixCache bounded(500);
    check(bounded.can_stage(300, 2) && bounded.prepare(entry(1)), "first candidate fits grant");
    bounded.commit(true);
    check(!bounded.can_stage(300, 2) && !bounded.prepare(entry(2)),
          "retained plus pending must fit, even when replacement would eventually fit");
    check(bounded.get(1) && bounded.bytes() <= bounded.budget(), "peak refusal preserves retained owner");
    check(!bounded.can_stage(501, 2), "oversized candidate refused before copy");
    check(!bounded.can_stage(UINT64_MAX, 2), "overflowing bound refused");
    return failures ? 1 : 0;
}
