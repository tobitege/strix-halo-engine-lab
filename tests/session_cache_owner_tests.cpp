#include "strix/session/prefix_cache.hpp"
#include "strix/model_content_hash.hpp"
#include <array>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <latch>
#include <thread>
#include <cstdlib>
#include <new>

namespace strix {
struct PrefixStateCacheTestAccess {
    static std::size_t receipts(PrefixStateCache& cache) {
        std::lock_guard lock(cache.mutex_); return cache.receipts_.size();
    }
};
}
using namespace strix;
namespace ss = strix::session;
namespace {
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
PrefixCacheBuffer heap(std::uint64_t n) {
    if (n > SIZE_MAX) throw PrefixCacheAllocationFailure();
    return {std::shared_ptr<std::byte>(new std::byte[static_cast<std::size_t>(n ? n : 1)], std::default_delete<std::byte[]>()), n};
}
const ss::Identity identity{"assets", "arithmetic", "immutable-v1", 2};
std::shared_ptr<ss::PrefixSnapshot> snapshot(std::vector<std::uint32_t> tokens,
    std::shared_ptr<const void> payload = std::make_shared<const std::array<std::byte, 64>>()) {
    auto s = std::make_shared<ss::PrefixSnapshot>(); s->identity = identity;
    s->state.tokens = std::move(tokens); s->state.consumed = s->state.tokens.size();
    // Two rank views/captures alias ONE immutable allocation, declared once per
    // view in the frozen StateSnapshot contract and once physically in the ledger.
    s->state.ranks = {{payload,64},{payload,64}}; s->state.bytes = 128;
    s->attention_frontier = s->recurrent_frontier = s->state.consumed;
    return s;
}
std::shared_ptr<PrefixCacheCheckpoint> segmented(PrefixStateCache& core) {
    auto c = std::make_shared<PrefixCacheCheckpoint>();
    c->identity = sha256_bytes(std::as_bytes(std::span("owned",5)));
    auto segment = std::make_shared<PrefixCacheSegment>();
    segment->identity = c->identity; segment->parent = c->identity;
    segment->block_rows = 2; segment->tokens = {1,2};
    segment->digest = prefix_cache_block_digest(c->identity, segment->tokens);
    c->digest = segment->digest; c->block_rows = 2; c->position = 2;
    c->segments = {segment}; c->producer_certificate = {7,11};
    auto credit = core.reserve(176);
    check(bool(credit), "segmented reservation");
    c->fixed = core.allocate(32,credit); segment->payload = core.allocate(128,credit,true);
    check(core.insert(c,&credit), "actual segmented checkpoint admission");
    return c;
}
ss::Hooks hooks(bool& fail) {
    ss::Hooks h; h.initialize=[](auto){}; h.reset=[](auto){}; h.release=[](auto){};
    h.view=[](auto,unsigned) { return ss::RankState{std::make_shared<const unsigned>(17),4}; };
    h.capture=h.view; h.capture_prompt=h.view;
    h.restore=[](auto,unsigned,const auto&){}; h.truncate=[](auto,unsigned,std::uint64_t){};
    h.restore_prompt=[&fail](auto,auto) { if(fail) throw std::runtime_error("restore fault"); return true; };
    h.resident_prefill_supported=true;
    return h;
}
void mixed_pressure() {
    auto core = ss::make_cache_owner(2048);
    ss::Policy p; p.max_entries=8; p.max_bytes=2048;
    ss::PrefixCache prefix(p,core); ss::NamedStateCache named(2048,8,core);
    check(prefix.owner()==named.owner(), "one service owner");
    auto qwen=segmented(*core); const auto key=qwen->digest; qwen.reset();
    auto s=snapshot({3,4}); check(prefix.admit(s), "opaque snapshot"); s.reset();
    auto held=prefix.lookup(std::array{3U,4U,5U},identity);
    check(held && core->counters().immutable_bytes==64, "aliased rank owners charged once");
    bool fail=false; ss::KvStore slots(2,2,hooks(fail)); const auto h=slots.acquire();
    const std::array tokens{10U,11U,12U,13U}; slots.append_consumed(h,tokens);
    auto certificate=slots.release_resident(h,tokens.size());
    check(prefix.admit_resident(slots,h,tokens,identity), "resident certificate in same directory");
    certificate.reset();
    check(named.put("first",std::vector<std::uint8_t>(256,7)), "named pool adoption");
    auto encoded=named.lease("first");
    check(encoded && encoded->fixed.data.get()==reinterpret_cast<const std::byte*>(encoded->encoded->data()), "codec allocation is pool backing without copy");
    encoded.reset();
    check(core->counters().session_entries==3 && core->counters().host_checkpoints==4,
        "all four payload kinds share entries");
    // LRU: segmented, opaque, resident, named. A held opaque lookup is evicted
    // from the directory but remains charged and readable until its last lease.
    check(named.put("pressure",std::vector<std::uint8_t>(1700,9)), "cross-kind pressure admission");
    check(!core->contains(key) && !prefix.lookup(std::array{3U,4U,5U},identity) &&
        !prefix.lookup_resident(slots,tokens,identity,1).entry, "common victim order crosses every kind");
    check(named.find("first") && named.find("pressure"), "newer named entries survive");
    check(held->state.tokens==std::vector<std::uint32_t>({3,4}) && core->counters().immutable_bytes==64,
        "eviction preserves live immutable backing and charge");
    held.reset(); check(core->counters().immutable_bytes==0, "last immutable lease releases charge");
    const auto& directory=named.entries();
    check(directory.contains("first"), "borrowed compatibility directory reflects the core");
    auto read=named.lease("first");
    std::latch acquired(1), evicted(1); bool concurrent_read=false;
    std::thread reader([&] {
        auto concurrent=named.lease("first"); acquired.count_down(); evicted.wait();
        concurrent_read=concurrent && concurrent->encoded->at(0)==7;
    });
    acquired.wait(); named.erase("first");
    const auto descriptor_removed=!directory.contains("first");
    const auto held_bytes=core->counters().encoded_host_bytes;
    evicted.count_down(); reader.join();
    check(descriptor_removed && held_bytes==1956, "core invalidates borrowed descriptors before freeing held backing");
    check(concurrent_read && read->encoded->at(0)==7, "concurrent lookup ownership survives named eviction");
    read.reset(); check(core->counters().encoded_host_bytes==1700, "pool extent released after final reader");
    auto credit=core->reserve(348); check(bool(credit), "remaining common credits");
    // Evicting the last directory entry cannot reclaim an outstanding pool lease.
    auto live=named.lease("pressure");
    check(!core->reserve(1701), "outstanding reservation plus held pool cannot be overspent");
    check(core->counters().reserved_bytes==348, "reservation remains charged after eviction");
    live.reset(); credit={}; check(core->counters().service_bytes==0, "complete lifetime ledger reconciles");
}
void attached_pool_commitment() {
    auto core=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{16384,{},0,PrefixCacheRetention::lru},heap);
    auto pool=std::make_shared<PrefixPayloadPool>(heap(8192)); core->attach_payload_pool(pool,8192);
    auto q=segmented(*core); q.reset();
    ss::Policy p; p.max_bytes=16384;
    ss::PrefixCache prefix(p,core); ss::NamedStateCache named(16384,8,core);
    auto s=snapshot({3,4}); check(prefix.admit(s), "attached slab shares typed quota"); s.reset();
    bool fail=false; ss::KvStore store(1,2,hooks(fail)); auto h=store.acquire();
    const std::array tokens{10U,11U,12U,13U}; store.append_consumed(h,tokens);
    auto certificate=store.release_resident(h,tokens.size());
    check(prefix.admit_resident(store,h,tokens,identity) && named.put("small",{1,2,3}), "all payload kinds beside attached slab");
    check(core->counters().service_bytes==16384 && core->counters().pool_bytes==8192,
        "full slab commitment and outside owners are never double-spent");
    check(!named.put("beyond-physical-reserve",std::vector<std::uint8_t>(8192)), "free slab ranges cannot fund outside codec storage twice");
}
void mixed_completed_turns() {
    for (auto mode : {PrefixCacheRetention::lru, PrefixCacheRetention::fair}) {
        auto core=std::make_shared<PrefixStateCache>(PrefixStateCachePolicy{8192,{},0,mode},heap);
        ss::Policy ordinary; ordinary.max_bytes=8192;
        ss::Policy completed=ordinary; completed.completed_turns=true;
        ss::PrefixCache prefix(ordinary,core), turns(completed,core);
        ss::NamedStateCache named(8192,8,core);
        auto qwen=segmented(*core); const auto digest=qwen->digest; qwen.reset();
        check(prefix.admit(snapshot({3,4})), "ordinary snapshot beside fair segmented checkpoint");
        bool fail=false; ss::KvStore slots(2,2,hooks(fail)); auto lane=slots.acquire();
        slots.append_consumed(lane,std::array{8U,9U});
        auto turn=std::make_shared<ss::PrefixSnapshot>(); turn->identity=identity;
        turn->state=slots.capture_prompt(lane); turn->attention_frontier=turn->recurrent_frontier=2;
        turn->turn=ss::TurnMetadata{"conversation","render","rank-order","protected"};
        check(turns.admit(turn), "completed-turn keep shares the same service directory");
        turn.reset();
        auto certificate=slots.release_resident(lane,2);
        check(prefix.admit_resident(slots,lane,std::array{8U,9U},identity), "resident beside completed turn");
        certificate.reset();
        check(named.put("named",{1,2,3}) && core->counters().host_checkpoints==5,
            "segmented, prompt, completed turn, resident and encoded payloads coexist");
        check(!core->reserve(8192), "protected completed turn prevents a full-budget reservation");
        check(!core->contains(digest) && !named.find("named") && prefix.size()==0 && turns.size()==1,
            "common pressure crosses every unprotected kind under both fair and LRU policy");
        check(turns.unpin("protected") && bool(core->reserve(8192)),
            "explicit unpin releases the final common retention commitment");
    }
}
void named_churn() {
    auto core=ss::make_cache_owner(16384); ss::NamedStateCache named(16384,2,core);
    for(unsigned i=0;i<32;++i) {
        check(named.put("entry"+std::to_string(i),std::vector<std::uint8_t>(i+1,7)), "named churn admission");
        check(PrefixStateCacheTestAccess::receipts(*core)<=4, "expired physical lease descriptors stay bounded");
    }
    check(named.size()==2 && named.bytes()==63, "one retained quota after repeated encoded churn");
}
void legacy_oversize_leases() {
    auto core=ss::make_cache_owner(1);
    ss::Policy p; p.max_bytes=1; p.retain_one_oversize=true; p.charge_token_bytes=false;
    ss::PrefixCache cache(p,core);
    auto first=snapshot({1,2}); check(cache.admit(first), "legacy sole oversize seed");
    auto held=cache.lookup(std::array{1U,2U},identity);
    check(cache.admit(snapshot({3,4})) && cache.size()==1 && !cache.lookup(std::array{1U,2U},identity),
        "legacy replacement preserves sole retained semantics while old producer lease lives");
    check(held && core->counters().immutable_bytes==128 && core->counters().legacy_live_over_budget_bytes>0,
        "legacy outstanding allocations stay charged with explicit over-budget gauge");
    ss::Policy strict; strict.completed_turns=true; strict.max_bytes=4096;
    ss::PrefixCache turns(strict,core);
    check(!turns.reserve_turn(identity,{"conversation","render","rank-order",""},2,128),
        "legacy exemption cannot finance strict turn sealing");
}
void capture_aliases() {
    auto core=ss::make_cache_owner(8192);
    auto capture=snapshot({1,2});
    {
        ss::PrefixCache first({},core), second({},core);
        check(first.admit(capture) && second.admit(capture), "same capture retained by independent query scopes");
        check(core->counters().immutable_bytes==64, "shared capture counted once across entries");
    }
    check(core->counters().immutable_bytes==64, "external capture still owns evicted immutable allocation");
    capture.reset(); check(core->counters().immutable_bytes==0, "final external capture releases ledger charge");
}
void oversize_overflow() {
    auto core=ss::make_cache_owner(UINT64_MAX);
    ss::Policy policy; policy.retain_one_oversize=true;
    ss::PrefixCache cache(policy,core);
    auto large=snapshot({1}); large->identity.ranks=1;
    large->state.ranks.resize(1);
    large->state.ranks[0].bytes=large->state.bytes=UINT64_MAX-2048;
    check(cache.admit(large), "large declared immutable allocation admitted without allocating its byte bound");
    const auto held=cache.lookup(std::array{1U},large->identity);
    auto next=snapshot({2}); next->identity.ranks=1; next->state.ranks.resize(1);
    next->state.ranks[0].bytes=next->state.bytes=4096;
    check(!cache.admit(next), "sole-oversize cannot wrap live physical ownership after eviction");
    const auto stats=core->counters();
    check(held && stats.immutable_bytes==UINT64_MAX-2048 && stats.service_bytes>=stats.immutable_bytes,
        "overflow refusal preserves the old lease and complete accounting");
}
void policies_and_failure() {
    auto core=ss::make_cache_owner(8192); bool fail=false; ss::KvStore slots(2,2,hooks(fail));
    auto h=slots.acquire(); slots.append_consumed(h,std::array{1U,2U});
    auto turn=std::make_shared<ss::PrefixSnapshot>(); turn->identity=identity;
    turn->state=slots.capture_prompt(h); turn->attention_frontier=turn->recurrent_frontier=2;
    turn->turn=ss::TurnMetadata{"conversation","render","rank-order","keep"};
    ss::Policy p; p.completed_turns=true; p.max_bytes=4096; p.max_entries=1;
    ss::PrefixCache prefix(p,core); ss::NamedStateCache named(8192,8,core);
    check(prefix.admit(turn), "sealed completed-turn keep");
    auto protected_hit=prefix.lookup_turn(std::array{1U,2U,3U},identity,*turn->turn);
    check(protected_hit && protected_hit->state.sealed(), "typed sealed ownership retained");
    check(!prefix.can_admit_turn(identity,*turn->turn,2,8), "protected conversation cannot be replaced before seal");
    check(!named.put("too-large",std::vector<std::uint8_t>(8193)), "oversize named refusal before victims");
    check(!named.put("pressure",std::vector<std::uint8_t>(8192)), "protected keep blocks cross-kind budget");
    check(prefix.size()==1, "keep survives common pressure");
    const auto destination=slots.acquire(); fail=true;
    bool refused=false; try { (void)slots.restore_prompt(destination,protected_hit->state); } catch (...) { refused=true; }
    check(refused && slots.consumed(h)==2 && protected_hit->state.sealed(), "restore fault preserves healthy source and immutable sibling");
    fail=false; slots.release(destination); slots.release(h);
    check(prefix.unpin("keep") && !prefix.unpin("keep"), "explicit unpin");
    auto replacement=std::make_shared<ss::PrefixSnapshot>(*turn); replacement->turn->keep.clear();
    check(prefix.admit(replacement), "per-conversation replacement under one cap");
    check(prefix.size()==1, "replacement refreshes one entry");
    ss::Policy off=p; off.store=false; ss::PrefixCache no_store(off,core);
    check(!no_store.admit(turn), "no-store refuses new retention");
    ss::Policy strict=p; strict.max_bytes=1; strict.retain_one_oversize=true; strict.charge_token_bytes=false;
    ss::PrefixCache bound(strict,core); check(!bound.admit(turn), "legacy exceptions cannot bypass strict turns");
    auto overflow=snapshot({7}); overflow->state.ranks[0].bytes=UINT64_MAX; overflow->state.bytes=UINT64_MAX;
    check(!prefix.admit(overflow) && !prefix.can_admit_turn(identity,*turn->turn,UINT64_MAX,8), "checked token/payload arithmetic");
    PrefixStateCache failing({128,{},0,PrefixCacheRetention::lru},[](std::uint64_t)->PrefixCacheBuffer { throw std::bad_alloc(); });
    auto credit=failing.reserve(128); refused=false;
    try { (void)failing.allocate(128,credit); } catch(const PrefixCacheAllocationFailure&) { refused=true; }
    check(refused && failing.counters().reserved_bytes==128 && failing.counters().service_bytes==128, "allocation failure preserves reservation");
    credit={}; check(failing.counters().service_bytes==0, "failed allocation leaks no credit");
    ss::NamedStateCache empty(1,1);
    check(empty.put("empty",{}) && empty.find("empty") && empty.find("empty")->empty(), "empty encoded vector has stable immutable ownership");
    PrefixStateCachePolicy legacy_policy; legacy_policy.host_byte_budget=4096; legacy_policy.legacy=true;
    auto legacy=std::make_shared<PrefixStateCache>(legacy_policy,heap);
    ss::PrefixCache legacy_prefix({},legacy); ss::NamedStateCache legacy_named(4096,2,legacy);
    check(legacy_prefix.admit(snapshot({7,8})) && legacy_named.put("legacy",{1,2,3}), "legacy Qwen mode and session ledger coexist");
    check(legacy->counters().immutable_bytes==64 && legacy->counters().encoded_host_bytes==3, "legacy mode keeps typed and encoded accounting");
    auto unlimited=ss::make_cache_owner(UINT64_MAX); check(unlimited->counters().service_bytes==0, "legacy sentinel allocates no slab");
}
}
int main() try {
    mixed_pressure(); attached_pool_commitment(); mixed_completed_turns(); named_churn(); legacy_oversize_leases(); capture_aliases(); oversize_overflow(); policies_and_failure();
    std::cout<<"PASS shared retention ownership, pressure, leases and failures\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
