#include "strix/deepseek41_prefix_cache.hpp"
#include "strix/deepseek41_attention_host.hpp"
#include "strix/ds41_attention.hpp"
#include <cstring>
#include <iostream>
#include <numeric>
#include <atomic>

namespace ds = strix::deepseek41;
void check(bool v, const char* text) { if (!v) throw std::runtime_error(text); }
std::uint64_t hash(const void* data, std::size_t n, std::uint64_t h=14695981039346656037ULL) {
    const auto* p=static_cast<const unsigned char*>(data);
    while(n--) h=(h^*p++)*1099511628211ULL;
    return h;
}
ds::AttentionState image(std::vector<std::uint32_t> t, unsigned p, bool unknown=false) {
    namespace a = strix::ds41::attn;
    ds::AttentionState s; s.position=p; t.resize(p); s.tokens=t;
    s.next_token=unknown ? ds::kPrefixUnknownNextToken : t.back()+7;
    for(unsigned l=0;l<5;++l) {
        ds::AttentionLayerState x; x.owns_kv=l<4; x.owns_index=l==0 || l==2;
        x.ratio=l==1 ? 2U : l==2 ? 8U : 1U;
        x.ring.resize(ds::AttentionHost::ring_capacity*a::kWindowEntryBytes);
        for(std::size_t j=0;j<x.ring.size();++j) x.ring[j]=static_cast<unsigned char>(j*13+l+t.back());
        if(x.owns_kv) {
            x.kv.resize(p/x.ratio*a::kKvEntryBytes); x.idx_keys.resize(p/x.ratio*a::kIdxEntryBytes);
            for(std::size_t j=0;j<x.kv.size();++j) x.kv[j]=static_cast<unsigned char>(t[(j/a::kKvEntryBytes+1)*x.ratio-1]+j+l);
            for(std::size_t j=0;j<x.idx_keys.size();++j) x.idx_keys[j]=static_cast<unsigned char>(t[(j/a::kIdxEntryBytes+1)*x.ratio-1]+j*3+l);
            if(x.ratio==2) { x.carry.resize(2*a::kHeadDim); for(unsigned j=0;j<x.carry.size();++j) x.carry[j]=static_cast<float>(j+1)/16; x.carry_valid=true; }
        }
        x.topk_known=false; s.layers.push_back(std::move(x));
    }
    return s;
}
ds::AttentionState delta(ds::AttentionState s,unsigned from) {
    for(auto& l:s.layers) if(l.owns_kv) {
        l.kv.erase(l.kv.begin(),l.kv.begin()+from/l.ratio*strix::ds41::attn::kKvEntryBytes);
        l.idx_keys.erase(l.idx_keys.begin(),l.idx_keys.begin()+from/l.ratio*strix::ds41::attn::kIdxEntryBytes);
    }
    return s;
}
std::uint64_t state_hash(const ds::AttentionState& s) {
    auto h=hash(s.tokens.data(),s.tokens.size()*4);
    h=hash(&s.position,4,h); h=hash(&s.next_token,4,h);
    for(const auto& l:s.layers) {
        for(const auto* v:{&l.kv,&l.idx_keys,&l.ring}) h=hash(v->data(),v->size(),h);
        h=hash(l.carry.data(),l.carry.size()*4,h); h=hash(&l.carry_valid,sizeof(l.carry_valid),h);
        check(!l.topk_known && l.topk.empty(),"topk contract");
    }
    return h;
}
void exact_state(const ds::AttentionState& a,const ds::AttentionState& b) {
    check(a.tokens==b.tokens && a.position==b.position && a.next_token==b.next_token && a.layers.size()==b.layers.size(),"exact frontiers");
    for(std::size_t l=0;l<a.layers.size();++l) {
        const auto& x=a.layers[l]; const auto& y=b.layers[l];
        check(x.owns_kv==y.owns_kv && x.owns_index==y.owns_index && x.ratio==y.ratio,"exact ownership layout");
        check(x.kv==y.kv && x.idx_keys==y.idx_keys && x.ring==y.ring,"exact native-shaped row/ring bytes");
        check(x.carry_valid==y.carry_valid && x.carry.size()==y.carry.size() &&
            (x.carry.empty() || std::memcmp(x.carry.data(),y.carry.data(),x.carry.size()*4)==0),"exact carry bytes");
    }
}
void trace(ds::PrefixCache& c,const char* op,const std::vector<std::uint32_t>& t) {
    auto q=c.lookup(t); auto s=c.stats();
    std::cout<<op<<' '<<q.resume<<' '<<q.key<<' '<<q.shared<<' '<<q.next_token<<' '<<q.logits_fnv<<' '<<q.disk<<' '<<q.disk_bytes;
    std::cout<<" gauges "<<s.bytes<<' '<<s.blocks<<' '<<s.snapshots<<" calls "<<s.lookups<<' '<<s.hits<<' '<<s.hit_tokens<<' '<<s.inserts<<' '<<s.evicted_snapshots<<' '<<s.evicted_blocks<<' '<<s.refused;
    std::cout<<" coverage "<<c.covered(t,static_cast<unsigned>(t.size()));
    if(q.resume) std::cout<<" state "<<state_hash(c.load(t,q.resume,q.key));
    std::cout<<'\n';
}
void original_trace() {
    std::vector<std::uint32_t> a(40); std::iota(a.begin(),a.end(),11U); auto b=a; b[16]+=99; auto z=a; z[0]+=511;
    const auto full=image(a,32); const auto cap=ds::PrefixCache::snapshot_bytes(full,8);
    for(auto budget:{0ULL,1ULL,static_cast<unsigned long long>(cap),static_cast<unsigned long long>(cap+cap/2)}) {
        ds::PrefixCacheOptions o; o.block=8; o.budget_bytes=budget; ds::PrefixCache c(o);
        std::cout<<"budget "<<budget<<'\n';
        for(unsigned p:{8U,16U,24U}) { std::cout<<"insert "<<p<<' '<<c.insert(image(a,p,p==24),100+p)<<'\n'; trace(c,"a",a); }
        std::cout<<"branch "<<c.insert(image(b,32),132)<<'\n'; trace(c,"branch",b); trace(c,"lookup-no-touch",a);
        auto q=c.lookup(b); if(q.resume) { auto s=c.load(b,q.resume,q.key); c.use(s,q.key); }
        trace(c,"use-touch",b);
        std::cout<<"delta "<<c.insert(delta(full,16),232,16)<<'\n'; trace(c,"delta",a);
        std::cout<<"duplicate "<<c.insert(full,999)<<'\n'; trace(c,"duplicate",a);
        std::cout<<"other "<<c.insert(image(z,32),332)<<'\n'; trace(c,"other",z); trace(c,"after-pressure",b);
        std::vector<std::uint32_t> whole(a.begin(),a.begin()+24); auto hit=c.lookup(whole); auto min=c.lookup(a,20);
        std::cout<<"unknown/min "<<hit.resume<<' '<<min.resume<<'\n';
        c.clear_ram(); trace(c,"clear-ram",a); std::cout<<"missing-delta "<<c.insert(delta(full,16),432,16)<<'\n';
        c.clear(); trace(c,"clear",a);
    }
}
#ifdef STRIX_DS41_OWNER_ADOPTED
void adoption_checks() {
    std::vector<std::uint32_t> a(40); std::iota(a.begin(),a.end(),11U);
    auto b=a; b[16]+=777; auto z=a; z[0]+=999;
    const auto a8=image(a,8), a32=image(a,32);
    const auto cap=ds::PrefixCache::snapshot_bytes(a32,8);
    std::uint64_t allocations=0,alive=0,physical=0; int fail_after=-1;
    std::vector<const std::byte*> released; released.reserve(64); bool record_release=false;
    strix::PrefixStateCachePolicy policy; policy.host_byte_budget=8*cap; policy.retention=strix::PrefixCacheRetention::lru;
    auto owner=std::make_shared<strix::PrefixStateCache>(policy,[&](std::uint64_t n) {
        if(fail_after==0) throw std::bad_alloc();
        if(fail_after>0) --fail_after;
        ++allocations; ++alive; physical+=n;
        return strix::PrefixCacheBuffer{std::shared_ptr<std::byte>(new std::byte[static_cast<std::size_t>(n)],
            [&,n](std::byte* p) {if(record_release) released.push_back(p); delete[] p; --alive; physical-=n;}),n};
    });
    ds::PrefixCacheOptions o; o.block=8; o.budget_bytes=4*cap;
    {
        ds::PrefixCache c(o,owner), other(o,owner);
        check(allocations==0,"no eager allocation");
        check(c.insert(a8,8) && c.insert(a32,32) && c.insert(image(b,32),132),"branch admission");
        check(other.insert(image(z,32),232),"second scope admission");
        auto entries=owner->inspect(1);
        check(entries.size()==3 && owner->inspect(2).size()==1,"actual common directory adoption");
        check(entries[1]->tokens()==a32.tokens,"common chain token reconstruction");
        auto branch=entries.back()->segments.front(); auto main=entries[1]->segments.front();
        check(branch->ancestor->ancestor==main->ancestor->ancestor,"shared branch ancestry pointer");
        check(allocations==2+4+3+5,"one allocation per distinct block plus snapshot");
        auto before=c.stats(); auto count=allocations; auto bytes=owner->counters().service_bytes;
        check(c.insert(a32,999),"duplicate admitted");
        check(allocations==count && c.stats().bytes==before.bytes && owner->counters().service_bytes==bytes,"duplicate no allocation/charge");
        check(c.lookup(a).logits_fnv==32,"duplicate preserves snapshot");
        auto loaded=c.state(a,32,c.lookup(a).key); const auto expected=state_hash(a32);
        check(state_hash(loaded)==expected,"independent native-shaped state bytes");
        exact_state(loaded,a32);
        loaded.layers[0].kv[0]^=71; loaded.layers[1].carry[0]+=1;
        check(state_hash(c.load(a,32,c.lookup(a).key))==expected,"branch load mutation isolation");
        check(owner->chain_stats(1).bytes==c.stats().bytes,"one budget authority");
        auto bad=std::make_shared<strix::PrefixCacheCheckpoint>(*entries[1]);
        auto wrong=std::make_shared<strix::PrefixCacheSegment>(*bad->segments.front()); wrong->tokens[0]^=1; bad->segments[0]=wrong;
        bool invoked=false;
        check(!owner->retain_chain(bad,[&](auto&,const auto&) {invoked=true;} ) && !invoked,"hash collision compares tokens");
        const auto retained=owner->counters().service_bytes;
        c.clear_ram();
        check(c.lookup(a).resume==0 && owner->inspect(1).empty(),"erased query directory despite read leases");
        check(owner->counters().service_bytes==retained,"held immutable leases stay charged");
        entries.clear(); branch.reset(); main.reset(); bad.reset(); wrong.reset();
        check(c.stats().bytes==0 && owner->counters().service_bytes==other.stats().bytes,"release credits once");
        const auto other_hash=state_hash(other.state(z,32,other.lookup(z).key));
        for(int arm=0;arm<5;++arm) {
            fail_after=arm;
            try { (void)c.insert(a32,32); check(false,"allocation failure required"); } catch(const std::bad_alloc&) {}
            fail_after=-1;
            check(owner->inspect(1).empty() && c.stats().bytes==0 && owner->counters().reserved_bytes==0,"failed capture rollback");
            check(owner->counters().service_bytes==other.stats().bytes,"partial allocation credits released");
        }
        check(state_hash(other.state(z,32,other.lookup(z).key))==other_hash,"failure isolated from second scope");
        check(c.insert(a32,32),"clean retry");
        for(unsigned arm=0;arm<6;++arm) {
            auto pressure=a; pressure[0]+=1000+arm;
            check(c.insert(image(pressure,32),500+arm),"scoped pressure insertion");
            check(state_hash(other.load(z,32,other.lookup(z).key))==other_hash,"pressure cannot evict another scope");
            check(owner->counters().service_bytes==c.stats().bytes+other.stats().bytes,"physical branch bytes charged once across scopes");
        }
    }
    check(alive==0 && physical==0 && owner->counters().service_bytes==0,"all physical allocations released");
    // Exact snapshot LRU. Looking up does not touch; use and duplicate do.
    o.budget_bytes=2*(ds::PrefixCache::snapshot_bytes(a8,8));
    ds::PrefixCache c(o,owner);
    auto x=a; x[0]+=33; auto y=a; y[0]+=44;
    check(c.insert(a8,1) && c.insert(image(x,8),2),"two LRU entries");
    auto held=owner->query(3,[&](const auto& cp) { return cp.tokens()==a8.tokens ? 1U:0U; },false);
    check(bool(held),"lookup read lease");
    (void)c.lookup(a);
    check(c.insert(image(y,8),3) && c.lookup(a).resume==0 && c.lookup(x).resume==8,"lookup no-touch victim");
    check(held->tokens()==a8.tokens && !owner->chain_checkpoint(3,held->digest) &&
        !owner->chain_segment(3,held->segments.front()->digest),"evicted lease readable after query invalidation");
    check(owner->counters().service_bytes==c.stats().bytes+ds::PrefixCache::snapshot_bytes(a8,8),"evicted lease credits remain inside service budget");
    held.reset(); check(owner->counters().service_bytes==c.stats().bytes,"evicted lease final release");
    auto q=c.lookup(x); auto state=c.load(x,q.resume,q.key); c.use(state,q.key);
    check(c.insert(a8,4) && c.lookup(y).resume==0 && c.lookup(x).resume==8,"use touch victim");
    check(owner->chain_stats(3).evicted_snapshots==2,"core eviction selection used");
    auto win=image(a,8); win.layers={win.layers.back()}; win.layers[0].ratio=0;
    ds::PrefixCache window(o,owner);
    check(window.insert(win,9),"metadata-only block with window ownership");
    exact_state(window.state(a,8,window.lookup(a).key),win);

    ds::PrefixCacheOptions bounded=o; bounded.budget_bytes=cap;
    ds::PrefixCache rollback(bounded,owner);
    const auto base_charge=owner->counters().service_bytes;
    check(rollback.insert(a8,1),"rollback ancestor admitted");
    fail_after=1;
    try { (void)rollback.insert(delta(a32,8),2,8); check(false,"pinned delta allocation failure required"); }
    catch(const std::bad_alloc&) {}
    fail_after=-1;
    check(rollback.stats().bytes==0 && rollback.stats().evicted_snapshots==1 && rollback.stats().evicted_blocks==1,
        "failed pinned capture releases ancestor after selected snapshot eviction");
    check(owner->counters().service_bytes==base_charge && owner->counters().reserved_bytes==0,"failed pinned capture releases all credits");
    check(!rollback.insert(delta(a32,8),2,8) && rollback.insert(a32,2),"missing ancestry refuses, full retry succeeds");

    // Independent rank images have the same shapes and protocol, different bytes.
    const auto rank1=[](ds::AttentionState s) {
        for(auto& l:s.layers) {
            for(auto* rows:{&l.kv,&l.idx_keys,&l.ring}) for(auto& v:*rows) v^=0x55;
            for(auto& v:l.carry) v+=2.0F;
        }
        return s;
    };
    ds::PrefixCache left(bounded),right(bounded);
    const auto paired=[&](const ds::AttentionState& s,unsigned from) {
        auto r=rank1(s);
        check(left.insert(s,701,from)==right.insert(r,701,from),"paired admission determinism");
        const auto xq=left.lookup(a),yq=right.lookup(a);
        check(xq.resume==yq.resume && xq.key==yq.key && xq.shared==yq.shared && xq.next_token==yq.next_token &&
            xq.logits_fnv==yq.logits_fnv && xq.disk==yq.disk && xq.disk_bytes==yq.disk_bytes,"paired lookup frontier determinism");
        const auto xs=left.stats(),ys=right.stats();
        check(xs.bytes==ys.bytes && xs.blocks==ys.blocks && xs.snapshots==ys.snapshots && xs.lookups==ys.lookups &&
            xs.hits==ys.hits && xs.hit_tokens==ys.hit_tokens && xs.inserts==ys.inserts && xs.refused==ys.refused &&
            xs.evicted_snapshots==ys.evicted_snapshots && xs.evicted_blocks==ys.evicted_blocks,"paired RAM call/stat determinism");
        if(xq.resume) {
            auto expected=image(a,xq.resume,xq.next_token==ds::kPrefixUnknownNextToken);
            exact_state(left.load(a,xq.resume,xq.key),expected);
            exact_state(right.load(a,yq.resume,yq.key),rank1(expected));
        }
    };
    paired(a8,0); paired(image(a,16),0); paired(image(a,24,true),0);
    paired(delta(a32,16),16); paired(image(b,32),0); paired(a32,0);
    auto no_promote=o; no_promote.promote=false;
    ds::PrefixCache wrong_boundary(no_promote);
    check(wrong_boundary.insert(a8,1) && wrong_boundary.insert(image(x,8),2),"use boundary setup");
    wrong_boundary.use(image(a,16),ds::prefix_key(a,8,1));
    check(wrong_boundary.insert(image(y,8),3) && wrong_boundary.lookup(a).resume==0 &&
        wrong_boundary.lookup(x).resume==8,"use with a different boundary preserves original no-touch behavior");
    auto large=a8; large.layers[1].carry.resize((1U<<20U)+1,0.25F);
    auto unlimited=o; unlimited.budget_bytes=0;
    ds::PrefixCache ram_vectors(unlimited);
    check(ram_vectors.insert(large,8),"RAM vector lengths retain their original unrestricted contract");
    exact_state(ram_vectors.load(a,8,ram_vectors.lookup(a).key),large);
    ds::PrefixCache release_order(o,owner);
    check(release_order.insert(a32,1),"release order capture");
    auto saved=owner->inspect(6).front();
    std::vector<const std::byte*> expected_release{saved->fixed.data.get()};
    for(auto p=saved->segments.front();p;p=p->ancestor) expected_release.push_back(p->payload.data.get());
    saved.reset(); record_release=true;
    release_order.clear_ram(); record_release=false;
    check(released==expected_release,"actual allocator releases snapshot then leaf-to-root blocks");
}
#endif
int main(int argc,char** argv) {
    try {
        if(argc>1 && std::string(argv[1])=="--fail-sentinel") { std::cerr<<"DS41 prefix owner FAIL: deliberate active-check sentinel\n"; return 1; }
        original_trace();
#ifdef STRIX_DS41_OWNER_ADOPTED
        if(argc<2 || std::string(argv[1])!="--trace") adoption_checks();
#endif
        return 0;
    } catch(const std::exception& e) { std::cerr<<"DS41 prefix owner FAIL: "<<e.what()<<'\n'; return 1; }
}
