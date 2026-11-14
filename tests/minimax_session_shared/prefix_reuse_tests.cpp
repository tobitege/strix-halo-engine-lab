// Executes the production owner/planner/backend/control seams. Each rank owns
// independent row images; suffix counters measure physical fixture writes.
#define main minimax_frozen_policy_entry
#include "../minimax_lead_tests.cpp"
#undef main
#include <stdexcept>
#include "minimax/serve/session_service.hpp"
#include "strix/minimax/session_cache.hpp"
namespace ss = strix::session;
static bool done(const std::shared_ptr<sv::GenerationTicket>& t) { bool value=false; (void)t->wait(std::chrono::milliseconds(0),value,false);return value; }
static void need(bool b,const char* why) { if(!b)throw std::runtime_error(why); }
template<class F> static void rejects(F f) { bool bad=false;try{f();}catch(...){bad=true;}need(bad,"expected refusal"); }
struct Rank : FakeEngine {
    std::array<std::vector<unsigned>,8> image;
    std::vector<unsigned> suffix;
    unsigned start=0, writes=0;
    bool failed=false, unaligned=false;  // unaligned: conversation mode agreed, a resident restart may open mid-chunk
    void cancel_all() { if(active.open)prefill_cancel(active.id);if(parked.open)prefill_cancel(parked.id); }
    void require_active(std::uint64_t id,unsigned bank) { need(active.open && active.id==id && active.bank==bank,"peer active task"); }
    void prefill_check(std::uint64_t id,unsigned next) {need(!active.open && active.id==id && active.next==next,"peer completed token");}
    std::uint64_t prefill_reuse_rows() const { return 4; }
    void prefill_begin(unsigned s,std::span<const unsigned> p,bool c,std::uint64_t id,
        std::uint64_t parent=0,std::uint64_t generation=0,std::uint64_t resident=0) {
        need(!resident || (!c && !parent && (unaligned || resident%4==0) && image[s].size()>=resident),"rank validates physical boundary");
        start=static_cast<unsigned>(resident);suffix.assign(p.begin(),p.end());
        FakeEngine::prefill_begin(s,p,c,id,parent,generation);
    }
    bool prefill_unit(unsigned layers,unsigned* next) {
        const auto slot=active.slot;
        const auto done=FakeEngine::prefill_unit(layers,next);
        if(failed)throw std::runtime_error("rank write failure");
        if(done) {
            image[slot].resize(start+suffix.size());
            std::copy(suffix.begin(),suffix.end(),image[slot].begin()+start);
            writes+=static_cast<unsigned>(suffix.size());
        }
        return done;
    }
    std::vector<unsigned> step(std::span<const mm::Row> rows,unsigned append) {
        for(auto r:rows) { image[r.slot].resize(static_cast<std::size_t>(r.pos+1));image[r.slot][r.pos]=r.token+100000; }
        return FakeEngine::step(rows,append);
    }
};
struct Pair : Rank {
    Rank other;
    ms::Control begin;
    bool voted=true;
    void exchange_bytes(const void* out,void* in,std::size_t n) {
        if(n!=512){std::memcpy(in,out,n);return;}
        ms::Control c;std::memcpy(&c,out,sizeof(c));
        need(c.active==other.active && c.parked==other.parked,"independent peer checkpoint");
        FakeEngine::exchange_bytes(out,in,n);
        if(!c.command)return;
        if(c.command==2 || c.command==10) {begin=c;return;}
        if(c.command==14)return;
        auto saved=FakeClock::advance;FakeClock::advance=[](double){};
        try { (void)ms::replay_command(c,other,other,[](std::size_t){return std::vector<unsigned>{};}); }
        catch(...) {FakeClock::advance=saved;throw;}
        FakeClock::advance=saved;
    }
    void send_prompt(std::vector<unsigned>& p) {
        auto receive=[&](std::size_t n){need(n==p.size(),"wire suffix count");return p;};
        if(!ms::replay_resident_begin(begin,other,other,receive,voted,other.unaligned))
            (void)ms::replay_command(begin,other,other,receive);
    }
    void invalidate_residents() {}
};
struct Hello : FakeEngine {
    unsigned vote=0;std::uint64_t alignment=4;
    void exchange_bytes(const void* o,void* i,std::size_t n) {
        FakeEngine::exchange_bytes(o,i,n);ms::Control c;std::memcpy(&c,i,sizeof(c));
        c.pflags=vote;c.dpos[0]=alignment;std::memcpy(i,&c,sizeof(c));
    }
};
struct Fixture {
    Pair pair;sv::GenerationQueue queue;ml::Shared shared;std::atomic<bool> stop{false};
    std::unique_ptr<ml::Lead<Pair,Pair,FakeClock>> lead;
    Fixture(bool enabled=true,bool draft=false,ms::SessionService* service=nullptr,bool conversation=false) {
        FakeClock::value={};FakeClock::advance=[](double n){FakeClock::value+=std::chrono::milliseconds(static_cast<int>(n));};
        sv::GenerationQueue::test_now=FakeClock::now;
        shared.prefix_reuse=enabled;shared.conversation=enabled && conversation;pair.unaligned=pair.other.unaligned=shared.conversation;
        shared.prefix_identity={"loaded-weights","profile-tp2","prefill-v1",2};
        shared.policy.sched=ms::Sched::off;shared.policy.nest=false;shared.policy.lookup=false;
        if(draft){pair.dd=&pair;shared.policy.dspark_mode=ml::ds::Mode::on;pair.taps=true;}
        lead=std::make_unique<ml::Lead<Pair,Pair,FakeClock>>(pair,pair,queue,shared,4096,draft,stop,
            std::function<std::string(const std::vector<unsigned>&)>{},service);
    }
    auto submit(std::vector<unsigned> p,sv::CacheRetention retention=sv::CacheRetention::server_policy,unsigned output=1) {
        sv::GenerationJob j;j.prompt=std::move(p);j.max_tokens=output;j.cache_retention=retention;
        return queue.submit(j);
    }
    void finish(const std::shared_ptr<sv::GenerationTicket>& t) {
        for(unsigned i=0;i<300 && !done(t);++i)lead->runtime().step(false);
        need(done(t) && t->error().empty(),"owner completes request");
        need(pair.writes==pair.other.writes && pair.image==pair.other.image,"both ranks really write the suffix");
    }
};
// Uses the same service-owned factories as NativeAssets. No frontend hook or
// second cache owner is involved; only native execution assets are fixtures.
struct OwnerAssets final : ms::ServiceAssets {
    Pair pair;
    std::atomic<bool> stop{false};
    std::unique_ptr<strix::minimax::session::Cache> named;
    std::unique_ptr<ml::Lead<Pair, Pair, FakeClock>> owner;
    OwnerAssets() {
        FakeClock::value={}; FakeClock::advance=[](double n){FakeClock::value+=std::chrono::milliseconds(static_cast<int>(n));};
        sv::GenerationQueue::test_now=FakeClock::now;
        shared.prefix_reuse=true; shared.prefix_identity={"loaded-weights","profile-tp2","prefill-v1",2};
        shared.policy.sched=ms::Sched::off; shared.policy.nest=false; shared.policy.lookup=false;
        named=with_cache_owner(32ULL<<20, [&](const auto& pooled) {
            return std::make_unique<strix::minimax::session::Cache>("fixture",0,1,32ULL<<20,
                std::filesystem::path{}, [](auto b){return std::vector<std::uint8_t>(b.begin(),b.end());},8,UINT64_MAX,pooled);
        });
        owner=make_lead<Pair,Pair,FakeClock>(pair,pair,4096,false,stop);
    }
    sv::SessionRuntime& runtime() override { return owner->runtime(); }
    int follow() override { return 0; }
};
int main() try {
    {
        // The 6000 -> 9000 / 8192 threshold case scaled to the fixture's four-row chunks.
        Fixture f; f.shared.prefill_int4_regime = [](std::uint64_t rows) { return rows >= 8; };
        auto a = f.submit({1,2,3,4,5,6}); f.finish(a);
        auto b = f.submit({1,2,3,4,5,6,7,8,9}); f.finish(b);
        need(b->stats().cached_tokens == 0, "threshold crossing must decline the old four-row prefix");
        auto c = f.submit({1,2,3,4,5,6,7,8,9,10}); f.finish(c);
        need(c->stats().cached_tokens == 8, "same effective regime still reuses its resident prefix");
    }
    {
        OwnerAssets assets;
        need(assets.named->owner()==assets.runtime().cache_owner() && assets.cache_owner==assets.named->owner(),
            "production service factories inject one named/resident owner");
        sv::GenerationJob j; j.prompt={1,2,3,4,5,6,7,8,9}; j.max_tokens=1;
        const auto t=assets.queue.submit(j);
        for(unsigned i=0;i<300 && !done(t);++i)assets.runtime().step(false);
        need(done(t) && t->error().empty() && assets.runtime().cache_entries()>0,"production owner retains resident certificate");
        need(assets.cache_owner->counters().retention_metadata_bytes>0 && assets.cache_owner->counters().immutable_bytes==0,
            "resident metadata only; slot KV is not a second copy");
        strix::minimax::session::Checkpoint checkpoint;
        checkpoint.id="named"; checkpoint.history={3,4,5};
        checkpoint.frontier.consumed=checkpoint.frontier.accepted=3;
        checkpoint.frontier.emitted=1; checkpoint.frontier.pending=6;
        checkpoint.sections=strix::minimax::session::split_state(std::vector<std::uint8_t>(3*76800,0),3);
        checkpoint.sections.push_back({"pending.logits",false,std::vector<std::uint8_t>(100096*4,0)});
        assets.named->capture(std::move(checkpoint));
        need(assets.named->resume("named",false).history==std::vector<unsigned>({3,4,5}),
            "production named codec restores from common pool ownership");
        need(assets.cache_owner->counters().encoded_host_bytes>0 && assets.cache_owner->counters().session_entries==2 &&
             assets.runtime().cache_entries()==1 && assets.named->stats().entries==1,
            "real named encoded checkpoint and resident certificate coexist in one service directory");
    }

    for(bool local:{false,true})for(bool peer:{false,true}) {
        Hello h;h.vote=peer?ms::kHelloPrefixVote:0;bool agreed=true;
        need(!ms::session_hello(h,false,local,&agreed,4) && agreed==(local&&peer),"both rank votes required");
    }
    { Hello h;h.vote=2;h.alignment=8;bool agreed=true;ms::session_hello(h,false,true,&agreed,4);need(!agreed,"alignment vote mismatch");
      h.alignment=0;ms::session_hello(h,false,true,&agreed,4);need(!agreed,"empty alignment vote"); }
    {
        Rank rank;ms::Control c;c.command=2;c.task_id=1;c.slot_generation=1;c.pstart=4;c.plen=1;
        auto receive=[](std::size_t n){return std::vector<unsigned>(n,7);};
        rejects([&]{ms::replay_resident_begin(c,rank,rank,receive,false);});
        c.pstart=3;rejects([&]{ms::replay_resident_begin(c,rank,rank,receive,true);});
        c.pstart=4;c.parent_task_id=7;rejects([&]{ms::replay_resident_begin(c,rank,rank,receive,true);});
        c.parent_task_id=0;c.pflags=1;rejects([&]{ms::replay_resident_begin(c,rank,rank,receive,true);});
        c.pflags=0;c.plen=0;rejects([&]{ms::replay_resident_begin(c,rank,rank,receive,true);});
        c.plen=1;rejects([&]{ms::replay_resident_begin(c,rank,rank,[](auto){return std::vector<unsigned>{};},true);});
        need(rank.writes==0,"malformed controls refuse before row work");
    }
    const std::vector<unsigned> p{1,2,3,4,5,6,7,8,9};
    {
        Fixture f;auto a=f.submit(p);f.finish(a);need(a->stats().cached_tokens==0 && f.pair.writes==9 && f.pair.image[0]==p,"cold writes all expected prompt rows");
        auto b=f.submit(p);f.finish(b);need(b->stats().cached_tokens==8 && b->stats().prefill_ms<a->stats().prefill_ms && f.pair.writes==10 && f.pair.image[0]==p,"full match preserves exact prefix and writes final token");
        auto changed=p;changed[6]=77;auto c=f.submit(changed);f.finish(c);
        need(c->stats().cached_tokens==4 && f.pair.writes==15 && f.pair.image[0]==changed,"aligned suffix yields independent expected prompt image");
        auto mismatch=p;mismatch[0]=99;auto d=f.submit(mismatch);f.finish(d);need(!d->stats().cached_tokens,"changed first token cold");
        auto short_prompt=f.submit({99});f.finish(short_prompt);need(!short_prompt->stats().cached_tokens,"one token cold");
        auto stats=f.lead->runtime().cache_stats();need(stats.hits==2 && stats.misses==3 && stats.entries==0,"one shared index and effective accounting");
    }
    {
        Fixture f;auto a=f.submit(p);f.finish(a);f.pair.other.failed=true;auto b=f.submit(p);
        rejects([&]{f.lead->runtime().step(false);});
        need(f.lead->runtime().cache_entries()==0,"failed independent peer cannot republish residency");
        f.pair.other.failed=false;
    }
    {
        Fixture f;auto a=f.submit({1,2,3},sv::CacheRetention::server_policy,2);f.finish(a);
        need(f.lead->runtime().cache_entries()==0,"decode cannot complete a reusable prefill group");
        auto b=f.submit({1,2,3,100,9});f.finish(b);
        need(!b->stats().cached_tokens,"generated token matches do not become prefill hits");
    }
    {Fixture f(false);auto a=f.submit(p);f.finish(a);auto b=f.submit(p);f.finish(b);need(!b->stats().cached_tokens && f.pair.writes==18,"OFF stays cold");}
    {Fixture f;auto a=f.submit(p);f.finish(a);auto b=f.submit(p,sv::CacheRetention::no_store);f.finish(b);
     need(b->stats().cached_tokens==8 && f.lead->runtime().cache_entries()==0,"no-store consumes but never retains descriptor");
     auto c=f.submit(p);f.finish(c);need(!c->stats().cached_tokens,"no-store leaves no new resident index");}
    {Fixture f(true,true);auto a=f.submit(p);f.finish(a);auto b=f.submit(p);f.finish(b);
     need(!b->stats().cached_tokens && f.pair.writes==18,"unqualified drafter context cold");}
    {
        // Direct production runtime/cache/adapter seam fills two FREE slots,
        // then proves that the owner leases the longest rather than first slot.
        Pair pair;ms::SessionBackend<Pair> backend(pair);ms::SessionEngine engine(backend,8,16384,[](auto){},[&]{return pair.active;});
        ss::KvStore store(8,2,engine.storage_hooks());engine.attach(store);sv::GenerationQueue q;sv::SessionHooks hooks;
        hooks.control_turn=[](auto&){};hooks.identity={"weights","profile","rows",2};
        sv::SessionRuntime runtime(q,engine,store,{},hooks);
        std::uint64_t task=0;
        auto open=[&](std::vector<unsigned> prompt){sv::GenerationJob j;j.prompt=prompt;j.max_tokens=1;q.submit(j);need(runtime.reserve(sv::QueueSelection::fifo),"reserve");return runtime.activate(4);};
        auto run=[&](sv::SessionLease lease,const std::vector<unsigned>& prompt){
            // Independently replay the same wire start on rank 1.
            const auto reused=lease.stats.cached_tokens;const auto suffix=std::span(prompt).subspan(static_cast<std::size_t>(reused));
            pair.other.prefill_begin(lease.state.slot,suffix,false,++task,0,lease.state.generation,reused);
            engine.begin(lease.state,suffix,false,task,0,reused);unsigned next=0;
            while(!engine.unit(lease.state,60,&next)){};pair.other.prefill_unit(60,&next);
        };
        auto a=open(p);run(a,p);auto longer=p;longer.insert(longer.end(),{10,11,12,13});auto b=open(longer);run(b,longer);
        runtime.complete(a.state,{},8);runtime.complete(b.state,{},12);
        auto c=open(longer);need(c.state.slot==b.state.slot && c.stats.cached_tokens==12,"longest FREE slot chosen");
        need(runtime.cache_entries()==1,"active resident entry removed before mutation");
        rejects([&]{store.acquire_resident(b.state,std::span(longer).first(4));});
        runtime.complete(c.state,"failed peer",0);
        auto d=open(longer);need(d.stats.cached_tokens==8,"failed peer cannot retain active slot");
        store.invalidate(d.state);runtime.complete(d.state,{},8);
        auto e=open(longer);need(!e.stats.cached_tokens,"poisoned state never reusable");runtime.complete(e.state);
        runtime.invalidate_residents();need(runtime.cache_entries()==0,"admin invalidates all resident entries");
    }
    {
        Fixture f;auto a=f.submit(p);f.finish(a);
        // A reserved reuse cancelled after begin retains ONLY old committed rows.
        f.shared.policy.sched=ms::Sched::interleave;++f.shared.policy.epoch;
        auto b=f.submit(p,sv::CacheRetention::server_policy,100);f.lead->runtime().step(false);b->cancel();
        for(unsigned i=0;i<100 && !done(b);++i)f.lead->runtime().step(false);
        need(done(b),"cancel completes");
        auto c=f.submit(p);f.finish(c);need(c->stats().cached_tokens<=8,"cancel never admits generated rows");
    }
    {
        // Session/admin invalidation precedes the callback's physical writes.
        ms::SessionService service;Fixture f(true,false,&service);
        auto a=f.submit(p);f.finish(a);need(f.lead->runtime().cache_entries()==1,"resident entry before admin");
        unsigned calls=0;
        service.execute=[&](const auto&,const auto&) {
            need(f.lead->runtime().cache_entries()==0,"admin invalidates before state work");++calls;
            for(auto& image:f.pair.image)image.clear();for(auto& image:f.pair.other.image)image.clear();
            return sv::Json::object();
        };
        auto task=std::make_shared<ms::SessionTasks::Task>();task->body="{}";
        service.tasks.tasks.push_back(task);service.tasks.pending=true;
        f.lead->runtime().step(false);need(calls==1,"production admin dispatch executed");
        auto b=f.submit(p);f.finish(b);need(!b->stats().cached_tokens,"admin physical overwrite forces cold");
    }
    {
        // Exercise the same adapter at an actual bounded partial prefill frontier.
        Rank rank;ms::SessionBackend<Rank> backend(rank);
        ms::SessionEngine engine(backend,8,16384,[&](auto){rank.cancel_all();},[&]{return rank.active;});
        ss::KvStore store(8,2,engine.storage_hooks());engine.attach(store);sv::GenerationQueue q;sv::SessionHooks hooks;
        hooks.control_turn=[](auto&){};hooks.identity={"loaded","arithmetic","rows",2};
        sv::SessionRuntime runtime(q,engine,store,{},hooks);std::uint64_t task=0;
        auto open=[&]{sv::GenerationJob j;j.prompt=p;j.max_tokens=1;q.submit(j);need(runtime.reserve(sv::QueueSelection::fifo),"reserve partial");return runtime.activate(4);};
        auto a=open();engine.begin(a.state,p,false,++task,0);unsigned next=0;engine.unit(a.state,60,&next);runtime.complete(a.state,{},8);
        auto b=open();need(b.stats.cached_tokens==8,"partial starts at certified reuse");
        engine.begin(b.state,std::span(p).subspan(8),false,++task,0,8);
        need(!engine.unit(b.state,1,&next) && store.consumed(b.state)==8,"partial never publishes suffix rows");
        runtime.complete(b.state,{},8);need(!rank.active.open,"release cancels actual partial task");
        auto c=open();need(c.stats.cached_tokens==8,"cancel retains exactly previous safe prefix");
        runtime.complete(c.state);auto d=open();need(!d.stats.cached_tokens,"ordinary release clears residency");
        engine.begin(d.state,p,false,++task,0);rank.failed=true;
        rejects([&]{engine.unit(d.state,60,&next);});runtime.complete(d.state,{},8);
        rank.failed=false;auto e=open();need(!e.stats.cached_tokens,"failed native seam cannot publish residency");runtime.complete(e.state);
    }
    {
        // Actual nested Lead admission refuses any resident boundary.
        Fixture f;auto seed=f.submit(p);f.finish(seed);
        f.shared.policy.sched=ms::Sched::interleave;f.shared.policy.nest=true;f.shared.nest_rows=512;++f.shared.policy.epoch;
        auto decoder=f.submit({31},sv::CacheRetention::server_policy,100);
        f.lead->runtime().step(false);
        FakeClock::advance(1);
        auto long_request=f.submit(std::vector<unsigned>(2000,44),sv::CacheRetention::server_policy,2);
        for(unsigned i=0;i<10 && !f.pair.active.open;++i)f.lead->runtime().step(false);
        need(f.pair.active.open,"bounded parent actually open");FakeClock::advance(1);
        auto child=f.submit(p);
        for(unsigned i=0;i<200 && !done(child);++i)f.lead->runtime().step(false);
        need(done(child),"nested child finishes");
        const auto nested=std::find_if(f.pair.controls.begin(),f.pair.controls.end(),[](const auto& c){return c.command==10;});
        need(nested!=f.pair.controls.end() && nested->pstart==0 && !child->stats().cached_tokens,"nested admission always cold");
        decoder->cancel();long_request->cancel();
        for(unsigned i=0;i<100 && f.lead->runtime().live();++i)f.lead->runtime().step(false);
    }
    {
        // Immutable checkpoints and residency compete in the SAME cache budget;
        // resident retention cannot bypass a protected completed-turn keep.
        ss::Hooks h;h.initialize=[](auto){};h.reset=[](auto){};h.release=[](auto){};
        h.view=[](auto,unsigned){return ss::RankState{std::make_shared<const unsigned>(17),4};};
        h.restore_prompt=[](auto,auto){return true;};
        h.capture=h.view;h.capture_prompt=h.view;h.restore=[](auto,unsigned,const auto&){};
        h.truncate=[](auto,unsigned,std::uint64_t){};h.resident_prefill_supported=true;
        ss::KvStore store(1,2,h);const auto lease=store.acquire();store.append_consumed(lease,p);
        ss::Identity identity{"weights","profile","sealed",2};
        ss::Policy policy;policy.completed_turns=true;policy.max_entries=2;policy.max_bytes=4096;
        ss::PrefixCache cache(policy);
        auto frozen=std::make_shared<ss::PrefixSnapshot>();frozen->identity=identity;
        frozen->state=store.capture_prompt(lease);frozen->attention_frontier=frozen->recurrent_frontier=p.size();
        frozen->turn=ss::TurnMetadata{"conversation","template","ordered-tp2","protected"};
        need(cache.admit(frozen),"protected immutable checkpoint admitted");
        const auto certificate=store.release_resident(lease,8);
        auto forged=p;forged[0]=999;
        need(!cache.admit_resident(store,lease,std::span(forged).first(8),identity),"forged resident token key refused");
        rejects([&]{(void)store.acquire_resident(lease,std::span(forged).first(4));});
        need(cache.admit_resident(store,lease,std::span(p).first(8),identity) && cache.size()==2,"resident shares checkpoint index");
        need(cache.lookup_resident(store,p,identity,4).rows==8,"resident hit distinct from snapshot");
        auto other_profile=identity;other_profile.profile="changed";
        need(!cache.lookup_resident(store,p,other_profile,4).entry &&
             !cache.lookup_resident(store,p,ss::Identity{},4).entry,"unknown or changed loaded identity misses");
        auto ordinary=std::make_shared<ss::PrefixSnapshot>(*frozen);ordinary->turn.reset();
        need(cache.admit(ordinary) && cache.size()==2 && !cache.lookup_resident(store,p,identity,4).entry,"one eviction loop evicts unpinned residency");
        need(cache.lookup_turn(p,identity,*frozen->turn)!=nullptr,"protected payload survives resident pressure");
        ss::Policy no_store;no_store.store=false;ss::PrefixCache refused(no_store);
        need(!refused.admit_resident(store,lease,std::span(p).first(8),identity),"cache no-store policy refuses residents");
        const auto fresh=store.acquire();
        need(!store.resident_valid(lease,4) && !cache.lookup_resident(store,p,identity,4).entry,"new generation invalidates old residency");
        store.reset(fresh);store.release(fresh);
        need(frozen->state.sealed() && *std::static_pointer_cast<const unsigned>(frozen->state.ranks[0].owner)==17,"frozen provenance/payload survives physical reuse");
        need(cache.unpin("protected"),"pin controls preserved");
    }
    {
        // Reply-extended resident records through the production runtime/store/engine seam (MiniMax adapter), fake ranks.
        Pair pair;pair.unaligned=pair.other.unaligned=true;ms::SessionBackend<Pair> backend(pair);
        ms::SessionEngine engine(backend,8,16384,[](auto){},[&]{return pair.active;});
        ss::KvStore store(8,2,engine.storage_hooks());engine.attach(store);sv::GenerationQueue q;sv::SessionHooks hooks;
        hooks.control_turn=[](auto&){};hooks.identity={"weights","profile","rows",2};
        sv::SessionRuntime runtime(q,engine,store,{},hooks);
        std::uint64_t task=0;
        auto open=[&](std::vector<unsigned> prompt,std::uint64_t alignment){sv::GenerationJob j;j.prompt=prompt;j.max_tokens=1;q.submit(j);
            need(runtime.reserve(sv::QueueSelection::fifo),"reserve");return runtime.activate(alignment);};
        auto prefill=[&](sv::SessionLease& lease,const std::vector<unsigned>& prompt){
            const auto reused=lease.stats.cached_tokens;const auto suffix=std::span(prompt).subspan(static_cast<std::size_t>(reused));
            pair.other.prefill_begin(lease.state.slot,suffix,false,++task,0,lease.state.generation,reused);
            engine.begin(lease.state,suffix,false,task,0,reused);unsigned next=0;
            while(!engine.unit(lease.state,60,&next)){};pair.other.prefill_unit(60,&next);
        };
        // Two committed reply tokens: commit() consumes the previous anchor, so rows = prompt + 1 and one pending token.
        auto reply=[&](sv::SessionLease& lease,unsigned slot,std::uint64_t rows){
            engine.commit(lease.state,50);engine.commit(lease.state,51);
            pair.image[slot].resize(rows);pair.other.image[slot].resize(rows);
        };
        auto a=open(p,4);prefill(a,p);const auto slot=a.state.slot;reply(a,slot,10);
        need(store.consumed(a.state)==10 && store.pending(a.state)==51u,"decode leaves a pending anchor outside the consumed rows");
        // Default completion keeps today's refusal: decode rows cannot certify a prefill prefix (slot quarantined).
        runtime.complete(a.state,{},10);
        need(runtime.cache_entries()==0,"default completion refuses decode rows");
        auto probe=open(p,4);need(probe.state.slot!=slot || !probe.stats.cached_tokens,"quarantined slot never reused");runtime.complete(probe.state);
        // Explicit claim: certified at the exact frontier, with the prompt as the head of the history.
        auto b=open(p,4);prefill(b,p);const auto slot_b=b.state.slot;reply(b,slot_b,10);
        runtime.complete(b.state,{},sv::ResidentClaim{10,1});
        need(runtime.cache_entries()==1,"reply claim keeps one resident record");
        auto st=runtime.cache_stats();
        need(st.resident_kept==1 && st.resident_reply_kept==1 && st.resident_claim_refused==0,"kept counters");
        auto next=p;next.insert(next.end(),{50,60,61});
        auto c=open(next,1);
        need(c.state.slot==slot_b && c.stats.cached_tokens==10 && c.resident_reply,"exact-frontier lookup restarts at prompt+reply rows");
        need(runtime.cache_entries()==0,"the record is removed before the slot is mutated");
        st=runtime.cache_stats();
        need(st.resident_hits==1 && st.resident_hit_rows==10 && st.resident_reply_hits==1,"hit counters");
        prefill(c,next);
        // The new lane owns the slot; its own claim replaces the old record (tokens = new prompt + reply).
        reply(c,slot_b,static_cast<std::uint64_t>(next.size()+1));
        runtime.complete(c.state,{},sv::ResidentClaim{next.size()+1,1});
        need(runtime.cache_entries()==1 && runtime.cache_stats().resident_reply_kept==2,"second turn keeps a record of its own");
        // A prompt equal to the record leaves one suffix token; one that diverges inside the reply reuses only the head.
        auto same=next;same.push_back(50);
        auto d=open(same,1);need(d.state.slot==slot_b && d.stats.cached_tokens==same.size()-1 && !d.resident_reply,
                                 "at least one suffix token stays");
        prefill(d,same);runtime.complete(d.state);
        // Failed lanes, cancelled work and a frontier that is not the store's keep nothing.
        auto e=open(p,1);prefill(e,p);const auto slot_e=e.state.slot;reply(e,slot_e,10);
        runtime.complete(e.state,"failed peer",sv::ResidentClaim{10,1});
        need(runtime.cache_entries()==0,"failed lane keeps no reply record");
        auto f=open(p,1);prefill(f,p);reply(f,f.state.slot,10);
        runtime.complete(f.state,{},sv::ResidentClaim{9,0});
        need(runtime.cache_entries()==0,"a claim off the store frontier is refused");
        auto g=open(p,1);prefill(g,p);reply(g,g.state.slot,10);
        runtime.complete(g.state,{},sv::ResidentClaim{11,2});
        need(runtime.cache_entries()==0 && runtime.cache_stats().resident_claim_refused==3,"a claim beyond consumed rows is refused");
        // Slot pressure: a record never holds a slot back. All eight slots live => the record is gone, nothing starves.
        auto h=open(p,1);prefill(h,p);reply(h,h.state.slot,10);runtime.complete(h.state,{},sv::ResidentClaim{10,1});
        need(runtime.cache_entries()==1,"record parked on a free slot");
        std::vector<sv::SessionLease> live;
        for(unsigned i=0;i<7;++i){std::vector<unsigned> other(9,200+i);  // slot of the default-refused decode rows stays quarantined
            live.push_back(open(other,1));
            need(!live.back().stats.cached_tokens,"unrelated prompt reuses nothing");}
        need(runtime.live()==7 && runtime.cache_entries()==0,"every usable slot live: the record is gone, nothing starves");
        for(auto& l:live)runtime.complete(l.state);
        st=runtime.cache_stats();
        need(st.resident_hits==2 && st.resident_reply_hits==1 && st.resident_hit_rows==22 && st.resident_misses>=5,"stats keep counting");
        runtime.invalidate_residents();need(runtime.cache_entries()==0,"admin invalidation drops reply records too");
    }
    {
        // Conversation mode through the whole Lead: three turns, each restarts at the previous reply-end frontier on both
        // ranks (the rank images prove the rows really exist on rank 1), prefilling only the new suffix.
        Fixture f(true,false,nullptr,true);
        std::vector<unsigned> t1{1,2,3,4,5,6,7,8,9};
        auto a=f.submit(t1,sv::CacheRetention::server_policy,3);f.finish(a);need(!a->stats().cached_tokens && f.pair.writes==9,"turn 1 is cold");
        auto t2=t1;t2.insert(t2.end(),{100,100,100,20,21});
        auto b=f.submit(t2,sv::CacheRetention::server_policy,3);f.finish(b);
        need(b->stats().cached_tokens==11 && f.pair.writes==9+3,"turn 2 restarts at prompt + reply rows mid-chunk, writes only the suffix");
        auto t3=t2;t3.insert(t3.end(),{100,100,100,30});
        auto c=f.submit(t3,sv::CacheRetention::server_policy,3);f.finish(c);
        need(c->stats().cached_tokens==16 && f.pair.writes==9+3+2,"turn 3 restarts at turn 2's reply-end frontier");
        auto st=f.lead->runtime().cache_stats();
        need(st.resident_reply_hits==2 && st.resident_hits==2 && st.resident_hit_rows==27 && st.resident_reply_kept==3,"reply-extended counters");
        // Same conversation with the switch OFF on both ranks: chunk-aligned reuse only, a 9-14 row prompt reuses nothing.
        Fixture g(true,false,nullptr,false);
        auto x=g.submit(t1,sv::CacheRetention::server_policy,3);g.finish(x);
        auto y=g.submit(t2,sv::CacheRetention::server_policy,3);g.finish(y);
        need(!y->stats().cached_tokens || y->stats().cached_tokens%4==0,"conversation OFF restarts only on aligned boundaries");
        need(g.lead->runtime().cache_stats().resident_reply_hits==0 && g.lead->runtime().cache_stats().resident_reply_kept==0,
             "conversation OFF never keeps a reply record");
    }
    std::cout<<"PASS resident prefix owner/protocol/rank suffix checks\n";return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
