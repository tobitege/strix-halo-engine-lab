// Production adapter/store and owner tests, with independently executed ranks.
#define main minimax_current_policy_suite
#include "../minimax_lead_tests.cpp"
#undef main
#include <cassert>
#include <future>

namespace ss = strix::session;
static void need(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
template<class F> static void rejects(F f) {
    bool threw = false; try { f(); } catch (const std::exception&) { threw = true; }
    need(threw, "operation must fail closed");
}
struct PairRunner {
    FakeEngine rank[2];
    unsigned units[2]{}, steps[2]{}, releases[2]{};
    bool fail = false;
    bool freeze = false;
    void prefill_begin(unsigned s, std::span<const unsigned> p, bool c,
                       std::uint64_t t, std::uint64_t parent, std::uint64_t g) {
        for (auto& r : rank) r.prefill_begin(s,p,c,t,parent,g);
    }
    bool prefill_unit(unsigned layers, unsigned* next) {
        if (freeze) return false;
        unsigned tokens[2]{}; bool done[2]{};
        for (unsigned r=0;r<2;++r) { ++units[r]; done[r]=rank[r].prefill_unit(layers,&tokens[r]); }
        if (fail) throw std::runtime_error("paired operation failed");
        need(done[0]==done[1] && tokens[0]==tokens[1],"independent prefill disagreement");
        *next=tokens[0]; return done[0];
    }
    std::vector<unsigned> step(std::span<const mm::Row> rows, unsigned append) {
        std::vector<unsigned> result[2];
        for (unsigned r=0;r<2;++r) { ++steps[r]; result[r]=rank[r].step(rows,append); }
        need(result[0]==result[1],"independent row disagreement"); return result[0];
    }
    ml::Drafts draft(unsigned s,unsigned a,std::uint64_t p) { return rank[0].draft(s,a,p); }
    void synchronize() { for(auto& r:rank)r.synchronize(); }
    void release(ss::KvHandle h) {
        for(unsigned r=0;r<2;++r) {
            ++releases[r];
            if(rank[r].active.open && rank[r].active.slot==h.slot)rank[r].prefill_cancel(rank[r].active.id);
        }
    }
};
struct WindowEngine : FakeEngine {
    ml::Drafts draft(unsigned slot,unsigned,std::uint64_t pos) {
        ml::Drafts d;
        for(unsigned i=0;i<d.tokens.size();++i) {
            const auto p=pos+i;
            d.tokens[i]=100+slot+static_cast<unsigned>(p%27==13 ? 7 : p%9%4);
            d.confidence[i]=1;
        }
        return d;
    }
};
struct FaultEngine : FakeEngine {
    unsigned steps=0;
    std::vector<unsigned> step(std::span<const mm::Row> rows,unsigned append) {
        if(++steps==2)throw std::runtime_error("injected grouped failure");
        return FakeEngine::step(rows,append);
    }
};
struct SmallContextEngine : FakeEngine { unsigned capacity() const { return 128; } };
int main() try {
    FakeClock::advance=[](double){};
    sv::GenerationQueue::test_now=FakeClock::now;
    PairRunner pair; ms::SessionBackend<PairRunner> backend(pair);
    ms::SessionEngine engine(backend,2,16384,[&](auto h){pair.release(h);},[&]{return pair.rank[0].prefill_progress();});
    ss::KvStore store(2,2,engine.storage_hooks()); engine.attach(store);
    const auto a=store.acquire(), b=store.acquire();
    need(a.slot==0 && b.slot==1,"store chooses physical slots");
    rejects([&]{(void)store.acquire();});
    std::vector<unsigned> prompt{1,2,3};
    engine.begin(a,prompt,false,1,0); unsigned next=0;
    need(!engine.unit(a,1,&next) && store.consumed(a)==0 && !store.pending(a) && engine.checkpoint(a)==1,
         "partial layers preserve truthful canonical frontier");
    need(engine.unit(a,~0U,&next) && next==100 && store.tokens(a)==prompt,"whole prefill consumed once");
    engine.commit(a,next);
    const mm::Row row{a.slot,next,3};
    const auto ids=engine.step(std::span(&row,1),0,std::span(&a,1));engine.commit(a,ids[0]);
    rejects([&]{engine.step(std::span(&row,1),0,std::span(&a,1));});
    rejects([&]{engine.draft(a,100,3);});
    need(store.consumed(a)==4 && store.pending(a)==100 && pair.steps[0]==1 && pair.steps[1]==1 &&
         pair.units[0]==2 && pair.units[1]==2,"both ranks execute independent native plans");
    rejects([&]{(void)store.capture(a);});
    rejects([&]{(void)store.view(a,1);});
    store.release(a);const auto reused=store.acquire();
    need(reused.slot==a.slot && reused.generation==a.generation+1 && store.consumed(reused)==0 &&
         !store.pending(reused),"slot reuse hides old suffix");
    rejects([&]{engine.commit(a,42);}); rejects([&]{store.release(a);});
    rejects([&]{engine.step(std::span(&row,1),0,std::span(&a,1));});
    engine.begin(reused,prompt,false,2,0);store.release(reused);
    need(!pair.rank[0].active.open && !pair.rank[1].active.open,"release cancels partial state on each rank");
    const auto failed=store.acquire();engine.begin(failed,prompt,false,3,0);pair.fail=true;
    rejects([&]{engine.unit(failed,1,&next);});store.invalidate(failed);store.release(failed);pair.fail=false;
    need(!pair.rank[0].active.open && !pair.rank[1].active.open,"failed native work released on both ranks");
    store.release(b);
    const auto named=store.acquire();engine.observe_named(named,prompt,91);
    need(store.tokens(named)==prompt && store.pending(named)==91,"named codec reports authoritative local frontier");
    rejects([&]{engine.observe_named(named,std::vector<unsigned>{8},92);});store.release(named);
    const auto stuck=store.acquire();engine.begin(stuck,prompt,false,30,0);pair.freeze=true;
    rejects([&]{engine.unit(stuck,1,&next);});rejects([&]{(void)store.consumed(stuck);});
    store.release(stuck);pair.freeze=false;
    need(!engine.caps().fork_lanes && !engine.caps().shared_prefix_cow && !engine.caps().external_append &&
         !engine.caps().lane_resume && !engine.caps().prefix_cache,"unsupported capabilities stay off");
    {
        sv::GenerationQueue q;sv::SessionHooks hooks;hooks.control_turn=[](auto&){};
        hooks.now=FakeClock::now;
        sv::SessionRuntime runtime(q,engine,store,{},hooks);
        sv::GenerationJob j;j.prompt={1,2,3};j.max_tokens=3;
        auto t=q.submit(j);need(runtime.reserve(sv::QueueSelection::fifo),"runtime reserves");
        const auto* reserved=runtime.reservation();
        const auto submitted=reserved->submitted;
        need(runtime.reserve(sv::QueueSelection::short_first) && runtime.reservation()==reserved &&
             runtime.reservation()->submitted==submitted,"deferred reservation retains identity/age");
        auto lease=runtime.activate();engine.begin(lease.state,j.prompt,false,4,0);engine.unit(lease.state,~0U,&next);
        engine.commit(lease.state,next);need(runtime.publish(lease.state,std::span(&next,1)),"runtime publication");
        lease.stats.finish="length";runtime.complete(lease.state);
        need(t->stats().generated_tokens==1 && runtime.live()==0,"runtime owns completion/release");
        rejects([&]{runtime.publish(lease.state,std::span(&next,1));});
        j.on_tokens=[](auto)->bool {throw std::runtime_error("callback failed");};
        t=q.submit(j);runtime.reserve(sv::QueueSelection::fifo);auto throwing=runtime.activate();
        throwing.taken.job.on_tokens=[](auto)->bool {throw std::runtime_error("callback failed");};
        need(!runtime.publish(throwing.state,std::span(&next,1)),"callback exception stays lane local");
        runtime.complete(throwing.state);
        need(t->error()=="callback failed" && t->stats().generated_tokens==1,"published accounting survives callback failure");
    }
    // Existing constructor and loop go through SessionRuntime, with no arrival
    // needed for stop/heartbeat/quiesce. The complete policy suite is replayed
    // separately against immutable reference data.
    {
        Scenario idle(ms::Sched::interleave);auto r=idle.run(2500);
        need(idle.engine.quiesces>=2 && count(r,ms::Command::shutdown)==1,"idle stop/control lifecycle");
    }
    for(unsigned allowance : {5U,10U}) {
        WindowEngine native;native.pattern=1;native.dd=&native;native.taps=true;
        sv::GenerationQueue q;ml::Shared shared;shared.policy.sched=ms::Sched::off;
        shared.policy.dspark_mode=ml::ds::Mode::on;shared.policy.inflight=8;shared.policy.lookup=false;
        std::atomic<bool> stop=false;double elapsed=0;
        FakeClock::value={};FakeClock::advance=[&](double ms) {
            elapsed+=ms;FakeClock::value+=std::chrono::duration_cast<FakeClock::time_point::duration>(std::chrono::duration<double,std::milli>(ms));
            if(elapsed>=12000)stop=true;
        };
        sv::GenerationJob j;j.prompt.assign(10,7);j.max_tokens=allowance;j.stop_tokens={107};
        auto t=q.submit(j);
        ml::Lead<WindowEngine,WindowEngine,FakeClock> owner(native,native,q,shared,4096,true,stop);
        owner.runtime().run();
        bool done=false;const auto ids=t->wait(std::chrono::milliseconds(1),done,false);
        need(done && ids==std::vector<unsigned>({100,101,102,103,107}) && t->stats().finish=="stop",
             "speculative stop included at final allowance and middle of window");
        need(std::any_of(native.controls.begin(),native.controls.end(),[](const auto& c){return c.rows>1;}),
             "stop test executes actual speculative native window");
    }
    {
        // Independently executed archived Lead produced 65 IDs and context at
        // this boundary: lookup selects one pending ID after its last row.
        SmallContextEngine native;sv::GenerationQueue q;ml::Shared shared;shared.policy.sched=ms::Sched::off;
        std::atomic<bool> stop=false;double elapsed=0;FakeClock::value={};
        FakeClock::advance=[&](double ms) {
            elapsed+=ms;FakeClock::value+=std::chrono::duration_cast<FakeClock::time_point::duration>(std::chrono::duration<double,std::milli>(ms));
            if(elapsed>=30000)stop=true;
        };
        sv::GenerationJob j;j.prompt.assign(64,7);j.max_tokens=200;auto t=q.submit(j);
        ml::Lead<SmallContextEngine,SmallContextEngine,FakeClock> owner(native,native,q,shared,4096,false,stop);
        owner.runtime().run();bool done=false;const auto ids=t->wait(std::chrono::milliseconds(1),done,false);
        need(done && ids==std::vector<unsigned>(65,100) && t->stats().finish=="context" && t->error().empty(),
             "lookup preserves native final-row pending selection and context limit");
    }
    {
        Scenario s(ms::Sched::interleave);s.submit(0,1,10,2,"named");s.submit(0,2,10,2);
        ms::SessionService service;unsigned calls=0;
        service.execute=[&](const std::string&,const auto&) {++calls;return sv::session_error("session_incompatible","restore refused");};
        (void)s.run(3000,&service);
        need(calls==1 && s.tickets[1]->error()=="session_incompatible: restore refused" &&
             s.tickets[2]->stats().generated_tokens==2,"named restore refusal releases owner before next ordinary admission");
    }
    {
        Scenario s(ms::Sched::interleave);s.submit(0,1,10,100);s.submit(800,2,2000);
        bool cancelled_while_parked=false;
        s.submit(1000,3,10);s.events.push_back({2100,[&] {
            cancelled_while_parked=s.engine.parked.open && s.engine.parked.id==2;s.tickets.at(2)->cancel();
        }});const auto r=s.run(12000);
        need(std::any_of(r.controls.begin(),r.controls.end(),[](const auto& c) {
            return c.command==static_cast<unsigned>(ms::Command::cancel) && c.task_id==2 && c.active.id==2 && c.active.generation==1;
        }) && cancelled_while_parked,"parked parent cancellation is generation/task bound at next native boundary");
        need(s.tickets[2]->stats().finish=="cancelled" && s.tickets[3]->error().empty(),"parked cancellation preserves child completion");
    }
    {
        Scenario s(ms::Sched::interleave);ms::SessionService service;
        auto task=std::make_shared<ms::SessionTasks::Task>();task->body="{}";
        auto future=task->result.get_future();service.tasks.tasks.push_back(task);service.tasks.pending=true;
        unsigned operations=0;
        service.execute=[&](const std::string&,const auto&) {++operations;return sv::Json::object();};
        s.queue.set_idle_control([&]{need(s.queue.try_begin_external_work(),"idle external work admitted");
            s.queue.finish_external_work();need(s.queue.begin_drain() && s.queue.drained(),"idle drain completed");});
        s.queue.wake_control();const auto r=s.run(2500,&service);
        need(operations==1 && future.wait_for(std::chrono::milliseconds(0))==std::future_status::ready &&
             count(r,ms::Command::heartbeat)>=2 && count(r,ms::Command::shutdown)==1,
             "bounded admin/drain/heartbeat/shutdown run without generation");
    }
    {
        Scenario s(ms::Sched::interleave);s.submit(0,1,10,100,"named");ms::SessionService service;
        unsigned parks=0;
        service.execute=[&](const std::string& body,const ms::SessionService::Publish& publish) {
            if(sv::parse_json(body).find("op")->as_string()=="park_active") {
                ++parks;need(s.tickets[1]->stats().finish=="cancelled","named stop replies before paired park");return sv::Json::object();
            }
            FakeClock::advance(100);const unsigned token=42;
            need(!publish(std::span(&token,1)),"named publication observes owner stop without another request");
            return sv::parse_json(R"({"token_ids":[42],"hit":false,"finish":"length"})");
        };
        const auto r=s.run(50,&service);
        need(parks==1 && s.tickets[1]->stats().generated_tokens==1 && count(r,ms::Command::shutdown)==1,
             "named stop accounts committed output and parks before shutdown");
    }
    {
        FaultEngine native;sv::GenerationQueue q;ml::Shared shared;shared.policy.lookup=false;
        std::atomic<bool> stop=false;FakeClock::value={};
        FakeClock::advance=[](double ms) {FakeClock::value+=std::chrono::duration_cast<FakeClock::time_point::duration>(std::chrono::duration<double,std::milli>(ms));};
        sv::GenerationJob j;j.prompt.assign(10,7);j.max_tokens=4;auto t=q.submit(j);
        ml::Lead<FaultEngine,FaultEngine,FakeClock> owner(native,native,q,shared,4096,false,stop);
        rejects([&]{owner.runtime().run();});
        bool done=false;const auto ids=t->wait(std::chrono::milliseconds(1),done,false);
        need(done && ids.size()==2 && t->stats().generated_tokens==2 && t->error()=="injected grouped failure" &&
             q.snapshot().failed==1 && !native.prefill_open(),"native owner failure preserves partial publication and completes ticket");
    }
    std::cout<<"MiniMax session owner checks passed\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
