// Mutable independently executed ranks; production control framing/replay,
// paired transaction, neutral plan dispatch and KvStore rollback are unchanged.
#include "minimax/serve/session_engine.hpp"
#include <condition_variable>
#include <deque>
#include <future>
#include <iostream>
#include <thread>
#include <atomic>

namespace ms = strix::minimax::serve;
namespace ps = ms::paired;
namespace ss = strix::session;
namespace mm = strix::minimax::engine;
static std::atomic<unsigned> checks=0;
static void check(bool v,const char* why){++checks;if(!v)throw std::runtime_error(why);}
template<class F> static void refuses(F f){bool threw=false;try{f();}catch(const std::exception&){threw=true;}check(threw,"expected refusal");}
struct Wire {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<ps::Bytes> queues[2];
    bool broken=false;
    std::function<void(unsigned,ps::Bytes&)> fault;
    unsigned controls=0,states=0;
    void abort(){std::lock_guard lock(mutex);broken=true;ready.notify_all();}
    void exchange(unsigned rank,const void* out,void* in,std::size_t n){
        std::unique_lock lock(mutex);
        if(broken)throw std::runtime_error("transport aborted");
        check(n>0 && n<=65536,"bounded control/data transport");
        ps::Bytes b(static_cast<const std::uint8_t*>(out),static_cast<const std::uint8_t*>(out)+n);
        if(fault)fault(rank,b);
        if(rank==0 && n==512){++controls;if(b[8]==static_cast<unsigned>(ms::Command::state))++states;}
        queues[rank].push_back(std::move(b));ready.notify_all();
        if(!ready.wait_for(lock,std::chrono::seconds(3),[&]{return broken || !queues[1-rank].empty();})){
            broken=true;ready.notify_all();throw std::runtime_error("transport timeout");
        }
        if(broken)throw std::runtime_error("transport aborted");
        auto peer=std::move(queues[1-rank].front());queues[1-rank].pop_front();
        if(peer.size()!=n){broken=true;ready.notify_all();throw std::runtime_error("transport length");}
        std::copy(peer.begin(),peer.end(),static_cast<std::uint8_t*>(in));
    }
};
struct Memory {
    Wire& wire;unsigned rank;
    std::array<ps::Bytes,8> rows;
    mm::PfProgress progress{},parked{};
    std::vector<std::uint32_t> prompt;
    unsigned exports=0,imports=0,syncs=0;
    bool configured=false,named=false,deny_resource=false,fail_export=false;
    unsigned input_calls=0,input_rows=0;
    unsigned fail_input_call=0;
    std::vector<std::vector<mm::Row>> input_trace;
    std::vector<std::uint32_t> transient_next;
    int fail_import=0; // 1 next import after mutation, 2 every import
    std::shared_ptr<ps::Endpoint> endpoint;
    Memory(Wire& w,unsigned r):wire(w),rank(r){for(auto& row:rows)row.resize(8*ps::Endpoint::row_bytes);}
    mm::PfProgress prefill_progress(bool park=false)const{return park?parked:progress;}
    void exchange_bytes(const void* o,void* i,std::size_t n){wire.exchange(rank,o,i,n);}
    void abort(const char*){wire.abort();}
    void prefill_park(std::uint64_t){throw std::runtime_error("unsupported parked task");}
    void prefill_resume_parked(std::uint64_t){throw std::runtime_error("unsupported parked task");}
    bool prefill_open()const{return progress.open || parked.open;}
    void set_taps(bool){}
    void write(unsigned slot,std::uint64_t pos,std::uint32_t token){
        // Rank and slot-independent mathematical producer, rank-specific bytes.
        const auto offset=static_cast<std::size_t>(pos*ps::Endpoint::row_bytes);
        const auto previous=pos?rows[slot][offset-1]:0;
        for(std::size_t i=0;i<ps::Endpoint::row_bytes;++i)
            rows[slot][offset+i]=static_cast<std::uint8_t>(token+pos+i+rank*31+previous);
    }
    ps::Resources resources(){return {
        [&]{++syncs;},[&]{return !configured && !named && !prefill_open();},
        [&](auto slot,auto n,void* p){++exports;if(fail_export)throw std::runtime_error("export resource fault");
            std::copy_n(rows.at(slot).data(),n*ps::Endpoint::row_bytes,static_cast<std::uint8_t*>(p));},
        [&](auto slot,auto n,const void* p){++imports;std::copy_n(static_cast<const std::uint8_t*>(p),n*ps::Endpoint::row_bytes,rows.at(slot).data());
            if(fail_import){if(fail_import==1)fail_import=0;throw std::runtime_error("import threw AFTER mutation");}},
        [&](const void* o,void* i,std::size_t n){try{exchange_bytes(o,i,n);}catch(...){wire.abort();throw;}},
        [&]{wire.abort();},[&](std::uint64_t){return !deny_resource;}};}
};
struct RankRunner {
    Memory& m;
    void* dd=nullptr;
    void prefill_begin(unsigned slot,std::span<const std::uint32_t> prompt,bool context,
                       std::uint64_t task,std::uint64_t parent,std::uint64_t generation){
        check(!context && !parent,"plain profile");
        m.endpoint->begin(slot,generation,prompt);m.prompt.assign(prompt.begin(),prompt.end());
        m.progress={};m.progress.id=task;m.progress.slot=slot;m.progress.generation=generation;m.progress.open=1;
    }
    bool prefill_unit(unsigned,std::uint32_t* next){
        for(std::size_t i=0;i<m.prompt.size();++i)m.write(m.progress.slot,i,m.prompt[i]);
        m.progress.open=0;*next=91;m.endpoint->selected(m.progress.slot,*next);return true;
    }
    void require_active(std::uint64_t id,unsigned){check(m.progress.id==id && m.progress.open,"active identity");}
    void prefill_check(std::uint64_t,std::uint32_t){}
    void prefill_cancel(std::uint64_t){m.progress.open=0;}
    void retire_contexts(std::uint32_t){}
    void cancel_all(){m.progress.open=m.parked.open=0;}
    ms::SessionDraft draft(unsigned,unsigned,std::uint64_t){throw std::runtime_error("no drafter");}
    void synchronize(){}
    bool input_supported(ps::InputProfile p)const{return m.endpoint && m.endpoint->input_supported(p);}
    ps::Identity input_identity()const{return m.endpoint->identity();}
    std::vector<std::uint32_t> input(std::span<const mm::Row> rows,std::span<const ss::KvHandle> handles,bool follower=false) {
        std::vector<std::uint32_t> next;
        const auto ok=m.endpoint->input(rows,handles,[&]{
            ++m.input_calls;m.input_rows+=static_cast<unsigned>(rows.size());
            m.input_trace.emplace_back(rows.begin(),rows.end());
            std::vector<std::uint32_t> out;out.reserve(rows.size());
            for(const auto& row:rows) {
                auto& bytes=m.rows[row.slot];const auto offset=static_cast<std::size_t>(row.pos*ps::Endpoint::row_bytes);
                std::uint32_t prior=0;
                if(row.pos)for(unsigned j=0;j<4;++j)prior|=std::uint32_t(bytes[offset-ps::Endpoint::row_bytes+j])<<(8*j);
                const auto state=(prior*131+row.token+row.pos+1)%200064;
                m.write(row.slot,row.pos,row.token);
                for(unsigned j=0;j<4;++j)bytes[offset+j]=static_cast<std::uint8_t>(state>>(8*j));
                out.push_back(static_cast<std::uint32_t>((state*17+23)%200064));
            }
            m.transient_next=out; // overwritten by any other lane's next call
            if(m.input_calls==m.fail_input_call)throw std::runtime_error("native row fault AFTER writes");
            return out;
        },next);
        if(!ok && !follower)throw std::runtime_error("paired row execution failed");
        return next;
    }
    void input_control(const ms::Control& c) {
        std::array<ss::KvHandle,8> handles{};
        for(unsigned i=0;i<c.rows;++i)handles[i]={c.row[i].slot,c.dpos[i],c.parent_task_id};
        (void)input(std::span(c.row,c.rows),std::span(handles.data(),c.rows),true);
    }
    std::vector<std::uint32_t> step(std::span<const mm::Row> rows,std::uint32_t append){
        check(append==0,"plain step");m.endpoint->reserve_step(rows);
        std::vector<std::uint32_t> next;
        for(const auto& r:rows){m.write(r.slot,r.pos,r.token);next.push_back(r.token+1);}
        m.endpoint->stepped(rows,next);return next;
    }
};
struct OwnerRunner {
    RankRunner& local;Memory& m;strix::pair::Lockstep lock{0};
    std::shared_ptr<ps::Owner> owner;
    std::function<void(ms::Control&)> command_fault;
    std::function<void(std::vector<std::uint32_t>&)> input_result_fault;
    void send(ms::Control c){if(command_fault)command_fault(c);(void)ms::exchange(m,c,lock);}
    void prefill_begin(unsigned slot,std::span<const std::uint32_t> prompt,bool context,std::uint64_t task,
                       std::uint64_t parent,std::uint64_t generation){
        ms::Control c;c.command=static_cast<unsigned>(ms::Command::prefill);c.pslot=slot;c.plen=static_cast<unsigned>(prompt.size());
        c.task_id=task;c.parent_task_id=parent;c.slot_generation=generation;send(c);
        std::vector<std::uint32_t> out(prompt.begin(),prompt.end()),in(out.size());
        m.exchange_bytes(out.data(),in.data(),out.size()*4);local.prefill_begin(slot,prompt,context,task,parent,generation);
    }
    bool prefill_unit(unsigned layers,std::uint32_t* next){ms::Control c;c.command=static_cast<unsigned>(ms::Command::unit);
        c.task_id=m.progress.id;c.layers=layers;send(c);return local.prefill_unit(layers,next);}
    auto step(std::span<const mm::Row> rows,std::uint32_t append){ms::Control c;c.command=static_cast<unsigned>(ms::Command::step);
        c.rows=static_cast<unsigned>(rows.size());std::copy(rows.begin(),rows.end(),c.row);c.append=append;send(c);return local.step(rows,append);}
    auto draft(unsigned s,unsigned a,std::uint64_t p){return local.draft(s,a,p);}
    bool input_supported(ps::InputProfile p)const{return local.input_supported(p);}
    ps::Identity input_identity()const{return local.input_identity();}
    auto input(std::span<const mm::Row> rows,std::span<const ss::KvHandle> handles) {
        auto c=ms::input_control(rows,handles);send(c);auto next=local.input(rows,handles);
        if(input_result_fault)input_result_fault(next);return next;
    }
    void synchronize(){local.synchronize();}
    void state(ms::SessionPlan& p){if(p.kind==ms::SessionPlan::Kind::capture)p.state=owner->capture(p.frontier);
        else {const auto checkpoint=owner->restore(p.frontier.handle,p.rank,p.state[p.rank]);
            p.frontier.progress=checkpoint.first;p.frontier.progress_limit=checkpoint.second;
            if(m.endpoint->identity().schema==2){const auto& f=m.endpoint->frontier(p.frontier.handle);
                p.frontier.retained=f.retained;p.frontier.finish=f.finish;}}}
};
struct Pair {
    Wire wire;Memory rank0{wire,0},rank1{wire,1};RankRunner local{rank0},peer{rank1};
    OwnerRunner runner{local,rank0};ms::SessionBackend<OwnerRunner> backend{runner};
    ms::SessionEngine engine;
    ss::KvStore store;
    std::thread follower;
    std::exception_ptr follower_error;
    std::uint64_t task=0;
    bool fail_release=false;
    static ps::Identity identity(bool input=false,unsigned slots=3){ps::Identity id;id.capacity=8;id.slots=slots;id.assets_profile.fill(7);id.schema=input?2:1;return id;}
    ps::InputProfile prepare_input(bool input,std::uint64_t budget,unsigned slots) {
        if(!input)return ps::InputProfile::off;
        rank0.endpoint=std::make_shared<ps::Endpoint>(identity(true,slots),budget,rank0.resources(),ps::InputProfile::plain_greedy_r1);
        rank1.endpoint=std::make_shared<ps::Endpoint>(identity(true,slots),budget,rank1.resources(),ps::InputProfile::plain_greedy_r1);
        runner.owner=std::make_shared<ps::Owner>(*rank0.endpoint,[&](auto c){runner.send(c);},budget);
        return ps::InputProfile::plain_greedy_r1;
    }
    explicit Pair(std::uint64_t budget=32ULL<<20,bool active=true,bool input=false,unsigned slots=3)
        :engine(backend,slots,8,[&](auto h){
          if(rank0.progress.open && rank0.progress.slot==h.slot){ms::Control c;c.command=static_cast<unsigned>(ms::Command::cancel);
              c.task_id=rank0.progress.id;runner.send(c);local.prefill_cancel(c.task_id);}
          rank0.endpoint->forget(h.slot);if(fail_release)throw std::runtime_error("release resource fault");},
          [&]{return rank0.progress;},active,prepare_input(input,budget,slots)),store(slots,2,engine.storage_hooks()){
        if(!rank0.endpoint)rank0.endpoint=std::make_shared<ps::Endpoint>(identity(false,slots),budget,rank0.resources());
        if(!rank1.endpoint)rank1.endpoint=std::make_shared<ps::Endpoint>(identity(false,slots),budget,rank1.resources());
        runner.owner=std::make_shared<ps::Owner>(*rank0.endpoint,[&](auto c){runner.send(c);},budget);
        engine.attach(store);
        follower=std::thread([&]{try{
            strix::pair::Lockstep lock{1};
            for(;;){ms::Control mine;
                auto c=ms::exchange(rank1,mine,lock);
                if(c.command==static_cast<unsigned>(ms::Command::state)){rank1.endpoint->follow(c);continue;}
                if(!ms::replay_command(c,rank1,peer,[&](std::size_t n){std::vector<std::uint32_t> out(n),in(n);
                    rank1.exchange_bytes(out.data(),in.data(),n*4);return in;}))break;
            }
        }catch(...){follower_error=std::current_exception();wire.abort();}});
    }
    ~Pair(){wire.abort();if(follower.joinable())follower.join();}
    void fill(ss::KvHandle h,std::vector<std::uint32_t> prompt={1,2}){
        engine.begin(h,prompt,false,++task,0);std::uint32_t next=0;
        check(engine.unit(h,60,&next),"complete native plan");engine.commit(h,next);
    }
    void advance(ss::KvHandle h){mm::Row row{h.slot,*store.pending(h),store.consumed(h)};
        auto next=engine.step(std::span(&row,1),0,std::span(&h,1));engine.commit(h,next[0]);}
    static auto image(const ss::RankState& state){return std::static_pointer_cast<const ps::RankImage>(state.owner)->image;}
    void stop(){ms::Control c;c.command=static_cast<unsigned>(ms::Command::shutdown);runner.send(c);follower.join();
        if(follower_error)std::rethrow_exception(follower_error);}
};
static void restoration(){
    Pair p;auto source=p.store.acquire(),dest=p.store.acquire(),other=p.store.acquire();p.fill(source);p.advance(source);p.fill(other,{4,5});
    auto frozen=p.store.capture(source),bystander=p.store.capture(other);
    auto a=Pair::image(frozen.ranks[0]);check(a->rows[0]!=a->rows[1],"actual rank data differ");
    for(unsigned r=0;r<2;++r){auto view=p.store.view(source,r);auto payload=std::static_pointer_cast<const ps::RankImage>(view.owner);
        check(payload->rank==r && payload->image->frontier.handle==source && payload->image->rows[r]==a->rows[r],"requested slot/rank view");}
    auto before0=p.rank0.rows[other.slot],before1=p.rank1.rows[other.slot];
    p.store.release(source);source=p.store.acquire();p.fill(source,{7,8});
    p.store.restore(dest,frozen);auto restored=p.store.capture(dest);
    check(restored.tokens==frozen.tokens && restored.pending==frozen.pending && Pair::image(restored.ranks[1])->rows==a->rows,"exact complete paired restore");
    check(p.engine.checkpoint(dest)==1,
          "closed owner planner checkpoint restored");
    check(p.rank0.rows[other.slot]==before0 && p.rank1.rows[other.slot]==before1,"bystander untouched");
    check(Pair::image(frozen.ranks[0])->rows==a->rows,"frozen source survives release/reuse");
    p.advance(dest);check(p.store.consumed(dest)==4,"native plan can consume restored pending anchor");
    check(Pair::image(frozen.ranks[0])->rows!=Pair::image(p.store.capture(dest).ranks[0])->rows,"snapshot has no writable aliases");
    const auto commands=p.wire.states;refuses([&]{p.store.truncate(dest,1);});check(p.wire.states==commands,"unsafe truncate before commands");
    refuses([&]{p.store.restore(dest,frozen);});check(p.wire.states==commands,"shorter snapshot restore before work");
    p.store.truncate(dest,p.store.consumed(dest));check(!p.store.pending(dest),"same-length pending clear");
    auto cleared=p.store.capture(dest);check(!cleared.pending,"capture after shared pending clear");
    check(!p.engine.caps().fork_lanes && !p.engine.caps().shared_prefix_cow && !p.engine.caps().external_append &&
          !p.engine.caps().lane_resume && !p.engine.caps().prefix_cache && !p.engine.caps().turn_commit,"no capability inflation");
    p.stop();
}
static void rollback(unsigned failing_rank,bool persistent){
    Pair p;auto source=p.store.acquire(),dest=p.store.acquire(),other=p.store.acquire();p.fill(source);p.fill(dest,{6,7});p.fill(other,{8,9});
    auto source_state=p.store.capture(source),old=p.store.capture(dest),bystander=p.store.capture(other);
    auto& failing=failing_rank?p.rank1:p.rank0;failing.fail_import=persistent?2:1;
    refuses([&]{p.store.restore(dest,source_state);});
    if(persistent){refuses([&]{(void)p.store.tokens(dest);});p.store.release(dest);}
    else {auto restored=p.store.capture(dest);check(restored.tokens==old.tokens && restored.pending==old.pending &&
        Pair::image(restored.ranks[0])->rows==Pair::image(old.ranks[0])->rows,"throwing rank and earlier rank rolled back");}
    check(Pair::image(p.store.capture(source).ranks[0])->rows==Pair::image(source_state.ranks[0])->rows,"source preserved after restore failure");
    check(Pair::image(p.store.capture(other).ranks[1])->rows==Pair::image(bystander.ranks[1])->rows,"sibling preserved after restore failure");p.stop();
}
static void refusals(){
    {
        Pair p(32ULL<<20,false);auto h=p.store.acquire();p.fill(h);const auto n=p.wire.states;
        refuses([&]{(void)p.store.capture(h);});refuses([&]{(void)p.store.view(h,1);});check(p.wire.states==n && n==0,"OFF emits no state commands");p.stop();
    }
    {
        Pair p(300000);auto h=p.store.acquire();p.fill(h);refuses([&]{(void)p.store.capture(h);});
        check(!p.rank0.exports && !p.rank1.exports && !p.wire.states,"budget before export/transport");p.stop();
    }
    {
        Pair p(1400000);auto h=p.store.acquire();p.fill(h);auto held=p.store.capture(h);
        const auto exports=p.rank0.exports+p.rank1.exports;
        refuses([&]{(void)p.store.capture(h);});check(p.rank0.exports+p.rank1.exports==exports,"retained budget before export");
        held={};auto again=p.store.capture(h);check(again.sealed(),"released owners return retained budget");p.stop();
    }
    {
        Pair p;auto h=p.store.acquire();p.fill(h);
        p.rank0.deny_resource=true;refuses([&]{(void)p.store.capture(h);});
        check(!p.rank0.exports && !p.rank1.exports && !p.wire.states,"resource refusal before work");p.rank0.deny_resource=false;
        for(unsigned r=0;r<2;++r){auto& m=r?p.rank1:p.rank0;m.fail_export=true;
            refuses([&]{(void)p.store.capture(h);});m.fail_export=false;}
        check(p.store.capture(h).sealed(),"export failure leaves source and protocol usable");p.stop();
    }
    {
        Pair p;auto h=p.store.acquire();p.fill(h);auto frozen=p.store.capture(h);
        p.fail_release=true;refuses([&]{p.store.release(h);});p.fail_release=false;
        auto d=p.store.acquire();check(d.slot!=h.slot,"failed release quarantines capacity");p.store.restore(d,frozen);
        check(p.store.tokens(d)==frozen.tokens,"snapshot survives failed source release");p.stop();
    }
    {
        Pair p;auto h=p.store.acquire(),d=p.store.acquire();p.fill(h);auto frozen=p.store.capture(h);
        const auto commands=p.wire.states,imports=p.rank0.imports+p.rank1.imports;
        auto bad=frozen;std::swap(bad.ranks[0],bad.ranks[1]);refuses([&]{p.store.restore(d,bad);});
        bad=frozen;bad.tokens[0]++;refuses([&]{p.store.restore(d,bad);});
        bad=frozen;bad.pending=7;refuses([&]{p.store.restore(d,bad);});
        bad=frozen;bad.ranks[0].bytes++;refuses([&]{p.store.restore(d,bad);});
        check(p.wire.states==commands && p.rank0.imports+p.rank1.imports==imports,"provenance refuses before hooks");
        auto hooks=p.engine.storage_hooks();auto foreign=ss::RankState{std::make_shared<int>(3),8};
        refuses([&]{hooks.restore(d,0,foreign);});refuses([&]{hooks.restore(d,0,frozen.ranks[1]);});
        check(p.rank0.imports+p.rank1.imports==imports,"erased foreign/rank refused before mutation");
        p.rank0.configured=true;refuses([&]{(void)p.store.capture(h);});p.rank0.configured=false;
        p.rank0.named=true;refuses([&]{(void)p.store.capture(h);});p.rank0.named=false;
        p.rank1.configured=true;refuses([&]{(void)p.store.capture(h);});p.rank1.configured=false;
        p.rank0.parked.open=p.rank1.parked.open=1;refuses([&]{(void)p.store.capture(h);});p.rank0.parked.open=p.rank1.parked.open=0;
        check(p.rank0.imports+p.rank1.imports==imports,"unsupported profile has no imports");
        p.engine.begin(d,std::vector<std::uint32_t>{3,4},false,++p.task,0);
        const auto n=p.wire.states;refuses([&]{(void)p.store.capture(h);});check(p.wire.states==n,"partial prefill refuses before transport");
        p.store.release(d);p.stop();
    }
    {
        Pair p,q;auto a=p.store.acquire(),b=q.store.acquire();p.fill(a);auto f=p.store.capture(a);
        refuses([&]{q.store.restore(b,f);});check(!q.rank0.imports && !q.rank1.imports,"no cross-store compatibility");p.stop();q.stop();
    }
}
static void restore_transport_fault(unsigned mode){
    Pair p;auto source=p.store.acquire(),dest=p.store.acquire(),other=p.store.acquire();p.fill(source);p.fill(dest,{6,7});p.fill(other,{8,9});
    auto frozen=p.store.capture(source),old=p.store.capture(dest),bystander=p.store.capture(other);
    bool armed=false,injected=false;
    p.runner.command_fault=[&](ms::Control& c){if(c.command==15 && c.pflags==3)armed=true;};
    p.wire.fault=[&](unsigned rank,ps::Bytes& bytes){
        if(armed && rank==1 && bytes.size()==1 && !injected){
            // mode 0: malformed first install vote; mode 1: break the success
            // acknowledgement AFTER follower import. Both abort and quarantine.
            if(mode==0){bytes[0]=17;injected=true;}
            else if(p.rank1.imports){bytes.clear();injected=true;}
        }
    };
    refuses([&]{p.store.restore(dest,frozen);});check(injected,"fault reached real follower install acknowledgement");
    refuses([&]{(void)p.store.tokens(dest);});
    // Broken pair may not execute again. Immutable source/sibling checkpoints
    // remain valid and retain their exact bytes independently of pair teardown.
    check(Pair::image(frozen.ranks[0])->rows!=Pair::image(old.ranks[0])->rows && bystander.sealed(),"transport failure preserves frozen owners");
}
static void row_corruption(){
    Pair p;auto h=p.store.acquire();p.fill(h);bool injected=false;
    p.wire.fault=[&](unsigned rank,ps::Bytes& bytes){if(rank==1 && bytes.size()==65536 && !injected){bytes[123]^=1;injected=true;}};
    refuses([&]{(void)p.store.capture(h);});check(injected && !p.rank0.imports && !p.rank1.imports,"row digest refuses corrupted capture");
}
static void identities(){
    for(unsigned mode=0;mode<7;++mode){
        Wire wire;Memory m{wire,0};auto id=Pair::identity();
        if(mode==0)id.schema=2;else if(mode==1)id.world=3;else if(mode==2)id.slots=UINT32_MAX;
        else if(mode==3)id.capacity=UINT64_MAX;else if(mode==4)id.assets_profile.fill(0);
        else if(mode==5)id.slots=0;else id.capacity=0;
        refuses([&]{ps::Endpoint invalid(id,32ULL<<20,m.resources());});
        check(!m.exports && !m.imports && !wire.controls,"identity refusal before allocation/work");
    }
    for(unsigned mode=0;mode<3;++mode){
        Pair p;auto id=Pair::identity();
        if(mode==0)id.assets_profile[0]++;else if(mode==1)id.capacity++;else id.slots--;
        p.rank1.endpoint=std::make_shared<ps::Endpoint>(id,32ULL<<20,p.rank1.resources());
        auto h=p.store.acquire();p.fill(h);refuses([&]{(void)p.store.capture(h);});
        check(!p.rank0.exports && !p.rank1.exports,"peer asset/profile/capacity/topology identity before export");
    }
}
static void protocol_fault(unsigned mode){
    Pair p;auto h=p.store.acquire();p.fill(h);
    bool injected=false;
    if(mode<=1)p.runner.command_fault=[&](ms::Control& c){if(c.command==15 && !injected){injected=true;c.task_id=mode?0:2;}};
    else p.wire.fault=[&](unsigned rank,ps::Bytes& bytes){if(rank==0 && bytes.size()==144 && !injected){
        injected=true;if(mode==2)bytes[0]^=1;else if(mode==3)bytes[32]=2;else if(mode==4)bytes[40]=3;
        else if(mode==5)bytes[88]=255;else if(mode==6)bytes[136]=1;else bytes.pop_back();}};
    refuses([&]{(void)p.store.capture(h);});check(injected && !p.rank0.imports && !p.rank1.imports,"malformed/stale/identity refusal before mutation");
}
int main()try{
    restoration();for(unsigned r=0;r<2;++r){rollback(r,false);rollback(r,true);}refusals();
    for(unsigned mode=0;mode<8;++mode)protocol_fault(mode);
    restore_transport_fault(0);restore_transport_fault(1);row_corruption();
    identities();
    refuses([]{(void)ps::mul(UINT64_MAX,2);});refuses([]{(void)ps::add(UINT64_MAX,1);});
    std::cout<<"paired state checks "<<checks<<" (always active)\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
