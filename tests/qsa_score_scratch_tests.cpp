#include "strix/qsa_prefill_stage_plan.hpp"
#include "strix/qsa_score_scratch.hpp"
#include <chrono>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

namespace {
void expect(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
template<class F> void rejects(F f,const char* message) {
    bool rejected=false;try { f(); } catch(const std::exception&) { rejected=true; }expect(rejected,message);
}
struct State {
    int device=2, setup_call=0, fail_at=0;
    std::uintptr_t next=1;
    bool fail_record=false,fail_wait=false;
    unsigned waits=0,drains=0;
    std::map<void*,std::uint64_t> allocations;
    std::map<std::uintptr_t,bool> events;
    std::vector<std::uintptr_t> streams;
    void setup() { if(++setup_call==fail_at) throw std::runtime_error("injected setup failure"); }
};
struct Backend {
    using Event=std::uintptr_t;
    std::shared_ptr<State> s;
    int device() const { return s->device; }
    void activate(int d) { s->device=d; }
    void* allocate(std::uint64_t bytes) {
        s->setup();auto* p=reinterpret_cast<void*>(s->next++);s->allocations[p]=bytes;return p;
    }
    Event event() { s->setup();const auto e=s->next++;s->events[e]=false;return e; }
    void record(Event e,std::uintptr_t stream) {
        if(s->fail_record) throw std::runtime_error("injected event record failure");
        s->events.at(e)=true;s->streams.push_back(stream);
    }
    void wait(Event e) {
        if(s->fail_wait) throw std::runtime_error("injected incomplete consumer");
        expect(s->events.at(e),"wait must target a recorded consumer");s->events.at(e)=false;++s->waits;
    }
    void drain(std::uintptr_t stream) {
        for(auto& [e,pending]:s->events) { (void)e;pending=false; }
        s->streams.push_back(stream);++s->drains;
    }
    void destroy(Event e) { expect(!s->events.at(e),"event destroyed before consumer completion");s->events.erase(e); }
    void free(void* p) { expect(s->allocations.erase(p)==1,"allocation freed exactly once"); }
};
using Scratch=strix::QsaScoreScratch<Backend>;
void setup_and_lifetime() {
    const strix::QsaScratchBinding binding{2,17,23,262144};
    const auto bytes=strix::QsaScoreRingGeometry(binding.maximum).total_bytes();
    for(int failure=1;failure<=4;++failure) {
        auto s=std::make_shared<State>();s->fail_at=failure;
        rejects([&]{ Scratch scratch(Backend{s},binding,bytes); },"each setup allocation/event failure must propagate");
        expect(s->allocations.empty() && s->events.empty() && !s->drains,"setup rollback frees all partial resources before work");
    }
    auto s=std::make_shared<State>();
    {
        Scratch first(Backend{s},binding,bytes);
        const strix::QsaScratchBinding other{2,31,47,1048576};
        Scratch second(Backend{s},other,strix::QsaScoreRingGeometry(other.maximum).total_bytes());
        expect(first.data(0)!=second.data(0) && s->allocations.size()==4,"two owners have distinct admitted slots");
        expect(s->allocations.at(first.data(0))==bytes/2 &&
            s->allocations.at(second.data(0))==second.geometry().total_bytes()/2,"different capacities use their own plan geometry");
        first.validate(binding);second.validate(other);
        for(unsigned fault=0;fault<4;++fault) {
            auto wrong=binding;
            if(fault==0) ++wrong.device;
            if(fault==1) ++wrong.stream;
            if(fault==2) ++wrong.generation;
            if(fault==3) ++wrong.maximum;
            rejects([&]{first.validate(wrong);},"device/stream/generation/capacity mismatch rejects before work");
        }
        s->device=3;rejects([&]{first.validate(binding);},"current device must still match setup");s->device=2;
        for(unsigned group=0;group<7;++group) {
            Scratch::Lease lease(first);
            expect(lease.slot==group%2,"ring wraps in fixed slot order");
        }
        { Scratch::Lease lease(second); }
        expect(s->waits==5,"wrapped slots wait for their consumers");
    }
    expect(s->waits==8 && s->events.empty() && s->allocations.empty(),"teardown waits both owners' last events before freeing");
    rejects([&]{Scratch scratch(Backend{s},binding,bytes-1);},"admission mismatch rejects without allocation");
}
void slice_planning() {
    using strix::QsaPrefillStagePlan;
    constexpr std::uint32_t boundaries[]{4096U,4097U,65535U,65536U,65537U,262144U};
    constexpr unsigned expected[]{16U,32U,32U,64U,64U,64U};
    for (unsigned i=0;i<6U;++i) {
        const auto blocks=boundaries[i];
        expect(strix::qsa_auto_score_rows(blocks,true)==expected[i],"auto call-wide threshold");
        expect(strix::qsa_auto_score_rows(blocks,false)==(blocks>4096U?32U:16U),"R64 off switch");
        for (const std::uint32_t rows : {64U,128U,2048U,8192U,16384U,72U,120U,216U,8248U,16440U}) {
            for (const unsigned capacity : {64U,128U,256U,512U,1024U}) for (const unsigned shape : {16U,32U,64U,expected[i]}) {
                auto state=std::make_shared<State>();
                const strix::QsaScratchBinding binding{2,17,23,static_cast<std::uint64_t>(blocks)*4U};
                const QsaPrefillStagePlan plan(rows,shape,capacity);
                {
                    Scratch scratch(Backend{state},binding,strix::QsaScoreRingGeometry(binding.maximum,4U,capacity).total_bytes(),capacity);
                    std::uint32_t visited=0;
                    for (std::uint32_t slice=0;slice<plan.slices();++slice) {
                        Scratch::Lease lease(scratch);
                        expect(lease.slot==slice%2U,"slice alternates owner slots through repeated wraps");
                        expect(plan.first(slice)==visited,"slices cover rows once in original order");
                        const auto count=plan.count(slice);
                        expect(count && count<=capacity && count%8U==0U,"slice retains group/tail boundaries");
                        expect(plan.score_rows==shape,"every slice retains the call-wide shape");
                        // In particular a late call crossing a threshold must not
                        // choose R from its earlier slice's smaller block count.
                        const auto earlier=blocks>2048U?blocks-2048U:blocks;
                        if (shape==expected[i] && strix::qsa_auto_score_rows(earlier,true)!=shape)
                            expect(plan.score_rows==expected[i],"slice-local geometry cannot change R");
                        std::uint32_t tail=count%64U;
                        for (const unsigned group : {64U,32U,16U,8U})
                            if (tail>=group) tail-=group;
                        expect(!tail,"mixed tails decompose to incumbent 32/16/8 fallbacks");
                        visited+=count;
                    }
                    expect(visited==rows && plan.slices()==(rows+capacity-1U)/capacity,"complete slice coverage");
                    expect(state->waits==(plan.slices()>2U?plan.slices()-2U:0U),"each reuse waits its last consumer");
                }
                expect(state->allocations.empty() && state->events.empty(),"slice test leaves no pending owner");
                rejects([&]{(void)plan.count(plan.slices());},"out of range slice refuses");
            }
        }
    }
    rejects([]{QsaPrefillStagePlan plan(64U,16U,192U);},"unsupported slot rows refuse");
    rejects([]{QsaPrefillStagePlan plan(0U,16U);},"empty slice plan refuses");
    rejects([]{QsaPrefillStagePlan plan(65U,16U);},"non-tile tail refuses");
    rejects([]{QsaPrefillStagePlan plan(64U,8U);},"unsupported score shape refuses");
}
void geometry() {
    using strix::QsaScoreRingGeometry;
    const std::uint64_t capacities[]{262144U,524288U,1048576U};
    const std::uint32_t defaults[]{512U,512U,256U};
    const std::uint64_t bytes[]{335544320U,671088640U,671088640U};
    for(unsigned i=0;i<3U;++i) {
        const QsaScoreRingGeometry automatic(capacities[i]);
        expect(automatic.rows==defaults[i] && automatic.total_bytes()==bytes[i],"bounded default receipt at admitted capacities");
        for(const unsigned rows : {64U,128U,256U,512U,1024U}) {
            const QsaScoreRingGeometry geo(capacities[i],4U,rows);
            expect(geo.rows==rows && geo.blocks==capacities[i]/4U && geo.maxima_stride==capacities[i]/16U,
                "explicit groups retain score/maxima row geometry");
            expect(geo.total_bytes()==static_cast<std::uint64_t>(rows)*capacities[i]*5U/2U,
                "two slots charge every score and maxima byte");
            auto state=std::make_shared<State>();
            const strix::QsaScratchBinding binding{2,17,23,capacities[i]};
            { Scratch scratch(Backend{state},binding,geo.total_bytes(),rows);
              expect(state->allocations.at(scratch.data(0))==geo.total_bytes()/2U,"owner allocates explicit admitted geometry"); }
            rejects([&]{Scratch scratch(Backend{state},binding,geo.total_bytes()-1U,rows);},"explicit geometry admission mismatch rejects");
        }
    }
    expect(QsaScoreRingGeometry(16777216U).rows==64U,"default stops at one group");
    const QsaScoreRingGeometry rounded(262145U);
    expect(rounded.blocks==65537U && rounded.maxima_stride==16385U,"partial capacity rounds both strides up");
    expect(strix::qsa_score_ring_rows_override(nullptr)==0U,"missing override selects byte policy");
    for(const char* value : {"64","128","256","512","1024"})
        expect(strix::qsa_score_ring_rows_override(value)==std::stoul(value),"documented override accepted");
    for(const char* value : {"","0","32","192","2048","+64","064","64x"," 64"})
        rejects([&]{(void)strix::qsa_score_ring_rows_override(value);},"invalid override rejects");
    rejects([]{QsaScoreRingGeometry geo(0U);},"zero capacity refuses");
    rejects([]{QsaScoreRingGeometry geo(262144U,0U);},"zero ratio refuses");
    rejects([]{QsaScoreRingGeometry geo(262144U,4U,192U);},"non-power-of-two rows refuse");
    rejects([]{QsaScoreRingGeometry geo(UINT64_MAX);},"capacity rounding overflow refuses");
    rejects([]{QsaScoreRingGeometry geo(UINT64_MAX,1U);},"row byte overflow refuses");
    rejects([]{QsaScoreRingGeometry geo(UINT64_MAX/1024U,1U,1024U);},"slot multiplication overflow refuses");
}
void exceptional_lease() {
    auto s=std::make_shared<State>();
    const strix::QsaScratchBinding binding{2,17,23,262144};
    {
        Scratch ring(Backend{s},binding,strix::QsaScoreRingGeometry(binding.maximum).total_bytes());
        rejects([&]{Scratch::Lease lease(ring);throw std::runtime_error("after score submission");},"injected launch exception");
        { Scratch::Lease lease(ring); }
        s->fail_wait=true;
        rejects([&]{Scratch::Lease lease(ring);},"incomplete consumer refuses reuse");
        s->fail_wait=false;
        { Scratch::Lease lease(ring);expect(lease.slot==0,"failed wait retains pending slot and cursor"); }
    }
    expect(s->waits==3 && s->allocations.empty(),"exceptional leases retain their completion event through teardown");
    s=std::make_shared<State>();
    {
        Scratch ring(Backend{s},binding,strix::QsaScoreRingGeometry(binding.maximum).total_bytes());
        s->fail_record=true;
        for(unsigned i=0;i<4;++i) { Scratch::Lease lease(ring); }
        expect(s->drains==4,"failed consumer recording drains before slot can be reused");
    }
    expect(s->allocations.empty() && s->events.empty(),"record failure has quiescent teardown");
}
void pending_and_serialization() {
    struct Attend { bool active=true;std::vector<int> groups{1,2};const void* kv=this;unsigned union_at=7; } b;
    struct Select { bool active=true;std::vector<int> rows{1},traced{2};unsigned used=8; } sb;
    struct Index { bool active=true;std::vector<int> groups{1}; } ib;
    float value=1,*maxima=&value;std::uint32_t stride=64;bool drained=false;
    rejects([&]{
        strix::QsaPendingGuard guard([&]{drained=true;},[&]{
            expect(drained,"pending reset follows completion proof");
            strix::clear_qsa_pending(b,sb,ib,maxima,stride);
        });
        throw std::runtime_error("pending attend/selection submission failed");
    },"injected pending submission exception");
    expect(!b.active && b.groups.empty() && !b.kv && !b.union_at && !sb.active && sb.rows.empty() &&
        sb.traced.empty() && !sb.used && !ib.active && ib.groups.empty() && !maxima && !stride,
        "exception leaves no pending descriptors, active flags or stale maxima binding");
    strix::QsaScratchSerialGate gate;
    auto first=gate.enter(2);
    std::promise<void> started;
    auto ready=started.get_future();
    auto next=std::async(std::launch::async,[&]{started.set_value();const auto second=gate.enter(2);return true;});
    ready.wait();
    const bool blocked=next.wait_for(std::chrono::milliseconds(20))==std::future_status::timeout;
    first.unlock();expect(next.get() && blocked,"second stream waits while first owns shared caches, independent of pooling");
    rejects([&]{const auto wrong=gate.enter(3);},"a different device is refused at setup");
    strix::QsaScratchHandoff handoff;
    unsigned waits=0;
    const auto wait=[&]{++waits;};
    handoff.before(17,wait);handoff.recorded(17);handoff.before(17,wait);
    expect(waits==0,"same-stream submissions retain asynchronous ordering without a host fence");
    rejects([&]{handoff.before(31,[]{throw std::runtime_error("consumer incomplete");});},"cross-stream wait failure refuses handoff");
    expect(handoff.pending(),"failed handoff preserves pending consumer");
    handoff.before(31,wait);handoff.recorded(31);
    expect(waits==1,"different stream waits for the previous shared-scratch consumer");
    handoff.drained();handoff.before(17,wait);
    expect(waits==1 && !handoff.pending(),"exceptional drain clears the shared consumer before reuse");
    handoff.recorded(17);
    expect(handoff.pending_on(17) && !handoff.pending_on(31),"only the stream of the last submission owes its completion");
    handoff.drained();
    expect(!handoff.pending_on(17),"a drained handoff releases its stream");
}
}
int main() {
    try {
        geometry();setup_and_lifetime();slice_planning();exceptional_lease();pending_and_serialization();
        float embedded_storage=0,ring_storage=0;
        for(unsigned mask=0;mask<1024;++mask) {
            auto* embedded=strix::qsa_selector_maxima(mask,&embedded_storage);
            auto* compact=strix::qsa_selector_maxima(mask,&ring_storage);
            expect(bool(embedded)==bool(compact) && bool(compact)==bool(mask&2U),
                "compact and embedded preserve selector arm for every diagnostic mask");
            expect(!compact || (compact==&ring_storage && embedded==&embedded_storage),"maxima retain storage provenance");
            expect(!strix::qsa_selector_maxima(mask,nullptr),"unavailable maxima never select the assisted arm");
        }
        strix::require_qsa_score_source(false,false);strix::require_qsa_score_source(true,true);
        rejects([]{strix::require_qsa_score_source(true,false);},"unbound compact launcher is refused without device calls");
        std::cout<<"QSA score scratch CPU fixtures passed\n";return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
