#include "gemma4/engine/native_memory.hpp"
#include <iostream>
#include <memory>
#include <vector>

namespace m=strix::gemma4::memory;
#define CHECK(x) do {if(!(x))throw std::runtime_error(#x);}while(false)
template<class F> void overflow(F f) {bool caught=false;try{f();}catch(const std::overflow_error&){caught=true;}CHECK(caught);}
int main() {
    try {
        // Host algebra evidence, not physical native allocation measurement.
        // Individually listed capacities intentionally differ from used rows.
        const auto make=[] {
            auto n=m::stream_payload(144,7,12,8,64,9,32,6*56,true);
            m::stream_arrays(n,31*128*4,31*128*4,31*8); // admitted global
            m::stream_arrays(n,17*64*4,17*64*4,17*8); // full sliding ring/tags
            m::stream_arrays(n,8*128*4,8*128*4,0); // spare overlay global
            m::stream_arrays(n,8*64*4,8*64*4,0); // spare overlay sliding
            return n;
        };
        auto n=make();
        CHECK(n.host==144+7*12+8*64+9*32);
        CHECK(n.device==6*56+31*128*4*2+31*8+17*64*4*2+17*8+8*128*4*2+8*64*4*2);
        CHECK(n.pinned==6*56);
        CHECK(n.total()==n.host+n.device+n.pinned);
        auto no_pin=m::stream_payload(144,7,12,8,64,9,32,336,false);
        CHECK(no_pin.pinned==0 && no_pin.device==336);
        strix::gemma4::ModelConfig c;c.layers=3;c.kv_heads=1;c.window=17;c.max_positions=99;
        c.layer={{strix::gemma4::AttentionKind::global,128,0},{strix::gemma4::AttentionKind::sliding,64,1},
                 {strix::gemma4::AttentionKind::global,128,0}}; // aliases producer zero
        const auto geometry=m::exact_stream_bound(c,31,{144,64,12,64,32,56});CHECK(geometry);
        CHECK(geometry->additional.host==144+64+3*2*(12+64+2*32)+3*2*56+31*2*8);
        CHECK(geometry->additional.device==3*56+31*128*4*2+31*8+17*64*4*2+17*8+8*128*4*2+8*64*4*2);
        CHECK(geometry->additional.pinned==3*56);
        CHECK(!m::exact_stream_bound(c,0,{144,64,12,64,32,56}));
        CHECK(!m::exact_stream_bound(c,100,{144,64,12,64,32,56}));
        CHECK(!m::exact_stream_bound(c,31,{UINT64_MAX,64,12,64,32,56}));
        c.layer.back().producer=99;CHECK(!m::exact_stream_bound(c,31,{144,64,12,64,32,56}));
        const auto a=m::attention(65,3,8,64);
        CHECK(a.scores==8*3*128*4 && a.maxima==8*3*3*4 && a.partials==8*3*2*513*4);
        CHECK(m::attention(64,3,1,64).scores==3*64*4);
        overflow([]{(void)m::product(UINT64_MAX,2);});
        overflow([]{std::uint64_t x=UINT64_MAX;m::add(x,1);});
        overflow([]{(void)m::attention(UINT64_MAX,2,8,64);});
        overflow([]{(void)m::stream_payload(1,UINT64_MAX,8,1,1,1,1,0,false);});
        CHECK(!m::fits({},1,1,1));CHECK(!m::fits(0,0,1,1));CHECK(!m::fits(0,1,1,10));CHECK(!m::fits(1,1,1,0));
        CHECK(!m::fits(1,UINT64_MAX,2,10));CHECK(!m::fits(1,1,2,UINT64_MAX));
        CHECK(!m::fits(UINT64_MAX,2,2,100));
        const auto bytes=n.total();
        CHECK(m::fits(123,bytes,5,5*bytes+123));CHECK(!m::fits(123,bytes,5,5*bytes+122));
        // Source, candidate, install and rollback are independent owners. Aliases
        // to an immutable candidate do not allocate a second payload. Releasing
        // the source cannot release the captured owner; repeat without drift.
        for(int i=0;i<100;++i) {
            auto source=std::make_shared<m::Payload>(make());
            auto candidate=std::make_shared<m::Payload>(*source);
            auto alias=candidate;
            auto rollback=std::make_unique<m::Payload>(*source);
            auto install=std::make_unique<m::Payload>(*candidate);
            CHECK(source->total()+candidate->total()+rollback->total()+install->total()==4*bytes);
            source.reset();candidate.reset();CHECK(alias->total()==bytes && alias.use_count()==1);
            rollback.reset();install.reset();CHECK(alias->total()==bytes);
        }
        std::cout<<"PASS host accounting algebra/overflow/owner-lifetime evidence; native execution NOT RUN\n";
        return 0;
    }catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}
}
