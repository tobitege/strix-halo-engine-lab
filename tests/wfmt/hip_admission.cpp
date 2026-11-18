// No HIP runtime, device allocation or multi-GiB backing allocation is needed:
// these tests exercise the same launch envelope called by registry admission.
#include "../../src/hip/weight/registry.hpp"
#include "byte_equal.hpp"
#include <iostream>
#include <stdexcept>
#include <string_view>
namespace {
using namespace strix::weight;
using namespace strix::weight::hip;
void expect(bool condition,const char* message) {
    if(!condition) throw std::runtime_error(message);
}
WeightView tensor(std::uint64_t n,std::uint64_t k,unsigned block,unsigned bytes) {
    WeightView w{};
    w.shape={{n,k,0,0},2};
    w.payload.range.bytes=n*(k/block)*bytes;
    return w;
}
void addressing() {
    auto w=tensor(2,128,32,34);
    LaunchShape s{1,2,128,136,128,2,0};
    expect(bool(validate_launch(w,s,32,34,68)),"contiguous Q8 rejected");
    s.row_stride_bytes=272;
    expect(!validate_launch(w,s,32,34,68),"Q8 stride escapes owned payload");
    w=tensor(2,256,32,34);s.k_offset_bytes=1;
    expect(!validate_launch(w,s,32,34,68),"Q8 byte prefix admitted");
    s.k_offset_bytes=34;
    expect(!validate_launch(w,s,32,34,68),"Q8 pair-misaligned prefix admitted");
    expect(bool(validate_launch(w,s,32,34,34)),"scalar Q8 whole block rejected");
    s.k_offset_bytes=136;
    expect(bool(validate_launch(w,s,32,34,68)),"Q8 end-aligned subrow rejected");
    s.k_offset_bytes=204;
    expect(!validate_launch(w,s,32,34,68),"Q8 subrow crosses row boundary");
    s.k_offset_bytes=136;--w.payload.range.bytes;
    expect(!validate_launch(w,s,32,34,68),"last byte outside payload admitted");
    s.k_offset_bytes=UINT64_MAX;
    expect(!validate_launch(w,s,32,34,68),"overflowing prefix admitted");
    s.k_offset_bytes=0;w.shape.dims[1]=UINT64_MAX-31U;
    expect(!validate_launch(w,s,32,34,68),"overflowing storage row admitted");
    w=tensor(2,256,1,2);s={1,2,128,512,128,2,256};
    expect(bool(validate_launch(w,s,1,2,2)),"F16 subrow rejected");
    s.k_offset_bytes=1;
    expect(!validate_launch(w,s,1,2,2),"unaligned F16 prefix admitted");
    s.k_offset_bytes=0;w.has_byte_strides=true;w.byte_strides[0]=256;
    expect(!validate_launch(w,s,1,2,2),"declared stride mismatch admitted");
}
WeightView trellis_tensor(std::uint64_t n,std::uint64_t k,unsigned bits,bool gemma) {
    WeightView w{};
    w.storage={gemma?"exl3.gemma.transport":"exl3.mul1.tiles16",1};
    w.shape={{n,k,0,0},2};
    w.parameters.bits=bits;
    const std::uint64_t trellis=(n/16)*(k/16)*32*bits;
    w.payload.range.bytes=gemma?2*(n+k)+trellis:trellis;
    if(!gemma) {
        w.sidecars.push_back({"input_scale",Scalar::f16,{{{k,0,0,0}},1},{nullptr,{0,2*k}}});
        w.sidecars.push_back({"output_scale",Scalar::f16,{{{n,0,0,0}},1},{nullptr,{0,2*n}}});
    }
    return w;
}
// Heterogeneous experts: every resident expert is admitted against its own sidecars, the launch spans the whole ring, and a
// nominal packed row (K * bits / 8, Gemma's LinearView::row_bytes) is never accepted as a stride.
void trellis() {
    // Exact payloads and sidecars, with matching launches: only H128 tails fail.
    for(bool gemma:{false,true})
        for(const std::uint64_t n:{16ULL,128ULL,256ULL,272ULL})
            for(const std::uint64_t k:{16ULL,128ULL,256ULL,272ULL}) {
                const auto w=trellis_tensor(n,k,gemma?4U:3U,gemma);
                const LaunchShape shape{1,n,k,0,k,n,0};
                expect(bool(validate_trellis_launch(w,shape,gemma))==(n%128U==0 && k%128U==0),
                       "matching trellis shape must contain complete H128 blocks");
            }
    LaunchShape s{129,256,256,0,256,256,0};
    for(const bool gemma:{false,true}) {
        auto w=trellis_tensor(256,256,4,gemma);
        expect(bool(validate_trellis_launch(w,s,gemma)),"whole-tensor trellis launch rejected");
        for(const std::uint64_t rows:{1ULL,7ULL,129ULL,257ULL}) {auto c=s;c.m=rows;expect(bool(validate_trellis_launch(w,c,gemma)),"chunked row count rejected");}
        auto bad=s;bad.row_stride_bytes=256*4/8;
        expect(!validate_trellis_launch(w,bad,gemma),"nominal row_bytes accepted as a stride");
        bad=s;bad.k_offset_bytes=32;expect(!validate_trellis_launch(w,bad,gemma),"trellis K prefix admitted");
        bad=s;bad.n=128;expect(!validate_trellis_launch(w,bad,gemma),"trellis output slice admitted");
        bad=s;bad.k=128;expect(!validate_trellis_launch(w,bad,gemma),"trellis K slice admitted");
        bad=s;bad.m=0;expect(!validate_trellis_launch(w,bad,gemma),"zero rows admitted");
        auto t=w;--t.payload.range.bytes;expect(!validate_trellis_launch(t,s,gemma),"truncated payload admitted");
        t=w;++t.payload.range.bytes;expect(!validate_trellis_launch(t,s,gemma),"oversized payload admitted");
        t=w;t.parameters.bits=7;if(gemma) expect(!validate_trellis_launch(t,s,gemma),"unsupported Gemma bits admitted");
        t=w;t.storage={gemma?"exl3.mul1.tiles16":"exl3.gemma.transport",1};expect(!validate_trellis_launch(t,s,gemma),"wrong transport admitted");
        t=trellis_tensor(272,256,4,gemma);expect(!validate_trellis_launch(t,s,gemma),"shape differs from launch");
    }
    // mul1: each expert needs its own sidecars; one expert's sidecars never stand in for another's.
    auto a=trellis_tensor(256,256,3,false),b=a;
    b.sidecars.pop_back();expect(!validate_trellis_launch(b,s,false),"expert without svh admitted");
    b=a;b.sidecars[0].data.range.bytes=2*128;expect(!validate_trellis_launch(b,s,false),"short suh admitted");
    b=a;b.sidecars.push_back(a.sidecars[0]);expect(!validate_trellis_launch(b,s,false),"duplicate suh admitted");
    b=a;b.sidecars[1].scalar=Scalar::bf16;expect(!validate_trellis_launch(b,s,false),"wrong svh scalar admitted");
    // Gemma payload carries suh | svh itself.
    auto g=trellis_tensor(256,256,4,true);g.sidecars.push_back(a.sidecars[0]);
    expect(!validate_trellis_launch(g,s,true),"Gemma transport with canonical sidecars admitted");
}
void q6_span() {
    auto w=tensor(65536,131072,32,26);
    LaunchShape s{1,65536,131072,106496,131072,65536,0};
    expect(!validate_launch(w,s,32,26,26,true),"Q6 6.5-GiB span admitted");
    // Both native and signed-A8 protocols share this cap. Width 64 satisfies
    // the signed route's additional even-pair / whole-dword requirement.
    constexpr std::uint64_t rows=UINT32_MAX/52U;
    w=tensor(rows+1U,64,32,26);s={1,rows,64,52,64,rows,0};
    expect(bool(validate_launch(w,s,32,26,26,true)),"Q6 below-limit span rejected");
    ++s.n;++s.output_stride;
    expect(!validate_launch(w,s,32,26,26,true),"Q6 last group crosses 32-bit boundary");
    w=tensor(2,256,32,26);s={1,2,128,208,128,2,0};
    expect(!validate_launch(w,s,32,26,26,true),"Q6 shortened planar row admitted");
    s.k_offset_bytes=26;
    expect(!validate_launch(w,s,32,26,26,true),"Q6 planar prefix admitted");
}
void bytes() {
    using strix::weight::test::byte_equal;
    expect(!byte_equal({0.0F},{-0.0F}),"signed-zero change hidden");
    expect(byte_equal({-0.0F,1.0F},{-0.0F,1.0F}),"identical bytes rejected");
    expect(!byte_equal({1.0F},{}),"length mismatch hidden");
    expect(byte_equal({},{}),"empty vectors rejected");
}
}
int main(int argc,char** argv) try {
    if(argc!=2) throw std::runtime_error("expected addressing, q6-span or bytes");
    const std::string_view name=argv[1];
    if(name=="addressing") {addressing();trellis();}
    else if(name=="q6-span") q6_span();
    else if(name=="bytes") bytes();
    else throw std::runtime_error("unknown test");
    return 0;
} catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
}
