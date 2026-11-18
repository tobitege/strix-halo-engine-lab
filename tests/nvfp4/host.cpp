#include "strix/cuda/core/nvfp4.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace strix::cuda::core::nvfp4;
static void check(bool ok,const char* msg) { if(!ok) throw std::runtime_error(msg); }
template<class F> static void rejects(F f) {
    bool rejected=false; try { f(); } catch(const std::invalid_argument&) { rejected=true; }
    check(rejected,"expected invalid_argument");
}
int main() try {
    for(unsigned i=0;i<16;++i) check(encode_e2m1(e2m1(std::uint8_t(i)))==i,"FP4 roundtrip");
    for(unsigned i=0;i<127;++i) check(encode_e4m3(e4m3(std::uint8_t(i)))==i,"FP8 roundtrip");
    for(unsigned i=0;i<7;++i) {
        const float mid=(e2m1(std::uint8_t(i))+e2m1(std::uint8_t(i+1)))*.5f;
        check(encode_e2m1(mid)==(i%2?i+1:i),"FP4 RNE tie");
        check(encode_e2m1(-mid)==((i%2?i+1:i)|8),"negative FP4 RNE tie");
    }
    for(unsigned i=0;i<126;++i) {
        const float mid=(e4m3(std::uint8_t(i))+e4m3(std::uint8_t(i+1)))*.5f;
        check(encode_e4m3(mid)==(i%2?i+1:i),"FP8 RNE tie");
    }
    check(encode_e2m1(100)==7 && encode_e4m3(1000)==126,"saturate");
    for(const float magnitude : {1073741824.f, std::numeric_limits<float>::max()}) {
        check(encode_e2m1(magnitude)==7,"FP4 large positive finite saturation");
        check(encode_e2m1(-magnitude)==15,"FP4 large negative finite saturation");
    }
    check(std::isnan(e4m3(127)) && std::isnan(e4m3(128)),"invalid scale encoding");
    rejects([] { encode_e2m1(INFINITY); });
    rejects([] { encode_e4m3(-1); });
    // Canonical low-nibble-first storage, nontrivial scale and padding.
    std::vector<float> x(72,0); for(unsigned i=0;i<64;++i) x[i]=float(int(i%13)-6);
    std::vector<std::uint8_t> q(40,0xab),s(8,0xab);
    quantize(x.data(),72,1,64,q.data(),40,s.data(),8,1);
    for(unsigned i=0;i<64;++i)
        check(e2m1(std::uint8_t(q[i/2]>>((i&1)*4)))*e4m3(s[i/16])==
              e2m1(encode_e2m1(x[i])),"quantize decode");
    check(q[32]==0xab && s[4]==0xab,"quantizer overwrites padding");
    std::vector<float> out(7,-99),res(7,3);
    View v{{q.data(),s.data(),40,8,.5f},{q.data(),s.data(),40,8,2},out.data(),res.data(),7,7,1,1,64};
    const auto p=resolve(v); check(p.splits==1 && p.route==Route::decode,"decode plan");
    float dot=0; for(unsigned i=0;i<64;++i) {
        const float w=e2m1(encode_e2m1(x[i])); dot=std::fma(w,w,dot);
    }
    reference(v,p); check(out[0]==dot+3 && out[1]==-99,"reference with residual");
    v.residual=out.data(); reference(v,p); check(out[0]==dot*2+3,"residual alias");
    auto bad=p; ++bad.splits; rejects([&] {reference(v,bad);});
    v.k=63; rejects([&] {resolve(v);}); v.k=64;
    v.a.code_stride=31; rejects([&] {resolve(v);}); v.a.code_stride=40;
    v.a.tensor_scale=0; rejects([&] {resolve(v);}); v.a.tensor_scale=1;
    v.output_stride=std::numeric_limits<std::size_t>::max(); rejects([&] {resolve(v);}); v.output_stride=7;
    v.m=0; rejects([&] {resolve(v);});
    x[0]=NAN; rejects([&] {quantize(x.data(),72,1,64,q.data(),40,s.data(),8,1);});
    std::fill(x.begin(),x.end(),0.f); quantize(x.data(),72,1,64,q.data(),40,s.data(),8,1);
    for(unsigned i=0;i<32;++i) check(q[i]==0,"zero quantization");
    for(unsigned i=0;i<4;++i) check(s[i]==0,"zero block scale");
    std::cout<<"NVFP4 host: codecs, RNE, reference, padding, alias, rejection PASS\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
