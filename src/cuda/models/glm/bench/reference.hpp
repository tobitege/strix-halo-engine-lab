#pragma once
#include "../trellis.cuh"
#include <vector>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cstdio>

namespace strix::glm::rtxpf::bench {
struct HostMatrix { unsigned input,output,bits; std::vector<U16> packed,suh,svh; };
struct HostExpert { HostMatrix gate,up,down; };
// Increasing-k scalar FP32 FMA chain, with the exact decoded FP16 weights.
// The reference shares format helpers (exhaustively CPU checked), but no MMA,
// dispatch, accumulation schedule or fused epilogue implementation.
inline std::vector<float> linear(const std::vector<float>& x,const HostMatrix& w) {
    std::vector<float> a(w.input),q(std::size_t(w.input)*w.output);
    for(unsigned b=0;b<w.input;b+=128) {
        float v[128];for(unsigned j=0;j<128;++j)v[j]=hf(fh(x[b+j]*hf(w.suh[b+j])));
        hadamard(v);for(unsigned j=0;j<128;++j)a[b+j]=hf(fh(v[j]));
    }
    for(unsigned ki=0;ki<w.input/16;++ki)for(unsigned no=0;no<w.output/16;++no) {
        const U16* p=w.packed.data()+(std::size_t(ki)*(w.output/16)+no)*16*w.bits;
        for(unsigned pos=0;pos<256;++pos)q[std::size_t(ki*16+qi(pos))*w.output+no*16+qo(pos)]=hf(book(state(p,pos,w.bits)));
    }
    std::vector<float> y(w.output,0);
    // k outer permits independent output columns to be vectorized by MSVC.
    for(unsigned k=0;k<w.input;++k)for(unsigned n=0;n<w.output;++n)y[n]=std::fma(a[k],q[std::size_t(k)*w.output+n],y[n]);
    for(unsigned b=0;b<w.output;b+=128)hadamard(y.data()+b);
    for(unsigned n=0;n<w.output;++n)y[n]*=hf(w.svh[n]);
    return y;
}
inline std::vector<float> moe(const std::vector<float>& x,const std::vector<HostExpert>& ex,const U32* ids,const float* weights,unsigned routes) {
    std::vector<float> y(x.size(),0);
    for(unsigned s=0;s<routes;++s) {
        const auto& e=ex[ids?ids[s]:0];
        auto g=linear(x,e.gate),u=linear(x,e.up);
        for(unsigned i=0;i<g.size();++i)g[i]=swiglu(g[i],u[i]);
        auto d=linear(g,e.down);
        for(unsigned i=0;i<y.size();++i)y[i]=y[i]+d[i]*(weights?weights[s]:1.f);
    }
    return y;
}
inline void metrics(const float* a,const float* ref,std::size_t n,const char* label) {
    double sq=0,den=0;float mx=0;unsigned long long hist[8]{};
    auto ordered=[](float x) { U32 b;std::memcpy(&b,&x,4);if((b&0x7fffffff)==0)b=0;return b&0x80000000?~b:b|0x80000000; };
    for(std::size_t j=0;j<n;++j) {
        if(!std::isfinite(a[j])||!std::isfinite(ref[j]))throw std::runtime_error("nonfinite accuracy output");
        float d=a[j]-ref[j];mx=std::max(mx,std::fabs(d));sq+=double(d)*d;den+=double(ref[j])*ref[j];
        U32 aa=ordered(a[j]),bb=ordered(ref[j]),dist=aa>bb?aa-bb:bb-aa;
        unsigned bin=dist==0?0:dist==1?1:dist<=4?2:dist<=16?3:dist<=64?4:dist<=256?5:dist<=1024?6:7;++hist[bin];
    }
    std::printf("accuracy %s n=%zu max_abs=%.9g rel_L2=%.9g ulp_bins=[0,1,2..4,5..16,17..64,65..256,257..1024,>1024] counts=",label,n,mx,den?std::sqrt(sq/den):(sq?std::numeric_limits<double>::infinity():0));
    for(auto v:hist)std::printf(" %llu",v);std::puts("");
}
}
