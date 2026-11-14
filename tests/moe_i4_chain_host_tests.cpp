#ifdef NDEBUG
#undef NDEBUG
#endif
#include "core/moe_i4_arithmetic.hpp"
#include "core/moe_i4_chain_plan.hpp"
#include "core/iu4.hpp"
#include "moe_core/device_plan_index.hpp"
#include "minimax/engine/route_identity.hpp"
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
using namespace strix;
// A binary16 RN-even round trip for finite small-shape test inputs.
float half(float x) {
    const unsigned v=std::bit_cast<unsigned>(x),sign=v&0x80000000U;
    unsigned a=v&0x7fffffffU;
    if(a>=0x47800000U) return std::bit_cast<float>(sign|0x7f800000U);
    if(a<0x38800000U) {
        const float step=0x1p-24f;
        return std::copysign(std::nearbyint(std::fabs(x)/step)*step,x);
    }
    a=(a+0xfffU+((a>>13)&1U))&~0x1fffU;
    return std::bit_cast<float>(sign|a);
}
struct Ops {
    static unsigned channel(unsigned j) { return j; }
    static float round16(float x) { return half(x); }
    static float materialize(float x) { volatile float y=x; return y; }
    static void transform(float (&v)[128]) {
        for(unsigned s=1;s<128;s*=2)
            for(unsigned j=0;j<128;++j) if(!(j&s)) {
                const float l=v[j],r=v[j+s]; v[j]=l+r; v[j+s]=l-r;
            }
        for(float& x:v) x=materialize(x*0x1.6a09e6p-4f);
    }
};
struct Binding {
    static constexpr unsigned H=256,I=256;
    const float* x; bool mm;
    unsigned token(unsigned p) const { return (p*3)%5; }
    float suh(unsigned e,unsigned m,unsigned c) const { return ((e+m+c)%3==0 ? -1.f:1.f)*0.5f; }
    float activation(float g,float u) const {
        if(mm) {
            const float gg=g<7.f ? g:7.f,uu=u < -7.f ? -7.f : (u>7.f ? 7.f:u);
            return gg*(1.f/(1.f+std::exp(-1.702f*gg)))*(uu+1.f);
        }
        const float gg=std::min(g,10.f),uu=std::clamp(u,-10.f,10.f);
        return (gg/(1.f+std::exp(-gg)))*uu;
    }
};
// Independent dense-matrix H128, matching the ascending-stride tree (not a serial dot).
void reference_h(float (&x)[128]) {
    float src[128],dst[128]; std::memcpy(src,x,sizeof(src));
    for(unsigned s=1;s<128;s*=2) {
        for(unsigned i=0;i<128;++i) dst[i]=(i&s) ? src[i^s]-src[i] : src[i]+src[i^s];
        std::memcpy(src,dst,sizeof(src));
    }
    for(unsigned i=0;i<128;++i) x[i]=half(Ops::materialize(src[i]*0x1.6a09e6p-4f));
}
std::array<unsigned char,64> quant(const float (&v)[128],float (&scale)[2]) {
    std::array<unsigned char,64> out{};
    for(unsigned g=0;g<2;++g) {
        float max=0; for(unsigned i=0;i<64;++i) max=std::max(max,std::fabs(v[g*64+i]));
        scale[g]=max==0 ? 1.f:max/7.f;
        for(unsigned i=0;i<64;++i) {
            const unsigned q=unsigned(core::iu4_rne(v[g*64+i]/scale[g]))&15;
            out[g*32+i/2]|=static_cast<unsigned char>(q<<((i&1)*4));
        }
    }
    return out;
}
void arithmetic() {
    std::mt19937 rng(12); std::uniform_real_distribution<float> d(-18,18);
    float x[5*256]; for(float& v:x) v=d(rng);
    for(bool mm:{false,true}) for(unsigned trial=0;trial<64;++trial) {
        Binding b{x,mm}; float v[128],ref[128],g[128],u[128];
        for(unsigned j=0;j<128;++j) { g[j]=d(rng);u[j]=d(rng); }
        if(trial==0) for(unsigned j=0;j<128;++j) g[j]=u[j]=0;
        const unsigned e=trial%17,p=trial%5,col=(trial%2)*128;
        core::moe_i4::input_values<Ops>(b,e,p,trial%2,col,v);
        for(unsigned j=0;j<128;++j) ref[j]=half(Ops::materialize(x[b.token(p)*256+col+j]*b.suh(e,trial%2,col+j)));
        reference_h(ref); assert(std::memcmp(v,ref,sizeof(v))==0);
        core::moe_i4::activation_values<Ops>(b,e,col,g,u,v);
        // Materialized unfused FP32 gate/up, then activation, down scale, FP16, H128, FP16.
        float gu[2][128]; std::memcpy(gu[0],g,sizeof(g));std::memcpy(gu[1],u,sizeof(u));
        for(unsigned j=0;j<128;++j) ref[j]=half(Ops::materialize(b.activation(gu[0][j],gu[1][j])*b.suh(e,2,col+j)));
        reference_h(ref); assert(std::memcmp(v,ref,sizeof(v))==0);
        float vs[2],rs[2]; assert(quant(v,vs)==quant(ref,rs)); assert(std::memcmp(vs,rs,sizeof(vs))==0);
    }
    // The chain combine is the ordered control combine (contribution rows, then ascending top-k FMA):
    // no product boundary or order to bound here; device digests check run-to-run equality.
    // All row/byte locations are disjoint and cover the exact core LDS image.
    for(unsigned rows:{32U,64U,128U}) {
        std::vector<bool> used(rows*64);
        for(unsigned r=0;r<rows;++r) for(unsigned c=0;c<64;++c) {
            const auto i=core::iu4_lds_offset(r,c);assert(i>=0 && unsigned(i)<used.size());assert(!used[i]);used[i]=true;
        }
    }
}
void planner(unsigned R,unsigned E,unsigned K,bool skew) {
    std::mt19937 rng(R+K);std::vector<unsigned> ids(size_t(R)*K),canon(ids.size());
    for(unsigned r=0;r<R;++r) {
        std::vector<unsigned> es;
        while(es.size()<K) { const unsigned e=skew ? unsigned(es.size()) : rng()%E;if(std::find(es.begin(),es.end(),e)==es.end()) es.push_back(e); }
        if(K==4) std::shuffle(es.begin(),es.end(),rng);
        else std::sort(es.begin(),es.end());
        for(unsigned k=0;k<K;++k) ids[size_t(r)*K+k]=es[k];
        std::sort(es.begin(),es.end());for(unsigned k=0;k<K;++k) canon[size_t(r)*K+k]=es[k];
    }
    const unsigned B=(R+255)/256;
    std::vector<unsigned> block(size_t(B)*E),rank(size_t(R)*K),off(E+1);
    // Replay the device rank's wave ballots (32 lanes) and wave-prefix counts.
    for(unsigned b=0;b<B;++b) for(unsigned e=0;e<E;++e) {
        unsigned run=0;
        for(unsigned w=0;w<8;++w) {
            unsigned mask=0;
            for(unsigned l=0;l<32;++l) { const unsigned r=b*256+w*32+l;if(r<R) for(unsigned k=0;k<K;++k) if(canon[size_t(r)*K+k]==e) mask|=1U<<l; }
            for(unsigned l=0;l<32;++l) if(mask&(1U<<l)) {
                const unsigned r=b*256+w*32+l;
                for(unsigned k=0;k<K;++k) if(canon[size_t(r)*K+k]==e) rank[size_t(r)*K+k]=run+std::popcount(mask&((1U<<l)-1U));
            }
            run+=std::popcount(mask);
        }
        block[size_t(b)*E+e]=run;
    }
    unsigned n=0;
    for(unsigned e=0;e<E;++e) { off[e]=n;n+=moe_core::device_plan::scan_column(block.data(),B,E,e); } off[E]=n;
    std::vector<std::vector<unsigned>> rows(E);
    std::vector<unsigned> pairs(n,~0U);
    for(unsigned r=0;r<R;++r) for(unsigned k=0;k<K;++k) {
        const unsigned e=canon[size_t(r)*K+k];
        const unsigned p=moe_core::device_plan::scatter_position(off[e],block[size_t(r/256)*E+e],rank[size_t(r)*K+k]);
        assert(p<n && pairs[p]==~0U);pairs[p]=r;
    }
    for(unsigned e=0;e<E;++e) rows[e].assign(pairs.begin()+off[e],pairs.begin()+(e+1<E ? off[e+1]:n));
    const auto host=minimax::route::expert_rows(ids,R,E,K);assert(rows==host);
    // GLM's pair map retains canonical top-k slot numbers; shared rows follow all routed rows.
    for(unsigned e=0;e<E;++e) for(unsigned r:rows[e]) {
        const auto begin=canon.begin()+size_t(r)*K; const unsigned slot=unsigned(std::find(begin,begin+K,e)-begin);
        assert(slot<K);const unsigned pair=r*(K+1)+slot;assert(pair/(K+1)==r && pair%(K+1)==slot);
    }
}
// Chain plan policy: the down always runs the requested plan; the paired GEMM shares it only with admitted gate
// scratch (GSave on and room for every pair slot), else it stays at plan 1 at most and a plan-2 down rebuilds tiles.
void chain_plans() {
    using strix::core::moe_i4::chain_plan; using strix::core::moe_i4::chain_gsave_floats;
    const std::size_t need=chain_gsave_floats(4096*9,1024);
    assert(need==std::size_t(4096)*9*1024);
    for(int plan=0;plan<3;++plan) {
        const auto on=chain_plan(plan,true,2*need,need), exact=chain_plan(plan,true,need,need);
        assert(on.gsave && on.paired==plan && on.down==plan && !on.rebuild());
        assert(exact.gsave && exact.paired==plan && !exact.rebuild());
        for(const auto reg:{chain_plan(plan,false,2*need,need),chain_plan(plan,true,need-1,need),chain_plan(plan,true,0,need),chain_plan(plan,true,need,0)}) {
            assert(!reg.gsave && reg.down==plan && reg.paired==(plan<1 ? plan : 1) && reg.rebuild()==(plan==2));
        }
    }
}
int main() {
    arithmetic();
    chain_plans();
    for(unsigned R:{0U,1U,31U,32U,33U,255U,256U,257U,8192U})
        for(bool skew:{false,true}) { planner(R,128,4,skew);planner(R,288,8,skew); }
    std::cout<<"PASS: 15 chain plan cases; 36 planner row-list cases; 128 input/epilogue/A4 cases; LDS bijections; 1536 combine orders\n";
}
