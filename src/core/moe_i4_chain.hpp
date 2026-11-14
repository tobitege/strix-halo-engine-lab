#pragma once
// Candidate FAST_I4 chain. Bindings name model geometry, row maps and nonlinear arithmetic;
// all gather/transform/quantise, paired mainloops and output accumulation live on this core.
#include "tiled_core.hpp"
#include "activation_quant.hpp"
#include "linear_plan.hpp"
#include "moe_i4_arithmetic.hpp"
#include "moe_i4_chain_plan.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strix::core::moe_i4 {
// Chain plan policy: auto-select BM128 (plan 2) when the call averages >= kPlan2Rows rows per group.
// Waves past a ragged tile's rows skip their WMMA (tiled_core), so BM128's padding no longer costs its
// compute: grouped down on MiniMax-like routing (gfx1151) BM128 vs BM64 -4 % at ~159 rows per group and
// equal at 64. The paired GEMM keeps its pass-0 gate in a global FP32 scratch (GSave, default on) so
// it runs the same plan as down; STRIX_MOE_I4_CHAIN_GSAVE=0 restores the register save (BM64 max).
inline constexpr unsigned kPlan2Rows = 64;
// STRIX_MOE_I4_CHAIN=0|1 overrides the engine's default (def).
inline bool enabled(bool def = false) {
    const char* p = std::getenv("STRIX_MOE_I4_CHAIN");
    if (!p || !*p) return def;
    if (std::strcmp(p, "0") == 0) return false;
    if (std::strcmp(p, "1") == 0) return true;
    throw std::invalid_argument("STRIX_MOE_I4_CHAIN must be 0 or 1");
}
// STRIX_MOE_I4_CHAIN_GSAVE=0|1: paired gate kept in global FP32 scratch (1) or registers (0).
inline bool gsave_enabled(bool def = true) {
    const char* p = std::getenv("STRIX_MOE_I4_CHAIN_GSAVE");
    if (!p || !*p) return def;
    if (std::strcmp(p, "0") == 0) return false;
    if (std::strcmp(p, "1") == 0) return true;
    throw std::invalid_argument("STRIX_MOE_I4_CHAIN_GSAVE must be 0 or 1");
}
struct Weights {
    const uint8_t* q[3];
    const _Float16* s[3];
    const uint16_t* const* svh[3];
    size_t q_stride[3], s_stride[3];
};
struct WaveOps {
    __device__ __forceinline__ static unsigned channel(unsigned j) { return (threadIdx.x&31U)+32*j; }
    __device__ __forceinline__ static float round16(float x) { return float(_Float16(x)); }
    __device__ __forceinline__ static float materialize(float x) { return materialize_fp32(x); }
    __device__ __forceinline__ static void transform(float (&v)[4]) { hadamard128(v); }
};
template<class Binding> struct InputPolicy {
    Binding b;
    __device__ __forceinline__ void load(unsigned e,unsigned p,unsigned m,unsigned c,float (&v)[4],const float* cached=nullptr) const {
        input_values<WaveOps>(b,e,p,m,c,v,cached);
    }
};
template<class Binding> struct EpiloguePolicy {
    Binding b;
    __device__ __forceinline__ void activation(unsigned e,unsigned c,const float (&g)[4],const float (&u)[4],float (&v)[4]) const {
        activation_values<WaveOps>(b,e,c,g,u,v);
    }
    // Deterministic combine, stage 1: the down epilogue stores each routed pair's FP32 output row
    // (the value the unfused control writes to its down image). Stage 2 is the model's existing
    // ordered combine (ascending top-k FMA, shared expert last), so the chain adds no rounding
    // point, no atomics and no run-to-run or cross-rank order dependence.
    __device__ __forceinline__ void contribution(unsigned pair, unsigned col, const float (&v)[4]) const {
        float* row = b.contribution(pair);
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) row[col + (threadIdx.x & 31U) + 32*j] = v[j];
    }
};
// Unfused control uses the same policies; schedules only name rows and destinations.
struct ReferenceRow {
    unsigned expert,pair,m,col,width;
    const float* gu=nullptr;
    uint8_t* aq=nullptr; float* as=nullptr; _Float16* fp16=nullptr;
    uint8_t* aq2=nullptr; float* as2=nullptr;   // optional remainder plane (iu4_quant128_planes, W4A8 as two IU4 GEMMs)
};
__device__ __forceinline__ void reference_store(const ReferenceRow& r,const float (&v)[4]) {
    if(r.aq && r.aq2) iu4_quant128_planes(v,r.aq+size_t(r.pair)*(r.width/2),r.as+size_t(r.pair)*(r.width/64),
        r.aq2+size_t(r.pair)*(r.width/2),r.as2+size_t(r.pair)*(r.width/64),r.col,threadIdx.x&31U);
    else if(r.aq) iu4_quant128(v,r.aq+size_t(r.pair)*(r.width/2),r.as+size_t(r.pair)*(r.width/64),r.col,threadIdx.x&31U);
    else {
        #pragma unroll
        for(unsigned j=0;j<4;++j) r.fp16[size_t(r.pair)*r.width+r.col+(threadIdx.x&31U)+32*j]=_Float16(v[j]);
    }
}
// dst += src over n floats (unfused W4A8 control: the remainder-plane GEMM output joins the first plane's).
template<class T> __global__ __launch_bounds__(256) void reference_add_kernel(T* dst,const T* src,size_t n) {
    const size_t i=size_t(blockIdx.x)*256U+threadIdx.x;
    if(i<n) dst[i]+=src[i];
}
template<class Schedule,class=void> struct ReferenceInputMatrices : std::integral_constant<unsigned,1> {};
template<class Schedule> struct ReferenceInputMatrices<Schedule,std::void_t<decltype(Schedule::kInputMatrices)>>
    : std::integral_constant<unsigned,Schedule::kInputMatrices> {};
// Multiple matrices preserve a cached row's outer slot loop and unrolled matrix loop.
// The default keeps the existing reference schedule traversal unchanged.
template<class Schedule> __global__ __launch_bounds__(256) void reference_input_kernel(Schedule s) {
    if constexpr (ReferenceInputMatrices<Schedule>::value>1) {
        const unsigned token=s.token(),col=s.col();
        float cached[4];
        #pragma unroll
        for(int j=0;j<4;++j) cached[j]=s.b.x[size_t(token)*Schedule::Binding::H+col+(threadIdx.x&31U)+32U*static_cast<unsigned>(j)];
        const auto rows=s.prepare();
        for(unsigned i=0;i<Schedule::kInputRows;++i) {
            if(!s.admit(rows,i)) continue;
            #pragma unroll
            for(unsigned m=0;m<ReferenceInputMatrices<Schedule>::value;++m) {
                ReferenceRow r{};
                s.input(rows,i,m,r);
                float v[4]; InputPolicy<typename Schedule::Binding>{s.b}.load(r.expert,s.source_pair(r.pair),r.m,r.col,v,cached);
                reference_store(r,v);
            }
        }
    } else {
    ReferenceRow r{};
    float cached[4];
    if constexpr (Schedule::kCachedInput) {
        if(!s.input(0,r)) return;
        const unsigned token=s.b.token(s.source_pair(r.pair));
        #pragma unroll
        for(unsigned j=0;j<4;++j) cached[j]=s.b.x[size_t(token)*Schedule::Binding::H+r.col+(threadIdx.x&31U)+32*j];
    }
    for(unsigned i=0;s.input(i,r);++i) {
        float v[4]; InputPolicy<typename Schedule::Binding>{s.b}.load(r.expert,s.source_pair(r.pair),r.m,r.col,v,Schedule::kCachedInput ? cached : nullptr);
        reference_store(r,v);
    }
    }
}
template<class Schedule> __global__ __launch_bounds__(256) void reference_activation_kernel(Schedule s) {
    ReferenceRow r{};
    for(unsigned i=0;s.activation(i,r);++i) {
        float g[4],u[4],v[4];
        #pragma unroll
        for(unsigned j=0;j<4;++j) { const unsigned c=r.col+(threadIdx.x&31U)+32*j;g[j]=r.gu[size_t(r.pair)*2*r.width+c];u[j]=r.gu[(size_t(r.pair)*2+1)*r.width+c]; }
        EpiloguePolicy<typename Schedule::Binding>{s.b}.activation(r.expert,r.col,g,u,v);
        reference_store(r,v);
    }
}
template<class Binding> __global__ __launch_bounds__(256) void reference_combine_kernel(Binding b,const float* dn,unsigned rows,unsigned topk) {
    const unsigned q=blockIdx.x*8+threadIdx.x/32,blocks=Binding::H/128,lane=threadIdx.x&31U;
    if(q>=rows*blocks) return;
    const unsigned tok=q/blocks,col=(q%blocks)*128;
    float acc[4]={};
    for(unsigned k=0;k<topk;++k) {
        const unsigned pair=b.slot(tok,k);const float weight=b.weight(pair);
        #pragma unroll
        for(unsigned j=0;j<4;++j) acc[j]=__builtin_fmaf(weight,dn[size_t(pair)*Binding::H+col+lane+32*j],acc[j]);
    }
    const unsigned shared=b.slot(tok,topk);
    #pragma unroll
    for(unsigned j=0;j<4;++j) b.out[size_t(tok)*Binding::H+col+lane+32*j]=acc[j]+dn[size_t(shared)*Binding::H+col+lane+32*j];
}
// Existing FP16 shared experts can use the same policy arithmetic without entering A4.
template<class Binding> __global__ void fp16_input_rows(Binding b,_Float16* input,unsigned rows,unsigned expert,unsigned stride,unsigned slot) {
    const unsigned lane=threadIdx.x&31U,wave=threadIdx.x/32,col=(blockIdx.x*8+wave)*128;
    for(unsigned r=0;r<16;++r) {
        const unsigned token=blockIdx.y*16+r;
        if(token>=rows) return;
        const unsigned pair=token*stride+slot;
        float v[4]; InputPolicy<Binding>{b}.load(expert,pair,blockIdx.z,col,v);
        #pragma unroll
        for(unsigned j=0;j<4;++j) input[(size_t(pair)*2+blockIdx.z)*Binding::H+col+lane+32*j]=_Float16(v[j]);
    }
}
template<class Binding> __global__ void fp16_activation_rows(Binding b,const float* gu,_Float16* at,unsigned rows,unsigned expert,unsigned stride,unsigned slot) {
    const unsigned lane=threadIdx.x&31U,wave=threadIdx.x/32,col=blockIdx.x*128;
    for(unsigned r=wave;r<16;r+=8) {
        const unsigned token=blockIdx.y*16+r;
        if(token>=rows) return;
        const unsigned pair=token*stride+slot;
        float g[4],u[4],v[4];
        #pragma unroll
        for(unsigned j=0;j<4;++j) { const unsigned c=col+lane+32*j;g[j]=gu[size_t(pair)*2*Binding::I+c];u[j]=gu[(size_t(pair)*2+1)*Binding::I+c]; }
        EpiloguePolicy<Binding>{b}.activation(expert,col,g,u,v);
        #pragma unroll
        for(unsigned j=0;j<4;++j) at[size_t(pair)*Binding::I+col+lane+32*j]=_Float16(v[j]);
    }
}
// Paired gate/up reads the interleaved pre-quantised A4 image written once by the shared input
// pre-pass (row 2*pair+m, m=0 gate, m=1 up): the pass selects the row, so the K loop never
// re-gathers, re-transforms or re-quantises the input. Bytes equal the LDS-staged ones.
// GSave: the paired pass-0 gate values (FP32, exact) go to a per-pair global scratch row
// (gsave[pair*I + col]) instead of registers, so BM128 paired does not spill; the same thread
// reads its own stores back in pass 1 (program order, L2-resident), no barrier needed.
// Fp16 (paired only): the SwiGLU epilogue stores the down input as FP16 rows (at16[pair*I + col], the
// values reference_store / fp16_activation_rows write) instead of the A4 image, for a routed down that
// keeps FP16 activations (gate/up-only INT4 layers). Same activation arithmetic, no quantiser.
template<class Binding,bool Pair,bool GSave=false,bool Fp16=false> struct Chain {
    static constexpr bool kActive = true;
    static constexpr int kPasses = Pair ? 2 : 1;
    static constexpr uint32_t kPassRow = Pair ? 1 : 0;
    static constexpr bool kSaveRegs = Pair && !GSave;
    Binding b; Weights w;
    uint8_t* aq; float* as;
    float* gsave = nullptr;
    _Float16* at16 = nullptr;
    unsigned expert = 0;
    __device__ __forceinline__ void bind(unsigned e, int pass, const uint32_t* __restrict__& q, const _Float16*& s, const uint16_t* __restrict__& svh) {
        expert=e;
        // Select, never index, the by-value argument arrays with the runtime pass: a dynamic index
        // would copy the whole kernel argument to scratch.
        auto pick=[&](const auto (&v)[3]) { if constexpr (Pair) return pass==0 ? v[0] : v[1]; else return v[2]; };
        q=reinterpret_cast<const uint32_t*>(pick(w.q)+size_t(e)*pick(w.q_stride));
        s=pick(w.s)+size_t(e)*pick(w.s_stride); svh=pick(w.svh)[e];
    }
    __device__ __forceinline__ void store(unsigned pos,unsigned col,int pass,int slot,float (&v)[4],float* saved) const {
        const unsigned pair=b.pair(pos);
        if constexpr (Pair) {
            float* gp=nullptr;
            if constexpr (GSave) gp=gsave+size_t(pair)*Binding::I+col+(threadIdx.x&31U);
            if (!pass) {
                #pragma unroll
                for (unsigned j=0;j<4;++j) { if constexpr (GSave) gp[32*j]=v[j]; else saved[unsigned(slot)*4+j]=v[j]; }
            } else {
                float g[4],a[4];
                #pragma unroll
                for (unsigned j=0;j<4;++j) { if constexpr (GSave) g[j]=gp[32*j]; else g[j]=saved[unsigned(slot)*4+j]; }
                EpiloguePolicy<Binding>{b}.activation(expert,col,g,v,a);
                if constexpr (Fp16) {
                    _Float16* row=at16+size_t(pair)*Binding::I+col+(threadIdx.x&31U);
                    #pragma unroll
                    for (unsigned j=0;j<4;++j) row[32*j]=_Float16(a[j]);
                } else iu4_quant128(a,aq+size_t(pair)*(Binding::I/2),as+size_t(pair)*(Binding::I/64),col,threadIdx.x&31U);
            }
        } else EpiloguePolicy<Binding>{b}.contribution(pair,col,v);
    }
};
template<int Plan,bool Pair,bool GSave,class Binding,bool Fp16=false> hipError_t launch(const Binding& b,const Weights& w,
    const IU4GroupTile* tiles,const uint32_t* ntiles,unsigned max_tiles,const uint8_t* gq,const float* gs,
    uint8_t* aq,float* as,float* gsave,hipStream_t stream,_Float16* at16=nullptr) {
    static_assert(!Fp16 || Pair, "FP16 down input is the paired epilogue's output");
    constexpr auto c=iu4_tiles[Plan];
    using G=wmma_tiled::Geometry<c.bm,c.bn,c.wm,c.wn,PackedIU4>;
    using C=Chain<Binding,Pair,GSave,Fp16>;
    wmma_tiled::GroupedArgs gp; gp.tiles=tiles; gp.ntiles=ntiles; gp.col_scale=w.svh[0]; gp.map=b.map();
    // Paired: A is the pre-quantised gate/up image gq/gs, row 2*pair+pass (pass offset via kPassRow).
    // Down: A is the down image aq/as written by the paired epilogue, row = pair.
    gp.a_mul=Pair ? 2U : 1U;
    const unsigned K=Pair ? Binding::H : Binding::I, N=Pair ? Binding::I : Binding::H;
    const auto kernel=wmma_tiled::tiled_core<TiledContract<PackedIU4>,c.bm,c.bn,c.wm,c.wn,true,0,false,false,64,true,wmma_tiled::NoAdapter,C>;
    // Reject spills/insufficient residency before launching. No token/route readback.
    hipFuncAttributes attr{}; auto status=hipFuncGetAttributes(&attr,reinterpret_cast<const void*>(kernel));
    if (status!=hipSuccess) return status;
    int blocks=0; status=hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,G::threads,0);
    if (status!=hipSuccess) return status;
    if (attr.localSizeBytes || blocks*G::waves<4) return hipErrorNotSupported;
    kernel<<<dim3(max_tiles*(N/c.bn)),G::threads,0,stream>>>(reinterpret_cast<const uint32_t*>(w.q[0]),nullptr,nullptr,
        K,N,0,1,nullptr,nullptr,nullptr,1.f,0.f,1,Pair ? gq : aq,Pair ? gs : as,w.s[0],nullptr,gp,{},C{b,w,aq,as,gsave,at16});
    return hipGetLastError();
}
// gsave != nullptr (paired only): pass-0 gate values go to global scratch (pairs*I floats),
// which removes the register save and lets the paired GEMM run plan 2 (BM128) without spilling.
template<bool Pair,class Binding> hipError_t launch(int plan,const Binding& b,const Weights& w,
    const IU4GroupTile* tiles,const uint32_t* ntiles,unsigned max_tiles,const uint8_t* gq,const float* gs,
    uint8_t* aq,float* as,hipStream_t stream,float* gsave=nullptr) {
    if constexpr (Pair) if (gsave) switch(plan) {
        case 0:return launch<0,Pair,true>(b,w,tiles,ntiles,max_tiles,gq,gs,aq,as,gsave,stream);
        case 1:return launch<1,Pair,true>(b,w,tiles,ntiles,max_tiles,gq,gs,aq,as,gsave,stream);
        case 2:return launch<2,Pair,true>(b,w,tiles,ntiles,max_tiles,gq,gs,aq,as,gsave,stream);
        default:return hipErrorInvalidValue;
    }
    switch(plan) {
        case 0:return launch<0,Pair,false>(b,w,tiles,ntiles,max_tiles,gq,gs,aq,as,nullptr,stream);
        case 1:return launch<1,Pair,false>(b,w,tiles,ntiles,max_tiles,gq,gs,aq,as,nullptr,stream);
        case 2:return launch<2,Pair,false>(b,w,tiles,ntiles,max_tiles,gq,gs,aq,as,nullptr,stream);
        default:return hipErrorInvalidValue;
    }
}
// The chain's plan for a call (moe_i4_chain_plan.hpp). gsave_on: the caller's resolved STRIX_MOE_I4_CHAIN_GSAVE
// (gsave_enabled(), read once by the caller; nothing here reads the environment). scratch_floats: FP32 values of
// gate scratch the caller can lend for the paired launch; pair_slots: pair indices the binding can return.
template<class Binding> ChainPlan chain_plan(int requested,bool gsave_on,std::size_t scratch_floats,std::size_t pair_slots) {
    return chain_plan(requested,gsave_on,scratch_floats,chain_gsave_floats(pair_slots,Binding::I));
}
// Tile list capacity every launch of the plan fits in, the register-save fallback (plan 1 at most) included.
inline unsigned chain_max_tiles(const ChainPlan& p,std::size_t map_rows,int experts) {
    const auto pt=iu4_grouped_max_tiles(map_rows,experts,p.paired<1 ? p.paired : 1), dt=iu4_grouped_max_tiles(map_rows,experts,p.down);
    return unsigned(pt>dt ? pt : dt);
}
// ran: the plan the launches were enqueued on (the request, or the register-save form after a rejected scratch
// form), so a caller or fixture can tell a fallback from the requested route. Meaningful when status is success.
struct ChainStatus { hipError_t status; const char* stage; ChainPlan ran{}; };
// Paired gate/up then routed down on one plan decision: one tile build when the plan is shared (GSave admitted or
// plan <= 1), else the register-save paired launch on plan 1 and a second tile build for the down. gsave: the
// admitted scratch (used only when p.gsave). A scratch-form kernel the device does not admit (spill or residency,
// rejected before anything is enqueued) falls back to that register-save form. between(): called after the paired
// launch is enqueued (phase marks).
template<class Binding,class Between> ChainStatus launch_chain(const ChainPlan& p,const Binding& b,const Weights& w,
    const IU4Group* groups,int experts,IU4GroupTile* tiles,uint32_t* ntiles,std::size_t map_rows,const uint8_t* gq,const float* gs,
    uint8_t* aq,float* as,float* gsave,hipStream_t stream,Between between) {
    if (p.gsave && !gsave) return {hipErrorInvalidValue,"chain gate scratch",{}};
    int paired=p.paired;
    bool gsaved=p.gsave;
    auto pt=unsigned(iu4_grouped_max_tiles(map_rows,experts,paired));
    const auto dt=unsigned(iu4_grouped_max_tiles(map_rows,experts,p.down));
    auto status=build_iu4_grouped_tiles(groups,experts,paired,tiles,ntiles,pt,stream);
    if (status!=hipSuccess) return {status,"chain tiles",{}};
    status=launch<true>(paired,b,w,tiles,ntiles,pt,gq,gs,aq,as,stream,p.gsave ? gsave : nullptr);
    if (status==hipErrorNotSupported && p.gsave && paired>1) {
        paired=1; pt=unsigned(iu4_grouped_max_tiles(map_rows,experts,paired));
        status=build_iu4_grouped_tiles(groups,experts,paired,tiles,ntiles,pt,stream);
        if (status!=hipSuccess) return {status,"chain tiles",{}};
        status=launch<true>(paired,b,w,tiles,ntiles,pt,gq,gs,aq,as,stream);
        gsaved=false;
    }
    if (status!=hipSuccess) return {status,"chain paired gate/up",{}};
    between();
    if (paired!=p.down) {
        status=build_iu4_grouped_tiles(groups,experts,p.down,tiles,ntiles,dt,stream);
        if (status!=hipSuccess) return {status,"chain down tiles",{}};
    }
    status=launch<false>(p.down,b,w,tiles,ntiles,dt,nullptr,nullptr,aq,as,stream);
    return {status,"chain down",ChainPlan{paired,p.down,gsaved}};
}
template<class Binding> ChainStatus launch_chain(const ChainPlan& p,const Binding& b,const Weights& w,
    const IU4Group* groups,int experts,IU4GroupTile* tiles,uint32_t* ntiles,std::size_t map_rows,const uint8_t* gq,const float* gs,
    uint8_t* aq,float* as,float* gsave,hipStream_t stream) {
    return launch_chain(p,b,w,groups,experts,tiles,ntiles,map_rows,gq,gs,aq,as,gsave,stream,[] {});
}
// Paired gate/up whose epilogue writes the FP16 down input rows at16[pair*I] (no A4 image, no down call):
// one kernel reads the gate and up INT4 weights and applies the SwiGLU/down-transform epilogue.
template<class Binding> hipError_t launch_paired_fp16(int plan,const Binding& b,const Weights& w,
    const IU4GroupTile* tiles,const uint32_t* ntiles,unsigned max_tiles,const uint8_t* gq,const float* gs,
    _Float16* at16,hipStream_t stream,float* gsave=nullptr) {
    if (gsave) switch(plan) {
        case 0:return launch<0,true,true,Binding,true>(b,w,tiles,ntiles,max_tiles,gq,gs,nullptr,nullptr,gsave,stream,at16);
        case 1:return launch<1,true,true,Binding,true>(b,w,tiles,ntiles,max_tiles,gq,gs,nullptr,nullptr,gsave,stream,at16);
        case 2:return launch<2,true,true,Binding,true>(b,w,tiles,ntiles,max_tiles,gq,gs,nullptr,nullptr,gsave,stream,at16);
        default:return hipErrorInvalidValue;
    }
    switch(plan) {
        case 0:return launch<0,true,false,Binding,true>(b,w,tiles,ntiles,max_tiles,gq,gs,nullptr,nullptr,nullptr,stream,at16);
        case 1:return launch<1,true,false,Binding,true>(b,w,tiles,ntiles,max_tiles,gq,gs,nullptr,nullptr,nullptr,stream,at16);
        default:return hipErrorInvalidValue;  // register save: BM64 max
    }
}
} // namespace strix::core::moe_i4
