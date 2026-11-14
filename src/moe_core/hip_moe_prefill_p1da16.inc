// qpf2k/moe-mi: P1 + DA16 (STRIX_QPF2K_MOE=257 with STRIX_QPF2K_MOE_DA16=1). Included after hip_moe_prefill_da16.inc.
// The owner's P1 ordered-claim launch (hip_moe_prefill_p1.inc, design-5.md: ticket decode, producer/consumer protocol,
// failure decision and strict control copied token for token) with the DA16 operand image: a gate/up item, after its body,
// counts its arrival on arrive[b] (separate from done[b]); the last of the G items of batch b converts b's h into the image
// (exactly the DA16 converter) BEFORE its own done[b] increment, so done[b] == G implies the image of b is complete and
// visible (its stores precede the release). Down items stage A from the image (the DA16 down body). Same bits as P1, DA16 and
// the base; the failure-path scratch rule of P1 (raw dots of a projection-failed call are never consumed) is unchanged.
template<bool Q51,unsigned Tune>
__device__ __forceinline__ void da16_down_item(const unsigned char* weights,const PrefillRows& view,unsigned hidden,unsigned width,
    const uint4* image,const unsigned p1_batch,const unsigned p1_tile,
    typename F16LdsOf<F16Tile<Tune>>::Stage* const p1_sb,typename F16LdsOf<F16Tile<Tune>>::Rows& p1_rows) {
#define STRIX_DA16_BATCH_TILE const unsigned batch=p1_batch,tile=p1_tile;
#define STRIX_DA16_GUARD
#define STRIX_DA16_LDS(S,R) S* const sb=p1_sb; R& rows=p1_rows;
#define STRIX_DA16_UNIFORM_A false
#include "hip_moe_da16_down_body.inc"
#undef STRIX_DA16_UNIFORM_A
#undef STRIX_DA16_BATCH_TILE
#undef STRIX_DA16_GUARD
#undef STRIX_DA16_LDS
}
template<bool Q5K,bool Q51,unsigned Tune>
__global__ __launch_bounds__(F16Tile<Tune>::Threads,(Tune==82U?8U:1U)) void p1_prefill_f16x_da16(const unsigned char* gate,const unsigned char* up,
    const unsigned char* down_weights,PrefillRows view,unsigned hidden,unsigned width,const uint4* a16,P1Args p,unsigned* arrive,uint4* image) {
    using L=F16LdsOf<F16Tile<Tune>>;
    static_assert(std::is_same_v<typename L::Stage,typename F16LdsOf<F16TileX<Tune,0U>>::Stage>&&
        std::is_same_v<typename L::Rows,typename F16LdsOf<F16TileX<Tune,0U>>::Rows>,"P1: gate/up and down share one LDS image");
    __shared__ __align__(16) typename L::Stage sb[1];
    __shared__ typename L::Rows rows;
    __shared__ unsigned claimed,proceed,last;
    __shared__ float scale_of[F16Tile<Tune>::Batch];
    const unsigned t=threadIdx.x;
    if(t==0U)claimed=__hip_atomic_fetch_add(p.counter,1U,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    const unsigned i=claimed,G=p.gtiles,D=p.dtiles,nb=p.batches,lag=p.lag;
    const unsigned head=G*min(lag,nb),full=nb>lag?nb-lag:0U;
    bool gu;unsigned b,tile;
    if(i<head){gu=true;b=i/G;tile=i%G;}
    else {
        const unsigned j=i-head;
        if(j<full*(G+D)){const unsigned r=j/(G+D),k=j%(G+D);if(k<G){gu=true;b=lag+r;tile=k;}else{gu=false;b=r;tile=k-G;}}
        else {const unsigned k=j-full*(G+D);gu=false;b=full+k/D;tile=k%D;}
    }
    if(b>=nb)return; // CTA-uniform (from LDS)
    const auto& csr=view.csr;
    if(gu) {
        project_f16x_body<Q5K,Tune,0U>(gate,up,view,hidden,width,a16,b,tile,sb,rows);
        // DA16: the body's guard (bit 8 masked; fixed before the launch otherwise) decides whether b has h at all.
        if(!((csr.summary[0]&~projection_nonfinite)||b>=csr.summary[2])) {
            __builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent");
            __syncthreads();
            if(t==0U)last=__hip_atomic_fetch_add(arrive+b,1U,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_AGENT)==G-1U?1U:0U;
            __syncthreads();
            if(last) { // CTA-uniform (LDS): the DA16 converter of hip_moe_prefill_da16.inc
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
                const unsigned begin=csr.begins[b],count=csr.begins[b+1U]-begin,per=width/8U,total=count*per;
                if(t<count)scale_of[t]=ldexpf(1.0F,f16_exponent(__uint_as_float(static_cast<unsigned>(at(view.down_sums,rows.row[t])[rows.slot[t]]))));
                __syncthreads();
                for(unsigned k=t;k<total;k+=2U*blockDim.x) {
                    const unsigned j=k+blockDim.x;
                    const bool two=j<total;
                    const unsigned a0=k/per,c0=k%per,a1=two?j/per:a0,c1=two?j%per:c0;
                    const float* h0=at(view.output,rows.row[a0])+2ULL*rows.slot[a0]*width+8U*c0;
                    const float* h1=at(view.output,rows.row[a1])+2ULL*rows.slot[a1]*width+8U*c1;
                    const uint4 lo0=*reinterpret_cast<const uint4*>(h0),hi0=*reinterpret_cast<const uint4*>(h0+4);
                    uint4 lo1{},hi1{};
                    if(two){lo1=*reinterpret_cast<const uint4*>(h1);hi1=*reinterpret_cast<const uint4*>(h1+4);}
                    image[static_cast<std::size_t>(begin+a0)*per+c0]=f16_from_f32(lo0,hi0,scale_of[a0]);
                    if(two)image[static_cast<std::size_t>(begin+a1)*per+c1]=f16_from_f32(lo1,hi1,scale_of[a1]);
                }
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent");
        __syncthreads();
        if(t==0U) {
            __hip_atomic_fetch_add(p.done+b,1U,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
            if(p.strict)__hip_atomic_fetch_add(p.done+nb,1U,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
        }
        return;
    }
    if(t==0U) {
        bool go=__hip_atomic_load(csr.summary,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT)==0U&&b<csr.summary[2];
        if(go) {
            unsigned polls=0U;
            const unsigned long long t0=p.stats?static_cast<unsigned long long>(wall_clock64()):0ULL;
            const unsigned* const w=p.strict?p.done+nb:p.done+b;
            const unsigned need=p.strict?G*nb:G;
            while(__hip_atomic_load(w,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT)!=need){__builtin_amdgcn_s_sleep(1);++polls;}
            const unsigned long long t1=p.stats?static_cast<unsigned long long>(wall_clock64()):0ULL; // before any stats work
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
            go=__hip_atomic_load(csr.summary,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT)==0U;
            if(p.stats&&polls){__hip_atomic_fetch_add(p.stats,1U,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
                __hip_atomic_fetch_add(p.stats+1,polls,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
                __hip_atomic_fetch_add(reinterpret_cast<unsigned long long*>(p.stats+4),t1-t0,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);}
        }
        if(p.stats)__hip_atomic_fetch_add(p.stats+(go?2:3),1U,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
        proceed=go?1U:0U;
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
    if(!proceed)return; // CTA-uniform (from LDS)
    da16_down_item<Q51,Tune>(down_weights,view,hidden,width,image,b,tile,sb,rows);
}
