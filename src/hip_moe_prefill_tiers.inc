// Opt-in DA16 batch service tiers. Two same-stream launches keep the narrow body's
// static resources independent of the 128-row body. CSR and its summary are read-only.
// STRIX_QPF2K_MOE_TIERS=1 splits tune 82 using the padded tune-80 narrow body;
// =2 uses tune 88 for the narrow list: 64-row A/table/scales, with XOR pitch 64
// instead of 72. The logical B panel is still 256 x K64, with identical halves.
// Mode 1 already sizes per-row LDS for 64 rows, but padding prevents a third CTA.
// Mode 2 changes only LDS addressing, not the per-wave tile or K16 order.
// The full list keeps tune 82; unsplit tunes 80/88 keep their existing path.
// STRIX_QPF2K_MOE_DOWN_LDS_EPI=1 independently enables the vector-store epilogue
// for DA16, including certificate FP32/BF16. Both controls default OFF and are
// sampled at startup; changing them requires a restart and pool setup.
// The r22 route-control rounding (STRIX_QPF2K_MOE_ROUTE_CTRL=1) has its own down
// kernel; neither the tier split nor the LDS epilogue reproduces its stores, so
// both controls stand down whenever it is requested.
inline bool prefill_route_ctrl_requested() {
    static const bool env=[]{const char* v=std::getenv("STRIX_QPF2K_MOE_ROUTE_CTRL");return v&&v[0]=='1'&&v[1]=='\0';}();
    return env;
}
inline int& prefill_tiers_override(){static int v=-1;return v;}
inline int prefill_tiers() {
    if(prefill_route_ctrl_requested())return 0;
    static const int env=[]{const char* v=std::getenv("STRIX_QPF2K_MOE_TIERS");
        return v&&(v[0]=='1'||v[0]=='2')&&v[1]=='\0'?v[0]-'0':0;}();
    return prefill_tiers_override()<0?env:prefill_tiers_override();
}
inline int& prefill_tiers_flag(){static thread_local int v=0;return v;}
inline int& prefill_down_lds_override(){static int v=-1;return v;}
inline bool& prefill_down_lds_flag(){static thread_local bool v=false;return v;}
inline const std::array<hipEvent_t,3>*& prefill_da16_events() {
    static thread_local const std::array<hipEvent_t,3>* events=nullptr;return events;
}
inline void prefill_da16_mark(unsigned index,hipStream_t stream) {
    if(const auto* events=prefill_da16_events())check(hipEventRecord((*events)[index],stream),"DA16 timing event");
    if(const auto* events=prefill_test_events())check(hipEventRecord(events[index],stream),"DA16 tile64 timing event");
}
inline bool prefill_down_lds() {
    static const bool env=[]{const char* v=std::getenv("STRIX_QPF2K_MOE_DOWN_LDS_EPI");return v&&v[0]=='1'&&v[1]=='\0';}();
    if(prefill_route_ctrl_requested())return false;
    return prefill_down_lds_override()<0?env:prefill_down_lds_override()==1;
}
inline bool prefill_down_lds_admitted(const PrefillRows& rows) {
    // The scalar contract permits dword-aligned row views. Only vector-store
    // views may use the 16-byte epilogue; other views retain scalar stores.
    return prefill_down_lds()&&reinterpret_cast<std::uintptr_t>(rows.rawdots.base)%16U==0U&&rows.rawdots.stride%16U==0U;
}
template<auto Kernel> void prefill_tier_resources() {
    static const bool once=[] {
        const char* v=std::getenv("STRIX_INTPREFILL_RESOURCES");
        if(v&&v[0]=='1') {
            hipFuncAttributes a{};
            check(hipFuncGetAttributes(&a,reinterpret_cast<const void*>(Kernel)),"DA16 tier resources");
            std::fprintf(stderr,"DA16 tier resource %s: vgpr=%d lds=%zu scratch=%zu\n",__PRETTY_FUNCTION__,a.numRegs,a.sharedSizeBytes,a.localSizeBytes);
        }
        return true;
    }();
    (void)once;
}
inline unsigned* prefill_tier_lists(bool setup=false) {
    static HipPrefillPool pool;
    return static_cast<unsigned*>(pool.get(2ULL*(da16_counter_words()+1U)*sizeof(unsigned),"DA16 tier lists",setup));
}
// list[0] is its length; list[1..] are ORIGINAL batch IDs. The grid is bounded
// by the existing host-side batch bound; no device count readback is necessary.
__global__ void build_prefill_tiers(PrefillRows view,unsigned* narrow,unsigned* full) {
    const auto& c=view.csr;
    if(c.summary[0]&~projection_nonfinite)return;
    for(unsigned b=blockIdx.x*blockDim.x+threadIdx.x;b<c.summary[2];b+=gridDim.x*blockDim.x) {
        const unsigned n=c.begins[b+1U]-c.begins[b];
        // Keep even a zero-length admitted batch: its column arrivals are still
        // observable to the DA16 producer. Padded slots are excluded by summary[2].
        unsigned* list=n<=64U?narrow:full;
        list[1U+atomicAdd(list,1U)]=b;
    }
}
template<bool Q5K,unsigned Tune,bool Reuse>
__global__ __launch_bounds__(F16Tile<Tune>::Threads) void project_prefill_da16_tier(const unsigned char* gate,const unsigned char* up,
    PrefillRows view,unsigned hidden,unsigned outputs,const uint4* a16,unsigned* arrive,uint4* image,const unsigned* list) {
    if constexpr(Tune==88U) {
        using L=F16LdsOf<F16TileX<Tune,0U>>;
        static_assert(F16Tile<Tune>::Batch==64U);
        static_assert(sizeof(typename L::Stage)+sizeof(typename L::Rows)+64U*sizeof(float)+sizeof(unsigned)<=128U*1024U/3U);
    }
    project_prefill_f16x_da16_entry<Q5K,Tune,Reuse,true,2U>(gate,up,view,hidden,outputs,a16,arrive,image,list);
}
template<bool Q51,unsigned Tune,bool Certificate,bool RouteBF16,bool LdsEpi>
__global__ __launch_bounds__(F16Tile<Tune>::Threads) void down_prefill_da16_tier(const unsigned char* weights,PrefillRows view,
    unsigned hidden,unsigned width,const uint4* image,const unsigned* list) {
    if constexpr(Tune==88U) {
        using L=F16LdsOf<F16Tile<Tune>>;
        static_assert(F16Tile<Tune>::Batch==64U);
        static_assert(sizeof(typename L::Stage)+sizeof(typename L::Rows)<=128U*1024U/3U);
    }
    if(list&&blockIdx.y>=list[0])return;
#define STRIX_DA16_BATCH_TILE const unsigned batch=list?list[1U+blockIdx.y]:blockIdx.y,tile=blockIdx.x;
#define STRIX_DA16_GUARD if(csr.summary[0]||batch>=csr.summary[2])return;
#define STRIX_DA16_LDS(S,R) __shared__ __align__(16) S sb[1]; __shared__ R rows;
#define STRIX_DA16_LDS_EPILOGUE LdsEpi
    constexpr bool RouteI8=false;   // one-byte route dots have their own entry (hip_moe_prefill_da16.inc)
    [[maybe_unused]] constexpr unsigned route_top_k=0U;
    if constexpr(Certificate) {
#define STRIX_DA16_CERTIFICATE
#include "moe_core/hip_moe_da16_down_body.inc"
#undef STRIX_DA16_CERTIFICATE
    } else {
#include "moe_core/hip_moe_da16_down_body.inc"
    }
#undef STRIX_DA16_BATCH_TILE
#undef STRIX_DA16_GUARD
#undef STRIX_DA16_LDS
#undef STRIX_DA16_LDS_EPILOGUE
}
template<unsigned Tune>
void launch_prefill_tier(const unsigned char* gate,const unsigned char* up,PrefillRows rows,
    unsigned hidden,unsigned width,const uint4* a16,unsigned* arrive,uint4* image,const unsigned* list,unsigned batches,
    bool q5k,bool reuse,hipStream_t stream) {
    const dim3 gu(width/(F16Tile<Tune>::Columns/2U),batches);
    const auto project=[&]<bool Q5K,bool Reuse>() {
        prefill_tier_resources<project_prefill_da16_tier<Q5K,Tune,Reuse>>();
        hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_da16_tier<Q5K,Tune,Reuse>),gu,dim3(F16Tile<Tune>::Threads),0,stream,
            gate,up,rows,hidden,width,a16,arrive,image,list);
    };
    if(q5k){if(reuse)project.template operator()<true,true>();else project.template operator()<true,false>();}
    else {if(reuse)project.template operator()<false,true>();else project.template operator()<false,false>();}
    check(hipGetLastError(),"DA16 tier gate/up");
    // Down is deliberately launched later, after BOTH gate/up tiers. A global
    // projection failure must suppress every down batch, including the first tier.
}
template<unsigned Tune>
void launch_prefill_down_tier(const unsigned char* down,PrefillRows rows,unsigned hidden,unsigned width,const uint4* image,
    const unsigned* list,unsigned batches,bool q51,bool certificate,hipStream_t stream) {
    const dim3 grid(hidden/F16Tile<Tune>::Columns,batches);
    const auto run=[&]<bool Q51,bool Cert,bool BF16,bool Epi>() {
        prefill_tier_resources<down_prefill_da16_tier<Q51,Tune,Cert,BF16,Epi>>();
        hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_da16_tier<Q51,Tune,Cert,BF16,Epi>),grid,dim3(F16Tile<Tune>::Threads),0,stream,
            down,rows,hidden,width,image,list);
    };
    const auto kind=[&]<bool Q51,bool Epi>() {
        if(certificate){if(rows.route_bf16)run.template operator()<Q51,true,true,Epi>();else run.template operator()<Q51,true,false,Epi>();}
        else run.template operator()<Q51,false,false,Epi>();
    };
    const bool lds=prefill_down_lds_admitted(rows);
    if(lds) {
        if(q51)kind.template operator()<true,true>();else kind.template operator()<false,true>();
    } else {
        if(q51)kind.template operator()<true,false>();else kind.template operator()<false,false>();
    }
    check(hipGetLastError(),"DA16 tier down");
    if(lds)prefill_down_lds_flag()=true;
}
