// qpf2k/moe P1 (STRIX_QPF2K_MOE=257, prefill only, tunes 80/82; design 5): the routed-expert gate/up and
// down in ONE launch whose CTAs claim (kind, batch, tile) tickets from an ordered list, so the memory-bound down of a batch
// overlaps the compute-bound gate/up of later batches. Included by hip_moe_q4k_a8_known_rows.hip inside its anonymous
// namespace after down_prefill_f16x.
//
// Same bits as the base: every item runs the base kernel body (the same source files the kernels include, Core 0) on the same
// operands in the same K order; only the CTA -> (batch, tile) mapping and the start time of each down item change. A down
// item of batch b needs h and max |h| of every (row, slot) of b, i.e. the G gate/up items of b (each (row, slot) belongs to
// exactly one CSR batch); it waits for done[b] == G. Ordered list, lag L batches: round r = 0, 1, ...: the G gate/up items of
// batch r (r < nb), then the D down items of batch r - L (0 <= r - L < nb); L >= nb is the phase-separated control (every
// gate/up ticket before every down ticket, same kernel and resources). Tickets are claimed in list order by one atomic per
// CTA, so a down ticket is claimed only after every gate/up ticket of its batch was claimed by a running CTA, and gate/up
// items never wait: no deadlock for any grid size or co-resident finite work.
// Visibility: producer (gate/up): every thread release fence (agent; waits for its stores and no-return atomics), barrier,
// thread 0 release fetch_add(done[b]). Consumer (down): thread 0 polls done[b] with relaxed agent-scope atomic loads, then
// an acquire fence; barrier; every thread an acquire fence (agent: L0/GL1 invalidated) before the body reads h / max |h|.
// Failure paths: the base down skips every batch once any h is non-finite (summary[0] != 0 when the down kernel starts). Here
// thread 0 reads summary[0] atomically when the down item starts and again after its wait (CTA-uniform decision through LDS;
// the body's own guard is compiled out), so down items that started before another batch's failure write raw dots. The merge
// rejects the whole call then (header[0]) and never reads them; every other word is identical.
// counter = ticket counter, done[b] = finished gate/up items of batch b, done[batches] = all finished gate/up items (strict
// control only), stats (tests) = P1Stats words. strict (lag 65535, measurement control): every down item waits until ALL gate/up
// items finished (no overlap at all, same kernel and resources).
struct P1Args { unsigned* counter; unsigned* done; unsigned* stats; unsigned batches,lag,gtiles,dtiles,strict; };
// Stats words (tests, STRIX_QPF2K_MOE_P1STATS=1): [0] down items that waited, [1] polls, [2] down items that proceeded,
// [3] down items that skipped (failure), [4..5] u64 total wait in wall-clock ticks (hipDeviceAttributeWallClockRate kHz).
constexpr unsigned p1_stats_words=8U;
// The LDS image of one f16x CTA of tile shape T (stage + rows table).
template<class T> struct F16LdsOf {
    using Stage=F16Stage<T::Batch,T::Columns,T::LTune>;
    using Rows=F16Rows<T::Table?T::Batch:1U>;
};
// The base kernel bodies (the same source the kernels include) for one (batch, tile) item, LDS passed in.
template<bool Q5K,unsigned Tune,unsigned Core,bool Reuse=false,unsigned Tag=0U>
__device__ __forceinline__ void project_f16x_body(const unsigned char* gate,const unsigned char* up,const PrefillRows& view,
    unsigned hidden,unsigned outputs,const uint4* a16,const unsigned p1_batch,const unsigned p1_tile,
    typename F16LdsOf<F16TileX<Tune,Core>>::Stage* const p1_sb,typename F16LdsOf<F16TileX<Tune,Core>>::Rows& p1_rows) {
#define STRIX_F16X_BATCH_TILE const unsigned batch=p1_batch,tile=p1_tile;
#define STRIX_F16X_LDS(S,R) S* const sb=p1_sb; R& rows=p1_rows;
#define STRIX_F16X_HDR_REUSE Reuse
#include "strix/env_flags.hpp"
#include "hip_moe_f16x_project_body.inc"
#undef STRIX_F16X_HDR_REUSE
#undef STRIX_F16X_BATCH_TILE
#undef STRIX_F16X_LDS
}
template<bool Q51,unsigned Tune,unsigned Core>
__device__ __forceinline__ void down_f16x_body(const unsigned char* weights,const PrefillRows& view,unsigned hidden,unsigned width,
    const FuseArgT<Core>& fuse,const unsigned p1_batch,const unsigned p1_tile,
    typename F16LdsOf<F16Tile<Tune>>::Stage* const p1_sb,typename F16LdsOf<F16Tile<Tune>>::Rows& p1_rows) {
#define STRIX_F16X_BATCH_TILE const unsigned batch=p1_batch,tile=p1_tile;
#define STRIX_F16X_LDS(S,R) S* const sb=p1_sb; R& rows=p1_rows;
#define STRIX_F16X_DOWN_GUARD
#include "hip_moe_f16x_down_body.inc"
#undef STRIX_F16X_BATCH_TILE
#undef STRIX_F16X_LDS
#undef STRIX_F16X_DOWN_GUARD
}
// Counter pool: [0] ticket counter, [1 .. batches] done[b], [batches + 1] all finished gate/up items (strict control), last
// p1_stats_words words: wait statistics (tests, STRIX_QPF2K_MOE_P1STATS=1). Sized for the largest admitted batch bound (prefill_max_rows x top_k 10 / cap 64
// + 512 experts, rounded up).
inline unsigned p1_pool_words(){return static_cast<unsigned>(prefill_pool_rows()*10U/64U+1024U);}
// Largest batch bound the pool admits: ticket counter + done[0..batches] + the stats words.
inline unsigned p1_max_batches(){return p1_pool_words()-2U-p1_stats_words;}
inline unsigned* p1_counter_pool(bool setup = false) {
    static HipPrefillPool pool;
    return static_cast<unsigned*>(pool.get(static_cast<std::size_t>(p1_pool_words())*sizeof(unsigned),
        "P1 counters",setup,true));
}
// STRIX_QPF2K_MOE_LAG (tests / measurement; 1..65535, default 6): the lag L in batches (>= the batch count: every gate/up
// ticket before every down ticket; 65535: the strict control, see P1Args).
inline unsigned p1_lag() {
    static const unsigned lag=[]{const char* v=std::getenv("STRIX_QPF2K_MOE_LAG");const int n=v?std::atoi(v):6;
        return n>=1&&n<=65535?static_cast<unsigned>(n):6U;}();
    return lag;
}
inline bool p1_stats_enabled() {
    static const bool on=strix::envcfg::flag("STRIX_QPF2K_MOE_P1STATS", false);
    return on;
}
template<bool Q5K,bool Q51,unsigned Tune>
__global__ __launch_bounds__(F16Tile<Tune>::Threads,(Tune==82U?8U:1U)) void p1_prefill_f16x(const unsigned char* gate,const unsigned char* up,
    const unsigned char* down_weights,PrefillRows view,unsigned hidden,unsigned width,const uint4* a16,P1Args p) {
    using L=F16LdsOf<F16Tile<Tune>>;
    static_assert(std::is_same_v<typename L::Stage,typename F16LdsOf<F16TileX<Tune,0U>>::Stage>&&
        std::is_same_v<typename L::Rows,typename F16LdsOf<F16TileX<Tune,0U>>::Rows>,"P1: gate/up and down share one LDS image");
    __shared__ __align__(16) typename L::Stage sb[1];
    __shared__ typename L::Rows rows;
    __shared__ unsigned claimed,proceed;
    const unsigned t=threadIdx.x;
    if(t==0U)claimed=__hip_atomic_fetch_add(p.counter,1U,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    const unsigned i=claimed,G=p.gtiles,D=p.dtiles,nb=p.batches,lag=p.lag;
    // Ticket i -> (gate/up?, batch, tile): rounds 0 .. min(L, nb) - 1 hold gate/up items only, then nb - L rounds of G + D
    // items, then the down items of the last min(L, nb) batches.
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
    down_f16x_body<Q51,Tune,0U>(down_weights,view,hidden,width,FuseNone{},b,tile,sb,rows);
}
