#pragma once
// Sync-free device MoE route planner shared by the prefill engines (GLM exact MoE, MiniMax prefill MoE), plus the
// item queue that lets grouped launches run from a device-built item table without a host round trip.
//
// Planner: three launches on the caller's stream, no host synchronization, no allocation.
//   rank    one thread per row: canonical ids, rank of the row among the block's earlier rows holding each expert
//   scan    one block: per-expert block prefixes, expert counts, then the policy's group/tile/item tables
//   scatter one thread per row: each (row, slot) pair at off[expert] + block prefix + rank, plus the shared pair
// Expert e's pairs come out in ascending row order, experts in ascending id, the shared expert (id kExperts, every row)
// last: the order of the engines' host plans, so the tables are byte-equal to them.
//
// Policy contract (engines keep only this glue):
//   static constexpr unsigned kExperts, kTopK;          routed experts, slots per row
//   static constexpr unsigned kRows;                    rows per rank/scatter block (multiple of 32)
//   static constexpr bool kDropInvalid;                 invalid row: true = no routed pairs, false = P::fallback ids
//   using Out = ...;                                    engine tables, carried by value in Args
//   __device__ static bool load(const Args<P>&, unsigned row, unsigned (&e)[kTopK], float (&w)[kTopK]);
//                                                       canonical slot order; returns true when the row is invalid
//   __device__ static void fallback(unsigned (&e)[kTopK], float (&w)[kTopK]);  (kDropInvalid == false)
//   __device__ static void finish(const Args<P>&, const unsigned* count, const unsigned* start);
//                                                       scan block, every thread; count/start [kExperts + 1] in LDS
//   __device__ static void routed(const Args<P>&, unsigned row, unsigned slot, unsigned e, float w, unsigned pos);
//   __device__ static void shared(const Args<P>&, unsigned row, unsigned pos);
#include <hip/hip_runtime.h>
#include "device_plan_index.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::moe_core::device_plan {

template <class P> struct Args {
    unsigned rows;
    const std::uint32_t* ids;  // device [rows][kTopK]
    const float* weights;      // device [rows][kTopK] (optional for policies that ignore weights)
    unsigned* blk;             // [blocks][kExperts] counts, then exclusive per-expert block prefixes
    unsigned* rank;            // [rows][kTopK] rank among the block's earlier rows holding the same expert
    unsigned* off;             // [kExperts + 1] expert offsets ([kExperts] = shared), policy may use further words
    unsigned* flag;            // set (|= 1) when any row is invalid
    typename P::Out out;
};

constexpr std::size_t align256(std::size_t v) { return (v + 255) & ~std::size_t{255}; }
// Scratch of the planner for up to `rows` rows: blk, rank, off (+ 2 policy words), each 256-byte aligned.
template <class P> constexpr std::size_t blocks_for(unsigned rows) { return (std::size_t{rows} + P::kRows - 1U) / P::kRows; }
template <class P> constexpr std::size_t scratch_bytes(unsigned rows) {
    return align256(blocks_for<P>(rows) * P::kExperts * sizeof(unsigned)) +
           align256(std::size_t{rows} * P::kTopK * sizeof(unsigned)) + align256((P::kExperts + 2U) * sizeof(unsigned));
}
// Carve the planner scratch out of `base` (sized by scratch_bytes(capacity)).
template <class P> inline void bind_scratch(Args<P>& a, void* base, unsigned capacity) {
    auto* b = static_cast<unsigned char*>(base);
    a.blk = reinterpret_cast<unsigned*>(b);
    a.rank = reinterpret_cast<unsigned*>(b + align256(blocks_for<P>(capacity) * P::kExperts * sizeof(unsigned)));
    a.off = a.rank + align256(std::size_t{capacity} * P::kTopK * sizeof(unsigned)) / sizeof(unsigned);
}

template <class P>
__device__ __forceinline__ bool load_row(const Args<P>& a, unsigned r, unsigned (&e)[P::kTopK], float (&w)[P::kTopK]) {
    bool bad = false;
    if (r < a.rows) {
        bad = P::load(a, r, e, w);
    } else {
#pragma unroll
        for (unsigned k = 0; k < P::kTopK; ++k) e[k] = P::kExperts, w[k] = 0.0F;
    }
    if (bad) {
        if constexpr (P::kDropInvalid) {
#pragma unroll
            for (unsigned k = 0; k < P::kTopK; ++k) e[k] = P::kExperts;
        } else {
            P::fallback(e, w);
        }
    }
    return bad;
}

template <class P> __global__ __launch_bounds__(P::kRows) void rank_kernel(Args<P> p) {
    constexpr unsigned E = P::kExperts, K = P::kTopK, W = P::kRows / 32U;
    __shared__ unsigned cnt[W][E];
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U, r = blockIdx.x * P::kRows + tid;
    for (unsigned i = tid; i < W * E; i += P::kRows) (&cnt[0][0])[i] = 0;
    unsigned e[K];
    float w[K];
    if (load_row(p, r, e, w)) atomicOr(p.flag, 1U);
    __syncthreads();
    // Wave-local ranks: one pass per distinct expert held by any lane; lanes below hold earlier rows.
    unsigned rk[K] = {}, done = 0;
    constexpr unsigned all = (1U << K) - 1U;
#pragma unroll
    for (unsigned k = 0; k < K; ++k) if (e[k] >= E) done |= 1U << k;
    for (;;) {
        const unsigned pending = static_cast<unsigned>(__ballot(done != all));
        if (!pending) break;
        const unsigned leader = static_cast<unsigned>(__builtin_ctz(pending));
        unsigned mine = E;
#pragma unroll
        for (unsigned k = K; k-- > 0;) if (!(done >> k & 1U)) mine = e[k];
        const unsigned x = static_cast<unsigned>(__shfl(static_cast<int>(mine), static_cast<int>(leader)));
        unsigned slot = K;
#pragma unroll
        for (unsigned k = 0; k < K; ++k) if (e[k] == x) slot = k;
        const unsigned holders = static_cast<unsigned>(__ballot(slot < K));
        if (slot < K) {
            rk[slot] = cnt[wave][x] + static_cast<unsigned>(__builtin_popcount(holders & ((1U << lane) - 1U)));
            done |= 1U << slot;
        }
        if (lane == leader) cnt[wave][x] += static_cast<unsigned>(__builtin_popcount(holders));
    }
    __syncthreads();
    // Waves cover consecutive rows: prefix the wave counts per expert, publish the block's counts.
    for (unsigned x = tid; x < E; x += P::kRows) {
        unsigned run = 0;
        for (unsigned v = 0; v < W; ++v) { const unsigned c = cnt[v][x]; cnt[v][x] = run; run += c; }
        p.blk[std::size_t{blockIdx.x} * E + x] = run;
    }
    __syncthreads();
    if (r < p.rows)
#pragma unroll
        for (unsigned k = 0; k < K; ++k) p.rank[std::size_t{r} * K + k] = e[k] < E ? rk[k] + cnt[wave][e[k]] : 0U;
}

template <class P> __global__ __launch_bounds__(1024) void scan_kernel(Args<P> p, unsigned blocks) {
    constexpr unsigned E = P::kExperts;
    __shared__ unsigned count[E + 1], start[E + 1];
    const unsigned tid = threadIdx.x;
    for (unsigned x = tid; x < E; x += blockDim.x) {
        count[x] = scan_column(p.blk,blocks,E,x);
    }
    if (tid == 0) count[E] = p.rows;
    __syncthreads();
    if (tid == 0) {
        unsigned n = 0;
        for (unsigned x = 0; x <= E; ++x) {
            start[x] = n;
            p.off[x] = n;
            n += count[x];
        }
    }
    __syncthreads();
    P::finish(p, count, start);
}

template <class P> __global__ __launch_bounds__(P::kRows) void scatter_kernel(Args<P> p) {
    constexpr unsigned E = P::kExperts, K = P::kTopK;
    const unsigned r = blockIdx.x * P::kRows + threadIdx.x;
    if (r >= p.rows) return;
    unsigned e[K];
    float w[K];
    const bool bad = load_row(p, r, e, w);
    if (!(bad && P::kDropInvalid))
#pragma unroll
        for (unsigned k = 0; k < K; ++k)
            P::routed(p, r, k, e[k], w[k], scatter_position(p.off[e[k]],p.blk[std::size_t{blockIdx.x} * E + e[k]],p.rank[std::size_t{r} * K + k]));
    P::shared(p, r, p.off[E] + r);
}

// The three planner launches (rows > 0). Calls sharing scratch/tables must be ordered on `stream`.
template <class P> inline hipError_t enqueue(const Args<P>& a, hipStream_t stream) {
    const auto blocks = static_cast<unsigned>(blocks_for<P>(a.rows));
    rank_kernel<P><<<blocks, P::kRows, 0, stream>>>(a);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return e;
    scan_kernel<P><<<1, 1024, 0, stream>>>(a, blocks);
    if ((e = hipGetLastError()) != hipSuccess) return e;
    scatter_kernel<P><<<blocks, P::kRows, 0, stream>>>(a);
    return hipGetLastError();
}

// Packed-assignment schedule of the same route planner. Unlike the row-rank schedule above, it accepts repeated
// experts within a row, runtime experts/top-k and whole-plan failure, and keeps ranks in packed 16-bit LDS counters
// during scatter (no per-assignment rank scratch). Each expert's output still ascends by row, then original slot.
// This schedule preserves the count/scan/scatter loop structure of its original implementation.
// Policy contract:
//   kExperts (even, power of two): table capacity; kTopK: scratch capacity bound, actual values passed to enqueue.
//   kThreads (multiple of 32), kAssignmentsPerWave: launch/partition policy, kCountLimit (65535): packed counter limit.
//   kHistogramBins: capped unique-row and batch-size histogram bins; kInvalid: whole-plan failure bit.
//   View; tables(View): offsets, touched, experts, begins, rows, slots pointers (only live words are written).
//   routes(View,row), expert(View,row,slot), status(View,row), invalid<Certificate>(View,row,slot,expert,experts,bool& unsafe).
//   kept(View,row,slot): only for enqueue(..., prune=true) (moe_core/route_prune.hpp): false = the assignment is not planned.
//   certificate(View): mark an unsafe wave; failure(View,tid,bits,count,rows) and summary(View,tid,status,
//   touched,batches,count,duplicates,maximum,rows,hist): engine summary adapters, including all reserved words.
// Histograms count unique rows per touched expert and assignments per batch, with the last bin saturated.
template<class P> struct PackedPlan {
    static constexpr unsigned kSpan=P::kThreads/32U*P::kAssignmentsPerWave;
    static constexpr unsigned kHead=2U+P::kExperts; // fail, status, repeat[experts], block counts
    static_assert(P::kExperts%2U==0U && (P::kExperts&(P::kExperts-1U))==0U);
    static_assert(P::kThreads%32U==0U && P::kThreads<=1024U && P::kExperts<=P::kThreads);
    static_assert(P::kHistogramBins>0U && 2U*P::kHistogramBins<=P::kThreads);
    static_assert(P::kAssignmentsPerWave%32U==0U && kSpan<=65535U);
    static_assert(P::kCountLimit==65535U); // two 16-bit counters in every LDS word
    static constexpr unsigned blocks_for(unsigned rows,unsigned top_k) { return (rows*top_k+kSpan-1U)/kSpan; }
    static constexpr std::size_t scratch_bytes(unsigned capacity) {
        return (std::size_t{kHead}+std::size_t{blocks_for(capacity,P::kTopK)}*P::kExperts)*sizeof(unsigned);
    }
    // Scratch is reserved once for capacity; enqueue never allocates. rows/top_k must fit that capacity.
    static hipError_t enqueue(typename P::View v,unsigned rows,unsigned experts,unsigned top_k,unsigned cap,
                              unsigned* scratch,hipStream_t stream,bool certificate=false,bool prune=false);
};

template<class P,bool Certificate=false,bool Prune=false>
__global__ __launch_bounds__(P::kThreads) void csr_count_kernel(typename P::View v,unsigned rows,unsigned experts,unsigned top_k,unsigned* scratch) {
    const unsigned t=threadIdx.x,count=rows*top_k,k=blockIdx.x,first=k*PackedPlan<P>::kSpan,last=min(count,first+PackedPlan<P>::kSpan);
    __shared__ unsigned n_of[P::kExperts],failure,status;
    if(t<P::kExperts)n_of[t]=0U;
    if(t==0U){failure=0U;status=0U;}
    __syncthreads();
    bool unsafe=false;
    for(unsigned a=first+t;a<last;a+=P::kThreads) {
        const unsigned r=a/top_k,s=a%top_k,e=P::expert(v,r,s);
        if(P::template invalid<Certificate>(v,r,s,e,experts,unsafe))atomicOr(&failure,P::kInvalid);
        else {
            if constexpr(Prune)if(!P::kept(v,r,s))continue;
            atomicAdd(&n_of[e],1U);
            const unsigned* const route=P::routes(v,r);
            bool repeat=false;
            if constexpr(Prune){for(unsigned q=0;q<s;++q)repeat|=route[q]==e&&P::kept(v,r,q);}
            else
            for(unsigned q=0;q<s;++q)repeat|=route[q]==e;
            if(repeat)atomicAdd(scratch+2U+e,1U);
        }
    }
    if constexpr(Certificate) {
        const unsigned bad=static_cast<unsigned>(__ballot(unsafe));
        if(bad&&(t&31U)==0U)P::certificate(v);
    }
    const unsigned per=(rows+gridDim.x-1U)/gridDim.x;
    for(unsigned r=k*per+t;r<min(rows,(k+1U)*per);r+=P::kThreads){const unsigned s=P::status(v,r);if(s)atomicOr(&status,s);}
    __syncthreads();
    if(t==0U){if(failure)atomicOr(scratch,failure);if(status)atomicOr(scratch+1U,status);}
    if(t<P::kExperts)scratch[PackedPlan<P>::kHead+k*P::kExperts+t]=n_of[t];
}
template<class P,bool Prune=false>
__global__ __launch_bounds__(P::kThreads) void csr_scan_kernel(typename P::View v,unsigned rows,unsigned experts,unsigned top_k,unsigned cap,unsigned blocks,unsigned* scratch) {
    const auto& csr=P::tables(v);
    const unsigned t=threadIdx.x,count=rows*top_k;
    __shared__ unsigned n_of[P::kExperts],scan[3][P::kExperts],hist[2U*P::kHistogramBins],failure,status,duplicates,maximum;
    if(t==0){failure=scratch[0];status=scratch[1];duplicates=0U;maximum=0U;}
    if(t<2U*P::kHistogramBins)hist[t]=0U;
    __syncthreads();
    if(failure){P::failure(v,t,failure|status,count,rows);return;}
    if(t<P::kExperts) {
        unsigned n=0U;
        for(unsigned k=0;k<blocks;++k){unsigned* const c=scratch+PackedPlan<P>::kHead+k*P::kExperts+t;const unsigned x=*c;*c=n;n+=x;}
        if(n>P::kCountLimit)atomicOr(&failure,P::kInvalid);
        n_of[t]=n;
    }
    __syncthreads();
    if(failure){if(t==0U)atomicOr(scratch,failure);P::failure(v,t,failure|status,count,rows);return;}
    const unsigned n=t<P::kExperts?n_of[t]:0U,nb=(n+cap-1U)/cap;
    if(t<P::kExperts){scan[0][t]=n;scan[1][t]=n!=0U;scan[2][t]=nb;}
    __syncthreads();
    for(unsigned d=1U;d<P::kExperts;d<<=1U) {
        unsigned x[3]={};
        if(t>=d&&t<P::kExperts)for(unsigned k=0;k<3U;++k)x[k]=scan[k][t-d];
        __syncthreads();
        if(t>=d&&t<P::kExperts)for(unsigned k=0;k<3U;++k)scan[k][t]+=x[k];
        __syncthreads();
    }
    if(t<experts) {
        const unsigned off=scan[0][t]-n,touched=scan[1][t]-(n!=0U),batch=scan[2][t]-nb;
        csr.offsets[t]=off;
        if(t==experts-1U)csr.offsets[experts]=off+n;
        if(n) {
            csr.touched[touched]=t;atomicMax(&maximum,n);
            for(unsigned b=0;b<nb;++b){csr.experts[batch+b]=t;csr.begins[batch+b]=off+b*cap;atomicAdd(&hist[P::kHistogramBins+min(P::kHistogramBins,min(cap,n-b*cap))-1U],1U);}
            const unsigned unique=n-scratch[2U+t];
            atomicAdd(&hist[min(P::kHistogramBins,unique)-1U],1U);atomicAdd(&duplicates,n-unique);
        }
    }
    __syncthreads();
    // Prune: the plan holds the kept assignments only (the prefix scans above count kept ones).
    const unsigned planned=Prune?scan[0][P::kExperts-1U]:count;
    if(t==0U)csr.begins[scan[2][P::kExperts-1U]]=planned;
    P::summary(v,t,status,scan[1][P::kExperts-1U],scan[2][P::kExperts-1U],planned,duplicates,maximum,rows,hist);
}
template<class P,bool Prune=false>
__global__ __launch_bounds__(P::kThreads) void csr_scatter_kernel(typename P::View v,unsigned rows,unsigned top_k,const unsigned* scratch) {
    constexpr unsigned Waves=P::kThreads/32U;
    const auto& csr=P::tables(v);
    if(*reinterpret_cast<const volatile unsigned*>(scratch))return; // a failed CSR writes no rows / slots (as the base)
    const unsigned t=threadIdx.x,lane=t&31U,wave=t/32U,count=rows*top_k,k=blockIdx.x;
    const unsigned first=min(count,k*PackedPlan<P>::kSpan+wave*P::kAssignmentsPerWave),last=min(count,first+P::kAssignmentsPerWave);
    __shared__ unsigned position[Waves][P::kExperts/2U];
    for(unsigned x=t;x<Waves*(P::kExperts/2U);x+=P::kThreads)position[x/(P::kExperts/2U)][x%(P::kExperts/2U)]=0U;
    __syncthreads();
    const auto expert_at=[&](unsigned a){return P::expert(v,a/top_k,a%top_k);};
    const auto planned=[&](unsigned a){if constexpr(Prune)return P::kept(v,a/top_k,a%top_k);else{(void)a;return true;}};
    for(unsigned a=first+lane;a<last;a+=32U)if(planned(a))atomicAdd(&position[wave][expert_at(a)/2U],1U<<(16U*(expert_at(a)&1U)));
    __syncthreads();
    if(t<(P::kExperts/2U)) {
        unsigned lo=0U,hi=0U;
        for(unsigned w=0;w<Waves;++w){const unsigned c=position[w][t];position[w][t]=lo|hi<<16U;lo+=c&P::kCountLimit;hi+=c>>16U;}
    }
    __syncthreads();
    const unsigned* const base=scratch+PackedPlan<P>::kHead+k*P::kExperts;
    for(unsigned step=first;step<last;step+=32U) {
        const unsigned a=step+lane;
        const bool valid=a<last&&planned(a);
        const unsigned e=valid?expert_at(a):0U;
        unsigned active=static_cast<unsigned>(__ballot(valid)),peers=0U;
        while(active) {
            const unsigned leader_e=static_cast<unsigned>(__shfl(static_cast<int>(e),__ffs(active)-1,32));
            const unsigned same=static_cast<unsigned>(__ballot(valid&&e==leader_e))&active;
            if(valid&&e==leader_e)peers=same;
            active&=~same;
        }
        if(valid) {
            const unsigned slot=csr.offsets[e]+base[e]+((position[wave][e/2U]>>(16U*(e&1U)))&P::kCountLimit);
            const unsigned at_pos=slot+static_cast<unsigned>(__popc(peers&((1U<<lane)-1U)));
            csr.rows[at_pos]=a/top_k;csr.slots[at_pos]=a%top_k;
            if((peers&((1U<<lane)-1U))==0U)atomicAdd(&position[wave][e/2U],static_cast<unsigned>(__popc(peers))<<(16U*(e&1U)));
        }
    }
}

template<class P>
inline hipError_t PackedPlan<P>::enqueue(typename P::View v,unsigned rows,unsigned experts,unsigned top_k,unsigned cap,
                                        unsigned* scratch,hipStream_t stream,bool certificate,bool prune) {
    const unsigned blocks=blocks_for(rows,top_k);
    hipError_t e=hipMemsetAsync(scratch,0,kHead*sizeof(unsigned),stream);
    if(e!=hipSuccess)return e;
    if(prune) {
        if constexpr(requires { P::kept(v,0U,0U); }) {
            // Pruned plan (P::kept): the same three launches over the kept assignments; the default instantiations are untouched.
            if(certificate)csr_count_kernel<P,true,true><<<blocks,P::kThreads,0,stream>>>(v,rows,experts,top_k,scratch);
            else csr_count_kernel<P,false,true><<<blocks,P::kThreads,0,stream>>>(v,rows,experts,top_k,scratch);
            csr_scan_kernel<P,true><<<1,P::kThreads,0,stream>>>(v,rows,experts,top_k,cap,blocks,scratch);
            csr_scatter_kernel<P,true><<<blocks,P::kThreads,0,stream>>>(v,rows,top_k,scratch);
            return hipGetLastError();
        } else {
            return hipErrorInvalidValue;  // Optional policy hook: ordinary plans need not provide it.
        }
    }
    if(certificate)csr_count_kernel<P,true><<<blocks,P::kThreads,0,stream>>>(v,rows,experts,top_k,scratch);
    else csr_count_kernel<P><<<blocks,P::kThreads,0,stream>>>(v,rows,experts,top_k,scratch);
    csr_scan_kernel<P><<<1,P::kThreads,0,stream>>>(v,rows,experts,top_k,cap,blocks,scratch);
    csr_scatter_kernel<P><<<blocks,P::kThreads,0,stream>>>(v,rows,top_k,scratch);
    return hipGetLastError();
}

// ---------------------------------------------------------------- device item queue
// A grouped launch whose item table (cta_begin prefixes ascending, `*count` items, `*total` CTAs) was built on the
// device. The grid is a host bound of `*total` (the planner flags a total above it); virtual CTA = blockIdx.x and the
// surplus CTAs return at once. Each CTA runs exactly the work of the host-planned launch's CTA of the same index, so
// outputs are bitwise those of the host table, and the body keeps the host kernel's registers (no loop: a persistent
// queue raised the tail kernels' VGPRs by 25..45 and cost occupancy).
template <class ItemT> struct ItemQueue {
    const ItemT* items;
    const unsigned* count;
    const unsigned* total;
};

template <class ItemT>
__device__ __forceinline__ const ItemT& find_item(const ItemT* items, unsigned count, unsigned vb) {
    unsigned lo = 0, hi = count;  // last item with cta_begin <= vb
    while (hi - lo > 1) {
        const unsigned mid = (lo + hi) / 2;
        if (items[mid].cta_begin <= vb) lo = mid;
        else hi = mid;
    }
    return items[lo];
}

// body(item, bx, by, bz) with the host kernels' decomposition of the CTA index inside the item.
template <class ItemT, class Body> __device__ __forceinline__ void run_items(const ItemQueue<ItemT>& q, Body&& body) {
    if (blockIdx.x >= *q.total) return;
    const ItemT it = find_item(q.items, *q.count, blockIdx.x);
    const unsigned local = blockIdx.x - it.cta_begin;
    body(it, local % it.gx, (local / it.gx) % it.gy, local / (it.gx * it.gy));
}

}  // namespace strix::moe_core::device_plan
