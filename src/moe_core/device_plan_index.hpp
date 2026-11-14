#pragma once
#if defined(__HIPCC__)
#define STRIX_DP_INLINE __host__ __device__ __forceinline__
#else
#define STRIX_DP_INLINE inline
#endif
namespace strix::moe_core::device_plan {
STRIX_DP_INLINE unsigned scatter_position(unsigned expert_start,unsigned block_prefix,unsigned rank) {
    return expert_start+block_prefix+rank;
}
// Per-expert exclusive prefix; used by the GPU scan and its CPU replay.
STRIX_DP_INLINE unsigned scan_column(unsigned* counts,unsigned blocks,unsigned experts,unsigned expert) {
    unsigned run=0;
    for(unsigned b=0;b<blocks;++b) {
        unsigned& c=counts[static_cast<unsigned long long>(b)*experts+expert];
        const unsigned n=c; c=run; run+=n;
    }
    return run;
}
}
#undef STRIX_DP_INLINE
