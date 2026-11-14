// qpf2k/moe-mi: routed prefill prologue and shared CSR (STRIX_QPF2K_MOE_ROUTE: resolver default 14, unset raw value 0).
// Included by hip_moe_q4k_a8_known_rows.hip inside its anonymous namespace after f16_prefill_prologue and PrefillCsrPolicy.
// Every word they write is the base kernels' word (same values, same set of written words, failure paths included); only
// the order of independent work and the number of serial memory round trips change.
//
// Bits 1 and 4: compatibility selectors; both now use moe_core::device_plan::PackedPlan, as does value 0.
// The removed serial, waves and waves_v2 builders produce the same CSR tables on admitted Qwen geometries.
// All masks 0..15 are accepted; other values, negative values (except override -1 = environment), and malformed
// environment strings error. Bit 2 and bit 8 retain their prologue/merge meanings below.
//
// Bit 2: f16_prefill_prologue_v2. The base runs clear -> barrier -> route checks + a 2-byte strided max pass -> barrier ->
// thread 0 exponent -> barrier -> a second read of the row to convert it. v2 issues the row's 16-byte loads first and keeps
// them in registers, clears with 16-byte stores (when the clear view is 16-byte aligned, else the base loop), checks routes
// with one wave ballot, and every thread folds the 8 wave maxima itself: one barrier, one read of the row. The max is the
// same value (fmaxf over the same set of |x|: order-free, NaNs ignored the same way, start 0), so the exponent, the FP16
// image (f16_from_bf16 on the same words), sums, status and the zeroed max |h| words are the base's bytes.
// Bit 8: the prologue does not zero the accumulation view (84 MB per call at T 8192); the caller's merge writes those zeros
// on its skip paths instead (known_moe_prefill_merge_kernel_z: header[0] or the row status set, or an invalid route), where
// the base merge returns without a store. A successful row is overwritten by the merge in both paths, so the view holds the
// same bytes after the merge in every case (only when clear_words == hidden, i.e. exactly the words the merge covers).
inline bool& prefill_certificate_flag(){static thread_local bool v=false;return v;}
inline bool& prefill_clear_skipped_flag(){static thread_local bool v=false;return v;}
inline int& prefill_route_override(){static int v=-1;return v;}
inline ::strix::moe_core::RoutePrune& prefill_route_prune_slot(){static thread_local ::strix::moe_core::RoutePrune v{};return v;}
inline ::strix::moe_core::RoutePrune prefill_route_prune_state(){return prefill_route_prune_slot();}
// Per-call (this thread): bit 1 DA16, bit 2 ROUTE defaulted ON but off for this GSQ call (see enqueue_prefill).
inline unsigned& prefill_moe_gsq_off(){static thread_local unsigned v=0U;return v;}
inline unsigned prefill_route() {
    static const unsigned env=[]{
        const char* v=std::getenv("STRIX_QPF2K_MOE_ROUTE");
        unsigned n=0U;
        if(v) {
            if(!*v)throw std::invalid_argument("STRIX_QPF2K_MOE_ROUTE must be a mask in 0..15");
            for(;*v;++v) {
                if(*v<'0'||*v>'9')throw std::invalid_argument("STRIX_QPF2K_MOE_ROUTE must be a mask in 0..15");
                n=n*10U+static_cast<unsigned>(*v-'0');
                if(n>15U)throw std::invalid_argument("STRIX_QPF2K_MOE_ROUTE must be a mask in 0..15");
            }
        }
        return n;
    }();
    const int o=prefill_route_override();
    if(o < -1 || o > 15)throw std::invalid_argument("prefill route override must be -1 or a mask in 0..15");
    if(prefill_moe_gsq_off()&2U)return 0U;
    return o==-1?env:static_cast<unsigned>(o);
}
// Chunks = 16-byte words of a row per thread (max_hidden / 8 / 256, rounded up). Requires: hidden % 8 == 0, 16-byte aligned
// input rows, top_k <= 32 (host checks; else the base prologue runs). clear16: the clear view is 16-byte aligned and
// clear_words % 4 == 0 (else the base clear loop, same words).
constexpr unsigned route_prologue_chunks=(max_hidden/8U+255U)/256U;
__global__ __launch_bounds__(256) void f16_prefill_prologue_v2(PrefillRows v,unsigned hidden,unsigned top_k,unsigned experts,
    unsigned clear_words,uint4* a16,unsigned clear16) {
    const unsigned r=blockIdx.x,t=threadIdx.x,n16=hidden/8U;
    __shared__ float wave_max[8];
    const auto* x16=reinterpret_cast<const uint4*>(at(v.input,r));
    uint4 x[route_prologue_chunks];
#pragma unroll
    for(unsigned j=0;j<route_prologue_chunks;++j){const unsigned k=t+256U*j;x[j]=k<n16?x16[k]:uint4{0U,0U,0U,0U};}
    float* clear=at(v.clear,r);
    if(clear16){uint4* c4=reinterpret_cast<uint4*>(clear);for(unsigned i=t;i<clear_words/4U;i+=256U)c4[i]=uint4{0U,0U,0U,0U};}
    else for(unsigned i=t;i<clear_words;i+=256U)clear[i]=0.0F;
    unsigned route_bits=0U;
    if(t<32U) {
        const bool bad=t<top_k&&(at(v.routes,r)[t]>=experts||!isfinite(at(v.route_weights,r)[t]));
        route_bits=__ballot(bad)?invalid_route:0U;
    }
    if(t<top_k)at(v.down_sums,r)[t]=0;
    float m=0.0F;
    const auto fold=[&](unsigned w){m=fmaxf(m,fabsf(__uint_as_float(w<<16U)));m=fmaxf(m,fabsf(__uint_as_float(w&0xffff0000U)));};
#pragma unroll
    for(unsigned j=0;j<route_prologue_chunks;++j){fold(x[j].x);fold(x[j].y);fold(x[j].z);fold(x[j].w);}
    for(int stride=16;stride!=0;stride/=2)m=fmaxf(m,strix::qwen_dpp::xor32(m, stride));
    if((t&31U)==0U)wave_max[t/32U]=m;
    __syncthreads();
    m=wave_max[0];
    for(unsigned w=1;w<8U;++w)m=fmaxf(m,wave_max[w]);
    const int row_exponent=f16_exponent(m);
    if(t==0U) {
        at(v.sums,r)[0]=row_exponent;
        *at(v.status,r)=route_bits;
    }
    if(!a16)return;
    const float scale=ldexpf(1.0F,row_exponent);
    uint4* out=a16+static_cast<std::size_t>(r)*(hidden/8U);
#pragma unroll
    for(unsigned j=0;j<route_prologue_chunks;++j){const unsigned k=t+256U*j;if(k<n16)out[k]=f16_from_bf16(x[j],scale);}
}
// Certificate-only entry: base prologue above retains its exact source.
__global__ __launch_bounds__(256) void f16_prefill_prologue_v2_certificate(PrefillRows v,unsigned hidden,unsigned top_k,unsigned experts,
    unsigned clear_words,uint4* a16,unsigned clear16) {
    if(blockIdx.x==0U&&threadIdx.x==0U)*v.unsafe=0U; // earlier launch than every OR
    const unsigned r=blockIdx.x,t=threadIdx.x,n16=hidden/8U;
    __shared__ float wave_max[8];
    const auto* x16=reinterpret_cast<const uint4*>(at(v.input,r));
    uint4 x[route_prologue_chunks];
#pragma unroll
    for(unsigned j=0;j<route_prologue_chunks;++j){const unsigned k=t+256U*j;x[j]=k<n16?x16[k]:uint4{0U,0U,0U,0U};}
    float* clear=at(v.clear,r);
    if(clear16){uint4* c4=reinterpret_cast<uint4*>(clear);for(unsigned i=t;i<clear_words/4U;i+=256U)c4[i]=uint4{0U,0U,0U,0U};}
    else for(unsigned i=t;i<clear_words;i+=256U)clear[i]=0.0F;
    unsigned route_bits=0U;
    if(t<32U) {
        const bool bad=t<top_k&&(at(v.routes,r)[t]>=experts||!isfinite(at(v.route_weights,r)[t]));
        route_bits=__ballot(bad)?invalid_route:0U;
    }
    if(t<top_k)at(v.down_sums,r)[t]=0;
    float m=0.0F;
    const auto fold=[&](unsigned w){m=fmaxf(m,fabsf(__uint_as_float(w<<16U)));m=fmaxf(m,fabsf(__uint_as_float(w&0xffff0000U)));};
#pragma unroll
    for(unsigned j=0;j<route_prologue_chunks;++j){fold(x[j].x);fold(x[j].y);fold(x[j].z);fold(x[j].w);}
    for(int stride=16;stride!=0;stride/=2)m=fmaxf(m,strix::qwen_dpp::xor32(m, stride));
    if((t&31U)==0U)wave_max[t/32U]=m;
    __syncthreads();
    m=wave_max[0];
    for(unsigned w=1;w<8U;++w)m=fmaxf(m,wave_max[w]);
    const int row_exponent=f16_exponent(m);
    if(t==0U) {
        at(v.sums,r)[0]=row_exponent;
        *at(v.status,r)=route_bits;
    }
    if(!a16)return;
    const float scale=ldexpf(1.0F,row_exponent);
    uint4* out=a16+static_cast<std::size_t>(r)*(hidden/8U);
#pragma unroll
    for(unsigned j=0;j<route_prologue_chunks;++j){const unsigned k=t+256U*j;if(k<n16)out[k]=f16_from_bf16(x[j],scale);}
}
// Scratch is a pre-reserved capacity pool. PackedPlan::scratch_bytes keeps the historical layout:
// fail, status, repeat[512], blockcount[ceil(capacity*10/2048)][512]; only the header is cleared per call.
inline unsigned csr_mw_blocks_max(){return PrefillCsrPlan::blocks_for(prefill_pool_rows(),PrefillCsrPolicy::kTopK);}
inline unsigned* csr_mw_scratch(bool setup = false) {
    static HipPrefillPool pool;
    return static_cast<unsigned*>(pool.get(PrefillCsrPlan::scratch_bytes(prefill_pool_rows()),"CSR scratch",setup));
}
