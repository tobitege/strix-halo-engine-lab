#include "strix/hip_check.hpp"
#include "strix/hip_qwen_dpp_reduce.hpp"
#include "strix/hip_mtp_fast_draft.hpp"
#include <cstdlib>
#include <hip/hip_fp16.h>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strix {
namespace {
constexpr unsigned hidden=2560,inter=640,experts=512,slots=11,routed=10,waves=8;
template<bool Q4> constexpr unsigned block_bytes=Q4?18U:34U;
template<bool Q4> constexpr unsigned gate_row_bytes=hidden/32U*block_bytes<Q4>;
template<bool Q4> constexpr unsigned down_row_bytes=inter/32U*block_bytes<Q4>;
using strix::hip_check;
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for(unsigned d=16;d;d>>=1U) v+=strix::qwen_dpp::xor32(v, d);
    return v;
}
__device__ __forceinline__ std::uint16_t to_bf16(float x) {
    const unsigned bits=__float_as_uint(x);
    if((bits&0x7f800000U)==0x7f800000U && (bits&0x007fffffU)) return static_cast<std::uint16_t>((bits>>16U)|0x0040U);
    return static_cast<std::uint16_t>((bits+0x7fffU+((bits>>16U)&1U))>>16U);
}
// Eight consecutive Q8_0 codes at feature (multiple of 8) of one row, dotted with x[0..7].
__device__ __forceinline__ float q8_dot8(const unsigned char* row,unsigned feature,const float* x,float sum) {
    const auto* block=row+(feature/32U)*34U;
    const float scale=__half2float(*reinterpret_cast<const __half*>(block));
    const auto* codes=reinterpret_cast<const signed char*>(block+2U+feature%32U);
    float local=0.0F;
#pragma unroll
    for(unsigned i=0;i<8;++i) local=fmaf(static_cast<float>(codes[i]),x[i],local);
    return fmaf(scale,local,sum);
}
// Q4_0 block: half scale, element j in the low nibble of byte j (j<16) or the high nibble of byte j-16.
__device__ __forceinline__ float q4_dot8(const unsigned char* row,unsigned feature,const float* x,float sum) {
    const auto* block=row+(feature/32U)*18U;
    const float scale=__half2float(*reinterpret_cast<const __half*>(block));
    const unsigned within=feature%32U,shift=within>=16U?4U:0U;
    const auto* codes=block+2U+(within&15U);
    float local=0.0F;
#pragma unroll
    for(unsigned i=0;i<8;++i) local=fmaf(static_cast<float>(static_cast<int>((codes[i]>>shift)&15U)-8),x[i],local);
    return fmaf(scale,local,sum);
}
template<bool Q4>
__device__ __forceinline__ float dot8(const unsigned char* row,unsigned feature,const float* x,float sum) {
    if constexpr(Q4) return q4_dot8(row,feature,x,sum); else return q8_dot8(row,feature,x,sum);
}
template<bool Q4>
__device__ __forceinline__ void mtp_fast_gate_up_body(const HipMtpFastMoeViews& v) {
    const unsigned slot=blockIdx.y,wave=threadIdx.x/32U,lane=threadIdx.x%32U,row=blockIdx.x*waves+wave;
    if(!slot && !blockIdx.x && threadIdx.x<routed &&
       (v.routes[threadIdx.x]>=experts || !isfinite(v.route_weights[threadIdx.x]))) atomicOr(v.status,1U);
    if(*v.status) return;
    float x[8*hidden/256U];
#pragma unroll
    for(unsigned c=0;c<hidden/256U;++c) {
        const uint4 bits=*reinterpret_cast<const uint4*>(v.input+c*256U+8U*lane);
        const unsigned w[4]={bits.x,bits.y,bits.z,bits.w};
#pragma unroll
        for(unsigned i=0;i<4;++i) {x[8*c+2*i]=__uint_as_float(w[i]<<16U);x[8*c+2*i+1]=__uint_as_float(w[i]&0xffff0000U);}
    }
    if(row>=inter) {
        if(slot!=routed || row!=inter) return;
        float s=0.0F;
#pragma unroll
        for(unsigned c=0;c<hidden/256U;++c)
#pragma unroll
            for(unsigned i=0;i<8;++i) s=fmaf(v.shared_scalar[c*256U+8U*lane+i],x[8*c+i],s);
        s=wave_sum(s);
        if(!lane) v.workspace[slots*inter]=1.0F/(1.0F+expf(-s));
        return;
    }
    const unsigned char *g,*u;
    if(slot<routed) {
        const auto e=v.routes[slot];
        if(e>=experts) {if(!lane) v.workspace[slot*inter+row]=0.0F;return;}
        const auto offset=(static_cast<std::uint64_t>(e)*inter+row)*gate_row_bytes<Q4>;
        g=static_cast<const unsigned char*>(v.gate)+offset;u=static_cast<const unsigned char*>(v.up)+offset;
    } else {
        g=static_cast<const unsigned char*>(v.shared_gate)+row*gate_row_bytes<Q4>;u=static_cast<const unsigned char*>(v.shared_up)+row*gate_row_bytes<Q4>;
    }
    float gs=0.0F,us=0.0F;
#pragma unroll
    for(unsigned c=0;c<hidden/256U;++c) {gs=dot8<Q4>(g,c*256U+8U*lane,x+8*c,gs);us=dot8<Q4>(u,c*256U+8U*lane,x+8*c,us);}
    gs=wave_sum(gs);us=wave_sum(us);
    if(!lane) v.workspace[slot*inter+row]=(gs/(1.0F+expf(-gs)))*us;
}
template<bool Q4>
__global__ __launch_bounds__(256) void mtp_fast_gate_up(HipMtpFastMoeViews v) {mtp_fast_gate_up_body<Q4>(v);}
// Draft batch: token z runs the one-token kernel's blocks on its own views (same body, same bits).
template<bool Q4>
__global__ __launch_bounds__(256) void mtp_fast_gate_up_rows(HipMtpFastMoeRows v) {mtp_fast_gate_up_body<Q4>(v.views[blockIdx.z]);}
// W waves per block (one output row each, as before) share one LDS copy of the activations.
template<bool Q4,unsigned W=waves>
__device__ __forceinline__ void mtp_fast_down_body(const HipMtpFastMoeViews& v) {
    __shared__ float act[slots*inter];
    __shared__ float weight[slots];
    __shared__ const unsigned char* rows[slots];
    if(*v.status) return;
    for(unsigned i=threadIdx.x;i<slots*inter;i+=32U*W) act[i]=v.workspace[i];
    if(threadIdx.x<slots) {
        const unsigned s=threadIdx.x;
        weight[s]=s<routed?v.route_weights[s]:v.workspace[slots*inter];
        rows[s]=s<routed?static_cast<const unsigned char*>(v.down)+static_cast<std::uint64_t>(v.routes[s])*hidden*down_row_bytes<Q4>:
            static_cast<const unsigned char*>(v.shared_down);
    }
    __syncthreads();
    const unsigned wave=threadIdx.x/32U,lane=threadIdx.x%32U,row=blockIdx.x*W+wave;
    float sum=0.0F;
    for(unsigned s=0;s<slots;++s) {
        const auto* r=rows[s]+static_cast<std::uint64_t>(row)*down_row_bytes<Q4>;
        float p=0.0F;
        for(unsigned group=lane;group<inter/8U;group+=32U) p=dot8<Q4>(r,8U*group,act+s*inter+8U*group,p);
        sum=fmaf(weight[s],p,sum);
    }
    sum=wave_sum(sum);
    if(!lane) v.output[row]=to_bf16(sum);
}
template<bool Q4,unsigned W=waves>
__global__ __launch_bounds__(32U*W) void mtp_fast_down(HipMtpFastMoeViews v) {mtp_fast_down_body<Q4,W>(v);}
template<bool Q4,unsigned W=waves>
__global__ __launch_bounds__(32U*W) void mtp_fast_down_rows(HipMtpFastMoeRows v) {mtp_fast_down_body<Q4,W>(v.views[blockIdx.z]);}
template<bool InBf16,bool OutBf16,bool Q4=false>
__global__ __launch_bounds__(256) void mtp_draft_q8_gemv(const void* input,const unsigned char* q8,std::uint32_t width,
    std::uint32_t rows,void* output) {
    const unsigned lane=threadIdx.x%32U,row=blockIdx.x*waves+threadIdx.x/32U;
    if(row>=rows) return;
    const auto* r=q8+static_cast<std::uint64_t>(row)*(width/32U)*block_bytes<Q4>;
    float sum=0.0F,x[8];
    for(unsigned base=8U*lane;base<width;base+=256U) {
        if constexpr(InBf16) {
            const uint4 bits=*reinterpret_cast<const uint4*>(static_cast<const std::uint16_t*>(input)+base);
            const unsigned w[4]={bits.x,bits.y,bits.z,bits.w};
#pragma unroll
            for(unsigned i=0;i<4;++i) {x[2*i]=__uint_as_float(w[i]<<16U);x[2*i+1]=__uint_as_float(w[i]&0xffff0000U);}
        } else {
            const auto* f=static_cast<const float*>(input)+base;
#pragma unroll
            for(unsigned i=0;i<8;++i) x[i]=f[i];
        }
        sum=dot8<Q4>(r,base,x,sum);
    }
    sum=wave_sum(sum);
    if(lane) return;
    if constexpr(OutBf16) static_cast<std::uint16_t*>(output)[row]=to_bf16(sum);
    else static_cast<float*>(output)[row]=sum;
}
// Draft batch: mtp_draft_q8_gemv over Count tokens and up to three weight segments (rows of segment s
// are outputs first[s]..first[s+1]-1, written to output[s][t]). The wave reads its weight row once and
// runs every token's unchanged chain on it: per token the same column groups, the same codes and scale,
// the same local/sum FMA order and the same wave tree as the one-token kernel.
template<bool Q4> __device__ __forceinline__ void decode8(const unsigned char* row,unsigned feature,float& scale,float* codes) {
    if constexpr(Q4) {
        const auto* block=row+(feature/32U)*18U;
        scale=__half2float(*reinterpret_cast<const __half*>(block));
        const unsigned within=feature%32U,shift=within>=16U?4U:0U;
        const auto* c=block+2U+(within&15U);
#pragma unroll
        for(unsigned i=0;i<8;++i) codes[i]=static_cast<float>(static_cast<int>((c[i]>>shift)&15U)-8);
    } else {
        const auto* block=row+(feature/32U)*34U;
        scale=__half2float(*reinterpret_cast<const __half*>(block));
        const auto* c=reinterpret_cast<const signed char*>(block+2U+feature%32U);
#pragma unroll
        for(unsigned i=0;i<8;++i) codes[i]=static_cast<float>(c[i]);
    }
}
template<bool InBf16> __device__ __forceinline__ void load8(const void* input,unsigned base,float* x) {
    if constexpr(InBf16) {
        const uint4 bits=*reinterpret_cast<const uint4*>(static_cast<const std::uint16_t*>(input)+base);
        const unsigned w[4]={bits.x,bits.y,bits.z,bits.w};
#pragma unroll
        for(unsigned i=0;i<4;++i) {x[2*i]=__uint_as_float(w[i]<<16U);x[2*i+1]=__uint_as_float(w[i]&0xffff0000U);}
    } else {
        const auto* f=static_cast<const float*>(input)+base;
#pragma unroll
        for(unsigned i=0;i<8;++i) x[i]=f[i];
    }
}
// Tokens come in groups of Count per block (blockIdx.y = group of `total` tokens), bounding the accumulators.
template<bool InBf16,bool OutBf16,bool Q4,unsigned Count>
__global__ __launch_bounds__(256) void mtp_draft_q8_gemv_rows(HipMtpDraftGemvRows g,std::uint32_t width,unsigned total) {
    const unsigned lane=threadIdx.x%32U,output=blockIdx.x*waves+threadIdx.x/32U;
    if(output>=g.first[g.segments]) return;
    const unsigned first=blockIdx.y*Count,valid=total-first<Count?total-first:Count;
    unsigned segment=0;
    while(output>=g.first[segment+1U]) ++segment;
    const unsigned row=output-g.first[segment];
    const auto* r=static_cast<const unsigned char*>(g.weights[segment])+static_cast<std::uint64_t>(row)*(width/32U)*block_bytes<Q4>;
    float sum[Count];
#pragma unroll
    for(unsigned t=0;t<Count;++t) sum[t]=0.0F;
    for(unsigned base=8U*lane;base<width;base+=256U) {
        float scale,codes[8];
        decode8<Q4>(r,base,scale,codes);
#pragma unroll
        for(unsigned t=0;t<Count;++t) {
            float x[8];
            load8<InBf16>(g.input[first+(t<valid?t:0U)],base,x);
            float local=0.0F;
#pragma unroll
            for(unsigned i=0;i<8;++i) local=fmaf(codes[i],x[i],local);
            sum[t]=fmaf(scale,local,sum[t]);
        }
    }
#pragma unroll
    for(unsigned t=0;t<Count;++t) sum[t]=wave_sum(sum[t]);
    if(lane) return;
#pragma unroll
    for(unsigned t=0;t<Count;++t) {
        if(t>=valid) break;
        if constexpr(OutBf16) static_cast<std::uint16_t*>(g.output[segment][first+t])[row]=to_bf16(sum[t]);
        else static_cast<float*>(g.output[segment][first+t])[row]=sum[t];
    }
}
// Q8_0 -> Q4_0 per 32-block, the draft head's rule: d=signed max/-8, q=min(15, v/d+8.5).
__global__ void mtp_q8_to_q4(const unsigned char* q8,std::uint64_t blocks,unsigned char* q4) {
    const auto b=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(b>=blocks) return;
    const auto* source=q8+b*34U;auto* target=q4+b*18U;
    const float scale=__half2float(*reinterpret_cast<const __half*>(source));
    float values[32],max=0.0F,amax=0.0F;
    for(unsigned i=0;i<32;++i) {
        values[i]=scale*static_cast<float>(static_cast<signed char>(source[2+i]));
        if(fabsf(values[i])>amax) {amax=fabsf(values[i]);max=values[i];}
    }
    const float d=max/-8.0F,inverse=d!=0.0F?1.0F/d:0.0F;
    *reinterpret_cast<__half*>(target)=__float2half(d);
    for(unsigned j=0;j<16;++j) {
        const auto lo=static_cast<unsigned>(fminf(15.0F,values[j]*inverse+8.5F));
        const auto hi=static_cast<unsigned>(fminf(15.0F,values[j+16]*inverse+8.5F));
        target[2+j]=static_cast<unsigned char>(lo|(hi<<4U));
    }
}
struct Q4Copy {
    const void* first;void* second;std::uint64_t elements,generation;
    ModelContentIdentity content;std::shared_ptr<const void> source_owner;
};
struct Q4Copies {std::mutex lock;std::vector<Q4Copy> copies;};
Q4Copies& q4_copies() {static Q4Copies c;return c;}
__global__ void mtp_dequantize_q8(const unsigned char* q8,std::uint64_t elements,float* output) {
    const auto i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=elements) return;
    const auto* block=q8+(i/32U)*34U;
    output[i]=__half2float(*reinterpret_cast<const __half*>(block))*static_cast<float>(static_cast<signed char>(block[2U+i%32U]));
}
// STRIX_MTP_DOWN_WAVES=8|16|32: output rows (waves) per block sharing the activation copy.
unsigned fast_down_waves() {
    static const unsigned n=[]{const char* e=std::getenv("STRIX_MTP_DOWN_WAVES");
        const unsigned v=e?static_cast<unsigned>(std::atoi(e)):16U;return v==8U||v==32U?v:16U;}();
    return n;
}
} // namespace
void launch_hip_mtp_dequantize_q8(const void* q8,std::uint64_t elements,float* output,void* stream) {
    if(!q8 || !output || !stream || !elements || elements%32U) throw std::invalid_argument("MTP Q8 dequantize views are invalid");
    hipLaunchKernelGGL(mtp_dequantize_q8,dim3(static_cast<unsigned>((elements+255U)/256U)),dim3(256),0,static_cast<hipStream_t>(stream),
        static_cast<const unsigned char*>(q8),elements,output);
    hip_check(hipGetLastError(),"MTP Q8 dequantize launch");
}
void register_hip_mtp_draft_q4(const void* q8,std::uint64_t elements,void* stream,
    std::shared_ptr<const void> source_owner,std::uint64_t generation,ModelContentIdentity content) {
    if(source_owner && (!generation || !content.valid())) throw std::invalid_argument("MTP Q4 retained source requires generation/content pins");
    if(!q8 || !elements || elements%32U || !stream) throw std::invalid_argument("MTP draft Q4 copy views are invalid");
    auto& c=q4_copies();const std::lock_guard guard(c.lock);
    // The table-backed CLI admits one pinned sidecar source per process. Retain
    // it with the converted images so recycled source addresses cannot match.
    if(source_owner) for(const auto& known:c.copies) if(known.source_owner &&
        (known.source_owner.get()!=source_owner.get() || known.content!=content))
        throw std::invalid_argument("YaRN MTP Q4 registry already owns another sidecar");
    for(const auto& known:c.copies) if(known.first==q8) {
        if(source_owner) require_same_mtp_q4_source({known.source_owner,known.generation,known.elements,known.content},
            {source_owner,generation,elements,content});
        return;
    }
    const auto blocks=elements/32U;void* q4=nullptr;
    hip_check(hipMalloc(&q4,blocks*18U),"MTP draft Q4 copy");
    hipLaunchKernelGGL(mtp_q8_to_q4,dim3(static_cast<unsigned>((blocks+255U)/256U)),dim3(256),0,static_cast<hipStream_t>(stream),
        static_cast<const unsigned char*>(q8),blocks,static_cast<unsigned char*>(q4));
    hip_check(hipGetLastError(),"MTP draft Q4 conversion launch");
    c.copies.push_back({q8,q4,elements,generation,content,std::move(source_owner)});
}
const void* hip_mtp_draft_q4(const void* q8) noexcept {
    auto& c=q4_copies();const std::lock_guard guard(c.lock);
    for(const auto& known:c.copies) if(known.first==q8) return known.second;
    return nullptr;
}
void launch_hip_mtp_draft_q8_gemv(const void* input,bool input_bf16,const void* q8,std::uint32_t width,
    std::uint32_t rows,void* output,bool output_bf16,void* stream,bool any_order) {
    if(!input || !q8 || !output || !stream || !rows || !width || width%32U ||
       reinterpret_cast<std::uintptr_t>(input)%(input_bf16?16U:4U) || reinterpret_cast<std::uintptr_t>(q8)%2U)
        throw std::invalid_argument("MTP draft Q8 GEMV views are invalid");
    const auto s=static_cast<hipStream_t>(stream);const dim3 grid((rows+waves-1U)/waves);
    const auto start=[&](auto kernel,const unsigned char* w) {
        if(any_order) hipExtLaunchKernelGGL(kernel,grid,dim3(256),0,s,nullptr,nullptr,hipExtAnyOrderLaunch,input,w,width,rows,output);
        else hipLaunchKernelGGL(kernel,grid,dim3(256),0,s,input,w,width,rows,output);
    };
    if(const auto* q4=static_cast<const unsigned char*>(hip_mtp_draft_q4(q8))) {
        if(input_bf16 && output_bf16) start(mtp_draft_q8_gemv<true,true,true>,q4);
        else if(input_bf16) start(mtp_draft_q8_gemv<true,false,true>,q4);
        else if(output_bf16) start(mtp_draft_q8_gemv<false,true,true>,q4);
        else start(mtp_draft_q8_gemv<false,false,true>,q4);
        hip_check(hipGetLastError(),"MTP draft Q4 GEMV launch");
        return;
    }
    const auto* w=static_cast<const unsigned char*>(q8);
    if(input_bf16 && output_bf16) start(mtp_draft_q8_gemv<true,true>,w);
    else if(input_bf16) start(mtp_draft_q8_gemv<true,false>,w);
    else if(output_bf16) start(mtp_draft_q8_gemv<false,true>,w);
    else start(mtp_draft_q8_gemv<false,false>,w);
    hip_check(hipGetLastError(),"MTP draft Q8 GEMV launch");
}
void launch_hip_mtp_draft_q8_gemv_rows(const HipMtpDraftGemvRows& rows,std::uint32_t count,bool input_bf16,std::uint32_t width,
    bool output_bf16,void* stream) {
    if(!count || count>hip_mtp_draft_batch_rows || !rows.segments || rows.segments>3U || !stream || !width || width%32U ||
       rows.first[0]!=0U)
        throw std::invalid_argument("MTP draft batch GEMV geometry is invalid");
    HipMtpDraftGemvRows q4=rows;
    bool all_q4=true,any_q4=false;
    for(unsigned s=0;s<rows.segments;++s) {
        if(!rows.weights[s] || rows.first[s+1U]<=rows.first[s] || reinterpret_cast<std::uintptr_t>(rows.weights[s])%2U)
            throw std::invalid_argument("MTP draft batch GEMV segment is invalid");
        q4.weights[s]=hip_mtp_draft_q4(rows.weights[s]);all_q4=all_q4 && q4.weights[s];any_q4=any_q4 || q4.weights[s];
        for(unsigned t=0;t<count;++t) if(!rows.output[s][t]) throw std::invalid_argument("MTP draft batch GEMV output is missing");
    }
    for(unsigned t=0;t<count;++t)
        if(!rows.input[t] || reinterpret_cast<std::uintptr_t>(rows.input[t])%(input_bf16?16U:4U))
            throw std::invalid_argument("MTP draft batch GEMV input is invalid");
    // Same weights as the one-token launcher (which decides per tensor): a segment set mixing Q4 copies
    // and Q8 weights is refused rather than guessed.
    if(any_q4 && !all_q4) throw std::invalid_argument("MTP draft batch GEMV segments mix Q4 copies and Q8 weights");
    const auto& g=all_q4?q4:rows;
    // STRIX_MTP_DRAFT_BATCH_GEMV_GROUP=1..8: tokens per block (default 4); larger groups are split across grid.y.
    static const unsigned group=[]{const char* v=std::getenv("STRIX_MTP_DRAFT_BATCH_GEMV_GROUP");
        const unsigned n=v?static_cast<unsigned>(std::atoi(v)):4U;return n>=1U&&n<=8U?n:4U;}();
    const auto per=std::min(group,count);
    const auto s=static_cast<hipStream_t>(stream);const dim3 grid((rows.first[rows.segments]+waves-1U)/waves,(count+per-1U)/per);
    const auto run=[&]<bool Q4,unsigned C>() {
        if(input_bf16 && output_bf16) hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_draft_q8_gemv_rows<true,true,Q4,C>),grid,dim3(256),0,s,g,width,count);
        else if(input_bf16) hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_draft_q8_gemv_rows<true,false,Q4,C>),grid,dim3(256),0,s,g,width,count);
        else if(output_bf16) hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_draft_q8_gemv_rows<false,true,Q4,C>),grid,dim3(256),0,s,g,width,count);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_draft_q8_gemv_rows<false,false,Q4,C>),grid,dim3(256),0,s,g,width,count);
    };
    const auto by_count=[&]<bool Q4>() {
        switch(per) {
        case 1:run.template operator()<Q4,1U>();break;case 2:run.template operator()<Q4,2U>();break;
        case 3:run.template operator()<Q4,3U>();break;case 4:run.template operator()<Q4,4U>();break;
        case 5:run.template operator()<Q4,5U>();break;case 6:run.template operator()<Q4,6U>();break;
        case 7:run.template operator()<Q4,7U>();break;default:run.template operator()<Q4,8U>();break;
        }
    };
    if(all_q4) by_count.template operator()<true>(); else by_count.template operator()<false>();
    hip_check(hipGetLastError(),"MTP draft batch GEMV launch");
}
void launch_hip_mtp_fast_moe_rows(std::span<const HipMtpFastMoeViews> views,void* stream) {
    if(views.empty() || views.size()>hip_mtp_draft_batch_rows || !stream) throw std::invalid_argument("MTP fast MoE batch size is invalid");
    HipMtpFastMoeRows q4{};bool all_q4=true;
    for(std::size_t t=0;t<views.size();++t) {
        const auto& v=views[t];
        if(!v.input || !v.routes || !v.route_weights || !v.gate || !v.up || !v.down || !v.shared_gate || !v.shared_up ||
           !v.shared_down || !v.shared_scalar || !v.workspace || !v.output || !v.status || reinterpret_cast<std::uintptr_t>(v.input)%16U)
            throw std::invalid_argument("MTP fast MoE batch views are incomplete or misaligned");
        auto c=v;
        c.gate=hip_mtp_draft_q4(v.gate);c.up=hip_mtp_draft_q4(v.up);c.down=hip_mtp_draft_q4(v.down);
        c.shared_gate=hip_mtp_draft_q4(v.shared_gate);c.shared_up=hip_mtp_draft_q4(v.shared_up);c.shared_down=hip_mtp_draft_q4(v.shared_down);
        all_q4=all_q4 && c.gate && c.up && c.down && c.shared_gate && c.shared_up && c.shared_down;
        q4.views[t]=c;
    }
    // The batch runs only the production Q4 form (the one-token launcher's first branch).
    if(!all_q4) throw std::invalid_argument("MTP fast MoE batch needs the registered Q4 copies");
    const auto s=static_cast<hipStream_t>(stream);const auto n=static_cast<unsigned>(views.size());
    hipLaunchKernelGGL(mtp_fast_gate_up_rows<true>,dim3((inter+waves)/waves,slots,n),dim3(256),0,s,q4);
    hip_check(hipGetLastError(),"MTP fast Q4 gate/up batch launch");
    const unsigned down_waves=fast_down_waves();
    if(down_waves==32U) hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_fast_down_rows<true,32U>),dim3(hidden/32U,1,n),dim3(1024),0,s,q4);
    else if(down_waves==16U) hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_fast_down_rows<true,16U>),dim3(hidden/16U,1,n),dim3(512),0,s,q4);
    else hipLaunchKernelGGL(mtp_fast_down_rows<true>,dim3(hidden/waves,1,n),dim3(256),0,s,q4);
    hip_check(hipGetLastError(),"MTP fast Q4 down batch launch");
}
void launch_hip_mtp_fast_moe(const HipMtpFastMoeViews& v,void* stream) {
    if(!v.input || !v.routes || !v.route_weights || !v.gate || !v.up || !v.down || !v.shared_gate || !v.shared_up ||
       !v.shared_down || !v.shared_scalar || !v.workspace || !v.output || !v.status || !stream ||
       reinterpret_cast<std::uintptr_t>(v.input)%16U)
        throw std::invalid_argument("MTP fast MoE views are incomplete or misaligned");
    const auto s=static_cast<hipStream_t>(stream);
    auto q4=v;
    q4.gate=hip_mtp_draft_q4(v.gate);q4.up=hip_mtp_draft_q4(v.up);q4.down=hip_mtp_draft_q4(v.down);
    q4.shared_gate=hip_mtp_draft_q4(v.shared_gate);q4.shared_up=hip_mtp_draft_q4(v.shared_up);q4.shared_down=hip_mtp_draft_q4(v.shared_down);
    if(q4.gate && q4.up && q4.down && q4.shared_gate && q4.shared_up && q4.shared_down) {
        hipLaunchKernelGGL(mtp_fast_gate_up<true>,dim3((inter+waves)/waves,slots),dim3(256),0,s,q4);
        hip_check(hipGetLastError(),"MTP fast Q4 gate/up launch");
        const unsigned down_waves=fast_down_waves();
        if(down_waves==32U) hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_fast_down<true,32U>),dim3(hidden/32U),dim3(1024),0,s,q4);
        else if(down_waves==16U) hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_fast_down<true,16U>),dim3(hidden/16U),dim3(512),0,s,q4);
        else hipLaunchKernelGGL(mtp_fast_down<true>,dim3(hidden/waves),dim3(256),0,s,q4);
        hip_check(hipGetLastError(),"MTP fast Q4 down launch");
        return;
    }
    hipLaunchKernelGGL(mtp_fast_gate_up<false>,dim3((inter+waves)/waves,slots),dim3(256),0,s,v);
    hip_check(hipGetLastError(),"MTP fast gate/up launch");
    hipLaunchKernelGGL(mtp_fast_down<false>,dim3(hidden/waves),dim3(256),0,s,v);
    hip_check(hipGetLastError(),"MTP fast down launch");
}
} // namespace strix
