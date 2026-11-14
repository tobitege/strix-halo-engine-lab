#include "strix/env_flags.hpp"
#include "strix/hip_argmax_core.hpp"
#include "strix/hip_check.hpp"
#include "strix/hip_qwen_dpp_reduce.hpp"
#include "strix/hip_mtp_draft_head.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>

namespace strix {
namespace {
constexpr unsigned threads=256;
constexpr std::uint32_t max_slots=16;
constexpr std::uint32_t max_batch=8;
// sum: sum of exp(logit - value) over the finite rows of the partial (softmax denominator at its max).
// second: the largest finite logit after the selected row (-inf when none; runner-up kernels only).
struct Partial {float value;std::uint32_t token,bad,valid;float sum,second;};
struct Reply {std::uint32_t tokens[max_slots];std::uint32_t status;float probabilities[max_slots];std::uint32_t completed;
    float runner_up[max_slots];};
void require(bool ok,const char* message) {if(!ok) throw std::invalid_argument(message);}
using strix::hip_check;
__device__ float block_sum(float* values,float value) {
    const auto lane=static_cast<unsigned>(threadIdx.x);
    values[lane]=value;__syncthreads();
    for(unsigned stride=threads/2;stride;stride>>=1U) {
        if(lane<stride) values[lane]+=values[lane+stride];
        __syncthreads();
    }
    return values[0];
}
// Shared first-max rule (strix::argmax, R2-07); sum/second ride along with the winner.
__device__ Partial choose(Partial a,const Partial b) {return strix::argmax::choose(a,b);}
// choose() that also keeps the runner-up value: the other partial's best and both seconds compete.
// Partials cover disjoint rows, so a valid winner's token tells which side it came from.
__device__ Partial choose_second(Partial a,const Partial b) {
    const auto best=choose(a,b);
    const float loser=a.valid && b.valid?(best.token==a.token?b.value:a.value):-INFINITY;
    const float second=fmaxf(fmaxf(a.second,b.second),loser);
    a=best;a.second=second;return a;
}
template<bool Second> __device__ __forceinline__ Partial combine(Partial a,const Partial b) {
    if constexpr(Second) return choose_second(a,b); else return choose(a,b);
}
__global__ __launch_bounds__(threads) void draft_gather_rows(const std::uint8_t* head,const std::uint32_t* ids,
    std::uint32_t first,std::uint32_t count,std::uint32_t row_bytes,std::uint8_t* compact) {
    const auto row=first+blockIdx.x;if(blockIdx.x>=count) return;
    const auto* source=head+static_cast<std::uint64_t>(ids[row])*row_bytes;
    auto* target=compact+static_cast<std::uint64_t>(row)*row_bytes;
    for(std::uint32_t i=threadIdx.x;i<row_bytes;i+=threads) target[i]=source[i];
}
// Q8_0 row -> Q4_0 row, one thread per 32-block: d=signed max/-8, q=clamp(v/d+8.5),
// low nibbles hold elements 0..15 and high nibbles 16..31.
__global__ __launch_bounds__(threads) void draft_gather_rows_q4(const std::uint8_t* head,const std::uint32_t* ids,
    std::uint32_t first,std::uint32_t count,std::uint32_t blocks,std::uint8_t* compact) {
    const auto row=first+blockIdx.x;if(blockIdx.x>=count) return;
    for(std::uint32_t b=threadIdx.x;b<blocks;b+=threads) {
        const auto* source=head+(static_cast<std::uint64_t>(ids[row])*blocks+b)*34U;
        auto* target=compact+(static_cast<std::uint64_t>(row)*blocks+b)*18U;
        const float scale=__half2float(*reinterpret_cast<const __half*>(source));
        float values[32],max=0.0F,amax=0.0F;
        for(unsigned i=0;i<32;++i) {
            values[i]=scale*static_cast<float>(static_cast<std::int8_t>(source[2+i]));
            if(fabsf(values[i])>amax) {amax=fabsf(values[i]);max=values[i];}
        }
        const float d=max/-8.0F,inverse=d!=0.0F?1.0F/d:0.0F;
        *reinterpret_cast<__half*>(target)=__float2half(d);
        for(unsigned j=0;j<16;++j) {
            const auto lo=static_cast<unsigned>(fminf(15.0F,values[j]*inverse+8.5F));
            const auto hi=static_cast<unsigned>(fminf(15.0F,values[j+16]*inverse+8.5F));
            target[2+j]=static_cast<std::uint8_t>(lo|(hi<<4U));
        }
    }
}
// Wave per row over Q4_0 rows; lanes own whole 32-blocks, then a wave tree.
__global__ __launch_bounds__(256) void draft_q4_head(const std::uint16_t* hidden,const std::uint8_t* rows,
    std::uint32_t blocks,std::uint32_t count,float* logits) {
    const auto lane=threadIdx.x%32U,row=blockIdx.x*8U+threadIdx.x/32U;
    if(row>=count) return;
    const auto* q=rows+static_cast<std::uint64_t>(row)*blocks*18U;
    float sum=0.0F;
    for(std::uint32_t b=lane;b<blocks;b+=32U) {
        const auto* block=q+b*18U;const auto* x=hidden+b*32U;
        float local=0.0F;
        for(unsigned j=0;j<16;++j) {
            const unsigned v=block[2+j];
            local=fmaf(static_cast<float>(static_cast<int>(v&15U)-8),__uint_as_float(static_cast<unsigned>(x[j])<<16U),local);
            local=fmaf(static_cast<float>(static_cast<int>(v>>4U)-8),__uint_as_float(static_cast<unsigned>(x[j+16])<<16U),local);
        }
        sum=fmaf(__half2float(*reinterpret_cast<const __half*>(block)),local,sum);
    }
    for(unsigned d=16;d;d>>=1U) sum+=strix::qwen_dpp::xor32(sum, d);
    if(!lane) logits[row]=sum;
}
// draft_q4_head's body for the draft batch's own-row (tail) kernel; the one-token kernel above stays as it was.
__device__ __forceinline__ void draft_q4_head_body(const std::uint16_t* hidden,const std::uint8_t* rows,
    std::uint32_t blocks,std::uint32_t count,float* logits,std::uint32_t row) {
    const auto lane=threadIdx.x%32U;
    if(row>=count) return;
    const auto* q=rows+static_cast<std::uint64_t>(row)*blocks*18U;
    float sum=0.0F;
    for(std::uint32_t b=lane;b<blocks;b+=32U) {
        const auto* block=q+b*18U;const auto* x=hidden+b*32U;
        float local=0.0F;
        for(unsigned j=0;j<16;++j) {
            const unsigned v=block[2+j];
            local=fmaf(static_cast<float>(static_cast<int>(v&15U)-8),__uint_as_float(static_cast<unsigned>(x[j])<<16U),local);
            local=fmaf(static_cast<float>(static_cast<int>(v>>4U)-8),__uint_as_float(static_cast<unsigned>(x[j+16])<<16U),local);
        }
        sum=fmaf(__half2float(*reinterpret_cast<const __half*>(block)),local,sum);
    }
    for(unsigned d=16;d;d>>=1U) sum+=strix::qwen_dpp::xor32(sum, d);
    if(!lane) logits[row]=sum;
}
// Draft batch. Rows [0, shared) hold the same ids in every session's compact copy (one gather rule from one
// Q8 head, so the same bytes): one wave reads each such row once and runs every session's unchanged chain
// on it (per session the same blocks per lane, the same code order, the same local/scale FMAs and tree as
// draft_q4_head), writing that session's logits[row]. Rows [shared, rows_t) of session t run the one-token
// body on its own copy (blockIdx.y = session).
struct DraftHeadRows {
    const std::uint16_t* hidden[max_batch];
    const std::uint8_t* compact[max_batch];
    float* logits[max_batch];
    std::uint32_t rows[max_batch];
    const std::uint32_t* ids[max_batch];
    Partial* partials[max_batch];
    std::uint32_t* slots[max_batch];
    Reply* reply[max_batch];
    std::uint32_t* done[max_batch];
};
template<unsigned Count>
__global__ __launch_bounds__(256) void draft_q4_head_shared(DraftHeadRows h,std::uint32_t blocks,std::uint32_t shared) {
    const auto lane=threadIdx.x%32U,row=blockIdx.x*8U+threadIdx.x/32U;
    if(row>=shared) return;
    const auto* q=h.compact[0]+static_cast<std::uint64_t>(row)*blocks*18U;
    float sum[Count];
#pragma unroll
    for(unsigned t=0;t<Count;++t) sum[t]=0.0F;
    for(std::uint32_t b=lane;b<blocks;b+=32U) {
        const auto* block=q+b*18U;
        float lo[16],hi[16];
#pragma unroll
        for(unsigned j=0;j<16;++j) {
            const unsigned v=block[2+j];
            lo[j]=static_cast<float>(static_cast<int>(v&15U)-8);hi[j]=static_cast<float>(static_cast<int>(v>>4U)-8);
        }
        const float scale=__half2float(*reinterpret_cast<const __half*>(block));
#pragma unroll
        for(unsigned t=0;t<Count;++t) {
            const auto* x=h.hidden[t]+b*32U;
            float local=0.0F;
#pragma unroll
            for(unsigned j=0;j<16;++j) {
                local=fmaf(lo[j],__uint_as_float(static_cast<unsigned>(x[j])<<16U),local);
                local=fmaf(hi[j],__uint_as_float(static_cast<unsigned>(x[j+16])<<16U),local);
            }
            sum[t]=fmaf(scale,local,sum[t]);
        }
    }
#pragma unroll
    for(unsigned t=0;t<Count;++t) for(unsigned d=16;d;d>>=1U) sum[t]+=strix::qwen_dpp::xor32(sum[t], d);
    if(lane) return;
#pragma unroll
    for(unsigned t=0;t<Count;++t) h.logits[t][row]=sum[t];
}
__global__ __launch_bounds__(256) void draft_q4_head_tails(DraftHeadRows h,std::uint32_t blocks,std::uint32_t shared) {
    const auto t=blockIdx.y;
    draft_q4_head_body(h.hidden[t],h.compact[t],blocks,h.rows[t],h.logits[t],shared+blockIdx.x*8U+threadIdx.x/32U);
}
template<bool Second> __device__ __forceinline__ void draft_argmax_finish_body(const Partial* partials,std::uint32_t count,
    std::uint32_t step,std::uint32_t* slots,Reply* reply) {
    __shared__ Partial values[threads];
    const auto lane=static_cast<unsigned>(threadIdx.x);
    __shared__ float sums[threads];
    Partial value{0.0F,0,0,0,0.0F,-INFINITY};
    for(std::uint32_t i=lane;i<count;i+=threads) value=combine<Second>(value,partials[i]);
    const auto best=strix::argmax::block_reduce<threads>(values,lane,value,[](const Partial& a,const Partial& b){return combine<Second>(a,b);});
    float sum=0.0F;
    for(std::uint32_t i=lane;i<count;i+=threads) {
        const auto& p=partials[i];if(p.valid && best.valid) sum+=p.sum*expf(p.value-best.value);
    }
    sum=block_sum(sums,sum);
    if(!lane) {
        const auto failed=strix::argmax::status(best);
        // A failed row still feeds a valid id forward; the chain reader rejects it.
        const auto token=failed?0U:best.token;
        slots[step]=token;reply->tokens[step]=token;reply->probabilities[step]=failed?0.0F:1.0F/sum;
        if constexpr(Second) reply->runner_up[step]=failed?0.0F:expf(best.second-best.value)/sum;
        if(failed) reply->status|=failed<<(2U*step);
        // Host pollers read the fields above once they observe this count (wait_step).
        __threadfence_system();
        __hip_atomic_store(&reply->completed,step+1U,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
    }
}
// One launch per depth for the argmax (R2-10): every block writes its partial and takes a ticket on `done`;
// the block holding the last ticket folds the partials exactly as the former separate finish kernel did
// (same array, same lane/stride order, same tree), so ids and probabilities are unchanged. It then rearms
// `done` for the next depth (stream order separates launches).
template<bool Second> __device__ __forceinline__ void draft_argmax_body(const float* logits,const std::uint32_t* ids,
    std::uint32_t rows,Partial* partials,std::uint32_t* done,std::uint32_t step,std::uint32_t* slots,Reply* reply) {
    __shared__ Partial values[threads];
    __shared__ float sums[threads];
    __shared__ bool last;
    const auto lane=static_cast<unsigned>(threadIdx.x);
    const auto index=static_cast<std::uint64_t>(blockIdx.x)*threads+lane;
    const std::uint32_t count=(rows-1U)/threads+1U;
    Partial value{0.0F,0,0,0,0.0F,-INFINITY};
    if(index<rows) {
        const auto loaded=strix::argmax::load(logits[index],ids?ids[index]:static_cast<std::uint32_t>(index));
        value.value=loaded.value;value.token=loaded.token;value.bad=loaded.bad;value.valid=loaded.valid;
    }
    auto best=strix::argmax::block_reduce<threads>(values,lane,value,[](const Partial& a,const Partial& b){return combine<Second>(a,b);});
    best.sum=block_sum(sums,value.valid && best.valid?expf(value.value-best.value):0.0F);
    if(!lane) {
        partials[blockIdx.x]=best;
        last=__hip_atomic_fetch_add(done,1U,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_AGENT)==count-1U;
    }
    __syncthreads();
    if(!last) return;
    // Acquire for every lane: the other blocks' partials are visible past this CU's caches.
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
    draft_argmax_finish_body<Second>(partials,count,step,slots,reply);
    if(!lane) __hip_atomic_store(done,0U,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
}
template<bool Second> __global__ __launch_bounds__(threads) void draft_argmax(const float* logits,const std::uint32_t* ids,
    std::uint32_t rows,Partial* partials,std::uint32_t* done,std::uint32_t step,std::uint32_t* slots,Reply* reply) {
    draft_argmax_body<Second>(logits,ids,rows,partials,done,step,slots,reply);
}
template<bool Second> __global__ __launch_bounds__(threads) void draft_argmax_rows(DraftHeadRows h,std::uint32_t step) {
    const auto t=blockIdx.y;
    if(blockIdx.x>=(h.rows[t]-1U)/threads+1U) return;
    draft_argmax_body<Second>(h.logits[t],h.ids[t],h.rows[t],h.partials[t],h.done[t],step,h.slots[t],h.reply[t]);
}
} // namespace

struct HipMtpDraftHead::Impl {
    HipMtpDraftHeadConfig config;
    const std::uint8_t* head=nullptr;
    std::uint32_t row_bytes=0,capacity=0,rows=0,base_rows=0,context_rows=0,context_dropped=0;
    std::vector<std::uint8_t> seen;
    std::uint8_t* compact=nullptr;
    std::uint32_t* ids=nullptr;
    std::uint32_t* host_ids=nullptr;
    float* logits=nullptr;
    Partial* partials=nullptr;
    std::uint32_t* done=nullptr; // argmax ticket: the last block to finish folds the partials
    std::uint32_t* slots=nullptr;
    Reply* reply=nullptr;
    Reply* device_reply=nullptr;
    float* runner_logits=nullptr; // pinned [max_slots][capacity] when the runner-up diagnostic is on
    std::uint32_t runner_rows[max_slots]{};
    hipStream_t stream=nullptr;
    ~Impl() {
        if(stream) (void)hipStreamSynchronize(stream);
        for(void* p:{static_cast<void*>(compact),static_cast<void*>(ids),static_cast<void*>(logits),
                     static_cast<void*>(partials),static_cast<void*>(done),static_cast<void*>(slots)}) if(p) (void)hipFree(p);
        if(host_ids) (void)hipHostFree(host_ids);
        if(reply) (void)hipHostFree(reply);
        if(runner_logits) (void)hipHostFree(runner_logits);
    }
    template<class T> T* device(std::uint64_t count) {
        void* p=nullptr;hip_check(hipMalloc(&p,static_cast<std::size_t>(count*sizeof(T))),"MTP draft head allocation");return static_cast<T*>(p);
    }
    // Appends unseen ids to the append-only pinned list, then gathers the new rows.
    std::uint32_t admit(std::span<const std::uint32_t> tokens,std::uint32_t limit) {
        const auto first=rows;std::uint32_t added=0;
        for(const auto token:tokens) {
            if(token>=config.vocab_size || seen[token]) continue;
            if(added>=limit || rows>=capacity) {++context_dropped;continue;}
            seen[token]=1U;host_ids[rows++]=token;++added;
        }
        if(rows==first) return 0;
        hip_check(hipMemcpyAsync(ids+first,host_ids+first,(rows-first)*sizeof(std::uint32_t),hipMemcpyHostToDevice,stream),"MTP draft ids upload");
        if(config.q4) hipLaunchKernelGGL(draft_gather_rows_q4,dim3(rows-first),dim3(threads),0,stream,head,ids,first,rows-first,
            config.hidden_size/32U,compact);
        else hipLaunchKernelGGL(draft_gather_rows,dim3(rows-first),dim3(threads),0,stream,head,ids,first,rows-first,row_bytes,compact);
        hip_check(hipGetLastError(),"MTP draft row gather");
        return added;
    }
};
HipMtpDraftHead::HipMtpDraftHead(const void* q8_head,HipMtpDraftHeadConfig config,void* stream):impl_(std::make_unique<Impl>()) {
    auto& p=*impl_;
    require(q8_head && stream && config.hidden_size && config.hidden_size%32U==0U && config.hidden_size<=10240U &&
        config.vocab_size && config.max_depth && config.max_depth<=max_slots,"MTP draft head geometry is invalid");
    p.head=static_cast<const std::uint8_t*>(q8_head);p.stream=static_cast<hipStream_t>(stream);
    p.row_bytes=config.hidden_size/32U*34U;
    if(config.full) {
        require(config.base_tokens.empty() && !config.context_capacity && !config.q4,"MTP full draft head takes no vocabulary subset or Q4 copy");
        p.capacity=p.rows=config.vocab_size;
    } else {
        const auto capacity=static_cast<std::uint64_t>(config.base_tokens.size())+config.context_capacity;
        require(capacity && capacity<=config.vocab_size,"MTP draft vocabulary subset is empty or exceeds the vocabulary");
        p.capacity=static_cast<std::uint32_t>(capacity);
    }
    p.config=std::move(config);
    try {
        p.logits=p.device<float>(p.capacity);
        p.partials=p.device<Partial>((p.capacity-1U)/threads+1U);
        p.done=p.device<std::uint32_t>(1);
        hip_check(hipMemsetAsync(p.done,0,sizeof(std::uint32_t),p.stream),"MTP draft argmax ticket init");
        p.slots=p.device<std::uint32_t>(max_slots);
        hip_check(hipMemsetAsync(p.slots,0,max_slots*sizeof(std::uint32_t),p.stream),"MTP draft slot init");
        hip_check(hipHostMalloc(reinterpret_cast<void**>(&p.reply),sizeof(Reply),hipHostMallocMapped|hipHostMallocCoherent),"MTP draft mapped reply");
        hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&p.device_reply),p.reply,0),"MTP draft mapped reply view");
        *p.reply=Reply{};
        if(!p.config.full) {
            p.seen.assign(p.config.vocab_size,0U);
            p.compact=p.device<std::uint8_t>(static_cast<std::uint64_t>(p.capacity)*(p.config.q4?p.config.hidden_size/32U*18U:p.row_bytes));
            p.ids=p.device<std::uint32_t>(p.capacity);
            hip_check(hipHostMalloc(reinterpret_cast<void**>(&p.host_ids),p.capacity*sizeof(std::uint32_t),hipHostMallocDefault),"MTP draft pinned ids");
            p.base_rows=p.admit(p.config.base_tokens,static_cast<std::uint32_t>(p.config.base_tokens.size()));
            p.context_dropped=0;
            require(p.base_rows || p.config.context_capacity,"MTP draft base vocabulary has no valid id");
        }
        if(const char* v=std::getenv("STRIX_MTP_DRAFT_TOP2");v && v[0]!='0')
            hip_check(hipHostMalloc(reinterpret_cast<void**>(&p.runner_logits),static_cast<std::size_t>(max_slots)*p.capacity*sizeof(float),
                hipHostMallocDefault),"MTP draft runner-up logits");
        hip_check(hipStreamSynchronize(p.stream),"MTP draft head setup");
    } catch(...) {impl_.reset();throw;}
}
HipMtpDraftHead::~HipMtpDraftHead()=default;
void HipMtpDraftHead::observe(std::span<const std::uint32_t> tokens) {
    auto& p=*impl_;if(p.config.full) return;
    p.context_rows+=p.admit(tokens,p.config.context_capacity-p.context_rows);
}
void HipMtpDraftHead::begin_chain() {*impl_->reply=Reply{};}
void HipMtpDraftHead::launch(const std::uint16_t* hidden,std::uint32_t step,void* stream) {
    auto& p=*impl_;
    require(step<p.config.max_depth && stream==p.stream && p.rows,"MTP draft head step/stream is invalid");
    if(p.config.q4) {
        hipLaunchKernelGGL(draft_q4_head,dim3((p.rows+7U)/8U),dim3(256),0,p.stream,hidden,p.compact,p.config.hidden_size/32U,p.rows,p.logits);
        hip_check(hipGetLastError(),"MTP draft Q4 head");
    } else launch_hip_q8_head_rows(hidden,p.config.full?static_cast<const void*>(p.head):p.compact,p.config.hidden_size,p.rows,p.logits,stream);
    if(p.runner_logits) {
        p.runner_rows[step]=p.rows;
        hip_check(hipMemcpyAsync(p.runner_logits+static_cast<std::uint64_t>(step)*p.capacity,p.logits,p.rows*sizeof(float),
            hipMemcpyDeviceToHost,p.stream),"MTP draft runner-up logits copy");
    }
    const auto blocks=(p.rows-1U)/threads+1U;
    // One launch: every block writes its partial, the last block to take the ticket folds them.
    if(p.config.runner_up)
        hipLaunchKernelGGL(draft_argmax<true>,dim3(blocks),dim3(threads),0,p.stream,p.logits,p.ids,p.rows,p.partials,p.done,step,p.slots,p.device_reply);
    else
        hipLaunchKernelGGL(draft_argmax<false>,dim3(blocks),dim3(threads),0,p.stream,p.logits,p.ids,p.rows,p.partials,p.done,step,p.slots,p.device_reply);
    hip_check(hipGetLastError(),"MTP draft argmax");
}
std::uint32_t HipMtpDraftHead::shared_rows(std::span<HipMtpDraftHead* const> heads) {
    require(!heads.empty() && heads.size()<=max_batch,"MTP draft head batch size is invalid");
    const auto& a=*heads.front()->impl_;
    std::uint32_t shared=a.rows;
    for(auto* head:heads) {
        const auto& b=*head->impl_;
        if(b.config.full || !b.config.q4 || b.head!=a.head || b.config.hidden_size!=a.config.hidden_size ||
           b.config.runner_up!=a.config.runner_up || b.runner_logits || !b.rows) return UINT32_MAX;
        std::uint32_t n=0;const auto limit=std::min(shared,b.rows);
        while(n<limit && a.host_ids[n]==b.host_ids[n]) ++n;
        shared=n;
    }
    return shared;
}
void HipMtpDraftHead::launch_rows(std::span<HipMtpDraftHead* const> heads,std::span<const std::uint16_t* const> hidden,
    std::uint32_t step,std::uint32_t shared,void* stream) {
    require(!heads.empty() && heads.size()<=max_batch && hidden.size()==heads.size() && stream,"MTP draft head batch is invalid");
    const auto& a=*heads.front()->impl_;
    DraftHeadRows h{};std::uint32_t tail_blocks=0,partial_blocks=0;
    for(std::size_t t=0;t<heads.size();++t) {
        auto& p=*heads[t]->impl_;
        require(step<p.config.max_depth && p.rows && p.config.q4 && !p.config.full && !p.runner_logits && p.head==a.head &&
            p.config.hidden_size==a.config.hidden_size && p.config.runner_up==a.config.runner_up && shared<=p.rows && hidden[t],
            "MTP draft head batch member is incompatible");
        h.hidden[t]=hidden[t];h.compact[t]=p.compact;h.logits[t]=p.logits;h.rows[t]=p.rows;h.ids[t]=p.ids;
        h.partials[t]=p.partials;h.done[t]=p.done;h.slots[t]=p.slots;h.reply[t]=p.device_reply;
        tail_blocks=std::max(tail_blocks,(p.rows-shared+7U)/8U);partial_blocks=std::max(partial_blocks,(p.rows-1U)/threads+1U);
    }
    const auto s=static_cast<hipStream_t>(stream);const auto n=static_cast<unsigned>(heads.size());
    const auto blocks=a.config.hidden_size/32U;
    if(shared) {
        const dim3 grid((shared+7U)/8U);
        switch(n) {
        case 1:hipLaunchKernelGGL(HIP_KERNEL_NAME(draft_q4_head_shared<1U>),grid,dim3(256),0,s,h,blocks,shared);break;
        case 2:hipLaunchKernelGGL(HIP_KERNEL_NAME(draft_q4_head_shared<2U>),grid,dim3(256),0,s,h,blocks,shared);break;
        case 3:hipLaunchKernelGGL(HIP_KERNEL_NAME(draft_q4_head_shared<3U>),grid,dim3(256),0,s,h,blocks,shared);break;
        case 4:hipLaunchKernelGGL(HIP_KERNEL_NAME(draft_q4_head_shared<4U>),grid,dim3(256),0,s,h,blocks,shared);break;
        case 5:hipLaunchKernelGGL(HIP_KERNEL_NAME(draft_q4_head_shared<5U>),grid,dim3(256),0,s,h,blocks,shared);break;
        case 6:hipLaunchKernelGGL(HIP_KERNEL_NAME(draft_q4_head_shared<6U>),grid,dim3(256),0,s,h,blocks,shared);break;
        case 7:hipLaunchKernelGGL(HIP_KERNEL_NAME(draft_q4_head_shared<7U>),grid,dim3(256),0,s,h,blocks,shared);break;
        default:hipLaunchKernelGGL(HIP_KERNEL_NAME(draft_q4_head_shared<8U>),grid,dim3(256),0,s,h,blocks,shared);break;
        }
        hip_check(hipGetLastError(),"MTP draft Q4 head batch (shared rows)");
    }
    if(tail_blocks) {
        hipLaunchKernelGGL(draft_q4_head_tails,dim3(tail_blocks,n),dim3(256),0,s,h,blocks,shared);
        hip_check(hipGetLastError(),"MTP draft Q4 head batch (own rows)");
    }
    if(a.config.runner_up)
        hipLaunchKernelGGL(draft_argmax_rows<true>,dim3(partial_blocks,n),dim3(threads),0,s,h,step);
    else
        hipLaunchKernelGGL(draft_argmax_rows<false>,dim3(partial_blocks,n),dim3(threads),0,s,h,step);
    hip_check(hipGetLastError(),"MTP draft argmax batch");
}
const std::uint32_t* HipMtpDraftHead::device_token(std::uint32_t step) const {
    require(step<impl_->config.max_depth,"MTP draft token slot is invalid");return impl_->slots+step;
}
std::vector<std::uint32_t> HipMtpDraftHead::read(std::uint32_t count) const {
    const auto& p=*impl_;require(count<=p.config.max_depth,"MTP draft read exceeds depth");
    const auto status=__atomic_load_n(&p.reply->status,__ATOMIC_ACQUIRE);
    require(!(status&((count>=16U?0U:(1U<<(2U*count)))-1U)),"MTP draft head selected a non-finite row");
    std::vector<std::uint32_t> tokens(p.reply->tokens,p.reply->tokens+count);
    for(const auto token:tokens) require(token<p.config.vocab_size,"MTP draft head produced out-of-vocabulary token");
    return tokens;
}
std::uint32_t HipMtpDraftHead::step_token(std::uint32_t step) const {
    const auto& p=*impl_;require(step<p.config.max_depth,"MTP draft token step is invalid");
    const volatile std::uint32_t* mapped=p.reply->tokens;return mapped[step];
}
float HipMtpDraftHead::probability(std::uint32_t step) const {
    const auto& p=*impl_;require(step<p.config.max_depth,"MTP draft probability step is invalid");
    const volatile float* mapped=p.reply->probabilities;return mapped[step];
}
void HipMtpDraftHead::wait_step(std::uint32_t step,void* raw_stream) const {
    const auto& p=*impl_;require(step<p.config.max_depth && raw_stream,"MTP draft wait step/stream is invalid");
    // The stream that carries the step: the head's own, or the multi-session draft batch stream.
    const auto stream=static_cast<hipStream_t>(raw_stream);
    // STRIX_MTP_DRAFT_POLL=0 synchronises the stream as before.
    static const bool poll=strix::envcfg::flag("STRIX_MTP_DRAFT_POLL", true);
    if(poll) {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(50);
        for(unsigned spin=0;;++spin) {
            if(__atomic_load_n(&p.reply->completed,__ATOMIC_ACQUIRE)>step) return;
            if((spin&1023U)==1023U) {
                const auto state=hipStreamQuery(stream);
                if(state!=hipErrorNotReady) break; // idle or failed: the synchronisation returns or reports it
                if(std::chrono::steady_clock::now()>deadline) break;
            }
        }
    }
    hip_check(hipStreamSynchronize(stream),"MTP draft early-stop step");
}
void HipMtpDraftHead::read_logits(std::vector<float>& out) const {
    const auto& p=*impl_;out.resize(p.rows);
    hip_check(hipMemcpyAsync(out.data(),p.logits,p.rows*sizeof(float),hipMemcpyDeviceToHost,p.stream),"MTP draft logits readback");
    hip_check(hipStreamSynchronize(p.stream),"MTP draft logits readback completion");
}
std::span<const std::uint32_t> HipMtpDraftHead::host_ids() const noexcept {
    const auto& p=*impl_;return p.config.full?std::span<const std::uint32_t>{}:std::span<const std::uint32_t>(p.host_ids,p.rows);
}
void HipMtpDraftHead::set_token(std::uint32_t step,std::uint32_t token,void* stream) {
    auto& p=*impl_;require(step<p.config.max_depth && stream==p.stream && token<p.config.vocab_size,"MTP draft token override is invalid");
    hip_check(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(p.slots+step),static_cast<int>(token),1,p.stream),"MTP draft token override");
    p.reply->tokens[step]=token;
}
std::uint32_t* HipMtpDraftHead::device_token_slot(std::uint32_t step) {
    require(step<impl_->config.max_depth,"MTP draft token slot is invalid");return impl_->slots+step;
}
std::uint32_t HipMtpDraftHead::capacity() const noexcept {return impl_->capacity;}
void HipMtpDraftHead::record_token(std::uint32_t step,std::uint32_t token) {
    auto& p=*impl_;require(step<p.config.max_depth && token<p.config.vocab_size,"MTP draft token record is invalid");
    p.reply->tokens[step]=token;
}
bool HipMtpDraftHead::records_runner_up() const noexcept {return impl_->runner_logits!=nullptr;}
bool HipMtpDraftHead::runner_up(std::uint32_t step,std::uint32_t& token,float& probability) const {
    const auto& p=*impl_;
    if(!p.runner_logits || step>=p.config.max_depth || p.runner_rows[step]<2U) return false;
    const auto* l=p.runner_logits+static_cast<std::uint64_t>(step)*p.capacity;const auto rows=p.runner_rows[step];
    auto id=[&](std::uint32_t r){return p.config.full?r:p.host_ids[r];};
    auto better=[&](std::uint32_t a,std::uint32_t b){return l[a]>l[b] || (l[a]==l[b] && id(a)<id(b));};
    std::uint32_t first=UINT32_MAX,second=UINT32_MAX;
    for(std::uint32_t r=0;r<rows;++r) {
        if(!std::isfinite(l[r])) continue;
        if(first==UINT32_MAX || better(r,first)) {second=first;first=r;}
        else if(second==UINT32_MAX || better(r,second)) second=r;
    }
    if(second==UINT32_MAX) return false;
    double sum=0.0;for(std::uint32_t r=0;r<rows;++r) if(std::isfinite(l[r])) sum+=std::exp(static_cast<double>(l[r])-l[first]);
    token=id(second);probability=static_cast<float>(std::exp(static_cast<double>(l[second])-l[first])/sum);return true;
}
float HipMtpDraftHead::runner_up_probability(std::uint32_t step) const {
    const auto& p=*impl_;require(step<p.config.max_depth,"MTP draft runner-up step is invalid");
    if(!p.config.runner_up) return 0.0F;
    const volatile float* mapped=p.reply->runner_up;return mapped[step];
}
bool HipMtpDraftHead::full() const noexcept {return impl_->config.full;}
HipMtpDraftHeadStats HipMtpDraftHead::stats() const noexcept {
    const auto& p=*impl_;return {p.rows,p.base_rows,p.context_rows,p.context_dropped};
}
const float* HipMtpDraftHead::logits() const noexcept {return impl_->logits;}
const std::uint32_t* HipMtpDraftHead::ids() const noexcept {return impl_->ids;}
} // namespace strix
