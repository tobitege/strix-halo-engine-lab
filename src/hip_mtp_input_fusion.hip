#include "strix/hip_mtp_prime_rows.hpp"
#include <type_traits>
#include "strix/hip_check.hpp"
#include "strix/hip_qwen_dpp_reduce.hpp"
#include "strix/hip_mtp_fast_draft.hpp"
#include "strix/hip_mtp_input_fusion.hpp"
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

namespace strix {
namespace {
static_assert(sizeof(HipMtpInputFusionRequest)+sizeof(HipMtpInputFusionPrimeRows)+32U<4096U);
std::uint64_t multiply(std::uint64_t a,std::uint64_t b) {
    if(b && a>UINT64_MAX/b) throw std::overflow_error("MTP input fusion byte overflow");return a*b;
}
unsigned element_bytes(HipMtpInputStorage s) {
    if(s==HipMtpInputStorage::bf16)return 2U;
    if(s==HipMtpInputStorage::fp32)return 4U;
    throw std::invalid_argument("MTP input fusion storage must be explicit BF16 or FP32");
}
void geometry(unsigned h,unsigned hc) {
    if(h<32U || h>4096U || h%32U || (hc!=2U && hc!=4U))
        throw std::invalid_argument("MTP input fusion geometry is unsupported");
}
void view(HipMtpInputFusionView v,std::uint64_t bytes,unsigned align) {
    if(!v.address || !v.allocation_generation || v.bytes!=bytes || v.address%align ||
       v.bytes>std::numeric_limits<std::uintptr_t>::max()-v.address)
        throw std::invalid_argument("MTP input fusion view extent, alignment or generation is invalid");
}
bool overlaps(HipMtpInputFusionView a,HipMtpInputFusionView b) {
    return a.address<b.address+b.bytes && b.address<a.address+a.bytes;
}
using strix::hip_check;
__device__ __forceinline__ float load_value(std::uintptr_t p,unsigned i,HipMtpInputStorage storage) {
    return storage==HipMtpInputStorage::bf16?
        __uint_as_float(static_cast<unsigned>(reinterpret_cast<const std::uint16_t*>(p)[i])<<16U):
        reinterpret_cast<const float*>(p)[i];
}
__device__ __forceinline__ std::uint16_t to_bf16(float x) {
    const unsigned bits=__float_as_uint(x);
    if((bits&0x7f800000U)==0x7f800000U && (bits&0x007fffffU))
        return static_cast<std::uint16_t>((bits>>16U)|0x0040U);
    return static_cast<std::uint16_t>((bits+0x7fffU+((bits>>16U)&1U))>>16U);
}
__device__ __forceinline__ float rounded(float v,HipMtpInputRounding rounding) {
    return rounding==HipMtpInputRounding::bf16_rne?__uint_as_float(static_cast<unsigned>(to_bf16(v))<<16U):v;
}
// Threads >= 256 (when launched wider) only join the element-wise output pass: the sum and
// its 256-lane tree are the first 256 threads' exactly as in a 256-thread launch.
template<unsigned Threads=256U>
__device__ __forceinline__ void mtp_input_norms_body(const HipMtpInputFusionRequest& r) {
    __shared__ float sums[256];
    __shared__ float shared_inverse;
    const unsigned lane=threadIdx.x;
    const bool hidden=blockIdx.x!=0U;
    const unsigned n=hidden?r.hidden_size*r.hc_count:r.hidden_size;
    const auto input=hidden?r.wide_hidden.address:r.embedding.address;
    const auto storage=hidden?r.hidden_storage:r.embedding_storage;
    const auto* weights=reinterpret_cast<const float*>(hidden?r.hidden_norm.address:r.embedding_norm.address);
    auto* out=reinterpret_cast<float*>(r.normalized_scratch.address)+(hidden?r.hidden_size:0U);
    float sum=0.0F;
    if(lane<256U) for(unsigned i=lane;i<n;i+=256U) {const float v=load_value(input,i,storage);sum=fmaf(v,v,sum);}
    if(lane<256U) sums[lane]=sum;
    __syncthreads();
    for(unsigned stride=128U;stride;stride>>=1U) {if(lane<stride)sums[lane]=sums[lane]+sums[lane+stride];__syncthreads();}
    if(lane==0U) shared_inverse=1.0F/sqrtf(sums[0]/static_cast<float>(n)+r.epsilon);
    __syncthreads();
    const float inverse=shared_inverse;
    for(unsigned i=lane;i<n;i+=Threads) {
        const float gain=r.norm_weight_encoding==MtpNormWeightEncoding::gemma_offset?1.0F+weights[i]:weights[i];
        const float normalized=load_value(input,i,storage)*inverse;
        out[i]=rounded(normalized*gain,r.normalized_rounding);
    }
}
template<unsigned Threads=256U>
__global__ __launch_bounds__(Threads) void mtp_input_norms(HipMtpInputFusionRequest r) {mtp_input_norms_body<Threads>(r);}
// Draft batch: token blockIdx.y runs the one-token norms on its own views (same body, same bits).
template<unsigned Threads=256U,class Rows=HipMtpInputFusionRows>
__global__ __launch_bounds__(Threads) void mtp_input_norms_rows(HipMtpInputFusionRequest r,Rows rows) {
    auto q=r;
    q.embedding.address=rows.embedding[blockIdx.y];q.wide_hidden.address=rows.wide_hidden[blockIdx.y];
    q.normalized_scratch.address=rows.normalized[blockIdx.y];
    mtp_input_norms_body<Threads>(q);
}
__device__ float coefficient(const unsigned char* row,unsigned column) {
    const auto* block=row+(column/32U)*34U;
    const float scale=__half2float(*reinterpret_cast<const __half*>(block));
    const auto code=static_cast<signed char>(block[2U+column%32U]);
    return scale*static_cast<float>(code);
}
__global__ __launch_bounds__(128) void mtp_input_projections(HipMtpInputFusionRequest r) {
    const unsigned output_row=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned h=r.hidden_size;
    if(output_row>=h)return;
    const auto* input=reinterpret_cast<const float*>(r.normalized_scratch.address);
    const auto* row=reinterpret_cast<const unsigned char*>(r.packed_projection.address)+output_row*r.projection_row_stride_bytes;
    float embedding=0.0F;
    for(unsigned col=0;col<h;++col)embedding=fmaf(coefficient(row,col),input[col],embedding);
    auto* output=reinterpret_cast<float*>(r.projection_scratch.address);
    output[output_row]=rounded(embedding,r.projection_rounding);
    // Reuse each hidden-half coefficient across the HC chains. Each chain still
    // evaluates coefficient-first FMA in ascending column order, rounding once.
    float sums[4]={0.0F,0.0F,0.0F,0.0F};
    for(unsigned col=0;col<h;++col){
        const float weight=coefficient(row,h+col);
        for(unsigned stream=0;stream<r.hc_count;++stream)
            sums[stream]=fmaf(weight,input[h+stream*h+col],sums[stream]);
    }
    for(unsigned stream=0;stream<r.hc_count;++stream)
        output[h+stream*h+output_row]=rounded(sums[stream],r.projection_rounding);
}
// Same chains as mtp_input_projections, one thread per chain: role 0 is the
// embedding chain, role 1 + s the HC stream s chain, each coefficient-first
// ascending FMA. 32 rows per block stage contiguous 128-column runs of both
// row halves (4 Q8 blocks = 136 bytes each) with coalesced loads, instead of
// every lane walking its own 5 KB row.
constexpr unsigned kStagedRows=32U,kStagedColumns=128U,kStagedRun=kStagedColumns/32U*34U;
__global__ __launch_bounds__(160) void mtp_input_projections_staged(HipMtpInputFusionRequest r) {
    __shared__ __align__(8) unsigned char stage[kStagedRows][2][kStagedRun];
    const unsigned h=r.hidden_size,first=blockIdx.x*kStagedRows;
    const unsigned local=threadIdx.x%kStagedRows,role=threadIdx.x/kStagedRows;
    const auto* input=reinterpret_cast<const float*>(r.normalized_scratch.address);
    const auto* packed=reinterpret_cast<const unsigned char*>(r.packed_projection.address);
    float sum=0.0F;
    constexpr unsigned words=kStagedRun/8U,pieces=kStagedRows*2U*words;
    for(unsigned column=0;column<h;column+=kStagedColumns) {
        for(unsigned piece=threadIdx.x;piece<pieces;piece+=blockDim.x) {
            const unsigned row=piece/(2U*words),half=(piece/words)%2U,word=piece%words;
            const auto offset=static_cast<std::uint64_t>(first+row)*r.projection_row_stride_bytes+
                static_cast<std::uint64_t>((half*h+column)/32U)*34U+word*8U;
            *reinterpret_cast<uint2*>(&stage[row][half][word*8U])=*reinterpret_cast<const uint2*>(packed+offset);
        }
        __syncthreads();
        if(role<=r.hc_count) {
            const auto* run=stage[local][role==0U?0:1];
            const auto* x=input+(role==0U?0U:h+(role-1U)*h)+column;
#pragma unroll
            for(unsigned block=0;block<4U;++block) {
                const auto* b=run+block*34U;
                const float scale=__half2float(*reinterpret_cast<const __half*>(b));
#pragma unroll 8
                for(unsigned k=0;k<32U;++k)
                    sum=fmaf(scale*static_cast<float>(static_cast<signed char>(b[2U+k])),x[block*32U+k],sum);
            }
        }
        __syncthreads();
    }
    if(role>r.hc_count)return;
    auto* output=reinterpret_cast<float*>(r.projection_scratch.address);
    output[(role==0U?0U:h+(role-1U)*h)+first+local]=rounded(sum,r.projection_rounding);
}
__global__ __launch_bounds__(128) void mtp_input_broadcast_add(HipMtpInputFusionRequest r) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=r.hidden_size*r.hc_count)return;
    const auto* projections=reinterpret_cast<const float*>(r.projection_scratch.address);
    const float v=projections[i%r.hidden_size]+projections[r.hidden_size+i];
    if(r.output_storage==HipMtpInputStorage::bf16)reinterpret_cast<std::uint16_t*>(r.output.address)[i]=to_bf16(v);
    else reinterpret_cast<float*>(r.output.address)[i]=v;
}
// Draft-only fused projection + broadcast add: one wave per output row reads the
// packed row once; lanes own 8-column groups, then a wave tree. Same rounding
// points as the exact split kernels, different summation order.
// Code j of a Q8_0 (34-byte) or Q4_0 (18-byte, low nibbles 0..15, high 16..31) block.
template<bool Q4> __device__ __forceinline__ float block_code(const unsigned char* block,unsigned j) {
    if constexpr(Q4) return static_cast<float>(static_cast<int>((block[2U+(j&15U)]>>(j>=16U?4U:0U))&15U)-8);
    else return static_cast<float>(static_cast<signed char>(block[2U+j]));
}
// R rows per wave (wave w of a block owns rows (block*8 + w)*R .. +R-1): each lane loads its
// eight normalised inputs of every role once per column group and runs each row's unchanged
// FMA sequence on them (per row: same column groups, same element order, same tree).
template<bool Q4,unsigned R>
__global__ __launch_bounds__(256) void mtp_input_projections_wave_rows(HipMtpInputFusionRequest r,const unsigned char* weights) {
    constexpr unsigned block=Q4?18U:34U,hc=4U;
    const unsigned lane=threadIdx.x%32U,first=(blockIdx.x*8U+threadIdx.x/32U)*R,h=r.hidden_size;
    if(first>=h)return;
    const auto* input=reinterpret_cast<const float*>(r.normalized_scratch.address);
    float embedding[R],sums[R][hc];
#pragma unroll
    for(unsigned k=0;k<R;++k){embedding[k]=0.0F;for(unsigned s=0;s<hc;++s)sums[k][s]=0.0F;}
    for(unsigned base=8U*lane;base<h;base+=256U) {
        float xe[8],xs[hc][8];
#pragma unroll
        for(unsigned i=0;i<8U;++i){xe[i]=input[base+i];for(unsigned s=0;s<hc;++s)xs[s][i]=input[h+s*h+base+i];}
#pragma unroll
        for(unsigned k=0;k<R;++k) {
            if(first+k>=h)break;
            const auto* packed=weights+static_cast<std::uint64_t>(first+k)*(2U*h/32U)*block;
            const auto* e=packed+(base/32U)*block;const auto* w=packed+((h+base)/32U)*block;
            const float es=__half2float(*reinterpret_cast<const __half*>(e)),ws=__half2float(*reinterpret_cast<const __half*>(w));
#pragma unroll
            for(unsigned i=0;i<8U;++i) {
                embedding[k]=fmaf(es*block_code<Q4>(e,base%32U+i),xe[i],embedding[k]);
                const float weight=ws*block_code<Q4>(w,base%32U+i);
#pragma unroll
                for(unsigned s=0;s<hc;++s) sums[k][s]=fmaf(weight,xs[s][i],sums[k][s]);
            }
        }
    }
#pragma unroll
    for(unsigned k=0;k<R;++k) {
        for(unsigned d=16U;d;d>>=1U) {
            embedding[k]+=strix::qwen_dpp::xor32(embedding[k], d);
            for(unsigned s=0;s<hc;++s) sums[k][s]+=strix::qwen_dpp::xor32(sums[k][s], d);
        }
    }
    if(lane)return;
#pragma unroll
    for(unsigned k=0;k<R;++k) {
        const unsigned row=first+k;
        if(row>=h)break;
        const float projected=rounded(embedding[k],r.projection_rounding);
        for(unsigned s=0;s<hc;++s) {
            const float v=projected+rounded(sums[k][s],r.projection_rounding);
            if(r.output_storage==HipMtpInputStorage::bf16)reinterpret_cast<std::uint16_t*>(r.output.address)[s*h+row]=to_bf16(v);
            else reinterpret_cast<float*>(r.output.address)[s*h+row]=v;
        }
    }
}
// Draft batch: mtp_input_projections_wave_rows with one output row per wave and Count tokens. The wave
// decodes its row's codes once per column group; every token keeps the one-token kernel's chains (same
// column groups, the same rounded es*code and ws*code products, element and stream order, same tree and
// rounding/output casts). Rows per wave only regroup independent rows, so R=1 here matches any R there.
// Tokens come in groups of Count per block (blockIdx.y = group, `total` tokens): 40 accumulators per token, so a
// group of 8 spilled (t3 trace: 1.9 ms at 8 tokens against 42 us for one); groups only regroup independent tokens.
template<bool Q4,unsigned Count,class Rows=HipMtpInputFusionRows>
__global__ __launch_bounds__(256) void mtp_input_projections_batch(HipMtpInputFusionRequest r,Rows all,
    const unsigned char* weights,unsigned total) {
    constexpr unsigned block=Q4?18U:34U,hc=4U;
    const unsigned lane=threadIdx.x%32U,row=blockIdx.x*8U+threadIdx.x/32U,h=r.hidden_size;
    if(row>=h)return;
    const unsigned first=blockIdx.y*Count,valid=total-first<Count?total-first:Count;
    HipMtpInputFusionRows rows{};
#pragma unroll
    for(unsigned t=0;t<Count;++t){const auto u=first+(t<valid?t:0U);rows.normalized[t]=all.normalized[u];rows.output[t]=all.output[u];}
    const auto* packed=weights+static_cast<std::uint64_t>(row)*(2U*h/32U)*block;
    float embedding[Count],sums[Count][hc];
#pragma unroll
    for(unsigned t=0;t<Count;++t){embedding[t]=0.0F;for(unsigned s=0;s<hc;++s)sums[t][s]=0.0F;}
    for(unsigned base=8U*lane;base<h;base+=256U) {
        const auto* e=packed+(base/32U)*block;const auto* w=packed+((h+base)/32U)*block;
        const float es=__half2float(*reinterpret_cast<const __half*>(e)),ws=__half2float(*reinterpret_cast<const __half*>(w));
        float ec[8],wc[8];
#pragma unroll
        for(unsigned i=0;i<8U;++i){ec[i]=es*block_code<Q4>(e,base%32U+i);wc[i]=ws*block_code<Q4>(w,base%32U+i);}
#pragma unroll
        for(unsigned t=0;t<Count;++t) {
            const auto* input=reinterpret_cast<const float*>(rows.normalized[t]);
#pragma unroll
            for(unsigned i=0;i<8U;++i) {
                embedding[t]=fmaf(ec[i],input[base+i],embedding[t]);
#pragma unroll
                for(unsigned s=0;s<hc;++s) sums[t][s]=fmaf(wc[i],input[h+s*h+base+i],sums[t][s]);
            }
        }
    }
#pragma unroll
    for(unsigned t=0;t<Count;++t) {
        for(unsigned d=16U;d;d>>=1U) {
            embedding[t]+=strix::qwen_dpp::xor32(embedding[t], d);
            for(unsigned s=0;s<hc;++s) sums[t][s]+=strix::qwen_dpp::xor32(sums[t][s], d);
        }
    }
    if(lane)return;
#pragma unroll
    for(unsigned t=0;t<Count;++t) {
        if(t>=valid) break;
        const float projected=rounded(embedding[t],r.projection_rounding);
        for(unsigned s=0;s<hc;++s) {
            const float v=projected+rounded(sums[t][s],r.projection_rounding);
            if(r.output_storage==HipMtpInputStorage::bf16)reinterpret_cast<std::uint16_t*>(rows.output[t])[s*h+row]=to_bf16(v);
            else reinterpret_cast<float*>(rows.output[t])[s*h+row]=v;
        }
    }
}
template<bool Q4>
__global__ __launch_bounds__(256) void mtp_input_projections_wave(HipMtpInputFusionRequest r,const unsigned char* weights) {
    constexpr unsigned block=Q4?18U:34U;
    const unsigned lane=threadIdx.x%32U,row=blockIdx.x*8U+threadIdx.x/32U,h=r.hidden_size,hc=r.hc_count;
    if(row>=h)return;
    const auto* input=reinterpret_cast<const float*>(r.normalized_scratch.address);
    const auto* packed=weights+static_cast<std::uint64_t>(row)*(2U*h/32U)*block;
    float embedding=0.0F,sums[4]={0.0F,0.0F,0.0F,0.0F};
    for(unsigned base=8U*lane;base<h;base+=256U) {
        const auto* e=packed+(base/32U)*block;const auto* w=packed+((h+base)/32U)*block;
        const float es=__half2float(*reinterpret_cast<const __half*>(e)),ws=__half2float(*reinterpret_cast<const __half*>(w));
        for(unsigned i=0;i<8U;++i) {
            embedding=fmaf(es*block_code<Q4>(e,base%32U+i),input[base+i],embedding);
            const float weight=ws*block_code<Q4>(w,base%32U+i);
            for(unsigned stream=0;stream<hc;++stream) sums[stream]=fmaf(weight,input[h+stream*h+base+i],sums[stream]);
        }
    }
    for(unsigned d=16U;d;d>>=1U) {
        embedding+=strix::qwen_dpp::xor32(embedding, d);
        for(unsigned stream=0;stream<4U;++stream) sums[stream]+=strix::qwen_dpp::xor32(sums[stream], d);
    }
    if(lane)return;
    const float projected=rounded(embedding,r.projection_rounding);
    for(unsigned stream=0;stream<hc;++stream) {
        const float v=projected+rounded(sums[stream],r.projection_rounding);
        if(r.output_storage==HipMtpInputStorage::bf16)reinterpret_cast<std::uint16_t*>(r.output.address)[stream*h+row]=to_bf16(v);
        else reinterpret_cast<float*>(r.output.address)[stream*h+row]=v;
    }
}
} // namespace
std::uint64_t hip_mtp_input_fusion_scratch_bytes(std::uint32_t h,std::uint32_t hc) {
    geometry(h,hc);return multiply(multiply(h,static_cast<std::uint64_t>(hc)+1U),4U);
}
void preflight_hip_mtp_input_fusion(const HipMtpInputFusionRequest& r) {
    geometry(r.hidden_size,r.hc_count);
    if(!std::isfinite(r.epsilon) || r.epsilon<=0.0F || !r.norm_encoding_admitted ||
       (r.norm_weight_encoding!=MtpNormWeightEncoding::gemma_offset && r.norm_weight_encoding!=MtpNormWeightEncoding::effective_scale) ||
       r.arithmetic!=MtpInputFusionArithmetic::split_projections_then_add ||
       (r.normalized_rounding!=HipMtpInputRounding::fp32 && r.normalized_rounding!=HipMtpInputRounding::bf16_rne) ||
       (r.projection_rounding!=HipMtpInputRounding::fp32 && r.projection_rounding!=HipMtpInputRounding::bf16_rne))
        throw std::invalid_argument("MTP input fusion needs explicit admitted norm encoding and split/cast profile");
    const std::uint64_t h=r.hidden_size,wide=h*r.hc_count,row=(2U*h/32U)*34U;
    if(r.projection_row_stride_bytes!=row)throw std::invalid_argument("MTP packed Q8 projection full-row stride is invalid");
    view(r.embedding,h*element_bytes(r.embedding_storage),element_bytes(r.embedding_storage));
    view(r.wide_hidden,wide*element_bytes(r.hidden_storage),element_bytes(r.hidden_storage));
    view(r.embedding_norm,h*4U,4U);view(r.hidden_norm,wide*4U,4U);view(r.packed_projection,row*h,2U);
    const auto scratch=hip_mtp_input_fusion_scratch_bytes(r.hidden_size,r.hc_count);
    view(r.normalized_scratch,scratch,4U);view(r.projection_scratch,scratch,4U);
    view(r.output,wide*element_bytes(r.output_storage),element_bytes(r.output_storage));
    const std::array reads{r.embedding,r.wide_hidden,r.embedding_norm,r.hidden_norm,r.packed_projection};
    const std::array writes{r.normalized_scratch,r.projection_scratch,r.output};
    for(std::size_t i=0;i<writes.size();++i) {
        for(auto input:reads)if(overlaps(writes[i],input))throw std::invalid_argument("MTP input fusion writable view aliases an input");
        for(std::size_t j=0;j<i;++j)if(overlaps(writes[i],writes[j]))throw std::invalid_argument("MTP input fusion writable views overlap");
    }
}
std::uint64_t hip_mtp_input_fusion_profile_identity(const HipMtpInputFusionRequest& r) {
    preflight_hip_mtp_input_fusion(r);
    std::uint64_t hash=14695981039346656037ULL;
    const auto add=[&](std::uint64_t v) {for(unsigned s=0;s<64U;s+=8U){hash^=(v>>s)&255U;hash*=1099511628211ULL;}};
    // Version specifies coefficient-first ascending FMA, 256-lane strided RMS
    // tree, IEEE denormals/correctly rounded divide-sqrt, separate rounded
    // projections and quiet-upper BF16 RNE. Rev3 pins the compiler FP policy.
    add(0x4d54504655533301ULL);add(r.hidden_size);add(r.hc_count);add(std::bit_cast<std::uint32_t>(r.epsilon));
    add(static_cast<unsigned>(r.embedding_storage));add(static_cast<unsigned>(r.hidden_storage));add(static_cast<unsigned>(r.output_storage));
    add(static_cast<unsigned>(r.normalized_rounding));add(static_cast<unsigned>(r.projection_rounding));add(static_cast<unsigned>(r.norm_weight_encoding));
    add(static_cast<unsigned>(r.arithmetic));return hash;
}
void launch_hip_mtp_input_fusion(const HipMtpInputFusionRequest& r,void* raw_stream) {
    preflight_hip_mtp_input_fusion(r);
    if(!raw_stream)throw std::invalid_argument("MTP input fusion requires an explicit owned stream");
    const auto stream=static_cast<hipStream_t>(raw_stream);
    hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;hip_check(hipStreamIsCapturing(stream,&capture),"MTP input fusion capture query");
    if(capture!=hipStreamCaptureStatusNone)throw std::invalid_argument("MTP input fusion is eager only");
    hip_check(hipPeekAtLastError(),"MTP input fusion prior HIP error");
    hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_input_norms<256U>),dim3(2U),dim3(256U),0U,stream,r);hip_check(hipGetLastError(),"MTP input norms launch");
    const unsigned jobs=r.hidden_size;
    if(r.hidden_size%kStagedColumns==0U && r.projection_row_stride_bytes%8U==0U && r.packed_projection.address%8U==0U)
        hipLaunchKernelGGL(mtp_input_projections_staged,dim3(r.hidden_size/kStagedRows),dim3(kStagedRows*(1U+r.hc_count)),0U,stream,r);
    else hipLaunchKernelGGL(mtp_input_projections,dim3((jobs+127U)/128U),dim3(128U),0U,stream,r);
    hip_check(hipGetLastError(),"MTP split projection launch");
    const unsigned outputs=r.hidden_size*r.hc_count;
    hipLaunchKernelGGL(mtp_input_broadcast_add,dim3((outputs+127U)/128U),dim3(128U),0U,stream,r);hip_check(hipGetLastError(),"MTP broadcast add launch");
}
void launch_hip_mtp_input_fusion_draft(const HipMtpInputFusionRequest& r,void* raw_stream) {
    preflight_hip_mtp_input_fusion(r);
    if(!raw_stream)throw std::invalid_argument("MTP input fusion requires an explicit owned stream");
    const auto stream=static_cast<hipStream_t>(raw_stream);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_input_norms<1024U>),dim3(2U),dim3(1024U),0U,stream,r);hip_check(hipGetLastError(),"MTP input norms launch");
    const auto* q8=reinterpret_cast<const unsigned char*>(r.packed_projection.address);
    const auto* q4=static_cast<const unsigned char*>(hip_mtp_draft_q4(q8));
    // STRIX_MTP_PROJ_ROWS=1|2|4: output rows per wave (each row's arithmetic is unchanged).
    static const unsigned rows_per_wave=[]{const char* v=std::getenv("STRIX_MTP_PROJ_ROWS");
        const unsigned n=v?static_cast<unsigned>(std::atoi(v)):2U;return n==1U||n==4U?n:2U;}();
    const auto rows=[&]<unsigned R>(){
        const dim3 grid((r.hidden_size+8U*R-1U)/(8U*R));
        if(q4) hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_input_projections_wave_rows<true,R>),grid,dim3(256U),0U,stream,r,q4);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_input_projections_wave_rows<false,R>),grid,dim3(256U),0U,stream,r,q8);
    };
    if(r.hc_count==4U && rows_per_wave==4U) rows.template operator()<4U>();
    else if(r.hc_count==4U && rows_per_wave==2U) rows.template operator()<2U>();
    else if(q4) hipLaunchKernelGGL(mtp_input_projections_wave<true>,dim3((r.hidden_size+7U)/8U),dim3(256U),0U,stream,r,q4);
    else hipLaunchKernelGGL(mtp_input_projections_wave<false>,dim3((r.hidden_size+7U)/8U),dim3(256U),0U,stream,r,q8);
    hip_check(hipGetLastError(),"MTP draft fused projection launch");
}
namespace {
template<class Rows> void launch_mtp_input_fusion_rows(std::span<const HipMtpInputFusionRequest> requests,void* raw_stream) {
    constexpr auto capacity=std::is_same_v<Rows,HipMtpInputFusionRows>?8U:hip_mtp_prime_rows_max;
    if(requests.empty() || requests.size()>capacity || !raw_stream)throw std::invalid_argument("MTP input fusion batch size/stream is invalid");
    const auto& r=requests.front();
    Rows rows{};
    for(std::size_t t=0;t<requests.size();++t) {
        const auto& q=requests[t];
        preflight_hip_mtp_input_fusion(q);
        // Only per-session views may differ: the profile, weights and norms are the batch's.
        if(hip_mtp_input_fusion_profile_identity(q)!=hip_mtp_input_fusion_profile_identity(r) ||
           q.embedding_norm.address!=r.embedding_norm.address || q.hidden_norm.address!=r.hidden_norm.address ||
           q.packed_projection.address!=r.packed_projection.address || q.projection_row_stride_bytes!=r.projection_row_stride_bytes ||
           q.embedding.bytes!=r.embedding.bytes || q.wide_hidden.bytes!=r.wide_hidden.bytes ||
           q.normalized_scratch.bytes!=r.normalized_scratch.bytes || q.output.bytes!=r.output.bytes)
            throw std::invalid_argument("MTP input fusion batch requests differ beyond their per-session views");
        rows.wide_hidden[t]=q.wide_hidden.address;
        if constexpr (std::is_same_v<Rows,HipMtpInputFusionRows>) {
            rows.embedding[t]=q.embedding.address;rows.normalized[t]=q.normalized_scratch.address;rows.output[t]=q.output.address;
        } else {
            if(!t) {
                const auto stride=requests.size()>1U?requests[1].embedding.address-r.embedding.address:0U;
                rows.embedding={r.embedding.address,stride};rows.normalized={r.normalized_scratch.address,stride};
                rows.output={r.output.address,stride};
            }
            if(rows.embedding[t]!=q.embedding.address || rows.normalized[t]!=q.normalized_scratch.address || rows.output[t]!=q.output.address)
                throw std::invalid_argument("MTP input fusion prime scratch is not strided");
        }
    }
    const auto* q8=reinterpret_cast<const unsigned char*>(r.packed_projection.address);
    const auto* q4=static_cast<const unsigned char*>(hip_mtp_draft_q4(q8));
    // The batch reproduces the one-token draft launcher's production branch only (Q4 copy, HC 4).
    if(!q4 || r.hc_count!=4U)throw std::invalid_argument("MTP input fusion batch needs the Q4 copy and four HC streams");
    const auto stream=static_cast<hipStream_t>(raw_stream);
    const auto n=static_cast<unsigned>(requests.size());
    hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_input_norms_rows<1024U,Rows>),dim3(2U,n),dim3(1024U),0U,stream,r,rows);
    hip_check(hipGetLastError(),"MTP input norms batch launch");
    // STRIX_MTP_DRAFT_BATCH_PROJ_GROUP=1|2|4: tokens per block (default 2).
    static const unsigned group=[]{const char* v=std::getenv("STRIX_MTP_DRAFT_BATCH_PROJ_GROUP");
        const unsigned g=v?static_cast<unsigned>(std::atoi(v)):2U;return g==1U||g==4U?g:2U;}();
    const auto g=std::min(group,n);
    const dim3 grid((r.hidden_size+7U)/8U,(n+g-1U)/g);
    if(g==1U) hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_input_projections_batch<true,1U,Rows>),grid,dim3(256U),0U,stream,r,rows,q4,n);
    else if(g==2U) hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_input_projections_batch<true,2U,Rows>),grid,dim3(256U),0U,stream,r,rows,q4,n);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(mtp_input_projections_batch<true,4U,Rows>),grid,dim3(256U),0U,stream,r,rows,q4,n);
    hip_check(hipGetLastError(),"MTP draft fused projection batch launch");
}
}
void launch_hip_mtp_input_fusion_draft_rows(std::span<const HipMtpInputFusionRequest> requests,void* stream) {
    launch_mtp_input_fusion_rows<HipMtpInputFusionRows>(requests,stream);
}
void launch_hip_mtp_input_fusion_prime_rows(std::span<const HipMtpInputFusionRequest> requests,void* stream) {
    launch_mtp_input_fusion_rows<HipMtpInputFusionPrimeRows>(requests,stream);
}

} // namespace strix
