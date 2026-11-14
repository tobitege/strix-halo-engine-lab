#include "strix/hip_check.hpp"
#include "strix/hip_qwen_dpp_reduce.hpp"
#include "strix/hip_arch_caps.hpp"
#include "strix/hip_moe_q4k_a8_probe.hpp"
#include "strix/hip_moe_a8_down.hpp"
#include "strix/hip_moe_q4k_a8_known_rows.hpp"
#include "strix/hip_launch_context.hpp"
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "Q4K A8G32 requires nonfinite checks and strict floating-point semantics"
#endif
#include <hip/hip_fp16.h>
#include <cmath>
#include <stdexcept>
#include <string>
#pragma clang fp contract(off)
#include "strix/hip_gsq_a8_groups.hpp"
#if defined(__HIP_DEVICE_COMPILE__)
#if !STRIX_HAS_SUDOT4
#error "Experimental Q4K A8 probe requires native signed-A unsigned-B dot4"
#endif
#if !__has_builtin(__builtin_amdgcn_sudot4)
#error "No native signed-A unsigned-B dot4 support"
#endif
#endif
namespace strix::moe_q4k_a8 {
namespace {
// Explicit target option -fhip-fp32-correctly-rounded-divide-sqrt is mandatory.
// The available compiler lowers /; the declared OCML rte entry point did not link.
// Actual IR/ISA, denormal modes and independent device RN oracles remain required.
__device__ float divide_rne(float a, float b) { return a/b; }
__device__ int dot(int a, unsigned b, int acc) {
    return __builtin_amdgcn_sudot4(true,a,false,static_cast<int>(b),acc,false);
}
__device__ unsigned factor(uint4 header, unsigned group, bool minimum) {
    const unsigned shift = 8U*(group&3U);
    const unsigned a=(header.y>>shift)&255U,b=(header.z>>shift)&255U,c=(header.w>>shift)&255U;
    return group<4U ? (minimum?b:a)&63U : minimum ? (c>>4U)|((b>>6U)<<4U) : (c&15U)|((a>>6U)<<4U);
}
__device__ float half_float(unsigned bits) { return __half2float(__ushort_as_half(static_cast<unsigned short>(bits))); }
__global__ void validate_routes(const unsigned* routes, unsigned top_k, unsigned experts, unsigned* status) {
    if (threadIdx.x<top_k && routes[threadIdx.x]>=experts) atomicOr(status,invalid_route);
}
// One CTA quantizes one row; returns the CTA-uniform failure bits (0 when skipped).
template<bool Bf16>
__device__ unsigned quantize_row(const void* input, unsigned hidden, signed char* codes, float* deltas,
    int* sums, unsigned initial_status) {
    __shared__ __align__(16) signed char local_codes[max_hidden];
    __shared__ __align__(16) float local_deltas[max_hidden/32U];
    __shared__ __align__(16) int local_sums[max_hidden/32U];
    __shared__ unsigned bad;
    if(threadIdx.x==0) bad=0;
    __syncthreads();
    if(initial_status!=0) return 0;
    const unsigned lane=threadIdx.x&31U,wave=threadIdx.x/32U;
    for(unsigned group=wave;group<hidden/32U;group+=blockDim.x/32U) {
        const unsigned i=group*32U+lane;
        float x;
        if constexpr(Bf16) x=__uint_as_float(static_cast<unsigned>(static_cast<const unsigned short*>(input)[i])<<16U);
        else x=static_cast<const float*>(input)[i];
        if(!isfinite(x)) {atomicOr(&bad,nonfinite_activation);x=0.0F;}
        float a=fabsf(x);
        for(int stride=16;stride!=0;stride/=2) a=fmaxf(a,strix::qwen_dpp::xor32(a, stride));
        float delta=a==0.0F?1.0F:divide_rne(a,127.0F);
        if(!(delta>0.0F) || !isfinite(delta)) {atomicOr(&bad,invalid_delta);delta=1.0F;}
        const float rounded=rintf(divide_rne(x,delta));
        const int code=static_cast<int>(fminf(127.0F,fmaxf(-127.0F,rounded)));
        local_codes[i]=static_cast<signed char>(code);
        int sum=code;
        for(int stride=16;stride!=0;stride/=2) sum+=strix::qwen_dpp::xor32(sum, stride);
        if(lane==0) {local_deltas[group]=delta;local_sums[group]=sum;}
    }
    __syncthreads();
    if(bad!=0) return bad;
    for(unsigned i=threadIdx.x;i<hidden;i+=blockDim.x) codes[i]=local_codes[i];
    for(unsigned i=threadIdx.x;i<hidden/32U;i+=blockDim.x) {deltas[i]=local_deltas[i];sums[i]=local_sums[i];}
    return 0;
}
template<bool Bf16>
__global__ void quantize(const void* input, unsigned hidden, signed char* codes, float* deltas,
    int* sums, unsigned* status) {
    __shared__ unsigned initial_status;
    if(threadIdx.x==0) initial_status=*status;
    __syncthreads();
    const unsigned bad=quantize_row<Bf16>(input,hidden,codes,deltas,sums,initial_status);
    if(bad!=0 && threadIdx.x==0) atomicOr(status,bad);
}
// Parallel transcription of the stable CSR: assignment a=r*top_k+s is placed at
// offsets[e]+#{a'<a with expert e}, i.e. rows then slots ascending per expert.
// Only defined entries are written. Experts <= 512 <= blockDim. The union holds
// the first `active` rows; batches keep the physical `rows` cap of the kernels (at most 8:
// opt-in 16-row cohorts use 16-assignment batches). Histogram
// buckets saturate at 8.
__device__ void build_csr(const known_rows::PrologueRows& v, unsigned rows, unsigned active, unsigned experts, unsigned top_k, unsigned batch_cap) {
    const auto& csr=v.csr;
    const unsigned t=threadIdx.x,count=active*top_k;
    __shared__ unsigned expert_of[known_rows::max_assignments],first[known_rows::max_assignments],start[512],scan[3][512],hist[16],failure,duplicates,maximum;
    const unsigned cap=min(rows,batch_cap);
    if(t==0){failure=0U;duplicates=0U;maximum=0U;}
    if(t<16U)hist[t]=0U;
    __syncthreads();
    if(t<count) {
        const unsigned e=v.routes[t/top_k][t%top_k];expert_of[t]=e;
        if(e>=experts||!isfinite(v.route_weights[t/top_k][t%top_k]))atomicOr(&failure,invalid_route);
    }
    __syncthreads();
    if(failure){if(t<32U)csr.summary[t]=t==0U?failure:t==3U?count:t==6U?active:t==7U?1U:0U;return;}
    if(t<count){unsigned f=1U;for(unsigned a=t-t%top_k;a<t;++a)f&=expert_of[a]!=expert_of[t];first[t]=f;}
    __syncthreads();
    unsigned n=0U,unique=0U;
    if(t<experts)for(unsigned a=0;a<count;++a)if(expert_of[a]==t){++n;unique+=first[a];}
    const unsigned nb=(n+cap-1U)/cap;
    if(t<512U){scan[0][t]=n;scan[1][t]=n!=0U;scan[2][t]=nb;}
    __syncthreads();
    for(unsigned d=1U;d<512U;d<<=1U) {
        unsigned x[3]={};
        if(t>=d&&t<512U)for(unsigned k=0;k<3U;++k)x[k]=scan[k][t-d];
        __syncthreads();
        if(t>=d&&t<512U)for(unsigned k=0;k<3U;++k)scan[k][t]+=x[k];
        __syncthreads();
    }
    if(t<experts) {
        const unsigned off=scan[0][t]-n,touched=scan[1][t]-(n!=0U),batch=scan[2][t]-nb;
        start[t]=off;csr.offsets[t]=off;
        if(t==experts-1U)csr.offsets[experts]=off+n;
        if(n) {
            csr.touched[touched]=t;atomicAdd(&hist[min(unique,8U)-1U],1U);atomicAdd(&duplicates,n-unique);atomicMax(&maximum,n);
            for(unsigned b=0;b<nb;++b){csr.experts[batch+b]=t;csr.begins[batch+b]=off+b*cap;atomicAdd(&hist[8U+min(8U,min(cap,n-b*cap))-1U],1U);}
        }
    }
    __syncthreads();
    if(t<count) {
        const unsigned e=expert_of[t];unsigned rank=0U;
        for(unsigned a=0;a<t;++a)rank+=expert_of[a]==e;
        csr.rows[start[e]+rank]=t/top_k;csr.slots[start[e]+rank]=t%top_k;
    }
    if(t==0U)csr.begins[scan[2][511]]=count;
    if(t<32U)csr.summary[t]=t==1U?scan[1][511]:t==2U?scan[2][511]:t==3U?count:t==4U?duplicates:t==5U?maximum:
        t==6U?active:t==7U?1U:t>=8U&&t<24U?hist[t-8U]:0U;
}
// Window prologue, 1024-thread CTA per row: the incumbent clear, both route
// validators and the quantizer in their original order; status is written
// once by thread 0. CTA `rows` builds the CSR of the first `active` rows concurrently.
template<bool Bf16>
__global__ void window_prologue(known_rows::PrologueRows v, unsigned rows, unsigned hidden, unsigned top_k, unsigned experts,
    unsigned clear_words, unsigned active, unsigned batch_cap) {
    const unsigned row=blockIdx.x;
    if(row==rows){build_csr(v,rows,active,experts,top_k,batch_cap);return;}
    __shared__ unsigned route_bits;
    if(threadIdx.x==0) route_bits=0;
    for(unsigned i=threadIdx.x;i<clear_words;i+=blockDim.x) v.clear[row][i]=0.0F;
    __syncthreads();
    if(threadIdx.x<top_k && (v.routes[row][threadIdx.x]>=experts || !isfinite(v.route_weights[row][threadIdx.x])))
        atomicOr(&route_bits,invalid_route);
    __syncthreads();
    const unsigned bad=quantize_row<Bf16>(v.input[row],hidden,v.codes[row],v.deltas[row],v.sums[row],route_bits);
    if(threadIdx.x==0) *v.status[row]=route_bits|bad;
}
template<class T> __device__ __forceinline__ T* stride_row(known_rows::StridedRows<T> v,unsigned r) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(v.base)+static_cast<std::uint64_t>(r)*v.stride);
}
// Prefill rows: the window prologue's row CTA on strided views (CSR built separately).
template<bool Bf16>
__global__ void prefill_prologue(known_rows::PrefillRows v, unsigned hidden, unsigned top_k, unsigned experts,
    unsigned clear_words) {
    const unsigned row=blockIdx.x;
    __shared__ unsigned route_bits;
    if(threadIdx.x==0) route_bits=0;
    float* clear=stride_row(v.clear,row);
    for(unsigned i=threadIdx.x;i<clear_words;i+=blockDim.x) clear[i]=0.0F;
    __syncthreads();
    if(threadIdx.x<top_k && (stride_row(v.routes,row)[threadIdx.x]>=experts || !isfinite(stride_row(v.route_weights,row)[threadIdx.x])))
        atomicOr(&route_bits,invalid_route);
    __syncthreads();
    const unsigned bad=quantize_row<Bf16>(stride_row(v.input,row),hidden,stride_row(v.codes,row),stride_row(v.deltas,row),
        stride_row(v.sums,row),route_bits);
    if(threadIdx.x==0) *stride_row(v.status,row)=route_bits|bad;
}
// One serial row (STRIX_MOE_R1_GLUE_FUSE): the window prologue's row CTA on the serial bindings. It replaces the clear of
// the accumulation and status words, validate_routes_kernel (route index or nonfinite weight -> invalid_route),
// validate_routes (route index -> invalid_route) and quantize: the same status word (route bits, else the quantizer's
// bits), the same codes/deltas/sums and the same zeroed accumulation.
template<bool Bf16>
__global__ void serial_prologue(const void* input, const unsigned* routes, const float* route_weights, float* clear,
    unsigned clear_words, unsigned hidden, unsigned top_k, unsigned experts, signed char* codes, float* deltas, int* sums,
    unsigned* status) {
    __shared__ unsigned route_bits;
    if(threadIdx.x==0) route_bits=0;
    for(unsigned i=threadIdx.x;i<clear_words;i+=blockDim.x) clear[i]=0.0F;
    __syncthreads();
    if(threadIdx.x<top_k && (routes[threadIdx.x]>=experts || !isfinite(route_weights[threadIdx.x])))
        atomicOr(&route_bits,invalid_route);
    __syncthreads();
    const unsigned bad=quantize_row<Bf16>(input,hidden,codes,deltas,sums,route_bits);
    if(threadIdx.x==0) *status=route_bits|bad;
}
__global__ void project(const unsigned char* gate,const unsigned char* up,const unsigned* routes,
    const signed char* codes,const float* deltas,const int* sums,const unsigned* status,
    float* output,unsigned hidden,unsigned rows) {
    __shared__ __align__(16) signed char local_codes[max_hidden];
    __shared__ __align__(16) float local_deltas[max_hidden/32U];
    __shared__ __align__(16) int local_sums[max_hidden/32U];
    // Prior kernels are on the same stream; this kernel never changes status.
    if(*status!=0) return;
    for(unsigned i=threadIdx.x;i<hidden;i+=256U) local_codes[i]=codes[i];
    for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U) {local_deltas[i]=deltas[i];local_sums[i]=sums[i];}
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,c=lane&7U,p=c/2U,h=c&1U;
    const unsigned row=blockIdx.x*32U+(threadIdx.x/32U)*4U+lane/8U;
    const unsigned which=blockIdx.y,slot=blockIdx.z,expert=routes[slot];
    const unsigned blocks=hidden/256U;
    const auto* weights=(which==0?gate:up)+(static_cast<unsigned long long>(expert)*rows+row)*blocks*144U;
    float acc=0.0F;
    #pragma unroll 1
    for(unsigned block=0;block<blocks;++block) {
        const auto* packed=weights+block*144U;
        const uint4 hd=*reinterpret_cast<const uint4*>(packed);
        const uint4 q=*reinterpret_cast<const uint4*>(packed+16U+c*16U);
        const int4 a0=*reinterpret_cast<const int4*>(local_codes+block*256U+p*64U+h*16U);
        const int4 a1=*reinterpret_cast<const int4*>(local_codes+block*256U+p*64U+32U+h*16U);
        constexpr unsigned mask=0x0f0f0f0fU;
        int r0=dot(a0.x,q.x&mask,0);r0=dot(a0.y,q.y&mask,r0);r0=dot(a0.z,q.z&mask,r0);r0=dot(a0.w,q.w&mask,r0);
        int r1=dot(a1.x,(q.x>>4U)&mask,0);r1=dot(a1.y,(q.y>>4U)&mask,r1);r1=dot(a1.z,(q.z>>4U)&mask,r1);r1=dot(a1.w,(q.w>>4U)&mask,r1);
        r0+=strix::qwen_dpp::xor32(r0, 1);r1+=strix::qwen_dpp::xor32(r1, 1);
        const unsigned g0=2U*p,g1=g0+1U;
        const int i0=static_cast<int>(factor(hd,g0,false))*r0,i1=static_cast<int>(factor(hd,g1,false))*r1;
        const int j0=static_cast<int>(factor(hd,g0,true))*local_sums[block*8U+g0];
        const int j1=static_cast<int>(factor(hd,g1,true))*local_sums[block*8U+g1];
        const float d=half_float(hd.x&65535U),dm=half_float(hd.x>>16U);
        const float u0=d*static_cast<float>(i0),v0=dm*static_cast<float>(j0),t0=u0-v0;
        const float u1=d*static_cast<float>(i1),v1=dm*static_cast<float>(j1),t1=u1-v1;
        const float c0=local_deltas[block*8U+g0]*t0,c1=local_deltas[block*8U+g1]*t1;
        const float paired=c0+c1;
        acc=acc+paired;
    }
    // Only c0 is observable. c0 receives exactly ((p0+p1)+(p2+p3)).
    acc=acc+strix::qwen_dpp::xor32(acc, 2);
    acc=acc+strix::qwen_dpp::xor32(acc, 4);
    if(c==0) output[(slot*2U+which)*rows+row]=acc;
}
// Distinct unsigned-five-bit format; incumbent Q4K symbol/body above stays literal.
__global__ void project_q5k(const unsigned char* gate,const unsigned char* up,const unsigned* routes,
    const signed char* codes,const float* deltas,const int* sums,const unsigned* status,
    float* output,unsigned hidden,unsigned rows) {
    __shared__ __align__(16) signed char local_codes[max_hidden];
    __shared__ __align__(16) float local_deltas[max_hidden/32U];
    __shared__ __align__(16) int local_sums[max_hidden/32U];
    // Prior kernels are on the same stream; this kernel never changes status.
    if(*status!=0) return;
    for(unsigned i=threadIdx.x;i<hidden;i+=256U) local_codes[i]=codes[i];
    for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U) {local_deltas[i]=deltas[i];local_sums[i]=sums[i];}
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,c=lane&7U,p=c/2U,h=c&1U;
    const unsigned row=blockIdx.x*32U+(threadIdx.x/32U)*4U+lane/8U;
    const unsigned which=blockIdx.y,slot=blockIdx.z,expert=routes[slot];
    const unsigned blocks=hidden/256U;
    const auto* weights=(which==0?gate:up)+(static_cast<unsigned long long>(expert)*rows+row)*blocks*176U;
    float acc=0.0F;
    #pragma unroll 1
    for(unsigned block=0;block<blocks;++block) {
        const auto* packed=weights+block*176U;
        const uint4 hd=*reinterpret_cast<const uint4*>(packed);
        const uint4 q=*reinterpret_cast<const uint4*>(packed+48U+c*16U);
        const int4 a0=*reinterpret_cast<const int4*>(local_codes+block*256U+p*64U+h*16U);
        const int4 a1=*reinterpret_cast<const int4*>(local_codes+block*256U+p*64U+32U+h*16U);
        const uint4 qh=*reinterpret_cast<const uint4*>(packed+16U+h*16U);
        constexpr unsigned mask=0x0f0f0f0fU;
        const auto g0=2U*p,g1=g0+1U;
        const uint4 q0{(q.x&mask)|(((qh.x>>g0)&0x01010101U)<<4U),
            (q.y&mask)|(((qh.y>>g0)&0x01010101U)<<4U),
            (q.z&mask)|(((qh.z>>g0)&0x01010101U)<<4U),
            (q.w&mask)|(((qh.w>>g0)&0x01010101U)<<4U)};
        const uint4 q1{((q.x>>4U)&mask)|(((qh.x>>g1)&0x01010101U)<<4U),
            ((q.y>>4U)&mask)|(((qh.y>>g1)&0x01010101U)<<4U),
            ((q.z>>4U)&mask)|(((qh.z>>g1)&0x01010101U)<<4U),
            ((q.w>>4U)&mask)|(((qh.w>>g1)&0x01010101U)<<4U)};
        int r0=dot(a0.x,q0.x,0);r0=dot(a0.y,q0.y,r0);r0=dot(a0.z,q0.z,r0);r0=dot(a0.w,q0.w,r0);
        int r1=dot(a1.x,q1.x,0);r1=dot(a1.y,q1.y,r1);r1=dot(a1.z,q1.z,r1);r1=dot(a1.w,q1.w,r1);
        r0+=strix::qwen_dpp::xor32(r0, 1);r1+=strix::qwen_dpp::xor32(r1, 1);
        const int i0=static_cast<int>(factor(hd,g0,false))*r0,i1=static_cast<int>(factor(hd,g1,false))*r1;
        const int j0=static_cast<int>(factor(hd,g0,true))*local_sums[block*8U+g0];
        const int j1=static_cast<int>(factor(hd,g1,true))*local_sums[block*8U+g1];
        const float d=half_float(hd.x&65535U),dm=half_float(hd.x>>16U);
        const float u0=d*static_cast<float>(i0),v0=dm*static_cast<float>(j0),t0=u0-v0;
        const float u1=d*static_cast<float>(i1),v1=dm*static_cast<float>(j1),t1=u1-v1;
        const float c0=local_deltas[block*8U+g0]*t0,c1=local_deltas[block*8U+g1]*t1;
        const float paired=c0+c1;
        acc=acc+paired;
    }
    // Only c0 is observable. c0 receives exactly ((p0+p1)+(p2+p3)).
    acc=acc+strix::qwen_dpp::xor32(acc, 2);
    acc=acc+strix::qwen_dpp::xor32(acc, 4);
    if(c==0) output[(slot*2U+which)*rows+row]=acc;
}
// GSQ-RCO Q2_0 gate/up (q20_numerical_identity): 8 lanes per output row, lane c owns the A8 groups
// g = c (mod 8). Group g = half g%2 of the 18-byte block g/2 (fp16 d, then 64 2-bit codes, element j
// at bits 2(j%4) of byte j/4): each code byte expands to one u8x4 word in element order.
__device__ __forceinline__ unsigned q20_word(unsigned byte) {
    return (byte|(byte<<6U)|(byte<<12U)|(byte<<18U))&0x03030303U;
}
__device__ __forceinline__ unsigned q20_u16(const unsigned char* p) {
    return static_cast<unsigned>(p[0])|(static_cast<unsigned>(p[1])<<8U);
}
// delta_g * (d * float(I - S)) of one 32-code group: half of `block`, activation image `a` (8 words).
__device__ __forceinline__ float q20_group(const unsigned char* block,unsigned half,const int* a,int sum,float delta) {
    const auto* codes=block+2U+half*8U;
    const unsigned w01=q20_u16(codes),w23=q20_u16(codes+2U),w45=q20_u16(codes+4U),w67=q20_u16(codes+6U);
    int integer=dot(a[0],q20_word(w01&255U),0);integer=dot(a[1],q20_word(w01>>8U),integer);
    integer=dot(a[2],q20_word(w23&255U),integer);integer=dot(a[3],q20_word(w23>>8U),integer);
    integer=dot(a[4],q20_word(w45&255U),integer);integer=dot(a[5],q20_word(w45>>8U),integer);
    integer=dot(a[6],q20_word(w67&255U),integer);integer=dot(a[7],q20_word(w67>>8U),integer);
    const float d=half_float(q20_u16(block));
    const float scaled=d*static_cast<float>(integer-sum);
    return delta*scaled;
}
__global__ void project_q20(const unsigned char* gate,const unsigned char* up,const unsigned* routes,
    const signed char* codes,const float* deltas,const int* sums,const unsigned* status,
    float* output,unsigned hidden,unsigned rows) {
    __shared__ __align__(16) signed char local_codes[max_hidden];
    __shared__ __align__(16) float local_deltas[max_hidden/32U];
    __shared__ __align__(16) int local_sums[max_hidden/32U];
    if(*status!=0) return;
    for(unsigned i=threadIdx.x;i<hidden;i+=256U) local_codes[i]=codes[i];
    for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U) {local_deltas[i]=deltas[i];local_sums[i]=sums[i];}
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,c=lane&7U;
    const unsigned row=blockIdx.x*32U+(threadIdx.x/32U)*4U+lane/8U;
    const unsigned which=blockIdx.y,slot=blockIdx.z,expert=routes[slot];
    const unsigned groups=hidden/32U;
    const auto* weights=(which==0?gate:up)+(static_cast<unsigned long long>(expert)*rows+row)*(hidden/64U)*18U;
    float acc=0.0F;
    #pragma unroll 1
    for(unsigned g=c;g<groups;g+=8U) {
        const float value=q20_group(weights+(g/2U)*18U,g&1U,reinterpret_cast<const int*>(local_codes+g*32U),
            local_sums[g],local_deltas[g]);
        acc=acc+value;
    }
    acc=acc+strix::qwen_dpp::xor32(acc, 1);
    acc=acc+strix::qwen_dpp::xor32(acc, 2);
    acc=acc+strix::qwen_dpp::xor32(acc, 4);
    if(c==0) output[(slot*2U+which)*rows+row]=acc;
}
// GSQ IQ gate/up (gsq_a8_groups.hpp): project_q20 lanes/tree with signed int8 group weights.
template<std::uint32_t T>
__global__ void project_iq(const unsigned char* gate,const unsigned char* up,const unsigned* routes,
    const signed char* codes,const float* deltas,const unsigned* status,
    float* output,unsigned hidden,unsigned rows) {
    __shared__ __align__(16) signed char local_codes[max_hidden];
    __shared__ __align__(16) float local_deltas[max_hidden/32U];
    if(*status!=0) return;
    for(unsigned i=threadIdx.x;i<hidden;i+=256U) local_codes[i]=codes[i];
    for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U) local_deltas[i]=deltas[i];
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,c=lane&7U;
    const unsigned row=blockIdx.x*32U+(threadIdx.x/32U)*4U+lane/8U;
    const unsigned which=blockIdx.y,slot=blockIdx.z,expert=routes[slot];
    const unsigned groups=hidden/32U;
    const auto* weights=(which==0?gate:up)+(static_cast<unsigned long long>(expert)*rows+row)*gsq_a8::row_bytes(T,hidden);
    float acc=0.0F;
    #pragma unroll 1
    for(unsigned g=c;g<groups;g+=8U) {
        const auto x=gsq_a8::iq_group<T>(weights,g);
        const float value=gsq_a8::iq_apply<T>(x,reinterpret_cast<const int*>(local_codes+g*32U),local_deltas[g]);
        acc=acc+value;
    }
    acc=acc+strix::qwen_dpp::xor32(acc, 1);
    acc=acc+strix::qwen_dpp::xor32(acc, 2);
    acc=acc+strix::qwen_dpp::xor32(acc, 4);
    if(c==0) output[(slot*2U+which)*rows+row]=acc;
}
// Two independent projection DAGs consume the same staged activation values.
// Keep all affine corrections, operand order and rounding boundaries literal.
__device__ __forceinline__ float paired_shared(const unsigned char* packed,int4 a0,int4 a1,
    int s0,int s1,float e0,float e1,unsigned c,unsigned p) {
    const uint4 hd=*reinterpret_cast<const uint4*>(packed);
    const uint4 q=*reinterpret_cast<const uint4*>(packed+16U+c*16U);
    constexpr unsigned mask=0x0f0f0f0fU;
    int r0=dot(a0.x,q.x&mask,0);r0=dot(a0.y,q.y&mask,r0);r0=dot(a0.z,q.z&mask,r0);r0=dot(a0.w,q.w&mask,r0);
    int r1=dot(a1.x,(q.x>>4U)&mask,0);r1=dot(a1.y,(q.y>>4U)&mask,r1);r1=dot(a1.z,(q.z>>4U)&mask,r1);r1=dot(a1.w,(q.w>>4U)&mask,r1);
    r0+=strix::qwen_dpp::xor32(r0, 1);r1+=strix::qwen_dpp::xor32(r1, 1);
    const unsigned g0=2U*p,g1=g0+1U;
    const int i0=static_cast<int>(factor(hd,g0,false))*r0,i1=static_cast<int>(factor(hd,g1,false))*r1;
    const int j0=static_cast<int>(factor(hd,g0,true))*s0;
    const int j1=static_cast<int>(factor(hd,g1,true))*s1;
    const float d=half_float(hd.x&65535U),dm=half_float(hd.x>>16U);
    const float u0=d*static_cast<float>(i0),v0=dm*static_cast<float>(j0),t0=u0-v0;
    const float u1=d*static_cast<float>(i1),v1=dm*static_cast<float>(j1),t1=u1-v1;
    const float c0=e0*t0,c1=e1*t1;
    return c0+c1;
}
__global__ void project_shared_gate_up(const unsigned char* gate,const unsigned char* up,const unsigned* routes,
    const signed char* codes,const float* deltas,const int* sums,const unsigned* status,
    float* output,unsigned hidden,unsigned rows) {
    __shared__ __align__(16) signed char local_codes[max_hidden];
    __shared__ __align__(16) float local_deltas[max_hidden/32U];
    __shared__ __align__(16) int local_sums[max_hidden/32U];
    if(*status!=0) return;
    for(unsigned i=threadIdx.x;i<hidden;i+=256U) local_codes[i]=codes[i];
    for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U) {local_deltas[i]=deltas[i];local_sums[i]=sums[i];}
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,c=lane&7U,p=c/2U,h=c&1U;
    const unsigned row=blockIdx.x*32U+(threadIdx.x/32U)*4U+lane/8U;
    const unsigned slot=blockIdx.z,expert=routes[slot],blocks=hidden/256U;
    const auto offset=(static_cast<unsigned long long>(expert)*rows+row)*blocks*144U;
    const auto* gate_weights=gate+offset;const auto* up_weights=up+offset;
    const unsigned g0=2U*p,g1=g0+1U;
    float acc_gate=0.0F,acc_up=0.0F;
    #pragma unroll 1
    for(unsigned block=0;block<blocks;++block) {
        const int4 a0=*reinterpret_cast<const int4*>(local_codes+block*256U+p*64U+h*16U);
        const int4 a1=*reinterpret_cast<const int4*>(local_codes+block*256U+p*64U+32U+h*16U);
        const int s0=local_sums[block*8U+g0],s1=local_sums[block*8U+g1];
        const float e0=local_deltas[block*8U+g0],e1=local_deltas[block*8U+g1];
        const float paired_gate=paired_shared(gate_weights+block*144U,a0,a1,s0,s1,e0,e1,c,p);
        acc_gate=acc_gate+paired_gate;
        const float paired_up=paired_shared(up_weights+block*144U,a0,a1,s0,s1,e0,e1,c,p);
        acc_up=acc_up+paired_up;
    }
    acc_gate=acc_gate+strix::qwen_dpp::xor32(acc_gate, 2);acc_gate=acc_gate+strix::qwen_dpp::xor32(acc_gate, 4);
    acc_up=acc_up+strix::qwen_dpp::xor32(acc_up, 2);acc_up=acc_up+strix::qwen_dpp::xor32(acc_up, 4);
    if(c==0) {output[(slot*2U+0U)*rows+row]=acc_gate;output[(slot*2U+1U)*rows+row]=acc_up;}
}
__global__ void division_receipt(const unsigned* a,const unsigned* b,unsigned* output,unsigned count) {
    const unsigned i=blockIdx.x*256U+threadIdx.x;
    if(i<count) output[i]=__float_as_uint(divide_rne(__uint_as_float(a[i]),__uint_as_float(b[i])));
}
__global__ void half_conversion(unsigned* output) {
    const unsigned i=blockIdx.x*256U+threadIdx.x;
    output[i]=__float_as_uint(half_float(i));
}
using strix::hip_check;
}
namespace {
void admit_q5k_stream(hipStream_t stream) {
    if (!stream || stream == hipStreamLegacy || stream == hipStreamPerThread)
        throw std::invalid_argument("Q5K A8 requires explicitly owned eager stream");
    const auto* context = current_hip_launch_context();
    if (context && (context->stream != stream || context->graph_capture || context->status.arena_base ||
        context->arguments.memory.arena_base))
        throw std::invalid_argument("Q5K A8 rejects graph or foreign launch context");
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    hip_check(hipStreamIsCapturing(stream,&capture),"Q5K A8 capture query");
    if (capture != hipStreamCaptureStatusNone)
        throw std::invalid_argument("Q5K A8 rejects capture before enqueue");
}
}
namespace {
void enqueue_projection(const Request& request,const Bindings& b,hipStream_t stream) {
    auto* status=reinterpret_cast<unsigned*>(b.status.address);
    const auto* routes=reinterpret_cast<const unsigned*>(b.routes.address);
    auto* codes=reinterpret_cast<signed char*>(b.codes.address);auto* deltas=reinterpret_cast<float*>(b.deltas.address);
    auto* sums=reinterpret_cast<int*>(b.sums.address);
    const auto iq=[&]<std::uint32_t T>(){
        project_iq<T><<<dim3(request.shape.rows/32U,2,request.shape.top_k),256,0,stream>>>(
            reinterpret_cast<const unsigned char*>(b.gate.address),reinterpret_cast<const unsigned char*>(b.up.address),
            routes,codes,deltas,status,reinterpret_cast<float*>(b.output.address),request.shape.hidden,request.shape.rows);
    };
    if (request.gate_type == gsq_a8::type_iq2_s) iq.template operator()<gsq_a8::type_iq2_s>();
    else if (request.gate_type == gsq_a8::type_iq3_xxs) iq.template operator()<gsq_a8::type_iq3_xxs>();
    else if (request.gate_type == gsq_a8::type_iq3_s) iq.template operator()<gsq_a8::type_iq3_s>();
    else if (request.gate_type == gsq_a8::type_iq4_xs) iq.template operator()<gsq_a8::type_iq4_xs>();
    else if (request.gate_type == 42) {
        project_q20<<<dim3(request.shape.rows/32U,2,request.shape.top_k),256,0,stream>>>(
            reinterpret_cast<const unsigned char*>(b.gate.address),reinterpret_cast<const unsigned char*>(b.up.address),
            routes,codes,deltas,sums,status,reinterpret_cast<float*>(b.output.address),request.shape.hidden,request.shape.rows);
    } else if (request.gate_type == 13) {
        project_q5k<<<dim3(request.shape.rows/32U,2,request.shape.top_k),256,0,stream>>>(
            reinterpret_cast<const unsigned char*>(b.gate.address),reinterpret_cast<const unsigned char*>(b.up.address),
            routes,codes,deltas,sums,status,reinterpret_cast<float*>(b.output.address),request.shape.hidden,request.shape.rows);
    } else {
        project<<<dim3(request.shape.rows/32U,2,request.shape.top_k),256,0,stream>>>(
            reinterpret_cast<const unsigned char*>(b.gate.address),reinterpret_cast<const unsigned char*>(b.up.address),
            routes,codes,deltas,sums,status,reinterpret_cast<float*>(b.output.address),request.shape.hidden,request.shape.rows);
    }
    hip_check(hipGetLastError(),"Q4K A8 projection enqueue");
}
} // namespace
void enqueue_probe(const Request& request,const Bindings& b,hipStream_t stream) {
    validate(request,b);
    if (request.gate_type == 13 || gsq_a8::gate_type(request.gate_type)) admit_q5k_stream(stream);
    hip_check(hipPeekAtLastError(),"Q4K A8 pre-enqueue stale error");
    auto* status=reinterpret_cast<unsigned*>(b.status.address);
    const auto* routes=reinterpret_cast<const unsigned*>(b.routes.address);
    validate_routes<<<1,32,0,stream>>>(routes,request.shape.top_k,request.shape.experts,status);
    hip_check(hipGetLastError(),"Q4K A8 route enqueue");
    const void* input=reinterpret_cast<const void*>(b.input.address);
    auto* codes=reinterpret_cast<signed char*>(b.codes.address);auto* deltas=reinterpret_cast<float*>(b.deltas.address);
    auto* sums=reinterpret_cast<int*>(b.sums.address);
    if(request.input==InputKind::bf16) quantize<true><<<1,256,0,stream>>>(input,request.shape.hidden,codes,deltas,sums,status);
    else quantize<false><<<1,256,0,stream>>>(input,request.shape.hidden,codes,deltas,sums,status);
    hip_check(hipGetLastError(),"Q4K A8 quantizer enqueue");
    enqueue_projection(request,b,stream);
}
void enqueue_serial_prologue_probe(const Request& request,const Bindings& b,float* clear,unsigned clear_words,
    const float* route_weights,hipStream_t stream) {
    validate(request,b);
    if (request.gate_type == 13 || gsq_a8::gate_type(request.gate_type)) admit_q5k_stream(stream);
    if(!clear || !route_weights || request.shape.top_k>1024U) throw std::invalid_argument("Q4K A8 serial prologue bindings are invalid");
    hip_check(hipPeekAtLastError(),"Q4K A8 pre-enqueue stale error");
    auto* status=reinterpret_cast<unsigned*>(b.status.address);
    const auto* routes=reinterpret_cast<const unsigned*>(b.routes.address);
    const void* input=reinterpret_cast<const void*>(b.input.address);
    auto* codes=reinterpret_cast<signed char*>(b.codes.address);auto* deltas=reinterpret_cast<float*>(b.deltas.address);
    auto* sums=reinterpret_cast<int*>(b.sums.address);
    if(request.input==InputKind::bf16) serial_prologue<true><<<1,1024,0,stream>>>(input,routes,route_weights,clear,clear_words,
        request.shape.hidden,request.shape.top_k,request.shape.experts,codes,deltas,sums,status);
    else serial_prologue<false><<<1,1024,0,stream>>>(input,routes,route_weights,clear,clear_words,
        request.shape.hidden,request.shape.top_k,request.shape.experts,codes,deltas,sums,status);
    hip_check(hipGetLastError(),"Q4K A8 serial prologue enqueue");
    enqueue_projection(request,b,stream);
}
void enqueue_probe_shared_gate_up(const Request& request,const Bindings& b,hipStream_t stream) {
    validate(request,b);
    if (request.gate_type != 12) throw std::invalid_argument("shared gate/up staging supports Q4K only");
    hip_check(hipPeekAtLastError(),"Q4K A8 pre-enqueue stale error");
    auto* status=reinterpret_cast<unsigned*>(b.status.address);
    const auto* routes=reinterpret_cast<const unsigned*>(b.routes.address);
    validate_routes<<<1,32,0,stream>>>(routes,request.shape.top_k,request.shape.experts,status);
    hip_check(hipGetLastError(),"Q4K A8 route enqueue");
    const void* input=reinterpret_cast<const void*>(b.input.address);
    auto* codes=reinterpret_cast<signed char*>(b.codes.address);auto* deltas=reinterpret_cast<float*>(b.deltas.address);
    auto* sums=reinterpret_cast<int*>(b.sums.address);
    if(request.input==InputKind::bf16) quantize<true><<<1,256,0,stream>>>(input,request.shape.hidden,codes,deltas,sums,status);
    else quantize<false><<<1,256,0,stream>>>(input,request.shape.hidden,codes,deltas,sums,status);
    hip_check(hipGetLastError(),"Q4K A8 quantizer enqueue");
    project_shared_gate_up<<<dim3(request.shape.rows/32U,1,request.shape.top_k),256,0,stream>>>(
        reinterpret_cast<const unsigned char*>(b.gate.address),reinterpret_cast<const unsigned char*>(b.up.address),
        routes,codes,deltas,sums,status,reinterpret_cast<float*>(b.output.address),request.shape.hidden,request.shape.rows);
    hip_check(hipGetLastError(),"Q4K A8 shared gate/up projection enqueue");
}
void enqueue_quantize_only(const Request& request,const Bindings& b,hipStream_t stream) {
    validate(request,b);
    if (request.gate_type == 13 || gsq_a8::gate_type(request.gate_type)) admit_q5k_stream(stream);
    hip_check(hipPeekAtLastError(),"Q4K A8 pre-enqueue stale error");
    auto* status=reinterpret_cast<unsigned*>(b.status.address);
    const auto* routes=reinterpret_cast<const unsigned*>(b.routes.address);
    validate_routes<<<1,32,0,stream>>>(routes,request.shape.top_k,request.shape.experts,status);
    hip_check(hipGetLastError(),"Q4K A8 route enqueue");
    const void* input=reinterpret_cast<const void*>(b.input.address);
    auto* codes=reinterpret_cast<signed char*>(b.codes.address);auto* deltas=reinterpret_cast<float*>(b.deltas.address);
    auto* sums=reinterpret_cast<int*>(b.sums.address);
    if(request.input==InputKind::bf16)quantize<true><<<1,256,0,stream>>>(input,request.shape.hidden,codes,deltas,sums,status);
    else quantize<false><<<1,256,0,stream>>>(input,request.shape.hidden,codes,deltas,sums,status);
    hip_check(hipGetLastError(),"Q4K A8 quantizer enqueue");
}
namespace known_rows {
void enqueue_prologue(const Request& request,const PrologueRows& rows,unsigned count,unsigned clear_words,hipStream_t stream,
    unsigned active, unsigned batch_cap) {
    if(!active) active=count;
    if(count==0U||count>known_rows::max_union_rows||count*request.shape.top_k>known_rows::max_assignments||request.shape.experts>512U||active>count)
        throw std::invalid_argument("A8 window prologue geometry is invalid");
    hip_check(hipPeekAtLastError(),"A8 window prologue stale error");
    if(request.input==InputKind::bf16)window_prologue<true><<<count+1U,1024,0,stream>>>(rows,count,request.shape.hidden,request.shape.top_k,request.shape.experts,clear_words,active,batch_cap);
    else window_prologue<false><<<count+1U,1024,0,stream>>>(rows,count,request.shape.hidden,request.shape.top_k,request.shape.experts,clear_words,active,batch_cap);
    hip_check(hipGetLastError(),"A8 window prologue enqueue");
}
void enqueue_prefill_prologue(const Request& request,const PrefillRows& rows,unsigned count,unsigned clear_words,hipStream_t stream) {
    if(count<16U||count%8U||count>q51_known_rows::prefill_max_rows||request.shape.top_k>10U||request.shape.experts>512U)
        throw std::invalid_argument("A8 prefill prologue geometry is invalid");
    hip_check(hipPeekAtLastError(),"A8 prefill prologue stale error");
    if(request.input==InputKind::bf16)prefill_prologue<true><<<count,1024,0,stream>>>(rows,request.shape.hidden,request.shape.top_k,request.shape.experts,clear_words);
    else prefill_prologue<false><<<count,1024,0,stream>>>(rows,request.shape.hidden,request.shape.top_k,request.shape.experts,clear_words);
    hip_check(hipGetLastError(),"A8 prefill prologue enqueue");
}
} // namespace known_rows
void enqueue_division_probe(const std::uint32_t* a,const std::uint32_t* b,std::uint32_t* output,unsigned count,hipStream_t stream) {
    if(!a || !b || !output || count==0) throw std::invalid_argument("invalid division receipt bindings");
    division_receipt<<<(count+255U)/256U,256,0,stream>>>(a,b,output,count);hip_check(hipGetLastError(),"Q4K A8 division receipt enqueue");
}
void enqueue_half_conversion_probe(std::uint32_t* output,hipStream_t stream) {
    if(output==nullptr) throw std::invalid_argument("null half conversion destination");
    half_conversion<<<256,256,0,stream>>>(output);hip_check(hipGetLastError(),"Q4K A8 half conversion enqueue");
}

// A8 down is a separate, explicitly selected numerical contract.
// The incumbent Q4K quantize/project arithmetic bodies above remain unchanged.
namespace {
__device__ void down_error(unsigned* status,unsigned* aggregate,unsigned bits) {
    atomicOr(status,bits);
    if(aggregate) atomicOr(aggregate,down_aggregate_error);
}
__global__ void down_activation_quantize(float* intermediates,const unsigned* routes,
    signed char* codes,float* deltas,int* sums,unsigned* status,unsigned* aggregate,
    unsigned width,unsigned experts) {
    if(*status!=0) return;
    const unsigned lane=threadIdx.x&31U,wave=threadIdx.x/32U;
    const unsigned group=blockIdx.x*8U+wave,slot=blockIdx.y;
    if(group>=width/32U) return; // whole wave inactive, no cross-wave collectives
    if(routes[slot]>=experts) {if(lane==0) down_error(status,aggregate,down_invalid_route);return;}
    const unsigned at=slot*2U*width+group*32U+lane;
    const float g=intermediates[at],u=intermediates[at+width];
    const float h=(g/(1.0F+expf(-g)))*u;
    const unsigned bad=static_cast<unsigned>(__ballot(!isfinite(g)||!isfinite(u)||!isfinite(h)));
    if(bad!=0) {if(lane==0) down_error(status,aggregate,down_nonfinite_activation);return;}
    float a=fabsf(h);
    for(int stride=16;stride!=0;stride/=2) a=fmaxf(a,strix::qwen_dpp::xor32(a, stride));
    const bool tiny=a>0.0F && a<0x1p-119F;
    const float delta=a==0.0F || tiny?1.0F:divide_rne(a,127.0F);
    const float rounded=tiny?0.0F:rintf(divide_rne(h,delta));
    const int code=static_cast<int>(fminf(127.0F,fmaxf(-127.0F,rounded)));
    int sum=code;
    for(int stride=16;stride!=0;stride/=2) sum+=strix::qwen_dpp::xor32(sum, stride);
    intermediates[at]=h;
    codes[slot*width+group*32U+lane]=static_cast<signed char>(code);
    if(lane==0) {deltas[slot*(width/32U)+group]=delta;sums[slot*(width/32U)+group]=sum;}
}
__device__ unsigned down_u16(const unsigned char* p) {
    return static_cast<unsigned>(p[0])|(static_cast<unsigned>(p[1])<<8U);
}
__device__ unsigned down_u32(const unsigned char* p) {
    // Q8_0 rows have34-byte blocks: never require a misaligned dword load.
    return down_u16(p)|(down_u16(p+2U)<<16U);
}
__device__ unsigned down_q51_word(const unsigned char* block,unsigned first) {
    const unsigned high=down_u32(block+4U);
    unsigned result=0;
    #pragma unroll
    for(unsigned j=0;j<4U;++j) {
        const unsigned i=first+j;
        const unsigned packed=block[8U+i%16U];
        const unsigned low=(i<16U?packed:packed>>4U)&15U;
        const unsigned q=low|(((high>>i)&1U)<<4U);
        result|=q<<(8U*j);
    }
    return result;
}
template<bool Q51>
__global__ void down_rawdots(const unsigned char* weights,const unsigned* routes,
    const signed char* codes,const float* deltas,const int* sums,float* rawdots,
    unsigned* status,unsigned* aggregate,unsigned hidden,unsigned width,unsigned experts) {
    if(*status!=0) return;
    const unsigned lane=threadIdx.x&31U,owner=lane&3U;
    const unsigned row=blockIdx.x*64U+(threadIdx.x/32U)*8U+lane/4U;
    const unsigned slot=blockIdx.y,expert=routes[slot],groups=width/32U;
    if(expert>=experts) {if(threadIdx.x==0) down_error(status,aggregate,down_invalid_route);return;}
    constexpr unsigned block_bytes=Q51?24U:34U;
    const auto* row_weights=weights+(static_cast<unsigned long long>(expert)*hidden+row)*groups*block_bytes;
    float acc=0.0F;
    #pragma unroll 1
    for(unsigned group=owner;group<groups;group+=4U) {
        const auto* packed=row_weights+group*block_bytes;
        const auto* image=reinterpret_cast<const int*>(codes+slot*width+group*32U);
        int integer=0;
        #pragma unroll
        for(unsigned word=0;word<8U;++word) {
            if constexpr(Q51) integer=dot(image[word],down_q51_word(packed,word*4U),integer);
            else integer=__builtin_amdgcn_sudot4(true,image[word],true,
                static_cast<int>(down_u32(packed+2U+word*4U)),integer,false);
        }
        const float d=half_float(down_u16(packed));
        const float first=d*static_cast<float>(integer);
        float combined=first;
        if constexpr(Q51) {
            const float m=half_float(down_u16(packed+2U));
            const float correction=m*static_cast<float>(sums[slot*groups+group]);
            combined=first+correction;
        }
        const float contribution=deltas[slot*groups+group]*combined;
        acc=acc+contribution;
    }
    acc=acc+strix::qwen_dpp::xor32(acc, 1);
    acc=acc+strix::qwen_dpp::xor32(acc, 2);
    if(owner==0) rawdots[slot*hidden+row]=acc;
}
// Q2_0 down (q20_a8_down_numerical_identity): down_rawdots lanes and tree, group g = half g%2 of block
// g/2; contribution delta * (d * float(I - S)).
__global__ void down_rawdots_q20(const unsigned char* weights,const unsigned* routes,
    const signed char* codes,const float* deltas,const int* sums,float* rawdots,
    unsigned* status,unsigned* aggregate,unsigned hidden,unsigned width,unsigned experts) {
    if(*status!=0) return;
    const unsigned lane=threadIdx.x&31U,owner=lane&3U;
    const unsigned row=blockIdx.x*64U+(threadIdx.x/32U)*8U+lane/4U;
    const unsigned slot=blockIdx.y,expert=routes[slot],groups=width/32U;
    if(expert>=experts) {if(threadIdx.x==0) down_error(status,aggregate,down_invalid_route);return;}
    const auto* row_weights=weights+(static_cast<unsigned long long>(expert)*hidden+row)*(width/64U)*18U;
    float acc=0.0F;
    #pragma unroll 1
    for(unsigned group=owner;group<groups;group+=4U) {
        const float contribution=q20_group(row_weights+(group/2U)*18U,group&1U,
            reinterpret_cast<const int*>(codes+slot*width+group*32U),sums[slot*groups+group],deltas[slot*groups+group]);
        acc=acc+contribution;
    }
    acc=acc+strix::qwen_dpp::xor32(acc, 1);
    acc=acc+strix::qwen_dpp::xor32(acc, 2);
    if(owner==0) rawdots[slot*hidden+row]=acc;
}
// GSQ IQ4_NL down: down_rawdots lanes and tree, contribution delta * (d * float(I)).
template<std::uint32_t T>
__global__ void down_rawdots_iq(const unsigned char* weights,const unsigned* routes,
    const signed char* codes,const float* deltas,float* rawdots,
    unsigned* status,unsigned* aggregate,unsigned hidden,unsigned width,unsigned experts) {
    if(*status!=0) return;
    const unsigned lane=threadIdx.x&31U,owner=lane&3U;
    const unsigned row=blockIdx.x*64U+(threadIdx.x/32U)*8U+lane/4U;
    const unsigned slot=blockIdx.y,expert=routes[slot],groups=width/32U;
    if(expert>=experts) {if(threadIdx.x==0) down_error(status,aggregate,down_invalid_route);return;}
    const auto* row_weights=weights+(static_cast<unsigned long long>(expert)*hidden+row)*gsq_a8::row_bytes(T,width);
    float acc=0.0F;
    #pragma unroll 1
    for(unsigned group=owner;group<groups;group+=4U) {
        const auto x=gsq_a8::iq_group<T>(row_weights,group);
        const float contribution=gsq_a8::iq_apply<T>(x,reinterpret_cast<const int*>(codes+slot*width+group*32U),
            deltas[slot*groups+group]);
        acc=acc+contribution;
    }
    acc=acc+strix::qwen_dpp::xor32(acc, 1);
    acc=acc+strix::qwen_dpp::xor32(acc, 2);
    if(owner==0) rawdots[slot*hidden+row]=acc;
}
void down_admit(const DownRequest& request,const DownBindings& b,hipStream_t stream) {
    validate_down(request,b);
    if(!stream || stream==hipStreamLegacy || stream==hipStreamPerThread)
        throw std::invalid_argument("A8 down requires explicitly owned eager stream");
    const auto* context=current_hip_launch_context();
    if(context && (context->stream!=stream || context->graph_capture || context->status.arena_base ||
        context->arguments.memory.arena_base))
        throw std::invalid_argument("A8 down rejects graph or foreign launch context");
    hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
    hip_check(hipStreamIsCapturing(stream,&capture),"A8 down capture query");
    if(capture!=hipStreamCaptureStatusNone) throw std::invalid_argument("A8 down rejects capture before enqueue");
    hip_check(hipPeekAtLastError(),"A8 down pre-enqueue stale error");
}
}
void enqueue_down_activation_quantize(const DownRequest& r,const DownBindings& b,hipStream_t stream) {
    down_admit(r,b,stream);
    down_activation_quantize<<<dim3((r.shape.intermediate/32U+7U)/8U,r.shape.top_k),256,0,stream>>>(
        reinterpret_cast<float*>(b.intermediates.address),reinterpret_cast<const unsigned*>(b.routes.address),
        reinterpret_cast<signed char*>(b.codes.address),reinterpret_cast<float*>(b.deltas.address),
        reinterpret_cast<int*>(b.sums.address),reinterpret_cast<unsigned*>(b.status.address),
        reinterpret_cast<unsigned*>(b.aggregate_status.address),r.shape.intermediate,r.shape.experts);
    hip_check(hipGetLastError(),"A8 down activation/quantization enqueue");
}
void enqueue_down_rawdots(const DownRequest& r,const DownBindings& b,hipStream_t stream) {
    down_admit(r,b,stream);
    if(r.weight_type==gsq_a8::type_iq4_nl) {
        down_rawdots_iq<gsq_a8::type_iq4_nl><<<dim3(r.shape.hidden/64U,r.shape.top_k),256,0,stream>>>(
            reinterpret_cast<const unsigned char*>(b.weights.address),reinterpret_cast<const unsigned*>(b.routes.address),
            reinterpret_cast<const signed char*>(b.codes.address),reinterpret_cast<const float*>(b.deltas.address),
            reinterpret_cast<float*>(b.rawdots.address),
            reinterpret_cast<unsigned*>(b.status.address),reinterpret_cast<unsigned*>(b.aggregate_status.address),
            r.shape.hidden,r.shape.intermediate,r.shape.experts);
    } else if(r.weight_type==42) {
        down_rawdots_q20<<<dim3(r.shape.hidden/64U,r.shape.top_k),256,0,stream>>>(
            reinterpret_cast<const unsigned char*>(b.weights.address),reinterpret_cast<const unsigned*>(b.routes.address),
            reinterpret_cast<const signed char*>(b.codes.address),reinterpret_cast<const float*>(b.deltas.address),
            reinterpret_cast<const int*>(b.sums.address),reinterpret_cast<float*>(b.rawdots.address),
            reinterpret_cast<unsigned*>(b.status.address),reinterpret_cast<unsigned*>(b.aggregate_status.address),
            r.shape.hidden,r.shape.intermediate,r.shape.experts);
    } else if(r.weight_type==7) {
        down_rawdots<true><<<dim3(r.shape.hidden/64U,r.shape.top_k),256,0,stream>>>(
            reinterpret_cast<const unsigned char*>(b.weights.address),reinterpret_cast<const unsigned*>(b.routes.address),
            reinterpret_cast<const signed char*>(b.codes.address),reinterpret_cast<const float*>(b.deltas.address),
            reinterpret_cast<const int*>(b.sums.address),reinterpret_cast<float*>(b.rawdots.address),
            reinterpret_cast<unsigned*>(b.status.address),reinterpret_cast<unsigned*>(b.aggregate_status.address),
            r.shape.hidden,r.shape.intermediate,r.shape.experts);
    } else {
        down_rawdots<false><<<dim3(r.shape.hidden/64U,r.shape.top_k),256,0,stream>>>(
            reinterpret_cast<const unsigned char*>(b.weights.address),reinterpret_cast<const unsigned*>(b.routes.address),
            reinterpret_cast<const signed char*>(b.codes.address),reinterpret_cast<const float*>(b.deltas.address),
            reinterpret_cast<const int*>(b.sums.address),reinterpret_cast<float*>(b.rawdots.address),
            reinterpret_cast<unsigned*>(b.status.address),reinterpret_cast<unsigned*>(b.aggregate_status.address),
            r.shape.hidden,r.shape.intermediate,r.shape.experts);
    }
    hip_check(hipGetLastError(),"A8 down integer rawdot enqueue");
}

}
