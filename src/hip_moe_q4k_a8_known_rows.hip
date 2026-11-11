#include "strix/cohort_step_plan.hpp"
#include "strix/hip_qwen_dpp_reduce.hpp"
#include "strix/hip_arch_caps.hpp"
#include "moe_core/grouped_policy.hpp"
#include "strix/moe_route_bf16.hpp"
#include "strix/hip_moe_q4k_a8_known_rows.hpp"
#include "strix/moe_a8_down_contract.hpp"
#include "strix/qpf2k_defaults.hpp"
#include "hip_prefill_pool.hpp"
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "A8 CSR gate/up requires strict floating-point semantics"
#endif
#include <hip/hip_fp16.h>
#include "moe_core/grouped_f16.hpp"
#include <algorithm>
#include <array>
#include <utility>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#pragma clang fp contract(off)
#include "strix/hip_gsq_a8_groups.hpp"
#if defined(__HIP_DEVICE_COMPILE__)
#if !STRIX_HAS_RDNA35_WMMA || !STRIX_HAS_SUDOT4
#error "A8 CSR gate/up requires RDNA 3.5 packed WMMA and signed-A unsigned-B dot4"
#endif
#endif

namespace strix::moe_q4k_a8::known_rows {
namespace {
HipPrefillPoolOwner& prefill_pool_owner(){static HipPrefillPoolOwner owner;return owner;}
unsigned prefill_pool_rows(){return prefill_pool_owner().rows();}
struct DeviceRows {
    const signed char* codes[max_union_rows]{}; const float* deltas[max_union_rows]{}; const int* sums[max_union_rows]{};
    const unsigned* routes[max_union_rows]{}; unsigned* status[max_union_rows]{}; float* output[max_union_rows]{};
    signed char* down_codes[max_union_rows]{}; float* down_deltas[max_union_rows]{}; int* down_sums[max_union_rows]{};
    float* rawdots[max_union_rows]{};
};
struct DeviceCsr {
    unsigned* summary; unsigned* offsets; unsigned* touched;
    unsigned* rows; unsigned* slots; unsigned* experts; unsigned* begins;
};
void check(hipError_t error,const char* operation) {
    if(error!=hipSuccess)throw std::runtime_error(std::string(operation)+": "+hipGetErrorString(error));
}
__device__ int dot(int a,unsigned b,int acc) {
    return __builtin_amdgcn_sudot4(true,a,false,static_cast<int>(b),acc,false);
}
__device__ unsigned factor(uint4 header,unsigned group,bool minimum) {
    const unsigned shift=8U*(group&3U);
    const unsigned a=(header.y>>shift)&255U,b=(header.z>>shift)&255U,c=(header.w>>shift)&255U;
    return group<4U?(minimum?b:a)&63U:minimum?(c>>4U)|((b>>6U)<<4U):(c&15U)|((a>>6U)<<4U);
}
struct Coefficient {
    uint4 low,high;
    unsigned scale0,scale1,minimum0,minimum1;
    float d,dm;
};
__device__ Coefficient coefficient(uint4 hd,uint4 q,unsigned p) {
    constexpr unsigned mask=0x0f0f0f0fU;
    return {{q.x&mask,q.y&mask,q.z&mask,q.w&mask},
        {(q.x>>4U)&mask,(q.y>>4U)&mask,(q.z>>4U)&mask,(q.w>>4U)&mask},
        factor(hd,2U*p,false),factor(hd,2U*p+1U,false),factor(hd,2U*p,true),factor(hd,2U*p+1U,true),
        __half2float(__ushort_as_half(static_cast<unsigned short>(hd.x&65535U))),
        __half2float(__ushort_as_half(static_cast<unsigned short>(hd.x>>16U)))};
}
__device__ Coefficient coefficient(const unsigned char* packed,unsigned c,unsigned p) {
    return coefficient(*reinterpret_cast<const uint4*>(packed),*reinterpret_cast<const uint4*>(packed+16U+c*16U),p);
}
__device__ __forceinline__ float paired(Coefficient q,int4 a0,int4 a1,int s0,int s1,float e0,float e1) {
    int r0=dot(a0.x,q.low.x,0);r0=dot(a0.y,q.low.y,r0);r0=dot(a0.z,q.low.z,r0);r0=dot(a0.w,q.low.w,r0);
    int r1=dot(a1.x,q.high.x,0);r1=dot(a1.y,q.high.y,r1);r1=dot(a1.z,q.high.z,r1);r1=dot(a1.w,q.high.w,r1);
    r0+=strix::qwen_dpp::xor32(r0, 1);r1+=strix::qwen_dpp::xor32(r1, 1);
    const int i0=static_cast<int>(q.scale0)*r0,i1=static_cast<int>(q.scale1)*r1;
    const int j0=static_cast<int>(q.minimum0)*s0,j1=static_cast<int>(q.minimum1)*s1;
    const float u0=q.d*static_cast<float>(i0),v0=q.dm*static_cast<float>(j0),t0=u0-v0;
    const float u1=q.d*static_cast<float>(i1),v1=q.dm*static_cast<float>(j1),t1=u1-v1;
    const float c0=e0*t0,c1=e1*t1;
    return c0+c1;
}

// Row status (and prefill summary) bit set by the gate/up projection itself for a non-finite
// (row, slot). The projection kernels' entry checks ignore it: another CTA of the same launch may
// already have set it, and skipping on it would make the written bytes depend on CTA order.
constexpr unsigned projection_nonfinite=8U;
// The public component does not trust caller-provided device CSR contents.
// Admit the complete <=80 assignment permutation before any projection CTA.
__global__ void admit_csr(DeviceRows view,DeviceCsr csr,unsigned rows,unsigned experts,unsigned top_k) {
    if(threadIdx.x) return;
    unsigned failure=csr.summary[0];
    for(unsigned r=0;r<rows;++r)failure|=*view.status[r];
    if(failure){csr.summary[0]=failure;return;}
    const unsigned count=rows*top_k,batches=csr.summary[2],touched=csr.summary[1];
    bool bad=csr.summary[3]!=count||csr.summary[6]!=rows||csr.summary[7]!=1U||
        !batches||batches>count||!touched||touched>experts||touched>count;
    if(!bad) {
        bad=csr.offsets[0]!=0U||csr.offsets[experts]!=count||csr.begins[0]!=0U||csr.begins[batches]!=count;
        for(unsigned e=0;e<experts;++e)bad|=csr.offsets[e]>csr.offsets[e+1U]||csr.offsets[e+1U]>count;
        for(unsigned t=0;t<touched;++t) {
            const auto e=csr.touched[t];
            bad|=e>=experts||(t&&e<=csr.touched[t-1U]);
            if(e<experts)bad|=csr.offsets[e]==csr.offsets[e+1U];
        }
    }
    bool seen[80]={};
    if(!bad) for(unsigned b=0;b<batches;++b) {
        const auto begin=csr.begins[b],end=csr.begins[b+1U],e=csr.experts[b];
        if(e>=experts||begin>=end||end>count||end-begin>rows||(b&&e<csr.experts[b-1U])){bad=true;break;}
        if(begin<csr.offsets[e]||end>csr.offsets[e+1U]){bad=true;break;}
        for(unsigned a=begin;a<end;++a) {
            const auto r=csr.rows[a],slot=csr.slots[a];
            if(r>=rows||slot>=top_k){bad=true;break;}
            const auto key=r*top_k+slot;
            if(seen[key]||view.routes[r][slot]!=e){bad=true;break;}
            seen[key]=true;
        }
        if(bad)break;
    }
    if(!bad)for(unsigned a=0;a<count;++a)bad|=!seen[a];
    if(bad) {
        atomicOr(csr.summary,invalid_route);
        for(unsigned r=0;r<rows;++r)atomicOr(view.status[r],invalid_route);
    }
}

template<unsigned Rows>
__global__ void project_csr(const unsigned char* gate,const unsigned char* up,DeviceRows view,DeviceCsr csr,
    unsigned hidden,unsigned outputs) {
    if(csr.summary[0]||blockIdx.y>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.y],end=csr.begins[blockIdx.y+1U],expert=csr.experts[blockIdx.y];
    __shared__ __align__(16) signed char local_codes[Rows][max_hidden];
    __shared__ __align__(16) float local_deltas[Rows][max_hidden/32U];
    __shared__ __align__(16) int local_sums[Rows][max_hidden/32U];
#pragma unroll
    for(unsigned a=0;a<Rows;++a)if(begin+a<end) {
        const auto r=csr.rows[begin+a];
        for(unsigned i=threadIdx.x;i<hidden;i+=256U)local_codes[a][i]=view.codes[r][i];
        for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U){local_deltas[a][i]=view.deltas[r][i];local_sums[a][i]=view.sums[r][i];}
    }
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,c=lane&7U,p=c/2U,h=c&1U;
    const unsigned out=blockIdx.x*32U+(threadIdx.x/32U)*4U+lane/8U,blocks=hidden/256U;
    const auto offset=(static_cast<unsigned long long>(expert)*outputs+out)*blocks*144U;
    const unsigned g0=2U*p,g1=g0+1U;
    float acc_gate[Rows]={},acc_up[Rows]={};
#pragma unroll 1
    for(unsigned block=0;block<blocks;++block) {
        // Packed bytes, factors and half headers are decoded once per tile.
        const auto gate_coefficient=coefficient(gate+offset+block*144U,c,p);
        const auto up_coefficient=coefficient(up+offset+block*144U,c,p);
#pragma unroll
        for(unsigned a=0;a<Rows;++a)if(begin+a<end) {
            const int4 a0=*reinterpret_cast<const int4*>(local_codes[a]+block*256U+p*64U+h*16U);
            const int4 a1=*reinterpret_cast<const int4*>(local_codes[a]+block*256U+p*64U+32U+h*16U);
            const int s0=local_sums[a][block*8U+g0],s1=local_sums[a][block*8U+g1];
            const float e0=local_deltas[a][block*8U+g0],e1=local_deltas[a][block*8U+g1];
            const float gate_pair=paired(gate_coefficient,a0,a1,s0,s1,e0,e1);
            acc_gate[a]=acc_gate[a]+gate_pair;
            const float up_pair=paired(up_coefficient,a0,a1,s0,s1,e0,e1);
            acc_up[a]=acc_up[a]+up_pair;
        }
    }
#pragma unroll
    for(unsigned a=0;a<Rows;++a)if(begin+a<end) {
        acc_gate[a]=acc_gate[a]+strix::qwen_dpp::xor32(acc_gate[a], 2);acc_gate[a]=acc_gate[a]+strix::qwen_dpp::xor32(acc_gate[a], 4);
        acc_up[a]=acc_up[a]+strix::qwen_dpp::xor32(acc_up[a], 2);acc_up[a]=acc_up[a]+strix::qwen_dpp::xor32(acc_up[a], 4);
        if(c==0U) {
            const auto r=csr.rows[begin+a],slot=csr.slots[begin+a];
            view.output[r][(slot*2U)*outputs+out]=acc_gate[a];view.output[r][(slot*2U+1U)*outputs+out]=acc_up[a];
        }
    }
}
__device__ bool nonfinite(float x){return (__float_as_uint(x)&0x7fffffffU)>=0x7f800000U;}
// Serial Q5_K A8G32 pair (project_q5k): unsigned five-bit codes from qs/qh,
// then the project/project_csr affine and pair order.
__device__ __forceinline__ float paired_q5k(uint4 hd,uint4 q,uint4 qh,unsigned p,int4 a0,int4 a1,int s0,int s1,float e0,float e1) {
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
    const int j0=static_cast<int>(factor(hd,g0,true))*s0;
    const int j1=static_cast<int>(factor(hd,g1,true))*s1;
    const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(hd.x&65535U)));
    const float dm=__half2float(__ushort_as_half(static_cast<unsigned short>(hd.x>>16U)));
    const float u0=d*static_cast<float>(i0),v0=dm*static_cast<float>(j0),t0=u0-v0;
    const float u1=d*static_cast<float>(i1),v1=dm*static_cast<float>(j1),t1=u1-v1;
    const float c0=e0*t0,c1=e1*t1;
    return c0+c1;
}
__device__ float divide_rne(float a,float b){return a/b;}
// project_csr arithmetic (Q4_K, or the serial Q5_K pair) with every K block of
// a Q4_K matrix loaded before its chains (each wave fetches four complete
// packed rows at once). Batch rows are staged G at a time; a batch wider than
// G re-reads its weight slice from cache per group. Epilogue per
// (row,slot,output): the float profile's finite scan/SiLU/scan, or (A8Down)
// the serial A8 down activation quantizer on the block's 32-output group.
// Chunk: Q4_K K blocks held in registers at once (fewer lets more waves stay resident;
// each accumulator still adds its blocks in ascending order).
template<unsigned G,bool Q5K,bool A8Down,unsigned Chunk=max_hidden/256U>
__global__ __launch_bounds__(256) void project_window(const unsigned char* gate,const unsigned char* up,DeviceRows view,DeviceCsr csr,
    unsigned rows,unsigned hidden,unsigned outputs) {
    unsigned failure=csr.summary[0];
    for(unsigned r=0;r<rows;++r)failure|=*view.status[r];
    if((failure&~projection_nonfinite)||blockIdx.y>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.y],count=csr.begins[blockIdx.y+1U]-begin,expert=csr.experts[blockIdx.y];
    __shared__ __align__(16) signed char local_codes[G][max_hidden];
    __shared__ __align__(16) float local_deltas[G][max_hidden/32U];
    __shared__ __align__(16) int local_sums[G][max_hidden/32U];
    __shared__ float tile[2][G][32];
    const unsigned lane=threadIdx.x&31U,c=lane&7U,p=c/2U,h=c&1U,wave=threadIdx.x/32U;
    const unsigned out=blockIdx.x*32U+wave*4U+lane/8U,blocks=hidden/256U;
    constexpr unsigned block_bytes=Q5K?176U:144U;
    const auto offset=(static_cast<unsigned long long>(expert)*outputs+out)*blocks*block_bytes;
    const unsigned g0=2U*p,g1=g0+1U;
    constexpr unsigned max_blocks=max_hidden/256U;
    for(unsigned group=0;group<count;group+=G) {
        if(group)__syncthreads();
#pragma unroll
        for(unsigned a=0;a<G;++a)if(group+a<count) {
            const auto r=csr.rows[begin+group+a];
            for(unsigned i=16U*threadIdx.x;i<hidden;i+=4096U)
                *reinterpret_cast<int4*>(local_codes[a]+i)=*reinterpret_cast<const int4*>(view.codes[r]+i);
            for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U){local_deltas[a][i]=view.deltas[r][i];local_sums[a][i]=view.sums[r][i];}
        }
        __syncthreads();
        float acc[2][G]={};
#pragma unroll 1
        for(unsigned m=0;m<2U;++m) {
            const unsigned char* w=(m==0U?gate:up)+offset;
            if constexpr(Q5K) {
#pragma unroll 1
                for(unsigned b=0;b<blocks;++b) {
                    const auto* packed=w+b*176U;
                    const uint4 hd=*reinterpret_cast<const uint4*>(packed);
                    const uint4 q=*reinterpret_cast<const uint4*>(packed+48U+c*16U);
                    const uint4 qh=*reinterpret_cast<const uint4*>(packed+16U+h*16U);
#pragma unroll
                    for(unsigned a=0;a<G;++a)if(group+a<count) {
                        const int4 a0=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+h*16U);
                        const int4 a1=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+32U+h*16U);
                        const float pair=paired_q5k(hd,q,qh,p,a0,a1,local_sums[a][b*8U+g0],local_sums[a][b*8U+g1],
                            local_deltas[a][b*8U+g0],local_deltas[a][b*8U+g1]);
                        acc[m][a]=acc[m][a]+pair;
                    }
                }
            } else {
#pragma unroll
                for(unsigned first=0;first<max_blocks;first+=Chunk) {
                uint4 hd[Chunk],q[Chunk];
#pragma unroll
                for(unsigned i=0;i<Chunk;++i)if(first+i<blocks) {
                    const unsigned b=first+i;
                    hd[i]=*reinterpret_cast<const uint4*>(w+b*144U);q[i]=*reinterpret_cast<const uint4*>(w+b*144U+16U+c*16U);
                }
#pragma unroll
                for(unsigned i=0;i<Chunk;++i)if(first+i<blocks) {
                    const unsigned b=first+i;
                    const auto k=coefficient(hd[i],q[i],p);
#pragma unroll
                    for(unsigned a=0;a<G;++a)if(group+a<count) {
                        const int4 a0=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+h*16U);
                        const int4 a1=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+32U+h*16U);
                        const float pair=paired(k,a0,a1,local_sums[a][b*8U+g0],local_sums[a][b*8U+g1],
                            local_deltas[a][b*8U+g0],local_deltas[a][b*8U+g1]);
                        acc[m][a]=acc[m][a]+pair;
                    }
                }
                }
            }
        }
#pragma unroll
        for(unsigned a=0;a<G;++a)if(group+a<count) {
#pragma unroll
            for(unsigned m=0;m<2U;++m){acc[m][a]=acc[m][a]+strix::qwen_dpp::xor32(acc[m][a], 2);acc[m][a]=acc[m][a]+strix::qwen_dpp::xor32(acc[m][a], 4);}
            if constexpr(A8Down) {
                if(c==0U){tile[0][a][out%32U]=acc[0][a];tile[1][a][out%32U]=acc[1][a];}
            } else if(c==0U) {
                const auto r=csr.rows[begin+group+a],slot=csr.slots[begin+group+a];
                const float g=acc[0][a],u=acc[1][a],silu=(g/(1.0F+expf(-g)))*u;
                view.output[r][(slot*2U)*outputs+out]=silu;view.output[r][(slot*2U+1U)*outputs+out]=u;
                if(nonfinite(g)||nonfinite(u)||nonfinite(silu))atomicOr(view.status[r],projection_nonfinite);
            }
        }
        if constexpr(A8Down) {
            // Serial down_activation_quantize for group blockIdx.x of each staged
            // (row,slot): same lane = feature, collectives and fail-closed writes.
            // One wave per staged (row, slot): rows are independent, each keeps the serial quantizer.
            __syncthreads();
            {
                for(unsigned a=wave;a<G;a+=8U)if(group+a<count) {
                    const auto r=csr.rows[begin+group+a],slot=csr.slots[begin+group+a];
                    const unsigned feature=blockIdx.x*32U+lane,at=slot*2U*outputs+feature;
                    const float g=tile[0][a][lane],u=tile[1][a][lane];
                    const float hv=(g/(1.0F+expf(-g)))*u;
                    view.output[r][at+outputs]=u;
                    const unsigned bad=static_cast<unsigned>(__ballot(!isfinite(g)||!isfinite(u)||!isfinite(hv)));
                    if(bad!=0){view.output[r][at]=g;if(lane==0)atomicOr(view.status[r],projection_nonfinite);continue;}
                    float x=fabsf(hv);
                    for(int stride=16;stride!=0;stride/=2) x=fmaxf(x,strix::qwen_dpp::xor32(x, stride));
                    const bool tiny=x>0.0F && x<0x1p-119F;
                    const float delta=x==0.0F || tiny?1.0F:divide_rne(x,127.0F);
                    const float rounded=tiny?0.0F:rintf(divide_rne(hv,delta));
                    const int code=static_cast<int>(fminf(127.0F,fmaxf(-127.0F,rounded)));
                    int sum=code;
                    for(int stride=16;stride!=0;stride/=2) sum+=strix::qwen_dpp::xor32(sum, stride);
                    view.output[r][at]=hv;
                    view.down_codes[r][slot*outputs+feature]=static_cast<signed char>(code);
                    if(lane==0){view.down_deltas[r][slot*(outputs/32U)+blockIdx.x]=delta;view.down_sums[r][slot*(outputs/32U)+blockIdx.x]=sum;}
                }
            }
        }
    }
}
__device__ unsigned down_u16(const unsigned char* p) {
    return static_cast<unsigned>(p[0])|(static_cast<unsigned>(p[1])<<8U);
}
__device__ unsigned down_u32(const unsigned char* p) {return down_u16(p)|(down_u16(p+2U)<<16U);}
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
// Opt-in R16 projection: G-row activation staging, one weight load per expert batch.
template<unsigned G,bool Q5K,bool A8Down>
__global__ __launch_bounds__(256) void project_window_split(const unsigned char* gate,const unsigned char* up,DeviceRows view,DeviceCsr csr,
    unsigned rows,unsigned hidden,unsigned outputs) {
    unsigned failure=csr.summary[0];
    for(unsigned r=0;r<rows;++r)failure|=*view.status[r];
    if((failure&~projection_nonfinite)||blockIdx.y>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.y],count=csr.begins[blockIdx.y+1U]-begin,expert=csr.experts[blockIdx.y];
    __shared__ __align__(16) signed char local_codes[G][max_hidden];
    __shared__ __align__(16) float local_deltas[G][max_hidden/32U];
    __shared__ __align__(16) int local_sums[G][max_hidden/32U];
    __shared__ float tile[2][16][32];
    const unsigned lane=threadIdx.x&31U,c=lane&7U,p=c/2U,h=c&1U,wave=threadIdx.x/32U;
    const unsigned out=blockIdx.x*32U+wave*4U+lane/8U,blocks=hidden/256U;
    constexpr unsigned block_bytes=Q5K?176U:144U;
    const auto offset=(static_cast<unsigned long long>(expert)*outputs+out)*blocks*block_bytes;
    const unsigned g0=2U*p,g1=g0+1U;
    constexpr unsigned max_blocks=max_hidden/256U;
    static_assert(G==4U || G==8U);
#pragma unroll 1
    for(unsigned m=0;m<2U;++m) {
        const unsigned char* w=(m==0U?gate:up)+offset;
        uint4 staged_weights[Q5K?3U:2U][max_blocks];
#pragma unroll
        for(unsigned b=0;b<max_blocks;++b)if(b<blocks) {
            const auto* packed=w+b*block_bytes;
            staged_weights[0][b]=*reinterpret_cast<const uint4*>(packed);
            staged_weights[1][b]=*reinterpret_cast<const uint4*>(packed+(Q5K?48U:16U)+c*16U);
            if constexpr(Q5K)staged_weights[2][b]=*reinterpret_cast<const uint4*>(packed+16U+h*16U);
        }
        for(unsigned group=0;group<count;group+=G) {
            __syncthreads();
#pragma unroll
        for(unsigned a=0;a<G;++a)if(group+a<count) {
            const auto r=csr.rows[begin+group+a];
            for(unsigned i=16U*threadIdx.x;i<hidden;i+=4096U)
                *reinterpret_cast<int4*>(local_codes[a]+i)=*reinterpret_cast<const int4*>(view.codes[r]+i);
            for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U){local_deltas[a][i]=view.deltas[r][i];local_sums[a][i]=view.sums[r][i];}
        }
        __syncthreads();
        float acc[G]={};
#pragma unroll
        for(unsigned b=0;b<max_blocks;++b)if(b<blocks) {
#pragma unroll
            for(unsigned a=0;a<G;++a)if(group+a<count) {
                const int4 a0=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+h*16U);
                const int4 a1=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+32U+h*16U);
                float pair;
                if constexpr(Q5K)pair=paired_q5k(staged_weights[0][b],staged_weights[1][b],staged_weights[2][b],p,a0,a1,
                    local_sums[a][b*8U+g0],local_sums[a][b*8U+g1],local_deltas[a][b*8U+g0],local_deltas[a][b*8U+g1]);
                else pair=paired(coefficient(staged_weights[0][b],staged_weights[1][b],p),a0,a1,
                    local_sums[a][b*8U+g0],local_sums[a][b*8U+g1],local_deltas[a][b*8U+g0],local_deltas[a][b*8U+g1]);
                acc[a]=acc[a]+pair;
            }
        }
#pragma unroll
        for(unsigned a=0;a<G;++a)if(group+a<count) {
            acc[a]=acc[a]+__shfl_xor(acc[a],2,32);
            acc[a]=acc[a]+__shfl_xor(acc[a],4,32);
            if(c==0U)tile[m][group+a][out%32U]=acc[a];
        }
        }
    }
    __syncthreads();
    for(unsigned group=0;group<count;group+=G) {
        if constexpr(!A8Down) {
#pragma unroll
            for(unsigned a=0;a<G;++a)if(group+a<count && c==0U) {
                const auto r=csr.rows[begin+group+a],slot=csr.slots[begin+group+a];
                const float g=tile[0][group+a][out%32U],u=tile[1][group+a][out%32U];
                const float silu=(g/(1.0F+expf(-g)))*u;
                view.output[r][(slot*2U)*outputs+out]=silu;view.output[r][(slot*2U+1U)*outputs+out]=u;
                if(nonfinite(g)||nonfinite(u)||nonfinite(silu))atomicOr(view.status[r],projection_nonfinite);
            }
        }
        if constexpr(A8Down) {
            // Serial down_activation_quantize for group blockIdx.x of each staged
            // (row,slot): same lane = feature, collectives and fail-closed writes.
            // One wave per staged (row, slot): rows are independent, each keeps the serial quantizer.
            __syncthreads();
            {
                for(unsigned a=wave;a<G;a+=8U)if(group+a<count) {
                    const auto r=csr.rows[begin+group+a],slot=csr.slots[begin+group+a];
                    const unsigned feature=blockIdx.x*32U+lane,at=slot*2U*outputs+feature;
                    const float g=tile[0][group+a][lane],u=tile[1][group+a][lane];
                    const float hv=(g/(1.0F+expf(-g)))*u;
                    view.output[r][at+outputs]=u;
                    const unsigned bad=static_cast<unsigned>(__ballot(!isfinite(g)||!isfinite(u)||!isfinite(hv)));
                    if(bad!=0){view.output[r][at]=g;if(lane==0)atomicOr(view.status[r],projection_nonfinite);continue;}
                    float x=fabsf(hv);
                    for(int stride=16;stride!=0;stride/=2) x=fmaxf(x,__shfl_xor(x,stride,32));
                    const bool tiny=x>0.0F && x<0x1p-119F;
                    const float delta=x==0.0F || tiny?1.0F:divide_rne(x,127.0F);
                    const float rounded=tiny?0.0F:rintf(divide_rne(hv,delta));
                    const int code=static_cast<int>(fminf(127.0F,fmaxf(-127.0F,rounded)));
                    int sum=code;
                    for(int stride=16;stride!=0;stride/=2) sum+=__shfl_xor(sum,stride,32);
                    view.output[r][at]=hv;
                    view.down_codes[r][slot*outputs+feature]=static_cast<signed char>(code);
                    if(lane==0){view.down_deltas[r][slot*(outputs/32U)+blockIdx.x]=delta;view.down_sums[r][slot*(outputs/32U)+blockIdx.x]=sum;}
                }
            }
        }
    }
}

// Serial down_rawdots (Q5_1 s8xu5 / Q8_0 s8xs8 A8G32) per CSR batch: each
// lane decodes its block words once and runs every staged row's chain with
// the serial lane/group/tree order. Writes the incumbent raw slot dots. A
// failing block (0,0) folds all row statuses into the summary.
template<unsigned Rows,bool Q51>
__global__ __launch_bounds__(256) void down_window(const unsigned char* weights,DeviceRows view,DeviceCsr csr,
    unsigned rows,unsigned hidden,unsigned width) {
    unsigned failure=csr.summary[0];
    for(unsigned r=0;r<rows;++r)failure|=*view.status[r];
    if(failure){if(blockIdx.x==0U&&blockIdx.y==0U&&threadIdx.x==0U)csr.summary[0]=failure;return;}
    if(blockIdx.y>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.y],count=csr.begins[blockIdx.y+1U]-begin,expert=csr.experts[blockIdx.y];
    __shared__ __align__(16) signed char codes[Rows][max_rows];
    __shared__ float deltas[Rows][max_rows/32U];
    __shared__ int sums[Rows][max_rows/32U];
    const unsigned groups=width/32U;
#pragma unroll
    for(unsigned a=0;a<Rows;++a)if(a<count) {
        const auto r=csr.rows[begin+a],slot=csr.slots[begin+a];
        for(unsigned i=16U*threadIdx.x;i<width;i+=4096U)
            *reinterpret_cast<int4*>(codes[a]+i)=*reinterpret_cast<const int4*>(view.down_codes[r]+slot*width+i);
        for(unsigned i=threadIdx.x;i<groups;i+=256U){deltas[a][i]=view.down_deltas[r][slot*groups+i];sums[a][i]=view.down_sums[r][slot*groups+i];}
    }
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,owner=lane&3U;
    const unsigned row=blockIdx.x*64U+(threadIdx.x/32U)*8U+lane/4U;
    constexpr unsigned block_bytes=Q51?24U:34U;
    const auto* row_weights=weights+(static_cast<unsigned long long>(expert)*hidden+row)*groups*block_bytes;
    float acc[Rows]={};
#pragma unroll 1
    for(unsigned group=owner;group<groups;group+=4U) {
        const auto* packed=row_weights+group*block_bytes;
        unsigned words[8];
#pragma unroll
        for(unsigned word=0;word<8U;++word)words[word]=Q51?down_q51_word(packed,word*4U):down_u32(packed+2U+word*4U);
        const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(down_u16(packed))));
        const float m=Q51?__half2float(__ushort_as_half(static_cast<unsigned short>(down_u16(packed+2U)))):0.0F;
#pragma unroll
        for(unsigned a=0;a<Rows;++a)if(a<count) {
            const auto* image=reinterpret_cast<const int*>(codes[a]+group*32U);
            int integer=0;
#pragma unroll
            for(unsigned word=0;word<8U;++word) {
                if constexpr(Q51) integer=dot(image[word],words[word],integer);
                else integer=__builtin_amdgcn_sudot4(true,image[word],true,static_cast<int>(words[word]),integer,false);
            }
            const float first=d*static_cast<float>(integer);
            float combined=first;
            if constexpr(Q51) {
                const float correction=m*static_cast<float>(sums[a][group]);
                combined=first+correction;
            }
            const float contribution=deltas[a][group]*combined;
            acc[a]=acc[a]+contribution;
        }
    }
#pragma unroll
    for(unsigned a=0;a<Rows;++a)if(a<count) {
        acc[a]=acc[a]+strix::qwen_dpp::xor32(acc[a], 1);
        acc[a]=acc[a]+strix::qwen_dpp::xor32(acc[a], 2);
        if(owner==0U)view.rawdots[csr.rows[begin+a]][csr.slots[begin+a]*hidden+row]=acc[a];
    }
}
// GSQ-RCO Q2_0 window forms: project_q20 / down_rawdots_q20 lane arithmetic per (row, slot), shared
// weight loads per CSR batch, then project_window's A8 down quantizer epilogue (identical bytes).
__device__ __forceinline__ unsigned q20_word(unsigned byte) {
    return (byte|(byte<<6U)|(byte<<12U)|(byte<<18U))&0x03030303U;
}
__device__ __forceinline__ unsigned q20_u16(const unsigned char* p) {
    return static_cast<unsigned>(p[0])|(static_cast<unsigned>(p[1])<<8U);
}
struct Q20Group { unsigned w[8]; float d; };
__device__ __forceinline__ Q20Group q20_load(const unsigned char* block,unsigned half) {
    Q20Group x;
    const auto* codes=block+2U+half*8U;
#pragma unroll
    for(unsigned i=0;i<4U;++i){const unsigned v=q20_u16(codes+2U*i);x.w[2U*i]=q20_word(v&255U);x.w[2U*i+1U]=q20_word(v>>8U);}
    x.d=__half2float(__ushort_as_half(static_cast<unsigned short>(q20_u16(block))));
    return x;
}
__device__ __forceinline__ float q20_apply(const Q20Group& x,const int* a,int sum,float delta) {
    int integer=dot(a[0],x.w[0],0);
#pragma unroll
    for(unsigned i=1;i<8U;++i)integer=dot(a[i],x.w[i],integer);
    const float scaled=x.d*static_cast<float>(integer-sum);
    return delta*scaled;
}
template<unsigned G>
__global__ __launch_bounds__(256) void project_window_q20(const unsigned char* gate,const unsigned char* up,DeviceRows view,DeviceCsr csr,
    unsigned rows,unsigned hidden,unsigned outputs) {
    unsigned failure=csr.summary[0];
    for(unsigned r=0;r<rows;++r)failure|=*view.status[r];
    if((failure&~projection_nonfinite)||blockIdx.y>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.y],count=csr.begins[blockIdx.y+1U]-begin,expert=csr.experts[blockIdx.y];
    __shared__ __align__(16) signed char local_codes[G][max_hidden];
    __shared__ __align__(16) float local_deltas[G][max_hidden/32U];
    __shared__ __align__(16) int local_sums[G][max_hidden/32U];
    __shared__ float tile[2][G][32];
    const unsigned lane=threadIdx.x&31U,c=lane&7U,wave=threadIdx.x/32U;
    const unsigned out=blockIdx.x*32U+wave*4U+lane/8U,groups=hidden/32U;
    const auto offset=(static_cast<unsigned long long>(expert)*outputs+out)*(hidden/64U)*18U;
    for(unsigned group=0;group<count;group+=G) {
        if(group)__syncthreads();
#pragma unroll
        for(unsigned a=0;a<G;++a)if(group+a<count) {
            const auto r=csr.rows[begin+group+a];
            for(unsigned i=16U*threadIdx.x;i<hidden;i+=4096U)
                *reinterpret_cast<int4*>(local_codes[a]+i)=*reinterpret_cast<const int4*>(view.codes[r]+i);
            for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U){local_deltas[a][i]=view.deltas[r][i];local_sums[a][i]=view.sums[r][i];}
        }
        __syncthreads();
        float acc[2][G]={};
#pragma unroll
        for(unsigned m=0;m<2U;++m) {
            const unsigned char* w=(m==0U?gate:up)+offset;
#pragma unroll 1
            for(unsigned g=c;g<groups;g+=8U) {
                const auto x=q20_load(w+(g/2U)*18U,g&1U);
#pragma unroll
                for(unsigned a=0;a<G;++a)if(group+a<count) {
                    const float value=q20_apply(x,reinterpret_cast<const int*>(local_codes[a]+g*32U),local_sums[a][g],local_deltas[a][g]);
                    acc[m][a]=acc[m][a]+value;
                }
            }
        }
#pragma unroll
        for(unsigned a=0;a<G;++a)if(group+a<count) {
#pragma unroll
            for(unsigned m=0;m<2U;++m){acc[m][a]=acc[m][a]+strix::qwen_dpp::xor32(acc[m][a], 1);acc[m][a]=acc[m][a]+strix::qwen_dpp::xor32(acc[m][a], 2);
                acc[m][a]=acc[m][a]+strix::qwen_dpp::xor32(acc[m][a], 4);}
            if(c==0U){tile[0][a][out%32U]=acc[0][a];tile[1][a][out%32U]=acc[1][a];}
        }
        __syncthreads();
        for(unsigned a=wave;a<G;a+=8U)if(group+a<count) {
            const auto r=csr.rows[begin+group+a],slot=csr.slots[begin+group+a];
            const unsigned feature=blockIdx.x*32U+lane,at=slot*2U*outputs+feature;
            const float g=tile[0][a][lane],u=tile[1][a][lane];
            const float hv=(g/(1.0F+expf(-g)))*u;
            view.output[r][at+outputs]=u;
            const unsigned bad=static_cast<unsigned>(__ballot(!isfinite(g)||!isfinite(u)||!isfinite(hv)));
            if(bad!=0){view.output[r][at]=g;if(lane==0)atomicOr(view.status[r],projection_nonfinite);continue;}
            float x=fabsf(hv);
            for(int stride=16;stride!=0;stride/=2) x=fmaxf(x,strix::qwen_dpp::xor32(x, stride));
            const bool tiny=x>0.0F && x<0x1p-119F;
            const float delta=x==0.0F || tiny?1.0F:divide_rne(x,127.0F);
            const float rounded=tiny?0.0F:rintf(divide_rne(hv,delta));
            const int code=static_cast<int>(fminf(127.0F,fmaxf(-127.0F,rounded)));
            int sum=code;
            for(int stride=16;stride!=0;stride/=2) sum+=strix::qwen_dpp::xor32(sum, stride);
            view.output[r][at]=hv;
            view.down_codes[r][slot*outputs+feature]=static_cast<signed char>(code);
            if(lane==0){view.down_deltas[r][slot*(outputs/32U)+blockIdx.x]=delta;view.down_sums[r][slot*(outputs/32U)+blockIdx.x]=sum;}
        }
    }
}
template<unsigned Rows>
__global__ __launch_bounds__(256) void down_window_q20(const unsigned char* weights,DeviceRows view,DeviceCsr csr,
    unsigned rows,unsigned hidden,unsigned width) {
    unsigned failure=csr.summary[0];
    for(unsigned r=0;r<rows;++r)failure|=*view.status[r];
    if(failure){if(blockIdx.x==0U&&blockIdx.y==0U&&threadIdx.x==0U)csr.summary[0]=failure;return;}
    if(blockIdx.y>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.y],count=csr.begins[blockIdx.y+1U]-begin,expert=csr.experts[blockIdx.y];
    __shared__ __align__(16) signed char codes[Rows][max_rows];
    __shared__ float deltas[Rows][max_rows/32U];
    __shared__ int sums[Rows][max_rows/32U];
    const unsigned groups=width/32U;
#pragma unroll
    for(unsigned a=0;a<Rows;++a)if(a<count) {
        const auto r=csr.rows[begin+a],slot=csr.slots[begin+a];
        for(unsigned i=16U*threadIdx.x;i<width;i+=4096U)
            *reinterpret_cast<int4*>(codes[a]+i)=*reinterpret_cast<const int4*>(view.down_codes[r]+slot*width+i);
        for(unsigned i=threadIdx.x;i<groups;i+=256U){deltas[a][i]=view.down_deltas[r][slot*groups+i];sums[a][i]=view.down_sums[r][slot*groups+i];}
    }
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,owner=lane&3U;
    const unsigned row=blockIdx.x*64U+(threadIdx.x/32U)*8U+lane/4U;
    const auto* row_weights=weights+(static_cast<unsigned long long>(expert)*hidden+row)*(width/64U)*18U;
    float acc[Rows]={};
#pragma unroll 1
    for(unsigned group=owner;group<groups;group+=4U) {
        const auto x=q20_load(row_weights+(group/2U)*18U,group&1U);
#pragma unroll
        for(unsigned a=0;a<Rows;++a)if(a<count) {
            const float contribution=q20_apply(x,reinterpret_cast<const int*>(codes[a]+group*32U),sums[a][group],deltas[a][group]);
            acc[a]=acc[a]+contribution;
        }
    }
#pragma unroll
    for(unsigned a=0;a<Rows;++a)if(a<count) {
        acc[a]=acc[a]+strix::qwen_dpp::xor32(acc[a], 1);
        acc[a]=acc[a]+strix::qwen_dpp::xor32(acc[a], 2);
        if(owner==0U)view.rawdots[csr.rows[begin+a]][csr.slots[begin+a]*hidden+row]=acc[a];
    }
}
template<class T> __device__ __forceinline__ T* at(StridedRows<T> v,unsigned r) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(v.base)+static_cast<unsigned long long>(r)*v.stride);
}
// Prefill CSR over T*top_k assignments, one 1024-thread CTA: counting sort by
// expert; assignment a=r*top_k+s goes to offsets[e]+#{a'<a with expert e}, so
// each expert's rows (then slots) ascend. Batches are runs of at most `cap`
// assignments of one expert. Runs after the prologue and folds every row
// status into summary[0], so the projection kernels read one word.
constexpr unsigned serial_csr_max_rows=2048U; // build_prefill_csr keeps the expert of every assignment in LDS
__global__ __launch_bounds__(1024) void build_prefill_csr(PrefillRows v,unsigned rows,unsigned experts,unsigned top_k,unsigned cap) {
    const auto& csr=v.csr;
    const unsigned t=threadIdx.x,count=rows*top_k;
    __shared__ unsigned short expert_of[serial_csr_max_rows*10U];
    __shared__ unsigned n_of[512],scan[3][512],hist[16],failure,status,duplicates,maximum;
    if(t==0){failure=0U;status=0U;duplicates=0U;maximum=0U;}
    if(t<16U)hist[t]=0U;
    if(t<512U)n_of[t]=0U;
    __syncthreads();
    for(unsigned a=t;a<count;a+=1024U) {
        const unsigned r=a/top_k,s=a%top_k,e=at(v.routes,r)[s];
        if(e>=experts||!isfinite(at(v.route_weights,r)[s]))atomicOr(&failure,invalid_route);
        expert_of[a]=static_cast<unsigned short>(e<experts?e:0U);
    }
    for(unsigned r=t;r<rows;r+=1024U){const unsigned s=*at(v.status,r);if(s)atomicOr(&status,s);}
    __syncthreads();
    if(failure){if(t<32U)csr.summary[t]=t==0U?failure|status:t==3U?count:t==6U?rows:t==7U?1U:0U;return;}
    for(unsigned a=t;a<count;a+=1024U)atomicAdd(&n_of[expert_of[a]],1U);
    __syncthreads();
    const unsigned n=t<512U?n_of[t]:0U,nb=(n+cap-1U)/cap;
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
        csr.offsets[t]=off;
        if(t==experts-1U)csr.offsets[experts]=off+n;
        if(n) {
            csr.touched[touched]=t;atomicMax(&maximum,n);
            for(unsigned b=0;b<nb;++b){csr.experts[batch+b]=t;csr.begins[batch+b]=off+b*cap;atomicAdd(&hist[8U+min(8U,min(cap,n-b*cap))-1U],1U);}
            unsigned k=0U,unique=0U,last=UINT32_MAX;
            for(unsigned a=0;a<count&&k<n;++a)if(expert_of[a]==t) {
                const unsigned r=a/top_k;
                csr.rows[off+k]=r;csr.slots[off+k]=a%top_k;++k;
                unique+=r!=last;last=r;
            }
            atomicAdd(&hist[min(8U,unique)-1U],1U);atomicAdd(&duplicates,n-unique);
        }
    }
    __syncthreads();
    if(t==0U)csr.begins[scan[2][511]]=count;
    if(t<32U)csr.summary[t]=t==0U?status:t==1U?scan[1][511]:t==2U?scan[2][511]:t==3U?count:t==4U?duplicates:t==5U?maximum:
        t==6U?rows:t==7U?1U:t>=8U&&t<24U?hist[t-8U]:0U;
}
// build_prefill_csr with the same outputs (every array and summary word) in parallel: wave w
// owns the assignments [w*S, (w+1)*S) and counts its experts (two 16-bit counters per word);
// exclusive prefixes over the waves (16-bit, relative to the expert's offset) plus the expert
// offsets give each wave's first position per expert; then each wave walks its range in order,
// 32 assignments at a time, ranking lanes of one expert by lane order, so each expert's
// assignments ascend exactly as in the serial scan. An expert with 65536 or more assignments
// (possible only with repeated routes, T * top_k > 65535) fails closed as an invalid route.
__global__ __launch_bounds__(1024) void build_prefill_csr_waves(PrefillRows v,unsigned rows,unsigned experts,unsigned top_k,unsigned cap) {
    constexpr unsigned Waves=32U;
    const auto& csr=v.csr;
    const unsigned t=threadIdx.x,lane=t&31U,wave=t/32U,count=rows*top_k,span=(count+Waves-1U)/Waves;
    const unsigned first=wave*span,last=min(count,first+span);
    __shared__ unsigned position[Waves][256],n_of[512],scan[3][512],hist[16],failure,status,duplicates,maximum;
    if(t==0){failure=0U;status=0U;duplicates=0U;maximum=0U;}
    if(t<16U)hist[t]=0U;
    for(unsigned k=t;k<Waves*256U;k+=1024U)position[k/256U][k%256U]=0U;
    __syncthreads();
    const auto expert_at=[&](unsigned a){return at(v.routes,a/top_k)[a%top_k];};
    for(unsigned a=first+lane;a<last;a+=32U) {
        const unsigned e=expert_at(a);
        if(e>=experts||!isfinite(at(v.route_weights,a/top_k)[a%top_k]))atomicOr(&failure,invalid_route);
        else atomicAdd(&position[wave][e/2U],1U<<(16U*(e&1U)));
    }
    for(unsigned r=t;r<rows;r+=1024U){const unsigned s=*at(v.status,r);if(s)atomicOr(&status,s);}
    __syncthreads();
    if(failure){if(t<32U)csr.summary[t]=t==0U?failure|status:t==3U?count:t==6U?rows:t==7U?1U:0U;return;}
    if(t<256U) {
        unsigned lo=0U,hi=0U;
        for(unsigned w=0;w<Waves;++w){const unsigned c=position[w][t];position[w][t]=lo|hi<<16U;lo+=c&65535U;hi+=c>>16U;}
        if(lo>65535U||hi>65535U)atomicOr(&failure,invalid_route);
        n_of[2U*t]=lo;n_of[2U*t+1U]=hi;
    }
    __syncthreads();
    if(failure){if(t<32U)csr.summary[t]=t==0U?failure|status:t==3U?count:t==6U?rows:t==7U?1U:0U;return;}
    const unsigned n=t<512U?n_of[t]:0U,nb=(n+cap-1U)/cap;
    if(t<512U){scan[0][t]=n;scan[1][t]=n!=0U;scan[2][t]=nb;}
    __syncthreads();
    for(unsigned d=1U;d<512U;d<<=1U) {
        unsigned x[3]={};
        if(t>=d&&t<512U)for(unsigned k=0;k<3U;++k)x[k]=scan[k][t-d];
        __syncthreads();
        if(t>=d&&t<512U)for(unsigned k=0;k<3U;++k)scan[k][t]+=x[k];
        __syncthreads();
    }
    for(unsigned step=first;step<last;step+=32U) {
        const unsigned a=step+lane;
        const bool valid=a<last;
        const unsigned e=valid?expert_at(a):0U;
        unsigned active=static_cast<unsigned>(__ballot(valid)),peers=0U;
        while(active) {
            const unsigned leader_e=static_cast<unsigned>(__shfl(static_cast<int>(e),__ffs(active)-1,32));
            const unsigned same=static_cast<unsigned>(__ballot(valid&&e==leader_e))&active;
            if(valid&&e==leader_e)peers=same;
            active&=~same;
        }
        if(valid) {
            const unsigned slot=scan[0][e]-n_of[e]+((position[wave][e/2U]>>(16U*(e&1U)))&65535U);
            const unsigned at_pos=slot+static_cast<unsigned>(__popc(peers&((1U<<lane)-1U)));
            csr.rows[at_pos]=a/top_k;csr.slots[at_pos]=a%top_k;
            if((peers&((1U<<lane)-1U))==0U)atomicAdd(&position[wave][e/2U],static_cast<unsigned>(__popc(peers))<<(16U*(e&1U)));
        }
    }
    __syncthreads();
    if(t<experts) {
        const unsigned off=scan[0][t]-n,touched=scan[1][t]-(n!=0U),batch=scan[2][t]-nb;
        csr.offsets[t]=off;
        if(t==experts-1U)csr.offsets[experts]=off+n;
        if(n) {
            csr.touched[touched]=t;atomicMax(&maximum,n);
            for(unsigned b=0;b<nb;++b){csr.experts[batch+b]=t;csr.begins[batch+b]=off+b*cap;atomicAdd(&hist[8U+min(8U,min(cap,n-b*cap))-1U],1U);}
            unsigned unique=0U,previous=UINT32_MAX;
            for(unsigned k=0;k<n;++k){const unsigned r=csr.rows[off+k];unique+=r!=previous;previous=r;}
            atomicAdd(&hist[min(8U,unique)-1U],1U);atomicAdd(&duplicates,n-unique);
        }
    }
    __syncthreads();
    if(t==0U)csr.begins[scan[2][511]]=count;
    if(t<32U)csr.summary[t]=t==0U?status:t==1U?scan[1][511]:t==2U?scan[2][511]:t==3U?count:t==4U?duplicates:t==5U?maximum:
        t==6U?rows:t==7U?1U:t>=8U&&t<24U?hist[t-8U]:0U;
}
// project_window over strided prefill rows: same per-(row,slot) lane arithmetic,
// staging, chains and A8 down quantizer epilogue. Grid x = CSR batch (runs of one
// expert are adjacent, so their weight slices hit L2), y = 32-output tile.
template<unsigned G,bool Q5K,unsigned Chunk=max_hidden/256U>
__global__ __launch_bounds__(256) void project_prefill(const unsigned char* gate,const unsigned char* up,PrefillRows view,
    unsigned hidden,unsigned outputs) {
    const auto& csr=view.csr;
    if((csr.summary[0]&~projection_nonfinite)||blockIdx.x>=csr.summary[2])return;
    const unsigned tile_index=blockIdx.y;
    const auto begin=csr.begins[blockIdx.x],count=csr.begins[blockIdx.x+1U]-begin,expert=csr.experts[blockIdx.x];
    __shared__ __align__(16) signed char local_codes[G][max_hidden];
    __shared__ __align__(16) float local_deltas[G][max_hidden/32U];
    __shared__ __align__(16) int local_sums[G][max_hidden/32U];
    __shared__ float tile[2][G][32];
    const unsigned lane=threadIdx.x&31U,c=lane&7U,p=c/2U,h=c&1U,wave=threadIdx.x/32U;
    const unsigned out=tile_index*32U+wave*4U+lane/8U,blocks=hidden/256U;
    constexpr unsigned block_bytes=Q5K?176U:144U;
    const auto offset=(static_cast<unsigned long long>(expert)*outputs+out)*blocks*block_bytes;
    const unsigned g0=2U*p,g1=g0+1U;
    constexpr unsigned max_blocks=max_hidden/256U;
    for(unsigned group=0;group<count;group+=G) {
        if(group)__syncthreads();
#pragma unroll
        for(unsigned a=0;a<G;++a)if(group+a<count) {
            const auto r=csr.rows[begin+group+a];
            const auto* codes=at(view.codes,r);const auto* deltas=at(view.deltas,r);const auto* sums=at(view.sums,r);
            for(unsigned i=16U*threadIdx.x;i<hidden;i+=4096U)
                *reinterpret_cast<int4*>(local_codes[a]+i)=*reinterpret_cast<const int4*>(codes+i);
            for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U){local_deltas[a][i]=deltas[i];local_sums[a][i]=sums[i];}
        }
        __syncthreads();
        float acc[2][G]={};
#pragma unroll
        for(unsigned m=0;m<2U;++m) {
            const unsigned char* w=(m==0U?gate:up)+offset;
            if constexpr(Q5K) {
#pragma unroll 1
                for(unsigned b=0;b<blocks;++b) {
                    const auto* packed=w+b*176U;
                    const uint4 hd=*reinterpret_cast<const uint4*>(packed);
                    const uint4 q=*reinterpret_cast<const uint4*>(packed+48U+c*16U);
                    const uint4 qh=*reinterpret_cast<const uint4*>(packed+16U+h*16U);
#pragma unroll
                    for(unsigned a=0;a<G;++a)if(group+a<count) {
                        const int4 a0=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+h*16U);
                        const int4 a1=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+32U+h*16U);
                        const float pair=paired_q5k(hd,q,qh,p,a0,a1,local_sums[a][b*8U+g0],local_sums[a][b*8U+g1],
                            local_deltas[a][b*8U+g0],local_deltas[a][b*8U+g1]);
                        acc[m][a]=acc[m][a]+pair;
                    }
                }
            } else {
#pragma unroll
                for(unsigned first=0;first<max_blocks;first+=Chunk) {
                uint4 hd[Chunk],q[Chunk];
#pragma unroll
                for(unsigned i=0;i<Chunk;++i)if(first+i<blocks) {
                    const unsigned b=first+i;
                    hd[i]=*reinterpret_cast<const uint4*>(w+b*144U);q[i]=*reinterpret_cast<const uint4*>(w+b*144U+16U+c*16U);
                }
#pragma unroll
                for(unsigned i=0;i<Chunk;++i)if(first+i<blocks) {
                    const unsigned b=first+i;
                    const auto k=coefficient(hd[i],q[i],p);
#pragma unroll
                    for(unsigned a=0;a<G;++a)if(group+a<count) {
                        const int4 a0=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+h*16U);
                        const int4 a1=*reinterpret_cast<const int4*>(local_codes[a]+b*256U+p*64U+32U+h*16U);
                        const float pair=paired(k,a0,a1,local_sums[a][b*8U+g0],local_sums[a][b*8U+g1],
                            local_deltas[a][b*8U+g0],local_deltas[a][b*8U+g1]);
                        acc[m][a]=acc[m][a]+pair;
                    }
                }
                }
            }
        }
#pragma unroll
        for(unsigned a=0;a<G;++a)if(group+a<count) {
#pragma unroll
            for(unsigned m=0;m<2U;++m){acc[m][a]=acc[m][a]+strix::qwen_dpp::xor32(acc[m][a], 2);acc[m][a]=acc[m][a]+strix::qwen_dpp::xor32(acc[m][a], 4);}
            if(c==0U){tile[0][a][out%32U]=acc[0][a];tile[1][a][out%32U]=acc[1][a];}
        }
        // Serial down_activation_quantize for group tile_index of each staged (row, slot).
        __syncthreads();
        for(unsigned a=wave;a<G;a+=8U)if(group+a<count) {
            const auto r=csr.rows[begin+group+a],slot=csr.slots[begin+group+a];
            float* output=at(view.output,r);
            const unsigned feature=tile_index*32U+lane,at_slot=slot*2U*outputs+feature;
            const float g=tile[0][a][lane],u=tile[1][a][lane];
            const float hv=(g/(1.0F+expf(-g)))*u;
            output[at_slot+outputs]=u;
            const unsigned bad=static_cast<unsigned>(__ballot(!isfinite(g)||!isfinite(u)||!isfinite(hv)));
            if(bad!=0){output[at_slot]=g;if(lane==0){atomicOr(at(view.status,r),projection_nonfinite);atomicOr(csr.summary,projection_nonfinite);}continue;}
            float x=fabsf(hv);
            for(int stride=16;stride!=0;stride/=2) x=fmaxf(x,strix::qwen_dpp::xor32(x, stride));
            const bool tiny=x>0.0F && x<0x1p-119F;
            const float delta=x==0.0F || tiny?1.0F:divide_rne(x,127.0F);
            const float rounded=tiny?0.0F:rintf(divide_rne(hv,delta));
            const int code=static_cast<int>(fminf(127.0F,fmaxf(-127.0F,rounded)));
            int sum=code;
            for(int stride=16;stride!=0;stride/=2) sum+=strix::qwen_dpp::xor32(sum, stride);
            output[at_slot]=hv;
            at(view.down_codes,r)[slot*outputs+feature]=static_cast<signed char>(code);
            if(lane==0){at(view.down_deltas,r)[slot*(outputs/32U)+tile_index]=delta;at(view.down_sums,r)[slot*(outputs/32U)+tile_index]=sum;}
        }
    }
}
// Q2_0 prefill (dot4 form): project_window_q20 / down_window_q20 arithmetic per (row, slot) over the
// strided prefill rows; grid x = CSR batch, y = 32-output (gate/up) or 64-row (down) tile.
template<unsigned G>
__global__ __launch_bounds__(256) void project_prefill_q20(const unsigned char* gate,const unsigned char* up,PrefillRows view,
    unsigned hidden,unsigned outputs) {
    const auto& csr=view.csr;
    if((csr.summary[0]&~projection_nonfinite)||blockIdx.x>=csr.summary[2])return;
    const unsigned tile_index=blockIdx.y;
    const auto begin=csr.begins[blockIdx.x],count=csr.begins[blockIdx.x+1U]-begin,expert=csr.experts[blockIdx.x];
    __shared__ __align__(16) signed char local_codes[G][max_hidden];
    __shared__ __align__(16) float local_deltas[G][max_hidden/32U];
    __shared__ __align__(16) int local_sums[G][max_hidden/32U];
    __shared__ float tile[2][G][32];
    const unsigned lane=threadIdx.x&31U,c=lane&7U,wave=threadIdx.x/32U;
    const unsigned out=tile_index*32U+wave*4U+lane/8U,groups=hidden/32U;
    const auto offset=(static_cast<unsigned long long>(expert)*outputs+out)*(hidden/64U)*18U;
    for(unsigned group=0;group<count;group+=G) {
        if(group)__syncthreads();
#pragma unroll
        for(unsigned a=0;a<G;++a)if(group+a<count) {
            const auto r=csr.rows[begin+group+a];
            const auto* codes=at(view.codes,r);const auto* deltas=at(view.deltas,r);const auto* sums=at(view.sums,r);
            for(unsigned i=16U*threadIdx.x;i<hidden;i+=4096U)
                *reinterpret_cast<int4*>(local_codes[a]+i)=*reinterpret_cast<const int4*>(codes+i);
            for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U){local_deltas[a][i]=deltas[i];local_sums[a][i]=sums[i];}
        }
        __syncthreads();
        float acc[2][G]={};
#pragma unroll
        for(unsigned m=0;m<2U;++m) {
            const unsigned char* w=(m==0U?gate:up)+offset;
#pragma unroll 1
            for(unsigned g=c;g<groups;g+=8U) {
                const auto x=q20_load(w+(g/2U)*18U,g&1U);
#pragma unroll
                for(unsigned a=0;a<G;++a)if(group+a<count) {
                    const float value=q20_apply(x,reinterpret_cast<const int*>(local_codes[a]+g*32U),local_sums[a][g],local_deltas[a][g]);
                    acc[m][a]=acc[m][a]+value;
                }
            }
        }
#pragma unroll
        for(unsigned a=0;a<G;++a)if(group+a<count) {
#pragma unroll
            for(unsigned m=0;m<2U;++m){acc[m][a]=acc[m][a]+strix::qwen_dpp::xor32(acc[m][a], 1);acc[m][a]=acc[m][a]+strix::qwen_dpp::xor32(acc[m][a], 2);
                acc[m][a]=acc[m][a]+strix::qwen_dpp::xor32(acc[m][a], 4);}
            if(c==0U){tile[0][a][out%32U]=acc[0][a];tile[1][a][out%32U]=acc[1][a];}
        }
        __syncthreads();
        for(unsigned a=wave;a<G;a+=8U)if(group+a<count) {
            const auto r=csr.rows[begin+group+a],slot=csr.slots[begin+group+a];
            float* output=at(view.output,r);
            const unsigned feature=tile_index*32U+lane,at_slot=slot*2U*outputs+feature;
            const float g=tile[0][a][lane],u=tile[1][a][lane];
            const float hv=(g/(1.0F+expf(-g)))*u;
            output[at_slot+outputs]=u;
            const unsigned bad=static_cast<unsigned>(__ballot(!isfinite(g)||!isfinite(u)||!isfinite(hv)));
            if(bad!=0){output[at_slot]=g;if(lane==0){atomicOr(at(view.status,r),projection_nonfinite);atomicOr(csr.summary,projection_nonfinite);}continue;}
            float x=fabsf(hv);
            for(int stride=16;stride!=0;stride/=2) x=fmaxf(x,strix::qwen_dpp::xor32(x, stride));
            const bool tiny=x>0.0F && x<0x1p-119F;
            const float delta=x==0.0F || tiny?1.0F:divide_rne(x,127.0F);
            const float rounded=tiny?0.0F:rintf(divide_rne(hv,delta));
            const int code=static_cast<int>(fminf(127.0F,fmaxf(-127.0F,rounded)));
            int sum=code;
            for(int stride=16;stride!=0;stride/=2) sum+=strix::qwen_dpp::xor32(sum, stride);
            output[at_slot]=hv;
            at(view.down_codes,r)[slot*outputs+feature]=static_cast<signed char>(code);
            if(lane==0){at(view.down_deltas,r)[slot*(outputs/32U)+tile_index]=delta;at(view.down_sums,r)[slot*(outputs/32U)+tile_index]=sum;}
        }
    }
}
template<unsigned Rows>
__global__ __launch_bounds__(256) void down_prefill_q20(const unsigned char* weights,PrefillRows view,unsigned hidden,unsigned width) {
    const auto& csr=view.csr;
    if(csr.summary[0]||blockIdx.x>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.x],count=csr.begins[blockIdx.x+1U]-begin,expert=csr.experts[blockIdx.x];
    __shared__ __align__(16) signed char codes[Rows][max_rows];
    __shared__ float deltas[Rows][max_rows/32U];
    __shared__ int sums[Rows][max_rows/32U];
    const unsigned groups=width/32U;
    const unsigned lane=threadIdx.x&31U,owner=lane&3U;
    const unsigned row=blockIdx.y*64U+(threadIdx.x/32U)*8U+lane/4U;
    const auto* row_weights=weights+(static_cast<unsigned long long>(expert)*hidden+row)*(width/64U)*18U;
    for(unsigned first=0;first<count;first+=Rows) {
        if(first)__syncthreads();
#pragma unroll
        for(unsigned a=0;a<Rows;++a)if(first+a<count) {
            const auto r=csr.rows[begin+first+a],slot=csr.slots[begin+first+a];
            const auto* down_codes=at(view.down_codes,r)+slot*width;
            const auto* down_deltas=at(view.down_deltas,r)+slot*groups;const auto* down_sums=at(view.down_sums,r)+slot*groups;
            for(unsigned i=16U*threadIdx.x;i<width;i+=4096U)
                *reinterpret_cast<int4*>(codes[a]+i)=*reinterpret_cast<const int4*>(down_codes+i);
            for(unsigned i=threadIdx.x;i<groups;i+=256U){deltas[a][i]=down_deltas[i];sums[a][i]=down_sums[i];}
        }
        __syncthreads();
        float acc[Rows]={};
#pragma unroll 1
        for(unsigned group=owner;group<groups;group+=4U) {
            const auto x=q20_load(row_weights+(group/2U)*18U,group&1U);
#pragma unroll
            for(unsigned a=0;a<Rows;++a)if(first+a<count) {
                const float contribution=q20_apply(x,reinterpret_cast<const int*>(codes[a]+group*32U),sums[a][group],deltas[a][group]);
                acc[a]=acc[a]+contribution;
            }
        }
#pragma unroll
        for(unsigned a=0;a<Rows;++a)if(first+a<count) {
            acc[a]=acc[a]+strix::qwen_dpp::xor32(acc[a], 1);
            acc[a]=acc[a]+strix::qwen_dpp::xor32(acc[a], 2);
            if(owner==0U)at(view.rawdots,csr.rows[begin+first+a])[csr.slots[begin+first+a]*hidden+row]=acc[a];
        }
    }
}
// GSQ IQ window / prefill forms: project_iq / down_rawdots_iq lane arithmetic per (row, slot).
template<std::uint32_t T,unsigned G,bool Prefill,class View>
__device__ __forceinline__ void project_iq_body(const unsigned char* gate,const unsigned char* up,const View& view,
    const DeviceCsr& csr,unsigned batch,unsigned tile_index,unsigned hidden,unsigned outputs) {
    const auto begin=csr.begins[batch],count=csr.begins[batch+1U]-begin,expert=csr.experts[batch];
    __shared__ __align__(16) signed char local_codes[G][max_hidden];
    __shared__ __align__(16) float local_deltas[G][max_hidden/32U];
    __shared__ float tile[2][G][32];
    const unsigned lane=threadIdx.x&31U,c=lane&7U,wave=threadIdx.x/32U;
    const unsigned out=tile_index*32U+wave*4U+lane/8U,groups=hidden/32U;
    const auto offset=(static_cast<unsigned long long>(expert)*outputs+out)*gsq_a8::row_bytes(T,hidden);
    for(unsigned group=0;group<count;group+=G) {
        if(group)__syncthreads();
#pragma unroll
        for(unsigned a=0;a<G;++a)if(group+a<count) {
            const auto r=csr.rows[begin+group+a];
            const signed char* codes;const float* deltas;
            if constexpr(Prefill){codes=at(view.codes,r);deltas=at(view.deltas,r);}else{codes=view.codes[r];deltas=view.deltas[r];}
            for(unsigned i=16U*threadIdx.x;i<hidden;i+=4096U)
                *reinterpret_cast<int4*>(local_codes[a]+i)=*reinterpret_cast<const int4*>(codes+i);
            for(unsigned i=threadIdx.x;i<hidden/32U;i+=256U)local_deltas[a][i]=deltas[i];
        }
        __syncthreads();
        float acc[2][G]={};
#pragma unroll
        for(unsigned m=0;m<2U;++m) {
            const unsigned char* w=(m==0U?gate:up)+offset;
#pragma unroll 1
            for(unsigned g=c;g<groups;g+=8U) {
                const auto x=gsq_a8::iq_group<T>(w,g);
#pragma unroll
                for(unsigned a=0;a<G;++a)if(group+a<count) {
                    const float value=gsq_a8::iq_apply<T>(x,reinterpret_cast<const int*>(local_codes[a]+g*32U),local_deltas[a][g]);
                    acc[m][a]=acc[m][a]+value;
                }
            }
        }
#pragma unroll
        for(unsigned a=0;a<G;++a)if(group+a<count) {
#pragma unroll
            for(unsigned m=0;m<2U;++m){acc[m][a]=acc[m][a]+strix::qwen_dpp::xor32(acc[m][a], 1);acc[m][a]=acc[m][a]+strix::qwen_dpp::xor32(acc[m][a], 2);
                acc[m][a]=acc[m][a]+strix::qwen_dpp::xor32(acc[m][a], 4);}
            if(c==0U){tile[0][a][out%32U]=acc[0][a];tile[1][a][out%32U]=acc[1][a];}
        }
        __syncthreads();
        for(unsigned a=wave;a<G;a+=8U)if(group+a<count) {
            const auto r=csr.rows[begin+group+a],slot=csr.slots[begin+group+a];
            float* output;unsigned* status;signed char* down_codes;float* down_deltas;int* down_sums;
            if constexpr(Prefill){output=at(view.output,r);status=at(view.status,r);down_codes=at(view.down_codes,r);
                down_deltas=at(view.down_deltas,r);down_sums=at(view.down_sums,r);}
            else{output=view.output[r];status=view.status[r];down_codes=view.down_codes[r];down_deltas=view.down_deltas[r];down_sums=view.down_sums[r];}
            const unsigned feature=tile_index*32U+lane,at_slot=slot*2U*outputs+feature;
            const float g=tile[0][a][lane],u=tile[1][a][lane];
            const float hv=(g/(1.0F+expf(-g)))*u;
            output[at_slot+outputs]=u;
            const unsigned bad=static_cast<unsigned>(__ballot(!isfinite(g)||!isfinite(u)||!isfinite(hv)));
            if(bad!=0){output[at_slot]=g;if(lane==0){atomicOr(status,projection_nonfinite);if constexpr(Prefill)atomicOr(csr.summary,projection_nonfinite);}continue;}
            float x=fabsf(hv);
            for(int stride=16;stride!=0;stride/=2) x=fmaxf(x,strix::qwen_dpp::xor32(x, stride));
            const bool tiny=x>0.0F && x<0x1p-119F;
            const float delta=x==0.0F || tiny?1.0F:divide_rne(x,127.0F);
            const float rounded=tiny?0.0F:rintf(divide_rne(hv,delta));
            const int code=static_cast<int>(fminf(127.0F,fmaxf(-127.0F,rounded)));
            int sum=code;
            for(int stride=16;stride!=0;stride/=2) sum+=strix::qwen_dpp::xor32(sum, stride);
            output[at_slot]=hv;
            down_codes[slot*outputs+feature]=static_cast<signed char>(code);
            if(lane==0){down_deltas[slot*(outputs/32U)+tile_index]=delta;down_sums[slot*(outputs/32U)+tile_index]=sum;}
        }
    }
}
template<std::uint32_t T,unsigned G>
__global__ __launch_bounds__(256) void project_window_iq(const unsigned char* gate,const unsigned char* up,DeviceRows view,DeviceCsr csr,
    unsigned rows,unsigned hidden,unsigned outputs) {
    unsigned failure=csr.summary[0];
    for(unsigned r=0;r<rows;++r)failure|=*view.status[r];
    if((failure&~projection_nonfinite)||blockIdx.y>=csr.summary[2])return;
    project_iq_body<T,G,false>(gate,up,view,csr,blockIdx.y,blockIdx.x,hidden,outputs);
}
template<std::uint32_t T,unsigned G>
__global__ __launch_bounds__(256) void project_prefill_iq(const unsigned char* gate,const unsigned char* up,PrefillRows view,
    unsigned hidden,unsigned outputs) {
    const auto& csr=view.csr;
    if((csr.summary[0]&~projection_nonfinite)||blockIdx.x>=csr.summary[2])return;
    const DeviceCsr d{csr.summary,csr.offsets,csr.touched,csr.rows,csr.slots,csr.experts,csr.begins};
    project_iq_body<T,G,true>(gate,up,view,d,blockIdx.x,blockIdx.y,hidden,outputs);
}
template<std::uint32_t T,unsigned Rows,bool Prefill,class View>
__device__ __forceinline__ void down_iq_body(const unsigned char* weights,const View& view,const DeviceCsr& csr,
    unsigned batch,unsigned tile_index,unsigned hidden,unsigned width) {
    const auto begin=csr.begins[batch],count=csr.begins[batch+1U]-begin,expert=csr.experts[batch];
    __shared__ __align__(16) signed char codes[Rows][max_rows];
    __shared__ float deltas[Rows][max_rows/32U];
    const unsigned groups=width/32U;
    const unsigned lane=threadIdx.x&31U,owner=lane&3U;
    const unsigned row=tile_index*64U+(threadIdx.x/32U)*8U+lane/4U;
    const auto* row_weights=weights+(static_cast<unsigned long long>(expert)*hidden+row)*gsq_a8::row_bytes(T,width);
    for(unsigned first=0;first<count;first+=Rows) {
        if(first)__syncthreads();
#pragma unroll
        for(unsigned a=0;a<Rows;++a)if(first+a<count) {
            const auto r=csr.rows[begin+first+a],slot=csr.slots[begin+first+a];
            const signed char* dc;const float* dd;
            if constexpr(Prefill){dc=at(view.down_codes,r)+slot*width;dd=at(view.down_deltas,r)+slot*groups;}
            else{dc=view.down_codes[r]+slot*width;dd=view.down_deltas[r]+slot*groups;}
            for(unsigned i=16U*threadIdx.x;i<width;i+=4096U)
                *reinterpret_cast<int4*>(codes[a]+i)=*reinterpret_cast<const int4*>(dc+i);
            for(unsigned i=threadIdx.x;i<groups;i+=256U)deltas[a][i]=dd[i];
        }
        __syncthreads();
        float acc[Rows]={};
#pragma unroll 1
        for(unsigned group=owner;group<groups;group+=4U) {
            const auto x=gsq_a8::iq_group<T>(row_weights,group);
#pragma unroll
            for(unsigned a=0;a<Rows;++a)if(first+a<count) {
                const float contribution=gsq_a8::iq_apply<T>(x,reinterpret_cast<const int*>(codes[a]+group*32U),deltas[a][group]);
                acc[a]=acc[a]+contribution;
            }
        }
#pragma unroll
        for(unsigned a=0;a<Rows;++a)if(first+a<count) {
            acc[a]=acc[a]+strix::qwen_dpp::xor32(acc[a], 1);
            acc[a]=acc[a]+strix::qwen_dpp::xor32(acc[a], 2);
            if(owner==0U) {
                float* raw;
                if constexpr(Prefill)raw=at(view.rawdots,csr.rows[begin+first+a]);else raw=view.rawdots[csr.rows[begin+first+a]];
                raw[csr.slots[begin+first+a]*hidden+row]=acc[a];
            }
        }
        if constexpr(!Prefill)break;  // window batches hold at most Rows assignments
    }
}
template<std::uint32_t T,unsigned Rows>
__global__ __launch_bounds__(256) void down_window_iq(const unsigned char* weights,DeviceRows view,DeviceCsr csr,
    unsigned rows,unsigned hidden,unsigned width) {
    unsigned failure=csr.summary[0];
    for(unsigned r=0;r<rows;++r)failure|=*view.status[r];
    if(failure){if(blockIdx.x==0U&&blockIdx.y==0U&&threadIdx.x==0U)csr.summary[0]=failure;return;}
    if(blockIdx.y>=csr.summary[2])return;
    down_iq_body<T,Rows,false>(weights,view,csr,blockIdx.y,blockIdx.x,hidden,width);
}
template<std::uint32_t T,unsigned Rows>
__global__ __launch_bounds__(256) void down_prefill_iq(const unsigned char* weights,PrefillRows view,unsigned hidden,unsigned width) {
    const auto& csr=view.csr;
    if(csr.summary[0]||blockIdx.x>=csr.summary[2])return;
    const DeviceCsr d{csr.summary,csr.offsets,csr.touched,csr.rows,csr.slots,csr.experts,csr.begins};
    down_iq_body<T,Rows,true>(weights,view,d,blockIdx.x,blockIdx.y,hidden,width);
}
// down_window over strided prefill rows, Rows staged rows at a time: same lane
// chains, block decode and reduction per (row, slot). Grid x = batch, y = 64-row tile.
template<unsigned Rows,bool Q51>
__global__ __launch_bounds__(256) void down_prefill(const unsigned char* weights,PrefillRows view,unsigned hidden,unsigned width) {
    const auto& csr=view.csr;
    if(csr.summary[0]||blockIdx.x>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.x],count=csr.begins[blockIdx.x+1U]-begin,expert=csr.experts[blockIdx.x];
    __shared__ __align__(16) signed char codes[Rows][max_rows];
    __shared__ float deltas[Rows][max_rows/32U];
    __shared__ int sums[Rows][max_rows/32U];
    const unsigned groups=width/32U;
    const unsigned lane=threadIdx.x&31U,owner=lane&3U;
    const unsigned row=blockIdx.y*64U+(threadIdx.x/32U)*8U+lane/4U;
    constexpr unsigned block_bytes=Q51?24U:34U;
    const auto* row_weights=weights+(static_cast<unsigned long long>(expert)*hidden+row)*groups*block_bytes;
    for(unsigned first=0;first<count;first+=Rows) {
        if(first)__syncthreads();
#pragma unroll
        for(unsigned a=0;a<Rows;++a)if(first+a<count) {
            const auto r=csr.rows[begin+first+a],slot=csr.slots[begin+first+a];
            const auto* down_codes=at(view.down_codes,r)+slot*width;
            const auto* down_deltas=at(view.down_deltas,r)+slot*groups;const auto* down_sums=at(view.down_sums,r)+slot*groups;
            for(unsigned i=16U*threadIdx.x;i<width;i+=4096U)
                *reinterpret_cast<int4*>(codes[a]+i)=*reinterpret_cast<const int4*>(down_codes+i);
            for(unsigned i=threadIdx.x;i<groups;i+=256U){deltas[a][i]=down_deltas[i];sums[a][i]=down_sums[i];}
        }
        __syncthreads();
        float acc[Rows]={};
#pragma unroll 1
        for(unsigned group=owner;group<groups;group+=4U) {
            const auto* packed=row_weights+group*block_bytes;
            unsigned words[8];
#pragma unroll
            for(unsigned word=0;word<8U;++word)words[word]=Q51?down_q51_word(packed,word*4U):down_u32(packed+2U+word*4U);
            const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(down_u16(packed))));
            const float m=Q51?__half2float(__ushort_as_half(static_cast<unsigned short>(down_u16(packed+2U)))):0.0F;
#pragma unroll
            for(unsigned a=0;a<Rows;++a)if(first+a<count) {
                const auto* image=reinterpret_cast<const int*>(codes[a]+group*32U);
                int integer=0;
#pragma unroll
                for(unsigned word=0;word<8U;++word) {
                    if constexpr(Q51) integer=dot(image[word],words[word],integer);
                    else integer=__builtin_amdgcn_sudot4(true,image[word],true,static_cast<int>(words[word]),integer,false);
                }
                const float first_term=d*static_cast<float>(integer);
                float combined=first_term;
                if constexpr(Q51) {
                    const float correction=m*static_cast<float>(sums[a][group]);
                    combined=first_term+correction;
                }
                const float contribution=deltas[a][group]*combined;
                acc[a]=acc[a]+contribution;
            }
        }
#pragma unroll
        for(unsigned a=0;a<Rows;++a)if(first+a<count) {
            acc[a]=acc[a]+strix::qwen_dpp::xor32(acc[a], 1);
            acc[a]=acc[a]+strix::qwen_dpp::xor32(acc[a], 2);
            if(owner==0U)at(view.rawdots,csr.rows[begin+first+a])[csr.slots[begin+first+a]*hidden+row]=acc[a];
        }
    }
}
// iu8 WMMA forms of project_prefill/down_prefill, bit-identical to them. A CSR batch holds at
// most 16 assignments of one expert: the M = 16 rows of v_wmma_i32_16x16x16_iu8 (rows past the
// batch are zero codes, their D elements are never stored and skip the epilogue in pairs). Per
// 32-value group, two WMMAs (K = 16 + 16) on an accumulator of 0x4b400000 give the exact int32
// group dot r of every (row, output) as the bits of 1.5*2^23 + r (|r| < 2^22), so one exact
// FP32 subtraction yields float(r); the epilogue then runs the incumbent per-lane expressions
// on (row, output) elements: gate/up u = d*float(sc*r) with float(sc*r) = float(sc)*float(r)
// (exact, |sc*r| < 2^24), v = dm*float(m*s) likewise, (e0*(u0-v0) + e1*(u1-v1)) added to the
// accumulator of pair p in ascending K blocks, and the lanes' tree (acc0+acc1)+(acc2+acc3);
// down d*float(r) (+ m*float(sum)), times the delta, added to the accumulator of group%4 in
// ascending groups, same tree. Q4_K/Q5_K nibbles (plus the Q5 high bit) and Q5_1 u5 codes are
// unsigned 0..31, Q8_0 signed: all exact as signed int8 operands. Layouts (wave32): A lane
// i%16 = row i, B lane i%16 = output i, D element e of lane l = (row 2e + l/16, output l%16);
// the upper half wave repeats A/B. The epilogue quantizer and the stores are project_prefill's.
typedef int wmma_v4i __attribute__((ext_vector_type(4)));
typedef int wmma_v8i __attribute__((ext_vector_type(8)));
struct __attribute__((aligned(4))) Dword4 { unsigned x,y,z,w; };
constexpr unsigned wmma_batch=16U;
constexpr int wmma_magic=0x4b400000;
__device__ __forceinline__ wmma_v4i operand(uint4 v) {
    return wmma_v4i{static_cast<int>(v.x),static_cast<int>(v.y),static_cast<int>(v.z),static_cast<int>(v.w)};
}
__device__ __forceinline__ wmma_v8i group_dot(wmma_v4i a0,wmma_v4i b0,wmma_v4i a1,wmma_v4i b1) {
    const wmma_v8i magic{wmma_magic,wmma_magic,wmma_magic,wmma_magic,wmma_magic,wmma_magic,wmma_magic,wmma_magic};
    const wmma_v8i c=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a0,true,b0,magic,false);
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a1,true,b1,c,false);
}
__device__ __forceinline__ float group_float(int bits){return __int_as_float(bits)-0x1.8p23F;}
// Row r of a batch at slot (r&1)*8 + r/2, so D element e of half h reads slot 8h + e.
__device__ __forceinline__ unsigned wmma_row_slot(unsigned r){return (r&1U)<<3U|r>>1U;}
template<unsigned Groups> struct WmmaRowsStage {
    wmma_v4i a[2U*Groups][wmma_batch]; // 16-value plane k of every row: codes [16k, 16k+16)
    float delta[Groups][wmma_batch];
    float sum[Groups][wmma_batch];  // float(int sum), exact
};
// Batch rows (row, slot) and their codes/deltas/sums into a stage; rows >= count are zero.
template<unsigned Groups,class Codes,class Deltas,class Sums>
__device__ __forceinline__ void wmma_stage_rows(WmmaRowsStage<Groups>& s,unsigned count,unsigned width,unsigned threads,
    const Codes& codes,const Deltas& deltas,const Sums& sums) {
    const unsigned planes=width/16U,groups=width/32U;
    for(unsigned item=threadIdx.x;item<wmma_batch*planes;item+=threads) {
        const unsigned a=item/planes,k=item%planes;
        wmma_v4i v{0,0,0,0};
        if(a<count)v=*reinterpret_cast<const wmma_v4i*>(codes(a)+16U*k);
        s.a[k][a]=v;
    }
    for(unsigned item=threadIdx.x;item<wmma_batch*groups;item+=threads) {
        const unsigned a=item/groups,g=item%groups,slot=wmma_row_slot(a);
        float e=0.0F,f=0.0F;
        if(a<count){e=deltas(a)[g];f=static_cast<float>(sums(a)[g]);}
        s.delta[g][slot]=e;s.sum[g][slot]=f;
    }
}
// x[e] = p[e], e < V (two 16-byte reads for V = 8).
template<unsigned V>
__device__ __forceinline__ void row_values(const float* p,float (&x)[8]) {
    const float4 lo=reinterpret_cast<const float4*>(p)[0];
    x[0]=lo.x;x[1]=lo.y;
    if constexpr(V>2U){x[2]=lo.z;x[3]=lo.w;}
    if constexpr(V>4U){const float4 hi=reinterpret_cast<const float4*>(p)[1];x[4]=hi.x;x[5]=hi.y;x[6]=hi.z;x[7]=hi.w;}
}
// The (row, output) elements e < V of one wave: four chain accumulators, folded at the end as the
// lanes' xor-2/xor-4 (gate/up) or xor-1/xor-2 (down) tree: (acc0+acc1)+(acc2+acc3).
// FP32 value of an fp16 scale (exact) behind an empty asm, so its products stay v_mul_f32 (the
// compiler would otherwise fold the conversion into every product as v_fma_mix_f32 with a -0
// addend: the same value, but not dual-issuable).
__device__ __forceinline__ float scale_f32(unsigned short bits) {
    float x=__half2float(__ushort_as_half(bits));
    __asm__("" : "+v"(x));
    return x;
}
template<unsigned V> struct Chains {
    float acc[4][8];
    __device__ Chains(){
#pragma unroll
        for(unsigned c=0;c<4U;++c)
#pragma unroll
            for(unsigned e=0;e<8U;++e)acc[c][e]=0.0F;
    }
    __device__ void fold(float (&out)[8]) const {
#pragma unroll
        for(unsigned e=0;e<V;++e){const float s01=acc[0][e]+acc[1][e],s23=acc[2][e]+acc[3][e];out[e]=s01+s23;}
    }
};
// One matrix (gate or up) of a wave: K blocks ascending, the four pairs p of a block in order,
// two block buffers in turn (the next block loads while one computes). Elements e < V only
// (rows 2e, 2e+1 of the batch); the WMMAs always cover all 16 rows.
template<bool Q5K,unsigned V,unsigned Groups>
__device__ __forceinline__ void gate_up_chains(const unsigned char* w,const WmmaRowsStage<Groups>& s,unsigned blocks,unsigned i,unsigned h,
    float (&out)[8]) {
    constexpr unsigned block_bytes=Q5K?176U:144U,qs_at=Q5K?48U:16U,mask=0x0f0f0f0fU;
    struct Block { uint4 hd,q[8],qh[2]; };
    const auto load=[&](unsigned b,Block& x) {
        const auto* packed=w+b*block_bytes;
        x.hd=*reinterpret_cast<const uint4*>(packed);
#pragma unroll
        for(unsigned k=0;k<8U;++k)x.q[k]=*reinterpret_cast<const uint4*>(packed+qs_at+16U*k);
        if constexpr(Q5K){x.qh[0]=*reinterpret_cast<const uint4*>(packed+16U);x.qh[1]=*reinterpret_cast<const uint4*>(packed+32U);}
    };
    Chains<V> chains;
    const auto step=[&](unsigned b,const Block& cur) {
        const float d=scale_f32(static_cast<unsigned short>(cur.hd.x&65535U));
        const float dm=scale_f32(static_cast<unsigned short>(cur.hd.x>>16U));
#pragma unroll
        for(unsigned p=0;p<4U;++p) {
            const unsigned g0=2U*p,g1=g0+1U;
            uint4 w0[2],w1[2];
#pragma unroll
            for(unsigned k=0;k<2U;++k) {
                const uint4 q=cur.q[2U*p+k];
                w0[k]={q.x&mask,q.y&mask,q.z&mask,q.w&mask};
                w1[k]={(q.x>>4U)&mask,(q.y>>4U)&mask,(q.z>>4U)&mask,(q.w>>4U)&mask};
                if constexpr(Q5K) {
                    const uint4 qh=cur.qh[k];
                    w0[k].x|=((qh.x>>g0)&0x01010101U)<<4U;w0[k].y|=((qh.y>>g0)&0x01010101U)<<4U;
                    w0[k].z|=((qh.z>>g0)&0x01010101U)<<4U;w0[k].w|=((qh.w>>g0)&0x01010101U)<<4U;
                    w1[k].x|=((qh.x>>g1)&0x01010101U)<<4U;w1[k].y|=((qh.y>>g1)&0x01010101U)<<4U;
                    w1[k].z|=((qh.z>>g1)&0x01010101U)<<4U;w1[k].w|=((qh.w>>g1)&0x01010101U)<<4U;
                }
            }
            const unsigned plane=16U*b+4U*p;
            const wmma_v8i r0=group_dot(s.a[plane][i],operand(w0[0]),s.a[plane+1U][i],operand(w0[1]));
            const wmma_v8i r1=group_dot(s.a[plane+2U][i],operand(w1[0]),s.a[plane+3U][i],operand(w1[1]));
            float e0[8],e1[8],s0[8],s1[8];
            row_values<V>(&s.delta[8U*b+g0][8U*h],e0);row_values<V>(&s.delta[8U*b+g1][8U*h],e1);
            row_values<V>(&s.sum[8U*b+g0][8U*h],s0);row_values<V>(&s.sum[8U*b+g1][8U*h],s1);
            const float sc0=static_cast<float>(factor(cur.hd,g0,false)),sc1=static_cast<float>(factor(cur.hd,g1,false));
            const float mn0=static_cast<float>(factor(cur.hd,g0,true)),mn1=static_cast<float>(factor(cur.hd,g1,true));
#pragma unroll
            for(unsigned e=0;e<V;++e) {
                const float i0=sc0*group_float(r0[e]),i1=sc1*group_float(r1[e]);
                const float j0=mn0*s0[e],j1=mn1*s1[e];
                const float u0=d*i0,v0=dm*j0,t0=u0-v0;
                const float u1=d*i1,v1=dm*j1,t1=u1-v1;
                const float c0=e0[e]*t0,c1=e1[e]*t1;
                const float pair=c0+c1;
                chains.acc[p][e]=chains.acc[p][e]+pair;
            }
        }
    };
    // Two block buffers in turn: block b + 1 loads while block b computes.
    Block x[2];
    load(0U,x[0]);
#pragma unroll 1
    for(unsigned b=0;b<blocks;b+=2U) {
        if(b+1U<blocks)load(b+1U,x[1]);
        step(b,x[0]);
        if(b+1U<blocks) {
            if(b+2U<blocks)load(b+2U,x[0]);
            step(b+1U,x[1]);
        }
    }
    chains.fold(out);
}
// Gate/up of one CSR batch (<= 16 assignments) for Waves/2 16-output tiles: wave w runs matrix
// w%2 (gate, up) of tile w/2. Grid x = batch, y = output tile.
template<bool Q5K,unsigned Waves>
__global__ __launch_bounds__(32U*Waves) void project_prefill_wmma(const unsigned char* gate,const unsigned char* up,PrefillRows view,
    unsigned hidden,unsigned outputs) {
    constexpr unsigned Outs=8U*Waves,Groups=max_hidden/32U;
    const auto& csr=view.csr;
    if((csr.summary[0]&~projection_nonfinite)||blockIdx.x>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.x],count=csr.begins[blockIdx.x+1U]-begin,expert=csr.experts[blockIdx.x];
    __shared__ union { WmmaRowsStage<Groups> s; float tile[2][wmma_batch][Outs]; } lds;
    __shared__ unsigned row_of[wmma_batch],slot_of[wmma_batch];
    if(threadIdx.x<wmma_batch&&threadIdx.x<count){row_of[threadIdx.x]=csr.rows[begin+threadIdx.x];slot_of[threadIdx.x]=csr.slots[begin+threadIdx.x];}
    __syncthreads();
    wmma_stage_rows<Groups>(lds.s,count,hidden,32U*Waves,[&](unsigned a){return at(view.codes,row_of[a]);},
        [&](unsigned a){return at(view.deltas,row_of[a]);},[&](unsigned a){return at(view.sums,row_of[a]);});
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,wave=threadIdx.x/32U,i=lane&15U,h=lane>>4U,m=wave&1U,t=wave>>1U;
    const unsigned tile_base=blockIdx.y*Outs,out=tile_base+16U*t+i,blocks=hidden/256U,valid=(count+1U)/2U;
    constexpr unsigned block_bytes=Q5K?176U:144U;
    const auto* w=(m==0U?gate:up)+(static_cast<unsigned long long>(expert)*outputs+out)*blocks*block_bytes;
    float result[8]={};
    if(valid<=2U)gate_up_chains<Q5K,2U>(w,lds.s,blocks,i,h,result);
    else if(valid<=4U)gate_up_chains<Q5K,4U>(w,lds.s,blocks,i,h,result);
    else gate_up_chains<Q5K,8U>(w,lds.s,blocks,i,h,result);
    __syncthreads(); // every wave is done with the staged rows: the tile reuses them
#pragma unroll
    for(unsigned e=0;e<8U;++e)if(2U*e+h<count)lds.tile[m][2U*e+h][16U*t+i]=result[e];
    __syncthreads();
    // Serial down_activation_quantize (project_prefill's epilogue) for each staged (row, slot)
    // and 32-output group of the CTA, one wave per pair.
    constexpr unsigned Quads=Outs/32U;
    for(unsigned job=wave;job<count*Quads;job+=Waves) {
        const unsigned a=job/Quads,quad=job%Quads,tile_index=tile_base/32U+quad;
        const auto r=row_of[a],slot=slot_of[a];
        float* output=at(view.output,r);
        const unsigned feature=tile_index*32U+lane,at_slot=slot*2U*outputs+feature;
        const float g=lds.tile[0][a][32U*quad+lane],u=lds.tile[1][a][32U*quad+lane];
        const float hv=(g/(1.0F+expf(-g)))*u;
        output[at_slot+outputs]=u;
        const unsigned bad=static_cast<unsigned>(__ballot(!isfinite(g)||!isfinite(u)||!isfinite(hv)));
        if(bad!=0){output[at_slot]=g;if(lane==0){atomicOr(at(view.status,r),projection_nonfinite);atomicOr(csr.summary,projection_nonfinite);}continue;}
        float x=fabsf(hv);
        for(int stride=16;stride!=0;stride/=2) x=fmaxf(x,strix::qwen_dpp::xor32(x, stride));
        const bool tiny=x>0.0F && x<0x1p-119F;
        const float delta=x==0.0F || tiny?1.0F:divide_rne(x,127.0F);
        const float rounded=tiny?0.0F:rintf(divide_rne(hv,delta));
        const int code=static_cast<int>(fminf(127.0F,fmaxf(-127.0F,rounded)));
        int sum=code;
        for(int stride=16;stride!=0;stride/=2) sum+=strix::qwen_dpp::xor32(sum, stride);
        output[at_slot]=hv;
        at(view.down_codes,r)[slot*outputs+feature]=static_cast<signed char>(code);
        if(lane==0){at(view.down_deltas,r)[slot*(outputs/32U)+tile_index]=delta;at(view.down_sums,r)[slot*(outputs/32U)+tile_index]=sum;}
    }
}
// Raw bytes of one down block for this lane's output (Q5_1: d|m, qh, qs; Q8_0: nine dwords from
// the block's dword-aligned start: rows hold an even block count, so an even block starts at its
// scale and an odd one two bytes before it).
template<bool Q51> struct DownBlock;
template<> struct DownBlock<true> { uint2 head; Dword4 qs; };
template<> struct DownBlock<false> { Dword4 lo,hi; unsigned last; };
template<bool Q51> __device__ __forceinline__ DownBlock<Q51> down_block(const unsigned char* row_weights,unsigned group) {
    if constexpr(Q51) {
        const auto* block=row_weights+group*24U;
        return {*reinterpret_cast<const uint2*>(block),*reinterpret_cast<const Dword4*>(block+8U)};
    } else {
        const auto* p=row_weights+((34U*group)&~3U);
        return {reinterpret_cast<const Dword4*>(p)[0],reinterpret_cast<const Dword4*>(p)[1],reinterpret_cast<const unsigned*>(p)[8]};
    }
}
// WMMA B operands (K halves) and scales of a down block; Q5_1 value k = nibble | qh bit k << 4.
template<bool Q51>
__device__ __forceinline__ void down_operands(const DownBlock<Q51>& x,unsigned group,wmma_v4i& b0,wmma_v4i& b1,float& d,float& m) {
    if constexpr(Q51) {
        const unsigned q[4]{x.qs.x,x.qs.y,x.qs.z,x.qs.w};
        unsigned lo[4],hi[4];
#pragma unroll
        for(unsigned k=0;k<4U;++k) {
            const unsigned l=(x.head.y>>(4U*k))&15U,u=(x.head.y>>(16U+4U*k))&15U;
            lo[k]=(q[k]&0x0f0f0f0fU)|((l*0x00204081U)&0x01010101U)<<4U;
            hi[k]=((q[k]>>4U)&0x0f0f0f0fU)|((u*0x00204081U)&0x01010101U)<<4U;
        }
        b0=wmma_v4i{static_cast<int>(lo[0]),static_cast<int>(lo[1]),static_cast<int>(lo[2]),static_cast<int>(lo[3])};
        b1=wmma_v4i{static_cast<int>(hi[0]),static_cast<int>(hi[1]),static_cast<int>(hi[2]),static_cast<int>(hi[3])};
        d=scale_f32(static_cast<unsigned short>(x.head.x&65535U));
        m=scale_f32(static_cast<unsigned short>(x.head.x>>16U));
    } else {
        const unsigned odd=group&1U,select=odd?0x07060504U:0x05040302U;
        const unsigned q[9]{x.lo.x,x.lo.y,x.lo.z,x.lo.w,x.hi.x,x.hi.y,x.hi.z,x.hi.w,x.last};
        unsigned c[8];
#pragma unroll
        for(unsigned k=0;k<8U;++k)c[k]=__builtin_amdgcn_perm(q[k+1U],q[k],select);
        b0=wmma_v4i{static_cast<int>(c[0]),static_cast<int>(c[1]),static_cast<int>(c[2]),static_cast<int>(c[3])};
        b1=wmma_v4i{static_cast<int>(c[4]),static_cast<int>(c[5]),static_cast<int>(c[6]),static_cast<int>(c[7])};
        d=scale_f32(static_cast<unsigned short>((q[0]>>(odd?16U:0U))&65535U));
        m=0.0F;
    }
}
// Down chains of one wave for width = max_rows: groups ascending in phases of four (group
// 4k + j adds to chain j), two phase buffers in turn: the next phase loads while one computes.
template<bool Q51,unsigned V>
__device__ __forceinline__ void down_chains(const unsigned char* row_weights,const WmmaRowsStage<max_rows/32U>& s,unsigned i,unsigned h,
    float (&out)[8]) {
    constexpr unsigned Phases=max_rows/128U;
    struct Phase { DownBlock<Q51> block[4]; };
    const auto load=[&](unsigned phase,Phase& x) {
#pragma unroll
        for(unsigned j=0;j<4U;++j)x.block[j]=down_block<Q51>(row_weights,4U*phase+j);
    };
    Chains<V> chains;
    const auto step=[&](unsigned phase,const Phase& cur) {
#pragma unroll
        for(unsigned j=0;j<4U;++j) {
            const unsigned group=4U*phase+j;
            wmma_v4i b0,b1;float d,m;
            down_operands<Q51>(cur.block[j],j,b0,b1,d,m);
            const wmma_v8i r=group_dot(s.a[2U*group][i],b0,s.a[2U*group+1U][i],b1);
            float deltas[8],sums[8];
            row_values<V>(&s.delta[group][8U*h],deltas);
            if constexpr(Q51)row_values<V>(&s.sum[group][8U*h],sums);
#pragma unroll
            for(unsigned e=0;e<V;++e) {
                const float first_term=d*group_float(r[e]);
                float combined=first_term;
                if constexpr(Q51) {
                    const float correction=m*sums[e];
                    combined=first_term+correction;
                }
                const float contribution=deltas[e]*combined;
                chains.acc[j][e]=chains.acc[j][e]+contribution;
            }
        }
    };
    Phase x[2];
    load(0U,x[0]);
#pragma unroll 1
    for(unsigned phase=0;phase<Phases;phase+=2U) {
        if(phase+1U<Phases)load(phase+1U,x[1]);
        step(phase,x[0]);
        if(phase+1U<Phases) {
            if(phase+2U<Phases)load(phase+2U,x[0]);
            step(phase+1U,x[1]);
        }
    }
    chains.fold(out);
}
// Down of one CSR batch (<= 16 assignments) for Waves 16-row tiles of the hidden output, width =
// max_rows. Grid x = batch, y = row tile.
template<bool Q51,unsigned Waves>
__global__ __launch_bounds__(32U*Waves) void down_prefill_wmma(const unsigned char* weights,PrefillRows view,unsigned hidden,unsigned width) {
    constexpr unsigned Outs=16U*Waves,Groups=max_rows/32U;
    const auto& csr=view.csr;
    if(csr.summary[0]||blockIdx.x>=csr.summary[2])return;
    const auto begin=csr.begins[blockIdx.x],count=csr.begins[blockIdx.x+1U]-begin,expert=csr.experts[blockIdx.x];
    __shared__ WmmaRowsStage<Groups> s;
    __shared__ unsigned row_of[wmma_batch],slot_of[wmma_batch];
    if(threadIdx.x<wmma_batch&&threadIdx.x<count){row_of[threadIdx.x]=csr.rows[begin+threadIdx.x];slot_of[threadIdx.x]=csr.slots[begin+threadIdx.x];}
    __syncthreads();
    const unsigned groups=width/32U;
    wmma_stage_rows<Groups>(s,count,width,32U*Waves,[&](unsigned a){return at(view.down_codes,row_of[a])+slot_of[a]*width;},
        [&](unsigned a){return at(view.down_deltas,row_of[a])+slot_of[a]*groups;},
        [&](unsigned a){return at(view.down_sums,row_of[a])+slot_of[a]*groups;});
    __syncthreads();
    const unsigned lane=threadIdx.x&31U,wave=threadIdx.x/32U,i=lane&15U,h=lane>>4U,valid=(count+1U)/2U;
    constexpr unsigned block_bytes=Q51?24U:34U;
    const unsigned row=blockIdx.y*Outs+16U*wave+i;
    const auto* row_weights=weights+(static_cast<unsigned long long>(expert)*hidden+row)*groups*block_bytes;
    float result[8]={};
    if(valid<=2U)down_chains<Q51,2U>(row_weights,s,i,h,result);
    else if(valid<=4U)down_chains<Q51,4U>(row_weights,s,i,h,result);
    else down_chains<Q51,8U>(row_weights,s,i,h,result);
#pragma unroll
    for(unsigned e=0;e<8U;++e)if(2U*e+h<count) {
        const unsigned a=2U*e+h;
        at(view.rawdots,row_of[a])[slot_of[a]*hidden+row]=result[e];
    }
}
// Fast prefill profile (fast-prefill-f16wmma-v1, form 3): not the A8 contract. Weights are
// dequantized inside the kernels to FP16 (the FP32 value d*sc*q - dm*m, Q5_1 d*q + m, Q8_0 d*q,
// rounded once to nearest), activations are FP16, and v_wmma_f32_16x16x16_f16 accumulates over
// the whole K in FP32, with no per-group epilogue. Activations carry a power-of-two scale per
// row that puts the row's largest magnitude in [2^13, 2^14): exact for every value that stays
// normal (29 binades below the maximum) and undone exactly on the FP32 sums. Input x: one scale
// per prompt row (f16_prefill_prologue); h = silu(g)*u: gate/up writes it in FP32 and the largest |h|
// of each (row, slot), and down scales and rounds it while staging. A CSR batch holds at most 64
// assignments of one expert (M tile 64; waves whose 16 rows are all past the batch skip their
// WMMAs). Each (row, output) sums its K steps in the same order whatever batch it lands in, so
// results do not depend on the batch composition. A non-finite g, u or h fails the row closed
// (status 8), as the A8 quantizer does; down writes the raw slot dots of the unchanged merge.
// Scratch: the A8 views the profile does not use: sums[r][0] = the x scale exponent,
// down_sums[r][slot] = bits of max |h|, intermediates[r][slot*2*outputs + o] = h.
typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
constexpr unsigned f16_batch=64U,f16_pitch=72U; // M tile rows; LDS row pitch in halves (K step 64 + 8)
__device__ __forceinline__ unsigned f16_pair(float a,float b) {
    const _Float16 x=static_cast<_Float16>(a),y=static_cast<_Float16>(b);
    return static_cast<unsigned>(__builtin_bit_cast(unsigned short,x))|static_cast<unsigned>(__builtin_bit_cast(unsigned short,y))<<16U;
}
// Exponent s with peak * 2^s in [2^13, 2^14) (0 for a zero or non-finite max), clamped so 2^s and
// 2^-s are normal FP32 values.
__device__ __forceinline__ int f16_exponent(float peak) {
    if(!(peak>0.0F)||!isfinite(peak))return 0;
    int e;(void)frexpf(peak,&e); // peak in [2^(e-1), 2^e)
    return min(120,max(-120,14-e));
}
// 8 BF16 (one uint4) times scale as 8 FP16.
__device__ __forceinline__ uint4 f16_from_bf16(uint4 v,float scale) {
    const auto lo=[&](unsigned w){return __uint_as_float(w<<16U)*scale;};
    const auto hi=[&](unsigned w){return __uint_as_float(w&0xffff0000U)*scale;};
    return {f16_pair(lo(v.x),hi(v.x)),f16_pair(lo(v.y),hi(v.y)),f16_pair(lo(v.z),hi(v.z)),f16_pair(lo(v.w),hi(v.w))};
}
// 8 FP32 (the bits in two uint4) times scale as 8 FP16.
__device__ __forceinline__ uint4 f16_from_f32(uint4 a,uint4 b,float scale) {
    const auto x=[&](unsigned w){return __uint_as_float(w)*scale;};
    return {f16_pair(x(a.x),x(a.y)),f16_pair(x(a.z),x(a.w)),f16_pair(x(b.x),x(b.y)),f16_pair(x(b.z),x(b.w))};
}
// The four unsigned bytes of w as a*byte + b (one FP32 rounding), rounded to two FP16 pairs.
#include "moe_core/qwen_k_scalar_decode.inc"

// The four signed bytes of w times a, as two FP16 pairs.
__device__ __forceinline__ uint2 f16_scaled_s8(unsigned w,float a) {
    const int s=static_cast<int>(w);
    const float v0=a*static_cast<float>((s<<24)>>24),v1=a*static_cast<float>((s<<16)>>24);
    const float v2=a*static_cast<float>((s<<8)>>24),v3=a*static_cast<float>(s>>24);
    return {f16_pair(v0,v1),f16_pair(v2,v3)};
}
// Prologue of the FP16 forms, one CTA per prompt row: the A8 prologue's clear, route checks and
// row status, the row's input scale exponent and zeroed max |h| words. No A8 quantizer: a
// non-finite input reaches g and u and fails the row in the gate/up epilogue (status 8).
// a16 (MoE tune bit 4, else null): also the row's scaled FP16 image (f16_from_bf16 with the row's
// scale, the values gate/up would stage), so gate/up converts each prompt row once instead of once
// per (slot, output tile).
__global__ __launch_bounds__(256) void f16_prefill_prologue(PrefillRows v,unsigned hidden,unsigned top_k,unsigned experts,
    unsigned clear_words,uint4* a16) {
    const unsigned r=blockIdx.x,t=threadIdx.x;
    __shared__ unsigned route_bits;
    __shared__ int row_exponent;
    __shared__ float wave_max[8];
    if(t==0U)route_bits=0U;
    float* clear=at(v.clear,r);
    for(unsigned i=t;i<clear_words;i+=256U)clear[i]=0.0F;
    __syncthreads();
    if(t<top_k&&(at(v.routes,r)[t]>=experts||!isfinite(at(v.route_weights,r)[t])))atomicOr(&route_bits,invalid_route);
    if(t<top_k)at(v.down_sums,r)[t]=0;
    const auto* x=reinterpret_cast<const unsigned short*>(at(v.input,r));
    float m=0.0F;
    for(unsigned k=t;k<hidden;k+=256U)m=fmaxf(m,fabsf(__uint_as_float(static_cast<unsigned>(x[k])<<16U)));
    for(int stride=16;stride!=0;stride/=2)m=fmaxf(m,strix::qwen_dpp::xor32(m, stride));
    if((t&31U)==0U)wave_max[t/32U]=m;
    __syncthreads();
    if(t==0U) {
        for(unsigned w=1;w<8U;++w)m=fmaxf(m,wave_max[w]);
        row_exponent=f16_exponent(m);
        at(v.sums,r)[0]=row_exponent;
        *at(v.status,r)=route_bits;
    }
    if(!a16)return;
    __syncthreads();
    const float scale=ldexpf(1.0F,row_exponent);
    const auto* x16=reinterpret_cast<const uint4*>(x);
    uint4* out=a16+static_cast<std::size_t>(r)*(hidden/8U);
    for(unsigned k=t;k<hidden/8U;k+=256U)out[k]=f16_from_bf16(x16[k],scale);
}
// LDS-only barrier: the staged tiles are complete (lgkmcnt) but the next step's global loads stay
// in flight (__syncthreads would also wait for them, as its workgroup fence does).
#include "moe_core/qwen_grouped_tiles.inc"
#include "moe_core/hip_moe_prefill_core2.inc"
// qpf2k core2 bit 6 (gate/up only): one CTA owns two consecutive 128-assignment CSR batches of one expert (256 rows) x
// 64 outputs of gate and up (128 B columns), so each dequantized weight tile serves up to 256 rows; same per-wave tile
// (64 rows x 16 gate + 16 up columns), operands and K order as the base. Otherwise the F16Tile shape.
template<unsigned Tune,unsigned Core> struct F16TileX {
    static constexpr bool Wide=(Core&64U)!=0U;
    static constexpr unsigned Batch=Wide?256U:F16Tile<Tune>::Batch,Columns=Wide?128U:F16Tile<Tune>::Columns;
    static constexpr bool Table=true;
    static constexpr unsigned LTune=Tune;
    static constexpr unsigned Waves=Batch*Columns/2048U,Threads=32U*Waves,Groups=Columns/32U;
    static constexpr unsigned TPC=Threads/Columns,AChunks=Batch*8U/Threads;
};
template<bool Wide> struct f16_qv { using type=uint4; };
template<> struct f16_qv<false> { using type=uint2; };
__device__ __forceinline__ uint2 f16_load8(const void* p) {
    typedef unsigned f16_u32x2 __attribute__((ext_vector_type(2)));
    const f16_u32x2 v=*reinterpret_cast<const __attribute__((address_space(1))) f16_u32x2*>(reinterpret_cast<std::uintptr_t>(p));
    return uint2{v.x,v.y};
}
// The K loop shared by both projections: two register stages (or one with bit 0), LDS-only barriers.
template<unsigned Tune,class Regs,class Fetch,class Stage,class Step>
__device__ __forceinline__ void f16_pipeline(unsigned steps,const Fetch& fetch,const Stage& stage,const Step& step_fn) {
    if constexpr(Tune&1U) {
        Regs r;
        fetch(0U,r);
        for(unsigned step=0;step<steps;++step) {
            if(step)f16_barrier();
            stage(step,r,0U);
            f16_barrier();
            if(step+1U<steps)fetch(step+1U,r);
            step_fn(0U);
        }
    } else {
        Regs r0,r1;
        fetch(0U,r0);fetch(1U,r1); // steps is even
#pragma unroll 1
        for(unsigned step=0;step<steps;step+=2U) {
            if(step)f16_barrier();
            stage(step,r0,0U);
            f16_barrier();
            if(step+2U<steps)fetch(step+2U,r0);
            step_fn(0U);
            f16_barrier();
            stage(step+1U,r1,0U);
            f16_barrier();
            if(step+3U<steps)fetch(step+3U,r1);
            step_fn(0U);
        }
    }
}
template<bool Q5K,unsigned Tune,unsigned Core=0U>
__global__ __launch_bounds__((F16TileX<Tune,Core>::Threads),prefill_core2_min_waves(Tune,Core)) void project_prefill_f16x(const unsigned char* gate,const unsigned char* up,PrefillRows view,
    unsigned hidden,unsigned outputs,const uint4* a16) {
#define STRIX_F16X_BATCH_TILE const unsigned batch=blockIdx.y,tile=blockIdx.x;
#define STRIX_F16X_LDS(S,R) __shared__ __align__(16) S sb[1]; __shared__ R rows;
#include "moe_core/hip_moe_f16x_project_body.inc"
#undef STRIX_F16X_BATCH_TILE
#undef STRIX_F16X_LDS
}
inline int& direct_b_override(){static int v=-1;return v;}
inline bool direct_b_requested(){
    static const bool env=[]{const char* v=std::getenv("STRIX_QPF2K_MOE_DIRECT_B");return v&&v[0]=='1'&&v[1]=='\0';}();
    return direct_b_override()<0?env:direct_b_override()==1;
}
inline bool& direct_b_ran(){static thread_local bool v=false;return v;}
inline const char*& direct_b_reason(){static thread_local const char* v="off";return v;}
#include "hip_moe_direct_b.inc"
template<bool PairedPlanes>
__global__ __launch_bounds__(32) void direct_b_fragment_probe(const unsigned char* raw,const unsigned char* packed,
    unsigned long long* stats) {
    const unsigned unit=blockIdx.x/(PairedPlanes?2U:1U),matrix_of_wave=blockIdx.x%2U;
    const unsigned lane=threadIdx.x,i=lane%16U,h=lane/16U;
    const auto* prepared=packed+static_cast<std::size_t>(unit*2U+(PairedPlanes?matrix_of_wave:h))*2304U;
    const uint4 head=f16_load(prepared+16U*i);
    const bool fast=__all(QwenKDecodeOps::f16_affine_exact(head.x&65535U,63U*15U,head.x>>16U,63U));
    unsigned long long mismatch=0,first=~0ULL;
    for(unsigned step=0;step<4U;++step)for(unsigned kk=0;kk<4U;++kk) {
        const unsigned group=2U*step+kk/2U;
        const auto q=f16_load(prepared+256U+(group*16U+i)*16U);
        const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(head.x)));
        const float dm=__half2float(__ushort_as_half(static_cast<unsigned short>(head.x>>16U)));
        const float da=d*static_cast<float>(factor(head,group,false)),db=-(dm*static_cast<float>(factor(head,group,true)));
        f16x16 b0,b1;
        if constexpr(PairedPlanes)direct_b_decode_pair(q,h,da,db,fast,b0,b1);
        else direct_b_exchange(direct_b_decode(q,kk&1U,da,db,fast),h,b0,b1);
        for(unsigned which=0;which<(PairedPlanes?1U:2U);++which) {
            const unsigned matrix=PairedPlanes?matrix_of_wave:which;
            // Cast the complete fragment before indexing. A bit_cast directly
            // on an FP16 vector subscript can compile as element 0 for every e.
            typedef unsigned short Bits16 __attribute__((ext_vector_type(16)));
            const Bits16 operand_bits=__builtin_bit_cast(Bits16,(PairedPlanes?(kk&1U):matrix)?b1:b0);
            const auto* original=raw+static_cast<std::size_t>(unit*2U+matrix)*2304U+144U*i;
            const uint4 rh=f16_load(original);
            const float rd=__half2float(__ushort_as_half(static_cast<unsigned short>(rh.x)));
            const float rm=__half2float(__ushort_as_half(static_cast<unsigned short>(rh.x>>16U)));
            const float a=rd*static_cast<float>(factor(rh,group,false)),b=-(rm*static_cast<float>(factor(rh,group,true)));
            for(unsigned e=0;e<16U;++e) {
                const unsigned k=step*64U+kk*16U+e;
                const unsigned code=(original[16U+(k/64U)*32U+k%32U]>>(4U*((k/32U)%2U)))&15U;
                const unsigned short expected=static_cast<unsigned short>(f16_pair(fmaf(a,static_cast<float>(code),b),0.0F));
                const unsigned actual=operand_bits[e];
                mismatch+=expected!=actual;
                if(expected!=actual) {
                    const auto coordinate=((((static_cast<unsigned long long>(unit)*2U+matrix)*16U+i)*2U+h)*16U+step*4U+kk)*16U+e;
                    // One atomic carries both the coordinate and its values:
                    // the first failure stays coherent regardless of CTA order.
                    first=min(first,(coordinate<<32U)|(static_cast<unsigned long long>(expected)<<16U)|actual);
                }
            }
        }
    }
    atomicAdd(stats,mismatch);atomicAdd(stats+1,fast?1ULL:0ULL);atomicAdd(stats+2,fast?0ULL:1ULL);
    atomicMin(stats+3,first);
}
template<bool Q51,unsigned Tune,unsigned Core=0U>
__global__ __launch_bounds__(F16Tile<Tune>::Threads,prefill_core2_min_waves(Tune,Core)) void down_prefill_f16x(const unsigned char* weights,PrefillRows view,unsigned hidden,unsigned width,FuseArgT<Core> fuse) {
#define STRIX_F16X_BATCH_TILE const unsigned batch=blockIdx.y,tile=blockIdx.x;
#define STRIX_F16X_LDS(S,R) __shared__ __align__(16) S sb[1]; __shared__ R rows;
#define STRIX_F16X_DOWN_GUARD if(csr.summary[0]||batch>=csr.summary[2])return;
#include "moe_core/hip_moe_f16x_down_body.inc"
#undef STRIX_F16X_BATCH_TILE
#undef STRIX_F16X_LDS
#undef STRIX_F16X_DOWN_GUARD
}
#include "moe_core/hip_moe_gsq_weights.inc"
#include "moe_core/hip_moe_gsq_f16_prefill.inc"
#include "moe_core/hip_moe_gsq_wmma_prefill.inc"
#include "hip_moe_prefill_route.inc"
#include "moe_core/hip_moe_prefill_g3.inc"
#include "moe_core/hip_moe_prefill_p1.inc"
#include "hip_moe_route_bf16.inc"
#include "moe_core/hip_moe_prefill_da16.inc"
#include "hip_moe_prefill_tile64.inc"
#include "hip_moe_prefill_tiers.inc"
#include "moe_core/hip_moe_prefill_p1da16.inc"
#include "hip_moe_prefill_small.inc"
// MoE tune bit 4: the FP16 images of the prompt rows (configured maximum rows x max_hidden halves, reserved at setup).
HipPrefillPool& f16_prefill_a16_pool(){static HipPrefillPool pool;return pool;}
uint4* f16_prefill_a16(bool setup = false) {
    auto& pool = f16_prefill_a16_pool();
    return static_cast<uint4*>(pool.get(static_cast<std::size_t>(prefill_pool_rows())*max_hidden*2U,
        "FP16 prefill A image allocation",setup));
}
// Bit 6 self-test: random FP16 scale pairs (every binade, signs, zeros, subnormals, non-finite) and codes; where the
// check admits the short path, its FP16 words must equal the base conversion's for every code. counts: tested, admitted,
// mismatches for K-quant (a = d*sc, b = -dmin*m) and Q5_1 (d*q + m); the last three count the rejected Q8_0 offset form
// (d*(q + 128) - 128*d), which differs from the base in the sign of zero, and are informational.
__device__ __forceinline__ unsigned long long selftest_mix(unsigned long long x) {
    x^=x>>33U;x*=0xff51afd7ed558ccdULL;x^=x>>33U;x*=0xc4ceb9fe1a85ec53ULL;x^=x>>33U;return x;
}
__device__ __forceinline__ unsigned selftest_f16(unsigned long long r) {
    const unsigned pick=static_cast<unsigned>(r&15U);
    if(pick==0U)return static_cast<unsigned>(r>>8U)&0x83ffU;          // zero / subnormal
    if(pick==1U)return 0x7c00U|(static_cast<unsigned>(r>>8U)&0x83ffU); // inf / nan
    if(pick<8U)return (static_cast<unsigned>(r>>8U)&0x8000U)|((3U+static_cast<unsigned>(r>>16U)%7U)<<10U)|(static_cast<unsigned>(r>>24U)&1023U);
    return static_cast<unsigned>(r>>8U)&0xffffU;                      // any bits
}
__global__ void fast_dequant_selftest_kernel(unsigned long long seed,unsigned long long* counts) {
    const unsigned long long id=blockIdx.x*static_cast<unsigned long long>(blockDim.x)+threadIdx.x;
    unsigned long long r=selftest_mix(seed^(id*0x9e3779b97f4a7c15ULL));
    unsigned long long local[9]={};
    for(unsigned it=0;it<16U;++it) {
        r=selftest_mix(r+it);
        const unsigned hx=selftest_f16(r),hy=selftest_f16(selftest_mix(r^1U));
        const unsigned sc=static_cast<unsigned>(r>>40U)&63U,mn=static_cast<unsigned>(r>>46U)&63U;
        const bool q5=(r>>52U)&1U;
        const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(hx))),dm=__half2float(__ushort_as_half(static_cast<unsigned short>(hy)));
        // K-quant sub-block.
        {
            const float a=d*static_cast<float>(sc),b=-(dm*static_cast<float>(mn));
            ++local[0];
            if(QwenKDecode::f16_affine_exact(hx,63U*(q5?31U:15U),hy,63U)) {
                ++local[1];
                for(unsigned q=0;q<(q5?32U:16U);q+=4U) {
                    const unsigned w=q|(q+1U)<<8U|(q+2U)<<16U|(q+3U)<<24U;
                    const uint2 x=QwenKScalarDecode::f16_affine_u8(w,a,b),y=QwenKDecode::f16_affine_u8_fast(w,a*16777216.0F,b);
                    local[2]+=(x.x!=y.x)+(x.y!=y.y);
                }
            }
        }
        // Q5_1 block.
        {
            ++local[3];
            if(QwenKDecode::f16_affine_exact(hx,31U,hy,1U)) {
                ++local[4];
                for(unsigned q=0;q<32U;q+=4U) {
                    const unsigned w=q|(q+1U)<<8U|(q+2U)<<16U|(q+3U)<<24U;
                    const uint2 x=QwenKScalarDecode::f16_affine_u8(w,d,dm),y=QwenKDecode::f16_affine_u8_fast(w,d*16777216.0F,dm);
                    local[5]+=(x.x!=y.x)+(x.y!=y.y);
                }
            }
        }
        // Q8_0: every signed byte.
        {
            ++local[6];
            if(((hx>>10U)&31U)!=31U) {
                ++local[7];
                for(unsigned q=0;q<256U;q+=4U) {
                    const unsigned w=q|(q+1U)<<8U|(q+2U)<<16U|(q+3U)<<24U;
                    const uint2 x=f16_scaled_s8(w,d),y=QwenKDecode::f16_affine_u8_fast(w^0x80808080U,d*16777216.0F,-128.0F*d);
                    local[8]+=(x.x!=y.x)+(x.y!=y.y);
                }
            }
        }
    }
    for(unsigned k=0;k<9U;++k)atomicAdd(counts+k,local[k]);
}
bool overlap(DeviceBytes a,DeviceBytes b){return a.address<b.address+b.bytes&&b.address<a.address+a.bytes;}
template<bool Q51,unsigned Tune> void prefill_route_i8_resources() {
    static const bool admitted=[]{
        constexpr unsigned Threads=F16Tile<Tune>::Threads;
        const auto base=down_prefill_f16x_da16_certificate_bf16<Q51,Tune>;
        const auto compact=down_prefill_f16x_da16_certificate_i8<Q51,Tune>;
        hipFuncAttributes a{},b{};int old_blocks=0,new_blocks=0;
        check(hipFuncGetAttributes(&a,reinterpret_cast<const void*>(base)),"route BF16 resource attributes");
        check(hipFuncGetAttributes(&b,reinterpret_cast<const void*>(compact)),"route I8 resource attributes");
        check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&old_blocks,base,Threads,0),"route BF16 occupancy");
        check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&new_blocks,compact,Threads,0),"route I8 occupancy");
        const bool ok=old_blocks>0&&new_blocks>=old_blocks&&b.localSizeBytes<=a.localSizeBytes;
        std::fprintf(stderr,"moe-route-i8-resource tune=%u q51=%u vgpr=%d->%d lds=%zu->%zu local=%zu->%zu blocks=%d->%d %s\n",
            Tune,static_cast<unsigned>(Q51),a.numRegs,b.numRegs,a.sharedSizeBytes,b.sharedSizeBytes,a.localSizeBytes,b.localSizeBytes,
            old_blocks,new_blocks,ok?"ADMIT":"BEST-ALREADY");
        return ok;
    }();
    if(!admitted)throw std::runtime_error("BEST-ALREADY: MoE route I8 drops occupancy or increases private storage; restore route BF16");
}
void prefill_resources_at(const void* kernel,const char* name,unsigned tune,unsigned threads) {
    static const bool enabled=[]{const char* v=std::getenv("STRIX_INTPREFILL_RESOURCES");return v&&v[0]=='1';}();
    if(!enabled)return;
    hipFuncAttributes attributes{};int blocks=0;
    check(hipFuncGetAttributes(&attributes,kernel),"prefill resource attributes");
    check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,static_cast<int>(threads),0),"prefill resident block query");
    std::fprintf(stderr,"intprefill-resource %s tune=%u threads=%u vgpr=%d lds=%zu local=%zu resident_blocks_per_mp=%d\n",
        name,tune,threads,attributes.numRegs,attributes.sharedSizeBytes,attributes.localSizeBytes,blocks);
}
template<auto Kernel,unsigned Threads> void prefill_resources(const char* name,unsigned tune) {
    prefill_resources_at(reinterpret_cast<const void*>(Kernel),name,tune,Threads);
}
// Bound resource growth before publishing engagement. Query only for requested
// prepared consumers; OFF never performs an additional runtime resource query.
template<unsigned Tune,bool DA16> bool direct_b_resource_admitted() {
    static const bool admitted=[]{
        hipFuncAttributes a{};
        check(hipFuncGetAttributes(&a,reinterpret_cast<const void*>(direct_b_kernel<Tune,DA16>())),"direct B resources");
        // Split gate/up waves keep tune 82's eight row16 accumulators within
        // the same register ceiling as the four paired row16 tiles at 80/88.
        constexpr int ceiling=160;
        return a.numRegs<=ceiling && a.sharedSizeBytes<=20000U && a.localSizeBytes==0U;
    }();
    return admitted;
}
void range(DeviceBytes v){if(!v.address||!v.bytes||v.bytes>UINTPTR_MAX-v.address)throw std::invalid_argument("A8 CSR gate/up owner range is invalid");}
struct Prepared {DeviceRows rows;DeviceCsr csr;};
Prepared prepare(const Request& request,std::span<const Bindings> rows,const q51_known_rows::Workspace& workspace,hipStream_t stream) {
    if(rows.size()!=2U&&rows.size()!=4U&&rows.size()!=8U&&(rows.size()<9U||rows.size()>max_union_rows))
        throw std::invalid_argument("A8 CSR gate/up requires R2/R4/R8 (or a cohort union of 9..32 rows)");
    // The bound of the all-rows staging form (project_csr); cohort unions run the window kernels only.
    if(rows.size()<=8U&&staged_activation_bytes(static_cast<unsigned>(rows.size()))>65536U)
        throw std::invalid_argument("A8 CSR gate/up staging exceeds bounded LDS budget");
    const auto layout=q51_known_rows::scratch_layout(static_cast<unsigned>(rows.size()),request.shape.experts,request.shape.top_k);
    const DeviceBytes csr{workspace.memory.arena_base,workspace.memory.arena_bytes};range(csr);
    if(!workspace.owner||!workspace.memory.execution_identity||!workspace.memory.allocation_generation||
        csr.address%256U||csr.bytes!=layout.bytes)throw std::invalid_argument("A8 CSR gate/up retained CSR owner differs");
    if(!stream||stream==hipStreamLegacy||stream==hipStreamPerThread)throw std::invalid_argument("A8 CSR gate/up admits explicit eager streams only");
    hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;check(hipStreamIsCapturing(stream,&capture),"A8 CSR gate/up capture query");
    if(capture!=hipStreamCaptureStatusNone)throw std::invalid_argument("A8 CSR gate/up admits explicit eager streams only");
    Prepared out;std::array<DeviceBytes,5U*max_union_rows> writes{};std::array<DeviceBytes,4U*max_union_rows> reads{};
    for(unsigned r=0;r<rows.size();++r) {
        const auto& b=rows[r];validate(request,b);
        if(b.gate.address!=rows.front().gate.address||b.gate.bytes!=rows.front().gate.bytes||
           b.up.address!=rows.front().up.address||b.up.bytes!=rows.front().up.bytes)
            throw std::invalid_argument("A8 CSR gate/up weight owners differ");
        const std::array mutable_views{b.codes,b.deltas,b.sums,b.status,b.output};
        const std::array readonly_views{b.input,b.gate,b.up,b.routes};
        for(unsigned v=0;v<5U;++v)writes[5U*r+v]=mutable_views[v];
        for(unsigned v=0;v<4U;++v)reads[4U*r+v]=readonly_views[v];
        out.rows.codes[r]=reinterpret_cast<const signed char*>(b.codes.address);out.rows.deltas[r]=reinterpret_cast<const float*>(b.deltas.address);
        out.rows.sums[r]=reinterpret_cast<const int*>(b.sums.address);out.rows.routes[r]=reinterpret_cast<const unsigned*>(b.routes.address);
        out.rows.status[r]=reinterpret_cast<unsigned*>(b.status.address);out.rows.output[r]=reinterpret_cast<float*>(b.output.address);
    }
    for(unsigned w=0;w<5U*rows.size();++w) {
        range(writes[w]);if(overlap(csr,writes[w]))throw std::invalid_argument("A8 CSR gate/up CSR aliases row storage");
        for(unsigned previous=0;previous<w;++previous)if(overlap(writes[w],writes[previous]))throw std::invalid_argument("A8 CSR gate/up mutable row owners alias");
        for(unsigned r=0;r<4U*rows.size();++r){range(reads[r]);if(overlap(writes[w],reads[r]))throw std::invalid_argument("A8 CSR gate/up writable/read-only owners alias");}
    }
    for(unsigned r=0;r<4U*rows.size();++r)if(overlap(csr,reads[r]))throw std::invalid_argument("A8 CSR gate/up CSR aliases read-only owner");
    const auto ptr=[&](std::uint64_t offset){return reinterpret_cast<unsigned*>(csr.address+offset);};
    out.csr={ptr(layout.header),ptr(layout.offsets),ptr(layout.touched),ptr(layout.rows),ptr(layout.slots),ptr(layout.batch_experts),ptr(layout.batch_begins)};
    return out;
}
} // namespace
void enqueue_window(const Request& request,std::span<const Bindings> rows,const q51_known_rows::Workspace& workspace,
    const WindowRows& window,hipStream_t stream,unsigned active) {
    auto p=prepare(request,rows,workspace,stream);
    const DeviceBytes csr{workspace.memory.arena_base,workspace.memory.arena_bytes};
    const bool a8=window.down_type!=0U,q5k=request.gate_type==13U,q20=request.gate_type==42U,
        iq=gsq_a8::gate_type(request.gate_type)&&!q20;
    if((a8&&window.down_type!=7U&&window.down_type!=8U&&!gsq_a8::down_type(window.down_type))||((q5k||q20||iq)&&!a8))
        throw std::invalid_argument("A8 window down/gate format combination is invalid");
    const auto count=static_cast<unsigned>(rows.size()),groups=request.shape.rows/32U;
    PrologueRows v;
    for(unsigned r=0;r<count;++r) {
        const auto& b=rows[r];const auto weights=window.route_weights[r],clear=window.clear[r];
        range(weights);range(clear);
        if(weights.bytes<4ULL*request.shape.top_k||clear.bytes!=window.clear[0].bytes||clear.bytes%4U||clear.address%4U||
           clear.bytes/4U>UINT32_MAX||overlap(clear,csr)||overlap(weights,csr))
            throw std::invalid_argument("A8 window row extent is invalid");
        for(unsigned q=0;q<count;++q)for(const auto view:{rows[q].codes,rows[q].deltas,rows[q].sums,rows[q].status,
            rows[q].output,rows[q].input,rows[q].routes,window.route_weights[q]})if(overlap(clear,view))
            throw std::invalid_argument("A8 window cleared region aliases row storage");
        if(a8) {
            const std::array down{window.down_codes[r],window.down_deltas[r],window.down_sums[r],window.rawdots[r]};
            const std::array<std::uint64_t,4> need{static_cast<std::uint64_t>(request.shape.top_k)*request.shape.rows,
                4ULL*request.shape.top_k*groups,4ULL*request.shape.top_k*groups,4ULL*request.shape.top_k*request.shape.hidden};
            for(unsigned i=0;i<4U;++i) {
                range(down[i]);
                if(down[i].bytes<need[i]||down[i].address%16U||overlap(down[i],csr)||overlap(down[i],window.down_weights))
                    throw std::invalid_argument("A8 window down row view is invalid");
                for(unsigned q=0;q<count;++q)for(const auto view:{rows[q].codes,rows[q].deltas,rows[q].sums,rows[q].status,
                    rows[q].output,rows[q].input,rows[q].routes,rows[q].gate,rows[q].up,window.clear[q],window.route_weights[q]})
                    if(overlap(down[i],view))throw std::invalid_argument("A8 window down row view aliases row storage");
                for(unsigned q=0;q<=r;++q)for(unsigned j=0;j<4U;++j)if((q<r||j<i)&&overlap(down[i],
                    std::array{window.down_codes[q],window.down_deltas[q],window.down_sums[q],window.rawdots[q]}[j]))
                    throw std::invalid_argument("A8 window down row views alias");
            }
            p.rows.down_codes[r]=reinterpret_cast<signed char*>(window.down_codes[r].address);
            p.rows.down_deltas[r]=reinterpret_cast<float*>(window.down_deltas[r].address);
            p.rows.down_sums[r]=reinterpret_cast<int*>(window.down_sums[r].address);
            p.rows.rawdots[r]=reinterpret_cast<float*>(window.rawdots[r].address);
        }
        v.input[r]=reinterpret_cast<const void*>(b.input.address);v.routes[r]=p.rows.routes[r];
        v.route_weights[r]=reinterpret_cast<const float*>(weights.address);v.codes[r]=reinterpret_cast<signed char*>(b.codes.address);
        v.deltas[r]=reinterpret_cast<float*>(b.deltas.address);v.sums[r]=reinterpret_cast<int*>(b.sums.address);
        v.status[r]=p.rows.status[r];v.clear[r]=reinterpret_cast<float*>(clear.address);
    }
    if(a8) {
        range(window.down_weights);
        if(window.down_weights.bytes!=static_cast<std::uint64_t>(request.shape.experts)*request.shape.hidden*
               down_row_bytes(window.down_type,request.shape.rows)||
           request.shape.rows>max_rows||request.shape.rows%128U||request.shape.hidden%64U||overlap(window.down_weights,csr))
            throw std::invalid_argument("A8 window down weights/geometry are invalid");
    }
    v.csr={p.csr.summary,p.csr.offsets,p.csr.touched,p.csr.rows,p.csr.slots,p.csr.experts,p.csr.begins};
    const bool step16 = count==16U && cohort_step_rows()==16U;
    static const unsigned r16_project_g=[] {
        const char* value=std::getenv("STRIX_MOE_R16_PROJECT_G");
        if(!value)return strix::cohort_auto()?4U:16U;
        if(value[0]=='1' && value[1]=='6' && value[2]=='\0')return 16U;
        if((value[0]=='8' && value[1]=='\0'))return 8U;
        if((value[0]=='4' && value[1]=='\0'))return 4U;
        throw std::invalid_argument("STRIX_MOE_R16_PROJECT_G must be 4, 8 or 16");
    }();
    enqueue_prologue(request,v,count,static_cast<unsigned>(window.clear[0].bytes/4U),stream,active,step16?16U:8U);
    const auto* gate=reinterpret_cast<const unsigned char*>(rows.front().gate.address);
    const auto* up=reinterpret_cast<const unsigned char*>(rows.front().up.address);
    const dim3 project_grid(request.shape.rows/32U,count*request.shape.top_k);
#define STRIX_A8_WINDOW_PROJECT(G,Q5,A8) \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(project_window<G,Q5,A8>),project_grid,dim3(256),0,stream,gate,up,p.rows,p.csr, \
        count,request.shape.hidden,request.shape.rows)
    // STRIX_MOE_GATEUP_CHUNK=5: half the Q4_K blocks in registers at a time (same sums).
    static const bool half_chunk=[]{const char* v=std::getenv("STRIX_MOE_GATEUP_CHUNK");return v && v[0]=='5';}();
    const auto chunked=[&]<unsigned G>(){
        hipLaunchKernelGGL(HIP_KERNEL_NAME(project_window<G,false,true,5U>),project_grid,dim3(256),0,stream,gate,up,p.rows,p.csr,
            count,request.shape.hidden,request.shape.rows);
    };
    const auto iq_project=[&]<std::uint32_t T>(){
        if(step16)hipLaunchKernelGGL(HIP_KERNEL_NAME(project_window_iq<T,16U>),project_grid,dim3(256),0,stream,gate,up,p.rows,p.csr,
            count,request.shape.hidden,request.shape.rows);
        else if(count==2U)hipLaunchKernelGGL(HIP_KERNEL_NAME(project_window_iq<T,2U>),project_grid,dim3(256),0,stream,gate,up,p.rows,p.csr,
            count,request.shape.hidden,request.shape.rows);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_window_iq<T,4U>),project_grid,dim3(256),0,stream,gate,up,p.rows,p.csr,
            count,request.shape.hidden,request.shape.rows);
    };
    if(request.gate_type==gsq_a8::type_iq2_s)iq_project.template operator()<gsq_a8::type_iq2_s>();
    else if(request.gate_type==gsq_a8::type_iq3_xxs)iq_project.template operator()<gsq_a8::type_iq3_xxs>();
    else if(request.gate_type==gsq_a8::type_iq3_s)iq_project.template operator()<gsq_a8::type_iq3_s>();
    else if(request.gate_type==gsq_a8::type_iq4_xs)iq_project.template operator()<gsq_a8::type_iq4_xs>();
    else if(q20){
        if(step16)hipLaunchKernelGGL(HIP_KERNEL_NAME(project_window_q20<16U>),project_grid,dim3(256),0,stream,gate,up,p.rows,p.csr,
            count,request.shape.hidden,request.shape.rows);
        else if(count==2U)hipLaunchKernelGGL(HIP_KERNEL_NAME(project_window_q20<2U>),project_grid,dim3(256),0,stream,gate,up,p.rows,p.csr,
            count,request.shape.hidden,request.shape.rows);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_window_q20<4U>),project_grid,dim3(256),0,stream,gate,up,p.rows,p.csr,
            count,request.shape.hidden,request.shape.rows);
    }
    else if(step16 && r16_project_g!=16U) {
        const auto split_project=[&]<unsigned G,bool Q5,bool A8>() {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(project_window_split<G,Q5,A8>),project_grid,dim3(256),0,stream,
                gate,up,p.rows,p.csr,count,request.shape.hidden,request.shape.rows);
        };
        const auto split_width=[&]<unsigned G>() {
            if(!a8)split_project.template operator()<G,false,false>();
            else if(q5k)split_project.template operator()<G,true,true>();
            else split_project.template operator()<G,false,true>();
        };
        if(r16_project_g==8U)split_width.template operator()<8U>();
        else split_width.template operator()<4U>();
    }
    else if(!a8){if(step16){STRIX_A8_WINDOW_PROJECT(16U,false,false);}else if(count==2U){STRIX_A8_WINDOW_PROJECT(2U,false,false);}else{STRIX_A8_WINDOW_PROJECT(4U,false,false);}}
    else if(!q5k && half_chunk){if(step16)chunked.template operator()<16U>();else if(count==2U)chunked.template operator()<2U>();else chunked.template operator()<4U>();}
    else if(!q5k){if(step16){STRIX_A8_WINDOW_PROJECT(16U,false,true);}else if(count==2U){STRIX_A8_WINDOW_PROJECT(2U,false,true);}else{STRIX_A8_WINDOW_PROJECT(4U,false,true);}}
    else {if(step16){STRIX_A8_WINDOW_PROJECT(16U,true,true);}else if(count==2U){STRIX_A8_WINDOW_PROJECT(2U,true,true);}else{STRIX_A8_WINDOW_PROJECT(4U,true,true);}}
#undef STRIX_A8_WINDOW_PROJECT
    check(hipGetLastError(),"A8 window shared gate/up projection");
    if(!a8)return;
    const auto* down=reinterpret_cast<const unsigned char*>(window.down_weights.address);
    const dim3 down_grid(request.shape.hidden/64U,count*request.shape.top_k);
#define STRIX_A8_WINDOW_DOWN(R,Q51) \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(down_window<R,Q51>),down_grid,dim3(256),0,stream,down,p.rows,p.csr, \
        count,request.shape.hidden,request.shape.rows)
    // Opt-in R16 stages a whole expert batch; all other unions retain the R8 batch cap.
    if(window.down_type==gsq_a8::type_iq4_nl){
        const auto iq_down=[&]<unsigned R>(){hipLaunchKernelGGL(HIP_KERNEL_NAME(down_window_iq<gsq_a8::type_iq4_nl,R>),down_grid,dim3(256),0,
            stream,down,p.rows,p.csr,count,request.shape.hidden,request.shape.rows);};
        if(step16)iq_down.template operator()<16U>();else if(count==2U)iq_down.template operator()<2U>();else if(count==4U)iq_down.template operator()<4U>();
        else iq_down.template operator()<8U>();
    }
    else if(window.down_type==42U){
        const auto q20_down=[&]<unsigned R>(){hipLaunchKernelGGL(HIP_KERNEL_NAME(down_window_q20<R>),down_grid,dim3(256),0,stream,
            down,p.rows,p.csr,count,request.shape.hidden,request.shape.rows);};
        if(step16)q20_down.template operator()<16U>();else if(count==2U)q20_down.template operator()<2U>();else if(count==4U)q20_down.template operator()<4U>();
        else q20_down.template operator()<8U>();
    }
    else if(window.down_type==7U){if(step16){STRIX_A8_WINDOW_DOWN(16U,true);}else if(count==2U){STRIX_A8_WINDOW_DOWN(2U,true);}else if(count==4U){STRIX_A8_WINDOW_DOWN(4U,true);}else{STRIX_A8_WINDOW_DOWN(8U,true);}}
    else {if(step16){STRIX_A8_WINDOW_DOWN(16U,false);}else if(count==2U){STRIX_A8_WINDOW_DOWN(2U,false);}else if(count==4U){STRIX_A8_WINDOW_DOWN(4U,false);}else{STRIX_A8_WINDOW_DOWN(8U,false);}}
#undef STRIX_A8_WINDOW_DOWN
    check(hipGetLastError(),"A8 window shared integer down");
}
PrefillKernelForm prefill_kernel_form(const Request& request,unsigned down_type,int form) {
    const bool q20=gsq_a8::gate_type(request.gate_type);
    // STRIX_PREFILL_MOE_WMMA=0/1: dot4 or iu8 WMMA kernels (bit-identical results; default WMMA).
    static const int wmma_env=[]{const char* v=std::getenv("STRIX_PREFILL_MOE_WMMA");return v?std::atoi(v):-1;}();
    // GSQ experts take the FP16 forms (hip_moe_gsq_f16_prefill.inc: every GSQ gate/up type with a GSQ down type, Q2_0 or
    // IQ4_NL) unless STRIX_GSQ_PREFILL_F16=0 (gsq_prefill_f16_enabled).
    static const bool gsq_f16_env=gsq_prefill_f16_enabled();
    // The GSQ decoders read rows as whole dwords and 18-byte blocks in even/odd pairs (hip_moe_gsq_weights.inc): rows of
    // another geometry keep the A8 forms.
    const auto rows_ok=[](std::uint32_t type,std::uint32_t width) {
        return width%gsq_a8::block_elements(type)==0U&&gsq_a8::row_bytes(type,width)%4U==0U&&
            (gsq_a8::block_bytes(type)!=18U||(width/gsq_a8::block_elements(type))%2U==0U);
    };
    const bool gsq_f16=q20&&gsq_a8::down_type(down_type)&&rows_ok(request.gate_type,request.shape.hidden)&&
        rows_ok(down_type,request.shape.rows);
    // GSQ Q2_0 gate/up with Q2_0 down also has the iu8 WMMA form (hip_moe_gsq_wmma_prefill.inc, bit-identical to dot4).
    const bool gsq_q20=q20&&request.gate_type==gsq_a8::type_q2_0&&down_type==gsq_a8::type_q2_0;
    const bool fast_form=form>=3&&form<=66;
    if(fast_form&&(!q20||(gsq_f16&&gsq_f16_env)))return PrefillKernelForm::f16_wmma;
    // A GSQ expert the fast profile leaves on the A8 forms selects iu8 or dot4 as the A8 profile does (form -1:
    // STRIX_PREFILL_MOE_WMMA, iu8 by default, dot4 with =0).
    const int a8_form=fast_form?-1:form;
    // Whole CTA tiles; the WMMA down is unrolled for width = max_rows.
    const bool wmma_shape=request.shape.rows==max_rows&&request.shape.hidden%256U==0U;
    return (!q20||gsq_q20)&&wmma_shape&&(a8_form>=0?a8_form>=1:wmma_env>=0?wmma_env==1:true)?PrefillKernelForm::iu8_wmma:
        PrefillKernelForm::dot4;
}
inline unsigned& prefill_last_tune_value(){static thread_local unsigned v=0U;return v;}
// STRIX_QPF2K_SCHED bit 64: the armed one-shot stage hook of this thread (see set_prefill_stage_hook).
struct PrefillStageHookSlot { PrefillStageHook hook=nullptr; void* user=nullptr; unsigned stage=0U; };
inline PrefillStageHookSlot& prefill_stage_hook_slot(){static thread_local PrefillStageHookSlot v;return v;}
void set_prefill_stage_hook(PrefillStageHook hook,void* user,unsigned stage) {
    auto& slot=prefill_stage_hook_slot();
    if(hook&&(slot.hook||stage<1U||stage>3U))throw std::logic_error("known prefill stage hook armed twice or at an unknown stage");
    slot=hook?PrefillStageHookSlot{hook,user,stage}:PrefillStageHookSlot{};
}
// Runs the armed hook once when `stage` reaches its stage (3 runs it for every earlier stage the call did not reach).
void prefill_stage_reached(unsigned stage,hipStream_t stream) {
    auto& slot=prefill_stage_hook_slot();
    if(!slot.hook||stage<slot.stage)return;
    const auto armed=slot;slot={};
    armed.hook(armed.user,stream);
}
bool prefill_stage_hook_armed(){return prefill_stage_hook_slot().hook!=nullptr;}
unsigned prefill_effective_tune(const Request& request,unsigned count,unsigned down_type,int form) {
    static const unsigned tune_env=[]{const char* v=std::getenv("STRIX_INTPREFILL_MOE_TUNE");
        const int n=v?std::atoi(v):82;return n>=0&&n<=65535?static_cast<unsigned>(n):0U;}();
    static const bool tune_automatic=std::getenv("STRIX_INTPREFILL_MOE_TUNE")==nullptr;
    // Forms 4..34 = tunes 1..31, forms 35..66 = tunes 64..95 (bit 6).
    // Bit 1 takes 128-assignment batches only when the prompt averages >= 96 assignments per expert: below that the
    // 64-assignment batches already hold most experts' rows (one dequantization either way) and half of the 16 waves
    // would have no rows.
    const bool f16=prefill_kernel_form(request,down_type,form)==PrefillKernelForm::f16_wmma;
    const unsigned tune_req=form>=4&&f16?static_cast<unsigned>(form<=34?form-3:form-3+32):tune_env;
    // qpf2k core2 bit 3 (STRIX_QPF2K_MOE 9/11/13/15): keep the 128-assignment batches of bit 1 at every T (same bits:
    // results do not depend on the batch composition). Never for GSQ experts (their kernels ignore STRIX_QPF2K_MOE).
    const bool q20=gsq_a8::gate_type(request.gate_type);
    const unsigned tune=(tune_req&2U)&&static_cast<unsigned long long>(count)*request.shape.top_k<96ULL*request.shape.experts&&
        !(!q20&&tune_req==82U&&(prefill_core2()&8U))?tune_req&~2U:tune_req;
    // MoE XOR rows (hip_moe_prefill_da16.inc): the automatic tune's 64-assignment form from 3072 rows, only where the DA16
    // chain runs (the dispatch predicate of enqueue_prefill); explicit tunes and forms are never rewritten.
    if(tune==80U&&tune_req==82U&&tune_automatic&&form<4&&!q20&&count>=prefill_xor64_min_rows&&prefill_da16()&&
       prefill_route()==14U&&prefill_core2()==0U&&prefill_small()==0U&&request.shape.rows<=max_rows&&request.shape.rows%64U==0U&&
       request.shape.hidden/64U>=5U&&request.shape.rows/64U>=5U&&prefill_xor64(qpf2k::gsq_model)) {
        static const bool witness=[]{std::fprintf(stderr,"qpf2k moe xor64 rows active\n");return true;}();(void)witness;
        return 88U;
    }
    return tune;
}
void enqueue_prefill(const Request& request,const PrefillRows& rows,unsigned count,unsigned clear_words,
    const unsigned char* gate,const unsigned char* up,unsigned down_type,const unsigned char* down_weights,hipStream_t stream,int form,DirectBOperands direct_assets) {
    const bool q5k=request.gate_type==13U,q20=gsq_a8::gate_type(request.gate_type);  // every GSQ gate type: dot4 form
    direct_b_ran()=false;direct_b_reason()=direct_b_requested()?"not-selected":"off";
    prefill_certificate_flag()=false;prefill_tiers_flag()=0;prefill_down_lds_flag()=false;
    prefill_fused_flag()=false;prefill_da16_flag()=false;prefill_tile64_flag()=false;prefill_small_flag()=false;prefill_clear_skipped_flag()=false;
    if((down_type!=7U&&down_type!=8U&&!gsq_a8::down_type(down_type))||(request.gate_type!=12U&&!q5k&&!q20)||
       (gsq_a8::down_type(down_type)&&!q20)||
       !gate||!up||!down_weights||
       count<16U||count%8U||count>q51_known_rows::prefill_max_rows||request.shape.top_k>10U||request.shape.experts>512U||
       request.shape.hidden>max_hidden||request.shape.hidden%256U||request.shape.rows>max_rows||request.shape.rows%128U||
       request.shape.hidden%64U)
        throw std::invalid_argument("A8 prefill geometry/format is invalid");
    (void)prefill_pool_owner().rows(count);
    // qpf2k/moe-mi ROUTE bits and DA16 are gated on the production Q4_K/Q5_K prefill only: with a GSQ gate type they are
    // refused (fail-closed) until that combination has its own gate.
    // Defaulted (not explicitly set) ROUTE / DA16 are turned off for a GSQ gate type (STRIX_QPF2K_MOE_DEFAULTED, set by the
    // default-ON initialiser in hip_known_prefix_window.hip); explicit settings are still refused.
    {
        static const unsigned defaulted=[]{const char* v=std::getenv("STRIX_QPF2K_MOE_DEFAULTED");return v?static_cast<unsigned>(std::atoi(v))&3U:0U;}();
        prefill_moe_gsq_off()=q20?defaulted:0U;
    }
    if(q20&&(prefill_route()!=0U||prefill_da16()))
        throw std::invalid_argument("STRIX_QPF2K_MOE_ROUTE / STRIX_QPF2K_MOE_DA16 are not qualified with the GSQ prefill profile");
    // STRIX_PREFILL_MOE_BATCH: assignments of one expert per CTA (default 8).
    static const unsigned dot4_cap=[]{const char* v=std::getenv("STRIX_PREFILL_MOE_BATCH");const int n=v?std::atoi(v):8;
        return n>=1&&n<=64?static_cast<unsigned>(n):8U;}();
    const PrefillKernelForm kernel_form=prefill_kernel_form(request,down_type,form);
    const bool f16=kernel_form==PrefillKernelForm::f16_wmma;
    const unsigned tune=prefill_effective_tune(request,count,down_type,form);
    prefill_last_tune_value()=tune;
    if(direct_b_requested()) {
        direct_b_reason()=request.gate_type!=12U?"format":!f16?"FP16-prefill-required":
            (count!=4096U&&count!=8192U)?"prompt-shape":
            (tune!=80U&&tune!=82U&&tune!=88U)?"tune":
            prefill_core2()!=0U?"core-schedule":prefill_small()!=0U?"small-schedule":
            rows.route_bf16?"route-BF16":(!direct_assets.gate||!direct_assets.up)?"prepared-pair-unavailable":"resource-ceiling";
    }
    // Binding requests certification; the launcher supplies it only with MC on. Fail closed on every other recipe.
    const bool certificate=rows.unsafe&&form==3&&!q20&&f16&&(tune==82U||tune==80U||tune==88U)&&prefill_route()==14U&&
        prefill_da16()&&prefill_core2()==0U&&prefill_small()==0U&&clear_words==request.shape.hidden&&
        request.shape.hidden/64U>=5U&&request.shape.rows/64U>=5U&&
        reinterpret_cast<std::uintptr_t>(rows.input.base)%16U==0U&&rows.input.stride%16U==0U;
    if(rows.route_bf16 && (!certificate || prefill_core2()!=0U || q20 || (down_type!=7U && down_type!=8U)))
        throw std::invalid_argument("MoE route BF16 requires DA16 certificate writer and matching ordered merge");
    if(rows.route_i8) {
        const auto layout=strix::moe_route_i8_layout(request.shape.hidden,request.shape.top_k);
        if(!rows.route_bf16||!strix::moe_route_i8_enabled()||layout.bytes>4U*layout.values||rows.rawdots.stride<4U*layout.values)
            throw std::invalid_argument("MoE route I8 requires enabled BF16 route and its reserved FP32 row");
        const auto resources=[&]<unsigned Tune>() {
            if(down_type==7U)prefill_route_i8_resources<true,Tune>();else prefill_route_i8_resources<false,Tune>();
        };
        if(tune==80U)resources.template operator()<80U>();
        else if(tune==82U)resources.template operator()<82U>();
        else resources.template operator()<88U>(); // BF16 admission above fixes the tune set.
        static const bool witness=[]{std::fprintf(stderr,"MoE FAST lever route i8: ON (bit-changing; STRIX_PREFILL_MOE_ROUTE_I8=0 restores route bf16)\n");return true;}();
        (void)witness;
    }
    if(f16&&(request.shape.rows%128U||request.shape.hidden%256U))throw std::invalid_argument("A8 prefill FP16 WMMA geometry is invalid");
    const bool wmma=kernel_form!=PrefillKernelForm::dot4;
    const unsigned cap=f16?((tune&2U)?2U*f16_batch:f16_batch):wmma?wmma_batch:dot4_cap;
    if(f16) {
        if(request.input!=InputKind::bf16)throw std::invalid_argument("A8 prefill FP16 WMMA requires BF16 input rows");
        check(hipPeekAtLastError(),"FP16 prefill prologue stale error");
        // qpf2k/moe-mi STRIX_QPF2K_MOE_ROUTE bit 2 (hip_moe_prefill_route.inc): same words, one read of each row, one barrier.
        const auto aligned16=[](const void* p,std::size_t stride){return reinterpret_cast<std::uintptr_t>(p)%16U==0U&&stride%16U==0U;};
        // ROUTE bit 8 (hip_moe_prefill_route.inc): the accumulation view is not zeroed here; the caller's merge
        // (known_moe_prefill_merge_kernel_z) writes the same zeros on its skip paths (clear_words == hidden: the same words).
        const bool skip_clear=(prefill_route()&8U)!=0U&&(prefill_core2()&512U)==0U&&clear_words==request.shape.hidden;
        if(skip_clear){prefill_clear_skipped_flag()=true;
            static const bool witness=[]{std::fprintf(stderr,"qpf2k-moe-mi route clear moved to the merge\n");return true;}();(void)witness;}
        const unsigned prologue_clear=skip_clear?0U:clear_words;
        if((prefill_route()&2U)&&request.shape.hidden%8U==0U&&request.shape.hidden<=max_hidden&&request.shape.top_k<=32U&&
            aligned16(rows.input.base,rows.input.stride)) {
            static const bool witness=[]{std::fprintf(stderr,"qpf2k-moe-mi route prologue v2 active\n");return true;}();(void)witness;
            if(certificate)hipLaunchKernelGGL(f16_prefill_prologue_v2_certificate,dim3(count),dim3(256),0,stream,rows,request.shape.hidden,request.shape.top_k,
                request.shape.experts,prologue_clear,(tune&16U)?f16_prefill_a16():nullptr,
                aligned16(rows.clear.base,rows.clear.stride)&&clear_words%4U==0U?1U:0U);
            else
            hipLaunchKernelGGL(f16_prefill_prologue_v2,dim3(count),dim3(256),0,stream,rows,request.shape.hidden,request.shape.top_k,
                request.shape.experts,prologue_clear,(tune&16U)?f16_prefill_a16():nullptr,
                aligned16(rows.clear.base,rows.clear.stride)&&clear_words%4U==0U?1U:0U);
        } else
        hipLaunchKernelGGL(f16_prefill_prologue,dim3(count),dim3(256),0,stream,rows,request.shape.hidden,request.shape.top_k,
            request.shape.experts,prologue_clear,(tune&16U)?f16_prefill_a16():nullptr);
        check(hipGetLastError(),"FP16 prefill prologue");
    } else enqueue_prefill_prologue(request,rows,count,clear_words,stream);
    const unsigned assignments=count*request.shape.top_k;
    // The WMMA forms (and every T above the serial builder's 2048 rows) build the identical CSR with
    // the parallel builder; form 2 keeps the serial one.
    if(form==2&&count>serial_csr_max_rows)throw std::invalid_argument("A8 prefill serial CSR builder admits at most 2048 rows");
    if(((wmma&&form!=2)||count>serial_csr_max_rows)&&(prefill_route()&4U)) { // qpf2k/moe-mi route bit 4: multi-CTA CSR, same words
        static const bool witness=[]{std::fprintf(stderr,"qpf2k-moe-mi route csr multi-cta active\n");return true;}();(void)witness;
        const unsigned blocks=(count*request.shape.top_k+csr_mw_span-1U)/csr_mw_span;
        if(blocks==0U||blocks>csr_mw_blocks_max())throw std::invalid_argument("qpf2k CSR multi-CTA block count out of range");
        unsigned* scratch=csr_mw_scratch();
        check(hipMemsetAsync(scratch,0,csr_mw_head*sizeof(unsigned),stream),"qpf2k CSR scratch clear");
        if(certificate)hipLaunchKernelGGL(csr_mw_count_certificate,dim3(blocks),dim3(1024),0,stream,rows,count,request.shape.experts,request.shape.top_k,scratch);
        else
        hipLaunchKernelGGL(csr_mw_count,dim3(blocks),dim3(1024),0,stream,rows,count,request.shape.experts,request.shape.top_k,scratch);
        hipLaunchKernelGGL(csr_mw_scan,dim3(1),dim3(1024),0,stream,rows,count,request.shape.experts,request.shape.top_k,cap,blocks,scratch);
        hipLaunchKernelGGL(csr_mw_scatter,dim3(blocks),dim3(1024),0,stream,rows,count,request.shape.top_k,scratch);
    } else if(((wmma&&form!=2)||count>serial_csr_max_rows)&&(prefill_route()&1U)) { // qpf2k/moe-mi route bit 1: same words
        static const bool witness=[]{std::fprintf(stderr,"qpf2k-moe-mi route csr v2 active\n");return true;}();(void)witness;
        hipLaunchKernelGGL(build_prefill_csr_waves_v2,dim3(1),dim3(1024),0,stream,rows,count,request.shape.experts,request.shape.top_k,cap);
    } else if((wmma&&form!=2)||count>serial_csr_max_rows)
        hipLaunchKernelGGL(build_prefill_csr_waves,dim3(1),dim3(1024),0,stream,rows,count,request.shape.experts,request.shape.top_k,cap);
    else hipLaunchKernelGGL(build_prefill_csr,dim3(1),dim3(1024),0,stream,rows,count,request.shape.experts,request.shape.top_k,cap);
    check(hipGetLastError(),"A8 prefill CSR");
    prefill_stage_reached(1U,stream);
    // STRIX_PREFILL_MOE_CSR_DUMP=1 (diagnostic, synchronizes): rows-per-expert histogram of each call.
    static const bool csr_dump=[]{const char* v=std::getenv("STRIX_PREFILL_MOE_CSR_DUMP");return v&&v[0]=='1';}();
    if(csr_dump) {
        std::vector<unsigned> offsets(request.shape.experts+1U);
        check(hipMemcpyAsync(offsets.data(),rows.csr.offsets,4U*offsets.size(),hipMemcpyDeviceToHost,stream),"CSR dump");
        check(hipStreamSynchronize(stream),"CSR dump");
        std::array<unsigned,8> bucket{};unsigned largest=0;
        for(unsigned e=0;e<request.shape.experts;++e) {
            const unsigned n=offsets[e+1U]-offsets[e];largest=std::max(largest,n);
            bucket[n==0U?0U:n<=16U?1U:n<=32U?2U:n<=64U?3U:n<=128U?4U:n<=256U?5U:n<=512U?6U:7U]++;
        }
        std::fprintf(stderr,"moe-csr T=%u experts n=0:%u 1-16:%u 17-32:%u 33-64:%u 65-128:%u 129-256:%u 257-512:%u >512:%u max %u\n",
            count,bucket[0],bucket[1],bucket[2],bucket[3],bucket[4],bucket[5],bucket[6],bucket[7],largest);
        // Batch fill at both caps (64/128 rows): batches, partial batches, their rows, padded tile rows; partial < 32 rows too.
        for(const unsigned c:{64U,128U}) {
            unsigned long long b=0,partial=0,partial_rows=0,pad=0,small=0,small_rows=0;
            for(unsigned e=0;e<request.shape.experts;++e) {
                const unsigned n=offsets[e+1U]-offsets[e];if(!n)continue;
                b+=(n+c-1U)/c;
                if(const unsigned r=n%c) {++partial;partial_rows+=r;pad+=c-r;if(r<32U){++small;small_rows+=r;}}
            }
            std::fprintf(stderr,"moe-csr-fill T=%u cap %u assignments %u batches %llu partial %llu partial_rows %llu pad_rows %llu fill %.4f small32 %llu small32_rows %llu\n",
                count,c,offsets[request.shape.experts],b,partial,partial_rows,pad,b?static_cast<double>(offsets[request.shape.experts])/static_cast<double>(b*c):0.0,small,small_rows);
        }
    }
    const unsigned batches=std::min(assignments,request.shape.experts+(assignments+cap-1U)/cap);
    if(f16&&q20) {
        // GSQ FP16 kernels: tune bits 1 and 4 and the diagnostic bits (the other bits are layout/dequantization
        // forms of the production kernels with the same bytes).
        const unsigned gtune=tune&gsq_f16_tune_mask;
        const auto launch=[&]<unsigned Tune>() {
            using Tile=F16Tile<Tune>;
            constexpr unsigned Threads=Tile::Threads,Outs=Tile::Columns/2U;
            const uint4* a16=(Tune&16U)?f16_prefill_a16():nullptr;
            const auto gate_up=[&]<std::uint32_t G>() {
                prefill_resources<project_prefill_f16g<G,Tune>,Threads>("gate_up_gsq",Tune);
                hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16g<G,Tune>),dim3(request.shape.rows/Outs,batches),dim3(Threads),0,
                    stream,gate,up,rows,request.shape.hidden,request.shape.rows,a16);
            };
            const auto down=[&]<std::uint32_t D>() {
                prefill_resources<down_prefill_f16g<D,Tune>,Threads>("down_gsq",Tune);
                hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16g<D,Tune>),dim3(request.shape.hidden/Tile::Columns,batches),dim3(Threads),0,
                    stream,down_weights,rows,request.shape.hidden,request.shape.rows);
            };
            switch(request.gate_type) {
            case gsq_a8::type_q2_0: gate_up.template operator()<gsq_a8::type_q2_0>(); break;
            case gsq_a8::type_iq2_s: gate_up.template operator()<gsq_a8::type_iq2_s>(); break;
            case gsq_a8::type_iq3_xxs: gate_up.template operator()<gsq_a8::type_iq3_xxs>(); break;
            case gsq_a8::type_iq3_s: gate_up.template operator()<gsq_a8::type_iq3_s>(); break;
            default: gate_up.template operator()<gsq_a8::type_iq4_xs>(); break;
            }
            check(hipGetLastError(),"GSQ FP16 prefill gate/up projection");
            if(down_type==gsq_a8::type_q2_0)down.template operator()<gsq_a8::type_q2_0>();
            else down.template operator()<gsq_a8::type_iq4_nl>();
            check(hipGetLastError(),"GSQ FP16 prefill down");
        };
        const bool launched=[&]<unsigned... T>(std::integer_sequence<unsigned,T...>){
            return ((gtune==T&&(launch.template operator()<T>(),true))||...);
        }(std::integer_sequence<unsigned,0U,2U,16U,18U,272U,274U,528U,530U,1040U,1042U>{});
        if(!launched)throw std::invalid_argument("GSQ FP16 prefill tune is not instantiated");
        return;
    }
    if(f16) {
        // Forms 4..18 expose each schedule to byte-comparison fixtures. Form 3 uses
        // the opt-in environment selector; no arithmetic/profile identity changes.
        const auto launch=[&]<unsigned Tune>() {
            if constexpr((Tune&~9U)!=0U) {
                // Redone tiles (bits 1, 2, 4): grid x = output tile (the tiles of a batch run together), y = batch.
                using Tile=F16Tile<Tune>;
                constexpr unsigned Threads=Tile::Threads,Outs=Tile::Columns/2U;
                if(q5k)prefill_resources<project_prefill_f16x<true,Tune>,Threads>("gate_up_q5k",Tune);
                else prefill_resources<project_prefill_f16x<false,Tune>,Threads>("gate_up_q4k",Tune);
                if(down_type==7U)prefill_resources<down_prefill_f16x<true,Tune>,Threads>("down_q51",Tune);
                else prefill_resources<down_prefill_f16x<false,Tune>,Threads>("down_q80",Tune);
                const uint4* a16=(Tune&16U)?f16_prefill_a16():nullptr;
                const dim3 project_grid(request.shape.rows/Outs,batches);
                bool direct=direct_b_requested()&&direct_assets.gate&&direct_assets.up&&!q5k&&
                    (count==4096U||count==8192U)&&prefill_core2()==0U&&prefill_small()==0U&&!rows.route_bf16;
                if constexpr(Tune==80U||Tune==82U||Tune==88U) {
                    if(direct)direct=prefill_da16()?direct_b_resource_admitted<Tune,true>():direct_b_resource_admitted<Tune,false>();
                } else direct=false;
                const auto direct_project=[&]<bool DA16>(unsigned* arrive,uint4* image) {
                    using DT=DirectBTile<Tune>;
                    prefill_resources_at(reinterpret_cast<const void*>(direct_b_kernel<Tune,DA16>()),"gate_up_q4k_direct_b",Tune,DT::Threads);
                    launch_prefill_direct_b<Tune,DA16>(dim3(request.shape.rows/DT::Outputs,batches),stream,
                        direct_assets.gate,direct_assets.up,rows,request.shape.hidden,request.shape.rows,a16,arrive,image);
                    direct_b_ran()=true;direct_b_reason()="engaged";
                };
                // qpf2k core2 (STRIX_QPF2K_MOE = 1, 3, 5 or 7; production tunes 80 and 82 only): same bits.
                // The depth-D pipeline primes D stages: K steps (hidden/64 for gate/up, rows/64 for down) must be >= 5.
                if constexpr(Tune==80U||Tune==82U) if(const unsigned core=request.shape.hidden/64U>=5U&&request.shape.rows/64U>=5U?prefill_core2():0U) {
                    if(core==257U) {
                        // qpf2k P1 (hip_moe_prefill_p1.inc, design-5.md): gate/up and down in one ordered-claim launch; same bits.
                        const unsigned lag=p1_lag();
                        static bool seen=false;
                        if(!seen){seen=true;std::fprintf(stderr,"qpf2k-moe p1 active tune=%u core=257 lag=%u\n",Tune,lag);}
                        const unsigned G=request.shape.rows/Outs,Dn=request.shape.hidden/Tile::Columns;
                        if(batches>p1_max_batches()||p1_pool_words()%2U)throw std::invalid_argument("qpf2k P1 batch counters exceed the pool");
                        unsigned* pool=p1_counter_pool();
                        check(hipMemsetAsync(pool,0,(2U+static_cast<std::size_t>(batches))*sizeof(unsigned),stream),"qpf2k P1 counters clear");
                        unsigned* stats=p1_stats_enabled()?pool+p1_pool_words()-p1_stats_words:nullptr;
                        if(stats)check(hipMemsetAsync(stats,0,p1_stats_words*sizeof(unsigned),stream),"qpf2k P1 stats clear");
                        const P1Args pa{pool,pool+1,stats,batches,lag,G,Dn,lag==65535U?1U:0U};
                        const dim3 grid((G+Dn)*batches);
                        // qpf2k/moe-mi: P1 + DA16 (hip_moe_prefill_p1da16.inc) when STRIX_QPF2K_MOE_DA16=1. Same bits.
                        const bool p1da16=prefill_da16()&&request.shape.rows<=max_rows;
                        unsigned* arrive=nullptr;uint4* image=nullptr;
                        if(p1da16) {
                            prefill_da16_flag()=true;
                            static bool seen16=false;
                            if(!seen16){seen16=true;std::fprintf(stderr,"qpf2k-moe-mi p1+da16 active tune=%u\n",Tune);}
                            if(batches>da16_counter_words())throw std::invalid_argument("qpf2k DA16 batch counters exceed the pool");
                            arrive=da16_counters();image=da16_image();
                            check(hipMemsetAsync(arrive,0,static_cast<std::size_t>(batches)*sizeof(unsigned),stream),"qpf2k P1+DA16 counters clear");
                        }
                        const auto go=[&]<bool QG,bool QD>(){
                            if(p1da16)hipLaunchKernelGGL(HIP_KERNEL_NAME(p1_prefill_f16x_da16<QG,QD,Tune>),grid,dim3(Threads),0,stream,
                                gate,up,down_weights,rows,request.shape.hidden,request.shape.rows,a16,pa,arrive,image);
                            else
                            hipLaunchKernelGGL(HIP_KERNEL_NAME(p1_prefill_f16x<QG,QD,Tune>),grid,dim3(Threads),0,stream,
                                gate,up,down_weights,rows,request.shape.hidden,request.shape.rows,a16,pa);
                        };
                        if(q5k){if(down_type==7U)go.template operator()<true,true>();else go.template operator()<true,false>();}
                        else {if(down_type==7U)go.template operator()<false,true>();else go.template operator()<false,false>();}
                        check(hipGetLastError(),"FP16 prefill WMMA gate/up + down (qpf2k P1)");
                        return;
                    }
                    if(core&1024U) {
                        // qpf2k G3 core (hip_moe_prefill_g3.inc): same bits. Launch witness once per (tune, core).
                        static bool seen[8192]{};
                        if(!seen[core]){seen[core]=true;std::fprintf(stderr,"qpf2k-moe g3 active tune=%u core=%u\n",Tune,core);}
                        const unsigned probe=(core>>11U)&3U;
                        const auto by_probe=[&](const auto& f){
                            if(probe==1U)f.template operator()<1U>();else if(probe==2U)f.template operator()<2U>();
                            else if(probe==3U)f.template operator()<3U>();else f.template operator()<0U>();
                        };
                        // Depth (bit 5: 2 stages in flight; c22: depth 3 spills ~0.5 KB at BM 256): probes only at depth 1.
                        const auto by_depth=[&](const auto& f){
                            if(core&32U)f.template operator()<2U,0U>();else by_probe([&]<unsigned P>(){f.template operator()<1U,P>();});
                        };
                        const auto gu=[&]<unsigned BM>(){by_depth([&]<unsigned D,unsigned P>(){
                            const dim3 grid(request.shape.rows/128U,batches);
                            if(q5k)hipLaunchKernelGGL(HIP_KERNEL_NAME(g3_project<true,BM,P,D>),grid,dim3(2U*BM),0,stream,gate,up,rows,request.shape.hidden,request.shape.rows,a16,cap);
                            else hipLaunchKernelGGL(HIP_KERNEL_NAME(g3_project<false,BM,P,D>),grid,dim3(2U*BM),0,stream,gate,up,rows,request.shape.hidden,request.shape.rows,a16,cap);
                        });};
                        const auto dn=[&]<unsigned BM>(){by_depth([&]<unsigned D,unsigned P>(){
                            const dim3 grid(request.shape.hidden/256U,batches);
                            if(down_type==7U)hipLaunchKernelGGL(HIP_KERNEL_NAME(g3_down<true,BM,P,D>),grid,dim3(2U*BM),0,stream,down_weights,rows,request.shape.hidden,request.shape.rows,cap);
                            else hipLaunchKernelGGL(HIP_KERNEL_NAME(g3_down<false,BM,P,D>),grid,dim3(2U*BM),0,stream,down_weights,rows,request.shape.hidden,request.shape.rows,cap);
                        });};
                        if(core&16U) {
                            if(q5k)hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16x<true,Tune>),project_grid,dim3(Threads),0,stream,
                                gate,up,rows,request.shape.hidden,request.shape.rows,a16);
                            else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16x<false,Tune>),project_grid,dim3(Threads),0,stream,
                                gate,up,rows,request.shape.hidden,request.shape.rows,a16);
                        } else if(core&2U)gu.template operator()<128U>();
                        else gu.template operator()<256U>();
                        check(hipGetLastError(),"FP16 prefill WMMA gate/up projection (g3)");
                        if(core&4U) {
                            const dim3 down_grid(request.shape.hidden/Tile::Columns,batches);
                            if(down_type==7U)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16x<true,Tune>),down_grid,dim3(Threads),0,
                                stream,down_weights,rows,request.shape.hidden,request.shape.rows,FuseNone{});
                            else hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16x<false,Tune>),down_grid,dim3(Threads),0,
                                stream,down_weights,rows,request.shape.hidden,request.shape.rows,FuseNone{});
                        } else if(core&2U)dn.template operator()<128U>();
                        else dn.template operator()<256U>();
                        check(hipGetLastError(),"FP16 prefill WMMA down (g3)");
                        return;
                    }
                    const auto run=[&]<unsigned Core>() {
                        // Launch witness (engine gates): once per process and (tune, core).
                        static const bool witness=[]{std::fprintf(stderr,"qpf2k-moe core2 active tune=%u core=%u\n",Tune,Core);return true;}();
                        (void)witness;
                        if(q5k)prefill_resources<project_prefill_f16x<true,Tune,Core>,F16TileX<Tune,Core>::Threads>("gate_up_q5k_core2",Tune+1000U*Core);
                        else prefill_resources<project_prefill_f16x<false,Tune,Core>,F16TileX<Tune,Core>::Threads>("gate_up_q4k_core2",Tune+1000U*Core);
                        if(down_type==7U)prefill_resources<down_prefill_f16x<true,Tune,Core>,Threads>("down_q51_core2",Tune+1000U*Core);
                        else prefill_resources<down_prefill_f16x<false,Tune,Core>,Threads>("down_q80_core2",Tune+1000U*Core);
                        using TX=F16TileX<Tune,Core>;
                        const dim3 pgrid(request.shape.rows/(TX::Columns/2U),batches);
                        if(q5k)hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16x<true,Tune,Core>),pgrid,dim3(TX::Threads),0,stream,
                            gate,up,rows,request.shape.hidden,request.shape.rows,a16);
                        else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16x<false,Tune,Core>),pgrid,dim3(TX::Threads),0,stream,
                            gate,up,rows,request.shape.hidden,request.shape.rows,a16);
                        if constexpr((Core&256U)!=0U&&(Core&64U)!=0U) {
                            // Split: the lone batches of the 256-row launch run as 128-row CTAs (same per-output chains).
                            constexpr unsigned CN=Core&~64U;
                            using TN=F16TileX<Tune,CN>;
                            const dim3 ngrid(request.shape.rows/(TN::Columns/2U),batches);
                            if(q5k)hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16x<true,Tune,CN>),ngrid,dim3(TN::Threads),0,stream,
                                gate,up,rows,request.shape.hidden,request.shape.rows,a16);
                            else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16x<false,Tune,CN>),ngrid,dim3(TN::Threads),0,stream,
                                gate,up,rows,request.shape.hidden,request.shape.rows,a16);
                        }
                        check(hipGetLastError(),"FP16 prefill WMMA gate/up projection (core2)");
                        const dim3 down_grid(request.shape.hidden/Tile::Columns,batches);
                        FuseArgT<Core> fa{};
                        if constexpr((Core&512U)!=0U) {
                            fa=FuseArgs{fuse_counter_pool(),request.shape.top_k,request.shape.hidden/Tile::Columns,count};
                            check(hipMemsetAsync(fa.count,0,(static_cast<std::size_t>(count)*fa.tiles+1U)*sizeof(unsigned),stream),"fused merge counters clear");
                            prefill_fused_flag()=true;
                        }
                        if(down_type==7U)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16x<true,Tune,Core>),down_grid,dim3(Threads),0,
                            stream,down_weights,rows,request.shape.hidden,request.shape.rows,fa);
                        else hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16x<false,Tune,Core>),down_grid,dim3(Threads),0,
                            stream,down_weights,rows,request.shape.hidden,request.shape.rows,fa);
                        check(hipGetLastError(),"FP16 prefill WMMA down (core2)");
                    };
                    switch(core&~8U) {
                    case 3U: run.template operator()<3U>(); break;
                    case 5U: run.template operator()<5U>(); break;
                    case 7U: run.template operator()<7U>(); break;
                    case 17U: run.template operator()<17U>(); break;
                    case 33U: run.template operator()<33U>(); break;
                    case 49U: run.template operator()<49U>(); break;
                    case 65U: if constexpr(Tune==82U) run.template operator()<65U>(); else run.template operator()<1U>(); break;
                    case 81U: if constexpr(Tune==82U) run.template operator()<81U>(); else run.template operator()<17U>(); break;
                    case 97U: if constexpr(Tune==82U) run.template operator()<97U>(); else run.template operator()<33U>(); break;
                    case 129U: run.template operator()<129U>(); break;
                    case 513U: run.template operator()<513U>(); break;
                    case 641U: run.template operator()<641U>(); break;
                    case 321U: if constexpr(Tune==82U) run.template operator()<321U>(); else run.template operator()<1U>(); break;
                    case 449U: if constexpr(Tune==82U) run.template operator()<449U>(); else run.template operator()<129U>(); break;
                    case 145U: run.template operator()<145U>(); break;
                    default: run.template operator()<1U>(); break;
                    }
                    return;
                }
                // qpf2k/moe-mi DA16 (STRIX_QPF2K_MOE_DA16=1, hip_moe_prefill_da16.inc): the last gate/up tile of a batch writes the
                // batch's FP16 down operand image once; the down stages it as a copy. Same bits.
                if constexpr(Tune==80U||Tune==82U||Tune==88U) if(prefill_da16()&&request.shape.rows<=max_rows&&request.shape.rows%64U==0U&&
                    request.shape.hidden/64U>=5U&&request.shape.rows/64U>=5U) {
                    prefill_da16_flag()=true;
                    static const bool witness=[]{std::fprintf(stderr,"qpf2k-moe-mi da16 active\n");return true;}();(void)witness;
                    if(batches>da16_counter_words())throw std::invalid_argument("qpf2k DA16 batch counters exceed the pool");
                    unsigned* arrive=da16_counters();
                    uint4* image=da16_image();
                    prefill_da16_mark(0U,stream);
                    check(hipMemsetAsync(arrive,0,static_cast<std::size_t>(batches)*sizeof(unsigned),stream),"qpf2k DA16 counters clear");
                    // Tunes 80/88 already have 64-row CTAs; retain their launch path.
                    if constexpr(Tune==82U) if(prefill_tiers()) {
                        if(prefill_tile64()) {
                            static const bool declined=[] {
                                std::fprintf(stderr,"qpf2k moe tile64 declined: tier organization active\n");return true;
                            }();(void)declined;
                        }
                        unsigned* narrow=prefill_tier_lists();
                        unsigned* full=narrow+da16_counter_words()+1U;
                        check(hipMemsetAsync(narrow,0,sizeof(unsigned),stream),"DA16 narrow count clear");
                        check(hipMemsetAsync(full,0,sizeof(unsigned),stream),"DA16 full count clear");
                        prefill_tier_resources<build_prefill_tiers>();
                        hipLaunchKernelGGL(build_prefill_tiers,dim3((batches+255U)/256U),dim3(256),0,stream,rows,narrow,full);
                        check(hipGetLastError(),"DA16 tier lists");
                        const bool reuse=prefill_hdr_reuse(q20||qpf2k::gsq_model);
                        if(prefill_tiers()==2)
                            launch_prefill_tier<88U>(gate,up,rows,request.shape.hidden,request.shape.rows,a16,arrive,image,narrow,batches,q5k,reuse,stream);
                        else launch_prefill_tier<Tune&~2U>(gate,up,rows,request.shape.hidden,request.shape.rows,a16,arrive,image,narrow,batches,q5k,reuse,stream);
                        if constexpr(Tune==82U)launch_prefill_tier<82U>(gate,up,rows,request.shape.hidden,request.shape.rows,a16,arrive,image,full,batches,q5k,reuse,stream);
                        prefill_da16_mark(1U,stream);
                        if(prefill_tiers()==2)
                            launch_prefill_down_tier<88U>(down_weights,rows,request.shape.hidden,request.shape.rows,image,narrow,batches,down_type==7U,certificate,stream);
                        else launch_prefill_down_tier<Tune&~2U>(down_weights,rows,request.shape.hidden,request.shape.rows,image,narrow,batches,down_type==7U,certificate,stream);
                        if constexpr(Tune==82U)launch_prefill_down_tier<82U>(down_weights,rows,request.shape.hidden,request.shape.rows,image,full,batches,down_type==7U,certificate,stream);
                        prefill_da16_mark(2U,stream);
                        prefill_tiers_flag()=prefill_tiers();
                        prefill_certificate_flag()=certificate;
                        return;
                    }
                    // TILE64 (bit 0 gate/up, bit 1 down) is default OFF until node timing, admitted only for the 128-row DA16 tune.
                    // Q5_K gate/up always retains its incumbent specialization.
                    // Keep individually qualified organizations independent until joint native gates exist.
                    const unsigned requested_tile64=Tune==82U&&!q20&&(down_type==7U||down_type==8U)?prefill_tile64():0U;
                    const unsigned tile64=prefill_down_lds_admitted(rows)?0U:requested_tile64;
                    if(requested_tile64&&!tile64) {
                        static const bool declined=[] {
                            std::fprintf(stderr,"qpf2k moe tile64 declined: LDS epilogue organization active\n");return true;
                        }();(void)declined;
                    }
                    if(((tile64&1U)&&!q5k)||(tile64&2U)){prefill_tile64_flag()=true;static const bool t64_witness=[]{std::fprintf(stderr,"qpf2k moe tile64 active\n");return true;}();(void)t64_witness;}
                    if constexpr(Tune==80U||Tune==82U||Tune==88U) {
                        if(direct)direct_project.template operator()<true>(arrive,image);
                    }
                    if(direct_b_ran()) {} else
                    if((tile64&1U)&&!q5k) {
                        launch_tile64_project(gate,up,rows,request.shape.hidden,request.shape.rows,a16,arrive,image,batches,stream,
                            prefill_hdr_reuse(q20||qpf2k::gsq_model));
                    } else if(prefill_hdr_reuse(q20||qpf2k::gsq_model)) {
                        static const bool hdr_witness=[]{std::fprintf(stderr,"qpf2k moe gate/up header reuse active\n");return true;}();(void)hdr_witness;
                        if(q5k)hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16x_da16_mg<true,Tune>),project_grid,dim3(Threads),0,stream,
                            gate,up,rows,request.shape.hidden,request.shape.rows,a16,arrive,image);
                        else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16x_da16_mg<false,Tune>),project_grid,dim3(Threads),0,stream,
                            gate,up,rows,request.shape.hidden,request.shape.rows,a16,arrive,image);
                    } else
                    if(q5k)hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16x_da16<true,Tune>),project_grid,dim3(Threads),0,stream,
                        gate,up,rows,request.shape.hidden,request.shape.rows,a16,arrive,image);
                    else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16x_da16<false,Tune>),project_grid,dim3(Threads),0,stream,
                        gate,up,rows,request.shape.hidden,request.shape.rows,a16,arrive,image);
                    check(hipGetLastError(),"FP16 prefill WMMA gate/up projection (DA16)");
                    prefill_da16_mark(1U,stream);
                    prefill_stage_reached(2U,stream);
                    const dim3 down_grid(request.shape.hidden/Tile::Columns,batches);
                    if(tile64&2U) {
                        launch_tile64_down_mode(down_type==7U,certificate,down_weights,rows,request.shape.hidden,request.shape.rows,image,batches,stream);
                        check(hipGetLastError(),"FP16 prefill WMMA down (TILE64)");
                        if(certificate)prefill_certificate_flag()=true;
                        prefill_da16_mark(2U,stream);
                        return;
                    }
                    if constexpr(Tune==82U||Tune==80U||Tune==88U) if(certificate) {  // 80 = 82 below 96 assignments per expert; 88 = 80 with XOR rows
                        if(prefill_down_lds_admitted(rows)) {
                            launch_prefill_down_tier<Tune>(down_weights,rows,request.shape.hidden,request.shape.rows,image,nullptr,batches,down_type==7U,true,stream);
                            prefill_certificate_flag()=true;
                            prefill_da16_mark(2U,stream);
                            return;
                        }
                        if(rows.route_i8) {
                            if(down_type==7U)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16x_da16_certificate_i8<true,Tune>),down_grid,dim3(Threads),0,
                                stream,down_weights,rows,request.shape.hidden,request.shape.rows,image,request.shape.top_k);
                            else hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16x_da16_certificate_i8<false,Tune>),down_grid,dim3(Threads),0,
                                stream,down_weights,rows,request.shape.hidden,request.shape.rows,image,request.shape.top_k);
                        } else if(rows.route_bf16) {
                            if(down_type==7U)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16x_da16_certificate_bf16<true,Tune>),down_grid,dim3(Threads),0,
                                stream,down_weights,rows,request.shape.hidden,request.shape.rows,image);
                            else hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16x_da16_certificate_bf16<false,Tune>),down_grid,dim3(Threads),0,
                                stream,down_weights,rows,request.shape.hidden,request.shape.rows,image);
                        } else {
                            if(down_type==7U)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16x_da16_certificate<true,Tune>),down_grid,dim3(Threads),0,
                                stream,down_weights,rows,request.shape.hidden,request.shape.rows,image);
                            else hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16x_da16_certificate<false,Tune>),down_grid,dim3(Threads),0,
                                stream,down_weights,rows,request.shape.hidden,request.shape.rows,image);
                        }
                        check(hipGetLastError(),"FP16 prefill WMMA down (DA16 certificate)");
                        prefill_certificate_flag()=true; // publish only after the entire producer chain was enqueued
                        prefill_da16_mark(2U,stream);
                        return;
                    }
                    if(prefill_down_lds_admitted(rows)) {
                        launch_prefill_down_tier<Tune>(down_weights,rows,request.shape.hidden,request.shape.rows,image,nullptr,batches,down_type==7U,false,stream);
                        prefill_da16_mark(2U,stream);
                        return;
                    }
                    if(down_type==7U)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16x_da16<true,Tune>),down_grid,dim3(Threads),0,
                        stream,down_weights,rows,request.shape.hidden,request.shape.rows,image);
                    else hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16x_da16<false,Tune>),down_grid,dim3(Threads),0,
                        stream,down_weights,rows,request.shape.hidden,request.shape.rows,image);
                    check(hipGetLastError(),"FP16 prefill WMMA down (DA16)");
                    prefill_da16_mark(2U,stream);
                    return;
                }
                // qpf2k/moe-mi SMALL (STRIX_QPF2K_MOE_SMALL=<rows>, hip_moe_prefill_small.inc): gate/up batches with <= rows assignments
                // run the core2 depth-3 body, the others the base body, in one launch; the base down. Same bits.
                if constexpr(Tune==80U||Tune==82U) if(const unsigned small=prefill_small();small!=0U&&
                    request.shape.hidden/64U>=5U&&request.shape.rows/64U>=5U) {
                    prefill_small_flag()=true;
                    static const bool witness=[]{std::fprintf(stderr,"qpf2k-moe-mi small active\n");return true;}();(void)witness;
                    if(q5k)hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16x_small<true,Tune>),project_grid,dim3(Threads),0,stream,
                        gate,up,rows,request.shape.hidden,request.shape.rows,a16,small);
                    else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16x_small<false,Tune>),project_grid,dim3(Threads),0,stream,
                        gate,up,rows,request.shape.hidden,request.shape.rows,a16,small);
                    check(hipGetLastError(),"FP16 prefill WMMA gate/up projection (SMALL)");
                    const dim3 down_grid(request.shape.hidden/Tile::Columns,batches);
                    if(down_type==7U)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16x<true,Tune>),down_grid,dim3(Threads),0,
                        stream,down_weights,rows,request.shape.hidden,request.shape.rows,FuseNone{});
                    else hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16x<false,Tune>),down_grid,dim3(Threads),0,
                        stream,down_weights,rows,request.shape.hidden,request.shape.rows,FuseNone{});
                    check(hipGetLastError(),"FP16 prefill WMMA down (SMALL: base down)");
                    return;
                }
                if constexpr(Tune==80U||Tune==82U||Tune==88U) {
                    if(direct)direct_project.template operator()<false>(nullptr,nullptr);
                }
                if(direct_b_ran()) {} else
                if(q5k)hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16x<true,Tune>),project_grid,dim3(Threads),0,stream,
                    gate,up,rows,request.shape.hidden,request.shape.rows,a16);
                else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16x<false,Tune>),project_grid,dim3(Threads),0,stream,
                    gate,up,rows,request.shape.hidden,request.shape.rows,a16);
                check(hipGetLastError(),"FP16 prefill WMMA gate/up projection");
                const dim3 down_grid(request.shape.hidden/Tile::Columns,batches);
                if(down_type==7U)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16x<true,Tune>),down_grid,dim3(Threads),0,
                    stream,down_weights,rows,request.shape.hidden,request.shape.rows,FuseNone{});
                else hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16x<false,Tune>),down_grid,dim3(Threads),0,
                    stream,down_weights,rows,request.shape.hidden,request.shape.rows,FuseNone{});
                check(hipGetLastError(),"FP16 prefill WMMA down");
            } else {
            constexpr unsigned Threads=256U,Outs=Threads/2U;
            if(q5k)prefill_resources<project_prefill_f16<true,Tune>,Threads>("gate_up_q5k",Tune);
            else prefill_resources<project_prefill_f16<false,Tune>,Threads>("gate_up_q4k",Tune);
            if(down_type==7U)prefill_resources<down_prefill_f16<true,Tune>,Threads>("down_q51",Tune);
            else prefill_resources<down_prefill_f16<false,Tune>,Threads>("down_q80",Tune);
            const dim3 project_grid(request.shape.rows/Outs,batches);
            if(q5k)hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16<true,Tune>),project_grid,dim3(Threads),0,stream,
                gate,up,rows,request.shape.hidden,request.shape.rows);
            else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16<false,Tune>),project_grid,dim3(Threads),0,stream,
                gate,up,rows,request.shape.hidden,request.shape.rows);
            check(hipGetLastError(),"FP16 prefill WMMA gate/up projection");
            const dim3 down_grid(request.shape.hidden/Threads,batches);
            if(down_type==7U)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16<true,Tune>),down_grid,dim3(Threads),0,
                stream,down_weights,rows,request.shape.hidden,request.shape.rows);
            else hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_f16<false,Tune>),down_grid,dim3(Threads),0,
                stream,down_weights,rows,request.shape.hidden,request.shape.rows);
            check(hipGetLastError(),"FP16 prefill WMMA down");
            }
        };
        [&]<unsigned... T>(std::integer_sequence<unsigned,T...>){(void)((tune==T&&(launch.template operator()<T>(),true))||...);}
            (std::make_integer_sequence<unsigned,32U>{});
        [&]<unsigned... T>(std::integer_sequence<unsigned,T...>){(void)((tune==64U+T&&(launch.template operator()<64U+T>(),true))||...);}
            (std::make_integer_sequence<unsigned,32U>{});
        // Diagnostic attribution tunes: 16, 18, 80 or 82 plus one or two of the bits 256/512/1024, or 82 plus 2048/4096.
        [&]<unsigned... T>(std::integer_sequence<unsigned,T...>){(void)((tune==T&&(launch.template operator()<T>(),true))||...);}
            (std::integer_sequence<unsigned,272U,528U,1040U,1552U,274U,530U,1042U,1554U,336U,592U,1104U,2130U,4178U,338U,594U>{});
        // qpf2k/moe attribution (82 + 8192 hot A, + 16384 hot B, + 32768 no down raw-dot stores; wrong bits by design).
        [&]<unsigned... T>(std::integer_sequence<unsigned,T...>){(void)((tune==T&&(launch.template operator()<T>(),true))||...);}
            (std::integer_sequence<unsigned,8274U,16466U,24658U,32850U>{});
        return;
    }
    if(wmma) {
        // 16 waves per CTA (measured best against 8; down also against 4).
        constexpr unsigned project_waves=16U,down_waves=16U;
        if(q20) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_wmma_q20<project_waves>),dim3(batches,request.shape.rows/(8U*project_waves)),
                dim3(32U*project_waves),0,stream,gate,up,rows,request.shape.hidden,request.shape.rows);
            check(hipGetLastError(),"A8 prefill WMMA Q2_0 gate/up projection");
            hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_wmma_q20<down_waves>),dim3(batches,request.shape.hidden/(16U*down_waves)),
                dim3(32U*down_waves),0,stream,down_weights,rows,request.shape.hidden,request.shape.rows);
            check(hipGetLastError(),"A8 prefill WMMA Q2_0 down");
            return;
        }
        const dim3 project_grid(batches,request.shape.rows/(8U*project_waves));
        if(q5k)hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_wmma<true,project_waves>),project_grid,dim3(32U*project_waves),0,stream,
            gate,up,rows,request.shape.hidden,request.shape.rows);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_wmma<false,project_waves>),project_grid,dim3(32U*project_waves),0,stream,
            gate,up,rows,request.shape.hidden,request.shape.rows);
        check(hipGetLastError(),"A8 prefill WMMA gate/up projection");
        const dim3 down_grid(batches,request.shape.hidden/(16U*down_waves));
        if(down_type==7U)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_wmma<true,down_waves>),down_grid,dim3(32U*down_waves),0,
            stream,down_weights,rows,request.shape.hidden,request.shape.rows);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_wmma<false,down_waves>),down_grid,dim3(32U*down_waves),0,
            stream,down_weights,rows,request.shape.hidden,request.shape.rows);
        check(hipGetLastError(),"A8 prefill WMMA integer down");
        return;
    }
    const dim3 project_grid(batches,request.shape.rows/32U);
    if(q20) {
        const auto iq_project=[&]<std::uint32_t T>(){hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_iq<T,4U>),project_grid,dim3(256),0,
            stream,gate,up,rows,request.shape.hidden,request.shape.rows);};
        if(request.gate_type==gsq_a8::type_iq2_s)iq_project.template operator()<gsq_a8::type_iq2_s>();
        else if(request.gate_type==gsq_a8::type_iq3_xxs)iq_project.template operator()<gsq_a8::type_iq3_xxs>();
        else if(request.gate_type==gsq_a8::type_iq3_s)iq_project.template operator()<gsq_a8::type_iq3_s>();
        else if(request.gate_type==gsq_a8::type_iq4_xs)iq_project.template operator()<gsq_a8::type_iq4_xs>();
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_q20<4U>),project_grid,dim3(256),0,stream,gate,up,rows,
            request.shape.hidden,request.shape.rows);
        check(hipGetLastError(),"A8 prefill GSQ gate/up projection");
        if(down_type==gsq_a8::type_iq4_nl)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_iq<gsq_a8::type_iq4_nl,8U>),
            dim3(batches,request.shape.hidden/64U),dim3(256),0,stream,down_weights,rows,request.shape.hidden,request.shape.rows);
        else if(down_type==42U)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_q20<8U>),dim3(batches,request.shape.hidden/64U),dim3(256),0,
            stream,down_weights,rows,request.shape.hidden,request.shape.rows);
        else if(down_type==7U)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill<8U,true>),dim3(batches,request.shape.hidden/64U),dim3(256),0,
            stream,down_weights,rows,request.shape.hidden,request.shape.rows);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill<8U,false>),dim3(batches,request.shape.hidden/64U),dim3(256),0,
            stream,down_weights,rows,request.shape.hidden,request.shape.rows);
        check(hipGetLastError(),"A8 prefill Q2_0 integer down");
        return;
    }
    if(q5k)hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill<4U,true>),project_grid,dim3(256),0,stream,gate,up,rows,
        request.shape.hidden,request.shape.rows);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill<4U,false>),project_grid,dim3(256),0,stream,gate,up,rows,
        request.shape.hidden,request.shape.rows);
    check(hipGetLastError(),"A8 prefill gate/up projection");
    const dim3 down_grid(batches,request.shape.hidden/64U);
    if(down_type==7U)hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill<8U,true>),down_grid,dim3(256),0,stream,down_weights,rows,
        request.shape.hidden,request.shape.rows);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill<8U,false>),down_grid,dim3(256),0,stream,down_weights,rows,
        request.shape.hidden,request.shape.rows);
    check(hipGetLastError(),"A8 prefill integer down");
}
void set_prefill_core2(int core){prefill_core2_override()=core;}
void set_prefill_route(int bits){prefill_route_override()=bits;}
void set_prefill_da16(int on){prefill_da16_override()=on;}
void set_prefill_tile64(int on){prefill_tile64_override()=on;}
void set_prefill_test_events(hipEvent_t* events){prefill_test_events()=events;}
void set_prefill_hdr(int on){prefill_hdr_override()=on;}
void set_prefill_tiers(int on){prefill_tiers_override()=on;}
bool prefill_tiers_ran(){return prefill_tiers_flag();}
int prefill_tiers_mode_ran(){return prefill_tiers_flag();}
void set_prefill_down_lds(int on){prefill_down_lds_override()=on;}
bool prefill_down_lds_ran(){return prefill_down_lds_flag();}
void prefill_test_da16_events(const std::array<hipEvent_t,3>* events){prefill_da16_events()=events;}
const void* prefill_test_da16_data(){return da16_image();}
const unsigned* prefill_test_da16_arrivals(){return da16_counters();}
void set_prefill_xor64(int on){prefill_xor64_override()=on;}
unsigned prefill_last_tune(){return prefill_last_tune_value();}
void set_prefill_small(int rows){prefill_small_override()=rows;}
bool prefill_small_ran(){return prefill_small_flag();}
bool prefill_da16_ran(){return prefill_da16_flag();}
bool prefill_tile64_ran(){return prefill_tile64_flag();}
bool prefill_certificate_produced(){return prefill_certificate_flag();}
bool prefill_clear_skipped(){return prefill_clear_skipped_flag();}
void reserve_prefill_pools(unsigned rows) {
    prefill_pool_owner().setup(rows, [] {
        (void)f16_prefill_a16(true); (void)da16_image(true);
        if(prefill_tiers())(void)prefill_tier_lists(true);
        (void)da16_counters(true); (void)p1_counter_pool(true);
        (void)fuse_counter_pool(true); (void)csr_mw_scratch(true);
    });
}
std::array<std::uint64_t, 2> prefill_test_image_generations() {
    return {f16_prefill_a16_pool().generation(), da16_image_pool().generation()};
}
void prefill_test_poison_da16(hipStream_t stream, unsigned rows) {
    reserve_prefill_pools(rows);
    check(hipMemsetAsync(da16_image(),0xa5,static_cast<std::size_t>(prefill_pool_rows())*max_top_k*max_rows*2U,stream),"poison DA16 image");
}
const void* prefill_test_da16_image(){return da16_image();}
void prefill_test_poison_a16(hipStream_t stream, unsigned rows) {
    reserve_prefill_pools(rows);
    check(hipMemsetAsync(f16_prefill_a16(),0xa5,static_cast<std::size_t>(prefill_pool_rows())*max_hidden*2U,stream),"poison FP16 prefill A image");
}
bool prefill_fused_merge(){return prefill_fused_flag();}
std::array<unsigned,8> prefill_p1_stats() {
    std::array<unsigned,8> out{};
    check(hipDeviceSynchronize(),"P1 stats sync");
    check(hipMemcpy(out.data(),p1_counter_pool()+p1_pool_words()-p1_stats_words,sizeof(out),hipMemcpyDeviceToHost),"P1 stats read");
    return out;
}
namespace {
// The ordered merge's guards and non-finite flag after a fused down (known_moe_prefill_merge_kernel with the dot chain
// already done by the last-arriving down CTA): a row the merge skips (any failure in header[0], a row status, an invalid
// route) keeps the prologue's zero accumulation - the fused tail may have written it, so it is zeroed again here.
__global__ void prefill_merge_check_kernel(PrefillRows v,unsigned* header,unsigned hidden,unsigned experts,unsigned top_k,const unsigned* flag) {
    const unsigned row=blockIdx.y;
    unsigned* status=at(v.status,row);
    float* const acc_row=at(v.clear,row);
    const unsigned out=blockIdx.x*blockDim.x+threadIdx.x;
    // Guards in the merge kernel's order; a skipped row keeps (here: gets back) the prologue's +0 accumulation.
    if(header[0]||*status){if(out<hidden)acc_row[out]=0.0F;return;}
    if(out>=hidden)return;
    for(unsigned slot=0;slot<top_k;++slot)if(at(v.routes,row)[slot]>=experts){acc_row[out]=0.0F;return;}
    float current;
    if(*flag) {
        // Some fused value was non-finite: run the merge kernel's chain itself (same dots, order, instruction), so the
        // header/status race outcome follows the serial merge.
        const float* raw=at(v.rawdots,row);const float* w=at(v.route_weights,row);
        float dots[10];
#pragma unroll
        for(unsigned slot=0;slot<10U;++slot)if(slot<top_k)dots[slot]=raw[static_cast<std::uint64_t>(slot)*hidden+out];
        current=0.0F;
#pragma unroll
        for(unsigned slot=0;slot<10U;++slot)if(slot<top_k) {
            const float weight=w[slot];
            asm volatile("v_fmac_f32 %0, %1, %2" : "+v"(current) : "s"(weight), "v"(dots[slot]));
        }
        acc_row[out]=current;
    } else current=acc_row[out];
    if((__float_as_uint(current)&0x7fffffffU)>=0x7f7f8000U){atomicOr(status,8U);atomicOr(header,8U);}
}
} // namespace
void enqueue_prefill_merge_check(const PrefillRows& rows,unsigned count,unsigned hidden,unsigned experts,unsigned top_k,hipStream_t stream) {
    const unsigned* flag=fuse_counter_pool()+static_cast<std::size_t>(count)*(hidden/256U);
    hipLaunchKernelGGL(prefill_merge_check_kernel,dim3((hidden+255U)/256U,count),dim3(256),0,stream,rows,rows.csr.summary,hidden,experts,top_k,flag);
    check(hipGetLastError(),"fused merge check");
}
std::array<unsigned long long,9> fast_dequant_selftest(unsigned long long seed,unsigned blocks) {
    unsigned long long* counts=nullptr;
    check(hipMalloc(&counts,9U*sizeof(unsigned long long)),"fast dequant self-test allocation");
    check(hipMemset(counts,0,9U*sizeof(unsigned long long)),"fast dequant self-test clear");
    hipLaunchKernelGGL(fast_dequant_selftest_kernel,dim3(blocks),dim3(256),0,nullptr,seed,counts);
    check(hipGetLastError(),"fast dequant self-test launch");
    std::array<unsigned long long,9> out{};
    check(hipMemcpy(out.data(),counts,sizeof(out),hipMemcpyDeviceToHost),"fast dequant self-test read");
    check(hipFree(counts),"fast dequant self-test free");
    return out;
}
std::array<unsigned long long,5> gsq_q20_decode_selftest(unsigned seed,unsigned rows) {
    const unsigned count=64U*rows;
    unsigned char* blocks=nullptr;unsigned long long* counts=nullptr;
    check(hipMalloc(&blocks,18ULL*count+16U),"GSQ Q2_0 decode self-test blocks");
    check(hipMalloc(&counts,5U*sizeof(unsigned long long)),"GSQ Q2_0 decode self-test counts");
    check(hipMemset(counts,0,5U*sizeof(unsigned long long)),"GSQ Q2_0 decode self-test clear");
    hipLaunchKernelGGL(gsq_q20_fill_kernel,dim3((count+255U)/256U),dim3(256),0,nullptr,blocks,count,seed);
    hipLaunchKernelGGL(gsq_q20_decode_kernel,dim3((count+255U)/256U),dim3(256),0,nullptr,blocks,count,counts);
    check(hipGetLastError(),"GSQ Q2_0 decode self-test launch");
    std::array<unsigned long long,5> out{};
    check(hipMemcpy(out.data(),counts,sizeof(out),hipMemcpyDeviceToHost),"GSQ Q2_0 decode self-test read");
    check(hipFree(blocks),"GSQ Q2_0 decode self-test free");check(hipFree(counts),"GSQ Q2_0 decode self-test free");
    return out;
}
std::array<unsigned long long,5> gsq_iq_decode_selftest(std::uint32_t type,unsigned seed,unsigned blocks) {
    if(type!=gsq_a8::type_iq2_s&&type!=gsq_a8::type_iq3_xxs&&type!=gsq_a8::type_iq3_s&&type!=gsq_a8::type_iq4_xs&&type!=gsq_a8::type_iq4_nl)
        throw std::invalid_argument("GSQ IQ decode self-test: not an IQ type");
    const unsigned width=type==gsq_a8::type_iq4_nl?640U:2560U,be=gsq_a8::block_elements(type),per_row=width/be;
    blocks=(blocks+per_row-1U)/per_row*per_row;
    const unsigned groups=blocks*(be/32U);
    unsigned char* bytes=nullptr;unsigned long long* counts=nullptr;
    const std::size_t total=static_cast<std::size_t>(blocks)*gsq_a8::block_bytes(type);
    check(hipMalloc(&bytes,total),"GSQ IQ decode self-test blocks");  // no slack: the decoders read inside the rows
    check(hipMalloc(&counts,3U*sizeof(unsigned long long)),"GSQ IQ decode self-test counts");
    check(hipMemset(counts,0,3U*sizeof(unsigned long long)),"GSQ IQ decode self-test clear");
    const auto run=[&]<std::uint32_t T>() {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gsq_iq_fill_kernel<T>),dim3((blocks+255U)/256U),dim3(256),0,nullptr,bytes,blocks,seed);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gsq_iq_decode_kernel<T>),dim3((groups+255U)/256U),dim3(256),0,nullptr,bytes,groups,width,counts);
    };
    if(type==gsq_a8::type_iq2_s)run.template operator()<gsq_a8::type_iq2_s>();
    else if(type==gsq_a8::type_iq3_xxs)run.template operator()<gsq_a8::type_iq3_xxs>();
    else if(type==gsq_a8::type_iq3_s)run.template operator()<gsq_a8::type_iq3_s>();
    else if(type==gsq_a8::type_iq4_xs)run.template operator()<gsq_a8::type_iq4_xs>();
    else run.template operator()<gsq_a8::type_iq4_nl>();
    check(hipGetLastError(),"GSQ IQ decode self-test launch");
    std::array<unsigned long long,5> out{};
    check(hipMemcpy(out.data(),counts,3U*sizeof(unsigned long long),hipMemcpyDeviceToHost),"GSQ IQ decode self-test read");
    // Coverage of the fixture (host): distinct grid indices (or IQ4 nibble values) and sign patterns over all groups.
    std::vector<unsigned char> host(total);
    check(hipMemcpy(host.data(),bytes,total,hipMemcpyDeviceToHost),"GSQ IQ decode self-test bytes");
    std::vector<char> grid(1024,0),signs(256,0);
    const unsigned bb=gsq_a8::block_bytes(type);
    for(unsigned b=0;b<blocks;++b) {
        const unsigned char* x=host.data()+static_cast<std::size_t>(b)*bb;
        if(type==gsq_a8::type_iq4_nl||type==gsq_a8::type_iq4_xs) {
            const unsigned at=type==gsq_a8::type_iq4_nl?2U:8U;
            for(unsigned i=at;i<bb;++i){grid[x[i]&15U]=1;grid[x[i]>>4U]=1;}
            continue;
        }
        for(unsigned ib=0;ib<8U;++ib) {
            if(type==gsq_a8::type_iq3_s) {
                for(unsigned m=0;m<8U;++m)grid[x[2U+8U*ib+m]|(((x[66U+ib]>>m)&1U)<<8U)]=1;
                for(unsigned l=0;l<4U;++l)signs[x[74U+4U*ib+l]]=1;
            } else if(type==gsq_a8::type_iq3_xxs) {
                for(unsigned m=0;m<8U;++m)grid[x[2U+8U*ib+m]]=1;
                const unsigned aux=static_cast<unsigned>(x[66U+4U*ib])|static_cast<unsigned>(x[67U+4U*ib])<<8U|static_cast<unsigned>(x[68U+4U*ib])<<16U|static_cast<unsigned>(x[69U+4U*ib])<<24U;
                for(unsigned l=0;l<4U;++l)signs[(aux>>(7U*l))&127U]=1;
            } else {
                for(unsigned l=0;l<4U;++l){grid[x[2U+4U*ib+l]|(((x[66U+ib]>>(2U*l))&3U)<<8U)]=1;signs[x[34U+4U*ib+l]]=1;}
            }
        }
    }
    for(const char c:grid)out[3]+=c?1ULL:0ULL;
    for(const char c:signs)out[4]+=c?1ULL:0ULL;
    check(hipFree(bytes),"GSQ IQ decode self-test free");check(hipFree(counts),"GSQ IQ decode self-test free");
    return out;
}
void preflight(const Request& request,std::span<const Bindings> rows,const q51_known_rows::Workspace& workspace,hipStream_t stream) {
    (void)prepare(request,rows,workspace,stream);
}
void enqueue(const Request& request,std::span<const Bindings> rows,const q51_known_rows::Workspace& workspace,hipStream_t stream) {
    const auto p=prepare(request,rows,workspace,stream);check(hipPeekAtLastError(),"A8 CSR gate/up stale launch error");
    admit_csr<<<1,1,0,stream>>>(p.rows,p.csr,static_cast<unsigned>(rows.size()),request.shape.experts,request.shape.top_k);
    check(hipGetLastError(),"A8 CSR gate/up device CSR admission");
#define STRIX_A8_CSR_PROJECT(R) \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(project_csr<R>),dim3(request.shape.rows/32U,static_cast<unsigned>(rows.size())*request.shape.top_k), \
        dim3(256),0,stream,reinterpret_cast<const unsigned char*>(rows.front().gate.address), \
        reinterpret_cast<const unsigned char*>(rows.front().up.address),p.rows,p.csr,request.shape.hidden,request.shape.rows)
    if(rows.size()==2U){STRIX_A8_CSR_PROJECT(2);}
    else if(rows.size()==4U){STRIX_A8_CSR_PROJECT(4);}
    else {STRIX_A8_CSR_PROJECT(8);}
#undef STRIX_A8_CSR_PROJECT
    check(hipGetLastError(),"A8 CSR shared gate/up projection");
}

void set_prefill_direct_b(int on){direct_b_override()=on;}
bool prefill_direct_b_ran(){return direct_b_ran();}
const char* prefill_direct_b_reason(){return direct_b_reason();}
void prefill_project_test(const PrefillRows& rows,unsigned hidden,unsigned outputs,const unsigned char* gate,
    const unsigned char* up,unsigned batches,unsigned tune,bool direct,hipStream_t stream,DirectBOperands direct_assets,
    unsigned* arrive,uint4* image) {
    if(batches>da16_counter_words()||!arrive||!image)throw std::invalid_argument("direct B timing completion assets");
    const auto run=[&]<unsigned Tune>() {
        const uint4* a16=(Tune&16U)?f16_prefill_a16():nullptr;
        if(direct) {
            using DT=DirectBTile<Tune>;
            static bool reported=false;
            if(!reported) {
                prefill_resources_at(reinterpret_cast<const void*>(direct_b_kernel<Tune,true>()),"gate_up_q4k_direct_b_da16",Tune,DT::Threads);
                reported=true;
            }
            launch_prefill_direct_b<Tune,true>(dim3(outputs/DT::Outputs,batches),stream,
                direct_assets.gate,direct_assets.up,rows,hidden,outputs,a16,arrive,image);
        } else {
            using T=F16Tile<Tune>;
            static bool reported=false;
            if(!reported) {
                prefill_resources_at(reinterpret_cast<const void*>(project_prefill_f16x_da16_mg<false,Tune>),"gate_up_q4k_da16_mg",Tune,T::Threads);
                reported=true;
            }
            hipLaunchKernelGGL(HIP_KERNEL_NAME(project_prefill_f16x_da16_mg<false,Tune>),dim3(outputs/(T::Columns/2U),batches),dim3(T::Threads),0,stream,
                gate,up,rows,hidden,outputs,a16,arrive,image);
        }
        check(hipGetLastError(),"gate/up fixture projection");
    };
    switch(tune){case 80:run.template operator()<80>();break;case 82:run.template operator()<82>();break;
        case 88:run.template operator()<88>();break;default:throw std::invalid_argument("direct B fixture tune");}
}
DirectBFragmentResult direct_b_fragment_test(const unsigned char* raw,const unsigned char* packed,unsigned pairs,bool paired_planes) {
    // Keep the flattened coordinate within 32 bits above the two FP16 values.
    if(!pairs||pairs>(1U<<18U))throw std::invalid_argument("direct B fragment pair count");
    unsigned long long* stats=nullptr;check(hipMalloc(&stats,32),"direct B fragment counters");
    check(hipMemset(stats,0,24),"direct B fragment clear");
    check(hipMemset(stats+3,255,8),"direct B first fragment clear");
    if(paired_planes) {
        hipLaunchKernelGGL(direct_b_fragment_probe<true>,dim3(2U*pairs),dim3(32),0,nullptr,raw,packed,stats);
    } else {
        hipLaunchKernelGGL(direct_b_fragment_probe<false>,dim3(pairs),dim3(32),0,nullptr,raw,packed,stats);
    }
    check(hipGetLastError(),"direct B fragment probe");
    std::array<unsigned long long,4> result{};
    check(hipMemcpy(result.data(),stats,32,hipMemcpyDeviceToHost),"direct B fragment result");
    check(hipFree(stats),"direct B fragment free");
    DirectBFragmentResult out{result[0],result[1],result[2]};
    if(out.mismatched_elements) {
        out.actual=static_cast<unsigned>(result[3]&65535U);out.expected=static_cast<unsigned>((result[3]>>16U)&65535U);
        auto coordinate=result[3]>>32U;
        out.element=coordinate%16U;coordinate/=16U;out.plane=coordinate%16U;coordinate/=16U;
        out.replica=coordinate%2U;coordinate/=2U;out.lane=coordinate%16U;out.unit=static_cast<unsigned>(coordinate/16U);
    }
    return out;
}
} // namespace strix::moe_q4k_a8::known_rows
