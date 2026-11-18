#include "arch/sm120/nvfp4.cuh"
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cmath>
#include <limits>
namespace strix::cuda::core::nvfp4 {
static cudaError_t capability() {
    int device=0,major=0,minor=0;
    auto err=cudaGetDevice(&device); if(err!=cudaSuccess) return err;
    err=cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device); if(err!=cudaSuccess) return err;
    err=cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device); if(err!=cudaSuccess) return err;
    return major==12 && minor==0?cudaSuccess:cudaErrorNotSupported;
}
__global__ void quantize_kernel(const float* input,std::size_t xs,unsigned rows,unsigned k,
    std::uint8_t* q,std::size_t qs,std::uint8_t* s,std::size_t ss,float ts) {
    const std::size_t group=(std::size_t(blockIdx.x)*blockDim.x+threadIdx.x)/8;
    if(group>=std::size_t(rows)*(k/16)) return; // complete 8-lane subgroups
    const unsigned lane=threadIdx.x&7;
    const std::size_t row=group/(k/16),block=group%(k/16);
    const float x=input[row*xs+block*16+lane*2],y=input[row*xs+block*16+lane*2+1];
    float amax=fmaxf(fabsf(x),fabsf(y));
    #pragma unroll
    for(unsigned d=4;d;d/=2) amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,d,8));
    const float desired=fminf(448.f,__fdiv_rn(__fdiv_rn(amax,6.f),ts));
    __nv_fp8_e4m3 sc;sc.__x=__nv_cvt_float_to_fp8(desired,__NV_SATFINITE,__NV_E4M3);
    const float scale=__fmul_rn(float(sc),ts);
    const float u=scale?fminf(6.f,fmaxf(-6.f,__fdiv_rn(x,scale))):0.f;
    const float v=scale?fminf(6.f,fmaxf(-6.f,__fdiv_rn(y,scale))):0.f;
    q[row*qs+block*8+lane]=__nv_cvt_float2_to_fp4x2(make_float2(u,v),__NV_E2M1,cudaRoundNearest);
    if(lane==0) s[row*ss+block]=sc.__x;
}
cudaError_t quantize_enqueue(const float* input,std::size_t xs,unsigned rows,unsigned k,
    std::uint8_t* q,std::size_t qs,std::uint8_t* s,std::size_t ss,float ts,cudaStream_t stream) {
    const auto max=std::numeric_limits<std::size_t>::max();
    if(!input || !q || !s || !rows || !k || k%64 || xs<k || qs<k/2 || ss<k/16 ||
       reinterpret_cast<std::uintptr_t>(input)%alignof(float)!=0 ||
       !std::isfinite(ts) || ts<=0 || xs>max/rows/sizeof(float) || qs>max/rows || ss>max/rows ||
       std::uint64_t(rows)*(k/16)>std::uint64_t(2147483647)*16)
        return cudaErrorInvalidValue;
    const auto err=capability();if(err!=cudaSuccess) return err;
    quantize_kernel<<<unsigned((std::uint64_t(rows)*(k/16)+15)/16),128,0,stream>>>(input,xs,rows,k,q,qs,s,ss,ts);
    return cudaGetLastError();
}
cudaError_t enqueue(const View& v,const Plan& p,void* workspace,std::size_t bytes,cudaStream_t stream) {
    Plan expected;
    try { expected=resolve(v); } catch(...) { return cudaErrorInvalidValue; }
    if(p.m!=expected.m || p.n!=expected.n || p.k!=expected.k || p.splits!=expected.splits ||
       p.route!=expected.route || p.workspace_bytes!=expected.workspace_bytes ||
       bytes<p.workspace_bytes || (p.workspace_bytes && (!workspace || reinterpret_cast<std::uintptr_t>(workspace)%256)))
        return cudaErrorInvalidValue;
    auto err=capability(); if(err!=cudaSuccess) return err;
    // Explicit grid limits; reject rather than silently dropping tail tiles.
    if(v.m>65535u*64u) return cudaErrorInvalidValue;
    auto* scratch=static_cast<float*>(workspace);
    if(p.route==Route::decode)
        sm120::dense<true><<<dim3((v.n-1)/64+1,1,p.splits),128,0,stream>>>(v,p,scratch);
    else sm120::dense<false><<<dim3((v.n-1)/32+1,(v.m-1)/64+1,1),128,0,stream>>>(v,p,scratch);
    err=cudaGetLastError(); if(err!=cudaSuccess) return err;
    if(p.splits>1) {
        sm120::merge<<<unsigned((std::size_t(v.m)*v.n+255)/256),256,0,stream>>>(v,p,scratch);
        return cudaGetLastError();
    }
    return cudaSuccess;
}
}
