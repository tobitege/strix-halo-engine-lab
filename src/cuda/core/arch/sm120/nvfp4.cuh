#pragma once
#include "strix/cuda/core/nvfp4_cuda.hpp"
namespace strix::cuda::core::nvfp4::sm120 {
// PTX ISA: warp m16n8k64 NVFP4, four E4M3 scales per row.
// g=lane/4,t=lane%4: A words (g,8t),(g+8,8t), then K+32;
// B words (g,8t),(g,8t+32). Scale A row g+8*(lane%2), B row g.
__device__ __forceinline__ void mma(float* d,const unsigned* a,const unsigned* b,
                                   unsigned sa,unsigned sb) {
#if __CUDA_ARCH__ == 1200
    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
#else
    asm volatile("trap;");
#endif
}
template<bool Decode> __global__ void dense(View v,Plan p,float* scratch) {
    constexpr unsigned BM=64, BN=Decode?8:32;
    __shared__ unsigned qa[BM*8], qb[BN*8], sa[BM], sb[BN];
    const unsigned lane=threadIdx.x&31, warp=threadIdx.x/32;
    const unsigned g=lane/4,t=lane&3;
    const Matrix a=Decode?v.b:v.a, b=Decode?v.a:v.b;
    const unsigned rows=Decode?v.n:v.m, cols=Decode?v.m:v.n;
    const unsigned r0=(Decode?blockIdx.x:blockIdx.y)*BM;
    const unsigned c0=(Decode?blockIdx.y:blockIdx.x)*BN;
    float d[BN/8][4] = {};
    const unsigned blocks=v.k/64;
    const unsigned begin=unsigned(std::uint64_t(blocks)*blockIdx.z/p.splits);
    const unsigned end=unsigned(std::uint64_t(blocks)*(blockIdx.z+1)/p.splits);
    for(unsigned block=begin;block<end;++block) {
        for(unsigned i=threadIdx.x;i<BM*8;i+=blockDim.x) {
            const unsigned row=r0+i/8;
            qa[i]=row<rows?reinterpret_cast<const unsigned*>(a.codes+row*a.code_stride+block*32)[i%8]:0;
        }
        for(unsigned i=threadIdx.x;i<BN*8;i+=blockDim.x) {
            const unsigned row=c0+i/8;
            qb[i]=row<cols?reinterpret_cast<const unsigned*>(b.codes+row*b.code_stride+block*32)[i%8]:0;
        }
        if(threadIdx.x<BM) {
            const unsigned row=r0+threadIdx.x;
            sa[threadIdx.x]=row<rows?reinterpret_cast<const unsigned*>(a.scales+row*a.scale_stride)[block]:0;
        }
        if(threadIdx.x<BN) {
            const unsigned row=c0+threadIdx.x;
            sb[threadIdx.x]=row<cols?reinterpret_cast<const unsigned*>(b.scales+row*b.scale_stride)[block]:0;
        }
        __syncthreads();
        const unsigned r=warp*16+g;
        const unsigned af[4]={qa[r*8+t],qa[(r+8)*8+t],qa[r*8+t+4],qa[(r+8)*8+t+4]};
        const unsigned sf=sa[warp*16+g+(lane&1)*8];
        #pragma unroll
        for(unsigned j=0;j<BN/8;++j) {
            const unsigned bf[2]={qb[(j*8+g)*8+t],qb[(j*8+g)*8+t+4]};
            mma(d[j],af,bf,sf,sb[j*8+g]);
        }
        __syncthreads();
    }
    #pragma unroll
    for(unsigned j=0;j<BN/8;++j) {
        #pragma unroll
        for(unsigned z=0;z<4;++z) {
            const unsigned row=r0+warp*16+g+(z/2)*8,col=c0+j*8+t*2+z%2;
            const unsigned m=Decode?col:row,n=Decode?row:col;
            if(m<v.m && n<v.n) {
                if(p.splits>1) scratch[(std::size_t(blockIdx.z)*v.m+m)*v.n+n]=d[j][z];
                else {
                    float out=__fmul_rn(d[j][z],__fmul_rn(v.a.tensor_scale,v.b.tensor_scale));
                    if(v.residual) out=__fadd_rn(out,v.residual[m*v.residual_stride+n]);
                    v.output[m*v.output_stride+n]=out;
                }
            }
        }
    }
}
__global__ void merge(View v,Plan p,const float* scratch) {
    const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=std::size_t(v.m)*v.n) return;
    float sum=0;
    for(unsigned z=0;z<p.splits;++z) sum=__fadd_rn(sum,scratch[std::size_t(z)*v.m*v.n+i]);
    float out=__fmul_rn(sum,__fmul_rn(v.a.tensor_scale,v.b.tensor_scale));
    const std::size_t m=i/v.n,n=i%v.n;
    if(v.residual) out=__fadd_rn(out,v.residual[m*v.residual_stride+n]);
    v.output[m*v.output_stride+n]=out;
}
}
