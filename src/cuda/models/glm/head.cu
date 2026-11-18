#include "head.cuh"
#include "producer/head_select.hpp"
#include "row_mma.cuh"
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <algorithm>

namespace strix::glm::rtxpf {
namespace {
#include "projection.inl"
__global__ __launch_bounds__(1024) void argmax_kernel(const float* x,unsigned n,U32* result) {
    __shared__ float sv[1024];__shared__ U32 si[1024],sn[1024];unsigned l=threadIdx.x;
    float best=0;U32 id=producer::no_token,bad=0;
    for(unsigned j=l;j<n;j+=1024){float v=x[j];if(!isfinite(v)){++bad;continue;}if(producer::head_better(v,j,best,id)){best=v;id=j;}}
    sv[l]=best;si[l]=id;sn[l]=bad;__syncthreads();
    for(unsigned h=512;h;h/=2){if(l<h){if(producer::head_better(sv[l+h],si[l+h],sv[l],si[l])){sv[l]=sv[l+h];si[l]=si[l+h];}sn[l]+=sn[l+h];}__syncthreads();}
    if(l==0){result[0]=si[0];result[1]=sn[0];result[2]=__float_as_uint(sv[0]);}
}
}
std::size_t head_decoded_bytes(){return std::size_t(head_hidden)*head_vocab*2;}
std::size_t head_workspace_bytes(unsigned rows){return ((std::size_t(rows)*head_hidden*2+255)&~std::size_t(255))+std::size_t(rows)*head_vocab*4;}
cudaError_t decode_head(Matrix w,U16* q,cudaStream_t s){
    if(!w.packed||!q||w.input!=head_hidden||w.output!=head_vocab||w.bits!=6)return cudaErrorInvalidValue;
    layer_decode<<<w.input/16*(w.output/16),256,0,s>>>(w,q);return cudaGetLastError();
}
cudaError_t enqueue_head(const HeadWeights& h,const float* x,float* logits,unsigned rows,void* ws,std::size_t bytes,cudaStream_t s){
    if(!h.decoded||!h.suh||!h.svh||!x||!logits||!rows||rows>64||!ws||reinterpret_cast<std::uintptr_t>(ws)%256||bytes<head_workspace_bytes(rows))return cudaErrorInvalidValue;
    auto* operand=static_cast<U16*>(ws);auto* raw=reinterpret_cast<float*>(static_cast<char*>(ws)+((std::size_t(rows)*head_hidden*2+255)&~std::size_t(255)));
    input_transform<<<(rows*(head_hidden/128)+7)/8,256,0,s>>>(x,operand,h.suh,rows,head_hidden);
    projection_reference<<<unsigned((std::size_t(rows)*head_vocab+255)/256),256,0,s>>>(nullptr,operand,h.decoded,raw,rows,head_hidden,head_vocab,false);
    output_transform<<<(rows*(head_vocab/128)+7)/8,256,0,s>>>(raw,logits,h.svh,rows,head_vocab);
    return cudaGetLastError();
}
cudaError_t enqueue_argmax(const float* x,unsigned n,U32* result,cudaStream_t s){
    if(!x||!n||!result)return cudaErrorInvalidValue;argmax_kernel<<<1,1024,0,s>>>(x,n,result);return cudaGetLastError();
}
}
