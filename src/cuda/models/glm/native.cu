#include "native.cuh"
#include "c1_layout.cuh"
#include "c1_schedule.hpp"
#include "row_mma.cuh"
#include <mma.h>
#include <algorithm>
#include <limits>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace strix::glm::rtxpf {
namespace {
std::size_t align(std::size_t x) { return (x+255)&~std::size_t(255); }
bool shape(unsigned r,unsigned e,unsigned h,unsigned i) {
    return r>0 && r<=32768 && e>0 && e<=288 && h>0 && h<=16384 && i>0 && i<=16384 && h%128==0 && i%128==0;
}
struct Layout {
    U32 *counts,*offsets,*cursor,*map,*tiles,*counter;
    U16 *qg,*qu,*qd,*ag,*au,*ad,*qg2,*qu2,*qd2;
    float *activation,*contribution;
    float *scratch_g,*scratch_u;
    void* problems;
    std::size_t size;
    Layout(void* base,unsigned r,unsigned e,unsigned h,unsigned i,unsigned routes,bool c1=false) {
        std::size_t o=0;
        auto take=[&](std::size_t n) { auto p=reinterpret_cast<unsigned char*>(base); auto a=o; o+=align(n); return p ? p+a : nullptr; };
        counts=(U32*)take(e*4); offsets=(U32*)take((e+1)*4); cursor=(U32*)take(e*4);
        map=(U32*)take(std::size_t(r)*routes*4); tiles=(U32*)take(17*4); counter=(U32*)take(4);
        std::size_t q=std::size_t(std::min(group_size,e))*h*i*2, n=std::size_t(r)*routes;
        qg=(U16*)take(q); qu=(U16*)take(q); qd=(U16*)take(q);
        ag=(U16*)take(n*h*2); au=(U16*)take(n*h*2); ad=(U16*)take(n*i*2);
        // The C1 (CUTLASS) path never reads `activation`: its transform runs with down=false only and
        // its down input is c1_down_transform over `ad`; only the legacy path keeps the buffer.
        activation=c1?nullptr:(float*)take(n*i*4); contribution=(float*)take(n*h*4);
        if(c1) {
            qg2=(U16*)take(q); qu2=(U16*)take(q); qd2=(U16*)take(q);
            scratch_g=(float*)take(n*std::max(h,i)*4); scratch_u=(float*)take(n*i*4);
            problems=take(32768);
        }
        size=o;
    }
};
__global__ void histogram(const U32* ids,U32* counts,unsigned n,unsigned e) {
    unsigned j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<n && ids[j]<e) atomicAdd(counts+ids[j],1);
}
__global__ void prefix(U32* counts,U32* offsets,U32* cursor,unsigned e) {
    if(threadIdx.x==0) { unsigned n=0; for(unsigned j=0;j<e;++j) { offsets[j]=n; cursor[j]=n; n+=counts[j]; } offsets[e]=n; }
}
__global__ void scatter(const U32* ids,U32* cursor,U32* map,unsigned n,unsigned e) {
    unsigned j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<n && ids[j]<e) map[atomicAdd(cursor+ids[j],1)]=j;
}
__global__ void dense_dispatch(U32* counts,U32* offsets,U32* map,unsigned n) {
    unsigned j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<n) map[j]=j;
    if(j==0) { counts[0]=n; offsets[0]=0; offsets[1]=n; }
}
__global__ void tasks(const U32* counts,U32* tiles,U32* counter,unsigned first,unsigned count,unsigned width) {
    if(threadIdx.x==0) {
        unsigned n=0; for(unsigned j=0;j<count;++j) { tiles[j]=n; n+=((counts[first+j]+15)/16)*(width/128); }
        tiles[count]=n; *counter=0;
    }
}
__global__ void decode_kernel(Matrix w,U16* q) {
    std::size_t tile=blockIdx.x;
    unsigned pos=threadIdx.x;
    q[(tile/(w.output/16)*16+qi(pos))*w.output + tile%(w.output/16)*16+qo(pos)] =
        book(state(w.packed+tile*16*w.bits,pos,w.bits));
}
// A group is decoded once, never per row tile. Matrix descriptors may span
// independently loaded safetensor shards; no model-specific file layout here.
__global__ void decode_group(const Expert* ex,U16* qg,U16* qu,U16* qd,unsigned first,unsigned count,unsigned h,unsigned i) {
    unsigned local=blockIdx.y, which=blockIdx.z;
    if(local>=count) return;
    Expert e=ex[first+local]; Matrix w=which==0?e.gate:(which==1?e.up:e.down);
    U16* q=(which==0?qg:(which==1?qu:qd))+std::size_t(local)*h*i;
    unsigned pos=threadIdx.x; std::size_t tile=blockIdx.x;
    q[(tile/(w.output/16)*16+qi(pos))*w.output+tile%(w.output/16)*16+qo(pos)]=book(state(w.packed+tile*16*w.bits,pos,w.bits));
}
// One warp per H128 block, exactly the node's lane/register butterfly order.
__device__ void h128(float (&v)[4],unsigned lane) {
    #pragma unroll
    for(unsigned s=1;s<32;s*=2) {
        #pragma unroll
        for(unsigned j=0;j<4;++j) { float p=__shfl_xor_sync(0xffffffff,v[j],s); v[j]=(lane&s)?__fsub_rn(p,v[j]):__fadd_rn(v[j],p); }
    }
    #pragma unroll
    for(unsigned s=1;s<4;s*=2) {
        #pragma unroll
        for(unsigned j=0;j<4;++j) if(!(j&s)) { float a=v[j],b=v[j+s]; v[j]=__fadd_rn(a,b); v[j+s]=__fsub_rn(a,b); }
    }
    #pragma unroll
    for(unsigned j=0;j<4;++j) v[j]=__fmul_rn(v[j],0x1.6a09e6p-4f);
}
__global__ void transform(const Expert* ex,const float* x,const float* activation,const U32* offsets,const U32* map,
    U16* ag,U16* au,U16* ad,unsigned first,unsigned count,unsigned width,unsigned routes,bool down,const U32* act=nullptr) {
    if(blockIdx.y>=count) return;
    unsigned e=c1_expert(act,first+blockIdx.y);
    unsigned lane=threadIdx.x%32, warp=threadIdx.x/32;
    unsigned nblocks=width/128;
    for(unsigned block=blockIdx.x*8+warp;block<(offsets[e+1]-offsets[e])*nblocks;block+=gridDim.x*8) {
    unsigned row=block/nblocks, col=(block%nblocks)*128;
    unsigned entry=offsets[e]+row, source=map[entry]/routes;
    Expert w=ex[e]; float g[4],u[4];
    #pragma unroll
    for(unsigned j=0;j<4;++j) {
        unsigned c=col+lane+j*32;
        float value=down?activation[std::size_t(entry)*width+c]:x[std::size_t(source)*width+c];
        g[j]=hf(fh(__fmul_rn(value,hf((down?w.down.suh:w.gate.suh)[c]))));
        if(!down) u[j]=hf(fh(__fmul_rn(value,hf(w.up.suh[c]))));
    }
    h128(g,lane); if(!down) h128(u,lane);
    #pragma unroll
    for(unsigned j=0;j<4;++j) {
        std::size_t p=std::size_t(entry)*width+col+lane+j*32;
        (down?ad:ag)[p]=fh(g[j]); if(!down) au[p]=fh(u[j]);
    }
    }
}
// Persistent 16x128 CTA, eight warp 16x16x16 MMA chains. No split-K.
// Gate and up share a task and epilogue; distinct suh transforms are respected.
template<bool GU,bool Reference=false>
__global__ void grouped_mma(const Expert* ex,const U32* offsets,const U32* map,const U32* tiles,U32* counter,
    const U16* ag,const U16* au,const U16* qg,const U16* qu,float* activation,float* out,
    unsigned first,unsigned count,unsigned kdim,unsigned ndim) {
    using namespace nvcuda;
    __shared__ __align__(32) __half a[256],b[16*128],a2[GU?256:1],b2[GU?16*128:1];
    __shared__ __align__(32) float c[16*128],c2[GU?16*128:1];
    __shared__ unsigned task;
    unsigned t=threadIdx.x, warp=t/32;
    for(;;) {
        if(t==0) task=atomicAdd(counter,1); __syncthreads();
        if(task>=tiles[count]) return;
        unsigned local=0; while(local+1<count && task>=tiles[local+1]) ++local;
        unsigned expert=first+local, nt=ndim/128, idx=task-tiles[local];
        unsigned row0=(idx/nt)*16, col0=(idx%nt)*128, begin=offsets[expert], rows=offsets[expert+1]-begin;
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af,af2;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> bf,bf2;
        wmma::fragment<wmma::accumulator,16,16,16,float> cf,cf2;
        wmma::fill_fragment(cf,0.f); if constexpr(GU) wmma::fill_fragment(cf2,0.f);
        float rg[8]{},ru[8]{};
        for(unsigned k=0;k<kdim;k+=16) {
            unsigned r=row0+t/16, kc=k+t%16;
            a[t]=r<rows?__ushort_as_half(ag[std::size_t(begin+r)*kdim+kc]):__float2half_rn(0.f);
            if constexpr(GU) a2[t]=r<rows?__ushort_as_half(au[std::size_t(begin+r)*kdim+kc]):__float2half_rn(0.f);
            for(unsigned j=t;j<16*128;j+=256) {
                std::size_t p=std::size_t(local)*kdim*ndim+std::size_t(k+j/128)*ndim+col0+j%128;
                b[j]=__ushort_as_half(qg[p]); if constexpr(GU) b2[j]=__ushort_as_half(qu[p]);
            }
            __syncthreads();
            if constexpr(Reference) {
                // Each independent output has one scalar ascending-K FP32
                // FMA chain; tiled shared loads change no reduction order.
                #pragma unroll
                for(unsigned kk=0;kk<16;++kk) {
                    #pragma unroll
                    for(unsigned z=0;z<8;++z) {
                        unsigned j=t+z*256;
                        rg[z]=__fmaf_rn(__half2float(a[j/128*16+kk]),__half2float(b[kk*128+j%128]),rg[z]);
                        if constexpr(GU) ru[z]=__fmaf_rn(__half2float(a2[j/128*16+kk]),__half2float(b2[kk*128+j%128]),ru[z]);
                    }
                }
            } else {
                wmma::load_matrix_sync(af,a,16); wmma::load_matrix_sync(bf,b+warp*16,128);
                wmma::mma_sync(cf,af,bf,cf);
                if constexpr(GU) { wmma::load_matrix_sync(af2,a2,16); wmma::load_matrix_sync(bf2,b2+warp*16,128); wmma::mma_sync(cf2,af2,bf2,cf2); }
            }
            __syncthreads();
        }
        if constexpr(Reference) {
            #pragma unroll
            for(unsigned z=0;z<8;++z) { c[t+256*z]=rg[z];if constexpr(GU)c2[t+256*z]=ru[z]; }
        } else {
            wmma::store_matrix_sync(c+warp*16,cf,128,wmma::mem_row_major);
            if constexpr(GU) wmma::store_matrix_sync(c2+warp*16,cf2,128,wmma::mem_row_major);
        }
        __syncthreads();
        // Shared butterflies: each thread owns disjoint output locations,
        // every stride reads the previous complete H128 stage.
        for(unsigned s=1;s<128;s*=2) {
            // Eight independent register slots per thread.
            float vg[8],vu[8];
            #pragma unroll
            for(unsigned z=0;z<8;++z) { unsigned j=t+256*z,lo=j&~s,hi=j|s; vg[z]=(j&s)?__fsub_rn(c[lo],c[hi]):__fadd_rn(c[lo],c[hi]); if constexpr(GU) vu[z]=(j&s)?__fsub_rn(c2[lo],c2[hi]):__fadd_rn(c2[lo],c2[hi]); }
            __syncthreads();
            #pragma unroll
            for(unsigned z=0;z<8;++z) { c[t+256*z]=vg[z]; if constexpr(GU) c2[t+256*z]=vu[z]; }
            __syncthreads();
        }
        Expert e=ex[expert];
        for(unsigned j=t;j<16*128;j+=256) {
            unsigned r=row0+j/128,n=col0+j%128;
            if(r<rows) {
                float g=__fmul_rn(__fmul_rn(c[j],0x1.6a09e6p-4f),hf((GU?e.gate.svh:e.down.svh)[n]));
                if constexpr(GU) { float u=__fmul_rn(__fmul_rn(c2[j],0x1.6a09e6p-4f),hf(e.up.svh[n])); activation[std::size_t(begin+r)*ndim+n]=swiglu(g,u); }
                else out[std::size_t(map[begin+r])*ndim+n]=g;
            }
        }
        __syncthreads();
    }
}
__global__ void merge(const float* contribution,const float* weights,float* y,unsigned r,unsigned h,unsigned routes) {
    std::size_t j=std::size_t(blockIdx.x)*256+threadIdx.x;
    if(j>=std::size_t(r)*h) return;
    unsigned row=unsigned(j/h),c=unsigned(j%h); float acc=0;
    for(unsigned s=0;s<routes;++s) {
        float w=weights?weights[row*routes+s]:1.f;
        acc=__fadd_rn(acc,__fmul_rn(w,contribution[(std::size_t(row)*routes+s)*h+c]));
    }
    y[j]=acc;
}
#include "cutlass_path.inl"

cudaError_t launch(const Call& v,void* ws,std::size_t bytes,cudaStream_t stream,unsigned routes,Profile* prof,bool reference=false) {
    // This path decodes every expert of each group: expert batches are CUTLASS-only.
    if(!v.combine || !shape(v.rows,v.experts,v.hidden,v.inter) ||!ws || !v.x || !v.y || !v.matrices || (routes==8 && (v.experts<8||!v.ids||!v.weights))) return cudaErrorInvalidValue;
    Layout l(ws,v.rows,v.experts,v.hidden,v.inter,routes);
    if(bytes<l.size || reinterpret_cast<std::uintptr_t>(ws)%256) return cudaErrorInvalidValue;
    auto err=cudaMemsetAsync(l.counts,0,v.experts*4,stream); if(err!=cudaSuccess) return err;
    unsigned entries=v.rows*routes;
    if(routes==1) dense_dispatch<<<(entries+255)/256,256,0,stream>>>(l.counts,l.offsets,l.map,entries);
    else {
        histogram<<<(entries+255)/256,256,0,stream>>>(v.ids,l.counts,entries,v.experts);
        prefix<<<1,1,0,stream>>>(l.counts,l.offsets,l.cursor,v.experts);
        scatter<<<(entries+255)/256,256,0,stream>>>(v.ids,l.cursor,l.map,entries,v.experts);
    }
    for(unsigned first=0;first<v.experts;first+=group_size) {
        unsigned n=std::min(group_size,v.experts-first),g=first/group_size;
        if(prof) { err=cudaEventRecord(prof->begin[g],stream); if(err!=cudaSuccess)return err; }
        decode_group<<<dim3(v.hidden/16*(v.inter/16),n,3),256,0,stream>>>(v.matrices,l.qg,l.qu,l.qd,first,n,v.hidden,v.inter);
        if(prof) { err=cudaEventRecord(prof->decoded[g],stream); if(err!=cudaSuccess)return err; }
        // Upper bound avoids host readback of dispatch counts. Idle warps exit.
        transform<<<dim3(256,n),256,0,stream>>>(v.matrices,v.x,l.activation,l.offsets,l.map,l.ag,l.au,l.ad,first,n,v.hidden,routes,false);
        tasks<<<1,1,0,stream>>>(l.counts,l.tiles,l.counter,first,n,v.inter);
        if(reference) grouped_mma<true,true><<<256,256,0,stream>>>(v.matrices,l.offsets,l.map,l.tiles,l.counter,l.ag,l.au,l.qg,l.qu,l.activation,l.contribution,first,n,v.hidden,v.inter);
        else grouped_mma<true><<<256,256,0,stream>>>(v.matrices,l.offsets,l.map,l.tiles,l.counter,l.ag,l.au,l.qg,l.qu,l.activation,l.contribution,first,n,v.hidden,v.inter);
        if(prof) { err=cudaEventRecord(prof->gate_up[g],stream); if(err!=cudaSuccess)return err; }
        transform<<<dim3(256,n),256,0,stream>>>(v.matrices,v.x,l.activation,l.offsets,l.map,l.ag,l.au,l.ad,first,n,v.inter,routes,true);
        tasks<<<1,1,0,stream>>>(l.counts,l.tiles,l.counter,first,n,v.hidden);
        if(reference) grouped_mma<false,true><<<256,256,0,stream>>>(v.matrices,l.offsets,l.map,l.tiles,l.counter,l.ad,nullptr,l.qd,nullptr,l.activation,l.contribution,first,n,v.inter,v.hidden);
        else grouped_mma<false><<<256,256,0,stream>>>(v.matrices,l.offsets,l.map,l.tiles,l.counter,l.ad,nullptr,l.qd,nullptr,l.activation,l.contribution,first,n,v.inter,v.hidden);
        if(prof) { err=cudaEventRecord(prof->down[g],stream); if(err!=cudaSuccess)return err; }
    }
    merge<<<unsigned((std::size_t(v.rows)*v.hidden+255)/256),256,0,stream>>>(l.contribution,v.weights,v.y,v.rows,v.hidden,routes);
    return cudaGetLastError();
}
}

std::size_t workspace_bytes_cutlass(unsigned r,unsigned e,unsigned h,unsigned i) {
    if(!shape(r,e,h,i)) return 0;
    return Layout(nullptr,r,e,h,i,e==1?1:topk,true).size;
}
cudaError_t prepare_cutlass(CutlassExecution& e) {
    e.hot_blocks=HotGemm::sufficient(); e.cold_blocks=ColdGemm::sufficient();
    int device=0,sms=0,resident=0;
    auto err=cudaGetDevice(&device);if(err!=cudaSuccess)return err;
    err=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);if(err!=cudaSuccess)return err;
    err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,decode_window,256,0);
    if(err!=cudaSuccess)return err;
    // At most two decoder CTAs per SM: leave capacity for concurrent GEMMs.
    e.decode_blocks=sms*std::min(2,resident);
    return e.hot_blocks>0 && e.cold_blocks>0 && e.decode_blocks>0 ? cudaSuccess : cudaErrorInvalidConfiguration;
}
cudaError_t enqueue_cutlass(const Call& v,void* p,std::size_t n,cudaStream_t s,
    const CutlassExecution& e,Profile* prof) { return launch_cutlass(v,p,n,s,e,prof); }
cudaError_t decode_expert_fp16(const Expert* x,unsigned h,unsigned i,U16* qg,U16* qu,U16* qd,const U32* one,
    const CutlassExecution& e,cudaStream_t s) { return decode_one_expert(x,h,i,qg,qu,qd,one,e,s); }

std::size_t workspace_bytes(unsigned r,unsigned e,unsigned h,unsigned i) {
    if(!shape(r,e,h,i)) return 0;
    return Layout(nullptr,r,e,h,i,topk).size;
}
std::size_t workspace_bytes_dense(unsigned r,unsigned h,unsigned i) {
    if(!shape(r,1,h,i)) return 0;
    return Layout(nullptr,r,1,h,i,1).size;
}
cudaError_t enqueue(const Call& v,void* p,std::size_t n,cudaStream_t s,Profile* prof) { return launch(v,p,n,s,8,prof); }
cudaError_t enqueue_reference(const Call& v,void* p,std::size_t n,cudaStream_t s) {
    Call ref=v;
    if(v.experts==1) { ref.ids=nullptr;ref.weights=nullptr; }
    return launch(ref,p,n,s,v.experts==1?1:8,nullptr,true);
}
cudaError_t enqueue_dense(unsigned r,unsigned h,unsigned i,const float* x,const Expert* m,float* y,void* p,std::size_t n,cudaStream_t s,Profile* prof) {
    return launch({r,1,h,i,x,nullptr,nullptr,m,y},p,n,s,1,prof);
}
cudaError_t decode(Matrix w,U16* q,cudaStream_t s) {
    if(!w.packed || !q || !w.input || !w.output || w.input>16384 || w.output>16384 || w.input%16 || w.output%16 || w.bits<3 || w.bits>6) return cudaErrorInvalidValue;
    decode_kernel<<<w.input/16*(w.output/16),256,0,s>>>(w,q); return cudaGetLastError();
}
}
