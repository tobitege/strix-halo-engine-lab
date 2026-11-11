// 64x64 per-wave DA16 register tile (STRIX_QPF2K_MOE_TILE64=1); default OFF until node timing.
// Same K16 chains, operand conversions, image publication and storage epilogues.
struct MoeTile64 {
 static constexpr unsigned Batch=128U,Columns=256U,Threads=256U,Groups=4U,TPC=1U,AChunks=4U,LTune=82U;
 static constexpr bool Table=true,Wide=false;
};
template<unsigned Batch,unsigned Columns,unsigned Tune>
__device__ __forceinline__ void moe_tile64_step(const F16Stage<Batch,Columns,Tune>& s,unsigned rbase,unsigned mb,const unsigned (&cols)[4],unsigned i,f32x8 (&acc)[4][4]) {
#pragma unroll
 for(unsigned kk=0;kk<4U;++kk) {
  constexpr unsigned P=F16Stage<Batch,Columns,Tune>::pitch;
  const unsigned key=(Tune&8U)?i%8U*8U:0U,lo=(16U*kk)^key,hi=(16U*kk+8U)^key;
  f16x16 b[4];
#pragma unroll
  for(unsigned u=0;u<4U;++u)b[u]=f16_pair(s.b+(cols[u]+i)*P,lo,hi);
#pragma unroll
  for(unsigned mi=0;mi<4U;++mi)if(mi<mb) {
   const f16x16 a=f16_pair(s.a+(rbase+16U*mi+i)*P,lo,hi);
#pragma unroll
   for(unsigned u=0;u<4U;++u)acc[mi][u]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b[u],acc[mi][u]);
  }
 }
}

template<bool Q5K,unsigned Tune,unsigned Core,bool Reuse>
__device__ __forceinline__ void moe_tile64_project_body(const unsigned char* gate,const unsigned char* up,const PrefillRows& view,
 unsigned hidden,unsigned outputs,const uint4* a16,unsigned p1_batch,unsigned p1_tile,
 F16LdsOf<MoeTile64>::Stage* p1_sb,F16LdsOf<MoeTile64>::Rows& p1_rows) {
#define STRIX_F16X_BATCH_TILE const unsigned batch=p1_batch,tile=p1_tile;
#define STRIX_F16X_LDS(S,R) S* const sb=p1_sb; R& rows=p1_rows;
#define STRIX_F16X_HDR_REUSE Reuse
// qpf2k/moe: the body of project_prefill_f16x, included verbatim by the kernel (hip_moe_q4k_a8_known_rows.hip, identical tokens: the
// macros below expand to the original lines) and by the P1 fused launch (hip_moe_prefill_p1.inc). STRIX_F16X_BATCH_TILE
// declares batch / tile, STRIX_F16X_LDS(Stage, Rows) the stage `sb` (indexable) and the rows table `rows`.
    using T=MoeTile64;
    constexpr unsigned Batch=T::Batch,Columns=T::Columns,Outputs=Columns/2U,TPC=T::TPC,AChunks=T::AChunks;
    STRIX_F16X_BATCH_TILE
    const auto& csr=view.csr;
    if((csr.summary[0]&~projection_nonfinite)||batch>=csr.summary[2])return;
    const unsigned begin=csr.begins[batch];
    unsigned pair_end=0U; // core2 bit 6: end of the paired batch (0 = this batch alone)
    if constexpr(T::Wide||(Core&256U)!=0U) {
        // Batches of an expert are consecutive 128-assignment runs from its CSR offset: the even one of each pair runs both.
        const unsigned e=csr.experts[batch];
        const bool odd=((begin-csr.offsets[e])/128U)&1U;
        const bool paired=odd||(batch+1U<csr.summary[2]&&csr.experts[batch+1U]==e);
        if constexpr(T::Wide) {
            // Core bit 8 split: the 256-row kernel runs only the pairs (a lone batch goes to the 128-row launch).
            if(odd||((Core&256U)!=0U&&!paired))return;
            if(paired)pair_end=csr.begins[batch+2U];
        } else if(paired)return; // 128-row launch of the split: lone batches only
    }
    const unsigned count=(pair_end?pair_end:csr.begins[batch+1U])-begin,expert=csr.experts[batch];
    using Stage=F16Stage<Batch,Columns,T::LTune>;
    STRIX_F16X_LDS(Stage,F16Rows<T::Table?Batch:1U>)
    const unsigned t=threadIdx.x;
    if constexpr(T::Table) if(t<Batch) {
        const bool live=t<count;const unsigned r=live?csr.rows[begin+t]:0U;
        rows.row[t]=r;rows.slot[t]=live?csr.slots[begin+t]:0U;rows.exponent[t]=live?at(view.sums,r)[0]:0;
    }
    const auto row_of=[&](unsigned a)->unsigned{if constexpr(T::Table)return rows.row[a];else return a<count?csr.rows[begin+a]:0U;};
    const auto slot_of=[&](unsigned a)->unsigned{if constexpr(T::Table)return rows.slot[a];else return a<count?csr.slots[begin+a]:0U;};
    const auto exp_of=[&](unsigned a)->int{if constexpr(T::Table)return rows.exponent[a];else return a<count?at(view.sums,row_of(a))[0]:0;};
    __syncthreads();
    constexpr unsigned block_bytes=Q5K?176U:144U,qs_at=Q5K?48U:16U,mask=0x0f0f0f0fU;
    // A: chunk c = t + Threads*j is row c/8, halves 8*(c%8)..+8 of the step (rows past the batch stay zero).
    const unsigned blocks=hidden/256U,steps=hidden/64U;
    const unsigned char* a_src[AChunks];float a_scale[AChunks];bool a_live[AChunks];unsigned a_row[AChunks],a_half[AChunks];
#pragma unroll
    for(unsigned j=0;j<AChunks;++j) {
        const unsigned c=t+T::Threads*j;
        a_row[j]=c/8U;a_half[j]=8U*(c%8U);a_live[j]=a_row[j]<count;
        a_scale[j]=ldexpf(1.0F,exp_of(a_row[j]));
        // Probe 8192 (hot A, attribution only): every row reads the batch's first row (same instructions, L0/L2 hits).
        const unsigned ar=(Tune&8192U)?0U:a_row[j];
        a_src[j]=(Tune&16U)?reinterpret_cast<const unsigned char*>(a16+static_cast<std::size_t>(row_of(ar))*(hidden/8U))+2U*a_half[j]:
            reinterpret_cast<const unsigned char*>(at(view.input,row_of(ar)))+2U*a_half[j];
    }
    // B: column t % Columns, part t / Columns: qs dwords 8/TPC*part.. of the step's pair (and the same qh dwords).
    const unsigned b_col=t%Columns,b_part=t/Columns;
    constexpr unsigned BWords=8U/TPC;
    // Probe 16384 (hot B, attribution only): every batch reads expert 0's weights.
    const unsigned b_expert=(Tune&16384U)?0U:expert;
    const unsigned char* b_src=(b_col<Outputs?gate:up)+
        (static_cast<unsigned long long>(b_expert)*outputs+tile*Outputs+(b_col%Outputs))*blocks*block_bytes;
    // BWords 2 (four threads per B column, core bit 6): one 8-byte qs (and qh) load per thread and step.
    constexpr unsigned QN=BWords>=4U?BWords/4U:1U;
    using QV=typename f16_qv<(BWords>=4U)>::type;
    struct Regs { uint4 a[AChunks]{},h{}; QV q[QN]{},qh[QN]{}; };
    struct Cvt { uint4 a[AChunks],b[BWords]; };
    const auto fetch=[&](unsigned step,Regs& r) {
        if constexpr((Tune&1024U)!=0U)return;
        if constexpr((Tune&2048U)==0U)
#pragma unroll
        for(unsigned j=0;j<AChunks;++j) {
            // Branch-free loads (a row past the batch reads the first table row; staging writes zero for it, as the base), so
            // the wait-count pass can wait for exactly the oldest register stage (a skipped exec-masked load forces
            // vmcnt(0), which drains every younger prefetch).
            if constexpr(Core!=0U)r.a[j]=f16_load(a_src[j]+128U*step);
            else if(a_live[j])r.a[j]=f16_load(a_src[j]+128U*step);
        }
        if constexpr((Tune&4096U)!=0U)return;
        const unsigned char* blk=b_src+(step/4U)*block_bytes;
#ifdef STRIX_F16X_HDR_REUSE
        // MoE header reuse (STRIX_QPF2K_MOE_HDR): two register stages alternate even/odd steps (f16_pipeline, Tune bit 0
        // clear, Core 0), so the stage of step s last held step s-2: the same superblock (same header and Q5_K qh words,
        // whose addresses do not depend on s%4) whenever s%4 >= 2. Those loads are skipped; the register holds the bytes.
        constexpr bool hdr_reuse=STRIX_F16X_HDR_REUSE&&Core==0U&&(Tune&1U)==0U;
#else
        constexpr bool hdr_reuse=false;
#endif
        const bool hdr_load=!hdr_reuse||step%4U<2U;
        if(hdr_load)r.h=f16_load(blk);
        if constexpr(BWords>=4U) {
#pragma unroll
        for(unsigned w=0;w<BWords/4U;++w) {
            r.q[w]=f16_load(blk+qs_at+32U*(step%4U)+4U*BWords*b_part+16U*w);
            if constexpr(Q5K)if(hdr_load)r.qh[w]=f16_load(blk+16U+4U*BWords*b_part+16U*w);
        }
        } else {
            r.q[0]=f16_load8(blk+qs_at+32U*(step%4U)+4U*BWords*b_part);
            if constexpr(Q5K)if(hdr_load)r.qh[0]=f16_load8(blk+16U+4U*BWords*b_part);
        }
    };
    bool fast_block=false;
    const auto convert=[&](unsigned step,const Regs& r,Cvt& cv) {
#pragma unroll
        for(unsigned j=0;j<AChunks;++j) {
            cv.a[j]=(Tune&16U)?r.a[j]:f16_from_bf16(r.a[j],a_scale[j]);
            if constexpr(Core!=0U)if(!a_live[j])cv.a[j]=uint4{0U,0U,0U,0U};
        }
        const uint4 rh=r.h;
        const unsigned p=step%4U,g0=2U*p,g1=g0+1U;
        const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(rh.x&65535U)));
        const float dm=__half2float(__ushort_as_half(static_cast<unsigned short>(rh.x>>16U)));
        const float a0=d*static_cast<float>(factor(rh,g0,false)),b0=-(dm*static_cast<float>(factor(rh,g0,true)));
        const float a1=d*static_cast<float>(factor(rh,g1,false)),b1=-(dm*static_cast<float>(factor(rh,g1,true)));
        // a = d*sc, b = -dmin*m with sc, m <= 63 and q <= 15 (Q4_K) or 31 (Q5_K): checked once per superblock (the
        // steps 4s..4s+3 convert in order and share the header).
        if constexpr((Tune&64U)!=0U)if(p==0U)fast_block=__all(QwenKDecodeOps::f16_affine_exact(rh.x&65535U,63U*(Q5K?31U:15U),rh.x>>16U,63U));
        const bool fast=(Tune&64U)!=0U&&fast_block;
        unsigned q[BWords],qh[BWords];
        if constexpr(BWords>=4U) {
#pragma unroll
        for(unsigned w=0;w<BWords/4U;++w) {
            q[4U*w]=r.q[w].x;q[4U*w+1U]=r.q[w].y;q[4U*w+2U]=r.q[w].z;q[4U*w+3U]=r.q[w].w;
            qh[4U*w]=r.qh[w].x;qh[4U*w+1U]=r.qh[w].y;qh[4U*w+2U]=r.qh[w].z;qh[4U*w+3U]=r.qh[w].w;
        }
        } else {q[0]=r.q[0].x;q[1]=r.q[0].y;qh[0]=r.qh[0].x;qh[1]=r.qh[0].y;}
#pragma unroll
        for(unsigned k=0;k<BWords;k+=2U) {
            unsigned l0=q[k]&mask,u0=(q[k]>>4U)&mask,l1=q[k+1U]&mask,u1=(q[k+1U]>>4U)&mask;
            if constexpr(Q5K) {
                l0|=((qh[k]>>g0)&0x01010101U)<<4U;u0|=((qh[k]>>g1)&0x01010101U)<<4U;
                l1|=((qh[k+1U]>>g0)&0x01010101U)<<4U;u1|=((qh[k+1U]>>g1)&0x01010101U)<<4U;
            }
            if(fast) {
                const uint2 x0=QwenKDecodeOps::f16_affine_u8_fast(l0,a0*16777216.0F,b0),x1=QwenKDecodeOps::f16_affine_u8_fast(l1,a0*16777216.0F,b0);
                const uint2 y0=QwenKDecodeOps::f16_affine_u8_fast(u0,a1*16777216.0F,b1),y1=QwenKDecodeOps::f16_affine_u8_fast(u1,a1*16777216.0F,b1);
                cv.b[k]={x0.x,x0.y,x1.x,x1.y};cv.b[k+1U]={y0.x,y0.y,y1.x,y1.y};
            } else {
                const uint2 x0=QwenKDecodeOps::f16_affine_u8(l0,a0,b0),x1=QwenKDecodeOps::f16_affine_u8(l1,a0,b0),y0=QwenKDecodeOps::f16_affine_u8(u0,a1,b1),y1=QwenKDecodeOps::f16_affine_u8(u1,a1,b1);
                cv.b[k]={x0.x,x0.y,x1.x,x1.y};cv.b[k+1U]={y0.x,y0.y,y1.x,y1.y};
            }
        }
    };
    const auto store=[&](const Cvt& cv,Stage& st) {
#pragma unroll
        for(unsigned j=0;j<AChunks;++j)*st.a16(a_row[j],a_half[j])=cv.a[j];
#pragma unroll
        for(unsigned k=0;k<BWords;k+=2U) {
            const unsigned kq=BWords*b_part+k;
            *st.b16(b_col,4U*kq)=cv.b[k];*st.b16(b_col,32U+4U*kq)=cv.b[k+1U];
        }
    };
    const auto stage=[&](unsigned step,const Regs& r,unsigned buf) {
        Stage& st=sb[buf];
        if constexpr((Tune&512U)!=0U) {
#pragma unroll
            for(unsigned j=0;j<AChunks;++j)*st.a16(a_row[j],a_half[j])=r.a[j];
if constexpr(BWords>=4U)
#pragma unroll
            for(unsigned w=0;w<BWords/4U;++w){*st.b16(b_col,BWords*4U*b_part+8U*w)=r.q[w];*st.b16(b_col,32U+BWords*4U*b_part+8U*w)=r.h;}
            return;
        }
        Cvt cv;
        convert(step,r,cv);
        store(cv,st);
    };
    const unsigned lane=t&31U,wave=Core?static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(t/32U))):t/32U,i=lane&15U,h=lane>>4U,rh=wave/T::Groups,cg=wave%T::Groups;
    const unsigned rbase=64U*rh,mb=count>rbase?min(4U,(count-rbase+15U)/16U):0U;
    const unsigned cols[4]{32U*cg,32U*cg+16U,Outputs+32U*cg,Outputs+32U*cg+16U};
    f32x8 acc[4][4]={};
    f16_pipeline<Tune,Regs>(steps,fetch,stage,[&](unsigned){moe_tile64_step(sb[0],rbase,mb,cols,i,acc);});
    const unsigned out=tile*Outputs+32U*cg+i;
#pragma unroll
    for(unsigned mi=0;mi<4U;++mi)if(mi<mb)
#pragma unroll
    for(unsigned e=0;e<8U;++e)
#pragma unroll
    for(unsigned uo=0;uo<2U;++uo) {
        const unsigned a=rbase+16U*mi+2U*e+h;
        const bool live=a<count;
        const unsigned r=row_of(live?a:0U),slot=slot_of(live?a:0U);
        const float unscale=ldexpf(1.0F,-exp_of(live?a:0U));
        const float g=acc[mi][uo][e]*unscale,u=acc[mi][uo+2U][e]*unscale;
        const float hv=(g/(1.0F+expf(-g)))*u;
        const bool bad=!isfinite(g)||!isfinite(u)||!isfinite(hv);
        if(live) {
            if constexpr((Core&4U)!=0U)f16_store_global(at(view.output,r)+slot*2U*outputs+out+16U*uo,hv);
            else at(view.output,r)[slot*2U*outputs+out+16U*uo]=hv;
        }
        float m=fabsf(hv);
        for(int stride=8;stride!=0;stride/=2)m=fmaxf(m,__shfl_xor(m,stride,32));
        if(live&&i==0U)atomicMax(reinterpret_cast<unsigned*>(at(view.down_sums,r))+slot,__float_as_uint(m));
        if(live&&bad){atomicOr(at(view.status,r),projection_nonfinite);atomicOr(csr.summary,projection_nonfinite);}
    }

#undef STRIX_F16X_HDR_REUSE
#undef STRIX_F16X_BATCH_TILE
#undef STRIX_F16X_LDS
}
template<bool Q5K,unsigned Tune,bool Reuse>
__device__ __forceinline__ void moe_tile64_project_entry(const unsigned char* gate,const unsigned char* up,
    const PrefillRows& view,unsigned hidden,unsigned outputs,const uint4* a16,unsigned* arrive,uint4* image) {
    using L=F16LdsOf<MoeTile64>;
    __shared__ __align__(16) typename L::Stage sb[1];
    __shared__ typename L::Rows rows;
    __shared__ unsigned last;
    __shared__ float scale_of[MoeTile64::Batch];
    const unsigned batch=blockIdx.y,t=threadIdx.x;
    moe_tile64_project_body<Q5K,Tune,0U,Reuse>(gate,up,view,hidden,outputs,a16,batch,blockIdx.x,sb,rows);
    const auto& csr=view.csr;
    // The body's own guard (bit 8 masked: it may be set by other CTAs meanwhile, the rest is fixed before the launch).
    if((csr.summary[0]&~projection_nonfinite)||batch>=csr.summary[2])return;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent"); // this CTA's h stores and max |h| atomics before its arrival
    __syncthreads();
    if(t==0U)last=__hip_atomic_fetch_add(arrive+batch,1U,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_AGENT)==gridDim.x-1U?1U:0U;
    __syncthreads();
    if(!last)return; // CTA-uniform (LDS)
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent"); // every tile's h and the final max |h| words
    // rows.row / rows.slot: the body's table of this batch (written before its K loop, never overwritten).
    const unsigned begin=csr.begins[batch],count=csr.begins[batch+1U]-begin,per=outputs/8U,total=count*per;
    // The down's a_scale of each (row, slot): ldexpf(1, f16_exponent(final max |h|)) (its rows.exponent, then ldexpf).
    if(t<count)scale_of[t]=ldexpf(1.0F,f16_exponent(__uint_as_float(static_cast<unsigned>(at(view.down_sums,rows.row[t])[rows.slot[t]]))));
    __syncthreads();
    // Two words per thread and trip, both loads issued before either store.
    for(unsigned i=t;i<total;i+=2U*blockDim.x) {
        const unsigned j=i+blockDim.x;
        const bool two=j<total;
        const unsigned a0=i/per,c0=i%per,a1=two?j/per:a0,c1=two?j%per:c0;
        const float* h0=at(view.output,rows.row[a0])+2ULL*rows.slot[a0]*outputs+8U*c0;
        const float* h1=at(view.output,rows.row[a1])+2ULL*rows.slot[a1]*outputs+8U*c1;
        const uint4 lo0=*reinterpret_cast<const uint4*>(h0),hi0=*reinterpret_cast<const uint4*>(h0+4);
        uint4 lo1{},hi1{};
        if(two){lo1=*reinterpret_cast<const uint4*>(h1);hi1=*reinterpret_cast<const uint4*>(h1+4);}
        image[static_cast<std::size_t>(begin+a0)*per+c0]=f16_from_f32(lo0,hi0,scale_of[a0]);
        if(two)image[static_cast<std::size_t>(begin+a1)*per+c1]=f16_from_f32(lo1,hi1,scale_of[a1]);
    }
}

__global__ __launch_bounds__(256) void project_prefill_da16_tile64(const unsigned char* g,const unsigned char* u,PrefillRows v,unsigned h,unsigned o,const uint4* a,unsigned* c,uint4* image) {
 if(v.csr.summary[2]<=blockIdx.y||v.csr.begins[blockIdx.y+1U]-v.csr.begins[blockIdx.y]!=128U)return;
 moe_tile64_project_entry<false,82U,true>(g,u,v,h,o,a,c,image);
}
template<bool Q51,bool RouteBF16>
__global__ __launch_bounds__(256) void down_prefill_da16_tile64(const unsigned char* weights,PrefillRows view,unsigned hidden,unsigned width,const uint4* image) {
 if(view.csr.summary[0]||blockIdx.y>=view.csr.summary[2]||view.csr.begins[blockIdx.y+1U]-view.csr.begins[blockIdx.y]!=128U)return;
 constexpr unsigned Tune=82U;
#define STRIX_DA16_BATCH_TILE const unsigned batch=blockIdx.y,tile=blockIdx.x;
#define STRIX_DA16_GUARD if(csr.summary[0]||batch>=csr.summary[2])return;
#define STRIX_DA16_LDS(S,R) __shared__ __align__(16) S sb[1]; __shared__ R rows;
#define STRIX_DA16_CERTIFICATE
// qpf2k/moe-mi: the body of down_prefill_f16x_da16 (hip_moe_prefill_da16.inc), included by that kernel (the macros expand to
// its original lines) and by the P1 + DA16 item function. STRIX_DA16_BATCH_TILE declares batch / tile, STRIX_DA16_GUARD the
// early return (empty in P1: decided per CTA before), STRIX_DA16_LDS(Stage, Rows) the stage `sb` (indexable) and `rows`.
#include "hip_moe_tile64_down_body.inc"
#undef STRIX_DA16_CERTIFICATE
#undef STRIX_DA16_BATCH_TILE
#undef STRIX_DA16_GUARD
#undef STRIX_DA16_LDS
}
template<bool Q51>
__global__ __launch_bounds__(256) void down_prefill_da16_tile64_raw(const unsigned char* weights,PrefillRows view,unsigned hidden,unsigned width,const uint4* image) {
 if(view.csr.summary[0]||blockIdx.y>=view.csr.summary[2]||view.csr.begins[blockIdx.y+1U]-view.csr.begins[blockIdx.y]!=128U)return;
 constexpr unsigned Tune=82U;
#define STRIX_DA16_BATCH_TILE const unsigned batch=blockIdx.y,tile=blockIdx.x;
#define STRIX_DA16_GUARD if(csr.summary[0]||batch>=csr.summary[2])return;
#define STRIX_DA16_LDS(S,R) __shared__ __align__(16) S sb[1]; __shared__ R rows;
// qpf2k/moe-mi: the body of down_prefill_f16x_da16 (hip_moe_prefill_da16.inc), included by that kernel (the macros expand to
// its original lines) and by the P1 + DA16 item function. STRIX_DA16_BATCH_TILE declares batch / tile, STRIX_DA16_GUARD the
// early return (empty in P1: decided per CTA before), STRIX_DA16_LDS(Stage, Rows) the stage `sb` (indexable) and `rows`.
#include "hip_moe_tile64_down_body.inc"
#undef STRIX_DA16_BATCH_TILE
#undef STRIX_DA16_GUARD
#undef STRIX_DA16_LDS
}

// Partial CSR batches retain the incumbent tile, including its image publication.
// Separate filtered launches keep every existing specialization's machine code intact.
template<bool Reuse>
__global__ __launch_bounds__(512) void project_prefill_da16_tile64_remainder(
    const unsigned char* g,const unsigned char* u,PrefillRows v,unsigned h,unsigned o,
    const uint4* a,unsigned* c,uint4* image) {
    if(blockIdx.y>=v.csr.summary[2]||v.csr.begins[blockIdx.y+1U]-v.csr.begins[blockIdx.y]==128U)return;
    project_prefill_f16x_da16_entry<false,82U,Reuse,false,1U>(g,u,v,h,o,a,c,image);
}
template<bool Q51,bool RouteBF16,bool Certificate>
__global__ __launch_bounds__(512) void down_prefill_da16_tile64_remainder(
    const unsigned char* weights,PrefillRows view,unsigned hidden,unsigned width,const uint4* image) {
    constexpr unsigned Tune=82U;
#define STRIX_DA16_BATCH_TILE const unsigned batch=blockIdx.y,tile=blockIdx.x;
#define STRIX_DA16_GUARD if(csr.summary[0]||batch>=csr.summary[2]||csr.begins[batch+1U]-csr.begins[batch]==128U)return;
#define STRIX_DA16_LDS(S,R) __shared__ __align__(16) S sb[1]; __shared__ R rows;
    constexpr bool RouteI8=false;   // one-byte route dots have their own entry (hip_moe_prefill_da16.inc)
    [[maybe_unused]] constexpr unsigned route_top_k=0U;
    if constexpr(Certificate) {
#define STRIX_DA16_CERTIFICATE
#include "moe_core/hip_moe_da16_down_body.inc"
#undef STRIX_DA16_CERTIFICATE
    } else {
#include "moe_core/hip_moe_da16_down_body.inc"
    }
#undef STRIX_DA16_BATCH_TILE
#undef STRIX_DA16_GUARD
#undef STRIX_DA16_LDS
}
inline int& prefill_tile64_override(){static int v=-1;return v;}
inline bool& prefill_tile64_flag(){static thread_local bool v=false;return v;}
inline hipEvent_t*& prefill_test_events(){static thread_local hipEvent_t* v=nullptr;return v;}
// Bit 0: TILE64 gate/up, bit 1: TILE64 down. STRIX_QPF2K_MOE_TILE64=1 selects both; unset or any other value, neither.
// The test override (1 both, 2 gate/up only, 3 down only) lets the byte gate check each kernel against today's other one.
inline unsigned prefill_tile64() {
    static const bool env=[]{const char* v=std::getenv("STRIX_QPF2K_MOE_TILE64");return v&&v[0]=='1'&&v[1]=='\0';}();
    if(qpf2k::baseline||(prefill_moe_gsq_off()&1U))return 0U;
    const int o=prefill_tile64_override();
    if(o<0)return env?3U:0U;
    return o==1?3U:o==2?1U:o==3?2U:0U;
}

// The full and remainder grids are disjoint in CSR batch space. No new stream or merge order.
inline void launch_tile64_project(const unsigned char* g,const unsigned char* u,PrefillRows v,
    unsigned h,unsigned o,const uint4* a,unsigned* c,uint4* image,unsigned batches,hipStream_t stream,bool reuse) {
    const dim3 grid(o/128U,batches);
    hipLaunchKernelGGL(project_prefill_da16_tile64,grid,dim3(256),0,stream,g,u,v,h,o,a,c,image);
    if(reuse)hipLaunchKernelGGL(project_prefill_da16_tile64_remainder<true>,grid,dim3(512),0,stream,g,u,v,h,o,a,c,image);
    else hipLaunchKernelGGL(project_prefill_da16_tile64_remainder<false>,grid,dim3(512),0,stream,g,u,v,h,o,a,c,image);
}
template<bool Q51,bool BF16,bool Certificate>
inline void launch_tile64_down(const unsigned char* w,PrefillRows v,unsigned h,unsigned o,
    const uint4* image,unsigned batches,hipStream_t stream) {
    const dim3 grid(h/256U,batches);
    if constexpr(Certificate)
        hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_da16_tile64<Q51,BF16>),grid,dim3(256),0,stream,w,v,h,o,image);
    else hipLaunchKernelGGL(down_prefill_da16_tile64_raw<Q51>,grid,dim3(256),0,stream,w,v,h,o,image);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(down_prefill_da16_tile64_remainder<Q51,BF16,Certificate>),grid,dim3(512),0,stream,w,v,h,o,image);
}
inline void launch_tile64_down_mode(bool q51,bool certificate,const unsigned char* w,PrefillRows v,
    unsigned h,unsigned o,const uint4* image,unsigned batches,hipStream_t stream) {
    const auto run=[&]<bool Q51>() {
        if(!certificate)launch_tile64_down<Q51,false,false>(w,v,h,o,image,batches,stream);
        else if(v.route_bf16)launch_tile64_down<Q51,true,true>(w,v,h,o,image,batches,stream);
        else launch_tile64_down<Q51,false,true>(w,v,h,o,image,batches,stream);
    };
    if(q51)run.template operator()<true>();else run.template operator()<false>();
}
