// The barrier and the global-address loads live in moe_core/grouped_f16.hpp (GroupedF16Issue); these are the Qwen names.
__device__ __forceinline__ void f16_barrier(){strix::moe_core::GroupedF16Issue::barrier();}
__device__ __forceinline__ uint4 f16_load(const void* p){return strix::moe_core::GroupedF16Issue::load16<uint4>(p);}
// One K64 step of a CTA: s.a holds the batch rows (64 x K64), s.b 256 B columns (x K64). A wave
// owns two B columns tiles (cols[0], cols[1]) and every 16-row tile below mb: the B fragments are
// read once per K16 plane and reused by up to four row tiles. K16 planes ascend.
// Tune bit 3: 64-half rows whose 16-byte chunks are XOR-permuted by row % 8, so the 8 rows a 128-bit LDS
// load serves per pass hit distinct banks without the 8-half padding (staging 40960 instead of 46080 bytes: three
// 8-wave CTAs per WGP instead of two). Layout only; every operand and accumulation is unchanged.
template<unsigned Tune> __device__ __forceinline__ unsigned f16_lds(unsigned row,unsigned half) {
    using Ring=strix::moe_core::GroupedLdsRing<strix::moe_core::GroupedPrefillPolicy<(Tune&8U)!=0U>,f16_pitch>;
    return Ring::offset(row,half);
}
template<unsigned Batch,unsigned Columns,unsigned Tune> struct F16Stage {
    static constexpr unsigned pitch=strix::moe_core::GroupedLdsRing<strix::moe_core::GroupedPrefillPolicy<(Tune&8U)!=0U>,f16_pitch>::pitch;
    _Float16 a[Batch*pitch]; _Float16 b[Columns*pitch];
    __device__ uint4* a16(unsigned row,unsigned half){return reinterpret_cast<uint4*>(a+f16_lds<Tune>(row,half));}
    __device__ uint4* b16(unsigned row,unsigned half){return reinterpret_cast<uint4*>(b+f16_lds<Tune>(row,half));}
};
// Fragment row base + 16-aligned tile offset: the halves at lo and hi of that row.
__device__ __forceinline__ f16x16 f16_pair(const _Float16* row,unsigned lo_half,unsigned hi_half) {
    const uint4 lo=*reinterpret_cast<const uint4*>(row+lo_half),hi=*reinterpret_cast<const uint4*>(row+hi_half);
    f16x16 v;
    __builtin_memcpy(&v,&lo,16U);__builtin_memcpy(reinterpret_cast<char*>(&v)+16,&hi,16U);
    return v;
}
template<unsigned Batch,unsigned Columns,unsigned Tune>
__device__ __forceinline__ void f16_step(const F16Stage<Batch,Columns,Tune>& s,unsigned mb,const unsigned (&cols)[2],unsigned i,f32x8 (&acc)[4][2]) {
#pragma unroll
    for(unsigned kk=0;kk<4U;++kk) {
        // Rows are 16-aligned bases + i, so the XOR key of f16_lds is i % 8 for every fragment of this lane.
        constexpr unsigned P=F16Stage<Batch,Columns,Tune>::pitch;
        const unsigned key=(Tune&8U)?i%8U*8U:0U,lo=(16U*kk)^key,hi=(16U*kk+8U)^key;
        const f16x16 b0=f16_pair(s.b+(cols[0]+i)*P,lo,hi);
        const f16x16 b1=f16_pair(s.b+(cols[1]+i)*P,lo,hi);
#pragma unroll
        for(unsigned mi=0;mi<4U;++mi)if(mi<mb) {
            const f16x16 a=f16_pair(s.a+(16U*mi+i)*P,lo,hi);
            acc[mi][0]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b0,acc[mi][0]);
            acc[mi][1]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b1,acc[mi][1]);
        }
    }
}
// Batch rows of a CTA into LDS (row, slot, scale exponent).
template<unsigned Batch> struct F16Rows { unsigned row[Batch],slot[Batch]; int exponent[Batch]; };
// Gate/up of one CSR batch (<= 64 assignments) for 128 outputs of both matrices. Grid x = output
// tile (the tiles of a batch run together and share its rows in L2), y = batch. B columns 0..127
// are gate outputs, 128..255 the same up outputs; wave w owns gate and up columns 16w.. of every
// row, so one lane holds g and u of the same (row, output) for the fused SwiGLU.
template<bool Q5K,unsigned Tune=0U>
__global__ __launch_bounds__(256) void project_prefill_f16(const unsigned char* gate,const unsigned char* up,PrefillRows view,
    unsigned hidden,unsigned outputs) {
    constexpr unsigned Batch=64U,Columns=256U,Outputs=Columns/2U;
    const unsigned batch=blockIdx.y,tile=blockIdx.x;
    const auto& csr=view.csr;
    if((csr.summary[0]&~projection_nonfinite)||batch>=csr.summary[2])return;
    const unsigned begin=csr.begins[batch],count=csr.begins[batch+1U]-begin,expert=csr.experts[batch];
    __shared__ __align__(16) F16Stage<Batch,Columns,Tune> s;
    __shared__ F16Rows<Batch> rows;
    const unsigned t=threadIdx.x;
    if(t<Batch) {
        const bool live=t<count;const unsigned r=live?csr.rows[begin+t]:0U;
        rows.row[t]=r;rows.slot[t]=live?csr.slots[begin+t]:0U;rows.exponent[t]=live?at(view.sums,r)[0]:0;
    }
    __syncthreads();
    constexpr unsigned block_bytes=Q5K?176U:144U,qs_at=Q5K?48U:16U,mask=0x0f0f0f0fU;
    // Staging roles: A row t/4, K values 16(t%4)..; B column t: the 32 qs bytes of the step's pair.
    const unsigned a_row=t/4U,a_part=t%4U;
    const bool a_live=a_row<count;
    const float a_scale=ldexpf(1.0F,rows.exponent[a_row]);
    const auto* a_src=reinterpret_cast<const unsigned char*>(at(view.input,rows.row[a_row]))+32U*a_part;
    const unsigned blocks=hidden/256U,steps=hidden/64U;
    const unsigned char* b_src=(t<Outputs?gate:up)+
        (static_cast<unsigned long long>(expert)*outputs+tile*Outputs+(t%Outputs))*blocks*block_bytes;
    struct Regs { uint4 a0{},a1{},h{},q0{},q1{},qh0{},qh1{}; };
    const auto fetch=[&](unsigned step,Regs& r) {
        if(a_live){r.a0=f16_load(a_src+128U*step);r.a1=f16_load(a_src+128U*step+16U);}
        const unsigned char* blk=b_src+(step/4U)*block_bytes;
        r.h=f16_load(blk);
        r.q0=f16_load(blk+qs_at+32U*(step%4U));r.q1=f16_load(blk+qs_at+32U*(step%4U)+16U);
        if constexpr(Q5K){r.qh0=f16_load(blk+16U);r.qh1=f16_load(blk+32U);}
    };
    const auto stage=[&](unsigned step,const Regs& r) {
        const uint4 rh=r.h,rq0=r.q0,rq1=r.q1,rqh0=r.qh0,rqh1=r.qh1;
        *s.a16(a_row,16U*a_part)=f16_from_bf16(r.a0,a_scale);*s.a16(a_row,16U*a_part+8U)=f16_from_bf16(r.a1,a_scale);
        const unsigned p=step%4U,g0=2U*p,g1=g0+1U;
        const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(rh.x&65535U)));
        const float dm=__half2float(__ushort_as_half(static_cast<unsigned short>(rh.x>>16U)));
        const float a0=d*static_cast<float>(factor(rh,g0,false)),b0=-(dm*static_cast<float>(factor(rh,g0,true)));
        const float a1=d*static_cast<float>(factor(rh,g1,false)),b1=-(dm*static_cast<float>(factor(rh,g1,true)));
        const unsigned q[8]{rq0.x,rq0.y,rq0.z,rq0.w,rq1.x,rq1.y,rq1.z,rq1.w};
        const unsigned qh[8]{rqh0.x,rqh0.y,rqh0.z,rqh0.w,rqh1.x,rqh1.y,rqh1.z,rqh1.w};
#pragma unroll
        for(unsigned k=0;k<8U;k+=2U) {
            unsigned l0=q[k]&mask,u0=(q[k]>>4U)&mask,l1=q[k+1U]&mask,u1=(q[k+1U]>>4U)&mask;
            if constexpr(Q5K) {
                l0|=((qh[k]>>g0)&0x01010101U)<<4U;u0|=((qh[k]>>g1)&0x01010101U)<<4U;
                l1|=((qh[k+1U]>>g0)&0x01010101U)<<4U;u1|=((qh[k+1U]>>g1)&0x01010101U)<<4U;
            }
            const uint2 x0=QwenKScalarDecode::f16_affine_u8(l0,a0,b0),x1=QwenKScalarDecode::f16_affine_u8(l1,a0,b0),y0=QwenKScalarDecode::f16_affine_u8(u0,a1,b1),y1=QwenKScalarDecode::f16_affine_u8(u1,a1,b1);
            *s.b16(t,4U*k)={x0.x,x0.y,x1.x,x1.y};*s.b16(t,32U+4U*k)={y0.x,y0.y,y1.x,y1.y};
        }
    };
    const unsigned lane=t&31U,wave=t/32U,i=lane&15U,h=lane>>4U,mb=(count+15U)/16U;
    const unsigned cols[2]{16U*wave,Outputs+16U*wave};
    f32x8 acc[4][2]={};
    if constexpr(Tune&1U) {
        Regs r;
        fetch(0U,r);
        for(unsigned step=0;step<steps;++step) {
            if(step)f16_barrier();
            stage(step,r);
            f16_barrier();
            if(step+1U<steps)fetch(step+1U,r);
            f16_step(s,mb,cols,i,acc);
        }
    } else {
        Regs r0,r1;
        fetch(0U,r0);fetch(1U,r1); // steps = hidden/64 is even (hidden % 256 == 0)
#pragma unroll 1
        for(unsigned step=0;step<steps;step+=2U) {
            if(step)f16_barrier();
            stage(step,r0);
            f16_barrier();
            if(step+2U<steps)fetch(step+2U,r0);
            f16_step(s,mb,cols,i,acc);
            f16_barrier();
            stage(step+1U,r1);
            f16_barrier();
            if(step+3U<steps)fetch(step+3U,r1);
            f16_step(s,mb,cols,i,acc);
        }
    }
    const unsigned out=tile*Outputs+16U*wave+i;
#pragma unroll
    for(unsigned mi=0;mi<4U;++mi)if(mi<mb)
#pragma unroll
    for(unsigned e=0;e<8U;++e) {
        const unsigned a=16U*mi+2U*e+h;
        const bool live=a<count;
        const unsigned r=rows.row[live?a:0U],slot=rows.slot[live?a:0U];
        const float unscale=ldexpf(1.0F,-rows.exponent[live?a:0U]);
        const float g=acc[mi][0][e]*unscale,u=acc[mi][1][e]*unscale;
        const float hv=(g/(1.0F+expf(-g)))*u;
        const bool bad=!isfinite(g)||!isfinite(u)||!isfinite(hv);
        if(live)at(view.output,r)[slot*2U*outputs+out]=hv;
        // Largest |h| of the wave's 16 outputs of this row (the lanes of one half hold one row).
        float m=fabsf(hv);
        for(int stride=8;stride!=0;stride/=2)m=fmaxf(m,strix::qwen_dpp::xor32(m, stride));
        if(live&&i==0U)atomicMax(reinterpret_cast<unsigned*>(at(view.down_sums,r))+slot,__float_as_uint(m));
        if(live&&bad){atomicOr(at(view.status,r),projection_nonfinite);atomicOr(csr.summary,projection_nonfinite);}
    }
}
// The 32 FP16 weights of a Q8_0 down block (down_operands' byte selection, written without arrays).
__device__ __forceinline__ void f16_q80_block(const DownBlock<false>& x,unsigned group,uint4 (&out)[4]) {
    const unsigned odd=group&1U,select=odd?0x07060504U:0x05040302U;
    const float d=__half2float(__ushort_as_half(static_cast<unsigned short>((x.lo.x>>(odd?16U:0U))&65535U)));
    const auto pair=[&](unsigned hi0,unsigned lo0,unsigned hi1,unsigned lo1) {
        const uint2 a=f16_scaled_s8(__builtin_amdgcn_perm(hi0,lo0,select),d),b=f16_scaled_s8(__builtin_amdgcn_perm(hi1,lo1,select),d);
        return uint4{a.x,a.y,b.x,b.y};
    };
    out[0]=pair(x.lo.y,x.lo.x,x.lo.z,x.lo.y);out[1]=pair(x.lo.w,x.lo.z,x.hi.x,x.lo.w);
    out[2]=pair(x.hi.y,x.hi.x,x.hi.z,x.hi.y);out[3]=pair(x.hi.w,x.hi.z,x.last,x.hi.w);
}
// Down of one CSR batch (<= 64 assignments) for 256 hidden outputs, K = width (FP32 h of each
// (row, slot), scaled and rounded to FP16 while staging). Grid x = output tile, y = batch. B
// column t decodes K blocks 2*step, 2*step + 1 of its output row; wave w owns columns 32w..+32.
template<bool Q51,unsigned Tune=0U>
__global__ __launch_bounds__(256) void down_prefill_f16(const unsigned char* weights,PrefillRows view,unsigned hidden,unsigned width) {
    constexpr unsigned Batch=64U,Columns=256U;
    const unsigned batch=blockIdx.y,tile=blockIdx.x;
    const auto& csr=view.csr;
    if(csr.summary[0]||batch>=csr.summary[2])return;
    const unsigned begin=csr.begins[batch],count=csr.begins[batch+1U]-begin,expert=csr.experts[batch];
    __shared__ __align__(16) F16Stage<Batch,Columns,Tune> s;
    __shared__ F16Rows<Batch> rows;
    const unsigned t=threadIdx.x;
    if(t<Batch) {
        const bool live=t<count;const unsigned r=live?csr.rows[begin+t]:0U,slot=live?csr.slots[begin+t]:0U;
        rows.row[t]=r;rows.slot[t]=slot;
        rows.exponent[t]=live?f16_exponent(__uint_as_float(static_cast<unsigned>(at(view.down_sums,r)[slot]))):0;
    }
    __syncthreads();
    constexpr unsigned block_bytes=Q51?24U:34U;
    const unsigned a_row=t/4U,a_part=t%4U,groups=width/32U,steps=width/64U;
    const bool a_live=a_row<count;
    const float a_scale=ldexpf(1.0F,rows.exponent[a_row]);
    const auto* a_src=at(view.output,rows.row[a_row])+2ULL*rows.slot[a_row]*width+16U*a_part;
    const auto* b_src=weights+(static_cast<unsigned long long>(expert)*hidden+tile*Columns+t)*groups*block_bytes;
    struct Regs { uint4 a0{},a1{},a2{},a3{}; DownBlock<Q51> b0,b1; };
    const auto fetch=[&](unsigned step,Regs& r) {
        if(a_live){r.a0=f16_load(a_src+64U*step);r.a1=f16_load(a_src+64U*step+4U);r.a2=f16_load(a_src+64U*step+8U);r.a3=f16_load(a_src+64U*step+12U);}
        r.b0=down_block<Q51>(b_src,2U*step);r.b1=down_block<Q51>(b_src,2U*step+1U);
    };
    const auto stage=[&](unsigned step,const Regs& r) {
        const uint4 ra[4]{r.a0,r.a1,r.a2,r.a3};const DownBlock<Q51>& rb0=r.b0;const DownBlock<Q51>& rb1=r.b1;
        *s.a16(a_row,16U*a_part)=f16_from_f32(ra[0],ra[1],a_scale);*s.a16(a_row,16U*a_part+8U)=f16_from_f32(ra[2],ra[3],a_scale);
#pragma unroll
        for(unsigned half=0;half<2U;++half) {
            if constexpr(Q51) {
                wmma_v4i b0,b1;float d,m;
                down_operands<true>(half?rb1:rb0,2U*step+half,b0,b1,d,m);
                const unsigned c[8]{static_cast<unsigned>(b0.x),static_cast<unsigned>(b0.y),static_cast<unsigned>(b0.z),static_cast<unsigned>(b0.w),
                    static_cast<unsigned>(b1.x),static_cast<unsigned>(b1.y),static_cast<unsigned>(b1.z),static_cast<unsigned>(b1.w)};
#pragma unroll
                for(unsigned k=0;k<8U;k+=2U) {
                    const uint2 x=QwenKScalarDecode::f16_affine_u8(c[k],d,m),y=QwenKScalarDecode::f16_affine_u8(c[k+1U],d,m);
                    *s.b16(t,8U*(4U*half+k/2U))={x.x,x.y,y.x,y.y};
                }
            } else {
                uint4 w[4];
                f16_q80_block(half?rb1:rb0,2U*step+half,w);
#pragma unroll
                for(unsigned k=0;k<4U;++k)*s.b16(t,8U*(4U*half+k))=w[k];
            }
        }
    };
    const unsigned lane=t&31U,wave=t/32U,i=lane&15U,h=lane>>4U,mb=(count+15U)/16U;
    const unsigned cols[2]{32U*wave,32U*wave+16U};
    f32x8 acc[4][2]={};
    if constexpr(Tune&1U) {
        Regs r;
        fetch(0U,r);
        for(unsigned step=0;step<steps;++step) {
            if(step)f16_barrier();
            stage(step,r);
            f16_barrier();
            if(step+1U<steps)fetch(step+1U,r);
            f16_step(s,mb,cols,i,acc);
        }
    } else {
        Regs r0,r1;
        fetch(0U,r0);fetch(1U,r1); // steps = width/64 is even (width % 128 == 0)
#pragma unroll 1
        for(unsigned step=0;step<steps;step+=2U) {
            if(step)f16_barrier();
            stage(step,r0);
            f16_barrier();
            if(step+2U<steps)fetch(step+2U,r0);
            f16_step(s,mb,cols,i,acc);
            f16_barrier();
            stage(step+1U,r1);
            f16_barrier();
            if(step+3U<steps)fetch(step+3U,r1);
            f16_step(s,mb,cols,i,acc);
        }
    }
#pragma unroll
    for(unsigned mi=0;mi<4U;++mi)if(mi<mb)
#pragma unroll
    for(unsigned e=0;e<8U;++e) {
        const unsigned a=16U*mi+2U*e+h;
        if(a>=count)continue;
        const float unscale=ldexpf(1.0F,-rows.exponent[a]);
        float* out=at(view.rawdots,rows.row[a])+rows.slot[a]*hidden+tile*Columns+32U*wave+i;
        out[0]=acc[mi][0][e]*unscale;out[16]=acc[mi][1][e]*unscale;
    }
}
// Bit 6 helpers. An FP16 value (bits h) is M * 2^E with M < 2^11 an integer; lsb = E + ctz(M).
#include "qwen_k_weight_decode.inc"

// Tune bits 1, 2 and 4 (values 2, 4, 16) select the redone tiles of the FP16 forms. A CTA owns Batch
// assignments of one expert (a CSR batch) x Columns B columns; each wave owns 64 rows x 32 columns
// (four 16-row tiles x two column tiles, as the base tile), so per wave the operands, WMMAs and K order
// are the base ones.
//   bit 1 (batch-major): 128-assignment CSR batches, 16 waves (two row halves). Each dequantized weight
//     tile serves every assignment of an expert up to 128 (the base dequantizes it again per 64), and the
//     grid still runs the output tiles of one batch together (its rows stay in L2).
//   bit 2 (compact): 128 columns per CTA (gate/up 64 outputs, down 128) with all the batch rows, so the
//     weights are not dequantized more often per row; LDS halves (4 CTAs per WGP).
//   bit 4 (gate/up): stage A from the prologue's FP16 image (a copy) instead of converting BF16 per tile.
// Threads per B column (TPC) split a column's K step (gate/up: 16 qs bytes of the pair; down: one block).
//   bit 6 (value 64): exact short dequantization. On gfx1151 VALU and WMMA instructions do not overlap (a VALU
//     instruction costs WMMA issue time), and the dequantization is the kernel's VALU: byte extract, cvt_f32_ubyte,
//     fma, cvt to FP16 (~4 per weight). Here each code q enters v_fma_mix{lo,hi}_f16 as the FP16 subnormal q * 2^-24
//     (one v_perm per two codes) with a * 2^24, giving round_f16(a*q + b) in one instruction per weight. When every
//     a*q + b of the block is an exact FP32 value (checked from the block's FP16 scales, below), rounding it once to
//     FP16 equals the base's FP32-then-FP16 rounding, so the bytes are identical; a wave with any block outside the
//     check takes the base conversion. Q8_0 keeps the base conversion: its offset form d*(q + 128) - 128*d is exact
//     but turns the base's d * 0 = -0 (negative d) into +0 (self-test: 1 in 128 words).
//     (Measured and dropped: staging the next step while multiplying, with two LDS images or with a register-only
//     convert-ahead, is 1.0-1.7x slower: VALU and WMMA serialize anyway and the WMMA operand reads wait on lgkmcnt.)
// Diagnostic bits (STRIX_INTPREFILL_MOE_TUNE >= 256, never exact, attribution of the kernel time only): 256 skips the
// WMMAs, 512 stages the raw loaded words without dequantization/conversion, 1024 skips the global loads, 2048 only the
// A (activation) loads, 4096 only the B (weight) loads.
template<unsigned Tune> struct F16Tile {
    static constexpr unsigned Batch=(Tune&2U)?128U:64U,Columns=(Tune&4U)?128U:256U;
    static constexpr bool Table=true;
    static constexpr unsigned LTune=Tune;
    static constexpr unsigned Waves=Batch*Columns/2048U,Threads=32U*Waves,Groups=Columns/32U;
    static constexpr unsigned TPC=Threads/Columns,AChunks=Batch*8U/Threads;
};
template<unsigned Batch,unsigned Columns,unsigned Tune>
__device__ __forceinline__ void f16_step_rows(const F16Stage<Batch,Columns,Tune>& s,unsigned rbase,unsigned mb,const unsigned (&cols)[2],unsigned i,f32x8 (&acc)[4][2]) {
#pragma unroll
    for(unsigned kk=0;kk<4U;++kk) {
        constexpr unsigned P=F16Stage<Batch,Columns,Tune>::pitch;
        const unsigned key=(Tune&8U)?i%8U*8U:0U,lo=(16U*kk)^key,hi=(16U*kk+8U)^key;
        const f16x16 b0=f16_pair(s.b+(cols[0]+i)*P,lo,hi);
        const f16x16 b1=f16_pair(s.b+(cols[1]+i)*P,lo,hi);
#pragma unroll
        for(unsigned mi=0;mi<4U;++mi)if(mi<mb) {
            const f16x16 a=f16_pair(s.a+(rbase+16U*mi+i)*P,lo,hi);
            acc[mi][0]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b0,acc[mi][0]);
            acc[mi][1]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b1,acc[mi][1]);
        }
    }
}
