// qpf2k/moe-mi: the body of down_prefill_f16x_da16 (hip_moe_prefill_da16.inc), included by that kernel (the macros expand to
// its original lines) and by the P1 + DA16 item function. STRIX_DA16_BATCH_TILE declares batch / tile, STRIX_DA16_GUARD the
// early return (empty in P1: decided per CTA before), STRIX_DA16_LDS(Stage, Rows) the stage `sb` (indexable) and `rows`.
// STRIX_DA16_UNIFORM_A (a constant expression) selects the activation issue policy of moe_core/grouped_f16.hpp.
    using T=F16Tile<Tune>;
    constexpr unsigned Batch=T::Batch,Columns=T::Columns,TPC=T::TPC,AParts=Batch*4U/T::Threads;
    STRIX_DA16_BATCH_TILE
    const auto& csr=view.csr;
    STRIX_DA16_GUARD
    const unsigned begin=csr.begins[batch],count=csr.begins[batch+1U]-begin,expert=csr.experts[batch];
    using Stage=F16Stage<Batch,Columns,T::LTune>;
    STRIX_DA16_LDS(Stage,F16Rows<T::Table?Batch:1U>)
    const unsigned t=threadIdx.x;
    if constexpr(T::Table) if(t<Batch) {
        const bool live=t<count;const unsigned r=live?csr.rows[begin+t]:0U,slot=live?csr.slots[begin+t]:0U;
        rows.row[t]=r;rows.slot[t]=slot;
        rows.exponent[t]=live?f16_exponent(__uint_as_float(static_cast<unsigned>(at(view.down_sums,r)[slot]))):0;
    }
    const auto row_of=[&](unsigned a)->unsigned{if constexpr(T::Table)return rows.row[a];else return a<count?csr.rows[begin+a]:0U;};
    const auto slot_of=[&](unsigned a)->unsigned{if constexpr(T::Table)return rows.slot[a];else return a<count?csr.slots[begin+a]:0U;};
    const auto exp_of=[&](unsigned a)->int{if constexpr(T::Table)return rows.exponent[a];else return a<count?f16_exponent(__uint_as_float(static_cast<unsigned>(at(view.down_sums,row_of(a))[slot_of(a)]))):0;};
    __syncthreads();
    constexpr unsigned block_bytes=Q51?24U:34U;
    const unsigned groups=width/32U,steps=width/64U,per=width/8U;
    // A: part c = t + Threads*j is row c/4, K values 16*(c%4).. of the step = two image words (8 halves each).
    const uint4* a_src[AParts];bool a_live[AParts];unsigned a_row[AParts],a_part[AParts];
#pragma unroll
    for(unsigned j=0;j<AParts;++j) {
        const unsigned c=t+T::Threads*j;
        a_row[j]=c/4U;a_part[j]=c%4U;a_live[j]=a_row[j]<count;
        a_src[j]=image+static_cast<std::size_t>(begin+(a_live[j]?a_row[j]:0U))*per+2U*a_part[j];
    }
    const unsigned b_col=t%Columns,b_part=t/Columns;
    constexpr unsigned BBlocks=2U/TPC;
    const auto* b_src=weights+(static_cast<unsigned long long>(expert)*hidden+tile*Columns+b_col)*groups*block_bytes;
    struct Regs { uint4 a[AParts][2]{}; DownBlock<Q51> b[BBlocks]; };
    struct Cvt { uint4 a[AParts][2],b[BBlocks][4]; };
    using AIssue=strix::moe_core::GroupedF16ActivationIssue<STRIX_DA16_UNIFORM_A>;
    const auto fetch=[&](unsigned step,Regs& r) {
#pragma unroll
        for(unsigned j=0;j<AParts;++j)AIssue::issue(a_live[j],a_src[j]+8U*step,r.a[j]);
#pragma unroll
        for(unsigned k=0;k<BBlocks;++k)r.b[k]=down_block<Q51>(b_src,2U*step+BBlocks*b_part+k);
    };
    const auto convert=[&](unsigned step,const Regs& r,Cvt& cv) {
#pragma unroll
        for(unsigned j=0;j<AParts;++j){cv.a[j][0]=r.a[j][0];cv.a[j][1]=r.a[j][1];}
#pragma unroll
        for(unsigned k=0;k<BBlocks;++k) {
            const unsigned half=BBlocks*b_part+k;
            if constexpr(Q51) {
                wmma_v4i b0,b1;float d,m;
                down_operands<true>(r.b[k],2U*step+half,b0,b1,d,m);
                const unsigned c[8]{static_cast<unsigned>(b0.x),static_cast<unsigned>(b0.y),static_cast<unsigned>(b0.z),static_cast<unsigned>(b0.w),
                    static_cast<unsigned>(b1.x),static_cast<unsigned>(b1.y),static_cast<unsigned>(b1.z),static_cast<unsigned>(b1.w)};
                // Q5_1: d*q + m, q <= 31.
                if((Tune&64U)!=0U&&__all(QwenKDecode::f16_q51_exact(r.b[k].head.x))) {
#pragma unroll
                    for(unsigned q=0;q<8U;q+=2U) {
                        const uint2 x=QwenKDecode::f16_affine_u8_fast(c[q],d*16777216.0F,m),y=QwenKDecode::f16_affine_u8_fast(c[q+1U],d*16777216.0F,m);
                        cv.b[k][q/2U]={x.x,x.y,y.x,y.y};
                    }
                } else {
#pragma unroll
                    for(unsigned q=0;q<8U;q+=2U) {
                        const uint2 x=QwenKScalarDecode::f16_affine_u8(c[q],d,m),y=QwenKScalarDecode::f16_affine_u8(c[q+1U],d,m);
                        cv.b[k][q/2U]={x.x,x.y,y.x,y.y};
                    }
                }
            } else f16_q80_block(r.b[k],2U*step+half,cv.b[k]);
        }
    };
    const auto store=[&](const Cvt& cv,Stage& st) {
#pragma unroll
        for(unsigned j=0;j<AParts;++j){*st.a16(a_row[j],16U*a_part[j])=cv.a[j][0];*st.a16(a_row[j],16U*a_part[j]+8U)=cv.a[j][1];}
#pragma unroll
        for(unsigned k=0;k<BBlocks;++k)
#pragma unroll
            for(unsigned q=0;q<4U;++q)*st.b16(b_col,8U*(4U*(BBlocks*b_part+k)+q))=cv.b[k][q];
    };
    const auto stage=[&](unsigned step,const Regs& r,unsigned buf) {
        Stage& st=sb[buf];
        Cvt cv;
        convert(step,r,cv);
        store(cv,st);
    };
    const unsigned lane=t&31U,wave=t/32U,i=lane&15U,h=lane>>4U,rh=wave/T::Groups,cg=wave%T::Groups;
    const unsigned rbase=64U*rh,mb=count>rbase?min(4U,(count-rbase+15U)/16U):0U;
    const unsigned cols[2]{32U*cg,32U*cg+16U};
    f32x8 acc[4][2]={};
    f16_pipeline<Tune,Regs>(steps,fetch,stage,[&](unsigned){f16_step_rows(sb[0],rbase,mb,cols,i,acc);});
#ifdef STRIX_DA16_LDS_EPILOGUE
    if constexpr(STRIX_DA16_LDS_EPILOGUE) {
#include "hip_moe_da16_lds_epilogue.inc"
    } else {
#endif
#ifdef STRIX_DA16_CERTIFICATE
    bool unsafe=false; // starts after K: no accumulator live-range extension into the pipeline
#endif
#pragma unroll
    for(unsigned mi=0;mi<4U;++mi)if(mi<mb)
#pragma unroll
    for(unsigned e=0;e<8U;++e) {
        const unsigned a=rbase+16U*mi+2U*e+h;
        if(a>=count)continue;
        const float unscale=ldexpf(1.0F,-exp_of(a));
        float* out=at(view.rawdots,row_of(a));
        const auto dot_index=slot_of(a)*hidden+tile*Columns+32U*cg+i;
#ifdef STRIX_DA16_CERTIFICATE
        float d0,d1;
        if constexpr(RouteI8) {
            const auto stored=moe_route_i8_store_pair(out,dot_index-i,i,32U,hidden,route_top_k,
                acc[mi][0][e]*unscale,acc[mi][1][e]*unscale);
            d0=stored.x;d1=stored.y;
        } else {
            d0=moe_route_store<RouteBF16>(out,dot_index,acc[mi][0][e]*unscale);
            d1=moe_route_store<RouteBF16>(out,dot_index+16U,acc[mi][1][e]*unscale);
        }
        unsafe|=(__float_as_uint(d0)&0x7fffffffU)>=0x7b800000U;
        unsafe|=(__float_as_uint(d1)&0x7fffffffU)>=0x7b800000U;
#else
        out[dot_index]=acc[mi][0][e]*unscale;out[dot_index+16U]=acc[mi][1][e]*unscale;
#endif
    }
#ifdef STRIX_DA16_CERTIFICATE
    const unsigned bad=static_cast<unsigned>(__ballot(unsafe));
    if(bad&&lane==0U)atomicOr(view.unsafe,1U);
#endif
#ifdef STRIX_DA16_LDS_EPILOGUE
    }
#endif
