#pragma once
#include "strix/gsq_a8_groups.hpp"
#include "strix/hip_ggml_weight_formats.hpp"
#include "arch/gfx1151_gsq.hpp"
#include <hip/hip_fp16.h>
namespace strix::weight::hip::gsq_detail {
// Internal linkage, as the incumbent definitions inside the known-rows TU's unnamed namespace: the non-forceinline
// fetch/convert members are inlined under the same heuristics, so the kernels keep their incumbent code.
namespace {
// The incumbent gfx1151 FP16 pair/mix instructions. Bound by one alias, not a template parameter: templating the
// group structs on the instruction policy changed the scheduling of the IQ4_XS F16 prefill kernels.
using Instructions=::strix::weight::hip::GsqGfx1151;
namespace gsq_a8=::strix::gsq_a8;
namespace hip_ggml=::strix::hip_ggml;
// GSQ routed-expert weight decoders of the prefill kernels (hip_moe_gsq_f16_prefill.inc, hip_moe_gsq_wmma_prefill.inc).
// Included by hip_moe_q4k_a8_known_rows.hip after down_prefill_f16x, inside its anonymous namespace.
//
// Q2_0 (type 42): 18-byte blocks of 64 weights (FP16 d, 16 code bytes; weight k at bits 2(k%4) of byte k/4), value
// d * float(q - 1), one block per K64 step. Every value is an FP16 value (q - 1 in {-1, 0, 1, 2}), so the FP16 rounding is
// exact except 2d beyond the FP16 range (infinity; g, u or the raw dot is then non-finite and the row fails closed).
// Rows hold an even block count (K % 128 == 0), so an even block starts at a dword and an odd one two bytes after a
// dword: a thread loads the five dwords from the block's dword-aligned start. FP16 conversion: the four values of a block
// (q = 0..3) are formed once per thread and step; their low and high bytes make two dword tables, and one v_perm with
// selector bytes (c0, c0 + 4, c1, c1 + 4) returns the FP16 pair of codes c0, c1 (~2 VALU per weight, no FP32 work).
typedef unsigned gsq_u32x4 __attribute__((ext_vector_type(4))) __attribute__((aligned(4)));
struct GsqQ20Raw { unsigned w[5]; };
// Global (not flat) loads of the five dwords holding Q2_0 block `block` of a row (dword-aligned start).
__device__ __forceinline__ GsqQ20Raw gsq_q20_fetch(const unsigned char* row,unsigned block) {
    const std::uintptr_t at=reinterpret_cast<std::uintptr_t>(row)+((18U*block)&~3U);
    const gsq_u32x4 v=*reinterpret_cast<const __attribute__((address_space(1))) gsq_u32x4*>(at);
    const unsigned last=*reinterpret_cast<const __attribute__((address_space(1))) unsigned*>(at+16U);
    return {{v.x,v.y,v.z,v.w,last}};
}
// Low (lt) and high (ht) bytes of the block's four FP16 values round_f16(d * float(q - 1)), byte q = value q.
__device__ __forceinline__ void gsq_q20_tables(unsigned d_bits,unsigned& lt,unsigned& ht) {
    const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(d_bits)));
    const unsigned v01=Instructions::f16_pair(d*-1.0F,d*0.0F),v23=Instructions::f16_pair(d*1.0F,d*2.0F);
    lt=__builtin_amdgcn_perm(v23,v01,0x06040200U);
    ht=__builtin_amdgcn_perm(v23,v01,0x07050301U);
}
// The 16 codes of one code dword (K order) as 8 FP16 pairs. Byte k of y_j is code 4k + j; p/q interleave the
// codes of pairs (4k, 4k + 1) / (4k + 2, 4k + 3); a pair's selector puts c and c + 4 next to each other.
__device__ __forceinline__ void gsq_q20_word(unsigned w,unsigned lt,unsigned ht,unsigned (&out)[8]) {
    const unsigned y0=w&0x03030303U,y1=(w>>2U)&0x03030303U,y2=(w>>4U)&0x03030303U,y3=(w>>6U)&0x03030303U;
    const unsigned p0=__builtin_amdgcn_perm(y1,y0,0x05010400U),p1=__builtin_amdgcn_perm(y1,y0,0x07030602U);
    const unsigned q0=__builtin_amdgcn_perm(y3,y2,0x05010400U),q1=__builtin_amdgcn_perm(y3,y2,0x07030602U);
    const auto pair=[&](unsigned c,unsigned select){return __builtin_amdgcn_perm(ht,lt,__builtin_amdgcn_perm(c|0x04040404U,c,select));};
    out[0]=pair(p0,0x05010400U);out[1]=pair(q0,0x05010400U);out[2]=pair(p0,0x07030602U);out[3]=pair(q0,0x07030602U);
    out[4]=pair(p1,0x05010400U);out[5]=pair(q1,0x05010400U);out[6]=pair(p1,0x07030602U);out[7]=pair(q1,0x07030602U);
}
// FP16 weights of a thread's share of Q2_0 block `block`: 64/TPC halves from half 64/TPC*part, as 8/TPC uint4.
template<unsigned TPC>
__device__ __forceinline__ void gsq_q20_halves(const GsqQ20Raw& r,unsigned block,unsigned part,uint4 (&out)[8U/TPC]) {
    const unsigned odd=block&1U,select=odd?0x07060504U:0x05040302U;
    unsigned lt,ht;
    gsq_q20_tables(odd?r.w[0]>>16U:r.w[0]&65535U,lt,ht);
    // Code dword k = bytes 4k..4k+3 of the codes: dwords k, k + 1 of the raw words (part 1 of two: k = 2, 3).
    unsigned w[4U/TPC+1U];
    if constexpr(TPC==1U){
#pragma unroll
        for(unsigned k=0;k<5U;++k)w[k]=r.w[k];
    } else {
        w[0]=part?r.w[2]:r.w[0];w[1]=part?r.w[3]:r.w[1];w[2]=part?r.w[4]:r.w[2];
    }
#pragma unroll
    for(unsigned k=0;k<4U/TPC;++k) {
        unsigned v[8];
        gsq_q20_word(__builtin_amdgcn_perm(w[k+1U],w[k],select),lt,ht,v);
        out[2U*k]={v[0],v[1],v[2],v[3]};out[2U*k+1U]={v[4],v[5],v[6],v[7]};
    }
}
// GSQ IQ routed-expert weights for the fast prefill profile's FP16 tiles (hip_moe_gsq_f16_prefill.inc kernels).
// Included by hip_moe_q4k_a8_known_rows.hip before hip_moe_gsq_f16_prefill.inc, inside its anonymous namespace.
//
// Each weight value is the format's own dequantization (hip_ggml::block_element: llama.cpp's value) formed in FP32 and
// rounded once to FP16: v = s * float(m) with the group scale s of the GSQ A8 decomposition (IQ3_S d*(1+2n), IQ3_XXS
// (d*(0.5+k))*0.5, IQ2_S (d*(0.5+n))*0.25 per 16-weight half, IQ4_XS d*(ls-32), IQ4_NL d) and the signed grid / table
// value m. s*m is an exact FP32 value for every type (d has 11 significant bits; the scale factor at most 6 and the
// magnitude at most 7 more: <= 24 bits), so round_f16(s*m) is computed in one v_fma_mix{lo,hi}_f16 with m entering as the
// sign-magnitude FP16 subnormal |m| * 2^-24 (|m| <= 127), s * 2^24 as the FP32 factor and -0.0 as the addend (keeping the
// sign of a zero product: block_element's (s*|m|)*(+-1) is -0 exactly when the product is). Grids are staged in LDS once
// per CTA (IQ2_S 8 KiB, IQ3_S 2 KiB, IQ3_XXS 1 KiB). Block starts are dword-aligned (IQ4_XS) or 0 / 2 mod 4 (82, 98, 110,
// 18-byte blocks in rows of a whole number of dwords; prefill_kernel_form admits only such rows): a field is read as the
// aligned dwords it touches (gsq_ld_field), so no read leaves the row.
__device__ __forceinline__ unsigned gsq_ld(const unsigned char* p) {
    return *reinterpret_cast<const __attribute__((address_space(1))) unsigned*>(reinterpret_cast<std::uintptr_t>(p)&~std::uintptr_t{3});
}
__device__ __forceinline__ unsigned gsq_byte(unsigned dword,unsigned at){return (dword>>(8U*(at&3U)))&255U;}
// The N aligned dwords covering a field of `bytes` bytes at p: only the dwords the field touches are loaded (the others are
// 0), so a field ending at the row end never reads past it.
template<unsigned N>
__device__ __forceinline__ void gsq_ld_field(const unsigned char* p,unsigned bytes,unsigned (&out)[N]) {
    const unsigned span=(static_cast<unsigned>(reinterpret_cast<std::uintptr_t>(p))&3U)+bytes;
#pragma unroll
    for(unsigned k=0;k<N;++k)out[k]=4U*k<span?gsq_ld(p+4U*k):0U;
}
// FP16 pairs of four magnitudes (bytes of mags, each < 128) with their sign bits (bit j of signs = byte j negative).
__device__ __forceinline__ uint2 gsq_signed_pairs(unsigned mags,unsigned signs,float a2) {
    const unsigned neg=((signs*0x00204081U)&0x01010101U)<<7U;
    return {Instructions::f16_mix_pair(a2,__builtin_amdgcn_perm(neg,mags,0x05010400U),-0.0F),Instructions::f16_mix_pair(a2,__builtin_amdgcn_perm(neg,mags,0x07030602U),-0.0F)};
}
// FP16 pairs of four IQ4 nibbles (bytes 0..15): kvalues_iq4nl = -127 -104 -83 -65 -49 -35 -22 -10 | 1 13 25 38 53 69 89 113.
__device__ __forceinline__ uint2 gsq_iq4_pairs(unsigned nibbles,float a2) {
    const unsigned idx=nibbles&0x07070707U;
    const unsigned neg_mag=__builtin_amdgcn_perm(0x0A162331U,0x4153687FU,idx),pos_mag=__builtin_amdgcn_perm(0x71594535U,0x26190D01U,idx);
    const unsigned pos=((nibbles>>3U)&0x01010101U)*0xffU;
    const unsigned mag=(pos_mag&pos)|(neg_mag&~pos),neg=~pos&0x80808080U;
    return {Instructions::f16_mix_pair(a2,__builtin_amdgcn_perm(neg,mag,0x05010400U),-0.0F),Instructions::f16_mix_pair(a2,__builtin_amdgcn_perm(neg,mag,0x07030602U),-0.0F)};
}
__device__ __forceinline__ float gsq_half(unsigned bits){return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));}
// One 32-weight group of a GSQ weight row as 4 uint4 of FP16 (K order). Policy per type: Raw (the loaded dwords),
// fetch(row, group) (global loads only), convert(raw, group, tables, out); tables: LDS grid bytes and its loader.
template<std::uint32_t T> struct F16Group;
template<> struct F16Group<gsq_a8::type_iq3_s> {  // d, qs[64], qh[8], signs[32], scales[4]; 110 B per 256
    static constexpr unsigned table_bytes=512U*4U;
    struct Raw { unsigned d,qs[3],qh,sg[2],sc; };
    __device__ static void load_tables(unsigned* t,unsigned i,unsigned n){for(unsigned k=i;k<512U;k+=n)t[k]=hip_ggml::iq3s_grid[k];}
    __device__ static void fetch(const unsigned char* row,unsigned g,Raw& r) {
        const unsigned char* b=row+(g/8U)*110U;const unsigned ib=g%8U;
        r.d=gsq_ld(b);r.qh=gsq_ld(b+66U+ib);r.sc=gsq_ld(b+106U+ib/2U);
        gsq_ld_field(b+2U+8U*ib,8U,r.qs);
        gsq_ld_field(b+74U+4U*ib,4U,r.sg);
    }
    __device__ static void convert(const Raw& r,unsigned g,const unsigned* grid,uint4 (&out)[4]) {
        const unsigned o=((g/8U)*110U)&3U,ib=g%8U;
        const unsigned nib=(gsq_byte(r.sc,o+106U+ib/2U)>>(4U*(ib&1U)))&15U;
        const float s=__fmul_rn(gsq_half(r.d>>(8U*o)),static_cast<float>(1+2*static_cast<int>(nib))),a2=s*16777216.0F;
        const unsigned so=(o+2U+8U*ib)&3U,q0=__builtin_amdgcn_alignbyte(r.qs[1],r.qs[0],so),q1=__builtin_amdgcn_alignbyte(r.qs[2],r.qs[1],so);
        const unsigned qh=gsq_byte(r.qh,o+66U+ib),signs=__builtin_amdgcn_alignbyte(r.sg[1],r.sg[0],(o+74U+4U*ib)&3U);
        unsigned pairs[16];
#pragma unroll
        for(unsigned m=0;m<8U;++m) {
            const unsigned idx=(((m<4U?q0:q1)>>(8U*(m&3U)))&255U)|(((qh>>m)&1U)<<8U);
            const uint2 p=gsq_signed_pairs(grid[idx],(signs>>(4U*m))&15U,a2);
            pairs[2U*m]=p.x;pairs[2U*m+1U]=p.y;
        }
#pragma unroll
        for(unsigned q=0;q<4U;++q)out[q]={pairs[4U*q],pairs[4U*q+1U],pairs[4U*q+2U],pairs[4U*q+3U]};
    }
};
template<> struct F16Group<gsq_a8::type_iq3_xxs> {  // d, qs[64] grid indices, 8 x u32 scale (4 bits) and 4 x 7-bit signs; 98 B
    static constexpr unsigned table_bytes=256U*4U;
    struct Raw { unsigned d,qs[3],aux[2]; };
    __device__ static void load_tables(unsigned* t,unsigned i,unsigned n){for(unsigned k=i;k<256U;k+=n)t[k]=hip_ggml::iq3xxs_grid[k];}
    __device__ static void fetch(const unsigned char* row,unsigned g,Raw& r) {
        const unsigned char* b=row+(g/8U)*98U;const unsigned ib=g%8U;
        r.d=gsq_ld(b);
        gsq_ld_field(b+2U+8U*ib,8U,r.qs);
        gsq_ld_field(b+66U+4U*ib,4U,r.aux);
    }
    __device__ static void convert(const Raw& r,unsigned g,const unsigned* grid,uint4 (&out)[4]) {
        const unsigned o=((g/8U)*98U)&3U,ib=g%8U;
        const unsigned aux=__builtin_amdgcn_alignbyte(r.aux[1],r.aux[0],(o+66U+4U*ib)&3U);
        const float s=__fmul_rn(__fmul_rn(gsq_half(r.d>>(8U*o)),__fadd_rn(0.5F,static_cast<float>(aux>>28U))),0.5F),a2=s*16777216.0F;
        const unsigned so=(o+2U+8U*ib)&3U,q0=__builtin_amdgcn_alignbyte(r.qs[1],r.qs[0],so),q1=__builtin_amdgcn_alignbyte(r.qs[2],r.qs[1],so);
        unsigned pairs[16];
#pragma unroll
        for(unsigned l=0;l<4U;++l) {
            const unsigned s7=(aux>>(7U*l))&127U,sb=s7|((__popc(s7)&1U)<<7U);
#pragma unroll
            for(unsigned h=0;h<2U;++h) {
                const unsigned m=2U*l+h,idx=((m<4U?q0:q1)>>(8U*(m&3U)))&255U;
                const uint2 p=gsq_signed_pairs(grid[idx],(sb>>(4U*h))&15U,a2);
                pairs[2U*m]=p.x;pairs[2U*m+1U]=p.y;
            }
        }
#pragma unroll
        for(unsigned q=0;q<4U;++q)out[q]={pairs[4U*q],pairs[4U*q+1U],pairs[4U*q+2U],pairs[4U*q+3U]};
    }
};
template<> struct F16Group<gsq_a8::type_iq2_s> {  // d, qs[64] (32 grid low bytes + 32 sign bytes), qh[8], scales[8]; 82 B
    static constexpr unsigned table_bytes=1024U*8U;
    struct Raw { unsigned d,qs[2],sg[2],qh,sc; };
    __device__ static void load_tables(unsigned* t,unsigned i,unsigned n) {
        for(unsigned k=i;k<1024U;k+=n){const std::uint64_t e=hip_ggml::iq2s_grid[k];t[2U*k]=static_cast<unsigned>(e);t[2U*k+1U]=static_cast<unsigned>(e>>32U);}
    }
    __device__ static void fetch(const unsigned char* row,unsigned g,Raw& r) {
        const unsigned char* b=row+(g/8U)*82U;const unsigned ib=g%8U;
        r.d=gsq_ld(b);r.qh=gsq_ld(b+66U+ib);r.sc=gsq_ld(b+74U+ib);
        gsq_ld_field(b+2U+4U*ib,4U,r.qs);gsq_ld_field(b+34U+4U*ib,4U,r.sg);
    }
    __device__ static void convert(const Raw& r,unsigned g,const unsigned* grid,uint4 (&out)[4]) {
        const unsigned o=((g/8U)*82U)&3U,ib=g%8U;
        const unsigned sc=gsq_byte(r.sc,o+74U+ib),qh=gsq_byte(r.qh,o+66U+ib);
        const float d=gsq_half(r.d>>(8U*o));
        const float a0=__fmul_rn(__fmul_rn(d,__fadd_rn(0.5F,static_cast<float>(sc&15U))),0.25F)*16777216.0F;
        const float a1=__fmul_rn(__fmul_rn(d,__fadd_rn(0.5F,static_cast<float>(sc>>4U))),0.25F)*16777216.0F;
        const unsigned q=__builtin_amdgcn_alignbyte(r.qs[1],r.qs[0],(o+2U+4U*ib)&3U),signs=__builtin_amdgcn_alignbyte(r.sg[1],r.sg[0],(o+34U+4U*ib)&3U);
        unsigned pairs[16];
#pragma unroll
        for(unsigned l=0;l<4U;++l) {
            const unsigned idx=((q>>(8U*l))&255U)|(((qh>>(2U*l))&3U)<<8U),sb=(signs>>(8U*l))&255U;
            const float a2=l<2U?a0:a1;
            const uint2 lo=gsq_signed_pairs(grid[2U*idx],sb&15U,a2),hi=gsq_signed_pairs(grid[2U*idx+1U],sb>>4U,a2);
            pairs[4U*l]=lo.x;pairs[4U*l+1U]=lo.y;pairs[4U*l+2U]=hi.x;pairs[4U*l+3U]=hi.y;
        }
#pragma unroll
        for(unsigned k=0;k<4U;++k)out[k]={pairs[4U*k],pairs[4U*k+1U],pairs[4U*k+2U],pairs[4U*k+3U]};
    }
};
template<> struct F16Group<gsq_a8::type_iq4_xs> {  // d, scales_h (u16), scales_l[4], qs[128]; 136 B (dword-aligned)
    static constexpr unsigned table_bytes=0U;
    struct Raw { unsigned head,sl,qs[4]; };
    __device__ static void load_tables(unsigned*,unsigned,unsigned){}
    __device__ static void fetch(const unsigned char* row,unsigned g,Raw& r) {
        const unsigned char* b=row+(g/8U)*136U;const unsigned ib=g%8U;
        r.head=gsq_ld(b);r.sl=gsq_ld(b+4U);
        for(unsigned k=0;k<4U;++k)r.qs[k]=gsq_ld(b+8U+16U*ib+4U*k);
    }
    __device__ static void convert(const Raw& r,unsigned g,const unsigned*,uint4 (&out)[4]) {
        const unsigned ib=g%8U;
        const int ls=static_cast<int>(((gsq_byte(r.sl,ib/2U)>>(4U*(ib&1U)))&15U)|(((r.head>>(16U+2U*ib))&3U)<<4U));
        const float a2=__fmul_rn(gsq_half(r.head),static_cast<float>(ls-32))*16777216.0F;
        unsigned lo[8],hi[8];
#pragma unroll
        for(unsigned k=0;k<4U;++k) {
            const uint2 a=gsq_iq4_pairs(r.qs[k]&0x0f0f0f0fU,a2),b=gsq_iq4_pairs((r.qs[k]>>4U)&0x0f0f0f0fU,a2);
            lo[2U*k]=a.x;lo[2U*k+1U]=a.y;hi[2U*k]=b.x;hi[2U*k+1U]=b.y;
        }
        out[0]={lo[0],lo[1],lo[2],lo[3]};out[1]={lo[4],lo[5],lo[6],lo[7]};out[2]={hi[0],hi[1],hi[2],hi[3]};out[3]={hi[4],hi[5],hi[6],hi[7]};
    }
};
template<> struct F16Group<gsq_a8::type_iq4_nl> {  // d, qs[16]; 18 B per 32 (the five dwords of gsq_q20_fetch)
    static constexpr unsigned table_bytes=0U;
    using Raw=GsqQ20Raw;
    __device__ static void load_tables(unsigned*,unsigned,unsigned){}
    __device__ static void fetch(const unsigned char* row,unsigned g,Raw& r){r=gsq_q20_fetch(row,g);}
    __device__ static void convert(const Raw& r,unsigned g,const unsigned*,uint4 (&out)[4]) {
        const unsigned odd=g&1U,select=odd?0x07060504U:0x05040302U;
        const float a2=gsq_half(odd?r.w[0]>>16U:r.w[0])*16777216.0F;
        unsigned lo[8],hi[8];
#pragma unroll
        for(unsigned k=0;k<4U;++k) {
            const unsigned w=__builtin_amdgcn_perm(r.w[k+1U],r.w[k],select);
            const uint2 a=gsq_iq4_pairs(w&0x0f0f0f0fU,a2),b=gsq_iq4_pairs((w>>4U)&0x0f0f0f0fU,a2);
            lo[2U*k]=a.x;lo[2U*k+1U]=a.y;hi[2U*k]=b.x;hi[2U*k+1U]=b.y;
        }
        out[0]={lo[0],lo[1],lo[2],lo[3]};out[1]={lo[4],lo[5],lo[6],lo[7]};out[2]={hi[0],hi[1],hi[2],hi[3]};out[3]={hi[4],hi[5],hi[6],hi[7]};
    }
};

// B operands of the FP16 kernels: thread part `part` of TPC fills 64/TPC halves of a K64 step of its weight row. Q2_0: one
// block per step, the part's halves of it; the 32-weight group types: groups 2*step + part (TPC 2) or both (TPC 1).
template<std::uint32_t W,unsigned TPC> struct F16Block {
    using G=F16Group<W>;
    static constexpr unsigned table_bytes=G::table_bytes;
    struct Raw { typename G::Raw g[2U/TPC]; };
    __device__ static void load_tables(unsigned* t,unsigned i,unsigned n){G::load_tables(t,i,n);}
    __device__ static void fetch(const unsigned char* row,unsigned step,unsigned part,Raw& r) {
#pragma unroll
        for(unsigned j=0;j<2U/TPC;++j)G::fetch(row,2U*step+(TPC==2U?part:j),r.g[j]);
    }
    __device__ static void convert(const Raw& r,unsigned step,unsigned part,const unsigned* tables,uint4 (&out)[8U/TPC]) {
#pragma unroll
        for(unsigned j=0;j<2U/TPC;++j) {
            uint4 o[4];
            G::convert(r.g[j],2U*step+(TPC==2U?part:j),tables,o);
#pragma unroll
            for(unsigned q=0;q<4U;++q)out[4U*j+q]=o[q];
        }
    }
    __device__ static uint4 raw4(const Raw& r){uint4 v{};__builtin_memcpy(&v,&r,sizeof(v)<sizeof(r)?sizeof(v):sizeof(r));return v;}
};
template<unsigned TPC> struct F16Block<gsq_a8::type_q2_0,TPC> {
    static constexpr unsigned table_bytes=0U;
    using Raw=GsqQ20Raw;
    __device__ static void load_tables(unsigned*,unsigned,unsigned){}
    __device__ static void fetch(const unsigned char* row,unsigned step,unsigned,Raw& r){r=gsq_q20_fetch(row,step);}
    __device__ static void convert(const Raw& r,unsigned step,unsigned part,const unsigned*,uint4 (&out)[8U/TPC]){gsq_q20_halves<TPC>(r,step,part,out);}
    __device__ static uint4 raw4(const Raw& r){return {r.w[0],r.w[1],r.w[2],r.w[3]};}
};

}  // namespace
}
