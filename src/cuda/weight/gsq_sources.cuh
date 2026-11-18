#pragma once
#include "source.cuh"
#include "strix/cuda/qwen_quant.cuh"
#include "strix/weight/formats/ggml.hpp"

namespace strix::cuda::weight {

struct CudaFourChainImma {};
struct CudaNativeFp32 {};
struct GsqIq3SA8Dot4Cuda {};
using Iq3STag=GgmlTag<21>;
using Iq3XXSTag=GgmlTag<18>;
using Iq2STag=GgmlTag<22>;
using Iq4XSTag=GgmlTag<23>;
using Iq4NLTag=GgmlTag<20>;
using Q2_0Tag=GgmlTag<42>;

struct GsqView {
    const std::uint8_t* blocks;
    std::uint64_t row_stride_bytes,expert_stride_bytes;
    std::uint32_t k,n;
};
struct CudaSignedGroup32 { std::int8_t codes[32]; float scales[2]; };
struct A8RowFragment { unsigned words[2]; float scale,offset; };

// These policies retain the incumbent raw transports and arithmetic. Format
// choices are resolved at the launch boundary, never inside the tuned cores.
template<unsigned T,class Arch> struct LegacyProjectionSource {
    static_assert(T==12 || T==13);
    static constexpr unsigned project_wb=T==13?176:144,project_qs=T==13?48:16;
    static constexpr bool legacy_fast=true;
    __host__ __device__ static std::uint64_t project_row_bytes(unsigned k) { return std::uint64_t(k/256)*project_wb; }
    template<unsigned OUTS,unsigned THREADS>
    __device__ static void stage_project(unsigned char* dst,const std::uint8_t* gate,const std::uint8_t* up,
                                         unsigned expert,unsigned out,unsigned hidden,unsigned ffn,unsigned b,unsigned t) {
        constexpr unsigned CH=project_wb/16;
        for(unsigned i=t;i<2U*OUTS*CH;i+=THREADS) {
            const unsigned m=i/(OUTS*CH),rem=i%(OUTS*CH),o=rem/CH,c=rem%CH;
            const auto* w=(m?up:gate)+((std::uint64_t(expert)*ffn+out+o)*(hidden/256U)+b)*project_wb;
            Arch::copy16(dst+(m*OUTS+o)*project_wb+16U*c,w+16U*c);
        }
    }
    __device__ static void project_scales(const unsigned char* row,unsigned j,float& d,float& m) {
        const uint4 h=*reinterpret_cast<const uint4*>(row);
        d=__fmul_rn(Arch::half(h.x&65535U),float(Arch::factor(h,j,false)));
        m=__fmul_rn(Arch::half(h.x>>16U),float(Arch::factor(h,j,true)));
    }
    struct Packed { unsigned words[4]; };
    __device__ static Packed project_load(const unsigned char* row,unsigned p,unsigned q) {
        Packed v{};
        v.words[0]=Arch::load32(row+project_qs+32U*p+4U*q);
        v.words[1]=Arch::load32(row+project_qs+32U*p+16U+4U*q);
        if constexpr(T==13) { v.words[2]=Arch::load32(row+16U+4U*q); v.words[3]=Arch::load32(row+32U+4U*q); }
        return v;
    }
    __device__ static void project_unpack(const Packed& v,unsigned j,unsigned G,unsigned (&b)[2]) {
        b[0]=(v.words[0]>>(4U*G))&0x0f0f0f0fU; b[1]=(v.words[1]>>(4U*G))&0x0f0f0f0fU;
        if constexpr(T==13) { b[0]|=((v.words[2]>>j)&0x01010101U)<<4U; b[1]|=((v.words[3]>>j)&0x01010101U)<<4U; }
    }
    __device__ static void project_words(const unsigned char* row,unsigned p,unsigned q,unsigned (&w)[4]) {
        const auto v=project_load(row,p,q); unsigned a[2],b[2];
        project_unpack(v,2*p,0,a); project_unpack(v,2*p+1,1,b);
        w[0]=a[0]; w[1]=a[1]; w[2]=b[0]; w[3]=b[1];
    }
    __device__ static A8RowFragment project_row(const std::uint8_t* row,unsigned b,unsigned j,unsigned sub) {
        const auto* blk=row+b*project_wb; const auto h=*reinterpret_cast<const uint4*>(blk);
        const auto* qs=reinterpret_cast<const unsigned*>(blk+project_qs+32U*(j/2)+8U*sub);
        const unsigned q0=qs[0],q1=qs[1];
        unsigned h0=0,h1=0;
        if constexpr(T==13) { const auto* qh=reinterpret_cast<const unsigned*>(blk+16U+8U*sub); h0=qh[0]; h1=qh[1]; }
        A8RowFragment r{};
        r.scale=__fmul_rn(Arch::half(h.x&65535U),float(Arch::factor(h,j,false)));
        r.offset=__fmul_rn(Arch::half(h.x>>16U),float(Arch::factor(h,j,true)));
        r.words[0]=j%2?(q0>>4)&0x0f0f0f0fU:q0&0x0f0f0f0fU;
        r.words[1]=j%2?(q1>>4)&0x0f0f0f0fU:q1&0x0f0f0f0fU;
        if constexpr(T==13) { r.words[0]|=((h0>>j)&0x01010101U)<<4; r.words[1]|=((h1>>j)&0x01010101U)<<4; }
        return r;
    }
    __device__ static void mma(const unsigned (&a)[4],unsigned b0,unsigned b1,int (&r)[4],int (&)[4]) { Arch::mma(a,b0,b1,r); }
    __device__ static float project_value(float d,float m,int dot,int,float sum) {
        return __fsub_rn(__fmul_rn(d,Arch::group_float(dot)),__fmul_rn(m,sum));
    }
    __device__ static float row_project_value(float d,float m,int dot,float sum) {
        dot+=__shfl_xor_sync(0xffffffffU,dot,1); dot+=__shfl_xor_sync(0xffffffffU,dot,2);
        return __fsub_rn(__fmul_rn(d,float(dot)),__fmul_rn(m,sum));
    }
    __device__ static float fast_project_acc(int dot,float e,float d,float m,float es,float acc) {
        const float x=Arch::block_acc(dot,e,d,acc); return __fmaf_rn(-m,es,x);
    }
};
template<unsigned T,class Arch> struct LegacyDownSource {
    static_assert(T==7 || T==8);
    static constexpr unsigned down_wb=T==7?24U:34U,down_ws=T==7?112U:144U;
    static constexpr bool legacy_fast=true;
    __host__ __device__ static std::uint64_t down_row_bytes(unsigned k) { return std::uint64_t(k/32)*down_wb; }
    template<unsigned OUTS,unsigned THREADS>
    __device__ static void stage_down(unsigned char* dst,const std::uint8_t* weights,unsigned expert,
                                      unsigned out,unsigned hidden,unsigned ffn,unsigned c,unsigned t) {
        constexpr unsigned groups_per_row=4;
        const unsigned groups=ffn/32U;
        if constexpr(T==7) {
            for(unsigned i=t;i<OUTS*6U;i+=THREADS) { const unsigned o=i/6U,k=i%6U;
                const auto* w=weights+((std::uint64_t(expert)*hidden+out+o)*groups+groups_per_row*c)*down_wb;
                Arch::copy16(dst+o*down_ws+16U*k,w+16U*k); }
        } else {
            for(unsigned i=t;i<OUTS*17U;i+=THREADS) { const unsigned o=i/17U,k=i%17U;
                const auto* w=weights+((std::uint64_t(expert)*hidden+out+o)*groups+groups_per_row*c)*down_wb;
                Arch::copy8(dst+o*down_ws+8U*k,w+8U*k); }
        }
    }
    __device__ static void down_scales(const unsigned char* row,unsigned j,float& d,float& m) {
        const auto* h=reinterpret_cast<const unsigned short*>(row+j*down_wb);
        d=Arch::half(h[0]); if constexpr(T==7) m=Arch::half(h[1]); else m=0.0F;
    }
    __device__ static void down_words(const unsigned char* row,unsigned j,unsigned q,unsigned& b0,unsigned& b1) {
        const auto* blk=row+j*down_wb;
        if constexpr(T==7) {
            const unsigned qh=Arch::load32(blk+4U),qs=Arch::load32(blk+8U+4U*q);
            const unsigned l=(qh>>(4U*q))&15U,u=(qh>>(16U+4U*q))&15U;
            b0=(qs&0x0f0f0f0fU)|((l*0x00204081U)&0x01010101U)<<4U;
            b1=((qs>>4U)&0x0f0f0f0fU)|((u*0x00204081U)&0x01010101U)<<4U;
        } else {
            const unsigned base=j*down_wb+2U;
            if(base&3U) {
                const unsigned al=base&~3U;
                const unsigned x0=Arch::load32(row+al+4U*q),x1=Arch::load32(row+al+4U*q+4U);
                const unsigned y0=Arch::load32(row+al+16U+4U*q),y1=Arch::load32(row+al+16U+4U*q+4U);
                b0=__byte_perm(x0,x1,0x5432U); b1=__byte_perm(y0,y1,0x5432U);
            } else { b0=Arch::load32(row+base+4U*q); b1=Arch::load32(row+base+16U+4U*q); }
        }
    }
    __device__ static A8RowFragment down_row(const std::uint8_t* row,unsigned g,unsigned sub) {
        const auto* blk=row+g*down_wb; const auto* hw=reinterpret_cast<const unsigned short*>(blk);
        A8RowFragment r{}; r.scale=Arch::half(hw[0]);
        if constexpr(T==7) {
            r.offset=Arch::half(hw[1]); const unsigned qh=*reinterpret_cast<const unsigned*>(blk+4U);
            const unsigned qs=*reinterpret_cast<const unsigned*>(blk+8U+4U*(sub&3U)),hb=(qh>>(4U*sub))&15U;
            r.words[0]=(sub<4U?qs&0x0f0f0f0fU:(qs>>4U)&0x0f0f0f0fU)|((hb*0x00204081U)&0x01010101U)<<4U;
        } else r.words[0]=unsigned(hw[1U+2U*sub])|(unsigned(hw[2U+2U*sub])<<16U);
        return r;
    }
    __device__ static float down_value(float d,float m,int dot,float sum) {
        const float first=__fmul_rn(d,Arch::group_float(dot));
        if constexpr(T==7) return __fadd_rn(first,__fmul_rn(m,sum)); else return first;
    }
    __device__ static float row_down_value(float d,float m,int dot,float sum) {
        const float first=__fmul_rn(d,float(dot));
        if constexpr(T==7) return __fadd_rn(first,__fmul_rn(m,sum)); else return first;
    }
    __device__ static float fast_down_acc(int dot,float e,float d,float m,float es,float acc) {
        float x=Arch::block_acc(dot,e,d,acc);
        if constexpr(T==7) x=__fmaf_rn(m,es,x); return x;
    }
};

template<class Arch> struct OperandSource<GgmlTag<12>,SignedA8Tag,Arch,CudaFourChainImma>:LegacyProjectionSource<12,Arch> {};
template<class Arch> struct OperandSource<GgmlTag<13>,SignedA8Tag,Arch,CudaFourChainImma>:LegacyProjectionSource<13,Arch> {};
template<class Arch> struct OperandSource<GgmlTag<7>,SignedA8Tag,Arch,CudaFourChainImma>:LegacyDownSource<7,Arch> {};
template<class Arch> struct OperandSource<GgmlTag<8>,SignedA8Tag,Arch,CudaFourChainImma>:LegacyDownSource<8,Arch> {};

// Scalar extraction is shared by the MMA and dp4a operand protocols. Grid
// placement is supplied by the canonical CUDA decoder, with its original notices.
template<unsigned T> struct GsqDecode {
    static_assert(T==18 || T==20 || T==21 || T==22 || T==23 || T==42);
    static constexpr unsigned be=::strix::weight::formats::gsq_block_elements(T);
    static constexpr unsigned bb=::strix::weight::formats::gsq_block_bytes(T);
    static constexpr bool halves=T==22;
    __host__ __device__ static const std::uint8_t* block(const std::uint8_t* row,unsigned group,unsigned& sub) {
        sub=group%(be/32); return row+std::uint64_t(group/(be/32))*bb;
    }
    __host__ __device__ static void scales(const std::uint8_t* row,unsigned group,float& a,float& b) {
        unsigned s; const auto* x=block(row,group,s);
        const float d=quant::detail::fp16(std::uint16_t(x[0])|(std::uint16_t(x[1])<<8));
        b=0.0F;
        if constexpr(T==20 || T==42) a=d;
        if constexpr(T==21) a=d*float(1+2*int((x[::strix::weight::formats::Iq3S::scales+s/2]>>(4*(s%2)))&15));
        if constexpr(T==18) {
            const unsigned off=66+4*s;
            const unsigned aux=unsigned(x[off])|(unsigned(x[off+1])<<8)|(unsigned(x[off+2])<<16)|(unsigned(x[off+3])<<24);
            a=(d*(0.5F+float(aux>>28)))*0.5F;
        }
        if constexpr(T==22) { a=(d*(0.5F+float(x[74+s]&15)))*0.25F; b=(d*(0.5F+float(x[74+s]>>4)))*0.25F; }
        if constexpr(T==23) {
            const unsigned high=unsigned(x[2])|(unsigned(x[3])<<8);
            const unsigned ls=((x[4+s/2]>>(4*(s%2)))&15)|(((high>>(2*s))&3)<<4);
            a=d*float(int(ls)-32);
        }
    }
    __host__ __device__ static int code(const std::uint8_t* row,unsigned group,unsigned k) {
        unsigned s; const auto* x=block(row,group,s);
        if constexpr(T==42) { const unsigned e=32*s+k; return int((x[2+e/4]>>(2*(e%4)))&3)-1; }
        if constexpr(T==20 || T==23) {
            const auto* qs=T==20?x+2:x+8+16*s;
            return int(quant::detail::kvalue((qs[k%16]>>(4*(k/16)))&15));
        }
        if constexpr(T==21) {
            using Iq=::strix::weight::formats::Iq3S;
            const unsigned l=k/8,j=k%8,qh=x[Iq::qh+s];
            const unsigned idx=x[Iq::qs+8*s+2*l+j/4]|((qh<<(8-2*l-j/4))&256);
            const unsigned grid=quant::detail::iq3s_grid(idx);
            return int((grid>>(8*(j%4)))&255)*((x[Iq::signs+4*s+l]>>j)&1?-1:1);
        }
        if constexpr(T==18) {
            const unsigned off=66+4*s;
            const unsigned aux=unsigned(x[off])|(unsigned(x[off+1])<<8)|(unsigned(x[off+2])<<16)|(unsigned(x[off+3])<<24);
            const unsigned l=k/8,j=k%8,sign=quant::detail::ksigns((aux>>(7*l))&127);
            const unsigned grid=quant::detail::iq3xxs_grid(x[2+8*s+2*l+j/4]);
            return int((grid>>(8*(j%4)))&255)*((sign>>j)&1?-1:1);
        }
        if constexpr(T==22) {
            const unsigned l=k/8,j=k%8,qh=x[66+s];
            const auto grid=quant::detail::iq2s_grid(x[2+4*s+l]|((qh<<(8-2*l))&0x300));
            return int((grid>>(8*j))&255)*((x[34+4*s+l]>>j)&1?-1:1);
        }
        return 0;
    }
    __host__ __device__ static unsigned word(const std::uint8_t* row,unsigned group,unsigned index) {
        unsigned r=0;
        #pragma unroll
        for(unsigned j=0;j<4;++j) r|=(unsigned(code(row,group,4*index+j))&255)<<(8*j);
        return r;
    }
};

template<unsigned T,class Arch>
struct OperandSource<GgmlTag<T>,SignedA8Tag,Arch,GsqIq3SA8Dot4Cuda> {
    using View=GsqView;
    using Group=typename Arch::SignedGroup;
    using Decode=GsqDecode<T>;
    static constexpr unsigned group_k=32,alignment=2,shared_bytes=0;
    static constexpr bool has_input_transform=false,has_output_transform=false;
    static constexpr bool legacy_fast=false;
    static constexpr unsigned project_wb=((256/Decode::be*Decode::bb)+15)&~15U;
    static constexpr unsigned down_wb=32,down_ws=160;
    __host__ __device__ static std::uint64_t project_row_bytes(unsigned k) { return std::uint64_t(k/Decode::be)*Decode::bb; }
    __host__ __device__ static std::uint64_t down_row_bytes(unsigned k) { return std::uint64_t(k/Decode::be)*Decode::bb; }
    static ::strix::weight::Status validate(const ::strix::weight::WeightView& view,const LaunchShape& shape) {
        using namespace ::strix::weight;
        constexpr const char* storage=T==18?"gguf.iq3_xxs":T==20?"gguf.iq4_nl":T==21?"gguf.iq3_s":
                                      T==22?"gguf.iq2_s":T==23?"gguf.iq4_xs":"gguf.q2_0.strix";
        if(view.storage.name!=storage || view.storage.revision!=1)
            return {Error::unsupported_arithmetic,"GSQ source storage differs from bound trait"};
        if(!shape.k || !shape.n || shape.k>UINT32_MAX || shape.n>UINT32_MAX || shape.k%Decode::be || shape.split_count!=1)
            return {Error::unsupported_shape,"GSQ source requires complete storage blocks and one split"};
        const auto row=shape.k/Decode::be*Decode::bb;
        if(shape.expert_stride && (shape.n>UINT64_MAX/row || shape.expert_stride<shape.n*row))
            return {Error::unsupported_shape,"GSQ expert stride overlaps rows"};
        const auto* descriptor=find_format(builtin_registry(),view.storage);
        if(!descriptor) return {Error::unknown_format,"GSQ storage descriptor missing"};
        Geometry geometry{}; auto s=descriptor->geometry(view.parameters,geometry);
        if(!s) return s;
        if(geometry.block_elements!=Decode::be || geometry.block_bytes!=Decode::bb)
            return {Error::unsupported_arithmetic,"GSQ source storage differs from bound trait"};
        if(view.payload.range.bytes<row || view.payload.range.offset%alignment)
            return {Error::malformed,"GSQ source payload alignment or extent"};
        return descriptor->validate(view);
    }
    static ::strix::weight::Status make_view(const ::strix::weight::WeightView& view,const UploadedSlices& slices,
                                             const LaunchShape& shape,View& out) {
        auto s=validate(view,shape); if(!s) return s;
        const auto row=shape.k/Decode::be*Decode::bb;
        if(!slices.payload.data || slices.payload.bytes<view.payload.range.bytes ||
           reinterpret_cast<std::uintptr_t>(slices.payload.data)%alignment)
            return {::strix::weight::Error::malformed,"GSQ uploaded payload extent or alignment"};
        out={static_cast<const std::uint8_t*>(slices.payload.data),row,
             shape.expert_stride?shape.expert_stride:shape.n*row,std::uint32_t(shape.k),std::uint32_t(shape.n)};
        return {::strix::weight::Error::ok,{}};
    }
    __host__ __device__ static Group signed_group(const View& view,std::uint64_t row,std::uint64_t group) {
        Group result{}; const auto* ptr=view.blocks+row*view.row_stride_bytes;
        Decode::scales(ptr,unsigned(group),result.scales[0],result.scales[1]);
        #pragma unroll
        for(unsigned i=0;i<32;++i) result.codes[i]=std::int8_t(Decode::code(ptr,unsigned(group),i));
        return result;
    }
    // Only byte loads are used for GSQ raw blocks: IQ3_S rows may start at
    // 2-byte alignment, so the incumbent 16-byte cp.async contract is ineligible.
    template<unsigned OUTS,unsigned THREADS>
    __device__ static void stage_project(unsigned char* dst,const std::uint8_t* gate,const std::uint8_t* up,
                                         unsigned expert,unsigned out,unsigned hidden,unsigned ffn,unsigned b,unsigned t) {
        constexpr unsigned raw=256/Decode::be*Decode::bb;
        for(unsigned i=t;i<2*OUTS*raw;i+=THREADS) {
            const unsigned m=i/(OUTS*raw),r=(i/raw)%OUTS,k=i%raw;
            const auto* w=(m?up:gate)+(std::uint64_t(expert)*ffn+out+r)*(hidden/Decode::be*Decode::bb)+b*raw;
            dst[(m*OUTS+r)*project_wb+k]=w[k];
        }
    }
    __device__ static void project_scales(const unsigned char* row,unsigned j,float& d,float& m) { Decode::scales(row,j,d,m); }
    __device__ static void project_words(const unsigned char* row,unsigned p,unsigned q,unsigned (&w)[4]) {
        w[0]=Decode::word(row,2*p,q); w[1]=Decode::word(row,2*p,q+4);
        w[2]=Decode::word(row,2*p+1,q); w[3]=Decode::word(row,2*p+1,q+4);
    }
    __device__ static A8RowFragment project_row(const std::uint8_t* row,unsigned b,unsigned j,unsigned sub) {
        A8RowFragment f{}; Decode::scales(row,8*b+j,f.scale,f.offset);
        f.words[0]=Decode::word(row,8*b+j,2*sub); f.words[1]=Decode::word(row,8*b+j,2*sub+1); return f;
    }
    __device__ static void mma(const unsigned (&a)[4],unsigned b0,unsigned b1,int (&r)[4],int (&second)[4]) {
        if constexpr(Decode::halves) { Arch::mma(a,b0,0,r); Arch::mma(a,0,b1,second); }
        else Arch::mma(a,b0,b1,r);
    }
    __device__ static float project_value(float d,float m,int r,int second,float) {
        const float a=__fmul_rn(d,Arch::group_float(r));
        if constexpr(Decode::halves) return __fadd_rn(a,__fmul_rn(m,Arch::group_float(second)));
        else return a;
    }
    __device__ static float row_project_value(float d,float m,int dot,float) {
        if constexpr(Decode::halves) {
            dot+=__shfl_xor_sync(0xffffffffU,dot,1);
            const unsigned lane=threadIdx.x&31U;
            const int low=__shfl_sync(0xffffffffU,dot,int(lane&~3U));
            const int high=__shfl_sync(0xffffffffU,dot,int((lane&~3U)+2));
            return __fadd_rn(__fmul_rn(d,float(low)),__fmul_rn(m,float(high)));
        } else {
            dot+=__shfl_xor_sync(0xffffffffU,dot,1); dot+=__shfl_xor_sync(0xffffffffU,dot,2);
            return __fmul_rn(d,float(dot));
        }
    }
    // Down uses a compact four-group transport: exact signed bytes plus FP32
    // scales. Each source group is decoded once at the stage boundary.
    template<unsigned OUTS,unsigned THREADS>
    __device__ static void stage_down(unsigned char* dst,const std::uint8_t* weights,unsigned expert,
                                      unsigned out,unsigned hidden,unsigned ffn,unsigned c,unsigned t) {
        static_assert(T==20 || T==42,"down source requires admitted 640-wide storage");
        for(unsigned i=t;i<OUTS*4;i+=THREADS) {
            const unsigned r=i/4,j=i%4;
            const auto* row=weights+(std::uint64_t(expert)*hidden+out+r)*(ffn/Decode::be*Decode::bb);
            // Each 4-group row stores 128 code bytes, then 4 scales.
            for(unsigned w=0;w<8;++w) reinterpret_cast<unsigned*>(dst+r*160)[8*j+w]=Decode::word(row,4*c+j,w);
            float a,b; Decode::scales(row,4*c+j,a,b); reinterpret_cast<float*>(dst+r*160+128)[j]=a;
        }
    }
    static constexpr unsigned down_stage_stride=160;
    __device__ static void down_scales(const unsigned char* row,unsigned j,float& d,float& m) { d=reinterpret_cast<const float*>(row+128)[j]; m=0.0F; }
    __device__ static void down_words(const unsigned char* row,unsigned j,unsigned q,unsigned& b0,unsigned& b1) {
        b0=reinterpret_cast<const unsigned*>(row)[8*j+q]; b1=reinterpret_cast<const unsigned*>(row)[8*j+q+4];
    }
    __device__ static A8RowFragment down_row(const std::uint8_t* row,unsigned group,unsigned sub) {
        A8RowFragment f{}; Decode::scales(row,group,f.scale,f.offset); f.words[0]=Decode::word(row,group,sub); return f;
    }
    __device__ static float down_value(float d,float,int dot,float) { return __fmul_rn(d,Arch::group_float(dot)); }
    __device__ static float row_down_value(float d,float,int dot,float) { return __fmul_rn(d,float(dot)); }
};

struct RuntimeGgmlTag {};
template<class Arch>
struct OperandSource<RuntimeGgmlTag,NativeScalarTag,Arch,CudaNativeFp32> {
    __host__ __device__ static std::uint64_t row_bytes(unsigned type,unsigned width) {
        const auto geometry=quant::geometry(type);
        return std::uint64_t(width)/geometry.block_elements*geometry.block_bytes;
    }
    __device__ static void decode(unsigned type,const std::uint8_t* row,unsigned group,float* out) {
        quant::decode_row_group(type,row,group,out);
    }
};

} // namespace strix::cuda::weight
