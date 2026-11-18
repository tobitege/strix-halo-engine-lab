#pragma once
#include <hip/hip_fp16.h>
#include "arch/gfx1151_sources.hpp"
#include "../../hip_q6p_weights.hpp"
#include <cstring>
#include <algorithm>
#include <type_traits>
namespace strix::weight::hip {
template<> struct OperandSource<Q8_0Tag,F16TileTag,Gfx1151,DenseQ8A8F16V1> {
    using U4=gfx1151::U4;
    using View=MatrixView;
    using NumericContract=DenseQ8A8F16V1;
    using Protocol=F16TileTag;
    using Fragment=gfx1151::Half16;
    static constexpr unsigned group_k=32, alignment=2, shared_bytes=0;
    static constexpr bool has_input_transform=false, has_output_transform=false;
    static Status validate(const WeightView&,const LaunchShape&);
    template<class V> static bool supported(const V& v) {
        return v.weight && v.codes && v.deltas && v.out && v.rows && v.count && v.in && v.in % 128U == 0U && v.x_stride % 32U == 0U &&
               v.in <= v.x_stride && v.out_stride >= v.count;
    }
    __device__ __forceinline__ static _Float16 scalar(std::uint32_t word,unsigned byte,float scale) {
        return gfx1151::w16_scalar(word,byte,scale);
    }
    static constexpr std::uint32_t part_bytes(std::uint32_t k) {return k/32U*34U;}
    template<class Core> __device__ __forceinline__ static void load_fragment(const View& v,
        const typename Core::Coord& c,typename Core::BFragment& fragment,typename Core::Scratch&) {
        const auto* p=v.blocks+static_cast<std::size_t>(c.row)*v.row_stride_bytes+c.k_bytes+(c.unit/2U)*34U;
        stage<Core::form>(p,fragment);
    }
    template<unsigned Form> __device__ __forceinline__ static void stage(const std::uint8_t* p,U4* nb) {
        std::uint32_t d[17];
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) {
            const U4 t = *(gfx1151::GlobalBytes16*)(p + 16U * i);
            d[4U * i] = t[0], d[4U * i + 1U] = t[1], d[4U * i + 2U] = t[2], d[4U * i + 3U] = t[3];
        }
        d[16] = *(gfx1151::GlobalBytes4*)(p + 64U);
        if constexpr ((Form & 1U) != 0U) {
            std::uint32_t x[17];
#pragma unroll
            for (unsigned i = 0; i < 17U; ++i) x[i] = d[i] ^ 0x80808080U;
#pragma unroll
            for (unsigned b = 0; b < 2U; ++b) {
                const unsigned at = 34U * b;  // the scale's half of word at / 4 (at % 4 is 0 or 2)
                const std::uint32_t s = (d[at / 4U] >> (8U * (at % 4U))) & 0xffffU;
                const gfx1151::Half2 scale2 = __builtin_bit_cast(gfx1151::Half2, s | (s << 16U));
#pragma unroll
                for (unsigned q = 0; q < 4U; ++q) {
                    U4 h;
#pragma unroll
                    for (unsigned j = 0; j < 4U; ++j) {  // codes at bytes at + 2 + 8 q + 2 j, + 1: one half of a word
                        const unsigned byte = at + 2U + 8U * q + 2U * j;
                        h[j] = __builtin_bit_cast(std::uint32_t, gfx1151::w16_pair(x[byte / 4U], (byte % 4U) / 2U, scale2));
                    }
                    nb[4U * b + q] = h;
                }
            }
            return;
        }
#pragma unroll
        for (unsigned b = 0; b < 2U; ++b) {
            const unsigned at = 34U * b;  // block: scale at bytes at, at + 1; codes from at + 2
            const float scale = static_cast<float>(
                __builtin_bit_cast(_Float16, static_cast<std::uint16_t>(d[at / 4U] >> (8U * (at % 4U)))));
#pragma unroll
            for (unsigned q = 0; q < 4U; ++q) {
                gfx1151::Half8 h;
#pragma unroll
                for (unsigned i = 0; i < 8U; ++i) {
                    const unsigned byte = at + 2U + 8U * q + i;
                    h[i] = gfx1151::w16_scalar(d[byte / 4U], byte % 4U, scale);
                }
                nb[4U * b + q] = __builtin_bit_cast(U4, h);
            }
        }
    }
    __device__ __forceinline__ static Fragment activation_fragment(const std::int8_t* codes,const float* deltas,
                                                                   unsigned stride,unsigned row,unsigned unit) {
        const std::size_t at = static_cast<std::size_t>(row) * stride + 16U * unit;
        const float delta = deltas[static_cast<std::size_t>(row) * (stride / 32U) + unit / 2U];
        const U4 t = *(gfx1151::GlobalBytes16*)(reinterpret_cast<const std::uint8_t*>(codes) + at);
        Fragment h;
#pragma unroll
        for (unsigned i = 0; i < 16U; ++i) h[i] = gfx1151::a16_value(t[i / 4U], i % 4U, delta);
        return h;
    }
};
template <unsigned Outs> struct Q8Fetch {
    std::uint64_t codes[Outs];
    unsigned scale[Outs];
    __device__ __forceinline__ void load(const unsigned char* const (&w)[Outs], std::uint32_t base) {
#pragma unroll
        for (unsigned o = 0; o < Outs; ++o) {
            const unsigned char* block = w[o] + (base / 32U) * 34U;
            scale[o] = *reinterpret_cast<const std::uint16_t*>(block);
            __builtin_memcpy(&codes[o], block + 2U + base % 32U, 8U);
        }
    }
};


template <unsigned Outs> struct Q6Fetch {
    ::strix::q6p::Group group[Outs];
    __device__ __forceinline__ void load(const ::strix::q6p::Row (&w)[Outs], std::uint32_t base) {
#pragma unroll
        for (unsigned o = 0; o < Outs; ++o) group[o] = ::strix::q6p::load_group(w[o], base);
    }
};

template<class Arithmetic> struct OperandSource<Q8_0Tag,NativeScalarTag,Gfx1151,Arithmetic> : PacketSource {
    using View=MatrixView;
    using Row=const unsigned char*;
    using Protocol=NativeScalarTag;
    using NumericContract=Arithmetic;
    using Fragment=Q8Fetch<1U>;
    template<class Segments> __device__ __forceinline__ static Row row(const Segments& s,std::uint32_t out,
        std::uint32_t width,unsigned& m,std::uint32_t& r) {
        m=0;while(m+1U<s.count && out>=s.first[m+1U]) ++m;
        r=out-s.first[m];
        return s.weights[m]+static_cast<std::uint64_t>(r)*(width/32U)*34U;
    }
    template<class Segments,unsigned Outs> __device__ static void rows(const Segments& s,std::uint32_t g,
        std::uint32_t outputs,std::uint32_t width,Row (&w)[Outs]) {
#pragma unroll
        for(unsigned o=0;o<Outs;++o) {unsigned m;std::uint32_t r;w[o]=row(s,min(g*Outs+o,outputs-1U),width,m,r);}
    }

    template<unsigned Outs> using Fetch=Q8Fetch<Outs>;
    __device__ __forceinline__ static float coefficient(unsigned scale,std::int8_t code) {return Arithmetic::coefficient(scale,code);}
    static constexpr unsigned group_k=32, alignment=2, shared_bytes=0;
    static constexpr bool has_input_transform=false, has_output_transform=false;
    static constexpr auto validate=&validate_q8_rows;
    template<unsigned Outs> __device__ static std::uint64_t codes(const Fetch<Outs>& f,unsigned o,std::uint32_t) {return f.codes[o];}
    template<unsigned Outs> __device__ static unsigned scale(const Fetch<Outs>& f,unsigned o) {return f.scale[o];}
    __device__ __forceinline__ static float scalar(const View& v,std::uint64_t row,std::uint64_t k) {
        const auto* p=v.blocks+row*v.row_stride_bytes+(k/32U)*34U;
        return coefficient(*reinterpret_cast<const std::uint16_t*>(p),static_cast<std::int8_t>(p[2U+k%32U]));
    }
    __device__ __forceinline__ static std::uint8_t code_at(const unsigned char* row,unsigned feature) {
        return row[(feature/32U)*34U+2U+feature%32U];
    }
    __device__ __forceinline__ static std::uint16_t scale_at(const unsigned char* row,unsigned group) {
        return *reinterpret_cast<const std::uint16_t*>(row+group*34U);
    }
    template<class Codes,class Scales> __device__ __forceinline__ static void load_lane(const unsigned char* pcodes,
        const std::uint16_t* pscales,std::uint32_t at,std::uint32_t chunks,Codes& codes,Scales& scales) {
        codes=*reinterpret_cast<const Codes*>(pcodes+static_cast<std::uint64_t>(at)*1024U);
        scales=*reinterpret_cast<const Scales*>(pscales+static_cast<std::uint64_t>(at%chunks)*64U*8U);
    }
    __device__ __forceinline__ static const unsigned char* block(const unsigned char* row,unsigned feature) {return row+(feature/32U)*34U;}
    template<class Codes> __device__ __forceinline__ static void load(const unsigned char* row,unsigned feature,unsigned& scale,Codes& codes) {
        const unsigned char* p=block(row,feature);
        scale=*reinterpret_cast<const std::uint16_t*>(p);
        __builtin_memcpy(&codes,p+2U+feature%32U,sizeof(Codes));
    }
};
template<class Arithmetic> struct OperandSource<Q6P32Tag,NativeScalarTag,Gfx1151,Arithmetic> : PacketSource {
    using View=MatrixView;
    using Row=::strix::q6p::Row;
    using Protocol=NativeScalarTag;
    using NumericContract=Arithmetic;
    using Fragment=Q6Fetch<1U>;
    template<class Segments> __device__ __forceinline__ static Row row(const Segments& s,std::uint32_t out,
        std::uint32_t width,unsigned& m,std::uint32_t& r) {
        m=0;while(m+1U<s.count && out>=s.first[m+1U]) ++m;
        r=out-s.first[m];return ::strix::q6p::row_of(s.weights[m],width,r);
    }
    template<class Segments,unsigned Outs> __device__ static void rows(const Segments& s,std::uint32_t g,
        std::uint32_t outputs,std::uint32_t width,Row (&w)[Outs]) {
#pragma unroll
        for(unsigned o=0;o<Outs;++o) {unsigned m;std::uint32_t r;w[o]=::strix::q6p::uniform(row(s,min(g*Outs+o,outputs-1U),width,m,r));}
    }

    template<unsigned Outs> using Fetch=Q6Fetch<Outs>;
    __device__ __forceinline__ static float coefficient(unsigned scale,std::int8_t code) {return Arithmetic::coefficient(scale,code);}
    static constexpr unsigned group_k=32, alignment=2, shared_bytes=0;
    static constexpr bool has_input_transform=false, has_output_transform=false;
    static constexpr auto validate=&validate_q6_rows;
    __device__ __forceinline__ static float scalar(const View& v,std::uint64_t row_index,std::uint64_t k) {
        Row r{v.blocks,static_cast<std::uint32_t>(row_index*v.row_stride_bytes),v.k};
        const auto feature=static_cast<std::uint32_t>(k)&~7U;
        const auto group=::strix::q6p::load_group(r,feature);
        const auto code=::strix::q6p::codes8(group.qs,group.qh,::strix::q6p::shift_of(feature));
        return coefficient(group.scale,static_cast<std::int8_t>(code>>(8U*(k&7U))));
    }
    template<unsigned Outs> __device__ static std::uint64_t codes(const Fetch<Outs>& f,unsigned o,std::uint32_t base) {
        return ::strix::q6p::codes8(f.group[o].qs,f.group[o].qh,::strix::q6p::shift_of(base));
    }
    template<unsigned Outs> __device__ static unsigned scale(const Fetch<Outs>& f,unsigned o) {return f.group[o].scale;}
};
struct __attribute__((aligned(4))) DenseA8Dword4 {unsigned x,y,z,w;};
template <unsigned Outs, bool Nontemporal = false> struct DenseA8Weights {
    unsigned d[Outs][9];
    __device__ __forceinline__ void load(const unsigned char* const (&w)[Outs], std::uint32_t b) {
#pragma unroll
        for (unsigned o = 0; o < Outs; ++o) {
            const unsigned char* p = w[o] + (34U * b & ~3U);
            if constexpr (Nontemporal) {
                using Words = unsigned __attribute__((ext_vector_type(4), aligned(4)));
                const auto lo = __builtin_nontemporal_load(reinterpret_cast<const Words*>(p));
                const auto hi = __builtin_nontemporal_load(reinterpret_cast<const Words*>(p + 16U));
#pragma unroll
                for (unsigned k = 0; k < 4U; ++k) { d[o][k] = lo[k]; d[o][k + 4U] = hi[k]; }
                d[o][8] = __builtin_nontemporal_load(reinterpret_cast<const unsigned*>(p) + 8U);
            } else {
            const auto lo = reinterpret_cast<const DenseA8Dword4*>(p)[0], hi = reinterpret_cast<const DenseA8Dword4*>(p)[1];
            d[o][0] = lo.x; d[o][1] = lo.y; d[o][2] = lo.z; d[o][3] = lo.w;
            d[o][4] = hi.x; d[o][5] = hi.y; d[o][6] = hi.z; d[o][7] = hi.w;
            d[o][8] = reinterpret_cast<const unsigned*>(p)[8];
            }
        }
    }
};
// Q6P32 form of one block per lane: the block's 16 low-nibble bytes, 8 high-pair bytes and
// scale (q6p chunk layout; Q6P32 rows of dense A8 widths are dword aligned). codes() gives
// the dwords of the block's 32 int8 codes, the same values DenseA8Weights selects.
struct __attribute__((aligned(4))) DenseA8Dword2 { unsigned x, y; };
template <unsigned Outs> struct DenseA8Q6Weights {
    unsigned qs[Outs][4], qh[Outs][2], d[Outs];
    __device__ __forceinline__ void load(const ::strix::q6p::Row (&w)[Outs], std::uint32_t b) {
#pragma unroll
        for (unsigned o = 0; o < Outs; ++o) {
            const auto c = ::strix::q6p::chunk_offsets(w[o].cols, b / 8U);
            const std::uint32_t j = b % 8U;
            const unsigned char* row = w[o].tensor + w[o].offset;
            const auto lo = *reinterpret_cast<const DenseA8Dword4*>(row + c.qs + 16U * j);
            const auto hi = *reinterpret_cast<const DenseA8Dword2*>(row + c.qh + 8U * j);
            qs[o][0] = lo.x; qs[o][1] = lo.y; qs[o][2] = lo.z; qs[o][3] = lo.w;
            qh[o][0] = hi.x; qh[o][1] = hi.y;
            d[o] = *reinterpret_cast<const std::uint16_t*>(row + c.d + 2U * j);
        }
    }
    __device__ __forceinline__ void codes(unsigned o, int (&q)[8]) const {
#pragma unroll
        for (unsigned g = 0; g < 4U; ++g) {
            const std::uint64_t c = ::strix::q6p::codes8(qs[o][g], qh[o][g / 2U], 4U * (g & 1U));
            q[2U * g] = static_cast<int>(static_cast<std::uint32_t>(c));
            q[2U * g + 1U] = static_cast<int>(static_cast<std::uint32_t>(c >> 32U));
        }
    }
};
template<> struct OperandSource<Q8_0Tag,SignedA8Tag,Gfx1151,LegacyRowsArithmetic>
    : OperandSource<Q8_0Tag,NativeScalarTag,Gfx1151,LegacyRowsArithmetic> {
    using Protocol=SignedA8Tag;
    template<unsigned Outs,bool Nontemporal=false> using Weights=DenseA8Weights<Outs,Nontemporal>;
    struct Selector {unsigned select,shift;};
    __device__ __forceinline__ static Selector selector(unsigned lane) {const unsigned odd=lane&1U;return {odd?0x07060504U:0x05040302U,odd?16U:0U};}
    template<unsigned Outs,bool NT> __device__ __forceinline__ static void decode_group(const DenseA8Weights<Outs,NT>& current,
        unsigned o,Selector s,int (&q)[8],float& scale) {
#pragma unroll
        for(unsigned k=0;k<8U;++k) q[k]=static_cast<int>(__builtin_amdgcn_perm(current.d[o][k+1U],current.d[o][k],s.select));
        scale=__half2float(__ushort_as_half(static_cast<unsigned short>((current.d[o][0]>>s.shift)&0xffffU)));
    }

    static constexpr unsigned alignment=4, paired_k=64;
    static Status validate(const WeightView&,const LaunchShape&);
    __device__ __forceinline__ static const DenseA8Dword4* paired_blocks(const unsigned char* row,unsigned pair) {
        return reinterpret_cast<const DenseA8Dword4*>(row+68U*pair);
    }
    __device__ __forceinline__ static unsigned paired_last(const unsigned char* row,unsigned pair) {
        return reinterpret_cast<const unsigned*>(row+68U*pair)[16];
    }
};
template<> struct OperandSource<Q6P32Tag,SignedA8Tag,Gfx1151,LegacyRowsArithmetic>
    : OperandSource<Q6P32Tag,NativeScalarTag,Gfx1151,LegacyRowsArithmetic> {
    using Protocol=SignedA8Tag;
    template<unsigned Outs,bool Nontemporal=false> using Weights=DenseA8Q6Weights<Outs>;
    struct Selector {};
    __device__ __forceinline__ static Selector selector(unsigned) {return {};}
    template<unsigned Outs> __device__ __forceinline__ static void decode_group(const DenseA8Q6Weights<Outs>& current,
        unsigned o,Selector,int (&q)[8],float& scale) {
        current.codes(o,q);scale=__half2float(__ushort_as_half(static_cast<unsigned short>(current.d[o])));
    }
    template<class Segments,unsigned Outs> __device__ static void rows(const Segments& s,std::uint32_t g,
        std::uint32_t outputs,std::uint32_t width,Row (&w)[Outs]) {
#pragma unroll
        for(unsigned o=0;o<Outs;++o) {unsigned m;std::uint32_t r;w[o]=row(s,std::min<std::uint32_t>(g*Outs+o,outputs-1U),width,m,r);}
    }

    static constexpr unsigned alignment=4, paired_k=64;
    static Status validate(const WeightView&,const LaunchShape&);
};
using Q8F16Source=OperandSource<Q8_0Tag,F16TileTag,Gfx1151,DenseQ8A8F16V1>;
using Q8RowsSource=OperandSource<Q8_0Tag,NativeScalarTag,Gfx1151,LegacyRowsArithmetic>;
using Q6RowsSource=OperandSource<Q6P32Tag,NativeScalarTag,Gfx1151,LegacyRowsArithmetic>;
using Q8SignedSource=OperandSource<Q8_0Tag,SignedA8Tag,Gfx1151,LegacyRowsArithmetic>;
using Q6SignedSource=OperandSource<Q6P32Tag,SignedA8Tag,Gfx1151,LegacyRowsArithmetic>;
template<bool Q6,class Arithmetic> using NativeSourceFor=OperandSource<
    std::conditional_t<Q6,Q6P32Tag,Q8_0Tag>,NativeScalarTag,Gfx1151,Arithmetic>;
template<bool Q6> using SignedSourceFor=std::conditional_t<Q6,Q6SignedSource,Q8SignedSource>;
}  // namespace strix::weight::hip
