#pragma once
#include "../source.hpp"
namespace strix::weight::hip::gfx1151 {
using U4 = std::uint32_t __attribute__((ext_vector_type(4)));
using Half16 = _Float16 __attribute__((ext_vector_type(16)));

using GlobalBytes16 = const U4 __attribute__((address_space(1), aligned(1)));
using GlobalBytes4 = const std::uint32_t __attribute__((address_space(1), aligned(1)));
using Half8 = _Float16 __attribute__((ext_vector_type(8)));
using Half2 = _Float16 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float code(std::uint32_t word, unsigned byte) {  // signed byte `byte` of a little-endian word
    return static_cast<float>(static_cast<std::int32_t>(word << (24U - 8U * byte)) >> 24);
}
// w16 of the contract for byte `byte` of a word of codes: the FP32 product is exact (8 x 11 significant bits).
__device__ __forceinline__ _Float16 w16_scalar(std::uint32_t word, unsigned byte, float scale) {
    return static_cast<_Float16>(code(word, byte) * scale);
}
// The same bits for the two codes of one 16-bit half of a word (hi: bytes 2 and 3). `biased` = word ^ 0x80808080 holds
// code + 128 per byte; v_perm_b32 puts 0x64 above each byte, i.e. the FP16 integer 1024 + code + 128; the packed
// subtraction of 1152 is exact (|code| <= 128, so code is an FP16 integer) and one packed FP16 multiply rounds the exact
// product code * scale once (RNE), the single rounding of the FP32 path: equal for subnormal, infinite and NaN scales
// too (the device test compares all 65,536 scales x 256 codes bit for bit).
__device__ __forceinline__ Half2 w16_pair(std::uint32_t biased, unsigned hi, Half2 scale2) {
    const std::uint32_t bits = __builtin_amdgcn_perm(0x64646464U, biased, hi ? 0x04030402U : 0x04010400U);
    const _Float16 bias = static_cast<_Float16>(1152.0F);
    return (__builtin_bit_cast(Half2, bits) - Half2{bias, bias}) * scale2;
}
// a16 of the contract: fp16_rne(float(code) * delta), delta FP32 (never narrowed)
__device__ __forceinline__ _Float16 a16_value(std::uint32_t word, unsigned byte, float delta) {
    return static_cast<_Float16>(code(word, byte) * delta);
}

}  // namespace strix::weight::hip::gfx1151

namespace strix::weight::hip {
template<> struct OperandSource<RawF16Tag,F16TileTag,Gfx1151,RawF16Arithmetic> {
    using View=MatrixView;
    using NumericContract=RawF16Arithmetic;
    using Protocol=F16TileTag;
    using Fragment=gfx1151::Half16;
    using U4=gfx1151::U4;
    using GlobalU4 = const U4 __attribute__((address_space(1), aligned(2)));
    static constexpr unsigned group_k=1, alignment=2, shared_bytes=0;
    static constexpr bool has_input_transform=false, has_output_transform=false;
    static Status validate(const WeightView&,const LaunchShape&);
    __device__ __forceinline__ static float scalar(const View& v,std::uint64_t row,std::uint64_t k) {
        return static_cast<float>(*reinterpret_cast<const _Float16*>(v.blocks+row*v.row_stride_bytes+2U*k));
    }
    template<class Core> __device__ __forceinline__ static void load_fragment(const View& v,
        const typename Core::Coord& c,typename Core::BFragment& fragment,typename Core::Scratch&) {
        stage(v.blocks,c.unit,fragment);
    }
    __device__ __forceinline__ static void stage(const unsigned char* wp,unsigned unit,U4* nb) {
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) nb[i] = *(GlobalU4*)(wp + std::size_t{unit} * 32U + 16U * i);
    }
    __device__ __forceinline__ static void strided_stage(const unsigned char* p,std::size_t stride,U4* nb) {
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) nb[i] = *(GlobalU4*)(p + static_cast<std::size_t>(i) * stride);
    }
};

using RawF16Source=OperandSource<RawF16Tag,F16TileTag,Gfx1151,RawF16Arithmetic>;
}
