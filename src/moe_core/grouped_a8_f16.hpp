#pragma once
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "grouped_f16.hpp"

namespace strix::moe_core {
// Shared operand policy: form FP32 values first, then round exactly once.
using A8F16Half16 = _Float16 __attribute__((ext_vector_type(16)));
using A8F16Float8 = float __attribute__((ext_vector_type(8)));
using A8F16Word8 = unsigned __attribute__((ext_vector_type(8)));
using A8F16Word4 = unsigned __attribute__((ext_vector_type(4)));
struct A8F16Types {
    using Word4=A8F16Word4; using Word8=A8F16Word8;
    using Half16=A8F16Half16; using Float8=A8F16Float8;
};
using A8F16Fragments=GroupedF16Fragments<A8F16Types>;
__device__ __forceinline__ unsigned a8_f16_pair(float a, float b) {
    return unsigned(__half_as_ushort(__float2half_rn(a))) |
           (unsigned(__half_as_ushort(__float2half_rn(b))) << 16U);
}
// Four candidate halves in two byte tables; one perm selects each pair.
struct A8F16Table {
    unsigned lo, hi;
    __device__ A8F16Table(float d, float m) : A8F16Table(d*0.0F-m,d-m,d*2.0F-m,d*3.0F-m) {}
    __device__ A8F16Table(float a,float b,float c,float d) {
        const unsigned ab=a8_f16_pair(a,b),cd=a8_f16_pair(c,d);
        lo=__builtin_amdgcn_perm(cd,ab,0x06040200U);
        hi=__builtin_amdgcn_perm(cd,ab,0x07050301U);
    }
    // codes holds four byte codes (0..3), as in the shared GSQ decoder.
    template<unsigned Pair>
    __device__ __forceinline__ unsigned packed_pair(unsigned codes) const {
        const unsigned select=__builtin_amdgcn_perm(codes|0x04040404U,codes,Pair==0 ? 0x05010400U : 0x07030602U);
        return __builtin_amdgcn_perm(hi,lo,select);
    }
};
} // namespace strix::moe_core
