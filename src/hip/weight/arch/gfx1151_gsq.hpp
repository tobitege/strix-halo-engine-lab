#pragma once
#include "gfx1151_sources.hpp"
namespace strix::weight::hip {
// Incumbent gfx1151 instructions; shared by all GSQ F16 source protocols.
struct GsqGfx1151 : Gfx1151 {
    __device__ __forceinline__ static unsigned f16_pair(float a,float b) {
        const _Float16 x=static_cast<_Float16>(a),y=static_cast<_Float16>(b);
        return static_cast<unsigned>(__builtin_bit_cast(unsigned short,x))|
            static_cast<unsigned>(__builtin_bit_cast(unsigned short,y))<<16U;
    }
    __device__ __forceinline__ static unsigned f16_mix_pair(float a2,unsigned q2,float b) {
        unsigned r;
        __asm__("v_fma_mixlo_f16 %0, %1, %2, %3 op_sel_hi:[0,1,0]\n\t"
                "v_fma_mixhi_f16 %0, %1, %2, %3 op_sel:[0,1,0] op_sel_hi:[0,1,0]" : "=&v"(r) : "v"(a2),"v"(q2),"v"(b));
        return r;
    }
};
}
