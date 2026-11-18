#pragma once
#include "trellis.cuh"
#include <cstddef>

namespace strix::glm::rtxpf {
// KxN row-major, eight FP16 elements per CUTLASS vector access. With N a
// multiple of 128, every K row and expert base remains vector-aligned.
__host__ __device__ inline std::size_t c1_b_index(unsigned tile,unsigned pos,unsigned output) {
    return std::size_t(tile/(output/16)*16+qi(pos))*output+tile%(output/16)*16+qo(pos);
}
// Inverse fragment permutation, enumerated in coalesced KxN store order.
__host__ __device__ inline unsigned c1_position(unsigned k,unsigned n) {
    return 8*((n%8)*4+(k%8)/2)+k%2+2*(k/8)+4*(n/8);
}
template<unsigned Bits>
__host__ __device__ inline U32 c1_state(const U32* words,unsigned pos) {
    static_assert(Bits>=3 && Bits<=6);
    unsigned start=(pos+1)*Bits+256*Bits-16;
    if(start>=256*Bits)start-=256*Bits;
    unsigned i=start/32,s=start%32,j=i+1==8*Bits?0:i+1;
    return ((words[i]<<s)|(s?words[j]>>(32-s):0))>>16;
}
}
