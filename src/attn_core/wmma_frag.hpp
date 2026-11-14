#pragma once
// attn_core fragment loads and the accumulator row map (RDNA3 layout): A lane l holds row l % 16, k 0..15; B lane l
// holds column l % 16, k 0..15 (both half-waves hold the same data); accumulator element e of lane l is
// (row acc_row(e, l) = 2 e + l / 16, column l % 16).
#include "attn_core/wmma_types.hpp"

namespace strix::attn_core::tile {

__device__ __forceinline__ Frag frag32(const void* p) {  // 32 B (16 operand elements), 16-B aligned (LDS or global)
    const U4* v = static_cast<const U4*>(p);
    const U4 lo = v[0], hi = v[1];
    Frag f;
    __builtin_memcpy(&f, &lo, 16U);
    __builtin_memcpy(reinterpret_cast<char*>(&f) + 16U, &hi, 16U);
    return f;
}
// 16 FP32 values -> one A/B fragment of operand elements.
template <Operand O> __device__ __forceinline__ Frag frag_from_f32(const float* p) {
    const float4* q = reinterpret_cast<const float4*>(p);
    const U4 lo = pack8<O>(q[0], q[1]), hi = pack8<O>(q[2], q[3]);
    Frag f;
    __builtin_memcpy(&f, &lo, 16U);
    __builtin_memcpy(reinterpret_cast<char*>(&f) + 16U, &hi, 16U);
    return f;
}
// XOR-swizzled row-major tile of `pieces` 16-B pieces per row (idx-wmma's 128 x 128 key tile): row r's piece c at
// r * pieces + (c ^ (r & 15)); a fragment read (16 rows, one piece each) and a row-per-lane write hit distinct banks.
__host__ __device__ constexpr unsigned piece_slot(unsigned row, unsigned piece, unsigned pieces) { return row * pieces + (piece ^ (row & 15U)); }
__host__ __device__ constexpr unsigned slot(unsigned row, unsigned d, unsigned pieces) { return piece_slot(row, d / 8U, pieces) * 8U + d % 8U; }
__device__ __forceinline__ Frag frag_swz(const U4* tile, unsigned row, unsigned step, unsigned pieces) {
    const U4 lo = tile[piece_slot(row, 2U * step, pieces)], hi = tile[piece_slot(row, 2U * step + 1U, pieces)];
    Frag f;
    __builtin_memcpy(&f, &lo, 16U);
    __builtin_memcpy(reinterpret_cast<char*>(&f) + 16U, &hi, 16U);
    return f;
}
__host__ __device__ constexpr unsigned acc_row(unsigned e, unsigned lane) { return 2U * e + (lane >> 4U); }
// Value of accumulator row acc_row(e, lane) from a per-row scalar held by lane `row % 16` (dense's __shfl form).
__device__ __forceinline__ float row_bcast(float x, unsigned e, unsigned lane) { return __shfl(x, static_cast<int>(acc_row(e, lane))); }
// Per-row scalars (alpha, 1/l) of a 16-row M tile mt kept in LDS so one lane reads its 8 rows contiguously: row
// mt * 16 + acc_row(e, lane) at [mt * 16 + (lane / 16) * 8 + e].
__host__ __device__ constexpr unsigned row_slot(unsigned row) { return (row >> 4U) * 16U + (row & 1U) * 8U + ((row & 15U) >> 1U); }
__device__ __forceinline__ void row_scalars(const float* slots, unsigned mt, unsigned lane, float (&v)[8]) {
    const float4 a = *reinterpret_cast<const float4*>(slots + mt * 16U + (lane >> 4U) * 8U);
    const float4 b = *reinterpret_cast<const float4*>(slots + mt * 16U + (lane >> 4U) * 8U + 4U);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

}  // namespace strix::attn_core::tile
