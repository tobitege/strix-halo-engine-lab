#pragma once
// Shared small-row GEMV core for decode: out[r][o] = sum_k W[o][k] * A[r][k]
// for R activation rows (1..16) that share one sweep over the weights.
//
// Two sweeps over 16-weight chunks, shared as textual bodies (a kernel includes
// one and binds its operand, lane sum and epilogue; see the .inc headers):
//   gemv_rows_direct.inc  each wave loads its own activations and shares them
//                         between its T outputs: A is read once per T outputs.
//   gemv_rows_staged.inc  the block stages the next 32 chunks of every row in
//                         LDS and all Waves * T outputs share that copy: A is
//                         read once per Waves * T outputs.
// Both keep one arithmetic contract: accumulator (stream, output, row, lane)
// starts at +0 and visits chunks lane, lane + 32, ... in ascending order, the
// 16 elements of a chunk in ascending order, as acc = acc + weight *
// activation in FP32. The schedules only change where the activations come
// from and how many outputs share them, so they give the same bits.
//
// Families with another lane topology (q8_rows: eight leaves per lane) keep
// their own products and build the same stepped sweep from stage_window() and
// stepped() below. Admission is by measured geometry (rows, K, N) in each
// family's launcher.
#include <cstddef>
#include <cstdint>
#if defined(__HIP__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

namespace strix::core::gemv_rows {

inline constexpr unsigned max_rows = 16U;
inline constexpr std::size_t lds_limit = 65536U;
// LDS bytes of the staged sweep: rows x 4 float4 groups x 32 lanes per buffer.
constexpr std::size_t stage_bytes(unsigned rows, unsigned buffers) { return std::size_t(buffers) * rows * 4U * 32U * 16U; }

#if defined(__HIP__) || defined(__HIPCC__)
// LDS slot of vector v (16 bytes) of a row inside one step window: each row's
// window stays contiguous.
struct RowRuns {
    unsigned vectors;
    __device__ unsigned operator()(unsigned row, unsigned v) const { return row * vectors + v; }
};
// The block copies window `step` (Vectors 16-byte vectors per row) of every row
// into LDS; source(r) is row r as a pointer to Vec, `total` its vector count.
template <unsigned Rows, unsigned Vectors, unsigned Threads, class Vec, class Source, class Layout>
__device__ __forceinline__ void stage_window(Vec* lds, const Source& source, unsigned step, unsigned total,
                                             const Layout& layout) {
    for (unsigned i = threadIdx.x; i < Rows * Vectors; i += Threads) {
        const unsigned r = i / Vectors, v = i % Vectors, at = step * Vectors + v;
        if (at < total)
            lds[layout(r, v)] = source(r)[at];
    }
}
// The barrier protocol of a stepped sweep. stage(step, buffer) fills a window,
// consume(step, buffer) reads it. Two buffers: the next window is staged while
// this one is consumed (one barrier per step). One buffer: half the LDS, two
// barriers per step. Every thread of the block must call this with equal steps.
template <unsigned Buffers, class Stage, class Consume>
__device__ __forceinline__ void stepped(unsigned steps, const Stage& stage, const Consume& consume) {
    static_assert(Buffers == 1U || Buffers == 2U);
    stage(0U, 0U);
    __syncthreads();
    for (unsigned step = 0; step < steps; ++step) {
        if (Buffers == 2U && step + 1U < steps)
            stage(step + 1U, (step + 1U) & 1U);
        consume(step, Buffers == 2U ? step & 1U : 0U);
        if (step + 1U < steps) {
            __syncthreads();
            if (Buffers == 1U) {
                stage(step + 1U, 0U);
                __syncthreads();
            }
        }
    }
}
#endif // HIP device code
} // namespace strix::core::gemv_rows
