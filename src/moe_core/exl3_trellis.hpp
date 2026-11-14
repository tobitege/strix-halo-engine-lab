#pragma once
// minimax-m3 MoE/dense kernels: device building blocks of the strix-trellis 0.3 EXL3 linear (mul1 codebook), copied
// and adapted (not linked) so the fused kernels of src/minimax/moe/ reproduce its bits row by row:
//   input   t = RN16(H32(RN16(RN32(x * suh))))                       (transforms.hpp input_transform)
//   core    per output tile column, per split s: lane chains of v_dot2_f32_f16 over the split's k-tiles
//           (pairs 0,1 -> sum 0, pairs 2,3 -> sum 1), then v += shfl_down(v, 2, 4); v += shfl_down(v, 1, 4)
//   finish  acc = ((0 + p_0) + p_1) + ... + p_{S-1}; y = RN32(H32(acc) * svh)   (output_transform, alpha 1, no bias)
// The split count S is the library's vector policy for <= 8 rows (vector_splits below). Every helper keeps the
// library's operation order; only addressing, scheduling and launch structure differ.
// Compile with -ffp-contract=off (engine flags).
#include <cmath>
#include <cstdint>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "../core/hadamard.hpp"
#include "../core/trellis_codec.hpp"

namespace strix::moe_core {

using core::trellis::Half2;

// strix-trellis split policy for <= 8 rows (gemv_policy.hpp measured_vector_splits; no MiniMax shape is special-cased).
__host__ __device__ constexpr unsigned vector_splits(unsigned input, unsigned output) {
    return output <= 1024 ? 32U : (output <= 4096 && input >= 4096 ? 16U : 4U);
}

// RN32 boundary before a later RN16 conversion: the one core/hadamard.hpp materialize_fp32.
__device__ __forceinline__ float materialize(float v) { return strix::core::materialize_fp32(v); }

// H32 over 128 values held as lane + 32 j: the one core/hadamard.hpp hadamard128 (lane overload).
using strix::core::hadamard128;

__device__ __forceinline__ float half_bits(std::uint16_t h) { return __half2float(__ushort_as_half(h)); }

// One wave: input transform of the 128 channels [begin, begin + 128) of one row.
__device__ __forceinline__ void transform_block(const float* x, const std::uint16_t* suh, _Float16* out, unsigned begin,
                                                unsigned lane) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = begin + lane + 32U * static_cast<unsigned>(j);
        const float scaled = materialize(x[c] * half_bits(suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    hadamard128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) out[begin + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
}
// Same transform on the lane's four x values already in registers (xv[j] = x[begin + lane + 32 j]), so a caller that
// transforms one block for several (suh, out) pairs loads x once. transform_block_at is this helper behind the loads.
__device__ __forceinline__ void transform_block_values(const float (&xv)[4], const std::uint16_t* suh, _Float16* out,
                                                       unsigned begin, unsigned obegin, unsigned lane) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = begin + lane + 32U * static_cast<unsigned>(j);
        const float scaled = materialize(xv[j] * half_bits(suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    hadamard128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) out[obegin + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
}
// Same transform, writing the 128 results at out[obegin ...] (an LDS chunk that starts at channel begin - obegin).
__device__ __forceinline__ void transform_block_at(const float* x, const std::uint16_t* suh, _Float16* out, unsigned begin,
                                                   unsigned obegin, unsigned lane) {
    float xv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) xv[j] = x[begin + lane + 32U * static_cast<unsigned>(j)];
    transform_block_values(xv, suh, out, begin, obegin, lane);
}
// Same transform, returning the four per-lane values in registers (no store).
__device__ __forceinline__ void transform_values(const float* x, const std::uint16_t* suh,
                                                unsigned begin, unsigned lane, float (&v)[4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = begin + lane + 32U * static_cast<unsigned>(j);
        const float scaled = materialize(x[c] * half_bits(suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    hadamard128(v, lane);
}

// ---------------------------------------------------------------- mul1 decode (shared codec)
using core::trellis::hash16;
using core::trellis::mul1_pair;

// Aligned 16x16 tile reader of the shared codec: the lane owns eight consecutive states starting at bit
// ((lane * 8 + 1) * Bits - 16) of the tile's Bits * 8-word ring.
using core::trellis::LaneGeom;
using core::trellis::GlobalU32;
using core::trellis::LocalU32;
using core::trellis::uni;
using core::trellis::uniform;
using core::trellis::load_u32;
using core::trellis::TileStates;

// Activation pairs of one k-tile for one row: channels k * 16 + 2 (lane & 3) (+ 8).
__device__ __forceinline__ void load_act(const _Float16* row, unsigned k, unsigned lane, Half2& a0, Half2& a1) {
    const auto* p = (GlobalU32*)(reinterpret_cast<const std::uint32_t*>(row)) + k * 8U + (lane & 3U);
    a0 = __builtin_bit_cast(Half2, p[0]);
    a1 = __builtin_bit_cast(Half2, p[4]);
}

// One k-tile of the core for up to R rows (rows >= nrows are skipped; their sums are not touched).
template <int Bits, int R>
__device__ __forceinline__ void core_tile(const unsigned char* tile, const LaneGeom<Bits>& g, const _Float16* x,
                                          std::uint32_t xstride, unsigned nrows, unsigned k, unsigned lane,
                                          float (&sums)[R][2]) {
    Half2 act[R][2];
#pragma unroll
    for (int r = 0; r < R; ++r)
        if (static_cast<unsigned>(r) < nrows) load_act(x + static_cast<std::size_t>(r) * xstride, k, lane, act[r][0], act[r][1]);
    const TileStates<Bits> t(tile, g);
    const Half2 w0 = mul1_pair(t.template state<0>(), t.template state<1>());
    const Half2 w1 = mul1_pair(t.template state<2>(), t.template state<3>());
    const Half2 w2 = mul1_pair(t.template state<4>(), t.template state<5>());
    const Half2 w3 = mul1_pair(t.template state<6>(), t.template state<7>());
#pragma unroll
    for (int r = 0; r < R; ++r)
        if (static_cast<unsigned>(r) < nrows) {
            sums[r][0] = __builtin_amdgcn_fdot2(w0, act[r][0], sums[r][0], false);
            sums[r][0] = __builtin_amdgcn_fdot2(w1, act[r][1], sums[r][0], false);
            sums[r][1] = __builtin_amdgcn_fdot2(w2, act[r][0], sums[r][1], false);
            sums[r][1] = __builtin_amdgcn_fdot2(w3, act[r][1], sums[r][1], false);
        }
}

// The 4-lane reduction of one sum; lane (lane & 3) == 0 holds output tile_col * 16 + lane / 4 + h * 8.
__device__ __forceinline__ float reduce4(float v) {
    v += __shfl_down(v, 2, 4);
    v += __shfl_down(v, 1, 4);
    return v;
}

// Prefetched form: the raw words and activation pairs of the next D k-tiles stay in flight in a register ring (the
// activations ride in the same ring because vector-memory counters retire in issue order). Same arithmetic, same bits.
template <int Bits, int R> struct Slot {
    std::uint32_t a, b, c;
    Half2 act[R][2];
};
template <int Bits, int R>
__device__ __forceinline__ void load_slot(Slot<Bits, R>& s, const unsigned char* tile, const LaneGeom<Bits>& g,
                                          const _Float16* x, std::uint32_t xstride, unsigned nrows, unsigned k,
                                          unsigned lane) {
    s.a = load_u32(tile, g.oa);
    s.b = load_u32(tile, g.ob);
    s.c = load_u32(tile, g.oc);
#pragma unroll
    for (int r = 0; r < R; ++r)
        if (static_cast<unsigned>(r) < nrows) load_act(x + static_cast<std::size_t>(r) * xstride, k, lane, s.act[r][0], s.act[r][1]);
}
template <int Bits, int R>
__device__ __forceinline__ void compute_slot(const Slot<Bits, R>& s, const LaneGeom<Bits>& g, unsigned nrows,
                                             float (&sums)[R][2]) {
    TileStates<Bits> t;
    t.high = __builtin_amdgcn_alignbit(s.a, s.b, g.align);
    t.low = __builtin_amdgcn_alignbit(s.b, s.c, g.align);
    t.tail = s.c << (32U - g.align);
    const Half2 w0 = mul1_pair(t.template state<0>(), t.template state<1>());
    const Half2 w1 = mul1_pair(t.template state<2>(), t.template state<3>());
    const Half2 w2 = mul1_pair(t.template state<4>(), t.template state<5>());
    const Half2 w3 = mul1_pair(t.template state<6>(), t.template state<7>());
#pragma unroll
    for (int r = 0; r < R; ++r)
        if (static_cast<unsigned>(r) < nrows) {
            sums[r][0] = __builtin_amdgcn_fdot2(w0, s.act[r][0], sums[r][0], false);
            sums[r][0] = __builtin_amdgcn_fdot2(w1, s.act[r][1], sums[r][0], false);
            sums[r][1] = __builtin_amdgcn_fdot2(w2, s.act[r][0], sums[r][1], false);
            sums[r][1] = __builtin_amdgcn_fdot2(w3, s.act[r][1], sums[r][1], false);
        }
}

// LDS form: a ring of Depth k-tiles of weight words only (3 registers per tile) and the activation pairs read from LDS
// right before use (LDS reads retire on their own counter, so they never wait behind the weight loads in flight).
// xs: LDS rows (stride xs_stride halves) of the transformed input holding channels from kofs * 16 on.
// prime() issues the first Depth tiles' loads; a caller can prime before staging its LDS inputs (with lds_barrier(),
// which waits for LDS operations only) so the weight stream is already in flight during the staging.
template <int Bits, int Depth> struct LdsRing {
    static_assert(Depth >= 1);
    LaneGeom<Bits> g;
    const unsigned char* base;
    std::size_t kstride;
    unsigned kt, splits, s0, k0, k1;
    std::uint32_t ra[Depth], rb[Depth], rc[Depth];
    __device__ __forceinline__ LdsRing(const std::uint32_t* trellis, unsigned output, unsigned col, unsigned kt_,
                                       unsigned splits_, unsigned s0_, unsigned s1_, unsigned lane)
        : g(lane), kt(kt_), splits(splits_), s0(s0_) {
        kstride = static_cast<std::size_t>(output / 16U) * Bits * 32U;
        base = uniform(reinterpret_cast<const unsigned char*>(trellis)) + static_cast<std::size_t>(col) * Bits * 32U;
        k0 = kt * s0_ / splits;
        k1 = kt * s1_ / splits;
#pragma unroll
        for (int j = 0; j < Depth; ++j)
            if (k0 + static_cast<unsigned>(j) < k1) {
                const unsigned char* t = base + (k0 + static_cast<unsigned>(j)) * kstride;
                ra[j] = load_u32(t, g.oa);
                rb[j] = load_u32(t, g.ob);
                rc[j] = load_u32(t, g.oc);
            }
    }
    template <int R>
    __device__ __forceinline__ void run(const _Float16* xs, unsigned xs_stride, unsigned kofs, unsigned nrows, unsigned lane,
                                        float (&acc)[R][2]) {
        unsigned s = s0, send = kt * (s0 + 1U) / splits;
        float sums[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) sums[r][0] = sums[r][1] = 0.0F;
        const auto* xw = (LocalU32*)(reinterpret_cast<const std::uint32_t*>(xs)) + (lane & 3U);
        for (unsigned kb = k0; kb < k1; kb += Depth) {
#pragma unroll
            for (int j = 0; j < Depth; ++j) {
                const unsigned k = kb + static_cast<unsigned>(j);
                if (k < k1) {
                    Slot<Bits, R> cur;
                    cur.a = ra[j];
                    cur.b = rb[j];
                    cur.c = rc[j];
                    if (k + Depth < k1) {
                        const unsigned char* t = base + (k + Depth) * kstride;
                        ra[j] = load_u32(t, g.oa);
                        rb[j] = load_u32(t, g.ob);
                        rc[j] = load_u32(t, g.oc);
                    }
#pragma unroll
                    for (int r = 0; r < R; ++r)
                        if (static_cast<unsigned>(r) < nrows) {
                            const auto* p = xw + (static_cast<unsigned>(r) * xs_stride) / 2U + (k - kofs) * 8U;
                            cur.act[r][0] = __builtin_bit_cast(Half2, p[0]);
                            cur.act[r][1] = __builtin_bit_cast(Half2, p[4]);
                        }
                    compute_slot<Bits, R>(cur, g, nrows, sums);
                    if (k + 1U == send) {
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            if (static_cast<unsigned>(r) < nrows) {
                                acc[r][0] = acc[r][0] + reduce4(sums[r][0]);
                                acc[r][1] = acc[r][1] + reduce4(sums[r][1]);
                            }
                            sums[r][0] = sums[r][1] = 0.0F;
                        }
                        ++s;
                        send = kt * (s + 1U) / splits;
                    }
                }
            }
        }
    }
};

// Optimization: LdsRing for the MoE expert kernels (split4_core) with the wave-uniform k range / stride / tile
// base forced into scalar registers and the K3 two-word reader (gu4p2: moe n1 -3 %, n2 -7 %, n4 -8 %, exact). The linear
// cores keep LdsRing unchanged (with this addressing they measured +1.5..3 % at N1 in gu4p2).
template <int Bits, int Depth> struct LdsRingU {
    static_assert(Depth >= 1);
    LaneGeom<Bits> g;
    const unsigned char* base;
    std::size_t kstride;
    unsigned kt, splits, s0, k0, k1;
    std::uint32_t ra[Depth], rb[Depth], rc[Depth];
    // optimization: K3 never reads the third word. Lane shift = ((lane * 8 + 1) * 3 + 768 - 16) & 31 is 19 / 11 / 3 / 27 (<= 27),
    // and the 8 states span offsets 0..21, so a state reads at most low >> 27 = the top 5 bits of low, which come from
    // b << shift alone; tail is never read. With c = 0 compute_slot builds the same high/low (same bits); one load less.
    static constexpr bool kNeedC = Bits != 3;
    static __device__ __forceinline__ unsigned u(unsigned v) { return uni(v); }
    __device__ __forceinline__ LdsRingU(const std::uint32_t* trellis, unsigned output, unsigned col, unsigned kt_,
                                       unsigned splits_, unsigned s0_, unsigned s1_, unsigned lane)
        : g(lane), kt(kt_), splits(splits_), s0(s0_) {
        // Optimization: the k range, row stride and tile base are wave-uniform by construction; force them
        // into scalar registers so the k loop, its bounds checks and the tile addresses stay on the SALU (saddr loads)
        // instead of per-lane 64-bit VALU adds and exec-mask branches. Addressing only: the same words, same order.
        kstride = static_cast<std::size_t>(u(output / 16U)) * Bits * 32U;  // 64-bit product as before
        base = uniform(reinterpret_cast<const unsigned char*>(trellis) + static_cast<std::size_t>(col) * Bits * 32U);
        k0 = u(kt * s0_ / splits);
        k1 = u(kt * s1_ / splits);
#pragma unroll
        for (int j = 0; j < Depth; ++j)
            if (k0 + static_cast<unsigned>(j) < k1) {
                const unsigned char* t = base + (k0 + static_cast<unsigned>(j)) * kstride;
                ra[j] = load_u32(t, g.oa);
                rb[j] = load_u32(t, g.ob);
                rc[j] = kNeedC ? load_u32(t, g.oc) : 0U;
            }
    }
    template <int R>
    __device__ __forceinline__ void run(const _Float16* xs, unsigned xs_stride, unsigned kofs, unsigned nrows, unsigned lane,
                                        float (&acc)[R][2]) {
        unsigned s = u(s0), send = u(kt * (s0 + 1U) / splits);
        float sums[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) sums[r][0] = sums[r][1] = 0.0F;
        const auto* xw = (LocalU32*)(reinterpret_cast<const std::uint32_t*>(xs)) + (lane & 3U);
        for (unsigned kb = u(k0); kb < k1; kb = u(kb + static_cast<unsigned>(Depth))) {
#pragma unroll
            for (int j = 0; j < Depth; ++j) {
                const unsigned k = kb + static_cast<unsigned>(j);
                if (k < k1) {
                    Slot<Bits, R> cur;
                    cur.a = ra[j];
                    cur.b = rb[j];
                    cur.c = rc[j];
                    if (k + Depth < k1) {
                        const unsigned char* t = base + (k + Depth) * kstride;
                        ra[j] = load_u32(t, g.oa);
                        rb[j] = load_u32(t, g.ob);
                        rc[j] = kNeedC ? load_u32(t, g.oc) : 0U;
                    }
#pragma unroll
                    for (int r = 0; r < R; ++r)
                        if (static_cast<unsigned>(r) < nrows) {
                            const auto* p = xw + (static_cast<unsigned>(r) * xs_stride) / 2U + (k - kofs) * 8U;
                            cur.act[r][0] = __builtin_bit_cast(Half2, p[0]);
                            cur.act[r][1] = __builtin_bit_cast(Half2, p[4]);
                        }
                    compute_slot<Bits, R>(cur, g, nrows, sums);
                    if (k + 1U == send) {
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            if (static_cast<unsigned>(r) < nrows) {
                                acc[r][0] = acc[r][0] + reduce4(sums[r][0]);
                                acc[r][1] = acc[r][1] + reduce4(sums[r][1]);
                            }
                            sums[r][0] = sums[r][1] = 0.0F;
                        }
                        s = u(s + 1U);
                        send = u(kt * (s + 1U) / splits);
                    }
                }
            }
        }
    }
};

// Workgroup barrier that waits for this wave's LDS operations only (outstanding global loads stay in flight).
__device__ __forceinline__ void lds_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }

template <int Bits, int R, int Depth>
__device__ __forceinline__ void stream_column_lds(const std::uint32_t* trellis, unsigned output, unsigned col, unsigned kt,
                                                  unsigned splits, unsigned s0, unsigned s1, const _Float16* xs,
                                                  unsigned xs_stride, unsigned kofs, unsigned nrows, unsigned lane,
                                                  float (&acc)[R][2]) {
    LdsRing<Bits, Depth> ring(trellis, output, col, kt, splits, s0, s1, lane);
    ring.template run<R>(xs, xs_stride, kofs, nrows, lane, acc);
}

// optimization: stream_column_lds through LdsRingU (MoE expert kernels only).
template <int Bits, int R, int Depth>
__device__ __forceinline__ void stream_column_lds_u(const std::uint32_t* trellis, unsigned output, unsigned col, unsigned kt,
                                                    unsigned splits, unsigned s0, unsigned s1, const _Float16* xs,
                                                    unsigned xs_stride, unsigned kofs, unsigned nrows, unsigned lane,
                                                    float (&acc)[R][2]) {
    LdsRingU<Bits, Depth> ring(trellis, output, col, kt, splits, s0, s1, lane);
    ring.template run<R>(xs, xs_stride, kofs, nrows, lane, acc);
}

// One wave streams tile column `col` over splits [s0, s1) of a matrix with kt = input / 16 k-tiles, accumulating the
// reduced split partials in ascending split order into acc (lanes with (lane & 3) == 0 are meaningful).
// tiles: base of the trellis words; row stride of k-tiles = (output / 16) * Bits * 32 bytes.
// Depth 0: plain loop (compiler-scheduled, unroll 4); depth D > 0: explicit ring of D k-tiles in flight across the split
// boundaries of the wave's range.
template <int Bits, int R, int Depth = 0>
__device__ __forceinline__ void stream_column(const std::uint32_t* trellis, unsigned output, unsigned col, unsigned kt,
                                              unsigned splits, unsigned s0, unsigned s1, const _Float16* x,
                                              std::uint32_t xstride, unsigned nrows, unsigned lane, float (&acc)[R][2]) {
    const LaneGeom<Bits> g(lane);
    const std::size_t kstride = static_cast<std::size_t>(output / 16U) * Bits * 32U;
    const auto* base = uniform(reinterpret_cast<const unsigned char*>(trellis)) + static_cast<std::size_t>(col) * Bits * 32U;
    if constexpr (Depth > 0) {
        const unsigned k0 = kt * s0 / splits, k1 = kt * s1 / splits;
        Slot<Bits, R> ring[Depth];
#pragma unroll
        for (int j = 0; j < Depth; ++j)
            if (k0 + static_cast<unsigned>(j) < k1)
                load_slot<Bits, R>(ring[j], base + (k0 + static_cast<unsigned>(j)) * kstride, g, x, xstride, nrows,
                                   k0 + static_cast<unsigned>(j), lane);
        unsigned s = s0, send = kt * (s0 + 1U) / splits;
        float sums[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) sums[r][0] = sums[r][1] = 0.0F;
        for (unsigned kb = k0; kb < k1; kb += Depth) {
#pragma unroll
            for (int j = 0; j < Depth; ++j) {
                const unsigned k = kb + static_cast<unsigned>(j);
                if (k < k1) {
                const Slot<Bits, R> cur = ring[j];
                if (k + Depth < k1)
                    load_slot<Bits, R>(ring[j], base + (k + Depth) * kstride, g, x, xstride, nrows, k + Depth, lane);
                compute_slot<Bits, R>(cur, g, nrows, sums);
                if (k + 1U == send) {
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        if (static_cast<unsigned>(r) < nrows) {
                            acc[r][0] = acc[r][0] + reduce4(sums[r][0]);
                            acc[r][1] = acc[r][1] + reduce4(sums[r][1]);
                        }
                        sums[r][0] = sums[r][1] = 0.0F;
                    }
                    ++s;
                    send = kt * (s + 1U) / splits;
                }
                }
            }
        }
    } else {
    for (unsigned s = s0; s < s1; ++s) {
        const unsigned kb = kt * s / splits, ke = kt * (s + 1U) / splits;
        float sums[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) sums[r][0] = sums[r][1] = 0.0F;
        for (unsigned k = kb; k < ke; ++k) core_tile<Bits, R>(base + k * kstride, g, x, xstride, nrows, k, lane, sums);
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (static_cast<unsigned>(r) < nrows) {
                acc[r][0] = acc[r][0] + reduce4(sums[r][0]);
                acc[r][1] = acc[r][1] + reduce4(sums[r][1]);
            }
    }
    }
}


// EXL3 mul1 weight decode: the shared codec (core/trellis_codec.hpp) plus this library's input transform.
// Input/output transforms retain explicit RN32/RN16 boundaries in the device helpers.
template <int Bits> using Window = core::trellis::Window<Bits>;
template <int Bits> using Norm = core::trellis::Norm<Bits>;
using Mul1 = core::trellis::Mul1;
template <int Bits> using Mul1PairDecode = core::trellis::PairDecode8<Bits, core::trellis::Mul1>;

template <int Bits> struct Exl3Trellis : core::trellis::Codec<Bits, core::trellis::Mul1> {
    using InputTransform = Exl3Trellis<Bits>;
    static __host__ __device__ __forceinline__ float codebook_value(std::uint32_t state, Mul1 = {}) { return Mul1::value(state); }
    static __device__ __forceinline__ void input_block(const float* x, const std::uint16_t* suh, _Float16* out,
                                                      unsigned begin, unsigned lane) {
        transform_block(x, suh, out, begin, lane);
    }
    static __device__ __forceinline__ void h128(float (&v)[4], unsigned lane) { hadamard128(v, lane); }
    static __device__ __forceinline__ float scale(float x, std::uint16_t sign) {
        return materialize(x * half_bits(sign));
    }
};

// Scalar-default policy: preserve the previous codebook conversion and order.
template <int Bits, template <int> class WeightDecode = Exl3Trellis> struct ScalarDecode {
    static constexpr bool kPacked = false;
    static __host__ __device__ __forceinline__ void decode8(const std::uint32_t (&st)[8], Half2 (&out)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out[i] = Half2{static_cast<_Float16>(WeightDecode<Bits>::codebook_value(st[2 * i])),
                           static_cast<_Float16>(WeightDecode<Bits>::codebook_value(st[2 * i + 1]))};
        }
    }
};

}  // namespace strix::moe_core
