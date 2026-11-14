// Multi-row form of the Q8_0 x A8 decode sites (hip_ds41_rn_wide.hpp, STRIX_DS41_RN_WIDE=1). Per (row, output)
// the arithmetic is the one of q8_0_a8_kernel in hip_ds41_layer.hip: lane l of a 32-lane wave takes the groups
// l, l + 32, ... of the output's row in ascending order, acc = acc + float(sudot4 dot) * (scale * delta), then the
// butterfly v + shfl_xor(v, d), d = 16 ... 1. Only which wave owns which output, when the weights are loaded and
// where the activations are read from change.

#include "strix/env_flags.hpp"
#include "strix/hip_check.hpp"
#include "strix/hip_ds41_rn_wide.hpp"
#include "strix/dist/staged_ready.hpp"

#include "strix/ds41_formats.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::ds41 {
namespace {

using strix::hip_check;
void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

constexpr unsigned wave = 32;
constexpr unsigned rn_max_rows = 8;

__device__ inline float rn_butterfly(float v) {
    for (int d = 16; d; d >>= 1) v = v + __shfl_xor(v, d, wave);
    return v;
}

// Raw bytes of one 32-weight group (loaded apart from the decoding, so the next group of every output stays in
// flight in its compact form). Layouts as hip_ds41_layer.hip's load_weights: Q8_0 {f16 d, 32 int8}, CB4 {d, 16
// bytes of nibbles} into v.table, U5 {d, 16 bytes of nibbles, 4 bytes of high bits}, U6 {d, nibbles, 8 bytes}.
template <unsigned P> struct RnRaw;
template <> struct RnRaw<0> { unsigned short d; unsigned q[8]; };
template <> struct RnRaw<4> { unsigned short d; unsigned n[4]; };
template <> struct RnRaw<5> { unsigned short d; unsigned n[4]; unsigned h; };
template <> struct RnRaw<6> { unsigned short d; unsigned n[4]; unsigned h[2]; };
template <unsigned P> inline constexpr unsigned rn_block_bytes = P == 4U ? 18U : P == 5U ? 22U : P == 6U ? 26U : 34U;

template <unsigned P>
__device__ __forceinline__ RnRaw<P> rn_load(const std::uint8_t* __restrict__ row, unsigned g) {
    RnRaw<P> raw;
    const std::uint8_t* block = row + g * rn_block_bytes<P>;
    __builtin_memcpy(&raw.d, block, 2);
    if constexpr (P == 0U) {
        __builtin_memcpy(raw.q, block + 2, 32);
    } else {
        __builtin_memcpy(raw.n, block + 2, 16);
        if constexpr (P == 5U) __builtin_memcpy(&raw.h, block + 18, 4);
        if constexpr (P == 6U) __builtin_memcpy(raw.h, block + 18, 8);
    }
    return raw;
}
__device__ __forceinline__ unsigned rn_cb4(unsigned idx, const unsigned* t) {
    const unsigned sel = idx & 0x07070707U;
    const unsigned lo = __builtin_amdgcn_perm(t[1], t[0], sel), hi = __builtin_amdgcn_perm(t[3], t[2], sel);
    const unsigned high = ((idx >> 3U) & 0x01010101U) * 0xffU;
    return (hi & high) | (lo & ~high);
}
__device__ __forceinline__ unsigned rn_u5(unsigned nib, unsigned h4) {
    const unsigned v = nib | (((h4 & 1U) | ((h4 & 2U) << 7U) | ((h4 & 4U) << 14U) | ((h4 & 8U) << 21U)) << 4U);
    const unsigned b = v ^ 0x10101010U;
    return b | ((b & 0x10101010U) * 14U);
}
__device__ __forceinline__ unsigned rn_u6(unsigned nib, unsigned h8) {
    const unsigned v = nib | (((h8 & 3U) | ((h8 & 0xcU) << 6U) | ((h8 & 0x30U) << 12U) | ((h8 & 0xc0U) << 18U)) << 4U);
    const unsigned b = v ^ 0x20202020U;
    return b | ((b & 0x20202020U) * 6U);
}
// The eight words of int8 codes of the equivalent Q8_0 block.
template <unsigned P>
__device__ __forceinline__ void rn_decode(const RnRaw<P>& raw, const unsigned* table, int (&q)[8]) {
    if constexpr (P == 0U) {
#pragma unroll
        for (unsigned k = 0; k < 8; ++k) q[k] = static_cast<int>(raw.q[k]);
    } else {
#pragma unroll
        for (unsigned k = 0; k < 4; ++k) {
            const unsigned lo = raw.n[k] & 0x0f0f0f0fU, hi = (raw.n[k] >> 4U) & 0x0f0f0f0fU;
            if constexpr (P == 4U) {
                q[k] = static_cast<int>(rn_cb4(lo, table));
                q[4 + k] = static_cast<int>(rn_cb4(hi, table));
            } else if constexpr (P == 5U) {
                q[k] = static_cast<int>(rn_u5(lo, (raw.h >> (4U * k)) & 0xfU));
                q[4 + k] = static_cast<int>(rn_u5(hi, (raw.h >> (16U + 4U * k)) & 0xfU));
            } else {
                q[k] = static_cast<int>(rn_u6(lo, (raw.h[0] >> (8U * k)) & 0xffU));
                q[4 + k] = static_cast<int>(rn_u6(hi, (raw.h[1] >> (8U * k)) & 0xffU));
            }
        }
    }
}
__device__ __forceinline__ int rn_dot32(const int (&q)[8], const int4& x0, const int4& x1) {
    int d = 0;
    d = __builtin_amdgcn_sudot4(true, q[0], true, x0.x, d, false);
    d = __builtin_amdgcn_sudot4(true, q[1], true, x0.y, d, false);
    d = __builtin_amdgcn_sudot4(true, q[2], true, x0.z, d, false);
    d = __builtin_amdgcn_sudot4(true, q[3], true, x0.w, d, false);
    d = __builtin_amdgcn_sudot4(true, q[4], true, x1.x, d, false);
    d = __builtin_amdgcn_sudot4(true, q[5], true, x1.y, d, false);
    d = __builtin_amdgcn_sudot4(true, q[6], true, x1.z, d, false);
    d = __builtin_amdgcn_sudot4(true, q[7], true, x1.w, d, false);
    return d;
}

// R rows (exact), W outputs per wave, Waves waves per block (Waves * W outputs, all in one group), P the
// weight format, Lds: the block's R activation rows staged in LDS (else read in place).
template <unsigned R, unsigned W, unsigned Waves, unsigned P, bool Lds>
__device__ __forceinline__ void rn_body(Q8A8View v, const std::int8_t* __restrict__ codes,
                                                          const float* __restrict__ deltas, float* __restrict__ out) {
    extern __shared__ int4 rn_lds[];
    const unsigned w_id = threadIdx.x / wave, l = threadIdx.x % wave;
    const unsigned block_o0 = blockIdx.x * (Waves * W), o0 = block_o0 + w_id * W;
    constexpr unsigned block_bytes = rn_block_bytes<P>;
    const unsigned groups = v.width / 32U, row_bytes = (v.row_width ? v.row_width : v.width) / 32U * block_bytes;
    const unsigned grp = block_o0 / v.per_group;  // the whole block is in this group
    const signed char* xg = reinterpret_cast<const signed char*>(codes) + static_cast<std::size_t>(grp) * v.width;
    const float* dg = deltas + static_cast<std::size_t>(grp) * groups;
    const std::uint8_t* wbase = v.weight + v.k_offset / 32U * block_bytes;
    const unsigned* table = v.table;
    const std::uint8_t* wrow[W];
#pragma unroll
    for (unsigned w = 0; w < W; ++w) wrow[w] = wbase + static_cast<std::size_t>(v.first + min(o0 + w, v.count - 1U)) * row_bytes;
    // Early: the first group of every output's row before the activations are staged.
    RnRaw<P> current[W];
    if (l < groups)
#pragma unroll
        for (unsigned w = 0; w < W; ++w) current[w] = rn_load<P>(wrow[w], l);
    auto* xs = reinterpret_cast<signed char*>(rn_lds);
    auto* ds = reinterpret_cast<float*>(xs + static_cast<std::size_t>(R) * v.width);
    if constexpr (Lds) {
        const unsigned words = v.width / 16U;
        for (unsigned i = threadIdx.x; i < R * words; i += wave * Waves) {
            const unsigned r = i / words, q = i % words;
            reinterpret_cast<int4*>(xs)[i] = reinterpret_cast<const int4*>(xg + static_cast<std::size_t>(r) * v.x_stride)[q];
        }
        for (unsigned i = threadIdx.x; i < R * groups; i += wave * Waves) {
            const unsigned r = i / groups, q = i % groups;
            ds[i] = dg[static_cast<std::size_t>(r) * (v.x_stride / 32U) + q];
        }
        __syncthreads();
    }
    if (o0 >= v.count) return;
    float acc[W][R];
#pragma unroll
    for (unsigned w = 0; w < W; ++w)
#pragma unroll
        for (unsigned r = 0; r < R; ++r) acc[w][r] = 0.0F;
    for (unsigned g = l; g < groups; g += wave) {
        const bool more = g + wave < groups;
        RnRaw<P> next[W];
        if (more)
#pragma unroll
            for (unsigned w = 0; w < W; ++w) next[w] = rn_load<P>(wrow[w], g + wave);
        int q[W][8];
        float scale[W];
#pragma unroll
        for (unsigned w = 0; w < W; ++w) {
            rn_decode<P>(current[w], table, q[w]);
            scale[w] = f16_to_float(current[w].d);
        }
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            int4 x0, x1;
            float delta;
            if constexpr (Lds) {
                const auto* x = reinterpret_cast<const int4*>(xs + static_cast<std::size_t>(r) * v.width + 32U * g);
                x0 = x[0]; x1 = x[1];
                delta = ds[r * groups + g];
            } else {
                const auto* x = reinterpret_cast<const int4*>(xg + static_cast<std::size_t>(r) * v.x_stride + 32U * g);
                x0 = x[0]; x1 = x[1];
                delta = dg[static_cast<std::size_t>(r) * (v.x_stride / 32U) + g];
            }
#pragma unroll
            for (unsigned w = 0; w < W; ++w) acc[w][r] = acc[w][r] + static_cast<float>(rn_dot32(q[w], x0, x1)) * (scale[w] * delta);
        }
        if (more)
#pragma unroll
            for (unsigned w = 0; w < W; ++w) current[w] = next[w];
    }
#pragma unroll
    for (unsigned w = 0; w < W; ++w)
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            const float total = rn_butterfly(acc[w][r]);
            if (l == (w * R + r) % wave && o0 + w < v.count) {
                float* y = out + static_cast<std::size_t>(r) * v.out_stride + o0 + w;
                *y = v.accumulate ? *y + total : total;
            }
        }
}

template <unsigned R, unsigned W, unsigned Waves, unsigned P, bool Lds>
__global__ __launch_bounds__(wave * Waves) void rn_kernel(Q8A8View v, const std::int8_t* __restrict__ codes,
                                                        const float* __restrict__ deltas, float* __restrict__ out) {
    rn_body<R, W, Waves, P, Lds>(v, codes, deltas, out);
}
template <unsigned R, unsigned P>
__global__ __launch_bounds__(wave * 16) void rn_ready_kernel(Q8A8View v, dist::ProducerStage stage) {
    // Production o_b uses W=1, Waves=16, LDS. Keep that exact body and grid.
    rn_body<R, 1, 16, P, true>(v, v.codes, v.deltas, v.out);
    dist::publish_staged_view(stage, gridDim.x);
}

// Flat form for widths whose group count is not a multiple of 32 (attn_q_b: 1,280 values = 40 groups, so lanes 8-31
// idle in a wave's second pass): a wave owns W outputs and spreads their W * groups (output, group) items over all
// 32 lanes (item i = pass * 32 + lane: output i / groups, group i % groups). Each item's terms float(dot) * (scale *
// delta) go to LDS; then lane l of output j sums the terms of groups l, l + 32, ... ascending from 0.0 as the declared
// order does, and the butterfly follows: the same operations on the same values, so the same bits.
template <unsigned R, unsigned W, unsigned Waves, unsigned P>
__global__ __launch_bounds__(wave * Waves) void rn_flat_kernel(Q8A8View v, const std::int8_t* __restrict__ codes,
                                                               const float* __restrict__ deltas, float* __restrict__ out) {
    extern __shared__ int4 rn_lds[];
    const unsigned w_id = threadIdx.x / wave, l = threadIdx.x % wave;
    const unsigned o0 = (blockIdx.x * Waves + w_id) * W;
    constexpr unsigned block_bytes = rn_block_bytes<P>;
    const unsigned groups = v.width / 32U, row_bytes = (v.row_width ? v.row_width : v.width) / 32U * block_bytes;
    const unsigned grp = (blockIdx.x * Waves * W) / v.per_group;
    const signed char* xg = reinterpret_cast<const signed char*>(codes) + static_cast<std::size_t>(grp) * v.width;
    const float* dg = deltas + static_cast<std::size_t>(grp) * groups;
    const std::uint8_t* wbase = v.weight + v.k_offset / 32U * block_bytes;
    const unsigned* table = v.table;
    float* terms = reinterpret_cast<float*>(rn_lds) + static_cast<std::size_t>(w_id) * W * R * groups;  // [W][R][groups]
    const bool active = o0 < v.count;
    const unsigned items = active ? W * groups : 0U;
    const auto row_of = [&](unsigned i) {
        const unsigned o = o0 + i / groups;
        return wbase + static_cast<std::size_t>(v.first + (o < v.count ? o : v.count - 1U)) * row_bytes;
    };
    RnRaw<P> current;
    if (l < items) current = rn_load<P>(row_of(l), l % groups);
    for (unsigned i = l; i < items; i += wave) {
        const bool more = i + wave < items;
        RnRaw<P> next;
        if (more) next = rn_load<P>(row_of(i + wave), (i + wave) % groups);
        const unsigned j = i / groups, g = i % groups;
        int q[8];
        rn_decode<P>(current, table, q);
        const float scale = f16_to_float(current.d);
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            const auto* x = reinterpret_cast<const int4*>(xg + static_cast<std::size_t>(r) * v.x_stride + 32U * g);
            const float delta = dg[static_cast<std::size_t>(r) * (v.x_stride / 32U) + g];
            terms[(j * R + r) * groups + g] = static_cast<float>(rn_dot32(q, x[0], x[1])) * (scale * delta);
        }
        if (more) current = next;
    }
    __syncthreads();
    if (!active) return;
#pragma unroll
    for (unsigned j = 0; j < W; ++j)
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            float acc = 0.0F;
            for (unsigned g = l; g < groups; g += wave) acc = acc + terms[(j * R + r) * groups + g];
            const float total = rn_butterfly(acc);
            if (l == (j * R + r) % wave && o0 + j < v.count) {
                float* y = out + static_cast<std::size_t>(r) * v.out_stride + o0 + j;
                *y = v.accumulate ? *y + total : total;
            }
        }
}

// Form 2: the flat form with wave-level synchronisation. Each wave writes and reads only its own [W][R][groups] terms,
// so the block barrier is replaced by a wave barrier between wavefront-scope release / acquire fences (LDS operations
// of one wave execute in order on RDNA; the fences keep the compiler from moving the term loads above the stores).
// Same items, same terms, same order of the additions as rn_flat_kernel: the same bits.
template <unsigned R, unsigned W, unsigned Waves, unsigned P>
__global__ __launch_bounds__(wave * Waves) void rn_flatw_kernel(Q8A8View v, const std::int8_t* __restrict__ codes,
                                                                const float* __restrict__ deltas, float* __restrict__ out) {
    extern __shared__ int4 rn_lds[];
    const unsigned w_id = threadIdx.x / wave, l = threadIdx.x % wave;
    const unsigned o0 = (blockIdx.x * Waves + w_id) * W;
    if (o0 >= v.count) return;  // nothing shared with the other waves
    constexpr unsigned block_bytes = rn_block_bytes<P>;
    const unsigned groups = v.width / 32U, row_bytes = (v.row_width ? v.row_width : v.width) / 32U * block_bytes;
    const unsigned grp = (blockIdx.x * Waves * W) / v.per_group;
    const signed char* xg = reinterpret_cast<const signed char*>(codes) + static_cast<std::size_t>(grp) * v.width;
    const float* dg = deltas + static_cast<std::size_t>(grp) * groups;
    const std::uint8_t* wbase = v.weight + v.k_offset / 32U * block_bytes;
    const unsigned* table = v.table;
    float* terms = reinterpret_cast<float*>(rn_lds) + static_cast<std::size_t>(w_id) * W * R * groups;  // [W][R][groups]
    const unsigned items = W * groups;
    const auto row_of = [&](unsigned i) {
        const unsigned o = o0 + i / groups;
        return wbase + static_cast<std::size_t>(v.first + (o < v.count ? o : v.count - 1U)) * row_bytes;
    };
    RnRaw<P> current;
    if (l < items) current = rn_load<P>(row_of(l), l % groups);
    for (unsigned i = l; i < items; i += wave) {
        const bool more = i + wave < items;
        RnRaw<P> next;
        if (more) next = rn_load<P>(row_of(i + wave), (i + wave) % groups);
        const unsigned j = i / groups, g = i % groups;
        int q[8];
        rn_decode<P>(current, table, q);
        const float scale = f16_to_float(current.d);
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            const auto* x = reinterpret_cast<const int4*>(xg + static_cast<std::size_t>(r) * v.x_stride + 32U * g);
            const float delta = dg[static_cast<std::size_t>(r) * (v.x_stride / 32U) + g];
            terms[(j * R + r) * groups + g] = static_cast<float>(rn_dot32(q, x[0], x[1])) * (scale * delta);
        }
        if (more) current = next;
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
    for (unsigned j = 0; j < W; ++j)
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            float acc = 0.0F;
            for (unsigned g = l; g < groups; g += wave) acc = acc + terms[(j * R + r) * groups + g];
            const float total = rn_butterfly(acc);
            if (l == (j * R + r) % wave && o0 + j < v.count) {
                float* y = out + static_cast<std::size_t>(r) * v.out_stride + o0 + j;
                *y = v.accumulate ? *y + total : total;
            }
        }
}

// Segment forms (form 3: L = 8 lanes per output, form 4: L = 4) for G-group widths (attn_q_b: G = 40). The declared
// sum of one (row, output) is a fixed tree: leaf d (declared lane d = 0..31) = ((0.0 + term(d)) + term(d + 32)) over
// the groups d, d + 32 below G, ascending; then the butterfly levels 16, 8, 4, 2, 1 add the values of lanes d and
// d ^ level. Here a segment of L lanes owns an output: lane m holds the leaves of the declared lanes m + k L
// (k = 0 .. 32 / L - 1) and folds the levels 16 .. L in registers (the pairs (k, k + 16 / L), then (k, k + 8 / L) ...:
// the butterfly's pairs, IEEE addition being commutative); the levels L / 2 .. 1 run across the segment's lanes with
// DPP row_xmask moves (exact). The same operations on the same values, so the same bits as the 1-row kernel. Every
// lane is busy (40 groups = 8 lanes x 5), no term goes through LDS, and each (row, output) pays log2(L) cross-lane
// levels instead of 5. Every weight group of the lane (G / L of them per output) is issued at the top of the kernel.
// A wave owns 32 / L segments x Wl outputs per lane (Wl consecutive outputs per segment).
// rn_decode with the sign extension of the U6 codes written as shifts: s = the sign bits (bit 5 of each byte),
// s * 6 = (s << 1) | (s << 2), the bits never overlap. The same codes; used by the segment forms only.
__device__ __forceinline__ unsigned rn_u6_shift(unsigned nib, unsigned h8) {
    const unsigned v = nib | (((h8 & 3U) | ((h8 & 0xcU) << 6U) | ((h8 & 0x30U) << 12U) | ((h8 & 0xc0U) << 18U)) << 4U);
    const unsigned b = v ^ 0x20202020U, s = b & 0x20202020U;
    return b | (s << 1U) | (s << 2U);
}
template <unsigned P>
__device__ __forceinline__ void rn_decode_shift(const RnRaw<P>& raw, const unsigned* table, int (&q)[8]) {
    if constexpr (P == 6U) {
#pragma unroll
        for (unsigned k = 0; k < 4; ++k) {
            const unsigned lo = raw.n[k] & 0x0f0f0f0fU, hi = (raw.n[k] >> 4U) & 0x0f0f0f0fU;
            q[k] = static_cast<int>(rn_u6_shift(lo, (raw.h[0] >> (8U * k)) & 0xffU));
            q[4 + k] = static_cast<int>(rn_u6_shift(hi, (raw.h[1] >> (8U * k)) & 0xffU));
        }
    } else {
        rn_decode<P>(raw, table, q);
    }
}
constexpr unsigned rn_seg_groups = 40;  // the registered G: attn_q_b's 1,280 values per row
__device__ __forceinline__ float rn_xmask(float v, unsigned d) {
    const int s = __builtin_bit_cast(int, v);
    int p = 0;
    if (d == 1U) p = __builtin_amdgcn_mov_dpp(s, 0x161, 0xf, 0xf, true);
    else if (d == 2U) p = __builtin_amdgcn_mov_dpp(s, 0x162, 0xf, 0xf, true);
    else if (d == 4U) p = __builtin_amdgcn_mov_dpp(s, 0x164, 0xf, 0xf, true);
    else p = __builtin_amdgcn_mov_dpp(s, 0x168, 0xf, 0xf, true);
    return __builtin_bit_cast(float, p);
}
template <unsigned K, unsigned Start, unsigned Stride, unsigned Wl, unsigned R, class Leaf>
__device__ __forceinline__ void rn_seg_tree(const Leaf& leaf, float (&v)[Wl][R]) {
    if constexpr (Stride == K) {
        leaf.template operator()<Start>(v);
    } else {
        float a[Wl][R], b[Wl][R];
        rn_seg_tree<K, Start, Stride * 2U, Wl, R>(leaf, a);
        rn_seg_tree<K, Start + Stride, Stride * 2U, Wl, R>(leaf, b);
#pragma unroll
        for (unsigned w = 0; w < Wl; ++w)
#pragma unroll
            for (unsigned r = 0; r < R; ++r) v[w][r] = a[w][r] + b[w][r];
    }
}
template <unsigned R, unsigned Wl, unsigned L, unsigned Waves, unsigned P, unsigned G, bool Lds>
__global__ __launch_bounds__(wave * Waves) void rn_seg_kernel(Q8A8View v, const std::int8_t* __restrict__ codes,
                                                              const float* __restrict__ deltas, float* __restrict__ out) {
    constexpr unsigned K = wave / L, S = wave / L, J = (G + 31U) / 32U;
    static_assert(G % L == 0U && G <= 64U, "segment form: every lane takes the same number of groups");
    extern __shared__ int4 rn_lds[];
    const unsigned w_id = threadIdx.x / wave, l = threadIdx.x % wave, seg = l / L, m = l % L;
    const unsigned block_o0 = blockIdx.x * (Waves * S * Wl), o0 = block_o0 + (w_id * S + seg) * Wl;
    constexpr unsigned block_bytes = rn_block_bytes<P>;
    const unsigned row_bytes = (v.row_width ? v.row_width : v.width) / 32U * block_bytes;
    const unsigned grp = block_o0 / v.per_group;
    const signed char* xg = reinterpret_cast<const signed char*>(codes) + static_cast<std::size_t>(grp) * v.width;
    const float* dg = deltas + static_cast<std::size_t>(grp) * G;
    const std::uint8_t* wbase = v.weight + v.k_offset / 32U * block_bytes;
    const unsigned* table = v.table;
    const std::uint8_t* wrow[Wl];
#pragma unroll
    for (unsigned w = 0; w < Wl; ++w) wrow[w] = wbase + static_cast<std::size_t>(v.first + min(o0 + w, v.count - 1U)) * row_bytes;
    // Every group of the lane first: slot (k, j) is group m + k L + 32 j (valid for every lane or for none).
    RnRaw<P> raw[K][J][Wl];
#pragma unroll
    for (unsigned j = 0; j < J; ++j)
#pragma unroll
        for (unsigned k = 0; k < K; ++k)
            if (k * L + 32U * j < G)
#pragma unroll
                for (unsigned w = 0; w < Wl; ++w) raw[k][j][w] = rn_load<P>(wrow[w], m + k * L + 32U * j);
    auto* xs = reinterpret_cast<signed char*>(rn_lds);
    auto* ds = reinterpret_cast<float*>(xs + static_cast<std::size_t>(R) * G * 32U);
    if constexpr (Lds) {
        constexpr unsigned words = G * 2U;
        for (unsigned i = threadIdx.x; i < R * words; i += wave * Waves) {
            const unsigned r = i / words, q = i % words;
            reinterpret_cast<int4*>(xs)[i] = reinterpret_cast<const int4*>(xg + static_cast<std::size_t>(r) * v.x_stride)[q];
        }
        for (unsigned i = threadIdx.x; i < R * G; i += wave * Waves) {
            const unsigned r = i / G, q = i % G;
            ds[i] = dg[static_cast<std::size_t>(r) * (v.x_stride / 32U) + q];
        }
        __syncthreads();
    }
    const auto leaf = [&]<unsigned k>(float (&acc)[Wl][R]) {
#pragma unroll
        for (unsigned w = 0; w < Wl; ++w)
#pragma unroll
            for (unsigned r = 0; r < R; ++r) acc[w][r] = 0.0F;
#pragma unroll
        for (unsigned j = 0; j < J; ++j) {
            if (k * L + 32U * j >= G) continue;
            // One slot at a time: without the two empty asm statements (ordered among themselves) the compiler
            // loads every slot's activations first (5 slots x R rows x 9 registers) and spills.
            unsigned g = m + k * L + 32U * j;
            asm volatile("" : "+v"(g));
            int q[Wl][8];
            float scale[Wl];
#pragma unroll
            for (unsigned w = 0; w < Wl; ++w) {
                rn_decode_shift<P>(raw[k][j][w], table, q[w]);
                scale[w] = f16_to_float(raw[k][j][w].d);
            }
#pragma unroll
            for (unsigned r = 0; r < R; ++r) {
                int4 x0, x1;
                float delta;
                if constexpr (Lds) {
                    const auto* x = reinterpret_cast<const int4*>(xs + static_cast<std::size_t>(r) * G * 32U + 32U * g);
                    x0 = x[0]; x1 = x[1];
                    delta = ds[r * G + g];
                } else {
                    const auto* x = reinterpret_cast<const int4*>(xg + static_cast<std::size_t>(r) * v.x_stride + 32U * g);
                    x0 = x[0]; x1 = x[1];
                    delta = dg[static_cast<std::size_t>(r) * (v.x_stride / 32U) + g];
                }
#pragma unroll
                for (unsigned w = 0; w < Wl; ++w)
                    acc[w][r] = acc[w][r] + static_cast<float>(rn_dot32(q[w], x0, x1)) * (scale[w] * delta);
            }
#pragma unroll
            for (unsigned w = 0; w < Wl; ++w)
#pragma unroll
                for (unsigned r = 0; r < R; ++r) asm volatile("" : "+v"(acc[w][r]));
        }
    };
    float tot[Wl][R];
    rn_seg_tree<K, 0U, 1U, Wl, R>(leaf, tot);
#pragma unroll
    for (unsigned d = L / 2U; d; d >>= 1U)
#pragma unroll
        for (unsigned w = 0; w < Wl; ++w)
#pragma unroll
            for (unsigned r = 0; r < R; ++r) tot[w][r] = tot[w][r] + rn_xmask(tot[w][r], d);
    // Every lane of the segment holds the totals; lane m stores rows m, m + L, ...
#pragma unroll
    for (unsigned w = 0; w < Wl; ++w) {
        if (o0 + w >= v.count) continue;
#pragma unroll
        for (unsigned r0 = 0; r0 < R; r0 += L) {
            float total = tot[w][r0];
#pragma unroll
            for (unsigned r = r0 + 1U; r < R && r < r0 + L; ++r)
                if (m == r - r0) total = tot[w][r];
            if (r0 + m < R) {
                float* y = out + static_cast<std::size_t>(r0 + m) * v.out_stride + o0 + w;
                *y = v.accumulate ? *y + total : total;
            }
        }
    }
}

std::size_t rn_lds_bytes(unsigned rows, unsigned width, bool lds) {
    return lds ? static_cast<std::size_t>(rows) * width + static_cast<std::size_t>(rows) * (width / 32U) * 4U : 0U;
}

template <unsigned R, unsigned W, unsigned Waves, unsigned P>
void rn_flat_launch(const Q8A8View& v, hipStream_t q) {
    const unsigned outputs = Waves * W;
    const dim3 grid((v.count + outputs - 1U) / outputs), block(wave * Waves);
    const std::size_t lds = static_cast<std::size_t>(Waves) * W * R * (v.width / 32U) * 4U;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rn_flat_kernel<R, W, Waves, P>), grid, block, lds, q, v, v.codes, v.deltas, v.out);
}
template <unsigned R, unsigned W, unsigned Waves, unsigned P, bool Lds>
void rn_launch(const Q8A8View& v, hipStream_t q) {
    const unsigned outputs = Waves * W;
    const dim3 grid((v.count + outputs - 1U) / outputs), block(wave * Waves);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rn_kernel<R, W, Waves, P, Lds>), grid, block, rn_lds_bytes(R, v.width, Lds), q, v, v.codes,
                       v.deltas, v.out);
}
template <unsigned R, unsigned W, unsigned Waves, unsigned P>
void rn_flatw_launch(const Q8A8View& v, hipStream_t q) {
    const unsigned outputs = Waves * W;
    const dim3 grid((v.count + outputs - 1U) / outputs), block(wave * Waves);
    const std::size_t lds = static_cast<std::size_t>(Waves) * W * R * (v.width / 32U) * 4U;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rn_flatw_kernel<R, W, Waves, P>), grid, block, lds, q, v, v.codes, v.deltas, v.out);
}
template <unsigned R, unsigned Wl, unsigned L, unsigned Waves, unsigned P, bool Lds>
void rn_seg_launch(const Q8A8View& v, hipStream_t q) {
    const unsigned outputs = Waves * (wave / L) * Wl;
    const dim3 grid((v.count + outputs - 1U) / outputs), block(wave * Waves);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rn_seg_kernel<R, Wl, L, Waves, P, rn_seg_groups, Lds>), grid, block,
                       rn_lds_bytes(R, rn_seg_groups * 32U, Lds), q, v, v.codes, v.deltas, v.out);
}
// Forms 2 (flat, wave sync), 3 (segments of 8 lanes) and 4 (segments of 4 lanes): Q8_0 and U6 only.
template <unsigned R, unsigned P>
bool rn_dispatch_form(const Q8A8View& v, const RnWideShape& s, hipStream_t q) {
    if constexpr (P != 0U && P != 6U) {
        return false;
    } else {
        const unsigned W = s.outputs_per_wave, N = s.waves;
        if (s.form == 2U) {
            if (W == 4 && N == 4) { rn_flatw_launch<R, 4, 4, P>(v, q); return true; }
            if (W == 4 && N == 8) { rn_flatw_launch<R, 4, 8, P>(v, q); return true; }
            if (W == 2 && N == 8) { rn_flatw_launch<R, 2, 8, P>(v, q); return true; }
            if (W == 8 && N == 4) { rn_flatw_launch<R, 8, 4, P>(v, q); return true; }
            return false;
        }
        if (s.form == 3U) {
            if (s.lds) {
                if (W == 4 && N == 8) { rn_seg_launch<R, 1, 8, 8, P, true>(v, q); return true; }
                if (W == 4 && N == 4) { rn_seg_launch<R, 1, 8, 4, P, true>(v, q); return true; }
                if (W == 8 && N == 4) { rn_seg_launch<R, 2, 8, 4, P, true>(v, q); return true; }
            } else {
                if (W == 4 && N == 8) { rn_seg_launch<R, 1, 8, 8, P, false>(v, q); return true; }
                if (W == 4 && N == 4) { rn_seg_launch<R, 1, 8, 4, P, false>(v, q); return true; }
                if (W == 4 && N == 2) { rn_seg_launch<R, 1, 8, 2, P, false>(v, q); return true; }
                if (W == 8 && N == 4) { rn_seg_launch<R, 2, 8, 4, P, false>(v, q); return true; }
                if (W == 8 && N == 8) { rn_seg_launch<R, 2, 8, 8, P, false>(v, q); return true; }
            }
            return false;
        }
        if (s.form == 4U) {
            if (W == 8 && N == 4 && !s.lds) { rn_seg_launch<R, 1, 4, 4, P, false>(v, q); return true; }
            if (W == 8 && N == 4 && s.lds) { rn_seg_launch<R, 1, 4, 4, P, true>(v, q); return true; }
            return false;
        }
        return false;
    }
}
// Shapes registered for every format: the engine candidates. The full sweep (bench) for Q8_0 and U6 only.
template <unsigned P> inline constexpr bool rn_full_sweep = P == 0U || P == 6U;
template <unsigned R, unsigned P>
bool rn_dispatch_shape(const Q8A8View& v, const RnWideShape& s, hipStream_t q) {
    if (s.form >= 2U) return rn_dispatch_form<R, P>(v, s, q);
    const unsigned W = s.outputs_per_wave, N = s.waves;
    if (s.flat) {
        if (W == 4 && N == 4) { rn_flat_launch<R, 4, 4, P>(v, q); return true; }
        if constexpr (rn_full_sweep<P>) {
            if (W == 4 && N == 8) { rn_flat_launch<R, 4, 8, P>(v, q); return true; }
            if (W == 2 && N == 8) { rn_flat_launch<R, 2, 8, P>(v, q); return true; }
            if (W == 8 && N == 4) { rn_flat_launch<R, 8, 4, P>(v, q); return true; }
        }
        return false;
    }
    if (s.lds) {
        if (W == 2 && N == 16) { rn_launch<R, 2, 16, P, true>(v, q); return true; }
        if (W == 4 && N == 8) { rn_launch<R, 4, 8, P, true>(v, q); return true; }
        if (W == 1 && N == 16) { rn_launch<R, 1, 16, P, true>(v, q); return true; }
    }
    if constexpr (rn_full_sweep<P>) {
        if (s.lds) {
            if (W == 1 && N == 8) { rn_launch<R, 1, 8, P, true>(v, q); return true; }
            if (W == 2 && N == 8) { rn_launch<R, 2, 8, P, true>(v, q); return true; }
            if (W == 4 && N == 16) { rn_launch<R, 4, 16, P, true>(v, q); return true; }
        } else {
            if (W == 1 && N == 8) { rn_launch<R, 1, 8, P, false>(v, q); return true; }
            if (W == 2 && N == 8) { rn_launch<R, 2, 8, P, false>(v, q); return true; }
            if (W == 4 && N == 8) { rn_launch<R, 4, 8, P, false>(v, q); return true; }
            if (W == 1 && N == 16) { rn_launch<R, 1, 16, P, false>(v, q); return true; }
            if (W == 2 && N == 16) { rn_launch<R, 2, 16, P, false>(v, q); return true; }
            if (W == 4 && N == 16) { rn_launch<R, 4, 16, P, false>(v, q); return true; }
        }
    }
    return false;
}
template <unsigned P>
bool rn_dispatch_rows(const Q8A8View& v, const RnWideShape& s, hipStream_t q) {
    switch (v.rows) {
    case 1: return rn_dispatch_shape<1, P>(v, s, q);
    case 2: return rn_dispatch_shape<2, P>(v, s, q);
    case 3: return rn_dispatch_shape<3, P>(v, s, q);
    case 4: return rn_dispatch_shape<4, P>(v, s, q);
    case 5: return rn_dispatch_shape<5, P>(v, s, q);
    case 6: return rn_dispatch_shape<6, P>(v, s, q);
    case 7: return rn_dispatch_shape<7, P>(v, s, q);
    case 8: return rn_dispatch_shape<8, P>(v, s, q);
    default: return false;
    }
}

bool parse_shape(const char* e, RnWideShape& s) {
    unsigned w = 0, n = 0, lds = 0, flat = 0;
    const int got = e ? std::sscanf(e, "%u,%u,%u,%u", &w, &n, &lds, &flat) : 0;
    if (got < 3) return false;
    s.flat = got == 4 && flat == 1U;
    s.form = got == 4 && flat >= 2U ? flat : 0U;
    s.outputs_per_wave = w;
    s.waves = n;
    s.lds = lds != 0U;
    return true;
}

} // namespace

bool rn_wide_enabled() {
    static const bool on = strix::envcfg::flag("STRIX_DS41_RN_WIDE", false);
    return on;
}

RnWideShape rn_wide_default_shape(const Q8A8View& view) {
    static const RnWideShape forced = [] {
        RnWideShape s;
        s.waves = 0;
        parse_shape(std::getenv("STRIX_DS41_RN_SHAPE"), s);
        return s;
    }();
    if (forced.waves != 0U) return forced;
    // From the cold bench (strix-hip-ds41-rn-wide-bench, sr-r6-bench2): widths whose group count is not a multiple of
    // 32 (attn_q_b, 40 groups) take 2 outputs per wave, 8 waves, activations read in place (U6 R6 125 -> 175 GB/s; the
    // flat form measured lower and is not used by default); the others one output per wave, 16 waves, LDS-staged rows
    // (attn_output_a U6 R6 149 -> 197 GB/s).
    RnWideShape s;
    s.outputs_per_wave = 1;
    s.waves = 16;
    s.lds = true;
    if ((view.width / 32U) % 32U != 0U) {
        RnWideShape n;
        n.outputs_per_wave = 2;
        n.waves = 8;
        n.lds = false;
        if (rn_wide_supported(view, n)) return n;
    }
    return s;
}

// The (shape, format) pairs rn_dispatch_shape launches: every format takes the engine candidates (LDS 2x16, 4x8,
// 1x16, flat 4x4); the full sweep only Q8_0 and U6.
static bool rn_registered(unsigned packed, const RnWideShape& s) {
    const unsigned W = s.outputs_per_wave, N = s.waves;
    const bool common = s.flat ? W == 4 && N == 4 : s.lds && ((W == 2 && N == 16) || (W == 4 && N == 8) || (W == 1 && N == 16));
    if (common) return true;
    if (packed != 0U && packed != 6U) return false;
    if (s.flat) return (W == 4 && N == 8) || (W == 2 && N == 8) || (W == 8 && N == 4);
    return (N == 8 || N == 16) && (W == 1 || W == 2 || W == 4);
}

bool rn_wide_supported(const Q8A8View& v, const RnWideShape& s) {
    const unsigned per_group = v.per_group ? v.per_group : v.count;
    const unsigned outputs = s.waves * s.outputs_per_wave;
    if (s.form >= 2U) {
        // Forms 2-4: Q8_0 and U6; the segment forms only for the registered group count, whole segments per wave.
        // Exactly the shapes rn_dispatch_form registers, so a forced STRIX_DS41_RN_SHAPE never reaches a missing kernel.
        const bool seg = s.form == 3U || s.form == 4U;
        const unsigned W = s.outputs_per_wave, N = s.waves;
        const bool registered =
            s.form == 2U ? !s.lds && ((W == 4U && (N == 4U || N == 8U)) || (W == 2U && N == 8U) || (W == 8U && N == 4U))
            : s.form == 3U ? (s.lds ? (W == 4U && (N == 8U || N == 4U)) || (W == 8U && N == 4U)
                                    : (W == 4U && (N == 8U || N == 4U || N == 2U)) || (W == 8U && (N == 4U || N == 8U)))
            : s.form == 4U && W == 8U && N == 4U;
        const std::size_t lds = seg ? rn_lds_bytes(v.rows, v.width, s.lds) : std::size_t{outputs} * v.rows * (v.width / 32U) * 4U;
        return registered && !v.planar && (v.packed == 0U || v.packed == 6U) && v.rows >= 1U && v.rows <= rn_max_rows &&
               outputs != 0U && per_group % outputs == 0U && lds <= 64U * 1024U &&
               (!seg || v.width == rn_seg_groups * 32U) && v.x_stride % 16U == 0U &&
               reinterpret_cast<std::uintptr_t>(v.codes) % 16U == 0U;
    }
    if (!rn_registered(v.packed, s)) return false;  // forms 0 / 1: unregistered (shape, format) -> the caller keeps the engine kernel
    const std::size_t lds = s.flat ? std::size_t{outputs} * v.rows * (v.width / 32U) * 4U : rn_lds_bytes(v.rows, v.width, s.lds);
    return !v.planar && (v.packed == 0U || (v.packed >= 4U && v.packed <= 6U)) && v.rows >= 1U && v.rows <= rn_max_rows &&
           outputs != 0U && per_group % outputs == 0U && lds <= 64U * 1024U &&
           v.x_stride % 16U == 0U && reinterpret_cast<std::uintptr_t>(v.codes) % 16U == 0U;
}

void enqueue_q8_0_a8_rn(const Q8A8View& v, const RnWideShape& shape, void* stream) {
    Q8A8View view = v;
    if (!view.per_group) view.per_group = view.count;
    require(view.weight && view.codes && view.deltas && view.out && view.count && view.width && view.width % 32U == 0 &&
                view.count % view.per_group == 0 && (view.rows == 1 || view.x_stride >= (view.count / view.per_group) * view.width) &&
                (!view.row_width || (view.row_width % 32U == 0 && view.k_offset % 32U == 0 && view.k_offset + view.width <= view.row_width &&
                                     view.per_group == view.count)),
            "Q8_0 x A8 rn site view");
    require(rn_wide_supported(view, shape), "Q8_0 x A8 rn: shape not supported for this view");
    const auto q = static_cast<hipStream_t>(stream);
    bool ok = false;
    if (view.packed == 0U) ok = rn_dispatch_rows<0>(view, shape, q);
    else if (view.packed == 4U) ok = rn_dispatch_rows<4>(view, shape, q);
    else if (view.packed == 5U) ok = rn_dispatch_rows<5>(view, shape, q);
    else ok = rn_dispatch_rows<6>(view, shape, q);
    require(ok, "Q8_0 x A8 rn: shape not registered for this format");
    hip_check(hipGetLastError(), "Q8_0 x A8 rn launch");
}

namespace {
template <unsigned R, unsigned P>
void rn_ready_launch(const Q8A8View& v, dist::ProducerStage stage, hipStream_t q) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(rn_ready_kernel<R, P>), dim3((v.count + 15U) / 16U), dim3(wave * 16),
                       rn_lds_bytes(R, v.width, true), q, v, stage);
}
template <unsigned P>
void rn_ready_rows(const Q8A8View& v, dist::ProducerStage stage, hipStream_t q) {
    switch (v.rows) {
    case 1: rn_ready_launch<1, P>(v, stage, q); break;
    case 2: rn_ready_launch<2, P>(v, stage, q); break;
    case 3: rn_ready_launch<3, P>(v, stage, q); break;
    case 4: rn_ready_launch<4, P>(v, stage, q); break;
    case 5: rn_ready_launch<5, P>(v, stage, q); break;
    case 6: rn_ready_launch<6, P>(v, stage, q); break;
    case 7: rn_ready_launch<7, P>(v, stage, q); break;
    case 8: rn_ready_launch<8, P>(v, stage, q); break;
    }
}
}

bool enqueue_q8_0_a8_rn_ready(const Q8A8View& v, const RnWideShape& shape, dist::ProducerStage stage, void* stream) {
    if (shape.form || shape.flat || !shape.lds || shape.outputs_per_wave != 1U || shape.waves != 16U ||
        !rn_wide_supported(v, shape)) return false;
    require(stage.data == v.out && stage.flag && stage.token && !v.a8_codes && !v.accumulate && v.out_stride == v.count,
            "Q8 x A8 RN ready producer view");
    const auto q = static_cast<hipStream_t>(stream);
    if (v.packed == 0U) rn_ready_rows<0>(v, stage, q);
    else if (v.packed == 4U) rn_ready_rows<4>(v, stage, q);
    else if (v.packed == 5U) rn_ready_rows<5>(v, stage, q);
    else rn_ready_rows<6>(v, stage, q);
    hip_check(hipGetLastError(), "Q8 x A8 RN ready epilogue launch");
    return true;
}

} // namespace strix::ds41
