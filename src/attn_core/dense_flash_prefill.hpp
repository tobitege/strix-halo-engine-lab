#pragma once
// Shared FAST dense causal GQA flash-attention prefill core (f16/bf16 WMMA 16x16x16 -> FP32, FP32 online softmax).
// Tile helpers (operand rounding, fragments, WMMA call) come from flash_tile.hpp (namespace tile, canonical).
//
// Identity: FAST, opt-in per engine (gemma4: STRIX_G4_PREFILL_ATTN=flash). Nothing here runs unless an engine
// launches it. Lossy against exact attention: Q, K, V and the probabilities are Policy::kOp WMMA operands, the score and
// PV sums run in WMMA order, and the softmax is the online (running max) form with exp2. Judged on end metrics.
//
// Shape: one work-group = one KV head x Group query heads x Positions consecutive query positions (Group *
// Positions query rows, 16-row WMMA tiles). Every K/V tile of 16 keys is staged into LDS ONCE per work-group and
// reused by all its query rows (GQA: the whole KV group), instead of once per 16-row tile. Wave w = (row tile
// w / Split, head-dim slice w % Split): the slice's FP16 queries stay in registers as A fragments for the whole
// row tile, the wave computes the slice's partial scores Q[16][D/Split] . K[16][D/Split]^T and owns the output
// accumulators O[16][D/Split]. Per key tile:
//   1. stage: the prefetched K rows ([key][D], padded pitch) and V^T rows ([d][16 keys]) go to LDS;   barrier
//      the next tile's global loads are issued right away and land during the math;
//   2. QK on WMMA, the Split partials go to LDS as FP32;                                             barrier
//   3. every wave of the row tile sums the partials (fixed order), scales by scale*log2(e), applies the causal and
//      sliding-window mask, updates its running max/sum and builds the FP16 P fragment (the two half-waves split
//      the exponentials), rescales O by alpha and adds P[16][16] . V[16][D/Split] on WMMA;           barrier
// Split-K (splits > 1, grid z): work-group z walks its share of the block's key tiles and writes the unnormalised
// O, the running max m (log2 units) and sum l to a workspace; combine<P> merges the splits per (row, head):
// out = sum_z 2^(m_z - M) O_z / sum_z 2^(m_z - M) l_z. Fills the GPU when rows x KV heads give few work-groups.
// Key tiles outside [lo(first row), last row] are never visited (causal + window block skip); tiles a row tile
// cannot see are skipped per wave. Epilogue: O / l to the policy's FP32 output.
// LDS: 16 x (D + 8) x 2 (K) + D x 32 (V^T) + waves x 16 x 20 x 4 (partials) = 53,504 B at D512 / 16 waves.
//
// Policy P (all __device__ static unless noted):
//   using Args;  using Shape = dense::Shape<D, Group, Split, Positions>;  static constexpr float kScaleLog2;
//   static constexpr tile::Operand kOp;             operand type of Q/K/V/P (K/V stored in it)
//   bool valid(const Args&)                         geometry check (false: failure |= 2, nothing written)
//   unsigned rows(const Args&)                      query rows (consecutive positions) - also host
//   unsigned position(const Args&)                  absolute position of row 0
//   unsigned window(const Args&)                    sliding window (keys p-window+1 .. p), 0 = full causal
//   const float* query(const Args&, row, head)      FP32 [D], 16-B aligned
//   float* out(const Args&, row, head)              FP32 [D]
//   const uint16_t* key(const Args&, pos, kv_head)  kOp [D] of key pos, 16-B aligned (finite for every pos of a
//                                                   visited 16-key tile; masked keys are multiplied by zero)
//   const uint16_t* value_t(const Args&, kv_head, d) kOp V^T row: element pos = V[pos][d]; 16-B aligned at pos%16==0
//   unsigned* failure(const Args&)                  bit 1 nonfinite output, bit 4 operand query overflow
// The denominator sums the operand-rounded p (what PV accumulates) in key order across both half-waves.
// rows() is __host__ __device__. Launch: launch<P>(args, heads, stream, workspace, splits) - grid (row blocks, heads / Group, splits), longest
// blocks first; workspace_floats(rows, heads, D, splits) FP32 when splits > 1.
#include <hip/hip_runtime.h>

#include <cstdint>

#include "flash_core.hpp"

namespace strix::attn_core::dense {

using tile::Acc;
using tile::Frag;
using tile::U4;

template <unsigned Dim, unsigned Group, unsigned Split, unsigned Positions = 16>
struct Shape {
    static_assert(Dim % (16 * Split) == 0, "head dim must split into 16-wide slices");
    static_assert(Positions % 16 == 0, "16 consecutive positions per row tile");
    static constexpr unsigned dim = Dim, group = Group, split = Split, positions = Positions;
    static constexpr unsigned tiles = Group * Positions / 16, waves = tiles * Split, threads = 32 * waves;
    static constexpr unsigned slice = Dim / Split, frags = slice / 16;
    static constexpr unsigned kpitch = Dim + 8, spitch = 20; // halfs / floats
    static constexpr unsigned pieces = 2 * Dim, per_thread = (pieces + threads - 1) / threads;
    static constexpr unsigned lds_bytes = 16 * kpitch * 2 + Dim * 16 * 2 + waves * 16 * spitch * 4;
    static_assert(threads <= 1024, "work-group too large");
    static_assert(lds_bytes <= 65536, "LDS budget");
};

__device__ inline unsigned window_lo(unsigned p, unsigned window) { return window && p >= window ? p - window + 1U : 0U; }

template <class P>
__device__ __forceinline__ void flash_body(typename P::Args a, float* ws, unsigned splits) {
    using S = typename P::Shape;
    constexpr unsigned D = S::dim, KP = S::kpitch, SP = S::spitch, F = S::frags, T = S::threads;
    constexpr auto O = P::kOp;
    __shared__ __attribute__((aligned(16))) std::uint16_t ks[16 * KP];
    __shared__ __attribute__((aligned(16))) std::uint16_t vs[D * 16];
    __shared__ __attribute__((aligned(16))) float part[S::waves * 16 * SP];
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5, rr = lane & 15U, hw = lane >> 4;
    const unsigned tile = wave / S::split, piece = wave % S::split;
    if (!P::valid(a)) {
        if (!tid) atomicOr(P::failure(a), 2U);
        return;
    }
    const unsigned rows = P::rows(a), blocks = (rows + S::positions - 1U) / S::positions;
    const unsigned block = blocks - 1U - blockIdx.x, kvh = blockIdx.y, row_base = block * S::positions;
    const unsigned p0 = P::position(a), window = P::window(a);
    const unsigned block_rows = min(S::positions, rows - row_base);
    const unsigned pos_first = p0 + row_base, pos_last = pos_first + block_rows - 1U;
    const unsigned key_begin = window_lo(pos_first, window) & ~15U;
    const unsigned tiles_total = (pos_last >> 4) - (key_begin >> 4) + 1U, share = (tiles_total + splits - 1U) / splits;
    const unsigned kt_begin = key_begin + 16U * min(tiles_total, blockIdx.z * share);
    const unsigned kt_end = key_begin + 16U * min(tiles_total, (blockIdx.z + 1U) * share); // exclusive
    // This wave's 16 rows: one query head, 16 consecutive positions.
    const unsigned head = kvh * S::group + tile % S::group, sub = tile / S::group * 16U;
    const bool wave_live = sub < block_rows;
    const unsigned tile_last = pos_first + min(sub + 15U, block_rows - 1U), tile_lo = window_lo(pos_first + sub, window);
    const unsigned my_row = row_base + sub + rr, my_pos = p0 + my_row, my_lo = window_lo(my_pos, window);
    const bool live = wave_live && sub + rr < block_rows;
    const unsigned d0 = piece * S::slice;
    bool bad_half = false;
    Frag qf[F];
    {
        const float* q = live ? P::query(a, my_row, head) + d0 : nullptr;
        constexpr unsigned short exponent = O == tile::Operand::f16 ? 0x7c00U : 0x7f80U;
#pragma unroll
        for (unsigned f = 0; f < F; ++f) {
            qf[f] = live ? tile::frag_from_f32<O>(q + 16U * f) : Frag{};
#pragma unroll
            for (unsigned i = 0; i < 16; ++i) bad_half |= (qf[f][i] & exponent) == exponent;
        }
    }
    U4 kreg[S::per_thread], vreg[S::per_thread];
    const auto fetch = [&](unsigned kt) {
#pragma unroll
        for (unsigned i = 0; i < S::per_thread; ++i) {
            const unsigned idx = tid + i * T;
            if (S::pieces % T == 0 || idx < S::pieces) {
                const unsigned key = idx / (D / 8U), c = idx % (D / 8U);
                kreg[i] = *reinterpret_cast<const U4*>(P::key(a, kt + key, kvh) + 8U * c);
                const unsigned d = idx >> 1, h = idx & 1U;
                vreg[i] = *reinterpret_cast<const U4*>(P::value_t(a, kvh, d) + kt + 8U * h);
            }
        }
    };
    const auto stage = [&] {
#pragma unroll
        for (unsigned i = 0; i < S::per_thread; ++i) {
            const unsigned idx = tid + i * T;
            if (S::pieces % T == 0 || idx < S::pieces) {
                const unsigned key = idx / (D / 8U), c = idx % (D / 8U);
                *reinterpret_cast<U4*>(&ks[key * KP + 8U * c]) = kreg[i];
                const unsigned d = idx >> 1, h = idx & 1U;
                *reinterpret_cast<U4*>(&vs[d * 16U + 8U * h]) = vreg[i];
            }
        }
    };
    constexpr float neg_inf = -__builtin_huge_valf();
    float maximum = neg_inf, denominator = 0.0F;
    Acc acc[F];
#pragma unroll
    for (unsigned f = 0; f < F; ++f) acc[f] = tile::acc_zero();
    if (kt_begin < kt_end) fetch(kt_begin);
    for (unsigned kt = kt_begin; kt < kt_end; kt += 16U) {
        stage();
        __syncthreads();
        if (kt + 16U < kt_end) fetch(kt + 16U);
        const bool active = wave_live && kt <= tile_last && kt + 15U >= tile_lo; // wave-uniform
        if (active) {
            // Two independent WMMA chains (even / odd 16-wide steps) halve the dependent latency.
            Acc s = tile::acc_zero(), s1 = tile::acc_zero();
#pragma unroll
            for (unsigned f = 0; f < F; f += 2) {
                s = tile::mma<O>(qf[f], tile::frag32(&ks[rr * KP + d0 + 16U * f]), s);
                if (f + 1 < F) s1 = tile::mma<O>(qf[f + 1], tile::frag32(&ks[rr * KP + d0 + 16U * f + 16U]), s1);
            }
            s += s1;
            float* pw = part + wave * 16U * SP;
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) pw[(2U * e + hw) * SP + rr] = s[e];
        }
        __syncthreads();
        if (active) {
            // Each half-wave owns 8 of the 16 keys of its row (j = 8 hw + i); max and presence meet over lane ^ 16.
            float sc[8];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) sc[i] = 0.0F;
#pragma unroll
            for (unsigned p = 0; p < S::split; ++p) {
                const float* pp = part + (tile * S::split + p) * 16U * SP + rr * SP + 8U * hw;
#pragma unroll
                for (unsigned c = 0; c < 2; ++c) {
                    const float4 v = reinterpret_cast<const float4*>(pp)[c];
                    sc[4U * c] += v.x; sc[4U * c + 1] += v.y; sc[4U * c + 2] += v.z; sc[4U * c + 3] += v.w;
                }
            }
            float m = maximum;
            bool mine = false;
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                const unsigned key = kt + 8U * hw + i;
                const bool v = live && key >= my_lo && key <= my_pos;
                sc[i] = v ? sc[i] * P::kScaleLog2 : neg_inf;
                m = sc[i] > m ? sc[i] : m;
                mine |= v;
            }
            {
                const float other = __shfl_xor(m, 16, 32);
                m = other > m ? other : m;
            }
            const bool present = mine || __shfl_xor(int(mine), 16, 32) != 0;
            // The online update is spelled out, not routed through online_step (flash_core.hpp): identical
            // math through the shared step adds 4-24 B of scratch per kernel on gfx1151 (remarks, R1-7).
            Frag pa;
            const float alpha = present ? __builtin_amdgcn_exp2f(maximum - m) : 1.0F;
            {
                float own[8];
#pragma unroll
                for (unsigned i = 0; i < 8; ++i) own[i] = present ? __builtin_amdgcn_exp2f(sc[i] - m) : 0.0F;
                float sum = 0.0F;
#pragma unroll
                for (unsigned j = 0; j < 16; ++j) {
                    const float other = __shfl_xor(own[j % 8U], 16, 32);
                    const std::uint16_t p = tile::to_operand<O>(j / 8U == hw ? own[j % 8U] : other);
                    sum += tile::operand_value<O>(p);
                    pa[j] = p;
                }
                denominator = denominator * alpha + sum;
                maximum = m;
            }
            float scale[8];
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) scale[e] = __shfl(alpha, int(2U * e + hw), 32);
#pragma unroll
            for (unsigned f = 0; f < F; ++f) {
                Acc c = acc[f];
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) c[e] *= scale[e];
                acc[f] = tile::mma<O>(pa, tile::frag32(&vs[(d0 + 16U * f + rr) * 16U]), c);
            }
        }
        __syncthreads();
    }
    bool bad_value = false;
    const unsigned heads = gridDim.y * S::group;
    if (splits > 1U) {
        if (wave_live) {
            float* acc_ws = ws + ((std::size_t(blockIdx.z) * rows + row_base + sub) * heads + head) * D + d0;
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) {
                if (sub + 2U * e + hw >= block_rows) continue;
                float* o = acc_ws + std::size_t(2U * e + hw) * heads * D;
#pragma unroll
                for (unsigned f = 0; f < F; ++f) o[16U * f + rr] = acc[f][e];
            }
            if (live && piece == 0U && hw == 0U) {
                float* st = ws + std::size_t(splits) * rows * heads * D + ((std::size_t(blockIdx.z) * rows + my_row) * heads + head) * 2U;
                st[0] = maximum; st[1] = denominator;
            }
        }
        if (bad_half) atomicOr(P::failure(a), 4U);
        return;
    }
    if (wave_live) {
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
            const unsigned r = sub + 2U * e + hw;
            const float total = __shfl(denominator, int(2U * e + hw), 32);
            if (r < block_rows) {
                float* o = P::out(a, row_base + r, head) + d0;
#pragma unroll
                for (unsigned f = 0; f < F; ++f) {
                    const float value = acc[f][e] / total;
                    bad_value |= (__builtin_bit_cast(unsigned, value) & 0x7f800000U) == 0x7f800000U;
                    o[16U * f + rr] = value;
                }
            }
        }
    }
    if (bad_value) atomicOr(P::failure(a), 1U);
    if (bad_half) atomicOr(P::failure(a), 4U);
}
// Register cap for 8 waves per SIMD (192 VGPRs on gfx1151, 44 B/lane scratch): two 16-wave work-groups per WGP;
// 1.3x the uncapped 238-VGPR kernel (one work-group per WGP) at 32K.
template <class P>
__global__ __launch_bounds__(P::Shape::threads) __attribute__((amdgpu_waves_per_eu(8))) void flash_prefill(
    typename P::Args a, float* ws, unsigned splits) {
    flash_body<P>(a, ws, splits);
}

// Split-K merge: one work-group per (row, head), D / 4 threads x 4 components.
template <class P>
__global__ __launch_bounds__(P::Shape::dim / 4U) void combine(typename P::Args a, const float* ws, unsigned splits) {
    constexpr unsigned D = P::Shape::dim;
    const unsigned row = blockIdx.x, head = blockIdx.y, heads = gridDim.y, rows = P::rows(a), d = threadIdx.x * 4U;
    const float* st = ws + std::size_t(splits) * rows * heads * D;
    float m = -__builtin_huge_valf();
    for (unsigned z = 0; z < splits; ++z) { const float v = st[((std::size_t(z) * rows + row) * heads + head) * 2U]; m = v > m ? v : m; }
    float l = 0.0F, o[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (unsigned z = 0; z < splits; ++z) {
        const std::size_t at = (std::size_t(z) * rows + row) * heads + head;
        const float mz = st[at * 2U];
        if (!(mz > -__builtin_huge_valf())) continue; // empty split
        const float w = __builtin_amdgcn_exp2f(mz - m);
        l += w * st[at * 2U + 1U];
        const float4 v = *reinterpret_cast<const float4*>(ws + at * D + d);
        o[0] += w * v.x; o[1] += w * v.y; o[2] += w * v.z; o[3] += w * v.w;
    }
    float* out = P::out(a, row, head) + d;
    bool bad = false;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const float value = o[i] / l;
        bad |= (__builtin_bit_cast(unsigned, value) & 0x7f800000U) == 0x7f800000U;
        out[i] = value;
    }
    if (bad) atomicOr(P::failure(a), 1U);
}
__host__ __device__ constexpr std::size_t workspace_floats(unsigned rows, unsigned heads, unsigned dim, unsigned splits) {
    return splits > 1U ? std::size_t(splits) * rows * heads * (dim + 2U) : 0U;
}
template <class P>
inline void launch(const typename P::Args& a, unsigned heads, hipStream_t stream, float* ws = nullptr, unsigned splits = 1) {
    using S = typename P::Shape;
    const unsigned rows = P::rows(a);
    if (!rows) return;
    if (!ws) splits = 1;
    const dim3 grid((rows + S::positions - 1U) / S::positions, heads / S::group, splits);
    hipLaunchKernelGGL(flash_prefill<P>, grid, dim3(S::threads), 0, stream, a, ws, splits);
    if (splits > 1U) hipLaunchKernelGGL(combine<P>, dim3(rows, heads), dim3(S::dim / 4U), 0, stream, a, ws, splits);
}

} // namespace strix::attn_core::dense
