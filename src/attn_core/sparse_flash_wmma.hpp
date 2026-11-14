#pragma once
// Shared FAST sparse attention core: per-row key lists (GLM MLA/DSA top-k, MiniMax MSA selected blocks; later DS41
// MLA), flash form, 16-bit WMMA 16x16x16 -> FP32, FP32 online softmax. Tiles and softmax: flash_tile.hpp.
//
// Identity: FAST, per engine (GLM: default with STRIX_GLM_PF_WMMA=1, STRIX_GLM_PF_ATTN_WMMA=0 opts out). Nothing
// here runs unless an engine launches it, so the exact paths keep their bytes. Lossy against the exact kernels: queries and probabilities are
// rounded to the operand type (RNE), the score and PV sums run in WMMA order, and the softmax is the online form with
// exp2. Keys/values are used as stored (bf16 caches).
//
// Shape: one work-group = one query row (GLM: a token, all its 32 heads; MSA: a (token, KV group), its 16 query
// heads) x the row's own key list walked in blocks of T keys. Waves = (H / 16 M tiles) x 4 head-dim quarters. Wave w
// = (M tile w % kMt, quarter w / kMt): its quarter of the queries stays in registers as A fragments for the whole
// row, it computes the quarter's partial scores Q[16][D/4] . K[T][D/4]^T with K fragments read straight from the key
// rows, and it owns the output accumulators O[16][D/4]. Per block:
//   1. gather: every thread loads 4 keys x 8 value dims (16-B pieces) and stores them transposed (V^T[dim][key],
//      8-B stores) into LDS for the PV B operand (MLA: value row = key row, L0/L1 hits);
//   2. QK on WMMA, the four quarter partials go to LDS as FP32;                                       barrier
//   3. softmax thread = (head, T/8 keys): sum of the 4 partials (fixed order), x scale*log2e, mask keys >= n,
//      tile::online_step (8-lane max, running m/l, rounded p), P[head][key] and alpha[head] to LDS;  barrier
//   4. PV: O *= alpha (per accumulator row), O += P[16][T] . V[T][D/4] on WMMA;                      barrier
// Epilogue: O / l to the policy's FP32 output [H][D].
// LDS: D x (2T + 16) B (V^T, odd 16-B pitch: 16 consecutive rows hit distinct bank groups) + 4 x H x T x 4 B
// (partials) + H x (2T + 16) B (P) + H x 4 B (alpha). GLM (H 32, D 512): 60,032 B at T 32, 34,432 B at T 16 (200 /
// 255 VGPR, 0 scratch, 4 / 5 waves per SIMD, gfx1151). MSA shape (H 16, D 128, T 32): 4 waves, 19,776 B.
// Not built yet: D_qk != D_v (DS41 MLA rope part: key width 576, value width 512) - add a K-only k-step tail.
//
// Policy P:
//   using Args;                                            kernel argument (trivially copyable)
//   static constexpr unsigned kHeads, kDim;                H (16 | 32), D (key width = value width, % 64 == 0)
//   static constexpr tile::Operand kOp;                    operand type (bf16 caches: bf16)
//   static constexpr float kScaleLog2;                     softmax scale x log2(e)
//   static __device__ unsigned count(const Args&, unsigned row)                      keys in the row's list
//   static __device__ unsigned key(const Args&, unsigned row, unsigned j)            cache position of entry j < count
//   static __device__ const std::uint16_t* key_row(const Args&, unsigned row, unsigned pos)    [D], 16-B aligned
//   static __device__ const std::uint16_t* value_row(const Args&, unsigned row, unsigned pos)  [D], 16-B aligned
//   static __device__ const float* query(const Args&, unsigned row)                 FP32 [H][D]
//   static __device__ float* out(const Args&, unsigned row)                         FP32 [H][D]
// Launch: flash_rows_kernel<P, T><<<rows, Geometry<P::kHeads, P::kDim, T>::kThreads, 0, stream>>>(args).
#include "attn_core/flash_tile.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::attn_core::sparse {

template <unsigned H, unsigned D, unsigned T> struct Geometry {
    static_assert(H == 16 || H == 32, "M tiles of 16 heads");
    static_assert(D % 64 == 0 && D <= 512, "a head-dim quarter is a whole number of 16-wide k steps / tiles");
    static_assert(T == 16 || T == 32, "key block");
    static constexpr unsigned kMt = H / 16, kQuarters = 4, kWaves = kMt * kQuarters, kThreads = 32 * kWaves;
    static constexpr unsigned kQd = D / kQuarters;
    static constexpr unsigned kKs = kQd / 16;   // QK k steps per wave
    static constexpr unsigned kDt = kQd / 16;   // output tiles per wave
    static constexpr unsigned kNt = T / 16;     // key tiles per block
    static constexpr unsigned kPitch = 2 * T + 16;
    static constexpr unsigned kVt = 0, kSp = kVt + D * kPitch, kP = kSp + kQuarters * H * T * 4, kAl = kP + H * kPitch;
    static constexpr unsigned kBytes = kAl + H * 4;
    static constexpr unsigned kSoftKeys = T * H / kThreads;  // keys per softmax thread
    static_assert(T / kSoftKeys == 8 && kThreads / 8 == H, "8 lanes per head in the softmax");
    static constexpr unsigned kTg = T / 4, kDg = kThreads / kTg, kGPass = (D / 8) / kDg;  // gather: 4 keys x 8 dims
    static_assert(kGPass >= 1 && kGPass * kDg * 8 == D, "gather covers the value width");
};

template <class P, unsigned T>
__global__ __launch_bounds__((Geometry<P::kHeads, P::kDim, T>::kThreads)) void flash_rows_kernel(typename P::Args a) {
    using G = Geometry<P::kHeads, P::kDim, T>;
    using tile::Acc;
    using tile::Frag;
    using tile::U2;
    using tile::U4;
    constexpr unsigned H = P::kHeads, D = P::kDim;
    constexpr tile::Operand O = P::kOp;
    __shared__ __attribute__((aligned(16))) unsigned char lds[G::kBytes];
    const unsigned r = blockIdx.x, tid = threadIdx.x, w = tid >> 5U, lane = tid & 31U, ln = lane & 15U, hf = lane >> 4U;
    const unsigned mt = w % G::kMt, qd = w / G::kMt;
    const unsigned n = P::count(a, r);
    float* const out = P::out(a, r);
    if (n == 0) {  // uniform: nothing attended (not produced by the engines; defined as zeros)
        for (unsigned x = tid; x < H * D; x += G::kThreads) out[x] = 0.0F;
        return;
    }
    Frag qf[G::kKs];
    {
        const float* qr = P::query(a, r) + static_cast<std::size_t>(mt * 16U + ln) * D + qd * G::kQd;
#pragma unroll
        for (unsigned k = 0; k < G::kKs; ++k) qf[k] = tile::frag_from_f32<O>(qr + 16U * k);
    }
    Acc o[G::kDt];
#pragma unroll
    for (unsigned c = 0; c < G::kDt; ++c) o[c] = tile::acc_zero();
    float* const sp = reinterpret_cast<float*>(lds + G::kSp);
    float* const al = reinterpret_cast<float*>(lds + G::kAl);
    const unsigned sh = tid >> 3U, sg = tid & 7U;          // softmax: head, key group
    const unsigned gt = tid % G::kTg, gd = tid / G::kTg;   // gather: key group (4 keys), dim group (8 dims)
    tile::SoftmaxState st;
    const unsigned blocks = (n + T - 1U) / T;
    for (unsigned b = 0; b < blocks; ++b) {
        const unsigned j0 = b * T;
        // 1. gather loads (in flight during QK)
        U4 gv[G::kGPass][4];
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const unsigned j = j0 + 4U * gt + i;
            const std::uint16_t* vr = P::value_row(a, r, P::key(a, r, mask::list_clamp(j, n)));
#pragma unroll
            for (unsigned p = 0; p < G::kGPass; ++p) gv[p][i] = *reinterpret_cast<const U4*>(vr + 8U * (gd + p * G::kDg));
        }
        // 2. QK partials of this wave's quarter
#pragma unroll
        for (unsigned tt = 0; tt < G::kNt; ++tt) {
            const unsigned j = j0 + 16U * tt + ln;
            const std::uint16_t* kr = P::key_row(a, r, P::key(a, r, mask::list_clamp(j, n))) + qd * G::kQd;
            Acc s = tile::acc_zero();
#pragma unroll
            for (unsigned k = 0; k < G::kKs; ++k) s = tile::mma<O>(qf[k], tile::frag32(kr + 16U * k), s);
            float* spw = sp + (static_cast<std::size_t>(qd) * H + mt * 16U + hf) * T + 16U * tt + ln;
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) spw[2U * e * T] = s[e];
        }
        // V^T stores: dim row 8 (gd + p kDg) + k, keys 4 gt .. 4 gt + 3
#pragma unroll
        for (unsigned p = 0; p < G::kGPass; ++p) tile::store_vt4x8<G::kPitch>(lds + G::kVt, gv[p], 8U * (gd + p * G::kDg), gt);
        __syncthreads();
        // 3. softmax (thread = head sh, keys sg * K ..)
        {
            constexpr unsigned K = G::kSoftKeys;
            const float* s0 = sp + sh * T + sg * K;
            float sv[K];
            tile::sum_partials<G::kQuarters, K>(s0, H * T, sv);
#pragma unroll
            for (unsigned i = 0; i < K; ++i) sv[i] = mask::list_ok(j0 + sg * K + i, n) ? sv[i] * P::kScaleLog2 : -__builtin_huge_valf();
            std::uint16_t pb[K];
            const float alpha = tile::online_step<O, K, 8, tile::SumOf::rounded>(sv, st, pb);
            unsigned char* pr = lds + G::kP + sh * G::kPitch + sg * K * 2U;
            if constexpr (K == 4)
                *reinterpret_cast<U2*>(pr) = U2{pb[0] | (static_cast<unsigned>(pb[1]) << 16U), pb[2] | (static_cast<unsigned>(pb[3]) << 16U)};
            else
                *reinterpret_cast<unsigned*>(pr) = pb[0] | (static_cast<unsigned>(pb[1]) << 16U);
            if (sg == 0) al[tile::row_slot(sh)] = alpha;
        }
        __syncthreads();
        // 4. PV
        {
            float av[8];
            tile::row_scalars(al, mt, lane, av);
#pragma unroll
            for (unsigned c = 0; c < G::kDt; ++c)
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) o[c][e] *= av[e];
#pragma unroll
            for (unsigned tt = 0; tt < G::kNt; ++tt) {
                const Frag pf = tile::frag32(lds + G::kP + (mt * 16U + ln) * G::kPitch + 32U * tt);
#pragma unroll
                for (unsigned c = 0; c < G::kDt; ++c)
                    o[c] = tile::mma<O>(pf, tile::frag32(lds + G::kVt + (qd * G::kQd + 16U * c + ln) * G::kPitch + 32U * tt), o[c]);
            }
        }
        __syncthreads();
    }
    if (sg == 0) al[tile::row_slot(sh)] = tile::finalize_scale(st);
    __syncthreads();
    float iv[8];
    tile::row_scalars(al, mt, lane, iv);
#pragma unroll
    for (unsigned c = 0; c < G::kDt; ++c)
#pragma unroll
        for (unsigned e = 0; e < 8; ++e)
            out[static_cast<std::size_t>(mt * 16U + 2U * e + hf) * D + qd * G::kQd + 16U * c + ln] = o[c][e] * iv[e];
}

}  // namespace strix::attn_core::sparse
