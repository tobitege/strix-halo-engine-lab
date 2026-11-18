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
// Pipelined loads (template Pf, default; STRIX_ATTN_FLASH_PF=0 = the single-stage loop above, same bits): the list
// positions run two blocks ahead and the gather (step 1) one block ahead, issued right after the V^T store, so the
// vm latency of the position -> row chain overlaps softmax / PV, and the QK key loads hit lines the gather fetched.
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
// Optional (a policy without them compiles to the code above):
//   static __device__ void query16(const Args&, unsigned row, unsigned head, unsigned chunk, float (&x)[16])
//                                 replaces query(): dims 16 chunk .. + 15 of a head, computed by the policy (e.g. RoPE)
//   static __device__ const std::uint16_t* query_op(const Args&, unsigned row)
//                                 replaces query() and query16(): the row's queries already in the operand type, [H][D],
//                                 16-B aligned, loaded as fragments with no conversion. Same bits as query() when the
//                                 producer stored tile::to_operand<kOp>(x) of the FP32 values (GLM: the absorb GEMM
//                                 epilogue); halves the query bytes written and read
//   static constexpr tile::Operand kOutOp; static __device__ std::uint16_t* out_op(const Args&, unsigned row)
//                                 replaces out(): the row's normalised outputs stored as tile::to_operand<kOutOp>(o),
//                                 [H][D] u16 (a row without keys: zeros); half the output bytes, for a consumer that
//                                 rounds its operand the same way (GLM: the FAST MLA out GEMM on FP16 WMMA)
//   static __device__ void emit_state(const Args&, unsigned row, unsigned head, float m, float l)
//   static __device__ void emit(const Args&, unsigned row, unsigned head, unsigned dim, float o)
//                                 replace out(): the raw online state (m in log2 units of the scaled score, l, and
//                                 the accumulator relative to m) for an engine epilogue (sink term, inverse RoPE);
//                                 a row without keys then writes nothing
// Launch: flash_rows_kernel<P, T><<<rows, Geometry<P::kHeads, P::kDim, T>::kThreads, 0, stream>>>(args).
#include "attn_core/flash_tile.hpp"
#include "strix/env_arms.hpp"
#include "strix/env_flags.hpp"

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

template <class P>
concept PolicyQuery16 = requires(const typename P::Args& a, float (&x)[16]) { P::query16(a, 0U, 0U, 0U, x); };
template <class P>
concept PolicyQueryOp = requires(const typename P::Args& a, const std::uint16_t*& p) { p = P::query_op(a, 0U); };
template <class P>
concept PolicyOutOp = requires(const typename P::Args& a, std::uint16_t*& p) { p = P::out_op(a, 0U); P::kOutOp; };
template <class P>
concept PolicyRawState = requires(const typename P::Args& a) { P::emit_state(a, 0U, 0U, 0.0F, 0.0F); P::emit(a, 0U, 0U, 0U, 0.0F); };

template <class P, unsigned T, bool Pf = true>
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
    float* out = nullptr;
    std::uint16_t* out16 = nullptr;
    if constexpr (PolicyRawState<P>) {
        if (n == 0) return;
    } else if constexpr (PolicyOutOp<P>) {
        out16 = P::out_op(a, r);
        if (n == 0) {  // zeros, as out() (the operand image of +0 is 0)
            for (unsigned x = tid; x < H * D; x += G::kThreads) out16[x] = 0U;
            return;
        }
    } else {
        out = P::out(a, r);
        if (n == 0) {  // uniform: nothing attended (not produced by the engines; defined as zeros)
            for (unsigned x = tid; x < H * D; x += G::kThreads) out[x] = 0.0F;
            return;
        }
    }
    Frag qf[G::kKs];
    if constexpr (PolicyQueryOp<P>) {
        const std::uint16_t* qr = P::query_op(a, r) + static_cast<std::size_t>(mt * 16U + ln) * D + qd * G::kQd;
#pragma unroll
        for (unsigned k = 0; k < G::kKs; ++k) qf[k] = tile::frag32(qr + 16U * k);
    } else if constexpr (PolicyQuery16<P>) {
#pragma unroll
        for (unsigned k = 0; k < G::kKs; ++k) {
            float x[16];
            P::query16(a, r, mt * 16U + ln, (qd * G::kQd) / 16U + k, x);
            qf[k] = tile::frag_from_f32<O>(x);
        }
    } else {
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
    // Pf (pipelined loads, same operands and order = bitwise): the list positions of block b + 2 and the gathered
    // rows of block b + 1 are issued during block b, so the gather never waits on the position load and, when the
    // value row is the key row (MLA latent), the QK key loads of block b + 1 find the lines already near the CU.
    U4 gv[G::kGPass][4];
    unsigned gq[4], kp[G::kNt], kq[G::kNt];  // gather positions of block b + 1; QK positions of blocks b, b + 1
    const auto gather = [&](unsigned j0) {
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const std::uint16_t* vr = P::value_row(a, r, P::key(a, r, mask::list_clamp(j0 + 4U * gt + i, n)));
#pragma unroll
            for (unsigned p = 0; p < G::kGPass; ++p) gv[p][i] = *reinterpret_cast<const U4*>(vr + 8U * (gd + p * G::kDg));
        }
    };
    const auto positions = [&](unsigned j0) {  // clamped: in bounds past the last block
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) gq[i] = P::key(a, r, mask::list_clamp(j0 + 4U * gt + i, n));
#pragma unroll
        for (unsigned tt = 0; tt < G::kNt; ++tt) kq[tt] = P::key(a, r, mask::list_clamp(j0 + 16U * tt + ln, n));
    };
    if constexpr (Pf) {
        gather(0U);
#pragma unroll
        for (unsigned tt = 0; tt < G::kNt; ++tt) kp[tt] = P::key(a, r, mask::list_clamp(16U * tt + ln, n));
        positions(T);
    }
    for (unsigned b = 0; b < blocks; ++b) {
        const unsigned j0 = b * T;
        // 1. gather loads (Pf: issued one block earlier; else in flight during QK)
        if constexpr (!Pf) gather(j0);
        // 2. QK partials of this wave's quarter
#pragma unroll
        for (unsigned tt = 0; tt < G::kNt; ++tt) {
            const unsigned j = j0 + 16U * tt + ln;
            const unsigned pos = Pf ? kp[tt] : P::key(a, r, mask::list_clamp(j, n));
            const std::uint16_t* kr = P::key_row(a, r, pos) + qd * G::kQd;
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
        if constexpr (Pf) {
            if (b + 1U < blocks) {  // uniform: rows of block b + 1 from the positions loaded last block, then b + 2's
#pragma unroll
                for (unsigned i = 0; i < 4; ++i) {
                    const std::uint16_t* vr = P::value_row(a, r, gq[i]);
#pragma unroll
                    for (unsigned p = 0; p < G::kGPass; ++p) gv[p][i] = *reinterpret_cast<const U4*>(vr + 8U * (gd + p * G::kDg));
                }
#pragma unroll
                for (unsigned tt = 0; tt < G::kNt; ++tt) kp[tt] = kq[tt];
                positions(j0 + 2U * T);
            }
        }
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
    if constexpr (PolicyRawState<P>) {
        if (sg == 0) P::emit_state(a, r, sh, st.m, st.l);
#pragma unroll
        for (unsigned c = 0; c < G::kDt; ++c)
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) P::emit(a, r, mt * 16U + 2U * e + hf, qd * G::kQd + 16U * c + ln, o[c][e]);
        return;
    }
    if (sg == 0) al[tile::row_slot(sh)] = tile::finalize_scale(st);
    __syncthreads();
    float iv[8];
    tile::row_scalars(al, mt, lane, iv);
#pragma unroll
    for (unsigned c = 0; c < G::kDt; ++c)
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
            const std::size_t x = static_cast<std::size_t>(mt * 16U + 2U * e + hf) * D + qd * G::kQd + 16U * c + ln;
            if constexpr (PolicyOutOp<P>) out16[x] = tile::to_operand<P::kOutOp>(o[c][e] * iv[e]);
            else out[x] = o[c][e] * iv[e];
        }
}

// Pipelined loads (Pf, bitwise: same operands, same order), shared by every engine that binds this core.
// Auto by the launch's longest key list (host bound, max_keys): Pf hides latency only when there are blocks to overlap,
// and its extra VGPRs (gfx1151 VGPR Pf on / off: T 16 210 / 253, T 32 238 / 199) cost occupancy on short lists. Measured receipts
// (fp32glm3r, ABAB, bitwise in every case): GLM T 16 lists <= 2,051 flash ~-19 %; DS41 T 32 lists 640 -4.3..-4.7 %,
// 228 +3.3 %, <= 128 +12..15 % -> on from kPfMinKeys. STRIX_ATTN_FLASH_PF=1 / 0 forces it; resolved once per in-process
// arm (env_arms).
inline constexpr unsigned kPfMinKeys = 320;
inline int flash_pf_mode() {  // -1 auto, 0 off, 1 on
    static strix::env_arms::Resolved<int> mode;
    return mode.get([] {
        const char* p = strix::envcfg::text("STRIX_ATTN_FLASH_PF");
        if (p == nullptr || *p == '\0' || std::strcmp(p, "auto") == 0) return -1;
        return strix::envcfg::flag("STRIX_ATTN_FLASH_PF", true) ? 1 : 0;
    });
}
inline bool flash_pf_enabled(unsigned max_keys = ~0U) {
    const int m = flash_pf_mode();
    return m < 0 ? max_keys >= kPfMinKeys : m == 1;
}
inline const char* flash_pf_reason(unsigned max_keys = ~0U) {
    const int m = flash_pf_mode();
    if (m >= 0) return m == 1 ? "on (STRIX_ATTN_FLASH_PF=1)" : "off (STRIX_ATTN_FLASH_PF=0)";
    return max_keys >= kPfMinKeys ? "on (auto: key lists >= 320)" : "off (auto: key lists < 320, nothing to overlap)";
}
template <class P, unsigned T> inline void launch_flash_rows(const typename P::Args& a, unsigned grid, hipStream_t s, unsigned max_keys = ~0U) {
    constexpr unsigned threads = Geometry<P::kHeads, P::kDim, T>::kThreads;
    static std::atomic<int> shown{-1};  // print the decision (and its reason) whenever it changes
    const bool pf = flash_pf_enabled(max_keys);
    if (shown.exchange(pf ? 1 : 0, std::memory_order_relaxed) != (pf ? 1 : 0))
        std::fprintf(stderr, "attn_core flash T %u: pipelined loads %s\n", T, flash_pf_reason(max_keys));
    if (pf) flash_rows_kernel<P, T, true><<<grid, threads, 0, s>>>(a);
    else flash_rows_kernel<P, T, false><<<grid, threads, 0, s>>>(a);
}

}  // namespace strix::attn_core::sparse
