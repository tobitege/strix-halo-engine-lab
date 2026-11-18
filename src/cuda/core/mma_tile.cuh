// Shared CUDA core: FP16 tensor-core tile mechanics for FAST forms (sm_80+, built for sm_120). One mechanism, two drivers:
//   (1) register-B warp GEMM (warp_gemm_k32 + Int8Codes / Int4Codes traits, perm8 K order, below), and
//   (2) the CTA-level double-buffered smem tile (Tile<NT8, WeightFmt>, at the end): weights staged to smem as exact f16 codes,
//       B through ldmatrix; WeightFmt (Q40Blocks = 18-byte Q4_0 blocks) is the weight-format trait of its load / store half.
// Both share the exact unpack (i8/u4 magic numbers) and the per-32-K-block FP32 scale epilogue (block_fma).
// A = activations (FP32 in global, FP16 in smem), B = quantised weights (int8 or int4 codes + one FP32 scale per 32 K),
// accumulation FP32 (mma.sync m16n8k16 f16). Codes are exact in FP16; the scale is applied to the FP32 block partial:
//   acc += scale * (sum over the 32-K block of act * code)     -> the weights are never rounded.
//
// K permutation "perm8" (A and B agree, no runtime cost). Inside every 32-wide K block (the scale block) and for MMA step
// j in {0,1}, the thread t (lane & 3) holds the actual K indices 8t + 4j + {0,1} in MMA slots 2t, 2t+1 and 8t + 4j + {2,3} in
// slots 8 + 2t, 9 + 2t. So one thread's 8 consecutive codes of a weight row (uint2 of int8 / one uint32 of nibbles)
// feed both k16 steps of the block, and A is stored to smem in permuted order by stage_k4().
//
// Fragment map (PTX m16n8k16, g = lane >> 2, t = lane & 3): a0 (row g, slots 2t..2t+1), a1 (row g+8), a2 (row g, slots 8+2t..),
// a3 (row g+8); b0 (slots 2t..2t+1, col g), b1 (slots 8+2t.., col g); c0,c1 (row g, cols 2t, 2t+1), c2,c3 (row g+8).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "ptx.cuh"
#include "../weight/arch/sm120_sources.cuh"

namespace strix::cuda::core::mma {

constexpr unsigned kPadHalves = 8;  // smem row pad (16 B): ldmatrix rows hit distinct banks when pitch/8 is odd

// FP32 x4 (actual K = 4m .. 4m+3 of a 32-block, m in 0..7) -> FP16, stored in perm8 order into the 32-half block at `blk`.
__device__ __forceinline__ void stage_k4(__half* blk, unsigned m, float4 v) {
    const unsigned t = m >> 1U, j = m & 1U;
    *reinterpret_cast<__half2*>(blk + j * 16U + 2U * t) = __floats2half2_rn(v.x, v.y);
    *reinterpret_cast<__half2*>(blk + j * 16U + 8U + 2U * t) = __floats2half2_rn(v.z, v.w);
}

// A 16x16 fragment (16 rows, 16 halves = one MMA step) at `tile` (row 0, first slot), row pitch in halves.
__device__ __forceinline__ void ldsm_a(unsigned (&a)[4], const __half* tile, unsigned pitch_halves, unsigned lane) {
    ptx::ldsm_x4(a, tile + (lane & 15U) * pitch_halves + (lane >> 4U) * 8U);
}

// Exact integer -> half2 unpack (magic-number trick: 0x6400 | u8 = 1024 + u8).
__device__ __forceinline__ unsigned i8_pair_to_half2(unsigned w, unsigned sel) {  // sel: 0x4140 (bytes 0,1) or 0x4342 (bytes 2,3)
    const unsigned v = __byte_perm(w ^ 0x80808080U, 0x64646464U, sel);  // {1024 + (c0^0x80), 1024 + (c1^0x80)}
    const __half2 r = __hsub2(*reinterpret_cast<const __half2*>(&v), __float2half2_rn(1152.0F));
    return *reinterpret_cast<const unsigned*>(&r);
}
__device__ __forceinline__ unsigned u4_pair_to_half2(unsigned w, unsigned shift) {  // nibbles at bit shift, shift + 4; offset 8
    const unsigned v = (((w >> shift) & 0xFU) | (((w >> (shift + 4U)) & 0xFU) << 16U)) | 0x64006400U;
    const __half2 r = __hsub2(*reinterpret_cast<const __half2*>(&v), __float2half2_rn(1032.0F));
    return *reinterpret_cast<const unsigned*>(&r);
}
// Same, for u already shifted so the two nibbles sit at bits 0 and 8 (Q4_0 byte pairs: low / high halves of the block).
__device__ __forceinline__ unsigned u4_bytes_to_half2(unsigned u) {
    return ::strix::cuda::weight::Q40F16Source::unpack_pair(u);
}

// Per-32-K-block FP32 scale epilogue: acc += scale * (block partial) for one c fragment (s0 / s1: scales of columns 2t, 2t+1).
__device__ __forceinline__ void block_fma(float (&acc)[4], const float (&blk)[4], float s0, float s1) {
    acc[0] = __fmaf_rn(s0, blk[0], acc[0]);
    acc[1] = __fmaf_rn(s1, blk[1], acc[1]);
    acc[2] = __fmaf_rn(s0, blk[2], acc[2]);
    acc[3] = __fmaf_rn(s1, blk[3], acc[3]);
}

// Code traits: W words hold the thread's 8 consecutive codes (k 8t .. 8t+7 of the 32-block) of one weight column n = g.
struct Int8Codes {  // signed int8 (Q8_0 style), 2 words, byte i of word 0 = k 8t + i, word 1 = k 8t + 4 + i
    static constexpr unsigned W = 2;
    __device__ static __forceinline__ void b_frag(const unsigned (&w)[W], unsigned step, unsigned& b0, unsigned& b1) {
        b0 = i8_pair_to_half2(w[step], 0x4140U);
        b1 = i8_pair_to_half2(w[step], 0x4342U);
    }
};
struct Int4Codes {  // unsigned nibbles with offset 8 (Q4_0 style value = nibble - 8), 1 word, nibble i = k 8t + i
    static constexpr unsigned W = 1;
    __device__ static __forceinline__ void b_frag(const unsigned (&w)[W], unsigned step, unsigned& b0, unsigned& b1) {
        b0 = u4_pair_to_half2(w[0], 16U * step);
        b1 = u4_pair_to_half2(w[0], 16U * step + 8U);
    }
};

// One 32-K block for MT m16 tiles x one n8 tile. a[mt][step] are the A fragments (ldsm_a of the 2 steps of the block),
// w the thread's codes of column g, s0/s1 the FP32 scales of the columns 2t and 2t+1 (c-fragment columns).
template <unsigned MT, class Codes>
__device__ __forceinline__ void warp_gemm_k32(float (&acc)[MT][4], const unsigned (&a)[MT][2][4], const unsigned (&w)[Codes::W],
    float s0, float s1) {
    unsigned b[2][2];
    Codes::b_frag(w, 0U, b[0][0], b[0][1]);
    Codes::b_frag(w, 1U, b[1][0], b[1][1]);
#pragma unroll
    for (unsigned mt = 0; mt < MT; ++mt) {
        float blk[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        ptx::mma_f16(blk, a[mt][0][0], a[mt][0][1], a[mt][0][2], a[mt][0][3], b[0][0], b[0][1]);
        ptx::mma_f16(blk, a[mt][1][0], a[mt][1][1], a[mt][1][2], a[mt][1][3], b[1][0], b[1][1]);
        block_fma(acc[mt], blk, s0, s1);
    }
}


// Weight-format trait of the smem tile: Q4_0-style blocks, 18 B (2-byte aligned): fp16 d, 16 bytes; byte j low nibble = element j,
// high nibble = element j + 16. Regs = one block as 9 u16 (loaded for all threads, two threads per output row); store() turns the
// nibbles into exact f16 codes (code = nibble - 8) of elements kh * 16 .. +15 and the scale into FP32 (kh == 0 only).
using Q40Blocks = ::strix::cuda::weight::Q40TileSource;

// CTA tile: BM = 128 rows x BN = 16 * NT8 outputs per CTA of 256 threads (8 warps: 4 along M x 2 along N, warp tile 32 x 8 * NT8).
template <int NT8, class WeightFmt = Q40Blocks>
struct Tile {
    static constexpr int kBM = 128, kBK = 32, kLd = 40;  // smem row stride in halves (80 B: conflict-free ldmatrix)
    static constexpr int kBN = 16 * NT8;
    static_assert(kBN == 128 || kBN == 64, "weight loader: two threads per output row");
    struct Smem {
        __half a[2][kBM * kLd];
        __half w[2][kBN * kLd];
        float s[2][kBN];
    };
    struct Regs { float4 a[4]; typename WeightFmt::Regs w; };

    // Global -> registers for K step kb. x: A row base of the stream (row stride ld floats, K offset applied by caller);
    // wq: weight block 0 of the stream for output 0 (row stride row_bytes); n0: first output of this CTA.
    __device__ static __forceinline__ void load(Regs& r, const float* x, std::uint64_t ld, unsigned m0, unsigned rows, int kb,
                                                const std::uint8_t* wq, std::uint64_t row_bytes, unsigned n0) {
        const unsigned t = threadIdx.x, row = t >> 1U, kh = t & 1U;
        const float* p = x + static_cast<std::uint64_t>(min(m0 + row, rows - 1U)) * ld + kb * kBK + kh * 16U;
#pragma unroll
        for (int i = 0; i < 4; ++i) r.a[i] = *reinterpret_cast<const float4*>(p + 4 * i);
        const unsigned nl = (t % (kBN * 2U)) >> 1U;
        WeightFmt::load(r.w, wq + static_cast<std::uint64_t>(n0 + nl) * row_bytes + static_cast<std::uint64_t>(kb) * WeightFmt::kBlockBytes);
    }
    // Registers -> shared (FP32 -> f16 for A; nibbles -> exact f16 codes for W; scale -> FP32 for the epilogue).
    __device__ static __forceinline__ void store(const Regs& r, Smem& sm, int buf) {
        const unsigned t = threadIdx.x, row = t >> 1U, kh = t & 1U;
        uint4 pa[2];
        unsigned* pw = reinterpret_cast<unsigned*>(pa);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            __half2 lo = __floats2half2_rn(r.a[i].x, r.a[i].y), hi = __floats2half2_rn(r.a[i].z, r.a[i].w);
            pw[2 * i] = *reinterpret_cast<unsigned*>(&lo);
            pw[2 * i + 1] = *reinterpret_cast<unsigned*>(&hi);
        }
        uint4* da = reinterpret_cast<uint4*>(&sm.a[buf][row * kLd + kh * 16U]);
        da[0] = pa[0];
        da[1] = pa[1];
        // W: thread t -> output t / 2, half kh: elements kh * 16 .. +15 are nibble kh of bytes 0..15.
        if (t < kBN * 2U) {
            const unsigned nl = t >> 1U;
            WeightFmt::store(r.w, &sm.w[buf][nl * kLd + kh * 16U], &sm.s[buf][nl], kh);
        }
    }
    // One K step (2 x k16) on shared buffer `buf`: partial = A * codes in fresh accumulators, acc += scale * partial.
    __device__ static __forceinline__ void compute(float (&acc)[2][NT8][4], const Smem& sm, int buf) {
        const unsigned lane = threadIdx.x & 31U, wid = threadIdx.x >> 5U, wm = wid & 3U, wn = wid >> 2U;
        unsigned af[2][2][4];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks)
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
                ptx::ldsm_x4(af[ks][mi], &sm.a[buf][(wm * 32U + mi * 16U + (lane & 15U)) * kLd + ks * 16U + (lane >> 4U) * 8U]);
#pragma unroll
        for (int jp = 0; jp < NT8 / 2; ++jp) {
            unsigned bf[2][4];
            const unsigned nrow = wn * (NT8 * 8U) + jp * 16U + ((lane >> 4U) * 8U) + (lane & 7U);
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
                ptx::ldsm_x4(bf[ks], &sm.w[buf][nrow * kLd + ks * 16U + ((lane >> 3U) & 1U) * 8U]);
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int j = jp * 2 + jj;
                const float2 sc = *reinterpret_cast<const float2*>(&sm.s[buf][wn * (NT8 * 8U) + j * 8U + 2U * (lane & 3U)]);
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    float t[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
                    for (int ks = 0; ks < 2; ++ks)
                        ptx::mma_f16(t, af[ks][mi][0], af[ks][mi][1], af[ks][mi][2], af[ks][mi][3], bf[ks][2 * jj], bf[ks][2 * jj + 1]);
                    block_fma(acc[mi][j], t, sc.x, sc.y);
                }
            }
        }
    }
    // acc = sum over `kblocks` K steps of one stream. Ends with a barrier (shared buffers reusable by the next stream).
    __device__ static void run(float (&acc)[2][NT8][4], Smem& sm, const float* x, std::uint64_t ld, unsigned m0, unsigned rows,
                               const std::uint8_t* wq, std::uint64_t row_bytes, unsigned n0, int kblocks) {
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int j = 0; j < NT8; ++j) acc[mi][j][0] = acc[mi][j][1] = acc[mi][j][2] = acc[mi][j][3] = 0.0F;
        Regs r;
        load(r, x, ld, m0, rows, 0, wq, row_bytes, n0);
        store(r, sm, 0);
        __syncthreads();
        for (int kb = 0; kb < kblocks; ++kb) {
            const int cur = kb & 1;
            if (kb + 1 < kblocks) load(r, x, ld, m0, rows, kb + 1, wq, row_bytes, n0);
            compute(acc, sm, cur);
            if (kb + 1 < kblocks) store(r, sm, cur ^ 1);
            __syncthreads();
        }
    }
    // Tile-local output column / row of accumulator element e (0..3) of fragment (mi, j) for this thread.
    __device__ static __forceinline__ unsigned col(int j, int e) {
        return ((threadIdx.x >> 5U) >> 2U) * (NT8 * 8U) + j * 8U + 2U * (threadIdx.x & 3U) + (e & 1);
    }
    __device__ static __forceinline__ unsigned row(int mi, int e) {
        return ((threadIdx.x >> 5U) & 3U) * 32U + mi * 16U + ((threadIdx.x & 31U) >> 2U) + (e >> 1) * 8U;
    }
};

}  // namespace strix::cuda::core::mma
