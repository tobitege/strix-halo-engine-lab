#pragma once
// Cooperative paired-key flash schedule. Key decoding, row access, scale and final combine are supplied by P.
// The schedule retains quarter-score reduction, f16 exponent bands and the two ordered PV rescale forms.
#include "flash_core.hpp"

namespace strix::attn_core {

template <class P, unsigned Cpb, unsigned H>
__device__ __forceinline__ void banded_flash_prefill(typename P::Args v) {
    constexpr unsigned kHeadDim = P::kDim, kChunk = P::kChunk, kMaxChunks = P::kMaxChunks;
    constexpr unsigned kAttnGroups = P::kGroups, kFastKeys = 32, kFastPairs = 16, kFastQStride = kHeadDim / 2 + 4;
    constexpr int kFastPScale = P::kProbabilityShift;
    using Raw = typename P::Raw;
    using fhalf16 = Half16;
    using ffloat8 = tile::Acc;
    using fhalf2 = Half2;
    using fu8 = Word8;
    static_assert(H == 16U || H == 32U);
    constexpr unsigned MT = H / 16U, HW = H / 8U;   // head row tiles; heads per wave in the softmax and the combine
    // the key tile (32 KB), or before the first tile the Q staging (H rows of kFastQStride dwords)
    constexpr unsigned kt_words = kFastPairs * kHeadDim > H * kFastQStride ? kFastPairs * kHeadDim : H * kFastQStride;
    __shared__ __attribute__((aligned(16))) unsigned kt[kt_words];
    __shared__ float spart[4][H][kFastKeys];
    __shared__ __attribute__((aligned(16))) float pf[H][kFastKeys];   // the tile's probabilities (FP32; banded tiles)
    __shared__ __attribute__((aligned(16))) unsigned pt[H][kFastPairs]; // the same as f16 pairs times 2^(x_k - hi) (one band)
    __shared__ int band_s[2];                                            // the tile's largest shift, and whether it needs bands
    __shared__ float scl[H], kfac[kFastKeys], qfac[H];
    __shared__ int kexp[kFastKeys];
    __shared__ std::uint32_t cidx[Cpb * kChunk];
    __shared__ unsigned last_s;
    const unsigned ablate = P::ablation();
    const unsigned r = blockIdx.z, tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U, lo = lane & 15U, half = lane >> 4U;
    const unsigned hb = blockIdx.y * H;
    unsigned nw;
    const unsigned n = P::visible(v, r, nw), chunks = (n + kChunk - 1) / kChunk, blocks = (chunks + Cpb - 1) / Cpb;
    if (P::max_visible(v) != 0U && n > P::max_visible(v)) __builtin_trap();
    const unsigned c0 = blockIdx.x * Cpb;
    if (c0 >= chunks) return;
    const unsigned j0 = c0 * kChunk, end = n - j0 < Cpb * kChunk ? n : j0 + Cpb * kChunk;   // the block's keys [j0, end)
    const std::uint64_t p = P::position(v, r);
    const unsigned first_slot = static_cast<unsigned>((p + 1U - nw) % P::capacity(v));
    for (unsigned t = tid; t < Cpb * kChunk; t += 256U)
        if (j0 + t < end && j0 + t >= nw) cidx[t] = P::index(v, r, j0 + t - nw);
    // Q: wave w stages heads HW w .. (lane: 16 dims, RoPE in FP32) as f16 pairs, rows padded by 16 bytes
#pragma unroll
    for (unsigned hh = 0; hh < HW; ++hh) {
        const unsigned h = HW * wave + hh;
        float q[16];
        P::query(v, r, hb + h, lane, ablate, q);
        // a head whose largest |q| is above 2^14 is stored times 2^-k (qfac = 2^k undoes it on its scores in FP32)
        float amax = 0.0F;
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(q[i]));
        amax = P::wave_max(amax);
        const int k = amax > 16384.0F && __builtin_isfinite(amax) ? P::log2_ceil(amax * (1.0F / 16384.0F)) : 0;
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) q[i] = ldexpf(q[i], -k);
        if (lane == 0U) qfac[h] = ldexpf(1.0F, k);
        unsigned w[8];
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) w[i] = __builtin_bit_cast(unsigned, fhalf2{static_cast<_Float16>(q[2 * i]), static_cast<_Float16>(q[2 * i + 1])});
        uint4* dst = reinterpret_cast<uint4*>(kt + h * kFastQStride + 8 * lane);
        dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
        dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
    }
    __syncthreads();
    // QK roles. H 16: key tile kq = wave & 1, dim quarter dq = wave >> 1. H 32: head tile mq = wave & 1 (both key tiles),
    // dim quarter dq = wave >> 1. A operands (the wave's head tile x dim quarter) in registers.
    const unsigned kq = wave & 1U, dq = wave >> 1U, mq = H == 32U ? (wave & 1U) : 0U;
    fhalf16 qa[8];
#pragma unroll
    for (unsigned s = 0; s < 8; ++s) {
        const uint4* src = reinterpret_cast<const uint4*>(kt + (16U * mq + lo) * kFastQStride + (128U * dq + 16U * s) / 2U);
        const uint4 x0 = src[0], x1 = src[1];
        qa[s] = __builtin_bit_cast(fhalf16, fu8{x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w});
    }
    // loads of a tile: thread -> pair tid / 16 (keys t0 + 2 pair, + 1), dims 32 part ..
    const unsigned pr = tid >> 4U, part = tid & 15U;
    const auto load = [&](unsigned j, Raw& key) {
        P::load(v, j, end, nw, first_slot, part, j >= nw && j < end ? cidx[j - j0] : 0U, ablate, key);
    };
    __syncthreads();   // cidx visible; every wave holds its Q operands (the tile may be overwritten)
    Raw ra{}, rb{};
    load(j0 + 2U * pr, ra);
    load(j0 + 2U * pr + 1U, rb);
    float m[MT], l[MT];   // softmax roles: heads HW wave + 2 i + half
    ffloat8 acc[MT][4];
#pragma unroll
    for (unsigned i = 0; i < MT; ++i) {
        m[i] = -__builtin_inff(); l[i] = 0.0F;
#pragma unroll
        for (unsigned t = 0; t < 4; ++t) acc[i][t] = ffloat8{0, 0, 0, 0, 0, 0, 0, 0};
    }
#pragma unroll 1
    for (unsigned t0 = j0; t0 < end; t0 += kFastKeys) {
        const unsigned cnt = end - t0 < kFastKeys ? end - t0 : kFastKeys;
        P::stage(kt, pr, part, ra, rb, t0, nw, cnt, kexp, kfac, ablate);
        __syncthreads();
        if (t0 + kFastKeys < end) { load(t0 + kFastKeys + 2U * pr, ra); load(t0 + kFastKeys + 2U * pr + 1U, rb); }
        // QK: the wave's key tiles (H 16: kq; H 32: both), its head tile mq, dims 128 dq + 16 s ..
#pragma unroll
        for (unsigned kk = 0; kk < MT; ++kk) {
            const unsigned ktile = H == 32U ? kk : kq;
            const unsigned key = 16U * ktile + lo, kp = key >> 1U, sel = (key & 1U) != 0U ? 0x07060302U : 0x05040100U;
            ffloat8 d = ffloat8{0, 0, 0, 0, 0, 0, 0, 0};
            if ((ablate & 4U) == 0U) {
#pragma unroll
                for (unsigned s = 0; s < 8; ++s) {
                    const unsigned d0 = 128U * dq + 16U * s;
                    unsigned w[16];
#pragma unroll
                    for (unsigned v4 = 0; v4 < 4; ++v4) {
                        const uint4 x = *reinterpret_cast<const uint4*>(kt + pair_slot<kHeadDim>(kp, d0 + 4U * v4));
                        w[4 * v4] = x.x; w[4 * v4 + 1] = x.y; w[4 * v4 + 2] = x.z; w[4 * v4 + 3] = x.w;
                    }
                    fu8 b;
#pragma unroll
                    for (unsigned j = 0; j < 8; ++j) b[j] = __builtin_amdgcn_perm(w[2 * j + 1], w[2 * j], sel);
                    d = tile::mma<tile::Operand::f16>(__builtin_bit_cast(tile::Frag, qa[s]), __builtin_bit_cast(tile::Frag, b), d);
                }
            }
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) spart[dq][16U * mq + 2 * e + half][16U * ktile + lo] = d[e];
        }
        __syncthreads();
#pragma unroll
        for (unsigned i = 0; i < MT; ++i) {   // softmax of head HW wave + 2 i + half: lane lo holds keys 2 lo, 2 lo + 1
            const unsigned h = HW * wave + 2U * i + half, k0 = 2U * lo;
            float s[2];
#pragma unroll
            for (unsigned j = 0; j < 2; ++j) {
                const unsigned k = k0 + j;
                const float x = (spart[0][h][k] + spart[1][h][k]) + (spart[2][h][k] + spart[3][h][k]);
                s[j] = k < cnt ? (x * kfac[k]) * qfac[h] : -__builtin_inff();
            }
            float probabilities[2];
            const float sc = online_step<PairSoftmax<P>>(s, m[i], l[i], probabilities);
            const float p0 = probabilities[0], p1 = probabilities[1];
            // the tile's key shifts x (7 compressed, ek window): one band (largest - smallest <= 8, the usual tile) packs
            // f16 probabilities times 2^(x_k - hi) here; wider tiles keep FP32 probabilities for the banded passes below
            const bool v0 = k0 < cnt, v1 = k0 + 1U < cnt;
            int xh = max(v0 ? kexp[k0] : -1, v1 ? kexp[k0 + 1U] : -1), xl = min(v0 ? kexp[k0] : 1 << 20, v1 ? kexp[k0 + 1U] : 1 << 20);
#pragma unroll
            for (unsigned d = 1; d < 16; d *= 2) { xh = max(xh, __shfl_xor(xh, static_cast<int>(d), 32)); xl = min(xl, __shfl_xor(xl, static_cast<int>(d), 32)); }
            const bool banded = xh - xl > 8;
            if (banded) { pf[h][k0] = p0; pf[h][k0 + 1U] = p1; }
            else pt[h][lo] = __builtin_bit_cast(unsigned, fhalf2{static_cast<_Float16>(v0 ? ldexpf(p0, kexp[k0] - xh + kFastPScale) : 0.0F),
                                                                 static_cast<_Float16>(v1 ? ldexpf(p1, kexp[k0 + 1U] - xh + kFastPScale) : 0.0F)});
            if (lo == 0U) scl[h] = sc;
            if (tid == 0U) { band_s[0] = xh; band_s[1] = banded ? 1 : 0; }
        }
        __syncthreads();
        // PV: wave w owns dims 64 w + 16 t + lo of every head tile; rows = heads 16 i + 2 e + half. The stored keys carry
        // 2^-x_k (x = 7 compressed, ek window). Every f16 probability also carries 2^15: p * 2^(x_k - S + 15) <= 32768 for
        // the pass's top shift S, and f16 rounding takes a probability to zero only when p <= 2^(S - x_k - 40) of the
        // running softmax maximum (2^-40 at the top of a band, 2^-32 at its bottom) - the declared limit of this
        // approximation. Each product tile is formed from zero and added in FP32: acc = acc * sc + pv * 2^(S - 15) (the
        // accumulator is never scaled up, so it cannot overflow where the exact sum does not). One band (the usual tile,
        // largest - smallest shift <= 8): one pass with S = hi on the packed probabilities. Wider tiles: one pass per
        // band (S - 8 <= x_k <= S) on FP32 probabilities. Bands are uniform across the block.
        if (band_s[1] == 0) {
            const float up = ldexpf(1.0F, band_s[0] - kFastPScale);
#pragma unroll
            for (unsigned i = 0; i < MT; ++i)
#pragma unroll
                for (unsigned t = 0; t < 4; ++t) {   // one product tile at a time (one extra accumulator live)
                    ffloat8 pv = ffloat8{0, 0, 0, 0, 0, 0, 0, 0};
                    if ((ablate & 8U) == 0U) {
                        const unsigned dim = 64U * wave + 16U * t + lo;
#pragma unroll
                        for (unsigned ks = 0; ks < 2; ++ks) {
                            const uint4* ps = reinterpret_cast<const uint4*>(&pt[16U * i + lo][8U * ks]);
                            const uint4 x0 = ps[0], x1 = ps[1];
                            const fhalf16 a = __builtin_bit_cast(fhalf16, fu8{x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w});
                            fu8 b;
#pragma unroll
                            for (unsigned j = 0; j < 8; ++j) b[j] = kt[pair_slot<kHeadDim>(8U * ks + j, dim)];
                            pv = tile::mma<tile::Operand::f16>(__builtin_bit_cast(tile::Frag, a), __builtin_bit_cast(tile::Frag, b), pv);
                        }
                    }
#pragma unroll
                    for (unsigned e = 0; e < 8; ++e) acc[i][t][e] = __builtin_fmaf(pv[e], up, acc[i][t][e] * scl[16U * i + 2 * e + half]);
                }
            __syncthreads();
            continue;
        }
        int hi = -1;
        for (unsigned k = 0; k < cnt; ++k) hi = max(hi, kexp[k]);
#pragma unroll
        for (unsigned i = 0; i < MT; ++i)
#pragma unroll
            for (unsigned t = 0; t < 4; ++t)
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) acc[i][t][e] *= scl[16U * i + 2 * e + half];
#pragma unroll 1
        while (hi >= 0 && (ablate & 8U) == 0U) {
            const int band = hi, floor = hi - 8;
            const float up = ldexpf(1.0F, band - kFastPScale);
#pragma unroll
            for (unsigned i = 0; i < MT; ++i)
#pragma unroll
                for (unsigned t = 0; t < 4; ++t) {   // one product tile at a time, k-steps rolled: few live values (rare path)
                    ffloat8 pv = ffloat8{0, 0, 0, 0, 0, 0, 0, 0};
                    const unsigned dim = 64U * wave + 16U * t + lo;
#pragma unroll 1
                    for (unsigned ks = 0; ks < 2; ++ks) {
                        fhalf16 a;
#pragma unroll
                        for (unsigned j = 0; j < 16; ++j) {
                            const unsigned k = 16U * ks + j;
                            const int x = kexp[k];
                            a[j] = static_cast<_Float16>(k < cnt && x <= band && x >= floor ? ldexpf(pf[16U * i + lo][k], x - band + kFastPScale) : 0.0F);
                        }
                        fu8 b;
#pragma unroll
                        for (unsigned j = 0; j < 8; ++j) b[j] = kt[pair_slot<kHeadDim>(8U * ks + j, dim)];
                        pv = tile::mma<tile::Operand::f16>(__builtin_bit_cast(tile::Frag, a), __builtin_bit_cast(tile::Frag, b), pv);
                    }
#pragma unroll
                    for (unsigned e = 0; e < 8; ++e) acc[i][t][e] = __builtin_fmaf(pv[e], up, acc[i][t][e]);
                }
            int next = -1;
            for (unsigned k = 0; k < cnt; ++k) if (kexp[k] < floor) next = max(next, kexp[k]);
            hi = next;
        }
        __syncthreads();
    }
    // the block's partial (slot blockIdx.x): (m, l) of the softmax heads; acc[i][t][e] = head 16 i + 2 e + half, dim 64 wave + 16 t + lo
#pragma unroll
    for (unsigned i = 0; i < MT; ++i) {
        if (lo == 0U) {
            float* pp = P::partial(v) + ((static_cast<std::size_t>(r) * P::heads(v) + hb + HW * wave + 2U * i + half) * kMaxChunks + blockIdx.x) * (2 + kHeadDim);
            pp[0] = m[i]; pp[1] = l[i];
        }
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
            float* pp = P::partial(v) + ((static_cast<std::size_t>(r) * P::heads(v) + hb + 16U * i + 2U * e + half) * kMaxChunks + blockIdx.x) * (2 + kHeadDim) + 2U + 64U * wave + lo;
#pragma unroll
            for (unsigned t = 0; t < 4; ++t) pp[16U * t] = acc[i][t][e];
        }
    }
    if (P::tickets(v) == nullptr) return;
    __threadfence();
    __syncthreads();
    // tickets: one counter per (row, 8-head group); a block of H heads counts on its first group's
    std::uint32_t* ticket = P::tickets(v) + r * kAttnGroups + blockIdx.y * MT;
    if (tid == 0U) last_s = atomicAdd(ticket, 1U) == blocks - 1U ? 1U : 0U;
    __syncthreads();
    if (last_s == 0U) return;
    __threadfence();
#pragma unroll 1
    for (unsigned hh = 0; hh < HW; ++hh) P::combine(v, r, hb + HW * wave + hh, lane, blocks);
    if (tid == 0U) *ticket = 0U;
}

}  // namespace strix::attn_core
