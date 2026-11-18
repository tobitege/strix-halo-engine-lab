// sm_120 kernels of the deepseek41 CSA2 attention. Each kernel's arithmetic is its host reference
// (src/ds41_attention.cpp, f16_rows of src/ds41_layer.cpp): the same chains in the same order, the same xor
// butterflies and the same scalar helpers, so results match bit for bit. Built with --fmad=false (no contraction
// except the explicit fmaf of the declared order), -ftz=false, -prec-div=true, -prec-sqrt=true.
#include "strix/cuda/ds41_attention.hpp"

#include <cub/block/block_scan.cuh>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace strix::ds41::cuda::attn {
namespace {

constexpr unsigned kFull = 0xffffffffU;


__device__ __forceinline__ float warp_sum(float v) {  // butterfly: v[l] + v[l ^ d], d = 16..1
#pragma unroll
    for (unsigned d = 16; d != 0; d /= 2) v = v + __shfl_xor_sync(kFull, v, d);
    return v;
}
__device__ __forceinline__ float warp_max(float v, unsigned from) {
    for (unsigned d = from; d != 0; d /= 2) v = fmaxf(v, __shfl_xor_sync(kFull, v, d));
    return v;
}
// RMS norm of the warp's 512 vector (lane: dims 16l..16l+15), in place: (x * r) * w.
__device__ __forceinline__ void norm512(float (&x)[16], const float* w, unsigned lane) {
    float acc = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) acc = fmaf(x[i], x[i], acc);
    const float r = 1.0F / sqrtf(warp_sum(acc) / 512.0F + kNormEps);
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) x[i] = (x[i] * r) * w[16 * lane + i];
}
// lanes 28..31 own dims 448..511 = RoPE pairs 8(lane - 28) .. +7
__device__ __forceinline__ void rope512(float (&x)[16], const RopeRow& rope, unsigned lane, bool inverse) {
    if (lane < 28U) return;
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned pair = 8U * (lane - 28U) + i;
        rope_pair(x[2 * i], x[2 * i + 1], rope.c[pair], rope.s[pair], inverse);
    }
}

// ---- F16 rows ---------------------------------------------------------------------------------------------------------
// A warp per OW consecutive output rows of one job and a tile of up to 8 activation rows. 256-element segments of the
// weight rows are loaded as 16 bytes per lane (Batch segments in flight), transposed through the warp's shared buffer,
// and lane l takes elements l + 32k: its class chain in ascending order. Decode uses OW = 1 (more warps, more loads in
// flight), many rows OW = 4 (each activation load serves four outputs).
constexpr unsigned kF16RowTile = 8, kF16Warps = 8;

struct F16Launch {
    F16RowsView v;
    std::uint32_t end[4];  // cumulative output counts per job
};

template <unsigned OW, unsigned RT>
__global__ __launch_bounds__(kF16Warps * 32) void f16_rows_kernel(F16Launch L) {
    __shared__ uint4 stage[kF16Warps][OW][32];
    const unsigned lane = threadIdx.x & 31U, w = threadIdx.x >> 5U;
    const unsigned g = (blockIdx.x * kF16Warps + w) * OW;
    if (g >= L.end[L.v.job_count - 1U]) return;
    unsigned j = 0;
    while (g >= L.end[j]) ++j;
    const F16Job job = L.v.jobs[j];
    const unsigned o0 = g - (j == 0U ? 0U : L.end[j - 1U]);
    const unsigned r0 = blockIdx.y * RT, nr = min(RT, L.v.rows - r0);
    const uint4* wrow = reinterpret_cast<const uint4*>(job.weight + std::size_t{o0} * job.width);
    const unsigned row_u4 = job.width / 8U, xs = job.x_stride;
    const float* x = job.x + std::size_t{r0} * xs;
    float acc[OW][RT];
#pragma unroll
    for (unsigned o = 0; o < OW; ++o)
#pragma unroll
        for (unsigned r = 0; r < RT; ++r) acc[o][r] = 0.0F;
    const unsigned segs = job.width / 256U;
    const __half* hs = reinterpret_cast<const __half*>(&stage[w][0][0]);
    uint4 nxt[OW];
#pragma unroll
    for (unsigned o = 0; o < OW; ++o) nxt[o] = __ldg(wrow + std::size_t{o} * row_u4 + lane);
    for (unsigned sg = 0; sg < segs; ++sg) {
#pragma unroll
        for (unsigned o = 0; o < OW; ++o) stage[w][o][lane] = nxt[o];
        if (sg + 1U < segs) {
#pragma unroll
            for (unsigned o = 0; o < OW; ++o) nxt[o] = __ldg(wrow + std::size_t{o} * row_u4 + (sg + 1U) * 32U + lane);
        }
        __syncwarp();
        const float* xc = x + sg * 256U + lane;
#pragma unroll
        for (unsigned k = 0; k < 8; ++k) {
            float wf[OW];
#pragma unroll
            for (unsigned o = 0; o < OW; ++o) wf[o] = __half2float(hs[o * 256U + lane + 32U * k]);
#pragma unroll
            for (unsigned r = 0; r < RT; ++r) {
                if (r < nr) {
                    const float xv = __ldg(xc + std::size_t{r} * xs + 32U * k);
#pragma unroll
                    for (unsigned o = 0; o < OW; ++o) acc[o][r] = __fadd_rn(acc[o][r], __fmul_rn(wf[o], xv));
                }
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (unsigned o = 0; o < OW; ++o)
#pragma unroll
        for (unsigned r = 0; r < RT; ++r) {
            const float s = warp_sum(acc[o][r]);
            if (lane == r && r < nr) job.out[std::size_t{r0 + r} * job.out_stride + o0 + o] = s;
        }
}

// ---- window ring ------------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(32) void window_write_kernel(WindowView v) {
    const unsigned r = blockIdx.x, lane = threadIdx.x;
    const std::uint64_t p = v.pos[r];
    float x[16];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) x[i] = v.kv[std::size_t{r} * kHeadDim + 16 * lane + i];
    norm512(x, v.norm, lane);
    rope512(x, v.rope[p], lane, false);
    float amax = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(x[i]));
    amax = fmaxf(fmaxf(amax, __shfl_xor_sync(kFull, amax, 1)), 1e-4F);
    const int e = log2_ceil(amax * (1.0F / 448.0F));
    const float s = pow2f(e);
    std::uint8_t* entry = v.ring + (p % v.capacity) * kWindowEntryBytes;
    std::uint32_t q[4] = {0, 0, 0, 0};
#pragma unroll
    for (unsigned i = 0; i < 16; ++i)
        q[i / 4] |= static_cast<std::uint32_t>(e4m3_encode(clampf(x[i] / s, -448.0F, 448.0F))) << (8U * (i % 4));
    *reinterpret_cast<uint4*>(entry + 16 * lane) = make_uint4(q[0], q[1], q[2], q[3]);
    if ((lane & 1U) == 0U) entry[512 + lane / 2] = static_cast<std::uint8_t>(e + 127);
}

// ---- compressor -------------------------------------------------------------------------------------------------------
// A block per tile of 8 rows: warp w pools + normalises row w's group (if the row completes one), the 256 threads run
// the 128 index-key chains for the tile's rows (each thread streaming its wk row), and
// warp w finishes row w (key norm, RoPE, FP4 index key; RoPE and FP4 KV entry).
constexpr unsigned kCompRows = 8;

__global__ __launch_bounds__(256) void compress_kernel(CompressView v) {
    __shared__ __align__(16) float lat[kCompRows][kHeadDim];
    __shared__ float ks[kCompRows][kIdxDim];
    __shared__ unsigned active[kCompRows];
    const unsigned tid = threadIdx.x, lane = tid & 31U, w = tid >> 5U;
    const unsigned r = blockIdx.x * kCompRows + w;
    // phase A: pooled group and its norm
    {
        bool act = false;
        if (r < v.rows) {
            const std::uint64_t p = v.pos[r];
            float x[16];
            const float* kv = v.kv + std::size_t{r} * kHeadDim + 16 * lane;
            if (v.ratio == 2U) {
                if ((p & 1U) != 0U) {
                    act = true;
                    const float* sc = v.score + std::size_t{r} * kHeadDim + 16 * lane;
                    const float* kv0 = r > 0U ? v.kv + std::size_t{r - 1U} * kHeadDim + 16 * lane : v.state + 16 * lane;
                    const float* sc0 = r > 0U ? v.score + std::size_t{r - 1U} * kHeadDim + 16 * lane : v.state + kHeadDim + 16 * lane;
#pragma unroll
                    for (unsigned i = 0; i < 16; ++i) {
                        const float m = fmaxf(sc0[i], sc[i]);
                        const float e0 = exp_f(sc0[i] - m), e1 = exp_f(sc[i] - m);
                        const float sum = e0 + e1;
                        x[i] = kv0[i] * (e0 / sum) + kv[i] * (e1 / sum);
                    }
                }
            } else {
                act = true;
#pragma unroll
                for (unsigned i = 0; i < 16; ++i) x[i] = kv[i];
            }
            if (act) {
                norm512(x, v.norm, lane);
#pragma unroll
                for (unsigned i = 0; i < 16; ++i) lat[w][16 * lane + i] = x[i];
            }
        }
        if (lane == 0U) active[w] = act ? 1U : 0U;
    }
    __syncthreads();
    unsigned any = 0;
#pragma unroll
    for (unsigned i = 0; i < kCompRows; ++i) any |= active[i];
    if (any == 0U) return;
    // phase B: k[row][o] = fmaf chain over d of f16(wk[o][d]) * latent[d]; thread: o = tid % 128, rows 4 * (tid / 128) ..
    // +3; the thread streams its own wk row (16 bytes per load, the next 64 columns in flight while it computes).
    {
        const unsigned o = tid & 127U, rh = (tid >> 7U) * 4U;
        if ((active[rh] | active[rh + 1U] | active[rh + 2U] | active[rh + 3U]) != 0U) {
            const uint4* wr = reinterpret_cast<const uint4*>(v.wk + std::size_t{o} * kHeadDim);
            float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            uint4 cur[8], nxt[8];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) cur[i] = __ldg(wr + i);
            for (unsigned d0 = 0; d0 < kHeadDim; d0 += 64U) {
                if (d0 + 64U < kHeadDim) {
#pragma unroll
                    for (unsigned i = 0; i < 8; ++i) nxt[i] = __ldg(wr + (d0 + 64U) / 8U + i);
                }
#pragma unroll
                for (unsigned i = 0; i < 8; ++i) {
                    const __half* h = reinterpret_cast<const __half*>(&cur[i]);
#pragma unroll
                    for (unsigned k4 = 0; k4 < 2; ++k4) {
                        const unsigned d = d0 + 8U * i + 4U * k4;
                        float4 lv[4];
#pragma unroll
                        for (unsigned q = 0; q < 4; ++q) lv[q] = *reinterpret_cast<const float4*>(&lat[rh + q][d]);
#pragma unroll
                        for (unsigned k = 0; k < 4; ++k) {
                            const float wv = __half2float(h[4 * k4 + k]);
#pragma unroll
                            for (unsigned q = 0; q < 4; ++q) {
                                const float lk = k == 0 ? lv[q].x : k == 1 ? lv[q].y : k == 2 ? lv[q].z : lv[q].w;
                                acc[q] = fmaf(wv, lk, acc[q]);
                            }
                        }
                    }
                }
#pragma unroll
                for (unsigned i = 0; i < 8; ++i) cur[i] = nxt[i];
            }
#pragma unroll
            for (unsigned q = 0; q < 4; ++q) ks[rh + q][o] = acc[q];
        }
    }
    __syncthreads();
    // phase C: warp w finishes row w
    if (active[w] == 0U) return;
    const std::uint64_t p = v.pos[r];
    const std::uint64_t slot = p / v.ratio, rope_pos = slot * v.ratio;
    const RopeRow& rope = v.rope[rope_pos];
    {
        float k[4];
        float acc = 0.0F;
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) { k[i] = ks[w][4 * lane + i]; acc = fmaf(k[i], k[i], acc); }
        const float rk = 1.0F / sqrtf(warp_sum(acc) / 128.0F + kNormEps);
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) k[i] = (k[i] * rk) * v.knorm[4 * lane + i];
        if (lane >= 16U) {  // dims 64..127: pairs 2(lane - 16), +1
#pragma unroll
            for (unsigned i = 0; i < 2; ++i) {
                const unsigned pair = 2U * (lane - 16U) + i;
                rope_pair(k[2 * i], k[2 * i + 1], rope.c[pair], rope.s[pair], false);
            }
        }
        float amax = 0.0F;
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) amax = fmaxf(amax, fabsf(k[i]));
        amax = fmaxf(warp_max(amax, 4U), 7.05296657e-38F);
        const int e = log2_ceil(amax * (1.0F / 6.0F));
        const float s = pow2f(e);
        std::uint8_t c[4];
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) c[i] = e2m1_encode(clampf(k[i] / s, -6.0F, 6.0F));
        std::uint8_t* ie = v.idx_cache + slot * kIdxEntryBytes;
        ie[2 * lane] = static_cast<std::uint8_t>(c[0] | (c[1] << 4U));
        ie[2 * lane + 1] = static_cast<std::uint8_t>(c[2] | (c[3] << 4U));
        if ((lane & 7U) == 0U) ie[64 + lane / 8U] = static_cast<std::uint8_t>(e + 127);
    }
    {
        float x[16];
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) x[i] = lat[w][16 * lane + i];
        rope512(x, rope, lane, false);
        float amax = 0.0F;
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(x[i]));
        amax = fmaxf(amax, 0.01171875F);
        const std::uint8_t sc = e4m3_encode(amax / 6.0F);
        const float s = e4m3_value(sc);
        std::uint32_t q[2] = {0, 0};
#pragma unroll
        for (unsigned i = 0; i < 16; i += 2) {
            const std::uint8_t lo = e2m1_encode(clampf(x[i] / s, -6.0F, 6.0F));
            const std::uint8_t hi = e2m1_encode(clampf(x[i + 1] / s, -6.0F, 6.0F));
            q[i / 8] |= static_cast<std::uint32_t>(lo | (hi << 4U)) << (8U * ((i / 2) % 4));
        }
        std::uint8_t* ke = v.kv_cache + slot * kKvEntryBytes;
        *reinterpret_cast<uint2*>(ke + 8 * lane) = make_uint2(q[0], q[1]);
        ke[256 + lane] = sc;
    }
}

// ratio 2: the pending group after the last row (its first token) into state, after compress_kernel read it.
__global__ __launch_bounds__(256) void compress_state_kernel(CompressView v) {
    const std::uint64_t p = v.pos[v.rows - 1U];
    const bool pending = (p & 1U) == 0U;
    if (pending) {
        const float* kv = v.kv + std::size_t{v.rows - 1U} * kHeadDim;
        const float* sc = v.score + std::size_t{v.rows - 1U} * kHeadDim;
        for (unsigned i = threadIdx.x; i < kHeadDim; i += blockDim.x) { v.state[i] = kv[i]; v.state[kHeadDim + i] = sc[i]; }
    }
    if (threadIdx.x == 0U) *v.state_valid = pending ? 1U : 0U;
    if (v.stash != nullptr)
        for (unsigned r = 0; r < v.rows; ++r) {
            const float* kv = v.kv + std::size_t{r} * kHeadDim;
            const float* sc = v.score + std::size_t{r} * kHeadDim;
            float* d = v.stash + std::size_t{r} * 2U * kHeadDim;
            for (unsigned i = threadIdx.x; i < kHeadDim; i += blockDim.x) { d[i] = kv[i]; d[kHeadDim + i] = sc[i]; }
        }
}

}  // namespace

// partial commit: the carry of a call whose last row is `row` (at position p)
__global__ __launch_bounds__(256) void carry_restore_kernel(const float* stash, float* state, std::uint32_t* valid, std::uint64_t p, unsigned row) {
    const bool pending = (p & 1U) == 0U;
    if (pending) {
        const float* s = stash + std::size_t{row} * 2U * kHeadDim;
        for (unsigned i = threadIdx.x; i < 2U * kHeadDim; i += blockDim.x) state[i] = s[i];
    }
    if (threadIdx.x == 0U) *valid = pending ? 1U : 0U;
}

namespace {

// ---- indexer ----------------------------------------------------------------------------------------------------------
// qprep per row: int32 words [32 heads][32] of 4 doubled E2M1 codes (dims 4i..4i+3, low byte first), float scales
// [32][4] (2^e of each block of 32), float head weights [32].
struct QPrep {
    int codes[kIdxHeads][kIdxDim / 4];
    float sq[kIdxHeads][4];
    float w[kIdxHeads];
};
static_assert(sizeof(QPrep) == kIdxPrepBytes);

__global__ __launch_bounds__(1024) void index_prep_kernel(IndexView v) {
    const unsigned r = blockIdx.x, h = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    QPrep* out = reinterpret_cast<QPrep*>(v.qprep) + r;
    const RopeRow& rope = v.rope[v.pos[r]];
    float q[4];
    const float* src = v.q + (std::size_t{r} * kIdxHeads + h) * kIdxDim + 4 * lane;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) q[i] = src[i];
    if (lane >= 16U) {
#pragma unroll
        for (unsigned i = 0; i < 2; ++i) {
            const unsigned pair = 2U * (lane - 16U) + i;
            rope_pair(q[2 * i], q[2 * i + 1], rope.c[pair], rope.s[pair], false);
        }
    }
    float amax = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) amax = fmaxf(amax, fabsf(q[i]));
    amax = fmaxf(warp_max(amax, 4U), 7.05296657e-38F);
    const int e = log2_ceil(amax * (1.0F / 6.0F));
    const float s = pow2f(e);
    std::uint32_t word = 0;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const int c = e2m1_int2(e2m1_encode(clampf(q[i] / s, -6.0F, 6.0F)));
        word |= (static_cast<std::uint32_t>(c) & 255U) << (8U * i);
    }
    out->codes[h][lane] = static_cast<int>(word);
    if ((lane & 7U) == 0U) out->sq[h][lane / 8U] = s;
    if (lane == 0U) out->w[h] = v.w[std::size_t{r} * kIdxHeads + h] * kIdxWeightScale;
}

// Scores: a block of 4 warps covers G groups of 32 positions x RT rows (G * RT = 4); the tile's keys are decoded once
// into shared memory (doubled E2M1 codes as packed int8 words, E8M0 scales). Warp (group, row): lane h owns head h,
// computes d_h of the group's 32 positions (exact integer block dots by __dp4a, then the declared float terms) into a
// shared [head][position] buffer; then lane t runs the declared head chain of position t.
constexpr unsigned kScoreWarps = 4;
// Four E2M1 codes (the low 16 bits, code k at bits 4k) as four int8 of the doubled values (e2m1_doubled): the
// magnitudes {0, 1, 2, 3, 4, 6, 8, 12} by one byte permute, then the signed bytes negated.
__device__ __forceinline__ std::uint32_t e2m1x4_doubled(std::uint32_t c) {
    const std::uint32_t mag = __byte_perm(0x03020100U, 0x0C080604U, c & 0x7777U);
    const std::uint32_t s = (c >> 3U) & 0x1111U;
    const std::uint32_t sb = (s & 1U) | ((s & 0x10U) << 4U) | ((s & 0x100U) << 8U) | ((s & 0x1000U) << 12U);
    const std::uint32_t mask = sb * 0xFFU;
    return (mag & ~mask) | (__vneg4(mag) & mask);
}

template <unsigned G, unsigned RT>
__global__ __launch_bounds__(kScoreWarps * 32) void index_scores_kernel(IndexView v) {
    static_assert(G * RT == kScoreWarps);
    __shared__ int4 kw[G][32][8];          // key words of the tile: [group][position][32 words]
    __shared__ float ksc[G][32][4];
    __shared__ float dsm[kScoreWarps][32][33];
    __shared__ float wsm[RT][kIdxHeads];
    const unsigned tid = threadIdx.x, lane = tid & 31U, w = tid >> 5U;
    const unsigned g = w % G, rr = w / G;
    const unsigned t_base = blockIdx.x * 32U * G;
    const unsigned r = blockIdx.y * RT + rr;
    const QPrep* prep = reinterpret_cast<const QPrep*>(v.qprep);
    if (tid < 32U * G) {  // decode one position per thread
        const unsigned t = t_base + tid;
        if (t < v.max_count) {
            const std::uint32_t* e = reinterpret_cast<const std::uint32_t*>(v.idx_cache + std::size_t{t} * kIdxEntryBytes);
            int words[32];
#pragma unroll
            for (unsigned i = 0; i < 16; ++i) {
                const std::uint32_t b = e[i];  // dims 8i .. 8i + 7
#pragma unroll
                for (unsigned half = 0; half < 2; ++half) words[2 * i + half] = static_cast<int>(e2m1x4_doubled(b >> (16U * half)));
            }
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) kw[tid / 32U][tid % 32U][i] = make_int4(words[4 * i], words[4 * i + 1], words[4 * i + 2], words[4 * i + 3]);
            const std::uint32_t sw = e[16];
#pragma unroll
            for (unsigned b = 0; b < 4; ++b) ksc[tid / 32U][tid % 32U][b] = e8m0_value(static_cast<std::uint8_t>((sw >> (8U * b)) & 255U));
        }
    }
    {
        const unsigned rt = tid / 32U, h = tid % 32U, rw = blockIdx.y * RT + rt;
        if (rt < RT && rw < v.rows) wsm[rt][h] = prep[rw].w[h];
    }
    __syncthreads();
    if (r >= v.rows) return;
    const unsigned count = static_cast<unsigned>((v.pos[r] + 1U) / v.ratio);
    const unsigned t0 = t_base + 32U * g;
    if (t0 >= count) return;
    int q[32];
    {
        const int4* src = reinterpret_cast<const int4*>(prep[r].codes[lane]);
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const int4 u = src[i];
            q[4 * i] = u.x; q[4 * i + 1] = u.y; q[4 * i + 2] = u.z; q[4 * i + 3] = u.w;
        }
    }
    float sq[4];
#pragma unroll
    for (unsigned b = 0; b < 4; ++b) sq[b] = prep[r].sq[lane][b];
    const unsigned n = min(32U, count - t0);
    for (unsigned t = 0; t < n; ++t) {
        float term[4];
#pragma unroll
        for (unsigned b = 0; b < 4; ++b) {
            int dot = 0;
#pragma unroll
            for (unsigned i = 0; i < 2; ++i) {
                const int4 k = kw[g][t][2 * b + i];
                dot = __dp4a(q[8 * b + 4 * i], k.x, dot);
                dot = __dp4a(q[8 * b + 4 * i + 1], k.y, dot);
                dot = __dp4a(q[8 * b + 4 * i + 2], k.z, dot);
                dot = __dp4a(q[8 * b + 4 * i + 3], k.w, dot);
            }
            term[b] = ((static_cast<float>(dot) * sq[b]) * ksc[g][t][b]) * 0.25F;
        }
        dsm[w][lane][t] = ((term[0] + term[1]) + term[2]) + term[3];
    }
    __syncwarp();
    if (lane < n) {
        float score = 0.0F;
#pragma unroll 8
        for (unsigned h = 0; h < kIdxHeads; ++h) score = fmaf(fmaxf(dsm[w][h][lane], 0.0F), wsm[rr][h], score);
        v.scores[std::size_t{r} * v.max_count + t0 + lane] = score;
    }
}

// ---- selection --------------------------------------------------------------------------------------------------------
// Exact top-k (key descending, position ascending), written in ascending position order: the k-th key K by a 4-pass
// radix histogram (integer counts, warp-aggregated), then every key > K and the first (k - #greater) keys == K in
// position order, compacted with block scans. Rows up to kSelDirect positions run in one block with the keys in shared
// memory; longer rows in two levels: the exact top-k of every slice of kSelSlice positions (survivors kept in position
// order), then the top-k over the survivors. Exact: an element of the row's top-k has fewer than k elements of its
// slice before it, so it survives its slice.
constexpr unsigned kSelThreads = 1024;
using SelScan = cub::BlockScan<unsigned, kSelThreads>;

struct SelShared {
    typename SelScan::TempStorage scan;
    unsigned hist[256];
    unsigned prefix, remaining, base_eq, base_sel;
};

// K and the number of keys == K to take, over n keys key(i), the best k (k < n).
template <class Key>
__device__ void radix_kth(SelShared& s, Key key, unsigned n, unsigned k, unsigned& K, unsigned& need) {
    const unsigned lane = threadIdx.x & 31U;
    unsigned prefix = 0, mask = 0, remaining = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (unsigned i = threadIdx.x; i < 256U; i += kSelThreads) s.hist[i] = 0U;
        __syncthreads();
        for (unsigned base = 0; base < n; base += kSelThreads) {
            const unsigned i = base + threadIdx.x;
            unsigned bin = 256U;
            if (i < n) {
                const unsigned kv = key(i);
                if ((kv & mask) == prefix) bin = (kv >> shift) & 255U;
            }
            const unsigned peers = __match_any_sync(kFull, bin);
            if (bin != 256U && lane == static_cast<unsigned>(__ffs(peers) - 1)) atomicAdd(&s.hist[bin], static_cast<unsigned>(__popc(peers)));
        }
        __syncthreads();
        if (threadIdx.x < 32U) {  // warp 0: the bin holding the remaining-th key from the top
            unsigned c[8], tot = 0;
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) { c[i] = s.hist[255U - (8U * lane + i)]; tot += c[i]; }
            unsigned incl = tot;  // inclusive scan over lanes (descending bins)
#pragma unroll
            for (unsigned d = 1; d < 32U; d *= 2U) {
                const unsigned t = __shfl_up_sync(kFull, incl, d);
                if (lane >= d) incl += t;
            }
            const unsigned excl = incl - tot;
            const unsigned hit = __ballot_sync(kFull, excl < remaining && incl >= remaining);
            if (lane == static_cast<unsigned>(__ffs(hit) - 1)) {
                unsigned above = excl, i = 0;
                while (above + c[i] < remaining) { above += c[i]; ++i; }
                s.prefix = prefix | ((255U - (8U * lane + i)) << shift);
                s.remaining = remaining - above;
            }
        }
        __syncthreads();
        prefix = s.prefix;
        remaining = s.remaining;
        mask |= 255U << shift;
        __syncthreads();
    }
    K = prefix;
    need = remaining;
}

// Calls emit(i, slot) for the selected i in ascending order (slot = rank among the selected).
template <class Key, class Emit>
__device__ void compact(SelShared& s, Key key, unsigned n, unsigned K, unsigned need, Emit emit) {
    if (threadIdx.x == 0U) { s.base_eq = 0U; s.base_sel = 0U; }
    __syncthreads();
    for (unsigned c0 = 0; c0 < n; c0 += kSelThreads) {
        const unsigned i = c0 + threadIdx.x;
        const unsigned kv = i < n ? key(i) : 0U;
        const unsigned eq = (i < n && kv == K) ? 1U : 0U;
        unsigned eq_before, eq_total;
        SelScan(s.scan).ExclusiveSum(eq, eq_before, eq_total);
        const unsigned base_eq = s.base_eq;
        const unsigned sel = (i < n && (kv > K || (eq != 0U && base_eq + eq_before < need))) ? 1U : 0U;
        __syncthreads();
        unsigned sel_before, sel_total;
        SelScan(s.scan).ExclusiveSum(sel, sel_before, sel_total);
        const unsigned base_sel = s.base_sel;
        if (sel != 0U) emit(i, base_sel + sel_before);
        __syncthreads();
        if (threadIdx.x == 0U) { s.base_eq = base_eq + eq_total; s.base_sel = base_sel + sel_total; }
        __syncthreads();
    }
}

// The best k of n keys (all of them when n <= k), emitted in ascending order.
template <class Key, class Emit>
__device__ void select_best(SelShared& s, Key key, unsigned n, unsigned k, Emit emit) {
    if (n <= k) {
        for (unsigned i = threadIdx.x; i < n; i += kSelThreads) emit(i, i);
        return;
    }
    unsigned K, need;
    radix_kth(s, key, n, k, K, need);
    compact(s, key, n, K, need, emit);
}

struct SelRow {
    unsigned count;
    const float* sc;
    std::uint8_t* cand;
    bool user;
    unsigned bs;
    __device__ unsigned key(unsigned t) const {
        return (user && cand[t / bs] == 0U) ? order_key(-__builtin_inff()) : order_key(sc[t]);
    }
};
__device__ SelRow sel_row(const SelectView& v, unsigned r) {
    return {static_cast<unsigned>((v.pos[r] + 1U) / v.ratio), v.scores + std::size_t{r} * v.max_count, v.candidates + std::size_t{r} * v.cand_stride,
            v.pool == Pool::candidate_user, v.block_size};
}

// Per row: the candidate blocks (source), then the whole selection when count <= kSelDirect.
__global__ __launch_bounds__(kSelThreads) void select_kernel(SelectView v) {
    __shared__ SelShared s;
    __shared__ unsigned keys[kSelDirect];
    const unsigned r = blockIdx.x;
    const SelRow row = sel_row(v, r);
    const unsigned count = row.count;
    const unsigned ninf_key = order_key(-__builtin_inff());
    if (v.pool == Pool::candidate_source && count != 0U) {
        const unsigned bs = v.block_size, nb = (count + bs - 1U) / bs, last = (count - 1U) / bs;
        std::uint32_t* bk = v.block_keys + std::size_t{r} * v.cand_stride;
        for (unsigned b = threadIdx.x; b < nb; b += kSelThreads) {
            float m = -__builtin_inff();
            const unsigned e = min(count, (b + 1U) * bs);
            for (unsigned t = b * bs; t < e; ++t) m = fmaxf(m, row.sc[t]);
            bk[b] = order_key(b == last ? __builtin_inff() : m);
            row.cand[b] = 0U;
        }
        __syncthreads();
        select_best(s, [&](unsigned b) { return bk[b]; }, nb, v.cand_blocks, [&](unsigned b, unsigned) { if (bk[b] > ninf_key) row.cand[b] = 1U; });
        __syncthreads();
    }
    if (count > kSelDirect) return;  // two levels
    std::uint32_t* idx = v.idx + std::size_t{r} * v.topk;
    if (count <= v.topk) {
        for (unsigned t = threadIdx.x; t < count; t += kSelThreads) idx[t] = t;
    } else {
        for (unsigned t = threadIdx.x; t < count; t += kSelThreads) keys[t] = row.key(t);
        __syncthreads();
        select_best(s, [&](unsigned t) { return keys[t]; }, count, v.topk, [&](unsigned t, unsigned slot) { idx[slot] = t; });
    }
    if (threadIdx.x == 0U) v.idx_count[r] = min(count, v.topk);
}

// Level one: slice z of row r -> its survivors (key, position) in position order and their count.
__global__ __launch_bounds__(kSelThreads) void select_slice_kernel(SelectView v, unsigned slices) {
    __shared__ SelShared s;
    __shared__ unsigned keys[kSelSlice];
    const unsigned z = blockIdx.x, r = blockIdx.y;
    const SelRow row = sel_row(v, r);
    if (row.count <= kSelDirect) return;
    const unsigned t0 = z * kSelSlice;
    std::uint32_t* sk = v.survivors + (std::size_t{r} * slices + z) * v.topk * 2U;
    std::uint32_t* sp = sk + v.topk;
    std::uint32_t* sn = v.survivors + std::size_t{v.rows} * slices * v.topk * 2U + std::size_t{r} * slices + z;
    if (t0 >= row.count) { if (threadIdx.x == 0U) *sn = 0U; return; }
    const unsigned n = min(kSelSlice, row.count - t0);
    for (unsigned i = threadIdx.x; i < n; i += kSelThreads) keys[i] = row.key(t0 + i);
    __syncthreads();
    select_best(s, [&](unsigned i) { return keys[i]; }, n, v.topk, [&](unsigned i, unsigned slot) { sk[slot] = keys[i]; sp[slot] = t0 + i; });
    if (threadIdx.x == 0U) *sn = min(n, v.topk);
}

// Level two: the top-k over the row's survivors (slices in order, so positions ascend).
template <unsigned MaxSlices>
__global__ __launch_bounds__(kSelThreads) void select_merge_kernel(SelectView v, unsigned slices) {
    __shared__ SelShared s;
    __shared__ unsigned offs[MaxSlices + 1];
    const unsigned r = blockIdx.x;
    const SelRow row = sel_row(v, r);
    if (row.count <= kSelDirect) return;
    const std::uint32_t* base = v.survivors + std::size_t{r} * slices * v.topk * 2U;
    const std::uint32_t* sn = v.survivors + std::size_t{v.rows} * slices * v.topk * 2U + std::size_t{r} * slices;
    if (threadIdx.x == 0U) {
        unsigned o = 0;
        for (unsigned z = 0; z < slices; ++z) { offs[z] = o; o += sn[z]; }
        offs[slices] = o;
    }
    __syncthreads();
    const unsigned n = offs[slices];
    // survivor i: slice z with offs[z] <= i < offs[z + 1]
    const auto at = [&](unsigned i) {
        unsigned lo = 0, hi = slices;
        while (hi - lo > 1U) { const unsigned mid = (lo + hi) / 2U; if (offs[mid] <= i) lo = mid; else hi = mid; }
        return base + std::size_t{lo} * v.topk * 2U + (i - offs[lo]);
    };
    std::uint32_t* idx = v.idx + std::size_t{r} * v.topk;
    select_best(s, [&](unsigned i) { return *at(i); }, n, v.topk, [&](unsigned i, unsigned slot) { idx[slot] = at(i)[v.topk]; });
    if (threadIdx.x == 0U) v.idx_count[r] = v.topk;
}

// ---- attention --------------------------------------------------------------------------------------------------------
// Block = (chunk of 64 keys, group of 2 * Warps heads, row); a warp runs two heads. Tiles of 16 keys are dequantised once
// per block into shared memory (float4 i of lane l of key j at (j * 4 + i) * 32 + l: conflict-free reads), then each
// warp runs the declared online softmax of its two heads over the tile:
//  - the 32 lane partials (2 heads x 16 keys) are reduced by a transposed xor butterfly: at distance d every lane keeps
//    half of its items and adds the partner's partial of the same item, which are the additions of the per-key
//    butterfly; item l (head l / 16, key l % 16) ends in lane l, so each lane evaluates one exp_f;
//  - the tile maximum is a max over the head's 16 lanes; p_j are broadcast for the in-order sums of l and acc.
// Partials (m, l, acc) per chunk; the combine kernel finishes in chunk order.
// AllTiles (set_attention_prefetch): the chunk's 4 tiles are copied into 4 raw buffers at once, so the block waits for
// one memory latency instead of one per tile (double buffering hid only the short dequant/compute of a tile); the
// tiles are then processed exactly as before.
__device__ __forceinline__ float e2m1_lut(const float* lut, unsigned c) { return lut[c]; }

constexpr unsigned kChunkTiles = kChunk / kTile;
constexpr std::size_t attn_smem(unsigned raw_buffers) {
    return sizeof(float4) * kTile * 4 * 32 + std::size_t{raw_buffers} * kTile * kWindowEntryBytes + 16 * 4 + kChunk * 4;
}
constexpr std::size_t kAttnSmem = attn_smem(2), kAttnSmemAll = attn_smem(kChunkTiles);

template <unsigned Warps, unsigned H = 2, bool AllTiles = false>
__global__ __launch_bounds__(Warps * 32) void attention_chunk_kernel(AttentionView v) {
    static_assert(H == 1 || H == 2, "heads per warp");
    static_assert(kChunkTiles == 4, "AllTiles waits by tile index");
    constexpr unsigned kRawBuffers = AllTiles ? kChunkTiles : 2U;
    extern __shared__ float4 dyn[];
    float4* keys = dyn;                                                                  // the tile, dequantised
    std::uint8_t* raw = reinterpret_cast<std::uint8_t*>(dyn + kTile * 4 * 32);            // 2 (4) x 16 entries of 528 B
    float* lut = reinterpret_cast<float*>(raw + kRawBuffers * kTile * kWindowEntryBytes);  // E2M1 values
    std::uint32_t* sel_s = reinterpret_cast<std::uint32_t*>(lut + 16);                   // the chunk's selected positions
    const unsigned c = blockIdx.x, r = blockIdx.z, tid = threadIdx.x, lane = tid & 31U, w = tid >> 5U;
    const unsigned h0 = blockIdx.y * H * Warps + H * w;
    const std::uint64_t p = v.pos[r];
    const unsigned nw = static_cast<unsigned>(p + 1U < kWindow ? p + 1U : kWindow);
    const unsigned nc = v.idx_count != nullptr ? v.idx_count[r] : 0U;
    const unsigned n = nw + nc;
    if (c * kChunk >= n) return;
    if (tid < 16U) lut[tid] = e2m1_value(static_cast<std::uint8_t>(tid));
    const RopeRow& rope = v.rope[p];
    float q[H][16];
#pragma unroll
    for (unsigned hh = 0; hh < H; ++hh) {
        const float4* src = reinterpret_cast<const float4*>(v.q + (std::size_t{r} * v.heads + h0 + hh) * kHeadDim + 16 * lane);
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const float4 f = src[i];
            q[hh][4 * i] = f.x; q[hh][4 * i + 1] = f.y; q[hh][4 * i + 2] = f.z; q[hh][4 * i + 3] = f.w;
        }
        rope512(q[hh], rope, lane, false);
    }
    float acc[H][16];
#pragma unroll
    for (unsigned hh = 0; hh < H; ++hh)
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) acc[hh][i] = 0.0F;
    float mc[H], lc[H];
#pragma unroll
    for (unsigned hh = 0; hh < H; ++hh) { mc[hh] = -__builtin_inff(); lc[hh] = 0.0F; }
    const unsigned cb = c * kChunk, end = min(n, cb + kChunk);
    for (unsigned t = tid; t < kChunk; t += Warps * 32U)  // every slot also for one-warp blocks
        if (cb + t >= nw && cb + t < end) sel_s[t] = v.idx[std::size_t{r} * v.topk + (cb + t - nw)];
    __syncthreads();
    // raw entries of a tile into raw[buf] by asynchronous 16-byte copies (window 33 pieces, compressed 18)
    const auto issue = [&](unsigned t0, unsigned buf) {
        const unsigned cnt = min(kTile, n - t0);
        for (unsigned piece = tid; piece < cnt * 33U; piece += Warps * 32U) {
            const unsigned j = piece / 33U, part = piece % 33U, key = t0 + j;
            std::uint8_t* dst = raw + (buf * kTile + j) * kWindowEntryBytes + 16U * part;
            if (key < nw) {
                __pipeline_memcpy_async(dst, v.ring + ((p + 1U - nw + key) % v.capacity) * kWindowEntryBytes + 16U * part, 16);
            } else if (part < kKvEntryBytes / 16U) {
                __pipeline_memcpy_async(dst, v.kv_cache + std::size_t{sel_s[key - cb]} * kKvEntryBytes + 16U * part, 16);
            }
        }
        __pipeline_commit();
    };
    // H = 2: item l = (head l / 16, key l % 16) in lane l; H = 1: key j in lanes 2 j and 2 j + 1
    const unsigned item_h = H == 2 ? lane >> 4U : 0U, item_j = H == 2 ? lane & 15U : lane >> 1U;
    if constexpr (AllTiles) {  // one commit group per tile slot (empty past the chunk's end), tile i in raw buffer i
#pragma unroll
        for (unsigned i = 0; i < kChunkTiles; ++i) {
            if (cb + i * kTile < end) issue(cb + i * kTile, i);
            else __pipeline_commit();
        }
    } else {
        issue(cb, 0U);
    }
    unsigned it = 0;
    for (unsigned t0 = cb; t0 < end; t0 += kTile, ++it) {
        const unsigned cnt = min(kTile, n - t0), buf = AllTiles ? it : it & 1U;
        if constexpr (AllTiles) {  // tile it's group done: at most 3 - it later groups pending
            if (it == 0U) __pipeline_wait_prior(3);
            else if (it == 1U) __pipeline_wait_prior(2);
            else if (it == 2U) __pipeline_wait_prior(1);
            else __pipeline_wait_prior(0);
        } else if (t0 + kTile < end) {
            issue(t0 + kTile, buf ^ 1U);
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();
        // dequantise: job = (key j, lane slot l): 16 dims
        for (unsigned job = tid; job < cnt * 32U; job += Warps * 32U) {
            const unsigned j = job >> 5U, l = job & 31U, key = t0 + j;
            const std::uint8_t* e = raw + (buf * kTile + j) * kWindowEntryBytes;
            float d[16];
            if (key < nw) {
                const uint4 u = *reinterpret_cast<const uint4*>(e + 16 * l);
                const float s = e8m0_value(e[512 + l / 2]);
                const std::uint32_t wv[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (unsigned i = 0; i < 8; ++i) {
                    const __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>((wv[i / 2] >> (16U * (i % 2))) & 0xFFFFU), __NV_E4M3);
                    const float2 f = __half22float2(__half2(h2));
                    d[2 * i] = f.x * s;
                    d[2 * i + 1] = f.y * s;
                }
            } else {
                const uint2 u = *reinterpret_cast<const uint2*>(e + 8 * l);
                const float s = e4m3_value(e[256 + l]);
                const std::uint32_t wv[2] = {u.x, u.y};
#pragma unroll
                for (unsigned i = 0; i < 16; ++i) d[i] = e2m1_lut(lut, (wv[i / 8] >> (4U * (i % 8))) & 15U) * s;
            }
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) keys[(j * 4 + i) * 32 + l] = make_float4(d[4 * i], d[4 * i + 1], d[4 * i + 2], d[4 * i + 3]);
        }
        __syncthreads();
        // lane partials of the 16 H items (head hh, key j): fmaf chains over the lane's 16 dims
        float part[16 * H];
#pragma unroll
        for (unsigned j = 0; j < kTile; ++j) {
            float a0 = 0.0F, a1 = 0.0F;
            if (j < cnt) {
#pragma unroll
                for (unsigned i = 0; i < 4; ++i) {
                    const float4 k = keys[(j * 4 + i) * 32 + lane];
                    a0 = fmaf(q[0][4 * i], k.x, a0); a0 = fmaf(q[0][4 * i + 1], k.y, a0);
                    a0 = fmaf(q[0][4 * i + 2], k.z, a0); a0 = fmaf(q[0][4 * i + 3], k.w, a0);
                    if constexpr (H == 2) {
                        a1 = fmaf(q[H - 1][4 * i], k.x, a1); a1 = fmaf(q[H - 1][4 * i + 1], k.y, a1);
                        a1 = fmaf(q[H - 1][4 * i + 2], k.z, a1); a1 = fmaf(q[H - 1][4 * i + 3], k.w, a1);
                    }
                }
            }
            part[j] = a0;
            if constexpr (H == 2) part[16 * (H - 1) + j] = a1;
        }
        // transposed butterfly: H = 2: item l ends in lane l; H = 1: item j ends in lanes 2 j and 2 j + 1 (the last
        // distance adds the partner's partial of the same item, as the per-key butterfly does)
#pragma unroll
        for (unsigned d = 16, m = 8 * H; d != 0; d /= 2, m /= 2) {
            if (m == 0U) {
                part[0] = part[0] + __shfl_xor_sync(kFull, part[0], d);
                continue;
            }
            const bool upper = (lane & d) != 0U;
#pragma unroll
            for (unsigned i = 0; i < m; ++i) {
                const float send = upper ? part[i] : part[i + m];
                const float keep = upper ? part[i + m] : part[i];
                part[i] = keep + __shfl_xor_sync(kFull, send, d);
            }
        }
        const bool valid = item_j < cnt;
        const float s = part[0] * kAttnScale;
        float tm = valid ? s : -__builtin_inff();
        // the tile maximum over the head's 16 items: xor distances 8, 4, 2, 1 on the item index
#pragma unroll
        for (unsigned d = 8; d != 0; d /= 2) tm = fmaxf(tm, __shfl_xor_sync(kFull, tm, H == 2 ? d : 2U * d));
        float tmh[H];
        if constexpr (H == 2) {
            const float tm_other = __shfl_xor_sync(kFull, tm, 16);
            tmh[0] = item_h == 0U ? tm : tm_other;
            tmh[H - 1] = item_h == 0U ? tm_other : tm;
        } else {
            tmh[0] = tm;
        }
        float mn[H], scl[H];
#pragma unroll
        for (unsigned hh = 0; hh < H; ++hh) { mn[hh] = fmaxf(mc[hh], tmh[hh]); scl[hh] = exp_f(mc[hh] - mn[hh]); }
        const float pv = valid ? exp_f(s - mn[item_h]) : 0.0F;
        float pr[16 * H];
#pragma unroll
        for (unsigned i = 0; i < 16 * H; ++i) pr[i] = __shfl_sync(kFull, pv, H == 2 ? i : 2U * i);
#pragma unroll
        for (unsigned hh = 0; hh < H; ++hh) {
            lc[hh] = lc[hh] * scl[hh];
#pragma unroll
            for (unsigned j = 0; j < kTile; ++j)
                if (j < cnt) lc[hh] = lc[hh] + pr[16 * hh + j];
#pragma unroll
            for (unsigned i = 0; i < 16; ++i) acc[hh][i] = acc[hh][i] * scl[hh];
        }
#pragma unroll
        for (unsigned j = 0; j < kTile; ++j) {
            if (j < cnt) {
#pragma unroll
                for (unsigned i = 0; i < 4; ++i) {
                    const float4 k = keys[(j * 4 + i) * 32 + lane];
#pragma unroll
                    for (unsigned hh = 0; hh < H; ++hh) {
                        acc[hh][4 * i] = fmaf(pr[16 * hh + j], k.x, acc[hh][4 * i]);
                        acc[hh][4 * i + 1] = fmaf(pr[16 * hh + j], k.y, acc[hh][4 * i + 1]);
                        acc[hh][4 * i + 2] = fmaf(pr[16 * hh + j], k.z, acc[hh][4 * i + 2]);
                        acc[hh][4 * i + 3] = fmaf(pr[16 * hh + j], k.w, acc[hh][4 * i + 3]);
                    }
                }
            }
        }
#pragma unroll
        for (unsigned hh = 0; hh < H; ++hh) mc[hh] = mn[hh];
    }
#pragma unroll
    for (unsigned hh = 0; hh < H; ++hh) {
        float* part = v.partial + ((std::size_t{r} * v.heads + h0 + hh) * kMaxChunks + c) * kPartStride;
        if (lane == 0U) { part[0] = mc[hh]; part[1] = lc[hh]; }
        float4* pa = reinterpret_cast<float4*>(part + 4);
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) pa[lane * 4 + i] = make_float4(acc[hh][4 * i], acc[hh][4 * i + 1], acc[hh][4 * i + 2], acc[hh][4 * i + 3]);
    }
}

// AllChunks (set_attention_prefetch): every chunk's accumulator is loaded together with the (m, l) pairs, before the
// max / exp chains (one memory latency instead of one per 4 chunks after them); the same fmaf chains in chunk order.
template <bool AllChunks = false>
__global__ __launch_bounds__(32) void attention_combine_kernel(AttentionView v) {
    const unsigned h = blockIdx.x, r = blockIdx.y, lane = threadIdx.x;
    const std::uint64_t p = v.pos[r];
    const unsigned nw = static_cast<unsigned>(p + 1U < kWindow ? p + 1U : kWindow);
    const unsigned nc = v.idx_count != nullptr ? v.idx_count[r] : 0U;
    const unsigned chunks = (nw + nc + kChunk - 1U) / kChunk;
    const float* base = v.partial + (std::size_t{r} * v.heads + h) * kMaxChunks * kPartStride;
    // every chunk's (m, l) loaded first (independent loads), then the same max / exp / fmaf chains in chunk order
    float pm[kMaxChunks], pl[kMaxChunks];
    float4 all[AllChunks ? kMaxChunks : 1][4];
#pragma unroll
    for (unsigned c = 0; c < kMaxChunks; ++c) {
        pm[c] = c < chunks ? base[c * kPartStride] : 0.0F;
        pl[c] = c < chunks ? base[c * kPartStride + 1] : 0.0F;
        if constexpr (AllChunks) {
            const float4* pa = reinterpret_cast<const float4*>(base + c * kPartStride + 4);
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) all[c][i] = c < chunks ? pa[lane * 4 + i] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        }
    }
    float mm = -__builtin_inff();
#pragma unroll
    for (unsigned c = 0; c < kMaxChunks; ++c)
        if (c < chunks) mm = fmaxf(mm, pm[c]);
    float e[kMaxChunks], lsum = 0.0F;
#pragma unroll
    for (unsigned c = 0; c < kMaxChunks; ++c) {
        if (c < chunks) {
            e[c] = exp_f(pm[c] - mm);
            lsum = fmaf(pl[c], e[c], lsum);
        }
    }
    lsum = lsum + exp_f(v.sink[h] - mm);
    float o[16];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) o[i] = 0.0F;
    if constexpr (AllChunks) {
#pragma unroll
        for (unsigned c = 0; c < kMaxChunks; ++c) {
            if (c >= chunks) break;
            const float ec = e[c];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                o[4 * i] = fmaf(all[c][i].x, ec, o[4 * i]);
                o[4 * i + 1] = fmaf(all[c][i].y, ec, o[4 * i + 1]);
                o[4 * i + 2] = fmaf(all[c][i].z, ec, o[4 * i + 2]);
                o[4 * i + 3] = fmaf(all[c][i].w, ec, o[4 * i + 3]);
            }
        }
    }
    // chunks in steps of 4: the step's accumulator loads are issued first, then the same fmaf chains in chunk order
    // (one chunk at a time paid an L2 round trip per chunk)
    constexpr unsigned kStep = 4;
    for (unsigned c0 = 0; c0 < (AllChunks ? 0U : chunks); c0 += kStep) {
        float4 a[kStep][4];
#pragma unroll
        for (unsigned k = 0; k < kStep; ++k) {
            const float4* pa = reinterpret_cast<const float4*>(base + (c0 + k) * kPartStride + 4);
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) a[k][i] = c0 + k < chunks ? pa[lane * 4 + i] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        }
#pragma unroll
        for (unsigned k = 0; k < kStep; ++k) {
            if (c0 + k >= chunks) break;
            const float ec = e[c0 + k];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                o[4 * i] = fmaf(a[k][i].x, ec, o[4 * i]);
                o[4 * i + 1] = fmaf(a[k][i].y, ec, o[4 * i + 1]);
                o[4 * i + 2] = fmaf(a[k][i].z, ec, o[4 * i + 2]);
                o[4 * i + 3] = fmaf(a[k][i].w, ec, o[4 * i + 3]);
            }
        }
    }
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) o[i] = o[i] / lsum;
    rope512(o, v.rope[p], lane, true);
    float4* dst = reinterpret_cast<float4*>(v.o + (std::size_t{r} * v.heads + h) * kHeadDim + 16 * lane);
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) dst[i] = make_float4(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
}

}  // namespace

void enqueue_f16_rows(const F16RowsView& v, cudaStream_t stream) {
    if (v.job_count == 0U || v.job_count > 4U || v.rows == 0U) throw std::invalid_argument("ds41 cuda f16 rows: 1..4 jobs, rows > 0");
    F16Launch L{v, {}};
    std::uint32_t total = 0;
    for (unsigned j = 0; j < v.job_count; ++j) {
        const auto& job = v.jobs[j];
        if (job.width == 0U || job.width % 256U != 0U || job.count == 0U || (reinterpret_cast<std::uintptr_t>(job.weight) & 15U) != 0U)
            throw std::invalid_argument("ds41 cuda f16 rows: width % 256, 16-byte aligned weights");
        total += job.count;
        L.end[j] = total;
    }
    const unsigned ytiles = (v.rows + kF16RowTile - 1U) / kF16RowTile;
    bool four = v.rows > 4U;  // more rows: four outputs per warp (each activation load serves four)
    for (unsigned j = 0; j < v.job_count; ++j) four = four && v.jobs[j].count % 4U == 0U;
    const dim3 grid1((total + kF16Warps - 1U) / kF16Warps, ytiles);
    if (four) {
        f16_rows_kernel<4, 8><<<dim3((total / 4U + kF16Warps - 1U) / kF16Warps, ytiles), kF16Warps * 32, 0, stream>>>(L);
    } else if (v.rows == 1U) {
        f16_rows_kernel<1, 1><<<grid1, kF16Warps * 32, 0, stream>>>(L);
    } else if (v.rows == 2U) {
        f16_rows_kernel<1, 2><<<grid1, kF16Warps * 32, 0, stream>>>(L);
    } else if (v.rows <= 4U) {
        f16_rows_kernel<1, 4><<<grid1, kF16Warps * 32, 0, stream>>>(L);
    } else {
        f16_rows_kernel<1, 8><<<grid1, kF16Warps * 32, 0, stream>>>(L);
    }
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_window_write(const WindowView& v, cudaStream_t stream) {
    if (v.rows == 0U || v.capacity < kWindow + v.rows) throw std::invalid_argument("ds41 cuda window: capacity >= 128 + rows");
    window_write_kernel<<<v.rows, 32, 0, stream>>>(v);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_compress(const CompressView& v, cudaStream_t stream) {
    if (v.rows == 0U || (v.ratio != 1U && v.ratio != 2U)) throw std::invalid_argument("ds41 cuda compress: ratio 1 or 2, rows > 0");
    compress_kernel<<<(v.rows + kCompRows - 1U) / kCompRows, 256, 0, stream>>>(v);
    STRIX_CUDA_CHECK(cudaGetLastError());
    if (v.ratio == 2U) {
        compress_state_kernel<<<1, 256, 0, stream>>>(v);
        STRIX_CUDA_CHECK(cudaGetLastError());
    }
}

void enqueue_carry_restore(const float* stash, float* state, std::uint32_t* state_valid, std::uint64_t pos, std::uint32_t row, cudaStream_t stream) {
    if (stash == nullptr || state == nullptr || state_valid == nullptr) throw std::invalid_argument("ds41 cuda carry restore: buffers");
    carry_restore_kernel<<<1, 256, 0, stream>>>(stash, state, state_valid, pos, row);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_index_scores(const IndexView& v, cudaStream_t stream) {
    if (v.rows == 0U || v.ratio == 0U) throw std::invalid_argument("ds41 cuda index: rows > 0, ratio > 0");
    index_prep_kernel<<<v.rows, 1024, 0, stream>>>(v);
    STRIX_CUDA_CHECK(cudaGetLastError());
    if (v.max_count == 0U) return;
    if (v.rows >= 4U) {
        index_scores_kernel<1, 4><<<dim3((v.max_count + 31U) / 32U, (v.rows + 3U) / 4U), kScoreWarps * 32, 0, stream>>>(v);
    } else if (v.rows >= 2U) {
        index_scores_kernel<2, 2><<<dim3((v.max_count + 63U) / 64U, (v.rows + 1U) / 2U), kScoreWarps * 32, 0, stream>>>(v);
    } else {
        index_scores_kernel<4, 1><<<dim3((v.max_count + 127U) / 128U, v.rows), kScoreWarps * 32, 0, stream>>>(v);
    }
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_select(const SelectView& v, cudaStream_t stream) {
    if (v.rows == 0U || v.ratio == 0U || v.topk == 0U || v.block_size == 0U) throw std::invalid_argument("ds41 cuda select geometry");
    const auto slices_wide = (std::size_t{v.max_count} + kSelSlice - 1U) / kSelSlice;
    const auto limit = v.long_context ? kSelMaxSlices : 128U;
    if (v.max_count > kSelDirect && (v.survivors == nullptr || slices_wide > limit))
        throw std::invalid_argument("ds41 cuda select: survivors scratch or entry limit (long_context required above 512K)");
    const auto slices = static_cast<unsigned>(slices_wide);
    select_kernel<<<v.rows, kSelThreads, 0, stream>>>(v);
    STRIX_CUDA_CHECK(cudaGetLastError());
    if (v.max_count > kSelDirect) {
        select_slice_kernel<<<dim3(slices, v.rows), kSelThreads, 0, stream>>>(v, slices);
        STRIX_CUDA_CHECK(cudaGetLastError());
        if (slices > 128U) select_merge_kernel<kSelMaxSlices><<<v.rows, kSelThreads, 0, stream>>>(v, slices);
        else select_merge_kernel<128><<<v.rows, kSelThreads, 0, stream>>>(v, slices);
        STRIX_CUDA_CHECK(cudaGetLastError());
    }
}

namespace {
unsigned g_chunk_warps = 2;  // decode rows: warps per attention block, 2 or 1 (2 heads each) or 4 (1 head each)
bool g_attn_prefetch = false;  // the 4-warp chunk kernel with every tile issued at once + the all-chunk combine
}  // namespace
void set_attention_chunk_warps(unsigned warps) { g_chunk_warps = warps == 1U || warps == 4U ? warps : 2U; }
void set_attention_prefetch(bool on) { g_attn_prefetch = on; }

void enqueue_attention(const AttentionView& v, cudaStream_t stream) {
    if (v.rows == 0U || v.heads == 0U || v.heads % 4U != 0U || v.capacity < kWindow + v.rows || v.topk > kTopK)
        throw std::invalid_argument("ds41 cuda attention geometry");
    // 16 heads per block when there are enough blocks for the device, else 4 (more blocks for a decode row), or 2
    // (set_attention_chunk_warps(1): twice the blocks, each head's arithmetic unchanged)
    if (g_chunk_warps == 4U && !(v.heads % 16U == 0U && std::size_t{v.rows} * kMaxChunks * (v.heads / 16U) >= 376U)) {
        // one head per warp, 4 per block (set_attention_chunk_warps(4)): twice the warps of <2, 2>, one tile per block
        if (g_attn_prefetch) {
            static const bool opt41a =
                cudaFuncSetAttribute(attention_chunk_kernel<4, 1, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kAttnSmemAll) == cudaSuccess;
            if (!opt41a) throw std::runtime_error("ds41 cuda attention: shared memory opt-in");
            attention_chunk_kernel<4, 1, true><<<dim3(kMaxChunks, v.heads / 4U, v.rows), 4 * 32, kAttnSmemAll, stream>>>(v);
        } else {
            static const bool opt41 = cudaFuncSetAttribute(attention_chunk_kernel<4, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, kAttnSmem) == cudaSuccess;
            if (!opt41) throw std::runtime_error("ds41 cuda attention: shared memory opt-in");
            attention_chunk_kernel<4, 1><<<dim3(kMaxChunks, v.heads / 4U, v.rows), 4 * 32, kAttnSmem, stream>>>(v);
        }
    } else if (g_chunk_warps == 1U && !(v.heads % 16U == 0U && std::size_t{v.rows} * kMaxChunks * (v.heads / 16U) >= 376U)) {
        static const bool opt1 = cudaFuncSetAttribute(attention_chunk_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, kAttnSmem) == cudaSuccess;
        if (!opt1) throw std::runtime_error("ds41 cuda attention: shared memory opt-in");
        attention_chunk_kernel<1><<<dim3(kMaxChunks, v.heads / 2U, v.rows), 32, kAttnSmem, stream>>>(v);
    } else if (v.heads % 16U == 0U && std::size_t{v.rows} * kMaxChunks * (v.heads / 16U) >= 376U) {
        static const bool opt8 = cudaFuncSetAttribute(attention_chunk_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, kAttnSmem) == cudaSuccess;
        if (!opt8) throw std::runtime_error("ds41 cuda attention: shared memory opt-in");
        attention_chunk_kernel<8><<<dim3(kMaxChunks, v.heads / 16U, v.rows), 8 * 32, kAttnSmem, stream>>>(v);
    } else {
        static const bool opt2 = cudaFuncSetAttribute(attention_chunk_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, kAttnSmem) == cudaSuccess;
        if (!opt2) throw std::runtime_error("ds41 cuda attention: shared memory opt-in");
        attention_chunk_kernel<2><<<dim3(kMaxChunks, v.heads / 4U, v.rows), 2 * 32, kAttnSmem, stream>>>(v);
    }
    STRIX_CUDA_CHECK(cudaGetLastError());
    if (g_attn_prefetch) attention_combine_kernel<true><<<dim3(v.heads, v.rows), 32, 0, stream>>>(v);
    else attention_combine_kernel<<<dim3(v.heads, v.rows), 32, 0, stream>>>(v);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

// ---- RoPE tables ----------------------------------------------------------------------------------------------------
RopeTables::RopeTables(std::uint32_t positions) : positions_(positions) {
    std::vector<RopeRow> h(positions);
    for (int k = 0; k < 2; ++k) {
        const RopeKind kind = k == 0 ? RopeKind::window : RopeKind::compressed;
        for (std::uint32_t p = 0; p < positions; ++p) h[p] = rope_row(kind, p);
        RopeRow*& d = k == 0 ? window_ : compressed_;
        STRIX_CUDA_CHECK(cudaMalloc(&d, rope_table_bytes(positions)));
        STRIX_CUDA_CHECK(cudaMemcpy(d, h.data(), sizeof(RopeRow) * positions, cudaMemcpyHostToDevice));
    }
}
RopeTables::~RopeTables() {
    cudaFree(window_);
    cudaFree(compressed_);
}

}  // namespace strix::ds41::cuda::attn
