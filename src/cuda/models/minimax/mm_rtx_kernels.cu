// MiniMax M3.1 Flash prefill on the RTX PRO: non-linear kernels in the engine's declared orders.
// See mm_rtx_kernels.cuh for the reference of each kernel.
#include "mm_rtx_kernels.cuh"

#include <chrono>
#include <cstdlib>
#include <stdexcept>

namespace strix::minimax::rtxpf::cuda {

namespace {

constexpr float kEps = 1e-6F, kScale = 0.08838834764831845F, kSumEps = 1e-20F, kRouteScale = 2.0F;

__device__ __forceinline__ float u2f(std::uint32_t b) { return __uint_as_float(b); }
__device__ __forceinline__ std::uint32_t f2u(float x) { return __float_as_uint(x); }
__device__ __forceinline__ float bf16_to_float(std::uint16_t b) { return u2f(static_cast<std::uint32_t>(b) << 16U); }
__device__ __forceinline__ std::uint16_t float_to_bf16(float x) {  // RNE (finite inputs), strix/minimax_msa.hpp
    const std::uint32_t b = f2u(x);
    return static_cast<std::uint16_t>((b + 0x7FFFU + ((b >> 16U) & 1U)) >> 16U);
}
__device__ __forceinline__ float fmaxs(float a, float b) { return a < b ? b : a; }  // std::max(a, b)
__device__ __forceinline__ float neg_inf() { return __uint_as_float(0xff800000U); }
__device__ float exp_f(float x) {  // strix/minimax_msa.hpp exp_f
    if (x < -87.0F) return 0.0F;
    if (x > 88.0F) return u2f(0x7f800000U);
    const float n = rintf(x * 1.44269504088896341F);
    float r = __fmaf_rn(n, -0.693145751953125F, x);
    r = __fmaf_rn(n, -1.42860676533018e-06F, r);
    float p = 2.48015873e-05F;
    p = __fmaf_rn(p, r, 1.98412698e-04F);
    p = __fmaf_rn(p, r, 1.38888889e-03F);
    p = __fmaf_rn(p, r, 8.33333333e-03F);
    p = __fmaf_rn(p, r, 4.16666667e-02F);
    p = __fmaf_rn(p, r, 1.66666667e-01F);
    p = __fmaf_rn(p, r, 0.5F);
    p = __fmaf_rn(p, r, 1.0F);
    p = __fmaf_rn(p, r, 1.0F);
    return p * u2f(static_cast<std::uint32_t>(static_cast<int>(n) + 127) << 23U);
}
__device__ __forceinline__ float shx(float v, int d) { return __shfl_xor_sync(0xffffffffU, v, d, 32); }
// Butterfly of strix/minimax_msa.hpp (v = v + v[lane ^ d], d = 16..1): every lane ends with lanes[0]'s value.
__device__ __forceinline__ float butterfly(float v) {
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) v = v + shx(v, d);
    return v;
}
// dot128 of one thread: the 32 lane chains (4 dims each, in order) and the butterfly's tree for lanes[0].
__device__ __forceinline__ float dot128_thread(const float* a, const float* b) {
    float c[32];
#pragma unroll
    for (int l = 0; l < 32; ++l) {
        float acc = 0.0F;
#pragma unroll
        for (int i = 0; i < 4; ++i) acc = __fmaf_rn(a[4 * l + i], b[4 * l + i], acc);
        c[l] = acc;
    }
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1)
#pragma unroll
        for (int l = 0; l < d; ++l) c[l] = c[l] + c[l + d];
    return c[0];
}

__global__ void embed_kernel(const std::uint16_t* table, const std::uint32_t* tokens, float* out) {
    const std::uint16_t* src = table + static_cast<std::size_t>(tokens[blockIdx.x]) * kHidden;
    float* dst = out + static_cast<std::size_t>(blockIdx.x) * kHidden;
    for (unsigned i = threadIdx.x; i < kHidden; i += 256) dst[i] = bf16_to_float(src[i]);
}

// Gemma norm of one 6144 row (thread t chains j = t, t + 256, ...; tree 128..1; r = 1 / sqrt(p / 6144 + eps)).
__device__ void norm_row(const float* x, const float* w, float* y, float* red) {
    float s = 0.0F;
    for (unsigned i = threadIdx.x; i < kHidden; i += 256) s = __fmaf_rn(x[i], x[i], s);
    red[threadIdx.x] = s;
    __syncthreads();
    for (unsigned k = 128; k > 0; k >>= 1U) {
        if (threadIdx.x < k) red[threadIdx.x] = red[threadIdx.x] + red[threadIdx.x + k];
        __syncthreads();
    }
    const float r = 1.0F / sqrtf(red[0] / static_cast<float>(kHidden) + kEps);
    for (unsigned i = threadIdx.x; i < kHidden; i += 256) y[i] = (x[i] * r) * (1.0F + w[i]);
}
__global__ void rmsnorm_kernel(const float* x, const float* w, float* y) {
    __shared__ float red[256];
    const std::size_t o = static_cast<std::size_t>(blockIdx.x) * kHidden;
    norm_row(x + o, w, y + o, red);
}
__global__ void post_exchange_kernel(float* h, const float* p0, const float* p1, const float* w, float* y) {
    __shared__ float red[256];
    const std::size_t o = static_cast<std::size_t>(blockIdx.x) * kHidden;
    for (unsigned i = threadIdx.x; i < kHidden; i += 256) h[o + i] = h[o + i] + (p0[o + i] + p1[o + i]);
    __syncthreads();
    norm_row(h + o, w, y + o, red);
}

// Router logits: one warp per (row, expert): lane l chains j = l, l + 32, ...; butterfly; s = sigmoid, c = s + bias.
__global__ void router_logits_kernel(const float* y, const float* rw, const float* bias, std::size_t rows, float* scratch) {
    const std::size_t warp = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const unsigned lane = threadIdx.x & 31U;
    if (warp >= rows * kExperts) return;
    const std::size_t r = warp / kExperts;
    const unsigned e = static_cast<unsigned>(warp % kExperts);
    const float* yr = y + r * kHidden;
    const float* we = rw + static_cast<std::size_t>(e) * kHidden;
    float acc = 0.0F;
    for (unsigned j = lane; j < kHidden; j += 32) acc = __fmaf_rn(yr[j], we[j], acc);
    acc = butterfly(acc);
    if (lane == 0) {
        const float s = 1.0F / (1.0F + exp_f(-acc));
        scratch[(r * kExperts + e) * 2] = s;
        scratch[(r * kExperts + e) * 2 + 1] = s + bias[e];
    }
}
__global__ void router_top4_kernel(const float* scratch, std::size_t rows, std::uint32_t* ids, float* weights) {
    const std::size_t r = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* sc = scratch + r * kExperts * 2;
    std::uint32_t pick[kTopE];
    for (unsigned k = 0; k < kTopE; ++k) {  // (c desc, id asc), excluding the earlier picks
        std::uint32_t best = 0xffffffffU;
        for (unsigned e = 0; e < kExperts; ++e) {
            bool taken = false;
            for (unsigned j = 0; j < k; ++j) taken |= pick[j] == e;
            if (taken) continue;
            if (best == 0xffffffffU || sc[e * 2 + 1] > sc[best * 2 + 1]) best = e;
        }
        pick[k] = best;
    }
    float sum = 0.0F;
    for (unsigned k = 0; k < kTopE; ++k) sum = sum + sc[pick[k] * 2];
    for (unsigned k = 0; k < kTopE; ++k) {
        ids[r * kTopE + k] = pick[k];
        weights[r * kTopE + k] = (sc[pick[k] * 2] / (sum + kSumEps)) * kRouteScale;
    }
}

// Index projections (engine dense_t16_kernel arithmetic): block = 16 rows x 128 outputs, inputs staged in 64-wide tiles.
__global__ void dense_t_kernel(const float* x, const float* wt, float* y, std::size_t rows, unsigned n, unsigned ys) {
    __shared__ float xs[16][64];
    const std::size_t t0 = static_cast<std::size_t>(blockIdx.x) * 16;
    const unsigned o = blockIdx.y * 128 + threadIdx.x;
    float acc[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) acc[t] = 0.0F;
    for (unsigned i0 = 0; i0 < kHidden; i0 += 64) {
        __syncthreads();
        for (unsigned j = threadIdx.x; j < 16 * 64; j += 128) {
            const std::size_t t = j / 64;
            xs[t][j % 64] = t0 + t < rows ? x[(t0 + t) * kHidden + i0 + j % 64] : 0.0F;
        }
        __syncthreads();
        if (o < n)
            for (unsigned i = 0; i < 64; ++i) {
                const float w = wt[static_cast<std::size_t>(i0 + i) * n + o];
#pragma unroll
                for (int t = 0; t < 16; ++t) acc[t] = __fmaf_rn(xs[t][i], w, acc[t]);
            }
    }
    if (o < n)
        for (std::size_t t = 0; t < 16 && t0 + t < rows; ++t) y[(t0 + t) * ys + o] = acc[t];
}

// head_norm_rope on one 128 head held by a warp (lane l: dims 4l..4l+3). Returns the lane's 4 values.
__device__ void head_norm_rope_warp(float (&x)[4], const float* w, const RopeRow& rope, unsigned lane) {
    float acc = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i) acc = __fmaf_rn(x[i], x[i], acc);
    acc = butterfly(acc);
    const float r = 1.0F / sqrtf(acc / 128.0F + kEps);
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = (x[i] * r) * (1.0F + w[4 * lane + i]);
    float partner[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) partner[i] = __shfl_sync(0xffffffffU, x[i], lane < 8 ? lane + 8 : (lane < 16 ? lane - 8 : lane), 32);
    if (lane < 8) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned d = 4 * lane + i;
            x[i] = __fmaf_rn(-partner[i], rope.s[d], x[i] * rope.c[d]);  // a' = fmaf(-b, sin, a cos)
        }
    } else if (lane < 16) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned d = 4 * (lane - 8) + i;
            x[i] = __fmaf_rn(partner[i], rope.s[d], x[i] * rope.c[d]);  // b' = fmaf(a, sin, b cos)
        }
    }
}

// Prep: grid (rows, 73 heads: 64 q, 4 k, 4 iq, 1 ik) + 4 v heads copied; one warp per block.
__global__ void msa_prep_kernel(const SeqView* seqs, const std::uint32_t* row_seq, MsaLayer L, float* q, const float* k, const float* v,
                                float* iq, const float* ik, const RopeRow* rope) {
    const std::size_t row = blockIdx.x;
    const unsigned head = blockIdx.y, lane = threadIdx.x;
    const SeqView sq = seqs[row_seq[row]];
    const std::uint64_t pos = row - sq.row0 + sq.pos0;
    const RopeRow& rr = rope[pos];
    float x[4];
    if (head < 64) {
        float* p = q + (row * 64 + head) * 128 + 4 * lane;
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = p[i];
        head_norm_rope_warp(x, L.q_w, rr, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) p[i] = x[i];
    } else if (head < 68) {
        const unsigned g = head - 64;
        const float* p = k + (row * 4 + g) * 128 + 4 * lane;
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = p[i];
        head_norm_rope_warp(x, L.k_w, rr, lane);
        std::uint16_t* c = sq.k + ((static_cast<std::size_t>(L.layer) * sq.cap + pos) * 4 + g) * 128 + 4 * lane;
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] = float_to_bf16(x[i]);
        const float* pv = v + (row * 4 + g) * 128 + 4 * lane;
        std::uint16_t* cv = sq.v + ((static_cast<std::size_t>(L.layer) * sq.cap + pos) * 4 + g) * 128 + 4 * lane;
#pragma unroll
        for (int i = 0; i < 4; ++i) cv[i] = float_to_bf16(pv[i]);
    } else if (head < 72) {
        float* p = iq + (row * 4 + (head - 68)) * 128 + 4 * lane;
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = p[i];
        head_norm_rope_warp(x, L.iq_w, rr, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) p[i] = x[i];
    } else {
        const float* p = ik + row * 128 + 4 * lane;
#pragma unroll
        for (int i = 0; i < 4; ++i) x[i] = p[i];
        head_norm_rope_warp(x, L.ik_w, rr, lane);
        std::uint16_t* c = sq.ik + (static_cast<std::size_t>(L.layer) * sq.cap + pos) * 128 + 4 * lane;
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] = float_to_bf16(x[i]);
    }
}

// Index block scores: one warp per (row, index head g, block b); lane j scores keys b*128 + j + 32 m (full dot128 per
// lane, the reference's lane chains + tree), max over the visible keys (comparisons only feed the selection).
__global__ void index_scores_kernel(const SeqView* seqs, const std::uint32_t* row_seq, std::size_t rows, unsigned max_blocks,
                                    unsigned layer, const float* iq, float* scores) {
    __shared__ float qs[4][128];
    const unsigned w = threadIdx.x / 32, lane = threadIdx.x & 31U;
    const std::size_t item = static_cast<std::size_t>(blockIdx.x) * 4 + w;  // (row, g, b)
    const std::size_t row = item / (4 * max_blocks);
    const unsigned g = static_cast<unsigned>(item / max_blocks % 4), b = static_cast<unsigned>(item % max_blocks);
    const bool live = row < rows;
    SeqView sq{};
    std::uint64_t count = 0;
    if (live) {
        sq = seqs[row_seq[row]];
        count = row - sq.row0 + sq.pos0 + 1;
        for (unsigned i = lane; i < 128; i += 32) qs[w][i] = iq[(row * 4 + g) * 128 + i];
    }
    __syncwarp();
    if (!live || static_cast<std::uint64_t>(b) * kBlock >= count) return;
    const std::uint64_t blocks = (count + kBlock - 1) / kBlock;
    if (blocks <= kTopK) return;  // dense: no scores needed
    float best = neg_inf();
    float kv[128];
    for (unsigned m = 0; m < 4; ++m) {
        const std::uint64_t t = static_cast<std::uint64_t>(b) * kBlock + lane + 32 * m;
        if (t >= count) break;
        const std::uint16_t* kc = sq.ik + (static_cast<std::size_t>(layer) * sq.cap + t) * 128;
#pragma unroll 8
        for (int i = 0; i < 128; ++i) kv[i] = bf16_to_float(kc[i]);
        best = fmaxs(best, dot128_thread(qs[w], kv) * kScale);
    }
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) best = fmaxs(best, shx(best, d));
    if (lane == 0) scores[(row * 4 + g) * max_blocks + b] = best;
}

// Selection per (row, g): all blocks when there are at most 16; else the newest + the best 15 others (score desc, id asc),
// ascending block order.
__global__ void select_kernel(const SeqView* seqs, const std::uint32_t* row_seq, std::size_t rows, unsigned max_blocks,
                              const float* scores, std::uint32_t* sel, std::uint32_t* sel_count) {
    const std::size_t item = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= rows * 4) return;
    const std::size_t row = item / 4;
    const SeqView sq = seqs[row_seq[row]];
    const std::uint64_t count = row - sq.row0 + sq.pos0 + 1, blocks = (count + kBlock - 1) / kBlock;
    std::uint32_t* out = sel + item * kTopK;
    if (blocks <= kTopK) {
        for (std::uint32_t b = 0; b < blocks; ++b) out[b] = b;
        sel_count[item] = static_cast<std::uint32_t>(blocks);
        return;
    }
    const float* sc = scores + item * max_blocks;
    const auto newest = static_cast<std::uint32_t>(blocks - 1);
    std::uint32_t pick[kTopK];
    for (unsigned k = 0; k + 1 < kTopK; ++k) {
        std::uint32_t best = 0xffffffffU;
        for (std::uint32_t b = 0; b < newest; ++b) {
            bool taken = false;
            for (unsigned j = 0; j < k; ++j) taken |= pick[j] == b;
            if (taken) continue;
            if (best == 0xffffffffU || sc[b] > sc[best]) best = b;  // ties keep the smaller id
        }
        pick[k] = best;
    }
    pick[kTopK - 1] = newest;
    for (unsigned i = 1; i < kTopK; ++i)  // ascending
        for (unsigned j = i; j > 0 && pick[j - 1] > pick[j]; --j) {
            const std::uint32_t tmp = pick[j];
            pick[j] = pick[j - 1];
            pick[j - 1] = tmp;
        }
    for (unsigned i = 0; i < kTopK; ++i) out[i] = pick[i];
    sel_count[item] = kTopK;
}

// Sparse attention: one warp per (row, query head h); lane l owns dims 4l..4l+3.
__global__ void attend_kernel(const SeqView* seqs, const std::uint32_t* row_seq, std::size_t rows, unsigned layer, const float* q,
                              const std::uint32_t* sel, const std::uint32_t* sel_count, float* o) {
    const std::size_t warp = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const unsigned lane = threadIdx.x & 31U;
    if (warp >= rows * 64) return;
    const std::size_t row = warp / 64;
    const unsigned h = static_cast<unsigned>(warp % 64), g = h / 16;
    const SeqView sq = seqs[row_seq[row]];
    const std::uint64_t count = row - sq.row0 + sq.pos0 + 1;
    float qv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) qv[i] = q[(row * 64 + h) * 128 + 4 * lane + i];
    const std::uint32_t* blocks = sel + (row * 4 + g) * kTopK;
    const unsigned chunks = sel_count[row * 4 + g];
    float cm[kTopK], cl[kTopK], cacc[kTopK][4];
    for (unsigned c = 0; c < chunks; ++c) {
        float m = neg_inf(), l = 0.0F, acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        bool first = true;
        const std::uint64_t lo = static_cast<std::uint64_t>(blocks[c]) * kBlock;
        const std::uint64_t hi = count < lo + kBlock ? count : lo + kBlock;
        for (std::uint64_t t0 = lo; t0 < hi; t0 += 16) {
            const unsigned tn = static_cast<unsigned>((hi - t0) < 16 ? (hi - t0) : 16);
            float s[16];
            float tm = neg_inf();
            for (unsigned j = 0; j < tn; ++j) {
                const std::uint16_t* kc = sq.k + ((static_cast<std::size_t>(layer) * sq.cap + t0 + j) * 4 + g) * 128 + 4 * lane;
                float part = 0.0F;
#pragma unroll
                for (int i = 0; i < 4; ++i) part = __fmaf_rn(qv[i], bf16_to_float(kc[i]), part);
                s[j] = butterfly(part) * kScale;
                tm = fmaxs(tm, s[j]);
            }
            const float mn = fmaxs(m, tm);
            const float corr = first ? 0.0F : exp_f(m - mn);
            l = l * corr;
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[i] = acc[i] * corr;
            for (unsigned j = 0; j < tn; ++j) {
                const float p = exp_f(s[j] - mn);
                l = l + p;
                const std::uint16_t* vc = sq.v + ((static_cast<std::size_t>(layer) * sq.cap + t0 + j) * 4 + g) * 128 + 4 * lane;
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[i] = __fmaf_rn(p, bf16_to_float(vc[i]), acc[i]);
            }
            m = mn;
            first = false;
        }
        cm[c] = m;
        cl[c] = l;
#pragma unroll
        for (int i = 0; i < 4; ++i) cacc[c][i] = acc[i];
    }
    float big = neg_inf();
    for (unsigned c = 0; c < chunks; ++c) big = fmaxs(big, cm[c]);
    float L = 0.0F, e[kTopK];
    for (unsigned c = 0; c < chunks; ++c) {
        e[c] = exp_f(cm[c] - big);
        L = __fmaf_rn(cl[c], e[c], L);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a = 0.0F;
        for (unsigned c = 0; c < chunks; ++c) a = __fmaf_rn(cacc[c][i], e[c], a);
        o[(row * 64 + h) * 128 + 4 * lane + i] = a / L;
    }
}

// Index block scores, tiled (same arithmetic as index_scores_kernel): CTA = (32-row tile of ONE sequence, index head g,
// block b); the block's 128 keys and the tile's 32 query rows staged in shared memory as FP32; thread t scores rows
// t / 8 against keys (t % 8) + 8 j: per dot the reference's 32 lane chains (one float4 of each operand) and its tree;
// max over the visible keys (comparison only, so the reduction order does not matter). Key rows are padded to 33 float4
// (the 8 lanes of a quarter-warp read 8 different keys: at 32 float4 per row they all hit the same banks).
constexpr unsigned kKs4 = 33;
__global__ __launch_bounds__(256) void index_scores_tiled_kernel(const SeqView* seqs, const std::uint32_t* row_seq, const uint2* tiles,
                                                                 unsigned max_blocks, unsigned layer, const float* iq, float* scores) {
    extern __shared__ float4 smem4[];
    float4* ks = smem4;                 // [128 keys][kKs4 float4]
    float4* qs = smem4 + 128 * kKs4;    // [32 rows][32 float4]
    const uint2 tile = tiles[blockIdx.x];  // (first row, rows)
    const unsigned g = blockIdx.y, b = blockIdx.z, t = threadIdx.x;
    const SeqView sq = seqs[row_seq[tile.x]];
    const std::uint64_t last_count = tile.x + tile.y - sq.row0 + sq.pos0;  // visible positions of the tile's last row
    if (static_cast<std::uint64_t>(b) * kBlock >= last_count || (last_count + kBlock - 1) / kBlock <= kTopK) return;
    for (unsigned i = t; i < 128 * 32; i += 256) {
        const unsigned key = i / 32, q4 = i % 32;
        const std::uint64_t pos = static_cast<std::uint64_t>(b) * kBlock + key;
        float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        if (pos < last_count) {
            const std::uint16_t* kc = sq.ik + (static_cast<std::size_t>(layer) * sq.cap + pos) * 128 + 4 * q4;
            v = make_float4(bf16_to_float(kc[0]), bf16_to_float(kc[1]), bf16_to_float(kc[2]), bf16_to_float(kc[3]));
        }
        ks[key * kKs4 + q4] = v;
    }
    for (unsigned i = t; i < 32 * 32; i += 256) {
        const unsigned r = i / 32, q4 = i % 32;
        qs[i] = r < tile.y ? *reinterpret_cast<const float4*>(iq + ((static_cast<std::size_t>(tile.x) + r) * 4 + g) * 128 + 4 * q4)
                           : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    }
    __syncthreads();
    const unsigned r = t / 8;
    const std::uint64_t count = static_cast<std::uint64_t>(tile.x) + r - sq.row0 + sq.pos0 + 1;
    float best = neg_inf();
    if (r < tile.y && (count + kBlock - 1) / kBlock > kTopK) {
        for (unsigned j = 0; j < 16; ++j) {
            const unsigned key = (t % 8) + 8 * j;
            if (static_cast<std::uint64_t>(b) * kBlock + key >= count) continue;
            float c[32];
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                const float4 a = qs[r * 32 + l], k = ks[key * kKs4 + l];
                float acc = __fmaf_rn(a.x, k.x, 0.0F);
                acc = __fmaf_rn(a.y, k.y, acc);
                acc = __fmaf_rn(a.z, k.z, acc);
                acc = __fmaf_rn(a.w, k.w, acc);
                c[l] = acc;
            }
#pragma unroll
            for (int d = 16; d >= 1; d >>= 1)
#pragma unroll
                for (int l = 0; l < d; ++l) c[l] = c[l] + c[l + d];
            best = fmaxs(best, c[0] * kScale);
        }
    }
#pragma unroll
    for (int d = 4; d >= 1; d >>= 1) best = fmaxs(best, __shfl_xor_sync(0xffffffffU, best, d, 8));
    if ((t % 8) == 0 && r < tile.y && (count + kBlock - 1) / kBlock > kTopK && static_cast<std::uint64_t>(b) * kBlock < count)
        scores[((static_cast<std::size_t>(tile.x) + r) * 4 + g) * max_blocks + b] = best;
}

// Sparse attention, shared chunks + key-parallel scores (same arithmetic as attend_kernel): CTA = (row, KV group g), 16
// warps = the group's 16 query heads; each selected block's K and V rows (BF16) are staged once in shared memory. Lane j
// computes the whole score of key j (the reference's 32 lane chains + tree, in registers) and its p = exp_f(s - m');
// the online-softmax state then advances key by key in the reference order, p broadcast by one shuffle, while lane l
// keeps the PV accumulators of dims 4l..4l+3. K rows are padded to 132 BF16 (264 B): lane j reads row j, and at the
// natural 256-B stride every lane hit the same banks (32-way conflicts on each score load); V is read row-wise.
constexpr unsigned kKStride = 132;
__global__ __launch_bounds__(512) void attend_shared_kernel(const SeqView* seqs, const std::uint32_t* row_seq, unsigned layer, const float* q,
                                                            const std::uint32_t* sel, const std::uint32_t* sel_count, float* o,
                                                            int diag) {
    extern __shared__ uint4 smem16[];
    std::uint16_t* kc = reinterpret_cast<std::uint16_t*>(smem16);  // [128][kKStride] BF16
    std::uint16_t* vc = kc + 128 * kKStride;                        // [128][128] BF16
    float* qs = reinterpret_cast<float*>(vc + 128 * 128);           // [16 heads][128]
    const std::size_t row = blockIdx.x;
    const unsigned g = blockIdx.y, w = threadIdx.x / 32, lane = threadIdx.x & 31U, h = 16 * g + w;
    const SeqView sq = seqs[row_seq[row]];
    const std::uint64_t count = row - sq.row0 + sq.pos0 + 1;
    for (unsigned i = lane; i < 128; i += 32) qs[w * 128 + i] = q[(row * 64 + h) * 128 + i];
    const float* qh = qs + w * 128;
    const std::uint32_t* blocks = sel + (row * 4 + g) * kTopK;
    const unsigned chunks = sel_count[row * 4 + g];
    float cm[kTopK], cl[kTopK], cacc[kTopK][4];
    for (unsigned c = 0; c < chunks; ++c) {
        const std::uint64_t lo = static_cast<std::uint64_t>(blocks[c]) * kBlock;
        const std::uint64_t hi = count < lo + kBlock ? count : lo + kBlock;
        const unsigned keys = static_cast<unsigned>(hi - lo);
        __syncthreads();  // the previous chunk is consumed (and qs is written before the first)
        for (unsigned i = threadIdx.x; diag != 2 && i < keys * 16; i += 512) {  // 16 uint4 (8 BF16 each) per key row
            const unsigned j = i / 16, part = i % 16;
            const std::size_t at = ((static_cast<std::size_t>(layer) * sq.cap + lo + j) * 4 + g) * 128;
            const uint4 kv = reinterpret_cast<const uint4*>(sq.k + at)[part];
            uint2* kd = reinterpret_cast<uint2*>(kc + j * kKStride + 8 * part);  // 8-B aligned rows
            kd[0] = make_uint2(kv.x, kv.y);
            kd[1] = make_uint2(kv.z, kv.w);
            reinterpret_cast<uint4*>(vc)[j * 16 + part] = reinterpret_cast<const uint4*>(sq.v + at)[part];
        }
        __syncthreads();
        float m = neg_inf(), l = 0.0F, acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        bool first = true;
        for (unsigned g0 = 0; diag != 1 && g0 < keys; g0 += 32) {  // 32 keys = two softmax tiles of 16
            const unsigned key = g0 + lane;
            float sc = neg_inf();
            if (key < keys) {
                const std::uint16_t* kr = kc + key * kKStride;
                float cc[32];
#pragma unroll
                for (int ll = 0; ll < 32; ++ll) {
                    const uint2 k2 = *reinterpret_cast<const uint2*>(kr + 4 * ll);
                    float a = __fmaf_rn(qh[4 * ll], u2f(k2.x << 16U), 0.0F);
                    a = __fmaf_rn(qh[4 * ll + 1], u2f(k2.x & 0xffff0000U), a);
                    a = __fmaf_rn(qh[4 * ll + 2], u2f(k2.y << 16U), a);
                    a = __fmaf_rn(qh[4 * ll + 3], u2f(k2.y & 0xffff0000U), a);
                    cc[ll] = a;
                }
#pragma unroll
                for (int d = 16; d >= 1; d >>= 1)
#pragma unroll
                    for (int ll = 0; ll < d; ++ll) cc[ll] = cc[ll] + cc[ll + d];
                sc = cc[0] * kScale;
            }
            for (unsigned half = 0; half < 2; ++half) {
                const unsigned t0 = g0 + 16 * half;
                if (t0 >= keys) break;
                const unsigned tn = (keys - t0) < 16 ? (keys - t0) : 16;
                // tile max over the tile's keys (lanes 16 half .. 16 half + tn - 1); comparison only
                float tm = (lane / 16 == half && lane % 16 < tn) ? sc : neg_inf();
#pragma unroll
                for (int d = 16; d >= 1; d >>= 1) tm = fmaxs(tm, shx(tm, d));
                const float mn = fmaxs(m, tm);
                const float corr = first ? 0.0F : exp_f(m - mn);
                l = l * corr;
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[i] = acc[i] * corr;
                const float pl = (lane / 16 == half && lane % 16 < tn) ? exp_f(sc - mn) : 0.0F;
                for (unsigned j = 0; j < tn; ++j) {
                    const float pj = __shfl_sync(0xffffffffU, pl, 16 * half + j, 32);
                    l = l + pj;
                    const uint2 v2 = *reinterpret_cast<const uint2*>(vc + (t0 + j) * 128 + 4 * lane);
                    acc[0] = __fmaf_rn(pj, u2f(v2.x << 16U), acc[0]);
                    acc[1] = __fmaf_rn(pj, u2f(v2.x & 0xffff0000U), acc[1]);
                    acc[2] = __fmaf_rn(pj, u2f(v2.y << 16U), acc[2]);
                    acc[3] = __fmaf_rn(pj, u2f(v2.y & 0xffff0000U), acc[3]);
                }
                m = mn;
                first = false;
            }
        }
        cm[c] = m;
        cl[c] = l;
#pragma unroll
        for (int i = 0; i < 4; ++i) cacc[c][i] = acc[i];
    }
    float big = neg_inf();
    for (unsigned c = 0; c < chunks; ++c) big = fmaxs(big, cm[c]);
    float L = 0.0F, e[kTopK];
    for (unsigned c = 0; c < chunks; ++c) {
        e[c] = exp_f(cm[c] - big);
        L = __fmaf_rn(cl[c], e[c], L);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a = 0.0F;
        for (unsigned c = 0; c < chunks; ++c) a = __fmaf_rn(cacc[c][i], e[c], a);
        o[(row * 64 + h) * 128 + 4 * lane + i] = a / L;
    }
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

void embed(const std::uint16_t* table, const std::uint32_t* tokens, std::size_t rows, float* out, cudaStream_t s) {
    if (rows) embed_kernel<<<static_cast<unsigned>(rows), 256, 0, s>>>(table, tokens, out);
}
void rmsnorm(const float* x, const float* w, float* y, std::size_t rows, cudaStream_t s) {
    if (rows) rmsnorm_kernel<<<static_cast<unsigned>(rows), 256, 0, s>>>(x, w, y);
}
void post_exchange(float* h, const float* p0, const float* p1, const float* w, float* y, std::size_t rows, cudaStream_t s) {
    if (rows) post_exchange_kernel<<<static_cast<unsigned>(rows), 256, 0, s>>>(h, p0, p1, w, y);
}
void route(const float* y, const float* router_w, const float* bias, std::size_t rows, float* scratch, std::uint32_t* ids, float* weights,
           cudaStream_t s) {
    if (!rows) return;
    const std::size_t warps = rows * kExperts;
    router_logits_kernel<<<static_cast<unsigned>((warps * 32 + 255) / 256), 256, 0, s>>>(y, router_w, bias, rows, scratch);
    router_top4_kernel<<<static_cast<unsigned>((rows + 127) / 128), 128, 0, s>>>(scratch, rows, ids, weights);
}
void dense_t(const float* x, const float* wt, float* y, std::size_t rows, unsigned n, unsigned ys, cudaStream_t s) {
    if (rows) dense_t_kernel<<<dim3(static_cast<unsigned>((rows + 15) / 16), (n + 127) / 128), 128, 0, s>>>(x, wt, y, rows, n, ys);
}
void msa_prep(const SeqView* seqs_d, const std::uint32_t* row_seq, std::size_t rows, const MsaLayer& L, float* q, const float* k, const float* v,
              float* iq, const float* ik, const RopeRow* rope, float*, cudaStream_t s) {
    if (rows) msa_prep_kernel<<<dim3(static_cast<unsigned>(rows), 73), 32, 0, s>>>(seqs_d, row_seq, L, q, k, v, iq, ik, rope);
}
double g_attn_prof[3] = {0, 0, 0};  // MM_RTX_PROFILE: scores / select / attend seconds (synchronised)
namespace {
struct SubLap {
    bool on;
    std::chrono::steady_clock::time_point t;
    explicit SubLap(bool o) : on(o) { if (on) { cudaDeviceSynchronize(); t = std::chrono::steady_clock::now(); } }
    void lap(int i) {
        if (!on) return;
        cudaDeviceSynchronize();
        const auto n = std::chrono::steady_clock::now();
        g_attn_prof[i] += std::chrono::duration<double>(n - t).count();
        t = n;
    }
};
}  // namespace
void msa_attend(const SeqView* seqs_d, const std::uint32_t* row_seq, std::size_t rows, unsigned max_blocks, unsigned layer, const float* q,
                const float* iq, float* scores, std::uint32_t* sel, std::uint32_t* sel_count, float* o, cudaStream_t s, const uint2* tiles,
                unsigned n_tiles, bool legacy) {
    if (!rows) return;
    static const bool prof = std::getenv("MM_RTX_PROFILE") != nullptr;
    SubLap sl(prof);
    if (legacy) {
        const std::size_t items = rows * 4 * max_blocks;
        index_scores_kernel<<<static_cast<unsigned>((items + 3) / 4), 128, 0, s>>>(seqs_d, row_seq, rows, max_blocks, layer, iq, scores);
    } else if (max_blocks > kTopK) {
        static bool attr = false;
        constexpr int kSmem = (128 * kKs4 + 32 * 32) * 16;
        if (!attr) {
            check(cudaFuncSetAttribute(index_scores_tiled_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem), "smem attr");
            attr = true;
        }
        index_scores_tiled_kernel<<<dim3(n_tiles, 4, max_blocks), 256, kSmem, s>>>(seqs_d, row_seq, tiles, max_blocks, layer, iq, scores);
    }
    sl.lap(0);
    select_kernel<<<static_cast<unsigned>((rows * 4 + 127) / 128), 128, 0, s>>>(seqs_d, row_seq, rows, max_blocks, scores, sel, sel_count);
    sl.lap(1);
    if (legacy)
        attend_kernel<<<static_cast<unsigned>((rows * 64 * 32 + 255) / 256), 256, 0, s>>>(seqs_d, row_seq, rows, layer, q, sel, sel_count, o);
    else
    {
        static bool attr2 = false;
        constexpr int kSmem2 = 128 * kKStride * 2 + 128 * 128 * 2 + 16 * 128 * 4;
        if (!attr2) {
            check(cudaFuncSetAttribute(attend_shared_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem2), "smem attr");
            attr2 = true;
        }
        // MM_RTX_ATTN_DIAG (timing diagnosis only, wrong output): 1 = staging without compute, 2 = compute without staging
        static const int diag = std::getenv("MM_RTX_ATTN_DIAG") ? std::atoi(std::getenv("MM_RTX_ATTN_DIAG")) : 0;
        attend_shared_kernel<<<dim3(static_cast<unsigned>(rows), 4), 512, kSmem2, s>>>(seqs_d, row_seq, layer, q, sel, sel_count, o, diag);
    }
    sl.lap(2);
}
void export_layer(const SeqView& seq, unsigned layer, unsigned rank, std::uint64_t n, std::uint8_t* dst, cudaStream_t s) {
    export_layer(seq, layer, rank, 0, n, dst, s);
}
void export_layer(const SeqView& seq, unsigned layer, unsigned rank, std::uint64_t r0, std::uint64_t n, std::uint8_t* dst, cudaStream_t s) {
    if (r0 > seq.cap || n > seq.cap - r0 || rank >= 2) throw std::invalid_argument("export_layer: bad range/rank");
    const std::size_t base = static_cast<std::size_t>(layer) * seq.cap + r0;
    // K and V: rows of 4 heads x 128 BF16 (1024 B) in the cache, this rank's 2 heads (512 B) per row in the blob
    check(cudaMemcpy2DAsync(dst, 512, reinterpret_cast<const std::uint8_t*>(seq.k + base * 512) + rank * 512, 1024, 512, n,
                            cudaMemcpyDeviceToHost, s), "export k");
    check(cudaMemcpy2DAsync(dst + n * 512, 512, reinterpret_cast<const std::uint8_t*>(seq.v + base * 512) + rank * 512, 1024, 512, n,
                            cudaMemcpyDeviceToHost, s), "export v");
    check(cudaMemcpyAsync(dst + n * 1024, seq.ik + base * 128, n * 256, cudaMemcpyDeviceToHost, s), "export ik");
}

}  // namespace strix::minimax::rtxpf::cuda
