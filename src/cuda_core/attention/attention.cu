// GQA decode attention over a BF16 KV cache, rows 1..32 with causal masking inside the window. See attention.hpp.
// Three kernels (scores, values, combine) parallel over position chunks; the reduction order is defined per 256-position chunk.
#include "strix/cuda_core/attention.hpp"

#include "../common.cuh"

namespace strix::cuda_core {
namespace {

constexpr std::uint32_t CH = kAttnChunk;  // 256 positions per chunk
constexpr int D = 256;

__global__ void kv_store_kernel(AttentionDims d, const float* k, const float* v, __nv_bfloat16* kc, __nv_bfloat16* vc,
                                const std::uint32_t* pos0p, RowMap rm) {
    pdl_trigger();
    pdl_wait();
    const std::uint32_t r = blockIdx.y, h = blockIdx.x;
    const std::uint32_t pos = rm.pos ? rm.pos[r] : *pos0p + r;
    const std::size_t so = rm.slot ? rm.slot[r] * rm.slot_stride : 0;
    const std::size_t src = (static_cast<std::size_t>(r) * d.kv_heads + h) * d.head_dim;
    const std::size_t dst = so + (static_cast<std::size_t>(h) * d.max_positions + pos) * d.head_dim;
    for (std::uint32_t i = threadIdx.x; i < d.head_dim; i += blockDim.x) {
        kc[dst + i] = __float2bfloat16_rn(k[src + i]);
        vc[dst + i] = __float2bfloat16_rn(v[src + i]);
    }
}

struct Scratch {
    float* scores;  // [rows][heads][max_positions]
    float* cmax;    // [rows][heads][maxchunks]
    float* dpart;   // [rows][heads][maxchunks]
    float* part;    // [rows][heads][maxchunks][D]
};
__host__ __device__ inline Scratch carve(float* base, const AttentionDims& d, std::uint32_t rows) {
    const std::size_t rh = static_cast<std::size_t>(rows) * d.heads, mc = attn_max_chunks(d);
    Scratch s;
    s.scores = base;
    s.cmax = s.scores + rh * d.max_positions;
    s.dpart = s.cmax + rh * mc;
    s.part = s.dpart + rh * mc;
    return s;
}

// Phase A: block (chunk, kv head, row); warp w handles positions base + w + 8 i for all G query heads of the group
// (the K row is read once). dot: lane owns 8 consecutive dims (sequential), xor butterfly 16..1; score = fl(dot) * scale.
__global__ void __launch_bounds__(256) attn_scores_kernel(AttentionDims d, const float* q, GroupedView qv, const __nv_bfloat16* kc,
                                                          const std::uint32_t* pos0p, std::uint32_t rows, Scratch sc, RowMap rm) {
    pdl_trigger();
    pdl_wait();
    const std::uint32_t chunk = blockIdx.x, kvh = blockIdx.y, r = blockIdx.z;
    const std::uint32_t len = (rm.pos ? rm.pos[r] : *pos0p + r) + 1, base = chunk * CH;
    if (base >= len) return;
    if (rm.slot) kc += rm.slot[r] * rm.slot_stride;
    const std::uint32_t G = d.heads / d.kv_heads, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const std::uint32_t mc = attn_max_chunks(d);
    float qr[8][8];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        if (g < static_cast<int>(G)) {
            const float* qp = q + static_cast<std::size_t>(r) * qv.rs() + static_cast<std::size_t>(kvh * G + g) * qv.gs() + lane * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) qr[g][j] = qp[j];
        }
    }
    float mx[8];
#pragma unroll
    for (int g = 0; g < 8; ++g) mx[g] = __int_as_float(0xff800000);
    const std::uint32_t end = min(len, base + CH);
    const __nv_bfloat16* kb = kc + static_cast<std::size_t>(kvh) * d.max_positions * D + lane * 8;
    for (std::uint32_t p = base + w; p < end; p += 8) {
        const uint4 u = __ldg(reinterpret_cast<const uint4*>(kb + static_cast<std::size_t>(p) * D));
        const float kf[8] = {bf16_lo(u.x), bf16_hi(u.x), bf16_lo(u.y), bf16_hi(u.y), bf16_lo(u.z), bf16_hi(u.z), bf16_lo(u.w), bf16_hi(u.w)};
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            if (g < static_cast<int>(G)) {
                float dot = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) dot = __fadd_rn(dot, __fmul_rn(qr[g][j], kf[j]));
                dot = warp_sum_xor(dot);
                const float s = __fmul_rn(dot, d.scale);
                mx[g] = fmaxf(mx[g], s);
                if (lane == 0) sc.scores[(static_cast<std::size_t>(r) * d.heads + kvh * G + g) * d.max_positions + p] = s;
            }
        }
    }
    __shared__ float wm[8][8];
#pragma unroll
    for (int g = 0; g < 8; ++g) wm[w][g] = mx[g];
    __syncthreads();
    if (threadIdx.x < G) {
        float m = wm[0][threadIdx.x];
        for (int i = 1; i < 8; ++i) m = fmaxf(m, wm[i][threadIdx.x]);
        sc.cmax[(static_cast<std::size_t>(r) * d.heads + kvh * G + threadIdx.x) * mc + chunk] = m;
    }
}

// Phase C: block (chunk, kv head, row); thread t = output dim and (for the weights) position base + t.
// e_p = expf(s_p - m) with the exact global max m; chunk denominator = block tree of the e (thread t holds position base + t);
// chunk value partial = sum over the chunk positions ascending of e_p * v[p][t] (separate rounded mul, add).
__global__ void __launch_bounds__(256) attn_values_kernel(AttentionDims d, const __nv_bfloat16* vc, const std::uint32_t* pos0p,
                                                          std::uint32_t rows, Scratch sc, RowMap rm) {
    pdl_trigger();
    pdl_wait();
    const std::uint32_t chunk = blockIdx.x, kvh = blockIdx.y, r = blockIdx.z;
    const std::uint32_t len = (rm.pos ? rm.pos[r] : *pos0p + r) + 1, base = chunk * CH;
    if (base >= len) return;
    if (rm.slot) vc += rm.slot[r] * rm.slot_stride;
    const std::uint32_t G = d.heads / d.kv_heads, t = threadIdx.x, lane = t & 31, w = t >> 5;
    const std::uint32_t mc = attn_max_chunks(d), nch = (len + CH - 1) / CH;
    __shared__ float mg[8];
    __shared__ float es[8][CH];
    __shared__ float red[8];
    if (w < G) {
        const float* cm = sc.cmax + (static_cast<std::size_t>(r) * d.heads + kvh * G + w) * mc;
        float m = __int_as_float(0xff800000);
        for (std::uint32_t c = lane; c < nch; c += 32) m = fmaxf(m, cm[c]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        if (lane == 0) mg[w] = m;
    }
    __syncthreads();
    const std::uint32_t p = base + t;
    for (std::uint32_t g = 0; g < G; ++g) {
        float e = 0.0f;
        if (p < len) e = cc_expf(__fsub_rn(sc.scores[(static_cast<std::size_t>(r) * d.heads + kvh * G + g) * d.max_positions + p], mg[g]));
        es[g][t] = e;
        const float ds = block_sum256(e, red);
        if (t == 0) sc.dpart[(static_cast<std::size_t>(r) * d.heads + kvh * G + g) * mc + chunk] = ds;
    }
    __syncthreads();
    float acc[8];
#pragma unroll
    for (int g = 0; g < 8; ++g) acc[g] = 0.0f;
    const std::uint32_t n = (len - base) < CH ? (len - base) : CH;
    const __nv_bfloat16* vp = vc + (static_cast<std::size_t>(kvh) * d.max_positions + base) * D + t;
#pragma unroll 4
    for (std::uint32_t i = 0; i < n; ++i) {
        const float vv = __bfloat162float(vp[static_cast<std::size_t>(i) * D]);
#pragma unroll
        for (int g = 0; g < 8; ++g)
            if (g < static_cast<int>(G)) acc[g] = __fadd_rn(acc[g], __fmul_rn(es[g][i], vv));
    }
#pragma unroll
    for (int g = 0; g < 8; ++g)
        if (g < static_cast<int>(G)) sc.part[((static_cast<std::size_t>(r) * d.heads + kvh * G + g) * mc + chunk) * D + t] = acc[g];
}

// Phase D: block (head, row), thread = dim. out = (sum over chunks ascending of the partials) / (sum of chunk denominators
// ascending), times sigmoid(gate).
__global__ void __launch_bounds__(256) attn_combine_kernel(AttentionDims d, const float* gate, GroupedView gv, const std::uint32_t* pos0p,
                                                           float* out, Scratch sc, RowMap rm) {
    pdl_trigger();
    pdl_wait();
    const std::uint32_t h = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    const std::uint32_t len = (rm.pos ? rm.pos[r] : *pos0p + r) + 1, nch = (len + CH - 1) / CH, mc = attn_max_chunks(d);
    const std::size_t rh = static_cast<std::size_t>(r) * d.heads + h;
    float denom = 0.0f, acc = 0.0f;
    for (std::uint32_t c = 0; c < nch; ++c) {
        denom = __fadd_rn(denom, sc.dpart[rh * mc + c]);
        acc = __fadd_rn(acc, sc.part[(rh * mc + c) * D + t]);
    }
    float o = __fdiv_rn(acc, denom);
    if (gate != nullptr) o = __fmul_rn(o, cc_sigmoid(gate[static_cast<std::size_t>(r) * gv.rs() + static_cast<std::size_t>(h) * gv.gs() + t]));
    out[rh * D + t] = o;
}

}  // namespace

cudaError_t kv_store(const AttentionDims& d, const float* k, const float* v, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
                     const std::uint32_t* pos0, std::uint32_t rows, cudaStream_t stream, const RowMap& rm) {
    if (rows == 0) return cudaErrorInvalidValue;
    return launch_pdl(kv_store_kernel, dim3(d.kv_heads, rows), dim3(256), stream, d, k, v, k_cache, v_cache, pos0, rm);
}

cudaError_t attention_decode(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate,
                             const GroupedView& gv, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                             const std::uint32_t* pos0, std::uint32_t rows, float* out, float* scratch, cudaStream_t stream, const RowMap& rm) {
    if (d.head_dim != 256 || rows == 0 || rows > kMaxRows || d.kv_heads == 0 || d.heads % d.kv_heads != 0 || d.heads / d.kv_heads > 8)
        return cudaErrorInvalidValue;
    const Scratch sc = carve(scratch, d, rows);
    const std::uint32_t mc = attn_max_chunks(d);
    if (auto e = launch_pdl(attn_scores_kernel, dim3(mc, d.kv_heads, rows), dim3(256), stream, d, q, qv, k_cache, pos0, rows, sc, rm);
        e != cudaSuccess)
        return e;
    if (auto e = launch_pdl(attn_values_kernel, dim3(mc, d.kv_heads, rows), dim3(256), stream, d, v_cache, pos0, rows, sc, rm); e != cudaSuccess)
        return e;
    return launch_pdl(attn_combine_kernel, dim3(d.heads, rows), dim3(256), stream, d, gate, gv, pos0, out, sc, rm);
}

}  // namespace strix::cuda_core
