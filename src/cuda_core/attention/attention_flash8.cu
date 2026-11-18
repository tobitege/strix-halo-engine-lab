// Prefill attention, FAST8 class: causal flash attention with both products on e4m3 tensor cores (mma.sync m16n8k32 e4m3, FP32
// accumulate), head_dim 256, GQA. See attention.hpp attention_prefill_flash8 / kv_shadow_e4m3 for the contract (NOT bitwise equal to
// attention_prefill_flash; gated by the FAST quality gate).
//
// Shadow cache (written by kv_shadow_e4m3 from the BF16 cache, per kv head): K8 [P8][256] e4m3 with a per-position scale ks (k = K8 * ks),
// V8T [256][P8] e4m3 transposed with a per-position scale vs (v = V8T * vs); inside every 32-key group the V8T keys are stored in the
// mma k order (byte b of a group holds key 16 w + 2 q + (i & 1) + 8 (i >> 1), b = 8 q + 4 w + i), so a thread reads its B fragment
// with one LDS.64. P8 = 0 rounds the shadow capacity up to a multiple of 64 (kAttn8Align).
//
// CTA = 128 query rows x 1 query head, 8 warps of 16 rows, key tiles of 64 through a 3-stage cp.async ring (one barrier per tile).
// Q is quantised per (row, head) to e4m3 in registers (scale sq = amax / 448). S = (Q8 K8^T) * sq * ks * scale (log2 domain), online
// softmax FP32 (exp2), P'' = p * vs / c_t (c_t = max vs of the tile / 256, so P'' <= 256) rounded to e4m3, O_acc += P'' V8 with the
// running multiplier mu (O = mu * O_acc; per tile O_acc *= alpha * mu / c_t, mu = c_t), out = mu * O_acc / l * sigmoid(gate).
#include "strix/cuda_core/attention.hpp"

#include "../common.cuh"

namespace strix::cuda_core {
namespace {

constexpr int QB = 128, KT = 64, D = 256, STAGES = 3;
constexpr int K_TILE = KT * D;                       // 16 KiB: [64 keys][256 B], 32-B units swizzled by key & 3
constexpr int V_TILE = D * KT;                       // 16 KiB: [256 dims][64 B], 32-B units swizzled by (dim >> 1) & 1
constexpr int S_TILE = 2 * KT * 4;                   // ks[64], vs[64]
constexpr int STAGE = K_TILE + V_TILE + S_TILE;      // 33,280 B
constexpr int SMEM = STAGES * STAGE;                 // 99,840 B (<= 99 KiB opt-in)

__device__ __forceinline__ void cp16(std::uint32_t smem_addr, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr), "l"(g));
}
__device__ __forceinline__ void mma_e4m3(float (&c)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// 4 floats (lowest k first) -> 4 e4m3 bytes, round to nearest even, saturating
__device__ __forceinline__ std::uint32_t pack4_e4m3(float e0, float e1, float e2, float e3) {
    std::uint16_t lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(e1), "f"(e0));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(e3), "f"(e2));
    return static_cast<std::uint32_t>(lo) | (static_cast<std::uint32_t>(hi) << 16);
}

__device__ __forceinline__ std::uint32_t k_chunk(std::uint32_t r, std::uint32_t ch) {  // ch 0..15 of a 256-B key row
    return r * 256 + ((((ch >> 1) ^ (r & 3)) << 5) | ((ch & 1) << 4));
}
__device__ __forceinline__ std::uint32_t k_read(std::uint32_t r, int ks, int q4) { return r * 256 + ((ks ^ (r & 3)) << 5) + 8 * q4; }
__device__ __forceinline__ std::uint32_t v_chunk(std::uint32_t dim, std::uint32_t ch) {  // ch 0..3 of a 64-B dim row
    return K_TILE + dim * 64 + ((((ch >> 1) ^ ((dim >> 1) & 1)) << 5) | ((ch & 1) << 4));
}
__device__ __forceinline__ std::uint32_t v_read(std::uint32_t dim, int kk, int q4) {
    return K_TILE + dim * 64 + ((kk ^ ((dim >> 1) & 1)) << 5) + 8 * q4;
}

struct Shadow {
    const std::uint8_t* k8;
    const float* ks;
    const std::uint8_t* v8t;
    const float* vs;
    std::uint32_t p8;
};

__device__ __forceinline__ void load_tile(std::uint32_t st, const Shadow& sh, std::uint32_t kvh, std::uint32_t key0, std::uint32_t nkeys) {
    const std::uint8_t* kb = sh.k8 + static_cast<std::size_t>(kvh) * sh.p8 * D;
    const std::uint8_t* vb = sh.v8t + static_cast<std::size_t>(kvh) * D * sh.p8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const std::uint32_t idx = threadIdx.x + 256 * i;
        const std::uint32_t r = idx >> 4, ch = idx & 15;  // K: 64 rows x 16 chunks
        const std::uint32_t key = min(key0 + r, nkeys - 1);
        cp16(st + k_chunk(r, ch), kb + static_cast<std::size_t>(key) * D + ch * 16);
        const std::uint32_t dim = idx >> 2, vc = idx & 3;  // V: 256 dims x 4 chunks (keys key0 .. key0 + 63, inside the padded capacity)
        cp16(st + v_chunk(dim, vc), vb + static_cast<std::size_t>(dim) * sh.p8 + key0 + vc * 16);
    }
    if (threadIdx.x < 32) {
        const std::uint32_t c = threadIdx.x & 15;
        const float* src = (threadIdx.x < 16 ? sh.ks : sh.vs) + static_cast<std::size_t>(kvh) * sh.p8 + key0 + c * 4;
        cp16(st + K_TILE + V_TILE + (threadIdx.x < 16 ? 0 : KT * 4) + c * 16, src);
    }
}

__global__ void __launch_bounds__(256, 1)
    flash8_kernel(AttentionDims d, const float* q, GroupedView qv, const float* gate, GroupedView gv, Shadow sh, std::uint32_t pos0,
                  std::uint32_t rows, float* out) {
    extern __shared__ __align__(16) unsigned char smem[];
    const std::uint32_t qb = gridDim.x - 1 - blockIdx.x;  // heavy (late) blocks first
    const std::uint32_t head = blockIdx.y, G = d.heads / d.kv_heads, kvh = head / G;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q4 = lane & 3;
    const std::uint32_t row_base = qb * QB, wrow0 = row_base + warp * 16;
    const std::uint32_t nkeys = pos0 + min(rows, row_base + QB);
    const std::uint32_t ntiles = (nkeys + KT - 1) / KT;
    const std::uint32_t smem_base = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));

#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < static_cast<int>(ntiles)) load_tile(smem_base + st * STAGE, sh, kvh, st * KT, nkeys);
        asm volatile("cp.async.commit_group;");
    }

    // Q8 fragments: logical k 4 q4 + i <-> dim 32 ks + 8 q4 + i, logical 16 + 4 q4 + i <-> dim 32 ks + 8 q4 + 4 + i
    std::uint32_t qa[8][4];
    float sq0, sq1;
    {
        const std::uint32_t r0 = wrow0 + g, r1 = wrow0 + g + 8;
        const float* p0 = q + static_cast<std::size_t>(min(r0, rows - 1)) * qv.rs() + static_cast<std::size_t>(head) * qv.gs() + 8 * q4;
        const float* p1 = q + static_cast<std::size_t>(min(r1, rows - 1)) * qv.rs() + static_cast<std::size_t>(head) * qv.gs() + 8 * q4;
        float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const float4 x = *reinterpret_cast<const float4*>(p0 + 32 * ks), y = *reinterpret_cast<const float4*>(p0 + 32 * ks + 4);
            const float4 u = *reinterpret_cast<const float4*>(p1 + 32 * ks), w = *reinterpret_cast<const float4*>(p1 + 32 * ks + 4);
            a0 = fmaxf(a0, fmaxf(fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w))),
                                 fmaxf(fmaxf(fabsf(y.x), fabsf(y.y)), fmaxf(fabsf(y.z), fabsf(y.w)))));
            a1 = fmaxf(a1, fmaxf(fmaxf(fmaxf(fabsf(u.x), fabsf(u.y)), fmaxf(fabsf(u.z), fabsf(u.w))),
                                 fmaxf(fmaxf(fabsf(w.x), fabsf(w.y)), fmaxf(fabsf(w.z), fabsf(w.w)))));
        }
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, 1));
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, 2));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, 1));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, 2));
        sq0 = fmaxf(a0, 1e-30f) * (1.0f / 448.0f);
        sq1 = fmaxf(a1, 1e-30f) * (1.0f / 448.0f);
        const float i0 = 1.0f / sq0, i1 = 1.0f / sq1;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const float4 x = *reinterpret_cast<const float4*>(p0 + 32 * ks), y = *reinterpret_cast<const float4*>(p0 + 32 * ks + 4);
            const float4 u = *reinterpret_cast<const float4*>(p1 + 32 * ks), w = *reinterpret_cast<const float4*>(p1 + 32 * ks + 4);
            qa[ks][0] = pack4_e4m3(x.x * i0, x.y * i0, x.z * i0, x.w * i0);
            qa[ks][1] = pack4_e4m3(u.x * i1, u.y * i1, u.z * i1, u.w * i1);
            qa[ks][2] = pack4_e4m3(y.x * i0, y.y * i0, y.z * i0, y.w * i0);
            qa[ks][3] = pack4_e4m3(w.x * i1, w.y * i1, w.z * i1, w.w * i1);
        }
    }
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f, mu = 1.0f;
    const float sl2 = d.scale * 1.4426950408889634f, r0s = sl2 * sq0, r1s = sl2 * sq1;
    const std::uint32_t qpos0 = pos0 + wrow0 + g, qpos1 = qpos0 + 8;

    std::uint32_t stage_idx = 0;
    for (std::uint32_t t = 0; t < ntiles; ++t) {
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2) : "memory");
        __syncthreads();
        {
            const std::uint32_t t_load = t + STAGES - 1;
            if (t_load < ntiles) {
                const std::uint32_t ls = stage_idx == 0 ? STAGES - 1 : stage_idx - 1;
                load_tile(smem_base + ls * STAGE, sh, kvh, t_load * KT, nkeys);
            }
            asm volatile("cp.async.commit_group;");
        }
        const std::uint32_t stage = stage_idx * STAGE;
        stage_idx = stage_idx + 1 == STAGES ? 0 : stage_idx + 1;
        const unsigned char* sb = smem + stage;
        const float* sks = reinterpret_cast<const float*>(sb + K_TILE + V_TILE);
        const float* svs = sks + KT;

        float s[8][4];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const uint2 b = *reinterpret_cast<const uint2*>(sb + k_read(nt * 8 + g, ks, q4));
                mma_e4m3(s[nt], qa[ks], b.x, b.y);
            }
        }
        const std::uint32_t key0 = t * KT;
        const bool need_mask = key0 + KT - 1 > pos0 + wrow0;
        float vmax = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const float2 kf = *reinterpret_cast<const float2*>(sks + nt * 8 + 2 * q4);
            const float2 vv = *reinterpret_cast<const float2*>(svs + nt * 8 + 2 * q4);
            vmax = fmaxf(vmax, fmaxf(vv.x, vv.y));
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float v = s[nt][j] * ((j < 2 ? r0s : r1s) * ((j & 1) ? kf.y : kf.x));
                if (need_mask) {
                    const std::uint32_t kp = key0 + nt * 8 + 2 * q4 + (j & 1), qp = j < 2 ? qpos0 : qpos1;
                    if (kp > qp) v = -INFINITY;
                }
                s[nt][j] = v;
            }
        }
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, 1));
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, 2));
        const float ct = fmaxf(vmax, 1e-30f) * (1.0f / 256.0f), ict = 1.0f / ct;
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float n0s = n0 == -INFINITY ? 0.0f : n0, n1s = n1 == -INFINITY ? 0.0f : n1;
        const float al0 = exp2f(m0 - n0s), al1 = exp2f(m1 - n1s);
        m0 = n0;
        m1 = n1;
        l0 *= al0;
        l1 *= al1;
        const float f0 = al0 * mu * ict, f1 = al1 * mu * ict;
        mu = ct;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            o[i][0] *= f0; o[i][1] *= f0; o[i][2] *= f1; o[i][3] *= f1;
        }
        // P'' fragments: k step kk covers keys 32 kk .. 32 kk + 31 = n-tiles 4 kk .. 4 kk + 3 (see the V8T key order above)
        std::uint32_t pa[2][4];
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            float p[4][4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int nt = 4 * kk + u;
                p[u][0] = exp2f(s[nt][0] - n0s);
                p[u][1] = exp2f(s[nt][1] - n0s);
                p[u][2] = exp2f(s[nt][2] - n1s);
                p[u][3] = exp2f(s[nt][3] - n1s);
                l0 += p[u][0] + p[u][1];
                l1 += p[u][2] + p[u][3];
                const float2 vv = *reinterpret_cast<const float2*>(svs + nt * 8 + 2 * q4);  // re-read: keeps registers for O
                const float w0 = vv.x * ict, w1 = vv.y * ict;
                p[u][0] *= w0;
                p[u][1] *= w1;
                p[u][2] *= w0;
                p[u][3] *= w1;
            }
            pa[kk][0] = pack4_e4m3(p[0][0], p[0][1], p[1][0], p[1][1]);
            pa[kk][1] = pack4_e4m3(p[0][2], p[0][3], p[1][2], p[1][3]);
            pa[kk][2] = pack4_e4m3(p[2][0], p[2][1], p[3][0], p[3][1]);
            pa[kk][3] = pack4_e4m3(p[2][2], p[2][3], p[3][2], p[3][3]);
        }
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
            for (int np = 0; np < 32; ++np) {
                const uint2 b = *reinterpret_cast<const uint2*>(sb + v_read(np * 8 + g, kk, q4));
                mma_e4m3(o[np], pa[kk], b.x, b.y);
            }
        }
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = mu / l0, i1 = mu / l1;
    const std::uint32_t r0 = wrow0 + g, r1 = r0 + 8;
#pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        const std::uint32_t dd = 8 * nt + 2 * q4;
        for (int h = 0; h < 2; ++h) {
            const std::uint32_t r = h ? r1 : r0;
            if (r >= rows) continue;
            float v0 = o[nt][2 * h] * (h ? i1 : i0), v1 = o[nt][2 * h + 1] * (h ? i1 : i0);
            if (gate != nullptr) {
                const float2 gt = *reinterpret_cast<const float2*>(gate + static_cast<std::size_t>(r) * gv.rs() + static_cast<std::size_t>(head) * gv.gs() + dd);
                v0 *= cc_sigmoid(gt.x);
                v1 *= cc_sigmoid(gt.y);
            }
            *reinterpret_cast<float2*>(out + (static_cast<std::size_t>(r) * d.heads + head) * D + dd) = make_float2(v0, v1);
        }
    }
}

// One warp per (position, kv head): K row -> K8 + ks, V row -> V8T column (mma key order inside 32-key groups) + vs.
__global__ void shadow_kernel(AttentionDims d, const __nv_bfloat16* kc, const __nv_bfloat16* vc, std::uint8_t* k8, float* ks, std::uint8_t* v8t,
                              float* vs, std::uint32_t p8, std::uint32_t pos_begin, std::uint32_t count) {
    const std::uint32_t wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (wid >= count * d.kv_heads) return;
    const std::uint32_t kvh = wid % d.kv_heads, pos = pos_begin + wid / d.kv_heads;
    const std::size_t src = (static_cast<std::size_t>(kvh) * d.max_positions + pos) * D + 8 * lane;
    float kv[8], vv[8];
    const uint4 kr = *reinterpret_cast<const uint4*>(kc + src), vr = *reinterpret_cast<const uint4*>(vc + src);
    const std::uint32_t kw[4] = {kr.x, kr.y, kr.z, kr.w}, vw[4] = {vr.x, vr.y, vr.z, vr.w};
    float ka = 0.0f, va = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        kv[2 * i] = bf16_lo(kw[i]);
        kv[2 * i + 1] = bf16_hi(kw[i]);
        vv[2 * i] = bf16_lo(vw[i]);
        vv[2 * i + 1] = bf16_hi(vw[i]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        ka = fmaxf(ka, fabsf(kv[i]));
        va = fmaxf(va, fabsf(vv[i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ka = fmaxf(ka, __shfl_xor_sync(0xffffffffu, ka, o));
        va = fmaxf(va, __shfl_xor_sync(0xffffffffu, va, o));
    }
    const float ksc = fmaxf(ka, 1e-30f) * (1.0f / 448.0f), vsc = fmaxf(va, 1e-30f) * (1.0f / 448.0f);
    const float ik = 1.0f / ksc, iv = 1.0f / vsc;
    const std::uint32_t lo = pack4_e4m3(kv[0] * ik, kv[1] * ik, kv[2] * ik, kv[3] * ik), hi = pack4_e4m3(kv[4] * ik, kv[5] * ik, kv[6] * ik, kv[7] * ik);
    *reinterpret_cast<uint2*>(k8 + (static_cast<std::size_t>(kvh) * p8 + pos) * D + 8 * lane) = make_uint2(lo, hi);
    // V8T byte position of this key inside its 32-key group: j = pos & 31, w = j >> 4, jj = j & 15, b = 8 q + 4 w + i
    const std::uint32_t j = pos & 31u, w = j >> 4, jj = j & 15u, qq = (jj & 7u) >> 1, ii = ((jj >> 3) << 1) | (jj & 1u);
    const std::uint32_t col = (pos & ~31u) + 8 * qq + 4 * w + ii;
    const std::uint32_t vp[2] = {pack4_e4m3(vv[0] * iv, vv[1] * iv, vv[2] * iv, vv[3] * iv), pack4_e4m3(vv[4] * iv, vv[5] * iv, vv[6] * iv, vv[7] * iv)};
#pragma unroll
    for (int i = 0; i < 8; ++i)
        v8t[(static_cast<std::size_t>(kvh) * D + 8 * lane + i) * p8 + col] = static_cast<std::uint8_t>(vp[i >> 2] >> (8 * (i & 3)));
    if (lane == 0) {
        ks[static_cast<std::size_t>(kvh) * p8 + pos] = ksc;
        vs[static_cast<std::size_t>(kvh) * p8 + pos] = vsc;
    }
}

}  // namespace

std::uint32_t attn8_capacity(const AttentionDims& d) { return (d.max_positions + kAttn8Align - 1) / kAttn8Align * kAttn8Align; }

std::size_t kv_shadow_e4m3_bytes(const AttentionDims& d) {
    const std::size_t p8 = attn8_capacity(d);
    return static_cast<std::size_t>(d.kv_heads) * p8 * (2 * D + 2 * sizeof(float));
}

cudaError_t kv_shadow_e4m3(const AttentionDims& d, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, const KvShadowE4m3& sh,
                           std::uint32_t pos_begin, std::uint32_t count, cudaStream_t stream) {
    if (d.head_dim != D || count == 0 || pos_begin + count > d.max_positions) return count == 0 ? cudaSuccess : cudaErrorInvalidValue;
    const std::uint32_t warps = count * d.kv_heads;
    shadow_kernel<<<(warps + 7) / 8, 256, 0, stream>>>(d, k_cache, v_cache, sh.k8, sh.ks, sh.v8t, sh.vs, attn8_capacity(d), pos_begin, count);
    return cudaGetLastError();
}

cudaError_t attention_prefill_flash8(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate, const GroupedView& gv,
                                     const KvShadowE4m3& sh, std::uint32_t pos0, std::uint32_t rows, float* out, cudaStream_t stream) {
    if (d.head_dim != D || rows == 0 || d.kv_heads == 0 || d.heads % d.kv_heads != 0 || d.heads / d.kv_heads > 8 || pos0 + rows > d.max_positions)
        return cudaErrorInvalidValue;
    // Attributes belong to the current device; CUDA serializes the update.
    if (auto e = cudaFuncSetAttribute(flash8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); e != cudaSuccess) return e;
    const Shadow s{sh.k8, sh.ks, sh.v8t, sh.vs, attn8_capacity(d)};
    flash8_kernel<<<dim3((rows + QB - 1) / QB, d.heads), 256, SMEM, stream>>>(d, q, qv, gate, gv, s, pos0, rows, out);
    return cudaGetLastError();
}

}  // namespace strix::cuda_core
