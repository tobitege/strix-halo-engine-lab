// Prefill attention (FAST class): causal flash attention, head_dim 256, GQA, BF16 KV cache, tensor cores (mma.sync m16n8k16 bf16, FP32
// accumulate). See attention.hpp attention_prefill_flash for the contract (NOT bitwise equal to attention_decode).
//
// CTA = 128 query rows x 1 query head, 8 warps of 16 query rows. Key tiles of 32 keys through a 2-stage cp.async pipeline. Q lives in
// registers as bf16 A fragments (the head-dimension k permutation: k step ks, quad lane q holds d = 16 ks + 4 q .. +3, the same
// permutation on the K side, so a thread reads its 4 K values with one LDS.64). S = Q K^T; online softmax in FP32 (exp2); P (bf16) feeds
// O += P V with V B-fragments from ldmatrix.trans. K rows are padded to 544 B (conflict-free LDS.64), V rows to 528 B (conflict-free ldmatrix).
#include "strix/cuda_core/attention.hpp"

#include "../common.cuh"

namespace strix::cuda_core {
namespace {

constexpr int QB = 128, KT = 32, D = 256;

// Shared-memory layouts of one K/V tile stage (byte offsets inside a stage). Both give conflict-free K LDS.64 and V ldmatrix;
// the arithmetic per row is identical (same tiles, same order), so every layout produces the same bits.
struct PadLayout {  // FlashPipe::pad2: rows padded (K 544 B, V 528 B), 2 stages, prefetch distance 1, two barriers per tile
    static constexpr int STAGES = 2, K_TILE = KT * 544, STAGE = K_TILE + KT * 528;
    __device__ static std::uint32_t k_chunk(std::uint32_t r, std::uint32_t ch) { return r * 544 + ch * 16; }
    __device__ static std::uint32_t k_read(std::uint32_t r, int ks, int q4) { return r * 544 + 32 * ks + 8 * q4; }
    __device__ static std::uint32_t v_chunk(std::uint32_t key, std::uint32_t ch) { return K_TILE + key * 528 + ch * 16; }
};
struct SwzLayout {  // FlashPipe::swz3: dense 512-B rows, XOR swizzle (K 32-B units ^ row&3, V 16-B chunks ^ key&7), 3 stages,
                    // prefetch distance 2, one barrier per tile
    static constexpr int STAGES = 3, K_TILE = KT * 512, STAGE = K_TILE + KT * 512;
    __device__ static std::uint32_t k_chunk(std::uint32_t r, std::uint32_t ch) { return r * 512 + ((((ch >> 1) ^ (r & 3)) << 5) | ((ch & 1) << 4)); }
    __device__ static std::uint32_t k_read(std::uint32_t r, int ks, int q4) { return r * 512 + ((ks ^ (r & 3)) << 5) + 8 * q4; }
    __device__ static std::uint32_t v_chunk(std::uint32_t key, std::uint32_t ch) { return K_TILE + key * 512 + ((ch ^ (key & 7)) << 4); }
};
template <class LY>
constexpr int smem_bytes() { return LY::STAGES * LY::STAGE; }

__device__ __forceinline__ void cp16(std::uint32_t smem_addr, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr), "l"(g));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ std::uint32_t pack2(float lo, float hi) {
    std::uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ void ldsm_x4_trans(std::uint32_t (&r)[4], std::uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

template <class LY>
__device__ __forceinline__ void load_kv_tile(std::uint32_t smem_stage, const __nv_bfloat16* kbase, const __nv_bfloat16* vbase, std::uint32_t key0,
                                             std::uint32_t nkeys_total) {
    // 32 keys x 512 B for K and V: 1024 16-byte chunks each over 256 threads
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const std::uint32_t idx = threadIdx.x + 256 * i, r = idx >> 5, ch = idx & 31;
        const std::uint32_t key = min(key0 + r, nkeys_total - 1);
        cp16(smem_stage + LY::k_chunk(r, ch), kbase + static_cast<std::size_t>(key) * D + ch * 8);
        cp16(smem_stage + LY::v_chunk(r, ch), vbase + static_cast<std::size_t>(key) * D + ch * 8);
    }
}

template <class LY>
__global__ void __launch_bounds__(256, 1)
    flash_kernel(AttentionDims d, const float* q, GroupedView qv, const float* gate, GroupedView gv, const __nv_bfloat16* kc, const __nv_bfloat16* vc,
                 std::uint32_t pos0, std::uint32_t rows, float* out, std::uint32_t nqb, std::uint32_t n_full, std::uint32_t tail_w) {
    extern __shared__ __align__(16) unsigned char smem[];
    // job -> tile (linear order = the former 2D grid: head-major, x fastest). Jobs below n_full are whole tiles (8 warps); the rest
    // split the remaining tiles into 8 / tail_w jobs of tail_w active warps (inactive warps still copy K/V and meet every barrier).
    std::uint32_t tile = blockIdx.x, wlo = 0, wn = 8;
    if (blockIdx.x >= n_full) {
        const std::uint32_t k = blockIdx.x - n_full, per = 8 / tail_w;
        tile = n_full + k / per;
        wlo = (k % per) * tail_w;
        wn = tail_w;
    }
    const std::uint32_t qb = nqb - 1 - tile % nqb;  // heavy (late) blocks first
    const std::uint32_t head = tile / nqb, G = d.heads / d.kv_heads, kvh = head / G;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q4 = lane & 3;
    const bool act = static_cast<std::uint32_t>(warp) - wlo < wn;  // warp-uniform
    const std::uint32_t row_base = qb * QB, wrow0 = row_base + warp * 16;
    const std::uint32_t nkeys = pos0 + min(rows, row_base + QB);  // keys any row of this block may see
    const std::uint32_t ntiles = (nkeys + KT - 1) / KT;
    const __nv_bfloat16* kbase = kc + static_cast<std::size_t>(kvh) * d.max_positions * D;
    const __nv_bfloat16* vbase = vc + static_cast<std::size_t>(kvh) * d.max_positions * D;
    const std::uint32_t smem_base = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));

    // prefetch tiles 0 .. STAGES - 2 (one commit group per tile slot; empty groups past the end keep the count uniform)
#pragma unroll
    for (int st = 0; st < LY::STAGES - 1; ++st) {
        if (st < static_cast<int>(ntiles)) load_kv_tile<LY>(smem_base + st * LY::STAGE, kbase, vbase, st * KT, nkeys);
        asm volatile("cp.async.commit_group;");
    }

    // Q fragments (rows g and g + 8 of the warp's 16)
    std::uint32_t qa[16][4];
    {
        const std::uint32_t r0 = wrow0 + g, r1 = wrow0 + g + 8;
        const float* p0 = q + static_cast<std::size_t>(min(r0, rows - 1)) * qv.rs() + static_cast<std::size_t>(head) * qv.gs() + 4 * q4;
        const float* p1 = q + static_cast<std::size_t>(min(r1, rows - 1)) * qv.rs() + static_cast<std::size_t>(head) * qv.gs() + 4 * q4;
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
            const float4 a = *reinterpret_cast<const float4*>(p0 + 16 * ks), b = *reinterpret_cast<const float4*>(p1 + 16 * ks);
            qa[ks][0] = pack2(a.x, a.y);
            qa[ks][1] = pack2(b.x, b.y);
            qa[ks][2] = pack2(a.z, a.w);
            qa[ks][3] = pack2(b.z, b.w);
        }
    }
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float sl2 = d.scale * 1.4426950408889634f;
    const std::uint32_t qpos0 = pos0 + wrow0 + g, qpos1 = qpos0 + 8;

    std::uint32_t stage_idx = 0;  // t % STAGES
    for (std::uint32_t t = 0; t < ntiles; ++t) {
        const std::uint32_t stage = stage_idx * LY::STAGE;
        if constexpr (LY::STAGES == 2) {
            if (t + 1 < ntiles) {
                load_kv_tile<LY>(smem_base + (stage_idx ^ 1) * LY::STAGE, kbase, vbase, (t + 1) * KT, nkeys);
                asm volatile("cp.async.commit_group;\ncp.async.wait_group 1;" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;" ::: "memory");
            }
            __syncthreads();
        } else {
            // committed groups: tiles 0 .. t + STAGES - 2; tile t has landed once at most STAGES - 2 groups are pending
            asm volatile("cp.async.wait_group %0;" ::"n"(LY::STAGES - 2) : "memory");
            __syncthreads();  // tile t visible to every warp, and every warp is done with tile t - 1, whose stage is refilled now
            const std::uint32_t t_load = t + LY::STAGES - 1;
            if (t_load < ntiles) {
                const std::uint32_t ls = stage_idx == 0 ? LY::STAGES - 1 : stage_idx - 1;  // (t + STAGES - 1) % STAGES
                load_kv_tile<LY>(smem_base + ls * LY::STAGE, kbase, vbase, t_load * KT, nkeys);
            }
            asm volatile("cp.async.commit_group;");
        }
        stage_idx = stage_idx + 1 == static_cast<std::uint32_t>(LY::STAGES) ? 0 : stage_idx + 1;
        if (!act) {  // copy/barrier participant only
            if constexpr (LY::STAGES == 2) __syncthreads();
            continue;
        }
        const unsigned char* sK = smem + stage;
        const std::uint32_t sV = smem_base + stage;
        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const uint2 b = *reinterpret_cast<const uint2*>(sK + LY::k_read(nt * 8 + g, ks, q4));
                mma_bf16(s[nt], qa[ks], b.x, b.y);
            }
        }
        const std::uint32_t key0 = t * KT;
        // scale (log2 domain) and causal mask
        const bool need_mask = key0 + KT - 1 > pos0 + wrow0;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float v = s[nt][j] * sl2;
                if (need_mask) {
                    const std::uint32_t kp = key0 + nt * 8 + 2 * q4 + (j & 1), qp = j < 2 ? qpos0 : qpos1;
                    if (kp > qp) v = -INFINITY;
                }
                s[nt][j] = v;
            }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float n0s = n0 == -INFINITY ? 0.0f : n0, n1s = n1 == -INFINITY ? 0.0f : n1;
        const float al0 = exp2f(m0 - n0s), al1 = exp2f(m1 - n1s);  // m = -inf -> 0
        m0 = n0;
        m1 = n1;
        l0 *= al0;
        l1 *= al1;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            o[i][0] *= al0; o[i][1] *= al0; o[i][2] *= al1; o[i][3] *= al1;
        }
        std::uint32_t pa[2][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float p00 = exp2f(s[nt][0] - n0s), p01 = exp2f(s[nt][1] - n0s), p10 = exp2f(s[nt][2] - n1s), p11 = exp2f(s[nt][3] - n1s);
            l0 += p00 + p01;
            l1 += p10 + p11;
            const std::uint32_t lo = pack2(p00, p01), hi = pack2(p10, p11);
            pa[nt >> 1][(nt & 1) * 2 + 0] = lo;
            pa[nt >> 1][(nt & 1) * 2 + 1] = hi;
        }
        // O += P V : a0 = (row g, keys 2q..) of n-tile 2j, a1 = (row g+8, n-tile 2j), a2 = (row g, n-tile 2j+1), a3 = (g+8, 2j+1)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const std::uint32_t a[4] = {pa[j][0], pa[j][1], pa[j][2], pa[j][3]};
#pragma unroll
            for (int np = 0; np < 16; ++np) {
                std::uint32_t b[4];
                const std::uint32_t key = 16 * j + ((lane >> 3) & 1) * 8 + (lane & 7), dcol = 8 * (2 * np + (lane >> 4));
                ldsm_x4_trans(b, sV + LY::v_chunk(key, dcol / 8));
                mma_bf16(o[2 * np], a, b[0], b[1]);
                mma_bf16(o[2 * np + 1], a, b[2], b[3]);
            }
        }
        if constexpr (LY::STAGES == 2) __syncthreads();
    }
    if (!act) return;  // no barrier below
    // finalize
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = 1.0f / l0, i1 = 1.0f / l1;
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

__global__ void iota_kernel(std::uint32_t* p, std::uint32_t n, std::uint32_t base, std::uint32_t step) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = base + step * i;
}

}  // namespace

cudaError_t attention_prefill_flash(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate, const GroupedView& gv,
                                    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, std::uint32_t pos0, std::uint32_t rows,
                                    float* out, cudaStream_t stream, FlashPipe pipe, FlashTail tail, std::uint32_t tail_warps) {
    if (d.head_dim != D || rows == 0 || d.kv_heads == 0 || d.heads % d.kv_heads != 0 || d.heads / d.kv_heads > 8 || pos0 + rows > d.max_positions)
        return cudaErrorInvalidValue;
    static int sms = 0;
    if (sms == 0 && tail != FlashTail::off) {
        int dev = 0;
        if (auto e = cudaGetDevice(&dev); e != cudaSuccess) return e;
        if (auto e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev); e != cudaSuccess) return e;
    }
    const FlashPlan pl = flash_plan(rows, d.heads, static_cast<std::uint32_t>(sms), tail, tail_warps);
    const std::uint32_t nqb = (rows + QB - 1) / QB;
    const dim3 grid(pl.jobs);
    // Attributes belong to the current device; CUDA serializes the update.
    if (pipe == FlashPipe::swz3) {
        if (auto e = cudaFuncSetAttribute(flash_kernel<SwzLayout>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<SwzLayout>());
            e != cudaSuccess)
            return e;
        flash_kernel<SwzLayout><<<grid, 256, smem_bytes<SwzLayout>(), stream>>>(d, q, qv, gate, gv, k_cache, v_cache, pos0, rows, out, nqb,
                                                                                 pl.full, pl.tail_warps);
    } else {
        if (auto e = cudaFuncSetAttribute(flash_kernel<PadLayout>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<PadLayout>());
            e != cudaSuccess)
            return e;
        flash_kernel<PadLayout><<<grid, 256, smem_bytes<PadLayout>(), stream>>>(d, q, qv, gate, gv, k_cache, v_cache, pos0, rows, out, nqb,
                                                                                 pl.full, pl.tail_warps);
    }
    return cudaGetLastError();
}

FlashPlan flash_plan(std::uint32_t rows, std::uint32_t heads, std::uint32_t sms, FlashTail tail, std::uint32_t tail_warps) {
    FlashPlan p;
    p.tiles = (rows + QB - 1) / QB * heads;
    p.full = p.jobs = p.tiles;
    if (tail == FlashTail::off || sms == 0 || p.tiles <= sms) return p;  // no partial wave behind a full one
    const std::uint32_t r = p.tiles % sms;
    if (r == 0) return p;
    std::uint32_t w = 0;
    if (tail_warps == 1 || tail_warps == 2 || tail_warps == 4) {
        w = tail_warps;
    } else {
        for (std::uint32_t c : {1u, 2u, 4u})
            if (r * (8 / c) <= sms) {
                w = c;
                break;
            }
    }
    if (w == 0) return p;
    p.full = p.tiles - r;
    p.tail_tiles = r;
    p.tail_warps = w;
    p.jobs = p.full + r * (8 / w);
    return p;
}

cudaError_t attention_prefill_flash_grid(const AttentionDims& d, FlashPipe pipe, int device, FlashGrid* out) {
    if (out == nullptr || d.heads == 0) return cudaErrorInvalidValue;
    int sms = 0, per_sm = 0;
    if (auto e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device); e != cudaSuccess) return e;
    cudaError_t e = cudaSuccess;
    if (pipe == FlashPipe::swz3) {
        e = cudaFuncSetAttribute(flash_kernel<SwzLayout>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<SwzLayout>());
        if (e == cudaSuccess) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, flash_kernel<SwzLayout>, 256, smem_bytes<SwzLayout>());
    } else {
        e = cudaFuncSetAttribute(flash_kernel<PadLayout>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<PadLayout>());
        if (e == cudaSuccess) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, flash_kernel<PadLayout>, 256, smem_bytes<PadLayout>());
    }
    if (e != cudaSuccess) return e;
    *out = FlashGrid{static_cast<std::uint32_t>(QB), d.heads, static_cast<std::uint32_t>(sms), static_cast<std::uint32_t>(per_sm)};
    return cudaSuccess;
}

cudaError_t iota_u32(std::uint32_t* p, std::uint32_t n, std::uint32_t base, std::uint32_t step, cudaStream_t stream) {
    iota_kernel<<<(n + 127) / 128, 128, 0, stream>>>(p, n, base, step);
    return cudaGetLastError();
}

}  // namespace strix::cuda_core
