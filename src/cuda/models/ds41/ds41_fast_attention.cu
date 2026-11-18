// Fast prefill attention (cuda_ds41_fast_attention.hpp).
//
// enqueue_fast_kv: the cache entries as FP16 rows of 512 (exact: E4M3 x 2^e and E2M1 x E4M3 fit in FP16, but for
// window scales outside 2^-14 .. 2^15, rounded once), a warp per entry, so the attention loads keys without decoding.
//
// enqueue_fast_attention: a block of 16 warps per (row, 32 heads) runs a tiled online softmax over the row's keys
// (window entries, then the selected compressed entries) in tiles of 32 keys:
//  - the 32 query heads (RoPE, scaled by 512^-0.5 * log2(e) and a power of two per head) become FP16 A fragments held
//    in registers (warp: 16 heads x 128 dims);
//  - tiles of FP16 key rows are copied into swizzled shared memory by cp.async (two buffers: the next tile lands while
//    the current one is computed; keys past the row's count are zero-filled);
//  - S = Q K^T and O += P K by mma.sync m16n8k16 (FP16 inputs, FP32 accumulation), ldmatrix (.trans for K as B of PV);
//  - base-2 online softmax per head (ex2.approx, P rounded to FP16 and summed as rounded), the sink added to the
//    denominator at the end;
//  - epilogue: O / l, inverse RoPE of the last 64 dims, FP32 o and / or the A8 rows of o (quantize_a8 per 32 values).
// Only mma.sync / ldmatrix / cp.async (sm_80+); the shared layout fits the 99 KB opt-in of sm_89 and sm_120.
#include "../../core/ptx.cuh"
#include "strix/cuda/ds41_fast_attention.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <stdexcept>

namespace strix::ds41::cuda::attn {
namespace {

constexpr unsigned kFull = 0xffffffffU;
constexpr unsigned kFH = 32;        // heads per block
constexpr unsigned kFK = 32;        // keys per tile
constexpr unsigned kFWarps = 16;
constexpr unsigned kSD = kFWarps / 4;                 // S: dimension slices (warps: 2 head x 2 key groups each)
constexpr unsigned kSDims = kHeadDim / kSD;
constexpr unsigned kPD = kFWarps / 2;                 // PV: dimension slices (warps: 2 head groups each)
constexpr unsigned kPTiles = kHeadDim / kPD / 8;      // 8-dim tiles per PV warp
constexpr unsigned kHPW = kFH / kFWarps;              // softmax / query heads per warp
constexpr unsigned kFThreads = kFWarps * 32;
constexpr unsigned kChunksPerTile = kFK * kHeadDim / 8;  // 16-byte pieces
constexpr unsigned kSStride = kFK + 4;   // floats per S row (float2 stores)
constexpr unsigned kPStride = kFK + 8;   // halves per P row (80 bytes: conflict-free ldmatrix)
constexpr float kLog2e = 1.4426950408889634F;

struct alignas(16) FastSmem {
    __half k[2][kFK * kHeadDim];   // two tiles of swizzled rows (k[1] stages the queries first)
    float s[kSD][kFH][kSStride];   // partial S of the dimension slices
    __half p[kFH * kPStride];
    float m[kFH], l[kFH], alpha[kFH], qscale[kFH];
    std::uint32_t sel[kTopK];      // the row's selected compressed entries
};
constexpr std::size_t kFastSmem = sizeof(FastSmem);
static_assert(kFastSmem <= 99U * 1024U);

// half offset of 16-byte chunk c of row `row` (64 chunks per 512-half row; XOR of the low 3 bits by the row)
__device__ __forceinline__ unsigned swz(unsigned row, unsigned chunk) { return row * kHeadDim + ((chunk ^ (row & 7U)) << 3U); }

using ::strix::cuda::core::ptx::smem_addr;
using ::strix::cuda::core::ptx::ldsm_x4;
__device__ __forceinline__ void ldsm_x4_t(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// 16 bytes global -> shared, zero-filled past src_bytes (0 or 16)
using ::strix::cuda::core::ptx_mem::cp_async16;
using ::strix::cuda::core::ptx_mem::cp_async_commit;
using ::strix::cuda::core::ptx_mem::cp_async_wait_all;
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (unsigned d = 16; d != 0; d /= 2) v = fmaxf(v, __shfl_xor_sync(kFull, v, d));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (unsigned d = 16; d != 0; d /= 2) v += __shfl_xor_sync(kFull, v, d);
    return v;
}
__device__ __forceinline__ unsigned h2u(__half2 h) { return *reinterpret_cast<unsigned*>(&h); }

// ---- FP16 entries ---------------------------------------------------------------------------------------------------
// warp per entry (window entries first, then compressed), lane: dims 16 lane .. +15
__global__ __launch_bounds__(256) void fast_kv_kernel(FastKvView v) {
    const unsigned e = (blockIdx.x * blockDim.x + threadIdx.x) >> 5U, lane = threadIdx.x & 31U;
    unsigned out[8];
    std::uint16_t* dst;
    if (e < v.count) {
        const std::uint8_t* src = v.ring + ((std::size_t{v.first} + e) % v.capacity) * kWindowEntryBytes;
        const uint4 b = __ldg(reinterpret_cast<const uint4*>(src) + lane);
        const int ex = static_cast<int>(__ldg(src + 512U + lane / 2U)) - 127;
        const unsigned wv[4] = {b.x, b.y, b.z, b.w};
        const float sc = pow2f(ex);
#pragma unroll
        for (unsigned k = 0; k < 8; ++k) {
            const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>((wv[k / 2] >> (16U * (k % 2))) & 0xFFFFU), __NV_E4M3);
            const float2 f = __half22float2(__half2(h));
            out[k] = h2u(__floats2half2_rn(f.x * sc, f.y * sc));
        }
        dst = v.win16 + std::size_t{e} * kHeadDim;
    } else if (e < v.count + v.kv_count) {
        const unsigned c = e - v.count;
        const std::uint8_t* src = v.kv_cache + std::size_t{c} * kKvEntryBytes;
        const uint2 b = __ldg(reinterpret_cast<const uint2*>(src) + lane);
        const float sc = e4m3_value(__ldg(src + 256U + lane));
        const unsigned wv[2] = {b.x, b.y};
#pragma unroll
        for (unsigned k = 0; k < 8; ++k) {
            const unsigned byte = (wv[k / 4] >> (8U * (k % 4))) & 255U;
            out[k] = h2u(__floats2half2_rn(e2m1_value(static_cast<std::uint8_t>(byte & 15U)) * sc,
                                           e2m1_value(static_cast<std::uint8_t>(byte >> 4U)) * sc));
        }
        dst = v.kv16 + std::size_t{c} * kHeadDim;
    } else {
        return;
    }
    uint4* d = reinterpret_cast<uint4*>(dst + 16U * lane);
    d[0] = make_uint4(out[0], out[1], out[2], out[3]);
    d[1] = make_uint4(out[4], out[5], out[6], out[7]);
}

// ---- attention ------------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(kFThreads) void fast_attention_kernel(FastAttentionView v) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    FastSmem& sm = *reinterpret_cast<FastSmem*>(smem_raw);
    const unsigned r = blockIdx.y, hb0 = blockIdx.x * kFH, tid = threadIdx.x, lane = tid & 31U, w = tid >> 5U;
    const std::uint64_t p = v.pos[r];
    const unsigned nw = static_cast<unsigned>(p + 1U < kWindow ? p + 1U : kWindow);
    const unsigned nc = v.idx_count != nullptr ? v.idx_count[r] : 0U;
    const unsigned n = nw + nc;
    const RopeRow& rope = v.rope[p];
    const std::uint32_t* idx = v.idx != nullptr ? v.idx + std::size_t{r} * v.topk : nullptr;
    const __half* win16 = reinterpret_cast<const __half*>(v.win16) + (p + 1U - nw - v.win_first) * kHeadDim;
    const __half* kv16 = reinterpret_cast<const __half*>(v.kv16);

    if (tid < kFH) { sm.m[tid] = -__builtin_inff(); sm.l[tid] = 0.0F; }
    for (unsigned i = tid; i < nc; i += kFThreads) sm.sel[i] = __ldg(idx + i);
    // queries: warp w heads kHPW w .., lane dims 16 lane .. +15, staged in k[1]
#pragma unroll
    for (unsigned hh = 0; hh < kHPW; ++hh) {
        const unsigned h = kHPW * w + hh;
        const float4* src = reinterpret_cast<const float4*>(v.q + (std::size_t{r} * v.heads + hb0 + h) * kHeadDim + 16U * lane);
        float x[16];
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const float4 f = __ldg(src + i);
            x[4 * i] = f.x; x[4 * i + 1] = f.y; x[4 * i + 2] = f.z; x[4 * i + 3] = f.w;
        }
        if (lane >= 28U) {
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                const unsigned pair = 8U * (lane - 28U) + i;
                rope_pair(x[2 * i], x[2 * i + 1], rope.c[pair], rope.s[pair], false);
            }
        }
        float amax = 0.0F;
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) { x[i] = x[i] * (kAttnScale * kLog2e); amax = fmaxf(amax, fabsf(x[i])); }
        amax = warp_max(amax);
        // a power of two per head: max |q| in (128, 256] (FP16 range and precision)
        const int e = amax > 0.0F && amax < 3.0e38F ? min(max(log2_ceil(amax) - 8, -100), 100) : 0;
        const float down = pow2f(-e);
        unsigned packed[8];
#pragma unroll
        for (unsigned i = 0; i < 8; ++i) packed[i] = h2u(__floats2half2_rn(x[2 * i] * down, x[2 * i + 1] * down));
        *reinterpret_cast<uint4*>(&sm.k[1][swz(h, 2U * lane)]) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
        *reinterpret_cast<uint4*>(&sm.k[1][swz(h, 2U * lane + 1U)]) = make_uint4(packed[4], packed[5], packed[6], packed[7]);
        if (lane == 0U) sm.qscale[h] = pow2f(e);
    }
    // tile t0 into buffer kt: piece c = tid + kFThreads i -> key c / 64, chunk c % 64 (= tid % 64)
    const unsigned ch = tid & 63U;
    const auto issue = [&](unsigned t0, __half* kt) {
#pragma unroll
        for (unsigned i = 0; i < kChunksPerTile / kFThreads; ++i) {
            const unsigned j = (tid >> 6U) + (kFThreads / 64U) * i, key = t0 + j;
            const bool win = key < nw, any = key < n;
            const __half* src = win ? win16 + std::size_t{key} * kHeadDim : any ? kv16 + std::size_t{sm.sel[key - nw]} * kHeadDim : win16;
            cp_async16(&kt[swz(j, ch)], src + 8U * ch, any ? 16U : 0U);
        }
        cp_async_commit();
    };

    // warp roles. S: heads 16 sh, keys 16 sk, dims kSDims sd. PV: heads 16 ph, dims 8 kPTiles pd.
    const unsigned sh = w & 1U, sk = (w >> 1U) & 1U, sd = w >> 2U;
    const unsigned ph = w & 1U, pd = w >> 1U;
    float acc[kPTiles][4];
#pragma unroll
    for (unsigned t = 0; t < kPTiles; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0F;

    // the queries' A fragments of the S role (16 heads x kSDims dims) into registers
    __syncthreads();
    unsigned qa[kSDims / 16][4];
    {
        const unsigned arow = 16U * sh + (lane & 15U);
#pragma unroll
        for (unsigned ks = 0; ks < kSDims / 16U; ++ks) ldsm_x4(qa[ks], &sm.k[1][swz(arow, (kSDims * sd + 16U * ks) / 8U + (lane >> 4U))]);
    }
    issue(0U, sm.k[0]);
    // per tile (buffer b): tile landed and the previous PV done (barrier) | the next tile into k[b ^ 1], S from k[b] |
    // softmax | O += P k[b]
    unsigned b = 0;
    for (unsigned t0 = 0; t0 < n; t0 += kFK, b ^= 1U) {
        const unsigned cnt = min(kFK, n - t0);
        cp_async_wait_all();
        __syncthreads();
        if (t0 + kFK < n) issue(t0 + kFK, sm.k[b ^ 1U]);
        const __half* kt = sm.k[b];
        // S partial: 16 heads x 16 keys over kSDims dims
        {
            // two accumulators per key tile (even / odd k steps): four independent mma chains
            float s[2][2][4] = {};
            const unsigned brow = 16U * sk + (lane & 7U) + ((lane >> 4U) << 3U);
#pragma unroll
            for (unsigned ks = 0; ks < kSDims / 16U; ++ks) {
                unsigned bb[4];
                ldsm_x4(bb, &kt[swz(brow, (kSDims * sd + 16U * ks) / 8U + ((lane >> 3U) & 1U))]);
                mma16816(s[ks & 1U][0], qa[ks], bb[0], bb[1]);
                mma16816(s[ks & 1U][1], qa[ks], bb[2], bb[3]);
            }
#pragma unroll
            for (unsigned nt = 0; nt < 2; ++nt)
#pragma unroll
                for (unsigned i = 0; i < 4; ++i) s[0][nt][i] += s[1][nt][i];
            const unsigned g = lane >> 2U, cq = 2U * (lane & 3U);
#pragma unroll
            for (unsigned nt = 0; nt < 2; ++nt) {
                const unsigned key = 16U * sk + 8U * nt + cq;
                *reinterpret_cast<float2*>(&sm.s[sd][16U * sh + g][key]) = make_float2(s[0][nt][0], s[0][nt][1]);
                *reinterpret_cast<float2*>(&sm.s[sd][16U * sh + g + 8U][key]) = make_float2(s[0][nt][2], s[0][nt][3]);
            }
        }
        __syncthreads();
        // online softmax: warp w heads kHPW w .., lane = key
#pragma unroll
        for (unsigned hh = 0; hh < kHPW; ++hh) {
            const unsigned h = kHPW * w + hh;
            const bool valid = lane < cnt;
            float s = sm.s[0][h][lane];
#pragma unroll
            for (unsigned z = 1; z < kSD; ++z) s += sm.s[z][h][lane];
            s = valid ? s * sm.qscale[h] : -__builtin_inff();
            const float mo = sm.m[h];
            const float mn = fmaxf(mo, warp_max(s));
            const float a = ex2(mo - mn);
            const __half ph16 = __float2half_rn(valid ? ex2(s - mn) : 0.0F);
            sm.p[h * kPStride + lane] = ph16;
            const float sum = warp_sum(__half2float(ph16));
            __syncwarp();
            if (lane == 0U) { sm.l[h] = sm.l[h] * a + sum; sm.m[h] = mn; sm.alpha[h] = a; }
        }
        __syncthreads();
        // O = O * alpha + P K: 16 heads x 8 kPTiles dims
        {
            const unsigned g = lane >> 2U;
            const float a0 = sm.alpha[16U * ph + g], a1 = sm.alpha[16U * ph + g + 8U];
#pragma unroll
            for (unsigned t = 0; t < kPTiles; ++t) { acc[t][0] *= a0; acc[t][1] *= a0; acc[t][2] *= a1; acc[t][3] *= a1; }
#pragma unroll
            for (unsigned kk = 0; kk < 2; ++kk) {
                unsigned a[4];
                ldsm_x4(a, &sm.p[(16U * ph + (lane & 15U)) * kPStride + 16U * kk + (lane >> 4U) * 8U]);
                const unsigned krow = 16U * kk + (lane & 7U) + ((lane >> 3U) & 1U) * 8U;
#pragma unroll
                for (unsigned np = 0; np < kPTiles / 2; ++np) {
                    unsigned bb[4];
                    ldsm_x4_t(bb, &kt[swz(krow, kPTiles * pd + 2U * np + (lane >> 4U))]);
                    mma16816(acc[2 * np], a, bb[0], bb[1]);
                    mma16816(acc[2 * np + 1], a, bb[2], bb[3]);
                }
            }
        }
    }
    __syncthreads();
    // epilogue: rows g, g + 8 of the warp's 16 heads, dims 8 kPTiles pd + 8 t + 2 (lane & 3) + {0, 1}
    const unsigned g = lane >> 2U, cq = 2U * (lane & 3U);
#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        const unsigned h = 16U * ph + g + 8U * half;
        const float inv = 1.0F / (sm.l[h] + ex2(v.sink[hb0 + h] * kLog2e - sm.m[h]));
        float o[kPTiles][2];
#pragma unroll
        for (unsigned t = 0; t < kPTiles; ++t) {
            o[t][0] = acc[t][2 * half] * inv;
            o[t][1] = acc[t][2 * half + 1] * inv;
            const unsigned d = 8U * kPTiles * pd + 8U * t + cq;
            if (d >= kHeadDim - kRopeDim) {
                const unsigned pair = (d - (kHeadDim - kRopeDim)) / 2U;
                rope_pair(o[t][0], o[t][1], rope.c[pair], rope.s[pair], true);
            }
        }
        const std::size_t col0 = std::size_t{hb0 + h} * kHeadDim + 8U * kPTiles * pd;
        if (v.o != nullptr) {
            float* dst = v.o + std::size_t{r} * v.heads * kHeadDim + col0;
#pragma unroll
            for (unsigned t = 0; t < kPTiles; ++t) *reinterpret_cast<float2*>(dst + 8U * t + cq) = make_float2(o[t][0], o[t][1]);
        }
        if (v.a8.codes != nullptr) {
            // quantize_a8 per group of 32 values (4 tiles of 8 dims; the quad holds the group)
#pragma unroll
            for (unsigned gi = 0; gi < kPTiles / 4; ++gi) {
                float amax = 0.0F;
#pragma unroll
                for (unsigned t = 4 * gi; t < 4 * gi + 4; ++t)
#pragma unroll
                    for (unsigned c = 0; c < 2; ++c) {
                        const float a = fabsf(o[t][c]);
                        amax = a > amax ? a : amax;  // NaN -> ignored
                    }
#pragma unroll
                for (unsigned d = 1; d <= 2; d *= 2) {
                    const float x = __shfl_xor_sync(kFull, amax, d);
                    amax = x > amax ? x : amax;
                }
                const bool tiny = amax > 0.0F && amax < 0x1p-119F;
                const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
                int sum = 0;
                std::int8_t* codes = v.a8.codes + std::size_t{r} * v.a8.stride + col0 + 32U * gi;
#pragma unroll
                for (unsigned t = 4 * gi; t < 4 * gi + 4; ++t) {
                    int cc[2];
#pragma unroll
                    for (unsigned c = 0; c < 2; ++c) {
                        const float rounded = tiny ? 0.0F : rintf(o[t][c] / delta);
                        const float lo = -127.0F < rounded ? rounded : -127.0F;
                        cc[c] = static_cast<int>(lo < 127.0F ? lo : 127.0F);
                        sum += cc[c];
                    }
                    *reinterpret_cast<char2*>(codes + 8U * (t - 4 * gi) + cq) = make_char2(static_cast<char>(cc[0]), static_cast<char>(cc[1]));
                }
                sum += __shfl_xor_sync(kFull, sum, 1);
                sum += __shfl_xor_sync(kFull, sum, 2);
                if ((lane & 3U) == 0U) {
                    const std::size_t gidx = (std::size_t{r} * v.a8.stride + col0) / 32U + gi;
                    v.a8.deltas[gidx] = delta;
                    v.a8.sums[gidx] = sum;
                }
            }
        }
    }
}

}  // namespace

void enqueue_fast_kv(const FastKvView& v, cudaStream_t stream) {
    if ((v.count != 0U && (v.ring == nullptr || v.win16 == nullptr || v.capacity == 0U)) ||
        (v.kv_count != 0U && (v.kv_cache == nullptr || v.kv16 == nullptr)))
        throw std::invalid_argument("ds41 cuda fast kv: buffers");
    const std::size_t warps = std::size_t{v.count} + v.kv_count;
    if (warps == 0U) return;
    fast_kv_kernel<<<static_cast<unsigned>((warps + 7U) / 8U), 256, 0, stream>>>(v);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_fast_attention(const FastAttentionView& v, cudaStream_t stream) {
    if (v.rows == 0U || v.heads == 0U || v.heads % kFH != 0U || v.topk > kTopK || v.win16 == nullptr ||
        (v.o == nullptr && v.a8.codes == nullptr) || (v.a8.codes != nullptr && v.a8.stride < v.heads * kHeadDim) ||
        (v.idx != nullptr && v.kv16 == nullptr))
        throw std::invalid_argument("ds41 cuda fast attention geometry");
    static const bool opt = cudaFuncSetAttribute(fast_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 static_cast<int>(kFastSmem)) == cudaSuccess;
    if (!opt) throw std::runtime_error("ds41 cuda fast attention: shared memory opt-in");
    fast_attention_kernel<<<dim3(v.heads / kFH, v.rows), kFThreads, kFastSmem, stream>>>(v);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace strix::ds41::cuda::attn
