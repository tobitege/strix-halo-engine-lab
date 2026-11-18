// Fast prefill index scores (cuda_ds41_fast_attention.hpp): the indexer's E2M1 x E2M1 products on FP8 tensor cores.
//
// Every E2M1 value times a power of two is an E4M3 value as long as the power is within 2^-8 of the largest block
// scale of its vector (smaller blocks round towards zero), so:
//  - index keys (enqueue_fast_index_keys): position t's 128 values as E4M3 relative to its largest block scale
//    2^ck(t), and the factor 2^ck(t);
//  - queries (prep in enqueue_fast_index_scores): RoPE and E2M1 / E8M0 per 32 exactly as the declared indexer, then
//    E4M3 relative to the head's largest block scale 2^c(h), folded into the head weight: w(h) / 64 * 2^c(h);
//  - score(t) = 2^ck(t) * sum_h relu(q'(h) . k'(t)) * w'(h): q' . k' by mma.sync m16n8k32 E4M3 with FP32 accumulation
//    (positions x heads, 128 values = 4 k steps), relu and the weighted head sum in registers.
// A block of 8 warps covers 16 rows (2 per warp, the query fragments held in registers) and a range of positions in
// tiles of 64 (cp.async into two swizzled buffers). FP8 mma needs sm_89 or later.
#include "../../core/ptx.cuh"
#include "strix/cuda/ds41_fast_attention.hpp"

#include <cuda_fp8.h>

#include <stdexcept>

namespace strix::ds41::cuda::attn {
namespace {

constexpr unsigned kFull = 0xffffffffU;
constexpr unsigned kXWarps = 8, kXThreads = kXWarps * 32;
constexpr unsigned kXRowsPerWarp = 2, kXRows = kXWarps * kXRowsPerWarp;  // rows per block
constexpr unsigned kXTile = 64;                                             // positions per tile
constexpr unsigned kXRange = 1024;                                          // positions per block

struct FastQPrep {
    std::uint8_t q[kIdxHeads][kIdxDim];  // E4M3
    float w[kIdxHeads];                  // w / 64 * 2^c
};
static_assert(sizeof(FastQPrep) == kFastIdxPrepBytes);

using ::strix::cuda::core::ptx::smem_addr;
using ::strix::cuda::core::ptx_mem::cp_async16;
using ::strix::cuda::core::ptx_mem::cp_async_commit;
using ::strix::cuda::core::ptx_mem::cp_async_wait_all;
using ::strix::cuda::core::ptx::ldsm_x4;
__device__ __forceinline__ void mma_e4m3(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}
__device__ __forceinline__ std::uint8_t to_e4m3(float x) {
    return static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}
// 2^d for d <= 0 (0 below 2^-20: E4M3 keeps nothing of an E2M1 value there)
__device__ __forceinline__ float rel_pow2(int d) { return d < -20 ? 0.0F : pow2f(d); }
// byte offset of 16-byte chunk c (of 8) of key row `row`
__device__ __forceinline__ unsigned kswz(unsigned row, unsigned chunk) { return row * kIdxDim + ((chunk ^ (row & 7U)) << 4U); }

// ---- keys: warp per position, lane dims 4 lane .. +3 (bytes 2 lane, 2 lane + 1; block lane / 8)
__global__ __launch_bounds__(256) void fast_index_keys_kernel(FastIndexKeysView v) {
    const unsigned t = (blockIdx.x * blockDim.x + threadIdx.x) >> 5U, lane = threadIdx.x & 31U;
    if (t >= v.count) return;
    const std::uint8_t* e = v.idx_cache + std::size_t{t} * kIdxEntryBytes;
    const unsigned codes = __ldg(e + 2U * lane) | (static_cast<unsigned>(__ldg(e + 2U * lane + 1U)) << 8U);
    const unsigned eb = __ldg(e + 64U + lane / 8U);
    unsigned emax = eb;
#pragma unroll
    for (unsigned d = 16; d >= 8; d /= 2) emax = max(emax, __shfl_xor_sync(kFull, emax, d));
    const float rel = rel_pow2(static_cast<int>(eb) - static_cast<int>(emax));
    unsigned out = 0;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) out |= static_cast<unsigned>(to_e4m3(e2m1_value(static_cast<std::uint8_t>((codes >> (4U * i)) & 15U)) * rel)) << (8U * i);
    reinterpret_cast<unsigned*>(v.k8 + std::size_t{t} * kIdxDim)[lane] = out;
    if (lane == 0U) v.kscale[t] = e8m0_value(static_cast<std::uint8_t>(emax));
}

// ---- queries: warp per (row, head) as the declared index prep, then E4M3 relative to the head's largest block
__global__ __launch_bounds__(1024) void fast_index_prep_kernel(FastIndexView v) {
    const unsigned r = blockIdx.x, h = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    FastQPrep* out = reinterpret_cast<FastQPrep*>(v.qprep) + r;
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
#pragma unroll
    for (unsigned d = 4; d != 0; d /= 2) amax = fmaxf(amax, __shfl_xor_sync(kFull, amax, d));
    amax = fmaxf(amax, 7.05296657e-38F);
    const int e = log2_ceil(amax * (1.0F / 6.0F));
    const float s = pow2f(e);
    int c = e;
#pragma unroll
    for (unsigned d = 16; d >= 8; d /= 2) c = max(c, __shfl_xor_sync(kFull, c, d));
    const float rel = rel_pow2(e - c);
    unsigned word = 0;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const float x = e2m1_value(e2m1_encode(clampf(q[i] / s, -6.0F, 6.0F)));
        word |= static_cast<unsigned>(to_e4m3(x * rel)) << (8U * i);
    }
    reinterpret_cast<unsigned*>(out->q[h])[lane] = word;
    if (lane == 0U) out->w[h] = (v.w[std::size_t{r} * kIdxHeads + h] * kIdxWeightScale) * pow2f(c);
}

// ---- scores
__global__ __launch_bounds__(kXThreads) void fast_index_scores_kernel(FastIndexView v) {
    __shared__ __align__(16) std::uint8_t ks[2][kXTile * kIdxDim];
    const unsigned tid = threadIdx.x, lane = tid & 31U, w = tid >> 5U;
    const unsigned r0 = blockIdx.y * kXRows;
    const unsigned last = min(v.rows, r0 + kXRows) - 1U;
    const unsigned cta_count = min(v.max_count, static_cast<unsigned>((v.pos[last] + 1U) / v.ratio));
    const unsigned p0 = blockIdx.x * kXRange;
    if (p0 >= cta_count) return;
    const unsigned p1 = min(cta_count, p0 + kXRange);
    const FastQPrep* prep = reinterpret_cast<const FastQPrep*>(v.qprep);
    const unsigned g = lane >> 2U, tq = lane & 3U;
    // query B fragments (heads 8 nt + g, bytes 32 ks + 4 tq (+16)) and the weights of heads 8 nt + 2 tq + {0, 1}
    unsigned qb[kXRowsPerWarp][4][4][2];
    float wh[kXRowsPerWarp][4][2];
    unsigned rcount[kXRowsPerWarp];
#pragma unroll
    for (unsigned rr = 0; rr < kXRowsPerWarp; ++rr) {
        const unsigned r = min(r0 + kXRowsPerWarp * w + rr, v.rows - 1U);
        rcount[rr] = r0 + kXRowsPerWarp * w + rr < v.rows ? static_cast<unsigned>((v.pos[r] + 1U) / v.ratio) : 0U;
        const FastQPrep& pr = prep[r];
#pragma unroll
        for (unsigned nt = 0; nt < 4; ++nt) {
            const unsigned* qrow = reinterpret_cast<const unsigned*>(pr.q[8U * nt + g]);
#pragma unroll
            for (unsigned k = 0; k < 4; ++k) {
                qb[rr][nt][k][0] = __ldg(qrow + 8U * k + tq);
                qb[rr][nt][k][1] = __ldg(qrow + 8U * k + 4U + tq);
            }
            wh[rr][nt][0] = __ldg(&pr.w[8U * nt + 2U * tq]);
            wh[rr][nt][1] = __ldg(&pr.w[8U * nt + 2U * tq + 1U]);
        }
    }
    const auto issue = [&](unsigned t0, std::uint8_t* dst) {
#pragma unroll
        for (unsigned i = 0; i < kXTile * kIdxDim / 16U / kXThreads; ++i) {
            const unsigned c = tid + kXThreads * i, row = c >> 3U, ch = c & 7U;
            const bool ok = t0 + row < p1;
            cp_async16(dst + kswz(row, ch), v.k8 + std::size_t{ok ? t0 + row : 0U} * kIdxDim + 16U * ch, ok ? 16U : 0U);
        }
        cp_async_commit();
    };
    issue(p0, ks[0]);
    unsigned b = 0;
    for (unsigned t0 = p0; t0 < p1; t0 += kXTile, b ^= 1U) {
        cp_async_wait_all();
        __syncthreads();
        if (t0 + kXTile < p1) issue(t0 + kXTile, ks[b ^ 1U]);
        const std::uint8_t* kt = ks[b];
#pragma unroll 1
        for (unsigned sub = 0; sub < kXTile / 16U; ++sub) {
            const unsigned tb = t0 + 16U * sub;
            if (tb >= p1) break;
            unsigned a[4][4];
#pragma unroll
            for (unsigned k = 0; k < 4; ++k) ldsm_x4(a[k], kt + kswz(16U * sub + (lane & 15U), 2U * k + (lane >> 4U)));
            const unsigned t_lo = tb + g, t_hi = tb + g + 8U;
            const float k_lo = t_lo < p1 ? __ldg(v.kscale + t_lo) : 0.0F, k_hi = t_hi < p1 ? __ldg(v.kscale + t_hi) : 0.0F;
#pragma unroll
            for (unsigned rr = 0; rr < kXRowsPerWarp; ++rr) {
                if (tb >= rcount[rr]) continue;
                float lo = 0.0F, hi = 0.0F;
#pragma unroll
                for (unsigned nt = 0; nt < 4; ++nt) {
                    float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
                    for (unsigned k = 0; k < 4; ++k) mma_e4m3(c, a[k], qb[rr][nt][k][0], qb[rr][nt][k][1]);
                    lo = fmaf(fmaxf(c[0], 0.0F), wh[rr][nt][0], lo);
                    lo = fmaf(fmaxf(c[1], 0.0F), wh[rr][nt][1], lo);
                    hi = fmaf(fmaxf(c[2], 0.0F), wh[rr][nt][0], hi);
                    hi = fmaf(fmaxf(c[3], 0.0F), wh[rr][nt][1], hi);
                }
                lo += __shfl_xor_sync(kFull, lo, 1);
                lo += __shfl_xor_sync(kFull, lo, 2);
                hi += __shfl_xor_sync(kFull, hi, 1);
                hi += __shfl_xor_sync(kFull, hi, 2);
                float* out = v.scores + std::size_t{r0 + kXRowsPerWarp * w + rr} * v.max_count;
                if (tq == 0U) {
                    if (t_lo < rcount[rr]) out[t_lo] = lo * k_lo;
                    if (t_hi < rcount[rr]) out[t_hi] = hi * k_hi;
                }
            }
        }
    }
}

}  // namespace

bool fast_index_supported(int device) {
    int major = 0, minor = 0;
    STRIX_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    STRIX_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    return major * 10 + minor >= 89;
}

void enqueue_fast_index_keys(const FastIndexKeysView& v, cudaStream_t stream) {
    if (v.count == 0U) return;
    if (v.idx_cache == nullptr || v.k8 == nullptr || v.kscale == nullptr) throw std::invalid_argument("ds41 cuda fast index keys: buffers");
    fast_index_keys_kernel<<<(v.count + 7U) / 8U, 256, 0, stream>>>(v);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_fast_index_scores(const FastIndexView& v, cudaStream_t stream) {
    if (v.rows == 0U || v.ratio == 0U || v.qprep == nullptr || v.scores == nullptr) throw std::invalid_argument("ds41 cuda fast index geometry");
    fast_index_prep_kernel<<<v.rows, 1024, 0, stream>>>(v);
    STRIX_CUDA_CHECK(cudaGetLastError());
    if (v.max_count == 0U) return;
    fast_index_scores_kernel<<<dim3((v.max_count + kXRange - 1U) / kXRange, (v.rows + kXRows - 1U) / kXRows), kXThreads, 0, stream>>>(v);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace strix::ds41::cuda::attn
