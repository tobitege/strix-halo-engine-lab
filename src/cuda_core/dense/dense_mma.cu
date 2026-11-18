// Dense rows 1..32 on tensor cores: BF16 mma.sync m16n8k16 (FP32 accumulate). The FP32 activation is split exactly into three
// bf16 terms (hi + mid + lo = x, products with bf16 weights exact in FP32), accumulated lo, mid, hi. Fixed order (independent of
// the row count: rows ride the N dimension, every output element is its own accumulator): the K axis is cut into blocks of 32;
// warp w (of 8) owns groups of U consecutive blocks (U = 4 bf16, 8 q8_0): blocks U(8i+w) .. U(8i+w)+U-1 for i = 0, 1, ... (blocks ascending); inside a block two k16 steps, each with its terms; the 8 warp partials are summed in
// ascending warp order. q8_0: int8 values are exact in bf16; each 32-block partial starts from zero and is folded into the
// running sum as tot = tot + fl(d * part), blocks ascending per warp. Thread-level k permutation: thread q of a quad owns 8
// consecutive k of the 32-block (same permutation for A and B, so the sum is unchanged).
#include "dense_mma.hpp"

#include "../common.cuh"

namespace strix::cuda_core {
namespace {

struct MmaArgs {
    const void* w[kMmaMaxMulti];
    float* y[kMmaMaxMulti];
    std::uint32_t out[kMmaMaxMulti];
    std::uint32_t tile_end[kMmaMaxMulti];
};

__device__ __forceinline__ void mma16816(float* c, const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// low half = lo_val, high half = hi_val (round to nearest even)
__device__ __forceinline__ std::uint32_t pack_bf16(float lo_val, float hi_val) {
    std::uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi_val), "f"(lo_val));
    return r;
}

// x terms of 8 consecutive k for one activation row: t[term][word] (term 0 = hi, 1 = mid, 2 = lo), word w = elements 2w, 2w+1.
template <int TERMS>
__device__ __forceinline__ void split8(const float* xp, bool valid, std::uint32_t (&t)[3][4]) {
    float v[8];
    if (valid) {
        const float4 a = reinterpret_cast<const float4*>(xp)[0], b = reinterpret_cast<const float4*>(xp)[1];
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = 0.0f;
    }
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        t[0][w] = pack_bf16(v[2 * w], v[2 * w + 1]);
        t[1][w] = t[2][w] = 0;
        if constexpr (TERMS > 1) {
            const float r0 = __fsub_rn(v[2 * w], __uint_as_float(t[0][w] << 16));
            const float r1 = __fsub_rn(v[2 * w + 1], __uint_as_float(t[0][w] & 0xFFFF0000u));
            t[1][w] = pack_bf16(r0, r1);
            if constexpr (TERMS > 2)
                t[2][w] = pack_bf16(__fsub_rn(r0, __uint_as_float(t[1][w] << 16)), __fsub_rn(r1, __uint_as_float(t[1][w] & 0xFFFF0000u)));
        }
    }
}

// Raw weight data of one k32 block for the thread's two rows (g and g + 8).
template <WeightFormat F>
struct WLoad {
    uint4 u[2];             // bf16: 8 elements as 4 words
    std::uint32_t q[2][2];  // q8: 8 int8 as two words
    float d[2];             // q8 scales
};

template <WeightFormat F>
__device__ __forceinline__ void wload(WLoad<F>& l, const void* base, std::uint32_t in, std::uint32_t o0, int g, int qd, std::uint32_t kb,
                                      const unsigned char* sbase, int ub) {
    if constexpr (F == WeightFormat::bf16) {
        const uint4* p = reinterpret_cast<const uint4*>(static_cast<const std::uint16_t*>(base) + static_cast<std::size_t>(o0 + g) * in) + kb * 4 + qd;
        l.u[0] = __ldg(p);
        l.u[1] = __ldg(p + in);  // row + 8: 8 * in elements = in uint4
    } else {
        const std::size_t rb = static_cast<std::size_t>(in >> 5) * 34;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const std::uint16_t* p = reinterpret_cast<const std::uint16_t*>(
                sbase != nullptr ? sbase + (g + 8 * h) * 272 + ub * 34 : static_cast<const unsigned char*>(base) + (o0 + g + 8 * h) * rb + kb * 34);
            l.d[h] = __half2float(__ushort_as_half(*p));
            const std::uint16_t* q = p + 1 + 4 * qd;
            const std::uint32_t v0 = q[0], v1 = q[1], v2 = q[2], v3 = q[3];
            l.q[h][0] = v0 | (v1 << 16);
            l.q[h][1] = v2 | (v3 << 16);
        }
    }
}

// the 4 words of the thread's 8 elements of row half h
template <WeightFormat F>
__device__ __forceinline__ void a_words(const WLoad<F>& l, int h, std::uint32_t (&w)[4]) {
    if constexpr (F == WeightFormat::bf16) {
        const uint4 u = l.u[h];
        w[0] = u.x; w[1] = u.y; w[2] = u.z; w[3] = u.w;
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const std::uint32_t v = l.q[h][i >> 1] >> (16 * (i & 1));
            w[i] = pack_bf16(static_cast<float>(static_cast<std::int8_t>(v & 0xFF)), static_cast<float>(static_cast<std::int8_t>((v >> 8) & 0xFF)));
        }
    }
}

template <WeightFormat F, int NT, int TERMS, int U>
__device__ __forceinline__ void mma_dot(const void* base, std::uint32_t in, std::uint32_t o0, const float* x, std::uint32_t rows, int warp,
                                        int g, int qd, float (&tot)[NT][4], uint4* stg, int lane) {
    const std::uint32_t nkb = in >> 5;
    const bool staged = F == WeightFormat::q8_0 && (in & 255) == 0;  // rows 16 B aligned: stage 16 rows x 8 blocks (272 B each) through smem
    for (std::uint32_t kb0 = warp * U; kb0 < nkb; kb0 += 8 * U) {
        const unsigned char* sbase = nullptr;
        if constexpr (F == WeightFormat::q8_0) {
            if (staged) {
                const std::size_t rb = static_cast<std::size_t>(nkb) * 34;
                uint4 v[9];
#pragma unroll
                for (int t = 0; t < 9; ++t) {
                    const int j = lane + 32 * t, r = j / 17, c = j % 17;
                    if (j < 272) v[t] = __ldg(reinterpret_cast<const uint4*>(static_cast<const unsigned char*>(base) + (o0 + r) * rb + kb0 * 34) + c);
                }
                __syncwarp();
#pragma unroll
                for (int t = 0; t < 9; ++t)
                    if (lane + 32 * t < 272) stg[lane + 32 * t] = v[t];
                __syncwarp();
                sbase = reinterpret_cast<const unsigned char*>(stg);
            }
        }
        WLoad<F> l[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const std::uint32_t kb = kb0 + u;
            if (kb < nkb) wload<F>(l[u], base, in, o0, g, qd, kb, sbase, u);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const std::uint32_t kb = kb0 + u;
            if (kb >= nkb) break;
            std::uint32_t aw[2][4];
            a_words<F>(l[u], 0, aw[0]);
            a_words<F>(l[u], 1, aw[1]);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                std::uint32_t t[3][4];
                const std::uint32_t r = nt * 8 + g;
                split8<TERMS>(x + static_cast<std::size_t>(r) * in + kb * 32 + 8 * qd, r < rows, t);
                float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float* c = (F == WeightFormat::bf16) ? tot[nt] : part;
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    const std::uint32_t a[4] = {aw[0][2 * s], aw[1][2 * s], aw[0][2 * s + 1], aw[1][2 * s + 1]};
#pragma unroll
                    for (int tt = TERMS - 1; tt >= 0; --tt) mma16816(c, a, t[tt][2 * s], t[tt][2 * s + 1]);
                }
                if constexpr (F != WeightFormat::bf16) {
                    tot[nt][0] = __fadd_rn(tot[nt][0], __fmul_rn(l[u].d[0], part[0]));
                    tot[nt][1] = __fadd_rn(tot[nt][1], __fmul_rn(l[u].d[0], part[1]));
                    tot[nt][2] = __fadd_rn(tot[nt][2], __fmul_rn(l[u].d[1], part[2]));
                    tot[nt][3] = __fadd_rn(tot[nt][3], __fmul_rn(l[u].d[1], part[3]));
                }
            }
        }
    }
}

__device__ __forceinline__ int warp_id_of(int t) { return t >> 5; }

template <WeightFormat F, int NT, int TERMS, bool FUSED>
__global__ void __launch_bounds__(256) mma_kernel(MmaArgs ma, const void* w1, const float* x, const float* resid, std::uint32_t in,
                                                  std::uint32_t rows) {
    constexpr int U = F == WeightFormat::bf16 ? 4 : 8;
    constexpr std::size_t kRed = (FUSED ? 2 : 1) * 8 * NT * 128 * sizeof(float), kStg = F == WeightFormat::q8_0 ? 8 * 272 * 16 : 0;
    __shared__ __align__(16) unsigned char smem[kRed > kStg ? kRed : kStg];
    float(*red)[8][NT * 128] = reinterpret_cast<float(*)[8][NT * 128]>(smem);
    uint4* stg = reinterpret_cast<uint4*>(smem) + warp_id_of(threadIdx.x) * 272;
    pdl_trigger();
    const std::uint32_t tile = blockIdx.x;
    const void* wsel = ma.w[0];
    float* y = ma.y[0];
    std::uint32_t start = 0, out = ma.out[0];
#pragma unroll
    for (int j = 0; j < kMmaMaxMulti - 1; ++j) {
        if (tile >= ma.tile_end[j]) {
            start = ma.tile_end[j];
            wsel = ma.w[j + 1];
            y = ma.y[j + 1];
            out = ma.out[j + 1];
        }
    }
    const std::uint32_t o0 = (tile - start) * 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, qd = lane & 3;
    float a0[NT][4], a1[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int i = 0; i < 4; ++i) a0[nt][i] = a1[nt][i] = 0.0f;
    pdl_wait();
    mma_dot<F, NT, TERMS, U>(wsel, in, o0, x, rows, warp, g, qd, a0, stg, lane);
    if constexpr (FUSED) { __syncwarp(); mma_dot<F, NT, TERMS, U>(w1, in, o0, x, rows, warp, g, qd, a1, stg, lane); }
    __syncthreads();  // the staging area is reused as the reduction area
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        // element e = nt * 128 + o * 8 + c (o = output row within the tile, c = activation row within the n-tile)
        const int e0 = nt * 128 + g * 8 + 2 * qd, e2 = nt * 128 + (g + 8) * 8 + 2 * qd;
        red[0][warp][e0] = a0[nt][0]; red[0][warp][e0 + 1] = a0[nt][1];
        red[0][warp][e2] = a0[nt][2]; red[0][warp][e2 + 1] = a0[nt][3];
        if constexpr (FUSED) {
            red[1][warp][e0] = a1[nt][0]; red[1][warp][e0 + 1] = a1[nt][1];
            red[1][warp][e2] = a1[nt][2]; red[1][warp][e2 + 1] = a1[nt][3];
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < NT * 128; e += 256) {
        float s = red[0][0][e], u = 0.0f;
        if constexpr (FUSED) u = red[1][0][e];
#pragma unroll
        for (int w = 1; w < 8; ++w) {
            s = __fadd_rn(s, red[0][w][e]);
            if constexpr (FUSED) u = __fadd_rn(u, red[1][w][e]);
        }
        const std::uint32_t r = (e >> 7) * 8 + (e & 7), o = o0 + ((e & 127) >> 3);
        if (r >= rows) continue;
        const std::size_t idx = static_cast<std::size_t>(r) * out + o;
        float v = s;
        if constexpr (FUSED) v = __fmul_rn(cc_silu(s), u);
        else if (resid != nullptr) v = __fadd_rn(resid[idx], s);
        y[idx] = v;
    }
}

template <WeightFormat F, int NT, int TERMS, bool FUSED>
cudaError_t go(const MmaCall& c, const MmaArgs& ma, std::uint32_t tiles, cudaStream_t s) {
    return launch_pdl(mma_kernel<F, NT, TERMS, FUSED>, dim3(tiles), dim3(256), s, ma, c.w1, c.x, c.residual, c.in, c.rows);
}
template <WeightFormat F, int TERMS, bool FUSED>
cudaError_t by_nt(const MmaCall& c, const MmaArgs& ma, std::uint32_t tiles, cudaStream_t s) {
    if (c.rows <= 8) return go<F, 1, TERMS, FUSED>(c, ma, tiles, s);
    if (c.rows <= 16) return go<F, 2, TERMS, FUSED>(c, ma, tiles, s);
    if (c.rows <= 24) return go<F, 3, TERMS, FUSED>(c, ma, tiles, s);
    return go<F, 4, TERMS, FUSED>(c, ma, tiles, s);
}
template <WeightFormat F, bool FUSED>
cudaError_t by_terms(const MmaCall& c, const MmaArgs& ma, std::uint32_t tiles, cudaStream_t s) {
    if (c.terms <= 1) return by_nt<F, 1, FUSED>(c, ma, tiles, s);
    if (c.terms == 2) return by_nt<F, 2, FUSED>(c, ma, tiles, s);
    return by_nt<F, 3, FUSED>(c, ma, tiles, s);
}

}  // namespace

bool dense_mma_supported(const MmaCall& c) {
    if (c.format == WeightFormat::f32 || c.in % 32 != 0 || c.rows == 0 || c.rows > 32) return false;
    for (int j = 0; j < c.n; ++j)
        if (c.out[j] % 16 != 0) return false;
    return true;
}

cudaError_t dense_mma_launch(const MmaCall& c, cudaStream_t s) {
    MmaArgs ma{};
    std::uint32_t tiles = 0;
    for (int j = 0; j < c.n; ++j) {
        ma.w[j] = c.w[j];
        ma.y[j] = c.y[j];
        ma.out[j] = c.out[j];
        tiles += c.out[j] / 16;
        ma.tile_end[j] = tiles;
    }
    for (int j = c.n; j < kMmaMaxMulti; ++j) ma.tile_end[j] = tiles;
    const bool fused = c.w1 != nullptr;
    if (c.format == WeightFormat::bf16)
        return fused ? by_terms<WeightFormat::bf16, true>(c, ma, tiles, s) : by_terms<WeightFormat::bf16, false>(c, ma, tiles, s);
    return fused ? by_terms<WeightFormat::q8_0, true>(c, ma, tiles, s) : by_terms<WeightFormat::q8_0, false>(c, ma, tiles, s);
}

}  // namespace strix::cuda_core
