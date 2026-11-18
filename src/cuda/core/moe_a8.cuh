#pragma once
// One tuned core implementation. The family binding supplies dimensions,
// activation/quantization epilogue and route-storage operations.
#include "../weight/gsq_sources.cuh"
#include "imma_tile.cuh"
#include <cuda_fp16.h>
namespace strix::cuda::core::moe_a8 {
constexpr int kMagic=0x4b400000;
constexpr float kMagicF=12582912.0F;
constexpr unsigned tile_rows=64;
constexpr unsigned kProjThreads=512,kProjOuts=64,kAStride=272;
template <class View> __device__ __forceinline__ auto at(View v, std::uint32_t r) -> decltype(v.base) {
    return reinterpret_cast<decltype(v.base)>(reinterpret_cast<std::uintptr_t>(v.base) + static_cast<std::uint64_t>(r) * v.stride);
}
__device__ __forceinline__ bool finite(float x) { return (__float_as_uint(x) & 0x7f800000U) != 0x7f800000U; }
// NaN stored to memory as the gfx1151 default quiet NaN. Every NaN of this pipeline is generated inside it (inf - inf,
// 0 * inf: inputs and weights are checked finite), and the AMD default NaN is 0x7fc00000; its sign after NaN propagation
// through v_sub/neg modifiers is not modelled (open question), which only matters on the fail-closed path.
__device__ __forceinline__ float canon(float x) { return x != x ? __uint_as_float(0x7fc00000U) : x; }
__device__ __forceinline__ float half_bits(std::uint32_t b) { return __half2float(__ushort_as_half(static_cast<unsigned short>(b))); }

__device__ __forceinline__ void cp16(void* s, const void* g) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(s));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(g));
}
__device__ __forceinline__ void cp8(void* s, const void* g) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(s));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(a), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

// D = A(16x32 s8, row) * B(32x8 s8, col) + magic. Fragments (PTX ISA, m16n8k32 .s8): g = lane/4, q = lane%4;
// a0: (row g, k 4q..4q+3) a1: (g+8, 4q..) a2: (g, 16+4q..) a3: (g+8, 16+4q..); b0: (k 4q.., col g) b1: (k 16+4q.., col g);
// d0,d1: (row g, cols 2q, 2q+1) d2,d3: (row g+8, cols 2q, 2q+1).
__device__ __forceinline__ void mma_magic(const unsigned (&a)[4], unsigned b0, unsigned b1, int (&d)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(kMagic));
}
__device__ __forceinline__ float group_float(int bits) { return __fsub_rn(__int_as_float(bits), kMagicF); }
__device__ __forceinline__ unsigned lds32(const unsigned char* p) { return *reinterpret_cast<const unsigned*>(p); }

// Q4_K / Q5_K scale and min of group 0..7 from the 16-byte header (known_rows.hip:39-43, = ggml get_scale_min_k4).
__device__ __forceinline__ unsigned factor(uint4 header, unsigned group, bool minimum) {
    const unsigned shift = 8U * (group & 3U);
    const unsigned a = (header.y >> shift) & 255U, b = (header.z >> shift) & 255U, c = (header.w >> shift) & 255U;
    return group < 4U ? (minimum ? b : a) & 63U : minimum ? (c >> 4U) | ((b >> 6U) << 4U) : (c & 15U) | ((a >> 6U) << 4U);
}


struct OperandArch {
 using SignedGroup=weight::CudaSignedGroup32;
 __device__ static void copy16(void* s,const void* g) { cp16(s,g); }
 __device__ static void copy8(void* s,const void* g) { cp8(s,g); }
 __device__ static unsigned load32(const unsigned char* p) { return lds32(p); }
 __device__ static float half(unsigned h) { return half_bits(h); }
 __device__ static unsigned factor(uint4 h,unsigned g,bool m) { return moe_a8::factor(h,g,m); }
 __device__ static void mma(const unsigned (&a)[4],unsigned b0,unsigned b1,int (&r)[4]) { mma_magic(a,b0,b1,r); }
 __device__ static float group_float(int bits) { return moe_a8::group_float(bits); }
 __device__ static float block_acc(int dot,float e,float d,float acc) { return imma::block_acc(dot,e,d,acc); }
};

template <class Source> struct ProjLayout {
    static constexpr unsigned WB = Source::project_wb;
    static constexpr unsigned a_bytes = tile_rows * kAStride, w_bytes = 2U * kProjOuts * WB, es_bytes = tile_rows * 64U;
    static constexpr unsigned stage = a_bytes + w_bytes + es_bytes;
    static constexpr unsigned scales = 2U * stage;                               // D[2][8][64], DM[2][8][64], E[8][64], S[8][64]
    static constexpr unsigned rowinfo = scales + (2U * 2U * 8U * 64U + 2U * 8U * 64U) * 4U;
    static constexpr unsigned total = rowinfo + 2U * tile_rows * 4U;
};


template <class Source, class Binding>
__global__ __launch_bounds__(kProjThreads, 1) void project_kernel(const std::uint8_t* gate, const std::uint8_t* up, typename Binding::Rows v,
                                                                   typename Binding::Tables tables) {
    constexpr unsigned hidden=Binding::hidden, ffn=Binding::ffn, projection_nonfinite=Binding::projection_nonfinite;
    using Csr=typename Binding::Csr;
    using L = ProjLayout<Source>;
    const Csr& csr = v.csr;
    if ((csr.summary[0] & ~projection_nonfinite) || blockIdx.x >= *csr.tile_count) return;
    const unsigned begin = csr.tile_begins[blockIdx.x], count = csr.tile_begins[blockIdx.x + 1U] - begin;
    const unsigned expert = csr.tile_experts[blockIdx.x], out0 = blockIdx.y * kProjOuts;
    extern __shared__ __align__(16) unsigned char smem[];
    float* D = reinterpret_cast<float*>(smem + L::scales);  // [m][j][o]
    float* DM = D + 2U * 8U * 64U;
    float* E = DM + 2U * 8U * 64U;                          // [j][row]
    float* S = E + 8U * 64U;
    unsigned* row_of = reinterpret_cast<unsigned*>(smem + L::rowinfo);
    unsigned* slot_of = row_of + tile_rows;
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5;
    if (t < tile_rows) {
        const unsigned a = t < count ? t : 0U;
        row_of[t] = csr.rows[begin + a]; slot_of[t] = csr.slots[begin + a];
    }
    __syncthreads();
    const auto load = [&](unsigned b) {
        unsigned char* st = smem + (b & 1U) * L::stage;
        for (unsigned i = t; i < tile_rows * 16U; i += kProjThreads) {
            const unsigned r = i >> 4, c = i & 15U;
            cp16(st + r * kAStride + 16U * c, at(v.codes, row_of[r]) + 256U * b + 16U * c);
        }
        Source::template stage_project<kProjOuts,kProjThreads>(st+L::a_bytes,gate,up,expert,out0,hidden,ffn,b,t);
        for (unsigned i = t; i < tile_rows * 4U; i += kProjThreads) {
            const unsigned r = i >> 2, c = i & 3U;
            const void* src = c < 2U ? static_cast<const void*>(at(v.deltas, row_of[r]) + 8U * b + 4U * c)
                                     : static_cast<const void*>(at(v.sums, row_of[r]) + 8U * b + 4U * (c - 2U));
            cp16(st + L::a_bytes + L::w_bytes + r * 64U + 16U * c, src);
        }
        cp_commit();
    };
    const unsigned g = lane >> 2, q = lane & 3U, wm = warp & 3U, wn = warp >> 2;
    const bool active = 16U * wm < count;
    float acc[2][2][4][4];
#pragma unroll
    for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
        for (unsigned n = 0; n < 2U; ++n)
#pragma unroll
            for (unsigned e = 0; e < 4U; ++e)
#pragma unroll
                for (unsigned p = 0; p < 4U; ++p) acc[m][n][e][p] = 0.0F;
    constexpr unsigned blocks = hidden / 256U;
    load(0U);
#pragma unroll 1
    for (unsigned b = 0; b < blocks; ++b) {
        cp_wait_all();
        __syncthreads();  // block b landed; every warp finished block b-1 (its stage and the scale arrays are free)
        if (b + 1U < blocks) load(b + 1U);
        const unsigned char* st = smem + (b & 1U) * L::stage;
        const unsigned char* A = st;
        const unsigned char* W = st + L::a_bytes;
        const unsigned char* ES = st + L::a_bytes + L::w_bytes;
        for (unsigned i = t; i < 2U * 8U * 64U; i += kProjThreads) {
            const unsigned m = i >> 9, j = (i >> 6) & 7U, o = i & 63U;
            Source::project_scales(W+(m*kProjOuts+o)*L::WB,j,D[i],DM[i]);
        }
        for (unsigned i = t; i < 8U * 64U; i += kProjThreads) {
            const unsigned j = i >> 6, r = i & 63U;
            E[i] = reinterpret_cast<const float*>(ES + r * 64U)[j];
            S[i] = static_cast<float>(reinterpret_cast<const int*>(ES + r * 64U + 32U)[j]);
        }
        __syncthreads();
        if (!active) continue;
        const unsigned char* A0 = A + (16U * wm + g) * kAStride;
        const unsigned r0 = 16U * wm + g;
#pragma unroll
        for (unsigned p = 0; p < 4U; ++p) {
            unsigned a[2][4];
            float e[2][2], s[2][2];
#pragma unroll
            for (unsigned G = 0; G < 2U; ++G) {
                const unsigned k0 = 64U * p + 32U * G, j = 2U * p + G;
                a[G][0] = lds32(A0 + k0 + 4U * q); a[G][1] = lds32(A0 + 8U * kAStride + k0 + 4U * q);
                a[G][2] = lds32(A0 + k0 + 16U + 4U * q); a[G][3] = lds32(A0 + 8U * kAStride + k0 + 16U + 4U * q);
                e[G][0] = E[j * 64U + r0]; e[G][1] = E[j * 64U + r0 + 8U];
                s[G][0] = S[j * 64U + r0]; s[G][1] = S[j * 64U + r0 + 8U];
            }
#pragma unroll
            for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
                for (unsigned n = 0; n < 2U; ++n) {
                    const unsigned ob = 16U * wn + 8U * n;
                    const unsigned char* wq = W + (m * kProjOuts + ob + g) * L::WB;
                    unsigned words[4]; Source::project_words(wq,p,q,words);
                    int c0[4], c1[4], second0[4]={}, second1[4]={};
                    Source::mma(a[0],words[0],words[1],c0,second0);
                    Source::mma(a[1],words[2],words[3],c1,second1);
                    const unsigned oc = ob + 2U * q;
                    const float2 d0 = *reinterpret_cast<const float2*>(D + (m * 8U + 2U * p) * 64U + oc);
                    const float2 d1 = *reinterpret_cast<const float2*>(D + (m * 8U + 2U * p + 1U) * 64U + oc);
                    const float2 m0 = *reinterpret_cast<const float2*>(DM + (m * 8U + 2U * p) * 64U + oc);
                    const float2 m1 = *reinterpret_cast<const float2*>(DM + (m * 8U + 2U * p + 1U) * 64U + oc);
#pragma unroll
                    for (unsigned el = 0; el < 4U; ++el) {
                        const unsigned hr = el >> 1, col = el & 1U;
                        const float t0=Source::project_value(col?d0.y:d0.x,col?m0.y:m0.x,c0[el],second0[el],s[0][hr]);
                        const float t1=Source::project_value(col?d1.y:d1.x,col?m1.y:m1.x,c1[el],second1[el],s[1][hr]);
                        const float x0 = __fmul_rn(e[0][hr], t0), x1 = __fmul_rn(e[1][hr], t1);
                        const float pair = __fadd_rn(x0, x1);
                        acc[m][n][el][p] = __fadd_rn(acc[m][n][el][p], pair);
                    }
                }
        }
    }
    __syncthreads();  // stages are reused for the g/u tile
    float* Gt = reinterpret_cast<float*>(smem);   // [64 rows][64 outputs]
    float* Ut = Gt + tile_rows * kProjOuts;
    if (active) {
#pragma unroll
        for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
            for (unsigned n = 0; n < 2U; ++n)
#pragma unroll
                for (unsigned el = 0; el < 4U; ++el) {
                    const float* c = acc[m][n][el];
                    const float s01 = __fadd_rn(c[0], c[1]), s23 = __fadd_rn(c[2], c[3]);
                    const unsigned r = 16U * wm + g + 8U * (el >> 1), o = 16U * wn + 8U * n + 2U * q + (el & 1U);
                    (m ? Ut : Gt)[r * kProjOuts + o] = __fadd_rn(s01, s23);
                }
    }
    __syncthreads();
    Binding::template activation_tile<kProjOuts, kProjThreads>(Gt, Ut, v, tables, row_of, slot_of, count, out0);
}

// ---------------------------------------------------------------------------------------------------------------------
// down: down_prefill_wmma<Q51,16> (known_rows.hip:1005-1032) for a tile of <= 64 assignments and 64 hidden outputs.
// 8 warps: warp w covers rows 16*(w%4).. and outputs 32*(w/4).. (4 n8 tiles). Per element four chains, group 4k+j into
// chain j, groups ascending (down_chains :966-987):
//   first = d*float(r); Q5_1: combined = first + m*sum (Q8_0: combined = first); contribution = delta*combined;
//   acc[j] = acc[j] + contribution;   fold (acc0 + acc1) + (acc2 + acc3).
// K = 640 in 5 chunks of 4 groups, cp.async double buffered.
constexpr unsigned kDownThreads = 256U, kDownOuts = 64U, kDAStride = 144U;
template <class Source> struct DownLayout {
    static constexpr unsigned WB = Source::down_wb, WS = Source::down_ws;  // raw bytes / smem stride per output
    static constexpr unsigned a_bytes = tile_rows * kDAStride, w_bytes = kDownOuts * WS, es_bytes = tile_rows * 32U;
    static constexpr unsigned stage = a_bytes + w_bytes + es_bytes;
    static constexpr unsigned scales = 2U * stage;   // D[4][64], M[4][64], E[4][64], S[4][64]
    static constexpr unsigned rowinfo = scales + 4U * 4U * 64U * 4U;
    static constexpr unsigned total = rowinfo + 2U * tile_rows * 4U;
};

template <class Source, class Binding, bool RouteBF16 = false>
__global__ __launch_bounds__(kDownThreads) void down_kernel(const std::uint8_t* weights, typename Binding::Rows v) {
    constexpr unsigned hidden=Binding::hidden, ffn=Binding::ffn, projection_nonfinite=Binding::projection_nonfinite;
    using Csr=typename Binding::Csr;
    using L = DownLayout<Source>;
    const Csr& csr = v.csr;
    if (csr.summary[0] || blockIdx.x >= *csr.tile_count) return;
    const unsigned begin = csr.tile_begins[blockIdx.x], count = csr.tile_begins[blockIdx.x + 1U] - begin;
    const unsigned expert = csr.tile_experts[blockIdx.x], out0 = blockIdx.y * kDownOuts;
    extern __shared__ __align__(16) unsigned char smem[];
    float* Dv = reinterpret_cast<float*>(smem + L::scales);  // [j][o]
    float* Mv = Dv + 4U * 64U;
    float* E = Mv + 4U * 64U;                                // [j][row]
    float* S = E + 4U * 64U;
    unsigned* row_of = reinterpret_cast<unsigned*>(smem + L::rowinfo);
    unsigned* slot_of = row_of + tile_rows;
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5;
    if (t < tile_rows) {
        const unsigned a = t < count ? t : 0U;
        row_of[t] = csr.rows[begin + a]; slot_of[t] = csr.slots[begin + a];
    }
    __syncthreads();
    constexpr unsigned groups = ffn / 32U;
    const auto load = [&](unsigned c) {
        unsigned char* st = smem + (c & 1U) * L::stage;
        for (unsigned i = t; i < tile_rows * 8U; i += kDownThreads) {
            const unsigned r = i >> 3, k = i & 7U;
            cp16(st + r * kDAStride + 16U * k, at(v.down_codes, row_of[r]) + slot_of[r] * ffn + 128U * c + 16U * k);
        }
        Source::template stage_down<kDownOuts,kDownThreads>(st+L::a_bytes,weights,expert,out0,hidden,ffn,c,t);
        for (unsigned i = t; i < tile_rows * 2U; i += kDownThreads) {
            const unsigned r = i >> 1, k = i & 1U;
            const void* src = k == 0U ? static_cast<const void*>(at(v.down_deltas, row_of[r]) + slot_of[r] * groups + 4U * c)
                                      : static_cast<const void*>(at(v.down_sums, row_of[r]) + slot_of[r] * groups + 4U * c);
            cp16(st + L::a_bytes + L::w_bytes + r * 32U + 16U * k, src);
        }
        cp_commit();
    };
    const unsigned g = lane >> 2, q = lane & 3U, wm = warp & 3U, wn = warp >> 2;
    const bool active = 16U * wm < count;
    float acc[4][4][4];
#pragma unroll
    for (unsigned n = 0; n < 4U; ++n)
#pragma unroll
        for (unsigned e = 0; e < 4U; ++e)
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) acc[n][e][j] = 0.0F;
    constexpr unsigned chunks = groups / 4U;
    load(0U);
#pragma unroll 1
    for (unsigned c = 0; c < chunks; ++c) {
        cp_wait_all();
        __syncthreads();
        if (c + 1U < chunks) load(c + 1U);
        const unsigned char* st = smem + (c & 1U) * L::stage;
        const unsigned char* A = st;
        const unsigned char* W = st + L::a_bytes;
        const unsigned char* ES = st + L::a_bytes + L::w_bytes;
        for (unsigned i = t; i < 4U * 64U; i += kDownThreads) {
            const unsigned j = i >> 6, o = i & 63U;
            Source::down_scales(W+o*L::WS,j,Dv[i],Mv[i]);
            E[i] = reinterpret_cast<const float*>(ES + o * 32U)[j];
            S[i] = static_cast<float>(reinterpret_cast<const int*>(ES + o * 32U + 16U)[j]);
        }
        __syncthreads();
        if (!active) continue;
        const unsigned r0 = 16U * wm + g;
        const unsigned char* A0 = A + r0 * kDAStride;
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            unsigned a[4];
            a[0] = lds32(A0 + 32U * j + 4U * q); a[1] = lds32(A0 + 8U * kDAStride + 32U * j + 4U * q);
            a[2] = lds32(A0 + 32U * j + 16U + 4U * q); a[3] = lds32(A0 + 8U * kDAStride + 32U * j + 16U + 4U * q);
            const float e0 = E[j * 64U + r0], e1 = E[j * 64U + r0 + 8U];
            const float s0 = S[j * 64U + r0], s1 = S[j * 64U + r0 + 8U];
#pragma unroll
            for (unsigned n = 0; n < 4U; ++n) {
                const unsigned ob = 32U * wn + 8U * n;
                unsigned b0,b1; Source::down_words(W+(ob+g)*L::WS,j,q,b0,b1);
                int r[4];
                mma_magic(a, b0, b1, r);
                const unsigned oc = ob + 2U * q;
                const float2 dd = *reinterpret_cast<const float2*>(Dv + j * 64U + oc);
                const float2 mm = *reinterpret_cast<const float2*>(Mv + j * 64U + oc);
#pragma unroll
                for (unsigned el = 0; el < 4U; ++el) {
                    const unsigned hr = el >> 1, col = el & 1U;
                    const float combined=Source::down_value(col?dd.y:dd.x,col?mm.y:mm.x,r[el],hr?s1:s0);
                    const float contribution = __fmul_rn(hr ? e1 : e0, combined);
                    acc[n][el][j] = __fadd_rn(acc[n][el][j], contribution);
                }
            }
        }
    }
    if (!active) return;
#pragma unroll
    for (unsigned n = 0; n < 4U; ++n)
#pragma unroll
        for (unsigned hr = 0; hr < 2U; ++hr) {
            const unsigned r = 16U * wm + g + 8U * hr;
            if (r >= count) continue;
            float out[2];
#pragma unroll
            for (unsigned col = 0; col < 2U; ++col) {
                const float* c = acc[n][2U * hr + col];
                out[col] = __fadd_rn(__fadd_rn(c[0], c[1]), __fadd_rn(c[2], c[3]));
            }
            const unsigned o = out0 + 32U * wn + 8U * n + 2U * q;
            if constexpr (RouteBF16) {
                const auto index = static_cast<std::uint64_t>(slot_of[r]) * hidden + o;
                Binding::template route_store<true>(at(v.rawdots, row_of[r]), index, out[0]);
                Binding::template route_store<true>(at(v.rawdots, row_of[r]), index + 1U, out[1]);
            } else {
            *reinterpret_cast<float2*>(at(v.rawdots, row_of[r]) + slot_of[r] * hidden + o) = make_float2(out[0], out[1]);
            }
        }
}

// ---------------------------------------------------------------------------------------------------------------------
// moe_mma FAST form (Config::forms.moe_mma, identity qwen-cuda-moe-i8fma-v1): the same int8 IMMA block dots as the exact
// kernels (activations stay int8, one m16n8k32 IMMA = one exact 32-K group dot) but ONE FP32 chain per output instead of four:
//   project (Q4_K/Q5_K): acc = fma(e*dsc, float(dot), acc); acc = fma(-dmn, e*s, acc)   per group, groups ascending
//   down (Q5_1/Q8_0):    acc = fma(e*d,  float(dot), acc); acc = fma(m, e*s, acc)       (Q8_0: first only)
// with e = activation delta, s = activation sum, dsc = d*sc, dmn = dm*mn. The FMAs (and the pre-multiplied e*s) round differently
// from the HIP chains, so outputs differ by FP32 rounding noise: a bit-changing form, never part of the exact profile. The single
// chain frees registers: a warp owns MT 16-row m-tiles x 16 outputs, so every unpacked weight fragment feeds MT IMMAs and A comes
// from ldmatrix (core::imma::load_a_frag). Tiles stay the CSR's <= 64 assignments (partial tiles: rows >= count replicate row 0,
// warps whose first row is past `count` skip). CTA outputs = 16 * NW.
#ifndef MOE_FP_MT
#define MOE_FP_MT 1
#endif
#ifndef MOE_FP_NW
#define MOE_FP_NW 4
#endif
#ifndef MOE_FD_MT
#define MOE_FD_MT 2
#endif
#ifndef MOE_FD_NW
#define MOE_FD_NW 4
#endif
template <class Source, unsigned MT, unsigned NW> struct FastProjLayout {
    static_assert(MT == 1U || MT == 2U || MT == 4U);
    static constexpr unsigned OUTS = 16U * NW, WR = 4U / MT, THREADS = 32U * WR * NW;
    static constexpr unsigned WB = Source::project_wb;
    static constexpr unsigned a_bytes = tile_rows * kAStride, w_bytes = 2U * OUTS * WB, es_bytes = tile_rows * 64U;
    static constexpr unsigned stage = a_bytes + w_bytes + es_bytes;
    static constexpr unsigned scales = 2U * stage;  // D[2][8][OUTS], DM[2][8][OUTS], E[8][64], ES[8][64]
    static constexpr unsigned rowinfo = scales + (4U * 8U * OUTS + 2U * 8U * 64U) * 4U;
    static constexpr unsigned total = rowinfo + 2U * tile_rows * 4U;
    static_assert(2U * tile_rows * OUTS * 4U <= scales, "g/u tile reuses the stages");
};

template <class Source, class Binding, unsigned MT, unsigned NW>
__global__ __launch_bounds__(FastProjLayout<Source, MT, NW>::THREADS, 1) void project_fast_kernel(
    const std::uint8_t* gate, const std::uint8_t* up, typename Binding::Rows v, typename Binding::Tables tables) {
    constexpr unsigned hidden=Binding::hidden, ffn=Binding::ffn, projection_nonfinite=Binding::projection_nonfinite;
    using Csr=typename Binding::Csr;
    using L = FastProjLayout<Source, MT, NW>;
    constexpr unsigned OUTS = L::OUTS, THREADS = L::THREADS, WR = L::WR;
    const Csr& csr = v.csr;
    if ((csr.summary[0] & ~projection_nonfinite) || blockIdx.x >= *csr.tile_count) return;
    const unsigned begin = csr.tile_begins[blockIdx.x], count = csr.tile_begins[blockIdx.x + 1U] - begin;
    const unsigned expert = csr.tile_experts[blockIdx.x], out0 = blockIdx.y * OUTS;
    extern __shared__ __align__(16) unsigned char smem[];
    float* D = reinterpret_cast<float*>(smem + L::scales);  // [m][j][o]
    float* DM = D + 2U * 8U * OUTS;
    float* E = DM + 2U * 8U * OUTS;                          // [j][row]
    float* ES = E + 8U * 64U;                                // e * s
    unsigned* row_of = reinterpret_cast<unsigned*>(smem + L::rowinfo);
    unsigned* slot_of = row_of + tile_rows;
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5;
    if (t < tile_rows) {
        const unsigned a = t < count ? t : 0U;
        row_of[t] = csr.rows[begin + a]; slot_of[t] = csr.slots[begin + a];
    }
    __syncthreads();
    const auto load = [&](unsigned b) {
        unsigned char* st = smem + (b & 1U) * L::stage;
        for (unsigned i = t; i < tile_rows * 16U; i += THREADS) {
            const unsigned r = i >> 4, c = i & 15U;
            cp16(st + r * kAStride + 16U * c, at(v.codes, row_of[r]) + 256U * b + 16U * c);
        }
        Source::template stage_project<OUTS,THREADS>(st+L::a_bytes,gate,up,expert,out0,hidden,ffn,b,t);
        for (unsigned i = t; i < tile_rows * 4U; i += THREADS) {
            const unsigned r = i >> 2, c = i & 3U;
            const void* src = c < 2U ? static_cast<const void*>(at(v.deltas, row_of[r]) + 8U * b + 4U * c)
                                     : static_cast<const void*>(at(v.sums, row_of[r]) + 8U * b + 4U * (c - 2U));
            cp16(st + L::a_bytes + L::w_bytes + r * 64U + 16U * c, src);
        }
        cp_commit();
    };
    const unsigned g = lane >> 2, q = lane & 3U, wr = warp % WR, wn = warp / WR, rbase = 16U * MT * wr;
    const bool active = rbase < count;
    float acc[MT][2][2][4];
#pragma unroll
    for (unsigned mt = 0; mt < MT; ++mt)
#pragma unroll
        for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
            for (unsigned n = 0; n < 2U; ++n)
#pragma unroll
                for (unsigned e = 0; e < 4U; ++e) acc[mt][m][n][e] = 0.0F;
    constexpr unsigned blocks = hidden / 256U;
    load(0U);
#pragma unroll 1
    for (unsigned b = 0; b < blocks; ++b) {
        cp_wait_all();
        __syncthreads();
        if (b + 1U < blocks) load(b + 1U);
        const unsigned char* st = smem + (b & 1U) * L::stage;
        const unsigned char* A = st;
        const unsigned char* W = st + L::a_bytes;
        const unsigned char* AS = st + L::a_bytes + L::w_bytes;
        for (unsigned i = t; i < 2U * 8U * OUTS; i += THREADS) {
            const unsigned m = i / (8U * OUTS), j = (i / OUTS) & 7U, o = i % OUTS;
            Source::project_scales(W+(m*OUTS+o)*L::WB,j,D[i],DM[i]);
        }
        for (unsigned i = t; i < 8U * 64U; i += THREADS) {
            const unsigned j = i >> 6, r = i & 63U;
            const float ev = reinterpret_cast<const float*>(AS + r * 64U)[j];
            E[i] = ev;
            ES[i] = __fmul_rn(ev, static_cast<float>(reinterpret_cast<const int*>(AS + r * 64U + 32U)[j]));
        }
        __syncthreads();
        if (!active) continue;
#pragma unroll
        for (unsigned p = 0; p < 4U; ++p) {
            typename Source::Packed wv[2][2];
#pragma unroll
            for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
                for (unsigned n = 0; n < 2U; ++n) {
                    const unsigned char* wq = W + (m * OUTS + 16U * wn + 8U * n + g) * L::WB;
                    wv[m][n]=Source::project_load(wq,p,q);
                }
#pragma unroll
            for (unsigned G = 0; G < 2U; ++G) {
                const unsigned j = 2U * p + G;
                std::uint32_t a[MT][4];
                float e[MT][2], es[MT][2];
#pragma unroll
                for (unsigned mt = 0; mt < MT; ++mt) {
                    core::imma::load_a_frag(a[mt], A + (rbase + 16U * mt) * kAStride + 32U * j, kAStride, static_cast<int>(lane));
                    const unsigned r0 = rbase + 16U * mt + g;
                    e[mt][0] = E[j * 64U + r0]; e[mt][1] = E[j * 64U + r0 + 8U];
                    es[mt][0] = ES[j * 64U + r0]; es[mt][1] = ES[j * 64U + r0 + 8U];
                }
#pragma unroll
                for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
                    for (unsigned n = 0; n < 2U; ++n) {
                        std::uint32_t bb[2]; Source::project_unpack(wv[m][n],j,G,bb);
                        const unsigned oc = 16U * wn + 8U * n + 2U * q;
                        const float2 dd = *reinterpret_cast<const float2*>(D + (m * 8U + j) * OUTS + oc);
                        const float2 mm = *reinterpret_cast<const float2*>(DM + (m * 8U + j) * OUTS + oc);
#pragma unroll
                        for (unsigned mt = 0; mt < MT; ++mt) {
                            int c[4];
                            core::imma::mma_seeded(c, a[mt], bb);
#pragma unroll
                            for (unsigned el = 0; el < 4U; ++el) {
                                const unsigned hr = el >> 1, col = el & 1U;
                                acc[mt][m][n][el]=Source::fast_project_acc(c[el],e[mt][hr],col?dd.y:dd.x,col?mm.y:mm.x,es[mt][hr],acc[mt][m][n][el]);
                            }
                        }
                    }
            }
        }
    }
    __syncthreads();
    float* Gt = reinterpret_cast<float*>(smem);  // [64 rows][OUTS]
    float* Ut = Gt + tile_rows * OUTS;
    if (active) {
#pragma unroll
        for (unsigned mt = 0; mt < MT; ++mt)
#pragma unroll
            for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
                for (unsigned n = 0; n < 2U; ++n)
#pragma unroll
                    for (unsigned el = 0; el < 4U; ++el) {
                        const unsigned r = rbase + 16U * mt + g + 8U * (el >> 1), o = 16U * wn + 8U * n + 2U * q + (el & 1U);
                        (m ? Ut : Gt)[r * OUTS + o] = acc[mt][m][n][el];
                    }
    }
    __syncthreads();
    Binding::template activation_tile<OUTS, THREADS>(Gt, Ut, v, tables, row_of, slot_of, count, out0);
}


template <class Source, unsigned MT, unsigned NW> struct FastDownLayout {
    static_assert(MT == 1U || MT == 2U || MT == 4U);
    static constexpr unsigned OUTS = 16U * NW, WR = 4U / MT, THREADS = 32U * WR * NW;
    static constexpr unsigned WB = Source::down_wb, WS = Source::down_ws;
    static constexpr unsigned a_bytes = tile_rows * kDAStride, w_bytes = OUTS * WS, es_bytes = tile_rows * 32U;
    static constexpr unsigned stage = a_bytes + w_bytes + es_bytes;
    static constexpr unsigned scales = 2U * stage;  // Dv[4][OUTS], Mv[4][OUTS], E[4][64], ES[4][64]
    static constexpr unsigned rowinfo = scales + (2U * 4U * OUTS + 2U * 4U * 64U) * 4U;
    static constexpr unsigned total = rowinfo + 2U * tile_rows * 4U;
};

template <class Source, class Binding, bool RouteBF16, unsigned MT, unsigned NW>
__global__ __launch_bounds__(FastDownLayout<Source, MT, NW>::THREADS) void down_fast_kernel(const std::uint8_t* weights, typename Binding::Rows v) {
    constexpr unsigned hidden=Binding::hidden, ffn=Binding::ffn, projection_nonfinite=Binding::projection_nonfinite;
    using Csr=typename Binding::Csr;
    using L = FastDownLayout<Source, MT, NW>;
    constexpr unsigned OUTS = L::OUTS, THREADS = L::THREADS, WR = L::WR;
    const Csr& csr = v.csr;
    if (csr.summary[0] || blockIdx.x >= *csr.tile_count) return;
    const unsigned begin = csr.tile_begins[blockIdx.x], count = csr.tile_begins[blockIdx.x + 1U] - begin;
    const unsigned expert = csr.tile_experts[blockIdx.x], out0 = blockIdx.y * OUTS;
    extern __shared__ __align__(16) unsigned char smem[];
    float* Dv = reinterpret_cast<float*>(smem + L::scales);  // [j][o]
    float* Mv = Dv + 4U * OUTS;
    float* E = Mv + 4U * OUTS;                               // [j][row]
    float* ES = E + 4U * 64U;
    unsigned* row_of = reinterpret_cast<unsigned*>(smem + L::rowinfo);
    unsigned* slot_of = row_of + tile_rows;
    const unsigned t = threadIdx.x, lane = t & 31U;
    if (t < tile_rows) {
        const unsigned a = t < count ? t : 0U;
        row_of[t] = csr.rows[begin + a]; slot_of[t] = csr.slots[begin + a];
    }
    __syncthreads();
    constexpr unsigned groups = ffn / 32U;
    const auto load = [&](unsigned c) {
        unsigned char* st = smem + (c & 1U) * L::stage;
        for (unsigned i = t; i < tile_rows * 8U; i += THREADS) {
            const unsigned r = i >> 3, k = i & 7U;
            cp16(st + r * kDAStride + 16U * k, at(v.down_codes, row_of[r]) + slot_of[r] * ffn + 128U * c + 16U * k);
        }
        Source::template stage_down<OUTS,THREADS>(st+L::a_bytes,weights,expert,out0,hidden,ffn,c,t);
        for (unsigned i = t; i < tile_rows * 2U; i += THREADS) {
            const unsigned r = i >> 1, k = i & 1U;
            const void* src = k == 0U ? static_cast<const void*>(at(v.down_deltas, row_of[r]) + slot_of[r] * groups + 4U * c)
                                      : static_cast<const void*>(at(v.down_sums, row_of[r]) + slot_of[r] * groups + 4U * c);
            cp16(st + L::a_bytes + L::w_bytes + r * 32U + 16U * k, src);
        }
        cp_commit();
    };
    const unsigned warp = t >> 5;
    const unsigned g = lane >> 2, q = lane & 3U, wr = warp % WR, wn = warp / WR, rbase = 16U * MT * wr;
    const bool active = rbase < count;
    float acc[MT][2][4];
#pragma unroll
    for (unsigned mt = 0; mt < MT; ++mt)
#pragma unroll
        for (unsigned n = 0; n < 2U; ++n)
#pragma unroll
            for (unsigned e = 0; e < 4U; ++e) acc[mt][n][e] = 0.0F;
    constexpr unsigned chunks = groups / 4U;
    load(0U);
#pragma unroll 1
    for (unsigned c = 0; c < chunks; ++c) {
        cp_wait_all();
        __syncthreads();
        if (c + 1U < chunks) load(c + 1U);
        const unsigned char* st = smem + (c & 1U) * L::stage;
        const unsigned char* A = st;
        const unsigned char* W = st + L::a_bytes;
        const unsigned char* AS = st + L::a_bytes + L::w_bytes;
        for (unsigned i = t; i < 4U * OUTS; i += THREADS) {
            const unsigned j = i / OUTS, o = i % OUTS;
            Source::down_scales(W+o*L::WS,j,Dv[i],Mv[i]);
        }
        for (unsigned i = t; i < 4U * 64U; i += THREADS) {
            const unsigned j = i >> 6, r = i & 63U;
            const float ev = reinterpret_cast<const float*>(AS + r * 32U)[j];
            E[i] = ev;
            ES[i] = __fmul_rn(ev, static_cast<float>(reinterpret_cast<const int*>(AS + r * 32U + 16U)[j]));
        }
        __syncthreads();
        if (!active) continue;
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            std::uint32_t a[MT][4];
            float e[MT][2], es[MT][2];
#pragma unroll
            for (unsigned mt = 0; mt < MT; ++mt) {
                core::imma::load_a_frag(a[mt], A + (rbase + 16U * mt) * kDAStride + 32U * j, kDAStride, static_cast<int>(lane));
                const unsigned r0 = rbase + 16U * mt + g;
                e[mt][0] = E[j * 64U + r0]; e[mt][1] = E[j * 64U + r0 + 8U];
                es[mt][0] = ES[j * 64U + r0]; es[mt][1] = ES[j * 64U + r0 + 8U];
            }
#pragma unroll
            for (unsigned n = 0; n < 2U; ++n) {
                const unsigned ob = 16U * wn + 8U * n;
                std::uint32_t bb[2]; Source::down_words(W+(ob+g)*L::WS,j,q,bb[0],bb[1]);
                const unsigned oc = ob + 2U * q;
                const float2 dd = *reinterpret_cast<const float2*>(Dv + j * OUTS + oc);
                const float2 mm = *reinterpret_cast<const float2*>(Mv + j * OUTS + oc);
#pragma unroll
                for (unsigned mt = 0; mt < MT; ++mt) {
                    int r[4];
                    core::imma::mma_seeded(r, a[mt], bb);
#pragma unroll
                    for (unsigned el = 0; el < 4U; ++el) {
                        const unsigned hr = el >> 1, col = el & 1U;
                        acc[mt][n][el]=Source::fast_down_acc(r[el],e[mt][hr],col?dd.y:dd.x,col?mm.y:mm.x,es[mt][hr],acc[mt][n][el]);
                    }
                }
            }
        }
    }
    if (!active) return;
#pragma unroll
    for (unsigned mt = 0; mt < MT; ++mt)
#pragma unroll
        for (unsigned n = 0; n < 2U; ++n)
#pragma unroll
            for (unsigned hr = 0; hr < 2U; ++hr) {
                const unsigned r = rbase + 16U * mt + g + 8U * hr;
                if (r >= count) continue;
                const float out0v = acc[mt][n][2U * hr], out1v = acc[mt][n][2U * hr + 1U];
                const unsigned o = out0 + 16U * wn + 8U * n + 2U * q;
                if constexpr (RouteBF16) {
                    const auto index = static_cast<std::uint64_t>(slot_of[r]) * hidden + o;
                    Binding::template route_store<true>(at(v.rawdots, row_of[r]), index, out0v);
                    Binding::template route_store<true>(at(v.rawdots, row_of[r]), index + 1U, out1v);
                } else {
                    *reinterpret_cast<float2*>(at(v.rawdots, row_of[r]) + slot_of[r] * hidden + o) = make_float2(out0v, out1v);
                }
            }
}

// ---------------------------------------------------------------------------------------------------------------------
// Decode shapes (rows <= 8, so a tile holds <= 8 assignments of one expert). Same per-element chains as project_kernel /
// down_kernel; the exact integer group dots come from dp4a over lane groups (integer adds, order-free).
// STRIX_QRTX_DECODE_KERNELS=0 keeps the tiled kernels (reference seam).
constexpr unsigned kRowsMax = 8U;

// gate/up: block = (tile, 32 outputs), warp w = output w, gate and up chains in the same loop (their loads overlap). Lane L: pair p = L / 8, group of the
// pair G = (L / 4) % 2 (group j = 2p + G), elements 8 (L % 4) .. + 7 of group j. Per K block b: u = D_j * float(dot),
// v = DM_j * S, t = u - v, x = E * t; pair = x(G=0) + x(G=1); acc[p] += pair (p = this lane's pair); then
// (acc0 + acc1) + (acc2 + acc3). Then the SwiGLU + down activation quantizer of project_kernel per assignment.
template <class Source, class Binding>
__global__ __launch_bounds__(1024) void project_rows_kernel(const std::uint8_t* gate, const std::uint8_t* up, typename Binding::Rows v,
                                                           typename Binding::Tables tables) {
    constexpr unsigned hidden=Binding::hidden, ffn=Binding::ffn, projection_nonfinite=Binding::projection_nonfinite;
    using Csr=typename Binding::Csr;
    using L = ProjLayout<Source>;
    const Csr& csr = v.csr;
    if ((csr.summary[0] & ~projection_nonfinite) || blockIdx.x >= *csr.tile_count) return;
    const unsigned tile_begin = csr.tile_begins[blockIdx.x], total = csr.tile_begins[blockIdx.x + 1U] - tile_begin;
    // a tile holds <= 8 assignments when every route of a row is distinct; any more run in further chunks of 8
    for (unsigned base = 0; base < total; base += kRowsMax) {
    const unsigned begin = tile_begin + base, count = min(total - base, kRowsMax);
    __syncthreads();  // the previous chunk finished with the shared buffers
    const unsigned expert = csr.tile_experts[blockIdx.x], out0 = blockIdx.y * 32U;
    __shared__ __align__(16) std::uint32_t A[kRowsMax][hidden / 4U];
    __shared__ float E[kRowsMax][hidden / 32U], S[kRowsMax][hidden / 32U];
    __shared__ float GU[2][kRowsMax][32];
    __shared__ unsigned row_of[kRowsMax], slot_of[kRowsMax];
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5;
    if (t < kRowsMax) {
        const unsigned a = t < count ? t : 0U;
        row_of[t] = csr.rows[begin + a]; slot_of[t] = csr.slots[begin + a];
    }
    __syncthreads();
    for (unsigned i = t; i < count * (hidden / 4U); i += 1024U) {
        const unsigned a = i / (hidden / 4U), w = i % (hidden / 4U);
        A[a][w] = reinterpret_cast<const std::uint32_t*>(at(v.codes, row_of[a]))[w];
    }
    for (unsigned i = t; i < count * (hidden / 32U); i += 1024U) {
        const unsigned a = i / (hidden / 32U), g = i % (hidden / 32U);
        E[a][g] = at(v.deltas, row_of[a])[g];
        S[a][g] = static_cast<float>(at(v.sums, row_of[a])[g]);
    }
    __syncthreads();
    const unsigned p = lane >> 3, G = (lane >> 2) & 1U, sub = lane & 3U, j = 2U * p + G;
    constexpr unsigned blocks = hidden / 256U;
    const unsigned o = out0 + warp;
    const std::uint8_t* wrow[2] = {gate + (static_cast<std::uint64_t>(expert) * ffn + o) * Source::project_row_bytes(hidden),
                                   up + (static_cast<std::uint64_t>(expert) * ffn + o) * Source::project_row_bytes(hidden)};
    float acc[2][kRowsMax];
#pragma unroll
    for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
        for (unsigned a = 0; a < kRowsMax; ++a) acc[m][a] = 0.0F;
#pragma unroll 2
    for (unsigned b = 0; b < blocks; ++b) {
        weight::A8RowFragment fragment[2];
#pragma unroll
        for(unsigned m=0;m<2;++m) fragment[m]=Source::project_row(wrow[m],b,j,sub);
#pragma unroll
        for(unsigned m=0;m<2;++m) {
            const auto& f=fragment[m];
#pragma unroll
            for(unsigned a=0;a<kRowsMax;++a) {
                if(a>=count) break;
                const unsigned* x=&A[a][(256U*b+32U*j+8U*sub)/4U];
                int dot=__dp4a(int(f.words[0]),int(x[0]),0);
                dot=__dp4a(int(f.words[1]),int(x[1]),dot);
                const float tt=Source::row_project_value(f.scale,f.offset,dot,S[a][8U*b+j]);
                const float xx = __fmul_rn(E[a][8U * b + j], tt);
                const float other = __shfl_xor_sync(0xffffffffU, xx, 4);
                const float pair = G ? __fadd_rn(other, xx) : __fadd_rn(xx, other);
                acc[m][a] = __fadd_rn(acc[m][a], pair);
            }
        }
    }
#pragma unroll
    for (unsigned m = 0; m < 2U; ++m)
#pragma unroll
        for (unsigned a = 0; a < kRowsMax; ++a) {
            if (a >= count) break;
            const float c0 = __shfl_sync(0xffffffffU, acc[m][a], 0), c1 = __shfl_sync(0xffffffffU, acc[m][a], 8);
            const float c2 = __shfl_sync(0xffffffffU, acc[m][a], 16), c3 = __shfl_sync(0xffffffffU, acc[m][a], 24);
            if (lane == 0U) GU[m][a][warp] = __fadd_rn(__fadd_rn(c0, c1), __fadd_rn(c2, c3));
        }
    __syncthreads();
    Binding::template activation_tile<32,1024>(&GU[0][0][0],&GU[1][0][0],v,tables,row_of,slot_of,count,out0);
    }
}

// down: block = (tile, 8 hidden outputs), warp = one output. Lane L: chain j = L / 8, elements 4 (L % 8) .. + 3 of
// group g = 4c + j, chunks c ascending: first = d * float(dot), Q5_1: combined = first + m * S, contribution = E *
// combined, acc_j += contribution; then (acc0 + acc1) + (acc2 + acc3) into rawdots.
template <class Source, class Binding, bool RouteBF16 = false>
__global__ __launch_bounds__(256) void down_rows_kernel(const std::uint8_t* weights, typename Binding::Rows v) {
    constexpr unsigned hidden=Binding::hidden, ffn=Binding::ffn, projection_nonfinite=Binding::projection_nonfinite;
    using Csr=typename Binding::Csr;
    using L = DownLayout<Source>;
    const Csr& csr = v.csr;
    if (csr.summary[0] || blockIdx.x >= *csr.tile_count) return;
    const unsigned tile_begin = csr.tile_begins[blockIdx.x], total = csr.tile_begins[blockIdx.x + 1U] - tile_begin;
    // a tile holds <= 8 assignments when every route of a row is distinct; any more run in further chunks of 8
    for (unsigned base = 0; base < total; base += kRowsMax) {
    const unsigned begin = tile_begin + base, count = min(total - base, kRowsMax);
    __syncthreads();  // the previous chunk finished with the shared buffers
    const unsigned expert = csr.tile_experts[blockIdx.x];
    constexpr unsigned groups = ffn / 32U;
    __shared__ __align__(16) std::uint32_t A[kRowsMax][ffn / 4U];
    __shared__ float E[kRowsMax][groups], S[kRowsMax][groups];
    __shared__ unsigned row_of[kRowsMax], slot_of[kRowsMax];
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5;
    if (t < kRowsMax) {
        const unsigned a = t < count ? t : 0U;
        row_of[t] = csr.rows[begin + a]; slot_of[t] = csr.slots[begin + a];
    }
    __syncthreads();
    for (unsigned i = t; i < count * (ffn / 4U); i += 256U) {
        const unsigned a = i / (ffn / 4U), w = i % (ffn / 4U);
        A[a][w] = reinterpret_cast<const std::uint32_t*>(at(v.down_codes, row_of[a]) + slot_of[a] * ffn)[w];
    }
    for (unsigned i = t; i < count * groups; i += 256U) {
        const unsigned a = i / groups, g = i % groups;
        E[a][g] = at(v.down_deltas, row_of[a])[slot_of[a] * groups + g];
        S[a][g] = static_cast<float>(at(v.down_sums, row_of[a])[slot_of[a] * groups + g]);
    }
    __syncthreads();
    const unsigned o = blockIdx.y * 8U + warp, j = lane >> 3, sub = lane & 7U;
    const std::uint8_t* wrow = weights + (static_cast<std::uint64_t>(expert) * hidden + o) * Source::down_row_bytes(ffn);
    float acc[kRowsMax];
#pragma unroll
    for (unsigned a = 0; a < kRowsMax; ++a) acc[a] = 0.0F;
#pragma unroll
    for (unsigned c = 0; c < groups / 4U; ++c) {
        const unsigned g = 4U * c + j;
        const auto fragment=Source::down_row(wrow,g,sub);
        const unsigned word=fragment.words[0];
#pragma unroll
        for (unsigned a = 0; a < kRowsMax; ++a) {
            if (a >= count) break;
            int dot = __dp4a(static_cast<int>(word), static_cast<int>(A[a][8U * g + sub]), 0);
            dot += __shfl_xor_sync(0xffffffffU, dot, 1);
            dot += __shfl_xor_sync(0xffffffffU, dot, 2);
            dot += __shfl_xor_sync(0xffffffffU, dot, 4);
            const float combined=Source::row_down_value(fragment.scale,fragment.offset,dot,S[a][g]);
            acc[a] = __fadd_rn(acc[a], __fmul_rn(E[a][g], combined));
        }
    }
#pragma unroll
    for (unsigned a = 0; a < kRowsMax; ++a) {
        if (a >= count) break;
        const float c0 = __shfl_sync(0xffffffffU, acc[a], 0), c1 = __shfl_sync(0xffffffffU, acc[a], 8);
        const float c2 = __shfl_sync(0xffffffffU, acc[a], 16), c3 = __shfl_sync(0xffffffffU, acc[a], 24);
        if constexpr (RouteBF16) {
            if (lane == 0U) Binding::template route_store<true>(at(v.rawdots, row_of[a]),
                static_cast<std::uint64_t>(slot_of[a]) * hidden + o,
                __fadd_rn(__fadd_rn(c0, c1), __fadd_rn(c2, c3)));
        } else {
        if (lane == 0U) at(v.rawdots, row_of[a])[slot_of[a] * hidden + o] = __fadd_rn(__fadd_rn(c0, c1), __fadd_rn(c2, c3));
        }
    }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Native expert types (the GSQ-RCO IQ file: gate/up IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS, down IQ4_NL / Q2_0; also any other
// cuda_qwen_quant.cuh type). FAST arithmetic, not a node profile: FP32 activations (BF16 input rows, FP32 h), weights
// decoded per 32-value group with the shared decoders (bitwise the host reference values), FP32 dots, SwiGLU with expf. A warp owns
// one output; each lane decodes its groups once per chunk of up to 8 assignments and applies them to every assignment.
constexpr unsigned kNativeChunk = 8U;
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (unsigned d = 16U; d; d >>= 1U) v += __shfl_xor_sync(0xffffffffU, v, d);
    return v;
}
// grid (tiles, ffn / 32), 1024 threads: warp w = output 32 * blockIdx.y + w, gate and up. Writes h = silu(g) * u into
// intermediates[row][slot * 2 * ffn + o].
template <class Source, class Binding>
__global__ __launch_bounds__(1024) void native_project_kernel(const std::uint8_t* gate, const std::uint8_t* up, std::uint32_t type,
                                                              typename Binding::Rows v) {
    constexpr unsigned hidden=Binding::hidden, ffn=Binding::ffn, projection_nonfinite=Binding::projection_nonfinite;
    using Csr=typename Binding::Csr;
    const Csr& csr = v.csr;
    if (csr.summary[0] || blockIdx.x >= *csr.tile_count) return;
    const unsigned tile_begin = csr.tile_begins[blockIdx.x], total = csr.tile_begins[blockIdx.x + 1U] - tile_begin;
    const unsigned expert = csr.tile_experts[blockIdx.x];
    __shared__ std::uint16_t X[kNativeChunk][hidden];  // BF16 input rows
    __shared__ unsigned row_of[kNativeChunk], slot_of[kNativeChunk];
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5, o = blockIdx.y * 32U + warp;
    const std::uint64_t row_bytes = Source::row_bytes(type,hidden);
    const std::uint8_t* wg = gate + (static_cast<std::uint64_t>(expert) * ffn + o) * row_bytes;
    const std::uint8_t* wu = up + (static_cast<std::uint64_t>(expert) * ffn + o) * row_bytes;
    for (unsigned base = 0; base < total; base += kNativeChunk) {
        const unsigned count = min(total - base, kNativeChunk);
        __syncthreads();
        if (t < kNativeChunk) {
            const unsigned a = t < count ? t : 0U;
            row_of[t] = csr.rows[tile_begin + base + a]; slot_of[t] = csr.slots[tile_begin + base + a];
        }
        __syncthreads();
        for (unsigned i = t; i < count * hidden; i += 1024U) {
            const unsigned a = i / hidden, f = i % hidden;
            X[a][f] = at(v.input, row_of[a])[f];
        }
        __syncthreads();
        float g[kNativeChunk], u[kNativeChunk];
#pragma unroll
        for (unsigned a = 0; a < kNativeChunk; ++a) { g[a] = 0.0F; u[a] = 0.0F; }
        for (unsigned grp = lane; grp < hidden / 32U; grp += 32U) {
            float wv[32];
#pragma unroll
            for (unsigned m = 0; m < 2U; ++m) {
                Source::decode(type, m ? wu : wg, grp, wv);
#pragma unroll
                for (unsigned a = 0; a < kNativeChunk; ++a) {
                    if (a >= count) break;
                    const std::uint16_t* x = &X[a][32U * grp];
                    float sacc = 0.0F;
#pragma unroll
                    for (unsigned j = 0; j < 32U; ++j) sacc = __fmaf_rn(wv[j], __uint_as_float(static_cast<unsigned>(x[j]) << 16U), sacc);
                    if (m) u[a] += sacc; else g[a] += sacc;
                }
            }
        }
#pragma unroll
        for (unsigned a = 0; a < kNativeChunk; ++a) {
            if (a >= count) break;
            const float gv = warp_sum(g[a]), uv = warp_sum(u[a]);
            if (lane == 0U) {
                const float h = Binding::native_activation(gv,uv);
                at(v.intermediates, row_of[a])[slot_of[a] * 2U * ffn + o] = h;
                if (!isfinite(h)) { atomicOr(at(v.status, row_of[a]), projection_nonfinite); atomicOr(csr.summary, projection_nonfinite); }
            }
        }
    }
}
// grid (tiles, hidden / 32), 1024 threads: warp w = hidden output 32 * blockIdx.y + w; lanes 0..19 own the 20 groups of
// 640. Writes rawdots[row][slot * hidden + o].
template <class Source, class Binding>
__global__ __launch_bounds__(1024) void native_down_kernel(const std::uint8_t* down, std::uint32_t type, typename Binding::Rows v) {
    constexpr unsigned hidden=Binding::hidden, ffn=Binding::ffn, projection_nonfinite=Binding::projection_nonfinite;
    using Csr=typename Binding::Csr;
    const Csr& csr = v.csr;
    if (csr.summary[0] || blockIdx.x >= *csr.tile_count) return;
    const unsigned tile_begin = csr.tile_begins[blockIdx.x], total = csr.tile_begins[blockIdx.x + 1U] - tile_begin;
    const unsigned expert = csr.tile_experts[blockIdx.x];
    __shared__ float H[kNativeChunk][ffn];
    __shared__ unsigned row_of[kNativeChunk], slot_of[kNativeChunk];
    const unsigned t = threadIdx.x, lane = t & 31U, warp = t >> 5, o = blockIdx.y * 32U + warp;
    const std::uint64_t row_bytes = Source::row_bytes(type,ffn);
    const std::uint8_t* w = down + (static_cast<std::uint64_t>(expert) * hidden + o) * row_bytes;
    for (unsigned base = 0; base < total; base += kNativeChunk) {
        const unsigned count = min(total - base, kNativeChunk);
        __syncthreads();
        if (t < kNativeChunk) {
            const unsigned a = t < count ? t : 0U;
            row_of[t] = csr.rows[tile_begin + base + a]; slot_of[t] = csr.slots[tile_begin + base + a];
        }
        __syncthreads();
        for (unsigned i = t; i < count * ffn; i += 1024U) {
            const unsigned a = i / ffn, f = i % ffn;
            H[a][f] = at(v.intermediates, row_of[a])[slot_of[a] * 2U * ffn + f];
        }
        __syncthreads();
        float acc[kNativeChunk];
#pragma unroll
        for (unsigned a = 0; a < kNativeChunk; ++a) acc[a] = 0.0F;
        if (lane < ffn / 32U) {
            float wv[32];
            Source::decode(type, w, lane, wv);
#pragma unroll
            for (unsigned a = 0; a < kNativeChunk; ++a) {
                if (a >= count) break;
                float sacc = 0.0F;
#pragma unroll
                for (unsigned j = 0; j < 32U; ++j) sacc = __fmaf_rn(wv[j], H[a][32U * lane + j], sacc);
                acc[a] = sacc;
            }
        }
#pragma unroll
        for (unsigned a = 0; a < kNativeChunk; ++a) {
            if (a >= count) break;
            const float d = warp_sum(acc[a]);
            if (lane == 0U) at(v.rawdots, row_of[a])[slot_of[a] * hidden + o] = d;
        }
    }
}


} // namespace strix::cuda::core::moe_a8
