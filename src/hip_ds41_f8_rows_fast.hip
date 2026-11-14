// F8 rows with 16-byte weight loads, one output row per wave; arithmetic of f16_rows (see hip_ds41_f8_rows_fast.hpp).
#include "strix/env_flags.hpp"
#include "strix/hip_ds41_f8_rows_fast.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace strix::ds41 {
namespace {

constexpr unsigned kWave = 32, kWaves = 8, kMaxRows = 8;

template <unsigned D> __device__ __forceinline__ float xor_lane(float v) {  // DPP row_xmask: lane ^ D (D < 16)
    return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(__builtin_bit_cast(int, v), __builtin_bit_cast(int, v),
                                                                  static_cast<int>(0x160U + D), 0xF, 0xF, false));
}

// E4M3 code `c` (bits 0..7 of `b`) times the block scale as F32. Every packed value is an F16 value (ds41_packed), so
// the F32 product equals the F16 value the row kernel widens to: no rounding on any path.
template <unsigned Deq> __device__ __forceinline__ float f8_value(unsigned b, float scale) {
    if constexpr (Deq == 0) {  // the row kernel's formula without its F16 round trip
        const unsigned sign = (b & 0x80U) << 24U, e = (b >> 3U) & 15U, m = b & 7U;
        const float normal = __builtin_bit_cast(float, sign | ((e + 120U) << 23U) | (m << 20U));
        const float sub = __builtin_bit_cast(float, sign | __builtin_bit_cast(unsigned, static_cast<float>(m) * 0x1p-9F));
        return (e != 0U ? normal : sub) * scale;
    } else if constexpr (Deq == 1) {
        // sign | eeee mmm at F32 bits 31 | 26..20: the E4M3 value times 2^-120 for normal codes (exponent e) and for
        // subnormal ones (an F32 subnormal m x 2^-129; denormals are kept, -fdenormal-fp-math=ieee); both products exact
        const float v = __builtin_bit_cast(float, ((b & 0x80U) << 24U) | ((b & 0x7fU) << 20U));
        return (v * 0x1p120F) * scale;
    } else {
        return static_cast<float>(b & 0xffU) * scale;  // timing roof only
    }
}

// Block: 8 waves x RPW output rows each (wave wv owns rows RPW (8 block + wv) .. + RPW - 1). The activations of every
// row are staged in LDS once per block, by spans of Span elements, so a block reads them once per 8 RPW weight rows
// (the row kernel of hip_ds41_f16_rows.hip stages all 6144 x 4 bytes for every 8 rows of 6336 bytes: its bottleneck on
// the Engram rows). Inside a span the wave walks its rows in turn; per row and pair p (512 codes) lane l holds codes
// 16 l .. 16 l + 15 (one 16-byte load), widens them into its wave's tile, and lane c then takes elements 32 m + c
// (m = 0..15: chunk 2p then 2p + 1), so every class sums its elements in ascending order. A consumed load is replaced
// at once by the same row's load one span later (RPW x Span / 512 loads in flight per lane).
template <unsigned R, unsigned RPW, unsigned Span, unsigned Deq>
__global__ __launch_bounds__(kWaves * kWave) void f8_rows_fast_kernel(F16RowsView v) {
    constexpr unsigned PP = Span / 512U;  // pairs per span
    __shared__ float xs[R * Span];
    __shared__ float tile[kWaves][512];
    const unsigned l = threadIdx.x % kWave, wv = threadIdx.x / kWave, rows = v.rows, rb = v.width + v.width / 32U;
    const unsigned base = (blockIdx.x * kWaves + wv) * RPW, pairs = v.width / 512U;
    const std::uint8_t* rowp[RPW];
#pragma unroll
    for (unsigned i = 0; i < RPW; ++i) {
        const unsigned o = base + i, oc = o < v.count ? o : v.count - 1U;
        rowp[i] = v.weight_f8 + static_cast<std::size_t>(v.first + oc) * rb;
    }
    float acc[RPW][R];
#pragma unroll
    for (unsigned i = 0; i < RPW; ++i)
#pragma unroll
        for (unsigned r = 0; r < R; ++r) acc[i][r] = 0.0F;
    uint4 buf[RPW][PP];
    unsigned ex[RPW][PP];
#pragma unroll
    for (unsigned i = 0; i < RPW; ++i)
#pragma unroll
        for (unsigned q = 0; q < PP; ++q) {
            buf[i][q] = q < pairs ? *reinterpret_cast<const uint4*>(rowp[i] + 512U * q + 16U * l) : uint4{0, 0, 0, 0};
            ex[i][q] = q < pairs ? rowp[i][v.width + 16U * q + l / 2U] : 0U;
        }
    for (unsigned s0 = 0; s0 < v.width; s0 += Span) {
        const unsigned n = v.width - s0 < Span ? v.width - s0 : Span, sp = n / 512U, p0 = s0 / 512U;
        __syncthreads();
        for (unsigned i = threadIdx.x; i < rows * n; i += kWaves * kWave) {
            const unsigned r = i / n, q = i % n;
            xs[r * Span + q] = v.x[static_cast<std::size_t>(r) * v.x_stride + s0 + q];
        }
        __syncthreads();
#pragma unroll
        for (unsigned i = 0; i < RPW; ++i) {
#pragma unroll
            for (unsigned q = 0; q < PP; ++q) {
                if (q < sp) {
                    const unsigned x = ex[i][q];
                    // E8M0 exponent x: 2^(x - 127) as F32 (x = 0 is 2^-127, an F32 subnormal)
                    const float scale = x != 0U ? __builtin_bit_cast(float, x << 23U) : 0x1p-127F;
                    const unsigned words[4] = {buf[i][q].x, buf[i][q].y, buf[i][q].z, buf[i][q].w};
                    const unsigned nxt = p0 + q + PP;  // the same row's pair one span later
                    if (nxt < pairs) {
                        buf[i][q] = *reinterpret_cast<const uint4*>(rowp[i] + 512U * nxt + 16U * l);
                        ex[i][q] = rowp[i][v.width + 16U * nxt + l / 2U];
                    }
                    float f[16];
#pragma unroll
                    for (unsigned j = 0; j < 16; ++j) f[j] = f8_value<Deq>(words[j / 4U] >> (8U * (j % 4U)), scale);
#pragma unroll
                    for (unsigned j = 0; j < 4; ++j)
                        reinterpret_cast<float4*>(tile[wv])[4U * l + j] = float4{f[4 * j], f[4 * j + 1], f[4 * j + 2], f[4 * j + 3]};
                    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                    __builtin_amdgcn_wave_barrier();
                    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
                    const unsigned xo = 512U * q;
#pragma unroll
                    for (unsigned m = 0; m < 16; ++m) {   // chunk 2p (m < 8) then chunk 2p + 1: elements 32 m + l, ascending
                        const float w = tile[wv][32U * m + l];
#pragma unroll
                        for (unsigned r = 0; r < R; ++r)
                            if (r < rows) acc[i][r] = acc[i][r] + w * xs[r * Span + xo + 32U * m + l];
                    }
                    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                    __builtin_amdgcn_wave_barrier();
                    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
                }
            }
        }
    }
#pragma unroll
    for (unsigned i = 0; i < RPW; ++i) {
        const unsigned o = base + i;
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            if (r >= rows) break;
            float t = acc[i][r];
            t = t + __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(unsigned, t), __builtin_bit_cast(unsigned, t),
                                                                           0x76543210U, 0xfedcba98U, false, false));
            t = t + xor_lane<8>(t);
            t = t + xor_lane<4>(t);
            t = t + xor_lane<2>(t);
            t = t + xor_lane<1>(t);
            if (l == 0U && o < v.count) v.out[static_cast<std::size_t>(r) * v.out_stride + o] = t;
        }
    }
}

template <unsigned R, unsigned RPW, unsigned Deq>
void launch_rows(const F16RowsView& v, hipStream_t s) {
    // R x Span floats of LDS for the activations: 8 KB at 1 row, 16 KB at 2-4 rows, 32 KB at 5-8 rows
    constexpr unsigned Span = R <= 2 ? 2048U : 1024U;
    const dim3 g((v.count + kWaves * RPW - 1U) / (kWaves * RPW)), b(kWaves * kWave);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(f8_rows_fast_kernel<R, RPW, Span, Deq>), g, b, 0, s, v);
}
template <unsigned RPW, unsigned Deq>
void launch(const F16RowsView& v, hipStream_t s) {
    if (v.rows == 1) launch_rows<1, RPW, Deq>(v, s);
    else if (v.rows == 2) launch_rows<2, RPW, Deq>(v, s);
    else if (v.rows <= 4) launch_rows<4, RPW, Deq>(v, s);
    else launch_rows<8, RPW, Deq>(v, s);
}


// E8M0 exponent byte x as the F32 2^(x - 127) (x = 0 is 2^-127, an F32 subnormal)
__device__ __forceinline__ float e8m0_scale(unsigned x) {
    x &= 0xffU;
    return x != 0U ? __builtin_bit_cast(float, x << 23U) : 0x1p-127F;
}
// E4M3 code in bits 0..7 of `b` times `scale`, exactly as f8_value<1>
__device__ __forceinline__ float f8_code(unsigned b, float scale) {
    const float v = __builtin_bit_cast(float, ((b & 0x80U) << 24U) | ((b & 0x7fU) << 20U));
    return (v * 0x1p120F) * scale;
}

// ---- 1 row, compact codes (round 2) ----
// Wave wv owns RPW consecutive output rows and walks them one after the other, so its loads are one linear stream:
// row r pair p = 512 codes, lane l loads codes 16 l .. 16 l + 15 (16 bytes), D pairs in flight (D independent of
// RPW and of the activation staging). Each consumed pair goes through a wave-private LDS tile as CODES (4 dwords per
// lane, 640 bytes per wave; the round-1 kernel wrote 2 KB of widened F32): dword k of lane l (block m = l / 2,
// half h = l % 2: classes 16 h + 4 k .. + 3) lands at group 4 h + k, slot m, so lane c then reads its class group
// c / 4 for the 16 blocks with 4 16-byte reads and takes byte c % 4. The row's exponents are one 16-byte load per
// pair (lanes 0..pairs-1), issued a row ahead; pair p's 16 scales come from lane p by readlane (wave-uniform).
// The activations are staged ONCE per block, transposed (lane c's elements 32 m + c contiguous, row stride
// width / 32 + 4 floats: 4 16-byte reads per pair, no bank conflicts), with one block barrier; after it every wave
// runs on its own. Lane c = class c sums elements 32 m + c in ascending order (pair by pair, m ascending), then the
// permlanex16 / DPP butterfly 16, 8, 4, 2, 1 of f16_rows: the same bits as the row and lanes kernels.
constexpr unsigned kTileG = 20;  // dwords per class group in the code tile (16 blocks + 4 pad: conflict-free writes)
template <unsigned W, unsigned RPW, unsigned D>
__global__ __launch_bounds__(W * kWave) void f8_rows_compact_kernel(F16RowsView v) {
    extern __shared__ uint4 f8c_smem[];
    const unsigned width = v.width, pairs = width / 512U, P = width / 32U + 4U, rb = width + width / 32U;
    float* xT = reinterpret_cast<float*>(f8c_smem);
    const unsigned l = threadIdx.x % kWave, wv = threadIdx.x / kWave;
    unsigned* tile = reinterpret_cast<unsigned*>(f8c_smem) + 32U * P + wv * (8U * kTileG);
    const unsigned base = (blockIdx.x * W + wv) * RPW;
    const auto rowp = [&](unsigned i) {
        const unsigned o = base + i, oc = o < v.count ? o : v.count - 1U;
        return v.weight_f8 + static_cast<std::size_t>(v.first + oc) * rb;
    };
    uint4 buf[D];
    unsigned lr = 0, lp = 0;  // next pair to load: row lr, pair lp
    const std::uint8_t* lrow = rowp(0);
#pragma unroll
    for (unsigned d = 0; d < D; ++d) {
        buf[d] = uint4{0, 0, 0, 0};
        if (lr < RPW) {
            buf[d] = *reinterpret_cast<const uint4*>(lrow + 512U * lp + 16U * l);
            if (++lp == pairs) { lp = 0; ++lr; lrow = rowp(lr); }
        }
    }
    uint4 exc{0, 0, 0, 0}, exn{0, 0, 0, 0};
    if (l < pairs) {
        exc = *reinterpret_cast<const uint4*>(rowp(0) + width + 16U * l);
        if (RPW > 1U) exn = *reinterpret_cast<const uint4*>(rowp(1) + width + 16U * l);
    }
    for (unsigned i = threadIdx.x; i < width; i += W * kWave) xT[(i % 32U) * P + i / 32U] = v.x[i];
    __syncthreads();
    const unsigned sh = 8U * (l % 4U), wofs = 4U * (l % 2U) * kTileG + l / 2U;
    const unsigned* const trd = tile + (l / 4U) * kTileG;
    const float* const xl = xT + l * P;
    float acc = 0.0F;
    unsigned cr = 0, cp = 0;  // pair being consumed: row cr, pair cp
    const unsigned total = RPW * pairs;
    for (unsigned s0 = 0; s0 < total; s0 += D) {
#pragma unroll
        for (unsigned d = 0; d < D; ++d) {
            if (s0 + d < total) {
                const uint4 cur = buf[d];
                if (lr < RPW) {
                    buf[d] = *reinterpret_cast<const uint4*>(lrow + 512U * lp + 16U * l);
                    if (++lp == pairs) { lp = 0; ++lr; lrow = rowp(lr); }
                }
                tile[wofs] = cur.x;
                tile[wofs + kTileG] = cur.y;
                tile[wofs + 2U * kTileG] = cur.z;
                tile[wofs + 3U * kTileG] = cur.w;
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
                uint4 c4[4];
                float4 x4[4];
#pragma unroll
                for (unsigned j = 0; j < 4; ++j) {
                    c4[j] = *reinterpret_cast<const uint4*>(trd + 4U * j);
                    x4[j] = *reinterpret_cast<const float4*>(xl + 16U * cp + 4U * j);
                }
                const unsigned e[4] = {static_cast<unsigned>(__builtin_amdgcn_readlane(static_cast<int>(exc.x), static_cast<int>(cp))),
                                       static_cast<unsigned>(__builtin_amdgcn_readlane(static_cast<int>(exc.y), static_cast<int>(cp))),
                                       static_cast<unsigned>(__builtin_amdgcn_readlane(static_cast<int>(exc.z), static_cast<int>(cp))),
                                       static_cast<unsigned>(__builtin_amdgcn_readlane(static_cast<int>(exc.w), static_cast<int>(cp)))};
#pragma unroll
                for (unsigned j = 0; j < 4; ++j) {
                    const unsigned cw[4] = {c4[j].x, c4[j].y, c4[j].z, c4[j].w};
                    const float xv[4] = {x4[j].x, x4[j].y, x4[j].z, x4[j].w};
#pragma unroll
                    for (unsigned q = 0; q < 4; ++q) {  // block m = 4 j + q: element 512 cp + 32 m + c
                        const float w = f8_code(cw[q] >> sh, e8m0_scale(e[j] >> (8U * q)));
                        acc = acc + w * xv[q];
                    }
                }
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
                if (++cp == pairs) {
                    float t = acc;
                    t = t + __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(unsigned, t), __builtin_bit_cast(unsigned, t),
                                                                                   0x76543210U, 0xfedcba98U, false, false));
                    t = t + xor_lane<8>(t);
                    t = t + xor_lane<4>(t);
                    t = t + xor_lane<2>(t);
                    t = t + xor_lane<1>(t);
                    const unsigned o = base + cr;
                    if (l == 0U && o < v.count) v.out[o] = t;
                    acc = 0.0F;
                    cp = 0;
                    ++cr;
                    exc = exn;
                    if (cr + 1U < RPW && l < pairs) exn = *reinterpret_cast<const uint4*>(rowp(cr + 1U) + width + 16U * l);
                }
            }
        }
    }
}

template <unsigned W, unsigned RPW, unsigned D>
void launch_compact(const F16RowsView& v, hipStream_t s) {
    const std::size_t lds = (32U * (v.width / 32U + 4U) + W * 8U * kTileG) * 4U;
    const dim3 g((v.count + W * RPW - 1U) / (W * RPW)), b(W * kWave);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(f8_rows_compact_kernel<W, RPW, D>), g, b, lds, s, v);
}

// ---- 2..8 rows: lanes form with exact R, codes prefetched compact (round 2) ----
// 32 / CL lanes per output, lane cg holds classes CL cg .. CL cg + CL - 1 (CL = 8: 8-byte code loads, 4 lanes per
// output, the lanes kernel's layout; CL = 16: 16-byte loads, 2 lanes per output) of K outputs. Step n reads the
// codes of elements 32 n + CL cg .. + CL - 1 of each output (A steps in flight, kept as raw codes: 2 or 4 VGPRs per
// load) and one exponent dword per 4 steps (blocks 4 q .. 4 q + 3), and widens a step's codes only when it is
// consumed. The activations are staged per 1024 elements for the block's W x CL x K outputs (the lanes kernel: 64),
// and each lane's x reads serve its K outputs. Class c sums its elements in ascending n; the butterfly pairs classes
// ^16, ^8 (DPP between the output's lanes, in-lane for ^8 when CL = 16), then ^4, ^2, ^1 in-lane: the same bits.
template <unsigned CL> struct CodeLoad;
template <> struct CodeLoad<8> { using T = uint2; };
template <> struct CodeLoad<16> { using T = uint4; };
template <unsigned R, unsigned CL, unsigned K, unsigned W, unsigned A>
__global__ __launch_bounds__(W * kWave) void f8_rows_rn_kernel(F16RowsView v) {
    constexpr unsigned LPO = kWave / CL, OPW = kWave / LPO, OPB = W * OPW * K, kCh = 32;
    static_assert(A % 4U == 0 && kCh % A == 0);
    using CT = typename CodeLoad<CL>::T;
    __shared__ float4 xs[R][kCh * 8];
    const unsigned l = threadIdx.x % kWave, wv = threadIdx.x / kWave, cg = l % LPO, slot = l / LPO;
    const unsigned width = v.width, steps = width / 32U, rb = width + width / 32U;
    const std::uint8_t* rp[K];
    unsigned o[K];
#pragma unroll
    for (unsigned k = 0; k < K; ++k) {
        o[k] = blockIdx.x * OPB + k * (W * OPW) + wv * OPW + slot;
        const unsigned oc = o[k] < v.count ? o[k] : v.count - 1U;
        rp[k] = v.weight_f8 + static_cast<std::size_t>(v.first + oc) * rb;
    }
    float acc[K][R][CL];
#pragma unroll
    for (unsigned k = 0; k < K; ++k)
#pragma unroll
        for (unsigned r = 0; r < R; ++r)
#pragma unroll
            for (unsigned j = 0; j < CL; ++j) acc[k][r][j] = 0.0F;
    CT cw[K][A];
    unsigned ew[K][A / 4U];
#pragma unroll
    for (unsigned k = 0; k < K; ++k) {
#pragma unroll
        for (unsigned a = 0; a < A; ++a) cw[k][a] = *reinterpret_cast<const CT*>(rp[k] + 32U * a + CL * cg);
#pragma unroll
        for (unsigned q = 0; q < A / 4U; ++q) ew[k][q] = *reinterpret_cast<const unsigned*>(rp[k] + width + 4U * q);
    }
    for (unsigned c0 = 0; c0 < steps; c0 += kCh) {
        const unsigned cs = steps - c0 < kCh ? steps - c0 : kCh;
        __syncthreads();
        for (unsigned i = threadIdx.x; i < R * cs * 8U; i += W * kWave) {
            const unsigned r = i / (cs * 8U), q = i % (cs * 8U);
            xs[r][q] = reinterpret_cast<const float4*>(v.x + static_cast<std::size_t>(r) * v.x_stride + 32U * c0)[q];
        }
        __syncthreads();
        for (unsigned k0 = 0; k0 < cs; k0 += A) {
#pragma unroll
            for (unsigned a = 0; a < A; ++a) {
                const unsigned n = c0 + k0 + a;  // steps % A == 0, so every n < steps
                float wf[K][CL];
#pragma unroll
                for (unsigned k = 0; k < K; ++k) {
                    const CT cur = cw[k][a];
                    const float scale = e8m0_scale(ew[k][a / 4U] >> (8U * (a % 4U)));
                    // branch-free refill (past the end: the same step again, an L2 hit): a guarded load lets the compiler
                    // gather every step's refill ahead of the compute, with old and new codes live at once (spills)
                    const unsigned nn = n + A < steps ? n + A : n;
                    cw[k][a] = *reinterpret_cast<const CT*>(rp[k] + 32U * nn + CL * cg);
                    if (a % 4U == 3U) ew[k][a / 4U] = *reinterpret_cast<const unsigned*>(rp[k] + width + (nn - 3U));
                    unsigned words[CL / 4U];
                    if constexpr (CL == 8) {
                        words[0] = cur.x; words[1] = cur.y;
                    } else {
                        words[0] = cur.x; words[1] = cur.y; words[2] = cur.z; words[3] = cur.w;
                    }
#pragma unroll
                    for (unsigned j = 0; j < CL; ++j) wf[k][j] = f8_code(words[j / 4U] >> (8U * (j % 4U)), scale);
                }
#pragma unroll
                for (unsigned r = 0; r < R; ++r) {
                    float xv[CL];
#pragma unroll
                    for (unsigned jj = 0; jj < CL / 4U; ++jj) {
                        const float4 x4 = xs[r][8U * (k0 + a) + (CL / 4U) * cg + jj];
                        xv[4 * jj] = x4.x; xv[4 * jj + 1] = x4.y; xv[4 * jj + 2] = x4.z; xv[4 * jj + 3] = x4.w;
                    }
#pragma unroll
                    for (unsigned k = 0; k < K; ++k)
#pragma unroll
                        for (unsigned j = 0; j < CL; ++j) acc[k][r][j] = acc[k][r][j] + wf[k][j] * xv[j];
                }
                // keep each step's decode and x reads inside the step (no hoisting across the unrolled steps: the
                // scheduler otherwise pulls later steps' LDS reads and decodes up and spills)
                __builtin_amdgcn_sched_barrier(0);
            }
        }
    }
#pragma unroll
    for (unsigned k = 0; k < K; ++k) {
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            float total;
            if constexpr (CL == 8) {
                float s[8];
#pragma unroll
                for (unsigned j = 0; j < 8; ++j) {
                    float t = acc[k][r][j];
                    t = t + xor_lane<2>(t);  // class ^ 16
                    t = t + xor_lane<1>(t);  // class ^ 8
                    s[j] = t;
                }
                const float s4[4] = {s[0] + s[4], s[1] + s[5], s[2] + s[6], s[3] + s[7]};  // class ^ 4
                const float s2[2] = {s4[0] + s4[2], s4[1] + s4[3]};                        // class ^ 2
                total = s2[0] + s2[1];                                                      // class ^ 1
            } else {
                float s[16];
#pragma unroll
                for (unsigned j = 0; j < 16; ++j) s[j] = acc[k][r][j] + xor_lane<1>(acc[k][r][j]);  // class ^ 16
                float s8[8];
#pragma unroll
                for (unsigned j = 0; j < 8; ++j) s8[j] = s[j] + s[j + 8];  // class ^ 8
                const float s4[4] = {s8[0] + s8[4], s8[1] + s8[5], s8[2] + s8[6], s8[3] + s8[7]};  // class ^ 4
                const float s2[2] = {s4[0] + s4[2], s4[1] + s4[3]};                                // class ^ 2
                total = s2[0] + s2[1];                                                              // class ^ 1
            }
            if (cg == 0U && o[k] < v.count) v.out[static_cast<std::size_t>(r) * v.out_stride + o[k]] = total;
        }
    }
}

template <unsigned R, unsigned CL, unsigned K, unsigned W, unsigned A>
void launch_rn(const F16RowsView& v, hipStream_t s) {
    constexpr unsigned OPB = W * CL * K;  // W waves x CL outputs (32 lanes / (32 / CL) per output) x K
    hipLaunchKernelGGL(HIP_KERNEL_NAME(f8_rows_rn_kernel<R, CL, K, W, A>), dim3((v.count + OPB - 1U) / OPB), dim3(W * kWave), 0, s, v);
}

}  // namespace

namespace {
bool view_ok(const F16RowsView& v) noexcept {
    return v.weight_f8 && reinterpret_cast<std::uintptr_t>(v.weight_f8) % 16U == 0 && v.width && v.width % 512U == 0 &&
           (v.width + v.width / 32U) % 16U == 0 && v.x && v.out && v.count && v.rows >= 1 && v.rows <= kMaxRows &&
           (v.rows == 1 || v.x_stride % 4U == 0);
}
}  // namespace

namespace {
// STRIX_DS41_ENGRAM_FAST (1 row): unset or 1 = the round-1 kernel (the P13 default), 2 = the compact kernel (round 2),
// 0 = off (lanes row kernel). STRIX_DS41_ENGRAM_RN=1: the RN kernels at 2..8 rows (default off). Read once.
// Internal mode: 0 off, 1 round-1 kernel, 2 compact kernel.
unsigned fast_mode() noexcept {
    static const unsigned m = [] {
        const char* e = std::getenv("STRIX_DS41_ENGRAM_FAST");
        // default 2 (compact kernel) since P15 (approved after P14: R1 -0.6 ms, same bits); =1 the round-1 kernel
        return e && e[0] == '0' ? 0U : e && e[0] == '1' ? 1U : 2U;
    }();
    return m;
}
bool rn_on() noexcept {
    // on by default since P15 (P14: R6 -1.1 ms, same bits); STRIX_DS41_ENGRAM_RN=0 off
    static const bool on = strix::envcfg::flag("STRIX_DS41_ENGRAM_RN", true);
    return on;
}
}  // namespace

bool f8_rows_fast_supported(const F16RowsView& v) noexcept {
    if (v.rows == 1) return (fast_mode() == 2U && f8_rows_compact_supported(v)) || (fast_mode() == 1U && view_ok(v));
    return rn_on() && f8_rows_rn_supported(v);
}

bool f8_rows_fast_enabled() noexcept { return fast_mode() != 0U || rn_on(); }

void enqueue_f8_rows_fast_variant(const F16RowsView& v, unsigned rpw, unsigned dequant, void* stream) {
    if (!view_ok(v)) throw std::invalid_argument("F8 rows (fast) view: width % 512, 16-byte aligned rows, 1..8 rows");
    const auto s = static_cast<hipStream_t>(stream);
    const auto by_deq = [&]<unsigned RPW>() {
        if (dequant == 0U) launch<RPW, 0>(v, s);
        else if (dequant == 2U) launch<RPW, 2>(v, s);
        else launch<RPW, 1>(v, s);
    };
    if (rpw == 2U) by_deq.template operator()<2U>();
    else if (rpw == 8U) by_deq.template operator()<8U>();
    else by_deq.template operator()<4U>();
    if (hipGetLastError() != hipSuccess) throw std::runtime_error("F8 rows (fast) launch");
}


bool f8_rows_compact_supported(const F16RowsView& v) noexcept {
    // LDS: transposed activations 32 x (width / 32 + 4) floats + 8 waves x 640 bytes of code tiles; pairs <= 32 lanes
    return v.rows == 1 && view_ok(v) && v.width <= 12288U;
}

void enqueue_f8_rows_compact_variant(const F16RowsView& v, unsigned waves, unsigned rpw, unsigned depth, void* stream) {
    if (!f8_rows_compact_supported(v)) throw std::invalid_argument("F8 rows (compact) view: 1 row, width % 512 <= 12288, 16-byte aligned rows");
    const auto s = static_cast<hipStream_t>(stream);
    const auto by_d = [&]<unsigned W, unsigned RPW>() {
        if (depth == 2U) launch_compact<W, RPW, 2>(v, s);
        else if (depth == 8U) launch_compact<W, RPW, 8>(v, s);
        else launch_compact<W, RPW, 4>(v, s);
    };
    const auto by_rpw = [&]<unsigned W>() {
        if (rpw == 4U) by_d.template operator()<W, 4U>();
        else if (rpw == 16U) by_d.template operator()<W, 16U>();
        else by_d.template operator()<W, 8U>();
    };
    if (waves == 4U) by_rpw.template operator()<4U>();
    else by_rpw.template operator()<8U>();
    if (hipGetLastError() != hipSuccess) throw std::runtime_error("F8 rows (compact) launch");
}

bool f8_rows_rn_supported(const F16RowsView& v) noexcept {
    return v.weight_f8 && reinterpret_cast<std::uintptr_t>(v.weight_f8) % 16U == 0 && v.width && v.width % 256U == 0 &&
           (v.width + v.width / 32U) % 16U == 0 && v.x && reinterpret_cast<std::uintptr_t>(v.x) % 16U == 0 && v.out && v.count &&
           v.rows >= 2 && v.rows <= kMaxRows && v.x_stride % 4U == 0;
}

namespace {
// Bench sweep: rows 3 and 6 take every (classes per lane, outputs per lane, waves, ahead); the other row counts every
// (waves, ahead) of the 8-class 1-output form and 8 waves x 8 ahead of the others.
template <unsigned R>
bool launch_rn_variant(const F16RowsView& v, unsigned cl, unsigned k, unsigned w, unsigned a, hipStream_t s) {
    const auto by_wa = [&]<unsigned CL, unsigned K>() {
        if constexpr (R == 3 || R == 6 || (CL == 8 && K == 1)) {
            if (w == 4U && a == 4U) launch_rn<R, CL, K, 4, 4>(v, s);
            else if (w == 4U) launch_rn<R, CL, K, 4, 8>(v, s);
            else if (a == 4U) launch_rn<R, CL, K, 8, 4>(v, s);
            else launch_rn<R, CL, K, 8, 8>(v, s);
            return true;
        } else {
            if (w != 8U || a != 8U) return false;
            launch_rn<R, CL, K, 8, 8>(v, s);
            return true;
        }
    };
    if (cl == 8U && k == 1U) return by_wa.template operator()<8U, 1U>();
    if (cl == 8U && k == 2U) return by_wa.template operator()<8U, 2U>();
    if (cl == 16U && k == 1U) return by_wa.template operator()<16U, 1U>();
    if constexpr (R <= 4) {
        if (cl == 16U && k == 2U) return by_wa.template operator()<16U, 2U>();
    }
    return false;
}
}  // namespace

bool enqueue_f8_rows_rn_variant(const F16RowsView& v, unsigned cl, unsigned k, unsigned waves, unsigned ahead, void* stream) {
    if (!f8_rows_rn_supported(v)) throw std::invalid_argument("F8 rows (rn) view: 2..8 rows, width % 256, 16-byte aligned rows and x");
    const auto s = static_cast<hipStream_t>(stream);
    bool ok = false;
    switch (v.rows) {
        case 2: ok = launch_rn_variant<2>(v, cl, k, waves, ahead, s); break;
        case 3: ok = launch_rn_variant<3>(v, cl, k, waves, ahead, s); break;
        case 4: ok = launch_rn_variant<4>(v, cl, k, waves, ahead, s); break;
        case 5: ok = launch_rn_variant<5>(v, cl, k, waves, ahead, s); break;
        case 6: ok = launch_rn_variant<6>(v, cl, k, waves, ahead, s); break;
        case 7: ok = launch_rn_variant<7>(v, cl, k, waves, ahead, s); break;
        default: ok = launch_rn_variant<8>(v, cl, k, waves, ahead, s); break;
    }
    if (ok && hipGetLastError() != hipSuccess) throw std::runtime_error("F8 rows (rn) launch");
    return ok;
}

// Engine shapes (engram kv bench, cold, gfx1151): 1 row = the round-1 kernel (rows per wave 4), or with
// STRIX_DS41_ENGRAM_FAST=2 the compact kernel, 8 waves x 8 rows per wave x 4 pairs in flight; 2..8 rows = RN with
// 8 classes per lane, 1 output per lane, 8 waves x 8 steps ahead (kRnShape below).
void enqueue_f8_rows_fast(const F16RowsView& v, void* stream) {
    if (v.rows == 1) {
        if (fast_mode() != 2U || !f8_rows_compact_supported(v)) enqueue_f8_rows_fast_variant(v, 4U, 1U, stream);
        else enqueue_f8_rows_compact_variant(v, 8U, 8U, 4U, stream);
        return;
    }
    // waves, ahead per row count 2..8
    static constexpr unsigned kRnShape[9][2] = {{0, 0}, {0, 0}, {8, 8}, {8, 8}, {8, 8}, {8, 8}, {8, 8}, {8, 8}, {8, 8}};
    if (!enqueue_f8_rows_rn_variant(v, 8U, 1U, kRnShape[v.rows][0], kRnShape[v.rows][1], stream))
        throw std::runtime_error("F8 rows (rn): shape not instantiated");
}

}  // namespace strix::ds41
