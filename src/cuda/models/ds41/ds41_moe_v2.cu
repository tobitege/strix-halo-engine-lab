// deepseek41 routed experts on CUDA, decode v2 (cuda_ds41_moe_v2.hpp): the same declared arithmetic as ds41_moe.cu
// (lane classes, butterfly, SwiGLU, A8), with the weight bytes fetched as whole slabs by 16-byte cp.async (every
// load of a block in flight at once, overlapped with the activation staging) and small work items (8 gate/up
// outputs, 32 down outputs per block) so every SM keeps tens of KB in flight. The A8 of the SwiGLU output moves from
// the gate/up epilogue to the down kernel's prologue (the same a8_group on the same values: the same codes).

#include "../../core/ptx.cuh"
#include "ds41_moe_device.cuh"

#include <cuda_fp16.h>

#include "strix/cuda/ds41_moe_v2.hpp"
#include "strix/ds41_formats.hpp"

#include <stdexcept>
#include <vector>

namespace strix::ds41::cuda {

using namespace moe_detail;

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr std::uint32_t kGuOutputs = kWarps;         // gate/up outputs per block (one per warp)
constexpr std::uint32_t kDownPerWarp = 4;
constexpr std::uint32_t kDownOutputs = kWarps * kDownPerWarp;  // down outputs per block

__constant__ std::uint64_t c_grid_v2[256];
__constant__ int c_v2_mode;  // diagnostics only (0 = the exact kernel)
__constant__ unsigned long long* c_v3_stamps;  // diagnostics only: [block][8 items][4] clock64, nullptr = off
bool g_grid_v2 = false;

using ::strix::cuda::core::ptx_mem::cp_async16;
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

// The block's weight slab: n16 16-byte pieces from src to dst, all issued before any wait.
__device__ __forceinline__ void slab_async(unsigned char* dst, const unsigned char* src, std::uint32_t n16) {
    for (std::uint32_t i = threadIdx.x; i < n16; i += kThreads) cp_async16(dst + 16U * i, src + 16U * i);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

struct GateUpV2Params {
    const std::uint8_t* gate;
    const std::uint8_t* up;
    std::uint64_t expert_stride, row_bytes;
    std::uint32_t hidden, inter;
    const std::uint32_t *experts, *begins, *rows, *slots, *counts;
    const std::int8_t* x_codes;
    const float* x_deltas;
    float limit;
    float* y;  // [slot][inter] SwiGLU outputs
    float* gate_out;
    float* up_out;
};

// Block = (8 outputs, union expert j); warp w owns output o0 + w (its gate and up rows). Lane l takes groups
// l, l + 32, ... in ascending order: the declared lane classes; then the butterfly.
template <int R>
__global__ __launch_bounds__(kThreads) void gate_up_v2_kernel(GateUpV2Params p) {
    const std::uint32_t j = blockIdx.y;
    if (j >= p.counts[0]) return;
    extern __shared__ __align__(16) unsigned char smem[];
    const std::uint32_t G = p.hidden / 32U, J = (G + 31U) / 32U, rb = static_cast<std::uint32_t>(p.row_bytes);
    const std::uint32_t slab = kGuOutputs * rb;                                  // bytes per matrix
    unsigned char* wg = smem;                                                    // [8][row] gate
    unsigned char* wu = smem + slab;                                             // [8][row] up
    auto* grid = reinterpret_cast<std::uint64_t*>(smem + 2U * slab);             // [256]
    auto* xs = reinterpret_cast<std::int32_t*>(grid + 256);                      // [R][J][8][32]
    auto* xd = reinterpret_cast<float*>(xs + R * J * 256U);                      // [R][G]

    const std::uint32_t lane = threadIdx.x & 31U, warp = threadIdx.x / 32U, o0 = blockIdx.x * kGuOutputs;
    const std::uint64_t expert = p.experts[j];
    const std::uint32_t b0 = p.begins[j], b1 = p.begins[j + 1U];
    slab_async(wg, p.gate + expert * p.expert_stride + std::uint64_t{o0} * rb, slab / 16U);
    slab_async(wu, p.up + expert * p.expert_stride + std::uint64_t{o0} * rb, slab / 16U);
    for (std::uint32_t i = threadIdx.x; i < 256U; i += kThreads) grid[i] = c_grid_v2[i];
    const unsigned char* bg = wg + warp * rb;
    const unsigned char* bu = wu + warp * rb;

    for (std::uint32_t c0 = b0; c0 < b1; c0 += R) {
        const std::uint32_t n = min(static_cast<std::uint32_t>(R), b1 - c0);
        if (c0 != b0) __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < n * J * 256U; i += kThreads) {
            const std::uint32_t l = i & 31U, q = (i >> 5U) & 7U, jj = (i >> 8U) % J, r = (i >> 8U) / J, g = l + 32U * jj;
            std::int32_t w = 0;
            if (g < G) w = reinterpret_cast<const std::int32_t*>(p.x_codes + std::uint64_t{p.rows[c0 + r]} * p.hidden)[g * 8U + q];
            xs[i] = w;
        }
        for (std::uint32_t i = threadIdx.x; i < n * G; i += kThreads)
            xd[i] = p.x_deltas[std::uint64_t{p.rows[c0 + i / G]} * G + i % G];
        if (c0 == b0) cp_async_wait_all();
        __syncthreads();

        float acc[2][R];
#pragma unroll
        for (int r = 0; r < R; ++r) { acc[0][r] = 0.0F; acc[1][r] = 0.0F; }
        const int mode = c_v2_mode;
        for (std::uint32_t jj = 0; jj < (mode == 1 ? 0U : J); ++jj) {
            const std::uint32_t g = lane + 32U * jj;
            if (g >= G) break;
            const std::uint32_t blk = g >> 3U, s = g & 7U;
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const unsigned char* b = (m ? bu : bg) + blk * 66U;
                const std::uint32_t d = b[0] | (std::uint32_t{b[1]} << 8U);
                const unsigned char* q = b + 2U + 8U * s;
                const std::uint32_t aux0 = q[0] | (std::uint32_t{q[1]} << 8U) | (std::uint32_t{q[2]} << 16U) | (std::uint32_t{q[3]} << 24U);
                const std::uint32_t aux1 = q[4] | (std::uint32_t{q[5]} << 8U) | (std::uint32_t{q[6]} << 16U) | (std::uint32_t{q[7]} << 24U);
                const float scale = (half_bits(d) * (0.5F + static_cast<float>(aux1 >> 28U))) * 0.25F;
                std::uint32_t wv[8];
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const std::uint64_t gr = mode == 2 ? 0x0819082b19080819ULL + aux0 : grid[(aux0 >> (8 * l)) & 0xffU];
                    const std::uint32_t sb = sign_byte(aux1 >> (7 * l));
                    wv[2 * l] = apply_signs(static_cast<std::uint32_t>(gr), sign_mask4(sb & 15U));
                    wv[2 * l + 1] = apply_signs(static_cast<std::uint32_t>(gr >> 32U), sign_mask4(sb >> 4U));
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (r < static_cast<int>(n)) {
                        const std::int32_t* xr = xs + ((r * J + jj) * 8U) * 32U + lane;
                        int dot = 0;
#pragma unroll
                        for (int w = 0; w < 8; ++w) dot = __dp4a(static_cast<int>(wv[w]), xr[w * 32], dot);
                        const float term = static_cast<float>(dot) * (scale * xd[r * G + g]);
                        acc[m][r] = acc[m][r] + term;
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r < static_cast<int>(n)) {
                const float gv = butterfly(acc[0][r]), uv = butterfly(acc[1][r]);
                if (lane == 0U) {
                    const std::uint64_t at = std::uint64_t{p.slots[c0 + r]} * p.inter + o0 + warp;
                    p.y[at] = swiglu_clamped(gv, uv, p.limit);
                    if (p.gate_out) { p.gate_out[at] = gv; p.up_out[at] = uv; }
                }
            }
        }
    }
}

struct DownV2Params {
    const std::uint8_t* down;
    std::uint64_t expert_stride, row_bytes;
    std::uint32_t hidden, inter;
    const std::uint32_t *experts, *begins, *slots, *counts;
    const float* y;  // [slot][inter]
    float* raw;
    A8Rows act;      // optional: the A8 of y (block x == 0 writes it), for taps
};

// Block = (32 outputs, union expert j); warp w owns outputs o0 + 4 w .. + 3. Prologue: the A8 of the staged entries'
// SwiGLU outputs (a8_group per 32 values, as the v1 gate/up epilogue) and their 16-element half sums.
template <int R>
__global__ __launch_bounds__(kThreads) void down_v2_kernel(DownV2Params p) {
    const std::uint32_t j = blockIdx.y;
    if (j >= p.counts[0]) return;
    extern __shared__ __align__(16) unsigned char smem[];
    const std::uint32_t G = p.inter / 32U, J = (G + 31U) / 32U, rb = static_cast<std::uint32_t>(p.row_bytes);
    const std::uint32_t slab = kDownOutputs * rb;
    unsigned char* wd = smem;                                              // [32][row]
    auto* xs = reinterpret_cast<std::int32_t*>(smem + ((slab + 15U) & ~15U));  // [R][J][8][32]
    auto* xd = reinterpret_cast<float*>(xs + R * J * 256U);                // [R][G]
    auto* hs = reinterpret_cast<std::int32_t*>(xd + R * G);                // [R][2G]
    auto* cb = reinterpret_cast<std::int8_t*>(hs + R * 2U * G);            // [R][inter] codes

    const std::uint32_t lane = threadIdx.x & 31U, warp = threadIdx.x / 32U, o0 = blockIdx.x * kDownOutputs;
    const std::uint64_t expert = p.experts[j];
    const std::uint32_t b0 = p.begins[j], b1 = p.begins[j + 1U];
    slab_async(wd, p.down + expert * p.expert_stride + std::uint64_t{o0} * rb, slab / 16U);

    for (std::uint32_t c0 = b0; c0 < b1; c0 += R) {
        const std::uint32_t n = min(static_cast<std::uint32_t>(R), b1 - c0);
        if (c0 != b0) __syncthreads();
        // A8 of each entry's activation: warp per 32-value group
        for (std::uint32_t t = warp; t < n * G; t += kWarps) {
            const std::uint32_t r = t / G, g = t % G;
            const std::uint64_t slot = p.slots[c0 + r];
            const A8Group q = a8_group(p.y[slot * p.inter + 32U * g + lane]);
            cb[r * p.inter + 32U * g + lane] = q.code;
            if (lane == 0U) xd[r * G + g] = q.delta;
            if (p.act.codes && blockIdx.x == 0U) {
                p.act.codes[slot * p.inter + 32U * g + lane] = q.code;
                if (lane == 0U) {
                    p.act.deltas[slot * G + g] = q.delta;
                    if (p.act.sums) p.act.sums[slot * G + g] = q.sum;
                }
            }
        }
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < n * J * 256U; i += kThreads) {
            const std::uint32_t l = i & 31U, q = (i >> 5U) & 7U, jj = (i >> 8U) % J, r = (i >> 8U) / J, g = l + 32U * jj;
            xs[i] = g < G ? reinterpret_cast<const std::int32_t*>(cb + r * p.inter)[g * 8U + q] : 0;
        }
        for (std::uint32_t i = threadIdx.x; i < n * 2U * G; i += kThreads) {
            const std::uint32_t r = i / (2U * G), h = i % (2U * G);
            const std::int8_t* x = cb + r * p.inter + h * 16U;
            std::int32_t s = 0;
            for (int e = 0; e < 16; ++e) s += x[e];
            hs[i] = s;
        }
        if (c0 == b0) cp_async_wait_all();
        __syncthreads();

        for (std::uint32_t i = 0; i < kDownPerWarp; ++i) {
            const unsigned char* bytes = wd + (warp * kDownPerWarp + i) * rb;
            float acc[R];
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = 0.0F;
            for (std::uint32_t jj = 0; jj < (c_v2_mode == 1 ? 0U : J); ++jj) {
                const std::uint32_t g = lane + 32U * jj;
                if (g >= G) break;
                const std::uint32_t blk = g >> 3U, s = g & 7U, half = s >> 2U, shift = 2U * (s & 3U);
                const unsigned char* b = bytes + blk * 84U;
                const std::uint32_t dd = *reinterpret_cast<const std::uint32_t*>(b + 80U);
                const float d = half_bits(dd & 0xffffU), dmin = half_bits(dd >> 16U);
                const std::uint32_t sc0 = b[8U * half + 2U * (s & 3U)], sc1 = b[8U * half + 2U * (s & 3U) + 1U];
                const auto* qw = reinterpret_cast<const std::uint32_t*>(b + 16U + 32U * half);
                std::uint32_t wv[8];
#pragma unroll
                for (int w = 0; w < 8; ++w) wv[w] = (qw[w] >> shift) & 0x03030303U;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (r < static_cast<int>(n)) {
                        const std::int32_t* xr = xs + ((r * J + jj) * 8U) * 32U + lane;
                        int qx0 = 0, qx1 = 0;
#pragma unroll
                        for (int w = 0; w < 4; ++w) qx0 = __dp4a(static_cast<int>(wv[w]), xr[w * 32], qx0);
#pragma unroll
                        for (int w = 4; w < 8; ++w) qx1 = __dp4a(static_cast<int>(wv[w]), xr[w * 32], qx1);
                        const std::int32_t a = static_cast<std::int32_t>(sc0 & 15U) * qx0 + static_cast<std::int32_t>(sc1 & 15U) * qx1;
                        const std::int32_t bsum = static_cast<std::int32_t>(sc0 >> 4U) * hs[r * 2U * G + 2U * g] +
                                                  static_cast<std::int32_t>(sc1 >> 4U) * hs[r * 2U * G + 2U * g + 1U];
                        const float scaled = d * static_cast<float>(a);
                        const float offset = dmin * static_cast<float>(bsum);
                        const float term = (scaled - offset) * xd[r * G + g];
                        acc[r] = acc[r] + term;
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (r < static_cast<int>(n)) {
                    const float v = butterfly(acc[r]);
                    if (lane == 0U) p.raw[std::uint64_t{p.slots[c0 + r]} * p.hidden + o0 + warp * kDownPerWarp + i] = v;
                }
            }
        }
    }
}

template <class Kernel>
void set_smem(Kernel kernel, std::size_t bytes) {
    if (bytes > 48U * 1024U)
        STRIX_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
}
std::uint32_t pick_rows(std::uint32_t max_rows) { return max_rows <= 1U ? 1U : max_rows <= 2U ? 2U : max_rows <= 4U ? 4U : 8U; }

void check_v2(const MoeShape& s) {
    if (s.hidden % 256U || s.inter % 256U || s.hidden % kDownOutputs || s.inter % kGuOutputs || (kGuOutputs * s.gate_row_bytes()) % 16U ||
        (kDownOutputs * s.down_row_bytes()) % 16U || s.gate_expert_bytes() % 16U || s.down_expert_bytes() % 16U || s.inter > 4096U ||
        s.top_k == 0U || s.top_k > 16U)
        throw std::invalid_argument("ds41 cuda moe v2: unsupported expert shape");
}

}  // namespace

// ================================================================================================== v3: persistent
// Persistent blocks (one per SM by default) walk work items with a stride of the grid; every item's weight slab is
// fetched by cp.async into the other of two buffers while the current item computes. Shapes fixed at compile time
// (hidden 5120, inter 2304): gate/up items of T outputs (T warps, warp w = output w: its gate and up rows), down items
// of 32 outputs (16 warps x 2). Inner loops read the weights as aligned 32-bit shared words with funnel shifts, the
// IQ2_XXS grid from a 4-bit-code table expanded by byte_perm, and each activation word once for every row that uses
// it. The A8 of the SwiGLU outputs is made by the last block that finishes a (slot, 32-output group): the v1 epilogue's
// a8_group on the same 32 values.
namespace v3 {

constexpr std::uint32_t kH = 5120, kI = 2304, kGu = kH / 32U, kJu = 5, kGd = kI / 32U, kJd = 3;
constexpr std::uint32_t kRowU = kH / 256U * 66U, kRowD = kI / 256U * 84U;  // 1320, 756
constexpr std::uint32_t kMag = 0x002b1908U;                                // IQ2_XXS magnitudes 8, 25, 43 by code
constexpr std::uint32_t kDownTile = 32;                                    // outputs per down item (16 warps x 2)

__device__ __forceinline__ std::uint32_t lds32(const unsigned char* p) { return *reinterpret_cast<const std::uint32_t*>(p); }
// FP16 bits to FP32: half_bits (the hardware conversion and a NaN-payload select, no branches)
__device__ __forceinline__ float half_fast(std::uint32_t bits) { return half_bits(bits); }
__device__ __forceinline__ void wait_group(bool keep_one) {
    if (keep_one) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

struct GateUpArgs {
    const std::uint8_t* gate;
    const std::uint8_t* up;
    std::uint64_t expert_stride;
    const std::uint32_t *experts, *begins, *rows, *slots, *counts;
    const std::int8_t* x_codes;
    const float* x_deltas;
    std::uint32_t x_rows;
    const std::uint32_t* ntab;  // [256] 4-bit codes of the grid entries
    const std::uint64_t* stab;  // [128] byte masks (0xff = negative) of the 7-bit sign indices (iq2_xxs_sign_byte)
    float limit;
    float* y;                   // [slot][kI]
    float* gate_out;
    float* up_out;
    A8Rows act;                 // [slot][kI] (codes, deltas, sums)
    std::uint32_t* counters;    // [slot][kI / 32], zero between launches
};

// x rows staged once per block: word q of group g = l + 32 jj of row r at xs[((r kJu + jj) 8 + q) 32 + l]
__device__ __forceinline__ void stage_x(std::int32_t* xs, float* xd, const std::int8_t* codes, const float* deltas, std::uint32_t rows) {
    const std::uint32_t n16 = rows * kH / 16U;
    for (std::uint32_t t = threadIdx.x; t < n16; t += blockDim.x) {
        const int4 v = reinterpret_cast<const int4*>(codes)[t];
        const std::uint32_t r = t / (kH / 16U), u = t % (kH / 16U), g = u / 2U, q0 = 4U * (u & 1U), l = g & 31U, jj = g / 32U;
        std::int32_t* d = xs + ((r * kJu + jj) * 8U + q0) * 32U + l;
        d[0] = v.x; d[32] = v.y; d[64] = v.z; d[96] = v.w;
    }
    for (std::uint32_t t = threadIdx.x; t < rows * kGu; t += blockDim.x) xd[t] = deltas[t];
}

template <int R, int T>
__global__ __launch_bounds__(T * 32) void gate_up_v3_kernel(GateUpArgs a) {
    // every slab is followed by 16 never-written bytes: the last group's speculative third word (w2, unused when the
    // group is word aligned) reads them instead of the other matrix / buffer, which a cp.async may be filling
    constexpr std::uint32_t kSlab = T * kRowU, kSlabP = kSlab + 16U, kTiles = kI / T, kNeed = 32U / T;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* buf = smem;                                                     // [2][gate | up][T][kRowU]
    auto* xs = reinterpret_cast<std::int32_t*>(smem + 4U * kSlabP);                // [R][kJu][8][32]
    auto* xd = reinterpret_cast<float*>(xs + R * kJu * 256U);                      // [R][kGu]
    auto* ntab = reinterpret_cast<std::uint32_t*>(xd + R * kGu);                   // [256]
    auto* stab = reinterpret_cast<std::uint64_t*>(ntab + 256);                     // [128]
    __shared__ std::uint32_t fin[8];
    const std::uint32_t total = kTiles * a.counts[0];
    std::uint32_t it = blockIdx.x;
    if (it >= total) return;
    const std::uint32_t lane = threadIdx.x & 31U, warp = threadIdx.x / 32U;
    const auto issue = [&](std::uint32_t item, std::uint32_t b) {
        const std::uint32_t j = item / kTiles, o0 = (item % kTiles) * T;
        const std::uint64_t off = std::uint64_t{a.experts[j]} * a.expert_stride + std::uint64_t{o0} * kRowU;
        unsigned char* d = buf + b * 2U * kSlabP;
        for (std::uint32_t i = threadIdx.x; i < kSlab / 16U; i += T * 32U) {
            cp_async16(d + 16U * i, a.gate + off + 16U * i);
            cp_async16(d + kSlabP + 16U * i, a.up + off + 16U * i);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
    issue(it, 0);
    stage_x(xs, xd, a.x_codes, a.x_deltas, a.x_rows);
    for (std::uint32_t i = threadIdx.x; i < 256U; i += T * 32U) ntab[i] = a.ntab[i];
    for (std::uint32_t i = threadIdx.x; i < 128U; i += T * 32U) stab[i] = a.stab[i];
    unsigned long long* const stamp = c_v3_stamps ? c_v3_stamps + std::size_t{blockIdx.x} * 32U : nullptr;
    for (std::uint32_t k = 0; it < total; ++k) {
        if (stamp && threadIdx.x == 0 && k < 8U) stamp[k * 4U] = clock64();
        const std::uint32_t nxt = it + gridDim.x;
        if (nxt < total) issue(nxt, (k + 1U) & 1U);
        wait_group(nxt < total);
        __syncthreads();
        if (stamp && threadIdx.x == 0 && k < 8U) stamp[k * 4U + 1U] = clock64();
        const std::uint32_t j = it / kTiles, o0 = (it % kTiles) * T, o = o0 + warp;
        const std::uint32_t b0 = a.begins[j], b1 = a.begins[j + 1U];
        const unsigned char* rg = buf + (k & 1U) * 2U * kSlabP + warp * kRowU;
        const unsigned char* ru = rg + kSlabP;
        for (std::uint32_t c0 = b0; c0 < b1; c0 += R) {
            const std::uint32_t n = min(static_cast<std::uint32_t>(R), b1 - c0);
            std::uint32_t xrow[R];
#pragma unroll
            for (int r = 0; r < R; ++r) xrow[r] = r < static_cast<int>(n) ? a.rows[c0 + r] : 0U;
            float acc[2][R];
#pragma unroll
            for (int r = 0; r < R; ++r) { acc[0][r] = 0.0F; acc[1][r] = 0.0F; }
            const int mode = c_v2_mode;
#pragma unroll
            for (std::uint32_t jj = 0; jj < kJu; ++jj) {
                if (mode == 1) break;
                const std::uint32_t g = lane + 32U * jj, blk = g >> 3U, s = g & 7U;
                std::uint32_t wv[2][8];
                float scale[2];
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                    const unsigned char* row = m ? ru : rg;
                    const std::uint32_t off = blk * 66U, dw = lds32(row + (off & ~3U));
                    const std::uint32_t d = (off & 3U) ? dw >> 16U : dw & 0xffffU;
                    const std::uint32_t at = off + 2U + 8U * s, base = at & ~3U, sh = (at & 3U) * 8U;
                    const std::uint32_t w0 = lds32(row + base), w1 = lds32(row + base + 4U), w2 = lds32(row + base + 8U);
                    const std::uint32_t aux0 = __funnelshift_r(w0, w1, sh), aux1 = __funnelshift_r(w1, w2, sh);
                    scale[m] = (half_fast(d) * (0.5F + static_cast<float>(aux1 >> 28U))) * 0.25F;
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const std::uint32_t e = ntab[(aux0 >> (8 * l)) & 0xffU];
                        const std::uint64_t sm = stab[(aux1 >> (7 * l)) & 127U];
                        wv[m][2 * l] = apply_signs(__byte_perm(kMag, 0U, e & 0xffffU), static_cast<std::uint32_t>(sm));
                        wv[m][2 * l + 1] = apply_signs(__byte_perm(kMag, 0U, e >> 16U), static_cast<std::uint32_t>(sm >> 32U));
                    }
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (r < static_cast<int>(n)) {
                        const std::int32_t* xr = xs + ((xrow[r] * kJu + jj) * 8U) * 32U + lane;
                        int x8[8];
#pragma unroll
                        for (int w = 0; w < 8; ++w) x8[w] = xr[w * 32];
                        const float dx = xd[xrow[r] * kGu + g];
#pragma unroll
                        for (int m = 0; m < 2; ++m) {
                            int dot = 0;
#pragma unroll
                            for (int w = 0; w < 8; ++w) dot = __dp4a(static_cast<int>(wv[m][w]), x8[w], dot);
                            const float term = static_cast<float>(dot) * (scale[m] * dx);
                            acc[m][r] = acc[m][r] + term;
                        }
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (r < static_cast<int>(n)) {
                    const float gv = butterfly(acc[0][r]), uv = butterfly(acc[1][r]);
                    if (lane == 0U) {
                        const std::uint64_t at = std::uint64_t{a.slots[c0 + r]} * kI + o;
                        a.y[at] = swiglu_clamped(gv, uv, a.limit);
                        if (a.gate_out) { a.gate_out[at] = gv; a.up_out[at] = uv; }
                    }
                }
            }
            // the last of the kNeed items covering a (slot, 32-output group) makes its A8
            if (stamp && threadIdx.x == 0 && k < 8U) stamp[k * 4U + 2U] = clock64();
            if (mode == 3) continue;
            if (lane == 0U) __threadfence();  // the lanes that stored y publish them before the arrival
            __syncthreads();
            if (threadIdx.x < n) {
                const std::uint32_t slot = a.slots[c0 + threadIdx.x], c = slot * (kI / 32U) + o0 / 32U;
                const std::uint32_t old = atomicAdd(a.counters + c, 1U);
                fin[threadIdx.x] = old + 1U == kNeed ? 1U : 0U;
                if (old + 1U == kNeed) a.counters[c] = 0U;
            }
            __syncthreads();
            if (warp < n && fin[warp]) {
                __threadfence();
                const std::uint64_t slot = a.slots[c0 + warp], grp = o0 / 32U;
                const A8Group q = a8_group(__ldcg(a.y + slot * kI + grp * 32U + lane));
                a.act.codes[slot * kI + grp * 32U + lane] = q.code;
                if (lane == 0U) {
                    a.act.deltas[slot * (kI / 32U) + grp] = q.delta;
                    if (a.act.sums) a.act.sums[slot * (kI / 32U) + grp] = q.sum;
                }
            }
        }
        __syncthreads();
        if (stamp && threadIdx.x == 0 && k < 8U) stamp[k * 4U + 3U] = clock64();
        it = nxt;
    }
}

struct DownArgs {
    const std::uint8_t* down;
    std::uint64_t expert_stride;
    const std::uint32_t *experts, *begins, *slots, *counts;
    const std::int8_t* act_codes;
    const float* act_deltas;
    std::uint32_t n_slots;      // slots of the window (rows * top_k), staged once when kAll
    float* raw;
};

// act of `count` slots (ids from slot_of(i)) into xs [i][kJd][8][32], xd [i][kGd], hs [i][2 kGd]
template <class SlotOf>
__device__ __forceinline__ void stage_act(std::int32_t* xs, float* xd, std::int32_t* hs, const DownArgs& a, std::uint32_t count, SlotOf slot_of) {
    for (std::uint32_t t = threadIdx.x; t < count * (kI / 16U); t += blockDim.x) {
        const std::uint32_t i = t / (kI / 16U), u = t % (kI / 16U), g = u / 2U, q0 = 4U * (u & 1U), l = g & 31U, jj = g / 32U;
        const int4 v = reinterpret_cast<const int4*>(a.act_codes + std::uint64_t{slot_of(i)} * kI)[u];
        std::int32_t* d = xs + ((i * kJd + jj) * 8U + q0) * 32U + l;
        d[0] = v.x; d[32] = v.y; d[64] = v.z; d[96] = v.w;
        const std::uint32_t words[4] = {static_cast<std::uint32_t>(v.x), static_cast<std::uint32_t>(v.y), static_cast<std::uint32_t>(v.z),
                                        static_cast<std::uint32_t>(v.w)};
        std::int32_t sum = 0;
#pragma unroll
        for (int w = 0; w < 4; ++w)
#pragma unroll
            for (int b = 0; b < 4; ++b) sum += static_cast<std::int8_t>(words[w] >> (8 * b));
        hs[i * 2U * kGd + u] = sum;
    }
    for (std::uint32_t t = threadIdx.x; t < count * kGd; t += blockDim.x)
        xd[t] = a.act_deltas[std::uint64_t{slot_of(t / kGd)} * kGd + t % kGd];
}

template <int R, bool kAll>
__global__ __launch_bounds__(512) void down_v3_kernel(DownArgs a) {
    constexpr std::uint32_t kSlab = kDownTile * kRowD, kTiles = kH / kDownTile, kCap = kAll ? 6U * R : R;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* buf = smem;                                              // [2][32][kRowD]
    auto* xs = reinterpret_cast<std::int32_t*>(smem + 2U * kSlab);         // [kCap][kJd][8][32]
    auto* xd = reinterpret_cast<float*>(xs + kCap * kJd * 256U);           // [kCap][kGd]
    auto* hs = reinterpret_cast<std::int32_t*>(xd + kCap * kGd);           // [kCap][2 kGd]
    const std::uint32_t total = kTiles * a.counts[0];
    std::uint32_t it = blockIdx.x;
    if (it >= total) return;
    const std::uint32_t lane = threadIdx.x & 31U, warp = threadIdx.x / 32U;
    const auto issue = [&](std::uint32_t item, std::uint32_t b) {
        const std::uint32_t j = item / kTiles, o0 = (item % kTiles) * kDownTile;
        const unsigned char* src = a.down + std::uint64_t{a.experts[j]} * a.expert_stride + std::uint64_t{o0} * kRowD;
        unsigned char* d = buf + b * kSlab;
        for (std::uint32_t i = threadIdx.x; i < kSlab / 16U; i += 512U) cp_async16(d + 16U * i, src + 16U * i);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
    issue(it, 0);
    if constexpr (kAll) stage_act(xs, xd, hs, a, a.n_slots, [](std::uint32_t i) { return i; });
    for (std::uint32_t k = 0; it < total; ++k) {
        const std::uint32_t nxt = it + gridDim.x;
        const std::uint32_t j = it / kTiles, o0 = (it % kTiles) * kDownTile;
        const std::uint32_t b0 = a.begins[j], b1 = a.begins[j + 1U];
        for (std::uint32_t c0 = b0; c0 < b1; c0 += R) {
            const std::uint32_t n = min(static_cast<std::uint32_t>(R), b1 - c0);
            if constexpr (!kAll) {
                if (c0 != b0) __syncthreads();
                stage_act(xs, xd, hs, a, n, [&](std::uint32_t i) { return a.slots[c0 + i]; });
            }
            if (c0 == b0) {
                if (nxt < total) issue(nxt, (k + 1U) & 1U);
                wait_group(nxt < total);
            }
            __syncthreads();
            std::uint32_t xi[R], slot[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                slot[r] = r < static_cast<int>(n) ? a.slots[c0 + r] : 0U;
                xi[r] = kAll ? slot[r] : static_cast<std::uint32_t>(r);
            }
            const unsigned char* rows = buf + (k & 1U) * kSlab + warp * 2U * kRowD;
            float acc[2][R];
#pragma unroll
            for (int r = 0; r < R; ++r) { acc[0][r] = 0.0F; acc[1][r] = 0.0F; }
#pragma unroll
            for (std::uint32_t jj = 0; jj < kJd; ++jj) {
                const std::uint32_t g = lane + 32U * jj;
                if (g >= kGd) break;
                const std::uint32_t blk = g >> 3U, s = g & 7U, half = s >> 2U, shift = 2U * (s & 3U);
                std::uint32_t wv[2][8], scw[2];
                float d[2], dmin[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const unsigned char* b = rows + i * kRowD + blk * 84U;
                    const std::uint32_t dd = lds32(b + 80U);
                    d[i] = half_bits(dd & 0xffffU);
                    dmin[i] = half_bits(dd >> 16U);
                    const std::uint32_t so = 8U * half + 2U * (s & 3U), sw = lds32(b + (so & ~3U));
                    scw[i] = (so & 2U) ? sw >> 16U : sw & 0xffffU;
#pragma unroll
                    for (int w = 0; w < 8; ++w) wv[i][w] = (lds32(b + 16U + 32U * half + 4U * w) >> shift) & 0x03030303U;
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (r < static_cast<int>(n)) {
                        const std::int32_t* xr = xs + ((xi[r] * kJd + jj) * 8U) * 32U + lane;
                        int x8[8];
#pragma unroll
                        for (int w = 0; w < 8; ++w) x8[w] = xr[w * 32];
                        const std::int32_t h0 = hs[xi[r] * 2U * kGd + 2U * g], h1 = hs[xi[r] * 2U * kGd + 2U * g + 1U];
                        const float dx = xd[xi[r] * kGd + g];
#pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            const std::uint32_t sc0 = scw[i] & 0xffU, sc1 = scw[i] >> 8U;
                            int qx0 = 0, qx1 = 0;
#pragma unroll
                            for (int w = 0; w < 4; ++w) qx0 = __dp4a(static_cast<int>(wv[i][w]), x8[w], qx0);
#pragma unroll
                            for (int w = 4; w < 8; ++w) qx1 = __dp4a(static_cast<int>(wv[i][w]), x8[w], qx1);
                            const std::int32_t av = static_cast<std::int32_t>(sc0 & 15U) * qx0 + static_cast<std::int32_t>(sc1 & 15U) * qx1;
                            const std::int32_t bsum = static_cast<std::int32_t>(sc0 >> 4U) * h0 + static_cast<std::int32_t>(sc1 >> 4U) * h1;
                            const float scaled = d[i] * static_cast<float>(av);
                            const float offset = dmin[i] * static_cast<float>(bsum);
                            const float term = (scaled - offset) * dx;
                            acc[i][r] = acc[i][r] + term;
                        }
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (r < static_cast<int>(n)) {
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const float v = butterfly(acc[i][r]);
                        if (lane == 0U) a.raw[std::uint64_t{slot[r]} * kH + o0 + warp * 2U + i] = v;
                    }
                }
            }
        }
        __syncthreads();
        it = nxt;
    }
}

}  // namespace v3

// ================================================================================================== v1b
namespace v3 {

struct GateUpV1bArgs {
    const std::uint8_t* gate;
    const std::uint8_t* up;
    std::uint64_t expert_stride;
    const std::uint32_t *experts, *begins, *rows, *slots, *counts;
    const std::int8_t* x_codes;
    const float* x_deltas;
    const std::uint32_t* ntab;
    const std::uint64_t* stab;
    float limit;
    A8Rows act;
    float* gate_out;
    float* up_out;
};

// R = 1: at most 85 registers so 3 blocks share an SM (the R1 grid is 72 x 6 = 432 blocks; with 2 per SM it ran as
// two waves on 188 SMs, the second 56 blocks long). Register allocation only: the arithmetic is unchanged.
template <int R, int MinBlocks = (R == 1 ? 3 : 1)>
__global__ __launch_bounds__(256, MinBlocks) void gate_up_v1b_kernel(GateUpV1bArgs p) {
    const std::uint32_t j = blockIdx.y;
    if (j >= p.counts[0]) return;
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr std::uint32_t kWords = kRowU / 8U;  // 165 u64 per row
    auto* xs = reinterpret_cast<std::int32_t*>(smem);                          // [R][kJu][8][32]
    auto* xd = reinterpret_cast<float*>(xs + R * kJu * 256U);                  // [R][kGu]
    // [8 warps][2 kWords + 1]: the spare word after each warp's up row takes the last group's speculative tail read
    auto* rowbuf = reinterpret_cast<std::uint64_t*>(xd + R * kGu);
    auto* gu = reinterpret_cast<float*>(rowbuf + 8U * (2U * kWords + 1U));     // [R][32][2]
    auto* ntab = reinterpret_cast<std::uint32_t*>(gu + R * 64U);               // [256]
    auto* stab = reinterpret_cast<std::uint64_t*>(ntab + 256);                 // [128]
    for (std::uint32_t i = threadIdx.x; i < 256U; i += 256U) ntab[i] = p.ntab[i];
    for (std::uint32_t i = threadIdx.x; i < 128U; i += 256U) stab[i] = p.stab[i];

    const std::uint32_t lane = threadIdx.x & 31U, warp = threadIdx.x / 32U;
    const std::uint64_t expert = p.experts[j];
    const std::uint32_t b0 = p.begins[j], b1 = p.begins[j + 1U];
    const std::uint8_t* gate_base = p.gate + expert * p.expert_stride;
    const std::uint8_t* up_base = p.up + expert * p.expert_stride;
    std::uint64_t* buf = rowbuf + warp * (2U * kWords + 1U);
    const auto* rg = reinterpret_cast<const unsigned char*>(buf);
    const auto* ru = reinterpret_cast<const unsigned char*>(buf + kWords);

    for (std::uint32_t c0 = b0; c0 < b1; c0 += R) {
        const std::uint32_t n = min(static_cast<std::uint32_t>(R), b1 - c0);
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < n * kJu * 256U; i += 256U) {
            const std::uint32_t l = i & 31U, q = (i >> 5U) & 7U, jj = (i >> 8U) % kJu, r = (i >> 8U) / kJu, g = l + 32U * jj;
            xs[i] = reinterpret_cast<const std::int32_t*>(p.x_codes + std::uint64_t{p.rows[c0 + r]} * kH)[g * 8U + q];
        }
        for (std::uint32_t i = threadIdx.x; i < n * kGu; i += 256U) xd[i] = p.x_deltas[std::uint64_t{p.rows[c0 + i / kGu]} * kGu + i % kGu];
        __syncthreads();

        const std::uint32_t o0 = blockIdx.x * 32U + warp * 4U;
        uint2 pre[2][6];
        auto load = [&](std::uint32_t o) {
            const auto* g = reinterpret_cast<const uint2*>(gate_base + std::uint64_t{o} * kRowU);
            const auto* u = reinterpret_cast<const uint2*>(up_base + std::uint64_t{o} * kRowU);
#pragma unroll
            for (std::uint32_t k = 0; k < 6U; ++k) {
                const std::uint32_t w = lane + 32U * k;
                if (w < kWords) { pre[0][k] = __ldg(g + w); pre[1][k] = __ldg(u + w); }
            }
        };
        load(o0);
        for (std::uint32_t i = 0; i < 4U; ++i) {
#pragma unroll
            for (std::uint32_t k = 0; k < 6U; ++k) {
                const std::uint32_t w = lane + 32U * k;
                if (w < kWords) {
                    buf[w] = (std::uint64_t{pre[0][k].y} << 32U) | pre[0][k].x;
                    buf[kWords + w] = (std::uint64_t{pre[1][k].y} << 32U) | pre[1][k].x;
                }
            }
            __syncwarp();
            if (i + 1U < 4U) load(o0 + i + 1U);
            float acc[2][R];
#pragma unroll
            for (int r = 0; r < R; ++r) { acc[0][r] = 0.0F; acc[1][r] = 0.0F; }
#pragma unroll
            for (std::uint32_t jj = 0; jj < kJu; ++jj) {
                const std::uint32_t g = lane + 32U * jj, blk = g >> 3U, s = g & 7U;
                std::uint32_t wv[2][8];
                float scale[2];
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                    const unsigned char* row = m ? ru : rg;
                    const std::uint32_t off = blk * 66U, dw = lds32(row + (off & ~3U));
                    const std::uint32_t d = (off & 3U) ? dw >> 16U : dw & 0xffffU;
                    const std::uint32_t at = off + 2U + 8U * s, base = at & ~3U, sh = (at & 3U) * 8U;
                    const std::uint32_t w0 = lds32(row + base), w1 = lds32(row + base + 4U), w2 = lds32(row + base + 8U);
                    const std::uint32_t aux0 = __funnelshift_r(w0, w1, sh), aux1 = __funnelshift_r(w1, w2, sh);
                    scale[m] = (half_fast(d) * (0.5F + static_cast<float>(aux1 >> 28U))) * 0.25F;
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const std::uint32_t e = ntab[(aux0 >> (8 * l)) & 0xffU];
                        const std::uint64_t sm = stab[(aux1 >> (7 * l)) & 127U];
                        wv[m][2 * l] = apply_signs(__byte_perm(kMag, 0U, e & 0xffffU), static_cast<std::uint32_t>(sm));
                        wv[m][2 * l + 1] = apply_signs(__byte_perm(kMag, 0U, e >> 16U), static_cast<std::uint32_t>(sm >> 32U));
                    }
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (r < static_cast<int>(n)) {
                        const std::int32_t* xr = xs + ((r * kJu + jj) * 8U) * 32U + lane;
                        int x8[8];
#pragma unroll
                        for (int w = 0; w < 8; ++w) x8[w] = xr[w * 32];
                        const float dx = xd[r * kGu + g];
#pragma unroll
                        for (int m = 0; m < 2; ++m) {
                            int dot = 0;
#pragma unroll
                            for (int w = 0; w < 8; ++w) dot = __dp4a(static_cast<int>(wv[m][w]), x8[w], dot);
                            const float term = static_cast<float>(dot) * (scale[m] * dx);
                            acc[m][r] = acc[m][r] + term;
                        }
                    }
                }
            }
            __syncwarp();
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (r < static_cast<int>(n)) {
                    const float gv = butterfly(acc[0][r]), uv = butterfly(acc[1][r]);
                    if (lane == 0U) {
                        gu[(r * 32U + warp * 4U + i) * 2U] = gv;
                        gu[(r * 32U + warp * 4U + i) * 2U + 1U] = uv;
                    }
                }
            }
        }
        __syncthreads();
        // epilogue: warp r quantizes entry r's 32 activations (v1's)
        for (std::uint32_t r = warp; r < n; r += 8U) {
            const float gv = gu[(r * 32U + lane) * 2U], uv = gu[(r * 32U + lane) * 2U + 1U];
            const std::uint64_t slot = p.slots[c0 + r], o = blockIdx.x * 32U + lane;
            if (p.gate_out) { p.gate_out[slot * kI + o] = gv; p.up_out[slot * kI + o] = uv; }
            const A8Group q = a8_group(swiglu_clamped(gv, uv, p.limit));
            p.act.codes[slot * kI + o] = q.code;
            if (lane == 0U) {
                p.act.deltas[slot * (kI / 32U) + blockIdx.x] = q.delta;
                if (p.act.sums) p.act.sums[slot * (kI / 32U) + blockIdx.x] = q.sum;
            }
        }
    }
}

}  // namespace v3

void set_routed_v3_stamps(unsigned long long* stamps) { STRIX_CUDA_CHECK(cudaMemcpyToSymbol(c_v3_stamps, &stamps, sizeof(stamps))); }
void set_routed_v2_diagnostic(int mode) { STRIX_CUDA_CHECK(cudaMemcpyToSymbol(c_v2_mode, &mode, sizeof(int))); }

std::size_t routed_v2_y_floats(std::uint32_t slots, const MoeShape& shape) noexcept { return std::size_t{slots} * shape.inter; }

void enqueue_routed_gate_up_v2(const RoutedGateUpView& v, float* y, cudaStream_t stream) {
    check_v2(v.shape);
    if (!v.max_union) return;
    if (!g_grid_v2) {
        STRIX_CUDA_CHECK(cudaMemcpyToSymbol(c_grid_v2, iq2_xxs_grid.data(), sizeof(std::uint64_t) * 256U));
        g_grid_v2 = true;
    }
    const std::uint32_t x_stride = v.x.stride ? v.x.stride : v.shape.hidden;
    if (x_stride != v.shape.hidden || !y) throw std::invalid_argument("ds41 cuda moe v2 gate/up: A8 stride must equal hidden, y required");
    const GateUpV2Params p{v.weights.gate, v.weights.up, v.shape.gate_expert_bytes(), v.shape.gate_row_bytes(), v.shape.hidden, v.shape.inter,
                           v.routes.experts, v.routes.begins, v.routes.rows, v.routes.slots, v.routes.counts,
                           v.x.codes, v.x.deltas, v.shape.limit, y, v.gate_out, v.up_out};
    const std::uint32_t G = v.shape.hidden / 32U, J = (G + 31U) / 32U;
    const std::uint32_t R = pick_rows(std::min(v.max_rows, 8U));
    const std::size_t smem = 2U * kGuOutputs * v.shape.gate_row_bytes() + 256U * 8U + std::size_t{R} * J * 256U * 4U + std::size_t{R} * G * 4U;
    const dim3 grid(v.shape.inter / kGuOutputs, v.max_union);
    auto go = [&](auto kernel) {
        set_smem(kernel, smem);
        kernel<<<grid, kThreads, smem, stream>>>(p);
    };
    if (R == 1U) go(gate_up_v2_kernel<1>);
    else if (R == 2U) go(gate_up_v2_kernel<2>);
    else if (R == 4U) go(gate_up_v2_kernel<4>);
    else go(gate_up_v2_kernel<8>);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_routed_down_v2(const RoutedDownView& v, const float* y, cudaStream_t stream) {
    check_v2(v.shape);
    if (!v.max_union) return;
    if (v.raw1 || !y) throw std::invalid_argument("ds41 cuda moe v2 down: no expert TP, y required");
    if (v.act.codes && v.act.stride && v.act.stride != v.shape.inter) throw std::invalid_argument("ds41 cuda moe v2 down: A8 stride must equal inter");
    const DownV2Params p{v.weights.down, v.shape.down_expert_bytes(), v.shape.down_row_bytes(), v.shape.hidden, v.shape.inter,
                         v.routes.experts, v.routes.begins, v.routes.slots, v.routes.counts, y, v.raw, v.act};
    const std::uint32_t G = v.shape.inter / 32U, J = (G + 31U) / 32U;
    const std::uint32_t R = pick_rows(std::min(v.max_rows, 8U));
    const std::size_t smem = ((kDownOutputs * v.shape.down_row_bytes() + 15U) & ~std::size_t{15U}) + std::size_t{R} * J * 256U * 4U +
                             std::size_t{R} * G * 4U + std::size_t{R} * 2U * G * 4U + std::size_t{R} * v.shape.inter;
    const dim3 grid(v.shape.hidden / kDownOutputs, v.max_union);
    auto go = [&](auto kernel) {
        set_smem(kernel, smem);
        kernel<<<grid, kThreads, smem, stream>>>(p);
    };
    if (R == 1U) go(down_v2_kernel<1>);
    else if (R == 2U) go(down_v2_kernel<2>);
    else if (R == 4U) go(down_v2_kernel<4>);
    else go(down_v2_kernel<8>);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

namespace {
DeviceBuffer<std::uint32_t>& v3_ntab() {
    static DeviceBuffer<std::uint32_t> t;
    if (t.size() == 0U) {
        std::vector<std::uint32_t> h(256);
        for (std::size_t i = 0; i < 256U; ++i) {
            std::uint32_t e = 0;
            for (int b = 0; b < 8; ++b) {
                const auto m = static_cast<std::uint32_t>((iq2_xxs_grid[i] >> (8 * b)) & 0xffU);
                const std::uint32_t c = m == 8U ? 0U : m == 25U ? 1U : m == 43U ? 2U : 15U;
                if (c == 15U) throw std::runtime_error("ds41 cuda moe v3: IQ2_XXS grid magnitude outside {8, 25, 43}");
                e |= c << (4 * b);
            }
            h[i] = e;
        }
        t.resize(256);
        t.upload(h);
    }
    return t;
}
DeviceBuffer<std::uint64_t>& v3_stab() {
    static DeviceBuffer<std::uint64_t> t;
    if (t.size() == 0U) {
        std::vector<std::uint64_t> h(128);
        for (std::uint32_t i = 0; i < 128U; ++i) {
            const std::uint32_t sb = iq2_xxs_sign_byte(i);
            std::uint64_t m = 0;
            for (int b = 0; b < 8; ++b)
                if ((sb >> b) & 1U) m |= std::uint64_t{0xffU} << (8 * b);
            h[i] = m;
        }
        t.resize(128);
        t.upload(h);
    }
    return t;
}
int v3_sms() {
    static int sms = 0;
    if (!sms) {
        int dev = 0;
        STRIX_CUDA_CHECK(cudaGetDevice(&dev));
        STRIX_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    }
    return sms;
}
}  // namespace

bool routed_v3_supported(const MoeShape& s, std::uint32_t max_rows) noexcept {
    return s.hidden == v3::kH && s.inter == v3::kI && s.top_k >= 1U && s.top_k <= 6U && max_rows >= 1U && max_rows <= 8U;
}
namespace { bool g_v3_one_item = false; }
void set_routed_v3_one_item(bool on) { g_v3_one_item = on; }
std::size_t routed_v3_counter_words(std::uint32_t slots) noexcept { return std::size_t{slots} * (v3::kI / 32U); }

void enqueue_routed_gate_up_v3(const RoutedGateUpView& v, float* y, std::uint32_t* counters, cudaStream_t stream) {
    if (!routed_v3_supported(v.shape, v.max_rows) || !y || !counters || !v.act.codes || !v.act.deltas || v.x.rows == 0U || v.x.rows > v.max_rows ||
        (v.x.stride && v.x.stride != v.shape.hidden) || (v.act.stride && v.act.stride != v.shape.inter))
        throw std::invalid_argument("ds41 cuda moe v3 gate/up: unsupported view");
    if (!v.max_union) return;
    const v3::GateUpArgs a{v.weights.gate, v.weights.up, v.shape.gate_expert_bytes(), v.routes.experts, v.routes.begins, v.routes.rows,
                           v.routes.slots, v.routes.counts, v.x.codes, v.x.deltas, v.x.rows, v3_ntab().get(), v3_stab().get(), v.shape.limit, y,
                           v.gate_out, v.up_out, v.act, counters};
    const std::uint32_t R = pick_rows(v.max_rows);
    const bool one_item = g_v3_one_item;  // diagnostics: every block one item (ordinary scheduling)
    auto go = [&](auto kernel, std::uint32_t T, std::uint32_t RR) {
        const std::size_t smem = 4U * (T * v3::kRowU + 16U) + std::size_t{RR} * v3::kJu * 256U * 4U + std::size_t{RR} * v3::kGu * 4U + 256U * 4U + 128U * 8U;
        STRIX_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        int per_sm = 0;
        STRIX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, static_cast<int>(T * 32U), smem));
        const std::uint32_t items = v3::kI / T * v.max_union;
        const auto blocks = one_item ? items : std::min<std::uint32_t>(items, static_cast<std::uint32_t>(std::max(per_sm, 1) * v3_sms()));
        kernel<<<blocks, T * 32U, smem, stream>>>(a);
    };
    if (R == 1U && one_item) go(v3::gate_up_v3_kernel<1, 8>, 8U, 1U);
    else if (R == 1U) go(v3::gate_up_v3_kernel<1, 16>, 16U, 1U);
    else if (R == 2U) go(v3::gate_up_v3_kernel<2, 8>, 8U, 2U);
    else if (R == 4U) go(v3::gate_up_v3_kernel<4, 8>, 8U, 4U);
    else go(v3::gate_up_v3_kernel<8, 8>, 8U, 8U);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_routed_down_v3(const RoutedDownView& v, cudaStream_t stream) {
    if (!routed_v3_supported(v.shape, v.max_rows) || v.raw1 || !v.raw || !v.act.codes || !v.act.deltas || (v.act.stride && v.act.stride != v.shape.inter))
        throw std::invalid_argument("ds41 cuda moe v3 down: unsupported view");
    if (!v.max_union) return;
    const std::uint32_t R = pick_rows(v.max_rows);
    const v3::DownArgs a{v.weights.down, v.shape.down_expert_bytes(), v.routes.experts, v.routes.begins, v.routes.slots, v.routes.counts,
                         v.act.codes, v.act.deltas, R * v.shape.top_k, v.raw};
    auto go = [&](auto kernel, std::uint32_t cap) {
        const std::size_t smem = 2U * v3::kDownTile * v3::kRowD + std::size_t{cap} * (v3::kJd * 256U * 4U + v3::kGd * 4U + 2U * v3::kGd * 4U);
        STRIX_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        int per_sm = 0;
        STRIX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, 512, smem));
        const std::uint32_t items = v3::kH / v3::kDownTile * v.max_union;
        const auto blocks = std::min<std::uint32_t>(items, static_cast<std::uint32_t>(std::max(per_sm, 1) * v3_sms()));
        kernel<<<blocks, 512, smem, stream>>>(a);
    };
    if (R == 1U) go(v3::down_v3_kernel<1, true>, 6U);
    else if (R == 2U) go(v3::down_v3_kernel<2, true>, 12U);
    else if (R == 4U) go(v3::down_v3_kernel<4, false>, 4U);
    else go(v3::down_v3_kernel<8, false>, 8U);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

namespace {
int g_v1b_r1_form = 0;  // diagnostics: 0 bound + carveout, 1 bound only, 2 neither
}
void set_v1b_r1_form(int form) { g_v1b_r1_form = form == 1 || form == 2 ? form : 0; }

void enqueue_routed_gate_up_v1b(const RoutedGateUpView& v, cudaStream_t stream) {
    if (!routed_v3_supported(v.shape, v.max_rows) || !v.act.codes || !v.act.deltas || (v.x.stride && v.x.stride != v.shape.hidden) ||
        (v.act.stride && v.act.stride != v.shape.inter))
        throw std::invalid_argument("ds41 cuda moe v1b gate/up: unsupported view");
    if (!v.max_union) return;
    const v3::GateUpV1bArgs a{v.weights.gate, v.weights.up, v.shape.gate_expert_bytes(), v.routes.experts, v.routes.begins, v.routes.rows,
                              v.routes.slots, v.routes.counts, v.x.codes, v.x.deltas, v3_ntab().get(), v3_stab().get(), v.shape.limit, v.act,
                              v.gate_out, v.up_out};
    const std::uint32_t R = pick_rows(v.max_rows);
    auto go = [&](auto kernel) {
        const std::size_t smem = std::size_t{R} * v3::kJu * 256U * 4U + std::size_t{R} * v3::kGu * 4U + 8U * (2U * (v3::kRowU / 8U) + 1U) * 8U +
                                 std::size_t{R} * 64U * 4U + 256U * 4U + 128U * 8U;
        if (smem > 48U * 1024U) STRIX_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        // the largest shared-memory carveout, so the register-bound occupancy (3 blocks at R = 1) is not cut by smem
        if (R != 1U || g_v1b_r1_form == 0)
            STRIX_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
        kernel<<<dim3(v3::kI / 32U, v.max_union), 256, smem, stream>>>(a);
    };
    if (R == 1U && g_v1b_r1_form == 2) go(v3::gate_up_v1b_kernel<1, 1>);
    else if (R == 1U) go(v3::gate_up_v1b_kernel<1>);
    else if (R == 2U) go(v3::gate_up_v1b_kernel<2>);
    else if (R == 4U) go(v3::gate_up_v1b_kernel<4>);
    else go(v3::gate_up_v1b_kernel<8>);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace strix::ds41::cuda
