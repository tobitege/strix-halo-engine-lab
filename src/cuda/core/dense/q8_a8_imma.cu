// Shared CUDA dense core: Q8_0 x A8 sites on the int8 tensor cores (mma.m16n8k32.s8), bitwise product_q8_0.
//
// Tiled weights: a Q8_0 tensor is repacked once into 16-row tiles; per (tile, group) 32 lanes x 16 bytes in the
// fragment order of the A operand plus the 16 FP16 scales. One mma gives the exact integer dot of one 32-weight group
// for 16 outputs x 8 activation rows (the k order inside a group is permuted identically in A and B, which leaves the
// integer sum unchanged). The FP32 arithmetic is the declared one: term = float(dot) * (scale * delta); lane class
// c (groups c, c + 32, ...) is summed in ascending group order from 0; the 32 class sums are combined in the
// butterfly's tree (v[l] = v[l] + v[l + d], d = 16, 8, 4, 2, 1; result v[0]).
// Work split: unit u (of U = 8 * S: warp w of block s) owns the classes u, u + U, ... and runs their groups in
// ascending order; the tree levels d >= U are taken inside the unit, the levels d < U across units (in shared memory
// for S = 1, through a scratch buffer and a second kernel for S > 1).

#include "../../arch/capability.hpp"
#include "../ptx.cuh"
// Reuse the existing declared numerical helpers during the first tile extraction;
// common numerics and runtime helpers are the next extraction in the backend plan.
#include "../../models/ds41/ds41_cuda_common.cuh"

#include "strix/cuda/core/dense.hpp"

#include <cstdio>

namespace strix::cuda::core::dense {

namespace detail = ::strix::ds41::cuda::detail;
using ::strix::ds41::f16_to_float;
using ::strix::ds41::swiglu_clamped;
namespace {

using detail::require;

// Contract validation is host bookkeeping only; it never probes, allocates or synchronises a device.
void require_plan(const DensePlan& plan, std::uint32_t count, std::uint32_t width, std::uint32_t rows) {
    require(plan.count == count && plan.width == width && plan.rows == rows,
            "tiled Q8_0 x A8 plan shape");
    require(plan.format == DenseWeightFormat::q8_0_tiled_16x32_a8 &&
                plan.identity == DenseNumericIdentity::product_q8_0_bitwise &&
                plan.required_sm_major == 8 && plan.required_sm_minor == 6,
            "tiled Q8_0 x A8 plan identity and capability");
    require(plan.mode == (rows <= 8U ? DenseMode::decode : DenseMode::prefill),
            "tiled Q8_0 x A8 plan mode");
}

// ---- repack -----------------------------------------------------------------------------------------------------------
// Lane L (gid = L / 4, tig = L % 4) of tile t, group g: {W[gid][8 tig .. +3], W[gid + 8][8 tig ..], W[gid][8 tig + 4 ..],
// W[gid + 8][8 tig + 4 ..]} with W[m][k] the code k of row 16 t + m in group g; B is read as {x[8 tig .. +3],
// x[8 tig + 4 .. +7]} of the activation group, the same permutation of k.
__global__ void repack_kernel(const std::uint8_t* __restrict__ q8, std::uint32_t tiles, std::uint32_t groups, std::uint32_t* __restrict__ codes,
                              std::uint16_t* __restrict__ scales) {
    const std::size_t idx = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (idx >= std::size_t{tiles} * groups * 32U) return;
    const std::uint32_t lane = static_cast<std::uint32_t>(idx % 32U);
    const std::size_t tg = idx / 32U;
    const std::uint32_t g = static_cast<std::uint32_t>(tg % groups), t = static_cast<std::uint32_t>(tg / groups);
    const std::uint32_t gid = lane / 4U, tig = lane % 4U;
    const std::size_t row_bytes = std::size_t{groups} * 34U;
    auto word = [&](std::uint32_t m, std::uint32_t k0) {
        const std::uint8_t* b = q8 + (std::size_t{t} * 16U + m) * row_bytes + std::size_t{g} * 34U + 2U + k0;
        return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8U) | (static_cast<std::uint32_t>(b[2]) << 16U) |
               (static_cast<std::uint32_t>(b[3]) << 24U);
    };
    std::uint32_t* out = codes + idx * 4U;
    out[0] = word(gid, 8U * tig);
    out[1] = word(gid + 8U, 8U * tig);
    out[2] = word(gid, 8U * tig + 4U);
    out[3] = word(gid + 8U, 8U * tig + 4U);
    if (lane < 16U) {
        const std::uint8_t* b = q8 + (std::size_t{t} * 16U + lane) * row_bytes + std::size_t{g} * 34U;
        scales[tg * 16U + lane] = static_cast<std::uint16_t>(b[0] | (b[1] << 8U));
    }
}

using ::strix::cuda::core::ptx::mma_s8;

struct MmaArgs {
    const uint4* codes;
    const std::uint16_t* scales;
    std::uint32_t groups_total, g_off, groups, tile_first, per_group_tiles, x_group_width;
    const std::int8_t* x;
    const float* deltas;
    std::uint32_t x_stride, rows;
    float* out;
    std::uint32_t out_stride;
    bool accumulate;
    float* scratch;  // S > 1: [tile][U][rows][16]
    std::uint32_t inter = 0;
    float limit = 10.0F;
    A8Rows act;  // optional diagnostics, written by output tile 0 only
    const float* gate_up = nullptr;
    bool pdl = false;  // launched with programmatic serialization: the PDL prologue below, by pdl_mode
    int pdl_mode = 1;  // 0: wait at entry; 1: the block's slab prefetched into L2, then wait; 2: the first step's weights
                       // and scales into registers, wait before the first activation load (not SwigluDown)
    std::uint32_t halves = 1;  // 2: also groups [groups, 2 groups) with x shifted by groups, out = y0 + y1 (S = 1)
    // S = 1: hc_expand in the epilogue (Q8A8View::hc_*)
    const float* hc_residual = nullptr;
    const float* hc_mixes = nullptr;
    float* hc_out = nullptr;
    std::uint32_t hc_hidden = 0;
    int occ = 1;  // 3 = the 1-row-tile split-1 kernel compiled for 3 blocks per SM (host dispatch only)
};

// grid (tiles, S, row passes), 8 warps. T row tiles of 8 per pass. NC = 32 / U classes per unit.
// MB: minimum resident blocks per SM asked of the register allocator (--q8-occ3: MB = 3 for T = 1,
// S = 1, i.e. <= 85 registers instead of ~111, 564 block slots instead of 376 - o_a's 512 tiles in one wave). Register
// allocation only: the same arithmetic.
template <int T, int S, bool SwigluDown = false, int MB = 1>
__global__ void __launch_bounds__(256, MB) q8_mma_kernel(MmaArgs a) {
    extern __shared__ unsigned char activation[];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (a.pdl && (a.pdl_mode != 2 || SwigluDown)) {  // the weights of this block (no output of the previous kernel)
        constexpr int Up = 8 * S;                       // into L2 (mode 1), then the dependency
        const unsigned char* base = reinterpret_cast<const unsigned char*>(a.codes + std::size_t{a.tile_first + blockIdx.x} * a.groups_total * 32U);
        for (std::uint32_t i = threadIdx.x; a.pdl_mode == 1 && i < a.groups * 4U; i += blockDim.x) {
            const std::uint32_t g = i / 4U;
            if ((g % Up) / 8U != blockIdx.y) continue;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(base + (std::size_t{a.g_off} + g) * 512U + (i % 4U) * 128U));
        }
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
#endif
    // mode 2: this thread still owes griddepcontrol.wait (issued before its first activation load, or after the loop)
    bool pending_wait = !SwigluDown && a.pdl && a.pdl_mode == 2;
    (void)pending_wait;  // read only by the sm_90+ code
    if constexpr (SwigluDown) {
        const std::uint32_t row_base = blockIdx.z * 8U;
        const std::uint32_t rows = min(8U, a.rows - row_base);
        auto* codes = reinterpret_cast<std::int8_t*>(activation);
        auto* deltas = reinterpret_cast<float*>(activation + 8U * a.inter);
        const std::uint32_t lane = threadIdx.x & 31U;
        for (std::uint32_t group = threadIdx.x / 32U; group < rows * (a.inter / 32U); group += 8U) {
            const std::uint32_t r = group / (a.inter / 32U), g = group % (a.inter / 32U), i = g * 32U + lane;
            const std::size_t at = std::size_t{row_base + r} * 2U * a.inter + i;
            const float y = swiglu_clamped(a.gate_up[at], a.gate_up[at + a.inter], a.limit);
            float delta;
            std::int32_t sum;
            const auto code = detail::a8_group(y, delta, sum);
            codes[r * a.inter + i] = static_cast<std::int8_t>(code);
            if (lane == 0U) deltas[group] = delta;
            if (a.act.codes && blockIdx.x == 0U) {
                a.act.codes[std::size_t{row_base + r} * a.act.stride + i] = static_cast<std::int8_t>(code);
                if (lane == 0U) {
                    const auto ag = std::size_t{row_base + r} * a.act.stride / 32U + g;
                    a.act.deltas[ag] = delta; a.act.sums[ag] = sum;
                }
            }
        }
        __syncthreads();
        a.x = codes; a.deltas = deltas; a.x_stride = a.inter;
        a.rows = rows; a.out += std::size_t{row_base} * a.out_stride;
    }

    constexpr int U = 8 * S, NC = 32 / U;
    __shared__ float red[8][T * 8][16];
    const std::uint32_t tile = blockIdx.x, s = blockIdx.y, warp = threadIdx.x / 32U, lane = threadIdx.x & 31U;
    const std::uint32_t gid = lane / 4U, tig = lane % 4U, u = s * 8U + warp;
    const std::uint32_t r_base = SwigluDown ? 0U : blockIdx.z * (T * 8U);
    const std::uint32_t wt = a.tile_first + tile;
    const uint4* wc = a.codes + std::size_t{wt} * a.groups_total * 32U + lane;
    const std::uint16_t* ws = a.scales + std::size_t{wt} * a.groups_total * 16U;
    const std::size_t xoff0 = std::size_t{tile / a.per_group_tiles} * a.x_group_width;
    float yprev[(T * 8U * 16U + 255U) / 256U];
    for (std::uint32_t h = 0; h < a.halves; ++h) {
    const std::size_t xoff = xoff0 + std::size_t{h} * a.groups * 32U;
    const std::uint32_t g_base = a.g_off + h * a.groups;
    if (h) __syncthreads();  // red is reused
    float acc[NC][T][4];
#pragma unroll
    for (int k = 0; k < NC; ++k)
#pragma unroll
        for (int t = 0; t < T; ++t)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[k][t][j] = 0.0F;
    // Steps of 4 of the unit's groups (m = 4 j .. 4 j + 3, group u + U m, class slot m % NC = q % NC): every load
    // of a step is issued before its products; within a class the groups still add in ascending order.
    constexpr int B = 4;
    for (std::uint32_t m0 = 0; u + U * m0 < a.groups; m0 += B) {
        uint4 w[B];
        float sc0[B], sc1[B], d0[B][T], d1[B][T];
        int2 bv[B][T];
        // the step's weights and raw scales (constants), every load issued before any conversion: a conversion next to
        // its conditional load made each load its own branch region on sm_120 (the next load waited for the half)
        std::uint16_t s0r[B], s1r[B];
#pragma unroll
        for (int q = 0; q < B; ++q) {
            const std::uint32_t g = u + U * (m0 + q), gg = g_base + g;
            const bool ok = g < a.groups;
            w[q] = ok ? __ldg(wc + std::size_t{gg} * 32U) : make_uint4(0, 0, 0, 0);
            s0r[q] = ok ? ws[std::size_t{gg} * 16U + gid] : std::uint16_t{0};
            s1r[q] = ok ? ws[std::size_t{gg} * 16U + gid + 8U] : std::uint16_t{0};
        }
#pragma unroll
        for (int q = 0; q < B; ++q) {
            const bool ok = u + U * (m0 + q) < a.groups;
            sc0[q] = ok ? f16_to_float(s0r[q]) : 0.0F;
            sc1[q] = ok ? f16_to_float(s1r[q]) : 0.0F;
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        if (pending_wait) {  // the activations are the previous kernel's output
            asm volatile("griddepcontrol.wait;" ::: "memory");
            pending_wait = false;
        }
#endif
#pragma unroll
        for (int q = 0; q < B; ++q) {  // the step's activations and deltas
            const std::uint32_t g = u + U * (m0 + q);
            const bool ok = g < a.groups;
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const std::uint32_t r0 = r_base + 8U * t, rb = r0 + gid, rc = r0 + 2U * tig;
                bv[q][t] = ok && rb < a.rows ? *reinterpret_cast<const int2*>(a.x + std::size_t{rb} * a.x_stride + xoff + 32U * g + 8U * tig)
                                             : make_int2(0, 0);
                d0[q][t] = ok && rc < a.rows ? a.deltas[(std::size_t{rc} * a.x_stride + xoff) / 32U + g] : 0.0F;
                d1[q][t] = ok && rc + 1U < a.rows ? a.deltas[(std::size_t{rc + 1U} * a.x_stride + xoff) / 32U + g] : 0.0F;
            }
        }
#pragma unroll
        for (int q = 0; q < B; ++q) {
            if (u + U * (m0 + q) >= a.groups) break;
            const int k = q % NC;
#pragma unroll
            for (int t = 0; t < T; ++t) {
                if (r_base + 8U * t >= a.rows) break;
                int c[4];
                mma_s8(w[q], bv[q][t].x, bv[q][t].y, c);
                acc[k][t][0] = acc[k][t][0] + static_cast<float>(c[0]) * (sc0[q] * d0[q][t]);
                acc[k][t][1] = acc[k][t][1] + static_cast<float>(c[1]) * (sc0[q] * d1[q][t]);
                acc[k][t][2] = acc[k][t][2] + static_cast<float>(c[2]) * (sc1[q] * d0[q][t]);
                acc[k][t][3] = acc[k][t][3] + static_cast<float>(c[3]) * (sc1[q] * d1[q][t]);
            }
        }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (pending_wait) {  // a thread that ran no step: before any output, epilogue or later-half access
        asm volatile("griddepcontrol.wait;" ::: "memory");
        pending_wait = false;
    }
#endif
    // tree levels d >= U inside the unit (classes u + U q, q = 0 .. NC - 1)
    float v[T][4];
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if constexpr (NC == 4) v[t][j] = (acc[0][t][j] + acc[2][t][j]) + (acc[1][t][j] + acc[3][t][j]);
            else if constexpr (NC == 2) v[t][j] = acc[0][t][j] + acc[1][t][j];
            else v[t][j] = acc[0][t][j];
        }
    if constexpr (S == 1) {
#pragma unroll
        for (int t = 0; t < T; ++t)
#pragma unroll
            for (int j = 0; j < 4; ++j) red[warp][8 * t + 2 * tig + (j & 1)][gid + (j >= 2 ? 8U : 0U)] = v[t][j];
        __syncthreads();
        for (std::uint32_t cell = threadIdx.x, ci = 0; cell < T * 8U * 16U; cell += blockDim.x, ++ci) {
            const std::uint32_t rl = cell / 16U, mo = cell % 16U, row = r_base + rl;
            if (row >= a.rows) continue;
            float x4[4];
            for (int l = 0; l < 4; ++l) x4[l] = red[l][rl][mo] + red[l + 4][rl][mo];
            const float x2a = x4[0] + x4[2], x2b = x4[1] + x4[3];
            const float y = x2a + x2b;
            if (h + 1U < a.halves) { yprev[ci] = y; continue; }
            float* o = a.out + std::size_t{row} * a.out_stride + std::size_t{tile} * 16U + mo;
            const float yo = h ? yprev[ci] + y : a.accumulate ? *o + y : y;
            *o = yo;
            if (a.hc_out) {  // hc_expand_kernel's arithmetic for (row, i) with sub = yo
                const std::uint32_t i = tile * 16U + mo;
                const float* mix = a.hc_mixes + std::size_t{row} * 24U;
                float res[4];
                for (int j = 0; j < 4; ++j) res[j] = a.hc_residual[(std::size_t{row} * 4U + j) * a.hc_hidden + i];
                float yk[4];
                for (int k = 0; k < 4; ++k) {
                    float s = 0.0F;
                    for (int j = 0; j < 4; ++j) s = s + mix[8 + j * 4 + k] * res[j];
                    yk[k] = mix[4 + k] * yo + s;
                }
                for (int k = 0; k < 4; ++k) a.hc_out[(std::size_t{row} * 4U + k) * a.hc_hidden + i] = yk[k];
            }
        }
    } else {
#pragma unroll
        for (int t = 0; t < T; ++t)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::uint32_t row = r_base + 8U * t + 2U * tig + (j & 1U);
                if (row < a.rows)
                    a.scratch[((std::size_t{tile} * U + u) * a.rows + row) * 16U + gid + (j >= 2 ? 8U : 0U)] = v[t][j];
            }
    }
    }  // halves
}

// R1, S = 1 only: MMA columns are activation rows. Only tig == 0, c[0] (gid, 0)
// and c[2] (gid + 8, 0) survive; all lanes still participate in the MMA and its operand loads.
// launch bounds without a minimum-blocks cap (measured: the 3-blocks/SM cap cost +2 % in the graph)
__global__ void __launch_bounds__(256) q8_mma_r1_kernel(MmaArgs a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (a.pdl && a.pdl_mode != 2) {
        constexpr int Up = 8;
        const unsigned char* base = reinterpret_cast<const unsigned char*>(a.codes + std::size_t{a.tile_first + blockIdx.x} * a.groups_total * 32U);
        for (std::uint32_t i = threadIdx.x; a.pdl_mode == 1 && i < a.groups * 4U; i += blockDim.x) {
            const std::uint32_t g = i / 4U;
            if ((g % Up) / 8U != blockIdx.y) continue;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(base + (std::size_t{a.g_off} + g) * 512U + (i % 4U) * 128U));
        }
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
#endif
    bool pending_wait = a.pdl && a.pdl_mode == 2;
    (void)pending_wait;
    constexpr int U = 8, NC = 4, B = 4;
    __shared__ float red[8][16];
    const std::uint32_t tile = blockIdx.x, warp = threadIdx.x / 32U, lane = threadIdx.x & 31U;
    const std::uint32_t gid = lane / 4U, tig = lane % 4U, u = warp;
    const std::uint32_t wt = a.tile_first + tile;
    const uint4* wc = a.codes + std::size_t{wt} * a.groups_total * 32U + lane;
    const std::uint16_t* ws = a.scales + std::size_t{wt} * a.groups_total * 16U;
    const std::size_t xoff = std::size_t{tile / a.per_group_tiles} * a.x_group_width;
    float acc[NC][2];
#pragma unroll
    for (int k = 0; k < NC; ++k)
#pragma unroll
        for (int j = 0; j < 2; ++j) acc[k][j] = 0.0F;
    // Same four-group steps and ascending class chains as q8_mma_kernel<1, 1>.
    for (std::uint32_t m0 = 0; u + U * m0 < a.groups; m0 += B) {
        uint4 w[B];
        float sc0[B], sc1[B], d0[B];
        int2 bv[B];
        std::uint16_t s0r[B], s1r[B];
#pragma unroll
        for (int q = 0; q < B; ++q) {
            const std::uint32_t g = u + U * (m0 + q), gg = a.g_off + g;
            const bool ok = g < a.groups;
            w[q] = ok ? __ldg(wc + std::size_t{gg} * 32U) : make_uint4(0, 0, 0, 0);
            s0r[q] = ok ? ws[std::size_t{gg} * 16U + gid] : std::uint16_t{0};
            s1r[q] = ok ? ws[std::size_t{gg} * 16U + gid + 8U] : std::uint16_t{0};
        }
#pragma unroll
        for (int q = 0; q < B; ++q) {
            const bool ok = u + U * (m0 + q) < a.groups;
            sc0[q] = ok ? f16_to_float(s0r[q]) : 0.0F;
            sc1[q] = ok ? f16_to_float(s1r[q]) : 0.0F;
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        if (pending_wait) {
            asm volatile("griddepcontrol.wait;" ::: "memory");
            pending_wait = false;
        }
#endif
#pragma unroll
        for (int q = 0; q < B; ++q) {
            const std::uint32_t g = u + U * (m0 + q);
            const bool ok = g < a.groups;
            bv[q] = ok && gid == 0U ? *reinterpret_cast<const int2*>(a.x + xoff + 32U * g + 8U * tig) : make_int2(0, 0);
            d0[q] = ok && tig == 0U ? a.deltas[xoff / 32U + g] : 0.0F;
        }
#pragma unroll
        for (int q = 0; q < B; ++q) {
            if (u + U * (m0 + q) >= a.groups) break;
            const int k = q % NC;
            int c[4];
            mma_s8(w[q], bv[q].x, bv[q].y, c);
            if (tig == 0U) {
                acc[k][0] = acc[k][0] + static_cast<float>(c[0]) * (sc0[q] * d0[q]);
                acc[k][1] = acc[k][1] + static_cast<float>(c[2]) * (sc1[q] * d0[q]);
            }
        }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (pending_wait) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        pending_wait = false;
    }
#endif
    if (tig == 0U) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float v = (acc[0][j] + acc[2][j]) + (acc[1][j] + acc[3][j]);
            red[warp][gid + (j ? 8U : 0U)] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < 16U) {
        const std::uint32_t mo = threadIdx.x;
        float x4[4];
        for (int l = 0; l < 4; ++l) x4[l] = red[l][mo] + red[l + 4][mo];
        const float x2a = x4[0] + x4[2], x2b = x4[1] + x4[3];
        const float y = x2a + x2b;
        a.out[std::size_t{tile} * 16U + mo] = y;
    }
}

// S > 1: the tree levels d < U across the U unit values, per (tile, row, output).
__global__ void q8_mma_reduce_kernel(const float* __restrict__ scratch, std::uint32_t tiles, std::uint32_t units, std::uint32_t rows,
                                     float* out, std::uint32_t out_stride, bool accumulate) {
    // no early PDL trigger: a dependent GEMV launched at this kernel's entry sat resident beside it and slowed this
    // latency-bound kernel ~2x on the PRO; every CTA triggers implicitly when it exits
    const std::size_t idx = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (idx >= std::size_t{tiles} * rows * 16U) return;
    const std::uint32_t mo = static_cast<std::uint32_t>(idx % 16U), row = static_cast<std::uint32_t>((idx / 16U) % rows),
                        tile = static_cast<std::uint32_t>(idx / 16U / rows);
    float v[32];
    for (std::uint32_t l = 0; l < units; ++l) v[l] = scratch[((std::size_t{tile} * units + l) * rows + row) * 16U + mo];
    for (std::uint32_t d = units / 2U; d; d /= 2U)
        for (std::uint32_t l = 0; l < d; ++l) v[l] = v[l] + v[l + d];
    float* o = out + std::size_t{row} * out_stride + std::size_t{tile} * 16U + mo;
    *o = accumulate ? *o + v[0] : v[0];
}

int g_q8_occ = 1;
bool g_q8_r1_pad = false;  // measured: the 40 KB pad slowed the concurrent routed gate/up (+54 us/step) and shexp down
bool g_q8_r1 = false;

template <int T, int S, bool R1 = false>
void launch_mma(const MmaArgs& a, std::uint32_t tiles, cudaStream_t stream) {
    const dim3 grid(tiles, S, (a.rows + T * 8U - 1U) / (T * 8U));
    void (*kernel)(MmaArgs) = q8_mma_kernel<T, S>;
    if constexpr (T == 1 && S == 1) {
        if (a.occ == 3) kernel = q8_mma_kernel<1, 1, false, 3>;
    }
    std::size_t pad = 0;  // unused dynamic shared memory: caps the resident blocks per SM (scheduling only)
    if constexpr (R1) {
        static_assert(T == 1 && S == 1);
        kernel = q8_mma_r1_kernel;
        // measured: at 73 registers the R1 kernel runs 3 blocks per SM, and 3 blocks per SM cost +2 % in the decode
        // graph (--q8-occ3); 40 KB of unused shared memory per block keeps it at 2 like the generic kernel (sm_120: <= 100 KB
        // of shared memory per SM -> 2 x 41.5 KB fit, 3 do not)
        pad = g_q8_r1_pad ? 40U * 1024U : 0U;
    }
    if (a.pdl) {
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = grid;
        cfg.blockDim = dim3(256);
        cfg.dynamicSmemBytes = pad;
        cfg.stream = stream;
        cudaLaunchAttribute attr{};
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        STRIX_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, a));
    } else {
        kernel<<<grid, 256, pad, stream>>>(a);
    }
    STRIX_CUDA_CHECK(cudaGetLastError());
    if constexpr (S > 1) {
        const std::size_t n = std::size_t{tiles} * a.rows * 16U;
        q8_mma_reduce_kernel<<<static_cast<unsigned>((n + 255U) / 256U), 256, 0, stream>>>(a.scratch, tiles, 8U * S, a.rows, a.out,
                                                                                         a.out_stride, a.accumulate);
        STRIX_CUDA_CHECK(cudaGetLastError());
    }
}

int g_pdl_mode = 1;

}  // namespace

void set_q8_pdl_prefetch(bool on) { g_pdl_mode = on ? 1 : 0; }
void set_q8_occupancy(int blocks) { g_q8_occ = blocks == 3 ? 3 : 1; }
void set_q8_r1(bool on) { g_q8_r1 = on; }
void set_q8_r1_pad(bool on) { g_q8_r1_pad = on; }
void set_q8_pdl_mode(int mode) { g_pdl_mode = mode == 0 || mode == 2 ? mode : 1; }

std::size_t q8_tiled_code_bytes(std::uint32_t rows, std::uint32_t width) { return std::size_t{rows} / 16U * (width / 32U) * 512U; }
std::size_t q8_tiled_scale_count(std::uint32_t rows, std::uint32_t width) { return std::size_t{rows} / 16U * (width / 32U) * 16U; }

void enqueue_q8_tiled_repack(const std::uint8_t* q8, std::uint8_t* codes, std::uint16_t* scales,
                             std::uint32_t rows, std::uint32_t width, cudaStream_t stream) {
    require(rows % 16U == 0 && width % 32U == 0 && rows > 0, "tiled Q8_0 geometry (rows % 16, width % 32)");
    const std::size_t n = std::size_t{rows} / 16U * (width / 32U) * 32U;
    repack_kernel<<<static_cast<unsigned>((n + 255U) / 256U), 256, 0, stream>>>(q8, rows / 16U, width / 32U,
                                                                                reinterpret_cast<std::uint32_t*>(codes), scales);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

std::size_t q8_0_a8_split_scratch_floats(std::uint32_t count, std::uint32_t rows, std::uint32_t split) {
    return std::size_t{count} / 16U * 8U * split * rows * 16U;
}

void enqueue_swiglu_a8_down(const float* gate_up, float limit, Q8Tiled down,
                            float* out, std::uint32_t out_stride, A8Rows tap, const DensePlan& plan) {
    const auto rows = plan.rows;
    const auto stream = plan.stream;
    require_plan(plan, down.rows, down.width, rows);
    require(gate_up && rows > 0U && down.codes && down.scales && down.rows > 0U && down.rows % 16U == 0U &&
                down.width > 0U && down.width % 32U == 0U && down.width <= 4096U && out && (!out_stride || out_stride >= down.rows),
            "SwiGLU A8 down geometry");
    require(!tap.codes || (tap.rows == rows && tap.width == down.width && tap.stride >= down.width &&
                          tap.stride % 32U == 0U && tap.deltas && tap.sums), "SwiGLU A8 down tap geometry");
    MmaArgs a{reinterpret_cast<const uint4*>(down.codes), down.scales, down.width / 32U, 0U, down.width / 32U,
              0U, down.rows / 16U, down.width, nullptr, nullptr, down.width, rows, out,
              out_stride ? out_stride : down.rows, false, nullptr};
    a.gate_up = gate_up; a.inter = down.width; a.limit = limit; a.act = tap;
    const std::size_t smem = 8U * down.width + 8U * (down.width / 32U) * sizeof(float);
    q8_mma_kernel<1, 1, true><<<dim3(down.rows / 16U, 1U, (rows + 7U) / 8U), 256, smem, stream>>>(a);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_q8_0_a8_tiled(const Q8A8View& v, const DensePlan& plan) {
    const auto stream = plan.stream;
    const std::uint32_t width = v.width ? v.width : v.tiled.width, per_group = v.per_group ? v.per_group : v.count;
    const std::uint32_t out_stride = v.out_stride ? v.out_stride : v.count;
    require_plan(plan, v.count, width, v.x.rows);
    require(v.tiled.codes && v.first % 16U == 0 && v.count % 16U == 0 && per_group % 16U == 0 && v.count % per_group == 0 &&
                v.first + v.count <= v.tiled.rows && width % 32U == 0 && v.k_offset % 32U == 0 && v.k_offset + width <= v.tiled.width &&
                v.x.rows > 0 && v.x.stride % 8U == 0 && reinterpret_cast<std::uintptr_t>(v.x.codes) % 8U == 0,
            "tiled Q8_0 x A8 geometry");
    require((v.count / per_group - 1U) * width + width <= v.x.stride, "tiled Q8_0 x A8 activation stride");
    MmaArgs a{reinterpret_cast<const uint4*>(v.tiled.codes), v.tiled.scales, v.tiled.width / 32U, v.k_offset / 32U, width / 32U,
              v.first / 16U, per_group / 16U, width, v.x.codes, v.x.deltas, v.x.stride, v.x.rows, v.out, out_stride, v.accumulate,
              plan.split_scratch};
    // PDL needs sm_90 (the device's image); elsewhere a normal launch
    // PDL needs compute capability 9.0+: cached per device ordinal, API errors fail closed (no PDL)
    a.pdl = v.pdl && ::strix::cuda::arch::has_pdl();
    a.occ = v.occupancy ? v.occupancy : g_q8_occ;
    require(!v.hc_out || (v.hc_residual && v.hc_mixes && v.hc_hidden == v.count && v.first == 0U && !v.accumulate && v.k_halves <= 2U),
            "tiled Q8_0 x A8: the hc_expand epilogue needs count = hidden, first 0, no accumulate");
    a.hc_residual = v.hc_residual;
    a.hc_mixes = v.hc_mixes;
    a.hc_out = v.hc_out;
    a.hc_hidden = v.hc_hidden;
    a.pdl_mode = g_pdl_mode;
    a.halves = v.k_halves;
    require(v.k_halves == 1U || (v.k_halves == 2U && !v.accumulate && v.k_offset + 2U * width <= v.tiled.width &&
                                 (v.count / per_group - 1U) * width + 2U * width <= v.x.stride && per_group == v.count),
            "tiled Q8_0 x A8: two K-halves need one group, room for both halves, no accumulate");
    const std::uint32_t tiles = v.count / 16U;
    std::uint32_t split = v.split;
    if (split == 0) split = tiles >= 256U || !plan.split_scratch ? 1U : tiles >= 96U ? 2U : 4U;
    if (v.k_halves == 2U || v.hc_out) split = 1U;  // the second half / hc_expand run in the S = 1 epilogue
    require(split == 1U || (plan.split_scratch && plan.split_scratch_floats >= q8_0_a8_split_scratch_floats(v.count, v.x.rows, split)),
            "tiled Q8_0 x A8 split scratch");
    const bool big = plan.mode == DenseMode::prefill;
    if (g_q8_r1 && !(a.rows == 1U && split == 1U && a.halves == 1U && !a.accumulate && !a.hc_out)) {
        static bool noticed = false;
        if (!noticed) {
            std::fprintf(stderr, "q8-r1: shape requires generic Q8; using exact fallback\n");
            noticed = true;
        }
    }
    if (g_q8_r1 && a.rows == 1U && split == 1U && a.halves == 1U && !a.accumulate && !a.hc_out)
        launch_mma<1, 1, true>(a, tiles, stream);  // no SwiGLU preparation on this entry point
    else if (split == 1U) big ? launch_mma<4, 1>(a, tiles, stream) : launch_mma<1, 1>(a, tiles, stream);
    else if (split == 2U) big ? launch_mma<4, 2>(a, tiles, stream) : launch_mma<1, 2>(a, tiles, stream);
    else if (split == 4U) big ? launch_mma<4, 4>(a, tiles, stream) : launch_mma<1, 4>(a, tiles, stream);
    else require(false, "tiled Q8_0 x A8 split must be 1, 2 or 4");
}

}  // namespace strix::cuda::core::dense
