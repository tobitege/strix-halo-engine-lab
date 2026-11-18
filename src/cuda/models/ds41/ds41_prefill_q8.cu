// deepseek41 CUDA prefill-shape Q8_0 x A8 sites (cuda_ds41_prefill_q8.hpp): the tiled weights of
// enqueue_q8_0_a8_tiled over many rows per weight read, with its bits.
//
// Order of a warp's work: lane classes j = 0 .. 31 in bit-reversed order c = bitrev5(j); within a class the groups
// c, c + 32, ... ascending, term = float(dot) * (scale * delta), leaf = leaf + term from 0. The butterfly tree's leaf
// order (left to right) is exactly bitrev5(0), bitrev5(1), ...: its first level pairs c with c + 16 (j, j + 1 with
// j even), the next pairs those sums by bit 3 of c (bit 1 of j), and so on. So when class j ends, its sum is the right
// operand of the stack entries at the trailing one bits k of j (entry k = the left subtree of 2^k leaves), each
// cur = stk[k] + cur, and the result is pushed at the first zero bit. After j = 31 cur is v[0] of the tree.
// The block streams the groups in that same order through a cp.async pipeline (KG groups per stage): the tiled
// codes (512 bytes per 16 outputs, fragment order), the 16 FP16 scales of each tile, 32 activation codes and the delta
// of each row. float(dot) is formed as (1.5 * 2^23 + dot) - 1.5 * 2^23, exact for |dot| < 2^22 (|dot| <= 32 * 128 * 127).

#include "../../core/ptx.cuh"
#include "../../core/reduce.cuh"
#include "ds41_cuda_common.cuh"

#include "strix/cuda/ds41_prefill_q8.hpp"


namespace strix::ds41::cuda::pk {
namespace {

using detail::require;

using ::strix::cuda::core::ptx::cp_async16;
using ::strix::cuda::core::ptx::cp_async4;
using ::strix::cuda::core::ptx::cp_async_commit;
using ::strix::cuda::core::ptx::cp_async_wait;
using ::strix::cuda::core::ptx::mma_s8;

// float(i), exact for |i| < 2^22.
using ::strix::cuda::core::dot_float;

// IEEE half to float, exact (subnormal halves through the FP32 multiply by 2^112, denormals kept; Inf / NaN by the
// exponent mask), on the integer / FMA pipes instead of the conversion unit.
__device__ __forceinline__ float half_float(std::uint32_t h) {
    const std::uint32_t mag = (h & 0x7fffU) << 13U, sign = (h & 0x8000U) << 16U;
    const float f = __uint_as_float(mag) * 0x1p112F;
    const std::uint32_t bits = (h & 0x7c00U) == 0x7c00U ? mag | 0x7f800000U : __float_as_uint(f);
    return __uint_as_float(bits | sign);
}

__device__ __forceinline__ std::uint32_t bitrev5(std::uint32_t j) { return __brev(j) >> 27U; }

struct Args {
    const uint4* codes;
    const std::uint16_t* scales;
    std::uint32_t groups_total, g_off, groups, tile_first, tiles, per_group_tiles, x_group_width;
    const std::int8_t* x;
    const float* deltas;
    std::uint32_t x_stride, rows;
    float* out;
    std::uint32_t out_stride;
    bool accumulate;
};

// MT x NT mma tiles per warp (16 outputs x 8 rows each), WM x WN warps, KG groups per stage.
template <int MT, int NT, int WM, int WN, int KG, int STAGES>
struct Shape {
    static constexpr int BT = MT * WM, BM = 16 * BT, BN = 8 * NT * WN, threads = 32 * WM * WN;
    static constexpr int w_bytes = KG * BT * 512, s_bytes = KG * BM * 2, x_bytes = KG * BN * 32, d_bytes = KG * BN * 4;
    static constexpr int stage = w_bytes + s_bytes + x_bytes + d_bytes;
    static constexpr std::size_t smem = std::size_t{STAGES} * stage;
    static_assert(s_bytes % 16 == 0 && x_bytes % 16 == 0, "stage alignment");
};

template <int MT, int NT, int WM, int WN, int KG, int STAGES>
__global__ void __launch_bounds__(32 * WM * WN) q8_rows_kernel(const Args a) {
    using S = Shape<MT, NT, WM, WN, KG, STAGES>;
    constexpr int BT = S::BT, BN = S::BN, TPB = S::threads;
    extern __shared__ __align__(16) unsigned char smem[];
    const std::uint32_t tid = threadIdx.x, warp = tid / 32U, lane = tid & 31U, gid = lane / 4U, tig = lane % 4U;
    const std::uint32_t wm = warp % WM, wn = warp / WM;
    const std::uint32_t rb = blockIdx.x * BN, tb = blockIdx.y * BT;
    const std::size_t xoff = std::size_t{tb / a.per_group_tiles} * a.x_group_width;

    // ---- producer: the groups in the warp's order, KG per stage
    std::uint32_t pj = 0, pg = 0;
    const auto settle = [&] {
        while (pj < 32U && pg >= a.groups) {
            ++pj;
            pg = bitrev5(pj);
        }
    };
    settle();
    const auto load = [&](std::uint32_t slot, std::uint32_t kk, std::uint32_t g) {
        unsigned char* st = smem + std::size_t{slot} * S::stage;
        uint4* ws = reinterpret_cast<uint4*>(st) + kk * BT * 32U;
        std::uint16_t* ss = reinterpret_cast<std::uint16_t*>(st + S::w_bytes) + kk * BT * 16U;
        std::int8_t* xs = reinterpret_cast<std::int8_t*>(st + S::w_bytes + S::s_bytes) + kk * BN * 32U;
        float* ds = reinterpret_cast<float*>(st + S::w_bytes + S::s_bytes + S::x_bytes) + kk * BN;
        const std::uint32_t gg = a.g_off + g;
        for (std::uint32_t i = tid; i < BT * 32U; i += TPB) {
            const std::uint32_t t = min(tb + i / 32U, a.tiles - 1U);
            cp_async16(ws + i, a.codes + (std::size_t{a.tile_first + t} * a.groups_total + gg) * 32U + i % 32U);
        }
        for (std::uint32_t i = tid; i < BT * 2U; i += TPB) {
            const std::uint32_t t = min(tb + i / 2U, a.tiles - 1U);
            cp_async16(ss + 8U * i, a.scales + (std::size_t{a.tile_first + t} * a.groups_total + gg) * 16U + 8U * (i % 2U));
        }
        for (std::uint32_t i = tid; i < BN * 2U; i += TPB) {
            const std::uint32_t row = min(rb + i / 2U, a.rows - 1U);
            cp_async16(xs + 16U * i, a.x + std::size_t{row} * a.x_stride + xoff + 32U * g + 16U * (i % 2U));
        }
        for (std::uint32_t i = tid; i < BN; i += TPB) {
            const std::uint32_t row = min(rb + i, a.rows - 1U);
            cp_async4(ds + i, a.deltas + (std::size_t{row} * a.x_stride + xoff) / 32U + g);
        }
    };
    const auto load_stage = [&](std::uint32_t s) {
#pragma unroll
        for (int kk = 0; kk < KG; ++kk) {
            if (pj < 32U) {
                load(s % STAGES, kk, pg);
                pg += 32U;
                settle();
            }
        }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        load_stage(s);
        cp_async_commit();
    }

    // ---- consumer
    float cur[MT][NT][4], stk[5][MT][NT][4];
#pragma unroll
    for (int k = 0; k < 5; ++k)
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) stk[k][i][n][e] = 0.0F;
    std::uint32_t p = 0;
#pragma unroll 1
    for (std::uint32_t j = 0; j < 32U; ++j) {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) cur[i][n][e] = 0.0F;
#pragma unroll 1
        for (std::uint32_t g = bitrev5(j); g < a.groups; g += 32U, ++p) {
            const std::uint32_t kk = p % KG, s = p / KG;
            if (kk == 0U) {
                cp_async_wait<STAGES - 2>();
                __syncthreads();
                load_stage(s + STAGES - 1U);
                cp_async_commit();
            }
            const unsigned char* st = smem + std::size_t{s % STAGES} * S::stage;
            const uint4* ws = reinterpret_cast<const uint4*>(st) + kk * BT * 32U;
            const std::uint16_t* ss = reinterpret_cast<const std::uint16_t*>(st + S::w_bytes) + kk * BT * 16U;
            const std::int8_t* xs = reinterpret_cast<const std::int8_t*>(st + S::w_bytes + S::s_bytes) + kk * BN * 32U;
            const float* ds = reinterpret_cast<const float*>(st + S::w_bytes + S::s_bytes + S::x_bytes) + kk * BN;
            uint4 w[MT];
            float s0[MT], s1[MT];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const std::uint32_t tl = wm * MT + i;
                w[i] = ws[tl * 32U + lane];
                s0[i] = half_float(ss[tl * 16U + gid]);
                s1[i] = half_float(ss[tl * 16U + gid + 8U]);
            }
            int2 b[NT];
            float2 d[NT];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const std::uint32_t rl = (wn * NT + n) * 8U;
                b[n] = *reinterpret_cast<const int2*>(xs + (rl + gid) * 32U + 8U * tig);
                d[n] = *reinterpret_cast<const float2*>(ds + rl + 2U * tig);
            }
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    int c[4];
                    mma_s8(w[i], b[n].x, b[n].y, c);
                    cur[i][n][0] = cur[i][n][0] + dot_float(c[0]) * (s0[i] * d[n].x);
                    cur[i][n][1] = cur[i][n][1] + dot_float(c[1]) * (s0[i] * d[n].y);
                    cur[i][n][2] = cur[i][n][2] + dot_float(c[2]) * (s1[i] * d[n].x);
                    cur[i][n][3] = cur[i][n][3] + dot_float(c[3]) * (s1[i] * d[n].y);
                }
        }
        // fold class j into the tree
        bool placed = false;
#pragma unroll
        for (int k = 0; k < 5; ++k) {
            if (!placed) {
                if ((j >> k) & 1U) {
#pragma unroll
                    for (int i = 0; i < MT; ++i)
#pragma unroll
                        for (int n = 0; n < NT; ++n)
#pragma unroll
                            for (int e = 0; e < 4; ++e) cur[i][n][e] = stk[k][i][n][e] + cur[i][n][e];
                } else {
#pragma unroll
                    for (int i = 0; i < MT; ++i)
#pragma unroll
                        for (int n = 0; n < NT; ++n)
#pragma unroll
                            for (int e = 0; e < 4; ++e) stk[k][i][n][e] = cur[i][n][e];
                    placed = true;
                }
            }
        }
    }
    cp_async_wait<0>();

    // ---- epilogue: cur is v[0] of the tree
#pragma unroll
    for (int i = 0; i < MT; ++i) {
        const std::uint32_t t = tb + wm * MT + i;
        if (t >= a.tiles) continue;
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const std::uint32_t row = rb + (wn * NT + n) * 8U + 2U * tig + (e & 1U);
                if (row >= a.rows) continue;
                float* y = a.out + std::size_t{row} * a.out_stride + std::size_t{t} * 16U + gid + (e >= 2 ? 8U : 0U);
                *y = a.accumulate ? *y + cur[i][n][e] : cur[i][n][e];
            }
    }
}

template <int MT, int NT, int WM, int WN, int KG, int STAGES>
void launch(const Args& a, cudaStream_t stream) {
    using S = Shape<MT, NT, WM, WN, KG, STAGES>;
    const auto kernel = q8_rows_kernel<MT, NT, WM, WN, KG, STAGES>;
    static const bool configured = [&] {
        STRIX_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(S::smem)));
        return true;
    }();
    (void)configured;
    const dim3 grid((a.rows + S::BN - 1U) / S::BN, (a.tiles + S::BT - 1U) / S::BT);
    kernel<<<grid, S::threads, S::smem, stream>>>(a);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

// The block tile (outputs): grouped sites need per_group a multiple of it.
constexpr std::uint32_t kBlockOutputs = 64;

}  // namespace

bool q8_0_a8_rows_supported(const Q8A8View& v) {
    if (!v.tiled.codes || v.count == 0 || v.x.rows == 0) return false;
    const std::uint32_t width = v.width ? v.width : v.tiled.width, per_group = v.per_group ? v.per_group : v.count;
    return v.first % 16U == 0 && v.count % 16U == 0 && per_group % 16U == 0 && v.count % per_group == 0 && v.first + v.count <= v.tiled.rows &&
           width % 32U == 0 && width > 0 && v.k_offset % 32U == 0 && v.k_offset + width <= v.tiled.width && v.x.stride % 32U == 0 &&
           (v.count / per_group - 1U) * width + width <= v.x.stride && reinterpret_cast<std::uintptr_t>(v.x.codes) % 16U == 0 &&
           reinterpret_cast<std::uintptr_t>(v.x.deltas) % 4U == 0 && (per_group == v.count || per_group % kBlockOutputs == 0);
}

void enqueue_q8_0_a8_rows(const Q8A8View& v, cudaStream_t stream) {
    if (!q8_0_a8_rows_supported(v)) {
        Q8A8View t = v;
        t.split = 1;
        enqueue_q8_0_a8_tiled(t, stream);
        return;
    }
    const std::uint32_t width = v.width ? v.width : v.tiled.width, per_group = v.per_group ? v.per_group : v.count;
    const Args a{reinterpret_cast<const uint4*>(v.tiled.codes), v.tiled.scales, v.tiled.width / 32U, v.k_offset / 32U, width / 32U,
                 v.first / 16U, v.count / 16U, per_group / 16U, width, v.x.codes, v.x.deltas, v.x.stride, v.x.rows, v.out,
                 v.out_stride ? v.out_stride : v.count, v.accumulate};
    // 64 outputs x 64 rows per block (8 warps of 32 x 16), 4 groups per stage, 3 stages (54 KB): the fastest of the
    // shapes tried (larger warp tiles spill the class stack; smaller blocks read the weights more often)
    static_assert(Shape<2, 2, 2, 4, 4, 3>::BM == kBlockOutputs);
    launch<2, 2, 2, 4, 4, 3>(a, stream);
}

namespace {
std::size_t align256(std::size_t n) { return (n + 255U) / 256U * 256U; }
}  // namespace

void enqueue_shared_expert(const SharedExpertView& v, cudaStream_t stream) {
    if (!v.gate_up_tiled.codes) {
        cuda::enqueue_shared_expert(v, stream);
        return;
    }
    const std::uint32_t rows = v.x.rows, inter = v.gate_up_tiled.rows / 2U, hidden = v.gate_up_tiled.width;
    require(rows > 0 && v.workspace && v.x.width == hidden && v.down_tiled.codes && v.down_tiled.rows == hidden && v.down_tiled.width == inter,
            "shared expert geometry");
    // the workspace layout of shared_expert_workspace_bytes: [row][gate | up] floats, then the A8 of the activation
    auto* p = static_cast<std::uint8_t*>(v.workspace);
    const std::size_t f = align256(2U * std::size_t{rows} * inter * 4U), c = align256(std::size_t{rows} * inter),
                      g = align256(std::size_t{rows} * inter / 32U * 4U);
    float* gate = reinterpret_cast<float*>(p);
    const A8Rows act{reinterpret_cast<std::int8_t*>(p + f), reinterpret_cast<float*>(p + f + c), reinterpret_cast<std::int32_t*>(p + f + c + g),
                     rows, inter, inter};
    Q8A8View gu;
    gu.tiled = v.gate_up_tiled;
    gu.count = 2U * inter;
    gu.x = v.x;
    gu.out = gate;
    enqueue_q8_0_a8_rows(gu, stream);
    enqueue_swiglu_a8({gate, gate + inter, rows, inter, 2U * inter, v.limit, nullptr, act}, stream);
    Q8A8View d;
    d.tiled = v.down_tiled;
    d.count = hidden;
    d.x = act;
    d.out = v.out;
    d.out_stride = v.out_stride ? v.out_stride : hidden;
    enqueue_q8_0_a8_rows(d, stream);
}

}  // namespace strix::ds41::cuda::pk
