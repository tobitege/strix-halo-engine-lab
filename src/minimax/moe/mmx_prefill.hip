// minimax-m3 prefill MoE: grouped launches over all experts of a chunk (identity: mmx_prefill.hpp).
#include "minimax/moe/mmx_prefill.hpp"
#include "minimax/engine/route_identity.hpp"

#include "minimax/moe/mmx_prefill_core.hpp"
#include "minimax/moe/mmx_trellis_device.hpp"
#include "core/activation_quant.hpp"
#include "core/trellis_codec.hpp"
#ifndef STRIX_MMX_IU4_LIB
#define STRIX_MMX_IU4_LIB 0
#endif
#if STRIX_MMX_IU4_LIB
#include "core/moe_i4_chain.hpp"
#include "core/wmma_tiled.hpp"  // shared tiled core: grouped IU4 kernel and convert_exl3_iu4 (strix_tiled_core)
#endif
#include "moe_core/device_plan.hpp"
#include "strix/env_flags.hpp"
#include "strix/route_receipt.hpp"


#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace strix::minimax::moe::opt {

// Adopted strix-trellis 0.3 policy, shared with the host-only route census.
unsigned gemm_tile_rows(unsigned input, unsigned output, unsigned bits, unsigned rows) {
    return route::gemm_tile_rows(input, output, bits, rows);
}
unsigned gemm_splits(unsigned input, unsigned output, unsigned bits, unsigned rows) {
    return route::gemm_splits(input, output, bits, rows);
}

namespace {

namespace d = dev;

constexpr unsigned kHB = kHidden / 128, kIB = kInter / 128;  // 48, 12


// ---------------------------------------------------------------- library WMMA core pieces (gemm.hip, trellis_device.hpp)
template <int Bits> using WaveTile = strix::core::trellis::WaveTile<Bits>;
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));

// One grouped WMMA work item = one strix_gemm call's core grid (O/64 x ceil(rows/TileM) x splits CTAs).
struct WItem {
    const std::uint32_t* trellis;
    const _Float16* x;  // transformed rows [rows][input]
    float* part;        // [splits][rows][output]
    unsigned input, output, bits, rows, splits, cta_begin, gx, gy;
};

template <int Bits, int TileM>
__device__ void wmma_item(const WItem& it, unsigned bx, unsigned by, unsigned bz, _Float16* a, _Float16* q) {
    constexpr int AStride = 16;
    const int thread = static_cast<int>(threadIdx.x), lane = thread & 31, wave = thread / 32;
    const int first_row = static_cast<int>(by) * TileM;
    const int tile_n = static_cast<int>(bx) * 4 + wave;
    const int inputs = static_cast<int>(it.input), outputs = static_cast<int>(it.output), rows = static_cast<int>(it.rows);
    const int tiles_n = outputs / 16, tiles_k = inputs / 16;
    const int begin = static_cast<int>(static_cast<std::int64_t>(tiles_k) * bz / it.splits);
    const int end = static_cast<int>(static_cast<std::int64_t>(tiles_k) * (bz + 1) / it.splits);
    Float8 acc[TileM / 16] = {};
    for (int tile_k = begin; tile_k < end; ++tile_k) {
        for (int j = thread; j < TileM * 16; j += 128) {
            const int row = first_row + j / 16;
            a[(j / 16) * AStride + j % 16] =
                row < rows ? it.x[static_cast<std::size_t>(row) * static_cast<std::size_t>(inputs) + static_cast<std::size_t>(tile_k * 16 + j % 16)]
                           : static_cast<_Float16>(0);
        }
        if (tile_n < tiles_n) {
            const WaveTile<Bits> tile(it.trellis + (static_cast<std::size_t>(tile_k) * static_cast<std::size_t>(tiles_n) + static_cast<std::size_t>(tile_n)) * (Bits * 8));
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int r = 2 * (lane % 4) + (j % 2) + 8 * ((j / 2) % 2);
                const int c = lane / 4 + 8 * (j / 4);
                q[wave * 256 + r * 16 + c] = static_cast<_Float16>(strix::core::trellis::codebook_value(tile.state(8 * lane + j), 2));
            }
        }
        __syncthreads();
        if (tile_n < tiles_n) {
            Half16 right;
#pragma unroll
            for (int k = 0; k < 16; ++k) right[k] = q[wave * 256 + k * 16 + (lane % 16)];
#pragma unroll
            for (int m = 0; m < TileM / 16; ++m) {
                Half16 left;
#pragma unroll
                for (int k = 0; k < 16; ++k) left[k] = a[(m * 16 + lane % 16) * AStride + k];
                acc[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(left, right, acc[m]);
            }
        }
        __syncthreads();
    }
    if (tile_n < tiles_n) {
#pragma unroll
        for (int m = 0; m < TileM / 16; ++m) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int row = first_row + m * 16 + 2 * j + lane / 16;
                if (row < rows)
                    it.part[(static_cast<std::size_t>(bz) * static_cast<std::size_t>(rows) + static_cast<std::size_t>(row)) * static_cast<std::size_t>(outputs) +
                            static_cast<std::size_t>(tile_n * 16 + lane % 16)] = acc[m][j];
            }
        }
    }
}

template <int TileM>
__global__ __launch_bounds__(128) void wmma_grouped_kernel(const WItem* items, unsigned count) {
    __shared__ _Float16 a[TileM * 16];
    __shared__ _Float16 q[4 * 16 * 16];
    unsigned lo = 0, hi = count;  // last item with cta_begin <= blockIdx.x
    while (hi - lo > 1) {
        const unsigned mid = (lo + hi) / 2;
        if (items[mid].cta_begin <= blockIdx.x) lo = mid;
        else hi = mid;
    }
    const WItem it = items[lo];
    const unsigned local = blockIdx.x - it.cta_begin;
    const unsigned bx = local % it.gx, by = (local / it.gx) % it.gy, bz = local / (it.gx * it.gy);
    switch (it.bits) {
        case 3: wmma_item<3, TileM>(it, bx, by, bz, a, q); break;
        case 4: wmma_item<4, TileM>(it, bx, by, bz, a, q); break;
        case 5: wmma_item<5, TileM>(it, bx, by, bz, a, q); break;
        case 6: wmma_item<6, TileM>(it, bx, by, bz, a, q); break;
        default: wmma_item<8, TileM>(it, bx, by, bz, a, q); break;
    }
}

// ---------------------------------------------------------------- tables
struct PGroup {  // one expert (or the shared expert) of the chunk
    ExpertSlices w;
    unsigned n, pair_begin, s_gu, s_dn;
    unsigned tbegin;  // first row of the group in the tile-major (v4) xt / at layout, a multiple of 16
    std::size_t part_gu, part_dn;  // float offsets: gate [s_gu][n][1536], up right after; down [s_dn][n][6144]
    unsigned expert;  // routed expert id, kExperts = the shared expert (FAST_I4 tile list)
};
struct Tables {
    const PGroup* groups;
    const std::uint32_t* pair_row;    // [P]
    const std::uint32_t* pair_group;  // [P]
    const std::uint32_t* tok_pair;    // [T][4], ascending expert id
    const float* tok_w;               // [T][4]
    _Float16* xt;                     // [2][P][6144] (gate, up)
    _Float16* at;                     // [P][1536]
    float* part;
    unsigned pairs, rows, shared_group;
    unsigned tpairs;  // padded rows of the tile-major layout (v4); xt = [2][tpairs][6144], at = [tpairs][1536]
    unsigned fakeq;   // STRIX_MMX_FAKEQ_A4 (quality study): v4 transformed rows fake-quantised to A4 g64 before the FP16 store
    // Group-wide call: row r lives in chunk r / seg_rows (seg_rows = 0: one buffer, the kernels' y / out arguments).
    unsigned seg_rows;
    const float* seg_y[kPrefillMoeSegments];
    float* seg_out[kPrefillMoeSegments];
};
__device__ __forceinline__ const float* moe_row_in(const float* y, const Tables& t, unsigned r) {
    if (t.seg_rows == 0) return y + static_cast<std::size_t>(r) * kHidden;
    const unsigned s = r / t.seg_rows;
    return t.seg_y[s] + static_cast<std::size_t>(r - s * t.seg_rows) * kHidden;
}
__device__ __forceinline__ float* moe_row_out(float* out, const Tables& t, unsigned r) {
    if (t.seg_rows == 0) return out + static_cast<std::size_t>(r) * kHidden;
    const unsigned s = r / t.seg_rows;
    return t.seg_out[s] + static_cast<std::size_t>(r - s * t.seg_rows) * kHidden;
}

// FAST_I4 fake-quant (quality study): FP16-exact values -> q * s of the A4 g64 quantiser (core/activation_quant.hpp).
__device__ __forceinline__ void fakeq128(float (&v)[4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = static_cast<float>(static_cast<_Float16>(v[j]));
    strix::core::iu4_fake128(v);
}

// Tile-major position (v4) of row R (padded row index), column ch of a K-wide matrix: [R/16][ch/16][R%16][ch%16].
__device__ __forceinline__ std::size_t tiled_at(unsigned R, unsigned ch, unsigned K) {
    return ((static_cast<std::size_t>(R >> 4U) * (K >> 4U) + (ch >> 4U)) << 8U) + ((R & 15U) << 4U) + (ch & 15U);
}

// v4: gather + input transform (transform_block's operation order), stored tile-major.
__global__ __launch_bounds__(256) void gather_transform_tiled_kernel(const float* y, Tables t) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.pairs * 2 * kHB) return;
    const unsigned p = q / (2 * kHB), rem = q % (2 * kHB), m = rem / kHB, b = rem % kHB;
    const PGroup& g = t.groups[t.pair_group[p]];
    const std::uint16_t* suh = m == 0 ? g.w.gate.suh : g.w.up.suh;
    const float* x = moe_row_in(y, t, t.pair_row[p]);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = b * 128U + lane + 32U * static_cast<unsigned>(j);
        const float scaled = InputTransform::scale(x[c], suh[c]);
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    InputTransform::h128(v, lane);
    _Float16* out = t.xt + static_cast<std::size_t>(m) * t.tpairs * kHidden;
    const unsigned R = g.tbegin + (p - g.pair_begin);
#pragma unroll
    for (int j = 0; j < 4; ++j) out[tiled_at(R, b * 128U + lane + 32U * static_cast<unsigned>(j), kHidden)] = static_cast<_Float16>(v[j]);
}

// v4, token-major: one wave = one (token, 128-block). It reads the token's fp32 block once and runs
// gather_transform_tiled_kernel's per-(pair, matrix) statements for the token's 4 routed pairs and its shared pair, gate
// then up. Same values, same stores (padded rows untouched); the input block is read once instead of 10 times.
__global__ __launch_bounds__(256) void gather_transform_token_kernel(const float* y, Tables t) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.rows * kHB) return;
    const unsigned tok = q / kHB, b = q % kHB;
    const float* x = moe_row_in(y, t, tok);
    float xv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) xv[j] = x[b * 128U + lane + 32U * static_cast<unsigned>(j)];
#pragma unroll 1
    for (unsigned k = 0; k <= kTopE; ++k) {
        const unsigned p = k < kTopE ? t.tok_pair[tok * kTopE + k] : t.groups[t.shared_group].pair_begin + tok;
        const PGroup& g = t.groups[t.pair_group[p]];
        const unsigned R = g.tbegin + (p - g.pair_begin);
#pragma unroll
        for (unsigned m = 0; m < 2; ++m) {
            const std::uint16_t* suh = m == 0 ? g.w.gate.suh : g.w.up.suh;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned c = b * 128U + lane + 32U * static_cast<unsigned>(j);
                const float scaled = InputTransform::scale(xv[j], suh[c]);
                v[j] = static_cast<float>(static_cast<_Float16>(scaled));
            }
            InputTransform::h128(v, lane);
            if (t.fakeq) fakeq128(v);
            _Float16* out = t.xt + static_cast<std::size_t>(m) * t.tpairs * kHidden;
#pragma unroll
            for (int j = 0; j < 4; ++j) out[tiled_at(R, b * 128U + lane + 32U * static_cast<unsigned>(j), kHidden)] = static_cast<_Float16>(v[j]);
        }
    }
}



// output_transform of one 128-block of one row from split partials [splits][rows][output].


__device__ __forceinline__ float swiglu(float g, float u) {
    return Epilogue::swiglu(g, u);
}

__device__ __forceinline__ void finish_block(const float* part, unsigned splits, unsigned rows, unsigned row, unsigned output,
                                             unsigned b, const std::uint16_t* svh, unsigned lane, float (&v)[4]) {
    return Epilogue::finish_block(part, splits, rows, row, output, b, svh, lane, v);
}

// Shared epilogue: preserve output transforms, SwiGLU and both FP16 conversions.
__device__ __forceinline__ void gate_up_tiled_store(Tables t, const PGroup& g, unsigned p, unsigned i,
                                                  unsigned b, unsigned lane, const float (&gv)[4],
                                                  const float (&uv)[4], float* debug) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = b * 128 + lane + 32U * static_cast<unsigned>(j);
        const float act = swiglu(gv[j], uv[j]);
        const float scaled = InputTransform::scale(act, g.w.down.suh[c]);
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
        if (debug != nullptr) {
            const std::size_t idx = static_cast<std::size_t>(p) * kInter + c;
            const std::size_t stride = static_cast<std::size_t>(t.pairs) * kInter;
            debug[idx] = gv[j];
            debug[stride + idx] = uv[j];
            debug[2 * stride + idx] = act;
            debug[3 * stride + idx] = v[j];
        }
    }
    InputTransform::h128(v, lane);
    if (t.fakeq) fakeq128(v);
    const unsigned R = g.tbegin + i;
#pragma unroll
    for (int j = 0; j < 4; ++j) t.at[tiled_at(R, b * 128 + lane + 32U * static_cast<unsigned>(j), kInter)] = static_cast<_Float16>(v[j]);
}

// v4: gate/up finish with the down input stored tile-major.
__global__ __launch_bounds__(256) void gate_up_finish_tiled_kernel(Tables t, bool fused, float* debug) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.pairs * kIB) return;
    const unsigned p = q / kIB, b = q % kIB;
    const PGroup& g = t.groups[t.pair_group[p]];
    if (fused && g.s_gu == 1) return;
    const unsigned i = p - g.pair_begin;
    float gv[4], uv[4];
    finish_block(t.part + g.part_gu, g.s_gu, g.n, i, kInter, b, g.w.gate.svh, lane, gv);
    finish_block(t.part + g.part_gu + static_cast<std::size_t>(g.s_gu) * g.n * kInter, g.s_gu, g.n, i, kInter, b, g.w.up.svh, lane, uv);
    gate_up_tiled_store(t, g, p, i, b, lane, gv, uv, debug);
}

// Items are consecutive gate/up pairs, with identical row partitions and CTA ranges.
template <int MT>
__global__ __launch_bounds__(256) void gate_up_fused_kernel(const pfk::Item* items, unsigned count, Tables t,
                                                          bool partials, float* debug, bool producer_only) {
    __shared__ float drain[2 * 16 * 128];  // 16 KiB, reused for every sixteen-row slice
    unsigned lo = 0, hi = count;
    while (hi - lo > 1) {
        const unsigned mid = (lo + hi) / 2;
        if (items[ExpertLayout::kGateUpInterleave * mid].cta_begin <= blockIdx.x) lo = mid;
        else hi = mid;
    }
    const pfk::Item gate = items[ExpertLayout::kGateUpInterleave * lo];
    const unsigned matrix = threadIdx.x >> 7U;
    const pfk::Item it = items[ExpertLayout::kGateUpInterleave * lo + matrix];
    const unsigned local = blockIdx.x - gate.cta_begin;
    const unsigned bx = local % gate.gx, by = local / gate.gx;
    // Only wave-local control flow differs for mixed gate/up bit widths. All waves
    // converge before the first CTA barrier in the drain loop.
    pfk::Accum<MT> acc;
    switch (it.bits) {
        case 3: acc = pfk::item_acc_v3<3, MT, 0, true, 0, true>(it, bx, by, 0); break;
        case 4: acc = pfk::item_acc_v3<4, MT, 0, true, 0, true>(it, bx, by, 0); break;
        case 5: acc = pfk::item_acc_v3<5, MT, 0, true, 0, true>(it, bx, by, 0); break;
        case 6: acc = pfk::item_acc_v3<6, MT, 0, true, 0, true>(it, bx, by, 0); break;
        default: acc = pfk::item_acc_v3<8, MT, 0, true, 0, true>(it, bx, by, 0); break;
    }
    unsigned gl = 0, gh = t.shared_group + 1;
    while (gh - gl > 1) {
        const unsigned mid = (gl + gh) / 2;
        if (t.part + t.groups[mid].part_gu <= gate.part) gl = mid;
        else gh = mid;
    }
    const PGroup& g = t.groups[gl];
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U;
    const unsigned col = (wave & 3U) * 32U + (lane & 15U);
    const unsigned first = it.row0 + by * MT * 16U;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned r = 2U * static_cast<unsigned>(j) + (lane >> 4U);
            if (!producer_only) {
                drain[matrix * 2048U + r * 128U + col] = acc.acc0[m][j];
                drain[matrix * 2048U + r * 128U + col + 16U] = acc.acc1[m][j];
            }
            if (partials && first + 16U * static_cast<unsigned>(m) + r < it.rows) {
                float* dst = it.part + static_cast<std::size_t>(first + 16U * static_cast<unsigned>(m) + r) * kInter + bx * 128U + col;
                dst[0] = acc.acc0[m][j];
                dst[16] = acc.acc1[m][j];
            }
        }
        if (producer_only) continue;
        __syncthreads();
        for (unsigned r = wave; r < 16; r += 8) {
            const unsigned row = first + 16U * static_cast<unsigned>(m) + r;
            if (row < it.rows) {
                float gv[4], uv[4];
                // finish_block retains the initial +0.0F + partial, even for one split.
                finish_block(drain, 1, 16, r, 128, 0, g.w.gate.svh + bx * 128U, lane, gv);
                finish_block(drain + 2048, 1, 16, r, 128, 0, g.w.up.svh + bx * 128U, lane, uv);
                gate_up_tiled_store(t, g, g.pair_begin + row, row, bx, lane, gv, uv, debug);
            }
        }
        __syncthreads();  // all consumers finish before the next slice overwrites LDS
    }
}

__global__ __launch_bounds__(256) void down_combine_kernel(Tables t, float* out) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.rows * kHB) return;
    const unsigned tok = q / kHB, b = q % kHB;
    float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (unsigned k = 0; k < kTopE; ++k) {
        const unsigned p = t.tok_pair[tok * kTopE + k];
        const PGroup& g = t.groups[t.pair_group[p]];
        float v[4];
        finish_block(t.part + g.part_dn, g.s_dn, g.n, p - g.pair_begin, kHidden, b, g.w.down.svh, lane, v);
        const float w = t.tok_w[tok * kTopE + k];
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = Epilogue::router_accumulate(w, v[j], acc[j]);
    }
    const PGroup& sg = t.groups[t.shared_group];
    float v[4];
    finish_block(t.part + sg.part_dn, sg.s_dn, sg.n, tok, kHidden, b, sg.w.down.svh, lane, v);
    float* const o = moe_row_out(out, t, tok);
#pragma unroll
    for (int j = 0; j < 4; ++j) o[b * 128 + lane + 32U * static_cast<unsigned>(j)] = Epilogue::shared_add(acc[j], v[j]);
}

// Dense prefill linears on the v4 core (enqueue_prefill_dense): the library input transform, row-major x -> tile-major
// f16 per job (transform_block's statements, one x read for every job), and the library output transform per job.
struct DenseArgs {
    const std::uint16_t* suh[kMaxDenseJobs];
    const std::uint16_t* svh[kMaxDenseJobs];
    _Float16* xt[kMaxDenseJobs];
    const float* part[kMaxDenseJobs];
    float* out[kMaxDenseJobs];
    unsigned n[kMaxDenseJobs], splits[kMaxDenseJobs];
    unsigned count, rows, k;
};

__global__ __launch_bounds__(256) void dense_transform_kernel(const float* x, DenseArgs a) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave, kb = a.k / 128U;
    if (q >= a.rows * kb) return;
    const unsigned row = q / kb, b = q % kb;
    const float* xr = x + static_cast<std::size_t>(row) * a.k;
    float xv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) xv[j] = xr[b * 128U + lane + 32U * static_cast<unsigned>(j)];
#pragma unroll 1
    for (unsigned m = 0; m < a.count; ++m) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned c = b * 128U + lane + 32U * static_cast<unsigned>(j);
            const float scaled = InputTransform::scale(xv[j], a.suh[m][c]);
            v[j] = static_cast<float>(static_cast<_Float16>(scaled));
        }
        InputTransform::h128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            a.xt[m][tiled_at(row, b * 128U + lane + 32U * static_cast<unsigned>(j), a.k)] = static_cast<_Float16>(v[j]);
    }
}

__global__ __launch_bounds__(256) void dense_finish_kernel(DenseArgs a) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    unsigned q = blockIdx.x * 8U + wave, m = 0;
    for (; m < a.count; ++m) {  // job m owns rows * n / 128 consecutive waves
        const unsigned w = a.rows * (a.n[m] / 128U);
        if (q < w) break;
        q -= w;
    }
    if (m == a.count) return;
    const unsigned nb = a.n[m] / 128U, row = q / nb, b = q % nb;
    float v[4];
    finish_block(a.part[m], a.splits[m], a.rows, row, a.n[m], b, a.svh[m], lane, v);
#pragma unroll
    for (int j = 0; j < 4; ++j)
        a.out[m][static_cast<std::size_t>(row) * a.n[m] + b * 128U + lane + 32U * static_cast<unsigned>(j)] = v[j];
}

void launched(const char* what) {
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("minimax prefill moe ") + what + ": " + hipGetErrorString(e));
}
void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("minimax prefill moe ") + what + ": " + hipGetErrorString(e));
}

constexpr std::size_t align256(std::size_t v) { return (v + 255) & ~std::size_t{255}; }

std::size_t table_bound(std::uint32_t rows) {
    const std::size_t pairs = 5ULL * rows, groups = kExperts + 1;
    return align256(groups * sizeof(PGroup)) + 2 * align256(pairs * 4) +
           2 * align256(4ULL * rows * 4) + align256(groups * std::max(3 * sizeof(WItem), 6 * sizeof(pfk::Item)));
}

// Module state is used by one engine host thread; setters, enqueue and cleanup must not race.
// Slots protect only H2D sources. Device workspace reuse must be ordered on the caller's stream.
constexpr unsigned N_SLOTS = 4;
struct TableSlot {
    unsigned char* host = nullptr;
    std::size_t capacity = 0;
    hipEvent_t copied = nullptr;
    bool pending = false;
};
TableSlot g_table_slots[N_SLOTS];
unsigned g_table_next = 0;
int g_table_ring_override = -1;
bool table_ring_default() {
    static const bool on = [] {
        const char* e = std::getenv("STRIX_MM_PF_TABLE_RING");
        if (e == nullptr || std::string(e) == "0") return false;
        if (std::string(e) == "1") return true;
        throw std::invalid_argument("STRIX_MM_PF_TABLE_RING must be 0 or 1");
    }();
    return on;
}
bool table_ring() { return g_table_ring_override < 0 ? table_ring_default() : g_table_ring_override != 0; }
TableSlot& next_table_slot(std::uint32_t rows, std::size_t extra = 0) {
    TableSlot& slot = g_table_slots[g_table_next];
    if (slot.pending) {
        ok(hipEventSynchronize(slot.copied), "table slot wait");
        slot.pending = false;
    }
    // Reserve the bound's table prefix, never its large activation/partial regions.
    const std::size_t capacity = table_bound(std::max(rows, 4096U)) + extra;
    if (slot.copied == nullptr) ok(hipEventCreateWithFlags(&slot.copied, hipEventDisableTiming), "table event");
    if (slot.capacity < capacity) {
        if (slot.host != nullptr) {
            ok(hipHostFree(slot.host), "table slot free");
            slot.host = nullptr;
            slot.capacity = 0;
        }
        ok(hipHostMalloc(reinterpret_cast<void**>(&slot.host), capacity), "table slot malloc");
        slot.capacity = capacity;
    }
    g_table_next = (g_table_next + 1) % N_SLOTS;
    return slot;
}

// Host plan of one call: groups, pair/token tables and the workspace layout.
struct Plan {
    std::vector<PGroup> groups;
    std::vector<std::uint32_t> pair_row, pair_group, tok_pair;
    std::vector<float> tok_w;
    std::vector<WItem> gu_items, dn_items16, dn_items64;
    unsigned gu_ctas = 0, dn_ctas16 = 0, dn_ctas64 = 0;
    std::size_t part_floats = 0;
    unsigned tpairs = 0;  // padded rows of the tile-major layout (sum of the groups' rows rounded up to 16)
    std::size_t off_groups = 0, off_pair_row = 0, off_pair_group = 0, off_tok_pair = 0, off_tok_w = 0, off_items = 0,
                off_i4_groups = 0, off_xt = 0, off_at = 0, off_part = 0, bytes = 0;
};

Plan make_plan(const PrefillMoeView& v) {
    if (v.rows == 0) throw std::invalid_argument("minimax prefill moe: rows must be > 0");
    if (v.ids == nullptr || v.weights == nullptr || v.experts == nullptr || v.y == nullptr || v.out == nullptr)
        throw std::invalid_argument("minimax prefill moe: null pointer in view");
    const unsigned T = v.rows;
    Plan p;
    const auto erows = route::expert_rows(std::span<const std::uint32_t>(v.ids, std::size_t{T} * kTopE), T, kExperts, kTopE);
    std::vector<std::uint32_t> group_of(kExperts, ~0U);
    auto add_group = [&](const ExpertSlices& w, const std::vector<std::uint32_t>& rows, unsigned expert) {
        PGroup g{};
        g.w = w;
        g.expert = expert;
        g.n = static_cast<unsigned>(rows.size());
        g.pair_begin = static_cast<unsigned>(p.pair_row.size());
        g.tbegin = p.tpairs;
        p.tpairs += (g.n + 15U) / 16U * 16U;
        g.s_gu = gemm_splits(kHidden, kInter, w.gate.bits, g.n);
        g.s_dn = gemm_splits(kInter, kHidden, w.down.bits, g.n);
        if (gemm_splits(kHidden, kInter, w.up.bits, g.n) != g.s_gu || gemm_tile_rows(kHidden, kInter, w.gate.bits, g.n) != 16 ||
            gemm_tile_rows(kHidden, kInter, w.up.bits, g.n) != 16)
            throw std::invalid_argument("minimax prefill moe: gate/up policy differs between the two matrices");
        g.part_gu = p.part_floats;
        p.part_floats += 2 * static_cast<std::size_t>(g.s_gu) * g.n * kInter;
        g.part_dn = p.part_floats;
        p.part_floats += static_cast<std::size_t>(g.s_dn) * g.n * kHidden;
        const auto gi = static_cast<std::uint32_t>(p.groups.size());
        for (const std::uint32_t r : rows) {
            p.pair_row.push_back(r);
            p.pair_group.push_back(gi);
        }
        p.groups.push_back(g);
        return gi;
    };
    for (unsigned e = 0; e < kExperts; ++e)
        if (!erows[e].empty()) group_of[e] = add_group(v.experts[e], erows[e], e);
    std::vector<std::uint32_t> all(T);
    for (unsigned t = 0; t < T; ++t) all[t] = t;
    const std::uint32_t shared_group = add_group(v.shared, all, kExperts);
    // Token tables: the 4 pairs of each token in ascending expert id (its position in each expert's row list).
    p.tok_pair.resize(std::size_t{T} * kTopE);
    p.tok_w.resize(std::size_t{T} * kTopE);
    std::vector<std::uint32_t> cursor(kExperts, 0);
    for (unsigned t = 0; t < T; ++t) {
        std::uint32_t id[kTopE];
        float w[kTopE];
        for (unsigned k = 0; k < kTopE; ++k) {
            id[k] = v.ids[std::size_t{t} * kTopE + k];
            w[k] = v.weights[std::size_t{t} * kTopE + k];
        }
        for (unsigned i = 1; i < kTopE; ++i)
            for (unsigned j = i; j > 0 && id[j - 1] > id[j]; --j) {
                std::swap(id[j], id[j - 1]);
                std::swap(w[j], w[j - 1]);
            }
        for (unsigned k = 0; k < kTopE; ++k) {
            const PGroup& g = p.groups[group_of[id[k]]];
            p.tok_pair[std::size_t{t} * kTopE + k] = g.pair_begin + cursor[id[k]]++;
            p.tok_w[std::size_t{t} * kTopE + k] = w[k];
        }
    }
    // Workspace layout.
    const std::size_t P = p.pair_row.size();
    std::size_t off = 0;
    auto take = [&](std::size_t bytes) {
        const std::size_t at = off;
        off += align256(bytes);
        return at;
    };
    p.off_groups = take(p.groups.size() * sizeof(PGroup));
    p.off_pair_row = take(P * 4);
    p.off_pair_group = take(P * 4);
    p.off_tok_pair = take(p.tok_pair.size() * 4);
    p.off_tok_w = take(p.tok_w.size() * 4);
    const std::size_t item_bytes = std::max(3 * sizeof(WItem), 6 * sizeof(pfk::Item));
    if (v.i4 != nullptr && !v.i4_all) p.off_i4_groups = take(kI4Experts * 3U * sizeof(std::uint32_t));
    p.off_items = take(p.groups.size() * item_bytes);
    if (v.i4 == nullptr || v.i4_all) p.off_i4_groups = p.off_items;
    const std::size_t PX = std::max<std::size_t>(P, p.tpairs);  // the lib / v2 / v3 row-major or the v4 tiled layout
    p.off_xt = take(2 * PX * kHidden * 2);
    p.off_at = take(PX * kInter * 2);
    p.off_part = take(p.part_floats * 4);
    p.bytes = off;
    (void)shared_group;
    return p;
}

// STRIX_MMX_PREFILL_TILE: 0 (default) = the library's row tile per call (16 or 64 rows per CTA), 128 = 128 rows per
// CTA for every item. Same bits: every 16-row WMMA of a row tile is independent of the others, so grouping more row
// tiles per CTA only reuses the decoded weight tile (one weight pass per 128 rows instead of per 16).
unsigned prefill_tile() {
    static const unsigned t = [] {
        const char* e = std::getenv("STRIX_MMX_PREFILL_TILE");
        const unsigned x = e != nullptr ? static_cast<unsigned>(std::atoi(e)) : 0U;
        if (x != 0 && x != 128) throw std::invalid_argument("STRIX_MMX_PREFILL_TILE must be 0 or 128");
        return x;
    }();
    return t;
}

// The v4 WMMA core is the sole prefill core path.
int g_mtfit_override = -1;
int g_core8_override = -1;
int g_pipe_override = -1;
int g_pipe_pair_override = -1;
int g_dense_band_override = -1;
int g_tail_stream_override = -1;
int g_gufuse_override = -1;
int g_gather_tok_override = -1;
bool g_gufuse_debug = false;
bool g_gufuse_producer_only = false;
float* g_gufuse_intermediates = nullptr;
PrefillMarker g_marker = nullptr;  // profiling: called with a label after each launch (set_prefill_marker)
void* g_marker_user = nullptr;
void mark(const char* label, hipStream_t stream) {
    if (g_marker != nullptr) g_marker(label, stream, g_marker_user);
}
bool prefill_mtfit_default() {
    static const bool on = [] {
        const char* e = std::getenv("STRIX_MMX_PREFILL_MTFIT");
        return e == nullptr || std::string(e) != "0";  // default ON (prefill measurement: exact); =0 off
    }();
    return on;
}
bool prefill_mtfit() {
    return g_mtfit_override < 0 ? prefill_mtfit_default() : g_mtfit_override != 0;
}

int prefill_core8_default() {
    static const int value = [] {
        const char* e = std::getenv("STRIX_MMX_PREFILL_CORE8");
        if (e == nullptr || std::string(e) == "0") return 0;
        if (std::string(e) == "1") return 1;
        if (std::string(e) == "2") return 2;
        throw std::invalid_argument("STRIX_MMX_PREFILL_CORE8 must be 0, 1 or 2");
    }();
    return value;
}
int prefill_core8() { return g_core8_override < 0 ? prefill_core8_default() : g_core8_override; }

int prefill_pipe_default() {
    static const int value = [] {
        const char* e = std::getenv("STRIX_MMX_PREFILL_PIPE");
        if (e == nullptr) return 1;  // default PIPE=1 (measurement: exact, 8K B/A 0.973/0.935/0.924; b1 bench bitwise)
        if (std::string(e) == "0") return 0;
        if (std::string(e) == "1") return 1;
        if (std::string(e) == "2") return 2;
        throw std::invalid_argument("STRIX_MMX_PREFILL_PIPE must be 0, 1 or 2");
    }();
    return value;
}
int prefill_pipe() { return g_pipe_override < 0 ? prefill_pipe_default() : g_pipe_override; }

// STRIX_MMX_PREFILL_PIPE_PAIR (under PIPE): auto (default) = paired MT8 bands for calls of at least kPairMinRows rows
// (whole-call ABBA: 4096 rows -0.9..-1.5 %, 2048 rows -0.6..-1.0 %; 1024 rows mixed, -1.1..+0.7 %, and the two-node 1K
// prompt not faster, so it stays unpaired), 0 = never, 1 = every call. Same bits.
constexpr unsigned kPairMinRows = 2048;
int prefill_pipe_pair_default() {  // 2 = auto
    static const int value = [] {
        const char* e = std::getenv("STRIX_MMX_PREFILL_PIPE_PAIR");
        if (e == nullptr || std::string(e) == "auto") return 2;
        if (std::string(e) == "0") return 0;
        if (std::string(e) == "1") return 1;
        throw std::invalid_argument("STRIX_MMX_PREFILL_PIPE_PAIR must be auto, 0 or 1");
    }();
    return value;
}
bool prefill_pipe_pair(unsigned rows) {
    const int mode = g_pipe_pair_override < 0 ? prefill_pipe_pair_default() : g_pipe_pair_override;
    return mode == 1 || (mode == 2 && rows >= kPairMinRows);
}
// STRIX_MMX_PREFILL_DENSE_BAND (dense linears under PIPE): 0 (default) = MT8 PIPE CTAs, 4 / 8 = tall bands of that many
// waves over 512 / 1024 rows (one weight decode per band); rows beyond the last full band keep PIPE / the tail. Same bits.
int prefill_dense_band_default() {
    static const int value = static_cast<int>(
        4 * strix::env_choice("STRIX_MMX_PREFILL_DENSE_BAND", 0, {"0", "4", "8"}));  // 0 / 4 / 8
    return value;
}
unsigned prefill_dense_band() {
    if (prefill_pipe() == 0) return 0;
    return static_cast<unsigned>(g_dense_band_override < 0 ? prefill_dense_band_default() : g_dense_band_override);
}

// STRIX_MMX_PREFILL_TAIL_STREAM (v4 + mtfit): 1 (default) = the tail-class launches (MT 1..7) of each stage run on a side
// stream beside the MT8 launch, joined before the stage's consumer; 0 = every class launch on the caller's stream.
// Same items, kernels and bits; the classes write disjoint partials.
bool prefill_tail_stream_default() {
    static const bool on = [] {
        const char* e = std::getenv("STRIX_MMX_PREFILL_TAIL_STREAM");
        if (e == nullptr || std::string(e) == "1") return true;
        if (std::string(e) == "0") return false;
        throw std::invalid_argument("STRIX_MMX_PREFILL_TAIL_STREAM must be 0 or 1");
    }();
    return on;
}
bool prefill_tail_stream() { return g_tail_stream_override < 0 ? prefill_tail_stream_default() : g_tail_stream_override != 0; }

bool prefill_gufuse_default() {
    static const bool on = [] {
        const char* e = std::getenv("STRIX_MMX_PREFILL_GUFUSE");
        if (e == nullptr || std::string(e) == "0") return false;
        if (std::string(e) == "1") return true;
        throw std::invalid_argument("STRIX_MMX_PREFILL_GUFUSE must be 0 or 1");
    }();
    return on;
}
bool prefill_gufuse() { return g_gufuse_override < 0 ? prefill_gufuse_default() : g_gufuse_override != 0; }

// STRIX_MMX_PREFILL_GATHER_TOK: 1 (default) = v4 gather in token-major order (gather_transform_token_kernel),
// 0 = one wave per (pair, matrix, block). Same values and stores.
bool prefill_gather_tok_default() {
    static const bool on = [] {
        const char* e = std::getenv("STRIX_MMX_PREFILL_GATHER_TOK");
        if (e == nullptr || std::string(e) == "1") return true;
        if (std::string(e) == "0") return false;
        throw std::invalid_argument("STRIX_MMX_PREFILL_GATHER_TOK must be 0 or 1");
    }();
    return on;
}
bool prefill_gather_tok() { return g_gather_tok_override < 0 ? prefill_gather_tok_default() : g_gather_tok_override != 0; }

// Gate/up and down lists: v4 classes are MT - 1; older cores retain the four power-of-two classes.
struct V2Items {
    std::vector<pfk::Item> gu[8], dn[8], fused[8];
    unsigned gu_ctas[8] = {}, dn_ctas[8] = {}, fused_ctas[8] = {};
};
V2Items fill_items_v2(const Plan& p, unsigned char* ws, bool mtfit, bool gufuse) {
    V2Items v;
    auto* xt = reinterpret_cast<_Float16*>(ws + p.off_xt);
    auto* at = reinterpret_cast<_Float16*>(ws + p.off_at);
    auto* part = reinterpret_cast<float*>(ws + p.off_part);
    for (const PGroup& g : p.groups) {
        const unsigned cls = pfk::mt_class(g.n), tm = 16U * pfk::mt_of_class(cls);
        const bool fuse = gufuse && g.s_gu == 1;
        // Splits and part pointers always describe the whole group.
        auto append = [&](pfk::Item it, bool down) {
            auto* lists = down ? v.dn : fuse ? v.fused : v.gu;
            auto* ctas = down ? v.dn_ctas : fuse ? v.fused_ctas : v.gu_ctas;
            auto emit = [&](unsigned c, unsigned row0, unsigned gy) {
                it.row0 = row0;
                it.gy = gy;
                it.cta_begin = ctas[c];
                ctas[c] += it.gx * gy * it.splits;
                lists[c].push_back(it);
                if (!down && fuse) {
                    pfk::Item up = it;
                    up.trellis = g.w.up.trellis;
                    up.x += static_cast<std::size_t>(p.tpairs) * kHidden;
                    up.part += static_cast<std::size_t>(g.n) * kInter;
                    up.bits = g.w.up.bits;
                    lists[c].push_back(up);
                }
            };
            if (mtfit) {
                const auto fit = route::mtfit(g.n);
                if (fit.main != 0) emit(7, 0, fit.main);
                if (fit.remainder != 0) emit(fit.tail_class, fit.main * 128, 1);
            } else {
                emit(pfk::mt_of_class(cls) - 1, 0, (g.n + tm - 1) / tm);
            }
        };
        for (unsigned m = 0; m < (fuse ? 1U : 2U); ++m) {
            const Exl3& w = m == 0 ? g.w.gate : g.w.up;
            pfk::Item it{};
            it.trellis = w.trellis;
            it.x = xt + (static_cast<std::size_t>(m) * p.tpairs + g.tbegin) * kHidden;
            it.part = part + g.part_gu + static_cast<std::size_t>(m) * g.s_gu * g.n * kInter;
            it.input = kHidden;
            it.output = kInter;
            it.bits = w.bits;
            it.rows = g.n;
            it.splits = g.s_gu;
            it.gx = kInter / 128;
            append(it, false);
        }
        pfk::Item it{};
        it.trellis = g.w.down.trellis;
        it.x = at + static_cast<std::size_t>(g.tbegin) * kInter;
        it.part = part + g.part_dn;
        it.input = kInter;
        it.output = kHidden;
        it.bits = g.w.down.bits;
        it.rows = g.n;
        it.splits = g.s_dn;
        it.gx = kHidden / 128;
        append(it, true);
    }
    return v;
}

// STRIX_MMX_PREFILL_TAIL_STREAM side stream (non-blocking) and its fork / join events, created on first use.
hipStream_t g_tail_side = nullptr;
hipEvent_t g_tail_forked = nullptr, g_tail_joined = nullptr;
void tail_fork(hipStream_t stream) {
    if (g_tail_side == nullptr) {  // all three or none: a failed creation leaves nothing half-initialized
        hipStream_t side = nullptr;
        hipEvent_t forked = nullptr, joined = nullptr;
        hipError_t e = hipStreamCreateWithFlags(&side, hipStreamNonBlocking);
        if (e == hipSuccess) e = hipEventCreateWithFlags(&forked, hipEventDisableTiming);
        if (e == hipSuccess) e = hipEventCreateWithFlags(&joined, hipEventDisableTiming);
        if (e != hipSuccess) {
            if (joined != nullptr) (void)hipEventDestroy(joined);
            if (forked != nullptr) (void)hipEventDestroy(forked);
            if (side != nullptr) (void)hipStreamDestroy(side);
            ok(e, "tail side stream / events");
        }
        g_tail_side = side;
        g_tail_forked = forked;
        g_tail_joined = joined;
    }
    ok(hipEventRecord(g_tail_forked, stream), "tail fork");
    ok(hipStreamWaitEvent(g_tail_side, g_tail_forked, 0), "tail fork wait");
}
void tail_join(hipStream_t stream) {
    ok(hipEventRecord(g_tail_joined, g_tail_side), "tail join");
    ok(hipStreamWaitEvent(stream, g_tail_joined, 0), "tail join wait");
}

constexpr const char* kGuLabel[8] = {"gate_up.mt1", "gate_up.mt2", "gate_up.mt3", "gate_up.mt4",
                                   "gate_up.mt5", "gate_up.mt6", "gate_up.mt7", "gate_up.mt8"};
constexpr const char* kDnLabel[8] = {"down.mt1", "down.mt2", "down.mt3", "down.mt4",
                                   "down.mt5", "down.mt6", "down.mt7", "down.mt8"};

bool prefill_p1_packed() {
    static const bool enabled = [] {
        // Default on: the packed decode is bitwise-equal to the scalar decode (exhaustive E1, K3..K6) and faster on
        // PAIR/PIPE; STRIX_PF_P1=0 restores the scalar decode. Empty means unset.
        const char* value = std::getenv("STRIX_PF_P1");
        if (value == nullptr || value[0] == '\0') return true;
        char* end = nullptr;
        errno = 0;
        const unsigned long mask = std::strtoul(value, &end, 0);
        if (value[0] < '0' || value[0] > '9' || end == value || *end != '\0' || errno == ERANGE)
            throw std::invalid_argument("STRIX_PF_P1 must be an unsigned bitmask");
        return (mask & 1UL) != 0;
    }();
    return enabled;
}

template <bool Packed>
void launch_pipe_class(const pfk::Item* items_d, unsigned n, unsigned ctas, hipStream_t stream, bool pair, bool simplify) {
    if (pair && simplify)
        hipLaunchKernelGGL(pfk::grouped_pipe_pair_kernel<true>, dim3(ctas), dim3(128), 0, stream, items_d, n);
    else if (pair)
        hipLaunchKernelGGL((pfk::grouped_pipe_pair_kernel<false, Packed>), dim3(ctas), dim3(128), 0, stream, items_d, n);
    else if (simplify)
        hipLaunchKernelGGL((pfk::grouped_pipe_kernel<8, 4, true>), dim3(ctas), dim3(128), 0, stream, items_d, n);
    else
        hipLaunchKernelGGL((pfk::grouped_pipe_kernel<8, 4, false, Packed>), dim3(ctas), dim3(128), 0, stream, items_d, n);
}

template <int W>
void launch_band_class(const pfk::Item* items_d, unsigned n, unsigned ctas, hipStream_t stream) {
    if (prefill_pipe() == 2)
        hipLaunchKernelGGL((pfk::grouped_pipe_band_kernel<W, true>), dim3(ctas), dim3(W * 32), 0, stream, items_d, n);
    else if (prefill_p1_packed())
        hipLaunchKernelGGL((pfk::grouped_pipe_band_kernel<W, false, true>), dim3(ctas), dim3(W * 32), 0, stream, items_d, n);
    else
        hipLaunchKernelGGL((pfk::grouped_pipe_band_kernel<W, false, false>), dim3(ctas), dim3(W * 32), 0, stream, items_d, n);
}

void launch_class(const pfk::Item* items_d, unsigned n, unsigned ctas, unsigned cls, hipStream_t stream, bool pair = false) {
    if (cls == 7 && prefill_pipe() != 0) {
        const bool simplify = prefill_pipe() == 2;
        if (prefill_p1_packed())
            launch_pipe_class<true>(items_d, n, ctas, stream, pair, simplify);
        else
            launch_pipe_class<false>(items_d, n, ctas, stream, pair, simplify);
        return;
    }
    if (cls == 7 && prefill_core8() != 0) {
        if (prefill_core8() == 1)
            hipLaunchKernelGGL((pfk::grouped_core8_kernel<1>), dim3(ctas), dim3(128), 0, stream, items_d, n);
        else
            hipLaunchKernelGGL((pfk::grouped_core8_kernel<2>), dim3(ctas), dim3(128), 0, stream, items_d, n);
        return;
    }
    switch (cls) {
        case 0: hipLaunchKernelGGL((pfk::grouped_kernel<1, 4>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
        case 1: hipLaunchKernelGGL((pfk::grouped_kernel<2, 4>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
        case 2: hipLaunchKernelGGL((pfk::grouped_kernel<3, 4>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
        case 3: hipLaunchKernelGGL((pfk::grouped_kernel<4, 4>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
        case 4: hipLaunchKernelGGL((pfk::grouped_kernel<5, 4>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
        case 5: hipLaunchKernelGGL((pfk::grouped_kernel<6, 4>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
        case 6: hipLaunchKernelGGL((pfk::grouped_kernel<7, 4>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
        case 7: hipLaunchKernelGGL((pfk::grouped_kernel<8, 4>), dim3(ctas), dim3(128), 0, stream, items_d, n); break;
    }
}
// pair = this MoE call's paired MT8 PIPE bands
void launch_v2(const pfk::Item* items_d, const std::vector<pfk::Item>& list, unsigned ctas, unsigned cls, hipStream_t stream, bool pair) {
    if (list.empty()) return;
    launch_class(items_d, static_cast<unsigned>(list.size()), ctas, cls, stream, pair);
}

template <int MT>
void launch_gufuse(const pfk::Item* items, unsigned count, unsigned ctas, Tables t, hipStream_t stream) {
    static const bool qualified = [] {
        hipFuncAttributes attrs{};
        ok(hipFuncGetAttributes(&attrs, reinterpret_cast<const void*>(gate_up_fused_kernel<MT>)), "gufuse resources");
        if (attrs.localSizeBytes != 0) throw std::runtime_error("minimax prefill gufuse: scratch/spilling rejected");
        if (attrs.sharedSizeBytes > 16384) throw std::runtime_error("minimax prefill gufuse: LDS budget exceeded");
        return true;
    }();
    (void)qualified;
    hipLaunchKernelGGL((gate_up_fused_kernel<MT>), dim3(ctas), dim3(256), 0, stream,
                       items, count, t, g_gufuse_debug, g_gufuse_intermediates, g_gufuse_producer_only);
}
void launch_gufuse_class(const pfk::Item* items, unsigned count, unsigned ctas, unsigned cls, Tables t, hipStream_t stream) {
    switch (cls) {
        case 0: launch_gufuse<1>(items, count, ctas, t, stream); break;
        case 1: launch_gufuse<2>(items, count, ctas, t, stream); break;
        case 2: launch_gufuse<3>(items, count, ctas, t, stream); break;
        case 3: launch_gufuse<4>(items, count, ctas, t, stream); break;
        case 4: launch_gufuse<5>(items, count, ctas, t, stream); break;
        case 5: launch_gufuse<6>(items, count, ctas, t, stream); break;
        case 6: launch_gufuse<7>(items, count, ctas, t, stream); break;
        case 7: launch_gufuse<8>(items, count, ctas, t, stream); break;
    }
}

void fill_items(Plan& p, unsigned char* ws) {
    const unsigned override_tile = prefill_tile();
    auto* xt = reinterpret_cast<_Float16*>(ws + p.off_xt);
    auto* at = reinterpret_cast<_Float16*>(ws + p.off_at);
    auto* part = reinterpret_cast<float*>(ws + p.off_part);
    const std::size_t P = p.pair_row.size();
    for (const PGroup& g : p.groups) {
        for (unsigned m = 0; m < 2; ++m) {
            const Exl3& w = m == 0 ? g.w.gate : g.w.up;
            WItem it{};
            it.trellis = w.trellis;
            it.x = xt + (static_cast<std::size_t>(m) * P + g.pair_begin) * kHidden;
            it.part = part + g.part_gu + static_cast<std::size_t>(m) * g.s_gu * g.n * kInter;
            it.input = kHidden;
            it.output = kInter;
            it.bits = w.bits;
            it.rows = g.n;
            it.splits = g.s_gu;
            it.gx = (kInter / 16 + 3) / 4;
            it.gy = (g.n + (override_tile ? override_tile : 16U) - 1) / (override_tile ? override_tile : 16U);
            it.cta_begin = p.gu_ctas;
            p.gu_ctas += it.gx * it.gy * it.splits;
            p.gu_items.push_back(it);
        }
        const unsigned tile = override_tile ? override_tile : gemm_tile_rows(kInter, kHidden, g.w.down.bits, g.n);
        WItem it{};
        it.trellis = g.w.down.trellis;
        it.x = at + static_cast<std::size_t>(g.pair_begin) * kInter;
        it.part = part + g.part_dn;
        it.input = kInter;
        it.output = kHidden;
        it.bits = g.w.down.bits;
        it.rows = g.n;
        it.splits = g.s_dn;
        it.gx = (kHidden / 16 + 3) / 4;
        it.gy = (g.n + tile - 1) / tile;
        unsigned& ctas = tile != 16 ? p.dn_ctas64 : p.dn_ctas16;
        it.cta_begin = ctas;
        ctas += it.gx * it.gy * it.splits;
        (tile != 16 ? p.dn_items64 : p.dn_items16).push_back(it);
    }
}

// ---------------------------------------------------------------- device plan (STRIX_MMX_PF_PLAN, default device)
// The v4 + mtfit tables of make_plan / fill_items_v2 built on the GPU from the device routes by the shared planner
// (moe_core/device_plan.hpp): no route download, host sync, host table build or table upload per MoE call. Groups are
// not compacted (group x = expert x, shared = kExperts; an absent expert has n = 0 and no pairs or items), the
// tile-major stride `tpairs` is the bound 5T + 15 x 129, and every (stage, class) item list has a fixed slot; the pair,
// token, group-member and item contents are those of the host plan, so every CTA does the host CTA's work.
namespace dp = ::strix::moe_core::device_plan;
using Queue = dp::ItemQueue<pfk::Item>;
// Group-wide plan (enqueue_prefill_moe_group): one call over the chunks of a prefill group. A row's bits depend on the
// K split count of its item, and the adopted policy picks the splits from the rows an expert holds in the row's own
// chunk. The group-wide plan therefore merges, per expert, the chunks' rows by split class: every (chunk, expert) row
// set joins the group (expert, class of its chunk-local count), which runs with that class's splits. Rows keep their
// per-chunk arithmetic (same splits, same K ranges, row-independent fragments) while an expert's weights are decoded
// once for the group's rows of a class instead of once per chunk.
constexpr unsigned kGwClasses = 5;
constexpr unsigned kGwSplitGu[kGwClasses] = {1, 2, 4, 8, 16}, kGwSplitDn[kGwClasses] = {1, 1, 1, 2, 4};
constexpr unsigned gw_class(unsigned n) { return n >= 97 ? 0U : n >= 49 ? 1U : n >= 17 ? 2U : n >= 9 ? 3U : 4U; }
constexpr bool gw_classes_match_policy() {
    for (unsigned n = 1; n <= 300; ++n)
        for (const unsigned bits : {3U, 4U, 5U, 6U, 8U})
            if (route::gemm_splits(kHidden, kInter, bits, n) != kGwSplitGu[gw_class(n)] ||
                route::gemm_splits(kInter, kHidden, bits, n) != kGwSplitDn[gw_class(n)])
                return false;
    return true;
}
static_assert(gw_classes_match_policy(), "group-wide split classes must follow route::gemm_splits");
// C = split classes per expert: 1 = the per-chunk plan (129 groups), kGwClasses = the group-wide plan (641 groups).
template <unsigned C> struct DevDims {
    static constexpr unsigned kVirtual = C * kExperts, kGroups = kVirtual + 1;
    static constexpr unsigned kGuSlot = 2 * kGroups, kDnSlot = kGroups;  // items per class: gate+up per group / down
    static constexpr unsigned kItems = 8 * kGuSlot + 8 * kDnSlot;
};
constexpr unsigned kDevGroups = DevDims<1>::kGroups;
constexpr unsigned kGuSlot = DevDims<1>::kGuSlot, kDnSlot = DevDims<1>::kDnSlot;
struct DevMeta {  // [0..7] gate/up classes, [8..15] down classes
    unsigned count[16], total[16];
};
struct DevOut {
    PGroup* groups;
    std::uint32_t* pair_row;
    std::uint32_t* pair_group;
    std::uint32_t* tok_pair;
    float* tok_w;
    pfk::Item* items;
    DevMeta* meta;
    const ExpertSlices* experts;  // device [kExperts]
    ExpertSlices shared;
    _Float16* xt;
    _Float16* at;
    float* part;
    unsigned tpairs;
    bool i4=false;
    unsigned grid[16];  // the launch grid of every (stage, class): a bound of its CTAs (class_grids)
    // Group-wide plan only: the chunks' router tables ([n][4] ids, then [n][4] weights, n = the chunk's rows) and the
    // per-chunk expert counts [segments][128] of seg_count_kernel.
    const std::uint32_t* seg_routes[kPrefillMoeSegments];
    unsigned* seg_count;
    unsigned seg_rows, segments;
};
template <unsigned C> struct DevPolicyT {
    static constexpr unsigned kReal = ::strix::minimax::moe::opt::kExperts;
    static constexpr unsigned kExperts = DevDims<C>::kVirtual, kGroups = DevDims<C>::kGroups, kTopK = kTopE, kRows = 256;
    static constexpr unsigned kGuSlot = DevDims<C>::kGuSlot, kDnSlot = DevDims<C>::kDnSlot;
    static constexpr bool kDropInvalid = false;  // keep 5T pairs: an invalid row plans experts 0..3 with weight 0
    using Out = DevOut;
    using A = dp::Args<DevPolicyT>;
    // The token order of make_plan: the row's 4 (id, weight) in ascending id (a sorting network; the ids of a valid row
    // are distinct, so the order is that of the insertion sort). `seg` = the row's chunk (0 for the per-chunk plan).
    __device__ static bool load_real(const A& p, unsigned r, unsigned (&e)[kTopE], float (&w)[kTopE], unsigned& seg) {
        const std::uint32_t* ids = p.ids;
        const float* weights = p.weights;
        unsigned local = r;
        seg = 0;
        if constexpr (C > 1) {
            seg = r / p.out.seg_rows;
            local = r - seg * p.out.seg_rows;
            const unsigned n = seg + 1U == p.out.segments ? p.rows - seg * p.out.seg_rows : p.out.seg_rows;
            ids = p.out.seg_routes[seg];
            weights = reinterpret_cast<const float*>(ids + std::size_t{n} * kTopE);
        }
#pragma unroll
        for (unsigned k = 0; k < kTopE; ++k) {
            e[k] = ids[std::size_t{local} * kTopE + k];
            w[k] = weights[std::size_t{local} * kTopE + k];
        }
        auto cs = [&](unsigned i, unsigned j) {
            if (e[i] > e[j]) {
                const unsigned te = e[i];
                e[i] = e[j];
                e[j] = te;
                const float tw = w[i];
                w[i] = w[j];
                w[j] = tw;
            }
        };
        cs(0, 1);
        cs(2, 3);
        cs(0, 2);
        cs(1, 3);
        cs(1, 2);
        return e[3] >= kReal || e[0] == e[1] || e[1] == e[2] || e[2] == e[3];
    }
    // Planner ids. Group-wide: expert e of a chunk plans as (class of the chunk's count of e) x 128 + e, in the row's
    // real-id slot order (the combine order of the per-chunk plan). The fallback ids are mapped here as well, so
    // fallback() below has nothing left to do.
    __device__ static bool load(const A& p, unsigned r, unsigned (&e)[kTopE], float (&w)[kTopE]) {
        unsigned seg = 0;
        const bool bad = load_real(p, r, e, w, seg);
        if constexpr (C > 1) {
#pragma unroll
            for (unsigned k = 0; k < kTopE; ++k) {
                if (bad) e[k] = k, w[k] = 0.0F;
                e[k] += gw_class(p.out.seg_count[seg * kReal + e[k]]) * kReal;
            }
        }
        return bad;
    }
    __device__ static void fallback(unsigned (&e)[kTopE], float (&w)[kTopE]) {
        if constexpr (C == 1) {
#pragma unroll
            for (unsigned k = 0; k < kTopE; ++k) e[k] = k, w[k] = 0.0F;
        } else {
            (void)e;
            (void)w;
        }
    }
    // The groups of make_plan and the v4 + mtfit items of fill_items_v2 (gufuse off), in their order: thread x sizes
    // group x, exclusive scans over the groups (row tiles, partial floats, per-class item counts and CTAs), then
    // thread x writes its group and items.
    __device__ static void finish(const A& p, const unsigned* count, const unsigned* start) {
        if(p.out.i4) {
            const unsigned e=threadIdx.x;
            if(e<kGroups) {
                PGroup g{}; g.expert=e; g.n=count[e]; g.pair_begin=start[e];
                g.w=e<kExperts ? p.out.experts[e % kReal] : p.out.shared;
                p.out.groups[e]=g;
                // Reuse the item arena for the library's device group list.
                auto* tab=reinterpret_cast<unsigned*>(p.out.items);
                tab[3*e]=e; tab[3*e+1]=start[e]; tab[3*e+2]=count[e];
            }
            return;
        }
        // Per group and stage (0 gate/up, 1 down): [stage] = its MT8 items, [2 + stage] = its tail items. In: the CTAs
        // of the stage's items of that kind; after the scans: their CTA prefix (cta) and item prefix (slot) inside
        // their class queue. A group's MT8 and tail items share class 7 when its remainder is 113..127 rows.
        __shared__ unsigned cta[4][kGroups], slot[4][kGroups], tile_rows[kGroups + 1];
        __shared__ unsigned char tail_cls[kGroups], has_main[kGroups];
        __shared__ std::size_t floats[kGroups + 1];
        const unsigned x = threadIdx.x;
        const bool mine = x < kGroups;
        const unsigned n = mine ? count[x] : 0U;
        ExpertSlices w{};
        unsigned s_gu = 0, s_dn = 0;
        route::MtFit fit{};
        if (mine) {
            w = x < kExperts ? p.out.experts[x % kReal] : p.out.shared;
            if (C > 1 && x < kExperts) {
                s_gu = n ? kGwSplitGu[x / kReal] : 0U;
                s_dn = n ? kGwSplitDn[x / kReal] : 0U;
            } else {
                s_gu = n ? route::gemm_splits(kHidden, kInter, w.gate.bits, n) : 0U;
                s_dn = n ? route::gemm_splits(kInter, kHidden, w.down.bits, n) : 0U;
            }
            fit = route::mtfit(n);
            tile_rows[x] = (n + 15U) / 16U * 16U;
            floats[x] = 2 * static_cast<std::size_t>(s_gu) * n * kInter + static_cast<std::size_t>(s_dn) * n * kHidden;
            // gate/up: 2 items per used class (gate, up); down: 1. gx = 12 / 48.
            has_main[x] = fit.main != 0 ? 1 : 0;
            tail_cls[x] = static_cast<unsigned char>(fit.remainder != 0 ? fit.tail_class : 255U);
            cta[0][x] = 2 * (kInter / 128) * fit.main * s_gu;
            cta[1][x] = (kHidden / 128) * fit.main * s_dn;
            cta[2][x] = 2 * (kInter / 128) * s_gu;
            cta[3][x] = (kHidden / 128) * s_dn;
        }
        __syncthreads();
        // Exclusive scans down the groups: thread q = class queue q (stage q / 8, class q % 8).
        if (x < 16) {
            const unsigned st = x >> 3U, cls = x & 7U, per = st ? 1U : 2U;
            unsigned c = 0, t = 0;
            for (unsigned g = 0; g < kGroups; ++g) {
                if (cls == 7U && has_main[g] != 0) {
                    const unsigned d = cta[st][g];
                    cta[st][g] = t;
                    slot[st][g] = c;
                    c += per;
                    t += d;
                }
                if (tail_cls[g] == cls) {
                    const unsigned d = cta[2U + st][g];
                    cta[2U + st][g] = t;
                    slot[2U + st][g] = c;
                    c += per;
                    t += d;
                }
            }
            p.out.meta->count[x] = c;
            p.out.meta->total[x] = t;
            if (t > p.out.grid[x]) atomicOr(p.flag, 2U);  // above the launch grid (bug)
        } else if (x == 16) {
            unsigned t = 0;
            for (unsigned g = 0; g <= kGroups; ++g) { const unsigned d = g < kGroups ? tile_rows[g] : 0U; tile_rows[g] = t; t += d; }
        } else if (x == 17) {
            std::size_t t = 0;
            for (unsigned g = 0; g <= kGroups; ++g) { const std::size_t d = g < kGroups ? floats[g] : 0U; floats[g] = t; t += d; }
        }
        __syncthreads();
        if (mine) {
            PGroup g{};
            g.w = w;
            g.n = n;
            g.pair_begin = start[x];
            g.tbegin = tile_rows[x];
            g.s_gu = s_gu;
            g.s_dn = s_dn;
            g.part_gu = floats[x];
            g.part_dn = floats[x] + 2 * static_cast<std::size_t>(s_gu) * n * kInter;
            p.out.groups[x] = g;
            if (n != 0) {
                // Matrix m of a stage; a queue holds a group's items as gate main, gate tail, up main, up tail.
                const bool both = fit.main != 0 && fit.remainder != 0 && fit.tail_class == 7U;
                auto append = [&](pfk::Item it, unsigned stage, unsigned m) {
                    const unsigned main_ctas = it.gx * fit.main * it.splits, tail_ctas = it.gx * it.splits;
                    auto emit = [&](unsigned c, unsigned row0, unsigned gy, unsigned at, unsigned cta_begin) {
                        it.row0 = row0;
                        it.gy = gy;
                        it.cta_begin = cta_begin;
                        p.out.items[(stage ? 8U * kGuSlot + c * kDnSlot : c * kGuSlot) + at] = it;
                    };
                    if (fit.main != 0)
                        emit(7, 0, fit.main, slot[stage][x] + m * (both ? 2U : 1U),
                             cta[stage][x] + m * (main_ctas + (both ? tail_ctas : 0U)));
                    if (fit.remainder != 0) {
                        if (both)
                            emit(7, fit.main * 128, 1, slot[stage][x] + 1U + 2U * m,
                                 cta[stage][x] + main_ctas + m * (main_ctas + tail_ctas));
                        else
                            emit(fit.tail_class, fit.main * 128, 1, slot[2U + stage][x] + m, cta[2U + stage][x] + m * tail_ctas);
                    }
                };
                for (unsigned m = 0; m < 2U; ++m) {
                    const Exl3& mw = m == 0 ? w.gate : w.up;
                    pfk::Item it{};
                    it.trellis = mw.trellis;
                    it.x = p.out.xt + (static_cast<std::size_t>(m) * p.out.tpairs + g.tbegin) * kHidden;
                    it.part = p.out.part + g.part_gu + static_cast<std::size_t>(m) * s_gu * n * kInter;
                    it.input = kHidden;
                    it.output = kInter;
                    it.bits = mw.bits;
                    it.rows = n;
                    it.splits = s_gu;
                    it.gx = kInter / 128;
                    append(it, 0, m);
                }
                pfk::Item it{};
                it.trellis = w.down.trellis;
                it.x = p.out.at + static_cast<std::size_t>(g.tbegin) * kInter;
                it.part = p.out.part + g.part_dn;
                it.input = kInter;
                it.output = kHidden;
                it.bits = w.down.bits;
                it.rows = n;
                it.splits = s_dn;
                it.gx = kHidden / 128;
                append(it, 1, 0);
            }
        }
    }
    __device__ static void routed(const A& p, unsigned r, unsigned k, unsigned e, float w, unsigned pos) {
        p.out.pair_row[pos] = r;
        p.out.pair_group[pos] = e;
        p.out.tok_pair[std::size_t{r} * kTopE + k] = pos;
        p.out.tok_w[std::size_t{r} * kTopE + k] = w;
    }
    __device__ static void shared(const A& p, unsigned r, unsigned pos) {
        p.out.pair_row[pos] = r;
        p.out.pair_group[pos] = kExperts;
    }
};
static_assert(DevDims<1>::kItems == 24U * DevDims<1>::kGroups && DevDims<kGwClasses>::kItems == 24U * DevDims<kGwClasses>::kGroups);
using DevPolicy = DevPolicyT<1>;
using GwPolicy = DevPolicyT<kGwClasses>;

// Per-chunk expert counts of the group-wide plan: [segments][128], zeroed by the caller. One thread per row.
__global__ __launch_bounds__(256) void seg_count_kernel(dp::Args<GwPolicy> p) {
    const unsigned r = blockIdx.x * 256U + threadIdx.x;
    if (r >= p.rows) return;
    unsigned e[kTopE], seg = 0;
    float w[kTopE];
    const bool bad = GwPolicy::load_real(p, r, e, w, seg);
#pragma unroll
    for (unsigned k = 0; k < kTopE; ++k) atomicAdd(&p.out.seg_count[seg * GwPolicy::kReal + (bad ? k : e[k])], 1U);
}

// Device-plan workspace of a T-row call (a function of T only).
struct DevLayout {
    unsigned tpairs = 0;
    std::size_t groups = 0, pair_row = 0, pair_group = 0, tok_pair = 0, tok_w = 0, items = 0, meta = 0, scratch = 0, xt = 0,
                at = 0, part = 0, bytes = 0, seg_count = 0;
};
// Partial floats of any routes of T rows: the sum over the 129 groups of 2 s_gu(n) n 1536 + s_dn(n) n 6144 is at most
// 5T c + 129 max_n (f(n) - c n), c = 2 x 1536 + 6144 (the splits are 1 for large n). Bits do not enter the policy at
// these shapes; the bound still takes the maximum over them.
std::size_t dev_part_floats(unsigned T, unsigned groups = kDevGroups) {
    static unsigned cached_t = 0, cached_groups = 0;
    static std::size_t cached = 0;
    if (cached_t == T && cached_groups == groups) return cached;
    constexpr std::size_t c = 2ULL * kInter + kHidden;
    std::size_t extra = 0;
    for (unsigned n = 1; n <= T; ++n)
        for (const unsigned bits : {3U, 4U, 5U, 6U, 8U}) {
            const std::size_t f = 2ULL * route::gemm_splits(kHidden, kInter, bits, n) * n * kInter +
                                  std::size_t{route::gemm_splits(kInter, kHidden, bits, n)} * n * kHidden;
            if (f > c * n) extra = std::max(extra, f - c * n);
        }
    cached = 5ULL * T * c + std::size_t{groups} * extra;
    cached_t = T;
    cached_groups = groups;
    return cached;
}
// classes = 1: the per-chunk plan; kGwClasses: the group-wide plan (641 groups, their item queues and seg_count).
DevLayout dev_layout(unsigned T,bool i4=false,unsigned classes=1) {
    DevLayout l;
    const unsigned groups = classes * kExperts + 1U;
    l.tpairs = (5U * T + 15U * groups + 15U) / 16U * 16U;
    std::size_t off = 0;
    auto take = [&](std::size_t bytes) {
        const std::size_t at = off;
        off += align256(bytes);
        return at;
    };
    l.groups = take(groups * sizeof(PGroup));
    l.pair_row = take(5ULL * T * 4);
    l.pair_group = take(5ULL * T * 4);
    l.tok_pair = take(4ULL * T * 4);
    l.tok_w = take(4ULL * T * 4);
    l.items = take(std::size_t{24U * groups} * sizeof(pfk::Item));
    l.meta = take(sizeof(DevMeta));
    l.scratch = take(classes == 1 ? dp::scratch_bytes<DevPolicy>(T) : dp::scratch_bytes<GwPolicy>(T));
    if (classes != 1) l.seg_count = take(std::size_t{kPrefillMoeSegments} * kExperts * sizeof(unsigned));
    if(i4) {
        l.xt=off;
        // Largest tile list (BM32), count word, the global down operand (A4 + FP32 scales) and the
        // FP32 down rows of the ordered (deterministic) combine, then the gate/up A4 image
        // (row 2*pair+m, quantised once by the pre-pass) with its FP32 scales.
        take(align256(((5ULL*T+31)/32+kDevGroups)*12)+256+
             align256(5ULL*T*(kInter/2))+align256(5ULL*T*(kInter/64)*4)+align256(5ULL*T*kHidden*4)+
             align256(2ULL*5ULL*T*(kHidden/2))+align256(2ULL*5ULL*T*(kHidden/64)*4)+
             (strix::core::moe_i4::gsave_enabled() ? align256(5ULL*T*kInter*4) : 0));  // GSave: FP32 pass-0 gate of the paired chain GEMM
        l.bytes=off;
        return l;
    }
    l.xt = take(2ULL * l.tpairs * kHidden * 2);
    l.at = take(std::size_t{l.tpairs} * kInter * 2);
    l.part = take(dev_part_floats(T, groups) * 4);
    l.bytes = off;
    return l;
}

unsigned* g_dev_flag = nullptr;  // sticky invalid-route flag of device-planned calls (prefill_moe_device_check)
unsigned* dev_flag() {
    if (g_dev_flag == nullptr) {
        unsigned* f = nullptr;
        ok(hipMalloc(reinterpret_cast<void**>(&f), sizeof(unsigned)), "device plan flag");
        if (hipMemset(f, 0, sizeof(unsigned)) != hipSuccess) {
            (void)hipFree(f);
            throw std::runtime_error("minimax prefill moe: device plan flag init");
        }
        g_dev_flag = f;
    }
    return g_dev_flag;
}

#define STRIX_MMX_BITS_SWITCH(CALL) \
    switch (it.bits) {              \
        case 3: CALL(3); break;     \
        case 4: CALL(4); break;     \
        case 5: CALL(5); break;     \
        case 6: CALL(6); break;     \
        default: CALL(8); break;    \
    }
// grouped_kernel<MT, 4> (tail classes MT 1..7, and MT8 without PIPE / core8).
template <int MT> __global__ __launch_bounds__(128) void dq_v4_kernel(Queue q) {
    dp::run_items(q, [](const pfk::Item& it, unsigned bx, unsigned by, unsigned bz) {
#define STRIX_MMX_CALL(B) pfk::item_core_v3<B, MT, 0, true>(it, bx, by, bz)
        STRIX_MMX_BITS_SWITCH(STRIX_MMX_CALL)
#undef STRIX_MMX_CALL
    });
}
template <bool Simplify, bool Packed> __global__ __launch_bounds__(128) void dq_pipe_kernel(Queue q) {
    dp::run_items(q, [](const pfk::Item& it, unsigned bx, unsigned by, unsigned bz) {
#define STRIX_MMX_CALL(B) pfk::pipe_or_tail<B, 8, 4, Simplify, Packed>(it, bx, by, bz)
        STRIX_MMX_BITS_SWITCH(STRIX_MMX_CALL)
#undef STRIX_MMX_CALL
    });
}
template <bool Simplify, bool Packed> __global__ __launch_bounds__(128) void dq_pipe_pair_kernel(Queue q) {
    __shared__ pfk::Half16 exchange[128];  // as grouped_pipe_pair_kernel
    dp::run_items(q, [&](const pfk::Item& it, unsigned bx, unsigned by, unsigned bz) {
#define STRIX_MMX_CALL(B) pfk::pipe_pair_or_tail<B, Simplify, Packed>(it, bx, by, bz, exchange)
        STRIX_MMX_BITS_SWITCH(STRIX_MMX_CALL)
#undef STRIX_MMX_CALL
    });
}
template <int Core8> __global__ __launch_bounds__(128) void dq_core8_kernel(Queue q) {
    __shared__ pfk::Half16 exchange[256];  // as grouped_core8_kernel
    dp::run_items(q, [&](const pfk::Item& it, unsigned bx, unsigned by, unsigned bz) {
#define STRIX_MMX_CALL(B) pfk::item_core8<B, Core8>(it, bx, by, bz, exchange)
        STRIX_MMX_BITS_SWITCH(STRIX_MMX_CALL)
#undef STRIX_MMX_CALL
    });
}
#undef STRIX_MMX_BITS_SWITCH

void launch_dq_class(const Queue& q, unsigned cls, unsigned grid, hipStream_t stream, bool pair) {
    const dim3 g(grid);
    switch (cls) {
        case 0: hipLaunchKernelGGL(dq_v4_kernel<1>, g, dim3(128), 0, stream, q); return;
        case 1: hipLaunchKernelGGL(dq_v4_kernel<2>, g, dim3(128), 0, stream, q); return;
        case 2: hipLaunchKernelGGL(dq_v4_kernel<3>, g, dim3(128), 0, stream, q); return;
        case 3: hipLaunchKernelGGL(dq_v4_kernel<4>, g, dim3(128), 0, stream, q); return;
        case 4: hipLaunchKernelGGL(dq_v4_kernel<5>, g, dim3(128), 0, stream, q); return;
        case 5: hipLaunchKernelGGL(dq_v4_kernel<6>, g, dim3(128), 0, stream, q); return;
        case 6: hipLaunchKernelGGL(dq_v4_kernel<7>, g, dim3(128), 0, stream, q); return;
        default: break;
    }
    // Class 7: the kernel that launch_class<4> selects, from the same switches.
    if (prefill_pipe() != 0) {
        const bool simplify = prefill_pipe() == 2, packed = prefill_p1_packed();
        if (pair && simplify) hipLaunchKernelGGL((dq_pipe_pair_kernel<true, false>), g, dim3(128), 0, stream, q);
        else if (pair && packed) hipLaunchKernelGGL((dq_pipe_pair_kernel<false, true>), g, dim3(128), 0, stream, q);
        else if (pair) hipLaunchKernelGGL((dq_pipe_pair_kernel<false, false>), g, dim3(128), 0, stream, q);
        else if (simplify) hipLaunchKernelGGL((dq_pipe_kernel<true, false>), g, dim3(128), 0, stream, q);
        else if (packed) hipLaunchKernelGGL((dq_pipe_kernel<false, true>), g, dim3(128), 0, stream, q);
        else hipLaunchKernelGGL((dq_pipe_kernel<false, false>), g, dim3(128), 0, stream, q);
        return;
    }
    if (prefill_core8() == 1) hipLaunchKernelGGL((dq_core8_kernel<1>), g, dim3(128), 0, stream, q);
    else if (prefill_core8() == 2) hipLaunchKernelGGL((dq_core8_kernel<2>), g, dim3(128), 0, stream, q);
    else hipLaunchKernelGGL((dq_v4_kernel<8>), g, dim3(128), 0, stream, q);
}
// CTA bound of every (stage, class) for any routes of T rows: the launch grids (surplus CTAs return at once).
// MT8: an MT8 item is the 128-row tiles of a group (floor(n / 128)) or its remainder when that is >= 113 rows, both with
// splits 1 (n >= 113); floor(n / 128) + [n % 128 >= 113] <= n / 113 and the groups hold 5T rows, so these items have at
// most gx x matrices x floor(5T / 113) CTAs. Tail items of class c: at most one per matrix and group, so 129 groups x
// gx x matrices x the largest split count of a group size whose remainder falls in class c.
// Group-wide plan (classes > 1, `segs` chunks): a group (expert, class k) holds at most segs x (the class's largest
// chunk count) rows and runs with the class's splits; class 0 and the shared expert keep the bound above.
std::array<unsigned, 16> class_grids(unsigned T, unsigned classes = 1, unsigned segs = 1) {
    static unsigned cached_t = 0, cached_classes = 0, cached_segs = 0;
    static std::array<unsigned, 16> cached{};
    if (cached_t == T && cached_classes == classes && cached_segs == segs) return cached;
    std::array<unsigned, 16> g{};
    const unsigned tiles = std::max(1U, 5U * T / 113U);
    if (classes == 1) {
        for (unsigned n = 1; n <= T; ++n) {
            const auto fit = route::mtfit(n);
            if (fit.remainder == 0) continue;
            for (const unsigned bits : {3U, 4U, 5U, 6U, 8U}) {
                g[fit.tail_class] = std::max(g[fit.tail_class], 2U * (kInter / 128) * route::gemm_splits(kHidden, kInter, bits, n));
                g[8 + fit.tail_class] = std::max(g[8 + fit.tail_class], (kHidden / 128) * route::gemm_splits(kInter, kHidden, bits, n));
            }
        }
        for (unsigned c = 0; c < 16; ++c) g[c] *= kDevGroups;
        g[7] += 2U * (kInter / 128) * tiles;
        g[15] += (kHidden / 128) * tiles;
    } else {
        constexpr unsigned kChunkMax[kGwClasses] = {0, 96, 48, 16, 8};  // largest chunk count of a class (0: unbounded)
        for (unsigned k = 0; k < kGwClasses; ++k) {
            const unsigned nmax = k == 0 ? T : std::min(T, segs * kChunkMax[k]);
            bool tail[8] = {};
            for (unsigned n = 1; n <= nmax; ++n) {
                const auto fit = route::mtfit(n);
                if (fit.remainder != 0) tail[fit.tail_class] = true;
            }
            const unsigned groups = kExperts + (k == 0 ? 1U : 0U);  // class 0 also bounds the shared expert's tail
            for (unsigned c = 0; c < 8; ++c) {
                if (!tail[c]) continue;
                g[c] += groups * 2U * (kInter / 128) * kGwSplitGu[k];
                g[8 + c] += groups * (kHidden / 128) * kGwSplitDn[k];
            }
            const unsigned main = k == 0 ? tiles : std::min(tiles, kExperts * (nmax / 128U));
            g[7] += 2U * (kInter / 128) * main * kGwSplitGu[k];
            g[15] += (kHidden / 128) * main * kGwSplitDn[k];
        }
    }
    for (auto& v : g) v = std::max(v, 1U);
    cached = g;
    cached_t = T;
    cached_classes = classes;
    cached_segs = segs;
    return cached;
}

#if STRIX_MMX_IU4_LIB
void launch_i4_chain(const Tables&,const I4Layer&,const float*,float*,const strix::core::IU4Group*,unsigned char*,std::size_t,hipStream_t,bool);
#endif
// The launches of a device-planned call over its tables: gather, gate/up classes, finish, down classes, combine.
// The class queues hold gu_slot / dn_slot items (DevDims of the plan).
void launch_device_planned(const Tables& t, const float* y, float* out, pfk::Item* items, DevMeta* meta,
                           const std::array<unsigned, 16>& grids, unsigned gu_slot, unsigned dn_slot, bool token_gather,
                           hipStream_t stream) {
    const unsigned T = t.rows;
    if (token_gather) hipLaunchKernelGGL(gather_transform_token_kernel, dim3((T * kHB + 7) / 8), dim3(256), 0, stream, y, t);
    else hipLaunchKernelGGL(gather_transform_tiled_kernel, dim3((t.pairs * 2 * kHB + 7) / 8), dim3(256), 0, stream, y, t);
    launched("gather transform");
    mark("gather", stream);
    const bool pair = prefill_pipe_pair(T), side = prefill_tail_stream();
        auto queue = [&](unsigned stage, unsigned c) {
        const unsigned qi = stage * 8U + c;
        return Queue{items + (stage ? 8U * gu_slot + c * dn_slot : c * gu_slot), &meta->count[qi], &meta->total[qi]};
    };
    // As the stage() of enqueue_prefill_moe: tails (MT 1..7) on the side stream beside MT8, joined before the consumer.
    auto stage = [&](unsigned s, const char* const* labels) {
        if (side) tail_fork(stream);
        try {
            for (unsigned c = 0; c < 7; ++c) {
                launch_dq_class(queue(s, c), c, grids[s * 8U + c], side ? g_tail_side : stream, pair);
                if (!side) mark(labels[c], stream);
            }
            launch_dq_class(queue(s, 7), 7, grids[s * 8U + 7], stream, pair);
            mark(labels[7], stream);
        } catch (...) {
            if (side) tail_join(stream);
            throw;
        }
        if (side) {
            tail_join(stream);
            mark(labels[6], stream);
        }
    };
    stage(0, kGuLabel);
    launched("gate/up device plan");
    hipLaunchKernelGGL(gate_up_finish_tiled_kernel, dim3((t.pairs * kIB + 7) / 8), dim3(256), 0, stream, t, false, g_gufuse_intermediates);
    launched("gate/up finish");
    mark("finish", stream);
    stage(1, kDnLabel);
    launched("down device plan");
    hipLaunchKernelGGL(down_combine_kernel, dim3((T * kHB + 7) / 8), dim3(256), 0, stream, t, out);
    launched("down combine");
    mark("combine", stream);
}

void enqueue_device_moe(const PrefillMoeDeviceView& v, void* workspace, std::size_t workspace_bytes, hipStream_t stream) {
    const unsigned T = v.rows;
    if (T == 0 || v.y == nullptr || v.routes == nullptr || v.experts == nullptr || v.out == nullptr)
        throw std::invalid_argument("minimax prefill moe: invalid device view");
    const DevLayout l = dev_layout(T,v.i4!=nullptr);
    if (workspace == nullptr || workspace_bytes < l.bytes) throw std::invalid_argument("minimax prefill moe: workspace too small");
    auto* ws = static_cast<unsigned char*>(workspace);
    dp::Args<DevPolicy> a{};
    a.rows = T;
    a.ids = v.routes;
    a.weights = reinterpret_cast<const float*>(v.routes + std::size_t{T} * kTopE);
    dp::bind_scratch(a, ws + l.scratch, T);
    a.flag = dev_flag();
    a.out.groups = reinterpret_cast<PGroup*>(ws + l.groups);
    a.out.pair_row = reinterpret_cast<std::uint32_t*>(ws + l.pair_row);
    a.out.pair_group = reinterpret_cast<std::uint32_t*>(ws + l.pair_group);
    a.out.tok_pair = reinterpret_cast<std::uint32_t*>(ws + l.tok_pair);
    a.out.tok_w = reinterpret_cast<float*>(ws + l.tok_w);
    a.out.items = reinterpret_cast<pfk::Item*>(ws + l.items);
    a.out.meta = reinterpret_cast<DevMeta*>(ws + l.meta);
    a.out.experts = v.experts;
    a.out.shared = v.shared;
    a.out.i4 = v.i4 != nullptr;
    a.out.xt = reinterpret_cast<_Float16*>(ws + l.xt);
    a.out.at = reinterpret_cast<_Float16*>(ws + l.at);
    a.out.part = reinterpret_cast<float*>(ws + l.part);
    a.out.tpairs = l.tpairs;
    const auto grids = class_grids(T);
    for (unsigned c = 0; c < 16; ++c) a.out.grid[c] = grids[c];
    mark("start", stream);
    ok(dp::enqueue(a, stream), "device plan");
    Tables t{};
    t.groups = a.out.groups;
    t.pair_row = a.out.pair_row;
    t.pair_group = a.out.pair_group;
    t.tok_pair = a.out.tok_pair;
    t.tok_w = a.out.tok_w;
    t.xt = a.out.xt;
    t.at = a.out.at;
    t.part = a.out.part;
    t.pairs = 5U * T;
    t.rows = T;
    t.shared_group = kExperts;
    t.tpairs = l.tpairs;
    // The device planner historically ignores the environment fakeq switch.
    // Only an explicit per-call override changes its arithmetic.
    t.fakeq = v.i4 == nullptr && v.fakeq_a4 == PrefillFakeq::on ? 1U : 0U;
#if STRIX_MMX_IU4_LIB
    if(v.i4) {
        if(!v.i4_all || !v.i4_chain) throw std::invalid_argument("device i4 needs the all-scope candidate chain");
        launch_i4_chain(t,*v.i4,v.y,v.out,reinterpret_cast<const strix::core::IU4Group*>(a.out.items),
            ws+l.xt,workspace_bytes-l.xt,stream,true);
        return;
    }
#endif
    launch_device_planned(t, v.y, v.out, a.out.items, a.out.meta, grids, kGuSlot, kDnSlot,
                          v.fakeq_a4 == PrefillFakeq::on || prefill_gather_tok(), stream);
}

// Group-wide call: seg_count -> the planner over all rows with the split-class policy -> the same launches.
void enqueue_group_moe(const PrefillMoeGroupView& v, void* workspace, std::size_t workspace_bytes, hipStream_t stream) {
    const unsigned T = v.rows, S = v.segments;
    if (S == 0 || S > kPrefillMoeSegments || v.seg_rows == 0 || T <= (S - 1U) * v.seg_rows || T > S * v.seg_rows ||
        T - (S - 1U) * v.seg_rows < prefill_moe_group_min_rows() || v.experts == nullptr)
        throw std::invalid_argument("minimax prefill moe: invalid group view");
    for (unsigned c = 0; c < S; ++c)
        if (v.y[c] == nullptr || v.routes[c] == nullptr || v.out[c] == nullptr)
            throw std::invalid_argument("minimax prefill moe: invalid group view");
    const DevLayout l = dev_layout(T, false, kGwClasses);
    if (workspace == nullptr || workspace_bytes < l.bytes) throw std::invalid_argument("minimax prefill moe: workspace too small");
    auto* ws = static_cast<unsigned char*>(workspace);
    dp::Args<GwPolicy> a{};
    a.rows = T;
    dp::bind_scratch(a, ws + l.scratch, T);
    a.flag = dev_flag();
    a.out.groups = reinterpret_cast<PGroup*>(ws + l.groups);
    a.out.pair_row = reinterpret_cast<std::uint32_t*>(ws + l.pair_row);
    a.out.pair_group = reinterpret_cast<std::uint32_t*>(ws + l.pair_group);
    a.out.tok_pair = reinterpret_cast<std::uint32_t*>(ws + l.tok_pair);
    a.out.tok_w = reinterpret_cast<float*>(ws + l.tok_w);
    a.out.items = reinterpret_cast<pfk::Item*>(ws + l.items);
    a.out.meta = reinterpret_cast<DevMeta*>(ws + l.meta);
    a.out.experts = v.experts;
    a.out.shared = v.shared;
    a.out.xt = reinterpret_cast<_Float16*>(ws + l.xt);
    a.out.at = reinterpret_cast<_Float16*>(ws + l.at);
    a.out.part = reinterpret_cast<float*>(ws + l.part);
    a.out.tpairs = l.tpairs;
    a.out.seg_count = reinterpret_cast<unsigned*>(ws + l.seg_count);
    a.out.seg_rows = v.seg_rows;
    a.out.segments = S;
    for (unsigned c = 0; c < S; ++c) a.out.seg_routes[c] = v.routes[c];
    const auto grids = class_grids(T, kGwClasses, S);
    for (unsigned c = 0; c < 16; ++c) a.out.grid[c] = grids[c];
    mark("start", stream);
    ok(hipMemsetAsync(a.out.seg_count, 0, std::size_t{kPrefillMoeSegments} * kExperts * sizeof(unsigned), stream), "group counts");
    hipLaunchKernelGGL(seg_count_kernel, dim3((T + 255U) / 256U), dim3(256), 0, stream, a);
    launched("group counts");
    ok(dp::enqueue(a, stream), "device plan");
    Tables t{};
    t.groups = a.out.groups;
    t.pair_row = a.out.pair_row;
    t.pair_group = a.out.pair_group;
    t.tok_pair = a.out.tok_pair;
    t.tok_w = a.out.tok_w;
    t.xt = a.out.xt;
    t.at = a.out.at;
    t.part = a.out.part;
    t.pairs = 5U * T;
    t.rows = T;
    t.shared_group = GwPolicy::kExperts;
    t.tpairs = l.tpairs;
    t.seg_rows = v.seg_rows;
    for (unsigned c = 0; c < S; ++c) t.seg_y[c] = v.y[c], t.seg_out[c] = v.out[c];
    launch_device_planned(t, nullptr, nullptr, a.out.items, a.out.meta, grids, GwPolicy::kGuSlot, GwPolicy::kDnSlot,
                          prefill_gather_tok(), stream);
}

}  // namespace

void set_prefill_table_ring(int on) {
    if (on < -1 || on > 1) throw std::invalid_argument("set_prefill_table_ring: value must be -1, 0 or 1");
    (void)table_ring_default();  // capture once, including at Engine startup
    g_table_ring_override = on;
}
void cleanup_prefill_table_ring() noexcept {
    auto checked = [](hipError_t error) {
        if (error != hipSuccess) {
            std::fprintf(stderr, "minimax prefill moe table cleanup: %s\n", hipGetErrorString(error));
            std::abort();  // never free an in-flight source after a failed wait
        }
    };
    for (TableSlot& slot : g_table_slots) {
        if (slot.pending) checked(hipEventSynchronize(slot.copied));
        if (slot.host != nullptr) checked(hipHostFree(slot.host));
        if (slot.copied != nullptr) checked(hipEventDestroy(slot.copied));
        slot = {};
    }
    g_table_next = 0;
}

void set_prefill_mtfit(int on) {
    if (on < -1 || on > 1) throw std::invalid_argument("set_prefill_mtfit: value must be -1, 0 or 1");
    (void)prefill_mtfit_default();  // capture the environment once, including at Engine startup
    g_mtfit_override = on;
}
int prefill_pipe_override() { return g_pipe_override; }
void set_prefill_pipe_pair(int value) {
    if (value < -1 || value > 2) throw std::invalid_argument("set_prefill_pipe_pair: value must be -1, 0, 1 or 2 (auto)");
    (void)prefill_pipe_pair_default();  // capture once, including at Engine startup
    g_pipe_pair_override = value;
}
void set_prefill_dense_band(int value) {
    if (value != -1 && value != 0 && value != 4 && value != 8)
        throw std::invalid_argument("set_prefill_dense_band: value must be -1, 0, 4 or 8");
    (void)prefill_dense_band_default();  // capture once, including at Engine startup
    g_dense_band_override = value;
}
void set_prefill_pipe(int value) {
    if (value < -1 || value > 2) throw std::invalid_argument("set_prefill_pipe: value must be -1, 0, 1 or 2");
    (void)prefill_pipe_default();
    g_pipe_override = value;
}
void set_prefill_tail_stream(int on) {
    if (on < -1 || on > 1) throw std::invalid_argument("set_prefill_tail_stream: value must be -1, 0 or 1");
    (void)prefill_tail_stream_default();  // capture once, including at Engine startup
    g_tail_stream_override = on;
}
void cleanup_prefill_tail_stream() noexcept {
    if (g_tail_side != nullptr) (void)hipStreamSynchronize(g_tail_side);
    for (hipEvent_t* ev : {&g_tail_forked, &g_tail_joined}) {
        if (*ev != nullptr) (void)hipEventDestroy(*ev);
        *ev = nullptr;
    }
    if (g_tail_side != nullptr) (void)hipStreamDestroy(g_tail_side);
    g_tail_side = nullptr;
}
void set_prefill_core8(int value) {
    if (value < -1 || value > 2) throw std::invalid_argument("set_prefill_core8: value must be -1, 0, 1 or 2");
    (void)prefill_core8_default();  // capture once, including at Engine startup
    g_core8_override = value;
}
void set_prefill_gufuse(int on) {
    if (on < -1 || on > 1) throw std::invalid_argument("set_prefill_gufuse: value must be -1, 0 or 1");
    (void)prefill_gufuse_default();
    g_gufuse_override = on;
}
void set_prefill_gather_tok(int on) {
    if (on < -1 || on > 1) throw std::invalid_argument("set_prefill_gather_tok: value must be -1, 0 or 1");
    (void)prefill_gather_tok_default();
    g_gather_tok_override = on;
}
void set_prefill_gufuse_debug(bool on, float* intermediates, bool producer_only) {
    g_gufuse_debug = on || producer_only;
    g_gufuse_intermediates = intermediates;
    g_gufuse_producer_only = producer_only;
}
void set_prefill_marker(PrefillMarker marker, void* user) {
    g_marker = marker;
    g_marker_user = user;
}

PrefillDebugRegions prefill_moe_debug_regions(const PrefillMoeView& v) {
    const Plan p = make_plan(v);
    PrefillDebugRegions r;
    r.xt_offset = p.off_xt;
    r.xt_bytes = 2 * p.pair_row.size() * kHidden * 2;
    r.pairs = p.pair_row.size();
    r.tpairs = p.tpairs;
    for (const PGroup& g : p.groups) r.groups.push_back({g.pair_begin, g.tbegin, g.n});
    r.at_offset = p.off_at;
    r.at_bytes = p.pair_row.size() * kInter * 2;
    r.part_offset = p.off_part;
    r.part_bytes = p.part_floats * 4;
    return r;
}

std::size_t prefill_moe_workspace_bytes(const PrefillMoeView& v) { return make_plan(v).bytes; }

std::size_t prefill_moe_workspace_bound(std::uint32_t rows) {
    // Pairs 5T; per group splits x rows <= max(8 x 16, rows) for gate/up and <= max(2 x 16, rows) for down (the policy
    // stops doubling once ceil(rows/16) x O/64 x splits >= 160), so partial floats <= 129 x 128 x (2 x 1536 + 6144)
    // + 5T x (2 x 1536 + 6144), plus tables.
    const std::size_t P = 5ULL * rows, groups = kExperts + 1;
    std::size_t b = 0;
    b += table_bound(rows);
    const std::size_t PX = P + 15ULL * groups;  // v4 pads every group to 16 rows
    b += align256(2 * PX * kHidden * 2) + align256(PX * kInter * 2);
    b += align256((groups * 128 + P) * (2ULL * kInter + kHidden) * 4);
    return std::max(b, dev_layout(rows).bytes);  // the device plan's fixed layout (prefill_moe_device_plan)
}

std::size_t prefill_i4_workspace_bound(std::uint32_t rows) { return dev_layout(rows,true).bytes; }

bool prefill_i4_chain(std::size_t) {
#if STRIX_MMX_IU4_LIB
    return strix::core::moe_i4::enabled(true);  // FAST_I4 scope all; same bits as unfused
#else
    return false;
#endif
}

bool prefill_moe_device_plan(std::uint32_t rows) {
    static const bool device = [] {
        const char* e = std::getenv("STRIX_MMX_PF_PLAN");
        if (e == nullptr || std::string(e) == "device") return true;
        if (std::string(e) == "host") return false;
        throw std::invalid_argument("STRIX_MMX_PF_PLAN must be device or host");
    }();
    return device && rows > 0 && prefill_mtfit() && !(rows > 8 && prefill_gufuse());
}

void enqueue_prefill_moe_device(const PrefillMoeDeviceView& v, void* workspace, std::size_t workspace_bytes,
                                hipStream_t stream) {
    if (!prefill_moe_device_plan(v.rows)) throw std::invalid_argument("minimax prefill moe: device plan not selected for this call");
    enqueue_device_moe(v, workspace, workspace_bytes, stream);
    strix::route::launched(strix::route::Product::moe, v.i4 != nullptr ? strix::route::Family::iu4 : strix::route::Family::exact, v.rows);
}

unsigned prefill_moe_group_min_rows() { return 97; }  // every chunk's shared expert (and class 0) at splits 1
std::size_t prefill_moe_group_workspace_bound(std::uint32_t rows) { return dev_layout(rows, false, kGwClasses).bytes; }
void enqueue_prefill_moe_group(const PrefillMoeGroupView& v, void* workspace, std::size_t workspace_bytes, hipStream_t stream) {
    if (!prefill_moe_device_plan(v.rows)) throw std::invalid_argument("minimax prefill moe: device plan not selected for this call");
    enqueue_group_moe(v, workspace, workspace_bytes, stream);
    strix::route::launched(strix::route::Product::moe, strix::route::Family::exact, v.rows);  // group-wide calls are EXL3 only
}

void prefill_moe_device_check() {
    if (g_dev_flag == nullptr) return;
    unsigned flag = 0;
    ok(hipMemcpy(&flag, g_dev_flag, sizeof(flag), hipMemcpyDeviceToHost), "device plan flag read");
    if (flag != 0) {
        ok(hipMemset(g_dev_flag, 0, sizeof(unsigned)), "device plan flag reset");
        throw std::runtime_error(flag & 2U ? "minimax prefill moe: device plan class above its grid bound (bug)"
                                            : "minimax prefill moe: device plan saw invalid routes (expert id >= 128 or repeated in a row)");
    }
}

void cleanup_prefill_device_plan() noexcept {
    if (g_dev_flag != nullptr) (void)hipFree(g_dev_flag);
    g_dev_flag = nullptr;
}

#include "minimax/moe/mmx_prefill_iu4.inc"

void enqueue_prefill_moe(const PrefillMoeView& v, void* workspace, std::size_t workspace_bytes, hipStream_t stream) {
    Plan p = make_plan(v);
    if (workspace == nullptr || workspace_bytes < p.bytes) throw std::invalid_argument("minimax prefill moe: workspace too small");
    auto* ws = static_cast<unsigned char*>(workspace);
    const bool i4 = v.i4 != nullptr && v.i4->on();
    if (i4 && !iu4_built()) throw std::invalid_argument("minimax prefill moe: fast_i4 layer but no IU4 core in this build");
    const bool pair = prefill_pipe_pair(v.rows);
    const bool gufuse = v.rows > 8 && prefill_gufuse();
    V2Items v2i;
    std::vector<WItem> items;
    std::vector<pfk::Item> items2;  // gate/up classes, then down classes (same table region)
    std::vector<std::array<std::uint32_t, 3>> i4_groups;  // FAST_I4: strix::core::IU4Group [kI4Experts] in the items region
    if (i4) {
        i4_groups.resize(kI4Experts);
        for (unsigned e = 0; e < kI4Experts; ++e) i4_groups[e] = {e, 0U, 0U};
        for (const PGroup& g : p.groups) i4_groups[g.expert] = {g.expert, g.pair_begin, g.n};
        if (!v.i4_all) {
            fill_items(p, ws);
            items = p.dn_items16;
            items.insert(items.end(), p.dn_items64.begin(), p.dn_items64.end());
        }
    } else {
        v2i = fill_items_v2(p, ws, prefill_mtfit(), gufuse);
        for (unsigned c = 0; c < 8; ++c) items2.insert(items2.end(), v2i.gu[c].begin(), v2i.gu[c].end());
        for (unsigned c = 0; c < 8; ++c) items2.insert(items2.end(), v2i.fused[c].begin(), v2i.fused[c].end());
        for (unsigned c = 0; c < 8; ++c) items2.insert(items2.end(), v2i.dn[c].begin(), v2i.dn[c].end());
    }
    TableSlot* slot = table_ring() ? &next_table_slot(v.rows, i4 && !v.i4_all ? kI4GroupTableBytes : 0U) : nullptr;
    if (slot != nullptr && p.off_xt > slot->capacity) throw std::runtime_error("minimax prefill moe: table bound exceeded");
    auto copy = [&](std::size_t offset, const void* source, std::size_t bytes, const char* label) {
        if (slot != nullptr) {
            if (offset > slot->capacity || bytes > slot->capacity - offset)
                throw std::runtime_error("minimax prefill moe: table slot overflow");
            std::memcpy(slot->host + offset, source, bytes);
            source = slot->host + offset;
        }
        ok(hipMemcpyAsync(ws + offset, source, bytes, hipMemcpyHostToDevice, stream), label);
    };
    try {
        copy(p.off_groups, p.groups.data(), p.groups.size() * sizeof(PGroup), "groups");
        copy(p.off_pair_row, p.pair_row.data(), p.pair_row.size() * 4, "pairs");
        copy(p.off_pair_group, p.pair_group.data(), p.pair_group.size() * 4, "pairs");
        copy(p.off_tok_pair, p.tok_pair.data(), p.tok_pair.size() * 4, "tokens");
        copy(p.off_tok_w, p.tok_w.data(), p.tok_w.size() * 4, "tokens");
        if (i4) {
            copy(p.off_i4_groups, i4_groups.data(), i4_groups.size() * sizeof(i4_groups[0]), "iu4 groups");
            if (!v.i4_all) copy(p.off_items, items.data(), items.size() * sizeof(WItem), "iu4 EXL3 down items");
        }
        else copy(p.off_items, items2.data(), items2.size() * sizeof(pfk::Item), "items");
        if (slot != nullptr) {
            ok(hipEventRecord(slot->copied, stream), "table copies event");
            slot->pending = true;
        } else {
            ok(hipStreamSynchronize(stream), "tables");  // the host vectors die with this call
        }
    } catch (...) {
        // A partial upload or failed record has no completion event; retain sources until drained.
        if (hipStreamSynchronize(stream) != hipSuccess) std::abort();
        throw;
    }
    Tables t{};
    t.groups = reinterpret_cast<const PGroup*>(ws + p.off_groups);
    t.pair_row = reinterpret_cast<const std::uint32_t*>(ws + p.off_pair_row);
    t.pair_group = reinterpret_cast<const std::uint32_t*>(ws + p.off_pair_group);
    t.tok_pair = reinterpret_cast<const std::uint32_t*>(ws + p.off_tok_pair);
    t.tok_w = reinterpret_cast<const float*>(ws + p.off_tok_w);
    t.xt = reinterpret_cast<_Float16*>(ws + p.off_xt);
    t.at = reinterpret_cast<_Float16*>(ws + p.off_at);
    t.part = reinterpret_cast<float*>(ws + p.off_part);
    t.pairs = static_cast<unsigned>(p.pair_row.size());
    t.rows = v.rows;
    t.shared_group = static_cast<unsigned>(p.groups.size() - 1);
    t.tpairs = p.tpairs;
    t.fakeq = !i4 && (v.fakeq_a4 == PrefillFakeq::on ||
                     (v.fakeq_a4 == PrefillFakeq::environment && prefill_fakeq_a4())) ? 1U : 0U;
#if STRIX_MMX_IU4_LIB
    if (i4) {
        mark("start", stream);
        launch_iu4(p, t, v, ws, workspace_bytes, stream);
        // Shared route receipt: scope gate_up keeps the EXL3 down projection.
        if (v.i4_all) strix::route::launched(strix::route::Product::moe, strix::route::Family::iu4, v.rows);
        else {
            strix::route::launched(strix::route::Product::moe_gate_up, strix::route::Family::iu4, v.rows);
            strix::route::launched(strix::route::Product::moe_down, strix::route::Family::exact, v.rows);
        }
        return;
    }
#endif
    strix::route::launched(strix::route::Product::moe, strix::route::Family::exact, v.rows);  // host-planned EXL3 call
    mark("start", stream);
    // The token gather implements A4 fake quant; force it for the shadow call
    // even when the ordinary path selected the tiled gather.
    if (v.fakeq_a4 == PrefillFakeq::on || prefill_gather_tok()) hipLaunchKernelGGL(gather_transform_token_kernel, dim3((v.rows * kHB + 7) / 8), dim3(256), 0, stream, v.y, t);
    else hipLaunchKernelGGL(gather_transform_tiled_kernel, dim3((t.pairs * 2 * kHB + 7) / 8), dim3(256), 0, stream, v.y, t);
    launched("gather transform");
    mark("gather", stream);
    {
        const auto* it_d = reinterpret_cast<const pfk::Item*>(ws + p.off_items);
        std::size_t at = 0;
        // One stage's class launches; with the tail stream, classes 0..6 go to the side stream beside class 7.
        const bool tail_stream = prefill_mtfit() && prefill_tail_stream();
        auto stage = [&](const std::vector<pfk::Item>* lists, const unsigned* ctas, const char* const* labels) {
            bool tails = false;
            for (unsigned c = 0; c < 7; ++c) tails = tails || !lists[c].empty();
            const bool side = tail_stream && tails && !lists[7].empty();
            if (side) tail_fork(stream);
            std::size_t c_at = at;
            try {
                for (unsigned c = 0; c < 7 && side; ++c) {
                    launch_v2(it_d + c_at, lists[c], ctas[c], c, g_tail_side, pair);
                    c_at += lists[c].size();
                }
                for (unsigned c = side ? 7U : 0U; c < 8; ++c) {
                    launch_v2(it_d + c_at, lists[c], ctas[c], c, stream, pair);
                    c_at += lists[c].size();
                    if (!lists[c].empty()) mark(labels[c], stream);
                }
            } catch (...) {
                if (side) tail_join(stream);  // side work stays inside the caller's completion chain
                throw;
            }
            if (side) {
                tail_join(stream);
                mark(labels[6], stream);  // the joined tail classes (v4 only)
            }
            at = c_at;
        };
        stage(v2i.gu, v2i.gu_ctas, kGuLabel);
        for (unsigned c = 0; c < 8; ++c) {
            if (!v2i.fused[c].empty()) {
                launch_gufuse_class(it_d + at, static_cast<unsigned>(v2i.fused[c].size() / 2), v2i.fused_ctas[c], c, t, stream);
                mark(kGuLabel[c], stream);
            }
            at += v2i.fused[c].size();
        }
        launched("gate/up v2");
        hipLaunchKernelGGL(gate_up_finish_tiled_kernel, dim3((t.pairs * kIB + 7) / 8), dim3(256), 0, stream, t, gufuse && !g_gufuse_producer_only, g_gufuse_intermediates);
        launched("gate/up finish");
        mark("finish", stream);
        stage(v2i.dn, v2i.dn_ctas, kDnLabel);
        launched("down v2");
        hipLaunchKernelGGL(down_combine_kernel, dim3((v.rows * kHB + 7) / 8), dim3(256), 0, stream, t, v.out);
        launched("down combine");
        mark("combine", stream);
    }
}

namespace {
// Item list of one dense call: at most kMaxDenseJobs jobs x (bands + main + tail) items.
constexpr std::size_t kDenseItems = 3 * kMaxDenseJobs;
struct DensePlan {
    unsigned k = 0, tpad = 0;
    unsigned splits[kMaxDenseJobs] = {};
    std::size_t off_items = 0, off_xt[kMaxDenseJobs] = {}, off_part[kMaxDenseJobs] = {}, bytes = 0;
};
bool dense_bits_supported(unsigned bits) { return bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8; }
DensePlan make_dense_plan(const PrefillDenseJob* jobs, unsigned count, unsigned rows) {
    if (jobs == nullptr || count == 0 || count > kMaxDenseJobs || rows == 0)
        throw std::invalid_argument("minimax prefill dense: bad job count or rows");
    DensePlan p;
    p.k = jobs[0].w.input;
    p.tpad = (rows + 15U) / 16U * 16U;
    if (p.k == 0 || p.k % 128 != 0) throw std::invalid_argument("minimax prefill dense: input must be a multiple of 128");
    std::size_t off = 0;
    p.off_items = off;
    off += align256(kDenseItems * sizeof(pfk::Item));
    for (unsigned m = 0; m < count; ++m) {
        const Exl3& w = jobs[m].w;
        if (w.input != p.k || w.output == 0 || w.output % 128 != 0 || w.trellis == nullptr || w.suh == nullptr ||
            w.svh == nullptr || jobs[m].out == nullptr)
            throw std::invalid_argument("minimax prefill dense: jobs need one input width and outputs % 128 == 0");
        if (!dense_bits_supported(w.bits)) throw std::invalid_argument("minimax prefill dense: unsupported bit width");
        p.splits[m] = gemm_splits(w.input, w.output, w.bits, rows);
        p.off_xt[m] = off;
        off += align256(std::size_t{p.tpad} * p.k * 2);
        p.off_part[m] = off;
        off += align256(std::size_t{p.splits[m]} * rows * w.output * 4);
    }
    p.bytes = off;
    return p;
}
// Pinned host staging for the item lists: the H2D copy is stream-ordered into the call's workspace (consumed by the
// call's own launches before any later call on that stream overwrites it); a slot is reused only after its copy event.
struct DenseSlot {
    pfk::Item* host = nullptr;
    hipEvent_t copied = nullptr;
    bool pending = false;
};
constexpr unsigned kDenseSlots = 16;
DenseSlot g_dense_slots[kDenseSlots];
unsigned g_dense_next = 0;
std::mutex g_dense_mutex;  // reserve -> fill -> copy -> record, and cleanup: one transaction per thread
DenseSlot& next_dense_slot() {
    DenseSlot& slot = g_dense_slots[g_dense_next];
    g_dense_next = (g_dense_next + 1) % kDenseSlots;
    if (slot.host == nullptr) {
        ok(hipHostMalloc(reinterpret_cast<void**>(&slot.host), kDenseItems * sizeof(pfk::Item), hipHostMallocDefault), "dense items host");
        ok(hipEventCreateWithFlags(&slot.copied, hipEventDisableTiming), "dense items event");
    }
    if (slot.pending) {
        ok(hipEventSynchronize(slot.copied), "dense items slot reuse");
        slot.pending = false;
    }
    return slot;
}
}  // namespace

bool prefill_dense_supported(const Exl3& w) { return dense_bits_supported(w.bits) && w.input % 128 == 0 && w.output % 128 == 0; }

std::size_t prefill_dense_workspace_bytes(const PrefillDenseJob* jobs, unsigned count, unsigned rows) {
    return make_dense_plan(jobs, count, rows).bytes;
}

void enqueue_prefill_dense(const PrefillDenseJob* jobs, unsigned count, const float* x, unsigned rows, void* workspace,
                           std::size_t workspace_bytes, hipStream_t stream) {
    const DensePlan p = make_dense_plan(jobs, count, rows);
    if (x == nullptr || workspace == nullptr || workspace_bytes < p.bytes)
        throw std::invalid_argument("minimax prefill dense: null input or workspace too small");
    auto* ws = static_cast<unsigned char*>(workspace);
    constexpr unsigned kBandClass = 8;  // tall bands (STRIX_MMX_PREFILL_DENSE_BAND), beside the eight row-tile classes
    const unsigned band = prefill_dense_band();
    std::vector<pfk::Item> lists[kBandClass + 1];
    unsigned ctas[kBandClass + 1] = {};
    for (unsigned m = 0; m < count; ++m) {
        const Exl3& w = jobs[m].w;
        pfk::Item it{};
        it.trellis = w.trellis;
        it.x = reinterpret_cast<_Float16*>(ws + p.off_xt[m]);
        it.part = reinterpret_cast<float*>(ws + p.off_part[m]);
        it.input = w.input;
        it.output = w.output;
        it.bits = w.bits;
        it.rows = rows;
        it.splits = p.splits[m];
        it.gx = w.output / 128;
        auto emit = [&](unsigned c, unsigned row0, unsigned gy) {
            it.row0 = row0;
            it.gy = gy;
            it.cta_begin = ctas[c];
            ctas[c] += it.gx * gy * it.splits;
            lists[c].push_back(it);
        };
        const auto fit = route::mtfit(rows);  // the production v4 + mtfit item shapes
        const unsigned banded = band != 0 ? fit.main / band * band : 0U;  // 128-row tiles in full bands
        if (banded != 0) {
            it.gx = w.output / 32;
            emit(kBandClass, 0, banded / band);
            it.gx = w.output / 128;
        }
        if (fit.main != banded) emit(7, banded * 128, fit.main - banded);
        if (fit.remainder != 0) emit(fit.tail_class, fit.main * 128, 1);
    }
    std::lock_guard<std::mutex> staging(g_dense_mutex);
    DenseSlot& slot = next_dense_slot();
    unsigned at[kBandClass + 1] = {}, used = 0;
    for (unsigned c = 0; c <= kBandClass; ++c) {
        at[c] = used;
        for (const pfk::Item& it : lists[c]) slot.host[used++] = it;
    }
    auto* items_d = reinterpret_cast<pfk::Item*>(ws + p.off_items);
    ok(hipMemcpyAsync(items_d, slot.host, used * sizeof(pfk::Item), hipMemcpyHostToDevice, stream), "dense items");
    ok(hipEventRecord(slot.copied, stream), "dense items event");
    slot.pending = true;
    DenseArgs a{};
    a.count = count;
    a.rows = rows;
    a.k = p.k;
    unsigned waves = 0;
    for (unsigned m = 0; m < count; ++m) {
        a.suh[m] = jobs[m].w.suh;
        a.svh[m] = jobs[m].w.svh;
        a.xt[m] = reinterpret_cast<_Float16*>(ws + p.off_xt[m]);
        a.part[m] = reinterpret_cast<const float*>(ws + p.off_part[m]);
        a.out[m] = jobs[m].out;
        a.n[m] = jobs[m].w.output;
        a.splits[m] = p.splits[m];
        waves += rows * (jobs[m].w.output / 128);
    }
    hipLaunchKernelGGL(dense_transform_kernel, dim3((rows * (p.k / 128) + 7) / 8), dim3(256), 0, stream, x, a);
    launched("dense transform");
    if (!lists[kBandClass].empty()) {
        const auto n = static_cast<unsigned>(lists[kBandClass].size());
        if (band == 8) launch_band_class<8>(items_d + at[kBandClass], n, ctas[kBandClass], stream);
        else launch_band_class<4>(items_d + at[kBandClass], n, ctas[kBandClass], stream);
    }
    for (unsigned c = 0; c < 8; ++c)
        if (!lists[c].empty()) launch_class(items_d + at[c], static_cast<unsigned>(lists[c].size()), ctas[c], c, stream);
    launched("dense v4");
    hipLaunchKernelGGL(dense_finish_kernel, dim3((waves + 7) / 8), dim3(256), 0, stream, a);
    launched("dense finish");
}

void cleanup_prefill_dense() noexcept {
    std::lock_guard<std::mutex> staging(g_dense_mutex);
    for (DenseSlot& slot : g_dense_slots) {
        if (slot.pending) (void)hipEventSynchronize(slot.copied);
        if (slot.copied != nullptr) (void)hipEventDestroy(slot.copied);
        if (slot.host != nullptr) (void)hipHostFree(slot.host);
        slot = DenseSlot{};
    }
    g_dense_next = 0;
}

}  // namespace strix::minimax::moe::opt
