#include "strix/hip_check.hpp"
#include "minimax/engine/route_identity.hpp"
// minimax-m3 index projections in one launch, exact dense_t order (src/minimax/attn/index_proj.hpp).
#include "index_proj.hpp"
#include "core/f16_dense_tall.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::minimax::attn {
namespace {

constexpr unsigned kChunks = kProjIn / 8;  // 768 chunks of 8 inputs

__device__ __forceinline__ std::uint32_t f2u(float f) { return __builtin_bit_cast(std::uint32_t, f); }
__device__ __forceinline__ float u2f(std::uint32_t u) { return __builtin_bit_cast(float, u); }
__device__ __forceinline__ float h2f(std::uint32_t bits16) {
    return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(bits16)));
}
// acc[t] = fmaf(x[t][i], w_i, acc[t]) for the 8 inputs of one chunk, i ascending (dense_t's chain). x rows are at
// constant offsets t * kProjIn from one wave-uniform pointer (scalar loads, no per-row pointers).
template <unsigned R>
__device__ __forceinline__ void chunk_fma(const uint4& u, const float* __restrict__ x, unsigned i0, float (&acc)[R]) {
    const std::uint32_t wd[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
        const float wf = h2f((j & 1U) != 0U ? wd[j / 2] >> 16U : wd[j / 2] & 0xFFFFU);
#pragma unroll
        for (unsigned t = 0; t < R; ++t) acc[t] = __builtin_fmaf(x[t * kProjIn + i0 + j], wf, acc[t]);
    }
}
__device__ __forceinline__ void store_out(const IndexProjView& v, unsigned t, unsigned o, float a) {
    if (o < kProjQ) v.iq[static_cast<std::size_t>(t) * kProjQ + o] = a;
    else v.ik[static_cast<std::size_t>(t) * kProjK + (o - kProjQ)] = a;
}
// Chunk c of this lane's output through a raw buffer load: the descriptor is in SGPRs, the lane's 16-byte offset is
// one VGPR for every chunk and the chunk offset an SGPR, so the in-flight loads hold no address registers.
using Rsrc = __amdgpu_buffer_rsrc_t;
__device__ __forceinline__ Rsrc make_rsrc(const _Float16* w) {
    // gfx11 raw buffer word 3 (32-bit format, as composable_kernel's CK_BUFFER_RESOURCE_3RD_DWORD for gfx11)
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<_Float16*>(w), 0, static_cast<int>(kProjOut * kProjIn * 2), 0x31004000);
}
__device__ __forceinline__ uint4 load_chunk(Rsrc rs, unsigned c, unsigned lane_off) {
    using V4 = unsigned __attribute__((ext_vector_type(4)));
    const V4 r = __builtin_amdgcn_raw_buffer_load_b128(rs, static_cast<int>(lane_off), static_cast<int>(c * kProjOut * 16U), 0);
    return make_uint4(r.x, r.y, r.z, r.w);
}

// The same chunk with the activations already in registers (wave-uniform, so SGPRs).
template <unsigned R>
__device__ __forceinline__ void chunk_fma_x(const uint4& u, const float (&xv)[R][8], float (&acc)[R]) {
    const std::uint32_t wd[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (unsigned j = 0; j < 8; ++j) {
        const float wf = h2f((j & 1U) != 0U ? wd[j / 2] >> 16U : wd[j / 2] & 0xFFFFU);
#pragma unroll
        for (unsigned t = 0; t < R; ++t) acc[t] = __builtin_fmaf(xv[t][j], wf, acc[t]);
    }
}
template <unsigned R>
__device__ __forceinline__ void load_x(const float* __restrict__ x, unsigned rs, unsigned c, float (&xv)[R][8]) {
#pragma unroll
    for (unsigned t = 0; t < R; ++t)
#pragma unroll
        for (unsigned j = 0; j < 8; ++j) xv[t][j] = x[t * rs + c * 8 + j];
}

// Decode: thread = output o, R = rows chains. The weight ring keeps D chunk loads in flight per thread; the
// activations of the next PX chunks (8R floats each, wave-uniform: scalar loads) are prefetched the same way, so a
// chunk never waits for its scalar loads. Slot indices are static (the D loop is unrolled, PX divides D).
// Row split (grid.y > 1): block y runs rows y*R .. y*R+R-1 (a short last group repeats its last row, not stored); the
// extra waves re-read the weights, mostly from L2.
// MODE (diagnostics only, production = 0): 1 = no weight loads (the first chunk's words reused: FMA chain + activation
// loads), 2 = no activation loads either (pure FMA chain), 3 = no FMAs (weight + activation streams, words folded with
// XOR into the outputs).
template <unsigned R, unsigned D, int MODE = 0> __global__ __launch_bounds__(32) void proj_decode_kernel(IndexProjView v) {
    constexpr unsigned PX = R == 1 ? 8U : R == 2 ? 4U : R <= 4 ? 2U : 1U;
    const unsigned o = blockIdx.x * 32 + threadIdx.x;
    const unsigned row0 = blockIdx.y * R, here = v.rows - row0 < R ? v.rows - row0 : R;
    const float* __restrict__ x = v.x + static_cast<std::size_t>(row0) * kProjIn;
    const unsigned xs = here == R ? kProjIn : 0U;  // row stride; a short group (R = 2, one row) reads its row twice
    const Rsrc rs = make_rsrc(v.w);
    const unsigned off = o * 16U;
    uint4 ring[D];
#pragma unroll
    for (unsigned d = 0; d < D; ++d) ring[d] = load_chunk(rs, d, off);
    float xb[PX][R][8];
#pragma unroll
    for (unsigned p = 0; p < PX; ++p) load_x<R>(x, xs, p, xb[p]);
    float acc[R];
#pragma unroll
    for (unsigned t = 0; t < R; ++t) acc[t] = 0.0F;
    for (unsigned c0 = 0; c0 < kChunks; c0 += D) {
#pragma unroll
        for (unsigned d = 0; d < D; ++d) {
            const unsigned c = c0 + d;
            if constexpr (MODE == 3) {
#pragma unroll
                for (unsigned t = 0; t < R; ++t)
                    acc[t] = u2f(f2u(acc[t]) ^ ring[d].x ^ ring[d].y ^ ring[d].z ^ ring[d].w ^ f2u(xb[d % PX][t][0]));
            } else {
                chunk_fma_x<R>(ring[d], xb[d % PX], acc);
            }
            // Refill both windows only after the chunk is consumed (issuing first would keep two copies of every
            // slot; unconditional, so no merge copies); the barriers keep that order and stop the scheduler from
            // hoisting every later chunk's loads.
            __builtin_amdgcn_sched_barrier(0);
            if constexpr (MODE != 1 && MODE != 2) ring[d] = load_chunk(rs, c + D < kChunks ? c + D : kChunks - 1, off);
            if constexpr (MODE != 2) load_x<R>(x, xs, c + PX < kChunks ? c + PX : kChunks - 1, xb[d % PX]);
            __builtin_amdgcn_sched_barrier(0);
        }
    }
#pragma unroll
    for (unsigned t = 0; t < R; ++t)
        if (t < here) store_out(v, row0 + t, o, acc[t]);
}

// Decode, 2..8 rows (measurement: the row-pair waves of proj_decode_kernel each re-stream the whole matrix with a 16-deep ring
// and reach only ~125 GB/s): one block per 32 outputs, 4 waves. All 128 threads load each 16-chunk stage once (8 KB,
// coalesced) into an LDS ring of 4 stages, two stages in flight in registers; wave w runs rows 2w, 2w + 1 of every
// chunk from LDS (lane = output: one 16-byte read, conflict-free). Each output's chain is proj_decode_kernel's.
constexpr unsigned kLdsWaves = 4, kStageChunks = 16, kLdsStages = 4, kStages = kChunks / kStageChunks;  // 48 stages
constexpr unsigned kPieces = kStageChunks * 32 / (kLdsWaves * 32);                                      // 4 per thread
__global__ __launch_bounds__(kLdsWaves * 32) void proj_decode_lds_kernel(IndexProjView v) {
    __shared__ uint4 ring[kLdsStages][kStageChunks][32];  // 32 KB
    constexpr unsigned PX = 4;                            // activation chunks prefetched (2 rows: 64 SGPRs)
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U;
    const unsigned og = blockIdx.x, o = og * 32 + lane;
    const unsigned row0 = wave * 2, here = row0 < v.rows ? (v.rows - row0 < 2 ? v.rows - row0 : 2U) : 0U;
    const float* __restrict__ x = v.x + static_cast<std::size_t>(here != 0 ? row0 : 0U) * kProjIn;
    const unsigned xs = here == 2 ? kProjIn : 0U;  // a short pair repeats its row (not stored)
    const Rsrc rs = make_rsrc(v.w);
    auto load_stage = [&](unsigned s, uint4 (&r)[kPieces]) {
        const unsigned st = s < kStages ? s : kStages - 1;  // unconditional (no merge copies); the tail re-reads
#pragma unroll
        for (unsigned k = 0; k < kPieces; ++k) {
            const unsigned p = k * (kLdsWaves * 32) + tid;
            r[k] = load_chunk(rs, st * kStageChunks + p / 32, (og * 32 + p % 32) * 16U);
        }
    };
    auto store_stage = [&](unsigned s, const uint4 (&r)[kPieces]) {
#pragma unroll
        for (unsigned k = 0; k < kPieces; ++k) {
            const unsigned p = k * (kLdsWaves * 32) + tid;
            ring[s % kLdsStages][p / 32][p % 32] = r[k];
        }
    };
    float xb[PX][2][8];
#pragma unroll
    for (unsigned p = 0; p < PX; ++p) load_x<2>(x, xs, p, xb[p]);
    float acc[2] = {0.0F, 0.0F};
    auto compute_stage = [&](unsigned s) {
        if (here == 0) return;  // wave-uniform: this wave only loads
#pragma unroll
        for (unsigned k = 0; k < kStageChunks; ++k) {
            const unsigned c = s * kStageChunks + k;
            chunk_fma_x<2>(ring[s % kLdsStages][k][lane], xb[k % PX], acc);
            __builtin_amdgcn_sched_barrier(0);
            load_x<2>(x, xs, c + PX < kChunks ? c + PX : kChunks - 1, xb[k % PX]);
            __builtin_amdgcn_sched_barrier(0);
        }
    };
    uint4 ra[kPieces], rb[kPieces];
    load_stage(0, ra);
    load_stage(1, rb);
    store_stage(0, ra);
    store_stage(1, rb);
    load_stage(2, ra);
    load_stage(3, rb);
    for (unsigned s = 0; s < kStages; s += 2) {
        // Slot (s + 2) % 4 was last read for stage s - 2, finished by every wave before the previous barrier.
        store_stage(s + 2, ra);
        load_stage(s + 4, ra);
        __syncthreads();
        compute_stage(s);
        store_stage(s + 3, rb);
        load_stage(s + 5, rb);
        __syncthreads();
        compute_stage(s + 1);
    }
    for (unsigned t = 0; t < 2; ++t)
        if (t < here) store_out(v, row0 + t, o, acc[t]);
}

// (measurement: staging the activations in LDS was slower, 8K 8.1 -> 11.5 ms: 16 broadcast LDS reads per chunk and a
// 20 KB tile cost more than the scalar loads they replace; this is the baseline kernel.)
// Prefill: block = 32 outputs x 64 tokens (8 waves x 8 token chains per lane); stage = 8 chunks (64 inputs), the
// weight tile (4 KB) loaded once per block, the next stage's piece prefetched into a register.
constexpr unsigned kTokPerWave = 8, kWaves = 8, kStage = 8;
__global__ __launch_bounds__(kWaves * 32) void proj_tiled_kernel(IndexProjView v) {
    __shared__ uint4 tile[kStage][32];
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U;
    const unsigned og = blockIdx.x, o = og * 32 + lane;
    const unsigned t0 = blockIdx.y * (kWaves * kTokPerWave) + wave * kTokPerWave;
    // Rows past v.rows: the wave reads the last 8 valid rows instead (results not stored); needs rows >= 8.
    const unsigned tb = t0 + kTokPerWave <= v.rows ? t0 : v.rows - kTokPerWave;
    const float* __restrict__ xr = v.x + static_cast<std::size_t>(tb) * kProjIn;
    float acc[kTokPerWave];
#pragma unroll
    for (unsigned t = 0; t < kTokPerWave; ++t) acc[t] = 0.0F;
    const unsigned pk = tid >> 5U, po = tid & 31U;  // this thread's piece of a stage: chunk pk, output og*32+po
    const Rsrc rs = make_rsrc(v.w);
    const unsigned poff = (og * 32 + po) * 16U;
    uint4 next = load_chunk(rs, pk, poff);
    for (unsigned s = 0; s < kChunks / kStage; ++s) {
        __syncthreads();
        tile[pk][po] = next;
        __syncthreads();
        next = load_chunk(rs, (s + 1 < kChunks / kStage ? s + 1 : s) * kStage + pk, poff);
#pragma unroll
        for (unsigned k = 0; k < kStage; ++k) {
            chunk_fma<kTokPerWave>(tile[k][lane], xr, (s * kStage + k) * 8, acc);
            __builtin_amdgcn_sched_barrier(0);
        }
    }
#pragma unroll
    for (unsigned t = 0; t < kTokPerWave; ++t)
        if (tb == t0 || tb + t >= t0) store_out(v, tb + t, o, acc[t]);  // a shifted wave stores only rows >= t0
}

// Prefill, slab form (exact, same chains): the block of proj_tiled_kernel with R token chains per lane, and
//   - the wave index made wave-uniform (readfirstlane), so the wave's activations are scalar loads of XW inputs per
//     token (one load per XW MACs) and SGPR operands of the FMAs, instead of one per-lane load per MAC;
//   - the weight stage widened F16 -> F32 once by the thread that stages it (8 conversions per stage and thread, not
//     8 per chunk), double buffered in LDS (2 x 8 KB): one barrier per stage, the next stage is written while slow
//     waves still read the current one.
// R = 8, XW = 8: 64 tokens x 32 outputs per block, 512 FMAs of a stage in 344 instructions (dual-issue pairs), 65
// scalar loads and 16 LDS reads; 25 VGPRs (tiled: 107, 128 per-lane loads per stage). Measured (literal A B A x3, 60
// weight sets): 4096 rows 4092 -> 2744 us, 8192 rows 7587 -> 5085 us, 1024 rows 988 -> 691 us (0.67-0.70).
// (measurement: R = 16, XW = 4, 128 tokens per block, was 1.26-1.47x slower than the tiled kernel: 256 scalar
// loads per stage.)
template <unsigned R, unsigned XW> __global__ __launch_bounds__(kWaves * 32) void proj_slab_kernel(IndexProjView v) {
    using XV = float __attribute__((ext_vector_type(XW)));
    __shared__ float4 slab[2][kStage][2][32];
    const unsigned tid = threadIdx.x, lane = tid & 31U;
    const unsigned wave = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(tid >> 5U)));
    const unsigned og = blockIdx.x, o = og * 32 + lane;
    const unsigned t0 = blockIdx.y * (kWaves * R) + wave * R;
    // Rows past v.rows: the wave reads the last R valid rows instead (results not stored); needs rows >= R.
    const unsigned tb = t0 + R <= v.rows ? t0 : v.rows - R;
    const float* __restrict__ xr = v.x + static_cast<std::size_t>(tb) * kProjIn;
    float acc[R];
#pragma unroll
    for (unsigned t = 0; t < R; ++t) acc[t] = 0.0F;
    const unsigned pk = tid >> 5U;  // this thread's piece of a stage: chunk pk, output og*32+lane
    const Rsrc rs = make_rsrc(v.w);
    const unsigned poff = o * 16U;
    uint4 next = load_chunk(rs, pk, poff);
    for (unsigned s = 0; s < kChunks / kStage; ++s) {
        const unsigned b = s & 1U;
        // Buffer b was last read for stage s - 2, finished by every wave before the previous barrier.
        slab[b][pk][0][lane] = make_float4(h2f(next.x & 0xFFFFU), h2f(next.x >> 16U), h2f(next.y & 0xFFFFU), h2f(next.y >> 16U));
        slab[b][pk][1][lane] = make_float4(h2f(next.z & 0xFFFFU), h2f(next.z >> 16U), h2f(next.w & 0xFFFFU), h2f(next.w >> 16U));
        __syncthreads();
        next = load_chunk(rs, (s + 1 < kChunks / kStage ? s + 1 : s) * kStage + pk, poff);
#pragma unroll
        for (unsigned k = 0; k < kStage; ++k) {
            const float4 w0 = slab[b][k][0][lane], w1 = slab[b][k][1][lane];
            const float wf[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
            const float* __restrict__ xc = xr + (s * kStage + k) * 8;
#pragma unroll
            for (unsigned h = 0; h < 8; h += XW) {
                XV xv[R];
#pragma unroll
                for (unsigned t = 0; t < R; ++t) __builtin_memcpy(&xv[t], xc + t * kProjIn + h, sizeof(XV));
#pragma unroll
                for (unsigned j = 0; j < XW; ++j)
#pragma unroll
                    for (unsigned t = 0; t < R; ++t) acc[t] = __builtin_fmaf(xv[t][j], wf[h + j], acc[t]);
                __builtin_amdgcn_sched_barrier(0);
            }
        }
    }
#pragma unroll
    for (unsigned t = 0; t < R; ++t)
        if (tb == t0 || tb + t >= t0) store_out(v, tb + t, o, acc[t]);  // a shifted wave stores only rows >= t0
}

// FAST minimax-index-proj-wmma-v1 (index_proj.hpp): adapter of the shared dense FP16 WMMA core. Column tile i = 16
// consecutive outputs: tiles 0..15 are iq (row stride 256), 16..23 ik (row stride 128); 24 tiles, so tc = 4 / 2 / 1
// cover them with no dead tile. stage() copies the 64 FP16 weights of k units [unit, unit + 4) of one output from
// the packed layout: chunk c (inputs 8c..8c+7) of output o is 16 bytes at (c * kProjOut + o) * 8, and the 4 units are
// chunks 2 unit .. 2 unit + 7, so nb[i] = chunk 2 unit + i (16 stagers of a tile read 256 contiguous bytes per chunk).
namespace tall = strix::core::f16_dense_tall;
constexpr unsigned kWmmaTiles = kProjOut / 16U;
static_assert(kProjQ % 16U == 0 && kProjK % 16U == 0 && kProjIn % 128U == 0, "index_proj wmma: core tiling");
struct IndexCols {
    const _Float16* w;
    float* iq;
    float* ik;
    __device__ __forceinline__ tall::ColumnTile tile(unsigned i) const {
        return i < kProjQ / 16U ? tall::ColumnTile{nullptr, iq + 16U * i, kProjQ, true}
                                : tall::ColumnTile{nullptr, ik + 16U * (i - kProjQ / 16U), kProjK, true};
    }
    __device__ __forceinline__ void stage(unsigned tile, unsigned col, unsigned unit, tall::U4* nb) const {
        const auto* p = reinterpret_cast<const unsigned char*>(w) + (static_cast<std::size_t>(2U * unit) * kProjOut + tile * 16U + col) * 16U;
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) nb[i] = *(tall::detail::GlobalU4*)(p + static_cast<std::size_t>(i) * kProjOut * 16U);
    }
};
// CF: column blocks on grid.x (core kColumnFast): the 24 / TC CTAs of a row block share its x through L2. Rows on
// grid.x (CF = false, bench arm) re-stream x from memory once per column block: 4096 rows tc 4 4290 us vs slab 2740.
template <bool CF> struct IndexColsOrder : IndexCols {
    static constexpr bool kColumnFast = CF;
};
template <unsigned TC, bool CF> __global__ __launch_bounds__(256) void proj_wmma_kernel(IndexProjView v) {
    __shared__ tall::Storage<TC> lds;
    tall::core<TC>(v.x, kProjIn, v.rows, IndexColsOrder<CF>{{v.w, v.iq, v.ik}}, lds);
}
// x16 form: the shared core tile form (core/f16_dense_tall.hpp X16Cols / enqueue_x16_tiles: x rounded to FP16 once,
// k-tile-major, column-fast), same bits as the FP32 x path. Here: x read once by one column block alone, 4096 rows
// 1079 us; the tiles + WMMA 1229 us vs slab 2753 us.
using IndexColsX16 = tall::X16Cols<IndexCols>;
static_assert(kFastTilesRows == tall::kX16Rows, "index_proj: the FAST regime threshold is the shared x16 crossover");
__global__ __launch_bounds__(256) void proj_wmma_x16_kernel(IndexProjView v, const _Float16* xt) {
    __shared__ tall::Storage<4> lds;
    tall::core<4>(v.x, kProjIn, v.rows, IndexColsX16{{v.w, v.iq, v.ik}, xt, v.rows}, lds);
}
// Diagnostics (bench only): mode 1 = every row reads x row (row % 128): the x stream stays in cache (wrong
// outputs); mode 2 = the production kernel with at most 192 VGPRs (8 waves per SIMD: 2 CTAs per CU), same bits.
struct IndexColsXCached : IndexColsOrder<true> {
    const unsigned char* xb;
    __device__ __forceinline__ tall::Half16 fragment(unsigned row, unsigned unit) const {
        return tall::detail::row_fragment(tall::detail::uniform(xb), (row % 128U) * kProjIn * 4U + unit * 64U);
    }
};
__global__ __launch_bounds__(256) void proj_wmma_xcached_kernel(IndexProjView v) {
    __shared__ tall::Storage<4> lds;
    tall::core<4>(v.x, kProjIn, v.rows, IndexColsXCached{{{v.w, v.iq, v.ik}}, reinterpret_cast<const unsigned char*>(v.x)}, lds);
}
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8, 8))) void proj_wmma_cap_kernel(IndexProjView v) {
    __shared__ tall::Storage<4> lds;
    tall::core<4>(v.x, kProjIn, v.rows, IndexColsOrder<true>{{v.w, v.iq, v.ik}}, lds);
}

using strix::hip_check;

template <unsigned D> void launch_decode(const IndexProjView& v, unsigned rpw, hipStream_t stream) {
    if (rpw == 2 && v.rows > 2) {  // row split: pairs of rows per wave
        hipLaunchKernelGGL((proj_decode_kernel<2, D>), dim3(kProjOut / 32, (v.rows + 1) / 2), dim3(32), 0, stream, v);
        return;
    }
    const dim3 grid(kProjOut / 32);
    switch (v.rows) {
    case 1: hipLaunchKernelGGL((proj_decode_kernel<1, D>), grid, dim3(32), 0, stream, v); break;
    case 2: hipLaunchKernelGGL((proj_decode_kernel<2, D>), grid, dim3(32), 0, stream, v); break;
    case 3: hipLaunchKernelGGL((proj_decode_kernel<3, D>), grid, dim3(32), 0, stream, v); break;
    case 4: hipLaunchKernelGGL((proj_decode_kernel<4, D>), grid, dim3(32), 0, stream, v); break;
    case 5: hipLaunchKernelGGL((proj_decode_kernel<5, D>), grid, dim3(32), 0, stream, v); break;
    case 6: hipLaunchKernelGGL((proj_decode_kernel<6, D>), grid, dim3(32), 0, stream, v); break;
    case 7: hipLaunchKernelGGL((proj_decode_kernel<7, D>), grid, dim3(32), 0, stream, v); break;
    default: hipLaunchKernelGGL((proj_decode_kernel<8, D>), grid, dim3(32), 0, stream, v); break;
    }
}

unsigned split_default() {  // STRIX_MMATTN_PROJ_SPLIT=0 disables the row split (measurements determine the default)
    static const unsigned r = [] {
        const char* e = std::getenv("STRIX_MMATTN_PROJ_SPLIT");
        return e != nullptr && e[0] == '0' ? 0U : 2U;
    }();
    return r;
}

// OFF by default (measurement, ABA x3: exact but 1.8-2.2x slower at R2..R8, attn_proj seq +20-27 %). Root cause from the ISA:
// __syncthreads lowers to s_waitcnt vmcnt(0) + s_barrier, so every barrier waits for the stage just prefetched (no
// load overlap across barriers); the per-thread chunk index lands in the buffer load's SGPR soffset -> a readfirstlane
// waterfall loop per load; the wave-dependent activation row turns the scalar x loads into vector loads that share
// the in-order vmcnt with the weight stream. STRIX_MMATTN_PROJ_LDS=1 opts in (benches).
unsigned lds_default() {
    static const unsigned r = [] {
        const char* e = std::getenv("STRIX_MMATTN_PROJ_LDS");
        return e != nullptr && e[0] == '1' ? 1U : 0U;
    }();
    return r;
}

unsigned ring_override() {  // STRIX_MMATTN_PROJ_RING (8/16/32/48) overrides the measured choice; 0 = none
    static const unsigned r = [] {
        const char* e = std::getenv("STRIX_MMATTN_PROJ_RING");
        const long x = e != nullptr ? std::strtol(e, nullptr, 10) : 0;
        return x == 8 || x == 16 || x == 32 || x == 48 ? static_cast<unsigned>(x) : 0U;
    }();
    return r;
}

}  // namespace

std::vector<_Float16> pack_index_proj(const float* wq_rows, const float* wk_rows) {
    std::vector<_Float16> p(static_cast<std::size_t>(kProjOut) * kProjIn);
    for (unsigned o = 0; o < kProjOut; ++o) {
        const float* row = o < kProjQ ? wq_rows + static_cast<std::size_t>(o) * kProjIn
                                      : wk_rows + static_cast<std::size_t>(o - kProjQ) * kProjIn;
        for (unsigned i = 0; i < kProjIn; ++i) {
            const auto h = static_cast<_Float16>(row[i]);
            if (static_cast<float>(h) != row[i]) throw std::invalid_argument("pack_index_proj: weight is not an F16 value");
            p[(static_cast<std::size_t>(i / 8) * kProjOut + o) * 8 + i % 8] = h;
        }
    }
    return p;
}

void enqueue_index_proj_decode(const IndexProjView& v, unsigned ring, hipStream_t stream, unsigned rows_per_wave) {
    if (rows_per_wave != 0 && rows_per_wave != 2) throw std::invalid_argument("index_proj decode: rows_per_wave must be 0 or 2");
    if (v.rows == 0 || v.rows > 8) throw std::invalid_argument("index_proj decode: rows must be 1..8");
    switch (ring) {
    case 8: launch_decode<8>(v, rows_per_wave, stream); break;
    case 16: launch_decode<16>(v, rows_per_wave, stream); break;
    case 32: launch_decode<32>(v, rows_per_wave, stream); break;
    case 48: launch_decode<48>(v, rows_per_wave, stream); break;
    default: throw std::invalid_argument("index_proj decode: ring must be 8, 16, 32 or 48");
    }
    hip_check(hipGetLastError(), "index_proj decode");
}

void enqueue_index_proj_diag(const IndexProjView& v, int mode, hipStream_t stream) {
    const dim3 g1(kProjOut / 32), g2(kProjOut / 32, (v.rows + 1) / 2);
    if (v.rows == 1) {
        switch (mode) {
        case 1: hipLaunchKernelGGL((proj_decode_kernel<1, 32, 1>), g1, dim3(32), 0, stream, v); break;
        case 2: hipLaunchKernelGGL((proj_decode_kernel<1, 32, 2>), g1, dim3(32), 0, stream, v); break;
        case 3: hipLaunchKernelGGL((proj_decode_kernel<1, 32, 3>), g1, dim3(32), 0, stream, v); break;
        default: hipLaunchKernelGGL((proj_decode_kernel<1, 32, 0>), g1, dim3(32), 0, stream, v); break;
        }
    } else {
        switch (mode) {
        case 1: hipLaunchKernelGGL((proj_decode_kernel<2, 16, 1>), g2, dim3(32), 0, stream, v); break;
        case 2: hipLaunchKernelGGL((proj_decode_kernel<2, 16, 2>), g2, dim3(32), 0, stream, v); break;
        case 3: hipLaunchKernelGGL((proj_decode_kernel<2, 16, 3>), g2, dim3(32), 0, stream, v); break;
        default: hipLaunchKernelGGL((proj_decode_kernel<2, 16, 0>), g2, dim3(32), 0, stream, v); break;
        }
    }
    hip_check(hipGetLastError(), "index_proj diag");
}

void enqueue_index_proj_decode_lds(const IndexProjView& v, hipStream_t stream) {
    if (v.rows < 2 || v.rows > 8) throw std::invalid_argument("index_proj decode lds: rows must be 2..8");
    hipLaunchKernelGGL(proj_decode_lds_kernel, dim3(kProjOut / 32), dim3(kLdsWaves * 32), 0, stream, v);
    hip_check(hipGetLastError(), "index_proj decode lds");
}

void enqueue_index_proj_tiled(const IndexProjView& v, hipStream_t stream) {
    if (v.rows < kTokPerWave) throw std::invalid_argument("index_proj tiled: rows must be >= 8 (use the decode kernel)");
    const dim3 grid(kProjOut / 32, (v.rows + kWaves * kTokPerWave - 1) / (kWaves * kTokPerWave));
    hipLaunchKernelGGL(proj_tiled_kernel, grid, dim3(kWaves * 32), 0, stream, v);
    hip_check(hipGetLastError(), "index_proj tiled");
}

void enqueue_index_proj_slab(const IndexProjView& v, hipStream_t stream) {
    if (v.rows < kTokPerWave) throw std::invalid_argument("index_proj slab: rows must be >= 8 (use the decode kernel)");
    const dim3 grid(kProjOut / 32, (v.rows + kWaves * kTokPerWave - 1) / (kWaves * kTokPerWave));
    hipLaunchKernelGGL((proj_slab_kernel<kTokPerWave, 8>), grid, dim3(kWaves * 32), 0, stream, v);
    hip_check(hipGetLastError(), "index_proj slab");
}

template <unsigned TC> void launch_wmma(const IndexProjView& v, hipStream_t stream, bool rows_fast) {
    const unsigned rb = (v.rows + tall::kRows - 1U) / tall::kRows, cb = kWmmaTiles / TC;
    if (rows_fast) hipLaunchKernelGGL((proj_wmma_kernel<TC, false>), dim3(rb, cb), dim3(256), 0, stream, v);
    else hipLaunchKernelGGL((proj_wmma_kernel<TC, true>), dim3(cb, rb), dim3(256), 0, stream, v);
}
void enqueue_index_proj_wmma(const IndexProjView& v, hipStream_t stream, unsigned tc, bool rows_fast) {
    if (v.rows == 0 || static_cast<std::uint64_t>(v.rows) * kProjIn * 4U >= (std::uint64_t{1} << 32U))
        throw std::invalid_argument("index_proj wmma: rows must be 1 .. 174762");
    switch (tc) {
    case 4: launch_wmma<4>(v, stream, rows_fast); break;
    case 2: launch_wmma<2>(v, stream, rows_fast); break;
    case 1: launch_wmma<1>(v, stream, rows_fast); break;
    default: throw std::invalid_argument("index_proj wmma: tc must be 4, 2 or 1");
    }
    hip_check(hipGetLastError(), "index_proj wmma");
}
void enqueue_index_proj_wmma_x16(const IndexProjView& v, _Float16* xt, hipStream_t stream, bool convert) {
    if (v.rows == 0 || static_cast<std::uint64_t>(v.rows) * kProjIn * 4U >= (std::uint64_t{1} << 32U))
        throw std::invalid_argument("index_proj wmma x16: rows must be 1 .. 174762");
    if (convert) {
        tall::enqueue_x16_tiles(v.x, xt, v.rows, kProjIn, stream);
    }
    hipLaunchKernelGGL(proj_wmma_x16_kernel, dim3(kWmmaTiles / 4U, (v.rows + tall::kRows - 1U) / tall::kRows), dim3(256), 0, stream, v,
                       static_cast<const _Float16*>(xt));
    hip_check(hipGetLastError(), "index_proj wmma x16");
}
bool index_proj_fast_tiles(unsigned rows) { return rows >= kFastTilesRows; }
void enqueue_index_proj_fast(const IndexProjView& v, _Float16* xt, hipStream_t stream) {
    if (index_proj_fast_tiles(v.rows)) enqueue_index_proj_wmma_x16(v, xt, stream);
    else enqueue_index_proj_wmma(v, stream);
}
void enqueue_index_proj_wmma_diag(const IndexProjView& v, int mode, hipStream_t stream) {
    if (v.rows == 0 || static_cast<std::uint64_t>(v.rows) * kProjIn * 4U >= (std::uint64_t{1} << 32U))
        throw std::invalid_argument("index_proj wmma diag: rows must be 1 .. 174762");
    const dim3 grid(kWmmaTiles / 4U, (v.rows + tall::kRows - 1U) / tall::kRows);
    if (mode == 1) hipLaunchKernelGGL(proj_wmma_xcached_kernel, grid, dim3(256), 0, stream, v);
    else if (mode == 2) hipLaunchKernelGGL(proj_wmma_cap_kernel, grid, dim3(256), 0, stream, v);
    else if (mode == 3) hipLaunchKernelGGL((proj_wmma_kernel<4, true>), dim3(1, grid.y), dim3(256), 0, stream, v);
    else throw std::invalid_argument("index_proj wmma diag: mode must be 1, 2 or 3");
    hip_check(hipGetLastError(), "index_proj wmma diag");
}

void enqueue_index_proj(const IndexProjView& v, hipStream_t stream) {
    // Measured (ABA x3): one row = ring 32; two or more rows = ring 16, row pairs per wave (never slower).
    // Preserve the original lazy environment capture order: prefill must not
    // initialize the decode-only LDS/split defaults before the first decode.
    const unsigned ring = ring_override();
    const unsigned lds = v.rows >= 2 && v.rows <= 8 ? lds_default() : 0U;
    const unsigned split = v.rows <= 8 && v.rows > 2 && lds == 0 ? split_default() : 0U;
    const auto choice = route::index_route(v.rows, ring, split, lds);
    if (choice.lds) enqueue_index_proj_decode_lds(v, stream);
    else if (!choice.tiled) enqueue_index_proj_decode(v, choice.ring, stream, choice.rows_per_wave);
    else enqueue_index_proj_slab(v, stream);  // same bits as enqueue_index_proj_tiled
}

void index_proj_fork(const IndexProjView& v, hipStream_t main, hipStream_t side, hipEvent_t forked, _Float16* fast_xt) {
    hip_check(hipEventRecord(forked, main), "index_proj fork record");
    hip_check(hipStreamWaitEvent(side, forked, 0), "index_proj fork wait");
    if (fast_xt != nullptr) enqueue_index_proj_fast(v, fast_xt, side);
    else enqueue_index_proj(v, side);
}

void index_proj_join(hipStream_t main, hipStream_t side, hipEvent_t joined) {
    hip_check(hipEventRecord(joined, side), "index_proj join record");
    hip_check(hipStreamWaitEvent(main, joined, 0), "index_proj join wait");
}

}  // namespace strix::minimax::attn
