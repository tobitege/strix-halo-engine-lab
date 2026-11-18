// deepseek41 routed experts on CUDA: A8, device routes, decode GEMVs (gate/up IQ2_XXS + SwiGLU + A8, down Q2_K),
// the combine and the layer driver. Declared arithmetic: cuda_ds41_moe.hpp and docs/models/deepseek-v4.1/cuda-numeric-identity.md.

#include "../../core/reduce.cuh"
#include "ds41_moe_device.cuh"

#include "strix/ds41_formats.hpp"

#include <algorithm>
#include <cstring>

namespace strix::ds41::cuda {

using namespace moe_detail;

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr std::uint32_t kMaxRowWordsPerLane = 8;  // weight row <= 8 words per lane (IQ2 hidden <= 7936)

__constant__ std::uint64_t c_iq2_grid[256];
bool g_grid_ready = false;

void upload_grid() {
    if (g_grid_ready) return;
    STRIX_CUDA_CHECK(cudaMemcpyToSymbol(c_iq2_grid, iq2_xxs_grid.data(), sizeof(std::uint64_t) * 256U));
    g_grid_ready = true;
}

// ------------------------------------------------------------------------------------------------------ routes
// Block-wide exclusive scan of one value per thread (blockDim.x a multiple of 32, <= 1024); returns the total.
using ::strix::cuda::core::block_exclusive_scan;

struct RouteArgs {
    const std::uint32_t* ids;
    std::uint32_t rows, top_k, experts, split, first, end, base;
    MoeRoutes out;
};

// Fill the entries of expert e (local index le) from position `offset`, in ascending slot, one warp.
__device__ void fill_expert(const RouteArgs& a, std::uint32_t e, std::uint32_t offset) {
    const std::uint32_t lane = threadIdx.x & 31U, slots = a.rows * a.top_k;
    for (std::uint32_t s0 = 0; s0 < slots; s0 += 32U) {
        const std::uint32_t s = s0 + lane;
        const bool match = s < slots && a.ids[s] == e;
        const unsigned ballot = __ballot_sync(kFull, match);
        if (match) {
            const std::uint32_t pos = offset + __popc(ballot & ((1U << lane) - 1U));
            a.out.rows[pos] = s / a.top_k;
            a.out.slots[pos] = s;
        }
        offset += __popc(ballot);
    }
}

// Per row: its slots in ascending expert id and the count below split (one thread per row; top_k <= 16).
__device__ void row_order(const RouteArgs& a, std::uint32_t r) {
    std::uint32_t id[16], slot[16];
    for (std::uint32_t k = 0; k < a.top_k; ++k) {
        const std::uint32_t s = r * a.top_k + k, e = a.ids[s];
        std::uint32_t at = k;
        while (at > 0U && id[at - 1U] > e) { id[at] = id[at - 1U]; slot[at] = slot[at - 1U]; --at; }
        id[at] = e; slot[at] = s;
    }
    std::uint32_t low = 0;
    for (std::uint32_t k = 0; k < a.top_k; ++k) {
        a.out.order[r * a.top_k + k] = slot[k];
        low += id[k] < a.split ? 1U : 0U;
    }
    a.out.low_counts[r] = low;
}

// Union, begins and counts from per-expert counts in `count` (local index), one block of 1024 threads.
// Returns nothing; offsets of each local expert are left in `count` (shared) for the fill.
__device__ void routes_scan(const RouteArgs& a, std::uint32_t* count, std::uint32_t range, std::uint32_t* warp_sums) {
    __shared__ std::uint32_t s_max;
    if (threadIdx.x == 0U) s_max = 0U;
    __syncthreads();
    // one thread per local expert (range <= 1024)
    const std::uint32_t le = threadIdx.x;
    std::uint32_t c = le < range ? count[le] : 0U, present = c > 0U ? 1U : 0U, offset = c;
    if (c) atomicMax(&s_max, c);
    const std::uint32_t unions = block_exclusive_scan(present, warp_sums);
    const std::uint32_t total = block_exclusive_scan(offset, warp_sums);
    if (le < range && c) {
        a.out.experts[present] = le + a.first - a.base;
        a.out.begins[present] = offset;
    }
    if (le < range) count[le] = offset;
    if (threadIdx.x == 0U) {
        a.out.begins[unions] = total;
        a.out.counts[0] = unions;
        a.out.counts[1] = s_max;
    }
    __syncthreads();
}

__global__ __launch_bounds__(1024) void routes_small_kernel(RouteArgs a) {
    __shared__ std::uint32_t count[1024];
    __shared__ std::uint32_t warp_sums[33];
    const std::uint32_t range = a.end - a.first, slots = a.rows * a.top_k;
    for (std::uint32_t i = threadIdx.x; i < range; i += blockDim.x) count[i] = 0U;
    __syncthreads();
    for (std::uint32_t s = threadIdx.x; s < slots; s += blockDim.x) {
        const std::uint32_t e = a.ids[s];
        if (e >= a.first && e < a.end) atomicAdd(&count[e - a.first], 1U);
    }
    __syncthreads();
    // counts are needed for the fill: keep them while the scan replaces them by offsets
    std::uint32_t mine = threadIdx.x < range ? count[threadIdx.x] : 0U;
    routes_scan(a, count, range, warp_sums);
    // fill: warp w takes local experts w, w + 32, ...; only non-empty ones scan the slots
    const std::uint32_t warp = threadIdx.x / 32U, lane = threadIdx.x & 31U;
    __shared__ std::uint32_t present[1024];
    if (threadIdx.x < range) present[threadIdx.x] = mine;
    __syncthreads();
    for (std::uint32_t le = warp; le < range; le += 32U)
        if (present[le]) fill_expert(a, le + a.first, count[le]);
    for (std::uint32_t r = threadIdx.x; r < a.rows; r += blockDim.x) row_order(a, r);
    (void)lane;
}

__global__ __launch_bounds__(256) void routes_count_kernel(RouteArgs a, std::uint32_t* count) {
    const std::uint32_t slots = a.rows * a.top_k;
    for (std::uint32_t s = blockIdx.x * blockDim.x + threadIdx.x; s < slots; s += gridDim.x * blockDim.x) {
        const std::uint32_t e = a.ids[s];
        if (e >= a.first && e < a.end) atomicAdd(&count[e - a.first], 1U);
    }
}
__global__ __launch_bounds__(1024) void routes_scan_kernel(RouteArgs a, std::uint32_t* count) {
    __shared__ std::uint32_t s_count[1024];
    __shared__ std::uint32_t warp_sums[33];
    const std::uint32_t range = a.end - a.first;
    if (threadIdx.x < range) s_count[threadIdx.x] = count[threadIdx.x];
    __syncthreads();
    const std::uint32_t mine = threadIdx.x < range ? s_count[threadIdx.x] : 0U;
    routes_scan(a, s_count, range, warp_sums);
    // offsets replace counts for the fill; empty experts get UINT32_MAX
    if (threadIdx.x < range) count[threadIdx.x] = mine ? s_count[threadIdx.x] : 0xffffffffU;
}
__global__ __launch_bounds__(256) void routes_fill_kernel(RouteArgs a, const std::uint32_t* offsets) {
    const std::uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32U, range = a.end - a.first;
    if (blockIdx.y == 1U) {
        for (std::uint32_t r = blockIdx.x * blockDim.x + threadIdx.x; r < a.rows; r += gridDim.x * blockDim.x) row_order(a, r);
        return;
    }
    if (warp < range && offsets[warp] != 0xffffffffU) fill_expert(a, warp + a.first, offsets[warp]);
}

// ---------------------------------------------------------------------------------------------- decode gate/up
struct GateUpParams {
    const std::uint8_t* gate;
    const std::uint8_t* up;
    std::uint64_t expert_stride, row_bytes;
    std::uint32_t hidden, inter;
    const std::uint32_t *experts, *begins, *rows, *slots, *counts;
    const std::int8_t* x_codes;
    const float* x_deltas;
    float limit;
    std::int8_t* act_codes;
    float* act_deltas;
    std::int32_t* act_sums;
    float* gate_out;
    float* up_out;
};

// A block owns one union expert and 32 outputs (one A8 group of the activation); each of its 8 warps owns 4
// outputs (gate and up rows). Entries are staged R at a time. Lane l takes groups l, l + 32, ... of every row
// in ascending order: the declared lane classes.
template <int R>
__global__ __launch_bounds__(kThreads) void gate_up_gemv_kernel(GateUpParams p) {
    const std::uint32_t j = blockIdx.y;
    if (j >= p.counts[0]) return;
    extern __shared__ __align__(16) unsigned char smem[];
    const std::uint32_t G = p.hidden / 32U, J = (G + 31U) / 32U, words = static_cast<std::uint32_t>(p.row_bytes / 8U);
    auto* xs = reinterpret_cast<std::int32_t*>(smem);                         // [R][J][8][32]
    auto* xd = reinterpret_cast<float*>(xs + R * J * 256U);                   // [R][G]
    auto* rowbuf = reinterpret_cast<std::uint64_t*>(xd + ((R * G + 3U) & ~3U));  // [warps][2][words]
    auto* gu = reinterpret_cast<float*>(rowbuf + kWarps * 2U * words);        // [R][32][2]
    __shared__ std::uint64_t grid[256];
    for (std::uint32_t i = threadIdx.x; i < 256U; i += kThreads) grid[i] = c_iq2_grid[i];

    const std::uint32_t lane = threadIdx.x & 31U, warp = threadIdx.x / 32U;
    const std::uint64_t expert = p.experts[j];
    const std::uint32_t b0 = p.begins[j], b1 = p.begins[j + 1U];
    const std::uint8_t* gate_base = p.gate + expert * p.expert_stride;
    const std::uint8_t* up_base = p.up + expert * p.expert_stride;
    std::uint64_t* buf = rowbuf + warp * 2U * words;
    const auto* bytes = reinterpret_cast<const unsigned char*>(buf);

    for (std::uint32_t c0 = b0; c0 < b1; c0 += R) {
        const std::uint32_t n = min(static_cast<std::uint32_t>(R), b1 - c0);
        __syncthreads();
        // stage the entries' A8 rows: word q of group g = lane + 32 jj at xs[((r J + jj) 8 + q) 32 + lane]
        for (std::uint32_t i = threadIdx.x; i < n * J * 256U; i += kThreads) {
            const std::uint32_t l = i & 31U, q = (i >> 5U) & 7U, jj = (i >> 8U) % J, r = (i >> 8U) / J, g = l + 32U * jj;
            std::int32_t w = 0;
            if (g < G) w = reinterpret_cast<const std::int32_t*>(p.x_codes + std::uint64_t{p.rows[c0 + r]} * p.hidden)[g * 8U + q];
            xs[i] = w;
        }
        for (std::uint32_t i = threadIdx.x; i < n * G; i += kThreads)
            xd[i] = p.x_deltas[std::uint64_t{p.rows[c0 + i / G]} * G + i % G];
        __syncthreads();

        // this warp's 4 outputs; prefetch the next output's gate and up rows into registers
        const std::uint32_t o0 = blockIdx.x * 32U + warp * 4U;
        uint2 pre[2][kMaxRowWordsPerLane];
        auto load = [&](std::uint32_t o) {
            const auto* g = reinterpret_cast<const uint2*>(gate_base + o * p.row_bytes);
            const auto* u = reinterpret_cast<const uint2*>(up_base + o * p.row_bytes);
#pragma unroll
            for (std::uint32_t k = 0; k < kMaxRowWordsPerLane; ++k) {
                const std::uint32_t w = lane + 32U * k;
                if (w < words) { pre[0][k] = __ldg(g + w); pre[1][k] = __ldg(u + w); }
            }
        };
        load(o0);
        for (std::uint32_t i = 0; i < 4U; ++i) {
#pragma unroll
            for (std::uint32_t k = 0; k < kMaxRowWordsPerLane; ++k) {
                const std::uint32_t w = lane + 32U * k;
                if (w < words) {
                    buf[w] = (std::uint64_t{pre[0][k].y} << 32U) | pre[0][k].x;
                    buf[words + w] = (std::uint64_t{pre[1][k].y} << 32U) | pre[1][k].x;
                }
            }
            __syncwarp();
            if (i + 1U < 4U) load(o0 + i + 1U);
            float acc[2][R];
#pragma unroll
            for (int r = 0; r < R; ++r) { acc[0][r] = 0.0F; acc[1][r] = 0.0F; }
            for (std::uint32_t jj = 0; jj < J; ++jj) {
                const std::uint32_t g = lane + 32U * jj;
                if (g >= G) break;
                const std::uint32_t blk = g >> 3U, s = g & 7U;
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                    const unsigned char* b = bytes + m * words * 8U + blk * 66U;
                    const std::uint32_t d = b[0] | (std::uint32_t{b[1]} << 8U);
                    const unsigned char* q = b + 2U + 8U * s;
                    const std::uint32_t aux0 = q[0] | (std::uint32_t{q[1]} << 8U) | (std::uint32_t{q[2]} << 16U) | (std::uint32_t{q[3]} << 24U);
                    const std::uint32_t aux1 = q[4] | (std::uint32_t{q[5]} << 8U) | (std::uint32_t{q[6]} << 16U) | (std::uint32_t{q[7]} << 24U);
                    const float scale = (half_bits(d) * (0.5F + static_cast<float>(aux1 >> 28U))) * 0.25F;
                    std::uint32_t wv[8];
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const std::uint64_t gr = grid[(aux0 >> (8 * l)) & 0xffU];
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
        // epilogue: warp r quantizes entry r's 32 activations
        for (std::uint32_t r = warp; r < n; r += kWarps) {
            const float gv = gu[(r * 32U + lane) * 2U], uv = gu[(r * 32U + lane) * 2U + 1U];
            const std::uint64_t slot = p.slots[c0 + r], o = blockIdx.x * 32U + lane;
            if (p.gate_out) { p.gate_out[slot * p.inter + o] = gv; p.up_out[slot * p.inter + o] = uv; }
            const A8Group q = a8_group(swiglu_clamped(gv, uv, p.limit));
            p.act_codes[slot * p.inter + o] = q.code;
            if (lane == 0U) {
                p.act_deltas[slot * (p.inter / 32U) + blockIdx.x] = q.delta;
                if (p.act_sums) p.act_sums[slot * (p.inter / 32U) + blockIdx.x] = q.sum;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------- decode down
struct DownParams {
    const std::uint8_t* down;
    std::uint64_t expert_stride, row_bytes;
    std::uint32_t hidden, inter;
    const std::uint32_t *experts, *begins, *slots, *counts;
    const std::int8_t* act_codes;
    const float* act_deltas;
    float* raw;
    float* raw1;  // expert TP (ds41-expert-tp2-i1152-v1): raw = rank 0's K-half part, raw1 = rank 1's
};
constexpr std::uint32_t kDownOutputsPerWarp = 8;
constexpr std::uint32_t kDownOutputs = kWarps * kDownOutputsPerWarp;  // per block

// One Q2_K group's declared term (d * float(A) - dmin * float(B)) * delta of staged entry r.
__device__ __forceinline__ float q2k_group_term(const unsigned char* bytes, const std::int32_t* xs, const float* xd, const std::int32_t* hs,
                                                std::uint32_t J, std::uint32_t G, std::uint32_t g, int r) {
    const std::uint32_t blk = g >> 3U, s = g & 7U, half = s >> 2U, shift = 2U * (s & 3U);
    const unsigned char* b = bytes + blk * 84U;
    const std::uint32_t dd = *reinterpret_cast<const std::uint32_t*>(b + 80U);
    const float d = half_bits(dd & 0xffffU), dmin = half_bits(dd >> 16U);
    const std::uint32_t sc0 = b[8U * half + 2U * (s & 3U)], sc1 = b[8U * half + 2U * (s & 3U) + 1U];
    const auto* qw = reinterpret_cast<const std::uint32_t*>(b + 16U + 32U * half);
    const std::int32_t* xr = xs + ((r * J + g / 32U) * 8U) * 32U + (g & 31U);
    int qx0 = 0, qx1 = 0;
#pragma unroll
    for (int w = 0; w < 4; ++w) qx0 = __dp4a(static_cast<int>((qw[w] >> shift) & 0x03030303U), xr[w * 32], qx0);
#pragma unroll
    for (int w = 4; w < 8; ++w) qx1 = __dp4a(static_cast<int>((qw[w] >> shift) & 0x03030303U), xr[w * 32], qx1);
    const std::int32_t a = static_cast<std::int32_t>(sc0 & 15U) * qx0 + static_cast<std::int32_t>(sc1 & 15U) * qx1;
    const std::int32_t bsum = static_cast<std::int32_t>(sc0 >> 4U) * hs[r * 2U * G + 2U * g] +
                              static_cast<std::int32_t>(sc1 >> 4U) * hs[r * 2U * G + 2U * g + 1U];
    const float scaled = d * static_cast<float>(a);
    const float offset = dmin * static_cast<float>(bsum);
    return (scaled - offset) * xd[r * G + g];
}

// A block owns one union expert and 64 outputs; each warp 8 outputs. Entries staged R at a time with their
// 16-element half sums. TP (ds41-expert-tp2-i1152-v1, inter 2304 = 72 groups): every output is two K-half products,
// each with its own lane classes and butterfly: rank 0's local group g is global g (lane l: groups l, then l + 32 for
// l < 4); rank 1's local g is global 40 + g (g < 32) or 36 + (g - 32) (lane l: group 40 + l, then 36 + l for l < 4).
// The group terms are those of the unsplit product.
template <int R, bool TP>
__global__ __launch_bounds__(kThreads) void down_gemv_kernel(DownParams p) {
    const std::uint32_t j = blockIdx.y;
    if (j >= p.counts[0]) return;
    extern __shared__ __align__(16) unsigned char smem[];
    const std::uint32_t G = p.inter / 32U, J = (G + 31U) / 32U, words = static_cast<std::uint32_t>(p.row_bytes / 4U);
    auto* xs = reinterpret_cast<std::int32_t*>(smem);               // [R][J][8][32]
    auto* xd = reinterpret_cast<float*>(xs + R * J * 256U);         // [R][G]
    auto* hs = reinterpret_cast<std::int32_t*>(xd + R * G);         // [R][2G] half sums
    auto* rowbuf = reinterpret_cast<std::uint32_t*>(hs + R * 2U * G);  // [warps][words]

    const std::uint32_t lane = threadIdx.x & 31U, warp = threadIdx.x / 32U;
    const std::uint64_t expert = p.experts[j];
    const std::uint32_t b0 = p.begins[j], b1 = p.begins[j + 1U];
    const std::uint8_t* base = p.down + expert * p.expert_stride;
    std::uint32_t* buf = rowbuf + warp * words;
    const auto* bytes = reinterpret_cast<const unsigned char*>(buf);

    for (std::uint32_t c0 = b0; c0 < b1; c0 += R) {
        const std::uint32_t n = min(static_cast<std::uint32_t>(R), b1 - c0);
        __syncthreads();
        for (std::uint32_t i = threadIdx.x; i < n * J * 256U; i += kThreads) {
            const std::uint32_t l = i & 31U, q = (i >> 5U) & 7U, jj = (i >> 8U) % J, r = (i >> 8U) / J, g = l + 32U * jj;
            std::int32_t w = 0;
            if (g < G) w = reinterpret_cast<const std::int32_t*>(p.act_codes + std::uint64_t{p.slots[c0 + r]} * p.inter)[g * 8U + q];
            xs[i] = w;
        }
        for (std::uint32_t i = threadIdx.x; i < n * G; i += kThreads)
            xd[i] = p.act_deltas[std::uint64_t{p.slots[c0 + i / G]} * G + i % G];
        for (std::uint32_t i = threadIdx.x; i < n * 2U * G; i += kThreads) {
            const std::uint32_t r = i / (2U * G), h = i % (2U * G);
            const std::int8_t* x = p.act_codes + std::uint64_t{p.slots[c0 + r]} * p.inter + h * 16U;
            std::int32_t s = 0;
            for (int e = 0; e < 16; ++e) s += x[e];
            hs[i] = s;
        }
        __syncthreads();

        const std::uint32_t o0 = blockIdx.x * kDownOutputs + warp * kDownOutputsPerWarp;
        std::uint32_t pre[kMaxRowWordsPerLane];
        auto load = [&](std::uint32_t o) {
            const auto* w32 = reinterpret_cast<const std::uint32_t*>(base + o * p.row_bytes);
#pragma unroll
            for (std::uint32_t k = 0; k < kMaxRowWordsPerLane; ++k) {
                const std::uint32_t w = lane + 32U * k;
                if (w < words) pre[k] = __ldg(w32 + w);
            }
        };
        if (o0 < p.hidden) load(o0);
        for (std::uint32_t i = 0; i < kDownOutputsPerWarp && o0 + i < p.hidden; ++i) {
#pragma unroll
            for (std::uint32_t k = 0; k < kMaxRowWordsPerLane; ++k) {
                const std::uint32_t w = lane + 32U * k;
                if (w < words) buf[w] = pre[k];
            }
            __syncwarp();
            if (i + 1U < kDownOutputsPerWarp && o0 + i + 1U < p.hidden) load(o0 + i + 1U);
            if constexpr (TP) {
                float acc0[R], acc1[R];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    acc0[r] = 0.0F;
                    acc1[r] = 0.0F;
                    if (r < static_cast<int>(n)) {
                        acc0[r] = acc0[r] + q2k_group_term(bytes, xs, xd, hs, J, G, lane, r);
                        if (lane < 4U) acc0[r] = acc0[r] + q2k_group_term(bytes, xs, xd, hs, J, G, lane + 32U, r);
                        acc1[r] = acc1[r] + q2k_group_term(bytes, xs, xd, hs, J, G, 40U + lane, r);
                        if (lane < 4U) acc1[r] = acc1[r] + q2k_group_term(bytes, xs, xd, hs, J, G, 36U + lane, r);
                    }
                }
                __syncwarp();
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (r < static_cast<int>(n)) {
                        const float v0 = butterfly(acc0[r]), v1 = butterfly(acc1[r]);
                        if (lane == 0U) {
                            p.raw[std::uint64_t{p.slots[c0 + r]} * p.hidden + o0 + i] = v0;
                            p.raw1[std::uint64_t{p.slots[c0 + r]} * p.hidden + o0 + i] = v1;
                        }
                    }
                }
                continue;
            }
            float acc[R];
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = 0.0F;
            for (std::uint32_t jj = 0; jj < J; ++jj) {
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
            __syncwarp();
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (r < static_cast<int>(n)) {
                    const float v = butterfly(acc[r]);
                    if (lane == 0U) p.raw[std::uint64_t{p.slots[c0 + r]} * p.hidden + o0 + i] = v;
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------------------------------- combine
__global__ __launch_bounds__(kThreads) void combine_kernel(MoeCombineView c, int mode) {
    const std::uint64_t i = std::uint64_t{blockIdx.x} * kThreads + threadIdx.x;
    if (i >= std::uint64_t{c.rows} * c.hidden) return;
    const std::uint32_t r = static_cast<std::uint32_t>(i / c.hidden), o = static_cast<std::uint32_t>(i % c.hidden);
    const std::uint32_t low_count = c.low_counts[r];
    float low = 0.0F, high = 0.0F;
    // steps of 8 slots: the step's slot ids, then its weights and raw values, are loaded before the same in-order sums
    constexpr std::uint32_t kStep = 8;
    for (std::uint32_t k0 = 0; k0 < c.top_k; k0 += kStep) {
        std::uint64_t slot[kStep];
        float wv[kStep], rv[kStep];
#pragma unroll
        for (std::uint32_t q = 0; q < kStep; ++q) slot[q] = k0 + q < c.top_k ? c.order[r * c.top_k + k0 + q] : 0U;
#pragma unroll
        for (std::uint32_t q = 0; q < kStep; ++q) {
            const std::uint32_t k = k0 + q;
            const bool is_low = k < low_count;
            const bool use = k < c.top_k && !((mode == 1 && !is_low) || (mode == 2 && is_low));
            wv[q] = use ? c.weights[slot[q]] : 0.0F;
            rv[q] = use ? c.raw[slot[q] * c.hidden + o] : 0.0F;
        }
#pragma unroll
        for (std::uint32_t q = 0; q < kStep; ++q) {
            const std::uint32_t k = k0 + q;
            if (k >= c.top_k) break;
            const bool is_low = k < low_count;
            if ((mode == 1 && !is_low) || (mode == 2 && is_low)) continue;
            const float term = wv[q] * rv[q];
            if (is_low) low = low + term;
            else high = high + term;
        }
    }
    const float y = mode == 0 ? (c.shared[i] + low) + high : mode == 1 ? low : high;
    c.out[i] = y;
    if (mode == 0 && c.hc_out) {  // hc_expand_kernel's arithmetic for (r, o) with sub = y
        const float* mix = c.hc_mixes + std::size_t{r} * 24U;
        if (c.hc_pre_carry && o == 0U)
            for (int q = 0; q < 4; ++q) c.hc_pre_carry[std::size_t{r} * 4U + q] = mix[q];
        float res[4];
        for (int j = 0; j < 4; ++j) res[j] = c.hc_residual[(std::size_t{r} * 4U + j) * c.hidden + o];
        float yk[4];
        for (int k = 0; k < 4; ++k) {
            float s = 0.0F;
            for (int j = 0; j < 4; ++j) s = s + mix[8 + j * 4 + k] * res[j];
            yk[k] = mix[4 + k] * y + s;
        }
        for (int k = 0; k < 4; ++k) c.hc_out[(std::size_t{r} * 4U + k) * c.hidden + o] = yk[k];
    }
}
// Expert TP combine: each part folded over the row's slots in ascending expert id from 0, then (shared + S0) + S1.
__global__ __launch_bounds__(kThreads) void combine_tp_kernel(MoeCombineView c) {
    const std::uint64_t i = std::uint64_t{blockIdx.x} * kThreads + threadIdx.x;
    if (i >= std::uint64_t{c.rows} * c.hidden) return;
    const std::uint32_t r = static_cast<std::uint32_t>(i / c.hidden), o = static_cast<std::uint32_t>(i % c.hidden);
    float s0 = 0.0F, s1 = 0.0F;
    for (std::uint32_t k = 0; k < c.top_k; ++k) {
        const std::uint64_t slot = c.order[r * c.top_k + k];
        s0 = s0 + c.weights[slot] * c.raw[slot * c.hidden + o];
    }
    for (std::uint32_t k = 0; k < c.top_k; ++k) {
        const std::uint64_t slot = c.order[r * c.top_k + k];
        s1 = s1 + c.weights[slot] * c.raw1[slot * c.hidden + o];
    }
    c.out[i] = (c.shared[i] + s0) + s1;
}
__global__ __launch_bounds__(kThreads) void canonical_kernel(const float* shared, const float* low, const float* high, std::size_t count,
                                                             float* out) {
    const std::size_t i = std::size_t{blockIdx.x} * kThreads + threadIdx.x;
    if (i < count) out[i] = (shared[i] + low[i]) + high[i];
}

template <class Kernel>
void set_smem(Kernel kernel, std::size_t bytes) {
    if (bytes > 48U * 1024U)
        STRIX_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
}

std::uint32_t pick_rows(std::uint32_t max_rows) { return max_rows <= 1U ? 1U : max_rows <= 2U ? 2U : max_rows <= 4U ? 4U : 8U; }

void check_shape(const MoeShape& s) {
    if (s.hidden == 0U || s.inter == 0U || s.hidden % 256U || s.inter % 256U || s.gate_row_bytes() % 8U ||
        s.gate_row_bytes() / 8U > 32U * kMaxRowWordsPerLane || s.down_row_bytes() / 4U > 32U * kMaxRowWordsPerLane ||
        s.inter % 32U || s.hidden % kDownOutputs || s.top_k == 0U || s.top_k > 16U || s.experts == 0U || s.experts > 1024U)
        throw std::invalid_argument("ds41 cuda moe: unsupported expert shape");
}

}  // namespace

// ------------------------------------------------------------------------------------------------------- API
void enqueue_moe_routes(const MoeRoutesView& v, cudaStream_t stream) {
    const std::uint32_t end = v.end ? v.end : v.experts;
    if (v.rows == 0U || v.top_k == 0U || v.top_k > 16U || v.experts == 0U || v.experts > 1024U || v.first >= end || end > v.experts ||
        v.base > v.first)
        throw std::invalid_argument("ds41 cuda moe routes: inconsistent arguments");
    const RouteArgs a{v.ids, v.rows, v.top_k, v.experts, v.split, v.first, end, v.base, v.out};
    const std::uint64_t slots = std::uint64_t{v.rows} * v.top_k;
    if (slots <= 4096U) {
        routes_small_kernel<<<1, 1024, 0, stream>>>(a);
        STRIX_CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (!v.scratch) throw std::invalid_argument("ds41 cuda moe routes: more than 4096 slots need scratch");
    STRIX_CUDA_CHECK(cudaMemsetAsync(v.scratch, 0, sizeof(std::uint32_t) * (end - v.first), stream));
    routes_count_kernel<<<static_cast<unsigned>(std::min<std::uint64_t>((slots + 255U) / 256U, 1024U)), 256, 0, stream>>>(a, v.scratch);
    routes_scan_kernel<<<1, 1024, 0, stream>>>(a, v.scratch);
    const unsigned fill_blocks = std::max((end - v.first + 7U) / 8U, (v.rows + 255U) / 256U);
    routes_fill_kernel<<<dim3(fill_blocks, 2), 256, 0, stream>>>(a, v.scratch);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_routed_gate_up(const RoutedGateUpView& v, cudaStream_t stream) {
    check_shape(v.shape);
    if (!v.max_union) return;
    upload_grid();
    const std::uint32_t x_stride = v.x.stride ? v.x.stride : v.shape.hidden;
    if (x_stride != v.shape.hidden || (v.act.stride && v.act.stride != v.shape.inter))
        throw std::invalid_argument("ds41 cuda moe gate/up: A8 strides must equal the widths");
    const GateUpParams p{v.weights.gate, v.weights.up, v.shape.gate_expert_bytes(), v.shape.gate_row_bytes(), v.shape.hidden, v.shape.inter,
                         v.routes.experts, v.routes.begins, v.routes.rows, v.routes.slots, v.routes.counts,
                         v.x.codes, v.x.deltas, v.shape.limit, v.act.codes, v.act.deltas, v.act.sums, v.gate_out, v.up_out};
    const std::uint32_t G = v.shape.hidden / 32U, J = (G + 31U) / 32U, words = static_cast<std::uint32_t>(v.shape.gate_row_bytes() / 8U);
    const std::uint32_t R = pick_rows(std::min(v.max_rows, 8U));
    const std::size_t smem = std::size_t{R} * J * 256U * 4U + ((R * G + 3U) & ~3U) * 4U + kWarps * 2U * words * 8U + R * 64U * 4U;
    const dim3 grid(v.shape.inter / 32U, v.max_union);
    auto go = [&](auto kernel) {
        set_smem(kernel, smem);
        kernel<<<grid, kThreads, smem, stream>>>(p);
    };
    if (R == 1U) go(gate_up_gemv_kernel<1>);
    else if (R == 2U) go(gate_up_gemv_kernel<2>);
    else if (R == 4U) go(gate_up_gemv_kernel<4>);
    else go(gate_up_gemv_kernel<8>);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_routed_down(const RoutedDownView& v, cudaStream_t stream) {
    check_shape(v.shape);
    if (!v.max_union) return;
    if (v.act.stride && v.act.stride != v.shape.inter) throw std::invalid_argument("ds41 cuda moe down: A8 stride must equal inter");
    const DownParams p{v.weights.down, v.shape.down_expert_bytes(), v.shape.down_row_bytes(), v.shape.hidden, v.shape.inter,
                       v.routes.experts, v.routes.begins, v.routes.slots, v.routes.counts, v.act.codes, v.act.deltas, v.raw, v.raw1};
    if (v.raw1 && (v.shape.inter != 2304U || v.shape.down_row_bytes() != 9U * 84U))
        throw std::invalid_argument("ds41 cuda moe down: expert TP is declared for inter 2304 (Q2_K)");
    const std::uint32_t G = v.shape.inter / 32U, J = (G + 31U) / 32U, words = static_cast<std::uint32_t>(v.shape.down_row_bytes() / 4U);
    const std::uint32_t R = pick_rows(std::min(v.max_rows, 8U));
    const std::size_t smem = std::size_t{R} * J * 256U * 4U + std::size_t{R} * G * 4U + std::size_t{R} * 2U * G * 4U + kWarps * words * 4U;
    const dim3 grid(v.shape.hidden / kDownOutputs, v.max_union);
    auto go = [&](auto kernel) {
        set_smem(kernel, smem);
        kernel<<<grid, kThreads, smem, stream>>>(p);
    };
    if (v.raw1) {
        if (R == 1U) go(down_gemv_kernel<1, true>);
        else if (R == 2U) go(down_gemv_kernel<2, true>);
        else if (R == 4U) go(down_gemv_kernel<4, true>);
        else go(down_gemv_kernel<8, true>);
    } else if (R == 1U) go(down_gemv_kernel<1, false>);
    else if (R == 2U) go(down_gemv_kernel<2, false>);
    else if (R == 4U) go(down_gemv_kernel<4, false>);
    else go(down_gemv_kernel<8, false>);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_moe_combine(const MoeCombineView& c, cudaStream_t stream) {
    const std::uint64_t n = std::uint64_t{c.rows} * c.hidden;
    if (!n) return;
    if (c.raw1) {
        if (c.half >= 0) throw std::invalid_argument("ds41 cuda moe combine: expert TP has no half sums");
        combine_tp_kernel<<<static_cast<unsigned>((n + kThreads - 1U) / kThreads), kThreads, 0, stream>>>(c);
        STRIX_CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int mode = c.half < 0 ? 0 : c.half == 0 ? 1 : 2;
    if (c.hc_out && (mode != 0 || !c.hc_residual || !c.hc_mixes))
        throw std::invalid_argument("ds41 cuda moe combine: the hc_expand epilogue needs the whole combine");
    combine_kernel<<<static_cast<unsigned>((n + kThreads - 1U) / kThreads), kThreads, 0, stream>>>(c, mode);
    STRIX_CUDA_CHECK(cudaGetLastError());
}
void enqueue_moe_canonical_sum(const float* shared, const float* low, const float* high, std::size_t count, float* out,
                               cudaStream_t stream) {
    if (!count) return;
    canonical_kernel<<<static_cast<unsigned>((count + kThreads - 1U) / kThreads), kThreads, 0, stream>>>(shared, low, high, count, out);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace strix::ds41::cuda
