// deepseek41 routed experts on CUDA: prefill grouped GEMM and the layer driver.

#include "ds41_moe_device.cuh"

#include "strix/ds41_formats.hpp"

#include <algorithm>

namespace strix::ds41::cuda {

using namespace moe_detail;

namespace {

constexpr std::size_t align_up(std::size_t v) { return (v + 255U) & ~std::size_t{255U}; }

struct LayerSpace {
    MoeRoutes routes;
    std::uint32_t* scratch = nullptr;
    A8Rows act;
    void* grouped = nullptr;
    std::size_t bytes = 0;
};
LayerSpace carve(void* base, std::uint32_t capacity, const MoeShape& s) {
    LayerSpace w;
    auto* p = static_cast<unsigned char*>(base);
    std::size_t at = 0;
    const std::size_t slots = std::size_t{capacity} * s.top_k;
    auto take = [&](std::size_t bytes) {
        void* out = p ? p + at : nullptr;
        at += align_up(bytes);
        return out;
    };
    w.routes.experts = static_cast<std::uint32_t*>(take(4U * slots));
    w.routes.begins = static_cast<std::uint32_t*>(take(4U * (slots + 1U)));
    w.routes.rows = static_cast<std::uint32_t*>(take(4U * slots));
    w.routes.slots = static_cast<std::uint32_t*>(take(4U * slots));
    w.routes.order = static_cast<std::uint32_t*>(take(4U * slots));
    w.routes.low_counts = static_cast<std::uint32_t*>(take(4U * capacity));
    w.routes.counts = static_cast<std::uint32_t*>(take(8U));
    w.scratch = static_cast<std::uint32_t*>(take(4U * s.experts));
    w.act.codes = static_cast<std::int8_t*>(take(slots * s.inter));
    w.act.deltas = static_cast<float*>(take(4U * slots * (s.inter / 32U)));
    w.act.sums = nullptr;
    w.act.rows = capacity * s.top_k;
    w.act.width = w.act.stride = s.inter;
    w.grouped = take(routed_grouped_workspace_bytes(capacity, s));
    w.bytes = at;
    return w;
}

}  // namespace

namespace {

// ------------------------------------------------------------------------------------------- grouped GEMM (IMMA)
// A block owns one batch of up to 64 entries of one expert and 64 weight rows: for gate/up the 32 gate and the 32 up
// rows of 32 outputs (one A8 group of the activation), for down 64 output rows. m16n8k32 IMMAs per 32-weight group
// give the exact int32 dot of every (entry, row) for that group; the group's FP32 term is formed and accumulated per
// element exactly as in the declared product. Groups are visited class by class (class c = groups c, c + 32, ... in
// ascending order: the lane classes of the declared product), the classes in bit-reversed order, so the butterfly's
// tree (pairs (l, l + 16), then (l, l + 8), ...) folds with a five-level carry stack. Per class the weight rows'
// groups are decoded to int8 in shared memory and the entries' A operands copied there (cp.async); both are double
// buffered, so one barrier per class separates the products of class i from the decode and copies of the next one.
// 16 warps: a batch of up to 64 entries is 4 m-tiles x 8 n-tiles, two n-tiles per warp (smaller batches leave
// warps idle in the products, which then only decode and copy).
constexpr int kGThreads = 512;
constexpr std::uint32_t kWarps = kGThreads / 32;
constexpr std::uint32_t kBatch = 64;  // entries per batch
constexpr std::uint32_t kRows = 64;   // weight rows per block
constexpr std::uint32_t kMaxNt = 32U / kWarps;  // n-tiles per warp at 4 m-tiles

__constant__ std::uint64_t g_iq2_grid[256];
bool g_grouped_grid_ready = false;
void upload_grouped_grid() {
    if (g_grouped_grid_ready) return;
    STRIX_CUDA_CHECK(cudaMemcpyToSymbol(g_iq2_grid, iq2_xxs_grid.data(), sizeof(std::uint64_t) * 256U));
    g_grouped_grid_ready = true;
}

struct Batches {
    std::uint32_t* expert;    // union slot
    std::uint32_t* begin;     // first entry
    std::uint32_t* count;     // entries (<= kBatch)
    std::uint32_t* total;     // number of batches
    std::uint32_t* next;      // the next (batch, tile) item: batch * tiles + tile (blocks fetch items dynamically)
    std::int16_t* half_sums;  // [slots][inter / 16] (down)
};
std::uint32_t max_batches(std::uint32_t rows, const MoeShape& s) {
    const std::uint32_t slots = rows * s.top_k;
    return (slots + kBatch - 1U) / kBatch + std::min(slots, s.experts);
}
// The grid has at most kMaxGridY batch rows; its blocks take (batch, tile) items from one counter in the order a full
// grid would run them (a launch restricted to a few experts has few empty blocks; the load stays balanced).
constexpr std::uint32_t kMaxGridY = 48;
constexpr std::uint32_t kMaxTiles = 128;  // output tiles (grid x) the counters cover
// a launch over a restricted expert range (max_union below the expert count: enqueue_moe_layer_ranges) fetches items
// dynamically on a bounded grid; a whole layer keeps one item per block (its grid is mostly busy)
std::uint32_t dynamic_grid(std::uint32_t max_union, const MoeShape& s) { return max_union < s.experts ? 1U : 0U; }
std::uint32_t grid_y(std::uint32_t rows, const MoeShape& s, std::uint32_t dynamic) {
    return dynamic ? std::min(max_batches(rows, s), kMaxGridY) : max_batches(rows, s);
}
std::size_t round256(std::size_t v) { return (v + 255U) & ~std::size_t{255U}; }
Batches carve_batches(void* scratch, std::uint32_t rows, const MoeShape& s) {
    auto* p = static_cast<unsigned char*>(scratch);
    const std::size_t m = max_batches(rows, s);
    Batches b{};
    std::size_t at = 0;
    auto take = [&](std::size_t bytes) {
        void* out = p ? p + at : nullptr;
        at += round256(bytes);
        return out;
    };
    b.expert = static_cast<std::uint32_t*>(take(4U * m));
    b.begin = static_cast<std::uint32_t*>(take(4U * m));
    b.count = static_cast<std::uint32_t*>(take(4U * m));
    b.total = static_cast<std::uint32_t*>(take(4U * (1U + kMaxTiles)));
    b.next = b.total + 1;
    b.half_sums = static_cast<std::int16_t*>(take(2U * std::size_t{rows} * s.top_k * (s.inter / 16U)));
    return b;
}

// Batches of at most kBatch consecutive entries of each union expert, in union order (one block of 1024 threads).
__global__ __launch_bounds__(1024) void batches_kernel(const std::uint32_t* begins, const std::uint32_t* counts, Batches b) {
    __shared__ std::uint32_t warp_sums[33];
    const std::uint32_t unions = counts[0], lane = threadIdx.x & 31U, warp = threadIdx.x / 32U;
    std::uint32_t base = 0;
    for (std::uint32_t j0 = 0; j0 < unions; j0 += 1024U) {
        const std::uint32_t j = j0 + threadIdx.x;
        const std::uint32_t n = j < unions ? (begins[j + 1U] - begins[j] + kBatch - 1U) / kBatch : 0U;
        std::uint32_t inc = n;
        for (int d = 1; d < 32; d *= 2) {
            const std::uint32_t o = __shfl_up_sync(kFull, inc, d);
            if (lane >= static_cast<std::uint32_t>(d)) inc += o;
        }
        if (lane == 31U) warp_sums[warp] = inc;
        __syncthreads();
        if (warp == 0U) {
            const std::uint32_t v = warp_sums[lane];
            std::uint32_t w = v;
            for (int d = 1; d < 32; d *= 2) {
                const std::uint32_t o = __shfl_up_sync(kFull, w, d);
                if (lane >= static_cast<std::uint32_t>(d)) w += o;
            }
            warp_sums[lane] = w - v;
            if (lane == 31U) warp_sums[32] = w;
        }
        __syncthreads();
        const std::uint32_t first = base + warp_sums[warp] + inc - n;
        for (std::uint32_t k = 0; k < n; ++k) {
            const std::uint32_t e0 = begins[j] + k * kBatch;
            b.expert[first + k] = j;
            b.begin[first + k] = e0;
            b.count[first + k] = min(kBatch, begins[j + 1U] - e0);
        }
        base += warp_sums[32];
        __syncthreads();
    }
    if (threadIdx.x == 0U) *b.total = base;
    if (threadIdx.x < kMaxTiles) b.next[threadIdx.x] = 0U;
}

// 16-element half sums of the A8 activation codes of every routed slot (the Q2_K offset term).
__global__ __launch_bounds__(256) void half_sums_kernel(const std::int8_t* codes, std::uint64_t halves, std::int16_t* out) {
    const std::uint64_t i = std::uint64_t{blockIdx.x} * 256U + threadIdx.x;
    if (i >= halves) return;
    const int4 v = *reinterpret_cast<const int4*>(codes + i * 16U);
    int s = 0;
    s = __dp4a(v.x, 0x01010101, s);
    s = __dp4a(v.y, 0x01010101, s);
    s = __dp4a(v.z, 0x01010101, s);
    s = __dp4a(v.w, 0x01010101, s);
    out[i] = static_cast<std::int16_t>(s);
}

__device__ __forceinline__ void mma_s8(std::uint32_t a0, std::uint32_t a1, std::uint32_t a2, std::uint32_t a3, std::uint32_t b0,
                                       std::uint32_t b1, int (&d)[4]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}

struct GroupedParams {
    const std::uint8_t* w0;  // gate (IQ2_XXS) or down (Q2_K) tensor
    const std::uint8_t* w1;  // up (IQ2_XXS)
    std::uint64_t expert_stride, row_bytes;
    std::uint32_t hidden, inter;
    const std::uint32_t *experts, *rows, *slots;
    Batches batches;
    const std::int8_t* a_codes;  // gate/up: x [row][hidden]; down: act [slot][inter]
    const float* a_deltas;
    float limit;
    std::int8_t* act_codes;
    float* act_deltas;
    std::int32_t* act_sums;
    float* gate_out;
    float* up_out;
    float* raw;
    std::uint32_t dynamic;  // blocks fetch (batch, tile) items (a bounded grid) instead of one item per block
};

// ldmatrix of 8x8 b16 tiles (8 rows of 16 bytes): register j of lane l holds bytes 4 (l % 4) .. + 3 of row l / 4 of
// tile j, the m16n8k32 s8 fragment layout of A (tiles: rows 0-7 / 8-15 x bytes 0-15 / 16-31) and of B.
__device__ __forceinline__ void ldsm_x4(unsigned addr, std::uint32_t& r0, std::uint32_t& r1, std::uint32_t& r2, std::uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x2(unsigned addr, std::uint32_t& r0, std::uint32_t& r1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}
// The 32-byte rows of the staged A operands and decoded weights keep their two 16-byte halves swapped when bit 2 of
// the row index is set (ldmatrix reads 8 rows of one half without bank conflicts).
__device__ __forceinline__ std::uint32_t half_slot(std::uint32_t row, std::uint32_t half) { return half ^ ((row >> 2U) & 1U); }

// Shared memory of a grouped block, per class (groups c, c + 32, ... of one lane class), each double buffered: the
// decoded int8 B words [j][row][32 bytes] (IQ2_XXS signed codes; Q2_K codes times the half's scale) with the
// per-(group, row) scalars, and the A operands of the batch's entries [j][entry][32 bytes] with their deltas (and Q2_K
// half sums).
struct GroupedLayout {
    std::uint32_t dec, dec_stride, sc, sc_stride, a, a_stride, ad, ad_stride, ah, raw, bytes;
};

// The products of one staged class for a warp: m-tile msub (16 entries) x NT n-tiles from nt0; acc[k * 4 + q] gets
// the class's terms of the warp's elements, groups in ascending order.
template <bool kIq2, std::uint32_t NT>
__device__ __forceinline__ void class_products(const unsigned char* smem, const GroupedLayout& L, std::uint32_t J, std::uint32_t buf,
                                               std::uint32_t jc, std::uint32_t msub, std::uint32_t nt0, std::uint32_t lane, float* acc) {
    const std::uint32_t gid = lane >> 2U, t = lane & 3U, q = lane >> 3U, r8 = lane & 7U;
    const unsigned dec = static_cast<unsigned>(__cvta_generic_to_shared(smem + L.dec + buf * L.dec_stride));
    const unsigned abuf = static_cast<unsigned>(__cvta_generic_to_shared(smem + L.a + buf * L.a_stride));
    const auto* sc_a = reinterpret_cast<const float*>(smem + L.sc + buf * L.sc_stride);
    const auto* sc_b = sc_a + J * kRows;
    const auto* sc_m = reinterpret_cast<const std::uint32_t*>(sc_b + J * kRows);
    const auto* dbuf = reinterpret_cast<const float*>(smem + L.ad + buf * L.ad_stride);
    const auto* hbuf = reinterpret_cast<const std::int16_t*>(smem + L.ah + buf * L.ad_stride);
    // ldmatrix row addresses: A tile q = rows 8 (q & 1) .. + 7 of the m-tile, half q >> 1; B tile q = n-tile q >> 1,
    // half q & 1 (x2: lanes 0..15, one n-tile)
    const std::uint32_t ae = 16U * msub + r8 + 8U * (q & 1U);
    const unsigned a_off = abuf + ae * 32U + 16U * half_slot(ae, q >> 1U);
    const std::uint32_t br = 8U * (nt0 + (NT == 2U ? (q >> 1U) : 0U)) + r8;
    const unsigned b_off = dec + br * 32U + 16U * half_slot(br, q & 1U);
    const std::uint32_t m0 = 16U * msub + gid, m1 = m0 + 8U;
    for (std::uint32_t jj = 0; jj < jc; ++jj) {
        std::uint32_t a0, a1, a2, a3, b[4];
        ldsm_x4(a_off + jj * kBatch * 32U, a0, a1, a2, a3);
        if (NT == 2U) ldsm_x4(b_off + jj * kRows * 32U, b[0], b[1], b[2], b[3]);
        else ldsm_x2(b_off + jj * kRows * 32U, b[0], b[1]);
        const float dl0 = dbuf[jj * kBatch + m0], dl1 = dbuf[jj * kBatch + m1];
        std::int32_t s00 = 0, s01 = 0, s10 = 0, s11 = 0;
        if (!kIq2) {
            s00 = hbuf[2U * (jj * kBatch + m0)]; s01 = hbuf[2U * (jj * kBatch + m0) + 1U];
            s10 = hbuf[2U * (jj * kBatch + m1)]; s11 = hbuf[2U * (jj * kBatch + m1) + 1U];
        }
#pragma unroll
        for (std::uint32_t k = 0; k < NT; ++k) {
            int dot[4];
            mma_s8(a0, a1, a2, a3, b[2U * k], b[2U * k + 1U], dot);
            const std::uint32_t n0 = jj * kRows + 8U * (nt0 + k) + 2U * t;
            const float2 sa = *reinterpret_cast<const float2*>(sc_a + n0);
            if (kIq2) {
                acc[k * 4 + 0] = acc[k * 4 + 0] + static_cast<float>(dot[0]) * (sa.x * dl0);
                acc[k * 4 + 1] = acc[k * 4 + 1] + static_cast<float>(dot[1]) * (sa.y * dl0);
                acc[k * 4 + 2] = acc[k * 4 + 2] + static_cast<float>(dot[2]) * (sa.x * dl1);
                acc[k * 4 + 3] = acc[k * 4 + 3] + static_cast<float>(dot[3]) * (sa.y * dl1);
            } else {
                const float2 sb = *reinterpret_cast<const float2*>(sc_b + n0);
                const uint2 mm = *reinterpret_cast<const uint2*>(sc_m + n0);
                const std::int32_t ma0 = mm.x & 0xffU, ma1 = mm.x >> 8U, mb0 = mm.y & 0xffU, mb1 = mm.y >> 8U;
                // (d * float(A) - dmin * float(B)) * delta, B = m_h0 * sx_h0 + m_h1 * sx_h1
                acc[k * 4 + 0] = acc[k * 4 + 0] + (sa.x * static_cast<float>(dot[0]) - sb.x * static_cast<float>(ma0 * s00 + ma1 * s01)) * dl0;
                acc[k * 4 + 1] = acc[k * 4 + 1] + (sa.y * static_cast<float>(dot[1]) - sb.y * static_cast<float>(mb0 * s00 + mb1 * s01)) * dl0;
                acc[k * 4 + 2] = acc[k * 4 + 2] + (sa.x * static_cast<float>(dot[2]) - sb.x * static_cast<float>(ma0 * s10 + ma1 * s11)) * dl1;
                acc[k * 4 + 3] = acc[k * 4 + 3] + (sa.y * static_cast<float>(dot[3]) - sb.y * static_cast<float>(mb0 * s10 + mb1 * s11)) * dl1;
            }
        }
    }
}

// raw_bytes: the Q2_K block's 64 raw weight rows, staged once (0 for gate/up)
__host__ __device__ inline GroupedLayout grouped_layout(bool iq2, std::uint32_t J, std::uint32_t raw_bytes) {
    GroupedLayout l{};
    std::uint32_t at = 0;
    l.dec = at; l.dec_stride = J * kRows * 32U; at += 2U * l.dec_stride;
    l.sc = at; l.sc_stride = (iq2 ? 1U : 3U) * J * kRows * 4U; at += 2U * l.sc_stride;
    l.a = at; l.a_stride = J * kBatch * 32U; at += 2U * l.a_stride;
    l.ad = at; l.ad_stride = J * kBatch * 4U; at += 2U * l.ad_stride;
    l.ah = at; at += iq2 ? 0U : 2U * l.ad_stride;  // Q2_K half sums: two int16 per (j, entry)
    l.raw = at; at += raw_bytes;
    l.bytes = at;
    return l;
}
// decode items (row, group) per thread and class: gate/up J * 64 <= 512, down J * 64 <= 256
__host__ __device__ constexpr std::uint32_t grouped_items(bool iq2) { return ((iq2 ? 512U : 256U) + kGThreads - 1U) / kGThreads; }

// One batch of the grid-stride loop below (a block runs batches blockIdx.y, + gridDim.y, ...: the grid is bounded by
// kMaxGridY instead of the worst-case batch count, so a launch restricted to a few experts has few empty blocks).
template <bool kIq2>
__device__ __forceinline__ void grouped_batch(const GroupedParams& p, const std::uint32_t batch, const std::uint32_t tile) {
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ std::uint64_t grid[256];
    __shared__ std::uint32_t s_arow[kBatch];  // the A row of every entry (gate/up: its row, down: its slot)
    const std::uint32_t tid = threadIdx.x, lane = tid & 31U, warp = tid / 32U, gid = lane >> 2U, t = lane & 3U;
    const std::uint32_t j = p.batches.expert[batch], e0 = p.batches.begin[batch], n = p.batches.count[batch];
    const std::uint64_t expert = p.experts[j];
    const std::uint32_t rb = static_cast<std::uint32_t>(p.row_bytes);
    const std::uint32_t width = kIq2 ? p.hidden : p.inter, G = width / 32U, J = (G + 31U) / 32U;
    if (tid < n) s_arow[tid] = kIq2 ? p.rows[e0 + tid] : p.slots[e0 + tid];
    const GroupedLayout L = grouped_layout(kIq2, J, kIq2 ? 0U : kRows * rb);
    if (kIq2)
        for (std::uint32_t i = tid; i < 256U; i += kGThreads) grid[i] = g_iq2_grid[i];

    // weight rows of the block: gate/up r < 32 gate output o0 + r, else up output o0 + r - 32; down output o0 + r
    const std::uint32_t o0 = tile * (kIq2 ? 32U : kRows);
    auto row_ptr = [&](std::uint32_t r) -> const std::uint8_t* {
        if (kIq2) return (r < 32U ? p.w0 : p.w1) + expert * p.expert_stride + std::uint64_t{o0 + (r & 31U)} * rb;
        return p.w0 + expert * p.expert_stride + std::uint64_t{o0 + r} * rb;
    };
    // this thread's decode items (row, group) of a class: item = tid + kGThreads q; the raw weight bytes stay in
    // registers from the load (one class ahead) to the decode
    struct WItem {
        std::uint32_t w[10];
    };
    constexpr std::uint32_t kItems = grouped_items(kIq2);
    WItem wi[kItems];
    auto load_w = [&](std::uint32_t c, std::uint32_t jc) {
#pragma unroll
        for (std::uint32_t q = 0; q < kItems; ++q) {
            const std::uint32_t item = tid + kGThreads * q;
            if (item >= jc * kRows) continue;
            const std::uint32_t jj = item / kRows, r = item % kRows, g = c + 32U * jj;
            if (kIq2) {
                // 66-byte blocks on 4-byte aligned rows: odd blocks have the group's 8 bytes word aligned, even
                // blocks the FP16 scale (aligned words through the group, funnel-shifted)
                const std::uint8_t* blk = row_ptr(r) + (g >> 3U) * 66U;
                const std::uint32_t s = g & 7U;
                if ((g >> 3U) & 1U) {
                    const auto* w = reinterpret_cast<const std::uint32_t*>(blk + 2U + 8U * s);
                    wi[q].w[0] = __ldg(w);
                    wi[q].w[1] = __ldg(w + 1);
                    wi[q].w[2] = __ldg(reinterpret_cast<const std::uint16_t*>(blk));
                } else {
                    const auto* w = reinterpret_cast<const std::uint32_t*>(blk + 8U * s);
                    const std::uint32_t x0 = __ldg(w), x1 = __ldg(w + 1), x2 = __ldg(w + 2);
                    wi[q].w[0] = __funnelshift_r(x0, x1, 16U);
                    wi[q].w[1] = __funnelshift_r(x1, x2, 16U);
                    wi[q].w[2] = s == 0U ? (x0 & 0xffffU) : __ldg(reinterpret_cast<const std::uint16_t*>(blk));
                }
            } else {
                // from the staged raw rows (84-byte blocks: 4-byte aligned)
                const std::uint32_t s = g & 7U, half = s >> 2U;
                const unsigned char* blk = smem + L.raw + r * rb + (g >> 3U) * 84U;
                const auto* qs = reinterpret_cast<const std::uint32_t*>(blk + 16U + 32U * half);
#pragma unroll
                for (std::uint32_t w = 0; w < 8U; ++w) wi[q].w[w] = qs[w];
                wi[q].w[8] = *reinterpret_cast<const std::uint16_t*>(blk + 8U * half + 2U * (s & 3U));
                wi[q].w[9] = *reinterpret_cast<const std::uint32_t*>(blk + 80U);
            }
        }
    };
    auto store_w = [&](std::uint32_t c, std::uint32_t jc, std::uint32_t b) {
        auto* dec = reinterpret_cast<std::uint32_t*>(smem + L.dec + b * L.dec_stride);  // [J][64][8]
        auto* sc_a = reinterpret_cast<float*>(smem + L.sc + b * L.sc_stride);           // [J][64]: IQ2 scale / Q2_K d
        auto* sc_b = sc_a + J * kRows;                                                  // Q2_K dmin
        auto* sc_m = reinterpret_cast<std::uint32_t*>(sc_b + J * kRows);                // Q2_K mins (m0 | m1 << 8)
#pragma unroll
        for (std::uint32_t q = 0; q < kItems; ++q) {
            const std::uint32_t item = tid + kGThreads * q;
            if (item >= jc * kRows) continue;
            const std::uint32_t jj = item / kRows, r = item % kRows, g = c + 32U * jj;
            auto* out = reinterpret_cast<uint4*>(dec + (jj * kRows + r) * 8U);
            std::uint32_t o[8];
            if (kIq2) {
                const std::uint32_t aux0 = wi[q].w[0], aux1 = wi[q].w[1];
#pragma unroll
                for (std::uint32_t l = 0; l < 4U; ++l) {
                    const std::uint64_t gr = grid[(aux0 >> (8U * l)) & 0xffU];
                    const std::uint32_t sb = sign_byte(aux1 >> (7U * l));
                    o[2U * l] = apply_signs(static_cast<std::uint32_t>(gr), sign_mask4(sb & 15U));
                    o[2U * l + 1U] = apply_signs(static_cast<std::uint32_t>(gr >> 32U), sign_mask4(sb >> 4U));
                }
                sc_a[jj * kRows + r] = (half_bits(wi[q].w[2]) * (0.5F + static_cast<float>(aux1 >> 28U))) * 0.25F;
            } else {
                const std::uint32_t shift = 2U * (g & 3U), s0 = wi[q].w[8] & 0xffU, s1 = wi[q].w[8] >> 8U;
#pragma unroll
                for (std::uint32_t w = 0; w < 8U; ++w) o[w] = ((wi[q].w[w] >> shift) & 0x03030303U) * (w < 4U ? (s0 & 15U) : (s1 & 15U));
                sc_a[jj * kRows + r] = half_bits(wi[q].w[9] & 0xffffU);
                sc_b[jj * kRows + r] = half_bits(wi[q].w[9] >> 16U);
                sc_m[jj * kRows + r] = (s0 >> 4U) | ((s1 >> 4U) << 8U);
            }
            out[half_slot(r, 0U)] = make_uint4(o[0], o[1], o[2], o[3]);
            out[half_slot(r, 1U)] = make_uint4(o[4], o[5], o[6], o[7]);
        }
    };
    // A operands of a class into buffer b: per (j, entry) 32 code bytes, the delta, and (Q2_K) the two half sums
    auto copy_a = [&](std::uint32_t c, std::uint32_t jc, std::uint32_t b) {
        unsigned char* abuf = smem + L.a + b * L.a_stride;
        unsigned char* dbuf = smem + L.ad + b * L.ad_stride;
        unsigned char* hbuf = smem + L.ah + b * L.ad_stride;
        for (std::uint32_t item = tid; item < jc * kBatch * 2U; item += kGThreads) {
            const std::uint32_t half16 = item & 1U, m = (item >> 1U) % kBatch, jj = (item >> 1U) / kBatch;
            if (m >= n) continue;
            const std::uint64_t arow = s_arow[m];
            const std::uint32_t g = c + 32U * jj;
            const auto d = static_cast<unsigned>(__cvta_generic_to_shared(abuf + (jj * kBatch + m) * 32U + 16U * half_slot(m, half16)));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(p.a_codes + arow * width + g * 32U + 16U * half16));
            if (half16 == 0U) {
                const auto dd = static_cast<unsigned>(__cvta_generic_to_shared(dbuf + (jj * kBatch + m) * 4U));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dd), "l"(p.a_deltas + arow * G + g));
            } else if (!kIq2) {
                const auto dh = static_cast<unsigned>(__cvta_generic_to_shared(hbuf + (jj * kBatch + m) * 4U));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dh), "l"(p.batches.half_sums + arow * (2U * G) + 2U * g));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    auto class_of = [](std::uint32_t i) { return __brev(i) >> 27U; };
    auto class_groups = [&](std::uint32_t i) {
        const std::uint32_t c = class_of(i);
        return c < G ? (G - c + 31U) / 32U : 0U;
    };
    auto next_class = [&](std::uint32_t i) {  // the next class index after i with groups (32: none)
        ++i;
        while (i < 32U && !class_groups(i)) ++i;
        return i;
    };

    // warp layout by batch size: ms m-tiles of 16 entries, each over the 8 n-tiles of 8 weight rows, ntc per warp
    const std::uint32_t ms = n <= 16U ? 1U : n <= 32U ? 2U : 4U;
    const std::uint32_t ntc = 8U * ms >= kWarps ? 8U * ms / kWarps : 1U, wpm = 8U / ntc;
    const std::uint32_t msub = warp / wpm, nt0 = (warp % wpm) * ntc;
    const bool active = msub < ms && 16U * msub < n;
    const std::uint32_t m0 = 16U * msub + gid, m1 = m0 + 8U;
    const bool v0 = active && m0 < n, v1 = active && m1 < n;

    // prologue: class index 0 (class 0, which always has groups) staged in buffer 0; the next class's weights loaded
    __syncthreads();
    if (!kIq2) {  // the block's raw Q2_K rows (contiguous, 16-byte aligned) staged once, with class 0's A operands
        const auto* src = reinterpret_cast<const uint4*>(row_ptr(0U));
        const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem + L.raw));
        for (std::uint32_t k = tid; k < kRows * rb / 16U; k += kGThreads)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst + 16U * k), "l"(src + k));
    }
    copy_a(0U, class_groups(0U), 0U);
    if (!kIq2) {
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }
    load_w(0U, class_groups(0U));
    store_w(0U, class_groups(0U), 0U);
    std::uint32_t inext = next_class(0U);
    if (inext < 32U) load_w(class_of(inext), class_groups(inext));
    std::uint32_t buf = 0;
    // the class sums of every element folded in the butterfly's tree: class index i = 4 i4 + s, the classes' partial
    // sums p0 (i % 2 == 0), p1 (i % 4 == 1), and the levels above in p2..p4
    float p0[kMaxNt * 4], p1[kMaxNt * 4], p2[kMaxNt * 4], p3[kMaxNt * 4], p4[kMaxNt * 4], result[kMaxNt * 4];
    for (std::uint32_t i4 = 0; i4 < 32U; i4 += 4U) {
#pragma unroll
        for (std::uint32_t s = 0; s < 4U; ++s) {
            const std::uint32_t i = i4 + s, jc = class_groups(i);
            float acc[kMaxNt * 4];
#pragma unroll
            for (std::uint32_t e = 0; e < kMaxNt * 4; ++e) acc[e] = 0.0F;
            if (jc) {
                asm volatile("cp.async.wait_group 0;\n" ::);
                __syncthreads();  // class i staged in buf; the products of the previous class (buf ^ 1) are done
                if (inext < 32U) {
                    copy_a(class_of(inext), class_groups(inext), buf ^ 1U);
                    store_w(class_of(inext), class_groups(inext), buf ^ 1U);
                    const std::uint32_t i2 = next_class(inext);
                    if (i2 < 32U) load_w(class_of(i2), class_groups(i2));
                }
                if (active) {
                    if (ntc == 2U) class_products<kIq2, kMaxNt>(smem, L, J, buf, jc, msub, nt0, lane, acc);
                    else class_products<kIq2, 1U>(smem, L, J, buf, jc, msub, nt0, lane, acc);
                }
                buf ^= 1U;
                if (inext < 32U) inext = next_class(inext);
            }
#pragma unroll
            for (std::uint32_t e = 0; e < kMaxNt * 4; ++e) {
                if (s == 0U || s == 2U) {
                    p0[e] = acc[e];
                } else if (s == 1U) {
                    p1[e] = p0[e] + acc[e];
                } else {
                    float v = p1[e] + (p0[e] + acc[e]);
                    if (!(i4 & 4U)) { p2[e] = v; continue; }
                    v = p2[e] + v;
                    if (!(i4 & 8U)) { p3[e] = v; continue; }
                    v = p3[e] + v;
                    if (!(i4 & 16U)) { p4[e] = v; continue; }
                    result[e] = p4[e] + v;
                }
            }
        }
    }

    if (!kIq2) {
        if (active) {
            const std::uint64_t slot0 = v0 ? p.slots[e0 + m0] : 0U, slot1 = v1 ? p.slots[e0 + m1] : 0U;
#pragma unroll
            for (std::uint32_t k = 0; k < kMaxNt; ++k) {
                if (k >= ntc) break;
                const std::uint32_t o = o0 + 8U * (nt0 + k) + 2U * t;
                if (v0) *reinterpret_cast<float2*>(p.raw + slot0 * p.hidden + o) = make_float2(result[k * 4 + 0], result[k * 4 + 1]);
                if (v1) *reinterpret_cast<float2*>(p.raw + slot1 * p.hidden + o) = make_float2(result[k * 4 + 2], result[k * 4 + 3]);
            }
        }
        return;
    }
    // gate/up epilogue: [entry][32 outputs][gate, up] through shared memory (over the staged buffers)
    __syncthreads();
    auto* gu = reinterpret_cast<float*>(smem);
    if (active) {
#pragma unroll
        for (std::uint32_t k = 0; k < kMaxNt; ++k) {
            if (k >= ntc) break;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const std::uint32_t m = q >= 2 ? m1 : m0, r = 8U * (nt0 + k) + 2U * t + (q & 1);  // r < 32: gate, else up
                if (m < n) gu[(m * 32U + (r & 31U)) * 2U + (r >> 5U)] = result[k * 4 + q];
            }
        }
    }
    __syncthreads();
    for (std::uint32_t m = warp; m < n; m += kWarps) {
        const float gv = gu[(m * 32U + lane) * 2U], uv = gu[(m * 32U + lane) * 2U + 1U];
        const std::uint64_t slot = p.slots[e0 + m], o = tile * 32U + lane;
        if (p.gate_out) { p.gate_out[slot * p.inter + o] = gv; p.up_out[slot * p.inter + o] = uv; }
        const A8Group a = a8_group(swiglu_clamped(gv, uv, p.limit));
        p.act_codes[slot * p.inter + o] = a.code;
        if (lane == 0U) {
            p.act_deltas[slot * (p.inter / 32U) + tile] = a.delta;
            if (p.act_sums) p.act_sums[slot * (p.inter / 32U) + tile] = a.sum;
        }
    }
}
template <bool kIq2>
__global__ __launch_bounds__(kGThreads, 1) void grouped_kernel(GroupedParams p) {
    __shared__ std::uint32_t s_item;
    const std::uint32_t total = *p.batches.total;
    if (!p.dynamic) {
        if (blockIdx.y < total) grouped_batch<kIq2>(p, blockIdx.y, blockIdx.x);
        return;
    }
    for (;;) {
        // items in the order of a full grid (tiles of a batch together: its activations stay in L2)
        if (threadIdx.x == 0U) s_item = atomicAdd(p.batches.next, 1U);
        __syncthreads();
        const std::uint32_t item = s_item, batch = item / gridDim.x;
        if (batch >= total) return;
        grouped_batch<kIq2>(p, batch, item % gridDim.x);
        __syncthreads();  // the batch's epilogue is done with the shared memory (and s_item was read)
    }
}

std::size_t grouped_smem(bool iq2, const MoeShape& s) {
    const std::uint32_t G = (iq2 ? s.hidden : s.inter) / 32U, J = (G + 31U) / 32U;
    return std::max<std::size_t>(grouped_layout(iq2, J, iq2 ? 0U : static_cast<std::uint32_t>(kRows * s.down_row_bytes())).bytes, iq2 ? kBatch * 32U * 2U * 4U : 0U);
}

void check_grouped(const MoeShape& s) {
    if (s.hidden % 256U || s.inter % 256U || s.hidden % kRows || s.top_k == 0U || s.experts == 0U || s.experts > 1024U ||
        s.inter / 32U > kMaxTiles || s.hidden / kRows > kMaxTiles ||
        (s.hidden / 32U + 31U) / 32U * kRows > grouped_items(true) * kGThreads ||
        (s.inter / 32U + 31U) / 32U * kRows > grouped_items(false) * kGThreads ||
        grouped_smem(true, s) > 99U * 1024U - 2048U || grouped_smem(false, s) > 99U * 1024U)
        throw std::invalid_argument("ds41 cuda moe grouped: unsupported shape");
}

}  // namespace

std::size_t routed_grouped_workspace_bytes(std::uint32_t rows, const MoeShape& shape) noexcept {
    const std::size_t m = max_batches(rows, shape);
    return 3U * round256(4U * m) + round256(4U * (1U + kMaxTiles)) + round256(2U * std::size_t{rows} * shape.top_k * (shape.inter / 16U));
}

void enqueue_routed_gate_up_grouped(const RoutedGateUpView& v, std::uint32_t rows, void* scratch, cudaStream_t stream) {
    check_grouped(v.shape);
    if (!rows || !v.max_union) return;
    upload_grouped_grid();
    const Batches b = carve_batches(scratch, rows, v.shape);
    batches_kernel<<<1, 1024, 0, stream>>>(v.routes.begins, v.routes.counts, b);
    const GroupedParams p{v.weights.gate, v.weights.up, v.shape.gate_expert_bytes(), v.shape.gate_row_bytes(), v.shape.hidden,
                          v.shape.inter, v.routes.experts, v.routes.rows, v.routes.slots, b, v.x.codes, v.x.deltas, v.shape.limit,
                          v.act.codes, v.act.deltas, v.act.sums, v.gate_out, v.up_out, nullptr,
                          dynamic_grid(v.max_union, v.shape)};
    const std::size_t smem = grouped_smem(true, v.shape);
    STRIX_CUDA_CHECK(cudaFuncSetAttribute(grouped_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    grouped_kernel<true><<<dim3(v.shape.inter / 32U, grid_y(rows, v.shape, p.dynamic)), kGThreads, smem, stream>>>(p);
    STRIX_CUDA_CHECK(cudaGetLastError());
}
void enqueue_routed_down_grouped(const RoutedDownView& v, std::uint32_t rows, void* scratch, cudaStream_t stream) {
    check_grouped(v.shape);
    if (!rows || !v.max_union) return;
    const Batches b = carve_batches(scratch, rows, v.shape);
    batches_kernel<<<1, 1024, 0, stream>>>(v.routes.begins, v.routes.counts, b);
    const std::uint64_t halves = std::uint64_t{rows} * v.shape.top_k * (v.shape.inter / 16U);
    half_sums_kernel<<<static_cast<unsigned>((halves + 255U) / 256U), 256, 0, stream>>>(v.act.codes, halves, b.half_sums);
    const GroupedParams p{v.weights.down, nullptr, v.shape.down_expert_bytes(), v.shape.down_row_bytes(), v.shape.hidden, v.shape.inter,
                          v.routes.experts, v.routes.rows, v.routes.slots, b, v.act.codes, v.act.deltas, 0.0F,
                          nullptr, nullptr, nullptr, nullptr, nullptr, v.raw, dynamic_grid(v.max_union, v.shape)};
    if (reinterpret_cast<std::uintptr_t>(v.weights.down) % 16U)
        throw std::invalid_argument("ds41 cuda moe grouped: down weights not 16-byte aligned");
    const std::size_t smem = grouped_smem(false, v.shape);
    STRIX_CUDA_CHECK(cudaFuncSetAttribute(grouped_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    grouped_kernel<false><<<dim3(v.shape.hidden / kRows, grid_y(rows, v.shape, p.dynamic)), kGThreads, smem, stream>>>(p);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

MoeLayerScratch moe_layer_workspace_views(void* workspace, std::uint32_t capacity, const MoeShape& shape) {
    const LayerSpace w = carve(workspace, capacity, shape);
    return {w.routes, w.act, w.scratch, w.grouped};
}

std::size_t moe_layer_workspace_bytes(std::uint32_t capacity, const MoeShape& shape) noexcept { return carve(nullptr, capacity, shape).bytes; }

void enqueue_moe_layer(const MoeLayerView& l, cudaStream_t stream) {
    if (l.rows == 0U) return;
    if (l.rows > l.capacity || !l.workspace) throw std::invalid_argument("ds41 cuda moe layer: rows above the workspace capacity");
    const MoeShape& s = l.shape;
    const LayerSpace w = carve(l.workspace, l.capacity, s);
    MoeRoutesView b;
    b.ids = l.ids;
    b.rows = l.rows;
    b.top_k = s.top_k;
    b.experts = s.experts;
    b.split = l.split;
    b.scratch = w.scratch;
    b.out = w.routes;
    enqueue_moe_routes(b, stream);
    const std::uint32_t max_union = std::min(l.rows * s.top_k, s.experts);
    const bool grouped = l.grouped_min_rows != 0U && l.rows >= l.grouped_min_rows;
    if (grouped && l.raw1) throw std::invalid_argument("ds41 cuda moe layer: expert TP runs the decode GEMVs only");
    RoutedGateUpView g;
    g.shape = s;
    g.weights = l.weights;
    g.routes = w.routes;
    g.max_union = max_union;
    g.max_rows = std::min(l.rows, 8U);
    g.x = l.x;
    g.act = w.act;
    RoutedDownView d;
    d.shape = s;
    d.weights = l.weights;
    d.routes = w.routes;
    d.max_union = max_union;
    d.max_rows = g.max_rows;
    d.act = w.act;
    d.raw = l.raw;
    d.raw1 = l.raw1;
    if (grouped) {
        enqueue_routed_gate_up_grouped(g, l.rows, w.grouped, stream);
        enqueue_routed_down_grouped(d, l.rows, w.grouped, stream);
    } else {
        enqueue_routed_gate_up(g, stream);
        enqueue_routed_down(d, stream);
    }
    MoeCombineView c;
    c.raw = l.raw;
    c.shared = l.shared;
    c.weights = l.route_weights;
    c.order = w.routes.order;
    c.low_counts = w.routes.low_counts;
    c.rows = l.rows;
    c.top_k = s.top_k;
    c.hidden = s.hidden;
    c.out = l.y;
    c.raw1 = l.raw1;
    enqueue_moe_combine(c, stream);
}


namespace {
__global__ void map_ids_kernel(const std::uint32_t* ids, const std::uint32_t* perm, std::uint32_t n, std::uint32_t* out) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = perm[ids[i]];
}
}  // namespace

void enqueue_moe_layer_ranges(const MoeRangesView& r, cudaStream_t stream) {
    const MoeLayerView& l = r.layer;
    if (l.rows == 0U) return;
    if (l.rows > l.capacity || !l.workspace) throw std::invalid_argument("ds41 cuda moe layer: rows above the workspace capacity");
    if (r.perm && !r.mapped_ids) throw std::invalid_argument("ds41 cuda moe ranges: a permutation needs the mapped ids scratch");
    const MoeShape& s = l.shape;
    const LayerSpace w = carve(l.workspace, l.capacity, s);
    const std::uint32_t n = l.rows * s.top_k;
    const std::uint32_t* ids = l.ids;
    if (r.perm) {
        map_ids_kernel<<<(n + 255U) / 256U, 256, 0, stream>>>(l.ids, r.perm, n, r.mapped_ids);
        STRIX_CUDA_CHECK(cudaGetLastError());
        ids = r.mapped_ids;
    }
    const bool grouped = l.grouped_min_rows != 0U && l.rows >= l.grouped_min_rows;
    MoeRoutesView b;
    b.rows = l.rows;
    b.top_k = s.top_k;
    b.experts = s.experts;
    b.split = l.split;
    b.scratch = w.scratch;
    b.out = w.routes;
    bool order_built = false;
    for (std::uint32_t k = 0; k < r.range_count; ++k) {
        const MoeRange& g = r.ranges[k];
        if (g.first >= g.end) continue;
        if (g.ready) {
            if (g.wait_begin) STRIX_CUDA_CHECK(cudaEventRecord(g.wait_begin, stream));
            STRIX_CUDA_CHECK(cudaStreamWaitEvent(stream, g.ready, 0));
            if (g.wait_end) STRIX_CUDA_CHECK(cudaEventRecord(g.wait_end, stream));
        }
        b.ids = ids;
        b.first = g.first;
        b.end = g.end;
        b.base = g.base;
        enqueue_moe_routes(b, stream);
        order_built = !r.perm;  // the order / low counts of mapped ids are not the combine's
        const std::uint32_t max_union = std::min(n, g.end - g.first);
        RoutedGateUpView gu;
        gu.shape = s;
        gu.weights = g.weights;
        gu.routes = w.routes;
        gu.max_union = max_union;
        gu.max_rows = std::min(l.rows, 8U);
        gu.x = l.x;
        gu.act = w.act;
        RoutedDownView d;
        d.shape = s;
        d.weights = g.weights;
        d.routes = w.routes;
        d.max_union = max_union;
        d.max_rows = gu.max_rows;
        d.act = w.act;
        d.raw = l.raw;
        if (grouped) {
            enqueue_routed_gate_up_grouped(gu, l.rows, w.grouped, stream);
            enqueue_routed_down_grouped(d, l.rows, w.grouped, stream);
        } else {
            enqueue_routed_gate_up(gu, stream);
            enqueue_routed_down(d, stream);
        }
    }
    if (!order_built) {
        // routes of the original ids for the combine's order (one expert's entries: the cheapest restriction)
        b.ids = l.ids;
        b.first = 0;
        b.end = 1;
        b.base = 0;
        enqueue_moe_routes(b, stream);
    }
    MoeCombineView c;
    c.raw = l.raw;
    c.shared = l.shared;
    c.weights = l.route_weights;
    c.order = w.routes.order;
    c.low_counts = w.routes.low_counts;
    c.rows = l.rows;
    c.top_k = s.top_k;
    c.hidden = s.hidden;
    c.out = l.y;
    enqueue_moe_combine(c, stream);
}

}  // namespace strix::ds41::cuda
