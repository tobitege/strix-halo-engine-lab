// deepseek41 routed experts on CUDA, prefill FAST arm (cuda_ds41_moe_fast.hpp): FP16 tensor-core grouped GEMMs on
// expert weight tiles decoded in shared memory, FP32 accumulation, FP16 activations. Not the reference's bits.

#include "ds41_moe_device.cuh"
#include "strix/cuda/ds41_moe_fast.hpp"
#include "strix/ds41_formats.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <array>

namespace strix::ds41::cuda {

using namespace moe_detail;

namespace {

constexpr std::size_t align256(std::size_t v) { return (v + 255U) & ~std::size_t{255U}; }

// A block computes one batch of up to kMaxE entries of one expert against kTileRows weight rows: for gate/up the
// gate and up rows of 64 outputs (tile rows 16 b .. 16 b + 7 gate outputs 8 b .., 16 b + 8 .. up outputs 8 b ..: an
// m16 tile pairs 8 gate rows with their up rows, so a thread holds both products of its outputs), for down 128 output
// rows. Weights are the MMA's A operand (m16), the entries its B operand (n8), so batches of 32 / 64 / 128 entries
// keep the 8 warps busy (4 x 2 warps, 32 weight rows x E/2 entries each). Every stage holds 64 bytes per tile row
// (INT8: two 32-weight groups, one m16n8k32 each; FP16: one group, two m16n8k16), two buffers: the entries' operands
// by cp.async one stage ahead, the weights decoded from registers loaded two stages ahead.
constexpr int kThreads = 256;
constexpr std::uint32_t kTileRows = 128;
constexpr std::uint32_t kMaxE = 128;
constexpr std::uint32_t kRowBytes = 64;
constexpr std::uint32_t kTileBytes = kTileRows * kRowBytes;   // weights, then entries
constexpr std::uint32_t kScaleBytes = 2U * kTileRows * 4U;    // INT8: [2 groups][128] weight scales, then entry deltas
constexpr std::uint32_t kBufBytes = 2U * kTileBytes + 2U * kScaleBytes;
constexpr std::uint32_t kSmem = 2U * kBufBytes;
// INT8 down: the Q2_K min terms of the current 256-weight block, -dmin * m per (row, 16-weight sub-block) and the
// entries' delta * (sum of the 16 codes), FP16 [128][16] each, one m16n8k16 per block
constexpr std::uint32_t kMinBytes = kTileRows * 32U;
constexpr std::uint32_t kSmemDownI8 = kSmem + 2U * kMinBytes;

enum Kind : int { kGateI8 = 0, kGateF16 = 1, kDownF16 = 2, kDownI8 = 3 };

__device__ std::uint64_t g_fast_grid[256];
std::uint64_t g_fast_grid_ready = 0;  // devices holding the grid
__device__ std::uint64_t g_sign_masks[128U];
std::uint64_t g_sign_masks_ready = 0;
void upload_sign_masks() {
    int device = 0;
    STRIX_CUDA_CHECK(cudaGetDevice(&device));
    if (device < 64 && ((g_sign_masks_ready >> device) & 1U)) return;
    static const auto table = [] {
        std::array<std::uint64_t, 128U> out{};
        for (std::uint32_t signs = 0; signs < 128U; ++signs) {
            const auto mask = ::strix::ds41::iq2_xxs_sign_byte(signs);
            std::uint64_t word = 0;
            for (std::uint32_t b = 0; b < 8U; ++b)
                word |= std::uint64_t{(mask >> b) & 1U ? 255U : 0U} << (8U * b);
            out[signs] = word;
        }
        return out;
    }();
    STRIX_CUDA_CHECK(cudaMemcpyToSymbol(g_sign_masks, table.data(), table.size() * sizeof(std::uint64_t)));
    if (device < 64) g_sign_masks_ready |= std::uint64_t{1} << device;
}
void upload_fast_grid() {
    int device = 0;
    STRIX_CUDA_CHECK(cudaGetDevice(&device));
    if (device < 64 && ((g_fast_grid_ready >> device) & 1U)) return;
    STRIX_CUDA_CHECK(cudaMemcpyToSymbol(g_fast_grid, iq2_xxs_grid.data(), sizeof(std::uint64_t) * 256U));
    if (device < 64) g_fast_grid_ready |= std::uint64_t{1} << device;
}

struct FastBatches {
    std::uint32_t* expert;
    std::uint32_t* begin;
    std::uint32_t* count;
    std::uint32_t* total;
};
std::uint32_t max_batches(std::uint32_t rows, const MoeShape& s) {
    const std::uint32_t slots = rows * s.top_k;
    return (slots + kMaxE - 1U) / kMaxE + std::min(slots, s.experts);
}

struct FastSpace {
    MoeRoutes routes;
    std::uint32_t* scratch = nullptr;
    FastBatches batches{};
    __half* x16 = nullptr;
    __half* act16 = nullptr;
    std::int8_t* act_codes = nullptr;  // INT8 down: A8 of the activation [slot][inter], deltas [slot][inter / 32],
    float* act_deltas = nullptr;       // delta * (sum of each 16 codes) [slot][inter / 16]
    __half* act_bmin = nullptr;
    std::size_t bytes = 0;
};
FastSpace carve(void* base, std::uint32_t capacity, const MoeShape& s) {
    FastSpace w;
    auto* p = static_cast<unsigned char*>(base);
    std::size_t at = 0;
    const std::size_t slots = std::size_t{capacity} * s.top_k, mb = max_batches(capacity, s);
    auto take = [&](std::size_t bytes) {
        void* out = p ? p + at : nullptr;
        at += align256(bytes);
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
    w.batches.expert = static_cast<std::uint32_t*>(take(4U * mb));
    w.batches.begin = static_cast<std::uint32_t*>(take(4U * mb));
    w.batches.count = static_cast<std::uint32_t*>(take(4U * mb));
    w.batches.total = static_cast<std::uint32_t*>(take(4U));
    w.x16 = static_cast<__half*>(take(2U * std::size_t{capacity} * s.hidden));
    w.act16 = static_cast<__half*>(take(2U * slots * s.inter));
    w.act_codes = static_cast<std::int8_t*>(take(slots * s.inter));
    w.act_deltas = static_cast<float*>(take(4U * slots * (s.inter / 32U)));
    w.act_bmin = static_cast<__half*>(take(2U * slots * (s.inter / 16U)));
    w.bytes = at;
    return w;
}

// Batches of at most kMaxE consecutive entries of each union expert, in union order (one block of 1024 threads).
template <std::uint32_t entries = kMaxE>
__global__ __launch_bounds__(1024) void fast_batches_kernel(const std::uint32_t* begins, const std::uint32_t* counts, FastBatches b) {
    __shared__ std::uint32_t warp_sums[33];
    const std::uint32_t unions = counts[0], lane = threadIdx.x & 31U, warp = threadIdx.x / 32U;
    std::uint32_t base = 0;
    for (std::uint32_t j0 = 0; j0 < unions; j0 += 1024U) {
        const std::uint32_t j = j0 + threadIdx.x;
        const std::uint32_t n = j < unions ? (begins[j + 1U] - begins[j] + entries - 1U) / entries : 0U;
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
            const std::uint32_t e0 = begins[j] + k * entries;
            b.expert[first + k] = j;
            b.begin[first + k] = e0;
            b.count[first + k] = min(entries, begins[j + 1U] - e0);
        }
        base += warp_sums[32];
        __syncthreads();
    }
    if (threadIdx.x == 0U) *b.total = base;
}

// y[row] = shared[row] + sum_k w[row][k] * raw[row * top_k + k] (FP16 slot vectors, FP32 sums in slot order), 8
// outputs per thread.
__global__ __launch_bounds__(256) void fast_combine_kernel(const __half* raw, const float* shared, const float* w, std::uint32_t rows,
                                                           std::uint32_t top_k, std::uint32_t hidden, float* y) {
    const std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x, per_row = hidden / 8U;
    if (i >= std::size_t{rows} * per_row) return;
    const std::size_t row = i / per_row, o = (i % per_row) * 8U;
    float acc[8];
    const float4 s0 = *reinterpret_cast<const float4*>(shared + row * hidden + o), s1 = *reinterpret_cast<const float4*>(shared + row * hidden + o + 4U);
    acc[0] = s0.x; acc[1] = s0.y; acc[2] = s0.z; acc[3] = s0.w; acc[4] = s1.x; acc[5] = s1.y; acc[6] = s1.z; acc[7] = s1.w;
    for (std::uint32_t k = 0; k < top_k; ++k) {
        const std::size_t slot = row * top_k + k;
        const float wk = w[slot];
        const uint4 v = *reinterpret_cast<const uint4*>(raw + slot * hidden + o);
        const std::uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (std::uint32_t q = 0; q < 4U; ++q) {
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&words[q]));
            acc[2U * q] = __fmaf_rn(wk, f.x, acc[2U * q]);
            acc[2U * q + 1U] = __fmaf_rn(wk, f.y, acc[2U * q + 1U]);
        }
    }
    *reinterpret_cast<float4*>(y + row * hidden + o) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    *reinterpret_cast<float4*>(y + row * hidden + o + 4U) = make_float4(acc[4], acc[5], acc[6], acc[7]);
}

__global__ void to_half_kernel(const float* x, std::size_t n2, __half2* out) {
    const std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
    if (i < n2) {
        const float2 v = reinterpret_cast<const float2*>(x)[i];
        out[i] = __floats2half2_rn(v.x, v.y);
    }
}

struct FastParams {
    const std::uint8_t* w0;  // gate (IQ2_XXS) or down (Q2_K)
    const std::uint8_t* w1;  // up (IQ2_XXS)
    std::uint64_t expert_stride, row_bytes;
    std::uint32_t hidden, inter;
    const std::uint32_t *experts, *rows, *slots;
    FastBatches batches;
    const void* a;           // kGateI8: x codes [row][hidden]; kGateF16: x16 [row][hidden]; kDownF16: act16 [slot][inter];
                             // kDownI8: act codes [slot][inter]
    const float* a_deltas;   // kGateI8: x deltas [row][hidden / 32]; kDownI8: act deltas [slot][inter / 32]
    const __half* a_bmin;    // kDownI8: act delta * 16-code sums [slot][inter / 16]
    float limit;
    __half* act;             // gate/up, FP16 down: act16 [slot][inter]
    std::int8_t* act_codes;  // gate/up, INT8 down: A8 of the activation (with act_deltas, act_bmin)
    float* act_deltas;
    __half* act_bmin;
    __half* raw;             // down: [slot][hidden] in FP16 (saturated)
};

// 16-byte chunk c (0..3) of row r of a 64-byte-row tile, XOR-swizzled so ldmatrix's 8 rows hit distinct banks.
__device__ __forceinline__ std::uint32_t swz(std::uint32_t r, std::uint32_t c) { return r * kRowBytes + 16U * (c ^ ((r >> 1U) & 3U)); }

__device__ __forceinline__ void ldsm_x4(unsigned addr, std::uint32_t& r0, std::uint32_t& r1, std::uint32_t& r2, std::uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_f16(const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1, float (&d)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_s8(const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1, int (&d)[4]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
__device__ __forceinline__ void cp_async16(unsigned dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async8(unsigned dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ __half sat_half(float v) { return __float2half_rn(fminf(fmaxf(v, -65504.0F), 65504.0F)); }
__device__ __forceinline__ float fp16_bits(std::uint32_t bits) { return __half2float(__ushort_as_half(static_cast<unsigned short>(bits))); }
__device__ __forceinline__ std::uint32_t h2u(__half2 h) { return *reinterpret_cast<std::uint32_t*>(&h); }
__device__ __forceinline__ __half2 u2h(std::uint32_t u) { return *reinterpret_cast<__half2*>(&u); }
// bytes 2 i, 2 i + 1 of w (values < 256) as an exact FP16 pair
__device__ __forceinline__ __half2 bytes_h2(std::uint32_t w, std::uint32_t sel) {
    return __hsub2(u2h(__byte_perm(w, 0x6464U, sel)), __half2half2(__ushort_as_half(0x6400U)));
}

template <int kKind, std::uint32_t E, bool kSignedLut = false, std::uint32_t kThr = 256U>
__device__ __forceinline__ void fast_body(const FastParams& p, unsigned char* smem, const std::uint64_t* grid, const std::uint32_t* s_arow,
                                          const std::uint32_t* s_slot, std::uint32_t n, const std::uint8_t* wrow,
                                          const std::uint64_t* sign_masks = nullptr) {
    constexpr std::uint32_t kThreads = kThr;
    constexpr std::uint32_t kColumns = kThr / 128U, kEntryStep = 16U * kColumns;
    constexpr std::uint32_t kActivationBytes = (kThr == 512U ? 256U : kMaxE) * kRowBytes;
    constexpr std::uint32_t kEntryScaleBytes = (kThr == 512U ? 256U : kMaxE) * 8U;
    constexpr std::uint32_t kBufBytes = kTileBytes + kActivationBytes + kScaleBytes + kEntryScaleBytes;
    constexpr std::uint32_t kStages = kThr == 512U ? 3U : 2U;
    constexpr std::uint32_t kAhead = kStages - 1U;
    static_assert(kAhead < 4U);  // a Q2 min table is shared across its four stages
    constexpr std::uint32_t kSmem = kStages * kBufBytes;
    constexpr bool kI8 = kKind == kGateI8 || kKind == kDownI8, kGate = kKind == kGateI8 || kKind == kGateF16;
    constexpr std::uint32_t NT = E / (8U * kColumns);  // n8 tiles per warp
    const std::uint32_t tid = threadIdx.x, lane = tid & 31U, warp = tid / 32U, gid = lane >> 2U, t = lane & 3U;
    const std::uint32_t wm = warp & 3U, wn = warp >> 2U;
    const std::uint32_t width = kGate ? p.hidden : p.inter;
    const std::uint32_t S = kI8 ? width / 64U : width / 32U;  // stages
    // this thread's weight item per stage: tile row r; INT8 group 2 st + h (32 weights), FP16 weights 16 h .. + 15
    const std::uint32_t h = tid & 1U, r = tid >> 1U;
    const unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(smem));

    std::uint32_t rawc[8] = {}, rawn[8] = {};
    auto load_w = [&](std::uint32_t st, std::uint32_t (&raw)[8]) {
        if constexpr (kThr > 256U) { if (tid >= 256U) return; }
        if (kKind == kDownI8) {
            // groups g0 = 2 st, g0 + 1 share the 32 code bytes (shifts 2 (g0 & 3), + 2); this thread's 16 bytes 16 h ..
            const std::uint32_t g0 = 2U * st, s0 = g0 & 7U;
            const std::uint8_t* blk = wrow + (g0 >> 3U) * 84U;
            const auto* qs = reinterpret_cast<const std::uint32_t*>(blk + 16U + 32U * (s0 >> 2U) + 16U * h);
            raw[0] = __ldg(qs); raw[1] = __ldg(qs + 1); raw[2] = __ldg(qs + 2); raw[3] = __ldg(qs + 3);
            raw[4] = __ldg(blk + 2U * s0 + h) | (static_cast<std::uint32_t>(__ldg(blk + 2U * s0 + 2U + h)) << 8U);
            raw[5] = __ldg(reinterpret_cast<const std::uint32_t*>(blk + 80U));
            if (s0 == 0U) {  // a block's first stage: its sub-block mins 8 h .. 8 h + 7
                raw[6] = __ldg(reinterpret_cast<const std::uint32_t*>(blk + 8U * h));
                raw[7] = __ldg(reinterpret_cast<const std::uint32_t*>(blk + 8U * h + 4U));
            }
        } else if (kI8) {
            const std::uint32_t g = 2U * st + h;
            const std::uint8_t* blk = wrow + (g >> 3U) * 66U;
            const std::uint8_t* q = blk + 2U + 8U * (g & 7U);
            if ((g >> 3U) & 1U) {  // odd block: the group's 8 bytes word aligned
                raw[0] = __ldg(reinterpret_cast<const std::uint32_t*>(q));
                raw[1] = __ldg(reinterpret_cast<const std::uint32_t*>(q + 4U));
                raw[2] = 0U;
            } else {
                raw[0] = __ldg(reinterpret_cast<const std::uint32_t*>(q - 2U));
                raw[1] = __ldg(reinterpret_cast<const std::uint32_t*>(q + 2U));
                raw[2] = __ldg(reinterpret_cast<const std::uint32_t*>(q + 6U));
            }
            raw[3] = __ldg(reinterpret_cast<const std::uint16_t*>(blk));
        } else if (kGate) {
            const std::uint8_t* blk = wrow + (st >> 3U) * 66U;
            const std::uint8_t* q = blk + 2U + 8U * (st & 7U);
            raw[0] = __ldg(reinterpret_cast<const std::uint16_t*>(q + 2U * h));
            raw[1] = __ldg(reinterpret_cast<const std::uint16_t*>(q + 4U)) |
                     (static_cast<std::uint32_t>(__ldg(reinterpret_cast<const std::uint16_t*>(q + 6U))) << 16U);
            raw[2] = __ldg(reinterpret_cast<const std::uint16_t*>(blk));
        } else {
            const std::uint32_t s = st & 7U;
            const std::uint8_t* blk = wrow + (st >> 3U) * 84U;
            const auto* qs = reinterpret_cast<const std::uint32_t*>(blk + 16U + 32U * (s >> 2U) + 16U * h);
            raw[0] = __ldg(qs); raw[1] = __ldg(qs + 1); raw[2] = __ldg(qs + 2); raw[3] = __ldg(qs + 3);
            raw[4] = __ldg(blk + 2U * s + h);
            raw[5] = __ldg(reinterpret_cast<const std::uint32_t*>(blk + 80U));
        }
    };
    auto store_w = [&](std::uint32_t st, std::uint32_t buf, const std::uint32_t (&raw)[8]) {
        if constexpr (kThr > 256U) { if (tid >= 256U) return; }
        unsigned char* tile = smem + buf * kBufBytes;
        std::uint32_t o[8];
        if (kKind == kDownI8) {
            const std::uint32_t g0 = 2U * st, shift = 2U * (g0 & 3U), sc0 = raw[4] & 15U, sc1 = (raw[4] >> 8U) & 15U;
#pragma unroll
            for (std::uint32_t w = 0; w < 4U; ++w) {
                o[w] = ((raw[w] >> shift) & 0x03030303U) * sc0;
                o[4U + w] = ((raw[w] >> (shift + 2U)) & 0x03030303U) * sc1;
            }
            if (!h) {
                const float d = fp16_bits(raw[5] & 0xffffU);
                auto* sw = reinterpret_cast<float*>(tile + (kTileBytes + kActivationBytes));
                sw[r] = d;
                sw[kTileRows + r] = d;
            }
            if ((g0 & 7U) == 0U) {  // -dmin * m of sub-blocks 8 h .. 8 h + 7
                const float dmin = fp16_bits(raw[5] >> 16U);
                std::uint32_t m[4];
#pragma unroll
                for (std::uint32_t k = 0; k < 4U; ++k) {
                    const std::uint32_t wv = k < 2U ? raw[6] : raw[7], b = 16U * (k & 1U);
                    m[k] = h2u(__floats2half2_rn(-dmin * static_cast<float>((wv >> (b + 4U)) & 15U),
                                                 -dmin * static_cast<float>((wv >> (b + 12U)) & 15U)));
                }
                *reinterpret_cast<uint4*>(smem + kSmem + r * 32U + 16U * (h ^ ((r >> 2U) & 1U))) = make_uint4(m[0], m[1], m[2], m[3]);
            }
            *reinterpret_cast<uint4*>(tile + swz(r, h)) = make_uint4(o[0], o[1], o[2], o[3]);
            *reinterpret_cast<uint4*>(tile + swz(r, 2U + h)) = make_uint4(o[4], o[5], o[6], o[7]);
            return;
        }
        if (kI8) {
            const std::uint32_t g = 2U * st + h;
            std::uint32_t aux0 = raw[0], aux1 = raw[1];
            if (!((g >> 3U) & 1U)) {
                aux0 = __funnelshift_r(raw[0], raw[1], 16U);
                aux1 = __funnelshift_r(raw[1], raw[2], 16U);
            }
#pragma unroll
            for (std::uint32_t l = 0; l < 4U; ++l) {
                if constexpr (kSignedLut) {
                    const auto magnitudes = grid[(aux0 >> (8U * l)) & 255U];
                    const auto mask = sign_masks[(aux1 >> (7U * l)) & 127U];
                    const auto m0 = static_cast<std::uint32_t>(mask), m1 = static_cast<std::uint32_t>(mask >> 32U);
                    o[2U * l] = (static_cast<std::uint32_t>(magnitudes) ^ m0) + (m0 & 0x01010101U);
                    o[2U * l + 1U] = (static_cast<std::uint32_t>(magnitudes >> 32U) ^ m1) + (m1 & 0x01010101U);
                } else {
                    const std::uint64_t gr = grid[__byte_perm(aux0, 0U, 0x4440U + l)];
                    const std::uint32_t sb = sign_byte(aux1 >> (7U * l));
                    const std::uint32_t b0 = ((sb & 15U) * 0x00204081U) & 0x01010101U, b1 = ((sb >> 4U) * 0x00204081U) & 0x01010101U;
                    o[2U * l] = (static_cast<std::uint32_t>(gr) ^ (b0 * 0xffU)) + b0;
                    o[2U * l + 1U] = (static_cast<std::uint32_t>(gr >> 32U) ^ (b1 * 0xffU)) + b1;
                }
            }
            reinterpret_cast<float*>(tile + (kTileBytes + kActivationBytes))[h * kTileRows + r] =
                (fp16_bits(raw[3]) * (0.5F + static_cast<float>(aux1 >> 28U))) * 0.25F;
        } else if (kGate) {
            const std::uint32_t aux1 = raw[1];
            const float scale = fp16_bits(raw[2]) * (0.5F + static_cast<float>(aux1 >> 28U)) * 0.25F;
            const __half2 sc2 = __float2half2_rn(scale);
#pragma unroll
            for (std::uint32_t l2 = 0; l2 < 2U; ++l2) {
                const std::uint64_t gr = grid[(raw[0] >> (8U * l2)) & 0xffU];
                const std::uint32_t sb = sign_byte(aux1 >> (7U * (2U * h + l2)));
#pragma unroll
                for (std::uint32_t q = 0; q < 4U; ++q) {
                    const std::uint32_t word = q < 2U ? static_cast<std::uint32_t>(gr) : static_cast<std::uint32_t>(gr >> 32U);
                    const __half2 m = bytes_h2(word, (q & 1U) ? 0x5342U : 0x5140U);
                    const std::uint32_t sbits = sb >> (2U * q);
                    const std::uint32_t sign = ((sbits & 1U) << 15U) | ((sbits & 2U) << 30U);
                    o[4U * l2 + q] = h2u(__hmul2(u2h(h2u(m) ^ sign), sc2));
                }
            }
        } else {
            const std::uint32_t shift = 2U * (st & 3U), sc = raw[4];
            const float a = fp16_bits(raw[5] & 0xffffU) * static_cast<float>(sc & 15U);
            const float b = fp16_bits(raw[5] >> 16U) * static_cast<float>(sc >> 4U);
            const __half2 a2 = __float2half2_rn(a), nb2 = __float2half2_rn(-b);
#pragma unroll
            for (std::uint32_t w = 0; w < 4U; ++w) {
                const std::uint32_t x = (raw[w] >> shift) & 0x03030303U;
                o[2U * w] = h2u(__hfma2(bytes_h2(x, 0x5140U), a2, nb2));
                o[2U * w + 1U] = h2u(__hfma2(bytes_h2(x, 0x5342U), a2, nb2));
            }
        }
        *reinterpret_cast<uint4*>(tile + swz(r, 2U * h)) = make_uint4(o[0], o[1], o[2], o[3]);
        *reinterpret_cast<uint4*>(tile + swz(r, 2U * h + 1U)) = make_uint4(o[4], o[5], o[6], o[7]);
    };
    auto copy_a = [&](std::uint32_t st, std::uint32_t buf) {
        const unsigned tile = sbase + buf * kBufBytes + kTileBytes;
        const std::uint32_t row_bytes = kI8 ? width : 2U * width;
        const auto* a = static_cast<const unsigned char*>(p.a);
#pragma unroll
        for (std::uint32_t item = tid; item < E * 4U; item += kThreads) {
            const std::uint32_t m = item >> 2U, c = item & 3U;
            if (m < n) cp_async16(tile + swz(m, c), a + std::uint64_t{s_arow[m]} * row_bytes + st * 64U + c * 16U);
        }
        if (kI8 && tid < n) {  // the entries' deltas of the stage's two groups: [entry][2]
            const unsigned d = sbase + buf * kBufBytes + (kTileBytes + kActivationBytes) + kScaleBytes;
            cp_async8(d + 8U * tid, p.a_deltas + std::uint64_t{s_arow[tid]} * (width / 32U) + 2U * st);
        }
        if (kKind == kDownI8 && (st & 3U) == 0U && tid < 2U * n) {  // a block's first stage: the entries' 16 min sums
            const std::uint32_t m = tid >> 1U, cc = tid & 1U;
            cp_async16(sbase + kSmem + kMinBytes + m * 32U + 16U * (cc ^ ((m >> 2U) & 1U)),
                       p.a_bmin + std::uint64_t{s_arow[m]} * (width / 16U) + 16U * (st >> 2U) + 8U * cc);
        }
    };

    float acc[2][NT][4];
#pragma unroll
    for (std::uint32_t i = 0; i < 2U; ++i)
#pragma unroll
        for (std::uint32_t k = 0; k < NT; ++k)
#pragma unroll
            for (std::uint32_t e = 0; e < 4U; ++e) acc[i][k][e] = 0.0F;

    // Legacy: stage 0 ready, stage 1 weights in registers. Wide: stages 0/1 staged, stage 2 weights in registers.
    copy_a(0U, 0U);
    asm volatile("cp.async.commit_group;\n" ::);
    load_w(0U, rawc);
    store_w(0U, 0U, rawc);
    if (S > 1U) load_w(1U, rawc);
    if constexpr (kStages == 3U) {
        copy_a(1U, 1U);
        asm volatile("cp.async.commit_group;\n" ::);
        store_w(1U, 1U, rawc);
        load_w(2U, rawc);
    }
    // ldmatrix lane addresses: A (weights) rows 32 wm + 16 i + (lane & 15), chunk + (lane >> 4); B (entries) pair q:
    // entries E/2 wn + 16 q + (lane & 7) + 8 (lane >> 4), chunk + ((lane >> 3) & 1)
    const std::uint32_t a_row = 32U * wm + (lane & 15U), a_ch = lane >> 4U;
    // Entry pairs interleave over kColumns warp columns, kEntryStep entries per pair index.
    const std::uint32_t b_row = 16U * wn + (lane & 7U) + 8U * (lane >> 4U), b_ch = (lane >> 3U) & 1U;
    const std::uint32_t live_pairs = n > 16U * wn ? (n - 16U * wn + kEntryStep - 1U) / kEntryStep : 0U;

    for (std::uint32_t st = 0; st < S; ++st) {
        const std::uint32_t buf = kStages == 2U ? st & 1U : st % 3U;
        const std::uint32_t next_buf = kStages == 2U ? buf ^ 1U : (st + kAhead) % 3U;
        if constexpr (kStages == 3U) {
            if (st + kAhead < S) { asm volatile("cp.async.wait_group 1;\n" ::); }
            else { asm volatile("cp.async.wait_group 0;\n" ::); }
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        if (st + kAhead < S) copy_a(st + kAhead, next_buf);
        asm volatile("cp.async.commit_group;\n" ::);
        if (st + kStages < S) load_w(st + kStages, rawn);
        const unsigned tile = sbase + buf * kBufBytes;
        const float* sw = reinterpret_cast<const float*>(smem + buf * kBufBytes + (kTileBytes + kActivationBytes));
        const float* sa = sw + 2U * kTileRows;  // [entry][2]
#pragma unroll
        for (std::uint32_t kk = 0; kk < 2U; ++kk) {
            if (kk == 1U && st + kAhead < S) store_w(st + kAhead, next_buf, rawc);  // the next stage decoded among this one's MMAs
            std::uint32_t af[2][4];
#pragma unroll
            for (std::uint32_t i = 0; i < 2U; ++i)
                ldsm_x4(tile + swz(a_row + 16U * i, 2U * kk + a_ch), af[i][0], af[i][1], af[i][2], af[i][3]);
            if (kKind == kDownI8 && kk == 0U && (st & 3U) == 0U) {
                std::uint32_t am[2][4];
#pragma unroll
                for (std::uint32_t i = 0; i < 2U; ++i) {
                    const std::uint32_t rr = a_row + 16U * i;
                    ldsm_x4(sbase + kSmem + rr * 32U + 16U * (a_ch ^ ((rr >> 2U) & 1U)), am[i][0], am[i][1], am[i][2], am[i][3]);
                }
#pragma unroll
                for (std::uint32_t q = 0; q < NT / 2U; ++q) {
                    if (q >= live_pairs) break;
                    const std::uint32_t rr = b_row + kEntryStep * q;
                    std::uint32_t bm[4];
                    ldsm_x4(sbase + kSmem + kMinBytes + rr * 32U + 16U * (b_ch ^ ((rr >> 2U) & 1U)), bm[0], bm[1], bm[2], bm[3]);
#pragma unroll
                    for (std::uint32_t i = 0; i < 2U; ++i) {
                        mma_f16(am[i], bm[0], bm[1], acc[i][2U * q]);
                        mma_f16(am[i], bm[2], bm[3], acc[i][2U * q + 1U]);
                    }
                }
            }
            float w0[2] = {0.0F, 0.0F}, w1[2] = {0.0F, 0.0F};
            if (kI8) {
#pragma unroll
                for (std::uint32_t i = 0; i < 2U; ++i) {
                    w0[i] = sw[kk * kTileRows + 32U * wm + 16U * i + gid];
                    w1[i] = sw[kk * kTileRows + 32U * wm + 16U * i + gid + 8U];
                }
            }
#pragma unroll
            for (std::uint32_t q = 0; q < NT / 2U; ++q) {
                if (q >= live_pairs) break;
                std::uint32_t b[4];
                ldsm_x4(tile + kTileBytes + swz(b_row + kEntryStep * q, 2U * kk + b_ch), b[0], b[1], b[2], b[3]);
                if (kI8) {
                    // the pair's four IMMAs first, then their scaled terms
                    int d[2][2][4];
#pragma unroll
                    for (std::uint32_t hf = 0; hf < 2U; ++hf)
#pragma unroll
                        for (std::uint32_t i = 0; i < 2U; ++i) mma_s8(af[i], b[2U * hf], b[2U * hf + 1U], d[hf][i]);
#pragma unroll
                    for (std::uint32_t hf = 0; hf < 2U; ++hf) {
                        const std::uint32_t k = 2U * q + hf;
                        const float4 a4 = *reinterpret_cast<const float4*>(sa + 2U * (kEntryStep * q + 16U * wn + 8U * hf + 2U * t));
                        const float2 a = kk ? make_float2(a4.y, a4.w) : make_float2(a4.x, a4.z);
#pragma unroll
                        for (std::uint32_t i = 0; i < 2U; ++i) {
                            acc[i][k][0] = __fmaf_rn(static_cast<float>(d[hf][i][0]), __fmul_rn(w0[i], a.x), acc[i][k][0]);
                            acc[i][k][1] = __fmaf_rn(static_cast<float>(d[hf][i][1]), __fmul_rn(w0[i], a.y), acc[i][k][1]);
                            acc[i][k][2] = __fmaf_rn(static_cast<float>(d[hf][i][2]), __fmul_rn(w1[i], a.x), acc[i][k][2]);
                            acc[i][k][3] = __fmaf_rn(static_cast<float>(d[hf][i][3]), __fmul_rn(w1[i], a.y), acc[i][k][3]);
                        }
                    }
                } else {
#pragma unroll
                    for (std::uint32_t hf = 0; hf < 2U; ++hf)
#pragma unroll
                        for (std::uint32_t i = 0; i < 2U; ++i) mma_f16(af[i], b[2U * hf], b[2U * hf + 1U], acc[i][2U * q + hf]);
                }
            }
        }
#pragma unroll
        for (std::uint32_t i = 0; i < 8U; ++i) rawc[i] = rawn[i];
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    if (kGate && p.act_codes) {
        // act[entry][64 outputs] in FP32 through shared memory, then the A8 of every (entry, 32 outputs) by a warp
        __syncthreads();
        auto* tile = reinterpret_cast<float*>(smem);  // [E][64 + 4]
        constexpr std::uint32_t kPitch = 68U;
#pragma unroll
        for (std::uint32_t i = 0; i < 2U; ++i) {
            const std::uint32_t o = 8U * (2U * wm + i) + gid;
#pragma unroll
            for (std::uint32_t k = 0; k < NT; ++k) {
                const std::uint32_t e = kEntryStep * (k >> 1U) + 16U * wn + 8U * (k & 1U) + 2U * t;
#pragma unroll
                for (std::uint32_t c = 0; c < 2U; ++c) {
                    const float gv = fminf(acc[i][k][c], p.limit);
                    const float uv = fmaxf(fminf(acc[i][k][2U + c], p.limit), -p.limit);
                    tile[(e + c) * kPitch + o] = __fdividef(gv, 1.0F + __expf(-gv)) * uv;
                }
            }
        }
        __syncthreads();
        for (std::uint32_t item = warp; item < 2U * n; item += kThreads / 32U) {
            const std::uint32_t m = item >> 1U, g = item & 1U;
            const A8Group a = a8_group(tile[m * kPitch + 32U * g + lane]);
            const std::uint64_t slot = s_slot[m], grp = 2U * blockIdx.x + g;
            p.act_codes[slot * p.inter + 32U * grp + lane] = a.code;
            std::int32_t half = a.code;
#pragma unroll
            for (int d = 8; d >= 1; d /= 2) half += __shfl_xor_sync(kFull, half, d);
            if (lane == 0U) p.act_deltas[slot * (p.inter / 32U) + grp] = a.delta;
            if ((lane & 15U) == 0U) p.act_bmin[slot * (p.inter / 16U) + 2U * grp + (lane >> 4U)] = __float2half_rn(a.delta * static_cast<float>(half));
        }
    } else if (kGate) {
        // act[entry][64 outputs] in FP16 through shared memory, then 16-byte rows out
        __syncthreads();
        auto* tile = reinterpret_cast<__half*>(smem);  // [E][64 + 8]
        constexpr std::uint32_t kPitch = 72U;
#pragma unroll
        for (std::uint32_t i = 0; i < 2U; ++i) {
            const std::uint32_t o = 8U * (2U * wm + i) + gid;  // output within the block's 64
#pragma unroll
            for (std::uint32_t k = 0; k < NT; ++k) {
                const std::uint32_t e = kEntryStep * (k >> 1U) + 16U * wn + 8U * (k & 1U) + 2U * t;
#pragma unroll
                for (std::uint32_t c = 0; c < 2U; ++c) {
                    const float gv = fminf(acc[i][k][c], p.limit);
                    const float uv = fmaxf(fminf(acc[i][k][2U + c], p.limit), -p.limit);
                    const float a = __fdividef(gv, 1.0F + __expf(-gv)) * uv;
                    tile[(e + c) * kPitch + o] = __float2half_rn(a);
                }
            }
        }
        __syncthreads();
        for (std::uint32_t item = tid; item < n * 8U; item += kThreads) {
            const std::uint32_t m = item >> 3U, c = item & 7U;
            *reinterpret_cast<uint4*>(p.act + std::uint64_t{s_slot[m]} * p.inter + blockIdx.x * 64U + 8U * c) =
                *reinterpret_cast<const uint4*>(tile + m * kPitch + 8U * c);
        }
    } else {
#pragma unroll
        for (std::uint32_t i = 0; i < 2U; ++i) {
            const std::uint32_t o = blockIdx.x * kTileRows + 32U * wm + 16U * i + gid;
#pragma unroll
            for (std::uint32_t k = 0; k < NT; ++k) {
                const std::uint32_t e = kEntryStep * (k >> 1U) + 16U * wn + 8U * (k & 1U) + 2U * t;
#pragma unroll
                for (std::uint32_t c = 0; c < 2U; ++c)
                    if (e + c < n) {
                        __half* out = p.raw + std::uint64_t{s_slot[e + c]} * p.hidden + o;
                        out[0] = sat_half(acc[i][k][c]);
                        out[8] = sat_half(acc[i][k][2U + c]);
                    }
            }
        }
    }
}

template <int kKind, bool kSignedLut = false, std::uint32_t kThr = 256U, bool kNarrowOnly = false>
__global__ __launch_bounds__(kThr, kThr == 512U ? 1 : 2) void fast_kernel(FastParams p) {
    constexpr bool kGate = kKind == kGateI8 || kKind == kGateF16;
    const std::uint32_t batch = blockIdx.y;
    if (batch >= *p.batches.total) return;
    extern __shared__ __align__(128) unsigned char smem[];
    __shared__ std::uint32_t s_arow[kThr == 512U ? 256U : kMaxE], s_slot[kThr == 512U ? 256U : kMaxE];
    __shared__ std::uint64_t grid[256];
    __shared__ std::uint64_t sign_masks[kSignedLut ? 128U : 1U];
    const std::uint32_t tid = threadIdx.x;
    if (kGate && (kThr == 256U || tid < 256U)) grid[tid] = g_fast_grid[tid];
    if (kSignedLut && tid < 128U) sign_masks[tid] = g_sign_masks[tid];

    const std::uint32_t j = p.batches.expert[batch], e0 = p.batches.begin[batch], n = p.batches.count[batch];
    if constexpr (kThr == 512U) { if (n <= 128U) return; }
    if constexpr (kNarrowOnly) { if (n > 128U) return; }
    if (tid < n) {
        s_slot[tid] = p.slots[e0 + tid];
        s_arow[tid] = kGate ? p.rows[e0 + tid] : s_slot[tid];
    }
    const std::uint64_t expert = p.experts[j];
    const std::uint8_t* wrow = nullptr;
    if (kThr == 256U || tid < 256U) {
        const std::uint32_t r = tid >> 1U;
        if (kGate) {
            const std::uint32_t o = blockIdx.x * 64U + 8U * (r >> 4U) + (r & 7U);
            wrow = ((r & 8U) ? p.w1 : p.w0) + expert * p.expert_stride + std::uint64_t{o} * p.row_bytes;
        } else {
            wrow = p.w0 + expert * p.expert_stride + std::uint64_t{blockIdx.x * kTileRows + r} * p.row_bytes;
        }
    }
    __syncthreads();
    if constexpr (kThr == 512U) {
        fast_body<kKind, 256U, kSignedLut, 512U>(p, smem, grid, s_arow, s_slot, n, wrow, sign_masks);
    } else {
        if (n > 64U) fast_body<kKind, 128U, kSignedLut>(p, smem, grid, s_arow, s_slot, n, wrow, sign_masks);
        else if (n > 32U) fast_body<kKind, 64U, kSignedLut>(p, smem, grid, s_arow, s_slot, n, wrow, sign_masks);
        else fast_body<kKind, 32U, kSignedLut>(p, smem, grid, s_arow, s_slot, n, wrow, sign_masks);
    }
}

// ------------------------------------------------------------------------------------- FP16 kernels (warp-specialized)
// One block per SM, 12 warps: 8 MMA warps (4 x 2: 32 weight rows x up to 64 entries each) and 4 producer warps that
// copy the entries' FP16 operands (cp.async) and decode the weights to FP16, three stages of 64 weights (128-byte
// rows, 8 chunks XOR-swizzled by row & 7) in flight. Named barriers: FULL(s) producers arrive / MMA warps wait,
// EMPTY(s) MMA warps arrive / producers wait, 384 threads each.
constexpr std::uint32_t kF16Row = 128U;
constexpr std::uint32_t kF16Tile = kTileRows * kF16Row;  // 16 KB weights, then 16 KB entries
constexpr std::uint32_t kF16Stage = 2U * kF16Tile;
constexpr std::uint32_t kF16Smem = 3U * kF16Stage;

__device__ __forceinline__ std::uint32_t swz8(std::uint32_t r, std::uint32_t c) { return r * kF16Row + 16U * (c ^ (r & 7U)); }
__device__ __forceinline__ void bar_sync(std::uint32_t id, std::uint32_t count) {
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void bar_arrive(std::uint32_t id, std::uint32_t count) {
    asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(count) : "memory");
}

template <int kKind, std::uint32_t E, std::uint32_t kThr>
__device__ __forceinline__ void f16_body(const FastParams& p, unsigned char* smem, const std::uint64_t* grid, const std::uint32_t* s_arow,
                                         const std::uint32_t* s_slot, std::uint32_t n, const std::uint8_t* wrow) {
    constexpr bool kGate = kKind == kGateF16;
    constexpr std::uint32_t NT = E / 16U;
    const std::uint32_t tid = threadIdx.x, lane = tid & 31U, warp = tid / 32U, gid = lane >> 2U, t = lane & 3U;
    const std::uint32_t width = kGate ? p.hidden : p.inter, S = width / 64U;
    const unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(smem));

    if (warp >= 8U) {
        // ------------------------------------------------ producers: tile row r, both groups of every stage
        // kThr 384: 4 producer warps, both groups of row r; 512: 8 warps, group ptid >> 7 of row r
        constexpr std::uint32_t kProd = (kThr - 256U) / 32U;
        const std::uint32_t ptid = tid - 256U, r = ptid & 127U, myh = ptid >> 7U;
        std::uint32_t rawc[10] = {}, rawn[10] = {};
        auto load_w = [&](std::uint32_t st, std::uint32_t (&raw)[10]) {
            const std::uint32_t g0 = 2U * st, s0 = g0 & 7U;
            if (kGate) {  // the two groups' 16 bytes at blk + 2 + 8 s0 (2-byte aligned): five aligned words
                const std::uint8_t* blk = wrow + (g0 >> 3U) * 66U;
                const std::uint8_t* q = blk + 2U + 8U * s0;
                const auto* w = reinterpret_cast<const std::uint32_t*>(reinterpret_cast<std::uintptr_t>(q) & ~std::uintptr_t{3});
                raw[0] = __ldg(w); raw[1] = __ldg(w + 1); raw[2] = __ldg(w + 2); raw[3] = __ldg(w + 3);
                raw[4] = (reinterpret_cast<std::uintptr_t>(q) & 3U) ? __ldg(w + 4) : 0U;  // (a row's last bytes otherwise)
                raw[5] = __ldg(reinterpret_cast<const std::uint16_t*>(blk)) | ((reinterpret_cast<std::uintptr_t>(q) & 3U) << 16U);
            } else {  // the two groups share 32 code bytes; sub-block scales 2 s0 .. 2 s0 + 3; d, dmin
                const std::uint8_t* blk = wrow + (g0 >> 3U) * 84U;
                const auto* qs = reinterpret_cast<const std::uint32_t*>(blk + 16U + 32U * (s0 >> 2U));
#pragma unroll
                for (std::uint32_t w = 0; w < 8U; ++w) raw[w] = __ldg(qs + w);
                raw[8] = __ldg(reinterpret_cast<const std::uint32_t*>(blk + 2U * s0));
                raw[9] = __ldg(reinterpret_cast<const std::uint32_t*>(blk + 80U));
            }
        };
        auto store_w = [&](std::uint32_t st, unsigned char* tile, const std::uint32_t (&raw)[10]) {
            const std::uint32_t g0 = 2U * st;
            if (kGate) {
                const std::uint32_t sh = 8U * (raw[5] >> 16U);  // byte offset of q in its word: 0 or 2
                std::uint32_t wd[4];
#pragma unroll
                for (std::uint32_t k = 0; k < 4U; ++k) wd[k] = __funnelshift_r(raw[k], raw[k + 1U], sh);
                const float d = fp16_bits(raw[5] & 0xffffU);
#pragma unroll
                for (std::uint32_t h = 0; h < 2U; ++h) {
                    if (kProd == 8U && h != myh) continue;
                    const std::uint32_t aux0 = wd[2U * h], aux1 = wd[2U * h + 1U];
                    const __half2 sc2 = __float2half2_rn(d * (0.5F + static_cast<float>(aux1 >> 28U)) * 0.25F);
#pragma unroll
                    for (std::uint32_t l = 0; l < 4U; ++l) {
                        const std::uint64_t gr = grid[__byte_perm(aux0, 0U, 0x4440U + l)];
                        const std::uint32_t sb = sign_byte(aux1 >> (7U * l));
                        std::uint32_t o[4];
#pragma unroll
                        for (std::uint32_t q = 0; q < 4U; ++q) {
                            const std::uint32_t word = q < 2U ? static_cast<std::uint32_t>(gr) : static_cast<std::uint32_t>(gr >> 32U);
                            const __half2 m = bytes_h2(word, (q & 1U) ? 0x5342U : 0x5140U);
                            const std::uint32_t sbits = sb >> (2U * q);
                            const std::uint32_t sign = ((sbits & 1U) << 15U) | ((sbits & 2U) << 30U);
                            o[q] = h2u(__hmul2(u2h(h2u(m) ^ sign), sc2));
                        }
                        *reinterpret_cast<uint4*>(tile + swz8(r, 4U * h + l)) = make_uint4(o[0], o[1], o[2], o[3]);
                    }
                }
            } else {
                const float d = fp16_bits(raw[9] & 0xffffU), dmin = fp16_bits(raw[9] >> 16U);
#pragma unroll
                for (std::uint32_t h = 0; h < 2U; ++h) {
                    if (kProd == 8U && h != myh) continue;
                    const std::uint32_t shift = 2U * ((g0 + h) & 3U);
#pragma unroll
                    for (std::uint32_t half = 0; half < 2U; ++half) {
                        const std::uint32_t sc = (raw[8] >> (16U * h + 8U * half)) & 0xffU;
                        const __half2 a2 = __float2half2_rn(d * static_cast<float>(sc & 15U));
                        const __half2 nb2 = __float2half2_rn(-dmin * static_cast<float>(sc >> 4U));
#pragma unroll
                        for (std::uint32_t c2 = 0; c2 < 2U; ++c2) {
                            std::uint32_t o[4];
#pragma unroll
                            for (std::uint32_t w2 = 0; w2 < 2U; ++w2) {
                                const std::uint32_t x = (raw[4U * half + 2U * c2 + w2] >> shift) & 0x03030303U;
                                o[2U * w2] = h2u(__hfma2(bytes_h2(x, 0x5140U), a2, nb2));
                                o[2U * w2 + 1U] = h2u(__hfma2(bytes_h2(x, 0x5342U), a2, nb2));
                            }
                            *reinterpret_cast<uint4*>(tile + swz8(r, 4U * h + 2U * half + c2)) = make_uint4(o[0], o[1], o[2], o[3]);
                        }
                    }
                }
            }
        };
        const auto* a = static_cast<const unsigned char*>(p.a);
        load_w(0U, rawc);
        for (std::uint32_t st = 0; st < S; ++st) {
            const std::uint32_t s = st % 3U;
            if (st >= 3U) bar_sync(4U + s, kThr);  // the MMA warps are done with stage st - 3
            const unsigned btile = sbase + s * kF16Stage + kF16Tile;
            for (std::uint32_t item = ptid; item < n * 8U; item += 32U * kProd) {
                const std::uint32_t m = item >> 3U, c = item & 7U;
                cp_async16(btile + swz8(m, c), a + std::uint64_t{s_arow[m]} * (2U * width) + st * 128U + c * 16U);
            }
            asm volatile("cp.async.commit_group;\n" ::);
            if (st + 1U < S) load_w(st + 1U, rawn);
            store_w(st, smem + s * kF16Stage, rawc);
#pragma unroll
            for (std::uint32_t i = 0; i < 10U; ++i) rawc[i] = rawn[i];
            asm volatile("cp.async.wait_group 0;\n" ::);
            bar_arrive(1U + s, kThr);
        }
        return;
    }

    // ---------------------------------------------------- MMA warps
    const std::uint32_t wm = warp & 3U, wn = warp >> 2U;
    float acc[2][NT][4];
#pragma unroll
    for (std::uint32_t i = 0; i < 2U; ++i)
#pragma unroll
        for (std::uint32_t k = 0; k < NT; ++k)
#pragma unroll
            for (std::uint32_t e = 0; e < 4U; ++e) acc[i][k][e] = 0.0F;
    const std::uint32_t a_row = 32U * wm + (lane & 15U), a_ch = lane >> 4U;
    const std::uint32_t b_row = 16U * wn + (lane & 7U) + 8U * (lane >> 4U), b_ch = (lane >> 3U) & 1U;
    const std::uint32_t live_pairs = n > 16U * wn ? (n - 16U * wn + 31U) / 32U : 0U;
    for (std::uint32_t st = 0; st < S; ++st) {
        const std::uint32_t s = st % 3U;
        bar_sync(1U + s, kThr);
        const unsigned tile = sbase + s * kF16Stage;
        std::uint32_t af[2][2][4], bf[2][NT / 2U][4];
        auto load_frag = [&](std::uint32_t kk, std::uint32_t fb) {
#pragma unroll
            for (std::uint32_t i = 0; i < 2U; ++i)
                ldsm_x4(tile + swz8(a_row + 16U * i, 2U * kk + a_ch), af[fb][i][0], af[fb][i][1], af[fb][i][2], af[fb][i][3]);
#pragma unroll
            for (std::uint32_t q = 0; q < NT / 2U; ++q)
                if (q < live_pairs)
                    ldsm_x4(tile + kF16Tile + swz8(b_row + 32U * q, 2U * kk + b_ch), bf[fb][q][0], bf[fb][q][1], bf[fb][q][2], bf[fb][q][3]);
        };
        load_frag(0U, 0U);
#pragma unroll
        for (std::uint32_t kk = 0; kk < 4U; ++kk) {
            if (kk < 3U) load_frag(kk + 1U, (kk + 1U) & 1U);
            const std::uint32_t fb = kk & 1U;
#pragma unroll
            for (std::uint32_t q = 0; q < NT / 2U; ++q)
                if (q < live_pairs)
#pragma unroll
                    for (std::uint32_t hf = 0; hf < 2U; ++hf)
#pragma unroll
                        for (std::uint32_t i = 0; i < 2U; ++i) mma_f16(af[fb][i], bf[fb][q][2U * hf], bf[fb][q][2U * hf + 1U], acc[i][2U * q + hf]);
        }
        if (st + 3U < S) bar_arrive(4U + s, kThr);
    }

    if (kGate) {
        bar_sync(7U, 256U);  // every MMA warp done reading the stages (the producers finished writing them)
        auto* tile = reinterpret_cast<__half*>(smem);  // [E][64 + 8]
        constexpr std::uint32_t kPitch = 72U;
#pragma unroll
        for (std::uint32_t i = 0; i < 2U; ++i) {
            const std::uint32_t o = 8U * (2U * wm + i) + gid;
#pragma unroll
            for (std::uint32_t k = 0; k < NT; ++k) {
                const std::uint32_t e = 32U * (k >> 1U) + 16U * wn + 8U * (k & 1U) + 2U * t;
#pragma unroll
                for (std::uint32_t c = 0; c < 2U; ++c) {
                    const float gv = fminf(acc[i][k][c], p.limit);
                    const float uv = fmaxf(fminf(acc[i][k][2U + c], p.limit), -p.limit);
                    tile[(e + c) * kPitch + o] = __float2half_rn(__fdividef(gv, 1.0F + __expf(-gv)) * uv);
                }
            }
        }
        bar_sync(7U, 256U);
        for (std::uint32_t item = tid; item < n * 8U; item += 256U) {
            const std::uint32_t m = item >> 3U, c = item & 7U;
            *reinterpret_cast<uint4*>(p.act + std::uint64_t{s_slot[m]} * p.inter + blockIdx.x * 64U + 8U * c) =
                *reinterpret_cast<const uint4*>(tile + m * kPitch + 8U * c);
        }
    } else {
#pragma unroll
        for (std::uint32_t i = 0; i < 2U; ++i) {
            const std::uint32_t o = blockIdx.x * kTileRows + 32U * wm + 16U * i + gid;
#pragma unroll
            for (std::uint32_t k = 0; k < NT; ++k) {
                const std::uint32_t e = 32U * (k >> 1U) + 16U * wn + 8U * (k & 1U) + 2U * t;
#pragma unroll
                for (std::uint32_t c = 0; c < 2U; ++c)
                    if (e + c < n) {
                        __half* out = p.raw + std::uint64_t{s_slot[e + c]} * p.hidden + o;
                        out[0] = sat_half(acc[i][k][c]);
                        out[8] = sat_half(acc[i][k][2U + c]);
                    }
            }
        }
    }
}

template <int kKind, int kThr>
__global__ __launch_bounds__(kThr, 1) void f16_kernel(FastParams p) {
    constexpr bool kGate = kKind == kGateF16;
    const std::uint32_t batch = blockIdx.y;
    if (batch >= *p.batches.total) return;
    extern __shared__ __align__(128) unsigned char smem[];
    __shared__ std::uint32_t s_arow[kMaxE], s_slot[kMaxE];
    __shared__ std::uint64_t grid[256];
    const std::uint32_t tid = threadIdx.x;
    if (kGate && (kThr == 256U || tid < 256U)) grid[tid] = g_fast_grid[tid];
    const std::uint32_t j = p.batches.expert[batch], e0 = p.batches.begin[batch], n = p.batches.count[batch];
    if (tid < n) {
        s_slot[tid] = p.slots[e0 + tid];
        s_arow[tid] = kGate ? p.rows[e0 + tid] : s_slot[tid];
    }
    const std::uint64_t expert = p.experts[j];
    const std::uint32_t r = tid & 127U;  // producers' tile row
    const std::uint8_t* wrow =
        kGate ? ((r & 8U) ? p.w1 : p.w0) + expert * p.expert_stride + std::uint64_t{blockIdx.x * 64U + 8U * (r >> 4U) + (r & 7U)} * p.row_bytes
              : p.w0 + expert * p.expert_stride + std::uint64_t{blockIdx.x * kTileRows + r} * p.row_bytes;
    __syncthreads();
    if (n > 64U) f16_body<kKind, 128U, kThr>(p, smem, grid, s_arow, s_slot, n, wrow);
    else if (n > 32U) f16_body<kKind, 64U, kThr>(p, smem, grid, s_arow, s_slot, n, wrow);
    else f16_body<kKind, 32U, kThr>(p, smem, grid, s_arow, s_slot, n, wrow);
}

template <int kKind>
void launch_f16(dim3 grid, const FastParams& p, cudaStream_t stream) {
    // producer warps: 8 for gate/up (the IQ2_XXS decode), 4 for down (the MMA warps keep more registers)
    constexpr int kThr = kKind == kGateF16 ? 512 : 384;
    int device = 0;
    STRIX_CUDA_CHECK(cudaGetDevice(&device));
    static std::uint64_t attr = 0;
    if (device < 64 && !((attr >> device) & 1U)) {
        STRIX_CUDA_CHECK(cudaFuncSetAttribute(f16_kernel<kKind, kThr>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kF16Smem)));
        attr |= std::uint64_t{1} << device;
    }
    f16_kernel<kKind, kThr><<<grid, kThr, kF16Smem, stream>>>(p);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

template <int kKind, bool kSignedLut = false, std::uint32_t kThr = 256U, bool kNarrowOnly = false>
void launch(dim3 grid, const FastParams& p, cudaStream_t stream) {
    constexpr std::uint32_t wide_stage = kTileBytes + 256U * kRowBytes + kScaleBytes + 256U * 8U;
    constexpr std::uint32_t wide_smem = 3U * wide_stage + (kKind == kDownI8 ? kMinBytes + 256U * 32U : 0U);
    constexpr std::uint32_t smem = kThr == 512U ? wide_smem : (kKind == kDownI8 ? kSmemDownI8 : kSmem);
    int device = 0;
    STRIX_CUDA_CHECK(cudaGetDevice(&device));
    static std::uint64_t attr = 0;  // devices with the attribute set
    if (device < 64 && !((attr >> device) & 1U)) {
        STRIX_CUDA_CHECK(cudaFuncSetAttribute(fast_kernel<kKind, kSignedLut, kThr, kNarrowOnly>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        attr |= std::uint64_t{1} << device;
    }
    fast_kernel<kKind, kSignedLut, kThr, kNarrowOnly><<<grid, kThr, smem, stream>>>(p);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

template <int kKind, bool kSignedLut = false>
void launch_grouped(dim3 grid, const FastParams& p, cudaStream_t stream, bool wide) {
    if (wide) {
        launch<kKind, kSignedLut, 256U, true>(grid, p, stream);
        launch<kKind, kSignedLut, 512U>(grid, p, stream);
    } else {
        launch<kKind, kSignedLut>(grid, p, stream);
    }
}

void check_fast(const MoeShape& s) {
    if (s.hidden % 256U || s.inter % 256U || s.hidden % kTileRows || s.inter % 64U || s.top_k == 0U || s.experts == 0U ||
        s.experts > 1024U)
        throw std::invalid_argument("ds41 cuda moe fast: unsupported shape");
}

}  // namespace

std::size_t moe_fast_workspace_bytes(std::uint32_t capacity, const MoeShape& shape) noexcept { return carve(nullptr, capacity, shape).bytes; }

void enqueue_moe_fast_layer(const MoeFastLayerView& l, cudaStream_t stream) {
    if (l.rows == 0U) return;
    if (l.rows > l.capacity || !l.workspace) throw std::invalid_argument("ds41 cuda moe fast: rows above the workspace capacity");
    const MoeShape& s = l.shape;
    check_fast(s);
    if (l.wide_tiles && (!l.xa.codes || !l.down_i8)) throw std::invalid_argument("ds41 cuda moe fast: wide tiles require INT8");
    if (l.signed_lut && !l.xa.codes) throw std::invalid_argument("ds41 cuda moe fast: signed LUT requires INT8 gate/up");
    upload_fast_grid();
    if (l.signed_lut) upload_sign_masks();
    const FastSpace w = carve(l.workspace, l.capacity, s);
    // STRIX_FMOE_TIMES: device ms per phase on stderr (synchronizes; diagnostics only)
    static const bool times = std::getenv("STRIX_FMOE_TIMES") != nullptr;
    static cudaEvent_t ev[8];
    int nev = 0;
    const char* names[8] = {};
    auto step = [&](const char* what) {
        if (!times) return;
        if (!ev[0])
            for (auto& e : ev) STRIX_CUDA_CHECK(cudaEventCreate(&e));
        names[nev] = what;
        STRIX_CUDA_CHECK(cudaEventRecord(ev[nev++], stream));
    };
    step("start");
    MoeRoutesView b;
    b.ids = l.ids;
    b.rows = l.rows;
    b.top_k = s.top_k;
    b.experts = s.experts;
    b.split = l.split;
    b.scratch = w.scratch;
    b.out = w.routes;
    enqueue_moe_routes(b, stream);
    step("routes");
    if (l.wide_tiles) fast_batches_kernel<256U><<<1, 1024, 0, stream>>>(w.routes.begins, w.routes.counts, w.batches);
    else fast_batches_kernel<kMaxE><<<1, 1024, 0, stream>>>(w.routes.begins, w.routes.counts, w.batches);
    const std::size_t n2 = std::size_t{l.rows} * s.hidden / 2U;
    if (!l.xa.codes) to_half_kernel<<<static_cast<unsigned>((n2 + 255U) / 256U), 256, 0, stream>>>(l.x, n2, reinterpret_cast<__half2*>(w.x16));
    step("batches+half");
    const std::uint32_t mb = max_batches(l.rows, s);
    const bool i8 = l.xa.codes != nullptr;
    FastParams p{l.weights.gate, l.weights.up, s.gate_expert_bytes(), s.gate_row_bytes(), s.hidden, s.inter, w.routes.experts,
                 w.routes.rows, w.routes.slots, w.batches, i8 ? static_cast<const void*>(l.xa.codes) : w.x16, l.xa.deltas, nullptr,
                 s.limit, w.act16, l.down_i8 ? w.act_codes : nullptr, w.act_deltas, w.act_bmin, nullptr};
    if (i8 && l.signed_lut) launch_grouped<kGateI8, true>(dim3(s.inter / 64U, mb), p, stream, l.wide_tiles);
    else if (i8) launch_grouped<kGateI8>(dim3(s.inter / 64U, mb), p, stream, l.wide_tiles);
    else launch_f16<kGateF16>(dim3(s.inter / 64U, mb), p, stream);
    step("gate_up");
    p.w0 = l.weights.down;
    p.w1 = nullptr;
    p.expert_stride = s.down_expert_bytes();
    p.row_bytes = s.down_row_bytes();
    p.a = l.down_i8 ? static_cast<const void*>(w.act_codes) : w.act16;
    p.a_deltas = l.down_i8 ? w.act_deltas : nullptr;
    p.a_bmin = w.act_bmin;
    p.act = nullptr;
    p.act_codes = nullptr;
    p.raw = reinterpret_cast<__half*>(l.raw);  // the caller's raw buffer holds the FP16 slot vectors
    if (reinterpret_cast<std::uintptr_t>(l.weights.down) % 4U) throw std::invalid_argument("ds41 cuda moe fast: down weights not 4-byte aligned");
    if (l.down_i8) launch_grouped<kDownI8>(dim3(s.hidden / kTileRows, mb), p, stream, l.wide_tiles);
    else launch_f16<kDownF16>(dim3(s.hidden / kTileRows, mb), p, stream);
    step("down");
    const std::size_t groups = std::size_t{l.rows} * (s.hidden / 8U);
    fast_combine_kernel<<<static_cast<unsigned>((groups + 255U) / 256U), 256, 0, stream>>>(
        reinterpret_cast<const __half*>(l.raw), l.shared, l.route_weights, l.rows, s.top_k, s.hidden, l.y);
    STRIX_CUDA_CHECK(cudaGetLastError());
    step("combine");
    if (times) {
        STRIX_CUDA_CHECK(cudaEventSynchronize(ev[nev - 1]));
        std::fprintf(stderr, "fmoe rows %u:", l.rows);
        for (int i = 1; i < nev; ++i) {
            float ms = 0;
            cudaEventElapsedTime(&ms, ev[i - 1], ev[i]);
            std::fprintf(stderr, " %s %.2f", names[i], ms);
        }
        std::fprintf(stderr, "\n");
    }
}


namespace {
__global__ void fast_map_ids_kernel(const std::uint32_t* ids, const std::uint32_t* perm, std::uint32_t n, std::uint32_t* out) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = perm[ids[i]];
}
}  // namespace

void enqueue_moe_fast_layer_ranges(const MoeFastLayerView& l, const std::uint32_t* perm, std::uint32_t* mapped_ids, const MoeRange* ranges,
                                   std::uint32_t range_count, cudaStream_t stream) {
    if (l.rows == 0U) return;
    if (l.rows > l.capacity || !l.workspace) throw std::invalid_argument("ds41 cuda moe fast: rows above the workspace capacity");
    if (perm && !mapped_ids) throw std::invalid_argument("ds41 cuda moe fast ranges: a permutation needs the mapped ids scratch");
    const MoeShape& s = l.shape;
    check_fast(s);
    if (l.wide_tiles && (!l.xa.codes || !l.down_i8)) throw std::invalid_argument("ds41 cuda moe fast: wide tiles require INT8");
    if (l.signed_lut && !l.xa.codes) throw std::invalid_argument("ds41 cuda moe fast: signed LUT requires INT8 gate/up");
    upload_fast_grid();
    if (l.signed_lut) upload_sign_masks();
    const FastSpace w = carve(l.workspace, l.capacity, s);
    const std::uint32_t n = l.rows * s.top_k;
    const std::uint32_t* ids = l.ids;
    if (perm) {
        fast_map_ids_kernel<<<(n + 255U) / 256U, 256, 0, stream>>>(l.ids, perm, n, mapped_ids);
        STRIX_CUDA_CHECK(cudaGetLastError());
        ids = mapped_ids;
    }
    const std::size_t n2 = std::size_t{l.rows} * s.hidden / 2U;
    if (!l.xa.codes) to_half_kernel<<<static_cast<unsigned>((n2 + 255U) / 256U), 256, 0, stream>>>(l.x, n2, reinterpret_cast<__half2*>(w.x16));
    const std::uint32_t mb = max_batches(l.rows, s);
    const bool i8 = l.xa.codes != nullptr;
    for (std::uint32_t k = 0; k < range_count; ++k) {
        const MoeRange& g = ranges[k];
        if (g.first >= g.end) continue;
        if (g.ready) {
            if (g.wait_begin) STRIX_CUDA_CHECK(cudaEventRecord(g.wait_begin, stream));
            STRIX_CUDA_CHECK(cudaStreamWaitEvent(stream, g.ready, 0));
            if (g.wait_end) STRIX_CUDA_CHECK(cudaEventRecord(g.wait_end, stream));
        }
        if (reinterpret_cast<std::uintptr_t>(g.weights.down) % 4U) throw std::invalid_argument("ds41 cuda moe fast: down weights not 4-byte aligned");
        MoeRoutesView b;
        b.ids = ids;
        b.rows = l.rows;
        b.top_k = s.top_k;
        b.experts = s.experts;
        b.split = l.split;
        b.first = g.first;
        b.end = g.end;
        b.base = g.base;
        b.scratch = w.scratch;
        b.out = w.routes;
        enqueue_moe_routes(b, stream);
        if (l.wide_tiles) fast_batches_kernel<256U><<<1, 1024, 0, stream>>>(w.routes.begins, w.routes.counts, w.batches);
        else fast_batches_kernel<kMaxE><<<1, 1024, 0, stream>>>(w.routes.begins, w.routes.counts, w.batches);
        FastParams p{g.weights.gate, g.weights.up, s.gate_expert_bytes(), s.gate_row_bytes(), s.hidden, s.inter, w.routes.experts,
                     w.routes.rows, w.routes.slots, w.batches, i8 ? static_cast<const void*>(l.xa.codes) : w.x16, l.xa.deltas, nullptr,
                     s.limit, w.act16, l.down_i8 ? w.act_codes : nullptr, w.act_deltas, w.act_bmin, nullptr};
        if (i8 && l.signed_lut) launch_grouped<kGateI8, true>(dim3(s.inter / 64U, mb), p, stream, l.wide_tiles);
        else if (i8) launch_grouped<kGateI8>(dim3(s.inter / 64U, mb), p, stream, l.wide_tiles);
        else launch_f16<kGateF16>(dim3(s.inter / 64U, mb), p, stream);
        p.w0 = g.weights.down;
        p.w1 = nullptr;
        p.expert_stride = s.down_expert_bytes();
        p.row_bytes = s.down_row_bytes();
        p.a = l.down_i8 ? static_cast<const void*>(w.act_codes) : w.act16;
        p.a_deltas = l.down_i8 ? w.act_deltas : nullptr;
        p.a_bmin = w.act_bmin;
        p.act = nullptr;
        p.act_codes = nullptr;
        p.raw = reinterpret_cast<__half*>(l.raw);
        if (l.down_i8) launch_grouped<kDownI8>(dim3(s.hidden / kTileRows, mb), p, stream, l.wide_tiles);
        else launch_f16<kDownF16>(dim3(s.hidden / kTileRows, mb), p, stream);
    }
    const std::size_t groups = std::size_t{l.rows} * (s.hidden / 8U);
    fast_combine_kernel<<<static_cast<unsigned>((groups + 255U) / 256U), 256, 0, stream>>>(
        reinterpret_cast<const __half*>(l.raw), l.shared, l.route_weights, l.rows, s.top_k, s.hidden, l.y);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace strix::ds41::cuda
