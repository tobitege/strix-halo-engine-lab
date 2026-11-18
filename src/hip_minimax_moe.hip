// minimax-m3 fused post-exchange kernel (declared order: strix/minimax_moe.hpp).
#include "strix/hip_minimax_moe.hpp"
#include "xport/exchange_format_hip.hpp"
#include "moe_core/route.hpp"
#include "minimax/moe/mmx_trellis_device.hpp"  // dev::transform_block (STRIX_MM_XB_PREP)
#include "core/f16_dense_tall.hpp"             // FAST router product (enqueue_router_wmma)

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace strix::minimax::moe::hip {
namespace {

constexpr unsigned kWaves = kNormThreads / 32;

// Strided views of one row's scores in the [2][kExperts][rows] scratch.
struct RowScores {
    const float* base;
    std::uint32_t stride;
    __device__ float operator[](std::uint32_t e) const { return base[static_cast<std::size_t>(e) * stride]; }
};

template <class Scores>
struct StridedRoutePolicy {
    static constexpr unsigned experts = kExperts, top_k = kTopE;
    static constexpr float sum_eps = kSumEps, route_scale = kRouteScale;
    Scores sc, ch;
    unsigned r;
    std::uint32_t* ids;
    float* weights;
    __device__ float score(std::uint32_t e) const { return sc[e]; }
    __device__ bool better(std::uint32_t e, std::uint32_t pick) const { return ch[e] > ch[pick]; }
    __device__ void write_id(unsigned k, std::uint32_t id) const { ids[r * kTopE + k] = id; }
    __device__ void write_weight(unsigned k, std::uint32_t, float weight) const {
        weights[r * kTopE + k] = weight;
    }
};

// Sum: exchange row fold (xport/row_fold.hpp). part0 is the folded slot = rank 0's partial + rank 1's partial, already
// added by the exchange progress thread with the same binary32 addition; part1 is null. x = h + folded: same bits.
// Split: the partials are in the chunk-major split layout (xport::SplitLayout{rows, kHidden, v.split_head}); the sums and
// their order are the unsplit kernel's, only the element addresses differ. Split = false is the unchanged kernel.
struct SplitRow {
    const float* base;
    std::size_t o0, o1;  // element j of the row is base[(j < head ? o0 : o1) + j]
    unsigned head;
    __device__ const float* at(unsigned j) const { return base + (j < head ? o0 : o1) + j; }
};
__device__ __forceinline__ SplitRow split_row(const float* base, unsigned r, unsigned rows, unsigned head) {
    return SplitRow{base, std::size_t{r} * head, std::size_t{rows} * head + std::size_t{r} * (kHidden - head) - head, head};
}
template <unsigned Ilp, bool Sum = false, bool Split = false, bool Half = false>
__global__ __launch_bounds__(kNormThreads) void post_exchange_kernel(PostExchangeView v) {
    static_assert(!(Half && (Sum || Split)), "binary16 partials cannot be folded or split");
    __shared__ __attribute__((aligned(16))) float hs[kHidden];  // 16-byte aligned: mode 2 stages float4 rows here
    __shared__ __attribute__((aligned(16))) float ys[kHidden];
    __shared__ float part[kNormThreads];
    __shared__ float sc[kExperts], ch[kExperts];
    const unsigned r = blockIdx.x, t = threadIdx.x, lane = t & 31U, wave = t >> 5U;
    constexpr unsigned kPer = kHidden / kNormThreads;
    static_assert(kHidden % kNormThreads == 0);
    [[maybe_unused]] float hv[Ilp == 2 ? kPer : 1], nw[Ilp == 2 ? kPer : 1];
    if constexpr (Ilp == 2) {  // mode 2: the residual and the norm weights do not depend on the exchange: load them
                               // before the receive poll (registers), so only the partials remain after it
#pragma unroll
        for (unsigned q = 0; q < kPer; ++q) {
            hv[q] = v.h[static_cast<std::size_t>(r) * kHidden + t + q * kNormThreads];
            nw[q] = v.norm_w[t + q * kNormThreads];
        }
    }
    if (v.rx_flag != nullptr) {  // the exchange module's in-kernel wait (xfer::mmx_wait without the gate)
        // Every block (one per row, <= 8 in decode) polls the RX flag itself: no block depends on another block being
        // scheduled. A failed exchange releases the flag with ~0, so the wait always ends.
        if (t == 0)
            while (__hip_atomic_load(v.rx_flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < v.rx_seq) __builtin_amdgcn_s_sleep(2);
        __syncthreads();
    }
    float* h = v.h + static_cast<std::size_t>(r) * kHidden;
    const float* p0 = Half ? nullptr : v.part0 + static_cast<std::size_t>(r) * kHidden;
    const float* p1 = Sum || Half ? nullptr : v.part1 + static_cast<std::size_t>(r) * kHidden;
    [[maybe_unused]] SplitRow s0{}, s1{};
    if constexpr (Split) {
        s0 = split_row(v.part0, r, v.rows, v.split_head);
        if constexpr (!Sum) s1 = split_row(v.part1, r, v.rows, v.split_head);
    }
    [[maybe_unused]] const _Float16* q0 = Half ? reinterpret_cast<const _Float16*>(v.part0) + static_cast<std::size_t>(r) * kHidden : nullptr;
    [[maybe_unused]] const _Float16* q1 = Half ? reinterpret_cast<const _Float16*>(v.part1) + static_cast<std::size_t>(r) * kHidden : nullptr;
    float acc = 0.0F;
    if constexpr (Ilp == 2) {  // STRIX_MM_XB_POSTILP mode 2: the partials (GPU-uncached exchange slots) staged into LDS
        // with coalesced 16-byte loads (4x fewer uncached requests), then the plain chain reads them from LDS: same
        // values, same per-thread j order, same expressions -> identical bits. ys is overwritten by y only after the
        // reduction barriers; each hs[j] is read and then written by the same thread.
        const auto* p0v = reinterpret_cast<const float4*>(p0);
        const auto* p1v = reinterpret_cast<const float4*>(p1);
        if constexpr (Half) {
            const auto* h0v = reinterpret_cast<const uint4*>(q0);
            const auto* h1v = reinterpret_cast<const uint4*>(q1);
            for (unsigned k = t; k < kHidden / 8U; k += kNormThreads) {
                const uint4 x0 = h0v[k], x1 = h1v[k];
                strix::xport::hip::widen_f16x8(x0, hs + 8U * k);
                strix::xport::hip::widen_f16x8(x1, ys + 8U * k);
            }
        }
        for (unsigned k = t; !Half && k < kHidden / 4U; k += kNormThreads) {
            if constexpr (Split) {  // a float4 never straddles the head boundary (head % 4 == 0)
                const float4 x0 = *reinterpret_cast<const float4*>(s0.at(4U * k));
                if constexpr (Sum) {
                    reinterpret_cast<float4*>(hs)[k] = x0;
                } else {
                    const float4 x1 = *reinterpret_cast<const float4*>(s1.at(4U * k));
                    reinterpret_cast<float4*>(hs)[k] = x0;
                    reinterpret_cast<float4*>(ys)[k] = x1;
                }
            } else if constexpr (Sum) {
                reinterpret_cast<float4*>(hs)[k] = p0v[k];
            } else {
                const float4 x0 = p0v[k], x1 = p1v[k];
                reinterpret_cast<float4*>(hs)[k] = x0;
                reinterpret_cast<float4*>(ys)[k] = x1;
            }
        }
        __syncthreads();
#pragma unroll
        for (unsigned q = 0; q < kPer; ++q) {
            const unsigned j = t + q * kNormThreads;
            const float x = Sum ? hv[q] + hs[j] : hv[q] + (hs[j] + ys[j]);
            h[j] = x;
            hs[j] = x;
            acc = __builtin_fmaf(x, x, acc);
        }
    } else if constexpr (Ilp == 1) {  // STRIX_MM_XB_POSTILP mode 1: every load issued before the first store (the partials live in GPU-uncached
                  // exchange slots; the store to h may alias them for the compiler, which serialises the plain loop into
                  // 24 round trips). Same per-thread chain (j ascending) and the same expressions: identical bits.
        float a[kPer], b[kPer], c[kPer];
#pragma unroll
        for (unsigned q = 0; q < kPer; ++q) {
            const unsigned j = t + q * kNormThreads;
            a[q] = h[j];
            if constexpr (Split) {
                b[q] = *s0.at(j);
                c[q] = Sum ? 0.0F : *s1.at(j);
            } else {
                b[q] = Half ? static_cast<float>(q0[j]) : p0[j];
                c[q] = Sum ? 0.0F : Half ? static_cast<float>(q1[j]) : p1[j];
            }
        }
#pragma unroll
        for (unsigned q = 0; q < kPer; ++q) {
            const unsigned j = t + q * kNormThreads;
            const float x = Sum ? a[q] + b[q] : a[q] + (b[q] + c[q]);
            h[j] = x;
            hs[j] = x;
            acc = __builtin_fmaf(x, x, acc);
        }
    } else {
        for (unsigned j = t; j < kHidden; j += kNormThreads) {
            float x;
            if constexpr (Split) x = Sum ? h[j] + *s0.at(j) : h[j] + (*s0.at(j) + *s1.at(j));
            else {
                const float a0 = Half ? static_cast<float>(q0[j]) : p0[j];
                x = Sum ? h[j] + a0 : h[j] + (a0 + (Half ? static_cast<float>(q1[j]) : p1[j]));
            }
            h[j] = x;
            hs[j] = x;
            acc = __builtin_fmaf(x, x, acc);
        }
    }
    part[t] = acc;
    __syncthreads();
    for (unsigned s = kNormThreads / 2; s >= 1; s >>= 1U) {
        if (t < s) part[t] = part[t] + part[t + s];
        __syncthreads();
    }
    const float rs = 1.0F / __builtin_sqrtf(part[0] / static_cast<float>(kHidden) + kNormEps);
    float* y = v.y + static_cast<std::size_t>(r) * kHidden;
    if constexpr (Ilp == 2) {  // the same expression with the preloaded weights (no load behind a y store)
#pragma unroll
        for (unsigned q = 0; q < kPer; ++q) {
            const unsigned j = t + q * kNormThreads;
            const float o = (hs[j] * rs) * (1.0F + nw[q]);
            y[j] = o;
            ys[j] = o;
        }
    } else {
        for (unsigned j = t; j < kHidden; j += kNormThreads) {
            const float o = (hs[j] * rs) * (1.0F + v.norm_w[j]);
            y[j] = o;
            ys[j] = o;
        }
    }
    if (v.prep_jobs != 0) {  // the next linears' input transform (their prepare launch's function, same bits)
        __syncthreads();
        constexpr unsigned kBlocks = kHidden / 128U;
        for (unsigned q = wave; q < v.prep_jobs * kBlocks; q += kWaves) {
            const unsigned i = q / kBlocks, b = q % kBlocks;
            auto* xt = static_cast<_Float16*>(v.prep_xt[i]) + static_cast<std::size_t>(r) * kHidden;
            opt::dev::transform_block(ys, v.prep_suh[i], xt, b * 128U, lane);
        }
    }
    if (v.router_w == nullptr) return;
    __syncthreads();
    for (unsigned e = wave; e < kExperts; e += kWaves) {
        const float* w = v.router_w + static_cast<std::size_t>(e) * kHidden;
        float a = 0.0F;
        for (unsigned j = lane; j < kHidden; j += kRouterLanes) a = __builtin_fmaf(ys[j], w[j], a);
        for (unsigned d = 16; d >= 1; d >>= 1U) a = a + __shfl_xor(a, static_cast<int>(d), 32);
        if (lane == 0) {
            const float s = 1.0F / (1.0F + msa::exp_f(-a));
            sc[e] = s;
            ch[e] = s + v.bias[e];
        }
    }
    __syncthreads();
    if (t == 0) {
        strix::moe_core::route_topk(StridedRoutePolicy<const float*>{sc, ch, r, v.ids, v.weights});
    }
}

template <unsigned R>
__global__ __launch_bounds__(kNormThreads) void router_batch_kernel(
    std::uint32_t rows, const float* y, const float* router_w, const float* bias,
    std::uint32_t* ids, float* weights) {
    static_assert(R == 4 || R == 8);
    static_assert(kRouterLanes == 32);
    // The R rows' y are staged through LDS in chunks of kChunk inputs (ascending), each wave keeps the accumulators of
    // its kExperts / kWaves experts x R rows in registers across chunks: per (row, expert) lane L still runs
    // a = fmaf(y[j], w[e][j], a) over j = L, L + 32, ... ascending from 0.0F (kChunk % 32 == 0), so the same chain.
    constexpr unsigned kChunk = 512, kPerWave = kExperts / kWaves;
    static_assert(kHidden % kChunk == 0 && kChunk % kRouterLanes == 0 && kExperts % kWaves == 0);
    __shared__ float ys[R][kChunk];
    __shared__ float sc[R][kExperts], ch[R][kExperts];
    const unsigned t = threadIdx.x, lane = t & 31U, wave = t >> 5U;
    const unsigned first = blockIdx.x * R;
    const unsigned valid = rows - first < R ? rows - first : R;  // >= 1 by the grid
    float a[kPerWave][R];
#pragma unroll
    for (unsigned q = 0; q < kPerWave; ++q)
#pragma unroll
        for (unsigned row = 0; row < R; ++row) a[q][row] = 0.0F;
    for (unsigned c0 = 0; c0 < kHidden; c0 += kChunk) {
        __syncthreads();  // the previous chunk's readers are done
        for (unsigned i = t; i < R * kChunk; i += kNormThreads) {
            const unsigned row = i / kChunk, j = i % kChunk;
            ys[row][j] = row < valid ? y[static_cast<std::size_t>(first + row) * kHidden + c0 + j] : 0.0F;
        }
        __syncthreads();
#pragma unroll
        for (unsigned q = 0; q < kPerWave; ++q) {
            const float* w = router_w + static_cast<std::size_t>(wave + q * kWaves) * kHidden + c0;
            for (unsigned j = lane; j < kChunk; j += kRouterLanes) {
                const float wj = w[j];
#pragma unroll
                for (unsigned row = 0; row < R; ++row) a[q][row] = __builtin_fmaf(ys[row][j], wj, a[q][row]);
            }
        }
    }
#pragma unroll
    for (unsigned q = 0; q < kPerWave; ++q) {
        const unsigned e = wave + q * kWaves;
#pragma unroll
        for (unsigned row = 0; row < R; ++row) {
            float x = a[q][row];
            for (unsigned d = 16; d >= 1; d >>= 1U) x = x + __shfl_xor(x, static_cast<int>(d), 32);
            if (lane == 0 && row < valid) {
                const float s = 1.0F / (1.0F + msa::exp_f(-x));
                sc[row][e] = s;
                ch[row][e] = s + bias[e];
            }
        }
    }
    __syncthreads();
    if (t < R && first + t < rows) {
        const unsigned row = t, r = first + row;
        strix::moe_core::route_topk(StridedRoutePolicy<const float*>{sc[row], ch[row], r, ids, weights});
    }
}

// Prefill router, tiled (enqueue_router_tiled): CTA = kRtRows rows x kRtExperts experts. Per (row, expert) lane L still
// runs a = fmaf(y[j], w[e][j], a) over j = L, L + 32, ... ascending from 0.0F (kRtChunk % 32 == 0), then the same xor
// butterfly, sigmoid and bias: the same bits. The j loop is outside the expert loop, so each staged y element is read
// from LDS once per row for the wave's kRtPerWave experts. blockIdx.x = expert group (fastest), so the groups of one row
// tile run together and share its y rows in cache. Scores go to scratch [2][kExperts][rows] (rows adjacent: coalesced
// selection reads); router_select_kernel runs the selection of router_batch_kernel unchanged.
constexpr unsigned kRtRows = 16, kRtExperts = 32, kRtChunk = 256, kRtPerWave = kRtExperts / kWaves;
static_assert(kRtExperts % kWaves == 0 && kExperts % kRtExperts == 0 && kHidden % kRtChunk == 0 && kRtChunk % kRouterLanes == 0);

__global__ __launch_bounds__(kNormThreads) void router_tiled_kernel(std::uint32_t rows, const float* y, const float* router_w,
                                                                     const float* bias, float* scores) {
    static_assert(kRouterLanes == 32);
    __shared__ float ys[kRtRows][kRtChunk];
    const unsigned t = threadIdx.x, lane = t & 31U, wave = t >> 5U;
    const unsigned e0 = blockIdx.x * kRtExperts, first = blockIdx.y * kRtRows;
    const unsigned valid = rows - first < kRtRows ? rows - first : kRtRows;
    float a[kRtPerWave][kRtRows];
#pragma unroll
    for (unsigned q = 0; q < kRtPerWave; ++q)
#pragma unroll
        for (unsigned row = 0; row < kRtRows; ++row) a[q][row] = 0.0F;
    const float* w[kRtPerWave];
#pragma unroll
    for (unsigned q = 0; q < kRtPerWave; ++q) w[q] = router_w + static_cast<std::size_t>(e0 + wave + q * kWaves) * kHidden;
    for (unsigned c0 = 0; c0 < kHidden; c0 += kRtChunk) {
        __syncthreads();
        for (unsigned i = t; i < kRtRows * kRtChunk; i += kNormThreads) {
            const unsigned row = i / kRtChunk, j = i % kRtChunk;
            ys[row][j] = row < valid ? y[static_cast<std::size_t>(first + row) * kHidden + c0 + j] : 0.0F;
        }
        __syncthreads();
        for (unsigned j = lane; j < kRtChunk; j += kRouterLanes) {
            float wj[kRtPerWave];
#pragma unroll
            for (unsigned q = 0; q < kRtPerWave; ++q) wj[q] = w[q][c0 + j];
#pragma unroll
            for (unsigned row = 0; row < kRtRows; ++row) {
                const float yv = ys[row][j];
#pragma unroll
                for (unsigned q = 0; q < kRtPerWave; ++q) a[q][row] = __builtin_fmaf(yv, wj[q], a[q][row]);
            }
        }
    }
#pragma unroll
    for (unsigned q = 0; q < kRtPerWave; ++q) {
        const unsigned e = e0 + wave + q * kWaves;
#pragma unroll
        for (unsigned row = 0; row < kRtRows; ++row) {
            float x = a[q][row];
            for (unsigned d = 16; d >= 1; d >>= 1U) x = x + __shfl_xor(x, static_cast<int>(d), 32);
            if (lane == 0 && row < valid) {
                const float sv = 1.0F / (1.0F + msa::exp_f(-x));
                scores[static_cast<std::size_t>(e) * rows + first + row] = sv;
                scores[(static_cast<std::size_t>(kExperts) + e) * rows + first + row] = sv + bias[e];
            }
        }
    }
}

__global__ __launch_bounds__(256) void router_select_kernel(std::uint32_t rows, const float* scores, std::uint32_t* ids,
                                                            float* weights) {
    const unsigned r = blockIdx.x * 256U + threadIdx.x;
    if (r >= rows) return;
    const RowScores sc{scores + r, rows};
    const RowScores ch{scores + static_cast<std::size_t>(kExperts) * rows + r, rows};
    strix::moe_core::route_topk(StridedRoutePolicy<RowScores>{sc, ch, r, ids, weights});
}

// FAST router product (enqueue_router_wmma) on the shared dense FP16 WMMA core: the sum of (row, expert) goes through
// the sigmoid and bias of router_tiled_kernel into the same scores scratch. The launch's y is the scratch itself, so the
// expert of a column tile is its y offset.
namespace tall = strix::core::f16_dense_tall;
struct RouterScoreStore {
    float* scores;
    const float* bias;
    unsigned rows;
    struct Tile {
        float* s;
        float* c;
        float b;
    };
    __device__ __forceinline__ Tile open(unsigned, const tall::ColumnTile& ct, unsigned c) const {
        const unsigned e = static_cast<unsigned>(ct.y - scores) + c;
        return {scores + static_cast<std::size_t>(e) * rows, scores + (static_cast<std::size_t>(kExperts) + e) * rows, bias[e]};
    }
    __device__ __forceinline__ void put(const Tile& t, unsigned r, float v) const {
        const float sv = 1.0F / (1.0F + msa::exp_f(-v));
        t.s[r] = sv;
        t.c[r] = sv + t.b;
    }
    __device__ __forceinline__ void close() const {}
    template <class F> __device__ __forceinline__ void run(F&& f) { f(*this); }
};

// Rows flagged by the guarded product (non-finite sum: FP16 overflow of y or of a weight) get the exact scores of
// router_tiled_kernel: per (row, expert) lane L runs a = fmaf(y[j], w[e][j], a) over j = L, L + 32, ... from 0.0F, the
// same xor butterfly, sigmoid and bias; FP32 weights.
constexpr unsigned kRouterFallbackSpan = 1024;
__global__ __launch_bounds__(kNormThreads) void router_exact_rows_kernel(std::uint32_t rows, const float* y, const float* router_w,
                                                                         const float* bias, float* scores, std::uint32_t* flags) {
    __shared__ unsigned list[kNormThreads];
    __shared__ unsigned count;
    __shared__ float ys[kHidden];
    tall::exact_rows(flags, rows, kRouterFallbackSpan, list, &count, [&](unsigned r) __attribute__((always_inline)) {
        for (unsigned j = threadIdx.x; j < kHidden; j += kNormThreads) ys[j] = y[static_cast<std::size_t>(r) * kHidden + j];
        __syncthreads();
        const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U;
        for (unsigned e = wave; e < kExperts; e += kWaves) {
            const float* w = router_w + static_cast<std::size_t>(e) * kHidden;
            float a = 0.0F;
#pragma unroll 8  // the weight loads issue ahead; the chain order is unchanged
            for (unsigned j = lane; j < kHidden; j += kRouterLanes) a = __builtin_fmaf(ys[j], w[j], a);
            for (unsigned d = 16; d >= 1; d >>= 1U) a = a + __shfl_xor(a, static_cast<int>(d), 32);
            if (lane == 0) {
                const float sv = 1.0F / (1.0F + msa::exp_f(-a));
                scores[static_cast<std::size_t>(e) * rows + r] = sv;
                scores[(static_cast<std::size_t>(kExperts) + e) * rows + r] = sv + bias[e];
            }
        }
    });
}

__global__ __launch_bounds__(256) void router_f16_kernel(const float* w, _Float16* out, unsigned n) {
    const unsigned i = blockIdx.x * 256U + threadIdx.x;
    if (i < n) out[i] = static_cast<_Float16>(w[i]);
}

}  // namespace

void enqueue_post_exchange(const PostExchangeView& v, hipStream_t stream) {
    if (v.ilp) {  // the loads-first form needs h disjoint from both partials (it reads them all before writing h)
        const auto* hb = reinterpret_cast<const char*>(v.h);
        const auto* he = hb + std::size_t{v.rows} * kHidden * sizeof(float);
        for (const float* p : {v.part0, v.part1}) {
            if (p == nullptr) continue;  // folded exchange: one slot
            const auto* pb = reinterpret_cast<const char*>(p);
            const auto* pe = pb + std::size_t{v.rows} * kHidden * sizeof(float);
            if (pb < he && hb < pe) throw std::invalid_argument("minimax post-exchange: ilp form with h overlapping a partial");
        }
    }
    if (v.ilp == 2 && ((reinterpret_cast<std::uintptr_t>(v.part0) | reinterpret_cast<std::uintptr_t>(v.part1)) & 15U) != 0)
        throw std::invalid_argument("minimax post-exchange: ilp mode 2 needs 16-byte aligned partials");
    if (v.half_parts && v.split_head != 0) throw std::invalid_argument("minimax post-exchange: f16 partials cannot use split frames");
    if (v.half_parts) {
        if (v.part0 == nullptr || v.part1 == nullptr) throw std::invalid_argument("minimax post-exchange: f16 partials need both slots");
        if (v.ilp == 2) hipLaunchKernelGGL((post_exchange_kernel<2, false, false, true>), dim3(v.rows), dim3(kNormThreads), 0, stream, v);
        else if (v.ilp == 1) hipLaunchKernelGGL((post_exchange_kernel<1, false, false, true>), dim3(v.rows), dim3(kNormThreads), 0, stream, v);
        else hipLaunchKernelGGL((post_exchange_kernel<0, false, false, true>), dim3(v.rows), dim3(kNormThreads), 0, stream, v);
    } else if (v.split_head != 0) {  // split-producer chunk frame (opt-in): the same kernels with chunk-major partial addressing
        if (v.split_head % 4U != 0 || v.split_head >= kHidden) throw std::invalid_argument("minimax post-exchange: bad split head");
        if (v.part1 == nullptr) {
            if (v.ilp == 2) hipLaunchKernelGGL((post_exchange_kernel<2, true, true>), dim3(v.rows), dim3(kNormThreads), 0, stream, v);
            else if (v.ilp == 1) hipLaunchKernelGGL((post_exchange_kernel<1, true, true>), dim3(v.rows), dim3(kNormThreads), 0, stream, v);
            else hipLaunchKernelGGL((post_exchange_kernel<0, true, true>), dim3(v.rows), dim3(kNormThreads), 0, stream, v);
        } else if (v.ilp == 2) hipLaunchKernelGGL((post_exchange_kernel<2, false, true>), dim3(v.rows), dim3(kNormThreads), 0, stream, v);
        else if (v.ilp == 1) hipLaunchKernelGGL((post_exchange_kernel<1, false, true>), dim3(v.rows), dim3(kNormThreads), 0, stream, v);
        else hipLaunchKernelGGL((post_exchange_kernel<0, false, true>), dim3(v.rows), dim3(kNormThreads), 0, stream, v);
    } else if (v.part1 == nullptr) {  // folded exchange: part0 is the pair sum
        if (v.ilp == 2) hipLaunchKernelGGL((post_exchange_kernel<2, true>), dim3(v.rows), dim3(kNormThreads), 0, stream, v);
        else if (v.ilp == 1) hipLaunchKernelGGL((post_exchange_kernel<1, true>), dim3(v.rows), dim3(kNormThreads), 0, stream, v);
        else hipLaunchKernelGGL((post_exchange_kernel<0, true>), dim3(v.rows), dim3(kNormThreads), 0, stream, v);
    } else if (v.ilp == 2) hipLaunchKernelGGL(post_exchange_kernel<2>, dim3(v.rows), dim3(kNormThreads), 0, stream, v);
    else if (v.ilp == 1) hipLaunchKernelGGL(post_exchange_kernel<1>, dim3(v.rows), dim3(kNormThreads), 0, stream, v);
    else hipLaunchKernelGGL(post_exchange_kernel<0>, dim3(v.rows), dim3(kNormThreads), 0, stream, v);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("minimax post-exchange: ") + hipGetErrorString(e));
}

void enqueue_router_batch(std::uint32_t rows, const float* y, const float* router_w, const float* bias,
                          std::uint32_t* ids, float* weights, unsigned batch_rows, hipStream_t stream) {
    if (batch_rows != 4 && batch_rows != 8)
        throw std::invalid_argument("minimax router batch: rows must be 4 or 8");
    if (rows == 0) return;
    const dim3 grid(rows / batch_rows + (rows % batch_rows != 0));
    if (batch_rows == 4)
        hipLaunchKernelGGL(router_batch_kernel<4>, grid, dim3(kNormThreads), 0, stream, rows, y, router_w, bias, ids, weights);
    else
        hipLaunchKernelGGL(router_batch_kernel<8>, grid, dim3(kNormThreads), 0, stream, rows, y, router_w, bias, ids, weights);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("minimax router batch: ") + hipGetErrorString(e));
}

void enqueue_router_tiled(std::uint32_t rows, const float* y, const float* router_w, const float* bias,
                          std::uint32_t* ids, float* weights, float* scratch, hipStream_t stream) {
    if (rows == 0) return;
    if (scratch == nullptr) throw std::invalid_argument("minimax router tiled: scratch is required");
    const dim3 grid(kExperts / kRtExperts, rows / kRtRows + (rows % kRtRows != 0));
    hipLaunchKernelGGL(router_tiled_kernel, grid, dim3(kNormThreads), 0, stream, rows, y, router_w, bias, scratch);
    hipLaunchKernelGGL(router_select_kernel, dim3((rows + 255) / 256), dim3(256), 0, stream, rows, scratch, ids, weights);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("minimax router tiled: ") + hipGetErrorString(e));
}

void convert_router_f16(const float* router_w, std::uint16_t* w16, hipStream_t stream) {
    constexpr unsigned n = kExperts * kHidden;
    hipLaunchKernelGGL(router_f16_kernel, dim3((n + 255U) / 256U), dim3(256), 0, stream, router_w, reinterpret_cast<_Float16*>(w16), n);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("minimax router f16: ") + hipGetErrorString(e));
}

// MiniMax router: 8 column tiles of 6,144-long chains; the product is bound by the reads of y (25 KB per row). Automatic
// form by piece rows (u2 unit receipts, gfx1151): up to 1,536 rows y stays in the 32 MB MALL and narrow4 (two 4-tile
// column blocks sharing each y row) wins (1024: 0.183 vs wide16 0.415 ms); above it wide16 reads y once
// (2048: 0.858 vs narrow4 1.101; 4096: 1.274 vs 1.710 ms). STRIX_F16_MATRIX_FORM overrides.
constexpr unsigned kRouterWideRows = 1536;
static tall::MatrixForm router_form(unsigned form, unsigned rows) {
    if (form <= 2U) return static_cast<tall::MatrixForm>(form);
    return tall::matrix_form(rows <= kRouterWideRows ? tall::MatrixForm::narrow4 : tall::MatrixForm::wide16);
}
const char* router_wmma_form() {
    const tall::MatrixForm a = router_form(~0U, 0), b = router_form(~0U, ~0U);
    return a == b ? tall::matrix_form_name(a) : "auto (narrow4 <= 1536 rows < wide16)";
}

void enqueue_router_wmma(std::uint32_t rows, const float* y, const float* router_w, const std::uint16_t* w16, const float* bias,
                         std::uint32_t* ids, float* weights, float* scratch, std::uint32_t* flags, hipStream_t stream, unsigned form) {
    static_assert(kHidden % 128U == 0 && kExperts % 16U == 0);
    if (rows == 0) return;
    if (scratch == nullptr || flags == nullptr || w16 == nullptr) throw std::invalid_argument("minimax router wmma: scratch, flags and f16 weights are required");
    if (std::uint64_t{rows} * kHidden * 4U >= (std::uint64_t{1} << 32U)) throw std::invalid_argument("minimax router wmma: too many rows");
    const tall::Guarded<RouterScoreStore> store{{scratch, bias, rows}, flags};
    tall::launch_matrix(router_form(form, rows), y, rows, reinterpret_cast<const _Float16*>(w16), kHidden, kExperts, scratch, rows, store, stream);
    hipLaunchKernelGGL(router_exact_rows_kernel, dim3((rows + kRouterFallbackSpan - 1U) / kRouterFallbackSpan), dim3(kNormThreads), 0, stream,
                       rows, y, router_w, bias, scratch, flags);
    hipLaunchKernelGGL(router_select_kernel, dim3((rows + 255) / 256), dim3(256), 0, stream, rows, scratch, ids, weights);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("minimax router wmma: ") + hipGetErrorString(e));
}

}  // namespace strix::minimax::moe::hip
