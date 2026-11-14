#include "moe_core/decode.hpp"
// GLM-5.3-Flash decode MoE (identity and launch structure: glm_moe.hpp).
#include "glm/moe/glm_moe.hpp"

#include "glm/common/glm_math.hpp"
#include "glm/moe/glm_trellis_device.hpp"
#ifndef STRIX_GLM_FAST_I4_CORE
#define STRIX_GLM_FAST_I4_CORE 0
#endif
#if STRIX_GLM_FAST_I4_CORE
#include "core/w4a16.hpp"  // strix-trellis fast/i4-w4a16 src/core (shared stored INT4 core)
#endif

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace strix::glm::moe {
namespace {

namespace d = dev;

constexpr unsigned kGateSplits = d::vector_splits(kHidden, kInter);  // 32
constexpr unsigned kDownSplits = d::vector_splits(kInter, kHidden);  // 4
static_assert(kGateSplits == 32 && kDownSplits == 4, "strix-trellis 5acf090 split policy for the GLM expert shapes");
constexpr unsigned kGateKt = kHidden / 16, kDownKt = kInter / 16;                       // 256, 64 k-tiles
constexpr unsigned kGateKps = kGateKt / kGateSplits, kDownKps = kDownKt / kDownSplits;  // 8, 16 k-tiles per split
constexpr unsigned kSplitCh = kGateKps * 16;     // 128 channels = one H128 block per gate/up split
constexpr unsigned kInterBlocks = kInter / 128;  // 8 gate/up output blocks
constexpr unsigned kGuGroups = kInter / 256;     // 4 gate/up CTAs (16 columns) per split group
constexpr unsigned kDnGroups = kHidden / 256;    // 16 down CTAs (16 columns, 2 output blocks) per slot
constexpr unsigned kMaskWords = (kExperts + 31) / 32;
// Router: 8 experts per CTA; the F16 rows (row stride padded by 16 bytes) and the y rows are both staged through LDS
// in 256-channel chunks, double-buffered (measurement: with y read from global inside the chain, launch 1 took ~250 us).
constexpr unsigned kRouterEpc = 8, kRouterCtas = kExperts / kRouterEpc, kRouterChunk = 256, kRouterStride = kRouterChunk + 8;
static_assert(kExperts % kRouterEpc == 0 && kHidden % kRouterChunk == 0 && kExperts == 9 * 32);
constexpr std::size_t kRouterWBytes = std::size_t{2} * kRouterEpc * kRouterStride * 2;
constexpr std::size_t kRouterSmem = kRouterWBytes + std::size_t{2} * kMaxRows * kRouterChunk * 4;

constexpr std::size_t align256(std::size_t v) { return (v + 255) & ~std::size_t{255}; }
constexpr std::size_t kOffAt = 0;  // per pair: the transformed down input, kInter FP16
constexpr std::size_t kOffY = kOffAt + align256(std::size_t{kMaxPairs} * kInter * 2);  // per pair: o_e, kHidden FP32
constexpr std::size_t kOffPart = kOffY + align256(std::size_t{kMaxPairs} * kHidden * 4);  // gate/up split partials
constexpr std::size_t kOffLogits = kOffPart + align256(std::size_t{kMaxPairs} * 2 * kGateSplits * kInter * 4);
constexpr std::size_t kOffCount = kOffLogits + align256(std::size_t{kMaxRows} * kExperts * 4);
constexpr unsigned kCounters = kMaxSlots * kInterBlocks + kDnGroups + 1;
constexpr std::size_t kMoeBytes = kOffCount + align256(std::size_t{kCounters} * 4);

struct Args {
    unsigned rows;
    const float* y;
    const std::uint16_t* router_w;
    const float* bias;
    std::uint32_t* ids;
    float* weights;
    float* logits;
    const ExpertSlices* experts;
    ExpertSlices shared;
    float* out;
    unsigned char* ws;
    float* dbg_gu;  // MoeView::debug_gate_up
    I4Experts i4;   // MoeView::i4 (fast_i4: routed slots on the W4A16 core; read only by the I4 kernel instantiations)
    strix::pair::ProducerPublish pub;  // MoeView::publish (down_kernel; the core path publishes with one more launch)
    __device__ _Float16* at(unsigned pair) const {
        return reinterpret_cast<_Float16*>(ws + kOffAt) + static_cast<std::size_t>(pair) * kInter;
    }
    __device__ float* yslot(unsigned pair) const {
        return reinterpret_cast<float*>(ws + kOffY) + static_cast<std::size_t>(pair) * kHidden;
    }
    // Raw library split partials: [pair][matrix 0 gate / 1 up][split][kInter].
    __device__ float* part(unsigned pair, unsigned m, unsigned split) const {
        return reinterpret_cast<float*>(ws + kOffPart) + ((static_cast<std::size_t>(pair) * 2 + m) * kGateSplits + split) * kInter;
    }
    __device__ std::uint32_t* counters() const { return reinterpret_cast<std::uint32_t*>(ws + kOffCount); }
    __device__ std::uint32_t* gu_counter(unsigned slot, unsigned block) const { return counters() + slot * kInterBlocks + block; }
    __device__ std::uint32_t* dn_counter(unsigned group) const { return counters() + kMaxSlots * kInterBlocks + group; }
    __device__ std::uint32_t* router_counter() const { return counters() + kMaxSlots * kInterBlocks + kDnGroups; }
};

__device__ __forceinline__ float load_agent(const float* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// True in every thread of the CTA that arrives last at `counter` (expected arrivals); that CTA resets the counter.
__device__ __forceinline__ bool last_arrival(std::uint32_t* counter, unsigned expected) {
    __shared__ unsigned last;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned old = atomicAdd(counter, 1U);
        last = old + 1U == expected ? 1U : 0U;
        if (last) *counter = 0;
    }
    __syncthreads();
    if (last) __threadfence();
    return last != 0;
}

// ---------------------------------------------------------------- dispatch, derived per CTA from the router ids
// Slot 0 = shared expert (rows 0..R-1, pairs 0..R-1). Slot s >= 1 = the (s-1)-th smallest distinct routed expert e;
// its rows ascending; its pairs R + #(ids < e) + (rank of the row among e's rows). Pair (r, k) of row r's k-th
// expert (ascending id) follows from the same rule (down_kernel).
struct SlotInfo {
    unsigned count;   // distinct routed experts of the call (valid when ids were read)
    unsigned active;  // slot <= count
    unsigned expert;  // routed expert id (slot >= 1)
    unsigned nrows, first;
    unsigned row[kMaxRows];
};
// use_ids = false: only for slot 0 (the shared expert in the router launch, before the ids exist).
__device__ __forceinline__ void slot_info(const Args& a, unsigned slot, bool use_ids, SlotInfo& si, std::uint32_t* ids /* LDS, 64 */,
                          std::uint32_t* mask /* LDS, kMaskWords */) {
    const unsigned t = threadIdx.x, n = a.rows * kTopE;
    if (use_ids) {
        if (t < kMaskWords) mask[t] = 0;
        if (t < n) ids[t] = a.ids[t];
        __syncthreads();
        if (t < n) atomicOr(&mask[ids[t] >> 5U], 1U << (ids[t] & 31U));
        __syncthreads();
    }
    if (t == 0) {
        unsigned count = 0, e = kExperts;
        if (use_ids)
            for (unsigned w = 0; w < kMaskWords; ++w) {
                const unsigned bits = mask[w], c = static_cast<unsigned>(__builtin_popcount(bits));
                if (slot >= 1 && e == kExperts && slot - 1 < count + c) {  // the (slot - 1 - count)-th set bit
                    unsigned b = bits;
                    for (unsigned j = slot - 1 - count; j > 0; --j) b &= b - 1U;
                    e = w * 32U + static_cast<unsigned>(__builtin_ctz(b));
                }
                count += c;
            }
        si.count = count;
        si.active = (slot == 0 || slot <= count) ? 1U : 0U;
        si.expert = e;
        if (slot == 0) {
            si.nrows = a.rows;
            si.first = 0;
            for (unsigned r = 0; r < a.rows; ++r) si.row[r] = r;
        } else if (si.active) {
            unsigned nr = 0, below = 0;
            for (unsigned j = 0; j < n; ++j) {
                below += ids[j] < e ? 1U : 0U;
                if (ids[j] == e) si.row[nr++] = j / kTopE;
            }
            si.nrows = nr;
            si.first = a.rows + below;
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------- wide core (MiniMax wide form, mmx_moe.hip)
// Wave = 4 adjacent tile columns; per k it reads their contiguous run (4 x Bits x 32 bytes) with 16-byte loads
// through a register ring of kWideDepth k-tiles and a wave-private LDS row (no barrier), and reuses one activation
// read per k for the 4 columns. Library split boundaries every KPS k-tiles from k0 (a split start):
//   Store: the split partial reduce4(chain) goes to part0 + r * row_stride + split * kInter (raw, as the library);
//   else : fin[r][column-local output] = (first split ? +0 + p : fin + p), the output transform's ascending sum.
// Rows: static bucket R; rows past nrows read finite or stale LDS inputs and their results are never stored.
constexpr unsigned kWideDepth = 4;
constexpr unsigned kRowbufU = 8 * 6;  // uint4 per wave row buffer (Bits <= 6)
using WideV4 = std::uint32_t __attribute__((ext_vector_type(4)));
using GlobalWideV4 = const WideV4 __attribute__((address_space(1)));
// K6's 48 vectors round up to 64 so every lane stores both vectors unconditionally.
template <bool Pipe> constexpr unsigned kRowbufStride = Pipe ? 64U : kRowbufU;
template <int Bits, int R, unsigned KPS, bool Store>
__device__ __forceinline__ void wide_core(const Exl3 w, unsigned output, unsigned k0, unsigned k1, unsigned col0,
                                          const _Float16* xs, unsigned xs_stride, unsigned nrows, unsigned wave,
                                          unsigned lane, WideV4* rowbuf_all, float* part0, std::size_t row_stride,
                                          float* fin, unsigned fin_stride, unsigned fin_col0) {
    constexpr unsigned runU = 8U * Bits, UPL = (runU + 31U) / 32U;
    static_assert(runU <= kRowbufU && KPS % kWideDepth == 0, "row buffer size; split boundaries on ring boundaries");
    WideV4* rowbuf = rowbuf_all + wave * kRowbufU;
    const unsigned cols = output / 16U;
    const auto* gw = reinterpret_cast<const WideV4*>(d::uniform(w.trellis));
    const std::size_t kstride = static_cast<std::size_t>(cols) * 2U * Bits, cbase = static_cast<std::size_t>(col0) * 2U * Bits;
    WideV4 ring[kWideDepth][UPL] = {};
    auto load = [&](unsigned k, auto slot) __attribute__((always_inline)) {
        constexpr unsigned q = decltype(slot)::value;
#pragma unroll
        for (unsigned u = 0; u < UPL; ++u) {
            const unsigned e = lane + 32U * u;
            if (e < runU) ring[q][u] = gw[static_cast<std::size_t>(k) * kstride + cbase + e];
        }
    };
    const d::LaneGeom<Bits> geo(lane);
    float sums[4][R][2];
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int r = 0; r < R; ++r) sums[c][r][0] = sums[c][r][1] = 0.0F;
    const auto* rl = (d::LocalU32*)(reinterpret_cast<const std::uint32_t*>(rowbuf));
    const auto* xw = (d::LocalU32*)(reinterpret_cast<const std::uint32_t*>(xs)) + (lane & 3U);
    auto split_end = [&](unsigned split) __attribute__((always_inline)) {
        float* const ps = Store ? part0 + static_cast<std::size_t>(split) * kInter : nullptr;  // wave-uniform
#pragma unroll
        for (unsigned c = 0; c < 4; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float v0 = d::reduce4(sums[c][r][0]), v1 = d::reduce4(sums[c][r][1]);
                if ((lane & 3U) == 0 && static_cast<unsigned>(r) < nrows) {
                    if constexpr (Store) {
                        float* pr = ps + static_cast<std::size_t>(r) * row_stride + (col0 + c) * 16 + lane / 4;
                        pr[0] = v0;
                        pr[8] = v1;
                    } else {
                        float* pf = fin + static_cast<unsigned>(r) * fin_stride + (fin_col0 + c) * 16 + lane / 4;
                        pf[0] = split == 0 ? 0.0F + v0 : pf[0] + v0;
                        pf[8] = split == 0 ? 0.0F + v1 : pf[8] + v1;
                    }
                }
                sums[c][r][0] = sums[c][r][1] = 0.0F;
            }
    };
    auto step = [&](unsigned k, auto slot) __attribute__((always_inline)) {
        constexpr unsigned q = decltype(slot)::value;
#pragma unroll
        for (unsigned u = 0; u < UPL; ++u) {
            const unsigned e = lane + 32U * u;
            if (e < runU) rowbuf[e] = ring[q][u];
        }
        if (k + kWideDepth < k1) load(k + kWideDepth, slot);
        d::Half2 act[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const auto* pr = xw + (static_cast<unsigned>(r) * xs_stride) / 2U + (k - k0) * 8U;
            act[r][0] = __builtin_bit_cast(d::Half2, pr[0]);
            act[r][1] = __builtin_bit_cast(d::Half2, pr[4]);
        }
#pragma unroll
        for (unsigned c = 0; c < 4; ++c) {
            d::Slot<Bits, R> cur;
            const unsigned tw = c * Bits * 8U;
            cur.a = rl[tw + geo.oa / 4U];
            cur.b = rl[tw + geo.ob / 4U];
            cur.c = rl[tw + geo.oc / 4U];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                cur.act[r][0] = act[r][0];
                cur.act[r][1] = act[r][1];
            }
            d::compute_slot<Bits, R>(cur, geo, static_cast<unsigned>(R), sums[c]);
        }
    };
    using S0 = std::integral_constant<unsigned, 0>;
    using S1 = std::integral_constant<unsigned, 1>;
    using S2 = std::integral_constant<unsigned, 2>;
    using S3 = std::integral_constant<unsigned, 3>;
    load(k0 + 0, S0{});
    load(k0 + 1, S1{});
    load(k0 + 2, S2{});
    load(k0 + 3, S3{});
    for (unsigned split = 0; split < (k1 - k0) / KPS; ++split) {
        for (unsigned kk = 0; kk < KPS; kk += kWideDepth) {
            const unsigned k = k0 + split * KPS + kk;
            step(k, S0{});
            step(k + 1, S1{});
            step(k + 2, S2{});
            step(k + 3, S3{});
        }
        split_end(split);
    }
}
// Opt-in scheduling variant: arithmetic below is copied verbatim from wide_core.
template <int Bits, int R, unsigned KPS, bool Store>
__device__ __forceinline__ void wide_core_pipe(const Exl3 w, unsigned output, unsigned k0, unsigned k1, unsigned col0,
                                          const _Float16* xs, unsigned xs_stride, unsigned nrows, unsigned wave,
                                          unsigned lane, WideV4* rowbuf_all, float* part0, std::size_t row_stride,
                                          float* fin, unsigned fin_stride, unsigned fin_col0) {
    constexpr unsigned runU = 8U * Bits, UPL = (runU + 31U) / 32U;
    static_assert(runU <= kRowbufU && KPS % kWideDepth == 0, "row buffer size; split boundaries on ring boundaries");
    static_assert(UPL * 32U <= kRowbufStride<true>, "unconditional padded LDS stores stay wave-private");
    WideV4* rowbuf = rowbuf_all + wave * kRowbufStride<true>;
    const unsigned cols = output / 16U;
    const auto* gw = (GlobalWideV4*)(reinterpret_cast<const WideV4*>(d::uniform(w.trellis)));
    const std::size_t kstride = static_cast<std::size_t>(cols) * 2U * Bits, cbase = static_cast<std::size_t>(col0) * 2U * Bits;
    WideV4 ring[kWideDepth][UPL] = {};
    auto load = [&](unsigned k, auto slot) __attribute__((always_inline)) {
        constexpr unsigned q = decltype(slot)::value;
#pragma unroll
        for (unsigned u = 0; u < UPL; ++u) {
            const unsigned e = lane + 32U * u;
            ring[q][u] = gw[static_cast<std::size_t>(k) * kstride + cbase + (e < runU ? e : runU - 1U)];
        }
    };
    const d::LaneGeom<Bits> geo(lane);
    float sums[4][R][2];
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int r = 0; r < R; ++r) sums[c][r][0] = sums[c][r][1] = 0.0F;
    const auto* rl = (d::LocalU32*)(reinterpret_cast<const std::uint32_t*>(rowbuf));
    const auto* xw = (d::LocalU32*)(reinterpret_cast<const std::uint32_t*>(xs)) + (lane & 3U);
    auto split_end = [&](unsigned split) __attribute__((always_inline)) {
        float* const ps = Store ? part0 + static_cast<std::size_t>(split) * kInter : nullptr;  // wave-uniform
#pragma unroll
        for (unsigned c = 0; c < 4; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float v0 = d::reduce4(sums[c][r][0]), v1 = d::reduce4(sums[c][r][1]);
                if ((lane & 3U) == 0 && static_cast<unsigned>(r) < nrows) {
                    if constexpr (Store) {
                        float* pr = ps + static_cast<std::size_t>(r) * row_stride + (col0 + c) * 16 + lane / 4;
                        pr[0] = v0;
                        pr[8] = v1;
                    } else {
                        float* pf = fin + static_cast<unsigned>(r) * fin_stride + (fin_col0 + c) * 16 + lane / 4;
                        pf[0] = split == 0 ? 0.0F + v0 : pf[0] + v0;
                        pf[8] = split == 0 ? 0.0F + v1 : pf[8] + v1;
                    }
                }
                sums[c][r][0] = sums[c][r][1] = 0.0F;
            }
    };
    auto step = [&](unsigned k, auto slot, auto with_load) __attribute__((always_inline)) {
        constexpr unsigned q = decltype(slot)::value;
#pragma unroll
        for (unsigned u = 0; u < UPL; ++u) {
            const unsigned e = lane + 32U * u;
            rowbuf[e] = ring[q][u];
        }
        if constexpr (decltype(with_load)::value) load(k + kWideDepth, slot);
        d::Half2 act[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const auto* pr = xw + (static_cast<unsigned>(r) * xs_stride) / 2U + (k - k0) * 8U;
            act[r][0] = __builtin_bit_cast(d::Half2, pr[0]);
            act[r][1] = __builtin_bit_cast(d::Half2, pr[4]);
        }
#pragma unroll
        for (unsigned c = 0; c < 4; ++c) {
            d::Slot<Bits, R> cur;
            const unsigned tw = c * Bits * 8U;
            cur.a = rl[tw + geo.oa / 4U];
            cur.b = rl[tw + geo.ob / 4U];
            cur.c = rl[tw + geo.oc / 4U];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                cur.act[r][0] = act[r][0];
                cur.act[r][1] = act[r][1];
            }
            d::compute_slot<Bits, R>(cur, geo, static_cast<unsigned>(R), sums[c]);
        }
        __builtin_amdgcn_sched_barrier(0);
    };
    using S0 = std::integral_constant<unsigned, 0>;
    using S1 = std::integral_constant<unsigned, 1>;
    using S2 = std::integral_constant<unsigned, 2>;
    using S3 = std::integral_constant<unsigned, 3>;
    __builtin_amdgcn_sched_barrier(0);
    load(k0 + 0, S0{});
    __builtin_amdgcn_sched_barrier(0);
    load(k0 + 1, S1{});
    __builtin_amdgcn_sched_barrier(0);
    load(k0 + 2, S2{});
    __builtin_amdgcn_sched_barrier(0);
    load(k0 + 3, S3{});
    __builtin_amdgcn_sched_barrier(0);
    // Identical k order and split ends. Peel the final ring group: no conditional refill.
    const unsigned nk = k1 - k0;
    for (unsigned kk = 0; kk + kWideDepth < nk; kk += kWideDepth) {
        const unsigned k = k0 + kk;
        step(k, S0{}, std::true_type{});
        step(k + 1, S1{}, std::true_type{});
        step(k + 2, S2{}, std::true_type{});
        step(k + 3, S3{}, std::true_type{});
        if ((kk + kWideDepth) % KPS == 0) split_end((kk + kWideDepth) / KPS - 1U);
    }
    const unsigned k = k1 - kWideDepth;
    step(k, S0{}, std::false_type{});
    step(k + 1, S1{}, std::false_type{});
    step(k + 2, S2{}, std::false_type{});
    step(k + 3, S3{}, std::false_type{});
    split_end(nk / KPS - 1U);
}
template <int MaxR, unsigned KPS, bool Store, bool Pipe>
__device__ __forceinline__ void wide_rows(const Exl3 w, unsigned output, unsigned k0, unsigned k1, unsigned col0, const _Float16* xs,
                          unsigned xs_stride, unsigned nrows, unsigned wave, unsigned lane, WideV4* rowbuf, float* part0,
                          std::size_t row_stride, float* fin, unsigned fin_stride, unsigned fin_col0) {
    auto run = [&](auto bits_tag) {
        constexpr int Bits = decltype(bits_tag)::value;
#define GLM_WIDE(R) core(std::integral_constant<int, R>{})
        auto core = [&](auto rows_tag) {
            constexpr int R = decltype(rows_tag)::value;
            if constexpr (Pipe) wide_core_pipe<Bits, R, KPS, Store>(w, output, k0, k1, col0, xs, xs_stride, nrows, wave, lane, rowbuf, part0,
                                                                  row_stride, fin, fin_stride, fin_col0);
            else wide_core<Bits, R, KPS, Store>(w, output, k0, k1, col0, xs, xs_stride, nrows, wave, lane, rowbuf, part0,
                                                               row_stride, fin, fin_stride, fin_col0);
        };
        if (nrows == 1 || MaxR == 1) GLM_WIDE(1);
        else if (nrows == 2 || MaxR == 2) GLM_WIDE(2);
        else if constexpr (MaxR >= 4) {
            if (nrows <= 4 || MaxR == 4) GLM_WIDE(4);
            else GLM_WIDE(MaxR);
        }
#undef GLM_WIDE
    };
    if (w.bits == 6) run(std::integral_constant<int, 6>{});
    else run(std::integral_constant<int, 4>{});
}

__device__ __forceinline__ float swiglu(float g, float u) {  // R15, as glm_tier_a.hip swiglu_kernel
    const float gg = g > kClamp ? kClamp : g;
    const float uu = u > kClamp ? kClamp : (u < -kClamp ? -kClamp : u);
    return silu(gg) * uu;
}

// By value: a reference that may point into the kernel argument would force Args onto the stack (scratch).
__device__ __forceinline__ Exl3 slot_matrix(const Args& a, const SlotInfo& si, unsigned slot, unsigned m /* 0 gate, 1 up, 2 down */) {
    if (slot == 0) return m == 0 ? a.shared.gate : (m == 1 ? a.shared.up : a.shared.down);
    const ExpertSlices* e = a.experts + si.expert;
    return m == 0 ? e->gate : (m == 1 ? e->up : e->down);
}

struct DecodeTraits {
    using Schedule = strix::moe_core::DecodeSchedule<strix::glm::moe::kGateSplits, strix::glm::moe::kDownSplits, strix::glm::moe::kSplitCh, 4U>;
    using Args = strix::glm::moe::Args;
    using Dispatch = strix::glm::moe::SlotInfo;
    using ExpertSlices = strix::glm::moe::ExpertSlices;
    using Exl3 = strix::glm::moe::Exl3;
    using WideV4 = strix::glm::moe::WideV4;
    using SlotInfo = strix::glm::moe::SlotInfo;
    static constexpr auto kGateSplits = strix::glm::moe::kGateSplits;
    [[maybe_unused]] static constexpr auto kDownSplits = strix::glm::moe::kDownSplits;
    static constexpr auto kHidden = strix::glm::moe::kHidden;
    static constexpr auto kInter = strix::glm::moe::kInter;
    static constexpr auto kSplitCh = strix::glm::moe::kSplitCh;
    static constexpr auto kGateKps = strix::glm::moe::kGateKps;
    static constexpr auto kDownKps = strix::glm::moe::kDownKps;
    static constexpr auto kDownKt = strix::glm::moe::kDownKt;
    static constexpr auto kTopE = strix::glm::moe::kTopE;
    template <class... T> static __device__ __forceinline__ decltype(auto) last_arrival(T&&... values) {
        return strix::glm::moe::last_arrival(static_cast<T&&>(values)...);
    }
    template <class... T> static __device__ __forceinline__ decltype(auto) load_agent(T&&... values) {
        return strix::glm::moe::load_agent(static_cast<T&&>(values)...);
    }
    template <class... T> static __device__ __forceinline__ decltype(auto) swiglu(T&&... values) {
        return strix::glm::moe::swiglu(static_cast<T&&>(values)...);
    }
    template <class... T> static __device__ __forceinline__ decltype(auto) slot_matrix(T&&... values) {
        return strix::glm::moe::slot_matrix(static_cast<T&&>(values)...);
    }
    template <class... T> static __device__ __forceinline__ decltype(auto) slot_info(T&&... values) {
        return strix::glm::moe::slot_info(static_cast<T&&>(values)...);
    }
    template <auto A, auto B, auto C, auto D, class... T> static __device__ __forceinline__ decltype(auto) wide_rows(T&&... values) {
        return strix::glm::moe::wide_rows<A, B, C, D>(static_cast<T&&>(values)...);
    }
    template <class... T> static __device__ __forceinline__ decltype(auto) transform_block_at(T&&... values) {
        return d::transform_block_at(static_cast<T&&>(values)...);
    }
    template <class... T> static __device__ __forceinline__ decltype(auto) hadamard128(T&&... values) {
        return d::hadamard128(static_cast<T&&>(values)...);
    }
    template <class... T> static __device__ __forceinline__ decltype(auto) half_bits(T&&... values) {
        return d::half_bits(static_cast<T&&>(values)...);
    }
    template <class... T> static __device__ __forceinline__ decltype(auto) materialize(T&&... values) {
        return d::materialize(static_cast<T&&>(values)...);
    }
};
using DecodeCore = strix::moe_core::Decode<DecodeTraits>;

// ---------------------------------------------------------------- gate + up (+ R15 + down input transform)
// CTA = 4 waves x 4 columns (2 output blocks) of matrix m over SG library splits starting at split sg * SG.
template <int MaxR, unsigned SG, bool Pipe> constexpr std::size_t kGuSmem = std::size_t{MaxR} * SG * kSplitCh * 2 + 4 * kRowbufStride<Pipe> * 16;
template <int MaxR, unsigned SG, bool Pipe, bool I4 = false>
__device__ __forceinline__ void gate_up_body(const Args& a, const SlotInfo& si, unsigned slot, unsigned cg, unsigned sg, unsigned m,
                             unsigned char* smem) {
    auto* xs = reinterpret_cast<_Float16*>(smem);
    auto* rowbuf = reinterpret_cast<WideV4*>(smem + std::size_t{MaxR} * SG * kSplitCh * 2);
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const Exl3 w = slot_matrix(a, si, slot, m);
    const unsigned nrows = si.nrows, first = si.first;
    for (unsigned job = wave; job < nrows * SG; job += 4) {  // input transform of the CTA's SG H128 blocks per row
        const unsigned lr = job / SG, b = job % SG;
        d::transform_block_at(a.y + static_cast<std::size_t>(si.row[lr]) * kHidden, w.suh, xs + lr * SG * kSplitCh,
                              (sg * SG + b) * kSplitCh, b * kSplitCh, lane);
    }
    __syncthreads();
    const unsigned col0 = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(cg * 16U + wave * 4U)));
    const unsigned k0 = sg * SG * kGateKps;
    bool routed_i4 = false;
#if STRIX_GLM_FAST_I4_CORE
    if constexpr (I4) {
        if (slot >= 1) {  // fast_i4: W4A16 routed gate/up into the same raw split partials (part layout)
            routed_i4 = true;
            float* const part0 = a.part(first, m, sg * SG);
            constexpr std::size_t row_stride = std::size_t{2} * kGateSplits * kInter;
            strix::core::w4a16::wave_rows<MaxR, kGateKps, SG * 2U>(
                a.i4.wq[m] + std::size_t{si.expert} * (std::size_t{kHidden} * kInter / 2U),
                a.i4.ws[m] + std::size_t{si.expert} * kInter * (kHidden / 64U), kInter, kHidden, k0, k0 + SG * kGateKps, col0, xs,
                SG * kSplitCh, lane, nrows, [&](unsigned split, unsigned r, unsigned col, float v) {
                    if (r < nrows) part0[std::size_t{split} * kInter + r * row_stride + col] = v;
                });
        }
    }
#endif
    if (!routed_i4)
        wide_rows<MaxR, kGateKps, true, Pipe>(w, kInter, k0, k0 + SG * kGateKps, col0, xs, SG * kSplitCh, nrows, wave, lane, rowbuf,
                                        a.part(first, m, sg * SG), std::size_t{2} * kGateSplits * kInter, nullptr, 0, 0);
    auto finish = [&](unsigned block) {
        const std::uint16_t* gsvh = slot_matrix(a, si, slot, 0).svh;
        const std::uint16_t* usvh = slot_matrix(a, si, slot, 1).svh;
        const std::uint16_t* dsuh = slot_matrix(a, si, slot, 2).suh;
        for (unsigned lr = wave; lr < nrows; lr += 4) {
            float g[4], u[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) g[j] = u[j] = 0.0F;
            for (unsigned sp = 0; sp < kGateSplits; ++sp) {
                const float* pg = a.part(first + lr, 0, sp) + block * 128 + lane;
                const float* pu = a.part(first + lr, 1, sp) + block * 128 + lane;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    g[j] += load_agent(pg + 32 * j);
                    u[j] += load_agent(pu + 32 * j);
                }
            }
            d::hadamard128(g, lane);
            d::hadamard128(u, lane);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned i = block * 128 + lane + 32U * static_cast<unsigned>(j);
                const float gl = g[j] * d::half_bits(gsvh[i]), ul = u[j] * d::half_bits(usvh[i]);
                if (a.dbg_gu != nullptr) {
                    a.dbg_gu[(static_cast<std::size_t>(first + lr) * 2) * kInter + i] = gl;
                    a.dbg_gu[(static_cast<std::size_t>(first + lr) * 2 + 1) * kInter + i] = ul;
                }
                const float act = swiglu(gl, ul);
                const float scaled = d::materialize(act * d::half_bits(dsuh[i]));
                v[j] = static_cast<float>(static_cast<_Float16>(scaled));
            }
            d::hadamard128(v, lane);
            _Float16* out = a.at(first + lr);
#pragma unroll
            for (int j = 0; j < 4; ++j) out[block * 128 + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
        }
    };
#pragma unroll
    for (unsigned h = 0; h < 2; ++h)
        if (last_arrival(a.gu_counter(slot, cg * 2U + h), 2 * (kGateSplits / SG))) finish(cg * 2U + h);
}

// grid (kGuGroups x 32 / SG, slots, 2 matrices); FirstSlot 0 (shared included) or 1 (routed only).
template <int MaxR, unsigned SG, unsigned FirstSlot, bool Pipe, bool I4 = false>
__global__ __launch_bounds__(128) void gate_up_kernel(Args a) {
    __shared__ alignas(16) unsigned char smem[kGuSmem<MaxR, SG, Pipe>];
    __shared__ SlotInfo si;
    __shared__ std::uint32_t ids[kMaxRows * kTopE], mask[kMaskWords];
    const unsigned slot = blockIdx.y + FirstSlot;
    slot_info(a, slot, true, si, ids, mask);
    if (!si.active) return;
    gate_up_body<MaxR, SG, Pipe, I4>(a, si, slot, blockIdx.x % kGuGroups, blockIdx.x / kGuGroups, blockIdx.z, smem);
}

// ---------------------------------------------------------------- router (R1 logits, R14 / R5) + shared gate/up
// Router CTA: 8 experts; all 128 threads stage the next 512-channel chunk of the 8 F16 rows (16-byte loads into
// registers, then LDS, double-buffered); lane t < 8 R runs the chain of (expert t % 8, row t / 8):
// acc = +0; acc = fmaf(y[r][i], float(w[e][i]), acc), i ascending (= dev::dense). The last router CTA computes, one
// wave per row, s = sigmoid(logit), keys ordered_key(s + bias), the top 8 (key desc, id asc) by 8 wave arg-max rounds,
// ids ascending, den and weights (= dev::route).
__device__ __forceinline__ void router_cta(const Args& a, unsigned char* smem) {
    auto* wbuf = reinterpret_cast<std::uint16_t*>(smem);                // [2][kRouterEpc][kRouterStride] F16
    auto* xbuf = reinterpret_cast<float*>(smem + kRouterWBytes);        // [2][kMaxRows][kRouterChunk]
    const unsigned t = threadIdx.x, e0 = blockIdx.x * kRouterEpc;
    constexpr unsigned kQ = kRouterChunk / 8;   // uint4 per expert row chunk
    constexpr unsigned kXQ = kRouterChunk / 4;  // float4 per y row chunk
    constexpr unsigned kWJ = kRouterEpc * kQ / 128, kXJ = kMaxRows * kXQ / 128;
    WideV4 reg[kWJ];
    using F4 = float __attribute__((ext_vector_type(4)));  // HIP float4 struct arrays end up in scratch
    F4 xreg[kXJ];
    const unsigned xq = a.rows * kXQ;  // float4 of the active rows per chunk
    auto gload = [&](unsigned c) {
#pragma unroll
        for (unsigned j = 0; j < kWJ; ++j) {
            const unsigned q = t + 128U * j, e = q / kQ, o = q % kQ;
            reg[j] = *reinterpret_cast<const WideV4*>(a.router_w + static_cast<std::size_t>(e0 + e) * kHidden + c * kRouterChunk + o * 8);
        }
#pragma unroll
        for (unsigned j = 0; j < kXJ; ++j) {
            const unsigned q = t + 128U * j;
            if (q < xq)
                xreg[j] = *reinterpret_cast<const F4*>(a.y + static_cast<std::size_t>(q / kXQ) * kHidden + c * kRouterChunk +
                                                           (q % kXQ) * 4);
        }
    };
    auto lstore = [&](unsigned buf) {
#pragma unroll
        for (unsigned j = 0; j < kWJ; ++j) {
            const unsigned q = t + 128U * j, e = q / kQ, o = q % kQ;
            *reinterpret_cast<WideV4*>(wbuf + (buf * kRouterEpc + e) * kRouterStride + o * 8) = reg[j];
        }
#pragma unroll
        for (unsigned j = 0; j < kXJ; ++j) {
            const unsigned q = t + 128U * j;
            if (q < xq) *reinterpret_cast<F4*>(xbuf + (buf * kMaxRows + q / kXQ) * kRouterChunk + (q % kXQ) * 4) = xreg[j];
        }
    };
    const unsigned el = t % kRouterEpc, r = t / kRouterEpc;
    const bool active = r < a.rows;
    float acc = 0.0F;
    gload(0);
    lstore(0);
    __syncthreads();
    for (unsigned c = 0; c < kHidden / kRouterChunk; ++c) {
        if (c + 1 < kHidden / kRouterChunk) gload(c + 1);
        if (active) {
            const std::uint16_t* wr = wbuf + ((c & 1U) * kRouterEpc + el) * kRouterStride;
            const float* xc = xbuf + ((c & 1U) * kMaxRows + r) * kRouterChunk;
#pragma unroll 4
            for (unsigned i = 0; i < kRouterChunk; i += 8) {
                const WideV4 wv = *reinterpret_cast<const WideV4*>(wr + i);
                const F4 x0 = *reinterpret_cast<const F4*>(xc + i), x1 = *reinterpret_cast<const F4*>(xc + i + 4);
#pragma unroll
                for (unsigned q = 0; q < 8; ++q) {
                    const std::uint16_t h = static_cast<std::uint16_t>(wv[q / 2] >> (16U * (q & 1U)));
                    // Hardware F16 -> F32 conversion: exact for every non-NaN F16 incl. subnormals and infinities (= the
                    // f16_to_float of dev::dense) under the build's modes (FP16/FP32 denormals preserved, RNE). NaN weights
                    // are outside the contract (finite router weights); their payload equality is not claimed. measurement: the
                    // branchy integer f16_to_float compiled to ~50 instructions per chain step (launch 1 ~240 us).
                    acc = __builtin_fmaf(q < 4 ? x0[q] : x1[q - 4], static_cast<float>(__builtin_bit_cast(_Float16, h)), acc);
                }
            }
        }
        if (c + 1 < kHidden / kRouterChunk) lstore((c + 1) & 1U);
        __syncthreads();
    }
    if (active) a.logits[r * kExperts + e0 + el] = acc;
    if (!last_arrival(a.router_counter(), kRouterCtas)) return;
    const unsigned wave = t >> 5U, lane = t & 31U;
    for (unsigned row = wave; row < a.rows; row += 4) {
        float s[9];
        std::uint32_t key[9];
#pragma unroll
        for (unsigned j = 0; j < 9; ++j) {
            const unsigned e = lane + 32U * j;
            s[j] = sigmoid(load_agent(a.logits + row * kExperts + e));
            key[j] = ordered_key(s[j] + a.bias[e]);
        }
        unsigned taken = 0;
        std::uint32_t pick[kTopE];
        float ps[kTopE];
#pragma unroll
        for (unsigned k = 0; k < kTopE; ++k) {
            std::uint32_t bk = 0, be = kExperts;
            float bs = 0.0F;
#pragma unroll
            for (unsigned j = 0; j < 9; ++j)  // ascending id within the lane; strict > keeps the lower id on ties
                if (!((taken >> j) & 1U) && (be == kExperts || key[j] > bk)) {
                    bk = key[j];
                    be = lane + 32U * j;
                    bs = s[j];
                }
#pragma unroll
            for (int off = 16; off >= 1; off >>= 1) {
                const std::uint32_t ok = __shfl_xor(bk, off, 32), oe = __shfl_xor(be, off, 32);
                const float os = __shfl_xor(bs, off, 32);
                if (oe != kExperts && (be == kExperts || ok > bk || (ok == bk && oe < be))) {
                    bk = ok;
                    be = oe;
                    bs = os;
                }
            }
            pick[k] = be;
            ps[k] = bs;
            if ((be & 31U) == lane) taken |= 1U << (be >> 5U);
        }
#pragma unroll
        for (unsigned i = 1; i < kTopE; ++i)  // ascending id (insertion sort, uniform)
#pragma unroll
            for (unsigned j = i; j > 0; --j)
                if (pick[j - 1] > pick[j]) {
                    const std::uint32_t te = pick[j];
                    pick[j] = pick[j - 1];
                    pick[j - 1] = te;
                    const float tv = ps[j];
                    ps[j] = ps[j - 1];
                    ps[j - 1] = tv;
                }
        float den = 0.0F;
#pragma unroll
        for (unsigned k = 0; k < kTopE; ++k) den = den + ps[k];
        den = den + kRouteDenEps;
        if (lane == 0)
#pragma unroll
            for (unsigned k = 0; k < kTopE; ++k) {
                a.ids[row * kTopE + k] = pick[k];
                a.weights[row * kTopE + k] = (ps[k] / den) * kRouteScale;
            }
    }
}

template <int MaxR, unsigned SG, bool Pipe>
constexpr std::size_t kRsSmem = kGuSmem<MaxR, SG, Pipe> > kRouterSmem ? kGuSmem<MaxR, SG, Pipe> : kRouterSmem;
// grid kRouterCtas + kGuGroups x 32 / SG x 2 (the shared expert's gate/up, slot 0).
template <int MaxR, unsigned SG, bool Pipe>
__global__ __launch_bounds__(128) void router_shared_kernel(Args a) {
    __shared__ alignas(16) unsigned char smem[kRsSmem<MaxR, SG, Pipe>];
    __shared__ SlotInfo si;
    if (blockIdx.x < kRouterCtas) {
        router_cta(a, smem);
        return;
    }
    const unsigned local = blockIdx.x - kRouterCtas, per_m = kGuGroups * (kGateSplits / SG);
    slot_info(a, 0, false, si, nullptr, nullptr);
    gate_up_body<MaxR, SG, Pipe>(a, si, 0, (local % per_m) % kGuGroups, (local % per_m) / kGuGroups, local / per_m, smem);
}

// ---------------------------------------------------------------- down (+ output transform, + R16 combine)
// grid (kDnGroups, slots incl. shared). CTA = 4 waves x 4 columns (256 outputs) over the whole K: the 4 library splits
// accumulate in ascending order in the CTA's fin rows; H32 + svh per 128-output block into yslot; the last CTA of the
// 256-output group (over all active slots) runs R16 for every row.
template <int MaxR, bool Pipe, bool I4 = false>
__global__ __launch_bounds__(128) void down_kernel(Args a) {
    __shared__ _Float16 xs[MaxR * kInter];
    __shared__ WideV4 rowbuf[4 * kRowbufStride<Pipe>];
    __shared__ float fin[MaxR * 256];
    __shared__ SlotInfo si;
    __shared__ std::uint32_t ids[kMaxRows * kTopE], mask[kMaskWords], pairs[kMaxRows * kTopE];
    const unsigned slot = blockIdx.y, cg = blockIdx.x;
    slot_info(a, slot, true, si, ids, mask);
    if (!si.active) return;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const Exl3 w = slot_matrix(a, si, slot, 2);
    const unsigned nrows = si.nrows, first = si.first;
    for (unsigned q = threadIdx.x; q < nrows * (kInter / 2); q += 128) {
        const unsigned lr = q / (kInter / 2), c = q % (kInter / 2);
        reinterpret_cast<std::uint32_t*>(xs)[lr * (kInter / 2) + c] = reinterpret_cast<const std::uint32_t*>(a.at(first + lr))[c];
    }
    __syncthreads();
    const unsigned col0 = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(cg * 16U + wave * 4U)));
    bool routed_i4 = false;
#if STRIX_GLM_FAST_I4_CORE
    if constexpr (I4) {
        if (slot >= 1 && a.i4.wq[2] != nullptr) {  // fast_i4 scope all: W4A16 routed down; fin = the 4 splits accumulated in ascending order (as below)
            routed_i4 = true;
            strix::core::w4a16::wave_rows<MaxR, kDownKps, kDownKt / 4U>(
                a.i4.wq[2] + std::size_t{si.expert} * (std::size_t{kInter} * kHidden / 2U),
                a.i4.ws[2] + std::size_t{si.expert} * kHidden * (kInter / 64U), kHidden, kInter, 0, kDownKt, col0, xs, kInter, lane, nrows,
                [&](unsigned split, unsigned r, unsigned col, float v) {
                    if (r >= nrows) return;
                    float* pf = fin + r * 256U + (col - cg * 256U);
                    *pf = split == 0 ? 0.0F + v : *pf + v;
                });
        }
    }
#endif
    if (!routed_i4)
        wide_rows<MaxR, kDownKps, false, Pipe>(w, kHidden, 0, kDownKt, col0, xs, kInter, nrows, wave, lane, rowbuf, nullptr, 0, fin, 256,
                                         wave * 4U);
    __syncthreads();
    for (unsigned job = wave; job < nrows * 2; job += 4) {  // output transform per (row, 128-output block)
        const unsigned lr = job / 2, b = job % 2;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = fin[lr * 256 + b * 128 + lane + 32U * static_cast<unsigned>(j)];
        d::hadamard128(v, lane);
        float* y = a.yslot(first + lr);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned o = cg * 256 + b * 128 + lane + 32U * static_cast<unsigned>(j);
            y[o] = v[j] * d::half_bits(w.svh[o]);
        }
    }
    if (!last_arrival(a.dn_counter(cg), si.count + 1)) return;
    // R16 for the group's 256 outputs: pairs of (row, k-th expert ascending), from the same ids rule as slot_info.
    const unsigned R = a.rows, n = R * kTopE;
    if (threadIdx.x < n) {
        const unsigned r = threadIdx.x / kTopE, e = ids[threadIdx.x];
        unsigned below = 0, before = 0;
        for (unsigned j = 0; j < n; ++j) {
            below += ids[j] < e ? 1U : 0U;
            before += (ids[j] == e && j / kTopE < r) ? 1U : 0U;
        }
        pairs[threadIdx.x] = R + below + before;
    }
    __syncthreads();
    for (unsigned r = wave; r < R; r += 4) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned o = cg * 256 + lane + 32U * static_cast<unsigned>(j);
            float acc = 0.0F;
            for (unsigned k = 0; k < kTopE; ++k)
                acc = __builtin_fmaf(a.weights[r * kTopE + k], load_agent(a.yslot(pairs[r * kTopE + k]) + o), acc);
            a.out[static_cast<std::size_t>(r) * kHidden + o] = acc + load_agent(a.yslot(r) + o);
        }
    }
    // One R16 workgroup per 256-output group stores the partial: the last of the kDnGroups publishes it.
    if (a.pub.on()) strix::pair::producer_publish(a.pub, kDnGroups);
}

// The core down path does not publish in-kernel: one thread raises the flag after it (same message, one more launch).
__global__ void publish_after_kernel(strix::pair::ProducerPublish t) {
    if (threadIdx.x == 0) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
        __hip_atomic_store(t.flag, t.sequence, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

template <int MaxR, unsigned SG, unsigned FirstSlot, bool Pipe>
__global__ __launch_bounds__(128) void core_gate_up_kernel(Args a) {
    __shared__ alignas(16) unsigned char smem[kGuSmem<MaxR, SG, Pipe>];
    __shared__ SlotInfo si;
    __shared__ std::uint32_t ids[kMaxRows * kTopE], mask[kMaskWords];
    const unsigned slot = blockIdx.y + FirstSlot;
    slot_info(a, slot, true, si, ids, mask);
    if (!si.active) return;
    DecodeCore::gate_up4_body<MaxR, SG, Pipe>(a, si, slot, blockIdx.x % kGuGroups, blockIdx.x / kGuGroups, blockIdx.z, smem);
}

template <int MaxR, bool Pipe>
__global__ __launch_bounds__(128) void core_down_kernel(Args a) {
    __shared__ _Float16 xs[MaxR * kInter];
    __shared__ WideV4 rowbuf[4 * kRowbufStride<Pipe>];
    __shared__ float fin[MaxR * 256];
    __shared__ SlotInfo si;
    __shared__ std::uint32_t ids[kMaxRows * kTopE], mask[kMaskWords], pairs[kMaxRows * kTopE];
    DecodeCore::down4_cta<MaxR, Pipe>(a, xs, rowbuf, fin, si, ids, mask, pairs);
}

template <int MaxR, unsigned SG, bool Pipe>
__global__ __launch_bounds__(128) void core_router_shared_kernel(Args a) {
    __shared__ alignas(16) unsigned char smem[kRsSmem<MaxR, SG, Pipe>];
    __shared__ SlotInfo si;
    if (blockIdx.x < kRouterCtas) {
        router_cta(a, smem);
        return;
    }
    const unsigned local = blockIdx.x - kRouterCtas, per_m = kGuGroups * (kGateSplits / SG);
    slot_info(a, 0, false, si, nullptr, nullptr);
    DecodeCore::gate_up4_body<MaxR, SG, Pipe>(a, si, 0, (local % per_m) % kGuGroups, (local % per_m) / kGuGroups, local / per_m, smem);
}

Args make_args(const MoeView& v, bool router) {
    if (v.rows < 1 || v.rows > kMaxRows) throw std::invalid_argument("glm moe: rows must be 1..8");
    if (v.y == nullptr || v.ids == nullptr || v.weights == nullptr || v.experts == nullptr || v.out == nullptr ||
        v.ws.base == nullptr || (router && (v.router_w == nullptr || v.bias == nullptr)))
        throw std::invalid_argument("glm moe: null pointer in view");
    const ExpertSlices& s = v.shared;
    if (s.gate.bits != s.up.bits || (s.gate.bits != 4 && s.gate.bits != 6) || (s.down.bits != 4 && s.down.bits != 6) ||
        s.gate.input != kHidden || s.gate.output != kInter || s.up.input != kHidden || s.up.output != kInter ||
        s.down.input != kInter || s.down.output != kHidden)
        throw std::invalid_argument("glm moe: shared expert must be 4096 -> 1024 -> 4096 at K4 or K6");
    auto* ws = static_cast<unsigned char*>(v.ws.base);
    float* logits = v.logits != nullptr ? v.logits : reinterpret_cast<float*>(ws + kOffLogits);
    return {v.rows, v.y, v.router_w, v.bias, v.ids, v.weights, logits, v.experts, v.shared, v.out, ws, v.debug_gate_up, v.i4, v.publish};
}

// SG (library splits per gate_up CTA; bitwise at every value). Default = the measurement (node 0, cold synthetic
// layers, ABA x3): SG1 fastest at R1-2 (gate_up R1 183 vs 217 us at SG4), SG2 fastest at R4 (650 vs 682) and tied at
// R8 (1199 vs 1221); SG4 / SG8 never fastest. STRIX_GLM_MOE_SG overrides for every row count.
unsigned split_group(unsigned max_rows) {
    static const unsigned forced = [] {
        const char* s = std::getenv("STRIX_GLM_MOE_SG");
        const unsigned v = s != nullptr && *s != '\0' ? static_cast<unsigned>(std::stoul(s)) : 0U;
        if (v != 0 && v != 1 && v != 2 && v != 4 && v != 8) throw std::invalid_argument("STRIX_GLM_MOE_SG must be 1, 2, 4 or 8");
        return v;
    }();
    return forced != 0 ? forced : default_split_group(max_rows);
}

void check_launch(const char* what) {
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(std::string("glm moe: launch failed: ") + what + ": " + hipGetErrorString(e));
}

unsigned routed_slots(unsigned rows) { return rows * kTopE < kExperts ? rows * kTopE : kExperts; }

template <int MaxR, unsigned SG, bool Pipe>
void launch_rows_sg(const Args& a, bool router, hipStream_t stream, hipEvent_t* marks) {
    const unsigned routed = routed_slots(a.rows);
    static const bool core_enabled = [] {
        const char* value = std::getenv("STRIX_GLM_MOE_CORE");
        return value != nullptr && std::string(value) == "1";
    }();
    if constexpr (MaxR == 1) {
        if (core_enabled) {
            if (router) {
                core_router_shared_kernel<MaxR, SG, Pipe><<<kRouterCtas + kGuGroups * (kGateSplits / SG) * 2, 128, 0, stream>>>(a);
                check_launch("router_shared");
                if (marks != nullptr) (void)hipEventRecord(marks[0], stream);
                core_gate_up_kernel<MaxR, SG, 1, Pipe><<<dim3(kGuGroups * (kGateSplits / SG), routed, 2), 128, 0, stream>>>(a);
            } else {
                core_gate_up_kernel<MaxR, SG, 0, Pipe><<<dim3(kGuGroups * (kGateSplits / SG), routed + 1, 2), 128, 0, stream>>>(a);
            }
            check_launch("core_gate_up");
            if (marks != nullptr) (void)hipEventRecord(marks[router ? 1 : 0], stream);
            core_down_kernel<MaxR, Pipe><<<dim3(kDnGroups, routed + 1), 128, 0, stream>>>(a);
            check_launch("core_down");
            if (a.pub.on()) {
                publish_after_kernel<<<1, 32, 0, stream>>>(a.pub);
                check_launch("core_down publish");
            }
            return;
        }
    }
    if (router) {
        router_shared_kernel<MaxR, SG, Pipe><<<kRouterCtas + kGuGroups * (kGateSplits / SG) * 2, 128, 0, stream>>>(a);
        check_launch("router_shared");
        if (marks != nullptr) (void)hipEventRecord(marks[0], stream);
        gate_up_kernel<MaxR, SG, 1, Pipe><<<dim3(kGuGroups * (kGateSplits / SG), routed, 2), 128, 0, stream>>>(a);
    } else {
        gate_up_kernel<MaxR, SG, 0, Pipe><<<dim3(kGuGroups * (kGateSplits / SG), routed + 1, 2), 128, 0, stream>>>(a);
    }
    check_launch("gate_up");
    if (marks != nullptr) (void)hipEventRecord(marks[router ? 1 : 0], stream);
    down_kernel<MaxR, Pipe><<<dim3(kDnGroups, routed + 1), 128, 0, stream>>>(a);
    check_launch("down");
}
// fast_i4 (MoeView::i4): the same three launches with the routed slots on the W4A16 core (no core / pipe variants).
template <int MaxR, unsigned SG>
void launch_rows_i4(const Args& a, bool router, hipStream_t stream, hipEvent_t* marks) {
#if STRIX_GLM_FAST_I4_CORE
    const unsigned routed = routed_slots(a.rows);
    if (router) {
        router_shared_kernel<MaxR, SG, false><<<kRouterCtas + kGuGroups * (kGateSplits / SG) * 2, 128, 0, stream>>>(a);
        check_launch("router_shared");
        if (marks != nullptr) (void)hipEventRecord(marks[0], stream);
        gate_up_kernel<MaxR, SG, 1, false, true><<<dim3(kGuGroups * (kGateSplits / SG), routed, 2), 128, 0, stream>>>(a);
    } else {
        gate_up_kernel<MaxR, SG, 0, false, true><<<dim3(kGuGroups * (kGateSplits / SG), routed + 1, 2), 128, 0, stream>>>(a);
    }
    check_launch("i4 gate_up");
    if (marks != nullptr) (void)hipEventRecord(marks[router ? 1 : 0], stream);
    down_kernel<MaxR, false, true><<<dim3(kDnGroups, routed + 1), 128, 0, stream>>>(a);
    check_launch("i4 down");
#else
    (void)a, (void)router, (void)stream, (void)marks;
    throw std::invalid_argument("glm moe: fast_i4 W4A16 core not built (shared stored INT4 core)");
#endif
}
template <int MaxR, bool Pipe> void launch_rows(const Args& a, bool router, hipStream_t stream, hipEvent_t* marks) {
    if (a.i4.on()) {
        if (split_group(static_cast<unsigned>(MaxR)) == 1) launch_rows_i4<MaxR, 1>(a, router, stream, marks);
        else launch_rows_i4<MaxR, 2>(a, router, stream, marks);
        return;
    }
    switch (split_group(static_cast<unsigned>(MaxR))) {
    case 1: launch_rows_sg<MaxR, 1, Pipe>(a, router, stream, marks); break;
    case 2: launch_rows_sg<MaxR, 2, Pipe>(a, router, stream, marks); break;
    case 8: launch_rows_sg<MaxR, 8, Pipe>(a, router, stream, marks); break;
    default: launch_rows_sg<MaxR, 4, Pipe>(a, router, stream, marks); break;
    }
}
template <bool Pipe>
void launch(const MoeView& v, bool router, hipStream_t stream, hipEvent_t* marks) {
    const Args a = make_args(v, router);
    if (a.rows == 1) launch_rows<1, Pipe>(a, router, stream, marks);
    else if (a.rows == 2) launch_rows<2, Pipe>(a, router, stream, marks);
    else if (a.rows <= 4) launch_rows<4, Pipe>(a, router, stream, marks);
    else launch_rows<8, Pipe>(a, router, stream, marks);
}

}  // namespace

std::size_t moe_workspace_bytes() { return kMoeBytes; }
bool fast_i4_built() { return STRIX_GLM_FAST_I4_CORE != 0; }
const float* moe_pair_gate_partials(const MoeWorkspace& ws, unsigned pair) {
    return reinterpret_cast<const float*>(static_cast<const unsigned char*>(ws.base) + kOffPart) +
           static_cast<std::size_t>(pair) * 2 * kGateSplits * kInter;
}
const float* moe_pair_output(const MoeWorkspace& ws, unsigned pair) {
    return reinterpret_cast<const float*>(static_cast<const unsigned char*>(ws.base) + kOffY) + static_cast<std::size_t>(pair) * kHidden;
}
const std::uint16_t* moe_pair_down_input(const MoeWorkspace& ws, unsigned pair) {
    return reinterpret_cast<const std::uint16_t*>(static_cast<const unsigned char*>(ws.base) + kOffAt) + static_cast<std::size_t>(pair) * kInter;
}
void enqueue_moe(const MoeView& v, hipStream_t stream, hipEvent_t* marks) { launch<false>(v, true, stream, marks); }
void enqueue_experts(const MoeView& v, hipStream_t stream, hipEvent_t* marks) { launch<false>(v, false, stream, marks); }

void enqueue_moe_pipe(const MoeView& v, hipStream_t stream, hipEvent_t* marks) { launch<true>(v, true, stream, marks); }
void enqueue_experts_pipe(const MoeView& v, hipStream_t stream, hipEvent_t* marks) { launch<true>(v, false, stream, marks); }
bool pipe_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRIX_GLM_MOE_PIPE");
        return v != nullptr && std::string(v) == "1";
    }();
    return enabled;
}

}  // namespace strix::glm::moe
