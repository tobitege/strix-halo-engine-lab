#pragma once
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include "decode_schedule.hpp"
// Decode schedules share a family; traits retain engine-specific K chains,
// split materialization and epilogues. Device adapters supply the EXL3 decoder.
namespace strix::moe_core {
template <class Traits>
struct Decode {
    static_assert(Traits::Schedule::kAscendingSplitSums);
    static_assert(Traits::Schedule::kHalfMaterializedDownInput);
    using Args = typename Traits::Args;
    using Dispatch = typename Traits::Dispatch;
    using ExpertSlices = typename Traits::ExpertSlices;
    using Exl3 = typename Traits::Exl3;
    using WideV4 = typename Traits::WideV4;
    using SlotInfo = typename Traits::SlotInfo;
template <int MaxR, int Depth, int W, int DepthBig, bool Wide = false, bool Single = false>
static __device__ void gate_up4_body(const Args& a, const Dispatch* dp, const ExpertSlices& E, unsigned slot, unsigned nrows,
                              unsigned first, unsigned cg, unsigned split, unsigned m, _Float16* xs, WideV4* rowbuf = nullptr,
                              bool single = false, std::uint32_t ready = 0) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const Exl3& w = m == 0 ? E.gate : E.up;
    for (unsigned job = wave; job < nrows * 3; job += W) {  // input transform of the split's 3 H128 blocks per row
        const unsigned lr = job / 3, b = job % 3;
        const unsigned row = slot == Traits::kSharedSlot ? lr : dp->row[slot][lr];
        Traits::transform_block_at(a.y + static_cast<std::size_t>(row) * Traits::kHidden, w.suh, xs + lr * Traits::kSplitCh, split * Traits::kSplitCh + b * 128U,
                              b * 128U, lane);
    }
    __syncthreads();
    auto finish = [&](unsigned block) {
    for (unsigned r = wave; r < nrows; r += W) {
        float g[4], u[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) g[j] = u[j] = 0.0F;
        for (unsigned sp = 0; sp < Traits::kGateSplits; ++sp) {
            const float* pg = a.part_gu(first + r, 0, sp) + block * 128 + lane;
            const float* pu = a.part_gu(first + r, 1, sp) + block * 128 + lane;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                g[j] += Traits::load_agent(pg + 32 * j);
                u[j] += Traits::load_agent(pu + 32 * j);
            }
        }
        Traits::hadamard128(g, lane);
        Traits::hadamard128(u, lane);
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned i = block * 128 + lane + 32U * static_cast<unsigned>(j);
            const float act = Traits::swiglu(g[j] * Traits::half_bits(E.gate.svh[i]), u[j] * Traits::half_bits(E.up.svh[i]));
            const float scaled = Traits::materialize(act * Traits::half_bits(E.down.suh[i]));
            v[j] = static_cast<float>(static_cast<_Float16>(scaled));
        }
        Traits::hadamard128(v, lane);
        _Float16* out = a.at(first + r);
#pragma unroll
        for (int j = 0; j < 4; ++j) out[block * 128 + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
    }
    };
    if constexpr (Wide) {
        const unsigned col0 = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(cg * 16U + wave * 4U)));
        Traits::template split16_rows<MaxR>(w, Traits::kInter, Traits::kHidden / 16, Traits::kGateSplits, col0, split, xs, nrows, wave, lane, rowbuf,
                           a.part_gu(first, m, split), std::size_t{2} * 16 * Traits::kInter);
#pragma unroll
        for (unsigned h = 0; h < 2; ++h)
            if (Traits::last_arrival(a.gu4_counter(slot, cg * 2U + h), 2 * Traits::kGateSplits)) {
                finish(cg * 2U + h);
                if (ready != 0) Traits::publish_gu_ready(a, slot, cg * 2U + h, ready);
            }
    } else {
        const unsigned col = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(cg * static_cast<unsigned>(W) + wave)));
        if constexpr (Single) {
            if (single && w.bits == 3) {
                if (a.single_reader_counts != nullptr && threadIdx.x == 0) atomicAdd(a.single_reader_counts, 1U);
                Traits::singleton3_core(w, Traits::kInter, col, split, xs, lane, a.part_gu(first, m, split));
            }
            else Traits::template split4_rows<MaxR, Depth, DepthBig>(w, Traits::kInter, Traits::kHidden / 16, Traits::kGateSplits, col, split, xs, nrows, lane,
                                                     a.part_gu(first, m, split), std::size_t{2} * 16 * Traits::kInter);
        } else Traits::template split4_rows<MaxR, Depth, DepthBig>(w, Traits::kInter, Traits::kHidden / 16, Traits::kGateSplits, col, split, xs, nrows, lane,
                                                 a.part_gu(first, m, split), std::size_t{2} * 16 * Traits::kInter);
        const unsigned block = cg * W / 8;
        if (Traits::last_arrival(a.gu4_counter(slot, block), (8 / W) * 2 * Traits::kGateSplits)) {
            finish(block);
            if (ready != 0) Traits::publish_gu_ready(a, slot, block, ready);
        }
    }
}

template <int MaxR, int Depth, int W, int DepthBig, bool SkipShared, bool Wide = false, bool Single = false>
static __device__ __forceinline__ void gate_up4_cta(const Args& a, unsigned bxr, unsigned by, unsigned bz, _Float16* xs, WideV4* rowbuf,
                                             std::uint32_t ready = 0) {
    const Dispatch* dp = a.disp();
    constexpr unsigned groups = Wide ? Traits::kInter / 256U : Traits::kInter / (16U * W);
    // Former grid separated a slot's gate/up CTAs by an entire z plane spanning all slots. Pair the two
    // matrices per tile to reduce partial/counter lifetime; CTA counts and split arithmetic stay unchanged.
    const unsigned bx = a.interleave_gu ? bxr / 2U : bxr;
    const unsigned m = a.interleave_gu ? bxr % 2U : bz;
    unsigned slot;
    if (!Traits::map_slot(dp, by, slot)) return;
    if (SkipShared && slot == Traits::kSharedSlot) return;  // done by the router + shared launch
    gate_up4_body<MaxR, Depth, W, DepthBig, Wide, Single>(a, dp, Traits::slot_slices(a, dp, slot), slot, dp->nrows[slot], dp->first[slot],
                                                       bx % groups, bx / groups, m, xs, rowbuf,
                                                       Single && Traits::template single_stream_for<Depth>(a, dp, slot), ready);
}

// SplitOut (split-producer chunk frames, MiniMax only): out is stored through a.split_index(r, col) (a SplitLayout)
// instead of row-major; the default instantiation is the unchanged row-major store.
template <int W, bool SplitOut = false>
static __device__ __forceinline__ void down4_combine(const Args& a, const Dispatch* dp, unsigned block) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    // Finish every slot's block, then the G4 combine per row: acc = 0; fmaf over its experts in ascending id; + shared.
    const unsigned rows = dp->rows;
    auto finish = [&](unsigned pair, const std::uint16_t* svh, float (&v)[4]) {
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = 0.0F;
        for (unsigned sp = 0; sp < Traits::kDownSplits; ++sp) {
            const float* pp = a.part_dn(pair, sp) + block * 128 + lane;
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] += Traits::load_agent(pp + 32 * j);
        }
        Traits::hadamard128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = v[j] * Traits::half_bits(svh[block * 128 + lane + 32U * static_cast<unsigned>(j)]);
    };
    for (unsigned r = wave; r < rows; r += W) {
        float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        for (unsigned k = 0; k < Traits::kTopE; ++k) {
            float v[4];
            finish(dp->row_pair[r][k], a.experts[dp->row_expert[r][k]].down.svh, v);
            const float wk = dp->row_weight[r][k];
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[j] = __builtin_fmaf(wk, v[j], acc[j]);
        }
        float v[4];
        finish(dp->first[Traits::kSharedSlot] + r, a.shared.down.svh, v);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if constexpr (SplitOut)
                a.out[a.split_index(r, block * 128 + lane + 32U * static_cast<unsigned>(j))] = acc[j] + v[j];
            else
                a.out[static_cast<std::size_t>(r) * Traits::kHidden + block * 128 + lane + 32U * static_cast<unsigned>(j)] = acc[j] + v[j];
        }
    }
}

template <int MaxR, int Depth, int W, int DepthBig, bool Wide = false, bool Single = false, bool PartialOnly = false,
          bool SplitOut = false>
static __device__ __forceinline__ void down4_cta(const Args& a, unsigned bx, unsigned by, _Float16* xs, WideV4* rowbuf,
                                          std::uint32_t ready = 0) {
    const Dispatch* dp = a.disp();
    unsigned slot;
    if (!Traits::map_slot(dp, by, slot)) return;
    constexpr unsigned groups = Wide ? Traits::kHidden / 256U : Traits::kHidden / (16U * W);
    const unsigned cg = bx % groups, split = bx / groups;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const Exl3& w = Traits::slot_slices(a, dp, slot).down;
    const unsigned nrows = dp->nrows[slot], first = dp->first[slot];
    if (ready != 0 && slot != Traits::kSharedSlot) {  // the split's 384 inputs = gate/up blocks 3 split .. 3 split + 2 of this slot
        if (threadIdx.x == 0)
            for (unsigned b = 0; b < Traits::kSplitCh / 128; ++b) {
                const std::uint32_t* f = a.gu_ready(slot, split * (Traits::kSplitCh / 128) + b);
                for (unsigned spin = 0; __hip_atomic_load(f, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != ready; ++spin) {
                    if (spin == (1U << 26)) __builtin_trap();  // a producer never finished: fault instead of hanging
                    __builtin_amdgcn_s_sleep(1);
                }
            }
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    for (unsigned q = threadIdx.x; q < nrows * (Traits::kSplitCh / 2); q += 32U * W) {  // the split's transformed channels
        const unsigned lr = q / (Traits::kSplitCh / 2), c = q % (Traits::kSplitCh / 2);
        reinterpret_cast<std::uint32_t*>(xs)[lr * (Traits::kSplitCh / 2) + c] =
            reinterpret_cast<const std::uint32_t*>(a.at(first + lr) + split * Traits::kSplitCh)[c];
    }
    __syncthreads();
    if constexpr (Wide) {
        const unsigned col0 = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(cg * 16U + wave * 4U)));
        Traits::template split16_rows<MaxR>(w, Traits::kHidden, Traits::kInter / 16, Traits::kDownSplits, col0, split, xs, nrows, wave, lane, rowbuf,
                           a.part_dn(first, split), std::size_t{4} * Traits::kHidden);
        if constexpr (!PartialOnly) {
#pragma unroll
            for (unsigned h = 0; h < 2; ++h)
                if (Traits::last_arrival(a.dn4_counter(cg * 2U + h), (dp->count + 1) * Traits::kDownSplits)) {
                    down4_combine<W, SplitOut>(a, dp, cg * 2U + h);
                    Traits::lean_publish(a.pub, a.pub.tiles);
                }
        }
    } else {
        const unsigned col = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(cg * static_cast<unsigned>(W) + wave)));
        if constexpr (Single) {
            if (Traits::template single_stream_for<Depth>(a, dp, slot) && w.bits == 3) {
                if (a.single_reader_counts != nullptr && threadIdx.x == 0) atomicAdd(a.single_reader_counts + 1, 1U);
                Traits::singleton3_core(w, Traits::kHidden, col, split, xs, lane, a.part_dn(first, split));
            }
            else Traits::template split4_rows<MaxR, Depth, DepthBig>(w, Traits::kHidden, Traits::kInter / 16, Traits::kDownSplits, col, split, xs, nrows, lane,
                                                     a.part_dn(first, split), std::size_t{4} * Traits::kHidden);
        } else Traits::template split4_rows<MaxR, Depth, DepthBig>(w, Traits::kHidden, Traits::kInter / 16, Traits::kDownSplits, col, split, xs, nrows, lane,
                                                 a.part_dn(first, split), std::size_t{4} * Traits::kHidden);
        if constexpr (!PartialOnly) {
            const unsigned block = cg * W / 8;
            if (Traits::last_arrival(a.dn4_counter(block), (dp->count + 1) * (8 / W) * Traits::kDownSplits)) {
                down4_combine<W, SplitOut>(a, dp, block);
                Traits::lean_publish(a.pub, a.pub.tiles);
            }
        }
    }
}

template <int MaxR, unsigned SG, bool Pipe>
static __device__ __forceinline__ void gate_up4_body(const Args& a, const SlotInfo& si, unsigned slot, unsigned cg, unsigned sg, unsigned m,
                             unsigned char* smem) {
    auto* xs = reinterpret_cast<_Float16*>(smem);
    auto* rowbuf = reinterpret_cast<WideV4*>(smem + std::size_t{MaxR} * SG * Traits::kSplitCh * 2);
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const Exl3 w = Traits::slot_matrix(a, si, slot, m);
    const unsigned nrows = si.nrows, first = si.first;
    for (unsigned job = wave; job < nrows * SG; job += 4) {  // input transform of the CTA's SG H128 blocks per row
        const unsigned lr = job / SG, b = job % SG;
        Traits::transform_block_at(a.y + static_cast<std::size_t>(si.row[lr]) * Traits::kHidden, w.suh, xs + lr * SG * Traits::kSplitCh,
                              (sg * SG + b) * Traits::kSplitCh, b * Traits::kSplitCh, lane);
    }
    __syncthreads();
    const unsigned col0 = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(cg * 16U + wave * 4U)));
    const unsigned k0 = sg * SG * Traits::kGateKps;
    Traits::template wide_rows<MaxR, Traits::kGateKps, true, Pipe>(w, Traits::kInter, k0, k0 + SG * Traits::kGateKps, col0, xs, SG * Traits::kSplitCh, nrows, wave, lane, rowbuf,
                                    a.part(first, m, sg * SG), std::size_t{2} * Traits::kGateSplits * Traits::kInter, nullptr, 0U, 0U);
    auto finish = [&](unsigned block) {
        const std::uint16_t* gsvh = Traits::slot_matrix(a, si, slot, 0U).svh;
        const std::uint16_t* usvh = Traits::slot_matrix(a, si, slot, 1U).svh;
        const std::uint16_t* dsuh = Traits::slot_matrix(a, si, slot, 2U).suh;
        for (unsigned lr = wave; lr < nrows; lr += 4) {
            float g[4], u[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) g[j] = u[j] = 0.0F;
            for (unsigned sp = 0; sp < Traits::kGateSplits; ++sp) {
                const float* pg = a.part(first + lr, 0, sp) + block * 128 + lane;
                const float* pu = a.part(first + lr, 1, sp) + block * 128 + lane;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    g[j] += Traits::load_agent(pg + 32 * j);
                    u[j] += Traits::load_agent(pu + 32 * j);
                }
            }
            Traits::hadamard128(g, lane);
            Traits::hadamard128(u, lane);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned i = block * 128 + lane + 32U * static_cast<unsigned>(j);
                const float gl = g[j] * Traits::half_bits(gsvh[i]), ul = u[j] * Traits::half_bits(usvh[i]);
                if (a.dbg_gu != nullptr) {
                    a.dbg_gu[(static_cast<std::size_t>(first + lr) * 2) * Traits::kInter + i] = gl;
                    a.dbg_gu[(static_cast<std::size_t>(first + lr) * 2 + 1) * Traits::kInter + i] = ul;
                }
                const float act = Traits::swiglu(gl, ul);
                const float scaled = Traits::materialize(act * Traits::half_bits(dsuh[i]));
                v[j] = static_cast<float>(static_cast<_Float16>(scaled));
            }
            Traits::hadamard128(v, lane);
            _Float16* out = a.at(first + lr);
#pragma unroll
            for (int j = 0; j < 4; ++j) out[block * 128 + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
        }
    };
#pragma unroll
    for (unsigned h = 0; h < 2; ++h)
        if (Traits::last_arrival(a.gu_counter(slot, cg * 2U + h), 2 * (Traits::kGateSplits / SG))) finish(cg * 2U + h);
}

template <int MaxR, bool Pipe>
static __device__ __forceinline__ void down4_cta(Args a, _Float16* xs, WideV4* rowbuf, float* fin, SlotInfo& si, std::uint32_t* ids, std::uint32_t* mask, std::uint32_t* pairs) {
    const unsigned slot = blockIdx.y, cg = blockIdx.x;
    Traits::slot_info(a, slot, true, si, ids, mask);
    if (!si.active) return;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const Exl3 w = Traits::slot_matrix(a, si, slot, 2U);
    const unsigned nrows = si.nrows, first = si.first;
    for (unsigned q = threadIdx.x; q < nrows * (Traits::kInter / 2); q += 128) {
        const unsigned lr = q / (Traits::kInter / 2), c = q % (Traits::kInter / 2);
        reinterpret_cast<std::uint32_t*>(xs)[lr * (Traits::kInter / 2) + c] = reinterpret_cast<const std::uint32_t*>(a.at(first + lr))[c];
    }
    __syncthreads();
    const unsigned col0 = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(cg * 16U + wave * 4U)));
    Traits::template wide_rows<MaxR, Traits::kDownKps, false, Pipe>(w, Traits::kHidden, 0U, Traits::kDownKt, col0, xs, Traits::kInter, nrows, wave, lane, rowbuf, nullptr, std::size_t{0}, fin, 256U,
                                     wave * 4U);
    __syncthreads();
    for (unsigned job = wave; job < nrows * 2; job += 4) {  // output transform per (row, 128-output block)
        const unsigned lr = job / 2, b = job % 2;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = fin[lr * 256 + b * 128 + lane + 32U * static_cast<unsigned>(j)];
        Traits::hadamard128(v, lane);
        float* y = a.yslot(first + lr);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned o = cg * 256 + b * 128 + lane + 32U * static_cast<unsigned>(j);
            y[o] = v[j] * Traits::half_bits(w.svh[o]);
        }
    }
    if (!Traits::last_arrival(a.dn_counter(cg), si.count + 1)) return;
    // R16 for the group's 256 outputs: pairs of (row, k-th expert ascending), from the same ids rule as slot_info.
    const unsigned R = a.rows, n = R * Traits::kTopE;
    if (threadIdx.x < n) {
        const unsigned r = threadIdx.x / Traits::kTopE, e = ids[threadIdx.x];
        unsigned below = 0, before = 0;
        for (unsigned j = 0; j < n; ++j) {
            below += ids[j] < e ? 1U : 0U;
            before += (ids[j] == e && j / Traits::kTopE < r) ? 1U : 0U;
        }
        pairs[threadIdx.x] = R + below + before;
    }
    __syncthreads();
    for (unsigned r = wave; r < R; r += 4) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned o = cg * 256 + lane + 32U * static_cast<unsigned>(j);
            float acc = 0.0F;
            for (unsigned k = 0; k < Traits::kTopE; ++k)
                acc = __builtin_fmaf(a.weights[r * Traits::kTopE + k], Traits::load_agent(a.yslot(pairs[r * Traits::kTopE + k]) + o), acc);
            a.out[static_cast<std::size_t>(r) * Traits::kHidden + o] = acc + Traits::load_agent(a.yslot(r) + o);
        }
    }
}
template <int MaxR, int Depth, int W, int DepthBig, bool Wide, bool Single, bool SharedFirst>
static __device__ __forceinline__ void pipe4(Args a, unsigned gu_x, unsigned gu_z, unsigned slots, unsigned dn_x,
                                                     _Float16* xs, WideV4* rowbuf, unsigned& s_ticket) {
    if (threadIdx.x == 0) {
        const unsigned t = atomicAdd(a.pipe_ticket(), 1U);
        if (t + 1U == gridDim.x) __hip_atomic_store(a.pipe_ticket(), 0U, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        s_ticket = t;
    }
    __syncthreads();
    const std::uint32_t epoch = *a.pipe_epoch();
    unsigned t = s_ticket;
    if constexpr (SharedFirst) {
        if (t < dn_x) {
            down4_cta<MaxR, Depth, W, DepthBig, Wide, Single>(a, t, a.disp()->count, xs, rowbuf, epoch);
            return;
        }
        t -= dn_x;
    }
    const unsigned gu_n = gu_x * slots * gu_z;
    if (t < gu_n) {
        gate_up4_cta<MaxR, Depth, W, DepthBig, true, Wide, Single>(a, t % gu_x, (t / gu_x) % slots, t / (gu_x * slots), xs, rowbuf,
                                                                  epoch);
        return;
    }
    t -= gu_n;
    if (SharedFirst && t / dn_x == a.disp()->count) return;  // shared down already ran first
    down4_cta<MaxR, Depth, W, DepthBig, Wide, Single>(a, t % dn_x, t / dn_x, xs, rowbuf, epoch);
}
};
}  // namespace strix::moe_core
