#pragma once
// Shared FAST sparse attention core, row-group form of the key-split flash (sparse_flash_keysplit.hpp): one
// work-group = kRows neighbouring query rows of one KV group (16 heads each), one wave per row. The rows' key lists
// (ascending segments, the same Policy hooks as the key-split form) are merged into one union list; each 16-key
// tile of the union is loaded ONCE into LDS and consumed by every wave whose row selected that segment.
//
// Why: the key-split form moves 8 KB of K + V from L2 into the CU per (row, tile); at the MiniMax MSA prefill shape
// that byte stream is the bound (ATT: wait:vm 57-64 % of latency; cache reuse, load latency and line spread do not
// move it, scripts/bench/mmattn). Neighbouring rows share most of their blocks (dense windows: all of them), so the
// bytes per row-tile drop to 8 KB x union / (kRows x list).
//
// Identity: FAST (same class as the key-split form: bf16 queries and probabilities, WMMA sums, exp2 online softmax),
// but NOT the key-split form's bytes: one softmax stream per row over its tiles in ascending order instead of four
// wave streams merged at the end. Engines gate it like any FAST arithmetic change.
//
// Per tile (u, k0) of the union, stage s = tile & 1 (register-staged double buffer, one barrier per tile):
//   waves 0-1 (64 threads) load the V pieces (4 keys x 8 dims each, tile::store_vt4x8 -> V^T [dim][16 keys]),
//   waves 2-3 (64 threads) load the K rows (4 x 16 B each -> K [16 keys][128 dims], 16-B row pad);
//   wave w (row w): if its row selected segment u and k0 < its visible key count: QK (A = K rows from LDS, B = its
//   queries in registers), online softmax over its own visible keys, PV (A = V^T from LDS) - the key-split form's
//   fragment layouts (lane pair (ln, ln + 16) = head ln).
// Keys past the largest visible count of the tile are clamped (K) / zeroed (V); a row's keys past its own count are
// masked to -inf, so their probabilities are exactly 0.
// Union (prologue): segment identity = P::key_row(c, 0) (equal pointer <=> same keys and values); ranks from a
// duplicate pass and a counting pass over the <= kRows x kMaxSegs candidates in LDS (three barriers).
//
// Policy P: the key-split policy plus
//   static constexpr unsigned kMaxSegs;           segments per row bound (the launcher guarantees it; checked, traps)
//   static __device__ unsigned member(const Args&, unsigned group, unsigned i, unsigned rows)
//                                                 work-group index of row i of row group `group` for P::begin, or ~0U
//   static __device__ std::size_t key_stride(const Args&)   elements between keys k and k + 1 of a segment (K and V)
// Launch: flash_rowgroup_kernel<P><<<row_groups, RowGroup<...>::kThreads, 0, stream>>>(args).
#include "attn_core/sparse_flash_keysplit.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::attn_core::sparse {

// Rows: query rows per work-group (one wave each); 4 or 8. Every wave runs its own row's tiles in ascending order with
// the same expressions whatever Rows is, so the output bits do not depend on Rows (only the union and the schedule do).
template <unsigned H, unsigned D, unsigned Segs, unsigned Rows = 4> struct RowGroup {
    static_assert(H == 16 && D == 128, "key-split fragment layouts");
    static_assert(Rows == 4 || Rows == 8, "waves 0-3 load the tiles");
    static constexpr unsigned kRows = Rows, kWaves = kRows, kThreads = 32 * kWaves, kT = 16, kKs = D / 16;
    static constexpr unsigned kKPitch = 2 * D + 16;    // K row: 128 dims + 16-B pad (272 B: 8 lanes cover 32 banks)
    static constexpr unsigned kVPitch = 2 * kT + 16;   // V^T row: 16 keys + 16-B pad (key-split pitch)
    static constexpr unsigned kKBytes = kT * kKPitch, kVBytes = D * kVPitch, kStage = kKBytes + kVBytes;
    static constexpr unsigned kCand = kRows * Segs;
    static constexpr unsigned kQBytes = kRows * H * D * 2;       // query staging (prologue only), aliases the stages
    static constexpr unsigned kCv = kQBytes > 2 * kStage ? kQBytes : 2 * kStage;  // u64 candidate identity [kCand]
    static constexpr unsigned kKp = kCv + 8 * kCand;             // u64 union K segment base [kCand]
    static constexpr unsigned kVp = kKp + 8 * kCand;             // u64 union V segment base [kCand]
    static constexpr unsigned kNk = kVp + 8 * kCand;             // u16 visible keys [union][row], 0 = not selected
    static constexpr unsigned kDup = kNk + 2 * kRows * kCand;    // u32 duplicate flag [kCand]
    static constexpr unsigned kCnt = kDup + 4 * kCand;           // u32 union length
    static constexpr unsigned kBytes = kCnt + 16;
    static_assert(kRows * kCand % 2 == 0, "visible-key table cleared as u32");
    static_assert(Segs <= 32, "one candidate per lane");
};

// Opt bits (schedule / exactness notes per bit):
//   kRgLazyRescale  skip the accumulator rescale when no lane's alpha differs from 1 (exp2(0) = 1: same bits)
//   kRgQkPair       QK in two accumulator chains (even / odd 16-dim steps) added once: half the dependent WMMA chain,
//                   different score rounding (FAST: changes the bits of the default form)
inline constexpr unsigned kRgLazyRescale = 1U, kRgQkPair = 2U;

namespace rowgroup_detail {
// P[head ln][16 keys] B fragment from the lane's 8 probabilities (key 2e + hf) and the other half's (key-split form).
__device__ __forceinline__ tile::Frag p_frag(const std::uint16_t (&v)[8], unsigned hf) {
    unsigned word[8];
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned own = v[2U * i] | (static_cast<unsigned>(v[2U * i + 1U]) << 16U);
        const unsigned other = static_cast<unsigned>(__shfl_xor(static_cast<int>(own), 16));
        const unsigned o0 = own & 0xffffU, o1 = own >> 16U, p0 = other & 0xffffU, p1 = other >> 16U;
        word[2U * i] = hf == 0 ? o0 | (p0 << 16U) : p0 | (o0 << 16U);
        word[2U * i + 1U] = hf == 0 ? o1 | (p1 << 16U) : p1 | (o1 << 16U);
    }
    tile::Frag f;
    __builtin_memcpy(&f, word, 32U);
    return f;
}
}  // namespace rowgroup_detail

template <class P, unsigned Opt = 0U, unsigned Rows = 4U>
__global__ __launch_bounds__((RowGroup<P::kHeads, P::kDim, P::kMaxSegs, Rows>::kThreads)) void flash_rowgroup_kernel(typename P::Args a) {
    using G = RowGroup<P::kHeads, P::kDim, P::kMaxSegs, Rows>;
    using tile::Acc;
    using tile::Frag;
    using tile::U2;
    using tile::U4;
    constexpr unsigned H = P::kHeads, D = P::kDim, T = G::kT, S = P::kMaxSegs;
    constexpr tile::Operand O = P::kOp;
    static_assert(P::kSegKeys % T == 0, "segments are whole tiles");
    __shared__ __attribute__((aligned(16))) unsigned char lds[G::kBytes];
    auto* const cv = reinterpret_cast<std::uint64_t*>(lds + G::kCv);
    auto* const kp = reinterpret_cast<std::uint64_t*>(lds + G::kKp);
    auto* const vp = reinterpret_cast<std::uint64_t*>(lds + G::kVp);
    auto* const nkt = reinterpret_cast<std::uint16_t*>(lds + G::kNk);
    auto* const dup = reinterpret_cast<unsigned*>(lds + G::kDup);
    auto* const cnt_u = reinterpret_cast<unsigned*>(lds + G::kCnt);
    const unsigned tid = threadIdx.x, w = tid >> 5U, lane = tid & 31U, ln = lane & 15U, hf = lane >> 4U;
    typename P::Row row;
    const unsigned wg = P::member(a, blockIdx.x, w, G::kRows);
    const bool act = wg != ~0U && P::begin(a, wg, row);  // wave-uniform
    const unsigned cnt = act ? P::segments(a, row) : 0U;
    if (cnt > S) __builtin_trap();
    // 1. candidates (lane c = segment c of the wave's row), queries staged per wave, visible-key table cleared
    for (unsigned i = tid; i < G::kRows * G::kCand / 2U; i += G::kThreads) reinterpret_cast<unsigned*>(nkt)[i] = 0U;
    if (tid == 0) *cnt_u = 0U;
    std::uint64_t mine = ~0ULL;
    unsigned mine_nk = 0;
    if (lane < S) {
        if (lane < cnt) {
            mine = reinterpret_cast<std::uint64_t>(P::key_row(a, row, lane, 0U));
            mine_nk = P::seg_keys(a, row, lane);
        }
        cv[w * S + lane] = mine;
    }
    if (act) {
#pragma unroll 1
        for (unsigned head = 0; head < H; ++head) {
            float x[4];
            P::query(a, row, head, lane, x);
            *reinterpret_cast<U2*>(lds + ((w * H + head) * D + lane * 4U) * 2U) =
                U2{tile::pack2<O>(x[0], x[1]), tile::pack2<O>(x[2], x[3])};
        }
    }
    __syncthreads();
    // 2. duplicates (the segment is already a candidate of an earlier row) and the query fragments
    bool first = true;
    if (lane < S) {
        if (mine != ~0ULL) {
#pragma unroll 1
            for (unsigned j = 0; j < w * S; ++j) first = first && cv[j] != mine;
        }
        dup[w * S + lane] = mine != ~0ULL && first ? 0U : 1U;
    }
    Frag qf[G::kKs];
#pragma unroll
    for (unsigned k = 0; k < G::kKs; ++k) qf[k] = tile::frag32(lds + ((w * H + ln) * D + 16U * k) * 2U);
    __syncthreads();
    // 3. ranks: distinct candidates below this one (ascending identity = ascending keys within each row)
    if (lane < S && mine != ~0ULL) {
        unsigned rank = 0;
#pragma unroll 1
        for (unsigned j = 0; j < G::kCand; ++j) rank += cv[j] < mine && dup[j] == 0U ? 1U : 0U;
        nkt[rank * G::kRows + w] = static_cast<std::uint16_t>(mine_nk);
        if (first) {
            kp[rank] = mine;
            vp[rank] = reinterpret_cast<std::uint64_t>(P::value_row(a, row, lane, 0U));
            atomicMax(cnt_u, rank + 1U);
        }
    }
    __syncthreads();
    const unsigned nu = *cnt_u;
    if (nu == 0) return;  // no row with a visible key: outputs untouched
    const std::size_t ks = P::key_stride(a);
    const bool vload = tid < 64U, kload = tid >= 64U && tid < 128U;  // waves 4+ (Rows 8) only compute
    const unsigned t2 = tid & 63U, kg = t2 & 3U, dg = t2 >> 2U;  // V: keys 4 kg .. + 3, dims 8 dg .. + 7
    const auto keys_max = [&](unsigned u) {
        const unsigned* p = reinterpret_cast<const unsigned*>(nkt + u * G::kRows);
        unsigned m = 0;
#pragma unroll
        for (unsigned i = 0; i < G::kRows / 2U; ++i) {
            const unsigned x = p[i], lo = x & 0xffffU, hi = x >> 16U;
            m = lo > m ? lo : m;
            m = hi > m ? hi : m;
        }
        return m;
    };
    U4 stg[4];
    const auto fetch = [&](unsigned u, unsigned k0, unsigned nm) {
        if (vload) {
            const auto* base = reinterpret_cast<const std::uint16_t*>(vp[u]);
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const unsigned k = k0 + 4U * kg + i;
                const bool ok = k < nm;
                const U4 x = *reinterpret_cast<const U4*>(base + static_cast<std::size_t>(ok ? k : nm - 1U) * ks + 8U * dg);
                stg[i] = ok ? x : U4{0U, 0U, 0U, 0U};
            }
        } else if (kload) {
            const auto* base = reinterpret_cast<const std::uint16_t*>(kp[u]);
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const unsigned k = k0 + (t2 >> 4U) + 4U * i;
                stg[i] = *reinterpret_cast<const U4*>(base + static_cast<std::size_t>(k < nm ? k : nm - 1U) * ks + 8U * (t2 & 15U));
            }
        }
    };
    const auto put = [&](unsigned stage) {
        unsigned char* const buf = lds + stage * G::kStage;
        if (vload) {
            tile::store_vt4x8<G::kVPitch>(buf + G::kKBytes, stg, 8U * dg, kg);
        } else if (kload) {
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
                *reinterpret_cast<U4*>(buf + ((t2 >> 4U) + 4U * i) * G::kKPitch + 16U * (t2 & 15U)) = stg[i];
        }
    };
    Acc o[G::kKs];
#pragma unroll
    for (unsigned c = 0; c < G::kKs; ++c) o[c] = tile::acc_zero();
    tile::SoftmaxState st;
    unsigned u = 0, k0 = 0, nm = keys_max(0);
    fetch(u, k0, nm);
    put(0);
    __syncthreads();
#pragma unroll 1
    for (unsigned t = 0;; ++t) {
        unsigned u1 = u, k1 = k0 + T, nm1 = nm;
        if (k1 >= nm) {
            u1 = u + 1U;
            k1 = 0;
            nm1 = u1 < nu ? keys_max(u1) : 0U;
        }
        const bool more = u1 < nu;
        if (more) fetch(u1, k1, nm1);  // in flight during this tile's math
        const unsigned own = nkt[u * G::kRows + w];
        if (act && k0 < own) {
            const unsigned char* const buf = lds + (t & 1U) * G::kStage;
            Acc s = tile::acc_zero();
            if constexpr ((Opt & kRgQkPair) != 0U) {
                Acc s1 = tile::acc_zero();
#pragma unroll
                for (unsigned x = 0; x < G::kKs; x += 2U) {
                    s = tile::mma<O>(tile::frag32(buf + ln * G::kKPitch + 32U * x), qf[x], s);
                    s1 = tile::mma<O>(tile::frag32(buf + ln * G::kKPitch + 32U * (x + 1U)), qf[x + 1U], s1);
                }
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) s[e] += s1[e];
            } else {
#pragma unroll
                for (unsigned x = 0; x < G::kKs; ++x) s = tile::mma<O>(tile::frag32(buf + ln * G::kKPitch + 32U * x), qf[x], s);
            }
            float sv[8];
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) sv[e] = k0 + 2U * e + hf < own ? s[e] * P::kScaleLog2 : -__builtin_huge_valf();
            std::uint16_t pb[8];
            const float alpha = tile::online_step<O, 8, 2, tile::SumOf::rounded, 16>(sv, st, pb);
            const Frag pf = rowgroup_detail::p_frag(pb, hf);
            const bool rescale = (Opt & kRgLazyRescale) == 0U || __builtin_amdgcn_ballot_w32(alpha != 1.0F) != 0U;
#pragma unroll
            for (unsigned x = 0; x < G::kKs; ++x) {
                if (rescale) {
#pragma unroll
                    for (unsigned e = 0; e < 8; ++e) o[x][e] *= alpha;
                }
                o[x] = tile::mma<O>(tile::frag32(buf + G::kKBytes + (16U * x + ln) * G::kVPitch), pf, o[x]);
            }
        }
        if (!more) break;
        put((t + 1U) & 1U);
        __syncthreads();
        u = u1;
        k0 = k1;
        nm = nm1;
    }
    if (!act) return;
    // O^T[dim 16 x + 2 e + hf][head ln] / l -> FP32 [H][D] (a head without visible keys writes zeros, as the key-split form)
    const float scale = st.l == 0.0F ? 0.0F : 1.0F / st.l;
    float* const out = P::out(a, row) + static_cast<std::size_t>(ln) * D + hf;
#pragma unroll
    for (unsigned x = 0; x < G::kKs; ++x) {
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) out[16U * x + 2U * e] = o[x][e] * scale;
    }
}

}  // namespace strix::attn_core::sparse
