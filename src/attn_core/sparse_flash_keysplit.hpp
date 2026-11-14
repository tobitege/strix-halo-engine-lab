#pragma once
// Shared FAST sparse attention core, key-split form: per-row key lists made of segments (MiniMax MSA: the selected
// 128-key blocks of a (token, KV group)), flash form, 16-bit WMMA 16x16x16 -> FP32, FP32 online softmax.
//
// Identity: FAST, per engine. Nothing here runs unless an engine launches it, so the exact paths keep their bytes.
// Lossy against the exact kernels: queries and probabilities are rounded to the operand type (RNE), the score and PV
// sums run in WMMA order, the softmax is the online form with exp2, and the four per-wave streams are merged once.
// Keys/values are used as stored (bf16 caches).
// Split = true (two-term operands): queries and probabilities are carried as hi + lo, each rounded to the operand
// type (lo = the rounding remainder of hi), and both products run once per term (32 WMMA per tile instead of 16).
// The operand rounding error drops from 2^-9 to about 2^-17 relative; the result is still not the exact kernel's
// bytes (summation order, exp2).
//
// Shape: one work-group = one query row (16 query heads) = 4 waves. The row's key list is cut into 16-key tiles and
// wave w owns tiles w, w + 4, ...: a complete flash step per tile inside the wave, no barrier in the key loop.
// Every lane pair (ln, ln + 16) belongs to head ln:
//   QK   S^T[key][head] = K[key][D] . Q^T[D][head]: A fragments straight from the key rows (lane = key), B = the
//        head's query kept in registers (lane = head). Lane (hf, ln) receives keys 2e + hf of head ln.
//   soft the head's 16 scores sit in its two lanes: 8-register max / sum + one cross-half shuffle
//        (tile::online_step, Lanes 2, Stride 16); P[head][16 keys] is packed in the lane (4 shuffles).
//   PV   O^T[dim][head] += V^T[dim][key] . P^T[key][head]: A = V^T rows from a wave-private LDS tile (the wave
//        gathers its 16 value rows as 4 keys x 8 dims pieces, tile::store_vt4x8; a wave reads back only its own
//        stores), B = P. The accumulators hold column = head = lane, so the online rescale is a per-lane multiply.
// Epilogue (once per work-group): the four (m, l, O) states are merged in wave order through LDS (fixed order,
// deterministic), then written as FP32 [H][D].
// LDS: 4 x D x 48 B (V^T, 16-B pad per row) + 512 B (m, l) = 25,088 B at D 128; the query staging (H x D x 2 B) and
// the merge buffer (H x D x 4 B) reuse the V^T area before / after the key loop. Split adds H x D x 2 B (the query
// lo terms stay in LDS and are read per tile; the hi terms stay in registers).
//
// Policy P:
//   using Args;                                   kernel argument (trivially copyable)
//   struct Row;                                   per-work-group context filled by begin()
//   static constexpr unsigned kHeads, kDim;       16, 128
//   static constexpr unsigned kSegKeys;           keys per segment (multiple of 16)
//   static constexpr tile::Operand kOp;           operand type (bf16 caches: bf16)
//   static constexpr float kScaleLog2;            softmax scale x log2(e)
//   static __device__ bool begin(const Args&, unsigned work_group, Row&)   false: nothing to do, output untouched;
//                                                                          true: at least one visible key
//   static __device__ unsigned segments(const Args&, const Row&)
//   static __device__ unsigned seg_keys(const Args&, const Row&, unsigned c)                  1 .. kSegKeys
//   static __device__ const std::uint16_t* key_row(const Args&, const Row&, unsigned c, unsigned k)    [D], 16-B aligned
//   static __device__ const std::uint16_t* value_row(const Args&, const Row&, unsigned c, unsigned k)  [D], 16-B aligned
//   static __device__ void query(const Args&, const Row&, unsigned head, unsigned lane, float (&x)[4])
//                                                 called by a whole wave: lane owns dims 4 lane .. 4 lane + 3 (FP32,
//                                                 final values; the policy may use wave shuffles, e.g. a norm)
//   static __device__ float* out(const Args&, const Row&)                                     FP32 [H][D]
// Launch: flash_rows_keysplit_kernel<P, Split><<<work_groups, KeySplit<P::kHeads, P::kDim>::kThreads, 0, stream>>>(args).
#include "attn_core/flash_tile.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::attn_core::sparse {

template <unsigned H, unsigned D> struct KeySplit {
    static_assert(H == 16, "lane = head: one 16-head tile");
    static_assert(D == 128, "query staging: 4 dims per lane");
    static constexpr unsigned kWaves = 4, kThreads = 32 * kWaves, kT = 16;
    static constexpr unsigned kKs = D / 16;                 // QK k steps = output dim tiles
    static constexpr unsigned kPitch = 2 * kT + 16;         // V^T row: 16 keys + 16-B pad
    static constexpr unsigned kVtWave = D * kPitch;         // one wave's V^T tile
    static constexpr unsigned kMl = kWaves * kVtWave;       // (m, l) x 16 heads x 4 waves
    static constexpr unsigned kQlo = kMl + kWaves * H * 2 * 4;  // Split: query lo terms [H][D]
    static constexpr unsigned bytes(bool split) { return kQlo + (split ? H * D * 2 : 0U); }
    static_assert(H * D * 2 <= kMl && H * D * 4 <= kMl, "query staging and merge buffer fit the V^T area");
    // Merge buffer slot (FP32 index) of output (head, dim): lane (hf = dim & 1, ln = head) owns 8 x kKs floats.
    static constexpr unsigned merge_slot(unsigned head, unsigned dim) {
        return (((dim & 1U) * 16U + head) * kKs + dim / 16U) * 8U + (dim % 16U) / 2U;
    }
    // Every (head, dim) owns its own slot and the slots fill the buffer (checked at compile time).
    static constexpr bool merge_slots_cover() {
        bool seen[H * D] = {};
        for (unsigned h = 0; h < H; ++h)
            for (unsigned d = 0; d < D; ++d) {
                const unsigned s = merge_slot(h, d);
                if (s >= H * D || seen[s]) return false;
                seen[s] = true;
            }
        return true;
    }
};

template <class P, bool Split = false>
__global__ __launch_bounds__((KeySplit<P::kHeads, P::kDim>::kThreads)) void flash_rows_keysplit_kernel(typename P::Args a) {
    using G = KeySplit<P::kHeads, P::kDim>;
    static_assert(G::merge_slots_cover(), "merge buffer layout");
    using tile::Acc;
    using tile::Frag;
    using tile::U2;
    using tile::U4;
    constexpr unsigned H = P::kHeads, D = P::kDim, T = G::kT, TPS = P::kSegKeys / T;
    constexpr tile::Operand O = P::kOp;
    static_assert(P::kSegKeys % T == 0, "segments are whole tiles");
    __shared__ __attribute__((aligned(16))) unsigned char lds[G::bytes(Split)];
    typename P::Row row;
    if (!P::begin(a, blockIdx.x, row)) return;
    const unsigned tid = threadIdx.x, w = tid >> 5U, lane = tid & 31U, ln = lane & 15U, hf = lane >> 4U;
    // Queries: wave w prepares heads w, w + 4, ... (lane = 4 dims), rounded once to the operand type.
#pragma unroll
    for (unsigned i = 0; i < H / G::kWaves; ++i) {
        const unsigned head = i * G::kWaves + w;
        float x[4];
        P::query(a, row, head, lane, x);
        *reinterpret_cast<U2*>(lds + (head * D + lane * 4U) * 2U) = U2{tile::pack2<O>(x[0], x[1]), tile::pack2<O>(x[2], x[3])};
        if constexpr (Split) {
#pragma unroll
            for (unsigned d = 0; d < 4; ++d) x[d] -= tile::operand_value<O>(tile::to_operand<O>(x[d]));
            *reinterpret_cast<U2*>(lds + G::kQlo + (head * D + lane * 4U) * 2U) =
                U2{tile::pack2<O>(x[0], x[1]), tile::pack2<O>(x[2], x[3])};
        }
    }
    __syncthreads();
    Frag qf[G::kKs];
#pragma unroll
    for (unsigned k = 0; k < G::kKs; ++k) qf[k] = tile::frag32(lds + (ln * D + 16U * k) * 2U);
    __syncthreads();  // the V^T tiles reuse the staging area
    Acc o[G::kKs];
#pragma unroll
    for (unsigned c = 0; c < G::kKs; ++c) o[c] = tile::acc_zero();
    tile::SoftmaxState st;
    unsigned char* const vt = lds + w * G::kVtWave;
    const unsigned kg = lane & 3U, dg = lane >> 2U;  // gather: 4 keys x 8 dims, two passes (dims 8 dg, 8 (dg + 8))
    const unsigned tiles = P::segments(a, row) * TPS;
#pragma unroll 1
    for (unsigned f = w; f < tiles; f += G::kWaves) {
        const unsigned c = f / TPS, k0 = (f % TPS) * T;
        const unsigned nk = P::seg_keys(a, row, c);
        if (k0 >= nk) continue;  // wave-uniform: tile past the segment's visible keys
        // 1. value pieces (in flight during QK); keys past the visible count contribute zeros
        U4 gv[2][4];
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const unsigned k = k0 + 4U * kg + i;
            const bool ok = k < nk;
            const std::uint16_t* vr = P::value_row(a, row, c, ok ? k : nk - 1U);
#pragma unroll
            for (unsigned p = 0; p < 2; ++p) {
                const U4 x = *reinterpret_cast<const U4*>(vr + 8U * (dg + 8U * p));
                gv[p][i] = ok ? x : U4{0U, 0U, 0U, 0U};
            }
        }
        // 2. QK: S^T[key 2e + hf][head ln]
        Acc s = tile::acc_zero();
        {
            const unsigned k = k0 + ln;
            const std::uint16_t* kr = P::key_row(a, row, c, k < nk ? k : nk - 1U);
#pragma unroll
            for (unsigned x = 0; x < G::kKs; ++x) {
                const Frag kf = tile::frag32(kr + 16U * x);
                if constexpr (Split) s = tile::mma<O>(kf, tile::frag32(lds + G::kQlo + (ln * D + 16U * x) * 2U), s);
                s = tile::mma<O>(kf, qf[x], s);
            }
        }
        // 3. V^T tile of this wave
#pragma unroll
        for (unsigned p = 0; p < 2; ++p) tile::store_vt4x8<G::kPitch>(vt, gv[p], 8U * (dg + 8U * p), kg);
        // 4. online softmax of head ln over the tile's 16 keys (8 here, 8 in the other half)
        float sv[8];
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) sv[e] = k0 + 2U * e + hf < nk ? s[e] * P::kScaleLog2 : -__builtin_huge_valf();
        std::uint16_t pb[8], pl[8];
        float alpha;
        if constexpr (Split) {
            alpha = tile::online_step_split<O, 8, 2, 16>(sv, st, pb, pl);
        } else {
            alpha = tile::online_step<O, 8, 2, tile::SumOf::rounded, 16>(sv, st, pb);
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) pl[e] = 0;
        }
        // 5. P[head ln][16 keys]: element 2e + hf is this lane's, 2e + 1 - hf the other half's
        const auto p_frag = [&](const std::uint16_t (&v)[8]) {
            unsigned word[8];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const unsigned own = v[2U * i] | (static_cast<unsigned>(v[2U * i + 1U]) << 16U);
                const unsigned other = static_cast<unsigned>(__shfl_xor(static_cast<int>(own), 16));
                const unsigned o0 = own & 0xffffU, o1 = own >> 16U, p0 = other & 0xffffU, p1 = other >> 16U;
                word[2U * i] = hf == 0 ? o0 | (p0 << 16U) : p0 | (o0 << 16U);
                word[2U * i + 1U] = hf == 0 ? o1 | (p1 << 16U) : p1 | (o1 << 16U);
            }
            Frag f;
            __builtin_memcpy(&f, word, 32U);
            return f;
        };
        const Frag pf = p_frag(pb);
        Frag pf_lo{};
        if constexpr (Split) pf_lo = p_frag(pl);
        // 6. PV: O^T[dim 16 x + 2e + hf][head ln]
#pragma unroll
        for (unsigned x = 0; x < G::kKs; ++x) {
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) o[x][e] *= alpha;
            const Frag vf = tile::frag32(vt + (16U * x + ln) * G::kPitch);
            o[x] = tile::mma<O>(vf, pf, o[x]);
            if constexpr (Split) o[x] = tile::mma<O>(vf, pf_lo, o[x]);
        }
    }
    // Merge the four streams (wave order): M = max m, L = sum l 2^(m - M), O = sum O 2^(m - M) / L.
    float* const ml = reinterpret_cast<float*>(lds + G::kMl);
    if (hf == 0) {
        ml[(w * H + ln) * 2U] = st.m;
        ml[(w * H + ln) * 2U + 1U] = st.l;
    }
    __syncthreads();
    float big = -__builtin_huge_valf();
#pragma unroll
    for (unsigned i = 0; i < G::kWaves; ++i) {
        const float m = ml[(i * H + ln) * 2U];
        big = m > big ? m : big;
    }
    float total = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < G::kWaves; ++i) {
        const float l = ml[(i * H + ln) * 2U + 1U];
        // a wave without tiles has m = -inf, l = 0
        total += l == 0.0F ? 0.0F : l * __builtin_amdgcn_exp2f(ml[(i * H + ln) * 2U] - big);
    }
    const float scale = st.l == 0.0F ? 0.0F : __builtin_amdgcn_exp2f(st.m - big) / total;
    float* const sum = reinterpret_cast<float*>(lds) + static_cast<std::size_t>(lane) * (8U * G::kKs);
#pragma unroll 1
    for (unsigned i = 0; i < G::kWaves; ++i) {
        if (w == i) {
#pragma unroll
            for (unsigned x = 0; x < G::kKs; ++x) {
                float4 lo = make_float4(o[x][0] * scale, o[x][1] * scale, o[x][2] * scale, o[x][3] * scale);
                float4 hi = make_float4(o[x][4] * scale, o[x][5] * scale, o[x][6] * scale, o[x][7] * scale);
                float4* slot = reinterpret_cast<float4*>(sum + 8U * x);
                if (i != 0) {
                    const float4 a0 = slot[0], a1 = slot[1];
                    lo = make_float4(a0.x + lo.x, a0.y + lo.y, a0.z + lo.z, a0.w + lo.w);
                    hi = make_float4(a1.x + hi.x, a1.y + hi.y, a1.z + hi.z, a1.w + hi.w);
                }
                slot[0] = lo;
                slot[1] = hi;
            }
        }
        __syncthreads();
    }
    float* const out = P::out(a, row);
    const float* const merged = reinterpret_cast<const float*>(lds);
#pragma unroll
    for (unsigned j = 0; j < H * D / 4U / G::kThreads; ++j) {
        const unsigned q = tid + j * G::kThreads, head = q / (D / 4U), d = (q % (D / 4U)) * 4U;
        *reinterpret_cast<float4*>(out + static_cast<std::size_t>(head) * D + d) =
            make_float4(merged[G::merge_slot(head, d)], merged[G::merge_slot(head, d + 1U)], merged[G::merge_slot(head, d + 2U)],
                        merged[G::merge_slot(head, d + 3U)]);
    }
}

}  // namespace strix::attn_core::sparse
