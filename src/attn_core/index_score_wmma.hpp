#pragma once
// Shared FAST indexer score core for sparse-attention indexers (MiniMax MSA prefill, GLM DSA prefill).
//
// Identity: FAST, opt-in (STRIX_IDX_WMMA=1). Off (unset or 0) launches nothing from this file, so the exact and
// default paths keep their bytes. On, each (query vector, key) dot runs on the 16x16x16 WMMA units with 16-bit
// operands and FP32 accumulation; the queries are rounded to the operand type (round to nearest even) and the
// accumulation order changes. Keys that are already BF16 (MiniMax) stay exact; FP32 keys (GLM) are rounded too.
//
// Shape: one work-group = 4 waves = 128 query vectors (wave = 2 fragments of 16 query vectors, kept in registers
// for the whole launch) x key tiles of 128 keys staged once per tile in a 32 KiB swizzled LDS tile. Every key
// fragment read from LDS feeds both query fragments of its wave. The work-group walks key tiles blockIdx.x,
// blockIdx.x + gridDim.x, ...; no score matrix is materialised: the engine epilogue reduces each 16x16 result tile
// in registers (MiniMax: max over the block's keys; GLM: ReLU x head weight x head sum).
//
// An engine adapter is a policy type P with
//   using Args;                                     kernel argument (trivially copyable)
//   struct Ctx;  struct State;                      per work-group constants / per-lane epilogue state
//   static constexpr Operand kOp;
//   bool begin(const Args&, Ctx&)                   false: the whole work-group has nothing to score (uniform)
//   void stage_queries(const Args&, const Ctx&, std::uint16_t* tile)   write query vector q < 128, element d with
//                                                   tile[slot(q, d)] (all threads cooperate; operand bits)
//   void init(const Args&, const Ctx&, State&)      after the query fragments are loaded
//   unsigned tiles(const Args&, const Ctx&)         key tiles of the work-group
//   bool live(const Args&, const Ctx&, unsigned kt) uniform skip
//   uint4 key_piece(const Args&, const Ctx&, unsigned kt, unsigned key, unsigned piece)  8 operand elements
//   void tile_begin(const Args&, const Ctx&, State&, unsigned kt)
//   void fragment(const Args&, const Ctx&, State&, unsigned kt, unsigned kf, const Acc (&c)[2])
//       c[f][e] = dot(query vector wave * 32 + f * 16 + 2 e + lane / 16, key kt * 128 + kf * 16 + lane % 16)
//   void tile_end(const Args&, const Ctx&, State&, unsigned kt)
#include <hip/hip_runtime.h>

#include "attn_core/index_score_mode.hpp"
#include "attn_core/wmma_frag.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <iterator>
#include <string>
#include <vector>

namespace strix::attn_core::idx {

// Operand types, rounding, packing, the WMMA call and the swizzled-tile fragment load are the canonical tile::
// helpers (wmma_types.hpp / wmma_frag.hpp); this file binds them to the indexer's 128 x 128 key tile.
using tile::Acc;
using tile::Frag;
using tile::mma;
using tile::Operand;
using tile::to_bf16;
using tile::to_f16;
using tile::to_operand;

static_assert(static_cast<unsigned>(Operand::bf16) == 1U && static_cast<unsigned>(Operand::f16) == 2U,
              "wmma_mode() (index_score_mode.hpp) returns tile::Operand values");
// Key-tile split of the launch grid (blockIdx.x); STRIX_IDX_WMMA_SPLIT overrides (1..1024, tuning only).
inline unsigned wmma_split(unsigned tiles, unsigned dflt = 16U) {
    unsigned s = dflt;
    if (const char* e = std::getenv("STRIX_IDX_WMMA_SPLIT"); e != nullptr && *e != '\0') {
        const unsigned long v = std::strtoul(e, nullptr, 10);
        if (v < 1 || v > 1024) throw std::invalid_argument("STRIX_IDX_WMMA_SPLIT must be 1..1024");
        s = static_cast<unsigned>(v);
    }
    return tiles < s ? (tiles == 0 ? 1U : tiles) : s;
}

constexpr unsigned kWaves = 4, kFrags = 2, kThreads = kWaves * 32;
constexpr unsigned kKeys = 128, kDim = 128, kPieces = kDim / 8;  // 16-byte pieces of a 128-element row
constexpr unsigned kQueries = kWaves * kFrags * 16;               // 128 query vectors per work-group
static_assert(kQueries == kKeys, "the query vectors are staged through the key tile");
constexpr unsigned kTileBytes = kKeys * kDim * 2;                 // 32 KiB

// The key tile is tile::'s XOR-swizzled layout with kPieces pieces per row (see wmma_frag.hpp).
__host__ __device__ constexpr unsigned piece_slot(unsigned row, unsigned piece) { return tile::piece_slot(row, piece, kPieces); }
__host__ __device__ constexpr unsigned slot(unsigned row, unsigned d) { return tile::slot(row, d, kPieces); }
// tile::pack8 as the uint4 the policies' key_piece returns.
template <Operand O> __device__ __forceinline__ uint4 pack8(float4 lo, float4 hi) { return __builtin_bit_cast(uint4, tile::pack8<O>(lo, hi)); }

template <class P> __global__ __launch_bounds__(kThreads) void score_kernel(typename P::Args args) {
    __shared__ __align__(16) tile::U4 lds[kKeys * kPieces];
    const unsigned tid = threadIdx.x, lane = tid % 32U, wave = tid / 32U;
    typename P::Ctx ctx;
    if (!P::begin(args, ctx)) return;  // uniform, before any barrier
    P::stage_queries(args, ctx, reinterpret_cast<std::uint16_t*>(lds));
    __syncthreads();
    Frag a[kFrags][kDim / 16];
#pragma unroll
    for (unsigned f = 0; f < kFrags; ++f)
#pragma unroll
        for (unsigned s = 0; s < kDim / 16; ++s) a[f][s] = tile::frag_swz(lds, wave * kFrags * 16U + f * 16U + lane % 16U, s, kPieces);
    typename P::State st;
    P::init(args, ctx, st);
    __syncthreads();  // all query fragments are in registers before the first key tile overwrites LDS
    const unsigned tiles = P::tiles(args, ctx);
    for (unsigned kt = blockIdx.x; kt < tiles; kt += gridDim.x) {
        if (!P::live(args, ctx, kt)) continue;  // uniform
        {
            constexpr unsigned kPer = kKeys * kPieces / kThreads;
            uint4 pre[kPer];
#pragma unroll
            for (unsigned i = 0; i < kPer; ++i) {
                const unsigned e = i * kThreads + tid;
                pre[i] = P::key_piece(args, ctx, kt, e / kPieces, e % kPieces);
            }
#pragma unroll
            for (unsigned i = 0; i < kPer; ++i) {
                const unsigned e = i * kThreads + tid;
                lds[piece_slot(e / kPieces, e % kPieces)] = __builtin_bit_cast(tile::U4, pre[i]);
            }
        }
        __syncthreads();
        P::tile_begin(args, ctx, st, kt);
#pragma unroll 1
        for (unsigned kf = 0; kf < kKeys / 16U; kf += 2U) {
            Acc c0[kFrags], c1[kFrags];
#pragma unroll
            for (unsigned f = 0; f < kFrags; ++f) { c0[f] = Acc{}; c1[f] = Acc{}; }
#pragma unroll
            for (unsigned s = 0; s < kDim / 16; ++s) {
                const Frag b0 = tile::frag_swz(lds, kf * 16U + lane % 16U, s, kPieces);
                const Frag b1 = tile::frag_swz(lds, (kf + 1U) * 16U + lane % 16U, s, kPieces);
#pragma unroll
                for (unsigned f = 0; f < kFrags; ++f) {
                    c0[f] = mma<P::kOp>(a[f][s], b0, c0[f]);
                    c1[f] = mma<P::kOp>(a[f][s], b1, c1[f]);
                }
            }
            P::fragment(args, ctx, st, kt, kf, c0);
            P::fragment(args, ctx, st, kt, kf + 1U, c1);
        }
        P::tile_end(args, ctx, st, kt);
        __syncthreads();  // every wave is done with this tile before the next one is staged
    }
}

template <class P> inline void launch(const typename P::Args& args, unsigned split, unsigned groups, hipStream_t stream) {
    if (split == 0 || groups == 0) return;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(score_kernel<P>), dim3(split, groups), dim3(kThreads), 0, stream, args);
}

// ------------------------------------------------------------------ selection audit (host, diagnostic only)
// Per scored row: the top-k sets of the exact and the FAST scores (larger wins, ties by lower index) -> recall
// |A n B| / k and the union |A u B|. Accumulates over calls; print() writes one summary line.
struct Audit {
    double rows = 0, recall_sum = 0, union_sum = 0, exact_rows = 0, worst = 1.0, k_sum = 0;
    template <class T, class Valid>
    void add(const T* exact, const T* fast, std::size_t n, unsigned k, Valid valid) {
        std::vector<std::uint32_t> ia, ib;
        ia.reserve(n); ib.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            if (valid(i)) { ia.push_back(static_cast<std::uint32_t>(i)); ib.push_back(static_cast<std::uint32_t>(i)); }
        if (ia.size() <= k) return;  // every candidate is selected either way
        const auto top = [&](std::vector<std::uint32_t>& v, const T* s) {
            std::partial_sort(v.begin(), v.begin() + k, v.end(), [&](std::uint32_t x, std::uint32_t y) {
                return s[x] > s[y] || (!(s[y] > s[x]) && x < y);
            });
            v.resize(k);
            std::sort(v.begin(), v.end());
        };
        top(ia, exact);
        top(ib, fast);
        std::vector<std::uint32_t> both;
        std::set_intersection(ia.begin(), ia.end(), ib.begin(), ib.end(), std::back_inserter(both));
        const double r = static_cast<double>(both.size()) / k;
        rows += 1; recall_sum += r; union_sum += 2.0 * k - static_cast<double>(both.size()); k_sum += k;
        exact_rows += both.size() == k ? 1 : 0;
        worst = r < worst ? r : worst;
    }
    std::string line(const char* who) const {
        const bool any = rows > 0;
        char b[256];
        std::snprintf(b, sizeof b, "%s idx-wmma audit: rows %.0f recall@k mean %.5f min %.4f, rows with identical set %.4f, "
                      "mean |A u B| %.2f (k %.0f), block flip rate %.5f", who, rows, any ? recall_sum / rows : 1.0, worst,
                      any ? exact_rows / rows : 1.0, any ? union_sum / rows : 0.0, any ? k_sum / rows : 0.0,
                      any ? 1.0 - recall_sum / rows : 0.0);
        return b;
    }
};
inline bool audit_enabled() {
    const char* e = std::getenv("STRIX_IDX_WMMA_AUDIT");
    return e != nullptr && std::strcmp(e, "1") == 0;
}

}  // namespace strix::attn_core::idx
