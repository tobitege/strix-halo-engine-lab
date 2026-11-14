#pragma once
// Shared dense FP16 GEMM for prefill row batches on v_wmma_f32_16x16x16_f16: y[r][o] = sum_k fp16(x[r][k]) * w[o][k].
//
// Class: FAST (not the exact FP32 chains). Arithmetic contract: x is rounded to FP16 by the _Float16 cast (RNE, finite
// values beyond the FP16 range become inf, NaN stays NaN); the weights are FP16 as stored; per (row tile, column tile)
// one WMMA chain from +0 over the 16-wide k units in ascending order (FP32 accumulation in hardware order).
//
// Geometry (core<TC>): one CTA = 128 rows x 16 TC columns, 8 waves; wave w owns row tile w and all TC column tiles,
// so a row fragment (the only in-loop VALU work: 16 conversions) feeds TC WMMAs.
//   rows    : A fragment = 16 FP32 values of the lane's row, converted in registers (lane = row); never staged.
//   weights : per stage of four k units, thread i of the CTA copies the 64 FP16 weights of column i into LDS (once per
//             CTA, so a weight byte is read from memory once per 128 rows); every wave reads its B fragments from
//             there (lane = output column). The next stage's weights are requested before the current WMMAs.
// The k loop holds two stages per iteration: the accumulators are loop-carried and a back edge costs register moves.
// Cols policy: tile(i) -> ColumnTile for column tile i of the launch (16 consecutive outputs). Dead tiles (padding of
// the last CTA column) must still name readable weights; they are multiplied and never stored. plan() picks the column
// width of the remainder launch (TC 4, 2 or 1) so few such tiles exist.
// Store policy: where a finished sum goes (default RawStore: y[r][c] = sum). An adapter's policy can apply its epilogue
// (scale, bias, residual, a sum with an earlier pass) in the same store, so no second kernel re-reads the raw sums.
//   open(i, ct, c) -> per-lane state for live column tile i of the launch (c = lane's column inside the tile)
//   put(state, r, v): row r < rows, v = the WMMA chain's sum;  close(): once per thread after its last put.
//   run(f): calls f(s) with the object that has open/put/close. A policy with several modes switches on its
//   (wave-uniform) mode here and passes a store whose mode is a compile-time constant: no per-element mode branches.
// Row waves with no live row (the CTA past the last row) keep their staging and barrier duties and skip their row
// loads and WMMAs (skip_dead_waves; a wave with at least one live row is unchanged).
// Limits (callers check): in % 128 == 0, rows * in * 4 < 2^32.
#include "strix/env_flags.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::core::f16_dense_tall {
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
using Float4 = float __attribute__((ext_vector_type(4)));
using U4 = std::uint32_t __attribute__((ext_vector_type(4)));

inline constexpr unsigned kRows = 128;  // rows per CTA

struct ColumnTile {
    const _Float16* w;  // weight rows of the tile's 16 outputs, [16][in] row-major
    float* y;           // output of row 0 for the tile's first column
    unsigned stride;    // floats between consecutive rows of y
    bool live;          // false: computed, never stored
};

namespace detail {
using GlobalF4 = const Float4 __attribute__((address_space(1), aligned(4)));
using GlobalU4 = const U4 __attribute__((address_space(1), aligned(2)));

__device__ __forceinline__ unsigned uni(unsigned v) {  // wave-uniform value -> scalar register
    return static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(v)));
}
template <class T> __device__ __forceinline__ T* uniform(T* p) {  // wave-uniform pointer -> scalar registers (saddr loads)
    const auto v = reinterpret_cast<std::uintptr_t>(p);
    const auto lo = uni(static_cast<std::uint32_t>(v)), hi = uni(static_cast<std::uint32_t>(v >> 32U));
    return reinterpret_cast<T*>((static_cast<std::uintptr_t>(hi) << 32U) | lo);
}
__device__ __forceinline__ Half16 row_fragment(const unsigned char* x, std::uint32_t offset) {
    asm("" : "+v"(offset));
    Float4 f[4];
#pragma unroll
    for (unsigned i = 0; i < 4U; ++i) f[i] = *(GlobalF4*)(x + offset + 16U * i);
    Half16 h;
#pragma unroll
    for (unsigned i = 0; i < 16U; ++i) h[i] = static_cast<_Float16>(f[i / 4U][i % 4U]);
    return h;
}
__device__ __forceinline__ Half16 staged_fragment(const std::uint32_t* p) {
    struct { U4 l, h; } v{*reinterpret_cast<const U4*>(p), *reinterpret_cast<const U4*>(p + 4)};
    return __builtin_bit_cast(Half16, v);
}
}  // namespace detail

// Pitch 36 dwords per column: b128 accesses stay aligned and columns rotate banks. 2,304 bytes per column tile.
template <unsigned TC> struct alignas(16) Storage {
    std::uint32_t w[16U * TC][36];
};
static_assert(sizeof(Storage<16>) == 36864 && sizeof(Storage<4>) == 9216 && sizeof(Storage<1>) == 2304);

struct RawStore {
    struct Tile {
        float* y;
        unsigned stride, c;
    };
    __device__ __forceinline__ Tile open(unsigned, const ColumnTile& ct, unsigned c) const {
        return {detail::uniform(ct.y), detail::uni(ct.stride), c};
    }
    __device__ __forceinline__ void put(const Tile& t, unsigned r, float v) const { t.y[static_cast<std::size_t>(r) * t.stride + t.c] = v; }
    __device__ __forceinline__ void close() const {}
    template <class F> __device__ __forceinline__ void run(F&& f) { f(*this); }
};

// Host side: launches that cover `tiles` column tiles of `rows` rows. 16-tile CTA columns first; the remainder
// (rest < 16 tiles) in one launch of `tc`-tile columns, `gy` of them. small_tail = false is the previous rule (always
// 4-tile columns). The rule is the measured one (in = 1024): a 1- or 2-tile remainder runs fastest in 1-tile columns
// at every row count; up to 256 rows the launch is latency-bound and 2-tile columns win for every larger remainder;
// with more rows 2-tile columns win up to 5 tiles and 4-tile columns from 6.
struct Plan {
    unsigned full = 0;        // grid.y of the core<16> launch (0: none), tile0 = 0
    unsigned tc = 0, gy = 0;  // remainder launch core<tc>, grid.y = gy, tile0 = 16 * full (tc = 0: none)
};
inline Plan plan(unsigned tiles, unsigned rows, bool small_tail) {
    Plan p;
    p.full = tiles / 16U;
    const unsigned rest = tiles % 16U;
    if (!rest) return p;
    p.tc = 4U;
    p.gy = (rest + 3U) / 4U;
    if (!small_tail) return p;
    p.tc = rest <= 2U ? 1U : (rows <= 2U * kRows || rest <= 5U) ? 2U : 4U;
    p.gy = (rest + p.tc - 1U) / p.tc;
    return p;
}
// STRIX_F16_DENSE_TAIL (default on; 0 = previous behaviour): remainder launch sized by plan(), idle row waves skip
// their WMMAs. Results are bit-identical either way.
inline bool tail_policy() {
    static const bool on = envcfg::flag("STRIX_F16_DENSE_TAIL", true);
    return on;
}

template <unsigned TC, class Cols, class Store = RawStore>
__device__ __forceinline__ void core(const float* x, unsigned in, unsigned nr, Cols cols, Storage<TC>& lds, Store store = {},
                                     bool skip_dead_waves = true) {
    // No 8-tile width: its WMMAs sink below the fragment reads of the whole stage and the kernel spills (124+ VGPRs).
    static_assert(TC == 1 || TC == 2 || TC == 4 || TC == 16);
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U, c = lane & 15U;
    const unsigned row0 = blockIdx.x * kRows + wave * 16U;
    const unsigned units = detail::uni(in) / 16U;
    const auto* xb = reinterpret_cast<const unsigned char*>(detail::uniform(x));
    // Rows past nr read row 0 (their sums are never stored).
    const std::uint32_t xo = (row0 + c < nr ? row0 + c : 0U) * in * 4U;
    // Stager role: thread tid owns column tid of the CTA's column block.
    const bool stager = tid < 16U * TC;
    const auto* wp = reinterpret_cast<const unsigned char*>(cols.tile(blockIdx.y * TC + (stager ? tid / 16U : 0U)).w) +
                     static_cast<std::size_t>(tid & 15U) * in * 2U;
    U4 nb[8];
    const auto fetch = [&](unsigned unit) {
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) nb[i] = *(detail::GlobalU4*)(wp + std::size_t{unit} * 32U + 16U * i);
    };
    // A wave with no live row keeps its staging and barrier duties and skips its row loads and WMMAs.
    const bool idle = detail::uni(skip_dead_waves && row0 >= nr ? 1U : 0U) != 0U;
    Float8 acc[TC] = {};
    Half16 an = detail::row_fragment(xb, xo);
    if (stager) fetch(0U);
    for (unsigned base = 0; base < units; base += 8U) {
#pragma unroll
        for (unsigned s = 0; s < 2U; ++s) {
            const unsigned u0 = base + 4U * s;
            __syncthreads();  // every wave finished reading the previous stage
            if (stager) {
#pragma unroll
                for (unsigned i = 0; i < 8U; ++i) *reinterpret_cast<U4*>(&lds.w[tid][4U * i]) = nb[i];
            }
            __syncthreads();
            if (stager && (s == 0U || u0 + 4U < units)) fetch(u0 + 4U);
            asm volatile("" ::: "memory");  // the requests stay ahead of this stage's WMMAs
            if (idle) continue;
#pragma unroll
            for (unsigned kk = 0; kk < 4U; ++kk) {
                const Half16 a = an;
                if (kk + 1U < 4U || s == 0U || u0 + 4U < units) an = detail::row_fragment(xb, xo + (u0 + kk + 1U) * 64U);
#pragma unroll
                for (unsigned t = 0; t < TC; ++t)
                    acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, detail::staged_fragment(&lds.w[16U * t + c][kk * 8U]), acc[t]);
            }
        }
    }
    // acc[t][j] -> row row0 + 2 j + lane / 16, column 16 t + lane % 16. The policy's run() instantiates this loop once
    // per store mode it has, so a mode is chosen once per wave and never per element.
    store.run([&](auto& s) __attribute__((always_inline)) {
#pragma unroll
        for (unsigned t = 0; t < TC; ++t) {
            const ColumnTile ct = cols.tile(blockIdx.y * TC + t);
            if (!detail::uni(ct.live ? 1U : 0U)) continue;
            const auto st = s.open(blockIdx.y * TC + t, ct, c);
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) {
                const unsigned r = row0 + 2U * j + (lane >> 4U);
                if (r < nr) s.put(st, r, acc[t][j]);
            }
        }
        s.close();
    });
}

}  // namespace strix::core::f16_dense_tall
