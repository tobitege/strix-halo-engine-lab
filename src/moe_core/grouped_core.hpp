#pragma once
#include "grouped_policy.hpp"
#include "exl3_trellis.hpp"

namespace strix::moe_core {
namespace dev = ::strix::moe_core;

using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));

struct Item {
    const std::uint32_t* trellis;
    const _Float16* x;  // transformed rows [rows][input]
    float* part;        // [splits][rows][output]
    unsigned input, output, bits, rows, splits, cta_begin, gx, gy;  // gx = output / 128, gy = ceil(rows / (16 MT))
    unsigned row0;     // first row inside the group (multiple of 16)
};

template <int MT>
struct Accum {
    Float8 acc0[MT], acc1[MT];
};

// No launch policy lives here: adapters retain their kernel signatures and row policy.
template <template <int> class WeightDecode> struct GroupedDecodeTraits {
    template <int Bits> using Decode = ScalarDecode<Bits, WeightDecode>;
};

template <template <int> class WeightDecode, class ExpertLayout, class InputTransform, class Epilogue,
          class Traits = GroupedDecodeTraits<WeightDecode>>
struct GroupedCore : GroupedPrefillPolicy<> {
    using Input = InputTransform;
    using Output = Epilogue;



// One grouped work item: one matrix over row tiles starting at row0 inside the group.






template <int Bits>
static __device__ __forceinline__ void load_window(const std::uint32_t* tile, int w0, std::uint32_t (&w)[WeightDecode<Bits>::Window::kNW]) {
#pragma unroll
    for (int i = 0; i < WeightDecode<Bits>::Window::kNW; ++i) {
        int idx = w0 + i;
        while (idx >= WeightDecode<Bits>::Window::kWords) idx -= WeightDecode<Bits>::Window::kWords;  // w0 + i < 3 kWords
        w[i] = tile[idx];
    }
}

// Column c = lane % 16 of the lane's tile, rows k = 0..15 -> its B fragment (16 halves packed in 8 words).
template <int Bits>
static __device__ __forceinline__ void decode_fragment(const std::uint32_t (&w)[WeightDecode<Bits>::Window::kNW], bool h, std::uint32_t (&frag)[8]) {
    using W = typename WeightDecode<Bits>::Window;
#pragma unroll
    for (int r = 0; r < 16; r += 2) {
        std::uint16_t v[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int rr = r + i;
            const std::uint32_t hi = h ? w[W::last_word(1, rr) - 1] : w[W::last_word(0, rr) - 1];
            const std::uint32_t lo = h ? w[W::last_word(1, rr)] : w[W::last_word(0, rr)];
            const unsigned s = h ? W::shift(1, rr) : W::shift(0, rr);
            const std::uint32_t state = __builtin_amdgcn_alignbit(hi, lo, s) & 65535U;
            const _Float16 q = static_cast<_Float16>(WeightDecode<Bits>::codebook_value(state));
            v[i] = __builtin_bit_cast(std::uint16_t, q);
        }
        frag[r / 2] = static_cast<std::uint32_t>(v[0]) | (static_cast<std::uint32_t>(v[1]) << 16);
    }
}

static __device__ __forceinline__ Half16 as_half16(const std::uint32_t (&f)[8]) {
    Half16 out;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[2 * i] = __builtin_bit_cast(_Float16, static_cast<std::uint16_t>(f[i] & 0xffffU));
        out[2 * i + 1] = __builtin_bit_cast(_Float16, static_cast<std::uint16_t>(f[i] >> 16));
    }
    return out;
}

template <int Bits, int MT>
static __device__ void item_core(const Item& it, unsigned bx, unsigned by, unsigned bz) {
    using W = typename WeightDecode<Bits>::Window;
    const int lane = static_cast<int>(threadIdx.x & 31U), wave = static_cast<int>(threadIdx.x >> 5U);
    const int c = lane & 15;
    const bool upper = lane >= 16, h = c >= 8;
    const int inputs = static_cast<int>(it.input), outputs = static_cast<int>(it.output), rows = static_cast<int>(it.rows);
    const int tiles_n = outputs / 16, tiles_k = inputs / 16;
    const int tile_pair = static_cast<int>(bx) * 4 + wave;           // tiles 2p, 2p + 1
    const int my_tile = 2 * tile_pair + (upper ? 1 : 0);
    const int begin = static_cast<int>(static_cast<std::int64_t>(tiles_k) * bz / it.splits);
    const int end = static_cast<int>(static_cast<std::int64_t>(tiles_k) * (bz + 1) / it.splits);
    const int first_row = static_cast<int>(by) * MT * 16;
    const int w0 = W::first_word(c);
    const std::uint32_t* tp = it.trellis + static_cast<std::size_t>(my_tile) * W::kWords;
    const std::size_t k_stride = ExpertLayout::template packed_stride<W>(tiles_n);
    // A row pointers of the lane (row = first_row + 16 m + c), null past the item's rows.
    const _Float16* xr[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int row = first_row + 16 * m + c;
        xr[m] = row < rows ? it.x + static_cast<std::size_t>(row) * static_cast<std::size_t>(inputs) : nullptr;
    }
    Float8 acc0[MT] = {}, acc1[MT] = {};
    std::uint32_t wn[W::kNW];
    load_window<Bits>(tp + static_cast<std::size_t>(begin) * k_stride, w0, wn);
    for (int tile_k = begin; tile_k < end; ++tile_k) {
        std::uint32_t wc[W::kNW];
#pragma unroll
        for (int i = 0; i < W::kNW; ++i) wc[i] = wn[i];
        if (tile_k + 1 < end) load_window<Bits>(tp + static_cast<std::size_t>(tile_k + 1) * k_stride, w0, wn);
        std::uint32_t mine[8], other[8];
        decode_fragment<Bits>(wc, h, mine);
#pragma unroll
        for (int i = 0; i < 8; ++i)
            other[i] = __builtin_amdgcn_permlanex16(mine[i], mine[i], 0x76543210U, 0xfedcba98U, false, false);
        std::uint32_t f0[8], f1[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            f0[i] = upper ? other[i] : mine[i];
            f1[i] = upper ? mine[i] : other[i];
        }
        const Half16 b0 = as_half16(f0), b1 = as_half16(f1);
        // A fragments in groups of <= 4 row tiles (MT = 8 with all 8 live spills: 256 VGPRs).
        constexpr int G = MT < 4 ? MT : 4;
#pragma unroll
        for (int g = 0; g < MT; g += G) {
            Half16 a[G];
#pragma unroll
            for (int i = 0; i < G; ++i) {
                const int m = g + i;
                if (xr[m] != nullptr) {  // 32 bytes, 32-byte aligned (row stride and workspace offsets)
                    const auto* src = static_cast<const _Float16*>(__builtin_assume_aligned(xr[m] + tile_k * 16, 32));
                    __builtin_memcpy(&a[i], src, sizeof(Half16));
                } else {
                    a[i] = Half16{};
                }
            }
#pragma unroll
            for (int i = 0; i < G; ++i) {
                acc0[g + i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b0, acc0[g + i]);
                acc1[g + i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b1, acc1[g + i]);
            }
        }
    }
    // Store as wmma_item: acc[m][j] -> row first_row + 16 m + 2 j + lane / 16, column tile * 16 + lane % 16.
    const int col0 = (2 * tile_pair) * 16 + c, col1 = col0 + 16;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row = first_row + 16 * m + 2 * j + (lane >> 4);
            if (row < rows) {
                float* dst = it.part + (static_cast<std::size_t>(bz) * static_cast<std::size_t>(rows) + static_cast<std::size_t>(row)) *
                                           static_cast<std::size_t>(outputs);
                dst[col0] = acc0[m][j];
                dst[col1] = acc1[m][j];
            }
        }
    }
}


// ---------------------------------------------------------------- v3: normalized window
// Each lane funnel-shifts its window once per k-tile so that N[0] starts at stream bit S = s0 - 32 for EVERY lane;
// then state r = alignbit(N[g - 1], N[g], shift) with g and shift compile-time constants (no selects). Per-lane word
// offsets are hoisted out of the k loop and the tile base is uniform. CPU proof: qualification notes
// (v3). ISA (bits 3..8): 155-205 VALU per k-tile vs 157-333 in v2; no v_cmp / lane selects left but the 16 half-wave
// fragment selects. (Splitting the codebook multiply into two 24-bit multiplies was folded back by LLVM: not used.)


// Mode 0: exact. Diagnostics (bench only, outputs meaningless): 1 = no A loads, 2 = no decode (B from raw words).
// Tiled (v4): the group's rows are tile-major, [row tile of 16][k tile of 16][16 rows][16 halves] (it.x = the first
// tile of the group, rows padded to 16): one 16 x 16 A fragment is one contiguous 512-byte block (row-major: 16 rows x
// 32 bytes, one cache line each). Same values, same WMMA operands.


template <int Bits, int MT, int Mode, bool Tiled = false, int Core8 = 0, bool Paired = false>
static __device__ __forceinline__ Accum<MT> item_acc_v3(const Item& it, unsigned bx, unsigned by, unsigned bz, Half16* exchange = nullptr) {
    static_assert(Core8 == 0 || (MT == 8 && Tiled && Mode == 0));
    using NW = typename WeightDecode<Bits>::Norm;
    const int lane = static_cast<int>(threadIdx.x & 31U), wave = static_cast<int>((threadIdx.x >> 5U) & (Paired ? 3U : 7U));
    const int c = lane & 15;
    const bool upper = lane >= 16;
    const int inputs = static_cast<int>(it.input), outputs = static_cast<int>(it.output), rows = static_cast<int>(it.rows);
    const int tiles_n = outputs / 16, tiles_k = inputs / 16;
    const int tile_pair = static_cast<int>(bx) * 4 + (Core8 == 2 ? static_cast<int>(by & 1U) * 2 + (wave & 1) : wave);
    const int my_tile = 2 * tile_pair + (Core8 == 2 ? wave >> 1 : (upper ? 1 : 0));
    const int begin = static_cast<int>(static_cast<std::int64_t>(tiles_k) * bz / it.splits);
    const int end = static_cast<int>(static_cast<std::int64_t>(tiles_k) * (bz + 1) / it.splits);
    const int first_row = static_cast<int>(it.row0) + (Core8 == 2 ? static_cast<int>(by / 2) * 256 + (wave >> 1) * 128 : static_cast<int>(by) * MT * 16);
    // Window of this lane: start word wb, normalizing shift sh, word offsets inside the k-tile row of tiles.
    const int base = 32 * (c & 7) + 4 * (c >> 3);
    const int S = (base + 1) * Bits - 16 + 256 * Bits - 32;
    const int d = S & 31, wb = (S >> 5) - (d == 0 ? 1 : 0);
    const unsigned sh = static_cast<unsigned>((32 - d) & 31);
    int loff[NW::kNL];
#pragma unroll
    for (int i = 0; i < NW::kNL; ++i) loff[i] = my_tile * NW::kWords + (wb + i) % NW::kWords;
    const std::size_t k_stride = ExpertLayout::template packed_stride<NW>(tiles_n);
    const _Float16* xr[MT];  // row-major: the lane's row; tiled: the lane's 32 bytes in k tile 0 of its row tile
    constexpr int kStep = Tiled ? 256 : 16;  // halves between consecutive k tiles of one lane
#pragma unroll
    for (int m = 0; m < (Core8 == 1 ? 0 : MT); ++m) {
        const int row = first_row + 16 * m + c;
        if constexpr (Tiled)
            xr[m] = row < rows ? it.x + (static_cast<std::size_t>(first_row / 16 + m) * static_cast<std::size_t>(tiles_k)) * 256 +
                                     static_cast<std::size_t>(c) * 16
                               : nullptr;
        else
            xr[m] = row < rows ? it.x + static_cast<std::size_t>(row) * static_cast<std::size_t>(inputs) : nullptr;
    }
    // Each thread owns one 32-byte row of the CTA's 4 KB A tile.
    [[maybe_unused]] auto load_a = [&](int k) {
        const int m = static_cast<int>(threadIdx.x) / 16;
        const int r = static_cast<int>(threadIdx.x) & 15;
        Half16 a{};
        if (first_row + 16 * m + r < rows) {
            const auto* src = static_cast<const _Float16*>(__builtin_assume_aligned(
                it.x + (static_cast<std::size_t>(first_row / 16 + m) * static_cast<std::size_t>(tiles_k) + static_cast<std::size_t>(k)) * 256 +
                    static_cast<std::size_t>(r) * 16, 32));
            __builtin_memcpy(&a, src, sizeof(a));
        }
        return a;
    };
    if constexpr (Core8 == 1) {
        if (begin < end) {
            exchange[threadIdx.x] = load_a(begin);
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");  // LDS only: no vmcnt drain
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
        }
    }
    Accum<MT> result{};
    auto& acc0 = result.acc0;
    auto& acc1 = result.acc1;
    std::uint32_t ln[NW::kNL];
    {
        const std::uint32_t* kb = it.trellis + static_cast<std::size_t>(begin) * k_stride;
#pragma unroll
        for (int i = 0; i < NW::kNL; ++i) ln[i] = kb[loff[i]];
    }
    for (int tile_k = begin; tile_k < end; ++tile_k) {
        std::uint32_t lc[NW::kNL];
#pragma unroll
        for (int i = 0; i < NW::kNL; ++i) lc[i] = ln[i];
        // Issue A before the next weight words; its later LDS store must not wait for ln.
        [[maybe_unused]] Half16 next_a{};
        if constexpr (Core8 == 1) {
            if (tile_k + 1 < end) next_a = load_a(tile_k + 1);
        }
        if (tile_k + 1 < end) {
            const std::uint32_t* kb = it.trellis + static_cast<std::size_t>(tile_k + 1) * k_stride;  // uniform
#pragma unroll
            for (int i = 0; i < NW::kNL; ++i) ln[i] = kb[loff[i]];
        }
        std::uint32_t mine[8];
        if constexpr (Mode == 2) {
#pragma unroll
            for (int i = 0; i < 8; ++i) mine[i] = (lc[i % NW::kNL] & 0x3bff3bffU) | 0x30003000U;  // finite halves
        } else {
            std::uint32_t nrm[NW::kNL - 1];
#pragma unroll
            for (int i = 0; i < NW::kNL - 1; ++i) nrm[i] = __builtin_amdgcn_alignbit(lc[i], lc[i + 1], sh);
#pragma unroll
            for (int r = 0; r < (Core8 == 2 ? 8 : 16); r += 2) {
                std::uint16_t v[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int rr = r + i;
                    std::uint32_t state;
                    if constexpr (Core8 == 2) {
                        // Constant word indices avoid a lane-dependent private-array lookup.
                        const auto hi = upper ? nrm[NW::g(rr + 8) - 1] : nrm[NW::g(rr) - 1];
                        const auto low = upper ? nrm[NW::g(rr + 8)] : nrm[NW::g(rr)];
                        const auto shift = upper ? NW::shift(rr + 8) : NW::shift(rr);
                        state = __builtin_amdgcn_alignbit(hi, low, shift) & 65535U;
                    } else {
                        state = __builtin_amdgcn_alignbit(nrm[NW::g(rr) - 1], nrm[NW::g(rr)], NW::shift(rr)) & 65535U;
                    }
                    v[i] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(WeightDecode<Bits>::codebook_value(state)));
                }
                mine[r / 2] = static_cast<std::uint32_t>(v[0]) | (static_cast<std::uint32_t>(v[1]) << 16);
            }
        }
        Half16 b0, b1;
        if constexpr (Core8 == 2) {
            // Each wave decodes one tile: lower/upper lanes supply k=0..7/8..15.
            std::uint32_t full[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const auto peer = __builtin_amdgcn_permlanex16(mine[i], mine[i], 0x76543210U, 0xfedcba98U, false, false);
                full[i] = upper ? peer : mine[i];
                full[i + 4] = upper ? mine[i] : peer;
            }
            const int buffer = (tile_k - begin) & 1;
            if (!upper) exchange[buffer * 64 + wave * 16 + c] = as_half16(full);
            // LDS release only: keep next weight words outstanding across the barrier.
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");  // LDS only: no vmcnt drain
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
            b0 = exchange[buffer * 64 + (wave & 1) * 16 + c];
            b1 = exchange[buffer * 64 + ((wave & 1) + 2) * 16 + c];
        } else {
            std::uint32_t other[8], f0[8], f1[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                other[i] = __builtin_amdgcn_permlanex16(mine[i], mine[i], 0x76543210U, 0xfedcba98U, false, false);
                f0[i] = upper ? other[i] : mine[i];
                f1[i] = upper ? mine[i] : other[i];
            }
            b0 = as_half16(f0);
            b1 = as_half16(f1);
        }
        constexpr int G = MT < 4 ? MT : 4;
        // The extra bound folds away for MT 1/2/3/4/8; MT 5/6/7 have a partial last group.
#pragma unroll
        for (int g = 0; g < MT; g += G) {
            Half16 a[G];
#pragma unroll
            for (int i = 0; i < G && g + i < MT; ++i) {
                const int m = g + i;
                if constexpr (Mode == 1) {
                    a[i] = Half16{} + static_cast<_Float16>(0.001F * static_cast<float>(m + 1));
                } else if constexpr (Core8 == 1) {
                    a[i] = exchange[((tile_k - begin) & 1) * 128 + m * 16 + c];
                } else if (xr[m] != nullptr) {  // 32 bytes, 32-byte aligned
                    const auto* src = static_cast<const _Float16*>(__builtin_assume_aligned(xr[m] + static_cast<std::size_t>(tile_k) * kStep, 32));
                    __builtin_memcpy(&a[i], src, sizeof(Half16));
                } else {
                    a[i] = Half16{};
                }
            }
#pragma unroll
            for (int i = 0; i < G && g + i < MT; ++i) {
                acc0[g + i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b0, acc0[g + i]);
                acc1[g + i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b1, acc1[g + i]);
            }
        }
        if constexpr (Core8 == 1) {
            if (tile_k + 1 < end) {
                exchange[((tile_k - begin + 1) & 1) * 128 + threadIdx.x] = next_a;
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");  // LDS only: no vmcnt drain
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
            }
        }
    }
    return result;
}

template <int Bits, int MT, int Mode, bool Tiled = false, int Core8 = 0>
static __device__ void item_core_v3(const Item& it, unsigned bx, unsigned by, unsigned bz, Half16* exchange = nullptr) {
    const auto result = item_acc_v3<Bits, MT, Mode, Tiled, Core8>(it, bx, by, bz, exchange);
    const auto& acc0 = result.acc0;
    const auto& acc1 = result.acc1;
    const int lane = static_cast<int>(threadIdx.x & 31U), wave = static_cast<int>(threadIdx.x >> 5U);
    const int c = lane & 15, rows = static_cast<int>(it.rows), outputs = static_cast<int>(it.output);
    const int tile_pair = static_cast<int>(bx) * 4 + (Core8 == 2 ? static_cast<int>(by & 1U) * 2 + (wave & 1) : wave);
    const int first_row = static_cast<int>(it.row0) + (Core8 == 2 ? static_cast<int>(by / 2) * 256 + (wave >> 1) * 128 : static_cast<int>(by) * MT * 16);
    const int col0 = (2 * tile_pair) * 16 + c, col1 = col0 + 16;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row = first_row + 16 * m + 2 * j + (lane >> 4);
            if (row < rows) {
                float* dst = it.part + (static_cast<std::size_t>(bz) * static_cast<std::size_t>(rows) + static_cast<std::size_t>(row)) *
                                           static_cast<std::size_t>(outputs);
                dst[col0] = acc0[m][j];
                dst[col1] = acc1[m][j];
            }
        }
    }
}

// Full-row pipeline variant. The default core above stays unchanged.
static __device__ __forceinline__ std::uint32_t pipe_product(std::uint32_t state) {
    return WeightDecode<3>::pipe_product(state);
}
// For sum in [0,1020], both affine forms are integer multiples of 2^-17
// with <24 significant bits: (1024+sum)*887 - 1329*1024 == sum*887 - 221*2048.
// FP32 FMA is exact in either form, followed by the same RN16 conversion.
static __device__ __forceinline__ float pipe_codebook(std::uint32_t state) {
    return WeightDecode<3>::pipe_codebook(state);
}
using GlobalHalf16 = const Half16 __attribute__((address_space(1)));
static __device__ __forceinline__ Half16 pipe_load_a(const unsigned char* base, std::uint32_t offset) {
    asm("" : "+v"(offset));
    return *(GlobalHalf16*)(base + offset);
}

template <int Bits, int MT, int G, bool FullRows = true, bool Simplify = false>
static __device__ void item_pipe(const Item& it, unsigned bx, unsigned by, unsigned bz) {
    static_assert(MT == 8 && G == 4 && FullRows);
    using NW = typename WeightDecode<Bits>::Norm;
    const int lane = static_cast<int>(threadIdx.x & 31U);
    const int wave = static_cast<int>(threadIdx.x >> 5U);
    const int c = lane & 15;
    const bool upper = lane >= 16;
    const unsigned first_row = it.row0 + by * 16U * static_cast<unsigned>(MT);
    const unsigned tiles_k = it.input / 16U;
    const int begin = static_cast<int>(static_cast<std::uint64_t>(tiles_k) * bz / it.splits);
    const int end = static_cast<int>(static_cast<std::uint64_t>(tiles_k) * (bz + 1U) / it.splits);
    const int tile_pair = static_cast<int>(bx) * 4 + wave;
    const int my_tile = 2 * tile_pair + (upper ? 1 : 0);
    const int base = 32 * (c & 7) + 4 * (c >> 3);
    const int S = (base + 1) * Bits - 16 + 256 * Bits - 32;
    const int d = S & 31, wb = (S >> 5) - (d == 0 ? 1 : 0);
    const unsigned sh = static_cast<unsigned>((32 - d) & 31);
    std::uint32_t loff[NW::kNL];
#pragma unroll
    for (int i = 0; i < NW::kNL; ++i)
        loff[i] = static_cast<std::uint32_t>(my_tile * NW::kWords + (wb + i) % NW::kWords) * 4U;
    const std::size_t k_stride = ExpertLayout::template packed_stride<NW>(it.output / 16U);
    const auto* weights = dev::uniform(it.trellis);
    // Rebase per row tile using scalar 64-bit arithmetic, then use bounded lane offsets.
    const auto* x = dev::uniform(it.x + static_cast<std::size_t>(first_row / 16U) * tiles_k * 256U);
    Float8 acc0[MT] = {}, acc1[MT] = {};
    std::uint32_t ln[NW::kNL];
    if (begin < end) {
        const auto* kb = reinterpret_cast<const unsigned char*>(dev::uniform(weights + static_cast<std::size_t>(begin) * k_stride));
#pragma unroll
        for (int i = 0; i < NW::kNL; ++i) ln[i] = dev::load_u32(kb, loff[i]);
    }
    for (int tile_k = begin; tile_k < end; ++tile_k) {
        Half16 a0[G];
#pragma unroll
        for (int m = 0; m < G; ++m) {
            const auto* ab = reinterpret_cast<const unsigned char*>(dev::uniform(
                x + (static_cast<std::size_t>(m) * tiles_k + static_cast<unsigned>(tile_k)) * 256U));
            a0[m] = pipe_load_a(ab, static_cast<std::uint32_t>(c) * 32U);
        }
        Half16 b0, b1;
        { // Decode temporaries die before the second A bank becomes live.
            std::uint32_t nrm[NW::kNL - 1], mine[8], f0[8], f1[8];
#pragma unroll
            for (int i = 0; i < NW::kNL - 1; ++i) nrm[i] = __builtin_amdgcn_alignbit(ln[i], ln[i + 1], sh);
            if constexpr (!Simplify && Traits::template Decode<Bits>::kPacked) {
#pragma unroll
                for (int r = 0; r < 16; r += 8) {
                    std::uint32_t st[8];
                    Half2 decoded[4];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int rr = r + i;
                        st[i] = __builtin_amdgcn_alignbit(nrm[NW::g(rr) - 1], nrm[NW::g(rr)], NW::shift(rr)) & 65535U;
                    }
                    Traits::template Decode<Bits>::decode8(st, decoded);
#pragma unroll
                    for (int i = 0; i < 4; ++i) mine[r / 2 + i] = __builtin_bit_cast(std::uint32_t, decoded[i]);
                }
            } else {
#pragma unroll
                for (int r = 0; r < 16; r += 2) {
                    std::uint16_t v[2];
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const int rr = r + i;
                        const auto state = __builtin_amdgcn_alignbit(nrm[NW::g(rr) - 1], nrm[NW::g(rr)], NW::shift(rr)) & 65535U;
                        if constexpr (Simplify)
                            v[i] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(WeightDecode<Bits>::pipe_codebook(state)));
                        else
                            v[i] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(WeightDecode<Bits>::codebook_value(state)));
                    }
                    mine[r / 2] = static_cast<std::uint32_t>(v[0]) | (static_cast<std::uint32_t>(v[1]) << 16);
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const auto other = __builtin_amdgcn_permlanex16(mine[i], mine[i], 0x76543210U, 0xfedcba98U, false, false);
                f0[i] = upper ? other : mine[i];
                f1[i] = upper ? mine[i] : other;
            }
            b0 = as_half16(f0);
            b1 = as_half16(f1);
        }
        Half16 a1[G];
#pragma unroll
        for (int m = 0; m < G; ++m) {
            const auto* ab = reinterpret_cast<const unsigned char*>(dev::uniform(
                x + (static_cast<std::size_t>(G + m) * tiles_k + static_cast<unsigned>(tile_k)) * 256U));
            a1[m] = pipe_load_a(ab, static_cast<std::uint32_t>(c) * 32U);
        }
        if (tile_k + 1 < end) {
            const auto* kb = reinterpret_cast<const unsigned char*>(dev::uniform(weights + static_cast<std::size_t>(tile_k + 1) * k_stride));
#pragma unroll
            for (int i = 0; i < NW::kNL; ++i) ln[i] = dev::load_u32(kb, loff[i]);
        }
#pragma unroll
        for (int m = 0; m < G; ++m) {
            acc0[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0[m], b0, acc0[m]);
            acc1[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0[m], b1, acc1[m]);
        }
#pragma unroll
        for (int m = 0; m < G; ++m) {
            acc0[G + m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1[m], b0, acc0[G + m]);
            acc1[G + m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1[m], b1, acc1[G + m]);
        }
    }
    const unsigned col0 = static_cast<unsigned>(2 * tile_pair * 16 + c);
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned row = first_row + static_cast<unsigned>(16 * m + 2 * j + (lane >> 4));
            auto* dst = it.part + (static_cast<std::size_t>(bz) * it.rows + row) * it.output;
            dst[col0] = acc0[m][j];
            dst[col0 + 16U] = acc1[m][j];
        }
    }
}

template <int Bits, int MT, int G, bool Simplify>
static __device__ __forceinline__ void pipe_or_tail(const Item& it, unsigned bx, unsigned by, unsigned bz) {
    // Uniform check includes row0 for exact-fit remainder items.
    if (it.row0 + by * 16U * static_cast<unsigned>(MT) + 16U * static_cast<unsigned>(MT) <= it.rows)
        item_pipe<Bits, MT, G, true, Simplify>(it, bx, by, bz);
    else item_core_v3<Bits, MT, 0, true>(it, bx, by, bz);
}



// Paired PIPE bands (STRIX_MMX_PREFILL_PIPE_PAIR). CTAs by and by ^ 1 of a band of two full 128-row tiles take the two
// 64-column halves of the item's 128 outputs; wave w covers rows band + 128 (w >> 1) of column pair w & 1 and decodes only
// its own tile (8 states per lane; lanes 16-31 hold k 8..15, whose window starts 2 Bits later, so the word / shift
// constants stay those of k 0..7). The CTA shares the four decoded tiles through LDS (two buffers, one barrier per k
// tile) and keeps PIPE's schedule: a0 before the decode, a1 and the next weight words after it. Same WMMA chain and the
// same partial stores per output as item_pipe.
template <int Bits, bool Simplify>
static __device__ void item_pipe_pair(const Item& it, unsigned bx, unsigned by, unsigned bz, Half16* exchange) {
    using NW = typename WeightDecode<Bits>::Norm;
    constexpr int MT = 8, G = 4;
    constexpr int kNL = NW::g(7) + 2;  // words for k 0..7 of the shifted window
    const int lane = static_cast<int>(threadIdx.x & 31U);
    const int wave = static_cast<int>(threadIdx.x >> 5U);
    const int c = lane & 15;
    const bool upper = lane >= 16;
    const unsigned first_row = it.row0 + (by & ~1U) * 128U + static_cast<unsigned>(wave >> 1) * 128U;
    const unsigned tiles_k = it.input / 16U;
    const int begin = static_cast<int>(static_cast<std::uint64_t>(tiles_k) * bz / it.splits);
    const int end = static_cast<int>(static_cast<std::uint64_t>(tiles_k) * (bz + 1U) / it.splits);
    const int tile_pair = static_cast<int>(bx) * 4 + static_cast<int>(by & 1U) * 2 + (wave & 1);
    const int my_tile = 2 * tile_pair + (wave >> 1);
    const int base = 32 * (c & 7) + 4 * (c >> 3);
    const int S = (base + 1) * Bits - 16 + 256 * Bits - 32 + (upper ? 2 * Bits : 0);
    const int d = S & 31, wb = (S >> 5) - (d == 0 ? 1 : 0);
    const unsigned sh = static_cast<unsigned>((32 - d) & 31);
    std::uint32_t loff[kNL];
#pragma unroll
    for (int i = 0; i < kNL; ++i)
        loff[i] = static_cast<std::uint32_t>(my_tile * NW::kWords + (wb + i) % NW::kWords) * 4U;
    const std::size_t k_stride = ExpertLayout::template packed_stride<NW>(it.output / 16U);
    const auto* weights = dev::uniform(it.trellis);
    // Rebase per row tile using scalar 64-bit arithmetic, then use bounded lane offsets.
    const auto* x = dev::uniform(it.x + static_cast<std::size_t>(first_row / 16U) * tiles_k * 256U);
    Float8 acc0[MT] = {}, acc1[MT] = {};
    std::uint32_t ln[kNL];
    if (begin < end) {
        const auto* kb = reinterpret_cast<const unsigned char*>(dev::uniform(weights + static_cast<std::size_t>(begin) * k_stride));
#pragma unroll
        for (int i = 0; i < kNL; ++i) ln[i] = dev::load_u32(kb, loff[i]);
    }
    for (int tile_k = begin; tile_k < end; ++tile_k) {
        Half16 a0[G];
#pragma unroll
        for (int m = 0; m < G; ++m) {
            const auto* ab = reinterpret_cast<const unsigned char*>(dev::uniform(
                x + (static_cast<std::size_t>(m) * tiles_k + static_cast<unsigned>(tile_k)) * 256U));
            a0[m] = pipe_load_a(ab, static_cast<std::uint32_t>(c) * 32U);
        }
        const int buffer = (tile_k - begin) & 1;
        { // Decode temporaries die before the second A bank becomes live.
            std::uint32_t nrm[kNL - 1], mine[4], full[8];
#pragma unroll
            for (int i = 0; i < kNL - 1; ++i) nrm[i] = __builtin_amdgcn_alignbit(ln[i], ln[i + 1], sh);
            if constexpr (!Simplify && Traits::template Decode<Bits>::kPacked) {
                std::uint32_t st[8];
                Half2 decoded[4];
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    st[i] = __builtin_amdgcn_alignbit(nrm[NW::g(i) - 1], nrm[NW::g(i)], NW::shift(i)) & 65535U;
                Traits::template Decode<Bits>::decode8(st, decoded);
#pragma unroll
                for (int i = 0; i < 4; ++i) mine[i] = __builtin_bit_cast(std::uint32_t, decoded[i]);
            } else {
#pragma unroll
                for (int r = 0; r < 8; r += 2) {
                    std::uint16_t v[2];
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const int rr = r + i;
                        const auto state = __builtin_amdgcn_alignbit(nrm[NW::g(rr) - 1], nrm[NW::g(rr)], NW::shift(rr)) & 65535U;
                        if constexpr (Simplify)
                            v[i] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(WeightDecode<Bits>::pipe_codebook(state)));
                        else
                            v[i] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(WeightDecode<Bits>::codebook_value(state)));
                    }
                    mine[r / 2] = static_cast<std::uint32_t>(v[0]) | (static_cast<std::uint32_t>(v[1]) << 16);
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const auto peer = __builtin_amdgcn_permlanex16(mine[i], mine[i], 0x76543210U, 0xfedcba98U, false, false);
                full[i] = upper ? peer : mine[i];
                full[i + 4] = upper ? mine[i] : peer;
            }
            if (!upper) exchange[buffer * 64 + wave * 16 + c] = as_half16(full);
        }
        Half16 a1[G];
#pragma unroll
        for (int m = 0; m < G; ++m) {
            const auto* ab = reinterpret_cast<const unsigned char*>(dev::uniform(
                x + (static_cast<std::size_t>(G + m) * tiles_k + static_cast<unsigned>(tile_k)) * 256U));
            a1[m] = pipe_load_a(ab, static_cast<std::uint32_t>(c) * 32U);
        }
        if (tile_k + 1 < end) {
            const auto* kb = reinterpret_cast<const unsigned char*>(dev::uniform(weights + static_cast<std::size_t>(tile_k + 1) * k_stride));
#pragma unroll
            for (int i = 0; i < kNL; ++i) ln[i] = dev::load_u32(kb, loff[i]);
        }
        // LDS release only: A and the next weight words stay outstanding across the barrier. A buffer is rewritten only
        // after the next barrier, which every wave reaches after these reads.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
        const Half16 b0 = exchange[buffer * 64 + (wave & 1) * 16 + c];
        const Half16 b1 = exchange[buffer * 64 + ((wave & 1) + 2) * 16 + c];
#pragma unroll
        for (int m = 0; m < G; ++m) {
            acc0[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0[m], b0, acc0[m]);
            acc1[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0[m], b1, acc1[m]);
        }
#pragma unroll
        for (int m = 0; m < G; ++m) {
            acc0[G + m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1[m], b0, acc0[G + m]);
            acc1[G + m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1[m], b1, acc1[G + m]);
        }
    }
    const unsigned col0 = static_cast<unsigned>(2 * tile_pair * 16 + c);
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned row = first_row + static_cast<unsigned>(16 * m + 2 * j + (lane >> 4));
            auto* dst = it.part + (static_cast<std::size_t>(bz) * it.rows + row) * it.output;
            dst[col0] = acc0[m][j];
            dst[col0 + 16U] = acc1[m][j];
        }
    }
}

// A band of two full 128-row tiles runs paired, every other tile PIPE / the tail. The test reads only by & ~1 and
// by | 1, so both CTAs of a band agree.
template <int Bits, bool Simplify>
static __device__ __forceinline__ void pipe_pair_or_tail(const Item& it, unsigned bx, unsigned by, unsigned bz, Half16* exchange) {
    if ((by | 1U) < it.gy && it.row0 + (by & ~1U) * 128U + 256U <= it.rows)
        item_pipe_pair<Bits, Simplify>(it, bx, by, bz, exchange);
    else pipe_or_tail<Bits, 8, 4, Simplify>(it, bx, by, bz);
}

// Tall PIPE bands for many-row dense items. One CTA of W waves (4 or 8) owns one column pair (32 outputs, grid x) over
// W full 128-row tiles (grid y), so the two weight tiles of a k step are read and decoded once per 128 W rows instead of
// once per 128 (PIPE) or 256 (paired). Wave w runs the PIPE schedule on rows band + 128 w and decodes 16 / W states per
// lane: tile w & 1, k part w >> 1 of 0..7 (lanes 16-31: of 8..15, window 2 Bits later; part p starts 64 p Bits / W
// later, so the word / shift constants stay those of the first states). Every lane stores its dwords at their place of
// the B fragment in LDS (two buffers, one barrier per k tile). Same operands, WMMA chain and partial stores per output
// as item_pipe.
template <int Bits, int W, bool Simplify>
static __device__ void item_pipe_band(const Item& it, unsigned bx, unsigned by, unsigned bz, Half16* exchange) {
    using NW = typename WeightDecode<Bits>::Norm;
    using Word = std::uint32_t __attribute__((may_alias));
    static_assert(W == 4 || W == 8, "band CTAs have 4 or 8 waves");
    constexpr int MT = 8, G = 4;
    constexpr int N = 16 / W;            // states per lane and k tile
    constexpr int kNL = NW::g(N - 1) + 2;  // words for the first N states of the shifted window
    const int lane = static_cast<int>(threadIdx.x & 31U);
    const int wave = static_cast<int>(threadIdx.x >> 5U);
    const int c = lane & 15;
    const bool upper = lane >= 16;
    const int part = wave >> 1;
    const unsigned first_row = it.row0 + (by * static_cast<unsigned>(W) + static_cast<unsigned>(wave)) * 128U;
    const unsigned tiles_k = it.input / 16U;
    const int begin = static_cast<int>(static_cast<std::uint64_t>(tiles_k) * bz / it.splits);
    const int end = static_cast<int>(static_cast<std::uint64_t>(tiles_k) * (bz + 1U) / it.splits);
    const int my_tile = 2 * static_cast<int>(bx) + (wave & 1);
    const int base = 32 * (c & 7) + 4 * (c >> 3);
    const int S = (base + 1) * Bits - 16 + 256 * Bits - 32 + (upper ? 2 * Bits : 0) + part * (4 * N * Bits);
    const int d = S & 31, wb = (S >> 5) - (d == 0 ? 1 : 0);
    const unsigned sh = static_cast<unsigned>((32 - d) & 31);
    std::uint32_t loff[kNL];
#pragma unroll
    for (int i = 0; i < kNL; ++i)
        loff[i] = static_cast<std::uint32_t>(my_tile * NW::kWords + (wb + i) % NW::kWords) * 4U;
    const std::size_t k_stride = ExpertLayout::template packed_stride<NW>(it.output / 16U);
    const auto* weights = dev::uniform(it.trellis);
    const auto* x = dev::uniform(it.x + static_cast<std::size_t>(first_row / 16U) * tiles_k * 256U);
    auto* shared = reinterpret_cast<Word*>(exchange);
    const int slot = ((wave & 1) * 16 + c) * 8 + (upper ? 4 : 0) + part * (N / 2);
    Float8 acc0[MT] = {}, acc1[MT] = {};
    std::uint32_t ln[kNL];
    if (begin < end) {
        const auto* kb = reinterpret_cast<const unsigned char*>(dev::uniform(weights + static_cast<std::size_t>(begin) * k_stride));
#pragma unroll
        for (int i = 0; i < kNL; ++i) ln[i] = dev::load_u32(kb, loff[i]);
    }
    for (int tile_k = begin; tile_k < end; ++tile_k) {
        Half16 a0[G];
#pragma unroll
        for (int m = 0; m < G; ++m) {
            const auto* ab = reinterpret_cast<const unsigned char*>(dev::uniform(
                x + (static_cast<std::size_t>(m) * tiles_k + static_cast<unsigned>(tile_k)) * 256U));
            a0[m] = pipe_load_a(ab, static_cast<std::uint32_t>(c) * 32U);
        }
        const int buffer = (tile_k - begin) & 1;
        { // Decode temporaries die before the second A bank becomes live.
            std::uint32_t nrm[kNL - 1], state[N], mine[N / 2];
#pragma unroll
            for (int i = 0; i < kNL - 1; ++i) nrm[i] = __builtin_amdgcn_alignbit(ln[i], ln[i + 1], sh);
#pragma unroll
            for (int i = 0; i < N; ++i)
                state[i] = __builtin_amdgcn_alignbit(nrm[NW::g(i) - 1], nrm[NW::g(i)], NW::shift(i)) & 65535U;
            if constexpr (!Simplify && Traits::template Decode<Bits>::kPacked) {
                std::uint32_t st[8];  // decode8 pairs are independent: the repeats beyond N are dead
                Half2 decoded[4];
#pragma unroll
                for (int i = 0; i < 8; ++i) st[i] = state[i % N];
                Traits::template Decode<Bits>::decode8(st, decoded);
#pragma unroll
                for (int i = 0; i < N / 2; ++i) mine[i] = __builtin_bit_cast(std::uint32_t, decoded[i]);
            } else {
#pragma unroll
                for (int r = 0; r < N; r += 2) {
                    std::uint16_t v[2];
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        if constexpr (Simplify)
                            v[i] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(WeightDecode<Bits>::pipe_codebook(state[r + i])));
                        else
                            v[i] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(WeightDecode<Bits>::codebook_value(state[r + i])));
                    }
                    mine[r / 2] = static_cast<std::uint32_t>(v[0]) | (static_cast<std::uint32_t>(v[1]) << 16);
                }
            }
#pragma unroll
            for (int i = 0; i < N / 2; ++i) shared[buffer * 256 + slot + i] = mine[i];
        }
        Half16 a1[G];
#pragma unroll
        for (int m = 0; m < G; ++m) {
            const auto* ab = reinterpret_cast<const unsigned char*>(dev::uniform(
                x + (static_cast<std::size_t>(G + m) * tiles_k + static_cast<unsigned>(tile_k)) * 256U));
            a1[m] = pipe_load_a(ab, static_cast<std::uint32_t>(c) * 32U);
        }
        if (tile_k + 1 < end) {
            const auto* kb = reinterpret_cast<const unsigned char*>(dev::uniform(weights + static_cast<std::size_t>(tile_k + 1) * k_stride));
#pragma unroll
            for (int i = 0; i < kNL; ++i) ln[i] = dev::load_u32(kb, loff[i]);
        }
        // LDS release only: A and the next weight words stay outstanding across the barrier. A buffer is rewritten only
        // after the next barrier, which every wave reaches after these reads.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
        const Half16 b0 = exchange[buffer * 32 + c];
        const Half16 b1 = exchange[buffer * 32 + 16 + c];
#pragma unroll
        for (int m = 0; m < G; ++m) {
            acc0[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0[m], b0, acc0[m]);
            acc1[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0[m], b1, acc1[m]);
        }
#pragma unroll
        for (int m = 0; m < G; ++m) {
            acc0[G + m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1[m], b0, acc0[G + m]);
            acc1[G + m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1[m], b1, acc1[G + m]);
        }
    }
    const unsigned col0 = bx * 32U + static_cast<unsigned>(c);
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned row = first_row + static_cast<unsigned>(16 * m + 2 * j + (lane >> 4));
            auto* dst = it.part + (static_cast<std::size_t>(bz) * it.rows + row) * it.output;
            dst[col0] = acc0[m][j];
            dst[col0 + 16U] = acc1[m][j];
        }
    }
}



// V: 2 = v2 core (item_core), 3 = v3 exact, 4 = v4 (v3 + tiled A), 31 / 32 = v3 diagnostics (no A loads / no decode;
// bench only).


template <int Bits, int Core8>
static __device__ __forceinline__ void item_core8(const Item& it, unsigned bx, unsigned by, unsigned bz, Half16* exchange) {
    if constexpr (Core8 == 2) {
        if ((by | 1U) < it.gy && it.rows >= 256) item_core_v3<Bits, 8, 0, true, 2>(it, bx, by, bz, exchange);
        else item_core_v3<Bits, 8, 0, true, 1>(it, bx, by, bz, exchange);
    } else item_core_v3<Bits, 8, 0, true, 1>(it, bx, by, bz, exchange);
}

// Core8 preserves the item table and CTA count. Paired by values cover the two
// 64-column halves of 256 rows; an odd final by uses the original 128-column shape.



};

}  // namespace strix::moe_core
