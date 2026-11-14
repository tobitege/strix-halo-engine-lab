#pragma once
#include "grouped_core.hpp"

namespace strix::moe_core {
// One-split, tile-major variant. The adapter owns launches and host planning.
// Keep this schedule distinct from the multi-split MiniMax variants.
template <class Traits>
struct OneSplitPrefill : Traits {
    static_assert(Traits::kSplits == 1 && Traits::kAscendingRouteOrder);
    using Item = typename Traits::Item;
    using GroupDev = typename Traits::GroupDev;
    using Tables = typename Traits::Tables;
    template <int Bits> using Norm = typename Traits::template Norm<Bits>;
    template <typename T> using Glob = typename Traits::template Glob<T>;
    using FragmentCore = GroupedCore<Traits::template WeightDecode, Traits, Traits, Traits>;
    using Traits::kH; using Traits::kI; using Traits::kHB; using Traits::kIB;
    using Traits::kTop; using Traits::kG3Pitch; using Traits::kG3Blocks;
    using Traits::finish_block; using Traits::swiglu_r15;
    static __device__ __forceinline__ Half16 as_half16(const std::uint32_t (&f)[8]) {
        return FragmentCore::as_half16(f);
    }
    // Traits may set kScalarWeightDecode = true to keep the scalar codebook conversion per coefficient.
    static constexpr bool kScalarWeightDecode = requires { requires Traits::kScalarWeightDecode; };
    static __device__ __forceinline__ float codebook_mul1_v2(std::uint32_t state) {
        return Traits::template WeightDecode<3>::codebook_value_v2(state);
    }
// CTA = 4 waves = 8 tile columns = 128 outputs x MT row tiles of 16. Wave w owns tiles 2p, 2p + 1 (p = 4 bx + w):
// lanes 0-15 decode column lane of tile 2p, lanes 16-31 column lane - 16 of tile 2p + 1; v_permlanex16 hands each
// half the other's column so both B fragments (lane = column, k = 0..15) are complete without LDS.
// codebook_mul1_v2 and global-address-space loads. PF: the trellis words of k-tile t + PF are loaded while tile t is
// computed (a register ring of PF tiles, compile-time slots), so more weight bytes are in flight; PF 1 unrolls the k
// loop by 2 instead. Only load timing changes with PF; all PF values give the same bits.
template <int Bits, int MT, int K, int N, int PF>
static __device__ __forceinline__ void item_core(const Item& it, unsigned bx, unsigned by) {
    using NW = Norm<Bits>;
    constexpr int tiles_n = N / 16, tiles_k = K / 16;
    const int lane = static_cast<int>(threadIdx.x & 31U), wave = static_cast<int>(threadIdx.x >> 5U);
    const int c = lane & 15;
    const bool upper = lane >= 16;
    const int rows = static_cast<int>(it.rows);
    const int tile_pair = static_cast<int>(bx) * 4 + wave;
    const int my_tile = 2 * tile_pair + (upper ? 1 : 0);
    const int first_row = static_cast<int>(by) * MT * 16;
    const int base = 32 * (c & 7) + 4 * (c >> 3);
    const int S = (base + 1) * Bits - 16 + 256 * Bits - 32;
    const int d = S & 31, wb = (S >> 5) - (d == 0 ? 1 : 0);
    const unsigned sh = static_cast<unsigned>((32 - d) & 31);
    int loff[NW::kNL];
#pragma unroll
    for (int i = 0; i < NW::kNL; ++i) loff[i] = my_tile * NW::kWords + (wb + i) % NW::kWords;
    constexpr std::size_t k_stride = static_cast<std::size_t>(tiles_n) * NW::kWords;
    // A: the lane's 32 bytes (row first_row + 16 m + c) in k tile 0 of its row tile; null past the group's rows.
    const _Float16* xr[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int row = first_row + 16 * m + c;
        const _Float16* p = it.x + (static_cast<std::size_t>(first_row / 16 + m) * static_cast<std::size_t>(tiles_k)) * 256 +
                            static_cast<std::size_t>(c) * 16;
        // Rows past the group read row tile 0 of this CTA instead of zeros (no branch). A WMMA output row depends
        // only on its own A row and those rows are never stored, so every stored value keeps its bits.
        xr[m] = row < rows ? p : it.x + static_cast<std::size_t>(first_row / 16) * static_cast<std::size_t>(tiles_k) * 256;
    }
    Float8 acc0[MT] = {}, acc1[MT] = {};
    static_assert(tiles_k % (2 * PF) == 0, "k loop unrolled by 2 (PF 1) or by PF");
    auto tword = [&](int tile, int i) -> std::uint32_t {
        const std::uint32_t* kb = it.trellis + static_cast<std::size_t>(tile) * k_stride;  // uniform
        return ((Glob<std::uint32_t>)kb)[loff[i]];
    };
    std::uint32_t ring[PF][NW::kNL];
#pragma unroll
    for (int q = 0; q < PF; ++q)
#pragma unroll
        for (int i = 0; i < NW::kNL; ++i) ring[q][i] = tword(q, i);
    constexpr int kUnroll = PF == 1 ? 2 : 1;
#pragma unroll kUnroll
    for (int tk0 = 0; tk0 < tiles_k; tk0 += PF) {
#pragma unroll
     for (int q = 0; q < PF; ++q) {
        const int tile_k = tk0 + q;
        std::uint32_t lc[NW::kNL];
#pragma unroll
        for (int i = 0; i < NW::kNL; ++i) lc[i] = ring[q][i];
        if (tile_k + PF < tiles_k) {
#pragma unroll
            for (int i = 0; i < NW::kNL; ++i) ring[q][i] = tword(tile_k + PF, i);
        }
        std::uint32_t nrm[NW::kNL - 1];
#pragma unroll
        for (int i = 0; i < NW::kNL - 1; ++i) nrm[i] = __builtin_amdgcn_alignbit(lc[i], lc[i + 1], sh);
        std::uint32_t mine[8];
        if constexpr (!kScalarWeightDecode) {
            // The shared codec's packed pair decode (full-rate integer hash, one packed FP16 FMA per two coefficients):
            // the same FP16 bits as the scalar codebook below for all 65,536 states, with fewer and faster operations.
            Traits::template WeightDecode<Bits>::Fragment::template decode16<typename Traits::template WeightDecode<Bits>::Codebook>(nrm, mine);
        } else {
#pragma unroll
            for (int r = 0; r < 16; r += 2) {
                std::uint16_t v[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int rr = r + i;
                    const std::uint32_t state = __builtin_amdgcn_alignbit(nrm[NW::g(rr) - 1], nrm[NW::g(rr)], NW::shift(rr)) & 65535U;
                    v[i] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(codebook_mul1_v2(state)));
                }
                mine[r / 2] = static_cast<std::uint32_t>(v[0]) | (static_cast<std::uint32_t>(v[1]) << 16);
            }
        }
        std::uint32_t other[8], f0[8], f1[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            other[i] = __builtin_amdgcn_permlanex16(mine[i], mine[i], 0x76543210U, 0xfedcba98U, false, false);
            f0[i] = upper ? other[i] : mine[i];
            f1[i] = upper ? mine[i] : other[i];
        }
        const Half16 b0 = as_half16(f0), b1 = as_half16(f1);
        constexpr int G = MT < 4 ? MT : 4;  // A fragments in groups of <= 4 row tiles (VGPR budget at MT = 8)
#pragma unroll
        for (int g = 0; g < MT; g += G) {
            Half16 a[G];
#pragma unroll
            for (int i = 0; i < G; ++i) {
                const int m = g + i;
                a[i] = *((Glob<Half16>)__builtin_assume_aligned(xr[m] + static_cast<std::size_t>(tile_k) * 256, 32));
            }
#pragma unroll
            for (int i = 0; i < G; ++i) {
                acc0[g + i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b0, acc0[g + i]);
                acc1[g + i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b1, acc1[g + i]);
            }
        }
     }
    }
    // acc[m][j] -> row first_row + 16 m + 2 j + lane / 16, column tile * 16 + lane % 16 (wmma_core store layout).
    const int col0 = (2 * tile_pair) * 16 + c, col1 = col0 + 16;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row = first_row + 16 * m + 2 * j + (lane >> 4);
            if (row < rows) {
                float* dst = it.part + static_cast<std::size_t>(row) * N;
                dst[col0] = acc0[m][j];
                dst[col1] = acc1[m][j];
            }
        }
    }
}
static __device__ __forceinline__ std::size_t tiled_at(unsigned R, unsigned ch, unsigned K) {
    return ((static_cast<std::size_t>(R >> 4U) * (K >> 4U) + (ch >> 4U)) << 8U) + ((R & 15U) << 4U) + (ch & 15U);
}
static __device__ __forceinline__ void gather_transform_kernel(const float* x, Tables t) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.pairs * 2 * kHB) return;
    const unsigned p = q / (2 * kHB), rem = q % (2 * kHB), m = rem / kHB, b = rem % kHB;
    const GroupDev& g = t.groups[t.pair_group[p]];
    const std::uint16_t* suh = m == 0 ? g.w.gate.suh : g.w.up.suh;
    const float* xr = x + static_cast<std::size_t>(t.pair_row[p]) * kH;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = b * 128U + lane + 32U * static_cast<unsigned>(j);
        const float scaled = materialize(xr[c] * half_bits(suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    hadamard128(v, lane);
    _Float16* out = t.xt + static_cast<std::size_t>(m) * t.tpairs * kH;
    const unsigned R = g.tbegin + (p - g.pair_begin);
#pragma unroll
    for (int j = 0; j < 4; ++j) out[tiled_at(R, b * 128U + lane + 32U * static_cast<unsigned>(j), kH)] = static_cast<_Float16>(v[j]);
}
static __device__ __forceinline__ void gather_transform_v3_kernel(const float* x, Tables t) {
    __shared__ alignas(16) _Float16 sv[16][kG3Pitch];
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U, tile = blockIdx.x, m = blockIdx.y, R0 = 16U * tile;
    const GroupDev& g = t.groups[t.tile_group[tile]];
    const std::uint16_t* suh = m == 0 ? g.w.gate.suh : g.w.up.suh;
    const float* xr[2];
    bool live[2];
#pragma unroll
    for (unsigned h = 0; h < 2; ++h) {
        const unsigned local = R0 + wave + 8U * h - g.tbegin;
        live[h] = local < g.n;
        xr[h] = live[h] ? x + static_cast<std::size_t>(t.pair_row[g.pair_begin + local]) * kH : x;
    }
    const unsigned ct = threadIdx.x / 32U, w = threadIdx.x % 32U, r = w / 2U, c0 = 16U * ct + 8U * (w % 2U);
    for (unsigned bi = 0; bi < kG3Blocks; ++bi) {
        const unsigned b = blockIdx.z * kG3Blocks + bi;
#pragma unroll
        for (unsigned h = 0; h < 2; ++h) {
            float v[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            if (live[h]) {  // uniform per wave
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned c = b * 128U + lane + 32U * static_cast<unsigned>(j);
                    const float scaled = materialize(xr[h][c] * half_bits(suh[c]));
                    v[j] = static_cast<float>(static_cast<_Float16>(scaled));
                }
                hadamard128(v, lane);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) sv[wave + 8U * h][lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
        }
        __syncthreads();
        _Float16* out = t.xt + static_cast<std::size_t>(m) * t.tpairs * kH + tiled_at(R0, b * 128U, kH) + 256U * ct + 8U * w;
        *reinterpret_cast<uint4*>(out) = *reinterpret_cast<const uint4*>(&sv[r][c0]);
        __syncthreads();  // sv reused by the next block
    }
}
static __device__ __forceinline__ void finish_kernel(Tables t) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.pairs * kIB) return;
    const unsigned p = q / kIB, b = q % kIB;
    const GroupDev& g = t.groups[t.pair_group[p]];
    float gv[4], uv[4];
    finish_block(t.part_g + static_cast<std::size_t>(p) * kI, b, g.w.gate.svh, lane, gv);
    finish_block(t.part_u + static_cast<std::size_t>(p) * kI, b, g.w.up.svh, lane, uv);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = b * 128 + lane + 32U * static_cast<unsigned>(j);
        const float act = swiglu_r15(gv[j], uv[j]);
        const float scaled = materialize(act * half_bits(g.w.down.suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    hadamard128(v, lane);
    const unsigned R = g.tbegin + (p - g.pair_begin);
#pragma unroll
    for (int j = 0; j < 4; ++j) t.at[tiled_at(R, b * 128 + lane + 32U * static_cast<unsigned>(j), kI)] = static_cast<_Float16>(v[j]);
}
static __device__ __forceinline__ void combine_kernel(Tables t, const float* shared, float* out, float* eo) {
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const unsigned q = blockIdx.x * 8U + wave;
    if (q >= t.rows * kHB) return;
    const unsigned tok = q / kHB, b = q % kHB;
    float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (unsigned k = 0; k < kTop; ++k) {
        const unsigned p = t.tok_pair[tok * kTop + k];
        const GroupDev& g = t.groups[t.pair_group[p]];
        float v[4];
        finish_block(t.part_d + static_cast<std::size_t>(p) * kH, b, g.w.down.svh, lane, v);
        if (eo != nullptr) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                eo[(static_cast<std::size_t>(tok) * kTop + k) * kH + b * 128 + lane + 32U * static_cast<unsigned>(j)] = v[j];
        }
        const float w = t.tok_w[tok * kTop + k];
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = __builtin_fmaf(w, v[j], acc[j]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const std::size_t i = static_cast<std::size_t>(tok) * kH + b * 128 + lane + 32U * static_cast<unsigned>(j);
        out[i] = acc[j] + shared[i];
    }
}
};
}  // namespace strix::moe_core
