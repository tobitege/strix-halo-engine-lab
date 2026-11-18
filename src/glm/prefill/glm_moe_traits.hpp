#pragma once
#include <cstddef>
#include <cstdint>

namespace strix::glm::pf {
struct MoePolicy {
    static constexpr unsigned kSplits = 1;
    static constexpr unsigned kBitsMin = 3, kBitsMax = 6;
    static constexpr unsigned kRowBlock = 16, kGateUpInterleave = 2;
    static constexpr unsigned kG3Pitch = 136, kG3Blocks = 8;
    static constexpr bool kAscendingRouteOrder = true;
    static constexpr bool accepts_bits(unsigned bits) { return bits >= kBitsMin && bits <= kBitsMax; }
};
}  // namespace strix::glm::pf

#if defined(__HIPCC__)
#include "glm/prefill/glm_pf_moe.hpp"
#include "glm/common/glm_math.hpp"
#include "moe_core/grouped_core.hpp"
#include "glm/prefill/glm_pf_aux.hpp"

namespace strix::glm::pf {
using strix::moe_core::hadamard128;
using strix::moe_core::half_bits;
struct MoeTraits : MoePolicy {
    static constexpr unsigned kE = kMoeExperts, kTop = kMoeTop, kH = kMoeHidden, kI = kMoeInter;
    static constexpr unsigned kHB = kH / 128, kIB = kI / 128;
    template <int Bits> using WeightDecode = strix::moe_core::Exl3Trellis<Bits>;
    template <int Bits> using WeightSource = strix::weight::hip::LegacyMul1Source<static_cast<unsigned>(Bits)>;
    template <int Bits> using Norm = typename WeightDecode<Bits>::Norm;
    template <typename T> using Glob = const __attribute__((address_space(1))) T*;
// One grouped work item: one expert matrix over its group's rows.
struct Item {
    const std::uint32_t* trellis;
    const _Float16* x;  // tile-major transformed rows of the group: [rows/16 padded][K/16][16][16]
    float* part;        // [rows][N] fp32 WMMA results (one split)
    unsigned rows, cta_begin, gy, pad;
};
struct GroupDev {  // one routed expert of the chunk
    MoeExpert w;
    unsigned n, pair_begin, tbegin, pad;  // tbegin: first row in the tile-major layout (multiple of 16)
};
struct Tables {
    const GroupDev* groups;
    const std::uint32_t* tile_group;  // [tpairs / 16] group of each 16-row tile (gather v3)
    const std::uint32_t* pair_row;    // [P] token of the pair
    const std::uint32_t* pair_group;  // [P]
    const std::uint32_t* tok_pair;    // [T][8] pair of route (t, k)
    const float* tok_w;               // [T][8]
    _Float16* xt;                     // [2][tpairs][4096] tile-major (gate, up)
    _Float16* at;                     // [tpairs][1024] tile-major
    float* part_g;                    // [P][1024]
    float* part_u;                    // [P][1024]
    float* part_d;                    // [P][4096]
    unsigned pairs, rows, tpairs, pad;
};
static __device__ __forceinline__ void finish_block(const float* row, unsigned b, const std::uint16_t* svh, unsigned lane, float (&v)[4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = 0.0F;
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] += row[b * 128 + lane + 32U * static_cast<unsigned>(j)];
    hadamard128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = v[j] * half_bits(svh[b * 128 + lane + 32U * static_cast<unsigned>(j)]);
#if defined(STRIX_WFMT_CAPTURE) && STRIX_WFMT_CAPTURE
    for (int j = 0; j < 4; ++j) strix::wfmt_trace::trellis_trace(strix::wfmt_trace::kTrellisFinish, v[j]);
#endif
}
static __device__ __forceinline__ float swiglu_r15(float g, float u) {
    const float gg = g > kClamp ? kClamp : g;
    const float uu = u > kClamp ? kClamp : (u < -kClamp ? -kClamp : u);
    return silu(gg) * uu;
}
};
// Grouped X and WMMA cores share this H128 epilogue. Gate/up are paired
// within one CTA; shared down is launched after all routed-down producers.
template<bool Down>
struct MoeAuxEpilogue {
    template<class Rows, class Mat>
    __device__ __forceinline__ void operator()(Rows rows, Mat m, unsigned r, unsigned col0,
                                               float (&v)[4], unsigned lane) const {
        namespace d = strix::glm::moe::dev;
        const auto& a = rows.a;
        const unsigned pair = a.pairs[rows.tile.first + r];
        if constexpr (!Down) {
            if (a.aux_act && rows.m == 1U) {
                const auto down = a.experts[rows.tile.expert].down;
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j) {
                    const unsigned i = col0 + lane + 32U * j;
                    const float u = d::materialize(v[j] * d::half_bits(m.svh[i]));
                    const float g = a.gu[(std::size_t{pair} * 2U) * MoeTraits::kI + i];
                    const float scaled = d::materialize(MoeTraits::swiglu_r15(g, u) * d::half_bits(down.suh[i]));
                    v[j] = static_cast<float>(static_cast<_Float16>(scaled));
                }
                d::hadamard128(v, lane);
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j)
                    a.at[std::size_t{pair} * MoeTraits::kI + col0 + lane + 32U * j] = static_cast<_Float16>(v[j]);
                return;
            }
        } else if (a.aux_combine && rows.tile.expert == MoeTraits::kE) {
            constexpr unsigned P = MoeTraits::kTop + 1U, H = MoeTraits::kH;
            const unsigned row = pair / P;
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) {
                const unsigned i = col0 + lane + 32U * j;
                const float shared = d::materialize(v[j] * d::half_bits(m.svh[i]));
                float sum = 0.0F;
                for (unsigned k = 0; k < MoeTraits::kTop; ++k)
                    sum = __builtin_fmaf(a.weights[row * MoeTraits::kTop + k],
                                        a.output[(std::size_t{row} * P + k) * H + i], sum);
                a.out[std::size_t{row} * H + i] = sum + shared;
            }
            return;
        }
        aux::Store{}(rows, m, r, col0, v, lane);
    }
};
}  // namespace strix::glm::pf
#endif
