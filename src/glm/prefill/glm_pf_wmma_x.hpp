#pragma once
#include "glm/prefill/glm_pf_aux.hpp"
// GLM prefill FAST identity glm-pf-wmma-x (hwops G-01b): the exact X cores' operands on v_wmma_f32_16x16x16_f16.
// Same FP16 operands as exact_x::core_s (trellis tiles decoded with mul1_pair, transformed FP16 rows), same epilogue
// (H128 on the 128-output block, svh scale). The accumulation is new: per (row, output) and stage of four k-units one
// WMMA chain from +0, stages added in ascending order into an FP32 total (v_add_f32), no K splits and no reduce4, so
// the bits are a new identity (FAST only, opt-in).
//   CTA  = 8 waves x 128 outputs (wave w = output tile w, the tile it decodes in the exact cores) x 16 * MT rows.
//   k    = stages of four 16-wide k-units: the activation rows are staged in LDS once per CTA, each wave transposes its
//          decoded tile through its own LDS rows into a B fragment (lane = output, 16 k).
#include "glm/moe/glm_trellis_device.hpp"
#include "glm/prefill/glm_pf_wmma.hpp"
#include "moe_core/grouped_f16.hpp"
#include "strix/env_flags.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strix::glm::pf::wmma_x {
namespace d = moe::dev;

// enabled() (STRIX_GLM_PF_WMMA=1) lives in glm_pf_wmma.hpp (host side, also read by the rank startup agreement).

// Full tiles of 128 rows (MT 8: each decoded weight tile feeds eight WMMAs; measured whole-call MoE -27..-30 % vs
// -24..-26 % at 64 rows); STRIX_GLM_PF_WMMA_ROWS=64 keeps 64-row tiles. Same arithmetic per (row, output): same bits.
inline unsigned tile_rows() {
    const char* p = std::getenv("STRIX_GLM_PF_WMMA_ROWS");
    return p && std::strcmp(p, "64") == 0 ? 64U : 128U;
}

// 128-row tiles: each wave computes two output tiles x four row tiles (CP 2: A fragments reused across the pair,
// 1.5 instead of 2.25 b128 LDS reads per WMMA) or one output tile x eight row tiles (CP 1,
// STRIX_GLM_PF_WMMA_CP=1). Decode and every (row tile, output tile) WMMA chain are unchanged: same bits.
inline unsigned cols_per_wave() {
    const char* p = std::getenv("STRIX_GLM_PF_WMMA_CP");
    return p && std::strcmp(p, "1") == 0 ? 1U : 2U;
}

// Routed MoE groups: a remainder of 65..127 rows joins the 128-row tiles (STRIX_GLM_PF_WMMA_TAIL128=0 keeps it on the
// 64-row tail kernel). Same arithmetic per (row, output): same bits.
inline unsigned tail128() {
    const char* p = std::getenv("STRIX_GLM_PF_WMMA_TAIL128");
    return p && std::strcmp(p, "0") == 0 ? 0U : 1U;
}

// Routed MoE full tiles over S CTAs per (expert, output block): STRIX_GLM_PF_WMMA_SPLIT=1..8, default 2 (t8: MoE whole
// call -0.3..-1.3 % vs 1). Same bits.
inline unsigned split() {
    const char* p = std::getenv("STRIX_GLM_PF_WMMA_SPLIT");
    const unsigned v = p ? static_cast<unsigned>(std::strtoul(p, nullptr, 10)) : 2U;
    return v >= 1U && v <= 8U ? v : 2U;
}

// Ragged tiles skip the WMMA chains of row fragments that hold no row (default; STRIX_GLM_PF_WMMA_LIVE_FRAG=0 computes
// them as before). Stored bits are the same in both forms. Returns the kernels' all_fragments argument.
inline unsigned all_fragments() { return envcfg::flag("STRIX_GLM_PF_WMMA_LIVE_FRAG", true) ? 0U : 1U; }

// The device body is the shared folded F16 core (moe_core/grouped_f16.hpp); GLM binds its trellis decode, the K64
// fold and its input / output policies.
struct Exl3Dec {
    template<int Bits> using LaneGeom = d::LaneGeom<Bits>;
    template<int Bits> using TileStates = d::TileStates<Bits>;
    template<class T> static __device__ __forceinline__ const T* uniform(const T* p) { return d::uniform(p); }
    static __device__ __forceinline__ unsigned uni(unsigned v) { return static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(v))); }
    template<class... A> static __device__ __forceinline__ auto load_u32(A&&... a) { return d::load_u32(static_cast<A&&>(a)...); }
    template<class... A> static __device__ __forceinline__ auto mul1_pair(A&&... a) { return d::mul1_pair(static_cast<A&&>(a)...); }
    template<class... A> static __device__ __forceinline__ void hadamard128(A&&... a) { d::hadamard128(static_cast<A&&>(a)...); }
};
using Fold = moe_core::folded_f16::FoldK64;
using moe_core::folded_f16::kRowsMax;
template<unsigned NR, unsigned ActivationPitch = 36> using StorageR = moe_core::folded_f16::StorageR<NR, ActivationPitch>;
using Storage = StorageR<64>;
using Storage128 = StorageR<128>;

// Rows: input(r) = transformed FP16 row, output(r) = FP32 destination (exact_x interface). nr <= 16 * MT.
// all_fragments: compute the row fragments past nr too (STRIX_GLM_PF_WMMA_LIVE_FRAG=0; same stored bits).
template<int Bits, unsigned MT, unsigned CP = 1, class Mat, class Rows, class Lds, class Output = aux::Store, class Input = aux::StagedInput>
__device__ __forceinline__ void core(Mat m, Rows rows, unsigned nr, Lds& lds, Output output = {}, Input input = {}, bool all_fragments = false) {
    moe_core::folded_f16::core<Bits, MT, CP, Exl3Dec, Fold>(m, rows, nr, lds, output, input, all_fragments);
}
// A row span [0, n) in tiles of 64 (or 128) rows, the last one on the smallest MT covering it (uniform per CTA).
template<int Bits, unsigned CP = 1, class Mat, class RowsAt, class Lds, class Output = aux::Store, class Input = aux::StagedInput>
__device__ __forceinline__ void span(Mat m, RowsAt at, unsigned n, Lds& lds, Output output = {}, Input input = {}, bool all_fragments = false) {
    moe_core::folded_f16::span<Bits, CP, Exl3Dec, Fold>(m, at, n, lds, output, input, all_fragments);
}
} // namespace strix::glm::pf::wmma_x
