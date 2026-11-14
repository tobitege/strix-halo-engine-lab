#pragma once
// attn_core helper umbrella (canonical for all attention cores; layout per wave4 review t2 section 3):
//   wmma_types.hpp     Operand, Frag/Acc/U4/U2, to_bf16/to_f16/to_operand, operand_value, pack2/pack8, mma, acc_zero
//   wmma_frag.hpp      frag32, frag_from_f32, swizzled tile (piece_slot/slot/frag_swz), acc_row, row_bcast,
//                      row_slot/row_scalars
//   online_softmax.hpp SoftmaxState, online_step<O, K, Lanes, SumOf> (empty-block guard), finalize_scale
//   masks.hpp          mask::window_lo/causal_ok/tile_range/list_ok/list_clamp
//   tile_stage.hpp     pick, store_vt4x8 (V^T staging), sum_partials (fixed order)
// Users: sparse_flash_wmma.hpp (GLM MLA/DSA, MSA shape); flash_core.hpp policies -> dense_flash_prefill.hpp (G4),
// banded_flash_prefill.hpp (DS41 f16wmma), selected_flash_prefill.hpp (QSA); index_score_wmma.hpp (idx-wmma) rebases onto these. API stable: extend, do not change signatures.
#include "attn_core/masks.hpp"
#include "attn_core/online_softmax.hpp"
#include "attn_core/tile_stage.hpp"
#include "attn_core/wmma_frag.hpp"
#include "attn_core/wmma_types.hpp"

namespace strix::attn_core::tile {
// First-published form (m, l as two floats; 8 lanes, rounded sum). Same arithmetic as SoftmaxState's step.
template <Operand O, unsigned K> __device__ __forceinline__ float online_step(const float (&s)[K], float& m, float& l, std::uint16_t (&p)[K]) {
    SoftmaxState st{m, l};
    const float alpha = online_step<O, K, 8, SumOf::rounded>(s, st, p);
    m = st.m;
    l = st.l;
    return alpha;
}
}  // namespace strix::attn_core::tile
