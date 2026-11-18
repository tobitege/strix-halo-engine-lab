#include "schema.hpp"
namespace strix::weight {
Status bind_ds41(const TensorIndex& index,WeightMap& out) {
    static constexpr FormatId linear[] = {{"plain.f32",1},{"plain.f16",1},{"plain.bf16",1},{"gguf.iq2_xxs",1},{"gguf.q2_k",1},{"gguf.mxfp4",1},{"gguf.q8_0",1},{"row.e4m3-e8m0.g32",1},{"block.e4m3-e8m0.32x32",1},{"exl3.mul1.tiles16",1}};
    static constexpr FormatId plain[] = {{"plain.f32",1},{"plain.f16",1},{"plain.bf16",1}};
    static constexpr detail::RolePattern roles[] = {{"embed",Role::embedding},{"token_embd",Role::embedding},{"lm_head",Role::head},{"output.weight",Role::head},
    {"ffn_gate_inp",Role::router},{"router",Role::router},{"norm",Role::norm},{".bias",Role::bias},
    {"q_proj",Role::attention_q},{"attn_q",Role::attention_q},{"k_proj",Role::attention_k},{"attn_k",Role::attention_k},
    {"v_proj",Role::attention_v},{"attn_v",Role::attention_v},{"o_proj",Role::attention_o},{"attn_output",Role::attention_o},
    {"down",Role::dense_down},{"gate",Role::dense_gate},{"up",Role::dense_up}};
    const detail::FamilySchema schema{"ds41",linear,plain,roles,true,true,false,false};
    return detail::bind_model(index,schema,out);
}
}
