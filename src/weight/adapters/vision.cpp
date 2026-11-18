#include "schema.hpp"
namespace strix::weight {
Status bind_vision(const TensorIndex& index,WeightMap& out) {
    static constexpr FormatId linear[] = {{"plain.f32",1},{"plain.f16",1},{"plain.bf16",1}};
    static constexpr FormatId plain[] = {{"plain.f32",1},{"plain.f16",1},{"plain.bf16",1}};
    static constexpr detail::RolePattern roles[] = {{"embed",Role::embedding},{"token_embd",Role::embedding},{"lm_head",Role::head},{"output.weight",Role::head},
    {"ffn_gate_inp",Role::router},{"router",Role::router},{"norm",Role::norm},{".bias",Role::bias},
    {"q_proj",Role::attention_q},{"attn_q",Role::attention_q},{"k_proj",Role::attention_k},{"attn_k",Role::attention_k},
    {"v_proj",Role::attention_v},{"attn_v",Role::attention_v},{"o_proj",Role::attention_o},{"attn_output",Role::attention_o},
    {"down",Role::dense_down},{"gate",Role::dense_gate},{"up",Role::dense_up},{"",Role::dense_up,true}};
    const detail::FamilySchema schema{"vision",linear,plain,roles,false,false,false,false};
    return detail::bind_model(index,schema,out);
}
}
