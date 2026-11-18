#include "schema.hpp"
namespace strix::weight {
Status bind_qwen(const TensorIndex& index,WeightMap& out) {
    static constexpr FormatId linear[] = {{"plain.f32",1},{"plain.f16",1},{"plain.bf16",1},{"gguf.q4_0",1},{"gguf.q5_0",1},{"gguf.q5_1",1},{"gguf.q8_0",1},{"gguf.q3_k",1},{"gguf.q4_k",1},{"gguf.q5_k",1},{"gguf.q6_k",1},{"gguf.iq4_nl",1},{"gguf.iq4_xs",1},{"gguf.iq2_s",1},{"gguf.iq3_xxs",1},{"gguf.iq3_s",1},{"gguf.q2_0.strix",1}};
    static constexpr FormatId plain[] = {{"plain.f32",1},{"plain.f16",1},{"plain.bf16",1},{"gguf.q4_0",1},{"gguf.q5_0",1},{"gguf.q5_1",1},{"gguf.q8_0",1},{"gguf.q3_k",1},{"gguf.q4_k",1},{"gguf.q5_k",1},{"gguf.q6_k",1},{"gguf.iq4_nl",1},{"gguf.iq4_xs",1},{"gguf.iq2_s",1},{"gguf.iq3_xxs",1},{"gguf.iq3_s",1},{"gguf.q2_0.strix",1}};
    static constexpr FormatId auxiliary[] = {{"plain.f32",1}};
    static constexpr detail::RolePattern roles[] = {{".ssm_a",Role::auxiliary,true,true},{".ssm_d",Role::auxiliary,true,true},{"embed",Role::embedding},{"token_embd",Role::embedding},{"lm_head",Role::head},{"output.weight",Role::head},
    {"ffn_gate_inp",Role::router},{"router",Role::router},{"norm",Role::norm},{".bias",Role::bias},
    {"q_proj",Role::attention_q},{"attn_q",Role::attention_q},{"k_proj",Role::attention_k},{"attn_k",Role::attention_k},
    {"v_proj",Role::attention_v},{"attn_v",Role::attention_v},{"o_proj",Role::attention_o},{"attn_output",Role::attention_o},
    {"down",Role::dense_down},{"gate",Role::dense_gate},{"up",Role::dense_up}};
    auto native_linear=std::vector<FormatId>(std::begin(linear),std::end(linear));
    native_linear.push_back({"nvidia.nvfp4",1});
    const detail::FamilySchema schema{"qwen",native_linear,plain,roles,false,false,false,false,auxiliary};
    return detail::bind_model(index,schema,out);
}
}
