#include "dense_adapter.hpp"
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>

namespace strix::model_graph {
namespace {
std::string scalar(double value) {
    std::ostringstream out; out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10) << value; return out.str();
}
LayerDesc dense_layer(const model::DenseConfig& c, std::uint64_t id) {
    LayerDesc l;
    l.id = id; l.attention = AttentionFamily::dense;
    l.dimensions = {{"hidden",c.hidden},{"heads",c.heads},{"kv_heads",c.kv_heads},{"head_dim",c.head_dim}};
    l.mlps = {{MlpKind::dense,c.intermediate,0,0}};
    l.norms = {{"rmsnorm",scalar(c.rms_epsilon),NormPlacement::pre_attention},
               {"rmsnorm",scalar(c.rms_epsilon),NormPlacement::pre_mlp}};
    l.semantics = {{"attention","causal-gqa"},{"activation","silu-gated"},
        {"qk_norm",c.qk_norm ? "rmsnorm-per-head" : "none"},
        {"residual","sequential-pre-norm"},{"attention_bias",c.attention_bias ? "true" : "false"},
        {"mlp_bias",c.mlp_bias ? "true" : "false"}};
    const auto h = std::to_string(c.hidden), d = std::to_string(c.head_dim);
    const auto q = std::to_string(std::uint64_t(c.heads) * c.head_dim);
    const auto kv = std::to_string(std::uint64_t(c.kv_heads) * c.head_dim);
    const auto f = std::to_string(c.intermediate);
    l.tensors = {{"attention.q","weight",{q,h},"replicated"},{"attention.k","weight",{kv,h},"replicated"},
        {"attention.v","weight",{kv,h},"replicated"},{"attention.o","weight",{h,q},"replicated"},
        {"mlp.gate","weight",{f,h},"replicated"},{"mlp.up","weight",{f,h},"replicated"},
        {"mlp.down","weight",{h,f},"replicated"},{"attention.norm","scalar",{h},"replicated"},
        {"mlp.norm","scalar",{h},"replicated"}};
    if (c.qk_norm) {
        l.tensors.push_back({"attention.q_norm","scalar",{d},"replicated"});
        l.tensors.push_back({"attention.k_norm","scalar",{d},"replicated"});
    }
    if (c.attention_bias) for (const auto& [role, n] : std::vector<std::pair<std::string,std::string>>{
        {"q",q},{"k",kv},{"v",kv},{"o",h}}) l.tensors.push_back({"attention." + role + ".bias","scalar",{n},"replicated"});
    if (c.mlp_bias) for (const auto& [role, n] : std::vector<std::pair<std::string,std::string>>{
        {"gate",f},{"up",f},{"down",h}}) l.tensors.push_back({"mlp." + role + ".bias","scalar",{n},"replicated"});
    l.states = {{{"attention_kv","backend-selected",{"positions","2",std::to_string(c.kv_heads),d},"sequence-local"},"sequence"}};
    return l;
}
}
ModelGraph dense_decoder_graph(const model::DenseConfig& c, std::string family, std::string version) {
    model::validate(c);
    ModelGraph g; g.model = std::move(family); g.semantic_version = std::move(version);
    g.dimensions = {{"hidden",c.hidden},{"layers",c.layers},{"vocab",c.vocab},{"context",c.context},
        {"heads",c.heads},{"kv_heads",c.kv_heads},{"head_dim",c.head_dim},{"intermediate",c.intermediate},
        {"rope.dimensions",c.rope.dimensions},{"rope.original_context",c.rope.original_context},{"mtp.layers",c.mtp_layers}};
    g.semantics = {{"core.dense","core"},{"core.attention","attn_core"},{"core.norm","core"},
        {"core.mtp","shared-speculation-contract"},{"rope.type",c.rope.type},{"rope.theta",scalar(c.rope.theta)},
        {"rope.factor",scalar(c.rope.factor)},{"rope.attention_factor",scalar(c.rope.attention_factor)},
        {"rope.beta_fast",scalar(c.rope.beta_fast)},{"rope.beta_slow",scalar(c.rope.beta_slow)},
        {"rms_epsilon",scalar(c.rms_epsilon)},{"tied_embeddings",c.tied_embeddings ? "true" : "false"},
        {"qualification","host-declaration-only"}};
    g.inputs = {{"token_ids","u32",{"tokens"},"request"}};
    g.outputs = {{"logits","fp32",{"tokens",std::to_string(c.vocab)},"request"}};
    g.tensors = {{"embedding","weight",{std::to_string(c.vocab),std::to_string(c.hidden)},"replicated"},
        {"output_norm","scalar",{std::to_string(c.hidden)},"replicated"},
        {"lm_head",c.tied_embeddings ? "embedding-alias" : "weight",{std::to_string(c.vocab),std::to_string(c.hidden)},"replicated"}};
    for (std::uint32_t i = 0; i < c.layers; ++i) {
        g.layers.push_back(dense_layer(c,i));
        if (i) g.edges.push_back({"layer:" + std::to_string(i-1),"layer:" + std::to_string(i),EdgeKind::data});
    }
    if (c.mtp_layers) {
        AttachmentDesc a; a.kind = AttachmentKind::mtp; a.name = "mtp"; a.target = "trunk-hidden-and-token-embedding";
        for (std::uint32_t i = 0; i < c.mtp_layers; ++i) {
            auto l = dense_layer(c,std::uint64_t(c.layers) + i);
            l.block = BlockKind::auxiliary;
            l.semantics["mtp.input"] = "concat-normalized-embedding-and-hidden";
            l.tensors.push_back({"mtp.eh_proj","weight",{std::to_string(c.hidden),std::to_string(std::uint64_t(c.hidden)*2)},"replicated"});
            l.tensors.push_back({"mtp.enorm","scalar",{std::to_string(c.hidden)},"replicated"});
            l.tensors.push_back({"mtp.hnorm","scalar",{std::to_string(c.hidden)},"replicated"});
            a.layers.push_back(std::move(l));
        }
        g.attachments.push_back(std::move(a));
    }
    validate(g); return g;
}
} // namespace strix::model_graph
