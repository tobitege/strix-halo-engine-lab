#include "strix/qwen27_model.hpp"
#include "strix/model_family_registry.hpp"
#include "strix/ggml_dequant.hpp"
#include "model_graph/model_graph.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
namespace q = strix::qwen27;
namespace w = strix::weights;
using strix::serve::Json;
using Bytes = std::vector<std::uint8_t>;
int checks = 0;
void check(bool value, const char* message) {
    ++checks; if (!value) throw std::runtime_error(message);
}
void refuses(const std::function<void()>& f) {
    ++checks;
    try { f(); } catch (const std::exception&) { return; }
    throw std::runtime_error("expected malformed/unsupported input to fail");
}
void u32(Bytes& b, std::uint32_t v) { for (unsigned i = 0; i < 4; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8*i))); }
void u64(Bytes& b, std::uint64_t v) { for (unsigned i = 0; i < 8; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8*i))); }
void string(Bytes& b, const std::string& v) { u64(b,v.size()); b.insert(b.end(),v.begin(),v.end()); }
void write(const std::filesystem::path& path, const Bytes& bytes) {
    std::ofstream f(path,std::ios::binary); f.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
    if (!f) throw std::runtime_error("fixture write failed");
}
void json_file(const std::filesystem::path& path, const Json& j) { const auto s = j.dump(); write(path,Bytes(s.begin(),s.end())); }
Bytes floats(std::uint64_t n, float value = 1) {
    Bytes b; for (std::uint64_t i = 0; i < n; ++i) u32(b,std::bit_cast<std::uint32_t>(value)); return b;
}
Json config_json(bool gguf = false) {
    auto j = strix::serve::parse_json_unique(R"({"model_type":"qwen3_8","hidden_size":32,"num_hidden_layers":2,
        "num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":48,"head_dim":8,"vocab_size":4,
        "max_position_embeddings":128,"rope_theta":1000000,"rms_norm_eps":0.000001,"tie_word_embeddings":false,
        "use_sliding_window":false,"sliding_window":4096,
        "rope_scaling":{"rope_type":"yarn","factor":4,"original_max_position_embeddings":32,"beta_fast":32,"beta_slow":1}})");
    if (gguf) { j.set("hidden_size",256); j.set("head_dim",64); j.set("intermediate_size",256); }
    return j;
}
std::uint64_t count(const std::vector<std::uint64_t>& shape) { std::uint64_t n = 1; for (auto d : shape) n *= d; return n; }
struct SafeDesc { std::string name,dtype; std::vector<std::uint64_t> shape; Bytes data; };
void safe_file(const std::filesystem::path& path, const std::vector<SafeDesc>& tensors) {
    Json header = Json::object(); Bytes data;
    for (const auto& t : tensors) {
        Json desc = Json::object(), shape = Json::array(), offsets = Json::array();
        for (auto d : t.shape) shape.push(Json(d));
        offsets.push(Json(std::uint64_t(data.size()))); data.insert(data.end(),t.data.begin(),t.data.end()); offsets.push(Json(std::uint64_t(data.size())));
        desc.set("dtype",t.dtype); desc.set("shape",shape); desc.set("data_offsets",offsets); header.set(t.name,desc);
    }
    auto text = header.dump(); while (text.size() % 8) text += ' ';
    Bytes bytes; u64(bytes,text.size()); bytes.insert(bytes.end(),text.begin(),text.end()); bytes.insert(bytes.end(),data.begin(),data.end()); write(path,bytes);
}
std::vector<SafeDesc> safe_model(const q::Config& c, bool packed) {
    std::vector<SafeDesc> tensors;
    for (const auto& [name,shape] : q::required_tensors(c)) {
        if (packed && shape.size() == 2) {
            const auto base = name.substr(0,name.size()-7);
            Bytes codes(static_cast<std::size_t>(count(shape)/2));
            for (std::size_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<std::uint8_t>(((2*i)%16) | (((2*i+1)%16)<<4));
            Bytes scales(static_cast<std::size_t>(count(shape)/16));
            for (std::size_t i = 0; i < scales.size(); ++i) scales[i] = static_cast<std::uint8_t>(56 + 8*(i%3)); // 1,2,4
            tensors.push_back({name,"U8",{shape[0],shape[1]/2},codes});
            tensors.push_back({base+".weight_scale","F8_E4M3",{shape[0],shape[1]/16},scales});
            tensors.push_back({base+".weight_scale_2","F32",{},floats(1,0.5F)});
        } else tensors.push_back({name,"F32",shape,floats(count(shape))});
    }
    return tensors;
}
std::string gguf_name(const std::string& name, const q::Config& c) {
    if (name == "model.embed_tokens.weight") return "token_embd.weight";
    if (name == "model.norm.weight") return "output_norm.weight";
    if (name == "lm_head.weight") return "output.weight";
    const bool mtp = name.starts_with("mtp.layers.");
    const std::size_t start = mtp ? 11 : 13;
    const auto dot = name.find('.',start);
    const auto index = static_cast<std::uint32_t>(std::stoul(name.substr(start,dot-start))) + (mtp ? c.layers : 0);
    const auto tail = name.substr(dot+1);
    const auto suffix = tail.ends_with(".bias") ? ".bias" : ".weight";
    const auto stem = tail.substr(0,tail.size()-std::strlen(suffix));
    const std::map<std::string,std::string> names = {{"self_attn.q_proj","attn_q"},{"self_attn.k_proj","attn_k"},
        {"self_attn.v_proj","attn_v"},{"self_attn.o_proj","attn_output"},{"self_attn.q_norm","attn_q_norm"},
        {"self_attn.k_norm","attn_k_norm"},{"input_layernorm","attn_norm"},{"post_attention_layernorm","ffn_norm"},
        {"mlp.gate_proj","ffn_gate"},{"mlp.up_proj","ffn_up"},{"mlp.down_proj","ffn_down"},
        {"eh_proj","nextn.eh_proj"},{"enorm","nextn.enorm"},{"hnorm","nextn.hnorm"}};
    return "blk."+std::to_string(index)+"."+names.at(stem)+suffix;
}
Bytes quantized(const std::vector<std::uint64_t>& shape, std::uint32_t type) {
    const auto g = *strix::ggml_dequant::geometry(type);
    Bytes bytes;
    for (std::uint64_t row = 0; row < shape.front(); ++row) for (std::uint64_t col = 0; col < shape.back(); col += g.block_elements) {
        Bytes block(g.block_bytes,0);
        const std::uint16_t d[] = {0x3c00,0x4000,0x4200};
        block[0] = static_cast<std::uint8_t>(d[row%3]); block[1] = static_cast<std::uint8_t>(d[row%3] >> 8);
        if (type == 2) std::fill(block.begin()+2,block.end(),std::uint8_t{0xa9});
        if (type == 20) std::fill(block.begin()+2,block.end(),std::uint8_t{0x98});
        if (type == 12) {
            std::fill(block.begin()+4,block.begin()+8,std::uint8_t{1});
            std::fill(block.begin()+12,block.begin()+16,std::uint8_t{1});
            std::fill(block.begin()+16,block.end(),std::uint8_t{0x32});
        }
        if (type == 23) {
            block[2] = 0xaa; block[3] = 0xaa;
            std::fill(block.begin()+4,block.begin()+8,std::uint8_t{0x11});
            std::fill(block.begin()+8,block.end(),std::uint8_t{0x98});
        }
        bytes.insert(bytes.end(),block.begin(),block.end());
    }
    return bytes;
}
void gguf_file(const std::filesystem::path& path, const q::Config& c, std::uint32_t type, bool missing = false, bool wrong_shape = false) {
    struct Meta { std::string name; std::uint32_t type; Bytes data; };
    std::vector<Meta> meta;
    const auto dim = [&](const std::string& key,std::uint32_t n) { Bytes b; u32(b,n); meta.push_back({c.model_type+"."+key,4,b}); };
    const auto real = [&](const std::string& key,double n) { Bytes b; u64(b,std::bit_cast<std::uint64_t>(n)); meta.push_back({c.model_type+"."+key,12,b}); };
    Bytes arch; string(arch,c.model_type); meta.push_back({"general.architecture",8,arch});
    dim("embedding_length",c.hidden); dim("block_count",c.layers); dim("attention.head_count",c.heads);
    dim("attention.head_count_kv",c.kv_heads); dim("attention.key_length",c.head_dim); dim("feed_forward_length",c.intermediate);
    dim("context_length",c.context); dim("rope.dimension_count",c.rope.dimensions); dim("nextn_predict_layers",c.mtp_layers);
    real("rope.freq_base",c.rope.theta); real("attention.layer_norm_rms_epsilon",c.rms_epsilon);
    Bytes rope; string(rope,c.rope.type); meta.push_back({c.model_type+".rope.scaling.type",8,rope});
    real("rope.scaling.factor",c.rope.factor); dim("rope.scaling.original_context_length",c.rope.original_context);
    Bytes tokens; u32(tokens,8); u64(tokens,c.vocab); for (std::uint32_t i = 0; i < c.vocab; ++i) string(tokens,"t"+std::to_string(i));
    meta.push_back({"tokenizer.ggml.tokens",9,tokens});
    auto required = q::required_tensors(c);
    if (missing) required.erase("model.layers.0.self_attn.q_proj.weight");
    if (wrong_shape) required["model.layers.0.self_attn.q_proj.weight"][0] += 1;
    Bytes header = {'G','G','U','F'}; u32(header,3); u64(header,required.size()); u64(header,meta.size());
    for (const auto& m : meta) { string(header,m.name); u32(header,m.type); header.insert(header.end(),m.data.begin(),m.data.end()); }
    Bytes payload;
    for (const auto& [name,shape] : required) {
        while (payload.size()%32) payload.push_back(0);
        string(header,gguf_name(name,c)); u32(header,static_cast<std::uint32_t>(shape.size()));
        for (auto it = shape.rbegin(); it != shape.rend(); ++it) u64(header,*it);
        u32(header,shape.size() == 2 ? type : 0); u64(header,payload.size());
        const auto data = shape.size() == 2 ? quantized(shape,type) : floats(count(shape));
        payload.insert(payload.end(),data.begin(),data.end());
    }
    while (header.size()%32) header.push_back(0);
    header.insert(header.end(),payload.begin(),payload.end()); write(path,header);
}
void test_config(const std::filesystem::path& dir) {
    const auto path = dir/"config.json"; auto j = config_json(); json_file(path,j);
    const auto c = q::parse_hf_config(path);
    check(c.hidden == 32 && c.layers == 2 && c.heads == 4 && c.kv_heads == 2 && c.head_dim == 8 && c.intermediate == 48 && c.vocab == 4,"HF dimensions");
    check(c.rope.type == "yarn" && c.rope.factor == 4 && c.rope.original_context == 32 && c.rope.dimensions == 8,"HF YaRN");
    auto nested = Json::object(); nested.set("text_config",j); json_file(path,nested); check(q::parse_hf_config(path).hidden == 32,"nested HF text config");
    for (const auto& [key,value] : std::vector<std::pair<std::string,Json>>{{"hidden_size",-1},{"num_attention_heads",0},
        {"num_key_value_heads",3},{"head_dim",7},{"model_type","qwen3_moe"},{"num_experts",8},{"rms_norm_eps",0},
        {"hidden_size",Json(1.5)},{"num_hidden_layers",70000},{"head_dim",Json(std::uint64_t(UINT32_MAX)+1)},
        {"use_sliding_window",true},{"hidden_act","gelu"}}) {
        auto bad = j; bad = strix::serve::parse_json_unique(j.dump()); bad.set(key,value); json_file(path,bad); refuses([&] { (void)q::parse_hf_config(path); });
    }
    for (const auto& text : {R"({"rope_type":"yarn","factor":0,"original_max_position_embeddings":32})",
        R"({"rope_type":"yarn","factor":4})",R"({"rope_type":"dynamic","factor":4})",
        R"({"rope_type":"yarn","factor":4,"original_max_position_embeddings":32,"partial_rotary_factor":0.3})",
        R"({"rope_type":"yarn","type":"linear","factor":4,"original_max_position_embeddings":32})"}) {
        auto bad = strix::serve::parse_json_unique(j.dump()); bad.set("rope_scaling",strix::serve::parse_json_unique(text));
        json_file(path,bad); refuses([&] { (void)q::parse_hf_config(path); });
    }
    auto aliases = config_json();
    const auto linear = strix::serve::parse_json_unique(R"({"rope_type":"linear","factor":2})");
    aliases.set("rope_parameters",Json()); aliases.set("rope_scaling",linear); json_file(path,aliases);
    const auto fallback = q::parse_hf_config(path);
    check(fallback.rope.type == "linear" && fallback.rope.factor == 2,"null RoPE alias falls back to scaling");
    aliases.set("rope_parameters",linear); json_file(path,aliases);
    refuses([&] { (void)q::parse_hf_config(path); });
    aliases.set("rope_scaling",Json()); json_file(path,aliases);
    check(q::parse_hf_config(path).rope == fallback.rope,"null scaling preserves RoPE parameters");
    aliases.set("rope_parameters",Json()); json_file(path,aliases);
    check(q::parse_hf_config(path).rope.type == "default","both null RoPE aliases use native policy");
    j.set("rope_scaling",Json()); json_file(path,j); check(q::parse_hf_config(path).rope.type == "default","native RoPE");
    auto graph = q::graph(c); check(graph.layers.size() == 2 && graph.layers[0].attention == strix::model_graph::AttentionFamily::dense,"dense graph");
    check(graph.model == "qwen-dense","dense-family graph identity");
    check(graph.layers[0].mlps[0].kind == strix::model_graph::MlpKind::dense && graph.layers[0].states.size() == 1,"shared dense/attention declaration");
    check(graph.attachments.empty(),"MTP absent by default");
    auto mtp = c; mtp.mtp_layers = 1;
    const auto mg = q::graph(mtp); check(mg.attachments.size() == 1 && mg.attachments[0].layers[0].id == c.layers,"MTP attachment outside trunk");
    check(strix::model_graph::identity(graph) != strix::model_graph::identity(mg),"MTP descriptive identity");
    auto changed = c; changed.rope.beta_slow = 2;
    check(strix::model_graph::identity(graph) != strix::model_graph::identity(q::graph(changed)),"RoPE semantic identity");
}
void test_gguf(const std::filesystem::path& dir) {
    const auto cfg = dir/"config.json"; json_file(cfg,config_json(true));
    auto c = q::parse_hf_config(cfg); c.mtp_layers = 1;
    for (auto type : {2U,12U,20U,23U}) {
        const auto path = dir/("tiny-"+std::to_string(type)+".gguf"); gguf_file(path,c,type);
        auto model = q::load_gguf(path); const auto& gc = model.config;
        check(gc.hidden == c.hidden && gc.layers == c.layers && gc.heads == c.heads && gc.kv_heads == c.kv_heads &&
            gc.intermediate == c.intermediate && gc.vocab == c.vocab && gc.head_dim == c.head_dim && gc.rope == c.rope,"HF/GGUF configuration agreement");
        check(model.tensors.contains("mtp.layers.0.eh_proj.weight"),"MTP GGUF mapping");
        check(model.weight("model.layers.0.self_attn.k_proj.weight").tensor.shape == std::vector<std::uint64_t>({128,256}),"reverse GGUF dimensions once");
        auto view = model.weight("model.layers.0.self_attn.q_proj.weight"); model = {};
        for (auto row : {0U,1U,255U}) {
            const auto values = view.decode_row(row);
            for (std::size_t col = 0; col < values.size(); ++col) {
                const auto period = type == 12 ? 64U : 32U;
                const float lo = type == 12 ? 2.0F : 1.0F, hi = type == 12 ? 3.0F : type == 2 ? 2.0F : 13.0F;
                check(values[col] == (col%period < period/2 ? lo : hi) * static_cast<float>(row%3+1),"independent quantized row golden");
            }
        }
        refuses([&] { (void)view.decode_row(256); });
    }
    auto tied = c; tied.tied_embeddings = true; tied.mtp_layers = 0;
    const auto path = dir/"tied.gguf"; gguf_file(path,tied,2);
    const auto model = q::load_gguf(path); check(model.config.tied_embeddings,"GGUF inferred tied head");
    check(model.weight("lm_head.weight").tensor.storage == model.weight("model.embed_tokens.weight").tensor.storage,"tied head storage lifetime");
    gguf_file(dir/"missing.gguf",c,12,true); refuses([&] { (void)q::load_gguf(dir/"missing.gguf"); });
    gguf_file(dir/"wrong.gguf",c,12,false,true); refuses([&] { (void)q::load_gguf(dir/"wrong.gguf"); });
    gguf_file(dir/"truncated.gguf",c,12); std::filesystem::resize_file(dir/"truncated.gguf",64); refuses([&] { (void)q::load_gguf(dir/"truncated.gguf"); });
    auto info = strix::gguf::read_file(path);
    auto& vocab = *std::find_if(info.metadata.begin(),info.metadata.end(),[](const auto& m) { return m.key == "tokenizer.ggml.tokens"; });
    vocab.array_length = 99; refuses([&] { (void)q::parse_gguf_config(info); });
    refuses([&] { (void)q::canonical_gguf_name("blk.999.attn_q.weight",c); });
    refuses([&] { (void)q::canonical_gguf_name("blk.x.attn_q.weight",c); });
    refuses([&] { (void)q::canonical_gguf_name("blk.0.unknown.weight",c); });
    q::register_family(); q::register_family();
    check(strix::admit_model_file(path).family->name == "qwen27","family registry admission");
}
void test_safetensors(const std::filesystem::path& dir) {
    auto j = config_json(); j.set("tie_word_embeddings",true); j.set("num_nextn_predict_layers",1); json_file(dir/"config.json",j);
    const auto c = q::parse_hf_config(dir/"config.json"); auto tensors = safe_model(c,true);
    const auto base = std::string("model.layers.0.self_attn.q_proj");
    tensors.push_back({base+".pre_quant_scale","F32",{c.hidden},floats(c.hidden,2)});
    tensors.push_back({base+".input_scale","F16",{},Bytes{0,0x3c}});
    tensors.push_back({base+".input_scale_2","F32",{1},floats(1,3)});
    safe_file(dir/"model.safetensors",tensors);
    auto model = q::load_safetensors(dir); const auto& view = model.weight(base+".weight");
    check(view.nvfp4.has_value() && view.pre_quant_scale.has_value() && view.input_scale && view.input_scale_2,"NVFP4 and input scale views");
    check(model.weight("lm_head.weight").nvfp4.has_value(),"tied NVFP4 head");
    check(model.weight("mtp.layers.0.eh_proj.weight").nvfp4.has_value(),"NVFP4 MTP binding");
    constexpr float expected[16] = {0,.5F,1,1.5F,2,3,4,6,0,-.5F,-1,-1.5F,-2,-3,-4,-6};
    for (std::uint64_t row : {0ULL,1ULL,31ULL}) {
        const auto v = view.decode_row(row);
        for (std::size_t i = 0; i < v.size(); ++i) {
            const auto block = (row*c.hidden+i)/16;
            const float scale = block%3 == 0 ? 1.0F : block%3 == 1 ? 2.0F : 4.0F;
            check(v[i] == expected[i%16]*scale*.5F,"all E2M1 codes, adjacent packing, per-row block/global scales");
        }
    }
    check(view.tensor.bytes().size() == std::uint64_t(c.heads)*c.head_dim*c.hidden/2,"packed bytes retained");
    unsigned bad_case = 0;
    const auto mutate = [&](const std::function<void(std::vector<SafeDesc>&)>& f) {
        const auto target = dir/("bad-"+std::to_string(bad_case++));
        std::filesystem::create_directories(target); json_file(target/"config.json",j);
        auto bad = tensors; f(bad); safe_file(target/"model.safetensors",bad); refuses([&] { (void)q::load_safetensors(target); });
    };
    mutate([&](auto& t) { t.erase(std::find_if(t.begin(),t.end(),[&](const auto& a) { return a.name == base+".weight_scale"; })); });
    mutate([&](auto& t) { for (auto& a : t) if (a.name == base+".weight_scale_2") a.data = floats(1,0); });
    mutate([&](auto& t) { for (auto& a : t) if (a.name == base+".weight_scale_2") a.data = floats(1,std::numeric_limits<float>::infinity()); });
    mutate([&](auto& t) { for (auto& a : t) if (a.name == base+".weight_scale") a.data[0] = 127; });
    mutate([&](auto& t) { for (auto& a : t) if (a.name == base+".weight_scale") a.data[0] = 184; });
    mutate([&](auto& t) { for (auto& a : t) if (a.name == base+".weight_scale") a.dtype = "U8"; });
    mutate([&](auto& t) { for (auto& a : t) if (a.name == base+".weight") a.shape[1] *= 2; });
    mutate([](auto& t) { t.push_back({"unmapped.weight","F32",{1},floats(1)}); });
    mutate([&](auto& t) { for (auto& a : t) if (a.name == base+".pre_quant_scale") a.data = floats(c.hidden,-1); });
    const auto plain_dir = dir/"plain"; std::filesystem::create_directories(plain_dir); json_file(plain_dir/"config.json",j);
    auto plain = safe_model(c,false); safe_file(plain_dir/"model.safetensors",plain);
    check(q::load_safetensors(plain_dir).weight(base+".weight").decode_row(1)[0] == 1,"unquantized scalar matrix import");
    // Complete two-shard checkpoint including scales split from their packed matrices.
    std::vector<SafeDesc> a,b; Json index = Json::object(), map = Json::object();
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        (i%2 ? a : b).push_back(tensors[i]); map.set(tensors[i].name,i%2 ? "a.safetensors" : "b.safetensors");
    }
    safe_file(dir/"a.safetensors",a); safe_file(dir/"b.safetensors",b); index.set("weight_map",map);
    json_file(dir/"model.safetensors.index.json",index);
    check(q::load_safetensors(dir).weight(base+".weight").decode_row(0)[7] == 3,"sharded NVFP4 import");
    map.set(tensors[0].name,"../outside.safetensors"); index.set("weight_map",map); json_file(dir/"model.safetensors.index.json",index);
    refuses([&] { (void)q::load_safetensors(dir); });
    // Independent E4M3 edge values: zero, subnormal, minimum normal and maximum finite.
    const std::vector<std::uint8_t> scale_codes = {0,1,7,8,55,56,126};
    const float scale_values[] = {0,1.0F/512,7.0F/512,1.0F/64,15.0F/16,1,448};
    const auto edge_path = dir/"edge.safetensors";
    safe_file(edge_path,{{"edge.weight","U8",{1,56},Bytes(56,0x22)},
        {"edge.weight_scale","F8_E4M3FN",{1,7},scale_codes},{"edge.weight_scale_2","F32",{1},floats(1,0.5F)}});
    auto edge = w::nvfp4(w::load_safetensors(edge_path),"edge",1,112);
    const auto edge_values = edge.decode_row(0);
    for (std::size_t i = 0; i < edge_values.size(); ++i) check(edge_values[i] == scale_values[i/16]*0.5F,"E4M3 boundary golden");
    refuses([&] { (void)edge.decode_row(1); });
    refuses([&] { (void)w::nvfp4(w::load_safetensors(edge_path),"edge",1,111); });
    // Header bounds, duplicate keys, overlaps, holes and arithmetic overflow.
    for (const auto* text : {R"({"x":{"dtype":"U8","shape":[2],"data_offsets":[0,1]}})",
        R"({"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"y":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}})",
        R"({"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}})",R"({"x":{},"x":{}})",
        R"({"x":{"dtype":"F32","shape":[9223372036854775807,8],"data_offsets":[0,2]}})"}) {
        Bytes bytes; const std::string s(text); u64(bytes,s.size()); bytes.insert(bytes.end(),s.begin(),s.end()); bytes.resize(bytes.size()+2);
        write(dir/"bad.safetensors",bytes); refuses([&] { (void)w::read_safetensors(dir/"bad.safetensors"); });
    }
    Bytes short_header; u64(short_header,UINT64_MAX); write(dir/"short.safetensors",short_header);
    refuses([&] { (void)w::read_safetensors(dir/"short.safetensors"); });
}
}
int main() {
    // All fixtures are generated in the build tree; no model assets, network or GPU.
    try {
        const auto root = std::filesystem::current_path()/"qwen27-synthetic";
        std::filesystem::create_directories(root/"config"); std::filesystem::create_directories(root/"gguf");
        std::filesystem::create_directories(root/"safe");
        // Use fresh names on repeated runs so stale malformed index fixtures cannot influence loading.
        const auto run = root/"safe"/std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        std::filesystem::create_directories(run);
        test_config(root/"config"); test_gguf(root/"gguf"); test_safetensors(run);
        std::cout << checks << " host checks passed\n"; return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL after " << checks << " checks: " << e.what() << '\n'; return 1; }
}
