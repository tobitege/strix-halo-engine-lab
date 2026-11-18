#include "strix/weight/admission.hpp"
#include "strix/weight/registry.hpp"
#include "strix/serve/json.hpp"
#include "strix/ds41_native_fp8.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace strix::weight;
bool wfmt_reference_admission_test();
namespace {
void check(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
struct CountingSource final:Source {
    std::shared_ptr<const Source> inner;mutable std::size_t bytes_read=0;
    explicit CountingSource(std::shared_ptr<const Source> source):inner(std::move(source)){}
    Status read(ByteRange range,std::span<Byte> out) const override {bytes_read+=out.size();return inner->read(range,out);}
    std::uint64_t size() const noexcept override{return inner->size();}
    Digest256 identity() const override{return inner->identity();}
};
struct Fixture {std::string name,dtype;std::vector<std::uint64_t> shape;std::vector<Byte> bytes;};
std::vector<Byte> floats(std::size_t n){std::vector<Byte> out(n*4);float one=1;for(std::size_t i=0;i<n;++i)std::memcpy(out.data()+i*4,&one,4);return out;}
TensorIndex fixture(std::vector<Fixture> fields) {
    using strix::serve::Json;auto header=Json::object();std::uint64_t offset=0;
    for(const auto& f:fields){auto entry=Json::object();entry.set("dtype",f.dtype);auto shape=Json::array();for(auto n:f.shape)shape.as_array().push_back(Json(n));entry.set("shape",shape);auto offsets=Json::array();offsets.as_array().push_back(Json(offset));offset+=f.bytes.size();offsets.as_array().push_back(Json(offset));entry.set("data_offsets",offsets);header.set(f.name,entry);}
    const auto text=header.dump();std::vector<Byte> bytes;for(unsigned i=0;i<8;++i)bytes.push_back(Byte((text.size()>>(i*8))&255));for(char c:text)bytes.push_back(Byte(c));for(const auto& f:fields)bytes.insert(bytes.end(),f.bytes.begin(),f.bytes.end());
    TensorIndex index;check(bool(index_safetensors(memory_source(bytes,"schema-fixture"),index)),"schema fixture parses");return index;
}
std::vector<Byte> text_bytes(std::string_view text){return {reinterpret_cast<const Byte*>(text.data()),reinterpret_cast<const Byte*>(text.data()+text.size())};}
void rank_bounds() {
    auto index=fixture({{"x.weight","F32",{1},floats(1)}});
    TensorBinding binding{};binding.tensor=index.tensors[0].id;binding.role=Role::dense_up;
    binding.logical_shape=index.tensors[0].shape;binding.expert_axis=UINT32_MAX;
    binding.admitted_formats={{"plain.f32",1}};
    WeightMap map;check(bool(admit(builtin_registry(),{&index,{&binding,1},{}},map)),"rank fixture admitted");
    const auto original=describe_admission(map);
    index.tensors[0].shape.rank=5;binding.logical_shape.rank=5;
    auto status=admit(builtin_registry(),{&index,{&binding,1},{}},map);
    check(status.code==Error::unsupported_shape && status.detail=="storage rank exceeds four","constructed storage rank rejected before strides");
    check(describe_admission(map)==original,"over-rank admission leaves output intact");
    index.tensors[0].shape.rank=1;
    status=admit(builtin_registry(),{&index,{&binding,1},{}},map);
    check(status.code==Error::unsupported_shape && status.detail=="binding rank exceeds four","logical rank rejected before normalization");
    auto entry=map[0];entry.binding.logical_shape.rank=5;entry.weight.shape.rank=5;
    status=validate_binding(builtin_registry(),entry);
    check(status.code==Error::unsupported_shape && status.detail=="binding rank exceeds four","matching rank-five shapes rejected before comparison");
    check(validate_map(builtin_registry(),{entry}).code==Error::unsupported_shape,"map rejects constructed ranks");
    entry.binding.logical_shape.rank=1;
    status=validate_binding(builtin_registry(),entry);
    check(status.code==Error::unsupported_shape && status.detail=="weight rank exceeds four","weight rank checked before shape comparison");
}
void qwen_cuda_alpha_sources() {
    // CUDA transcodes these GGUF projection inputs to F32 after shared admission.
    for(const auto type:{1U,30U,8U}) {
        const std::size_t elements=48*2560;
        auto source=memory_source(std::vector<Byte>(type==8 ? elements/32*34 : elements*2));
        TensorIndex index;index.container=Container::gguf;index.owner=source;
        RawTensor raw{};raw.name="blk.0.ssm_alpha.weight";raw.wire_type=type;
        raw.id=tensor_id(*source,raw.name);raw.shape={{{48,2560,0,0}},2};raw.payload={source,{0,source->size()}};
        index.tensors.push_back(raw);WeightMap map;
        check(bool(bind_qwen(index,map)),"CUDA alpha F16/BF16/Q8_0 source admitted");
        check(map.size()==1 && map[0].binding.role==Role::dense_up,"alpha remains a projection");
    }
    for(const auto* name:{"blk.0.ssm_a","blk.0.ssm_d"}) {
        WeightMap map;auto auxiliary=fixture({{name,"F32",{48},floats(48)}});
        check(bool(bind_qwen(auxiliary,map)) && map.size()==1 && map[0].binding.role==Role::auxiliary,"standalone auxiliary keeps F32 role");
        auxiliary=fixture({{name,"F16",{48},std::vector<Byte>(96)}});
        check(!bind_qwen(auxiliary,map),"standalone auxiliary still refuses F16");
    }
    WeightMap map;auto bias=fixture({{"blk.0.ssm_dt.bias","F16",{48},std::vector<Byte>(96)}});
    check(bool(bind_qwen(bias,map)) && map[0].binding.role==Role::bias,"dt bias is not standalone ssm_d");
}
}
bool wfmt_admission_test() try {
    rank_bounds();qwen_cuda_alpha_sources();
    check(wfmt_reference_admission_test(),"host reference admission negative coverage");
    std::array<Byte,128> bytes{};auto source=memory_source(bytes,"admission");TensorIndex index;index.container=Container::gguf;index.owner=source;
    RawTensor t{};t.id=tensor_id(*source,"x.weight");t.name="x.weight";t.wire_type=1;t.shape={{{2,32,0,0}},2};t.payload={source,{0,128}};index.tensors.push_back(t);
    TensorBinding binding{};binding.tensor=t.id;binding.role=Role::dense_up;binding.logical_shape=t.shape;binding.expert_axis=UINT32_MAX;binding.admitted_formats={{"plain.f16",1}};
    WeightMap map;check(bool(admit(builtin_registry(),{&index,{&binding,1},{}},map)),"explicit plain admission");check(map.size()==1 && map[0].weight.storage==Id{"plain.f16",1},"source storage retained");
    // Reject ranks before comparing dimensions or constructing scalar strides.
    for (const std::uint8_t rank : {std::uint8_t{5}, std::uint8_t{255}}) {
        auto invalid=map[0]; invalid.binding.logical_shape.rank=rank;
        auto status=validate_binding(builtin_registry(),invalid);
        check(status.code==Error::unsupported_shape && status.detail.find("rank exceeds four")!=std::string::npos,"binding rank rejected before comparison");
        invalid.weight.shape.rank=rank;
        check(validate_binding(builtin_registry(),invalid).code==Error::unsupported_shape,"equal oversized ranks rejected");
        invalid.binding.logical_shape.rank=t.shape.rank;
        check(validate_binding(builtin_registry(),invalid).code==Error::unsupported_shape,"weight rank rejected");
        index.tensors[0].shape.rank=rank; binding.logical_shape.rank=rank;
        check(admit(builtin_registry(),{&index,{&binding,1},{}},map).code==Error::unsupported_shape,"plain raw rank rejected before strides");
        index.tensors[0].wire_type=UINT32_MAX;
        check(admit(builtin_registry(),{&index,{&binding,1},{}},map).code==Error::unsupported_shape,"raw rank rejected before format normalization");
        index.tensors[0]=t; binding.logical_shape=t.shape;
    }
    const auto original=describe_admission(map);binding.admitted_formats={{"plain.f32",1}};check(!admit(builtin_registry(),{&index,{&binding,1},{}},map),"format allowlist enforced");check(describe_admission(map)==original,"failed admission atomic");
    binding.admitted_formats.clear();binding.admitted_categories=1;check(bool(admit(builtin_registry(),{&index,{&binding,1},{}},map)),"explicit plain category opt-in");binding.admitted_categories=0;
    binding.admitted_formats={{"plain.f16",1}};binding.logical_shape.dims[0]=3;check(!admit(builtin_registry(),{&index,{&binding,1},{}},map),"shape mismatch rejected");binding.logical_shape=t.shape;
    binding.expert_axis=2;check(!admit(builtin_registry(),{&index,{&binding,1},{}},map),"expert axis rejected");binding.expert_axis=UINT32_MAX;
    index.tensors.push_back(t);check(!admit(builtin_registry(),{&index,{&binding,1},{}},map),"duplicate shard/tensor rejected");index.tensors.pop_back();
    auto shard_duplicate=t;shard_duplicate.id+=1;index.tensors.push_back(shard_duplicate);check(!admit(builtin_registry(),{&index,{&binding,1},{}},map),"duplicate names across distinct source shards rejected");index.tensors.pop_back();
    // Reader-recognised Q4_1 is not silently executable.
    index.tensors[0].wire_type=3;index.tensors[0].payload.range.bytes=40;binding.admitted_formats={{"gguf.q4_1",1}};
    auto rejected=admit(builtin_registry(),{&index,{&binding,1},{}},map);check(!rejected && rejected.code==Error::unsupported_arithmetic,"parse-only never executable");
    index.tensors[0]=t;binding.admitted_formats={{"plain.f16",1}};
    WeightMap family;check(bool(bind_qwen(index,family)),"Qwen data schema");check(bool(bind_ds41(index,family)),"DS41 data schema");check(bool(bind_minimax(index,family)),"MiniMax data schema");check(bool(bind_glm(index,family)),"GLM data schema");
    check(bool(bind_h3(index,family)),"H3 data schema");check(bool(bind_vision(index,family)),"vision data schema");
    index.tensors[0].wire_type=0;index.tensors[0].shape={{{2,16,0,0}},2};index.tensors[0].payload.range.bytes=128;check(bool(bind_gemma4(index,family)),"Gemma data schema");
    check(family[0].binding.admitted_categories==0,"legacy binder uses exact allowlist");
    const auto gemma_text=fixture({{"model.language_model.layer.weight","F32",{1},floats(1)},
        {"lm_head.weight","F32",{1},floats(1)},{"model.vision.layer.weight","F32",{1},floats(1)}});
    WeightMap gemma_filtered;
    check(bool(bind_gemma4(gemma_text,gemma_filtered)) && gemma_filtered.size()==2,"Gemma adapter selects text namespace and head only");
    WeightMap transformed;
    auto table=fixture({{"adaln_t_table","F32",{2,4},floats(8)}});check(bool(bind_h3(table,transformed)),"H3 existing F32 row lookup");
    check(transformed.size()==1 && transformed[0].binding.role==Role::lookup_row,"H3 row lookup role retained");
    const auto config=text_bytes(R"({"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256})");
    auto channel=fixture({{"layer.weight","I8",{1,256},std::vector<Byte>(256)},
        {"layer.weight_scale","F32",{1,1},floats(1)},
        {"layer.comfy_quant","U8",{config.size()},config}});
    check(bool(bind_h3(channel,transformed)),"H3 authoritative int8 schema");
    check(transformed.size()==1 && transformed[0].weight.storage==Id{"linear.i8-channel-convrot256",1} && transformed[0].weight.parameters.transform=="convrot256","H3 semantic identity");
    auto counted=std::make_shared<CountingSource>(transformed[0].weight.payload.source);
    transformed[0].weight.payload.source=counted;for(auto& side:transformed[0].weight.sidecars)side.data.source=counted;
    check(bool(validate_binding(builtin_registry(),transformed[0])) && counted->bytes_read==0,"resolver metadata check performs no numerical source reads");
    check(bool(validate_map(builtin_registry(),transformed)) && counted->bytes_read>0,"one-time admission still checks scale values");
    const auto packed_config=text_bytes(R"({"format":"asym_w4a8_int8","group_size":16})");
    auto packed=fixture({{"layer.weight","I8",{1,128},std::vector<Byte>(128)},
        {"layer.weight_s_rel","F32",{1,16},floats(16)}, {"layer.weight_s_channel","F32",{1},floats(1)},
        {"layer.weight_codebook","BF16",{16},std::vector<Byte>(32)},
        {"layer.comfy_quant","U8",{packed_config.size()},packed_config}});
    check(bool(bind_h3(packed,transformed)),"H3 packed schema and canonical codebook conversion");
    bool original_codebook=false;for(const auto& side:transformed[0].weight.sidecars)original_codebook|=side.name=="source_codebook";
    check(original_codebook,"original codebook provenance retained");
    const std::vector<Byte> marker={Byte{0x2d},Byte{0xd1},Byte{0xdc},Byte{0x83}};
    auto trellis=fixture({{"layer.up.trellis","I16",{8,8,48},std::vector<Byte>(6144)},
        {"layer.up.suh","F16",{128},std::vector<Byte>(256)}, {"layer.up.svh","F16",{128},std::vector<Byte>(256)},
        {"layer.up.mul1","I32",{1},marker}});
    check(bool(bind_minimax(trellis,transformed)),"MiniMax variable-bit transformed schema");
    check(transformed[0].weight.parameters.bits==3 && transformed[0].weight.parameters.transform=="hadamard128","trellis transform/bit identity");
    check(bool(bind_glm(trellis,transformed)),"GLM transformed schema");
    trellis.tensors[0].wire_dtype="U16";check(!bind_glm(trellis,transformed),"GLM retains I16-only trellis graph admission");
    check(bool(bind_minimax(trellis,transformed)),"MiniMax retains its existing U16 transport admission");trellis.tensors[0].wire_dtype="I16";
    trellis.tensors.pop_back();check(bool(bind_glm(trellis,transformed)),"legacy GLM fixed mul1 contract permits absent wire marker");
    check(!bind_ds41(trellis,transformed),"DS41 requires actual mul1 marker");
    trellis.tensors.pop_back();check(!bind_glm(trellis,transformed),"missing transform sidecar rejected");
    auto nan_codes=std::vector<Byte>(1024);nan_codes[0]=Byte{127};
    auto malformed_fp8=fixture({{"layer.weight","F8_E4M3",{32,32},nan_codes},
        {"layer.scale","F8_E8M0",{1,1},{Byte(strix::ds41::fp8_min_exponent)}}});
    auto bad_fp8=bind_ds41(malformed_fp8,transformed);
    check(!bad_fp8 && bad_fp8.code==Error::malformed,"full admission rejects malformed primary FP8 codes");
    // Ownership remains attached to maps after both index and caller source disappear.
    index={};source.reset();std::array<Byte,128> observed{};check(bool(read_slice(family[0].weight.payload,observed)),"admitted map retains source");
    return true;
}catch(const std::exception& e){std::cerr<<"admission: "<<e.what()<<'\n';return false;}
