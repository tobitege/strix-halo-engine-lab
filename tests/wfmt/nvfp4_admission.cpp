#include "strix/weight/admission.hpp"
#include "strix/weight/registry.hpp"
#include "strix/serve/json.hpp"
#include "strix/cuda/core/nvfp4.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace strix::weight;
namespace {
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
struct Field {std::string name,dtype;std::vector<std::uint64_t> shape;std::vector<Byte> bytes;};
std::vector<Field> fields() {
    std::vector<Byte> codes(96);for(unsigned i=0;i<codes.size();++i)codes[i]=Byte(i%256);
    std::vector<Byte> scales(12,Byte{56});float tensor=1.5f;std::vector<Byte> ts(4);std::memcpy(ts.data(),&tensor,4);
    return {{"model.layers.0.mlp.up_proj.weight.codes","U8",{3,32},codes},
        {"model.layers.0.mlp.up_proj.weight.block_scales","U8",{3,4},scales},
        {"model.layers.0.mlp.up_proj.weight.tensor_scale","F32",{1},ts}};
}
Status fixture(const std::vector<Field>& fields,std::string marker,TensorIndex& index) {
    using strix::serve::Json;auto root=Json::object();
    if(!marker.empty()){auto meta=Json::object();meta.set("strix.weight_schema",marker);root.set("__metadata__",meta);}
    std::uint64_t offset=0;
    for(const auto& f:fields) {
        auto row=Json::object(),shape=Json::array(),range=Json::array();row.set("dtype",f.dtype);
        for(auto n:f.shape)shape.as_array().emplace_back(n);row.set("shape",shape);
        range.as_array().emplace_back(offset);offset+=f.bytes.size();range.as_array().emplace_back(offset);
        row.set("data_offsets",range);root.set(f.name,row);
    }
    const auto header=root.dump();std::vector<Byte> bytes;
    for(unsigned i=0;i<8;++i)bytes.push_back(Byte((header.size()>>(8*i))&255));
    for(char c:header)bytes.push_back(Byte(c));for(const auto& f:fields)bytes.insert(bytes.end(),f.bytes.begin(),f.bytes.end());
    return index_safetensors(memory_source(bytes,"native-fp4-schema"),index);
}
}
bool wfmt_nvfp4_admission_test() try {
    constexpr auto schema="strix.nvfp4.safetensors@1";auto f=fields();TensorIndex index;
    require(bool(fixture(f,schema,index)),"canonical schema parses");require(index.schema==schema,"explicit schema selected");
    WeightMap map;require(bool(bind_qwen(index,map)),"Qwen native FP4 binding");require(map.size()==1,"companions not independent matrices");
    const auto& w=map[0].weight;require(w.storage==Id{"nvidia.nvfp4",1} && w.shape.dims[0]==3 && w.shape.dims[1]==64,"logical FP4 shape");
    require(w.sidecars.size()==2 && !w.has_byte_strides,"canonical descriptor-addressed sidecars");
    std::vector<Byte> raw(96);require(bool(read_slice(w.payload,raw)),"payload read");require(raw==f[0].bytes,"payload bytes retained");
    std::vector<Byte> decoded(192*sizeof(float));DecodeBuffer out{decoded,0};
    require(bool(decode_reference(builtin_registry(),w,{0,192,{},Scalar::f32},out)),"host FP4 oracle decode");
    for(unsigned i=0;i<192;++i) {
        const auto byte=std::to_integer<unsigned>(raw[i/2]);const auto code=(byte>>(4*(i%2)))&15;
        const float expected=(strix::cuda::core::nvfp4::e2m1(std::uint8_t(code))*1.f)*1.5f;
        require(std::memcmp(decoded.data()+i*4,&expected,4)==0,"FP4 oracle exact IEEE bytes");
    }
    require(!bind_ds41(index,map),"unbound family rejects native FP4");
    const auto negative=[&](std::vector<Field> bad,const char* reason) {
        TensorIndex invalid;require(bool(fixture(bad,schema,invalid)),"negative schema structurally parses");
        WeightMap rejected;require(!bind_qwen(invalid,rejected),reason);
    };
    auto bad=f;bad.pop_back();negative(bad,"missing tensor scale rejected");
    bad=f;bad[1].dtype="F8_E4M3FN";negative(bad,"canonical block scale wire dtype is U8");
    bad=f;float zero=0;std::memcpy(bad[2].bytes.data(),&zero,4);negative(bad,"nonpositive tensor scale rejected");
    bad=f;bad[1].bytes[0]=Byte{127};negative(bad,"NaN block scale rejected");
    bad=f;bad[1].bytes[0]=Byte{128};negative(bad,"negative block scale rejected");
    bad=f;bad[0].shape={3,16};bad[0].bytes.resize(48);bad[1].shape={3,2};bad[1].bytes.resize(6);negative(bad,"K alignment rejected");
    bad=f;bad.erase(bad.begin());negative(bad,"orphan scale rejected");
    bad=f;bad.push_back({"model.layers.0.mlp.up_proj.weight","F32",{3,64},std::vector<Byte>(3*64*4)});
    negative(bad,"duplicate logical weight rejected");
    bad=f;for(auto& field:bad)field.name.replace(15,1,"experts");negative(bad,"expert FP4 schema rejected");
    require(!fixture(f,"strix.nvfp4.safetensors@2",index),"unknown schema revision rejected");
    require(bool(fixture(f,"",index)),"unmarked U8 container parses");
    require(bool(bind_qwen(index,map)) && map.empty(),"unmarked suffixes do not infer FP4 semantics");
    return true;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return false;}
#ifdef STRIX_WFMT_NVFP4_STANDALONE
int main(){return wfmt_nvfp4_admission_test()?0:1;}
#endif
