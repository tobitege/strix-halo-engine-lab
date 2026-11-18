#include "adapters/schema.hpp"
#include "strix/weight/registry.hpp"
#include "strix/serve/json.hpp"
#include <algorithm>
#include <stdexcept>
namespace strix::weight::detail {
namespace {
const RawTensor* named(const TensorIndex& index,const std::string& name) {for(const auto& t:index.tensors)if(t.name==name)return &t;return nullptr;}
Status channel_parameters(const TensorIndex& index,const RawTensor& raw,Parameters& p,Shape& logical) {
    const auto prefix=raw.name.substr(0,raw.name.size()-7);
    const RawTensor* config=named(index,prefix+".comfy_quant");
    if(!config) {
        for(const auto& t:index.tensors) {
            if(!t.name.starts_with(prefix+".") || t.wire_dtype!="U8")continue;
            const auto suffix=t.name.substr(prefix.size()+1);
            if(suffix.find('.')!=std::string::npos || suffix=="weight" || suffix=="weight_scale" || suffix=="bias")continue;
            if(config)return {Error::malformed,"multiple channel quant configs"};config=&t;
        }
    }
    if(!config || config->wire_dtype!="U8" || config->payload.range.bytes>65536)return {Error::malformed,"channel quant needs bounded U8 config"};
    std::string bytes(std::size_t(config->payload.range.bytes),'\0');auto status=read_slice(config->payload,{reinterpret_cast<Byte*>(bytes.data()),bytes.size()});if(!status)return status;
    try {
        const auto json=serve::parse_json_unique(bytes);const auto* format=json.find("format");
        if(!format)return {Error::malformed,"channel quant config format absent"};
        const auto setting=[&](const char* key,std::int64_t fallback){auto v=json.find(key);if(!v)if(auto params=json.find("params"))v=params->find(key);return v?v->as_int():fallback;};
        if(format->as_string()=="int8_tensorwise") {
            const auto* rot=json.find("convrot");if(!rot || !rot->as_bool() || setting("convrot_groupsize",0)!=256)return {Error::malformed,"channel quant requires convrot256"};
            p.transform="linear.i8-channel-convrot256@1";p.group_size=256;
        }else if(format->as_string()=="asym_w4a8_int8") {
            const auto group=setting("group_size",16);if(group<4 || group>UINT32_MAX || setting("convrot_groupsize",256)!=256)return {Error::malformed,"unsupported packed grouping"};
            p.transform="linear.w4a8-grouped@1";p.bits=4;p.group_size=std::uint32_t(group);
            if(logical.rank!=2 || logical.dims[1]>UINT64_MAX/2)return {Error::unsupported_shape,"invalid packed channel shape"};logical.dims[1]*=2;
            if(named(index,prefix+".weight_correction") || named(index,prefix+".bias"))return {Error::unsupported_arithmetic,"packed correction/bias requires separate arithmetic"};
        }else return {Error::unknown_format,"unsupported channel quant format"};
        p.semantic_metadata=digest_bytes({reinterpret_cast<const Byte*>(bytes.data()),bytes.size()});
        return {Error::ok,{}};
    }catch(const std::exception& e){return {Error::malformed,e.what()};}
}
}
Status bind_model(const TensorIndex& index,const FamilySchema& schema,WeightMap& out) {
    std::vector<TensorBinding> bindings;std::vector<Parameters> parameters;
    const bool nvfp4=index.container==Container::safetensors && index.schema=="strix.nvfp4.safetensors@1";
    for(const auto& raw:index.tensors) {
        if(schema.text_only && index.container!=Container::gguf && !raw.name.starts_with(schema.text_prefix) && raw.name!=schema.text_head)continue;
        // A tensor above the WeightView rank cannot be a view; it stays in the index (byte range) and
        // only an engine that reads it uses it, so it never fails a family's text-weight admission.
        if(raw.opaque())continue;
        const bool trellis=raw.name.ends_with(".trellis");
        const bool standalone=std::any_of(schema.roles.begin(),schema.roles.end(),[&](const auto& p){return p.standalone && p.matches(raw.name);});
        const bool fp4_codes=nvfp4 && raw.name.ends_with(".codes");
        if(nvfp4 && (raw.name.ends_with(".block_scales") || raw.name.ends_with(".tensor_scale"))) {
            if(!named(index,raw.name.substr(0,raw.name.size()-13)+".codes"))
                return {Error::malformed,"orphan canonical NVFP4 scale sidecar"};
            continue;
        }
        const bool weight=raw.name.ends_with(".weight") || raw.name.ends_with("_weight") || standalone || fp4_codes;
        const bool bias=raw.name.ends_with(".bias");
        // Sidecars are attached by schema to their weight, not advertised as independent matrices.
        if(!weight && !trellis && !bias)continue;
        TensorBinding b{};b.tensor=raw.id;b.logical_shape=raw.shape;b.expert_axis=UINT32_MAX;b.role=Role::dense_up;
        for(const auto& pattern:schema.roles)if(pattern.matches(raw.name)){b.role=pattern.role;break;}
        const bool expert=raw.name.find("experts")!=std::string::npos || raw.name.find("_exps")!=std::string::npos;
        if(expert){if(raw.shape.rank>2)b.expert_axis=0;if(b.role==Role::dense_gate)b.role=Role::expert_gate;if(b.role==Role::dense_up)b.role=Role::expert_up;if(b.role==Role::dense_down)b.role=Role::expert_down;}
        Parameters p{};
        if(fp4_codes) {
            if(named(index,raw.name.substr(0,raw.name.size()-6)))
                return {Error::malformed,"duplicate plain and native FP4 logical tensor"};
            if(expert || b.role==Role::embedding || b.role==Role::lookup_row || b.role==Role::router ||
               b.role==Role::norm || b.role==Role::bias || b.role==Role::auxiliary)
                return {Error::unsupported_role,"native NVFP4 schema requires a dense matrix role"};
            if(raw.shape.rank!=2 || raw.shape.dims[1]>UINT64_MAX/2)return {Error::unsupported_shape,"invalid NVFP4 matrix shape"};
            b.logical_shape.dims[1]*=2;p.transform="strix.nvfp4.safetensors@1";
        }else if(trellis) {
            if(raw.wire_dtype=="U16" && !schema.unsigned_trellis)return {Error::unknown_format,"legacy family graph requires signed I16 trellis transport"};
            if(!schema.transformed)return {Error::unknown_format,"family does not admit tile trellis transport"};
            if(raw.shape.rank!=3 || raw.shape.dims[0]>UINT64_MAX/16 || raw.shape.dims[1]>UINT64_MAX/16)return {Error::unsupported_shape,"invalid trellis tile shape"};
            b.logical_shape={{{raw.shape.dims[1]*16,raw.shape.dims[0]*16,0,0}},2};p.transform="exl3.mul1.tiles16@1";
            if(schema.fixed_mul1)p.codebook_encoding="legacy-fixed-mul1@1";
            b.expert_axis=UINT32_MAX;
        }else if(schema.native_fp8 && raw.wire_dtype=="F8_E4M3")p.transform="block.e4m3-e8m0.32x32@1";
        else if(schema.channel_quant && raw.wire_dtype=="I8") {auto status=channel_parameters(index,raw,p,b.logical_shape);if(!status)return status;}
        const auto admitted=(b.role==Role::lookup_row || b.role==Role::auxiliary) && !schema.lookup.empty()?schema.lookup:
            b.role==Role::norm || b.role==Role::bias?schema.plain:schema.linear;
        b.admitted_formats.assign(admitted.begin(),admitted.end());b.admitted_categories=0;
        bindings.push_back(std::move(b));parameters.push_back(std::move(p));
    }
    return admit(builtin_registry(),{&index,bindings,parameters},out);
}
}
