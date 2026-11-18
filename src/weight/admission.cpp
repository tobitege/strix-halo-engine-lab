#include "strix/weight/admission.hpp"
#include "strix/weight/registry.hpp"
#include "core/trellis_format.hpp"
#include "strix/ggml_dequant.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include <set>
#include <sstream>
namespace strix::weight {
namespace {
bool same_shape(const Shape& a,const Shape& b) {if(a.rank!=b.rank)return false;for(unsigned i=0;i<a.rank;++i)if(a.dims[i]!=b.dims[i])return false;return true;}
Status check(const Registry& registry,const WeightMapEntry& e,bool scan_values=true) {
    const auto& b=e.binding;const auto& w=e.weight;
    if(b.logical_shape.rank>b.logical_shape.dims.size())return {Error::unsupported_shape,"binding rank exceeds four"};
    if(w.shape.rank>w.shape.dims.size())return {Error::unsupported_shape,"weight rank exceeds four"};
    if(b.tensor!=w.tensor)return {Error::malformed,"binding tensor identity mismatch"};
    if(!same_shape(b.logical_shape,w.shape))return {Error::unsupported_shape,"binding logical shape mismatch"};
    for(unsigned i=0;i<w.shape.rank;++i)if(!w.shape.dims[i])return {Error::unsupported_shape,"empty executable weight dimension"};
    if(b.expert_axis!=UINT32_MAX && b.expert_axis>=w.shape.rank)return {Error::unsupported_shape,"expert axis outside logical shape"};
    const auto* format=find_format(registry,w.storage);if(!format)return {Error::unknown_format,"unregistered weight format "+serialize_id(w.storage)};
    const auto role=unsigned(b.role);if(role>=64 || !(format->applicable_roles&(std::uint64_t{1}<<role)))return {Error::unsupported_role,"format cannot serve requested tensor role"};
    const bool exact=std::find(b.admitted_formats.begin(),b.admitted_formats.end(),w.storage)!=b.admitted_formats.end();
    if(!exact && !(b.admitted_categories&format->category))return {Error::unknown_format,"format outside tensor allowlist"};
    const bool typed=b.role==Role::auxiliary || b.role==Role::bias || b.role==Role::norm;
    if(!format->semantic_decode_supported && !typed)return {Error::unsupported_arithmetic,"container dtype has no semantic weight decoder"};
    if(!w.payload.source)return {Error::stale_source,"weight source owner absent"};
    const auto valid_slice=[](const Slice& s){return s.source && s.range.offset<=s.source->size() && s.range.bytes<=s.source->size()-s.range.offset;};
    if(!valid_slice(w.payload))return {Error::malformed,"weight payload outside source"};
    std::set<std::string> names;
    for(const auto& sidecar:w.sidecars)if(!valid_slice(sidecar.data) || !names.insert(sidecar.name).second)return {Error::malformed,"invalid or duplicate semantic sidecar"};
    if(scan_values && format->validate){auto status=format->validate(w);if(!status)return status;}
    else {auto status=validate_metadata(w);if(!status)return status;}
    return {Error::ok,{}};
}
const RawTensor* named(const TensorIndex& index,std::string_view name) {for(const auto& t:index.tensors)if(t.name==name)return &t;return nullptr;}
Status normalise(const Registry& registry,const TensorIndex& index,const RawTensor& raw,const TensorBinding& binding,const Parameters& params,WeightView& out) {
    if(raw.opaque())return {Error::unsupported_shape,"tensor rank exceeds WeightView rank: "+raw.name};
    if(raw.shape.rank>raw.shape.dims.size())return {Error::unsupported_shape,"storage rank exceeds four"};
    if(binding.logical_shape.rank>binding.logical_shape.dims.size())return {Error::unsupported_shape,"binding rank exceeds four"};
    WeightView w{};w.tensor=raw.id;w.shape=raw.shape;w.payload=raw.payload;w.parameters=params;
    w.storage=index.container==Container::gguf?gguf_format(raw.wire_type):safetensors_format(raw.wire_dtype);
    // A caller must explicitly select this semantic schema, then all companion dtype/shape rules are verified.
    if(params.transform=="strix.nvfp4.safetensors@1") {
        if(index.container!=Container::safetensors || index.schema!=params.transform || !raw.name.ends_with(".codes") ||
           raw.wire_dtype!="U8" || raw.shape.rank!=2 || !raw.shape.dims[0] || !raw.shape.dims[1] ||
           raw.shape.dims[1]>UINT64_MAX/2 || raw.shape.dims[1]%32)
            return {Error::malformed,"invalid canonical NVFP4 code tensor"};
        const auto base=raw.name.substr(0,raw.name.size()-6);
        const auto* scales=named(index,base+".block_scales");const auto* tensor=named(index,base+".tensor_scale");
        if(!scales || scales->wire_dtype!="U8" || scales->shape.rank!=2 || scales->shape.dims[0]!=raw.shape.dims[0] ||
           scales->shape.dims[1]!=raw.shape.dims[1]/8 || !tensor || tensor->wire_dtype!="F32" ||
           tensor->shape.rank!=1 || tensor->shape.dims[0]!=1)
            return {Error::malformed,"canonical NVFP4 requires U8 block scales and F32[1] tensor scale"};
        w.storage={"nvidia.nvfp4",1};w.shape={{{raw.shape.dims[0],raw.shape.dims[1]*2,0,0}},2};
        w.sidecars.push_back({"block_scales",Scalar::e4m3fn,scales->shape,scales->payload});
        w.sidecars.push_back({"tensor_scale",Scalar::f32,tensor->shape,tensor->payload});
        w.parameters.bits=4;w.parameters.group_size=16;
        w.parameters.scale_encoding="e4m3fn";w.parameters.transform.clear();
        // Packed nibbles use descriptor addressing, never scalar byte strides.
        w.has_byte_strides=false;
    }else if(params.transform=="block.e4m3-e8m0.32x32@1") {
        if(raw.wire_dtype!="F8_E4M3" || raw.shape.rank!=2 || raw.shape.dims[0]%32 || raw.shape.dims[1]%32 || !raw.name.ends_with(".weight"))return {Error::malformed,"invalid native FP8 weight schema"};
        const auto* scale=named(index,raw.name.substr(0,raw.name.size()-7)+".scale");
        if(!scale || scale->wire_dtype!="F8_E8M0" || scale->shape.rank!=2 || scale->shape.dims[0]!=raw.shape.dims[0]/32 || scale->shape.dims[1]!=raw.shape.dims[1]/32)return {Error::malformed,"native FP8 requires exact E8M0 block scale grid"};
        w.storage={"block.e4m3-e8m0.32x32",1};w.sidecars.push_back({"block_scales",Scalar::e8m0,scale->shape,scale->payload});
    }else if(params.transform=="exl3.mul1.tiles16@1") {
        if(!raw.name.ends_with(".trellis") || (raw.wire_dtype!="I16" && raw.wire_dtype!="U16") || raw.shape.rank!=3 || !raw.shape.dims[2] || raw.shape.dims[2]%16)return {Error::malformed,"invalid EXL3 trellis tensor schema"};
        const auto base=raw.name.substr(0,raw.name.size()-8);const auto* suh=named(index,base+".suh");const auto* svh=named(index,base+".svh");
        if(!suh || !svh || suh->wire_dtype!="F16" || svh->wire_dtype!="F16" || suh->shape.rank!=1 || svh->shape.rank!=1 || suh->shape.dims[0]!=raw.shape.dims[0]*16 || svh->shape.dims[0]!=raw.shape.dims[1]*16)return {Error::malformed,"EXL3 requires exact F16 input/output transforms"};
        w.storage={"exl3.mul1.tiles16",1};w.shape=binding.logical_shape;
        w.parameters.bits=std::uint32_t(raw.shape.dims[2]/16);
        w.parameters.transform="hadamard128";
        w.sidecars.push_back({"input_scale",Scalar::f16,suh->shape,suh->payload});w.sidecars.push_back({"output_scale",Scalar::f16,svh->shape,svh->payload});
        if(const auto* mul=named(index,base+".mul1")) {
            if(mul->wire_dtype!="I32" || mul->payload.range.bytes!=4)return {Error::malformed,"invalid EXL3 mul1 marker"};
            std::array<Byte,4> bytes{};auto status=read_slice(mul->payload,bytes);if(!status)return status;
            if(bytes!=std::array<Byte,4>{Byte{0x2d},Byte{0xd1},Byte{0xdc},Byte{0x83}})return {Error::malformed,"EXL3 marker is not mul1"};
            w.sidecars.push_back({"codebook_marker",Scalar::i32,{{{1,0,0,0}},1},mul->payload});
        }else if(params.codebook_encoding=="legacy-fixed-mul1@1") {
            // Legacy graph contract fixes the codebook even where its container omitted a marker.
            constexpr auto marker=core::trellis::Mul1Constants::kMultiplier;
            const std::array<Byte,4> bytes={Byte(marker&255),Byte((marker>>8)&255),Byte((marker>>16)&255),Byte(marker>>24)};
            auto owner=memory_source(bytes,"legacy-fixed-mul1");
            w.sidecars.push_back({"codebook_marker",Scalar::i32,{{{1,0,0,0}},1},{owner,{0,4}}});
        }else return {Error::malformed,"EXL3 requires verified codebook marker or fixed legacy graph contract"};
    }else if(params.transform=="linear.i8-channel-convrot256@1" || params.transform=="linear.w4a8-grouped@1") {
        if(raw.wire_dtype!="I8" || raw.shape.rank!=2 || !raw.name.ends_with(".weight"))return {Error::malformed,"invalid channel-quant weight schema"};
        const auto prefix=raw.name.substr(0,raw.name.size()-7);
        w.shape=binding.logical_shape;w.parameters.transform="convrot256";
        const auto attach=[&](const std::string& suffix,const char* semantic,Scalar scalar)->Status {
            const auto* side=named(index,prefix+suffix);if(!side)return {Error::malformed,"missing channel quant sidecar "+suffix};
            w.sidecars.push_back({semantic,scalar,side->shape,side->payload});return {Error::ok,{}};
        };
        if(params.transform=="linear.i8-channel-convrot256@1") {
            const auto* scale=named(index,prefix+".weight_scale");
            if(!scale || scale->wire_dtype!="F32" || scale->shape.rank!=2 || scale->shape.dims[0]!=raw.shape.dims[0] || scale->shape.dims[1]!=1)return {Error::malformed,"channel quant requires F32[N,1] scales"};
            w.storage={"linear.i8-channel-convrot256",1};auto status=attach(".weight_scale","channel_scales",Scalar::f32);if(!status)return status;
            w.sidecars.back().shape={{{raw.shape.dims[0],0,0,0}},1};
        }else {
            const auto* group=named(index,prefix+".weight_s_rel");const auto* channel=named(index,prefix+".weight_s_channel");
            const auto n=w.shape.dims[0],k=w.shape.dims[1];
            if(!params.group_size || k%params.group_size || !group || !channel || group->shape.rank!=2 || group->shape.dims[0]!=n || group->shape.dims[1]!=k/params.group_size || channel->wire_dtype!="F32" || channel->shape.rank!=1 || channel->shape.dims[0]!=n)return {Error::malformed,"invalid grouped weight scale geometry"};
            const bool f32=group->wire_dtype=="F32";
            if(!f32 && group->wire_dtype!="F8_E4M3" && group->wire_dtype!="F8_E4M3FN" && group->wire_dtype!="U8")return {Error::malformed,"group scales must be E4M3FN or F32"};
            w.storage={"linear.w4a8-grouped",1};w.parameters.scale_encoding=f32?"f32":"e4m3fn";
            auto status=attach(".weight_s_rel","group_scales",f32?Scalar::f32:Scalar::e4m3fn);if(!status)return status;
            status=attach(".weight_s_channel","channel_scales",Scalar::f32);if(!status)return status;
            // This model schema requires strictly positive channel scales, as the legacy loader does.
            std::array<Byte,4096> chunk{};
            for(std::uint64_t at=0;at<channel->payload.range.bytes;) {
                const auto bytes=std::size_t(std::min<std::uint64_t>(chunk.size(),channel->payload.range.bytes-at));
                auto part=channel->payload;part.range.offset+=at;part.range.bytes=bytes;
                status=read_slice(part,{chunk.data(),bytes});if(!status)return status;
                for(std::size_t i=0;i<bytes;i+=4){float scale;std::memcpy(&scale,chunk.data()+i,4);if(!std::isfinite(scale) || scale<=0)return {Error::malformed,"packed channel scale must be finite positive"};}
                at+=bytes;
            }
            if(const auto* codebook=named(index,prefix+".weight_codebook")) {
                Scalar scalar;
                if(codebook->wire_dtype=="F32")scalar=Scalar::f32;else if(codebook->wire_dtype=="F16")scalar=Scalar::f16;else if(codebook->wire_dtype=="BF16")scalar=Scalar::bf16;else return {Error::malformed,"unsupported channel codebook dtype"};
                if(codebook->shape.rank!=1 || codebook->shape.dims[0]!=16)return {Error::malformed,"channel codebook requires sixteen entries"};
                if(scalar==Scalar::f32)w.sidecars.push_back({"codebook",scalar,codebook->shape,codebook->payload});
                else {
                    std::vector<Byte> stored(std::size_t(codebook->payload.range.bytes));auto status=read_slice(codebook->payload,stored);if(!status)return status;
                    std::array<float,16> values{};ggml_dequant::dequantize_row(scalar==Scalar::f16?1U:30U,stored,values);
                    const auto bytes=std::as_bytes(std::span<const float>(values));auto owner=memory_source(bytes,"canonical-channel-codebook");
                    w.sidecars.push_back({"codebook",Scalar::f32,codebook->shape,{owner,{0,bytes.size()}}});
                    w.sidecars.push_back({"source_codebook",scalar,codebook->shape,codebook->payload});
                }
            }
        }
    }else if(!params.transform.empty())return {Error::unknown_format,"unsupported semantic transform "+params.transform};
    const auto* desc=find_format(registry,w.storage);if(!desc)return {Error::unknown_format,"unregistered storage dtype"};
    if(desc->geometry){auto status=desc->geometry(w.parameters,w.geometry);if(!status)return status;}
    // Descriptor-addressed packed/transform layouts never pretend to have scalar strides.
    w.has_byte_strides=w.geometry.block_elements==1 && params.transform.empty();
    if(w.has_byte_strides && w.shape.rank){std::uint64_t stride=w.geometry.block_bytes;for(unsigned i=w.shape.rank;i-->0;){w.byte_strides[i]=stride;if(w.shape.dims[i] && stride>UINT64_MAX/w.shape.dims[i])return {Error::malformed,"stride overflow"};stride*=w.shape.dims[i];}}
    out=std::move(w);return {Error::ok,{}};
}
}
Status namespace_index(TensorIndex& index,std::string shard) {
    if(!index.owner)return {Error::stale_source,"cannot namespace index without source owner"};
    auto owner=namespace_source(index.owner,std::move(shard));
    for(auto& t:index.tensors){t.payload.source=owner;t.id=tensor_id(*owner,t.name,index.header_identity);}
    for(auto& blob:index.metadata_blobs)blob.data.source=owner;
    index.owner=std::move(owner);return {Error::ok,{}};
}

Status validate_binding(const Registry& registry,const WeightMapEntry& entry) try {return check(registry,entry,false);}
catch(const std::bad_alloc&){return {Error::memory_budget,"binding allocation failed"};}
catch(const std::exception& e){return {Error::malformed,e.what()};}

Status validate_map(const Registry& registry,const WeightMap& map) try {
    std::set<TensorId> seen;
    for(const auto& e:map){if(!seen.insert(e.binding.tensor).second)return {Error::malformed,"duplicate tensor identity in map"};auto status=check(registry,e);if(!status)return status;}
    return {Error::ok,{}};
}
catch(const std::bad_alloc&){return {Error::memory_budget,"admission allocation failed"};}
catch(const std::exception& e){return {Error::malformed,e.what()};}
Status admit(const Registry& registry,const AdmissionRequest& request,WeightMap& out) try {
    if(!request.index)return {Error::malformed,"admission index absent"};
    if(!request.tensor_parameters.empty() && request.tensor_parameters.size()!=request.bindings.size())return {Error::malformed,"parameter/binding count mismatch"};
    std::set<TensorId> indexed;std::set<std::string> names;
    for(const auto& raw:request.index->tensors) {
        if(!indexed.insert(raw.id).second)return {Error::malformed,"duplicate tensor identity in source index"};
        if(!names.insert(raw.name).second)return {Error::malformed,"duplicate tensor name across source shards"};
    }
    WeightMap map;map.reserve(request.bindings.size());
    for(std::size_t i=0;i<request.bindings.size();++i){const auto& b=request.bindings[i];const RawTensor* raw=nullptr;for(const auto& t:request.index->tensors)if(t.id==b.tensor){raw=&t;break;}if(!raw)return {Error::malformed,"binding refers to absent tensor"};
        WeightView view;auto status=normalise(registry,*request.index,*raw,b,request.tensor_parameters.empty()?Parameters{}:request.tensor_parameters[i],view);if(!status)return status;
        map.push_back({b,std::move(view)});
    }
    auto status=validate_map(registry,map);if(status)out=std::move(map);return status;
}
catch(const std::bad_alloc&){return {Error::memory_budget,"admission allocation failed"};}
catch(const std::exception& e){return {Error::malformed,e.what()};}
std::string describe_admission(const WeightMap& map) {std::ostringstream out;for(const auto& e:map)out<<e.binding.tensor<<' '<<unsigned(e.binding.role)<<' '<<serialize_id(e.weight.storage)<<' '<<e.weight.payload.range.offset<<'+'<<e.weight.payload.range.bytes<<'\n';return out.str();}
}
