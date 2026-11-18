#include "strix/weight/admission.hpp"
#include "strix/serve/json.hpp"
#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
namespace strix::weight {
namespace {
const serve::Json& require(const serve::Json& root,const char* key) {
    auto p=root.find(key);if(!p)throw std::runtime_error(std::string("safetensors missing ")+key);return *p;
}
std::uint64_t integer(const serve::Json& j) {
    const auto n=j.as_int();if(n<0)throw std::runtime_error("safetensors negative extent");return std::uint64_t(n);
}
}
Status index_safetensors(std::shared_ptr<const Source> source,TensorIndex& out) {
    if(!source)return {Error::stale_source,"safetensors source is absent"};
    try {
        if(source->size()<8)return {Error::malformed,"truncated safetensors prefix"};
        std::array<Byte,8> prefix{};auto status=source->read({0,8},prefix);if(!status)return status;
        std::uint64_t n=0;for(unsigned i=0;i<8;++i)n|=std::uint64_t(std::to_integer<unsigned char>(prefix[i]))<<(i*8);
        if(n<2 || n>64ULL*1024*1024 || n>source->size()-8)return {Error::malformed,"invalid safetensors header size"};
        std::string text(std::size_t(n),'\0');status=source->read({8,n},{reinterpret_cast<Byte*>(text.data()),text.size()});if(!status)return status;
        const auto root=serve::parse_json_unique(text);const auto base=8+n;
        TensorIndex result;result.container=Container::safetensors;result.schema="safetensors@1";result.owner=source;
        result.header_identity=digest_bytes({reinterpret_cast<const Byte*>(text.data()),text.size()});
        const std::map<std::string,std::uint64_t> widths={{"BF16",2},{"F16",2},{"F32",4},{"F64",8},{"I8",1},{"U8",1},{"BOOL",1},{"I16",2},{"U16",2},{"I32",4},{"U32",4},{"I64",8},{"U64",8},{"F8_E4M3",1},{"F8_E4M3FN",1},{"F8_E5M2",1},{"F8_E8M0",1}};
        std::vector<std::pair<std::uint64_t,std::uint64_t>> spans;
        for(const auto& [name,d]:root.as_object()) {
            if(name=="__metadata__") {for(const auto& [key,value]:d.as_object())result.metadata.emplace_back(key,value.as_string());continue;}
            if(d.as_object().size()!=3)return {Error::malformed,"unsupported safetensors fields/strides"};
            RawTensor t;t.name=name;t.id=tensor_id(*source,name,result.header_identity);t.wire_dtype=require(d,"dtype").as_string();
            auto width=widths.find(t.wire_dtype);if(width==widths.end())return {Error::unknown_format,"unrecognised safetensors dtype "+t.wire_dtype};
            const auto& shape=require(d,"shape").as_array();if(shape.size()>255)return {Error::malformed,"safetensors rank out of range"};
            // Rank above the WeightView rank is indexed opaquely (see RawTensor::opaque) instead of failing the whole file.
            const bool viewable=shape.size()<=t.shape.dims.size();t.shape.rank=viewable?std::uint8_t(shape.size()):std::uint8_t(0);std::uint64_t count=1;
            for(std::size_t i=0;i<shape.size();++i){auto dim=integer(shape[i]);if(dim && count>UINT64_MAX/dim)return {Error::malformed,"safetensors shape overflow"};count*=dim;t.extents.push_back(dim);if(viewable)t.shape.dims[i]=dim;}
            const auto& offsets=require(d,"data_offsets").as_array();if(offsets.size()!=2)return {Error::malformed,"safetensors needs two offsets"};
            auto first=integer(offsets[0]),last=integer(offsets[1]);
            if(count>UINT64_MAX/width->second || last<first || last>source->size()-base || last-first!=count*width->second)
                return {Error::malformed,"safetensors extent mismatch "+name};
            t.payload={source,{base+first,last-first}};if(first!=last)spans.emplace_back(first,last);result.tensors.push_back(std::move(t));
        }
        std::sort(spans.begin(),spans.end());std::uint64_t end=0;
        for(auto [first,last]:spans){if(first!=end)return {Error::malformed,"safetensors overlap or unclaimed hole"};end=last;}
        if(end!=source->size()-base)return {Error::malformed,"unclaimed safetensors payload"};
        // Only an explicit versioned marker gives U8 codes/scales FP4 semantics.
        for(const auto& [key,value]:result.metadata)if(key=="strix.weight_schema") {
            if(value!="strix.nvfp4.safetensors@1")return {Error::unknown_format,"unsupported strix weight schema"};
            result.schema=value;
        }
        out=std::move(result);return {Error::ok,{}};
    }catch(const std::exception& e){return {Error::malformed,e.what()};}
}
FormatId safetensors_format(std::string_view dtype) noexcept {
    if(dtype=="F32")return {"plain.f32",1};if(dtype=="F16")return {"plain.f16",1};if(dtype=="BF16")return {"plain.bf16",1};
    if(dtype=="F64")return {"typed.f64",1};if(dtype=="I8")return {"typed.i8",1};if(dtype=="U8")return {"typed.u8",1};
    if(dtype=="I16")return {"typed.i16",1};if(dtype=="U16")return {"typed.u16",1};if(dtype=="I32")return {"typed.i32",1};
    if(dtype=="U32")return {"typed.u32",1};if(dtype=="I64")return {"typed.i64",1};if(dtype=="U64")return {"typed.u64",1};
    if(dtype=="BOOL")return {"typed.bool8",1};if(dtype=="F8_E4M3" || dtype=="F8_E4M3FN")return {"typed.e4m3fn",1};
    if(dtype=="F8_E8M0")return {"typed.e8m0",1};
    if(dtype=="F8_E5M2")return {"typed.e5m2",1};return {{},0};
}
}
