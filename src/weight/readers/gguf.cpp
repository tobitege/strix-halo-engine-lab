#include "strix/weight/admission.hpp"
#include "strix/gguf_reader.hpp"
#include <algorithm>
#include <exception>
namespace strix::weight {
Status index_gguf(const gguf::FileInfo& parsed,std::shared_ptr<const Source> source,TensorIndex& out) {
    if(!source || source->size()!=parsed.file_size)return {Error::malformed,"GGUF source extent differs from parsed header"};
        TensorIndex result;result.container=Container::gguf;result.schema="gguf@3";result.owner=source;
        auto status=digest_slice({source,{0,parsed.data_offset}},result.header_identity);if(!status)return status;
        for(const auto& t:parsed.tensors) {
            if(t.dimensions.empty() || t.dimensions.size()>4)return {Error::unsupported_shape,"GGUF logical rank outside 1..4"};
            RawTensor raw;raw.id=tensor_id(*source,t.name,result.header_identity);raw.name=t.name;raw.wire_type=t.ggml_type;
            raw.shape.rank=std::uint8_t(t.dimensions.size());raw.extents.assign(t.dimensions.rbegin(),t.dimensions.rend());
            // The public shape is row-major logical [N,K]; GGUF stores contiguous K first.
            std::reverse_copy(t.dimensions.begin(),t.dimensions.end(),raw.shape.dims.begin());
            raw.payload={source,{t.absolute_offset,t.byte_size}};result.tensors.push_back(std::move(raw));
        }
        for(const auto& m:parsed.metadata) if(m.string_value)result.metadata.emplace_back(m.key,*m.string_value);
        out=std::move(result);return {Error::ok,{}};
}

Status index_gguf(std::shared_ptr<const Source> source,TensorIndex& out) {
    if(!source)return {Error::stale_source,"GGUF source is absent"};
    try {
        auto stream=source_stream(source);
        gguf::Limits limits;limits.allow_hyphen_in_keys=true;
        auto parsed=gguf::read(*stream,source->size(),limits);
        return index_gguf(parsed,std::move(source),out);
    }catch(const std::exception& e){return {Error::malformed,e.what()};}
}
}
