#include "strix/weight/admission.hpp"
namespace strix::weight {
Status index_pack(std::shared_ptr<const Source> source,TensorIndex& out) {
    // Existing packed EXL3 transports use safetensors, not an invented binary dtype.
    TensorIndex result;auto status=index_safetensors(std::move(source),result);if(!status)return status;
    result.container=Container::pack;
    bool declared=false;
    for(const auto& [key,value]:result.metadata)if(key=="strix.weight.schema") {
        if(value!="exl3.mul1.tiles16@1" && value!="exl3.gemma.transport@1" && value!="strix.nvfp4.safetensors@1")
            return {Error::unknown_format,"unsupported pack schema "+value};
        result.schema=value;declared=true;
    }
    if(!declared)return {Error::unknown_format,"pack requires a verified strix.weight.schema marker"};
    out=std::move(result);return {Error::ok,{}};
}
}
