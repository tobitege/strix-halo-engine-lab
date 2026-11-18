#include "registry.hpp"
#include "ggml_sources.hpp"
#include "gsq_sources.hpp"
#include "gsq_plan.hpp"
#include "arch/gfx1151_gsq.hpp"
#include "strix/weight/registry.hpp"
#include "strix/weight/host_q8.hpp"
#include <limits>
namespace strix::weight::hip {
namespace {
Status shape(const WeightView& w,const LaunchShape& s,FormatId format,unsigned block,unsigned bytes,
             unsigned offset_alignment,bool planar=false) {
    if(!(w.storage==format)) return {Error::unknown_format,"HIP source storage identity differs"};
    auto status=validate_launch(w,s,block,bytes,offset_alignment,planar);
    if(!status) return status;
    return validate_weight(w);
}
}
Status Q8F16Source::validate(const WeightView& w,const LaunchShape& s) {
    auto status=shape(w,s,{"gguf.q8_0",1},32,34,68);
    if(!status) return status;
    if(s.k%128U || s.activation_stride%32U) return {Error::unsupported_shape,"HIP Q8A8 F16 staging requires K128/A32"};
    return {Error::ok,{}};
}
Status RawF16Source::validate(const WeightView& w,const LaunchShape& s) {
    auto status=shape(w,s,{"plain.f16",1},1,2,2);if(!status) return status;
    if(s.k%128U) return {Error::unsupported_shape,"HIP raw F16 core requires K128"};
    return {Error::ok,{}};
}
Status validate_q8_rows(const WeightView& w,const LaunchShape& s) {return shape(w,s,{"gguf.q8_0",1},32,34,34);}
Status validate_q6_rows(const WeightView& w,const LaunchShape& s) {return shape(w,s,{"image.q6p32",1},32,26,26,true);}
Status Q8SignedSource::validate(const WeightView& w,const LaunchShape& s) {
    auto status=Q8RowsSource::validate(w,s);if(!status) return status;
    if(s.k%64U || s.row_stride_bytes%4U || s.k_offset_bytes%68U) return {Error::unsupported_shape,"HIP signed-A8 source requires even whole-dword block pairs"};
    return {Error::ok,{}};
}
Status Q6SignedSource::validate(const WeightView& w,const LaunchShape& s) {
    auto status=Q6RowsSource::validate(w,s);if(!status) return status;
    if(s.k%64U || s.row_stride_bytes%4U) return {Error::unsupported_shape,"HIP signed-A8 planar source requires even whole-dword block pairs"};
    return {Error::ok,{}};
}
// GSQ (IQ* / Q2_0) sources: whole 32-weight groups addressed by row; the geometry is the shared semantic one.
template<std::uint32_t T,class Arch>
Status OperandSource<GsqTag<T>,SignedA8Tag,Arch,GsqA8Arithmetic>::validate(const WeightView& w,const LaunchShape& s) {
    return shape(w,s,::strix::weight::gguf_format(T),block_elements,block_bytes,block_bytes);
}
template<std::uint32_t T,class Arch>
Status OperandSource<GsqTag<T>,F16TileTag,Arch,GsqF16Arithmetic>::validate(const WeightView& w,const LaunchShape& s) {
    constexpr auto be=::strix::gsq_a8::block_elements(T),bb=::strix::gsq_a8::block_bytes(T);
    auto status=shape(w,s,::strix::weight::gguf_format(T),be,bb,bb);
    if(!status) return status;
    return gsq_f16_launch(w,s,T);
}
#define STRIX_GSQ_REGISTER(NAME,TYPE,PAIR)     {{NAME,1},"signed-a8","gsq-a8-groups",32,2,0,0,false,false,GsqSignedSource<TYPE>::validate},     {{NAME,1},"f16-tile","gsq-f16-groups",32,4,64,PAIR,false,false,GsqF16Source<TYPE,GsqGfx1151>::validate}
std::span<const SourceRegistration> sources() noexcept {
    static const SourceRegistration table[]={
        {{"plain.f16",1},"f16-tile","existing-f16-core",1,2,64,0,false,false,RawF16Source::validate},
        {{"gguf.q8_0",1},"f16-tile","dense-q8a8-f16-v1",32,2,64,0,false,false,Q8F16Source::validate},
        {{"gguf.q8_0",1},"native-scalar","existing-row-site",32,2,0,0,false,false,Q8RowsSource::validate},
        {{"image.q6p32",1},"native-scalar","existing-row-site",32,2,0,0,false,false,Q6RowsSource::validate},
        {{"gguf.q8_0",1},"signed-a8","existing-block-chain",32,4,0,64,false,false,Q8SignedSource::validate},
        {{"image.q6p32",1},"signed-a8","existing-block-chain",32,4,0,64,false,false,Q6SignedSource::validate},
        STRIX_GSQ_REGISTER("gguf.iq3_s",::strix::gsq_a8::type_iq3_s,0),
        STRIX_GSQ_REGISTER("gguf.iq3_xxs",::strix::gsq_a8::type_iq3_xxs,0),
        STRIX_GSQ_REGISTER("gguf.iq2_s",::strix::gsq_a8::type_iq2_s,0),
        STRIX_GSQ_REGISTER("gguf.iq4_xs",::strix::gsq_a8::type_iq4_xs,0),
        STRIX_GSQ_REGISTER("gguf.iq4_nl",::strix::gsq_a8::type_iq4_nl,64),
        STRIX_GSQ_REGISTER("gguf.q2_0.strix",::strix::gsq_a8::type_q2_0,128),
        {{"exl3.mul1.tiles16",1},"trellis-tile","trellis-mul1-legacy",16,4,0,0,true,true,validate_trellis_mul1},
        {{"exl3.gemma.transport",1},"trellis-tile","g4-exl3-v1",16,2,0,0,true,true,validate_gemma_transport},
        {{"exl3.gemma.transport",1},"trellis-tile","g4-exl3-v2",16,2,0,0,true,true,validate_gemma_transport},
        {{"exl3.gemma.transport",1},"trellis-tile","g4-exl3-v3",16,2,0,0,true,true,validate_gemma_transport},
        {{"exl3.gemma.transport",1},"trellis-tile","g4-exl3-wmma-pf-v1",16,2,0,0,true,true,validate_gemma_transport}
    };
    return table;
}
#undef STRIX_GSQ_REGISTER
const SourceRegistration* find_source(FormatId storage,std::string_view protocol) noexcept {
    return find_unique_source(sources(),storage,protocol);
}
const SourceRegistration* find_source(FormatId storage,std::string_view protocol,std::string_view arithmetic) noexcept {
    for(const auto& s:sources()) if(s.storage==storage && s.protocol==protocol && s.arithmetic==arithmetic) return &s;
    return nullptr;
}
}  // namespace strix::weight::hip
