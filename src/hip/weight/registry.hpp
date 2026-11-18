#pragma once
#include "strix/weight/resolve.hpp"
namespace strix::weight::hip {
struct LaunchShape {
    std::uint64_t m,n,k,row_stride_bytes,activation_stride,output_stride,k_offset_bytes;
};
// Host-testable launch envelope; storage metadata/ownership validation is also
// required by the registry. Q6 planar rows cannot be rebased to a shorter width.
inline Status validate_launch(const WeightView& w,const LaunchShape& s,
                              unsigned block,unsigned bytes,unsigned offset_alignment,bool planar=false) {
    if(!s.m || !s.n || !s.k || s.m>UINT32_MAX || s.n>UINT32_MAX || s.k>UINT32_MAX ||
       s.k%block || s.activation_stride<s.k || s.output_stride<s.n)
        return {Error::unsupported_shape,"HIP source dimensions/activation/output strides"};
    if(w.shape.rank!=2 || w.shape.dims[0]<s.n || w.shape.dims[1]<s.k ||
       w.shape.dims[1]%block || w.shape.dims[1]/block>UINT64_MAX/bytes)
        return {Error::unsupported_shape,"HIP source launch differs from owned tensor"};
    const auto stride=w.shape.dims[1]/block*bytes;
    const auto row=s.k/block*bytes;
    if(s.row_stride_bytes!=stride || (w.has_byte_strides && w.byte_strides[0]!=stride) ||
       s.k_offset_bytes%offset_alignment || s.k_offset_bytes>stride || row>stride-s.k_offset_bytes)
        return {Error::unsupported_shape,"HIP source row stride or K prefix"};
    const auto end=s.k_offset_bytes+row; // bounded by stride above
    if(end>w.payload.range.bytes || (s.n-1U)>(w.payload.range.bytes-end)/stride)
        return {Error::unsupported_shape,"HIP source addressed extent exceeds payload"};
    if(planar && (s.k_offset_bytes || s.k!=w.shape.dims[1] ||
       end>UINT32_MAX || (s.n-1U)>(UINT32_MAX-end)/stride))
        return {Error::unsupported_shape,"HIP planar source requires full rows and a below-4-GiB span"};
    return {Error::ok,{}};
}
// EXL3 / trellis tile rings (exl3.mul1.tiles16 per expert: trellis payload + input_scale / output_scale sidecars; and
// exl3.gemma.transport: suh | svh | trellis in one payload). A ring has no row stride and no K prefix: the launch must span the
// whole tensor, a nominal row_bytes is never a stride, and each resident expert is validated against its own sidecars.
inline Status validate_trellis_launch(const WeightView& w,const LaunchShape& s,bool gemma) {
    if(!(w.storage==FormatId{gemma?"exl3.gemma.transport":"exl3.mul1.tiles16",1}))
        return {Error::unknown_format,"HIP trellis source storage identity differs"};
    const auto bits=w.parameters.bits;
    if(gemma?!(bits==4U||bits==5U||bits==6U||bits==8U):(bits<1U||bits>8U))
        return {Error::unsupported_shape,"HIP trellis source bit width"};
    if(w.shape.rank!=2) return {Error::unsupported_shape,"HIP trellis source needs a rank-2 tensor"};
    const auto n=w.shape.dims[0],k=w.shape.dims[1];
    if(!n||!k||n%128U||k%128U||n>UINT32_MAX||k>UINT32_MAX)
        return {Error::unsupported_shape,"HIP trellis source needs complete H128 transform blocks"};
    if(!s.m||s.m>UINT32_MAX||s.n!=n||s.k!=k||s.activation_stride<k||s.output_stride<n)
        return {Error::unsupported_shape,"HIP trellis launch must span the whole tensor"};
    if(s.row_stride_bytes||s.k_offset_bytes)
        return {Error::unsupported_shape,"HIP trellis tiles have no row stride or K prefix"};
    const std::uint64_t trellis=(n/16U)*(k/16U)*32U*bits;
    if(gemma) {
        for(const auto& side:w.sidecars) if(side.name=="input_scale"||side.name=="output_scale")
            return {Error::unsupported_shape,"Gemma transport carries its scales in the payload"};
        if(w.payload.range.bytes!=2U*(n+k)+trellis) return {Error::unsupported_shape,"Gemma transport payload extent"};
        return {Error::ok,{}};
    }
    if(w.payload.range.bytes!=trellis) return {Error::unsupported_shape,"EXL3 trellis payload extent"};
    unsigned suh=0,svh=0;
    for(const auto& side:w.sidecars) {
        if(side.name=="input_scale") {++suh; if(side.data.range.bytes!=2U*k||side.scalar!=Scalar::f16) return {Error::malformed,"EXL3 suh sidecar extent"};}
        if(side.name=="output_scale") {++svh; if(side.data.range.bytes!=2U*n||side.scalar!=Scalar::f16) return {Error::malformed,"EXL3 svh sidecar extent"};}
    }
    if(suh!=1U||svh!=1U) return {Error::malformed,"EXL3 expert needs its own input_scale and output_scale"};
    return {Error::ok,{}};
}
inline Status validate_trellis_mul1(const WeightView& w,const LaunchShape& s) {return validate_trellis_launch(w,s,false);}
inline Status validate_gemma_transport(const WeightView& w,const LaunchShape& s) {return validate_trellis_launch(w,s,true);}
struct SourceRegistration {
    FormatId storage;
    std::string_view protocol,arithmetic;
    std::uint32_t group_k,alignment,staged_k,paired_k;
    bool input_transform,output_transform;
    Status (*validate)(const WeightView&,const LaunchShape&);
};
std::span<const SourceRegistration> sources() noexcept;
// An omitted arithmetic is safe only when the storage/protocol pair is unique.
inline const SourceRegistration* find_unique_source(std::span<const SourceRegistration> rows,
                                                   FormatId storage,std::string_view protocol) noexcept {
    const SourceRegistration* match=nullptr;
    for(const auto& s:rows) if(s.storage==storage && s.protocol==protocol) {
        if(match) return nullptr;
        match=&s;
    }
    return match;
}
// Admission scaffolding tested by fixtures; production launchers do not use this
// table yet. Callers retain incumbent launch and numeric Site policy.
const SourceRegistration* find_source(FormatId,std::string_view protocol) noexcept;
// Exact (storage, protocol, arithmetic) row: several arithmetics may share a storage and a protocol (EXL3 transports).
const SourceRegistration* find_source(FormatId,std::string_view protocol,std::string_view arithmetic) noexcept;
}  // namespace strix::weight::hip
