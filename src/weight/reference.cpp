#include "formats/catalog.hpp"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <exception>
#include <cmath>
#include <array>
#include "strix/ggml_dequant.hpp"
#include "strix/ds41_native_fp8.hpp"
namespace strix::weight::detail {
namespace {
Status range_valid(const Slice& slice) {
 if(!slice.source || slice.range.offset>slice.source->size() || slice.range.bytes>slice.source->size()-slice.range.offset)
  return {Error::malformed,"source range outside owning source"};
 return {Error::ok,{}};
}
template<class Check> Status scan(const Slice& slice,std::size_t element_bytes,Check check) {
 std::array<Byte,65536> buffer{};
 if(slice.range.bytes%element_bytes) return {Error::malformed,"sidecar scalar alignment"};
 for(std::uint64_t offset=0;offset<slice.range.bytes;) {
  const auto bytes=static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(),slice.range.bytes-offset));
  auto status=slice.source->read({slice.range.offset+offset,bytes},std::span(buffer).first(bytes)); if(!status) return status;
  for(std::size_t i=0;i<bytes;i+=element_bytes) if(!check(buffer.data()+i)) return {Error::malformed,"invalid code/scale content"};
  offset+=bytes;
 }
 return {Error::ok,{}};
}
bool finite_f32(const Byte* p) { float x=0; std::memcpy(&x,p,4); return std::isfinite(x); }
bool nonnegative_f32(const Byte* p) { float x=0; std::memcpy(&x,p,4); return std::isfinite(x) && x>=0; }
bool finite_half(const Byte* p) { std::uint16_t x=0; std::memcpy(&x,p,2); return std::isfinite(ggml_dequant::fp16(x)); }
Status required(const WeightView& view,std::string_view name,Scalar scalar,Shape shape,const NamedSlice*& result) {
 result=sidecar(view,name);
 if(!result || result->scalar!=scalar || result->shape.rank!=shape.rank)
  return {Error::malformed,"missing or wrong sidecar: "+std::string(name)};
 for(unsigned i=0;i<shape.rank;++i) if(result->shape.dims[i]!=shape.dims[i]) return {Error::malformed,"sidecar shape: "+std::string(name)};
 return {Error::ok,{}};
}
Status validate_special(const WeightView& view,bool scan_values) {
 const auto inspect=[&](const Slice& slice,std::size_t stride,auto predicate) {return scan_values?scan(slice,stride,predicate):Status{Error::ok,{}};};
 const auto name=view.storage.name;
 const auto width=view.shape.dims[view.shape.rank-1],rows=view.shape.rank==2?view.shape.dims[0]:1;
 const NamedSlice* part=nullptr; Status status{Error::ok,{}};
 if(name=="image.cb4") {
  status=required(view,"codebook",Scalar::s8,{{16,0,0,0},1},part); if(!status) return status;
  if(scan_values) {
  std::vector<Byte> bytes; status=read(part->data,bytes); if(!status) return status;
  for(unsigned i=1;i<16;++i) if(static_cast<std::int8_t>(std::to_integer<unsigned char>(bytes[i]))<
     static_cast<std::int8_t>(std::to_integer<unsigned char>(bytes[i-1]))) return {Error::malformed,"CB4 codebook order"};
  }
 }
 if(name.starts_with("image.") && name!="image.q6p32" && !view.parameters.transform.empty() &&
    view.parameters.transform!="planar" && view.parameters.transform!="interleaved") return {Error::malformed,"unknown packed layout"};
 if(name=="block.e4m3-e8m0.32x32") {
  status=required(view,"block_scales",Scalar::e8m0,{{rows/32,width/32,0,0},2},part); if(!status) return status;
  status=inspect(part->data,1,[](const Byte* p) { auto e=std::to_integer<unsigned>(*p); return e>=ds41::fp8_min_exponent && e<=ds41::fp8_max_exponent; }); if(!status) return status;
  return inspect(view.payload,1,[](const Byte* p) { return (std::to_integer<unsigned>(*p)&127)!=127; });
 }
 if(name=="nvidia.nvfp4") {
  if(view.shape.rank!=2 || width%64) return {Error::unsupported_shape,"NVFP4 requires matrix K multiple of 64"};
  status=required(view,"block_scales",Scalar::e4m3fn,{{rows,width/16,0,0},2},part); if(!status) return status;
  status=inspect(part->data,1,[](const Byte* p) { return std::to_integer<unsigned>(*p)<127; }); if(!status) return status;
  status=required(view,"tensor_scale",Scalar::f32,{{1,0,0,0},1},part); if(!status) return status;
  return inspect(part->data,4,[](const Byte* p) { float x=0; std::memcpy(&x,p,4); return std::isfinite(x) && x>0; });
 }
 if(name.starts_with("linear.")) {
  if(view.shape.rank!=2 || width%256 || view.parameters.transform!="convrot256") return {Error::unsupported_shape,"channel quant requires K multiple of 256 and convrot256"};
  status=required(view,"channel_scales",Scalar::f32,{{rows,0,0,0},1},part); if(!status) return status;
  status=inspect(part->data,4,finite_f32); if(!status) return status;
  if(name=="linear.w4a8-grouped") {
   const auto group=view.parameters.group_size;
   if(!group || width%group) return {Error::unsupported_shape,"W4A8 group does not divide K"};
   const auto* scale=sidecar(view,"group_scales");
   if(!scale || (scale->scalar!=Scalar::e4m3fn && scale->scalar!=Scalar::f32)) return {Error::malformed,"W4A8 group scale dtype"};
   status=required(view,"group_scales",scale->scalar,{{rows,width/group,0,0},2},part); if(!status) return status;
   if(part->scalar==Scalar::f32) status=inspect(part->data,4,nonnegative_f32);
   else status=inspect(part->data,1,[](const Byte* p) { const auto code=std::to_integer<unsigned>(*p); return code<127 || code==128; });
   if(!status) return status;
   if(sidecar(view,"codebook")) {
    status=required(view,"codebook",Scalar::f32,{{16,0,0,0},1},part); if(!status) return status;
    status=inspect(part->data,4,finite_f32); if(!status) return status;
   }
  }
 }
 if(name.starts_with("exl3.")) {
  if(view.shape.rank!=2 || width%128 || rows%128 || view.parameters.transform!="hadamard128") return {Error::unsupported_shape,"EXL3 requires declared H128 transform geometry"};
  if(name=="exl3.mul1.tiles16") {
   status=required(view,"codebook_marker",Scalar::i32,{{1,0,0,0},1},part); if(!status) return status;
   status=inspect(part->data,4,[](const Byte* p) {
    return std::to_integer<unsigned>(p[0])==0x2d && std::to_integer<unsigned>(p[1])==0xd1 &&
           std::to_integer<unsigned>(p[2])==0xdc && std::to_integer<unsigned>(p[3])==0x83;
   }); if(!status) return status;
   status=required(view,"input_scale",Scalar::f16,{{width,0,0,0},1},part); if(!status) return status;
   status=inspect(part->data,2,finite_half); if(!status) return status;
   status=required(view,"output_scale",Scalar::f16,{{rows,0,0,0},1},part); if(!status) return status;
   return inspect(part->data,2,finite_half);
  }
  auto scales=view.payload; scales.range.bytes=2*(width+rows);
  return inspect(scales,2,finite_half);
 }
 return status;
}
}
Status elements(const Shape& shape, std::uint64_t& count) {
 count = 1;
 if (!shape.rank || shape.rank > 4) return {Error::unsupported_shape, "rank outside 1..4"};
 for (unsigned i=0;i<shape.rank;++i) {
   if (!shape.dims[i] || count > UINT64_MAX/shape.dims[i]) return {Error::malformed,"empty or overflowing shape"};
   count *= shape.dims[i];
 }
 return {Error::ok,{}};
}
Status read(const Slice& slice, std::vector<Byte>& bytes) {
 if (!slice.source || slice.range.offset > slice.source->size() ||
     slice.range.bytes > slice.source->size()-slice.range.offset || slice.range.bytes > SIZE_MAX)
   return {Error::malformed,"source range outside owning source"};
 bytes.resize(static_cast<std::size_t>(slice.range.bytes));
 return slice.source->read(slice.range, bytes);
}
const NamedSlice* sidecar(const WeightView& view, std::string_view name) {
 const NamedSlice* result = nullptr;
 for (const auto& part : view.sidecars) if (part.name == name) {
   if (result) return nullptr;
   result = &part;
 }
 return result;
}
Status unavailable(const WeightView&,const DecodeRequest&,DecodeBuffer& out) {
 out.written=0; return {Error::unsupported_arithmetic,"geometry-only storage has no semantic decoder"};
}
Status no_group(const WeightView&,std::uint64_t,SignedGroup32&) {
 return {Error::unsupported_arithmetic,"storage has no signed group32 operand contract"};
}
Status validate_impl(const WeightView& view,bool scan_values) {
 std::uint64_t count=0; auto status=elements(view.shape,count); if(!status) return status;
 const auto* format=find_format(builtin_registry(),view.storage);
 if(!format) return {Error::unknown_format,"unregistered storage identity"};
 Geometry g{}; status=format->geometry(view.parameters,g); if(!status) return status;
 if (!g.block_elements || !g.block_bytes || count%g.block_elements || view.shape.dims[view.shape.rank-1]%g.row_alignment)
   return {Error::unsupported_shape,"shape does not meet storage block/row alignment"};
 if (g.payload_alignment && view.payload.range.offset%g.payload_alignment)
   return {Error::malformed,"payload alignment"};
 if(count/g.block_elements > UINT64_MAX/g.block_bytes) return {Error::malformed,"payload size overflow"};
 auto bytes=count/g.block_elements*g.block_bytes;
 if(view.has_byte_strides) {
   // Only scalar row-major storage has a supported byte-stride contract.
   if(g.block_elements!=1 ||
      (!view.storage.name.starts_with("plain.") && !view.storage.name.starts_with("typed.")))
     return {Error::unsupported_shape,"byte strides require contiguous scalar storage"};
   std::uint64_t stride=g.block_bytes;
   for(unsigned axis=view.shape.rank;axis>0;--axis) {
     if(view.byte_strides[axis-1]!=stride)
       return {Error::unsupported_shape,"noncontiguous byte strides are unsupported"};
     if(stride>UINT64_MAX/view.shape.dims[axis-1])
       return {Error::malformed,"byte stride extent overflow"};
     stride*=view.shape.dims[axis-1];
   }
 }
 // Gemma transport has contiguous FP16 input/output sidecars before the rings.
 if(view.storage.name=="exl3.gemma.transport") {
   if(view.shape.rank!=2 || view.shape.dims[0]%128 || view.shape.dims[1]%128)
     return {Error::unsupported_shape,"Gemma transport requires I/O multiples of 128"};
   const auto scales=view.shape.dims[0]+view.shape.dims[1];
   if(scales> (UINT64_MAX-bytes)/2) return {Error::malformed,"transport size overflow"};
   bytes += scales*2;
 }
 if(view.storage.name=="block.e4m3-e8m0.32x32") {
   if(view.shape.rank!=2 || view.shape.dims[0]%32 || view.shape.dims[1]%32)
     return {Error::unsupported_shape,"FP8 matrix requires 32x32 blocks"};
 }
 if(view.payload.range.bytes!=bytes) return {Error::malformed,"payload byte length differs from exact geometry"};
 status=range_valid(view.payload); if(!status) return status;
 std::vector<std::string_view> names;
 for(const auto& part:view.sidecars) {
   if(std::find(names.begin(),names.end(),part.name)!=names.end()) return {Error::malformed,"duplicate sidecar name"};
   names.push_back(part.name);
   std::uint64_t count_sidecar=0; status=elements(part.shape,count_sidecar); if(!status) return status;
   unsigned bits=8;
   switch(part.scalar) {
    case Scalar::f64: case Scalar::i64: case Scalar::u64: bits=64; break;
    case Scalar::f32: case Scalar::i32: case Scalar::u32: bits=32; break;
    case Scalar::f16: case Scalar::bf16: case Scalar::i16: case Scalar::u16: bits=16; break;
    case Scalar::e2m1: bits=4; break;
    default: break;
   }
   if(count_sidecar>UINT64_MAX/bits || (count_sidecar*bits)%8 || part.data.range.bytes!=count_sidecar*bits/8)
    return {Error::malformed,"sidecar scalar byte length"};
   status=range_valid(part.data); if(!status) return status;
 }
 return validate_special(view,scan_values);
}
Status validate(const WeightView& view){return validate_impl(view,true);}
Status validate_metadata(const WeightView& view){return validate_impl(view,false);}

CostDescriptor cost(const WeightView& view,ImageId) {
 std::uint64_t count=0;
 if(!elements(view.shape,count)) return {};
 std::uint64_t sidecar_bytes=0; for(const auto& part:view.sidecars) sidecar_bytes+=part.data.range.bytes;
 if(sidecar_bytes>UINT64_MAX-view.payload.range.bytes) return {};
 const auto packed=view.payload.range.bytes+sidecar_bytes;
 auto divisor=std::gcd(packed,count);
 CostDescriptor result{};
 result.packed_bytes_per_weight={packed/divisor,count/divisor};
 result.image_bytes_per_weight={0,1}; // unknown image cost is never fabricated
 result.fixed_metadata_bytes=0;
 result.operand=Scalar::f32;
 return result;
}
Status decode(const WeightView& view,const DecodeRequest& request,DecodeBuffer& out) {
 out.written=0;
 auto status=validate(view); if(!status) return status;
 std::uint64_t count=0; status=elements(view.shape,count); if(!status) return status;
 if(request.first_element>count || request.count>count-request.first_element)
   return {Error::unsupported_shape,"decode slice outside tensor"};
 if(request.output!=Scalar::f32) return {Error::unsupported_arithmetic,"reference decoder only widens to F32"};
 if(request.count>SIZE_MAX/sizeof(float) || out.bytes.size()<request.count*sizeof(float))
   return {Error::malformed,"decode output buffer too small"};
 if(count>SIZE_MAX/sizeof(float)) return {Error::memory_budget,"reference tensor exceeds addressable scratch storage"};
 // Storage widening does not execute an ArithmeticDescriptor's operand policy.
 if(!request.arithmetic.name.empty() || request.arithmetic.revision!=0)
   return {Error::unsupported_arithmetic,"reference decoder supports storage widening only"};
 std::vector<Byte> raw; status=read(view.payload,raw); if(!status) return status;
 std::vector<float> decoded(static_cast<std::size_t>(count));
 const auto name=view.storage.name;
 if(name.starts_with("plain.")) status=decode_plain(view,raw,decoded);
 else if(name=="gguf.iq2_xxs" || name=="gguf.q2_k" || name=="gguf.mxfp4" || name.starts_with("row.")) status=decode_ds41(view,raw,decoded);
 else if(name.starts_with("gguf.")) status=decode_ggml(view,raw,decoded);
 else if(name.starts_with("image.")) status=decode_packed(view,raw,decoded);
 else if(name.starts_with("block.")) status=decode_fp8(view,raw,decoded);
 else if(name.starts_with("exl3.")) status=decode_exl3(view,raw,decoded);
 else if(name.starts_with("linear.")) status=decode_channel(view,raw,decoded);
 else if(name=="nvidia.nvfp4") status=decode_nvfp4(view,raw,decoded);
 else status={Error::unsupported_arithmetic,"typed storage is not a usable linear weight"};
 if(!status) return status;
 std::memcpy(out.bytes.data(),decoded.data()+request.first_element,request.count*sizeof(float));
 out.written=static_cast<std::size_t>(request.count*sizeof(float));
 return {Error::ok,{}};
}
}
namespace strix::weight {
Status decode_reference(const Registry& registry,const WeightView& view,const DecodeRequest& request,DecodeBuffer& out) {
 out.written=0;
 const auto* format=find_format(registry,view.storage);
 if(!format) return {Error::unknown_format,"unregistered storage identity"};
 try { return format->decode_reference(view,request,out); }
 catch(const std::exception& error) { return {Error::malformed,error.what()}; }
}
Status signed_group_reference(const Registry& registry,const WeightView& view,std::uint64_t group,SignedGroup32& out) {
 const auto* format=find_format(registry,view.storage);
 if(!format) return {Error::unknown_format,"unregistered storage identity"};
 try { return format->signed_group_reference(view,group,out); }
 catch(const std::exception& error) { return {Error::malformed,error.what()}; }
}
}
