#include "catalog.hpp"
#include "h3/gemm/operand.hpp"
#include <cstring>
namespace strix::weight::detail {
namespace {
Status grouped_geometry(const Parameters& parameters,Geometry& out) {
 if(parameters.group_size<4 || (16%parameters.group_size && parameters.group_size%16))
  return {Error::malformed,"W4A8 group size must divide or be divisible by 16"};
 out={2,1,parameters.group_size,1,1,256,1,true}; return {Error::ok,{}};
}
Status float_sidecar(const WeightView& view,std::string_view name,std::size_t count,std::vector<float>& out) {
 const auto* part=sidecar(view,name);
 if(!part || part->scalar!=Scalar::f32 || part->data.range.bytes!=count*4)
  return {Error::malformed,"missing or malformed F32 sidecar: "+std::string(name)};
 std::vector<Byte> bytes; auto status=read(part->data,bytes); if(!status) return status;
 out.resize(count); std::memcpy(out.data(),bytes.data(),bytes.size()); return {Error::ok,{}};
}
}
std::span<const FormatDescriptor> channel_formats() {
 static const FormatDescriptor formats[] = {
  descriptor({"linear.i8-channel-convrot256",1},"int8_tensorwise:convrot256",transformed_linear,true,fixed<1,1,256>),
  descriptor({"linear.w4a8-grouped",1},"asym_w4a8_int8",transformed_linear,true,grouped_geometry)
 };
 return formats;
}
Status decode_channel(const WeightView& view,const std::vector<Byte>& raw,std::vector<float>& out) {
 if(view.shape.rank!=2 || view.shape.dims[1]%256 || view.parameters.transform!="convrot256")
  return {Error::unsupported_shape,"channel quant requires K multiple of 256 and declared convrot256"};
 const auto rows=view.shape.dims[0],width=view.shape.dims[1];
 std::vector<float> channel;
 auto status=float_sidecar(view,"channel_scales",rows,channel); if(!status) return status;
 for(float scale:channel) if(!std::isfinite(scale)) return {Error::malformed,"channel scale must be finite"};
 if(view.storage.name=="linear.i8-channel-convrot256") {
  for(std::size_t i=0;i<out.size();++i) out[i]=float(static_cast<std::int8_t>(std::to_integer<std::uint8_t>(raw[i])))*channel[i/width];
  return {Error::ok,{}};
 }
 h3::gemm::PackedW4A8 packed;
 packed.group_size=view.parameters.group_size;
 const auto groups=out.size()/packed.group_size;
 const auto* scales=sidecar(view,"group_scales");
 if(!scales) return {Error::malformed,"W4A8 requires group scales"};
 if(scales->scalar==Scalar::f32) {
  status=float_sidecar(view,"group_scales",groups,packed.scales_f32); if(!status) return status;
 } else if(scales->scalar==Scalar::e4m3fn) {
  std::vector<Byte> bytes; status=read(scales->data,bytes); if(!status) return status;
  if(bytes.size()!=groups) return {Error::malformed,"W4A8 group-scale shape"};
  packed.scales_fp8.resize(groups); std::memcpy(packed.scales_fp8.data(),bytes.data(),groups);
 } else return {Error::malformed,"W4A8 scales require F32 or E4M3FN"};
 if(sidecar(view,"codebook")) { status=float_sidecar(view,"codebook",16,packed.codebook); if(!status) return status; }
 packed.validate(static_cast<std::uint32_t>(rows),static_cast<std::uint32_t>(width),raw.size());
 for(std::size_t i=0;i<out.size();++i)
  out[i]=float(packed.at(reinterpret_cast<const std::int8_t*>(raw.data()),i))*channel[i/width];
 return {Error::ok,{}};
}
}
