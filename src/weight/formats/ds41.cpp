#include "catalog.hpp"
#include "strix/ds41_formats.hpp"
namespace strix::weight::detail {
std::span<const FormatDescriptor> ds41_formats() {
 static const FormatDescriptor formats[] = {
  descriptor({"gguf.iq2_xxs",1},"GGUF:16",quantized_linear,true,fixed<256,66>,true),
  descriptor({"gguf.q2_k",1},"GGUF:10",quantized_linear,true,fixed<256,84>,true),
  descriptor({"gguf.mxfp4",1},"GGUF:39",quantized_linear,true,fixed<32,17>,true),
  descriptor({"row.e4m3-e8m0.g32",1},"row-e4m3-e8m0",row_lookup,true,fixed<256,264>,false,
   (1ull<<unsigned(Role::lookup_row)) | (1ull<<unsigned(Role::embedding)))
 };
 return formats;
}
Status decode_ds41(const WeightView& view,const std::vector<Byte>& raw,std::vector<float>& out) {
 const auto* bytes=reinterpret_cast<const std::uint8_t*>(raw.data());
 Geometry g{}; auto status=format_geometry(view.storage,view.parameters,g); if(!status) return status;
 for(std::size_t block=0;block<out.size()/g.block_elements;++block) {
  const auto* source=bytes+block*g.block_bytes; auto* dest=out.data()+block*g.block_elements;
  if(view.storage.name=="gguf.iq2_xxs") ds41::decode_iq2_xxs(std::span<const std::uint8_t,66>(source,66),std::span<float,256>(dest,256));
  else if(view.storage.name=="gguf.q2_k") ds41::decode_q2_k(std::span<const std::uint8_t,84>(source,84),std::span<float,256>(dest,256));
  else if(view.storage.name=="gguf.mxfp4") ds41::decode_mxfp4(std::span<const std::uint8_t,17>(source,17),std::span<float,32>(dest,32));
  else ds41::decode_engram_row(std::span<const std::uint8_t,264>(source,264),std::span<float,256>(dest,256));
 }
 return {Error::ok,{}};
}
}
