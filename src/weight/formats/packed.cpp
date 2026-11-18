#include "catalog.hpp"
#include "strix/ds41_packed.hpp"
#include "strix/ggml_dequant.hpp"
#include "strix/q6p_registry.hpp"
#include <algorithm>
#include <cstring>
namespace strix::weight::detail {
std::span<const FormatDescriptor> packed_formats() {
 static const FormatDescriptor formats[] = {
  descriptor({"image.cb4",1},"derived-cb4",quantized_linear,true,fixed<32,18>),
  descriptor({"image.u5",1},"derived-u5",quantized_linear,true,fixed<32,22>),
  descriptor({"image.u6",1},"derived-u6",quantized_linear,true,fixed<32,26>),
  descriptor({"image.q6p32",1},"derived-q6p32",quantized_linear,true,fixed<32,26>)
 };
 return formats;
}
Status decode_packed(const WeightView& view,const std::vector<Byte>& raw,std::vector<float>& out) {
 const auto width=view.shape.dims[view.shape.rank-1];
 const auto* bytes=reinterpret_cast<const std::uint8_t*>(raw.data());
 if(view.storage.name=="image.q6p32") {
  std::vector<std::uint8_t> q8(out.size()/32*34);
  q6p::to_q8_0(bytes,out.size()/width,width,q8.data());
  ggml_dequant::dequantize_row(8,std::as_bytes(std::span(q8)),out);
  return {Error::ok,{}};
 }
 ds41::PackedChoice choice;
 choice.kind=view.storage.name=="image.cb4"?ds41::PackedKind::cb4:
             view.storage.name=="image.u5"?ds41::PackedKind::u5:ds41::PackedKind::u6;
 if(choice.kind==ds41::PackedKind::cb4) {
  const auto* table=sidecar(view,"codebook");
  if(!table || table->scalar!=Scalar::s8 || table->shape.rank!=1 || table->shape.dims[0]!=16 || table->data.range.bytes!=16)
   return {Error::malformed,"CB4 requires signed 16-entry codebook"};
  std::vector<Byte> codes; auto status=read(table->data,codes); if(!status) return status;
  std::memcpy(choice.table.data(),codes.data(),16);
  if(!std::is_sorted(choice.table.begin(),choice.table.end())) return {Error::malformed,"CB4 codebook must be sorted"};
 }
 ds41::PackedLayout layout;
 if(view.parameters.transform.empty() || view.parameters.transform=="interleaved") layout=ds41::PackedLayout::interleaved;
 else if(view.parameters.transform=="planar") layout=ds41::PackedLayout::planar;
 else return {Error::malformed,"unknown packed layout"};
 const auto row_bytes=ds41::packed_row_bytes(choice.kind,width);
 for(std::size_t row=0;row<out.size()/width;++row) for(std::size_t g=0;g<width/32;++g) {
  std::array<std::uint8_t,34> q8{};
  ds41::unpack_group(bytes+row*row_bytes,width,g,choice,layout,q8.data());
  ggml_dequant::dequantize_row(8,std::as_bytes(std::span(q8)),std::span(out).subspan(row*width+32*g,32));
 }
 return {Error::ok,{}};
}
}
