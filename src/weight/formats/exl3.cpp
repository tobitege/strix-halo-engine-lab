#include "catalog.hpp"
#include "gemma4/exl3/format.hpp"
namespace strix::weight::detail {
namespace {
Status geometry(const Parameters& p,Geometry& out) {
 if(p.bits<1 || p.bits>8) return {Error::malformed,"EXL3 bits must be in 1..8"};
 out={256,32*p.bits,16,16,16,16,1,false}; return {Error::ok,{}};
}
Status gemma_geometry(const Parameters& p,Geometry& out) {
 if(p.bits!=4 && p.bits!=5 && p.bits!=6 && p.bits!=8) return {Error::malformed,"Gemma EXL3 bits must be 4,5,6,8"};
 return geometry(p,out);
}
}
std::span<const FormatDescriptor> exl3_formats() {
 static const FormatDescriptor formats[] = {
  descriptor({"exl3.mul1.tiles16",1},"EXL3:mul1:tiles16",transformed_linear,true,geometry),
  descriptor({"exl3.gemma.transport",1},"Gemma:EXL3:contiguous",transformed_linear,true,gemma_geometry)
 };
 return formats;
}
Status decode_exl3(const WeightView& view,const std::vector<Byte>& raw,std::vector<float>& out) {
 using namespace gemma4::exl3;
 if(view.shape.rank!=2 || view.shape.dims[0]%16 || view.shape.dims[1]%16)
  return {Error::unsupported_shape,"EXL3 requires complete 16x16 tiles"};
 const auto width=view.shape.dims[1],rows=view.shape.dims[0];
 const auto* data=reinterpret_cast<const std::uint8_t*>(raw.data());
 const bool gemma=view.storage.name=="exl3.gemma.transport";
 const auto* rings=data+(gemma?2*(width+rows):0);
 if(gemma) {
  strix::gemma4::LinearView linear{data,static_cast<unsigned>(rows),static_cast<unsigned>(width),
       static_cast<std::size_t>(width*view.parameters.bits/8),strix::gemma4::WeightFormat::exl3_mul1};
  gemma4::exl3::validate(linear);
 }
 // Decode the transformed-domain coefficients. Input/output transforms remain
 // attached to the arithmetic identity; this never invents an unrotated matrix.
 using Format=core::trellis::Format;
 for(std::size_t row=0;row<rows;++row) for(std::size_t col=0;col<width;++col) {
  const auto* tile=rings+Format::tile(col/16,row/16,rows/16,int(view.parameters.bits))*4;
  out[row*width+col]=codebook(Format::state_bytes(tile,int(view.parameters.bits),Format::position(unsigned(col%16),unsigned(row%16))));
 }
 return {Error::ok,{}};
}
}
