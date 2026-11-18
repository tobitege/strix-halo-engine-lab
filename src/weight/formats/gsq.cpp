#include "catalog.hpp"
#include "strix/weight/formats/ggml.hpp"
#include "strix/ggml_dequant.hpp"
#include "strix/iq4_nl.hpp"
#include "strix/ds41_formats.hpp"
#include <bit>
#include <cstring>
namespace strix::weight::detail {
namespace {
// Original canonical table fragments retain their existing provenance/notices.
constexpr std::uint64_t iq2s_grid[] = {
#include "strix/qwen_iq2_s_grid.inc"
};
constexpr std::uint32_t iq3xxs_grid[] = {
#include "strix/qwen_iq3_xxs_grid.inc"
};
constexpr std::uint32_t iq3s_grid[] = {
#include "strix/qwen_iq3_s_grid.inc"
};
std::uint16_t u16(const std::uint8_t* p) { return std::uint16_t(p[0]) | (std::uint16_t(p[1])<<8); }
std::uint32_t u32(const std::uint8_t* p) { return std::uint32_t(p[0]) | (std::uint32_t(p[1])<<8) | (std::uint32_t(p[2])<<16) | (std::uint32_t(p[3])<<24); }
}
Status group(const WeightView& view,std::uint64_t index,SignedGroup32& out) {
 auto status=validate(view); if(!status) return status;
 std::uint64_t count=0; status=elements(view.shape,count); if(!status) return status;
 if(index>=count/32) return {Error::unsupported_shape,"signed group outside tensor"};
 Geometry geometry{}; status=format_geometry(view.storage,view.parameters,geometry); if(!status) return status;
 // Validation bounds the tensor extent; index bounds make these products and
 // the source-relative offset safe. Read only the block owning this group.
 auto block=view.payload;
 block.range.offset+=(index*32/geometry.block_elements)*geometry.block_bytes;
 block.range.bytes=geometry.block_bytes;
 std::vector<Byte> raw; status=read(block,raw); if(!status) return status;
 const auto type=gguf_type(view.storage);
 const auto* x=reinterpret_cast<const std::uint8_t*>(raw.data());
 const auto ib=unsigned((index*32%geometry.block_elements)/32);
 out={}; out.scale_count=1;
 const float d=ggml_dequant::fp16(u16(x));
 if(type==8) {
  const auto g=ds41::q8_0_group(std::span<const std::uint8_t,34>(x,34));
  out.codes=g.codes; out.scales[0]=g.scale;
 } else if(type==16) {
  const auto g=ds41::iq2_xxs_group(std::span<const std::uint8_t,66>(x,66),ib);
  out.codes=g.codes; out.scales[0]=g.scale;
 } else if(type==39) {
  const auto g=ds41::mxfp4_group(std::span<const std::uint8_t,17>(x,17));
  out.codes=g.codes; out.scales[0]=g.scale;
 } else if(type==10) {
  const auto g=ds41::q2_k_group(std::span<const std::uint8_t,84>(x,84),ib);
  out.scale_count=2;
  for(unsigned h=0;h<2;++h) { out.scales[h]=g.d*float(g.scale[h]&15); out.offsets[h]=-g.dmin*float(g.scale[h]>>4); }
  for(unsigned j=0;j<32;++j) { out.codes[j]=static_cast<std::int8_t>(g.codes[j]); out.scale_index[j]=static_cast<std::uint8_t>(j/16); }
 } else if(type==42) {
  out.scales[0]=d;
  for(unsigned j=0;j<32;++j) { const auto element=32*ib+j; out.codes[j]=static_cast<std::int8_t>(int((x[2+element/4]>>(2*(element%4)))&3)-1); }
 } else if(type==20 || type==23) {
  const auto* qs=type==20?x+2:x+8+ib*16;
  if(type==20) out.scales[0]=d;
  else { const auto high=u16(x+2); const auto ls=((x[4+ib/2]>>(4*(ib%2)))&15) | (((high>>(2*ib))&3)<<4); out.scales[0]=d*float(int(ls)-32); }
  for(unsigned j=0;j<16;++j) { out.codes[j]=iq4_nl::codebook[qs[j]&15]; out.codes[j+16]=iq4_nl::codebook[qs[j]>>4]; }
 } else if(type==21) {
  using Iq=formats::Iq3S;
  const auto nib=(x[Iq::scales+ib/2]>>(4*(ib%2)))&15;
  out.scales[0]=d*float(1+2*int(nib));
  const auto qh=x[Iq::qh+ib]; const auto* qs=x+Iq::qs+ib*8; const auto* signs=x+Iq::signs+ib*4;
  for(unsigned l=0;l<4;++l) {
   const auto a=iq3s_grid[qs[2*l] | ((qh<<(8-2*l))&256)];
   const auto b=iq3s_grid[qs[2*l+1] | ((qh<<(7-2*l))&256)];
   for(unsigned j=0;j<8;++j) {
    const auto code=((j<4?a:b)>>(8*(j%4)))&255;
    out.codes[8*l+j]=static_cast<std::int8_t>(int(code)*((signs[l]>>j)&1?-1:1));
   }
  }
 } else if(type==18) {
  const auto aux=u32(x+66+4*ib); out.scales[0]=(d*(0.5F+float(aux>>28)))*0.5F;
  const auto* qs=x+2+8*ib;
  for(unsigned l=0;l<4;++l) {
   const auto s7=(aux>>(7*l))&127; const auto signs=s7|((std::popcount(s7)&1)<<7);
   const auto a=iq3xxs_grid[qs[2*l]],b=iq3xxs_grid[qs[2*l+1]];
   for(unsigned j=0;j<8;++j) out.codes[8*l+j]=static_cast<std::int8_t>(int(((j<4?a:b)>>(8*(j%4)))&255)*((signs>>j)&1?-1:1));
  }
 } else if(type==22) {
  const auto scale=x[74+ib],qh=x[66+ib]; out.scale_count=2;
  out.scales[0]=(d*(0.5F+float(scale&15)))*0.25F;
  out.scales[1]=(d*(0.5F+float(scale>>4)))*0.25F;
  const auto* qs=x+2+4*ib; const auto* signs=x+34+4*ib;
  for(unsigned l=0;l<4;++l) {
   const auto grid=iq2s_grid[qs[l]|((qh<<(8-2*l))&768)];
   for(unsigned j=0;j<8;++j) { out.codes[8*l+j]=static_cast<std::int8_t>(int((grid>>(8*j))&255)*((signs[l]>>j)&1?-1:1)); out.scale_index[8*l+j]=static_cast<std::uint8_t>(l/2); }
  }
 } else return no_group(view,index,out);
 return {Error::ok,{}};
}
}
