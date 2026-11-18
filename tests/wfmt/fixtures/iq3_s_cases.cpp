#include "../fixture.hpp"
#include "strix/weight/formats/ggml.hpp"
#include "strix/ggml_dequant.hpp"
#include <cmath>
#include <bit>
using namespace strix::weight;
using namespace strix::weight::test;
namespace {
bool compare_groups(const WeightView& view) {
 std::vector<float> expected; if(!legacy_decode(view,expected)) return false;
 for(std::size_t index=0;index<expected.size()/32;++index) {
  SignedGroup32 group{};
  const auto status=signed_group_reference(builtin_registry(),view,index,group);
  if(!status) { std::cerr << status.detail << '\n'; return false; }
  if(group.scale_count<1 || group.scale_count>32) return false;
  for(unsigned j=0;j<32;++j) {
   const auto slot=group.scale_index[j]; if(slot>=group.scale_count) return false;
   const auto value=group.scales[slot]*float(group.codes[j])+group.offsets[slot];
   const auto old=expected[index*32+j];
   // Adding the optional affine +0 would erase negative zero: compare the
   // non-affine operand using its declared multiply only.
   const auto operand=group.offsets[slot]==0?group.scales[slot]*float(group.codes[j]):value;
   if(std::bit_cast<std::uint32_t>(operand)!=std::bit_cast<std::uint32_t>(old)) {
    std::cerr << serialize_id(view.storage) << " group=" << index << " lane=" << j << '\n'; return false;
   }
  }
 }
 return true;
}
}
bool wfmt_iq3s_test() {
 const auto* iq3=find_format(builtin_registry(),{"gguf.iq3_s",1}); if(!iq3) return false;
 using Iq=formats::Iq3S;
 for(unsigned index=0;index<512;++index) {
  auto view=fixture(*iq3,0); std::vector<Byte> raw(view.payload.range.bytes,Byte{0}); raw[1]=Byte{0x3c};
  for(unsigned group=0;group<8;++group) {
   for(unsigned q=0;q<8;++q) raw[Iq::qs+group*8+q]=Byte(index&255);
   raw[Iq::qh+group]=Byte(index&256?255:0);
   for(unsigned sign=0;sign<4;++sign) raw[Iq::signs+group*4+sign]=Byte((index+37*sign+group)&255);
  }
  view.payload=memory(std::move(raw)); if(!compare_groups(view)) return false;
 }
 for(unsigned nibble=0;nibble<16;++nibble) for(unsigned sign=0;sign<256;++sign) {
  auto view=fixture(*iq3,sign+nibble); auto source=std::static_pointer_cast<const MemorySource>(view.payload.source);
  auto raw=source->bytes;
  for(unsigned i=Iq::signs;i<Iq::scales;++i) raw[i]=Byte(sign);
  for(unsigned i=Iq::scales;i<Iq::bytes;++i) raw[i]=Byte(nibble|(nibble<<4));
  view.payload=memory(std::move(raw)); if(!compare_groups(view)) return false;
 }
 // Exhaustive half scale encodings: finite values, signed zero and subnormals
 // preserve the ordered scale expression, independently of codebook extraction.
 for(unsigned bits=0;bits<65536;++bits) {
  const auto d=strix::ggml_dequant::fp16(static_cast<std::uint16_t>(bits));
  if(!std::isfinite(d)) continue;
  auto view=fixture(*iq3,bits); auto source=std::static_pointer_cast<const MemorySource>(view.payload.source); auto raw=source->bytes;
  raw[0]=Byte(bits&255); raw[1]=Byte(bits>>8);
  view.payload=memory(std::move(raw)); if(!compare_groups(view)) return false;
 }
 for(auto wire:{8u,16u,18u,20u,22u,23u,39u,42u}) {
  const auto* format=find_format(builtin_registry(),gguf_format(wire)); if(!format) return false;
  for(unsigned seed:{0u,1u,127u,255u}) if(!compare_groups(fixture(*format,seed))) return false;
 }
 auto bad=fixture(*iq3,1); bad.payload.range.bytes-=1;
 SignedGroup32 group{};
 if(signed_group_reference(builtin_registry(),bad,0,group)) return false;
 const auto valid=fixture(*iq3,1);
 if(signed_group_reference(builtin_registry(),valid,8,group)) return false;
 std::cout << "IQ3_S exhaustive grid/sign/scale boundaries PASS\n";
 return true;
}
