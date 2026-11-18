#include "fixture.hpp"
#include "cases.hpp"
#include "strix/ggml_dequant.hpp"
#include "strix/ds41_formats.hpp"
#include "strix/ds41_packed.hpp"
#include "strix/ds41_native_fp8.hpp"
#include "strix/q6p_registry.hpp"
#include "strix/cuda/core/nvfp4.hpp"
#include "gemma4/exl3/format.hpp"
#include "h3/gemm/operand.hpp"
#include <filesystem>
#include <cmath>
bool wfmt_strides_test();
bool wfmt_arithmetic_test();
bool wfmt_group_reads_test();
namespace strix::weight::test {
namespace {
std::vector<Byte> seeded(std::size_t count,std::uint32_t seed) {
 std::vector<Byte> result(count);
 for(auto& byte:result) { seed=seed*1664525u+1013904223u; byte=Byte(seed>>24); }
 return result;
}
std::vector<Byte> halves(std::size_t count) {
 std::vector<std::uint16_t> values(count,0x3c00);
 return bytes_of(std::span<const std::uint16_t>(values));
}
std::vector<Byte> floats(std::size_t count,float value) {
 std::vector<float> values(count,value); return bytes_of(std::span<const float>(values));
}
std::vector<Byte> source_bytes(const Slice& slice) {
 std::vector<Byte> out(static_cast<std::size_t>(slice.range.bytes));
 if(!slice.source->read(slice.range,out)) throw std::runtime_error("fixture source read");
 return out;
}
}
WeightView fixture(const FormatDescriptor& format,std::uint32_t seed) {
 WeightView view{}; view.storage=format.id; view.parameters.bits=4; view.parameters.group_size=16;
 const auto name=format.id.name;
 if(name.starts_with("exl3.")) view.parameters.bits=seed==0?4:seed==1?5:seed==UINT32_MAX?8:6;
 Geometry g{}; if(!format.geometry(view.parameters,g)) throw std::runtime_error("fixture geometry"); view.geometry=g;
 auto width=std::uint64_t(g.block_elements),rows=std::uint64_t(1);
 if(name=="block.e4m3-e8m0.32x32") { width=32; rows=32; }
 if(name=="nvidia.nvfp4") width=64;
 if(name.starts_with("linear.")) { width=256; rows=2; view.parameters.transform="convrot256"; }
 if(name.starts_with("exl3.")) { width=128; rows=128; view.parameters.transform="hadamard128"; }
 view.shape=matrix(rows,width);
 std::size_t size=static_cast<std::size_t>(rows*width/g.block_elements*g.block_bytes);
 if(name=="exl3.gemma.transport") size+=2*(rows+width);
 auto raw=seeded(size,seed);
 if(name.starts_with("gguf.") || name.starts_with("image.")) {
  for(std::size_t offset=0;offset+1<raw.size();offset+=g.block_bytes) {
   raw[offset]=Byte{0}; raw[offset+1]=Byte{0x3c};
   if(g.block_bytes>4) { raw[offset+2]=Byte{0}; raw[offset+3]=Byte{0}; }
  }
 }
 if(name=="plain.f32") raw=floats(rows*width,-0.0F);
 if(name=="plain.f16" || name=="plain.bf16") raw=halves(rows*width);
 if(name=="block.e4m3-e8m0.32x32") {
  for(auto& byte:raw) byte&=Byte{0x7e};
  view.sidecars.push_back(named("block_scales",Scalar::e8m0,matrix(rows/32,width/32),std::vector<Byte>(rows*width/1024,Byte{127})));
 }
 if(name=="image.cb4") {
  std::vector<Byte> table(16); for(unsigned i=0;i<16;++i) table[i]=Byte(static_cast<unsigned char>(int(i)-8));
  view.sidecars.push_back(named("codebook",Scalar::s8,{{16,0,0,0},1},std::move(table)));
 }
 if(name=="nvidia.nvfp4") {
  view.sidecars.push_back(named("block_scales",Scalar::e4m3fn,matrix(rows,width/16),std::vector<Byte>(rows*width/16,Byte{0x38})));
  view.sidecars.push_back(named("tensor_scale",Scalar::f32,{{1,0,0,0},1},floats(1,1.0F)));
 }
 if(name.starts_with("linear.")) {
  view.sidecars.push_back(named("channel_scales",Scalar::f32,{{rows,0,0,0},1},floats(rows,0.5F)));
  if(name=="linear.w4a8-grouped") view.sidecars.push_back(named("group_scales",Scalar::e4m3fn,
      matrix(rows,width/view.parameters.group_size),std::vector<Byte>(rows*width/view.parameters.group_size,Byte{0x38})));
 }
 if(name=="exl3.gemma.transport") {
  const auto scales=halves(rows+width); std::copy(scales.begin(),scales.end(),raw.begin());
 }
 if(name=="exl3.mul1.tiles16") {
  view.sidecars.push_back(named("codebook_marker",Scalar::i32,{{1,0,0,0},1},{Byte{0x2d},Byte{0xd1},Byte{0xdc},Byte{0x83}}));
  view.sidecars.push_back(named("input_scale",Scalar::f16,{{width,0,0,0},1},halves(width)));
  view.sidecars.push_back(named("output_scale",Scalar::f16,{{rows,0,0,0},1},halves(rows)));
 }
 view.payload=memory(std::move(raw));
 return view;
}
bool legacy_decode(const WeightView& view,std::vector<float>& out) {
 const auto raw=source_bytes(view.payload); const auto name=view.storage.name;
 const auto rows=view.shape.dims[0],width=view.shape.dims[1]; out.resize(rows*width);
 const auto* bytes=reinterpret_cast<const std::uint8_t*>(raw.data());
 const auto type=gguf_type(view.storage);
 if(type!=UINT32_MAX && ggml_dequant::supported(type)) {
  ggml_dequant::dequantize_row(type,raw,out); return true;
 }
 if(name=="gguf.iq2_xxs") {
  for(std::size_t block=0;block<out.size()/256;++block) ds41::decode_iq2_xxs(std::span<const std::uint8_t,66>(bytes+66*block,66),std::span<float,256>(out.data()+256*block,256)); return true;
 }
 if(name=="gguf.q2_k") {
  for(std::size_t block=0;block<out.size()/256;++block) ds41::decode_q2_k(std::span<const std::uint8_t,84>(bytes+84*block,84),std::span<float,256>(out.data()+256*block,256)); return true;
 }
 if(name=="gguf.mxfp4") {
  for(std::size_t block=0;block<out.size()/32;++block) ds41::decode_mxfp4(std::span<const std::uint8_t,17>(bytes+17*block,17),std::span<float,32>(out.data()+32*block,32)); return true;
 }
 if(name=="row.e4m3-e8m0.g32") {
  for(std::size_t block=0;block<out.size()/256;++block) ds41::decode_engram_row(std::span<const std::uint8_t,264>(bytes+264*block,264),std::span<float,256>(out.data()+256*block,256)); return true;
 }
 if(name=="image.q6p32") {
  std::vector<unsigned char> q8(out.size()/32*34); q6p::to_q8_0(bytes,rows,width,q8.data());
  ggml_dequant::dequantize_row(8,std::as_bytes(std::span(q8)),out); return true;
 }
 if(name.starts_with("image.")) {
  ds41::PackedChoice choice; choice.kind=name=="image.cb4"?ds41::PackedKind::cb4:name=="image.u5"?ds41::PackedKind::u5:ds41::PackedKind::u6;
  if(choice.kind==ds41::PackedKind::cb4) { const auto table=source_bytes(view.sidecars[0].data); std::memcpy(choice.table.data(),table.data(),16); }
  for(std::size_t row=0;row<rows;++row) for(std::size_t g=0;g<width/32;++g) {
   std::array<std::uint8_t,34> q8{}; ds41::unpack_group(bytes+row*ds41::packed_row_bytes(choice.kind,width),width,g,choice,
       view.parameters.transform=="planar"?ds41::PackedLayout::planar:ds41::PackedLayout::interleaved,q8.data());
   ds41::decode_q8_0(q8,std::span<float,32>(out.data()+row*width+32*g,32));
  } return true;
 }
 if(name=="block.e4m3-e8m0.32x32") {
  const auto scales=source_bytes(view.sidecars[0].data); std::vector<std::uint8_t> joined(raw.size()+scales.size());
  std::memcpy(joined.data(),raw.data(),raw.size()); std::memcpy(joined.data()+raw.size(),scales.data(),scales.size());
  for(std::size_t row=0;row<rows;++row) for(std::size_t k=0;k<width;++k) out[row*width+k]=ds41::fp8_weight(joined.data(),rows,width,row,k);
  return true;
 }
 if(name.starts_with("exl3.")) {
  std::vector<std::uint8_t> transport(2*(rows+width)+rows*width*view.parameters.bits/8);
  if(name=="exl3.gemma.transport") std::memcpy(transport.data(),raw.data(),raw.size());
  else {
   const auto scales=halves(rows+width); std::memcpy(transport.data(),scales.data(),scales.size());
   std::memcpy(transport.data()+scales.size(),raw.data(),raw.size());
  }
  const gemma4::LinearView w{transport.data(),static_cast<unsigned>(rows),static_cast<unsigned>(width),width*view.parameters.bits/8,gemma4::WeightFormat::exl3_mul1};
  for(unsigned row=0;row<rows;++row) for(unsigned k=0;k<width;++k) out[row*width+k]=gemma4::exl3::rotated_at(w,k,row);
  return true;
 }
 if(name.starts_with("linear.")) {
  h3::gemm::PackedW4A8 packed; packed.group_size=view.parameters.group_size;
  if(name=="linear.w4a8-grouped") packed.scales_fp8.assign(rows*width/packed.group_size,0x38);
  for(std::size_t i=0;i<out.size();++i) {
   const auto code=name=="linear.w4a8-grouped"?packed.at(reinterpret_cast<const std::int8_t*>(bytes),i):static_cast<std::int8_t>(bytes[i]);
   out[i]=float(code)*0.5F;
  } return true;
 }
 if(name=="nvidia.nvfp4") {
  namespace nv=cuda::core::nvfp4;
  for(std::size_t i=0;i<out.size();++i) out[i]=(nv::e2m1((bytes[i/2]>>(4*(i%2)))&15)*nv::e4m3(0x38))*1.0F;
  return true;
 }
 return false;
}
}
using namespace strix::weight;
using namespace strix::weight::test;
namespace {
bool run_references(std::string_view directory) {
 std::size_t count=0;
 for(const auto& format:builtin_registry().formats) {
  if(!format.semantic_decode_supported) continue;
  for(std::uint32_t seed:{0u,1u,0x12345678u,UINT32_MAX}) {
   const auto view=fixture(format,seed); std::vector<float> expected;
   if(!require(legacy_decode(view,expected),"missing legacy oracle")) return false;
   std::vector<Byte> result(expected.size()*4); DecodeBuffer output{result,0};
   const auto status=decode_reference(builtin_registry(),view,{0,expected.size(),{},Scalar::f32},output);
   if(!status) { std::cerr << serialize_id(view.storage) << ": " << status.detail << '\n'; return false; }
   const auto old=std::as_bytes(std::span(expected));
   if(!require(result.size()==old.size() && std::equal(result.begin(),result.end(),old.begin()),"legacy decoder bytes differ")) return false;
   if(!directory.empty()) {
    std::string name(format.id.name); std::replace(name.begin(),name.end(),'.','_'); name+="-"+std::to_string(seed);
    const auto base=std::filesystem::path(directory)/"baseline",candidate=std::filesystem::path(directory)/"candidate";
    const auto raw=source_bytes(view.payload); std::vector<Byte> sidecars;
    for(const auto& part:view.sidecars) { const auto bytes=source_bytes(part.data); sidecars.insert(sidecars.end(),bytes.begin(),bytes.end()); }
    if(!write_capture_record(base.string(),name,serialize_id(format.id),std::span(view.shape.dims).first(view.shape.rank),raw,old,sidecars) ||
       !write_capture_record(candidate.string(),name,serialize_id(format.id),std::span(view.shape.dims).first(view.shape.rank),raw,result,sidecars)) return false;
   }
   // Output slice and negative output-size checks are independent of format decoding.
   std::array<Byte,4> last{}; DecodeBuffer tail{last,0};
   if(!decode_reference(builtin_registry(),view,{expected.size()-1,1,{},Scalar::f32},tail) ||
      !std::equal(last.begin(),last.end(),old.end()-4)) return false;
   DecodeBuffer short_buffer{std::span(result).first(3),0};
   if(decode_reference(builtin_registry(),view,{0,1,{},Scalar::f32},short_buffer)) return false;
   ++count;
  }
 }
 std::cout << "legacy-reference-cases=" << count << '\n';
 return true;
}
}
bool wfmt_reference_test() {
 if(!wfmt_strides_test() || !wfmt_arithmetic_test() || !wfmt_group_reads_test()) return false;
 if(!run_references({})) return false;
 // Exhaustive scalar encodings exercise signed zeros, supported subnormals,
 // infinities and every NaN payload through the unchanged widening delegate.
 std::vector<std::uint16_t> encodings(65536);
 for(unsigned i=0;i<encodings.size();++i) encodings[i]=static_cast<std::uint16_t>(i);
 for(auto id:{FormatId{"plain.f16",1},FormatId{"plain.bf16",1}}) {
  const auto* format=find_format(builtin_registry(),id); if(!format) return false;
  auto view=fixture(*format,0); view.shape=matrix(1,encodings.size());
  view.payload=memory(bytes_of(std::span<const std::uint16_t>(encodings)));
  std::vector<float> old; if(!legacy_decode(view,old)) return false;
  std::vector<Byte> bytes(old.size()*4); DecodeBuffer output{bytes,0};
  if(!decode_reference(builtin_registry(),view,{0,old.size(),{},Scalar::f32},output) ||
     !std::equal(bytes.begin(),bytes.end(),std::as_bytes(std::span(old)).begin())) return false;
 }
 const auto* fp4=find_format(builtin_registry(),{"nvidia.nvfp4",1}); if(!fp4) return false;
 for(unsigned scale=0;scale<256;++scale) for(unsigned code=0;code<16;++code) {
  auto view=fixture(*fp4,0);
  view.payload=memory(std::vector<Byte>(32,Byte(code|(code<<4))));
  view.sidecars[0].data=memory(std::vector<Byte>(4,Byte(scale)));
  std::vector<Byte> bytes(64*4); DecodeBuffer output{bytes,0};
  const auto status=decode_reference(builtin_registry(),view,{0,64,{},Scalar::f32},output);
  if(scale>=127) { if(status) return false; continue; }
  if(!status) return false;
  namespace nv=strix::cuda::core::nvfp4;
  const float value=(nv::e2m1(static_cast<std::uint8_t>(code))*nv::e4m3(static_cast<std::uint8_t>(scale)))*1.0F;
  for(unsigned i=0;i<64;++i) if(std::memcmp(bytes.data()+4*i,&value,4)) return false;
 }
 const auto* trellis=find_format(builtin_registry(),{"exl3.mul1.tiles16",1}); if(!trellis) return false;
 for(unsigned bits=1;bits<=8;++bits) {
  auto view=fixture(*trellis,0); view.parameters.bits=bits;
  if(!trellis->geometry(view.parameters,view.geometry)) return false;
  view.payload=memory(seeded(view.shape.dims[0]*view.shape.dims[1]*bits/8,bits));
  std::vector<float> old; if(!legacy_decode(view,old)) return false;
  std::vector<Byte> bytes(old.size()*4); DecodeBuffer output{bytes,0};
  if(!decode_reference(builtin_registry(),view,{0,old.size(),{},Scalar::f32},output) ||
     !std::equal(bytes.begin(),bytes.end(),std::as_bytes(std::span(old)).begin())) return false;
 }
 const auto* w4=find_format(builtin_registry(),{"linear.w4a8-grouped",1}); if(!w4) return false;
 auto signed_zero=fixture(*w4,0);
 signed_zero.sidecars[1].data=memory(std::vector<Byte>(32,Byte{0x80}));
 std::vector<Byte> zero_output(512*4); DecodeBuffer zero_buffer{zero_output,0};
 if(!decode_reference(builtin_registry(),signed_zero,{0,512,{},Scalar::f32},zero_buffer) ||
    std::any_of(zero_output.begin(),zero_output.end(),[](Byte byte) { return byte!=Byte{0}; })) return false;
 for(const auto& format:builtin_registry().formats) {
  if(!format.semantic_decode_supported) continue;
  auto view=fixture(format,1); view.payload.range.offset=UINT64_MAX;
  if(validate_weight(view)) return false;
  view=fixture(format,1); ++view.shape.dims[view.shape.rank-1];
  if(validate_weight(view)) return false;
  view=fixture(format,1);
  if(!view.sidecars.empty()) {
   auto missing=view; missing.sidecars.erase(missing.sidecars.begin());
   if(validate_weight(missing)) return false;
   auto wrong=view; wrong.sidecars[0].scalar=Scalar::f64;
   if(validate_weight(wrong)) return false;
   auto duplicate=view; duplicate.sidecars.push_back(view.sidecars[0]);
   if(validate_weight(duplicate)) return false;
  }
 }
 std::cout << "exhaustive-F16/BF16=131072 NVFP4-code-scale=4096 trellis-bits=1..8 malformed-schema PASS\n";
 return true;
}
bool wfmt_reference_capture(std::string_view directory) { return run_references(directory); }
