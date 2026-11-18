#pragma once
#include "cases.hpp"
#include "strix/weight/host_q8.hpp"
#include "strix/weight/resolve.hpp"
#include "strix/ds41_formats.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
namespace strix::weight::test {
struct DenseFixture {
 std::uint32_t rows,n,k,prefix,row_bytes,x_stride,out_stride;
 std::shared_ptr<std::vector<std::uint8_t>> weights;
 std::vector<float> input,deltas,expected,actual;
 std::vector<std::int8_t> codes;
 std::vector<std::int32_t> sums;
};
inline DenseFixture dense_fixture(std::uint32_t rows,std::uint32_t n,std::uint32_t k,
                                  std::uint32_t prefix,std::uint32_t padding,std::uint32_t seed) {
 DenseFixture f{}; f.rows=rows; f.n=n; f.k=k; f.prefix=prefix;
 f.row_bytes=prefix+static_cast<std::uint32_t>(Q8Storage::row_bytes(k))+padding;
 f.x_stride=k+32; f.out_stride=n+3;
 f.weights=std::make_shared<std::vector<std::uint8_t>>(std::size_t(n)*f.row_bytes,0xa5);
 constexpr std::array<std::uint16_t,8> scales{0,0x8000,1,0x8001,0x0400,0x3c00,0xbc00,0x7bff};
 for(std::uint32_t o=0;o<n;++o) for(std::uint32_t g=0;g<k/32;++g) {
  auto* block=f.weights->data()+std::size_t(o)*f.row_bytes+prefix+std::size_t(g)*34;
  const auto scale=scales[(o+g+seed)%scales.size()]; block[0]=static_cast<std::uint8_t>(scale); block[1]=static_cast<std::uint8_t>(scale>>8);
  for(std::uint32_t j=0;j<32;++j) block[2+j]=static_cast<std::uint8_t>(o*13+g*37+j*53+seed);
 }
 f.input.resize(std::size_t(rows)*f.x_stride,123.0F);
 f.codes.resize(std::size_t(rows)*f.x_stride,17);
 f.deltas.resize(std::size_t(rows)*(f.x_stride/32),1.0F);
 f.sums.resize(f.deltas.size(),0);
 for(std::uint32_t r=0;r<rows;++r) {
  for(std::uint32_t j=0;j<k;++j) {
   const auto g=j/32;
   float value=float(int((j*7+r*19+seed)%257)-128)/33.0F;
   if(g%5==0) value=0;
   else if(g%7==0) value*=0x1p-120F;
   f.input[std::size_t(r)*f.x_stride+j]=value;
  }
  const auto activation=::strix::ds41::quantize_a8({f.input.data()+std::size_t(r)*f.x_stride,k});
  std::copy(activation.codes.begin(),activation.codes.end(),f.codes.begin()+std::size_t(r)*f.x_stride);
  std::copy(activation.deltas.begin(),activation.deltas.end(),f.deltas.begin()+std::size_t(r)*(f.x_stride/32));
  std::copy(activation.sums.begin(),activation.sums.end(),f.sums.begin()+std::size_t(r)*(f.x_stride/32));
 }
 f.expected.resize(std::size_t(rows)*f.out_stride,1.25F); f.actual=f.expected;
 return f;
}
template<class T> bool same_bytes(const std::vector<T>& a,const std::vector<T>& b) {
 return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0;
}
inline bool check(bool condition,std::string_view message) {
 if(!condition) std::cerr << message << '\n'; return condition;
}
template<class Arguments,class Launch>
bool binding_rejections(const WeightView& weight,const Arguments& args,Launch launch) {
 auto bad=args; bad.binding={};
 if(!check(launch(bad).code==Error::malformed,"missing Q8 owning binding accepted")) return false;
 bad=args; bad.view.row_bytes+=7;
 if(!check(launch(bad).code==Error::malformed,"compact Q8 image accepted stale padded stride")) return false;
 auto bytes=std::make_shared<std::vector<Byte>>(static_cast<std::size_t>(weight.payload.range.bytes));
 if(!read_slice(weight.payload,*bytes)) return false;
 (*bytes)[2]^=Byte{1};
 WeightView other{};
 if(!q8_weight_from_legacy(*bytes,bytes,args.view.count,args.view.in,args.view.row_bytes,0,weight.tensor,other)) return false;
 HostQ8Binding replacement;
 if(!prepare_host_q8(other,replacement)) return false;
 bad=args; bad.view.weight=replacement.data();
 if(!check(launch(bad).code==Error::malformed,"unowned Q8 operand accepted")) return false;
 bad.binding=replacement;
 if(!check(launch(bad).code==Error::stale_source,"same-shaped Q8 binding from another source accepted")) return false;
 return true;
}
inline bool resolved(const WeightView& weight,const Candidate& candidate,std::uint32_t rows,Plan& plan) {
 WeightMapEntry tensor{}; tensor.weight=weight; tensor.binding.tensor=weight.tensor;
 tensor.binding.role=Role::dense_up; tensor.binding.logical_shape=weight.shape; tensor.binding.expert_axis=UINT32_MAX;
 tensor.binding.admitted_formats.push_back(weight.storage);
 ResolveRequest request{}; request.tensor=&tensor; request.hardware.backend=candidate.backend;
 request.hardware.arch="host-reference"; request.hardware.compiler_id="host-gate";
 request.hardware.device_class="host-gate"; request.hardware.free_bytes=UINT64_MAX; request.hardware.cache_budget_bytes=UINT64_MAX;
 request.required_arithmetic=candidate.arithmetic; request.maximum_permission=Exactness::legacy_bitwise; request.incumbent=candidate.kernel;
 request.workload={rows,weight.shape.dims[0],weight.shape.dims[1],1,1,{},1,false,false};
 const auto status=resolve(builtin_registry(),request,{&candidate,1},{},plan);
 return check(bool(status),status.detail);
}
inline bool record_dense(std::string_view directory,std::string_view name,std::string_view arithmetic,
                         const WeightView& weight,const DenseFixture& f,std::span<const float> partials={}) {
 if(directory.empty()) return true;
 std::vector<Byte> packed(static_cast<std::size_t>(weight.payload.range.bytes));
 if(!read_slice(weight.payload,packed)) return false;
 std::vector<ByteSection> sections{{"source",packed},{"activation_codes",std::as_bytes(std::span(f.codes))},
  {"activation_scales",std::as_bytes(std::span(f.deltas))},{"output",std::as_bytes(std::span(f.actual))}};
 if(!partials.empty()) sections.push_back({"partials",std::as_bytes(partials)});
 const std::array<std::uint64_t,3> shape{f.rows,f.n,f.k};
 return write_capture_sections(directory,name,"gguf.q8_0@1",arithmetic,shape,sections);
}
inline constexpr std::array<std::uint32_t,14> tails{1,2,7,8,9,31,32,33,63,64,65,127,128,129};
}
