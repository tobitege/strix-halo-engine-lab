#include "backend_fixture.hpp"
#include "intel/weight/source.hpp"
namespace {
using namespace strix::weight;
using namespace strix::weight::test;
bool intel_case(std::string_view directory,std::uint32_t rows,std::uint32_t n,std::uint32_t k,std::uint32_t accumulate,std::uint32_t seed) {
 auto f=dense_fixture(rows,n,k,seed%2?17u:0u,7,seed);
 const ::strix::intel::dense::Q8A8View old{f.weights->data(),f.codes.data(),f.deltas.data(),f.expected.data(),
   f.row_bytes,f.prefix,k,n,rows,f.x_stride,f.out_stride,accumulate};
 ::strix::intel::dense::run_reference(old);
 WeightView weight{};
 auto status=intel::from_legacy(old,std::as_bytes(std::span(*f.weights)),f.weights,seed+1,weight);
 if(!check(bool(status),status.detail)) return false;
 HostQ8Binding binding;
 status=prepare_host_q8(weight,binding); if(!check(bool(status),status.detail)) return false;
 intel::DenseArguments args{old,binding};
 args.view.weight=binding.data(); args.view.row_bytes=binding.row_bytes(); args.view.k_bytes=0; args.view.out=f.actual.data();
 const intel::Q8F16Source<intel::IntelAscending> source{binding.data(),binding.row_bytes(),0};
 for(std::uint32_t o=0;o<n;++o) for(std::uint32_t g=0;g<k/32;++g) {
  const auto* raw=f.weights->data()+std::size_t(o)*f.row_bytes+f.prefix+std::size_t(g)*34;
  const auto* block=source.block(o,g);
  if(std::memcmp(raw,block,34)) return false;
  std::uint16_t bits=0; std::memcpy(&bits,raw,2);
  const float scale=::strix::intel::fp16::to_float(bits);
  for(unsigned lane=0;lane<32;++lane) {
   const float expected=::strix::intel::fp16::round(static_cast<float>(static_cast<std::int8_t>(raw[2+lane]))*scale);
   if(std::bit_cast<std::uint32_t>(expected)!=std::bit_cast<std::uint32_t>(source.weight(block,lane))) return false;
  }
 }
 for(std::uint32_t r=0;r<rows;++r) for(std::uint32_t j=0;j<k;++j) {
  const auto code=f.codes[std::size_t(r)*f.x_stride+j]; const auto delta=f.deltas[std::size_t(r)*(f.x_stride/32)+j/32];
  const float expected=::strix::intel::fp16::round(static_cast<float>(code)*delta);
  if(std::bit_cast<std::uint32_t>(expected)!=std::bit_cast<std::uint32_t>(source.activation(code,delta))) return false;
 }
 auto context=intel::reference_execution_context();
 status=intel::enqueue_dense(weight,args,context);
 if(!check(bool(status),status.detail) || !check(same_bytes(f.expected,f.actual),"Intel explicit host oracle registry output differs")) return false;
 std::fill(f.actual.begin(),f.actual.end(),1.25F);
 Plan plan{}; if(!resolved(weight,intel::candidates()[0],rows,plan)) return false;
 status=enqueue(plan,weight,arguments(args),f.actual.data(),nullptr,0,context);
 if(!check(bool(status),status.detail) || !check(same_bytes(f.expected,f.actual),"Intel generic host oracle output differs")) return false;
 const auto name="intel-"+std::to_string(rows)+"-"+std::to_string(n)+"-"+std::to_string(k)+"-"+std::to_string(accumulate);
 if(!record_dense(directory,name,"intel-q8a8-f16-ascending@1",weight,f)) return false;
 if(!binding_rejections(weight,args,[&](const intel::DenseArguments& bad) { return intel::enqueue_dense(weight,bad,context); })) return false;
 auto stale_prefix=args; stale_prefix.view.k_bytes=7;
 if(intel::enqueue_dense(weight,stale_prefix,context).code!=Error::malformed) return false;
 const auto final=f.actual;
 auto native=intel::native_execution_context({});
 const auto native_result=intel::enqueue_dense(weight,args,native);
 const auto expected_native=::strix::intel::sycl_built()?Error::device_failure:Error::unsupported_backend;
 if(!check(native_result.code==expected_native,"Intel native dispatch silently used a host oracle") || !same_bytes(final,f.actual)) return false;
 auto malformed=arguments(args); ++malformed.abi_revision;
 if(enqueue(plan,weight,malformed,f.actual.data(),nullptr,0,context).code!=Error::malformed) return false;
 for(auto id:{FormatId{"gguf.iq3_s",1},FormatId{"exl3.mul1.tiles16",1},FormatId{"nvidia.nvfp4",1}}) {
  auto unsupported=weight; unsupported.storage=id;
  Workload work{rows,n,k,1,1,{},1,false,false}; Hardware hardware{}; hardware.backend=Backend::intel; hardware.arch="host-reference";
  if(intel::candidates()[0].validate(unsupported,work,hardware).code!=Error::unsupported_arithmetic) return false;
 }
 return true;
}
bool intel_cases(std::string_view directory) {
 std::size_t cases=0;
 if(!intel_case(directory,1,2,128,0,1)) return false;
 for(auto rows:tails) for(auto k:{128u,256u,1024u}) for(unsigned accumulate:{0u,1u})
  if(!intel_case(directory,rows,1u+2u*(rows%4),k,accumulate,static_cast<std::uint32_t>(++cases))) return false;
 for(auto n:tails) for(unsigned accumulate:{0u,1u})
  if(!intel_case(directory,1,n,128,accumulate,static_cast<std::uint32_t>(++cases))) return false;
 if(std::string_view(::strix::intel::dense::kQ8A8PolicyName)!="dense-q8a8-f16-v1" ||
    !(intel::candidates()[0].arithmetic==ArithmeticId{"intel-q8a8-f16-ascending",1})) return false;
 auto wrong=cpu_execution_context(); if(intel::install_native_sources({},wrong).code!=Error::unsupported_backend) return false;
 std::cout << "Intel ascending F16 host-oracle cases=" << cases << "; native empty-queue dispatch never falls back\n";
 return true;
}
}
bool wfmt_intel_test() { return intel_cases({}); }
bool wfmt_intel_capture(std::string_view directory) { return intel_cases(directory); }
