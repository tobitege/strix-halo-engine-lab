#include "backend_fixture.hpp"
#include "cpu/weight/source.hpp"
namespace {
using namespace strix::weight;
using namespace strix::weight::test;
#if defined(STRIX_WFMT_CAPTURE)
void partial_sink(const float* values,std::size_t count,void* user) {
 auto& out=*static_cast<std::vector<float>*>(user); out.insert(out.end(),values,values+count);
}
#endif
bool cpu_case(std::string_view directory,std::uint32_t rows,std::uint32_t n,std::uint32_t k,std::uint32_t accumulate,std::uint32_t seed) {
 auto f=dense_fixture(rows,n,k,0,seed%2?7u:0u,seed);
 const auto& kernels=::strix::cpu::reference_kernels();
 auto codes=f.codes; auto deltas=f.deltas; auto sums=f.sums;
 for(std::uint32_t r=0;r<rows;++r) kernels.quantize_a8(f.input.data()+std::size_t(r)*f.x_stride,k,
   codes.data()+std::size_t(r)*f.x_stride,deltas.data()+std::size_t(r)*(f.x_stride/32),sums.data()+std::size_t(r)*(f.x_stride/32));
 if(!check(same_bytes(codes,f.codes) && same_bytes(deltas,f.deltas) && same_bytes(sums,f.sums),"CPU activation codes/scales changed")) return false;
 f.codes=std::move(codes); f.deltas=std::move(deltas);
 ::strix::cpu::Q8a8View old{f.weights->data(),f.codes.data(),f.deltas.data(),f.expected.data(),
   f.row_bytes,k,n,rows,f.x_stride,f.out_stride,accumulate};
 std::vector<float> expected_partials,actual_partials;
#if defined(STRIX_WFMT_CAPTURE)
 expected_partials.reserve(std::size_t(rows)*n*32); actual_partials.reserve(std::size_t(rows)*n*32);
 cpu::set_partial_sink(partial_sink,&expected_partials);
#endif
 kernels.gemv_q8_0_a8(old,{});
#if defined(STRIX_WFMT_CAPTURE)
 cpu::set_partial_sink(nullptr,nullptr);
#endif
 WeightView weight{};
 auto status=cpu::from_legacy(old,std::as_bytes(std::span(*f.weights)),f.weights,seed+1,weight);
 if(!check(bool(status),status.detail)) return false;
 HostQ8Binding binding;
 status=prepare_host_q8(weight,binding); if(!check(bool(status),status.detail)) return false;
 cpu::DenseArguments args{old,{},binding};
 args.view.weight=binding.data(); args.view.row_bytes=binding.row_bytes(); args.view.out=f.actual.data();
 const cpu::Q8RawSource<cpu::CpuTree32> source{binding.data(),binding.row_bytes()};
 for(std::uint32_t o=0;o<n;++o) for(std::uint32_t g=0;g<k/32;++g) {
  const auto* raw=f.weights->data()+std::size_t(o)*f.row_bytes+std::size_t(g)*34;
  const auto* block=source.block(o,g);
  if(std::memcmp(raw,block,34)) return false;
  for(unsigned lane=0;lane<32;++lane) if(source.code(block,lane)!=static_cast<std::int8_t>(raw[2+lane])) return false;
 }
#if defined(STRIX_WFMT_CAPTURE)
 cpu::set_partial_sink(partial_sink,&actual_partials);
#endif
 status=cpu::enqueue_dense(weight,args);
#if defined(STRIX_WFMT_CAPTURE)
 cpu::set_partial_sink(nullptr,nullptr);
#endif
 if(!check(bool(status),status.detail) || !check(same_bytes(f.expected,f.actual),"CPU direct registry output differs") ||
    !check(same_bytes(expected_partials,actual_partials),"CPU direct partials differ")) return false;
 std::fill(f.actual.begin(),f.actual.end(),1.25F); actual_partials.clear();
 Plan plan{}; if(!resolved(weight,cpu::candidates()[0],rows,plan)) return false;
 auto context=cpu::execution_context();
#if defined(STRIX_WFMT_CAPTURE)
 cpu::set_partial_sink(partial_sink,&actual_partials);
#endif
 status=enqueue(plan,weight,arguments(args),f.actual.data(),nullptr,0,context);
#if defined(STRIX_WFMT_CAPTURE)
 cpu::set_partial_sink(nullptr,nullptr);
#endif
 if(!check(bool(status),status.detail) || !check(same_bytes(f.expected,f.actual),"CPU generic registry output differs") ||
    !check(same_bytes(expected_partials,actual_partials),"CPU generic partials differ")) return false;
 auto malformed=arguments(args); ++malformed.abi_revision;
 if(enqueue(plan,weight,malformed,f.actual.data(),nullptr,0,context).code!=Error::malformed) return false;
 for(auto id:{FormatId{"gguf.iq3_s",1},FormatId{"exl3.mul1.tiles16",1},FormatId{"nvidia.nvfp4",1}}) {
  auto unsupported=weight; unsupported.storage=id;
  Workload work{rows,n,k,1,1,{},1,false,false}; Hardware hardware{}; hardware.backend=Backend::cpu;
  if(cpu::candidates()[0].validate(unsupported,work,hardware).code!=Error::unsupported_arithmetic) return false;
 }
 const auto name="cpu-"+std::to_string(rows)+"-"+std::to_string(n)+"-"+std::to_string(k)+"-"+std::to_string(accumulate);
 if(!record_dense(directory,name,"cpu-q8a8-tree32@1",weight,f,actual_partials)) return false;
 if(!binding_rejections(weight,args,[&](const cpu::DenseArguments& bad) { return cpu::enqueue_dense(weight,bad); })) return false;
 std::weak_ptr<std::vector<std::uint8_t>> owner=f.weights; f.weights.reset();
 if(owner.expired()) return false;
 weight={}; binding={}; args.binding={};
 return check(owner.expired(),"CPU weight source retained an unexpected owner");
}
bool cpu_cases(std::string_view directory) {
 std::size_t cases=0;
 if(!cpu_case(directory,1,2,32,0,1)) return false; // 68-byte binding, legacy row stride 41.
 for(auto rows:tails) for(auto k:{32u,64u,128u,256u,1024u,1056u}) for(unsigned accumulate:{0u,1u}) {
  const auto n=1u+2u*(rows%4);
  if(!cpu_case(directory,rows,n,k,accumulate,static_cast<std::uint32_t>(++cases))) return false;
 }
 for(auto n:tails) for(unsigned accumulate:{0u,1u})
  if(!cpu_case(directory,1,n,128,accumulate,static_cast<std::uint32_t>(++cases))) return false;
 if(!cpu::require_isa(::strix::cpu::Isa::scalar)) return false;
 for(auto isa:{::strix::cpu::Isa::avx2,::strix::cpu::Isa::avx512,::strix::cpu::Isa::amx,::strix::cpu::Isa::neon,::strix::cpu::Isa::sve})
  if(cpu::require_isa(isa).code!=Error::unavailable_isa) return false;
 std::cout << "CPU Q8 tree32 cases=" << cases << " capture=" <<
#if defined(STRIX_WFMT_CAPTURE)
 "partials+output\n";
#else
 "production-output\n";
#endif
 return true;
}
}
bool wfmt_cpu_test() { return cpu_cases({}); }
bool wfmt_cpu_capture(std::string_view directory) { return cpu_cases(directory); }
