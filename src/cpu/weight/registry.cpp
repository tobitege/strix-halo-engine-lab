#include "cpu/weight/source.hpp"
#include <exception>
namespace strix::weight::cpu {
namespace {
Status eligible(const WeightView& weight,const Workload& work,const Hardware& hardware) {
 if(hardware.backend!=Backend::cpu) return {Error::unsupported_backend,"CPU Q8 tree32 source requires CPU backend"};
 if(!work.m || work.m>UINT32_MAX || work.experts>1 || work.active_experts>1)
  return {Error::unsupported_shape,"CPU Q8 tree32 registration is dense rows only"};
 return validate_q8_shape(weight,work.n,work.k);
}
Status dispatch(const Plan& plan,const WeightView& weight,const KernelArguments& envelope,void* output) {
 if(plan.selected.kernel!=q8_tree32_kernel || !(plan.selected.arithmetic==ArithmeticId{"cpu-q8a8-tree32",1}) ||
    plan.selected.strategy!=Strategy::native) return {Error::unsupported_arithmetic,"CPU registry only provides raw Q8_0 tree32 incumbent"};
 if(envelope.schema!=q8_tree32_kernel || envelope.abi_revision!=2 || envelope.bytes!=sizeof(DenseArguments) || !envelope.data)
  return {Error::malformed,"CPU Q8 typed launch schema/ABI"};
 const auto& args=*static_cast<const DenseArguments*>(envelope.data);
 const auto binding=args.binding.retain();
 auto status=binding.validate(weight,args.view.weight,args.view.row_bytes); if(!status) return status;
 if(binding.is_image()) return {Error::unsupported_backend,"CPU launch requires a host binding"};
 auto view=args.view; view.out=static_cast<float*>(output);
 view.weight=binding.data(); view.row_bytes=binding.row_bytes();
 status=validate_view(view); if(!status) return status;
 status=validate_q8_shape(weight,view.count,view.in); if(!status) return status;
 const auto* kernels=::strix::cpu::registered_weight_kernels(::strix::cpu::Isa::scalar);
 if(!kernels || !kernels->gemv_q8_0_a8) return {Error::unavailable_isa,"CPU scalar Q8 registration missing"};
 try { kernels->gemv_q8_0_a8(view,args.stream); return {Error::ok,{}}; }
 catch(const std::exception& error) { return {Error::device_failure,error.what()}; }
}
}
std::span<const Candidate> candidates() noexcept {
 static const Candidate entries[] = {{q8_tree32_kernel,{"gguf.q8_0",1},{"cpu-q8a8-tree32",1},{"cpu.q8-raw",1},
  Backend::cpu,Strategy::native,Exactness::legacy_bitwise,0,{}, {},eligible}};
 return entries;
}
Status validate_view(const ::strix::cpu::Q8a8View& view) {
 if(!view.weight || !view.codes || !view.deltas || !view.out) return {Error::malformed,"CPU Q8 launch has null operand"};
 if(!view.in || !view.count || !view.rows || view.in%Q8Storage::elements || view.x_stride%Q8Storage::elements ||
    view.x_stride<view.in || Q8Storage::row_bytes(view.in)>view.row_bytes || view.out_stride<view.count)
  return {Error::unsupported_shape,"CPU Q8 launch dimensions/strides"};
 return {Error::ok,{}};
}
Status from_legacy(const ::strix::cpu::Q8a8View& view,std::span<const Byte> bytes,std::shared_ptr<const void> owner,TensorId tensor,WeightView& out) {
 auto status=validate_view(view); if(!status) return status;
 if(bytes.data()!=reinterpret_cast<const Byte*>(view.weight)) return {Error::malformed,"CPU source span differs from weight pointer"};
 return q8_weight_from_legacy(bytes,std::move(owner),view.count,view.in,view.row_bytes,0,tensor,out);
}
ExecutionContext execution_context() {
 static const auto callbacks=[] {
  const auto base=cpu_execution_context();
  auto result=std::make_shared<RuntimeCallbacks>(*base.callbacks);
  result->enqueue=[](const Plan& plan,const WeightView& weight,const KernelArguments& args,void* output,void*,std::size_t,ExecutionContext&) {
   return dispatch(plan,weight,args,output);
  };
  return result;
 }();
 return {Backend::cpu,nullptr,0,callbacks};
}
Status enqueue_dense(const WeightView& weight,const DenseArguments& args) {
 Plan plan{}; plan.selected=candidates()[0];
 auto context=execution_context();
 auto launch=args;
 return enqueue(plan,weight,arguments(launch),args.view.out,nullptr,0,context);
}
Status require_isa(::strix::cpu::Isa isa) {
 if(::strix::cpu::registered_weight_kernels(isa)) return {Error::ok,{}};
 return {Error::unavailable_isa,"CPU architecture weight source is a scaffold; dependency=qualified ISA kernel; fixture=CPU backend exactness; removal=register validated implementation"};
}
}
