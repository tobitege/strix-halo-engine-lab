#include "intel/weight/source.hpp"
#include <exception>
namespace strix::weight::intel {
namespace {
Status eligible(const WeightView& weight,const Workload& work,const Hardware& hardware) {
 if(hardware.backend!=Backend::intel) return {Error::unsupported_backend,"Intel ascending F16 source requires Intel backend"};
 if(!work.m || work.m>UINT32_MAX || work.k%128 || work.experts>1 || work.active_experts>1)
  return {Error::unsupported_shape,"Intel Q8 F16 incumbent requires dense rows and K multiple of 128"};
 if(hardware.arch!="host-reference" && !::strix::intel::sycl_built()) return {Error::unsupported_backend,
  "Intel native source unavailable; dependency=SYCL backend build; fixture=Intel native operand qualification; removal=build and qualify the existing SYCL boundary"};
 return validate_q8_shape(weight,work.n,work.k);
}
Status launch_view(const Plan& plan,const WeightView& weight,const KernelArguments& envelope,void* output,
                   ::strix::intel::dense::Q8A8View& view,HostQ8Binding& binding) {
 if(plan.selected.kernel!=q8_f16_ascending_kernel ||
    !(plan.selected.arithmetic==ArithmeticId{"intel-q8a8-f16-ascending",1}) || plan.selected.strategy!=Strategy::native)
  return {Error::unsupported_arithmetic,"Intel registry only provides the existing ascending F16 operand policy"};
 if(envelope.schema!=q8_f16_ascending_kernel || envelope.abi_revision!=2 || envelope.bytes!=sizeof(DenseArguments) || !envelope.data)
  return {Error::malformed,"Intel Q8 typed launch schema/ABI"};
 const auto& args=*static_cast<const DenseArguments*>(envelope.data);
 binding=args.binding.retain();
 auto status=binding.validate(weight,args.view.weight,args.view.row_bytes,args.view.k_bytes); if(!status) return status;
 view=args.view; view.out=static_cast<float*>(output);
 view.weight=binding.data(); view.row_bytes=binding.row_bytes(); view.k_bytes=0;
 status=validate_view(view); if(!status) return status;
 return validate_q8_shape(weight,view.count,view.in);
}
Status native_status(::strix::intel::Status status) {
 switch(status) {
  case ::strix::intel::Status::ok: return {Error::ok,{}};
  case ::strix::intel::Status::not_built: return {Error::unsupported_backend,
   "Intel native source unavailable; dependency=SYCL backend build; fixture=Intel native operand qualification; removal=build and qualify the existing SYCL boundary"};
  case ::strix::intel::Status::unsupported: return {Error::unsupported_shape,"Intel native boundary rejected its Q8 view"};
  case ::strix::intel::Status::invalid: return {Error::device_failure,"Intel native queue/operand is invalid"};
  case ::strix::intel::Status::device_error: return {Error::device_failure,"Intel native device or asynchronous failure"};
 }
 return {Error::device_failure,"unknown Intel native status"};
}
}
std::span<const Candidate> candidates() noexcept {
 static const Candidate entries[] = {{q8_f16_ascending_kernel,{"gguf.q8_0",1},{"intel-q8a8-f16-ascending",1},{"intel.q8-f16-ascending",1},
  Backend::intel,Strategy::native,Exactness::legacy_bitwise,0,{}, {},eligible}};
 return entries;
}
Status validate_view(const ::strix::intel::dense::Q8A8View& view) {
 return ::strix::intel::dense::supported(view)?Status{Error::ok,{}}:
  Status{Error::unsupported_shape,"Intel Q8 F16 view does not meet existing supported dimensions/strides"};
}
Status from_legacy(const ::strix::intel::dense::Q8A8View& view,std::span<const Byte> bytes,std::shared_ptr<const void> owner,TensorId tensor,WeightView& out) {
 auto status=validate_view(view); if(!status) return status;
 if(bytes.data()!=reinterpret_cast<const Byte*>(view.weight)) return {Error::malformed,"Intel source span differs from weight pointer"};
 return q8_weight_from_legacy(bytes,std::move(owner),view.count,view.in,view.row_bytes,view.k_bytes,tensor,out);
}
ExecutionContext reference_execution_context() {
 auto context=cpu_execution_context(); context.backend=Backend::intel;
 auto callbacks=std::make_shared<RuntimeCallbacks>(*context.callbacks);
 callbacks->enqueue=[](const Plan& plan,const WeightView& weight,const KernelArguments& args,void* output,void*,std::size_t,ExecutionContext&) -> Status {
  ::strix::intel::dense::Q8A8View view;
  HostQ8Binding binding;
  auto status=launch_view(plan,weight,args,output,view,binding); if(!status) return status;
  if(binding.is_image())
   return {Error::unsupported_backend,"host oracle requires a host binding"};
  ::strix::intel::dense::run_reference(view); return {Error::ok,{}};
 };
 context.callbacks=std::move(callbacks); return context;
}
ExecutionContext native_execution_context(::strix::intel::Queue queue) {
 ExecutionContext context{Backend::intel,queue.impl,0,{}};
 (void)install_native_sources(queue,context);
 return context;
}
Status install_native_sources(::strix::intel::Queue queue,ExecutionContext& context) {
 if(context.backend!=Backend::intel) return {Error::unsupported_backend,"Intel source registration requires an Intel runtime context"};
 if(context.stream!=queue.impl && (context.callbacks || context.stream))
  return {Error::unsupported_backend,"Intel adapter stream must already use Queue::impl"};
 // Preserve vendor allocation/event/copy callbacks installed by the runtime.
 // An empty context gains enqueue only, never CPU allocation or host fallback.
 auto callbacks=context.callbacks?std::make_shared<RuntimeCallbacks>(*context.callbacks):std::make_shared<RuntimeCallbacks>();
 callbacks->enqueue=[](const Plan& plan,const WeightView& weight,const KernelArguments& args,void* output,void*,std::size_t,ExecutionContext& executing) -> Status {
  ::strix::intel::dense::Q8A8View view;
  HostQ8Binding binding;
  auto status=launch_view(plan,weight,args,output,view,binding); if(!status) return status;
  if(!::strix::intel::sycl_built()) return native_status(::strix::intel::Status::not_built);
  if(!executing.stream) return native_status(::strix::intel::Status::invalid);
  ImageLease lease;
  status=binding.acquire(executing,lease); if(!status) return status;
  view.weight=static_cast<const std::uint8_t*>(lease.data());
  const auto launched=native_status(::strix::intel::enqueue_dense_q8a8({executing.stream},view));
  // Fence even a failed submission: the runtime may have accepted some work.
  status=record_last_use(lease,executing);
  return launched?status:launched;
 };
 context.callbacks=std::move(callbacks); context.stream=queue.impl;
 return {Error::ok,{}};
}
Status prepare_native_q8(const WeightView& weight,std::shared_ptr<ImageCache> cache,ExecutionContext& context,HostQ8Binding& out) {
 if(context.backend!=Backend::intel) return {Error::unsupported_backend,"Intel Q8 preparation requires Intel runtime"};
 Plan plan{}; plan.selected=candidates()[0]; plan.selected.resources.image_bytes=weight.payload.range.bytes;
 return prepare_q8_image(weight,plan,std::move(cache),context,out);
}
Status enqueue_dense(const WeightView& weight,const DenseArguments& args,ExecutionContext& context) {
 Plan plan{}; plan.selected=candidates()[0]; auto launch=args;
 return enqueue(plan,weight,arguments(launch),args.view.out,nullptr,0,context);
}
}
