#include "intel/weight/source.hpp"
#include <sycl/sycl.hpp>
#include <array>
#include <map>
#include <stdexcept>

void wfmt_intel_queue_test() {
 namespace w=strix::weight;
 namespace i=strix::intel;
 namespace fake=sycl::test;
 const auto require=[](bool ok,const char* message) { if(!ok) throw std::runtime_error(message); };
 struct QueueOwner {
  i::Queue queue;
  ~QueueOwner() { i::close_queue(queue); }
 } first,second;
 fake::execute=true;
 require(i::open_queue(0,&first.queue,nullptr)==i::Status::ok,"open Q1");
 const auto q1=fake::queues.back();
 require(i::open_queue(0,&second.queue,nullptr)==i::Status::ok,"open Q2");
 const auto q2=fake::queues.back();
 const std::map<void*,std::shared_ptr<fake::Work>> queues{{first.queue.impl,q1},{second.queue.impl,q2}};
 struct Event { std::shared_ptr<fake::Work> queue; std::size_t sequence; };
 std::vector<std::shared_ptr<std::vector<w::Byte>>> staging;
 unsigned retired=0;
 auto api=std::make_shared<w::RuntimeCallbacks>();
 api->allocate=[](std::uint64_t bytes,void*& out) {
  out=new w::Byte[static_cast<std::size_t>(bytes)]; return w::Status{w::Error::ok,{}};
 };
 api->free=[](void* pointer) { delete[] static_cast<w::Byte*>(pointer); };
 api->record_event=[&](w::ExecutionContext& context,void*& out) {
  const auto queue=queues.at(context.stream);
  out=new Event{queue,queue->submitted}; return w::Status{w::Error::ok,{}};
 };
 api->wait_event=[&](w::ExecutionContext& context,void* event) {
  const auto ready=*static_cast<Event*>(event);
  if(queues.at(context.stream)!=ready.queue)
   fake::submit(queues.at(context.stream),[ready] { fake::drain(ready.queue,ready.sequence); });
  return w::Status{w::Error::ok,{}};
 };
 api->event_complete=[](void* event) {
  const auto& e=*static_cast<Event*>(event); return e.queue->completed>=e.sequence;
 };
 api->destroy_event=[](void* event) { delete static_cast<Event*>(event); };
 api->convert=[&](const w::WeightView& weight,const w::Plan&,void* out,w::ExecutionContext& context) {
  auto bytes=std::make_shared<std::vector<w::Byte>>(static_cast<std::size_t>(weight.payload.range.bytes));
  auto status=w::read_slice(weight.payload,*bytes); if(!status) return status;
  staging.push_back(bytes);
  require(i::copy({context.stream},out,bytes->data(),bytes->size())==i::Status::ok,"queued image copy");
  return w::Status{w::Error::ok,{}};
 };
 api->retire=[&](void* pointer,w::Retirement retirement,w::ExecutionContext) {
  require(retirement.unresolved_streams.empty(),"retired image with an unresolved last-use stream");
  for(auto* event:retirement.events) {
   require(api->event_complete(event),"retired image before last use completed");
   api->destroy_event(event);
  }
  api->free(pointer); ++retired;
 };
 w::ExecutionContext context{w::Backend::intel,first.queue.impl,0,api};
 int foreign=0;
 auto incompatible=context; incompatible.stream=&foreign;
 require(w::intel::install_native_sources(first.queue,incompatible).code==w::Error::unsupported_backend,
         "foreign adapter stream accepted");
 require(incompatible.stream==&foreign && incompatible.callbacks==api,"failed install changed adapter");
 require(bool(w::intel::install_native_sources(first.queue,context)),"install native sources");
 auto executing=context; executing.stream=second.queue.impl;
 auto raw=std::make_shared<std::vector<w::Byte>>(136,w::Byte{1});
 for(std::size_t block=0;block<4;++block) { (*raw)[block*34]=w::Byte{0}; (*raw)[block*34+1]=w::Byte{0x3c}; }
 w::WeightView weight{};
 require(bool(w::q8_weight_from_legacy(*raw,raw,1,128,136,0,1,weight)),"owning source");
 auto cache=std::make_shared<w::ImageCache>(136);
 w::HostQ8Binding binding;
 require(bool(w::intel::prepare_native_q8(weight,cache,context,binding)),"prepare native image");
 require(q1->completed==0,"preparation should still be pending");
 bool unrelated=false;
 fake::submit(q1,[&] { unrelated=true; }); // Q2 must wait only for Q1 readiness.
 std::array<std::int8_t,128> codes{},input_codes{}; input_codes.fill(1);
 std::array<float,4> deltas{},input_deltas{}; input_deltas.fill(1.0F);
 float output=-1;
 require(i::copy(second.queue,codes.data(),input_codes.data(),codes.size())==i::Status::ok,"Q2 codes copy");
 require(i::copy(second.queue,deltas.data(),input_deltas.data(),sizeof(deltas))==i::Status::ok,"Q2 scales copy");
 w::intel::DenseArguments args{{binding.data(),codes.data(),deltas.data(),&output,136,0,128,1,1,128,1,0},binding};
 require(bool(w::intel::enqueue_dense(weight,args,executing)),"enqueue on executing Q2");
 args.binding={}; binding={}; raw.reset(); weight={};
 require(output==-1 && cache->evict_completed(136)==0 && retired==0,"pending image retired early");
 require(i::synchronize(second.queue)==i::Status::ok,"wait Q2");
 require(output==128.0F,"Q2 completion preceded kernel or its input/readiness copies");
 require(!unrelated,"kernel or readiness wait consumed unrelated Q1 work");
 require(cache->evict_completed(136)==136 && retired==1,"Q2 last-use event did not release image");
 require(i::synchronize(first.queue)==i::Status::ok && unrelated,"finish Q1");
 fake::execute=false;
}
