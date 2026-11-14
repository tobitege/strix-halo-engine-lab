#include "strix/hip_check.hpp"
#include "strix/hip_integer_projection_runtime.hpp"
#include "strix/hip_launch_context.hpp"
#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace strix::integer_projection {
namespace {
void require(bool ok,const char* message){if(!ok)throw std::invalid_argument(message);}
using strix::hip_check;
hipStream_t native_stream(hipStream_t stream) noexcept {return stream==hipStreamLegacy?nullptr:stream;}
std::uint64_t unique_generation(){
    static std::atomic<std::uint64_t> next{0x4950520000000000ULL};auto current=next.load();
    do{if(current==UINT64_MAX)throw std::overflow_error("integer runtime owner generations exhausted");}
    while(!next.compare_exchange_weak(current,current+1));return current;
}
std::uint64_t add(std::uint64_t a,std::uint64_t b){if(b>UINT64_MAX-a)throw std::overflow_error("integer runtime extent overflow");return a+b;}
void target(const DeviceIdentity& d,hipStream_t stream){
    const auto* override_arch=std::getenv("HSA_OVERRIDE_GFX_VERSION");require(!override_arch||!*override_arch,"integer runtime rejects architecture override");
    require(d.index>=0&&d.instance_generation&&d.architecture=="gfx1151","integer runtime requires explicit gfx1151 instance");
    int current=-1;hip_check(hipGetDevice(&current),"integer runtime current device");require(current==d.index,"integer runtime device changed");
    // HIP's legacy/default stream is a sentinel tied to the current device,
    // not an allocated stream handle. Some runtimes dereference that sentinel.
    hipDevice_t owner=current;
    if(stream!=nullptr&&stream!=hipStreamLegacy)
        hip_check(hipStreamGetDevice(stream,&owner),"integer runtime stream device");
    require(owner==d.index,"integer runtime stream device mismatch");
    hipDeviceProp_t props{};hip_check(hipGetDeviceProperties(&props,current),"integer runtime device properties");
    std::string architecture=props.gcnArchName;require(architecture.substr(0,architecture.find(':'))==d.architecture,"integer runtime target architecture mismatch");
}
AllocationBinding native_allocation(const DeviceIdentity& d,std::uintptr_t address,std::uint64_t generation,std::uint64_t identity){
    hipPointerAttribute_t attr{};hip_check(hipPointerGetAttributes(&attr,reinterpret_cast<const void*>(address)),"integer runtime pointer attributes");
    require(attr.type==hipMemoryTypeDevice&&!attr.isManaged&&attr.device==d.index,"integer runtime owner factory requires device-local allocation");
    hipDeviceptr_t base=nullptr;std::size_t bytes=0;hip_check(hipMemGetAddressRange(&base,&bytes,reinterpret_cast<hipDeviceptr_t>(address)),"integer runtime allocation extent");
    return {reinterpret_cast<std::uintptr_t>(base),bytes,identity,generation,d,1};
}
void physical_extent(const AllocationBinding& binding,bool registered){
    hipPointerAttribute_t attr{};hip_check(hipPointerGetAttributes(&attr,reinterpret_cast<const void*>(binding.base)),"integer runtime source attributes");
    if(registered){require(attr.type==hipMemoryTypeHost&&!attr.isManaged&&attr.devicePointer,"integer runtime registered mapping is not GPU visible");return;}
    const auto actual=native_allocation(binding.device,binding.base,binding.generation,binding.identity);
    require(actual.base==binding.base&&actual.bytes==binding.bytes,"integer runtime owning device allocation extent changed");
}
struct Owned final:AllocationPin {
    AllocationBinding binding;
    hipStream_t stream=nullptr;
    explicit Owned(const DeviceIdentity& device,std::uint64_t bytes,hipStream_t on_stream):stream(on_stream){
        void* p=nullptr;hip_check(hipMalloc(&p,static_cast<std::size_t>(bytes)),"integer runtime allocation");
        try{const auto id=unique_generation();binding=native_allocation(device,reinterpret_cast<std::uintptr_t>(p),id,id);}
        catch(...){(void)hipFree(p);throw;}
    }
    AllocationBinding current_binding()const override{return binding;}
    ~Owned()override{
        int active=-1;if(hipGetDevice(&active)==hipSuccess&&active==binding.device.index&&hipStreamQuery(native_stream(stream))==hipSuccess)
            (void)hipFree(reinterpret_cast<void*>(binding.base)); // Otherwise retain raw allocation until process teardown.
    }
};
template<class T>T* pointer(const std::shared_ptr<Owned>& owner){return reinterpret_cast<T*>(owner->binding.base);}
__global__ void widen_input(const void* input,double* output,unsigned k,bool bf16){
    const auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=k)return;
    const auto width=bf16?2U:4U;
    const auto* bytes=static_cast<const unsigned char*>(input)+static_cast<std::uint64_t>(i)*width;
    unsigned bits=static_cast<unsigned>(bytes[0])|(static_cast<unsigned>(bytes[1])<<8U);
    if(bf16)bits<<=16U;else bits|=(static_cast<unsigned>(bytes[2])<<16U)|(static_cast<unsigned>(bytes[3])<<24U);
    output[i]=__longlong_as_double(static_cast<long long>(binary64_widen_bits(bits)));
}
__global__ void repack_weights(const unsigned char* checkpoint,unsigned char* packed,double* scales,
    unsigned k,unsigned first,unsigned columns,bool iq){
    const auto index=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const auto groups=k/32U;if(index>=static_cast<std::uint64_t>(columns)*groups)return;
    const auto column=first+static_cast<unsigned>(index/groups),group=static_cast<unsigned>(index%groups);
    const auto block=iq?(static_cast<std::uint64_t>(column)*(k/256U)+group/8U)*136U:
        (static_cast<std::uint64_t>(column)*groups+group)*18U;
    const auto* source=checkpoint+block;
    const auto scale_bits=static_cast<std::uint16_t>(static_cast<unsigned>(source[0])|(static_cast<unsigned>(source[1])<<8U));
    double scale=__longlong_as_double(static_cast<long long>(binary64_widen_half_bits(scale_bits)));
    unsigned offset=2;
    if(iq){const auto subgroup=group%8U;const auto high=static_cast<unsigned>(source[2])|(static_cast<unsigned>(source[3])<<8U);
        const auto code=((source[4+subgroup/2U]>>(4U*(subgroup%2U)))&15U)|(((high>>(2U*subgroup))&3U)<<4U);
        scale*=static_cast<double>(static_cast<int>(code)-32);offset=8+16*subgroup;}
    scales[index]=scale;for(unsigned j=0;j<16;++j)packed[index*16+j]=source[offset+j];
}
__global__ void convert_logits(const double* input,float* staged,unsigned first,unsigned columns,unsigned* status){
    const auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=columns)return;
    const auto bits=binary32_rne_bits(static_cast<std::uint64_t>(__double_as_longlong(input[i])));
    if((bits&0x7f800000U)==0x7f800000U){atomicOr(status,1U);return;}
    staged[static_cast<std::uint64_t>(first)+i]=__uint_as_float(bits);
}
}
std::shared_ptr<HipIntegerRuntimeAllocationOwner> HipIntegerRuntimeAllocationOwner::pin(
    DeviceIdentity d,std::uintptr_t address,std::uint64_t generation,std::shared_ptr<void> lifetime){
    require(address&&generation&&lifetime,"integer runtime native pin requires true allocation lifetime and generation");
    target(d,hipStreamLegacy);auto result=std::shared_ptr<HipIntegerRuntimeAllocationOwner>(new HipIntegerRuntimeAllocationOwner);
    result->binding_=native_allocation(d,address,generation,unique_generation());result->owner_=std::move(lifetime);return result;
}
AllocationBinding HipIntegerRuntimeAllocationOwner::current_binding()const{
    physical_extent(binding_,false);return binding_;
}
void HipIntegerRuntimeAllocationOwner::advance_access(std::uint64_t next){require(next>binding_.access_generation,"integer runtime content epochs must increase");binding_.access_generation=next;}

struct HipIntegerProjectionRuntimeBundle::Impl {
    RuntimePlan plan;
    RuntimeOwners owners;
    hipStream_t stream;
    std::shared_ptr<void> stream_owner;
    HipIntegerProjectionLauncher launcher;
    std::array<std::shared_ptr<Owned>,buffer_count> buffers{};
    std::shared_ptr<Owned> staged_logits,conversion_status;
    RuntimeSnapshot snapshot;
    std::uint64_t next_operation=1,internal_bytes=0;
    unsigned host_status=0;
    bool submitted=false,failed=false;
    Impl(RuntimePlan p,RuntimeOwners o,hipStream_t s,std::shared_ptr<void> token):plan(std::move(p)),owners(std::move(o)),stream(s),
        stream_owner(std::move(token)),launcher(plan.config().device,s,stream_owner){}
    void live(){target(plan.config().device,stream);plan.validate_snapshot(owners,snapshot);
        physical_extent(snapshot.arena,false);physical_extent(snapshot.weights,plan.checkpoint().storage==RuntimeTensorStorage::registered_file_mapped);}
    void allocate(){
        const auto peak=plan.tile(0,1);internal_bytes=0;
        for(std::size_t i=0;i<buffer_count;++i){buffers[i]=std::make_shared<Owned>(plan.config().device,peak.accounting().buffers[i].bytes,stream);
            internal_bytes=add(internal_bytes,buffers[i]->binding.bytes);}
        staged_logits=std::make_shared<Owned>(plan.config().device,plan.accounting().staged_logits_bytes,stream);
        conversion_status=std::make_shared<Owned>(plan.config().device,4,stream);
        internal_bytes=add(internal_bytes,add(staged_logits->binding.bytes,conversion_status->binding.bytes));
        require(internal_bytes<=plan.config().maximum_internal_bytes,"integer runtime actual HIP internal allocations exceed admission ceiling");
        const auto retained=add(internal_bytes,add(snapshot.arena.bytes,snapshot.weights.bytes));
        require(retained<=plan.config().maximum_retained_bytes,"integer runtime actual whole retained allocations exceed admission ceiling");
        // Native allocations cannot alias either externally pinned source pool.
        for(const auto& owned:buffers)for(const auto& external:{snapshot.arena,snapshot.weights})
            require(owned->binding.identity!=external.identity&&!(owned->binding.base<external.base+external.bytes&&external.base<owned->binding.base+owned->binding.bytes),
                "integer runtime internal allocation aliases external source/destination");
    }
    void retire(){buffers={};staged_logits.reset();conversion_status.reset();submitted=false;}
};
HipIntegerProjectionRuntimeBundle::HipIntegerProjectionRuntimeBundle(RuntimePlan plan,RuntimeOwners owners,hipStream_t stream,std::shared_ptr<void> token)
    :impl_(std::make_unique<Impl>(std::move(plan),std::move(owners),stream,std::move(token))){}
HipIntegerProjectionRuntimeBundle::~HipIntegerProjectionRuntimeBundle(){if(!drain())static_cast<void>(impl_.release());}
bool HipIntegerProjectionRuntimeBundle::failed()const noexcept{return impl_->failed;}
bool HipIntegerProjectionRuntimeBundle::drain()noexcept{
    if(!impl_->submitted){impl_->retire();return true;}
    try{impl_->live();if(hipStreamQuery(native_stream(impl_->stream))!=hipSuccess||!impl_->launcher.drain())return false;impl_->live();impl_->retire();return true;}catch(...){return false;}
}
void HipIntegerProjectionRuntimeBundle::register_for_compiled_execution(const CompiledSerialExecution& compiled,ExecutionLauncherRegistry& registry){
    require(compiled.execution_identity==impl_->plan.execution_identity()&&impl_->plan.step_id()<compiled.bindings.bindings.size()&&
        compiled.bindings.bindings[impl_->plan.step_id()].kernel_id==runtime_kernel_id(impl_->plan.config()),"integer runtime launcher cannot attach to another compiled plan");
    registry.add_kernel_launcher({runtime_kernel_id(impl_->plan.config()),impl_->plan.implementation_identity(),&dispatch,this});
}
void HipIntegerProjectionRuntimeBundle::dispatch(const ExecutionDispatchContext& context,void* data){
    require(data,"integer runtime dispatch owner missing");auto outcome=static_cast<HipIntegerProjectionRuntimeBundle*>(data)->execute(context);
    if(!outcome.output_committed)throw std::runtime_error("experimental integer LM-head failed: "+outcome.reason);
}
RuntimeOutcome HipIntegerProjectionRuntimeBundle::execute(const ExecutionDispatchContext& context){
    RuntimeOutcome result;result.numerical_identity=impl_->plan.config().approved_numerical_identity;result.runtime_identity=impl_->plan.identity();
    if(impl_->failed||impl_->submitted){result.reason="integer runtime owner is failed or retains unproved work";return result;}
    bool commit_started=false;
    try{
        const auto* scope=current_hip_launch_context();
        require(scope?(!scope->graph_capture&&scope->stream==impl_->stream):(impl_->stream==hipStreamLegacy),
            "integer runtime requires matching eager stream scope or explicit legacy stream");
        hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;hip_check(hipStreamIsCapturing(native_stream(impl_->stream),&capture),"integer runtime capture query");
        require(capture==hipStreamCaptureStatusNone,"blocking integer runtime cannot execute during graph capture");
        impl_->snapshot=impl_->plan.validate_dispatch(context,impl_->owners);impl_->live();impl_->allocate();impl_->live();
        require(impl_->next_operation&&impl_->plan.accounting().tile_count<=UINT64_MAX-impl_->next_operation,"integer runtime operation generations exhausted");
        impl_->submitted=true; // Before any attempted launch or queued source read.
        const auto& config=impl_->plan.config();
        hipLaunchKernelGGL(widen_input,dim3(config.hidden_size/128U+static_cast<unsigned>(config.hidden_size%128U!=0)),dim3(128),0,native_stream(impl_->stream),
            reinterpret_cast<const void*>(impl_->plan.input().address),pointer<double>(impl_->buffers[index(BufferRole::input_f64)]),config.hidden_size,config.input==RuntimeInput::bf16);
        hip_check(hipGetLastError(),"integer runtime source widening");
        hip_check(hipMemsetAsync(pointer<unsigned>(impl_->conversion_status),0,4,native_stream(impl_->stream)),"integer runtime conversion status clear");
        for(std::uint32_t first=0;first<config.vocabulary_size;){
            const auto tile=impl_->plan.tile(first,impl_->next_operation++);const auto columns=static_cast<unsigned>(tile.request().shape.columns);
            impl_->live();HipProjectionInputs inputs;
            for(std::size_t i=0;i<buffer_count;++i){impl_->buffers[i]->binding.access_generation=tile.request().operation_generation;
                inputs.pins[i]=impl_->buffers[i];inputs.buffers[i]={impl_->buffers[i]->binding,0,impl_->buffers[i]->binding.bytes};}
            const auto groups=static_cast<std::uint64_t>(columns)*(config.hidden_size/32U);
            hipLaunchKernelGGL(repack_weights,dim3(static_cast<unsigned>((groups+127U)/128U)),dim3(128),0,native_stream(impl_->stream),
                reinterpret_cast<const unsigned char*>(impl_->plan.checkpoint().address),pointer<unsigned char>(impl_->buffers[index(BufferRole::packed_weights)]),
                pointer<double>(impl_->buffers[index(BufferRole::weight_scales_f64)]),config.hidden_size,first,columns,config.weights==WeightEncoding::iq4_xs);
            hip_check(hipGetLastError(),"integer runtime original checkpoint repack");
            inputs.producer_ready=HipProjectionReadyEvent::record(config.device,impl_->stream,inputs.pins,impl_->stream_owner);
            const auto outcome=impl_->launcher.execute(tile,inputs);
            require(outcome.status==HipProjectionStatus::complete,"integer runtime shared projection rejected tile");
            validate_completed_output(tile,inputs.buffers,outcome.evidence);impl_->live();
            hipLaunchKernelGGL(convert_logits,dim3((columns+127U)/128U),dim3(128),0,native_stream(impl_->stream),
                pointer<const double>(impl_->buffers[index(BufferRole::destination_f64)]),pointer<float>(impl_->staged_logits),first,columns,pointer<unsigned>(impl_->conversion_status));
            hip_check(hipGetLastError(),"integer runtime explicit FP32 conversion");
            hip_check(hipMemcpyAsync(&impl_->host_status,pointer<unsigned>(impl_->conversion_status),4,hipMemcpyDeviceToHost,native_stream(impl_->stream)),"integer runtime conversion status readback");
            hip_check(hipStreamSynchronize(native_stream(impl_->stream)),"integer runtime tile conversion completion");impl_->live();
            require(impl_->host_status==0,"integer runtime logits cannot be represented as finite FP32");
            ++result.completed_tiles;first+=columns;
        }
        (void)impl_->plan.validate_dispatch(context,impl_->owners);impl_->live();commit_started=true;
        hip_check(hipMemcpyAsync(reinterpret_cast<void*>(impl_->plan.output().address),pointer<const float>(impl_->staged_logits),
            static_cast<std::size_t>(impl_->plan.accounting().staged_logits_bytes),hipMemcpyDeviceToDevice,native_stream(impl_->stream)),"integer runtime whole logits commit");
        hip_check(hipStreamSynchronize(native_stream(impl_->stream)),"integer runtime logits commit completion");impl_->live();
        (void)impl_->plan.validate_dispatch(context,impl_->owners);
        result.actual_internal_bytes=impl_->internal_bytes;result.actual_retained_bytes=add(impl_->internal_bytes,add(impl_->snapshot.arena.bytes,impl_->snapshot.weights.bytes));
        result.output_committed=true;result.stream_quiescent=true;impl_->retire();return result;
    }catch(const std::exception& e){result.reason=e.what();}
    catch(...){result.reason="integer runtime source/resource provider threw unknown exception";}
    if(impl_->submitted)impl_->failed=true;
    result.destination_poisoned=commit_started;result.actual_internal_bytes=impl_->internal_bytes;
    result.stream_quiescent=drain();return result;
}
} // namespace strix::integer_projection
