#include "strix/hip_check.hpp"
#include "strix/hip_arch_caps.hpp"
#include "strix/hip_integer_projection.hpp"
#include <hip/hip_runtime.h>
#include <array>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <utility>

namespace strix::integer_projection {
namespace {
using int2=int __attribute__((ext_vector_type(2)));
using int4=int __attribute__((ext_vector_type(4)));
using int8=int __attribute__((ext_vector_type(8)));
__device__ __constant__ int book[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
using strix::hip_check;
void require(bool ok,const char* reason){if(!ok)throw std::invalid_argument(reason);}
// One thread per actual row/group. Canonical sequential absmax and exact
// binary64 division; no cross-row statistics, reciprocal, fast math or atomics.
__global__ void prepare_kernel(const double* input,std::uint8_t* codes,double* scales,
    std::int32_t* sums,unsigned* error,std::uint32_t rows,std::uint32_t k,bool a4) {
    const std::uint64_t group=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(group>=static_cast<std::uint64_t>(rows)*(k/32U))return;
    const auto start=group*32U;
    double maximum=0;
    for(unsigned j=0;j<32;++j){const double x=input[start+j];
        if(!isfinite(x)){atomicOr(error,1U);return;}maximum=fmax(maximum,fabs(x));}
    const int qmax=a4?7:127;
    const double scale=maximum==0?1.0:maximum/static_cast<double>(qmax);
    if(!isfinite(scale)||scale<=0){atomicOr(error,2U);return;}
    int sum=0;
    for(unsigned j=0;j<32;++j) {
        const double x=input[start+j]/scale;
        const double lower=floor(x),fraction=x-lower;
        int q=static_cast<int>(lower);
        if(fraction>0.5||(fraction==0.5&&q%2!=0))++q;
        q=q>qmax?qmax:(q< -qmax?-qmax:q);
        codes[start+j]=static_cast<std::uint8_t>(static_cast<unsigned>(q)&(a4?15U:255U));sum+=q;
    }
    scales[group]=scale;sums[group]=sum;
}
template<bool Four,bool IQ,bool A4>
__global__ void project_kernel(const std::uint8_t* activations,const double* activation_scales,
    const std::int32_t* sums,const std::uint8_t* weights,const double* weight_scales,
    double* output,std::int32_t* integer_groups,unsigned* error,
    std::uint32_t rows,std::uint32_t columns,std::uint32_t k) {
#if defined(__HIP_DEVICE_COMPILE__) && !STRIX_HAS_RDNA35_WMMA
#error This experimental fragment ABI requires RDNA 3.5 IU4/IU8 WMMA builtins.
#endif
    const unsigned lane=threadIdx.x; // exactly one Wave32 per workgroup
    const std::uint32_t input_row=blockIdx.y*16U+lane%16U;
    const std::uint32_t column=blockIdx.x*16U+lane%16U;
    const unsigned groups=k/32U;
    double result[8]={};
    for(unsigned g=0;g<groups;++g) {
        int8 accumulator{};
        for(unsigned panel=0;panel<2;++panel) {
            std::uint32_t aw[4]={},bw[4]={};
            constexpr unsigned bits=Four?4U:8U,per=32U/bits,mask=(1U<<bits)-1U;
            for(unsigned j=0;j<16;++j) {
                const auto kk=g*32U+panel*16U+j;
                int av=0,bv=0;
                if(input_row<rows) {
                    const unsigned code=activations[static_cast<std::uint64_t>(input_row)*k+kk];
                    if constexpr(A4)av=code<8U?static_cast<int>(code):static_cast<int>(code)-16;
                    else av=code<128U?static_cast<int>(code):static_cast<int>(code)-256;
                }
                if(column<columns) {
                    const auto packed=weights[(static_cast<std::uint64_t>(column)*groups+g)*16U+j];
                    const unsigned code=panel==0?packed&15U:packed>>4U;
                    if constexpr(IQ)bv=book[code];else bv=static_cast<int>(code);
                } // mask AFTER interpretation: IQ index zero is not padding.
                aw[j/per]|=(static_cast<unsigned>(av)&mask)<<((j%per)*bits);
                bw[j/per]|=(static_cast<unsigned>(bv)&mask)<<((j%per)*bits);
            }
            if constexpr(Four) {
                int2 a{},b{};for(unsigned v=0;v<2;++v){a[v]=__builtin_bit_cast(int,aw[v]);b[v]=__builtin_bit_cast(int,bw[v]);}
                accumulator=__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true,a,false,b,accumulator,false);
            } else {
                int4 a{},b{};for(unsigned v=0;v<4;++v){a[v]=__builtin_bit_cast(int,aw[v]);b[v]=__builtin_bit_cast(int,bw[v]);}
                accumulator=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a,IQ,b,accumulator,false);
            }
        }
        // Accumulator VGPR v maps row 2*v+lane/16, column lane%16.
        // Every group is bounded by 32*127*127=516128. No saturation.
        for(unsigned v=0;v<8;++v) {
            const auto row=blockIdx.y*16U+2U*v+lane/16U;
            if(row>=rows||column>=columns)continue;
            std::int64_t corrected=accumulator[v];
            if constexpr(!IQ)corrected-=8LL*sums[static_cast<std::uint64_t>(row)*groups+g];
            if(corrected< -516128LL||corrected>516128LL){atomicOr(error,4U);continue;}
            const auto index=static_cast<std::uint64_t>(row)*columns+column;
            integer_groups[index*groups+g]=static_cast<std::int32_t>(corrected);
            const double scale=weight_scales[static_cast<std::uint64_t>(column)*groups+g]*
                activation_scales[static_cast<std::uint64_t>(row)*groups+g];
            result[v]+=scale*static_cast<double>(corrected);
            if(!isfinite(result[v]))atomicOr(error,8U);
        }
    }
    for(unsigned v=0;v<8;++v){const auto row=blockIdx.y*16U+2U*v+lane/16U;
        if(row<rows&&column<columns)output[static_cast<std::uint64_t>(row)*columns+column]=result[v];}
}
__global__ void validate_weight_scales(const double* scales,std::uint64_t count,unsigned* status) {
    const auto i=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i==0){status[1]=static_cast<unsigned>(warpSize);if(warpSize!=32)atomicOr(status,32U);}
    if(i<count&&!isfinite(scales[i]))atomicOr(status,16U);
}
template<class T> T* pointer(const ExecutionBindings& b,BufferRole role) {
    return reinterpret_cast<T*>(buffer_address(b[index(role)]));
}
void target(const DeviceIdentity& device,hipStream_t stream) {
    const auto* override_arch=std::getenv("HSA_OVERRIDE_GFX_VERSION");
    require(!override_arch||!*override_arch,"architecture override prevents gfx1151 attribution");
    int active=-1;hip_check(hipGetDevice(&active),"projection active device");
    require(active==device.index,"projection active HIP device changed");
    // The legacy/default stream is a current-device sentinel, not an object.
    hipDevice_t stream_device=active;
    if(stream!=nullptr&&stream!=hipStreamLegacy)
        hip_check(hipStreamGetDevice(stream,&stream_device),"projection stream device");
    require(stream_device==device.index,"projection stream belongs to another device");
    hipDeviceProp_t properties{};hip_check(hipGetDeviceProperties(&properties,device.index),"projection device properties");
    const std::string architecture=properties.gcnArchName;
    require(device.architecture=="gfx1151"&&architecture.substr(0,architecture.find(':'))==device.architecture,
        "projection requires exact gfx1151");
}
hipStream_t native_stream(hipStream_t stream) noexcept {
    return stream==hipStreamLegacy?nullptr:stream;
}
std::uint64_t native_extents(const ExecutionPlan& p,const ExecutionBindings& b) {
    std::uint64_t actual=0;
    for(std::size_t i=0;i<buffer_count;++i) {
        const auto& allocation=b[i].allocation;
        bool seen=false;for(std::size_t j=0;j<i;++j)if(b[j].allocation.identity==allocation.identity)seen=true;
        if(seen)continue;
        hipPointerAttribute_t attributes{};
        hip_check(hipPointerGetAttributes(&attributes,reinterpret_cast<const void*>(allocation.base)),"projection pointer attributes");
        require(attributes.type==hipMemoryTypeDevice&&!attributes.isManaged&&attributes.device==p.request().device.index,
            "projection requires device-local HIP allocation on the bound device");
        hipDeviceptr_t base=nullptr;std::size_t bytes=0;
        hip_check(hipMemGetAddressRange(&base,&bytes,reinterpret_cast<hipDeviceptr_t>(allocation.base)),"projection native allocation extent");
        require(reinterpret_cast<std::uintptr_t>(base)==allocation.base&&bytes>=allocation.bytes,
            "projection declared allocation is not pinned within its native extent");
        require(bytes<=p.request().maximum_device_bytes&&actual<=p.request().maximum_device_bytes-bytes,
            "projection actual HIP allocations exceed admitted full working set");
        actual+=bytes;
    }
    return actual;
}
} // namespace

struct HipProjectionReadyEvent::Impl {
    DeviceIdentity device;
    hipStream_t stream=nullptr;
    std::shared_ptr<void> stream_owner;
    AllocationPins producer_pins;
    AllocationSnapshots recorded_bindings;
    hipEvent_t event=nullptr;
    bool recorded=false;
};
std::shared_ptr<const HipProjectionReadyEvent> HipProjectionReadyEvent::record(
    DeviceIdentity device,hipStream_t stream,AllocationPins pins,std::shared_ptr<void> owner) {
    auto event=std::shared_ptr<HipProjectionReadyEvent>(new HipProjectionReadyEvent);
    event->impl_=std::make_unique<Impl>();event->impl_->device=std::move(device);event->impl_->stream=stream;
    event->impl_->stream_owner=std::move(owner);event->impl_->producer_pins=std::move(pins);
    const auto& recorded_device=event->impl_->device;
    require(recorded_device.index>=0&&recorded_device.instance_generation&&recorded_device.architecture=="gfx1151","input producer requires explicit device instance");
    require(stream&&stream!=hipStreamPerThread&&(stream==hipStreamLegacy||event->impl_->stream_owner),"input producer requires explicit owned stream");
    event->impl_->recorded_bindings=snapshot_allocation_pins(event->impl_->producer_pins);
    for(std::size_t i=0;i<buffer_count;++i){
        require(event->impl_->recorded_bindings[i].device==recorded_device,"input ready event requires exact-device allocation pins");}
    target(recorded_device,stream);
    hip_check(hipEventCreateWithFlags(&event->impl_->event,hipEventDisableTiming),"projection producer event create");
    validate_allocation_snapshots(event->impl_->producer_pins,event->impl_->recorded_bindings);
    hip_check(hipEventRecord(event->impl_->event,native_stream(stream)),"projection producer event record");event->impl_->recorded=true;
    validate_allocation_snapshots(event->impl_->producer_pins,event->impl_->recorded_bindings);
    return event;
}
HipProjectionReadyEvent::~HipProjectionReadyEvent() {
    if(!impl_)return;
    try {target(impl_->device,impl_->stream);
        require(impl_->device.instance_generation&&impl_->stream&&impl_->stream!=hipStreamPerThread&&
            (impl_->stream==hipStreamLegacy||impl_->stream_owner),"unproved producer stream ownership");
        const auto completed=impl_->recorded?hipEventQuery(impl_->event):hipStreamQuery(native_stream(impl_->stream));
        if(completed==hipSuccess&&(!impl_->event||hipEventDestroy(impl_->event)==hipSuccess))return;
    }catch(...){}
    static_cast<void>(impl_.release()); // Event/stream owner survives unproved producer completion.
}
const DeviceIdentity& HipProjectionReadyEvent::device() const noexcept{return impl_->device;}
hipEvent_t HipProjectionReadyEvent::native_handle() const noexcept{return impl_->event;}
bool HipProjectionReadyEvent::matches_pins(const AllocationPins& pins) const noexcept {
    try {for(std::size_t i=0;i<buffer_count;++i)
        if(pins[i].get()!=impl_->producer_pins[i].get()||pins[i].owner_before(impl_->producer_pins[i])||
            impl_->producer_pins[i].owner_before(pins[i])||!pins[i]||pins[i]->current_binding()!=impl_->recorded_bindings[i])return false;
        validate_allocation_snapshots(pins,impl_->recorded_bindings);return true;
    }catch(...){return false;}
}

struct HipIntegerProjectionLauncher::Impl {
    struct Retained { ExecutionPlan plan; HipProjectionInputs inputs; };
    DeviceIdentity device;
    hipStream_t stream=nullptr;
    std::shared_ptr<void> stream_owner;
    std::optional<Retained> retained;
    std::array<unsigned,2> host_status{}; // Survives failed asynchronous readback.
    bool failed=false;
    std::uint64_t last_operation_generation=0;
    void live() {
        target(device,stream);
        if(retained){validate_execution_pins(retained->plan,retained->inputs.buffers,retained->inputs.pins);
            (void)native_extents(retained->plan,retained->inputs.buffers);}
    }
    void status() {
        hip_check(hipMemcpyAsync(host_status.data(),pointer<unsigned>(retained->inputs.buffers,BufferRole::status_u32),
            sizeof(host_status),hipMemcpyDeviceToHost,native_stream(stream)),"projection status readback");
        hip_check(hipStreamSynchronize(native_stream(stream)),"projection stage completion");
        live();
        require(host_status[1]==32,"projection kernel did not report Wave32");
        require(host_status[0]==0,"projection finite/preparation/integer/output status rejected execution");
    }
};
HipIntegerProjectionLauncher::HipIntegerProjectionLauncher(DeviceIdentity device,hipStream_t stream,std::shared_ptr<void> owner)
    :impl_(std::make_unique<Impl>()) {
    require(device.index>=0&&device.instance_generation&&device.architecture=="gfx1151","explicit projection device instance required");
    require(stream&&stream!=hipStreamPerThread,"explicit legacy or owned HIP stream required");
    require(stream==hipStreamLegacy||owner!=nullptr,"non-default projection stream requires an owning lifetime token");
    target(device,stream);impl_->device=std::move(device);impl_->stream=stream;impl_->stream_owner=std::move(owner);
}
HipIntegerProjectionLauncher::~HipIntegerProjectionLauncher() {
    if(!drain())static_cast<void>(impl_.release()); // Retain bounded pins/event/stream through process teardown.
}
bool HipIntegerProjectionLauncher::failed() const noexcept{return impl_->failed;}
bool HipIntegerProjectionLauncher::retains_resources() const noexcept{return impl_->retained.has_value();}
bool HipIntegerProjectionLauncher::drain() noexcept {
    if(!impl_->retained)return true;
    try {impl_->live();if(hipStreamQuery(native_stream(impl_->stream))!=hipSuccess)return false;impl_->live();
        impl_->retained.reset();return true;}catch(...){return false;}
}
HipProjectionOutcome HipIntegerProjectionLauncher::execute(const ExecutionPlan& p,HipProjectionInputs input) {
    HipProjectionOutcome result;
    result.evidence.execution_identity=p.identity();result.evidence.destination=input.buffers[index(BufferRole::destination_f64)];
    if(impl_->failed||impl_->retained){result.reason="projection launcher is failed or retains an operation";return result;}
    try {
        require(p.request().device==impl_->device,"projection plan belongs to another device instance");
        require(p.request().operation_generation>impl_->last_operation_generation,"projection operation generation is stale or replayed");
        validate_execution_pins(p,input.buffers,input.pins);target(impl_->device,impl_->stream);
        require(input.producer_ready&&input.producer_ready->device()==impl_->device&&input.producer_ready->matches_pins(input.pins),
            "projection requires owned input-ready event for the exact device instance");
        result.actual_device_bytes=native_extents(p,input.buffers);
    }catch(const std::exception& e){result.reason=e.what();return result;}
    catch(...){result.reason="projection contract provider threw an unknown exception";return result;}
    impl_->retained.emplace(Impl::Retained{p,std::move(input)});
    impl_->last_operation_generation=p.request().operation_generation;
    impl_->host_status={};bool commit_started=false;
    try {
        const auto& b=impl_->retained->inputs.buffers;const auto& r=p.request();const auto& a=p.accounting();
        const auto rows=static_cast<std::uint32_t>(r.shape.rows),columns=static_cast<std::uint32_t>(r.shape.columns),k=static_cast<std::uint32_t>(r.shape.inner);
        hip_check(hipStreamWaitEvent(native_stream(impl_->stream),impl_->retained->inputs.producer_ready->native_handle(),0),"projection input producer dependency");
        hip_check(hipMemsetAsync(pointer<unsigned>(b,BufferRole::status_u32),0,8,native_stream(impl_->stream)),"projection status initialization");
        hipLaunchKernelGGL(validate_weight_scales,dim3(a.weight_scale_blocks),dim3(128),0,native_stream(impl_->stream),
            pointer<const double>(b,BufferRole::weight_scales_f64),static_cast<std::uint64_t>(columns)*(k/32U),pointer<unsigned>(b,BufferRole::status_u32));
        hip_check(hipGetLastError(),"projection weight validation launch");
        hipLaunchKernelGGL(prepare_kernel,dim3(a.preparation_blocks),dim3(128),0,native_stream(impl_->stream),
            pointer<const double>(b,BufferRole::input_f64),pointer<std::uint8_t>(b,BufferRole::activation_codes),
            pointer<double>(b,BufferRole::activation_scales_f64),pointer<std::int32_t>(b,BufferRole::activation_sums_i32),
            pointer<unsigned>(b,BufferRole::status_u32),rows,k,r.activations==low_precision::ActivationEncoding::symmetric_a4);
        hip_check(hipGetLastError(),"projection preparation launch");impl_->status();
        const dim3 grid(a.grid_columns,a.grid_rows),threads(32);
#define LAUNCH(Four,IQ,A4) hipLaunchKernelGGL((project_kernel<Four,IQ,A4>),grid,threads,0,native_stream(impl_->stream),pointer<const std::uint8_t>(b,BufferRole::activation_codes),pointer<const double>(b,BufferRole::activation_scales_f64),pointer<const std::int32_t>(b,BufferRole::activation_sums_i32),pointer<const std::uint8_t>(b,BufferRole::packed_weights),pointer<const double>(b,BufferRole::weight_scales_f64),pointer<double>(b,BufferRole::staged_output_f64),pointer<std::int32_t>(b,BufferRole::integer_groups_i32),pointer<unsigned>(b,BufferRole::status_u32),rows,columns,k)
        if(r.instruction==Instruction::iu4){LAUNCH(true,false,true);}
        else if(r.weights==WeightEncoding::iq4_xs){if(r.activations==low_precision::ActivationEncoding::symmetric_a4){LAUNCH(false,true,true);}else{LAUNCH(false,true,false);}}
        else{if(r.activations==low_precision::ActivationEncoding::symmetric_a4){LAUNCH(false,false,true);}else{LAUNCH(false,false,false);}}
#undef LAUNCH
        hip_check(hipGetLastError(),"integer projection launch");impl_->status();
        impl_->live();commit_started=true;
        hip_check(hipMemcpyAsync(pointer<double>(b,BufferRole::destination_f64),pointer<const double>(b,BufferRole::staged_output_f64),
            static_cast<std::size_t>(a.buffers[index(BufferRole::destination_f64)].bytes),hipMemcpyDeviceToDevice,native_stream(impl_->stream)),"projection destination commit");
        hip_check(hipStreamSynchronize(native_stream(impl_->stream)),"projection destination completion");impl_->live();
        result.evidence.stream_quiescent=true;result.evidence.destination_copy_complete=true;
        result.evidence.destination_owner=impl_->retained->inputs.pins[index(BufferRole::destination_f64)];
        validate_completed_output(p,b,result.evidence);impl_->retained.reset();result.status=HipProjectionStatus::complete;return result;
    }catch(const std::exception& e){
        impl_->failed=true;result.reason=e.what();result.evidence.device_status=impl_->host_status[0];
        result.evidence.destination_poisoned=commit_started;
        result.evidence.stream_quiescent=drain();
        result.status=result.evidence.stream_quiescent?HipProjectionStatus::failed:HipProjectionStatus::failed_retained;return result;
    }catch(...){
        impl_->failed=true;result.reason="projection execution provider threw an unknown exception";
        result.evidence.device_status=impl_->host_status[0];result.evidence.destination_poisoned=commit_started;
        result.evidence.stream_quiescent=drain();result.status=result.evidence.stream_quiescent?HipProjectionStatus::failed:HipProjectionStatus::failed_retained;
        return result;
    }
}
} // namespace strix::integer_projection
