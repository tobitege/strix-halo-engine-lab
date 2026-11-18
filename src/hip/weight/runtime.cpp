#include "runtime.hpp"
namespace strix::hip::weight {
strix::weight::Status bind_runtime(const strix::weight::RuntimeAdapter& api, void* stream,
    std::uint32_t device, strix::weight::ExecutionContext& context, strix::weight::Hardware& hardware) {
    return strix::weight::bind_runtime(strix::weight::Backend::hip, api, stream, device, context, hardware);
}
}

#if defined(STRIX_WFMT_HIP_RUNTIME)
#include <hip/hip_runtime.h>
#include "../../weight/native_runtime.hpp"
#include <utility>
namespace strix::hip::weight {
namespace {
struct Api {
    using Error=hipError_t; using Properties=hipDeviceProp_t; using Event=hipEvent_t; using Stream=hipStream_t;
    static constexpr auto backend=strix::weight::Backend::hip;
    static constexpr auto Success=hipSuccess;
    static constexpr auto OutOfMemory=hipErrorOutOfMemory;
    static constexpr auto EventDisableTiming=hipEventDisableTiming;
    static constexpr auto MemcpyHostToDevice=hipMemcpyHostToDevice;
    template<class... Args> static auto SetDevice(Args&&... args) { return hipSetDevice(std::forward<Args>(args)...); }
    template<class... Args> static auto GetDeviceProperties(Args&&... args) { return hipGetDeviceProperties(std::forward<Args>(args)...); }
    template<class... Args> static auto MemGetInfo(Args&&... args) { return hipMemGetInfo(std::forward<Args>(args)...); }
    template<class... Args> static auto EventQuery(Args&&... args) { return hipEventQuery(std::forward<Args>(args)...); }
    template<class... Args> static auto EventDestroy(Args&&... args) { return hipEventDestroy(std::forward<Args>(args)...); }
    template<class... Args> static auto Free(Args&&... args) { return hipFree(std::forward<Args>(args)...); }
    template<class... Args> static auto GetErrorString(Args&&... args) { return hipGetErrorString(std::forward<Args>(args)...); }
    template<class... Args> static auto Malloc(Args&&... args) { return hipMalloc(std::forward<Args>(args)...); }
    template<class... Args> static auto EventCreateWithFlags(Args&&... args) { return hipEventCreateWithFlags(std::forward<Args>(args)...); }
    template<class... Args> static auto EventRecord(Args&&... args) { return hipEventRecord(std::forward<Args>(args)...); }
    template<class... Args> static auto StreamWaitEvent(Args&&... args) { return hipStreamWaitEvent(std::forward<Args>(args)...); }
    template<class... Args> static auto Memcpy(Args&&... args) { return hipMemcpy(std::forward<Args>(args)...); }
    template<class... Args> static auto StreamQuery(Args&&... args) { return hipStreamQuery(std::forward<Args>(args)...); }
    template<class... Args> static auto StreamSynchronize(Args&&... args) { return hipStreamSynchronize(std::forward<Args>(args)...); }
    template<class... Args> static auto EventSynchronize(Args&&... args) { return hipEventSynchronize(std::forward<Args>(args)...); }
    static std::string arch(const Properties& p) { return p.gcnArchName; }
};
}
void shutdown_native_runtime() { strix::weight::detail::NativeRuntime<Api>::shutdown(); }
strix::weight::Status bind_native_runtime(void* stream,std::uint32_t device,strix::weight::ExecutionContext& context,strix::weight::Hardware& hardware) {
    return strix::weight::detail::NativeRuntime<Api>::bind(stream,device,context,hardware);
}
}
#else
namespace strix::hip::weight {
void shutdown_native_runtime() {}
strix::weight::Status bind_native_runtime(void*,std::uint32_t,strix::weight::ExecutionContext&,strix::weight::Hardware&) {
    return {strix::weight::Error::unsupported_backend,"native runtime is not compiled; no implicit CPU offload"};
}
}
#endif
