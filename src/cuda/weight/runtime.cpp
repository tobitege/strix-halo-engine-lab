#include "runtime.hpp"
namespace strix::cuda::weight {
strix::weight::Status bind_runtime(const strix::weight::RuntimeAdapter& api, void* stream,
    std::uint32_t device, strix::weight::ExecutionContext& context, strix::weight::Hardware& hardware) {
    return strix::weight::bind_runtime(strix::weight::Backend::cuda, api, stream, device, context, hardware);
}
}

#if defined(STRIX_WFMT_CUDA_RUNTIME)
#include <cuda_runtime_api.h>
#include "../../weight/native_runtime.hpp"
#include <utility>
namespace strix::cuda::weight {
namespace {
struct Api {
    using Error=cudaError_t; using Properties=cudaDeviceProp; using Event=cudaEvent_t; using Stream=cudaStream_t;
    static constexpr auto backend=strix::weight::Backend::cuda;
    static constexpr auto Success=cudaSuccess;
    static constexpr auto OutOfMemory=cudaErrorMemoryAllocation;
    static constexpr auto EventDisableTiming=cudaEventDisableTiming;
    static constexpr auto MemcpyHostToDevice=cudaMemcpyHostToDevice;
    template<class... Args> static auto SetDevice(Args&&... args) { return cudaSetDevice(std::forward<Args>(args)...); }
    template<class... Args> static auto GetDeviceProperties(Args&&... args) { return cudaGetDeviceProperties(std::forward<Args>(args)...); }
    template<class... Args> static auto MemGetInfo(Args&&... args) { return cudaMemGetInfo(std::forward<Args>(args)...); }
    template<class... Args> static auto EventQuery(Args&&... args) { return cudaEventQuery(std::forward<Args>(args)...); }
    template<class... Args> static auto EventDestroy(Args&&... args) { return cudaEventDestroy(std::forward<Args>(args)...); }
    template<class... Args> static auto Free(Args&&... args) { return cudaFree(std::forward<Args>(args)...); }
    template<class... Args> static auto GetErrorString(Args&&... args) { return cudaGetErrorString(std::forward<Args>(args)...); }
    template<class... Args> static auto Malloc(Args&&... args) { return cudaMalloc(std::forward<Args>(args)...); }
    template<class... Args> static auto EventCreateWithFlags(Args&&... args) { return cudaEventCreateWithFlags(std::forward<Args>(args)...); }
    template<class... Args> static auto EventRecord(Args&&... args) { return cudaEventRecord(std::forward<Args>(args)...); }
    template<class... Args> static auto StreamWaitEvent(Args&&... args) { return cudaStreamWaitEvent(std::forward<Args>(args)...); }
    template<class... Args> static auto Memcpy(Args&&... args) { return cudaMemcpy(std::forward<Args>(args)...); }
    template<class... Args> static auto StreamQuery(Args&&... args) { return cudaStreamQuery(std::forward<Args>(args)...); }
    template<class... Args> static auto StreamSynchronize(Args&&... args) { return cudaStreamSynchronize(std::forward<Args>(args)...); }
    template<class... Args> static auto EventSynchronize(Args&&... args) { return cudaEventSynchronize(std::forward<Args>(args)...); }
    static std::string arch(const Properties& p) { return "sm_"+std::to_string(p.major)+std::to_string(p.minor); }
};
}
void shutdown_native_runtime() { strix::weight::detail::NativeRuntime<Api>::shutdown(); }
strix::weight::Status bind_native_runtime(void* stream,std::uint32_t device,strix::weight::ExecutionContext& context,strix::weight::Hardware& hardware) {
    return strix::weight::detail::NativeRuntime<Api>::bind(stream,device,context,hardware);
}
}
#else
namespace strix::cuda::weight {
void shutdown_native_runtime() {}
strix::weight::Status bind_native_runtime(void*,std::uint32_t,strix::weight::ExecutionContext&,strix::weight::Hardware&) {
    return {strix::weight::Error::unsupported_backend,"native runtime is not compiled; no implicit CPU offload"};
}
}
#endif
