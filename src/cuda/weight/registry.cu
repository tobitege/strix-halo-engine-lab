#include "registry.hpp"
#include "strix/cuda/qwen_transcode.hpp"
#include "strix/weight/registry.hpp"
#include <array>
#include <cstdint>

namespace strix::cuda::weight {
namespace {
namespace w = ::strix::weight;
w::Status device_status(cudaError_t error) {
    return error == cudaSuccess ? w::Status{w::Error::ok,{}} : w::Status{w::Error::device_failure,cudaGetErrorString(error)};
}
w::Status dense(const w::Plan& plan, const w::WeightView& view, const w::KernelArguments& record,
    void* output, void*, std::size_t, w::ExecutionContext& context) {
    if (plan.selected.storage != w::FormatId{"gguf.q8_0",1} ||
        plan.selected.arithmetic != w::ArithmeticId{"cuda-qwen-imma-four-chain",1})
        return {w::Error::unsupported_arithmetic,"dense source requires its incumbent Q8 arithmetic"};
    const auto& a = *static_cast<const DenseProjectArguments*>(record.data);
    if (view.shape.rank != 2 || a.site.width != view.shape.dims[1] || !a.site.segments ||
        a.site.segments > qwen_dense::kMaxSegments || !a.rows || a.pitch < a.rows || !a.codes || !a.deltas)
        return {w::Error::unsupported_shape,"dense launch/source shape mismatch"};
    if (!a.sources[0] || a.sources[0]->tensor != view.tensor || a.site.segment[0].output != output)
        return {w::Error::malformed,"dense output/source envelope differs"};
    for (unsigned i = 0; i < a.site.segments; ++i) {
        const auto* source = a.sources[i];
        const auto& segment = a.site.segment[i];
        if (!source || source->storage != w::FormatId{"gguf.q8_0",1} || source->shape.rank != 2 ||
            source->shape.dims[0] != segment.outputs || source->shape.dims[1] != a.site.width ||
            !segment.weight || !segment.output || segment.output_stride < segment.outputs)
            return {w::Error::unsupported_shape,"dense segment/source shape differs"};
    }
    return device_status(qwen_dense::project(static_cast<cudaStream_t>(context.stream),a.site,a.rows,a.codes,a.deltas,a.pitch));
}
w::Status launch_transcode(const w::Plan& plan, const w::WeightView& view, const w::KernelArguments& record,
    void* output, void*, std::size_t, w::ExecutionContext& context) {
    const auto& a = *static_cast<const TranscodeArguments*>(record.data);
    const auto type = w::gguf_type(view.storage);
    if (!a.source || !output || !a.elements || a.elements % 32 || !transcode::supported(type))
        return {w::Error::unsupported_shape,"transcode source or group geometry unsupported"};
    std::uint64_t count = 1;
    if (!view.shape.rank || view.shape.rank > view.shape.dims.size())
        return {w::Error::unsupported_shape,"transcode requires a ranked source"};
    for (unsigned i = 0; i < view.shape.rank; ++i) {
        if (!view.shape.dims[i] || count > UINT64_MAX / view.shape.dims[i])
            return {w::Error::malformed,"transcode source extent overflow"};
        count *= view.shape.dims[i];
    }
    if (a.elements != count)
        return {w::Error::unsupported_shape,"transcode element count differs from source"};
    const w::ImageId expected_image = record.schema == transcode_f32_schema
        ? w::ImageId{"legacy.qwen.cuda.decode-f32",1}
        : w::ImageId{"legacy.qwen.cuda.amax127-f16-roundf",1};
    if (plan.selected.image != expected_image)
        return {w::Error::unsupported_arithmetic,"converted image recipe differs"};
    if (plan.selected.strategy != w::Strategy::converted)
        return {w::Error::unsupported_arithmetic,"transcode requires an explicit converted image recipe"};
    auto stream = static_cast<cudaStream_t>(context.stream);
    if (record.schema == transcode_f32_schema) transcode::to_f32(a.source,type,a.elements,static_cast<float*>(output),stream);
    else transcode::to_q8_0(a.source,type,a.elements,output,stream);
    return device_status(cudaPeekAtLastError());
}
w::Status launch_nvfp4(const w::Plan& plan,const w::WeightView& view,const w::KernelArguments& record,
    void* output,void* workspace,std::size_t bytes,w::ExecutionContext& context) {
    if(plan.selected.storage!=w::FormatId{"nvidia.nvfp4",1} ||
       plan.selected.arithmetic!=w::ArithmeticId{"nvfp4-mma-f32-v1",1} ||
       plan.selected.strategy!=w::Strategy::native || plan.selected.image!=native_nvfp4_image ||
       plan.selected.exactness!=w::Exactness::fast)
        return {w::Error::unsupported_arithmetic,"native FP4 requires explicit native MMA FAST contract/image"};
    const auto& a=*static_cast<const Nvfp4Arguments*>(record.data);
    if(a.view.output!=output || view.shape.rank!=2 || a.view.n!=view.shape.dims[0] ||
       a.view.k!=view.shape.dims[1] || a.plan.workspace_bytes!=plan.workspace_bytes)
        return {w::Error::unsupported_shape,"native FP4 typed source/output/workspace differs from plan"};
    LaunchShape shape{};shape.m=a.view.m;shape.n=a.view.n;shape.k=a.view.k;
    auto status=NativeNvfp4Source::validate(view,shape);if(!status)return status;
    // a.view.b is made from the admitted weight's UploadedSlices; its image
    // lease and the activation/output buffers outlive stream completion.
    return enqueue_nvfp4(a,workspace,bytes,context);
}
constexpr std::array registrations{
    Registration{dense_project_schema,1,sizeof(DenseProjectArguments),alignof(DenseProjectArguments),dense},
    Registration{transcode_q8_schema,1,sizeof(TranscodeArguments),alignof(TranscodeArguments),launch_transcode},
    Registration{transcode_f32_schema,1,sizeof(TranscodeArguments),alignof(TranscodeArguments),launch_transcode},
    Registration{nvfp4_schema,1,sizeof(Nvfp4Arguments),alignof(Nvfp4Arguments),launch_nvfp4},
};
}
std::span<const Registration> source_registry() noexcept { return registrations; }
::strix::weight::Status enqueue_registered(const w::Plan& plan, const w::WeightView& view,
    const w::KernelArguments& record, void* output, void* workspace, std::size_t bytes, w::ExecutionContext& context) {
    if (context.backend != w::Backend::cuda || plan.selected.backend != w::Backend::cuda)
        return {w::Error::unsupported_backend,"CUDA source launcher requires CUDA context"};
    if (view.storage != plan.selected.storage)
        return {w::Error::unknown_format,"source and plan storage differ"};
    if (bytes < plan.workspace_bytes || (plan.workspace_bytes && !workspace))
        return {w::Error::memory_budget,"source workspace missing or short"};
    if (plan.selected.kernel != record.schema)
        return {w::Error::unsupported_arithmetic,"plan kernel and typed launch schema differ"};
    for (const auto& entry : registrations) if (entry.schema == record.schema) {
        if (record.abi_revision != entry.revision || record.bytes != entry.bytes || !record.data ||
            reinterpret_cast<std::uintptr_t>(record.data) % entry.alignment || !output)
            return {w::Error::malformed,"invalid CUDA typed launch ABI"};
        int current_device = -1;
        if (const auto status = cudaGetDevice(&current_device); status != cudaSuccess)
            return device_status(status);
        if (current_device < 0 || static_cast<std::uint32_t>(current_device) != context.device_index)
            return {w::Error::device_failure,"CUDA context belongs to a different current device"};
        return entry.launch(plan,view,record,output,workspace,bytes,context);
    }
    return {w::Error::unsupported_arithmetic,"no CUDA source registration for typed launch"};
}
::strix::weight::RuntimeAdapter with_source_registry(const w::RuntimeAdapter& api) {
    auto result = api;
    auto callbacks = api.callbacks ? std::make_shared<w::RuntimeCallbacks>(*api.callbacks) : std::make_shared<w::RuntimeCallbacks>();
    callbacks->enqueue = enqueue_registered;
    result.callbacks = std::move(callbacks);
    return result;
}
}
