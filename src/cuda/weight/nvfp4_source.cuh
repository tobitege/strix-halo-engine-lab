#pragma once
#include "source.cuh"
#include "strix/weight/admission.hpp"
#include "strix/weight/cache.hpp"
#include "strix/weight/registry.hpp"
#include "strix/cuda/core/nvfp4_cuda.hpp"
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

namespace strix::cuda::weight {
struct Nvfp4Tag {};
struct Sm120Nvfp4 {};
struct Nvfp4MmaF32 {};
using Nvfp4Matrix = core::nvfp4::Matrix;
inline constexpr bool supports_native_nvfp4(unsigned major,unsigned minor) noexcept {return major==12 && minor==0;}
static_assert(supports_native_nvfp4(12,0) && !supports_native_nvfp4(12,1) && !supports_native_nvfp4(9,0));

// The native core already consumes this matrix fragment. In particular there
// is no scalar decode fallback or hidden activation conversion in this source.
template<> struct OperandSource<Nvfp4Tag,Nvfp4TileTag,Sm120Nvfp4,Nvfp4MmaF32> {
    using View = Nvfp4Matrix;
    using Fragment = Nvfp4Matrix;
    static constexpr unsigned group_k=16, alignment=4, shared_bytes=0;
    static constexpr bool has_input_transform=false, has_output_transform=false;
    static strix::weight::Status validate(const strix::weight::WeightView& view,const LaunchShape& shape) {
        using namespace strix::weight;
        if(view.storage!=Id{"nvidia.nvfp4",1})return {Error::unknown_format,"native FP4 source storage differs"};
        auto status=validate_metadata(view);if(!status)return status;
        if(view.shape.rank!=2 || !shape.m || !shape.n || !shape.k || shape.k%64 ||
           shape.n!=view.shape.dims[0] || shape.k!=view.shape.dims[1] ||
           shape.m>UINT32_MAX || shape.n>UINT32_MAX || shape.k>UINT32_MAX)
            return {Error::unsupported_shape,"native FP4 requires dense [N,K], K multiple of 64"};
        return {Error::ok,{}};
    }
    template<class Core>
    STRIX_CUDA_WEIGHT_HD static void load_fragment(const View& view,const typename Core::Coord&,
        typename Core::BFragment& fragment,typename Core::Scratch&) {fragment=view;}
};
using NativeNvfp4Source=OperandSource<Nvfp4Tag,Nvfp4TileTag,Sm120Nvfp4,Nvfp4MmaF32>;
static_assert(std::is_trivially_copyable_v<NativeNvfp4Source::View>);

inline strix::weight::Status make_view(const strix::weight::WeightView& view,
    const UploadedSlices& uploaded,const LaunchShape& shape,Nvfp4Matrix& result) {
    using namespace strix::weight;
    auto status=NativeNvfp4Source::validate(view,shape);if(!status)return status;
    if(uploaded.sidecars.size()!=view.sidecars.size() || uploaded.payload.bytes!=view.payload.range.bytes)
        return {Error::malformed,"native FP4 upload slice count/size differs"};
    Nvfp4Matrix matrix{};matrix.codes=static_cast<const std::uint8_t*>(uploaded.payload.data);
    matrix.code_stride=std::size_t(shape.k/2);matrix.scale_stride=std::size_t(shape.k/16);
    bool scales=false,tensor=false;
    for(std::size_t i=0;i<view.sidecars.size();++i) {
        const auto& side=view.sidecars[i];
        if(side.name=="block_scales") {
            if(uploaded.sidecars[i].bytes!=side.data.range.bytes || !uploaded.sidecars[i].data)
                return {Error::malformed,"native FP4 uploaded block scales differ"};
            matrix.scales=static_cast<const std::uint8_t*>(uploaded.sidecars[i].data);scales=true;
        }else if(side.name=="tensor_scale") {
            if(uploaded.sidecars[i].bytes!=sizeof(float) || !uploaded.sidecars[i].data)
                return {Error::malformed,"native FP4 uploaded tensor scale differs"};
            std::array<Byte,sizeof(float)> bytes{};
            status=read_slice(side.data,bytes);if(!status)return status;
            std::memcpy(&matrix.tensor_scale,bytes.data(),sizeof(float));tensor=true;
        }else return {Error::malformed,"native FP4 unknown sidecar"};
    }
    if(!scales || !tensor || !matrix.codes || reinterpret_cast<std::uintptr_t>(matrix.codes)%4 ||
       reinterpret_cast<std::uintptr_t>(matrix.scales)%4 || !std::isfinite(matrix.tensor_scale) || matrix.tensor_scale<=0)
        return {Error::malformed,"native FP4 upload alignment/scale differs"};
    result=matrix;return {Error::ok,{}};
}

struct Nvfp4Arguments {
    core::nvfp4::View view;
    core::nvfp4::Plan plan;
};
static_assert(std::is_standard_layout_v<Nvfp4Arguments> && std::is_trivially_copyable_v<Nvfp4Arguments>);
inline strix::weight::Status enqueue_nvfp4(const Nvfp4Arguments& args,void* workspace,
    std::size_t bytes,strix::weight::ExecutionContext& context) {
    using namespace strix::weight;
    if(context.backend!=Backend::cuda)return {Error::unsupported_backend,"native FP4 source requires CUDA SM120; no qualified CPU, HIP or Intel source"};
    int device=-1;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess || device<0 || std::uint32_t(device)!=context.device_index)
        return {Error::device_failure,"native FP4 execution device differs from context"};
    int major=0,minor=0;
    error=cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device);
    if(error==cudaSuccess)error=cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device);
    if(error!=cudaSuccess)return {Error::device_failure,cudaGetErrorString(error)};
    if(!supports_native_nvfp4(unsigned(major),unsigned(minor)))return {Error::unavailable_isa,"native FP4 source requires SM120"};
    error=core::nvfp4::enqueue(args.view,args.plan,workspace,bytes,static_cast<cudaStream_t>(context.stream));
    if(error==cudaErrorNotSupported)return {Error::unavailable_isa,"native FP4 source requires SM120"};
    if(error!=cudaSuccess)return {Error::device_failure,cudaGetErrorString(error)};
    return {Error::ok,{}};
}
}
