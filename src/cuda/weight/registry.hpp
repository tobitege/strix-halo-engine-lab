#pragma once
#include "strix/weight/cache.hpp"
#include "strix/weight/runtime.hpp"
#include "strix/cuda/qwen_dense.hpp"
#include "nvfp4_source.cuh"
#include <span>
#include <type_traits>

namespace strix::cuda::weight {
// Build-local IDs describe complete, typed launch records. They are never pack IDs.
inline constexpr ::strix::weight::KernelId dense_project_schema = 0x435701;
inline constexpr ::strix::weight::KernelId transcode_q8_schema = 0x435702;
inline constexpr ::strix::weight::KernelId transcode_f32_schema = 0x435703;
inline constexpr ::strix::weight::KernelId nvfp4_schema = 0x435704;
inline constexpr ::strix::weight::ImageId native_nvfp4_image{"cuda.nvfp4.native",1};
struct DenseProjectArguments {
    qwen_dense::Site site;
    std::uint32_t rows;
    const std::int8_t* codes;
    const float* deltas;
    std::uint32_t pitch;
    const ::strix::weight::WeightView* sources[qwen_dense::kMaxSegments];
};
struct TranscodeArguments {
    const void* source;
    std::uint64_t elements;
};
static_assert(std::is_standard_layout_v<DenseProjectArguments> && std::is_trivially_copyable_v<DenseProjectArguments>);
static_assert(std::is_standard_layout_v<TranscodeArguments> && std::is_trivially_copyable_v<TranscodeArguments>);
static_assert(sizeof(void*) == 8, "CUDA source launch ABI requires 64-bit pointers");
static_assert(sizeof(qwen_dense::Site) == 192 && sizeof(DenseProjectArguments) == 264);
static_assert(sizeof(TranscodeArguments) == 16);
static_assert(alignof(DenseProjectArguments) == alignof(void*));
static_assert(alignof(TranscodeArguments) == alignof(void*));

inline ::strix::weight::KernelArguments arguments(const DenseProjectArguments& a) {
    return {dense_project_schema, 1, &a, sizeof(a)};
}
void arguments(DenseProjectArguments&&) = delete;
void arguments(const DenseProjectArguments&&) = delete;
inline ::strix::weight::KernelArguments transcode_arguments(const TranscodeArguments& a, bool f32) {
    return {f32 ? transcode_f32_schema : transcode_q8_schema, 1, &a, sizeof(a)};
}
void transcode_arguments(TranscodeArguments&&, bool) = delete;

inline ::strix::weight::KernelArguments arguments(const Nvfp4Arguments& a) {
    return {nvfp4_schema,1,&a,sizeof(a)};
}
void arguments(Nvfp4Arguments&&) = delete;
void arguments(const Nvfp4Arguments&&) = delete;
void transcode_arguments(const TranscodeArguments&&, bool) = delete;

using Launch = ::strix::weight::Status (*)(const ::strix::weight::Plan&, const ::strix::weight::WeightView&,
    const ::strix::weight::KernelArguments&, void*, void*, std::size_t, ::strix::weight::ExecutionContext&);
struct Registration {
    ::strix::weight::KernelId schema;
    std::uint32_t revision;
    std::size_t bytes, alignment;
    Launch launch;
};
std::span<const Registration> source_registry() noexcept;
::strix::weight::Status enqueue_registered(const ::strix::weight::Plan&, const ::strix::weight::WeightView&,
    const ::strix::weight::KernelArguments&, void*, void*, std::size_t, ::strix::weight::ExecutionContext&);
// Adds immutable launch dispatch to the vendor's existing lifecycle callbacks.
::strix::weight::RuntimeAdapter with_source_registry(const ::strix::weight::RuntimeAdapter&);
}
