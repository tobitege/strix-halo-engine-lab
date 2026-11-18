#pragma once
#include "intel/core/backend.hpp"
#include "strix/weight/host_q8.hpp"
#include "strix/weight/cache.hpp"
namespace strix::weight::intel {
struct IntelAscending {};
template<class Arithmetic> struct Q8F16Source;
template<> struct Q8F16Source<IntelAscending> {
 const std::uint8_t* base;
 std::size_t stride,prefix;
 const std::uint8_t* block(std::size_t output,std::size_t group) const noexcept {
  return Q8Storage::block(base+output*stride+prefix,group);
 }
 static float scale(const std::uint8_t* packed) noexcept { return ::strix::intel::fp16::to_float(Q8Storage::scale_bits(packed)); }
 static float weight(const std::uint8_t* packed,std::size_t lane) noexcept {
  return ::strix::intel::fp16::round(static_cast<float>(Q8Storage::code(packed,lane))*scale(packed));
 }
 static float activation(std::int8_t code,float delta) noexcept {
  return ::strix::intel::fp16::round(static_cast<float>(code)*delta);
 }
};
inline constexpr KernelId q8_f16_ascending_kernel=0x494e5401;
struct DenseArguments { ::strix::intel::dense::Q8A8View view; Q8BindingRef binding; };
std::span<const Candidate> candidates() noexcept;
Status validate_view(const ::strix::intel::dense::Q8A8View&);
Status from_legacy(const ::strix::intel::dense::Q8A8View&,std::span<const Byte>,std::shared_ptr<const void>,TensorId,WeightView&);
// This context is an explicit host oracle, never selected as a model offload fallback.
ExecutionContext reference_execution_context();
// Native dispatch always calls the existing SYCL boundary and preserves not_built.
// The stream is Queue::impl for enqueue AND all adapter callbacks. Installation
// rejects a different existing stream. Queues/activation operands remain borrowed.
ExecutionContext native_execution_context(::strix::intel::Queue);
Status install_native_sources(::strix::intel::Queue,ExecutionContext&);
Status prepare_native_q8(const WeightView&,std::shared_ptr<ImageCache>,ExecutionContext&,HostQ8Binding&);
Status enqueue_dense(const WeightView&,const DenseArguments&,ExecutionContext&);
}
namespace strix::weight {
template<> struct ArgumentTraits<intel::DenseArguments> {
 static constexpr KernelId schema=intel::q8_f16_ascending_kernel;
 static constexpr std::uint32_t abi_revision=2;
};
}
