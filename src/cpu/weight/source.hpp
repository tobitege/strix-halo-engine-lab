#pragma once
#include "cpu/core/backend.hpp"
#include "strix/weight/host_q8.hpp"
#include "strix/weight/cache.hpp"
namespace strix::weight::cpu {
struct CpuTree32 {};
template<class Arithmetic> struct Q8RawSource;
template<> struct Q8RawSource<CpuTree32> {
 const std::uint8_t* base;
 std::size_t stride;
 const std::uint8_t* block(std::size_t output,std::size_t group) const noexcept {
  return Q8Storage::block(base+output*stride,group);
 }
 static float scale(const std::uint8_t* packed) noexcept { return ::strix::cpu::half_to_float(Q8Storage::scale_bits(packed)); }
 static std::int8_t code(const std::uint8_t* packed,std::size_t lane) noexcept { return Q8Storage::code(packed,lane); }
};
inline constexpr KernelId q8_tree32_kernel=0x43505501;
struct DenseArguments {
 ::strix::cpu::Q8a8View view;
 ::strix::cpu::Stream stream;
 Q8BindingRef binding;
};
std::span<const Candidate> candidates() noexcept;
Status validate_view(const ::strix::cpu::Q8a8View&);
Status from_legacy(const ::strix::cpu::Q8a8View&,std::span<const Byte>,std::shared_ptr<const void>,TensorId,WeightView&);
Status enqueue_dense(const WeightView&,const DenseArguments&);
ExecutionContext execution_context();
Status require_isa(::strix::cpu::Isa);
#if defined(STRIX_WFMT_CAPTURE)
using PartialSink=void(*)(const float*,std::size_t,void*);
void set_partial_sink(PartialSink,void*) noexcept;
#endif
}
namespace strix::weight {
template<> struct ArgumentTraits<cpu::DenseArguments> {
 static constexpr KernelId schema=cpu::q8_tree32_kernel;
 static constexpr std::uint32_t abi_revision=2;
};
}
