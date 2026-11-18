#pragma once
#include "types.hpp"
#include <strix/device_profile.hpp>
#include <type_traits>

namespace strix::weight {
struct Hardware {
  Backend backend; std::string arch, compiler_id, device_class;
  std::uint64_t isa_mask, free_bytes, cache_budget_bytes;
  std::uint32_t registers_per_sm, registers_per_thread_limit;
  std::uint32_t shared_bytes_per_sm, warp_or_wave;
  double memory_bytes_per_second, lookup_per_second, dot4_per_second;
  double valu_ops_per_second, matrix_ops_per_second;
  bool valu_matrix_serialize;
};
struct Workload {
  std::uint64_t m, n, k, experts, active_experts;
  std::vector<std::uint32_t> rows_per_expert;
  std::uint32_t concurrent_requests;
  bool warm_image, warm_file;
};
struct Resources {
  std::uint32_t registers, threads, shared_bytes, spills, min_resident_blocks;
  std::uint64_t workspace_bytes, image_bytes;
};
using KernelId = std::uint32_t; // build-local dispatch ID, never persistent key
struct Candidate {
  KernelId kernel; FormatId storage; ArithmeticId arithmetic; ImageId image;
  Backend backend; Strategy strategy; Exactness exactness;
  std::uint64_t required_isa; Resources resources; CostDescriptor cost;
  Status (*validate)(const WeightView&, const Workload&, const Hardware&) = nullptr;
  // Content identity verified by the dispatch registrar against the loaded
  // implementation (including reductions/tails), never copied from a receipt.
  Digest256 implementation{};
};
struct ResolveRequest {
  const WeightMapEntry* tensor; Hardware hardware; Workload workload;
  ArithmeticId required_arithmetic; Exactness maximum_permission;
  KernelId incumbent; bool allow_fast;
  bool override_strategy; Strategy forced_strategy;
  bool use_qualified_decisions = true;
  // Verified content digest of the loaded executable and kernel modules.
  // A source commit/compiler label is insufficient. Absent identity fails closed.
  Digest256 build_identity{};
};
struct Qualification {
  Digest256 receipt; std::string device_class, compiler_id, shape_regime;
  Digest256 build_identity, incumbent_implementation, replacement_implementation;
  ArithmeticId arithmetic;
  bool byte_equal, faster, quality_passed;
};
struct Plan {
  Candidate selected; std::uint64_t workspace_bytes;
  double predicted_us; std::string reason; Digest256 qualification;
};
Status resolve(const Registry&, const ResolveRequest&,
               std::span<const Candidate>, std::span<const Qualification>, Plan&);
std::string effective_config(const WeightMap&, std::span<const Plan>);
struct ImageKey {
  Digest256 source, sidecars, metadata; TensorId tensor;
  ByteRange payload; std::array<std::uint64_t,4> byte_strides;
  bool has_byte_strides;
  Shape shape; FormatId storage; ImageId image; ArithmeticId arithmetic;
  Backend backend; std::string arch, kernel_abi;
};
class ImageLease; class ImageCache; struct ExecutionContext;
struct KernelArguments {
  KernelId schema; std::uint32_t abi_revision;
  const void* data; std::size_t bytes;
}; // checked typed launch-record envelope, not an untyped tensor pointer
Status acquire_image(ImageCache&, const ImageKey&, const Plan&,
                     const WeightView&, ExecutionContext&, ImageLease&);
Status record_last_use(ImageLease&, ExecutionContext&);
Status enqueue(const Plan&, const WeightView&, const KernelArguments&,
               void* output, void* workspace, std::size_t workspace_bytes,
               ExecutionContext&);
template<class Args> struct ArgumentTraits; // backend registration supplies schema and ABI
// An lvalue envelope is consumed synchronously; no temporary launch record may escape.
template<class Args> KernelArguments arguments(Args& args) {
  using Type = std::remove_cv_t<Args>;
  static_assert(std::is_standard_layout_v<Type> && std::is_trivially_copyable_v<Type>);
  return {ArgumentTraits<Type>::schema, ArgumentTraits<Type>::abi_revision, &args, sizeof(Type)};
}
template<class Args> KernelArguments arguments(const Args&&) = delete;
bool operator==(const Digest256&, const Digest256&) noexcept;
bool operator==(const ImageKey&, const ImageKey&) noexcept;
std::string shape_regime(const Workload&);
Status validate_qualification(const Qualification&);
struct DecisionTable {
  std::vector<Qualification> entries;
  std::vector<std::shared_ptr<const std::string>> owned_arithmetic;
  std::vector<std::string> source_commits, compiler_flags;
};
Status import_decisions(std::string_view json, DecisionTable&);
Status estimate_cost(const Candidate&, const Hardware&, const Workload&, double&);
Hardware hardware_from_profile(const strix::DeviceProfile&, std::string compiler_id);
} // namespace strix::weight
