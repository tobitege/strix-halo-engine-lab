#pragma once

#include "strix/weight/types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#ifdef __CUDACC__
#define STRIX_CUDA_WEIGHT_HD __host__ __device__ __forceinline__
#define STRIX_CUDA_WEIGHT_DEVICE __device__ __forceinline__
#else
#define STRIX_CUDA_WEIGHT_HD inline
#endif

namespace strix::cuda::weight {

// Protocols describe operands consumed by a tuned core, never a model or a
// runtime format switch. Unsupported combinations deliberately have no body.
struct NativeScalarTag {};
struct SignedA8Tag {};
struct F16TileTag {};
struct Bf16TileTag {};
struct Fp8TileTag {};
struct Nvfp4TileTag {};
template <class Storage, class Protocol, class Arch, class Arithmetic>
struct OperandSource;

template <std::uint32_t WireType> struct GgmlTag {};
using F32Tag = GgmlTag<0>;
using F16Tag = GgmlTag<1>;
using Q8_0Tag = GgmlTag<8>;
using Bf16Tag = GgmlTag<30>;

struct LaunchShape {
    std::uint64_t m{}, n{}, k{}, expert_stride{};
    std::span<const std::uint32_t> rows_per_expert{};
    std::uint32_t split_count{1};
};

// Ownership is held by the upload/image lease through its completion event;
// only these immutable addresses enter a trivially-copyable device View.
struct UploadedSlice {
    const void* data{};
    std::uint64_t bytes{};
};
struct UploadedSlices {
    UploadedSlice payload;
    std::span<const UploadedSlice> sidecars;
};

struct TransformJob {
    const void* input{};
    void* output{};
    std::uint64_t rows{}, width{}, input_stride_bytes{}, output_stride_bytes{};
    const std::uint32_t* expert_indices{};
    void* scratch{};
    std::size_t scratch_bytes{};
    ::strix::weight::Scalar input_scalar{}, output_scalar{};
};

template <class Source, class Arithmetic, class ExistingCore>
struct BoundCore : ExistingCore {
    using WeightSource = Source;
    using NumericContract = Arithmetic;
};

} // namespace strix::cuda::weight
