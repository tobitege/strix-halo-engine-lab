#pragma once
// CPU backend: the shared core interfaces of docs/architecture/backends.md, implemented on the host.
// Layer 2 of the device rule (one backend per vendor/kind); the ISA specialisations under
// src/cpu/arch/<isa> are layer 3 and are chosen by capability (caps()), never by model.
//
// Every operation is declared with the arithmetic contract of the shared core it implements, so a
// result is bitwise comparable with host references implementing the same contract:
//   - gemv_q8_0_a8: the Q8_0 x A8 decode product (32-weight groups, int32 dot per group,
//     float(dot) * (scale_w * delta_x), group g into lane g % 32, lanes combined by the butterfly
//     v[l] += v[l ^ d] for d = 16 .. 1, result lane 0).
//   - quantize_a8: A8 groups of 32, delta = amax / 127, code = rint(x / delta) clamped to +-127,
//     delta 1 and codes 0 for an all-zero group or amax < 2^-119.
//   - rmsnorm: strix::ops::norm::ReductionOrder::lane_strided_fma_lds_halving with the lane count
//     of NormTraits::lanes.
// Valid calls allocate no memory and start no hidden threads: the caller owns every buffer.
// Compile with contraction off.
#include <cstddef>
#include <cstdint>
#include "strix/weight/host_q8.hpp"

#include "ops/norm/norm.hpp"

namespace strix::cpu {

// ---------------------------------------------------------------------------------- capability
enum class Isa : std::uint8_t { scalar, avx2, avx512, amx, neon, sve };
struct Caps {
    bool avx2 = false, avx512f = false, avx512bw = false, avx512vnni = false, amx_int8 = false;
    bool neon = false, sve = false, dotprod = false;
    unsigned sve_bits = 0;
};
Caps caps() noexcept;               // host detection (cpuid / compile-time ARM features)
const char* isa_name(Isa isa) noexcept;

// ------------------------------------------------------------------------- memory and streams
// Host memory is the device memory: a DevicePtr is a plain pointer, upload is a copy or a mapping.
// A Stream is an in-order queue; the reference executes each call synchronously on the caller's
// thread, so record/wait are no-ops. A threaded specialisation keeps the same in-order semantics.
struct Stream {
    unsigned threads = 1;  // worker budget for one call (cores declared for the cpu unit)
};

// ------------------------------------------------------------------------------ weight formats
inline constexpr std::size_t q8_0_group = weight::Q8Storage::elements;
inline constexpr std::size_t q8_0_block_bytes = weight::Q8Storage::bytes;  // FP16 scale (little endian) + 32 x int8
inline constexpr std::size_t product_lanes = 32;
float half_to_float(std::uint16_t bits) noexcept;

// Related layout to src/core/f16_dense_q8a8.hpp, without its k_bytes/tile0 fields.
// That HIP core uses FAST FP16-rounded operands, not this integer-dot arithmetic contract.
// Caller supplies sufficient buffers and strides; x_stride is a multiple of 32.
struct Q8a8View {
    const std::uint8_t* weight;  // output o at weight + o * row_bytes
    const std::int8_t* codes;    // A8 codes, row r at codes + r * x_stride
    const float* deltas;         // deltas[r * (x_stride / 32) + k / 32]
    float* out;                  // out[r * out_stride + o]
    std::uint32_t row_bytes, in, count, rows, x_stride, out_stride;
    std::uint32_t accumulate;    // out += product
};

// --------------------------------------------------------------------------- kernel table
// One table per ISA specialisation; entries a specialisation does not provide stay null and the
// selector fills them from the reference. Every entry keeps the reference's bits.
// All quantize_a8 / gemv_q8_0_a8 entries throw std::invalid_argument for width / in
// not divisible by 32, before writing any output. Zero width is allowed.
struct Kernels {
    Isa isa = Isa::scalar;
    void (*quantize_a8)(const float* x, std::size_t width, std::int8_t* codes, float* deltas,
                        std::int32_t* sums) = nullptr;
    void (*gemv_q8_0_a8)(const Q8a8View& v, const Stream& s) = nullptr;
    void (*rmsnorm_f32)(const float* x, const float* w, unsigned n, float eps, float* y,
                        unsigned rows, const Stream& s) = nullptr;
};
const Kernels& reference_kernels() noexcept;
// Explicit source registration; architecture scaffolds remain unavailable.
const Kernels* registered_weight_kernels(Isa) noexcept;
// Best table for the host: specialisations in priority order, gaps filled from the reference.
// Returned by value: subsequent selections and selecting-thread exit cannot invalidate it.
Kernels select(const Caps& c) noexcept;

// ------------------------------------------------------------------- shared-norm binding
// Binds strix::ops::norm::Norm<> (the shared, backend-neutral norm interface) to this backend.
struct NormTraits {
    using Input = float;
    using Output = float;
    using Scale = float;
    using ScaleFormat = int;  // f32 only in the scaffold
    using Stream = const ::strix::cpu::Stream*;
    static constexpr float epsilon = 1e-6F;
    static constexpr auto reduction_order = ops::norm::ReductionOrder::lane_strided_fma_lds_halving;
    static constexpr unsigned lanes(unsigned n) { return n >= 512 ? 256U : 128U; }
    static void launch(const float* x, const float* w, int, unsigned n, float eps, float* y,
                       unsigned rows, Stream s) {
        select(caps()).rmsnorm_f32(x, w, n, eps, y, rows, *s);
    }
};
using Norm = ops::norm::Norm<NormTraits>;

}  // namespace strix::cpu
