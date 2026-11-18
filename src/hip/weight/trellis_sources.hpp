#pragma once
// Registered HIP operand sources of the EXL3 / trellis family (mul1 codebook, 16x16 tile rings).
// Home: shared HIP backend core. The statements here are the single definition of the input transform of the MiniMax / GLM
// MoE kernels (src/moe_core/exl3_trellis.hpp forwards to them) and of the generic mul1 tile codec; scheduling, split policy
// and every epilogue stay in the callers. No arithmetic is selected from the format: each arithmetic is its own tag.
// This header needs only the HIP and core headers (the engine kernels include it); the registry-facing primary template is
// the one of source.hpp.
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "../../core/hadamard.hpp"
#include "../../core/trellis_codec.hpp"
#include "trellis_partial_trace.hpp"

namespace strix::weight::hip {

template<class Storage, class Protocol, class Arch, class Arithmetic> struct OperandSource;
struct Gfx1151;

struct TrellisTileTag {};
template<unsigned Bits> struct Exl3Mul1TilesTag {};   // storage exl3.mul1.tiles16 (trellis | suh | svh per expert)
template<unsigned Bits> struct GemmaTransportTag {};  // storage exl3.gemma.transport (suh | svh | trellis)

// Distinct numeric tags, deliberately not aliases: the arithmetic is never inferred from the format.
struct TrellisMul1LegacyArithmetic {};  // RN16(H128(RN16(RN32(x * suh)))) input, v_dot2 split core (MiniMax / GLM / DS41 library)
struct GemmaExl3V1Arithmetic {};
struct GemmaExl3V2Arithmetic {};
struct GemmaExl3V3Arithmetic {};
struct GemmaExl3WmmaPrefillV1Arithmetic {};

// One resident expert after its own sidecars were uploaded: there is no shared transform and no nominal row stride.
struct TrellisExpertView {
    const std::uint32_t* trellis = nullptr;
    const std::uint16_t* suh = nullptr;
    const std::uint16_t* svh = nullptr;
    std::uint32_t k = 0, n = 0, bits = 0;
};
static_assert(std::is_trivially_copyable_v<TrellisExpertView>);

// Gemma's contiguous transport is suh | svh | trellis (the canonical library transport is trellis | suh | svh): the views
// are different types and are never cast into each other. row_bytes of the Gemma LinearView is nominal (K * bits / 8) and is
// not a stride; tiles are addressed from the end of the sidecars.
struct GemmaTransportView {
    const std::uint8_t* data = nullptr;
    std::uint32_t k = 0, n = 0, bits = 0;
    __host__ __device__ constexpr const std::uint8_t* suh_bytes() const { return data; }
    __host__ __device__ constexpr const std::uint8_t* svh_bytes() const { return data + 2ULL * k; }
    __host__ __device__ constexpr const std::uint8_t* tile(unsigned input_tile, unsigned output_tile) const {
        return data + 2ULL * (k + n) + (static_cast<std::size_t>(input_tile) * (n / 16U) + output_tile) * (32U * bits);
    }
};
static_assert(std::is_trivially_copyable_v<GemmaTransportView>);

namespace trellis_source_detail {
__device__ __forceinline__ float materialize(float v) { return core::materialize_fp32(v); }
__device__ __forceinline__ float half_bits(std::uint16_t h) { return __half2float(__ushort_as_half(h)); }

// One wave: input transform of the 128 channels [begin, begin + 128) of one row.
__device__ __forceinline__ void input_block(const float* x, const std::uint16_t* suh, _Float16* out, unsigned begin, unsigned lane) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = begin + lane + 32U * static_cast<unsigned>(j);
        const float scaled = materialize(x[c] * half_bits(suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    core::hadamard128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) out[begin + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
#if defined(STRIX_WFMT_CAPTURE) && STRIX_WFMT_CAPTURE
    for (int j = 0; j < 4; ++j) strix::wfmt_trace::trellis_trace(strix::wfmt_trace::kTrellisInput, static_cast<_Float16>(v[j]));
#endif
}
// Same transform on the lane's four x values already in registers (xv[j] = x[begin + lane + 32 j]); the 128 results go to
// out[obegin ...].
__device__ __forceinline__ void input_values(const float (&xv)[4], const std::uint16_t* suh, _Float16* out, unsigned begin,
                                             unsigned obegin, unsigned lane) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = begin + lane + 32U * static_cast<unsigned>(j);
        const float scaled = materialize(xv[j] * half_bits(suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    core::hadamard128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) out[obegin + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
#if defined(STRIX_WFMT_CAPTURE) && STRIX_WFMT_CAPTURE
    for (int j = 0; j < 4; ++j) strix::wfmt_trace::trellis_trace(strix::wfmt_trace::kTrellisInput, static_cast<_Float16>(v[j]));
#endif
}
// Same transform returning the four per-lane values in registers (no store).
__device__ __forceinline__ void input_registers(const float* x, const std::uint16_t* suh, unsigned begin, unsigned lane, float (&v)[4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = begin + lane + 32U * static_cast<unsigned>(j);
        const float scaled = materialize(x[c] * half_bits(suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    core::hadamard128(v, lane);
#if defined(STRIX_WFMT_CAPTURE) && STRIX_WFMT_CAPTURE
    for (int j = 0; j < 4; ++j) strix::wfmt_trace::trellis_trace(strix::wfmt_trace::kTrellisInput, v[j]);
#endif
}
}  // namespace trellis_source_detail

template<unsigned Bits>
struct OperandSource<Exl3Mul1TilesTag<Bits>, TrellisTileTag, Gfx1151, TrellisMul1LegacyArithmetic>
    : core::trellis::Codec<static_cast<int>(Bits), core::trellis::Mul1> {
    static_assert(Bits >= 1 && Bits <= 8);
    using View = TrellisExpertView;
    using NumericContract = TrellisMul1LegacyArithmetic;
    static constexpr unsigned group_k = 16, alignment = 4, shared_bytes = 0;
    static constexpr bool has_input_transform = true, has_output_transform = true;
    static __host__ __device__ __forceinline__ float codebook_value(std::uint32_t state, core::trellis::Mul1 = {}) {
        return core::trellis::Mul1::value(state);
    }
    static __device__ __forceinline__ void input_block(const float* x, const std::uint16_t* suh, _Float16* out, unsigned begin, unsigned lane) {
        trellis_source_detail::input_block(x, suh, out, begin, lane);
    }
    static __device__ __forceinline__ void h128(float (&v)[4], unsigned lane) { core::hadamard128(v, lane); }
    static __device__ __forceinline__ float scale(float x, std::uint16_t sign) {
        return trellis_source_detail::materialize(x * trellis_source_detail::half_bits(sign));
    }
};
template<unsigned Bits>
using LegacyMul1Source = OperandSource<Exl3Mul1TilesTag<Bits>, TrellisTileTag, Gfx1151, TrellisMul1LegacyArithmetic>;

// Gemma: the shared tile codec and the transport addressing only. Its flagged RN32/RN16 transforms, preload barriers and the
// per-identity cores stay in src/gemma4/exl3/device.hip; the generic mul1 input transform is never bound here.
template<unsigned Bits, class Arithmetic>
struct GemmaTransportSource : core::trellis::Codec<static_cast<int>(Bits), core::trellis::Mul1> {
    static_assert(Bits == 4 || Bits == 5 || Bits == 6 || Bits == 8);
    using View = GemmaTransportView;
    using NumericContract = Arithmetic;
    static constexpr unsigned group_k = 16, alignment = 2, shared_bytes = 0;
    static constexpr bool has_input_transform = true, has_output_transform = true;
    static __device__ __forceinline__ const std::uint8_t* tile(const View& v, unsigned input_tile, unsigned output_tile) {
        return v.tile(input_tile, output_tile);
    }
};
#define STRIX_WFMT_GEMMA_SOURCE(Arithmetic) \
    template<unsigned Bits> struct OperandSource<GemmaTransportTag<Bits>, TrellisTileTag, Gfx1151, Arithmetic> \
        : GemmaTransportSource<Bits, Arithmetic> {}
STRIX_WFMT_GEMMA_SOURCE(GemmaExl3V1Arithmetic);
STRIX_WFMT_GEMMA_SOURCE(GemmaExl3V2Arithmetic);
STRIX_WFMT_GEMMA_SOURCE(GemmaExl3V3Arithmetic);
STRIX_WFMT_GEMMA_SOURCE(GemmaExl3WmmaPrefillV1Arithmetic);
#undef STRIX_WFMT_GEMMA_SOURCE

}  // namespace strix::weight::hip
