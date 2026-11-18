#pragma once
// Shared 16-weight chunk decoders for the small-row GEMV sweeps (gemv_rows_direct.inc binds
// GEMV_ROWS_CHUNK / GEMV_ROWS_LOAD_CHUNK to these). One copy for BF16 and Q8_0 rows:
//   bf16  weight = float(bits << 16)                        (exact)
//   q8_0  34-byte blocks {f16 d, int8 q[32]}; weight = d * float(q)   (one FP32 product, as the g4 chunk decoder)
// Home: shared (core). Gemma4's gemma4/kernels/chunk.hpp WeightChunk<bf16|q8_0> (chunk.hpp:32,52) is the same
// arithmetic. Required adoption: replace those two WeightChunk specializations with adapters
// to Chunk below, then run Gemma4 BF16/Q8_0 projection byte gates before removing local decoders.
// Rebinding a HIP kernel is outside this change's permitted scope.
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>

namespace strix::core::weight_decode {

enum class Format : std::uint8_t { bf16, q8_0 };

__device__ __forceinline__ uint4 load_chunk_bytes(const std::uint8_t* p) {
    uint4 v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}
__device__ __forceinline__ unsigned word_of(uint4 v, unsigned i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
__device__ __forceinline__ float bf16_to_float(std::uint16_t h) { return __builtin_bit_cast(float, std::uint32_t(h) << 16U); }

template <Format F> struct Chunk;
template <> struct Chunk<Format::bf16> {
    uint4 a, b;
    __device__ Chunk() = default;
    __device__ Chunk(const std::uint8_t* row, unsigned i) : a(load_chunk_bytes(row + 2ULL * i)), b(load_chunk_bytes(row + 2ULL * i + 16)) {}
    __device__ float at(unsigned e) const {
        const auto w = word_of(e < 8U ? a : b, (e % 8U) / 2U);
        return bf16_to_float(std::uint16_t(w >> ((e % 2U) * 16U)));
    }
};
template <> struct Chunk<Format::q8_0> {
    uint4 codes;
    float d;
    __device__ Chunk() = default;
    // A 16-chunk never crosses a 32-block; the 34-byte block stride keeps the payload 2-byte aligned only.
    __device__ Chunk(const std::uint8_t* row, unsigned i) {
        const std::uint8_t* p = row + std::size_t(i / 32U) * 34U;
        std::uint16_t h;
        __builtin_memcpy(&h, p, 2);
        d = float(__builtin_bit_cast(_Float16, h));
        __builtin_memcpy(&codes, p + 2 + i % 32U, sizeof(codes));
    }
    __device__ float at(unsigned e) const {
        const unsigned q = (word_of(codes, e / 4U) >> ((e % 4U) * 8U)) & 255U;
        return d * float(q < 128U ? int(q) : int(q) - 256);
    }
};
inline constexpr std::size_t row_bytes(Format f, std::size_t inputs) { return f == Format::bf16 ? inputs * 2U : (inputs / 32U) * 34U; }

} // namespace strix::core::weight_decode
