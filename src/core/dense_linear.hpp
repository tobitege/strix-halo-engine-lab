#pragma once
// Dense decode projection binding of the shared small-row sweep (gemv_rows_direct.inc) to the shared
// BF16 / Q8_0 chunk decoders (weight_decode.hpp). out[r][o] = sum_k W[o][k] * x[r][k], rows 1..16.
// Epilogue kinds: store, or SiLU(gate) * up over two matrices that share every activation read.
// Home: shared (core): no model names; qwen35 binds it, Gemma4/Qwen4 can later.
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace strix::core {
// Exact small-row sweeps: never round a logical tail up into adjacent storage.
template <class F>
void dense_row_sweeps(unsigned rows, F&& run) {
    unsigned offset = 0;
    for (unsigned tile = 16; tile; tile /= 2)
        while (rows - offset >= tile) {
            run(offset, tile);
            offset += tile;
        }
}

// Non-owning execution scratch view; its backend context owns the allocations.
struct DenseWideScratch {
    std::int8_t* codes = nullptr;
    float *deltas = nullptr, *g = nullptr, *u = nullptr;
};

// Host admission for embedding/dense inputs, shared by backend adapters.
inline std::uint32_t parse_u32(std::string_view text) {
    std::uint32_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument("expected an unsigned 32-bit decimal integer");
    return value;
}
inline std::size_t checked_product(std::initializer_list<std::size_t> factors) {
    std::size_t result = 1;
    for (auto n : factors) {
        if (n && result > std::numeric_limits<std::size_t>::max() / n)
            throw std::invalid_argument("allocation size overflow");
        result *= n;
    }
    return result;
}
inline void require_capacity(std::size_t pos, std::size_t rows, std::size_t capacity) {
    if (pos > capacity || rows > capacity - pos)
        throw std::invalid_argument("max_positions exceeded");
}
template <class Token>
void validate_token_ids(const Token* ids, std::size_t count, std::uint32_t vocab) {
    if (count && !ids) throw std::invalid_argument("null token IDs");
    for (std::size_t i = 0; i < count; ++i)
        if (static_cast<std::uint64_t>(ids[i]) >= vocab)
            throw std::invalid_argument("token ID outside vocabulary");
}
} // namespace strix::core

#if defined(__HIPCC__)
#include "gemv_rows.hpp"
#include "weight_decode.hpp"

namespace strix::core::dense_linear {

enum class Epilogue : std::uint8_t { store, silu_mul };

struct View {
    const std::uint8_t* data;
    unsigned inputs, outputs;
    std::size_t row_bytes;
};

__device__ __forceinline__ float wave_sum32(float v) {
    // Fixed butterfly (xor 16,8,4,2,1): identical bits for every schedule.
#pragma unroll
    for (unsigned m = 16U; m > 0U; m >>= 1U)
        v += __shfl_xor(v, int(m), 32);
    return v;
}
__device__ __forceinline__ float silu(float x) { return x / (1.0F + expf(-x)); }

// 4 waves per block, T outputs per wave. grid = ceil(outputs / (4 * T)).
template <unsigned Rows, unsigned T, weight_decode::Format Format, Epilogue Ep>
__global__ void __launch_bounds__(128) kernel(View w, View up, const float* x, float* out) {
    const unsigned first = (blockIdx.x * 4U + threadIdx.x / 32U) * T, lane = threadIdx.x % 32U;
    constexpr unsigned Streams = Ep == Epilogue::silu_mul ? 2U : 1U;
    const auto inputs = w.inputs, outputs = w.outputs;
#define GEMV_ROWS_WEIGHT_ROW(stream, o) ((stream) ? up.data + std::size_t(o) * up.row_bytes : w.data + std::size_t(o) * w.row_bytes)
#define GEMV_ROWS_CHUNK weight_decode::Chunk<Format>
#define GEMV_ROWS_LOAD_CHUNK(row, start) weight_decode::Chunk<Format>(row, start)
#define GEMV_ROWS_LANE_SUM(partial) wave_sum32(partial)
#define GEMV_ROWS_STORE(row, o, sa, sb)                                                                              \
    if (lane == 0 && (o) < outputs)                                                                                \
    out[std::size_t(row) * outputs + (o)] = Ep == Epilogue::silu_mul ? silu(sa) * (sb) : (sa)
#include "gemv_rows_direct.inc"
#undef GEMV_ROWS_WEIGHT_ROW
#undef GEMV_ROWS_CHUNK
#undef GEMV_ROWS_LOAD_CHUNK
#undef GEMV_ROWS_LANE_SUM
#undef GEMV_ROWS_STORE
}

} // namespace strix::core::dense_linear
#endif
