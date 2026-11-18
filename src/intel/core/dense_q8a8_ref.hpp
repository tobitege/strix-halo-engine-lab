#pragma once
// Reference kernel of the shared dense policy dense-q8a8-f16-v1 (src/core/f16_dense_q8a8.hpp) for the Intel backend:
//   y[r][o] (+)= sum_k fp16(code_w * d) * fp16(code_a * delta), FP32 accumulation, k ascending, from +0.
// Data fields correspond to the HIP policy; its launch-only tile0 is omitted. One work item
// computes one (r, o) cell; the body is plain C++ and is the same function the host loop and a SYCL parallel_for call.
// It is the correctness oracle of the XMX path (arch/xe2), not a fast path: XMX (DPAS) sums k in hardware order, so
// device-vs-reference checks require a declared tolerance; host tests do not qualify device arithmetic.
#include "fp16.hpp"
#include "strix/weight/host_q8.hpp"

#include <cstdint>
#include <cstring>

namespace strix::intel::dense {

inline constexpr char kQ8A8PolicyName[] = "dense-q8a8-f16-v1";

struct Q8A8View {
    const std::uint8_t* weight = nullptr;  // Q8_0 rows: per 32 weights {FP16 scale, 32 x int8}
    const std::int8_t* codes = nullptr;    // A8 codes: row r at codes + r * x_stride
    const float* deltas = nullptr;         // deltas[r * (x_stride / 32) + k / 32]
    float* out = nullptr;                  // out[r * out_stride + o]
    std::uint32_t row_bytes = 0, k_bytes = 0;
    std::uint32_t in = 0, count = 0, rows = 0, x_stride = 0, out_stride = 0;
    std::uint32_t accumulate = 0;
};

inline bool supported(const Q8A8View& v) {
    const std::uint64_t required = static_cast<std::uint64_t>(v.k_bytes) + weight::Q8Storage::row_bytes(v.in);
    return v.weight && v.codes && v.deltas && v.out && v.rows && v.count && v.in &&
           v.in % 128U == 0U && v.x_stride % 32U == 0U && v.in <= v.x_stride &&
           required <= v.row_bytes && v.out_stride >= v.count;
}

inline void q8a8_cell(const Q8A8View& v, std::uint32_t r, std::uint32_t o) {
    const std::uint8_t* w = v.weight + static_cast<std::size_t>(o) * v.row_bytes + v.k_bytes;
    const std::int8_t* a = v.codes + static_cast<std::size_t>(r) * v.x_stride;
    const float* delta = v.deltas + static_cast<std::size_t>(r) * (v.x_stride / 32U);
    float acc = 0.0F;
    for (std::uint32_t b = 0; b < v.in / 32U; ++b) {
        const std::size_t block_offset = static_cast<std::size_t>(b) * 34U;
        std::uint16_t scale_bits = 0;
        std::memcpy(&scale_bits, w + block_offset, sizeof scale_bits);
        const float d = fp16::to_float(scale_bits);
        for (std::uint32_t j = 0; j < 32U; ++j) {
            const auto code_w = static_cast<std::int8_t>(w[block_offset + 2U + j]);
            const float w16 = fp16::round(static_cast<float>(code_w) * d);
            const float a16 = fp16::round(static_cast<float>(a[b * 32U + j]) * delta[b]);
            acc += w16 * a16;
        }
    }
    float& y = v.out[static_cast<std::size_t>(r) * v.out_stride + o];
    y = v.accumulate ? y + acc : acc;
}

// Host loop over every cell (the plain C++ build of the kernel).
inline void run_reference(const Q8A8View& v) {
    for (std::uint32_t r = 0; r < v.rows; ++r)
        for (std::uint32_t o = 0; o < v.count; ++o) q8a8_cell(v, r, o);
}

} // namespace strix::intel::dense
