#pragma once

#include <cstdint>

#if defined(__HIPCC__)
#define STRIX_A8_SMALL_HD __host__ __device__
#else
#define STRIX_A8_SMALL_HD
#endif

namespace strix::dense_a8_small {
// Transposed wave32 WMMA: A = output features, B = activation rows.
// Permute A so accumulator e owns consecutive outputs within a half-wave.
STRIX_A8_SMALL_HD constexpr unsigned weight_row(unsigned lane) {
    return 8U * (lane & 1U) + ((lane & 15U) >> 1U);
}
STRIX_A8_SMALL_HD constexpr unsigned output_row(unsigned lane, unsigned e) {
    return 8U * (lane >> 4U) + e;
}
STRIX_A8_SMALL_HD constexpr unsigned scale_lane(unsigned lane, unsigned e) {
    return 2U * e + (lane >> 4U);
}
// Two adjacent Q8_0 blocks: 17 aligned little-endian dwords, 68 bytes.
STRIX_A8_SMALL_HD constexpr unsigned scale_bits(const unsigned* q, unsigned block) {
    return block ? q[8] >> 16U : q[0] & 0xffffU;
}
STRIX_A8_SMALL_HD constexpr unsigned code_word(const unsigned* q, unsigned block, unsigned half, unsigned k) {
    const unsigned j = 4U * half + k;
    return block ? q[9U + j] : (q[j] >> 16U) | (q[j + 1U] << 16U);
}
}  // namespace strix::dense_a8_small

#undef STRIX_A8_SMALL_HD
