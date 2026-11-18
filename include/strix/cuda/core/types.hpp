#pragma once

#include <cstdint>

namespace strix::cuda::core {

// Device views only: storage and lifetime belong to the caller.
// Per 16-row tile and 32-weight group: 512 fragment-order code bytes and 16 FP16 scales.
struct Q8Tiled {
    const std::uint8_t* codes = nullptr;   // [rows / 16][width / 32][32 lanes][16 bytes]
    const std::uint16_t* scales = nullptr; // [rows / 16][width / 32][16]
    std::uint32_t rows = 0, width = 0;
};
// A8 codes [row][stride], deltas and sums [row][stride / 32]; width <= stride.
// Width and stride are multiples of 32.
struct A8Rows {
    std::int8_t* codes = nullptr;
    float* deltas = nullptr;
    std::int32_t* sums = nullptr;
    std::uint32_t rows = 0, width = 0, stride = 0;
};
}  // namespace strix::cuda::core
