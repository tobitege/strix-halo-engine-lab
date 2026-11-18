#pragma once
// Bit-exact IEEE binary16 conversions in plain C++ (no _Float16, no sycl::half): the reference kernels round exactly
// as the HIP cores' _Float16 casts (RNE, overflow to inf, NaN kept quiet), on every host compiler and inside SYCL
// device code, so a device result can be compared to the host reference without a toolchain-specific half type.
#include <cstdint>
#include <cstring>

namespace strix::intel::fp16 {

inline std::uint16_t from_float(float value) {
    std::uint32_t x = 0;
    std::memcpy(&x, &value, sizeof x);
    const auto sign = static_cast<std::uint16_t>((x >> 16) & 0x8000U);
    std::uint32_t ax = x & 0x7fffffffU;
    if (ax >= 0x7f800000U) return static_cast<std::uint16_t>(sign | (ax > 0x7f800000U ? 0x7e00U : 0x7c00U));
    if (ax >= 0x477ff000U) return static_cast<std::uint16_t>(sign | 0x7c00U);  // |v| >= 65520 rounds to inf
    if (ax < 0x38800000U) {  // below 2^-14: subnormal or zero; adding 0.5 rounds at the 2^-24 step (RNE in the FPU)
        float a = 0.0F;
        std::memcpy(&a, &ax, sizeof a);
        a += 0.5F;
        std::uint32_t r = 0;
        std::memcpy(&r, &a, sizeof r);
        return static_cast<std::uint16_t>(sign | (r - 0x3f000000U));
    }
    ax += 0xc8000fffU + ((ax >> 13) & 1U);  // rebias 127 -> 15 and round to nearest even
    return static_cast<std::uint16_t>(sign | (ax >> 13));
}

inline float to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1fU;
    std::uint32_t mant = h & 0x3ffU;
    std::uint32_t bits = 0;
    if (exp == 0x1fU) bits = sign | 0x7f800000U | (mant << 13);
    else if (exp != 0U) bits = sign | ((exp + 112U) << 23) | (mant << 13);
    else if (mant == 0U) bits = sign;
    else {
        std::uint32_t e = 113U;
        while ((mant & 0x400U) == 0U) {
            mant <<= 1;
            --e;
        }
        bits = sign | (e << 23) | ((mant & 0x3ffU) << 13);
    }
    float out = 0.0F;
    std::memcpy(&out, &bits, sizeof out);
    return out;
}

inline float round(float value) { return to_float(from_float(value)); }

} // namespace strix::intel::fp16
