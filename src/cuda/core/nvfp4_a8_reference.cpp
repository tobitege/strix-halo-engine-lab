// Host oracles of nvfp4-a8-v1 (include/strix/cuda/core/nvfp4_a8.hpp). The converter reuses the #683 quantizer.
#include "strix/cuda/core/nvfp4_a8.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "strix/cuda/core/nvfp4.hpp"

namespace strix::cuda::core::nvfp4_a8 {
namespace {
float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = (h >> 15) & 1U, exp = (h >> 10) & 31U, man = h & 1023U;
    float v;
    if (exp == 0) v = std::ldexp(static_cast<float>(man), -24);
    else if (exp == 31) v = man ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    else v = std::ldexp(static_cast<float>(man | 1024U), static_cast<int>(exp) - 25);
    return sign ? -v : v;
}
}  // namespace

float tensor_scale_for_amax(float amax) {
    if (!(amax > 0) || !std::isfinite(amax)) return 1.0F;
    const float ts = amax / (6.0F * 448.0F);
    return ts > 0 && std::isfinite(ts) ? ts : 1.0F;
}

void reference_convert_q8_0(const std::uint8_t* q8, std::uint32_t n, std::uint32_t k, std::uint8_t* codes,
                            std::uint8_t* scales, float* tensor_scale) {
    if (!q8 || !codes || !scales || !tensor_scale || n == 0 || k == 0 || k % kBlock) throw std::invalid_argument("q8_0 view");
    std::vector<float> x(std::size_t{n} * k);
    float amax = 0;
    for (std::size_t r = 0; r < n; ++r)
        for (std::uint32_t b = 0; b < k / kBlock; ++b) {
            const std::uint8_t* blk = q8 + (r * (k / kBlock) + b) * 34U;
            std::uint16_t h;
            std::memcpy(&h, blk, 2);
            const float d = half_to_float(h);
            for (std::uint32_t j = 0; j < kBlock; ++j) {
                const float v = d * static_cast<float>(static_cast<std::int8_t>(blk[2 + j]));
                if (!std::isfinite(v)) throw std::invalid_argument("nonfinite q8_0 weight");
                x[r * k + b * kBlock + j] = v;
                amax = std::max(amax, std::fabs(v));
            }
        }
    *tensor_scale = tensor_scale_for_amax(amax);
    nvfp4::quantize(x.data(), k, n, k, codes, k / 2, scales, k / 16, *tensor_scale);
}

float reference_dot(const std::uint8_t* codes, const std::uint8_t* scales, float tensor_scale, const std::int8_t* a,
                    const float* deltas_row, std::uint32_t k) {
    float lane[kLanes] = {};
    for (std::uint32_t g = 0; g < k / kGroup; ++g) {
        std::int32_t dot = 0;
        for (std::uint32_t j = 0; j < kGroup; ++j) {
            const std::uint32_t at = g * kGroup + j;
            const std::uint8_t c = (codes[at / 2] >> ((at & 1U) * 4U)) & 15U;
            dot += static_cast<std::int32_t>(a[at]) * static_cast<std::int32_t>(2.0F * nvfp4::e2m1(c));
        }
        volatile float f = nvfp4::e4m3(scales[g]) * deltas_row[g / 2];  // rounded product, no contraction
        lane[g % kLanes] = std::fma(static_cast<float>(f), static_cast<float>(dot), lane[g % kLanes]);
    }
    for (std::uint32_t off = 16; off > 0; off >>= 1) {
        float next[kLanes];
        for (std::uint32_t i = 0; i < kLanes; ++i) {
            volatile float s = lane[i] + lane[i ^ off];
            next[i] = s;
        }
        std::copy(next, next + kLanes, lane);
    }
    volatile float y = lane[0] * (0.5F * tensor_scale);
    return y;
}

}  // namespace strix::cuda::core::nvfp4_a8
