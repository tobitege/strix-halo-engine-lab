// Scalar, exact reference of the CPU backend. It is the oracle every ISA specialisation must match
// bit for bit; it is written for the declared operation order, not for speed.
#include "cpu/core/backend.hpp"
#include "cpu/weight/source.hpp"
#include "core/integer_math.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace strix::cpu {

float half_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    const std::uint32_t exponent = (bits >> 10U) & 0x1fU, mantissa = bits & 0x3ffU;
    if (exponent == 0U) {
        const float magnitude = std::ldexp(static_cast<float>(mantissa), -24);
        return sign ? -magnitude : magnitude;
    }
    if (exponent == 31U) return std::bit_cast<float>(sign | 0x7f800000U | (mantissa << 13U));
    return std::bit_cast<float>(sign | ((exponent + 112U) << 23U) | (mantissa << 13U));
}

#if defined(STRIX_WFMT_CAPTURE)
namespace weight_capture {
thread_local void (*sink)(const float*,std::size_t,void*)=nullptr;
thread_local void* user=nullptr;
}
#endif

namespace {

void ref_quantize_a8(const float* x, std::size_t width, std::int8_t* codes, float* deltas,
                     std::int32_t* sums) {
    if (width % q8_0_group != 0U)
        throw std::invalid_argument("CPU A8 activation width must be a multiple of 32");
    for (std::size_t g = 0; g < width / q8_0_group; ++g) {
        const float* xg = x + g * q8_0_group;
        float amax = 0.0F;
        for (std::size_t j = 0; j < q8_0_group; ++j) amax = std::max(amax, std::fabs(xg[j]));
        const bool tiny = amax > 0.0F && amax < 0x1p-119F;
        const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
        std::int32_t sum = 0;
        for (std::size_t j = 0; j < q8_0_group; ++j) {
            const float rounded = tiny ? 0.0F : std::nearbyint(xg[j] / delta);
            const auto code = static_cast<std::int8_t>(std::min(127.0F, std::max(-127.0F, rounded)));
            codes[g * q8_0_group + j] = code;
            sum += code;
        }
        deltas[g] = delta;
        if (sums) sums[g] = sum;
    }
}

void ref_gemv_q8_0_a8(const Q8a8View& v, const Stream&) {
    if (v.in % q8_0_group != 0U)
        throw std::invalid_argument("CPU Q8_0 product width must be a multiple of 32");
    const std::size_t groups = v.in / q8_0_group;
    using Source = weight::cpu::Q8RawSource<weight::cpu::CpuTree32>;
    const Source source{v.weight, v.row_bytes};
    for (std::uint32_t r = 0; r < v.rows; ++r) {
        const std::int8_t* xc = v.codes + static_cast<std::size_t>(r) * v.x_stride;
        const float* xd = v.deltas + static_cast<std::size_t>(r) * (v.x_stride / q8_0_group);
        for (std::uint32_t o = 0; o < v.count; ++o) {
            float lanes[product_lanes]{};
            for (std::size_t g = 0; g < groups; ++g) {
                const std::uint8_t* block = source.block(o, g);
                const float scale = Source::scale(block);
                std::int32_t dot = 0;
                for (std::size_t j = 0; j < q8_0_group; ++j)
                    dot += Source::code(block, j) * xc[g * q8_0_group + j];
                lanes[g % product_lanes] = lanes[g % product_lanes] + core::integer_math::product(dot, scale, xd[g]);
            }
            float& y = v.out[static_cast<std::size_t>(r) * v.out_stride + o];
#if defined(STRIX_WFMT_CAPTURE)
            if (weight_capture::sink) weight_capture::sink(lanes, product_lanes, weight_capture::user);
#endif
            const float p = core::integer_math::tree32(lanes);
            y = v.accumulate ? y + p : p;
        }
    }
}

void ref_rmsnorm_f32(const float* x, const float* w, unsigned n, float eps, float* y, unsigned rows,
                     const Stream&) {
    const unsigned lanes = NormTraits::lanes(n);
    std::array<float, 256> partial{};
    for (unsigned row = 0; row < rows; ++row) {
        const float* xr = x + static_cast<std::size_t>(row) * n;
        float* yr = y + static_cast<std::size_t>(row) * n;
        for (unsigned lane = 0; lane < lanes; ++lane) {
            float a = 0.0F;
            for (unsigned i = lane; i < n; i += lanes) a = std::fma(xr[i], xr[i], a);
            partial[lane] = a;
        }
        for (unsigned h = lanes / 2; h > 0; h >>= 1U)
            for (unsigned lane = 0; lane < h; ++lane) partial[lane] = partial[lane] + partial[lane + h];
        const float r = 1.0F / std::sqrt(partial[0] / static_cast<float>(n) + eps);
        for (unsigned i = 0; i < n; ++i) yr[i] = w ? (xr[i] * r) * w[i] : xr[i] * r;
    }
}

constexpr Kernels kReference{Isa::scalar, &ref_quantize_a8, &ref_gemv_q8_0_a8, &ref_rmsnorm_f32};

}  // namespace

const Kernels& reference_kernels() noexcept { return kReference; }

}  // namespace strix::cpu

#if defined(STRIX_WFMT_CAPTURE)
namespace strix::weight::cpu {
void set_partial_sink(PartialSink sink,void* user) noexcept {
    ::strix::cpu::weight_capture::sink=sink;
    ::strix::cpu::weight_capture::user=user;
}
}
#endif
