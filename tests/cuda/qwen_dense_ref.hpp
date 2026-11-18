// Literal host (CPU) reference of the HIP dense A8 contract and shared expert (src/hip_q8_rows_gemv.hpp :1030-1159,
// src/hip_shared_expert_launcher.hip:535-562, 1495-1527). One IEEE operation per HIP instruction, scalar loops in the
// HIP order (MSVC /fp:precise: no contraction; explicit std::fmaf where the HIP side has an FMA).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::qwen_dense_ref {

inline float bits_float(std::uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
inline std::uint32_t float_bits(float f) { std::uint32_t b; std::memcpy(&b, &f, 4); return b; }
inline std::uint16_t f2bf(float v) { const std::uint32_t b = float_bits(v); return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U); }
inline float bf2f(std::uint16_t h) { return bits_float(static_cast<std::uint32_t>(h) << 16U); }
inline float half_to_float(std::uint16_t h) {
    const std::uint32_t s = (h >> 15) & 1U, e = (h >> 10) & 31U, m = h & 1023U;
    float v;
    if (e == 0) v = std::ldexp(static_cast<float>(m), -24);
    else if (e == 31) v = m ? bits_float(0x7fc00000U | (m << 13)) : bits_float(0x7f800000U);
    else v = std::ldexp(static_cast<float>(m | 1024U), static_cast<int>(e) - 25);
    return s ? -v : v;
}
constexpr std::uint32_t kNan = 0x7fc00000U;

// codes [r * width + f], deltas [b * pitch + r]; x(r, f) = x[r * stride + f] (already FP32).
inline void quantize(std::uint32_t rows, std::uint32_t width, const float* x, std::uint64_t stride, std::int8_t* codes,
                     float* deltas, std::uint32_t pitch) {
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t b = 0; b < width / 32U; ++b) {
            const float* v = x + r * stride + 32U * b;
            bool bad = false;
            float amax = 0.0F;
            for (int k = 0; k < 32; ++k) { bad |= !std::isfinite(v[k]); amax = std::fmax(amax, std::fabs(v[k])); }
            const bool tiny = amax > 0.0F && amax < 0x1p-119F;
            const float delta = bad ? bits_float(kNan) : (amax == 0.0F || tiny) ? 1.0F : amax / 127.0F;
            for (int k = 0; k < 32; ++k) {
                const float q = v[k] / delta;
                const float rounded = bad || tiny ? 0.0F : std::nearbyint(q);   // default rounding: half-even
                codes[static_cast<std::uint64_t>(r) * width + 32U * b + k] =
                    static_cast<std::int8_t>(static_cast<int>(std::fmin(127.0F, std::fmax(-127.0F, rounded))));
            }
            deltas[static_cast<std::uint64_t>(b) * pitch + r] = delta;
        }
}

// One (row, output): weight row = Q8_0 blocks; returns the canonicalised y.
inline float project_one(const std::uint8_t* wrow, std::uint32_t width, const std::int8_t* code_row, const float* deltas,
                         std::uint32_t pitch, std::uint32_t row) {
    float acc = 0.0F;
    for (std::uint32_t b = 0; b < width / 32U; ++b) {
        const std::uint8_t* blk = wrow + 34U * b;
        const float dw = half_to_float(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
        int dot = 0;
        for (int k = 0; k < 32; ++k) dot += static_cast<int>(static_cast<std::int8_t>(blk[2 + k])) * code_row[32U * b + k];
        const float p = dw * deltas[static_cast<std::uint64_t>(b) * pitch + row];
        acc = std::fmaf(p, static_cast<float>(dot), acc);
    }
    return acc != acc ? bits_float(kNan) : acc;
}

// v_mul_f32 with RDNA3 NaN propagation (src0 first, quieted) [assumption, see src/cuda/models/qwen/qwen_dense.cu amd_mul].
inline float amd_mul(float a, float b) {
    if (a != a) return bits_float(float_bits(a) | 0x00400000U);
    if (b != b) return bits_float(float_bits(b) | 0x00400000U);
    return a * b;
}

// gfx1151 expf on the host: (float)exp((double)x) except the inputs listed in miss_expf.bin (inputs then outputs).
class HostExpf {
public:
    explicit HostExpf(const std::string& path) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("missing " + path);
        _fseeki64(f, 0, SEEK_END);
        const long long n = _ftelli64(f);
        _fseeki64(f, 0, SEEK_SET);
        std::vector<std::uint32_t> raw(static_cast<std::size_t>(n / 4));
        if (std::fread(raw.data(), 4, raw.size(), f) != raw.size()) throw std::runtime_error("short " + path);
        std::fclose(f);
        const std::size_t m = raw.size() / 2;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> v(m);
        for (std::size_t i = 0; i < m; ++i) v[i] = {raw[i], raw[m + i]};
        raw.clear(); raw.shrink_to_fit();
        if (!std::is_sorted(v.begin(), v.end())) std::sort(v.begin(), v.end());
        in_.resize(m); out_.resize(m);
        for (std::size_t i = 0; i < m; ++i) { in_[i] = v[i].first; out_[i] = v[i].second; }
    }
    [[nodiscard]] std::size_t size() const { return in_.size(); }
    float operator()(float x) const {
        const std::uint32_t b = float_bits(x);
        const auto it = std::lower_bound(in_.begin(), in_.end(), b);
        if (it != in_.end() && *it == b) return bits_float(out_[static_cast<std::size_t>(it - in_.begin())]);
        return static_cast<float>(std::exp(static_cast<double>(x)));
    }
private:
    std::vector<std::uint32_t> in_, out_;
};

// rows_shared_activation_scalar_kernel for one row (workspace: gate[I], up[I]; input BF16 [H]).
template <class Expf>
void shared_activation_row(float* ws, std::uint32_t I, const std::uint16_t* input, const float* scalar, std::uint32_t H,
                           const Expf& expf_) {
    for (std::uint32_t i = 0; i < I; ++i) {
        const float g = ws[i], u = ws[I + i];
        const float den = 1.0F + expf_(-g);
        const float q = g / den;
        ws[i] = q * u;
    }
    float scratch[256];
    for (std::uint32_t t = 0; t < 256U; ++t) {
        float sum = 0.0F;
        for (std::uint32_t f = t; f < H; f += 256U) sum = std::fmaf(scalar[f], bf2f(input[f]), sum);
        scratch[t] = sum;
    }
    for (std::uint32_t s = 128U; s > 0U; s >>= 1U)
        for (std::uint32_t t = 0; t < s; ++t) scratch[t] = scratch[t + s] + scratch[t];
    ws[I] = 1.0F / (1.0F + expf_(-scratch[0]));
}

}  // namespace strix::qwen_dense_ref
