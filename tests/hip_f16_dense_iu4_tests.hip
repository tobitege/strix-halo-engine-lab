// Shared policy dense-iu4a16-f16-v1 (src/core/f16_dense_iu4.hpp):
//  1. weight decode: every FP16 scale (65,536) x every code (16) through Cols::stage from memory (a check on register
//     words is not equivalent: the compiler folds their known bits), bit for bit vs host_w16 (any NaN matches any NaN);
//  2. a GEMM with a partial row CTA, a remainder column launch and a partial last column tile: error vs a double sum
//     of the contract's x16 * w16 products (tolerance 2^-10 of sum |terms|: FP32 WMMA accumulation), columns past
//     count untouched; 16-wave CTAs give the same bits as 8-wave CTAs.
#include "core/f16_dense_iu4.hpp"
#include "core/iu4.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace iu = strix::core::f16_dense_tall::iu4a16;
namespace tall = strix::core::f16_dense_tall;

#define CHECK(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s at %d\n", hipGetErrorString(e_), __LINE__); return 1; } } while (0)

__global__ void decode_kernel(iu::View v, std::uint16_t* out) {
    const unsigned o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= v.count) return;
    tall::U4 nb[8];
    iu::Cols<8>{v}.stage(o / 16U, o % 16U, 0U, nb);
    for (unsigned q = 0; q < 8U; ++q)
        for (unsigned p = 0; p < 4U; ++p) {
            out[std::size_t(o) * 64U + 8U * q + 2U * p] = std::uint16_t(nb[q][p] & 0xffffU);
            out[std::size_t(o) * 64U + 8U * q + 2U * p + 1U] = std::uint16_t(nb[q][p] >> 16U);
        }
}

static bool is_nan16(std::uint16_t b) { return (b & 0x7c00U) == 0x7c00U && (b & 0x3ffU); }

int main() {
    int failures = 0;
    {  // 1. decode: column o has scale o, codes (k % 16) - 8 at k = 0..63
        const unsigned N = 65536U, K = 64U;
        std::vector<std::uint8_t> w(std::size_t(N) * K / 2U + std::size_t(N) * 2U, 0U);
        for (unsigned o = 0; o < N; ++o)
            for (unsigned k = 0; k < K; ++k) {
                auto& byte = w[strix::core::iu4_weight_byte(int(k), int(o), int(N))];
                strix::core::iu4_put(byte, int(k % 2U), int(k % 16U) - 8);
            }
        auto* sc = reinterpret_cast<std::uint16_t*>(w.data() + std::size_t(N) * K / 2U);
        for (unsigned o = 0; o < N; ++o) sc[o] = std::uint16_t(o);
        std::uint8_t* dw = nullptr; std::uint16_t* dout = nullptr;
        CHECK(hipMalloc(&dw, w.size())); CHECK(hipMalloc(&dout, std::size_t(N) * K * 2U));
        CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
        iu::View v{};
        v.weight = dw; v.scale = reinterpret_cast<const std::uint16_t*>(dw + std::size_t(N) * K / 2U);
        v.in = K; v.n = N; v.count = N; v.rows = 1U; v.out_stride = N;
        hipLaunchKernelGGL(decode_kernel, dim3(N / 256U), dim3(256), 0, 0, v, dout);
        CHECK(hipGetLastError());
        std::vector<std::uint16_t> got(std::size_t(N) * K);
        CHECK(hipMemcpy(got.data(), dout, got.size() * 2U, hipMemcpyDeviceToHost));
        unsigned bad = 0;
        for (unsigned o = 0; o < N; ++o)
            for (unsigned k = 0; k < K; ++k) {
                const std::uint16_t want = __builtin_bit_cast(std::uint16_t, iu::host_w16(int(k % 16U) - 8, std::uint16_t(o)));
                const std::uint16_t g = got[std::size_t(o) * K + k];
                if (g != want && !(is_nan16(g) && is_nan16(want)) && bad++ < 5)
                    std::printf("decode scale 0x%04x code %d: got 0x%04x want 0x%04x\n", o, int(k % 16U) - 8, g, want);
            }
        std::printf("decode: %u mismatches of %u\n", bad, N * K);
        failures += bad != 0U;
        CHECK(hipFree(dw)); CHECK(hipFree(dout));
    }
    {  // 2. GEMM
        const unsigned R = 200U, K = 256U, N = 80U, C = 72U;
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> ux(-2.0F, 2.0F), us(0.001F, 0.02F);
        std::uniform_int_distribution<int> uc(-8, 7);
        std::vector<float> x(std::size_t(R) * K);
        for (auto& f : x) f = ux(rng);
        std::vector<int> code(std::size_t(N) * K);
        for (auto& c : code) c = uc(rng);
        std::vector<std::uint8_t> w(std::size_t(N) * K / 2U + std::size_t(N) * (K / 64U) * 2U, 0U);
        for (unsigned o = 0; o < N; ++o)
            for (unsigned k = 0; k < K; ++k)
                strix::core::iu4_put(w[strix::core::iu4_weight_byte(int(k), int(o), int(N))], int(k % 2U), code[std::size_t(o) * K + k]);
        auto* sc = reinterpret_cast<std::uint16_t*>(w.data() + std::size_t(N) * K / 2U);
        for (unsigned i = 0; i < N * (K / 64U); ++i) sc[i] = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(us(rng)));
        std::uint8_t* dw = nullptr; float *dx = nullptr, *dy = nullptr;
        CHECK(hipMalloc(&dw, w.size())); CHECK(hipMalloc(&dx, x.size() * 4U)); CHECK(hipMalloc(&dy, std::size_t(R) * N * 4U));
        CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dx, x.data(), x.size() * 4U, hipMemcpyHostToDevice));
        iu::View v{};
        v.weight = dw; v.scale = reinterpret_cast<const std::uint16_t*>(dw + std::size_t(N) * K / 2U);
        v.x = dx; v.out = dy; v.in = K; v.n = N; v.count = C; v.rows = R; v.out_stride = N;
        if (!iu::supported(v)) { std::printf("GEMM view not supported\n"); return 1; }
        std::vector<float> y8(std::size_t(R) * N), y16(std::size_t(R) * N);
        for (unsigned waves : {8U, 16U}) {
            std::vector<float> sentinel(std::size_t(R) * N, -12345.0F);
            CHECK(hipMemcpy(dy, sentinel.data(), sentinel.size() * 4U, hipMemcpyHostToDevice));
            iu::enqueue(v, 0, waves);
            CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
            CHECK(hipMemcpy((waves == 8U ? y8 : y16).data(), dy, y8.size() * 4U, hipMemcpyDeviceToHost));
        }
        double worst = 0.0;
        unsigned untouched_bad = 0;
        for (unsigned r = 0; r < R; ++r)
            for (unsigned o = 0; o < N; ++o) {
                const float got = y8[std::size_t(r) * N + o];
                if (o >= C) { untouched_bad += got != -12345.0F; continue; }
                double sum = 0.0, mag = 0.0;
                for (unsigned k = 0; k < K; ++k) {
                    const double t = double(static_cast<float>(static_cast<_Float16>(x[std::size_t(r) * K + k]))) *
                        double(static_cast<float>(iu::host_w16(code[std::size_t(o) * K + k], sc[o * (K / 64U) + k / 64U])));
                    sum += t, mag += std::fabs(t);
                }
                worst = std::max(worst, std::fabs(double(got) - sum) / (mag > 0.0 ? mag : 1.0));
            }
        const bool same = std::memcmp(y8.data(), y16.data(), y8.size() * 4U) == 0;
        std::printf("gemm: worst |err|/sum|terms| %.3g, columns past count written %u, 8 vs 16 waves %s\n", worst, untouched_bad,
                    same ? "bitwise" : "DIFFER");
        failures += !(worst < 1.0 / 1024.0) + (untouched_bad != 0U) + !same;
        CHECK(hipFree(dw)); CHECK(hipFree(dx)); CHECK(hipFree(dy));
    }
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
