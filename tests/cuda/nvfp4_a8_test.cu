// nvfp4-a8-v1 (strix/cuda/core/nvfp4_a8.hpp): device converter == host oracle (#683 quantizer), Qwen dense binding ==
// host GEMV oracle bit for bit at every row count (chunks of 32 included), rows-invariant, and the exact Q8_0 route
// unchanged outside the OnflyScope.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "strix/cuda/core/nvfp4.hpp"
#include "strix/cuda/core/nvfp4_a8.cuh"
#include "strix/cuda/core/nvfp4_a8_cuda.hpp"
#include "strix/cuda/qwen_dense.hpp"

namespace a8 = strix::cuda::core::nvfp4_a8;
namespace dn = strix::cuda::qwen_dense;

static int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { ++failures; std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
#define CK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); return 2; } } while (0)

static std::uint16_t half_bits(float f) {   // positive normal / zero only (test deltas)
    if (f == 0) return 0;
    int e;
    const float m = std::frexp(f, &e);   // f = m * 2^e, m in [.5, 1)
    const auto man = static_cast<std::uint32_t>(std::lround((m * 2 - 1) * 1024));
    return static_cast<std::uint16_t>(((e - 1 + 15) << 10) | (man & 1023U));
}

static std::vector<std::uint8_t> make_q8(std::uint32_t n, std::uint32_t k, std::mt19937& rng) {
    std::vector<std::uint8_t> q(std::size_t{n} * (k / 32) * 34);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> d(1e-4F, 3e-2F);
    for (std::size_t b = 0; b < q.size() / 34; ++b) {
        std::uint16_t h = half_bits(d(rng));
        if (b % 97 == 5) h = 0;               // zero block (scale 0 codes)
        if (b % 89 == 7) h = 0x8000U | h;     // negative delta: -0 values where q == 0
        std::memcpy(&q[b * 34], &h, 2);
        for (int j = 0; j < 32; ++j) q[b * 34 + 2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(b % 61 == 3 && j < 16 ? 0 : code(rng)));
    }
    return q;
}

static int test_convert(std::uint32_t n, std::uint32_t k, std::mt19937& rng) {
    const auto q8 = make_q8(n, k, rng);
    std::vector<std::uint8_t> hc(std::size_t{n} * k / 2), hs(std::size_t{n} * k / 16);
    float hts = 0;
    a8::reference_convert_q8_0(q8.data(), n, k, hc.data(), hs.data(), &hts);
    void* dq = nullptr; float* scratch = nullptr;
    a8::Image img;
    CK(cudaMalloc(&dq, q8.size()));
    CK(cudaMalloc(&img.codes, hc.size()));
    CK(cudaMalloc(&img.scales, hs.size()));
    CK(cudaMalloc(&scratch, 4));
    CK(cudaMemcpy(dq, q8.data(), q8.size(), cudaMemcpyHostToDevice));
    CK(a8::convert_q8_0(dq, n, k, img, scratch, nullptr));
    CK(cudaDeviceSynchronize());
    std::vector<std::uint8_t> dc(hc.size()), ds(hs.size());
    CK(cudaMemcpy(dc.data(), img.codes, dc.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ds.data(), img.scales, ds.size(), cudaMemcpyDeviceToHost));
    CHECK(std::memcmp(&hts, &img.tensor_scale, 4) == 0, "tensor scale %g vs %g", hts, img.tensor_scale);
    std::size_t bad_c = 0, bad_s = 0;
    for (std::size_t i = 0; i < dc.size(); ++i) bad_c += dc[i] != hc[i];
    for (std::size_t i = 0; i < ds.size(); ++i) bad_s += ds[i] != hs[i];
    CHECK(bad_c == 0 && bad_s == 0, "convert %ux%u: %zu code / %zu scale bytes differ", n, k, bad_c, bad_s);
    std::printf("convert %ux%u ts %.6g: codes/scales == host oracle (#683 quantize)\n", n, k, hts);
    cudaFree(dq); cudaFree(img.codes); cudaFree(img.scales); cudaFree(scratch);
    return 0;
}

static int test_project(std::mt19937& rng) {
    constexpr std::uint32_t k = 2560, n0 = 96, n1 = 64, maxr = 40;
    const auto q0 = make_q8(n0, k, rng), q1 = make_q8(n1, k, rng);
    void *d0, *d1, *in, *out, *out_ref;
    std::int8_t* codes; float *deltas, *scratch;
    a8::Image i0, i1;
    CK(cudaMalloc(&d0, q0.size())); CK(cudaMalloc(&d1, q1.size()));
    CK(cudaMemcpy(d0, q0.data(), q0.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d1, q1.data(), q1.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&i0.codes, n0 * k / 2)); CK(cudaMalloc(&i0.scales, n0 * k / 16));
    CK(cudaMalloc(&i1.codes, n1 * k / 2)); CK(cudaMalloc(&i1.scales, n1 * k / 16));
    CK(cudaMalloc(&scratch, 4));
    CK(a8::convert_q8_0(d0, n0, k, i0, scratch, nullptr));
    CK(a8::convert_q8_0(d1, n1, k, i1, scratch, nullptr));
    std::vector<float> x(std::size_t{maxr} * k);
    std::normal_distribution<float> nd(0, 1);
    for (auto& v : x) v = nd(rng);
    for (std::uint32_t j = 0; j < 32; ++j) x[std::size_t{5} * k + 64 + j] = 0;   // a zero activation block
    CK(cudaMalloc(&in, x.size() * 4)); CK(cudaMemcpy(in, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&codes, std::size_t{maxr} * k)); CK(cudaMalloc(&deltas, std::size_t{k / 32} * maxr * 4));
    CK(cudaMalloc(&out, std::size_t{maxr} * (n0 + n1) * 4)); CK(cudaMalloc(&out_ref, std::size_t{maxr} * (n0 + n1) * 4));
    dn::Site site;
    site.store = dn::Store::f32;
    site.width = k;
    site.segments = 2;
    site.segment[0] = {d0, n0, out, n0 + n1};
    site.segment[1] = {d1, n1, static_cast<float*>(out) + n0, n0 + n1};
    std::vector<std::uint8_t> hc0(n0 * k / 2), hs0(n0 * k / 16), hc1(n1 * k / 2), hs1(n1 * k / 16);
    CK(cudaMemcpy(hc0.data(), i0.codes, hc0.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs0.data(), i0.scales, hs0.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hc1.data(), i1.codes, hc1.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs1.data(), i1.scales, hs1.size(), cudaMemcpyDeviceToHost));
    dn::onfly_register(d0, {i0.codes, i0.scales, i0.tensor_scale, n0, k});
    dn::onfly_register(d1, {i1.codes, i1.scales, i1.tensor_scale, n1, k});
    std::vector<float> row1;   // rows-invariance: row 0 of every R equals row 0 of R = 1
    for (const std::uint32_t rows : {1U, 2U, 3U, 8U, 9U, 13U, 16U, 17U, 32U, 33U, 40U}) {
        const std::uint32_t pitch = rows;
        // Exact Q8_0 route first (no scope): must not see the registrations.
        CK(dn::quantize_project(nullptr, site, rows, dn::Input::f32, in, k, codes, deltas, pitch));
        CK(cudaMemcpy(out_ref, out, std::size_t{rows} * (n0 + n1) * 4, cudaMemcpyDeviceToDevice));
        dn::onfly_clear();
        CK(dn::quantize_project(nullptr, site, rows, dn::Input::f32, in, k, codes, deltas, pitch));
        std::vector<float> e1(std::size_t{rows} * (n0 + n1)), e2(e1.size()), t(e1.size());
        CK(cudaMemcpy(e1.data(), out_ref, e1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(e2.data(), out, e2.size() * 4, cudaMemcpyDeviceToHost));
        CHECK(std::memcmp(e1.data(), e2.data(), e1.size() * 4) == 0, "rows %u: Q8_0 route changed by registration", rows);
        dn::onfly_register(d0, {i0.codes, i0.scales, i0.tensor_scale, n0, k});
        dn::onfly_register(d1, {i1.codes, i1.scales, i1.tensor_scale, n1, k});
        {
            const dn::OnflyScope scope(true);
            CK(dn::quantize_project(nullptr, site, rows, dn::Input::f32, in, k, codes, deltas, pitch));
        }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(t.data(), out, t.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<std::int8_t> hcodes(std::size_t{rows} * k);
        std::vector<float> hdel(std::size_t{k / 32} * pitch);
        CK(cudaMemcpy(hcodes.data(), codes, hcodes.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hdel.data(), deltas, hdel.size() * 4, cudaMemcpyDeviceToHost));
        std::size_t bad = 0;
        double err = 0, ref = 0;
        for (std::uint32_t r = 0; r < rows; ++r) {
            std::vector<float> dr(k / 32);
            for (std::uint32_t b = 0; b < k / 32; ++b) dr[b] = hdel[std::size_t{b} * pitch + r];
            for (std::uint32_t o = 0; o < n0 + n1; ++o) {
                const bool s1 = o >= n0;
                const std::uint32_t i = s1 ? o - n0 : o;
                const float y = a8::reference_dot((s1 ? hc1 : hc0).data() + std::size_t{i} * k / 2,
                                                  (s1 ? hs1 : hs0).data() + std::size_t{i} * k / 16,
                                                  s1 ? i1.tensor_scale : i0.tensor_scale, hcodes.data() + std::size_t{r} * k,
                                                  dr.data(), k);
                const float got = t[std::size_t{r} * (n0 + n1) + o];
                bad += std::memcmp(&y, &got, 4) != 0;
                err += std::fabs(double(got) - double(e1[std::size_t{r} * (n0 + n1) + o]));
                ref += std::fabs(double(e1[std::size_t{r} * (n0 + n1) + o]));
            }
        }
        CHECK(bad == 0, "rows %u: %zu outputs differ from the host oracle", rows, bad);
        for (const std::uint32_t rw : {1U, 2U, 4U, 8U})   // launch shape = parallelism only: same bits
            for (const std::uint32_t ow : {1U, 2U}) {
                dn::onfly_set_launch(rw, ow);
                {
                    const dn::OnflyScope scope(true);
                    CK(dn::project(nullptr, site, rows, codes, deltas, pitch));
                }
                std::vector<float> u(t.size());
                CK(cudaMemcpy(u.data(), out, u.size() * 4, cudaMemcpyDeviceToHost));
                CHECK(std::memcmp(u.data(), t.data(), u.size() * 4) == 0, "rows %u: launch shape %u/%u changes bits", rows, rw, ow);
            }
        dn::onfly_set_launch(0, 0);
        if (row1.empty()) row1.assign(t.begin(), t.begin() + (n0 + n1));
        CHECK(std::memcmp(row1.data(), t.data(), (n0 + n1) * 4) == 0, "rows %u: row 0 differs from rows = 1", rows);
        std::printf("rows %2u: nvfp4-a8 == host oracle (%u outputs), rows-invariant; |nvfp4 - q8| / |q8| = %.4f\n", rows,
                    rows * (n0 + n1), err / ref);
    }
    dn::onfly_clear();
    return 0;
}

int main() {
    std::mt19937 rng(1234);
    // Device decode tables == #683 host functions.
    for (std::uint32_t c = 0; c < 16; ++c) CHECK(a8::dev_e2m1(c) == strix::cuda::core::nvfp4::e2m1(static_cast<std::uint8_t>(c)), "e2m1 %u", c);
    for (std::uint32_t c = 0; c < 127; ++c) CHECK(a8::dev_e4m3(c) == strix::cuda::core::nvfp4::e4m3(static_cast<std::uint8_t>(c)), "e4m3 %u", c);
    if (int rc = test_convert(48, 2560, rng)) return rc;
    if (int rc = test_convert(40, 640, rng)) return rc;
    if (int rc = test_convert(16, 6144, rng)) return rc;
    if (int rc = test_project(rng)) return rc;
    std::printf(failures ? "nvfp4_a8_test: %d FAILURES\n" : "nvfp4_a8_test: PASS\n", failures);
    return failures ? 1 : 0;
}
