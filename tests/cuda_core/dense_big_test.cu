// Large-M dense GEMM (dense_big.hpp): BITWISE equality with the decode GEMV/mma kernels (dense_rows, dense_gate_up_silu_rows,
// nvfp4_dense_rows) row by row for bf16, q8_0 and nvfp4, with 3-term (exact) activation splits, plus FAST terms 2/1 against
// dense_rows under the matching arithmetic, residual and silu_mul epilogues. Run: cuda_core_dense_big_test [--bench]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/dense_big.hpp"
#include "strix/cuda_core/nvfp4.hpp"

namespace cc = strix::cuda_core;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

static int g_fail = 0;
static void expect(bool ok, const std::string& what) {
    if (!ok) { ++g_fail; std::printf("  FAIL: %s\n", what.c_str()); }
}

template <class T>
static T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    CK(cudaMalloc(reinterpret_cast<void**>(&d), h.size() * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

struct W {
    cc::DenseWeight w;
    void* mem = nullptr;
};

static W make_weight(cc::WeightFormat f, std::uint32_t out, std::uint32_t in, std::mt19937& rng) {
    W r;
    r.w.format = f;
    r.w.out = out;
    r.w.in = in;
    std::normal_distribution<float> nd(0.0f, 0.02f);
    if (f == cc::WeightFormat::bf16 || f == cc::WeightFormat::nvfp4) {
        std::vector<std::uint16_t> h(static_cast<std::size_t>(out) * in);
        for (auto& v : h) { const __nv_bfloat16 b = __float2bfloat16_rn(nd(rng)); v = __bfloat16_as_ushort(b); }
        std::uint16_t* d = dev(h);
        if (f == cc::WeightFormat::bf16) {
            r.mem = d;
            r.w.data = d;
        } else {
            void* t = nullptr;
            CK(cudaMalloc(&t, cc::nvfp4_bytes(out, in)));
            float gs = 1.0f;
            CK(cc::nvfp4_quantize_bf16(d, out, in, t, &gs, nullptr));
            CK(cudaDeviceSynchronize());
            CK(cudaFree(d));
            r.mem = t;
            r.w.data = t;
            r.w.gscale = gs;
        }
    } else {
        const std::size_t nb = static_cast<std::size_t>(out) * (in / 32);
        std::vector<unsigned char> h(nb * 34);
        for (std::size_t b = 0; b < nb; ++b) {
            const __half d = __float2half_rn(std::abs(nd(rng)) * 0.05f + 1e-4f);
            std::memcpy(&h[b * 34], &d, 2);
            for (int i = 0; i < 32; ++i) h[b * 34 + 2 + i] = static_cast<unsigned char>(static_cast<int>(rng() % 255) - 127);
        }
        r.mem = dev(h);
        r.w.data = r.mem;
    }
    return r;
}

static std::vector<float> make_x(std::uint32_t rows, std::uint32_t in, std::mt19937& rng) {
    std::vector<float> x(static_cast<std::size_t>(rows) * in);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (std::size_t i = 0; i < x.size(); ++i) {
        float v = nd(rng) * (1.0f + static_cast<float>(i % 7));
        if (i % 97 == 0) v *= 40.0f;
        x[i] = v;
    }
    return x;
}

// windowed reference: dense_rows in windows of 32 rows
static void ref_rows(const cc::DenseWeight& w, const float* x, float* y, std::uint32_t rows, const float* resid) {
    for (std::uint32_t r = 0; r < rows; r += 32) {
        const std::uint32_t n = std::min<std::uint32_t>(32, rows - r);
        CK(cc::dense_rows(w, x + static_cast<std::size_t>(r) * w.in, y + static_cast<std::size_t>(r) * w.out, n,
                          resid ? resid + static_cast<std::size_t>(r) * w.out : nullptr, nullptr));
    }
}

static std::size_t diff_count(const std::vector<float>& a, const std::vector<float>& b) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], 4) != 0) ++n;
    return n;
}

static void test_one(cc::WeightFormat f, std::uint32_t out, std::uint32_t in, std::uint32_t rows, int terms, std::mt19937& rng) {
    const char* fn = f == cc::WeightFormat::bf16 ? "bf16" : f == cc::WeightFormat::q8_0 ? "q8_0" : "nvfp4";
    std::printf("%s out %u in %u rows %u terms %d\n", fn, out, in, rows, terms);
    const cc::DenseArith saved = cc::dense_arith();
    cc::set_dense_arith(terms == 3 ? cc::DenseArith::mma3 : terms == 2 ? cc::DenseArith::mma2 : cc::DenseArith::mma1);
    W w = make_weight(f, out, in, rng);
    const std::vector<float> hx = make_x(rows, in, rng);
    float* x = dev(hx);
    std::vector<float> hr(static_cast<std::size_t>(rows) * out);
    for (auto& v : hr) v = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) * 0.01f;
    float* resid = dev(hr);
    float *yref = nullptr, *ybig = nullptr, *yref2 = nullptr;
    CK(cudaMalloc(reinterpret_cast<void**>(&yref), hr.size() * 4));
    CK(cudaMalloc(reinterpret_cast<void**>(&yref2), hr.size() * 4));
    CK(cudaMalloc(reinterpret_cast<void**>(&ybig), hr.size() * 4));
    std::vector<float> a(hr.size()), b(hr.size());

    cc::BigAct act;
    CK(cc::dense_big_split(x, rows, in, f, terms, &act, nullptr));
    // plain
    ref_rows(w.w, x, yref, rows, nullptr);
    CK(cc::dense_big_gemm(w.w, act, ybig, nullptr, cc::BigEpi::none, nullptr));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(a.data(), yref, a.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), ybig, b.size() * 4, cudaMemcpyDeviceToHost));
    std::size_t d = diff_count(a, b);
    std::printf("  plain: %zu / %zu elements differ\n", d, a.size());
    expect(d == 0, std::string(fn) + " plain bitwise");
    // residual (in place)
    CK(cudaMemcpy(ybig, resid, hr.size() * 4, cudaMemcpyDeviceToDevice));
    ref_rows(w.w, x, yref, rows, resid);
    CK(cc::dense_big_gemm(w.w, act, ybig, ybig, cc::BigEpi::residual, nullptr));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(a.data(), yref, a.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), ybig, b.size() * 4, cudaMemcpyDeviceToHost));
    d = diff_count(a, b);
    std::printf("  residual: %zu differ\n", d);
    expect(d == 0, std::string(fn) + " residual bitwise");
    // gate/up silu: gate = w, up = second weight; reference dense_gate_up_silu_rows in windows
    {
        W up = make_weight(f, out, in, rng);
        for (std::uint32_t r = 0; r < rows; r += 32)
            CK(cc::dense_gate_up_silu_rows(w.w, up.w, x + static_cast<std::size_t>(r) * in, yref2 + static_cast<std::size_t>(r) * out,
                                           std::min<std::uint32_t>(32, rows - r), nullptr));
        CK(cc::dense_big_gemm(w.w, act, ybig, nullptr, cc::BigEpi::none, nullptr));  // gate sums
        float* y3 = nullptr;
        CK(cudaMalloc(reinterpret_cast<void**>(&y3), hr.size() * 4));
        CK(cc::dense_big_gemm(up.w, act, y3, ybig, cc::BigEpi::silu_mul, nullptr));
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(a.data(), yref2, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), y3, b.size() * 4, cudaMemcpyDeviceToHost));
        d = diff_count(a, b);
        std::printf("  gate/up silu: %zu differ\n", d);
        expect(d == 0, std::string(fn) + " gate_up_silu bitwise");
        CK(cudaFree(y3));
        CK(cudaFree(up.mem));
    }
    cc::set_dense_arith(saved);
    CK(cudaFree(x)); CK(cudaFree(resid)); CK(cudaFree(yref)); CK(cudaFree(yref2)); CK(cudaFree(ybig)); CK(cudaFree(w.mem));
}

// FAST/TURBO e4m3 x e4m3 GEMM vs the exact nvfp4 big GEMM: relative RMS error (activation e4m3 per row + weight rounded to e4m3).
static void test_f8(std::uint32_t out, std::uint32_t in, std::uint32_t rows, std::mt19937& rng) {
    std::printf("nvfp4 f8 out %u in %u rows %u\n", out, in, rows);
    W w = make_weight(cc::WeightFormat::nvfp4, out, in, rng);
    const std::vector<float> hx = make_x(rows, in, rng);
    float* x = dev(hx);
    float *y0 = nullptr, *y8 = nullptr;
    CK(cudaMalloc(reinterpret_cast<void**>(&y0), static_cast<std::size_t>(rows) * out * 4));
    CK(cudaMalloc(reinterpret_cast<void**>(&y8), static_cast<std::size_t>(rows) * out * 4));
    cc::BigAct a0, a8;
    CK(cc::dense_big_split(x, rows, in, cc::WeightFormat::nvfp4, 1, &a0, nullptr, nullptr, false));
    CK(cc::dense_big_gemm(w.w, a0, y0, nullptr, cc::BigEpi::none, nullptr));
    CK(cc::dense_big_split(x, rows, in, cc::WeightFormat::nvfp4, 1, &a8, nullptr, nullptr, true));
    CK(cc::dense_big_gemm(w.w, a8, y8, nullptr, cc::BigEpi::none, nullptr));
    CK(cudaDeviceSynchronize());
    std::vector<float> a(static_cast<std::size_t>(rows) * out), b(a.size());
    CK(cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), y8, b.size() * 4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0;
    for (std::size_t i = 0; i < a.size(); ++i) { const double d = double(a[i]) - b[i]; num += d * d; den += double(a[i]) * a[i]; }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    std::printf("  f8 rel RMS vs f16 path: %.4f\n", rel);
    {   // clustering: per-output and per-row relative RMS (uniform = precision, a few outliers = bug)
        std::vector<double> cn(out, 0.0), cd(out, 0.0), rn(rows, 0.0), rd(rows, 0.0);
        for (std::uint32_t r = 0; r < rows; ++r)
            for (std::uint32_t o = 0; o < out; ++o) {
                const double d = double(a[std::size_t(r) * out + o]) - b[std::size_t(r) * out + o], v = a[std::size_t(r) * out + o];
                cn[o] += d * d; cd[o] += v * v; rn[r] += d * d; rd[r] += v * v;
            }
        auto stat = [](const std::vector<double>& n, const std::vector<double>& d, const char* what) {
            std::vector<double> e(n.size());
            for (std::size_t i = 0; i < n.size(); ++i) e[i] = std::sqrt(n[i] / std::max(d[i], 1e-30));
            std::sort(e.begin(), e.end());
            std::printf("  %s rel RMS: min %.4f p50 %.4f p99 %.4f max %.4f\n", what, e.front(), e[e.size() / 2], e[e.size() * 99 / 100], e.back());
        };
        stat(cn, cd, "per-output");
        stat(rn, rd, "per-row");
    }
    expect(a8.f8 && rel < 0.08, "nvfp4 f8 relative RMS < 0.08");
    CK(cudaFree(x)); CK(cudaFree(y0)); CK(cudaFree(y8)); CK(cudaFree(w.mem));
}

static void bench(std::mt19937& rng) {
    struct Sh { const char* n; std::uint32_t out, in; };
    const Sh shapes[] = {{"qkv 10240x5120", 10240, 5120}, {"gate 17408x5120", 17408, 5120}, {"down 5120x17408", 5120, 17408}};
    const std::uint32_t rows = 4096;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (auto f : {cc::WeightFormat::bf16, cc::WeightFormat::q8_0, cc::WeightFormat::nvfp4})
        for (const auto& s : shapes) {
            for (int terms : {3, 1, 0}) {
                if (f == cc::WeightFormat::nvfp4 && terms != 3 && terms != 0) continue;
                if (f != cc::WeightFormat::nvfp4 && terms == 0) continue;
                const bool f8 = terms == 0;
                if (f8) terms = 1;
                W w = make_weight(f, s.out, s.in, rng);
                const std::vector<float> hx = make_x(rows, s.in, rng);
                float* x = dev(hx);
                float* y = nullptr;
                CK(cudaMalloc(reinterpret_cast<void**>(&y), static_cast<std::size_t>(rows) * s.out * 4));
                cc::BigAct act;
                CK(cc::dense_big_split(x, rows, s.in, f, terms, &act, nullptr, nullptr, f8));
                CK(cc::dense_big_gemm(w.w, act, y, nullptr, cc::BigEpi::none, nullptr));
                CK(cudaDeviceSynchronize());
                float split_ms = 0, gemm_ms = 0;
                const int reps = 3;
                CK(cudaEventRecord(e0));
                for (int i = 0; i < reps; ++i) CK(cc::dense_big_split(x, rows, s.in, f, terms, &act, nullptr, nullptr, f8));
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&split_ms, e0, e1));
                CK(cudaEventRecord(e0));
                for (int i = 0; i < reps; ++i) CK(cc::dense_big_gemm(w.w, act, y, nullptr, cc::BigEpi::none, nullptr));
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&gemm_ms, e0, e1));
                const double fl = 2.0 * rows * s.out * s.in;
                std::printf("bench %-5s %-16s %s terms %d rows %u: gemm+expand %.3f ms (%.1f TFLOP/s weight-flops, %.1f mma-flops), split %.3f ms\n",
                            f == cc::WeightFormat::bf16 ? "bf16" : f == cc::WeightFormat::q8_0 ? "q8_0" : "nvfp4", s.n, f8 ? "F8" : "  ", f == cc::WeightFormat::nvfp4 ? 1 : terms, rows,
                            gemm_ms / reps, fl / (gemm_ms / reps) / 1e9, fl * (f == cc::WeightFormat::nvfp4 ? 1 : terms) / (gemm_ms / reps) / 1e9, split_ms / reps);
                CK(cudaFree(x)); CK(cudaFree(y)); CK(cudaFree(w.mem));
            }
        }
}

int main(int argc, char** argv) {
    CK(cudaSetDevice(0));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    std::printf("device 0: %s cc %d.%d SMs %d\n", p.name, p.major, p.minor, p.multiProcessorCount);
    CK(cc::nvfp4_reserve_workspace(256u << 20));
    std::mt19937 rng(4321);
    const bool bench_only = argc > 1 && std::string(argv[1]) == "--bench-only";
    if (!bench_only) {
        for (auto f : {cc::WeightFormat::bf16, cc::WeightFormat::q8_0, cc::WeightFormat::nvfp4}) {
            test_one(f, 1040, 5120, 300, 3, rng);   // out not a multiple of 128, rows not a multiple of the tile
            test_one(f, 48, 5120, 37, 3, rng);      // tiny out (GDN beta/alpha)
            test_one(f, 5120, 17408, 200, 3, rng);  // down projection
            if (f != cc::WeightFormat::nvfp4) {
                test_one(f, 256, 4096, 130, 2, rng);
                test_one(f, 256, 4096, 130, 1, rng);
            }
        }
        test_f8(1040, 5120, 300, rng);
        test_f8(5120, 17408, 200, rng);
        test_f8(17408, 5120, 4096, rng);
        std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    }
    if (bench_only || (argc > 1 && std::string(argv[1]) == "--bench")) bench(rng);
    cc::dense_big_release();
    return g_fail ? 1 : 0;
}
