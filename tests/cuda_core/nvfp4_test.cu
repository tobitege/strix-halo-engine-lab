// NVFP4 (TURBO weights): converter vs host RTN reference, GEMV vs host dequant reference, rows 1..32 bitwise row
// invariance, and a bandwidth microbench on the 27B shapes. Run: cuda_core_nvfp4_test [--bench-only|--test-only].
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "strix/cuda_core/nvfp4.hpp"

namespace cc = strix::cuda_core;
#define CK(x)                                                                                              \
    do {                                                                                                   \
        cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) {                                                                           \
            std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);          \
            std::exit(2);                                                                                  \
        }                                                                                                  \
    } while (0)

static int g_fail = 0;
static void expect(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}
static float h2f(float v) { return __half2float(__float2half_rn(v)); }
static const float kV[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

static void test_shape(std::uint32_t out, std::uint32_t in, std::mt19937& rng) {
    std::printf("-- shape %u x %u\n", out, in);
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> w(static_cast<std::size_t>(out) * in);
    for (auto& v : w) v = nd(rng);
    for (int i = 0; i < 40; ++i) w[rng() % w.size()] *= 30.f;  // outliers
    for (std::uint32_t k = 0; k < 16; ++k) w[k] = 0.f;          // exact-zero group
    for (std::uint32_t k = 16; k < 32; ++k) w[k] *= 1e-6f;      // tiny group (scale underflow)
    float* d_w;
    CK(cudaMalloc(&d_w, w.size() * 4));
    CK(cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
    void* d_t;
    CK(cudaMalloc(&d_t, cc::nvfp4_bytes(out, in)));
    float gs = 0;
    CK(cc::nvfp4_quantize_f32(d_w, out, in, d_t, &gs, 0));
    std::vector<unsigned char> tiles(cc::nvfp4_bytes(out, in));
    CK(cudaMemcpy(tiles.data(), d_t, tiles.size(), cudaMemcpyDeviceToHost));
    float amax = 0;
    for (float v : w) amax = std::max(amax, std::fabs(v));
    expect(gs == amax / (6.0f * 448.0f), "global scale = amax/(6*448)");
    std::size_t mism = 0, scmis = 0;
    double se = 0, sn = 0;
    float maxsc = 0;
    std::vector<float> deq(w.size());
    for (std::uint32_t o = 0; o < out; ++o)
        for (std::uint32_t G = 0; G < in / 16; ++G) {
            unsigned char codes[16];
            const unsigned char sb = cc::nvfp4_host_quantize_group(&w[static_cast<std::size_t>(o) * in + G * 16], gs, codes);
            const std::uint32_t m = o >> 4, r = o & 15, h = r >> 3, g = r & 7, c = G >> 3, j = G & 7;
            const unsigned char* rec = tiles.data() + (static_cast<std::size_t>(m) * (in >> 7) + c) * cc::kNvfp4RecordBytes;
            if (rec[1024 + g * 16 + h * 8 + j] != sb) ++scmis;
            const float sd = cc::nvfp4_host_e4m3(sb);
            maxsc = std::max(maxsc, sd);
            for (int i = 0; i < 16; ++i) {
                const float dv = cc::nvfp4_host_dequant(tiles.data(), in, o, G * 16 + i);
                const float rv = ((codes[i] & 8) ? -1.f : 1.f) * kV[codes[i] & 7] * sd;
                if (dv != rv) ++mism;
                deq[static_cast<std::size_t>(o) * in + G * 16 + i] = dv * gs;
            }
        }
    for (std::size_t i = 0; i < w.size(); ++i) {
        se += double(deq[i] - w[i]) * (deq[i] - w[i]);
        sn += double(w[i]) * w[i];
    }
    std::printf("   scale bytes != host RTN: %zu, dequant != host RTN: %zu, max scale %.1f, rel RMS err %.4f\n", scmis, mism, maxsc,
                std::sqrt(se / sn));
    expect(scmis == 0 && mism == 0, "converter == host RTN reference (scales and values)");

    const std::uint32_t R = 32;
    std::vector<float> x(static_cast<std::size_t>(R) * in);
    std::normal_distribution<float> xd(0.f, 1.f);
    for (auto& v : x) v = xd(rng);
    float *d_x, *d_y, *d_res;
    CK(cudaMalloc(&d_x, x.size() * 4));
    CK(cudaMalloc(&d_y, std::size_t(R) * out * 4));
    CK(cudaMalloc(&d_res, std::size_t(R) * out * 4));
    CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    std::vector<float> res(std::size_t(R) * out);
    for (auto& v : res) v = xd(rng);
    CK(cudaMemcpy(d_res, res.data(), res.size() * 4, cudaMemcpyHostToDevice));
    cc::DenseWeight W;
    W.data = d_t;
    W.format = cc::WeightFormat::nvfp4;
    W.out = out;
    W.in = in;
    W.gscale = gs;
    std::vector<float> dq(static_cast<std::size_t>(out) * in);
    for (std::uint32_t o = 0; o < out; ++o)
        for (std::uint32_t k = 0; k < in; ++k) dq[std::size_t(o) * in + k] = cc::nvfp4_host_dequant(tiles.data(), in, o, k);
    std::vector<double> ref(std::size_t(R) * out);
    for (std::uint32_t r = 0; r < R; ++r) {
        std::vector<float> xr(in);
        for (std::uint32_t k = 0; k < in; ++k) xr[k] = h2f(x[std::size_t(r) * in + k]);
        for (std::uint32_t o = 0; o < out; ++o) {
            double s = 0;
            for (std::uint32_t k = 0; k < in; ++k) s += double(dq[std::size_t(o) * in + k]) * xr[k];
            ref[std::size_t(r) * out + o] = s * gs;
        }
    }
    std::vector<float> y32(std::size_t(R) * out), y(std::size_t(R) * out);
    CK(cc::nvfp4_dense_rows(W, d_x, d_y, R, nullptr, 0));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(y32.data(), d_y, y32.size() * 4, cudaMemcpyDeviceToHost));
    double maxabs = 0, refmax = 0;
    for (double v : ref) refmax = std::max(refmax, std::fabs(v));
    for (std::size_t i = 0; i < ref.size(); ++i) maxabs = std::max(maxabs, std::fabs(y32[i] - ref[i]));
    std::printf("   rows 32: max|y-ref|=%.3e (ref max %.3e) rel %.2e\n", maxabs, refmax, maxabs / refmax);
    expect(maxabs / refmax < 1e-5, "GEMV rows 32 within 1e-5 of the host double reference (f16 activations)");
    for (std::uint32_t Rr : {1u, 2u, 5u, 8u, 9u, 16u, 17u, 24u, 25u}) {
        CK(cc::nvfp4_dense_rows(W, d_x, d_y, Rr, nullptr, 0));
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(y.data(), d_y, std::size_t(Rr) * out * 4, cudaMemcpyDeviceToHost));
        char b[96];
        std::snprintf(b, sizeof b, "rows %u bitwise == rows 32 (same rows)", Rr);
        expect(std::memcmp(y.data(), y32.data(), std::size_t(Rr) * out * 4) == 0, b);
    }
    {
        bool same = true;
        for (std::uint32_t r : {0u, 7u, 31u}) {
            CK(cc::nvfp4_dense_rows(W, d_x + std::size_t(r) * in, d_y, 1, nullptr, 0));
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(y.data(), d_y, std::size_t(out) * 4, cudaMemcpyDeviceToHost));
            same = same && std::memcmp(y.data(), y32.data() + std::size_t(r) * out, std::size_t(out) * 4) == 0;
        }
        expect(same, "single-row calls bitwise == rows of the 32-row batch");
    }
    {
        CK(cc::nvfp4_dense_rows(W, d_x, d_y, 3, d_res, 0));
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(y.data(), d_y, std::size_t(3) * out * 4, cudaMemcpyDeviceToHost));
        bool same = true;
        for (std::size_t i = 0; i < std::size_t(3) * out; ++i) same = same && (y[i] == res[i] + y32[i]);
        expect(same, "residual = fl(residual + (W x)) bitwise");
    }
    {   // multi-job launch == separate calls, bitwise
        float* d_y2;
        CK(cudaMalloc(&d_y2, std::size_t(R) * out * 4));
        const cc::Nvfp4Job jobs[2] = {{W, d_y, nullptr}, {W, d_y2, d_res}};
        CK(cc::nvfp4_dense_rows_multi(jobs, 2, d_x, 5, 0));
        CK(cudaDeviceSynchronize());
        std::vector<float> ya(std::size_t(5) * out), yb(std::size_t(5) * out);
        CK(cudaMemcpy(ya.data(), d_y, ya.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(yb.data(), d_y2, yb.size() * 4, cudaMemcpyDeviceToHost));
        bool same = std::memcmp(ya.data(), y32.data(), ya.size() * 4) == 0;
        for (std::size_t i = 0; i < yb.size(); ++i) same = same && yb[i] == res[i] + y32[i];
        expect(same, "multi-job launch bitwise == separate calls (plain + residual)");
        cudaFree(d_y2);
    }
    {
        cc::DenseWeight U = W;  // up == gate weights: y = silu(g) * g
        CK(cc::nvfp4_gate_up_silu_rows(W, U, d_x, d_y, 4, 0));
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(y.data(), d_y, std::size_t(4) * out * 4, cudaMemcpyDeviceToHost));
        std::size_t bad = 0;
        for (std::size_t i = 0; i < std::size_t(4) * out; ++i) {
            const float g = y32[i];
            const float e = g / (1.0f + std::exp(-g)) * g;
            if (std::fabs(y[i] - e) > 1e-5f * (std::fabs(e) + 1e-3f)) ++bad;
        }
        expect(bad == 0, "fused gate/up silu close to host epilogue");
    }
    cudaFree(d_w); cudaFree(d_t); cudaFree(d_x); cudaFree(d_y); cudaFree(d_res);
}

__global__ void read_kernel(const uint4* p, std::size_t n, unsigned* sink) {
    unsigned a = 0;
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x) {
        uint4 v;
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
        a ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (a == 0x12345678u) *sink = a;
}

static void read_peak() {
    const std::size_t bytes = std::size_t(1) << 30;
    uint4* p;
    unsigned* sink;
    CK(cudaMalloc(&p, bytes));
    CK(cudaMalloc(&sink, 4));
    CK(cudaMemset(p, 1, bytes));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (int bpsm : {2, 4, 8}) {
        read_kernel<<<188 * bpsm, 256>>>(p, bytes / 16, sink);
        CK(cudaEventRecord(e0));
        for (int i = 0; i < 5; ++i) read_kernel<<<188 * bpsm, 256>>>(p, bytes / 16, sink);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        std::printf("read-only 1 GiB, %d blocks/SM x 256 thr: %.0f GB/s\n", bpsm, 5.0 * bytes / (ms * 1e-3) / 1e9);
    }
    cudaFree(p);
    cudaFree(sink);
}

static void bench() {
    read_peak();
    struct S {
        const char* name;
        std::uint32_t out, in;
        bool fused;
    };
    const S shapes[] = {{"qkv 5120->12288", 12288, 5120, false},        {"kv 5120->1024", 1024, 5120, false},
                        {"o 6144->5120", 5120, 6144, false},            {"gdn_qkv 5120->10240", 10240, 5120, false},
                        {"z 5120->6144", 6144, 5120, false},            {"gate+up 5120->17408 x2", 17408, 5120, true},
                        {"down 17408->5120", 5120, 17408, false},       {"lm_head 5120->248320", 248320, 5120, false}};
    const std::uint32_t rows_list[] = {1, 2, 4, 8, 16, 32};
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    float* d_x;
    CK(cudaMalloc(&d_x, std::size_t(32) * 17408 * 4));
    {
        std::vector<float> hx(std::size_t(32) * 17408);
        std::mt19937 r(5);
        std::normal_distribution<float> d(0, 1);
        for (auto& v : hx) v = d(r);
        CK(cudaMemcpy(d_x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    }
    float* d_y;
    CK(cudaMalloc(&d_y, std::size_t(32) * 248320 * 4));
    std::printf("%-24s %4s", "shape (GB/s over weight bytes)", "ks");
    for (auto r : rows_list) std::printf("   R=%-2u us  GB/s", r);
    std::printf("\n");
    for (const auto& s : shapes) {
        const std::size_t per = cc::nvfp4_bytes(s.out, s.in) * (s.fused ? 2 : 1);
        const std::size_t sets = std::max<std::size_t>(2, (std::size_t(1500) << 20) / per + 1);
        unsigned char* buf;
        CK(cudaMalloc(&buf, per * sets));
        std::vector<unsigned char> h(std::min<std::size_t>(per, 64u << 20));
        std::mt19937 rg(7);
        for (auto& b : h) b = static_cast<unsigned char>(rg());
        for (std::size_t o = 0; o < per * sets; o += h.size())
            CK(cudaMemcpy(buf + o, h.data(), std::min(h.size(), per * sets - o), cudaMemcpyHostToDevice));
        std::printf("%-24s %4u", s.name, cc::nvfp4_split_k(s.out, s.in));
        for (auto r : rows_list) {
            auto run_s = [&](std::size_t set, cudaStream_t st) {
                cc::DenseWeight W;
                W.format = cc::WeightFormat::nvfp4;
                W.out = s.out;
                W.in = s.in;
                W.gscale = 0.001f;
                W.data = buf + set * per;
                if (s.fused) {
                    cc::DenseWeight U = W;
                    U.data = buf + set * per + per / 2;
                    CK(cc::nvfp4_gate_up_silu_rows(W, U, d_x, d_y, r, st));
                } else {
                    CK(cc::nvfp4_dense_rows(W, d_x, d_y, r, nullptr, st));
                }
            };
            auto run = [&](std::size_t set) { run_s(set, 0); };
            for (std::size_t i = 0; i < sets; ++i) run(i);
            CK(cudaDeviceSynchronize());
            // graph of `sets` calls (each on its own weight copy) removes host launch gaps: device-side time per call
            cudaGraph_t gr;
            cudaGraphExec_t ge;
            cudaStream_t cs;
            CK(cudaStreamCreate(&cs));
            run_s(0, cs);  // workspaces are per (device, stream): size this stream's one outside the capture
            CK(cudaStreamSynchronize(cs));
            CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal));
            for (std::size_t i = 0; i < sets; ++i) run_s(i, cs);
            CK(cudaStreamEndCapture(cs, &gr));
            CK(cudaGraphInstantiate(&ge, gr, 0));
            CK(cudaGraphLaunch(ge, cs));
            CK(cudaStreamSynchronize(cs));
            CK(cudaEventRecord(e0, cs));
            for (int i = 0; i < 4; ++i) CK(cudaGraphLaunch(ge, cs));
            CK(cudaEventRecord(e1, cs));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            CK(cudaGraphExecDestroy(ge));
            CK(cudaGraphDestroy(gr));
            cc::nvfp4_release_workspace(cs);
            CK(cudaStreamDestroy(cs));
            const double us = ms * 1e3 / (4.0 * sets);
            std::printf("  %7.1f %5.0f", us, per / (us * 1e-6) / 1e9);
        }
        std::printf("   [%.1f MB]\n", per / 1e6);
        cudaFree(buf);
    }
}

int main(int argc, char** argv) {
    const std::string a = argc > 1 ? argv[1] : "";
    if (argc > 2) cc::nvfp4_set_split_target(static_cast<std::uint32_t>(std::atoi(argv[2])));
    CK(cudaSetDevice(0));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    std::printf("device 0: %s cc %d.%d SMs %d\n", p.name, p.major, p.minor, p.multiProcessorCount);
    if (a != "--bench-only") {
        std::mt19937 rng(1234);
        test_shape(256, 512, rng);
        test_shape(48, 5120, rng);
        test_shape(1024, 6144, rng);
        test_shape(5120, 17408, rng);
        std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
    }
    if (a != "--test-only") bench();
    return g_fail ? 1 : 0;
}
