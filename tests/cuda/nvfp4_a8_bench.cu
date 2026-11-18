// Microbenchmark of the Qwen dense decode sites: exact Q8_0 A8 route vs TURBO nvfp4-a8-v1 route (same activations),
// real Flash-Next shapes, rotating weight copies (> L2) so every call streams its weights from memory.
// Usage: nvfp4_a8_bench [reps=100] [rows list, default 1,2,4,8,16,32]. Prints per site and the per-window total
// (36 GDN layers x (in + out) + 12 QSA layers x (in + out) + 48 shared experts x (gate||up + down)).
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "strix/cuda/core/nvfp4_a8_cuda.hpp"
#include "strix/cuda/qwen_dense.hpp"

namespace a8 = strix::cuda::core::nvfp4_a8;
namespace dn = strix::cuda::qwen_dense;
#define CK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

struct Shape { const char* name; std::uint32_t width; std::vector<std::uint32_t> segs; std::uint32_t per_window; };

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 100;
    std::vector<std::uint32_t> rows_list{1, 2, 4, 8, 16, 32};
    if (argc > 2) {
        rows_list.clear();
        for (char* s = std::strtok(argv[2], ","); s; s = std::strtok(nullptr, ",")) rows_list.push_back(std::atoi(s));
    }
    const std::vector<Shape> shapes{{"gdn_in", 2560, {10240, 6144}, 36}, {"gdn_out", 6144, {2560}, 36},
                                    {"qsa_in", 2560, {12288, 512, 512}, 12}, {"qsa_out", 6144, {2560}, 12},
                                    {"shexp_gu", 2560, {640, 640}, 48}, {"shexp_down", 640, {2560}, 48}};
    std::mt19937 rng(7);
    std::vector<double> win_q8(rows_list.size()), win_f4(rows_list.size()), win_best(rows_list.size());
    double win_bytes_q8 = 0, win_bytes_f4 = 0;
    std::int8_t* codes; float* deltas; float* out; void* in;
    CK(cudaMalloc(&codes, 32 * 12288)); CK(cudaMalloc(&deltas, (12288 / 32) * 32 * 4));
    CK(cudaMalloc(&out, 32ull * 16384 * 4)); CK(cudaMalloc(&in, 32ull * 6144 * 4));
    std::vector<float> x(32ull * 6144);
    std::normal_distribution<float> nd(0, 1);
    for (auto& v : x) v = nd(rng);
    CK(cudaMemcpy(in, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    float* scratch; CK(cudaMalloc(&scratch, 4));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (const auto& sh : shapes) {
        std::uint32_t outs = 0;
        for (auto s : sh.segs) outs += s;
        const std::size_t q8_bytes = std::size_t{outs} * (sh.width / 32) * 34, f4_bytes = a8::image_bytes(outs, sh.width);
        const int copies = static_cast<int>(512e6 / q8_bytes) + 2;   // > 4 x L2
        std::vector<std::vector<void*>> w(copies);
        std::vector<std::uint8_t> host;
        double conv_ms = 0;
        for (int c = 0; c < copies; ++c)
            for (auto s : sh.segs) {
                host.resize(std::size_t{s} * (sh.width / 32) * 34);
                std::uniform_int_distribution<int> q(-127, 127);
                for (std::size_t b = 0; b < host.size() / 34; ++b) {
                    const std::uint16_t h = 0x2000 + (rng() & 0x3FF);   // ~0.0078 .. 0.0156
                    std::memcpy(&host[b * 34], &h, 2);
                    for (int j = 0; j < 32; ++j) host[b * 34 + 2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q(rng)));
                }
                void* d; CK(cudaMalloc(&d, host.size())); CK(cudaMemcpy(d, host.data(), host.size(), cudaMemcpyHostToDevice));
                a8::Image img;
                CK(cudaMalloc(&img.codes, std::size_t{s} * sh.width / 2)); CK(cudaMalloc(&img.scales, std::size_t{s} * sh.width / 16));
                CK(cudaEventRecord(e0));
                CK(a8::convert_q8_0(d, s, sh.width, img, scratch, nullptr));
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); conv_ms += ms;
                dn::onfly_register(d, {img.codes, img.scales, img.tensor_scale, s, sh.width});
                w[c].push_back(d);
            }
        win_bytes_q8 += double(q8_bytes) * sh.per_window; win_bytes_f4 += double(f4_bytes) * sh.per_window;
        std::printf("%-10s K=%u N=%u q8 %.1f MB nvfp4 %.1f MB copies %d convert %.2f GB/s\n", sh.name, sh.width, outs,
                    q8_bytes / 1e6, f4_bytes / 1e6, copies, double(q8_bytes) * copies / (conv_ms * 1e6));
        for (std::size_t ri = 0; ri < rows_list.size(); ++ri) {
            const std::uint32_t rows = rows_list[ri];
            CK(dn::quantize(nullptr, rows, sh.width, dn::Input::f32, in, sh.width, codes, deltas, rows));
            const auto run_any = [&](int i) {
                dn::Site site; site.store = dn::Store::f32; site.width = sh.width;
                site.segments = static_cast<std::uint32_t>(sh.segs.size());
                std::uint32_t at = 0;
                for (std::size_t m = 0; m < sh.segs.size(); ++m) {
                    site.segment[m] = {w[i % copies][m], sh.segs[m], out + at, outs};
                    at += sh.segs[m];
                }
                CK(dn::project(nullptr, site, rows, codes, deltas, rows));
            };
            double t[2];
            for (int route = 0; route < 2; ++route) {
                const dn::OnflyScope scope(route == 1);
                const auto run = [&](int i) {
                    dn::Site site; site.store = dn::Store::f32; site.width = sh.width;
                    site.segments = static_cast<std::uint32_t>(sh.segs.size());
                    std::uint32_t at = 0;
                    for (std::size_t m = 0; m < sh.segs.size(); ++m) {
                        site.segment[m] = {w[i % copies][m], sh.segs[m], out + at, outs};
                        at += sh.segs[m];
                    }
                    CK(dn::project(nullptr, site, rows, codes, deltas, rows));
                };
                for (int i = 0; i < copies; ++i) run(i);   // warm-up
                CK(cudaEventRecord(e0));
                for (int i = 0; i < reps; ++i) run(i);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                t[route] = ms * 1e3 / reps;
            }
            win_q8[ri] += t[0] * sh.per_window; win_f4[ri] += t[1] * sh.per_window;
            double best = t[1]; std::uint32_t brw = 0, bow = 0;
            for (const std::uint32_t rw : {1U, 2U, 4U, 8U})
                for (const std::uint32_t ow : {1U, 2U}) {
                    if (rw > 1U && rw >= 2U * rows) continue;
                    dn::onfly_set_launch(rw, ow);
                    const dn::OnflyScope scope(true);
                    for (int i = 0; i < copies; ++i) run_any(i);
                    CK(cudaEventRecord(e0));
                    for (int i = 0; i < reps; ++i) run_any(i);
                    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                    if (ms * 1e3 / reps < best) { best = ms * 1e3 / reps; brw = rw; bow = ow; }
                }
            dn::onfly_set_launch(0, 0);
            win_best[ri] += best * sh.per_window;
            std::printf("  rows %2u: best shape rw %u out %u %.2f us (auto %.2f)\n", rows, brw, bow, best, t[1]);
            std::printf("  rows %2u: q8 %7.2f us (%6.0f GB/s)  nvfp4 %7.2f us (%6.0f GB/s)  x%.2f\n", rows, t[0], q8_bytes / (t[0] * 1e3),
                        t[1], f4_bytes / (t[1] * 1e3), t[0] / t[1]);
        }
    }
    std::printf("window (192 calls): q8 %.2f GB, nvfp4 %.2f GB\n", win_bytes_q8 / 1e9, win_bytes_f4 / 1e9);
    for (std::size_t ri = 0; ri < rows_list.size(); ++ri)
        std::printf("WINDOW rows %2u: q8 %7.0f us  nvfp4 %7.0f us (best shapes %7.0f)  saved %6.0f us  x%.2f\n", rows_list[ri],
                    win_q8[ri], win_f4[ri], win_best[ri], win_q8[ri] - win_f4[ri], win_q8[ri] / win_f4[ri]);
    return 0;
}
