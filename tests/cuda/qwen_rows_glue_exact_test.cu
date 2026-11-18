// Decode-rows exactness + timing test for src/cuda/models/qwen/qwen_dense.cu (project / quantize / shared_expert at rows 1..8).
// The new kernels are default ON with env off switches (STRIX_QRTX_ROWS_CP, ...); run this exe once per setting:
//   qwen_rows_glue_exact_test [--time]  > log      prints "HASH <case> rows=<r> <fnv64 of every output byte>" and (--time) "TIME ..."
//   qwen_rows_glue_exact_test --compare old.log new.log   (host only) -> PASS/FAIL counts
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "strix/cuda/amd_math_tables.hpp"
#include "strix/cuda/qwen_dense.hpp"

namespace qd = strix::cuda::qwen_dense;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

static std::uint64_t fnv(const std::vector<unsigned char>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : v) { h ^= c; h *= 1099511628211ULL; }
    return h;
}
static std::uint16_t f2h(float f) {   // exact enough: round-to-nearest half
    std::uint32_t b; std::memcpy(&b, &f, 4);
    const std::uint32_t s = (b >> 16) & 0x8000U; int e = static_cast<int>((b >> 23) & 255) - 127 + 15; std::uint32_t m = b & 0x7fffffU;
    if (e <= 0) return static_cast<std::uint16_t>(s);
    if (e >= 31) return static_cast<std::uint16_t>(s | 0x7bffU);
    return static_cast<std::uint16_t>(s | (e << 10) | (m >> 13));
}
static std::uint16_t bf(float f) { std::uint32_t b; std::memcpy(&b, &f, 4); return static_cast<std::uint16_t>(b >> 16); }

struct Case { const char* name; qd::Store store; std::uint32_t width; std::vector<std::uint32_t> outs; bool timed; };

int compare(const char* a, const char* b) {
    auto load = [](const char* p) { std::map<std::string, std::string> m; std::FILE* f = std::fopen(p, "r"); char l[512];
        while (f && std::fgets(l, sizeof l, f)) if (!std::strncmp(l, "HASH ", 5)) { std::string s(l + 5); auto k = s.rfind(' '); m[s.substr(0, k)] = s.substr(k); }
        if (f) std::fclose(f); return m; };
    auto A = load(a), B = load(b); int pass = 0, fail = 0;
    for (auto& kv : A) { if (B.count(kv.first) && B[kv.first] == kv.second) ++pass; else { ++fail; std::printf("FAIL %s\n", kv.first.c_str()); } }
    std::printf("compare: PASS %d FAIL %d (of %zu/%zu)\n", pass, fail, A.size(), B.size());
    return fail || A.empty() || A.size() != B.size();
}

int main(int argc, char** argv) {
    if (argc >= 4 && !std::strcmp(argv[1], "--compare")) return compare(argv[2], argv[3]);
    const bool timing = argc >= 2 && !std::strcmp(argv[1], "--time");
    std::vector<Case> cases = {
        {"gdn_in", qd::Store::f32_round_bf16, 2560, {10240, 6144}, true},
        {"gdn_out", qd::Store::bf16, 6144, {2560}, true},
        {"qsa_in", qd::Store::f32_round_bf16, 2560, {8192, 512, 512}, true},
        {"qsa_out", qd::Store::bf16, 4096, {2560}, true},
        {"sh_gateup", qd::Store::f32, 2560, {512, 512}, true},
        {"sh_down", qd::Store::bf16_gated, 512, {2560}, true},
        {"odd_a", qd::Store::f32, 2560, {1001, 37, 3}, false},
        {"odd_b", qd::Store::bf16, 256, {9, 5}, false},
        {"odd_c", qd::Store::f32_round_bf16, 128, {1}, false},
        {"odd_d", qd::Store::bf16_gated, 64, {7}, false},
        {"odd_e", qd::Store::f32, 6144, {13, 2}, false},
    };
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    const std::uint32_t pitch = 16;
    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const char* only = std::getenv("STRIX_T_ONLY"); const char* onlyr = std::getenv("STRIX_T_ROWS");
    for (auto& c : cases) {
        if (only && std::strcmp(only, c.name)) continue;
        const std::uint32_t nb = c.width / 32;
        std::uint32_t total = 0; for (auto o : c.outs) total += o;
        const std::size_t seg_bytes_total = static_cast<std::size_t>(total) * nb * 34;
        const int copies = c.timed ? static_cast<int>(std::max<std::size_t>(1, (320u << 20) / seg_bytes_total)) : 1;
        std::vector<std::vector<unsigned char*>> w(copies);
        for (int k = 0; k < copies; ++k)
            for (auto o : c.outs) {
                std::vector<unsigned char> h(static_cast<std::size_t>(o) * nb * 34);
                for (std::size_t i = 0; i < static_cast<std::size_t>(o) * nb; ++i) {
                    const std::uint16_t d = f2h(0.002F + 0.02F * (rng() % 1000) / 1000.0F);
                    h[i * 34] = d & 255; h[i * 34 + 1] = d >> 8;
                    for (int j = 0; j < 32; ++j) h[i * 34 + 2 + j] = static_cast<unsigned char>(static_cast<int>(rng() % 255) - 127);
                }
                unsigned char* p; CK(cudaMalloc(&p, h.size() + 64)); CK(cudaMemcpy(p, h.data(), h.size(), cudaMemcpyHostToDevice)); w[k].push_back(p);
            }
        for (std::uint32_t rows = 1; rows <= 8; ++rows) {
            if (onlyr && rows != static_cast<std::uint32_t>(std::atoi(onlyr))) continue;
            std::vector<std::uint16_t> hin(static_cast<std::size_t>(rows) * c.width);
            for (auto& x : hin) x = bf(nd(rng) * (1.0F + (rng() % 8)));
            std::uint16_t* din; CK(cudaMalloc(&din, hin.size() * 2)); CK(cudaMemcpy(din, hin.data(), hin.size() * 2, cudaMemcpyHostToDevice));
            std::int8_t* codes; float* deltas; CK(cudaMalloc(&codes, 8 * c.width + 64)); CK(cudaMalloc(&deltas, nb * pitch * 4 + 64));
            CK(qd::quantize(st, rows, c.width, qd::Input::bf16, din, c.width, codes, deltas, pitch));
            std::vector<void*> outs; std::vector<std::size_t> osz;
            auto make_site = [&](int k, std::vector<void*>* ob, float* gate) {
                qd::Site s{}; s.store = c.store; s.width = c.width; s.segments = static_cast<std::uint32_t>(c.outs.size());
                for (std::size_t m = 0; m < c.outs.size(); ++m) {
                    const std::size_t es = (c.store == qd::Store::f32 || c.store == qd::Store::f32_round_bf16) ? 4 : 2;
                    const std::size_t bytes = es * 8 * c.outs[m] + 64;
                    void* o; if (ob->size() <= m) { CK(cudaMalloc(&o, bytes)); CK(cudaMemset(o, 0xAB, bytes)); ob->push_back(o); osz.push_back(bytes); }
                    s.segment[m] = {w[k][m], c.outs[m], (*ob)[m], c.outs[m]};
                }
                s.gate = gate; s.gate_stride = 1; return s;
            };
            float* gate; CK(cudaMalloc(&gate, 64)); { float g[8]; for (auto& x : g) x = 0.25F + (rng() % 100) / 50.0F; CK(cudaMemcpy(gate, g, 32, cudaMemcpyHostToDevice)); }
            qd::Site s0 = make_site(0, &outs, gate);
            CK(qd::project(st, s0, rows, codes, deltas, pitch)); CK(cudaStreamSynchronize(st));
            std::vector<unsigned char> all;
            for (std::size_t m = 0; m < outs.size(); ++m) { std::vector<unsigned char> h(osz[m]); CK(cudaMemcpy(h.data(), outs[m], osz[m], cudaMemcpyDeviceToHost)); all.insert(all.end(), h.begin(), h.end()); }
            std::printf("HASH %s rows=%u %016llx\n", c.name, rows, static_cast<unsigned long long>(fnv(all)));
            if (timing && c.timed && (rows == 1 || rows == 4 || rows == 8)) {
                std::vector<qd::Site> sites; for (int k = 0; k < copies; ++k) sites.push_back(make_site(k, &outs, gate));
                for (int k = 0; k < copies; ++k) CK(qd::project(st, sites[k], rows, codes, deltas, pitch));   // warm
                CK(cudaStreamSynchronize(st));
                const int reps = 50;
                cudaGraph_t g; cudaGraphExec_t ge;   // in-graph launches, like the decode verify window
                CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
                for (int i = 0; i < reps; ++i) CK(qd::project(st, sites[i % copies], rows, codes, deltas, pitch));
                CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
                CK(cudaGraphLaunch(ge, st)); CK(cudaStreamSynchronize(st));
                CK(cudaEventRecord(e0, st)); CK(cudaGraphLaunch(ge, st));
                CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
                cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
                float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
                const double us = ms * 1000.0 / reps, gbs = seg_bytes_total / us / 1e3;
                std::printf("TIME %s rows=%u %.2f us/call  %.0f GB/s (%.0f %% of 1790)\n", c.name, rows, us, gbs, 100.0 * gbs / 1790.0);
            }
            for (auto p : outs) cudaFree(p);
            cudaFree(din); cudaFree(codes); cudaFree(deltas); cudaFree(gate);
        }
        for (auto& v : w) for (auto p : v) cudaFree(p);
    }
    std::printf("DONE\n");
    return 0;
}
