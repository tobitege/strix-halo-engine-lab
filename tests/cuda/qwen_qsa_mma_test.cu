// Model-free device test of the QSA FAST attention form (Config::forms.qsa_mma): qsa::attention_mma against the exact
// qsa::attention on the same random inputs (selection built like select_kernel's output: all tokens while <= 512 complete
// blocks, else 512 random 4-token blocks in random order + the tail). Reports max abs / relative L2 of attn_out and the
// per-row relative L2 spread (errors must be uniform), then times both at 8K rows (history 0) and at a 32K position.
//   qwen_qsa_mma_test <tables dir> [nofail]
#include "strix/cuda/qwen_qsa.hpp"

#include "strix/cuda/amd_math_tables.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace qsa = strix::cuda::qwen_qsa;
namespace am = strix::cuda::amdmath;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
std::mt19937 rng(20261115);
float uniform(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
std::uint16_t to_bf16(float x) { std::uint32_t u; std::memcpy(&u, &x, 4); return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16) & 1U)) >> 16); }
float bf16_value(float x) { std::uint32_t u = static_cast<std::uint32_t>(to_bf16(x)) << 16; float y; std::memcpy(&y, &u, 4); return y; }
template <class T> T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    ck(cudaMalloc(&d, std::max<std::size_t>(v.size(), 1) * sizeof(T)), "malloc");
    ck(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <class T> std::vector<T> download(const T* d, std::size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

struct Inputs {
    std::uint16_t* kv = nullptr;
    float* q = nullptr;
    std::uint32_t* sel = nullptr;
    std::uint64_t positions = 0;
};
Inputs make_inputs(std::uint32_t rows, std::uint64_t first, float qscale) {
    Inputs in;
    in.positions = first + rows;
    std::vector<std::uint16_t> kv(in.positions * 1024);
    for (auto& x : kv) x = to_bf16(uniform(-1.5F, 1.5F));
    std::vector<float> q(std::uint64_t{rows} * qsa::kQGateWidth);
    for (auto& x : q) x = bf16_value(uniform(-qscale, qscale));
    std::vector<std::uint32_t> sel(std::uint64_t{rows} * qsa::kMaxSelected, 0U);
    std::vector<std::uint32_t> blocks;
    for (std::uint32_t r = 0; r < rows; ++r) {
        const auto visible = static_cast<std::uint32_t>(first + r + 1), nb = visible / qsa::kRatio;
        auto* s = &sel[std::uint64_t{r} * qsa::kMaxSelected];
        if (nb <= qsa::kTopK) { for (std::uint32_t i = 0; i < visible; ++i) s[i] = i; continue; }
        blocks.resize(nb);
        for (std::uint32_t i = 0; i < nb; ++i) blocks[i] = i;
        for (std::uint32_t i = 0; i < qsa::kTopK; ++i) std::swap(blocks[i], blocks[i + rng() % (nb - i)]);
        for (std::uint32_t i = 0; i < qsa::kTopK; ++i)
            for (std::uint32_t o = 0; o < qsa::kRatio; ++o) s[i * qsa::kRatio + o] = blocks[i] * qsa::kRatio + o;
        for (std::uint32_t t = nb * qsa::kRatio; t < visible; ++t) s[qsa::kTopK * qsa::kRatio + t - nb * qsa::kRatio] = t;
    }
    in.kv = upload(kv); in.q = upload(q); in.sel = upload(sel);
    return in;
}
void free_inputs(Inputs& in) { cudaFree(in.kv); cudaFree(in.q); cudaFree(in.sel); }

int failures = 0;
void compare(std::uint32_t rows, std::uint64_t first, float qscale, const am::Tables& tables, bool nofail) {
    auto in = make_inputs(rows, first, qscale);
    float *scores, *oa, *ob;
    ck(cudaMalloc(&scores, std::uint64_t{rows} * qsa::kHeads * qsa::kMaxSelected * 4), "scores");
    ck(cudaMalloc(&oa, std::uint64_t{rows} * qsa::kAttnWidth * 4), "oa");
    ck(cudaMalloc(&ob, std::uint64_t{rows} * qsa::kAttnWidth * 4), "ob");
    ck(cudaMemset(ob, 0xff, std::uint64_t{rows} * qsa::kAttnWidth * 4), "poison");
    qsa::attention(in.q, rows, first, in.sel, in.kv, scores, oa, tables, nullptr);
    qsa::attention_mma(in.q, rows, first, in.sel, in.kv, ob, nullptr);
    ck(cudaDeviceSynchronize(), "run");
    const auto a = download(oa, std::uint64_t{rows} * qsa::kAttnWidth), b = download(ob, std::uint64_t{rows} * qsa::kAttnWidth);
    double maxabs = 0, d2 = 0, n2 = 0, worst_row = 0, min_row = 1e30, sum_row = 0;
    std::uint32_t worst = 0, bad = 0;
    for (std::uint32_t r = 0; r < rows; ++r) {
        double rd = 0, rn = 0;
        for (std::uint32_t i = 0; i < qsa::kAttnWidth; ++i) {
            const double x = a[std::uint64_t{r} * qsa::kAttnWidth + i], y = b[std::uint64_t{r} * qsa::kAttnWidth + i];
            if (!std::isfinite(y)) { ++bad; continue; }
            maxabs = std::max(maxabs, std::fabs(x - y)); rd += (x - y) * (x - y); rn += x * x;
        }
        const double rel = std::sqrt(rd / std::max(rn, 1e-30));
        if (rel > worst_row) { worst_row = rel; worst = r; }
        min_row = std::min(min_row, rel); sum_row += rel; d2 += rd; n2 += rn;
    }
    const double rel = std::sqrt(d2 / std::max(n2, 1e-30));
    const bool ok = bad == 0 && rel < 1.5e-2 && worst_row < 4e-2;
    std::printf("rows %5u first %6llu qscale %.1f: max abs %.3e rel L2 %.3e | per-row rel L2 min %.2e mean %.2e max %.2e (row %u) nonfinite %u %s\n",
                rows, static_cast<unsigned long long>(first), qscale, maxabs, rel, min_row, sum_row / rows, worst_row, worst, bad, ok ? "ok" : "FAIL");
    if (!ok && !nofail) ++failures;
    cudaFree(scores); cudaFree(oa); cudaFree(ob); free_inputs(in);
}

void timing(std::uint32_t rows, std::uint64_t first, const am::Tables& tables) {
    const std::uint32_t chunk = 4096;
    auto in = make_inputs(rows, first, 2.0F);
    float *scores, *oa;
    ck(cudaMalloc(&scores, std::uint64_t{chunk} * qsa::kHeads * qsa::kMaxSelected * 4), "scores");
    ck(cudaMalloc(&oa, std::uint64_t{rows} * qsa::kAttnWidth * 4), "oa");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int mode = 0; mode < 2; ++mode) {
        const auto run = [&] {
            for (std::uint32_t c = 0; c < rows; c += chunk) {
                const auto n = std::min(chunk, rows - c);
                const auto* q = in.q + std::uint64_t{c} * qsa::kQGateWidth;
                const auto* s = in.sel + std::uint64_t{c} * qsa::kMaxSelected;
                auto* o = oa + std::uint64_t{c} * qsa::kAttnWidth;
                if (mode == 0) qsa::attention(q, n, first + c, s, in.kv, scores, o, tables, nullptr);
                else qsa::attention_mma(q, n, first + c, s, in.kv, o, nullptr);
            }
        };
        run(); run(); ck(cudaDeviceSynchronize(), "warm");
        const int reps = 5;
        cudaEventRecord(e0);
        for (int i = 0; i < reps; ++i) run();
        cudaEventRecord(e1); ck(cudaEventSynchronize(e1), "time");
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        std::printf("timing rows %u first %llu %s: %.3f ms per layer-call (%u rows)\n", rows, static_cast<unsigned long long>(first),
                    mode == 0 ? "exact(scores+softmax+values)" : "qsa_mma", ms / reps, rows);
    }
    cudaFree(scores); cudaFree(oa); free_inputs(in);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::strcmp(argv[2], "nofail") != 0)) {
        std::fprintf(stderr, "usage: qwen_qsa_mma_test <tables dir> [nofail]\n"); return 1;
    }
    const bool nofail = argc == 3;
    const auto tables = am::load_tables(argv[1]);
    for (const std::uint64_t first : {0ULL, 9000ULL})
        for (const std::uint32_t rows : {1U, 15U, 16U, 17U, 1000U, 4096U}) compare(rows, first, 2.0F, tables, nofail);
    compare(1000, 9000, 6.0F, tables, nofail);   // peaked softmax
    compare(17, 2040, 2.0F, tables, nofail);     // crosses the 512-block boundary (count 2048 -> selection starts)
    timing(8192, 0, tables);
    timing(8192, 32768, tables);
    if (nofail) {
        std::printf("DIAGNOSTIC ONLY: numerical checks not enforced; no qualification verdict\n");
        return 0;
    }
    std::printf(failures ? "FAIL (%d)\n" : "PASS\n", failures);
    return failures ? 1 : 0;
}
