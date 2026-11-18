// Exactness + timing test of the QSA decode kernels (rows 1..8): attention_output_head_kernel (reference) vs
// attention_output_head_pipe_kernel (STRIX_QRTX_QSA_HEAD), and indexer_kernel<64> (reference) vs indexer_kernel<kIdxWide>
// (STRIX_QRTX_QSA_IDX), on random BF16 data; every output compared with memcmp. The .cu is included so the unnamed-namespace
// kernels are reachable. Needs a CUDA device and the amd math tables dir; no model.
//   qwen_qsa_decode_exact_test <tables dir>
#include "../../src/cuda/models/qwen/qwen_qsa.cu"

#include "strix/cuda/amd_math_tables.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace strix::cuda::qwen_qsa {
namespace {
#define CKT(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)
int g_pass = 0, g_fail = 0;
std::mt19937 rng(20261115);
std::uint16_t bf(float v) { std::uint32_t b; std::memcpy(&b, &v, 4); return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16) & 1U)) >> 16); }
float nrm(float s) { static std::normal_distribution<float> d(0.F, 1.F); return d(rng) * s; }
template <class T> T* dev(const std::vector<T>& h) {
    T* p; CKT(cudaMalloc(&p, h.size() * sizeof(T))); CKT(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
template <class F> float time_us(F&& f, int reps = 50) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 5; ++i) f();
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i) f();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); cudaEventDestroy(a); cudaEventDestroy(b);
    return ms * 1000.F / reps;
}
void report(const char* what, bool ok) { (ok ? g_pass : g_fail)++; if (!ok) std::printf("FAIL %s\n", what); }

void head_case(const am::Tables& tables, std::uint32_t rows, std::uint64_t first_position, bool timing) {
    const std::uint64_t ntok = first_position + rows + 4;
    std::vector<std::uint16_t> kvh(ntok * 2ULL * kKvWidth);
    for (auto& x : kvh) x = bf(nrm(1.F));
    std::vector<std::uint32_t> sel(static_cast<std::size_t>(rows) * kMaxSelected, 0);
    std::vector<float> sc(static_cast<std::size_t>(rows) * kHeads * kMaxSelected, 0.F);
    std::uniform_real_distribution<float> u(0.F, 1.F);
    for (std::uint32_t r = 0; r < rows; ++r) {
        const auto cnt = selected_count(first_position + r);
        const auto vis = first_position + r + 1;
        for (std::uint32_t i = 0; i < cnt; ++i) sel[r * kMaxSelected + i] = static_cast<std::uint32_t>(rng() % vis);
        for (std::uint32_t h = 0; h < kHeads; ++h)
            for (std::uint32_t i = 0; i < cnt; ++i) sc[(static_cast<std::size_t>(r) * kHeads + h) * kMaxSelected + i] = u(rng) * 2.F / static_cast<float>(cnt);
    }
    std::vector<float> qg(static_cast<std::size_t>(rows) * kQGateWidth);
    for (auto& x : qg) x = nrm(2.F);
    auto* d_kv = dev(kvh); auto* d_sel = dev(sel); auto* d_sc = dev(sc); auto* d_qg = dev(qg);
    float *o0, *o1; const std::size_t on = static_cast<std::size_t>(rows) * kAttnWidth;
    CKT(cudaMalloc(&o0, on * 4)); CKT(cudaMalloc(&o1, on * 4)); CKT(cudaMemset(o0, 0xAB, on * 4)); CKT(cudaMemset(o1, 0xCD, on * 4));
    auto run_old = [&] { attention_output_head_kernel<<<dim3(kHeads, rows), kOutThreads>>>(d_qg, first_position, d_sel, d_kv, d_sc, o0, tables); };
    auto run_new = [&] { attention_output_head_pipe_kernel<<<dim3(kHeads, rows), kOutThreads>>>(d_qg, first_position, d_sel, d_kv, d_sc, o1, tables); };
    run_old(); run_new(); CKT(cudaDeviceSynchronize());
    std::vector<float> a(on), b(on);
    CKT(cudaMemcpy(a.data(), o0, on * 4, cudaMemcpyDeviceToHost)); CKT(cudaMemcpy(b.data(), o1, on * 4, cudaMemcpyDeviceToHost));
    char name[96]; std::snprintf(name, sizeof name, "head rows=%u pos=%llu", rows, static_cast<unsigned long long>(first_position));
    report(name, std::memcmp(a.data(), b.data(), on * 4) == 0);
    if (timing) {
        const float t0 = time_us(run_old), t1 = time_us(run_new);
        std::printf("TIMING head rows=%u count=%u: old %.1f us  new %.1f us\n", rows, selected_count(first_position + rows - 1), t0, t1);
    }
    cudaFree(d_kv); cudaFree(d_sel); cudaFree(d_sc); cudaFree(d_qg); cudaFree(o0); cudaFree(o1);
}

template <unsigned S>
void idx_variant(std::uint32_t rows, const std::uint16_t* h, const std::uint16_t* wq, const std::uint16_t* wk, const float* ref_q, const float* ref_k,
    float* oq, float* ok, bool timing, float t_old) {
    CKT(cudaFuncSetAttribute(indexer_kernel<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kIdxSmem)));
    auto run = [&] { indexer_kernel<S><<<dim3(1, kIdxOutputs / S), 256, kIdxSmem>>>(h, rows, wq, wk, oq, ok); };
    CKT(cudaMemset(oq, 0xAB, rows * kIndexQWidth * 4)); CKT(cudaMemset(ok, 0xAB, rows * kIndexDim * 4));
    run(); CKT(cudaDeviceSynchronize());
    std::vector<float> a(rows * kIndexQWidth), b(rows * kIndexDim), ra(a.size()), rb(b.size());
    CKT(cudaMemcpy(a.data(), oq, a.size() * 4, cudaMemcpyDeviceToHost)); CKT(cudaMemcpy(b.data(), ok, b.size() * 4, cudaMemcpyDeviceToHost));
    CKT(cudaMemcpy(ra.data(), ref_q, ra.size() * 4, cudaMemcpyDeviceToHost)); CKT(cudaMemcpy(rb.data(), ref_k, rb.size() * 4, cudaMemcpyDeviceToHost));
    char name[64]; std::snprintf(name, sizeof name, "indexer slice=%u rows=%u", S, rows);
    report(name, std::memcmp(a.data(), ra.data(), a.size() * 4) == 0 && std::memcmp(b.data(), rb.data(), b.size() * 4) == 0);
    if (timing) std::printf("TIMING indexer rows=%u slice=%u (%u CTAs): old %.1f us  new %.1f us\n", rows, S, kIdxOutputs / S, t_old, time_us(run));
}

void idx_case(std::uint32_t rows, bool timing) {
    std::vector<std::uint16_t> hid(static_cast<std::size_t>(rows) * kHidden), wq(static_cast<std::size_t>(kIndexQWidth) * kHidden), wk(static_cast<std::size_t>(kIndexDim) * kHidden);
    for (auto& x : hid) x = bf(nrm(1.F));
    for (auto& x : wq) x = bf(nrm(0.04F));
    for (auto& x : wk) x = bf(nrm(0.04F));
    auto* dh = dev(hid); auto* dq = dev(wq); auto* dk = dev(wk);
    float *rq, *rk, *oq, *ok;
    CKT(cudaMalloc(&rq, rows * kIndexQWidth * 4)); CKT(cudaMalloc(&rk, rows * kIndexDim * 4));
    CKT(cudaMalloc(&oq, rows * kIndexQWidth * 4)); CKT(cudaMalloc(&ok, rows * kIndexDim * 4));
    CKT(cudaFuncSetAttribute(indexer_kernel<kIdxSlice>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kIdxSmem)));
    auto run_old = [&] { indexer_kernel<kIdxSlice><<<dim3(1, kIdxOutputs / kIdxSlice), 256, kIdxSmem>>>(dh, rows, dq, dk, rq, rk); };
    run_old(); CKT(cudaDeviceSynchronize());
    const float t_old = timing ? time_us(run_old) : 0.F;
    idx_variant<kIdxWide>(rows, dh, dq, dk, rq, rk, oq, ok, timing, t_old);
    if (timing) { idx_variant<8>(rows, dh, dq, dk, rq, rk, oq, ok, true, t_old); idx_variant<16>(rows, dh, dq, dk, rq, rk, oq, ok, true, t_old); }
    cudaFree(dh); cudaFree(dq); cudaFree(dk); cudaFree(rq); cudaFree(rk); cudaFree(oq); cudaFree(ok);
}
}  // namespace

int run(const char* tables_dir) {
    auto tables = am::load_tables(tables_dir);
    // positions: ~300, 1K, partial counts, 2047/2048/2051 boundary, 4K+ (capped at kMaxSelected), tail != 0 and == 0
    const std::uint64_t positions[] = {0, 2, 7, 299, 300, 1023, 1030, 2043, 2046, 2047, 2048, 4093, 4097, 4100};
    for (auto pos : positions)
        for (std::uint32_t rows = 1; rows <= 8; ++rows) head_case(tables, rows, pos, false);
    for (std::uint32_t rows : {1U, 4U, 8U}) { head_case(tables, rows, 1023 - rows, true); head_case(tables, rows, 4100, true); }
    for (std::uint32_t rows = 1; rows <= 8; ++rows) idx_case(rows, rows == 1 || rows == 4 || rows == 8);
    std::printf("PASS %d FAIL %d\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
}  // namespace strix::cuda::qwen_qsa

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: qwen_qsa_decode_exact_test <tables dir>\n"); return 1; }
    return strix::cuda::qwen_qsa::run(argv[1]);
}
