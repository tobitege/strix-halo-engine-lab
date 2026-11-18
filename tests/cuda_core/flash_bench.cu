// Long-context prefill attention bench + identity check for cuda_core::attention_prefill_flash (head_dim 256, GQA).
//
//   cuda_core_flash_bench [--lens 32768,65536,131072,262144] [--rows 4096] [--heads 24] [--kv-heads 4] [--iters 3] [--pipes pad2,swz3]
//                         [--check-only] [--no-f8]
//
// Per length L: a filled BF16 KV cache of L positions (deterministic pseudo-random values), one prefill chunk of `rows` query rows at
// positions L - rows .. L - 1 (the last chunk of an L-token prefill, the most expensive one). Every pipe runs on the same inputs; the
// output bytes of each pipe are compared with the first pipe (identity of scheduling-only pipes) and the time per call (CUDA events,
// median of --iters after one warm-up) is printed with the causal FLOP rate (4 * head_dim * heads * visible keys per row).
// FAST8 arm (unless --no-f8): kv_shadow_e4m3 + attention_prefill_flash8 on the same inputs, printed with its error vs the first pipe
// (relative RMS over all outputs, max abs; non-finite outputs fail the run).
// --check-only: small lengths (and the partial last tile / short chunk edges) for ctest; exit 1 on any byte difference.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strix/cuda_core/attention.hpp"

namespace cc = strix::cuda_core;

namespace {

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(2);                                                                        \
        }                                                                                        \
    } while (0)

__device__ __forceinline__ float hash_unit(std::uint64_t i, std::uint32_t seed) {  // deterministic value in [-1, 1)
    std::uint64_t x = i * 0x9E3779B97F4A7C15ull + seed * 0xBF58476D1CE4E5B9ull;
    x ^= x >> 31;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 29;
    return static_cast<float>(static_cast<std::uint32_t>(x >> 40)) * (2.0f / 16777216.0f) - 1.0f;
}
__global__ void fill_bf16(__nv_bfloat16* p, std::uint64_t n, std::uint32_t seed, float amp) {
    for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        p[i] = __float2bfloat16_rn(amp * hash_unit(i, seed));
}
__global__ void fill_f32(float* p, std::uint64_t n, std::uint32_t seed, float amp) {
    for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        p[i] = amp * hash_unit(i, seed);
}

std::vector<std::uint32_t> parse_list(const char* s) {
    std::vector<std::uint32_t> v;
    for (const char* p = s; *p;) {
        v.push_back(static_cast<std::uint32_t>(std::strtoul(p, nullptr, 10)));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}
// arm = pipe [t[w]]: "pad2", "swz3", "pad2t" (wave-tail split, auto warps), "pad2t1" / "t2" / "t4" (forced tail warps)
struct Arm {
    cc::FlashPipe pipe = cc::FlashPipe::pad2;
    cc::FlashTail tail = cc::FlashTail::off;
    std::uint32_t warps = 0;
    char name[8] = "pad2";
};
Arm parse_arm(const char* p) {
    Arm a;
    a.pipe = std::strncmp(p, "swz3", 4) == 0 ? cc::FlashPipe::swz3 : cc::FlashPipe::pad2;
    if (p[0] && p[1] && p[2] && p[3] && p[4] == 't') {
        a.tail = cc::FlashTail::split;
        if (p[5] >= '1' && p[5] <= '4') a.warps = static_cast<std::uint32_t>(p[5] - '0');
    }
    std::size_t n = 0;
    while (p[n] && p[n] != ',' && n < 7) ++n;
    std::memcpy(a.name, p, n);
    a.name[n] = 0;
    return a;
}

struct Case {
    std::uint32_t L, rows;
};

// returns number of byte-mismatching floats vs the first pipe (summed over pipes)
bool with_f8 = true, f8_bad = false;

std::size_t run_case(const Case& c, std::uint32_t heads, std::uint32_t kvh, const std::vector<Arm>& pipes, int iters, bool timing) {
    constexpr std::uint32_t D = 256;
    cc::AttentionDims d;
    d.heads = heads;
    d.kv_heads = kvh;
    d.head_dim = D;
    d.scale = 1.0f / 16.0f;
    d.max_positions = c.L;
    const std::uint32_t pos0 = c.L - c.rows;
    const std::size_t kv_n = static_cast<std::size_t>(kvh) * c.L * D, q_n = static_cast<std::size_t>(c.rows) * heads * D;
    __nv_bfloat16 *kc = nullptr, *vc = nullptr;
    float *q = nullptr, *gate = nullptr, *out = nullptr;
    CK(cudaMalloc(&kc, kv_n * 2));
    CK(cudaMalloc(&vc, kv_n * 2));
    CK(cudaMalloc(&q, q_n * 4));
    CK(cudaMalloc(&gate, q_n * 4));
    CK(cudaMalloc(&out, q_n * 4));
    // amplitudes chosen so the scores spread over a few units (RMS-normed q and k of a real model): peaky and flat rows both occur
    fill_bf16<<<1024, 256>>>(kc, kv_n, 11u, 1.7f);
    fill_bf16<<<1024, 256>>>(vc, kv_n, 23u, 1.0f);
    fill_f32<<<1024, 256>>>(q, q_n, 37u, 2.5f);
    fill_f32<<<1024, 256>>>(gate, q_n, 41u, 3.0f);
    CK(cudaDeviceSynchronize());
    const cc::GroupedView qv{c.rows, heads, D, heads * D, D};
    double flop = 0;
    for (std::uint32_t r = 0; r < c.rows; ++r) flop += 4.0 * D * heads * (pos0 + r + 1.0);
    std::vector<float> ref(q_n), cur(q_n);
    std::size_t bad_total = 0;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (std::size_t pi = 0; pi < pipes.size(); ++pi) {
        CK(cudaMemset(out, 0xFF, q_n * 4));
        CK(cc::attention_prefill_flash(d, q, qv, gate, qv, kc, vc, pos0, c.rows, out, nullptr, pipes[pi].pipe, pipes[pi].tail, pipes[pi].warps));
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(pi == 0 ? ref.data() : cur.data(), out, q_n * 4, cudaMemcpyDeviceToHost));
        std::size_t bad = 0;
        if (pi > 0)
            for (std::size_t i = 0; i < q_n; ++i) bad += std::memcmp(&ref[i], &cur[i], 4) != 0;
        bad_total += bad;
        double ms_med = 0;
        if (timing) {
            std::vector<float> ms(static_cast<std::size_t>(iters));
            for (int it = 0; it < iters; ++it) {
                CK(cudaEventRecord(e0));
                CK(cc::attention_prefill_flash(d, q, qv, gate, qv, kc, vc, pos0, c.rows, out, nullptr, pipes[pi].pipe, pipes[pi].tail,
                                               pipes[pi].warps));
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms[static_cast<std::size_t>(it)], e0, e1));
            }
            std::sort(ms.begin(), ms.end());
            ms_med = ms[ms.size() / 2];
        }
        int clk = 0;
        cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
        std::printf("L=%u rows=%u pos0=%u heads=%u/%u pipe=%s ms=%.3f TF/s=%.1f %s%zu\n", c.L, c.rows, pos0, heads, kvh, pipes[pi].name,
                    ms_med, timing ? flop / (ms_med * 1e9) : 0.0, pi == 0 ? "ref " : "mismatch=", pi == 0 ? std::size_t{0} : bad);
        std::fflush(stdout);
    }
    if (with_f8) {  // FAST8 arm: e4m3 shadow of the whole cache (build timed for this chunk's positions), flash8, error vs pipe 0
        cc::KvShadowE4m3 sh;
        const std::size_t p8 = cc::attn8_capacity(d);
        CK(cudaMalloc(&sh.k8, kvh * p8 * D));
        CK(cudaMalloc(&sh.v8t, kvh * p8 * D));
        CK(cudaMalloc(&sh.ks, kvh * p8 * 4));
        CK(cudaMalloc(&sh.vs, kvh * p8 * 4));
        CK(cudaMemset(sh.k8, 0, kvh * p8 * D));
        CK(cudaMemset(sh.v8t, 0, kvh * p8 * D));
        CK(cudaMemset(sh.ks, 0, kvh * p8 * 4));
        CK(cudaMemset(sh.vs, 0, kvh * p8 * 4));
        CK(cc::kv_shadow_e4m3(d, kc, vc, sh, 0, pos0, nullptr));
        float shadow_ms = 0;
        CK(cudaEventRecord(e0));
        CK(cc::kv_shadow_e4m3(d, kc, vc, sh, pos0, c.rows, nullptr));
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(&shadow_ms, e0, e1));
        CK(cc::attention_prefill_flash8(d, q, qv, gate, qv, sh, pos0, c.rows, out, nullptr));
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(cur.data(), out, q_n * 4, cudaMemcpyDeviceToHost));
        double se = 0, sr = 0, mx = 0;
        std::size_t nonfinite = 0;
        for (std::size_t i = 0; i < q_n; ++i) {
            if (!std::isfinite(cur[i])) { ++nonfinite; continue; }
            const double e = static_cast<double>(cur[i]) - ref[i];
            se += e * e;
            sr += static_cast<double>(ref[i]) * ref[i];
            mx = std::max(mx, std::fabs(e));
        }
        double ms_med = 0;
        if (timing) {
            std::vector<float> ms(static_cast<std::size_t>(iters));
            for (int it = 0; it < iters; ++it) {
                CK(cudaEventRecord(e0));
                CK(cc::attention_prefill_flash8(d, q, qv, gate, qv, sh, pos0, c.rows, out, nullptr));
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                CK(cudaEventElapsedTime(&ms[static_cast<std::size_t>(it)], e0, e1));
            }
            std::sort(ms.begin(), ms.end());
            ms_med = ms[ms.size() / 2];
        }
        std::printf("L=%u rows=%u pos0=%u pipe=flash8 ms=%.3f TF/s=%.1f shadow_ms=%.3f rel_rms=%.4g max_abs=%.4g nonfinite=%zu\n", c.L, c.rows, pos0,
                    ms_med, timing ? flop / (ms_med * 1e9) : 0.0, shadow_ms, std::sqrt(se / std::max(sr, 1e-30)), mx, nonfinite);
        std::fflush(stdout);
        if (nonfinite) f8_bad = true;
        CK(cudaFree(sh.k8));
        CK(cudaFree(sh.v8t));
        CK(cudaFree(sh.ks));
        CK(cudaFree(sh.vs));
    }
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    CK(cudaFree(kc));
    CK(cudaFree(vc));
    CK(cudaFree(q));
    CK(cudaFree(gate));
    CK(cudaFree(out));
    return bad_total;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::uint32_t> lens{32768, 65536, 131072, 262144};
    std::uint32_t rows = 4096, heads = 24, kvh = 4;
    int iters = 3;
    bool check_only = false;
    std::vector<Arm> pipes{parse_arm("pad2"), parse_arm("swz3"), parse_arm("pad2t"), parse_arm("pad2t1"), parse_arm("pad2t2"),
                           parse_arm("pad2t4"), parse_arm("swz3t")};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--lens") lens = parse_list(next());
        else if (a == "--rows") rows = static_cast<std::uint32_t>(std::atoi(next()));
        else if (a == "--heads") heads = static_cast<std::uint32_t>(std::atoi(next()));
        else if (a == "--kv-heads") kvh = static_cast<std::uint32_t>(std::atoi(next()));
        else if (a == "--iters") iters = std::max(1, std::atoi(next()));
        else if (a == "--check-only") check_only = true;
        else if (a == "--no-f8") with_f8 = false;
        else if (a == "--pipes") {
            pipes.clear();
            for (const char* p = next(); *p;) {
                pipes.push_back(parse_arm(p));
                while (*p && *p != ',') ++p;
                if (*p == ',') ++p;
            }
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    std::size_t bad = 0;
    if (check_only) {
        // edges: partial last key tile, chunk shorter than one query block, chunk starting at position 0, long pos0
        const Case cases[] = {{4096, 4096}, {1000, 1000}, {5000, 77}, {9000, 2048}, {33, 33}, {20000, 130}};
        for (const Case& c : cases) bad += run_case(c, heads, kvh, pipes, 1, false);
    } else {
        for (std::uint32_t L : lens) bad += run_case({L, std::min(rows, L)}, heads, kvh, pipes, iters, true);
    }
    std::printf("flash_bench: %s (%zu mismatching floats)%s\n", bad == 0 ? "IDENTICAL" : "MISMATCH", bad, f8_bad ? ", flash8 NON-FINITE" : "");
    return bad == 0 && !f8_bad ? 0 : 1;
}
