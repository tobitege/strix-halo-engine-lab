// Shared sparse flash core (src/attn_core/sparse_flash_wmma.hpp): correctness against an FP64 CPU softmax attention
// on random data for the GLM MLA shape (H 32, D 512, key row = value row) and the MiniMax MSA shape (H 16, D 128,
// separate key / value rows), per-row key lists of varied length (1 .. 2051, masked tails), plus a throughput line
// per shape. FAST identity: the gate is a rel-L2 bound (bf16 operands), not bits.
#include "attn_core/sparse_flash_wmma.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
namespace T = strix::attn_core::tile;
namespace S = strix::attn_core::sparse;

void hok(hipError_t e, const char* w) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, hipGetErrorString(e));
        std::exit(2);
    }
}
std::uint16_t bf16(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U);
}
float fbf(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16U;
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}

template <unsigned H_, unsigned D_, bool Shared> struct TestPolicy {
    struct Args {
        const std::uint16_t* k;
        const std::uint16_t* v;
        const std::uint32_t* list;
        const std::uint32_t* n;
        const float* q;
        float* o;
        unsigned stride;
    };
    static constexpr unsigned kHeads = H_, kDim = D_;
    static constexpr T::Operand kOp = T::Operand::bf16;
    static constexpr float kScaleLog2 = 1.4426950408889634F / 16.0F;
    __device__ static unsigned count(const Args& a, unsigned r) { return a.n[r]; }
    __device__ static unsigned key(const Args& a, unsigned r, unsigned j) { return a.list[static_cast<std::size_t>(r) * a.stride + j]; }
    __device__ static const std::uint16_t* key_row(const Args& a, unsigned, unsigned p) { return a.k + static_cast<std::size_t>(p) * D_; }
    __device__ static const std::uint16_t* value_row(const Args& a, unsigned, unsigned p) {
        return (Shared ? a.k : a.v) + static_cast<std::size_t>(p) * D_;
    }
    __device__ static const float* query(const Args& a, unsigned r) { return a.q + static_cast<std::size_t>(r) * H_ * D_; }
    __device__ static float* out(const Args& a, unsigned r) { return a.o + static_cast<std::size_t>(r) * H_ * D_; }
};

template <unsigned H, unsigned D, bool Shared, unsigned TB> bool run(const char* name, unsigned rows, unsigned cache, unsigned maxn, bool timing) {
    using P = TestPolicy<H, D, Shared>;
    std::mt19937 g(1234U + H + D);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<std::uint16_t> k(static_cast<std::size_t>(cache) * D), v(k.size());
    for (auto& x : k) x = bf16(nd(g));
    for (auto& x : v) x = bf16(nd(g));
    std::vector<float> q(static_cast<std::size_t>(rows) * H * D);
    for (auto& x : q) x = 0.5F * nd(g);
    std::vector<std::uint32_t> list(static_cast<std::size_t>(rows) * maxn), n(rows);
    for (unsigned r = 0; r < rows; ++r) {
        n[r] = timing ? maxn : (r % 7 == 0 ? maxn : 1U + static_cast<unsigned>(g() % maxn));
        for (unsigned j = 0; j < n[r]; ++j) list[static_cast<std::size_t>(r) * maxn + j] = static_cast<std::uint32_t>(g() % cache);
    }
    std::uint16_t *dk, *dv;
    std::uint32_t *dl, *dn;
    float *dq, *dout;
    hok(hipMalloc(&dk, k.size() * 2), "m");
    hok(hipMalloc(&dv, v.size() * 2), "m");
    hok(hipMalloc(&dl, list.size() * 4), "m");
    hok(hipMalloc(&dn, n.size() * 4), "m");
    hok(hipMalloc(&dq, q.size() * 4), "m");
    hok(hipMalloc(&dout, q.size() * 4), "m");
    hok(hipMemcpy(dk, k.data(), k.size() * 2, hipMemcpyHostToDevice), "c");
    hok(hipMemcpy(dv, v.data(), v.size() * 2, hipMemcpyHostToDevice), "c");
    hok(hipMemcpy(dl, list.data(), list.size() * 4, hipMemcpyHostToDevice), "c");
    hok(hipMemcpy(dn, n.data(), n.size() * 4, hipMemcpyHostToDevice), "c");
    hok(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice), "c");
    const typename P::Args a{dk, dv, dl, dn, dq, dout, maxn};
    constexpr unsigned kThr = S::Geometry<H, D, TB>::kThreads;
    S::flash_rows_kernel<P, TB><<<rows, kThr>>>(a);
    hok(hipGetLastError(), "launch");
    hok(hipDeviceSynchronize(), "sync");
    bool pass = true;
    if (!timing) {
        std::vector<float> o(q.size());
        hok(hipMemcpy(o.data(), dout, o.size() * 4, hipMemcpyDeviceToHost), "c");
        double worst = 0, sum = 0;
        const std::vector<std::uint16_t>& vv = Shared ? k : v;
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned h = 0; h < H; ++h) {
                const float* qr = &q[(static_cast<std::size_t>(r) * H + h) * D];
                std::vector<double> s(n[r]);
                double mx = -1e300;
                for (unsigned j = 0; j < n[r]; ++j) {
                    const std::uint16_t* kr = &k[static_cast<std::size_t>(list[static_cast<std::size_t>(r) * maxn + j]) * D];
                    double d = 0;
                    for (unsigned i = 0; i < D; ++i) d += static_cast<double>(qr[i]) * fbf(kr[i]);
                    s[j] = d / 16.0;
                    mx = std::max(mx, s[j]);
                }
                double l = 0;
                for (auto& x : s) l += (x = std::exp(x - mx));
                double e2 = 0, r2 = 0;
                for (unsigned i = 0; i < D; ++i) {
                    double ref = 0;
                    for (unsigned j = 0; j < n[r]; ++j)
                        ref += s[j] * fbf(vv[static_cast<std::size_t>(list[static_cast<std::size_t>(r) * maxn + j]) * D + i]);
                    ref /= l;
                    const double d = o[(static_cast<std::size_t>(r) * H + h) * D + i] - ref;
                    e2 += d * d;
                    r2 += ref * ref;
                }
                const double rel = std::sqrt(e2 / std::max(r2, 1e-30));
                worst = std::max(worst, std::isfinite(rel) ? rel : 1e9);
                sum += rel;
            }
        pass = worst < 2e-2;
        std::printf("%s T%u: rows %u, row/head rel L2 mean %.3e worst %.3e (bound 2e-2) -> %s\n", name, TB, rows, sum / (rows * H), worst,
                    pass ? "PASS" : "FAIL");
    } else {
        hipEvent_t e0, e1;
        hok(hipEventCreate(&e0), "e");
        hok(hipEventCreate(&e1), "e");
        float best = 1e30F;
        for (unsigned it = 0; it < 5; ++it) {
            hok(hipEventRecord(e0, nullptr), "e");
            S::flash_rows_kernel<P, TB><<<rows, kThr>>>(a);
            hok(hipEventRecord(e1, nullptr), "e");
            hok(hipEventSynchronize(e1), "e");
            float ms = 0;
            hok(hipEventElapsedTime(&ms, e0, e1), "e");
            best = std::min(best, ms);
        }
        const double flop = 4.0 * rows * H * D * maxn;
        std::printf("%s T%u timing: rows %u x keys %u: %.3f ms, %.2f TF/s (bf16 WMMA peak ~57)\n", name, TB, rows, maxn, best, flop / best * 1e-9);
    }
    (void)hipFree(dk);
    (void)hipFree(dv);
    (void)hipFree(dl);
    (void)hipFree(dn);
    (void)hipFree(dq);
    (void)hipFree(dout);
    return pass;
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "bench";
    bool ok = true;
    ok &= run<32, 512, true, 32>("glm-mla H32 D512", 48, 8192, 2051, false);
    ok &= run<32, 512, true, 16>("glm-mla H32 D512", 48, 8192, 2051, false);
    ok &= run<16, 128, false, 32>("msa H16 D128", 96, 8192, 2048, false);
    if (bench) {
        run<32, 512, true, 32>("glm-mla H32 D512", 4096, 8192, 2048, true);
        run<32, 512, true, 16>("glm-mla H32 D512", 4096, 8192, 2048, true);
        run<16, 128, false, 32>("msa H16 D128", 8192, 32768, 2048, true);
    }
    std::printf("attn-core sparse tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
