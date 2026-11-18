// Tests + microbench of the one-launch index projections (src/minimax/attn/index_proj.hpp): every output bit for bit
// equal to the gate's dense_t (FP32 transposed weights) and to the engine's dense_t16 (both copied below verbatim from
// src/minimax/engine/mm_kernels.hpp at minimax/engine f2f84f43), decode rows 1..8 x ring 8/16/32 and the tiled
// prefill kernel at ragged token counts. STRIX_MMATTN_BENCH=1 adds ABA x3 timings with the weights of 60 layers
// rotated (cold, like the real layer loop). FAST minimax-index-proj-wmma-v1: contract / drift / same-bits checks
// always; STRIX_MMATTN_BENCH=wmma its strict ABAB bench (STRIX_MMATTN_WMMA_SERIES = x16 | diag | orders).
// Build (gfx1151): clang++ -x hip --offload-arch=gfx1151 -mno-wavefrontsize64 --rocm-path=<rocm> -O3 -std=c++20
//   -Isrc -Iinclude --hip-link src/minimax/attn/index_proj.hip src/minimax/attn/index_proj_tests.hip -o index-proj-tests
#include "index_proj.hpp"
#include "strix/env_flags.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#if defined(__linux__)
#include <filesystem>
#include <fstream>
#include <unistd.h>
#endif

namespace ma = strix::minimax::attn;

namespace {

constexpr unsigned kHidden = 6144;

// --- references (verbatim from minimax/engine mm_kernels.hpp) ---
__global__ void dense_t_kernel(const float* x, const float* wt, float* y, unsigned d, unsigned n) {
    __shared__ float xs[kHidden];
    const float* xr = x + static_cast<std::size_t>(blockIdx.x) * d;
    for (unsigned i = threadIdx.x; i < d; i += 128) xs[i] = xr[i];
    __syncthreads();
    const unsigned o = blockIdx.y * 128 + threadIdx.x;
    if (o >= n) return;
    float acc = 0.0F;
    for (unsigned i = 0; i < d; ++i) acc = __builtin_fmaf(xs[i], wt[static_cast<std::size_t>(i) * n + o], acc);
    y[static_cast<std::size_t>(blockIdx.x) * n + o] = acc;
}
template <unsigned TT>
__global__ void dense_t16_kernel(const float* x, const _Float16* wt, float* y, unsigned d, unsigned n, unsigned rows) {
    __shared__ float xs[TT][64];
    const unsigned t0 = blockIdx.x * TT;
    const unsigned o = blockIdx.y * 128 + threadIdx.x;
    float acc[TT];
    for (unsigned t = 0; t < TT; ++t) acc[t] = 0.0F;
    for (unsigned i0 = 0; i0 < d; i0 += 64) {
        __syncthreads();
        for (unsigned j = threadIdx.x; j < TT * 64; j += 128) {
            const unsigned t = j / 64, i = j % 64;
            xs[t][i] = t0 + t < rows ? x[static_cast<std::size_t>(t0 + t) * d + i0 + i] : 0.0F;
        }
        __syncthreads();
        if (o < n)
            for (unsigned i = 0; i < 64; ++i) {
                const float w = static_cast<float>(wt[static_cast<std::size_t>(i0 + i) * n + o]);
                for (unsigned t = 0; t < TT; ++t) acc[t] = __builtin_fmaf(xs[t][i], w, acc[t]);
            }
    }
    if (o < n)
        for (unsigned t = 0; t < TT; ++t)
            if (t0 + t < rows) y[static_cast<std::size_t>(t0 + t) * n + o] = acc[t];
}

int g_fail = 0;
void hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "HIP error %s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}
template <class T> T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    hip_ok(hipMalloc(&d, std::max<std::size_t>(1, h.size()) * sizeof(T)), "malloc");
    if (!h.empty()) hip_ok(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return d;
}
template <class T> std::vector<T> host(const T* d, std::size_t n) {
    std::vector<T> h(n);
    hip_ok(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
    return h;
}
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// One layer's weights in every layout.
struct Weights {
    std::vector<float> wq, wk;  // [256][6144], [128][6144] (F16 values)
    float *tq, *tk;             // dense_t: transposed FP32 [6144][256], [6144][128]
    _Float16 *hq, *hk;          // dense_t16: transposed F16
    _Float16* packed;           // index_proj
};
Weights make_weights(std::mt19937& rng, bool keep_host) {
    Weights w{};
    std::normal_distribution<float> n(0.0F, 0.02F);
    w.wq.resize(std::size_t(ma::kProjQ) * kHidden);
    w.wk.resize(std::size_t(ma::kProjK) * kHidden);
    for (auto* v : {&w.wq, &w.wk})
        for (auto& x : *v) x = static_cast<float>(static_cast<_Float16>(n(rng)));
    auto transpose = [](const std::vector<float>& m, unsigned rows) {
        std::vector<float> t(m.size());
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned i = 0; i < kHidden; ++i) t[std::size_t(i) * rows + r] = m[std::size_t(r) * kHidden + i];
        return t;
    };
    const auto tq = transpose(w.wq, ma::kProjQ), tk = transpose(w.wk, ma::kProjK);
    auto to16 = [](const std::vector<float>& f) {
        std::vector<_Float16> h(f.size());
        for (std::size_t i = 0; i < f.size(); ++i) h[i] = static_cast<_Float16>(f[i]);
        return h;
    };
    w.tq = keep_host ? dev(tq) : nullptr;
    w.tk = keep_host ? dev(tk) : nullptr;
    w.hq = dev(to16(tq));
    w.hk = dev(to16(tk));
    w.packed = dev(ma::pack_index_proj(w.wq.data(), w.wk.data()));
    if (!keep_host) {
        w.wq.clear();
        w.wk.clear();
    }
    return w;
}
void free_weights(Weights& w) {
    for (void* p : {static_cast<void*>(w.tq), static_cast<void*>(w.tk), static_cast<void*>(w.hq), static_cast<void*>(w.hk),
                    static_cast<void*>(w.packed)})
        if (p != nullptr) (void)hipFree(p);
}

void engine_t16(const float* x, const Weights& w, float* iq, float* ik, unsigned rows, hipStream_t st) {
    if (rows <= 8) {
        hipLaunchKernelGGL(dense_t16_kernel<8>, dim3(1, ma::kProjQ / 128), dim3(128), 0, st, x, w.hq, iq, kHidden, ma::kProjQ, rows);
        hipLaunchKernelGGL(dense_t16_kernel<8>, dim3(1, 1), dim3(128), 0, st, x, w.hk, ik, kHidden, ma::kProjK, rows);
    } else {
        const unsigned gx = (rows + 15) / 16;
        hipLaunchKernelGGL(dense_t16_kernel<16>, dim3(gx, ma::kProjQ / 128), dim3(128), 0, st, x, w.hq, iq, kHidden, ma::kProjQ, rows);
        hipLaunchKernelGGL(dense_t16_kernel<16>, dim3(gx, 1), dim3(128), 0, st, x, w.hk, ik, kHidden, ma::kProjK, rows);
    }
}

void tests(std::mt19937& rng) {
    Weights w = make_weights(rng, true);
    const unsigned maxT = 2050;  // one guard row even for the largest boundary case
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> xh(std::size_t(maxT) * kHidden);
    for (auto& v : xh) v = n(rng);
    float* x = dev(xh);
    float *rq = dev(std::vector<float>(std::size_t(maxT) * ma::kProjQ)), *rk = dev(std::vector<float>(std::size_t(maxT) * ma::kProjK));
    float *iq = dev(std::vector<float>(std::size_t(maxT) * ma::kProjQ)), *ik = dev(std::vector<float>(std::size_t(maxT) * ma::kProjK));
    auto same = [&](unsigned rows, const char* what) {
        hip_ok(hipDeviceSynchronize(), "sync");
        const bool ok = host(iq, std::size_t(rows) * ma::kProjQ) == host(rq, std::size_t(rows) * ma::kProjQ) &&
                        host(ik, std::size_t(rows) * ma::kProjK) == host(rk, std::size_t(rows) * ma::kProjK);
        // == on floats: also require the bits (no -0/+0 or NaN slack)
        const auto a = host(iq, std::size_t(rows) * ma::kProjQ), b = host(rq, std::size_t(rows) * ma::kProjQ);
        const auto c = host(ik, std::size_t(rows) * ma::kProjK), d = host(rk, std::size_t(rows) * ma::kProjK);
        const bool bits = std::memcmp(a.data(), b.data(), a.size() * 4) == 0 && std::memcmp(c.data(), d.data(), c.size() * 4) == 0;
        if (!(ok && bits)) {
            ++g_fail;
            std::fprintf(stderr, "FAIL: %s rows %u\n", what, rows);
        }
        hip_ok(hipMemset(iq, 0xFF, std::size_t(maxT) * ma::kProjQ * 4), "memset");
        hip_ok(hipMemset(ik, 0xFF, std::size_t(maxT) * ma::kProjK * 4), "memset");
    };
    for (unsigned rows : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 16U, 17U, 63U, 64U, 65U, 300U, 1100U, 2047U, 2048U, 2049U}) {
        hipLaunchKernelGGL(dense_t_kernel, dim3(rows, ma::kProjQ / 128), dim3(128), 0, nullptr, x, w.tq, rq, kHidden, ma::kProjQ);
        hipLaunchKernelGGL(dense_t_kernel, dim3(rows, 1), dim3(128), 0, nullptr, x, w.tk, rk, kHidden, ma::kProjK);
        engine_t16(x, w, iq, ik, rows, nullptr);
        same(rows, "engine dense_t16 vs dense_t");
        if (rows <= 8)
            for (unsigned ring : {8U, 16U, 32U, 48U})
                for (unsigned rpw : {0U, 2U}) {
                    ma::enqueue_index_proj_decode({x, w.packed, iq, ik, rows}, ring, nullptr, rpw);
                    same(rows, ("decode ring " + std::to_string(ring) + " rows/wave " + std::to_string(rpw)).c_str());
                }
        if (rows >= 8) {
            ma::enqueue_index_proj_tiled({x, w.packed, iq, ik, rows}, nullptr);
            same(rows, "tiled");
        }
        if (rows >= 8) {
            ma::enqueue_index_proj_slab({x, w.packed, iq, ik, rows}, nullptr);
            same(rows, "slab");
        }
        if (rows >= 2 && rows <= 8) {
            ma::enqueue_index_proj_decode_lds({x, w.packed, iq, ik, rows}, nullptr);
            same(rows, "decode lds");
        }
        ma::enqueue_index_proj({x, w.packed, iq, ik, rows}, nullptr);
        same(rows, "default");
        std::printf("index_proj rows %u: %s\n", rows, g_fail == 0 ? "bitwise equal (dense_t, dense_t16, decode rings, tiled, slab)" : "MISMATCH");
    }
    for (void* p : {static_cast<void*>(x), static_cast<void*>(rq), static_cast<void*>(rk), static_cast<void*>(iq), static_cast<void*>(ik)})
        (void)hipFree(p);
    free_weights(w);
}

// Check bytes rather than floating-point values: the sentinel is an all-ones NaN.
template <class T> bool ff_tail(const std::vector<T>& values, std::size_t live) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
    for (std::size_t i = live * sizeof(T); i < values.size() * sizeof(T); ++i)
        if (bytes[i] != 0xFFU) return false;
    return true;
}

// FAST minimax-index-proj-wmma-v1 contract: (a) against dense_t on x rounded to FP16 (the same products in an FP32
// chain: only the accumulation order differs), (b) drift against the exact chains on x (the FP16 rounding of x),
// (c) tc 4 / 2 / 1 bit for bit equal, (d) finite outputs and rows past `rows` untouched. x carries a few outlier
// channels (x 24) like a norm output.
void wmma_tests(std::mt19937& rng) {
    Weights w = make_weights(rng, true);
    const unsigned maxT = 2050;  // one guard row even for the largest boundary case
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> xh(std::size_t(maxT) * kHidden), x16h(xh.size());
    for (std::size_t i = 0; i < xh.size(); ++i) {
        xh[i] = n(rng) * (i % kHidden % 997U == 5U ? 24.0F : 1.0F);
        x16h[i] = static_cast<float>(static_cast<_Float16>(xh[i]));
    }
    float *x = dev(xh), *x16 = dev(x16h);
    const std::size_t nq = std::size_t(maxT) * ma::kProjQ, nk = std::size_t(maxT) * ma::kProjK;
    float *eq = dev(std::vector<float>(nq)), *ek = dev(std::vector<float>(nk)), *cq = dev(std::vector<float>(nq)),
          *ck = dev(std::vector<float>(nk)), *iq = dev(std::vector<float>(nq)), *ik = dev(std::vector<float>(nk));
    _Float16* xt = dev(std::vector<_Float16>(std::size_t(maxT + 1) * kHidden));  // prefix and suffix guards
    auto rel_l2 = [](const std::vector<float>& a, const std::vector<float>& b, double& max_rel) {
        double num = 0, den = 0, mx = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double d = double(a[i]) - double(b[i]);
            num += d * d;
            den += double(b[i]) * double(b[i]);
            mx = std::max(mx, std::abs(d));
        }
        max_rel = mx / std::sqrt(den / double(std::max<std::size_t>(1, b.size())));
        return std::sqrt(num / std::max(den, 1e-300));
    };
    for (unsigned rows : {1U, 15U, 16U, 127U, 128U, 129U, 300U, 1100U, 2047U, 2048U, 2049U}) {
        for (auto [xin, q, k] : {std::tuple{x, eq, ek}, std::tuple{x16, cq, ck}}) {
            hipLaunchKernelGGL(dense_t_kernel, dim3(rows, ma::kProjQ / 128), dim3(128), 0, nullptr, xin, w.tq, q, kHidden, ma::kProjQ);
            hipLaunchKernelGGL(dense_t_kernel, dim3(rows, 1), dim3(128), 0, nullptr, xin, w.tk, k, kHidden, ma::kProjK);
        }
        std::vector<float> first_q, first_k;
        for (const auto& [tc, rows_fast] : {std::pair{4U, false}, std::pair{2U, false}, std::pair{1U, false}, std::pair{4U, true}}) {
            hip_ok(hipMemset(iq, 0xFF, nq * 4), "memset");
            hip_ok(hipMemset(ik, 0xFF, nk * 4), "memset");
            ma::enqueue_index_proj_wmma({x, w.packed, iq, ik, rows}, nullptr, tc, rows_fast);
            hip_ok(hipDeviceSynchronize(), "sync");
            const auto aq = host(iq, std::size_t(rows + (rows < maxT ? 1 : 0)) * ma::kProjQ);
            const auto ak = host(ik, std::size_t(rows + (rows < maxT ? 1 : 0)) * ma::kProjK);
            bool guard = true, finite = true;
            auto scan = [&](const std::vector<float>& a, std::size_t live) {  // [0, live) finite, the rest still 0xFF
                for (std::size_t i = 0; i < a.size(); ++i) {
                    if (i < live) finite = finite && std::isfinite(a[i]);
                    else guard = guard && std::bit_cast<std::uint32_t>(a[i]) == 0xFFFFFFFFU;
                }
            };
            scan(aq, std::size_t(rows) * ma::kProjQ);
            scan(ak, std::size_t(rows) * ma::kProjK);
            std::vector<float> lq(aq.begin(), aq.begin() + std::size_t(rows) * ma::kProjQ), lk(ak.begin(), ak.begin() + std::size_t(rows) * ma::kProjK);
            if (tc == 4 && !rows_fast) {
                double mq = 0, mk = 0, dq = 0, dk = 0;
                const double cq_l2 = rel_l2(lq, host(cq, lq.size()), mq), ck_l2 = rel_l2(lk, host(ck, lk.size()), mk);
                const double eq_l2 = rel_l2(lq, host(eq, lq.size()), dq), ek_l2 = rel_l2(lk, host(ek, lk.size()), dk);
                std::printf("wmma rows %u: contract rel-l2 iq %.2e ik %.2e | drift vs exact rel-l2 iq %.2e ik %.2e, max |err|/rms iq %.2e ik %.2e\n",
                            rows, cq_l2, ck_l2, eq_l2, ek_l2, dq, dk);
                // contract: accumulation order only, measured 1.9-2.0e-5 at every row count 1..1100 (uniform: the WMMA's
                // internal accumulation, 6144 terms); drift (x rounded to FP16): measured 1.8-2.2e-4.
                if (!(cq_l2 < 4e-5 && ck_l2 < 4e-5 && eq_l2 < 2e-3 && ek_l2 < 2e-3)) {
                    ++g_fail;
                    std::fprintf(stderr, "FAIL: wmma rows %u contract / drift bound\n", rows);
                }
                first_q = std::move(lq);
                first_k = std::move(lk);
            } else if (std::memcmp(lq.data(), first_q.data(), lq.size() * 4) != 0 || std::memcmp(lk.data(), first_k.data(), lk.size() * 4) != 0) {
                ++g_fail;
                std::fprintf(stderr, "FAIL: wmma rows %u tc %u rows_fast %d differs from tc 4\n", rows, tc, int(rows_fast));
            }
            if (!guard || !finite) {
                ++g_fail;
                std::fprintf(stderr, "FAIL: wmma rows %u tc %u rows_fast %d guard %d finite %d\n", rows, tc, int(rows_fast), int(guard), int(finite));
            }
        }
        for (int form = 0; form < 3; ++form) {  // the 192-VGPR form, the x16 tiles form and the FAST entry: same bits as tc 4
            hip_ok(hipMemset(iq, 0xFF, nq * 4), "memset");
            hip_ok(hipMemset(ik, 0xFF, nk * 4), "memset");
            hip_ok(hipMemset(xt, 0xFF, std::size_t(maxT + 1) * kHidden * sizeof(*xt)), "scratch memset");
            if (form == 0) ma::enqueue_index_proj_wmma_diag({x, w.packed, iq, ik, rows}, 2, nullptr);
            else if (form == 1) ma::enqueue_index_proj_wmma_x16({x, w.packed, iq, ik, rows}, xt + kHidden, nullptr);
            else ma::enqueue_index_proj_fast({x, w.packed, iq, ik, rows}, xt + kHidden, nullptr);
            hip_ok(hipDeviceSynchronize(), "sync");
            const auto aq = host(iq, std::size_t(rows + 1) * ma::kProjQ);
            const auto ak = host(ik, std::size_t(rows + 1) * ma::kProjK);
            const bool guard = ff_tail(aq, first_q.size()) && ff_tail(ak, first_k.size()) &&
                               ff_tail(host(xt, kHidden), 0) &&
                               ff_tail(host(xt + std::size_t(rows + 1) * kHidden, kHidden), 0);
            if (!guard || std::memcmp(aq.data(), first_q.data(), first_q.size() * sizeof(float)) != 0 ||
                std::memcmp(ak.data(), first_k.data(), first_k.size() * sizeof(float)) != 0) {
                ++g_fail;
                std::fprintf(stderr, "FAIL: wmma rows %u %s form differs from tc 4 or corrupts guards\n", rows, form == 0 ? "192-VGPR" : form == 1 ? "x16" : "fast entry");
            }
        }
    }
    for (void* p : {static_cast<void*>(x), static_cast<void*>(x16), static_cast<void*>(eq), static_cast<void*>(ek), static_cast<void*>(cq),
                    static_cast<void*>(ck), static_cast<void*>(iq), static_cast<void*>(ik), static_cast<void*>(xt)})
        (void)hipFree(p);
    free_weights(w);
}

#if defined(__linux__)
// Sum of evicted_ms over this process's KFD stats entries (0 when absent).
long kfd_evicted_ms() {
    long total = 0;
    std::error_code ec;
    const std::string dir = "/sys/class/kfd/kfd/proc/" + std::to_string(::getpid());
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (e.path().filename().string().rfind("stats_", 0) != 0) continue;
        std::ifstream f(e.path() / "evicted_ms");
        long v = 0;
        if (f >> v) total += v;
    }
    return total;
}
#else
long kfd_evicted_ms() { return 0; }
#endif

// STRIX_MMATTN_BENCH=wmma: strict A B A B A B (3 pairs), fixed first 12/30 of 60 weight sets. Series 1: A = slab
// (exact default), B = wmma tc 4; series 2: A = wmma tc 4 rows-fast CTA order, B = tc 4 (column-fast); series 3:
// A = tc 4, B = tc 2. Cold = the first call of each arm in the process (code object load excluded by one 8-row call
// of every kernel first), warm = the medians. First run (rows-fast only, 18:05 11-15): 4096 rows slab 2740 tc4 4290
// tc2 6878 tc1 14687 us; 1024 rows 673 / 209 / 326 / 602 us.
void bench_wmma(std::mt19937& rng) {
    const unsigned L = 60;
    std::vector<Weights> ws;
    for (unsigned l = 0; l < L; ++l) ws.push_back(make_weights(rng, false));
    const unsigned maxT = 8192;
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> xh(std::size_t(maxT) * kHidden);
    for (auto& v : xh) v = n(rng);
    float* x = dev(xh);
    float *iq = dev(std::vector<float>(std::size_t(maxT) * ma::kProjQ)), *ik = dev(std::vector<float>(std::size_t(maxT) * ma::kProjK));
    _Float16* xt = dev(std::vector<_Float16>(std::size_t(maxT) * kHidden));  // x16 tiles (arm 8 times the WMMA on them alone)
    hipStream_t st;
    hip_ok(hipStreamCreateWithFlags(&st, hipStreamNonBlocking), "stream");
    hipEvent_t e0, e1;
    hip_ok(hipEventCreate(&e0), "event");
    hip_ok(hipEventCreate(&e1), "event");
    auto launch = [&](int arm, unsigned l, unsigned rows) {
        const ma::IndexProjView v{x, ws[l].packed, iq, ik, rows};
        if (arm == 0) ma::enqueue_index_proj_slab(v, st);
        else if (arm >= 7) ma::enqueue_index_proj_wmma_x16(v, xt, st, arm == 7);
        else if (arm >= 4) ma::enqueue_index_proj_wmma_diag(v, arm - 3, st);
        else ma::enqueue_index_proj_wmma(v, st, arm == 3 ? 2U : 4U, arm == 2);
    };
    auto timed = [&](int arm, unsigned rows, unsigned layers, bool warm) {
        // The packed stride includes rows. Repack on this stream, outside the events,
        // even when a preceding case happened to use the same shape.
        if (arm == 8) launch(7, 0, rows);
        if (warm) launch(arm, 0, rows);
        hip_ok(hipEventRecord(e0, st), "rec");
        for (unsigned l = 0; l < layers; ++l) launch(arm, l, rows);
        hip_ok(hipEventRecord(e1, st), "rec");
        hip_ok(hipEventSynchronize(e1), "sync");
        float ms = 0;
        hip_ok(hipEventElapsedTime(&ms, e0, e1), "elapsed");
        return 1000.0 * ms / layers;
    };
    auto validate_x16 = [&](unsigned rows, unsigned layers) {
        launch(7, 0, rows);  // prepare the current shape on st before any conversion-free use
        for (unsigned l = 0; l < layers; ++l) {
            launch(1, l, rows);
            hip_ok(hipStreamSynchronize(st), "reference sync");
            const auto rq = host(iq, std::size_t(rows) * ma::kProjQ);
            const auto rk = host(ik, std::size_t(rows) * ma::kProjK);
            launch(8, l, rows);
            hip_ok(hipStreamSynchronize(st), "x16 sync");
            const auto aq = host(iq, rq.size()), ak = host(ik, rk.size());
            if (std::memcmp(aq.data(), rq.data(), rq.size() * sizeof(float)) != 0 ||
                std::memcmp(ak.data(), rk.data(), rk.size() * sizeof(float)) != 0) {
                ++g_fail;
                std::fprintf(stderr, "FAIL: x16-only rows %u layer %u differs from direct WMMA; timings rejected\n", rows, l);
                return false;
            }
        }
        return true;
    };
    for (int arm = 0; arm < 9; ++arm) launch(arm, 0, 8);  // code objects loaded, not timed
    hip_ok(hipStreamSynchronize(st), "sync");
    const double cold_a = timed(0, 4096, 1, false), cold_b = timed(1, 4096, 1, false);
    std::printf("wmma bench cold (first 4096-row call, layer 0) us: slab %.0f wmma-tc4 %.0f\n", cold_a, cold_b);
    // Run 2 (18:25 11-15, column-fast): 4096 rows tc4-rowsfast 4426 vs tc4 1715; tc2 1.21-1.74x of tc4 at 1024..8192.
    // Run 3 (18:47 11-15, gfxclk busy median 2401 MHz): 4096 rows tc4 1705 vs x-cached 599 us, vs 192-VGPR 3366 us.
    // Run 4 (19:05): one column block alone (x read once) 4096 rows 1079 us, 2048 431, 1024 100: x stream ~93-114 GB/s.
    const char* names[9] = {"slab",           "wmma-tc4",       "wmma-tc4-rowsfast", "wmma-tc2", "diag-xcached", "wmma-tc4-vgpr192",
                            "diag-1colblock", "wmma-x16(conv+wmma)", "wmma-x16-only"};
    // "x16" (default): slab|tc4, tc4|x16, tc4|x16-only, slab|x16; "diag": slab|tc4, tc4|x-cached, tc4|one column block;
    // "orders": slab|tc4, rowsfast|tc4, tc4|tc2, tc4|vgpr192
    const int series = strix::envcfg::choice("STRIX_MMATTN_WMMA_SERIES", {"x16", "diag", "orders"}, 0);
    const std::vector<std::pair<int, int>> pairs =
        series == 2 ? std::vector<std::pair<int, int>>{{0, 1}, {2, 1}, {1, 3}, {1, 5}}
        : series == 1 ? std::vector<std::pair<int, int>>{{0, 1}, {1, 4}, {1, 6}}
                      : std::vector<std::pair<int, int>>{{0, 1}, {1, 7}, {1, 8}, {0, 7}};
    for (const auto& [a, b] : pairs)
        for (unsigned rows : {4096U, 1024U, 2048U, 8192U}) {
            const unsigned layers = rows > 2048 ? 12 : 30;
            if ((a == 8 || b == 8) && !validate_x16(rows, layers)) continue;
            std::vector<double> ta, tb, ratio;
            std::string runs;
            for (int pair = 0; pair < 3; ++pair) {
                const double x1 = timed(a, rows, layers, true), y1 = timed(b, rows, layers, true);
                ta.push_back(x1);
                tb.push_back(y1);
                ratio.push_back(y1 / x1);
                char buf[64];
                std::snprintf(buf, sizeof buf, " [%.0f %.0f]", x1, y1);
                runs += buf;
            }
            std::printf("wmma bench T %u sampled layers [0,%u) of %u us/call: A %s %.0f B %s %.0f (B/A median %.3f, worst %.3f, last pair %.3f) ABAB:%s\n", rows, layers, L,
                        names[a], median(ta), names[b], median(tb), median(ratio), *std::max_element(ratio.begin(), ratio.end()),
                        ratio.back(), runs.c_str());
        }
    std::printf("wmma bench kfd evicted_ms %ld\n", kfd_evicted_ms());
    (void)hipEventDestroy(e0);
    (void)hipEventDestroy(e1);
    (void)hipStreamDestroy(st);
    for (auto& w : ws) free_weights(w);
    (void)hipFree(x);
    (void)hipFree(iq);
    (void)hipFree(ik);
    (void)hipFree(xt);
}

// slab_only (STRIX_MMATTN_BENCH=slab): only the prefill slab gate.
void bench(std::mt19937& rng, bool slab_only) {
    const unsigned L = 60;
    std::vector<Weights> ws;
    for (unsigned l = 0; l < L; ++l) ws.push_back(make_weights(rng, false));
    const unsigned maxT = 8192;
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> xh(std::size_t(maxT) * kHidden);
    for (auto& v : xh) v = n(rng);
    float* x = dev(xh);
    float *iq = dev(std::vector<float>(std::size_t(maxT) * ma::kProjQ)), *ik = dev(std::vector<float>(std::size_t(maxT) * ma::kProjK));
    hipStream_t st;
    hip_ok(hipStreamCreateWithFlags(&st, hipStreamNonBlocking), "stream");
    hipEvent_t e0, e1;
    hip_ok(hipEventCreate(&e0), "event");
    hip_ok(hipEventCreate(&e1), "event");
    auto time_run = [&](auto&& body, unsigned layers, int reps) {
        body(0U);
        hip_ok(hipEventRecord(e0, st), "rec");
        for (int r = 0; r < reps; ++r)
            for (unsigned l = 0; l < layers; ++l) body(l);
        hip_ok(hipEventRecord(e1, st), "rec");
        hip_ok(hipEventSynchronize(e1), "sync");
        float ms = 0;
        hip_ok(hipEventElapsedTime(&ms, e0, e1), "elapsed");
        return 1000.0 * ms / (double(reps) * layers);
    };
    // Change B gate: literal A B A x3 for every R2..R8, A = the round-2 default (ring 16 + row split), B = LDS kernel.
    if (!slab_only) for (unsigned rows = 2; rows <= 8; ++rows) {
        auto old_k = [&](unsigned l) { ma::enqueue_index_proj_decode({x, ws[l].packed, iq, ik, rows}, 16U, st, rows > 2 ? 2U : 0U); };
        auto lds_k = [&](unsigned l) { ma::enqueue_index_proj_decode_lds({x, ws[l].packed, iq, ik, rows}, st); };
        std::vector<double> ta, tb, ratio;
        std::string runs;
        for (int round = 0; round < 3; ++round) {
            const double a1 = time_run(old_k, L, 10), bb = time_run(lds_k, L, 10), a2 = time_run(old_k, L, 10);
            ta.push_back(a1);
            ta.push_back(a2);
            tb.push_back(bb);
            ratio.push_back(bb / (0.5 * (a1 + a2)));
            char b[80];
            std::snprintf(b, sizeof b, " [%.1f %.1f %.1f r%.3f]", a1, bb, a2, ratio.back());
            runs += b;
        }
        std::printf("decode rows %u us/layer: round-2 %.1f lds %.1f (B/A median %.3f, worst bracket %.3f)  ABA:%s\n", rows, median(ta),
                    median(tb), median(ratio), *std::max_element(ratio.begin(), ratio.end()), runs.c_str());
        std::fflush(stdout);
    }
    // Diagnostic (not an ABA): engine dense_t16<8> (2 launches), ring32, round-2 default, LDS; order E R O L E x3.
    if (!slab_only) for (unsigned rows : {1U, 2U, 4U, 6U, 8U}) {
        auto arm = [&](int a) {
            return [&, a](unsigned l) {
                if (a == 0) engine_t16(x, ws[l], iq, ik, rows, st);
                else if (a == 3 && rows >= 2) ma::enqueue_index_proj_decode_lds({x, ws[l].packed, iq, ik, rows}, st);
                else ma::enqueue_index_proj_decode({x, ws[l].packed, iq, ik, rows}, a == 2 && rows >= 2 ? 16U : 32U, st, a == 1 ? 0U : 2U);
            };
        };
        std::vector<double> t[4];
        std::string runs;
        for (int round = 0; round < 3; ++round) {
            runs += " [";
            for (int k = 0; k <= 4; ++k) {
                const int a = k == 4 ? 0 : k;
                const double us = time_run(arm(a), L, 10);
                t[a].push_back(us);
                char b[24];
                std::snprintf(b, sizeof b, k == 0 ? "%.1f" : " %.1f", us);
                runs += b;
            }
            runs += "]";
        }
        const double ma0 = median(t[0]);
        std::printf("decode rows %u us/layer (60 layers' weights rotated, 4.72 MB/layer): A(engine dense_t16) %.1f ring32 %.1f (%.3f)"
                    " current-default %.1f (%.3f) lds-coop %.1f (%.3f)  order E R O L E:%s\n", rows, ma0, median(t[1]), median(t[1]) / ma0, median(t[2]),
                    median(t[2]) / ma0, median(t[3]), median(t[3]) / ma0, runs.c_str());
        std::fflush(stdout);
    }
    // Diagnostic decomposition of the decode kernel (rows 1 and 2): full / no weight loads / pure FMA chain / no FMAs.
    if (!slab_only) for (unsigned rows : {1U, 2U, 8U}) {
        std::vector<double> t[4];
        for (int round = 0; round < 3; ++round)
            for (int mode = 0; mode < 4; ++mode)
                t[mode].push_back(time_run([&, mode](unsigned l) { ma::enqueue_index_proj_diag({x, ws[l].packed, iq, ik, rows}, mode, st); }, L, 10));
        std::printf("diag rows %u us/layer: full %.1f | no weight loads %.1f | pure FMA chain %.1f | streams only (no FMA) %.1f\n", rows,
                    median(t[0]), median(t[1]), median(t[2]), median(t[3]));
    }
    // Prefill: A = engine dense_t16<16>, B = tiled.
    if (!slab_only) for (unsigned rows : {1024U, 8192U}) {
        const unsigned layers = rows > 2048 ? 6 : 20;
        auto arm = [&](int a) {
            return [&, a](unsigned l) {
                if (a == 0) engine_t16(x, ws[l], iq, ik, rows, st);
                else ma::enqueue_index_proj_tiled({x, ws[l].packed, iq, ik, rows}, st);
            };
        };
        std::vector<double> t[2];
        std::string runs;
        for (int round = 0; round < 3; ++round) {
            const double a1 = time_run(arm(0), layers, 1), b = time_run(arm(1), layers, 1), a2 = time_run(arm(0), layers, 1);
            t[0].push_back(a1);
            t[0].push_back(a2);
            t[1].push_back(b);
            char buf[64];
            std::snprintf(buf, sizeof buf, " [%.0f %.0f %.0f]", a1, b, a2);
            runs += buf;
        }
        std::printf("prefill T %u us/layer: A(engine dense_t16<16>) %.0f tiled %.0f (%.3f)  ABA:%s\n", rows, median(t[0]), median(t[1]),
                    median(t[1]) / median(t[0]), runs.c_str());
        std::fflush(stdout);
    }
    // Prefill slab gate: literal A B A x3, A = tiled, B = slab.
    for (unsigned rows : {4096U, 8192U, 1024U, 4096U}) {
        {
            const unsigned layers = rows > 2048 ? 12 : 30;
            auto arm = [&](int a) {
                return [&, a](unsigned l) {
                    if (a == 0) ma::enqueue_index_proj_tiled({x, ws[l].packed, iq, ik, rows}, st);
                    else ma::enqueue_index_proj_slab({x, ws[l].packed, iq, ik, rows}, st);
                };
            };
            std::vector<double> t[2], ratio;
            std::string runs;
            for (int round = 0; round < 3; ++round) {
                const double a1 = time_run(arm(0), layers, 1), b = time_run(arm(1), layers, 1), a2 = time_run(arm(0), layers, 1);
                t[0].push_back(a1);
                t[0].push_back(a2);
                t[1].push_back(b);
                ratio.push_back(b / (0.5 * (a1 + a2)));
                char buf[64];
                std::snprintf(buf, sizeof buf, " [%.0f %.0f %.0f]", a1, b, a2);
                runs += buf;
            }
            std::printf("prefill T %u us/layer: tiled %.0f slab %.0f (B/A median %.3f, worst %.3f)  ABA:%s\n", rows, median(t[0]),
                        median(t[1]), median(ratio), *std::max_element(ratio.begin(), ratio.end()), runs.c_str());
            std::fflush(stdout);
        }
    }
    (void)hipEventDestroy(e0);
    (void)hipEventDestroy(e1);
    (void)hipStreamDestroy(st);
    for (auto& w : ws) free_weights(w);
    (void)hipFree(x);
    (void)hipFree(iq);
    (void)hipFree(ik);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::mt19937 rng(20261024);
    tests(rng);
    wmma_tests(rng);
    if (g_fail != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("strix-mmattn-index-proj-tests: all passed\n");
    if (const char* b = std::getenv("STRIX_MMATTN_BENCH"); b != nullptr && (std::strcmp(b, "1") == 0 || std::strcmp(b, "slab") == 0))
        bench(rng, std::strcmp(b, "slab") == 0);
    else if (b != nullptr && std::strcmp(b, "wmma") == 0)
        bench_wmma(rng);
    return g_fail == 0 ? 0 : 1;
}
