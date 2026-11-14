// Tests + microbench of the one-launch index projections (src/minimax/attn/index_proj.hpp): every output bit for bit
// equal to the gate's dense_t (FP32 transposed weights) and to the engine's dense_t16 (both copied below verbatim from
// src/minimax/engine/mm_kernels.hpp at minimax/engine f2f84f43), decode rows 1..8 x ring 8/16/32 and the tiled
// prefill kernel at ragged token counts. STRIX_MMATTN_BENCH=1 adds ABA x3 timings with the weights of 60 layers
// rotated (cold, like the real layer loop).
#include "index_proj.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

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
    const unsigned maxT = 1100;
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
    for (unsigned rows : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 16U, 17U, 63U, 64U, 65U, 300U, 1100U}) {
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
    if (g_fail != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("strix-mmattn-index-proj-tests: all passed\n");
    if (const char* b = std::getenv("STRIX_MMATTN_BENCH"); b != nullptr && (std::strcmp(b, "1") == 0 || std::strcmp(b, "slab") == 0))
        bench(rng, std::strcmp(b, "slab") == 0);
    return 0;
}
