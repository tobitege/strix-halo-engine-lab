// Byte-identity + timing of the fused GDN decode path (STRIX_QRTX_GDN_DECODE, default on) against the old kernel sequence.
// OLD = launch_gates, raw_qkv cudaMemcpy2D, launch_conv (prepass+conv), launch_recurrence_split (decay+recurrence), launch_norm;
//       commit: launch_recurrence_split + launch_conv_rollback.
// NEW = launch_core_split / launch_core (in-place state) / launch_commit_replay (fused, count <= 8).
// Build: build_qwen_test.bat <worktree> qwen_gdn_decode_exact_test <outdir> <worktree>\src\cuda\models\qwen\qwen_gdn.cu
// Run:   qwen_gdn_decode_exact_test.exe [tables dir]   (rows 1..8, three seeds, lanes 2/4/8/16; PASS/FAIL counts, timings)
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strix/cuda/amd_math_tables.hpp"
#include "strix/cuda/qwen_gdn.hpp"

namespace am = strix::cuda::amdmath;
namespace g = strix::cuda::qwen_gdn;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

static int g_pass = 0, g_fail = 0;
constexpr u32 H = g::kHidden, VH = g::kValueHeads, CW = g::kConvWidth, D = g::kHeadDim;
constexpr u64 kStride = 10240 + 6144 + 48 + 48 + 6144;  // driver gdn_stride
constexpr u32 kLayers = 36;
static am::Tables g_t;
static cudaStream_t g_s;

struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed * 0x9e3779b97f4a7c15ULL + 1) {}
    u64 next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    float uni() { return static_cast<float>((next() >> 40) * (1.0 / 16777216.0)); }
    float normal() {
        const float u1 = std::max(uni(), 1e-7F), u2 = uni();
        return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
    }
};
static u16 f2bf(float v) { u32 b; std::memcpy(&b, &v, 4); return static_cast<u16>((b + 0x7fffU + ((b >> 16) & 1U)) >> 16); }

template <class T> static T* up(const std::vector<T>& v) {
    T* d = nullptr; CK(cudaMalloc(&d, v.size() * sizeof(T))); CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d;
}
template <class T> static std::vector<T> down(const T* d, size_t n) {
    std::vector<T> v(n); CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
}

struct Weights {
    float *db, *da, *dc, *ddt, *dA, *dn;
    explicit Weights(u64 seed) {
        Rng r(seed);
        std::vector<float> beta(VH * H), alpha(VH * H), conv(CW * 4), dt(VH), a(VH), norm(D);
        for (auto& x : beta) x = r.normal() * 0.05F;
        for (u32 h = 0; h < VH; ++h)  // alpha heads at three scales: softplus branches x>20, x<-20, between
            for (u32 f = 0; f < H; ++f) alpha[h * H + f] = r.normal() * (h % 3 == 0 ? 2.0F : h % 3 == 1 ? 0.05F : 0.5F);
        for (auto& x : conv) x = r.normal() * 0.5F;
        for (u32 h = 0; h < VH; ++h) { dt[h] = r.normal() * 3.0F; a[h] = -std::exp(r.normal()); }
        for (auto& x : norm) x = 1.0F + r.normal() * 0.2F;
        db = up(beta); da = up(alpha); dc = up(conv); ddt = up(dt); dA = up(a); dn = up(norm);
    }
    ~Weights() { cudaFree(db); cudaFree(da); cudaFree(dc); cudaFree(ddt); cudaFree(dA); cudaFree(dn); }
    Weights(const Weights&) = delete;
    g::Weights w() const { return g::Weights{db, da, dc, ddt, dA, dn}; }
};

struct Case {  // host inputs of one window
    std::vector<u16> hidden, hist;
    std::vector<float> rows, state;
};
static Case make_case(u64 seed) {
    Rng r(seed); Case c;
    c.hidden.resize(size_t{8} * H);
    for (u32 row = 0; row < 8; ++row)
        for (u32 f = 0; f < H; ++f) {
            float x = r.normal();
            if (seed % 3 == 1 && row == 5) x = 0.0F; else if (seed % 3 == 1 && row == 6) x *= 3000.0F;
            c.hidden[size_t{row} * H + f] = f2bf(x);
        }
    c.hist.resize(size_t{CW} * 4);
    for (auto& x : c.hist) x = f2bf(r.normal() * 2.0F);
    c.rows.assign(8 * kStride, 0.0F);
    for (u32 row = 0; row < 8; ++row) {
        for (u32 i = 0; i < CW; ++i) {  // raw qkv (FP32 GEMV output): mixed scales, one near-zero row, one large row
            float x = r.normal() * 1.5F;
            if (seed % 3 == 2 && row == 2) x *= 1e-30F; else if (seed % 3 == 2 && row == 4) x *= 400.0F;
            c.rows[row * kStride + i] = x;
        }
        for (u32 i = 0; i < 6144; ++i) c.rows[row * kStride + 10240 + i] = r.normal();  // z
    }
    c.state.resize(g::kMatrixFloats);
    for (auto& x : c.state) x = r.normal() * 0.3F;
    return c;
}

struct Layer {
    float* rows = nullptr; float* raw = nullptr; u16* conv = nullptr; u16* before = nullptr; float* s_in = nullptr; float* s_out = nullptr;
    void* scratch = nullptr;
    void alloc() {
        CK(cudaMalloc(&rows, 8 * kStride * 4)); CK(cudaMalloc(&raw, 8 * CW * 4));
        CK(cudaMalloc(&conv, CW * 4 * 2)); CK(cudaMalloc(&before, CW * 4 * 2));
        CK(cudaMalloc(&s_in, g::kMatrixFloats * 4)); CK(cudaMalloc(&s_out, g::kMatrixFloats * 4));
        CK(cudaMalloc(&scratch, g::core_scratch_bytes(8) + 1024));
    }
    void release() { cudaFree(rows); cudaFree(raw); cudaFree(conv); cudaFree(before); cudaFree(s_in); cudaFree(s_out); cudaFree(scratch); }
    g::Rows R() const { return g::Rows{rows, rows + 10240, rows + 16384, rows + 16432, rows + 16480, kStride}; }
    void* dscratch(u32 count) const {
        return static_cast<unsigned char*>(scratch) + (g::conv_scratch_bytes(count) + 255U) / 256U * 256U;
    }
    void load(const Case& c) {  // window start: raw rows, history, S0; everything else poisoned
        CK(cudaMemcpy(rows, c.rows.data(), 8 * kStride * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(raw, 0xCD, 8 * CW * 4));
        CK(cudaMemcpy(conv, c.hist.data(), CW * 4 * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(before, c.hist.data(), CW * 4 * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(s_in, c.state.data(), g::kMatrixFloats * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(s_out, 0xCD, g::kMatrixFloats * 4));
        CK(cudaMemset(scratch, 0xCD, g::core_scratch_bytes(8)));
        CK(cudaDeviceSynchronize());
    }
};

static void old_gates(Layer& L, const u16* hidden, const Weights& W, u32 n) { g::launch_gates(hidden, H, L.R(), n, W.w(), g_t, g_s); }
static void old_copy(Layer& L, u32 n) {
    CK(cudaMemcpy2DAsync(L.raw, size_t{CW} * 4, L.rows, kStride * 4, size_t{CW} * 4, n, cudaMemcpyDeviceToDevice, g_s));
}
static void old_conv(Layer& L, const Weights& W, u32 n) { g::launch_conv(L.R(), n, L.conv, L.scratch, W.w(), g_t, g_s); }
static void old_rec(Layer& L, u32 n) { g::launch_recurrence_split(L.R(), n, L.s_in, L.s_out, L.dscratch(n), g_t, g_s); }
static void old_norm(Layer& L, const Weights& W, u32 n) { g::launch_norm(L.R(), n, W.w(), g_t, g_s); }
static void old_all(Layer& L, const u16* hidden, const Weights& W, u32 n) {
    old_gates(L, hidden, W, n); old_copy(L, n); old_conv(L, W, n); old_rec(L, n); old_norm(L, W, n);
}
static void new_all(Layer& L, const u16* hidden, const Weights& W, u32 n, u32 lanes) {
    g::launch_core_split(hidden, H, L.R(), n, L.conv, L.s_in, L.s_out, L.raw, L.scratch, W.w(), g_t, g_s, nullptr, g::Tuning{lanes});
}
static void old_commit(Layer& L, u32 keep) {
    g::launch_recurrence_split(L.R(), keep, L.s_in, L.s_out, L.dscratch(keep), g_t, g_s);
    g::launch_conv_rollback(L.raw, keep, L.before, L.conv, g_s);
}
static void new_commit(Layer& L, u32 keep, u32 lanes) {
    g::launch_commit_replay(L.R(), keep, L.s_in, L.s_out, L.raw, L.before, L.conv, L.dscratch(keep), g_t, g_s, g::Tuning{lanes});
}

static bool same_bytes(const char* what, u32 n, u32 lanes, u64 seed, const void* a, const void* b, size_t bytes) {
    const bool ok = std::memcmp(a, b, bytes) == 0;
    if (ok) ++g_pass; else { ++g_fail; std::printf("FAIL %s rows=%u lanes=%u seed=%llu\n", what, n, lanes, static_cast<unsigned long long>(seed)); }
    return ok;
}
struct Snap { std::vector<float> rows, raw, state; std::vector<u16> conv; };
static Snap snap(const Layer& L, const float* state) {
    CK(cudaStreamSynchronize(g_s));
    return Snap{down(L.rows, 8 * kStride), down(L.raw, 8 * CW), down(state, g::kMatrixFloats), down(L.conv, CW * 4)};
}
static void compare(const char* tag, const Snap& a, const Snap& b, u32 n, u32 lanes, u64 seed, bool with_raw) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%s rows(qkv,z,beta,gate,core)", tag); same_bytes(buf, n, lanes, seed, a.rows.data(), b.rows.data(), a.rows.size() * 4);
    if (with_raw) { std::snprintf(buf, sizeof buf, "%s raw_qkv", tag); same_bytes(buf, n, lanes, seed, a.raw.data(), b.raw.data(), a.raw.size() * 4); }
    std::snprintf(buf, sizeof buf, "%s state", tag); same_bytes(buf, n, lanes, seed, a.state.data(), b.state.data(), a.state.size() * 4);
    std::snprintf(buf, sizeof buf, "%s conv", tag); same_bytes(buf, n, lanes, seed, a.conv.data(), b.conv.data(), a.conv.size() * 2);
}

static void exactness() {
    Weights W(7);
    Layer A, B, C; A.alloc(); B.alloc(); C.alloc();
    u16* dh = nullptr;
    for (u64 seed = 1; seed <= 3; ++seed) {
        const Case c = make_case(seed);
        dh = up(c.hidden);
        for (u32 n = 1; n <= 8; ++n)
            for (u32 lanes : {2U, 4U, 8U, 16U}) {
                A.load(c); B.load(c);
                old_all(A, dh, W, n);
                new_all(B, dh, W, n, lanes);
                const Snap sa = snap(A, A.s_out), sb = snap(B, B.s_out);
                compare("verify", sa, sb, n, lanes, seed, true);
                // in-place state (launch_core, no raw copy): final state and rows must equal the old split result
                C.load(c);
                g::launch_core(dh, H, C.R(), n, C.conv, C.s_in, C.scratch, W.w(), g_t, g_s, nullptr, g::Tuning{lanes});
                const Snap sc = snap(C, C.s_in);
                same_bytes("inplace state", n, lanes, seed, sa.state.data(), sc.state.data(), sa.state.size() * 4);
                same_bytes("inplace rows", n, lanes, seed, sa.rows.data(), sc.rows.data(), sa.rows.size() * 4);
                same_bytes("inplace conv", n, lanes, seed, sa.conv.data(), sc.conv.data(), sa.conv.size() * 2);
                // commit replay of keep = 1..n rows from the verified window (rows after K4/K5, raw rows, history before the window)
                for (u32 keep = 1; keep <= n; ++keep) {
                    for (Layer* L : {&A, &B}) {
                        CK(cudaMemcpy(L->rows, sa.rows.data(), 8 * kStride * 4, cudaMemcpyHostToDevice));
                        CK(cudaMemcpy(L->raw, sa.raw.data(), 8 * CW * 4, cudaMemcpyHostToDevice));
                        CK(cudaMemcpy(L->conv, sa.conv.data(), CW * 4 * 2, cudaMemcpyHostToDevice));
                        CK(cudaMemset(L->s_out, 0xCD, g::kMatrixFloats * 4));
                    }
                    old_commit(A, keep);
                    new_commit(B, keep, lanes);
                    const Snap ca = snap(A, A.s_out), cb = snap(B, B.s_out);
                    compare("commit", ca, cb, keep, lanes, seed, true);
                }
            }
        cudaFree(dh);
    }
    A.release(); B.release(); C.release();
}

// per-call time of `f(layer)` over kLayers distinct layers captured into one graph (warm, 50 reps)
template <class F> static double time_graph(F&& f) {
    cudaGraph_t gr; cudaGraphExec_t ex;
    CK(cudaStreamBeginCapture(g_s, cudaStreamCaptureModeThreadLocal));
    for (u32 l = 0; l < kLayers; ++l) f(l);
    CK(cudaStreamEndCapture(g_s, &gr));
    CK(cudaGraphInstantiate(&ex, gr, 0));
    for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(ex, g_s));
    CK(cudaStreamSynchronize(g_s));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, g_s));
    for (int i = 0; i < 50; ++i) CK(cudaGraphLaunch(ex, g_s));
    CK(cudaEventRecord(e1, g_s)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    cudaGraphExecDestroy(ex); cudaGraphDestroy(gr); cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms * 1000.0 / 50.0 / kLayers;
}
template <class F> static double time_eager(F&& f) {
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (u32 l = 0; l < kLayers; ++l) f(l);
    CK(cudaStreamSynchronize(g_s));
    CK(cudaEventRecord(e0, g_s));
    for (int i = 0; i < 50; ++i) for (u32 l = 0; l < kLayers; ++l) f(l);
    CK(cudaEventRecord(e1, g_s)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms * 1000.0 / 50.0 / kLayers;
}

static void timing() {
    std::vector<Weights*> W;
    std::vector<Layer> L(kLayers);
    const Case c = make_case(11);
    u16* dh = up(c.hidden);
    for (u32 l = 0; l < kLayers; ++l) { W.push_back(new Weights(100 + l)); L[l].alloc(); L[l].load(c); }
    int clock = 0; CK(cudaDeviceGetAttribute(&clock, cudaDevAttrClockRate, 0));
    std::printf("timing: us per call per layer, %u distinct layers, graph-captured (in-graph), max SM clock %d MHz\n", kLayers, clock / 1000);
    for (u32 n : {1U, 4U, 8U}) {
        std::printf("rows %u OLD: gates %.2f copy %.2f conv(prepass+conv) %.2f rec(decay+rec) %.2f norm %.2f | sequence %.2f\n", n,
            time_graph([&](u32 l) { old_gates(L[l], dh, *W[l], n); }), time_graph([&](u32 l) { old_copy(L[l], n); }),
            time_graph([&](u32 l) { old_conv(L[l], *W[l], n); }), time_graph([&](u32 l) { old_rec(L[l], n); }),
            time_graph([&](u32 l) { old_norm(L[l], *W[l], n); }), time_graph([&](u32 l) { old_all(L[l], dh, *W[l], n); }));
        for (u32 lanes : {4U, 8U})
            std::printf("rows %u NEW lanes %u: sequence (front+rec+norm) %.2f\n", n, lanes,
                time_graph([&](u32 l) { new_all(L[l], dh, *W[l], n, lanes); }));
        std::printf("rows %u commit OLD graph %.2f eager %.2f | NEW lanes4 graph %.2f eager %.2f | NEW lanes8 graph %.2f eager %.2f\n", n,
            time_graph([&](u32 l) { old_commit(L[l], n); }), time_eager([&](u32 l) { old_commit(L[l], n); }),
            time_graph([&](u32 l) { new_commit(L[l], n, 4); }), time_eager([&](u32 l) { new_commit(L[l], n, 4); }),
            time_graph([&](u32 l) { new_commit(L[l], n, 8); }), time_eager([&](u32 l) { new_commit(L[l], n, 8); }));
    }
    for (u32 l = 0; l < kLayers; ++l) { L[l].release(); delete W[l]; }
    cudaFree(dh);
}

int main(int argc, char** argv) {
    const std::string tables = argc > 1 ? argv[1] : "D:/tmp/qrtxpf-build/tables";
    CK(cudaStreamCreate(&g_s));
    g_t = am::load_tables(tables);
    exactness();
    std::printf("qwen_gdn_decode_exact_test: %d passed, %d failed\n", g_pass, g_fail);
    if (!g_fail) timing();
    am::free_tables(g_t);
    return g_fail ? 1 : 0;
}
