// Tests of the CUDA GDN prefill core (src/cuda/models/qwen/qwen_gdn.cu) against literal host references of the HIP kernels
// (src/hip_gdn_launcher.hip :2750 gates, :2807 conv, :2896 lanes, :3000 norm) and, when present, the HIP step dump.
//   qwen_gdn_tests [--tables DIR] [--no-real] [--replay-only] [--bench ROWS] [--fixture DIR]
// Host references follow the HIP lane structure literally (256-lane chains + LDS/shuffle trees, permlanex16/DPP key sums,
// the G=4 lanes recurrence with DPP row_xmask). Transcendentals come from a device oracle running the same
// amd_math.cuh emulation (verified separately by qwen_amd_math_tests), so these tests check the arithmetic structure.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "strix/cuda/qwen_gdn.hpp"
#include "strix/cuda/amd_math_tables.hpp"
#include "qwen_fixture.hpp"
#include "qwen_gguf_host.hpp"

namespace am = strix::cuda::amdmath;
namespace g = strix::cuda::qwen_gdn;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

static int g_pass = 0, g_fail = 0;
static am::Tables g_tables;

// ------------------------------------------------------------------------------------------------ host helpers
static float bits_f(u32 b) { float f; std::memcpy(&f, &b, 4); return f; }
static u32 f_bits(float f) { u32 b; std::memcpy(&b, &f, 4); return b; }
static u16 h_f2bf(float v) { const u32 b = f_bits(v); return static_cast<u16>((b + 0x7fffU + ((b >> 16) & 1U)) >> 16); }
static float h_bf2f(u16 b) { return bits_f(static_cast<u32>(b) << 16); }
static float h_rbf(float v) { return h_bf2f(h_f2bf(v)); }

struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed * 0x9e3779b97f4a7c15ULL + 1) {}
    u64 next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    float uni() { return static_cast<float>((next() >> 40) * (1.0 / 16777216.0)); }  // [0,1)
    float normal() {
        const float u1 = std::max(uni(), 1e-7F), u2 = uni();
        return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
    }
};

template <class T> static T* dev_upload(const std::vector<T>& v) {
    T* d = nullptr; CK(cudaMalloc(&d, std::max<size_t>(v.size(), 1) * sizeof(T)));
    if (!v.empty()) CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> static std::vector<T> dev_download(const T* d, size_t n) {
    std::vector<T> v(n); if (n) CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
}

// bitwise comparison of two arrays of 32/16-bit values; reports the first difference
template <class T> static bool same(const char* what, const std::vector<T>& got, const std::vector<T>& want) {
    size_t diff = 0, first = SIZE_MAX;
    if (got.size() != want.size()) { std::printf("FAIL %s: size %zu vs %zu\n", what, got.size(), want.size()); ++g_fail; return false; }
    for (size_t i = 0; i < got.size(); ++i)
        if (std::memcmp(&got[i], &want[i], sizeof(T))) { if (first == SIZE_MAX) first = i; ++diff; }
    if (diff) {
        if constexpr (sizeof(T) == 4) {
            u32 a, b; std::memcpy(&a, &got[first], 4); std::memcpy(&b, &want[first], 4);
            std::printf("FAIL %s: %zu/%zu differ, first [%zu] cuda %08x (%.9g) ref %08x (%.9g)\n", what, diff, got.size(), first, a,
                bits_f(a), b, bits_f(b));
        } else {
            u64 a = 0, b = 0; std::memcpy(&a, &got[first], sizeof(T)); std::memcpy(&b, &want[first], sizeof(T));
            std::printf("FAIL %s: %zu/%zu differ, first [%zu] cuda %llx ref %llx\n", what, diff, got.size(), first,
                static_cast<unsigned long long>(a), static_cast<unsigned long long>(b));
        }
        ++g_fail; return false;
    }
    std::printf("ok   %s (%zu values)\n", what, got.size());
    ++g_pass; return true;
}

// ------------------------------------------------------------------------------------------------ device oracle
__global__ void oracle_kernel(int op, const float* x, float* y, size_t n, am::Tables t) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    y[i] = op == 0 ? am::amd_expf(t, x[i]) : op == 1 ? am::amd_log1pf(t, x[i]) : am::amd_rsqrtf(t, x[i]);
}
enum { kExp = 0, kLog1p = 1, kRsqrt = 2 };
static std::vector<float> oracle(int op, const std::vector<float>& x) {
    if (x.empty()) return {};
    float* dx = dev_upload(x); float* dy = nullptr; CK(cudaMalloc(&dy, x.size() * 4));
    oracle_kernel<<<static_cast<unsigned>((x.size() + 255) / 256), 256>>>(op, dx, dy, x.size(), g_tables);
    CK(cudaGetLastError());
    auto y = dev_download(dy, x.size());
    CK(cudaFree(dx)); CK(cudaFree(dy));
    return y;
}

// ------------------------------------------------------------------------------------------------ host references
constexpr u32 H = g::kHidden, CW = g::kConvWidth, VW = g::kValueWidth, VH = g::kValueHeads, KH = g::kKeyHeads, D = g::kHeadDim;

// gdn_window_key_sum, literal on 32 lanes: (p0+p2)+(p1+p3); permlanex16 (lane l reads l^16); DPP row_shl d (lane l reads
// l+d when inside its 16-lane row, else keeps its own value, bound_ctrl off); readfirstlane = lane 0.
static float ref_key_sum(const float* v128 /* channel c = part*32+lane */, bool squares) {
    float s[32], tmp[32];
    for (int l = 0; l < 32; ++l) {
        float p[4];
        for (int part = 0; part < 4; ++part) { const float x = v128[part * 32 + l]; p[part] = squares ? x * x : x; }
        s[l] = (p[0] + p[2]) + (p[1] + p[3]);
    }
    for (int l = 0; l < 32; ++l) tmp[l] = s[l ^ 16];
    for (int l = 0; l < 32; ++l) s[l] = s[l] + tmp[l];
    for (int d = 8; d; d >>= 1) {
        for (int l = 0; l < 32; ++l) tmp[l] = (l % 16) + d < 16 ? s[l + d] : s[l];
        for (int l = 0; l < 32; ++l) s[l] = s[l] + tmp[l];
    }
    return s[0];
}

// gdn_prefill_gates_kernel: hidden BF16 [count][2560] -> beta, gate [count][48]
static void ref_gates(const std::vector<u16>& hidden, u32 count, const std::vector<float>& wb, const std::vector<float>& wa,
    const std::vector<float>& dt, const std::vector<float>& a, std::vector<float>& beta, std::vector<float>& gate) {
    std::vector<float> rawb(count * VH), rawa(count * VH);
    for (u32 row = 0; row < count; ++row)
        for (u32 kind = 0; kind < 2; ++kind)
            for (u32 out = 0; out < VH; ++out) {
                const float* w = (kind ? wa : wb).data() + static_cast<u64>(out) * H;
                const u16* x = hidden.data() + static_cast<u64>(row) * H;
                float s[256], t[32], tmp[32];
                for (int l = 0; l < 256; ++l) {
                    float acc = 0.0F;
                    for (u32 f = l; f < H; f += 256) acc = std::fmaf(h_bf2f(x[f]), w[f], acc);
                    s[l] = acc;
                }
                for (int l = 0; l < 128; ++l) s[l] = s[l] + s[l + 128];
                for (int l = 0; l < 64; ++l) s[l] = s[l] + s[l + 64];
                for (int l = 0; l < 32; ++l) t[l] = s[l] + s[l + 32];
                for (int d = 16; d; d >>= 1) {  // HIP __shfl_down(width 32): lane l reads l+d if < 32, else itself
                    for (int l = 0; l < 32; ++l) tmp[l] = l + d < 32 ? t[l + d] : t[l];
                    for (int l = 0; l < 32; ++l) t[l] = t[l] + tmp[l];
                }
                (kind ? rawa : rawb)[row * VH + out] = h_rbf(t[0]);
            }
    std::vector<float> negb(rawb.size()), biased(rawa.size());
    for (size_t i = 0; i < rawb.size(); ++i) { negb[i] = -rawb[i]; biased[i] = rawa[i] + dt[i % VH]; }
    const auto eb = oracle(kExp, negb), ea = oracle(kExp, biased), la = oracle(kLog1p, ea);
    beta.resize(rawb.size()); gate.resize(rawa.size());
    for (size_t i = 0; i < rawb.size(); ++i) {
        beta[i] = h_rbf(1.0F / (1.0F + eb[i]));
        const float x = biased[i];
        const float sp = x > 20.0F ? x : x < -20.0F ? ea[i] : la[i];
        gate[i] = a[i % VH] * sp;
    }
}

// Serial conv (gdn_window_prepare_rows structure: shift window, append, fma chain oldest first) with the prefill
// kernel's arithmetic. qkv [count][10240] rewritten; history [10240][4] updated.
static void ref_conv(std::vector<float>& qkv, u32 count, std::vector<u16>& history, const std::vector<float>& cw, float eps) {
    std::vector<float> sums(static_cast<size_t>(count) * CW), neg(sums.size());
    for (u32 ch = 0; ch < CW; ++ch) {
        u16 win[4];
        for (int i = 0; i < 4; ++i) win[i] = history[ch * 4 + i];
        for (u32 row = 0; row < count; ++row) {
            for (int i = 0; i < 3; ++i) win[i] = win[i + 1];
            win[3] = h_f2bf(qkv[static_cast<u64>(row) * CW + ch]);
            float sum = 0.0F;
            for (int i = 0; i < 4; ++i) sum = std::fmaf(h_bf2f(win[i]), cw[ch * 4 + i], sum);
            sums[static_cast<u64>(row) * CW + ch] = sum;
        }
        for (int i = 0; i < 4; ++i) history[ch * 4 + i] = win[i];
    }
    for (size_t i = 0; i < sums.size(); ++i) neg[i] = -sums[i];
    const auto e = oracle(kExp, neg);
    std::vector<float> value(sums.size());
    for (size_t i = 0; i < sums.size(); ++i) value[i] = h_rbf(sums[i] / (1.0F + e[i]));
    std::vector<float> ks(static_cast<size_t>(count) * 2 * KH);
    for (u32 row = 0; row < count; ++row)
        for (u32 head = 0; head < 2 * KH; ++head)
            ks[row * 2 * KH + head] = ref_key_sum(&value[static_cast<u64>(row) * CW + head * D], true) + eps;
    const auto inv = oracle(kRsqrt, ks);
    const float q_scale = oracle(kRsqrt, {static_cast<float>(D)})[0];
    for (u32 row = 0; row < count; ++row)
        for (u32 ch = 0; ch < CW; ++ch) {
            float v = value[static_cast<u64>(row) * CW + ch];
            const u32 head = ch / D;
            if (head < 2 * KH) v = (v * inv[row * 2 * KH + head]) * (head >= KH ? 1.0F : q_scale);
            qkv[static_cast<u64>(row) * CW + ch] = v;
        }
}

// gdn_prefill_lanes_kernel<key_value, G=4>, literal: lane sub owns keys sub+4j; in-lane tree halves 16..1; DPP row_xmask 2
// then 1 (lane sub reads sub^m; all four lanes end with the same root, checked).
static float ref_lane_tree4(const float* p128) {
    float lanes[4];
    for (int sub = 0; sub < 4; ++sub) {
        float r[32];
        for (int j = 0; j < 32; ++j) r[j] = p128[sub + 4 * j];
        for (int half = 16; half; half /= 2)
            for (int j = 0; j < half; ++j) r[j] = r[j] + r[j + half];
        lanes[sub] = r[0];
    }
    for (int m = 2; m; m /= 2) {
        float n[4];
        for (int sub = 0; sub < 4; ++sub) n[sub] = lanes[sub] + lanes[sub ^ m];
        std::memcpy(lanes, n, sizeof(n));
    }
    if (f_bits(lanes[0]) != f_bits(lanes[1]) || f_bits(lanes[0]) != f_bits(lanes[2]) || f_bits(lanes[0]) != f_bits(lanes[3])) {
        if (lanes[0] == lanes[0]) { std::printf("ref lane tree: lanes disagree\n"); std::exit(3); }
    }
    return lanes[0];
}
// qkv (after conv) [count][10240], beta/gate [count][48], state [48][128 k][128 v] -> core [count][6144]
static void ref_recurrence(const std::vector<float>& qkv, const std::vector<float>& beta, const std::vector<float>& gate, u32 count,
    std::vector<float>& state, std::vector<float>& core) {
    const auto decay = oracle(kExp, gate);
    core.assign(static_cast<size_t>(count) * VW, 0.0F);
    std::vector<float> S(D), p(D);
    for (u32 vh = 0; vh < VH; ++vh)
        for (u32 col = 0; col < D; ++col) {
            for (u32 k = 0; k < D; ++k) S[k] = state[(static_cast<u64>(vh) * D + k) * D + col];
            const u32 kh = vh % KH;
            for (u32 row = 0; row < count; ++row) {
                const float* r = qkv.data() + static_cast<u64>(row) * CW;
                const float* key = r + 2 * KH * 0 + KH * D + kh * D;
                const float* query = r + kh * D;
                const float v = r[2 * KH * D + vh * D + col];
                const float dcy = decay[row * VH + vh], b = beta[row * VH + vh];
                for (u32 k = 0; k < D; ++k) { S[k] = S[k] * dcy; p[k] = S[k] * key[k]; }
                const float pred = ref_lane_tree4(p.data());
                const float delta = (v - pred) * b;
                for (u32 k = 0; k < D; ++k) { S[k] = std::fmaf(key[k], delta, S[k]); p[k] = S[k] * query[k]; }
                core[static_cast<u64>(row) * VW + vh * D + col] = ref_lane_tree4(p.data());
            }
            for (u32 k = 0; k < D; ++k) state[(static_cast<u64>(vh) * D + k) * D + col] = S[k];
        }
}

// dense_a8_quantize8 semantics on one 32-feature block (hip_q8_rows_gemv.hpp:1107-1126)
static void ref_quantize_block(const float* x, std::int8_t* codes, float& delta) {
    bool bad = false; float amax = 0.0F;
    for (int l = 0; l < 32; ++l) { bad |= !std::isfinite(x[l]); amax = std::fmax(amax, std::fabs(x[l])); }
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    delta = bad ? bits_f(0x7fc00000U) : (amax == 0.0F || tiny) ? 1.0F : amax / 127.0F;
    for (int l = 0; l < 32; ++l) {
        const float rounded = bad || tiny ? 0.0F : std::nearbyint(x[l] / delta);
        codes[l] = static_cast<std::int8_t>(static_cast<int>(std::fmin(127.0F, std::fmax(-127.0F, rounded))));
    }
}
// gdn_prefill_norm_rows_kernel: core [count][6144] in place, z [count][6144]
static void ref_norm(std::vector<float>& core, const std::vector<float>& z, u32 count, const std::vector<float>& nw, float eps,
    std::vector<std::int8_t>* codes, std::vector<float>* deltas) {
    std::vector<float> q(static_cast<size_t>(count) * VH), nz(z.size());
    for (u32 row = 0; row < count; ++row)
        for (u32 h = 0; h < VH; ++h) q[row * VH + h] = ref_key_sum(&core[static_cast<u64>(row) * VW + h * D], true) / 128.0F + eps;
    const auto inv = oracle(kRsqrt, q);
    for (size_t i = 0; i < z.size(); ++i) nz[i] = -z[i];
    const auto e = oracle(kExp, nz);
    for (u32 row = 0; row < count; ++row)
        for (u32 c = 0; c < VW; ++c) {
            const u64 i = static_cast<u64>(row) * VW + c;
            const float gate = 1.0F / (1.0F + e[i]);
            core[i] = ((core[i] * inv[row * VH + c / D]) * nw[c % D]) * gate;
        }
    if (codes) {
        codes->assign(static_cast<size_t>(count) * VW, 0);
        deltas->assign(static_cast<size_t>((count + 255) / 256) * (VW / 32) * 256, 0.0F);
        for (u32 row = 0; row < count; ++row)
            for (u32 b = 0; b < VW / 32; ++b) {
                float d;
                ref_quantize_block(&core[static_cast<u64>(row) * VW + b * 32], codes->data() + static_cast<u64>(row) * VW + b * 32, d);
                (*deltas)[static_cast<u64>(row / 256) * (VW / 32 * 256) + b * 256 + row % 256] = d;
            }
    }
}

// ------------------------------------------------------------------------------------------------ device workspace
constexpr u64 kStride = CW + VW + VH + VH + VW;  // 22624 floats, %4 == 0
constexpr u64 kOffZ = CW, kOffBeta = CW + VW, kOffGate = kOffBeta + VH, kOffCore = kOffGate + VH;
struct Workspace {
    float* base = nullptr; u32 count = 0;
    explicit Workspace(u32 n) : count(n) { CK(cudaMalloc(&base, std::max<u64>(n, 1) * kStride * 4)); CK(cudaMemset(base, 0, std::max<u64>(n, 1) * kStride * 4)); }
    ~Workspace() { cudaFree(base); }
    g::Rows rows(u32 first = 0) const {
        float* b = base + static_cast<u64>(first) * kStride;
        return {b, b + kOffZ, b + kOffBeta, b + kOffGate, b + kOffCore, kStride};
    }
    void put(u64 off, u32 width, const std::vector<float>& v) const {
        CK(cudaMemcpy2D(base + off, kStride * 4, v.data(), width * 4ULL, width * 4ULL, count, cudaMemcpyHostToDevice));
    }
    std::vector<float> get(u64 off, u32 width) const {
        std::vector<float> v(static_cast<size_t>(count) * width);
        CK(cudaMemcpy2D(v.data(), width * 4ULL, base + off, kStride * 4, width * 4ULL, count, cudaMemcpyDeviceToHost));
        return v;
    }
};

struct HostWeights { std::vector<float> beta, alpha, conv, dt, a, norm; };
struct DevWeights {
    float *beta, *alpha, *conv, *dt, *a, *norm;
    explicit DevWeights(const HostWeights& h) : beta(dev_upload(h.beta)), alpha(dev_upload(h.alpha)), conv(dev_upload(h.conv)),
        dt(dev_upload(h.dt)), a(dev_upload(h.a)), norm(dev_upload(h.norm)) {}
    ~DevWeights() { cudaFree(beta); cudaFree(alpha); cudaFree(conv); cudaFree(dt); cudaFree(a); cudaFree(norm); }
    g::Weights w() const { g::Weights x{beta, alpha, conv, dt, a, norm}; return x; }
};

static std::vector<float> f32_of(const std::vector<std::uint8_t>& b) {
    std::vector<float> v(b.size() / 4); std::memcpy(v.data(), b.data(), v.size() * 4); return v;
}
static HostWeights synthetic_weights(u64 seed) {
    Rng r(seed); HostWeights w;
    w.beta.resize(VH * H); w.alpha.resize(VH * H); w.conv.resize(CW * 4); w.dt.resize(VH); w.a.resize(VH); w.norm.resize(D);
    for (auto& x : w.beta) x = r.normal() * 0.05F;
    for (u32 h = 0; h < VH; ++h)  // alpha heads with very different scales: the softplus branches x>20, x<-20 and between
        for (u32 f = 0; f < H; ++f) w.alpha[h * H + f] = r.normal() * (h % 3 == 0 ? 2.0F : h % 3 == 1 ? 0.05F : 0.5F);
    for (auto& x : w.conv) x = r.normal() * 0.5F;
    for (u32 h = 0; h < VH; ++h) { w.dt[h] = r.normal() * 3.0F; w.a[h] = -std::exp(r.normal()); }
    for (auto& x : w.norm) x = 1.0F + r.normal() * 0.2F;
    return w;
}
static HostWeights real_weights(const strix::qwen_gguf::Model& m, u32 layer) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    HostWeights w;
    w.beta = f32_of(m.bytes(p + "ssm_beta.weight")); w.alpha = f32_of(m.bytes(p + "ssm_alpha.weight"));
    w.conv = f32_of(m.bytes(p + "ssm_conv1d.weight")); w.dt = f32_of(m.bytes(p + "ssm_dt.bias"));
    w.a = f32_of(m.bytes(p + "ssm_a")); w.norm = f32_of(m.bytes(p + "ssm_norm.weight"));
    if (w.beta.size() != VH * H || w.alpha.size() != VH * H || w.conv.size() != CW * 4 || w.dt.size() != VH || w.a.size() != VH ||
        w.norm.size() != D) { std::printf("real weights: unexpected sizes\n"); std::exit(2); }
    return w;
}

// synthetic inputs: hidden BF16 rows with edge rows; qkv/z BF16-valued
static std::vector<u16> make_hidden(u32 count, u64 seed) {
    Rng r(seed); std::vector<u16> h(static_cast<size_t>(count) * H);
    for (u32 row = 0; row < count; ++row)
        for (u32 f = 0; f < H; ++f) {
            float x = r.normal();
            if (row % 17 == 5) x = 0.0F;                 // zero row
            else if (row % 17 == 9) x *= 1e-38F;         // tiny (subnormal BF16)
            else if (row % 17 == 13) x *= 3000.0F;       // huge
            h[static_cast<u64>(row) * H + f] = h_f2bf(x);
        }
    return h;
}
static std::vector<float> make_bf16_values(size_t n, u64 seed, float scale, u32 width) {
    Rng r(seed); std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        const u64 row = i / width;
        float x = r.normal() * scale;
        if (row % 13 == 3) x = 0.0F;
        else if (row % 13 == 7) x *= 1e-36F;
        else if (row % 13 == 11) x *= 200.0F;
        v[i] = h_rbf(x);
    }
    return v;
}
static std::vector<u16> make_history(u64 seed) {
    Rng r(seed); std::vector<u16> h(CW * 4);
    for (auto& x : h) x = h_f2bf(r.normal() * 2.0F);
    return h;
}

// ------------------------------------------------------------------------------------------------ tests
static void test_gates(const HostWeights& hw, const char* tag, u32 count) {
    const auto hidden = make_hidden(count, 11 + count);
    std::vector<float> beta, gate;
    ref_gates(hidden, count, hw.beta, hw.alpha, hw.dt, hw.a, beta, gate);
    DevWeights dw(hw); Workspace ws(count);
    u16* dh = dev_upload(hidden);
    g::launch_gates(dh, H, ws.rows(), count, dw.w(), g_tables, nullptr);
    CK(cudaDeviceSynchronize());
    std::string a = std::string("gates beta ") + tag, b = std::string("gates gate ") + tag;
    same(a.c_str(), ws.get(kOffBeta, VH), beta);
    same(b.c_str(), ws.get(kOffGate, VH), gate);
    CK(cudaFree(dh));
}

static void test_conv(const HostWeights& hw, const char* tag, u32 count, bool zero_history) {
    auto qkv = make_bf16_values(static_cast<size_t>(count) * CW, 21 + count, 2.0F, CW);
    auto hist = zero_history ? std::vector<u16>(CW * 4, 0) : make_history(31 + count);
    DevWeights dw(hw); Workspace ws(count);
    ws.put(0, CW, qkv);
    u16* dhist = dev_upload(hist); void* scratch = nullptr; CK(cudaMalloc(&scratch, g::conv_scratch_bytes(count)));
    g::launch_conv(ws.rows(), count, dhist, scratch, dw.w(), g_tables, nullptr);
    CK(cudaDeviceSynchronize());
    ref_conv(qkv, count, hist, hw.conv, 1e-6F);
    std::string a = std::string("conv qkv ") + tag + " T=" + std::to_string(count), b = std::string("conv history ") + tag + " T=" + std::to_string(count);
    same(a.c_str(), ws.get(0, CW), qkv);
    same(b.c_str(), dev_download(dhist, CW * 4), hist);
    CK(cudaFree(dhist)); CK(cudaFree(scratch));
}

// inputs of the recurrence: q/k from the conv of random data (unit-norm rows), v BF16, beta in (0,1), gate <= 0 with edges
struct RecInputs { std::vector<float> qkv, beta, gate, state; };
static RecInputs make_rec_inputs(const HostWeights& hw, u32 count, u64 seed, bool zero_state) {
    RecInputs in;
    in.qkv = make_bf16_values(static_cast<size_t>(count) * CW, seed, 2.0F, CW);
    auto hist = make_history(seed + 1);
    ref_conv(in.qkv, count, hist, hw.conv, 1e-6F);
    Rng r(seed + 2);
    in.beta.resize(static_cast<size_t>(count) * VH); in.gate.resize(in.beta.size());
    for (size_t i = 0; i < in.beta.size(); ++i) {
        in.beta[i] = h_rbf(r.uni());
        const u64 row = i / VH;
        in.gate[i] = row % 11 == 4 ? 0.0F : row % 11 == 8 ? -150.0F : -std::exp(r.normal() * 1.5F) * 0.3F;
    }
    in.state.assign(g::kMatrixFloats, 0.0F);
    if (!zero_state) for (auto& x : in.state) x = r.normal() * 0.1F;
    return in;
}
static void test_recurrence(const HostWeights& hw, u32 count, bool zero_state) {
    const auto in = make_rec_inputs(hw, count, 41 + count, zero_state);
    auto state = in.state; std::vector<float> core;
    ref_recurrence(in.qkv, in.beta, in.gate, count, state, core);
    for (u32 lanes : {2U, 4U, 8U, 16U}) {
        Workspace ws(count);
        ws.put(0, CW, in.qkv); ws.put(kOffBeta, VH, in.beta); ws.put(kOffGate, VH, in.gate);
        float* ds = dev_upload(in.state); void* scratch = nullptr; CK(cudaMalloc(&scratch, g::decay_scratch_bytes(count)));
        g::launch_recurrence(ws.rows(), count, ds, scratch, g_tables, nullptr, {lanes});
        CK(cudaDeviceSynchronize());
        const std::string tag = " G=" + std::to_string(lanes) + " T=" + std::to_string(count) + (zero_state ? " zero state" : " state");
        same(("recurrence core" + tag).c_str(), ws.get(kOffCore, VW), core);
        same(("recurrence state" + tag).c_str(), dev_download(ds, g::kMatrixFloats), state);
        CK(cudaFree(ds)); CK(cudaFree(scratch));
    }
}

static void test_norm(const HostWeights& hw, u32 count, bool a8) {
    Rng r(51 + count);
    std::vector<float> core(static_cast<size_t>(count) * VW);
    for (size_t i = 0; i < core.size(); ++i) {
        const u64 row = i / VW;
        float x = r.normal() * 0.3F;
        if (row % 9 == 2) x = 0.0F; else if (row % 9 == 5) x *= 1e-30F; else if (row % 9 == 7) x *= 1e5F;
        core[i] = x;
    }
    const auto z = make_bf16_values(core.size(), 61 + count, 3.0F, VW);
    DevWeights dw(hw); Workspace ws(count);
    ws.put(kOffCore, VW, core); ws.put(kOffZ, VW, z);
    std::int8_t* dcodes = nullptr; float* ddeltas = nullptr;
    const size_t ndelta = static_cast<size_t>((count + 255) / 256) * (VW / 32) * 256;
    g::A8Out out{};
    if (a8) {
        CK(cudaMalloc(&dcodes, static_cast<size_t>(count) * VW)); CK(cudaMalloc(&ddeltas, ndelta * 4));
        CK(cudaMemset(ddeltas, 0, ndelta * 4)); out = {dcodes, ddeltas};
    }
    g::launch_norm(ws.rows(), count, dw.w(), g_tables, nullptr, a8 ? &out : nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<std::int8_t> codes; std::vector<float> deltas;
    ref_norm(core, z, count, hw.norm, 1e-6F, a8 ? &codes : nullptr, a8 ? &deltas : nullptr);
    const std::string tag = " T=" + std::to_string(count) + (a8 ? " A8" : "");
    same(("norm core" + tag).c_str(), ws.get(kOffCore, VW), core);
    if (a8) {
        same(("norm A8 codes" + tag).c_str(), dev_download(dcodes, codes.size()), codes);
        same(("norm A8 deltas" + tag).c_str(), dev_download(ddeltas, ndelta), deltas);
        CK(cudaFree(dcodes)); CK(cudaFree(ddeltas));
    }
}

// whole core K3..K6: CUDA vs the chained host references, and CUDA in several blocks vs CUDA in one block
struct CoreOut { std::vector<float> qkv, beta, gate, core, state; std::vector<u16> hist; };
static CoreOut run_core_cuda(const HostWeights& hw, const std::vector<u16>& hidden, const std::vector<float>& qkv,
    const std::vector<float>& z, const std::vector<u16>& hist0, const std::vector<float>& state0, u32 count,
    const std::vector<u32>& splits) {
    DevWeights dw(hw); Workspace ws(count);
    ws.put(0, CW, qkv); ws.put(kOffZ, VW, z);
    u16* dh = dev_upload(hidden); u16* dhist = dev_upload(hist0); float* ds = dev_upload(state0);
    void* scratch = nullptr; CK(cudaMalloc(&scratch, g::core_scratch_bytes(count)));
    u32 first = 0;
    for (u32 s : splits) {
        const u32 n = s - first;
        g::launch_core(dh + static_cast<u64>(first) * H, H, ws.rows(first), n, dhist, ds, scratch, dw.w(), g_tables, nullptr);
        first = s;
    }
    CK(cudaDeviceSynchronize());
    CoreOut o{ws.get(0, CW), ws.get(kOffBeta, VH), ws.get(kOffGate, VH), ws.get(kOffCore, VW), dev_download(ds, g::kMatrixFloats),
        dev_download(dhist, CW * 4)};
    CK(cudaFree(dh)); CK(cudaFree(dhist)); CK(cudaFree(ds)); CK(cudaFree(scratch));
    return o;
}
static void test_core(const HostWeights& hw, const char* tag, u32 count) {
    const auto hidden = make_hidden(count, 71);
    const auto qkv = make_bf16_values(static_cast<size_t>(count) * CW, 72, 2.0F, CW);
    const auto z = make_bf16_values(static_cast<size_t>(count) * VW, 73, 3.0F, VW);
    const std::vector<u16> hist0(CW * 4, 0);
    const std::vector<float> state0(g::kMatrixFloats, 0.0F);
    const auto one = run_core_cuda(hw, hidden, qkv, z, hist0, state0, count, {count});
    // host chain
    CoreOut ref; ref.qkv = qkv; ref.hist = hist0; ref.state = state0;
    ref_gates(hidden, count, hw.beta, hw.alpha, hw.dt, hw.a, ref.beta, ref.gate);
    ref_conv(ref.qkv, count, ref.hist, hw.conv, 1e-6F);
    ref_recurrence(ref.qkv, ref.beta, ref.gate, count, ref.state, ref.core);
    ref_norm(ref.core, z, count, hw.norm, 1e-6F, nullptr, nullptr);
    const std::string t = std::string(" ") + tag + " T=" + std::to_string(count);
    same(("core chain beta" + t).c_str(), one.beta, ref.beta);
    same(("core chain gate" + t).c_str(), one.gate, ref.gate);
    same(("core chain qkv" + t).c_str(), one.qkv, ref.qkv);
    same(("core chain out" + t).c_str(), one.core, ref.core);
    same(("core chain state" + t).c_str(), one.state, ref.state);
    same(("core chain history" + t).c_str(), one.hist, ref.hist);
    // continuation: blocks of uneven sizes, including blocks shorter than the conv kernel
    for (const auto& splits : {std::vector<u32>{40, count}, std::vector<u32>{2, 3, 37, 101, count}, std::vector<u32>{64, 128, count}}) {
        const auto multi = run_core_cuda(hw, hidden, qkv, z, hist0, state0, count, splits);
        std::string s = " splits";
        for (u32 x : splits) s += " " + std::to_string(x);
        const bool ok = multi.core == one.core && multi.state == one.state && multi.hist == one.hist && multi.qkv == one.qkv &&
            multi.beta == one.beta && multi.gate == one.gate;  // operator== is exact for these (no NaN produced; also memcmp below)
        const bool bits = !std::memcmp(multi.core.data(), one.core.data(), one.core.size() * 4) &&
            !std::memcmp(multi.state.data(), one.state.data(), one.state.size() * 4) && multi.hist == one.hist &&
            !std::memcmp(multi.qkv.data(), one.qkv.data(), one.qkv.size() * 4) &&
            !std::memcmp(multi.beta.data(), one.beta.data(), one.beta.size() * 4) &&
            !std::memcmp(multi.gate.data(), one.gate.data(), one.gate.size() * 4);
        if (bits) { std::printf("ok   continuation%s%s == one block\n", t.c_str(), s.c_str()); ++g_pass; }
        else { std::printf("FAIL continuation%s%s (value-equal %d)\n", t.c_str(), s.c_str(), ok ? 1 : 0); ++g_fail; }
    }
}

// ------------------------------------------------------------------------------------------------ fixture replay
// Dense A8 projection (H1.1 of notes/qwen-prefill-map-a8.md): per 32-feature block quantize (dense_a8_quantize8), then per
// (row, out) acc = fma(dw * delta, (float)dot, acc) over blocks ascending, canonical NaN. Test-only, simple kernels.
__global__ void a8_quantize_kernel(const float* x, u32 rows, u32 width, std::int8_t* codes, float* deltas) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x, blocks = width / 32;
    if (i >= rows * blocks) return;
    const u32 r = i / blocks, b = i % blocks;
    const float* v = x + static_cast<u64>(r) * width + b * 32;
    bool bad = false; float amax = 0.0F;
    for (int l = 0; l < 32; ++l) { bad |= !isfinite(v[l]); amax = fmaxf(amax, fabsf(v[l])); }
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float delta = bad ? __uint_as_float(0x7fc00000U) : amax == 0.0F || tiny ? 1.0F : __fdiv_rn(amax, 127.0F);
    for (int l = 0; l < 32; ++l) {
        const float rounded = bad || tiny ? 0.0F : rintf(__fdiv_rn(v[l], delta));
        codes[static_cast<u64>(r) * width + b * 32 + l] = static_cast<std::int8_t>(static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded))));
    }
    deltas[static_cast<u64>(r) * blocks + b] = delta;
}
__global__ void a8_project_kernel(const std::int8_t* codes, const float* deltas, u32 rows, u32 width, const std::uint8_t* q8,
    u32 outs, float* y) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * outs) return;
    const u32 r = i / outs, n = i % outs, blocks = width / 32;
    const std::uint8_t* w = q8 + static_cast<u64>(n) * blocks * 34;
    float acc = 0.0F;
    for (u32 b = 0; b < blocks; ++b) {
        const std::uint8_t* blk = w + b * 34;
        const u16 hb = static_cast<u16>(blk[0] | (blk[1] << 8));
        const float dw = __half2float(__ushort_as_half(hb));
        int dot = 0;
        const std::int8_t* c = codes + static_cast<u64>(r) * width + b * 32;
        for (int l = 0; l < 32; ++l) dot += static_cast<int>(c[l]) * static_cast<int>(static_cast<std::int8_t>(blk[2 + l]));
        acc = __fmaf_rn(__fmul_rn(dw, deltas[static_cast<u64>(r) * blocks + b]), static_cast<float>(dot), acc);
    }
    y[static_cast<u64>(r) * outs + n] = acc != acc ? __uint_as_float(0x7fc00000U) : acc;
}
static std::vector<float> a8_project(const std::vector<float>& x, u32 rows, u32 width, const std::vector<std::uint8_t>& q8, u32 outs) {
    float* dx = dev_upload(x); std::uint8_t* dw = dev_upload(q8);
    std::int8_t* codes = nullptr; float* deltas = nullptr; float* dy = nullptr;
    CK(cudaMalloc(&codes, static_cast<size_t>(rows) * width)); CK(cudaMalloc(&deltas, static_cast<size_t>(rows) * width / 32 * 4));
    CK(cudaMalloc(&dy, static_cast<size_t>(rows) * outs * 4));
    a8_quantize_kernel<<<(rows * (width / 32) + 255) / 256, 256>>>(dx, rows, width, codes, deltas);
    a8_project_kernel<<<(rows * outs + 255) / 256, 256>>>(codes, deltas, rows, width, dw, outs, dy);
    CK(cudaGetLastError());
    auto y = dev_download(dy, static_cast<size_t>(rows) * outs);
    cudaFree(dx); cudaFree(dw); cudaFree(codes); cudaFree(deltas); cudaFree(dy);
    return y;
}
static std::vector<std::uint8_t> read_file(const std::string& p) {
    std::FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return {};
    _fseeki64(f, 0, SEEK_END); const long long n = _ftelli64(f); _fseeki64(f, 0, SEEK_SET);
    std::vector<std::uint8_t> v(static_cast<size_t>(n));
    if (n && std::fread(v.data(), 1, v.size(), f) != v.size()) v.clear();
    std::fclose(f);
    return v;
}
static void replay_fixture(const strix::qwen_gguf::Model& model, const std::string& dir) {
    const std::string dump_path = dir + "\\d64.dump";
    if (read_file(dump_path).empty()) { std::printf("skip fixture replay: %s not present\n", dump_path.c_str()); return; }
    strix::qwen_fixture::Dump dump(dump_path);
    const auto matrix = read_file(dir + "\\d64-state\\recurrent-matrix.raw");
    const auto conv = read_file(dir + "\\d64-state\\gdn-convolution.raw");
    // gated_delta_net = SerialOp 5; reads index 0 = hidden(0) BF16[2560], writes index 0 = hidden(1) BF16[2560]
    std::map<u32, std::map<u64, const strix::qwen_fixture::Record*>> in, out;
    for (const auto& r : dump.records())
        if (r.op == 5 && r.index == 0 && r.bytes == H * 2) (r.is_read ? in : out)[r.layer][r.position] = &r;
    std::printf("fixture: %zu GDN layers in the dump\n", in.size());
    u32 layers_ok = 0;
    for (const auto& [layer, rows_in] : in) {
        const u32 count = static_cast<u32>(rows_in.size());
        u64 expect_pos = 0; bool contiguous = true;
        for (const auto& [pos, rec] : rows_in) contiguous &= pos == expect_pos++;
        std::vector<u16> hidden(static_cast<size_t>(count) * H);
        u32 i = 0;
        for (const auto& [pos, rec] : rows_in) { const auto b = dump.load(*rec); std::memcpy(&hidden[static_cast<u64>(i++) * H], b.data(), H * 2); }
        const std::string p = "blk." + std::to_string(layer) + ".";
        const auto hw = real_weights(model, layer);
        // in projection: qkv ‖ z, round_bf16 kept as FP32 (GdnInputRowsSite::store, hip_gdn_launcher.hip:1063-1067)
        std::vector<float> xin(hidden.size());
        for (size_t k = 0; k < hidden.size(); ++k) xin[k] = h_bf2f(hidden[k]);
        auto qkv = a8_project(xin, count, H, model.bytes(p + "attn_qkv.weight"), CW);
        auto z = a8_project(xin, count, H, model.bytes(p + "attn_gate.weight"), VW);
        for (auto& v : qkv) v = h_rbf(v);
        for (auto& v : z) v = h_rbf(v);
        const std::vector<u16> hist0(CW * 4, 0);
        const std::vector<float> state0(g::kMatrixFloats, 0.0F);
        const auto o = run_core_cuda(hw, hidden, qkv, z, hist0, state0, count, {count});
        // out projection: BF16 (GdnOutputRowsSite::store :1081)
        const auto y = a8_project(o.core, count, VW, model.bytes(p + "ssm_out.weight"), H);
        std::vector<u16> got(y.size()), want;
        for (size_t k = 0; k < y.size(); ++k) got[k] = h_f2bf(y[k]);
        for (const auto& [pos, rec] : out[layer]) { const auto b = dump.load(*rec); const size_t at = want.size(); want.resize(at + H); std::memcpy(&want[at], b.data(), H * 2); }
        const std::string tag = " layer " + std::to_string(layer) + " rows " + std::to_string(count) + (contiguous ? "" : " (positions not 0..n-1)");
        bool ok = same(("fixture hidden(1)" + tag).c_str(), got, want);
        const u32 ordinal = layer - (layer + 1) / 4;
        if (matrix.size() >= (ordinal + 1ULL) * g::kMatrixFloats * 4 && conv.size() >= (ordinal + 1ULL) * g::kConvStateHalfs * 2) {
            std::vector<float> ms(g::kMatrixFloats); std::memcpy(ms.data(), matrix.data() + ordinal * g::kMatrixFloats * 4, ms.size() * 4);
            std::vector<u16> cs(g::kConvStateHalfs); std::memcpy(cs.data(), conv.data() + ordinal * g::kConvStateHalfs * 2, cs.size() * 2);
            ok &= same(("fixture recurrent matrix" + tag).c_str(), o.state, ms);
            ok &= same(("fixture conv history" + tag).c_str(), o.hist, cs);
        } else std::printf("note fixture state files missing or short: state not compared for layer %u\n", layer);
        layers_ok += ok;
    }
    std::printf("fixture: %u/%zu GDN layers exact\n", layers_ok, in.size());
}

// ------------------------------------------------------------------------------------------------ bench
static void bench(u32 count) {
    const auto hw = synthetic_weights(5);
    DevWeights dw(hw); Workspace ws(count);
    ws.put(0, CW, make_bf16_values(static_cast<size_t>(count) * CW, 81, 2.0F, CW));
    ws.put(kOffZ, VW, make_bf16_values(static_cast<size_t>(count) * VW, 82, 3.0F, VW));
    const auto hidden = make_hidden(count, 83);
    u16* dh = dev_upload(hidden); u16* dhist = nullptr; float* ds = nullptr; void* scratch = nullptr;
    CK(cudaMalloc(&dhist, CW * 8)); CK(cudaMemset(dhist, 0, CW * 8));
    CK(cudaMalloc(&ds, g::kMatrixFloats * 4)); CK(cudaMemset(ds, 0, g::kMatrixFloats * 4));
    CK(cudaMalloc(&scratch, g::core_scratch_bytes(count)));
    void* decay_scratch = static_cast<unsigned char*>(scratch) + (g::conv_scratch_bytes(count) + 255U) / 256U * 256U;
    cudaEvent_t e[6]; for (auto& x : e) CK(cudaEventCreate(&x));
    for (u32 lanes : {4U, 8U, 16U})
        for (int rep = 0; rep < 2; ++rep) {
            CK(cudaEventRecord(e[0]));
            g::launch_gates(dh, H, ws.rows(), count, dw.w(), g_tables, nullptr); CK(cudaEventRecord(e[1]));
            g::launch_conv(ws.rows(), count, dhist, scratch, dw.w(), g_tables, nullptr); CK(cudaEventRecord(e[2]));
            g::launch_recurrence(ws.rows(), count, ds, decay_scratch, g_tables, nullptr, {lanes}); CK(cudaEventRecord(e[3]));
            g::launch_norm(ws.rows(), count, dw.w(), g_tables, nullptr); CK(cudaEventRecord(e[4]));
            CK(cudaEventSynchronize(e[4]));
            float t[4]; for (int k = 0; k < 4; ++k) CK(cudaEventElapsedTime(&t[k], e[k], e[k + 1]));
            if (rep) std::printf("bench T=%u G=%u: gates %.3f ms conv %.3f ms recurrence %.3f ms norm %.3f ms total %.3f ms/layer\n",
                count, lanes, t[0], t[1], t[2], t[3], t[0] + t[1] + t[2] + t[3]);
        }
    cudaFree(dh); cudaFree(dhist); cudaFree(ds); cudaFree(scratch);
}

int main(int argc, char** argv) {
    std::string tables = "qrtx-build\\tables", fixture = "qrtx-build\\fx";
    bool real = true, replay_only = false; u32 bench_rows = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--tables" && i + 1 < argc) tables = argv[++i];
        else if (a == "--fixture" && i + 1 < argc) fixture = argv[++i];
        else if (a == "--no-real") real = false;
        else if (a == "--replay-only") replay_only = true;
        else if (a == "--bench" && i + 1 < argc) bench_rows = static_cast<u32>(std::atoi(argv[++i]));
    }
    g_tables = am::load_tables(tables);
    if (bench_rows) { bench(bench_rows); am::free_tables(g_tables); return 0; }
    if (replay_only) {
        strix::qwen_gguf::Model model;
        replay_fixture(model, fixture);
        am::free_tables(g_tables);
        std::printf("qwen_gdn_tests: %d passed, %d failed\n", g_pass, g_fail);
        return g_fail ? 1 : 0;
    }
    const auto syn = synthetic_weights(1);
    test_gates(syn, "synthetic", 40);
    test_conv(syn, "synthetic", 77, false);
    test_conv(syn, "synthetic", 2, false);
    test_conv(syn, "synthetic zero history", 64, true);
    test_recurrence(syn, 45, false);
    test_recurrence(syn, 16, true);
    test_norm(syn, 37, false);
    test_norm(syn, 300, true);
    test_core(syn, "synthetic", 150);
    if (real) {
        strix::qwen_gguf::Model model;
        const auto w0 = real_weights(model, 0);
        test_gates(w0, "blk.0", 48);
        test_conv(w0, "blk.0", 70, false);
        test_core(w0, "blk.0", 150);
        replay_fixture(model, fixture);
    }
    am::free_tables(g_tables);
    std::printf("qwen_gdn_tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
