// Tests of src/cuda/models/qwen/qwen_hc.cu (family "hc"): literal host references of the HIP kernels vs the CUDA kernels, bit for bit.
//   qwen_hc_tests [--tables DIR] [--fixture FILE] [--skip-real] [--skip-perf]
// Tables default qrtx-build\tables; the GGUF comes from tests/cuda/qwen_gguf_host.hpp. With --fixture (or when
// qrtx-build\fx\d64.dump exists) the HIP step dump is replayed: hyper mixes, router, inject, combine, expand,
// embedding and LM head steps, compared byte for byte with the HIP writes.
//
// The host references follow the HIP source and the gfx1151 ISA literally (see the header comment of qwen_hc.cu):
// each FMA is std::fma, every other operation one IEEE single operation (no contraction: fp_contract off below).
// expf/rsqrtf on the host are a transliteration of include/strix/cuda/amd_math.cuh over the same table files.
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma fp_contract(off)
#pragma float_control(precise, on)
#endif
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/cuda/amd_math_tables.hpp"
#include "qwen_fixture.hpp"
#include "qwen_gguf_host.hpp"
#include "strix/cuda/qwen_hc.hpp"
#include "strix/cuda/amd_math.cuh"

namespace hc = strix::cuda::qwen_hc;
namespace am = strix::cuda::amdmath;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

static int g_pass = 0, g_fail = 0;
static void report(const std::string& name, bool ok, const std::string& detail = {}) {
    (ok ? g_pass : g_fail)++;
    std::printf("[%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.empty() ? "" : " - ", detail.c_str());
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------------------------------------------------
// Host scalar helpers.
static float u2f(std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static std::uint32_t f2u(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float bf(std::uint16_t v) { return u2f(static_cast<std::uint32_t>(v) << 16U); }
// BF16 store with gfx1151's canonical generated NaN (0x7fc00000; x86 would give 0xffc00000).
static std::uint16_t to_bf(float v) { if (v != v) return 0x7fc0; const std::uint32_t b = f2u(v); return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U); }
static float half_to_float(std::uint16_t h) {
    const std::uint32_t s = (h >> 15) & 1U, e = (h >> 10) & 31U, m = h & 1023U;
    if (e == 0) { const float v = std::ldexp(static_cast<float>(m), -24); return s ? -v : v; }
    if (e == 31) return u2f((s << 31) | 0x7f800000U | (m << 13));
    return u2f((s << 31) | ((e + 112U) << 23) | (m << 13));
}
static std::uint16_t float_to_half(float f) {  // RNE, used only to build synthetic scales
    const std::uint32_t b = f2u(f), s = (b >> 16) & 0x8000U;
    const int e = static_cast<int>((b >> 23) & 255U) - 127 + 15;
    std::uint32_t m = b & 0x7fffffU;
    if (e <= 0) return static_cast<std::uint16_t>(s);  // synthetic scales are never that small
    if (e >= 31) return static_cast<std::uint16_t>(s | 0x7c00U);
    std::uint32_t r = (static_cast<std::uint32_t>(e) << 10) | (m >> 13);
    const std::uint32_t rem = m & 0x1fffU;
    if (rem > 0x1000U || (rem == 0x1000U && (r & 1U))) ++r;
    return static_cast<std::uint16_t>(s | r);
}

// Host port of amd_math.cuh (same tables, same algorithm).
struct HostMath {
    std::vector<std::uint32_t> exp_bits, rsq_bits;
    std::map<std::uint32_t, float> exp_hard, rsq_hard;
};
static HostMath g_math;
static std::vector<std::uint32_t> read_u32_file(const std::string& p) {
    std::FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) throw std::runtime_error("missing " + p);
    std::fseek(f, 0, SEEK_END); const long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<std::uint32_t> v(static_cast<std::size_t>(n) / 4);
    if (std::fread(v.data(), 4, v.size(), f) != v.size()) throw std::runtime_error("short " + p);
    std::fclose(f);
    return v;
}
static void load_host_math(const std::string& dir) {
    g_math.exp_bits = read_u32_file(dir + "/tbl_exp_pm.bin");
    g_math.rsq_bits = read_u32_file(dir + "/tbl_rsq_1_4.bin");
    auto raw = read_u32_file(dir + "/hard_v_exp_f32.bin");
    for (std::size_t i = 0, n = raw.size() / 2; i < n; ++i) {
        const std::uint32_t mag = raw[i] & 0x7fffffffU;
        if (mag < (102U << 23) || mag >= (127U << 23)) continue;
        g_math.exp_hard[raw[i]] = u2f(raw[n + i]);
    }
    raw = read_u32_file(dir + "/hard_v_rsq_f32.bin");
    for (std::size_t i = 0, n = raw.size() / 2; i < n; ++i) {
        std::uint32_t b = raw[i], y = raw[n + i];
        if (((b >> 23) & 255U) == 0U || ((b >> 23) & 255U) == 255U) continue;  // outside the emulated domain (as the loader)
        const int e = static_cast<int>(b >> 23) - 127;
        const int k = e >= 0 ? e / 2 : -((-e + 1) / 2);
        b = (b & 0x007fffffU) | (static_cast<std::uint32_t>(e - 2 * k + 127) << 23);
        y = static_cast<std::uint32_t>(static_cast<int>(y) + (k << 23));
        g_math.rsq_hard[b] = u2f(y);
    }
}
static bool tbit(const std::vector<std::uint32_t>& t, std::uint32_t i) { return (t[i >> 5] >> (i & 31U)) & 1U; }
static float other_neighbour(float f, double r) {
    const std::uint32_t b = f2u(f);
    const bool up = r > static_cast<double>(f);
    const bool away = (b >> 31) ? !up : up;
    return u2f(away ? b + 1U : b - 1U);
}
static float h_v_exp(float x) {
    const std::uint32_t b = f2u(x), mag = b & 0x7fffffffU;
    const double r = std::exp2(static_cast<double>(x));
    const float f = static_cast<float>(r);
    if (mag < (102U << 23)) return f;
    const auto it = g_math.exp_hard.find(b);
    if (it != g_math.exp_hard.end()) return it->second;
    const std::uint32_t index = (b >> 31) * (25U << 23) + (mag - (102U << 23));  // positive binades, then negative
    return tbit(g_math.exp_bits, index) ? other_neighbour(f, r) : f;
}
static float h_v_rsq(float x) {
    const std::uint32_t b = f2u(x);
    const int e = static_cast<int>(b >> 23) - 127;
    const int k = e >= 0 ? e / 2 : -((-e + 1) / 2);
    const std::uint32_t cb = (b & 0x007fffffU) | (static_cast<std::uint32_t>(e - 2 * k + 127) << 23);
    float y;
    const auto it = g_math.rsq_hard.find(cb);
    if (it != g_math.rsq_hard.end()) y = it->second;
    else {
        const double r = 1.0 / std::sqrt(static_cast<double>(u2f(cb)));
        const float f = static_cast<float>(r);
        y = tbit(g_math.rsq_bits, cb - (127U << 23)) ? other_neighbour(f, r) : f;
    }
    return u2f(static_cast<std::uint32_t>(static_cast<int>(f2u(y)) - (k << 23)));
}
static float h_expf(float x) {
    const float kLog2e = u2f(0x3fb8aa3bU), kLog2eLo = u2f(0x32a5705fU);
    const float ph = x * kLog2e;
    const float e = std::nearbyint(ph);
    float pl = std::fma(x, kLog2e, -ph);
    pl = std::fma(x, kLog2eLo, pl);
    const float a = (ph - e) + pl;
    float y;
    if (std::fabs(a) < 1.0F) y = std::scalbn(h_v_exp(a), static_cast<int>(e));
    else y = a != a ? a : 0.0F;
    if (u2f(0xc2ce8ed0U) > x) y = 0.0F;
    if (u2f(0x42b17218U) < x) y = u2f(0x7f800000U);
    return y;
}
static float h_rsqrtf(float x) {
    const bool small = u2f(0x00800000U) > x;
    const float s = small ? x * u2f(0x4b800000U) : x;
    float y;
    const std::uint32_t sb = f2u(s);
    if (sb == 0U) y = u2f(0x7f800000U);
    else if (sb == 0x80000000U) y = u2f(0xff800000U);
    else if (sb >= 0x7f800000U) y = sb == 0x7f800000U ? 0.0F : u2f(0x7fc00000U);
    else y = h_v_rsq(s);
    return small ? y * u2f(0x45800000U) : y;
}
static float h_sigmoid(float x) { return 1.0F / (1.0F + h_expf(-x)); }

// ---------------------------------------------------------------------------------------------------------------------
// Literal host references.
constexpr unsigned H = hc::kHidden, S = hc::kStreams, W = hc::kWide, RK = hc::kRank, P = hc::kNormalizedPitch;

static float q8_value(const std::uint8_t* tensor, std::size_t row, unsigned row_width, unsigned feature) {
    const std::uint8_t* block = tensor + (row * (row_width / 32U) + feature / 32U) * 34U;
    const float scale = half_to_float(static_cast<std::uint16_t>(block[0] | (block[1] << 8)));
    return scale * static_cast<float>(static_cast<std::int8_t>(block[2 + feature % 32U]));
}

struct HcWeights { std::vector<std::uint8_t> down, up; std::vector<float> norm, inject; };  // inject empty: final mix

// One row of the HC mix: normalized[10240] + low[320] (P floats), injection[4], output[2560].
static void ref_hc_row(const HcWeights& w, float eps, const std::uint16_t* in, float* normalized, float* injection, std::uint16_t* out) {
    // rows_prepare_hyper_norm_kernel (:1757-1776)
    for (unsigned s = 0; s < S; ++s) {
        float red[256];
        for (unsigned l = 0; l < 256; ++l) {
            float sum = 0.0F;
            for (unsigned f = l; f < H; f += 256) { const float v = bf(in[s * H + f]); sum = std::fma(v, v, sum); }
            red[l] = sum;
        }
        for (unsigned stride = 128; stride > 0; stride >>= 1)
            for (unsigned l = 0; l < stride; ++l) red[l] = red[l] + red[l + stride];
        const float inverse = h_rsqrtf(red[0] / static_cast<float>(H) + eps);
        for (unsigned f = 0; f < H; ++f) normalized[s * H + f] = (bf(in[s * H + f]) * inverse) * w.norm[s * H + f];
    }
    // split-K projection (:2907-3034 / serial :978-1087, :1239-1265)
    const auto project = [&](auto coefficient) {
        float total = 0.0F;
        for (unsigned tile = 0; tile < 16; ++tile) {
            float leaf[32];
            for (unsigned l = 0; l < 32; ++l) {
                float acc = 0.0F;
                for (unsigned sc = 0; sc < 5; ++sc)
                    for (unsigned j = 0; j < 4; ++j) {
                        const unsigned f = tile * 640 + sc * 128 + 4 * l + j;
                        acc = std::fma(normalized[f], coefficient(f), acc);
                    }
                leaf[l] = acc;
            }
            for (unsigned d = 16; d > 0; d >>= 1) {  // xor butterfly over window lanes
                float next[32];
                for (unsigned l = 0; l < 32; ++l) next[l] = leaf[l] + leaf[l ^ d];
                std::memcpy(leaf, next, sizeof(leaf));
            }
            total = total + leaf[0];
        }
        return total / static_cast<float>(S);
    };
    for (unsigned o = 0; o < RK; ++o) {
        const float scaled = project([&](unsigned f) { return q8_value(w.down.data(), o, W, f); });
        normalized[W + o] = scaled * h_sigmoid(scaled);
    }
    if (!w.inject.empty())
        for (unsigned o = 0; o < S; ++o) {
            const float scaled = project([&](unsigned f) { return w.inject[static_cast<std::size_t>(o) * W + f]; });
            const float sig = h_sigmoid(scaled);
            injection[o] = sig + sig;  // 2 * sig, ISA v_add_f32 sig, sig
        }
    // wide2_hyper_output_kernel (:3193-3268): gate chains then the contracted stream mix
    for (unsigned f = 0; f < H; ++f) {
        float mixed = 0.0F;
        for (unsigned s = 0; s < S; ++s) {
            float g = 0.0F;
            for (unsigned k = 0; k < RK; ++k) g = std::fma(normalized[W + k], q8_value(w.up.data(), s * H + f, RK, k), g);
            mixed = std::fma(normalized[s * H + f], h_sigmoid(g), mixed);
        }
        out[f] = to_bf(mixed * 0.25F);
    }
}

static void ref_route_logits(const std::uint16_t* x, const float* wt, float* logits) {
    for (unsigned e = 0; e < hc::kExperts; ++e) {
        float sum = 0.0F;
        for (unsigned f = 0; f < H; ++f) sum = std::fma(bf(x[f]), wt[static_cast<std::size_t>(e) * H + f], sum);
        logits[e] = sum;
    }
}
static bool better(float lv, unsigned li, float rv, unsigned ri) { return lv > rv || (lv == rv && li < ri); }
static void ref_route_select(const float* in, unsigned top_k, bool normalize, std::uint32_t* ids, float* weights) {
    constexpr unsigned E = hc::kExperts;
    float lg[E];
    std::memcpy(lg, in, sizeof(lg));
    float scratch[256];
    for (unsigned l = 0; l < 256; ++l) scratch[l] = std::fmax(std::fmax(-INFINITY, lg[l]), lg[l + 256]);
    for (unsigned st = 128; st > 0; st >>= 1) for (unsigned l = 0; l < st; ++l) scratch[l] = std::fmax(scratch[l], scratch[l + st]);
    const float maximum = scratch[0];
    for (unsigned l = 0; l < 256; ++l) {
        float sum = 0.0F;
        for (unsigned e = l; e < E; e += 256) { lg[e] = h_expf(lg[e] - maximum); sum = sum + lg[e]; }
        scratch[l] = sum;
    }
    for (unsigned st = 128; st > 0; st >>= 1) for (unsigned l = 0; l < st; ++l) scratch[l] = scratch[l] + scratch[l + st];
    const float den = scratch[0];
    for (unsigned e = 0; e < E; ++e) lg[e] = lg[e] / den;
    // top-k: the wave butterfly returns the maximum under better(); NaN excluded; winner removed.
    std::vector<bool> taken(E, false);
    float bv[32]; unsigned bi[32];
    for (unsigned slot = 0; slot < top_k; ++slot) {
        bool has = false; float best = -INFINITY; unsigned index = E;
        for (unsigned e = 0; e < E; ++e) {
            if (taken[e] || !(lg[e] == lg[e])) continue;
            if (!has || better(lg[e], e, best, index)) { has = true; best = lg[e]; index = e; }
        }
        bv[slot] = has ? best : -INFINITY; bi[slot] = has ? index : E;
        if (has) taken[index] = true;
    }
    float ssum = 0.0F;
    for (unsigned s = 0; s < top_k; ++s) ssum = ssum + bv[s];
    if (!normalize) ssum = 1.0F;
    for (unsigned s = 0; s < top_k; ++s) { ids[s] = bi[s]; weights[s] = bv[s] / ssum; }
}
static float ref_lm_row(const std::uint16_t* x, const std::uint8_t* wt, std::size_t row) {
    float partial[32][8] = {};
    for (unsigned lane = 0; lane < 32; ++lane)
        for (unsigned chunk = 0; chunk < H; chunk += 256)
            for (unsigned l = 0; l < 8; ++l) {
                const unsigned f = chunk + 8 * lane + l;
                partial[lane][l] = std::fma(bf(x[f]), q8_value(wt, row, H, f), partial[lane][l]);
            }
    for (unsigned d = 16; d > 0; d /= 2)
        for (unsigned lane = 0; lane < d; ++lane)
            for (unsigned l = 0; l < 8; ++l) partial[lane][l] = partial[lane][l] + partial[lane + d][l];
    float* p = partial[0];
    p[0] = p[0] + p[4]; p[1] = p[1] + p[5]; p[2] = p[2] + p[6]; p[3] = p[3] + p[7];
    p[0] = p[0] + p[2]; p[1] = p[1] + p[3];
    p[0] = p[0] + p[1];
    return p[0];
}

// ---------------------------------------------------------------------------------------------------------------------
// Device buffers and comparison.
template <class T> struct Dev {
    T* p = nullptr; std::size_t n = 0;
    explicit Dev(std::size_t count) : n(count) { CK(cudaMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T))); }
    Dev(const std::vector<T>& v) : Dev(v.size()) { if (n) CK(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete; Dev& operator=(const Dev&) = delete;
    std::vector<T> get() const { std::vector<T> v(n); if (n) CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
};

// Compares element bit patterns; nan_equiv: any NaN equals any NaN (payloads are not reproduced, plan rule 3).
template <class T>
static bool same(const std::string& name, const std::vector<T>& want, const std::vector<T>& got, bool nan_equiv = false,
    std::size_t row_elems = 0) {
    if (want.size() != got.size()) { report(name, false, "size " + std::to_string(want.size()) + " vs " + std::to_string(got.size())); return false; }
    std::size_t bad = 0, first = SIZE_MAX, nan_pairs = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (std::memcmp(&want[i], &got[i], sizeof(T)) == 0) continue;
        bool both_nan = false;
        if constexpr (sizeof(T) == 4) { float a, b; std::memcpy(&a, &want[i], 4); std::memcpy(&b, &got[i], 4); both_nan = a != a && b != b; }
        if constexpr (sizeof(T) == 2) { const std::uint16_t a = static_cast<std::uint16_t>(want[i]), b = static_cast<std::uint16_t>(got[i]);
            both_nan = (a & 0x7fffU) > 0x7f80U && (b & 0x7fffU) > 0x7f80U; }
        if (nan_equiv && both_nan) { ++nan_pairs; continue; }
        if (first == SIZE_MAX) first = i;
        ++bad;
    }
    char buf[256] = {};
    if (bad) {
        std::uint32_t a = 0, b = 0; std::memcpy(&a, &want[first], sizeof(T)); std::memcpy(&b, &got[first], sizeof(T));
        std::snprintf(buf, sizeof(buf), "%zu/%zu differ; first at %zu (row %zu, elem %zu): want 0x%08x got 0x%08x", bad, want.size(), first,
            row_elems ? first / row_elems : 0, row_elems ? first % row_elems : first, a, b);
    } else std::snprintf(buf, sizeof(buf), "%zu elements bit-exact%s", want.size(),
        nan_pairs ? (" (" + std::to_string(nan_pairs) + " NaN/NaN pairs)").c_str() : "");
    report(name, bad == 0, buf);
    return bad == 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// Synthetic data.
static std::mt19937_64 g_rng(0x51e7c0de);
static float nrand(float sd) { return std::normal_distribution<float>(0.0F, sd)(g_rng); }
static std::vector<std::uint8_t> synth_q8(std::size_t rows, unsigned width, float scale_lo, float scale_hi) {
    std::vector<std::uint8_t> t(rows * (width / 32U) * 34U);
    std::uniform_real_distribution<float> sd(scale_lo, scale_hi);
    std::uniform_int_distribution<int> code(-128, 127), edge(0, 63);
    for (std::size_t b = 0; b < rows * (width / 32U); ++b) {
        std::uint8_t* p = t.data() + b * 34U;
        float sc = sd(g_rng);
        const int pick = edge(g_rng);
        if (pick == 0) sc = 0.0F; else if (pick == 1) sc = -sc;
        const std::uint16_t h = float_to_half(sc);
        p[0] = static_cast<std::uint8_t>(h & 255U); p[1] = static_cast<std::uint8_t>(h >> 8);
        for (unsigned i = 0; i < 32; ++i) p[2 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(pick == 2 ? -128 : code(g_rng)));
    }
    return t;
}
static std::vector<std::uint16_t> synth_bf16(std::size_t n, float sd) {
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) x = to_bf(nrand(sd));
    return v;
}
template <class T> static std::vector<T> f32_vector(const std::vector<std::uint8_t>& bytes) {
    std::vector<T> v(bytes.size() / sizeof(T)); std::memcpy(v.data(), bytes.data(), v.size() * sizeof(T)); return v;
}

// ---------------------------------------------------------------------------------------------------------------------
// HC mix runner.
struct HcRun { std::vector<float> normalized, injection; std::vector<std::uint16_t> output; double ms = 0; };
static HcRun run_hc(const HcWeights& w, const std::vector<std::uint16_t>& input, unsigned rows, float eps, const am::Tables& t, int reps = 1,
    bool split = false) {
    Dev<std::uint8_t> down(w.down), up(w.up);
    Dev<float> norm(w.norm), inj(w.inject.empty() ? std::vector<float>{} : w.inject);
    Dev<std::uint16_t> in(input);
    Dev<float> normalized(static_cast<std::size_t>(rows) * P), injection(static_cast<std::size_t>(rows) * S);
    Dev<std::uint16_t> out(static_cast<std::size_t>(rows) * H);
    Dev<std::uint8_t> packed(hc::hc_packed_bytes());
    Dev<float> gates(split ? static_cast<std::size_t>(rows) * S * H : 0);
    CK(cudaMemset(normalized.p, 0xff, normalized.n * 4)); CK(cudaMemset(injection.p, 0xff, injection.n * 4));
    hc::HcMixArgs a;
    a.input = in.p; a.norm = norm.p; a.inject = w.inject.empty() ? nullptr : inj.p; a.packed = hc::hc_packed_view(packed.p);
    a.epsilon = eps; a.rows = rows; a.normalized = normalized.p; a.injection = injection.p; a.output = out.p;
    if (split) a.gates = gates.p;
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    hc::hc_pack(down.p, up.p, a.packed, nullptr);
    hc::hc_mix(a, t, nullptr);
    CK(cudaDeviceSynchronize());
    HcRun r;
    if (reps > 1) {
        CK(cudaEventRecord(e0));
        for (int i = 0; i < reps; ++i) { hc::hc_pack(down.p, up.p, a.packed, nullptr); hc::hc_mix(a, t, nullptr); }
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); r.ms = ms / reps;
        const char* names[5] = {"pack", "norm", "projection", split ? "output gates (split)" : "output (fused)", "split epilogue"};
        for (unsigned k = 0; k < (split ? 5U : 4U); ++k) {
            CK(cudaEventRecord(e0));
            for (int i = 0; i < reps; ++i) {
                if (k == 0) hc::hc_pack(down.p, up.p, a.packed, nullptr); else hc::hc_mix(a, t, nullptr, 1U << (k - 1));
            }
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&ms, e0, e1));
            std::printf("perf:   %s %.3f ms\n", names[k], ms / reps);
        }
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    r.normalized = normalized.get(); r.injection = injection.get(); r.output = out.get();
    return r;
}
static void check_hc(const std::string& name, const HcWeights& w, const std::vector<std::uint16_t>& input, unsigned rows, float eps,
    const am::Tables& t, const std::vector<unsigned>& check_rows = {}) {
    const HcRun got = run_hc(w, input, rows, eps, t);
    std::vector<unsigned> list = check_rows;
    if (list.empty()) for (unsigned r = 0; r < rows; ++r) list.push_back(r);
    std::vector<float> wn, gn, wi, gi; std::vector<std::uint16_t> wo, go;
    for (unsigned r : list) {
        std::vector<float> n(P), inj(S, 0.0F); std::vector<std::uint16_t> o(H);
        ref_hc_row(w, eps, input.data() + static_cast<std::size_t>(r) * W, n.data(), inj.data(), o.data());
        wn.insert(wn.end(), n.begin(), n.end());
        gn.insert(gn.end(), got.normalized.begin() + static_cast<std::ptrdiff_t>(r) * P, got.normalized.begin() + static_cast<std::ptrdiff_t>(r + 1) * P);
        if (!w.inject.empty()) {
            wi.insert(wi.end(), inj.begin(), inj.end());
            gi.insert(gi.end(), got.injection.begin() + static_cast<std::ptrdiff_t>(r) * S, got.injection.begin() + static_cast<std::ptrdiff_t>(r + 1) * S);
        }
        wo.insert(wo.end(), o.begin(), o.end());
        go.insert(go.end(), got.output.begin() + static_cast<std::ptrdiff_t>(r) * H, got.output.begin() + static_cast<std::ptrdiff_t>(r + 1) * H);
    }
    same(name + " normalized+low", wn, gn, false, P);
    if (!w.inject.empty()) same(name + " injection", wi, gi, false, S);
    same(name + " output", wo, go, false, H);
}

static HcWeights synth_hc(bool with_inject) {
    HcWeights w;
    w.down = synth_q8(RK, W, 2e-3F, 2e-2F);
    w.up = synth_q8(W, RK, 2e-3F, 2e-2F);
    w.norm.resize(W); for (auto& v : w.norm) v = 1.0F + nrand(0.2F);
    if (with_inject) { w.inject.resize(static_cast<std::size_t>(S) * W); for (auto& v : w.inject) v = nrand(0.02F); }
    return w;
}
static std::vector<std::uint16_t> hc_input(unsigned rows, float sd) {
    auto v = synth_bf16(static_cast<std::size_t>(rows) * W, sd);
    if (rows > 3) {
        std::fill(v.begin(), v.begin() + W, std::uint16_t{0});                         // row 0: all zero (sum 0, rsqrt(eps))
        for (unsigned f = 0; f < W; ++f) v[W + f] = to_bf(nrand(1.0F) * 1e-30F);       // row 1: tiny (red/2560 + eps ~ eps)
        for (unsigned f = 0; f < W; ++f) v[2 * W + f] = to_bf(nrand(1.0F) * 3e18F);    // row 2: huge (v*v sums overflow to inf)
        for (unsigned f = 0; f < W; f += 7) v[3 * W + f] = to_bf(nrand(1.0F) * 1e-39F); // row 3: subnormal bf16 mixed in
    }
    return v;
}

// ---------------------------------------------------------------------------------------------------------------------
static void test_hc_synthetic(const am::Tables& t) {
    const HcWeights w = synth_hc(true);
    check_hc("hc synthetic 40 rows (inject)", w, hc_input(40, 1.0F), 40, 1e-6F, t);
    const HcWeights wf = synth_hc(false);
    check_hc("hc synthetic 17 rows (final, no inject)", wf, hc_input(17, 0.5F), 17, 1e-6F, t);
}
static void test_hc_real(const strix::qwen_gguf::Model& m, const am::Tables& t) {
    HcWeights w;
    w.down = m.bytes("blk.0.hc_attn_down.weight"); w.up = m.bytes("blk.0.hc_attn_up.weight");
    w.norm = f32_vector<float>(m.bytes("blk.0.hc_attn_norm.weight")); w.inject = f32_vector<float>(m.bytes("blk.0.hc_attn_inject.weight"));
    check_hc("hc real blk.0.hc_attn 72 rows", w, hc_input(72, 0.05F), 72, 1e-6F, t);
    HcWeights f;
    f.down = m.bytes("output_hc_down.weight"); f.up = m.bytes("output_hc_up.weight");
    f.norm = f32_vector<float>(m.bytes("output_hc_norm.weight"));
    check_hc("hc real output_hc 33 rows", f, hc_input(33, 1.5F), 33, 1e-6F, t);
}
static void test_hc_perf(const strix::qwen_gguf::Model* m, const am::Tables& t) {
    HcWeights w;
    if (m) {
        w.down = m->bytes("blk.0.hc_ffn_down.weight"); w.up = m->bytes("blk.0.hc_ffn_up.weight");
        w.norm = f32_vector<float>(m->bytes("blk.0.hc_ffn_norm.weight")); w.inject = f32_vector<float>(m->bytes("blk.0.hc_ffn_inject.weight"));
    } else w = synth_hc(true);
    const unsigned rows = 8192;
    const auto input = hc_input(rows, 0.05F);
    const HcRun got = run_hc(w, input, rows, 1e-6F, t, 5);
    std::printf("perf: hc mix fused (pack + norm + projection + output), 8192 rows: %.3f ms per call\n", got.ms);
    const HcRun split = run_hc(w, input, rows, 1e-6F, t, 5, true);
    std::printf("perf: hc mix split epilogue, 8192 rows: %.3f ms per call\n", split.ms);
    same("hc 8192 rows split-epilogue output == fused output", got.output, split.output, false, H);
    std::vector<unsigned> rs;
    for (unsigned r = 0; r < 16; ++r) rs.push_back(r);
    for (unsigned r = 4000; r < 4040; ++r) rs.push_back(r);
    for (unsigned r = rows - 24; r < rows; ++r) rs.push_back(r);
    std::vector<float> wn, gn, wi, gi; std::vector<std::uint16_t> wo, go;
    for (unsigned r : rs) {
        std::vector<float> n(P), inj(S); std::vector<std::uint16_t> o(H);
        ref_hc_row(w, 1e-6F, input.data() + static_cast<std::size_t>(r) * W, n.data(), inj.data(), o.data());
        wn.insert(wn.end(), n.begin(), n.end()); wi.insert(wi.end(), inj.begin(), inj.end()); wo.insert(wo.end(), o.begin(), o.end());
        gn.insert(gn.end(), got.normalized.begin() + static_cast<std::ptrdiff_t>(r) * P, got.normalized.begin() + static_cast<std::ptrdiff_t>(r + 1) * P);
        gi.insert(gi.end(), got.injection.begin() + static_cast<std::ptrdiff_t>(r) * S, got.injection.begin() + static_cast<std::ptrdiff_t>(r + 1) * S);
        go.insert(go.end(), got.output.begin() + static_cast<std::ptrdiff_t>(r) * H, got.output.begin() + static_cast<std::ptrdiff_t>(r + 1) * H);
    }
    same("hc 8192 rows (80 sampled rows) normalized+low", wn, gn, false, P);
    same("hc 8192 rows (80 sampled rows) injection", wi, gi, false, S);
    same("hc 8192 rows (80 sampled rows) output", wo, go, false, H);
}

static void test_router(const strix::qwen_gguf::Model* m, const am::Tables& t, bool perf) {
    std::vector<float> wt;
    if (m) wt = f32_vector<float>(m->bytes("blk.0.ffn_gate_inp.weight"));
    else { wt.resize(static_cast<std::size_t>(hc::kExperts) * H); for (auto& v : wt) v = nrand(0.02F); }
    // ties: expert 7 = expert 5 and expert 300 = expert 44 (equal logits -> equal probabilities -> lower index first)
    std::copy(wt.begin() + 5 * H, wt.begin() + 6 * H, wt.begin() + 7 * H);
    std::copy(wt.begin() + 44 * H, wt.begin() + 45 * H, wt.begin() + 300 * H);
    const unsigned rows = 72;
    auto x = synth_bf16(static_cast<std::size_t>(rows) * H, 1.0F);
    std::fill(x.begin(), x.begin() + H, std::uint16_t{0});  // row 0: all logits 0 -> uniform softmax, top-10 = 0..9
    Dev<float> dw(wt); Dev<std::uint16_t> dx(x); Dev<float> dl(static_cast<std::size_t>(rows) * hc::kExperts);
    hc::route_logits(dx.p, dw.p, rows, dl.p, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<float> want(dl.n);
    for (unsigned r = 0; r < rows; ++r) ref_route_logits(x.data() + static_cast<std::size_t>(r) * H, wt.data(), want.data() + static_cast<std::size_t>(r) * hc::kExperts);
    const std::string tag = m ? "real blk.0.ffn_gate_inp" : "synthetic";
    auto logits = dl.get();
    same("router logits " + tag + " 72 rows", want, logits, false, hc::kExperts);
    // selection on these logits plus crafted rows
    const unsigned extra = 6, total = rows + extra;
    std::vector<float> sel_in = want;
    sel_in.resize(static_cast<std::size_t>(total) * hc::kExperts, 0.0F);
    float* e = sel_in.data() + static_cast<std::size_t>(rows) * hc::kExperts;
    for (unsigned i = 0; i < hc::kExperts; ++i) e[i] = 3.0F;                                   // all equal
    for (unsigned i = 0; i < hc::kExperts; ++i) e[512 + i] = (i % 3 == 0) ? -INFINITY : nrand(2.0F);  // -inf experts
    for (unsigned i = 0; i < hc::kExperts; ++i) e[1024 + i] = i == 77 ? 1e30F : nrand(1.0F);        // one dominant
    for (unsigned i = 0; i < hc::kExperts; ++i) e[1536 + i] = static_cast<float>(i % 5);            // many ties at the top
    for (unsigned i = 0; i < hc::kExperts; ++i) e[2048 + i] = i < 500 ? -INFINITY : static_cast<float>(i);  // only 12 finite
    for (unsigned i = 0; i < hc::kExperts; ++i) e[2560 + i] = i == 3 ? NAN : nrand(1.0F);         // a NaN logit
    Dev<float> ds(sel_in); Dev<std::uint32_t> di(static_cast<std::size_t>(total) * hc::kTopK); Dev<float> dwt(static_cast<std::size_t>(total) * hc::kTopK);
    for (int normalize = 1; normalize >= 0; --normalize) {
        hc::route_select(ds.p, total, hc::kTopK, normalize != 0, t, di.p, dwt.p, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<std::uint32_t> wi(di.n); std::vector<float> ww(dwt.n);
        for (unsigned r = 0; r < total; ++r)
            ref_route_select(sel_in.data() + static_cast<std::size_t>(r) * hc::kExperts, hc::kTopK, normalize != 0,
                wi.data() + static_cast<std::size_t>(r) * hc::kTopK, ww.data() + static_cast<std::size_t>(r) * hc::kTopK);
        const std::string nt = normalize ? " (normalized)" : " (raw)";
        same("router select ids " + tag + " + 6 edge rows" + nt, wi, di.get(), false, hc::kTopK);
        same("router select weights " + tag + " + 6 edge rows" + nt, ww, dwt.get(), true, hc::kTopK);
        if (normalize) {
            const auto ids = di.get();
            bool ok = true;
            for (unsigned s = 0; s < 10; ++s) ok = ok && ids[static_cast<std::size_t>(rows) * 10 + s] == s;  // all equal -> 0..9
            const std::uint32_t* tie = ids.data() + static_cast<std::size_t>(rows + 3) * 10;               // i%5 == 4 first
            for (unsigned s = 0; s < 10; ++s) ok = ok && tie[s] == 4 + 5 * s;
            report("router select tie rules (equal row -> 0..9, ties -> ascending index)", ok);
        }
    }
    if (perf) {
        const unsigned big = 8192;
        auto xb = synth_bf16(static_cast<std::size_t>(big) * H, 1.0F);
        Dev<std::uint16_t> dxb(xb); Dev<float> dlb(static_cast<std::size_t>(big) * hc::kExperts);
        Dev<std::uint32_t> dib(static_cast<std::size_t>(big) * 10); Dev<float> dwb(static_cast<std::size_t>(big) * 10);
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        hc::route_logits(dxb.p, dw.p, big, dlb.p, nullptr);
        CK(cudaEventRecord(a));
        for (int i = 0; i < 5; ++i) hc::route_logits(dxb.p, dw.p, big, dlb.p, nullptr);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms1 = 0; CK(cudaEventElapsedTime(&ms1, a, b));
        CK(cudaEventRecord(a));
        for (int i = 0; i < 5; ++i) hc::route_select(dlb.p, big, 10, true, t, dib.p, dwb.p, nullptr);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms2 = 0; CK(cudaEventElapsedTime(&ms2, a, b));
        std::printf("perf: router 8192 rows: logits %.3f ms, select %.3f ms\n", ms1 / 5, ms2 / 5);
        const auto lb = dlb.get();
        std::vector<float> wl, gl;
        for (unsigned r : {0U, 1U, 4097U, 8191U}) {
            std::vector<float> ref(hc::kExperts);
            ref_route_logits(xb.data() + static_cast<std::size_t>(r) * H, wt.data(), ref.data());
            wl.insert(wl.end(), ref.begin(), ref.end());
            gl.insert(gl.end(), lb.begin() + static_cast<std::ptrdiff_t>(r) * hc::kExperts, lb.begin() + static_cast<std::ptrdiff_t>(r + 1) * hc::kExperts);
        }
        same("router logits 8192 rows (4 sampled rows)", wl, gl, false, hc::kExperts);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
}

static void test_elementwise(const strix::qwen_gguf::Model* m) {
    const unsigned rows = 24;
    // embedding
    {
        std::vector<std::uint8_t> table;
        std::size_t vocab;
        if (m) { table = m->bytes("token_embd.weight"); vocab = hc::kVocab; }
        else { vocab = 4096; table = synth_q8(vocab, H, 1e-3F, 5e-2F); }
        std::vector<std::uint32_t> ids(rows);
        for (unsigned r = 0; r < rows; ++r) ids[r] = static_cast<std::uint32_t>((static_cast<std::uint64_t>(r) * 104729ULL + 17ULL) % vocab);
        ids[0] = 0; ids[1] = static_cast<std::uint32_t>(vocab - 1);
        Dev<std::uint8_t> dt(table); Dev<std::uint32_t> di(ids); Dev<std::uint16_t> out(static_cast<std::size_t>(rows) * H);
        hc::token_embedding(dt.p, di.p, rows, out.p, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<std::uint16_t> want(out.n);
        for (unsigned r = 0; r < rows; ++r) for (unsigned f = 0; f < H; ++f) want[static_cast<std::size_t>(r) * H + f] = to_bf(q8_value(table.data(), ids[r], H, f));
        same(std::string("token_embedding ") + (m ? "real token_embd" : "synthetic"), want, out.get(), false, H);
    }
    // specials for BF16 elementwise inputs
    auto specials = [](std::vector<std::uint16_t>& v) {
        const std::uint16_t sp[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x3f80, 0xbf80, 0x0080};
        for (std::size_t i = 0; i < v.size(); i += 97) v[i] = sp[(i / 97) % (sizeof(sp) / 2)];
    };
    auto hyper = synth_bf16(static_cast<std::size_t>(rows) * W, 1.0F), hidden = synth_bf16(static_cast<std::size_t>(rows) * H, 2.0F),
         routed = synth_bf16(static_cast<std::size_t>(rows) * H, 1.0F);
    specials(hyper); specials(hidden); specials(routed);
    std::vector<float> inj(static_cast<std::size_t>(rows) * S);
    for (auto& v : inj) v = 2.0F / (1.0F + std::exp(-nrand(1.0F)));
    inj[0] = 0.0F; inj[1] = -0.0F; inj[2] = 1e38F; inj[5] = 1e-40F;
    Dev<std::uint16_t> dh(hyper), dd(hidden), dr(routed), o1(hyper.size()), o2(hyper.size()), o3(hidden.size());
    Dev<float> dinj(inj);
    hc::hyper_expand(dd.p, rows, o1.p, nullptr);
    hc::inject(dh.p, dd.p, dinj.p, rows, o2.p, nullptr);
    hc::combine(dd.p, dr.p, rows, o3.p, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<std::uint16_t> we(hyper.size()), wi(hyper.size()), wc(hidden.size());
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned s = 0; s < S; ++s)
            for (unsigned f = 0; f < H; ++f) {
                const std::size_t i = static_cast<std::size_t>(r) * W + s * H + f, j = static_cast<std::size_t>(r) * H + f;
                we[i] = to_bf(bf(hidden[j]));
                wi[i] = to_bf(std::fma(bf(hidden[j]), inj[static_cast<std::size_t>(r) * S + s], bf(hyper[i])));
            }
    for (std::size_t j = 0; j < hidden.size(); ++j) wc[j] = to_bf(bf(hidden[j]) + bf(routed[j]));
    same("hyper_expand (with special bf16 values)", we, o1.get(), false, W);
    same("inject (contracted fma, special values, inf*0 -> canonical NaN)", wi, o2.get(), false, W);
    same("combine (special values)", wc, o3.get(), false, H);
}

static void test_lm_head(const strix::qwen_gguf::Model* m, bool perf) {
    std::vector<std::uint8_t> wt;
    std::size_t vocab;
    if (m) { wt = m->bytes("output.weight"); vocab = hc::kVocab; }
    else { vocab = 8192; wt = synth_q8(vocab, H, 1e-3F, 3e-2F); }
    auto x = synth_bf16(H, 1.0F);
    Dev<std::uint8_t> dw(wt); Dev<std::uint16_t> dx(x); Dev<float> dl(vocab);
    hc::lm_head(dx.p, dw.p, static_cast<std::uint32_t>(vocab), dl.p, nullptr);
    CK(cudaDeviceSynchronize());
    if (perf) {
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        for (int i = 0; i < 10; ++i) hc::lm_head(dx.p, dw.p, static_cast<std::uint32_t>(vocab), dl.p, nullptr);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
        std::printf("perf: lm head one row over %zu vocab rows: %.3f ms\n", vocab, ms / 10);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    const auto got = dl.get();
    std::vector<float> want(vocab);
    for (std::size_t v = 0; v < vocab; ++v) want[v] = ref_lm_row(x.data(), wt.data(), v);
    same(std::string("lm_head ") + (m ? "real output.weight" : "synthetic") + " all logits", want, got, false);
    std::uint32_t ref_best = 0;
    for (std::size_t v = 1; v < vocab; ++v) if (want[v] > want[ref_best]) ref_best = static_cast<std::uint32_t>(v);
    report("lm_head first_max_token", hc::first_max_token(got.data(), got.size()) == ref_best, "token " + std::to_string(ref_best));
}
static void test_first_max() {
    std::vector<float> v = {1.0F, 3.0F, -2.0F, 3.0F, 0.5F};
    bool ok = hc::first_max_token(v.data(), v.size()) == 1U;
    v = {-0.0F, 0.0F}; ok = ok && hc::first_max_token(v.data(), v.size()) == 0U;
    v = {-5.0F, -7.0F}; ok = ok && hc::first_max_token(v.data(), v.size()) == 0U;
    auto throws = [](std::vector<float> x) { try { (void)hc::first_max_token(x.data(), x.size()); return false; } catch (const std::invalid_argument&) { return true; } };
    ok = ok && throws({1.0F, NAN}) && throws({INFINITY}) && throws({-INFINITY, 0.0F}) && throws({});
    report("first_max_token (strict >, lowest index, throws on non-finite/empty)", ok);
}

// ---------------------------------------------------------------------------------------------------------------------
// Fixture replay (STRIX_QRTX_STEP_DUMP of a HIP tiled block).
static void replay(const std::string& path, const strix::qwen_gguf::Model& m, const am::Tables& t) {
    namespace qf = strix::qwen_fixture;
    const qf::Dump dump(path);
    std::map<std::uint32_t, const qf::Record*> steps;  // step -> first record
    for (const auto& r : dump.records()) if (!steps.count(r.step)) steps[r.step] = &r;
    std::printf("fixture %s: %zu records, %zu steps\n", path.c_str(), dump.records().size(), steps.size());
    std::vector<std::uint8_t> token_embd;
    std::map<std::string, unsigned> done;
    for (const auto& [step, rec] : steps) {
        const unsigned op = rec->op, layer = rec->layer;
        const auto rd = [&](unsigned i) { return dump.load_rows(step, true, i); };
        const auto wr = [&](unsigned i) { return dump.load_rows(step, false, i); };
        const std::string where = "step " + std::to_string(step) + " op " + std::to_string(op) +
            (layer == UINT32_MAX ? std::string() : " layer " + std::to_string(layer));
        try {
            if (op == 4 || op == 8 || op == 14) {  // attn_hyper_mix, mlp_hyper_mix, final_hyper_mix
                const auto in = rd(0), out = wr(0);
                const unsigned rows = static_cast<unsigned>(in.size() / (W * 2));
                if (!rows || in.size() != rows * W * 2ULL || out.size() != rows * H * 2ULL) { std::printf("skip %s: sizes %zu/%zu\n", where.c_str(), in.size(), out.size()); continue; }
                HcWeights w;
                const std::string base = op == 14 ? "output_hc_" : "blk." + std::to_string(layer) + (op == 4 ? ".hc_attn_" : ".hc_ffn_");
                w.down = m.bytes(base + "down.weight"); w.up = m.bytes(base + "up.weight"); w.norm = f32_vector<float>(m.bytes(base + "norm.weight"));
                if (op != 14) w.inject = f32_vector<float>(m.bytes(base + "inject.weight"));
                const HcRun got = run_hc(w, f32_vector<std::uint16_t>(in), rows, 1e-6F, t);
                same("fixture " + where + " hc output", f32_vector<std::uint16_t>(out), got.output, false, H);
                if (op != 14) {
                    const auto inj = wr(1);
                    if (inj.size() == rows * S * 4ULL) same("fixture " + where + " hc injection", f32_vector<float>(inj), got.injection, false, S);
                    else std::printf("note %s: injection write %zu bytes\n", where.c_str(), inj.size());
                }
            } else if (op == 9) {  // moe_route
                const auto in = rd(0), ids = wr(0), wts = wr(1);
                const unsigned rows = static_cast<unsigned>(in.size() / (H * 2));
                if (ids.size() != rows * 40ULL || wts.size() != rows * 40ULL) { std::printf("skip %s: writes %zu/%zu\n", where.c_str(), ids.size(), wts.size()); continue; }
                const auto wt = f32_vector<float>(m.bytes("blk." + std::to_string(layer) + ".ffn_gate_inp.weight"));
                Dev<std::uint16_t> dx(f32_vector<std::uint16_t>(in)); Dev<float> dw(wt), dl(static_cast<std::size_t>(rows) * hc::kExperts);
                Dev<std::uint32_t> di(static_cast<std::size_t>(rows) * 10); Dev<float> dwt(static_cast<std::size_t>(rows) * 10);
                hc::route_logits(dx.p, dw.p, rows, dl.p, nullptr);
                hc::route_select(dl.p, rows, 10, true, t, di.p, dwt.p, nullptr);
                CK(cudaDeviceSynchronize());
                same("fixture " + where + " router ids", f32_vector<std::uint32_t>(ids), di.get(), false, 10);
                same("fixture " + where + " router weights", f32_vector<float>(wts), dwt.get(), false, 10);
            } else if (op == 7 || op == 13) {  // attn_inject, mlp_inject
                const auto hy = rd(0), hid = rd(1), inj = rd(2), out = wr(0);
                const unsigned rows = static_cast<unsigned>(hy.size() / (W * 2));
                if (hid.size() != rows * H * 2ULL || inj.size() != rows * S * 4ULL || out.size() != hy.size()) { std::printf("skip %s\n", where.c_str()); continue; }
                Dev<std::uint16_t> a(f32_vector<std::uint16_t>(hy)), b(f32_vector<std::uint16_t>(hid)), o(static_cast<std::size_t>(rows) * W);
                Dev<float> c(f32_vector<float>(inj));
                hc::inject(a.p, b.p, c.p, rows, o.p, nullptr);
                CK(cudaDeviceSynchronize());
                same("fixture " + where + " inject", f32_vector<std::uint16_t>(out), o.get(), false, W);
            } else if (op == 12) {  // moe_combine
                const auto sh = rd(0), ro = rd(1), out = wr(0);
                const unsigned rows = static_cast<unsigned>(sh.size() / (H * 2));
                if (ro.size() != sh.size() || out.size() != sh.size()) { std::printf("skip %s\n", where.c_str()); continue; }
                Dev<std::uint16_t> a(f32_vector<std::uint16_t>(sh)), b(f32_vector<std::uint16_t>(ro)), o(static_cast<std::size_t>(rows) * H);
                hc::combine(a.p, b.p, rows, o.p, nullptr);
                CK(cudaDeviceSynchronize());
                same("fixture " + where + " combine", f32_vector<std::uint16_t>(out), o.get(), false, H);
            } else if (op == 2) {  // hyper_expand
                const auto in = rd(0), out = wr(0);
                const unsigned rows = static_cast<unsigned>(in.size() / (H * 2));
                if (out.size() != rows * W * 2ULL) { std::printf("skip %s\n", where.c_str()); continue; }
                Dev<std::uint16_t> a(f32_vector<std::uint16_t>(in)), o(static_cast<std::size_t>(rows) * W);
                hc::hyper_expand(a.p, rows, o.p, nullptr);
                CK(cudaDeviceSynchronize());
                same("fixture " + where + " hyper_expand", f32_vector<std::uint16_t>(out), o.get(), false, W);
            } else if (op == 0) {  // token_embedding: the record carries no token id; recover it from the row's first values
                const auto out = f32_vector<std::uint16_t>(wr(0));
                const unsigned rows = static_cast<unsigned>(out.size() / H);
                if (token_embd.empty()) token_embd = m.bytes("token_embd.weight");
                std::vector<std::uint32_t> ids(rows, 0);
                for (unsigned r = 0; r < rows; ++r) {
                    long found = -1;
                    for (std::size_t v = 0; v < hc::kVocab && found < 0; ++v) {
                        bool ok = true;
                        for (unsigned f = 0; f < 64 && ok; ++f) ok = to_bf(q8_value(token_embd.data(), v, H, f)) == out[static_cast<std::size_t>(r) * H + f];
                        if (ok) found = static_cast<long>(v);
                    }
                    ids[r] = found < 0 ? 0U : static_cast<std::uint32_t>(found);
                }
                Dev<std::uint8_t> dt(token_embd); Dev<std::uint32_t> di(ids); Dev<std::uint16_t> o(out.size());
                hc::token_embedding(dt.p, di.p, rows, o.p, nullptr);
                CK(cudaDeviceSynchronize());
                same("fixture " + where + " token_embedding (ids recovered from 64 features)", out, o.get(), false, H);
            } else if (op == 15) {  // lm_head: terminal-row policy, only the block's last row runs the head (the other rows'
                                     // logits buffers are not written: known_prefix_omitted_lm_heads = rows - 1)
                std::vector<std::uint8_t> head;
                for (std::uint32_t r = rec->rows ? rec->rows - 1U : 0U;; ++r) {
                    const qf::Record* in = dump.find(step, true, 0, r);
                    const qf::Record* out = dump.find(step, false, 0, r);
                    if (!in) break;
                    if (!out || out->bytes != hc::kVocab * 4ULL) continue;
                    if (head.empty()) head = m.bytes("output.weight");
                    Dev<std::uint8_t> dw(head); Dev<std::uint16_t> dx(f32_vector<std::uint16_t>(dump.load(*in))); Dev<float> dl(hc::kVocab);
                    hc::lm_head(dx.p, dw.p, hc::kVocab, dl.p, nullptr);
                    CK(cudaDeviceSynchronize());
                    const auto want = f32_vector<float>(dump.load(*out));
                    const auto got = dl.get();
                    same("fixture " + where + " lm_head row " + std::to_string(r), want, got, false);
                    const std::uint32_t tw = hc::first_max_token(want.data(), want.size()), tg = hc::first_max_token(got.data(), got.size());
                    report("fixture " + where + " first token row " + std::to_string(r), tw == tg,
                        "HIP " + std::to_string(tw) + ", CUDA " + std::to_string(tg));
                }
            } else continue;
            ++done[std::to_string(op)];
        } catch (const std::exception& e) {
            report("fixture " + where, false, e.what());
        }
    }
    for (const auto& [op, n] : done) std::printf("fixture: op %s replayed in %u steps\n", op.c_str(), n);
}

// ---------------------------------------------------------------------------------------------------------------------
// FAST form (HcMixArgs::mma): tensor-core projection + output vs the exact kernels on the same packed weights and inputs.
struct Err {
    double d2 = 0, r2 = 0, mx = 0;
    void add(double ref, double got) { const double d = got - ref; d2 += d * d; r2 += ref * ref; mx = std::max(mx, std::fabs(d)); }
    double rel() const { return r2 > 0 ? std::sqrt(d2 / r2) : 0.0; }
};
static void report_err(const std::string& name, const Err& all, const Err* cls, const char* const* names, unsigned n, double tol) {
    double lo = 1e30, hi = 0; std::string detail;
    char buf[160];
    std::snprintf(buf, sizeof buf, "max abs %.3e rel L2 %.3e;", all.mx, all.rel()); detail = buf;
    for (unsigned i = 0; i < n; ++i) {
        if (cls[i].r2 <= 0) continue;
        lo = std::min(lo, cls[i].rel()); hi = std::max(hi, cls[i].rel());
        std::snprintf(buf, sizeof buf, " %s %.2e", names[i], cls[i].rel()); detail += buf;
    }
    report(name, all.rel() < tol && hi <= 3.0 * std::max(all.rel(), 1e-9), detail);
}
// Exercise the actual FAST output kernels with identical low-rank input, bypassing norm/projection.
static void test_hc_mma_split(const am::Tables& t) {
    constexpr unsigned rows = 65;  // Includes a partial output tile.
    auto wu = synth_q8(W, RK, 2e-4F, 2e-3F);
    const std::size_t stream_bytes = wu.size() / S;
    for (unsigned s = 1; s < S; ++s)
        std::copy_n(wu.data(), stream_bytes, wu.data() + s * stream_bytes);
    Dev<std::uint8_t> down(synth_q8(RK, W, 2e-4F, 2e-3F)), up(wu);
    Dev<std::uint8_t> packed(hc::hc_packed_bytes());
    std::vector<float> n(static_cast<std::size_t>(rows) * P, 0.0F);
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned k = 0; k < RK; ++k)
            n[static_cast<std::size_t>(r) * P + W + k] = r == 0 ? 0.0F : nrand(2.0F);
        for (unsigned f = 0; f < H; ++f) {
            // Row 0: zero gates make sigmoid exactly 1/2. Probe a BF16 tie and
            // its adjacent FP32 values. Other rows cancel equal, nonzero gates.
            const float midpoint = 1.00390625F;
            const float boundary = f % 3U == 0 ? std::nextafter(midpoint, 0.0F) :
                (f % 3U == 1 ? midpoint : std::nextafter(midpoint, 2.0F));
            const std::size_t i = static_cast<std::size_t>(r) * P + f;
            n[i] = r == 0 ? 0.0F : 4096.0F;
            n[i + H] = 8.0F * boundary;
            n[i + 2U * H] = -n[i];
        }
    }
    Dev<float> normalized(n), gates(static_cast<std::size_t>(rows) * W);
    Dev<std::uint16_t> output(static_cast<std::size_t>(rows) * H);
    hc::HcMixArgs a;
    a.rows = rows; a.mma = true; a.normalized = normalized.p; a.output = output.p;
    a.packed = hc::hc_packed_view(packed.p);
    hc::hc_pack(down.p, up.p, a.packed, nullptr);
    hc::hc_mix(a, t, nullptr, 4U);
    CK(cudaDeviceSynchronize());
    const auto fused = output.get();
    CK(cudaMemset(output.p, 0xff, output.n * sizeof(std::uint16_t)));
    a.gates = gates.p;
    hc::hc_mix(a, t, nullptr, 12U);
    CK(cudaDeviceSynchronize());
    same("hc FAST fused == split: finite cancellation and BF16 boundaries", fused, output.get(), false, H);
    bool boundary_ok = true, finite = true;
    for (unsigned f = 0; f < H; ++f)
        boundary_ok = boundary_ok && fused[f] == (f % 3U == 2 ? 0x3f81U : 0x3f80U);
    for (auto v : fused) finite = finite && (v & 0x7f80U) != 0x7f80U;
    report("hc FAST split fixture covers BF16 tie and neighbors", boundary_ok);
    report("hc FAST cancellation outputs remain finite", finite);
}
static void test_hc_mma(const am::Tables& t, const char* tag, float down_lo, float down_hi, bool with_inject, bool outliers) {
    HcWeights w;
    w.down = synth_q8(RK, W, down_lo, down_hi); w.up = synth_q8(W, RK, 2e-4F, 2e-3F);
    w.norm.resize(W); for (auto& v : w.norm) v = 1.0F + nrand(0.2F);
    if (with_inject) { w.inject.resize(static_cast<std::size_t>(S) * W); for (auto& v : w.inject) v = nrand(0.02F); }
    Dev<std::uint8_t> down(w.down), up(w.up);
    Dev<float> norm(w.norm), inj(w.inject.empty() ? std::vector<float>{} : w.inject);
    Dev<std::uint8_t> packed(hc::hc_packed_bytes());
    hc::HcMixArgs a;
    a.packed = hc::hc_packed_view(packed.p); a.norm = norm.p; a.inject = w.inject.empty() ? nullptr : inj.p; a.epsilon = 1e-6F;
    hc::hc_pack(down.p, up.p, a.packed, nullptr);
    const unsigned list[] = {1, 15, 16, 17, 65, 1000, 8192};
    for (unsigned rows : list) {
        auto input = synth_bf16(static_cast<std::size_t>(rows) * W, 1.0F);
        if (outliers) for (std::size_t i = 0; i < input.size(); i += 997) input[i] = to_bf(nrand(40.0F));
        Dev<std::uint16_t> in(input);
        Dev<float> normalized(static_cast<std::size_t>(rows) * P), injection(static_cast<std::size_t>(rows) * S);
        Dev<std::uint16_t> out(static_cast<std::size_t>(rows) * H);
        a.input = in.p; a.rows = rows; a.normalized = normalized.p; a.injection = injection.p; a.output = out.p;
        a.mma = false; hc::hc_mix(a, t, nullptr); CK(cudaDeviceSynchronize());
        const auto ne = normalized.get(); const auto ie = injection.get(); const auto oe = out.get();
        CK(cudaMemset(normalized.p, 0xff, normalized.n * 4)); CK(cudaMemset(injection.p, 0xff, injection.n * 4)); CK(cudaMemset(out.p, 0xff, out.n * 2));
        a.mma = true; hc::hc_mix(a, t, nullptr); CK(cudaDeviceSynchronize());
        const auto nm = normalized.get(); const auto im = injection.get(); const auto om = out.get();
        // Keep the independent full-workspace references above. Compact owns only
        // inverses and low values; compare its low/injection/output for both forms,
        // including separate producer/consumer calls.
        if (rows <= 65U) for (bool mma : {false, true}) {
            a.mma = mma;
            a.workspace_layout = hc::HcWorkspaceLayout::compact;
            for (bool staged : {false, true}) {
                CK(cudaMemset(normalized.p, 0xff, normalized.n * sizeof(float)));
                if (staged) for (unsigned stage : {1U, 2U, 4U}) hc::hc_mix(a, t, nullptr, stage);
                else hc::hc_mix(a, t, nullptr);
                CK(cudaDeviceSynchronize());
                const auto compact = normalized.get();
                std::vector<float> expected_low, compact_low;
                const auto& full = mma ? nm : ne;
                for (unsigned row = 0; row < rows; ++row) for (unsigned k = 0; k < RK; ++k) {
                    const auto at = static_cast<std::size_t>(row) * P + W + k;
                    expected_low.push_back(full[at]); compact_low.push_back(compact[at]);
                }
                const std::string label = std::string("compact ") + (mma ? "mma" : "exact") +
                    (staged ? " staged" : " complete") + " rows=" + std::to_string(rows);
                same(label + " low", expected_low, compact_low, false, RK);
                if (with_inject) same(label + " injection", mma ? im : ie, injection.get(), false, S);
                same(label + " output", mma ? om : oe, out.get(), false, H);
            }
        }
        // A valid scalar input view need not satisfy compact vector alignment.
        if (rows == 16U) {
            auto shifted_input = input; shifted_input.insert(shifted_input.begin(), 0);
            auto shifted_norm = w.norm; shifted_norm.insert(shifted_norm.begin(), 0.0F);
            Dev<std::uint16_t> unaligned_input(shifted_input);
            Dev<float> unaligned_norm(shifted_norm);
            a.input = unaligned_input.p + 1; a.norm = unaligned_norm.p + 1;
            a.workspace_layout = hc::HcWorkspaceLayout::automatic;
            for (bool mma : {false, true}) {
                a.mma = mma; hc::hc_mix(a, t, nullptr); CK(cudaDeviceSynchronize());
                same("unaligned full workspace", mma ? nm : ne, normalized.get(), false, P);
                same("unaligned output", mma ? om : oe, out.get(), false, H);
            }
            a.input = in.p; a.norm = norm.p;
        }
        a.workspace_layout = hc::HcWorkspaceLayout::full;
        const unsigned tail = rows % 64U ? rows - rows % 64U : (rows > 64U ? rows - 64U : 0U);
        Err low, lowc[3], lowk[2], io, ioc[3], ou, ouc[3], ouf[3];
        const auto rc = [&](unsigned r) { return r < 16U ? 0U : (r >= tail ? 2U : 1U); };
        for (unsigned r = 0; r < rows; ++r) {
            for (unsigned k = 0; k < RK; ++k) {
                const double e = ne[static_cast<std::size_t>(r) * P + W + k], g = nm[static_cast<std::size_t>(r) * P + W + k];
                low.add(e, g); lowc[rc(r)].add(e, g); lowk[k >= RK - 32U ? 1U : 0U].add(e, g);
            }
            if (with_inject) for (unsigned s = 0; s < S; ++s) { const double e = ie[r * S + s], g = im[r * S + s]; io.add(e, g); ioc[rc(r)].add(e, g); }
            for (unsigned f = 0; f < H; ++f) {
                const double e = bf(oe[static_cast<std::size_t>(r) * H + f]), g = bf(om[static_cast<std::size_t>(r) * H + f]);
                ou.add(e, g); ouc[rc(r)].add(e, g); ouf[f < 32U ? 0U : (f >= H - 32U ? 2U : 1U)].add(e, g);
            }
        }
        bool nan_hit = false;  // the poison fill must be gone (every output written)
        for (std::size_t i = 0; i < om.size(); ++i) if (om[i] == 0xffffU) { nan_hit = true; break; }
        const std::string base = std::string("hc_mma ") + tag + " rows=" + std::to_string(rows) + " ";
        const char* rn[3] = {"rows<16", "mid", "lasttile"}; const char* kn[2] = {"k<288", "klast32"}; const char* fn[3] = {"f<32", "mid", "flast32"};
        report_err(base + "low", low, lowc, rn, 3, 2e-2);
        report_err(base + "low (by output block)", low, lowk, kn, 2, 2e-2);
        if (with_inject) report_err(base + "injection", io, ioc, rn, 3, 2e-2);
        report_err(base + "output bf16", ou, ouc, rn, 3, 2e-2);
        report_err(base + "output bf16 (by feature)", ou, ouf, fn, 3, 2e-2);
        report(base + "all outputs written", !nan_hit);
    }
}
static void time_hc_mma(const am::Tables& t) {
    HcWeights w = synth_hc(true);
    Dev<std::uint8_t> down(w.down), up(w.up);
    Dev<float> norm(w.norm), inj(w.inject);
    Dev<std::uint8_t> packed(hc::hc_packed_bytes());
    hc::HcMixArgs a;
    a.packed = hc::hc_packed_view(packed.p); a.norm = norm.p; a.inject = inj.p; a.epsilon = 1e-6F;
    hc::hc_pack(down.p, up.p, a.packed, nullptr);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (unsigned rows : {8192U, 1024U}) {
        Dev<std::uint16_t> in(synth_bf16(static_cast<std::size_t>(rows) * W, 1.0F));
        Dev<float> normalized(static_cast<std::size_t>(rows) * P), injection(static_cast<std::size_t>(rows) * S);
        Dev<std::uint16_t> out(static_cast<std::size_t>(rows) * H);
        a.input = in.p; a.rows = rows; a.normalized = normalized.p; a.injection = injection.p; a.output = out.p;
        Dev<float> gates_buf(static_cast<std::size_t>(rows) * S * H);
        for (int mode = 0; mode < 2; ++mode) {
            a.mma = mode == 1;
            for (unsigned stages : {1U, 2U, 4U, 4U | 0x100U}) {
                a.gates = stages & 0x100U ? gates_buf.p : nullptr; stages &= 15U;
                for (int i = 0; i < 3; ++i) hc::hc_mix(a, t, nullptr, stages);
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(e0));
                for (int i = 0; i < 20; ++i) hc::hc_mix(a, t, nullptr, stages);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
                std::printf("perf: rows=%u %s %s %.3f ms\n", rows, a.mma ? "mma  " : "exact", stages == 1U ? "norm      " : stages == 2U ? "projection" : a.gates ? "out(gates)" : "output    ", ms / 20);
            }
        }
    }
}

int main(int argc, char** argv) {
    std::string tables = "qrtx-build\\tables", fixture;
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--mma") {
        for (int j = 1; j + 1 < argc; ++j) if (std::string(argv[j]) == "--tables") tables = argv[j + 1];
        try {
            load_host_math(tables);
            am::Tables t = am::load_tables(tables);
            test_hc_mma_split(t);
            test_hc_mma(t, "mild", 2e-4F, 2e-3F, true, false);
            test_hc_mma(t, "mild+outliers", 2e-4F, 2e-3F, true, true);
            test_hc_mma(t, "saturating", 2e-3F, 2e-2F, false, false);
            time_hc_mma(t);
            am::free_tables(t);
        } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); ++g_fail; }
        std::printf("qwen_hc_tests --mma: %d passed, %d failed\n", g_pass, g_fail);
        return g_fail ? 1 : 0;
    }
    bool real = true, perf = true, only_fixture = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--tables" && i + 1 < argc) tables = argv[++i];
        else if (a == "--fixture" && i + 1 < argc) fixture = argv[++i];
        else if (a == "--skip-real") real = false;
        else if (a == "--skip-perf") perf = false;
        else if (a == "--fixture-only") only_fixture = true;
    }
    if (fixture.empty() && std::filesystem::exists("qrtx-build\\fx\\d64.dump")) fixture = "qrtx-build\\fx\\d64.dump";
    try {
        load_host_math(tables);
        am::Tables t = am::load_tables(tables);
        std::unique_ptr<strix::qwen_gguf::Model> model;
        if (real || only_fixture) model = std::make_unique<strix::qwen_gguf::Model>();
        if (!only_fixture) {
            test_first_max();
            test_hc_synthetic(t);
            if (model) test_hc_real(*model, t);
            test_router(model.get(), t, perf);
            test_elementwise(model.get());
            test_lm_head(model.get(), perf);
            if (perf) test_hc_perf(model.get(), t);
        }
        if (!fixture.empty() && model) replay(fixture, *model, t);
        am::free_tables(t);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        ++g_fail;
    }
    std::printf("qwen_hc_tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
