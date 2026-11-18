// Bit-exact tests of src/cuda/models/qwen/qwen_qsa.cu (QSA layer core) against literal host references of the HIP kernels
// (src/hip_indexed_attention_launcher.hip). Usage: qwen_qsa_tests [--no-real] [--bench]
//   tables: qrtx-build\tables (STRIX_QRTX_TABLES); RoPE table: qrtx-build\fx\rope_cs_262144.bin when
//   present (else a synthetic cos/sin table: the kernels only consume it); GGUF via qwen_gguf_host.hpp.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "qwen_fixture.hpp"
#include "qwen_gguf_host.hpp"
#include "strix/cuda/qwen_qsa.hpp"
#include "strix/cuda/amd_math_tables.hpp"

namespace q = strix::cuda::qwen_qsa;
namespace am = strix::cuda::amdmath;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

static int g_pass = 0, g_fail = 0;

// ---------------------------------------------------------------- host emulation of the gfx1151 expf / rsqrtf
namespace hm {
std::vector<std::uint32_t> exp_bits, rsq_bits;
std::map<std::uint32_t, std::uint32_t> exp_hard, rsq_hard;
static std::vector<std::uint32_t> read_u32(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "missing %s\n", p.c_str()); std::exit(2); }
    _fseeki64(f, 0, SEEK_END); const auto n = _ftelli64(f); _fseeki64(f, 0, SEEK_SET);
    std::vector<std::uint32_t> v(static_cast<size_t>(n) / 4);
    if (std::fread(v.data(), 4, v.size(), f) != v.size()) std::exit(2);
    std::fclose(f);
    return v;
}
static void load(const std::string& dir) {
    exp_bits = read_u32(dir + "/tbl_exp_pm.bin");
    rsq_bits = read_u32(dir + "/tbl_rsq_1_4.bin");
    auto e = read_u32(dir + "/hard_v_exp_f32.bin");
    for (size_t i = 0, n = e.size() / 2; i < n; ++i) {
        const auto mag = e[i] & 0x7fffffffU;
        if (mag >= (102U << 23) && mag < (127U << 23)) exp_hard[e[i]] = e[n + i];
    }
    auto r = read_u32(dir + "/hard_v_rsq_f32.bin");
    for (size_t i = 0, n = r.size() / 2; i < n; ++i) {
        std::uint32_t b = r[i], y = r[n + i];
        if (((b >> 23) & 255U) == 0U || ((b >> 23) & 255U) == 255U) continue;  // outside the emulated domain
        const int ex = static_cast<int>(b >> 23) - 127;
        const int k = ex >= 0 ? ex / 2 : -((-ex + 1) / 2);
        b = (b & 0x007fffffU) | (static_cast<std::uint32_t>(ex - 2 * k + 127) << 23);
        y = static_cast<std::uint32_t>(static_cast<int>(y) + (k << 23));
        rsq_hard[b] = y;
    }
}
static float u2f(std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static std::uint32_t f2u(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool bit(const std::vector<std::uint32_t>& t, std::uint32_t i) { return (t[i >> 5] >> (i & 31U)) & 1U; }
static float other(float f, double r) {
    const auto b = f2u(f);
    const bool up = r > static_cast<double>(f);
    const bool away = (b >> 31) ? !up : up;
    return u2f(away ? b + 1U : b - 1U);
}
static float v_exp(float x) {
    const auto b = f2u(x), mag = b & 0x7fffffffU;
    const double r = std::exp2(static_cast<double>(x));
    const float f = static_cast<float>(r);
    if (mag < (102U << 23)) return f;
    if (auto it = exp_hard.find(b); it != exp_hard.end()) return u2f(it->second);
    const std::uint32_t index = (b >> 31) * (25U << 23) + (mag - (102U << 23));  // as amd_math.cuh 95ac29be
    return bit(exp_bits, index) ? other(f, r) : f;
}
static float v_rsq(float x) {
    const auto b = f2u(x);
    const int e = static_cast<int>(b >> 23) - 127;
    const int k = e >= 0 ? e / 2 : -((-e + 1) / 2);
    const std::uint32_t cb = (b & 0x007fffffU) | (static_cast<std::uint32_t>(e - 2 * k + 127) << 23);
    float y;
    if (auto it = rsq_hard.find(cb); it != rsq_hard.end()) y = u2f(it->second);
    else {
        const double r = 1.0 / std::sqrt(static_cast<double>(u2f(cb)));
        const float f = static_cast<float>(r);
        y = bit(rsq_bits, cb - (127U << 23)) ? other(f, r) : f;
    }
    return u2f(static_cast<std::uint32_t>(static_cast<int>(f2u(y)) - (k << 23)));
}
static float expf_(float x) {
    const float kLog2e = u2f(0x3fb8aa3bU), kLog2eLo = u2f(0x32a5705fU);
    const float ph = x * kLog2e;
    const float e = std::nearbyint(ph);
    float pl = std::fma(x, kLog2e, -ph);
    pl = std::fma(x, kLog2eLo, pl);
    const float a = (ph - e) + pl;
    float y;
    if (std::fabs(a) < 1.0F) y = std::scalbn(v_exp(a), static_cast<int>(e));
    else y = a != a ? a : 0.0F;
    if (u2f(0xc2ce8ed0U) > x) y = 0.0F;
    if (u2f(0x42b17218U) < x) y = u2f(0x7f800000U);
    return y;
}
static float rsqrtf_(float x) {
    const bool small = u2f(0x00800000U) > x;
    const float s = small ? x * u2f(0x4b800000U) : x;
    float y;
    const auto sb = f2u(s);
    if (sb == 0U) y = u2f(0x7f800000U);
    else if (sb == 0x80000000U) y = u2f(0xff800000U);
    else if (sb >= 0x7f800000U) y = sb == 0x7f800000U ? 0.0F : u2f(0x7fc00000U);
    else y = v_rsq(s);
    return small ? y * u2f(0x45800000U) : y;
}
}  // namespace hm

// ---------------------------------------------------------------- helpers
static float bf(std::uint16_t b) { return hm::u2f(static_cast<std::uint32_t>(b) << 16); }
static std::uint16_t tobf(float v) { const auto b = hm::f2u(v); return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16) & 1U)) >> 16); }
static float rbf(float v) { return bf(tobf(v)); }

template <class T> struct Dev {
    T* p = nullptr; size_t n = 0;
    explicit Dev(size_t count) : n(count) { CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T))); CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T))); }
    Dev(const std::vector<T>& v) : Dev(v.size()) { put(v); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    void put(const std::vector<T>& v) { CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> v(n); CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
};

template <class T> static bool same(const char* what, const std::vector<T>& got, const std::vector<T>& want, size_t count = SIZE_MAX) {
    count = std::min({count, got.size(), want.size()});
    size_t diff = 0, first = SIZE_MAX;
    for (size_t i = 0; i < count; ++i)
        if (std::memcmp(&got[i], &want[i], sizeof(T)) != 0) { if (first == SIZE_MAX) first = i; ++diff; }
    if (diff) {
        ++g_fail;
        if constexpr (sizeof(T) == 4 && !std::is_integral_v<T>)
            std::printf("FAIL %s: %zu/%zu differ, first [%zu] got %.9g (0x%08x) want %.9g (0x%08x)\n", what, diff, count, first,
                static_cast<double>(got[first]), hm::f2u(static_cast<float>(got[first])), static_cast<double>(want[first]),
                hm::f2u(static_cast<float>(want[first])));
        else std::printf("FAIL %s: %zu/%zu differ, first [%zu] got %llu want %llu\n", what, diff, count, first,
            static_cast<unsigned long long>(got[first]), static_cast<unsigned long long>(want[first]));
        return false;
    }
    ++g_pass;
    std::printf("ok   %s (%zu values)\n", what, count);
    return true;
}

static std::mt19937 rng(12345);
static std::vector<std::uint16_t> rand_bf16(size_t n, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) x = tobf(d(rng));
    return v;
}
static std::vector<float> rand_bf16f(size_t n, float scale) {
    auto b = rand_bf16(n, scale);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = bf(b[i]);
    return v;
}

// RoPE table: the node file when present, else synthetic from the formula in double (only consumed by the kernels).
static std::vector<float2> g_cs;
static bool g_cs_real = false;
static void load_cs(size_t positions) {
    const char* path = "qrtx-build\\fx\\rope_cs_262144.bin";
    if (FILE* f = std::fopen(path, "rb")) {
        g_cs.resize(262144ULL * 32U);
        g_cs_real = std::fread(g_cs.data(), sizeof(float2), g_cs.size(), f) == g_cs.size();
        std::fclose(f);
        if (g_cs_real) { std::printf("rope table: %s\n", path); return; }
    }
    g_cs.resize(positions * 32U);
    for (size_t p = 0; p < positions; ++p)
        for (unsigned i = 0; i < 32U; ++i) {
            const float ex = 2.0F * static_cast<float>(i) / 64.0F;
            const float ang = static_cast<float>(p) * std::pow(1e7F, -ex);
            g_cs[p * 32U + i] = make_float2(std::cos(ang), std::sin(ang));
        }
    std::printf("rope table: synthetic (%zu positions)\n", positions);
}

// ---------------------------------------------------------------- literal host references
// linear_to_float_rows_kernel<bf16,16> (H:3566-3591)
static void ref_indexer(const std::vector<std::uint16_t>& hidden, unsigned rows, const std::vector<std::uint16_t>& wq,
    const std::vector<std::uint16_t>& wk, std::vector<float>& iq, std::vector<float>& ik) {
    iq.assign(static_cast<size_t>(rows) * 512U, 0.0F); ik.assign(static_cast<size_t>(rows) * 128U, 0.0F);
    float s[256];
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned o = 0; o < 640U; ++o) {
            const std::uint16_t* w = o < 512U ? &wq[static_cast<size_t>(o) * 2560U] : &wk[static_cast<size_t>(o - 512U) * 2560U];
            for (unsigned t = 0; t < 256U; ++t) {
                float sum = 0.0F;
                for (unsigned f = t; f < 2560U; f += 256U) sum = std::fma(bf(hidden[static_cast<size_t>(r) * 2560U + f]), bf(w[f]), sum);
                s[t] = sum;
            }
            for (unsigned stride = 128U; stride > 0; stride >>= 1U) for (unsigned t = 0; t < stride; ++t) s[t] = s[t] + s[t + stride];
            if (o < 512U) iq[static_cast<size_t>(r) * 512U + o] = rbf(s[0]); else ik[static_cast<size_t>(r) * 128U + o - 512U] = rbf(s[0]);
        }
}
// norm_rope_body (H:865-996) with the ISA contraction a*cos then fma(+-b, sin, .)
static void ref_norm_rope(std::vector<float>& values, unsigned rows, size_t row_stride, unsigned heads, unsigned hd,
    unsigned head_stride, const std::vector<float>& w, std::uint64_t p0) {
    std::vector<float> src(hd), red(hd);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned h = 0; h < heads; ++h) {
            float* v = &values[r * row_stride + static_cast<size_t>(h) * head_stride];
            for (unsigned f = 0; f < hd; ++f) { src[f] = v[f]; red[f] = v[f] * v[f]; }
            for (unsigned stride = hd / 2U; stride > 0; stride >>= 1U) for (unsigned f = 0; f < stride; ++f) red[f] = red[f] + red[f + stride];
            const float mean = red[0] / static_cast<float>(hd) + 1e-6F;
            const float inv = hm::rsqrtf_(mean);
            for (unsigned f = 0; f < hd; ++f) src[f] = rbf(src[f] * inv * w[f]);
            for (unsigned f = 0; f < hd; ++f) {
                float res = src[f];
                if (f < 64U) {
                    const float2 c = g_cs[(p0 + r) * 32U + f % 32U];
                    const float ac = src[f] * c.x;
                    res = f < 32U ? std::fma(-src[f + 32U], c.y, ac) : std::fma(src[f - 32U], c.y, ac);
                }
                v[f] = rbf(res);
            }
        }
}
// block_score_rows_kernel (H:3693-3749): in-kernel pooling + LDS trees (the reference form of the pooled kernels).
static float ref_pool_feature_vec(const std::vector<std::uint16_t>& raw, std::uint32_t block, const std::vector<float>& kn, float* out) {
    float pooled[128], scratch[128];
    const std::uint64_t ts = static_cast<std::uint64_t>(block) * 4U;
    for (unsigned f = 0; f < 128U; ++f) {
        float mean = 0.0F;
        for (unsigned o = 0; o < 4U; ++o) mean += bf(raw[(ts + o) * 128U + f]);
        mean /= 4.0F;
        pooled[f] = rbf(mean);
        scratch[f] = pooled[f] * pooled[f];
    }
    for (unsigned stride = 64U; stride > 0; stride >>= 1U) for (unsigned f = 0; f < stride; ++f) scratch[f] = scratch[f] + scratch[f + stride];
    const float inv = hm::rsqrtf_(scratch[0] / 128.0F + 1e-6F);
    float y[128];
    for (unsigned f = 0; f < 128U; ++f) y[f] = rbf(pooled[f] * inv * kn[f]);
    for (unsigned f = 0; f < 128U; ++f) {
        float rot = y[f];
        if (f < 64U) {
            const float2 c = g_cs[ts * 32U + f % 32U];
            const float ac = y[f] * c.x;
            rot = f < 32U ? std::fma(-y[f + 32U], c.y, ac) : std::fma(y[f - 32U], c.y, ac);
        }
        out[f] = rbf(rot);
    }
    return 0.0F;
}
static float ref_block_score(const float* iq, const float* key) {
    float total = 0.0F, scratch[128];
    for (unsigned h = 0; h < 4U; ++h) {
        for (unsigned f = 0; f < 128U; ++f) scratch[f] = iq[h * 128U + f] * key[f];
        for (unsigned stride = 64U; stride > 0; stride >>= 1U) for (unsigned f = 0; f < stride; ++f) scratch[f] = scratch[f] + scratch[f + stride];
        total += std::fmax(scratch[0], 0.0F);
    }
    return total / std::sqrt(128.0F);
}
// select_blocks_body semantics (serial, H:1360-1440): repeatedly the lowest-index maximal score > -inf (NaN never),
// block 0 once none is left; ascending multiset; then the tail tokens. Selected scores (and block 0 on fallback) -> -inf.
static std::vector<std::uint32_t> ref_select(std::vector<float>& scores, std::uint32_t visible) {
    const std::uint32_t complete = visible / 4U;
    std::vector<std::uint32_t> sel;
    if (complete <= 512U) { for (std::uint32_t i = 0; i < visible; ++i) sel.push_back(i); return sel; }
    std::vector<std::uint32_t> blocks;
    for (unsigned s = 0; s < 512U; ++s) {
        std::int64_t best = -1;
        for (std::uint32_t b = 0; b < complete; ++b) {
            const float v = scores[b];
            if (!(v > -INFINITY)) continue;
            if (best < 0 || v > scores[static_cast<size_t>(best)]) best = b;
        }
        if (best < 0) { blocks.push_back(0); scores[0] = -INFINITY; }
        else { blocks.push_back(static_cast<std::uint32_t>(best)); scores[static_cast<size_t>(best)] = -INFINITY; }
    }
    std::sort(blocks.begin(), blocks.end());
    for (auto b : blocks) for (unsigned o = 0; o < 4U; ++o) sel.push_back(b * 4U + o);
    for (std::uint32_t t = complete * 4U; t < visible; ++t) sel.push_back(t);
    return sel;
}
// attention_score_body / softmax_body / output_body per row (H:1600-1680, 1680-1786, 1790-1850 forms; LDS trees).
static void ref_attention_row(const float* qg, const std::vector<std::uint32_t>& sel, const std::vector<std::uint16_t>& kv, float* out,
    std::vector<float>* scores_out = nullptr) {
    const unsigned count = static_cast<unsigned>(sel.size());
    std::vector<float> sc(static_cast<size_t>(24) * count);
    float scratch[256];
    for (unsigned h = 0; h < 24U; ++h) {
        const unsigned kvh = h / 12U;
        for (unsigned s = 0; s < count; ++s) {
            const std::uint16_t* k = &kv[static_cast<size_t>(sel[s]) * 1024U + kvh * 256U];
            for (unsigned f = 0; f < 256U; ++f) scratch[f] = qg[h * 512U + f] * bf(k[f]);
            for (unsigned stride = 128U; stride > 0; stride >>= 1U) for (unsigned f = 0; f < stride; ++f) scratch[f] = scratch[f] + scratch[f + stride];
            sc[static_cast<size_t>(h) * count + s] = scratch[0] / std::sqrt(256.0F);
        }
        float* row = &sc[static_cast<size_t>(h) * count];
        float lm[256];
        for (unsigned l = 0; l < 256U; ++l) { lm[l] = -INFINITY; for (unsigned i = l; i < count; i += 256U) lm[l] = std::fmax(lm[l], row[i]); }
        for (unsigned stride = 128U; stride > 0; stride >>= 1U) for (unsigned l = 0; l < stride; ++l) lm[l] = std::fmax(lm[l], lm[l + stride]);
        const float mx = lm[0];
        float ls[256];
        for (unsigned l = 0; l < 256U; ++l) { ls[l] = 0.0F; for (unsigned i = l; i < count; i += 256U) ls[l] += hm::expf_(row[i] - mx); }
        for (unsigned stride = 128U; stride > 0; stride >>= 1U) for (unsigned l = 0; l < stride; ++l) ls[l] = ls[l] + ls[l + stride];
        const float den = ls[0];
        for (unsigned i = 0; i < count; ++i) row[i] = rbf(hm::expf_(row[i] - mx) / den);
        for (unsigned f = 0; f < 256U; ++f) {
            float sum = 0.0F;
            for (unsigned s = 0; s < count; ++s) sum = std::fma(row[s], bf(kv[static_cast<size_t>(sel[s]) * 1024U + 512U + kvh * 256U + f]), sum);
            const float attn = rbf(sum);
            const float g = qg[h * 512U + 256U + f];
            const float gate = rbf(1.0F / (1.0F + hm::expf_(-g)));
            out[h * 256U + f] = rbf(attn * gate);
        }
    }
    if (scores_out) *scores_out = sc;
}

// ---------------------------------------------------------------- tests
static am::Tables g_tables;

static void test_indexer(const std::vector<std::uint16_t>& wq, const std::vector<std::uint16_t>& wk, unsigned rows, float hscale, const char* tag) {
    const auto hidden = rand_bf16(static_cast<size_t>(rows) * 2560U, hscale);
    std::vector<float> iq, ik;
    ref_indexer(hidden, rows, wq, wk, iq, ik);
    Dev<std::uint16_t> dh(hidden), dwq(wq), dwk(wk);
    Dev<float> diq(static_cast<size_t>(rows) * 512U), dik(static_cast<size_t>(rows) * 128U);
    q::LayerWeights w{dwq.p, dwk.p, nullptr, nullptr, nullptr, nullptr};
    q::indexer_projection(dh.p, rows, w, diq.p, dik.p, nullptr);
    CK(cudaDeviceSynchronize());
    same((std::string("indexer q ") + tag).c_str(), diq.get(), iq);
    same((std::string("indexer k ") + tag).c_str(), dik.get(), ik);
}

static void test_norm_rope(const std::vector<float>& qn, const std::vector<float>& kn, const std::vector<float>& iqn, std::uint64_t p0,
    unsigned rows, const char* tag) {
    Dev<float2> dcs(g_cs);
    struct Case { unsigned heads, hd, stride; size_t row_stride; const std::vector<float>* w; const char* name; };
    const Case cases[] = {{4, 128, 128, 512, &iqn, "index_q"}, {24, 256, 512, 12288, &qn, "q"}, {2, 256, 256, 512, &kn, "k"}};
    for (const auto& c : cases) {
        auto v = rand_bf16f(c.row_stride * rows, 1.5F);
        // edge rows: one head all zeros (rsqrt(eps)), one tiny head (subnormal-scale mean)
        for (unsigned f = 0; f < c.hd; ++f) v[f] = 0.0F;
        if (rows > 1) for (unsigned f = 0; f < c.hd; ++f) v[c.row_stride + f] = bf(tobf(v[c.row_stride + f] * 1e-21F));
        Dev<float> dv(v), dw(*c.w);
        ref_norm_rope(v, rows, c.row_stride, c.heads, c.hd, c.stride, *c.w, p0);
        q::norm_rope(dv.p, rows, c.row_stride, c.heads, c.hd, c.stride, dw.p, p0, dcs.p, g_tables, nullptr);
        CK(cudaDeviceSynchronize());
        same((std::string("norm_rope ") + c.name + " " + tag).c_str(), dv.get(), v);
    }
}

static void test_select() {
    // 64 rows at positions with 600 .. 8300 complete blocks; scores: random, heavy ties, +-0, -inf, NaN, too few finite.
    const unsigned rows = 64;
    const std::uint64_t p0 = 2050;  // visible 2051 .. -> complete 512 (row 0..1: all visible) then 513 ..
    std::vector<std::uint64_t> pos(rows);
    for (unsigned r = 0; r < rows; ++r) pos[r] = r < 8 ? p0 + r : 2400 + static_cast<std::uint64_t>(r) * 470U;  // not contiguous: one launch per row below
    const std::uint32_t max_blocks = 33000 / 4;
    std::vector<float> all(static_cast<size_t>(rows) * max_blocks, 0.0F);
    std::uniform_real_distribution<float> u(0.0F, 4.0F);
    for (unsigned r = 0; r < rows; ++r) {
        const auto complete = static_cast<std::uint32_t>((pos[r] + 1) / 4);
        float* s = &all[static_cast<size_t>(r) * max_blocks];
        const unsigned kind = r % 8U;
        for (std::uint32_t b = 0; b < complete; ++b) {
            float v = u(rng);
            if (kind == 1) v = static_cast<float>(static_cast<int>(v));           // 4 distinct values: massive ties
            if (kind == 2) v = (b % 3U) ? 0.0F : -0.0F;                           // all zeros, both signs
            if (kind == 3) v = b % 5U ? -INFINITY : v;                            // ~1/5 finite (> 512 when complete > 2560)
            if (kind == 4) v = b % 50U ? -INFINITY : v;                           // too few finite -> zeros fallback
            if (kind == 5) v = b % 7U == 0 ? std::nanf("") : v;                   // NaN never selected
            if (kind == 6) v = b % 2U ? INFINITY : v;                             // +inf counts
            if (kind == 7) v = rbf(v * 1e-3F);                                    // BF16-like coarse values: ties
            s[b] = v;
        }
    }
    Dev<float> ds(all);
    Dev<std::uint32_t> dsel(static_cast<size_t>(rows) * 2051U);
    for (unsigned r = 0; r < rows; ++r)
        q::select_rows(ds.p + static_cast<size_t>(r) * max_blocks, max_blocks, 1, pos[r], dsel.p + static_cast<size_t>(r) * 2051U, nullptr);
    CK(cudaDeviceSynchronize());
    const auto got = dsel.get();
    const auto got_scores = ds.get();
    unsigned bad = 0, bad_scores = 0;
    for (unsigned r = 0; r < rows; ++r) {
        std::vector<float> s(all.begin() + static_cast<long long>(r) * max_blocks, all.begin() + static_cast<long long>(r + 1) * max_blocks);
        const auto want = ref_select(s, static_cast<std::uint32_t>(pos[r] + 1));
        if (want.size() != q::selected_count(pos[r])) { std::printf("count mismatch row %u\n", r); ++bad; continue; }
        for (size_t i = 0; i < want.size(); ++i) if (got[r * 2051ULL + i] != want[i]) {
            if (!bad) std::printf("select row %u (pos %llu kind %u) slot %zu got %u want %u\n", r, static_cast<unsigned long long>(pos[r]), r % 8U, i, got[r * 2051ULL + i], want[i]);
            ++bad; break;
        }
        for (std::uint32_t b = 0; b < (pos[r] + 1) / 4; ++b)
            if (std::memcmp(&s[b], &got_scores[static_cast<size_t>(r) * max_blocks + b], 4) != 0) { ++bad_scores; break; }
    }
    if (bad || bad_scores) { ++g_fail; std::printf("FAIL select: %u rows differ in selection, %u in scores\n", bad, bad_scores); }
    else { ++g_pass; std::printf("ok   select (64 rows: random, ties, +-0, -inf, too-few-finite repeats, NaN, +inf)\n"); }
}

static void test_attention_direct() {
    // Synthetic selections incl. the zeros fallback (tokens 0..3 repeated), count 2051 and small counts.
    const unsigned rows = 6;
    const std::uint64_t p0s[rows] = {0, 3, 700, 2050, 9001, 20003};
    const std::uint64_t ctx = 20004;
    auto kv = rand_bf16(ctx * 1024ULL, 1.0F);
    for (size_t i = 512; i < kv.size(); i += 1024) for (unsigned f = 0; f < 512; ++f) kv[i + f] = tobf(bf(kv[i + f]) * 2.0F);
    Dev<std::uint16_t> dkv(kv);
    for (unsigned r = 0; r < rows; ++r) {
        const auto p = p0s[r];
        const auto count = q::selected_count(p);
        std::vector<std::uint32_t> sel(2051, 0);
        std::uniform_int_distribution<std::uint32_t> ub(0, static_cast<std::uint32_t>(p / 4));
        if (count > 2048) {
            const unsigned zeros = r == 4 ? 37 : 0;
            std::vector<std::uint32_t> blocks;
            while (blocks.size() < 512 - zeros) { auto b = ub(rng); if (b < (p + 1) / 4 && std::find(blocks.begin(), blocks.end(), b) == blocks.end()) blocks.push_back(b); }
            std::sort(blocks.begin(), blocks.end());
            unsigned k = 0;
            for (unsigned z = 0; z < zeros * 4; ++z) sel[k++] = z % 4;
            for (auto b : blocks) for (unsigned o = 0; o < 4; ++o) sel[k++] = b * 4 + o;
            for (auto t = ((p + 1) / 4) * 4; t <= p; ++t) sel[k++] = static_cast<std::uint32_t>(t);
        } else for (unsigned i = 0; i < count; ++i) sel[i] = i;
        auto qg = rand_bf16f(12288, 1.0F);
        for (unsigned f = 0; f < 256; ++f) qg[256 + f] = rbf(qg[256 + f] * 8.0F);  // gate of head 0: saturating sigmoid
        std::vector<float> out(6144);
        std::vector<std::uint32_t> selv(sel.begin(), sel.begin() + count);
        std::vector<float> sc;
        ref_attention_row(qg.data(), selv, kv, out.data(), &sc);
        Dev<float> dq(qg), dsc(24ULL * 2051), dout(6144);
        Dev<std::uint32_t> dsel(sel);
        q::attention(dq.p, 1, p, dsel.p, dkv.p, dsc.p, dout.p, g_tables, nullptr);
        CK(cudaDeviceSynchronize());
        const auto gs = dsc.get();
        std::vector<float> gsc(static_cast<size_t>(24) * count);
        for (unsigned h = 0; h < 24; ++h) for (unsigned s = 0; s < count; ++s) gsc[h * count + s] = gs[h * 2051ULL + s];
        char name[96];
        std::snprintf(name, sizeof(name), "attention weights pos %llu count %u", static_cast<unsigned long long>(p), count);
        same(name, gsc, sc);
        std::snprintf(name, sizeof(name), "attention output pos %llu", static_cast<unsigned long long>(p));
        same(name, dout.get(), out);
    }
}

// End-to-end layer_core vs host references for rows [p0, p0 + rows) over a random prior state.
static void test_layer(const std::vector<std::uint16_t>& wq, const std::vector<std::uint16_t>& wk, const std::vector<float>& qn,
    const std::vector<float>& kn, const std::vector<float>& iqn, const std::vector<float>& ikn, std::uint64_t p0, unsigned rows,
    unsigned chunk, bool ties, const char* tag) {
    const std::uint64_t ctx = p0 + rows;
    auto kv = rand_bf16(ctx * 1024ULL, 1.0F);
    auto raw = rand_bf16(ctx * 128ULL, 2.0F);
    if (ties && p0 >= 4) {
        // zero raw keys of the prior-state blocks b >= 7 except every 5th -> pooled key 0 -> block score exactly 0 for
        // every row: fewer than 512 positive scores, so the threshold is 0 with ~1000+ ties, resolved to the lowest indices
        for (std::uint64_t b = 7; b * 4 + 4 <= p0; ++b) if (b % 5U) std::memset(&raw[b * 4 * 128], 0, 4 * 256);
    }
    const auto hidden = rand_bf16(static_cast<size_t>(rows) * 2560U, 1.0F);
    auto qg = rand_bf16f(static_cast<size_t>(rows) * 12288U, 1.0F);
    auto k = rand_bf16f(static_cast<size_t>(rows) * 512U, 1.0F);
    auto v = rand_bf16f(static_cast<size_t>(rows) * 512U, 1.0F);
    Dev<std::uint16_t> dkv(kv), draw(raw), dh(hidden), dwq(wq), dwk(wk);
    Dev<float> dqg(qg), dk(k), dv(v), dqn(qn), dkn(kn), diqn(iqn), dikn(ikn), diq(static_cast<size_t>(rows) * 512U),
        dik(static_cast<size_t>(rows) * 128U), dout(static_cast<size_t>(rows) * 6144U);
    Dev<float2> dcs(g_cs);
    q::LayerWeights w{dwq.p, dwk.p, dqn.p, dkn.p, diqn.p, dikn.p};
    q::LayerState st{dkv.p, draw.p};
    q::RowBuffers b{dh.p, dqg.p, dk.p, dv.p, diq.p, dik.p, dout.p};
    auto ws = q::allocate_workspace(chunk, ctx);
    q::layer_core(b, rows, p0, w, st, dcs.p, g_tables, ws, nullptr);
    CK(cudaDeviceSynchronize());
    // host
    std::vector<float> iq, ik;
    ref_indexer(hidden, rows, wq, wk, iq, ik);
    ref_norm_rope(iq, rows, 512, 4, 128, 128, iqn, p0);
    ref_norm_rope(qg, rows, 12288, 24, 256, 512, qn, p0);
    ref_norm_rope(k, rows, 512, 2, 256, 256, kn, p0);
    for (unsigned r = 0; r < rows; ++r) {
        const auto p = p0 + r;
        for (unsigned f = 0; f < 128; ++f) raw[p * 128 + f] = tobf(ik[r * 128ULL + f]);
        for (unsigned f = 0; f < 512; ++f) { kv[p * 1024 + f] = tobf(k[r * 512ULL + f]); kv[p * 1024 + 512 + f] = tobf(v[r * 512ULL + f]); }
    }
    const auto last_complete = static_cast<std::uint32_t>(ctx / 4);
    std::vector<float> pooled(static_cast<size_t>(last_complete) * 128);
    if (last_complete > 512) for (std::uint32_t bl = 0; bl < last_complete; ++bl) ref_pool_feature_vec(raw, bl, ikn, &pooled[bl * 128ULL]);
    std::vector<float> out(static_cast<size_t>(rows) * 6144);
    unsigned selected_rows = 0;
    for (unsigned r = 0; r < rows; ++r) {
        const auto p = p0 + r;
        const auto complete = static_cast<std::uint32_t>((p + 1) / 4);
        std::vector<float> bs(complete);
        if (complete > 512) { ++selected_rows; for (std::uint32_t bl = 0; bl < complete; ++bl) bs[bl] = ref_block_score(&iq[r * 512ULL], &pooled[bl * 128ULL]); }
        const auto sel = ref_select(bs, static_cast<std::uint32_t>(p + 1));
        ref_attention_row(&qg[r * 12288ULL], sel, kv, &out[r * 6144ULL]);
    }
    char name[128];
    const auto nm = [&](const char* what) { std::snprintf(name, sizeof(name), "%s %s [%llu,+%u) chunk %u", what, tag, static_cast<unsigned long long>(p0), rows, chunk); return name; };
    same(nm("layer index_query"), diq.get(), iq);
    same(nm("layer q_gate"), dqg.get(), qg);
    same(nm("layer k"), dk.get(), k);
    same(nm("layer kv state"), dkv.get(), kv);
    same(nm("layer raw index state"), draw.get(), raw);
    if (last_complete > 512) {
        Dev<float> tmp(0);
        std::vector<float> gp(static_cast<size_t>(last_complete) * 128);
        CK(cudaMemcpy(gp.data(), ws.pooled_keys, gp.size() * 4, cudaMemcpyDeviceToHost));
        same(nm("layer pooled keys"), gp, pooled);
    }
    std::printf("     (%u of %u rows use block selection)\n", selected_rows, rows);
    same(nm("layer attention_output"), dout.get(), out);
    q::free_workspace(ws);
}

static void bench(const std::vector<std::uint16_t>& wq, const std::vector<std::uint16_t>& wk, const std::vector<float>& qn,
    const std::vector<float>& kn, const std::vector<float>& iqn, const std::vector<float>& ikn, std::uint64_t p0, unsigned rows) {
    const std::uint64_t ctx = p0 + rows;
    Dev<std::uint16_t> dkv(rand_bf16(ctx * 1024ULL, 1.0F)), draw(rand_bf16(ctx * 128ULL, 2.0F)), dh(rand_bf16(static_cast<size_t>(rows) * 2560U, 1.0F)),
        dwq(wq), dwk(wk);
    const auto qg = rand_bf16f(static_cast<size_t>(rows) * 12288U, 1.0F);
    Dev<float> dqg(qg), dk(rand_bf16f(static_cast<size_t>(rows) * 512U, 1.0F)), dv(rand_bf16f(static_cast<size_t>(rows) * 512U, 1.0F)), dqn(qn), dkn(kn),
        diqn(iqn), dikn(ikn), diq(static_cast<size_t>(rows) * 512U), dik(static_cast<size_t>(rows) * 128U), dout(static_cast<size_t>(rows) * 6144U);
    Dev<float2> dcs(g_cs);
    q::LayerWeights w{dwq.p, dwk.p, dqn.p, dkn.p, diqn.p, dikn.p};
    q::LayerState st{dkv.p, draw.p};
    q::RowBuffers b{dh.p, dqg.p, dk.p, dv.p, diq.p, dik.p, dout.p};
    const unsigned chunk = std::min(rows, 512U);
    auto ws = q::allocate_workspace(chunk, ctx);
    cudaEvent_t e[10];
    for (auto& x : e) CK(cudaEventCreate(&x));
    float t[9] = {};
    for (int it = 0; it < 2; ++it) {
        dqg.put(qg);
        CK(cudaEventRecord(e[0]));
        q::indexer_projection(b.hidden, rows, w, b.index_query, b.index_key, nullptr);
        CK(cudaEventRecord(e[1]));
        q::norm_rope(b.index_query, rows, 512, 4, 128, 128, w.index_q_norm, p0, dcs.p, g_tables, nullptr);
        q::norm_rope(b.q_gate, rows, 12288, 24, 256, 512, w.q_norm, p0, dcs.p, g_tables, nullptr);
        q::norm_rope(b.k, rows, 512, 2, 256, 256, w.k_norm, p0, dcs.p, g_tables, nullptr);
        q::store_rows(b.index_key, b.k, b.v, rows, p0, st, nullptr);
        CK(cudaEventRecord(e[2]));
        const auto lc = static_cast<std::uint32_t>(ctx / 4);
        if (lc > 512) q::block_pool(st.raw_index, 0, lc, w.index_k_norm, dcs.p, g_tables, ws.pooled_keys, nullptr);
        CK(cudaEventRecord(e[3]));
        float acc[5] = {};
        for (unsigned c = 0; c < rows; c += chunk) {
            const auto n = std::min(chunk, rows - c);
            cudaEvent_t a[6];
            for (auto& x : a) CK(cudaEventCreate(&x));
            CK(cudaEventRecord(a[0]));
            q::block_scores(b.index_query + c * 512ULL, n, p0 + c, ws.pooled_keys, ws.block_scores, ws.max_blocks, nullptr);
            CK(cudaEventRecord(a[1]));
            q::select_rows(ws.block_scores, ws.max_blocks, n, p0 + c, ws.selected, nullptr);
            CK(cudaEventRecord(a[2]));
            q::attention_scores(b.q_gate + c * 12288ULL, n, p0 + c, ws.selected, st.kv, ws.scores, nullptr);
            CK(cudaEventRecord(a[3]));
            q::attention_softmax(n, p0 + c, ws.scores, g_tables, nullptr);
            CK(cudaEventRecord(a[4]));
            q::attention_values(b.q_gate + c * 12288ULL, n, p0 + c, ws.selected, st.kv, ws.scores, b.attention_output + c * 6144ULL, g_tables, nullptr);
            CK(cudaEventRecord(a[5]));
            CK(cudaEventSynchronize(a[5]));
            for (int i = 0; i < 5; ++i) { float ms; CK(cudaEventElapsedTime(&ms, a[i], a[i + 1])); acc[i] += ms; }
            for (auto& x : a) cudaEventDestroy(x);
        }
        CK(cudaDeviceSynchronize());
        for (int i = 0; i < 3; ++i) CK(cudaEventElapsedTime(&t[i], e[i], e[i + 1]));
        t[3] = acc[0]; t[4] = acc[1]; t[5] = acc[2]; t[6] = acc[3]; t[7] = acc[4];
    }
    std::printf("bench rows %u at positions [%llu, %llu): indexer %.3f ms, norms+store %.3f, pool %.3f, block scores %.3f, select %.3f, scores %.3f, softmax %.3f, values+gate %.3f ms\n",
        rows, static_cast<unsigned long long>(p0), static_cast<unsigned long long>(ctx), t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    q::free_workspace(ws);
}

static std::vector<float> f32_tensor(const strix::qwen_gguf::Model& m, const std::string& name) {
    const auto b = m.bytes(name);
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), b.size());
    return v;
}
static std::vector<std::uint16_t> bf16_tensor(const strix::qwen_gguf::Model& m, const std::string& name) {
    const auto b = m.bytes(name);
    std::vector<std::uint16_t> v(b.size() / 2);
    std::memcpy(v.data(), b.data(), b.size());
    return v;
}

static void fixture_probe() {
    const char* path = "qrtx-build\\fx\\d64.dump";
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::printf("fixture %s absent: indexed_attention replay skipped\n", path); return; }
    std::fclose(f);
    strix::qwen_fixture::Dump dump(path);
    std::map<std::string, unsigned> ids;
    for (const auto& r : dump.records()) if (r.kernel_id.find("indexed") != std::string::npos || r.kernel_id.find("qsa") != std::string::npos) ++ids[r.kernel_id];
    std::printf("fixture %s: %zu records; QSA-step kernel ids:\n", path, dump.records().size());
    for (const auto& [id, n] : ids) std::printf("  %s (%u records)\n", id.c_str(), n);
    std::printf("  (step replay needs the dense A8 q/k/v/o projections of the dense family; not wired here)\n");
}

int main(int argc, char** argv) {
    bool real = true, do_bench = false;
    for (int i = 1; i < argc; ++i) { if (!std::strcmp(argv[i], "--no-real")) real = false; if (!std::strcmp(argv[i], "--bench")) do_bench = true; }
    const char* td = std::getenv("STRIX_QRTX_TABLES");
    const std::string tables_dir = td && *td ? td : "qrtx-build\\tables";
    g_tables = am::load_tables(tables_dir);
    hm::load(tables_dir);
    load_cs(33000);

    // synthetic weights (BF16 projections; F32 norm weights around 1)
    auto wq = rand_bf16(512ULL * 2560U, 0.02F), wk = rand_bf16(128ULL * 2560U, 0.02F);
    auto qn = rand_bf16f(256, 0.3F), kn = rand_bf16f(256, 0.3F), iqn = rand_bf16f(128, 0.3F), ikn = rand_bf16f(128, 0.3F);
    for (auto* v : {&qn, &kn, &iqn, &ikn}) for (auto& x : *v) x += 1.0F;

    test_indexer(wq, wk, 40, 1.0F, "synthetic 40 rows");
    test_norm_rope(qn, kn, iqn, 9000, 5, "synthetic pos 9000");
    test_norm_rope(qn, kn, iqn, 0, 3, "synthetic pos 0");
    test_select();
    test_attention_direct();
    test_layer(wq, wk, qn, kn, iqn, ikn, 0, 70, 32, false, "synthetic");
    test_layer(wq, wk, qn, kn, iqn, ikn, 2040, 24, 16, true, "synthetic ties (complete 510..516)");
    test_layer(wq, wk, qn, kn, iqn, ikn, 8190, 40, 64, true, "synthetic ties (pos > 8192)");
    test_layer(wq, wk, qn, kn, iqn, ikn, 20000, 12, 8, false, "synthetic deep");

    if (real) {
        strix::qwen_gguf::Model m;
        for (const auto& e : m.info().metadata)
            if (e.key.find("rope") != std::string::npos || e.key.find("indexer") != std::string::npos || e.key.find("rms") != std::string::npos)
                std::printf("gguf %s = %s\n", e.key.c_str(), e.floating_value ? std::to_string(*e.floating_value).c_str()
                    : e.unsigned_value ? std::to_string(*e.unsigned_value).c_str() : e.signed_value ? std::to_string(*e.signed_value).c_str()
                    : e.string_value ? e.string_value->c_str() : "(array)");
        const auto rwq = bf16_tensor(m, "blk.3.indexer.q_proj.weight"), rwk = bf16_tensor(m, "blk.3.indexer.k_proj.weight");
        const auto rqn = f32_tensor(m, "blk.3.attn_q_norm.weight"), rkn = f32_tensor(m, "blk.3.attn_k_norm.weight");
        const auto riqn = f32_tensor(m, "blk.3.indexer.q_norm.weight"), rikn = f32_tensor(m, "blk.3.indexer.k_norm.weight");
        test_indexer(rwq, rwk, 32, 1.0F, "blk.3");
        test_norm_rope(rqn, rkn, riqn, 4097, 4, "blk.3");
        test_layer(rwq, rwk, rqn, rkn, riqn, rikn, 0, 64, 64, false, "blk.3");
        test_layer(rwq, rwk, rqn, rkn, riqn, rikn, 9000, 20, 20, true, "blk.3");
        if (do_bench) {
            bench(rwq, rwk, rqn, rkn, riqn, rikn, 0, 8192);
            bench(rwq, rwk, rqn, rkn, riqn, rikn, 32768 - 1024, 1024);
        }
    }
    fixture_probe();
    am::free_tables(g_tables);
    std::printf("qwen_qsa_tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
