// cuda_core device test: every kernel vs a float64 CPU reference (tolerance) and bitwise row-count / repeat invariance.
// Needs one CUDA device (run with CUDA_VISIBLE_DEVICES=0). Exit 0 = pass.
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <limits>
#include <vector>

#include "strix/cuda_core/attention.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/gdn.hpp"
#include "strix/cuda_core/norms.hpp"
#include "strix/cuda_core/rope.hpp"
#include "strix/cuda_core/nvfp4.hpp"
#include "../../src/cuda_core/common.cuh"

namespace cc = strix::cuda_core;
static int g_fail = 0;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); return 2; } } while (0)
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

template <class T> T* dev(const std::vector<T>& h) {
    T* p = nullptr; cudaMalloc(&p, h.size() * sizeof(T)); cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); return p;
}
template <class T> T* devz(std::size_t n) { T* p = nullptr; cudaMalloc(&p, n * sizeof(T)); cudaMemset(p, 0, n * sizeof(T)); return p; }
template <class T> std::vector<T> get(const T* p, std::size_t n) {
    std::vector<T> h(n); cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost); return h;
}
static bool bits_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}
static double maxrel(const std::vector<float>& a, const std::vector<double>& ref) {
    if (a.size() != ref.size()) return std::numeric_limits<double>::infinity();
    for (float v : a) if (!std::isfinite(v)) return std::numeric_limits<double>::infinity();
    for (double v : ref) if (!std::isfinite(v)) return std::numeric_limits<double>::infinity();
    double worst = 0, scale = 1e-30;
    for (double r : ref) scale = std::max(scale, std::fabs(r));
    for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - ref[i]) / scale);
    return worst;
}
static std::uint32_t* dpos(std::uint32_t v) { std::uint32_t* p = nullptr; cudaMalloc((void**)&p, 4); cudaMemcpy(p, &v, 4, cudaMemcpyHostToDevice); return p; }
static std::uint16_t bf16_bits(float f) {
    std::uint32_t u; std::memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return static_cast<std::uint16_t>(u >> 16);
}
static float bf16_val(std::uint16_t b) { std::uint32_t u = static_cast<std::uint32_t>(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
static std::uint16_t half_bits(float f) {  // exact for the values used (d = k/1024)
    __half h = __float2half(f); std::uint16_t b; std::memcpy(&b, &h, 2); return b;
}
static float half_val(std::uint16_t b) { __half h; std::memcpy(&h, &b, 2); return __half2float(h); }

static std::mt19937 rng(1234);
static float rnd(float s) { return std::uniform_real_distribution<float>(-s, s)(rng); }

static int test_dense_arith(cc::DenseArith arith, std::uint32_t in) {
    cc::set_dense_arith(arith);
    const bool mma = arith != cc::DenseArith::fp32;
    const std::uint32_t out = mma ? 80 : 70, R = 32;
    std::vector<std::uint16_t> wb(out * in);
    std::vector<double> wd(out * in);
    for (std::size_t i = 0; i < wb.size(); ++i) { wb[i] = bf16_bits(rnd(0.1f)); wd[i] = bf16_val(wb[i]); }
    std::vector<unsigned char> wq(static_cast<std::size_t>(out) * (in / 32) * 34);
    std::vector<double> wqd(out * in);
    for (std::uint32_t o = 0; o < out; ++o)
        for (std::uint32_t b = 0; b < in / 32; ++b) {
            unsigned char* blk = &wq[(static_cast<std::size_t>(o) * (in / 32) + b) * 34];
            const std::uint16_t hb = half_bits(static_cast<float>(1 + (rng() % 50)) / 1024.0f);
            blk[0] = hb & 0xFF; blk[1] = hb >> 8;
            for (int j = 0; j < 32; ++j) { const int q = static_cast<int>(rng() % 255) - 127; blk[2 + j] = static_cast<unsigned char>(static_cast<signed char>(q)); wqd[o * in + b * 32 + j] = half_val(hb) * q; }
        }
    std::vector<float> wf(out * in);
    for (auto& v : wf) v = rnd(0.1f);
    std::vector<float> x(R * in), res(R * out);
    for (auto& v : x) v = rnd(1.0f);
    for (auto& v : res) v = rnd(1.0f);
    auto* dx = dev(x); auto* dres = dev(res);
    auto* dwb = dev(wb); auto* dwq = dev(wq); auto* dwf = dev(wf);
    auto* dy = devz<float>(R * out); auto* dy1 = devz<float>(R * out);
    struct Case { const char* name; cc::DenseWeight w; const std::vector<double>* ref; std::vector<double> reff; };
    std::vector<double> wfd(wf.begin(), wf.end());
    Case cases[3] = {{"bf16", {dwb, cc::WeightFormat::bf16, out, in}, &wd, {}}, {"q8_0", {dwq, cc::WeightFormat::q8_0, out, in}, &wqd, {}},
                     {"f32", {dwf, cc::WeightFormat::f32, out, in}, &wfd, {}}};
    for (auto& cs : cases) {
        for (std::uint32_t rows : {1u, 2u, 3u, 5u, 8u, 11u, 16u, 17u, 24u, 32u}) {
            if (!mma && rows > 16) continue;
            CK(cc::dense_rows(cs.w, dx, dy, rows, nullptr, nullptr));
            auto y = get(dy, rows * out);
            std::vector<double> ref(rows * out);
            for (std::uint32_t r = 0; r < rows; ++r)
                for (std::uint32_t o = 0; o < out; ++o) { double s = 0; for (std::uint32_t k = 0; k < in; ++k) s += (*cs.ref)[o * in + k] * x[r * in + k]; ref[r * out + o] = s; }
            CHECK(maxrel(y, ref) < (arith == cc::DenseArith::mma1 && cs.w.format != cc::WeightFormat::f32 ? 1e-2 : 1e-5), "%s rows %u relerr %g", cs.name, rows, maxrel(y, ref));
            // row invariance: each row alone must equal the batched row, bitwise
            for (std::uint32_t r = 0; r < rows; ++r) {
                CK(cc::dense_rows(cs.w, dx + r * in, dy1 + r * out, 1, nullptr, nullptr));
            }
            auto y1 = get(dy1, rows * out);
            CHECK(bits_equal(y, y1), "%s rows %u not bitwise equal to rows=1", cs.name, rows);
        }
        // residual: y = res + (W x) bitwise
        CK(cc::dense_rows(cs.w, dx, dy, 7, nullptr, nullptr));
        CK(cc::dense_rows(cs.w, dx, dy1, 7, dres, nullptr));
        auto a = get(dy, 7 * out), b = get(dy1, 7 * out);
        bool ok = true;
        for (std::size_t i = 0; i < a.size(); ++i) { const float e = res[i] + a[i]; ok &= std::memcmp(&e, &b[i], 4) == 0; }
        CHECK(ok, "%s residual not bitwise res + dot", cs.name);
    }
    // fused gate|up: silu(gate)*up vs two dense calls + host silu (tolerance), fused == fused rows=1
    CK(cc::dense_gate_up_silu_rows(cases[0].w, cases[0].w, dx, dy, 9, nullptr));
    CK(cc::dense_rows(cases[0].w, dx, dy1, 9, nullptr, nullptr));
    auto f = get(dy, 9 * out), g = get(dy1, 9 * out);
    std::vector<double> ref(9 * out);
    for (std::size_t i = 0; i < ref.size(); ++i) ref[i] = g[i] / (1.0 + std::exp(-(double)g[i])) * g[i];
    CHECK(maxrel(f, ref) < 1e-5, "fused relerr %g", maxrel(f, ref));
    for (std::uint32_t r = 0; r < 9; ++r) CK(cc::dense_gate_up_silu_rows(cases[0].w, cases[0].w, dx + r * in, dy1 + r * out, 1, nullptr));
    CHECK(bits_equal(f, get(dy1, 9 * out)), "fused rows invariance");
    cc::set_dense_arith(cc::DenseArith::mma3);
    return 0;
}
int test_dense() {
    if (int e = test_dense_arith(cc::DenseArith::fp32, 1056)) return e;
    if (int e = test_dense_arith(cc::DenseArith::mma3, 1056)) return e;
    if (int e = test_dense_arith(cc::DenseArith::mma3, 1280)) return e;  // q8_0 staged path (in % 256 == 0)
    return test_dense_arith(cc::DenseArith::mma1, 1280);
}

int test_norms_rope() {
    const std::uint32_t rows = 5, groups = 3, n = 300;
    std::vector<float> x(rows * groups * n), gain(n), z(x.size());
    for (auto& v : x) v = rnd(2.0f);
    for (auto& v : gain) v = rnd(1.0f);
    for (auto& v : z) v = rnd(2.0f);
    auto *dx = dev(x), *dg = dev(gain), *dz = dev(z), *dy = devz<float>(x.size());
    const cc::GroupedView v{rows, groups, n, 0, 0};
    CK(cc::rmsnorm(dx, v, dg, dy, v, 1e-6f, nullptr));
    auto y = get(dy, x.size());
    std::vector<double> ref(x.size()), ref2(x.size()), ref3(x.size());
    for (std::uint32_t r = 0; r < rows * groups; ++r) {
        double ss = 0; for (std::uint32_t i = 0; i < n; ++i) ss += (double)x[r * n + i] * x[r * n + i];
        const double inv = 1.0 / std::sqrt(ss / n + 1e-6), inv2 = 1.0 / std::sqrt(ss + 1e-6);
        for (std::uint32_t i = 0; i < n; ++i) {
            ref[r * n + i] = x[r * n + i] * inv * gain[i];
            ref2[r * n + i] = x[r * n + i] * inv * gain[i] * (z[r * n + i] / (1.0 + std::exp(-(double)z[r * n + i])));
            ref3[r * n + i] = x[r * n + i] * inv2 * 0.25;
        }
    }
    CHECK(maxrel(y, ref) < 1e-6, "rmsnorm %g", maxrel(y, ref));
    CK(cc::gated_rmsnorm(dx, v, dg, dz, v, cc::GateKind::swish, dy, v, 1e-6f, nullptr));
    CHECK(maxrel(get(dy, x.size()), ref2) < 1e-6, "gated %g", maxrel(get(dy, x.size()), ref2));
    CK(cc::l2norm(dx, v, dy, v, 1e-6f, 0.25f, nullptr));
    CHECK(maxrel(get(dy, x.size()), ref3) < 1e-6, "l2 %g", maxrel(get(dy, x.size()), ref3));
    // in place == out of place
    CK(cc::rmsnorm(dx, v, dg, dx, v, 1e-6f, nullptr));
    CHECK(bits_equal(get(dx, x.size()), y), "in-place rmsnorm");
    // rope
    const std::uint32_t heads = 4, hd = 256, rd = 64, rr = 3, pos0 = 10;
    std::vector<float> ct, st; cc::rope_table_host(32, rd, 1e7, ct, st);
    std::vector<float> q(rr * heads * hd); for (auto& e : q) e = rnd(1.0f);
    auto *dq = dev(q), *dc = dev(ct), *ds = dev(st);
    const cc::GroupedView qv{rr, heads, hd, 0, 0};
    CK(cc::rope_apply(dq, qv, rd, dc, ds, dpos(pos0), nullptr));
    auto qo = get(dq, q.size());
    std::vector<double> qr(q.size());
    for (std::uint32_t r = 0; r < rr; ++r) for (std::uint32_t h = 0; h < heads; ++h) for (std::uint32_t i = 0; i < hd; ++i) {
        const std::size_t o = (r * heads + h) * hd; double val = q[o + i];
        if (i < rd) {
            const std::uint32_t j = i % 32; const double ang = (double)(pos0 + r) * std::pow(1e7, -2.0 * j / rd);
            const double x1 = q[o + j], x2 = q[o + j + 32];
            val = i < 32 ? x1 * std::cos(ang) - x2 * std::sin(ang) : x2 * std::cos(ang) + x1 * std::sin(ang);
        }
        qr[o + i] = val;
    }
    CHECK(maxrel(qo, qr) < 1e-6, "rope %g", maxrel(qo, qr));
    return 0;
}

int test_attention() {
    cc::AttentionDims d; d.heads = 6; d.kv_heads = 2; d.head_dim = 256; d.scale = 1.0f / 16.0f; d.max_positions = 64;
    const std::uint32_t pos0 = 37, R = 4, total = pos0 + R;
    std::vector<float> kf(static_cast<std::size_t>(total) * 2 * 256), vf(kf.size());
    for (auto& e : kf) e = rnd(1.0f);
    for (auto& e : vf) e = rnd(1.0f);
    auto *dk = dev(kf), *dv = dev(vf);
    auto *kc = devz<__nv_bfloat16>(2 * 64 * 256), *vc = devz<__nv_bfloat16>(2 * 64 * 256);
    // history stored in one call, then window
    CK(cc::kv_store(d, dk, dv, kc, vc, dpos(0), 16, nullptr));
    for (std::uint32_t p = 16; p < pos0; p += 16) CK(cc::kv_store(d, dk + p * 512, dv + p * 512, kc, vc, dpos(p), std::min(16u, pos0 - p), nullptr));
    CK(cc::kv_store(d, dk + pos0 * 512, dv + pos0 * 512, kc, vc, dpos(pos0), R, nullptr));
    std::vector<float> q(R * 6 * 256), gt(q.size());
    for (auto& e : q) e = rnd(2.0f);
    for (auto& e : gt) e = rnd(2.0f);
    auto *dq = dev(q), *dgt = dev(gt), *dout = devz<float>(q.size()), *dsc = devz<float>(cc::attention_scratch_floats(d, R)), *dout1 = devz<float>(q.size());
    const cc::GroupedView qv{R, 6, 256, 0, 0};
    CK(cc::attention_decode(d, dq, qv, dgt, qv, kc, vc, dpos(pos0), R, dout, dsc, nullptr));
    auto o = get(dout, q.size());
    // reference with bf16-rounded K/V
    std::vector<double> ref(q.size());
    for (std::uint32_t r = 0; r < R; ++r) for (std::uint32_t h = 0; h < 6; ++h) {
        const std::uint32_t kvh = h / 3, len = pos0 + r + 1;
        std::vector<double> s(len); double m = -1e300;
        for (std::uint32_t p = 0; p < len; ++p) {
            double dot = 0; for (int i = 0; i < 256; ++i) dot += (double)q[(r * 6 + h) * 256 + i] * bf16_val(bf16_bits(kf[(p * 2 + kvh) * 256 + i]));
            s[p] = dot / 16.0; m = std::max(m, s[p]);
        }
        double den = 0; for (auto& e : s) { e = std::exp(e - m); den += e; }
        for (int i = 0; i < 256; ++i) {
            double acc = 0; for (std::uint32_t p = 0; p < len; ++p) acc += s[p] * bf16_val(bf16_bits(vf[(p * 2 + kvh) * 256 + i]));
            const double g = gt[(r * 6 + h) * 256 + i];
            ref[(r * 6 + h) * 256 + i] = acc / den / (1.0 + std::exp(-g));
        }
    }
    CHECK(maxrel(o, ref) < 1e-5, "attention relerr %g", maxrel(o, ref));
    for (std::uint32_t r = 0; r < R; ++r)
        CK(cc::attention_decode(d, dq + r * 1536, cc::GroupedView{1, 6, 256, 0, 0}, dgt + r * 1536, cc::GroupedView{1, 6, 256, 0, 0}, kc, vc, dpos(pos0 + r), 1, dout1 + r * 1536, dsc, nullptr));
    CHECK(bits_equal(o, get(dout1, q.size())), "attention rows invariance");
    return 0;
}

int test_gdn() {
    cc::GdnDims d; d.key_heads = 4; d.value_heads = 12; d.key_dim = 128; d.value_dim = 128;
    const std::uint32_t C = d.conv_channels(), R = 6, hv = d.value_heads, K = 128;
    std::vector<float> convw(C * 4), a(hv), dt(hv), ng(128), qkv(R * C), z(R * hv * 128), ar(R * hv), br(R * hv);
    for (auto& e : convw) e = rnd(0.5f);
    for (auto& e : a) e = -std::exp(rnd(1.0f));
    for (auto& e : dt) e = rnd(1.0f);
    for (auto& e : ng) e = 0.5f + rnd(0.3f);
    for (auto& e : qkv) e = rnd(1.0f);
    for (auto& e : z) e = rnd(1.0f);
    for (auto& e : ar) e = rnd(1.0f);
    for (auto& e : br) e = rnd(1.0f);
    // two initial states: zero conv state, random delta state
    std::vector<float> s0(static_cast<std::size_t>(hv) * K * K), cs0(C * 3);
    for (auto& e : s0) e = rnd(0.2f);
    for (auto& e : cs0) e = rnd(0.5f);
    auto *dcw = dev(convw), *da = dev(a), *ddt = dev(dt), *dng = dev(ng), *dz = dev(z), *dar = dev(ar), *dbr = dev(br);
    cc::GdnLayer L{dcw, da, ddt, dng};
    auto run = [&](std::uint32_t chunk, std::vector<float>& outv, std::vector<float>& sout) {
        std::vector<float> two(2 * s0.size(), 0.0f);
        std::copy(s0.begin(), s0.end(), two.begin());
        auto *draw = dev(qkv), *dqc = devz<float>(R * C), *dout = devz<float>(R * hv * 128), *dst = dev(two), *dcs = dev(cs0);
        auto* dpar = devz<std::uint32_t>(1);
        for (std::uint32_t r = 0; r < R; r += chunk) {
            const std::uint32_t n = std::min(chunk, R - r);
            cc::GdnWindow w{draw + r * C, dqc + r * C, dar + r * hv, dbr + r * hv};
            if (cc::gdn_window(d, L, {dcs, dst}, w, dz + r * hv * 128, dout + r * hv * 128, n, 1e-6f, dpar, nullptr) != cudaSuccess) std::printf("gdn launch fail\n");
            cc::GdnCommitLayer cl{L, {dcs, dst}, w};
            if (cc::gdn_commit(d, &cl, 1, n, n, dpar, nullptr) != cudaSuccess) std::printf("gdn commit fail\n");
        }
        std::uint32_t par = 0;
        cudaMemcpy(&par, dpar, 4, cudaMemcpyDeviceToHost);
        outv = get(dout, R * hv * 128);
        auto all = get(dst, two.size());
        sout.assign(all.begin() + par * s0.size(), all.begin() + (par + 1) * s0.size());
        cudaFree(draw); cudaFree(dqc); cudaFree(dout); cudaFree(dst); cudaFree(dcs); cudaFree(dpar);
    };
    std::vector<float> o6, s6, o1, s1, o3, s3;
    run(6, o6, s6); run(1, o1, s1); run(3, o3, s3);
    CHECK(bits_equal(o6, o1) && bits_equal(s6, s1), "gdn rows=6 vs rows=1 not bitwise");
    CHECK(bits_equal(o6, o3) && bits_equal(s6, s3), "gdn rows=6 vs rows=3 not bitwise");
    {  // prefill entry (one call over all rows, delta lands on the shadow side, then flip): bitwise the window path
        std::vector<float> two(2 * s0.size(), 0.0f);
        std::copy(s0.begin(), s0.end(), two.begin());
        auto *draw = dev(qkv), *dqc = devz<float>(R * C), *dout = devz<float>(R * hv * 128), *dst = dev(two), *dcs = dev(cs0);
        auto* dpar = devz<std::uint32_t>(1);
        cc::GdnWindow w{draw, dqc, dar, dbr};
        CK(cc::gdn_prefill_layer(d, L, {dcs, dst}, w, dz, dout, R, 1e-6f, dpar, nullptr));
        CK(cc::gdn_prefill_flip(dpar, nullptr));
        std::uint32_t par = 0;
        CK(cudaMemcpy(&par, dpar, 4, cudaMemcpyDeviceToHost));
        const auto po = get(dout, R * hv * 128);
        const auto all = get(dst, two.size());
        const std::vector<float> ps(all.begin() + par * s0.size(), all.begin() + (par + 1) * s0.size());
        cudaFree(draw); cudaFree(dqc); cudaFree(dout); cudaFree(dst); cudaFree(dcs); cudaFree(dpar);
        CHECK(bits_equal(po, o6) && bits_equal(ps, s6), "gdn prefill entry vs windows not bitwise");
    }
    {  // build-to-build identity: compare this line between two binaries (same fixture seed) for schedule-only kernel changes
        auto fnv = [](const std::vector<float>& v) {
            std::uint64_t h = 1469598103934665603ull;
            for (float f : v) { std::uint32_t u; std::memcpy(&u, &f, 4); for (int i = 0; i < 4; ++i) { h ^= (u >> (8 * i)) & 0xffu; h *= 1099511628211ull; } }
            return static_cast<unsigned long long>(h);
        };
        std::printf("gdn digest out %016llx state %016llx\n", fnv(o6), fnv(s6));
    }
    // float64 reference
    std::vector<double> cs(cs0.begin(), cs0.end()), S(s0.begin(), s0.end()), ref(R * hv * 128);
    auto silu = [](double v) { return v / (1.0 + std::exp(-v)); };
    for (std::uint32_t r = 0; r < R; ++r) {
        std::vector<double> mix(C);
        for (std::uint32_t c = 0; c < C; ++c) {
            const double in3[4] = {cs[c * 3], cs[c * 3 + 1], cs[c * 3 + 2], qkv[r * C + c]};
            double s = 0; for (int j = 0; j < 4; ++j) s += convw[c * 4 + j] * in3[j];
            mix[c] = silu(s); cs[c * 3] = in3[1]; cs[c * 3 + 1] = in3[2]; cs[c * 3 + 2] = in3[3];
        }
        std::vector<double> qn(4 * K), kn(4 * K);
        for (int h = 0; h < 4; ++h) {
            double sq = 0, sk = 0; for (int i = 0; i < K; ++i) { sq += mix[h * K + i] * mix[h * K + i]; sk += mix[512 + h * K + i] * mix[512 + h * K + i]; }
            for (int i = 0; i < K; ++i) { qn[h * K + i] = mix[h * K + i] / std::sqrt(sq + 1e-6) / std::sqrt(128.0); kn[h * K + i] = mix[512 + h * K + i] / std::sqrt(sk + 1e-6); }
        }
        for (std::uint32_t h = 0; h < hv; ++h) {
            const double beta = 1.0 / (1.0 + std::exp(-(double)br[r * hv + h]));
            const double xin = ar[r * hv + h] + dt[h], sp = xin > 20 ? xin : std::log1p(std::exp(xin));
            const double decay = std::exp(a[h] * sp);
            const int kh = h % 4;
            std::vector<double> od(K);
            for (int j = 0; j < K; ++j) {
                double kv = 0;
                for (int k = 0; k < K; ++k) { S[(h * K + k) * K + j] *= decay; kv += S[(h * K + k) * K + j] * kn[kh * K + k]; }
                const double dl = (mix[1024 + h * K + j] - kv) * beta;
                double acc = 0;
                for (int k = 0; k < K; ++k) { S[(h * K + k) * K + j] += kn[kh * K + k] * dl; acc += S[(h * K + k) * K + j] * qn[kh * K + k]; }
                od[j] = acc;
            }
            double ss = 0; for (double e : od) ss += e * e;
            const double inv = 1.0 / std::sqrt(ss / K + 1e-6);
            for (int j = 0; j < K; ++j) ref[(r * hv + h) * K + j] = od[j] * inv * ng[j] * silu(z[(r * hv + h) * K + j]);
        }
    }
    CHECK(maxrel(o6, ref) < 1e-4, "gdn relerr %g", maxrel(o6, ref));
    std::vector<double> sref = S;
    CHECK(maxrel(s6, sref) < 1e-4, "gdn state relerr %g", maxrel(s6, sref));
    return 0;
}

__global__ void nvfp4_test_epilogue(const float* gate, const float* up, float* y, std::uint32_t out) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out) y[i] = __fmul_rn(cc::cc_silu(gate[i]), up[i]);
}

static int test_nvfp4_fused() {
    constexpr std::uint32_t in = 256;
    cc::nvfp4_set_split_target(0);
    for (const std::uint32_t out : {248320u, 16u}) {
        CHECK(cc::nvfp4_split_k(out, in) == (out == 248320u ? 1u : 2u), "NVFP4 split coverage out=%u", out);
        // Exact packed weights: gate = 1, up = 2, all local scales = 1 (e4m3 0x38).
        std::vector<unsigned char> gate(cc::nvfp4_bytes(out, in), 0x22), up(gate.size(), 0x44);
        for (std::size_t base = 0; base < gate.size(); base += cc::kNvfp4RecordBytes)
            for (std::size_t i = 1024; i < cc::kNvfp4RecordBytes; ++i) gate[base + i] = up[base + i] = 0x38;
        auto* dg = dev(gate);
        auto* du = dev(up);
        std::vector<float> x(in, 0.0f);
        // One split retains the small term after cancellation; two splits lose it.
        x[0] = 32768.0f; x[128] = -32768.0f; x[144] = 1.0f / 1024.0f;
        if (out == 16u) x[128] = -16384.0f;  // Nonzero split-K epilogue coverage.
        auto* dx = dev(x);
        auto* g = devz<float>(out);
        auto* u = devz<float>(out);
        auto* separate = devz<float>(out);
        auto* fused = devz<float>(out);
        const cc::DenseWeight gw{dg, cc::WeightFormat::nvfp4, out, in, 0.5f};
        const cc::DenseWeight uw{du, cc::WeightFormat::nvfp4, out, in, 0.25f};
        CK(cc::nvfp4_dense_rows(gw, dx, g, 1, nullptr, nullptr));
        CK(cc::nvfp4_dense_rows(uw, dx, u, 1, nullptr, nullptr));
        nvfp4_test_epilogue<<<(out + 255) / 256, 256>>>(g, u, separate, out);
        CK(cudaGetLastError());
        CK(cc::nvfp4_gate_up_silu_rows(gw, uw, dx, fused, 1, nullptr));
        CK(cudaDeviceSynchronize());
        CHECK(bits_equal(get(fused, out), get(separate, out)), "NVFP4 fused vs separate bits out=%u", out);
        CK(cudaFree(dg)); CK(cudaFree(du)); CK(cudaFree(dx));
        CK(cudaFree(g)); CK(cudaFree(u)); CK(cudaFree(separate)); CK(cudaFree(fused));
    }
    cc::nvfp4_release_workspace(nullptr);  // the test runs on the default stream
    return 0;
}

int main() {
    if (cudaSetDevice(0) != cudaSuccess) return 2;
    int rc = test_dense() | test_norms_rope() | test_attention() | test_gdn() | test_nvfp4_fused();
    // repeat determinism is covered by the identical-bits checks (each compares two separate launches)
    std::printf(g_fail || rc ? "cuda_core_test: FAILED (%d checks)\n" : "cuda_core_test: PASS\n", g_fail);
    return (g_fail || rc) ? 1 : 0;
}
