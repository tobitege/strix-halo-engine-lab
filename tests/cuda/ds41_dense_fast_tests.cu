// deepseek41 CUDA fast dense sites (cuda_ds41_dense_fast.hpp) against the exact kernels: relative L2 error, max error
// relative to the largest value and cosine per site (Q8_0 x A8 sites of a layer, shared expert, hc mixes, hc expand +
// mixes, norm with the collapse and A8), on synthetic data with model-like scales.
// With --bench: device ms of the sites at 2K / 8K / 32K rows (exact vs fast).

#include "strix/cuda/ds41_common.hpp"
#include "strix/cuda/ds41_core.hpp"
#include "strix/cuda/ds41_dense_fast.hpp"
#include "strix/cuda/ds41_prefill_q8.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace c = strix::ds41::cuda;
namespace pk = strix::ds41::cuda::pk;
namespace fast = strix::ds41::cuda::fast;

namespace {

int failures = 0;
std::mt19937_64 rng(20261020);

template <class T>
c::DeviceBuffer<T> to_device(const std::vector<T>& v) {
    c::DeviceBuffer<T> d(v.size());
    d.upload(v);
    return d;
}

std::uint16_t to_half(float f) {
    // round to nearest even, normal range only (the test values)
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    const std::uint32_t sign = (b >> 16) & 0x8000U;
    const int e = static_cast<int>((b >> 23) & 0xff) - 127 + 15;
    if (e <= 0) return static_cast<std::uint16_t>(sign);
    std::uint32_t m = b & 0x7fffffU;
    std::uint32_t h = (static_cast<std::uint32_t>(e) << 10) | (m >> 13);
    const std::uint32_t rem = m & 0x1fffU;
    if (rem > 0x1000U || (rem == 0x1000U && (h & 1U))) ++h;
    return static_cast<std::uint16_t>(sign | h);
}

// GGUF Q8_0 rows with model-like scales (|w| ~ 0.02).
std::vector<std::uint8_t> random_q8(std::size_t rows, std::size_t width) {
    std::vector<std::uint8_t> w(rows * width / 32 * 34);
    std::normal_distribution<float> g(0.0F, 0.02F);
    std::vector<float> blk(32);
    for (std::size_t b = 0; b < w.size(); b += 34) {
        float amax = 0;
        for (auto& v : blk) { v = g(rng); amax = std::max(amax, std::fabs(v)); }
        const std::uint16_t h = to_half(amax / 127.0F);
        std::memcpy(&w[b], &h, 2);
        for (int j = 0; j < 32; ++j) w[b + 2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(std::lround(blk[j] / (amax / 127.0F))));
    }
    return w;
}

struct Err {
    double rel_l2 = 0, max_rel = 0, cosine = 0;
    std::size_t nonfinite = 0;
};
Err compare(const float* got_d, const float* want_d, std::size_t n) {
    std::vector<float> a(n), b(n);
    STRIX_CUDA_CHECK(cudaDeviceSynchronize());
    STRIX_CUDA_CHECK(cudaMemcpy(a.data(), got_d, n * 4, cudaMemcpyDeviceToHost));
    STRIX_CUDA_CHECK(cudaMemcpy(b.data(), want_d, n * 4, cudaMemcpyDeviceToHost));
    double dd = 0, bb = 0, aa = 0, ab = 0, mx = 0, md = 0;
    Err e;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { ++e.nonfinite; continue; }
        const double d = double(a[i]) - b[i];
        dd += d * d; bb += double(b[i]) * b[i]; aa += double(a[i]) * a[i]; ab += double(a[i]) * b[i];
        mx = std::max(mx, std::fabs(double(b[i])));
        md = std::max(md, std::fabs(d));
    }
    e.rel_l2 = std::sqrt(dd / std::max(bb, 1e-300));
    e.max_rel = md / std::max(mx, 1e-300);
    e.cosine = ab / std::sqrt(std::max(aa * bb, 1e-300));
    return e;
}
void report(const std::string& what, const Err& e, double tol) {
    const bool ok = e.nonfinite == 0 && e.rel_l2 <= tol && e.cosine > 1.0 - tol;
    if (!ok) ++failures;
    std::printf("%s %-44s rel L2 %.3e  max err / max %.3e  cosine 1-%.3e%s\n", ok ? "ok  " : "FAIL", what.c_str(), e.rel_l2, e.max_rel,
                1.0 - e.cosine, e.nonfinite ? "  NON-FINITE" : "");
}

struct Site {
    const char* name;
    std::uint32_t count, width;   // weight rows, full weight width
    std::uint32_t per_group = 0;  // grouped (o_a)
    std::uint32_t parts = 1;      // K-split (o_b) of the exact form (the fast form takes it in one call)
};

c::A8Buffer random_a8(std::uint32_t rows, std::uint32_t width, cudaStream_t s) {
    std::normal_distribution<float> g(0.0F, 1.0F);
    std::vector<float> xf(std::size_t{rows} * width);
    for (auto& f : xf) f = g(rng);
    auto dx = to_device(xf);
    c::A8Buffer a;
    a.resize(rows, width);
    c::enqueue_quantize_a8({dx.get(), rows, width, 0, a.view()}, s);
    STRIX_CUDA_CHECK(cudaStreamSynchronize(s));
    return a;
}

void run_site(const Site& k, const c::Q8Tiled& tiled, c::A8Rows x, float* out, bool fast_form, cudaStream_t s) {
    const std::uint32_t groups = k.per_group ? k.count / k.per_group : 1U;
    const std::uint32_t parts = fast_form ? 1U : k.parts, part = k.width / parts;
    for (std::uint32_t p = 0; p < parts; ++p) {
        c::Q8A8View v;
        v.tiled = tiled;
        v.count = k.count;
        v.per_group = k.per_group;
        v.split = 1;
        v.x = x;
        if (k.parts > 1) {
            v.x.codes += p * part;
            v.x.deltas += p * part / 32U;
            v.x.sums += p * part / 32U;
            v.x.width = part;
            v.k_offset = p * part;
            v.width = part;
        } else {
            v.x.width = groups * k.width;
        }
        v.accumulate = p > 0;
        v.out = out;
        if (fast_form) fast::enqueue_q8_0_a8(v, s);
        else pk::enqueue_q8_0_a8_rows(v, s);
    }
}

const Site kSites[] = {{"q_b 1280->32768", 32768, 1280},        {"o_a 8x(4096->1024)", 8192, 4096, 1024},
                       {"o_b 8192->5120", 5120, 8192, 0, 2},     {"qa_kv 5120->1792", 1792, 5120},
                       {"gate_up 5120->4608", 4608, 5120},       {"down 2304->5120", 5120, 2304}};

void test_sites(cudaStream_t s) {
    for (const auto& k : kSites) {
        auto dw = to_device(random_q8(k.count, k.width));
        c::Q8TiledBuffer tiled;
        tiled.repack(dw.get(), k.count, k.width, s);
        const std::uint32_t groups = k.per_group ? k.count / k.per_group : 1U;
        const std::uint32_t xin = k.parts > 1 ? k.width : groups * k.width;
        for (const std::uint32_t rows : {1U, 37U, 300U, 1024U}) {
            auto x = random_a8(rows, xin, s);
            const std::size_t n = std::size_t{rows} * k.count;
            c::DeviceBuffer<float> want(n), got(n);
            run_site(k, tiled.view(), x.view(), want.get(), false, s);
            run_site(k, tiled.view(), x.view(), got.get(), true, s);
            report(std::string(k.name) + " rows " + std::to_string(rows), compare(got.get(), want.get(), n), 1e-5);
        }
    }
}

void test_shared(cudaStream_t s) {
    const std::uint32_t hidden = 5120, inter = 2304;
    auto dgu = to_device(random_q8(2U * inter, hidden)), dd = to_device(random_q8(hidden, inter));
    c::Q8TiledBuffer tgu, tdown;
    tgu.repack(dgu.get(), 2U * inter, hidden, s);
    tdown.repack(dd.get(), hidden, inter, s);
    for (const std::uint32_t rows : {5U, 1000U}) {
        auto a = random_a8(rows, hidden, s);
        c::DeviceBuffer<std::uint8_t> ws1(c::shared_expert_workspace_bytes(rows, inter)), ws2(c::shared_expert_workspace_bytes(rows, inter));
        c::DeviceBuffer<float> o1(std::size_t{rows} * hidden), o2(std::size_t{rows} * hidden);
        c::SharedExpertView v;
        v.gate_up_tiled = tgu.view();
        v.down_tiled = tdown.view();
        v.x = a.view();
        v.out = o1.get();
        v.workspace = ws1.get();
        pk::enqueue_shared_expert(v, s);
        v.out = o2.get();
        v.workspace = ws2.get();
        fast::enqueue_shared_expert(v, s);
        // the A8 of the activation may round a code differently: a looser bound
        report("shared expert rows " + std::to_string(rows), compare(o2.get(), o1.get(), std::size_t{rows} * hidden), 1e-3);
    }
}

c::DeviceBuffer<float> random_stream(std::size_t n, float sd);

// ---- A8 quantize --------------------------------------------------------------------------------------------------
void test_quantize(cudaStream_t s) {
    for (const std::uint32_t width : {32768U, 8192U}) {
        const std::uint32_t rows = 300;
        auto x = random_stream(std::size_t{rows} * width, 0.5F);
        c::A8Buffer a1, a2;
        a1.resize(rows, width);
        a2.resize(rows, width);
        c::enqueue_quantize_a8({x.get(), rows, width, 0, a1.view()}, s);
        fast::enqueue_quantize_a8({x.get(), rows, width, 0, a2.view()}, s);
        STRIX_CUDA_CHECK(cudaStreamSynchronize(s));
        const auto c1 = a1.codes.download(), c2 = a2.codes.download();
        const auto d1 = a1.deltas.download(), d2 = a2.deltas.download();
        const auto s1 = a1.sums.download(), s2 = a2.sums.download();
        std::size_t diff = 0, big = 0, dd = 0, sd = 0;
        for (std::size_t i = 0; i < c1.size(); ++i) {
            diff += c1[i] != c2[i];
            big += std::abs(int(c1[i]) - int(c2[i])) > 1;
        }
        for (std::size_t g = 0; g < d1.size(); ++g) {
            dd += d1[g] != d2[g];
            // a sum differs by the codes that differ
            int want = 0;
            for (int k = 0; k < 32; ++k) want += c2[g * 32 + k];
            sd += want != s2[g];
        }
        const bool ok = big == 0 && dd == 0 && sd == 0 && diff * 1000 < c1.size();
        if (!ok) ++failures;
        std::printf("%s A8 quantize width %-30u %zu of %zu codes differ (%zu by more than one), deltas differ %zu, sums wrong %zu\n",
                    ok ? "ok  " : "FAIL", width, diff, c1.size(), big, dd, sd);
    }
}

// ---- hc ------------------------------------------------------------------------------------------------------------
struct Hc {
    std::uint32_t hidden = 5120;
    c::DeviceBuffer<std::uint16_t> fn;
    c::DeviceBuffer<float> base, scale, norm_w;
    Hc() {
        std::normal_distribution<float> g(0.0F, 1.0F);
        std::vector<std::uint16_t> f(std::size_t{24} * 4 * hidden);
        for (auto& v : f) v = to_half(g(rng) * 0.01F);
        fn = to_device(f);
        std::vector<float> b(24), sc{0.5F, 0.3F, 0.8F}, w(hidden);
        for (auto& v : b) v = g(rng);
        for (auto& v : w) v = 1.0F + 0.1F * g(rng);
        base = to_device(b);
        scale = to_device(sc);
        norm_w = to_device(w);
    }
    [[nodiscard]] c::HcMixView mix(const float* stream, std::uint32_t rows, float* mixes, float* partial) const {
        c::HcMixView v;
        v.stream = stream;
        v.rows = rows;
        v.hidden = hidden;
        v.fn = fn.get();
        v.base = base.get();
        v.scale = scale.get();
        v.iters = 20;
        v.mixes = mixes;
        v.partial = partial;
        return v;
    }
};

c::DeviceBuffer<float> random_stream(std::size_t n, float sd) {
    std::normal_distribution<float> g(0.0F, sd);
    std::vector<float> v(n);
    for (auto& f : v) f = g(rng);
    return to_device(v);
}

void test_hc(cudaStream_t s) {
    Hc hc;
    const std::uint32_t H = hc.hidden;
    for (const std::uint32_t rows : {1U, 45U, 700U}) {
        auto stream = random_stream(std::size_t{rows} * 4 * H, 3.0F);
        c::DeviceBuffer<float> m1(std::size_t{rows} * 24), m2(std::size_t{rows} * 24);
        c::DeviceBuffer<std::uint8_t> ws(c::hc_mix_workspace_bytes(rows, H));
        c::enqueue_hc_mixes(hc.mix(stream.get(), rows, m1.get(), reinterpret_cast<float*>(ws.get())), s);
        fast::enqueue_hc_mixes(hc.mix(stream.get(), rows, m2.get(), nullptr), s);
        report("hc mixes rows " + std::to_string(rows), compare(m2.get(), m1.get(), std::size_t{rows} * 24), 1e-5);

        // expand (with the mixes m1) then the mixes, vs fused
        auto sub = random_stream(std::size_t{rows} * H, 1.0F);
        c::DeviceBuffer<float> s1(std::size_t{rows} * 4 * H), s2(std::size_t{rows} * 4 * H);
        STRIX_CUDA_CHECK(cudaMemcpyAsync(s1.get(), stream.get(), s1.size() * 4, cudaMemcpyDeviceToDevice, s));  // ordered with the kernels on s
        STRIX_CUDA_CHECK(cudaMemcpyAsync(s2.get(), stream.get(), s2.size() * 4, cudaMemcpyDeviceToDevice, s));  // ordered with the kernels on s
        c::DeviceBuffer<float> n1(std::size_t{rows} * 24), n2(std::size_t{rows} * 24);
        c::enqueue_hc_expand({sub.get(), 0, s1.get(), m1.get(), rows, H, s1.get()}, s);
        c::enqueue_hc_mixes(hc.mix(s1.get(), rows, n1.get(), reinterpret_cast<float*>(ws.get())), s);
        const c::HcExpandView e{sub.get(), 0, s2.get(), m1.get(), rows, H, s2.get()};
        fast::enqueue_hc_mixes(hc.mix(s2.get(), rows, n2.get(), nullptr), s, &e);
        report("hc expand stream rows " + std::to_string(rows), compare(s2.get(), s1.get(), s1.size()), 1e-6);
        report("hc expand + mixes rows " + std::to_string(rows), compare(n2.get(), n1.get(), std::size_t{rows} * 24), 1e-5);

        // everything in one pass: expand, collapse (pre of the expand's mixes) + norm + A8, mixes
        {
            c::DeviceBuffer<float> s3(std::size_t{rows} * 4 * H), n3(std::size_t{rows} * 24), xe(std::size_t{rows} * H), xf(std::size_t{rows} * H);
            STRIX_CUDA_CHECK(cudaMemcpyAsync(s3.get(), stream.get(), s3.size() * 4, cudaMemcpyDeviceToDevice, s));  // ordered with the kernels on s
            c::A8Buffer ae, af;
            ae.resize(rows, H);
            af.resize(rows, H);
            c::NormView ev;
            ev.hc_stream = s1.get();
            ev.pre = m1.get();
            ev.pre_stride = 24;
            ev.weight = hc.norm_w.get();
            ev.rows = rows;
            ev.n = H;
            ev.out = xe.get();
            ev.a8 = ae.view();
            c::enqueue_norm(ev, s);
            fast::HcFusedView f;
            f.mix = hc.mix(s3.get(), rows, n3.get(), nullptr);
            f.sub = sub.get();
            f.emix = m1.get();
            f.pre = m1.get();
            f.weight = hc.norm_w.get();
            f.x = xf.get();
            f.a8 = af.view();
            f.rows_per_block = rows == 45U ? 16U : rows == 700U ? 32U : 8U;  // every block shape
            fast::enqueue_hc_fused(f, s);
            report("fused stream rows " + std::to_string(rows), compare(s3.get(), s1.get(), s1.size()), 1e-6);
            report("fused mixes rows " + std::to_string(rows), compare(n3.get(), n1.get(), std::size_t{rows} * 24), 1e-5);
            report("fused norm rows " + std::to_string(rows), compare(xf.get(), xe.get(), xe.size()), 1e-6);
            const auto c1 = ae.codes.download(), c2 = af.codes.download();
            std::size_t diff = 0, big = 0;
            for (std::size_t i = 0; i < c1.size(); ++i) {
                diff += c1[i] != c2[i];
                big += std::abs(int(c1[i]) - int(c2[i])) > 1;
            }
            const bool ok = big == 0 && diff * 1000 < c1.size();
            if (!ok) ++failures;
            std::printf("%s fused A8 rows %-34u %zu of %zu codes differ (%zu by more than one)\n", ok ? "ok  " : "FAIL", rows, diff, c1.size(), big);
        }

        // norm of the collapse with pre = the mixes' pre (stride 24), with A8
        c::DeviceBuffer<float> x1(std::size_t{rows} * H), x2(std::size_t{rows} * H);
        c::A8Buffer a1, a2;
        a1.resize(rows, H);
        a2.resize(rows, H);
        c::NormView nv;
        nv.hc_stream = s1.get();
        nv.pre = n1.get();
        nv.pre_stride = 24;
        nv.weight = hc.norm_w.get();
        nv.rows = rows;
        nv.n = H;
        nv.out = x1.get();
        nv.a8 = a1.view();
        c::enqueue_norm(nv, s);
        nv.out = x2.get();
        nv.a8 = a2.view();
        fast::enqueue_norm(nv, s);
        report("norm (collapse) rows " + std::to_string(rows), compare(x2.get(), x1.get(), x1.size()), 1e-6);
        // the A8 codes: count of codes that differ (by one step at a rounding tie)
        STRIX_CUDA_CHECK(cudaDeviceSynchronize());
        const auto c1 = a1.codes.download(), c2 = a2.codes.download();
        std::size_t diff = 0, big = 0;
        for (std::size_t i = 0; i < c1.size(); ++i) {
            diff += c1[i] != c2[i];
            big += std::abs(int(c1[i]) - int(c2[i])) > 1;
        }
        const bool ok = big == 0 && diff * 1000 < c1.size();
        if (!ok) ++failures;
        std::printf("%s A8 of the norm rows %-26u %zu of %zu codes differ (%zu by more than one)\n", ok ? "ok  " : "FAIL", rows, diff, c1.size(), big);
        // q_norm form (x rows with a stride)
        nv.hc_stream = nullptr;
        nv.pre = nullptr;
        nv.x = s1.get();
        nv.x_stride = 4 * H;
        nv.out = x1.get();
        nv.a8 = a1.view();
        c::enqueue_norm(nv, s);
        nv.out = x2.get();
        nv.a8 = a2.view();
        fast::enqueue_norm(nv, s);
        report("norm (strided x) rows " + std::to_string(rows), compare(x2.get(), x1.get(), x1.size()), 1e-6);
    }
}

// ---- bench ---------------------------------------------------------------------------------------------------------
float time_ms(cudaStream_t s, const std::function<void()>& body, int reps) {
    body();
    float best = 1e30F;
    for (int i = 0; i < reps; ++i) {
        c::EventTimer t;
        t.start(s);
        body();
        best = std::min(best, t.stop(s));
    }
    return best;
}

void bench(cudaStream_t s, std::uint32_t C, bool q8) {
    std::printf("bench: device ms (min of runs) over T rows in calls of %u rows; exact = the prefill kernels, fast = cuda_ds41_dense_fast\n", C);
    for (const auto& k : kSites) {
        if (!q8) break;
        auto dw = to_device(random_q8(k.count, k.width));
        c::Q8TiledBuffer tiled;
        tiled.repack(dw.get(), k.count, k.width, s);
        const std::uint32_t groups = k.per_group ? k.count / k.per_group : 1U;
        const std::uint32_t xin = k.parts > 1 ? k.width : groups * k.width;
        auto x = random_a8(C, xin, s);
        c::DeviceBuffer<float> out(std::size_t{C} * k.count);
        for (const std::uint32_t T : {2048U, 8192U, 32768U}) {
            const int reps = T >= 32768U ? 3 : 5;
            const auto run = [&](bool f) {
                for (std::uint32_t r0 = 0; r0 < T; r0 += C) run_site(k, tiled.view(), x.view(), out.get(), f, s);
            };
            const float a = time_ms(s, [&] { run(false); }, reps), b = time_ms(s, [&] { run(true); }, reps);
            const double ops = 2.0 * T * k.count * (k.per_group ? k.width : k.width);
            std::printf("  T %5u %-20s exact %9.2f ms (%4.0f TOPS)  fast %8.2f ms (%4.0f TOPS)  x%.2f\n", T, k.name, a, ops / (a * 1e-3) / 1e12, b,
                        ops / (b * 1e-3) / 1e12, a / b);
        }
    }
    {
        auto xo = random_stream(std::size_t{C} * 32768U, 0.5F);
        c::A8Buffer a;
        a.resize(C, 32768U);
        for (const std::uint32_t T : {2048U, 8192U, 32768U}) {
            const auto run = [&](bool f) {
                for (std::uint32_t r0 = 0; r0 < T; r0 += C) {
                    const c::QuantizeA8View v{xo.get(), C, 32768U, 0, a.view()};
                    if (f) fast::enqueue_quantize_a8(v, s);
                    else c::enqueue_quantize_a8(v, s);
                }
            };
            const float e = time_ms(s, [&] { run(false); }, 3), f = time_ms(s, [&] { run(true); }, 3);
            std::printf("  T %5u A8 quantize of o (32768)  exact %7.2f ms  fast %6.2f ms  x%.2f\n", T, e, f, e / f);
        }
    }
    // hc: per T rows, mixes (exact / fast), expand then mixes (exact) vs fused (fast), norm (exact / fast)
    Hc hc;
    const std::uint32_t H = hc.hidden;
    for (const std::uint32_t T : {2048U, 8192U, 32768U}) {
        auto stream = random_stream(std::size_t{T} * 4 * H, 3.0F);
        auto sub = random_stream(std::size_t{T} * H, 1.0F);
        c::DeviceBuffer<float> mixes(std::size_t{T} * 24), mixes2(std::size_t{T} * 24), x(std::size_t{T} * H);
        c::DeviceBuffer<std::uint8_t> ws(c::hc_mix_workspace_bytes(C, H));
        c::A8Buffer a8;
        a8.resize(T, H);
        fast::enqueue_hc_mixes(hc.mix(stream.get(), T, mixes2.get(), nullptr), s);  // valid mixes for the expand
        const int reps = T >= 32768U ? 3 : 5;
        const auto chunks = [&](auto&& f) {
            for (std::uint32_t r0 = 0; r0 < T; r0 += C) f(r0, std::min(C, T - r0));
        };
        const auto mix_at = [&](std::uint32_t r0, std::uint32_t n) {
            return hc.mix(stream.get() + std::size_t{r0} * 4 * H, n, mixes.get() + std::size_t{r0} * 24, reinterpret_cast<float*>(ws.get()));
        };
        const auto expand_at = [&](std::uint32_t r0, std::uint32_t n) {
            float* st = stream.get() + std::size_t{r0} * 4 * H;
            return c::HcExpandView{sub.get() + std::size_t{r0} * H, 0, st, mixes2.get() + std::size_t{r0} * 24, n, H, st};
        };
        const float m_ex = time_ms(s, [&] { chunks([&](auto r0, auto n) { c::enqueue_hc_mixes(mix_at(r0, n), s); }); }, reps);
        const float m_fa = time_ms(s, [&] { chunks([&](auto r0, auto n) { fast::enqueue_hc_mixes(mix_at(r0, n), s); }); }, reps);
        const float e_ex = time_ms(s, [&] { chunks([&](auto r0, auto n) { c::enqueue_hc_expand(expand_at(r0, n), s); }); }, reps);
        const float em_fa = time_ms(s, [&] {
            chunks([&](auto r0, auto n) {
                const auto e = expand_at(r0, n);
                fast::enqueue_hc_mixes(mix_at(r0, n), s, &e);
            });
        }, reps);
        const float all_fa = time_ms(s, [&] {
            chunks([&](auto r0, auto n) {
                fast::HcFusedView f;
                f.mix = mix_at(r0, n);
                f.sub = sub.get() + std::size_t{r0} * H;
                f.emix = mixes2.get() + std::size_t{r0} * 24;
                f.pre = f.emix;
                f.weight = hc.norm_w.get();
                f.x = x.get() + std::size_t{r0} * H;
                f.a8 = a8.view();
                f.a8.codes += std::size_t{r0} * H;
                f.a8.deltas += std::size_t{r0} * H / 32;
                f.a8.sums += std::size_t{r0} * H / 32;
                f.a8.rows = n;
                fast::enqueue_hc_fused(f, s);
            });
        }, reps);
        const auto norm_at = [&](std::uint32_t r0, std::uint32_t n) {
            c::NormView nv;
            nv.hc_stream = stream.get() + std::size_t{r0} * 4 * H;
            nv.pre = mixes2.get() + std::size_t{r0} * 24;
            nv.pre_stride = 24;
            nv.weight = hc.norm_w.get();
            nv.rows = n;
            nv.n = H;
            nv.out = x.get() + std::size_t{r0} * H;
            nv.a8 = a8.view();
            nv.a8.codes += std::size_t{r0} * H;
            nv.a8.deltas += std::size_t{r0} * H / 32;
            nv.a8.sums += std::size_t{r0} * H / 32;
            nv.a8.rows = n;
            return nv;
        };
        const float n_ex = time_ms(s, [&] { chunks([&](auto r0, auto n) { c::enqueue_norm(norm_at(r0, n), s); }); }, reps);
        const float n_fa = time_ms(s, [&] { chunks([&](auto r0, auto n) { fast::enqueue_norm(norm_at(r0, n), s); }); }, reps);
        std::printf("  T %5u hc mixes exact %7.2f fast %6.2f ms | expand %6.2f + mixes %6.2f exact vs fused %6.2f ms | norm exact %6.2f fast %6.2f ms"
                    " | expand + norm + mixes: exact %6.2f, one pass %6.2f ms\n",
                    T, m_ex, m_fa, e_ex, m_ex, em_fa, n_ex, n_fa, e_ex + m_ex + n_ex, all_fa);
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool do_bench = false, only = false, q8 = true;
    std::uint32_t C = 1024;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--bench") do_bench = true;
        else if (a == "--only") only = true;
        else if (a == "--hc") q8 = false;
        else if (a == "--chunk" && i + 1 < argc) C = static_cast<std::uint32_t>(std::stoul(argv[++i]));
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
    c::select_device(0);
    c::Stream stream;
    const cudaStream_t s = stream.get();
    if (!only) {
        test_sites(s);
        test_shared(s);
        test_hc(s);
        test_quantize(s);
        std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    }
    if (do_bench) bench(s, C, q8);
    } catch (const std::exception& e) {
        std::printf("ERROR %s\n", e.what());
        return 2;
    }
    return failures ? 1 : 0;
}
