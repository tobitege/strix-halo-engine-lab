// deepseek41 CUDA prefill-shape lane-dot kernels (cuda_ds41_prefill_kernels.hpp): bitwise against the decode kernels
// (core enqueue_f16_rows / enqueue_router / enqueue_engram, attn::enqueue_f16_rows) and the host references
// (f16_rows, reference_router) for 1..13 rows and large row counts, on synthetic data with edge values (NaN, Inf,
// subnormals, signed zeros, the largest half). With --bench: device times of the prefill uses at 2K / 8K / 32K rows
// (chunks of 1024 rows, as the prefill driver calls them), the decode kernels against these.

#include "strix/cuda/ds41_attention.hpp"
#include "strix/cuda/ds41_common.hpp"
#include "strix/cuda/ds41_core.hpp"
#include "strix/cuda/ds41_prefill_kernels.hpp"

#include "strix/ds41_layer.hpp"
#include "strix/ds41_moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace c = strix::ds41::cuda;
namespace pk = strix::ds41::cuda::pk;
namespace at = strix::ds41::cuda::attn;
namespace d = strix::ds41;

namespace {

int failures = 0;
std::mt19937_64 rng(20261020);

void expect_bits(const std::string& what, const float* got, const float* want, std::size_t n, bool quiet = false) {
    std::size_t bad = 0, first = n;
    for (std::size_t i = 0; i < n; ++i)
        if (std::memcmp(got + i, want + i, 4) != 0) {
            if (first == n) first = i;
            ++bad;
        }
    if (bad) {
        ++failures;
        std::printf("FAIL %s: %zu of %zu differ (first %zu: %.9g vs %.9g)\n", what.c_str(), bad, n, first, got[first], want[first]);
    } else if (!quiet) {
        std::printf("ok   %s (%zu values, fnv %016llx)\n", what.c_str(), n, static_cast<unsigned long long>(c::fnv1a(got, n)));
    }
}
template <class T>
void expect_equal(const std::string& what, const std::vector<T>& got, const std::vector<T>& want) {
    if (got.size() != want.size() || std::memcmp(got.data(), want.data(), got.size() * sizeof(T)) != 0) {
        ++failures;
        std::printf("FAIL %s\n", what.c_str());
    } else {
        std::printf("ok   %s (%zu values)\n", what.c_str(), got.size());
    }
}

// Halves: normal values of exponent [8, 18) (|w| ~ 2^-7 .. 2^3); with edges, 1 in 64 is special.
std::vector<std::uint16_t> random_halves(std::size_t n, bool edges) {
    static const std::uint16_t special[] = {0x7c00, 0xfc00, 0x7e00, 0x0001, 0x8003, 0x03ff, 0x8000, 0x0000, 0x7bff, 0xfbff, 0x0400};
    std::uniform_int_distribution<int> e(8, 17), m(0, 1023), s(0, 1), pick(0, 63), sp(0, 10);
    std::vector<std::uint16_t> w(n);
    for (auto& h : w)
        h = edges && pick(rng) == 0 ? special[sp(rng)] : static_cast<std::uint16_t>((s(rng) << 15) | (e(rng) << 10) | m(rng));
    return w;
}
// Floats in [-scale, scale); with edges, 1 in 64 is special (subnormals, zeros, NaN, Inf, huge).
std::vector<float> random_floats(std::size_t n, float scale, bool edges = false) {
    static const float special[] = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity(), 1e-40F, -3e-42F, 0.0F, -0.0F, 3e38F, -1e30F,
                                    std::numeric_limits<float>::denorm_min()};
    std::uniform_real_distribution<float> u(-scale, scale);
    std::uniform_int_distribution<int> pick(0, 63), sp(0, 9);
    std::vector<float> v(n);
    for (auto& x : v) x = edges && pick(rng) == 0 ? special[sp(rng)] : u(rng);
    return v;
}
// Host-checked data: finite, with subnormal and zero values but no overflow.
std::vector<float> host_floats(std::size_t n, float scale) {
    auto v = random_floats(n, scale);
    std::uniform_int_distribution<int> pick(0, 63);
    for (auto& x : v)
        if (pick(rng) == 0) x = pick(rng) < 32 ? 1e-40F : -0.0F;
    return v;
}
template <class T>
c::DeviceBuffer<T> to_device(const std::vector<T>& h) {
    c::DeviceBuffer<T> b(h.size());
    b.upload(h.data(), h.size());
    return b;
}

// ---- F16 projections ------------------------------------------------------------------------------------------------
struct F16Case {
    const char* name;
    std::uint32_t count, width, x_stride, out_stride;
    bool attn;  // the attention kernel takes it (width % 256)
};

void test_f16(const F16Case& k, const std::vector<std::uint32_t>& row_counts, bool edges, cudaStream_t s) {
    const std::uint32_t max_rows = *std::max_element(row_counts.begin(), row_counts.end());
    const auto w = random_halves(std::size_t{k.count} * k.width, edges);
    const auto x = edges ? random_floats(std::size_t{max_rows} * k.x_stride, 2.0F, true) : host_floats(std::size_t{max_rows} * k.x_stride, 2.0F);
    auto dw = to_device(w);
    auto dx = to_device(x);
    const std::size_t outn = std::size_t{max_rows} * k.out_stride;
    c::DeviceBuffer<float> a(outn), b(outn), e(outn);
    // host reference of the first rows (finite data only)
    const std::uint32_t host_rows = edges ? 0U : std::min<std::uint32_t>(max_rows, 13U);
    std::vector<float> ref(std::size_t{host_rows} * k.count);
    for (std::uint32_t r = 0; r < host_rows; ++r) d::f16_rows(w.data(), k.width, 0, k.count, x.data() + std::size_t{r} * k.x_stride, ref.data() + std::size_t{r} * k.count);
    for (const std::uint32_t rows : row_counts) {
        STRIX_CUDA_CHECK(cudaMemsetAsync(a.get(), 0xff, outn * 4U, s));
        STRIX_CUDA_CHECK(cudaMemsetAsync(b.get(), 0xff, outn * 4U, s));
        STRIX_CUDA_CHECK(cudaMemsetAsync(e.get(), 0xff, outn * 4U, s));
        pk::LaneDotView p;
        p.jobs[0] = pk::LaneDotJob{dw.get(), k.width, k.count, dx.get(), k.x_stride, a.get(), k.out_stride};
        p.job_count = 1;
        p.rows = rows;
        pk::enqueue_lane_dot(p, s);
        c::F16RowsView v;
        v.weight = {dw.get(), k.count, k.width};
        v.count = k.count;
        v.x = dx.get();
        v.rows = rows;
        v.x_stride = k.x_stride;
        v.out = b.get();
        v.out_stride = k.out_stride;
        c::enqueue_f16_rows(v, s);
        if (k.attn) {
            at::F16RowsView av{};
            av.rows = rows;
            av.jobs[av.job_count++] = at::F16Job{dw.get(), k.width, k.count, dx.get(), k.x_stride, e.get(), k.out_stride};
            at::enqueue_f16_rows(av, s);
        }
        STRIX_CUDA_CHECK(cudaStreamSynchronize(s));
        const std::size_t n = std::size_t{rows} * k.out_stride;
        const auto ga = a.download(n), gb = b.download(n);
        const std::string tag = std::string(k.name) + (edges ? " edges" : "") + " rows " + std::to_string(rows);
        expect_bits(tag + " == core f16_rows", ga.data(), gb.data(), n);  // also the gaps of out_stride (untouched)
        if (k.attn) {
            const auto ge = e.download(n);
            expect_bits(tag + " == attn f16_rows", ga.data(), ge.data(), n, true);
        }
        for (std::uint32_t r = 0; r < std::min(rows, host_rows); ++r)
            expect_bits(tag + " == host f16_rows row " + std::to_string(r), ga.data() + std::size_t{r} * k.out_stride,
                        ref.data() + std::size_t{r} * k.count, k.count, true);
    }
}

// Two jobs of one launch (the indexer's q_b and proj; the compressor's kv and gate) against the attention kernel.
void test_two_jobs(cudaStream_t s) {
    const std::uint32_t QA = 1280, H = 5120, QB = 4096, P = 32;
    for (const std::uint32_t rows : {1U, 5U, 13U, 1000U}) {
        const auto wq = random_halves(std::size_t{QB} * QA, true), wp = random_halves(std::size_t{P} * H, true);
        const auto xq = random_floats(std::size_t{rows} * QA, 3.0F, true), xp = random_floats(std::size_t{rows} * H, 3.0F, true);
        auto dwq = to_device(wq), dwp = to_device(wp);
        auto dxq = to_device(xq), dxp = to_device(xp);
        c::DeviceBuffer<float> a1(std::size_t{rows} * QB), a2(std::size_t{rows} * P), b1(std::size_t{rows} * QB), b2(std::size_t{rows} * P);
        pk::LaneDotView p;
        p.rows = rows;
        p.jobs[p.job_count++] = pk::LaneDotJob{dwq.get(), QA, QB, dxq.get(), QA, a1.get(), QB};
        p.jobs[p.job_count++] = pk::LaneDotJob{dwp.get(), H, P, dxp.get(), H, a2.get(), P};
        pk::enqueue_lane_dot(p, s);
        at::F16RowsView v{};
        v.rows = rows;
        v.jobs[v.job_count++] = at::F16Job{dwq.get(), QA, QB, dxq.get(), QA, b1.get(), QB};
        v.jobs[v.job_count++] = at::F16Job{dwp.get(), H, P, dxp.get(), H, b2.get(), P};
        at::enqueue_f16_rows(v, s);
        STRIX_CUDA_CHECK(cudaStreamSynchronize(s));
        const std::string tag = "index q_b + proj (2 jobs) rows " + std::to_string(rows);
        auto g1 = a1.download(), g2 = a2.download(), r1 = b1.download(), r2 = b2.download();
        expect_bits(tag + " q_b == attn", g1.data(), r1.data(), g1.size());
        expect_bits(tag + " proj == attn", g2.data(), r2.data(), g2.size());
    }
}

// ---- router -----------------------------------------------------------------------------------------------------------
void test_router(const std::vector<std::uint32_t>& row_counts, bool edges, cudaStream_t s) {
    const std::uint32_t hidden = 5120, experts = 384, top_k = 6;
    const std::uint32_t max_rows = *std::max_element(row_counts.begin(), row_counts.end());
    const auto w = random_floats(std::size_t{experts} * hidden, 0.02F, edges);
    auto x = edges ? random_floats(std::size_t{max_rows} * hidden, 1.0F, true) : host_floats(std::size_t{max_rows} * hidden, 1.0F);
    auto bias = random_floats(experts, 0.1F), bias_image = random_floats(experts, 0.1F);
    bias[300] = bias[7];
    if (max_rows > 4) std::fill(x.begin() + 4U * hidden, x.begin() + 5U * hidden, 0.0F);  // row 4: every score equal
    std::vector<std::uint8_t> image(max_rows);
    for (std::uint32_t r = 0; r < max_rows; ++r) image[r] = r % 3U == 1U;
    auto dw = to_device(w), dx = to_device(x), db = to_device(bias), dbi = to_device(bias_image);
    auto dimg = to_device(image);
    for (const std::uint32_t rows : row_counts) {
        c::DeviceBuffer<float> s1(std::size_t{rows} * experts), w1(std::size_t{rows} * top_k), s2(std::size_t{rows} * experts),
            w2(std::size_t{rows} * top_k);
        c::DeviceBuffer<std::uint32_t> i1(std::size_t{rows} * top_k), i2(std::size_t{rows} * top_k);
        c::RouterView v;
        v.x = dx.get();
        v.rows = rows;
        v.weight = {dw.get(), experts, hidden};
        v.bias = db.get();
        v.bias_image = dbi.get();
        v.image_rows = dimg.get();
        v.top_k = top_k;
        v.route_scale = 1.5F;
        v.scores = s1.get();
        v.ids = i1.get();
        v.weights = w1.get();
        pk::enqueue_router(v, s);
        v.scores = s2.get();
        v.ids = i2.get();
        v.weights = w2.get();
        c::enqueue_router(v, s);
        STRIX_CUDA_CHECK(cudaStreamSynchronize(s));
        const std::string tag = std::string("router") + (edges ? " edges" : "") + " rows " + std::to_string(rows);
        const auto gs = s1.download(), gw = w1.download(), cs = s2.download(), cw = w2.download();
        const auto gi = i1.download(), ci = i2.download();
        expect_bits(tag + " scores == core", gs.data(), cs.data(), gs.size());
        expect_equal(tag + " ids == core", gi, ci);
        expect_bits(tag + " weights == core", gw.data(), cw.data(), gw.size());
        if (!edges && rows <= 13U) {
            const auto ref = d::reference_router(std::span<const float>(x.data(), std::size_t{rows} * hidden), rows, hidden, w, experts, bias,
                                                 bias_image, std::span<const std::uint8_t>(image.data(), rows), top_k, 1.5F);
            expect_bits(tag + " scores == host", gs.data(), ref.scores.data(), ref.scores.size(), true);
            expect_equal(tag + " ids == host", gi, ref.ids);
            expect_bits(tag + " weights == host", gw.data(), ref.weights.data(), ref.weights.size(), true);
        }
    }
}

// ---- Engram -----------------------------------------------------------------------------------------------------------
void test_engram(const std::vector<std::uint32_t>& row_counts, cudaStream_t s) {
    const std::uint32_t hidden = 5120, cols = 24, width = cols * 256U;
    const std::uint32_t max_rows = *std::max_element(row_counts.begin(), row_counts.end());
    const auto w = random_halves(std::size_t{5U * hidden} * width, false);
    std::vector<std::uint8_t> bytes(std::size_t{max_rows} * cols * 264U);
    std::uniform_int_distribution<int> byte(0, 255), scale(118, 134);
    for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i % 264U < 256U ? byte(rng) : scale(rng));
    const auto stream0 = random_floats(std::size_t{max_rows} * 4U * hidden, 1.0F, true);
    const auto qn = random_floats(4U * hidden, 1.0F), kn = random_floats(4U * hidden, 1.0F);
    auto dw = to_device(w);
    auto db = to_device(bytes);
    auto dq = to_device(qn), dk = to_device(kn);
    for (const std::uint32_t rows : row_counts) {
        auto st1 = to_device(stream0), st2 = to_device(stream0);
        c::DeviceBuffer<float> kv1(std::size_t{rows} * 5U * hidden), kv2(std::size_t{rows} * 5U * hidden);
        c::DeviceBuffer<std::uint8_t> ws(c::engram_workspace_bytes(rows, cols, hidden));
        c::EngramView v;
        v.row_bytes = db.get();
        v.rows = rows;
        v.cols = cols;
        v.hidden = hidden;
        v.kv = {dw.get(), 5U * hidden, width};
        v.q_norm = dq.get();
        v.k_norm = dk.get();
        v.workspace = ws.get();
        v.stream = st1.get();
        v.kv_out = kv1.get();
        pk::enqueue_engram(v, s);
        v.stream = st2.get();
        v.kv_out = kv2.get();
        c::enqueue_engram(v, s);
        STRIX_CUDA_CHECK(cudaStreamSynchronize(s));
        const std::string tag = "engram rows " + std::to_string(rows);
        auto a = kv1.download(), b = kv2.download();
        expect_bits(tag + " kv == core", a.data(), b.data(), a.size());
        auto sa = st1.download(), sb = st2.download();
        expect_bits(tag + " stream == core", sa.data(), sb.data(), sa.size());
    }
}

// ---- bench ------------------------------------------------------------------------------------------------------------
float time_ms(cudaStream_t s, const std::function<void()>& body, int reps) {
    body();  // warm up
    c::EventTimer t;
    t.start(s);
    for (int i = 0; i < reps; ++i) body();
    return t.stop(s) / static_cast<float>(reps);
}

void bench(cudaStream_t s) {
    constexpr std::uint32_t C = 1024;
    const std::uint32_t H = 5120, QA = 1280;
    struct Use {
        const char* name;
        std::uint32_t count, width;
    };
    // weights and one chunk of rows (every chunk of a call reads the same buffers: the driver's access pattern)
    const auto wbig = random_halves(std::size_t{5U * H} * 6144U, false);
    auto dwe = to_device(wbig);
    const auto xbig = random_floats(std::size_t{C} * 6144U, 1.0F);
    auto dx = to_device(xbig);
    c::DeviceBuffer<float> out(std::size_t{C} * 5U * H);
    const auto wr = random_floats(std::size_t{384} * H, 0.02F);
    auto dwr = to_device(wr);
    const auto bias = random_floats(384, 0.1F);
    auto db = to_device(bias);
    c::DeviceBuffer<std::uint32_t> ids(std::size_t{C} * 6U);
    c::DeviceBuffer<float> rw(std::size_t{C} * 6U);
    const Use uses[] = {{"engram_kv", 5U * H, 6144U}, {"compress kv+gate", 512U, H}, {"index q_b+proj", 4096U, QA}};
    std::printf("bench: device ms per call over T rows in chunks of %u (old = decode kernel, new = prefill kernel)\n", C);
    for (const std::uint32_t T : {2048U, 8192U, 32768U}) {
        const int reps = T >= 32768U ? 1 : 3;
        for (const auto& u : uses) {
            const bool two = u.name[0] != 'e';
            const auto run_old = [&] {
                for (std::uint32_t r0 = 0; r0 < T; r0 += C) {
                    if (u.name[0] == 'e') {
                        c::F16RowsView v;
                        v.weight = {dwe.get(), u.count, u.width};
                        v.count = u.count;
                        v.x = dx.get();
                        v.rows = C;
                        v.out = out.get();
                        c::enqueue_f16_rows(v, s);
                    } else {
                        at::F16RowsView v{};
                        v.rows = C;
                        v.jobs[v.job_count++] = at::F16Job{dwe.get(), u.width, u.count, dx.get(), u.width, out.get(), u.count};
                        if (u.name[0] == 'c')
                            v.jobs[v.job_count++] = at::F16Job{dwe.get() + std::size_t{u.count} * u.width, u.width, u.count, dx.get(), u.width,
                                                               out.get() + std::size_t{C} * u.count, u.count};
                        else
                            v.jobs[v.job_count++] = at::F16Job{dwe.get() + std::size_t{u.count} * u.width, H, 32, dx.get(), H,
                                                               out.get() + std::size_t{C} * u.count, 32};
                        at::enqueue_f16_rows(v, s);
                    }
                }
            };
            const auto run_new = [&] {
                for (std::uint32_t r0 = 0; r0 < T; r0 += C) {
                    pk::LaneDotView p;
                    p.rows = C;
                    p.jobs[p.job_count++] = pk::LaneDotJob{dwe.get(), u.width, u.count, dx.get(), u.width, out.get(), u.count};
                    if (two) {
                        if (u.name[0] == 'c')
                            p.jobs[p.job_count++] = pk::LaneDotJob{dwe.get() + std::size_t{u.count} * u.width, u.width, u.count, dx.get(), u.width,
                                                                   out.get() + std::size_t{C} * u.count, u.count};
                        else
                            p.jobs[p.job_count++] = pk::LaneDotJob{dwe.get() + std::size_t{u.count} * u.width, H, 32, dx.get(), H,
                                                                   out.get() + std::size_t{C} * u.count, 32};
                    }
                    pk::enqueue_lane_dot(p, s);
                }
            };
            const float a = time_ms(s, run_old, reps), b = time_ms(s, run_new, reps);
            std::printf("  T %5u %-17s old %9.2f ms  new %8.2f ms  x%.1f\n", T, u.name, a, b, a / b);
        }
        // router: logits + selection
        const auto router = [&](bool fresh) {
            for (std::uint32_t r0 = 0; r0 < T; r0 += C) {
                c::RouterView v;
                v.x = dx.get();
                v.rows = C;
                v.weight = {dwr.get(), 384, H};
                v.bias = db.get();
                v.top_k = 6;
                v.scores = out.get();
                v.ids = ids.get();
                v.weights = rw.get();
                if (fresh) pk::enqueue_router(v, s);
                else c::enqueue_router(v, s);
            }
        };
        const float a = time_ms(s, [&] { router(false); }, reps), b = time_ms(s, [&] { router(true); }, reps);
        std::printf("  T %5u %-17s old %9.2f ms  new %8.2f ms  x%.1f\n", T, "router", a, b, a / b);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    const bool only_bench = argc > 2 && std::string(argv[2]) == "--only";
    c::select_device(0);
    c::Stream stream;
    const cudaStream_t s = stream.get();
    if (!only_bench) {
        std::vector<std::uint32_t> small;
        for (std::uint32_t r = 1; r <= 13; ++r) small.push_back(r);
        auto with_large = small;
        with_large.push_back(1000);
        with_large.push_back(4096);
        // odd geometry: counts not multiples of the tile, strides wider than the rows
        test_f16({"odd 200x512 strided", 200, 512, 520, 210, false}, with_large, false, s);
        test_f16({"odd 200x512 strided", 200, 512, 520, 210, false}, small, true, s);
        test_f16({"compress 512x5120", 512, 5120, 5120, 512, true}, with_large, false, s);
        test_f16({"compress 512x5120", 512, 5120, 5120, 512, true}, {1, 7, 13, 1000}, true, s);
        test_f16({"index q_b 4096x1280", 4096, 1280, 1280, 4096, true}, {1, 2, 9, 13, 1000, 4096}, true, s);
        test_f16({"index proj 32x5120", 32, 5120, 5120, 32, true}, with_large, false, s);
        test_f16({"engram_kv 25600x6144", 25600, 6144, 6144, 25600, false}, {1, 2, 3, 8, 9, 13, 1000, 4096}, false, s);
        test_f16({"engram_kv 25600x6144", 25600, 6144, 6144, 25600, false}, {5, 1024}, true, s);
        test_two_jobs(s);
        test_router(with_large, false, s);
        test_router({1, 3, 8, 13, 1000}, true, s);
        test_engram({1, 2, 7, 8, 13, 1000}, s);
        std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    }
    if (do_bench) bench(s);
    return failures ? 1 : 0;
}
