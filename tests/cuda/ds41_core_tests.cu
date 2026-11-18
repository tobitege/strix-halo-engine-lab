// deepseek41 CUDA layer core: every kernel bitwise against its host reference (ds41_layer.hpp / ds41_moe.hpp /
// ds41_formats.hpp) on synthetic inputs. Short: runs in seconds and allocates a few hundred MB at most.

#include "strix/cuda/ds41_common.hpp"
#include "strix/cuda/ds41_core.hpp"

#include "strix/ds41_formats.hpp"
#include "strix/ds41_layer.hpp"
#include "strix/ds41_moe.hpp"

#include <algorithm>
#include <array>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace c = strix::ds41::cuda;
namespace d = strix::ds41;

namespace {

int failures = 0;

void expect_bits(const std::string& what, const float* got, const float* want, std::size_t n) {
    std::size_t bad = 0, first = n;
    for (std::size_t i = 0; i < n; ++i)
        if (std::memcmp(got + i, want + i, 4) != 0) {
            if (first == n) first = i;
            ++bad;
        }
    if (bad) {
        ++failures;
        std::printf("FAIL %s: %zu of %zu differ (first %zu: %.9g vs %.9g)\n", what.c_str(), bad, n, first, got[first], want[first]);
    } else {
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

std::mt19937_64 rng(20261020);

std::uint16_t random_half_scale(int min_exp, int max_exp) {  // positive or negative finite normal half
    std::uniform_int_distribution<int> e(min_exp, max_exp), m(0, 1023), s(0, 1);
    return static_cast<std::uint16_t>((s(rng) << 15) | (e(rng) << 10) | m(rng));
}
std::vector<std::uint8_t> random_q8(std::size_t rows, std::size_t width) {
    std::vector<std::uint8_t> w(rows * width / 32 * 34);
    std::uniform_int_distribution<int> code(-127, 127);
    for (std::size_t b = 0; b < w.size(); b += 34) {
        const std::uint16_t h = random_half_scale(3, 12);
        std::memcpy(&w[b], &h, 2);
        for (int j = 0; j < 32; ++j) w[b + 2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code(rng)));
    }
    return w;
}
std::vector<float> random_floats(std::size_t n, float scale) {
    std::normal_distribution<float> g(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = g(rng);
    return v;
}

// A8 of rows on the host, concatenated.
struct HostA8 {
    std::vector<std::int8_t> codes;
    std::vector<float> deltas;
    std::vector<std::int32_t> sums;
};
HostA8 host_a8(const std::vector<float>& x, std::size_t rows, std::size_t width) {
    HostA8 h;
    for (std::size_t r = 0; r < rows; ++r) {
        const auto a = d::quantize_a8(std::span<const float>(x.data() + r * width, width));
        h.codes.insert(h.codes.end(), a.codes.begin(), a.codes.end());
        h.deltas.insert(h.deltas.end(), a.deltas.begin(), a.deltas.end());
        h.sums.insert(h.sums.end(), a.sums.begin(), a.sums.end());
    }
    return h;
}

void test_quantize_a8(cudaStream_t s) {
    const std::uint32_t rows = 5, width = 5120;
    auto x = random_floats(std::size_t{rows} * width, 3.0F);
    // special groups: zero, tiny, a NaN, an infinity, signed zeros, exact halves
    for (int j = 0; j < 32; ++j) x[32 + j] = 0.0F;
    for (int j = 0; j < 32; ++j) x[64 + j] = 1e-37F * static_cast<float>(j % 3);
    x[100] = NAN;
    x[200] = INFINITY;
    x[300] = -0.0F;
    for (int j = 0; j < 32; ++j) x[320 + j] = 127.0F * (static_cast<float>(j) + 0.5F) / 32.0F;
    c::DeviceBuffer<float> dx(x.size());
    dx.upload(x);
    c::A8Buffer a;
    a.resize(rows, width);
    c::enqueue_quantize_a8({dx.get(), rows, width, 0, a.view()}, s);
    STRIX_CUDA_CHECK(cudaStreamSynchronize(s));
    const auto h = host_a8(x, rows, width);
    expect_equal("quantize_a8 codes", a.codes.download(), h.codes);
    auto gd = a.deltas.download();
    expect_bits("quantize_a8 deltas", gd.data(), h.deltas.data(), gd.size());
    expect_equal("quantize_a8 sums", a.sums.download(), h.sums);
}

// Plain, grouped and K-split Q8_0 sites at 1..11 rows.
void test_q8_0_a8(cudaStream_t s) {
    struct Case {
        const char* name;
        std::uint32_t width, count, per_group, parts;
    };
    const Case cases[] = {{"q_a 5120->1280", 5120, 1280, 0, 1}, {"q_b 1280->2048", 1280, 2048, 0, 1},
                          {"o_a grouped 8x(4096->1024)", 4096, 8192, 1024, 1}, {"o_b k-split 2 8192->640", 8192, 640, 0, 2}};
    for (const auto& k : cases) {
        const auto w = random_q8(k.count, k.width);
        c::DeviceBuffer<std::uint8_t> dw(w.size());
        dw.upload(w);
        const std::uint32_t groups = k.per_group ? k.count / k.per_group : 1U, xw = groups * k.width;
        for (const std::uint32_t rows : {1U, 3U, 8U, 11U}) {
            const auto x = random_floats(std::size_t{rows} * (k.parts > 1 ? k.width : xw), 1.0F);
            const std::uint32_t xin = k.parts > 1 ? k.width : xw;
            // host
            std::vector<float> want(std::size_t{rows} * k.count);
            for (std::uint32_t r = 0; r < rows; ++r) {
                const auto a = d::quantize_a8(std::span<const float>(x.data() + std::size_t{r} * xin, xin));
                if (k.parts > 1)
                    d::q8_0_a8_split_rows(w.data(), k.width, k.count, k.parts, a, want.data() + std::size_t{r} * k.count);
                else
                    d::q8_0_a8_rows(w.data(), k.width, 0, k.count, k.per_group ? k.per_group : k.count, a,
                                    want.data() + std::size_t{r} * k.count);
            }
            // device
            c::DeviceBuffer<float> dx(x.size()), dout(want.size());
            dx.upload(x);
            c::A8Buffer a;
            a.resize(rows, xin);
            c::enqueue_quantize_a8({dx.get(), rows, xin, 0, a.view()}, s);
            if (k.parts > 1) {
                const std::uint32_t part = k.width / k.parts;
                for (std::uint32_t p = 0; p < k.parts; ++p) {
                    c::A8Rows xs = a.view();
                    xs.codes += p * part;
                    xs.deltas += p * part / 32U;
                    xs.sums += p * part / 32U;
                    xs.width = part;
                    c::Q8A8View v;
                    v.weight = {dw.get(), k.count, k.width};
                    v.count = k.count;
                    v.k_offset = p * part;
                    v.width = part;
                    v.x = xs;
                    v.out = dout.get();
                    v.accumulate = p > 0;
                    c::enqueue_q8_0_a8(v, s);
                }
            } else {
                c::Q8A8View v;
                v.weight = {dw.get(), k.count, k.width};
                v.count = k.count;
                v.per_group = k.per_group;
                v.x = a.view();
                v.out = dout.get();
                c::enqueue_q8_0_a8(v, s);
            }
            STRIX_CUDA_CHECK(cudaStreamSynchronize(s));
            const auto got = dout.download();
            expect_bits(std::string(k.name) + " rows " + std::to_string(rows), got.data(), want.data(), want.size());
        }
    }
}

// The tensor-core form: plain, grouped and K-split sites, splits 1/2/4, decode and prefill row counts.
void test_q8_0_a8_tiled(cudaStream_t s) {
    struct Case {
        const char* name;
        std::uint32_t width, count, per_group, parts;
    };
    const Case cases[] = {{"tiled q_a+kv 5120->1792", 5120, 1792, 0, 1}, {"tiled q_b 1280->2048", 1280, 2048, 0, 1},
                          {"tiled o_a grouped 8x(4096->1024)", 4096, 8192, 1024, 1}, {"tiled o_b k-split 2 8192->640", 8192, 640, 0, 2},
                          {"tiled down 2304->512", 2304, 512, 0, 1}};
    for (const auto& k : cases) {
        const auto w = random_q8(k.count, k.width);
        c::DeviceBuffer<std::uint8_t> dw(w.size());
        dw.upload(w);
        c::Q8TiledBuffer tiled;
        tiled.repack(dw.get(), k.count, k.width, s);
        const std::uint32_t groups = k.per_group ? k.count / k.per_group : 1U;
        const std::uint32_t xin = k.parts > 1 ? k.width : groups * k.width;
        for (const std::uint32_t rows : {1U, 3U, 8U, 11U, 70U}) {
            const auto x = random_floats(std::size_t{rows} * xin, 1.0F);
            std::vector<float> want(std::size_t{rows} * k.count);
            for (std::uint32_t r = 0; r < rows; ++r) {
                const auto a = d::quantize_a8(std::span<const float>(x.data() + std::size_t{r} * xin, xin));
                if (k.parts > 1)
                    d::q8_0_a8_split_rows(w.data(), k.width, k.count, k.parts, a, want.data() + std::size_t{r} * k.count);
                else
                    d::q8_0_a8_rows(w.data(), k.width, 0, k.count, k.per_group ? k.per_group : k.count, a,
                                    want.data() + std::size_t{r} * k.count);
            }
            c::DeviceBuffer<float> dx(x.size()), dout(want.size());
            dx.upload(x);
            c::A8Buffer a;
            a.resize(rows, xin);
            c::enqueue_quantize_a8({dx.get(), rows, xin, 0, a.view()}, s);
            for (const std::uint32_t split : {1U, 2U, 4U}) {
                if (rows == 70U && split != 1U) continue;
                c::DeviceBuffer<float> scratch(c::q8_0_a8_split_scratch_floats(k.count, rows, split));
                const std::uint32_t part = k.width / k.parts;
                for (std::uint32_t p = 0; p < k.parts; ++p) {
                    c::Q8A8View v;
                    v.tiled = tiled.view();
                    v.count = k.count;
                    v.per_group = k.per_group;
                    v.split = split;
                    v.split_scratch = scratch.get();
                    v.split_scratch_floats = scratch.size();
                    v.x = a.view();
                    if (k.parts > 1) {
                        v.x.codes += p * part;
                        v.x.deltas += p * part / 32U;
                        v.x.sums += p * part / 32U;
                        v.x.width = part;
                        v.k_offset = p * part;
                        v.width = part;
                        v.accumulate = p > 0;
                    }
                    v.out = dout.get();
                    c::enqueue_q8_0_a8(v, s);
                }
                STRIX_CUDA_CHECK(cudaStreamSynchronize(s));
                const auto got = dout.download();
                expect_bits(std::string(k.name) + " rows " + std::to_string(rows) + " split " + std::to_string(split), got.data(),
                            want.data(), want.size());
            }
        }
    }
}

std::vector<std::uint16_t> random_f16(std::size_t n, int min_exp, int max_exp) {
    std::vector<std::uint16_t> v(n);
    for (auto& h : v) h = random_half_scale(min_exp, max_exp);
    return v;
}
template <class T>
c::DeviceBuffer<T> to_device(const std::vector<T>& v) {
    c::DeviceBuffer<T> d(v.size());
    d.upload(v);
    return d;
}

// collapse + RMSNorm + A8 (the site input), plain norms in place, collapse only.
void test_norm(cudaStream_t s) {
    const std::uint32_t rows = 3, n = 5120;
    const auto stream = random_floats(std::size_t{rows} * 4U * n, 2.0F);
    auto mix = random_floats(std::size_t{rows} * 24U, 0.5F);
    const auto w = random_floats(n, 1.0F);
    auto dstream = to_device(stream), dmix = to_device(mix), dw = to_device(w);
    c::DeviceBuffer<float> dout(std::size_t{rows} * n), dcol(std::size_t{rows} * n);
    c::A8Buffer a;
    a.resize(rows, n);
    for (const bool identity : {false, true}) {
        c::NormView v;
        v.hc_stream = dstream.get();
        v.pre = identity ? nullptr : dmix.get();
        v.pre_stride = 24;
        v.weight = dw.get();
        v.rows = rows;
        v.n = n;
        v.out = dout.get();
        v.collapsed = dcol.get();
        v.a8 = a.view();
        c::enqueue_norm(v, s);
        std::vector<float> want_col(std::size_t{rows} * n), want(std::size_t{rows} * n);
        HostA8 want_a8;
        for (std::uint32_t r = 0; r < rows; ++r) {
            std::array<float, 4> pre = d::hc_identity_pre();
            if (!identity) std::copy_n(mix.begin() + r * 24, 4, pre.begin());
            std::span<float> col(want_col.data() + std::size_t{r} * n, n);
            d::hc_collapse(std::span<const float>(stream.data() + std::size_t{r} * 4U * n, 4U * n), pre, col);
            const auto q = d::collapse_norm_a8(std::span<const float>(stream.data() + std::size_t{r} * 4U * n, 4U * n), pre, w, 1e-20F,
                                               std::span<float>(want.data() + std::size_t{r} * n, n));
            want_a8.codes.insert(want_a8.codes.end(), q.codes.begin(), q.codes.end());
            want_a8.deltas.insert(want_a8.deltas.end(), q.deltas.begin(), q.deltas.end());
        }
        const std::string tag = identity ? " (identity pre)" : "";
        auto got_col = dcol.download(), got = dout.download();
        expect_bits("hc_collapse" + tag, got_col.data(), want_col.data(), want_col.size());
        expect_bits("collapse + rms_norm" + tag, got.data(), want.data(), want.size());
        expect_equal("collapse + rms_norm A8 codes" + tag, a.codes.download(), want_a8.codes);
        auto gd = a.deltas.download();
        expect_bits("collapse + rms_norm A8 deltas" + tag, gd.data(), want_a8.deltas.data(), gd.size());
    }
    // q_a_norm in place (1280) at 5 rows
    const std::uint32_t qn = 1280, qrows = 5;
    const auto x = random_floats(std::size_t{qrows} * qn, 30.0F);
    const auto qw = random_floats(qn, 1.0F);
    auto dx = to_device(x), dqw = to_device(qw);
    c::NormView v;
    v.x = dx.get();
    v.weight = dqw.get();
    v.rows = qrows;
    v.n = qn;
    v.out = dx.get();
    c::enqueue_norm(v, s);
    std::vector<float> want(x.size());
    for (std::uint32_t r = 0; r < qrows; ++r)
        d::rms_norm(std::span<const float>(x.data() + std::size_t{r} * qn, qn), qw, 1e-20F, std::span<float>(want.data() + std::size_t{r} * qn, qn));
    auto got = dx.download();
    expect_bits("rms_norm 1280 in place", got.data(), want.data(), want.size());
}

void test_hc(cudaStream_t s) {
    const std::uint32_t hidden = 5120, n = 4U * hidden;
    const auto fn = random_f16(std::size_t{24} * n, 3, 10);  // ~2^-12 .. 2^-5
    const auto base = random_floats(24, 0.5F), scale = random_floats(3, 1.0F);
    auto dfn = to_device(fn);
    auto dbase = to_device(base), dscale = to_device(scale);
    for (const std::uint32_t rows : {1U, 4U, 9U}) {
        const auto stream = random_floats(std::size_t{rows} * n, 1.5F);
        auto dstream = to_device(stream);
        c::DeviceBuffer<float> dmix(std::size_t{rows} * 24U), dpart(c::hc_mix_workspace_bytes(rows, hidden) / 4U);
        c::HcMixView v;
        v.stream = dstream.get();
        v.rows = rows;
        v.hidden = hidden;
        v.fn = dfn.get();
        v.base = dbase.get();
        v.scale = dscale.get();
        v.mixes = dmix.get();
        v.partial = dpart.get();
        c::enqueue_hc_mixes(v, s);
        std::vector<float> want(std::size_t{rows} * 24U);
        const d::HcSiteWeights site{fn, base, scale};
        for (std::uint32_t r = 0; r < rows; ++r) {
            const auto m = d::hc_mixes(std::span<const float>(stream.data() + std::size_t{r} * n, n), site, 20, 1e-6F, 1e-20F);
            std::copy(m.pre.begin(), m.pre.end(), want.begin() + r * 24);
            std::copy(m.post.begin(), m.post.end(), want.begin() + r * 24 + 4);
            std::copy(m.comb.begin(), m.comb.end(), want.begin() + r * 24 + 8);
        }
        auto got = dmix.download();
        expect_bits("hc_mixes + sinkhorn rows " + std::to_string(rows), got.data(), want.data(), want.size());
        // expansion with these mixes, in place on the stream
        const auto sub = random_floats(std::size_t{rows} * hidden, 1.0F);
        auto dsub = to_device(sub);
        c::enqueue_hc_expand({dsub.get(), 0, dstream.get(), dmix.get(), rows, hidden, dstream.get()}, s);
        std::vector<float> want_s(stream.size());
        for (std::uint32_t r = 0; r < rows; ++r) {
            d::HcMix m;
            std::copy_n(want.begin() + r * 24, 4, m.pre.begin());
            std::copy_n(want.begin() + r * 24 + 4, 4, m.post.begin());
            std::copy_n(want.begin() + r * 24 + 8, 16, m.comb.begin());
            d::hc_expand(std::span<const float>(sub.data() + std::size_t{r} * hidden, hidden),
                         std::span<const float>(stream.data() + std::size_t{r} * n, n), m,
                         std::span<float>(want_s.data() + std::size_t{r} * n, n));
        }
        auto got_s = dstream.download();
        expect_bits("hc_expand in place rows " + std::to_string(rows), got_s.data(), want_s.data(), want_s.size());
    }
}

void test_router(cudaStream_t s) {
    const std::uint32_t hidden = 5120, experts = 384, top_k = 6, rows = 5;
    const auto w = random_floats(std::size_t{experts} * hidden, 0.02F);
    auto x = random_floats(std::size_t{rows} * hidden, 1.0F);
    auto bias = random_floats(experts, 0.1F), bias_image = random_floats(experts, 0.1F);
    // ties: experts 7 and 300 get the same key, row 4 is all zero (every score equal)
    bias[300] = bias[7];
    for (std::uint32_t i = 0; i < hidden; ++i) x[4U * hidden + i] = 0.0F;
    std::vector<std::uint8_t> image{0, 1, 0, 0, 1};
    auto dw = to_device(w), dx = to_device(x), db = to_device(bias), dbi = to_device(bias_image);
    auto dimg = to_device(image);
    c::DeviceBuffer<float> scores(std::size_t{rows} * experts), weights(std::size_t{rows} * top_k);
    c::DeviceBuffer<std::uint32_t> ids(std::size_t{rows} * top_k);
    c::RouterView v;
    v.x = dx.get();
    v.rows = rows;
    v.weight = {dw.get(), experts, hidden};
    v.bias = db.get();
    v.bias_image = dbi.get();
    v.image_rows = dimg.get();
    v.top_k = top_k;
    v.route_scale = 1.5F;
    v.scores = scores.get();
    v.ids = ids.get();
    v.weights = weights.get();
    c::enqueue_router(v, s);
    const auto ref = d::reference_router(x, rows, hidden, w, experts, bias, bias_image, image, top_k, 1.5F);
    auto gs = scores.download(), gw = weights.download();
    expect_bits("router scores", gs.data(), ref.scores.data(), ref.scores.size());
    expect_equal("router ids", ids.download(), ref.ids);
    expect_bits("router weights", gw.data(), ref.weights.data(), ref.weights.size());
}

void test_shared_expert(cudaStream_t s) {
    const std::uint32_t hidden = 5120, inter = 2304;
    const auto gate = random_q8(inter, hidden), up = random_q8(inter, hidden), down = random_q8(hidden, inter);
    auto dg = to_device(gate), du = to_device(up), dd = to_device(down);
    std::vector<std::uint8_t> gate_up(gate);
    gate_up.insert(gate_up.end(), up.begin(), up.end());
    auto dgu = to_device(gate_up);
    c::Q8TiledBuffer tgu, tdown;
    tgu.repack(dgu.get(), 2U * inter, hidden, s);
    tdown.repack(dd.get(), hidden, inter, s);
    for (const std::uint32_t rows : {1U, 3U}) {
        const auto x = random_floats(std::size_t{rows} * hidden, 0.002F);
        auto dx = to_device(x);
        c::A8Buffer a;
        a.resize(rows, hidden);
        c::enqueue_quantize_a8({dx.get(), rows, hidden, 0, a.view()}, s);
        c::DeviceBuffer<std::uint8_t> ws(c::shared_expert_workspace_bytes(rows, inter));
        c::DeviceBuffer<float> out(std::size_t{rows} * hidden);
        c::SharedExpertView v;
        v.gate = {dg.get(), inter, hidden};
        v.up = {du.get(), inter, hidden};
        v.down = {dd.get(), hidden, inter};
        v.x = a.view();
        v.out = out.get();
        v.workspace = ws.get();
        c::enqueue_shared_expert(v, s);
        d::ExpertTensors t;
        t.gate_up_format = t.down_format = d::WeightFormat::q8_0;
        t.gate = gate;
        t.up = up;
        t.down = down;
        t.hidden = hidden;
        t.inter = inter;
        t.limit = 10.0F;
        std::vector<float> want;
        for (std::uint32_t r = 0; r < rows; ++r) {
            const std::vector<d::A8Activation> xs{d::quantize_a8(std::span<const float>(x.data() + std::size_t{r} * hidden, hidden))};
            const std::vector<std::uint32_t> zero{0U};
            const std::vector<float> one{1.0F}, none(hidden, 0.0F);
            const auto raw = d::reference_window(t, xs, d::build_window_routes(zero, 1U, 1U, 1U), one, none, 1U).raw;
            want.insert(want.end(), raw.begin(), raw.end());
        }
        auto got = out.download();
        expect_bits("shared expert rows " + std::to_string(rows), got.data(), want.data(), want.size());
        c::SharedExpertView tv;
        tv.gate_up_tiled = tgu.view();
        tv.down_tiled = tdown.view();
        tv.x = a.view();
        tv.out = out.get();
        tv.workspace = ws.get();
        out.zero();
        c::enqueue_shared_expert(tv, s);
        got = out.download();
        expect_bits("shared expert tiled (gate+up one site) rows " + std::to_string(rows), got.data(), want.data(), want.size());
    }
}

void test_embed_engram(cudaStream_t s) {
    // embedding
    const std::uint32_t vocab = 1000, hidden = 1280, rows = 3;
    const auto table = random_f16(std::size_t{vocab} * hidden, 5, 20);
    const std::vector<std::uint32_t> tokens{0, 999, 17};
    auto dt = to_device(table);
    auto dtok = to_device(tokens);
    c::DeviceBuffer<float> stream(std::size_t{rows} * 4U * hidden), pre(std::size_t{rows} * 4U);
    c::enqueue_embed({{dt.get(), vocab, hidden}, dtok.get(), rows, stream.get(), pre.get()}, s);
    std::vector<float> want(std::size_t{rows} * 4U * hidden);
    for (std::uint32_t r = 0; r < rows; ++r)
        d::embed_token(table.data(), hidden, tokens[r], std::span<float>(want.data() + std::size_t{r} * 4U * hidden, 4U * hidden));
    auto got = stream.download();
    expect_bits("embedding", got.data(), want.data(), want.size());
    // Engram: gather from a small table, decode, projection, injection (hidden 1280 keeps the F16 weight small)
    const std::uint32_t cols = 24, table_rows = 512;
    std::vector<std::uint8_t> etable(std::size_t{table_rows} * 264U);
    std::uniform_int_distribution<int> code(0, 255), sc(118, 126);
    for (std::size_t r = 0; r < table_rows; ++r) {
        for (int j = 0; j < 256; ++j) {
            int cv = code(rng);
            if ((cv & 0x7f) == 0x7f) cv &= 0xfe;  // no NaN codes
            etable[r * 264 + j] = static_cast<std::uint8_t>(cv);
        }
        for (int j = 0; j < 8; ++j) etable[r * 264 + 256 + j] = static_cast<std::uint8_t>(sc(rng));
    }
    std::vector<std::uint64_t> eids(std::size_t{rows} * cols);
    std::uniform_int_distribution<std::uint64_t> pick(0, table_rows - 1);
    for (auto& e : eids) e = pick(rng);
    const auto kv = random_f16(std::size_t{5} * hidden * cols * 256U, 5, 12);
    const auto qn = random_floats(4U * hidden, 1.0F), kn = random_floats(4U * hidden, 1.0F);
    auto detable = to_device(etable);
    auto deids = to_device(eids);
    auto dkv = to_device(kv);
    auto dqn = to_device(qn), dkn = to_device(kn);
    c::DeviceBuffer<std::uint8_t> erows(std::size_t{rows} * cols * 264U), ews(c::engram_workspace_bytes(rows, cols, hidden));
    c::DeviceBuffer<float> kv_out(std::size_t{rows} * 5U * hidden);
    c::enqueue_engram_gather(detable.get(), deids.get(), rows * cols, erows.get(), s);
    c::EngramView ev;
    ev.row_bytes = erows.get();
    ev.rows = rows;
    ev.cols = cols;
    ev.hidden = hidden;
    ev.kv = {dkv.get(), 5U * hidden, cols * 256U};
    ev.q_norm = dqn.get();
    ev.k_norm = dkn.get();
    ev.stream = stream.get();
    ev.kv_out = kv_out.get();
    ev.workspace = ews.get();
    c::enqueue_engram(ev, s);
    std::vector<float> want_kv(std::size_t{rows} * 5U * hidden);
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::vector<std::uint8_t> rb(std::size_t{cols} * 264U);
        for (std::uint32_t k = 0; k < cols; ++k)
            std::copy_n(etable.data() + eids[r * cols + k] * 264U, 264, rb.data() + std::size_t{k} * 264U);
        std::vector<float> emb(std::size_t{cols} * 256U);
        d::engram_rows_to_bf16(rb, emb);
        d::f16_rows(kv.data(), emb.size(), 0, 5U * hidden, emb.data(), want_kv.data() + std::size_t{r} * 5U * hidden);
        d::engram_inject(std::span<float>(want.data() + std::size_t{r} * 4U * hidden, 4U * hidden),
                         std::span<const float>(want_kv.data() + std::size_t{r} * 5U * hidden, 5U * hidden), qn, kn, 1e-20F);
    }
    auto got_kv = kv_out.download(), got_s = stream.download();
    expect_bits("engram_kv projection (F16 lane dot)", got_kv.data(), want_kv.data(), want_kv.size());
    expect_bits("engram injection", got_s.data(), want.data(), want.size());
}

void test_argmax(cudaStream_t s) {
    const std::uint32_t vocab = 129280, rows = 3;
    auto logits = random_floats(std::size_t{rows} * vocab, 3.0F);
    // row 0: a tie of the maximum across the two vocabulary halves; row 2: a NaN
    logits[100] = 50.0F;
    logits[vocab / 2 + 5] = 50.0F;
    logits[2U * vocab + 7] = NAN;
    auto dl = to_device(logits);
    c::DeviceBuffer<d::ArgmaxCandidate> out(rows);
    c::enqueue_argmax({dl.get(), rows, vocab, 0, 0, out.get()}, s);
    auto got = out.download();
    for (std::uint32_t r = 0; r < rows; ++r) {
        const auto s0 = d::vocab_slice(vocab, 2, 0), s1 = d::vocab_slice(vocab, 2, 1);
        const float* l = logits.data() + std::size_t{r} * vocab;
        const auto want = d::combine_argmax(d::argmax_slice(std::span<const float>(l + s0.first, s0.count), s0.first),
                                            d::argmax_slice(std::span<const float>(l + s1.first, s1.count), s1.first));
        const bool same = got[r].index == want.index && got[r].finite == want.finite && std::memcmp(&got[r].value, &want.value, 4) == 0;
        if (!same) {
            ++failures;
            std::printf("FAIL argmax row %u: %u/%u vs %u/%u\n", r, got[r].index, got[r].finite, want.index, want.finite);
        } else {
            std::printf("ok   argmax row %u (index %u, finite %u)\n", r, want.index, want.finite);
        }
    }
}

}  // namespace

int main() try {
    const auto dev = c::select_device(0);
    std::printf("device %s (sm_%d%d, %d SMs)\n", dev.name.c_str(), dev.major, dev.minor, dev.sms);
    c::Stream stream;
    test_quantize_a8(stream.get());
    test_q8_0_a8(stream.get());
    test_q8_0_a8_tiled(stream.get());
    test_norm(stream.get());
    test_hc(stream.get());
    test_router(stream.get());
    test_shared_expert(stream.get());
    test_embed_engram(stream.get());
    test_argmax(stream.get());
    if (failures) {
        std::printf("%d check(s) FAILED\n", failures);
        return 1;
    }
    std::printf("ds41 cuda core tests passed\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
}
