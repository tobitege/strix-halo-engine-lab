// Decode micro-benchmarks of the deepseek41 layer sites (not a ctest): cold-weight GB/s of the Q8_0 x A8
// projections against their byte floor, and the latency of the small per-token kernels. Every timed
// kernel is first checked bitwise against its host reference.
// usage: strix-hip-ds41-layer-bench [iterations]
//        STRIX_DS41_HC_SITE_BENCH=1 strix-hip-ds41-layer-bench [iterations]: per-boundary time of enqueue_hc_site
//        (expansion + mixes + collapse + norm + A8, hidden 5120) at 1..8 rows, one launch against the separate kernels.

#include "strix/ds41_formats.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_layer.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace strix::ds41;

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(2); }
}
template <class T> T* dev(std::size_t n) { T* p = nullptr; check(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "hipMalloc"); return p; }
template <class T> T* dev(const std::vector<T>& h) { T* p = dev<T>(h.size()); check(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); return p; }
template <class T> std::vector<T> down(const T* p, std::size_t n) {
    std::vector<T> h(n);
    check(hipDeviceSynchronize(), "sync");
    check(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "download");
    return h;
}

std::uint16_t to_f16(float x) { return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(x)); }
std::vector<std::uint8_t> random_q8(std::mt19937& rng, std::size_t rows, std::size_t width) {
    std::vector<std::uint8_t> w(rows * width / 32 * 34);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.001F, 0.05F);
    for (std::size_t b = 0; b < rows * width / 32; ++b) {
        const auto s = to_f16(sc(rng));
        w[b * 34] = static_cast<std::uint8_t>(s & 0xffU);
        w[b * 34 + 1] = static_cast<std::uint8_t>(s >> 8U);
        for (std::size_t k = 0; k < 32; ++k) w[b * 34 + 2 + k] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code(rng)));
    }
    return w;
}

// Median kernel time (ms) of `launch(i)` over iterations, one event pair per call.
double time_ms(int iterations, const std::function<void(int)>& launch) {
    hipEvent_t a, b;
    check(hipEventCreate(&a), "event");
    check(hipEventCreate(&b), "event");
    for (int i = 0; i < 3; ++i) launch(i);
    std::vector<float> t;
    for (int i = 0; i < iterations; ++i) {
        check(hipEventRecord(a, nullptr), "record");
        launch(i);
        check(hipEventRecord(b, nullptr), "record");
        check(hipEventSynchronize(b), "sync");
        float ms = 0;
        check(hipEventElapsedTime(&ms, a, b), "elapsed");
        t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

struct Site { const char* name; std::uint32_t width, outputs, groups; };

// Projection: copies of the weight so that consecutive calls read cold bytes (> 512 MB working set).
void bench_projection(std::mt19937& rng, const Site& s, int iterations) {
    const std::size_t row_bytes = s.width / 32 * 34, bytes = row_bytes * s.outputs;
    const std::size_t copies = std::max<std::size_t>(2, (512ULL << 20) / bytes + 1);
    const auto host_w = random_q8(rng, s.outputs, s.width);
    auto* w = dev<std::uint8_t>(bytes * copies);
    for (std::size_t c = 0; c < copies; ++c) check(hipMemcpy(w + c * bytes, host_w.data(), bytes, hipMemcpyHostToDevice), "upload");
    for (const std::uint32_t rows : {1U, 2U, 4U, 8U}) {
        // A8 activations: rows x (groups x width) values.
        const std::size_t in = static_cast<std::size_t>(s.groups) * s.width;
        std::vector<A8Activation> act;
        std::vector<std::int8_t> codes;
        std::vector<float> deltas;
        std::normal_distribution<float> d(0.0F, 1.0F);
        for (std::uint32_t r = 0; r < rows; ++r) {
            std::vector<float> x(in);
            for (auto& v : x) v = d(rng);
            act.push_back(quantize_a8(x));
            codes.insert(codes.end(), act.back().codes.begin(), act.back().codes.end());
            deltas.insert(deltas.end(), act.back().deltas.begin(), act.back().deltas.end());
        }
        auto* dc = dev(codes);
        auto* dd = dev(deltas);
        auto* out = dev<float>(static_cast<std::size_t>(rows) * s.outputs);
        const std::uint32_t per_group = s.outputs / s.groups;
        std::vector<std::uint32_t> experts(s.groups);
        for (std::uint32_t g = 0; g < s.groups; ++g) experts[g] = g;
        auto* de = dev(experts);
        // P2's product: dense view for a plain projection; for the grouped site, routes with one "expert" per
        // group whose entry r reads activation row r * groups + g (width values) and writes output row r * groups + g.
        std::vector<std::uint32_t> begins(s.groups + 1), inputs, outputs;
        for (std::uint32_t g = 0; g < s.groups; ++g) {
            begins[g] = static_cast<std::uint32_t>(inputs.size());
            for (std::uint32_t r = 0; r < rows; ++r) { inputs.push_back(r * s.groups + g); outputs.push_back(r * s.groups + g); }
        }
        begins[s.groups] = static_cast<std::uint32_t>(inputs.size());
        auto* db = dev(begins);
        auto* di = dev(inputs);
        auto* dout_rows = dev(outputs);
        auto launch_p2 = [&](int i) {
            const auto* base = w + (static_cast<std::size_t>(i) % copies) * bytes;
            if (s.groups == 1) {
                ProductView v;
                v.weights = base; v.row_bytes = row_bytes; v.width = s.width; v.outputs = s.outputs;
                v.experts = de; v.expert_count = 1; v.rows = rows; v.codes = dc; v.deltas = dd; v.out = out;
                enqueue_product(WeightFormat::q8_0, v, nullptr);
            } else {
                RoutesProductView v;
                v.weights = base; v.expert_stride_bytes = static_cast<std::uint64_t>(per_group) * row_bytes; v.row_bytes = row_bytes;
                v.width = s.width; v.outputs = per_group;
                v.routes.experts = de; v.routes.begins = db; v.routes.inputs = di; v.routes.outputs = dout_rows;
                v.routes.expert_count = s.groups; v.routes.max_rows = rows;
                v.codes = dc; v.deltas = dd; v.out = out;
                enqueue_routes_product(WeightFormat::q8_0, v, nullptr);
            }
        };
        const bool fits = s.width * rows + s.width / 32 * 4 * rows <= 64 * 1024;
        bool exact = fits;
        if (fits) {
            launch_p2(0);
            const auto got = down(out, static_cast<std::size_t>(rows) * s.outputs);
            for (std::uint32_t r = 0; r < rows && exact; ++r)
                for (std::uint32_t o = 0; o < s.outputs; o += 997) {
                    const std::uint32_t g = o / per_group;
                    A8Activation slice;
                    slice.codes.assign(act[r].codes.begin() + static_cast<std::ptrdiff_t>(g) * s.width,
                                       act[r].codes.begin() + static_cast<std::ptrdiff_t>(g + 1) * s.width);
                    slice.deltas.assign(act[r].deltas.begin() + static_cast<std::ptrdiff_t>(g) * s.width / 32,
                                        act[r].deltas.begin() + static_cast<std::ptrdiff_t>(g + 1) * s.width / 32);
                    slice.sums.assign(s.width / 32, 0);
                    exact = exact && got[static_cast<std::size_t>(r) * s.outputs + o] ==
                        product_q8_0(std::span<const std::uint8_t>(host_w.data() + o * row_bytes, row_bytes), slice, s.width);
                }
        }
        double ms = -1.0;
        if (fits) ms = time_ms(iterations, launch_p2);
        std::printf("{\"site\":\"%s\",\"kernel\":\"p2-product\",\"rows\":%u,\"bytes\":%zu,\"ms\":%.4f,\"gbs\":%.1f,\"exact\":%s,\"fits_lds\":%s}\n",
                    s.name, rows, bytes, ms, ms > 0 ? static_cast<double>(bytes) / (ms * 1e-3) / 1e9 : 0.0,
                    exact ? "true" : "false", fits ? "true" : "false");
        // This branch's site kernel (activations read in place, any width, grouping by per_group).
        auto launch_d = [&](int i) {
            Q8A8View v;
            v.weight = w + (static_cast<std::size_t>(i) % copies) * bytes; v.width = s.width; v.count = s.outputs;
            v.per_group = per_group; v.codes = dc; v.deltas = dd; v.rows = rows; v.x_stride = static_cast<std::uint32_t>(in);
            v.out_stride = s.outputs; v.out = out;
            enqueue_q8_0_a8(v, nullptr);
        };
        check(hipMemset(out, 0, static_cast<std::size_t>(rows) * s.outputs * 4), "clear");
        launch_d(0);
        bool exact_d = true;
        {
            const auto got = down(out, static_cast<std::size_t>(rows) * s.outputs);
            for (std::uint32_t r = 0; r < rows && exact_d; ++r)
                for (std::uint32_t o = 0; o < s.outputs; o += 331) {
                    const std::uint32_t g = o / per_group;
                    A8Activation slice;
                    slice.codes.assign(act[r].codes.begin() + static_cast<std::ptrdiff_t>(g) * s.width,
                                       act[r].codes.begin() + static_cast<std::ptrdiff_t>(g + 1) * s.width);
                    slice.deltas.assign(act[r].deltas.begin() + static_cast<std::ptrdiff_t>(g) * s.width / 32,
                                        act[r].deltas.begin() + static_cast<std::ptrdiff_t>(g + 1) * s.width / 32);
                    slice.sums.assign(s.width / 32, 0);
                    exact_d = exact_d && got[static_cast<std::size_t>(r) * s.outputs + o] ==
                        product_q8_0(std::span<const std::uint8_t>(host_w.data() + o * row_bytes, row_bytes), slice, s.width);
                }
        }
        const double ms_d = time_ms(iterations, launch_d);
        std::printf("{\"site\":\"%s\",\"kernel\":\"site-q8a8\",\"rows\":%u,\"bytes\":%zu,\"ms\":%.4f,\"gbs\":%.1f,\"exact\":%s}\n",
                    s.name, rows, bytes, ms_d, static_cast<double>(bytes) / (ms_d * 1e-3) / 1e9, exact_d ? "true" : "false");
        (void)hipFree(dc); (void)hipFree(dd); (void)hipFree(out); (void)hipFree(de); (void)hipFree(db); (void)hipFree(di); (void)hipFree(dout_rows);
    }
    (void)hipFree(w);
}

std::vector<float> random_floats(std::mt19937& rng, std::size_t n, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

// One hyper-connection boundary as the decode chain enqueues it (expansion of the previous sublayer, this site's
// mixes, collapse by the previous mixes, RMSNorm, A8), back to back over `burst` launches per event pair so that
// the launch gaps count as in the chain. fn rotates over copies (> 32 MB) so that its bytes are cold as in
// decode; the stream inputs stay warm. Every mode is first compared bitwise with the separate kernels.
void bench_hc_site(std::mt19937& rng, int iterations) {
    constexpr std::uint32_t H = 5120, n = H * hc_copies, max_r = 8, burst = 20;
    // STRIX_DS41_HC_SITE_BENCH_COPIES (default 48; 1: warm fn), STRIX_DS41_HC_SITE_BENCH_ITERS (Sinkhorn, default 20)
    const char* copies_env = std::getenv("STRIX_DS41_HC_SITE_BENCH_COPIES");
    const char* iters_env = std::getenv("STRIX_DS41_HC_SITE_BENCH_ITERS");
    const std::size_t fn_count = std::size_t{hc_mix_count} * n, copies = copies_env ? std::max(1UL, std::stoul(copies_env)) : 48;
    const std::uint32_t iters = iters_env ? static_cast<std::uint32_t>(std::max(1UL, std::stoul(iters_env))) : 20U;
    // STRIX_DS41_HC_SITE_BENCH_TILE / _PART: HcSiteView::tile / norm_part of the fast form (default 0, by the row count)
    const char* tile_env = std::getenv("STRIX_DS41_HC_SITE_BENCH_TILE");
    const char* part_env = std::getenv("STRIX_DS41_HC_SITE_BENCH_PART");
    const auto tile = static_cast<std::uint32_t>(tile_env ? std::stoul(tile_env) : 0UL);
    const auto norm_part = static_cast<std::uint32_t>(part_env ? std::stoul(part_env) : 0UL);
    std::vector<std::uint16_t> fn(fn_count);
    {
        std::normal_distribution<float> d(0.0F, 0.02F);
        for (auto& x : fn) x = to_f16(d(rng));
    }
    auto* dfn = dev<std::uint16_t>(fn_count * copies);
    for (std::size_t c = 0; c < copies; ++c) check(hipMemcpy(dfn + c * fn_count, fn.data(), fn_count * 2, hipMemcpyHostToDevice), "upload");
    const auto base = random_floats(rng, hc_mix_count, 0.5F), w = random_floats(rng, H, 1.0F);
    const std::vector<float> scale{0.7F, 1.3F, 2.1F};
    std::vector<HcMix> em(max_r), cm(max_r);
    std::uniform_real_distribution<float> u(0.05F, 0.6F);
    for (auto* m : {&em, &cm})
        for (auto& x : *m) {
            for (auto& v : x.pre) v = u(rng);
            for (auto& v : x.post) v = 2.0F * u(rng);
            for (auto& v : x.comb) v = u(rng);
        }
    auto* dbase = dev(base);
    auto* dscale = dev(scale);
    auto* dw = dev(w);
    auto* dres = dev(random_floats(rng, std::size_t{max_r} * n, 1.0F));
    auto* dsub = dev(random_floats(rng, std::size_t{max_r} * H, 1.0F));
    auto* dem = dev(em);
    auto* dcm = dev(cm);
    auto* st = dev<float>(std::size_t{max_r} * n);
    auto* partial = dev<float>(hc_mix_partial_floats(max_r, H));
    auto* counters = dev(std::vector<std::uint32_t>(hc_site_counter_count(max_r), 0U));
    auto* mixes = dev<HcMix>(max_r);
    auto* out = dev<float>(std::size_t{max_r} * H);
    auto* codes = dev<std::int8_t>(std::size_t{max_r} * H);
    auto* deltas = dev<float>(std::size_t{max_r} * H / 32);
    auto* sums = dev<std::int32_t>(std::size_t{max_r} * H / 32);
    // parts: 0 the full boundary, 1 without the norm (expansion + mixes), 2 without the mixes (expansion + norm)
    for (const int parts : {0, 1, 2})
        for (const std::uint32_t R : {1U, 2U, 4U, 6U, 8U}) {
            auto view = [&](std::uint32_t launches, std::uint32_t form, int i) {
                HcSiteView v;
                v.rows = R; v.hidden = H; v.sub = dsub; v.residual = dres; v.expand_mixes = dem; v.stream = st;
                if (parts != 2) {
                    v.fn = dfn + (static_cast<std::size_t>(i) % copies) * fn_count; v.base = dbase; v.scale = dscale;
                    v.mixes_out = mixes; v.partial = partial; v.counters = counters; v.iters = iters;
                }
                if (parts != 1) {
                    v.collapse_mixes = dcm; v.norm_weight = dw;
                    v.out = out; v.out_stride = H; v.codes = codes; v.deltas = deltas; v.sums = sums; v.a8_stride = H;
                }
                v.launches = launches;
                v.form = form;
                if (form != 1) { v.tile = tile; v.norm_part = norm_part; }
                return v;
            };
            auto outputs = [&] {
                std::vector<std::uint8_t> all;
                auto add = [&](const void* p, std::size_t bytes) {
                    std::vector<std::uint8_t> h(bytes);
                    check(hipDeviceSynchronize(), "sync");
                    check(hipMemcpy(h.data(), p, bytes, hipMemcpyDeviceToHost), "download");
                    all.insert(all.end(), h.begin(), h.end());
                };
                add(st, std::size_t{R} * n * 4);
                if (parts != 2) add(mixes, R * sizeof(HcMix));
                if (parts != 1) {
                    add(out, std::size_t{R} * H * 4); add(codes, std::size_t{R} * H); add(deltas, std::size_t{R} * H / 32 * 4);
                    add(sums, std::size_t{R} * H / 32 * 4);
                }
                return all;
            };
            enqueue_hc_site(view(2, 0, 0), nullptr);
            const auto ref = outputs();
            // modes: 0 by the flag and the row count (as the decode chain), 1 the previous one-launch kernel, 2 the fast
            // one, 3 the separate kernels
            for (const std::uint32_t mode : {0U, 1U, 2U, 3U}) {
                const std::uint32_t launches = mode == 0 ? 0U : mode == 3 ? 2U : 1U, form = mode == 1 ? 1U : mode == 2 ? 2U : 0U;
                check(hipMemset(st, 0, std::size_t{R} * n * 4), "clear");
                check(hipMemset(out, 0, std::size_t{R} * H * 4), "clear");
                check(hipMemset(codes, 0, std::size_t{R} * H), "clear");
                check(hipMemset(mixes, 0, R * sizeof(HcMix)), "clear");
                enqueue_hc_site(view(launches, form, 1), nullptr);
                const bool exact = outputs() == ref;
                const double ms = time_ms(iterations, [&](int i) {
                    for (std::uint32_t k = 0; k < burst; ++k)
                        enqueue_hc_site(view(launches, form, i * static_cast<int>(burst) + static_cast<int>(k)), nullptr);
                });
                static const char* const names[] = {"auto", "previous", "fast", "separate"};
                std::printf("{\"kernel\":\"hc-site\",\"parts\":\"%s\",\"mode\":\"%s\",\"rows\":%u,\"us_per_boundary\":%.2f,\"exact\":%s}\n",
                            parts == 0 ? "full" : parts == 1 ? "no-norm" : "no-mix", names[mode], R, ms * 1e3 / burst,
                            exact ? "true" : "false");
            }
        }
    for (void* p : {static_cast<void*>(dfn), static_cast<void*>(dbase), static_cast<void*>(dscale), static_cast<void*>(dw),
                    static_cast<void*>(dres), static_cast<void*>(dsub), static_cast<void*>(dem), static_cast<void*>(dcm),
                    static_cast<void*>(st), static_cast<void*>(partial), static_cast<void*>(counters), static_cast<void*>(mixes),
                    static_cast<void*>(out), static_cast<void*>(codes), static_cast<void*>(deltas), static_cast<void*>(sums)})
        (void)hipFree(p);
}

// The prefill site with the collapse (HcSiteView::collapse_out) at prefill row counts: each collapse tile (4, 2, 1)
// against tile 4 bitwise, in rounds (ABAB), beside the bytes it must move (stream and sub in, stream and collapse out)
// and a device-to-device copy of the stream as the reachable bandwidth. STRIX_DS41_HC_COLLAPSE_BENCH=1, _ROWS (4096).
bool bench_hc_collapse(std::mt19937& rng, int iterations) {
    constexpr std::uint32_t H = 5120, n = H * hc_copies;
    const char* rows_env = std::getenv("STRIX_DS41_HC_COLLAPSE_BENCH_ROWS");
    const auto R = static_cast<std::uint32_t>(rows_env ? std::stoul(rows_env) : 4096UL);
    std::vector<std::uint16_t> fn(std::size_t{hc_mix_count} * n);
    {
        std::normal_distribution<float> d(0.0F, 0.02F);
        for (auto& x : fn) x = to_f16(d(rng));
    }
    std::vector<HcMix> em(R);
    std::uniform_real_distribution<float> u(0.05F, 0.6F);
    for (auto& x : em) {
        for (auto& v : x.pre) v = u(rng);
        for (auto& v : x.post) v = 2.0F * u(rng);
        for (auto& v : x.comb) v = u(rng);
    }
    auto* dfn = dev(fn);
    auto* dbase = dev(random_floats(rng, hc_mix_count, 0.5F));
    auto* dscale = dev(std::vector<float>{0.7F, 1.3F, 2.1F});
    auto* dres = dev(random_floats(rng, std::size_t{R} * n, 1.0F));
    auto* dsub = dev(random_floats(rng, std::size_t{R} * H, 1.0F));
    auto* dem = dev(em);
    auto* st = dev<float>(std::size_t{R} * n);
    auto* partial = dev<float>(hc_mix_partial_floats(R, H));
    auto* counters = dev(std::vector<std::uint32_t>(hc_site_counter_count(R), 0U));
    auto* mixes = dev<HcMix>(R);
    auto* cout = dev<float>(std::size_t{R} * H);
    auto view = [&](std::uint32_t tile, std::uint32_t once) {
        HcSiteView v;
        v.collapse_once = once;
        v.rows = R; v.hidden = H; v.sub = dsub; v.residual = dres; v.expand_mixes = dem; v.stream = st;
        v.fn = dfn; v.base = dbase; v.scale = dscale; v.iters = 20; v.mix_norm_eps = 1e-20F;
        v.mixes_out = mixes; v.partial = partial; v.counters = counters;
        v.launches = 1; v.form = 2; v.tile = tile; v.collapse_out = cout;
        return v;
    };
    auto outputs = [&] {
        std::vector<std::uint8_t> all;
        auto add = [&](const void* p, std::size_t bytes) {
            std::vector<std::uint8_t> h(bytes);
            check(hipDeviceSynchronize(), "sync");
            check(hipMemcpy(h.data(), p, bytes, hipMemcpyDeviceToHost), "download");
            all.insert(all.end(), h.begin(), h.end());
        };
        add(st, std::size_t{R} * n * 4); add(mixes, R * sizeof(HcMix)); add(cout, std::size_t{R} * H * 4);
        return all;
    };
    enqueue_hc_site(view(4, 1), nullptr);  // reference: the previous form, tile 4
    const auto ref = outputs();
    const double bytes = static_cast<double>(R) * (2.0 * n + 2.0 * H) * 4.0;
    bool form_exact[2][3] = {{true, true, true}, {true, true, true}};
    bool all_exact = true;
    for (int round = 0; round < 3; ++round) {
        for (const std::uint32_t once : {1U, 2U})
        for (const std::uint32_t tile : {4U, 2U, 1U}) {
            if (once == 2U && tile == 4U) continue;  // one read caps the tile at 2 rows
            check(hipMemset(st, 0, std::size_t{R} * n * 4), "clear");
            check(hipMemset(cout, 0, std::size_t{R} * H * 4), "clear");
            check(hipMemset(mixes, 0, R * sizeof(HcMix)), "clear");
            enqueue_hc_site(view(tile, once), nullptr);
            const bool launch_exact = outputs() == ref;
            const double ms = time_ms(iterations, [&](int) { enqueue_hc_site(view(tile, once), nullptr); });
            const bool repeated_exact = outputs() == ref;
            bool& exact = form_exact[once - 1U][tile / 2U];
            exact = exact && launch_exact && repeated_exact;
            all_exact = all_exact && exact;
            std::printf("{\"kernel\":\"hc-collapse\",\"round\":%d,\"rows\":%u,\"once\":%u,\"tile\":%u,\"ms\":%.4f,\"gbps\":%.1f,\"exact\":%s}\n",
                        round, R, once == 2 ? 1U : 0U, tile, ms, bytes / ms * 1e-6, exact ? "true" : "false");
        }
        const double cms = time_ms(iterations, [&](int) {
            check(hipMemcpyAsync(st, dres, std::size_t{R} * n * 4, hipMemcpyDeviceToDevice, nullptr), "copy");
        });
        std::printf("{\"kernel\":\"d2d-copy\",\"round\":%d,\"bytes_moved\":%.0f,\"ms\":%.4f,\"gbps\":%.1f,\"site_floor_ms\":%.4f}\n", round,
                    2.0 * R * n * 4, cms, 2.0 * R * n * 4 / cms * 1e-6, bytes / (2.0 * R * n * 4 / cms));
    }
    for (void* p : {static_cast<void*>(dfn), static_cast<void*>(dbase), static_cast<void*>(dscale), static_cast<void*>(dres),
                    static_cast<void*>(dsub), static_cast<void*>(dem), static_cast<void*>(st), static_cast<void*>(partial),
                    static_cast<void*>(counters), static_cast<void*>(mixes), static_cast<void*>(cout)})
        (void)hipFree(p);
    return all_exact;
}

} // namespace

int main(int argc, char** argv) {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 50;
    std::mt19937 rng(20261019);
    if (const char* b = std::getenv("STRIX_DS41_HC_SITE_BENCH"); b && b[0] == '1') {
        bench_hc_site(rng, iterations);
        return 0;
    }
    if (const char* b = std::getenv("STRIX_DS41_HC_COLLAPSE_BENCH"); b && b[0] == '1') {
        return bench_hc_collapse(rng, iterations) ? 0 : 1;
    }
    const Site sites[] = {
        {"attn_q_a 5120->1280", 5120, 1280, 1}, {"attn_q_b 1280->32768", 1280, 32768, 1},
        {"attn_kv 5120->512", 5120, 512, 1},    {"attn_output_a 8x(4096->1024)", 4096, 8192, 8},
        {"attn_output_b 8192->5120", 8192, 5120, 1}, {"head half 5120->64640", 5120, 64640, 1},
    };
    for (const auto& s : sites) bench_projection(rng, s, iterations);
    return 0;
}
