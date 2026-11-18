// FAST lever ds41-dense-f16-v1 (shared policy core::f16_dense_tall::q8a8, "dense-q8a8-f16-v1"): the dense Q8_0 x A8
// products in one FP16 WMMA chain against the exact integer products (relative L2 per case) and the time of both.
// Lines "fd ...": rows in count parts acc | exact ms | fast ms at 128-row CTAs | at 256-row CTAs | rule | rel_l2.
// --no-timing: one repetition (the ctest form). Both modes first check the packed weight decode (all scales x codes) and
// that every form of the policy gives form 0's bits; --forms [reps [rounds [rows]]] times the forms on DS41 chunk shapes.
#include "core/f16_dense_q8a8.hpp"
#include "strix/ds41_attention.hpp"
#include "strix/hip_ds41_layer.hpp"
#include "hip_ds41_fast_dense_args.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
void ck(hipError_t e, const char* what) {
    if (e == hipSuccess) return;
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(2);
}
template <class T> struct Dev {
    T* p = nullptr;
    std::size_t n;
    explicit Dev(std::size_t count) : n(count) { ck(hipMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T)), "hipMalloc"); }
    ~Dev() { (void)hipFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void up(const std::vector<T>& h) { ck(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    std::vector<T> down() {
        std::vector<T> h(n);
        ck(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "download");
        return h;
    }
};
struct Case {
    unsigned rows, in, count, parts;
    bool acc;
};
std::mt19937 rng(20261114);

template <class F> double timed(F&& f, unsigned reps) {
    f();
    ck(hipDeviceSynchronize(), "sync");
    const auto t0 = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < reps; ++i) f();
    ck(hipDeviceSynchronize(), "sync");
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}

// The DS41 flash prologue's query math (hip_ds41_attention_flash.inl query16: FP32 q, rope512 on the pairs of dims
// 448..511 of each 512-wide head; then frag_from_f32 = to_bf16 per value), one thread per 16-dim chunk.
__global__ void prologue_image(const float* q, const strix::ds41::attn::RopeRow* rope, unsigned rows, unsigned count, std::uint16_t* out) {
    const std::size_t i = blockIdx.x * 256ULL + threadIdx.x;
    if (i >= std::size_t{rows} * count / 16U) return;
    const unsigned r = static_cast<unsigned>(i * 16U / count), chunk = static_cast<unsigned>(i * 16U % count % 512U / 16U);
    float x[16];
    for (unsigned k = 0; k < 16U; ++k) x[k] = q[i * 16U + k];
    if (chunk >= 28U)
        for (unsigned k = 0; k < 8U; ++k) {
            const unsigned p = 8U * (chunk - 28U) + k;
            strix::ds41::attn::rope_pair(x[2U * k], x[2U * k + 1U], rope[r].c[p], rope[r].s[p], false);
        }
    for (unsigned k = 0; k < 16U; ++k) out[i * 16U + k] = strix::attn_core::tile::to_operand<strix::attn_core::tile::Operand::bf16>(x[k]);
}
// enqueue_q_b_rope_image (q_b's image for the flash core's query_op) against the prologue on the FAST FP32 output:
// bitwise equal, taken from the FAST dense rule's rows on, declined (nothing launched) below.
bool run_image(strix::ds41::Q8A8DensePrefillView v, const Dev<float>& fast, const Case& c, unsigned reps) {
    namespace attn = strix::ds41::attn;
    std::vector<attn::RopeRow> hr(c.rows);
    std::uniform_real_distribution<float> th(-3.14159F, 3.14159F);
    for (auto& row : hr)
        for (unsigned p = 0; p < attn::kRopePairs; ++p) {
            const float t = th(rng);
            row.c[p] = std::cos(t), row.s[p] = std::sin(t);
        }
    Dev<attn::RopeRow> dr(c.rows);
    dr.up(hr);
    Dev<std::uint16_t> image(std::size_t{c.rows} * c.count), ref(std::size_t{c.rows} * c.count);
    strix::ds41::set_fast_dense(1);
    v.out = fast.p;  // run()'s FAST FP32 output; the image form does not write it
    const bool want = c.rows >= 128U && c.parts == 1U && !c.acc && c.count % 512U == 0U;
    const bool taken = strix::ds41::enqueue_q_b_rope_image(v, image.p, dr.p, nullptr);
    if (!taken) {
        std::printf("img %5u %5u | declined %s\n", c.rows, c.count, want ? "FAIL" : "ok");
        return !want;
    }
    if (!want) {
        std::printf("img %5u %5u | unexpected admission FAIL\n", c.rows, c.count);
        return false;
    }
    const double ti = timed([&] { (void)strix::ds41::enqueue_q_b_rope_image(v, image.p, dr.p, nullptr); }, reps);
    const unsigned chunks = c.rows * c.count / 16U;
    hipLaunchKernelGGL(prologue_image, dim3((chunks + 255U) / 256U), dim3(256), 0, nullptr, fast.p, dr.p, c.rows, c.count, ref.p);
    ck(hipDeviceSynchronize(), "image sync");
    const auto hi = image.down(), hp = ref.down();
    std::size_t diff = 0;
    for (std::size_t i = 0; i < hi.size(); ++i) diff += hi[i] != hp[i];
    const bool ok = want && diff == 0U;
    std::printf("img %5u %5u | image %8.4f ms | differing %zu of %zu %s\n", c.rows, c.count, ti, diff, hi.size(), ok ? "ok" : "FAIL");
    return ok;
}

bool run(const Case& c, unsigned reps) {
    const unsigned row_bytes = c.in / 32U * 34U, groups = c.in / 32U;
    std::vector<std::uint8_t> w(std::size_t{c.count} * row_bytes);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> scale(0.0005F, 0.02F), step(0.002F, 0.05F);
    for (std::size_t b = 0; b < std::size_t{c.count} * groups; ++b) {
        const _Float16 d = static_cast<_Float16>(scale(rng));
        std::memcpy(&w[b * 34U], &d, 2);
        for (unsigned i = 0; i < 32U; ++i) w[b * 34U + 2U + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code(rng)));
    }
    std::vector<std::int8_t> a(std::size_t{c.rows} * c.in);
    for (auto& x : a) x = static_cast<std::int8_t>(code(rng));
    std::vector<float> dl(std::size_t{c.rows} * groups);
    for (auto& x : dl) x = step(rng);
    Dev<std::uint8_t> dw(w.size() + 64U);  // the stagers read 68 bytes from a block pair: slack past the last row
    Dev<std::int8_t> da(a.size());
    Dev<float> dd(dl.size()), exact(std::size_t{c.rows} * c.count), fast(std::size_t{c.rows} * c.count);
    w.resize(w.size() + 64U);
    dw.up(w), da.up(a), dd.up(dl);
    const std::vector<float> ones(std::size_t{c.rows} * c.count, 1.0F);

    strix::ds41::Q8A8DensePrefillView v;
    v.weight = dw.p, v.codes = da.p, v.deltas = dd.p;
    v.width = c.in / c.parts, v.count = c.count, v.rows = c.rows, v.x_stride = c.in, v.out_stride = c.count;
    if (c.parts == 2U) v.row_width = c.in, v.k_parts = 2U;
    v.accumulate = c.acc;
    v.tall = 1U;
    const auto call = [&](int mode, float* out) {
        strix::ds41::set_fast_dense(mode);
        v.out = out;
        (void)strix::ds41::enqueue_q8_0_a8_dense_prefill(v, nullptr);
    };
    const unsigned r = c.acc ? 1U : reps;  // an accumulating call is not repeatable
    if (c.acc) exact.up(ones), fast.up(ones);
    const double te = c.acc ? (call(0, exact.p), 0.0) : timed([&] { call(0, exact.p); }, r);
    const double tf = c.acc ? (call(1, fast.p), 0.0) : timed([&] { call(1, fast.p); }, r);
    ck(hipDeviceSynchronize(), "sync");
    const auto he = exact.down(), hf = fast.down();
    // Gate both CTA heights even in --no-timing mode; timing alone cannot establish identity.
    namespace q8a8 = strix::core::f16_dense_tall::q8a8;
    double tw[2] = {0.0, 0.0};
    bool geometry_equal = true;
    if (!c.acc) {
        Dev<float> scratch(std::size_t{c.rows} * c.count);
        q8a8::View f{};
        f.weight = dw.p, f.codes = da.p, f.deltas = dd.p, f.out = scratch.p;
        f.row_bytes = row_bytes, f.in = c.in, f.count = c.count, f.rows = c.rows, f.x_stride = c.in, f.out_stride = c.count;
        std::vector<float> wave8;
        if (c.parts == 1U && c.rows >= 128U) {  // the library's instantiation (rule form) against form 0, bit for bit
            q8a8::enqueue(f, nullptr, 0U, 0U);
            ck(hipDeviceSynchronize(), "form 0 sync");
            const auto f0 = scratch.down();
            geometry_equal = std::memcmp(f0.data(), hf.data(), f0.size() * sizeof(float)) == 0;
        }
        for (unsigned i = 0; i < 2U; ++i) {
            tw[i] = timed([&] { q8a8::enqueue(f, nullptr, i ? 16U : 8U); }, r);
            ck(hipDeviceSynchronize(), "geometry sync");
            const auto result = scratch.down();
            if (i == 0U) wave8 = result;
            else geometry_equal = geometry_equal && std::memcmp(wave8.data(), result.data(), result.size() * sizeof(float)) == 0;
        }
    }
    double num = 0.0, den = 0.0;
    unsigned bad = 0;
    for (std::size_t i = 0; i < he.size(); ++i) {
        if (!std::isfinite(hf[i])) ++bad;
        const double d = double(hf[i]) - double(he[i]);
        num += d * d, den += double(he[i]) * double(he[i]);
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    const bool lever = c.rows >= 128U;  // below: the exact kernels (rel must be 0)
    const bool ok = geometry_equal && bad == 0U && (lever ? rel > 0.0 && rel < 3e-3 : rel == 0.0);
    std::printf("fd %5u %5u %5u parts %u acc %d | exact %8.4f | fast %8.4f | w8 %8.4f w16 %8.4f | rule %2u | rel_l2 %.3e %s\n", c.rows,
                c.in, c.count, c.parts, int(c.acc), te, tf, tw[0], tw[1], q8a8::waves_for(c.rows, c.in), rel, ok ? "ok" : "FAIL");
    const bool image_ok = run_image(v, fast, c, reps);
    return ok && image_ok;
}

namespace q8a8 = strix::core::f16_dense_tall::q8a8;
using U4 = strix::core::f16_dense_tall::U4;
// The stagers' packed decode (Cols<8, kWPacked>::stage) against the scalar one (Cols<8, 0>::stage) on weights in memory,
// as the kernel reads them (a check built on register words is not equivalent: the compiler folds their known bits): rows of two Q8_0 blocks, 4 rows per FP16 scale (the 256 code bytes), all 65,536 scales.
__global__ void stage_decode_check(const std::uint8_t* weight, unsigned* bad) {
    const unsigned o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= 65536U * 4U) return;
    q8a8::View v{};
    v.weight = weight, v.row_bytes = 68U, v.count = 65536U * 4U;
    U4 ref[8], pk[8];
    q8a8::Cols<8U, 0U>{v}.stage(o / 16U, o % 16U, 0U, ref);
    q8a8::Cols<8U, q8a8::kWPacked>{v}.stage(o / 16U, o % 16U, 0U, pk);
#pragma unroll
    for (unsigned i = 0; i < 8U; ++i)
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) {
            const std::uint32_t rb = (ref[i][j / 2U] >> (16U * (j % 2U))) & 0xffffU, pb = (pk[i][j / 2U] >> (16U * (j % 2U))) & 0xffffU;
            if (rb == pb) continue;
            atomicAdd(&bad[0], 1U);
            if ((rb & 0x7fffU) > 0x7c00U && (pb & 0x7fffU) > 0x7c00U) atomicAdd(&bad[1], 1U);
            atomicMin(&bad[2], (o / 4U) << 8U | (64U * (o % 4U) + 8U * i + j));
        }
}
bool decode_check() {
    Dev<unsigned> bad(3);
    bad.up({0U, 0U, ~0U});
    // the gate: the real stagers on weights read from memory
    std::vector<std::uint8_t> w(std::size_t{65536U} * 4U * 68U + 64U, 0U);
    for (std::size_t o = 0; o < 65536U * 4U; ++o)
        for (unsigned b = 0; b < 2U; ++b) {
            const std::uint16_t s = static_cast<std::uint16_t>(o / 4U);
            std::memcpy(&w[o * 68U + 34U * b], &s, 2);
            for (unsigned i = 0; i < 32U; ++i) w[o * 68U + 34U * b + 2U + i] = static_cast<std::uint8_t>(64U * (o % 4U) + 32U * b + i);
        }
    Dev<std::uint8_t> dw(w.size());
    dw.up(w);
    hipLaunchKernelGGL(stage_decode_check, dim3(65536U * 4U / 256U), dim3(256), 0, nullptr, dw.p, bad.p);
    ck(hipGetLastError(), "stage decode check launch");
    const auto g = bad.down();
    std::printf("wpack stage decode: %u of 16777216 differ (%u both NaN), first scale 0x%04x code byte 0x%02x %s\n", g[0], g[1],
                g[0] ? g[2] >> 8U : 0U, g[0] ? g[2] & 255U : 0U, g[0] ? "FAIL" : "ok");
    return g[0] == 0U;
}

// Forms of the policy (0, q8a8::kWPacked): each against form 0 bit for bit and, with timing, in ABAB rounds.
// Shapes: DS41 rank 0 per 1024-row chunk (q_b 1280 -> 32768, o_a 8 groups of 4096 -> 1024, o_b two K parts 4096 -> 5120,
// q_a 5120 -> 1280, kv 5120 -> 512); `calls` = launches per layer-chunk.
struct Shape {
    const char* name;
    unsigned rows, in, count, parts, calls;
};
constexpr unsigned kForms[] = {0U, q8a8::kWPacked};
constexpr unsigned kFormCount = sizeof(kForms) / sizeof(kForms[0]);
double chunk_ms[kFormCount] = {};

bool forms(const Shape& c, unsigned reps, unsigned rounds) {
    const unsigned width = c.in * c.parts, row_bytes = width / 32U * 34U, groups = width / 32U;
    std::vector<std::uint8_t> w(std::size_t{c.count} * row_bytes + 64U);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> scale(0.0005F, 0.02F), step(0.002F, 0.05F);
    for (std::size_t b = 0; b < std::size_t{c.count} * groups; ++b) {
        const _Float16 d = static_cast<_Float16>(scale(rng));
        std::memcpy(&w[b * 34U], &d, 2);
        for (unsigned i = 0; i < 32U; ++i) w[b * 34U + 2U + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code(rng)));
    }
    std::vector<std::int8_t> a(std::size_t{c.rows} * width);
    for (auto& x : a) x = static_cast<std::int8_t>(code(rng));
    std::vector<float> dl(std::size_t{c.rows} * groups);
    for (auto& x : dl) x = step(rng);
    Dev<std::uint8_t> dw(w.size());
    Dev<std::int8_t> da(a.size());
    Dev<float> dd(dl.size()), base(std::size_t{c.rows} * c.count), out(std::size_t{c.rows} * c.count);
    dw.up(w), da.up(a), dd.up(dl);
    q8a8::View f{};
    f.weight = dw.p, f.codes = da.p, f.deltas = dd.p, f.row_bytes = row_bytes, f.in = c.in, f.count = c.count, f.rows = c.rows;
    f.x_stride = width, f.out_stride = c.count;
    f.out = base.p;
    if (!q8a8::supported(f)) {
        std::fprintf(stderr, "FAIL unsupported forms shape %s (rows %u)\n", c.name, c.rows);
        return false;
    }
    const auto call = [&](unsigned form, float* y) {
        f.out = y;
        q8a8::enqueue_parts(f, c.parts, nullptr, form);
    };
    call(0U, base.p);
    ck(hipDeviceSynchronize(), "form 0");
    const auto hb = base.down();
    bool equal = true;
    for (unsigned k = 1; k < kFormCount; ++k) {
        ck(hipMemset(out.p, 0xff, out.n * sizeof(float)), "poison");
        call(kForms[k], out.p);
        ck(hipDeviceSynchronize(), "form");
        const auto ho = out.down();
        const bool eq = std::memcmp(hb.data(), ho.data(), hb.size() * sizeof(float)) == 0;
        if (!eq) std::printf("fm %s form %u: DIFFERENT bits from form 0\n", c.name, kForms[k]);
        equal = equal && eq;
    }
    if (!rounds) {
        std::printf("fm %-5s %5u %5u %5u parts %u | forms bit-equal %s\n", c.name, c.rows, width, c.count, c.parts, equal ? "ok" : "FAIL");
        return equal;
    }
    hipEvent_t e0, e1;
    ck(hipEventCreate(&e0), "event"), ck(hipEventCreate(&e1), "event");
    const auto timed_ms = [&](auto&& g) {
        ck(hipEventRecord(e0, nullptr), "record");
        for (unsigned i = 0; i < reps; ++i) g();
        ck(hipEventRecord(e1, nullptr), "record");
        ck(hipEventSynchronize(e1), "event sync");
        float ms = 0.0F;
        ck(hipEventElapsedTime(&ms, e0, e1), "elapsed");
        return double(ms) / reps;
    };
    std::vector<double> t[kFormCount];
    for (unsigned r = 0; r < rounds; ++r)  // strict alternation: every round runs every form once, in the same order
        for (unsigned k = 0; k < kFormCount; ++k) t[k].push_back(timed_ms([&] { call(kForms[k], out.p); }));
    const auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v.size() % 2U ? v[v.size() / 2U] : 0.5 * (v[v.size() / 2U - 1U] + v[v.size() / 2U]);
    };
    for (unsigned k = 0; k < kFormCount; ++k) {
        std::vector<double> warm(t[k].begin() + 1, t[k].end()), d;
        for (unsigned r = 1; r < rounds; ++r) d.push_back(t[k][r] - t[0][r]);
        const double m = median(warm);
        chunk_ms[k] += c.calls * m;
        std::printf("fm %-5s %5u %5u %5u parts %u | form %u | cold %8.4f | warm %8.4f | d %+8.4f (%+.4f..%+.4f) | %s\n", c.name,
                    c.rows, width, c.count, c.parts, kForms[k], t[k][0], m, k ? median(d) : 0.0,
                    k ? *std::min_element(d.begin(), d.end()) : 0.0, k ? *std::max_element(d.begin(), d.end()) : 0.0,
                    equal ? "bits ok" : "BITS FAIL");
    }
    (void)hipEventDestroy(e0), (void)hipEventDestroy(e1);
    return equal;
}

// Batched form (grid z) against one call per group, bit for bit; with rounds, strict ABAB timing (median of warm rounds).
// The o-groups layout of DS41: input row = groups x in (group g at column g * in), output row = groups x count.
bool batched(const char* name, unsigned rows, unsigned in, unsigned count, unsigned groups, unsigned reps, unsigned rounds) {
    const unsigned row_bytes = in / 32U * 34U, width = in * groups, outs = count * groups;
    std::vector<std::uint8_t> w(std::size_t{outs} * row_bytes + 64U);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> scale(0.0005F, 0.02F), step(0.002F, 0.05F);
    for (std::size_t b = 0; b < std::size_t{outs} * (in / 32U); ++b) {
        const _Float16 d = static_cast<_Float16>(scale(rng));
        std::memcpy(&w[b * 34U], &d, 2);
        for (unsigned i = 0; i < 32U; ++i) w[b * 34U + 2U + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code(rng)));
    }
    std::vector<std::int8_t> a(std::size_t{rows} * width);
    for (auto& x : a) x = static_cast<std::int8_t>(code(rng));
    std::vector<float> dl(std::size_t{rows} * (width / 32U));
    for (auto& x : dl) x = step(rng);
    Dev<std::uint8_t> dw(w.size());
    Dev<std::int8_t> da(a.size());
    Dev<float> dd(dl.size()), base(std::size_t{rows} * outs), out(std::size_t{rows} * outs);
    dw.up(w), da.up(a), dd.up(dl);
    q8a8::View f{};
    f.weight = dw.p, f.codes = da.p, f.deltas = dd.p, f.row_bytes = row_bytes, f.in = in, f.count = count, f.rows = rows;
    f.x_stride = width, f.out_stride = outs, f.out = base.p;
    const auto per_group = [&](float* y) {
        for (unsigned g = 0; g < groups; ++g) {
            auto v = f;
            v.weight = dw.p + std::size_t{g} * count * row_bytes, v.codes = da.p + g * in, v.deltas = dd.p + g * in / 32U;
            v.out = y + g * count;
            q8a8::enqueue(v, nullptr);
        }
    };
    const auto one = [&](float* y) {
        auto v = f;
        v.out = y, v.batch = groups, v.batch_in = in, v.batch_out = count, v.batch_weight_bytes = std::uint64_t{count} * row_bytes;
        q8a8::enqueue(v, nullptr);
    };
    if (!q8a8::supported(f)) {
        std::fprintf(stderr, "FAIL unsupported batched shape %s\n", name);
        return false;
    }
    ck(hipMemset(base.p, 0xff, base.n * sizeof(float)), "poison"), ck(hipMemset(out.p, 0x7f, out.n * sizeof(float)), "poison");
    per_group(base.p), one(out.p);
    ck(hipDeviceSynchronize(), "batched");
    const auto hb = base.down(), ho = out.down();
    const bool equal = std::memcmp(hb.data(), ho.data(), hb.size() * sizeof(float)) == 0;
    if (!rounds) {
        std::printf("bt %-5s %5u %5u %5u x %u groups | batched bit-equal %s\n", name, rows, in, count, groups, equal ? "ok" : "FAIL");
        return equal;
    }
    hipEvent_t e0, e1;
    ck(hipEventCreate(&e0), "event"), ck(hipEventCreate(&e1), "event");
    const auto timed_ms = [&](auto&& g) {
        ck(hipEventRecord(e0, nullptr), "record");
        for (unsigned i = 0; i < reps; ++i) g();
        ck(hipEventRecord(e1, nullptr), "record");
        ck(hipEventSynchronize(e1), "event sync");
        float ms = 0.0F;
        ck(hipEventElapsedTime(&ms, e0, e1), "elapsed");
        return double(ms) / reps;
    };
    std::vector<double> ta, tb;
    for (unsigned r = 0; r < rounds; ++r) ta.push_back(timed_ms([&] { per_group(out.p); })), tb.push_back(timed_ms([&] { one(out.p); }));
    const auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v.size() % 2U ? v[v.size() / 2U] : 0.5 * (v[v.size() / 2U - 1U] + v[v.size() / 2U]);
    };
    const double ma = median({ta.begin() + 1, ta.end()}), mb = median({tb.begin() + 1, tb.end()});
    std::printf("bt %-5s %5u %5u %5u x %u groups | per-group %8.4f | batched %8.4f ms (warm median, %u rounds ABAB) | d %+8.4f | %s\n",
                name, rows, in, count, groups, ma, mb, rounds, mb - ma, equal ? "bits ok" : "BITS FAIL");
    (void)hipEventDestroy(e0), (void)hipEventDestroy(e1);
    return equal;
}
}  // namespace

int main(int argc, char** argv) {
    std::printf("dispatch form %u\n", q8a8::form_for());
    // --forms [reps [rounds [rows]]]: the forms bench on the DS41 chunk shapes only
    if (argc > 1 && std::string(argv[1]) == "--forms") {
        strix::test::DenseFormsArgs args;
        if (!strix::test::parse_dense_forms_args(argc, argv, args)) {
            std::fprintf(stderr, "usage: --forms [reps:1..1000 [rounds:2..101 [rows:1..16384]]]\n");
            return 2;
        }
        const auto [reps, rounds, rows] = args;
        const Shape shapes[] = {{"q_b", rows, 1280, 32768, 1, 1}, {"o_a", rows, 4096, 1024, 1, 8}, {"o_b", rows, 4096, 5120, 2, 1},
                                {"q_a", rows, 5120, 1280, 1, 1},  {"kv", rows, 5120, 512, 1, 1}};
        bool ok = decode_check();
        for (const auto& s : shapes) ok = forms(s, reps, rounds) && ok;
        // o_a's 8 groups: one batched launch against 8 calls (the 1024-row chunk and the 256-row S1 layers)
        ok = batched("o_a", rows, 4096, 1024, 8, reps, rounds) && ok;
        ok = batched("o_a", 256, 4096, 1024, 8, reps, rounds) && ok;
        for (unsigned k = 0; k < kFormCount; ++k)
            std::printf("census form %u: %8.3f ms per layer-chunk (d %+7.3f) | x113 layer-chunks (11.6K rank 0) %8.1f ms (d %+6.1f)\n",
                        kForms[k], chunk_ms[k], chunk_ms[k] - chunk_ms[0], 113.0 * chunk_ms[k], 113.0 * (chunk_ms[k] - chunk_ms[0]));
        std::printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    const bool timing = !(argc > 1 && std::string(argv[1]) == "--no-timing");
    const unsigned reps = timing ? 20U : 1U;
    bool ok = decode_check();
    // every form bit-equal to form 0: partial last tile, two K parts (a16 advances with the codes), a small input
    const Shape small[] = {{"s1", 256, 1280, 2048, 1, 1}, {"s2", 300, 1024, 1000, 1, 1}, {"s3", 513, 256, 40, 1, 1},
                           {"s4", 640, 1024, 1024, 2, 1}};
    for (const auto& s : small) ok = forms(s, 1U, 0U) && ok;
    // batched form bit-equal to the per-group calls: partial row block, partial last tile (remainder launch), 2 / 8 groups
    ok = batched("b1", 300, 1024, 1000, 2, 1U, 0U) && ok;
    ok = batched("b2", 1024, 4096, 1024, 8, 1U, 0U) && ok;
    ok = batched("b3", 513, 256, 40, 3, 1U, 0U) && ok;
    // q_b-like (in 1024 / 1536, many outputs), o-like (in 4096, two K parts of 4096), partial last tile, accumulate, small
    const Case cases[] = {
        {64, 1024, 512, 1, false},    {128, 1024, 2048, 1, false},  {256, 1024, 2048, 1, false},  {512, 1024, 4096, 1, false},
        {1024, 1024, 4096, 1, false}, {2048, 1024, 8192, 1, false}, {4096, 1024, 8192, 1, false}, {1024, 1536, 4096, 1, false},
        {1024, 4096, 1024, 1, false}, {1024, 8192, 1024, 2, false}, {2048, 8192, 1024, 2, false}, {1024, 4096, 4096, 1, false},
        {300, 1024, 1000, 1, false},  {513, 256, 40, 1, false},    {256, 1024, 512, 1, true},
        {129, 1024, 512, 1, false},   {300, 1024, 512, 1, false},  {513, 256, 512, 1, false},
    };
    for (const auto& c : cases) ok = run(c, reps) && ok;
    strix::ds41::set_fast_dense(-1);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
