// Tests of src/cuda/models/qwen/qwen_dense.cu (dense A8 projections + shared expert) against the literal host reference
// tests/cuda/qwen_dense_ref.hpp, bit for bit.
//   qwen_dense_tests                 host-only tests (no CUDA call)
//   qwen_dense_tests --gpu           + CUDA vs host: synthetic edge cases, real Q8_0 weights, shared expert
//   ctest -R qwen_dense_rows16       isolated ON/OFF dispatch qualification (no model/table assets required)
//   qwen_dense_tests --bench         GMAC/s of 8192 x 2560 x 16384 (GDN qkv||z shape)
//   qwen_dense_tests --bench-shape M,K,N [...]  one site (M rows, K width, N outputs), best of 8, TOPS; repeatable, no model
//   qwen_dense_tests --fixture F     list the records of a STRIX_QRTX_STEP_DUMP file (and replay when recognised)
// Env: STRIX_QWEN_GGUF (model), STRIX_QRTX_TABLES (default qrtx-build\tables).
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "strix/cuda/qwen_dense.hpp"
#include "strix/cuda/amd_math_tables.hpp"
#include "qwen_dense_ref.hpp"
#include "qwen_fixture.hpp"
#include "qwen_gguf_host.hpp"
#ifdef STRIX_WFMT_CAPTURE
#include "../wfmt/cuda_capture.hpp"
static std::string wfmt_capture_directory;
static std::size_t wfmt_capture_index = 0;
#endif

namespace qd = strix::cuda::qwen_dense;
namespace ref = strix::qwen_dense_ref;

static int g_fail = 0, g_pass = 0;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)
static void check(bool ok, const std::string& what) {
    if (ok) ++g_pass; else { ++g_fail; std::printf("FAIL %s\n", what.c_str()); }
}

static int g_rows16 = -1;
static unsigned g_dispatches = 0;
namespace strix::cuda::qwen_dense {
void test_dispatch(std::uint32_t rows, bool one_pass) {
    if (g_rows16 < 0) return;
    ++g_dispatches;
    std::printf("dispatch rows=%u form=%s\n", rows, one_pass ? "rp16" : "split8");
    check(one_pass == (g_rows16 == 1), "requested dense dispatch");
}
}

// ---------------------------------------------------------------------------------------------------------------------
struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ULL + 1) {}
    std::uint64_t next() { s += 0x9E3779B97F4A7C15ULL; std::uint64_t z = s; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
    std::uint32_t u32() { return static_cast<std::uint32_t>(next() >> 32); }
    double uni() { return (next() >> 11) * 0x1p-53; }
    float normal() { const double u = uni() + 1e-300, v = uni(); return static_cast<float>(std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * v)); }
};

static void parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn) {
    const unsigned t = std::max(1U, std::thread::hardware_concurrency());
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> th;
    for (unsigned i = 0; i < t; ++i) th.emplace_back([&] { for (std::size_t k; (k = next.fetch_add(1)) < n;) fn(k); });
    for (auto& x : th) x.join();
}

// Activations with edge blocks. kind per (row, block): see the switch. FP32 values, BF16-exact when bf16 is set.
static std::vector<float> make_rows(std::uint32_t rows, std::uint32_t width, bool bf16, std::uint64_t seed, bool edges = true) {
    std::vector<float> x(static_cast<std::size_t>(rows) * width);
    Rng g(seed);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t b = 0; b < width / 32U; ++b) {
            float* v = &x[static_cast<std::size_t>(r) * width + 32U * b];
            const std::uint32_t kind = edges ? g.u32() % 24U : 99U;
            const float scale = std::ldexp(1.0F, static_cast<int>(g.u32() % 31U) - 20);
            for (int k = 0; k < 32; ++k) v[k] = g.normal() * scale;
            switch (kind) {
            case 0: for (int k = 0; k < 32; ++k) v[k] = (k & 1) ? -0.0F : 0.0F; break;            // zero block (signed zeros)
            case 1: for (int k = 0; k < 32; ++k) v[k] = g.normal() * 0x1p-122F; v[3] = 0x1p-120F; v[4] = -0x1.fep-120F;
                    v[5] = bf16 ? ref::bits_float(0x00010000U) : ref::bits_float(0x00000003U); break;  // tiny (< 2^-119, subnormals)
            case 2: v[g.u32() % 32U] = ref::bits_float(0x7fc00000U); break;                         // NaN
            case 3: v[g.u32() % 32U] = ref::bits_float(0x7f800000U); break;                         // +inf
            case 4: v[g.u32() % 32U] = ref::bits_float(0xff800000U); break;                         // -inf
            case 5: case 6: {                                                                       // ties: delta = 2^e exactly
                const float s = std::ldexp(1.0F, kind == 5 ? 0 : static_cast<int>(g.u32() % 20U) - 10);
                for (int k = 0; k < 32; ++k) v[k] = (static_cast<float>(static_cast<int>(g.u32() % 253U) - 126) + 0.5F) * s;
                v[0] = 127.0F * s; v[1] = -126.5F * s; v[2] = 0.5F * s; v[3] = -0.5F * s; v[4] = 2.5F * s; v[5] = -3.5F * s;
                break;
            }
            case 7: v[7] = 0x1p-119F; for (int k = 0; k < 32; ++k) if (k != 7) v[k] = g.normal() * 0x1p-122F; break;  // amax = 2^-119: not tiny
            case 8: for (int k = 0; k < 32; ++k) v[k] = g.normal() * 1e30F; v[9] = -3.0e38F; break;  // huge, finite
            case 9: v[g.u32() % 32U] = -127.0F * scale; v[(g.u32() % 31U) + 1U] = 127.0F * scale; break;   // +-127 codes
            case 10: if (!bf16) { v[1] = ref::bits_float(0x00000001U); v[2] = ref::bits_float(0x807fffffU); } break;
            default: break;
            }
            if (bf16) for (int k = 0; k < 32; ++k) v[k] = ref::bf2f(ref::f2bf(v[k]) ) ;
            if (bf16) for (int k = 0; k < 32; ++k) if (v[k] != v[k]) v[k] = ref::bits_float(0x7fc00000U);
        }
    return x;
}
static std::vector<std::uint16_t> to_bf16(const std::vector<float>& x) {
    std::vector<std::uint16_t> h(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) h[i] = static_cast<std::uint16_t>(ref::float_bits(x[i]) >> 16U);  // exact (BF16-valued)
    return h;
}

// Synthetic Q8_0 rows: scales normal/subnormal/zero halfs, codes over the full int8 range (-128 included).
static std::vector<std::uint8_t> make_q8(std::uint32_t outputs, std::uint32_t width, std::uint64_t seed) {
    const std::uint32_t nb = width / 32U;
    std::vector<std::uint8_t> w(static_cast<std::size_t>(outputs) * nb * 34U);
    Rng g(seed);
    for (std::size_t blk = 0; blk < static_cast<std::size_t>(outputs) * nb; ++blk) {
        std::uint8_t* p = &w[blk * 34U];
        const std::uint32_t k = g.u32() % 64U;
        std::uint16_t h;
        if (k == 0) h = 0;
        else if (k == 1) h = static_cast<std::uint16_t>(g.u32() % 1024U);   // subnormal half
        else h = static_cast<std::uint16_t>(((g.u32() % 14U + 1U) << 10) | (g.u32() % 1024U) | ((g.u32() & 1U) << 15));
        p[0] = static_cast<std::uint8_t>(h); p[1] = static_cast<std::uint8_t>(h >> 8);
        for (int i = 0; i < 32; ++i) p[2 + i] = static_cast<std::uint8_t>(g.u32());
    }
    return w;
}

// ---------------------------------------------------------------------------------------------------------------------
template <class T> struct Dev {
    T* p = nullptr; std::size_t n = 0;
    explicit Dev(std::size_t count) : n(count) { CK(cudaMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T))); }
    Dev(const std::vector<T>& v) : Dev(v.size()) { up(v); }
    ~Dev() { cudaFree(p); }
    void up(const std::vector<T>& v) { CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> down() const { std::vector<T> v(n); CK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
};

static std::string hex(std::uint32_t v) { char b[16]; std::snprintf(b, sizeof b, "0x%08x", v); return b; }

// ---------------------------------------------------------------------------------------------------------------------
// Host-only tests: hand-computed quantizer and store values.
static void host_tests() {
    std::vector<float> x(32 * 8);
    std::vector<std::int8_t> c(x.size());
    std::vector<float> d(8);
    // block 0: zeros; 1: tiny; 2: NaN; 3: +inf; 4: amax 127 with ties; 5: amax 254 (delta 2) ties; 6: amax 2^-119; 7: -0 / 1
    for (int k = 0; k < 32; ++k) { x[k] = 0.0F; x[32 + k] = 0x1p-121F; x[64 + k] = 1.0F; x[96 + k] = 1.0F; x[128 + k] = 0.0F;
                                   x[160 + k] = 0.0F; x[192 + k] = 0x1p-121F; x[224 + k] = -0.0F; }
    x[66] = ref::bits_float(0x7fc00000U); x[100] = ref::bits_float(0x7f800000U);
    const float t4[] = {127.0F, 2.5F, 3.5F, -0.5F, 0.5F, 126.5F, -126.5F, -127.0F, 1.5F};
    const int c4[] = {127, 2, 4, 0, 0, 126, -126, -127, 2};
    for (int i = 0; i < 9; ++i) x[128 + i] = t4[i];
    const float t5[] = {254.0F, 5.0F, 7.0F, -1.0F, 3.0F, -254.0F};
    const int c5[] = {127, 2, 4, 0, 2, -127};
    for (int i = 0; i < 6; ++i) x[160 + i] = t5[i];
    x[192] = 0x1p-119F;
    x[230] = 1.0F;
    ref::quantize(1, 256, x.data(), 256, c.data(), d.data(), 1);
    check(d[0] == 1.0F && std::all_of(c.begin(), c.begin() + 32, [](std::int8_t v) { return v == 0; }), "host q zero block");
    check(d[1] == 1.0F && std::all_of(c.begin() + 32, c.begin() + 64, [](std::int8_t v) { return v == 0; }), "host q tiny block");
    check(ref::float_bits(d[2]) == 0x7fc00000U && std::all_of(c.begin() + 64, c.begin() + 96, [](std::int8_t v) { return v == 0; }), "host q NaN block");
    check(ref::float_bits(d[3]) == 0x7fc00000U && c[100] == 0, "host q inf block");
    bool ok4 = d[4] == 1.0F; for (int i = 0; i < 9; ++i) ok4 &= c[128 + i] == c4[i];
    check(ok4, "host q ties delta 1 (half-even rint, +-127)");
    bool ok5 = d[5] == 2.0F; for (int i = 0; i < 6; ++i) ok5 &= c[160 + i] == c5[i];
    check(ok5, "host q ties delta 2");
    check(d[6] == 0x1p-119F / 127.0F && c[192] == 127 && c[193] == 32, "host q amax 2^-119 is not tiny");   // 2^-121/(2^-119/127) = 31.75
    check(d[7] == 1.0F / 127.0F && c[230] == 127 && c[224] == 0, "host q signed zero block");
    // BF16 RNE without NaN special case (HIP float_to_bf16)
    check(ref::f2bf(ref::bits_float(0x3f808000U)) == 0x3f80 && ref::f2bf(ref::bits_float(0x3f818000U)) == 0x3f82 &&
          ref::f2bf(ref::bits_float(0x7fc00000U)) == 0x7fc0 && ref::f2bf(ref::bits_float(0x7f7fffffU)) == 0x7f80 &&
          ref::f2bf(ref::bits_float(0x7fffffffU)) == 0x8000, "host bf16 rounding");
    check(ref::half_to_float(0x3c00) == 1.0F && ref::half_to_float(0x0001) == 0x1p-24F && ref::half_to_float(0xfbff) == -65504.0F,
          "host half conversion");
    // projection: 2 blocks, known dot
    std::vector<std::uint8_t> w(68, 0);
    w[0] = 0x00; w[1] = 0x3c;  // 1.0
    w[34] = 0x00; w[35] = 0x40;  // 2.0
    for (int k = 0; k < 32; ++k) { w[2 + k] = 1; w[36 + k] = static_cast<std::uint8_t>(-1); }
    std::vector<std::int8_t> code(64, 3);
    const float dl[2] = {0.5F, 0.25F};
    check(ref::project_one(w.data(), 64, code.data(), dl, 1, 0) == 1.0F * 0.5F * 96.0F + 2.0F * 0.25F * -96.0F, "host project known");
}

// ---------------------------------------------------------------------------------------------------------------------
// GPU: quantizer vs host.
static void gpu_quantize_tests() {
    for (std::uint32_t width : {2560U, 6144U, 640U})
        for (int bf : {1, 0})
            for (std::uint32_t rows : {1U, 3U, 64U, 257U, 4096U}) {
                const auto x = make_rows(rows, width, bf != 0, width * 131U + rows * 7U + bf);
                const std::uint32_t pitch = rows + 5U;
                std::vector<std::int8_t> hc(static_cast<std::size_t>(rows) * width);
                std::vector<float> hd(static_cast<std::size_t>(width / 32U) * pitch, -1.0F);
                ref::quantize(rows, width, x.data(), width, hc.data(), hd.data(), pitch);
                Dev<std::int8_t> dc(hc.size());
                Dev<float> dd(hd.size());
                CK(cudaMemset(dd.p, 0xff, dd.n * 4));
                for (std::size_t i = 0; i < hd.size(); ++i) if (i % pitch >= rows) hd[i] = ref::bits_float(0xffffffffU);
                if (bf) {
                    Dev<std::uint16_t> in(to_bf16(x));
                    CK(qd::quantize(nullptr, rows, width, qd::Input::bf16, in.p, width, dc.p, dd.p, pitch));
                } else {
                    Dev<float> in(x);
                    CK(qd::quantize(nullptr, rows, width, qd::Input::f32, in.p, width, dc.p, dd.p, pitch));
                }
                CK(cudaDeviceSynchronize());
                const auto gc = dc.down();
                const auto gd = dd.down();
                const bool cok = std::memcmp(gc.data(), hc.data(), hc.size()) == 0;
                const bool dok = std::memcmp(gd.data(), hd.data(), hd.size() * 4) == 0;
                std::string what = "quantize w=" + std::to_string(width) + (bf ? " bf16" : " f32") + " rows=" + std::to_string(rows);
                if (!cok) for (std::size_t i = 0; i < hc.size(); ++i) if (gc[i] != hc[i]) {
                    what += " code[" + std::to_string(i) + "] gpu " + std::to_string(gc[i]) + " host " + std::to_string(hc[i]); break; }
                if (!dok) for (std::size_t i = 0; i < hd.size(); ++i) if (ref::float_bits(gd[i]) != ref::float_bits(hd[i])) {
                    what += " delta[" + std::to_string(i) + "] gpu " + hex(ref::float_bits(gd[i])) + " host " + hex(ref::float_bits(hd[i])); break; }
                check(cok && dok, what);
            }
}

// ---------------------------------------------------------------------------------------------------------------------
// GPU: projection vs host for one site configuration.
struct Case {
    const char* name;
    qd::Store store;
    qd::Input input;
    std::uint32_t width;
    std::vector<std::uint32_t> outs;
};

// weights: one host Q8_0 buffer per segment. Returns the number of compared elements.
static std::size_t run_projection(const Case& cs, const std::vector<std::vector<std::uint8_t>>& weights, std::uint32_t rows,
                                  std::uint64_t seed, bool edges, std::size_t max_checked_rows) {
    const std::uint32_t width = cs.width, nseg = static_cast<std::uint32_t>(cs.outs.size());
    const bool bf = cs.input == qd::Input::bf16;
    const auto x = make_rows(rows, width, bf, seed, edges);
    const std::uint32_t pitch = rows;
    // device
    std::vector<std::unique_ptr<Dev<std::uint8_t>>> dw;
    for (const auto& w : weights) dw.emplace_back(std::make_unique<Dev<std::uint8_t>>(w));
    Dev<std::int8_t> codes(static_cast<std::size_t>(rows) * width);
    Dev<float> deltas(static_cast<std::size_t>(width / 32U) * pitch);
    const bool f32out = cs.store == qd::Store::f32 || cs.store == qd::Store::f32_round_bf16;
    const std::uint32_t esz = f32out ? 4U : 2U;
    std::vector<std::unique_ptr<Dev<std::uint8_t>>> outs;
    qd::Site site{};
    site.store = cs.store; site.width = width; site.segments = nseg;
    // output strides: segment outputs + 3 padding elements (strided rows)
    for (std::uint32_t m = 0; m < nseg; ++m) {
        outs.emplace_back(std::make_unique<Dev<std::uint8_t>>(static_cast<std::size_t>(rows) * (cs.outs[m] + 4U) * esz));
        CK(cudaMemset(outs.back()->p, 0xcd, outs.back()->n));
        site.segment[m] = {dw[m]->p, cs.outs[m], outs.back()->p, cs.outs[m] + 4U};
    }
    std::vector<float> gate(static_cast<std::size_t>(rows) * 3U);
    Rng g(seed ^ 0xabc);
    for (std::uint32_t r = 0; r < rows; ++r) {
        gate[r * 3U] = static_cast<float>(g.uni());
        if (r % 97U == 5U) gate[r * 3U] = ref::bits_float(0x7fc00000U);
        if (r % 89U == 7U) gate[r * 3U] = 0.0F;
    }
    Dev<float> dgate(gate);
    if (cs.store == qd::Store::bf16_gated) { site.gate = dgate.p; site.gate_stride = 3U; }
    if (bf) {
        Dev<std::uint16_t> in(to_bf16(x));
        CK(qd::quantize_project(nullptr, site, rows, cs.input, in.p, width, codes.p, deltas.p, pitch));
        CK(cudaDeviceSynchronize());
    } else {
        Dev<float> in(x);
        CK(qd::quantize_project(nullptr, site, rows, cs.input, in.p, width, codes.p, deltas.p, pitch));
        CK(cudaDeviceSynchronize());
    }
#ifdef STRIX_WFMT_CAPTURE
    std::vector<float> observed_partials;
    std::uint32_t captured_outputs = 0;
    if (!wfmt_capture_directory.empty()) {
        std::vector<std::vector<std::uint8_t>> production;
        for (const auto& out : outs) production.push_back(out->down());
        for (const auto n : cs.outs) captured_outputs += n;
        Dev<float> partials(static_cast<std::size_t>(width / 32U) * rows * captured_outputs);
        CK(qd::capture_project_partials(nullptr, site, rows, codes.p, deltas.p, pitch, partials.p));
        CK(cudaDeviceSynchronize());
        observed_partials = partials.down();
        for (std::size_t m = 0; m < outs.size(); ++m)
            check(production[m] == outs[m]->down(), "diagnostic matches production output");
    }
#endif
    // host reference on the checked rows
    std::vector<std::int8_t> hc(static_cast<std::size_t>(rows) * width);
    std::vector<float> hd(static_cast<std::size_t>(width / 32U) * pitch);
    ref::quantize(rows, width, x.data(), width, hc.data(), hd.data(), pitch);
    std::vector<std::uint32_t> check_rows;
    if (rows <= max_checked_rows) for (std::uint32_t r = 0; r < rows; ++r) check_rows.push_back(r);
    else {
        for (std::uint32_t r = 0; r < 8U; ++r) { check_rows.push_back(r); check_rows.push_back(rows - 1U - r); }
        Rng s(seed + 99);
        while (check_rows.size() < 48U) check_rows.push_back(s.u32() % rows);
    }
    std::size_t compared = 0;
    bool ok = true;
    std::string first;
    for (std::uint32_t m = 0; m < nseg; ++m) {
        const auto got = outs[m]->down();
#ifdef STRIX_WFMT_CAPTURE
        if (!wfmt_capture_directory.empty()) {
            // Observe the exact production buffers, not host recomputations.
            const auto observed_codes = codes.down();
            const auto observed_scales = deltas.down();
            std::uint32_t begin = 0;
            for (std::uint32_t j = 0; j < m; ++j) begin += cs.outs[j];
            std::vector<float> segment_partials;
            for (std::size_t line = 0; line < static_cast<std::size_t>(width / 32U) * rows; ++line)
                segment_partials.insert(segment_partials.end(), observed_partials.data() + line * captured_outputs + begin,
                                         observed_partials.data() + line * captured_outputs + begin + cs.outs[m]);
            const std::uint64_t shape[]{rows, width, cs.outs[m], static_cast<std::uint64_t>(cs.store), m};
            const std::string name = "cuda-dense-" + std::to_string(wfmt_capture_index++);
            check(strix::weight::test::write_cuda_capture(wfmt_capture_directory, name,
                "cuda-qwen-imma-four-chain@1", shape,
                {{"source", std::as_bytes(std::span(weights[m]))},
                 {"activation_codes", std::as_bytes(std::span(observed_codes))},
                 {"activation_scales", std::as_bytes(std::span(observed_scales))},
                 {"sidecars", std::as_bytes(std::span(gate))},
                 {"partials", std::as_bytes(std::span(segment_partials))},
                 {"output", std::as_bytes(std::span(got))}}), "write " + name);
        }
#endif
        const std::uint32_t N = cs.outs[m], ld = N + 4U;
        const std::uint32_t row_bytes = width / 32U * 34U;
        std::vector<std::uint8_t> expect_row_ok(check_rows.size(), 1);
        std::vector<std::string> msg(check_rows.size());
        parallel_for(check_rows.size(), [&](std::size_t i) {
            const std::uint32_t r = check_rows[i];
            for (std::uint32_t n = 0; n < N; ++n) {
                const float y = ref::project_one(weights[m].data() + static_cast<std::size_t>(n) * row_bytes, width,
                                                 hc.data() + static_cast<std::size_t>(r) * width, hd.data(), pitch, r);
                std::uint32_t e;
                switch (cs.store) {
                case qd::Store::f32: e = ref::float_bits(y); break;
                case qd::Store::f32_round_bf16: e = static_cast<std::uint32_t>(ref::f2bf(y)) << 16U; break;
                case qd::Store::bf16: e = ref::f2bf(y); break;
                default: e = ref::f2bf(ref::amd_mul(gate[r * 3U], y)); break;
                }
                std::uint32_t gv = 0;
                std::memcpy(&gv, got.data() + (static_cast<std::size_t>(r) * ld + n) * esz, esz);
                if (gv != e) {
                    expect_row_ok[i] = 0;
                    msg[i] = std::string(" seg ") + std::to_string(m) + " row " + std::to_string(r) + " out " + std::to_string(n) +
                             " gpu " + hex(gv) + " host " + hex(e);
                    return;
                }
            }
        });
        for (std::size_t i = 0; i < check_rows.size(); ++i) if (!expect_row_ok[i]) { if (ok) first = msg[i]; ok = false; }
        compared += check_rows.size() * N;
        // padding untouched
        bool pad = true;
        for (std::uint32_t r = 0; r < rows && pad; ++r)
            for (std::uint32_t n = N; n < ld; ++n) if (got[(static_cast<std::size_t>(r) * ld + n) * esz] != 0xcd) { pad = false; break; }
        if (!pad) { ok = false; first += " padding overwritten"; }
    }
    check(ok, std::string("project ") + cs.name + " rows=" + std::to_string(rows) + (edges ? " edges" : "") + first);
    return compared;
}

static const Case kCases[] = {
    {"gdn_in qkv||z", qd::Store::f32_round_bf16, qd::Input::bf16, 2560U, {10240U, 6144U}},
    {"gdn_out", qd::Store::bf16, qd::Input::f32, 6144U, {2560U}},
    {"qsa_in q|k|v", qd::Store::f32_round_bf16, qd::Input::bf16, 2560U, {12288U, 512U, 512U}},
    {"qsa_out", qd::Store::bf16, qd::Input::f32, 6144U, {2560U}},
    {"shared gate||up", qd::Store::f32, qd::Input::bf16, 2560U, {640U, 640U}},
    {"shared down", qd::Store::bf16_gated, qd::Input::f32, 640U, {2560U}},
};

static void gpu_projection_synthetic() {
    std::size_t total = 0;
    int ci = 0;
    for (const auto& cs : kCases) {
        std::vector<std::vector<std::uint8_t>> w;
        for (std::size_t m = 0; m < cs.outs.size(); ++m) w.push_back(make_q8(cs.outs[m], cs.width, 1000U * ci + m));
        for (std::uint32_t rows : {1U, 5U, 8U, 9U, 13U, 16U, 17U, 64U, 130U, 256U, 1000U, 1024U, 4096U}) {
            if (g_rows16 >= 0 && rows != 9U && rows != 13U && rows != 16U) continue;
            const auto before = g_dispatches;
            total += run_projection(cs, w, rows, 77U * rows + ci, true, 136U);
            if (g_rows16 >= 0) check(g_dispatches == before + 1U, "eligible projection dispatched exactly once");
        }
        ++ci;
    }
    std::printf("synthetic projection: %zu (row, output) values compared\n", total);
}

static void gpu_projection_real(const strix::qwen_gguf::Model& model) {
    struct Real { int ci; std::vector<const char*> tensors; };
    const Real reals[] = {
        {0, {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"}},
        {1, {"blk.0.ssm_out.weight"}},
        {2, {"blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight"}},
        {3, {"blk.3.attn_output.weight"}},
        {4, {"blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight"}},
        {5, {"blk.0.ffn_down_shexp.weight"}},
    };
    std::size_t total = 0;
    for (const auto& rl : reals) {
        const Case& cs = kCases[rl.ci];
        std::vector<std::vector<std::uint8_t>> w;
        bool shape = true;
        for (std::size_t m = 0; m < rl.tensors.size(); ++m) {
            const auto& t = model.tensor(rl.tensors[m]);
            shape &= t.ggml_type == 8U && t.byte_size == static_cast<std::uint64_t>(cs.outs[m]) * cs.width / 32U * 34U;
            w.push_back(model.bytes(rl.tensors[m]));
        }
        check(shape, std::string("real weight shapes ") + cs.name);
        if (!shape) continue;
        for (std::uint32_t rows : {64U, 300U}) total += run_projection(cs, w, rows, 5U * rows + rl.ci, false, 72U);
    }
    std::printf("real-weight projection: %zu (row, output) values compared\n", total);
}

// ---------------------------------------------------------------------------------------------------------------------
// GPU: shared activation (synthetic wide-range gates) and the whole shared expert with real weights.
static void host_shared_expert(std::uint32_t rows, const std::vector<std::uint16_t>& in, const std::vector<std::uint8_t>& gw,
                               const std::vector<std::uint8_t>& uw, const std::vector<std::uint8_t>& dwn, const std::vector<float>& sc,
                               std::uint32_t H, std::uint32_t I, std::uint64_t ws_stride, const ref::HostExpf& ex,
                               std::vector<float>& ws, std::vector<std::uint16_t>& out) {
    std::vector<float> x(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) x[i] = ref::bf2f(in[i]);
    std::vector<std::int8_t> c(static_cast<std::size_t>(rows) * H);
    std::vector<float> d(static_cast<std::size_t>(H / 32U) * rows);
    ref::quantize(rows, H, x.data(), H, c.data(), d.data(), rows);
    ws.assign(static_cast<std::size_t>(rows) * ws_stride, 0.0F);
    const std::uint32_t gb = H / 32U * 34U, db = I / 32U * 34U;
    parallel_for(rows, [&](std::size_t r) {
        for (std::uint32_t n = 0; n < I; ++n) {
            ws[r * ws_stride + n] = ref::project_one(gw.data() + static_cast<std::size_t>(n) * gb, H, c.data() + r * H, d.data(), rows, static_cast<std::uint32_t>(r));
            ws[r * ws_stride + I + n] = ref::project_one(uw.data() + static_cast<std::size_t>(n) * gb, H, c.data() + r * H, d.data(), rows, static_cast<std::uint32_t>(r));
        }
        ref::shared_activation_row(&ws[r * ws_stride], I, &in[r * H], sc.data(), H, ex);
    });
    std::vector<std::int8_t> c2(static_cast<std::size_t>(rows) * I);
    std::vector<float> d2(static_cast<std::size_t>(I / 32U) * rows);
    ref::quantize(rows, I, ws.data(), ws_stride, c2.data(), d2.data(), rows);
    out.assign(static_cast<std::size_t>(rows) * H, 0);
    parallel_for(rows, [&](std::size_t r) {
        for (std::uint32_t n = 0; n < H; ++n) {
            const float y = ref::project_one(dwn.data() + static_cast<std::size_t>(n) * db, I, c2.data() + r * I, d2.data(), rows, static_cast<std::uint32_t>(r));
            out[r * H + n] = ref::f2bf(ref::amd_mul(ws[r * ws_stride + I], y));
        }
    });
}

static void gpu_shared_tests(const strix::qwen_gguf::Model* model, const strix::cuda::amdmath::Tables& tables, const ref::HostExpf& ex) {
    // activation on synthetic gates over a wide range (exp overflow/underflow, subnormals, signed zeros)
    {
        const std::uint32_t rows = 64, I = 640, H = 2560;
        const std::uint64_t stride = 2 * I + 7;
        std::vector<float> ws(rows * stride);
        Rng g(4242);
        for (auto& v : ws) {
            const std::uint32_t k = g.u32() % 16U;
            if (k == 0) v = static_cast<float>(g.uni() * 240.0 - 120.0);
            else if (k == 1) v = ref::bits_float(g.u32() & 0x807fffffU);
            else if (k == 2) v = (g.u32() & 1) ? 0.0F : -0.0F;
            else v = g.normal() * std::ldexp(1.0F, static_cast<int>(g.u32() % 12U) - 4);
        }
        std::vector<std::uint16_t> in(rows * H);
        for (auto& v : in) v = ref::f2bf(g.normal());
        std::vector<float> sc(H);
        for (auto& v : sc) v = g.normal() * 0.05F;
        for (std::uint32_t r = 0; r < 8; ++r) for (std::uint32_t f = 0; f < H; ++f) in[r * H + f] = ref::f2bf(g.normal() * 200.0F);  // saturated gates
        Dev<float> dws(ws);
        Dev<std::uint16_t> din(in);
        Dev<float> dsc(sc);
        CK(qd::shared_activation(nullptr, rows, dws.p, stride, din.p, H, dsc.p, I, H, tables));
        CK(cudaDeviceSynchronize());
        const auto got = dws.down();
        auto expect = ws;
        for (std::uint32_t r = 0; r < rows; ++r) ref::shared_activation_row(&expect[r * stride], I, &in[r * H], sc.data(), H, ex);
        std::size_t bad = 0, first = SIZE_MAX;
        for (std::size_t i = 0; i < got.size(); ++i) {
            const bool nan_both = got[i] != got[i] && expect[i] != expect[i];
            if (!nan_both && ref::float_bits(got[i]) != ref::float_bits(expect[i])) { ++bad; if (first == SIZE_MAX) first = i; }
        }
        check(bad == 0, "shared activation synthetic (" + std::to_string(got.size()) + " floats, mismatches " + std::to_string(bad) +
                            (bad ? " first " + std::to_string(first) + " gpu " + hex(ref::float_bits(got[first])) + " host " + hex(ref::float_bits(expect[first])) : "") + ")");
    }
    if (!model) return;
    const auto gw = model->bytes("blk.0.ffn_gate_shexp.weight"), uw = model->bytes("blk.0.ffn_up_shexp.weight"),
               dwn = model->bytes("blk.0.ffn_down_shexp.weight"), scb = model->bytes("blk.0.ffn_gate_inp_shexp.weight");
    std::vector<float> sc(scb.size() / 4);
    std::memcpy(sc.data(), scb.data(), scb.size());
    const std::uint32_t H = 2560, I = 640;
    Dev<std::uint8_t> dg(gw), du(uw), dd(dwn);
    Dev<float> dsc(sc);
    for (std::uint32_t rows : {1U, 64U, 257U}) {
        const std::uint64_t ws_stride = 2 * I;
        std::vector<std::uint16_t> in(static_cast<std::size_t>(rows) * H);
        Rng g(rows * 31U);
        for (auto& v : in) v = ref::f2bf(g.normal() * 0.7F);
        Dev<std::uint16_t> din(in);
        Dev<float> dws(static_cast<std::size_t>(rows) * ws_stride);
        Dev<std::uint16_t> dout(static_cast<std::size_t>(rows) * H);
        Dev<std::int8_t> codes(static_cast<std::size_t>(rows) * H);
        Dev<float> deltas(static_cast<std::size_t>(H / 32U) * rows);
        qd::SharedExpert w{dg.p, du.p, dd.p, dsc.p, H, I};
        CK(qd::shared_expert(nullptr, rows, w, din.p, H, dws.p, ws_stride, dout.p, H, codes.p, deltas.p, rows, tables));
        CK(cudaDeviceSynchronize());
        std::vector<float> ews;
        std::vector<std::uint16_t> eout;
        host_shared_expert(rows, in, gw, uw, dwn, sc, H, I, ws_stride, ex, ews, eout);
        const auto gws = dws.down();
        const auto gout = dout.down();
        const bool wok = std::memcmp(gws.data(), ews.data(), ews.size() * 4) == 0;
        const bool ook = std::memcmp(gout.data(), eout.data(), eout.size() * 2) == 0;
        std::string what = "shared expert real blk.0 rows=" + std::to_string(rows);
        if (!wok) for (std::size_t i = 0; i < ews.size(); ++i) if (ref::float_bits(gws[i]) != ref::float_bits(ews[i])) {
            what += " ws[" + std::to_string(i) + "] gpu " + hex(ref::float_bits(gws[i])) + " host " + hex(ref::float_bits(ews[i])); break; }
        if (!ook) for (std::size_t i = 0; i < eout.size(); ++i) if (gout[i] != eout[i]) {
            what += " out[" + std::to_string(i) + "] gpu " + hex(gout[i]) + " host " + hex(eout[i]); break; }
        check(wok && ook, what);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
static void bench() {
    const std::uint32_t rows = 8192, width = 2560;
    const auto w0 = make_q8(10240, width, 1), w1 = make_q8(6144, width, 2);
    Dev<std::uint8_t> d0(w0), d1(w1);
    const auto x = make_rows(rows, width, true, 3, false);
    Dev<std::uint16_t> in(to_bf16(x));
    Dev<std::int8_t> codes(static_cast<std::size_t>(rows) * width);
    Dev<float> deltas(static_cast<std::size_t>(width / 32U) * rows);
    Dev<float> o0(static_cast<std::size_t>(rows) * 10240), o1(static_cast<std::size_t>(rows) * 6144);
    qd::Site site{};
    site.store = qd::Store::f32_round_bf16; site.width = width; site.segments = 2;
    site.segment[0] = {d0.p, 10240, o0.p, 10240};
    site.segment[1] = {d1.p, 6144, o1.p, 6144};
    CK(qd::quantize(nullptr, rows, width, qd::Input::bf16, in.p, width, codes.p, deltas.p, rows));
    CK(qd::project(nullptr, site, rows, codes.p, deltas.p, rows));
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    const int iters = 10;
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) CK(qd::project(nullptr, site, rows, codes.p, deltas.p, rows));
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    const double mac = static_cast<double>(rows) * width * 16384.0;
    std::printf("bench project 8192 x 2560 x 16384: %.3f ms/iter, %.1f GMAC/s\n", ms / iters, mac / (ms / iters * 1e-3) / 1e9);
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) CK(qd::quantize(nullptr, rows, width, qd::Input::bf16, in.p, width, codes.p, deltas.p, rows));
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&ms, a, b));
    std::printf("bench quantize 8192 x 2560 (bf16): %.3f ms/iter\n", ms / iters);
}

// --bench-rows: decode-shaped projections (the Qwen sites, rows 1/8/9/16) with L2-cold weights: the weights rotate over
// copies totalling >= 384 MB (the sm_120 L2 holds a whole site). STRIX_QRTX_DENSE_ROWS16=0/1 picks the 9..16 form.
static void bench_rows() {
    for (const auto& cs : kCases) {
        std::size_t bytes = 0;
        for (auto o : cs.outs) bytes += static_cast<std::size_t>(o) * (cs.width / 32U) * 34U;
        const int copies = static_cast<int>((384ULL << 20) / bytes + 1U);
        std::vector<std::unique_ptr<Dev<std::uint8_t>>> w;
        for (int c = 0; c < copies; ++c)
            for (std::size_t m = 0; m < cs.outs.size(); ++m) w.push_back(std::make_unique<Dev<std::uint8_t>>(make_q8(cs.outs[m], cs.width, 100U * c + m)));
        const std::uint32_t rmax = 16U, width = cs.width;
        const auto x = make_rows(rmax, width, true, 9, false);
        Dev<std::uint16_t> in(to_bf16(x));
        Dev<std::int8_t> codes(static_cast<std::size_t>(rmax) * width);
        Dev<float> deltas(static_cast<std::size_t>(width / 32U) * rmax);
        std::vector<std::unique_ptr<Dev<float>>> outs;
        for (auto o : cs.outs) outs.push_back(std::make_unique<Dev<float>>(static_cast<std::size_t>(rmax) * o));
        std::vector<float> g(rmax * 3U, 0.5F);
        Dev<float> gate(g);
        CK(qd::quantize(nullptr, rmax, width, qd::Input::bf16, in.p, width, codes.p, deltas.p, rmax));
        std::vector<qd::Site> sites(copies);
        for (int c = 0; c < copies; ++c) {
            auto& site = sites[c];
            site.store = cs.store; site.width = width; site.segments = static_cast<std::uint32_t>(cs.outs.size());
            for (std::size_t m = 0; m < cs.outs.size(); ++m)
                site.segment[m] = {w[c * cs.outs.size() + m]->p, cs.outs[m], outs[m]->p, cs.outs[m]};
            if (cs.store == qd::Store::bf16_gated) { site.gate = gate.p; site.gate_stride = 3U; }
        }
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        for (std::uint32_t rows : {1U, 8U, 9U, 16U}) {
            for (int c = 0; c < copies; ++c) CK(qd::project(nullptr, sites[c], rows, codes.p, deltas.p, rmax));
            const int iters = 40 * copies;
            CK(cudaEventRecord(a));
            for (int i = 0; i < iters; ++i) CK(qd::project(nullptr, sites[i % copies], rows, codes.p, deltas.p, rmax));
            CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
            float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
            const double us = 1000.0 * ms / iters;
            std::printf("bench-rows %-16s rows %2u: %7.2f us/call  %6.0f GB/s (weights %.1f MB)\n", cs.name, rows, us,
                        static_cast<double>(bytes) / (us * 1e-6) / 1e9, bytes / 1e6);
        }
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    }
}

// --bench-shape M,K,N (repeatable): one projection site of M rows x K width x N outputs (one segment), best-of iterations.
static void bench_shape(std::uint32_t rows, std::uint32_t width, std::uint32_t outputs, int iters) {
    const auto w = make_q8(outputs, width, 7);
    Dev<std::uint8_t> dw(w);
    const auto x = make_rows(rows, width, true, 3, false);
    Dev<std::uint16_t> in(to_bf16(x));
    Dev<std::int8_t> codes(static_cast<std::size_t>(rows) * width);
    Dev<float> deltas(static_cast<std::size_t>(width / 32U) * rows);
    Dev<float> o(static_cast<std::size_t>(rows) * outputs);
    qd::Site site{};
    site.store = qd::Store::f32_round_bf16; site.width = width; site.segments = 1;
    site.segment[0] = {dw.p, outputs, o.p, outputs};
    CK(qd::quantize(nullptr, rows, width, qd::Input::bf16, in.p, width, codes.p, deltas.p, rows));
    CK(qd::project(nullptr, site, rows, codes.p, deltas.p, rows));
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    float best = 1e30F;
    for (int i = 0; i < iters; ++i) {
        CK(cudaEventRecord(a));
        CK(qd::project(nullptr, site, rows, codes.p, deltas.p, rows));
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
        best = ms < best ? ms : best;
    }
    const double mac = static_cast<double>(rows) * width * outputs;
    std::vector<std::uint8_t> host(o.n * sizeof(float));
    CK(cudaMemcpy(host.data(), o.p, host.size(), cudaMemcpyDeviceToHost));
    std::uint64_t h = 1469598103934665603ULL;   // FNV-1a over the output bytes: byte identity across kernel variants
    for (const std::uint8_t c : host) h = (h ^ c) * 1099511628211ULL;
    std::printf("bench-shape %u x %u x %u: best %.3f ms, %.1f TOPS (int8, 2 ops/MAC), out fnv %016llx\n", rows, width, outputs, best,
                2.0 * mac / (best * 1e-3) / 1e12, static_cast<unsigned long long>(h));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
}

static void list_fixture(const std::string& path) {
    strix::qwen_fixture::Dump dump(path);
    std::map<std::tuple<std::uint32_t, bool, std::uint32_t>, const strix::qwen_fixture::Record*> first;
    for (const auto& r : dump.records()) if (r.row == 0) first.emplace(std::make_tuple(r.step, r.is_read, r.index), &r);
    std::printf("fixture %s: %zu records\n", path.c_str(), dump.records().size());
    for (const auto& [k, r] : first)
        std::printf("  step %u %s[%u] op %u layer %d rows %u bytes/row %llu kernel '%s'\n", r->step, r->is_read ? "read" : "write",
                    r->index, r->op, r->layer == UINT32_MAX ? -1 : static_cast<int>(r->layer), r->rows,
                    static_cast<unsigned long long>(r->bytes), r->kernel_id.c_str());
}

// Replay of every shared_expert step (op 11) of a STRIX_QRTX_STEP_DUMP fixture: read 0 = BF16 MoE input rows, write 0 =
// BF16 shared-expert output rows. GPU (and the host reference) vs the HIP bytes, every row, every byte.
static void replay_shared(const std::string& path, const strix::qwen_gguf::Model& model, const strix::cuda::amdmath::Tables& tables,
                          const ref::HostExpf& ex) {
    strix::qwen_fixture::Dump dump(path);
    std::map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> steps;   // step -> (layer, rows)
    for (const auto& r : dump.records())
        if (r.op == 11U && r.kernel_id.find("shared_expert") != std::string::npos) steps[r.step] = {r.layer, r.rows};
    const std::uint32_t H = 2560, I = 640;
    std::size_t gpu_equal = 0, host_equal = 0, bytes = 0;
    for (const auto& [step, lr] : steps) {
        const auto [layer, rows] = lr;
        const auto inb = dump.load_rows(step, true, 0), outb = dump.load_rows(step, false, 0);
        if (inb.size() != static_cast<std::size_t>(rows) * H * 2U || outb.size() != inb.size()) {
            check(false, "replay shared step " + std::to_string(step) + " unexpected sizes"); continue; }
        std::vector<std::uint16_t> in(inb.size() / 2), expect(outb.size() / 2);
        std::memcpy(in.data(), inb.data(), inb.size()); std::memcpy(expect.data(), outb.data(), outb.size());
        const std::string L = "blk." + std::to_string(layer) + ".";
        const auto gw = model.bytes(L + "ffn_gate_shexp.weight"), uw = model.bytes(L + "ffn_up_shexp.weight"),
                   dwn = model.bytes(L + "ffn_down_shexp.weight"), scb = model.bytes(L + "ffn_gate_inp_shexp.weight");
        std::vector<float> sc(scb.size() / 4);
        std::memcpy(sc.data(), scb.data(), scb.size());
        Dev<std::uint8_t> dg(gw), du(uw), dd(dwn);
        Dev<float> dsc(sc);
        Dev<std::uint16_t> din(in);
        Dev<float> dws(static_cast<std::size_t>(rows) * 2U * I);
        Dev<std::uint16_t> dout(static_cast<std::size_t>(rows) * H);
        Dev<std::int8_t> codes(static_cast<std::size_t>(rows) * H);
        Dev<float> deltas(static_cast<std::size_t>(H / 32U) * rows);
        qd::SharedExpert w{dg.p, du.p, dd.p, dsc.p, H, I};
        CK(qd::shared_expert(nullptr, rows, w, din.p, H, dws.p, 2U * I, dout.p, H, codes.p, deltas.p, rows, tables));
        CK(cudaDeviceSynchronize());
        const auto got = dout.down();
        std::vector<float> hws;
        std::vector<std::uint16_t> hout;
        host_shared_expert(rows, in, gw, uw, dwn, sc, H, I, 2U * I, ex, hws, hout);
        bytes += outb.size();
        std::string what = "replay shared_expert step " + std::to_string(step) + " layer " + std::to_string(layer) + " rows " + std::to_string(rows);
        bool gok = true, hok = true;
        for (std::size_t i = 0; i < expect.size(); ++i) {
            if (gok && got[i] != expect[i]) { gok = false; what += " GPU first diff row " + std::to_string(i / H) + " out " + std::to_string(i % H) +
                " cuda " + hex(got[i]) + " hip " + hex(expect[i]); }
            if (hok && hout[i] != expect[i]) { hok = false; what += " HOST first diff row " + std::to_string(i / H) + " out " + std::to_string(i % H) +
                " host " + hex(hout[i]) + " hip " + hex(expect[i]); }
        }
        gpu_equal += gok; host_equal += hok;
        check(gok, what);
    }
    std::printf("fixture shared_expert replay: %zu steps, %zu bytes compared; CUDA equal in %zu, host reference equal in %zu\n",
                steps.size(), bytes, gpu_equal, host_equal);
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // a crash must not lose the log
#ifdef STRIX_WFMT_CAPTURE
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string_view(argv[i]) != "--wfmt-backend-capture") continue;
        wfmt_capture_directory = argv[i + 1];
        CK(cudaSetDevice(0));
        // Boundary rows, output tails, even/odd Q8 group pairs and every store.
        // Small synthetic tensors keep this a bounded pre-model GPU gate.
        for (std::uint32_t store = 0; store < 4; ++store) {
            const Case cs{"wfmt", static_cast<qd::Store>(store), qd::Input::bf16, 256U, {17U, 129U}};
            std::vector<std::vector<std::uint8_t>> weights;
            for (std::size_t m = 0; m < cs.outs.size(); ++m) weights.push_back(make_q8(cs.outs[m], cs.width, 90U + m));
            for (std::uint32_t rows : {1U,2U,7U,8U,9U,31U,32U,33U,63U,64U,65U,127U,128U,129U})
                run_projection(cs, weights, rows, rows + store * 101U, true, 129U);
        }
        // A single segment starts at a tile boundary and exercises tile2;
        // the preceding split segments exercise the raw136 fallback.
        for (std::uint32_t store = 0; store < 4; ++store) {
            const Case cs{"wfmt-tile2", static_cast<qd::Store>(store), qd::Input::bf16, 256U, {129U}};
            const std::vector<std::vector<std::uint8_t>> weights{make_q8(129U, 256U, 92U)};
            for (std::uint32_t rows : {1U,2U,7U,8U,9U,31U,32U,33U,63U,64U,65U,127U,128U,129U})
                run_projection(cs, weights, rows, rows + store * 101U, true, 129U);
        }
        std::printf("wfmt CUDA dense: %zu production records, %d failures\n", wfmt_capture_index, g_fail);
        return g_fail ? 1 : 0;
    }
#endif
    bool gpu = false, do_bench = false, do_bench_rows = false;
    std::vector<std::array<std::uint32_t, 3>> shapes;
    std::string fixture;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gpu") gpu = true;
        else if (a == "--bench") do_bench = true;
        else if (a == "--bench-rows") do_bench_rows = true;
        else if (a == "--bench-shape" && i + 1 < argc) {
            std::array<std::uint32_t, 3> v{};
            if (std::sscanf(argv[++i], "%u,%u,%u", &v[0], &v[1], &v[2]) == 3) shapes.push_back(v);
        }
        else if (a == "--fixture" && i + 1 < argc) fixture = argv[++i];
        else if (a == "--rows16" && i + 1 < argc) {
            const std::string arm = argv[++i];
            if (arm != "0" && arm != "1") return 2;
            g_rows16 = arm == "1" ? 1 : 0;
        }
    }
    if (g_rows16 >= 0) {
        CK(cudaSetDevice(0));
        gpu_projection_synthetic();
        return g_fail ? 1 : 0;
    }
    if (!shapes.empty()) {
        CK(cudaSetDevice(0));
        for (const auto& v : shapes) bench_shape(v[0], v[1], v[2], 8);
        return g_fail ? 1 : 0;
    }
    host_tests();
    std::unique_ptr<strix::qwen_gguf::Model> model;
    try {
        model = std::make_unique<strix::qwen_gguf::Model>();
        for (const char* n : {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight", "blk.0.ffn_gate_shexp.weight",
                              "blk.0.ffn_up_shexp.weight", "blk.0.ffn_down_shexp.weight", "blk.3.attn_q.weight", "blk.3.attn_output.weight"})
            check(model->tensor(n).ggml_type == 8U, std::string("gguf Q8_0 ") + n);
        check(model->tensor("blk.0.ffn_gate_inp_shexp.weight").ggml_type == 0U, "gguf F32 ffn_gate_inp_shexp");
        // host reference vs FP64 dequantised dot on real weights (sanity of the reference itself)
        const auto w = model->bytes("blk.0.ssm_out.weight");
        const auto x = make_rows(2, 6144, false, 11, false);
        std::vector<std::int8_t> c(2 * 6144);
        std::vector<float> d(192 * 2);
        ref::quantize(2, 6144, x.data(), 6144, c.data(), d.data(), 2);
        double worst = 0;
        for (std::uint32_t n = 0; n < 2560; n += 37) {
            const float y = ref::project_one(w.data() + n * 6528ULL, 6144, c.data(), d.data(), 2, 0);
            double e = 0, mag = 0;
            for (std::uint32_t b = 0; b < 192; ++b) {
                const std::uint8_t* blk = w.data() + n * 6528ULL + 34U * b;
                const double dw = ref::half_to_float(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
                for (int k = 0; k < 32; ++k) { const double t = dw * static_cast<std::int8_t>(blk[2 + k]) * x[32U * b + k]; e += t; mag += std::fabs(t); }
            }
            worst = std::max(worst, std::fabs(y - e) / (mag + 1e-30));
        }
        check(worst < 0.02, "host reference ~ fp64 dequant dot on ssm_out (worst rel " + std::to_string(worst) + ")");
    } catch (const std::exception& e) {
        std::printf("note: gguf unavailable (%s); real-weight tests skipped\n", e.what());
        model.reset();
    }
    if (gpu || do_bench || do_bench_rows) {
        CK(cudaSetDevice(0));
        cudaDeviceProp p{};
        CK(cudaGetDeviceProperties(&p, 0));
        std::printf("device: %s sm_%d%d\n", p.name, p.major, p.minor);
    }
    if (gpu) {
        std::printf("note: --gpu inherits dispatch policy; RP16 qualification requires both qwen_dense_rows16 CTests\n");
        const auto t0 = std::chrono::steady_clock::now();
        gpu_quantize_tests();
        gpu_projection_synthetic();
        if (model) gpu_projection_real(*model);
        const char* td = std::getenv("STRIX_QRTX_TABLES");
        const std::string tdir = td && *td ? td : "qrtx-build\\tables";
        auto tables = strix::cuda::amdmath::load_tables(tdir);
        ref::HostExpf ex(tdir + "\\miss_expf.bin");
        std::printf("host expf miss list: %zu inputs\n", ex.size());
        gpu_shared_tests(model.get(), tables, ex);
        strix::cuda::amdmath::free_tables(tables);
        std::printf("gpu tests %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    if (do_bench) bench();
    if (do_bench_rows) bench_rows();
    if (!fixture.empty()) {
        if (std::getenv("QWEN_DENSE_LIST")) list_fixture(fixture);
        if (model) {
            CK(cudaSetDevice(0));
            const char* td = std::getenv("STRIX_QRTX_TABLES");
            const std::string tdir = td && *td ? td : "qrtx-build\\tables";
            auto tables = strix::cuda::amdmath::load_tables(tdir);
            ref::HostExpf ex(tdir + "\\miss_expf.bin");
            replay_shared(fixture, *model, tables, ex);
            strix::cuda::amdmath::free_tables(tables);
        }
    }
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}

