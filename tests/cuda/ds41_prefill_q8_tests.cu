// deepseek41 CUDA prefill-shape Q8_0 x A8 (cuda_ds41_prefill_q8.hpp): bitwise against enqueue_q8_0_a8_tiled for the
// five site shapes of a layer (q_b, grouped o_a, K-split o_b, q_a + kv, shared gate + up and down), 1..13, 64, 1000
// and 4096 rows, on synthetic data and on data with edge values (zero groups, extreme / subnormal / signed-zero / Inf /
// NaN scales and deltas, codes -128); odd geometries (few groups, empty lane classes, a count not a multiple of the
// block tile, wider strides, accumulate); the shared-expert entry against enqueue_shared_expert.
// With --bench: device ms of the sites at 2K / 8K / 32K rows in chunks of 1024 rows (as the prefill driver calls them).

#include "strix/cuda/ds41_common.hpp"
#include "strix/cuda/ds41_core.hpp"
#include "strix/cuda/ds41_prefill_q8.hpp"

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

namespace {

int failures = 0;
std::mt19937_64 rng(20261020);

// Bitwise comparison on the device: differing words, the first one, non-finite expected values.
__global__ void compare_kernel(const std::uint32_t* got, const std::uint32_t* want, std::size_t n, unsigned long long* stats) {
    for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x) {
        if ((want[i] & 0x7f800000U) == 0x7f800000U) atomicAdd(&stats[2], 1ULL);
        if (got[i] != want[i]) {
            atomicAdd(&stats[0], 1ULL);
            atomicMin(&stats[1], static_cast<unsigned long long>(i));
        }
    }
}

// Deterministic values in [-1, 1) (the out rows an accumulating call starts from).
__global__ void fill_kernel(float* p, std::size_t n, std::uint32_t seed) {
    for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x; i < n; i += std::size_t{gridDim.x} * blockDim.x) {
        std::uint32_t h = static_cast<std::uint32_t>(i) * 0x9E3779B1U ^ seed;
        h ^= h >> 15U;
        h *= 0x85EBCA77U;
        h ^= h >> 13U;
        p[i] = static_cast<float>(static_cast<std::int32_t>(h >> 8U) - (1 << 23)) * 0x1p-23F;
    }
}

void expect_bits(const std::string& what, const float* got, const float* want, std::size_t n) {
    c::DeviceBuffer<unsigned long long> st(3);
    const std::vector<unsigned long long> init{0ULL, ~0ULL, 0ULL};
    st.upload(init);
    compare_kernel<<<1024, 256>>>(reinterpret_cast<const std::uint32_t*>(got), reinterpret_cast<const std::uint32_t*>(want), n, st.get());
    STRIX_CUDA_CHECK(cudaDeviceSynchronize());
    const auto r = st.download();
    if (r[0]) {
        ++failures;
        float a = 0, b = 0;
        STRIX_CUDA_CHECK(cudaMemcpy(&a, got + r[1], 4, cudaMemcpyDeviceToHost));
        STRIX_CUDA_CHECK(cudaMemcpy(&b, want + r[1], 4, cudaMemcpyDeviceToHost));
        std::printf("FAIL %s: %llu of %zu differ (first %llu: %.9g vs %.9g)\n", what.c_str(), r[0], n, r[1], a, b);
    } else {
        std::printf("ok   %s (%zu values, %llu non-finite)\n", what.c_str(), n, r[2]);
    }
}

template <class T>
c::DeviceBuffer<T> to_device(const std::vector<T>& v) {
    c::DeviceBuffer<T> d(v.size());
    d.upload(v);
    return d;
}

// GGUF Q8_0 rows. Scales: normal halves of |s| ~ 2^-12 .. 2^-3; with edges 1 block in 32 is special (zero group,
// +-0 scale, largest half, subnormal half, Inf, NaN) and codes include -128.
std::vector<std::uint8_t> random_q8(std::size_t rows, std::size_t width, bool edges) {
    std::vector<std::uint8_t> w(rows * width / 32 * 34);
    std::uniform_int_distribution<int> e(3, 12), m(0, 1023), sgn(0, 1), code(-127, 127), code_e(-128, 127), pick(0, 31), kind(0, 6),
        rare(0, 63);
    for (std::size_t b = 0; b < w.size(); b += 34) {
        auto h = static_cast<std::uint16_t>((sgn(rng) << 15) | (e(rng) << 10) | m(rng));
        bool zero = false;
        if (edges && pick(rng) == 0) {
            switch (kind(rng)) {
            case 0: zero = true; h = 0; break;
            case 1: h = 0x8000; break;
            case 2: h = 0x7BFF; break;
            case 3: h = 0x0001; break;
            case 4: h = rare(rng) ? 0x0200 : 0x7C00; break;  // Inf (rare, else a subnormal)
            case 5: h = rare(rng) ? 0x8001 : 0x7E00; break;  // NaN (rare)
            default: h = 0xFBFF; break;
            }
        }
        std::memcpy(&w[b], &h, 2);
        for (int j = 0; j < 32; ++j)
            w[b + 2 + j] = zero ? 0 : static_cast<std::uint8_t>(static_cast<std::int8_t>(edges ? code_e(rng) : code(rng)));
    }
    return w;
}

// A8 rows (codes in [-127, 127], deltas, sums unused by the product). Deltas: |d| ~ 2^-10 .. 2^-4; with edges 1 group
// in 32 is special (zero codes, delta 0 / -0 / FLT_MAX / subnormal / Inf / NaN / 2^100).
struct HostA8 {
    std::vector<std::int8_t> codes;
    std::vector<float> deltas;
    std::vector<std::int32_t> sums;
};
HostA8 random_a8(std::size_t rows, std::size_t stride, bool edges) {
    HostA8 h;
    h.codes.resize(rows * stride);
    h.deltas.resize(rows * stride / 32);
    h.sums.assign(rows * stride / 32, 0);
    std::uniform_int_distribution<int> code(-127, 127), pick(0, 31), kind(0, 7), rare(0, 63);
    std::uniform_real_distribution<float> ex(-10.0F, -4.0F);
    for (auto& x : h.codes) x = static_cast<std::int8_t>(code(rng));
    for (std::size_t g = 0; g < h.deltas.size(); ++g) {
        float d = std::exp2(ex(rng));
        if (edges && pick(rng) == 0) {
            switch (kind(rng)) {
            case 0: std::memset(&h.codes[g * 32], 0, 32); break;
            case 1: d = 0.0F; break;
            case 2: d = -0.0F; break;
            case 3: d = rare(rng) ? 0x1p60F : std::numeric_limits<float>::max(); break;
            case 4: d = std::numeric_limits<float>::denorm_min(); break;
            case 5: d = rare(rng) ? 0x1p-130F : std::numeric_limits<float>::infinity(); break;
            case 6: d = rare(rng) ? -0x1p-8F : std::numeric_limits<float>::quiet_NaN(); break;
            default: d = 0x1p100F; break;
            }
        }
        h.deltas[g] = d;
    }
    return h;
}

struct DeviceA8 {
    c::DeviceBuffer<std::int8_t> codes;
    c::DeviceBuffer<float> deltas;
    c::DeviceBuffer<std::int32_t> sums;
    std::uint32_t rows = 0, stride = 0;
    explicit DeviceA8(const HostA8& h, std::uint32_t rows_, std::uint32_t stride_)
        : codes(to_device(h.codes)), deltas(to_device(h.deltas)), sums(to_device(h.sums)), rows(rows_), stride(stride_) {}
    [[nodiscard]] c::A8Rows view() { return {codes.get(), deltas.get(), sums.get(), rows, stride, stride}; }
};

struct Site {
    const char* name;
    std::uint32_t count, width;   // weight rows, full weight width
    std::uint32_t per_group = 0;  // grouped (o_a)
    std::uint32_t parts = 1;      // K-split (o_b): parts calls with accumulate
    std::uint32_t first = 0;      // first weight row of the view (the weights have first + count rows)
    std::uint32_t x_stride = 0;   // 0: the activation width
    std::uint32_t out_stride = 0; // 0: count
    bool accumulate_into = false; // the first call also accumulates (into a non-zero out)
};

// Runs the site's calls (all parts) with one of the two kernels.
void run_site(const Site& k, const c::Q8Tiled& tiled, c::A8Rows x, float* out, bool fresh, cudaStream_t s) {
    const std::uint32_t groups = k.per_group ? k.count / k.per_group : 1U;
    const std::uint32_t part = k.width / k.parts;
    for (std::uint32_t p = 0; p < k.parts; ++p) {
        c::Q8A8View v;
        v.tiled = tiled;
        v.first = k.first;
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
        v.accumulate = p > 0 || k.accumulate_into;
        v.out = out;
        v.out_stride = k.out_stride;
        if (fresh) pk::enqueue_q8_0_a8_rows(v, s);
        else c::enqueue_q8_0_a8_tiled(v, s);
    }
}

void test_site(const Site& k, const std::vector<std::uint32_t>& row_counts, bool edges, cudaStream_t s) {
    const auto w = random_q8(std::size_t{k.first} + k.count, k.width, edges);
    auto dw = to_device(w);
    c::Q8TiledBuffer tiled;
    tiled.repack(dw.get(), k.first + k.count, k.width, s);
    const std::uint32_t groups = k.per_group ? k.count / k.per_group : 1U;
    const std::uint32_t xin = k.parts > 1 ? k.width : groups * k.width, xs = k.x_stride ? k.x_stride : xin;
    const std::uint32_t os = k.out_stride ? k.out_stride : k.count;
    for (const std::uint32_t rows : row_counts) {
        const auto h = random_a8(rows, xs, edges);
        DeviceA8 x(h, rows, xs);
        const std::size_t n = std::size_t{rows} * os;
        c::DeviceBuffer<float> want(n), got(n);
        fill_kernel<<<1024, 256>>>(want.get(), n, rows);
        fill_kernel<<<1024, 256>>>(got.get(), n, rows);
        STRIX_CUDA_CHECK(cudaDeviceSynchronize());
        run_site(k, tiled.view(), x.view(), want.get(), false, s);
        run_site(k, tiled.view(), x.view(), got.get(), true, s);
        STRIX_CUDA_CHECK(cudaStreamSynchronize(s));
        expect_bits(std::string(k.name) + (edges ? " edges" : "") + " rows " + std::to_string(rows), got.get(), want.get(), n);
    }
}

void test_shared(const std::vector<std::uint32_t>& row_counts, bool edges, cudaStream_t s) {
    const std::uint32_t hidden = 5120, inter = 2304;
    auto dgu = to_device(random_q8(2U * inter, hidden, edges)), dd = to_device(random_q8(hidden, inter, edges));
    c::Q8TiledBuffer tgu, tdown;
    tgu.repack(dgu.get(), 2U * inter, hidden, s);
    tdown.repack(dd.get(), hidden, inter, s);
    for (const std::uint32_t rows : row_counts) {
        // activations from real values (the shared expert's A8 input is quantize_a8 of the norm output)
        std::normal_distribution<float> g(0.0F, 1.0F);
        std::vector<float> xf(std::size_t{rows} * hidden);
        for (auto& f : xf) f = g(rng);
        auto dx = to_device(xf);
        c::A8Buffer a;
        a.resize(rows, hidden);
        c::enqueue_quantize_a8({dx.get(), rows, hidden, 0, a.view()}, s);
        c::DeviceBuffer<std::uint8_t> ws1(c::shared_expert_workspace_bytes(rows, inter)), ws2(c::shared_expert_workspace_bytes(rows, inter));
        c::DeviceBuffer<float> o1(std::size_t{rows} * hidden), o2(std::size_t{rows} * hidden);
        c::SharedExpertView v;
        v.gate_up_tiled = tgu.view();
        v.down_tiled = tdown.view();
        v.x = a.view();
        v.out = o1.get();
        v.workspace = ws1.get();
        c::enqueue_shared_expert(v, s);
        v.out = o2.get();
        v.workspace = ws2.get();
        pk::enqueue_shared_expert(v, s);
        STRIX_CUDA_CHECK(cudaStreamSynchronize(s));
        expect_bits(std::string("shared expert") + (edges ? " edges" : "") + " rows " + std::to_string(rows), o2.get(), o1.get(),
                    std::size_t{rows} * hidden);
    }
}

// ---- bench ------------------------------------------------------------------------------------------------------------
// Minimum over reps of single timed runs (the device may be shared: the minimum is the least disturbed run).
float time_ms(cudaStream_t s, const std::function<void()>& body, int reps) {
    body();  // warm up
    float best = 1e30F;
    for (int i = 0; i < reps; ++i) {
        c::EventTimer t;
        t.start(s);
        body();
        best = std::min(best, t.stop(s));
    }
    return best;
}

void bench(cudaStream_t s) {
    constexpr std::uint32_t C = 1024;
    const Site sites[] = {{"q_b 1280->32768", 32768, 1280},           {"o_a 8x(4096->1024)", 8192, 4096, 1024},
                          {"o_b 2x(4096)->5120", 5120, 8192, 0, 2},    {"qa_kv 5120->1792", 1792, 5120},
                          {"gate_up 5120->4608", 4608, 5120},          {"down 2304->5120", 5120, 2304}};
    std::printf("bench: device ms (min of runs) over T rows in chunks of %u (old = enqueue_q8_0_a8_tiled, new = enqueue_q8_0_a8_rows)\n", C);
    for (const auto& k : sites) {
        const auto w = random_q8(k.count, k.width, false);
        auto dw = to_device(w);
        c::Q8TiledBuffer tiled;
        tiled.repack(dw.get(), k.count, k.width, s);
        const std::uint32_t groups = k.per_group ? k.count / k.per_group : 1U;
        const std::uint32_t xin = k.parts > 1 ? k.width : groups * k.width;
        DeviceA8 x(random_a8(C, xin, false), C, xin);
        c::DeviceBuffer<float> out(std::size_t{C} * k.count);
        for (const std::uint32_t T : {2048U, 8192U, 32768U}) {
            const int reps = T >= 32768U ? 3 : 7;
            const auto run = [&](bool fresh) {
                for (std::uint32_t r0 = 0; r0 < T; r0 += C) run_site(k, tiled.view(), x.view(), out.get(), fresh, s);
            };
            const float a = time_ms(s, [&] { run(false); }, reps), b = time_ms(s, [&] { run(true); }, reps);
            const double tops = 2.0 * T * k.count * (k.parts > 1 ? k.width : k.width) / (b * 1e-3) / 1e12;
            std::printf("  T %5u %-20s old %9.2f ms  new %8.2f ms  x%.2f  (new %.0f TOPS)\n", T, k.name, a, b, a / b, tops);
        }
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
        std::vector<std::uint32_t> rows;
        for (std::uint32_t r = 1; r <= 13; ++r) rows.push_back(r);
        rows.push_back(64);
        rows.push_back(1000);
        rows.push_back(4096);
        const std::vector<std::uint32_t> some{1, 5, 8, 13, 64, 1000};
        const Site sites[] = {{"q_b 1280->32768", 32768, 1280},           {"o_a 8x(4096->1024)", 8192, 4096, 1024},
                              {"o_b 2x(4096)->5120", 5120, 8192, 0, 2},    {"qa_kv 5120->1792", 1792, 5120},
                              {"gate_up 5120->4608", 4608, 5120},          {"down 2304->5120", 5120, 2304}};
        for (const auto& k : sites) test_site(k, rows, false, s);
        for (const auto& k : sites) test_site(k, some, true, s);
        // odd geometry: 3 groups (29 empty lane classes), 33 groups, count not a multiple of 64, first != 0,
        // wider strides, accumulate into a non-zero out
        test_site({"odd 96->80 first 32 strided acc", 80, 96, 0, 1, 32, 128, 100, true}, {1, 7, 65, 300}, false, s);
        test_site({"odd 1056->208", 208, 1056}, {3, 64, 129}, true, s);
        test_site({"odd grouped 2x(64->128)", 256, 64, 128}, {2, 70}, false, s);
        test_shared({1, 7, 13, 64, 1000}, false, s);
        test_shared({9, 300}, true, s);
        std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    }
    if (do_bench) bench(s);
    return failures ? 1 : 0;
}
