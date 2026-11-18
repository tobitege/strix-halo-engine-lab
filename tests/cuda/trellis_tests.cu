// Native bounded stage-level tests. CPU oracles are generated independently by
// tools/trellis_reference.py. This executable never downloads or loads a model.
#include "strix/cuda/trellis.hpp"

#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace c = strix::ds41::cuda;
namespace t = c::trellis;
namespace {
void check(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }
template <class F> void rejects(F f, const char* what) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(std::string("invalid descriptor accepted: ") + what);
}
void contract_tests() {
    t::Linear v;
    v.weights = {reinterpret_cast<const std::uint16_t*>(0x1000000), reinterpret_cast<const std::uint16_t*>(0x2000000),
        reinterpret_cast<const std::uint16_t*>(0x3000000), nullptr, 4096, 256, 256, 0, 128, 128, 2, 0x83dcd12dU};
    v.rows = 1; v.x_stride = v.y_stride = 128;
    v.x = reinterpret_cast<const float*>(0x4000000); v.x_bytes = 512;
    v.y = reinterpret_cast<float*>(0x5000000); v.y_bytes = 512;
    v.workspace = reinterpret_cast<void*>(0x6000000); v.workspace_bytes = 2304;
    rejects([&] { (void)t::workspace_bytes(v); }, "default disabled");
    v.mode = t::Mode::hip_bitwise_required;
    rejects([&] { (void)t::workspace_bytes(v); }, "unqualified identity");
    v.mode = t::Mode::gemv_quality;
    check(t::workspace_bytes(v) == 2304, "workspace formula");
    unsigned tests = 2;
    #define BAD(FIELD, VALUE) { auto bad = v; bad.FIELD = VALUE; rejects([&] { t::enqueue_linear(bad, nullptr); }, #FIELD); ++tests; }
    BAD(weights.input, 129); BAD(weights.output, 0); BAD(weights.bits, 0); BAD(weights.bits, 9);
    BAD(weights.marker, 1); BAD(weights.packed_bytes, 4095); BAD(weights.suh_bytes, 255);
    BAD(weights.svh_bytes, 255); BAD(weights.bias_bytes, 512); BAD(rows, 9); BAD(splits, 0);
    BAD(splits, 9); BAD(x_stride, 127); BAD(y_stride, 127); BAD(x_bytes, 511);
    BAD(y_bytes, 511); BAD(workspace_bytes, 2303); BAD(workspace, reinterpret_cast<void*>(0x6000002));
    BAD(y, const_cast<float*>(v.x)); BAD(workspace, v.y); BAD(weights.suh, reinterpret_cast<const std::uint16_t*>(v.y));
    BAD(x, nullptr); BAD(weights.packed, nullptr);
    BAD(y_bytes, std::numeric_limits<std::size_t>::max());
    #undef BAD
    auto empty = v; empty.rows = 0; empty.x = nullptr; empty.y = nullptr; empty.workspace = nullptr;
    check(t::workspace_bytes(empty) == 0, "empty workspace");
    t::enqueue_linear(empty, nullptr);  // must not initialize CUDA
    std::printf("PASS CPU contracts %u rejections + workspace + empty no-op (no CUDA calls)\n", tests);
}

class Reader {
public:
    explicit Reader(const std::filesystem::path& p) : in(p, std::ios::binary) {
        check(bool(in), "cannot open fixture " + p.string());
        const auto n = std::filesystem::file_size(p);
        check(n >= 40 && n <= 256ULL * 1024 * 1024, "fixture size outside 40 B..256 MiB");
        remaining = n;
    }
    template <class T> std::vector<T> take(std::size_t n) {
        check(n <= remaining / sizeof(T), "truncated fixture");
        std::vector<T> out(n);
        if (n) in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n * sizeof(T)));
        check(bool(in), "read fixture"); remaining -= n * sizeof(T); return out;
    }
    std::ifstream in;
    std::uintmax_t remaining = 0;
};
struct Case {
    std::uint32_t i, o, k, rows, splits, xs, ys;
    std::vector<std::uint16_t> packed, suh, svh, q, a;
    std::vector<float> bias, x, p, y;
    explicit Case(const std::filesystem::path& path) {
        Reader f(path);
        const auto magic = f.take<char>(8);
        check(std::memcmp(magic.data(), "TRLCASE1", 8) == 0, "fixture magic");
        auto h = f.take<std::uint32_t>(8);
        i = h[0]; o = h[1]; k = h[2]; rows = h[3]; splits = h[4]; xs = h[6]; ys = h[7];
        const auto pb = t::packed_bytes(i, o, k);
        check(rows > 0 && rows <= 64 && splits > 0 && splits <= std::min(32U, i / 16) && h[5] <= 1 &&
            xs >= i && xs <= i + 64 && ys >= o && ys <= o + 64, "fixture geometry");
        const auto expected = pb + 2ULL * (i + o) + 4ULL * h[5] * o + 4ULL * rows * xs + 2ULL * i * o +
            2ULL * rows * i + 4ULL * splits * rows * o + 4ULL * rows * ys;
        check(expected == f.remaining, "fixture exact payload length");
        packed = f.take<std::uint16_t>(pb / 2); suh = f.take<std::uint16_t>(i); svh = f.take<std::uint16_t>(o);
        bias = f.take<float>(h[5] * o); x = f.take<float>(std::size_t{rows} * xs);
        q = f.take<std::uint16_t>(std::size_t{i} * o); a = f.take<std::uint16_t>(std::size_t{rows} * i);
        p = f.take<float>(std::size_t{splits} * rows * o); y = f.take<float>(std::size_t{rows} * ys);
        check(f.remaining == 0, "fixture trailing bytes");
    }
};
template <class T>
void equal(const std::vector<T>& got, const T* want, std::size_t count, const std::string& label) {
    check(got.size() >= count, label + " short output");
    for (std::size_t i = 0; i < count; ++i)
        if (std::memcmp(got.data() + i, want + i, sizeof(T)))
            throw std::runtime_error(label + " bits differ at " + std::to_string(i));
}
float hf(std::uint16_t h) { return __half2float(__ushort_as_half(h)); }

// Numerical envelope for MMA partials, independent FP64 dot from saved FP16 A/Q.
// Propagate each partial's bound through ordered split folding, FP32 H128,
// normalization and svh; compare each output absolutely, including cancellation.
void mma_check(const Case& f, unsigned rows, const std::vector<float>& partial, const std::vector<float>& y) {
    std::vector<double> core(std::size_t{rows} * f.o, 0), bound(core.size(), 0);
    const double u = std::ldexp(1.0, -24);
    for (unsigned s = 0; s < f.splits; ++s) {
        const unsigned b = f.i / 16 * s / f.splits * 16, e = f.i / 16 * (s + 1) / f.splits * 16;
        const double gamma = (2.0 * (e - b) * u) / (1 - 2.0 * (e - b) * u);
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned n = 0; n < f.o; ++n) {
                double dot = 0, norm = 0;
                for (unsigned i = b; i < e; ++i) {
                    const double p = double(hf(f.a[std::size_t{r} * f.i + i])) * double(hf(f.q[std::size_t{i} * f.o + n]));
                    dot += p; norm += std::abs(p);
                }
                const auto j = std::size_t{r} * f.o + n;
                const double err = gamma * norm + 1e-30;
                check(std::isfinite(partial[std::size_t{s} * rows * f.o + j]) &&
                    std::abs(double(partial[std::size_t{s} * rows * f.o + j]) - dot) <= err, "MMA partial FP64 bound");
                bound[j] += err + u * (std::abs(core[j]) + std::abs(dot) + err);
                core[j] += dot;
            }
    }
    const double hn = double(float(0x1.6a09e6p-4F));
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned n0 = 0; n0 < f.o; n0 += 128) {
            std::array<double, 128> z{}, er{};
            for (unsigned j = 0; j < 128; ++j) { z[j] = core[std::size_t{r} * f.o + n0 + j]; er[j] = bound[std::size_t{r} * f.o + n0 + j]; }
            for (unsigned stride = 1; stride < 128; stride *= 2)
                for (unsigned block = 0; block < 128; block += 2 * stride)
                    for (unsigned j = 0; j < stride; ++j) {
                        const auto lo = block + j, hi = lo + stride;
                        const double a = z[lo], b = z[hi];
                        const double ee = er[lo] + er[hi] + u * (std::abs(a) + std::abs(b) + er[lo] + er[hi]);
                        z[lo] = a + b; z[hi] = a - b; er[lo] = er[hi] = ee;
                    }
            for (unsigned j = 0; j < 128; ++j) {
                const double scale = double(hf(f.svh[n0 + j]));
                double value = z[j] * hn * scale;
                double error = er[j] * hn * std::abs(scale) + 4 * u * std::abs(value) + 1e-25;
                if (!f.bias.empty()) { value += f.bias[n0 + j]; error += u * (std::abs(value) + error); }
                const float got = y[std::size_t{r} * f.ys + n0 + j];
                check(std::isfinite(got) && std::abs(double(got) - value) <= error, "MMA epilogue propagated bound");
            }
        }
    }
}

void run_case(const std::filesystem::path& path, c::Stream& stream) {
    Case f(path);
    c::DeviceBuffer<std::uint16_t> packed, suh, svh, q(std::size_t{f.i} * f.o);
    c::DeviceBuffer<float> bias, x, y(std::size_t{f.rows} * f.ys);
    packed.upload(f.packed); suh.upload(f.suh); svh.upload(f.svh); bias.upload(f.bias); x.upload(f.x);
    t::Weights w{packed.get(), suh.get(), svh.get(), bias.get(), f.packed.size()*2, f.suh.size()*2, f.svh.size()*2,
        f.bias.size()*4, f.i, f.o, f.k, 0x83dcd12dU};
    t::enqueue_decode(w, q.get(), q.size()*2, stream.get()); stream.sync();
    equal(q.download(), f.q.data(), f.q.size(), "Q " + path.string());
    unsigned runs = 0;
    for (auto mode : {t::Mode::gemv_quality, t::Mode::mma_prefill_quality}) {
        const std::vector<unsigned> counts = mode == t::Mode::gemv_quality ? std::vector<unsigned>{1,2,3,4,5,6,7,8} : std::vector<unsigned>{1,8,15,16,17,31,32,33,64};
        for (unsigned rows : counts) {
            if (rows > f.rows) continue;
            t::Linear v{w, x.get(), y.get(), x.size()*4, y.size()*4, f.xs, f.ys, rows, f.splits, mode};
            c::DeviceBuffer<std::uint8_t> scratch(t::workspace_bytes(v));
            v.workspace = scratch.get(); v.workspace_bytes = scratch.size();
            std::vector<float> initial(y.size(), -123456.0F);
            y.upload(initial);
            t::enqueue_linear(v, stream.get()); stream.sync();
            const auto got = y.download();
            const auto ao = (std::size_t{2} * rows * f.i + 255) & ~std::size_t{255};
            std::vector<std::uint16_t> actual_a(std::size_t{rows} * f.i);
            std::vector<float> actual_p(std::size_t{f.splits} * rows * f.o);
            STRIX_CUDA_CHECK(cudaMemcpy(actual_a.data(), scratch.get(), actual_a.size()*2, cudaMemcpyDeviceToHost));
            STRIX_CUDA_CHECK(cudaMemcpy(actual_p.data(), scratch.get()+ao, actual_p.size()*4, cudaMemcpyDeviceToHost));
            equal(actual_a, f.a.data(), actual_a.size(), "A");
            for (unsigned r = 0; r < f.rows; ++r)
                for (unsigned n = r < rows ? f.o : 0; n < f.ys; ++n)
                    check(got[std::size_t{r} * f.ys + n] == -123456.0F, "row tail / padding overwritten");
            if (mode == t::Mode::gemv_quality) {
                for (unsigned s = 0; s < f.splits; ++s)
                    for (unsigned r = 0; r < rows; ++r)
                        check(std::memcmp(actual_p.data() + (std::size_t{s}*rows+r)*f.o,
                            f.p.data() + (std::size_t{s}*f.rows+r)*f.o, f.o*4) == 0, "P bits / row invariance");
                equal(got, f.y.data(), std::size_t{rows} * f.ys, "Y GEMV");
            } else mma_check(f, rows, actual_p, got);

            // Replay changing X and poisoned scratch: catches captured stale
            // pointers, skipped writes, old-Y reads and hidden initialization.
            cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
            STRIX_CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
            t::enqueue_linear(v, stream.get());
            STRIX_CUDA_CHECK(cudaStreamEndCapture(stream.get(), &graph));
            STRIX_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            for (unsigned replay = 0; replay < 3; ++replay) {
                auto changed = f.x;
                if (replay == 1) for (auto& value : changed) value = -value;
                x.upload(changed);
                STRIX_CUDA_CHECK(cudaMemsetAsync(scratch.get(), 0xFF, scratch.size(), stream.get()));
                STRIX_CUDA_CHECK(cudaMemsetAsync(y.get(), 0xFF, y.size()*4, stream.get()));
                STRIX_CUDA_CHECK(cudaGraphLaunch(exec, stream.get())); stream.sync();
                auto captured = y.download();
                // Compare against a fresh eager call on exactly that input.
                t::enqueue_linear(v, stream.get()); stream.sync();
                const auto eager = y.download();
                bool changed_any = false;
                for (unsigned r = 0; r < rows; ++r)
                    for (unsigned n = 0; n < f.o; ++n) {
                        const auto j = std::size_t{r}*f.ys+n;
                        check(std::isfinite(captured[j]), "graph nonfinite active output");
                        check(std::memcmp(captured.data()+j, eager.data()+j, 4) == 0, "graph vs changed-X eager bits");
                        changed_any = changed_any || captured[j] != got[j];
                    }
                if (replay == 1) check(changed_any, "graph ignored changed input");
                if (replay != 1)
                    for (unsigned r = 0; r < rows; ++r)
                        check(std::memcmp(captured.data()+std::size_t{r}*f.ys, got.data()+std::size_t{r}*f.ys, f.o*4) == 0, "graph restoration bits");
            }
            STRIX_CUDA_CHECK(cudaGraphExecDestroy(exec)); STRIX_CUDA_CHECK(cudaGraphDestroy(graph));
            x.upload(f.x); ++runs;
        }
    }
    // Non-prefix views exercise real row placement, not just padded storage.
    // They must give the same A/P/Y as those rows in the independent oracle.
    unsigned views = 0;
    for (unsigned first : {1U, 7U, 17U, 25U}) {
        if (first >= f.rows) continue;
        const unsigned rows = std::min(8U, f.rows - first);
        t::Linear v{w, x.get()+std::size_t{first}*f.xs, y.get()+std::size_t{first}*f.ys,
            (x.size()-std::size_t{first}*f.xs)*4, (y.size()-std::size_t{first}*f.ys)*4,
            f.xs, f.ys, rows, f.splits, t::Mode::gemv_quality};
        c::DeviceBuffer<std::uint8_t> scratch(t::workspace_bytes(v));
        v.workspace = scratch.get(); v.workspace_bytes = scratch.size();
        t::enqueue_linear(v, stream.get()); stream.sync();
        const auto got = y.download();
        for (unsigned r = first; r < first + rows; ++r)
            check(std::memcmp(got.data()+std::size_t{r}*f.ys, f.y.data()+std::size_t{r}*f.ys, f.o*4) == 0, "shifted row-view oracle bits");
        ++views;
    }
    std::printf("PASS %s K%u I%u N%u Q/A/P/Y, stride/tails, %u row/mode cases, %u shifted views, changed-X graphs\n", path.filename().string().c_str(), f.k, f.i, f.o, runs, views);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        contract_tests();
        if (argc == 2 && std::string(argv[1]) == "--contract-only") return 0;
        check(argc >= 3, "usage: strix-cuda-trellis-tests --book mul1.bin case.trlcase [...]; or --contract-only");
        check(std::string(argv[1]) == "--book", "exhaustive codebook oracle required");
        check(argc >= 4, "at least one fixture required");
        cudaDeviceProp prop{};
        STRIX_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        check(std::string(prop.name).find("RTX PRO 6000") != std::string::npos && prop.major == 12 && prop.minor == 0,
            "device 0 is not the authorized sm_120 RTX PRO 6000; set PCI_BUS_ID and CUDA_VISIBLE_DEVICES=0");
        const auto device = c::select_device(0);
        std::printf("GPU %s sm_%d%d (correctness only, no timing)\n", device.name.c_str(), device.major, device.minor);
        std::ifstream in(argv[2], std::ios::binary);
        check(bool(in) && std::filesystem::file_size(argv[2]) == 131072, "codebook oracle must contain exactly 65536 halves");
        std::vector<std::uint16_t> expected(65536);
        in.read(reinterpret_cast<char*>(expected.data()), 131072); check(bool(in), "codebook oracle read");
        c::DeviceBuffer<std::uint16_t> book(65536); c::Stream stream;
        t::enqueue_codebook(book.get(), 131072, stream.get()); stream.sync();
        equal(book.download(), expected.data(), expected.size(), "all 65536 codebook states");
        std::printf("PASS exhaustive 65536 mul1 half states\n");
        for (int i = 3; i < argc; ++i) run_case(argv[i], stream);
        std::printf("PASS native compressed linear suite; no model qualification or speed claim\n");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
