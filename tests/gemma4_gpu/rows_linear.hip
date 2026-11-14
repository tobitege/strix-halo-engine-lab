// Byte gate of the exact many-row linears against the eight-row g4-v1 passes.
#include <hip/hip_runtime.h>
#include "gemma4/kernels/rows_linear.hpp"
#include "gemma4/kernels/internal.hpp"
#include "fixture.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <string_view>

using namespace strix::gemma4;
using namespace strix::gemma4::gpu;
namespace {
template <class T> Buffer device(const std::vector<T>& values) {
    Buffer b(values.size() * sizeof(T));
    b.upload(values.data(), b.size());
    return b;
}
std::vector<float> read(const Buffer& b, std::size_t count) {
    std::vector<float> v(count);
    b.download(v.data(), count * sizeof(float));
    return v;
}
// Mixed magnitudes, exact zeros of both signs and rounding-heavy mantissas.
std::vector<float> inputs(std::size_t n, unsigned seed) {
    std::vector<float> x(n);
    std::mt19937 rng(seed);
    for (auto& a : x) {
        const auto kind = rng() % 16U;
        if (kind == 0)
            a = (rng() & 1U) ? -0.0F : 0.0F;
        else
            a = std::ldexp(float(int(rng() % 2000001U) - 1000000), int(rng() % 24U) - 32);
    }
    return x;
}
const char* name(WeightFormat f) {
    switch (f) {
    case WeightFormat::bf16: return "bf16";
    case WeightFormat::f32: return "f32";
    case WeightFormat::q8_0: return "q8_0";
    case WeightFormat::q4_k: return "q4_k";
    case WeightFormat::q6_k: return "q6_k";
    default: return "other";
    }
}
struct Matrix {
    std::vector<std::uint8_t> host;
    Buffer data;
    LinearView view;
    Matrix(WeightFormat f, unsigned outputs, unsigned inputs, unsigned seed)
        : host(fixture::packed(f, outputs, inputs, seed)), data(device(host)),
          view{data.as<std::uint8_t>(), outputs, inputs, packed_row_bytes(f, inputs), f} {}
};
std::size_t cases = 0;
// Reference: the eight-row passes; candidate: every tile variant and the default.
void compare(const Matrix& w, const Matrix* up, unsigned rows, unsigned seed) {
    const auto inputs_count = w.view.inputs, outputs = w.view.outputs;
    const auto x = inputs(std::size_t(rows) * inputs_count, seed);
    auto dx = device(x);
    const auto count = std::size_t(rows) * outputs;
    Buffer reference(count * sizeof(float)), candidate(count * sizeof(float)), failure(sizeof(unsigned));
    const std::vector<float> poison(count, bit_cast<float>(0x7fc0dead));
    reference.upload(poison.data(), reference.size());
    failure.zero();
    // Explicit slices of at most eight rows through the g4-v1 passes.
    for (unsigned r = 0; r < rows; r += 8U) {
        const auto n = std::min(8U, rows - r);
        if (up)
            gate_up(w.view, up->view, dx.as<float>() + std::size_t(r) * inputs_count, n,
                    reference.as<float>() + std::size_t(r) * outputs, true, nullptr, failure.as<unsigned>());
        else
            linear(w.view, dx.as<float>() + std::size_t(r) * inputs_count, n, reference.as<float>() + std::size_t(r) * outputs,
                   true, nullptr, failure.as<unsigned>());
    }
    synchronize();
    const auto expected = read(reference, count);
    // Without the many-row tiles: one staged pass up to sixteen rows where admitted, else the slices.
    {
        candidate.upload(poison.data(), candidate.size());
        if (up)
            gate_up_rows(w.view, up->view, dx.as<float>(), rows, candidate.as<float>(), nullptr, failure.as<unsigned>(), false);
        else
            linear_rows(w.view, dx.as<float>(), rows, candidate.as<float>(), nullptr, failure.as<unsigned>(), false);
        synchronize();
        const auto got = read(candidate, count);
        if (std::memcmp(got.data(), expected.data(), count * sizeof(float))) {
            std::cerr << "mismatch " << name(w.view.format) << (up ? " gate_up" : " linear") << " outputs " << outputs
                      << " inputs " << inputs_count << " rows " << rows << " staged or sliced\n";
            bytes(got, expected);
        }
        ++cases;
    }
    unsigned flag = 1;
    failure.download(&flag, sizeof(flag));
    CHECK(flag == 0);
    Buffer scratch(up ? count * sizeof(float) : 0);
    for (int variant = -1; variant < rows_linear_variants + (up ? 2 : 0); ++variant) {
        candidate.upload(poison.data(), candidate.size());
        failure.zero();
        // Gate/up: every variant fused, then the split schedule with the default and one other tile.
        const bool split = variant >= rows_linear_variants;
        if (up)
            gate_up_rows(w.view, up->view, dx.as<float>(), rows, candidate.as<float>(), nullptr, failure.as<unsigned>(),
                         true, split ? (variant - rows_linear_variants) * 3 - 1 : variant, scratch.as<float>(),
                         split ? 1 : 0);
        else
            linear_rows(w.view, dx.as<float>(), rows, candidate.as<float>(), nullptr, failure.as<unsigned>(), true,
                        variant);
        synchronize();
        const auto got = read(candidate, count);
        if (std::memcmp(got.data(), expected.data(), count * sizeof(float))) {
            std::cerr << "mismatch " << name(w.view.format) << (up ? " gate_up" : " linear") << " outputs " << outputs
                      << " inputs " << inputs_count << " rows " << rows << " variant " << variant << '\n';
            bytes(got, expected);
        }
        failure.download(&flag, sizeof(flag));
        CHECK(flag == 0);
        ++cases;
    }
}
void host_identity(const Matrix& w, unsigned rows) {
    const auto x = inputs(std::size_t(rows) * w.view.inputs, 77);
    auto dx = device(x);
    Buffer out(std::size_t(rows) * w.view.outputs * sizeof(float));
    linear_rows(w.view, dx.as<float>(), rows, out.as<float>());
    synchronize();
    const LinearView host{w.host.data(), w.view.outputs, w.view.inputs, w.view.row_bytes, w.view.format};
    std::vector<float> expected(std::size_t(rows) * w.view.outputs);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned o = 0; o < w.view.outputs; ++o)
            expected[std::size_t(r) * w.view.outputs + o] = chunk16_dot(
                [&](unsigned i) { return weight_at(host, o, i); }, x.data() + std::size_t(r) * w.view.inputs, w.view.inputs);
    bytes(read(out, expected.size()), expected);
    ++cases;
}
void tests(bool full) {
    const std::vector<unsigned> many = {1, 2, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 31, 32, 33, 47, 64, 65, 100, 129, 255, 257, 511, 512};
    const std::vector<unsigned> few = {9, 17, 33, 100, 512};
    for (const auto f : {WeightFormat::bf16, WeightFormat::f32, WeightFormat::q8_0, WeightFormat::q4_k, WeightFormat::q6_k}) {
        const bool dense = f == WeightFormat::bf16 || f == WeightFormat::f32;
        struct Shape {
            unsigned outputs, inputs;
        };
        // Partial last K blocks (256, 768, 272), production widths and fallbacks (17).
        std::vector<Shape> shapes = dense ? std::vector<Shape>{{33, 16}, {5, 17}, {70, 272}, {257, 2560}, {40, 10240}}
                                          : std::vector<Shape>{{33, 256}, {70, 768}, {257, 2560}, {65, 2048}, {40, 10240}};
        unsigned seed = 1;
        for (const auto s : shapes) {
            const Matrix w(f, s.outputs, s.inputs, 100 + seed), up(f, s.outputs, s.inputs, 200 + seed);
            for (const auto rows : (full || s.inputs <= 2560 ? many : few)) {
                compare(w, nullptr, rows, seed * 1000 + rows);
                compare(w, &up, rows, seed * 1000 + rows + 1);
            }
            ++seed;
        }
        if (!dense || f == WeightFormat::bf16) {
            const Matrix w(f, 19, dense ? 272U : 512U, 9);
            host_identity(w, 37);
        }
        std::cout << "{\"format\":\"" << name(f) << "\",\"cases\":" << cases << "}\n";
    }
    // Mixed gate/up formats keep the eight-row passes.
    const Matrix g(WeightFormat::q4_k, 40, 2560, 5), u(WeightFormat::q6_k, 40, 2560, 6);
    compare(g, &u, 100, 4242);
}
// Production geometry, synthetic bytes. Diagnostic only, not a ctest.
// probe != nullptr: run only that shape, at probe_rows rows, variant probe_variant
// (gate shapes: split schedule), ten times, for hardware counter collection.
const char* probe = nullptr;
unsigned probe_rows = 512;
int probe_variant = -1;
void bench() {
    struct Shape {
        const char* name;
        WeightFormat format;
        unsigned outputs, inputs;
        bool gate;
    };
    const Shape shapes[] = {{"q4_gate_up", WeightFormat::q4_k, 10240, 2560, true},
                            {"q4_down", WeightFormat::q4_k, 2560, 10240, false},
                            {"q6_down", WeightFormat::q6_k, 2560, 10240, false},
                            {"q4_q", WeightFormat::q4_k, 2048, 2560, false},
                            {"q4_o", WeightFormat::q4_k, 2560, 2048, false},
                            {"q4_k", WeightFormat::q4_k, 512, 2560, false},
                            {"q6_v", WeightFormat::q6_k, 512, 2560, false},
                            {"q4_ple_projection", WeightFormat::q4_k, 2560, 256, false},
                            {"bf16_ple_model", WeightFormat::bf16, 10752, 2560, false}};
    for (const auto& s : shapes) {
        if (probe && std::string_view(probe) != s.name)
            continue;
        const Matrix w(s.format, s.outputs, s.inputs, 3), up(s.format, s.outputs, s.inputs, 4);
        if (probe) {
            const auto x = inputs(std::size_t(probe_rows) * s.inputs, 5);
            auto dx = device(x);
            Buffer out(std::size_t(probe_rows) * s.outputs * sizeof(float));
            Buffer scratch(s.gate ? std::size_t(probe_rows) * s.outputs * sizeof(float) : 0);
            for (unsigned r = 0; r < 10U; ++r) {
                if (s.gate)
                    gate_up_rows(w.view, up.view, dx.as<float>(), probe_rows, out.as<float>(), nullptr, nullptr, true,
                                 probe_variant, scratch.as<float>(), 1);
                else
                    linear_rows(w.view, dx.as<float>(), probe_rows, out.as<float>(), nullptr, nullptr, true, probe_variant);
            }
            synchronize();
            std::cout << "{\"probe\":\"" << s.name << "\",\"rows\":" << probe_rows << ",\"variant\":" << probe_variant << "}\n";
            continue;
        }
        for (unsigned rows : {64U, 512U}) {
            const auto x = inputs(std::size_t(rows) * s.inputs, 5);
            auto dx = device(x);
            Buffer out(std::size_t(rows) * s.outputs * sizeof(float));
            Buffer scratch(s.gate ? std::size_t(rows) * s.outputs * sizeof(float) : 0);
            // Two passes, forward then reverse variant order, so clock ramp and
            // cache state do not favour one end of the list; each prints its own line.
            const int last = rows_linear_variants * (s.gate ? 2 : 1);
            for (int pass = 0; pass < 2; ++pass)
            for (int step = -2; step < last; ++step) {
                const int variant = pass == 0 ? step : last - 3 - step;
                // Gate/up variants >= rows_linear_variants run the split schedule.
                const bool split = variant >= rows_linear_variants;
                const int tile = split ? variant - rows_linear_variants : variant;
                const auto run = [&] {
                    if (s.gate)
                        gate_up_rows(w.view, up.view, dx.as<float>(), rows, out.as<float>(), nullptr, nullptr,
                                     variant >= -1, tile, scratch.as<float>(), split ? 1 : 0);
                    else
                        linear_rows(w.view, dx.as<float>(), rows, out.as<float>(), nullptr, nullptr, variant >= -1,
                                    variant);
                };
                run();
                synchronize();
                const auto start = std::chrono::steady_clock::now();
                const unsigned repeats = variant == -2 ? 2U : 20U;
                for (unsigned r = 0; r < repeats; ++r)
                    run();
                synchronize();
                const double ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / repeats;
                const double macs = double(rows) * s.outputs * s.inputs * (s.gate ? 2.0 : 1.0);
                std::cout << std::setprecision(6) << "{\"bench\":\"" << s.name << "\",\"rows\":" << rows
                          << ",\"pass\":" << pass << ",\"variant\":\"" << (variant == -2 ? "rows8" : (split ? "split" : "") + std::to_string(tile)) << "\",\"ms\":" << ms
                          << ",\"gmac_s\":" << macs / (ms * 1e6) << "}\n";
            }
        }
    }
}
} // namespace
int main(int argc, char** argv) {
    try {
        require_device();
        if (argc == 2 && std::string_view(argv[1]) == "bench") {
            bench();
            return 0;
        }
        if (argc == 5 && std::string_view(argv[1]) == "probe") {
            probe = argv[2];
            probe_rows = unsigned(std::atoi(argv[3]));
            probe_variant = std::atoi(argv[4]);
            bench();
            return 0;
        }
        tests(argc == 2 && std::string_view(argv[1]) == "--full");
        std::cout << "{\"gate\":\"rows-linear-vs-rows8\",\"cases\":" << cases << ",\"verdict\":\"EXACT\"}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
