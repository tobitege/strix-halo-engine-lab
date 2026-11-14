// Byte gate of the EXL3 panel prefill route (g4-exl3-wmma-pf-v1 on the prefill
// GEMMs) against the pack's own 64x64 WMMA kernel, and of its fused FFN against
// separate linears + gpu::gelu_product. "bench" times both routes and the exact
// many-row core at 512 rows.
#include <hip/hip_runtime.h>
#include "gemma4/exl3/device.hpp"
#include "gemma4/prefill/device.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <iostream>
#include <random>
#include <string_view>

namespace g4 = strix::gemma4;
namespace ex = g4::exl3;
namespace pf = g4::prefill;
namespace gpu = g4::gpu;
namespace {
// Synthetic pack: random trellis bits and finite FP16 scales in [0.5, 2).
std::vector<std::uint8_t> pack(unsigned inputs, unsigned outputs, unsigned k, unsigned seed) {
    std::mt19937 rng(seed);
    const std::size_t bytes = 2ULL * (inputs + outputs) + std::size_t(inputs / 16) * (outputs / 16) * 32 * k;
    std::vector<std::uint8_t> data(bytes);
    for (unsigned i = 0; i < inputs + outputs; ++i) {
        const auto h = std::uint16_t(0x3800U + rng() % 0x0800U);
        data[2ULL * i] = std::uint8_t(h); data[2ULL * i + 1] = std::uint8_t(h >> 8U);
    }
    for (std::size_t i = 2ULL * (inputs + outputs); i < bytes; ++i) data[i] = std::uint8_t(rng());
    return data;
}
std::vector<float> inputs(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<float> x(n);
    for (auto& a : x) a = (rng() % 16U == 0) ? ((rng() & 1U) ? -0.0F : 0.0F)
                                             : std::ldexp(float(int(rng() % 2000001U) - 1000000), int(rng() % 8U) - 22);
    return x;
}
struct Pack {
    std::vector<std::uint8_t> host;
    gpu::Buffer data;
    g4::LinearView view;
    std::unique_ptr<pf::PreparedLinear> prepared;
    Pack(unsigned in, unsigned out, unsigned k, unsigned seed) : host(pack(in, out, k, seed)), data(host.size()) {
        data.upload(host.data(), host.size());
        view = {data.as<std::uint8_t>(), out, in, std::size_t(in) * k / 8, g4::WeightFormat::exl3_mul1};
        prepared = std::make_unique<pf::PreparedLinear>(view);
    }
};
std::vector<float> download(const gpu::Buffer& b, std::size_t count) {
    std::vector<float> v(count); b.download(v.data(), count * 4); return v;
}
void same(const std::vector<float>& a, const std::vector<float>& b, const std::string& what) {
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], 4)) {
            std::cerr << what << " at " << i << ": " << std::hexfloat << a[i] << " vs " << b[i] << '\n';
            throw std::runtime_error("EXL3 panel route byte mismatch");
        }
}
std::size_t cases = 0, ffn_cases = 0, packed_cases = 0;
// Panel route == the pack's WMMA kernel (on the same panels): outputs, failure bits, input maximum.
void compare(const Pack& w, unsigned rows, unsigned seed) {
    const auto x = inputs(std::size_t(rows) * w.view.inputs, seed);
    gpu::Buffer dx(x.size() * 4); dx.upload(x.data(), x.size() * 4);
    const auto count = std::size_t(rows) * w.view.outputs;
    gpu::Buffer reference(count * 4), candidate(count * 4), failure(8), maximum(8);
    gpu::Buffer scratch(std::size_t(rows) * w.view.inputs * 2);
    const std::vector<float> poison(count, g4::bit_cast<float>(0x7fc0deadU));
    reference.upload(poison.data(), count * 4); candidate.upload(poison.data(), count * 4);
    failure.zero(); maximum.zero();
    ex::device_prefill(w.view, dx.as<float>(), rows, reference.as<float>(), scratch.data(), failure.as<unsigned>(),
                       maximum.as<unsigned>(), w.prepared->panels().data());
    pf::linear(w.view, dx.as<float>(), rows, candidate.as<float>(), scratch.data(), failure.as<unsigned>() + 1,
               maximum.as<unsigned>() + 1, nullptr, true, w.prepared.get());
    gpu::synchronize();
    unsigned flags[2] = {}, maxima[2] = {};
    failure.download(flags, 8); maximum.download(maxima, 8);
    if (flags[0] != flags[1] || maxima[0] != maxima[1]) throw std::runtime_error("panel route failure bits or maximum differ");
    same(download(reference, count), download(candidate, count),
         "linear I " + std::to_string(w.view.inputs) + " O " + std::to_string(w.view.outputs) + " K " +
         std::to_string(ex::bits(w.view)) + " rows " + std::to_string(rows));
    // The same kernel reading its B fragments from the packed trellis (no resident panels): the same bytes.
    if (rows <= 129) {
        gpu::Buffer direct(count * 4), direct_failure(4), direct_maximum(4);
        direct.upload(poison.data(), count * 4);
        direct_failure.zero(); direct_maximum.zero();
        ex::device_prefill(w.view, dx.as<float>(), rows, direct.as<float>(), scratch.data(), direct_failure.as<unsigned>(),
                           direct_maximum.as<unsigned>(), nullptr);
        gpu::synchronize();
        unsigned direct_flag = 0, direct_max = 0;
        direct_failure.download(&direct_flag, 4); direct_maximum.download(&direct_max, 4);
        if (direct_flag != flags[0] || direct_max != maxima[0]) throw std::runtime_error("packed WMMA failure bits or maximum differ");
        same(download(reference, count), download(direct, count),
             "packed I " + std::to_string(w.view.inputs) + " O " + std::to_string(w.view.outputs) + " K " +
             std::to_string(ex::bits(w.view)) + " rows " + std::to_string(rows));
        ++packed_cases;
    }
    ++cases;
}
// Fused panel FFN == linear(gate), linear(up), gelu_product, linear(down) on the panel route.
void compare_ffn(const Pack& gate, const Pack& up, const Pack& down, unsigned rows, unsigned seed) {
    const unsigned hidden = gate.view.inputs, inter = gate.view.outputs;
    if (!pf::exl3_gelu_pair_supported(*gate.prepared, *up.prepared, down.view)) throw std::runtime_error("panel FFN declined");
    const auto x = inputs(std::size_t(rows) * hidden, seed);
    gpu::Buffer dx(x.size() * 4); dx.upload(x.data(), x.size() * 4);
    const auto count = std::size_t(rows) * hidden;
    gpu::Buffer g(std::size_t(rows) * inter * 4), u(std::size_t(rows) * inter * 4), reference(count * 4), candidate(count * 4);
    gpu::Buffer failure(8), maximum(24), scratch(std::size_t(rows) * std::max(hidden, inter) * 2),
        product(std::size_t(rows) * inter * 2);
    const std::vector<float> poison(count, g4::bit_cast<float>(0x7fc0deadU));
    reference.upload(poison.data(), count * 4); candidate.upload(poison.data(), count * 4);
    failure.zero(); maximum.zero();
    auto* f = failure.as<unsigned>();
    auto* m = maximum.as<unsigned>();
    pf::linear(gate.view, dx.as<float>(), rows, g.as<float>(), scratch.data(), f, m, nullptr, true, gate.prepared.get());
    pf::linear(up.view, dx.as<float>(), rows, u.as<float>(), scratch.data(), f, m + 1, nullptr, true, up.prepared.get());
    gpu::gelu_product(g.as<float>(), u.as<float>(), rows * inter);
    pf::linear(down.view, g.as<float>(), rows, reference.as<float>(), scratch.data(), f, m + 2, nullptr, true, down.prepared.get());
    pf::exl3_gelu_pair(*gate.prepared, *up.prepared, down.view, dx.as<float>(), rows, scratch.data(), g.as<float>(), u.as<float>(),
                       product.data(), f + 1, m + 3);
    pf::linear(down.view, nullptr, rows, candidate.as<float>(), product.data(), f + 1, m + 5, nullptr, true, down.prepared.get(),
               nullptr, true);
    gpu::synchronize();
    unsigned flags[2] = {}, maxima[6] = {};
    failure.download(flags, 8); maximum.download(maxima, 24);
    if (flags[0] != flags[1]) throw std::runtime_error("panel FFN failure bits differ");
    for (unsigned i = 0; i < 3; ++i) if (maxima[i] != maxima[3 + i]) throw std::runtime_error("panel FFN maxima differ");
    same(download(reference, count), download(candidate, count),
         "ffn hidden " + std::to_string(hidden) + " inter " + std::to_string(inter) + " rows " + std::to_string(rows));
    ++ffn_cases;
}
void tests() {
    struct Shape { unsigned in, out; };
    // Family shapes (q, k/v, o, gate/up, down, global q/o) and small edges.
    const Shape shapes[] = {{256, 128}, {256, 2560}, {2560, 256}, {2560, 512}, {2560, 1024}, {2048, 2560}, {2560, 2048},
                            {4096, 2560}, {2560, 4096}, {2560, 10240}, {10240, 2560}};
    const unsigned many[] = {1, 8, 9, 17, 64, 100, 129, 256, 257, 512};
    unsigned seed = 1;
    for (const unsigned k : {4U, 5U, 6U, 8U}) {
        for (const auto s : shapes) {
            const Pack w(s.in, s.out, k, seed++);
            for (const auto rows : many) compare(w, rows, seed * 131 + rows);
        }
        std::cout << "{\"k\":" << k << ",\"cases\":" << cases << "}\n";
    }
    struct Ffn { unsigned hidden, inter, k; };
    const Ffn ffns[] = {{256, 512, 4}, {512, 256, 5}, {2560, 1024, 4}, {2560, 10240, 4}};
    for (const auto& f : ffns) {
        const Pack gate(f.hidden, f.inter, f.k, seed++), up(f.hidden, f.inter, f.k, seed++), down(f.inter, f.hidden, f.k, seed++);
        for (const auto rows : {9U, 17U, 100U, 257U, 512U}) compare_ffn(gate, up, down, rows, seed * 137 + rows);
    }
}
template <class F> double timed(F run) {
    run(); gpu::synchronize();
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < 5; ++i) run();
    gpu::synchronize();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 5;
}
void bench() {
    struct Shape { const char* name; unsigned in, out, k; };
    const Shape shapes[] = {{"gate", 2560, 10240, 4}, {"down", 10240, 2560, 4}, {"q", 2560, 2048, 5}, {"kv", 2560, 512, 6},
                            {"o", 2048, 2560, 5}, {"q_global", 2560, 4096, 5}, {"o_global", 4096, 2560, 5}};
    const unsigned rows = 512;
    for (const auto& s : shapes) {
        const Pack w(s.in, s.out, s.k, 7);
        const auto x = inputs(std::size_t(rows) * s.in, 3);
        gpu::Buffer dx(x.size() * 4), out(std::size_t(rows) * s.out * 4), failure(4), maximum(4);
        gpu::Buffer scratch(std::max({ex::device_scratch_bytes, ex::rows_scratch_bytes(w.view, rows), std::size_t(rows) * s.in * 2}));
        dx.upload(x.data(), x.size() * 4);
        auto* f = failure.as<unsigned>();
        const double exact = timed([&] { (void)ex::device_linear_rows(w.view, dx.as<float>(), rows, out.as<float>(), scratch.data(), scratch.size(), f); });
        const double kernel = timed([&] { ex::device_prefill(w.view, dx.as<float>(), rows, out.as<float>(), scratch.data(), f, maximum.as<unsigned>(), w.prepared->panels().data()); });
        const double panel = timed([&] { pf::linear(w.view, dx.as<float>(), rows, out.as<float>(), scratch.data(), f, maximum.as<unsigned>(), nullptr, true, w.prepared.get()); });
        const double packed = timed([&] { ex::device_prefill(w.view, dx.as<float>(), rows, out.as<float>(), scratch.data(), f, maximum.as<unsigned>(), nullptr); });
        const double macs = double(rows) * s.in * s.out;
        std::cout << std::setprecision(5) << "{\"bench\":\"" << s.name << "\",\"rows\":" << rows << ",\"exact_ms\":" << exact
                  << ",\"wmma64_ms\":" << kernel << ",\"wmma64_packed_ms\":" << packed << ",\"panel_ms\":" << panel << ",\"panel_tflops\":" << 2 * macs / (panel * 1e9)
                  << ",\"exact_tflops\":" << 2 * macs / (exact * 1e9) << "}\n";
    }
    const unsigned hidden = 2560, inter = 10240;
    const Pack gate(hidden, inter, 4, 7), up(hidden, inter, 4, 8), down(inter, hidden, 4, 9);
    const auto x = inputs(std::size_t(rows) * hidden, 3);
    gpu::Buffer dx(x.size() * 4), g(std::size_t(rows) * inter * 4), u(std::size_t(rows) * inter * 4), out(std::size_t(rows) * hidden * 4),
        failure(4), maximum(12), product(std::size_t(rows) * inter * 2);
    gpu::Buffer scratch(std::max({ex::ffn_rows_scratch_bytes(gate.view, up.view, down.view, rows), std::size_t(rows) * inter * 2}));
    dx.upload(x.data(), x.size() * 4);
    auto* f = failure.as<unsigned>();
    const double exact = timed([&] { (void)ex::device_ffn_rows(gate.view, up.view, down.view, dx.as<float>(), rows, g.as<float>(), u.as<float>(), out.as<float>(), scratch.data(), scratch.size(), f); });
    const double panel = timed([&] {
        pf::exl3_gelu_pair(*gate.prepared, *up.prepared, down.view, dx.as<float>(), rows, scratch.data(), g.as<float>(), u.as<float>(),
                           product.data(), f, maximum.as<unsigned>());
        pf::linear(down.view, nullptr, rows, out.as<float>(), product.data(), f, maximum.as<unsigned>() + 2, nullptr, true,
                   down.prepared.get(), nullptr, true);
    });
    std::cout << std::setprecision(5) << "{\"bench\":\"ffn\",\"rows\":" << rows << ",\"exact_ms\":" << exact << ",\"panel_ms\":" << panel
              << ",\"panel_tflops\":" << 6.0 * rows * hidden * inter / (panel * 1e9) << "}\n";
}
} // namespace
int main(int argc, char** argv) {
    try {
        gpu::require_device();
        if (argc == 2 && std::string_view(argv[1]) == "bench") { bench(); return 0; }
        tests();
        std::cout << "{\"gate\":\"exl3-panels-vs-wmma64\",\"identity\":\"" << ex::prefill_identity << "\",\"cases\":" << cases
                  << ",\"ffn_cases\":" << ffn_cases << ",\"packed_cases\":" << packed_cases << ",\"verdict\":\"EXACT\"}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
