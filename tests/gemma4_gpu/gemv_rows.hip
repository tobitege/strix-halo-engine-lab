// Small-row decode GEMV fixture: output hash, time per call and bytes per call
// of the shared passes at 1..16 rows. Diagnostic only, not a ctest.
//   gemv_rows [--format q4_k|q6_k|q8_0|bf16|f32] [--inputs K] [--outputs N] [--gate 0|1]
//             [--rows 1,2,4,8,12,16] [--iters 300] [--route slices|many]
// route slices: linear_rows / gate_up_rows without the many-row tiles (one
// staged pass up to sixteen rows where admitted, otherwise eight-row slices);
// route many: the many-row tiles above eight rows.
#include <hip/hip_runtime.h>
#include "gemma4/kernels/rows_linear.hpp"
#include "gemma4/kernels/internal.hpp"
#include "fixture.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

using namespace strix::gemma4;
using namespace strix::gemma4::gpu;
namespace {
WeightFormat format_of(std::string_view s) {
    if (s == "q4_k") return WeightFormat::q4_k;
    if (s == "q6_k") return WeightFormat::q6_k;
    if (s == "q8_0") return WeightFormat::q8_0;
    if (s == "bf16") return WeightFormat::bf16;
    if (s == "f32") return WeightFormat::f32;
    throw std::runtime_error("format");
}
std::vector<unsigned> list(std::string_view s) {
    std::vector<unsigned> v;
    std::size_t at = 0;
    while (at < s.size()) {
        const auto end = std::min(s.find(',', at), s.size());
        v.push_back(unsigned(std::atoi(std::string(s.substr(at, end - at)).c_str())));
        at = end + 1;
    }
    return v;
}
std::uint64_t fnv(const std::vector<float>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    const auto* p = reinterpret_cast<const unsigned char*>(v.data());
    for (std::size_t i = 0; i < v.size() * sizeof(float); ++i)
        h = (h ^ p[i]) * 1099511628211ULL;
    return h;
}
} // namespace

int main(int argc, char** argv) try {
    std::string format = "q4_k", rows_arg = "1,2,4,8,12,16", route = "slices";
    unsigned inputs = 2560, outputs = 10240, iters = 300;
    bool gate = true;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string_view key = argv[i], value = argv[i + 1];
        if (key == "--format") format = value;
        else if (key == "--inputs") inputs = unsigned(std::atoi(argv[i + 1]));
        else if (key == "--outputs") outputs = unsigned(std::atoi(argv[i + 1]));
        else if (key == "--gate") gate = value == "1";
        else if (key == "--rows") rows_arg = value;
        else if (key == "--iters") iters = unsigned(std::atoi(argv[i + 1]));
        else if (key == "--route") route = value;
        else throw std::runtime_error("unknown option");
    }
    const auto f = format_of(format);
    const bool many = route == "many";
    const auto row_bytes = packed_row_bytes(f, inputs);
    const auto host_w = fixture::packed(f, outputs, inputs, 101), host_u = fixture::packed(f, outputs, inputs, 202);
    Buffer dw(host_w.size()), du(host_u.size());
    dw.upload(host_w.data(), dw.size());
    du.upload(host_u.data(), du.size());
    const LinearView w{dw.as<std::uint8_t>(), outputs, inputs, row_bytes, f}, up{du.as<std::uint8_t>(), outputs, inputs, row_bytes, f};
    for (const auto rows : list(rows_arg)) {
        std::vector<float> x(std::size_t(rows) * inputs);
        std::mt19937 rng(7000U + rows);
        for (auto& a : x)
            a = rng() % 16U == 0 ? ((rng() & 1U) ? -0.0F : 0.0F)
                                 : std::ldexp(float(int(rng() % 2000001U) - 1000000), int(rng() % 24U) - 32);
        Buffer dx(x.size() * sizeof(float)), out(std::size_t(rows) * outputs * sizeof(float)), failure(sizeof(unsigned));
        dx.upload(x.data(), dx.size());
        failure.zero();
        const auto call = [&] {
            if (gate)
                gate_up_rows(w, up, dx.as<float>(), rows, out.as<float>(), nullptr, failure.as<unsigned>(), many);
            else
                linear_rows(w, dx.as<float>(), rows, out.as<float>(), nullptr, failure.as<unsigned>(), many);
        };
        for (unsigned i = 0; i < 30U; ++i)
            call();
        synchronize();
        // Three timed windows; the median is reported with the spread.
        double ms[3];
        for (auto& window : ms) {
            const auto begin = std::chrono::steady_clock::now();
            for (unsigned i = 0; i < iters; ++i)
                call();
            synchronize();
            window = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() / iters;
        }
        std::sort(ms, ms + 3);
        std::vector<float> got(std::size_t(rows) * outputs);
        out.download(got.data(), got.size() * sizeof(float));
        unsigned flag = 1;
        failure.download(&flag, sizeof(flag));
        const double weight_bytes = double(outputs) * row_bytes * (gate ? 2.0 : 1.0);
        std::cout << "{\"format\":\"" << format << "\",\"gate\":" << gate << ",\"inputs\":" << inputs << ",\"outputs\":" << outputs
                  << ",\"route\":\"" << route << "\",\"rows\":" << rows << ",\"hash\":\"" << std::hex << fnv(got) << std::dec
                  << "\",\"failure\":" << flag << std::fixed << std::setprecision(4) << ",\"ms\":" << ms[1]
                  << ",\"ms_min\":" << ms[0] << ",\"ms_max\":" << ms[2] << std::setprecision(0)
                  << ",\"weight_bytes_sweep\":" << weight_bytes << ",\"activation_bytes_copy\":" << double(rows) * inputs * 4.0
                  << "}" << std::endl;
        std::cout.unsetf(std::ios::fixed);
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << "gemv_rows: " << e.what() << '\n';
    return 1;
}
