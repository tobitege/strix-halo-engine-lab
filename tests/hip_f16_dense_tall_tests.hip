// Device self-test of the shared dense FP16 core (core/f16_dense_tall.hpp): every column width and the idle-wave skip
// give the same bytes for the same shape, padding columns and the guard row keep their sentinel, and the remainder
// rule of plan() is timed against the other widths. Needs a GPU. Exit 0 = PASS.
#include "core/f16_dense_tall.hpp"
#include "dense_fixture_checks.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tall = strix::core::f16_dense_tall;

namespace {

constexpr float kGuard = -1234.5F;
constexpr unsigned kPad = 16;  // sentinel columns after the outputs of every row

struct Args {
    const float* x;
    const _Float16* w;
    float* y;
    unsigned in, rows, stride, tiles, tile0, skip;
};
struct Cols {
    const Args& a;
    __device__ __forceinline__ tall::ColumnTile tile(unsigned i) const {
        const unsigned t = a.tile0 + i;
        const bool live = t < a.tiles;
        const unsigned u = live ? t : a.tiles - 1U;
        return {a.w + static_cast<std::size_t>(u) * 16U * a.in, a.y + u * 16U, a.stride, live};
    }
};
template <unsigned TC> __global__ __launch_bounds__(256) void kernel(Args a) {
    __shared__ tall::Storage<TC> lds;
    tall::core<TC>(a.x, a.in, a.rows, Cols{a}, lds, tall::RawStore{}, a.skip != 0U);
}

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

struct Fixture {
    unsigned rows, in, n;
    float *dx = nullptr, *dy = nullptr;
    _Float16* dw = nullptr;
    std::vector<float> fill, reference;
    Fixture(unsigned r, unsigned i, unsigned nn) : rows(r), in(i), n(nn) {
        std::vector<float> x(std::size_t{rows} * in);
        std::vector<_Float16> w(std::size_t{n} * in);
        std::uint64_t s = 0x9E3779B97F4A7C15ULL + rows * 131U + n;
        const auto next = [&] {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            return static_cast<float>(static_cast<int>(s >> 60U) - 8) / 8.0F;
        };
        for (auto& v : x) v = next() * 4.0F;
        for (auto& v : w) v = static_cast<_Float16>(next() * 0.25F);
        // |product| <= 1 and each product is a multiple of 1/64. At in=1024,
        // every possible partial sum is exact in FP32; no WMMA oracle is reused.
        const std::vector<float> wf(w.begin(), w.end());
        reference = dense_fixture::exact_reference(x, wf, rows, in, n, kPad, kGuard);
        fill.assign(std::size_t{rows + 1U} * (n + kPad), kGuard);
        ok(hipMalloc(&dx, x.size() * 4U), "x");
        ok(hipMalloc(&dw, w.size() * 2U), "w");
        ok(hipMalloc(&dy, fill.size() * 4U), "y");
        ok(hipMemcpy(dx, x.data(), x.size() * 4U, hipMemcpyHostToDevice), "x upload");
        ok(hipMemcpy(dw, w.data(), w.size() * 2U, hipMemcpyHostToDevice), "w upload");
    }
    ~Fixture() {
        (void)hipFree(dx);
        (void)hipFree(dw);
        (void)hipFree(dy);
    }
    // tail: column width of the remainder launch (0: plan()), skip: idle-wave skip.
    void launch(unsigned tail, bool skip) const {
        Args a{dx, dw, dy, in, rows, n + kPad, n / 16U, 0U, skip ? 1U : 0U};
        const unsigned gx = (rows + tall::kRows - 1U) / tall::kRows, full = a.tiles / 16U, rest = a.tiles % 16U;
        if (full) hipLaunchKernelGGL(kernel<16>, dim3(gx, full), dim3(256), 0, nullptr, a);
        if (!rest) return;
        a.tile0 = full * 16U;
        const unsigned tc = tail ? tail : tall::plan(a.tiles, rows, true).tc;
        const dim3 grid(gx, (rest + tc - 1U) / tc);
        switch (tc) {
        case 16: hipLaunchKernelGGL(kernel<16>, grid, dim3(256), 0, nullptr, a); break;
        case 4: hipLaunchKernelGGL(kernel<4>, grid, dim3(256), 0, nullptr, a); break;
        case 2: hipLaunchKernelGGL(kernel<2>, grid, dim3(256), 0, nullptr, a); break;
        default: hipLaunchKernelGGL(kernel<1>, grid, dim3(256), 0, nullptr, a); break;
        }
    }
    struct Result {
        std::uint64_t fnv;
        bool sentinels;
        double ms;
    };
    Result run(unsigned tail, bool skip, unsigned reps) const {
        ok(hipMemcpy(dy, fill.data(), fill.size() * 4U, hipMemcpyHostToDevice), "y fill");
        launch(tail, skip);
        ok(hipDeviceSynchronize(), "sync");
        std::vector<float> y(fill.size());
        ok(hipMemcpy(y.data(), dy, y.size() * 4U, hipMemcpyDeviceToHost), "y download");
        dense_fixture::check_dense(y, reference, rows, n, kPad, kGuard);
        Result r{1469598103934665603ULL, true, 0.0};
        for (const float f : y) {
            std::uint32_t b;
            std::memcpy(&b, &f, 4);
            for (unsigned k = 0; k < 4U; ++k) r.fnv = (r.fnv ^ ((b >> (8U * k)) & 0xFFU)) * 1099511628211ULL;
        }
        for (unsigned q = 0; q <= rows; ++q)
            for (unsigned c = q < rows ? n : 0U; c < n + kPad; ++c) r.sentinels = r.sentinels && y[std::size_t{q} * (n + kPad) + c] == kGuard;
        // Correctness above is mandatory for every arm; timing is not a verdict.
        double best = 1e30;
        for (unsigned b = 0; b < 5U; ++b) {
            const auto t0 = std::chrono::steady_clock::now();
            for (unsigned i = 0; i < reps; ++i) launch(tail, skip);
            ok(hipDeviceSynchronize(), "sync");
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps);
        }
        r.ms = best;
        return r;
    }
};

}  // namespace

int main() {
    unsigned failures = 0;
    try {
        // Shapes: last CTA with one live row / full CTAs / one row; outputs with a 1-tile remainder, alone, and none.
        for (const unsigned rows : {129U, 128U, 1024U, 1U, 113U}) {
            for (const unsigned n : {272U, 16U, 1024U, 48U, 208U}) {
                const Fixture f(rows, 1024U, n);
                const unsigned reps = rows * n > 200000U ? 30U : 100U;
                const auto legacy = f.run(4U, false, reps), now = f.run(0U, true, reps);
                bool same = legacy.fnv == now.fnv && legacy.sentinels && now.sentinels;
                std::string widths;
                if (n % 256U) {
                    for (const unsigned tc : {16U, 4U, 2U, 1U}) {
                        for (const bool skip : {false, true}) {
                            const auto r = f.run(tc, skip, reps);
                            same = same && r.fnv == now.fnv && r.sentinels;
                            if (skip) widths += " tc" + std::to_string(tc) + " " + std::to_string(r.ms).substr(0, 6);
                        }
                    }
                }
                std::printf("SHAPE rows %4u in 1024 n %4u  fnv %016llx sentinels %d  tail-4/no-skip %.4f ms  plan(tc %u)+skip %.4f ms  ratio %.3f %s |%s\n",
                            rows, n, static_cast<unsigned long long>(now.fnv), now.sentinels ? 1 : 0, legacy.ms, tall::plan(n / 16U, rows, true).tc, now.ms,
                            now.ms / legacy.ms, same ? "SAME" : "DIFFERENT", widths.c_str());
                failures += same ? 0U : 1U;
            }
        }
        // Remainder rule: every width for every remainder, with full CTAs (1,024 rows) so only the columns differ.
        for (unsigned rest = 1; rest < 16U; ++rest) {
            const Fixture f(1024U, 1024U, rest * 16U);
            std::string line;
            unsigned best = 0;
            double best_ms = 1e30;
            for (const unsigned tc : {0U, 16U, 4U, 2U, 1U}) {
                for (const bool skip : {false, true}) {
                    const auto r = f.run(tc, skip, 100U);
                    // run() checks the independent reference, live writes and all guards,
                    // including every remainder; a failure throws to main's nonzero exit.
                    if (!skip) continue;
                    line += " tc" + std::to_string(tc) + " " + std::to_string(r.ms).substr(0, 6);
                    if (r.ms < best_ms) best_ms = r.ms, best = tc;
                }
            }
            std::printf("TAIL rest %2u  plan tc %u  fastest tc %u |%s\n", rest, tall::plan(rest, 1024U, true).tc, best, line.c_str());
        }
    } catch (const std::exception& e) {
        std::printf("ERROR %s\n", e.what());
        return 2;
    }
    std::printf("f16 dense tall: %u failures\n", failures);
    return failures ? 1 : 0;
}
