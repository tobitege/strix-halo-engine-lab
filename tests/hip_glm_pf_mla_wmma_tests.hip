// FAST glm-pf-mla-wmma-v1 (STRIX_GLM_PF_MLA_WMMA=1): the MLA absorb / out per-head GEMMs on the shared short-K FP16
// WMMA tile GEMM (core/f16_tile_gemm.hpp) against (a) the contract in double, sum_k fp16(x) * w (accumulation error
// only: tight) and (b) the FP32 inputs in double (FP16 rounding of x: drift gate, the exact FP32 chains sit within 1e-6
// of it). Sampled rows of every head, row counts that end inside a 16-row fragment and a 128-row CTA, untouched guard
// rows, finite outputs; then the per-launch time at 4096 and 8192 rows x 32 heads (the pair prefill chunk shapes).
#include "glm/prefill/glm_pf_mla_wmma.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace M = strix::glm::pf::mla_wmma;
namespace {
constexpr unsigned kLora = 512, kDim = 256;
unsigned checks = 0, failures = 0;
void check(bool c, const std::string& what) {
    ++checks;
    if (!c) { ++failures; std::printf("FAIL %s\n", what.c_str()); }
}
void ok(hipError_t e, const char* w) { if (e != hipSuccess) throw std::runtime_error(std::string(w) + ": " + hipGetErrorString(e)); }
constexpr float kGuard = -777.25F;
float h2f(std::uint16_t h) { return static_cast<float>(*reinterpret_cast<const _Float16*>(&h)); }
std::uint16_t f2h(float f) { const _Float16 h = static_cast<_Float16>(f); std::uint16_t u; std::memcpy(&u, &h, 2); return u; }
float r16(float f) { return static_cast<float>(static_cast<_Float16>(f)); }

template <bool kOut> void run(unsigned heads, unsigned C, std::mt19937& g, bool timing) {
    constexpr unsigned K = kOut ? kLora : kDim, N = kOut ? kDim : kLora;
    const std::size_t xn = std::size_t{C} * heads * K, yn = std::size_t{C} * heads * N, guard = std::size_t{heads} * N;
    std::normal_distribution<float> nx(0.0F, kOut ? 0.6F : 2.0F), nw(0.0F, 0.04F);
    std::vector<float> x(xn);
    for (auto& v : x) v = nx(g);
    std::vector<std::uint16_t> kv(std::size_t{heads} * 2 * kDim * kLora);
    for (auto& v : kv) v = f2h(nw(g));
    float *dx, *dy;
    std::uint16_t* dk;
    ok(hipMalloc(&dx, xn * 4), "x"); ok(hipMalloc(&dy, (yn + guard) * 4), "y"); ok(hipMalloc(&dk, kv.size() * 2), "kv");
    ok(hipMemcpy(dx, x.data(), xn * 4, hipMemcpyHostToDevice), "x h2d");
    ok(hipMemcpy(dk, kv.data(), kv.size() * 2, hipMemcpyHostToDevice), "kv h2d");
    std::vector<float> y(yn + guard, kGuard);
    ok(hipMemcpy(dy, y.data(), y.size() * 4, hipMemcpyHostToDevice), "y h2d");
    M::gemm<kLora, kDim, kOut>(dx, dk, heads, C, dy, nullptr);
    ok(hipGetLastError(), "launch");
    ok(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost), "y d2h");
    if constexpr (!kOut) {  // bf16 q-tilde image (fused core 22 / 23): bitwise tile::to_bf16 of the FP32 sums, guard kept
        std::uint16_t* dz;
        const std::size_t zn = yn + guard;
        ok(hipMalloc(&dz, zn * 2), "z");
        std::vector<std::uint16_t> z(zn, 0xA5A5U);
        ok(hipMemcpy(dz, z.data(), zn * 2, hipMemcpyHostToDevice), "z h2d");
        M::gemm<kLora, kDim, kOut>(dx, dk, heads, C, reinterpret_cast<float*>(dz), nullptr, true);
        ok(hipGetLastError(), "image launch");
        ok(hipMemcpy(z.data(), dz, zn * 2, hipMemcpyDeviceToHost), "z d2h");
        ok(hipFree(dz), "z free");
        std::size_t bad = 0;
        for (std::size_t i = 0; i < yn; ++i) {
            std::uint32_t u;
            std::memcpy(&u, &y[i], 4);
            const std::uint16_t ref = (u & 0x7FFFFFFFU) > 0x7F800000U ? static_cast<std::uint16_t>((u >> 16) | 0x40U)
                                                                         : static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
            bad += z[i] != ref;
        }
        for (std::size_t i = yn; i < zn; ++i) bad += z[i] != 0xA5A5U;
        check(bad == 0, "absorb H" + std::to_string(heads) + " C" + std::to_string(C) + " bf16 image == to_bf16(FP32 sums), guard kept");
    }
    if constexpr (kOut) {  // FP16 o-bar image input (flash out_op): bitwise equal to the FP32-input run, guard kept
        std::vector<std::uint16_t> xh(xn);
        for (std::size_t i = 0; i < xn; ++i) xh[i] = f2h(x[i]);
        std::uint16_t* dxh;
        float* dy2;
        ok(hipMalloc(&dxh, xn * 2), "xh"); ok(hipMalloc(&dy2, (yn + guard) * 4), "y2");
        ok(hipMemcpy(dxh, xh.data(), xn * 2, hipMemcpyHostToDevice), "xh h2d");
        std::vector<float> y2(yn + guard, kGuard);
        ok(hipMemcpy(dy2, y2.data(), y2.size() * 4, hipMemcpyHostToDevice), "y2 h2d");
        M::gemm<kLora, kDim, kOut>(reinterpret_cast<const float*>(dxh), dk, heads, C, dy2, nullptr, false, true);
        ok(hipGetLastError(), "a16 launch");
        ok(hipMemcpy(y2.data(), dy2, y2.size() * 4, hipMemcpyDeviceToHost), "y2 d2h");
        ok(hipFree(dxh), "xh free"); ok(hipFree(dy2), "y2 free");
        check(std::memcmp(y.data(), y2.data(), y.size() * 4) == 0,
              "out H" + std::to_string(heads) + " C" + std::to_string(C) + " FP16 A image == FP32 A (bitwise, guard kept)");
    }
    // Sampled rows: the first, the last, fragment / CTA edges and a stride; every head, every output.
    std::vector<unsigned> rows{0, C - 1};
    for (unsigned r : {15U, 16U, 127U, 128U, 129U}) if (r < C) rows.push_back(r);
    for (unsigned r = 7; r < C; r += std::max(1U, C / 13U)) rows.push_back(r);
    double e_contract = 0, e_drift = 0, ref2 = 0, ref2c = 0;
    bool finite = true;
    for (unsigned r : rows)
        for (unsigned h = 0; h < heads; ++h)
            for (unsigned o = 0; o < N; ++o) {
                double a16 = 0, a32 = 0;
                for (unsigned k = 0; k < K; ++k) {
                    const std::size_t wi = kOut ? (std::size_t{h} * 2 * kDim + kDim + o) * kLora + k : (std::size_t{h} * 2 * kDim + k) * kLora + o;
                    const double w = h2f(kv[wi]), xv = x[(std::size_t{r} * heads + h) * K + k];
                    a16 += static_cast<double>(r16(static_cast<float>(xv))) * w;
                    a32 += xv * w;
                }
                const double v = y[(std::size_t{r} * heads + h) * N + o];
                finite = finite && std::isfinite(v);
                e_contract += (v - a16) * (v - a16); ref2c += a16 * a16;
                e_drift += (v - a32) * (v - a32); ref2 += a32 * a32;
            }
    bool guard_ok = true;
    for (std::size_t i = yn; i < yn + guard; ++i) guard_ok = guard_ok && y[i] == kGuard;
    const double l2c = std::sqrt(e_contract / ref2c), l2d = std::sqrt(e_drift / ref2);
    const std::string name = std::string(kOut ? "out" : "absorb") + " heads " + std::to_string(heads) + " C " + std::to_string(C);
    std::printf("%s rel-L2 contract %.3g drift %.3g\n", name.c_str(), l2c, l2d);
    check(finite, name + " finite");
    check(guard_ok, name + " guard rows untouched");
    // FP32 accumulation in WMMA hardware order: rel-L2 ~8e-7 at K 256 and ~1.6e-6 at K 512 against the double contract,
    // the same value at every row count (uniform precision, not an indexing error); bound 4e-9 * K.
    check(l2c <= 4.0e-9 * K, name + " contract rel-L2 <= 4e-9 K");
    check(l2d <= 5.0e-4, name + " drift vs FP32 inputs rel-L2 <= 5e-4");
    if (timing) {
        // Arm 0 = the engine call; the others are placement diagnostics on the same buffers (outputs ignored): order 0 /
        // 1 / 2 (head groups of 4, 8, 16) on the engine strides (L), and on head-major [heads][C][K] strides (H).
        struct Arm { unsigned order, group; bool head_major; };
        const Arm arms[] = {{0, 1, false}, {0, 1, false}, {1, 1, false}, {2, 4, false}, {2, 8, false}, {2, 16, false},
                            {0, 1, true}, {1, 1, true}, {2, 8, true}};
        for (unsigned arm = 0; arm < sizeof(arms) / sizeof(arms[0]); ++arm) {
            auto p = M::args<kLora, kDim, kOut>(dx, dk, heads, C, dy);
            p.order = arms[arm].order;
            p.group = arms[arm].group;
            if (arms[arm].head_major) { p.a_row = K; p.a_batch = std::size_t{C} * K; p.y_row = N; p.y_batch = std::size_t{C} * N; }
            const auto launch = [&] {
                if constexpr (kOut) strix::core::f16_tile_gemm::gemm<strix::core::f16_tile_gemm::Layout::NK>(p, nullptr);
                else strix::core::f16_tile_gemm::gemm<strix::core::f16_tile_gemm::Layout::KN>(p, nullptr);
            };
            std::vector<double> t;
            for (unsigned i = 0; i < 9; ++i) {
                ok(hipDeviceSynchronize(), "sync");
                const auto a = std::chrono::steady_clock::now();
                if (arm == 0) M::gemm<kLora, kDim, kOut>(dx, dk, heads, C, dy, nullptr);
                else launch();
                ok(hipDeviceSynchronize(), "sync");
                t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
            }
            std::sort(t.begin(), t.end());
            const double mac = static_cast<double>(C) * heads * K * N;
            std::printf("TIME %s arm %u order %u group %u %s %.3f ms (%.2f TMAC/s)\n", name.c_str(), arm, arms[arm].order,
                        arms[arm].group, arms[arm].head_major ? "H" : "L", t[4], mac / t[4] / 1e9);
        }
    }
    ok(hipFree(dx), "free"); ok(hipFree(dy), "free"); ok(hipFree(dk), "free");
}
}  // namespace

int main(int argc, char** argv) {
    const bool timing = !(argc > 1 && std::strcmp(argv[1], "--no-timing") == 0);
    std::mt19937 g(20261115);
    try {
        for (unsigned C : {1U, 17U, 130U, 1000U}) {
            run<false>(32, C, g, false);
            run<true>(32, C, g, false);
        }
        run<false>(64, 300, g, false);
        run<true>(64, 300, g, false);
        for (unsigned C : {4096U, 8192U}) {
            run<false>(32, C, g, timing);
            run<true>(32, C, g, timing);
        }
    } catch (const std::exception& e) {
        std::printf("ERROR %s\n", e.what());
        return 2;
    }
    std::printf("checks: %u failures: %u\n", checks, failures);
    return failures ? 1 : 0;
}
