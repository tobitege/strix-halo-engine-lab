// FAST glm-pf-router-wmma-v1 (STRIX_GLM_PF_ROUTER_WMMA=1): the GLM prefill router product (288 x 4096 F16) on the shared
// dense FP16 WMMA core, against the exact R1 chains of dense_rows. Checks: (a) the three launch forms give the same
// bytes; (b) rows whose x overflows FP16 are flagged and get dense_rows' bits, and the flags are zero after every call;
// (c) drift report: max |d logit| and top-8 set agreement under sigmoid + bias (host selection, ties to the lower id).
// Timing: exact vs each form, median of reps. Needs a GPU. Exit 0 = PASS.
#include "glm/prefill/glm_pf_router_wmma.hpp"
#include "glm/prefill/glm_prefill.hpp"
#include "router_wmma_gate_checks.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace P = strix::glm::pf;
namespace RW = strix::glm::pf::router_wmma;
namespace D = strix::glm::dev;

namespace {
unsigned checks = 0, failures = 0;
void check(bool c, const std::string& what) {
    ++checks;
    if (!c) { ++failures; std::printf("FAIL %s\n", what.c_str()); }
}
void ok(hipError_t e, const char* w) { if (e != hipSuccess) throw std::runtime_error(std::string(w) + ": " + hipGetErrorString(e)); }
std::vector<void*> allocs;
template <class T> T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    ok(hipMalloc(&p, std::max<std::size_t>(h.size(), 1) * sizeof(T)), "malloc");
    ok(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    allocs.push_back(p);
    return p;
}
template <class T> std::vector<T> host(const T* d, std::size_t n) {
    std::vector<T> h(n);
    ok(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
    return h;
}
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

std::vector<unsigned> top8(const float* lg, const std::vector<float>& bias) {
    std::vector<unsigned> id(RW::kOut);
    std::iota(id.begin(), id.end(), 0U);
    std::vector<float> key(RW::kOut);
    for (unsigned e = 0; e < RW::kOut; ++e) key[e] = 1.0F / (1.0F + std::exp(-lg[e])) + bias[e];
    std::partial_sort(id.begin(), id.begin() + 8, id.end(), [&](unsigned a, unsigned b) { return key[a] > key[b] || (key[a] == key[b] && a < b); });
    id.resize(8);
    std::sort(id.begin(), id.end());
    return id;
}

void run_case(std::mt19937& rng, unsigned rows, bool timing) {
    constexpr unsigned H = RW::kIn, E = RW::kOut;
    std::normal_distribution<float> nx(0.0F, 1.0F), nw(0.0F, 0.02F), nb(0.0F, 0.01F);
    std::vector<float> x(std::size_t{rows} * H), bias(E);
    std::vector<std::uint16_t> w(std::size_t{E} * H);
    for (auto& v : x) v = nx(rng);
    for (auto& v : w) v = __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(nw(rng)));
    for (auto& v : bias) v = nb(rng);
    router_wmma_test::prepare_probe(x, H, w, E, std::uint16_t{0x3c00});
    std::vector<unsigned> big;  // rows with an FP16 overflow of x (finite FP32): the exact fallback must take them
    for (unsigned r : {3U, rows / 2U + 1U, rows - 1U})
        if (r < rows && std::find(big.begin(), big.end(), r) == big.end()) big.push_back(r);
    const std::vector<float> clean = x;  // timing input: no overflow rows (the fallback is the rare path)
    for (unsigned r : big) router_wmma_test::inject_overflow(x, H, r);
    float* dx = dev(x);
    auto* dw = dev(w);
    auto* flags = dev(std::vector<std::uint32_t>(rows, 0U));
    float* dy = dev(std::vector<float>(std::size_t{rows} * E, -77.5F));
    const auto exact = [&] { P::dense_rows(dx, dw, D::WType::F16, H, 0, E, dy, rows, nullptr); };
    const auto wmma = [&](unsigned form) { RW::logits(dx, dw, dy, rows, flags, nullptr, form); };
    const auto collect = [&] {
        ok(hipDeviceSynchronize(), "sync");
        return host(dy, std::size_t{rows} * E);
    };
    exact();
    const auto ref = collect();
    std::vector<float> form[3];
    for (unsigned f = 0; f < 3; ++f) {
        // All-ones is a NaN: a skipped store cannot inherit exact or another form's output.
        ok(hipMemset(dy, 0xff, std::size_t{rows} * E * sizeof(float)), "poison logits");
        wmma(f);
        form[f] = collect();
        check(router_wmma_test::complete(form[f]), "all logits written and finite form " + std::to_string(f));
        if (rows > 1U)
            for (unsigned e = 0; e < E; ++e)
                check(router_wmma_test::wmma_probe(form[f][e], ref[e], 1.0F, 0.0F), "finite WMMA probe (not exact fallback)");
        for (unsigned r : big)
            check(std::memcmp(&form[f][std::size_t{r} * E], &ref[std::size_t{r} * E], E * sizeof(float)) == 0,
                  "injected NaN product replaced by exact fallback form " + std::to_string(f));
        const auto fl = host(flags, rows);
        check(std::all_of(fl.begin(), fl.end(), [](std::uint32_t v) { return v == 0U; }), "flags zero after form " + std::to_string(f));
    }
    const std::string tag = "rows " + std::to_string(rows);
    for (unsigned f = 1; f < 3; ++f)
        check(std::memcmp(form[f].data(), form[0].data(), form[0].size() * 4) == 0, tag + " logits form " + std::to_string(f) + " == plan");
    unsigned same = 0, n = 0;
    double max_d = 0.0, sum_d = 0.0;
    for (unsigned r = 0; r < rows; ++r) {
        const float* a = &ref[std::size_t{r} * E];
        const float* b = &form[0][std::size_t{r} * E];
        if (std::find(big.begin(), big.end(), r) != big.end()) {
            check(std::memcmp(a, b, E * 4) == 0, tag + " overflow row " + std::to_string(r) + " == dense_rows bits");
            continue;
        }
        for (unsigned e = 0; e < E; ++e) {
            const double d = std::fabs(static_cast<double>(a[e]) - b[e]);
            max_d = std::max(max_d, d);
            sum_d += d;
        }
        same += top8(a, bias) == top8(b, bias);
        ++n;
    }
    if (n != 0) {
        std::printf("DRIFT glm %s: top-8 set agreement %.4f, max |d logit| %.3e, mean %.3e\n", tag.c_str(), static_cast<double>(same) / n,
                    max_d, sum_d / (static_cast<double>(n) * E));
        check(same >= n * 95U / 100U, tag + " top-8 set agreement >= 95 %");
    }
    if (!timing) return;
    ok(hipMemcpy(dx, clean.data(), clean.size() * sizeof(float), hipMemcpyHostToDevice), "clean x");
    hipEvent_t ea, eb;
    ok(hipEventCreate(&ea), "event");
    ok(hipEventCreate(&eb), "event");
    const auto time = [&](auto&& f) {
        std::vector<double> ms;
        for (unsigned i = 0; i < 25; ++i) {
            ok(hipEventRecord(ea, nullptr), "rec");
            f();
            ok(hipEventRecord(eb, nullptr), "rec");
            ok(hipEventSynchronize(eb), "sync");
            float t = 0.0F;
            ok(hipEventElapsedTime(&t, ea, eb), "elapsed");
            if (i >= 5) ms.push_back(t);
        }
        return median(ms);
    };
    const double te = time(exact), t0 = time([&] { wmma(0); }), t1 = time([&] { wmma(1); }), t2 = time([&] { wmma(2); });
    std::printf("TIME glm rows %u: exact dense_rows %.3f ms (%.3f us/row) | wmma plan %.3f narrow4 %.3f wide16 %.3f ms | auto %s\n", rows, te,
                1000.0 * te / rows, t0, t1, t2, strix::core::f16_dense_tall::matrix_form_name(RW::form()));
    ok(hipEventDestroy(ea), "event");
    ok(hipEventDestroy(eb), "event");
}
}  // namespace

int main(int argc, char** argv) {
    const bool timing = argc > 1 && std::strcmp(argv[1], "--timing") == 0;
    try {
        std::mt19937 rng(20261115U);
        for (const unsigned rows : {1U, 129U, 1000U, 2048U, 4096U, 8192U}) run_case(rng, rows, timing && rows >= 2048U);
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    for (void* p : allocs) (void)hipFree(p);
    std::printf("%s glm pf router wmma: %u checks, %u failures\n", failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
