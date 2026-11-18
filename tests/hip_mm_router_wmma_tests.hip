// FAST mm-pf-router-wmma-v1 (enqueue_router_wmma): the MiniMax prefill router product (128 x 6144) on the shared dense
// FP16 WMMA core, against the exact tiled router (enqueue_router_tiled). Checks: (a) the three launch forms give the
// same bytes (scores, ids, weights); (b) rows whose y overflows FP16 are flagged and get the exact router's bits, and
// the flags are zero after every call; (c) drift report (top-4 set agreement, max score difference). Timing: exact vs
// each form, median of reps per row count. Needs a GPU. Exit 0 = PASS.
#include "strix/hip_minimax_moe.hpp"
#include "router_wmma_gate_checks.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace moe = strix::minimax::moe;

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

struct Outs {
    std::vector<float> scores, w;
    std::vector<std::uint32_t> ids;
};

void run_case(std::mt19937& rng, unsigned rows, bool timing) {
    constexpr unsigned H = moe::kHidden, E = moe::kExperts, K = moe::kTopE;
    std::normal_distribution<float> ny(0.0F, 1.0F), nw(0.0F, 0.02F), nb(0.0F, 0.01F);
    std::vector<float> y(std::size_t{rows} * H), rw(std::size_t{E} * H), bias(E);
    for (auto& v : y) v = ny(rng);
    for (auto& v : rw) v = nw(rng);
    for (auto& v : bias) v = nb(rng);
    router_wmma_test::prepare_probe(y, H, rw, E, 1.0F);
    std::vector<unsigned> big;  // rows with an FP16 overflow of y (finite FP32): the exact fallback must take them
    for (unsigned r : {3U, rows / 2U + 1U, rows - 1U})
        if (r < rows && std::find(big.begin(), big.end(), r) == big.end()) big.push_back(r);
    const std::vector<float> clean = y;  // timing input: no overflow rows (the fallback is the rare path)
    for (unsigned r : big) router_wmma_test::inject_overflow(y, H, r);
    float *dy = dev(y), *drw = dev(rw), *db = dev(bias);
    auto* dw16 = dev(std::vector<std::uint16_t>(rw.size()));
    moe::hip::convert_router_f16(drw, dw16, nullptr);
    auto* scratch = dev(std::vector<float>(moe::hip::router_tiled_scratch_floats(rows), -77.5F));
    auto* flags = dev(std::vector<std::uint32_t>(rows, 0U));
    auto* di = dev(std::vector<std::uint32_t>(std::size_t{rows} * K, 0xdeadbeefU));
    auto* dwt = dev(std::vector<float>(std::size_t{rows} * K, -99.25F));
    const auto exact = [&] { moe::hip::enqueue_router_tiled(rows, dy, drw, db, di, dwt, scratch, nullptr); };
    const auto wmma = [&](unsigned form) { moe::hip::enqueue_router_wmma(rows, dy, drw, dw16, db, di, dwt, scratch, flags, nullptr, form); };
    const auto collect = [&] {
        ok(hipDeviceSynchronize(), "sync");
        return Outs{host(scratch, moe::hip::router_tiled_scratch_floats(rows)), host(dwt, std::size_t{rows} * K), host(di, std::size_t{rows} * K)};
    };
    exact();
    const Outs ref = collect();
    Outs form[3];
    for (unsigned f = 0; f < 3; ++f) {
        ok(hipMemset(scratch, 0xff, moe::hip::router_tiled_scratch_floats(rows) * sizeof(float)), "poison scores");
        ok(hipMemset(dwt, 0xff, std::size_t{rows} * K * sizeof(float)), "poison weights");
        ok(hipMemset(di, 0xff, std::size_t{rows} * K * sizeof(std::uint32_t)), "poison ids");
        wmma(f);
        form[f] = collect();
        check(router_wmma_test::complete(form[f].scores) && router_wmma_test::complete(form[f].w),
              "all scores and weights written and finite form " + std::to_string(f));
        check(std::all_of(form[f].ids.begin(), form[f].ids.end(), [](std::uint32_t id) { return id < E; }),
              "all ids written and valid form " + std::to_string(f));
        if (rows > 1U)
            for (unsigned e = 0; e < E; ++e)
                check(router_wmma_test::wmma_probe(form[f].scores[std::size_t{e} * rows], ref.scores[std::size_t{e} * rows],
                                                  1.0F / (1.0F + std::exp(-1.0F)), 1.0e-6F), "finite WMMA probe (not exact fallback)");
        for (unsigned r : big)
            for (unsigned e = 0; e < 2U * E; ++e)
                check(std::memcmp(&form[f].scores[std::size_t{e} * rows + r], &ref.scores[std::size_t{e} * rows + r], sizeof(float)) == 0,
                      "injected NaN product replaced by exact fallback form " + std::to_string(f));
        const auto fl = host(flags, rows);
        check(std::all_of(fl.begin(), fl.end(), [](std::uint32_t v) { return v == 0U; }), "flags zero after form " + std::to_string(f));
    }
    const std::string tag = "rows " + std::to_string(rows);
    for (unsigned f = 1; f < 3; ++f) {
        check(std::memcmp(form[f].scores.data(), form[0].scores.data(), form[0].scores.size() * 4) == 0, tag + " scores form " + std::to_string(f) + " == plan");
        check(form[f].ids == form[0].ids, tag + " ids form " + std::to_string(f) + " == plan");
        check(std::memcmp(form[f].w.data(), form[0].w.data(), form[0].w.size() * 4) == 0, tag + " weights form " + std::to_string(f) + " == plan");
    }
    unsigned same_set = 0, same_order = 0, n = 0;
    double max_ds = 0.0;
    for (unsigned r = 0; r < rows; ++r) {
        const bool overflow = std::find(big.begin(), big.end(), r) != big.end();
        bool bitwise = true;
        for (unsigned e = 0; e < 2U * E; ++e) {
            const float a = ref.scores[std::size_t{e} * rows + r], b = form[0].scores[std::size_t{e} * rows + r];
            bitwise = bitwise && std::memcmp(&a, &b, 4) == 0;
            if (!overflow) max_ds = std::max(max_ds, static_cast<double>(std::fabs(a - b)));
        }
        const auto* ia = &ref.ids[std::size_t{r} * K];
        const auto* ib = &form[0].ids[std::size_t{r} * K];
        if (overflow) {
            check(bitwise && std::equal(ia, ia + K, ib) && std::memcmp(&ref.w[std::size_t{r} * K], &form[0].w[std::size_t{r} * K], K * 4) == 0,
                  tag + " overflow row " + std::to_string(r) + " == exact bits");
            continue;
        }
        std::vector<std::uint32_t> sa(ia, ia + K), sb(ib, ib + K);
        same_order += sa == sb;
        std::sort(sa.begin(), sa.end());
        std::sort(sb.begin(), sb.end());
        same_set += sa == sb;
        ++n;
    }
    if (n != 0) {
        std::printf("DRIFT mm %s: top-4 set agreement %.4f (order %.4f), max |d score| %.3e\n", tag.c_str(),
                    static_cast<double>(same_set) / n, static_cast<double>(same_order) / n, max_ds);
        check(same_set >= n * 95U / 100U, tag + " top-4 set agreement >= 95 %");
    }
    if (!timing) return;
    ok(hipMemcpy(dy, clean.data(), clean.size() * sizeof(float), hipMemcpyHostToDevice), "clean y");
    hipEvent_t a, b;
    ok(hipEventCreate(&a), "event");
    ok(hipEventCreate(&b), "event");
    const auto time = [&](auto&& f) {
        std::vector<double> ms;
        for (unsigned i = 0; i < 25; ++i) {
            ok(hipEventRecord(a, nullptr), "rec");
            f();
            ok(hipEventRecord(b, nullptr), "rec");
            ok(hipEventSynchronize(b), "sync");
            float t = 0.0F;
            ok(hipEventElapsedTime(&t, a, b), "elapsed");
            if (i >= 5) ms.push_back(t);
        }
        return median(ms);
    };
    const double te = time(exact), t0 = time([&] { wmma(0); }), t1 = time([&] { wmma(1); }), t2 = time([&] { wmma(2); });
    std::printf("TIME mm rows %u: exact tiled %.3f ms (%.3f us/row) | wmma plan %.3f narrow4 %.3f wide16 %.3f ms | auto %s\n", rows, te,
                1000.0 * te / rows, t0, t1, t2, moe::hip::router_wmma_form());
    ok(hipEventDestroy(a), "event");
    ok(hipEventDestroy(b), "event");
}
}  // namespace

int main(int argc, char** argv) {
    const bool timing = argc > 1 && std::strcmp(argv[1], "--timing") == 0;
    try {
        std::mt19937 rng(20261115U);
        for (const unsigned rows : {1U, 129U, 1000U, 1024U, 2048U, 4096U}) run_case(rng, rows, timing && rows >= 1024U);
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    for (void* p : allocs) (void)hipFree(p);
    std::printf("%s mm router wmma: %u checks, %u failures\n", failures ? "FAIL" : "PASS", checks, failures);
    return failures ? 1 : 0;
}
