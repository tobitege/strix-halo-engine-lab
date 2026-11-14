// FAST glm-pf-kda-wmma-v1 (STRIX_GLM_PF_KDA_WMMA=1): the KDA plain-weight projections on the shared dense FP16 WMMA
// core against (a) a host reference of the contract, sum_k fp16(x) * w in double (accumulation error only: tight), and
// (b) the exact dense_rows / dense_rows_group chains (FP16 rounding of x: drift gate). Also finite outputs, untouched
// guard rows, then interleaved whole-call timing exact vs WMMA.
#include "glm/prefill/glm_pf_kda_wmma.hpp"
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

namespace P = strix::glm::pf;
namespace K = strix::glm::pf::kda_wmma;
namespace D = strix::glm::dev;
namespace {
unsigned checks = 0, failures = 0;
void check(bool c, const std::string& what) {
    ++checks;
    if (!c) { ++failures; std::printf("FAIL %s\n", what.c_str()); }
}
void ok(hipError_t e, const char* w) { if (e != hipSuccess) throw std::runtime_error(std::string(w) + ": " + hipGetErrorString(e)); }
std::vector<void*> allocs;
template<class T> T* alloc(std::size_t n) { T* p; ok(hipMalloc(&p, n * sizeof(T)), "alloc"); allocs.push_back(p); return p; }
constexpr float kGuard = -777.25F;

struct Proj { unsigned n; std::vector<_Float16> w; _Float16* dw; float* dy_exact; float* dy_wmma; };

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

// One shape: `in` inputs, projections of widths ns, rows R (the outputs have one guard row).
void run_case(std::mt19937& rng, unsigned in, const std::vector<unsigned>& ns, unsigned R, bool timing) {
    std::normal_distribution<float> nx(0.0F, 1.0F), nw(0.0F, 0.02F);
    std::vector<float> x(std::size_t{R} * in);
    for (auto& v : x) v = nx(rng);
    auto* dx = alloc<float>(x.size());
    ok(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice), "x");
    std::vector<Proj> pr;
    std::vector<P::DenseProj> pe, pw;
    for (const unsigned n : ns) {
        Proj p{n, std::vector<_Float16>(std::size_t{n} * in), nullptr, nullptr, nullptr};
        for (auto& v : p.w) v = static_cast<_Float16>(nw(rng));
        p.dw = alloc<_Float16>(p.w.size());
        ok(hipMemcpy(p.dw, p.w.data(), p.w.size() * 2, hipMemcpyHostToDevice), "w");
        const std::size_t yn = std::size_t{R + 1} * n;
        std::vector<float> guard(yn, kGuard);
        p.dy_exact = alloc<float>(yn);
        p.dy_wmma = alloc<float>(yn);
        ok(hipMemcpy(p.dy_exact, guard.data(), yn * 4, hipMemcpyHostToDevice), "g");
        ok(hipMemcpy(p.dy_wmma, guard.data(), yn * 4, hipMemcpyHostToDevice), "g");
        pe.push_back({p.dw, 0, n, p.dy_exact});
        pw.push_back({p.dw, 0, n, p.dy_wmma});
        pr.push_back(std::move(p));
    }
    const auto count = static_cast<unsigned>(ns.size());
    const std::string tag = "in=" + std::to_string(in) + " n0=" + std::to_string(ns[0]) + " count=" + std::to_string(count) + " rows=" + std::to_string(R);
    check(K::supported(D::WType::F16, in, pw.data(), count, R), tag + " supported");
    const auto exact = [&] {
        if (count > 1) P::dense_rows_group(dx, D::WType::F16, in, pe.data(), count, R, nullptr);
        else P::dense_rows(dx, pr[0].dw, D::WType::F16, in, 0, pr[0].n, pr[0].dy_exact, R, nullptr);
    };
    const auto wmma = [&] { K::dense_group(dx, in, pw.data(), count, R, nullptr); };
    exact();
    wmma();
    ok(hipDeviceSynchronize(), "sync");
    for (unsigned i = 0; i < count; ++i) {
        const unsigned n = pr[i].n;
        std::vector<float> ye(std::size_t{R + 1} * n), yw(ye.size());
        ok(hipMemcpy(ye.data(), pr[i].dy_exact, ye.size() * 4, hipMemcpyDeviceToHost), "ye");
        ok(hipMemcpy(yw.data(), pr[i].dy_wmma, yw.size() * 4, hipMemcpyDeviceToHost), "yw");
        bool finite = true, guard = true;
        double num = 0, den = 0, cnum = 0, cden = 0;
        // Contract reference on a sample of rows (every row when small).
        const unsigned step = R > 64 ? R / 61 : 1;
        for (unsigned r = 0; r < R; ++r) {
            const bool sample = r % step == 0 || r + 1 == R;
            for (unsigned o = 0; o < n; ++o) {
                const double e = ye[std::size_t{r} * n + o], g = yw[std::size_t{r} * n + o];
                finite = finite && std::isfinite(g);
                num += (g - e) * (g - e);
                den += e * e;
                if (!sample) continue;
                double ref = 0;
                const float* xr = &x[std::size_t{r} * in];
                const _Float16* wr = &pr[i].w[std::size_t{o} * in];
                for (unsigned k = 0; k < in; ++k) ref += static_cast<double>(static_cast<_Float16>(xr[k])) * static_cast<double>(wr[k]);
                cnum += (g - ref) * (g - ref);
                cden += ref * ref;
            }
        }
        for (unsigned o = 0; o < n; ++o) guard = guard && yw[std::size_t{R} * n + o] == kGuard;
        const double drift = std::sqrt(num / std::max(den, 1e-300)), contract = std::sqrt(cnum / std::max(cden, 1e-300));
        std::uint64_t fnv = 1469598103934665603ULL;  // FNV-1a of the output bytes, guard row included
        for (const float v : yw) {
            std::uint32_t b;
            std::memcpy(&b, &v, 4);
            for (unsigned k = 0; k < 4; ++k) fnv = (fnv ^ ((b >> (8U * k)) & 0xFFU)) * 1099511628211ULL;
        }
        std::printf("CASE %s proj=%u n=%u drift-vs-exact rel-l2 %.3e contract rel-l2 %.3e fnv %016llx\n", tag.c_str(), i, n, drift, contract,
                    static_cast<unsigned long long>(fnv));
        check(finite, tag + " finite");
        check(guard, tag + " guard row untouched");
        check(contract < 2e-5, tag + " contract (accumulation error only)");
        check(drift < 2e-3, tag + " drift against the exact chains");
    }
    if (!timing) return;
    std::vector<double> te, tw;
    for (unsigned rep = 0; rep < 9; ++rep) {
        for (unsigned arm = 0; arm < 2; ++arm) {
            ok(hipDeviceSynchronize(), "sync");
            const auto a = std::chrono::steady_clock::now();
            for (unsigned q = 0; q < 8; ++q) { if (arm) wmma(); else exact(); }
            ok(hipDeviceSynchronize(), "sync");
            (arm ? tw : te).push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count() / 8.0);
        }
    }
    double mac = 0;
    for (const unsigned n : ns) mac += static_cast<double>(n) * in * R;
    const double e = median(te), w = median(tw);
    std::printf("TIME %s exact %.3f ms (%.2f TMAC/s) kda-wmma %.3f ms (%.2f TMAC/s) ratio %.3f\n", tag.c_str(), e, mac / e * 1e-9, w,
                mac / w * 1e-9, w / e);
}
}  // namespace

int main() {
    try {
        std::mt19937 rng(20261112U);
        for (const unsigned R : {1U, 15U, 16U, 127U, 128U, 129U, 300U, 2048U}) {
            run_case(rng, 4096, {128, 128, 64}, R, false);  // f_a | g_a | b (64 heads): 16-tile column + 4-tile remainder
            run_case(rng, 4096, {128, 128, 32}, R, false);  // 32 heads: 2 live tiles of the 4-tile remainder
            run_case(rng, 128, {4096}, R, false);           // f_b / g_b
        }
        run_case(rng, 128, {8192}, 513, false);
        run_case(rng, 256, {48}, 33, false);  // 4-tile kernel only, dead tile
        const P::DenseProj bad{nullptr, 0, 24, nullptr};
        check(!K::supported(D::WType::F16, 4096, &bad, 1, 8), "ragged width is not supported");
        check(!K::supported(D::WType::BF16, 4096, &bad, 1, 8), "BF16 weights are not supported");
        const P::DenseProj odd{&bad, 0, 32, const_cast<float*>(reinterpret_cast<const float*>(&bad))};
        check(!K::supported(D::WType::F16, 96, &odd, 1, 8), "in must be a multiple of 128");
        run_case(rng, 4096, {128, 128, 64}, 4096, true);
        run_case(rng, 4096, {128, 128, 32}, 4096, true);
        run_case(rng, 128, {8192}, 4096, true);
        run_case(rng, 128, {4096}, 4096, true);
        for (void* p : allocs) (void)hipFree(p);
        std::printf("glm pf kda wmma: %u checks, %u failures\n", checks, failures);
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("ERROR %s\n", e.what());
        return 2;
    }
}
