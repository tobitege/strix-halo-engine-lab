// FAST glm-pf-wmma-x (STRIX_GLM_PF_WMMA=1) against the exact X3 cores on random EXL3 weights: dense linears through
// pf::exact::linear_rows (X3 route forced), grouped MoE through pf::exact_moe::enqueue (V11 forced). Not bitwise (new
// identity): the gate is rel-L2 and max-abs drift against the exact outputs, finite outputs and untouched padding
// rows; then interleaved whole-call timing exact vs WMMA.
#include "glm/prefill/glm_pf_exact.hpp"
#include "glm/prefill/glm_pf_exact_moe.hpp"
#include "glm/moe/glm_trellis_device.hpp"
#include "strix/env_flags.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace M = strix::glm::moe;
namespace X = strix::glm::pf::exact_moe;
namespace DX = strix::glm::pf::exact;
namespace {
constexpr unsigned H = 4096, I = 1024, E = 288, Top = 8;
unsigned checks = 0, failures = 0;
void check(bool c, const std::string& what) {
    ++checks;
    if (!c) { ++failures; std::printf("FAIL %s\n", what.c_str()); }
}
void ok(hipError_t e, const char* w) { if (e != hipSuccess) throw std::runtime_error(std::string(w) + ": " + hipGetErrorString(e)); }
void env(const char* k, const char* v) {
    if ((v ? setenv(k, v, 1) : unsetenv(k)) != 0) throw std::runtime_error("setenv");
}
std::uint16_t half(float x) { const _Float16 h = static_cast<_Float16>(x); std::uint16_t b; std::memcpy(&b, &h, 2); return b; }
std::vector<void*> allocs;
template<class T> T* alloc(std::size_t n) { T* p; ok(hipMalloc(&p, n * sizeof(T)), "alloc"); allocs.push_back(p); return p; }
M::Exl3 make_matrix(std::mt19937& rng, unsigned K, unsigned N, unsigned bits) {
    std::vector<std::uint32_t> tr(std::size_t{K} * N * bits / 32);
    for (auto& v : tr) v = static_cast<std::uint32_t>(rng());
    std::vector<std::uint16_t> suh(K), svh(N);
    for (auto& v : suh) v = half((rng() & 1U ? 1.0F : -1.0F) * (0.5F + static_cast<float>(rng() % 1024) / 2048.0F));
    for (auto& v : svh) v = half(rng() & 1U ? 0.02F : -0.02F);
    auto* dt = alloc<std::uint32_t>(tr.size()); auto* ds = alloc<std::uint16_t>(K); auto* dv = alloc<std::uint16_t>(N);
    ok(hipMemcpy(dt, tr.data(), tr.size() * 4, hipMemcpyHostToDevice), "t");
    ok(hipMemcpy(ds, suh.data(), K * 2, hipMemcpyHostToDevice), "s");
    ok(hipMemcpy(dv, svh.data(), N * 2, hipMemcpyHostToDevice), "v");
    return {dt, ds, dv, K, N, bits};
}
struct Drift { double rel_l2 = 0, max_rel = 0; std::size_t equal = 0, n = 0, nonfinite = 0, first_nonfinite = 0; bool finite = true; };
Drift drift(const std::vector<float>& ref, const std::vector<float>& got) {
    Drift d;
    double num = 0, den = 0, amax = 0, dmax = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        d.finite = d.finite && std::isfinite(ref[i]) && std::isfinite(got[i]);
        if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) { if (!d.nonfinite++) d.first_nonfinite = i; continue; }
        const double e = static_cast<double>(got[i]) - ref[i];
        num += e * e; den += static_cast<double>(ref[i]) * ref[i];
        amax = std::max(amax, std::fabs(static_cast<double>(ref[i])));
        dmax = std::max(dmax, std::fabs(e));
        d.equal += std::memcmp(&ref[i], &got[i], 4) == 0;
    }
    d.n = ref.size();
    d.rel_l2 = den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
    d.max_rel = amax > 0 ? dmax / amax : dmax;
    return d;
}
// Drift vs the exact cores. Linear outputs (FP32 sums of the same FP16 products): rel-L2 <= 2e-6, max <= 5e-6 of
// max |ref| (measured 2.4-7.2e-7; one WMMA chain over all of K measured 1.3e-5 at K 4096 from the non-RNE
// accumulator add, and a wrong operand or layout is >= 1e-2). Outputs behind the FP16 down input (pair outputs, R16)
// inherit FP16 rounding flips (half step 4.9e-4 relative): rel-L2 <= 1e-4, max <= 1e-3. Accuracy itself: FP64 TRUTH.
constexpr double kLinL2 = 2e-6, kLinMax = 5e-6, kF16L2 = 1e-4, kF16Max = 1e-3;
bool gate(const Drift& d, const std::string& name, double l2 = kLinL2, double mx = kLinMax) {
    std::printf("DRIFT %s rel_l2=%.3e max_rel=%.3e bitequal=%.4f n=%zu\n", name.c_str(), d.rel_l2, d.max_rel,
                d.n ? static_cast<double>(d.equal) / static_cast<double>(d.n) : 0.0, d.n);
    if (!d.finite) std::printf("NONFINITE %s count=%zu first=%zu\n", name.c_str(), d.nonfinite, d.first_nonfinite);
    const bool pass = d.finite && d.rel_l2 <= l2 && d.max_rel <= mx;
    check(pass, name + " drift band");
    return pass;
}
// FP64 truth for the dense check: decoded weights (the cores' mul1_pair values) as FP32 [K][N], FP64 dot products of
// the transformed FP16 rows, then the H128 output transform and svh on the host in FP64.
template<int Bits>
__global__ void decode_kernel(const unsigned char* packed, unsigned N, float* w) {
    namespace dv = strix::glm::moe::dev;
    const unsigned lane = threadIdx.x, nt = blockIdx.x, kt = blockIdx.y;
    const dv::LaneGeom<Bits> geo(lane);
    const dv::TileStates<Bits> st(packed + (std::size_t{kt} * (N / 16U) + nt) * Bits * 32U, geo);
    const dv::Half2 v[4] = {dv::mul1_pair(st.template state<0>(), st.template state<1>()),
                            dv::mul1_pair(st.template state<2>(), st.template state<3>()),
                            dv::mul1_pair(st.template state<4>(), st.template state<5>()),
                            dv::mul1_pair(st.template state<6>(), st.template state<7>())};
    const unsigned o = lane / 4U, p = lane & 3U;
    for (unsigned i = 0; i < 4U; ++i) {
        const unsigned out = nt * 16U + o + (i >= 2U ? 8U : 0U), pair = p + (i & 1U ? 4U : 0U);
        for (unsigned e = 0; e < 2U; ++e) w[(std::size_t{kt} * 16U + 2U * pair + e) * N + out] = static_cast<float>(v[i][e]);
    }
}
__global__ void dot64_kernel(const _Float16* t, const float* w, unsigned K, unsigned N, unsigned rows, double* acc) {
    const unsigned o = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (o >= N || r >= rows) return;
    double a = 0;
    for (unsigned k = 0; k < K; ++k) a += static_cast<double>(t[std::size_t{r} * K + k]) * static_cast<double>(w[std::size_t{k} * N + o]);
    acc[std::size_t{r} * N + o] = a;
}
// Arms: 0 exact X3 / V11, 1 WMMA 64-row tiles, 2 WMMA default (128-row tiles, two output tiles per wave, MoE remainders
// of 65..127 rows on a 128-row tile), 3 the same with those remainders on the 64-row tail kernel, 4 default with each
// group's 128-row tiles in one CTA instead of two (the default), 5 default with one output tile per wave (CP 1).
// Arm 6: default with the WMMA chains of row fragments past a ragged tile's rows computed (LIVE_FRAG=0).
// Arms 1-6: same bits. Every arm sets every knob explicitly (arm 0 sets WMMA=0), so inherited settings cannot
// collapse two arms into one.
constexpr unsigned kArms = 7;
void arm_env(unsigned arm) {
    env("STRIX_GLM_PF_WMMA", arm ? "1" : "0");
    env("STRIX_GLM_PF_WMMA_ROWS", arm == 1 ? "64" : "128");
    env("STRIX_GLM_PF_WMMA_TAIL128", arm == 3 ? "0" : "1");
    env("STRIX_GLM_PF_WMMA_SPLIT", arm == 4 ? "1" : "2");
    env("STRIX_GLM_PF_WMMA_CP", arm == 5 ? "1" : "2");
    env("STRIX_GLM_PF_WMMA_LIVE_FRAG", arm == 6 ? "0" : "1");
    env("STRIX_GLM_PF_EXACTMOE_TM", nullptr);
    env("STRIX_GLM_PF_EXACTMOE_PLAN", nullptr);
    env("STRIX_GLM_PF_AUX_COMBINE", "1");  // pinned: the shared pair slot is only meaningful with the separate combine
}
bool same_bits(const std::vector<float>& a, const std::vector<float>& b, const std::string& name) {
    const bool same = a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
    check(same, name + " WMMA forms bitwise");
    return same;
}
template<class F> double median_ms(F&& f, hipStream_t s, unsigned reps) {
    std::vector<double> t;
    for (unsigned k = 0; k < reps; ++k) {
        const auto a = std::chrono::steady_clock::now();
        f(); ok(hipStreamSynchronize(s), "timed");
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

void dense(std::mt19937& rng, hipStream_t s, unsigned reps) {
    struct Shape { unsigned K, N, bits; };
    for (const Shape sh : {Shape{4096, 1024, 4}, Shape{1024, 4096, 5}, Shape{4096, 6144, 6}, Shape{2048, 4096, 3}}) {
        const M::Exl3 w = make_matrix(rng, sh.K, sh.N, sh.bits);
        strix::glm::engine::Linear l;
        l.raw.trellis = w.trellis; l.raw.suh = w.suh; l.raw.svh = w.svh;
        l.raw.input = sh.K; l.raw.output = sh.N; l.raw.bits = sh.bits; l.raw.codebook = 2;
        check(DX::supported(l), "dense supported");
        constexpr unsigned Rmax = 8192;
        float* x = alloc<float>(std::size_t{Rmax} * sh.K);
        float* y = alloc<float>(std::size_t{Rmax + 1} * sh.N);
        const std::size_t wsb = DX::workspace_bytes(Rmax, sh.K);
        void* ws = alloc<unsigned char>(wsb);
        std::vector<float> hx(std::size_t{Rmax} * sh.K);
        std::normal_distribution<float> n01(0, 1);
        for (auto& v : hx) v = n01(rng);
        ok(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice), "x");
        const std::string shape = "dense K" + std::to_string(sh.K) + " N" + std::to_string(sh.N) + " b" + std::to_string(sh.bits);
        for (unsigned rows : {1U, 17U, 64U, 100U, 129U, 200U, 257U, 2048U, 8192U}) {  // 129..257: partial tile after full ones
            std::vector<float> out[kArms];
            for (unsigned arm = 0; arm < kArms; ++arm) {
                env("STRIX_GLM_PF_EXACT_X", "4");
                arm_env(arm);
                ok(hipMemsetAsync(y, 0xff, std::size_t{Rmax + 1} * sh.N * 4, s), "poison");
                DX::linear_rows(l, x, sh.K, y, sh.N, rows, ws, wsb, s);
                ok(hipStreamSynchronize(s), "dense");
                out[arm].resize(std::size_t{rows} * sh.N);
                ok(hipMemcpy(out[arm].data(), y, out[arm].size() * 4, hipMemcpyDeviceToHost), "y");
                std::uint32_t tail[64];
                ok(hipMemcpy(tail, y + std::size_t{rows} * sh.N, sizeof tail, hipMemcpyDeviceToHost), "tail");
                check(std::all_of(tail, tail + 64, [](std::uint32_t t) { return t == 0xffffffffU; }), shape + " rows=" + std::to_string(rows) + " arm=" + std::to_string(arm) + " padding untouched");
            }
            gate(drift(out[0], out[1]), shape + " rows=" + std::to_string(rows));
            for (unsigned arm = 2; arm < kArms; ++arm) same_bits(out[1], out[arm], shape + " rows=" + std::to_string(rows) + " arm " + std::to_string(arm));
            if (rows == 17U || rows == 8192U) {
                // ws still holds the transformed FP16 rows (same input_kernel in both arms).
                const unsigned rr = std::min(rows, 16U);
                float* wd = alloc<float>(std::size_t{sh.K} * sh.N);
                double* acc = alloc<double>(std::size_t{rr} * sh.N);
                const dim3 g(sh.N / 16U, sh.K / 16U);
                const auto* packed = reinterpret_cast<const unsigned char*>(w.trellis);
                switch (sh.bits) {
                case 3: decode_kernel<3><<<g, 32, 0, s>>>(packed, sh.N, wd); break;
                case 4: decode_kernel<4><<<g, 32, 0, s>>>(packed, sh.N, wd); break;
                case 5: decode_kernel<5><<<g, 32, 0, s>>>(packed, sh.N, wd); break;
                default: decode_kernel<6><<<g, 32, 0, s>>>(packed, sh.N, wd); break;
                }
                dot64_kernel<<<dim3((sh.N + 255U) / 256U, rr), 256, 0, s>>>(static_cast<const _Float16*>(ws), wd, sh.K, sh.N, rr, acc);
                ok(hipStreamSynchronize(s), "truth");
                std::vector<double> a(std::size_t{rr} * sh.N);
                std::vector<std::uint16_t> svh(sh.N);
                ok(hipMemcpy(a.data(), acc, a.size() * 8, hipMemcpyDeviceToHost), "acc");
                ok(hipMemcpy(svh.data(), w.svh, sh.N * 2, hipMemcpyDeviceToHost), "svh");
                double e[2] = {0, 0}, den = 0;
                for (unsigned r = 0; r < rr; ++r) for (unsigned b = 0; b < sh.N; b += 128U) for (unsigned i = 0; i < 128U; ++i) {
                    double yv = 0;
                    for (unsigned m = 0; m < 128U; ++m)
                        yv += (__builtin_popcount(i & m) & 1 ? -1.0 : 1.0) * a[std::size_t{r} * sh.N + b + m];
                    _Float16 h;
                    std::memcpy(&h, &svh[b + i], 2);
                    yv = yv / std::sqrt(128.0) * static_cast<double>(h);
                    den += yv * yv;
                    for (unsigned arm = 0; arm < 2; ++arm) {
                        const double d = static_cast<double>(out[arm][std::size_t{r} * sh.N + b + i]) - yv;
                        e[arm] += d * d;
                    }
                }
                const double ex = std::sqrt(e[0] / den), wm = std::sqrt(e[1] / den);
                std::printf("TRUTH %s rows=%u (first %u) rel_l2 vs FP64: exact %.3e wmma %.3e ratio %.2f\n", shape.c_str(), rows, rr,
                            ex, wm, ex > 0 ? wm / ex : 0.0);
                // Stage-partial WMMA chains folded with RNE adds: at most 2x exact's error (+2e-7) and 1e-6 (t2: 2.5-2.8e-7).
                check(wm <= 2.0 * ex + 2e-7 && wm <= 1e-6, shape + " rows=" + std::to_string(rows) + " WMMA vs FP64 truth");
            }
        }
        if (reps) {
            const unsigned rows = 8192;
            double t[kArms][2];
            for (unsigned rep = 0; rep < 2; ++rep) for (unsigned o = 0; o < kArms; ++o) {
                const unsigned arm = rep ? kArms - 1U - o : o;  // ABCD DCBA
                env("STRIX_GLM_PF_EXACT_X", "4");
                arm_env(arm);
                DX::linear_rows(l, x, sh.K, y, sh.N, rows, ws, wsb, s);
                t[arm][rep] = median_ms([&] { DX::linear_rows(l, x, sh.K, y, sh.N, rows, ws, wsb, s); }, s, reps);
            }
            const double a = std::min(t[0][0], t[0][1]), b = std::min(t[1][0], t[1][1]), c8 = std::min(t[2][0], t[2][1]), c1 = std::min(t[3][0], t[3][1]), c2 = std::min(t[4][0], t[4][1]), c5 = std::min(t[5][0], t[5][1]);
            const double tf = 2.0 * rows * sh.K * sh.N / 1e9;
            std::printf("TIME %s rows=%u exact-X3 %.3f ms (%.2f TF) wmma-64 %.3f ms (%.2f TF) ratio %.3f wmma-default %.3f ms (%.2f TF) ratio %.3f wmma-tail64 %.3f ms ratio %.3f wmma-split1 %.3f ms ratio %.3f wmma-cp1 %.3f ms ratio %.3f\n",
                        shape.c_str(), rows, a, tf / a, b, tf / b, b / a, c8, tf / c8, c8 / a, c1, c1 / a, c2, c2 / a, c5, c5 / a);
        }
        arm_env(0);
        env("STRIX_GLM_PF_EXACT_X", nullptr);
    }
}

// --dense-rows: whole-call timing by rows per call (the chunked serving route issues 64..2048-row calls): the TM16
// exact core (the short-call route), the exact X3 core and the WMMA core, interleaved ABC CBA, minimum of the medians.
// Timing completion only: this mode does not compare outputs or qualify exactness/fidelity.
void dense_rows(std::mt19937& rng, hipStream_t s, unsigned reps) {
    struct BaselineTile {
        const char* inherited = strix::envcfg::text("STRIX_GLM_PF_EXACT_TM");
        const bool present = inherited != nullptr;
        const std::string saved = present ? inherited : "";
        BaselineTile() { env("STRIX_GLM_PF_EXACT_TM", "16"); }
        ~BaselineTile() { env("STRIX_GLM_PF_EXACT_TM", present ? saved.c_str() : nullptr); }
    } baseline_tile;
    struct Shape { unsigned K, N, bits; };
    for (const Shape sh : {Shape{4096, 1024, 4}, Shape{1024, 4096, 5}, Shape{4096, 6144, 6}, Shape{2048, 4096, 3}}) {
        const M::Exl3 w = make_matrix(rng, sh.K, sh.N, sh.bits);
        strix::glm::engine::Linear l;
        l.raw.trellis = w.trellis; l.raw.suh = w.suh; l.raw.svh = w.svh;
        l.raw.input = sh.K; l.raw.output = sh.N; l.raw.bits = sh.bits; l.raw.codebook = 2;
        constexpr unsigned Rmax = 4096;
        float* x = alloc<float>(std::size_t{Rmax} * sh.K);
        float* y = alloc<float>(std::size_t{Rmax + 1} * sh.N);
        const std::size_t wsb = DX::workspace_bytes(Rmax, sh.K);
        void* ws = alloc<unsigned char>(wsb);
        std::vector<float> hx(std::size_t{Rmax} * sh.K);
        std::normal_distribution<float> n01(0, 1);
        for (auto& v : hx) v = n01(rng);
        ok(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice), "x");
        for (unsigned rows : {32U, 64U, 128U, 256U, 512U, 1024U, 2048U, 4096U}) {
            double t[3][2];
            for (unsigned rep = 0; rep < 2; ++rep) for (unsigned o = 0; o < 3; ++o) {
                const unsigned arm = rep ? 2U - o : o;
                arm_env(arm == 2 ? 2U : 0U);
                env("STRIX_GLM_PF_EXACT_X", arm == 0 ? "0" : "4");
                if (arm == 0 && (DX::route(rows) != DX::Route::tm || DX::tile_rows() != 16))
                    throw std::runtime_error("dense-rows baseline must resolve to TM16");
                DX::linear_rows(l, x, sh.K, y, sh.N, rows, ws, wsb, s);
                t[arm][rep] = median_ms([&] { DX::linear_rows(l, x, sh.K, y, sh.N, rows, ws, wsb, s); }, s, reps);
            }
            const double a = std::min(t[0][0], t[0][1]), b = std::min(t[1][0], t[1][1]), c = std::min(t[2][0], t[2][1]);
            std::printf("ROWS dense K%u N%u b%u rows=%u tm%u %.3f ms exact-X3 %.3f ms (x%.3f) wmma %.3f ms (x%.3f of tm16, x%.3f of X3)\n",
                        sh.K, sh.N, sh.bits, rows, DX::tile_rows(), a, b, b / a, c, c / a, c / b);
        }
        arm_env(0);
        env("STRIX_GLM_PF_EXACT_X", nullptr);
    }
}

void moe(std::mt19937& rng, hipStream_t s, unsigned reps, bool classes_only) {
    constexpr unsigned MaxR = 4096;
    std::vector<M::ExpertSlices> experts(E);
    for (auto& e : experts) e = {make_matrix(rng, H, I, 4), make_matrix(rng, H, I, 4), make_matrix(rng, I, H, 4)};
    const M::ExpertSlices shared{make_matrix(rng, H, I, 6), make_matrix(rng, H, I, 6), make_matrix(rng, I, H, 6)};
    const std::size_t wsb = X::workspace_bytes(MaxR);
    void* ws = alloc<unsigned char>(wsb);
    float* x = alloc<float>(std::size_t{MaxR} * H);
    float* w = alloc<float>(std::size_t{MaxR} * Top);
    float* out = alloc<float>(std::size_t{MaxR + 1} * H);
    X::Plan plan(MaxR);
    // Device plan inputs: expert table [289] (288 = shared), device ids, invalid-id flag.
    auto* table = alloc<M::ExpertSlices>(E + 1);
    ok(hipMemcpy(table, experts.data(), sizeof(M::ExpertSlices) * E, hipMemcpyHostToDevice), "table");
    ok(hipMemcpy(table + E, &shared, sizeof(M::ExpertSlices), hipMemcpyHostToDevice), "table shared");
    auto* dids = alloc<std::uint32_t>(std::size_t{MaxR} * Top);
    auto* flag = alloc<unsigned>(1);
    // Mode 2 forced group sizes: the 64/128-row tile edges, the 65..127 remainder promotion, split-2 ownership, and one
    // hot expert spanning several split iterations (1100 rows; sizes that do not fit in R are skipped).
    const unsigned forced[] = {64U, 65U, 127U, 128U, 129U, 192U, 193U, 255U, 256U, 257U, 1100U};
    // Mode 3 (row classes): every expert holds exactly R / 36 rows, so every routed tile is the same ragged tile:
    // 33, 40, 48 rows (64-row tiles, MT 4: 3 of 4 row fragments live), 65, 80, 96, 112 rows (128-row tiles: 5..7 of 8).
    std::vector<std::pair<unsigned, unsigned>> cases;
    if (!classes_only) for (unsigned R : {128U, 333U, 2048U, 4096U}) for (unsigned mode : {0U, 1U, 2U}) cases.emplace_back(R, mode);
    for (unsigned c : {33U, 40U, 48U, 65U, 80U, 96U, 112U}) cases.emplace_back(c * 36U, 3U);
    for (const auto& [R, mode] : cases) {
        // mode 0 lognormal popularity, mode 1 group sizes 1..64 first (every tail bucket), mode 2 forced sizes, then
        // lognormal
        std::vector<double> pop(E); std::normal_distribution<double> nd(0.0, 1.2);
        for (auto& p : pop) p = std::exp(nd(rng));
        std::discrete_distribution<unsigned> dd(pop.begin(), pop.end());
        std::vector<std::vector<unsigned>> rows(R);
        if (mode == 1) { unsigned first = 0; for (unsigned c = 1; c <= 64 && first + c <= R; ++c) { for (unsigned r = first; r < first + c; ++r) rows[r].push_back(c - 1); first += c; } }
        if (mode == 3) for (unsigned r = 0; r < R; ++r) for (unsigned j = 0; j < Top; ++j) rows[r].push_back((r * Top + j) % E);
        if (mode == 2) {
            // Experts 0..10 get exactly the forced sizes (each row takes at most one of them; the rest of the row
            // draws from 64..287, so no other row lands in a forced group).
            if (R < 333) continue;
            for (unsigned e = 0, first = 0; e < std::size(forced); ++e) {
                if (first + forced[e] > R) continue;
                for (unsigned r = first; r < first + forced[e]; ++r) rows[r].push_back(e);
                first += forced[e];
            }
        }
        std::vector<std::uint32_t> ids; std::vector<float> wt;
        for (auto& row : rows) {
            while (row.size() < Top) { unsigned e = mode != 0 ? 64U + dd(rng) % (E - 64) : dd(rng); if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e); }
            std::sort(row.begin(), row.end());
            for (unsigned e : row) { ids.push_back(e); wt.push_back(0.02F + static_cast<float>(rng() % 1000) * 0.0005F); }
        }
        std::vector<float> hx(std::size_t{R} * H); std::normal_distribution<float> n01(0, 1);
        for (auto& v : hx) v = n01(rng);
        ok(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice), "x");
        ok(hipMemcpy(w, wt.data(), wt.size() * 4, hipMemcpyHostToDevice), "w");
        const X::View v{R, experts.data(), shared, x, ids.data(), wt.data(), w, out};
        const std::string name = "moe R=" + std::to_string(R) + " mode=" + std::to_string(mode);
        std::vector<float> res[kArms][3], res_dev[kArms];
        ok(hipMemcpy(dids, ids.data(), ids.size() * 4, hipMemcpyHostToDevice), "ids");
        const X::DeviceView dv{R, table, shared, x, dids, w, out};
        env("STRIX_GLM_PF_EXACTMOE_V", "11");
        for (unsigned arm = 0; arm < kArms; ++arm) {
            arm_env(arm);
            if (X::device_plan(R, 16)) {  // the device-built plan (production path) must give the host plan's bits
                ok(hipMemsetAsync(ws, 0xff, wsb, s), "poison"); ok(hipMemsetAsync(out, 0xff, std::size_t{MaxR + 1} * H * 4, s), "poison");
                ok(hipMemsetAsync(flag, 0, 4, s), "flag");
                X::enqueue_device(dv, plan, ws, wsb, s, flag); ok(hipStreamSynchronize(s), "run device");
                res_dev[arm].resize(std::size_t{R} * H); ok(hipMemcpy(res_dev[arm].data(), out, res_dev[arm].size() * 4, hipMemcpyDeviceToHost), "od");
                unsigned f = 1; ok(hipMemcpy(&f, flag, 4, hipMemcpyDeviceToHost), "flag");
                check(f == 0, name + " arm=" + std::to_string(arm) + " device plan ids valid");
                std::uint32_t dtail[64];
                ok(hipMemcpy(dtail, out + std::size_t{R} * H, sizeof dtail, hipMemcpyDeviceToHost), "dtail");
                check(std::all_of(dtail, dtail + 64, [](std::uint32_t t) { return t == 0xffffffffU; }), name + " arm=" + std::to_string(arm) + " device plan padding untouched");
            }
            ok(hipMemsetAsync(ws, 0xff, wsb, s), "poison"); ok(hipMemsetAsync(out, 0xff, std::size_t{MaxR + 1} * H * 4, s), "poison");
            X::enqueue(v, plan, ws, wsb, s, 16); ok(hipStreamSynchronize(s), "run");
            res[arm][0].resize(std::size_t{R} * H); ok(hipMemcpy(res[arm][0].data(), out, res[arm][0].size() * 4, hipMemcpyDeviceToHost), "o");
            res[arm][1].resize(std::size_t{R} * 9 * 2 * I); ok(hipMemcpy(res[arm][1].data(), X::pair_gate_up(ws, MaxR, 0), res[arm][1].size() * 4, hipMemcpyDeviceToHost), "gu");
            res[arm][2].resize(std::size_t{R} * 9 * H); ok(hipMemcpy(res[arm][2].data(), X::pair_output(ws, MaxR, 0), res[arm][2].size() * 4, hipMemcpyDeviceToHost), "po");
            std::uint32_t tail[64];
            ok(hipMemcpy(tail, out + std::size_t{R} * H, sizeof tail, hipMemcpyDeviceToHost), "tail");
            check(std::all_of(tail, tail + 64, [](std::uint32_t t) { return t == 0xffffffffU; }), name + " arm=" + std::to_string(arm) + " padding untouched");
        }
        gate(drift(res[0][1], res[1][1]), name + " gate/up");
        // Routed slots only: with the combine in the shared down's epilogue (the default) neither route writes a
        // pair-output row for the shared slot (slot 8 of each row keeps the poison), so that slot has nothing to gate.
        const auto routed = [&](const std::vector<float>& v) {
            std::vector<float> o;
            o.reserve(std::size_t{R} * 8 * H);
            for (unsigned r = 0; r < R; ++r) o.insert(o.end(), v.begin() + static_cast<std::ptrdiff_t>(std::size_t{r} * 9 * H), v.begin() + static_cast<std::ptrdiff_t>((std::size_t{r} * 9 + 8) * H));
            return o;
        };
        gate(drift(routed(res[0][2]), routed(res[1][2])), name + " pair-output", kF16L2, kF16Max);
        gate(drift(res[0][0], res[1][0]), name + " R16", kF16L2, kF16Max);
        // The pinned fused combine really ran in both routes: the shared slot still holds the poison.
        const auto shared_slot_unwritten = [&](const std::vector<float>& v) {
            for (unsigned r = 0; r < R; ++r)
                for (unsigned c = 0; c < H; ++c) {
                    std::uint32_t bits;
                    std::memcpy(&bits, &v[(std::size_t{r} * 9 + 8) * H + c], 4);
                    if (bits != 0xffffffffU) return false;
                }
            return true;
        };
        check(shared_slot_unwritten(res[0][2]) && shared_slot_unwritten(res[1][2]), name + " fused combine leaves the shared pair slot unwritten");
        // Separate combine (STRIX_GLM_PF_AUX_COMBINE=0): the shared expert writes its pair-output row, so all nine
        // slots of every row are gated, shared slot included, and the final output again.
        {
            std::vector<float> po[2], o16[2];
            for (unsigned arm = 0; arm < 2; ++arm) {
                arm_env(arm);
                env("STRIX_GLM_PF_AUX_COMBINE", "0");
                ok(hipMemsetAsync(ws, 0xff, wsb, s), "poison"); ok(hipMemsetAsync(out, 0xff, std::size_t{MaxR + 1} * H * 4, s), "poison");
                X::enqueue(v, plan, ws, wsb, s, 16); ok(hipStreamSynchronize(s), "run unfused");
                o16[arm].resize(std::size_t{R} * H); ok(hipMemcpy(o16[arm].data(), out, o16[arm].size() * 4, hipMemcpyDeviceToHost), "o");
                po[arm].resize(std::size_t{R} * 9 * H); ok(hipMemcpy(po[arm].data(), X::pair_output(ws, MaxR, 0), po[arm].size() * 4, hipMemcpyDeviceToHost), "po");
            }
            env("STRIX_GLM_PF_AUX_COMBINE", "1");
            check(!shared_slot_unwritten(po[0]) && !shared_slot_unwritten(po[1]), name + " separate combine writes the shared pair slot");
            gate(drift(po[0], po[1]), name + " pair-output separate combine (9 slots)", kF16L2, kF16Max);
            gate(drift(o16[0], o16[1]), name + " R16 separate combine", kF16L2, kF16Max);
        }
        for (unsigned arm = 0; arm < kArms; ++arm)
            if (!res_dev[arm].empty()) same_bits(res[arm][0], res_dev[arm], name + " arm " + std::to_string(arm) + " device plan");
        for (unsigned arm = 2; arm < kArms; ++arm)
            for (unsigned b = 0; b < 3; ++b) same_bits(res[1][b], res[arm][b], name + " arm " + std::to_string(arm) + " buffer " + std::to_string(b));
        if (reps && (R >= 2048 || mode == 3)) {
            double t[kArms][2];
            for (unsigned rep = 0; rep < 2; ++rep) for (unsigned o = 0; o < kArms; ++o) {
                const unsigned arm = rep ? kArms - 1U - o : o;  // ABCD DCBA
                arm_env(arm);
                X::enqueue(v, plan, ws, wsb, s, 16);
                t[arm][rep] = median_ms([&] { X::enqueue(v, plan, ws, wsb, s, 16); }, s, reps);
            }
            const double a = std::min(t[0][0], t[0][1]), b = std::min(t[1][0], t[1][1]), c8 = std::min(t[2][0], t[2][1]), c1 = std::min(t[3][0], t[3][1]), c2 = std::min(t[4][0], t[4][1]), c5 = std::min(t[5][0], t[5][1]);
            const double tf = static_cast<double>(R) * 9.0 * 6.0 * H * I / 1e9;
            std::printf("TIME %s whole-call V11 %.3f ms (%.2f TF) wmma-64 %.3f ms (%.2f TF) ratio %.3f wmma-default %.3f ms (%.2f TF) ratio %.3f wmma-tail64 %.3f ms ratio %.3f wmma-split1 %.3f ms ratio %.3f wmma-cp1 %.3f ms ratio %.3f\n",
                        name.c_str(), a, tf / a, b, tf / b, b / a, c8, tf / c8, c8 / a, c1, c1 / a, c2, c2 / a, c5, c5 / a);
            const double c6 = std::min(t[6][0], t[6][1]);
            std::printf("TIME %s rows/expert %u live-fragments %.3f ms all-fragments %.3f ms ratio %.4f (pairs %.4f %.4f)\n", name.c_str(),
                        mode == 3 ? R / 36U : 0U, c8, c6, c8 / c6, t[2][0] / t[6][0], t[2][1] / t[6][1]);
        }
        arm_env(0);
        env("STRIX_GLM_PF_EXACTMOE_V", nullptr);
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        unsigned reps = 7;
        bool do_dense = true, do_moe = true, classes = false, rows_only = false;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--no-timing") == 0) reps = 0;
            else if (std::strcmp(argv[i], "--dense") == 0) do_moe = false;
            else if (std::strcmp(argv[i], "--moe") == 0) do_dense = false;
            else if (std::strcmp(argv[i], "--moe-classes") == 0) { do_dense = false; classes = true; }
            else if (std::strcmp(argv[i], "--dense-rows") == 0) { do_dense = false; do_moe = false; rows_only = true; }
            else throw std::invalid_argument("usage: strix-hip-glm5-pf-wmma-tests [--no-timing] [--dense | --moe | --moe-classes | --dense-rows]");
        }
        std::mt19937 rng(0x77a11ee5U);
        hipStream_t s; ok(hipStreamCreate(&s), "stream");
        if (rows_only) dense_rows(rng, s, reps ? reps : 7U);
        if (do_dense) dense(rng, s, reps);
        if (do_moe) moe(rng, s, reps, classes);
        ok(hipStreamDestroy(s), "stream");
    } catch (const std::exception& e) {
        check(false, e.what());
    }
    for (void* p : allocs) (void)hipFree(p);
    std::printf("checks: %u failures: %u\n", checks, failures);
    return failures ? 1 : 0;
}
