// FAST_I4 paired chain on the GLM prefill MoE call: plan policy fixture (shared moe_i4::chain_plan / launch_chain).
// Arms (same integers, so same bits everywhere they are compared):
//   0 gate scratch lent and admitted: paired gate/up and down on the call's plan, one tile list
//   1 default policy: register save, paired at most plan 1, second tile list for a plan-2 down
//   2 STRIX_GLM_PF_IU4_PLAN=1 with the scratch: both GEMMs on BM64 tiles
//   3 the scratch on offer but the shared switch off: the register save is kept
//   4 STRIX_GLM_PF_IU4_PLAN=0 with the scratch: both GEMMs on BM32 tiles
// The gate-scratch policy of each arm is explicit (iu4::set_chain_policy_for_test: production resolves it once per
// process), and every arm asserts the plan the call was enqueued on (iu4::last_chain_run), so a resource fallback
// to the register save fails the fixture instead of passing the bit comparisons on another route. Arm 1 enqueues
// the same launches as that fallback (paired plan 1 without the scratch, then the down on its own tile list).
// Compared: the routed down's A4 image and scales, the final tile list (arms 0 and 1), the ordered combine output.
// Row classes: every routed expert holds exactly c rows (c * 36 rows of 8 ascending experts out of 288).
// --lognormal R: R rows routed by a lognormal expert popularity (sigma 1.2), the skew of a real prefill chunk.
#include "core/moe_i4_chain.hpp"
#include "core/wmma_tiled.hpp"
#include "glm/prefill/glm_pf_exact_moe.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace M = strix::glm::moe;
namespace X = strix::glm::pf::exact_moe;
namespace {
constexpr unsigned H = 4096, I = 1024, E = 288, Top = 8, P = 9, kArms = 5;
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
std::uint64_t fnv(const void* p, std::size_t n) {
    std::uint64_t h = 14695981039346656037ULL;
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}
template<class T> std::uint64_t device_hash(const T* p, std::size_t n) {
    std::vector<T> host(n);
    ok(hipMemcpy(host.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "download");
    return fnv(host.data(), n * sizeof(T));
}
void arm_env(unsigned arm) {
    env("STRIX_GLM_PF_WMMA", "1");
    env("STRIX_GLM_PF_EXACTMOE_V", "11");
    env("STRIX_MOE_I4_CHAIN", "1");
    env("STRIX_GLM_PF_IU4_PLAN", arm == 2 ? "1" : arm == 4 ? "0" : nullptr);
    const X::iu4::ChainPolicy policy{arm != 3, arm != 1};  // shared switch, scratch lent
    X::iu4::set_chain_policy_for_test(&policy);
}
// Plan the chain must have been enqueued on: requested = the override of arms 2 / 4, else plan 2 from
// kPlan2Rows rows per routed expert (rows * 8 pairs over 288 experts).
struct Expected { int paired, down; bool gsave; };
Expected expected_run(unsigned arm, unsigned rows) {
    const int requested = arm == 2 ? 1 : arm == 4 ? 0 : rows * 8U >= strix::core::moe_i4::kPlan2Rows * E ? 2 : 1;
    const bool scratch = arm != 1 && arm != 3;
    return {scratch || requested < 1 ? requested : 1, requested, scratch};
}
template<class F> double median_ms(F&& f, hipStream_t s, unsigned reps) {
    std::vector<double> t;
    for (unsigned i = 0; i < reps; ++i) {
        ok(hipStreamSynchronize(s), "sync");
        const auto t0 = std::chrono::steady_clock::now();
        f(); ok(hipStreamSynchronize(s), "sync");
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}
void chain(std::mt19937& rng, hipStream_t s, unsigned reps, std::vector<unsigned> classes, const std::vector<unsigned>& skewed) {
    constexpr unsigned MaxR = 256U * 36U;
    std::vector<M::ExpertSlices> experts(E);
    for (auto& e : experts) e = {make_matrix(rng, H, I, 4), make_matrix(rng, H, I, 4), make_matrix(rng, I, H, 4)};
    const M::ExpertSlices shared{make_matrix(rng, H, I, 6), make_matrix(rng, H, I, 6), make_matrix(rng, I, H, 6)};
    // Stored INT4 routed experts of the same layer (the load-time conversion).
    X::iu4::Layer q;
    for (unsigned m = 0; m < 3; ++m) {
        q.wq[m] = alloc<std::uint8_t>(X::iu4::weight_bytes());
        q.ws[m] = reinterpret_cast<std::uint16_t*>(alloc<std::uint8_t>(X::iu4::scale_bytes(m)));
        q.svh[m] = alloc<const std::uint16_t*>(E);
    }
    X::iu4::convert_into(experts.data(), q, s);
    X::iu4::Scratch scratch = X::iu4::make_scratch(MaxR);
    const std::size_t wsb = X::workspace_bytes(MaxR);
    void* ws = alloc<unsigned char>(wsb);
    float* x = alloc<float>(std::size_t{MaxR} * H);
    float* w = alloc<float>(std::size_t{MaxR} * Top);
    float* out = alloc<float>(std::size_t{MaxR + 1} * H);
    X::Plan plan(MaxR);
    auto* table = alloc<M::ExpertSlices>(E + 1);
    ok(hipMemcpy(table, experts.data(), sizeof(M::ExpertSlices) * E, hipMemcpyHostToDevice), "table");
    ok(hipMemcpy(table + E, &shared, sizeof(M::ExpertSlices), hipMemcpyHostToDevice), "table shared");
    auto* dids = alloc<std::uint32_t>(std::size_t{MaxR} * Top);
    auto* flag = alloc<unsigned>(1);
    if (classes.empty() && skewed.empty()) classes = {31U, 63U, 64U, 65U, 96U, 112U, 128U, 129U};  // BM64 / BM128 tile edges, 96..128 = the plan-2 class
    struct Case { unsigned rows; bool skew; };
    std::vector<Case> cases;
    for (unsigned c : classes) {
        if (c == 0 || c > 256U) throw std::invalid_argument("--rows takes 1..256 rows per expert");
        cases.push_back({c * 36U, false});
    }
    for (unsigned r : skewed) {
        if (r == 0 || r > MaxR) throw std::invalid_argument("--lognormal takes 1..9216 rows");
        cases.push_back({r, true});
    }
    for (const Case& cs : cases) {
        const unsigned R = cs.rows, c = cs.skew ? 0U : R / 36U;
        std::vector<double> pop(E); std::normal_distribution<double> nd(0.0, 1.2);
        for (auto& p : pop) p = std::exp(nd(rng));
        std::discrete_distribution<unsigned> dd(pop.begin(), pop.end());
        std::vector<std::uint32_t> ids; std::vector<float> wt;
        for (unsigned r = 0; r < R; ++r) {
            std::uint32_t row[Top];
            for (unsigned j = 0; j < Top; ++j) {
                if (!cs.skew) { row[j] = (r * Top + j) % E; continue; }
                do row[j] = dd(rng); while (std::find(row, row + j, row[j]) != row + j);
            }
            std::sort(row, row + Top);
            for (unsigned e : row) { ids.push_back(e); wt.push_back(0.02F + static_cast<float>(rng() % 1000) * 0.0005F); }
        }
        std::vector<float> hx(std::size_t{R} * H); std::normal_distribution<float> n01(0, 1);
        for (auto& v : hx) v = n01(rng);
        ok(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice), "x");
        ok(hipMemcpy(w, wt.data(), wt.size() * 4, hipMemcpyHostToDevice), "w");
        ok(hipMemcpy(dids, ids.data(), ids.size() * 4, hipMemcpyHostToDevice), "ids");
        X::DeviceView dv{R, table, shared, x, dids, w, out};
        dv.iu4 = &q; dv.iu4_scratch = &scratch;
        const std::string name = "iu4 chain " + (cs.skew ? std::string("lognormal") : "rows/expert=" + std::to_string(c)) + " R=" + std::to_string(R);
        const std::size_t slots = std::size_t{R} * P;
        std::uint64_t h_out[kArms], h_q[kArms], h_s[kArms], h_tiles[kArms]; std::uint32_t nt[kArms];
        for (unsigned arm = 0; arm < kArms; ++arm) {
            arm_env(arm);
            ok(hipMemsetAsync(ws, 0xff, wsb, s), "poison"); ok(hipMemsetAsync(out, 0xff, std::size_t{MaxR + 1} * H * 4, s), "poison");
            ok(hipMemsetAsync(scratch.aq_down, 0xff, slots * (I / 2), s), "poison"); ok(hipMemsetAsync(scratch.as_down, 0xff, slots * (I / 64) * 4, s), "poison");
            ok(hipMemsetAsync(flag, 0, 4, s), "flag");
            X::enqueue_device(dv, plan, ws, wsb, s, flag); ok(hipStreamSynchronize(s), "run");
            unsigned f = 1; ok(hipMemcpy(&f, flag, 4, hipMemcpyDeviceToHost), "flag");
            check(f == 0, name + " arm " + std::to_string(arm) + " ids valid");
            const auto ran = X::iu4::last_chain_run(); const auto want = expected_run(arm, R);
            check(ran.valid && ran.paired == want.paired && ran.down == want.down && ran.gsave == want.gsave,
                  name + " arm " + std::to_string(arm) + " effective plan paired " + std::to_string(ran.paired) + " down " + std::to_string(ran.down) +
                      " gsave " + std::to_string(ran.gsave) + " (want " + std::to_string(want.paired) + " " + std::to_string(want.down) + " " + std::to_string(want.gsave) + ")");
            h_out[arm] = device_hash(out, std::size_t{R} * H);
            h_q[arm] = device_hash(scratch.aq_down, slots * (I / 2));
            h_s[arm] = device_hash(scratch.as_down, slots * (I / 64));
            ok(hipMemcpy(&nt[arm], scratch.ntiles, 4, hipMemcpyDeviceToHost), "ntiles");
            check(nt[arm] != 0 && nt[arm] <= scratch.max_tiles, name + " arm " + std::to_string(arm) + " tile count");
            h_tiles[arm] = nt[arm] <= scratch.max_tiles ? device_hash(static_cast<const strix::core::IU4GroupTile*>(scratch.tiles), nt[arm]) : 0;
            std::uint32_t tail[64];
            ok(hipMemcpy(tail, out + std::size_t{R} * H, sizeof tail, hipMemcpyDeviceToHost), "tail");
            check(std::all_of(tail, tail + 64, [](std::uint32_t t) { return t == 0xffffffffU; }), name + " arm " + std::to_string(arm) + " padding untouched");
        }
        std::printf("HASH %s out %016llx down-image %016llx down-scales %016llx tiles %u %016llx\n", name.c_str(),
                    static_cast<unsigned long long>(h_out[0]), static_cast<unsigned long long>(h_q[0]), static_cast<unsigned long long>(h_s[0]), nt[0],
                    static_cast<unsigned long long>(h_tiles[0]));
        for (unsigned arm = 1; arm < kArms; ++arm) {
            check(h_out[arm] == h_out[0], name + " arm " + std::to_string(arm) + " combine output");
            check(h_q[arm] == h_q[0], name + " arm " + std::to_string(arm) + " down A4 image");
            check(h_s[arm] == h_s[0], name + " arm " + std::to_string(arm) + " down scales");
        }
        check(nt[1] == nt[0] && h_tiles[1] == h_tiles[0], name + " final tile list (register save)");
        check(nt[3] == nt[0] && h_tiles[3] == h_tiles[0], name + " final tile list (shared switch)");
        if (reps) {
            double t[kArms][2];
            for (unsigned rep = 0; rep < 2; ++rep) for (unsigned o = 0; o < 3U; ++o) {
                const unsigned arm = rep ? 2U - o : o;  // ABC CBA
                arm_env(arm);
                X::enqueue_device(dv, plan, ws, wsb, s, flag);
                t[arm][rep] = median_ms([&] { X::enqueue_device(dv, plan, ws, wsb, s, flag); }, s, reps);
            }
            const double a = std::min(t[0][0], t[0][1]), b = std::min(t[1][0], t[1][1]), p1 = std::min(t[2][0], t[2][1]);
            std::printf("TIME %s whole-call shared-plan %.3f ms register-save %.3f ms ratio %.4f (passes %.4f %.4f) plan1+scratch %.3f ms ratio %.4f\n",
                        name.c_str(), a, b, a / b, t[0][0] / t[1][0], t[0][1] / t[1][1], p1, p1 / b);
        }
    }
    X::iu4::set_chain_policy_for_test(nullptr);
    X::iu4::release(scratch);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        unsigned reps = 7;
        std::vector<unsigned> classes, skewed;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--no-timing") == 0) reps = 0;
            else if (std::strcmp(argv[i], "--rows") == 0 && i + 1 < argc) classes.push_back(static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10)));
            else if (std::strcmp(argv[i], "--lognormal") == 0 && i + 1 < argc) skewed.push_back(static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10)));
            else throw std::invalid_argument("usage: strix-hip-glm5-pf-iu4-chain-tests [--no-timing] [--rows N]... [--lognormal R]...");
        }
        if (!X::iu4::built()) { std::printf("SKIP: the IU4 core is not linked\n"); return 0; }
        std::mt19937 rng(0x1c4a11eeU);
        hipStream_t s; ok(hipStreamCreate(&s), "stream");
        chain(rng, s, reps, classes, skewed);
        ok(hipStreamDestroy(s), "stream");
    } catch (const std::exception& e) {
        check(false, e.what());
    }
    for (void* p : allocs) (void)hipFree(p);
    std::printf("checks: %u failures: %u\n", checks, failures);
    return failures ? 1 : 0;
}
