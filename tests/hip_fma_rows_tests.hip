// Shared exact FMA-rows core (src/core/fma_rows.hpp) against the GLM dense_rows kernels (version 3 is the reference
// arm): every output byte equal (memcmp against the retained reference bytes), untouched guard rows, then interleaved
// whole-call timing at the GLM KDA / indexer / router tile sizes and at the MiniMax index-projection size.
// STRIX_FMA_ROWS_BENCH=<rounds> sets the timing rounds (0 = gate only).
// The reference arm itself is checked against a host oracle (the same ascending fmaf chain from +0) on sampled rows,
// so two arms that leave the same output unwritten cannot pass. Gate-only shapes also cover a single 32-input stage,
// 127 / 128 / 129 stages (4064 / 4096 / 4128 inputs), the 4095 / 4096 / 4097-row selection boundary and the resident
// block boundaries, with the form launch_auto returns asserted.
#include "core/fma_rows.hpp"
#include "glm/prefill/glm_prefill.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace P = strix::glm::pf;
namespace D = strix::glm::dev;
namespace F = strix::core::fma_rows;
namespace {
unsigned checks = 0, failures = 0;
void check(bool c, const std::string& what) {
    ++checks;
    if (!c) { ++failures; std::printf("FAIL %s\n", what.c_str()); }
}
void ok(hipError_t e, const char* w) { if (e != hipSuccess) throw std::runtime_error(std::string(w) + ": " + hipGetErrorString(e)); }
template<class T> T* alloc(std::size_t n) { T* p; ok(hipMalloc(&p, n * sizeof(T)), "alloc"); return p; }
constexpr float kGuard = -777.25F;
constexpr unsigned kGuardRows = 2;
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
std::uint64_t fnv(const void* p, std::size_t n) {
    std::uint64_t h = 1469598103934665603ULL;
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}

struct Arm {
    std::string name;
    std::function<void(float* const* y)> run;
};

struct Shape {
    const char* name;
    unsigned in, rows, count, n[3];
    D::WType wt;
    unsigned o0;
    F::Form want = F::Form::Auto;  // form launch_auto must return for this shape (Auto: not asserted)
};

template <unsigned RT, unsigned C, unsigned NP, unsigned KS, bool DB>
Arm core_arm(const char* name, const float* dx, const Shape& sh, void* const* dw) {
    return {name, [=](float* const* y) {
                F::Group g{};
                g.count = sh.count;
                for (unsigned p = 0; p < sh.count; ++p) { g.w[p] = dw[p]; g.y[p] = y[p]; g.o0[p] = sh.o0; g.n[p] = sh.n[p]; }
                F::launch<RT, C, NP, KS, DB>(dx, static_cast<F::Weight>(static_cast<unsigned>(sh.wt)), sh.in, sh.rows, g, nullptr);
            }};
}

template <unsigned RT, unsigned C, unsigned KX, unsigned CB, unsigned KS>
Arm xres_arm(const char* name, const float* dx, const Shape& sh, void* const* dw) {
    return {name, [=](float* const* y) {
                F::Group g{};
                g.count = sh.count;
                for (unsigned p = 0; p < sh.count; ++p) { g.w[p] = dw[p]; g.y[p] = y[p]; g.o0[p] = sh.o0; g.n[p] = sh.n[p]; }
                F::launch_xres<RT, C, KX, CB, KS>(dx, static_cast<F::Weight>(static_cast<unsigned>(sh.wt)), sh.in, sh.rows, g, nullptr);
            }};
}

void run_shape(const Shape& sh, unsigned rounds) {
    std::mt19937 rng(1234U + sh.in + sh.rows);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> hx(static_cast<std::size_t>(sh.rows) * sh.in);
    for (auto& v : hx) v = dist(rng);
    float* dx = alloc<float>(hx.size());
    ok(hipMemcpy(dx, hx.data(), hx.size() * sizeof(float), hipMemcpyHostToDevice), "x");
    void* dw[3]{};
    std::vector<float> hw[3];  // the weights as the kernels widen them (exact)
    std::size_t ybytes[3]{}, total = 0;
    double macs = 0.0;
    for (unsigned p = 0; p < sh.count; ++p) {
        const std::size_t wn = static_cast<std::size_t>(sh.o0 + sh.n[p]) * sh.in;
        if (sh.wt == D::WType::F32) {
            std::vector<float> w(wn);
            for (auto& v : w) v = dist(rng) * 0.05F;
            hw[p] = w;
            dw[p] = alloc<float>(wn);
            ok(hipMemcpy(dw[p], w.data(), wn * sizeof(float), hipMemcpyHostToDevice), "w");
        } else if (sh.wt == D::WType::F16) {
            std::vector<_Float16> w(wn);
            for (auto& v : w) v = static_cast<_Float16>(dist(rng) * 0.05F);
            hw[p].resize(wn);
            for (std::size_t i = 0; i < wn; ++i) hw[p][i] = static_cast<float>(w[i]);
            dw[p] = alloc<_Float16>(wn);
            ok(hipMemcpy(dw[p], w.data(), wn * 2, hipMemcpyHostToDevice), "w");
        } else {
            std::vector<std::uint16_t> w(wn);
            for (auto& v : w) v = static_cast<std::uint16_t>(__builtin_bit_cast(std::uint32_t, dist(rng) * 0.05F) >> 16U);
            hw[p].resize(wn);
            for (std::size_t i = 0; i < wn; ++i) hw[p][i] = __builtin_bit_cast(float, static_cast<std::uint32_t>(w[i]) << 16U);
            dw[p] = alloc<std::uint16_t>(wn);
            ok(hipMemcpy(dw[p], w.data(), wn * 2, hipMemcpyHostToDevice), "w");
        }
        ybytes[p] = static_cast<std::size_t>(sh.rows + kGuardRows) * sh.n[p] * sizeof(float);
        total += ybytes[p];
        macs += static_cast<double>(sh.rows) * sh.n[p] * sh.in;
    }
    const bool glm_shape = sh.in % 32 == 0;
    std::vector<Arm> arms;
    arms.push_back({"glm-v3", [=](float* const* y) {
                        P::DenseProj pr[3];
                        for (unsigned p = 0; p < sh.count; ++p) pr[p] = {dw[p], sh.o0, sh.n[p], y[p]};
                        if (sh.count > 1) P::dense_rows_group_versioned(dx, sh.wt, sh.in, pr, sh.count, sh.rows, nullptr, 3);
                        else P::dense_rows_versioned(dx, dw[0], sh.wt, sh.in, sh.o0, sh.n[0], y[0], sh.rows, nullptr, 3);
                    }});
    arms.push_back(core_arm<128, 64, 1, 32, true>("core 128x64 db", dx, sh, dw));
    arms.push_back(core_arm<128, 64, 1, 32, false>("core 128x64 sb", dx, sh, dw));
    arms.push_back(core_arm<128, 64, 1, 16, true>("core 128x64 k16 db", dx, sh, dw));
    arms.push_back(core_arm<64, 64, 1, 32, true>("core 64x64 db", dx, sh, dw));
    arms.push_back(core_arm<128, 32, 1, 32, true>("core 128x32 db", dx, sh, dw));
    if (sh.count > 1) {
        arms.push_back(core_arm<64, 128, 3, 32, true>("core 64x128 np3 db", dx, sh, dw));
        arms.push_back(core_arm<64, 128, 3, 32, false>("core 64x128 np3 sb", dx, sh, dw));
    }
    if (sh.in <= 128 && sh.in % 32 == 0) {
        arms.push_back(xres_arm<64, 64, 128, 4, 32>("xres 64x64 cb4", dx, sh, dw));
        arms.push_back(xres_arm<64, 64, 128, 8, 32>("xres 64x64 cb8", dx, sh, dw));
        arms.push_back(xres_arm<64, 64, 128, 16, 32>("xres 64x64 cb16", dx, sh, dw));
        arms.push_back(xres_arm<64, 64, 128, 8, 16>("xres 64x64 cb8 k16", dx, sh, dw));
        arms.push_back(xres_arm<32, 64, 128, 8, 32>("xres 32x64 cb8", dx, sh, dw));
    }
    for (unsigned f = 1; f < F::kFormCount; ++f) {
        const auto form = static_cast<F::Form>(f);
        if (!F::form_admits(form, sh.in, sh.rows, sh.count)) continue;
        static const char* const names[] = {"", "form pipe128", "form pipe64", "form np3", "form xres16", "form xres8", "form xres4", "form pipe128 k16"};
        arms.push_back({names[f], [=](float* const* y) {
                            F::Group g{};
                            g.count = sh.count;
                            for (unsigned p = 0; p < sh.count; ++p) { g.w[p] = dw[p]; g.y[p] = y[p]; g.o0[p] = sh.o0; g.n[p] = sh.n[p]; }
                            F::launch_auto(dx, static_cast<F::Weight>(static_cast<unsigned>(sh.wt)), sh.in, sh.rows, g, nullptr, form);
                        }});
    }
    {
        F::Group g{};
        g.count = sh.count;
        for (unsigned p = 0; p < sh.count; ++p) g.n[p] = sh.n[p];
        if (F::admits(sh.in, sh.rows, 32)) std::printf("  auto form for %s: %u\n", sh.name, static_cast<unsigned>(F::select(sh.in, sh.rows, g)));
    }
    // Automatic selection: the returned form is the contract (and, where the shape names one, the expected form).
    F::Form auto_form = F::Form::Auto;
    F::Form* const auto_out = &auto_form;
    if (F::admits(sh.in, sh.rows, 32))
        arms.push_back({"form auto", [=](float* const* y) {
                            F::Group g{};
                            g.count = sh.count;
                            for (unsigned p = 0; p < sh.count; ++p) { g.w[p] = dw[p]; g.y[p] = y[p]; g.o0[p] = sh.o0; g.n[p] = sh.n[p]; }
                            *auto_out = F::launch_auto(dx, static_cast<F::Weight>(static_cast<unsigned>(sh.wt)), sh.in, sh.rows, g, nullptr);
                        }});
    // Version 4 of the GLM adapter: the core for chains of <= 4096 inputs, the version 3 route above that (gate only).
    if (glm_shape && (sh.in <= 4096 || rounds == 0))
        arms.push_back({"glm-v4", [=](float* const* y) {
                            P::DenseProj pr[3];
                            for (unsigned p = 0; p < sh.count; ++p) pr[p] = {dw[p], sh.o0, sh.n[p], y[p]};
                            if (sh.count > 1) P::dense_rows_group_versioned(dx, sh.wt, sh.in, pr, sh.count, sh.rows, nullptr, 4);
                            else P::dense_rows_versioned(dx, dw[0], sh.wt, sh.in, sh.o0, sh.n[0], y[0], sh.rows, nullptr, 4);
                        }});

    float* dy[3]{};
    for (unsigned p = 0; p < sh.count; ++p) dy[p] = alloc<float>(ybytes[p] / sizeof(float));
    std::vector<unsigned char> host(total), ref;
    std::vector<std::uint64_t> hash(arms.size());
    std::vector<char> same(arms.size(), 1);
    std::vector<std::vector<double>> ms(arms.size());
    auto fill_guard = [&] {
        for (unsigned p = 0; p < sh.count; ++p) {
            std::vector<float> gv(ybytes[p] / sizeof(float), kGuard);
            ok(hipMemcpy(dy[p], gv.data(), ybytes[p], hipMemcpyHostToDevice), "guard");
        }
    };
    for (std::size_t a = 0; a < arms.size(); ++a) {
        fill_guard();
        arms[a].run(dy);
        ok(hipDeviceSynchronize(), "sync");
        ok(hipGetLastError(), arms[a].name.c_str());
        std::size_t at = 0;
        bool guard_ok = true;
        for (unsigned p = 0; p < sh.count; ++p) {
            ok(hipMemcpy(host.data() + at, dy[p], ybytes[p], hipMemcpyDeviceToHost), "read");
            const auto* f = reinterpret_cast<const float*>(host.data() + at);
            for (std::size_t i = static_cast<std::size_t>(sh.rows) * sh.n[p]; i < ybytes[p] / sizeof(float); ++i) guard_ok = guard_ok && f[i] == kGuard;
            at += ybytes[p];
        }
        hash[a] = fnv(host.data(), total);
        check(guard_ok, std::string(sh.name) + " " + arms[a].name + " guard rows untouched");
        if (a == 0) {
            // Reference arm: host oracle on sampled rows (every row of a small shape), every output of those rows.
            ref = host;
            const unsigned step = static_cast<double>(sh.rows) * sh.in <= 262144.0 ? 1U : std::max(1U, sh.rows / 5U);
            bool oracle_ok = true;
            std::size_t off = 0;
            for (unsigned p = 0; p < sh.count; ++p) {
                const auto* f = reinterpret_cast<const float*>(ref.data() + off);
                for (unsigned r = 0; r < sh.rows; r = (r + step < sh.rows || r == sh.rows - 1) ? r + step : sh.rows - 1)
                    for (unsigned o = 0; o < sh.n[p]; ++o) {
                        float acc = 0.0F;
                        const float* xr = hx.data() + static_cast<std::size_t>(r) * sh.in;
                        const float* wr = hw[p].data() + static_cast<std::size_t>(sh.o0 + o) * sh.in;
                        for (unsigned k = 0; k < sh.in; ++k) acc = std::fmaf(xr[k], wr[k], acc);
                        oracle_ok = oracle_ok && std::memcmp(&acc, &f[static_cast<std::size_t>(r) * sh.n[p] + o], sizeof(float)) == 0;
                    }
                off += ybytes[p];
            }
            check(oracle_ok, std::string(sh.name) + " glm-v3 bytes equal to the host fmaf oracle");
        } else {
            same[a] = std::memcmp(host.data(), ref.data(), total) == 0;
            check(same[a] != 0, std::string(sh.name) + " " + arms[a].name + " bytes equal to glm-v3");
        }
    }
    if (F::admits(sh.in, sh.rows, 32)) {
        F::Group g{};
        g.count = sh.count;
        for (unsigned p = 0; p < sh.count; ++p) g.n[p] = sh.n[p];
        check(auto_form == F::select(sh.in, sh.rows, g) && auto_form != F::Form::Auto, std::string(sh.name) + " launch_auto returns select()");
        if (sh.want != F::Form::Auto)
            check(auto_form == sh.want, std::string(sh.name) + " automatic form " + std::to_string(static_cast<unsigned>(auto_form)) +
                                            " (want " + std::to_string(static_cast<unsigned>(sh.want)) + ")");
    }
    for (unsigned r = 0; r < rounds + 2; ++r)
        for (std::size_t a = 0; a < arms.size(); ++a) {
            const auto t0 = std::chrono::steady_clock::now();
            arms[a].run(dy);
            ok(hipDeviceSynchronize(), "sync");
            if (r >= 2) ms[a].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
    std::printf("shape %s: in %u rows %u n %u+%u+%u hash %016llx\n", sh.name, sh.in, sh.rows, sh.n[0], sh.count > 1 ? sh.n[1] : 0,
                sh.count > 2 ? sh.n[2] : 0, static_cast<unsigned long long>(hash[0]));
    if (rounds)
        for (std::size_t a = 0; a < arms.size(); ++a) {
            const double m = median(ms[a]), base = median(ms[0]);
            std::printf("  %-20s %8.3f ms  %6.2f TMAC/s  x%.3f  %s\n", arms[a].name.c_str(), m, macs / m * 1e-9, m / base,
                        same[a] ? "same" : "DIFF");
        }
    for (unsigned p = 0; p < sh.count; ++p) { ok(hipFree(dy[p]), "free"); ok(hipFree(dw[p]), "free"); }
    ok(hipFree(dx), "free");
}
}  // namespace

int main() {
    try {
        const char* e = std::getenv("STRIX_FMA_ROWS_BENCH");
        const unsigned rounds = e ? static_cast<unsigned>(std::strtoul(e, nullptr, 10)) : 0U;
        const Shape gate[] = {
            {"gate group ragged", 256, 77, 3, {128, 96, 40}, D::WType::F16, 0},
            {"gate single f32", 128, 131, 1, {200, 0, 0}, D::WType::F32, 0},
            {"gate single bf16", 96, 9, 1, {33, 0, 0}, D::WType::BF16, 0},
            {"gate pair f16", 4096, 300, 2, {64, 24, 0}, D::WType::F16, 0},
            {"gate offset rows", 64, 200, 2, {130, 70, 0}, D::WType::F16, 37},
            {"gate offset f32", 4096, 70, 1, {96, 0, 0}, D::WType::F32, 5},
            // One 32-input stage (no look-ahead load after it), short and resident forms.
            {"gate one stage", 32, 300, 1, {200, 0, 0}, D::WType::F16, 0, F::Form::Pipe128K16},
            {"gate one stage group", 32, 4096, 3, {128, 128, 64}, D::WType::F16, 0, F::Form::Xres4},
            {"gate one stage bf16", 32, 70, 2, {96, 33, 0}, D::WType::BF16, 3, F::Form::Pipe128K16},
            // 127 / 128 / 129 stages: 4064 and 4096 inputs run the core under GLM version 4, 4128 keeps the version 3 route.
            {"gate 127 stages", 4064, 300, 2, {64, 24, 0}, D::WType::F16, 0, F::Form::Pipe64},
            {"gate 129 stages", 4128, 300, 2, {64, 24, 0}, D::WType::F16, 0, F::Form::Pipe64},
            {"gate 127 stages long", 4064, 4096, 3, {128, 128, 64}, D::WType::F16, 0, F::Form::PipeNp3},
            {"gate 129 stages long", 4128, 4096, 1, {96, 0, 0}, D::WType::F16, 0, F::Form::Pipe64},
            // Row selection boundary (kLongRows = 4096): long chains, grouped and single.
            {"gate group 4095 rows", 4096, 4095, 3, {128, 128, 64}, D::WType::F16, 0, F::Form::Pipe128},
            {"gate group 4096 rows", 4096, 4096, 3, {128, 128, 64}, D::WType::F16, 0, F::Form::PipeNp3},
            {"gate group 4097 rows", 4096, 4097, 3, {128, 128, 64}, D::WType::F16, 0, F::Form::PipeNp3},
            {"gate single 4095 rows", 4096, 4095, 1, {288, 0, 0}, D::WType::F16, 0, F::Form::Pipe128},
            {"gate single 4096 rows", 4096, 4096, 1, {288, 0, 0}, D::WType::F16, 0, F::Form::Pipe64},
            // Same boundary on resident chains (<= 128 inputs), then the resident block boundaries at 4096 rows:
            // 8 / 9 / 16 / 17 column blocks of 64 outputs = 4 / 8 / 8 / 16 blocks per work group.
            {"gate resident 4095 rows", 128, 4095, 1, {576, 0, 0}, D::WType::F16, 0, F::Form::Pipe128K16},
            {"gate resident 4097 rows", 128, 4097, 1, {576, 0, 0}, D::WType::F16, 0, F::Form::Xres8},
            {"gate resident 8 blocks", 128, 4096, 1, {512, 0, 0}, D::WType::F16, 0, F::Form::Xres4},
            {"gate resident 9 blocks", 128, 4096, 1, {576, 0, 0}, D::WType::F16, 0, F::Form::Xres8},
            {"gate resident 16 blocks", 128, 4096, 1, {1024, 0, 0}, D::WType::F16, 0, F::Form::Xres8},
            {"gate resident 17 blocks", 128, 4096, 1, {1088, 0, 0}, D::WType::F16, 0, F::Form::Xres16},
        };
        for (const auto& s : gate) run_shape(s, 0);
        if (rounds && !std::getenv("STRIX_FMA_ROWS_SET2")) {
            const Shape bench[] = {
                {"glm kda f_a+g_a+b", 4096, 4096, 3, {128, 128, 64}, D::WType::F16, 0},
                {"glm kda f_a+g_a+b 8192", 4096, 8192, 3, {128, 128, 64}, D::WType::F16, 0},
                {"glm kda f_b", 128, 4096, 1, {8192, 0, 0}, D::WType::F16, 0},
                {"glm kda g_b", 128, 4096, 1, {4096, 0, 0}, D::WType::F16, 0},
                {"glm router", 4096, 4096, 1, {288, 0, 0}, D::WType::F16, 0},
                {"glm short rows", 4096, 1024, 3, {128, 128, 64}, D::WType::F16, 0},
                {"minimax index proj", 6144, 4096, 1, {384, 0, 0}, D::WType::F16, 0},
            };
            for (const auto& s : bench) run_shape(s, rounds);
        }
        if (rounds && std::getenv("STRIX_FMA_ROWS_SET2")) {
            const Shape bench2[] = {
                {"kda group 256", 4096, 256, 3, {128, 128, 64}, D::WType::F16, 0},
                {"kda group 2048", 4096, 2048, 3, {128, 128, 64}, D::WType::F16, 0},
                {"kda f_b 256", 128, 256, 1, {8192, 0, 0}, D::WType::F16, 0},
                {"kda f_b 1024", 128, 1024, 1, {8192, 0, 0}, D::WType::F16, 0},
                {"kda f_b 8192", 128, 8192, 1, {8192, 0, 0}, D::WType::F16, 0},
                {"router 1024", 4096, 1024, 1, {288, 0, 0}, D::WType::F16, 0},
                {"router 2048", 4096, 2048, 1, {288, 0, 0}, D::WType::F16, 0},
                {"router 8192", 4096, 8192, 1, {288, 0, 0}, D::WType::F16, 0},
            };
            for (const auto& s : bench2) run_shape(s, rounds);
        }
        std::printf("%u checks, %u failures\n", checks, failures);
        return failures ? 1 : 0;
    } catch (const std::exception& ex) {
        std::printf("EXCEPTION %s\n", ex.what());
        return 2;
    }
}
