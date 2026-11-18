// Gate K (batch 1): GLM Tier A device kernels == host emulation bit for bit (design notes section 4).
// Real weights from tools/glm/export_host_case.py cases (mHC L3, KDA L4 rank 0, router L3) + randomized inputs; rows
// R = 1..8; rows are independent (row r of an R-row call == the same row alone); inactive rows beyond R stay untouched
// (NaN-poisoned buffers) and set no flag; forced ties / +-0 in top-k and argmax; non-finite flag on a NaN logit.
// usage: hip_glm_tier_a_tests CASES_DIR
#include "glm/host/glm_host.hpp"
#include "glm/kernels/glm_tier_a.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using namespace strix::glm;
namespace H = strix::glm::host;
namespace D = strix::glm::dev;

namespace {

int failures = 0, checks = 0;
void check(bool ok, const std::string& what) {
    ++checks;
    std::printf("%s %s\n", ok ? "OK  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}
void hok(hipError_t e, const char* w) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, hipGetErrorString(e));
        std::exit(3);
    }
}
template <class T> void fr(T* p) { hok(hipFree(const_cast<std::remove_const_t<T>*>(p)), "free"); }
template <class T> T* up(const std::vector<T>& h) {
    T* p = nullptr;
    hok(hipMalloc(&p, std::max<std::size_t>(h.size(), 1) * sizeof(T)), "malloc");
    hok(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return p;
}
template <class T> std::vector<T> down(const T* p, std::size_t n) {
    std::vector<T> h(n);
    hok(hipDeviceSynchronize(), "sync");
    hok(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
    return h;
}
template <class T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
std::vector<float> rnd(std::size_t n, std::mt19937& g, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}
std::vector<std::uint16_t> rnd16(std::size_t n, std::mt19937& g, bool bf) {  // finite F16 / BF16 weights ~N(0, 0.05)
    std::normal_distribution<float> d(0.0F, 0.05F);
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) {
        const float f = d(g);
        if (bf) {
            x = float_to_bf16(f);
        } else {
            _Float16 hh = static_cast<_Float16>(f);
            std::memcpy(&x, &hh, 2);
        }
    }
    return v;
}
const float kNaN = __builtin_nanf("");
unsigned* new_flag() {
    unsigned* f = nullptr;
    hok(hipMalloc(&f, 4), "flag");
    hok(hipMemset(f, 0, 4), "flag0");
    return f;
}
unsigned read_flag(unsigned* f) { return down(f, 1)[0]; }

struct CaseDir {
    std::map<std::string, std::vector<char>> raw;
    explicit CaseDir(const std::string& dir) {
        std::ifstream m(dir + "/manifest.txt");
        if (!m) throw std::runtime_error("no manifest in " + dir);
        std::string line;
        while (std::getline(m, line)) {
            std::istringstream ss(line);
            std::string name, dt;
            std::size_t count = 0;
            ss >> name >> dt >> count;
            const std::size_t es = dt == "u16" ? 2 : (dt == "f64" ? 8 : 4);
            std::vector<char> b(count * es);
            std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
            f.read(b.data(), static_cast<std::streamsize>(b.size()));
            if (static_cast<std::size_t>(f.gcount()) != b.size()) throw std::runtime_error("short " + name);  // fail closed
            raw.emplace(name, std::move(b));
        }
    }
    template <class T> std::vector<T> v(const std::string& n) const {
        const auto& b = raw.at(n);
        std::vector<T> out(b.size() / sizeof(T));
        std::memcpy(out.data(), b.data(), b.size());
        return out;
    }
};

// ---------------------------------------------------------------- dense + rmsnorm
void test_dense(std::mt19937& g) {
    const unsigned in = 4096, n = 300;
    for (int t = 0; t < 3; ++t) {
        const auto wt = static_cast<D::WType>(t);
        std::vector<std::uint16_t> w16;
        std::vector<float> w32;
        if (t < 2) w16 = rnd16(static_cast<std::size_t>(n) * in, g, t == 1); else w32 = rnd(static_cast<std::size_t>(n) * in, g, 0.05F);
        const void* hw = t < 2 ? static_cast<const void*>(w16.data()) : w32.data();
        void* dw = t < 2 ? static_cast<void*>(up(w16)) : static_cast<void*>(up(w32));
        const H::Mat hm{hw, static_cast<H::WType>(t), n, in};
        bool all = true;
        for (unsigned R = 1; R <= 8; ++R) {
            const auto x = rnd(static_cast<std::size_t>(R) * in, g, 1.0F);
            float* dx = up(x);
            std::vector<float> yinit(static_cast<std::size_t>(8) * 280, kNaN);
            float* dy = up(yinit);
            D::dense(dx, dw, wt, in, 20, 280, dy, R, nullptr);  // rows of [20, 300)
            const auto yd = down(dy, yinit.size());
            std::vector<float> yh(yinit);
            for (unsigned r = 0; r < R; ++r)
                H::dense({x.data() + static_cast<std::size_t>(r) * in, in}, hm, {yh.data() + static_cast<std::size_t>(r) * 280, 280}, 20);
            all &= same(yd, yh);  // includes the untouched NaN rows >= R
            fr(dx);
            fr(dy);
        }
        check(all, std::string("R1 dense chain ") + (t == 0 ? "F16" : t == 1 ? "BF16" : "F32") + " [20,300) x 4096, R=1..8, inactive rows untouched");
        fr(dw);
    }
}

void test_rmsnorm(std::mt19937& g) {
    for (unsigned n : {16384U, 4096U, 1536U, 512U, 128U}) {
        const auto w = rnd16(n, g, true);
        std::uint16_t* dw = up(w);
        bool all = true;
        for (unsigned R = 1; R <= 8; ++R) {
            const auto x = rnd(static_cast<std::size_t>(R) * n, g, 3.0F);
            float* dx = up(x);
            for (int weighted = 0; weighted < 2; ++weighted) {
                float* dy = up(std::vector<float>(static_cast<std::size_t>(R) * n));
                D::rmsnorm(dx, weighted ? dw : nullptr, D::WType::BF16, n, kEps, dy, R, nullptr);
                const auto yd = down(dy, static_cast<std::size_t>(R) * n);
                std::vector<float> yh(yd.size());
                const H::Vec hv{w.data(), H::WType::BF16, n};
                for (unsigned r = 0; r < R; ++r)
                    H::rmsnorm({x.data() + static_cast<std::size_t>(r) * n, n}, weighted ? &hv : nullptr, kEps, {yh.data() + static_cast<std::size_t>(r) * n, n});
                all &= same(yd, yh);
                fr(dy);
            }
            fr(dx);
        }
        check(all, "R2/R3 rmsnorm n=" + std::to_string(n) + " (L=" + std::to_string(norm_lanes(n)) + "), weighted + unweighted, R=1..8");
        fr(dw);
    }
}

// ---------------------------------------------------------------- mHC
void test_hc(const std::string& cases, std::mt19937& g) {
    CaseDir c(cases + "/hc_l3");
    const auto fn = c.v<float>("fn"), base = c.v<float>("base"), scale = c.v<float>("scale");
    float *dfn = up(fn), *dbase = up(base), *dscale = up(scale);
    const H::HcWeights hw{H::Mat{fn.data(), H::WType::F32, kHcMix, kHc * kHidden}, base.data(), scale.data()};
    bool all = true, all_x = true, all_f = true;
    std::vector<float> one_x;
    for (unsigned R = 1; R <= 8; ++R) {
        auto X = rnd(static_cast<std::size_t>(R) * kHc * kHidden, g, 1.0F);
        if (R == 8) std::copy(one_x.begin(), one_x.end(), X.begin() + 5 * kHc * kHidden);  // row 5 == the R=1 row
        if (R == 1) one_x = X;
        const auto y = rnd(static_cast<std::size_t>(R) * kHidden, g, 1.0F);
        float *dX = up(X), *dy = up(y), *dF = up(std::vector<float>(X.size())), *dmix = up(std::vector<float>(R * kHcMix));
        float *dxc = up(std::vector<float>(static_cast<std::size_t>(R) * kHidden)), *dXo = up(std::vector<float>(X.size())), *dh = up(std::vector<float>(static_cast<std::size_t>(R) * kHidden));
        unsigned* flag = new_flag();
        D::hc_pre(dX, dfn, dbase, dscale, dF, dmix, dxc, R, nullptr);
        D::hc_expand(dX, dmix, dy, dXo, flag, R, nullptr);
        D::hc_final(dXo, dh, R, nullptr);
        const auto mixd = down(dmix, R * kHcMix);
        const auto xcd = down(dxc, static_cast<std::size_t>(R) * kHidden);
        const auto Xod = down(dXo, X.size());
        const auto hd = down(dh, static_cast<std::size_t>(R) * kHidden);
        for (unsigned r = 0; r < R; ++r) {
            std::span<const float> Xr(X.data() + static_cast<std::size_t>(r) * kHc * kHidden, kHc * kHidden);
            const H::HcMix m = H::hc_mix(Xr, hw);
            std::vector<float> mh(kHcMix), xh(kHidden), Xoh(kHc * kHidden), hh(kHidden);
            std::copy(m.pre.begin(), m.pre.end(), mh.begin());
            std::copy(m.post.begin(), m.post.end(), mh.begin() + 4);
            std::copy(m.comb.begin(), m.comb.end(), mh.begin() + 8);
            H::hc_collapse(Xr, m, xh);
            H::hc_expand(Xr, m, {y.data() + static_cast<std::size_t>(r) * kHidden, kHidden}, Xoh);
            H::hc_final(Xoh, hh);
            all &= std::memcmp(mh.data(), mixd.data() + r * kHcMix, kHcMix * 4) == 0;
            all_x &= std::memcmp(xh.data(), xcd.data() + static_cast<std::size_t>(r) * kHidden, kHidden * 4) == 0 &&
                     std::memcmp(Xoh.data(), Xod.data() + static_cast<std::size_t>(r) * kHc * kHidden, kHc * kHidden * 4) == 0;
            all_f &= std::memcmp(hh.data(), hd.data() + static_cast<std::size_t>(r) * kHidden, kHidden * 4) == 0;
        }
        all &= read_flag(flag) == 0;
        for (float* p : {dX, dy, dF, dmix, dxc, dXo, dh}) fr(p);
        fr(flag);
    }
    check(all, "R6 mHC mix (pre/post/sinkhorn comb), real L3 fn, R=1..8, no flag");
    check(all_x, "R7 mHC collapse + expand, R=1..8");
    check(all_f, "R7 final collapse (mean of 4 streams), R=1..8");
    for (float* p : {dfn, dbase, dscale}) fr(p);
}

// ---------------------------------------------------------------- KDA
void test_kda(const std::string& cases, std::mt19937& g) {
    CaseDir c(cases + "/kda_l4_r0");
    constexpr unsigned Hh = 32, C = Hh * kKdaDim, steps = 12;
    const auto conv = c.v<std::uint16_t>("conv"), onorm = c.v<std::uint16_t>("o_norm");
    const auto A_log = c.v<float>("A_log"), dt = c.v<float>("dt_bias");
    const auto qkv = c.v<float>("qkv"), f = c.v<float>("f"), gate = c.v<float>("gate"), b = c.v<float>("b");
    const D::KdaDev kd{Hh, up(conv), up(A_log), up(dt), up(onorm)};
    const H::KdaWeights hw{conv.data(), A_log.data(), dt.data(), H::Vec{onorm.data(), H::WType::BF16, kKdaDim}};
    // rows: row 0 = the real case sequence; rows 1..R-1 = random sequences (own state slots)
    for (unsigned R : {1U, 3U, 8U}) {
        std::vector<H::KdaState> hs(R, H::KdaState{Hh, std::vector<float>(static_cast<std::size_t>(Hh) * kKdaDim * kKdaDim, 0.0F),
                                                   std::vector<float>(static_cast<std::size_t>(3) * C * 3, 0.0F)});
        float* dS = up(std::vector<float>(static_cast<std::size_t>(R) * Hh * kKdaDim * kKdaDim, 0.0F));
        float* dconv = up(std::vector<float>(static_cast<std::size_t>(R) * 3 * C * 3, 0.0F));
        float *dscr = up(std::vector<float>(static_cast<std::size_t>(R) * Hh * 4 * kKdaDim)), *dbeta = up(std::vector<float>(R * Hh));
        float *dO = up(std::vector<float>(static_cast<std::size_t>(R) * C)), *dY = up(std::vector<float>(static_cast<std::size_t>(R) * C));
        unsigned* flag = new_flag();
        bool all = true;
        for (unsigned t = 0; t < steps; ++t) {
            std::vector<float> iq(static_cast<std::size_t>(R) * 3 * C), ifv(static_cast<std::size_t>(R) * C), ig(ifv.size()), ib(R * Hh);
            for (unsigned r = 0; r < R; ++r) {
                if (r == 0) {
                    std::copy_n(qkv.data() + static_cast<std::size_t>(t) * 3 * C, 3 * C, iq.begin());
                    std::copy_n(f.data() + static_cast<std::size_t>(t) * C, C, ifv.begin());
                    std::copy_n(gate.data() + static_cast<std::size_t>(t) * C, C, ig.begin());
                    std::copy_n(b.data() + static_cast<std::size_t>(t) * Hh, Hh, ib.begin());
                } else {
                    const auto a = rnd(3 * C, g, 1.0F), bb = rnd(C, g, 1.0F), cc = rnd(C, g, 1.0F), dd = rnd(Hh, g, 1.0F);
                    std::copy(a.begin(), a.end(), iq.begin() + static_cast<std::ptrdiff_t>(r * 3 * C));
                    std::copy(bb.begin(), bb.end(), ifv.begin() + static_cast<std::ptrdiff_t>(r * C));
                    std::copy(cc.begin(), cc.end(), ig.begin() + static_cast<std::ptrdiff_t>(r * C));
                    std::copy(dd.begin(), dd.end(), ib.begin() + static_cast<std::ptrdiff_t>(r * Hh));
                }
            }
            float *dq = up(iq), *df = up(ifv), *dg = up(ig), *db = up(ib);
            D::kda_step(kd, dq, df, dg, db, dS, dconv, dscr, dbeta, dO, dY, flag, R, nullptr);
            const auto yd = down(dY, static_cast<std::size_t>(R) * C);
            for (unsigned r = 0; r < R; ++r) {
                std::vector<float> yh(C);
                H::kda_step(hs[r], H::KdaInputs{{iq.data() + static_cast<std::size_t>(r) * 3 * C, 3 * C}, {ifv.data() + static_cast<std::size_t>(r) * C, C},
                                                {ig.data() + static_cast<std::size_t>(r) * C, C}, {ib.data() + r * Hh, Hh}},
                            hw, yh);
                all &= std::memcmp(yh.data(), yd.data() + static_cast<std::size_t>(r) * C, C * 4) == 0;
            }
            for (float* p : {dq, df, dg, db}) fr(p);
        }
        const auto Sd = down(dS, static_cast<std::size_t>(R) * Hh * kKdaDim * kKdaDim);
        const auto cd = down(dconv, static_cast<std::size_t>(R) * 3 * C * 3);
        for (unsigned r = 0; r < R; ++r) {
            all &= std::memcmp(hs[r].S.data(), Sd.data() + static_cast<std::size_t>(r) * hs[r].S.size(), hs[r].S.size() * 4) == 0;
            all &= std::memcmp(hs[r].conv.data(), cd.data() + static_cast<std::size_t>(r) * hs[r].conv.size(), hs[r].conv.size() * 4) == 0;
        }
        all &= read_flag(flag) == 0;
        check(all, "R8-R10 KDA step, real L4 rank-0 weights, 12 steps, R=" + std::to_string(R) + ": outputs + recurrent state + conv state");
        for (float* p : {dS, dconv, dscr, dbeta, dO, dY}) fr(p);
        fr(flag);
    }
}

// ---------------------------------------------------------------- router, combine, swiglu, argmax
void test_route(const std::string& cases, std::mt19937& g) {
    CaseDir c(cases + "/moe_l3");
    const auto router = c.v<std::uint16_t>("router");
    const auto bias = c.v<float>("bias");
    std::uint16_t* drouter = up(router);
    float* dbias = up(bias);
    bool all = true;
    for (unsigned R = 1; R <= 8; ++R) {
        const auto x = rnd(static_cast<std::size_t>(R) * kHidden, g, 0.3F);
        float *dx = up(x), *dl = up(std::vector<float>(R * kExperts)), *dw = up(std::vector<float>(R * kTopE));
        std::uint32_t* dids = up(std::vector<std::uint32_t>(R * kTopE));
        D::dense(dx, drouter, D::WType::F16, kHidden, 0, kExperts, dl, R, nullptr);
        D::route(dl, dbias, dids, dw, R, nullptr);
        const auto idd = down(dids, R * kTopE);
        const auto wd = down(dw, R * kTopE);
        for (unsigned r = 0; r < R; ++r) {
            const H::Route hr = H::route({x.data() + static_cast<std::size_t>(r) * kHidden, kHidden}, H::Mat{router.data(), H::WType::F16, kExperts, kHidden}, bias.data());
            all &= std::memcmp(hr.id.data(), idd.data() + r * kTopE, kTopE * 4) == 0 && std::memcmp(hr.w.data(), wd.data() + r * kTopE, kTopE * 4) == 0;
        }
        for (float* p : {dx, dl, dw}) fr(p);
        fr(dids);
    }
    check(all, "R14 router (real L3 F16 router + bias), top-8 ids + weights, R=1..8");
    // forced ties: equal logits and bias -> the 8 lowest ids; +-0 bias
    std::vector<float> lg(kExperts, 0.25F), zb(kExperts, 0.0F);
    zb[3] = -0.0F;
    float *dl = up(lg), *dzb = up(zb), *dw = up(std::vector<float>(kTopE));
    std::uint32_t* dids = up(std::vector<std::uint32_t>(kTopE));
    D::route(dl, dzb, dids, dw, 1, nullptr);
    const auto idd = down(dids, kTopE);
    bool tie = true;
    for (unsigned k = 0; k < kTopE; ++k) tie &= idd[k] == k;
    check(tie, "R5 router ties (all equal, one -0 bias) -> ids 0..7");
    for (float* p : {dl, dzb, dw}) fr(p);
    fr(dids);
    fr(drouter);
    fr(dbias);
}

// Oracle for the wave-parallel route_kernel: the former lane-0 serial selection, verbatim.
__global__ void route_serial_oracle(const float* logits, const float* bias, std::uint32_t* ids, float* w) {
    __shared__ float sg[kExperts];
    __shared__ std::uint32_t key[kExperts];
    __shared__ bool taken[kExperts];
    if (threadIdx.x != 0) return;
    const unsigned r = blockIdx.x;
    const float* lg = logits + static_cast<std::size_t>(r) * kExperts;
    for (unsigned e = 0; e < kExperts; ++e) {
        sg[e] = sigmoid(lg[e]);
        key[e] = ordered_key(sg[e] + bias[e]);
    }
    for (unsigned e = 0; e < kExperts; ++e) taken[e] = false;
    std::uint32_t pick[kTopE];
    for (unsigned k = 0; k < kTopE; ++k) {
        unsigned best = kExperts;
        for (unsigned e = 0; e < kExperts; ++e)
            if (!taken[e] && (best == kExperts || key[e] > key[best])) best = e;
        taken[best] = true;
        pick[k] = best;
    }
    for (unsigned a = 1; a < kTopE; ++a)
        for (unsigned b = a; b > 0 && pick[b - 1] > pick[b]; --b) {
            const std::uint32_t t = pick[b];
            pick[b] = pick[b - 1];
            pick[b - 1] = t;
        }
    float den = 0.0F;
    for (unsigned k = 0; k < kTopE; ++k) den = den + sg[pick[k]];
    den = den + kRouteDenEps;
    for (unsigned k = 0; k < kTopE; ++k) {
        ids[static_cast<std::size_t>(r) * kTopE + k] = pick[k];
        w[static_cast<std::size_t>(r) * kTopE + k] = (sg[pick[k]] / den) * kRouteScale;
    }
}

// Device stress set: 4096 rows of random, quantized (mass ties), signed-zero, infinite, NaN and saturated logits with
// random / zero / -0 / tie-making biases; the wave route must equal the serial oracle bit for bit (ids and weights).
void test_route_stress(std::mt19937& g) {
    constexpr unsigned R = 4096;
    std::vector<float> lg(static_cast<std::size_t>(R) * kExperts);
    std::normal_distribution<float> n(0.0F, 3.0F);
    std::uniform_int_distribution<int> q(-4, 4), pick(0, kExperts - 1);
    for (unsigned r = 0; r < R; ++r) {
        float* row = lg.data() + static_cast<std::size_t>(r) * kExperts;
        const unsigned kind = r % 8;
        for (unsigned e = 0; e < kExperts; ++e) {
            float v = n(g);
            if (kind == 1) v = 0.5F * static_cast<float>(q(g));      // few distinct values: ties across lanes and slots
            if (kind == 2) v = (e % 2) ? 0.0F : -0.0F;               // signed zeros only
            if (kind == 3) v = n(g) * 40.0F;                         // saturated sigmoids (1 and +0)
            if (kind == 4 && e % 37 == 5) v = INFINITY;
            if (kind == 4 && e % 41 == 7) v = -INFINITY;
            if (kind == 5) v = 1.0F;                                 // all equal
            row[e] = v;
        }
        if (kind == 6) row[pick(g)] = NAN;
        if (kind == 7) { row[pick(g)] = NAN; row[pick(g)] = -NAN; row[pick(g)] = INFINITY; }
    }
    for (unsigned variant = 0; variant < 3; ++variant) {
        std::vector<float> bias(kExperts);
        for (unsigned e = 0; e < kExperts; ++e)
            bias[e] = variant == 0 ? 0.01F * n(g) : variant == 1 ? ((e % 3) ? 0.0F : -0.0F) : 0.25F * static_cast<float>(q(g));
        float *dl = up(lg), *db = up(bias);
        float *dw = up(std::vector<float>(static_cast<std::size_t>(R) * kTopE)), *dwo = up(std::vector<float>(static_cast<std::size_t>(R) * kTopE));
        std::uint32_t* di = up(std::vector<std::uint32_t>(static_cast<std::size_t>(R) * kTopE));
        std::uint32_t* dio = up(std::vector<std::uint32_t>(static_cast<std::size_t>(R) * kTopE));
        for (unsigned rows : {1U, 7U, 32U, R}) {
            D::route(dl, db, di, dw, rows, nullptr);
            route_serial_oracle<<<rows, 32>>>(dl, db, dio, dwo);
            hok(hipGetLastError(), "oracle");
            const std::size_t m = static_cast<std::size_t>(rows) * kTopE;
            check(same(down(di, m), down(dio, m)) && same(down(dw, m), down(dwo, m)),
                  "P3 wave route == serial oracle (ids + weights bitwise), bias variant " + std::to_string(variant) +
                      ", rows " + std::to_string(rows));
        }
        for (float* p : {dl, db, dw, dwo}) fr(p);
        fr(di);
        fr(dio);
    }
}

void test_small(std::mt19937& g) {
    for (unsigned R : {1U, 8U}) {  // combine
        const auto w = rnd(R * kTopE, g, 0.5F), eo = rnd(static_cast<std::size_t>(R) * kTopE * kHidden, g, 1.0F), sh = rnd(static_cast<std::size_t>(R) * kHidden, g, 1.0F);
        float *dw = up(w), *deo = up(eo), *dsh = up(sh), *dy = up(std::vector<float>(static_cast<std::size_t>(R) * kHidden));
        D::moe_combine(dw, deo, dsh, dy, R, nullptr);
        const auto yd = down(dy, static_cast<std::size_t>(R) * kHidden);
        bool all = true;
        for (unsigned r = 0; r < R; ++r) {
            H::Route hr;
            for (unsigned k = 0; k < kTopE; ++k) hr.w[k] = w[r * kTopE + k];
            std::vector<float> yh(kHidden);
            H::moe_combine(hr, {eo.data() + static_cast<std::size_t>(r) * kTopE * kHidden, kTopE * kHidden}, {sh.data() + static_cast<std::size_t>(r) * kHidden, kHidden}, yh);
            all &= std::memcmp(yh.data(), yd.data() + static_cast<std::size_t>(r) * kHidden, kHidden * 4) == 0;
        }
        check(all, "R16 MoE combine R=" + std::to_string(R));
        for (float* p : {dw, deo, dsh, dy}) fr(p);
    }
    {  // swiglu incl. clamp edges
        auto gg = rnd(1 << 16, g, 6.0F), uu = rnd(1 << 16, g, 6.0F);
        gg[0] = 10.0F; gg[1] = 11.0F; gg[2] = -1000.0F; uu[0] = -10.0F; uu[1] = 10.5F; uu[2] = -0.0F;
        float *dg = up(gg), *du = up(uu), *da = up(std::vector<float>(gg.size()));
        D::swiglu(dg, du, da, gg.size(), nullptr);
        const auto ad = down(da, gg.size());
        bool all = true;
        for (std::size_t i = 0; i < gg.size(); ++i) {
            const float h = H::swiglu(gg[i], uu[i]);
            all &= std::memcmp(&h, &ad[i], 4) == 0;
        }
        check(all, "R15 swiglu (65536 values incl. clamp edges)");
        for (float* p : {dg, du, da}) fr(p);
    }
    {  // argmax: ties, +-0, slice offset; NaN -> flag
        const unsigned n = 77440;
        auto lg = rnd(n, g, 1.0F);
        lg[100] = 9.0F; lg[50000] = 9.0F; lg[200] = 9.0F;  // tie: lowest id 100
        float *dl = up(lg), *dv = up(std::vector<float>(1));
        std::uint32_t* di = up(std::vector<std::uint32_t>(1));
        unsigned* flag = new_flag();
        D::argmax(dl, n, 77440, dv, di, flag, 1, nullptr);
        const H::ArgMax hm = H::argmax_local(lg, 77440);
        const auto vd = down(dv, 1);
        const auto id = down(di, 1);
        check(id[0] == hm.id && std::memcmp(&vd[0], &hm.value, 4) == 0 && id[0] == 77440 + 100 && read_flag(flag) == 0,
              "R18 argmax over a 77,440-vocab slice, 3-way tie -> lowest id");
        std::vector<float> z(1000, -0.0F);
        z[7] = 0.0F;
        float* dz = up(z);
        D::argmax(dz, 1000, 0, dv, di, flag, 1, nullptr);
        check(down(di, 1)[0] == 0, "R18 argmax all +-0 -> id 0 (equal values)");
        lg[333] = kNaN;
        float* dn = up(lg);
        D::argmax(dn, n, 0, dv, di, flag, 1, nullptr);
        check(read_flag(flag) == 1, "C5 non-finite logit sets the device flag");
        for (float* p : {dl, dv, dz, dn}) fr(p);
        fr(di);
        fr(flag);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s CASES_DIR\n", argv[0]);
        return 2;
    }
    std::mt19937 g(20261024);
    test_dense(g);
    test_rmsnorm(g);
    test_hc(argv[1], g);
    test_kda(argv[1], g);
    test_route(argv[1], g);
    test_route_stress(g);
    test_small(g);
    std::printf("checks: %d failures: %d\n", checks, failures);
    return failures ? 1 : 0;
}
