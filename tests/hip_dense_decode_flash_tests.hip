// Shared FAST wide-window attention (attn_core/dense_decode_flash.hpp) vs the decode form (attn_core/dense_decode.hpp)
// on random data, Dim 256, GQA 24 / 4 heads, gated, partial RoPE: two windows (positions 0..39, then 40..92: the
// second refreshes V^T only from 40 and its last 16-key tile is partial). Pass: max |flash - decode| <= 2e-2 x max |decode|
// (BF16 Q / P operands) and identical cache bits (the prep stage is shared).
#include <hip/hip_runtime.h>

#include "attn_core/dense_decode_flash.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define CHECK(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP %s at %d\n", hipGetErrorString(e_), __LINE__); return 1; } } while (0)

namespace ad = strix::attn_core;

int main() {
    constexpr unsigned D = 256, H = 24, KV = 4, MAXP = 128, VS = 128;
    const unsigned windows[2][2] = {{0U, 40U}, {40U, 53U}};
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    auto fill = [&](std::vector<float>& v) { for (auto& x : v) x = nd(rng); };
    std::vector<float> qg(D), kg(D);
    for (auto& x : qg) x = 1.0F + 0.1F * nd(rng);
    for (auto& x : kg) x = 1.0F + 0.1F * nd(rng);
    float *dq, *dk, *dv, *dqg, *dkg, *qs, *ctx_a, *ctx_b, *ws;
    std::uint16_t *kc_a, *vc_a, *kc_b, *vc_b, *vt;
    unsigned* fail;
    const std::size_t wsf = ad::dense::workspace_floats(64, H, D, 4);
    CHECK(hipMalloc(&dq, 64 * H * 2 * D * 4)); CHECK(hipMalloc(&dk, 64 * KV * D * 4)); CHECK(hipMalloc(&dv, 64 * KV * D * 4));
    CHECK(hipMalloc(&dqg, D * 4)); CHECK(hipMalloc(&dkg, D * 4)); CHECK(hipMalloc(&qs, 64 * H * D * 4));
    CHECK(hipMalloc(&ctx_a, 64 * H * D * 4)); CHECK(hipMalloc(&ctx_b, 64 * H * D * 4)); CHECK(hipMalloc(&ws, wsf * 4));
    for (auto** p : {&kc_a, &vc_a, &kc_b, &vc_b}) { CHECK(hipMalloc(p, MAXP * KV * D * 2)); CHECK(hipMemset(*p, 0, MAXP * KV * D * 2)); }
    CHECK(hipMalloc(&vt, KV * D * VS * 2)); CHECK(hipMemset(vt, 0, KV * D * VS * 2));
    CHECK(hipMalloc(&fail, 4)); CHECK(hipMemset(fail, 0, 4));
    CHECK(hipMemcpy(dqg, qg.data(), D * 4, hipMemcpyHostToDevice)); CHECK(hipMemcpy(dkg, kg.data(), D * 4, hipMemcpyHostToDevice));
    double worst = 0.0, peak = 0.0;
    for (const auto& w : windows) {
        const unsigned pos0 = w[0], n = w[1];
        std::vector<float> q(std::size_t(n) * H * 2 * D), k(std::size_t(n) * KV * D), v(std::size_t(n) * KV * D);
        fill(q); fill(k); fill(v);
        CHECK(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice)); CHECK(hipMemcpy(dk, k.data(), k.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dv, v.data(), v.size() * 4, hipMemcpyHostToDevice));
        for (unsigned r = 0; r < n; r += 16U) {  // decode form, 16-row windows
            const unsigned m = n - r < 16U ? n - r : 16U;
            const ad::dense_decode::Params p{m, H, KV, 2U * D, 64U, pos0 + r, 1e-6F, 0.0625F, 1e7, true};
            ad::dense_decode::launch<D>(p, dq + std::size_t(r) * H * 2 * D, dk + std::size_t(r) * KV * D, dv + std::size_t(r) * KV * D, dqg, dkg,
                                        qs, kc_a, vc_a, ctx_a + std::size_t(r) * H * D, nullptr);
        }
        const ad::dense_decode::Params p{n, H, KV, 2U * D, 64U, pos0, 1e-6F, 0.0625F, 1e7, true};
        ad::dense_decode_flash::launch<D, 6>(p, dq, dk, dv, dqg, dkg, qs, kc_b, vc_b, vt, VS, pos0, fail, ws, wsf, ctx_b, nullptr);
        CHECK(hipDeviceSynchronize());
        std::vector<float> a(std::size_t(n) * H * D), b(a.size());
        CHECK(hipMemcpy(a.data(), ctx_a, a.size() * 4, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(b.data(), ctx_b, b.size() * 4, hipMemcpyDeviceToHost));
        for (std::size_t i = 0; i < a.size(); ++i) { worst = std::fmax(worst, std::fabs(double(a[i]) - b[i])); peak = std::fmax(peak, std::fabs(double(a[i]))); }
    }
    std::vector<std::uint16_t> ca(MAXP * KV * D), cb(ca.size());
    bool same = true;
    for (int c = 0; c < 2; ++c) {
        CHECK(hipMemcpy(ca.data(), c ? vc_a : kc_a, ca.size() * 2, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(cb.data(), c ? vc_b : kc_b, cb.size() * 2, hipMemcpyDeviceToHost));
        same = same && ca == cb;
    }
    unsigned f = 0;
    CHECK(hipMemcpy(&f, fail, 4, hipMemcpyDeviceToHost));
    const bool ok = same && f == 0U && worst <= 2e-2 * peak && peak > 0.0;
    std::printf("dense_decode_flash: max |flash - decode| %.3e (peak %.3e, rel %.2e), cache bits %s, failure %u -> %s\n", worst, peak,
                worst / peak, same ? "equal" : "DIFFER", f, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
