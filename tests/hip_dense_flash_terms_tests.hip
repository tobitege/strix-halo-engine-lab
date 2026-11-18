// Device test of the shared dense flash core's operand forms (dense_flash_prefill.hpp, kTerms 1 and 2) against the
// host model of the same products in the same order and against exact FP64 attention (src/attn_core/two_term_model.hpp).
// Causal policy from position 0, f16 operands, one D piece (the vision tower's Shape<DP, 1, 1, 64>). Data classes:
//   normal  |q|, |k| <= 3, |v| <= 1: remainders normal f16, the two-term gain is the full u^2 factor
//   small   all operands ~2^-6: lo terms subnormal (2^-25 absolute floor)
//   tiny    v ~2^-16: hi itself subnormal, two terms equal one term (no gain by construction)
//   cancel  q . k sums of +-64 products that cancel: the score's relative error is large in both forms; the bound
//           is relative to sum |q_d k_d|
// Tails: rows 1, 17, 100 (partial 16-row tiles and key tiles); D 64 and D 72 padded to DP 80 (odd WMMA step count).
// Checks per case and variant: failure word 0, every output finite; |device - exact| <= the a-priori bound built from
// the model's operand / product bounds (each WMMA step assumed to round up to 16 times: the internal order is not
// documented); the device is closer to the model of its own variant than to the other variant's; two terms beat one
// term by >= 8x on normal and cancel data and are no worse on tiny data. Needs a GPU: not a CTest test.
#include "attn_core/dense_flash_prefill.hpp"
#include "attn_core/two_term_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace dense = strix::attn_core::dense;
namespace tile = strix::attn_core::tile;
namespace tm = strix::attn_core::model;

struct TArgs {
    const float* q;
    float* out;
    const std::uint16_t *k, *vt, *klo, *vtlo;
    unsigned* failure;
    unsigned n, heads, vstride;
};
template <unsigned DP, unsigned D, unsigned Terms>
struct TPolicy {
    using Args = TArgs;
    static constexpr unsigned kTerms = Terms;
    using Shape = dense::Shape<DP, 1, 1, 64>;
    static constexpr float kScaleLog2 = static_cast<float>(1.0 / (D == 64U ? 8.0 : 8.48528137423857)) * 1.44269504088896341F;
    static constexpr tile::Operand kOp = tile::Operand::f16;
    __device__ static bool valid(const Args& a) { return a.n != 0U; }
    __host__ __device__ static unsigned rows(const Args& a) { return a.n; }
    __device__ static unsigned position(const Args&) { return 0U; }
    __device__ static unsigned window(const Args&) { return 0U; }
    __device__ static const float* query(const Args& a, unsigned row, unsigned h) { return a.q + (std::size_t(row) * a.heads + h) * DP; }
    __device__ static float* out(const Args& a, unsigned row, unsigned h) { return a.out + (std::size_t(row) * a.heads + h) * DP; }
    __device__ static const std::uint16_t* key(const Args& a, unsigned p, unsigned h) { return a.k + (std::size_t(p) * a.heads + h) * DP; }
    __device__ static const std::uint16_t* value_t(const Args& a, unsigned h, unsigned d) { return a.vt + std::size_t(h * DP + d) * a.vstride; }
    __device__ static const std::uint16_t* key_lo(const Args& a, unsigned p, unsigned h) { return a.klo + (std::size_t(p) * a.heads + h) * DP; }
    __device__ static const std::uint16_t* value_t_lo(const Args& a, unsigned h, unsigned d) { return a.vtlo + std::size_t(h * DP + d) * a.vstride; }
    __device__ static unsigned* failure(const Args& a) { return a.failure; }
};

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
std::uint16_t f16_bits(float x) { return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(x)); }

enum class Data { normal, small, tiny, cancel };
const char* name(Data d) { return d == Data::normal ? "normal" : d == Data::small ? "small" : d == Data::tiny ? "tiny" : "cancel"; }

struct Problem {
    unsigned n, heads, dp, d;
    std::vector<float> q, k, v;  // [n][heads][dp], padded dims zero
};
Problem make(unsigned n, unsigned dp, unsigned d, Data kind, std::uint64_t seed) {
    Problem p{n, 2U, dp, d, {}, {}, {}};
    const std::size_t count = std::size_t(n) * p.heads * dp;
    p.q.assign(count, 0.0F);
    p.k.assign(count, 0.0F);
    p.v.assign(count, 0.0F);
    const auto uni = [&seed] {  // [-1, 1)
        seed ^= seed << 13U;
        seed ^= seed >> 7U;
        seed ^= seed << 17U;
        return static_cast<float>(static_cast<double>(seed >> 11U) * 0x1p-52 - 1.0);
    };
    const float aq = kind == Data::small ? 0x1p-6F : kind == Data::tiny ? 0.5F : kind == Data::cancel ? 8.0F : 3.0F;
    const float av = kind == Data::small ? 0x1p-6F : kind == Data::tiny ? 0x1p-16F : 1.0F;
    for (unsigned r = 0; r < n; ++r)
        for (unsigned h = 0; h < p.heads; ++h) {
            const std::size_t o = (std::size_t(r) * p.heads + h) * dp;
            for (unsigned i = 0; i < d; ++i) {
                p.v[o + i] = av * uni();
                if (kind == Data::cancel) {  // pairs (2i, 2i+1): q equal, k opposite -> exact cancellation + a small part
                    const unsigned base = i & ~1U;
                    const std::size_t ob = (std::size_t(base) * 2654435761U + h) % 997U;  // q shared by every row
                    const float qv = aq * (0.5F + 0.5F * static_cast<float>(ob) / 997.0F);
                    p.q[o + i] = qv;
                    p.k[o + i] = (i % 2U == 0U ? 1.0F : -1.0F) * aq * (0.75F + 0.25F * std::fabs(uni())) + 0.01F * uni();
                } else {
                    p.q[o + i] = aq * uni();
                    p.k[o + i] = aq * uni();
                }
            }
        }
    if (kind == Data::cancel)  // k_{2i+1} = -k_{2i} up to the small part, q_{2i} = q_{2i+1}
        for (std::size_t o = 0; o < count; o += dp)
            for (unsigned i = 0; i + 1U < d; i += 2U) p.k[o + i + 1U] = -p.k[o + i] + 0.01F * uni();
    return p;
}

template <unsigned DP, unsigned D, unsigned Terms>
std::vector<float> run_device(const Problem& p, unsigned& failure) {
    const unsigned npad = (p.n + 15U) & ~15U, hw = p.heads * DP;
    std::vector<std::uint16_t> k(std::size_t(npad) * hw, 0), klo(k.size(), 0), vt(std::size_t(hw) * npad, 0), vtlo(vt.size(), 0);
    for (unsigned r = 0; r < p.n; ++r)
        for (unsigned c = 0; c < hw; ++c) {
            const tm::Terms a = tm::split(p.k[std::size_t(r) * hw + c], tm::Format::f16, 2U);
            const tm::Terms b = tm::split(p.v[std::size_t(r) * hw + c], tm::Format::f16, 2U);
            k[std::size_t(r) * hw + c] = f16_bits(a.hi);
            klo[std::size_t(r) * hw + c] = f16_bits(a.lo);
            vt[std::size_t(c) * npad + r] = f16_bits(b.hi);
            vtlo[std::size_t(c) * npad + r] = f16_bits(b.lo);
        }
    float *dq = nullptr, *dout = nullptr;
    std::uint16_t *dk = nullptr, *dklo = nullptr, *dvt = nullptr, *dvtlo = nullptr;
    unsigned* df = nullptr;
    const std::size_t qbytes = p.q.size() * 4U, kbytes = k.size() * 2U;
    ok(hipMalloc(&dq, qbytes), "q");
    ok(hipMalloc(&dout, qbytes), "out");
    ok(hipMalloc(&dk, kbytes), "k");
    ok(hipMalloc(&dklo, kbytes), "klo");
    ok(hipMalloc(&dvt, kbytes), "vt");
    ok(hipMalloc(&dvtlo, kbytes), "vtlo");
    ok(hipMalloc(&df, 4U), "failure");
    ok(hipMemcpy(dq, p.q.data(), qbytes, hipMemcpyHostToDevice), "q up");
    ok(hipMemcpy(dk, k.data(), kbytes, hipMemcpyHostToDevice), "k up");
    ok(hipMemcpy(dklo, klo.data(), kbytes, hipMemcpyHostToDevice), "klo up");
    ok(hipMemcpy(dvt, vt.data(), kbytes, hipMemcpyHostToDevice), "vt up");
    ok(hipMemcpy(dvtlo, vtlo.data(), kbytes, hipMemcpyHostToDevice), "vtlo up");
    ok(hipMemset(dout, 0xFF, qbytes), "out fill");  // NaN: a row the core does not write fails the finite check
    ok(hipMemset(df, 0, 4U), "failure clear");
    const TArgs a{dq, dout, dk, dvt, Terms == 2U ? dklo : nullptr, Terms == 2U ? dvtlo : nullptr, df, p.n, p.heads, npad};
    dense::launch<TPolicy<DP, D, Terms>>(a, p.heads, nullptr);
    ok(hipGetLastError(), "launch");
    ok(hipDeviceSynchronize(), "sync");
    std::vector<float> out(p.q.size());
    ok(hipMemcpy(out.data(), dout, qbytes, hipMemcpyDeviceToHost), "out down");
    ok(hipMemcpy(&failure, df, 4U, hipMemcpyDeviceToHost), "failure down");
    for (void* x : {static_cast<void*>(dq), static_cast<void*>(dout), static_cast<void*>(dk), static_cast<void*>(dklo), static_cast<void*>(dvt),
                    static_cast<void*>(dvtlo), static_cast<void*>(df)})
        (void)hipFree(x);
    return out;
}

struct Ref {
    std::vector<double> exact;
    std::vector<float> model1, model2;
};
// Exact FP64 attention (the policy's kScaleLog2 as the exact constant) and the host model of both variants.
Ref reference(const Problem& p, float scale_log2) {
    const unsigned dp = p.dp, H = p.heads;
    Ref r{std::vector<double>(p.q.size()), std::vector<float>(p.q.size()), std::vector<float>(p.q.size())};
    std::vector<tm::Terms> cut[2][3];  // [terms - 1][q, k, v]
    for (unsigned terms = 1; terms <= 2U; ++terms) {
        const std::vector<float>* x[3] = {&p.q, &p.k, &p.v};
        for (unsigned i = 0; i < 3U; ++i) {
            cut[terms - 1U][i].resize(x[i]->size());
            for (std::size_t e = 0; e < x[i]->size(); ++e) cut[terms - 1U][i][e] = tm::split((*x[i])[e], tm::Format::f16, terms);
        }
    }
    for (unsigned row = 0; row < p.n; ++row)
        for (unsigned h = 0; h < H; ++h) {
            const std::size_t o = (std::size_t(row) * H + h) * dp;
            const auto key = [&](unsigned j) { return &p.k[(std::size_t(j) * H + h) * dp]; };
            const auto value = [&](unsigned j) { return &p.v[(std::size_t(j) * H + h) * dp]; };
            for (unsigned terms = 1; terms <= 2U; ++terms) {
                const std::vector<tm::Terms>* t = cut[terms - 1U];
                const auto kt = [&](unsigned j) { return &t[1][(std::size_t(j) * H + h) * dp]; };
                const auto vt = [&](unsigned j) { return &t[2][(std::size_t(j) * H + h) * dp]; };
                tm::dense_row(&t[0][o], row + 1U, kt, vt, dp, scale_log2, terms, tm::Format::f16, terms == 1U ? &r.model1[o] : &r.model2[o]);
            }
            std::vector<double> s(row + 1U), w(row + 1U);
            double top = -1e300, l = 0.0;
            for (unsigned j = 0; j <= row; ++j) {
                double dot = 0.0;
                for (unsigned d = 0; d < dp; ++d) dot += double(p.q[o + d]) * key(j)[d];
                s[j] = dot * scale_log2;
                top = std::max(top, s[j]);
            }
            for (unsigned j = 0; j <= row; ++j) l += (w[j] = std::exp2(s[j] - top));
            for (unsigned d = 0; d < dp; ++d) {
                double acc = 0.0;
                for (unsigned j = 0; j <= row; ++j) acc += w[j] * value(j)[d];
                r.exact[o + d] = acc / l;
            }
        }
    return r;
}

// A-priori bound of |device - exact| for one (row, head), variant `terms` (see the header comment).
void bound_row(const Problem& p, unsigned row, unsigned h, float scale_log2, unsigned terms, const std::vector<double>& exact, double* out) {
    const unsigned dp = p.dp, H = p.heads, nk = row + 1U, tiles = (nk + 15U) / 16U;
    const tm::Format f = tm::Format::f16;
    const double u24 = 0x1p-24, kS = scale_log2, ln2 = std::log(2.0);
    const double step_roundings = 16.0;  // per WMMA step, conservative (internal order undocumented)
    const double qk_roundings = step_roundings * tm::dot_roundings(dp, terms);
    const std::size_t o = (std::size_t(row) * H + h) * dp;
    std::vector<double> s(nk), delta(nk), w(nk);
    double top = -1e300, dmax = 0.0;
    for (unsigned j = 0; j < nk; ++j) {
        const float* k = &p.k[(std::size_t(j) * H + h) * dp];
        double dot = 0.0, pb = 0.0, mag = 0.0;
        for (unsigned d = 0; d < dp; ++d) {
            dot += double(p.q[o + d]) * k[d];
            pb += tm::product_bound(p.q[o + d], k[d], f, terms);
            mag += std::fabs(double(p.q[o + d]) * k[d]);
        }
        s[j] = dot * kS;
        delta[j] = kS * (pb + qk_roundings * u24 * mag) + 2.0 * u24 * std::fabs(s[j]);  // + the FP32 scale multiply
        top = std::max(top, s[j]);
        dmax = std::max(dmax, delta[j]);
    }
    double l = 0.0, rho = 0.0;
    for (unsigned j = 0; j < nk; ++j) {
        l += (w[j] = std::exp2(s[j] - top));
        // log2-unit errors of the score and of the running max, the FP32 subtraction, the hardware exp2 (4 ulp)
        rho = std::max(rho, ln2 * (delta[j] + dmax + u24 * (std::fabs(s[j] - top) + 2.0 * dmax)) + 4.0 * 0x1p-23);
    }
    const double wfac = std::expm1(2.0 * rho);
    for (unsigned d = 0; d < dp; ++d) {
        const double oe = exact[o + d];
        double spread = 0.0, pvb = 0.0, pv = 0.0, pden = 0.0;
        for (unsigned j = 0; j < nk; ++j) {
            const double v = p.v[(std::size_t(j) * H + h) * dp + d];
            spread += w[j] * std::fabs(v - oe);
            pvb += tm::product_bound(w[j], v, f, terms);  // P and V operand errors of this product
            pv += w[j] * std::fabs(v);
            pden += tm::operand_bound(w[j], f, terms) + 2.0 * u24 * w[j];
        }
        const double pv_round = (step_roundings * (terms == 2U ? 3.0 : 1.0) + 1.0) * tiles * u24 * pv;
        const double den_round = (16.0 + 2.0) * tiles * u24 * l + pden;
        out[o + d] = 1.5 * (wfac * spread / l + (pvb + pv_round) / l + std::fabs(oe) * den_round / l + u24 * std::fabs(oe)) + 0x1p-40;
    }
}

struct Stat {
    double err = 0.0, ratio = 0.0, to_own = 0.0, to_other = 0.0;
    bool finite = true;
};
Stat measure(const Problem& p, const std::vector<float>& dev, const Ref& r, const std::vector<double>& bound, unsigned terms) {
    Stat s;
    const std::vector<float>& own = terms == 2U ? r.model2 : r.model1;
    const std::vector<float>& other = terms == 2U ? r.model1 : r.model2;
    for (unsigned row = 0; row < p.n; ++row)
        for (unsigned h = 0; h < p.heads; ++h)
            for (unsigned d = 0; d < p.d; ++d) {  // padded dims excluded
                const std::size_t i = (std::size_t(row) * p.heads + h) * p.dp + d;
                if (!std::isfinite(dev[i])) s.finite = false;
                const double e = std::fabs(double(dev[i]) - r.exact[i]);
                s.err = std::max(s.err, e);
                s.ratio = std::max(s.ratio, e / bound[i]);
                s.to_own = std::max(s.to_own, std::fabs(double(dev[i]) - own[i]));
                s.to_other = std::max(s.to_other, std::fabs(double(dev[i]) - other[i]));
            }
    return s;
}

int fails = 0;
void check(bool c, const char* what, const char* tag) {
    if (!c) {
        ++fails;
        std::printf("FAIL %s: %s\n", tag, what);
    }
}

template <unsigned DP, unsigned D>
void run_all() {
    const float kS = TPolicy<DP, D, 1>::kScaleLog2;
    for (const unsigned n : {1U, 17U, 100U})
        for (const Data kind : {Data::normal, Data::small, Data::tiny, Data::cancel}) {
            const Problem p = make(n, DP, D, kind, 0x51ED2701ULL + n * 7919U + static_cast<unsigned>(kind));
            const Ref r = reference(p, kS);
            std::vector<double> b1(p.q.size()), b2(p.q.size());
            for (unsigned row = 0; row < n; ++row)
                for (unsigned h = 0; h < p.heads; ++h) {
                    bound_row(p, row, h, kS, 1U, r.exact, b1.data());
                    bound_row(p, row, h, kS, 2U, r.exact, b2.data());
                }
            unsigned f1 = 0, f2 = 0;
            const std::vector<float> d1 = run_device<DP, D, 1>(p, f1), d2 = run_device<DP, D, 2>(p, f2);
            const Stat s1 = measure(p, d1, r, b1, 1U), s2 = measure(p, d2, r, b2, 2U);
            char tag[64];
            std::snprintf(tag, sizeof tag, "D%u/DP%u rows %u %s", D, DP, n, name(kind));
            std::printf("%-26s terms1 err %.3g bound-ratio %.3f own %.3g other %.3g | terms2 err %.3g bound-ratio %.3f own %.3g other %.3g | gain %.1fx\n",
                        tag, s1.err, s1.ratio, s1.to_own, s1.to_other, s2.err, s2.ratio, s2.to_own, s2.to_other,
                        s2.err > 0.0 ? s1.err / s2.err : 0.0);
            check(f1 == 0U && f2 == 0U, "failure word", tag);
            check(s1.finite && s2.finite, "finite outputs", tag);
            check(s1.ratio <= 1.0, "terms 1 within the a-priori bound", tag);
            check(s2.ratio <= 1.0, "terms 2 within the a-priori bound", tag);
            if (kind == Data::normal || kind == Data::cancel) {
                check(s1.to_own < s1.to_other / 4.0 && s2.to_own < s2.to_other / 4.0, "device closer to its own variant's model", tag);
                check(s2.err * 8.0 <= s1.err, "two terms >= 8x better", tag);
            }
            if (kind == Data::tiny) check(s2.err <= s1.err * 1.01 + 1e-30, "tiny: two terms no worse", tag);
        }
}

}  // namespace

int main() {
    try {
        run_all<64, 64>();
        run_all<80, 72>();
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 2;
    }
    std::printf("dense flash terms tests: %s (%d failures)\n", fails == 0 ? "PASS" : "FAIL", fails);
    return fails == 0 ? 0 : 1;
}
