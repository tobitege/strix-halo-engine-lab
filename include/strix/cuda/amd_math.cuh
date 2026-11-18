// Bit-exact CUDA emulation of the gfx1151 (ROCm OCML / hardware) single-precision functions used by the Qwen prefill.
//
// The hardware approximations v_exp_f32, v_rsq_f32 and v_rcp_f32 are faithful (measured exhaustively: never more than one
// ulp from the exact value), so wherever one is not the correctly rounded result it is the other neighbour of the exact
// value. One bit per input selects it. The bits come from an exhaustive gfx1151 sweep (qwen-rtx-prefill probe p2):
//   - v_exp_f32 differs from the rounding of the fp64 reference only for |x| in [2^-25, 1): bitmap over those inputs;
//   - v_rsq_f32(x * 4^k) == v_rsq_f32(x) * 2^-k and v_rcp_f32(x * 2^k) == v_rcp_f32(x) * 2^-k for every normal input with
//     a normal result (0 violations in ~2.1e9 inputs each): bitmaps over [1, 4) and [1, 2) only;
//   - "hard" inputs, whose fp64 reference lies within 2^-48 (relative) of a float midpoint, are listed with their hardware
//     result (rsq/rcp canonicalised to [1, 4) / [1, 2) by the invariance).
// The reference is the same expression on both sides: (float) of the fp64 function (exp2, 1/sqrt, 1/x).
// The OCML sequences around the hardware instructions (expf, rsqrtf) are transliterated from the gfx1151 ISA of the engine
// build flags (-fno-fast-math, correctly rounded division/sqrt, IEEE denormals): every operation is one IEEE operation with
// the same rounding, written with explicit __fmul_rn/__fadd_rn/__fmaf_rn so that nvcc cannot contract or reorder.
// Domains: amd_v_exp: |x| < 1 (all expf reductions); amd_v_rsq: positive normal x; amd_v_rcp: normal x with |x| < 2^126.
// NaN payloads are not reproduced (any NaN in the prefill is a failure anyway).
#pragma once

#include <cstdint>

namespace strix::cuda::amdmath {

struct Tables {
    const std::uint32_t* exp_bits;   // [2 signs][25 * 2^23 bits]: |x| patterns [102<<23, 127<<23), positive then negative
    const std::uint32_t* rsq_bits;   // [2^24 >> 5] over patterns [127<<23, 129<<23)
    const std::uint32_t* rcp_bits;   // [2^23 >> 5] over patterns [127<<23, 128<<23)
    const std::uint32_t* exp_hard_in; const float* exp_hard_out; int exp_hard_n;   // sorted by input
    const std::uint32_t* rsq_hard_in; const float* rsq_hard_out; int rsq_hard_n;   // canonical [1, 4), sorted
    const std::uint32_t* rcp_hard_in; const float* rcp_hard_out; int rcp_hard_n;   // canonical [1, 2) positive, sorted
    const float2* exp2_tab;           // 2^(n/64), n = -64..64, as (hi, lo) floats (hi + lo within 2^-48 relative)
};

__device__ __forceinline__ bool table_bit(const std::uint32_t* t, std::uint32_t index) {
    return (__ldg(t + (index >> 5)) >> (index & 31U)) & 1U;
}

__device__ __forceinline__ int find_hard(const std::uint32_t* in, int n, std::uint32_t key) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const std::uint32_t v = __ldg(in + mid);
        if (v == key) return mid;
        if (v < key) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

// The neighbour of f on the side of the exact value r (r != f).
__device__ __forceinline__ float other_neighbour(float f, double r) {
    const std::uint32_t b = __float_as_uint(f);
    const bool up = r > static_cast<double>(f);
    // magnitude grows when moving away from zero
    const bool away = (b >> 31) ? !up : up;
    return __uint_as_float(away ? b + 1U : b - 1U);
}

// v_exp_f32 for |x| < 1. The correctly rounded value comes from a float double-float evaluation (error < 2^-37
// relative: 2^(n/64) table, degree-5 series on |r| <= 2^-7); only when the exact value lies within 2^-35 of a rounding
// boundary (about 1 input in 2^10) is the fp64 reference used, so the result is the same as the pure fp64 form.
__device__ __forceinline__ float amd_v_exp_slow(const Tables& t, float x) {
    const std::uint32_t b = __float_as_uint(x), mag = b & 0x7fffffffU;
    const double r = exp2(static_cast<double>(x));
    const float f = static_cast<float>(r);
    if (mag < (102U << 23)) return f;  // |x| < 2^-25: the hardware equals the rounding (probe p2)
    if (t.exp_hard_n) {
        const int h = find_hard(t.exp_hard_in, t.exp_hard_n, b);
        if (h >= 0) return __ldg(t.exp_hard_out + h);
    }
    const std::uint32_t index = (b >> 31) * (25U << 23) + (mag - (102U << 23));  // positive binades, then negative
    return table_bit(t.exp_bits, index) ? other_neighbour(f, r) : f;
}

__device__ __forceinline__ float amd_v_exp(const Tables& t, float x) {
    if (!t.exp2_tab) return amd_v_exp_slow(t, x);
    const std::uint32_t b = __float_as_uint(x), mag = b & 0x7fffffffU;
    const float n = rintf(__fmul_rn(x, 64.0F));
    const float r = __fsub_rn(x, __fmul_rn(n, 0.015625F));  // exact
    const float2 tab = __ldg(t.exp2_tab + static_cast<int>(n) + 64);
    const float lh = __uint_as_float(0x3f317218U), ll = __uint_as_float(0xb102e308U);  // ln 2 = lh + ll
    const float qh = __fmul_rn(r, lh);
    float ql = __fmaf_rn(r, lh, -qh);
    ql = __fmaf_rn(r, ll, ql);
    float p = __fmaf_rn(qh, 1.0F / 120.0F, 1.0F / 24.0F);
    p = __fmaf_rn(qh, p, 1.0F / 6.0F);
    p = __fmaf_rn(qh, p, 0.5F);
    const float tail = __fmul_rn(__fmul_rn(qh, qh), p);
    const float sh = __fadd_rn(1.0F, qh);
    float sl = __fsub_rn(qh, __fsub_rn(sh, 1.0F));
    sl = __fadd_rn(sl, __fadd_rn(ql, tail));
    const float s1 = __fadd_rn(sh, sl);
    const float s2 = __fsub_rn(sl, __fsub_rn(s1, sh));
    const float ph = __fmul_rn(tab.x, s1);
    float pl = __fmaf_rn(tab.x, s1, -ph);
    pl = __fmaf_rn(tab.x, s2, pl);
    pl = __fmaf_rn(tab.y, s1, pl);
    const float f = __fadd_rn(ph, pl);
    const float e = __fsub_rn(pl, __fsub_rn(f, ph));  // (ph + pl) - f, |e| <= half an ulp of f
    // half an ulp of f on the side of e (below a power of two the ulp halves)
    const std::uint32_t fb = __float_as_uint(f);
    const float half_up = __uint_as_float(((fb >> 23) - 24U) << 23);
    const float half = (e < 0.0F && (fb & 0x007fffffU) == 0U) ? __fmul_rn(half_up, 0.5F) : half_up;
    const float bound = __uint_as_float(((fb >> 23) - 35U) << 23);  // 2^-35 relative
    if (__fsub_rn(half, fabsf(e)) <= bound) return amd_v_exp_slow(t, x);  // near a rounding boundary
    if (mag < (102U << 23)) return f;
    const std::uint32_t index = (b >> 31) * (25U << 23) + (mag - (102U << 23));
    if (!table_bit(t.exp_bits, index)) return f;
    if (fabsf(e) <= bound) return amd_v_exp_slow(t, x);  // the side of the exact value is not certain
    return __uint_as_float(((e > 0.0F) == ((fb >> 31) == 0U)) ? fb + 1U : fb - 1U);
}

// v_rsq_f32 for a positive normal input.
__device__ __forceinline__ float amd_v_rsq(const Tables& t, float x) {
    const std::uint32_t b = __float_as_uint(x);
    const int e = static_cast<int>(b >> 23) - 127;
    const int k = e >= 0 ? e / 2 : -((-e + 1) / 2);  // floor(e / 2)
    const std::uint32_t cb = (b & 0x007fffffU) | (static_cast<std::uint32_t>(e - 2 * k + 127) << 23);  // [1, 4)
    const float c = __uint_as_float(cb);
    float y;
    const int h = t.rsq_hard_n ? find_hard(t.rsq_hard_in, t.rsq_hard_n, cb) : -1;
    if (h >= 0) y = __ldg(t.rsq_hard_out + h);
    else {
        // Outside the hard list the fp64 reference rounds to the correctly rounded value, which __frsqrt_rn gives.
        const float f = __frsqrt_rn(c);
        y = f;
        if (table_bit(t.rsq_bits, cb - (127U << 23))) {
            // side of the exact value: sign(1 - c*f^2), with f^2 and c*f^2 split exactly
            const float sh = __fmul_rn(f, f), sl = __fmaf_rn(f, f, -sh);
            const float p = __fmul_rn(c, sh), q = __fmaf_rn(c, sh, -p);
            const float u = __fmul_rn(c, sl), v = __fmaf_rn(c, sl, -u);
            const float d = __fsub_rn(__fsub_rn(__fsub_rn(__fsub_rn(1.0F, p), q), u), v);
            if (fabsf(d) < 0x1p-44F) {
                const double r = 1.0 / sqrt(static_cast<double>(c));
                y = other_neighbour(static_cast<float>(r), r);
            } else {
                y = __uint_as_float(d > 0.0F ? __float_as_uint(f) + 1U : __float_as_uint(f) - 1U);
            }
        }
    }
    // exact power-of-two scaling (the result stays normal for normal inputs)
    return __uint_as_float(static_cast<std::uint32_t>(static_cast<int>(__float_as_uint(y)) - (k << 23)));
}

// v_rcp_f32 for a normal input with |x| < 2^126.
__device__ __forceinline__ float amd_v_rcp(const Tables& t, float x) {
    const std::uint32_t b = __float_as_uint(x);
    const int e = static_cast<int>((b >> 23) & 255U) - 127;
    const std::uint32_t cb = (b & 0x007fffffU) | (127U << 23);  // positive, [1, 2)
    const float c = __uint_as_float(cb);
    float y;
    const int h = t.rcp_hard_n ? find_hard(t.rcp_hard_in, t.rcp_hard_n, cb) : -1;
    if (h >= 0) y = __ldg(t.rcp_hard_out + h);
    else {
        // Outside the hard list the fp64 reference rounds to the correctly rounded value, which __frcp_rn gives;
        // the side of the exact value is the sign of the exact residual 1 - c*f (one fma).
        const float f = __frcp_rn(c);
        y = f;
        if (table_bit(t.rcp_bits, cb - (127U << 23))) {
            const float d = __fmaf_rn(-c, f, 1.0F);
            y = __uint_as_float(d > 0.0F ? __float_as_uint(f) + 1U : __float_as_uint(f) - 1U);
        }
    }
    const std::uint32_t yb = static_cast<std::uint32_t>(static_cast<int>(__float_as_uint(y)) - (e << 23));
    return __uint_as_float(yb | (b & 0x80000000U));
}

// v_ldexp_f32 (exact scaling, IEEE rounding of a subnormal result, overflow to inf).
__device__ __forceinline__ float amd_ldexp(float v, int n) { return scalbnf(v, n); }

// expf: OCML / LLVM AMDGPU lowering at -fno-fast-math (ISA of eval_f, probe p1):
//   ph = x * log2e; e = rndne(ph); pl = fma(x, log2e, -ph); pl = fma(x, log2e_lo, pl); a = (ph - e) + pl;
//   y = ldexp(v_exp(a), (int)e); x < -103.97208 -> 0; x > 88.72284 -> +inf.
__device__ __forceinline__ float amd_expf(const Tables& t, float x) {
    const float kLog2e = __uint_as_float(0x3fb8aa3bU), kLog2eLo = __uint_as_float(0x32a5705fU);
    const float ph = __fmul_rn(x, kLog2e);
    const float e = rintf(ph);
    float pl = __fmaf_rn(x, kLog2e, -ph);
    pl = __fmaf_rn(x, kLog2eLo, pl);
    const float a = __fadd_rn(__fsub_rn(ph, e), pl);
    float y;
    if (fabsf(a) < 1.0F) y = amd_ldexp(amd_v_exp(t, a), static_cast<int>(e));
    else y = a != a ? a : 0.0F;  // only reached for infinite/NaN x (then the selects below decide) [checked by the tests]
    // v_cmp_ngt_f32 vcc, -103.97208, x : vcc = !(K > x)  -> keep y, else 0
    if (__uint_as_float(0xc2ce8ed0U) > x) y = 0.0F;
    // v_cmp_nlt_f32 vcc, 88.72284, x : vcc = !(K < x)  -> keep y, else +inf
    if (__uint_as_float(0x42b17218U) < x) y = __uint_as_float(0x7f800000U);
    return y;
}

// rsqrtf: subnormal inputs scaled by 2^24 before v_rsq_f32 and the result by 2^12.
__device__ __forceinline__ float amd_rsqrtf(const Tables& t, float x) {
    const bool small = __uint_as_float(0x00800000U) > x;  // v_cmp_gt_f32 vcc, 0x800000, x
    const float s = small ? __fmul_rn(x, __uint_as_float(0x4b800000U)) : x;
    float y;
    const std::uint32_t sb = __float_as_uint(s);
    if (sb == 0U) y = __uint_as_float(0x7f800000U);
    else if (sb == 0x80000000U) y = __uint_as_float(0xff800000U);
    else if (sb >= 0x7f800000U) y = sb == 0x7f800000U ? 0.0F : __uint_as_float(0x7fc00000U);
    else y = amd_v_rsq(t, s);
    return small ? __fmul_rn(y, __uint_as_float(0x45800000U)) : y;
}

// log1pf: the OCML double-float algorithm as compiled for gfx1151 (ISA of eval_f, probe p1). Every line is one IEEE
// instruction; VOPD pairs (v_dual_*) read both sources before either writes, kept with temporaries. The single hardware
// approximation is v_rcp_f32 of a value in [1.66, 2.34].
__device__ __forceinline__ float amd_log1pf(const Tables& t, float x) {
    auto add = [](float a, float b) { return __fadd_rn(a, b); };
    auto sub = [](float a, float b) { return __fsub_rn(a, b); };
    auto mul = [](float a, float b) { return __fmul_rn(a, b); };
    auto fma = [](float a, float b, float c) { return __fmaf_rn(a, b, c); };
    const float v1 = x;
    float v0 = add(1.0F, v1);
    int e;
    (void)frexp(static_cast<double>(v0), &e);                 // v_frexp_exp_i32_f64 of cvt_f64(v0)
    int ef;
    float v3 = frexpf(v0, &ef);                               // v_frexp_mant_f32
    if (!(v0 == v0) || v0 == 0.0F || fabsf(v0) == __uint_as_float(0x7f800000U)) { e = 0; v3 = v0; }
    const bool lt = __uint_as_float(0x3f2aaaabU) > v3;
    v3 = add(-1.0F, v0);
    float v5 = sub(v3, v0);
    v3 = sub(v1, v3);
    int v2i = e - (lt ? 1 : 0);
    const int v4i = -v2i;
    const float v2 = static_cast<float>(v2i);
    v0 = scalbnf(v0, v4i);
    v5 = add(1.0F, v5);
    float v6 = add(1.0F, v0);
    v3 = add(v3, v5);
    v3 = scalbnf(v3, v4i);
    float v4 = add(-1.0F, v0);
    v5 = add(-1.0F, v6);
    float v7 = add(1.0F, v4);
    v5 = sub(v0, v5);
    v0 = sub(v0, v7);
    v0 = add(v3, v0);
    v5 = add(v3, v5);
    float v8 = add(v4, v0);
    v7 = add(v6, v5);
    v4 = sub(v8, v4);
    v3 = amd_v_rcp(t, v7);
    v6 = sub(v7, v6);
    v5 = sub(v5, v6);
    v0 = sub(v0, v4);
    const float v9 = mul(v8, v3);
    float v10 = mul(v7, v9);
    v6 = fma(v9, v7, -v10);
    v6 = fma(v9, v5, v6);
    float v11 = add(v10, v6);
    float v12 = sub(v8, v11);
    v8 = sub(v8, v12);
    v4 = sub(v11, v10);
    v8 = sub(v8, v11);
    v4 = sub(v4, v6);
    v0 = add(v0, v8);
    v0 = add(v4, v0);
    v4 = add(v12, v0);
    v6 = mul(v3, v4);
    v11 = sub(v12, v4);
    v8 = mul(v7, v6);
    v0 = add(v0, v11);
    v7 = fma(v6, v7, -v8);
    v7 = fma(v6, v5, v7);
    v5 = add(v8, v7);
    v10 = sub(v4, v5);
    v4 = sub(v4, v10);
    v4 = sub(v4, v5);
    v0 = add(v0, v4);
    v4 = add(v9, v6);
    v8 = sub(v5, v8);
    v5 = sub(v8, v7);
    { const float t0 = add(v5, v0); v5 = sub(v4, v9); v0 = t0; }
    v0 = add(v10, v0);
    v5 = sub(v6, v5);
    v0 = mul(v3, v0);
    v0 = add(v5, v0);
    v3 = add(v4, v0);
    v5 = mul(v3, v3);
    v6 = fma(__uint_as_float(0x3e9b6dacU), v5, __uint_as_float(0x3ecc95a3U));
    v7 = mul(v3, v5);
    v5 = fma(v5, v6, __uint_as_float(0x3f2aaadaU));
    v6 = scalbnf(v3, 1);
    v3 = sub(v3, v4);
    v5 = mul(v7, v5);
    v7 = mul(__uint_as_float(0x3f317218U), v2);
    v0 = sub(v0, v3);
    v4 = add(v6, v5);
    v0 = scalbnf(v0, 1);
    v3 = sub(v4, v6);
    v6 = fma(__uint_as_float(0x3f317218U), v2, -v7);
    v3 = sub(v5, v3);
    v0 = add(v0, v3);
    float w2 = fma(v2, __uint_as_float(0xb102e308U), v6);
    v5 = add(v4, v0);
    v3 = add(v7, w2);
    v4 = sub(v5, v4);
    v7 = sub(v3, v7);
    v0 = sub(v0, v4);
    w2 = sub(w2, v7);
    v6 = add(v3, v5);
    v8 = sub(v6, v3);
    v4 = sub(v5, v8);
    v5 = add(w2, v0);
    const float w9 = sub(v6, v8);
    v3 = sub(v3, w9);
    { const float t0 = add(v4, v3); v4 = sub(v5, w2); v3 = t0; }
    v3 = add(v5, v3);
    v5 = sub(v5, v4);
    v0 = sub(v0, v4);
    v7 = add(v6, v3);
    w2 = sub(w2, v5);
    v4 = sub(v7, v6);
    v0 = add(v0, w2);
    w2 = sub(v3, v4);
    v0 = add(v0, w2);
    v0 = add(v7, v0);
    if (!(v1 != __uint_as_float(0x7f800000U))) v0 = __uint_as_float(0x7f800000U);
    if (-1.0F > v1) v0 = __uint_as_float(0x7fc00000U);
    if (!(v1 != -1.0F)) v0 = __uint_as_float(0xff800000U);
    if (__uint_as_float(0x33800000U) > fabsf(v1)) v0 = v1;
    return v0;
}

// Correctly rounded IEEE operations (the AMD division and sqrtf are correctly rounded at the engine's flags).
__device__ __forceinline__ float amd_div(float a, float b) { return __fdiv_rn(a, b); }
__device__ __forceinline__ float amd_sqrtf(float x) { return __fsqrt_rn(x); }

}  // namespace strix::cuda::amdmath
