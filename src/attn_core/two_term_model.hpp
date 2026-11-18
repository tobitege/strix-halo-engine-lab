#pragma once
// Host model of the 16-bit WMMA operands of the shared attention cores, one term (tile::to_operand) and two terms
// (tile::to_operand2: hi = RNE(x), lo = RNE(x - hi)), and of the dense core's products in their real order
// (dense_flash_prefill.hpp flash_body, kTerms 1 / 2). Diagnostic and test oracle only: no device path includes it.
//
// Operand error. RNE, finite x below the format's overflow threshold; u = unit roundoff (f16 2^-11, bf16 2^-8),
// h = half the smallest subnormal step (f16 2^-25; bf16 has FP32's exponent range, 2^-150):
//   one term   |x - hi|      <= max(u |x|, h)
//   two terms  |x - hi - lo| <= max(u |x - hi|, h) <= max(u^2 |x|, h)
// The relative form holds only while the remainder x - hi is a normal number of the format. In f16 a remainder
// below 2^-14 is subnormal: the error floor is h = 2^-25 absolute, so the gain over one term shrinks below |x| ~ 2^-3
// and vanishes for |x| < 2^-14 (hi itself subnormal, lo = 0). f16 overflows at |x| >= 65520 (hi infinite; the cores
// flag it). bf16 two terms carry 16 significant bits: u^2 = 2^-16, not the f16 figure.
//
// One product a.b of two-term operands as the cores form it, hi.hi + hi.lo + lo.hi (lo.lo dropped):
//   a.b - device = ah eb + ea bh + ra rb   (ra = a - ah, ea = ra - al; the same for b)
//   |.| <= (|a| + ra) max(u^2|b|, h) + max(u^2|a|, h)(|b| + rb) + ra rb,  ra <= max(u|a|, h)   ~ 3 u^2 |a b|
// i.e. about 2^-20.4 |ab| in f16 and 2^-14.4 |ab| in bf16 in the normal range (one term: about 2u |ab|).
// A dot product adds the FP32 rounding of every WMMA step of its chain (model: the 16 products of a step are exact,
// the step's sum is rounded once to FP32; RDNA3 does not document the internal order): about steps x 2^-24 x
// sum |a_i b_i|. Both parts are relative to sum |a_i b_i|, not to |sum a_i b_i|: under cancellation the score's
// relative error is unbounded with one or two terms; two terms only remove the operand part. There is no
// format-independent 2^-17 figure.
//
// Order (dense core, one D piece, Split 1 - the vision tower's Shape<DP, 1, 1, 64>):
//   QK  D in 16-wide steps; chain s takes the even steps, chain s1 the odd ones; per step at two terms
//       s += q_hi.k_lo, s += q_lo.k_hi, s += q_hi.k_hi (the small products first); score = (s + s1) * kScaleLog2.
//   P   per 16-key tile: m = max(running max, tile scores), alpha = exp2(old max - m), p = exp2(score - m),
//       l = l * alpha + sum_j (p_hi + p_lo) in key order.
//   PV  per output element and tile: c = acc * alpha; c += p_hi.v_lo; c += p_lo.v_hi; acc = c + p_hi.v_hi.
//   out = acc / l.
// The device's exp2 is the hardware approximation (about 1 ulp); the model uses std::exp2. Split-K merges
// (splits > 1) and a policy's key placement (the vision tower's cyclic bidirectional slots) are not modelled:
// the model walks the keys it is given, 16 per tile, in order.
// Agreement is diagnostic evidence, not a bitwise device oracle or a proof of the full numerical bounds.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace strix::attn_core::model {

enum class Format { f16, bf16 };

inline double unit_roundoff(Format f) { return f == Format::f16 ? 0x1p-11 : 0x1p-8; }
inline double half_min_step(Format f) { return f == Format::f16 ? 0x1p-25 : 0x1p-150; }

// Bit-based FP16 conversion shared by host diagnostics; independent of the floating-point rounding mode.
inline float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
    std::uint32_t exp = (h >> 10U) & 0x1FU, man = h & 0x3FFU, bits;
    if (exp == 0U) {
        if (man == 0U) {
            bits = sign;
        } else {
            unsigned shift = 0;
            while ((man & 0x400U) == 0U) {
                man <<= 1U;
                ++shift;
            }
            bits = sign | ((113U - shift) << 23U) | ((man & 0x3FFU) << 13U);
        }
    } else if (exp == 31U) {
        bits = sign | 0x7F800000U | (man << 13U);
    } else {
        bits = sign | ((exp + 112U) << 23U) | (man << 13U);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline float round_to_f16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const auto sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
    const std::uint32_t a = bits & 0x7FFFFFFFU;
    if (a >= 0x7F800000U) return f;  // preserve infinities and NaNs
    if (a >= 0x477FF000U) return half_to_float(static_cast<std::uint16_t>(sign | 0x7C00U));
    if (a < 0x33000000U) return half_to_float(sign);  // below half the least FP16 subnormal
    const unsigned exponent = a >> 23U;
    const unsigned shift = exponent < 113U ? 126U - exponent : 13U;
    const std::uint32_t mantissa = (a & 0x7FFFFFU) | 0x800000U;
    const std::uint32_t remainder = mantissa & ((1U << shift) - 1U);
    const std::uint32_t halfway = 1U << (shift - 1U);
    std::uint32_t rounded = mantissa >> shift;
    rounded += remainder > halfway || (remainder == halfway && (rounded & 1U));
    if (exponent >= 113U) rounded += (exponent - 113U) << 10U;
    return half_to_float(static_cast<std::uint16_t>(sign | rounded));
}

// RNE to the operand format, as a float (f16: subnormals kept, |x| >= 65520 -> infinity; bf16: FP32 bits).
inline float round_operand(float x, Format f) {
    if (!std::isfinite(x)) return x;
    if (f == Format::bf16) {
        std::uint32_t u;
        std::memcpy(&u, &x, 4);
        u = (u + 0x7FFFU + ((u >> 16) & 1U)) & 0xFFFF0000U;
        float r;
        std::memcpy(&r, &u, 4);
        return r;
    }
    return round_to_f16(x);
}

struct Terms {
    float hi, lo;
};
// terms 1: {RNE(x), 0}; terms 2: {RNE(x), RNE(x - hi)} (x - hi is exact in FP32); an infinite hi keeps lo 0.
inline Terms split(float x, Format f, unsigned terms) {
    const float hi = round_operand(x, f);
    if (terms < 2U || !std::isfinite(hi)) return Terms{hi, 0.0F};
    return Terms{hi, round_operand(x - hi, f)};
}

// A-priori bound of |x - (hi + lo)| (see the header): infinity at overflow.
inline double operand_bound(double x, Format f, unsigned terms) {
    if (f == Format::f16 && std::fabs(x) >= 65520.0) return std::numeric_limits<double>::infinity();
    const double u = unit_roundoff(f), h = half_min_step(f), a = std::fabs(x);
    return terms < 2U ? std::fmax(u * a, h) : std::fmax(u * u * a, h);
}

// The value of one product as the device forms it: hi.hi (+ hi.lo + lo.hi at two terms), exact in double.
inline double device_product(const Terms& a, const Terms& b) {
    return double(a.hi) * b.hi + double(a.hi) * b.lo + double(a.lo) * b.hi;
}
// A-priori bound of |a b - device_product| from the magnitudes alone (see the header).
inline double product_bound(double a, double b, Format f, unsigned terms) {
    const double u = unit_roundoff(f), h = half_min_step(f), x = std::fabs(a), y = std::fabs(b);
    const double rx = std::fmax(u * x, h), ry = std::fmax(u * y, h);  // |a - ah|, |b - bh|; |ah| <= |a| + rx
    if (terms < 2U) return (x + rx) * ry + rx * (y + ry) + rx * ry;
    return (x + rx) * operand_bound(b, f, 2U) + operand_bound(a, f, 2U) * (y + ry) + rx * ry;
}
// FP32 roundings in the longest QK chain of a dp-wide dot (WMMA steps + the chain merge).
inline unsigned dot_roundings(unsigned dp, unsigned terms) {
    const unsigned steps = (dp + 15U) / 16U;
    return (terms < 2U ? 1U : 3U) * ((steps + 1U) / 2U) + 1U;
}

// The dense core's score dot q . k (before kScaleLog2) of split operands, dp dims (zero padding adds nothing),
// real chain order. terms 1 operands have lo = 0, so the same code is the one-term chain without the small steps.
inline float dense_dot(const Terms* q, const Terms* k, unsigned dp, unsigned terms) {
    float chain[2] = {0.0F, 0.0F};
    for (unsigned step = 0; 16U * step < dp; ++step) {
        double hh = 0.0, hl = 0.0, lh = 0.0;
        for (unsigned d = 16U * step; d < dp && d < 16U * step + 16U; ++d) {
            const Terms a = q[d], b = k[d];
            hh += double(a.hi) * b.hi;
            hl += double(a.hi) * b.lo;
            lh += double(a.lo) * b.hi;
        }
        float& s = chain[step % 2U];
        if (terms >= 2U) {
            s = static_cast<float>(double(s) + hl);
            s = static_cast<float>(double(s) + lh);
        }
        s = static_cast<float>(double(s) + hh);
    }
    return chain[0] + chain[1];
}

// One (query row, head) of the dense core over nk keys given in order: q and key(j) / value(j) are split operands
// (split(x, f, terms)) [dp]; P is split here.
template <class KeyAt, class ValueAt>
void dense_row(const Terms* q, unsigned nk, KeyAt key, ValueAt value, unsigned dp, float scale_log2, unsigned terms, Format f,
               float* out) {
    std::vector<float> acc(dp, 0.0F);
    float maximum = -std::numeric_limits<float>::infinity(), denominator = 0.0F;
    for (unsigned kt = 0; kt < nk; kt += 16U) {
        const unsigned n = nk - kt < 16U ? nk - kt : 16U;
        float sc[16];
        float m = maximum;
        for (unsigned i = 0; i < 16U; ++i) {
            sc[i] = i < n ? dense_dot(q, key(kt + i), dp, terms) * scale_log2 : -std::numeric_limits<float>::infinity();
            m = sc[i] > m ? sc[i] : m;
        }
        const float alpha = std::exp2(maximum - m);
        Terms p[16];
        float sum = 0.0F;
        for (unsigned j = 0; j < 16U; ++j) {
            p[j] = split(std::exp2(sc[j] - m), f, terms);
            sum += p[j].hi + p[j].lo;
        }
        denominator = denominator * alpha + sum;
        maximum = m;
        for (unsigned d = 0; d < dp; ++d) {
            double hh = 0.0, hl = 0.0, lh = 0.0;
            for (unsigned j = 0; j < n; ++j) {
                const Terms v = value(kt + j)[d];
                hh += double(p[j].hi) * v.hi;
                hl += double(p[j].hi) * v.lo;
                lh += double(p[j].lo) * v.hi;
            }
            float c = acc[d] * alpha;
            if (terms >= 2U) {
                c = static_cast<float>(double(c) + hl);
                c = static_cast<float>(double(c) + lh);
            }
            acc[d] = static_cast<float>(double(c) + hh);
        }
    }
    for (unsigned d = 0; d < dp; ++d) out[d] = acc[d] / denominator;
}

}  // namespace strix::attn_core::model
