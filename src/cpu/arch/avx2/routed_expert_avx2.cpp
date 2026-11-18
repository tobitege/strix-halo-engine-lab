// AVX2 / AVX-VNNI row kernels of the host routed-expert executor. Compile with /arch:AVX2 (MSVC) or -mavx2 -mavxvnni.
// Integer group dots are exact in any order (u4/u5 x int8 via maddubs+madd, or vpdpbusd; Q8_0 signed x signed via the sign
// trick), so only the float chain defines the bits; it is vectorised across the eight groups of a super-block (gate/up)
// or the four chains (down) with the same single IEEE operations as the scalar oracle (mul/add/sub, never FMA).
#include "cpu/arch/avx2/routed_expert_avx2.hpp"

#if (defined(__AVX2__) || defined(__AVX512F__)) && (defined(__x86_64__) || defined(_M_X64))
#define STRIX_EXPERT_AVX2 1
#include <immintrin.h>

#include <cstring>
#if defined(_MSC_VER)
#include <intrin.h>
#pragma fp_contract(off)
#else
#include <cpuid.h>
#endif

namespace strix::cpu::arch::avx2 {
namespace {

using namespace strix::cpu::expert;
using namespace strix::cpu::expert::detail;
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using i8 = std::int8_t;
using i32 = std::int32_t;

bool has_avxvnni() {
#if defined(_MSC_VER)
    int v[4];
    __cpuidex(v, 7, 1);
    return (static_cast<unsigned>(v[0]) >> 4) & 1U;
#else
    unsigned r[4] = {0, 0, 0, 0};
    __cpuid_count(7, 1, r[0], r[1], r[2], r[3]);
    return (r[0] >> 4) & 1U;
#endif
}

// u8 weights (<= 31, or abs of an int8) x s8 codes, 8 int32 partial sums.
template <bool VNNI>
inline __m256i dot_u8s8(__m256i w, __m256i c) {
    if constexpr (VNNI) {
        return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), w, c);
    } else {
        return _mm256_madd_epi16(_mm256_maddubs_epi16(w, c), _mm256_set1_epi16(1));
    }
}
template <bool VNNI>
inline __m256i dot_s8s8(__m256i w, __m256i c) { return dot_u8s8<VNNI>(_mm256_sign_epi8(w, w), _mm256_sign_epi8(c, w)); }

inline float half_f(u16 h) { return half_to_float(h); }

template <bool Q5K, bool VNNI>
void gate_row(const u8* wrow, u32 blocks, const RowA8* const* rows, u32 na, float* out) {
    constexpr u32 WB = Q5K ? 176 : 144, QS = Q5K ? 48 : 16;
    __m256 acc[kMaxAssign];
    for (u32 a = 0; a < na; ++a) acc[a] = _mm256_setzero_ps();
    const __m256i m0f = _mm256_set1_epi8(0x0f), one = _mm256_set1_epi8(1);
    for (u32 b = 0; b < blocks; ++b) {
        const u8* blk = wrow + static_cast<std::size_t>(b) * WB;
        u16 dh, dmh;
        std::memcpy(&dh, blk, 2); std::memcpy(&dmh, blk + 2, 2);
        // factor() of the CUDA kernel: 6-bit scales and mins of the 8 groups
        u32 y, z, w;
        std::memcpy(&y, blk + 4, 4); std::memcpy(&z, blk + 8, 4); std::memcpy(&w, blk + 12, 4);
        alignas(16) u8 sc[16], mn[16];
        for (u32 j = 0; j < 8; ++j) {
            const unsigned shift = 8U * (j & 3U);
            const unsigned a = (y >> shift) & 255U, bb = (z >> shift) & 255U, c = (w >> shift) & 255U;
            sc[j] = static_cast<u8>(j < 4U ? a & 63U : (c & 15U) | ((a >> 6U) << 4U));
            mn[j] = static_cast<u8>(j < 4U ? bb & 63U : (c >> 4U) | ((bb >> 6U) << 4U));
        }
        const __m256 Dv = _mm256_mul_ps(_mm256_set1_ps(half_f(dh)), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(sc)))));
        const __m256 DMv = _mm256_mul_ps(_mm256_set1_ps(half_f(dmh)), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(mn)))));
        __m256i wv[8];
        for (u32 p = 0; p < 4; ++p) {
            const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + QS + 32 * p));
            __m256i lo = _mm256_and_si256(q, m0f), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m0f);
            if constexpr (Q5K) {
                const __m256i qh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 16));
                lo = _mm256_or_si256(lo, _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(static_cast<int>(2 * p))), one), 4));
                hi = _mm256_or_si256(hi, _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(static_cast<int>(2 * p + 1))), one), 4));
            }
            wv[2 * p] = lo; wv[2 * p + 1] = hi;
        }
        for (u32 a = 0; a < na; ++a) {
            const i8* c = rows[a]->codes + 256 * static_cast<std::size_t>(b);
            __m256i v[8];
            for (u32 j = 0; j < 8; ++j)
                v[j] = dot_u8s8<VNNI>(wv[j], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + 32 * j)));
            const __m256i e = _mm256_hadd_epi32(_mm256_hadd_epi32(v[0], v[1]), _mm256_hadd_epi32(v[2], v[3]));
            const __m256i f = _mm256_hadd_epi32(_mm256_hadd_epi32(v[4], v[5]), _mm256_hadd_epi32(v[6], v[7]));
            const __m256i dots = _mm256_add_epi32(_mm256_permute2x128_si256(e, f, 0x20), _mm256_permute2x128_si256(e, f, 0x31));
            const __m256 S = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(rows[a]->sums + 8 * static_cast<std::size_t>(b))));
            const __m256 E = _mm256_loadu_ps(rows[a]->deltas + 8 * static_cast<std::size_t>(b));
            const __m256 u = _mm256_mul_ps(Dv, _mm256_cvtepi32_ps(dots));
            const __m256 vv = _mm256_mul_ps(DMv, S);
            const __m256 x = _mm256_mul_ps(E, _mm256_sub_ps(u, vv));
            acc[a] = _mm256_add_ps(acc[a], _mm256_hadd_ps(x, x));  // lanes 0,1 = pairs 0,1; lanes 4,5 = pairs 2,3
        }
    }
    for (u32 a = 0; a < na; ++a) {
        alignas(32) float t[8];
        _mm256_store_ps(t, acc[a]);
        out[a] = (t[0] + t[1]) + (t[4] + t[5]);
    }
}

// four 32-weight dots -> four int32 (group order 0..3)
inline __m128i reduce4(__m256i a, __m256i b, __m256i c, __m256i d) {
    const __m256i t = _mm256_hadd_epi32(_mm256_hadd_epi32(a, b), _mm256_hadd_epi32(c, d));
    return _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
}

template <bool Q51, bool VNNI>
void down_row(const u8* wrow, u32 groups, const i8* const* codes, const float* const* deltas, const i32* const* sums, u32 na,
              float* out) {
    constexpr u32 WB = Q51 ? 24 : 34;
    __m128 acc[kMaxAssign];
    for (u32 a = 0; a < na; ++a) acc[a] = _mm_setzero_ps();
    const __m256i m0f = _mm256_set1_epi8(0x0f), hbit = _mm256_set1_epi8(0x10);
    const __m256i bitmask = _mm256_set1_epi64x(static_cast<long long>(0x8040201008040201ULL));
    const __m256i spread = _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL, 0x0101010101010101LL, 0x0000000000000000LL);
    for (u32 c4 = 0; c4 < groups / 4; ++c4) {
        __m256i wv[4];
        alignas(16) u16 dd[8] = {0, 0, 0, 0, 0, 0, 0, 0}, mm[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (u32 j = 0; j < 4; ++j) {
            const u8* blk = wrow + static_cast<std::size_t>(4 * c4 + j) * WB;
            std::memcpy(&dd[j], blk, 2);
            if constexpr (Q51) {
                u32 qh;
                std::memcpy(&mm[j], blk + 2, 2);
                std::memcpy(&qh, blk + 4, 4);
                const __m128i qs = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 8));
                const __m256i q = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(qs, 4), _mm256_castsi256_si128(m0f)), _mm_and_si128(qs, _mm256_castsi256_si128(m0f)));
                const __m256i sel = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(qh)), spread), bitmask);
                wv[j] = _mm256_or_si256(q, _mm256_and_si256(_mm256_cmpeq_epi8(sel, bitmask), hbit));
            } else {
                wv[j] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 2));
            }
        }
        const __m128 dv = _mm_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(dd)));
        const __m128 mv = Q51 ? _mm_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(mm))) : _mm_setzero_ps();
        for (u32 a = 0; a < na; ++a) {
            const i8* c = codes[a] + 128 * static_cast<std::size_t>(c4);
            __m256i v[4];
            for (u32 j = 0; j < 4; ++j) {
                const __m256i cv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + 32 * j));
                v[j] = Q51 ? dot_u8s8<VNNI>(wv[j], cv) : dot_s8s8<VNNI>(wv[j], cv);
            }
            const __m128 first = _mm_mul_ps(dv, _mm_cvtepi32_ps(reduce4(v[0], v[1], v[2], v[3])));
            __m128 combined = first;
            if constexpr (Q51) {
                const __m128 S = _mm_cvtepi32_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sums[a] + 4 * static_cast<std::size_t>(c4))));
                combined = _mm_add_ps(first, _mm_mul_ps(mv, S));
            }
            const __m128 E = _mm_loadu_ps(deltas[a] + 4 * static_cast<std::size_t>(c4));
            acc[a] = _mm_add_ps(acc[a], _mm_mul_ps(E, combined));
        }
    }
    for (u32 a = 0; a < na; ++a) {
        alignas(16) float t[4];
        _mm_store_ps(t, acc[a]);
        out[a] = (t[0] + t[1]) + (t[2] + t[3]);
    }
}

template <bool VNNI>
Kernels make(const char* name) {
    Kernels k;
    k.name = name;
    k.gate[0] = gate_row<false, VNNI>; k.gate[1] = gate_row<true, VNNI>;
    k.down[0] = down_row<true, VNNI>;  k.down[1] = down_row<false, VNNI>;
    return k;
}

}  // namespace

const Kernels* expert_kernels() noexcept {
    static const Kernels plain = make<false>("avx2");
    static const Kernels vnni = make<true>("avx2+vnni");
    static const bool use_vnni = has_avxvnni();
    return use_vnni ? &vnni : &plain;
}

}  // namespace strix::cpu::arch::avx2

#else  // not compiled for AVX2: no table, the executor keeps the scalar one

namespace strix::cpu::arch::avx2 {
const expert::detail::Kernels* expert_kernels() noexcept { return nullptr; }
}

#endif
