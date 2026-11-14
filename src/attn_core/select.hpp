#pragma once
// Exact ordered radix selection. Policies retain the original CTA/scan/compaction schedules.
// Callers own LDS so each kernel keeps its declaration order, sizes and launch attributes.
#include <hip/hip_runtime.h>
#include <cstdint>

namespace strix::attn_core::select {
enum class Variant { Chunked, Segmented, FourPerThread };
template <unsigned Pool, bool GuardEmpty> struct ChunkPolicy {
    static constexpr Variant kVariant = Variant::Chunked;
    static constexpr unsigned kKpool = Pool;
    static constexpr bool kGuardEmpty = GuardEmpty;
};
template <unsigned Pool> struct SegmentPolicy {
    static constexpr Variant kVariant = Variant::Segmented;
    using Key = std::uint32_t;
    static constexpr unsigned kThreads = 1024;
    static constexpr unsigned kKpool = Pool;
};
template <unsigned Pool, unsigned Threads, unsigned Waves> struct QuadPolicy {
    static constexpr Variant kVariant = Variant::FourPerThread;
    using Key = unsigned;
    static constexpr unsigned kKpool = Pool, kThreads = Threads, kWaves = Waves;
};
struct ChunkStorage {
    unsigned* hist;
    unsigned &sh_prefix, &sh_mask, &sh_remaining, &sh_carry_eq, &sh_carry_out;
    unsigned *scan_eq, *scan_sel;
};
struct SegmentStorage {
    unsigned *hist, *wsum;
    unsigned &sh_prefix, &sh_rem;
};
struct QuadStorage {
    unsigned *hist, *ws;
    unsigned &sh_digit, &sh_rem;
};

// Whole-row prefill shortcut, including the unpooled tail.
template <unsigned Threads>
__device__ __forceinline__ void all_tokens(unsigned l, unsigned t, std::uint32_t* tokens, std::uint32_t* n_out) {
    for (unsigned pos = l; pos <= t; pos += Threads) tokens[pos] = pos;
    if (l == 0) *n_out = t + 1;
}

template <class Policy>
__device__ __forceinline__ unsigned block_excl_scan_1024(unsigned v, unsigned* wsum) {
    const unsigned l = threadIdx.x, lane = l & 31U, w = l >> 5U;
    unsigned x = v;
#pragma unroll
    for (unsigned off = 1; off < 32; off <<= 1U) {
        const unsigned y = __shfl_up(x, off, 32);
        if (lane >= off) x += y;
    }
    __syncthreads();
    if (lane == 31) wsum[w] = x;
    __syncthreads();
    if (w == 0) {
        unsigned s = wsum[lane];
#pragma unroll
        for (unsigned off = 1; off < 32; off <<= 1U) {
            const unsigned y = __shfl_up(s, off, 32);
            if (lane >= off) s += y;
        }
        wsum[lane] = s;  // inclusive totals per wave
    }
    __syncthreads();
    return (w ? wsum[w - 1] : 0U) + x - v;
}

template <class Policy>
__device__ __forceinline__ unsigned sel2_excl_scan(unsigned v, unsigned* ws, unsigned& total) {
    const unsigned lane = threadIdx.x % 32, w = threadIdx.x / 32;
    unsigned x = v;
#pragma unroll
    for (unsigned off = 1; off < 32; off <<= 1U) {
        const unsigned y = __shfl_up(x, off, 32);
        if (lane >= off) x += y;
    }
    if (lane == 31) ws[w] = x;
    __syncthreads();
    unsigned before = 0, tot = 0;
#pragma unroll
    for (unsigned i = 0; i < Policy::kWaves; ++i) {
        const unsigned s = ws[i];
        tot += s;
        before += i < w ? s : 0U;
    }
    __syncthreads();  // ws is reused by the next scan
    total = tot;
    return before + x - v;
}

// Keep the uncached chunked load form separate from the cached-key row schedules.
template <class Policy>
__device__ __forceinline__ void histogram_pass(const std::uint32_t* keys, unsigned P, unsigned l,
                                              unsigned prefix, unsigned mask, int shift, unsigned* hist) {
    if constexpr (Policy::kVariant == Variant::Chunked) {
        for (unsigned p = l; p < P; p += 1024)
            if ((keys[p] & mask) == prefix) atomicAdd(&hist[(keys[p] >> shift) & 255U], 1U);
    } else {
        for (unsigned p = l; p < P; p += Policy::kThreads) {
            const typename Policy::Key key = keys[p];
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255U], 1U);
        }
    }
}

// Called only by lane zero, preserving when the selected count is read.
template <class Policy>
__device__ __forceinline__ void append_tail(unsigned P, unsigned t, unsigned count,
                                           std::uint32_t* tokens, std::uint32_t* n_out) {
    unsigned n = count * Policy::kKpool;
    for (unsigned pos = P * Policy::kKpool; pos <= t; ++pos) tokens[n++] = pos;
    *n_out = n;
}

template <class Policy>
__device__ __forceinline__ void chunked_passes(const std::uint32_t* keys, unsigned P, unsigned l,
                                              std::uint32_t* tokens, ChunkStorage storage) {
    unsigned* hist = storage.hist;
    unsigned &sh_prefix = storage.sh_prefix, &sh_mask = storage.sh_mask, &sh_remaining = storage.sh_remaining;
    unsigned &sh_carry_eq = storage.sh_carry_eq, &sh_carry_out = storage.sh_carry_out;
    unsigned *scan_eq = storage.scan_eq, *scan_sel = storage.scan_sel;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (unsigned d = l; d < 256; d += 1024) hist[d] = 0;
        __syncthreads();
        const unsigned prefix = sh_prefix, mask = sh_mask;
        histogram_pass<Policy>(keys, P, l, prefix, mask, shift, hist);
        __syncthreads();
        if (l == 0) {
            unsigned rem = sh_remaining;
            for (int d = 255; d >= 0; --d) {
                if (hist[d] >= rem) {
                    sh_prefix = prefix | (static_cast<unsigned>(d) << shift);
                    break;
                }
                rem -= hist[d];
            }
            sh_remaining = rem;  // pools with key == K* still to take (lowest indices first)
            sh_mask = mask | (255U << shift);
        }
        __syncthreads();
    }
    const unsigned kstar = sh_prefix, take_eq = sh_remaining;
    for (unsigned base = 0; base < P; base += 1024) {  // ordered compaction, chunk by chunk
        const unsigned p = base + l;
        const unsigned key = p < P ? keys[p] : 0U;
        const unsigned eq = (p < P && key == kstar) ? 1U : 0U;
        scan_eq[l] = eq;
        __syncthreads();
        for (unsigned off = 1; off < 1024; off <<= 1U) {  // inclusive scan (Hillis-Steele)
            const unsigned v = l >= off ? scan_eq[l - off] : 0U;
            __syncthreads();
            scan_eq[l] += v;
            __syncthreads();
        }
        const unsigned eq_rank = sh_carry_eq + scan_eq[l] - eq;  // equal keys before p
        const unsigned sel = (p < P && (key > kstar || (eq && eq_rank < take_eq))) ? 1U : 0U;
        scan_sel[l] = sel;
        __syncthreads();
        for (unsigned off = 1; off < 1024; off <<= 1U) {
            const unsigned v = l >= off ? scan_sel[l - off] : 0U;
            __syncthreads();
            scan_sel[l] += v;
            __syncthreads();
        }
        if (sel) {
            const unsigned at = sh_carry_out + scan_sel[l] - 1;
            for (unsigned j = 0; j < Policy::kKpool; ++j) tokens[at * Policy::kKpool + j] = p * Policy::kKpool + j;
        }
        __syncthreads();
        if (l == 1023) {
            sh_carry_eq += scan_eq[1023];
            sh_carry_out += scan_sel[1023];
        }
        __syncthreads();
    }
}

// One entry point; the three schedules are distinct algorithmic variants, not engine copies.
// Row addressing and LDS ownership stay in the thin kernel adapters.
template <class Policy, class Storage>
__device__ __forceinline__ void select_row(const std::uint32_t* keys, unsigned P, unsigned t, unsigned k,
                                          unsigned l, std::uint32_t* tokens, std::uint32_t* n_out, Storage storage,
                                          const std::uint32_t* hist0 = nullptr, unsigned r = 0) {
    if constexpr (Policy::kVariant == Variant::Chunked) {
        unsigned &sh_prefix = storage.sh_prefix, &sh_mask = storage.sh_mask, &sh_remaining = storage.sh_remaining;
        unsigned &sh_carry_eq = storage.sh_carry_eq, &sh_carry_out = storage.sh_carry_out;
        if (l == 0) {
            sh_prefix = 0;
            sh_mask = 0;
            sh_remaining = k;
            sh_carry_eq = 0;
            sh_carry_out = 0;
        }
        __syncthreads();
        if constexpr (Policy::kGuardEmpty) {
            if (k > 0) chunked_passes<Policy>(keys, P, l, tokens, storage);
        } else {
            chunked_passes<Policy>(keys, P, l, tokens, storage);
        }
        if (l == 0) append_tail<Policy>(P, t, sh_carry_out, tokens, n_out);
    } else if constexpr (Policy::kVariant == Variant::Segmented) {
        unsigned *hist = storage.hist, *wsum = storage.wsum;
        unsigned &sh_prefix = storage.sh_prefix, &sh_rem = storage.sh_rem;
        if (k == P) {  // every visible pool is selected
            for (unsigned i = l; i < P * Policy::kKpool; i += 1024) tokens[i] = i;
        } else {
            unsigned prefix = 0, mask = 0, rem = k;
            for (int shift = 24; shift >= 0; shift -= 8) {
                __syncthreads();
                if (shift == 24) {
                    if (l < 256) hist[l] = hist0[r * 256 + l];
                } else {
                    if (l < 256) hist[l] = 0;
                    __syncthreads();
                    histogram_pass<Policy>(keys, P, l, prefix, mask, shift, hist);
                }
                __syncthreads();
                const unsigned v = l < 256 ? hist[255 - l] : 0U;  // digits from the top
                const unsigned above = block_excl_scan_1024<Policy>(v, wsum);  // keys (under the prefix) with a larger digit
                if (l < 256 && above < rem && above + v >= rem) {
                    sh_prefix = prefix | (static_cast<unsigned>(255 - l) << shift);
                    sh_rem = rem - above;
                }
                __syncthreads();
                prefix = sh_prefix;
                rem = sh_rem;
                mask |= 255U << shift;
            }
            const unsigned kstar = prefix, take_eq = rem;
            const unsigned seg = (P + 1023) / 1024, b0 = l * seg < P ? l * seg : P, b1 = b0 + seg < P ? b0 + seg : P;
            unsigned ce = 0;
            for (unsigned p = b0; p < b1; ++p) ce += keys[p] == kstar ? 1U : 0U;
            const unsigned eq_base = block_excl_scan_1024<Policy>(ce, wsum);
            unsigned cs = 0, er = eq_base;
            for (unsigned p = b0; p < b1; ++p) {
                const std::uint32_t key = keys[p];
                const bool eq = key == kstar;
                cs += (key > kstar || (eq && er < take_eq)) ? 1U : 0U;
                er += eq ? 1U : 0U;
            }
            unsigned o = block_excl_scan_1024<Policy>(cs, wsum);
            er = eq_base;
            for (unsigned p = b0; p < b1; ++p) {
                const std::uint32_t key = keys[p];
                const bool eq = key == kstar;
                if (key > kstar || (eq && er < take_eq)) {
                    for (unsigned j = 0; j < Policy::kKpool; ++j) tokens[o * Policy::kKpool + j] = p * Policy::kKpool + j;
                    ++o;
                }
                er += eq ? 1U : 0U;
            }
        }
        if (l == 0) append_tail<Policy>(P, t, k, tokens, n_out);
    } else {
        static_assert(Policy::kVariant == Variant::FourPerThread);
        constexpr unsigned kSel2Threads = Policy::kThreads;
        unsigned *hist = storage.hist, *ws = storage.ws;
        unsigned &sh_digit = storage.sh_digit, &sh_rem = storage.sh_rem;
        unsigned prefix = 0, mask = 0, rem = k;
        for (int shift = 24; shift >= 0; shift -= 8) {
            hist[l] = 0;
            __syncthreads();
            histogram_pass<Policy>(keys, P, l, prefix, mask, shift, hist);
            __syncthreads();
            // thread l holds digit d = 255 - l; cum = count of matching keys with digit >= d. The chosen digit is the one
            // where cum first reaches rem (= the descending walk of the v1 kernel); rem becomes rem - (cum - h).
            const unsigned h = hist[255 - l];
            unsigned tot;
            const unsigned above = sel2_excl_scan<Policy>(h, ws, tot);
            if (above < rem && above + h >= rem) {
                sh_digit = 255 - l;
                sh_rem = rem - above;
            }
            __syncthreads();
            prefix |= sh_digit << shift;
            mask |= 255U << shift;
            rem = sh_rem;
            __syncthreads();  // sh_* and hist are rewritten by the next pass
        }
        const unsigned kstar = prefix, take_eq = rem;
        unsigned carry_eq = 0, carry_out = 0;
        for (unsigned base = 0; base < P; base += 4 * kSel2Threads) {
            unsigned key[4], eq_n = 0;
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const unsigned p = base + 4 * l + i;
                key[i] = p < P ? keys[p] : 0U;
                eq_n += (p < P && key[i] == kstar) ? 1U : 0U;
            }
            unsigned eq_tot;
            unsigned eq_rank = carry_eq + sel2_excl_scan<Policy>(eq_n, ws, eq_tot);
            bool sel[4];
            unsigned sel_n = 0;
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const unsigned p = base + 4 * l + i;
                const bool eq = p < P && key[i] == kstar;
                sel[i] = p < P && (key[i] > kstar || (eq && eq_rank < take_eq));
                eq_rank += eq ? 1U : 0U;
                sel_n += sel[i] ? 1U : 0U;
            }
            unsigned sel_tot;
            unsigned at = carry_out + sel2_excl_scan<Policy>(sel_n, ws, sel_tot);
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
                if (sel[i]) {
                    const unsigned p = base + 4 * l + i;
                    for (unsigned j = 0; j < Policy::kKpool; ++j) tokens[at * Policy::kKpool + j] = p * Policy::kKpool + j;
                    ++at;
                }
            carry_eq += eq_tot;
            carry_out += sel_tot;
        }
        if (l == 0) append_tail<Policy>(P, t, carry_out, tokens, n_out);
    }
}
} // namespace strix::attn_core::select
