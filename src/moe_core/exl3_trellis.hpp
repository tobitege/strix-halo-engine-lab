#pragma once
// minimax-m3 MoE/dense kernels: device building blocks of the strix-trellis 0.3 EXL3 linear (mul1 codebook), copied
// and adapted (not linked) so the fused kernels of src/minimax/moe/ reproduce its bits row by row:
//   input   t = RN16(H32(RN16(RN32(x * suh))))                       (transforms.hpp input_transform)
//   core    per output tile column, per split s: lane chains of v_dot2_f32_f16 over the split's k-tiles
//           (pairs 0,1 -> sum 0, pairs 2,3 -> sum 1), then v += shfl_down(v, 2, 4); v += shfl_down(v, 1, 4)
//   finish  acc = ((0 + p_0) + p_1) + ... + p_{S-1}; y = RN32(H32(acc) * svh)   (output_transform, alpha 1, no bias)
// The split count S is the library's vector policy for <= 8 rows (vector_splits below). Every helper keeps the
// library's operation order; only addressing, scheduling and launch structure differ.
// Compile with -ffp-contract=off (engine flags).
#include <cmath>
#include <cstdint>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace strix::moe_core {

using Half2 = _Float16 __attribute__((ext_vector_type(2)));

// strix-trellis split policy for <= 8 rows (gemv_policy.hpp measured_vector_splits; no MiniMax shape is special-cased).
__host__ __device__ constexpr unsigned vector_splits(unsigned input, unsigned output) {
    return output <= 1024 ? 32U : (output <= 4096 && input >= 4096 ? 16U : 4U);
}

// RN32 boundary before a later RN16 conversion (hadamard.hpp materialize_fp32).
__device__ __forceinline__ float materialize(float v) {
    asm volatile("" : "+v"(v));
    return v;
}

// H32 over 128 values held as lane + 32 j (hadamard.hpp hadamard128): strides 1..16 across lanes, 32 and 64 in
// registers, then one multiply by RN32(1/sqrt(128)).
__device__ __forceinline__ void hadamard128(float (&v)[4], unsigned lane) {
#pragma unroll
    for (int stride = 1; stride < 32; stride *= 2) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float peer = __shfl_xor(v[j], stride, 32);
            v[j] = (lane & static_cast<unsigned>(stride)) ? peer - v[j] : v[j] + peer;
        }
    }
#pragma unroll
    for (int stride = 1; stride < 4; stride *= 2) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (!(j & stride)) {
                const float l = v[j], r = v[j + stride];
                v[j] = l + r;
                v[j + stride] = l - r;
            }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = materialize(v[j] * 0x1.6a09e6p-4F);
}

__device__ __forceinline__ float half_bits(std::uint16_t h) { return __half2float(__ushort_as_half(h)); }

// One wave: input transform of the 128 channels [begin, begin + 128) of one row.
__device__ __forceinline__ void transform_block(const float* x, const std::uint16_t* suh, _Float16* out, unsigned begin,
                                                unsigned lane) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = begin + lane + 32U * static_cast<unsigned>(j);
        const float scaled = materialize(x[c] * half_bits(suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    hadamard128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) out[begin + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
}
// Same transform on the lane's four x values already in registers (xv[j] = x[begin + lane + 32 j]), so a caller that
// transforms one block for several (suh, out) pairs loads x once. transform_block_at is this helper behind the loads.
__device__ __forceinline__ void transform_block_values(const float (&xv)[4], const std::uint16_t* suh, _Float16* out,
                                                       unsigned begin, unsigned obegin, unsigned lane) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = begin + lane + 32U * static_cast<unsigned>(j);
        const float scaled = materialize(xv[j] * half_bits(suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    hadamard128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) out[obegin + lane + 32U * static_cast<unsigned>(j)] = static_cast<_Float16>(v[j]);
}
// Same transform, writing the 128 results at out[obegin ...] (an LDS chunk that starts at channel begin - obegin).
__device__ __forceinline__ void transform_block_at(const float* x, const std::uint16_t* suh, _Float16* out, unsigned begin,
                                                   unsigned obegin, unsigned lane) {
    float xv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) xv[j] = x[begin + lane + 32U * static_cast<unsigned>(j)];
    transform_block_values(xv, suh, out, begin, obegin, lane);
}
// Same transform, returning the four per-lane values in registers (no store).
__device__ __forceinline__ void transform_values(const float* x, const std::uint16_t* suh,
                                                unsigned begin, unsigned lane, float (&v)[4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned c = begin + lane + 32U * static_cast<unsigned>(j);
        const float scaled = materialize(x[c] * half_bits(suh[c]));
        v[j] = static_cast<float>(static_cast<_Float16>(scaled));
    }
    hadamard128(v, lane);
}

// ---------------------------------------------------------------- mul1 decode (packed_decoder.hpp, qualified bitwise
// against trellis_optimized.hpp mul1_half2)
template <std::uint32_t Multiplier> __host__ __device__ __forceinline__ std::uint32_t hash16(std::uint32_t state) {
#if defined(__HIP_DEVICE_COMPILE__)
    constexpr std::uint32_t adjusted = Multiplier - 1U;
    constexpr std::uint32_t upper = (adjusted & 0xffff0000U) | 1U;
    constexpr std::uint32_t lower = adjusted & 65535U;
    std::uint32_t high, result;
    asm("v_pk_mul_lo_u16 %0, %1, %2 op_sel:[0,0] op_sel_hi:[0,1]" : "=v"(high) : "v"(state), "s"(upper));
    asm("v_mad_u32_u16 %0, %1, %2, %1" : "=v"(result) : "v"(high), "s"(lower));
    return result;
#else
    // Emulate the low-u16 operands of the two packed ISA instructions.
    return (state & 65535U) * Multiplier;
#endif
}
__host__ __device__ __forceinline__ Half2 mul1_pair(std::uint32_t first, std::uint32_t second) {
    const std::uint32_t a = hash16<0x83dcd12dU>(first), b = hash16<0x83dcd12dU>(second);
#if defined(__HIP_DEVICE_COMPILE__)
    std::uint32_t low, both;
    constexpr std::uint32_t seed = 0x64006400U;
    asm("v_sad_u8 %0, %1, 0, %2" : "=v"(low) : "v"(a), "s"(seed));
    asm("v_sad_hi_u8 %0, %1, 0, %2" : "=v"(both) : "v"(b), "v"(low));
    const __half2 d = __hfma2(__builtin_bit_cast(__half2, both), __builtin_bit_cast(__half2, 0x1eee1eeeU),
                              __builtin_bit_cast(__half2, 0xc931c931U));
    return __builtin_bit_cast(Half2, d);
#else
    const auto packed_input = [](std::uint32_t product) {
        const unsigned sum = (product & 255U) + ((product >> 8U) & 255U) +
                             ((product >> 16U) & 255U) + (product >> 24U);
        return __builtin_bit_cast(_Float16, static_cast<std::uint16_t>(0x6400U + sum));
    };
    const float scale = static_cast<float>(__builtin_bit_cast(_Float16, std::uint16_t{0x1eee}));
    const float bias = static_cast<float>(__builtin_bit_cast(_Float16, std::uint16_t{0xc931}));
    // Both FP16 multiplicands have <=11 significant bits. Their exact product
    // and this bounded sum fit FP32, so RN16(fmaf) emulates packed RN16 FMA.
    return Half2{static_cast<_Float16>(fmaf(static_cast<float>(packed_input(a)), scale, bias)),
                 static_cast<_Float16>(fmaf(static_cast<float>(packed_input(b)), scale, bias))};
#endif
}

// Per-lane constants of the aligned 16x16 tile reader (trellis_optimized.hpp AlignedLane<Bits, true>): the lane owns
// eight consecutive states starting at bit ((lane * 8 + 1) * Bits - 16) of the tile's Bits * 8-word ring.
template <int Bits> struct LaneGeom {
    std::uint32_t oa, ob, oc;  // byte offsets of the three source words inside the tile
    std::uint32_t align;       // 32 - shift (shift is never 0 for integer Bits in 1..8)
    __device__ explicit LaneGeom(unsigned lane) {
        constexpr int count = Bits * 8;
        const int start = (static_cast<int>(lane) * 8 + 1) * Bits + 256 * Bits - 16;
        const int index = start / 32 - count, shift = start & 31;
        const int ia = index < 0 ? index + count : index;
        const int ib = index + 1 >= count ? index + 1 - count : index + 1;
        const int ic = index + 2 >= count ? index + 2 - count : index + 2;
        oa = static_cast<std::uint32_t>(ia) * 4U;
        ob = static_cast<std::uint32_t>(ib) * 4U;
        oc = static_cast<std::uint32_t>(ic) * 4U;
        align = 32U - static_cast<std::uint32_t>(shift);
    }
};

// Explicit address spaces: pointers read from memory (the Exl3 tables) are generic to the compiler, which would emit
// flat loads (64-bit lane addresses, and counted on the LDS counter too).
using GlobalU32 = const std::uint32_t __attribute__((address_space(1)));
using LocalU32 = const std::uint32_t __attribute__((address_space(3)));
// Wave-uniform copy of a pointer (scalar registers), so global loads take the saddr form.
__device__ __forceinline__ unsigned uni(unsigned v) {  // wave-uniform value -> scalar register
    return static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(v)));
}
template <class T> __device__ __forceinline__ const T* uniform(const T* p) {
    const auto v = reinterpret_cast<std::uintptr_t>(p);
    const auto lo = static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(static_cast<std::uint32_t>(v))));
    const auto hi = static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(static_cast<std::uint32_t>(v >> 32U))));
    return reinterpret_cast<const T*>((static_cast<std::uintptr_t>(hi) << 32U) | lo);
}
__device__ __forceinline__ std::uint32_t load_u32(const unsigned char* base, std::uint32_t offset) {
    asm("" : "+v"(offset));  // keep a 32-bit VGPR offset on a wave-uniform base (saddr form)
    return *(GlobalU32*)(base + offset);
}

template <int Bits> struct TileStates {
    std::uint32_t high, low, tail;
    TileStates() = default;
    __device__ __forceinline__ TileStates(const unsigned char* tile, const LaneGeom<Bits>& g) {
        const std::uint32_t a = load_u32(tile, g.oa), b = load_u32(tile, g.ob), c = load_u32(tile, g.oc);
        high = __builtin_amdgcn_alignbit(a, b, g.align);  // (a << shift) | (b >> (32 - shift))
        low = __builtin_amdgcn_alignbit(b, c, g.align);
        tail = c << (32U - g.align);
    }
    template <int J> __device__ __forceinline__ std::uint32_t state() const {
        constexpr int offset = J * Bits;
        if constexpr (offset <= 16) return (high >> (16 - offset)) & 65535U;
        else if constexpr (offset < 32) return ((high << (offset - 16)) | (low >> (48 - offset))) & 65535U;
        else if constexpr (offset <= 48) return (low >> (48 - offset)) & 65535U;
        else return ((low << (offset - 48)) | (tail >> (80 - offset))) & 65535U;
    }
};

// Activation pairs of one k-tile for one row: channels k * 16 + 2 (lane & 3) (+ 8).
__device__ __forceinline__ void load_act(const _Float16* row, unsigned k, unsigned lane, Half2& a0, Half2& a1) {
    const auto* p = (GlobalU32*)(reinterpret_cast<const std::uint32_t*>(row)) + k * 8U + (lane & 3U);
    a0 = __builtin_bit_cast(Half2, p[0]);
    a1 = __builtin_bit_cast(Half2, p[4]);
}

// One k-tile of the core for up to R rows (rows >= nrows are skipped; their sums are not touched).
template <int Bits, int R>
__device__ __forceinline__ void core_tile(const unsigned char* tile, const LaneGeom<Bits>& g, const _Float16* x,
                                          std::uint32_t xstride, unsigned nrows, unsigned k, unsigned lane,
                                          float (&sums)[R][2]) {
    Half2 act[R][2];
#pragma unroll
    for (int r = 0; r < R; ++r)
        if (static_cast<unsigned>(r) < nrows) load_act(x + static_cast<std::size_t>(r) * xstride, k, lane, act[r][0], act[r][1]);
    const TileStates<Bits> t(tile, g);
    const Half2 w0 = mul1_pair(t.template state<0>(), t.template state<1>());
    const Half2 w1 = mul1_pair(t.template state<2>(), t.template state<3>());
    const Half2 w2 = mul1_pair(t.template state<4>(), t.template state<5>());
    const Half2 w3 = mul1_pair(t.template state<6>(), t.template state<7>());
#pragma unroll
    for (int r = 0; r < R; ++r)
        if (static_cast<unsigned>(r) < nrows) {
            sums[r][0] = __builtin_amdgcn_fdot2(w0, act[r][0], sums[r][0], false);
            sums[r][0] = __builtin_amdgcn_fdot2(w1, act[r][1], sums[r][0], false);
            sums[r][1] = __builtin_amdgcn_fdot2(w2, act[r][0], sums[r][1], false);
            sums[r][1] = __builtin_amdgcn_fdot2(w3, act[r][1], sums[r][1], false);
        }
}

// The 4-lane reduction of one sum; lane (lane & 3) == 0 holds output tile_col * 16 + lane / 4 + h * 8.
__device__ __forceinline__ float reduce4(float v) {
    v += __shfl_down(v, 2, 4);
    v += __shfl_down(v, 1, 4);
    return v;
}

// Prefetched form: the raw words and activation pairs of the next D k-tiles stay in flight in a register ring (the
// activations ride in the same ring because vector-memory counters retire in issue order). Same arithmetic, same bits.
template <int Bits, int R> struct Slot {
    std::uint32_t a, b, c;
    Half2 act[R][2];
};
template <int Bits, int R>
__device__ __forceinline__ void load_slot(Slot<Bits, R>& s, const unsigned char* tile, const LaneGeom<Bits>& g,
                                          const _Float16* x, std::uint32_t xstride, unsigned nrows, unsigned k,
                                          unsigned lane) {
    s.a = load_u32(tile, g.oa);
    s.b = load_u32(tile, g.ob);
    s.c = load_u32(tile, g.oc);
#pragma unroll
    for (int r = 0; r < R; ++r)
        if (static_cast<unsigned>(r) < nrows) load_act(x + static_cast<std::size_t>(r) * xstride, k, lane, s.act[r][0], s.act[r][1]);
}
template <int Bits, int R>
__device__ __forceinline__ void compute_slot(const Slot<Bits, R>& s, const LaneGeom<Bits>& g, unsigned nrows,
                                             float (&sums)[R][2]) {
    TileStates<Bits> t;
    t.high = __builtin_amdgcn_alignbit(s.a, s.b, g.align);
    t.low = __builtin_amdgcn_alignbit(s.b, s.c, g.align);
    t.tail = s.c << (32U - g.align);
    const Half2 w0 = mul1_pair(t.template state<0>(), t.template state<1>());
    const Half2 w1 = mul1_pair(t.template state<2>(), t.template state<3>());
    const Half2 w2 = mul1_pair(t.template state<4>(), t.template state<5>());
    const Half2 w3 = mul1_pair(t.template state<6>(), t.template state<7>());
#pragma unroll
    for (int r = 0; r < R; ++r)
        if (static_cast<unsigned>(r) < nrows) {
            sums[r][0] = __builtin_amdgcn_fdot2(w0, s.act[r][0], sums[r][0], false);
            sums[r][0] = __builtin_amdgcn_fdot2(w1, s.act[r][1], sums[r][0], false);
            sums[r][1] = __builtin_amdgcn_fdot2(w2, s.act[r][0], sums[r][1], false);
            sums[r][1] = __builtin_amdgcn_fdot2(w3, s.act[r][1], sums[r][1], false);
        }
}

// LDS form: a ring of Depth k-tiles of weight words only (3 registers per tile) and the activation pairs read from LDS
// right before use (LDS reads retire on their own counter, so they never wait behind the weight loads in flight).
// xs: LDS rows (stride xs_stride halves) of the transformed input holding channels from kofs * 16 on.
// prime() issues the first Depth tiles' loads; a caller can prime before staging its LDS inputs (with lds_barrier(),
// which waits for LDS operations only) so the weight stream is already in flight during the staging.
template <int Bits, int Depth> struct LdsRing {
    static_assert(Depth >= 1);
    LaneGeom<Bits> g;
    const unsigned char* base;
    std::size_t kstride;
    unsigned kt, splits, s0, k0, k1;
    std::uint32_t ra[Depth], rb[Depth], rc[Depth];
    __device__ __forceinline__ LdsRing(const std::uint32_t* trellis, unsigned output, unsigned col, unsigned kt_,
                                       unsigned splits_, unsigned s0_, unsigned s1_, unsigned lane)
        : g(lane), kt(kt_), splits(splits_), s0(s0_) {
        kstride = static_cast<std::size_t>(output / 16U) * Bits * 32U;
        base = uniform(reinterpret_cast<const unsigned char*>(trellis)) + static_cast<std::size_t>(col) * Bits * 32U;
        k0 = kt * s0_ / splits;
        k1 = kt * s1_ / splits;
#pragma unroll
        for (int j = 0; j < Depth; ++j)
            if (k0 + static_cast<unsigned>(j) < k1) {
                const unsigned char* t = base + (k0 + static_cast<unsigned>(j)) * kstride;
                ra[j] = load_u32(t, g.oa);
                rb[j] = load_u32(t, g.ob);
                rc[j] = load_u32(t, g.oc);
            }
    }
    template <int R>
    __device__ __forceinline__ void run(const _Float16* xs, unsigned xs_stride, unsigned kofs, unsigned nrows, unsigned lane,
                                        float (&acc)[R][2]) {
        unsigned s = s0, send = kt * (s0 + 1U) / splits;
        float sums[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) sums[r][0] = sums[r][1] = 0.0F;
        const auto* xw = (LocalU32*)(reinterpret_cast<const std::uint32_t*>(xs)) + (lane & 3U);
        for (unsigned kb = k0; kb < k1; kb += Depth) {
#pragma unroll
            for (int j = 0; j < Depth; ++j) {
                const unsigned k = kb + static_cast<unsigned>(j);
                if (k < k1) {
                    Slot<Bits, R> cur;
                    cur.a = ra[j];
                    cur.b = rb[j];
                    cur.c = rc[j];
                    if (k + Depth < k1) {
                        const unsigned char* t = base + (k + Depth) * kstride;
                        ra[j] = load_u32(t, g.oa);
                        rb[j] = load_u32(t, g.ob);
                        rc[j] = load_u32(t, g.oc);
                    }
#pragma unroll
                    for (int r = 0; r < R; ++r)
                        if (static_cast<unsigned>(r) < nrows) {
                            const auto* p = xw + (static_cast<unsigned>(r) * xs_stride) / 2U + (k - kofs) * 8U;
                            cur.act[r][0] = __builtin_bit_cast(Half2, p[0]);
                            cur.act[r][1] = __builtin_bit_cast(Half2, p[4]);
                        }
                    compute_slot<Bits, R>(cur, g, nrows, sums);
                    if (k + 1U == send) {
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            if (static_cast<unsigned>(r) < nrows) {
                                acc[r][0] = acc[r][0] + reduce4(sums[r][0]);
                                acc[r][1] = acc[r][1] + reduce4(sums[r][1]);
                            }
                            sums[r][0] = sums[r][1] = 0.0F;
                        }
                        ++s;
                        send = kt * (s + 1U) / splits;
                    }
                }
            }
        }
    }
};

// Optimization: LdsRing for the MoE expert kernels (split4_core) with the wave-uniform k range / stride / tile
// base forced into scalar registers and the K3 two-word reader (gu4p2: moe n1 -3 %, n2 -7 %, n4 -8 %, exact). The linear
// cores keep LdsRing unchanged (with this addressing they measured +1.5..3 % at N1 in gu4p2).
template <int Bits, int Depth> struct LdsRingU {
    static_assert(Depth >= 1);
    LaneGeom<Bits> g;
    const unsigned char* base;
    std::size_t kstride;
    unsigned kt, splits, s0, k0, k1;
    std::uint32_t ra[Depth], rb[Depth], rc[Depth];
    // optimization: K3 never reads the third word. Lane shift = ((lane * 8 + 1) * 3 + 768 - 16) & 31 is 19 / 11 / 3 / 27 (<= 27),
    // and the 8 states span offsets 0..21, so a state reads at most low >> 27 = the top 5 bits of low, which come from
    // b << shift alone; tail is never read. With c = 0 compute_slot builds the same high/low (same bits); one load less.
    static constexpr bool kNeedC = Bits != 3;
    static __device__ __forceinline__ unsigned u(unsigned v) { return uni(v); }
    __device__ __forceinline__ LdsRingU(const std::uint32_t* trellis, unsigned output, unsigned col, unsigned kt_,
                                       unsigned splits_, unsigned s0_, unsigned s1_, unsigned lane)
        : g(lane), kt(kt_), splits(splits_), s0(s0_) {
        // Optimization: the k range, row stride and tile base are wave-uniform by construction; force them
        // into scalar registers so the k loop, its bounds checks and the tile addresses stay on the SALU (saddr loads)
        // instead of per-lane 64-bit VALU adds and exec-mask branches. Addressing only: the same words, same order.
        kstride = static_cast<std::size_t>(u(output / 16U)) * Bits * 32U;  // 64-bit product as before
        base = uniform(reinterpret_cast<const unsigned char*>(trellis) + static_cast<std::size_t>(col) * Bits * 32U);
        k0 = u(kt * s0_ / splits);
        k1 = u(kt * s1_ / splits);
#pragma unroll
        for (int j = 0; j < Depth; ++j)
            if (k0 + static_cast<unsigned>(j) < k1) {
                const unsigned char* t = base + (k0 + static_cast<unsigned>(j)) * kstride;
                ra[j] = load_u32(t, g.oa);
                rb[j] = load_u32(t, g.ob);
                rc[j] = kNeedC ? load_u32(t, g.oc) : 0U;
            }
    }
    template <int R>
    __device__ __forceinline__ void run(const _Float16* xs, unsigned xs_stride, unsigned kofs, unsigned nrows, unsigned lane,
                                        float (&acc)[R][2]) {
        unsigned s = u(s0), send = u(kt * (s0 + 1U) / splits);
        float sums[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) sums[r][0] = sums[r][1] = 0.0F;
        const auto* xw = (LocalU32*)(reinterpret_cast<const std::uint32_t*>(xs)) + (lane & 3U);
        for (unsigned kb = u(k0); kb < k1; kb = u(kb + static_cast<unsigned>(Depth))) {
#pragma unroll
            for (int j = 0; j < Depth; ++j) {
                const unsigned k = kb + static_cast<unsigned>(j);
                if (k < k1) {
                    Slot<Bits, R> cur;
                    cur.a = ra[j];
                    cur.b = rb[j];
                    cur.c = rc[j];
                    if (k + Depth < k1) {
                        const unsigned char* t = base + (k + Depth) * kstride;
                        ra[j] = load_u32(t, g.oa);
                        rb[j] = load_u32(t, g.ob);
                        rc[j] = kNeedC ? load_u32(t, g.oc) : 0U;
                    }
#pragma unroll
                    for (int r = 0; r < R; ++r)
                        if (static_cast<unsigned>(r) < nrows) {
                            const auto* p = xw + (static_cast<unsigned>(r) * xs_stride) / 2U + (k - kofs) * 8U;
                            cur.act[r][0] = __builtin_bit_cast(Half2, p[0]);
                            cur.act[r][1] = __builtin_bit_cast(Half2, p[4]);
                        }
                    compute_slot<Bits, R>(cur, g, nrows, sums);
                    if (k + 1U == send) {
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            if (static_cast<unsigned>(r) < nrows) {
                                acc[r][0] = acc[r][0] + reduce4(sums[r][0]);
                                acc[r][1] = acc[r][1] + reduce4(sums[r][1]);
                            }
                            sums[r][0] = sums[r][1] = 0.0F;
                        }
                        s = u(s + 1U);
                        send = u(kt * (s + 1U) / splits);
                    }
                }
            }
        }
    }
};

// Workgroup barrier that waits for this wave's LDS operations only (outstanding global loads stay in flight).
__device__ __forceinline__ void lds_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }

template <int Bits, int R, int Depth>
__device__ __forceinline__ void stream_column_lds(const std::uint32_t* trellis, unsigned output, unsigned col, unsigned kt,
                                                  unsigned splits, unsigned s0, unsigned s1, const _Float16* xs,
                                                  unsigned xs_stride, unsigned kofs, unsigned nrows, unsigned lane,
                                                  float (&acc)[R][2]) {
    LdsRing<Bits, Depth> ring(trellis, output, col, kt, splits, s0, s1, lane);
    ring.template run<R>(xs, xs_stride, kofs, nrows, lane, acc);
}

// optimization: stream_column_lds through LdsRingU (MoE expert kernels only).
template <int Bits, int R, int Depth>
__device__ __forceinline__ void stream_column_lds_u(const std::uint32_t* trellis, unsigned output, unsigned col, unsigned kt,
                                                    unsigned splits, unsigned s0, unsigned s1, const _Float16* xs,
                                                    unsigned xs_stride, unsigned kofs, unsigned nrows, unsigned lane,
                                                    float (&acc)[R][2]) {
    LdsRingU<Bits, Depth> ring(trellis, output, col, kt, splits, s0, s1, lane);
    ring.template run<R>(xs, xs_stride, kofs, nrows, lane, acc);
}

// One wave streams tile column `col` over splits [s0, s1) of a matrix with kt = input / 16 k-tiles, accumulating the
// reduced split partials in ascending split order into acc (lanes with (lane & 3) == 0 are meaningful).
// tiles: base of the trellis words; row stride of k-tiles = (output / 16) * Bits * 32 bytes.
// Depth 0: plain loop (compiler-scheduled, unroll 4); depth D > 0: explicit ring of D k-tiles in flight across the split
// boundaries of the wave's range.
template <int Bits, int R, int Depth = 0>
__device__ __forceinline__ void stream_column(const std::uint32_t* trellis, unsigned output, unsigned col, unsigned kt,
                                              unsigned splits, unsigned s0, unsigned s1, const _Float16* x,
                                              std::uint32_t xstride, unsigned nrows, unsigned lane, float (&acc)[R][2]) {
    const LaneGeom<Bits> g(lane);
    const std::size_t kstride = static_cast<std::size_t>(output / 16U) * Bits * 32U;
    const auto* base = uniform(reinterpret_cast<const unsigned char*>(trellis)) + static_cast<std::size_t>(col) * Bits * 32U;
    if constexpr (Depth > 0) {
        const unsigned k0 = kt * s0 / splits, k1 = kt * s1 / splits;
        Slot<Bits, R> ring[Depth];
#pragma unroll
        for (int j = 0; j < Depth; ++j)
            if (k0 + static_cast<unsigned>(j) < k1)
                load_slot<Bits, R>(ring[j], base + (k0 + static_cast<unsigned>(j)) * kstride, g, x, xstride, nrows,
                                   k0 + static_cast<unsigned>(j), lane);
        unsigned s = s0, send = kt * (s0 + 1U) / splits;
        float sums[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) sums[r][0] = sums[r][1] = 0.0F;
        for (unsigned kb = k0; kb < k1; kb += Depth) {
#pragma unroll
            for (int j = 0; j < Depth; ++j) {
                const unsigned k = kb + static_cast<unsigned>(j);
                if (k < k1) {
                const Slot<Bits, R> cur = ring[j];
                if (k + Depth < k1)
                    load_slot<Bits, R>(ring[j], base + (k + Depth) * kstride, g, x, xstride, nrows, k + Depth, lane);
                compute_slot<Bits, R>(cur, g, nrows, sums);
                if (k + 1U == send) {
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        if (static_cast<unsigned>(r) < nrows) {
                            acc[r][0] = acc[r][0] + reduce4(sums[r][0]);
                            acc[r][1] = acc[r][1] + reduce4(sums[r][1]);
                        }
                        sums[r][0] = sums[r][1] = 0.0F;
                    }
                    ++s;
                    send = kt * (s + 1U) / splits;
                }
                }
            }
        }
    } else {
    for (unsigned s = s0; s < s1; ++s) {
        const unsigned kb = kt * s / splits, ke = kt * (s + 1U) / splits;
        float sums[R][2];
#pragma unroll
        for (int r = 0; r < R; ++r) sums[r][0] = sums[r][1] = 0.0F;
        for (unsigned k = kb; k < ke; ++k) core_tile<Bits, R>(base + k * kstride, g, x, xstride, nrows, k, lane, sums);
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (static_cast<unsigned>(r) < nrows) {
                acc[r][0] = acc[r][0] + reduce4(sums[r][0]);
                acc[r][1] = acc[r][1] + reduce4(sums[r][1]);
            }
    }
    }
}


// EXL3 mul1 weight decode and its normalized packed-stream geometry.
// Input/output transforms retain explicit RN32/RN16 boundaries in the device helpers.
template <int Bits>
struct Window {
    static constexpr int kWords = Bits * 8;                   // one 16 x 16 tile
    static constexpr int kNW = (78 + 27 * Bits) / 32 + 1;     // words a lane reads
    // Stream bit offset of the window start, relative to word w0 = s0 / 32 - 1 (32..63), for h = column / 8.
    static constexpr int off(int h) { return 32 + ((4 * Bits * h + 257 * Bits - 16) & 31); }
    static constexpr int delta(int r) { return 8 * ((r & 7) >> 1) + (r & 1) + 2 * (r >> 3); }
    static constexpr int last_word(int h, int r) { return (off(h) + 15 + delta(r) * Bits) >> 5; }
    static constexpr unsigned shift(int h, int r) { return static_cast<unsigned>((32 - ((off(h) + 16 + delta(r) * Bits) & 31)) & 31); }
    __device__ static int first_word(int c) {
        const int base = 32 * (c & 7) + 4 * (c >> 3);
        return (((base + 1) * Bits - 16 + 256 * Bits) >> 5) - 1;
    }
};

template <int Bits>
struct Norm {
    static constexpr int kWords = Bits * 8;
    static constexpr int kNL = ((47 + 27 * Bits) >> 5) + 2;  // loaded words; kNL - 1 normalized words
    static constexpr int e(int r) { return 48 + Window<Bits>::delta(r) * Bits; }
    static constexpr int g(int r) { return (e(r) - 1) >> 5; }
    static constexpr unsigned shift(int r) { return static_cast<unsigned>((32 - (e(r) & 31)) & 31); }
};

struct Mul1 {};

template <int Bits> struct Mul1PairDecode {
    static constexpr bool kPacked = true;
    static __host__ __device__ __forceinline__ void decode8(const std::uint32_t (&st)[8], Half2 (&out)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i) out[i] = mul1_pair(st[2 * i], st[2 * i + 1]);
    }
};

template <int Bits> struct Exl3Trellis {
    using Window = moe_core::Window<Bits>;
    using Norm = moe_core::Norm<Bits>;
    using InputTransform = Exl3Trellis<Bits>;
    static __host__ __device__ __forceinline__ float codebook_value(std::uint32_t state, Mul1 = {}) {
  // library mul1 codebook (gemm.hip)
    const std::uint32_t product = state * 0x83dcd12dU;
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned sum = __builtin_amdgcn_udot4(product, 0x01010101U, 0, false);
#else
    const unsigned sum = (product & 255U) + ((product >> 8U) & 255U) +
                         ((product >> 16U) & 255U) + (product >> 24U);
#endif
    return __half2float(__float2half_rn(fmaf(static_cast<float>(1024 + sum), __half2float(__ushort_as_half(0x1eee)),
                                             __half2float(__ushort_as_half(0xc931)))));
    }
static __device__ __forceinline__ float codebook_value_v2(std::uint32_t state) {
    const std::uint32_t product = state * 0x83dcd12dU;
    const unsigned sum = __builtin_amdgcn_udot4(product, 0x01010101U, 0, false);
    return __half2float(__float2half_rn(fmaf(__uint_as_float(0x44800000U | (sum << 13)), __half2float(__ushort_as_half(0x1eee)),
                                             __half2float(__ushort_as_half(0xc931)))));
}
    static __device__ __forceinline__ void input_block(const float* x, const std::uint16_t* suh, _Float16* out,
                                                      unsigned begin, unsigned lane) {
        transform_block(x, suh, out, begin, lane);
    }
    static __device__ __forceinline__ void h128(float (&v)[4], unsigned lane) { hadamard128(v, lane); }
    static __device__ __forceinline__ float scale(float x, std::uint16_t sign) {
        return materialize(x * half_bits(sign));
    }
    static __device__ __forceinline__ std::uint32_t pipe_product(std::uint32_t state) {
        std::uint32_t lo, hi, product;
        asm("v_mul_u32_u24 %0, %1, 0xdcd12d" : "=v"(lo) : "v"(state));
        asm("v_mul_u32_u24 %0, %1, 0x83" : "=v"(hi) : "v"(state));
        asm("v_lshl_add_u32 %0, %1, 24, %2" : "=v"(product) : "v"(hi), "v"(lo));
        return product;
    }
    static __device__ __forceinline__ float pipe_codebook(std::uint32_t state) {
        const auto product = pipe_product(state);
        const unsigned sum = __builtin_amdgcn_udot4(product, 0x01010101U, 0, false);
        return __half2float(__float2half_rn(fmaf(static_cast<float>(sum), 887.0f / 131072.0f, -3.453125f)));
    }
};

// Scalar-default policy: preserve the previous codebook conversion and order.
template <int Bits, template <int> class WeightDecode = Exl3Trellis> struct ScalarDecode {
    static constexpr bool kPacked = false;
    static __host__ __device__ __forceinline__ void decode8(const std::uint32_t (&st)[8], Half2 (&out)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out[i] = Half2{static_cast<_Float16>(WeightDecode<Bits>::codebook_value(st[2 * i])),
                           static_cast<_Float16>(WeightDecode<Bits>::codebook_value(st[2 * i + 1]))};
        }
    }
};

}  // namespace strix::moe_core
