// Shared EXL3 trellis weight codec: the one operand source every engine's kernels read packed trellis weights
// through. Format and codebook semantics follow the adopted strix-trellis library (Apache License 2.0, see
// NOTICE.md and LICENSES/strix-trellis-Apache-2.0.txt); the EXL3 spelling identifies the exllamav3 format basis.
//
// The format itself (tile order, ring, positions, host reference) is defined in trellis_format.hpp.
//
// Operand-source interface for a tile geometry (Codec<Bits, Book>):
//   block          kTile = 16 (inputs and outputs), kWords, kTileBytes, kDecoderStateBytes = 0
//   random access  Format::tile(), Format::position(), Format::state() (host reference of every reader below)
//   readers        Lane      lane t of a wave owns states 8t .. 8t + 7 (three words; row-vector / GEMV geometries)
//                  Fragment  lane c owns output column c, its 16 input coordinates (WMMA B fragment geometries)
//                  WaveTile  the whole ring in two shuffle banks, any state from any lane
//                  LaneTile  16-byte windows of a whole or half tile per lane
//   lookahead      every reader splits into issue (raw words, no dependence on earlier loads) and decode (pure
//                  register arithmetic), so a geometry can keep kLoads words per lane in flight for any number of
//                  tiles ahead; issue points are the caller's choice
//   decode         Book::value (FP32 reference form), Book::half (one FP16 FMA), Book::pair (two states -> packed
//                  halves, full-rate integer ops); all forms give the same FP16 bits for all 65,536 states
// Layout policies (where the tile row of an input tile starts, how experts are addressed) stay with the engines
// (for example moe_core ExpertLayout::packed_stride); the codec only needs the tile base.
// Compile with -ffp-contract=off.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "trellis_format.hpp"

namespace strix::core::trellis {

using Half2 = _Float16 __attribute__((ext_vector_type(2)));
using Word4 = std::uint32_t __attribute__((ext_vector_type(4)));

// Explicit address spaces: pointers read from memory are generic to the compiler, which would emit flat loads
// (64-bit lane addresses, and counted on the LDS counter too).
using GlobalU32 = const std::uint32_t __attribute__((address_space(1)));
using LocalU32 = const std::uint32_t __attribute__((address_space(3)));

// ---------------------------------------------------------------- codebooks
__host__ __device__ __forceinline__ unsigned byte_sum(std::uint32_t product) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_udot4(product, 0x01010101U, 0, false);
#else
    return (product & 255U) + ((product >> 8U) & 255U) + ((product >> 16U) & 255U) + (product >> 24U);
#endif
}

// state * Multiplier (mod 2^32) for a 16-bit state with full-rate integer ops (v_mul_lo_u32 and v_dot4_u32_u8 issue
// at quarter and half rate on gfx1151). Only the low 16 bits of the state container are read.
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

// The mul1 codebook: value(s) = RN16((1024 + bytesum(s * 0x83dcd12d)) * F16(0x1eee) + F16(0xc931)).
// Both FP16 multiplicands have <= 11 significant bits, so the product and the bounded sum are exact in FP32: an FP16
// FMA, an FP32 FMA followed by RN16, and a separate FP32 multiply and add followed by RN16 are the same function.
struct Mul1 : Mul1Constants {
    // Only the low 16 bits of the container are read by pair() / half16(): a reader may leave the state unmasked.
    static constexpr bool kPairIgnoresHighBits = true;

    // FP32 reference form; returns the FP16 value widened to float.
    static __host__ __device__ __forceinline__ float value(std::uint32_t state) {
        const unsigned sum = byte_sum(state * kMultiplier);
        return __half2float(__float2half_rn(fmaf(static_cast<float>(1024 + sum), __half2float(__ushort_as_half(kScale)),
                                                 __half2float(__ushort_as_half(kBias)))));
    }
    // Same value; the FP32 addend is formed by bit insertion instead of an integer conversion.
    static __device__ __forceinline__ float value_v2(std::uint32_t state) {
        const std::uint32_t product = state * kMultiplier;
        const unsigned sum = __builtin_amdgcn_udot4(product, 0x01010101U, 0, false);
        return __half2float(__float2half_rn(fmaf(__uint_as_float(0x44800000U | (sum << 13)), __half2float(__ushort_as_half(kScale)),
                                                 __half2float(__ushort_as_half(kBias)))));
    }
    // One FP16 FMA per coefficient: 0x6400 + sum is the FP16 encoding of 1024 + sum (sum <= 1020).
    static __device__ __forceinline__ _Float16 half(std::uint32_t state) {
        const unsigned sum = __builtin_amdgcn_udot4(state * kMultiplier, 0x01010101U, 0x6400U, false);
        const auto scale = __builtin_bit_cast(_Float16, kScale);
        const auto bias = __builtin_bit_cast(_Float16, kBias);
        return __builtin_elementwise_fma(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(sum)), scale, bias);
    }
    // half() with full-rate integer ops (hash16 and v_sad_u8).
    static __device__ __forceinline__ _Float16 half16(std::uint32_t state) {
        unsigned sum;
        asm("v_sad_u8 %0, %1, 0, %2" : "=v"(sum) : "v"(hash16<kMultiplier>(state)), "s"(0x6400U));
        const auto scale = __builtin_bit_cast(_Float16, kScale);
        const auto bias = __builtin_bit_cast(_Float16, kBias);
        return __builtin_elementwise_fma(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(sum)), scale, bias);
    }
    // Two states -> packed halves (low half = first state): mul1_pair below.
    static __host__ __device__ __forceinline__ Half2 pair(std::uint32_t first, std::uint32_t second);
    // 24-bit multiply form of the product and the folded-constant codebook used by the pipelined grouped cores.
    static __device__ __forceinline__ std::uint32_t pipe_product(std::uint32_t state) {
        std::uint32_t lo, hi, product;
        asm("v_mul_u32_u24 %0, %1, 0xdcd12d" : "=v"(lo) : "v"(state));
        asm("v_mul_u32_u24 %0, %1, 0x83" : "=v"(hi) : "v"(state));
        asm("v_lshl_add_u32 %0, %1, 24, %2" : "=v"(product) : "v"(hi), "v"(lo));
        return product;
    }
    static __device__ __forceinline__ float pipe_value(std::uint32_t state) {
        const auto product = pipe_product(state);
        const unsigned sum = __builtin_amdgcn_udot4(product, 0x01010101U, 0, false);
        return __half2float(__float2half_rn(fmaf(static_cast<float>(sum), 887.0f / 131072.0f, -3.453125f)));
    }
};

// Two mul1 states -> packed halves (low half = first state): 2 x hash16, v_sad_u8 + v_sad_hi_u8, one packed FP16 FMA.
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
__host__ __device__ __forceinline__ Half2 Mul1::pair(std::uint32_t first, std::uint32_t second) { return mul1_pair(first, second); }

// A state as a reader extracted it: only the low 16 bits are the state. Converting to an integer masks; decoders that
// read only the low 16 bits of the container (hash16: mul1_pair, Mul1::half16) take it as it is and skip the mask.
struct State16 {
    std::uint32_t raw;
    __host__ __device__ __forceinline__ constexpr operator std::uint32_t() const { return raw & 65535U; }
};
static_assert(Mul1::kPairIgnoresHighBits, "mul1_pair(State16, State16) passes the states unmasked");
__forceinline__ __host__ __device__ Half2 mul1_pair(State16 first, State16 second) { return mul1_pair(first.raw, second.raw); }

// The library's three codebooks by id (0, 1: two-half sum codebooks; 2: mul1).
__host__ __device__ __forceinline__ float codebook_value(std::uint32_t state, int cb) {
    if (cb == 2) return Mul1::value(state);
    const std::uint32_t mixed = cb == 1 ? state * 0xcbac1fedU : state * 89226354U + 64248484U;
    const std::uint32_t halves = (mixed & 0x8fff8fffU) ^ 0x3b603b60U;
    return __half2float(__float2half_rn(__half2float(__ushort_as_half(static_cast<unsigned short>(halves & 65535U))) +
                                        __half2float(__ushort_as_half(static_cast<unsigned short>(halves >> 16)))));
}

// Reference coefficient (host and device): FP16 bits of weight (input k, output n) of an I x O matrix.
template <class Book = Mul1>
__host__ __device__ inline std::uint16_t reference_bits(const std::uint32_t* trellis, int bits, std::size_t outputs, std::size_t k,
                                                        std::size_t n) {
    const std::uint32_t* ring = trellis + Format::tile(k / 16U, n / 16U, outputs / 16U, bits);
    const float v = Book::value(Format::state(ring, bits, Format::position(static_cast<unsigned>(k % 16U), static_cast<unsigned>(n % 16U))));
    return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(v));
}

// ---------------------------------------------------------------- addressing helpers
// Wave-uniform value -> scalar register, so global loads take the saddr form.
__device__ __forceinline__ unsigned uni(unsigned v) {
    return static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(v)));
}
template <class T> __device__ __forceinline__ const T* uniform(const T* p) {
    const auto v = reinterpret_cast<std::uintptr_t>(p);
    const auto lo = static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(static_cast<std::uint32_t>(v))));
    const auto hi = static_cast<std::uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<int>(static_cast<std::uint32_t>(v >> 32U))));
    return reinterpret_cast<const T*>((static_cast<std::uintptr_t>(hi) << 32U) | lo);
}
// Keep the complete advanced base in scalar registers before a lane offset is added (uniformity alone lets
// instruction selection sink row strides into the vector address).
template <class T> __device__ __forceinline__ const T* scalar_advance(const T* base, unsigned elements) {
    auto address = reinterpret_cast<std::uintptr_t>(base) + static_cast<std::size_t>(elements) * sizeof(T);
    asm("" : "+s"(address));
    return reinterpret_cast<const T*>(address);
}
// Scalar base plus a 32-bit vector byte offset. Nonvolatile: an unused word stays dead-load eliminable.
__device__ __forceinline__ std::uint32_t load_u32(const unsigned char* base, std::uint32_t offset) {
    asm("" : "+v"(offset));  // keep a 32-bit VGPR offset on a wave-uniform base (saddr form)
    return *(GlobalU32*)(base + offset);
}

// ---------------------------------------------------------------- Lane reader: states 8t .. 8t + 7 of lane t
// The lane's eight consecutive states start at ring bit (t * 8 + 1) * Bits - 16; three words cover them.
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

// Raw words of one lane and tile: the issue half of the reader (kLoads loads, nothing waited on).
template <int Bits> struct LaneWords {
    static constexpr int kLoads = 3;
    std::uint32_t a, b, c;
    LaneWords() = default;  // uninitialized: a prefetch ring of these must not cost register clears
    __device__ __forceinline__ LaneWords(std::uint32_t a_, std::uint32_t b_, std::uint32_t c_) : a(a_), b(b_), c(c_) {}
    // Wave-uniform tile base (saddr loads).
    __device__ __forceinline__ static LaneWords issue(const void* tile, const LaneGeom<Bits>& g) {
        const auto* base = static_cast<const unsigned char*>(tile);
        return LaneWords(load_u32(base, g.oa), load_u32(base, g.ob), load_u32(base, g.oc));
    }
    // Per-lane tile pointer (vaddr loads).
    __device__ __forceinline__ static LaneWords issue_at(const std::uint32_t* tile, const LaneGeom<Bits>& g) {
        return LaneWords(tile[g.oa >> 2U], tile[g.ob >> 2U], tile[g.oc >> 2U]);  // word indices: one scaled address per load
    }
};

// The decode half: the aligned 3-word window and its eight states (compile-time shifts).
template <int Bits> struct TileStates {
    std::uint32_t high, low, tail;
    TileStates() = default;
    __device__ __forceinline__ TileStates(const unsigned char* tile, const LaneGeom<Bits>& g) {
        const std::uint32_t a = load_u32(tile, g.oa), b = load_u32(tile, g.ob), c = load_u32(tile, g.oc);
        high = __builtin_amdgcn_alignbit(a, b, g.align);  // (a << shift) | (b >> (32 - shift))
        low = __builtin_amdgcn_alignbit(b, c, g.align);
        tail = c << (32U - g.align);
    }
    // State J as extracted (State16): an integer conversion masks it to 16 bits exactly as before; mul1_pair takes it
    // unmasked, since hash16 reads only the low 16 bits.
    template <int J> __device__ __forceinline__ State16 state() const {
        constexpr int offset = J * Bits;
        if constexpr (offset <= 16) return State16{high >> (16 - offset)};
        else if constexpr (offset < 32) return State16{(high << (offset - 16)) | (low >> (48 - offset))};
        else if constexpr (offset <= 48) return State16{low >> (48 - offset)};
        else return State16{(low << (offset - 48)) | (tail >> (80 - offset))};
    }
};
// The same window from raw words issued earlier (align = LaneGeom::align). Plain shifts, not the funnel-shift
// intrinsic: the compiler then still proves the third word dead where the lane's shift keeps it out of every state
// (all lanes at 3 and 4 bits), and drops its load.
template <int Bits> __device__ __forceinline__ TileStates<Bits> align_words(const LaneWords<Bits>& w, std::uint32_t align) {
    const std::uint32_t shift = 32U - align;  // never 0 for integer Bits in 1..8
    TileStates<Bits> t;
    t.high = (w.a << shift) | (w.b >> (32U - shift));
    t.low = (w.b << shift) | (w.c >> (32U - shift));
    t.tail = w.c << shift;
    return t;
}

// ---------------------------------------------------------------- Fragment reader: lane c = output column
// Window (v2): the lane's states addressed inside kNW loaded words with half-tile dependent word index and shift.
template <int Bits> struct Window {
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

// Normalized window (v3): each lane funnel-shifts its kNL loaded words once so that word 0 starts at the same stream
// bit for every lane; state r is then alignbit(N[g(r) - 1], N[g(r)], shift(r)) with compile-time g and shift.
template <int Bits> struct Norm {
    static constexpr int kWords = Bits * 8;
    static constexpr int kNL = ((47 + 27 * Bits) >> 5) + 2;  // loaded words; kNL - 1 normalized words
    static constexpr int e(int r) { return 48 + Window<Bits>::delta(r) * Bits; }
    static constexpr int g(int r) { return (e(r) - 1) >> 5; }
    static constexpr unsigned shift(int r) { return static_cast<unsigned>((32 - (e(r) & 31)) & 31); }
};

// The normalized reader as one object: construct per lane (hoisted out of the k loop), issue raw words per tile,
// normalize and take states when the tile is consumed.
template <int Bits> struct Fragment {
    using N = Norm<Bits>;
    static constexpr int kLoads = N::kNL;       // words in flight per lane and tile
    static constexpr int kNormalized = N::kNL - 1;
    int wb;        // first word of the lane's window inside the tile ring
    unsigned sh;   // normalizing shift
    __device__ __forceinline__ explicit Fragment(int column) {
        const int base = 32 * (column & 7) + 4 * (column >> 3);
        const int S = (base + 1) * Bits - 16 + 256 * Bits - 32;
        const int d = S & 31;
        wb = (S >> 5) - (d == 0 ? 1 : 0);
        sh = static_cast<unsigned>((32 - d) & 31);
    }
    // Word index of load i inside the tile ring (add the tile's word offset).
    __device__ __forceinline__ int word(int i) const { return (wb + i) % N::kWords; }
    __device__ __forceinline__ void issue(const std::uint32_t* tile, std::uint32_t (&raw)[kLoads]) const {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) raw[i] = tile[word(i)];
    }
    __device__ __forceinline__ void normalize(const std::uint32_t (&raw)[kLoads], std::uint32_t (&nrm)[kNormalized]) const {
#pragma unroll
        for (int i = 0; i < kNormalized; ++i) nrm[i] = __builtin_amdgcn_alignbit(raw[i], raw[i + 1], sh);
    }
    template <int R> __device__ __forceinline__ static std::uint32_t state(const std::uint32_t (&nrm)[kNormalized]) {
        return __builtin_amdgcn_alignbit(nrm[N::g(R) - 1], nrm[N::g(R)], N::shift(R)) & 65535U;
    }
    // The lane's 16 coefficients (input coordinates 0..15 of its column) as 8 words of packed halves: the B fragment
    // of a 16x16x16 FP16 matrix operation.
    template <class Book = Mul1>
    __device__ __forceinline__ static void decode16(const std::uint32_t (&nrm)[kNormalized], std::uint32_t (&frag)[8]) {
        const auto pair = [&](auto first, auto second) { return __builtin_bit_cast(std::uint32_t, Book::pair(first, second)); };
        frag[0] = pair(state<0>(nrm), state<1>(nrm));   frag[1] = pair(state<2>(nrm), state<3>(nrm));
        frag[2] = pair(state<4>(nrm), state<5>(nrm));   frag[3] = pair(state<6>(nrm), state<7>(nrm));
        frag[4] = pair(state<8>(nrm), state<9>(nrm));   frag[5] = pair(state<10>(nrm), state<11>(nrm));
        frag[6] = pair(state<12>(nrm), state<13>(nrm)); frag[7] = pair(state<14>(nrm), state<15>(nrm));
    }
};

// ---------------------------------------------------------------- WaveTile reader: the ring in two shuffle banks
// All lanes participate, including lanes without a source word.
template <int Bits> struct WaveTile {
    std::uint32_t lo, hi;
    static constexpr int words = Bits * 8;
    __device__ explicit WaveTile(const std::uint32_t* data) {
        const int lane = static_cast<int>(threadIdx.x & 31U);
        lo = lane < words ? data[lane] : 0;
        if constexpr (words > 32) hi = lane + 32 < words ? data[lane + 32] : 0;
        else hi = 0;
    }
    __device__ __forceinline__ std::uint32_t word(int index) const {
        const std::uint32_t first = __shfl(lo, index & 31, 32);
        if constexpr (words > 32) {
            const std::uint32_t second = __shfl(hi, index & 31, 32);
            return index < 32 ? first : second;
        }
        return first;
    }
    template <int Group> __device__ __forceinline__ std::uint64_t group(int first_position) const {
        constexpr int width = 16 + (Group - 1) * Bits;
        const int end = (first_position + Group) * Bits + 256 * Bits;
        const int start = end - width;
        const std::uint64_t pair = (static_cast<std::uint64_t>(word((start / 32) % words)) << 32) | word(((end - 1) / 32) % words);
        return pair >> ((32 - (end & 31)) & 31);
    }
    __device__ __forceinline__ std::uint32_t state(int position) const {
        const int end = (position + 1) * Bits + 256 * Bits;
        const int start = end - 16;
        const std::uint32_t earlier = word((start / 32) % words);
        const std::uint32_t later = word(((end - 1) / 32) % words);
        const std::uint64_t pair = (static_cast<std::uint64_t>(earlier) << 32) | later;
        return static_cast<std::uint32_t>(pair >> ((32 - (end & 31)) & 31)) & 65535U;
    }
};

// ---------------------------------------------------------------- LaneTile reader: 16-byte windows per lane
// G = 8: the whole tile; G = 4: the half tile of column groups G0 .. G0 + 3 (and + 8), which also needs the word
// preceding its first group (ring wrap for G0 = 0). Any state P of the covered half is a compile-time selection.
template <unsigned K, unsigned G, unsigned G0> struct LaneTile {
    static_assert(G == 8 ? G0 == 0 : G == 4 && (G0 == 0 || G0 == 4));
    static constexpr unsigned ring = 8 * K;
    static constexpr unsigned base = G == 8 ? 0 : (G0 * K + ring - 4) % ring;
    static constexpr unsigned vectors = G == 8 ? 2 * K : G * K / 4 + 1;
    Word4 v[vectors];
    __device__ __forceinline__ void load(const std::uint32_t* tile) {
#pragma unroll
        for (unsigned i = 0; i < vectors; ++i) v[i] = *reinterpret_cast<const Word4*>(tile + (base + 4 * i) % ring);
    }
    template <unsigned J> __device__ __forceinline__ std::uint32_t word() const {
        static_assert(J < 4 * vectors);
        return v[J / 4][J % 4];
    }
    template <unsigned P> __device__ __forceinline__ unsigned state() const {
        constexpr unsigned start = ((P + 1) * K + 256 * K - 16) % (256 * K), off = start % 32;
        constexpr unsigned first = (start / 32 + ring - base) % ring;
        if constexpr (off <= 16) return (word<first>() >> (16 - off)) & 65535U;
        else {
            constexpr unsigned second = (start / 32 + 1 + ring - base) % ring;
            return ((word<first>() << (off - 16)) | (word<second>() >> (48 - off))) & 65535U;
        }
    }
};

// ---------------------------------------------------------------- the trait a kernel instantiates
template <int Bits, class Book = Mul1> struct Codec {
    static_assert(Bits >= 1 && Bits <= 8);
    static constexpr int kBits = Bits;
    static constexpr int kTile = Format::kTile;
    static constexpr int kWords = Format::words(Bits);
    static constexpr int kTileBytes = Format::tile_bytes(Bits);
    static constexpr int kDecoderStateBytes = Format::kDecoderStateBytes;
    using Codebook = Book;
    using Window = trellis::Window<Bits>;
    using Norm = trellis::Norm<Bits>;
    using Fragment = trellis::Fragment<Bits>;
    using Geom = trellis::LaneGeom<Bits>;
    using Words = trellis::LaneWords<Bits>;
    using States = trellis::TileStates<Bits>;
    using Wave = trellis::WaveTile<Bits>;
    static __host__ __device__ __forceinline__ float codebook_value(std::uint32_t state) { return Book::value(state); }
    static __device__ __forceinline__ float codebook_value_v2(std::uint32_t state) { return Book::value_v2(state); }
    static __device__ __forceinline__ std::uint32_t pipe_product(std::uint32_t state) { return Book::pipe_product(state); }
    static __device__ __forceinline__ float pipe_codebook(std::uint32_t state) { return Book::pipe_value(state); }
};

// Decode policies of an 8-state group (four packed pairs). Scalar keeps the FP32 reference conversion per state;
// Pair is the packed full-rate form. Same bits.
template <int Bits, class Book = Mul1> struct ScalarDecode8 {
    static constexpr bool kPacked = false;
    static __host__ __device__ __forceinline__ void decode8(const std::uint32_t (&st)[8], Half2 (&out)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            out[i] = Half2{static_cast<_Float16>(Book::value(st[2 * i])), static_cast<_Float16>(Book::value(st[2 * i + 1]))};
    }
};
template <int Bits, class Book = Mul1> struct PairDecode8 {
    static constexpr bool kPacked = true;
    static __host__ __device__ __forceinline__ void decode8(const std::uint32_t (&st)[8], Half2 (&out)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i) out[i] = Book::pair(st[2 * i], st[2 * i + 1]);
    }
};

}  // namespace strix::core::trellis
