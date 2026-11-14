#pragma once
// Included after hip_q8_rows_gemv.hpp (it uses strix::q8_rows: Segments, segment_row, dense_a8_width,
// dense_f16_overflow_counter).
//
// STRIX_QPF2K_DENSE=1 (default off): whole-T register-blocked form of the fast-prefill-f16wmma-v1 dense Q8_0
// projections (q8_rows::launch_dense_f16). Same numerics, bit for bit:
//   y[r][n] = sum over K16 tiles k = 0, 1, ... ascending of v_wmma_f32_16x16x16_f16(fp16(x[r][16k..]), fp16(w[n][16k..]))
//   accumulated from +0 into one FP32 accumulator per (row, output), the site's store() on the final value.
//   - x: the site's BF16/FP32 activations rounded to FP16 (RNE), as dense_f16_convert_kernel does;
//   - w: fp16(d * q) of each Q8_0 block, d the block's FP16 scale and q its int8 code (the product is exact in
//     FP32 and in the FP16 multiply, so v_mul_f16(d, q) rounds the same exact value once, RNE).
// Only the decomposition changes: one launch covers all rows (any count, rows tiled by 256), a block owns
// 256 rows x 128 outputs, 8 waves of 64 x 64 (16 accumulators per wave, every A and B fragment reused 4x),
// one Q8_0 block (32 K values) per stage, two LDS stages (one barrier per stage) with the next stage's
// global loads in registers while the current one is multiplied, grouped rasterization (raster row tiles per
// group) so resident blocks share A rows and weight rows in L2.
//   - BF16 inputs: one convert launch over all rows (same RNE values, same overflow counting) into a per-stream
//     FP16 scratch, copied into LDS by the projection.
//   - FP32 inputs: the projection converts its A tile itself while staging it (same RNE values); blocks of
//     output tile 0 count overflows (each activation once, as the convert kernel does).
// Stage width, tiling, grid order and LDS layout do not change any (row, output) value: each accumulator sees
// the same operands in the same K16 order whatever block, launch or row count it lands in.

#include "strix/env_flags.hpp"
#include <atomic>
#include <cstdio>
#include <map>
#include <mutex>
#include "strix/qpf2k_defaults.hpp"

namespace strix::q8_rows::tiled {

typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned u32x2 __attribute__((ext_vector_type(2)));
typedef unsigned Frag __attribute__((ext_vector_type(8), aligned(16)));  // 16 halves as 8 dwords (no true16 re-pack)
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef float f4 __attribute__((ext_vector_type(4)));

constexpr unsigned kBM = 256U, kBN = 128U, kWM = 64U, kWN = 64U, kMT = kWM / 16U, kNT = kWN / 16U;
constexpr unsigned kWavesN = kBN / kWN, kThreadsT = 256U;
constexpr unsigned kPitch = 40U;  // halves per LDS row of one 32-K stage (80 bytes: 16-row b128 reads hit distinct banks)
constexpr unsigned kStageHalves = (kBM + kBN) * kPitch;  // 30 KiB per stage, 60 KiB for both
static_assert((kBM / kWM) * kWavesN == kThreadsT / 32U);
// Row pitch (halves) of the FP16 activation scratch: 128 bytes of padding keep the rows a staging load instruction
// touches off one memory channel (t2: the unpadded 12288-byte pitch of 6144-wide rows ran the projection at 43 %).
__host__ __device__ constexpr std::uint32_t scratch_pitch(std::uint32_t width) { return width + 64U; }

inline bool enabled() {
    static const bool on = strix::envcfg::flag("STRIX_QPF2K_DENSE", false);
    // GSQ owners must keep their typed projections; these tiled loaders assume Q8_0.
    return on && !qpf2k::suppressed(qpf2k::Item::dense);
}
// Bit mask of sites (STRIX_QPF2K_DENSE_SITES, default all): 1 GDN in, 2 GDN out, 4 QSA q/k/v, 8 QSA out,
// 16 shared gate+up, 32 shared down.
inline bool site_enabled(unsigned bit) {
    static const unsigned mask = [] { const char* v = std::getenv("STRIX_QPF2K_DENSE_SITES"); return v ? static_cast<unsigned>(std::strtoul(v, nullptr, 0)) : 63U; }();
    return enabled() && (mask & bit) != 0U;
}
// Fewest rows a call must have to take the tiled form (STRIX_QPF2K_DENSE_MIN_ROWS overrides every site, default 512).
// Per site (bit as site_enabled): GDN in (1) and shared gate|up (16) need 1024 by default: the short-call bench (mi t1,
// T 512) had them slower than the engine launches (GDN in cold 1.019 / warm 1.042, shared gate|up warm 1.103), every
// other site and every T >= 1024 faster. Routing only: a call runs either form, the bits are the same.
inline unsigned min_rows(unsigned bit = 0U) {
    static const int n = strix::envcfg::integer("STRIX_QPF2K_DENSE_MIN_ROWS", -1);
    if (n >= 0) return static_cast<unsigned>(n);
    return bit == 1U || bit == 16U ? 1024U : 512U;
}
// Row tiles per rasterization group (block -> tile mapping only). Measured (t1, T 8192): FP16-scratch A best at 1 (the
// resident blocks share one A row tile in L2 and stream weight rows), inline A best at 2 (its FP32/BF16 A rows are 2x
// the bytes). STRIX_QPF2K_DENSE_RASTER overrides both (0 = these defaults).
inline unsigned raster(unsigned mode) {
    static const unsigned n = [] { const char* v = std::getenv("STRIX_QPF2K_DENSE_RASTER"); const int r = v ? std::atoi(v) : 0; return r >= 1 && r <= 64 ? static_cast<unsigned>(r) : 0U; }();
    return n ? n : mode == 0U ? 1U : 2U;
}

__device__ __forceinline__ u32x4 g16(const void* p) {
    return *reinterpret_cast<const __attribute__((address_space(1))) u32x4*>(reinterpret_cast<std::uintptr_t>(p));
}
__device__ __forceinline__ unsigned g4(const void* p) {
    return *reinterpret_cast<const __attribute__((address_space(1))) unsigned*>(reinterpret_cast<std::uintptr_t>(p));
}
__device__ __forceinline__ f4 g16f(const void* p) {
    return *reinterpret_cast<const __attribute__((address_space(1))) f4*>(reinterpret_cast<std::uintptr_t>(p));
}

// Four int8 codes (one dword) times the FP16 scale: two dwords of FP16 values, codes 0,1 | 2,3.
// (code ^ 0x80) under the exponent byte 0x64 is the FP16 value 1024 + code + 128 (exact), minus 1152 is the code
// (exact); the FP16 multiply then rounds d * code once (RNE), the value the reference fp16(float(d) * code) has.
__device__ __forceinline__ u32x2 dequant4(unsigned codes, h2 scale) {
    const unsigned x = codes ^ 0x80808080U;
    const unsigned lo = __builtin_amdgcn_perm(0x64646464U, x, 0x04010400U), hi = __builtin_amdgcn_perm(0x64646464U, x, 0x04030402U);
    const h2 bias{static_cast<_Float16>(1152.0F), static_cast<_Float16>(1152.0F)};
    const h2 a = (__builtin_bit_cast(h2, lo) - bias) * scale, b = (__builtin_bit_cast(h2, hi) - bias) * scale;
    return u32x2{__builtin_bit_cast(unsigned, a), __builtin_bit_cast(unsigned, b)};
}

template <class Site> struct DenseVectorEpilogue : Site {
    static constexpr bool dense_vector_epilogue = true;
};

// Sites address row r at input0 + r * input_stride.
template <class Site> __device__ __forceinline__ const typename Site::Input* tiled_row_input(const Site& site, std::uint32_t r) {
    return reinterpret_cast<const typename Site::Input*>(site.input0 + r * site.input_stride);
}

// All rows' activations rounded to FP16 (RNE) into a16 [rows][width]; finite values beyond the FP16 range
// count into overflow (dense_f16_convert_kernel per element).
template <class Site>
__global__ __launch_bounds__(256) void convert_kernel(Site site, std::uint32_t rows, _Float16* __restrict__ a16, unsigned* __restrict__ overflow) {
    const std::uint32_t width = site.width, groups = width / 8U;
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * 256U + threadIdx.x;
    if (item >= static_cast<std::uint64_t>(rows) * groups) return;
    const auto r = static_cast<std::uint32_t>(item / groups);
    const auto g = static_cast<std::uint32_t>(item % groups);
    float x[8];
    load8(tiled_row_input(site, r) + 8U * g, x);
    _Float16 h[8];
    bool bad = false;
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) {
        h[k] = static_cast<_Float16>(x[k]);
        bad |= isfinite(x[k]) && !(fabsf(x[k]) <= 65504.0F);
    }
    uint4 word;
    __builtin_memcpy(&word, h, 16U);
    *reinterpret_cast<uint4*>(a16 + static_cast<std::uint64_t>(r) * scratch_pitch(width) + 8U * g) = word;
    if (bad) atomicAdd(overflow, 1U);
}

// Probe (fixture builds only, timing decomposition, wrong outputs): 1 = no B dequant (raw codes stored),
// 2 = no WMMA, 3 = WMMA + fragment reads only (no stage loads/stores).
// A modes: kPre = FP16 rows (convert_kernel scratch, rows rounded up to 256); kF32 = the site's FP32 rows converted
// while staging; kBf16 = the site's BF16 rows converted while staging. Inline modes read the last row for rows past the end.
constexpr unsigned kPre = 0U, kF32 = 1U, kBf16 = 2U;
// LDS-only barrier: the stage buffers are the only data shared through it (global loads are consumed by their own
// thread), so no global acquire (the __syncthreads fence adds a buffer_gl0_inv per barrier).
__device__ __forceinline__ void lds_barrier() { __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
// Tile shape of project_kernel: BN = 128 (8 waves, 80-byte padded LDS rows) or 256 (16 waves, 64-byte LDS rows whose
// 16-byte chunks are XOR-permuted by (row / 2) % 4 so the 8 rows one LDS phase serves hit distinct banks; two stages
// = 64 KiB). Waves are 64 x 64 (16 accumulators) in both.
template <unsigned BN> struct TileShape {
    static constexpr unsigned threads = BN == 256U ? 512U : 256U, waves_n = BN / kWN, pitch = BN == 256U ? 32U : kPitch;
    static constexpr unsigned stage_halves = (kBM + BN) * pitch;
    static __device__ __forceinline__ unsigned at(unsigned row, unsigned half) {
        if constexpr (BN == 256U) return row * 32U + (((half / 8U) ^ ((row / 2U) % 4U)) * 8U) + half % 8U;
        else return row * kPitch + half;
    }
};
// Probes 4 and 5 (fixture only): 4 = the full kernel without the epilogue stores (a guard that is never true keeps the
// accumulators live), 5 = probe 3 without the epilogue stores.
// Lean (form 4): the compute loop holds the B fragments of two output sub-tiles at a time (24 fragment VGPRs instead of
// 40, fenced per group) so that the form-4 kernel fits 192 VGPRs and two 256 x 256 blocks share a WGP. Every accumulator still receives
// its WMMAs in the same K16 order; only the order between different accumulators changes. Occupancy: Lean requests 8
// waves per SIMD (<= 192 VGPRs); the others request 1, below the minimum their work-group size implies, which the
// backend ignores (the same as __launch_bounds__(threads) alone). The body stays in the kernel: passing the kernel's
// site argument to a function makes a private copy whose loads are divergent.
// STRIX_QPF2K_WMMA_PIN (default on, =0 off; form 4): the Lean WMMAs as inline asm with A in v[183:190] (VGPR bank 3) and the
// two live B fragments in v[166:173] / v[158:165] (bank 2), the accumulators left to the allocator (gfx1151 WMMA issue:
// 34 -> 32 cycles when A, B and C start in different banks, LLVM #204254). The accumulator bank is NOT constrained: the
// current allocator puts it in bank 1, and tools/check_wmma_banks.py verifies distinct banks, C == D, D disjoint from A/B,
// zero scratch and <= 192 VGPRs on each built code object. "+&v": the WMMA's tied, early-clobber D (as LLVM's builtin), so
// D never shares registers with A/B. Same instruction on the same values: only the register assignment changes.
inline bool wmma_pin() {
    static const bool on = strix::envcfg::flag("STRIX_QPF2K_WMMA_PIN", true);
    return on;
}
inline bool gdn_vec_epilogue() {
    static const bool on = [] { const char* v = std::getenv("STRIX_QPF2K_GDN_VEC_EPILOGUE"); return !v || v[0] != '0' || v[1] != '\0'; }();
    return on;
}
template <unsigned BSlot> __device__ __forceinline__ f8 wmma_pinned(Frag a, Frag b, f8 c) {
    if constexpr (BSlot == 0U) asm("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0" : "+&v"(c) : "{v[183:190]}"(a), "{v[166:173]}"(b));
    else asm("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0" : "+&v"(c) : "{v[183:190]}"(a), "{v[158:165]}"(b));
    return c;
}
template <class Site, unsigned AMode, unsigned ProbeIn = 0U, unsigned BN = 128U, bool Lean = false, bool Pin = false>
__global__ __attribute__((amdgpu_flat_work_group_size(1, TileShape<BN>::threads), amdgpu_waves_per_eu(Lean ? 8U : 1U))) void project_kernel(Site site,
    std::uint32_t rows, const _Float16* __restrict__ a16, std::uint32_t raster_m, unsigned* __restrict__ overflow) {
    static_assert(!Lean || BN == 256U);
    constexpr unsigned Probe = ProbeIn == 5U ? 3U : ProbeIn == 4U ? 0U : ProbeIn;
    constexpr bool NoStores = ProbeIn >= 4U;
    using TS = TileShape<BN>;
    constexpr unsigned Threads = TS::threads, WavesN = TS::waves_n, StageH = TS::stage_halves;
    __shared__ __align__(16) _Float16 lds[2U * StageH];
    const unsigned thread = threadIdx.x, lane = thread % 32U;
    const unsigned wave = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(thread / 32U)));
    const unsigned wm = wave / WavesN, wn = wave % WavesN;
    const std::uint32_t width = site.width, outputs = site.outputs;
    const std::uint32_t tiles_m = (rows + kBM - 1U) / kBM, tiles_n = (outputs + BN - 1U) / BN;
    // Grouped rasterization: consecutive blocks take raster_m row tiles of one output tile, then the next output tile.
    const std::uint32_t pid = blockIdx.x, per_group = raster_m * tiles_n, group = pid / per_group, first_m = group * raster_m;
    const std::uint32_t group_m = min(raster_m, tiles_m - first_m), local = pid - group * per_group;
    const std::uint32_t m0 = (first_m + local % group_m) * kBM, n0 = (local / group_m) * BN;
    std::uint32_t stages = width / 32U;
    asm volatile("" : "+s"(stages));

    // A sources (32-bit offsets from a uniform base). Thread t stages values 8 (t % 4) .. + 7 of rows t / 4 + 64 j:
    // kPre / kBf16 one 16-byte load per row (chunk i = row j), kF32 two adjacent 16-byte loads (chunks 2 j, 2 j + 1),
    // so every mode stores one 16-byte LDS word per row and an FP32 overflow group of 8 sits in one thread.
    constexpr bool Inline = AMode != kPre;
    constexpr unsigned ARows = kBM * 4U / Threads;  // rows staged per thread
    constexpr unsigned AChunks = AMode == kF32 ? 2U * ARows : ARows;
    constexpr unsigned ARowStep = Threads / 4U;
    constexpr unsigned AStageBytes = AMode == kF32 ? 128U : 64U;
    const unsigned arow = thread / 4U, apart = thread % 4U;
    const unsigned char* const abase = Inline ? site.input0 : reinterpret_cast<const unsigned char*>(a16);
    const std::uint64_t astride = Inline ? site.input_stride : 2ULL * scratch_pitch(width);
    std::uint32_t aoff[AChunks];
#pragma unroll
    for (unsigned i = 0; i < AChunks; ++i) {
        const unsigned j = AMode == kF32 ? i / 2U : i;
        const std::uint32_t r = Inline ? min(m0 + arow + ARowStep * j, rows - 1U) : m0 + arow + ARowStep * j;
        aoff[i] = static_cast<std::uint32_t>(r * astride) + (AMode == kF32 ? 32U * apart + 16U * (i % 2U) : 16U * apart);
    }
    // Overflow counting (inline modes): blocks of output tile 0 count each group of 8 values (aligned at 8) holding a
    // finite value beyond the FP16 range, once per real row, as convert_kernel does per item.
    const bool count = Inline && overflow != nullptr && n0 == 0U;
    // B: thread t dequantizes K values 16 h .. 16 h + 15 of the stage's Q8_0 block of output n0 + t / 2. The block's
    // outputs lie in one segment (boundaries are multiples of 128, checked by supported()): uniform segment base.
    static_assert(Threads == 2U * BN);
    const unsigned bo = thread / 2U, bh = thread % 2U;
    const std::uint32_t rowbytes = width / 32U * 34U;
    const unsigned char* wbase;
    std::uint32_t boff;
    {
        unsigned seg; std::uint32_t row;
        static_cast<void>(segment_row(site.segments, n0, width, seg, row));
        wbase = site.segments.weights[seg];
        const std::uint32_t last = min(site.segments.first[seg + 1U], outputs) - 1U - site.segments.first[seg];
        boff = min(row + bo, last) * rowbytes + 16U * bh;
    }
    u32x4 areg[AChunks];
    unsigned bscale, bq[5];
    // Stage s = Q8 block s (34 bytes at 34 s; the row start is dword aligned). Even s: scale = low half of the dword
    // at 34 s, codes 16 h .. at 34 s + 2 + 16 h (two bytes past a dword: five dwords + alignbyte). Odd s: scale = high
    // half of the dword at 34 s - 2, codes at 34 s + 2 + 16 h (dword aligned: four dwords).
    const auto fetch = [&]<bool Odd>(std::uint32_t s) {
        if constexpr (Probe == 3U) return;
        // Lean: an opaque (uniform) stage index keeps every load at a uniform base + one 32-bit offset VGPR (no 64-bit
        // pointer induction variables per load stream).
        if constexpr (Lean) asm volatile("" : "+s"(s));
        const unsigned char* const astage = abase + AStageBytes * s;
#pragma unroll
        for (unsigned i = 0; i < AChunks; ++i) areg[i] = g16(astage + aoff[i]);
        const unsigned char* const block = wbase + 34U * s;
        if constexpr (Odd) {
            bscale = g4(block - 2U + (boff - 16U * bh));
            const u32x4 q = g16(block + 2U + boff);
            bq[0] = q.x; bq[1] = q.y; bq[2] = q.z; bq[3] = q.w;
        } else {
            bscale = g4(block + (boff - 16U * bh));
            const u32x4 q = g16(block + boff);
            bq[0] = q.x; bq[1] = q.y; bq[2] = q.z; bq[3] = q.w; bq[4] = g4(block + boff + 16U);
        }
    };
    const auto store = [&]<bool Odd>(unsigned buffer) {
        if constexpr (Probe == 3U) return;
        _Float16* as = lds + buffer * StageH;
        _Float16* bs = as + kBM * TS::pitch;
#pragma unroll
        for (unsigned i = 0; i < (AMode == kF32 ? AChunks / 2U : AChunks); ++i) {
            const auto pack = [](_Float16 a, _Float16 b) {
                return static_cast<unsigned>(__builtin_bit_cast(unsigned short, a)) | static_cast<unsigned>(__builtin_bit_cast(unsigned short, b)) << 16U;
            };
            const auto beyond = [](float v) { return isfinite(v) && !(fabsf(v) <= 65504.0F); };
            if constexpr (AMode == kF32) {
                const f4 x = __builtin_bit_cast(f4, areg[2U * i]), y = __builtin_bit_cast(f4, areg[2U * i + 1U]);
                if (count) {
                    bool bad = false;
#pragma unroll
                    for (unsigned k = 0; k < 4U; ++k) bad |= beyond(x[k]) || beyond(y[k]);
                    // Rows past the end repeat the last row: count real rows only.
                    if (bad && m0 + arow + ARowStep * i < rows) atomicAdd(overflow, 1U);
                }
                const u32x4 v{pack(static_cast<_Float16>(x.x), static_cast<_Float16>(x.y)), pack(static_cast<_Float16>(x.z), static_cast<_Float16>(x.w)),
                    pack(static_cast<_Float16>(y.x), static_cast<_Float16>(y.y)), pack(static_cast<_Float16>(y.z), static_cast<_Float16>(y.w))};
                *reinterpret_cast<u32x4*>(as + TS::at(arow + ARowStep * i, 8U * apart)) = v;
            } else if constexpr (AMode == kBf16) {
                // BF16 -> FP32 (exact) -> FP16 (RNE), the convert kernel's load8 + static_cast.
                float x[8];
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j) { x[2U * j] = __uint_as_float(areg[i][j] << 16U); x[2U * j + 1U] = __uint_as_float(areg[i][j] & 0xffff0000U); }
                if (count) {
                    bool bad = false;
#pragma unroll
                    for (unsigned k = 0; k < 8U; ++k) bad |= beyond(x[k]);
                    if (bad && m0 + arow + ARowStep * i < rows) atomicAdd(overflow, 1U);
                }
                u32x4 v;
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j) v[j] = pack(static_cast<_Float16>(x[2U * j]), static_cast<_Float16>(x[2U * j + 1U]));
                *reinterpret_cast<u32x4*>(as + TS::at(arow + ARowStep * i, 8U * apart)) = v;
            } else *reinterpret_cast<u32x4*>(as + TS::at(arow + ARowStep * i, 8U * apart)) = areg[i];
        }
        unsigned w[4];
        if constexpr (Odd) { w[0] = bq[0]; w[1] = bq[1]; w[2] = bq[2]; w[3] = bq[3]; }
        else {
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) w[j] = __builtin_amdgcn_alignbyte(bq[j + 1U], bq[j], 2U);
        }
        const unsigned short sbits = static_cast<unsigned short>(Odd ? bscale >> 16U : bscale & 0xffffU);
        const _Float16 sh = __builtin_bit_cast(_Float16, sbits);
        const h2 scale{sh, sh};
        u32x4 lo, hi;
        if constexpr (Probe == 1U) { lo = u32x4{w[0], w[1], w[2], w[3]}; hi = lo ^ bscale; }
        else {
            const u32x2 d0 = dequant4(w[0], scale), d1 = dequant4(w[1], scale), d2 = dequant4(w[2], scale), d3 = dequant4(w[3], scale);
            lo = u32x4{d0.x, d0.y, d1.x, d1.y}; hi = u32x4{d2.x, d2.y, d3.x, d3.y};
        }
        *reinterpret_cast<u32x4*>(bs + TS::at(bo, 16U * bh)) = lo;
        *reinterpret_cast<u32x4*>(bs + TS::at(bo, 16U * bh + 8U)) = hi;
    };
    f8 acc[kMT][kNT];
#pragma unroll
    for (unsigned t = 0; t < kMT; ++t)
#pragma unroll
        for (unsigned u = 0; u < kNT; ++u) acc[t][u] = f8{};
    const auto compute = [&](unsigned buffer) {
        const _Float16* as = lds + buffer * StageH;
        const _Float16* bs = as + kBM * TS::pitch;
        // Fragment of LDS row `row`, K halves 16 kt .. + 15 (two 16-byte chunks; contiguous in the padded layout).
        const auto frag = [](const _Float16* base, unsigned row, unsigned kt) {
            if constexpr (BN == 256U) {
                const u32x4 lo = *reinterpret_cast<const u32x4*>(base + TS::at(row, 16U * kt));
                const u32x4 hi = *reinterpret_cast<const u32x4*>(base + TS::at(row, 16U * kt + 8U));
                return Frag{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
            } else return *reinterpret_cast<const Frag*>(base + TS::at(row, 16U * kt));
        };
        if constexpr (Lean) {
            // Two B fragments per group, a scheduling fence after each group: without it the scheduler hoists the
            // fragment reads of later groups and spills accumulators at 192 VGPRs.
            constexpr unsigned NB = 2U;
            static_assert(!Pin || NB == 2U, "wmma_pinned holds two B fragments");
#pragma unroll
            for (unsigned kt = 0; kt < 2U; ++kt)
#pragma unroll
                for (unsigned up = 0; up < kNT; up += NB) {
                    Frag b[NB];
#pragma unroll
                    for (unsigned v = 0; v < NB; ++v) b[v] = frag(bs, wn * kWN + 16U * (up + v) + lane % 16U, kt);
#pragma unroll
                    for (unsigned t = 0; t < kMT; ++t) {
                        const Frag a = frag(as, wm * kWM + 16U * t + lane % 16U, kt);
#pragma unroll
                        for (unsigned v = 0; v < NB; ++v) {
                            if constexpr (Probe == 2U) acc[t][up + v][0] += __builtin_bit_cast(float, a[0] ^ b[v][1]);
                            else if constexpr (Pin) acc[t][up + v] = v == 0U ? wmma_pinned<0U>(a, b[v], acc[t][up + v]) : wmma_pinned<1U>(a, b[v], acc[t][up + v]);
                            else acc[t][up + v] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(h16, a), __builtin_bit_cast(h16, b[v]), acc[t][up + v]);
                        }
                    }
                    __builtin_amdgcn_sched_barrier(0);
                }
            return;
        }
#pragma unroll
        for (unsigned kt = 0; kt < 2U; ++kt) {
            Frag b[kNT];
#pragma unroll
            for (unsigned u = 0; u < kNT; ++u) b[u] = frag(bs, wn * kWN + 16U * u + lane % 16U, kt);
#pragma unroll
            for (unsigned t = 0; t < kMT; ++t) {
                const Frag a = frag(as, wm * kWM + 16U * t + lane % 16U, kt);
#pragma unroll
                for (unsigned u = 0; u < kNT; ++u) {
                    if constexpr (Probe == 2U) {
                        acc[t][u][0] += __builtin_bit_cast(float, a[0] ^ b[u][1]);
                    } else acc[t][u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(h16, a), __builtin_bit_cast(h16, b[u]), acc[t][u]);
                }
            }
        }
    };
    if constexpr (Probe != 0U) {
        for (unsigned j = thread; j < 2U * StageH; j += Threads) lds[j] = static_cast<_Float16>(0.0F);
        __syncthreads();
    }
    fetch.template operator()<false>(0U);
    store.template operator()<false>(0U);
    lds_barrier();
    // stages = width / 32 is even (width % 64 == 0): even stages use buffer 0, odd stages buffer 1.
#pragma unroll 1
    for (std::uint32_t s = 0; s < stages; s += 2U) {
        fetch.template operator()<true>(s + 1U);
        compute(0U);
        store.template operator()<true>(1U);
        lds_barrier();
        const bool more = s + 2U < stages;
        if (more) fetch.template operator()<false>(s + 2U);
        compute(1U);
        if (more) store.template operator()<false>(0U);
        lds_barrier();
    }
    if constexpr (requires { Site::dense_vector_epilogue; } && !NoStores) {
        // Reuse the dead A/B LDS arena. Each wave scatters a 16x64 FP32
        // accumulator strip, then reads four adjacent output columns per lane.
        // XOR the low row bit into column bit 4 to avoid two-way LDS conflicts.
        // 16 waves * 16 rows * 64 floats = the existing 64 KiB, no new LDS.
        auto* image = reinterpret_cast<float*>(lds);
        const auto at = [=](unsigned row, unsigned col) {
            return wave * 1024U + row * 64U + (col ^ ((row & 1U) * 16U));
        };
#pragma unroll
        for (unsigned t = 0; t < kMT; ++t) {
#pragma unroll
            for (unsigned u = 0; u < kNT; ++u)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e)
                    image[at(2U * e + lane / 16U, 16U * u + lane % 16U)] = acc[t][u][e];
            lds_barrier();
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) {
                const unsigned linear = 4U * lane + 128U * j, rr = linear / 64U, cc = linear % 64U;
                const auto r = m0 + wm * kWM + 16U * t + rr, out = n0 + wn * kWN + cc;
                const f4 value = *reinterpret_cast<const f4*>(image + at(rr, cc));
                if (r < rows && out + 3U < outputs) site.store4(out, r, value);
            }
            lds_barrier(); // every reader finishes before the next strip
        }
        return;
    }
#pragma unroll
    for (unsigned t = 0; t < kMT; ++t)
#pragma unroll
        for (unsigned u = 0; u < kNT; ++u) {
            const std::uint32_t out = n0 + wn * kWN + 16U * u + lane % 16U;
            if (out >= outputs) continue;
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const std::uint32_t r = m0 + wm * kWM + 16U * t + 2U * e + lane / 16U;
                if constexpr (NoStores) {
                    if (__builtin_bit_cast(unsigned, acc[t][u][e]) == 0x7fc0deadU) site.store(out, r, acc[t][u][e]);
                } else if (r < rows) site.store(out, r, acc[t][u][e]);
            }
        }
}

// Form 1/2 (STRIX_QPF2K_DENSE_FORM): the same 256 x 128 block tile, operands, K16 order and stores, run by 16 waves
// (512 threads, waves of 64 x WN) so that every thread stages half the bytes and Depth stages of global loads stay in
// flight (t1: the 8-wave form waits on its one-stage prefetch; WMMA + fragments alone reach 98 %). Lower register
// blocking per wave (WN = 32: 8 accumulators, one fragment per WMMA) for twice the resident waves.
template <class Site, unsigned AMode, unsigned Threads, unsigned WN, unsigned Depth, unsigned Probe = 0U>
__global__ __launch_bounds__(Threads) void project2_kernel(Site site, std::uint32_t rows, const _Float16* __restrict__ a16,
    std::uint32_t raster_m, unsigned* __restrict__ overflow) {
    constexpr unsigned Waves = Threads / 32U, WavesN = kBN / WN, WM = kBM / (Waves / WavesN), MT = WM / 16U, NT = WN / 16U;
    static_assert(Waves % WavesN == 0U && WM % 16U == 0U && (Depth == 1U || Depth == 2U));
    __shared__ __align__(16) _Float16 lds[2U * kStageHalves];
    const unsigned thread = threadIdx.x, lane = thread % 32U;
    const unsigned wave = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(thread / 32U)));
    const unsigned wm = wave / WavesN, wn = wave % WavesN;
    const std::uint32_t width = site.width, outputs = site.outputs;
    const std::uint32_t tiles_m = (rows + kBM - 1U) / kBM, tiles_n = (outputs + kBN - 1U) / kBN;
    const std::uint32_t pid = blockIdx.x, per_group = raster_m * tiles_n, group = pid / per_group, first_m = group * raster_m;
    const std::uint32_t group_m = min(raster_m, tiles_m - first_m), local = pid - group * per_group;
    const std::uint32_t m0 = (first_m + local % group_m) * kBM, n0 = (local / group_m) * kBN;
    std::uint32_t stages = width / 32U;
    asm volatile("" : "+s"(stages));
    // A: 16-byte chunks, CPR per row (4: FP16/BF16, 8: FP32); chunk i of thread t = row t / CPR + (Threads / CPR) i.
    constexpr bool Inline = AMode != kPre;
    constexpr unsigned CPR = AMode == kF32 ? 8U : 4U, AChunks = kBM * CPR / Threads, ARowStep = Threads / CPR;
    constexpr unsigned AStageBytes = CPR * 16U;
    const unsigned arow = thread / CPR, apart = thread % CPR;
    const unsigned char* const abase = Inline ? site.input0 : reinterpret_cast<const unsigned char*>(a16);
    const std::uint64_t astride = Inline ? site.input_stride : 2ULL * scratch_pitch(width);
    std::uint32_t aoff[AChunks];
#pragma unroll
    for (unsigned i = 0; i < AChunks; ++i) {
        const std::uint32_t r = Inline ? min(m0 + arow + ARowStep * i, rows - 1U) : m0 + arow + ARowStep * i;
        aoff[i] = static_cast<std::uint32_t>(r * astride) + 16U * apart;
    }
    const bool count = Inline && overflow != nullptr && n0 == 0U;
    // B: TPO threads per output, VPT = 32 / TPO codes each (K values VPT h .. of the stage's block of output n0 + t / TPO).
    constexpr unsigned TPO = Threads / kBN, VPT = 32U / TPO, CW = VPT / 4U;  // code dwords per thread
    static_assert(TPO * kBN == Threads && (VPT == 8U || VPT == 16U));
    const unsigned bo = thread / TPO, bh = thread % TPO;
    const std::uint32_t rowbytes = width / 32U * 34U;
    const unsigned char* wbase;
    std::uint32_t boff;  // the output's row in its segment + VPT h
    {
        unsigned seg; std::uint32_t row;
        static_cast<void>(segment_row(site.segments, n0, width, seg, row));
        wbase = site.segments.weights[seg];
        const std::uint32_t last = min(site.segments.first[seg + 1U], outputs) - 1U - site.segments.first[seg];
        boff = min(row + bo, last) * rowbytes + VPT * bh;
    }
    struct Stage { u32x4 a[AChunks]; unsigned scale, q[CW + 1U]; };
    Stage reg[Depth];
    // Even block s (dword aligned): scale = low half of its first dword, codes 2 bytes past VPT h (CW + 1 dwords +
    // alignbyte). Odd block (2 bytes past a dword): scale = high half of the dword 2 bytes before it, codes dword aligned.
    const auto fetch = [&](std::uint32_t s, Stage& r) {
        if constexpr (Probe == 3U) return;
        const unsigned char* const astage = abase + AStageBytes * s;
#pragma unroll
        for (unsigned i = 0; i < AChunks; ++i) r.a[i] = g16(astage + aoff[i]);
        const unsigned char* const block = wbase + 34U * s;
        const unsigned row_start = boff - VPT * bh;
        if (s & 1U) {
            r.scale = g4(block - 2U + row_start);
#pragma unroll
            for (unsigned j = 0; j < CW; ++j) r.q[j] = g4(block + 2U + boff + 4U * j);
        } else {
            r.scale = g4(block + row_start);
#pragma unroll
            for (unsigned j = 0; j <= CW; ++j) r.q[j] = g4(block + boff + 4U * j);
        }
    };
    const auto store = [&](std::uint32_t s, const Stage& r, unsigned buffer) {
        if constexpr (Probe == 3U) return;
        _Float16* as = lds + buffer * kStageHalves;
        _Float16* bs = as + kBM * kPitch;
        const auto pack = [](_Float16 a, _Float16 b) {
            return static_cast<unsigned>(__builtin_bit_cast(unsigned short, a)) | static_cast<unsigned>(__builtin_bit_cast(unsigned short, b)) << 16U;
        };
        const auto beyond = [](float v) { return isfinite(v) && !(fabsf(v) <= 65504.0F); };
#pragma unroll
        for (unsigned i = 0; i < AChunks; ++i) {
            const unsigned row = arow + ARowStep * i;
            if constexpr (AMode == kF32) {
                const f4 x = __builtin_bit_cast(f4, r.a[i]);
                if (count) {
                    const bool bad = beyond(x.x) || beyond(x.y) || beyond(x.z) || beyond(x.w);
                    const int partner = __shfl_xor(static_cast<int>(bad), 1);
                    if ((bad || partner != 0) && apart % 2U == 0U && m0 + row < rows) atomicAdd(overflow, 1U);
                }
                const u32x2 v{pack(static_cast<_Float16>(x.x), static_cast<_Float16>(x.y)), pack(static_cast<_Float16>(x.z), static_cast<_Float16>(x.w))};
                *reinterpret_cast<u32x2*>(as + row * kPitch + 4U * apart) = v;
            } else if constexpr (AMode == kBf16) {
                float x[8];
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j) { x[2U * j] = __uint_as_float(r.a[i][j] << 16U); x[2U * j + 1U] = __uint_as_float(r.a[i][j] & 0xffff0000U); }
                if (count) {
                    bool bad = false;
#pragma unroll
                    for (unsigned k = 0; k < 8U; ++k) bad |= beyond(x[k]);
                    if (bad && m0 + row < rows) atomicAdd(overflow, 1U);
                }
                u32x4 v;
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j) v[j] = pack(static_cast<_Float16>(x[2U * j]), static_cast<_Float16>(x[2U * j + 1U]));
                *reinterpret_cast<u32x4*>(as + row * kPitch + 8U * apart) = v;
            } else *reinterpret_cast<u32x4*>(as + row * kPitch + 8U * apart) = r.a[i];
        }
        unsigned w[CW];
        if (s & 1U) {
#pragma unroll
            for (unsigned j = 0; j < CW; ++j) w[j] = r.q[j];
        } else {
#pragma unroll
            for (unsigned j = 0; j < CW; ++j) w[j] = __builtin_amdgcn_alignbyte(r.q[j + 1U], r.q[j], 2U);
        }
        const unsigned short sbits = static_cast<unsigned short>((s & 1U) ? r.scale >> 16U : r.scale & 0xffffU);
        const _Float16 sh = __builtin_bit_cast(_Float16, sbits);
        const h2 scale{sh, sh};
#pragma unroll
        for (unsigned j = 0; j < CW; j += 2U) {
            u32x4 v;
            if constexpr (Probe == 1U) v = u32x4{w[j], w[j + 1U], w[j] ^ r.scale, w[j + 1U]};
            else {
                const u32x2 d0 = dequant4(w[j], scale), d1 = dequant4(w[j + 1U], scale);
                v = u32x4{d0.x, d0.y, d1.x, d1.y};
            }
            *reinterpret_cast<u32x4*>(bs + bo * kPitch + VPT * bh + 4U * j) = v;
        }
    };
    f8 acc[MT][NT];
#pragma unroll
    for (unsigned t = 0; t < MT; ++t)
#pragma unroll
        for (unsigned u = 0; u < NT; ++u) acc[t][u] = f8{};
    const auto compute = [&](unsigned buffer) {
        const _Float16* as = lds + buffer * kStageHalves;
        const _Float16* bs = as + kBM * kPitch;
#pragma unroll
        for (unsigned kt = 0; kt < 2U; ++kt) {
            Frag b[NT];
#pragma unroll
            for (unsigned u = 0; u < NT; ++u) b[u] = *reinterpret_cast<const Frag*>(bs + (wn * WN + 16U * u + lane % 16U) * kPitch + 16U * kt);
#pragma unroll
            for (unsigned t = 0; t < MT; ++t) {
                const Frag a = *reinterpret_cast<const Frag*>(as + (wm * WM + 16U * t + lane % 16U) * kPitch + 16U * kt);
#pragma unroll
                for (unsigned u = 0; u < NT; ++u) {
                    if constexpr (Probe == 2U) acc[t][u][0] += __builtin_bit_cast(float, a[0] ^ b[u][1]);
                    else acc[t][u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(h16, a), __builtin_bit_cast(h16, b[u]), acc[t][u]);
                }
            }
        }
    };
    if constexpr (Probe != 0U) {
        for (unsigned j = thread; j < 2U * kStageHalves; j += Threads) lds[j] = static_cast<_Float16>(0.0F);
        __syncthreads();
    }
    // Pipeline: stage s computes from LDS buffer s % 2 while the loads of stages s + 1 .. s + Depth are in flight
    // (register sets reg[(s + 1) % Depth] ..); after the compute, stage s + 1 goes to LDS buffer (s + 1) % 2.
    fetch(0U, reg[0]);
    if constexpr (Depth == 2U) if (stages > 1U) fetch(1U, reg[1]);
    store(0U, reg[0], 0U);
    lds_barrier();
    if constexpr (Depth == 2U) { if (stages > 2U) fetch(2U, reg[0]); }
    else if (stages > 1U) fetch(1U, reg[0]);
    // stages is even: unrolled by 2 so the buffer and register-set indices are compile-time.
#pragma unroll 1
    for (std::uint32_t s = 0; s < stages; s += 2U) {
        compute(0U);
        // stage s + 1 is in reg[Depth == 2 ? 1 : 0]
        store(s + 1U, reg[Depth == 2U ? 1U : 0U], 1U);
        lds_barrier();
        if constexpr (Depth == 2U) { if (s + 3U < stages) fetch(s + 3U, reg[1]); }
        else if (s + 2U < stages) fetch(s + 2U, reg[0]);
        compute(1U);
        if (s + 2U < stages) store(s + 2U, reg[0], 0U);
        lds_barrier();
        if constexpr (Depth == 2U) { if (s + 4U < stages) fetch(s + 4U, reg[0]); }
        else if (s + 3U < stages) fetch(s + 3U, reg[0]);
    }
#pragma unroll
    for (unsigned t = 0; t < MT; ++t)
#pragma unroll
        for (unsigned u = 0; u < NT; ++u) {
            const std::uint32_t out = n0 + wn * WN + 16U * u + lane % 16U;
            if (out >= outputs) continue;
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const std::uint32_t r = m0 + wm * WM + 16U * t + 2U * e + lane / 16U;
                if (r < rows) site.store(out, r, acc[t][u][e]);
            }
        }
}

// FP16 activation scratch per stream (kPre mode), grown on demand (the stream is drained first).
inline _Float16* scratch(hipStream_t stream, std::uint64_t halves) {
    struct Slot { void* p = nullptr; std::uint64_t bytes = 0; };
    static std::mutex mutex;
    static std::map<hipStream_t, Slot> slots;
    const std::lock_guard lock(mutex);
    auto& s = slots[stream];
    if (s.bytes < halves * 2U) {
        if (s.p) {
            if (hipStreamSynchronize(stream) != hipSuccess || hipFree(s.p) != hipSuccess)
                throw std::runtime_error("q8_rows tiled: FP16 scratch release failed");
            s = {};
        }
        if (hipMalloc(&s.p, halves * 2U) != hipSuccess) throw std::runtime_error("q8_rows tiled: FP16 scratch allocation failed");
        s.bytes = halves * 2U;
    }
    return static_cast<_Float16*>(s.p);
}

// Kernel form (STRIX_QPF2K_DENSE_FORM overrides every site, STRIX_QPF2K_DENSE_FORM_PRE the FP16-scratch sites only):
// 0 = project_kernel (8 waves of 64 x 64, one stage in flight), 1 = project2_kernel 16 waves of 64 x 32 with two stages
// in flight, 2 = the same with one stage in flight, 3 = project_kernel with 256 x 256 blocks (16 waves of 64 x 64, 40 %
// fewer staged bytes per FLOP than form 0), 4 = form 3 at <= 192 VGPRs, two blocks per WGP (project_kernel Lean).
// Defaults: form 4 for FP16-scratch A where the site declares it (GDN in: two co-resident blocks hide the 256 KiB
// epilogue of each tile), else form 3; form 3 for inline A (t7: fastest there). Forms 3/4 need 256-aligned segments, else form 0; form 4 with
// FP32 A runs form 3 (inline FP32 staging spills at 192 VGPRs).
// Per-site default for FP16-scratch A: Site::qpf2k_pre_form if the site declares it, else 3. t8 (interleaved, rev 6):
// form 4 is faster for GDN in at T 2048/4096/8192 (-2.2 .. -4.1 %) but slower for QSA q|k|v at T 8192 (+2.4 / +2.7 %,
// under study), so only the GDN input site declares 4.
template <class Site> unsigned form(unsigned mode) {
    const auto read = [](const char* name) { const char* v = std::getenv(name); const int n = v ? std::atoi(v) : -1; return n >= 0 && n <= 4 ? static_cast<unsigned>(n) : ~0U; };
    static const unsigned all = read("STRIX_QPF2K_DENSE_FORM"), pre = read("STRIX_QPF2K_DENSE_FORM_PRE");
    if (mode == kPre && pre != ~0U) return pre;
    if (all != ~0U) return all;
    if (mode == kF32) return 3U;
    if (mode == kBf16) return 4U;  // rev 6c (falls back to form 0 on segments not 256-aligned: shared gate|up)
    if constexpr (requires { Site::qpf2k_pre_form; }) return Site::qpf2k_pre_form;
    else return 3U;
}

// A mode of a site: FP32 rows convert while staging; BF16 rows convert while staging when the outputs are few (the
// convert launch's DRAM bytes would cost more than the staging VALU: ~400 / outputs vs ~8 % of the projection), else
// one convert launch. STRIX_QPF2K_DENSE_INLINE = 0 (convert launch) / 1 (inline) overrides the BF16 choice.
template <class Site> unsigned mode_of(const Site& site) {
    if constexpr (sizeof(typename Site::Input) == 4U) return kF32;
    else {
        static const int forced = strix::envcfg::integer("STRIX_QPF2K_DENSE_INLINE", -1);
        if (forced == 0 || forced == 1) return forced ? kBf16 : kPre;
        // rev 6c: BF16 rows always convert while staging. With form 4 (two blocks per WGP) the staging VALU is hidden
        // and the convert launch's DRAM pass is gone (t8 bench: GDN in 17.02 vs 17.80 ms, QSA q|k|v 14.31 vs 15.23 ms
        // against rev 5's convert + form 3). STRIX_QPF2K_DENSE_INLINE=0 restores the convert launch for >= 4096 outputs.
        static_cast<void>(site);
        return kBf16;
    }
}

// Requirements of the tiled form (else the caller keeps its launches): width a positive multiple of 64 up to 8192,
// dword-aligned Q8 segments whose boundaries are multiples of 128, 32-bit operand offsets, at least one row.
template <class Site> bool supported(const Site& site, std::uint32_t rows, unsigned mode) {
    if (!rows || !dense_a8_width(site.width) || !site.outputs || !site.segments.count || site.segments.count > kMaxSegments) return false;
    for (unsigned m = 0; m < site.segments.count; ++m) {
        if (reinterpret_cast<std::uintptr_t>(site.segments.weights[m]) % 4U) return false;
        if (m && site.segments.first[m] % kBN) return false;  // a 128-output tile never straddles two segments
    }
    if (site.segments.first[site.segments.count] != site.outputs) return false;
    const std::uint64_t rows_up = (static_cast<std::uint64_t>(rows) + kBM - 1U) / kBM * kBM;
    const std::uint64_t row_bytes = sizeof(typename Site::Input) * static_cast<std::uint64_t>(site.width);
    if (mode != kPre && (static_cast<std::uint64_t>(rows) - 1U) * site.input_stride + row_bytes >= (1ULL << 32U)) return false;
    if (mode == kPre && rows_up * scratch_pitch(site.width) * 2U >= (1ULL << 32U)) return false;
    if (static_cast<std::uint64_t>(site.outputs) * (site.width / 32U) * 34U >= (1ULL << 32U)) return false;
    return true;
}
template <class Site> bool supported(const Site& site, std::uint32_t rows) { return supported(site, rows, mode_of(site)); }

// One launch over all rows (plus the convert launch in kPre mode). The site supplies Input (std::uint16_t BF16 or
// float), input0 / input_stride (row r at input0 + r * input_stride), segments, width, outputs and store(out, r, value).
// The caller checks supported() first. any_order: the caller proved the first kernel (the convert, or the projection
// itself when inline) independent of the kernel queued just before it, as launch_dense_f16's convert.
// Launch of one projection kernel (form, A mode) over all rows.
template <class Site, unsigned Mode, unsigned Probe = 0U>
void launch_projection(const Site& site, std::uint32_t rows, hipStream_t stream, bool any_order, unsigned form_id, std::uint32_t raster_m,
    const _Float16* a16) {
    if (form_id >= 3U)  // 256-output tiles: every segment boundary a multiple of 256, else form 0 (same bits)
        for (unsigned m = 1; m < site.segments.count; ++m) if (site.segments.first[m] % 256U) form_id = 0U;
    if (Mode == kF32 && form_id == 4U) form_id = 3U;  // FP32 A staged inline does not fit 192 VGPRs: form 3 (same bits)
    if constexpr (Probe == 0U && requires { site.vector_store_supported(); } && !requires { Site::dense_vector_epilogue; }) {
        // Unpinned scratch-A + Q8 + the vector epilogue spills; preserve the
        // original scalar route there. Decide once before wrapping the site.
        if (gdn_vec_epilogue() && (Mode != kPre || wmma_pin()) &&
            (form_id == 4U || (Mode == kBf16 && form_id == 0U)) && site.vector_store_supported()) {
            DenseVectorEpilogue<Site> vector_site{};
            static_cast<Site&>(vector_site) = site;
            launch_projection<decltype(vector_site), Mode, Probe>(vector_site, rows, stream, any_order, form_id, raster_m, a16);
            return;
        }
    }
    const std::uint32_t bn = form_id >= 3U ? 256U : kBN;
    const std::uint32_t tiles = ((rows + kBM - 1U) / kBM) * ((site.outputs + bn - 1U) / bn);
    unsigned* const overflow = Mode == kPre ? nullptr : dense_f16_overflow_counter();
    const auto go = [&](auto kernel, unsigned threads) {
        if (any_order)
            hipExtLaunchKernelGGL(kernel, dim3(tiles), dim3(threads), 0, stream, nullptr, nullptr, hipExtAnyOrderLaunch, site, rows, a16, raster_m, overflow);
        else hipLaunchKernelGGL(kernel, dim3(tiles), dim3(threads), 0, stream, site, rows, a16, raster_m, overflow);
    };
    if constexpr (requires { Site::dense_vector_epilogue; }) {
        const auto lean = [&] {
            if constexpr (Mode != kF32) {
                if constexpr (Mode == kPre)
                    go(project_kernel<Site, Mode, Probe, 256U, true, true>, 512U);
                else {
                    if (wmma_pin()) go(project_kernel<Site, Mode, Probe, 256U, true, true>, 512U);
                    else go(project_kernel<Site, Mode, Probe, 256U, true>, 512U);
                }
            }
        };
        if constexpr (Mode == kF32) go(project_kernel<Site, Mode, Probe, 256U>, 512U);
        else if constexpr (Mode == kPre) lean();
        else {
            if (form_id == 0U) go(project_kernel<Site, Mode, Probe>, kThreadsT);
            else lean();
        }
    } else {
        if (form_id == 1U) go(project2_kernel<Site, Mode, 512U, 32U, 2U, Probe>, 512U);
        else if (form_id == 2U) go(project2_kernel<Site, Mode, 512U, 32U, 1U, Probe>, 512U);
        else if (form_id == 3U) go(project_kernel<Site, Mode, Probe, 256U>, 512U);
        else if (form_id == 4U) {
            if constexpr (Mode != kF32) {
                if (wmma_pin()) go(project_kernel<Site, Mode, Probe, 256U, true, true>, 512U);
                else go(project_kernel<Site, Mode, Probe, 256U, true>, 512U);
            }
        }
        else go(project_kernel<Site, Mode, Probe>, kThreadsT);
    }
}
// One launch over all rows (plus the convert launch in kPre mode). The site supplies Input (std::uint16_t BF16 or
// float), input0 / input_stride (row r at input0 + r * input_stride), segments, width, outputs and store(out, r, value).
// The caller checks supported() first. any_order: the caller proved the first kernel (the convert, or the projection
// itself when inline) independent of the kernel queued just before it, as launch_dense_f16's convert.
template <class Site, unsigned Probe = 0U>
void launch(const Site& site, std::uint32_t rows, hipStream_t stream, bool any_order = false, unsigned mode = ~0U, unsigned raster_m = 0U,
    unsigned form_id = ~0U) {
    if (mode == ~0U) mode = mode_of(site);
    if (form_id == ~0U) {
        form_id = form<Site>(mode);
        // Default form 4 only up to K 2560: above it the 40 resident blocks' B tiles (256 x K x 34/32 B each) outgrow the
        // 32 MB MALL (mi t1 K sweep: form4/form3 0.995 @2560, 1.075 @3072, 1.61 @4096..5120). No engine site is affected
        // today (form-4 sites are K 2560); an explicit form_id is kept as given (fixture sweeps). Same bits either way.
        if (form_id == 4U && site.width > 2560U) form_id = 3U;
    }
    // Forms 3/4 tile 256 outputs: every segment boundary must be a multiple of 256 (else form 0, same bits).
    if (form_id >= 3U)
        for (unsigned m = 1; m < site.segments.count; ++m) if (site.segments.first[m] % 256U) form_id = 0U;
    if (mode == kF32 && form_id == 4U) form_id = 3U;  // as launch_projection (the trace shows the form that runs)
    if (!supported(site, rows, mode)) throw std::invalid_argument("q8_rows tiled: unsupported dense projection");
    if (!raster_m) raster_m = raster(mode);
    // STRIX_QPF2K_DENSE_TRACE=1: one stderr line per site type on its first launch (route witness).
    static const bool trace = strix::envcfg::flag("STRIX_QPF2K_DENSE_TRACE", false);
    if (trace) {
        static std::atomic<unsigned> launches{0};
        if (launches.fetch_add(1U, std::memory_order_relaxed) == 0U)
            std::fprintf(stderr, "qpf2k dense tiled: first launch width %u outputs %u segments %u rows %u mode %u raster %u form %u\n", site.width,
                site.outputs, site.segments.count, rows, mode, raster_m, form_id);
    }
    if (mode == kPre) {
        _Float16* a16 = scratch(stream, (static_cast<std::uint64_t>(rows) + kBM - 1U) / kBM * kBM * scratch_pitch(site.width));
        const std::uint64_t items = static_cast<std::uint64_t>(rows) * (site.width / 8U);
        const dim3 grid(static_cast<unsigned>((items + 255U) / 256U));
        if (any_order)
            hipExtLaunchKernelGGL(HIP_KERNEL_NAME(convert_kernel<Site>), grid, dim3(256U), 0, stream, nullptr, nullptr, hipExtAnyOrderLaunch,
                site, rows, a16, dense_f16_overflow_counter());
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(convert_kernel<Site>), grid, dim3(256U), 0, stream, site, rows, a16, dense_f16_overflow_counter());
        launch_projection<Site, kPre, Probe>(site, rows, stream, false, form_id, raster_m, a16);
        return;
    }
    if constexpr (sizeof(typename Site::Input) == 4U) launch_projection<Site, kF32, Probe>(site, rows, stream, any_order, form_id, raster_m, nullptr);
    else launch_projection<Site, kBf16, Probe>(site, rows, stream, any_order, form_id, raster_m, nullptr);
}

}  // namespace strix::q8_rows::tiled
