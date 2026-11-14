#pragma once
// Included by hip_q8_rows_gemv.hpp inside namespace strix::q8_rows.
//
// Dense FP16 projections of fast-prefill-f16wmma-v1 (prefill launches of 32..256 rows, a multiple of 32; not the
// exact or dense A8 contracts): y[r][n] = sum_k fp16(x[r][k]) * fp16(d[n][b] * q[n][k]) with
// v_wmma_f32_16x16x16_f16 (FP32 accumulation over the whole width, hardware order inside a tile).
//   - x: the site's BF16/FP32 activations rounded to FP16 (RNE) by dense_f16_convert_kernel into a
//     per-family scratch slot (rows x width, row-major); finite values beyond the FP16 range count
//     into dense_f16_overflow_count() (they become +-inf).
//   - weights: the effective Q8_0 value d * q (exact in FP32) rounded to FP16 (RNE), dequantized per
//     block into LDS by the projection itself; no FP16 weight copy exists in memory.
// A block owns BM rows x BN outputs (8 waves of WM x WN); per 64-feature stage (one Q8 block pair)
// the rows' FP16 values and the dequantized weights sit in LDS; operands of the next stage are
// prefetched into registers while the current one is multiplied.

typedef _Float16 dense_f16_v16h __attribute__((ext_vector_type(16)));
typedef float dense_f16_v8f __attribute__((ext_vector_type(8)));
constexpr unsigned kDenseF16Pitch = 72U;  // halves per LDS row of one 64-feature stage (144 bytes)

inline unsigned* dense_f16_overflow_counter() {
    static unsigned* const counter = [] {
        void* p = nullptr;
        if (hipMalloc(&p, 256U) != hipSuccess || hipMemset(p, 0, 256U) != hipSuccess)
            throw std::runtime_error("q8_rows: dense FP16 overflow counter allocation failed");
        return static_cast<unsigned*>(p);
    }();
    return counter;
}
// Activations that overflowed FP16 so far (synchronizes the device).
inline unsigned dense_f16_overflow_count() {
    unsigned value = 0;
    if (hipDeviceSynchronize() != hipSuccess ||
        hipMemcpy(&value, dense_f16_overflow_counter(), sizeof(value), hipMemcpyDeviceToHost) != hipSuccess)
        throw std::runtime_error("q8_rows: dense FP16 overflow counter read failed");
    return value;
}
// Two FP16 activation slots per family (rows <= 256, width <= 8192), alternating per launch.
inline _Float16* dense_f16_slot(unsigned family) {
    constexpr std::uint64_t slot_halves = std::uint64_t{kDenseA8WideRows} * kDenseA8MaxWidth;
    static _Float16* const pool = [] {
        void* p = nullptr;
        if (hipMalloc(&p, slot_halves * 2U * 2U * kDenseA8Families) != hipSuccess)
            throw std::runtime_error("q8_rows: dense FP16 scratch allocation failed");
        return static_cast<_Float16*>(p);
    }();
    static std::atomic<unsigned> turn[kDenseA8Families]{};
    family %= kDenseA8Families;
    const unsigned slot = family * 2U + (turn[family].fetch_add(1U, std::memory_order_relaxed) & 1U);
    return pool + slot * slot_halves;
}

// Launches of more than kDenseA8WideRows rows (STRIX_INTPREFILL_DENSE_ROWS strided sites, up to kDenseF16BigRows):
// two shared FP16 activation slots of kDenseF16BigRows x kDenseA8MaxWidth halves, alternating per launch.
constexpr unsigned kDenseF16BigRows = 1024U;
inline _Float16* dense_f16_big_slot() {
    constexpr std::uint64_t slot_halves = std::uint64_t{kDenseF16BigRows} * kDenseA8MaxWidth;
    static _Float16* const pool = [] {
        void* p = nullptr;
        if (hipMalloc(&p, slot_halves * 2U * 2U) != hipSuccess) throw std::runtime_error("q8_rows: dense FP16 large scratch allocation failed");
        return static_cast<_Float16*>(p);
    }();
    static std::atomic<unsigned> turn{};
    return pool + (turn.fetch_add(1U, std::memory_order_relaxed) & 1U) * slot_halves;
}

template <class Site>
__global__ __launch_bounds__(kThreads) void dense_f16_convert_kernel(Site site, _Float16* __restrict__ a16, unsigned* __restrict__ overflow) {
    const std::uint32_t width = site.width, groups = width / 8U;
    const std::uint32_t item = blockIdx.x * kThreads + threadIdx.x;
    if (item >= Site::rows * groups) return;
    const unsigned r = item / groups;
    const std::uint32_t g = item % groups;
    float x[8];
    load8(row_input(site, r) + 8U * g, x);
    _Float16 h[8];
    bool bad = false;
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) {
        h[k] = static_cast<_Float16>(x[k]);
        bad |= isfinite(x[k]) && !(fabsf(x[k]) <= 65504.0F);
    }
    uint4 word;
    __builtin_memcpy(&word, h, 16U);
    *reinterpret_cast<uint4*>(a16 + static_cast<std::uint64_t>(r) * width + 8U * g) = word;
    if (bad) atomicAdd(overflow, 1U);
}

// LDS half index of (row, half) in one 64-feature stage. Tune bit 0: 64-half rows whose 16-byte chunks are
// XOR-permuted by row % 8, so the 8 rows a 128-bit LDS load serves per pass hit distinct banks (an unpermuted
// 128-byte pitch maps them onto the same banks: 8-way conflicts). Otherwise the padded 72-half pitch. Layout only.
template <unsigned Tune> __device__ __forceinline__ unsigned dense_f16_lds(unsigned row, unsigned half) {
    if constexpr ((Tune & 1U) != 0U) return row * 64U + ((half / 8U) ^ (row % 8U)) * 8U + half % 8U;
    else return row * kDenseF16Pitch + half;
}
// Fragment of one LDS row: the 8 halves at lo and the 8 at hi (halves from the row start).
__device__ __forceinline__ dense_f16_v16h dense_f16_pair(const _Float16* row, unsigned lo_half, unsigned hi_half) {
    dense_f16_v16h v;
    const uint4 lo = *reinterpret_cast<const uint4*>(row + lo_half), hi = *reinterpret_cast<const uint4*>(row + hi_half);
    __builtin_memcpy(&v, &lo, 16U);
    __builtin_memcpy(reinterpret_cast<char*>(&v) + 16, &hi, 16U);
    return v;
}

template <class Site, unsigned BM, unsigned BN, unsigned WM, unsigned WN, unsigned Tune = 0U>
__global__ __launch_bounds__(256) void project_rows_dense_f16_wmma_kernel(Site site, const _Float16* __restrict__ a16) {
    constexpr unsigned Rows = Site::rows, WC = BN / WN, TM = WM / 16U, TN = WN / 16U;
    constexpr unsigned Pitch = (Tune & 1U) ? 64U : kDenseF16Pitch;  // halves per LDS row (see dense_f16_lds)
    static_assert((BM / WM) * WC == 8U && Rows % BM == 0U && WM % 16U == 0U && WN % 16U == 0U);
    constexpr unsigned ALoads = BM * 8U / 256U;            // 16-byte A loads per thread per stage
    constexpr unsigned BItems = BN * 2U;                   // Q8 blocks per stage
    static_assert(ALoads >= 1U && BItems <= 256U);
    __shared__ __align__(16) _Float16 as[BM * Pitch];
    __shared__ __align__(16) _Float16 bs[BN * Pitch];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U, wr = wave / WC, wc = wave % WC;
    const std::uint32_t width = site.width, outputs = site.outputs, stages = width / 64U;
    const unsigned ntiles = (outputs + BN - 1U) / BN;
    const unsigned m0 = ((Tune & 2U) ? blockIdx.x / ntiles : blockIdx.x % (Rows / BM)) * BM;
    const std::uint32_t n0 = ((Tune & 2U) ? blockIdx.x % ntiles : blockIdx.x / (Rows / BM)) * BN;
    // A: thread loads (row i / 8, 8 halves i % 8) for i = threadIdx.x + 256 m.
    const _Float16* asrc[ALoads];
#pragma unroll
    for (unsigned m = 0; m < ALoads; ++m) {
        const unsigned i = threadIdx.x + 256U * m;
        asrc[m] = a16 + static_cast<std::uint64_t>(m0 + i / 8U) * width + (i % 8U) * 8U;
    }
    // B: thread < BItems dequantizes Q8 block (2 stage + odd) of output n0 + threadIdx.x / 2.
    const bool loads_b = threadIdx.x < BItems;
    const unsigned odd = threadIdx.x % 2U, bo = threadIdx.x / 2U;
    const unsigned char* wrow = nullptr;
    {
        unsigned seg; std::uint32_t row;
        wrow = segment_row(site.segments, min(n0 + bo, outputs - 1U), width, seg, row);
    }
    uint4 areg[ALoads];
    unsigned breg[9];
    const auto load = [&](std::uint32_t stage) {
#pragma unroll
        for (unsigned m = 0; m < ALoads; ++m) areg[m] = *reinterpret_cast<const uint4*>(asrc[m] + stage * 64U);
        if (loads_b) {
            const unsigned char* p = wrow + 68U * stage + (odd ? 32U : 0U);
            const auto lo = reinterpret_cast<const uint4*>(p)[0], hi = reinterpret_cast<const uint4*>(p)[1];
            breg[0] = lo.x; breg[1] = lo.y; breg[2] = lo.z; breg[3] = lo.w;
            breg[4] = hi.x; breg[5] = hi.y; breg[6] = hi.z; breg[7] = hi.w;
            breg[8] = reinterpret_cast<const unsigned*>(p)[8];
        }
    };
    const auto store = [&]() {
#pragma unroll
        for (unsigned m = 0; m < ALoads; ++m) {
            const unsigned i = threadIdx.x + 256U * m;
            *reinterpret_cast<uint4*>(&as[dense_f16_lds<Tune>(i / 8U, (i % 8U) * 8U)]) = areg[m];
        }
        if (loads_b) {
            // Even block: dwords from its scale (codes two bytes in); odd block: from two bytes before its scale.
            const float scale = __half2float(__ushort_as_half(static_cast<unsigned short>(odd ? breg[0] >> 16U : breg[0] & 0xffffU)));
#pragma unroll
            for (unsigned q = 0; q < 4U; ++q) {
                _Float16 h[8];
#pragma unroll
                for (unsigned half = 0; half < 2U; ++half) {
                    const unsigned k = 2U * q + half;
                    const unsigned word = odd ? breg[k + 1U] : __builtin_amdgcn_alignbyte(breg[k + 1U], breg[k], 2U);
#pragma unroll
                    for (unsigned j = 0; j < 4U; ++j)
                        h[4U * half + j] = static_cast<_Float16>(scale * static_cast<float>(static_cast<signed char>((word >> (8U * j)) & 255U)));
                }
                uint4 v;
                __builtin_memcpy(&v, h, 16U);
                *reinterpret_cast<uint4*>(&bs[dense_f16_lds<Tune>(bo, odd * 32U + 8U * q)]) = v;
            }
        }
    };
    dense_f16_v8f acc[TM][TN];
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
        for (unsigned u = 0; u < TN; ++u) acc[t][u] = dense_f16_v8f{};
    load(0U);
    for (std::uint32_t stage = 0; stage < stages; ++stage) {
        store();
        __syncthreads();
        if (stage + 1U < stages) load(stage + 1U);
#pragma unroll
        for (unsigned k = 0; k < 4U; ++k) {
            // Fragment rows are 16-aligned bases + lane % 16, so the XOR key of dense_f16_lds is lane % 8:
            // one pair of chunk offsets per k, the 16-row tiles at constant offsets.
            const unsigned key = (Tune & 1U) ? lane % 8U * 8U : 0U;
            const unsigned lo = (Tune & 1U) ? (16U * k) ^ key : 16U * k, hi = (Tune & 1U) ? (16U * k + 8U) ^ key : 16U * k + 8U;
            dense_f16_v16h b[TN];
#pragma unroll
            for (unsigned u = 0; u < TN; ++u) b[u] = dense_f16_pair(bs + (wc * WN + 16U * u + lane % 16U) * Pitch, lo, hi);
#pragma unroll
            for (unsigned t = 0; t < TM; ++t) {
                const dense_f16_v16h a = dense_f16_pair(as + (wr * WM + 16U * t + lane % 16U) * Pitch, lo, hi);
#pragma unroll
                for (unsigned u = 0; u < TN; ++u) acc[t][u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b[u], acc[t][u]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
        for (unsigned u = 0; u < TN; ++u) {
            const std::uint32_t out = n0 + wc * WN + 16U * u + lane % 16U;
            if (out >= outputs) continue;
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) site.store(out, m0 + wr * WM + 16U * t + 2U * e + lane / 16U, acc[t][u][e]);
        }
}

// Tune bit 2 (value 4..7): the same tile, operands and K order as project_rows_dense_f16_wmma_kernel, pipelined:
//   - two register stages of global loads in flight (stage s + 2 loads while s computes), issued as global (not flat)
//     loads so the LDS-only barrier (lgkmcnt) does not wait for them; __syncthreads drained them every stage;
//   - one LDS image, barriers only around its rewrite; no private arrays (the base spills 80 bytes to scratch).
// Bits 0 and 1 keep their meaning (XOR-swizzled 64-half rows, outputs-first block order).
__device__ __forceinline__ void dense_f16_lds_barrier() { __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
typedef unsigned dense_f16_u32x4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ uint4 dense_f16_global16(const void* p) {
    const dense_f16_u32x4 v = *reinterpret_cast<const __attribute__((address_space(1))) dense_f16_u32x4*>(reinterpret_cast<std::uintptr_t>(p));
    return uint4{v.x, v.y, v.z, v.w};
}
__device__ __forceinline__ unsigned dense_f16_global4(const void* p) {
    return *reinterpret_cast<const __attribute__((address_space(1))) unsigned*>(reinterpret_cast<std::uintptr_t>(p));
}
template <class Site, unsigned BM, unsigned BN, unsigned WM, unsigned WN, unsigned Tune>
__global__ __launch_bounds__(256) void project_rows_dense_f16_wmma2_kernel(Site site, const _Float16* __restrict__ a16) {
    constexpr unsigned Rows = Site::rows, WC = BN / WN, TM = WM / 16U, TN = WN / 16U;
    constexpr unsigned Pitch = (Tune & 1U) ? 64U : kDenseF16Pitch;
    static_assert((BM / WM) * WC == 8U && Rows % BM == 0U && WM % 16U == 0U && WN % 16U == 0U);
    constexpr unsigned ALoads = BM * 8U / 256U, BItems = BN * 2U;
    static_assert(ALoads >= 1U && ALoads <= 4U && BItems <= 256U);
    __shared__ __align__(16) _Float16 as[BM * Pitch];
    __shared__ __align__(16) _Float16 bs[BN * Pitch];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U, wr = wave / WC, wc = wave % WC;
    const std::uint32_t width = site.width, outputs = site.outputs, stages = width / 64U;
    const unsigned ntiles = (outputs + BN - 1U) / BN;
    const unsigned m0 = ((Tune & 2U) ? blockIdx.x / ntiles : blockIdx.x % (Rows / BM)) * BM;
    const std::uint32_t n0 = ((Tune & 2U) ? blockIdx.x % ntiles : blockIdx.x / (Rows / BM)) * BN;
    // A: load m of thread t is row (t + 256 m) / 8 = t / 8 + 32 m, halves 8 (t % 8): one base and a 32-row stride.
    const _Float16* abase = a16 + static_cast<std::uint64_t>(m0 + threadIdx.x / 8U) * width + (threadIdx.x % 8U) * 8U;
    const std::uint64_t astride = 32ULL * width;
    const bool loads_b = threadIdx.x < BItems;
    const unsigned odd = threadIdx.x % 2U, bo = threadIdx.x / 2U;
    const unsigned char* wrow;
    {
        unsigned seg; std::uint32_t row;
        wrow = segment_row(site.segments, min(n0 + bo, outputs - 1U), width, seg, row);
    }
    struct Regs { uint4 a0, a1, a2, a3, lo, hi; unsigned last; };
    const auto load = [&](std::uint32_t stage, Regs& r) {
        r.a0 = dense_f16_global16(abase + stage * 64U);
        if constexpr (ALoads > 1U) r.a1 = dense_f16_global16(abase + astride + stage * 64U);
        if constexpr (ALoads > 2U) r.a2 = dense_f16_global16(abase + 2U * astride + stage * 64U);
        if constexpr (ALoads > 3U) r.a3 = dense_f16_global16(abase + 3U * astride + stage * 64U);
        if (loads_b) {
            const unsigned char* p = wrow + 68U * stage + (odd ? 32U : 0U);
            r.lo = dense_f16_global16(p); r.hi = dense_f16_global16(p + 16U); r.last = dense_f16_global4(p + 32U);
        }
    };
    const auto store = [&](const Regs& r) {
        const unsigned arow = threadIdx.x / 8U, ahalf = (threadIdx.x % 8U) * 8U;
        *reinterpret_cast<uint4*>(&as[dense_f16_lds<Tune>(arow, ahalf)]) = r.a0;
        if constexpr (ALoads > 1U) *reinterpret_cast<uint4*>(&as[dense_f16_lds<Tune>(arow + 32U, ahalf)]) = r.a1;
        if constexpr (ALoads > 2U) *reinterpret_cast<uint4*>(&as[dense_f16_lds<Tune>(arow + 64U, ahalf)]) = r.a2;
        if constexpr (ALoads > 3U) *reinterpret_cast<uint4*>(&as[dense_f16_lds<Tune>(arow + 96U, ahalf)]) = r.a3;
        if (loads_b) {
            const unsigned w[9]{r.lo.x, r.lo.y, r.lo.z, r.lo.w, r.hi.x, r.hi.y, r.hi.z, r.hi.w, r.last};
            const float scale = __half2float(__ushort_as_half(static_cast<unsigned short>(odd ? w[0] >> 16U : w[0] & 0xffffU)));
#pragma unroll
            for (unsigned q = 0; q < 4U; ++q) {
                unsigned packed[4];
#pragma unroll
                for (unsigned half = 0; half < 2U; ++half) {
                    const unsigned k = 2U * q + half;
                    const unsigned word = odd ? w[k + 1U] : __builtin_amdgcn_alignbyte(w[k + 1U], w[k], 2U);
#pragma unroll
                    for (unsigned j = 0; j < 4U; j += 2U) {
                        const _Float16 x = static_cast<_Float16>(scale * static_cast<float>(static_cast<signed char>((word >> (8U * j)) & 255U)));
                        const _Float16 y = static_cast<_Float16>(scale * static_cast<float>(static_cast<signed char>((word >> (8U * j + 8U)) & 255U)));
                        packed[2U * half + j / 2U] = static_cast<unsigned>(__builtin_bit_cast(unsigned short, x)) |
                            static_cast<unsigned>(__builtin_bit_cast(unsigned short, y)) << 16U;
                    }
                }
                *reinterpret_cast<uint4*>(&bs[dense_f16_lds<Tune>(bo, odd * 32U + 8U * q)]) = uint4{packed[0], packed[1], packed[2], packed[3]};
            }
        }
    };
    dense_f16_v8f acc[TM][TN];
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
        for (unsigned u = 0; u < TN; ++u) acc[t][u] = dense_f16_v8f{};
    const auto compute = [&]() {
#pragma unroll
        for (unsigned k = 0; k < 4U; ++k) {
            const unsigned key = (Tune & 1U) ? lane % 8U * 8U : 0U;
            const unsigned lo = (Tune & 1U) ? (16U * k) ^ key : 16U * k, hi = (Tune & 1U) ? (16U * k + 8U) ^ key : 16U * k + 8U;
            dense_f16_v16h b[TN];
#pragma unroll
            for (unsigned u = 0; u < TN; ++u) b[u] = dense_f16_pair(bs + (wc * WN + 16U * u + lane % 16U) * Pitch, lo, hi);
#pragma unroll
            for (unsigned t = 0; t < TM; ++t) {
                const dense_f16_v16h a = dense_f16_pair(as + (wr * WM + 16U * t + lane % 16U) * Pitch, lo, hi);
#pragma unroll
                for (unsigned u = 0; u < TN; ++u) acc[t][u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b[u], acc[t][u]);
            }
        }
    };
    Regs r0, r1;
    load(0U, r0);
    if (stages > 1U) load(1U, r1);
#pragma unroll 1
    for (std::uint32_t stage = 0; stage < stages; stage += 2U) {
        if (stage) dense_f16_lds_barrier();
        store(r0);
        dense_f16_lds_barrier();
        if (stage + 2U < stages) load(stage + 2U, r0);
        compute();
        if (stage + 1U < stages) {
            dense_f16_lds_barrier();
            store(r1);
            dense_f16_lds_barrier();
            if (stage + 3U < stages) load(stage + 3U, r1);
            compute();
        }
    }
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
        for (unsigned u = 0; u < TN; ++u) {
            const std::uint32_t out = n0 + wc * WN + 16U * u + lane % 16U;
            if (out >= outputs) continue;
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) site.store(out, m0 + wr * WM + 16U * t + 2U * e + lane / 16U, acc[t][u][e]);
        }
}

// STRIX_DENSE_F16_TILE = 1 (BM 256, BN 128), 2 (128 x 128), 3 (128 x 64), 4 (64 x 64), 5 (64 x 128): forces
// a block shape where the rows allow it (tuning). Default: 128 x 128 when it gives >= 40 blocks, else the next smaller shape.
template <class Site> void launch_dense_f16(const Site& site, hipStream_t stream, bool any_order) {
    static_assert(Site::rows % 32U == 0U && Site::rows <= kDenseF16BigRows);
    if (!dense_a8_width(site.width))
        throw std::invalid_argument("q8_rows: dense FP16 requires a width that is a positive multiple of 64, at most 8192");
    for (unsigned m = 0; m < site.segments.count; ++m)
        if (reinterpret_cast<std::uintptr_t>(site.segments.weights[m]) % 4U)
            throw std::invalid_argument("q8_rows: dense FP16 requires dword-aligned Q8 weights");
    auto* a16 = Site::rows > kDenseA8WideRows ? dense_f16_big_slot() : dense_f16_slot(a8_family_of<Site>());
    const auto items = Site::rows * (site.width / 8U);
    const dim3 convert_blocks((items + kThreads - 1U) / kThreads);
    if (any_order)
        hipExtLaunchKernelGGL(dense_f16_convert_kernel<Site>, convert_blocks, dim3(kThreads), 0, stream, nullptr, nullptr,
            hipExtAnyOrderLaunch, site, a16, dense_f16_overflow_counter());
    else hipLaunchKernelGGL(dense_f16_convert_kernel<Site>, convert_blocks, dim3(kThreads), 0, stream, site, a16, dense_f16_overflow_counter());
    const auto run = [&]<unsigned BM, unsigned BN, unsigned WM, unsigned WN>() {
        if constexpr (Site::rows % BM == 0U) {
            const unsigned blocks = Site::rows / BM * ((site.outputs + BN - 1U) / BN);
            static const unsigned tune = strix::env_number<unsigned>("STRIX_INTPREFILL_DENSE_F16_TUNE", 4U);
            const auto launch = [&]<unsigned T>() {
                // Bit 2 covers tiles of up to 128 rows (the 256-row tile keeps the base pipeline).
                if constexpr ((T & 4U) != 0U && BM <= 128U)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(project_rows_dense_f16_wmma2_kernel<Site, BM, BN, WM, WN, T>),
                        dim3(blocks), dim3(256U), 0, stream, site, a16);
                else hipLaunchKernelGGL(HIP_KERNEL_NAME(project_rows_dense_f16_wmma_kernel<Site, BM, BN, WM, WN, (T & 3U)>),
                    dim3(blocks), dim3(256U), 0, stream, site, a16);
            };
            [&]<unsigned... T>(std::integer_sequence<unsigned, T...>) {
                (void)((tune == T && (launch.template operator()<T>(), true)) || ... || (launch.template operator()<0U>(), true));
            }(std::integer_sequence<unsigned, 1U, 2U, 3U, 4U, 5U, 6U, 7U>{});
            return true;
        } else return false;
    };
    static const unsigned forced = strix::env_number<unsigned>("STRIX_DENSE_F16_TILE", 0U);
    const auto blocks_of = [&](unsigned bm, unsigned bn) { return Site::rows / bm * ((site.outputs + bn - 1U) / bn); };
    if (forced == 1U && run.template operator()<256U, 128U, 64U, 64U>()) return;
    if (forced == 2U && run.template operator()<128U, 128U, 64U, 32U>()) return;
    if (forced == 3U && run.template operator()<128U, 64U, 32U, 32U>()) return;
    if (forced == 4U && run.template operator()<64U, 64U, 32U, 16U>()) return;
    if (forced == 5U && run.template operator()<64U, 128U, 32U, 32U>()) return;
    // 256-row launches: 128 x 128 measured fastest (GDN/QSA in, GDN out, shared gate/up and down); 256 x 128 leaves
    // the last of ~3 block rounds mostly idle on 40 CUs.
    if constexpr (Site::rows % 128U == 0U) {
        if (blocks_of(128U, 128U) >= 40U && run.template operator()<128U, 128U, 64U, 32U>()) return;
        if (run.template operator()<128U, 64U, 32U, 32U>()) return;
    }
    if constexpr (Site::rows % 64U == 0U) {
        if (blocks_of(64U, 128U) >= 40U && run.template operator()<64U, 128U, 32U, 32U>()) return;
        if (run.template operator()<64U, 64U, 32U, 16U>()) return;
    }
    if constexpr (Site::rows == 32U) { run.template operator()<32U, 128U, 16U, 32U>(); return; }  // 2 x 4 waves
}

// GSQ-RCO typed segments in the fast prefill (fast-prefill-f16wmma-v1 semantics, quality-gated, not exact):
// y[r][n] = sum_k fp16(x[r][k]) * fp16(w[n][k]) with w the llama.cpp dequantized value of the GSQ block
// (dense_group: s0 * v, s0|s1 * v for halves, s0 * v - s1 for min types) rounded to FP16 (RNE), v_wmma_f32_16x16x16_f16.
// Tile BM 32 (waves 2 x 4 of 16 x 32) or 16 (waves 1 x 8 of 16 x 16) x BN 128; thread t < 256 dequantizes the 32-weight group (2 stage + t % 2) of
// output n0 + t / 2 per 64-feature stage.
template <class Site, unsigned BM>
__global__ __launch_bounds__(256) void project_rows_dense_f16_gsq_kernel(Site site, const _Float16* __restrict__ a16) {
    constexpr unsigned Rows = Site::rows, BN = 128U, WM = 16U, WC = 8U / (BM / WM), WN = BN / WC, TN = WN / 16U;
    static_assert(Rows % BM == 0U);
    __shared__ __align__(16) _Float16 as[BM * kDenseF16Pitch];
    __shared__ __align__(16) _Float16 bs[BN * kDenseF16Pitch];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U, wr = wave / WC, wc = wave % WC;
    const std::uint32_t width = site.width, outputs = site.outputs, stages = width / 64U;
    const unsigned m0 = blockIdx.x % (Rows / BM) * BM;
    const std::uint32_t n0 = blockIdx.x / (Rows / BM) * BN;
    const unsigned odd = threadIdx.x % 2U, bo = threadIdx.x / 2U;
    const std::uint32_t out = min(n0 + bo, outputs - 1U);
    unsigned m = 0;
    while (m + 1U < site.segments.count && out >= site.segments.first[m + 1U]) ++m;
    const std::uint32_t type = site.segments.types[m] ? site.segments.types[m] : 8U;
    const unsigned char* wrow = site.segments.weights[m] +
        static_cast<std::uint64_t>(out - site.segments.first[m]) * gsq_a8::dense_row_bytes(type, width);
    dense_f16_v8f acc[TN];
#pragma unroll
    for (unsigned u = 0; u < TN; ++u) acc[u] = dense_f16_v8f{};
    for (std::uint32_t stage = 0; stage < stages; ++stage) {
        if (threadIdx.x < BM * 8U) {   // A: one 16-byte chunk per thread (BM rows x 64 halves)
            const unsigned i = threadIdx.x;
            *reinterpret_cast<uint4*>(&as[(i / 8U) * kDenseF16Pitch + (i % 8U) * 8U]) =
                *reinterpret_cast<const uint4*>(a16 + static_cast<std::uint64_t>(m0 + i / 8U) * width + stage * 64U + (i % 8U) * 8U);
        }
        gsq_dense_dispatch(type, [&]<std::uint32_t T>() {
            const auto x = gsq_a8::dense_group<T>(wrow, 2U * stage + odd);
            constexpr auto kind = gsq_a8::dense_kind<T>();
#pragma unroll
            for (unsigned q = 0; q < 4U; ++q) {
                _Float16 h[8];
#pragma unroll
                for (unsigned j = 0; j < 8U; ++j) {
                    const unsigned e = 8U * q + j;
                    const float v = static_cast<float>(static_cast<std::int8_t>(static_cast<std::uint32_t>(x.w[e / 4U]) >> (8U * (e % 4U))));
                    float w;
                    if constexpr (kind == gsq_a8::DenseKind::min) { const float p = x.s0 * v; w = p - x.s1; }
                    else if constexpr (kind == gsq_a8::DenseKind::halves) w = v * (e < 16U ? x.s0 : x.s1);
                    else w = v * x.s0;
                    h[j] = static_cast<_Float16>(w);
                }
                uint4 word;
                __builtin_memcpy(&word, h, 16U);
                *reinterpret_cast<uint4*>(&bs[bo * kDenseF16Pitch + odd * 32U + 8U * q]) = word;
            }
        });
        __syncthreads();
#pragma unroll
        for (unsigned k = 0; k < 4U; ++k) {
            dense_f16_v16h b[TN];
#pragma unroll
            for (unsigned u = 0; u < TN; ++u) b[u] = dense_f16_pair(bs + (wc * WN + 16U * u + lane % 16U) * kDenseF16Pitch, 16U * k, 16U * k + 8U);
            const dense_f16_v16h a = dense_f16_pair(as + (wr * WM + lane % 16U) * kDenseF16Pitch, 16U * k, 16U * k + 8U);
#pragma unroll
            for (unsigned u = 0; u < TN; ++u) acc[u] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b[u], acc[u]);
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned u = 0; u < TN; ++u) {
        const std::uint32_t o = n0 + wc * WN + 16U * u + lane % 16U;
        if (o >= outputs) continue;
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) site.store(o, m0 + wr * WM + 2U * e + lane / 16U, acc[u][e]);
    }
}
template <class Site> void launch_dense_f16_gsq(const Site& site, hipStream_t stream) {
    static_assert(Site::rows % 16U == 0U && Site::rows <= kDenseA8WideRows);
    constexpr unsigned BM = Site::rows % 32U == 0U ? 32U : 16U;
    auto* a16 = dense_f16_slot(a8_family_of<Site>());
    const auto items = Site::rows * (site.width / 8U);
    hipLaunchKernelGGL(dense_f16_convert_kernel<Site>, dim3((items + kThreads - 1U) / kThreads), dim3(kThreads), 0, stream, site, a16,
        dense_f16_overflow_counter());
    const unsigned blocks = Site::rows / BM * ((site.outputs + 127U) / 128U);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(project_rows_dense_f16_gsq_kernel<Site, BM>), dim3(blocks), dim3(256U), 0, stream, site, a16);
}

// Prompt launch of a dense site: with f16 (fast-prefill-f16wmma-v1 prefill launchers) 32..256-row launches
// run launch_dense_f16; otherwise (and for other row counts) the dense A8 form over 64 rows and
// launch(site, a8) up to 64 rows, as the prefill sites did before.
// g64 (component dense8): 64..256-row launches (a multiple of 64) run launch_dense_g64 instead.
template <class Site> void launch_prefill(const Site& site, bool a8, bool f16, hipStream_t stream, bool any_order = false, bool g64 = false) {
    // GSQ-RCO typed segments: the GSQ FP16 WMMA form for 32..256-row prompt launches; 1..8 rows take the dense A8 GSQ form.
    if (gsq_segments(site.segments)) {
        if constexpr (Site::rows >= 16U && Site::rows % 16U == 0U && Site::rows <= kDenseA8WideRows) {
            launch_dense_f16_gsq(site, stream);
            return;
        } else if constexpr (Site::rows <= 8U) {
            launch(site, true, stream, any_order);
            return;
        } else throw std::invalid_argument("q8_rows: GSQ prompt launches take 1..8 or 16..256 (multiple of 16) rows");
    }
    // Only where dense A8 takes its LDS-staged 4 x 2 form (>= 40 blocks of 8 waves): narrower launches (QSA output of
    // 64 rows, shared gate/up) were slower in G64 than in the 4 x 1 A8 form and keep dense A8 (fixture, gfx1151).
    if constexpr (Site::rows >= 64U && Site::rows % 64U == 0U && Site::rows <= kDenseA8WideRows)
        if (g64 && Site::rows / 64U * ((site.outputs + 255U) / 256U) >= 40U) { launch_dense_g64(site, stream, any_order); return; }
    if constexpr (Site::rows >= 32U && Site::rows % 32U == 0U && Site::rows <= kDenseA8WideRows)
        if (f16) { launch_dense_f16(site, stream, any_order); return; }
    if constexpr (Site::rows > 64U) launch_dense_a8(site, stream, any_order);
    else launch(site, a8, stream, any_order);
}
