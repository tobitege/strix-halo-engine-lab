#pragma once
// Shared exact FP32 row GEMM: y[r][o] = chain over k ascending from +0.0F of fmaf(x[r][k], w[o][k], acc), one chain per
// (row, output), plain row-major weights (F16 / BF16 widened exactly, or F32). The chain is the contract: any engine
// whose projection is defined this way gets the same bits from every tile and schedule in this file.
//
// Schedule (the only thing this core adds over a two-barrier staged kernel): the global loads of step t + 1 are issued
// right after the barrier of step t and stay in registers, unconverted, while the wave runs the FMAs of step t out of
// LDS. Nothing waits on memory between a barrier and the FMA loop, and with DB the stage needs one barrier instead of
// two: the weight tile of step t is written to buffer t & 1, last read in step t - 2, which every wave finished before
// the barrier of step t - 1. The activation tile is always double-buffered by stage.
//
// Tile: RT rows x C columns per work group of C * 4 threads; a thread owns RT / 16 rows x 4 columns (accumulators in
// registers), LDS holds KS inputs of the row tile and of the column tile, transposed so that a k step is two or three
// 16-byte LDS reads for RT / 4 FMAs. NP == 1: grid.y walks the column blocks of every projection of the group in order.
// NP > 1: one work group runs up to NP projections that share the activation tile (count <= NP).
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::core::fma_rows {

enum class Weight : unsigned { F16 = 0, BF16 = 1, F32 = 2 };
inline constexpr unsigned kGroupMax = 3;

struct Group {
    const void* w[kGroupMax];
    float* y[kGroupMax];
    unsigned o0[kGroupMax], n[kGroupMax], first[kGroupMax + 1];  // first[p] = first column block of p (NP == 1)
    unsigned count;
};

namespace detail {
struct Raw {
    std::uint32_t a, b;
};
template <Weight WT> __device__ __forceinline__ Raw load_pair(const void* w, std::size_t at) {
    Raw r{0U, 0U};
    if constexpr (WT == Weight::F32) {
        __builtin_memcpy(&r, static_cast<const float*>(w) + at, 8);
    } else {
        __builtin_memcpy(&r.a, static_cast<const std::uint16_t*>(w) + at, 4);
    }
    return r;
}
template <Weight WT> __device__ __forceinline__ void widen(const Raw& r, float& a, float& b) {
    if constexpr (WT == Weight::F32) {
        a = __builtin_bit_cast(float, r.a);
        b = __builtin_bit_cast(float, r.b);
    } else if constexpr (WT == Weight::F16) {
        a = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(r.a)));
        b = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<std::uint16_t>(r.a >> 16U)));
    } else {
        a = __builtin_bit_cast(float, (r.a & 0xFFFFU) << 16U);
        b = __builtin_bit_cast(float, r.a & 0xFFFF0000U);
    }
}
}  // namespace detail

template <unsigned RT, unsigned C, Weight WT, unsigned NP, unsigned KS, bool DB>
__global__ __launch_bounds__(C * 4) void pipe_kernel(const float* __restrict__ x, unsigned in, unsigned rows, Group g) {
    constexpr unsigned NT = C * 4, RI = RT / 16, XN = RT * KS / 4, WN = C * KS / 2;
    constexpr unsigned XL = (XN + NT - 1) / NT, WL = (WN + NT - 1) / NT;
    static_assert(RI == 2 || RI == 4 || RI == 8, "fma_rows row tile");
    static_assert(KS % 8 == 0 && NP >= 1 && NP <= kGroupMax && XN % NT == 0 && WN % NT == 0, "fma_rows geometry");
    __shared__ alignas(16) float as[2][KS][RT + 4];
    __shared__ alignas(16) float ws[DB ? 2 : 1][KS][C + 4];
    unsigned first_p = 0;
    if constexpr (NP == 1)
        while (first_p + 1 < g.count && blockIdx.y >= g.first[first_p + 1]) ++first_p;
    const unsigned tid = threadIdx.x, tr = tid / (C / 4), tc = tid % (C / 4);
    const unsigned r0 = blockIdx.x * RT, c0 = (NP == 1 ? blockIdx.y - g.first[first_p] : blockIdx.y) * C;
    const unsigned projections = NP == 1 ? 1U : g.count, stages = in / KS;
    float acc[NP][RI][4];
#pragma unroll
    for (unsigned pp = 0; pp < NP; ++pp)
#pragma unroll
        for (unsigned i = 0; i < RI; ++i)
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) acc[pp][i][j] = 0.0F;

    float4 xraw[XL];
    detail::Raw wraw[WL];
    // Rows / columns past the end read the last valid one (never stored): no branch and no select on the loads.
    auto load_x = [&](unsigned k0) {
#pragma unroll
        for (unsigned i = 0; i < XL; ++i) {
            const unsigned v = tid + i * NT, rr = v / (KS / 4), kk = 4 * (v % (KS / 4));
            const unsigned r = r0 + rr < rows ? r0 + rr : rows - 1;
            __builtin_memcpy(&xraw[i], x + static_cast<std::size_t>(r) * in + k0 + kk, 16);
        }
    };
    auto put_x = [&](unsigned b) {
#pragma unroll
        for (unsigned i = 0; i < XL; ++i) {
            const unsigned v = tid + i * NT, rr = v / (KS / 4), kk = 4 * (v % (KS / 4));
            as[b][kk][rr] = xraw[i].x;
            as[b][kk + 1][rr] = xraw[i].y;
            as[b][kk + 2][rr] = xraw[i].z;
            as[b][kk + 3][rr] = xraw[i].w;
        }
    };
    auto load_w = [&](unsigned p, unsigned k0) {
        const void* w = g.w[p];
        const unsigned o0 = g.o0[p], n = g.n[p];
#pragma unroll
        for (unsigned i = 0; i < WL; ++i) {
            const unsigned v = tid + i * NT, cc = v / (KS / 2), kk = 2 * (v % (KS / 2));
            const unsigned o = c0 + cc < n ? c0 + cc : n - 1;
            wraw[i] = detail::load_pair<WT>(w, static_cast<std::size_t>(o0 + o) * in + k0 + kk);
        }
    };
    auto put_w = [&](unsigned b) {
#pragma unroll
        for (unsigned i = 0; i < WL; ++i) {
            const unsigned v = tid + i * NT, cc = v / (KS / 2), kk = 2 * (v % (KS / 2));
            float a, c;
            detail::widen<WT>(wraw[i], a, c);
            ws[b][kk][cc] = a;
            ws[b][kk + 1][cc] = c;
        }
    };

    load_x(0);
    load_w(first_p, 0);
    unsigned t = 0;
    for (unsigned s = 0; s < stages; ++s) {
        const unsigned xb = s & 1U;
#pragma unroll
        for (unsigned pp = 0; pp < NP; ++pp) {
            if (pp >= projections) break;
            const unsigned wb = DB ? (t & 1U) : 0U;
            ++t;
            if (pp == 0) put_x(xb);
            put_w(wb);
            __syncthreads();
            if (pp + 1 < projections)
                load_w(first_p + pp + 1, s * KS);
            else if (s + 1 < stages)
                load_w(first_p, (s + 1) * KS);
            if (pp == 0 && s + 1 < stages) load_x((s + 1) * KS);
#pragma unroll 4
            for (unsigned kk = 0; kk < KS; ++kk) {
                float a[RI];
                if constexpr (RI == 2) {
                    const float2 a0 = *reinterpret_cast<const float2*>(&as[xb][kk][tr * RI]);
                    a[0] = a0.x;
                    a[1] = a0.y;
                } else {
#pragma unroll
                    for (unsigned h = 0; h < RI / 4; ++h) {
                        const float4 a4 = *reinterpret_cast<const float4*>(&as[xb][kk][tr * RI + 4 * h]);
                        a[4 * h] = a4.x;
                        a[4 * h + 1] = a4.y;
                        a[4 * h + 2] = a4.z;
                        a[4 * h + 3] = a4.w;
                    }
                }
                const float4 b4 = *reinterpret_cast<const float4*>(&ws[wb][kk][tc * 4]);
                const float b[4] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
                for (unsigned i = 0; i < RI; ++i)
#pragma unroll
                    for (unsigned j = 0; j < 4; ++j) acc[pp][i][j] = __builtin_fmaf(a[i], b[j], acc[pp][i][j]);
            }
            if constexpr (!DB) __syncthreads();  // retire the weight tile before the next step overwrites it
        }
    }
#pragma unroll
    for (unsigned pp = 0; pp < NP; ++pp) {
        if (pp >= projections) break;
        const unsigned p = first_p + pp, n = g.n[p];
        float* y = g.y[p];
#pragma unroll
        for (unsigned i = 0; i < RI; ++i) {
            const unsigned r = r0 + tr * RI + i;
            if (r >= rows) continue;
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const unsigned o = c0 + tc * 4 + j;
                if (o < n) y[static_cast<std::size_t>(r) * n + o] = acc[pp][i][j];
            }
        }
    }
}

// Short chains (in <= KX): the activation tile of the whole chain stays in LDS and one work group walks CB column
// blocks in a row, so x is staged once per CB blocks and only the first weight load of a work group is not hidden
// behind FMAs. Same thread layout, chain and group walk as pipe_kernel<.., NP = 1, ..>; weights double-buffered.
template <unsigned RT, unsigned C, Weight WT, unsigned KX, unsigned CB, unsigned KS>
__global__ __launch_bounds__(C * 4) void xres_kernel(const float* __restrict__ x, unsigned in, unsigned rows, Group g) {
    constexpr unsigned NT = C * 4, RI = RT / 16, XN = RT * KX / 4, WN = C * KS / 2, XL = XN / NT, WL = WN / NT;
    static_assert(RI == 2 || RI == 4 || RI == 8, "fma_rows row tile");
    static_assert(KS % 8 == 0 && KX % KS == 0 && XN % NT == 0 && WN % NT == 0, "fma_rows geometry");
    __shared__ alignas(16) float as[KX][RT + 4];
    __shared__ alignas(16) float ws[2][KS][C + 4];
    const unsigned tid = threadIdx.x, tr = tid / (C / 4), tc = tid % (C / 4);
    const unsigned r0 = blockIdx.x * RT, stages = in / KS, total = g.first[g.count], gb0 = blockIdx.y * CB;
    detail::Raw wraw[WL];
    auto load_w = [&](unsigned gb, unsigned k0) {
        unsigned p = 0;
        while (p + 1 < g.count && gb >= g.first[p + 1]) ++p;
        const void* w = g.w[p];
        const unsigned o0 = g.o0[p], n = g.n[p], c0 = (gb - g.first[p]) * C;
#pragma unroll
        for (unsigned i = 0; i < WL; ++i) {
            const unsigned v = tid + i * NT, cc = v / (KS / 2), kk = 2 * (v % (KS / 2));
            const unsigned o = c0 + cc < n ? c0 + cc : n - 1;
            wraw[i] = detail::load_pair<WT>(w, static_cast<std::size_t>(o0 + o) * in + k0 + kk);
        }
    };
    auto put_w = [&](unsigned b) {
#pragma unroll
        for (unsigned i = 0; i < WL; ++i) {
            const unsigned v = tid + i * NT, cc = v / (KS / 2), kk = 2 * (v % (KS / 2));
            float a, c;
            detail::widen<WT>(wraw[i], a, c);
            ws[b][kk][cc] = a;
            ws[b][kk + 1][cc] = c;
        }
    };
    load_w(gb0, 0);
    {
        float4 xraw[XL];
#pragma unroll
        for (unsigned i = 0; i < XL; ++i) {
            const unsigned v = tid + i * NT, rr = v / (KX / 4), kk = 4 * (v % (KX / 4));
            const unsigned r = r0 + rr < rows ? r0 + rr : rows - 1, k = kk < in ? kk : 0;
            __builtin_memcpy(&xraw[i], x + static_cast<std::size_t>(r) * in + k, 16);
        }
#pragma unroll
        for (unsigned i = 0; i < XL; ++i) {
            const unsigned v = tid + i * NT, rr = v / (KX / 4), kk = 4 * (v % (KX / 4));
            as[kk][rr] = xraw[i].x;
            as[kk + 1][rr] = xraw[i].y;
            as[kk + 2][rr] = xraw[i].z;
            as[kk + 3][rr] = xraw[i].w;
        }
    }
    unsigned t = 0;
    for (unsigned cb = 0; cb < CB; ++cb) {
        const unsigned gb = gb0 + cb;
        if (gb >= total) break;
        unsigned p = 0;
        while (p + 1 < g.count && gb >= g.first[p + 1]) ++p;
        const unsigned n = g.n[p], c0 = (gb - g.first[p]) * C;
        float acc[RI][4];
#pragma unroll
        for (unsigned i = 0; i < RI; ++i)
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) acc[i][j] = 0.0F;
        for (unsigned s = 0; s < stages; ++s) {
            const unsigned wb = t & 1U;
            ++t;
            put_w(wb);
            __syncthreads();
            if (s + 1 < stages)
                load_w(gb, (s + 1) * KS);
            else if (cb + 1 < CB && gb + 1 < total)
                load_w(gb + 1, 0);
#pragma unroll 4
            for (unsigned kk = 0; kk < KS; ++kk) {
                float a[RI];
                if constexpr (RI == 2) {
                    const float2 a0 = *reinterpret_cast<const float2*>(&as[s * KS + kk][tr * RI]);
                    a[0] = a0.x;
                    a[1] = a0.y;
                } else {
#pragma unroll
                    for (unsigned h = 0; h < RI / 4; ++h) {
                        const float4 a4 = *reinterpret_cast<const float4*>(&as[s * KS + kk][tr * RI + 4 * h]);
                        a[4 * h] = a4.x;
                        a[4 * h + 1] = a4.y;
                        a[4 * h + 2] = a4.z;
                        a[4 * h + 3] = a4.w;
                    }
                }
                const float4 b4 = *reinterpret_cast<const float4*>(&ws[wb][kk][tc * 4]);
                const float b[4] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
                for (unsigned i = 0; i < RI; ++i)
#pragma unroll
                    for (unsigned j = 0; j < 4; ++j) acc[i][j] = __builtin_fmaf(a[i], b[j], acc[i][j]);
            }
        }
        float* y = g.y[p];
#pragma unroll
        for (unsigned i = 0; i < RI; ++i) {
            const unsigned r = r0 + tr * RI + i;
            if (r >= rows) continue;
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const unsigned o = c0 + tc * 4 + j;
                if (o < n) y[static_cast<std::size_t>(r) * n + o] = acc[i][j];
            }
        }
    }
}

template <unsigned RT, unsigned C, unsigned KX, unsigned CB, unsigned KS>
inline void launch_xres(const float* x, Weight wt, unsigned in, unsigned rows, Group g, hipStream_t s) {
    g.first[0] = 0;
    for (unsigned p = 0; p < g.count; ++p) g.first[p + 1] = g.first[p] + (g.n[p] + C - 1) / C;
    const dim3 grid((rows + RT - 1) / RT, (g.first[g.count] + CB - 1) / CB);
    if (wt == Weight::F16)
        xres_kernel<RT, C, Weight::F16, KX, CB, KS><<<grid, C * 4, 0, s>>>(x, in, rows, g);
    else if (wt == Weight::BF16)
        xres_kernel<RT, C, Weight::BF16, KX, CB, KS><<<grid, C * 4, 0, s>>>(x, in, rows, g);
    else
        xres_kernel<RT, C, Weight::F32, KX, CB, KS><<<grid, C * 4, 0, s>>>(x, in, rows, g);
}

// True when launch<..., KS, ...> can run the call: every chain is a whole number of KS stages.
constexpr bool admits(unsigned in, unsigned rows, unsigned ks) { return rows != 0 && in != 0 && in % ks == 0; }

// Enqueues the group (count 1..kGroupMax, every n > 0) on stream s. NP == 1: any count; NP > 1: count <= NP.
template <unsigned RT, unsigned C, unsigned NP, unsigned KS, bool DB>
inline void launch(const float* x, Weight wt, unsigned in, unsigned rows, Group g, hipStream_t s) {
    unsigned blocks = 0;
    g.first[0] = 0;
    for (unsigned p = 0; p < g.count; ++p) {
        const unsigned nb = (g.n[p] + C - 1) / C;
        g.first[p + 1] = g.first[p] + nb;
        if (NP == 1) blocks += nb;
        else if (nb > blocks) blocks = nb;
    }
    const dim3 grid((rows + RT - 1) / RT, blocks);
    if (wt == Weight::F16)
        pipe_kernel<RT, C, Weight::F16, NP, KS, DB><<<grid, C * 4, 0, s>>>(x, in, rows, g);
    else if (wt == Weight::BF16)
        pipe_kernel<RT, C, Weight::BF16, NP, KS, DB><<<grid, C * 4, 0, s>>>(x, in, rows, g);
    else
        pipe_kernel<RT, C, Weight::F32, NP, KS, DB><<<grid, C * 4, 0, s>>>(x, in, rows, g);
}

// ---- Physical form per shape. Every form gives the same bytes; the choice is measured (kernel fixture, gfx1151,
// interleaved whole-call medians, the two-barrier staged kernel = 1.00). A form is used only where it won:
//   chains of <= 128 inputs, 8192 outputs:
//     4096 / 8192 rows: resident 16 blocks 0.42 / 0.24 (8 blocks 0.44 / 0.26, 4 blocks 0.71 / 0.47, 16-input pipe 0.90 / 0.88,
//       32-input pipe 1.40 / 1.32);
//     256 / 1024 rows: 16-input pipe 0.87 / 0.84 (resident forms 1.17-1.50, 32-input pipe 1.05 / 1.15);
//   long chains (4096 inputs), three projections:
//     4096 / 8192 rows: three-in-one 0.81 / 0.71, 64-row tile 0.87 / 0.71, 128-row tile 0.90 / 0.80;
//     1024 / 2048 rows: 128-row tile 0.65 / 0.85 (64-row 0.85 / 1.21, three-in-one 1.23 / 1.44);
//     256 rows (fewer work groups than compute units): 64-row tile 0.40 (128-row 0.55, three-in-one 1.78);
//   long chains, one projection: 1024 / 2048 rows 128-row tile 0.79 / 0.88 (64-row 1.03 / 1.26); 4096 / 8192 rows 64-row
//     tile 0.91 / 0.72.
enum class Form : unsigned { Auto = 0, Pipe128 = 1, Pipe64 = 2, PipeNp3 = 3, Xres16 = 4, Xres8 = 5, Xres4 = 6, Pipe128K16 = 7 };
inline constexpr unsigned kFormCount = 8;
inline constexpr unsigned kXresIn = 128, kLongRows = 4096, kFillGroups = 80, kComputeUnits = 40;

constexpr bool form_admits(Form f, unsigned in, unsigned rows, unsigned count) {
    if (!admits(in, rows, 32) || count == 0 || count > kGroupMax) return false;
    return f == Form::Pipe128 || f == Form::Pipe64 || f == Form::PipeNp3 || f == Form::Pipe128K16 ||
           (f >= Form::Xres16 && f <= Form::Xres4 && in <= kXresIn);
}

constexpr Form select(unsigned in, unsigned rows, const Group& g) {
    unsigned blocks = 0;
    for (unsigned p = 0; p < g.count; ++p) blocks += (g.n[p] + 63U) / 64U;
    if (in <= kXresIn) {
        if (rows < kLongRows) return Form::Pipe128K16;
        const unsigned tiles = (rows + 63U) / 64U;
        if (tiles * ((blocks + 15U) / 16U) >= kFillGroups) return Form::Xres16;
        if (tiles * ((blocks + 7U) / 8U) >= kFillGroups) return Form::Xres8;
        return Form::Xres4;
    }
    if (rows >= kLongRows) return g.count > 1 ? Form::PipeNp3 : Form::Pipe64;
    return ((rows + 127U) / 128U) * blocks < kComputeUnits ? Form::Pipe64 : Form::Pipe128;
}

// Enqueues the group in the selected form (Auto, or a form the shape does not admit, = select()). Requires
// admits(in, rows, 32). Returns the form that runs.
inline Form launch_auto(const float* x, Weight wt, unsigned in, unsigned rows, const Group& g, hipStream_t s, Form f = Form::Auto) {
    if (f == Form::Auto || !form_admits(f, in, rows, g.count)) f = select(in, rows, g);
    switch (f) {
    case Form::Pipe64: launch<64, 64, 1, 32, true>(x, wt, in, rows, g, s); break;
    case Form::PipeNp3: launch<64, 128, 3, 32, true>(x, wt, in, rows, g, s); break;
    case Form::Xres16: launch_xres<64, 64, kXresIn, 16, 32>(x, wt, in, rows, g, s); break;
    case Form::Xres8: launch_xres<64, 64, kXresIn, 8, 32>(x, wt, in, rows, g, s); break;
    case Form::Xres4: launch_xres<64, 64, kXresIn, 4, 32>(x, wt, in, rows, g, s); break;
    case Form::Pipe128K16: launch<128, 64, 1, 16, true>(x, wt, in, rows, g, s); break;
    default: f = Form::Pipe128; launch<128, 64, 1, 32, true>(x, wt, in, rows, g, s); break;
    }
    return f;
}

}  // namespace strix::core::fma_rows
