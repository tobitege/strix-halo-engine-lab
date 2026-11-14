#include "rows_linear.hpp"
#include "internal.hpp"
#include "chunk.hpp"
#include <algorithm>
#include <cstdlib>
#include <string_view>

namespace strix::gemma4::gpu {
namespace {
constexpr unsigned tile_waves = 8U;
// One workgroup = eight waves over T * 8 consecutive outputs (pairs for Gate)
// and R consecutive rows. Lane L of every wave is the shared-pass lane L: it
// owns chunks L, L+32, ... of K. Per 512-element K block, each wave decodes its
// lane chunk of T matrices once (the shared WeightChunk decoder) and applies
// every decoded value to R rows. The rows' FP32 inputs for that block are
// staged in LDS, G groups of four elements per barrier, as [row][group][lane]
// float4 so that the 32 lanes of a read touch 32 consecutive float4 values.
// Each accumulator (output, row) still starts at +0 and adds RN(w * x) for its
// lane's chunks and elements in ascending order, and its 32 lane values are
// combined by wave_sum, exactly as in pair_linear_kernel; no row affects
// another row's value. S = 2 alternates two input tiles, so each stage needs
// one barrier instead of two; the values read are the same.
// M24: Q4_K/Q6_K decode their integer scale products with the 24-bit multiply (same values).
template <WeightFormat Format, bool M24, class C> __device__ inline float decoded(const C& c, unsigned e) {
    if constexpr (M24 && (Format == WeightFormat::q4_k || Format == WeightFormat::q6_k))
        return c.at24(e);
    else
        return c.at(e);
}
template <WeightFormat Format, bool Gate, unsigned T, unsigned R, unsigned G, unsigned S = 1, bool M24 = false>
__global__ void __launch_bounds__(256) rows_linear_kernel(LinearView w, LinearView up, const float* x, unsigned rows,
                                                         float* out, unsigned* failure, bool hardware_half) {
    static_assert(4U % G == 0U && (R * 32U * G) % 256U == 0U && (S == 1U || S == 2U));
    __shared__ float4 tiles[S * R * G * 32U];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    const unsigned first = (blockIdx.y * tile_waves + wave) * T, row0 = blockIdx.x * R;
    const unsigned chunks = w.inputs / 16U, blocks = (chunks + 31U) / 32U;
    const std::uint8_t* wp[T];
    const std::uint8_t* uptr[T];
#pragma unroll
    for (unsigned t = 0; t < T; ++t) {
        // A tail wave reuses a valid row for its unused loads; it never stores it.
        const auto o = min(first + t, w.outputs - 1U);
        wp[t] = w.data + std::size_t(o) * w.row_bytes;
        uptr[t] = Gate ? up.data + std::size_t(o) * up.row_bytes : wp[t];
    }
    float acc[T][R] = {}, uacc[T][R] = {};
    for (unsigned b = 0; b < blocks; ++b) {
        const unsigned c = b * 32U + lane;
        const bool active = c < chunks && first < w.outputs;
        WeightChunk<Format> a[T], u[T];
        if (active) {
#pragma unroll
            for (unsigned t = 0; t < T; ++t) {
                a[t] = WeightChunk<Format>(wp[t], c * 16U, true, hardware_half);
                if constexpr (Gate)
                    u[t] = WeightChunk<Format>(uptr[t], c * 16U, true, hardware_half);
            }
        }
#pragma unroll
        for (unsigned h = 0; h < 4U / G; ++h) {
            float4* tile = tiles + (S == 1U ? 0U : ((b * (4U / G) + h) % S) * (R * G * 32U));
            if constexpr (S == 1U) __syncthreads();
#pragma unroll
            for (unsigned j = 0; j < R * 32U * G / 256U; ++j) {
                const unsigned i = threadIdx.x + 256U * j;
                const unsigned rr = i / (32U * G), l = (i / G) % 32U, g = i % G, cc = b * 32U + l;
                if (cc < chunks) {
                    const unsigned row = min(row0 + rr, rows - 1U);
                    tile[(rr * G + g) * 32U + l] = *reinterpret_cast<const float4*>(
                        x + std::size_t(row) * w.inputs + cc * 16U + (h * G + g) * 4U);
                }
            }
            __syncthreads();
            if (active) {
#pragma unroll
                for (unsigned g = 0; g < G; ++g) {
                    float v[T][4], uv[T][4];
#pragma unroll
                    for (unsigned t = 0; t < T; ++t)
#pragma unroll
                        for (unsigned k = 0; k < 4; ++k) {
                            v[t][k] = decoded<Format, M24>(a[t], (h * G + g) * 4U + k);
                            if constexpr (Gate)
                                uv[t][k] = decoded<Format, M24>(u[t], (h * G + g) * 4U + k);
                        }
#pragma unroll
                    for (unsigned rr = 0; rr < R; ++rr) {
                        const float4 in4 = tile[(rr * G + g) * 32U + lane];
#pragma unroll
                        for (unsigned k = 0; k < 4; ++k) {
                            const float in = k == 0 ? in4.x : k == 1 ? in4.y : k == 2 ? in4.z : in4.w;
#pragma unroll
                            for (unsigned t = 0; t < T; ++t) {
                                acc[t][rr] = acc[t][rr] + v[t][k] * in;
                                if constexpr (Gate)
                                    uacc[t][rr] = uacc[t][rr] + uv[t][k] * in;
                            }
                        }
                    }
                }
            }
        }
    }
    if constexpr (Gate) {
        // Pair p = t * R + rr is reduced by the whole wave, kept by lane p % 32,
        // and its gelu_tanh(gate) * up evaluated there: the same scalar
        // operations as one lane-0 epilogue per pair, 32 pairs per pass.
        constexpr unsigned pairs = T * R;
#pragma unroll
        for (unsigned base = 0; base < pairs; base += 32U) {
            float ga = 0.0F, ub = 0.0F;
#pragma unroll
            for (unsigned p = base; p < pairs && p < base + 32U; ++p) {
                const float sa = wave_sum(acc[p / R][p % R]), sb = wave_sum(uacc[p / R][p % R]);
                if (lane == p - base) { ga = sa; ub = sb; }
            }
            const unsigned p = base + lane, t = p / R, rr = p % R;
            if (p < pairs && first + t < w.outputs && row0 + rr < rows) {
                const float value = gelu_tanh(ga) * ub;
                out[std::size_t(row0 + rr) * w.outputs + first + t] = value;
                if (failure)
                    finite_flag(value, failure);
            }
        }
    } else {
#pragma unroll
        for (unsigned t = 0; t < T; ++t) {
#pragma unroll
            for (unsigned rr = 0; rr < R; ++rr) {
                const float sa = wave_sum(acc[t][rr]);
                if (lane == 0 && first + t < w.outputs && row0 + rr < rows) {
                    out[std::size_t(row0 + rr) * w.outputs + first + t] = sa;
                    if (failure)
                        finite_flag(sa, failure);
                }
            }
        }
    }
}
// Shared decoded weights: a workgroup owns O consecutive outputs and 8 * RW
// rows (RW per wave). Per 512-element K block, wave w decodes the lane-l chunk
// of outputs w, w + 8, ... once into LDS as FP32 ([output][group][lane]
// float4); every wave then applies them to its own RW rows, whose FP32 inputs
// each lane reads straight from x (its own chunk). Lane l still owns chunks
// l, l + 32, ... of K and adds their 16 elements in ascending order from +0 as
// RN(acc + RN(w * x)); wave_sum combines the 32 lane values. Same bytes as
// rows_linear_kernel; only who decodes a weight and how often changes.
template <WeightFormat Format, unsigned O, unsigned RW>
__global__ void __launch_bounds__(256) shared_rows_kernel(LinearView w, const float* x, unsigned rows, float* out,
                                                          unsigned* failure, bool hardware_half) {
    static_assert(O % tile_waves == 0U);
    __shared__ float4 weights[O * 4U * 32U];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    const unsigned first = blockIdx.y * O, row0 = (blockIdx.x * tile_waves + wave) * RW;
    const unsigned chunks = w.inputs / 16U, blocks = (chunks + 31U) / 32U;
    constexpr unsigned per_wave = O / tile_waves;
    const std::uint8_t* wp[per_wave];
#pragma unroll
    for (unsigned j = 0; j < per_wave; ++j)
        wp[j] = w.data + std::size_t(min(first + wave + j * tile_waves, w.outputs - 1U)) * w.row_bytes;
    const float* xr[RW];
#pragma unroll
    for (unsigned r = 0; r < RW; ++r)
        xr[r] = x + std::size_t(min(row0 + r, rows - 1U)) * w.inputs;
    float acc[O][RW] = {};
    for (unsigned b = 0; b < blocks; ++b) {
        const unsigned c = b * 32U + lane;
        const bool active = c < chunks;
        __syncthreads();
        if (active) {
#pragma unroll
            for (unsigned j = 0; j < per_wave; ++j) {
                const WeightChunk<Format> a(wp[j], c * 16U, true, hardware_half);
                const unsigned o = wave + j * tile_waves;
#pragma unroll
                for (unsigned g = 0; g < 4U; ++g)
                    weights[(o * 4U + g) * 32U + lane] =
                        make_float4(decoded<Format, true>(a, g * 4U), decoded<Format, true>(a, g * 4U + 1U),
                                    decoded<Format, true>(a, g * 4U + 2U), decoded<Format, true>(a, g * 4U + 3U));
            }
        }
        __syncthreads();
        if (active) {
#pragma unroll
            for (unsigned g = 0; g < 4U; ++g) {
                float4 in[RW];
#pragma unroll
                for (unsigned r = 0; r < RW; ++r)
                    in[r] = *reinterpret_cast<const float4*>(xr[r] + c * 16U + g * 4U);
#pragma unroll
                for (unsigned o = 0; o < O; ++o) {
                    const float4 wv = weights[(o * 4U + g) * 32U + lane];
#pragma unroll
                    for (unsigned r = 0; r < RW; ++r) {
                        acc[o][r] = acc[o][r] + wv.x * in[r].x;
                        acc[o][r] = acc[o][r] + wv.y * in[r].y;
                        acc[o][r] = acc[o][r] + wv.z * in[r].z;
                        acc[o][r] = acc[o][r] + wv.w * in[r].w;
                    }
                }
            }
        }
    }
#pragma unroll
    for (unsigned o = 0; o < O; ++o) {
#pragma unroll
        for (unsigned r = 0; r < RW; ++r) {
            const float sa = wave_sum(acc[o][r]);
            if (lane == 0 && first + o < w.outputs && row0 + r < rows) {
                out[std::size_t(row0 + r) * w.outputs + first + o] = sa;
                if (failure)
                    finite_flag(sa, failure);
            }
        }
    }
}
bool hardware_half() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_HARDWARE_HALF");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
// Measured on gfx1151 (rows 512, production shapes): 4x16x2 fastest except
// Q6_K, where 2x32x2 is. ROWS_TILE_V2 (default on; read per call so a lever
// A/B can switch it in one process): calls of at least 256 rows with more
// than 512 outputs stage Q4_K and BF16 inputs in two alternating tiles
// (4x16x2) and use 4x32x1 with two tiles for Q6_K; both won in forward and
// reverse bench passes at 512 rows. Narrow outputs and short calls keep the
// first table. STRIX_G4_ROWS_TILE forces one variant everywhere.
int default_variant(WeightFormat format, unsigned outputs, unsigned rows) {
    static const int forced = [] {
        const auto* p = std::getenv("STRIX_G4_ROWS_TILE");
        const int v = p ? std::atoi(p) : -1;
        return v >= 0 && v < rows_linear_variants ? v : -1;
    }();
    if (forced >= 0)
        return forced;
    // SHARED_Q6 (with DECODE24): the wide Q6_K linear decodes each weight once
    // per workgroup into LDS and applies it to 32 rows (16 outputs x 4 rows per
    // wave); it won in both bench passes there and lost on every Q4_K shape.
    // DECODE24: the same tiles with 24-bit integer scale products in the
    // Q4_K decoder (every Q4_K tile) and the wide Q6_K tile; both won in
    // forward and reverse bench passes. Narrow Q6_K keeps 2x32x2.
    const bool m24 = decode24();
    if (rows >= 256U && outputs > 512U && rows_tile_v2()) {
        if (format == WeightFormat::q6_k)
            return m24 ? (shared_q6() ? 17 : 15) : 8;
        if (format == WeightFormat::q4_k)
            return m24 ? 14 : 9;
        if (format == WeightFormat::bf16)
            return 9;
    }
    if (format == WeightFormat::q6_k)
        return 3;
    return format == WeightFormat::q4_k && m24 ? 12 : 4;
}
template <WeightFormat Format, bool Gate, unsigned T, unsigned R, unsigned G, unsigned S = 1, bool M24 = false>
void launch(LinearView w, LinearView up, const float* x, unsigned rows, float* out, Queue q, unsigned* failure) {
    const dim3 grid((rows + R - 1U) / R, (w.outputs + tile_waves * T - 1U) / (tile_waves * T));
    hipLaunchKernelGGL((rows_linear_kernel<Format, Gate, T, R, G, S, M24>), grid, dim3(256), 0, static_cast<hipStream_t>(q), w,
                       up, x, rows, out, failure, hardware_half());
}
// Variants (outputs per wave, rows, groups per barrier): 4x16x4, 4x32x2, 8x16x4,
// 2x32x2, 4x16x2, 4x16x1, 4x8x4, 8x8x2; two-tile stages 4x32x1, 4x16x2, 2x32x1;
// register-tiled 2x64x1. Gate kernels hold two accumulator sets with half the outputs per wave.
template <WeightFormat Format, unsigned O, unsigned RW>
void launch_shared(LinearView w, const float* x, unsigned rows, float* out, Queue q, unsigned* failure) {
    const dim3 grid((rows + tile_waves * RW - 1U) / (tile_waves * RW), (w.outputs + O - 1U) / O);
    hipLaunchKernelGGL((shared_rows_kernel<Format, O, RW>), grid, dim3(256), 0, static_cast<hipStream_t>(q), w, x, rows,
                       out, failure, hardware_half());
}
template <WeightFormat Format, bool Gate>
void launch_variant(int variant, LinearView w, LinearView up, const float* x, unsigned rows, float* out, Queue q,
                    unsigned* failure) {
    switch (variant) {
    case 1: launch<Format, Gate, Gate ? 2U : 4U, 32U, 2U>(w, up, x, rows, out, q, failure); break;
    case 2: launch<Format, Gate, Gate ? 4U : 8U, 16U, 4U>(w, up, x, rows, out, q, failure); break;
    case 3: launch<Format, Gate, Gate ? 1U : 2U, 32U, 2U>(w, up, x, rows, out, q, failure); break;
    case 4: launch<Format, Gate, Gate ? 2U : 4U, 16U, 2U>(w, up, x, rows, out, q, failure); break;
    case 5: launch<Format, Gate, Gate ? 2U : 4U, 16U, 1U>(w, up, x, rows, out, q, failure); break;
    case 6: launch<Format, Gate, Gate ? 2U : 4U, 8U, 4U>(w, up, x, rows, out, q, failure); break;
    case 7: launch<Format, Gate, Gate ? 4U : 8U, 8U, 2U>(w, up, x, rows, out, q, failure); break;
    case 8: launch<Format, Gate, Gate ? 2U : 4U, 32U, 1U, 2U>(w, up, x, rows, out, q, failure); break;
    case 9: launch<Format, Gate, Gate ? 2U : 4U, 16U, 2U, 2U>(w, up, x, rows, out, q, failure); break;
    case 10: launch<Format, Gate, Gate ? 1U : 2U, 32U, 1U, 2U>(w, up, x, rows, out, q, failure); break;
    case 11: launch<Format, Gate, Gate ? 1U : 2U, 64U, 1U>(w, up, x, rows, out, q, failure); break;
    // 24-bit integer scale products in the decoder: 4x16x2, 2x32x2, 4x16x2 S2, 4x32x1 S2.
    case 12: launch<Format, Gate, Gate ? 2U : 4U, 16U, 2U, 1U, true>(w, up, x, rows, out, q, failure); break;
    case 13: launch<Format, Gate, Gate ? 1U : 2U, 32U, 2U, 1U, true>(w, up, x, rows, out, q, failure); break;
    case 14: launch<Format, Gate, Gate ? 2U : 4U, 16U, 2U, 2U, true>(w, up, x, rows, out, q, failure); break;
    case 15: launch<Format, Gate, Gate ? 2U : 4U, 32U, 1U, 2U, true>(w, up, x, rows, out, q, failure); break;
    // Shared decoded weights (outputs x rows per wave): 8x8, 16x4, 32x2. The fused
    // gate/up tile has no shared form; it keeps 4x16x2 with two tiles and 24-bit decode.
    case 16: case 17: case 18:
        if constexpr (Gate) {
            launch<Format, Gate, 2U, 16U, 2U, 2U, true>(w, up, x, rows, out, q, failure);
        } else if (variant == 16) {
            launch_shared<Format, 8U, 8U>(w, x, rows, out, q, failure);
        } else if (variant == 17) {
            launch_shared<Format, 16U, 4U>(w, x, rows, out, q, failure);
        } else {
            launch_shared<Format, 32U, 2U>(w, x, rows, out, q, failure);
        }
        break;
    default: launch<Format, Gate, Gate ? 2U : 4U, 16U, 4U>(w, up, x, rows, out, q, failure); break;
    }
}
template <bool Gate>
bool many_rows(LinearView w, LinearView up, const float* x, unsigned rows, float* out, Queue q, unsigned* failure,
               int variant) {
    if (rows <= 8U || !w.data || w.inputs % 16U || (Gate && (up.format != w.format || up.inputs != w.inputs)))
        return false;
    if (variant < 0)
        variant = default_variant(w.format, w.outputs, rows);
    switch (w.format) {
#define G4_ROWS_FORMAT(F)                                                                                                  \
    case WeightFormat::F: launch_variant<WeightFormat::F, Gate>(variant, w, up, x, rows, out, q, failure); break
        G4_ROWS_FORMAT(bf16);
        G4_ROWS_FORMAT(f32);
        G4_ROWS_FORMAT(q8_0);
        G4_ROWS_FORMAT(q4_k);
        G4_ROWS_FORMAT(q6_k);
#undef G4_ROWS_FORMAT
    default: return false;
    }
    launch_checked();
    return true;
}
} // namespace
bool shared_q6() {
    const auto* p = std::getenv("STRIX_G4_PREFILL_SHARED_Q6");
    return !p || std::string_view(p) != "0";
}
bool decode24() {
    const auto* p = std::getenv("STRIX_G4_PREFILL_DECODE24");
    return !p || std::string_view(p) != "0";
}
bool rows_tile_v2() {
    const auto* p = std::getenv("STRIX_G4_PREFILL_ROWS_TILE_V2");
    return !p || std::string_view(p) != "0";
}
bool rows_linear_enabled() {
    static const bool enabled = [] {
        const auto* p = std::getenv("STRIX_G4_PREFILL_EXACT_MANYROW");
        return !p || std::string_view(p) != "0";
    }();
    return enabled;
}
void linear_rows(LinearView w, const float* x, unsigned rows, float* out, Queue q, unsigned* failure, bool many,
                 int variant) {
    if (!rows)
        return;
    if (!w.data || !w.inputs || !w.outputs)
        throw std::runtime_error("empty device linear");
    if (many && variant < 0 && staged_rows_preferred(w.format, rows, w.inputs, w.outputs, false) &&
        linear_staged_rows(w, nullptr, x, rows, out, q, failure))
        return;
    if (many && many_rows<false>(w, {}, x, rows, out, q, failure, variant))
        return;
    if (linear_staged_rows(w, nullptr, x, rows, out, q, failure))
        return;
    for (unsigned r = 0; r < rows; r += 8U)
        linear(w, x + std::size_t(r) * w.inputs, std::min(8U, rows - r), out + std::size_t(r) * w.outputs, true, q,
               failure);
}
void gate_up_rows(LinearView gate, LinearView up, const float* x, unsigned rows, float* out, Queue q, unsigned* failure,
                  bool many, int variant, float* scratch, int split, bool fuse_finite) {
    if (!rows)
        return;
    if (!gate.data || !up.data || gate.outputs != up.outputs || gate.inputs != up.inputs)
        throw std::runtime_error("gate/up geometry");
    // Fused by default: one tile reads x once for gate and up and evaluates
    // the GELU product on all 32 lanes (+4-5 % exact prefill at 512/2K over the
    // split schedule). STRIX_G4_ROWS_GATE_SPLIT=1 restores the split schedule.
    static const bool split_default = [] {
        const auto* p = std::getenv("STRIX_G4_ROWS_GATE_SPLIT");
        return p && std::string_view(p) == "1";
    }();
    if (many && scratch && rows > 8U && (split < 0 ? split_default : split != 0)) {
        linear_rows(gate, x, rows, out, q, failure, true, variant);
        linear_rows(up, x, rows, scratch, q, failure, true, variant);
        if (failure && fuse_finite) {
            gelu_product_finite(out, scratch, rows * gate.outputs, failure, q);
            return;
        }
        gelu_product(out, scratch, rows * gate.outputs, q);
        if (failure)
            check_finite(out, std::size_t(rows) * gate.outputs, failure, q);
        return;
    }
    if (many && variant < 0 && staged_rows_preferred(gate.format, rows, gate.inputs, gate.outputs, true) &&
        linear_staged_rows(gate, &up, x, rows, out, q, failure))
        return;
    if (many && many_rows<true>(gate, up, x, rows, out, q, failure, variant))
        return;
    if (linear_staged_rows(gate, &up, x, rows, out, q, failure))
        return;
    for (unsigned r = 0; r < rows; r += 8U)
        gpu::gate_up(gate, up, x + std::size_t(r) * gate.inputs, std::min(8U, rows - r),
                     out + std::size_t(r) * gate.outputs, true, q, failure);
}
} // namespace strix::gemma4::gpu
