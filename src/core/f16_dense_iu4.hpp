#pragma once
// Named policy dense-iu4a16-f16-v1 of the shared FP16 dense core (f16_dense_tall.hpp): IU4 g64 weights (core/iu4.hpp
// tile layout, the strix::gguf::type_strix_iu4_g64 pack: packed tiles then FP16 scales [N][K/64]) x FP16 activations in
// ONE FP16 WMMA chain per output, y[r][o] = sum_k fp16(x[r][k]) * w16[o][k] (FP32 accumulation, k ascending, from +0).
//
// Class: FAST (not the bits of the exact W4A16 small-M core of dense_linear_iu4.hpp, which scales each K64 group sum).
// Arithmetic contract:
//   w16[o][k] = fp16_rne(float(code) * float(scale))   code = signed nibble in [-8, 7], scale = FP16 of (o, k / 64)
//                                                       (the FP32 product is exact: 4 x 11 significant bits)
//   x16[r][k] = fp16_rne(x[r][k])                       (the core's register conversion, lane = row)
// Weights are decoded by the CTA's stagers once per (CTA, stage of 64 k = one scale group) into LDS: an IU4 byte is read
// once per 16 * Waves rows (0.53 bytes per weight, a quarter of an FP16 mirror). No FP16 copy of the weights exists.
// The decode is the packed FP16 form of f16_dense_q8a8.hpp: 0x6400 | (code + 8) is the FP16 integer 1032 + code, the
// packed subtraction of 1032 is exact and one packed FP16 multiply rounds the exact product once (RNE), the single
// rounding of the FP32 path (subnormal, infinite and NaN scales included; host_w16 below is the reference form).
// Any engine with IU4 g64 dense weights binds it by filling View; nothing here names a model.
// Home: shared (backend core, HIP; arch: gfx11 WMMA through f16_dense_tall). Bound: qwen35 HIP (rows > 16).
// Limits (callers check with supported()): in % 128 == 0, n % 16 == 0, count <= n, rows * in * 4 < 2^32.
#include "f16_dense_q8a8.hpp"

namespace strix::core::f16_dense_tall::iu4a16 {
inline constexpr char kPolicyName[] = "dense-iu4a16-f16-v1";

struct View {
    const std::uint8_t* weight;  // packed tiles [in/16][n/16][16][8] bytes
    const std::uint16_t* scale;  // FP16 bits [n][in/64] (the pack stores them right after the tiles)
    const float* x;              // FP32 rows [rows][in]
    float* out;                  // out[r * out_stride + o]
    std::uint32_t in, n, count, rows, out_stride;
    std::uint32_t tile0;         // first column tile of the launch (set by enqueue)
    std::uint32_t accumulate;    // out += sum
};

namespace detail {
using GlobalU2 = const std::uint32_t __attribute__((ext_vector_type(2), address_space(1), aligned(8)));
using Half2 = _Float16 __attribute__((ext_vector_type(2)));
// The two w16 of byte `byte` of `biased` (= word ^ 0x88888888: code + 8 per nibble): low half = low nibble (even k).
__device__ __forceinline__ std::uint32_t w16_pair(std::uint32_t biased, unsigned byte, Half2 scale2) {
    const std::uint32_t t = biased >> (8U * byte);
    const std::uint32_t bits = (t & 0xFU) | ((t & 0xF0U) << 12U) | 0x64006400U;
    const _Float16 bias = static_cast<_Float16>(1032.0F);
    return __builtin_bit_cast(std::uint32_t, (__builtin_bit_cast(Half2, bits) - Half2{bias, bias}) * scale2);
}
}  // namespace detail

// Reference form of the contract's weight value (FP16 bits), for tests.
inline _Float16 host_w16(int code, std::uint16_t scale_bits) {
    return static_cast<_Float16>(static_cast<float>(code) * static_cast<float>(__builtin_bit_cast(_Float16, scale_bits)));
}

template <unsigned Waves> struct Cols {
    static constexpr unsigned kWaves = Waves;
    View v;
    __device__ __forceinline__ ColumnTile tile(unsigned i) const {
        const unsigned o = 16U * (v.tile0 + i);
        return {nullptr, v.out + o, v.out_stride, o < v.count};
    }
    // 64 weights of the stager's column = four 8-byte rows of four K16 tiles (one K64 scale group); columns past count
    // read the last one (never stored). unit % 4 == 0 (the core stages four k units at a time from 0).
    __device__ __forceinline__ void stage(unsigned tile, unsigned col, unsigned unit, U4* nb) const {
        const unsigned o0 = 16U * (v.tile0 + tile) + col, o = o0 < v.count ? o0 : v.count - 1U;
        const std::size_t col_bytes = (std::size_t(o / 16U) * 16U + o % 16U) * 8U, kt_bytes = std::size_t(v.n) * 8U;
        std::uint32_t w[8];
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            const auto t = *(detail::GlobalU2*)(v.weight + std::size_t(unit + j) * kt_bytes + col_bytes);
            w[2U * j] = t[0], w[2U * j + 1U] = t[1];
        }
        const std::uint32_t s = v.scale[std::size_t(o) * (v.in / 64U) + unit / 4U];
        const detail::Half2 scale2 = __builtin_bit_cast(detail::Half2, s | (s << 16U));
#pragma unroll
        for (unsigned q = 0; q < 8U; ++q) {  // nb[q] = k 8q .. 8q + 7 of the group = word q (8 nibbles)
            const std::uint32_t b = w[q] ^ 0x88888888U;
            U4 h;
#pragma unroll
            for (unsigned p = 0; p < 4U; ++p) h[p] = detail::w16_pair(b, p, scale2);
            nb[q] = h;
        }
    }
};

using Store = q8a8::Store;

template <unsigned TC, unsigned Waves> __global__ __launch_bounds__(32 * Waves) void kernel(View v) {
    __shared__ Storage<TC> lds;
    core<TC>(v.x, v.in, v.rows, Cols<Waves>{v}, lds, Store{v.count, v.tile0, v.accumulate});
}

inline bool supported(const View& v) {
    return v.weight && v.scale && v.x && v.out && v.rows && v.count && v.in && v.in % 128U == 0U && v.n % 16U == 0U &&
           v.count <= v.n && v.out_stride >= v.count && std::uint64_t(v.rows) * v.in * 4U < (1ULL << 32U);
}

// Rows per weight read = 16 * waves (the same bits either way): STRIX_F16_IU4_WAVES (8 or 16) when set, else the rule
// measured for the sibling q8a8 policy on gfx1151 (q8a8::waves_for: 256-row CTAs for in >= 8192 or 512-1023 rows).
inline unsigned waves_for(unsigned rows, unsigned in, unsigned waves = 0U) {
    static const unsigned env = static_cast<unsigned>(envcfg::integer("STRIX_F16_IU4_WAVES", 0));
    if (!waves) waves = env;
    return waves == 8U || waves == 16U ? waves : rows >= 512U && (in >= 8192U || rows < 1024U) ? 16U : 8U;
}

template <unsigned Waves> inline void launch(View v, hipStream_t stream) {
    const unsigned tiles = (v.count + 15U) / 16U, gx = (v.rows + 16U * Waves - 1U) / (16U * Waves);
    const Plan pl = plan(tiles, v.rows, true);
    v.tile0 = 0U;
    if (pl.full) hipLaunchKernelGGL((kernel<16, Waves>), dim3(gx, pl.full), dim3(32U * Waves), 0, stream, v);
    v.tile0 = 16U * pl.full;
    if (pl.tc == 1U) hipLaunchKernelGGL((kernel<1, Waves>), dim3(gx, pl.gy), dim3(32U * Waves), 0, stream, v);
    if (pl.tc == 2U) hipLaunchKernelGGL((kernel<2, Waves>), dim3(gx, pl.gy), dim3(32U * Waves), 0, stream, v);
    if (pl.tc == 4U) hipLaunchKernelGGL((kernel<4, Waves>), dim3(gx, pl.gy), dim3(32U * Waves), 0, stream, v);
}

// Caller checks supported(v) and the launch error (hipGetLastError).
inline void enqueue(const View& v, hipStream_t stream, unsigned waves = 0U) {
    if (waves_for(v.rows, v.in, waves) == 16U) launch<16>(v, stream);
    else launch<8>(v, stream);
}
}  // namespace strix::core::f16_dense_tall::iu4a16
