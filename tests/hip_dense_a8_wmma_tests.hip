#include "../src/hip_q8_rows_gemv.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "dense A8 requires strict floating-point semantics"
#endif
#pragma clang fp contract(off)
#if defined(__HIP_DEVICE_COMPILE__)
#if !defined(__gfx1151__)
#error "dense A8 prototypes target gfx1151"
#endif
#endif

// Dense A8 profile prototype (Q8_0 weights x A8 activations).
//
// Quantizer (one row r, one 32-feature block b; x = the BF16/FP32 activations as FP32):
//   any x non-finite      -> delta = NaN (0x7fc00000), codes 0
//   amax = max |x|; tiny = 0 < amax < 2^-119
//   delta = amax == 0 || tiny ? 1 : amax / 127          (IEEE division, round to nearest even)
//   code  = tiny ? 0 : clamp(rint(x / delta), -127, 127) (IEEE division, rint ties to even)
// Projection, per (row r, output n), blocks in ascending order:
//   acc = +0
//   acc = fma(float(d_w[n][b]) * delta[r][b], float(sum_k q_w[n][b][k] * code[r][b][k]), acc)
//   y   = isnan(acc) ? 0x7fc00000 : acc
// The int32 block dot is exact in any order (|sum| < 2^20, so its float is exact); the FP32
// product, FMA and the ascending block chain are the contract. Deltas are stored per block,
// transposed: delta[b * stride + r].
//
// (a) dense_a8_dot4_kernel: 1..8 rows, G lanes share one output's block (32/G code bytes
//     each, v_dot4_i32_iu8, integer DPP reduction), every lane of the group runs the chain.
// (b) dense_a8_wmma_kernel: 16x16 tiles, two v_wmma_i32_16x16x16_iu8 per block on a zero
//     accumulator, then the per-element chain step in the epilogue.
//
// STRIX_DENSE_A8_BENCH=1 adds timings: (b) at 64/128/256 rows and the FP32
// project_wide_rows_kernel at 64 rows, (a) at 1..8 rows and the exact q8_rows kernels.
namespace {
using namespace strix;
void require(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
}

constexpr std::uint32_t kCanonicalNan = 0x7fc00000U;

// ---------------------------------------------------------------- device side
__device__ __forceinline__ float canonical(float v) { return v != v ? __uint_as_float(kCanonicalNan) : v; }
__device__ __forceinline__ float half_bits(unsigned bits) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits & 0xffffU)));
}
__device__ __forceinline__ float load_input(const std::uint16_t* p) { return __uint_as_float(static_cast<unsigned>(*p) << 16U); }
__device__ __forceinline__ float load_input(const float* p) { return *p; }
__device__ __forceinline__ float divide(float a, float b) { return a / b; }

// One wave per (row, block); lane = feature.
template <class Input>
__global__ __launch_bounds__(256) void dense_a8_quantize_kernel(const Input* input, std::uint32_t rows, std::uint32_t width,
    std::int8_t* codes, float* deltas, std::uint32_t stride) {
    const unsigned lane = threadIdx.x & 31U, blocks = width / 32U;
    const std::uint32_t item = blockIdx.x * 8U + threadIdx.x / 32U;
    if (item >= rows * blocks) return;
    const std::uint32_t r = item / blocks, b = item % blocks;
    const float x = load_input(input + static_cast<std::uint64_t>(r) * width + 32U * b + lane);
    const bool bad = __ballot(!isfinite(x)) != 0;
    float amax = fabsf(x);
    for (int d = 16; d != 0; d /= 2) amax = fmaxf(amax, __shfl_xor(amax, d, 32));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float delta = bad ? __uint_as_float(kCanonicalNan) : amax == 0.0F || tiny ? 1.0F : divide(amax, 127.0F);
    const float rounded = bad || tiny ? 0.0F : rintf(divide(x, delta));
    const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
    codes[static_cast<std::uint64_t>(r) * width + 32U * b + lane] = static_cast<std::int8_t>(code);
    if (lane == 0U) deltas[static_cast<std::uint64_t>(b) * stride + r] = delta;
}

__device__ __forceinline__ int dot4(int a, int b, int acc) { return __builtin_amdgcn_sudot4(true, a, true, b, acc, false); }
template <unsigned D> __device__ __forceinline__ int lane_xor_int(int v) {
    return __builtin_amdgcn_mov_dpp(v, 0x160 | static_cast<int>(D), 0xf, 0xf, false);
}
template <unsigned G> __device__ __forceinline__ int group_sum(int v) {
    if constexpr (G >= 2U) v += lane_xor_int<1U>(v);
    if constexpr (G >= 4U) v += lane_xor_int<2U>(v);
    if constexpr (G >= 8U) v += lane_xor_int<4U>(v);
    return v;
}
template <unsigned Words> struct Words4 { unsigned v[Words]; };

// (a) Rows rows starting at row0 of the quantized buffer; output n row r -> out[(row0 + r) * outputs + n].
template <unsigned Rows, unsigned G>
__global__ __launch_bounds__(256) void dense_a8_dot4_kernel(const std::int8_t* codes, const float* deltas, std::uint32_t stride,
    const unsigned char* weights, float* out, std::uint32_t row0, std::uint32_t width, std::uint32_t outputs) {
    constexpr unsigned Bytes = 32U / G, Words = Bytes / 4U, PerWave = 32U / G;
    const unsigned lane = threadIdx.x & 31U, j = lane % G;
    const std::uint32_t n = (blockIdx.x * 8U + threadIdx.x / 32U) * PerWave + lane / G, blocks = width / 32U;
    const std::uint32_t row = n < outputs ? n : outputs - 1U;
    const unsigned char* w = weights + static_cast<std::uint64_t>(row) * blocks * 34U;
    const std::int8_t* a = codes + static_cast<std::uint64_t>(row0) * width + j * Bytes;
    float acc[Rows];
#pragma unroll
    for (unsigned r = 0; r < Rows; ++r) acc[r] = 0.0F;
    for (std::uint32_t b = 0; b < blocks; ++b) {
        const unsigned char* block = w + b * 34U;
        const float dw = half_bits(*reinterpret_cast<const std::uint16_t*>(block));
        Words4<Words> q;
        __builtin_memcpy(&q, block + 2U + j * Bytes, Bytes);
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) {
            const auto x = *reinterpret_cast<const Words4<Words>*>(a + static_cast<std::uint64_t>(r) * width + 32U * b);
            int p = 0;
#pragma unroll
            for (unsigned k = 0; k < Words; ++k) p = dot4(static_cast<int>(x.v[k]), static_cast<int>(q.v[k]), p);
            p = group_sum<G>(p);
            const float s = dw * deltas[static_cast<std::uint64_t>(b) * stride + row0 + r];
            acc[r] = __builtin_fmaf(s, static_cast<float>(p), acc[r]);
        }
    }
    if (j == 0U && n < outputs)
#pragma unroll
        for (unsigned r = 0; r < Rows; ++r) out[static_cast<std::uint64_t>(row0 + r) * outputs + n] = canonical(acc[r]);
}

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
struct __attribute__((aligned(4))) Dword4 { unsigned x, y, z, w; };

// (b) A wave owns TM x TN 16x16 tiles (rows x outputs); a block of 4 waves shares the row tiles
// and consecutive blocks share the output tiles. Q8_0 block pairs (68 bytes) are dword aligned:
// the even block's codes are realigned by two bytes, the odd block's are aligned already.
// WMMA layouts (gfx11 wave32): A lane i%16 = row i (16 K bytes), B lane i%16 = output i,
// D element e of lane l = (row 2e + l/16, output l%16); the upper half wave repeats A/B.
template <unsigned TM, unsigned TN>
__global__ __launch_bounds__(128) void dense_a8_wmma_kernel(const std::int8_t* codes, const float* deltas, std::uint32_t stride,
    const unsigned char* weights, float* out, std::uint32_t rows, std::uint32_t width, std::uint32_t outputs) {
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x / 32U, i = lane & 15U, h = lane >> 4U;
    const std::uint32_t row_tiles = (rows + 16U * TM - 1U) / (16U * TM), pairs = width / 64U, blocks = width / 32U;
    const std::uint32_t m0 = blockIdx.x % row_tiles * 16U * TM, n0 = (blockIdx.x / row_tiles * 4U + wave) * 16U * TN;
    if (n0 >= outputs) return;
    const unsigned char* w[TN];
#pragma unroll
    for (unsigned t = 0; t < TN; ++t)
        w[t] = weights + static_cast<std::uint64_t>(min(n0 + 16U * t + i, outputs - 1U)) * blocks * 34U;
    const std::int8_t* a[TM];
#pragma unroll
    for (unsigned t = 0; t < TM; ++t) a[t] = codes + static_cast<std::uint64_t>(min(m0 + 16U * t + i, rows - 1U)) * width;
    // Epilogue rows of this lane: m0 + 16 t + 2 e + h. The delta stride covers whole row tiles
    // (padding rows are read but never stored), so they are immediate offsets of one index.
    const std::uint32_t erow = m0 + h;
    float acc[TM][TN][8];
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
        for (unsigned u = 0; u < TN; ++u)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) acc[t][u][e] = 0.0F;
    const v8i zero{};
    for (std::uint32_t p = 0; p < pairs; ++p) {
        v4i wb[2][TN][2];
        float dw[2][TN];
#pragma unroll
        for (unsigned u = 0; u < TN; ++u) {
            const auto* q4 = reinterpret_cast<const Dword4*>(w[u] + 68U * p);
            const Dword4 c0 = q4[0], c1 = q4[1], c2 = q4[2], c3 = q4[3];
            const unsigned last = reinterpret_cast<const unsigned*>(w[u] + 68U * p)[16];
            const unsigned q[17]{c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w, c2.x, c2.y, c2.z, c2.w, c3.x, c3.y, c3.z, c3.w, last};
            dw[0][u] = half_bits(q[0]);
            dw[1][u] = half_bits(q[8] >> 16U);
#pragma unroll
            for (unsigned k = 0; k < 4U; ++k) {
                wb[0][u][0][k] = static_cast<int>(__builtin_amdgcn_alignbyte(q[k + 1U], q[k], 2U));
                wb[0][u][1][k] = static_cast<int>(__builtin_amdgcn_alignbyte(q[k + 5U], q[k + 4U], 2U));
                wb[1][u][0][k] = static_cast<int>(q[9U + k]);
                wb[1][u][1][k] = static_cast<int>(q[13U + k]);
            }
        }
        v4i ab[2][TM][2];
#pragma unroll
        for (unsigned t = 0; t < TM; ++t) {
            const auto* x = reinterpret_cast<const v4i*>(a[t] + 64U * p);
            ab[0][t][0] = x[0]; ab[0][t][1] = x[1]; ab[1][t][0] = x[2]; ab[1][t][1] = x[3];
        }
#pragma unroll
        for (unsigned s = 0; s < 2U; ++s) {
            const float* d = deltas + ((2U * p + s) * stride + erow);
            float da[TM][8];
#pragma unroll
            for (unsigned t = 0; t < TM; ++t)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) da[t][e] = d[16U * t + 2U * e];
#pragma unroll
            for (unsigned t = 0; t < TM; ++t)
#pragma unroll
                for (unsigned u = 0; u < TN; ++u) {
                    v8i c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ab[s][t][0], true, wb[s][u][0], zero, false);
                    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ab[s][t][1], true, wb[s][u][1], c, false);
#pragma unroll
                    for (unsigned e = 0; e < 8U; ++e) {
                        const float scale = dw[s][u] * da[t][e];
                        acc[t][u][e] = __builtin_fmaf(scale, static_cast<float>(c[e]), acc[t][u][e]);
                    }
                }
        }
    }
#pragma unroll
    for (unsigned t = 0; t < TM; ++t)
#pragma unroll
        for (unsigned u = 0; u < TN; ++u) {
            const std::uint32_t n = n0 + 16U * u + i;
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const std::uint32_t m = m0 + 16U * t + 2U * e + h;
                if (m < rows && n < outputs) out[static_cast<std::uint64_t>(m) * outputs + n] = canonical(acc[t][u][e]);
            }
        }
}

// Current FP32 path (the GDN sites' coefficient/FMA/tree) for the baseline timings.
__device__ __forceinline__ float baseline_coefficient(unsigned scale, std::int8_t code) {
    float result;
    asm("v_fma_mix_f32 %0, %1, %2, neg(0) op_sel_hi:[1,0,0]" : "=v"(result) : "v"(scale), "v"(static_cast<float>(code)));
    return result;
}
template <class In, unsigned Rows> struct BaselineSite {
    using Input = In;
    static constexpr unsigned rows = Rows;
    const void* input[Rows];
    q8_rows::Segments segments;
    std::uint32_t width, outputs;
    float* output;
    __device__ static float coefficient(unsigned scale, std::int8_t code) { return baseline_coefficient(scale, code); }
    __device__ static void fma(float& acc, float c, float x) { acc = __builtin_fmaf(c, x, acc); }
    __device__ static float add(float lower, float upper) {
        float result;
        asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(lower), "v"(upper));
        return result;
    }
    __device__ void store(std::uint32_t out, unsigned r, float value) const { output[static_cast<std::uint64_t>(r) * outputs + out] = value; }
};

// ---------------------------------------------------------------- host side
struct Buffer {
    void* p = nullptr; std::size_t bytes = 0;
    explicit Buffer(std::size_t size) : bytes(size) { check(hipMalloc(&p, size), "allocate"); }
    ~Buffer() { if (hipFree(p) != hipSuccess) std::_Exit(3); }
    Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
    template <class T> T* as() const { return static_cast<T*>(p); }
    template <class T> void upload(const std::vector<T>& v) const {
        require(v.size() * sizeof(T) == bytes, "upload size"); check(hipMemcpy(p, v.data(), bytes, hipMemcpyHostToDevice), "upload");
    }
    template <class T> std::vector<T> download() const {
        std::vector<T> v(bytes / sizeof(T)); check(hipMemcpy(v.data(), p, bytes, hipMemcpyDeviceToHost), "download"); return v;
    }
    // On the fixture stream (non-blocking: it does not order against null-stream memsets).
    void poison(hipStream_t stream) const { check(hipMemsetAsync(p, 0xa5, bytes, stream), "poison"); }
};

float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U, e = (h >> 10U) & 31U, m = h & 1023U;
    if (e == 31U) return std::bit_cast<float>(sign | 0x7f800000U | (m << 13U));
    if (e == 0U) {
        const float v = std::ldexp(static_cast<float>(m), -24);
        return sign ? -v : v;
    }
    return std::bit_cast<float>(sign | ((e + 112U) << 23U) | (m << 13U));
}
std::uint16_t float_to_half(float v) {  // finite, in range; round to nearest even via the hardware type
    return std::bit_cast<std::uint16_t>(static_cast<_Float16>(v));
}
std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

// Reference quantizer and projection (the contract, serially).
struct Quantized { std::vector<std::int8_t> codes; std::vector<float> deltas; };
Quantized quantize_reference(const std::vector<float>& x, std::uint32_t rows, std::uint32_t width, std::uint32_t stride) {
    const std::uint32_t blocks = width / 32U;
    Quantized q{std::vector<std::int8_t>(static_cast<std::size_t>(rows) * width), std::vector<float>(static_cast<std::size_t>(blocks) * stride)};
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t b = 0; b < blocks; ++b) {
            const float* v = &x[static_cast<std::size_t>(r) * width + 32U * b];
            bool bad = false; float amax = 0.0F;
            for (unsigned k = 0; k < 32U; ++k) { bad |= !std::isfinite(v[k]); amax = std::max(amax, std::fabs(v[k])); }
            const bool tiny = amax > 0.0F && amax < 0x1p-119F;
            const float delta = bad ? std::bit_cast<float>(kCanonicalNan) : amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
            for (unsigned k = 0; k < 32U; ++k) {
                const float rounded = bad || tiny ? 0.0F : std::nearbyint(v[k] / delta);
                q.codes[static_cast<std::size_t>(r) * width + 32U * b + k] =
                    static_cast<std::int8_t>(static_cast<int>(std::fmin(127.0F, std::fmax(-127.0F, rounded))));
            }
            q.deltas[static_cast<std::size_t>(b) * stride + r] = delta;
        }
    return q;
}
float reference_output(const Quantized& q, std::uint32_t stride, const std::vector<std::uint8_t>& weights, std::uint32_t width,
    std::uint32_t r, std::uint32_t n) {
    const std::uint32_t blocks = width / 32U;
    float acc = 0.0F;
    for (std::uint32_t b = 0; b < blocks; ++b) {
        const std::uint8_t* block = &weights[(static_cast<std::size_t>(n) * blocks + b) * 34U];
        const float dw = half_to_float(static_cast<std::uint16_t>(block[0] | (block[1] << 8U)));
        int dot = 0;
        for (unsigned k = 0; k < 32U; ++k)
            dot += static_cast<int>(static_cast<std::int8_t>(block[2U + k])) * q.codes[static_cast<std::size_t>(r) * width + 32U * b + k];
        const float s = dw * q.deltas[static_cast<std::size_t>(b) * stride + r];
        acc = std::fma(s, static_cast<float>(dot), acc);
    }
    return std::isnan(acc) ? std::bit_cast<float>(kCanonicalNan) : acc;
}

// Fixture flavors. 0: random (Gaussian activations with a few outlier channels, realistic
// scales). 1: edges (zero / tiny / subnormal / extreme blocks, RNE ties, -128 codes, signed
// zero, subnormal/zero/negative fp16 scales). 2: sparse NaN/Inf activations and Inf/NaN scales.
std::vector<std::uint8_t> make_weights(std::uint32_t width, std::uint32_t outputs, unsigned flavor, std::mt19937& rng) {
    const std::size_t blocks = static_cast<std::size_t>(width / 32U) * outputs;
    std::vector<std::uint8_t> w(blocks * 34U);
    std::normal_distribution<float> code(0.0F, 40.0F);
    std::uniform_real_distribution<float> scale(0.0004F, 0.004F);
    constexpr std::array<std::uint16_t, 8> edge_scales{0x0000, 0x8000, 0x0001, 0x83ff, 0x0400, 0xa000, 0x7bff, 0x1c00};
    for (std::size_t b = 0; b < blocks; ++b) {
        std::uint16_t d = float_to_half(scale(rng) * (b % 7U == 3U ? -1.0F : 1.0F));
        if (flavor == 1U && b % 5U == 0U) d = edge_scales[(b / 5U) % edge_scales.size()];
        if (flavor == 2U && b % 997U == 13U) d = b % 2U ? 0x7c00 : 0x7e01;
        w[b * 34U] = static_cast<std::uint8_t>(d & 0xffU); w[b * 34U + 1U] = static_cast<std::uint8_t>(d >> 8U);
        for (unsigned k = 0; k < 32U; ++k) {
            int c = static_cast<int>(std::lround(code(rng)));
            c = std::clamp(c, -128, 127);
            if (flavor == 1U && b % 3U == 1U) c = k % 2U ? -128 : 127;
            w[b * 34U + 2U + k] = static_cast<std::uint8_t>(static_cast<std::int8_t>(c));
        }
    }
    return w;
}
// Activations as FP32 values exactly representable in the input type (BF16 rows are rounded).
std::vector<float> make_inputs(std::uint32_t rows, std::uint32_t width, bool bf, unsigned flavor, std::mt19937& rng) {
    std::vector<float> x(static_cast<std::size_t>(rows) * width);
    std::normal_distribution<float> g(0.0F, 1.0F);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t i = 0; i < width; ++i) {
            float v = g(rng) * (i % 97U == 5U ? 30.0F : 1.0F);
            const std::uint32_t b = i / 32U, k = i % 32U;
            if (flavor == 1U) {
                switch ((b + r) % 9U) {
                case 0: v = 0.0F; break;                                     // zero block
                case 1: v = k == 7U ? 0x1p-120F : 0.0F; break;               // tiny block
                case 2: v = std::ldexp(static_cast<float>(k) - 16.0F, -133); break;  // subnormal values
                case 3: v = (k % 2U ? -1.0F : 1.0F) * 3.0e38F; break;        // near FP32 max
                case 4: v = (static_cast<float>(k) - 15.5F) / 127.0F * 4.0F; break;  // half-step ties
                case 5: v = k == 0U ? -0.0F : (k == 1U ? 254.0F : 1.0F); break;  // exact x/delta = 0.5
                case 6: v = k % 2U ? -0.0F : 0.0F; break;                    // signed zeros
                default: break;
                }
            }
            if (flavor == 2U && r % 5U == 2U && i == (r * 37U) % width)
                v = std::bit_cast<float>(i % 3U == 0U ? 0x7f800000U : i % 3U == 1U ? 0xff800000U : 0x7fc01234U + r);
            if (bf) v = std::bit_cast<float>(static_cast<std::uint32_t>(bf16(v)) << 16U);
            if (bf && std::isnan(v)) v = std::bit_cast<float>(0x7fc10000U);
            x[static_cast<std::size_t>(r) * width + i] = v;
        }
    return x;
}

struct Shape { const char* name; std::uint32_t width, outputs; bool bf16; };
constexpr std::array<Shape, 5> shapes{{
    {"GDN qkv+z", 2560U, 18432U, true},
    {"GDN/QSA out", 6144U, 2560U, false},
    {"QSA q+k+v", 2560U, 13312U, true},
    {"shared gate+up", 2560U, 1280U, true},
    {"shared down", 640U, 2560U, false},
}};

struct Activations {
    std::uint32_t rows, width, stride;
    Buffer input, codes, deltas;
    bool bf;
    Activations(std::uint32_t r, std::uint32_t w, bool bf16_input)
        : rows(r), width(w), stride((r + 63U) / 64U * 64U), input(static_cast<std::size_t>(r) * w * (bf16_input ? 2U : 4U)),
          codes(static_cast<std::size_t>(r) * w), deltas(static_cast<std::size_t>(w / 32U) * stride * 4U), bf(bf16_input) {}
    void upload(const std::vector<float>& x) const {
        if (bf) {
            std::vector<std::uint16_t> h(x.size());
            for (std::size_t i = 0; i < x.size(); ++i) h[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(x[i]) >> 16U);
            input.upload(h);
        } else input.upload(x);
    }
    void quantize(hipStream_t stream) const {
        const std::uint32_t items = rows * (width / 32U), grid = (items + 7U) / 8U;
        if (bf) hipLaunchKernelGGL(dense_a8_quantize_kernel<std::uint16_t>, dim3(grid), dim3(256), 0, stream,
            input.as<std::uint16_t>(), rows, width, codes.as<std::int8_t>(), deltas.as<float>(), stride);
        else hipLaunchKernelGGL(dense_a8_quantize_kernel<float>, dim3(grid), dim3(256), 0, stream,
            input.as<float>(), rows, width, codes.as<std::int8_t>(), deltas.as<float>(), stride);
        check(hipGetLastError(), "quantize launch");
    }
};

template <unsigned Rows, unsigned G>
void launch_dot4(const Activations& a, const Buffer& w, Buffer& out, std::uint32_t row0, std::uint32_t outputs, hipStream_t stream) {
    constexpr unsigned PerBlock = 8U * (32U / G);
    hipLaunchKernelGGL((dense_a8_dot4_kernel<Rows, G>), dim3((outputs + PerBlock - 1U) / PerBlock), dim3(256), 0, stream,
        a.codes.as<std::int8_t>(), a.deltas.as<float>(), a.stride, w.as<unsigned char>(), out.as<float>(), row0, a.width, outputs);
}
template <unsigned G>
void launch_dot4_rows(unsigned rows, const Activations& a, const Buffer& w, Buffer& out, std::uint32_t row0, std::uint32_t outputs,
    hipStream_t stream) {
    switch (rows) {
    case 1: launch_dot4<1U, G>(a, w, out, row0, outputs, stream); break;
    case 2: launch_dot4<2U, G>(a, w, out, row0, outputs, stream); break;
    case 3: launch_dot4<3U, G>(a, w, out, row0, outputs, stream); break;
    case 4: launch_dot4<4U, G>(a, w, out, row0, outputs, stream); break;
    case 5: launch_dot4<5U, G>(a, w, out, row0, outputs, stream); break;
    case 6: launch_dot4<6U, G>(a, w, out, row0, outputs, stream); break;
    case 7: launch_dot4<7U, G>(a, w, out, row0, outputs, stream); break;
    default: launch_dot4<8U, G>(a, w, out, row0, outputs, stream); break;
    }
    check(hipGetLastError(), "dot4 launch");
}
template <unsigned TM, unsigned TN>
void launch_wmma(const Activations& a, const Buffer& w, Buffer& out, std::uint32_t outputs, hipStream_t stream) {
    const std::uint32_t row_tiles = (a.rows + 16U * TM - 1U) / (16U * TM), groups = (outputs + 16U * TN - 1U) / (16U * TN);
    hipLaunchKernelGGL((dense_a8_wmma_kernel<TM, TN>), dim3(row_tiles * ((groups + 3U) / 4U)), dim3(128), 0, stream,
        a.codes.as<std::int8_t>(), a.deltas.as<float>(), a.stride, w.as<unsigned char>(), out.as<float>(), a.rows, a.width, outputs);
    check(hipGetLastError(), "wmma launch");
}
enum class Tile { t11, t21, t12, t22, t41, t42 };
const char* tile_name(Tile t) {
    switch (t) { case Tile::t11: return "1x1"; case Tile::t21: return "2x1"; case Tile::t12: return "1x2";
    case Tile::t22: return "2x2"; case Tile::t41: return "4x1"; default: return "4x2"; }
}
void launch_wmma(Tile t, const Activations& a, const Buffer& w, Buffer& out, std::uint32_t outputs, hipStream_t stream) {
    switch (t) {
    case Tile::t11: launch_wmma<1U, 1U>(a, w, out, outputs, stream); break;
    case Tile::t21: launch_wmma<2U, 1U>(a, w, out, outputs, stream); break;
    case Tile::t12: launch_wmma<1U, 2U>(a, w, out, outputs, stream); break;
    case Tile::t22: launch_wmma<2U, 2U>(a, w, out, outputs, stream); break;
    case Tile::t41: launch_wmma<4U, 1U>(a, w, out, outputs, stream); break;
    default: launch_wmma<4U, 2U>(a, w, out, outputs, stream); break;
    }
}
constexpr std::array<Tile, 6> tiles{Tile::t11, Tile::t21, Tile::t12, Tile::t22, Tile::t41, Tile::t42};

std::string describe(std::uint32_t bits) {
    char text[64]; std::snprintf(text, sizeof text, "0x%08x (%g)", bits, std::bit_cast<float>(bits)); return text;
}
void compare(const std::vector<float>& expected, const std::vector<float>& actual, std::uint32_t outputs, const std::string& what) {
    require(expected.size() == actual.size(), what + ": size");
    if (!std::memcmp(expected.data(), actual.data(), expected.size() * 4U)) return;
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (std::bit_cast<std::uint32_t>(expected[i]) != std::bit_cast<std::uint32_t>(actual[i]))
            throw std::runtime_error(what + ": row " + std::to_string(i / outputs) + " output " + std::to_string(i % outputs) +
                " expected " + describe(std::bit_cast<std::uint32_t>(expected[i])) + " got " + describe(std::bit_cast<std::uint32_t>(actual[i])));
}

// One shape, rows, flavor: GPU quantizer == reference quantizer; WMMA (every tile) == dot4 (every
// group width, rows 1..8 per launch) bitwise; sampled (or all) outputs == the serial reference.
void exact_case(hipStream_t stream, const Shape& shape, std::uint32_t rows, unsigned flavor, std::uint32_t outputs,
    bool full_reference, std::uint32_t& cases) {
    std::mt19937 rng(1234U + flavor * 77U + rows + shape.width * 3U + outputs);
    const auto weights_host = make_weights(shape.width, outputs, flavor, rng);
    const auto x = make_inputs(rows, shape.width, shape.bf16, flavor, rng);
    Buffer weights(weights_host.size()); weights.upload(weights_host);
    Activations a(rows, shape.width, shape.bf16); a.upload(x);
    check(hipMemsetAsync(a.deltas.p, 0xff, a.deltas.bytes, stream), "poison padding deltas (NaN)");
    a.quantize(stream);
    check(hipStreamSynchronize(stream), "quantize");
    const auto q = quantize_reference(x, rows, shape.width, a.stride);
    const auto codes = a.codes.download<std::int8_t>();
    const auto deltas = a.deltas.download<float>();
    const std::string tag = std::string(shape.name) + " " + std::to_string(shape.width) + "->" + std::to_string(outputs) +
        " rows=" + std::to_string(rows) + " flavor=" + std::to_string(flavor);
    require(codes == q.codes, tag + ": quantizer codes differ from the reference");
    for (std::uint32_t b = 0; b < shape.width / 32U; ++b)
        require(!std::memcmp(&deltas[static_cast<std::size_t>(b) * a.stride], &q.deltas[static_cast<std::size_t>(b) * a.stride], rows * 4U),
            tag + ": quantizer deltas differ from the reference");

    Buffer out(static_cast<std::size_t>(rows) * outputs * 4U);
    // Decode form: 8 rows per launch, group width 8 (the reference result for the WMMA kernels).
    out.poison(stream);
    for (std::uint32_t r = 0; r < rows; r += 8U) launch_dot4_rows<8U>(std::min(8U, rows - r), a, weights, out, r, outputs, stream);
    check(hipStreamSynchronize(stream), "dot4");
    const auto decode = out.download<float>();
    // Serial reference: every output of every row, or a sample.
    const std::uint32_t step_n = full_reference ? 1U : 37U;
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t n = (r * 13U) % step_n; n < outputs; n += step_n) {
            const float e = reference_output(q, a.stride, weights_host, shape.width, r, n);
            const float g = decode[static_cast<std::size_t>(r) * outputs + n];
            if (std::bit_cast<std::uint32_t>(e) != std::bit_cast<std::uint32_t>(g))
                throw std::runtime_error(tag + ": dot4 row " + std::to_string(r) + " output " + std::to_string(n) + " reference " +
                    describe(std::bit_cast<std::uint32_t>(e)) + " dot4 " + describe(std::bit_cast<std::uint32_t>(g)));
        }
    // Other group widths and row counts (1..7 rows per launch at varying offsets).
    for (const unsigned g : {1U, 2U, 4U}) {
        out.poison(stream);
        for (std::uint32_t r = 0, k = 0; r < rows; ++k) {
            const unsigned count = std::min(1U + (k * 3U + g) % 8U, rows - r);
            if (g == 1U) launch_dot4_rows<1U>(count, a, weights, out, r, outputs, stream);
            else if (g == 2U) launch_dot4_rows<2U>(count, a, weights, out, r, outputs, stream);
            else launch_dot4_rows<4U>(count, a, weights, out, r, outputs, stream);
            r += count;
        }
        check(hipStreamSynchronize(stream), "dot4 groups");
        compare(decode, out.download<float>(), outputs, tag + " dot4 G=" + std::to_string(g));
    }
    // Prefill form.
    for (const auto t : tiles) {
        out.poison(stream);
        launch_wmma(t, a, weights, out, outputs, stream);
        check(hipStreamSynchronize(stream), "wmma");
        compare(decode, out.download<float>(), outputs, tag + " wmma " + tile_name(t));
    }
    if (flavor == 0U) {
        std::size_t finite = 0;
        for (const float v : decode) finite += std::isfinite(v) && v != 0.0F;
        require(finite * 10U > decode.size() * 9U, tag + ": degenerate fixture");
    }
    ++cases;
}

// Engine forms (q8_rows::launch_dense_a8: quantizer + dot4 rows <= 8 / WMMA 16, 32, 64) through a
// two-segment site, against the serial reference. Rows of one launch are rows base..base+R-1 of
// the fixture; every R must give each row the reference bits.
template <class In, unsigned Rows> struct EngineSite {
    using Input = In;
    static constexpr unsigned rows = Rows, a8_family = 3U;
    const void* input[Rows];
    q8_rows::Segments segments;
    std::uint32_t width, outputs;
    float* output;
    __device__ static float coefficient(unsigned, std::int8_t) { return 0.0F; }
    __device__ static void fma(float&, float, float) {}
    __device__ static float add(float lower, float upper) { return lower + upper; }
    __device__ void store(std::uint32_t out, unsigned r, float value) const { output[static_cast<std::uint64_t>(r) * outputs + out] = value; }
};
template <class In, unsigned Rows>
void launch_engine(const Activations& a, std::uint32_t base, const Buffer& weights, std::uint32_t outputs, Buffer& out,
    hipStream_t stream) {
    EngineSite<In, Rows> site{};
    const std::uint32_t split = outputs / 3U * 2U;
    site.segments = q8_rows::segments({{weights.p, split},
        {weights.as<unsigned char>() + static_cast<std::uint64_t>(split) * (a.width / 32U) * 34U, outputs - split}});
    for (unsigned r = 0; r < Rows; ++r)
        site.input[r] = a.input.as<unsigned char>() + static_cast<std::uint64_t>(base + r) * a.width * sizeof(In);
    site.width = a.width; site.outputs = outputs; site.output = out.as<float>();
    // Over 64 rows only the dense A8 form exists (the prompt sites call it directly).
    if constexpr (Rows > 64U) q8_rows::launch_dense_a8(site, stream, false);
    else q8_rows::launch(site, true, stream);
    check(hipGetLastError(), "engine dense A8 launch");
}
template <unsigned Rows>
void launch_engine_rows(const Activations& a, std::uint32_t base, const Buffer& weights, std::uint32_t outputs, Buffer& out,
    hipStream_t stream) {
    if (a.bf) launch_engine<std::uint16_t, Rows>(a, base, weights, outputs, out, stream);
    else launch_engine<float, Rows>(a, base, weights, outputs, out, stream);
}
void launch_engine_any(unsigned rows, const Activations& a, std::uint32_t base, const Buffer& weights, std::uint32_t outputs,
    Buffer& out, hipStream_t stream) {
    switch (rows) {
    case 1: launch_engine_rows<1U>(a, base, weights, outputs, out, stream); break;
    case 2: launch_engine_rows<2U>(a, base, weights, outputs, out, stream); break;
    case 3: launch_engine_rows<3U>(a, base, weights, outputs, out, stream); break;
    case 4: launch_engine_rows<4U>(a, base, weights, outputs, out, stream); break;
    case 5: launch_engine_rows<5U>(a, base, weights, outputs, out, stream); break;
    case 6: launch_engine_rows<6U>(a, base, weights, outputs, out, stream); break;
    case 7: launch_engine_rows<7U>(a, base, weights, outputs, out, stream); break;
    case 8: launch_engine_rows<8U>(a, base, weights, outputs, out, stream); break;
    case 16: launch_engine_rows<16U>(a, base, weights, outputs, out, stream); break;
    case 32: launch_engine_rows<32U>(a, base, weights, outputs, out, stream); break;
    case 128: launch_engine_rows<128U>(a, base, weights, outputs, out, stream); break;
    case 256: launch_engine_rows<256U>(a, base, weights, outputs, out, stream); break;
    default: launch_engine_rows<64U>(a, base, weights, outputs, out, stream); break;
    }
}
// Total = 64: rows 1..8 (dot4 forms) and 16/32/64 (WMMA, pitch-64 slots); Total = 256: the prompt
// sites of 128 and 256 rows (WMMA, pitch-256 slots).
template <std::uint32_t Total = 64U>
void engine_case(hipStream_t stream, const Shape& shape, unsigned flavor, std::uint32_t outputs, std::uint32_t& cases) {
    constexpr std::uint32_t total = Total;
    std::mt19937 rng(4321U + flavor * 31U + shape.width + outputs);
    const auto weights_host = make_weights(shape.width, outputs, flavor, rng);
    const auto x = make_inputs(total, shape.width, shape.bf16, flavor, rng);
    Buffer weights(weights_host.size()); weights.upload(weights_host);
    Activations a(total, shape.width, shape.bf16); a.upload(x);
    const auto q = quantize_reference(x, total, shape.width, total);
    const std::string tag = std::string("engine ") + shape.name + " " + std::to_string(shape.width) + "->" + std::to_string(outputs) +
        " flavor=" + std::to_string(flavor);
    // Header quantizer against the reference (codes [row][feature], deltas [block * Total + row]).
    {
        Buffer codes(static_cast<std::size_t>(total) * shape.width), deltas(static_cast<std::size_t>(shape.width / 32U) * total * 4U);
        const auto quantize = [&](auto site) {
            for (unsigned r = 0; r < total; ++r)
                site.input[r] = a.input.as<unsigned char>() + static_cast<std::uint64_t>(r) * shape.width * (shape.bf16 ? 2U : 4U);
            site.width = shape.width;
            const std::uint32_t items = total * (shape.width / 8U);
            hipLaunchKernelGGL(q8_rows::dense_a8_quantize_kernel<decltype(site)>, dim3((items + 255U) / 256U), dim3(256), 0, stream,
                site, codes.as<std::int8_t>(), deltas.as<float>());
            check(hipGetLastError(), "engine quantizer launch");
        };
        if (shape.bf16) quantize(EngineSite<std::uint16_t, total>{}); else quantize(EngineSite<float, total>{});
        check(hipStreamSynchronize(stream), "engine quantizer");
        require(codes.download<std::int8_t>() == q.codes, tag + ": header quantizer codes differ from the reference");
        const auto d = deltas.download<float>();
        require(!std::memcmp(d.data(), q.deltas.data(), d.size() * 4U), tag + ": header quantizer deltas differ from the reference");
    }
    const std::uint32_t step_n = outputs <= 2560U ? 7U : 37U;
    std::vector<float> reference(static_cast<std::size_t>(total) * outputs, 0.0F);
    for (std::uint32_t r = 0; r < total; ++r)
        for (std::uint32_t n = r % step_n; n < outputs; n += step_n)
            reference[static_cast<std::size_t>(r) * outputs + n] = reference_output(q, total, weights_host, shape.width, r, n);
    Buffer out(static_cast<std::size_t>(total) * outputs * 4U);
    const auto row_counts = Total == 64U ? std::vector<unsigned>{1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 16U, 32U, 64U}
                                         : std::vector<unsigned>{128U, 256U};
    for (const unsigned rows : row_counts) {
        const std::uint32_t base = rows <= 8U ? (rows * 5U) % (total - rows + 1U) : 0U;
        out.poison(stream);
        launch_engine_any(rows, a, base, weights, outputs, out, stream);
        check(hipStreamSynchronize(stream), "engine dense A8");
        const auto got = out.download<float>();
        for (std::uint32_t r = 0; r < rows; ++r)
            for (std::uint32_t n = (base + r) % step_n; n < outputs; n += step_n) {
                const float e = reference[static_cast<std::size_t>(base + r) * outputs + n];
                const float g = got[static_cast<std::size_t>(r) * outputs + n];
                if (std::bit_cast<std::uint32_t>(e) != std::bit_cast<std::uint32_t>(g))
                    throw std::runtime_error(tag + " rows=" + std::to_string(rows) + ": row " + std::to_string(base + r) + " output " +
                        std::to_string(n) + " reference " + describe(std::bit_cast<std::uint32_t>(e)) + " engine " +
                        describe(std::bit_cast<std::uint32_t>(g)));
            }
        ++cases;
    }
}

float median_us(const std::function<void()>& enqueue, hipStream_t stream, unsigned iterations = 30U) {
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "event"); check(hipEventCreate(&stop), "event");
    std::vector<float> samples;
    for (unsigned it = 0; it < iterations + 5U; ++it) {
        check(hipEventRecord(start, stream), "record"); enqueue(); check(hipEventRecord(stop, stream), "record");
        check(hipEventSynchronize(stop), "sync");
        float ms = 0; check(hipEventElapsedTime(&ms, start, stop), "elapsed");
        if (it >= 5U) samples.push_back(ms * 1000.0F);
    }
    check(hipEventDestroy(start), "event"); check(hipEventDestroy(stop), "event");
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

template <class In, unsigned Rows>
float baseline_us(const Shape& shape, const Buffer& weights, const std::vector<std::unique_ptr<Buffer>>& rows_in, Buffer& out,
    hipStream_t stream) {
    BaselineSite<In, Rows> site{};
    for (unsigned r = 0; r < Rows; ++r) site.input[r] = rows_in[r]->p;
    site.segments = q8_rows::segments({{weights.p, shape.outputs}});
    site.width = shape.width; site.outputs = shape.outputs; site.output = out.as<float>();
    return median_us([&] { q8_rows::launch(site, false, stream); }, stream);
}
template <unsigned Rows>
float baseline_shape_us(const Shape& shape, const Buffer& weights, hipStream_t stream) {
    std::vector<std::unique_ptr<Buffer>> rows_in;
    std::mt19937 rng(7U);
    const auto x = make_inputs(Rows, shape.width, shape.bf16, 0U, rng);
    for (unsigned r = 0; r < Rows; ++r) {
        rows_in.push_back(std::make_unique<Buffer>(static_cast<std::size_t>(shape.width) * (shape.bf16 ? 2U : 4U)));
        if (shape.bf16) {
            std::vector<std::uint16_t> h(shape.width);
            for (std::uint32_t i = 0; i < shape.width; ++i) h[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(x[r * shape.width + i]) >> 16U);
            rows_in.back()->upload(h);
        } else rows_in.back()->upload(std::vector<float>(x.begin() + r * shape.width, x.begin() + (r + 1U) * shape.width));
    }
    Buffer out(static_cast<std::size_t>(Rows) * shape.outputs * 4U);
    return shape.bf16 ? baseline_us<std::uint16_t, Rows>(shape, weights, rows_in, out, stream)
                      : baseline_us<float, Rows>(shape, weights, rows_in, out, stream);
}

void engine_bench(hipStream_t stream) {
    std::cout.setf(std::ios::fixed); std::cout.precision(1);
    for (const auto& shape : shapes) {
        std::mt19937 rng(99U);
        const auto weights_host = make_weights(shape.width, shape.outputs, 0U, rng);
        Buffer weights(weights_host.size()); weights.upload(weights_host);
        const double mb = static_cast<double>(weights_host.size()) / 1e6;
        Activations a(64U, shape.width, shape.bf16);
        a.upload(make_inputs(64U, shape.width, shape.bf16, 0U, rng));
        Buffer out(static_cast<std::size_t>(64U) * shape.outputs * 4U);
        std::cout << "engine " << shape.name << " " << shape.width << "->" << shape.outputs << " (" << mb
                  << " MB): T  exact  dense-A8 (us, incl. quantizer)\n";
        const auto exact = [&](unsigned rows) {
            switch (rows) {
            case 1: return baseline_shape_us<1U>(shape, weights, stream);
            case 2: return baseline_shape_us<2U>(shape, weights, stream);
            case 4: return baseline_shape_us<4U>(shape, weights, stream);
            case 8: return baseline_shape_us<8U>(shape, weights, stream);
            default: return baseline_shape_us<64U>(shape, weights, stream);
            }
        };
        for (const unsigned rows : {1U, 2U, 4U, 8U, 64U}) {
            const float dense = median_us([&] { launch_engine_any(rows, a, 0U, weights, shape.outputs, out, stream); }, stream);
            std::cout << "    T=" << rows << "  " << exact(rows) << "  " << dense << "  (" << mb * 1e3 / dense << " GB/s)\n";
        }
    }
}

// Exact-profile prompt projections: the engine's FP32 q8_rows forms for 64-row launches (sites whose
// fma is the fused v_fmac_f32 take the VOPD kernels, as GDN/QSA/shared sites do).
template <class In, unsigned Rows> struct FusedSite : BaselineSite<In, Rows> {
    static constexpr bool fused_fma = true;
};
void exact_bench(hipStream_t stream) {
    std::cout.setf(std::ios::fixed); std::cout.precision(1);
    for (const auto& shape : shapes) {
        std::mt19937 rng(99U);
        const auto weights_host = make_weights(shape.width, shape.outputs, 0U, rng);
        Buffer weights(weights_host.size()); weights.upload(weights_host);
        Activations a(64U, shape.width, shape.bf16);
        a.upload(make_inputs(64U, shape.width, shape.bf16, 0U, rng));
        Buffer out(static_cast<std::size_t>(64U) * shape.outputs * 4U);
        const double flop = 2.0 * 64.0 * shape.width * shape.outputs;
        const auto run = [&]<class In>() {
            FusedSite<In, 64U> site{};
            site.segments = q8_rows::segments({{weights.p, shape.outputs}});
            for (unsigned r = 0; r < 64U; ++r)
                site.input[r] = a.input.as<unsigned char>() + static_cast<std::uint64_t>(r) * a.width * sizeof(In);
            site.width = a.width; site.outputs = shape.outputs; site.output = out.as<float>();
            const float us = median_us([&] { q8_rows::launch(site, false, stream); }, stream);
            std::cout << "exact " << shape.name << " " << shape.width << "->" << shape.outputs << " T=64: " << us << " us ("
                      << flop / (us * 1e6) << " TFLOPS)\n";
        };
        if (shape.bf16) run.template operator()<std::uint16_t>(); else run.template operator()<float>();
    }
}

// Prompt sites (256 rows per launch, as prefill runs them): quantizer alone, projection alone
// (from the quantized slot) and both, per engine shape.
void prompt_bench(hipStream_t stream) {
    std::cout.setf(std::ios::fixed); std::cout.precision(1);
    for (const auto& shape : shapes) {
        std::mt19937 rng(99U);
        const auto weights_host = make_weights(shape.width, shape.outputs, 0U, rng);
        Buffer weights(weights_host.size()); weights.upload(weights_host);
        Activations a(256U, shape.width, shape.bf16);
        a.upload(make_inputs(256U, shape.width, shape.bf16, 0U, rng));
        Buffer out(static_cast<std::size_t>(256U) * shape.outputs * 4U);
        const double ops = 2.0 * 256.0 * shape.width * shape.outputs;
        const auto run = [&]<class In>() {
            EngineSite<In, 256U> site{};
            site.segments = q8_rows::segments({{weights.p, shape.outputs}});
            for (unsigned r = 0; r < 256U; ++r)
                site.input[r] = a.input.as<unsigned char>() + static_cast<std::uint64_t>(r) * a.width * sizeof(In);
            site.width = a.width; site.outputs = shape.outputs; site.output = out.as<float>();
            const float both = median_us([&] { q8_rows::launch_dense_a8(site, stream, false); }, stream);
            const auto slot = q8_rows::dense_a8_wide_slot(q8_rows::a8_family_of<decltype(site)>());
            const std::uint32_t items = 256U * (shape.width / 8U);
            const float quant = median_us([&] {
                hipLaunchKernelGGL(q8_rows::dense_a8_quantize_kernel<decltype(site)>, dim3((items + 255U) / 256U), dim3(256), 0, stream,
                    site, slot.codes, slot.deltas);
            }, stream);
            const float project = median_us([&] { q8_rows::launch_dense_a8_projection(site, slot, stream); }, stream);
            std::cout << "prompt " << shape.name << " " << shape.width << "->" << shape.outputs << " T=256: quantize " << quant
                      << " us, projection " << project << " us (" << ops / (project * 1e6) << " TOPS), both " << both << " us\n";
        };
        if (shape.bf16) run.template operator()<std::uint16_t>(); else run.template operator()<float>();
        // 64-row launches (the QSA output site runs 64-row prompt launches).
        const auto run64 = [&]<class In>() {
            EngineSite<In, 64U> site{};
            site.segments = q8_rows::segments({{weights.p, shape.outputs}});
            for (unsigned r = 0; r < 64U; ++r)
                site.input[r] = a.input.as<unsigned char>() + static_cast<std::uint64_t>(r) * a.width * sizeof(In);
            site.width = a.width; site.outputs = shape.outputs; site.output = out.as<float>();
            const auto slot = q8_rows::dense_a8_slot(q8_rows::a8_family_of<decltype(site)>());
            const std::uint32_t items = 64U * (shape.width / 8U);
            hipLaunchKernelGGL(q8_rows::dense_a8_quantize_kernel<decltype(site)>, dim3((items + 255U) / 256U), dim3(256), 0, stream,
                site, slot.codes, slot.deltas);
            const float project = median_us([&] { q8_rows::launch_dense_a8_projection(site, slot, stream); }, stream);
            std::cout << "prompt " << shape.name << " T=64: projection " << project << " us (" << ops / 4.0 / (project * 1e6) << " TOPS)\n";
        };
        if (shape.bf16) run64.template operator()<std::uint16_t>(); else run64.template operator()<float>();
    }
}

// fast-prefill-f16wmma-v1 dense projections (q8_rows::launch_dense_f16) against the exact FP32 q8_rows
// forms on the same inputs (flavor 0), next to dense A8: relative RMS / max error of each against
// exact, and the times of the whole launches (convert + projection for FP16, quantize + projection
// for A8). Rows 256 and 64 (the prompt launch sizes).
template <class In, unsigned Rows>
void schedule_shape(const Shape& shape, hipStream_t stream, unsigned flavor) {
    std::mt19937 rng(99U + Rows);
    const auto weights_host = make_weights(shape.width, shape.outputs, flavor, rng);
    Buffer weights(weights_host.size()); weights.upload(weights_host);
    Activations a(Rows, shape.width, shape.bf16);
    a.upload(make_inputs(Rows, shape.width, shape.bf16, flavor, rng));
    const std::size_t n = static_cast<std::size_t>(Rows) * shape.outputs;
    Buffer f16_out(n * 4U);
    const auto site_for = [&](Buffer& out) {
        FusedSite<In, Rows> site{};
        site.segments = q8_rows::segments({{weights.p, shape.outputs}});
        for (unsigned r = 0; r < Rows; ++r)
            site.input[r] = a.input.as<unsigned char>() + static_cast<std::uint64_t>(r) * a.width * sizeof(In);
        site.width = a.width; site.outputs = shape.outputs; site.output = out.as<float>();
        return site;
    };
    const auto s16 = site_for(f16_out);
    Buffer halves(static_cast<std::size_t>(Rows) * shape.width * 2U);
    using Site = decltype(site_for(f16_out));
    const auto items = Rows * (shape.width / 8U);
    hipLaunchKernelGGL(q8_rows::dense_f16_convert_kernel<Site>, dim3((items + 255U) / 256U), dim3(256), 0,
        stream, s16, halves.as<_Float16>(), q8_rows::dense_f16_overflow_counter());
    std::vector<unsigned char> baseline(n * 4U), tuned(n * 4U);
    const auto fp16 = [&]<unsigned BM,unsigned BN,unsigned WM,unsigned WN,unsigned Tune>() {
        hipFuncAttributes attr{};int blocks=0;
        const auto kernel=[]{
            if constexpr((Tune&4U)!=0U)return q8_rows::project_rows_dense_f16_wmma2_kernel<Site, BM, BN, WM, WN, Tune>;
            else return q8_rows::project_rows_dense_f16_wmma_kernel<Site, BM, BN, WM, WN, Tune>;
        }();
        check(hipFuncGetAttributes(&attr,reinterpret_cast<const void*>(kernel)),"FP16 attributes");
        check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,256,0),"FP16 occupancy");
        std::cout << "FP16 resource tune=" << Tune << " vgpr=" << attr.numRegs << " lds=" << attr.sharedSizeBytes
                  << " local=" << attr.localSizeBytes << " resident_blocks_per_mp=" << blocks << '\n';
        hipLaunchKernelGGL(kernel, dim3(Rows / BM * ((shape.outputs + BN-1U) / BN)), dim3(256), 0, stream, s16, halves.as<_Float16>());
        check(hipStreamSynchronize(stream), "FP16 schedule gate");
        check(hipMemcpy(tuned.data(), f16_out.p, n * 4U, hipMemcpyDeviceToHost), "FP16 schedule bytes");
        if constexpr (Tune == 0U) baseline = tuned;
        else require(tuned == baseline, "dense FP16 schedule changed bytes");
        // STRIX_INTPREFILL_SCHED_BENCH=1: kernel time of this tile and tune (median of 30, warm L2/MALL).
        if (std::getenv("STRIX_INTPREFILL_SCHED_BENCH") && flavor == 0U) {
            const float us = median_us([&] {
                hipLaunchKernelGGL(kernel, dim3(Rows / BM * ((shape.outputs + BN-1U) / BN)), dim3(256), 0, stream, s16, halves.as<_Float16>());
            }, stream);
            std::cout << "FP16 bench " << shape.name << " T=" << Rows << " tile " << BM << "x" << BN << " tune=" << Tune << ": "
                      << us << " us " << 2.0 * Rows * shape.width * shape.outputs / (us * 1e6) << " TFLOPS\n";
        }
    };
    const auto schedule_tiles = [&]<unsigned BM,unsigned BN,unsigned WM,unsigned WN>() {
        if constexpr(Rows%BM==0U) {
            fp16.template operator()<BM,BN,WM,WN,0U>(); fp16.template operator()<BM,BN,WM,WN,1U>();
            fp16.template operator()<BM,BN,WM,WN,2U>(); fp16.template operator()<BM,BN,WM,WN,3U>();
            if constexpr(BM<=128U) {
                fp16.template operator()<BM,BN,WM,WN,4U>(); fp16.template operator()<BM,BN,WM,WN,5U>();
                fp16.template operator()<BM,BN,WM,WN,6U>(); fp16.template operator()<BM,BN,WM,WN,7U>();
            }
        }
    };
    schedule_tiles.template operator()<32U,128U,16U,32U>();
    schedule_tiles.template operator()<256U,128U,64U,64U>();
    schedule_tiles.template operator()<128U,128U,64U,32U>();
    schedule_tiles.template operator()<128U,64U,32U,32U>();
    schedule_tiles.template operator()<64U,64U,32U,16U>();
    schedule_tiles.template operator()<64U,128U,32U,32U>();
    if constexpr (Rows >= 64U) {
    hipLaunchKernelGGL(q8_rows::dense_a8_quantize_kernel<Site>, dim3((items + 255U) / 256U), dim3(256), 0,
        stream, s16, a.codes.as<std::int8_t>(), a.deltas.as<float>());
    const auto iu8 = [&]<unsigned Waves,bool Single>() {
        hipFuncAttributes attr{};int blocks=0;
        const auto kernel=q8_rows::project_rows_dense_a8_wmma_lds2_kernel<Site, Waves, Single>;
        check(hipFuncGetAttributes(&attr,reinterpret_cast<const void*>(kernel)),"IU8 attributes");
        check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,32U*Waves,0),"IU8 occupancy");
        std::cout << "IU8 resource single=" << Single << " vgpr=" << attr.numRegs << " lds=" << attr.sharedSizeBytes
                  << " local=" << attr.localSizeBytes << " resident_blocks_per_mp=" << blocks << '\n';
        hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_rows::project_rows_dense_a8_wmma_lds2_kernel<Site, Waves, Single>),
            dim3(Rows / 64U * ((shape.outputs + 32U*Waves-1U) / (32U*Waves))), dim3(32U*Waves), 0, stream,
            s16, a.codes.as<std::int8_t>(), a.deltas.as<float>());
        check(hipStreamSynchronize(stream), "IU8 LDS gate");
        check(hipMemcpy(tuned.data(), f16_out.p, n * 4U, hipMemcpyDeviceToHost), "IU8 LDS bytes");
        if constexpr (!Single) baseline = tuned;
        else require(tuned == baseline, "dense IU8 single LDS changed bytes");
    };
    iu8.template operator()<4U,false>(); iu8.template operator()<4U,true>();
    iu8.template operator()<8U,false>(); iu8.template operator()<8U,true>();
    }
    std::cout << "schedule bytes all eligible FP16 tiles and IU8 waves=4,8 " << shape.name << " T=" << Rows << " PASS\n";
}

template <class In, unsigned Rows>
void f16_shape(const Shape& shape, hipStream_t stream, bool timing) {
    std::mt19937 rng(99U + Rows);
    const auto weights_host = make_weights(shape.width, shape.outputs, 0U, rng);
    Buffer weights(weights_host.size()); weights.upload(weights_host);
    Activations a(Rows, shape.width, shape.bf16);
    a.upload(make_inputs(Rows, shape.width, shape.bf16, 0U, rng));
    const std::size_t n = static_cast<std::size_t>(Rows) * shape.outputs;
    Buffer exact_out(n * 4U), a8_out(n * 4U), f16_out(n * 4U), g64_out(n * 4U);
    const auto site_for = [&](Buffer& out, bool fused) {
        (void)fused;
        FusedSite<In, Rows> site{};
        site.segments = q8_rows::segments({{weights.p, shape.outputs}});
        for (unsigned r = 0; r < Rows; ++r)
            site.input[r] = a.input.as<unsigned char>() + static_cast<std::uint64_t>(r) * a.width * sizeof(In);
        site.width = a.width; site.outputs = shape.outputs; site.output = out.as<float>();
        return site;
    };
    const auto s8 = site_for(a8_out, true), s16 = site_for(f16_out, true), s64 = site_for(g64_out, true);
    // Exact reference: the FP32 q8_rows forms in 64-row launches (the prompt sites' exact organization).
    for (unsigned g = 0; g < Rows / 64U; ++g) {
        FusedSite<In, 64U> se{};
        se.segments = q8_rows::segments({{weights.p, shape.outputs}});
        for (unsigned r = 0; r < 64U; ++r)
            se.input[r] = a.input.as<unsigned char>() + static_cast<std::uint64_t>(64U * g + r) * a.width * sizeof(In);
        se.width = a.width; se.outputs = shape.outputs; se.output = exact_out.as<float>() + static_cast<std::size_t>(64U * g) * shape.outputs;
        q8_rows::launch(se, false, stream);
    }
    q8_rows::launch_dense_a8(s8, stream, false);
    q8_rows::launch_dense_f16(s16, stream, false);
    q8_rows::dense_g64_invalidate();
    q8_rows::launch_dense_g64(s64, stream, false);
    check(hipStreamSynchronize(stream), "f16 compare");
    std::vector<float> e(n), x8(n), x16(n), x64(n);
    check(hipMemcpy(x64.data(), g64_out.p, n * 4U, hipMemcpyDeviceToHost), "download");
    check(hipMemcpy(e.data(), exact_out.p, n * 4U, hipMemcpyDeviceToHost), "download");
    check(hipMemcpy(x8.data(), a8_out.p, n * 4U, hipMemcpyDeviceToHost), "download");
    check(hipMemcpy(x16.data(), f16_out.p, n * 4U, hipMemcpyDeviceToHost), "download");
    const auto err = [&](const std::vector<float>& got, double& rel, double& mx) {
        double sq = 0, ref = 0; mx = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(got[i]) - e[i];
            sq += d * d; ref += static_cast<double>(e[i]) * e[i]; mx = std::max(mx, std::abs(d));
        }
        rel = std::sqrt(sq / ref); mx /= std::sqrt(ref / static_cast<double>(n));
    };
    double r8, m8, r16, m16, r64, m64;
    err(x8, r8, m8); err(x16, r16, m16); err(x64, r64, m64);
    std::cout << std::defaultfloat;
    std::cout.precision(6);
    std::cout << "f16 " << shape.name << " " << shape.width << "->" << shape.outputs << " T=" << Rows << ": error vs exact A8 rel " << r8
              << " max " << m8 << " | F16 rel " << r16 << " max " << m16 << " | G64 rel " << r64 << " max " << m64;
    require(r16 < 1e-2 && std::isfinite(r16), "dense FP16 outside the error bound");
    require(r64 < 5e-2 && std::isfinite(r64), "dense A8-G64 outside the error bound");
    if (timing) {
        const double flop = 2.0 * Rows * shape.width * shape.outputs;
        // A8 and G64 alternate three times (same process, same warm state); G64 runs on weights already requantized
        // (the engine requantizes once per prefill step). Reported: the medians of the three rounds' medians.
        float q8t[3], q64t[3];
        for (unsigned round = 0; round < 3U; ++round) {
            q8t[round] = median_us([&] { q8_rows::launch_dense_a8(s8, stream, false); }, stream);
            q64t[round] = median_us([&] { q8_rows::launch_dense_g64(s64, stream, false); }, stream);
        }
        std::sort(q8t, q8t + 3); std::sort(q64t, q64t + 3);
        const float t8 = q8t[1], t64 = q64t[1];
        const float t16 = median_us([&] { q8_rows::launch_dense_f16(s16, stream, false); }, stream);
        std::cout << std::fixed;
        std::cout.precision(1);
        std::cout << " | A8 " << t8 << " us, F16 " << t16 << " us (" << flop / (t16 * 1e6) << " TFLOPS), G64 " << t64 << " us";
    }
    std::cout << '\n';
}
void f16_bench(hipStream_t stream, bool timing) {
    for (const auto& shape : shapes) {
        if (shape.bf16) { f16_shape<std::uint16_t, 256U>(shape, stream, timing); f16_shape<std::uint16_t, 64U>(shape, stream, timing); }
        else { f16_shape<float, 256U>(shape, stream, timing); f16_shape<float, 64U>(shape, stream, timing); }
    }
    std::cout << "dense FP16 within the error bound: ok (overflows " << q8_rows::dense_f16_overflow_count() << ")\n";
}

void bench(hipStream_t stream) {
    std::cout.setf(std::ios::fixed); std::cout.precision(1);
    for (const auto& shape : shapes) {
        std::mt19937 rng(99U);
        const auto weights_host = make_weights(shape.width, shape.outputs, 0U, rng);
        Buffer weights(weights_host.size()); weights.upload(weights_host);
        const double mb = static_cast<double>(weights_host.size()) / 1e6;
        std::cout << shape.name << " " << shape.width << "->" << shape.outputs << " (" << mb << " MB Q8_0, warm L2/MALL)\n";
        const float base64 = baseline_shape_us<64U>(shape, weights, stream);
        const double ops64 = 2.0 * 64.0 * shape.width * shape.outputs;
        std::cout << "  FP32 project_wide_rows_kernel T=64: " << base64 << " us  " << ops64 / (base64 * 1e6) << " TOPS\n";
        for (const std::uint32_t rows : {64U, 128U, 256U}) {
            Activations a(rows, shape.width, shape.bf16);
            a.upload(make_inputs(rows, shape.width, shape.bf16, 0U, rng));
            a.quantize(stream);
            Buffer out(static_cast<std::size_t>(rows) * shape.outputs * 4U);
            const double ops = 2.0 * rows * shape.width * shape.outputs;
            const float quant = median_us([&] { a.quantize(stream); }, stream);
            float best = 1e30F; Tile best_tile = Tile::t11;
            std::cout << "  A8 WMMA T=" << rows << " (quantize " << quant << " us):";
            for (const auto t : tiles) {
                const float us = median_us([&] { launch_wmma(t, a, weights, out, shape.outputs, stream); }, stream);
                std::cout << "  " << tile_name(t) << " " << us;
                if (us < best) { best = us; best_tile = t; }
            }
            std::cout << "\n    best " << tile_name(best_tile) << ": " << best << " us  " << ops / (best * 1e6) << " TOPS  "
                      << mb * 1e3 / best << " GB/s weights";
            if (rows == 64U) std::cout << "  speedup vs FP32 x" << base64 / best << " (x" << base64 / (best + quant) << " with quantize)";
            std::cout << '\n';
        }
        // Decode form vs the exact q8_rows kernels.
        Activations a(8U, shape.width, shape.bf16);
        a.upload(make_inputs(8U, shape.width, shape.bf16, 0U, rng));
        a.quantize(stream);
        Buffer out(static_cast<std::size_t>(8U) * shape.outputs * 4U);
        std::cout << "  decode (us)   exact-q8_rows  A8 dot4 G=1 / 2 / 4 / 8\n";
        const auto exact = [&](unsigned rows) {
            switch (rows) {
            case 1: return baseline_shape_us<1U>(shape, weights, stream);
            case 2: return baseline_shape_us<2U>(shape, weights, stream);
            case 4: return baseline_shape_us<4U>(shape, weights, stream);
            default: return baseline_shape_us<8U>(shape, weights, stream);
            }
        };
        for (const unsigned rows : {1U, 2U, 4U, 8U}) {
            std::cout << "    T=" << rows << "  " << exact(rows) << "  ";
            std::cout << median_us([&] { launch_dot4_rows<1U>(rows, a, weights, out, 0U, shape.outputs, stream); }, stream) << " / "
                      << median_us([&] { launch_dot4_rows<2U>(rows, a, weights, out, 0U, shape.outputs, stream); }, stream) << " / "
                      << median_us([&] { launch_dot4_rows<4U>(rows, a, weights, out, 0U, shape.outputs, stream); }, stream) << " / "
                      << median_us([&] { launch_dot4_rows<8U>(rows, a, weights, out, 0U, shape.outputs, stream); }, stream) << "  ("
                      << mb * 1e3 / median_us([&] { launch_dot4_rows<4U>(rows, a, weights, out, 0U, shape.outputs, stream); }, stream)
                      << " GB/s at G=4)\n";
        }
    }
}
} // namespace

// STRIX_DENSE_F16_ROWS_BENCH=1: the production dense FP16 kernel (128 x 128, tune 4) over T = 256..2048 rows per launch
// on a strided site (rows at base + r * stride), weights cold (a 96 MB scratch write evicts L2/MALL before each launch):
// kernel time per 256 rows and TFLOP/s, so the cost of re-streaming the weights every 256 rows is measured, not assumed.
template <class In, unsigned Rows> struct StridedRowsSite {
    using Input = In;
    static constexpr unsigned rows = Rows;
    const unsigned char* input0;
    std::uint64_t input_stride;
    q8_rows::Segments segments;
    std::uint32_t width, outputs;
    float* output;
    __device__ const In* row_input(unsigned r) const { return reinterpret_cast<const In*>(input0 + r * input_stride); }
    __device__ static float coefficient(unsigned scale, std::int8_t code) { return baseline_coefficient(scale, code); }
    __device__ static void fma(float& acc, float c, float x) { acc = __builtin_fmaf(c, x, acc); }
    __device__ static float add(float lower, float upper) { return lower + upper; }
    __device__ void store(std::uint32_t out, unsigned r, float value) const { output[static_cast<std::uint64_t>(r) * outputs + out] = value; }
};
template <class In, unsigned Rows>
void rows_bench_shape(const Shape& shape, hipStream_t stream, Buffer& flush) {
    std::mt19937 rng(7U);
    const auto weights_host = make_weights(shape.width, shape.outputs, 0U, rng);
    Buffer weights(weights_host.size()); weights.upload(weights_host);
    Activations a(Rows, shape.width, shape.bf16);
    a.upload(make_inputs(Rows, shape.width, shape.bf16, 0U, rng));
    Buffer out(static_cast<std::size_t>(Rows) * shape.outputs * 4U), halves(static_cast<std::size_t>(Rows) * shape.width * 2U);
    using Site = StridedRowsSite<In, Rows>;
    Site site{};
    site.input0 = a.input.as<unsigned char>(); site.input_stride = static_cast<std::uint64_t>(shape.width) * sizeof(In);
    site.segments = q8_rows::segments({{weights.p, shape.outputs}});
    site.width = shape.width; site.outputs = shape.outputs; site.output = out.as<float>();
    const auto items = Rows * (shape.width / 8U);
    hipLaunchKernelGGL(q8_rows::dense_f16_convert_kernel<Site>, dim3((items + 255U) / 256U), dim3(256), 0, stream, site,
        halves.as<_Float16>(), q8_rows::dense_f16_overflow_counter());
    const auto tile = [&]<unsigned BM, unsigned BN, unsigned WM, unsigned WN, unsigned Tune>() {
    const auto kernel = [] {
        if constexpr ((Tune & 4U) != 0U) return q8_rows::project_rows_dense_f16_wmma2_kernel<Site, BM, BN, WM, WN, Tune>;
        else return q8_rows::project_rows_dense_f16_wmma_kernel<Site, BM, BN, WM, WN, Tune>;
    }();
    const dim3 grid(Rows / BM * ((shape.outputs + BN - 1U) / BN));
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "event"); check(hipEventCreate(&stop), "event");
    std::vector<float> cold, warm;
    for (unsigned it = 0; it < 13U; ++it) {
        check(hipMemsetAsync(flush.p, static_cast<int>(it), flush.bytes, stream), "flush");
        check(hipEventRecord(start, stream), "record");
        hipLaunchKernelGGL(kernel, grid, dim3(256), 0, stream, site, halves.as<_Float16>());
        check(hipEventRecord(stop, stream), "record");
        check(hipEventSynchronize(stop), "sync");
        float ms = 0; check(hipEventElapsedTime(&ms, start, stop), "elapsed");
        if (it >= 3U) cold.push_back(ms * 1000.0F);
    }
    const float hot = median_us([&] { hipLaunchKernelGGL(kernel, grid, dim3(256), 0, stream, site, halves.as<_Float16>()); }, stream, 10U);
    check(hipEventDestroy(start), "event"); check(hipEventDestroy(stop), "event");
    std::sort(cold.begin(), cold.end());
    const float us = cold[cold.size() / 2U];
    std::cout << "rows bench " << shape.name << " " << shape.width << "->" << shape.outputs << " T=" << Rows << " tile " << BM << "x"
              << BN << " tune " << Tune << " cold " << us << " us (" << us * 256.0F / Rows << " us/256 rows, "
              << 2.0 * Rows * shape.width * shape.outputs / (us * 1e6) << " TFLOPS) warm " << hot << " us (" << hot * 256.0F / Rows
              << " us/256 rows)\n";
    };
    tile.template operator()<128U, 128U, 64U, 32U, 4U>();
    tile.template operator()<128U, 64U, 32U, 32U, 4U>();
    tile.template operator()<64U, 128U, 32U, 32U, 4U>();
    tile.template operator()<256U, 128U, 64U, 64U, 1U>();
}
void rows_bench(hipStream_t stream) {
    std::cout.setf(std::ios::fixed); std::cout.precision(1);
    Buffer flush(96U << 20U);
    for (const auto& shape : shapes) {
        const auto run = [&]<class In>() {
            rows_bench_shape<In, 256U>(shape, stream, flush); rows_bench_shape<In, 512U>(shape, stream, flush);
            rows_bench_shape<In, 1024U>(shape, stream, flush); rows_bench_shape<In, 2048U>(shape, stream, flush);
        };
        if (shape.bf16) run.template operator()<std::uint16_t>(); else run.template operator()<float>();
    }
}

// Quantized producer path used by shared experts / fused projection producers.
// Allocations contain only real activation rows; output tail canaries detect stores.
template <unsigned Rows>
void small_quantized_case(hipStream_t stream, unsigned width, unsigned outputs, unsigned flavor) {
    std::mt19937 rng(19U + Rows + width + outputs + flavor);
    const auto wh = make_weights(width, outputs, flavor, rng);
    const auto x = make_inputs(Rows, width, false, flavor, rng);
    const auto q = quantize_reference(x, Rows, width, q8_rows::kDenseA8Rows);
    Buffer weights(wh.size()), codes(q.codes.size()), deltas(q.deltas.size() * 4U);
    weights.upload(wh); codes.upload(q.codes); deltas.upload(q.deltas);
    Buffer out((Rows * outputs + 16U) * 4U);
    EngineSite<float, Rows> site{};
    const unsigned split = outputs / 3U * 2U;
    site.segments = q8_rows::segments({{weights.p, split},
        {weights.as<unsigned char>() + static_cast<std::uint64_t>(split) * (width / 32U) * 34U, outputs - split}});
    site.width = width; site.outputs = outputs; site.output = out.as<float>();
    const q8_rows::DenseA8Slot slot{codes.as<std::int8_t>(), deltas.as<float>()};
    const auto verify = [&](const char* name) {
        check(hipGetLastError(), name);
        check(hipStreamSynchronize(stream), name);
        const auto got = out.download<float>();
        for (unsigned r = 0; r < Rows; ++r)
            for (unsigned n = 0; n < outputs; ++n)
                require(std::bit_cast<unsigned>(got[r * outputs + n]) ==
                    std::bit_cast<unsigned>(reference_output(q, q8_rows::kDenseA8Rows, wh, width, r, n)),
                    std::string(name) + ": serial reference mismatch");
        for (unsigned n = Rows * outputs; n < got.size(); ++n)
            require(std::bit_cast<unsigned>(got[n]) == 0xa5a5a5a5U, std::string(name) + ": output tail overwritten");
    };
    out.poison(stream);
    constexpr unsigned Outs = q8_rows::dense_a8_outs_of<EngineSite<float, Rows>>();
    hipLaunchKernelGGL(HIP_KERNEL_NAME(q8_rows::project_rows_dense_a8_kernel<EngineSite<float, Rows>, Outs, false, false>),
        dim3((outputs + Outs * q8_rows::kWaves - 1U) / (Outs * q8_rows::kWaves)), dim3(q8_rows::kThreads),
        0, stream, site, slot.codes, slot.deltas);
    verify("small quantized dot4 control");
    out.poison(stream);
    q8_rows::launch_dense_a8_quantized(site, slot, stream);
    verify("small quantized WMMA dispatch");
}

int main() try {
    hipStream_t stream = nullptr;
    check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create stream");
    if (q8_rows::dense_a8_small_enabled()) {
        for (unsigned width : {64U, 8192U}) for (unsigned outputs : {17U, 65U}) for (unsigned flavor = 0; flavor < 3U; ++flavor) {
            small_quantized_case<2U>(stream, width, outputs, flavor);
            small_quantized_case<3U>(stream, width, outputs, flavor);
            small_quantized_case<4U>(stream, width, outputs, flavor);
            small_quantized_case<5U>(stream, width, outputs, flavor);
            small_quantized_case<6U>(stream, width, outputs, flavor);
            small_quantized_case<7U>(stream, width, outputs, flavor);
            small_quantized_case<8U>(stream, width, outputs, flavor);
        }
        std::cout << "small quantized WMMA == dot4 == serial reference; canaries intact: 84 cases\n";
    }
    if (std::getenv("STRIX_DENSE_F16_ROWS_BENCH")) { rows_bench(stream); return 0; }
    if (std::getenv("STRIX_INTPREFILL_GATES")) {
        for (const auto& shape : shapes) {
            const auto run = [&]<class In>() {
                schedule_shape<In,32U>(shape,stream,0U);
                schedule_shape<In,64U>(shape,stream,0U);
                schedule_shape<In,128U>(shape,stream,0U);
                schedule_shape<In,256U>(shape,stream,0U);
                const Shape tail{shape.name,shape.width,1000U,shape.bf16};
                schedule_shape<In,64U>(tail,stream,1U);
            };
            if(shape.bf16)run.template operator()<std::uint16_t>();else run.template operator()<float>();
        }
        check(hipStreamDestroy(stream), "destroy schedule stream");
        return 0;
    }
    if (!std::getenv("STRIX_DENSE_A8_BENCH_ONLY")) {
        std::uint32_t engine_cases = 0;
        for (const auto& shape : shapes) {
            for (unsigned flavor = 0; flavor < 3U; ++flavor) engine_case(stream, shape, flavor, shape.outputs, engine_cases);
            for (const std::uint32_t outputs : {64U, 192U, 1000U}) engine_case(stream, shape, outputs % 3U, outputs, engine_cases);
            for (unsigned flavor = 0; flavor < 3U; ++flavor) engine_case<256U>(stream, shape, flavor, shape.outputs, engine_cases);
            engine_case<256U>(stream, shape, 1U, 1000U, engine_cases);
            std::cout << "engine " << shape.name << ": ok\n";
        }
        std::cout << "dense A8 engine forms (rows 1..8, 16, 32, 64, 128, 256) == serial reference: " << engine_cases << " cases\n";
    }
    if (!std::getenv("STRIX_DENSE_A8_BENCH_ONLY") && !std::getenv("STRIX_DENSE_A8_ENGINE_ONLY")) {
        std::uint32_t cases = 0;
        for (const auto& shape : shapes) {
            for (const std::uint32_t rows : {64U, 128U, 256U})
                for (unsigned flavor = 0; flavor < 3U; ++flavor)
                    exact_case(stream, shape, rows, flavor, shape.outputs, rows == 64U && flavor == 0U && shape.outputs <= 2560U, cases);
            // Row/output tails (not multiples of the tiles) and odd row counts.
            for (const std::uint32_t rows : {1U, 17U, 100U})
                for (const std::uint32_t outputs : {1U, 33U, 1000U})
                    exact_case(stream, shape, rows, rows % 3U, outputs, true, cases);
            std::cout << shape.name << ": ok\n";
        }
        std::cout << "dense A8: quantizer == reference, WMMA == dot4 == serial reference bitwise: " << cases << " cases\n";
    }
    if (std::getenv("STRIX_DENSE_F16")) f16_bench(stream, std::getenv("STRIX_DENSE_F16")[0] == '2');
    if (std::getenv("STRIX_DENSE_A8_PROMPT_BENCH")) prompt_bench(stream);
    if (std::getenv("STRIX_DENSE_A8_EXACT_BENCH")) exact_bench(stream);
    if (std::getenv("STRIX_DENSE_A8_BENCH")) engine_bench(stream);
    if (std::getenv("STRIX_DENSE_A8_BENCH") && !std::getenv("STRIX_DENSE_A8_ENGINE_ONLY")) bench(stream);
    check(hipStreamDestroy(stream), "destroy stream");
    return 0;
} catch (const std::exception& error) {
    std::cerr << "dense A8 fixture failed: " << error.what() << '\n';
    return 1;
}
