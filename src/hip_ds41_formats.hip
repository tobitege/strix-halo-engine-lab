#pragma clang fp contract(off)
#include "strix/env_flags.hpp"
#include "strix/hip_check.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_routes_fast.hpp"
#include "hip_ds41_routes_fast_body.hpp"
#include "strix/ds41_scalar_codecs.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "moe_core/route.hpp"

#include "strix/ds41_moe.hpp"
#include "strix/dist/staged_ready.hpp"

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <stdexcept>
#include <string>


namespace strix::ds41 {
namespace {

__device__ const unsigned long long iq2_grid_device[256] = {
#include "strix/ds41_iq2_xxs_grid.inc"
};

void check(hipError_t status, const char* what) {
    ::strix::hip_check_prefixed("ds41 formats: ", status, what);
}

__device__ __forceinline__ float half_bits_to_float(unsigned short bits) {
    return __half2float(__ushort_as_half(bits));
}
__device__ __forceinline__ unsigned short load_u16(const unsigned char* p) {
    return static_cast<unsigned short>(p[0] | (p[1] << 8));
}
__device__ __forceinline__ unsigned long long load_u64(const unsigned char* p) {
    unsigned long long v;
    __builtin_memcpy(&v, p, 8);
    return v;
}
#if defined(STRIX_EXPERIMENTAL_MALL_Q2_DOWN_NT)
// Cache policy only: preserve each Q2_K code, scale, group order and arithmetic.
// These unaligned types match the packed weight storage, including TP half tails.
using MallQ2Words = unsigned __attribute__((ext_vector_type(8), aligned(1)));
using MallQ2Word = unsigned __attribute__((aligned(1)));
using MallQ2Half = unsigned short __attribute__((aligned(1)));
__device__ __forceinline__ void mall_q2_codes(unsigned (&q)[8], const unsigned char* p) {
    const auto v = __builtin_nontemporal_load(reinterpret_cast<const MallQ2Words*>(p));
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) q[k] = v[k];
}
__device__ __forceinline__ unsigned mall_q2_word(const unsigned char* p) {
    return __builtin_nontemporal_load(reinterpret_cast<const MallQ2Word*>(p));
}
__device__ __forceinline__ unsigned short mall_q2_half(const unsigned char* p) {
    return __builtin_nontemporal_load(reinterpret_cast<const MallQ2Half*>(p));
}
#endif
// Scalar codecs: the shared host/device definitions (ds41_scalar_codecs.hpp).
__device__ __forceinline__ float e8m0_device(unsigned code) { return codec::e8m0(static_cast<std::uint8_t>(code)); }
__device__ __forceinline__ float e8m0_half_device(unsigned code) { return codec::e8m0_half(static_cast<std::uint8_t>(code)); }
__device__ __forceinline__ float e4m3_device(unsigned code) { return codec::e4m3(static_cast<std::uint8_t>(code)); }
// Byte masks 0xff for the set bits of a 4-bit value (bit j -> byte j).
__device__ __forceinline__ unsigned byte_masks(unsigned bits4) {
    return ((bits4 * 0x00204081U) & 0x01010101U) * 0xffU;
}
__device__ __forceinline__ unsigned sign_byte(unsigned index) {
    index &= 127U;
    return index | ((static_cast<unsigned>(__builtin_popcount(index)) & 1U) << 7U);
}
// Bytes of magnitude m (all < 128) negated where mask bytes are 0xff, without carries between bytes.
__device__ __forceinline__ int apply_signs(unsigned magnitudes, unsigned mask) {
    return static_cast<int>((magnitudes ^ mask) + (mask & 0x01010101U));
}
// Doubled E2M1 of the four nibbles selected into bytes: entries 0-7 and 8-15 by v_perm.
__device__ __forceinline__ int e2m1_bytes(unsigned nibbles) {
    const unsigned index = nibbles & 0x07070707U, negative = ((nibbles >> 3U) & 0x01010101U) * 0xffU;
    const unsigned positive_values = __builtin_amdgcn_perm(0x0c080604U, 0x03020100U, index);
    const unsigned negative_values = __builtin_amdgcn_perm(0xf4f8fafcU, 0xfdfeff00U, index);
    return static_cast<int>((positive_values & ~negative) | (negative_values & negative));
}

// One 32-weight group in integer form: eight dwords of int8 codes and the FP32 scale terms.
struct IqGroup { int w[8]; float scale; };
__device__ __forceinline__ IqGroup iq2_group(const unsigned char* block, unsigned group, const unsigned long long* grid) {
    IqGroup out;
    const float d = half_bits_to_float(load_u16(block));
    const unsigned long long aux = load_u64(block + 2U + 8U * group);
    const unsigned aux0 = static_cast<unsigned>(aux), aux1 = static_cast<unsigned>(aux >> 32U);
    out.scale = d * (0.5F + static_cast<float>(aux1 >> 28U)) * 0.25F;
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) {
        const unsigned long long g = grid[(aux0 >> (8U * l)) & 0xffU];
        const unsigned signs = sign_byte(aux1 >> (7U * l));
        out.w[2U * l] = apply_signs(static_cast<unsigned>(g), byte_masks(signs & 15U));
        out.w[2U * l + 1U] = apply_signs(static_cast<unsigned>(g >> 32U), byte_masks(signs >> 4U));
    }
    return out;
}
__device__ __forceinline__ IqGroup mxfp4_group_device(const unsigned char* block) {
    IqGroup out;
    out.scale = e8m0_half_device(block[0]);
    unsigned q[4];
    __builtin_memcpy(q, block + 1, 16);
#pragma unroll
    for (unsigned k = 0; k < 4U; ++k) {
        out.w[k] = e2m1_bytes(q[k] & 0x0f0f0f0fU);
        out.w[k + 4U] = e2m1_bytes((q[k] >> 4U) & 0x0f0f0f0fU);
    }
    return out;
}
// Q2_K group: unsigned 2-bit codes per byte, the two scale bytes and the block's d / dmin.
struct Q2Group { int w[8]; unsigned scale0, scale1; float d, dmin; };
__device__ __forceinline__ Q2Group q2_group(const unsigned char* block, unsigned group) {
    Q2Group out;
    const unsigned half = group / 4U, shift = 2U * (group % 4U);
    unsigned q[8];
#if defined(STRIX_EXPERIMENTAL_MALL_Q2_DOWN_NT)
    mall_q2_codes(q, block + 16U + 32U * half);
#else
    __builtin_memcpy(q, block + 16U + 32U * half, 32);
#endif
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) out.w[k] = static_cast<int>((q[k] >> shift) & 0x03030303U);
#if defined(STRIX_EXPERIMENTAL_MALL_Q2_DOWN_NT)
    const unsigned scales = mall_q2_half(block + 8U * half + 2U * (group % 4U));
    out.scale0 = scales & 0xffU;
    out.scale1 = scales >> 8U;
    out.d = half_bits_to_float(mall_q2_half(block + 80U));
    out.dmin = half_bits_to_float(mall_q2_half(block + 82U));
#else
    out.scale0 = block[8U * half + 2U * (group % 4U)];
    out.scale1 = block[8U * half + 2U * (group % 4U) + 1U];
    out.d = half_bits_to_float(load_u16(block + 80U));
    out.dmin = half_bits_to_float(load_u16(block + 82U));
#endif
    return out;
}
// Raw bytes of one 32-weight group, loaded apart from their decoding so that a wave can have the
// next group of every weight row in flight while it decodes the current one.
template <WeightFormat Format> struct RawGroup;
template <> struct RawGroup<WeightFormat::iq2_xxs> { unsigned short d; unsigned long long aux; };
template <> struct RawGroup<WeightFormat::q2_k> { unsigned q[8]; unsigned short scales; unsigned dm; unsigned shift; };
template <> struct RawGroup<WeightFormat::mxfp4> { unsigned e; unsigned q[4]; };
template <> struct RawGroup<WeightFormat::q8_0> { unsigned short d; unsigned q[8]; };
template <> struct RawGroup<WeightFormat::cb4> { unsigned short d; unsigned q[4]; };
template <> struct RawGroup<WeightFormat::u5> { unsigned short d; unsigned q[4]; unsigned h; };
template <> struct RawGroup<WeightFormat::u6> { unsigned short d; unsigned q[4]; unsigned h[2]; };

__device__ __forceinline__ unsigned short load_u16_raw(const unsigned char* p) {
    unsigned short v;
    __builtin_memcpy(&v, p, 2);
    return v;
}
// Group g of a weight row in the given format.
template <WeightFormat Format>
__device__ __forceinline__ RawGroup<Format> load_raw(const unsigned char* row, unsigned g) {
    RawGroup<Format> raw;
    if constexpr (Format == WeightFormat::iq2_xxs) {
        const auto* block = row + g / 8U * iq2_xxs_block_bytes;
        raw.d = load_u16_raw(block);
        raw.aux = load_u64(block + 2U + 8U * (g % 8U));
    } else if constexpr (Format == WeightFormat::q2_k) {
        const auto* block = row + g / 8U * q2_k_block_bytes;
        const unsigned group = g % 8U, half = group / 4U;
#if defined(STRIX_EXPERIMENTAL_MALL_Q2_DOWN_NT)
        mall_q2_codes(raw.q, block + 16U + 32U * half);
        raw.scales = mall_q2_half(block + 8U * half + 2U * (group % 4U));
        raw.dm = mall_q2_word(block + 80U);
#else
        __builtin_memcpy(raw.q, block + 16U + 32U * half, 32);
        raw.scales = load_u16_raw(block + 8U * half + 2U * (group % 4U));
        __builtin_memcpy(&raw.dm, block + 80U, 4);
#endif
        raw.shift = 2U * (group % 4U);
    } else if constexpr (Format == WeightFormat::q8_0) {
        const auto* block = row + g * q8_0_block_bytes;
        raw.d = load_u16_raw(block);
        __builtin_memcpy(raw.q, block + 2, 32);
    } else if constexpr (Format == WeightFormat::cb4) {
        const auto* block = row + g * cb4_block_bytes;
        raw.d = load_u16_raw(block);
        __builtin_memcpy(raw.q, block + 2, 16);
    } else if constexpr (Format == WeightFormat::u5) {
        const auto* block = row + g * u5_block_bytes;
        raw.d = load_u16_raw(block);
        __builtin_memcpy(raw.q, block + 2, 16);
        __builtin_memcpy(&raw.h, block + 18, 4);
    } else if constexpr (Format == WeightFormat::u6) {
        const auto* block = row + g * u6_block_bytes;
        raw.d = load_u16_raw(block);
        __builtin_memcpy(raw.q, block + 2, 16);
        __builtin_memcpy(raw.h, block + 18, 8);
    } else {
        const auto* block = row + g * mxfp4_block_bytes;
        raw.e = block[0];
        __builtin_memcpy(raw.q, block + 1, 16);
    }
    return raw;
}
__device__ __forceinline__ IqGroup decode_raw(const RawGroup<WeightFormat::iq2_xxs>& raw, const unsigned long long* grid) {
    IqGroup out;
    const float d = half_bits_to_float(raw.d);
    const unsigned aux0 = static_cast<unsigned>(raw.aux), aux1 = static_cast<unsigned>(raw.aux >> 32U);
    out.scale = d * (0.5F + static_cast<float>(aux1 >> 28U)) * 0.25F;
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) {
        const unsigned long long g = grid[(aux0 >> (8U * l)) & 0xffU];
        const unsigned signs = sign_byte(aux1 >> (7U * l));
        out.w[2U * l] = apply_signs(static_cast<unsigned>(g), byte_masks(signs & 15U));
        out.w[2U * l + 1U] = apply_signs(static_cast<unsigned>(g >> 32U), byte_masks(signs >> 4U));
    }
    return out;
}
__device__ __forceinline__ IqGroup decode_raw(const RawGroup<WeightFormat::mxfp4>& raw, const unsigned long long*) {
    IqGroup out;
    out.scale = e8m0_half_device(raw.e);
#pragma unroll
    for (unsigned k = 0; k < 4U; ++k) {
        out.w[k] = e2m1_bytes(raw.q[k] & 0x0f0f0f0fU);
        out.w[k + 4U] = e2m1_bytes((raw.q[k] >> 4U) & 0x0f0f0f0fU);
    }
    return out;
}
__device__ __forceinline__ IqGroup decode_raw(const RawGroup<WeightFormat::q8_0>& raw, const unsigned long long*) {
    IqGroup out;
    out.scale = half_bits_to_float(raw.d);
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) out.w[k] = static_cast<int>(raw.q[k]);
    return out;
}
// CB4: the table (entries 0-7 in grid[0], 8-15 in grid[1]) looked up by byte permutes, 4 indices per word.
__device__ __forceinline__ unsigned cb4_lookup(unsigned idx, unsigned t0, unsigned t1, unsigned t2, unsigned t3) {
    const unsigned sel = idx & 0x07070707U;
    const unsigned lo = __builtin_amdgcn_perm(t1, t0, sel), hi = __builtin_amdgcn_perm(t3, t2, sel);
    const unsigned high = ((idx >> 3U) & 0x01010101U) * 0xffU;
    return (hi & high) | (lo & ~high);
}
__device__ __forceinline__ IqGroup decode_raw(const RawGroup<WeightFormat::cb4>& raw, const unsigned long long* table) {
    IqGroup out;
    out.scale = half_bits_to_float(raw.d);
    const auto t0 = static_cast<unsigned>(table[0]), t1 = static_cast<unsigned>(table[0] >> 32U);
    const auto t2 = static_cast<unsigned>(table[1]), t3 = static_cast<unsigned>(table[1] >> 32U);
#pragma unroll
    for (unsigned k = 0; k < 4U; ++k) {
        out.w[k] = static_cast<int>(cb4_lookup(raw.q[k] & 0x0f0f0f0fU, t0, t1, t2, t3));
        out.w[k + 4U] = static_cast<int>(cb4_lookup((raw.q[k] >> 4U) & 0x0f0f0f0fU, t0, t1, t2, t3));
    }
    return out;
}
// U5 / U6: the uniform 5 / 6-bit values to signed codes (high bits spread to bit 4 / bits 4-5 of each byte).
__device__ __forceinline__ unsigned u5_word(unsigned nib, unsigned h4) {
    const unsigned v = nib | (((h4 & 1U) | ((h4 & 2U) << 7U) | ((h4 & 4U) << 14U) | ((h4 & 8U) << 21U)) << 4U);
    const unsigned b = v ^ 0x10101010U;
    return b | ((b & 0x10101010U) * 14U);
}
__device__ __forceinline__ unsigned u6_word(unsigned nib, unsigned h8) {
    const unsigned v = nib | (((h8 & 3U) | ((h8 & 0xcU) << 6U) | ((h8 & 0x30U) << 12U) | ((h8 & 0xc0U) << 18U)) << 4U);
    const unsigned b = v ^ 0x20202020U;
    return b | ((b & 0x20202020U) * 6U);
}
__device__ __forceinline__ IqGroup decode_raw(const RawGroup<WeightFormat::u5>& raw, const unsigned long long*) {
    IqGroup out;
    out.scale = half_bits_to_float(raw.d);
#pragma unroll
    for (unsigned k = 0; k < 4U; ++k) {
        out.w[k] = static_cast<int>(u5_word(raw.q[k] & 0x0f0f0f0fU, (raw.h >> (4U * k)) & 0xfU));
        out.w[k + 4U] = static_cast<int>(u5_word((raw.q[k] >> 4U) & 0x0f0f0f0fU, (raw.h >> (16U + 4U * k)) & 0xfU));
    }
    return out;
}
__device__ __forceinline__ IqGroup decode_raw(const RawGroup<WeightFormat::u6>& raw, const unsigned long long*) {
    IqGroup out;
    out.scale = half_bits_to_float(raw.d);
#pragma unroll
    for (unsigned k = 0; k < 4U; ++k) {
        out.w[k] = static_cast<int>(u6_word(raw.q[k] & 0x0f0f0f0fU, (raw.h[0] >> (8U * k)) & 0xffU));
        out.w[k + 4U] = static_cast<int>(u6_word((raw.q[k] >> 4U) & 0x0f0f0f0fU, (raw.h[1] >> (8U * k)) & 0xffU));
    }
    return out;
}
__device__ __forceinline__ Q2Group decode_raw(const RawGroup<WeightFormat::q2_k>& raw, const unsigned long long*) {
    Q2Group out;
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) out.w[k] = static_cast<int>((raw.q[k] >> raw.shift) & 0x03030303U);
    out.scale0 = raw.scales & 0xffU;
    out.scale1 = raw.scales >> 8U;
    out.d = half_bits_to_float(static_cast<unsigned short>(raw.dm & 0xffffU));
    out.dmin = half_bits_to_float(static_cast<unsigned short>(raw.dm >> 16U));
    return out;
}
__device__ __forceinline__ int signed_dot(const int (&w)[8], const int4 x0, const int4 x1) {
    int dot = 0;
    dot = __builtin_amdgcn_sudot4(true, w[0], true, x0.x, dot, false);
    dot = __builtin_amdgcn_sudot4(true, w[1], true, x0.y, dot, false);
    dot = __builtin_amdgcn_sudot4(true, w[2], true, x0.z, dot, false);
    dot = __builtin_amdgcn_sudot4(true, w[3], true, x0.w, dot, false);
    dot = __builtin_amdgcn_sudot4(true, w[4], true, x1.x, dot, false);
    dot = __builtin_amdgcn_sudot4(true, w[5], true, x1.y, dot, false);
    dot = __builtin_amdgcn_sudot4(true, w[6], true, x1.z, dot, false);
    dot = __builtin_amdgcn_sudot4(true, w[7], true, x1.w, dot, false);
    return dot;
}
__device__ __forceinline__ int unsigned_dot(int w0, int w1, int w2, int w3, int x0, int x1, int x2, int x3) {
    int dot = 0;
    dot = __builtin_amdgcn_sudot4(false, w0, true, x0, dot, false);
    dot = __builtin_amdgcn_sudot4(false, w1, true, x1, dot, false);
    dot = __builtin_amdgcn_sudot4(false, w2, true, x2, dot, false);
    dot = __builtin_amdgcn_sudot4(false, w3, true, x3, dot, false);
    return dot;
}

__global__ __launch_bounds__(256) void decode_kernel(WeightFormat format, const unsigned char* __restrict__ blocks,
                                                     unsigned long long groups, float* __restrict__ out) {
    __shared__ unsigned long long grid[256];
    grid[threadIdx.x] = iq2_grid_device[threadIdx.x];
    __syncthreads();
    const unsigned long long g = static_cast<unsigned long long>(blockIdx.x) * 256U + threadIdx.x;
    if (g >= groups) return;
    float* y = out + 32U * g;
    if (format == WeightFormat::q2_k) {
        const auto q = q2_group(blocks + g / 8U * q2_k_block_bytes, static_cast<unsigned>(g % 8U));
        for (unsigned j = 0; j < 32U; ++j) {
            const unsigned sc = j < 16U ? q.scale0 : q.scale1;
            const float dl = q.d * static_cast<float>(sc & 15U), ml = q.dmin * static_cast<float>(sc >> 4U);
            const unsigned code = (static_cast<unsigned>(q.w[j / 4U]) >> (8U * (j % 4U))) & 0xffU;
            y[j] = dl * static_cast<float>(code) - ml;
        }
        return;
    }
    const auto q = format == WeightFormat::iq2_xxs ? iq2_group(blocks + g / 8U * iq2_xxs_block_bytes, static_cast<unsigned>(g % 8U), grid)
        : format == WeightFormat::q8_0             ? decode_raw(load_raw<WeightFormat::q8_0>(blocks, static_cast<unsigned>(g)), grid)
                                                   : mxfp4_group_device(blocks + g * mxfp4_block_bytes);
    for (unsigned j = 0; j < 32U; ++j) {
        const auto code = static_cast<signed char>((static_cast<unsigned>(q.w[j / 4U]) >> (8U * (j % 4U))) & 0xffU);
        y[j] = format == WeightFormat::iq2_xxs ? q.scale * static_cast<float>(code) : static_cast<float>(code) * q.scale;
    }
}

__global__ __launch_bounds__(256) void engram_kernel(const unsigned char* __restrict__ rows, unsigned long long row_count,
                                                     float* __restrict__ out) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x) * 256U + threadIdx.x;
    if (index >= row_count * engram_row_elements) return;
    const auto* row = rows + index / engram_row_elements * engram_row_bytes;
    const unsigned j = static_cast<unsigned>(index % engram_row_elements);
    out[index] = e4m3_device(row[j]) * e8m0_device(row[256U + j / 32U]);
}

__global__ __launch_bounds__(256) void scalar_tables_kernel(float* out) {
    const unsigned c = threadIdx.x;
    out[c] = e4m3_device(c);
    out[256U + c] = e8m0_device(c);
    out[512U + c] = e8m0_half_device(c);
    if (c < 16U) {
        out[768U + c] = codec::e2m1(static_cast<std::uint8_t>(c));
    }
}

__global__ __launch_bounds__(32) void quantize_a8_kernel(const float* __restrict__ values, unsigned width, signed char* __restrict__ codes,
                                                         float* __restrict__ deltas, int* __restrict__ sums) {
    const unsigned lane = threadIdx.x, group = blockIdx.x, row = blockIdx.y, groups = width / 32U;
    const unsigned long long at = static_cast<unsigned long long>(row) * width + 32U * group + lane;
    const float x = values[at];
    float amax = fabsf(x);
    for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 32));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
    const float rounded = tiny ? 0.0F : rintf(x / delta);
    const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
    int sum = code;
    for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, 32);
    codes[at] = static_cast<signed char>(code);
    if (lane == 0U) {
        deltas[static_cast<unsigned long long>(row) * groups + group] = delta;
        sums[static_cast<unsigned long long>(row) * groups + group] = sum;
    }
}

// Window products. A block owns one expert of the routes and stages the A8 rows of its entries in
// LDS; each wave owns W weight rows, decodes every 32-weight group of them once and applies it to
// all staged rows. Per (entry, output) the lane, group order and butterfly are those of the declared
// product, independently of W and of the other entries.
// Blocks of 8 waves, or 16 when experts have four or more entries (more resident waves for the
// same staged rows); a block always covers 32 outputs.
template <unsigned MaxRows> inline constexpr unsigned window_waves = MaxRows >= 4U ? 16U : 8U;

struct EntryRange { unsigned begin, count, expert; };
// Routes built on the device give their expert count in counts[0]; blocks past it have no entries.
__device__ __forceinline__ EntryRange entry_range(const RoutesView& routes, unsigned dense_rows) {
    const unsigned j = blockIdx.y;
    if (!routes.begins) return {0U, dense_rows, routes.experts[j]};
    if (routes.counts && j >= routes.counts[0]) return {0U, 0U, 0U};
    return {routes.begins[j], routes.begins[j + 1U] - routes.begins[j], routes.experts[j]};
}
__device__ __forceinline__ unsigned entry_input(const RoutesView& routes, unsigned i) {
    return routes.begins ? routes.inputs[i] : i;
}
__device__ __forceinline__ unsigned entry_output(const RoutesView& routes, unsigned i, unsigned dense_experts) {
    return routes.begins ? routes.outputs[i] : i * dense_experts + blockIdx.y;
}
// Copies the codes and deltas of the block's entries into LDS: codes[e * width], deltas[e * groups].
__device__ __forceinline__ void stage_rows(const RoutesView& routes, const EntryRange& range, const signed char* __restrict__ codes,
                                           const float* __restrict__ deltas, unsigned width, signed char* __restrict__ lds_codes,
                                           float* __restrict__ lds_deltas) {
    const unsigned words = width / 16U, groups = width / 32U;
    for (unsigned e = 0; e < range.count; ++e) {
        const unsigned long long input = entry_input(routes, range.begin + e);
        const auto* source = reinterpret_cast<const int4*>(codes + input * width);
        auto* target = reinterpret_cast<int4*>(lds_codes + static_cast<unsigned long long>(e) * width);
        for (unsigned i = threadIdx.x; i < words; i += blockDim.x) target[i] = source[i];
        for (unsigned i = threadIdx.x; i < groups; i += blockDim.x)
            lds_deltas[e * groups + i] = deltas[input * groups + i];
    }
    __syncthreads();
}

// The first group of every weight row of the lane (issued before the activations are staged when the
// caller primes it, so the weight stream starts at the top of the block).
// Tail (expert tensor parallelism, Q2_K only): the row is width / 256 whole blocks then one packed half block of 44 bytes
// (scales 8, codes 32, d and dmin) holding the last four groups; the same group values as a whole block's first half.
template <WeightFormat Format, bool Tail>
__device__ __forceinline__ RawGroup<Format> load_group(const unsigned char* row, unsigned g, unsigned width) {
    if constexpr (Tail && Format == WeightFormat::q2_k) {
        const unsigned whole = width / 256U;
        if (g >= 8U * whole) {
            RawGroup<Format> raw;
            const auto* half = row + whole * q2_k_block_bytes;
            const unsigned group = g % 4U;
#if defined(STRIX_EXPERIMENTAL_MALL_Q2_DOWN_NT)
            mall_q2_codes(raw.q, half + 8U);
            raw.scales = mall_q2_half(half + 2U * group);
            raw.dm = mall_q2_word(half + 40U);
#else
            __builtin_memcpy(raw.q, half + 8U, 32);
            raw.scales = load_u16_raw(half + 2U * group);
            __builtin_memcpy(&raw.dm, half + 40U, 4);
#endif
            raw.shift = 2U * group;
            return raw;
        }
    }
    return load_raw<Format>(row, g);
}
template <WeightFormat Format, unsigned W, bool Tail = false>
__device__ __forceinline__ void load_first(const unsigned char* const (&rows)[W], unsigned width, RawGroup<Format> (&current)[W]) {
    const unsigned lane = threadIdx.x % 32U;
    if (lane < width / 32U)
#pragma unroll
        for (unsigned w = 0; w < W; ++w) current[w] = load_group<Format, Tail>(rows[w], lane, width);
}
template <WeightFormat Format, unsigned MaxRows, unsigned W, bool Tail = false>
__device__ __forceinline__ void accumulate_rows(const unsigned char* const (&rows)[W], const signed char* lds_codes,
                                                const float* lds_deltas, unsigned count, unsigned width,
                                                const unsigned long long* grid, float (&acc)[W][MaxRows],
                                                RawGroup<Format> (&current)[W], bool primed) {
    const unsigned lane = threadIdx.x % 32U, groups = width / 32U;
#pragma unroll
    for (unsigned w = 0; w < W; ++w)
#pragma unroll
        for (unsigned r = 0; r < MaxRows; ++r) acc[w][r] = 0.0F;
    // The next group of every weight row is loaded before the current one is decoded.
    if (!primed) load_first<Format, W, Tail>(rows, width, current);
    for (unsigned g = lane; g < groups; g += 32U) {
        const bool more = g + 32U < groups;
        RawGroup<Format> next[W];
        if (more)
#pragma unroll
            for (unsigned w = 0; w < W; ++w) next[w] = load_group<Format, Tail>(rows[w], g + 32U, width);
        int4 x0[MaxRows], x1[MaxRows];
        float delta[MaxRows];
        int sx0[MaxRows], sx1[MaxRows];
#pragma unroll
        for (unsigned r = 0; r < MaxRows; ++r)
            if (r < count) {
                const auto* x = reinterpret_cast<const int4*>(lds_codes + static_cast<unsigned long long>(r) * width + 32U * g);
                x0[r] = x[0]; x1[r] = x[1];
                delta[r] = lds_deltas[r * groups + g];
                if constexpr (Format == WeightFormat::q2_k) {
                    // Activation sums of the two 16-element halves, shared by every weight row.
                    const int ones = 0x01010101;
                    sx0[r] = unsigned_dot(ones, ones, ones, ones, x0[r].x, x0[r].y, x0[r].z, x0[r].w);
                    sx1[r] = unsigned_dot(ones, ones, ones, ones, x1[r].x, x1[r].y, x1[r].z, x1[r].w);
                }
            }
#pragma unroll
        for (unsigned w = 0; w < W; ++w) {
            const auto q = decode_raw(current[w], grid);
            if constexpr (Format == WeightFormat::q2_k) {
#pragma unroll
                for (unsigned r = 0; r < MaxRows; ++r)
                    if (r < count) {
                        const int qx0 = unsigned_dot(q.w[0], q.w[1], q.w[2], q.w[3], x0[r].x, x0[r].y, x0[r].z, x0[r].w);
                        const int qx1 = unsigned_dot(q.w[4], q.w[5], q.w[6], q.w[7], x1[r].x, x1[r].y, x1[r].z, x1[r].w);
                        // 24-bit products (full-rate): scales < 16, half sums |.| <= 3 * 127 * 16.
                        const int a = __mul24(static_cast<int>(q.scale0 & 15U), qx0) + __mul24(static_cast<int>(q.scale1 & 15U), qx1);
                        const int b = __mul24(static_cast<int>(q.scale0 >> 4U), sx0[r]) + __mul24(static_cast<int>(q.scale1 >> 4U), sx1[r]);
                        const float scaled = q.d * static_cast<float>(a);
                        const float offset = q.dmin * static_cast<float>(b);
                        acc[w][r] = acc[w][r] + (scaled - offset) * delta[r];
                    }
            } else {
#pragma unroll
                for (unsigned r = 0; r < MaxRows; ++r)
                    if (r < count) acc[w][r] = acc[w][r] + static_cast<float>(signed_dot(q.w, x0[r], x1[r])) * (q.scale * delta[r]);
            }
        }
        if (more)
#pragma unroll
            for (unsigned w = 0; w < W; ++w) current[w] = next[w];
    }
#pragma unroll
    for (unsigned w = 0; w < W; ++w)
#pragma unroll
        for (unsigned r = 0; r < MaxRows; ++r)
            if (r < count)
#pragma unroll
                for (unsigned d = 16U; d; d /= 2U) acc[w][r] = acc[w][r] + __shfl_xor(acc[w][r], static_cast<int>(d), 32);
}

__device__ __forceinline__ void load_grid(unsigned long long* grid) {
    if (threadIdx.x < 256U) grid[threadIdx.x] = iq2_grid_device[threadIdx.x];
}
// CB4: the launch's table into the block's two table words (made visible by stage_rows' barrier).
__device__ __forceinline__ void load_table(unsigned long long* grid, const std::uint32_t (&table)[4]) {
    if (threadIdx.x < 2U) grid[threadIdx.x] = table[2U * threadIdx.x] | (static_cast<unsigned long long>(table[2U * threadIdx.x + 1U]) << 32U);
}
template <WeightFormat Format> inline constexpr unsigned grid_words = Format == WeightFormat::iq2_xxs ? 256U : Format == WeightFormat::cb4 ? 2U : 1U;

// out[output(entry) * outputs + o] for the W outputs of each wave; Waves waves per block. Early: the first
// group of every weight row is issued before the activations are staged.
template <WeightFormat Format, unsigned MaxRows, unsigned W, unsigned Waves, bool Early, bool Tail = false>
__global__ __launch_bounds__(32U * Waves) void routes_product_kernel(RoutesProductView view, unsigned dense_rows,
                                                                            unsigned dense_experts) {
    extern __shared__ int4 staged[];
    __shared__ unsigned long long grid[grid_words<Format>];
    if constexpr (Format == WeightFormat::iq2_xxs) load_grid(grid);
    if constexpr (Format == WeightFormat::cb4) load_table(grid, view.table);
    const auto range = entry_range(view.routes, dense_rows);
    if (range.count == 0U) return; // padded union slot without entries (routes built on the device)
    auto* lds_codes = reinterpret_cast<signed char*>(staged);
    auto* lds_deltas = reinterpret_cast<float*>(lds_codes + static_cast<unsigned long long>(MaxRows) * view.width);
    const unsigned first = (blockIdx.x * Waves + threadIdx.x / 32U) * W;
    const bool active = first < view.outputs;
    const auto* base = static_cast<const unsigned char*>(view.weights) + range.expert * view.expert_stride_bytes;
    const unsigned char* rows[W];
#pragma unroll
    for (unsigned w = 0; w < W; ++w) rows[w] = base + static_cast<unsigned long long>(min(first + w, view.outputs - 1U)) * view.row_bytes;
    RawGroup<Format> first_groups[W];
    if constexpr (Early) load_first<Format, W, Tail>(rows, view.width, first_groups);
    // Entries are staged MaxRows at a time; an entry's result does not depend on the others.
    for (unsigned chunk = 0; chunk < range.count; chunk += MaxRows) {
        const EntryRange part{range.begin + chunk, min(MaxRows, range.count - chunk), range.expert};
        if (chunk) __syncthreads();
        stage_rows(view.routes, part, view.codes, view.deltas, view.width, lds_codes, lds_deltas);
        if (!active) continue;
        float acc[W][MaxRows];
        accumulate_rows<Format, MaxRows, W, Tail>(rows, lds_codes, lds_deltas, part.count, view.width, grid, acc, first_groups,
                                                  Early && chunk == 0U);
        if (threadIdx.x % 32U != 0U) continue;
#pragma unroll
        for (unsigned r = 0; r < MaxRows; ++r)
            if (r < part.count) {
                const unsigned long long output = entry_output(view.routes, part.begin + r, dense_experts);
#pragma unroll
                for (unsigned w = 0; w < W; ++w)
                    if (first + w < view.outputs) view.out[output * view.outputs + first + w] = acc[w][r];
            }
    }
}

// Fused gate/up: each wave owns 32 / waves outputs of both matrices (32 outputs per block); the block
// then applies the clamped SwiGLU to its 32 outputs of every entry and A8-quantizes them as one group.
// Waves: waves per block (32 / Waves outputs each). Early: every wave issues the first group of its
// weight rows before the activations are staged.
template <WeightFormat Format, unsigned MaxRows, unsigned Waves, bool Early>
__global__ __launch_bounds__(32U * Waves) void gate_up_kernel(GateUpView view) {
    constexpr unsigned outs = 32U / Waves;
    extern __shared__ int4 staged[];
    __shared__ unsigned long long grid[grid_words<Format>];
    __shared__ float tile[2][MaxRows][32];
    if constexpr (Format == WeightFormat::iq2_xxs) load_grid(grid);
    if constexpr (Format == WeightFormat::cb4) load_table(grid, view.table);
    const auto range = entry_range(view.routes, 0U);
    if (range.count == 0U) return; // padded union slot without entries (routes built on the device)
    auto* lds_codes = reinterpret_cast<signed char*>(staged);
    auto* lds_deltas = reinterpret_cast<float*>(lds_codes + static_cast<unsigned long long>(MaxRows) * view.hidden);
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, first = blockIdx.x * 32U + wave * outs;
    const unsigned long long expert_base = range.expert * view.expert_stride_bytes;
    const unsigned char* rows[2U * outs];
#pragma unroll
    for (unsigned w = 0; w < outs; ++w) {
        rows[w] = static_cast<const unsigned char*>(view.gate) + expert_base + static_cast<unsigned long long>(first + w) * view.row_bytes;
        rows[w + outs] = static_cast<const unsigned char*>(view.up) + expert_base + static_cast<unsigned long long>(first + w) * view.row_bytes;
    }
    RawGroup<Format> first_groups[2U * outs];
    if constexpr (Early) load_first<Format, 2U * outs>(rows, view.hidden, first_groups);
    // Entries are staged MaxRows at a time; an entry's result does not depend on the others.
    for (unsigned chunk = 0; chunk < range.count; chunk += MaxRows) {
        const EntryRange part{range.begin + chunk, min(MaxRows, range.count - chunk), range.expert};
        if (chunk) __syncthreads();
        stage_rows(view.routes, part, view.codes, view.deltas, view.hidden, lds_codes, lds_deltas);
        float acc[2U * outs][MaxRows];
        accumulate_rows<Format, MaxRows, 2U * outs>(rows, lds_codes, lds_deltas, part.count, view.hidden, grid, acc, first_groups,
                                                    Early && chunk == 0U);
        if (lane == 0U)
#pragma unroll
            for (unsigned r = 0; r < MaxRows; ++r)
                if (r < part.count)
#pragma unroll
                    for (unsigned w = 0; w < outs; ++w) {
                        tile[0][r][wave * outs + w] = acc[w][r];
                        tile[1][r][wave * outs + w] = acc[w + outs][r];
                    }
        __syncthreads();
        for (unsigned e = wave; e < part.count; e += Waves) {
            const unsigned long long slot = view.routes.outputs[part.begin + e];
            const unsigned at = blockIdx.x * 32U + lane, groups = view.inter / 32U;
            const float g = tile[0][e][lane], u = tile[1][e][lane];
            if (view.gate_out) { view.gate_out[slot * view.inter + at] = g; view.up_out[slot * view.inter + at] = u; }
            const float act = swiglu_clamped(g, u, view.limit);
            float amax = fabsf(act);
            for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 32));
            const bool tiny = amax > 0.0F && amax < 0x1p-119F;
            const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
            const float rounded = tiny ? 0.0F : rintf(act / delta);
            const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
            int sum = code;
            for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, 32);
            view.act_codes[slot * view.inter + at] = static_cast<signed char>(code);
            if (lane == 0U) { view.act_deltas[slot * groups + blockIdx.x] = delta; view.act_sums[slot * groups + blockIdx.x] = sum; }
        }
    }
}

// STRIX_DS41_EXPERT_SLOTS: the two kernels above for routes built on the device, the grid covering gridDim.y
// union slots; block (x, y) does slots y, y + gridDim.y, ... below counts[0], each exactly as the block
// (x, slot) of the kernels above (no early weight loads; a barrier before a later slot reuses the LDS).
template <WeightFormat Format, unsigned MaxRows, unsigned W, unsigned Waves, bool Tail = false>
__global__ __launch_bounds__(32U * Waves) void routes_product_slots_kernel(RoutesProductView view) {
    extern __shared__ int4 staged[];
    __shared__ unsigned long long grid[grid_words<Format>];
    if constexpr (Format == WeightFormat::iq2_xxs) load_grid(grid);
    if constexpr (Format == WeightFormat::cb4) load_table(grid, view.table);
    const unsigned union_count = view.routes.counts[0];
    auto* lds_codes = reinterpret_cast<signed char*>(staged);
    auto* lds_deltas = reinterpret_cast<float*>(lds_codes + static_cast<unsigned long long>(MaxRows) * view.width);
    const unsigned first = (blockIdx.x * Waves + threadIdx.x / 32U) * W;
    const bool active = first < view.outputs;
    for (unsigned j = blockIdx.y; j < union_count; j += gridDim.y) {
        if (j != blockIdx.y) __syncthreads();
        const EntryRange range{view.routes.begins[j], view.routes.begins[j + 1U] - view.routes.begins[j], view.routes.experts[j]};
        const auto* base = static_cast<const unsigned char*>(view.weights) + range.expert * view.expert_stride_bytes;
        const unsigned char* rows[W];
#pragma unroll
        for (unsigned w = 0; w < W; ++w) rows[w] = base + static_cast<unsigned long long>(min(first + w, view.outputs - 1U)) * view.row_bytes;
        RawGroup<Format> first_groups[W];
        for (unsigned chunk = 0; chunk < range.count; chunk += MaxRows) {
            const EntryRange part{range.begin + chunk, min(MaxRows, range.count - chunk), range.expert};
            if (chunk) __syncthreads();
            stage_rows(view.routes, part, view.codes, view.deltas, view.width, lds_codes, lds_deltas);
            if (!active) continue;
            float acc[W][MaxRows];
            accumulate_rows<Format, MaxRows, W, Tail>(rows, lds_codes, lds_deltas, part.count, view.width, grid, acc, first_groups, false);
            if (threadIdx.x % 32U != 0U) continue;
#pragma unroll
            for (unsigned r = 0; r < MaxRows; ++r)
                if (r < part.count) {
                    const unsigned long long output = view.routes.outputs[part.begin + r];
#pragma unroll
                    for (unsigned w = 0; w < W; ++w)
                        if (first + w < view.outputs) view.out[output * view.outputs + first + w] = acc[w][r];
                }
        }
    }
}
template <WeightFormat Format, unsigned MaxRows, unsigned Waves>
__global__ __launch_bounds__(32U * Waves) void gate_up_slots_kernel(GateUpView view) {
    constexpr unsigned outs = 32U / Waves;
    extern __shared__ int4 staged[];
    __shared__ unsigned long long grid[grid_words<Format>];
    __shared__ float tile[2][MaxRows][32];
    if constexpr (Format == WeightFormat::iq2_xxs) load_grid(grid);
    if constexpr (Format == WeightFormat::cb4) load_table(grid, view.table);
    const unsigned union_count = view.routes.counts[0];
    auto* lds_codes = reinterpret_cast<signed char*>(staged);
    auto* lds_deltas = reinterpret_cast<float*>(lds_codes + static_cast<unsigned long long>(MaxRows) * view.hidden);
    const unsigned wave = threadIdx.x / 32U, lane = threadIdx.x % 32U, first = blockIdx.x * 32U + wave * outs;
    for (unsigned j = blockIdx.y; j < union_count; j += gridDim.y) {
        if (j != blockIdx.y) __syncthreads();
        const EntryRange range{view.routes.begins[j], view.routes.begins[j + 1U] - view.routes.begins[j], view.routes.experts[j]};
        const unsigned long long expert_base = range.expert * view.expert_stride_bytes;
        const unsigned char* rows[2U * outs];
#pragma unroll
        for (unsigned w = 0; w < outs; ++w) {
            rows[w] = static_cast<const unsigned char*>(view.gate) + expert_base + static_cast<unsigned long long>(first + w) * view.row_bytes;
            rows[w + outs] = static_cast<const unsigned char*>(view.up) + expert_base + static_cast<unsigned long long>(first + w) * view.row_bytes;
        }
        RawGroup<Format> first_groups[2U * outs];
        for (unsigned chunk = 0; chunk < range.count; chunk += MaxRows) {
            const EntryRange part{range.begin + chunk, min(MaxRows, range.count - chunk), range.expert};
            if (chunk) __syncthreads();
            stage_rows(view.routes, part, view.codes, view.deltas, view.hidden, lds_codes, lds_deltas);
            float acc[2U * outs][MaxRows];
            accumulate_rows<Format, MaxRows, 2U * outs>(rows, lds_codes, lds_deltas, part.count, view.hidden, grid, acc, first_groups, false);
            if (lane == 0U)
#pragma unroll
                for (unsigned r = 0; r < MaxRows; ++r)
                    if (r < part.count)
#pragma unroll
                        for (unsigned w = 0; w < outs; ++w) {
                            tile[0][r][wave * outs + w] = acc[w][r];
                            tile[1][r][wave * outs + w] = acc[w + outs][r];
                        }
            __syncthreads();
            for (unsigned e = wave; e < part.count; e += Waves) {
                const unsigned long long slot = view.routes.outputs[part.begin + e];
                const unsigned at = blockIdx.x * 32U + lane, groups = view.inter / 32U;
                const float g = tile[0][e][lane], u = tile[1][e][lane];
                if (view.gate_out) { view.gate_out[slot * view.inter + at] = g; view.up_out[slot * view.inter + at] = u; }
                const float act = swiglu_clamped(g, u, view.limit);
                float amax = fabsf(act);
                for (int stride = 16; stride; stride /= 2) amax = fmaxf(amax, __shfl_xor(amax, stride, 32));
                const bool tiny = amax > 0.0F && amax < 0x1p-119F;
                const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
                const float rounded = tiny ? 0.0F : rintf(act / delta);
                const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
                int sum = code;
                for (int stride = 16; stride; stride /= 2) sum += __shfl_xor(sum, stride, 32);
                view.act_codes[slot * view.inter + at] = static_cast<signed char>(code);
                if (lane == 0U) { view.act_deltas[slot * groups + blockIdx.x] = delta; view.act_sums[slot * groups + blockIdx.x] = sum; }
            }
        }
    }
}

// STRIX_DS41_EXPERT_ROWS: the routed gate/up (IQ2_XXS) and down (Q2_K) with the loads restructured. Every
// (entry, output) keeps the lane classes, group order, terms and butterfly of accumulate_rows, so the bits are
// those of the kernels above.
// - Per union slot a wave issues the activation loads of the first staged part, then the first weight step of its
//   rows (Depth steps; Steps = groups / 32 rounded up per lane), then stores the activations to LDS behind an
//   LDS-only barrier (__syncthreads is a workgroup release fence, which would wait for the weight loads).
// - Depth 1: a rolled loop that issues the next step's loads unconditionally (the last one clamped to a valid group
//   and unused) before decoding the current step; the declared kernels' `if (more)` loads made the compiler wait for
//   the next group before decoding the current one. Depth = Steps (down only): every step held in registers and
//   reused by the later parts of an expert; otherwise later parts reload (from the caches).
// - Weight rows are buffer loads with the row in the scalar offset and one vector offset per step shared by the
//   rows (no 64-bit address registers); the IQ2_XXS scale and the Q2_K scale pair are dword loads (no d16 loads).
// - Signs 1: IQ2_XXS sign masks from a 128-entry LDS table of byte_masks(sign_byte(i)); 2: the same bytes by
//   v_perm (selector 12 + bit gives 0x00 / 0xff), without 32-bit multiplies.
// - The butterflies move the lanes of __shfl_xor(x, d, 32) by DPP (d < 16) and permlanex16 (d = 16).
struct IqHeld { unsigned d; unsigned long long aux; };      // d: the block's first dword (the FP16 scale below)
struct Q2Held { unsigned q[8]; unsigned scales; unsigned dm; };  // scales: the dword at the group's scale pair
template <WeightFormat Format> struct HeldOf;
template <> struct HeldOf<WeightFormat::iq2_xxs> { using type = IqHeld; };
template <> struct HeldOf<WeightFormat::q2_k> { using type = Q2Held; };
template <WeightFormat Format> using HeldGroup = typename HeldOf<Format>::type;

// The W weight rows of a wave: buffer resources over one expert's matrix (gate/up: rows [0, split) in the gate,
// the others in the up) and each row's byte offset. The offsets are wave-uniform: buffer loads with the row in the
// scalar offset and one vector offset per step shared by the W rows.
template <unsigned W> struct WaveRows { __amdgpu_buffer_rsrc_t rsrc[2]; int offset[W]; unsigned split; };
__device__ __forceinline__ __amdgpu_buffer_rsrc_t weight_rsrc(const void* base) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(base), 0, 0x7fffffff, 0x31004000);  // raw, offset-checked
}
template <WeightFormat Format>
__device__ __forceinline__ HeldGroup<Format> load_held_group(__amdgpu_buffer_rsrc_t rsrc, int row, unsigned g) {
    HeldGroup<Format> raw;
    if constexpr (Format == WeightFormat::iq2_xxs) {
        const unsigned block = g / 8U * static_cast<unsigned>(iq2_xxs_block_bytes);
        raw.d = __builtin_amdgcn_raw_buffer_load_b32(rsrc, static_cast<int>(block), row, 0);
        const auto aux = __builtin_amdgcn_raw_buffer_load_b64(rsrc, static_cast<int>(block + 2U + 8U * (g % 8U)), row, 0);
        raw.aux = aux[0] | (static_cast<unsigned long long>(aux[1]) << 32U);
    } else {
        const unsigned block = g / 8U * static_cast<unsigned>(q2_k_block_bytes), group = g % 8U, half = group / 4U;
#if defined(STRIX_EXPERIMENTAL_MALL_Q2_DOWN_NT)
        constexpr int policy = 6;  // gfx1151 SLC + DLC, as the non-temporal global loads above.
#else
        constexpr int policy = 0;
#endif
        const auto q0 = __builtin_amdgcn_raw_buffer_load_b128(rsrc, static_cast<int>(block + 16U + 32U * half), row, policy);
        const auto q1 = __builtin_amdgcn_raw_buffer_load_b128(rsrc, static_cast<int>(block + 32U + 32U * half), row, policy);
#pragma unroll
        for (unsigned k = 0; k < 4U; ++k) { raw.q[k] = q0[k]; raw.q[k + 4U] = q1[k]; }
        raw.scales = __builtin_amdgcn_raw_buffer_load_b32(rsrc, static_cast<int>(block + 8U * half + 2U * (group % 4U)), row, policy);
        raw.dm = __builtin_amdgcn_raw_buffer_load_b32(rsrc, static_cast<int>(block + 80U), row, policy);
    }
    return raw;
}
template <WeightFormat Format, unsigned W>
__device__ __forceinline__ void load_step(const WaveRows<W>& rows, unsigned g, HeldGroup<Format> (&held)[W]) {
#pragma unroll
    for (unsigned w = 0; w < W; ++w) held[w] = load_held_group<Format>(rows.rsrc[w < rows.split ? 0U : 1U], rows.offset[w], g);
}
__device__ __forceinline__ unsigned byte_masks_perm(unsigned bits4) {
    return __builtin_amdgcn_perm(0U, 0U, ((bits4 * 0x00204081U) & 0x01010101U) | 0x0c0c0c0cU);
}
__device__ __forceinline__ void load_sign_masks(unsigned long long* masks) {
    if (threadIdx.x < 128U) {
        const unsigned s = sign_byte(threadIdx.x);
        masks[threadIdx.x] = byte_masks(s & 15U) | (static_cast<unsigned long long>(byte_masks(s >> 4U)) << 32U);
    }
}
template <unsigned Signs>
__device__ __forceinline__ IqGroup decode_held(const IqHeld& raw, const unsigned long long* grid, const unsigned long long* masks) {
    IqGroup out;
    const float d = half_bits_to_float(static_cast<unsigned short>(raw.d & 0xffffU));
    const unsigned aux0 = static_cast<unsigned>(raw.aux), aux1 = static_cast<unsigned>(raw.aux >> 32U);
    out.scale = d * (0.5F + static_cast<float>(aux1 >> 28U)) * 0.25F;
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) {
        const unsigned long long g = grid[(aux0 >> (8U * l)) & 0xffU];
        unsigned m0, m1;
        if constexpr (Signs == 1U) {
            const unsigned long long m = masks[(aux1 >> (7U * l)) & 127U];
            m0 = static_cast<unsigned>(m); m1 = static_cast<unsigned>(m >> 32U);
        } else {
            const unsigned signs = sign_byte(aux1 >> (7U * l));
            m0 = Signs == 2U ? byte_masks_perm(signs & 15U) : byte_masks(signs & 15U);
            m1 = Signs == 2U ? byte_masks_perm(signs >> 4U) : byte_masks(signs >> 4U);
        }
        out.w[2U * l] = apply_signs(static_cast<unsigned>(g), m0);
        out.w[2U * l + 1U] = apply_signs(static_cast<unsigned>(g >> 32U), m1);
    }
    return out;
}
__device__ __forceinline__ Q2Group decode_held(const Q2Held& raw, unsigned shift) {
    Q2Group out;
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) out.w[k] = static_cast<int>((raw.q[k] >> shift) & 0x03030303U);
    out.scale0 = raw.scales & 0xffU;
    out.scale1 = (raw.scales >> 8U) & 0xffU;
    out.d = half_bits_to_float(static_cast<unsigned short>(raw.dm & 0xffffU));
    out.dmin = half_bits_to_float(static_cast<unsigned short>(raw.dm >> 16U));
    return out;
}
// The value of lane ^ D (as __shfl_xor(v, D, 32)).
template <unsigned D> __device__ __forceinline__ int lane_xor_bits(int v) {
    if constexpr (D == 16U) {
        const auto u = static_cast<unsigned>(v);
        return static_cast<int>(__builtin_amdgcn_permlanex16(u, u, 0x76543210U, 0xfedcba98U, false, false));
    }
    else return __builtin_amdgcn_update_dpp(v, v, 0x160 + static_cast<int>(D), 0xf, 0xf, false);  // DPP row_xmask
}
template <unsigned D> __device__ __forceinline__ float lane_xor(float v) { return __int_as_float(lane_xor_bits<D>(__float_as_int(v))); }
__device__ __forceinline__ float butterfly_sum(float v) {
    v = v + lane_xor<16U>(v); v = v + lane_xor<8U>(v); v = v + lane_xor<4U>(v); v = v + lane_xor<2U>(v);
    return v + lane_xor<1U>(v);
}
__device__ __forceinline__ float butterfly_max(float v) {
    v = fmaxf(v, lane_xor<16U>(v)); v = fmaxf(v, lane_xor<8U>(v)); v = fmaxf(v, lane_xor<4U>(v)); v = fmaxf(v, lane_xor<2U>(v));
    return fmaxf(v, lane_xor<1U>(v));
}
__device__ __forceinline__ int butterfly_isum(int v) {
    v += lane_xor_bits<16U>(v); v += lane_xor_bits<8U>(v); v += lane_xor_bits<4U>(v); v += lane_xor_bits<2U>(v);
    return v + lane_xor_bits<1U>(v);
}

// Steps [0, Depth) of the lane's weight rows, step-major (step i is group lane + 32 i, clamped to the last group).
template <WeightFormat Format, unsigned W, unsigned Steps, unsigned Depth>
__device__ __forceinline__ void load_held(const WaveRows<W>& rows, unsigned width, HeldGroup<Format> (&held)[Steps][W]) {
    const unsigned lane = threadIdx.x % 32U, last = width / 32U - 1U;
#pragma unroll
    for (unsigned i = 0; i < (Depth < Steps ? Depth : Steps); ++i) load_step<Format, W>(rows, min(lane + 32U * i, last), held[i]);
    asm volatile("" ::: "memory");  // issued here, ahead of the staging stores and barrier
}
// One step (group g of the lane) of accumulate_rows over the W held groups: the same terms in the same order.
template <WeightFormat Format, unsigned MaxRows, unsigned W, unsigned Signs>
__device__ __forceinline__ void accumulate_group(const HeldGroup<Format> (&held)[W], unsigned g, const signed char* lds_codes,
                                                 const float* lds_deltas, unsigned count, unsigned width, const unsigned long long* grid,
                                                 const unsigned long long* masks, float (&acc)[W][MaxRows]) {
    const unsigned groups = width / 32U;
    int4 x0[MaxRows], x1[MaxRows];
    float delta[MaxRows];
    int sx0[MaxRows], sx1[MaxRows];
#pragma unroll
    for (unsigned r = 0; r < MaxRows; ++r)
        if (r < count) {
            const auto* x = reinterpret_cast<const int4*>(lds_codes + static_cast<unsigned long long>(r) * width + 32U * g);
            x0[r] = x[0]; x1[r] = x[1];
            delta[r] = lds_deltas[r * groups + g];
            if constexpr (Format == WeightFormat::q2_k) {
                const int ones = 0x01010101;
                sx0[r] = unsigned_dot(ones, ones, ones, ones, x0[r].x, x0[r].y, x0[r].z, x0[r].w);
                sx1[r] = unsigned_dot(ones, ones, ones, ones, x1[r].x, x1[r].y, x1[r].z, x1[r].w);
            }
        }
#pragma unroll
    for (unsigned w = 0; w < W; ++w) {
        if constexpr (Format == WeightFormat::q2_k) {
            const auto q = decode_held(held[w], 2U * (g % 4U));
#pragma unroll
            for (unsigned r = 0; r < MaxRows; ++r)
                if (r < count) {
                    const int qx0 = unsigned_dot(q.w[0], q.w[1], q.w[2], q.w[3], x0[r].x, x0[r].y, x0[r].z, x0[r].w);
                    const int qx1 = unsigned_dot(q.w[4], q.w[5], q.w[6], q.w[7], x1[r].x, x1[r].y, x1[r].z, x1[r].w);
                    const int a = __mul24(static_cast<int>(q.scale0 & 15U), qx0) + __mul24(static_cast<int>(q.scale1 & 15U), qx1);
                    const int b = __mul24(static_cast<int>(q.scale0 >> 4U), sx0[r]) + __mul24(static_cast<int>(q.scale1 >> 4U), sx1[r]);
                    const float scaled = q.d * static_cast<float>(a);
                    const float offset = q.dmin * static_cast<float>(b);
                    acc[w][r] = acc[w][r] + (scaled - offset) * delta[r];
                }
        } else {
            const auto q = decode_held<Signs>(held[w], grid, masks);
#pragma unroll
            for (unsigned r = 0; r < MaxRows; ++r)
                if (r < count) acc[w][r] = acc[w][r] + static_cast<float>(signed_dot(q.w, x0[r], x1[r])) * (q.scale * delta[r]);
        }
    }
}
// accumulate_rows over held groups: steps [0, Depth) are loaded by the caller; step i issues the loads of step
// i + Depth before it decodes step i (no copies, no conditional loads). Then the butterfly of accumulate_rows.
template <WeightFormat Format, unsigned MaxRows, unsigned W, unsigned Steps, unsigned Depth, unsigned Signs>
__device__ __forceinline__ void accumulate_stream(const WaveRows<W>& rows, HeldGroup<Format> (&held)[Steps][W],
                                                  const signed char* lds_codes, const float* lds_deltas, unsigned count, unsigned width,
                                                  const unsigned long long* grid, const unsigned long long* masks, float (&acc)[W][MaxRows]) {
    const unsigned lane = threadIdx.x % 32U, groups = width / 32U;
    __builtin_assume(groups > 32U * (Steps - 1U));  // Steps = groups / 32 rounded up: only the last step can be partial
#pragma unroll
    for (unsigned w = 0; w < W; ++w)
#pragma unroll
        for (unsigned r = 0; r < MaxRows; ++r) acc[w][r] = 0.0F;
    if constexpr (Depth == 1U) {
        // Rolled: the next step's loads (the last one clamped, unused) are issued unconditionally before the
        // current step is decoded; the register footprint of accumulate_rows.
        HeldGroup<Format> current[W];
#pragma unroll
        for (unsigned w = 0; w < W; ++w) current[w] = held[0][w];
#pragma unroll 1
        for (unsigned i = 0; i < Steps; ++i) {
            HeldGroup<Format> next[W];
            load_step<Format, W>(rows, min(lane + 32U * (i + 1U), groups - 1U), next);
            asm volatile("" ::: "memory");
            const unsigned g = lane + 32U * i;
            if (g < groups) accumulate_group<Format, MaxRows, W, Signs>(current, g, lds_codes, lds_deltas, count, width, grid, masks, acc);
#pragma unroll
            for (unsigned w = 0; w < W; ++w) current[w] = next[w];
        }
    } else {
#pragma unroll
    for (unsigned i = 0; i < Steps; ++i) {
        if (i + Depth < Steps) {
            load_step<Format, W>(rows, min(lane + 32U * (i + Depth), groups - 1U), held[i + Depth]);
            asm volatile("" ::: "memory");  // keeps the loads here (not sunk into the partial last step's branch)
        }
        const unsigned g = lane + 32U * i;
        if (g < groups) accumulate_group<Format, MaxRows, W, Signs>(held[i], g, lds_codes, lds_deltas, count, width, grid, masks, acc);
        __builtin_amdgcn_sched_barrier(0);  // one step's decode at a time
    }
    }
#pragma unroll
    for (unsigned w = 0; w < W; ++w)
#pragma unroll
        for (unsigned r = 0; r < MaxRows; ++r)
            if (r < count) acc[w][r] = butterfly_sum(acc[w][r]);
}
// A block barrier for LDS data only: unlike __syncthreads (a workgroup release fence, which waits for every
// outstanding global load), the held weight loads stay in flight across it.
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
// stage_rows with lds_barrier (the later parts of an expert).
__device__ __forceinline__ void stage_rows_lds(const RoutesView& routes, const EntryRange& range, const signed char* __restrict__ codes,
                                               const float* __restrict__ deltas, unsigned width, signed char* __restrict__ lds_codes,
                                               float* __restrict__ lds_deltas) {
    const unsigned words = width / 16U, groups = width / 32U;
    for (unsigned e = 0; e < range.count; ++e) {
        const unsigned long long input = entry_input(routes, range.begin + e);
        const auto* source = reinterpret_cast<const int4*>(codes + input * width);
        auto* target = reinterpret_cast<int4*>(lds_codes + static_cast<unsigned long long>(e) * width);
        for (unsigned i = threadIdx.x; i < words; i += blockDim.x) target[i] = source[i];
        for (unsigned i = threadIdx.x; i < groups; i += blockDim.x)
            lds_deltas[e * groups + i] = deltas[input * groups + i];
    }
    lds_barrier();
}
// stage_rows split around the weight loads: the loads of the first part into registers (K int4 per thread and
// entry), then their LDS stores and the barrier. Threads: the block size; K * Threads covers width / 16.
using Words4 = int __attribute__((ext_vector_type(4)));
template <unsigned MaxRows, unsigned K> struct StageRegs { Words4 codes[MaxRows][K]; float deltas[MaxRows]; };
template <unsigned MaxRows, unsigned K, unsigned Threads>
__device__ __forceinline__ void stage_issue(const RoutesView& routes, const EntryRange& part, const signed char* __restrict__ codes,
                                            const float* __restrict__ deltas, unsigned width, StageRegs<MaxRows, K>& s) {
    const unsigned words = width / 16U, groups = width / 32U;
#pragma unroll
    for (unsigned e = 0; e < MaxRows; ++e)
        if (e < part.count) {
            const unsigned long long input = entry_input(routes, part.begin + e);
            const auto* source = reinterpret_cast<const Words4*>(codes + input * width);
#pragma unroll
            for (unsigned k = 0; k < K; ++k)
                if (threadIdx.x + k * Threads < words) s.codes[e][k] = source[threadIdx.x + k * Threads];
            if (threadIdx.x < groups) s.deltas[e] = deltas[input * groups + threadIdx.x];
        }
}
template <unsigned MaxRows, unsigned K, unsigned Threads>
__device__ __forceinline__ void stage_store(const EntryRange& part, unsigned width, const StageRegs<MaxRows, K>& s,
                                            signed char* __restrict__ lds_codes, float* __restrict__ lds_deltas) {
    const unsigned words = width / 16U, groups = width / 32U;
#pragma unroll
    for (unsigned e = 0; e < MaxRows; ++e)
        if (e < part.count) {
            auto* target = reinterpret_cast<Words4*>(lds_codes + static_cast<unsigned long long>(e) * width);
#pragma unroll
            for (unsigned k = 0; k < K; ++k)
                if (threadIdx.x + k * Threads < words) target[threadIdx.x + k * Threads] = s.codes[e][k];
            if (threadIdx.x < groups) lds_deltas[e * groups + threadIdx.x] = s.deltas[e];
        }
    lds_barrier();
}
__device__ __forceinline__ unsigned union_count(const RoutesView& routes) { return routes.counts ? routes.counts[0] : routes.expert_count; }

// routes_product_slots_kernel with held weight groups (W outputs per wave, Waves waves; gridDim.y union slots).
// Depth: steps loaded ahead (Steps: every step, kept for the later parts of an expert; else later parts reload).
template <WeightFormat Format, unsigned MaxRows, unsigned W, unsigned Waves, unsigned Steps, unsigned Depth, unsigned Signs>
__global__ __launch_bounds__(32U * Waves) void routes_product_rows_kernel(RoutesProductView view) {
    constexpr unsigned threads = 32U * Waves, k_words = (64U * Steps + threads - 1U) / threads;
    extern __shared__ int4 staged[];
    __shared__ unsigned long long grid[grid_words<Format>];
    __shared__ unsigned long long masks[Format == WeightFormat::iq2_xxs && Signs == 1U ? 128U : 1U];
    if constexpr (Format == WeightFormat::iq2_xxs) {
        load_grid(grid);
        if constexpr (Signs == 1U) load_sign_masks(masks);
    }
    const unsigned slots = union_count(view.routes);
    auto* lds_codes = reinterpret_cast<signed char*>(staged);
    auto* lds_deltas = reinterpret_cast<float*>(lds_codes + static_cast<unsigned long long>(MaxRows) * view.width);
    const unsigned wave = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / 32U)));
    const unsigned first = (blockIdx.x * Waves + wave) * W;
    const bool active = first < view.outputs;
    for (unsigned j = blockIdx.y; j < slots; j += gridDim.y) {
        const EntryRange range{view.routes.begins[j], view.routes.begins[j + 1U] - view.routes.begins[j], view.routes.experts[j]};
        const EntryRange part0{range.begin, min(MaxRows, range.count), range.expert};
        StageRegs<MaxRows, k_words> s;
        stage_issue<MaxRows, k_words, threads>(view.routes, part0, view.codes, view.deltas, view.width, s);
        WaveRows<W> rows;
        rows.rsrc[0] = rows.rsrc[1] = weight_rsrc(static_cast<const unsigned char*>(view.weights) + range.expert * view.expert_stride_bytes);
        rows.split = W;
#pragma unroll
        for (unsigned w = 0; w < W; ++w) rows.offset[w] = static_cast<int>(min(first + w, view.outputs - 1U) * view.row_bytes);
        HeldGroup<Format> held[Steps][W];
        load_held<Format, W, Steps, Depth>(rows, view.width, held);
        if (j != blockIdx.y) lds_barrier();
        stage_store<MaxRows, k_words, threads>(part0, view.width, s, lds_codes, lds_deltas);
        const auto part_product = [&](const EntryRange& part, HeldGroup<Format> (&h)[Steps][W]) {
            if (!active) return;
            float acc[W][MaxRows];
            accumulate_stream<Format, MaxRows, W, Steps, Depth, Signs>(rows, h, lds_codes, lds_deltas, part.count, view.width, grid, masks, acc);
            if (threadIdx.x % 32U != 0U) return;
#pragma unroll
            for (unsigned r = 0; r < MaxRows; ++r)
                if (r < part.count) {
                    const unsigned long long output = view.routes.outputs[part.begin + r];
#pragma unroll
                    for (unsigned w = 0; w < W; ++w)
                        if (first + w < view.outputs) view.out[output * view.outputs + first + w] = acc[w][r];
                }
        };
        part_product(part0, held);
        for (unsigned chunk = MaxRows; chunk < range.count; chunk += MaxRows) {
            const EntryRange part{range.begin + chunk, min(MaxRows, range.count - chunk), range.expert};
            lds_barrier();
            stage_rows_lds(view.routes, part, view.codes, view.deltas, view.width, lds_codes, lds_deltas);
            if constexpr (Depth >= Steps) {
                part_product(part, held);
            } else {
                HeldGroup<Format> again[Steps][W];
                load_held<Format, W, Steps, Depth>(rows, view.width, again);
                part_product(part, again);
            }
        }
    }
}
// gate_up_slots_kernel with held weight groups (32 / Waves outputs of gate and of up per wave; Depth as above).
template <WeightFormat Format, unsigned MaxRows, unsigned Waves, unsigned Steps, unsigned Depth, unsigned Signs>
__global__ __launch_bounds__(32U * Waves) void gate_up_rows_kernel(GateUpView view) {
    constexpr unsigned outs = 32U / Waves, W = 2U * outs, threads = 32U * Waves, k_words = (64U * Steps + threads - 1U) / threads;
    extern __shared__ int4 staged[];
    __shared__ unsigned long long grid[grid_words<Format>];
    __shared__ unsigned long long masks[Format == WeightFormat::iq2_xxs && Signs == 1U ? 128U : 1U];
    __shared__ float tile[2][MaxRows][32];
    if constexpr (Format == WeightFormat::iq2_xxs) {
        load_grid(grid);
        if constexpr (Signs == 1U) load_sign_masks(masks);
    }
    const unsigned slots = union_count(view.routes);
    auto* lds_codes = reinterpret_cast<signed char*>(staged);
    auto* lds_deltas = reinterpret_cast<float*>(lds_codes + static_cast<unsigned long long>(MaxRows) * view.hidden);
    const unsigned wave = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / 32U)));
    const unsigned lane = threadIdx.x % 32U, first = blockIdx.x * 32U + wave * outs;
    for (unsigned j = blockIdx.y; j < slots; j += gridDim.y) {
        const EntryRange range{view.routes.begins[j], view.routes.begins[j + 1U] - view.routes.begins[j], view.routes.experts[j]};
        const EntryRange part0{range.begin, min(MaxRows, range.count), range.expert};
        StageRegs<MaxRows, k_words> s;
        stage_issue<MaxRows, k_words, threads>(view.routes, part0, view.codes, view.deltas, view.hidden, s);
        const unsigned long long expert_base = range.expert * view.expert_stride_bytes;
        WaveRows<W> rows;
        rows.rsrc[0] = weight_rsrc(static_cast<const unsigned char*>(view.gate) + expert_base);
        rows.rsrc[1] = weight_rsrc(static_cast<const unsigned char*>(view.up) + expert_base);
        rows.split = outs;
#pragma unroll
        for (unsigned w = 0; w < outs; ++w) rows.offset[w] = rows.offset[w + outs] = static_cast<int>((first + w) * view.row_bytes);
        HeldGroup<Format> held[Steps][W];
        load_held<Format, W, Steps, Depth>(rows, view.hidden, held);
        if (j != blockIdx.y) lds_barrier();
        stage_store<MaxRows, k_words, threads>(part0, view.hidden, s, lds_codes, lds_deltas);
        const auto part_gate_up = [&](const EntryRange& part, HeldGroup<Format> (&h)[Steps][W]) {
            float acc[W][MaxRows];
            accumulate_stream<Format, MaxRows, W, Steps, Depth, Signs>(rows, h, lds_codes, lds_deltas, part.count, view.hidden, grid, masks, acc);
            if (lane == 0U)
#pragma unroll
                for (unsigned r = 0; r < MaxRows; ++r)
                    if (r < part.count)
#pragma unroll
                        for (unsigned w = 0; w < outs; ++w) {
                            tile[0][r][wave * outs + w] = acc[w][r];
                            tile[1][r][wave * outs + w] = acc[w + outs][r];
                        }
            lds_barrier();
            for (unsigned e = wave; e < part.count; e += Waves) {
                const unsigned long long slot = view.routes.outputs[part.begin + e];
                const unsigned at = blockIdx.x * 32U + lane, groups = view.inter / 32U;
                const float g = tile[0][e][lane], u = tile[1][e][lane];
                if (view.gate_out) { view.gate_out[slot * view.inter + at] = g; view.up_out[slot * view.inter + at] = u; }
                const float act = swiglu_clamped(g, u, view.limit);
                const float amax = butterfly_max(fabsf(act));
                const bool tiny = amax > 0.0F && amax < 0x1p-119F;
                const float delta = amax == 0.0F || tiny ? 1.0F : amax / 127.0F;
                const float rounded = tiny ? 0.0F : rintf(act / delta);
                const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
                const int sum = butterfly_isum(code);
                view.act_codes[slot * view.inter + at] = static_cast<signed char>(code);
                if (lane == 0U) { view.act_deltas[slot * groups + blockIdx.x] = delta; view.act_sums[slot * groups + blockIdx.x] = sum; }
            }
        };
        part_gate_up(part0, held);
        for (unsigned chunk = MaxRows; chunk < range.count; chunk += MaxRows) {
            const EntryRange part{range.begin + chunk, min(MaxRows, range.count - chunk), range.expert};
            lds_barrier();
            stage_rows_lds(view.routes, part, view.codes, view.deltas, view.hidden, lds_codes, lds_deltas);
            if constexpr (Depth >= Steps) {
                part_gate_up(part, held);
            } else {
                HeldGroup<Format> again[Steps][W];
                load_held<Format, W, Steps, Depth>(rows, view.hidden, again);
                part_gate_up(part, again);
            }
        }
    }
}

// Window routes from the router's ids, in one block (rows * top_k <= 64): the CSR over the union of the
// experts in [first, end) in ascending id with entries in ascending row, every row's slots in ascending id
// with its low-half count, counts[0] = union experts and counts[1] = the most entries of one expert.
// Equal to build_window_routes / restrict_routes.
__device__ __forceinline__ void build_routes_body(const BuildRoutesView& view) {
    __shared__ unsigned id[64];
    __shared__ unsigned leader[64];
    const unsigned t = threadIdx.x, n = view.rows * view.top_k;
    if (t < n) id[t] = view.ids[t];
    __syncthreads();
    const bool mine = t < n && id[t] >= view.first && id[t] < view.end;
    unsigned before = 0, below = 0;
    if (t < n) {
        for (unsigned s = 0; s < n; ++s) {
            const bool owned = id[s] >= view.first && id[s] < view.end;
            if (id[s] == id[t]) before += s < t ? 1U : 0U;
            if (owned && id[s] < id[t]) below += 1U;
        }
    }
    if (t < 64U) leader[t] = mine && before == 0U ? 1U : 0U;
    __syncthreads();
    if (mine) {
        unsigned position = 0;
        for (unsigned s = 0; s < n; ++s) position += leader[s] && id[s] < id[t] ? 1U : 0U;
        const unsigned entry = below + before;
        view.rows_out[entry] = t / view.top_k;
        view.slots[entry] = t;
        if (before == 0U) { view.experts[position] = id[t] - view.base; view.begins[position] = below; }
    }
    if (t < n) {
        const unsigned row = t / view.top_k;
        unsigned rank = 0, low = 0;
        for (unsigned k = 0; k < view.top_k; ++k) {
            const unsigned other = id[row * view.top_k + k];
            rank += other < id[t] ? 1U : 0U;
            low += other < view.split ? 1U : 0U;
        }
        view.order[row * view.top_k + rank] = t;
        if (t % view.top_k == 0U) view.low_counts[row] = low;
    }
    if (t == 0U) {
        unsigned experts = 0, entries = 0, most = 0;
        for (unsigned s = 0; s < n; ++s) {
            const bool owned = id[s] >= view.first && id[s] < view.end;
            experts += leader[s];
            entries += owned ? 1U : 0U;
            if (leader[s]) {
                unsigned count = 0;
                for (unsigned u = 0; u < n; ++u) count += id[u] == id[s] ? 1U : 0U;
                most = count > most ? count : most;
            }
        }
        view.begins[experts] = entries;
        view.counts[0] = experts;
        view.counts[1] = most;
    }
}

__global__ __launch_bounds__(64) void build_routes_kernel(BuildRoutesView view) { build_routes_body(view); }

__device__ __forceinline__ void combine_body(const float* __restrict__ raw, const float* __restrict__ shared,
                                             const float* __restrict__ weights, const std::uint32_t* __restrict__ order,
                                             const std::uint32_t* __restrict__ low_counts, unsigned top_k, unsigned hidden,
                                             float* __restrict__ out) {
    const unsigned o = blockIdx.x * 256U + threadIdx.x, r = blockIdx.y;
    if (o >= hidden) return;
    float low = 0.0F, high = 0.0F;
    const unsigned split = low_counts[r];
    for (unsigned k = 0; k < top_k; ++k) {
        const unsigned long long slot = order[r * top_k + k];
        const float term = weights[slot] * raw[slot * hidden + o];
        if (k < split) low = low + term;
        else high = high + term;
    }
    const unsigned long long at = static_cast<unsigned long long>(r) * hidden + o;
    out[at] = (shared[at] + low) + high;
}
__global__ __launch_bounds__(256) void combine_kernel(CombineView view) {
    combine_body(view.raw, view.shared, view.weights, view.order, view.low_counts, view.top_k, view.hidden, view.out);
}


// Router scores: one wave per expert, every window row (count <= 8); lanes take elements l, l + 32,
// ... in ascending order (acc = acc + w * x), then the butterfly, then router_score.
// The block (4 waves, 4 experts) stages the window rows of x in LDS a chunk at a time and every lane
// loads its weights of the chunk ahead of the products; the order of each lane's sum is unchanged.
constexpr unsigned router_waves = 4U, router_chunk = 512U;
// Router weight i of a row: F32, or the BF16 copy widened (exact for BF16-valued routers).
template <class W> __device__ __forceinline__ float router_weight(const W* w, unsigned i) {
    if constexpr (sizeof(W) == 2) return __uint_as_float(static_cast<unsigned>(w[i]) << 16U);
    else return w[i];
}
template <unsigned MaxRows, class W = float>
__device__ __forceinline__ void router_scores_body(const float* __restrict__ x, const W* __restrict__ weight,
                                                   float* __restrict__ scores, unsigned rows, unsigned hidden, unsigned experts) {
    constexpr unsigned per_lane = router_chunk / 32U;
    __shared__ float xs[MaxRows][router_chunk];
    const unsigned lane = threadIdx.x % 32U, expert = blockIdx.x * router_waves + threadIdx.x / 32U;
    const W* w = weight + static_cast<unsigned long long>(min(expert, experts - 1U)) * hidden;
    float acc[MaxRows];
#pragma unroll
    for (unsigned r = 0; r < MaxRows; ++r) acc[r] = 0.0F;
    float wv[per_lane];
#pragma unroll
    for (unsigned j = 0; j < per_lane; ++j) wv[j] = lane + 32U * j < hidden ? router_weight(w, lane + 32U * j) : 0.0F;
    for (unsigned c0 = 0; c0 < hidden; c0 += router_chunk) {
        const unsigned n = min(router_chunk, hidden - c0);
        if (c0) __syncthreads();
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned i = threadIdx.x; i < n; i += 32U * router_waves)
                xs[r][i] = x[static_cast<unsigned long long>(r) * hidden + c0 + i];
        __syncthreads();
        // The next chunk's weights are in flight while this chunk's products are formed.
        float next[per_lane];
        const unsigned c1 = c0 + router_chunk;
#pragma unroll
        for (unsigned j = 0; j < per_lane; ++j) next[j] = c1 + lane + 32U * j < hidden ? router_weight(w, c1 + lane + 32U * j) : 0.0F;
#pragma unroll
        for (unsigned j = 0; j < per_lane; ++j)
            if (lane + 32U * j < n)
#pragma unroll
                for (unsigned r = 0; r < MaxRows; ++r)
                    if (r < rows) acc[r] = acc[r] + wv[j] * xs[r][lane + 32U * j];
#pragma unroll
        for (unsigned j = 0; j < per_lane; ++j) wv[j] = next[j];
    }
    if (expert >= experts) return;
#pragma unroll
    for (unsigned r = 0; r < MaxRows; ++r)
        if (r < rows) {
#pragma unroll
            for (unsigned d = 16U; d; d /= 2U) acc[r] = acc[r] + __shfl_xor(acc[r], static_cast<int>(d), 32);
            if (lane == 0U) scores[static_cast<unsigned long long>(r) * experts + expert] = router_score(acc[r]);
        }
}
template <unsigned MaxRows>
__global__ __launch_bounds__(32U * router_waves) void router_scores_kernel(RouterView view) {
    if (view.weight_bf16) router_scores_body<MaxRows>(view.x, view.weight_bf16, view.scores, view.rows, view.hidden, view.experts);
    else router_scores_body<MaxRows>(view.x, view.weight, view.scores, view.rows, view.hidden, view.experts);
}
// The previous form (RouterView::staged = false, kept for A/B): 8 waves per block, x read from global
// memory inside the lane loop. Same order, same bits.
__global__ __launch_bounds__(256) void router_scores_wave_kernel(RouterView view) {
    const unsigned lane = threadIdx.x % 32U, expert = blockIdx.x * 8U + threadIdx.x / 32U;
    if (expert >= view.experts) return;
    const unsigned long long row = static_cast<unsigned long long>(expert) * view.hidden;
    float acc[product_max_rows];
#pragma unroll
    for (unsigned r = 0; r < product_max_rows; ++r) acc[r] = 0.0F;
    for (unsigned i = lane; i < view.hidden; i += 32U) {
        const float wi = view.weight_bf16 ? router_weight(view.weight_bf16 + row, i) : view.weight[row + i];
#pragma unroll
        for (unsigned r = 0; r < product_max_rows; ++r)
            if (r < view.rows) acc[r] = acc[r] + wi * view.x[static_cast<unsigned long long>(r) * view.hidden + i];
    }
#pragma unroll
    for (unsigned r = 0; r < product_max_rows; ++r)
        if (r < view.rows) {
#pragma unroll
            for (unsigned d = 16U; d; d /= 2U) acc[r] = acc[r] + __shfl_xor(acc[r], static_cast<int>(d), 32);
            if (lane == 0U) view.scores[static_cast<unsigned long long>(r) * view.experts + expert] = router_score(acc[r]);
        }
}

// Router scores, one wave per (expert, row): rows are independent, so a row's sum is the same chain of products as in
// the forms above (lane l: elements l, l + 32, ... ascending, acc = acc + w * x), then the same butterfly. The waves
// of one expert are adjacent (its weight row is read once from DRAM, then from the caches). Each lane keeps the next
// 16 products' operands in flight; hidden % 512 == 0.
constexpr unsigned router_row_waves = 8U, router_row_ahead = 16U;
template <class W>
__device__ __forceinline__ void router_row_body(const float* __restrict__ x, const W* __restrict__ weight, float* __restrict__ scores,
                                                unsigned rows, unsigned hidden, unsigned experts) {
    constexpr unsigned U = router_row_ahead, step = 32U * U;
    const unsigned lane = threadIdx.x % 32U, wave = blockIdx.x * router_row_waves + threadIdx.x / 32U;
    if (wave >= experts * rows) return;
    const unsigned expert = wave / rows, r = wave % rows;
    const W* w = weight + static_cast<unsigned long long>(expert) * hidden + lane;
    const float* xr = x + static_cast<unsigned long long>(r) * hidden + lane;
    float wv[U], xv[U];
#pragma unroll
    for (unsigned k = 0; k < U; ++k) { wv[k] = router_weight(w, 32U * k); xv[k] = xr[32U * k]; }
    float acc = 0.0F;
    for (unsigned c0 = 0; c0 < hidden; c0 += step) {
        const unsigned c1 = min(c0 + step, hidden - step);  // the last chunk loads itself again (unused)
        float wn[U], xn[U];
#pragma unroll
        for (unsigned k = 0; k < U; ++k) { wn[k] = router_weight(w, c1 + 32U * k); xn[k] = xr[c1 + 32U * k]; }
#pragma unroll
        for (unsigned k = 0; k < U; ++k) acc = acc + wv[k] * xv[k];
#pragma unroll
        for (unsigned k = 0; k < U; ++k) { wv[k] = wn[k]; xv[k] = xn[k]; }
    }
#pragma unroll
    for (unsigned d = 16U; d; d /= 2U) acc = acc + __shfl_xor(acc, static_cast<int>(d), 32);
    if (lane == 0U) scores[static_cast<unsigned long long>(r) * experts + expert] = router_score(acc);
}
__global__ __launch_bounds__(32U * router_row_waves) void router_row_kernel(RouterView view) {
    if (view.weight_bf16) router_row_body(view.x, view.weight_bf16, view.scores, view.rows, view.hidden, view.experts);
    else router_row_body(view.x, view.weight, view.scores, view.rows, view.hidden, view.experts);
}

// Top-k selection of one row per block: repeated arg-max of score + bias (larger value, then lower
// id), then the normalized and scaled weights computed by one thread in selection order.
struct BlockRoutePolicy {
    static constexpr unsigned threads = 256, max_top_k = 16;
    const RouterView& view;
    const float* scores;
    const float* biased;
    std::uint32_t* ids;
    float* weights;
    __device__ unsigned get_experts() const { return view.experts; }
    __device__ unsigned get_top_k() const { return view.top_k; }
    __device__ float score(unsigned e) const { return scores[e]; }
    __device__ void seed(unsigned t, unsigned k, const unsigned* chosen, float& value, unsigned& out_index) const {
        float v = 0.0F;
        unsigned index = view.experts;
        for (unsigned e = t; e < view.experts; e += 256U) {
            bool taken = false;
            for (unsigned j = 0; j < k; ++j) taken = taken || chosen[j] == e;
            if (taken) continue;
            if (index == view.experts || biased[e] > v) { v = biased[e]; index = e; }
        }
        value = v; out_index = index;
    }
    __device__ void write(unsigned r, unsigned k, unsigned id, float denominator) const {
        const float s = scores[id];
        ids[r * view.top_k + k] = id;
        weights[r * view.top_k + k] = (view.top_k > 1U ? s / denominator : s) * view.route_scale;
    }
};

__device__ __forceinline__ void router_select_body(const RouterView& view, const float* __restrict__ scores,
                                                   const float* __restrict__ bias, std::uint32_t* __restrict__ ids,
                                                   float* __restrict__ weights) {
    __shared__ float biased[1024];
    const unsigned r = blockIdx.x, t = threadIdx.x;
    for (unsigned e = t; e < view.experts; e += 256U) biased[e] = scores[e] + bias[e];
    __syncthreads();
    strix::moe_core::route_block_topk(BlockRoutePolicy{view, scores, biased, ids, weights}, r, t);
}
__global__ __launch_bounds__(256) void router_select_kernel(RouterView view) {
    const unsigned r = blockIdx.x;
    router_select_body(view, view.scores + static_cast<unsigned long long>(r) * view.experts,
                       view.image_rows && view.image_rows[r] ? view.bias_image : view.bias, view.ids, view.weights);
}

// ---- STRIX_DS41_ROUTER_FAST=1 (default off): wide scores, then a one-wave selection ----------------------------------
// Scores (BF16 copy, hidden % 256 == 0): one wave per expert over every window row. The wave reads its weight row 512
// contiguous bytes per 256-element chunk (lane l: elements 8 l .. 8 l + 7), router_fast_depth chunks in flight, passes
// each chunk through a wave-private LDS tile, and lane l takes the chunk's elements l + 32 m (m = 0..7) in order: its
// chain is the one of the forms above (elements l, l + 32, ... ascending, acc = acc + w * x), then the same butterfly
// and router_score. The block stages the rows of x in LDS (whole rows when they fit, else router_fast_x / R elements).
constexpr unsigned router_fast_depth = 4U, router_fast_x = 6144U;

// The selection of router_select_body done by one wave, the same comparisons in the same tree: lane l stands for the
// block's threads l + 32 i (i = 0..7), virtual thread t scans experts t, t + 256, ... in that order, the reduction
// pairs t with t + stride for strides 128, 64, 32 (inside the lane) and 16 .. 1 (across lanes), so the chosen ids are
// the block's for every input (NaN included); the weight sum is taken in selection order as there.
__device__ __forceinline__ bool router_better(float ov, unsigned oi, float bv, unsigned bi, unsigned experts) {
    return oi != experts && (bi == experts || ov > bv || (ov == bv && oi < bi));
}
__device__ void router_select_wave(const RouterView& view, unsigned r, unsigned lane) {
    const unsigned experts = view.experts, top_k = view.top_k;
    const float* scores = view.scores + static_cast<unsigned long long>(r) * experts;
    const float* bias = view.image_rows && view.image_rows[r] ? view.bias_image : view.bias;
    float b[32];  // expert lane + 32 q, biased
#pragma unroll
    for (unsigned q = 0; q < 32U; ++q) {
        const unsigned e = lane + 32U * q;
        b[q] = e < experts ? scores[e] + bias[e] : 0.0F;
    }
    unsigned taken = 0U, mine = experts;  // taken bit q: expert lane + 32 q chosen; lane k: the k-th chosen id
    for (unsigned k = 0; k < top_k; ++k) {
        float v[8];
        unsigned index[8];
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) {  // virtual thread lane + 32 i: experts lane + 32 (i + 8 j), j ascending
            v[i] = 0.0F;
            index[i] = experts;
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) {
                const unsigned q = i + 8U * j, e = lane + 32U * q;
                if (e < experts && ((taken >> q) & 1U) == 0U && (index[i] == experts || b[q] > v[i])) { v[i] = b[q]; index[i] = e; }
            }
        }
#pragma unroll
        for (unsigned s = 4U; s; s /= 2U)  // strides 128, 64, 32
#pragma unroll
            for (unsigned i = 0; i < s; ++i)
                if (router_better(v[i + s], index[i + s], v[i], index[i], experts)) { v[i] = v[i + s]; index[i] = index[i + s]; }
        float bv = v[0];
        unsigned bi = index[0];
#pragma unroll
        for (unsigned d = 16U; d; d /= 2U) {  // strides 16 .. 1
            const float ov = __shfl(bv, static_cast<int>(lane + d), 32);
            const unsigned oi = static_cast<unsigned>(__shfl(static_cast<int>(bi), static_cast<int>(lane + d), 32));
            if (lane < d && router_better(ov, oi, bv, bi, experts)) { bv = ov; bi = oi; }
        }
        const unsigned c = static_cast<unsigned>(__shfl(static_cast<int>(bi), 0, 32));
        if (lane == c % 32U) taken |= 1U << (c / 32U);
        if (lane == k) mine = c;
    }
    const float s = lane < top_k ? scores[mine] : 0.0F;
    float sum = 0.0F;
    for (unsigned k = 0; k < top_k; ++k) sum = sum + __shfl(s, static_cast<int>(k), 32);
    const float denominator = sum + 1e-20F;
    if (lane < top_k) {
        view.ids[r * top_k + lane] = mine;
        view.weights[r * top_k + lane] = (top_k > 1U ? s / denominator : s) * view.route_scale;
    }
}

template <unsigned R, unsigned Waves>
__global__ __launch_bounds__(32U * Waves) void router_fast_kernel(RouterView view) {
    __shared__ float xs[router_fast_x];
    __shared__ alignas(16) unsigned short tile[Waves][256];
    __shared__ unsigned last_s;
    const unsigned l = threadIdx.x % 32U, wv = threadIdx.x / 32U, rows = view.rows, hidden = view.hidden, chunks = hidden / 256U;
    const unsigned expert = blockIdx.x * Waves + wv, ec = min(expert, view.experts - 1U);
    const unsigned span = rows * hidden <= router_fast_x ? hidden : router_fast_x / R;
    const std::uint16_t* w = view.weight_bf16 + static_cast<unsigned long long>(ec) * hidden + 8U * l;
    float acc[R];
#pragma unroll
    for (unsigned r = 0; r < R; ++r) acc[r] = 0.0F;
    uint4 buf[router_fast_depth];
#pragma unroll
    for (unsigned d = 0; d < router_fast_depth; ++d)
        buf[d] = d < chunks ? *reinterpret_cast<const uint4*>(w + 256U * d) : uint4{0, 0, 0, 0};
    for (unsigned k0 = 0; k0 < chunks; k0 += router_fast_depth) {
#pragma unroll
        for (unsigned d = 0; d < router_fast_depth; ++d) {
            const unsigned k = k0 + d;
            if (k < chunks) {
                const unsigned e0 = 256U * k;
                if (e0 % span == 0U) {  // stage elements e0 .. e0 + span - 1 of every row
                    __syncthreads();
                    const unsigned n = min(span, hidden - e0);
                    for (unsigned i = threadIdx.x; i < rows * n; i += 32U * Waves) {
                        const unsigned r = i / n, q = i % n;
                        xs[r * span + q] = view.x[static_cast<unsigned long long>(r) * hidden + e0 + q];
                    }
                    __syncthreads();
                }
                reinterpret_cast<uint4*>(tile[wv])[l] = buf[d];
                if (k + router_fast_depth < chunks) buf[d] = *reinterpret_cast<const uint4*>(w + 256U * (k + router_fast_depth));
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
                const unsigned xo = e0 % span;
#pragma unroll
                for (unsigned m = 0; m < 8U; ++m) {
                    const float wf = __uint_as_float(static_cast<unsigned>(tile[wv][32U * m + l]) << 16U);
#pragma unroll
                    for (unsigned r = 0; r < R; ++r)
                        if (r < rows) acc[r] = acc[r] + wf * xs[r * span + xo + 32U * m + l];
                }
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            }
        }
    }
#pragma unroll
    for (unsigned r = 0; r < R; ++r)
        if (r < rows) {
#pragma unroll
            for (unsigned d = 16U; d; d /= 2U) acc[r] = acc[r] + __shfl_xor(acc[r], static_cast<int>(d), 32);
            if (l == 0U && expert < view.experts) view.scores[static_cast<unsigned long long>(r) * view.experts + expert] = router_score(acc[r]);
        }
    if (view.tickets == nullptr) return;
    // the last block to finish selects every row (one wave per row); the ticket is left zero
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0U) last_s = atomicAdd(view.tickets, 1U) == gridDim.x - 1U ? 1U : 0U;
    __syncthreads();
    if (last_s == 0U) return;
    __threadfence();
    for (unsigned r = wv; r < rows; r += Waves) router_select_wave(view, r, l);
    if (threadIdx.x == 0U) *view.tickets = 0U;
}
// Fast form 1 (the default of STRIX_DS41_ROUTER_FAST=1): the row-wave scores above, then with a ticket the last block
// selects every row (a block of router_row_waves = product_max_rows waves: wave r takes row r).
__global__ __launch_bounds__(32U * router_row_waves) void router_row_select_kernel(RouterView view) {
    __shared__ unsigned last_s;
    if (view.weight_bf16) router_row_body(view.x, view.weight_bf16, view.scores, view.rows, view.hidden, view.experts);
    else router_row_body(view.x, view.weight, view.scores, view.rows, view.hidden, view.experts);
    if (view.tickets == nullptr) return;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0U) last_s = atomicAdd(view.tickets, 1U) == gridDim.x - 1U ? 1U : 0U;
    __syncthreads();
    if (last_s == 0U) return;
    __threadfence();
    if (threadIdx.x / 32U < view.rows) router_select_wave(view, threadIdx.x / 32U, threadIdx.x % 32U);
    if (threadIdx.x == 0U) *view.tickets = 0U;
}
static_assert(router_row_waves >= product_max_rows);
// The selection alone (no ticket): one block, one wave per row.
__global__ __launch_bounds__(32U * product_max_rows) void router_select_waves_kernel(RouterView view) {
    router_select_wave(view, threadIdx.x / 32U, threadIdx.x % 32U);
}
// All selected ids are visible before the integer route builder reads them. No global counter or
// change to the score/weight arithmetic; the CSR retains ascending expert and slot order. The builder is
// build_routes_fast_kernel's (the default builder): r1fuse took build_routes_kernel's, whose
// thread 0 reduces every entry against every entry alone (2,352 serial steps at 8 rows), which made the fused form
// slower than selection + fast builder at R6/N8.
__global__ __launch_bounds__(256) void router_select_routes_kernel(RouterView view, BuildRoutesView routes, bool fast) {
    const unsigned r = threadIdx.x / 32U;
    if (r < view.rows) router_select_wave(view, r, threadIdx.x % 32U);
    __syncthreads();
    if (fast) build_routes_fast_body(routes);
    else build_routes_body(routes);
}
void launch_router_row_select(const RouterView& v, hipStream_t s) {
    hipLaunchKernelGGL(router_row_select_kernel, dim3((v.experts * v.rows + router_row_waves - 1U) / router_row_waves),
                       dim3(32U * router_row_waves), 0, s, v);
    if (v.tickets == nullptr) hipLaunchKernelGGL(router_select_waves_kernel, dim3(1), dim3(32U * v.rows), 0, s, v);
}
template <unsigned Waves>
void launch_router_fast(const RouterView& v, hipStream_t s) {
    const dim3 grid((v.experts + Waves - 1U) / Waves), block(32U * Waves);
    if (v.rows <= 1U) hipLaunchKernelGGL(HIP_KERNEL_NAME(router_fast_kernel<1, Waves>), grid, block, 0, s, v);
    else if (v.rows <= 2U) hipLaunchKernelGGL(HIP_KERNEL_NAME(router_fast_kernel<2, Waves>), grid, block, 0, s, v);
    else if (v.rows <= 4U) hipLaunchKernelGGL(HIP_KERNEL_NAME(router_fast_kernel<4, Waves>), grid, block, 0, s, v);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(router_fast_kernel<8, Waves>), grid, block, 0, s, v);
    if (v.tickets == nullptr) hipLaunchKernelGGL(router_select_waves_kernel, dim3(1), dim3(32U * v.rows), 0, s, v);
}

// One half of every row's weighted raw slot vectors, in ascending id from 0 (as in the combine).
__device__ __forceinline__ void half_sum_body(const float* __restrict__ raw, const float* __restrict__ weights,
                                              const std::uint32_t* __restrict__ order, const std::uint32_t* __restrict__ low_counts,
                                              unsigned top_k, unsigned hidden, unsigned half, float* __restrict__ out) {
    const unsigned o = blockIdx.x * 256U + threadIdx.x, r = blockIdx.y;
    if (o >= hidden) return;
    const unsigned low = low_counts[r];
    const unsigned first = half == 0U ? 0U : low, end = half == 0U ? low : top_k;
    float sum = 0.0F;
    for (unsigned k = first; k < end; ++k) {
        const unsigned long long slot = order[r * top_k + k];
        sum = sum + weights[slot] * raw[slot * hidden + o];
    }
    out[static_cast<unsigned long long>(r) * hidden + o] = sum;
}
__global__ __launch_bounds__(256) void half_sum_kernel(HalfSumView view) {
    half_sum_body(view.raw, view.weights, view.order, view.low_counts, view.top_k, view.hidden, view.half, view.out);
}
// Lanes past hidden skip the body but still take part in the release (padding included).
__global__ __launch_bounds__(256) void half_sum_ready_kernel(HalfSumView view, dist::ProducerStage stage) {
    if (blockIdx.x * 256U + threadIdx.x < view.hidden)
        half_sum_body(view.raw, view.weights, view.order, view.low_counts, view.top_k, view.hidden, view.half, view.out);
    dist::publish_staged_block_lean(stage.flag, stage.counter, stage.token, gridDim.x * gridDim.y);
}
__global__ __launch_bounds__(256) void canonical_sum_ready_kernel(const float* __restrict__ shared, const float* low, const float* high,
                                                                  unsigned long long count, float* __restrict__ out,
                                                                  dist::StagedReady* flag, unsigned* counter, std::uint64_t token) {
    dist::staged_acquire_lean();
    const unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * 256U + threadIdx.x;
    if (i < count) out[i] = (shared[i] + low[i]) + high[i];
    dist::publish_staged_reads_block_lean(flag, counter, token, gridDim.x);
}
__global__ __launch_bounds__(256) void canonical_sum_kernel(const float* __restrict__ shared, const float* __restrict__ low,
                                                            const float* __restrict__ high, unsigned long long count,
                                                            float* __restrict__ out) {
    const unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * 256U + threadIdx.x;
    if (i < count) out[i] = (shared[i] + low[i]) + high[i];
}

__global__ __launch_bounds__(256) void canonical_sum_split_kernel(const float* __restrict__ lo, const float* __restrict__ hi,
                                                                  const float* __restrict__ low, const float* __restrict__ high,
                                                                  unsigned hidden, float* __restrict__ out, float* __restrict__ shared_out) {
    const unsigned o = blockIdx.x * 256U + threadIdx.x, r = blockIdx.y, half = hidden / 2U;
    if (o >= hidden) return;
    const unsigned long long at = static_cast<unsigned long long>(r) * hidden + o;
    const float shared = o < half ? lo[static_cast<unsigned long long>(r) * half + o] : hi[static_cast<unsigned long long>(r) * half + o - half];
    out[at] = (shared + low[at]) + high[at];
    if (shared_out) shared_out[at] = shared;
}
__global__ __launch_bounds__(256) void assemble_a8_halves_kernel(const int* __restrict__ first_codes, const float* __restrict__ first_deltas,
                                                                 const int* __restrict__ second_codes, const float* __restrict__ second_deltas,
                                                                 unsigned half, int* __restrict__ codes, float* __restrict__ deltas) {
    const unsigned r = blockIdx.y, words = half / 4U, groups = half / 32U;
    for (unsigned i = blockIdx.x * 256U + threadIdx.x; i < 2U * words; i += gridDim.x * 256U) {
        const bool second = i >= words;
        const unsigned j = second ? i - words : i;
        codes[static_cast<unsigned long long>(r) * 2U * words + i] = (second ? second_codes : first_codes)[static_cast<unsigned long long>(r) * words + j];
    }
    for (unsigned i = blockIdx.x * 256U + threadIdx.x; i < 2U * groups; i += gridDim.x * 256U) {
        const bool second = i >= groups;
        const unsigned j = second ? i - groups : i;
        deltas[static_cast<unsigned long long>(r) * 2U * groups + i] = (second ? second_deltas : first_deltas)[static_cast<unsigned long long>(r) * groups + j];
    }
}

template <unsigned MaxRows>
std::size_t staged_bytes(unsigned width) { return static_cast<std::size_t>(MaxRows) * width + MaxRows * (width / 32U) * 4U; }

// STRIX_DS41_EXPERT_SLOTS: union slots covered by the grid at R rows (W1 routes: about the 99th percentile of one
// rank's union per layer; larger unions loop), 0 = the padded grid (rows * top_k slots).
unsigned expert_slots_y(unsigned rows, unsigned slots) {
    const auto& t = moe_kernel_tuning();
    if (t.slots <= 0) return 0U;
    static constexpr unsigned table[9] = {0U, 6U, 10U, 12U, 14U, 17U, 19U, 21U, 23U};
    const unsigned y = t.slots_y > 0 ? static_cast<unsigned>(t.slots_y) : table[std::min(rows, 8U)];
    return y < slots ? y : 0U;
}
template <WeightFormat Format, unsigned MaxRows, unsigned Waves, bool Early, bool Tail = false>
void launch_routes_product(const RoutesProductView& view, unsigned blocks_y, unsigned dense_rows, unsigned dense_experts,
                           hipStream_t stream) {
    constexpr unsigned W = 32U / Waves;
    if constexpr (!Early)
        if (view.routes.counts && view.routes.begins) {
            if (const unsigned y = expert_slots_y(view.routes.max_rows, blocks_y); y != 0U) {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(routes_product_slots_kernel<Format, MaxRows, W, Waves, Tail>),
                                   dim3((view.outputs + Waves * W - 1U) / (Waves * W), y), dim3(32U * Waves),
                                   staged_bytes<MaxRows>(view.width), stream, view);
                return;
            }
        }
    const dim3 grid((view.outputs + Waves * W - 1U) / (Waves * W), blocks_y);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(routes_product_kernel<Format, MaxRows, W, Waves, Early, Tail>), grid, dim3(32U * Waves),
                       staged_bytes<MaxRows>(view.width), stream, view, dense_rows, dense_experts);
}
// Kernel forms (the same bits in each: an entry's result never depends on the other entries, and
// experts with more entries than the staged rows are done in parts):
//   0: the original shapes;
//   1: early weight loads, 16 waves at 1-2 staged rows;  2: early weight loads only;
//   3, 4, 6: at most 4, 2 or 1 staged rows (smaller register tiles; an expert with more entries
//            re-reads its weights from the caches);  5: at most 2 staged rows, 16 waves.
struct FormShape { unsigned cap; bool wide; bool early; };
// Default forms (-1), from cold timings at the runner's shapes (layer 20, recorded routes, one rank):
// routed gate/up keeps the original shapes up to 4 rows and stages at most 4 rows in larger windows
// (R8 483 -> 394 us); the routed down stages at most 2 rows from 3 rows on (R8 398 -> 284 us, R4
// 169 -> 165 us); the Q8_0 shared expert's gate/up takes form 1 (R1 126 -> 121 us, R8 168 -> 153 us)
// and its down the original shapes.
inline int default_gate_up_form(WeightFormat format, unsigned rows) {
    if (format == WeightFormat::q8_0) return 1;
    return format == WeightFormat::iq2_xxs && rows > 4U ? 3 : 0;  // MXFP4 (draft experts) not measured: original
}
inline int default_product_form(WeightFormat format, unsigned rows) {
    if (format == WeightFormat::q8_0) return 0;
    return format == WeightFormat::q2_k && rows > 2U ? 4 : 0;
}
inline FormShape form_shape(int form) {
    switch (form) {
    case 1: return {8U, true, true};
    case 2: return {8U, false, true};
    case 3: return {4U, false, false};
    case 4: return {2U, false, false};
    case 5: return {2U, true, false};
    case 6: return {1U, false, false};
    default: return {8U, false, false};
    }
}
template <WeightFormat Format, unsigned MaxRows, bool Tail = false>
void launch_routes_product_form(const RoutesProductView& view, int form, unsigned blocks_y, unsigned dense_rows,
                                unsigned dense_experts, hipStream_t stream) {
    const auto f = form_shape(form);
    if constexpr (MaxRows <= 2U)
        if (f.wide) {
            if (f.early) launch_routes_product<Format, MaxRows, 16U, true, Tail>(view, blocks_y, dense_rows, dense_experts, stream);
            else launch_routes_product<Format, MaxRows, 16U, false, Tail>(view, blocks_y, dense_rows, dense_experts, stream);
            return;
        }
    if (f.early) launch_routes_product<Format, MaxRows, window_waves<MaxRows>, true, Tail>(view, blocks_y, dense_rows, dense_experts, stream);
    else launch_routes_product<Format, MaxRows, window_waves<MaxRows>, false, Tail>(view, blocks_y, dense_rows, dense_experts, stream);
}
template <WeightFormat Format, bool Tail = false>
void dispatch_routes_product(const RoutesProductView& view, int form, unsigned max_rows, unsigned blocks_y, unsigned dense_rows,
                             unsigned dense_experts, hipStream_t stream) {
    max_rows = std::min(max_rows, form_shape(form).cap);
    if (max_rows <= 1U) launch_routes_product_form<Format, 1, Tail>(view, form, blocks_y, dense_rows, dense_experts, stream);
    else if (max_rows <= 2U) launch_routes_product_form<Format, 2, Tail>(view, form, blocks_y, dense_rows, dense_experts, stream);
    else if (max_rows <= 4U) launch_routes_product_form<Format, 4, Tail>(view, form, blocks_y, dense_rows, dense_experts, stream);
    else launch_routes_product_form<Format, 8, Tail>(view, form, blocks_y, dense_rows, dense_experts, stream);
}
void routes_product(WeightFormat format, const RoutesProductView& view, int form, unsigned max_rows, unsigned blocks_y,
                    unsigned dense_rows, unsigned dense_experts, hipStream_t stream) {
    if (format == WeightFormat::iq2_xxs) dispatch_routes_product<WeightFormat::iq2_xxs>(view, form, max_rows, blocks_y, dense_rows, dense_experts, stream);
    else if (format == WeightFormat::q2_k) dispatch_routes_product<WeightFormat::q2_k>(view, form, max_rows, blocks_y, dense_rows, dense_experts, stream);
    else if (format == WeightFormat::q8_0) dispatch_routes_product<WeightFormat::q8_0>(view, form, max_rows, blocks_y, dense_rows, dense_experts, stream);
    else if (format == WeightFormat::cb4) dispatch_routes_product<WeightFormat::cb4>(view, form, max_rows, blocks_y, dense_rows, dense_experts, stream);
    else if (format == WeightFormat::u5) dispatch_routes_product<WeightFormat::u5>(view, form, max_rows, blocks_y, dense_rows, dense_experts, stream);
    else if (format == WeightFormat::u6) dispatch_routes_product<WeightFormat::u6>(view, form, max_rows, blocks_y, dense_rows, dense_experts, stream);
    else dispatch_routes_product<WeightFormat::mxfp4>(view, form, max_rows, blocks_y, dense_rows, dense_experts, stream);
}
template <WeightFormat Format, unsigned MaxRows, unsigned Waves, bool Early>
void launch_gate_up(const GateUpView& view, hipStream_t stream) {
    if constexpr (!Early)
        if (view.routes.counts) {
            if (const unsigned y = expert_slots_y(view.routes.max_rows, view.routes.expert_count); y != 0U) {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(gate_up_slots_kernel<Format, MaxRows, Waves>), dim3(view.inter / 32U, y),
                                   dim3(32U * Waves), staged_bytes<MaxRows>(view.hidden), stream, view);
                return;
            }
        }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gate_up_kernel<Format, MaxRows, Waves, Early>), dim3(view.inter / 32U, view.routes.expert_count),
                       dim3(32U * Waves), staged_bytes<MaxRows>(view.hidden), stream, view);
}
// Forms as for the products (form_shape). With 16 waves at 1-2 rows each wave owns 2 + 2 weight rows
// (with 8 waves, 4 + 4: the Q8_0 shared expert then needs more than 256 VGPRs and spills).
template <WeightFormat Format, unsigned MaxRows>
void launch_gate_up_form(const GateUpView& view, int form, hipStream_t stream) {
    const auto f = form_shape(form);
    if constexpr (MaxRows <= 2U)
        if (f.wide) {
            if (f.early) launch_gate_up<Format, MaxRows, 16U, true>(view, stream);
            else launch_gate_up<Format, MaxRows, 16U, false>(view, stream);
            return;
        }
    if (f.early) launch_gate_up<Format, MaxRows, window_waves<MaxRows>, true>(view, stream);
    else launch_gate_up<Format, MaxRows, window_waves<MaxRows>, false>(view, stream);
}
template <WeightFormat Format>
void dispatch_gate_up(const GateUpView& view, int form, hipStream_t stream) {
    const auto m = std::min(view.routes.max_rows, form_shape(form).cap);
    if (m <= 1U) launch_gate_up_form<Format, 1>(view, form, stream);
    else if (m <= 2U) launch_gate_up_form<Format, 2>(view, form, stream);
    else if (m <= 4U) launch_gate_up_form<Format, 4>(view, form, stream);
    else launch_gate_up_form<Format, 8>(view, form, stream);
}

// STRIX_DS41_EXPERT_ROWS shapes: staged rows per part and waves per block of the gate/up and of the down, the down's
// steps loaded ahead and the IQ2_XXS sign form. STRIX_DS41_EXPERT_ROWS_FORM=QGDABS overrides them: Q down steps
// ahead (1 or 3 = all, kept for the later parts), caps G, D (1, 2, 4 or 8), waves A, B (1: 8, 2: 16), signs S (1: LDS
// table, 2: v_perm). Only the model's widths (gate/up 5120 = 5 steps, down 2304 = 3 steps) have every shape; other
// widths take 8 waves, signs 2 and (down) every step ahead.
struct RowsShape { unsigned gate_cap = 4U, down_cap = 4U, gate_waves = 16U, down_waves = 8U, signs = 2U, down_depth = 1U; };
RowsShape rows_shape() {
    RowsShape s;
    const int f = moe_kernel_tuning().rows_form;
    if (f < 0) return s;
    const auto digit = [f](int place) { return static_cast<unsigned>(f / place % 10); };
    const auto cap = [](unsigned d, unsigned fallback) { return d == 1U || d == 2U || d == 4U || d == 8U ? d : fallback; };
    s.gate_cap = cap(digit(10000), s.gate_cap);
    s.down_cap = cap(digit(1000), s.down_cap);
    if (digit(100) != 0U) s.gate_waves = digit(100) == 2U ? 16U : 8U;
    if (digit(10) != 0U) s.down_waves = digit(10) == 2U ? 16U : 8U;
    if (digit(1) != 0U) s.signs = std::min(digit(1), 2U);
    if (digit(100000) != 0U) s.down_depth = digit(100000) >= 3U ? 3U : 1U;
    return s;
}
// Union slots of the grid: the STRIX_DS41_EXPERT_SLOTS bound for device-built routes, else every slot.
unsigned rows_grid_y(const RoutesView& routes) {
    if (routes.counts)
        if (const unsigned y = expert_slots_y(routes.max_rows, routes.expert_count); y != 0U) return y;
    return routes.expert_count;
}
template <unsigned MaxRows, unsigned Waves, unsigned Steps, unsigned Depth, unsigned Signs>
void launch_gate_up_rows(const GateUpView& view, hipStream_t stream) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(gate_up_rows_kernel<WeightFormat::iq2_xxs, MaxRows, Waves, Steps, Depth, Signs>),
                       dim3(view.inter / 32U, rows_grid_y(view.routes)), dim3(32U * Waves), staged_bytes<MaxRows>(view.hidden), stream, view);
}
template <unsigned MaxRows, unsigned Waves, unsigned Steps>
void gate_up_rows_signs(const GateUpView& view, const RowsShape& s, hipStream_t stream) {
    if (s.signs == 1U) launch_gate_up_rows<MaxRows, Waves, Steps, 1U, 1U>(view, stream);
    else launch_gate_up_rows<MaxRows, Waves, Steps, 1U, 2U>(view, stream);
}
template <unsigned MaxRows, unsigned Steps>
void gate_up_rows_shape(const GateUpView& view, const RowsShape& s, hipStream_t stream) {
    if constexpr (Steps == 5U) {
        if (s.gate_waves == 16U) gate_up_rows_signs<MaxRows, 16U, Steps>(view, s, stream);
        else gate_up_rows_signs<MaxRows, 8U, Steps>(view, s, stream);
        return;
    }
    launch_gate_up_rows<MaxRows, 8U, Steps, 1U, 2U>(view, stream);
}
template <unsigned Steps>
void gate_up_rows_steps(const GateUpView& view, const RowsShape& s, hipStream_t stream) {
    const auto m = std::min(view.routes.max_rows, s.gate_cap);
    if (m <= 1U) gate_up_rows_shape<1, Steps>(view, s, stream);
    else if (m <= 2U) gate_up_rows_shape<2, Steps>(view, s, stream);
    else if (m <= 4U) gate_up_rows_shape<4, Steps>(view, s, stream);
    else gate_up_rows_shape<8, Steps>(view, s, stream);
}
// The routed IQ2_XXS gate/up in the held-groups form; false (nothing launched) when the flag is off or the shape
// is not covered.
bool gate_up_rows(WeightFormat format, const GateUpView& view, hipStream_t stream) {
    if (moe_kernel_tuning().rows <= 0 || format != WeightFormat::iq2_xxs) return false;
    const auto s = rows_shape();
    switch ((view.hidden / 32U + 31U) / 32U) {
    case 1U: gate_up_rows_steps<1>(view, s, stream); return true;
    case 4U: gate_up_rows_steps<4>(view, s, stream); return true;
    case 5U: gate_up_rows_steps<5>(view, s, stream); return true;
    default: return false;
    }
}
template <unsigned MaxRows, unsigned Waves, unsigned Steps, unsigned Depth>
void launch_down_rows(const RoutesProductView& view, hipStream_t stream) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(routes_product_rows_kernel<WeightFormat::q2_k, MaxRows, 32U / Waves, Waves, Steps, Depth, 0U>),
                       dim3((view.outputs + 31U) / 32U, rows_grid_y(view.routes)), dim3(32U * Waves), staged_bytes<MaxRows>(view.width),
                       stream, view);
}
template <unsigned MaxRows, unsigned Steps>
void down_rows_shape(const RoutesProductView& view, const RowsShape& s, hipStream_t stream) {
    if constexpr (Steps == 3U) {
        if (s.down_waves == 16U) {
            if (s.down_depth >= Steps) launch_down_rows<MaxRows, 16U, Steps, Steps>(view, stream);
            else launch_down_rows<MaxRows, 16U, Steps, 1U>(view, stream);
        } else {
            if (s.down_depth >= Steps) launch_down_rows<MaxRows, 8U, Steps, Steps>(view, stream);
            else launch_down_rows<MaxRows, 8U, Steps, 1U>(view, stream);
        }
        return;
    }
    launch_down_rows<MaxRows, 8U, Steps, Steps>(view, stream);
}
template <unsigned Steps>
void down_rows_steps(const RoutesProductView& view, const RowsShape& s, hipStream_t stream) {
    const auto m = std::min(view.routes.max_rows, s.down_cap);
    if (m <= 1U) down_rows_shape<1, Steps>(view, s, stream);
    else if (m <= 2U) down_rows_shape<2, Steps>(view, s, stream);
    else if (m <= 4U) down_rows_shape<4, Steps>(view, s, stream);
    else down_rows_shape<8, Steps>(view, s, stream);
}
// The routed Q2_K down in the held-groups form (as gate_up_rows).
bool down_rows(WeightFormat format, const RoutesProductView& view, hipStream_t stream) {
    if (moe_kernel_tuning().rows <= 0 || format != WeightFormat::q2_k) return false;
    const auto s = rows_shape();
    switch ((view.width / 32U + 31U) / 32U) {
    case 1U: down_rows_steps<1>(view, s, stream); return true;
    case 2U: down_rows_steps<2>(view, s, stream); return true;
    case 3U: down_rows_steps<3>(view, s, stream); return true;
    default: return false;
    }
}

// STRIX_DS41_TP_DOWN=1: the expert-TP down (ds41-expert-tp2-i1152-v1: Q2_K rows of width 1152 = 36 groups, 4 whole
// blocks then the rank's half block, packed (44 bytes) or, Packed false, the first half of block 4 of a 420-byte row)
// with held groups. Lane l holds group l of each of its wave's W rows (the whole blocks) and ONE tail group: row l / 4,
// group 32 + l % 4 (W <= 8 rows: 4 W lanes). The tail step thus decodes one group per lane, where accumulate_rows
// decodes W groups on lanes 0..3 while the other lanes idle. A tail term is computed exactly as accumulate_rows
// computes it, moved to lane l % 4 by a lane permute and added there after the lane's whole-block term, so every
// (entry, output) keeps its lane classes (lane l: 0 + term(l), then + term(l + 32) for l < 4), its terms and its
// butterfly: the bits of routes_product_kernel<q2_k, ..., Tail>. Every group is loaded once per union slot and held
// across the staged-row parts (no re-reads); the slot loop, staging and barriers are those of routes_product_rows_kernel.
template <bool Packed>
__device__ __forceinline__ Q2Held load_tp_tail(__amdgpu_buffer_rsrc_t rsrc, int row, unsigned group) {
    Q2Held raw;
    const int half = static_cast<int>(4U * q2_k_block_bytes);
    // Packed: scales [0, 8), codes [8, 40), d and dmin [40, 44); else block 4 (scales [0, 16), codes [16, 80), dm at 80).
    const int codes = Packed ? 8 : 16, dm = Packed ? 40 : 80;
    const auto q0 = __builtin_amdgcn_raw_buffer_load_b128(rsrc, row + half + codes, 0, 0);
    const auto q1 = __builtin_amdgcn_raw_buffer_load_b128(rsrc, row + half + codes + 16, 0, 0);
#pragma unroll
    for (unsigned k = 0; k < 4U; ++k) { raw.q[k] = q0[k]; raw.q[k + 4U] = q1[k]; }
    raw.scales = __builtin_amdgcn_raw_buffer_load_b32(rsrc, row + half + static_cast<int>(2U * group), 0, 0);
    raw.dm = __builtin_amdgcn_raw_buffer_load_b32(rsrc, row + half + dm, 0, 0);
    return raw;
}
inline constexpr unsigned tp_down_width = 1152U, tp_down_groups = tp_down_width / 32U;
template <unsigned MaxRows, unsigned W, unsigned Waves, bool Packed>
__global__ __launch_bounds__(32U * Waves) void tp_down_kernel(RoutesProductView view) {
    static_assert(W >= 1U && W <= 8U && tp_down_groups == 36U);
    constexpr unsigned width = tp_down_width, groups = tp_down_groups, threads = 32U * Waves;
    constexpr unsigned k_words = (width / 16U + threads - 1U) / threads;
    extern __shared__ int4 staged[];
    const unsigned slots = union_count(view.routes);
    auto* lds_codes = reinterpret_cast<signed char*>(staged);
    auto* lds_deltas = reinterpret_cast<float*>(lds_codes + static_cast<unsigned long long>(MaxRows) * width);
    const unsigned wave = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / 32U)));
    const unsigned lane = threadIdx.x % 32U, first = (blockIdx.x * Waves + wave) * W;
    const bool active = first < view.outputs;
    const unsigned tail_group = lane % 4U, tail_row = min(first + min(lane / 4U, W - 1U), view.outputs - 1U);
    for (unsigned j = blockIdx.y; j < slots; j += gridDim.y) {
        const EntryRange range{view.routes.begins[j], view.routes.begins[j + 1U] - view.routes.begins[j], view.routes.experts[j]};
        const EntryRange part0{range.begin, min(MaxRows, range.count), range.expert};
        StageRegs<MaxRows, k_words> s;
        stage_issue<MaxRows, k_words, threads>(view.routes, part0, view.codes, view.deltas, width, s);
        const auto rsrc = weight_rsrc(static_cast<const unsigned char*>(view.weights) + range.expert * view.expert_stride_bytes);
        Q2Held held[W];
#pragma unroll
        for (unsigned w = 0; w < W; ++w)
            held[w] = load_held_group<WeightFormat::q2_k>(rsrc, static_cast<int>(min(first + w, view.outputs - 1U) * view.row_bytes), lane);
        const Q2Held tail = load_tp_tail<Packed>(rsrc, static_cast<int>(tail_row * view.row_bytes), tail_group);
        asm volatile("" ::: "memory");  // issued here, ahead of the staging stores and barrier
        if (j != blockIdx.y) lds_barrier();
        stage_store<MaxRows, k_words, threads>(part0, width, s, lds_codes, lds_deltas);
        const auto part_product = [&](const EntryRange& part) {
            if (!active) return;
            float acc[W][MaxRows];
#pragma unroll
            for (unsigned w = 0; w < W; ++w)
#pragma unroll
                for (unsigned r = 0; r < MaxRows; ++r) acc[w][r] = 0.0F;
            accumulate_group<WeightFormat::q2_k, MaxRows, W, 0U>(held, lane, lds_codes, lds_deltas, part.count, width, nullptr, nullptr, acc);
            // The tail terms of accumulate_rows' second step (group 32 + l % 4 of row l / 4), one per lane.
            float term[MaxRows];
            {
                const unsigned g = 32U + tail_group;
                const auto q = decode_held(tail, 2U * tail_group);
#pragma unroll
                for (unsigned r = 0; r < MaxRows; ++r)
                    if (r < part.count) {
                        const auto* x = reinterpret_cast<const int4*>(lds_codes + static_cast<unsigned long long>(r) * width + 32U * g);
                        const int4 x0 = x[0], x1 = x[1];
                        const float delta = lds_deltas[r * groups + g];
                        const int ones = 0x01010101;
                        const int sx0 = unsigned_dot(ones, ones, ones, ones, x0.x, x0.y, x0.z, x0.w);
                        const int sx1 = unsigned_dot(ones, ones, ones, ones, x1.x, x1.y, x1.z, x1.w);
                        const int qx0 = unsigned_dot(q.w[0], q.w[1], q.w[2], q.w[3], x0.x, x0.y, x0.z, x0.w);
                        const int qx1 = unsigned_dot(q.w[4], q.w[5], q.w[6], q.w[7], x1.x, x1.y, x1.z, x1.w);
                        const int a = __mul24(static_cast<int>(q.scale0 & 15U), qx0) + __mul24(static_cast<int>(q.scale1 & 15U), qx1);
                        const int b = __mul24(static_cast<int>(q.scale0 >> 4U), sx0) + __mul24(static_cast<int>(q.scale1 >> 4U), sx1);
                        const float scaled = q.d * static_cast<float>(a);
                        const float offset = q.dmin * static_cast<float>(b);
                        term[r] = (scaled - offset) * delta;
                    }
            }
#pragma unroll
            for (unsigned w = 0; w < W; ++w)
#pragma unroll
                for (unsigned r = 0; r < MaxRows; ++r)
                    if (r < part.count) {
                        const float t = __shfl(term[r], static_cast<int>(4U * w + tail_group), 32);
                        if (lane < 4U) acc[w][r] = acc[w][r] + t;
                    }
#pragma unroll
            for (unsigned w = 0; w < W; ++w)
#pragma unroll
                for (unsigned r = 0; r < MaxRows; ++r)
                    if (r < part.count) acc[w][r] = butterfly_sum(acc[w][r]);
            if (lane != 0U) return;
#pragma unroll
            for (unsigned r = 0; r < MaxRows; ++r)
                if (r < part.count) {
                    const unsigned long long output = view.routes.outputs[part.begin + r];
#pragma unroll
                    for (unsigned w = 0; w < W; ++w)
                        if (first + w < view.outputs) view.out[output * view.outputs + first + w] = acc[w][r];
                }
        };
        part_product(part0);
        for (unsigned chunk = MaxRows; chunk < range.count; chunk += MaxRows) {
            const EntryRange part{range.begin + chunk, min(MaxRows, range.count - chunk), range.expert};
            lds_barrier();
            stage_rows_lds(view.routes, part, view.codes, view.deltas, width, lds_codes, lds_deltas);
            part_product(part);
        }
    }
}
// STRIX_DS41_TP_DOWN_FORM=CW: at most C staged rows per part (1, 2, 4 or 8) and W rows per wave (2, 4 or 8; 32 / W waves
// per block). Default 2 rows, W 8 (4 waves): the fastest at R1-R8 (expert-once bench --tp, layers 5/20/35, both ranks:
// R6 down 251 -> 200 us per layer, R1 62 -> 58, R8 322 -> 248; cap 8 with W 8 is much slower).
struct TpDownShape { unsigned cap = 2U, w = 8U; };
TpDownShape tp_down_shape() {
    TpDownShape s;
    const int f = moe_kernel_tuning().tp_down_form;
    if (f < 0) return s;
    const auto c = static_cast<unsigned>(f / 10 % 10), w = static_cast<unsigned>(f % 10);
    if (c == 1U || c == 2U || c == 4U || c == 8U) s.cap = c;
    if (w == 2U || w == 4U || w == 8U) s.w = w;
    return s;
}
template <unsigned MaxRows, unsigned W, bool Packed>
void launch_tp_down(const RoutesProductView& view, hipStream_t stream) {
    constexpr unsigned waves = 32U / W;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(tp_down_kernel<MaxRows, W, waves, Packed>), dim3((view.outputs + 31U) / 32U, rows_grid_y(view.routes)),
                       dim3(32U * waves), staged_bytes<MaxRows>(tp_down_width), stream, view);
}
template <unsigned MaxRows, bool Packed>
void tp_down_w(const RoutesProductView& view, const TpDownShape& s, hipStream_t stream) {
    if (s.w == 2U) launch_tp_down<MaxRows, 2U, Packed>(view, stream);
    else if (s.w == 8U) launch_tp_down<MaxRows, 8U, Packed>(view, stream);
    else launch_tp_down<MaxRows, 4U, Packed>(view, stream);
}
template <bool Packed>
void tp_down_rows(const RoutesProductView& view, const TpDownShape& s, hipStream_t stream) {
    const auto m = std::min(view.routes.max_rows, s.cap);
    if (m <= 1U) tp_down_w<1, Packed>(view, s, stream);
    else if (m <= 2U) tp_down_w<2, Packed>(view, s, stream);
    else if (m <= 4U) tp_down_w<4, Packed>(view, s, stream);
    else tp_down_w<8, Packed>(view, s, stream);
}
// The expert-TP down in the form above; false (nothing launched) when the flag is off or the shape is not 1152 wide.
bool tp_down(const RoutesProductView& view, bool packed, hipStream_t stream) {
    if (moe_kernel_tuning().tp_down <= 0 || view.width != tp_down_width) return false;
    const auto s = tp_down_shape();
    if (packed) tp_down_rows<true>(view, s, stream);
    else tp_down_rows<false>(view, s, stream);
    return true;
}

} // namespace

MoeKernelTuning& moe_kernel_tuning() {
    static MoeKernelTuning tuning = [] {
        MoeKernelTuning t;
        const auto read = [](const char* name, int& form) {
            if (const char* v = std::getenv(name); v && *v) form = std::atoi(v);
        };
        read("STRIX_DS41_MOE_ROUTED", t.routed);
        read("STRIX_DS41_MOE_SHARED", t.shared);
        read("STRIX_DS41_MOE_HEAD", t.head);
        read("STRIX_DS41_MOE_PREFILL", t.prefill);
        read("STRIX_DS41_EXPERT_SLOTS", t.slots);
        read("STRIX_DS41_EXPERT_SLOTS_Y", t.slots_y);
        read("STRIX_DS41_EXPERT_ROWS", t.rows);
        read("STRIX_DS41_EXPERT_ROWS_FORM", t.rows_form);
        read("STRIX_DS41_TP_DOWN", t.tp_down);
        read("STRIX_DS41_TP_DOWN_FORM", t.tp_down_form);
        return t;
    }();
    return tuning;
}

void enqueue_decode_blocks(WeightFormat format, const void* blocks, std::size_t block_count, float* out, void* stream) {
    const auto groups = static_cast<unsigned long long>(block_count) * block_elements(format) / 32U;
    if (!groups) return;
    hipLaunchKernelGGL(decode_kernel, dim3(static_cast<unsigned>((groups + 255U) / 256U)), dim3(256), 0,
                       static_cast<hipStream_t>(stream), format, static_cast<const unsigned char*>(blocks), groups, out);
    check(hipGetLastError(), "decode launch");
}
void enqueue_decode_engram_rows(const void* rows, std::size_t row_count, float* out, void* stream) {
    const auto values = static_cast<unsigned long long>(row_count) * engram_row_elements;
    if (!values) return;
    hipLaunchKernelGGL(engram_kernel, dim3(static_cast<unsigned>((values + 255U) / 256U)), dim3(256), 0,
                       static_cast<hipStream_t>(stream), static_cast<const unsigned char*>(rows),
                       static_cast<unsigned long long>(row_count), out);
    check(hipGetLastError(), "engram decode launch");
}
void enqueue_decode_scalar_tables(float* out, void* stream) {
    hipLaunchKernelGGL(scalar_tables_kernel, dim3(1), dim3(256), 0, static_cast<hipStream_t>(stream), out);
    check(hipGetLastError(), "scalar tables launch");
}
void enqueue_quantize_a8(const float* values, std::uint32_t width, std::uint32_t rows, std::int8_t* codes, float* deltas,
                         std::int32_t* sums, void* stream) {
    if (width == 0U || width % 32U != 0U || rows == 0U) throw std::invalid_argument("ds41 A8 quantize geometry");
    hipLaunchKernelGGL(quantize_a8_kernel, dim3(width / 32U, rows), dim3(32), 0, static_cast<hipStream_t>(stream), values, width,
                       reinterpret_cast<signed char*>(codes), deltas, sums);
    check(hipGetLastError(), "A8 quantize launch");
}
void enqueue_product(WeightFormat format, const ProductView& view, void* stream) {
    const auto elements = block_elements(format);
    if (!view.weights || !view.experts || !view.codes || !view.deltas || !view.out || view.rows == 0U ||
        view.rows > product_max_rows || view.width == 0U || view.width % elements != 0U || view.outputs == 0U ||
        view.expert_count == 0U || view.row_bytes != view.width / elements * block_bytes(format) ||
        reinterpret_cast<std::uintptr_t>(view.codes) % 16U != 0U)
        throw std::invalid_argument("ds41 product view is inconsistent");
    RoutesProductView routes;
    routes.weights = view.weights; routes.expert_stride_bytes = view.expert_stride_bytes; routes.row_bytes = view.row_bytes;
    routes.width = view.width; routes.outputs = view.outputs; routes.routes.experts = view.experts;
    routes.routes.expert_count = view.expert_count; routes.routes.max_rows = view.rows;
    routes.codes = view.codes; routes.deltas = view.deltas; routes.out = view.out;
    std::copy(std::begin(view.table), std::end(view.table), std::begin(routes.table));
    routes_product(format, routes, 0, view.rows, view.expert_count, view.rows, view.expert_count, static_cast<hipStream_t>(stream));
    check(hipGetLastError(), "product launch");
}
void enqueue_routes_product(WeightFormat format, const RoutesProductView& view, void* stream) {
    const auto elements = block_elements(format);
    if (!view.weights || !view.routes.experts || !view.routes.begins || !view.routes.inputs || !view.routes.outputs ||
        !view.codes || !view.deltas || !view.out || view.routes.expert_count == 0U || view.routes.max_rows == 0U ||
        view.routes.max_rows > 16U || view.width == 0U || view.width % elements != 0U || view.outputs == 0U ||
        view.row_bytes != view.width / elements * block_bytes(format) || reinterpret_cast<std::uintptr_t>(view.codes) % 16U != 0U)
        throw std::invalid_argument("ds41 routes product view is inconsistent");
    if (down_rows(format, view, static_cast<hipStream_t>(stream))) {
        check(hipGetLastError(), "routes product launch");
        return;
    }
    // The Q8_0 product (or its requant packed forms) is the shared expert's down; the other formats are routed experts.
    const auto& tuning = moe_kernel_tuning();
    const bool shared = format == WeightFormat::q8_0 || format == WeightFormat::cb4 || format == WeightFormat::u5 || format == WeightFormat::u6;
    int form = shared ? tuning.shared : tuning.routed;
    if (form < 0) form = default_product_form(format, view.routes.max_rows);
    routes_product(format, view, form, view.routes.max_rows, view.routes.expert_count, 0U, 0U, static_cast<hipStream_t>(stream));
    check(hipGetLastError(), "routes product launch");
}
void enqueue_routes_product_groups(WeightFormat format, const RoutesProductView& view, void* stream, bool half_tail) {
    const std::uint64_t need = half_tail ? view.width / 256U * q2_k_block_bytes + 44U : (view.width + 255U) / 256U * q2_k_block_bytes;
    if (format != WeightFormat::q2_k || (half_tail && view.width % 256U != 128U) || view.row_bytes < need || !view.weights ||
        !view.routes.experts || !view.routes.begins || !view.routes.inputs || !view.routes.outputs || !view.codes || !view.deltas || !view.out || view.routes.expert_count == 0U || view.routes.max_rows == 0U ||
        view.routes.max_rows > 16U || view.width == 0U || view.width % 32U != 0U || view.outputs == 0U ||
        reinterpret_cast<std::uintptr_t>(view.codes) % 16U != 0U)
        throw std::invalid_argument("ds41 routes product (groups) view is inconsistent");
    int form = moe_kernel_tuning().routed;
    if (form < 0) form = default_product_form(format, view.routes.max_rows);
    const auto queue = static_cast<hipStream_t>(stream);
    if (tp_down(view, half_tail, queue)) {
        check(hipGetLastError(), "routes product (groups) launch");
        return;
    }
    if (half_tail) dispatch_routes_product<WeightFormat::q2_k, true>(view, form, view.routes.max_rows, view.routes.expert_count, 0U, 0U, queue);
    else routes_product(format, view, form, view.routes.max_rows, view.routes.expert_count, 0U, 0U, queue);
    check(hipGetLastError(), "routes product (groups) launch");
}
void enqueue_gate_up(WeightFormat format, const GateUpView& view, void* stream) {
    const auto elements = block_elements(format);
    if (format == WeightFormat::q2_k || !view.gate || !view.up || !view.routes.experts || !view.routes.begins ||
        !view.routes.inputs || !view.routes.outputs || !view.codes || !view.deltas || !view.act_codes || !view.act_deltas ||
        !view.act_sums || (!view.gate_out) != (!view.up_out) || view.routes.expert_count == 0U || view.routes.max_rows == 0U ||
        view.routes.max_rows > 16U || view.hidden % elements != 0U || view.inter == 0U || view.inter % 32U != 0U ||
        view.row_bytes != view.hidden / elements * block_bytes(format) || reinterpret_cast<std::uintptr_t>(view.codes) % 16U != 0U)
        throw std::invalid_argument("ds41 gate/up view is inconsistent");
    const auto queue = static_cast<hipStream_t>(stream);
    if (gate_up_rows(format, view, queue)) {
        check(hipGetLastError(), "gate/up launch");
        return;
    }
    // The Q8_0 gate/up is the shared expert's; the other formats are routed experts.
    const auto& tuning = moe_kernel_tuning();
    const bool shared = format == WeightFormat::q8_0 || format == WeightFormat::cb4 || format == WeightFormat::u5 ||
                        format == WeightFormat::u6;  // the shared expert in Q8_0 or a requant packed form
    int form = shared ? tuning.shared : tuning.routed;
    if (form < 0) form = default_gate_up_form(format, view.routes.max_rows);
    if (format == WeightFormat::iq2_xxs) dispatch_gate_up<WeightFormat::iq2_xxs>(view, form, queue);
    else if (format == WeightFormat::q8_0) dispatch_gate_up<WeightFormat::q8_0>(view, form, queue);
    else if (format == WeightFormat::cb4) dispatch_gate_up<WeightFormat::cb4>(view, form, queue);
    else if (format == WeightFormat::u5) dispatch_gate_up<WeightFormat::u5>(view, form, queue);
    else if (format == WeightFormat::u6) dispatch_gate_up<WeightFormat::u6>(view, form, queue);
    else dispatch_gate_up<WeightFormat::mxfp4>(view, form, queue);
    check(hipGetLastError(), "gate/up launch");
}
void enqueue_combine(const CombineView& view, void* stream) {
    if (!view.raw || !view.shared || !view.weights || !view.order || !view.low_counts || !view.out || view.rows == 0U ||
        view.top_k == 0U || view.hidden == 0U)
        throw std::invalid_argument("ds41 combine view is inconsistent");
    hipLaunchKernelGGL(combine_kernel, dim3((view.hidden + 255U) / 256U, view.rows), dim3(256), 0, static_cast<hipStream_t>(stream), view);
    check(hipGetLastError(), "combine launch");
}

static void enqueue_router_impl(const RouterView& view, void* stream, const BuildRoutesView* routes) {
    if (view.rows > product_max_rows && view.rows <= 16U && view.x && view.scores && view.ids && view.weights) {
        for (unsigned first = 0; first < view.rows; first += product_max_rows) {
            RouterView part = view;
            part.rows = std::min(product_max_rows, view.rows - first);
            part.x += static_cast<std::size_t>(first) * view.hidden;
            part.scores += static_cast<std::size_t>(first) * view.experts;
            part.ids += static_cast<std::size_t>(first) * view.top_k;
            part.weights += static_cast<std::size_t>(first) * view.top_k;
            if (part.image_rows) part.image_rows += first;
            enqueue_router(part, stream);
        }
        return;
    }
    if (!view.x || !(view.weight || view.weight_bf16) || !view.bias || !view.scores || !view.ids || !view.weights || view.rows == 0U ||
        view.rows > product_max_rows || view.hidden == 0U || view.experts == 0U || view.experts > 1024U || view.top_k == 0U ||
        view.top_k > 16U || view.top_k > view.experts || (view.image_rows && !view.bias_image))
        throw std::invalid_argument("ds41 router view is inconsistent");
    const auto queue = static_cast<hipStream_t>(stream);
    const dim3 grid((view.experts + router_waves - 1U) / router_waves), block(32U * router_waves);
    // router row waves by default (P11: -1.7 ms at R1, -7.6 ms per R6 window, same bits); STRIX_DS41_ROUTER_ROWS=0 off
    static const bool rows_env = strix::envcfg::flag("STRIX_DS41_ROUTER_ROWS", true);
    // STRIX_DS41_ROUTER_FAST=1 (default off; RouterView::fast for tests): the one-wave selection, fused in the scores
    // kernel's last block when RouterView::tickets is set, after the row-wave scores (1) or the wide ones (4 / 8 / 16).
    // Same bits.
    static const unsigned fast_env = [] { const char* e = std::getenv("STRIX_DS41_ROUTER_FAST"); return e && *e == '1' ? 1U : 0U; }();
    const unsigned fast = view.fast ? view.fast : fast_env;
    if (fast == 1U && view.hidden % (32U * router_row_ahead) == 0U) {
        launch_router_row_select(view, queue);
        check(hipGetLastError(), "router launch");
        return;
    }
    if (fast > 1U && view.weight_bf16 && view.hidden % 256U == 0U && reinterpret_cast<std::uintptr_t>(view.weight_bf16) % 16U == 0U) {
        if (fast == 4U) launch_router_fast<4>(view, queue);
        else if (fast == 16U) launch_router_fast<16>(view, queue);
        else launch_router_fast<8>(view, queue);
        check(hipGetLastError(), "router launch");
        return;
    }
    if ((view.row_waves || rows_env) && view.hidden % (32U * router_row_ahead) == 0U)
        hipLaunchKernelGGL(router_row_kernel, dim3((view.experts * view.rows + router_row_waves - 1U) / router_row_waves),
                           dim3(32U * router_row_waves), 0, queue, view);
    else if (!view.staged) hipLaunchKernelGGL(router_scores_wave_kernel, dim3((view.experts + 7U) / 8U), dim3(256), 0, queue, view);
    else if (view.rows <= 1U) hipLaunchKernelGGL(router_scores_kernel<1>, grid, block, 0, queue, view);
    else if (view.rows <= 2U) hipLaunchKernelGGL(router_scores_kernel<2>, grid, block, 0, queue, view);
    else if (view.rows <= 4U) hipLaunchKernelGGL(router_scores_kernel<4>, grid, block, 0, queue, view);
    else hipLaunchKernelGGL(router_scores_kernel<8>, grid, block, 0, queue, view);
    if (routes) hipLaunchKernelGGL(router_select_routes_kernel, dim3(1), dim3(256), 0, queue, view, *routes, routes_fast_enabled());
    else hipLaunchKernelGGL(router_select_kernel, dim3(view.rows), dim3(256), 0, queue, view);
    check(hipGetLastError(), "router launch");
}
void enqueue_router(const RouterView& view, void* stream) { enqueue_router_impl(view, stream, nullptr); }
bool enqueue_router_routes(const RouterView& view, const BuildRoutesView& routes, void* stream) {
    // Fast router modes already fuse selection into a producer. Preserve those launch contracts.
    const char* fast = std::getenv("STRIX_DS41_ROUTER_FAST");
    if (view.fast || (fast && fast[0] == '1') || !view.rows || view.rows > product_max_rows ||
        view.rows != routes.rows || view.top_k != routes.top_k || view.ids != routes.ids ||
        view.rows * view.top_k > 64U) return false;
    if (!routes.experts || !routes.begins || !routes.rows_out || !routes.slots || !routes.order ||
        !routes.low_counts || !routes.counts || routes.first >= routes.end || routes.base > routes.first)
        throw std::invalid_argument("ds41 fused router routes view is inconsistent");
    enqueue_router_impl(view, stream, &routes);
    return true;
}
void enqueue_half_sum(const HalfSumView& view, void* stream) {
    if (!view.raw || !view.weights || !view.order || !view.low_counts || !view.out || view.rows == 0U || view.top_k == 0U ||
        view.hidden == 0U || view.half > 1U)
        throw std::invalid_argument("ds41 half sum view is inconsistent");
    hipLaunchKernelGGL(half_sum_kernel, dim3((view.hidden + 255U) / 256U, view.rows), dim3(256), 0, static_cast<hipStream_t>(stream), view);
    check(hipGetLastError(), "half sum launch");
}
void enqueue_canonical_sum(const float* shared, const float* low, const float* high, std::size_t count, float* out, void* stream) {
    if (!shared || !low || !high || !out || count == 0U) throw std::invalid_argument("ds41 canonical sum view is inconsistent");
    hipLaunchKernelGGL(canonical_sum_kernel, dim3(static_cast<unsigned>((count + 255U) / 256U)), dim3(256), 0,
                       static_cast<hipStream_t>(stream), shared, low, high, static_cast<unsigned long long>(count), out);
    check(hipGetLastError(), "canonical sum launch");
}

void enqueue_half_sum_ready(const HalfSumView& view, dist::ProducerStage stage, void* stream) {
    if (!view.raw || !view.weights || !view.order || !view.low_counts || view.out != stage.data || !stage.data || !stage.flag ||
        !stage.counter || !stage.token || view.rows == 0U || view.top_k == 0U || view.hidden == 0U || view.half > 1U)
        throw std::invalid_argument("ds41 half sum ready view is inconsistent");
    hipLaunchKernelGGL(half_sum_ready_kernel, dim3((view.hidden + 255U) / 256U, view.rows), dim3(256), 0, static_cast<hipStream_t>(stream),
                       view, stage);
    check(hipGetLastError(), "half sum ready launch");
}
void enqueue_canonical_sum_ready(const float* shared, const float* low, const float* high, std::size_t count, float* out,
                                 dist::ConsumerStage peer, void* stream) {
    if (!shared || !low || !high || !out || count == 0U || !peer.data || !peer.flag || !peer.counter || !peer.token)
        throw std::invalid_argument("ds41 canonical sum ready view is inconsistent");
    hipLaunchKernelGGL(canonical_sum_ready_kernel, dim3(static_cast<unsigned>((count + 255U) / 256U)), dim3(256), 0,
                       static_cast<hipStream_t>(stream), shared, low, high, static_cast<unsigned long long>(count), out, peer.flag,
                       peer.counter, peer.token);
    check(hipGetLastError(), "canonical sum ready launch");
}

void enqueue_canonical_sum_split(const float* shared_lo, const float* shared_hi, const float* low, const float* high,
                                 std::uint32_t rows, std::uint32_t hidden, float* out, float* shared_out, void* stream) {
    if (!shared_lo || !shared_hi || !low || !high || !out || rows == 0U || hidden == 0U || hidden % 2U != 0U)
        throw std::invalid_argument("ds41 split canonical sum view is inconsistent");
    hipLaunchKernelGGL(canonical_sum_split_kernel, dim3((hidden + 255U) / 256U, rows), dim3(256), 0, static_cast<hipStream_t>(stream),
                       shared_lo, shared_hi, low, high, hidden, out, shared_out);
    check(hipGetLastError(), "split canonical sum launch");
}
void enqueue_assemble_a8_halves(const std::int8_t* first_codes, const float* first_deltas, const std::int8_t* second_codes,
                                const float* second_deltas, std::uint32_t rows, std::uint32_t half, std::int8_t* codes,
                                float* deltas, void* stream) {
    const auto aligned = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 4U == 0U; };
    if (!first_codes || !first_deltas || !second_codes || !second_deltas || !codes || !deltas || rows == 0U || half == 0U ||
        half % 32U != 0U || !aligned(first_codes) || !aligned(second_codes) || !aligned(codes))
        throw std::invalid_argument("ds41 A8 halves view is inconsistent");
    hipLaunchKernelGGL(assemble_a8_halves_kernel, dim3((half / 2U + 255U) / 256U, rows), dim3(256), 0, static_cast<hipStream_t>(stream),
                       reinterpret_cast<const int*>(first_codes), first_deltas, reinterpret_cast<const int*>(second_codes), second_deltas,
                       half, reinterpret_cast<int*>(codes), deltas);
    check(hipGetLastError(), "A8 halves launch");
}

void enqueue_build_routes(const BuildRoutesView& view, void* stream) {
    if (!view.ids || !view.experts || !view.begins || !view.rows_out || !view.slots || !view.order || !view.low_counts ||
        !view.counts || view.rows == 0U || view.top_k == 0U || view.rows > 16U || view.top_k > 128U / view.rows || view.first >= view.end ||
        view.base > view.first)
        throw std::invalid_argument("ds41 build routes view is inconsistent");
    if (routes_fast_enabled()) { enqueue_build_routes_fast(view, stream); return; }  // STRIX_DS41_ROUTES_FAST=1, same outputs
    if (view.rows * view.top_k > 64U)
        enqueue_build_routes_wide_serial(view, stream);
    else hipLaunchKernelGGL(build_routes_kernel, dim3(1), dim3(64), 0, static_cast<hipStream_t>(stream), view);
    check(hipGetLastError(), "build routes launch");
}
} // namespace strix::ds41
