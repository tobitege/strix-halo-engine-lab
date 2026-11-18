// CUDA Qwen prefill, family "hc": hyper-connection mixes, elementwise steps, MoE router and LM head, bit-identical to the
// HIP A8-profile tiled prefill on gfx1151 (engine bf60d9d2). See include/strix/cuda/qwen_hc.hpp for the API.
//
// Every floating-point operation below is one explicit IEEE operation (__fmaf_rn / __fadd_rn / __fmul_rn / __fdiv_rn)
// mirroring the HIP instruction, as read in the gfx1151 ISA (disassembly of each translation unit; ISA line numbers are
// relative to the kernel symbol's label line):
//   [N] rows_prepare_hyper_norm_kernel<u16,HyperTileRows> (hip_hyper_mix_launcher.s:398955): `sum += v*v` is
//       v_dual_fmac_f32 v3, v5, v5 (+66); LDS tree v_add_f32 strides 128..1 (+86..+185); `red/2560` is the
//       v_div_scale/fmas/fixup sequence (correctly rounded, +199..+216); `+ eps` v_add_f32 (+217); rsqrtf = the
//       2^24 / v_rsq_f32 / 2^12 sequence (+219..+226, amd_rsqrtf); `x*inverse*w` two v_mul_f32 (+257, +259).
//   [P] wide2_splitk_hyper_projection_kernel<8,true,false> (hip_hyper_mix_launcher.s:312631): chains are the
//       generated v_dual_fmac_f32 of hip_hc_wide2_vopd.inc (fma(x, c, acc), j = 0..3 per 128-feature chunk); fold =
//       DPP row_xmask 1/2/4/8 + v_permlanex16 then v_add_f32 (+822..+1025); epilogue: total/streams correctly rounded
//       (+1124..+1139, +2540..+2547), expf(-scaled) as amd_expf with negated constants (+2548..+2560), 1 + e
//       (v_add_f32), 1/(1+e) correctly rounded (+2561..+2574), low = scaled * sig v_mul_f32 (+2576), injection =
//       sig + sig v_add_f32 (+1177).
//   [O] wide2_hyper_output_kernel<u16,32,true,false> (hip_hyper_mix_launcher.s:340714): gate chains = the generated
//       v_dual_fmac_f32 of hip_hc_wide2_output_vopd.inc (fma(low, c, g), k ascending); the stream mix is CONTRACTED:
//       v_fma_f32 v3, n0, sig0, 0 then v_fmac_f32 for s = 1, 2, 3 (+838..+848); `/4` is v_mul_f32 0x3e800000 (+849);
//       BF16 RNE v_bfe/v_add3 (+851..+853).
//   [E] embedding_kernel<u16,q8_0> (hip_embedding_launcher.s:3165): v_fma_mix_f32 scale, code, -0 = the exact product.
//   [I] inject_rows_kernel<u16,ElementwiseTileRowsArgs> (hip_runtime_launchers.s:2602) and inject_kernel<u16> (:1560):
//       CONTRACTED, v_fmac_f32 hyper, weight, hidden. [C] combine_rows_kernel<u16,TileRows> (:3235): v_add_f32.
//   [S] rows_moe_route_select_kernel<wave,RouterTileRowsArgs> (hip_moe_route_launcher.s:107647): v_max_f32 (+86..+138),
//       expf(l - max) = amd_expf (+159..+175), adds (+178, +220), correctly rounded division (+236..+252), compare/
//       select top-k with ds_bpermute xor butterflies (+697..).
//   [L] prefetch8_wave_q8_lm_head_kernel<u16,8> (hip_lm_head_launcher.s:180867): coefficient = v_fma_mix_f32 scale,
//       code, -0 (exact product, +103..+123), leaf chain CONTRACTED v_fmac_f32 partial, coef, x (+114..+129),
//       __shfl_down tree adds (+146..).
#include "../../weight/arch/sm120_sources.cuh"
#include "strix/env_flags.hpp"
#include "strix/lever.hpp"
#include "strix/cuda/core/decode_rows.hpp"
#include "strix/cuda/qwen_hc.hpp"
#include "strix/cuda/amd_math.cuh"
#include "../../core/mma_tile.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <string>

namespace strix::cuda::qwen_hc {
namespace {

namespace am = strix::cuda::amdmath;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("qwen_hc ") + what + ": " + cudaGetErrorString(e));
}

// STRIX_QRTX_HC_COMPACT_NORM=0: the HC mix stores and re-reads the full FP32 normalized rows (40 KB per row). Bitwise
// (exact and FAST tiled paths): the projection and output kernels rebuild each normalized value from the BF16 input and
// the stored per-(row, stream) inverse with the norm kernel's own two roundings.
inline constexpr strix::lever::Decl hc_compact_norm_lever{
    .name = "qwen_hc_compact_norm", .env = "STRIX_QRTX_HC_COMPACT_NORM", .exactness = strix::lever::Exactness::bitwise,
    .default_on = true, .source = "scripts/bench/cuda/qwen_fast/ident.sh: exact and FAST logit dumps byte-identical on/off"};
bool hc_compact_norm_enabled() {
    static const bool on = strix::lever::select(hc_compact_norm_lever, {}).on;
    return on;
}
// STRIX_QRTX_HC_INJECT_VEC=0: the scalar inject kernel (one element per thread). Bitwise.
inline constexpr strix::lever::Decl hc_inject_vec_lever{
    .name = "qwen_hc_inject_vec", .env = "STRIX_QRTX_HC_INJECT_VEC", .exactness = strix::lever::Exactness::bitwise,
    .default_on = true, .source = "scripts/bench/cuda/qwen_fast/ident.sh: exact and FAST logit dumps byte-identical on/off"};
bool hc_inject_vec_enabled() {
    static const bool on = strix::lever::select(hc_inject_vec_lever, {}).on;
    return on;
}

__device__ __forceinline__ float bf16_to_float(std::uint16_t v) { return __uint_as_float(static_cast<std::uint32_t>(v) << 16U); }
// HIP store_activation (hip_hyper_mix_launcher.hip:202-214, hip_runtime_launchers.hip:118-128): RNE, no NaN case.
// NaN: gfx1151 arithmetic writes the canonical 0x7fc00000 for a generated NaN (-> BF16 0x7fc0), CUDA writes 0x7fffffff,
// which the rounding trick would wrap to 0x8000 (-0, a silently lost NaN). So a NaN is stored as HIP's canonical NaN
// (payloads of propagated input NaNs are not reproduced; plan rules 3-4).
__device__ __forceinline__ std::uint16_t float_to_bf16(float v) {
    if (v != v) return 0x7fc0U;
    const std::uint32_t b = __float_as_uint(v);
    return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U);
}
__device__ __forceinline__ float half_bits_to_float(std::uint16_t h) { return weight::Q8Block::scale_bits(h); }
// sigmoid(x) = 1.0F / (1.0F + expf(-x)) (hip_hyper_mix_launcher.hip:399-404; ISA [P], [O]).
__device__ __forceinline__ float amd_sigmoid(const am::Tables& t, float x) {
    return __fdiv_rn(1.0F, __fadd_rn(1.0F, am::amd_expf(t, -x)));
}

// ---------------------------------------------------------------------------------------------------------------------
// HC pack (replaces wide2_pack_down_kernel, hip_hyper_mix_launcher.hip:2887-2905: data movement only).
constexpr std::size_t kDownCodeBytes = std::size_t{kRank} * kWide;                  // int8
constexpr std::size_t kDownScaleBytes = std::size_t{kRank} * (kWide / 32U) * 4U;
constexpr std::size_t kUpCodeBytes = std::size_t{kRank / 4U} * kStreams * kHidden * 4U;
constexpr std::size_t kUpScaleBytes = std::size_t{kRank / 32U} * kStreams * kHidden * 4U;

__global__ void hc_pack_kernel(const unsigned char* __restrict__ down, const unsigned char* __restrict__ up, HcPacked p) {
    const unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr unsigned kDownItems = kRank * (kWide / 32U), kUpItems = kWide * (kRank / 32U);
    if (id < kDownItems) {
        const unsigned o = id / (kWide / 32U), b = id % (kWide / 32U);
        const unsigned char* block = down + (static_cast<std::size_t>(o) * (kWide / 32U) + b) * 34U;
        p.down_scales[id] = half_bits_to_float(static_cast<std::uint16_t>(block[0] | (block[1] << 8U)));
        std::uint32_t* dst = reinterpret_cast<std::uint32_t*>(p.down_codes + static_cast<std::size_t>(o) * kWide + b * 32U);
#pragma unroll
        for (unsigned w = 0; w < 8U; ++w)
            dst[w] = static_cast<std::uint32_t>(block[2 + 4 * w]) | (static_cast<std::uint32_t>(block[3 + 4 * w]) << 8U) |
                (static_cast<std::uint32_t>(block[4 + 4 * w]) << 16U) | (static_cast<std::uint32_t>(block[5 + 4 * w]) << 24U);
    } else if (id < kDownItems + kUpItems) {
        const unsigned u = id - kDownItems, row = u / (kRank / 32U), b = u % (kRank / 32U);
        const unsigned s = row / kHidden, f = row % kHidden;
        const unsigned char* block = up + (static_cast<std::size_t>(row) * (kRank / 32U) + b) * 34U;
        p.up_scales[(static_cast<std::size_t>(b) * kStreams + s) * kHidden + f] =
            half_bits_to_float(static_cast<std::uint16_t>(block[0] | (block[1] << 8U)));
#pragma unroll
        for (unsigned w = 0; w < 8U; ++w)
            p.up_codes[(static_cast<std::size_t>(b * 8U + w) * kStreams + s) * kHidden + f] =
                static_cast<std::uint32_t>(block[2 + 4 * w]) | (static_cast<std::uint32_t>(block[3 + 4 * w]) << 8U) |
                (static_cast<std::uint32_t>(block[4 + 4 * w]) << 16U) | (static_cast<std::uint32_t>(block[5 + 4 * w]) << 24U);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// [N] rows_prepare_hyper_norm_kernel (hip_hyper_mix_launcher.hip:1757-1776): one block per (row, stream), 256 threads.
// Compact (lever qwen_hc_compact_norm, tiled paths): only the inverse is stored, at normalized[row][stream]
// (the wide part of the row is otherwise unused); the consumers rebuild (x * inverse) * w from the BF16 input with the
// same two roundings (norm_value), so every value they use is bit-identical while 40 KB per row are neither written
// nor read twice.
__device__ __forceinline__ float norm_value(std::uint16_t x, float inverse, float w) {
    return __fmul_rn(__fmul_rn(bf16_to_float(x), inverse), w);
}
// Four normalized values of features col..col+3 (col % 4 == 0, inside one stream) of one row: the stored FP32 values,
// or (Compact) the bit-identical rebuild from the BF16 input and the stored inverse.
template <bool Compact>
__device__ __forceinline__ float4 load_x4(const float* x_rows, const std::uint16_t* xin, const float* xnorm, std::size_t row, unsigned col) {
    if constexpr (Compact) {
        const uint2 v = *reinterpret_cast<const uint2*>(xin + row * kWide + col);
        const float inv = x_rows[row * kNormalizedPitch + col / kHidden];
        const float4 w = *reinterpret_cast<const float4*>(xnorm + col);
        return make_float4(norm_value(static_cast<std::uint16_t>(v.x), inv, w.x), norm_value(static_cast<std::uint16_t>(v.x >> 16U), inv, w.y),
                           norm_value(static_cast<std::uint16_t>(v.y), inv, w.z), norm_value(static_cast<std::uint16_t>(v.y >> 16U), inv, w.w));
    } else {
        return *reinterpret_cast<const float4*>(x_rows + row * kNormalizedPitch + col);
    }
}
template <bool Compact = false>
__global__ __launch_bounds__(256) void hc_norm_kernel(const std::uint16_t* __restrict__ input, const float* __restrict__ norm,
    float epsilon, float* __restrict__ normalized, am::Tables t) {
    const unsigned row = blockIdx.x, stream = blockIdx.y, lane = threadIdx.x;
    const std::uint16_t* in = input + static_cast<std::size_t>(row) * kWide + stream * kHidden;
    float* out = normalized + static_cast<std::size_t>(row) * kNormalizedPitch + stream * kHidden;
    __shared__ float reduction[256];
    __shared__ float inverse;
    float values[kHidden / 256U];
    float sum = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < kHidden / 256U; ++i) {  // features lane, lane + 256, ... ascending
        values[i] = bf16_to_float(in[lane + 256U * i]);
        sum = __fmaf_rn(values[i], values[i], sum);
    }
    reduction[lane] = sum;
    __syncthreads();
    // LDS tree :1768-1771: reduction[l] += reduction[l + stride], stride 128, 64, ..., 1, barrier after each level.
    for (unsigned stride = 128U; stride > 16U; stride >>= 1U) {
        if (lane < stride) reduction[lane] = __fadd_rn(reduction[lane], reduction[lane + stride]);
        __syncthreads();
    }
    if (lane < 32U) {  // strides 16..1 inside warp 0: the same pairs (l, l + stride) as the LDS levels
        float v = __fadd_rn(reduction[lane], reduction[lane + 16U]);
#pragma unroll
        for (unsigned stride = 8U; stride > 0U; stride >>= 1U) v = __fadd_rn(v, __shfl_down_sync(0xffffffffU, v, stride));
        if (lane == 0U) inverse = am::amd_rsqrtf(t, __fadd_rn(__fdiv_rn(v, static_cast<float>(kHidden)), epsilon));
        if (Compact && lane == 0U) normalized[static_cast<std::size_t>(row) * kNormalizedPitch + stream] = inverse;
    }
    if constexpr (Compact) return;
    __syncthreads();
    const float inv = inverse;
    const float* w = norm + stream * kHidden;
#pragma unroll
    for (unsigned i = 0; i < kHidden / 256U; ++i)
        out[lane + 256U * i] = __fmul_rn(__fmul_rn(values[i], inv), w[lane + 256U * i]);
}

// ---------------------------------------------------------------------------------------------------------------------
// [P] wide2_splitk_hyper_projection_kernel<8, true> (hip_hyper_mix_launcher.hip:2907-3050).
// Per (row, output) the HIP DAG is (window lane l = 0..31, tile T = 0..15, sub-chunk s = 0..4, j = 0..3):
//   leaf(T, l)  = fma chain from 0 over s, j ascending: acc = fma(x[f], c[f], acc), f = 640 T + 128 s + 4 l + j
//   tile(T)     = xor butterfly over l with distances 16, 8, 4, 2, 1 (wide2_lane_tree :2865-2876 on bit-reversed
//                 hardware lanes = window-lane distances 16, 8, 4, 2, 1; same as the serial __shfl_xor 16..1 :1081-1086)
//   total       = ((0 + tile(0)) + tile(1)) + ... + tile(15)   (:3000-3006)
//   low         = (total/4) * sigmoid(total/4);  injection = 2 * sigmoid(total/4)   (:3016-3030)
// CUDA mapping (same as HIP): block = 8 warps x 8 outputs, 16 rows; hardware lane = window lane; 128 accumulators
// (16 rows x 8 outputs) per lane. The butterfly is a transpose-reduce: at each level a lane keeps the half of its
// items selected by its lane bit and receives its partner's value of the same items, so after 5 levels lane L holds
// 4 finished (row, output) pairs: row q = 8 b4 + 4 b3 + 2 b2 + b1, outputs 4 b0 + i (b = bits of L).
constexpr unsigned kProjRows = 16U, kProjOutputs = 8U, kProjWarps = 8U, kProjItems = kProjRows * kProjOutputs;
constexpr unsigned kProjLowBlocks = kRank / (kProjWarps * kProjOutputs);  // 5

// One butterfly level at distance 16 >> Level: items [0, half) keep the lane's half and add the partner's same items.
template <unsigned Level> __device__ __forceinline__ void fold_level(float (&acc)[kProjItems], unsigned lane) {
    constexpr unsigned distance = 16U >> Level, half = 64U >> Level;
    const bool upper = (lane & distance) != 0U;
#pragma unroll
    for (unsigned i = 0; i < half; ++i) {
        const float keep = upper ? acc[i + half] : acc[i];
        const float send = upper ? acc[i] : acc[i + half];
        acc[i] = __fadd_rn(keep, __shfl_xor_sync(0xffffffffU, send, distance));
    }
}

// x_rows and low_rows are the same workspace (normalized part read, low part written): no __restrict__ on them.
template <bool Compact = false>
__global__ __launch_bounds__(256, 1) void hc_projection_kernel(const float* x_rows, std::uint32_t rows,
    const std::int8_t* __restrict__ down_codes, const float* __restrict__ down_scales, const float* __restrict__ inject,
    float* low_rows, float* __restrict__ injection, am::Tables t, const std::uint16_t* __restrict__ xin = nullptr,
    const float* __restrict__ xnorm = nullptr) {
    __shared__ float4 xs[2][kProjRows][32];
    const unsigned warp = threadIdx.x / 32U, lane = threadIdx.x % 32U, row0 = blockIdx.y * kProjRows;
    const bool is_inject = blockIdx.x >= kProjLowBlocks;
    const unsigned first = is_inject ? 0U : (blockIdx.x * kProjWarps + warp) * kProjOutputs;
    const unsigned total_outputs = is_inject ? (warp == 0U ? kStreams : 0U) : kRank;
    const bool active = first < total_outputs;

    const auto stage = [&](unsigned chunk, unsigned buffer) {
#pragma unroll
        for (unsigned m = 0; m < 2U; ++m) {
            const unsigned e = threadIdx.x + 256U * m, r = e / 32U, k = e % 32U;
            float4 v = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            if (row0 + r < rows)
                v = load_x4<Compact>(x_rows, xin, xnorm, row0 + r, (chunk / 5U) * 640U + (chunk % 5U) * 128U + k * 4U);
            xs[buffer][r][k] = v;
        }
    };
    const auto fetch = [&](unsigned chunk, float4 (&next)[2]) {
#pragma unroll
        for (unsigned m = 0; m < 2U; ++m) {
            const unsigned e = threadIdx.x + 256U * m, r = e / 32U, k = e % 32U;
            next[m] = row0 + r < rows ? load_x4<Compact>(x_rows, xin, xnorm, row0 + r, (chunk / 5U) * 640U + (chunk % 5U) * 128U + k * 4U)
                                      : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        }
    };

    float acc[kProjItems], total[4];
#pragma unroll
    for (unsigned i = 0; i < kProjItems; ++i) acc[i] = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < 4U; ++i) total[i] = 0.0F;
    // Weights of the next chunk are loaded one chunk ahead (as HIP's load_weights, :2975): raw[] holds, for the Q8
    // blocks, the 8 code words then the 8 scales (bits); for the injection block, the 4 x 4 FP32 weights.
    std::uint32_t raw[16];
    const auto load_raw = [&](unsigned chunk) {
        const unsigned feature = (chunk / 5U) * 640U + (chunk % 5U) * 128U + lane * 4U;
        if (is_inject) {
#pragma unroll
            for (unsigned p = 0; p < 4U; ++p) {
                const float4 w = first + p < total_outputs ?
                    *reinterpret_cast<const float4*>(inject + static_cast<std::size_t>(first + p) * kWide + feature) :
                    make_float4(0.0F, 0.0F, 0.0F, 0.0F);
                raw[4U * p] = __float_as_uint(w.x); raw[4U * p + 1U] = __float_as_uint(w.y);
                raw[4U * p + 2U] = __float_as_uint(w.z); raw[4U * p + 3U] = __float_as_uint(w.w);
            }
        } else {
#pragma unroll
            for (unsigned p = 0; p < kProjOutputs; ++p) {
                raw[p] = *reinterpret_cast<const std::uint32_t*>(down_codes + static_cast<std::size_t>(first + p) * kWide + feature);
                raw[8U + p] = __float_as_uint(down_scales[static_cast<std::size_t>(first + p) * (kWide / 32U) + feature / 32U]);
            }
        }
    };
    stage(0U, 0U);
    if (active) load_raw(0U);
    __syncthreads();
    for (unsigned chunk = 0; chunk < 80U; ++chunk) {
        float4 next[2];
        if (chunk + 1U < 80U) fetch(chunk + 1U, next);
        if (active) {
            float c[kProjOutputs][4];
#pragma unroll
            for (unsigned p = 0; p < kProjOutputs; ++p)
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j)
                    // injection: FP32 weights (outputs 4..7 of the injection warp are zero, as HIP's padding);
                    // Q8: coefficient = scale * float(code), exact (:2967-2971)
                    c[p][j] = is_inject ? (p < 4U ? __uint_as_float(raw[4U * p + j]) : 0.0F) :
                        __fmul_rn(__uint_as_float(raw[8U + p]), static_cast<float>(static_cast<signed char>((raw[p] >> (8U * j)) & 255U)));
            if (chunk + 1U < 80U) load_raw(chunk + 1U);
            const float4 (*x)[32] = xs[chunk & 1U];
#pragma unroll
            for (unsigned r = 0; r < kProjRows; ++r) {
                const float4 v = x[r][lane];
                const float in[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j)
#pragma unroll
                    for (unsigned p = 0; p < kProjOutputs; ++p) acc[r * kProjOutputs + p] = __fmaf_rn(in[j], c[p][j], acc[r * kProjOutputs + p]);
            }
            if (chunk % 5U == 4U) {
                // Tile fold: xor butterfly, distances 16, 8, 4, 2, 1 (transpose-reduce form, see above).
                fold_level<0>(acc, lane); fold_level<1>(acc, lane); fold_level<2>(acc, lane); fold_level<3>(acc, lane);
                fold_level<4>(acc, lane);
#pragma unroll
                for (unsigned i = 0; i < 4U; ++i) total[i] = __fadd_rn(total[i], acc[i]);
#pragma unroll
                for (unsigned i = 0; i < kProjItems; ++i) acc[i] = 0.0F;
            }
        }
        if (chunk + 1U < 80U) {  // buffer (chunk + 1) & 1 was last read in chunk - 1, before the barrier below
#pragma unroll
            for (unsigned m = 0; m < 2U; ++m) {
                const unsigned e = threadIdx.x + 256U * m;
                xs[(chunk + 1U) & 1U][e / 32U][e % 32U] = next[m];
            }
        }
        __syncthreads();
    }
    if (!active) return;
    const unsigned b = lane;
    const unsigned r = 8U * ((b >> 4U) & 1U) + 4U * ((b >> 3U) & 1U) + 2U * ((b >> 2U) & 1U) + ((b >> 1U) & 1U);
    const unsigned out_base = first + 4U * (b & 1U);
    if (row0 + r >= rows) return;
#pragma unroll
    for (unsigned i = 0; i < 4U; ++i) {
        const unsigned o = out_base + i;
        if (o >= total_outputs) continue;
        const float scaled = __fdiv_rn(total[i], static_cast<float>(kStreams));
        const float sig = amd_sigmoid(t, scaled);
        if (is_inject) injection[static_cast<std::size_t>(row0 + r) * kStreams + o] = __fadd_rn(sig, sig);
        else low_rows[static_cast<std::size_t>(row0 + r) * kNormalizedPitch + kWide + o] = __fmul_rn(scaled, sig);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// [O] wide2_hyper_output_kernel<u16, 32, true> (hip_hyper_mix_launcher.hip:3193-3268). Thread = one output feature f,
// 32 rows x 4 streams gate chains g = fma(low[k], c_s(k), g), k = 0..319 ascending from 0; then
// mixed = fma(n3, sig3, fma(n2, sig2, fma(n1, sig1, fma(n0, sig0, 0)))) and bf16(mixed * 0.25).
// The 32 rows' low values sit in shared memory (HIP reads a transposed copy low_t through scalar loads; the values
// are the same FP32 lows, so this kernel has no 8192-row pitch limit: above 8192 rows HIP switches to
// wide_lane_chain_hyper_output_kernel<u16,32> (:3075-3180), whose ISA (hip_hyper_mix_launcher.s:355815) also
// contracts `gate += low * c` to v_fmac_f32 and the mix to v_fma(.., 0)/v_fmac: the same DAG, so no change).
constexpr unsigned kOutRows = 32U, kOutThreads = 128U;

template <bool Fused, bool Compact = false>
__global__ __launch_bounds__(128) void hc_output_kernel(const float* __restrict__ normalized, std::uint32_t rows,
    const std::uint32_t* __restrict__ up_codes, const float* __restrict__ up_scales, std::uint16_t* __restrict__ output,
    float* __restrict__ gates, am::Tables t, const std::uint16_t* __restrict__ xin = nullptr, const float* __restrict__ xnorm = nullptr) {
    __shared__ __align__(16) float lows[kRank][kOutRows];
    const unsigned row0 = blockIdx.x * kOutRows, f = blockIdx.y * kOutThreads + threadIdx.x;
    for (unsigned idx = threadIdx.x; idx < kRank * kOutRows; idx += kOutThreads) {
        const unsigned r = idx / kRank, k = idx % kRank;
        lows[k][r] = row0 + r < rows ? normalized[static_cast<std::size_t>(row0 + r) * kNormalizedPitch + kWide + k] : 0.0F;
    }
    __syncthreads();
    float g[kStreams][kOutRows];
#pragma unroll
    for (unsigned s = 0; s < kStreams; ++s)
#pragma unroll
        for (unsigned r = 0; r < kOutRows; ++r) g[s][r] = 0.0F;
    float scale_next[kStreams];
    std::uint32_t word_next[kStreams];
    const auto load_q = [&](unsigned q) {
#pragma unroll
        for (unsigned s = 0; s < kStreams; ++s) {
            scale_next[s] = up_scales[(static_cast<std::size_t>(q / 8U) * kStreams + s) * kHidden + f];
            word_next[s] = up_codes[(static_cast<std::size_t>(q) * kStreams + s) * kHidden + f];
        }
    };
    load_q(0U);
#pragma unroll 1
    for (unsigned q = 0; q < kRank / 4U; ++q) {
        float scale[kStreams];
        std::uint32_t word[kStreams];
#pragma unroll
        for (unsigned s = 0; s < kStreams; ++s) { scale[s] = scale_next[s]; word[s] = word_next[s]; }
        if (q + 1U < kRank / 4U) load_q(q + 1U);  // one k-quad ahead
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            float c[kStreams];
#pragma unroll
            for (unsigned s = 0; s < kStreams; ++s)  // scale * (float(code ^ 0x80) - 128) == scale * float(int8), exact
                c[s] = __fmul_rn(scale[s], static_cast<float>(static_cast<signed char>((word[s] >> (8U * j)) & 255U)));
            const float4* low = reinterpret_cast<const float4*>(lows[4U * q + j]);
#pragma unroll
            for (unsigned r4 = 0; r4 < kOutRows / 4U; ++r4) {
                const float4 v = low[r4];
#pragma unroll
                for (unsigned s = 0; s < kStreams; ++s) {
                    g[s][4U * r4 + 0U] = __fmaf_rn(v.x, c[s], g[s][4U * r4 + 0U]);
                    g[s][4U * r4 + 1U] = __fmaf_rn(v.y, c[s], g[s][4U * r4 + 1U]);
                    g[s][4U * r4 + 2U] = __fmaf_rn(v.z, c[s], g[s][4U * r4 + 2U]);
                    g[s][4U * r4 + 3U] = __fmaf_rn(v.w, c[s], g[s][4U * r4 + 3U]);
                }
            }
        }
    }
    if constexpr (!Fused) {
#pragma unroll
        for (unsigned r = 0; r < kOutRows; ++r) {
            if (row0 + r >= rows) break;
#pragma unroll
            for (unsigned s = 0; s < kStreams; ++s) gates[(static_cast<std::size_t>(row0 + r) * kStreams + s) * kHidden + f] = g[s][r];
        }
        return;
    }
#pragma unroll
    for (unsigned r = 0; r < kOutRows; ++r) {
        if (row0 + r >= rows) break;
        const float* n = normalized + static_cast<std::size_t>(row0 + r) * kNormalizedPitch + f;
        float mixed = 0.0F;
#pragma unroll
        for (unsigned s = 0; s < kStreams; ++s) {
            float nv;
            if constexpr (Compact)   // bit-identical rebuild (norm_value); the inverse sits at normalized[row][s]
                nv = norm_value(xin[static_cast<std::size_t>(row0 + r) * kWide + s * kHidden + f],
                                normalized[static_cast<std::size_t>(row0 + r) * kNormalizedPitch + s], xnorm[s * kHidden + f]);
            else nv = n[s * kHidden];
            mixed = __fmaf_rn(nv, amd_sigmoid(t, g[s][r]), mixed);
        }
        output[static_cast<std::size_t>(row0 + r) * kHidden + f] = float_to_bf16(__fmul_rn(mixed, 0.25F));
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// FAST forms (Config::forms.hc_mma): the [P] and [O] GEMMs on FP16 tensor cores (core/mma_tile.cuh: perm8 K order, int8 codes
// exact in FP16, FP32 accumulation, per-32-K FP32 scale epilogue). Activations x = normalized values (|x| <= sqrt(2560) * |norm
// weight|) and low = s * sigmoid(s) are far inside the FP16 range, so FP16 (not BF16) is used for the 10-bit mantissa. The
// FP32 inject weights are rounded to FP16 (the Q8 weights are not rounded). Everything that is not the GEMM (the /4, sigmoid,
// mix, 0.25, bf16) is the exact kernels' FP32 code.
namespace cm = strix::cuda::core::mma;
namespace cp = strix::cuda::core::ptx;

// FP32 sigmoid for the FAST form: the exact kernels emulate the HIP expf/division bit for bit (amd_sigmoid, ~10x the
// instructions); this is 1 / (1 + expf(-x)) with CUDA's FP32 expf (~1-2 ulp) and a correctly rounded reciprocal.
__device__ __forceinline__ float fast_sigmoid(float x) { return __frcp_rn(__fadd_rn(1.0F, expf(-x))); }

constexpr unsigned kMmaPitch = 64U + cm::kPadHalves;   // A smem row pitch (halves), chunk = 64 K = 2 scale blocks
constexpr unsigned kMmaCodePitch = 80U;                // bytes per weight row in smem (64 B + 16 pad)
constexpr unsigned kMmaOutRows = 64U;                  // rows per CTA of the output kernel (4 m16 tiles)

// MT m16 tiles (16 * MT rows) x (NW * 40) low outputs per CTA; NW warps of 5 n8 tiles each. gridDim.y = 8 / NW splits of the
// 320 outputs (x is read once per split). Warps also own the inject n tile (4 outputs) of the m tiles w, w + NW, ... (split 0).
template <unsigned MT, unsigned NW>
struct ProjCfg {
    static constexpr unsigned kRowsCta = 16U * MT, kThreads = NW * 32U, kOutCta = NW * 40U;
    static constexpr unsigned kSmem = 2U * kRowsCta * kMmaPitch * 2U + 2U * kOutCta * kMmaCodePitch + 2U * kOutCta * 8U;
};

template <unsigned MT, unsigned NW, bool Compact = false>
__global__ __launch_bounds__(NW * 32U, 1) void hc_projection_mma_kernel(const float* x_rows, std::uint32_t rows,
    const std::int8_t* __restrict__ down_codes, const float* __restrict__ down_scales, const float* __restrict__ inject,
    float* low_rows, float* __restrict__ injection, am::Tables t_, const std::uint16_t* __restrict__ xin,
    const float* __restrict__ xnorm) {
    using Cfg = ProjCfg<MT, NW>;
    constexpr unsigned kT = Cfg::kThreads, kBM = Cfg::kRowsCta, kNR = Cfg::kOutCta;
    constexpr unsigned kXI = (kBM * 16U + kT - 1U) / kT, kSI = (kNR + kT - 1U) / kT, kIK = (MT + NW - 1U) / NW;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* a_s = reinterpret_cast<__half*>(smem_raw);                                        // [2][BM][72]
    unsigned char* c_s = smem_raw + 2U * kBM * kMmaPitch * 2U;                                // [2][NR][80]
    float2* s_s = reinterpret_cast<float2*>(c_s + 2U * kNR * kMmaCodePitch);                  // [2][NR]
    const unsigned tid = threadIdx.x, warp = tid >> 5U, lane = tid & 31U, g = lane >> 2U, t = lane & 3U;
    const unsigned row0 = blockIdx.x * kBM, nbase = blockIdx.y * kNR;
    constexpr unsigned kChunks = kWide / 64U;

    float4 xr[kXI];
    float2 sr[kSI];
    const auto load_x = [&](unsigned c) {
#pragma unroll
        for (unsigned i = 0; i < kXI; ++i) {
            const unsigned e = tid + kT * i, r = e >> 4U, m4 = e & 15U;
            xr[i] = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            if (e < kBM * 16U && row0 + r < rows) {
                xr[i] = load_x4<Compact>(x_rows, xin, xnorm, row0 + r, c * 64U + m4 * 4U);   // 64-chunks stay in one stream
            }
        }
#pragma unroll
        for (unsigned i = 0; i < kSI; ++i) {
            const unsigned o = tid + kT * i;
            if (o < kNR) sr[i] = *reinterpret_cast<const float2*>(down_scales + static_cast<std::size_t>(nbase + o) * (kWide / 32U) + c * 2U);
        }
    };
    const auto store_x = [&](unsigned stage) {
#pragma unroll
        for (unsigned i = 0; i < kXI; ++i) {
            const unsigned e = tid + kT * i, r = e >> 4U, m4 = e & 15U;
            if (e < kBM * 16U) cm::stage_k4(a_s + (stage * kBM + r) * kMmaPitch + (m4 >> 3U) * 32U, m4 & 7U, xr[i]);
        }
#pragma unroll
        for (unsigned i = 0; i < kSI; ++i) {
            const unsigned o = tid + kT * i;
            if (o < kNR) s_s[stage * kNR + o] = sr[i];
        }
    };
    const auto issue_codes = [&](unsigned c, unsigned stage) {
#pragma unroll
        for (unsigned e = tid; e < kNR * 4U; e += kT) {
            const unsigned n = e >> 2U, q = e & 3U;
            cp::cp_async16(c_s + stage * kNR * kMmaCodePitch + n * kMmaCodePitch + q * 16U,
                down_codes + static_cast<std::size_t>(nbase + n) * kWide + c * 64U + q * 16U);
        }
        cp::cp_async_commit();
    };

    float acc[5][MT][4];  // [n tile][m tile][frag]
    float ai[kIK][4];
#pragma unroll
    for (unsigned m = 0; m < MT; ++m)
#pragma unroll
        for (unsigned n = 0; n < 5U; ++n)
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i) acc[n][m][i] = 0.0F;
#pragma unroll
    for (unsigned k = 0; k < kIK; ++k)
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) ai[k][i] = 0.0F;
    const bool inj_on = inject != nullptr && blockIdx.y == 0;
    float iw[2][8];  // inject weights (FP32) of the next chunk's 2 blocks: row g (< 4), k 8t .. 8t+7 of each block
    const auto load_inj = [&](unsigned c) {
#pragma unroll
        for (unsigned b = 0; b < 2U; ++b) {
            float4 v0 = make_float4(0.0F, 0.0F, 0.0F, 0.0F), v1 = v0;
            if (g < kStreams) {
                const float* p = inject + static_cast<std::size_t>(g) * kWide + c * 64U + b * 32U + 8U * t;
                v0 = *reinterpret_cast<const float4*>(p); v1 = *reinterpret_cast<const float4*>(p + 4);
            }
            iw[b][0] = v0.x; iw[b][1] = v0.y; iw[b][2] = v0.z; iw[b][3] = v0.w;
            iw[b][4] = v1.x; iw[b][5] = v1.y; iw[b][6] = v1.z; iw[b][7] = v1.w;
        }
    };

    load_x(0U);
    issue_codes(0U, 0U);
    if (inj_on && warp < MT) load_inj(0U);
    store_x(0U);
    cp::cp_async_wait<0>();
    __syncthreads();
    for (unsigned c = 0; c < kChunks; ++c) {
        const unsigned stage = c & 1U;
        if (c + 1U < kChunks) {
            issue_codes(c + 1U, stage ^ 1U);
            load_x(c + 1U);
        }
#pragma unroll
        for (unsigned b = 0; b < 2U; ++b) {
            unsigned a[MT][2][4];
#pragma unroll
            for (unsigned m = 0; m < MT; ++m)
#pragma unroll
                for (unsigned j = 0; j < 2U; ++j)
                    cm::ldsm_a(a[m][j], a_s + (stage * kBM + m * 16U) * kMmaPitch + b * 32U + j * 16U, kMmaPitch, lane);
#pragma unroll
            for (unsigned n = 0; n < 5U; ++n) {
                const unsigned nt = warp * 5U + n;
                unsigned w[2];
                const uint2 raw = *reinterpret_cast<const uint2*>(c_s + stage * kNR * kMmaCodePitch +
                    (nt * 8U + g) * kMmaCodePitch + b * 32U + 8U * t);
                w[0] = raw.x; w[1] = raw.y;
                const float2 q0 = s_s[stage * kNR + nt * 8U + 2U * t], q1 = s_s[stage * kNR + nt * 8U + 2U * t + 1U];
                const float s0 = b ? q0.y : q0.x, s1 = b ? q1.y : q1.x;
                cm::warp_gemm_k32<MT, cm::Int8Codes>(acc[n], a, w, s0, s1);
            }
        }
        if (inj_on) {
            float cur[2][8];
#pragma unroll
            for (unsigned b = 0; b < 2U; ++b)
#pragma unroll
                for (unsigned i = 0; i < 8U; ++i) cur[b][i] = iw[b][i];
            if (warp < MT && c + 1U < kChunks) load_inj(c + 1U);
#pragma unroll
            for (unsigned k = 0; k < kIK; ++k) {
                const unsigned m = warp + k * NW;
                if (m < MT) {
#pragma unroll
                    for (unsigned b = 0; b < 2U; ++b)
#pragma unroll
                        for (unsigned j = 0; j < 2U; ++j) {
                            unsigned a[4];
                            cm::ldsm_a(a, a_s + (stage * kBM + m * 16U) * kMmaPitch + b * 32U + j * 16U, kMmaPitch, lane);
                            __half2 h0 = __floats2half2_rn(cur[b][4 * j], cur[b][4 * j + 1]);
                            __half2 h1 = __floats2half2_rn(cur[b][4 * j + 2], cur[b][4 * j + 3]);
                            cp::mma_f16(ai[k], a[0], a[1], a[2], a[3], *reinterpret_cast<unsigned*>(&h0), *reinterpret_cast<unsigned*>(&h1));
                        }
                }
            }
        }
        if (c + 1U < kChunks) {
            store_x(stage ^ 1U);
            cp::cp_async_wait<0>();
        }
        __syncthreads();
    }
    // Epilogue (exact kernels' FP32 code): scaled = total / 4, sigmoid.
    const auto finish = [&](float v0, float v1, unsigned row, unsigned o, bool inj) {
        if (row >= rows) return;
        const float s0 = __fdiv_rn(v0, static_cast<float>(kStreams)), s1 = __fdiv_rn(v1, static_cast<float>(kStreams));
        const float g0 = fast_sigmoid(s0), g1 = fast_sigmoid(s1);
        if (inj) {
            if (o < kStreams) {
                injection[static_cast<std::size_t>(row) * kStreams + o] = __fadd_rn(g0, g0);
                injection[static_cast<std::size_t>(row) * kStreams + o + 1U] = __fadd_rn(g1, g1);
            }
        } else {
            *reinterpret_cast<float2*>(low_rows + static_cast<std::size_t>(row) * kNormalizedPitch + kWide + o) =
                make_float2(__fmul_rn(s0, g0), __fmul_rn(s1, g1));
        }
    };
#pragma unroll
    for (unsigned m = 0; m < MT; ++m)
#pragma unroll
        for (unsigned n = 0; n < 5U; ++n) {
            const unsigned o = nbase + (warp * 5U + n) * 8U + 2U * t;
            finish(acc[n][m][0], acc[n][m][1], row0 + m * 16U + g, o, false);
            finish(acc[n][m][2], acc[n][m][3], row0 + m * 16U + g + 8U, o, false);
        }
    if (inj_on) {
#pragma unroll
        for (unsigned k = 0; k < kIK; ++k) {
            const unsigned m = warp + k * NW;
            if (m < MT) {
                finish(ai[k][0], ai[k][1], row0 + m * 16U + g, 2U * t, true);
                finish(ai[k][2], ai[k][3], row0 + m * 16U + g + 8U, 2U * t, true);
            }
        }
    }
}

template <unsigned MT, unsigned NW, bool Compact>
void launch_projection_mma_c(const HcMixArgs& a, const am::Tables& tables, cudaStream_t stream) {
    using Cfg = ProjCfg<MT, NW>;
    static const bool attr = [] {
        check(cudaFuncSetAttribute(hc_projection_mma_kernel<MT, NW, Compact>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(Cfg::kSmem)), "mma projection attr");
        return true;
    }();
    (void)attr;
    hc_projection_mma_kernel<MT, NW, Compact><<<dim3((a.rows + Cfg::kRowsCta - 1U) / Cfg::kRowsCta, 8U / NW), Cfg::kThreads, Cfg::kSmem, stream>>>(
        a.normalized, a.rows, a.packed.down_codes, a.packed.down_scales, a.inject, a.normalized, a.injection, tables, a.input, a.norm);
}
template <unsigned MT, unsigned NW>
void launch_projection_mma(const HcMixArgs& a, const am::Tables& tables, cudaStream_t stream, bool compact) {
    if (compact) launch_projection_mma_c<MT, NW, true>(a, tables, stream);
    else launch_projection_mma_c<MT, NW, false>(a, tables, stream);
}
// Tile shape per row count: fill the 188 SMs (one wave, no tail) while keeping the weight re-reads (per row tile) low.
void projection_mma(const HcMixArgs& a, const am::Tables& tables, cudaStream_t stream, bool compact) {
    struct Opt { unsigned mt, nw; };
    static const Opt opts[] = {{4U, 8U}, {3U, 8U}, {2U, 8U}, {4U, 4U}, {2U, 4U}, {1U, 4U}};
    unsigned best = 0; double best_cost = 1e30;
    for (unsigned i = 0; i < 6U; ++i) {
        const double tiles = std::ceil(a.rows / (16.0 * opts[i].mt)) * (8U / opts[i].nw);
        const double res = opts[i].nw == 8U ? 1.0 : 2.0;  // resident CTAs per SM (smem)
        const double waves = std::ceil(tiles / (188.0 * res));
        // time ~ waves x CTA work; CTA work ~ (rows per CTA) x (outputs per CTA) / throughput (+ weight stream per row tile)
        const double work = (16.0 * opts[i].mt) * (opts[i].nw * 40.0) + 640.0 * (opts[i].nw * 40.0) / 8.0 / opts[i].mt;
        const double cost = waves * work / res;
        if (cost < best_cost) { best_cost = cost; best = i; }
    }
    switch (best) {
    case 0: launch_projection_mma<4U, 8U>(a, tables, stream, compact); break;
    case 1: launch_projection_mma<3U, 8U>(a, tables, stream, compact); break;
    case 2: launch_projection_mma<2U, 8U>(a, tables, stream, compact); break;
    case 3: launch_projection_mma<4U, 4U>(a, tables, stream, compact); break;
    case 4: launch_projection_mma<2U, 4U>(a, tables, stream, compact); break;
    default: launch_projection_mma<1U, 4U>(a, tables, stream, compact); break;
    }
}

// [O] up-projection + mix. CTA = 4 warps x (8 features x 4 streams) per pass, kMmaOutPasses passes over features; A = the
// CTA's 64 rows of low (K = 320) staged once as FP16.
constexpr unsigned kMmaOutPasses = 5U, kMmaOutPitch = kRank + cm::kPadHalves;
constexpr unsigned kMmaOutSmem = kMmaOutRows * kMmaOutPitch * 2U;

template <bool Compact = false>
__global__ __launch_bounds__(128) void hc_output_mma_kernel(const float* __restrict__ normalized, std::uint32_t rows,
    const std::uint32_t* __restrict__ up_codes, const float* __restrict__ up_scales, std::uint16_t* __restrict__ output,
    float* __restrict__ gates, am::Tables t_,
    const std::uint16_t* __restrict__ xin, const float* __restrict__ xnorm) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* a_s = reinterpret_cast<__half*>(smem_raw);  // [64][328]
    const unsigned tid = threadIdx.x, warp = tid >> 5U, lane = tid & 31U, g = lane >> 2U, t = lane & 3U;
    const unsigned row0 = blockIdx.x * kMmaOutRows;
    for (unsigned e0 = 0; e0 < kMmaOutRows * (kRank / 4U); e0 += 128U * 8U) {  // 8 loads in flight, then the stores
        float4 v[8];
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) {
            const unsigned e = e0 + tid + 128U * i, r = e / (kRank / 4U), m4 = e % (kRank / 4U);
            v[i] = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            if (row0 + r < rows) v[i] = *reinterpret_cast<const float4*>(normalized + static_cast<std::size_t>(row0 + r) * kNormalizedPitch + kWide + m4 * 4U);
        }
#pragma unroll
        for (unsigned i = 0; i < 8U; ++i) {
            const unsigned e = e0 + tid + 128U * i, r = e / (kRank / 4U), m4 = e % (kRank / 4U);
            cm::stage_k4(a_s + r * kMmaOutPitch + (m4 >> 3U) * 32U, m4 & 7U, v[i]);
        }
    }
    __syncthreads();
    for (unsigned pass = 0; pass < kMmaOutPasses; ++pass) {
        const unsigned f0 = (blockIdx.y * kMmaOutPasses + pass) * 32U + warp * 8U;
        float acc[kStreams][4][4];  // [stream][m tile][frag]
#pragma unroll
        for (unsigned s = 0; s < kStreams; ++s)
#pragma unroll
            for (unsigned m = 0; m < 4U; ++m)
#pragma unroll
                for (unsigned i = 0; i < 4U; ++i) acc[s][m][i] = 0.0F;
        unsigned wn[kStreams][2];
        float2 sn[kStreams];
        const auto load_b = [&](unsigned blk) {
#pragma unroll
            for (unsigned s = 0; s < kStreams; ++s) {
                wn[s][0] = up_codes[(static_cast<std::size_t>(8U * blk + 2U * t) * kStreams + s) * kHidden + f0 + g];
                wn[s][1] = up_codes[(static_cast<std::size_t>(8U * blk + 2U * t + 1U) * kStreams + s) * kHidden + f0 + g];
                sn[s] = *reinterpret_cast<const float2*>(up_scales + (static_cast<std::size_t>(blk) * kStreams + s) * kHidden + f0 + 2U * t);
            }
        };
        load_b(0U);
#pragma unroll
        for (unsigned blk = 0; blk < kRank / 32U; ++blk) {
            unsigned w[kStreams][2];
            float2 sc[kStreams];
#pragma unroll
            for (unsigned s = 0; s < kStreams; ++s) { w[s][0] = wn[s][0]; w[s][1] = wn[s][1]; sc[s] = sn[s]; }
            if (blk + 1U < kRank / 32U) load_b(blk + 1U);
            unsigned a[4][2][4];
#pragma unroll
            for (unsigned m = 0; m < 4U; ++m)
#pragma unroll
                for (unsigned j = 0; j < 2U; ++j)
                    cm::ldsm_a(a[m][j], a_s + (m * 16U) * kMmaOutPitch + blk * 32U + j * 16U, kMmaOutPitch, lane);
#pragma unroll
            for (unsigned s = 0; s < kStreams; ++s) cm::warp_gemm_k32<4U, cm::Int8Codes>(acc[s], a, w[s], sc[s].x, sc[s].y);
        }
        const unsigned f = f0 + 2U * t;
#pragma unroll
        for (unsigned m = 0; m < 4U; ++m)
#pragma unroll
            for (unsigned h = 0; h < 2U; ++h) {
                const unsigned row_raw = row0 + m * 16U + g + 8U * h, row = row_raw < rows ? row_raw : rows - 1U;  // clamp: loads stay in flight
                if (gates) {
                    if (row_raw >= rows) continue;
#pragma unroll
                    for (unsigned s = 0; s < kStreams; ++s)
                        *reinterpret_cast<float2*>(gates + (static_cast<std::size_t>(row) * kStreams + s) * kHidden + f) =
                            make_float2(acc[s][m][2U * h], acc[s][m][2U * h + 1U]);
                    continue;
                }
                float mixed0 = 0.0F, mixed1 = 0.0F;
#pragma unroll
                for (unsigned s = 0; s < kStreams; ++s) {
                    float2 n;
                    if constexpr (Compact) {   // bit-identical rebuild of the normalized pair (norm_value)
                        const std::uint32_t xv = *reinterpret_cast<const std::uint32_t*>(xin + static_cast<std::size_t>(row) * kWide + s * kHidden + f);
                        const float inv = normalized[static_cast<std::size_t>(row) * kNormalizedPitch + s];
                        const float2 w = *reinterpret_cast<const float2*>(xnorm + s * kHidden + f);
                        n = make_float2(norm_value(static_cast<std::uint16_t>(xv), inv, w.x), norm_value(static_cast<std::uint16_t>(xv >> 16U), inv, w.y));
                    } else {
                        n = *reinterpret_cast<const float2*>(normalized + static_cast<std::size_t>(row) * kNormalizedPitch + s * kHidden + f);
                    }
                    mixed0 = __fmaf_rn(n.x, fast_sigmoid(acc[s][m][2U * h]), mixed0);
                    mixed1 = __fmaf_rn(n.y, fast_sigmoid(acc[s][m][2U * h + 1U]), mixed1);
                }
                const std::uint32_t packed = static_cast<std::uint32_t>(float_to_bf16(__fmul_rn(mixed0, 0.25F))) |
                    (static_cast<std::uint32_t>(float_to_bf16(__fmul_rn(mixed1, 0.25F))) << 16U);
                if (row_raw < rows) *reinterpret_cast<std::uint32_t*>(output + static_cast<std::size_t>(row) * kHidden + f) = packed;
            }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Decode shapes (rows <= 8): the [P], [O] and router DAGs above, laid out for few rows.
// STRIX_QRTX_DECODE_KERNELS=0 keeps the tiled kernels (reference seam).
bool decode_rows_enabled() {
    static const bool on = strix::envcfg::flag("STRIX_QRTX_DECODE_KERNELS", true);
    return on;
}

// [P] for R rows: warp w = tile w (16 warps). Lane l keeps the leaf chain over s = 0..4, j = 0..3 of feature
// 640 w + 128 s + 4 l + j; the xor butterfly 16..1 gives tile(w) (every lane the same sum: IEEE addition is
// commutative); thread (o, r) then adds the tiles in order from 0. A block covers OG low outputs (or the 4 injection
// outputs, last block) of RG rows. OG = 1, RG = R (one output, all rows) was the only form; with STRIX_QRTX_HC_ROWS_SPLIT
// (default) OG/RG follow R so the FP32 rows (R x 40 KB) are read by 320/OG blocks instead of 320 (L2 traffic / OG)
// while the grid stays >= 160 blocks. Every (row, output) chain, butterfly and tile sum is unchanged: bit-identical.
template <unsigned R, unsigned OG, unsigned RG>
__global__ __launch_bounds__(512) void hc_projection_rows_kernel(const float* x_rows, const std::int8_t* __restrict__ down_codes,
    const float* __restrict__ down_scales, const float* __restrict__ inject, float* low_rows, float* __restrict__ injection,
    am::Tables t) {
    constexpr unsigned OT = OG > 4U ? OG : 4U;
    __shared__ float tiles[16][OT][RG];
    const unsigned warp = threadIdx.x / 32U, lane = threadIdx.x % 32U;
    const bool is_inject = blockIdx.x >= kRank / OG;
    const unsigned outputs = is_inject ? kStreams : OG, r0 = blockIdx.y * RG;
    float acc[OT][RG];
#pragma unroll
    for (unsigned o = 0; o < OT; ++o)
#pragma unroll
        for (unsigned r = 0; r < RG; ++r) acc[o][r] = 0.0F;
#pragma unroll
    for (unsigned sub = 0; sub < 5U; ++sub) {
        const unsigned f = 640U * warp + 128U * sub + 4U * lane;
        float x[RG][4];
#pragma unroll
        for (unsigned r = 0; r < RG; ++r) {
            const float4 v = r0 + r < R ? *reinterpret_cast<const float4*>(x_rows + static_cast<std::size_t>(r0 + r) * kNormalizedPitch + f)
                                        : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            x[r][0] = v.x; x[r][1] = v.y; x[r][2] = v.z; x[r][3] = v.w;
        }
        if (is_inject) {
#pragma unroll
            for (unsigned o = 0; o < 4U; ++o) {
                const float4 w = *reinterpret_cast<const float4*>(inject + static_cast<std::size_t>(o) * kWide + f);
                const float c[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j)
#pragma unroll
                    for (unsigned r = 0; r < RG; ++r) acc[o][r] = __fmaf_rn(x[r][j], c[j], acc[o][r]);
            }
        } else {
#pragma unroll
            for (unsigned oo = 0; oo < OG; ++oo) {
                const unsigned o = blockIdx.x * OG + oo;
                const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(down_codes + static_cast<std::size_t>(o) * kWide + f);
                const float scale = down_scales[static_cast<std::size_t>(o) * (kWide / 32U) + f / 32U];
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j) {
                    const float c = __fmul_rn(scale, static_cast<float>(static_cast<signed char>((word >> (8U * j)) & 255U)));
#pragma unroll
                    for (unsigned r = 0; r < RG; ++r) acc[oo][r] = __fmaf_rn(x[r][j], c, acc[oo][r]);
                }
            }
        }
    }
#pragma unroll
    for (unsigned o = 0; o < OT; ++o) {
        if (o >= outputs) break;
#pragma unroll
        for (unsigned r = 0; r < RG; ++r) {
            float v = acc[o][r];
#pragma unroll
            for (unsigned d = 16U; d > 0U; d >>= 1U) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffU, v, d));
            if (lane == 0U) tiles[warp][o][r] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x >= outputs * RG) return;
    const unsigned o = threadIdx.x / RG, r = r0 + threadIdx.x % RG;
    if (r >= R) return;
    float total = 0.0F;
#pragma unroll
    for (unsigned w = 0; w < 16U; ++w) total = __fadd_rn(total, tiles[w][o][r - r0]);
    const float scaled = __fdiv_rn(total, static_cast<float>(kStreams));
    const float sig = amd_sigmoid(t, scaled);
    if (is_inject) injection[static_cast<std::size_t>(r) * kStreams + o] = __fadd_rn(sig, sig);
    else low_rows[static_cast<std::size_t>(r) * kNormalizedPitch + kWide + blockIdx.x * OG + o] = __fmul_rn(scaled, sig);
}

// [O] for R rows: thread = (feature f, stream s), lane = 4 f' + s; the 4 streams of a feature are adjacent lanes, so
// the stream mix fma(n_s, sig_s, ..) s ascending from 0 is done by lane s = 0 after shuffles. A block covers RG rows
// (blockIdx.y). RG = R was the only form (80 blocks of 4 warps: 1.7 warps per SM); STRIX_QRTX_HC_ROWS_SPLIT (default)
// uses RG = 1..2 (320-640 blocks). Same chains: bit-identical.
template <unsigned R, unsigned RG>
__global__ __launch_bounds__(128) void hc_output_rows_kernel(const float* __restrict__ normalized,
    const std::uint32_t* __restrict__ up_codes, const float* __restrict__ up_scales, std::uint16_t* __restrict__ output,
    am::Tables t) {
    __shared__ float lows[kRank][RG];
    const unsigned r0 = blockIdx.y * RG;
    for (unsigned i = threadIdx.x; i < kRank * RG; i += blockDim.x)
        lows[i / RG][i % RG] =
            r0 + i % RG < R ? normalized[static_cast<std::size_t>(r0 + i % RG) * kNormalizedPitch + kWide + i / RG] : 0.0F;
    __syncthreads();
    const unsigned s = threadIdx.x % 4U, f = blockIdx.x * (blockDim.x / 4U) + threadIdx.x / 4U;
    float g[RG];
#pragma unroll
    for (unsigned r = 0; r < RG; ++r) g[r] = 0.0F;
#pragma unroll 4
    for (unsigned q = 0; q < kRank / 4U; ++q) {
        const float scale = up_scales[(static_cast<std::size_t>(q / 8U) * kStreams + s) * kHidden + f];
        const std::uint32_t word = up_codes[(static_cast<std::size_t>(q) * kStreams + s) * kHidden + f];
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            const float c = __fmul_rn(scale, static_cast<float>(static_cast<signed char>((word >> (8U * j)) & 255U)));
#pragma unroll
            for (unsigned r = 0; r < RG; ++r) g[r] = __fmaf_rn(lows[4U * q + j][r], c, g[r]);
        }
    }
    const unsigned base = (threadIdx.x % 32U) & ~3U;
#pragma unroll
    for (unsigned rr = 0; rr < RG; ++rr) {
        if (r0 + rr >= R) break;
        const unsigned r = r0 + rr;
        const float term = amd_sigmoid(t, g[rr]);
        const float n = normalized[static_cast<std::size_t>(r) * kNormalizedPitch + s * kHidden + f];
        float sig[4], nv[4];
#pragma unroll
        for (unsigned k = 0; k < 4U; ++k) {
            sig[k] = __shfl_sync(0xffffffffU, term, base + k);
            nv[k] = __shfl_sync(0xffffffffU, n, base + k);
        }
        if (s == 0U) {
            float mixed = 0.0F;
#pragma unroll
            for (unsigned k = 0; k < 4U; ++k) mixed = __fmaf_rn(nv[k], sig[k], mixed);
            output[static_cast<std::size_t>(r) * kHidden + f] = float_to_bf16(__fmul_rn(mixed, 0.25F));
        }
    }
}

// Split epilogue: the same stream mix per (row, feature) from the stored gates.
template <bool Fast>
__global__ __launch_bounds__(256) void hc_mix_epilogue_kernel(const float* __restrict__ normalized, const float* __restrict__ gates,
    std::uint32_t rows, std::uint16_t* __restrict__ output, am::Tables t) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::size_t>(rows) * kHidden) return;
    const std::size_t row = i / kHidden, f = i % kHidden;
    const float* n = normalized + row * kNormalizedPitch + f;
    const float* g = gates + row * kStreams * kHidden + f;
    float mixed = 0.0F;
#pragma unroll
    for (unsigned s = 0; s < kStreams; ++s) {
        if constexpr (Fast) mixed = __fmaf_rn(n[s * kHidden], fast_sigmoid(g[s * kHidden]), mixed);
        else mixed = __fmaf_rn(n[s * kHidden], amd_sigmoid(t, g[s * kHidden]), mixed);
    }
    output[i] = float_to_bf16(__fmul_rn(mixed, 0.25F));
}

// ---------------------------------------------------------------------------------------------------------------------
// Elementwise.
__global__ void embedding_kernel(const unsigned char* __restrict__ table, const std::uint32_t* __restrict__ ids,
    std::uint16_t* __restrict__ out) {
    const unsigned row = blockIdx.x;
    const std::size_t token = ids[row];
    for (unsigned f = threadIdx.x; f < kHidden; f += blockDim.x) {
        const unsigned char* block = table + (token * (kHidden / 32U) + f / 32U) * 34U;
        const float scale = half_bits_to_float(static_cast<std::uint16_t>(block[0] | (block[1] << 8U)));
        out[static_cast<std::size_t>(row) * kHidden + f] =
            float_to_bf16(__fmul_rn(scale, static_cast<float>(weight::Q8Block::code(block, f % 32U))));
    }
}
__global__ void expand_kernel(const std::uint16_t* __restrict__ in, std::uint16_t* __restrict__ out, std::size_t total) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const std::size_t row = i / kWide, f = i % kHidden;
    out[i] = float_to_bf16(bf16_to_float(in[row * kHidden + f]));  // BF16 -> FP32 -> BF16: the identity
}
// Same per-element arithmetic as inject_kernel, 8 elements (16 B) per thread, row = blockIdx.y (lever qwen_hc_inject_vec,
// bitwise: one fma + one BF16 rounding per element as before; 2560 % 8 == 0 so the 8 elements share one stream).
__global__ __launch_bounds__(256) void inject_vec_kernel(const std::uint16_t* __restrict__ hyper, const std::uint16_t* __restrict__ hidden,
    const float* __restrict__ weights, std::uint16_t* __restrict__ out) {
    const std::size_t row = blockIdx.y;
    const unsigned within = (blockIdx.x * 256U + threadIdx.x) * 8U, s = within / kHidden, f = within % kHidden;
    if (within >= kWide) return;
    const uint4 h = *reinterpret_cast<const uint4*>(hyper + row * kWide + within);
    const uint4 v = *reinterpret_cast<const uint4*>(hidden + row * kHidden + f);
    const float w = weights[row * kStreams + s];
    const unsigned hw[4] = {h.x, h.y, h.z, h.w}, vw[4] = {v.x, v.y, v.z, v.w};
    unsigned o[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const std::uint16_t lo = float_to_bf16(__fmaf_rn(bf16_to_float(static_cast<std::uint16_t>(vw[k])), w, bf16_to_float(static_cast<std::uint16_t>(hw[k]))));
        const std::uint16_t hi = float_to_bf16(__fmaf_rn(bf16_to_float(static_cast<std::uint16_t>(vw[k] >> 16U)), w, bf16_to_float(static_cast<std::uint16_t>(hw[k] >> 16U))));
        o[k] = static_cast<unsigned>(lo) | (static_cast<unsigned>(hi) << 16U);
    }
    *reinterpret_cast<uint4*>(out + row * kWide + within) = make_uint4(o[0], o[1], o[2], o[3]);
}
__global__ void inject_kernel(const std::uint16_t* __restrict__ hyper, const std::uint16_t* __restrict__ hidden,
    const float* __restrict__ weights, std::uint16_t* __restrict__ out, std::size_t total) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const std::size_t row = i / kWide, within = i % kWide, s = within / kHidden, f = within % kHidden;
    out[i] = float_to_bf16(__fmaf_rn(bf16_to_float(hidden[row * kHidden + f]), weights[row * kStreams + s], bf16_to_float(hyper[i])));
}
__global__ void combine_kernel(const std::uint16_t* __restrict__ a, const std::uint16_t* __restrict__ b,
    std::uint16_t* __restrict__ out, std::size_t total) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    out[i] = float_to_bf16(__fadd_rn(bf16_to_float(a[i]), bf16_to_float(b[i])));
}

// ---------------------------------------------------------------------------------------------------------------------
// Router logits (lanes_tiles_moe_route_logits_kernel, hip_moe_route_launcher.hip:1280-1336): per (row, expert) one
// chain sum = fma(x_f, w_f, sum), f = 0..2559 ascending from 0. CUDA: 64 rows x 64 experts per block, 4 x 4 outputs
// per thread, K staged 32 at a time in shared memory; each output keeps its own sequential chain.
constexpr unsigned kRtRows = 64U, kRtExperts = 64U, kRtK = 32U;
__global__ __launch_bounds__(256) void route_logits_kernel(const std::uint16_t* __restrict__ hidden, const float* __restrict__ w,
    std::uint32_t rows, float* __restrict__ logits) {
    __shared__ __align__(16) float xs[kRtK][kRtRows + 4U];  // +4: fewer bank conflicts on the transposing stores
    __shared__ __align__(16) float ws[kRtK][kRtExperts + 4U];
    const unsigned tx = threadIdx.x % 16U, ty = threadIdx.x / 16U;
    const unsigned row0 = blockIdx.x * kRtRows, e0 = blockIdx.y * kRtExperts;
    float acc[4][4];
#pragma unroll
    for (unsigned i = 0; i < 4U; ++i)
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) acc[i][j] = 0.0F;
    for (unsigned k0 = 0; k0 < kHidden; k0 += kRtK) {
#pragma unroll
        for (unsigned m = 0; m < 8U; ++m) {
            const unsigned idx = threadIdx.x + 256U * m, r = idx / kRtK, k = idx % kRtK;
            xs[k][r] = row0 + r < rows ? bf16_to_float(hidden[static_cast<std::size_t>(row0 + r) * kHidden + k0 + k]) : 0.0F;
            ws[k][r] = w[static_cast<std::size_t>(e0 + r) * kHidden + k0 + k];
        }
        __syncthreads();
#pragma unroll
        for (unsigned k = 0; k < kRtK; ++k) {
            const float4 a = *reinterpret_cast<const float4*>(&xs[k][ty * 4U]);
            const float4 b = *reinterpret_cast<const float4*>(&ws[k][tx * 4U]);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (unsigned i = 0; i < 4U; ++i)
#pragma unroll
                for (unsigned j = 0; j < 4U; ++j) acc[i][j] = __fmaf_rn(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned i = 0; i < 4U; ++i) {
        const unsigned r = row0 + ty * 4U + i;
        if (r >= rows) continue;
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) logits[static_cast<std::size_t>(r) * kExperts + e0 + tx * 4U + j] = acc[i][j];
    }
}

// Router logits for R rows: block = 4 experts (128 blocks); its 4 x 2560 FP32 weights and the R rows are staged in
// shared memory in one go, then the sequential chain over f ascending from 0 runs per (expert, row). SPLIT (default,
// STRIX_QRTX_ROUTE_SPLIT=0 = the old form): thread (e, r) < 4 R owns one chain (was: thread e < 4 ran all R chains,
// 2560 x R FMAs on 4 lanes); row pitch 2562 BF16 so the R chains read distinct banks. Same chains, bit-identical.
constexpr unsigned kRtePerBlock = 4U, kRtePitch = kHidden + 1U;   // +1: the 4 chains read distinct banks
template <unsigned R, bool SPLIT>
__global__ __launch_bounds__(256) void route_logits_rows_kernel(const std::uint16_t* __restrict__ hidden, const float* __restrict__ w,
    float* __restrict__ logits) {
    constexpr unsigned kXs = SPLIT ? kHidden + 2U : kHidden;
    extern __shared__ float rt_smem[];
    float* ws = rt_smem;                                   // [4][kRtePitch]
    auto* xs = reinterpret_cast<std::uint16_t*>(rt_smem + kRtePerBlock * kRtePitch);   // BF16 [R][kXs]
    const unsigned e0 = blockIdx.x * kRtePerBlock;
    for (unsigned i = threadIdx.x; i < kRtePerBlock * kHidden; i += 256U)
        ws[(i / kHidden) * kRtePitch + i % kHidden] = w[static_cast<std::size_t>(e0) * kHidden + i];
    for (unsigned i = threadIdx.x; i < R * kHidden; i += 256U) xs[(i / kHidden) * kXs + i % kHidden] = hidden[i];
    __syncthreads();
    if constexpr (SPLIT) {
        if (threadIdx.x >= kRtePerBlock * R) return;
        const unsigned e = threadIdx.x / R, r = threadIdx.x % R;
        const float* we = ws + e * kRtePitch;
        const std::uint16_t* xr = xs + r * kXs;
        float acc = 0.0F;
#pragma unroll 8
        for (unsigned k = 0; k < kHidden; ++k) acc = __fmaf_rn(bf16_to_float(xr[k]), we[k], acc);
        logits[static_cast<std::size_t>(r) * kExperts + e0 + e] = acc;
    } else {
        if (threadIdx.x >= kRtePerBlock) return;
        const float* we = ws + threadIdx.x * kRtePitch;
        float acc[R];
#pragma unroll
        for (unsigned r = 0; r < R; ++r) acc[r] = 0.0F;
#pragma unroll 8
        for (unsigned k = 0; k < kHidden; ++k) {
            const float wv = we[k];
#pragma unroll
            for (unsigned r = 0; r < R; ++r) acc[r] = __fmaf_rn(bf16_to_float(xs[r * kHidden + k]), wv, acc[r]);
        }
#pragma unroll
        for (unsigned r = 0; r < R; ++r) logits[static_cast<std::size_t>(r) * kExperts + e0 + threadIdx.x] = acc[r];
    }
}

// [S] wave_parallel_moe_route_select_row (hip_moe_route_launcher.hip:866-1010), one 256-thread block per row.
__device__ __forceinline__ bool better(float lv, unsigned li, float rv, unsigned ri) {  // :271-279
    return lv > rv || (lv == rv && li < ri);
}
__global__ __launch_bounds__(256) void route_select_kernel(const float* __restrict__ workspace, std::uint32_t top_k,
    bool normalize, std::uint32_t* __restrict__ ids, float* __restrict__ weights, am::Tables t) {
    constexpr unsigned kThreads = 256U, kWave = 32U, kCandidates = kExperts / kWave;
    const unsigned row = blockIdx.x, lane = threadIdx.x;
    __shared__ float logits[kExperts];
    __shared__ float scratch[kThreads];
    __shared__ float best_values[32];
    __shared__ unsigned best_indices[32];
    for (unsigned e = lane; e < kExperts; e += kThreads) logits[e] = workspace[static_cast<std::size_t>(row) * kExperts + e];
    __syncthreads();
    float local_max = -INFINITY;
    for (unsigned e = lane; e < kExperts; e += kThreads) local_max = fmaxf(local_max, logits[e]);
    scratch[lane] = local_max;
    __syncthreads();
    for (unsigned stride = kThreads / 2U; stride > 0U; stride >>= 1U) {
        if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    const float maximum = scratch[0];
    __syncthreads();
    float local_sum = 0.0F;
    for (unsigned e = lane; e < kExperts; e += kThreads) {
        const float p = am::amd_expf(t, __fsub_rn(logits[e], maximum));
        logits[e] = p;
        local_sum = __fadd_rn(local_sum, p);
    }
    scratch[lane] = local_sum;
    __syncthreads();
    for (unsigned stride = kThreads / 2U; stride > 0U; stride >>= 1U) {
        if (lane < stride) scratch[lane] = __fadd_rn(scratch[lane], scratch[lane + stride]);
        __syncthreads();
    }
    const float denominator = scratch[0];
    for (unsigned e = lane; e < kExperts; e += kThreads) logits[e] = __fdiv_rn(logits[e], denominator);
    __syncthreads();
    if (lane < kWave) {
        float values[kCandidates];
        unsigned available = 0;
#pragma unroll
        for (unsigned i = 0; i < kCandidates; ++i) {
            values[i] = logits[lane + kWave * i];
            if (values[i] == values[i]) available |= 1U << i;  // NaN probabilities are excluded (:957)
        }
        for (unsigned slot = 0; slot < top_k; ++slot) {
            bool has = false;
            float best = -INFINITY;
            unsigned index = kExperts;
#pragma unroll
            for (unsigned i = 0; i < kCandidates; ++i)
                if ((available & (1U << i)) != 0U) {
                    const unsigned expert = lane + kWave * i;
                    if (!has || better(values[i], expert, best, index)) { has = true; best = values[i]; index = expert; }
                }
#pragma unroll
            for (unsigned mask = kWave / 2U; mask > 0U; mask >>= 1U) {
                const float other = __shfl_xor_sync(0xffffffffU, best, static_cast<int>(mask));
                const unsigned other_index = __shfl_xor_sync(0xffffffffU, index, static_cast<int>(mask));
                const int other_has = __shfl_xor_sync(0xffffffffU, static_cast<int>(has), static_cast<int>(mask));
                if (other_has != 0 && (!has || better(other, other_index, best, index))) { has = true; best = other; index = other_index; }
            }
            if (lane == 0U) { best_values[slot] = has ? best : -INFINITY; best_indices[slot] = has ? index : kExperts; }
            if (has && index % kWave == lane) available &= ~(1U << (index / kWave));
        }
    }
    __syncthreads();
    if (lane == 0U) {
        float selected_sum = 0.0F;
        for (unsigned slot = 0; slot < top_k; ++slot) selected_sum = __fadd_rn(selected_sum, best_values[slot]);
        if (!normalize) selected_sum = 1.0F;
        for (unsigned slot = 0; slot < top_k; ++slot) {
            ids[static_cast<std::size_t>(row) * top_k + slot] = best_indices[slot];
            weights[static_cast<std::size_t>(row) * top_k + slot] = __fdiv_rn(best_values[slot], selected_sum);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Latency-hiding forms of the decode router / select / HC-up kernels (STRIX_QRTX_ROUTE_PIPE, STRIX_QRTX_SELECT_WARP,
// STRIX_QRTX_HC_OUT_REG; default on). Same FP32 chains in the same order as the kernels above: bit-identical.
__device__ __forceinline__ void hc_cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(dst))), "l"(src)
                 : "memory");
}
__device__ __forceinline__ void hc_cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void hc_cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// Router logits: block = 4 experts (128 blocks, 128 threads). The 4 x 2560 FP32 weights and the R BF16 rows are staged with
// 16 B cp.async in 10 chunks of 256 k, ALL issued up front (10 MB in flight card-wide); the chain threads (e, r) < 4 R
// wait chunk by chunk and run the same serial fmaf chain k = 0..2559 from 0, 8 k per LDS.128 pair.
constexpr unsigned kRtvChunk = 256U, kRtvChunks = kHidden / kRtvChunk, kRtvWP = kHidden + 4U, kRtvXP = kHidden + 8U;
// One 256-k chunk of the pipelined router chain, then the next (compile-time recursion: wait_group needs an immediate).
template <unsigned C>
__device__ __forceinline__ void route_pipe_chunk(const float* wrow, const std::uint16_t* xrow, bool chain, float& acc) {
    hc_cp_wait<static_cast<int>(kRtvChunks - 1U - C)>();
    __syncthreads();
    if (chain) {
        const float4* wp = reinterpret_cast<const float4*>(wrow + C * kRtvChunk);
        const uint4* xp = reinterpret_cast<const uint4*>(xrow + C * kRtvChunk);
#pragma unroll 4
        for (unsigned g = 0; g < kRtvChunk / 8U; ++g) {
            const uint4 xv = xp[g];
            const float4 a = wp[2U * g], b = wp[2U * g + 1U];
            acc = __fmaf_rn(__uint_as_float(xv.x << 16U), a.x, acc);
            acc = __fmaf_rn(__uint_as_float(xv.x & 0xffff0000U), a.y, acc);
            acc = __fmaf_rn(__uint_as_float(xv.y << 16U), a.z, acc);
            acc = __fmaf_rn(__uint_as_float(xv.y & 0xffff0000U), a.w, acc);
            acc = __fmaf_rn(__uint_as_float(xv.z << 16U), b.x, acc);
            acc = __fmaf_rn(__uint_as_float(xv.z & 0xffff0000U), b.y, acc);
            acc = __fmaf_rn(__uint_as_float(xv.w << 16U), b.z, acc);
            acc = __fmaf_rn(__uint_as_float(xv.w & 0xffff0000U), b.w, acc);
        }
    }
    if constexpr (C + 1U < kRtvChunks) route_pipe_chunk<C + 1U>(wrow, xrow, chain, acc);
}
template <unsigned R>
__global__ __launch_bounds__(128) void route_logits_pipe_kernel(const std::uint16_t* __restrict__ hidden, const float* __restrict__ w,
    float* __restrict__ logits) {
    extern __shared__ __align__(16) float rtv_smem[];
    float* ws = rtv_smem;                                                                  // [4][kRtvWP]
    auto* xs = reinterpret_cast<std::uint16_t*>(rtv_smem + kRtePerBlock * kRtvWP);          // BF16 [R][kRtvXP]
    const unsigned e0 = blockIdx.x * kRtePerBlock, tid = threadIdx.x;
#pragma unroll
    for (unsigned c = 0; c < kRtvChunks; ++c) {
        for (unsigned i = tid; i < kRtePerBlock * (kRtvChunk / 4U); i += 128U) {
            const unsigned e = i / (kRtvChunk / 4U), v = i % (kRtvChunk / 4U);
            hc_cp16(ws + e * kRtvWP + c * kRtvChunk + v * 4U, w + static_cast<std::size_t>(e0 + e) * kHidden + c * kRtvChunk + v * 4U);
        }
        for (unsigned i = tid; i < R * (kRtvChunk / 8U); i += 128U) {
            const unsigned r = i / (kRtvChunk / 8U), v = i % (kRtvChunk / 8U);
            hc_cp16(xs + r * kRtvXP + c * kRtvChunk + v * 8U, hidden + static_cast<std::size_t>(r) * kHidden + c * kRtvChunk + v * 8U);
        }
        hc_cp_commit();
    }
    const unsigned e = tid / R, r = tid % R;
    const bool chain = tid < kRtePerBlock * R;
    float acc = 0.0F;
    route_pipe_chunk<0U>(ws + e * kRtvWP, xs + r * kRtvXP, chain, acc);
    if (chain) logits[static_cast<std::size_t>(r) * kExperts + e0 + e] = acc;
}

// Route select with 4 barriers instead of 20: thread t owns experts t and t + 256 (as route_select_kernel); max by shuffles +
// one smem exchange (order independent); the sum tree levels 128/64/32 are read by warp 0 from smem (a_m = a[l + 32 m]:
// (m, m + 4), (m, m + 2), (0, 1)), levels 16..1 are shfl_down. Same adds in the same pairs: bit-identical.
__global__ __launch_bounds__(256) void route_select_warp_kernel(const float* __restrict__ workspace, std::uint32_t top_k,
    bool normalize, std::uint32_t* __restrict__ ids, float* __restrict__ weights, am::Tables t) {
    constexpr unsigned kWave = 32U, kCandidates = kExperts / kWave;
    const unsigned row = blockIdx.x, tid = threadIdx.x, lane = tid % 32U, warp = tid / 32U;
    __shared__ float sa[256];
    __shared__ float sl[kExperts];
    __shared__ float wmax[8];
    __shared__ float sden;
    __shared__ float best_values[32];
    __shared__ unsigned best_indices[32];
    const float x0 = workspace[static_cast<std::size_t>(row) * kExperts + tid], x1 = workspace[static_cast<std::size_t>(row) * kExperts + tid + 256U];
    float local_max = fmaxf(fmaxf(-INFINITY, x0), x1);
#pragma unroll
    for (unsigned mask = 16U; mask > 0U; mask >>= 1U) local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffU, local_max, mask));
    if (lane == 0U) wmax[warp] = local_max;
    __syncthreads();
    float maximum = wmax[0];
#pragma unroll
    for (unsigned w = 1; w < 8U; ++w) maximum = fmaxf(maximum, wmax[w]);
    const float p0 = am::amd_expf(t, __fsub_rn(x0, maximum)), p1 = am::amd_expf(t, __fsub_rn(x1, maximum));
    sa[tid] = __fadd_rn(__fadd_rn(0.0F, p0), p1);
    __syncthreads();
    if (warp == 0U) {
        float a[8];
#pragma unroll
        for (unsigned m = 0; m < 8U; ++m) a[m] = sa[lane + 32U * m];
        float t1[4], t2[2];
#pragma unroll
        for (unsigned m = 0; m < 4U; ++m) t1[m] = __fadd_rn(a[m], a[m + 4U]);
#pragma unroll
        for (unsigned m = 0; m < 2U; ++m) t2[m] = __fadd_rn(t1[m], t1[m + 2U]);
        float v = __fadd_rn(t2[0], t2[1]);
#pragma unroll
        for (unsigned stride = 16U; stride > 0U; stride >>= 1U) v = __fadd_rn(v, __shfl_down_sync(0xffffffffU, v, stride));
        if (lane == 0U) sden = v;
    }
    __syncthreads();
    const float denominator = sden;
    sl[tid] = __fdiv_rn(p0, denominator);
    sl[tid + 256U] = __fdiv_rn(p1, denominator);
    __syncthreads();
    if (warp != 0U) return;
    float values[kCandidates];
#pragma unroll
    for (unsigned i = 0; i < kCandidates; ++i) values[i] = sl[lane + kWave * i];
    unsigned available = 0;
#pragma unroll
    for (unsigned i = 0; i < kCandidates; ++i)
        if (values[i] == values[i]) available |= 1U << i;
    for (unsigned slot = 0; slot < top_k; ++slot) {
        bool has = false;
        float best = -INFINITY;
        unsigned index = kExperts;
#pragma unroll
        for (unsigned i = 0; i < kCandidates; ++i)
            if ((available & (1U << i)) != 0U) {
                const unsigned expert = lane + kWave * i;
                if (!has || better(values[i], expert, best, index)) { has = true; best = values[i]; index = expert; }
            }
#pragma unroll
        for (unsigned mask = kWave / 2U; mask > 0U; mask >>= 1U) {
            const float other = __shfl_xor_sync(0xffffffffU, best, static_cast<int>(mask));
            const unsigned other_index = __shfl_xor_sync(0xffffffffU, index, static_cast<int>(mask));
            const int other_has = __shfl_xor_sync(0xffffffffU, static_cast<int>(has), static_cast<int>(mask));
            if (other_has != 0 && (!has || better(other, other_index, best, index))) { has = true; best = other; index = other_index; }
        }
        if (lane == 0U) { best_values[slot] = has ? best : -INFINITY; best_indices[slot] = has ? index : kExperts; }
        if (has && index % kWave == lane) available &= ~(1U << (index / kWave));
    }
    __syncwarp();
    if (lane == 0U) {
        float selected_sum = 0.0F;
        for (unsigned slot = 0; slot < top_k; ++slot) selected_sum = __fadd_rn(selected_sum, best_values[slot]);
        if (!normalize) selected_sum = 1.0F;
        for (unsigned slot = 0; slot < top_k; ++slot) {
            ids[static_cast<std::size_t>(row) * top_k + slot] = best_indices[slot];
            weights[static_cast<std::size_t>(row) * top_k + slot] = __fdiv_rn(best_values[slot], selected_sum);
        }
    }
}

// [O] with the whole 80-word weight chain of a thread loaded into registers before any compute (all loads in flight at
// once), FB features x 4 streams per block. The chain per (row, feature, stream) is unchanged.
template <unsigned R, unsigned RG, unsigned FB>
__global__ __launch_bounds__(FB * 4U) void hc_output_rows_reg_kernel(const float* __restrict__ normalized,
    const std::uint32_t* __restrict__ up_codes, const float* __restrict__ up_scales, std::uint16_t* __restrict__ output,
    am::Tables t) {
    __shared__ float lows[kRank][RG];
    const unsigned r0 = blockIdx.y * RG;
    const unsigned s = threadIdx.x % 4U, f = blockIdx.x * FB + threadIdx.x / 4U;
    std::uint32_t words[kRank / 4U];
    float scales[kRank / 32U];
#pragma unroll
    for (unsigned q = 0; q < kRank / 4U; ++q) words[q] = up_codes[(static_cast<std::size_t>(q) * kStreams + s) * kHidden + f];
#pragma unroll
    for (unsigned q = 0; q < kRank / 32U; ++q) scales[q] = up_scales[(static_cast<std::size_t>(q) * kStreams + s) * kHidden + f];
    for (unsigned i = threadIdx.x; i < kRank * RG; i += FB * 4U)
        lows[i / RG][i % RG] =
            r0 + i % RG < R ? normalized[static_cast<std::size_t>(r0 + i % RG) * kNormalizedPitch + kWide + i / RG] : 0.0F;
    __syncthreads();
    float g[RG];
#pragma unroll
    for (unsigned r = 0; r < RG; ++r) g[r] = 0.0F;
#pragma unroll
    for (unsigned q = 0; q < kRank / 4U; ++q) {
        const float scale = scales[q / 8U];
        const std::uint32_t word = words[q];
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) {
            const float c = __fmul_rn(scale, static_cast<float>(static_cast<signed char>((word >> (8U * j)) & 255U)));
#pragma unroll
            for (unsigned r = 0; r < RG; ++r) g[r] = __fmaf_rn(lows[4U * q + j][r], c, g[r]);
        }
    }
    const unsigned base = (threadIdx.x % 32U) & ~3U;
#pragma unroll
    for (unsigned rr = 0; rr < RG; ++rr) {
        if (r0 + rr >= R) break;
        const unsigned r = r0 + rr;
        const float term = amd_sigmoid(t, g[rr]);
        const float n = normalized[static_cast<std::size_t>(r) * kNormalizedPitch + s * kHidden + f];
        float sig[4], nv[4];
#pragma unroll
        for (unsigned k = 0; k < 4U; ++k) {
            sig[k] = __shfl_sync(0xffffffffU, term, base + k);
            nv[k] = __shfl_sync(0xffffffffU, n, base + k);
        }
        if (s == 0U) {
            float mixed = 0.0F;
#pragma unroll
            for (unsigned k = 0; k < 4U; ++k) mixed = __fmaf_rn(nv[k], sig[k], mixed);
            output[static_cast<std::size_t>(r) * kHidden + f] = float_to_bf16(__fmul_rn(mixed, 0.25F));
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// [L] prefetch8_wave_q8_lm_head_kernel<u16, 8> (hip_lm_head_launcher.hip:475-520): warp per vocab row; lane L leaf l
// accumulates partial[l] = fma(x[c + 8L + l], scale * code, partial[l]) over chunks c = 0, 256, ..., 2304; then
// for d = 16, 8, 4, 2, 1: lanes < d add lane + d; lane 0: p0+=p4, p1+=p5, p2+=p6, p3+=p7; p0+=p2, p1+=p3; p0+=p1.
__global__ __launch_bounds__(256) void lm_head_kernel(const std::uint16_t* __restrict__ hidden, const unsigned char* __restrict__ w,
    std::uint32_t vocab, float* __restrict__ logits) {
    __shared__ float x[kHidden];
    for (unsigned f = threadIdx.x; f < kHidden; f += blockDim.x) x[f] = bf16_to_float(hidden[f]);
    __syncthreads();
    const unsigned row = blockIdx.x * 8U + threadIdx.x / 32U, lane = threadIdx.x % 32U;
    if (row >= vocab) return;
    const unsigned char* blocks = w + static_cast<std::size_t>(row) * (kHidden / 32U) * 34U;
    float partial[8];
#pragma unroll
    for (unsigned l = 0; l < 8U; ++l) partial[l] = 0.0F;
#pragma unroll
    for (unsigned chunk = 0; chunk < kHidden; chunk += 256U) {
        const unsigned feature = chunk + 8U * lane;
        const unsigned char* block = blocks + (feature / 32U) * 34U;
        const std::uint16_t* h = reinterpret_cast<const std::uint16_t*>(block);  // block offsets are even
        const float scale = half_bits_to_float(h[0]);
        const unsigned o = 1U + (feature % 32U) / 2U;
        const std::uint16_t c01 = h[o], c23 = h[o + 1U], c45 = h[o + 2U], c67 = h[o + 3U];
        const std::uint16_t pairs[4] = {c01, c23, c45, c67};
#pragma unroll
        for (unsigned l = 0; l < 8U; ++l) {
            const signed char code = static_cast<signed char>((pairs[l / 2U] >> (8U * (l % 2U))) & 255U);
            partial[l] = __fmaf_rn(x[feature + l], __fmul_rn(scale, static_cast<float>(code)), partial[l]);
        }
    }
#pragma unroll
    for (unsigned d = 16U; d > 0U; d /= 2U)
#pragma unroll
        for (unsigned l = 0; l < 8U; ++l) {
            const float partner = __shfl_down_sync(0xffffffffU, partial[l], d);
            if (lane < d) partial[l] = __fadd_rn(partial[l], partner);
        }
    if (lane == 0U) {
        partial[0] = __fadd_rn(partial[0], partial[4]); partial[1] = __fadd_rn(partial[1], partial[5]);
        partial[2] = __fadd_rn(partial[2], partial[6]); partial[3] = __fadd_rn(partial[3], partial[7]);
        partial[0] = __fadd_rn(partial[0], partial[2]); partial[1] = __fadd_rn(partial[1], partial[3]);
        partial[0] = __fadd_rn(partial[0], partial[1]);
        logits[row] = partial[0];
    }
}

// lm_head_kernel for R rows sharing each weight read: row r's partials, leaf order and reduction tree are exactly the
// one-row kernel's (the HIP rows variant keeps the same DAG, hip_lm_head_launcher.hip:573-664). RPW vocab rows per warp
// (STRIX_QRTX_HEAD_ROWS_TILE, default 8; 1 = the old form): the R x 5 KB BF16 staging was done per 8 vocab rows (about
// twice the weight bytes from L2 at R = 8); with RPW = 8 once per 64, and the 8 features of a lane are one 16-byte LDS.
// Same values, same chains: bit-identical.
template <unsigned R, unsigned RPW>
__global__ __launch_bounds__(256) void lm_head_rows_kernel(const std::uint16_t* __restrict__ hidden, std::uint64_t hidden_stride,
    const unsigned char* __restrict__ w, std::uint32_t vocab, float* __restrict__ logits, std::uint64_t logits_stride) {
    __shared__ __align__(16) std::uint16_t xs[R][kHidden];
    for (unsigned i = threadIdx.x; i < R * kHidden; i += blockDim.x)
        xs[i / kHidden][i % kHidden] = hidden[static_cast<std::size_t>(i / kHidden) * hidden_stride + i % kHidden];
    __syncthreads();
    const unsigned lane = threadIdx.x % 32U;
    for (unsigned it = 0; it < RPW; ++it) {
    const unsigned row = (blockIdx.x * RPW + it) * 8U + threadIdx.x / 32U;
    if (row >= vocab) return;
    const unsigned char* blocks = w + static_cast<std::size_t>(row) * (kHidden / 32U) * 34U;
    float partial[R][8];
#pragma unroll
    for (unsigned r = 0; r < R; ++r)
#pragma unroll
        for (unsigned l = 0; l < 8U; ++l) partial[r][l] = 0.0F;
#pragma unroll
    for (unsigned chunk = 0; chunk < kHidden; chunk += 256U) {
        const unsigned feature = chunk + 8U * lane;
        const unsigned char* block = blocks + (feature / 32U) * 34U;
        const std::uint16_t* h = reinterpret_cast<const std::uint16_t*>(block);
        const float scale = half_bits_to_float(h[0]);
        const unsigned o = 1U + (feature % 32U) / 2U;
        const std::uint16_t pairs[4] = {h[o], h[o + 1U], h[o + 2U], h[o + 3U]};
        float coef[8];
#pragma unroll
        for (unsigned l = 0; l < 8U; ++l)
            coef[l] = __fmul_rn(scale, static_cast<float>(static_cast<signed char>((pairs[l / 2U] >> (8U * (l % 2U))) & 255U)));
#pragma unroll
        for (unsigned r = 0; r < R; ++r) {
            std::uint16_t xv[8];
            if constexpr (RPW > 1U) {
                const uint4 v = *reinterpret_cast<const uint4*>(&xs[r][feature]);
                const std::uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (unsigned l = 0; l < 8U; ++l) xv[l] = static_cast<std::uint16_t>(u[l / 2U] >> (16U * (l % 2U)));
            } else {
#pragma unroll
                for (unsigned l = 0; l < 8U; ++l) xv[l] = xs[r][feature + l];
            }
#pragma unroll
            for (unsigned l = 0; l < 8U; ++l) partial[r][l] = __fmaf_rn(bf16_to_float(xv[l]), coef[l], partial[r][l]);
        }
    }
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
#pragma unroll
        for (unsigned d = 16U; d > 0U; d /= 2U)
#pragma unroll
            for (unsigned l = 0; l < 8U; ++l) {
                const float partner = __shfl_down_sync(0xffffffffU, partial[r][l], d);
                if (lane < d) partial[r][l] = __fadd_rn(partial[r][l], partner);
            }
        if (lane == 0U) {
            float* p = partial[r];
            p[0] = __fadd_rn(p[0], p[4]); p[1] = __fadd_rn(p[1], p[5]);
            p[2] = __fadd_rn(p[2], p[6]); p[3] = __fadd_rn(p[3], p[7]);
            p[0] = __fadd_rn(p[0], p[2]); p[1] = __fadd_rn(p[1], p[3]);
            p[0] = __fadd_rn(p[0], p[1]);
            logits[static_cast<std::size_t>(r) * logits_stride + row] = p[0];
        }
    }
    }
}

// Greedy selection on the device (select_greedy_token semantics): first maximum, lowest index on ties (+0 == -0);
// a non-finite logit sets out[row] = 0xffffffff.
__global__ __launch_bounds__(1024) void first_max_kernel(const float* __restrict__ logits, std::uint64_t stride, std::uint32_t count,
    std::uint32_t* __restrict__ out) {
    const float* x = logits + static_cast<std::size_t>(blockIdx.x) * stride;
    float best = 0.0F;
    unsigned index = 0xffffffffU;
    bool bad = false;
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x) {
        const float v = x[i];
        if (!isfinite(v)) bad = true;
        else if (index == 0xffffffffU || v > best) { best = v; index = i; }
    }
    __shared__ float bv[1024];
    __shared__ unsigned bi[1024];
    __shared__ int any_bad;
    if (threadIdx.x == 0) any_bad = 0;
    __syncthreads();
    if (bad) any_bad = 1;
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = index;
    __syncthreads();
    for (unsigned d = blockDim.x / 2U; d > 0U; d /= 2U) {
        if (threadIdx.x < d) {
            const float v = bv[threadIdx.x + d];
            const unsigned j = bi[threadIdx.x + d];
            if (j != 0xffffffffU && (bi[threadIdx.x] == 0xffffffffU || v > bv[threadIdx.x] || (v == bv[threadIdx.x] && j < bi[threadIdx.x]))) {
                bv[threadIdx.x] = v;
                bi[threadIdx.x] = j;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = any_bad || bi[0] == 0xffffffffU ? 0xffffffffU : bi[0];
}

unsigned blocks_for(std::size_t total, unsigned threads) { return static_cast<unsigned>((total + threads - 1U) / threads); }

}  // namespace

std::size_t hc_packed_bytes() noexcept { return kDownCodeBytes + kDownScaleBytes + kUpCodeBytes + kUpScaleBytes; }
HcPacked hc_packed_view(void* base) noexcept {
    auto* b = static_cast<unsigned char*>(base);
    HcPacked p;
    p.down_codes = reinterpret_cast<std::int8_t*>(b);
    p.down_scales = reinterpret_cast<float*>(b + kDownCodeBytes);
    p.up_codes = reinterpret_cast<std::uint32_t*>(b + kDownCodeBytes + kDownScaleBytes);
    p.up_scales = reinterpret_cast<float*>(b + kDownCodeBytes + kDownScaleBytes + kUpCodeBytes);
    return p;
}
void hc_pack(const void* down_q8, const void* up_q8, const HcPacked& packed, cudaStream_t stream) {
    const unsigned items = kRank * (kWide / 32U) + kWide * (kRank / 32U);
    hc_pack_kernel<<<blocks_for(items, 256U), 256U, 0, stream>>>(static_cast<const unsigned char*>(down_q8),
        static_cast<const unsigned char*>(up_q8), packed);
    check(cudaGetLastError(), "pack launch");
}

namespace {
bool hc_rows_split_enabled() {
    static const bool on = strix::envcfg::flag("STRIX_QRTX_HC_ROWS_SPLIT", true);
    return on;
}
template <unsigned R, unsigned OG, unsigned RG, unsigned RO>
void hc_mix_rows_g(const HcMixArgs& a, const am::Tables& tables, cudaStream_t stream, bool split) {
    hc_projection_rows_kernel<R, OG, RG><<<dim3(kRank / OG + (a.inject ? 1U : 0U), (R + RG - 1U) / RG), 512U, 0, stream>>>(
        a.normalized, a.packed.down_codes, a.packed.down_scales, a.inject, a.normalized, a.injection, tables);
    check(cudaGetLastError(), "projection rows launch");
    static const bool reg = strix::envcfg::flag("STRIX_QRTX_HC_OUT_REG", true);
    if (reg && split)  // legacy seam (HC_ROWS_SPLIT=0) keeps the old kernel
        hc_output_rows_reg_kernel<R, RO, 16U><<<dim3(kHidden / 16U, (R + RO - 1U) / RO), 64U, 0, stream>>>(a.normalized,
            a.packed.up_codes, a.packed.up_scales, a.output, tables);
    else
    hc_output_rows_kernel<R, RO><<<dim3(kHidden / 32U, (R + RO - 1U) / RO), 128U, 0, stream>>>(a.normalized, a.packed.up_codes,
        a.packed.up_scales, a.output, tables);
    check(cudaGetLastError(), "output rows launch");
}
template <unsigned R> void hc_mix_rows_r(const HcMixArgs& a, const am::Tables& tables, cudaStream_t stream) {
    constexpr unsigned OG = R >= 8U ? 8U : R >= 4U ? 4U : R >= 2U ? 2U : 1U, RG = R >= 8U ? 2U : 1U, RO = R >= 4U ? 2U : 1U;
    if (hc_rows_split_enabled()) hc_mix_rows_g<R, OG, RG, RO>(a, tables, stream, true);
    else hc_mix_rows_g<R, 1U, R, R>(a, tables, stream, false);
}
void hc_mix_rows(const HcMixArgs& a, const am::Tables& tables, cudaStream_t stream) {
    switch (a.rows) {
    case 1: hc_mix_rows_r<1>(a, tables, stream); break;
    case 2: hc_mix_rows_r<2>(a, tables, stream); break;
    case 3: hc_mix_rows_r<3>(a, tables, stream); break;
    case 4: hc_mix_rows_r<4>(a, tables, stream); break;
    case 5: hc_mix_rows_r<5>(a, tables, stream); break;
    case 6: hc_mix_rows_r<6>(a, tables, stream); break;
    case 7: hc_mix_rows_r<7>(a, tables, stream); break;
    case 8: hc_mix_rows_r<8>(a, tables, stream); break;
    // 9..16 rows (core::decode_rows_max, split forms only): R is only the row bound of the 2-row groups, so each row's
    // chains are the R = 8 ones (OG 8, RG 2, RO 2)
    case 9: hc_mix_rows_g<9, 8U, 2U, 2U>(a, tables, stream, true); break;
    case 10: hc_mix_rows_g<10, 8U, 2U, 2U>(a, tables, stream, true); break;
    case 11: hc_mix_rows_g<11, 8U, 2U, 2U>(a, tables, stream, true); break;
    case 12: hc_mix_rows_g<12, 8U, 2U, 2U>(a, tables, stream, true); break;
    case 13: hc_mix_rows_g<13, 8U, 2U, 2U>(a, tables, stream, true); break;
    case 14: hc_mix_rows_g<14, 8U, 2U, 2U>(a, tables, stream, true); break;
    case 15: hc_mix_rows_g<15, 8U, 2U, 2U>(a, tables, stream, true); break;
    default: hc_mix_rows_g<16, 8U, 2U, 2U>(a, tables, stream, true); break;
    }
}
bool route_split_enabled() {
    static const bool on = strix::envcfg::flag("STRIX_QRTX_ROUTE_SPLIT", true);
    return on;
}
template <unsigned R, bool SPLIT> void route_rows_rs(const std::uint16_t* hidden, const float* w, float* logits, cudaStream_t stream) {
    constexpr unsigned smem = kRtePerBlock * kRtePitch * 4U + R * (SPLIT ? kHidden + 2U : kHidden) * 2U;
    static const cudaError_t attr =
        cudaFuncSetAttribute(route_logits_rows_kernel<R, SPLIT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    check(attr, "route logits rows smem");
    static const bool pipe = strix::envcfg::flag("STRIX_QRTX_ROUTE_PIPE", true);
    if (SPLIT && pipe && (reinterpret_cast<std::uintptr_t>(hidden) & 15U) == 0U && (reinterpret_cast<std::uintptr_t>(w) & 15U) == 0U) {
        constexpr unsigned psmem = kRtePerBlock * kRtvWP * 4U + R * kRtvXP * 2U;
        static const cudaError_t pattr =
            cudaFuncSetAttribute(route_logits_pipe_kernel<R>, cudaFuncAttributeMaxDynamicSharedMemorySize, psmem);
        check(pattr, "route logits pipe smem");
        route_logits_pipe_kernel<R><<<kExperts / kRtePerBlock, 128U, psmem, stream>>>(hidden, w, logits);
        return;
    }
    route_logits_rows_kernel<R, SPLIT><<<kExperts / kRtePerBlock, 256U, smem, stream>>>(hidden, w, logits);
}
template <unsigned R> void route_rows_r(const std::uint16_t* hidden, const float* w, float* logits, cudaStream_t stream) {
    if (route_split_enabled()) route_rows_rs<R, true>(hidden, w, logits, stream);
    else route_rows_rs<R, false>(hidden, w, logits, stream);
}
}  // namespace

void hc_mix(const HcMixArgs& a, const am::Tables& tables, cudaStream_t stream, unsigned stages) {
    if (!a.rows) return;
    if (a.rows > 65535U * kProjRows) throw std::invalid_argument("qwen_hc mix: too many rows");
    const bool compact = hc_compact_workspace(a, stages, hc_compact_norm_enabled(), decode_rows_enabled());
    if ((stages & 1U) && compact) hc_norm_kernel<true><<<dim3(a.rows, kStreams), 256U, 0, stream>>>(a.input, a.norm, a.epsilon, a.normalized, tables);
    else if (stages & 1U) hc_norm_kernel<<<dim3(a.rows, kStreams), 256U, 0, stream>>>(a.input, a.norm, a.epsilon, a.normalized, tables);
    check(cudaGetLastError(), "norm launch");
    const unsigned out_blocks = kProjLowBlocks + (a.inject ? 1U : 0U);
    // Output blocks fastest (HIP default order): the 6 blocks of one row block run together and share its x rows in L2.
    if (a.mma) {  // FAST form: tensor-core projection + output (stages bit 1 / bit 2 as the exact path)
        static const bool attr = [] {
            check(cudaFuncSetAttribute(hc_output_mma_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kMmaOutSmem)), "mma attr");
            check(cudaFuncSetAttribute(hc_output_mma_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kMmaOutSmem)), "mma attr");
            return true;
        }();
        (void)attr;
        const unsigned row_tiles = (a.rows + kMmaOutRows - 1U) / kMmaOutRows;
        if (stages & 2U) projection_mma(a, tables, stream, compact);
        check(cudaGetLastError(), "mma projection launch");
        const dim3 og(row_tiles, kHidden / (32U * kMmaOutPasses));
        if ((stages & 4U) && compact) hc_output_mma_kernel<true><<<og, 128U, kMmaOutSmem, stream>>>(a.normalized,
            a.rows, a.packed.up_codes, a.packed.up_scales, a.output, a.gates, tables, a.input, a.norm);
        else if (stages & 4U) hc_output_mma_kernel<false><<<og, 128U, kMmaOutSmem, stream>>>(a.normalized,
            a.rows, a.packed.up_codes, a.packed.up_scales, a.output, a.gates, tables, a.input, a.norm);
        check(cudaGetLastError(), "mma output launch");
        if ((stages & 8U) && a.gates)
            hc_mix_epilogue_kernel<true><<<blocks_for(std::size_t{a.rows} * kHidden, 256U), 256U, 0, stream>>>(a.normalized, a.gates, a.rows,
                a.output, tables);
        return;
    }
    const bool rows_form = a.rows <= 8U || (a.rows <= core::decode_rows_max() && hc_rows_split_enabled());
    if (!compact && rows_form && stages == 15U && !a.gates && decode_rows_enabled()) {
        hc_mix_rows(a, tables, stream);
        return;
    }
    const dim3 proj_grid(out_blocks, (a.rows + kProjRows - 1U) / kProjRows);
    if ((stages & 2U) && compact) hc_projection_kernel<true><<<proj_grid, 256U, 0, stream>>>(a.normalized, a.rows,
        a.packed.down_codes, a.packed.down_scales, a.inject, a.normalized, a.injection, tables, a.input, a.norm);
    else if (stages & 2U) hc_projection_kernel<<<proj_grid, 256U, 0, stream>>>(a.normalized, a.rows,
        a.packed.down_codes, a.packed.down_scales, a.inject, a.normalized, a.injection, tables);
    check(cudaGetLastError(), "projection launch");
    const dim3 out_grid((a.rows + kOutRows - 1U) / kOutRows, kHidden / kOutThreads);
    if ((stages & 4U) && !a.gates && compact)
        hc_output_kernel<true, true><<<out_grid, kOutThreads, 0, stream>>>(a.normalized, a.rows, a.packed.up_codes, a.packed.up_scales,
            a.output, nullptr, tables, a.input, a.norm);
    else if ((stages & 4U) && !a.gates)
        hc_output_kernel<true><<<out_grid, kOutThreads, 0, stream>>>(a.normalized, a.rows, a.packed.up_codes, a.packed.up_scales,
            a.output, nullptr, tables);
    if ((stages & 4U) && a.gates)
        hc_output_kernel<false><<<out_grid, kOutThreads, 0, stream>>>(a.normalized, a.rows, a.packed.up_codes, a.packed.up_scales,
            a.output, a.gates, tables);
    if ((stages & 8U) && a.gates)
        hc_mix_epilogue_kernel<false><<<blocks_for(std::size_t{a.rows} * kHidden, 256U), 256U, 0, stream>>>(a.normalized, a.gates, a.rows,
            a.output, tables);
    check(cudaGetLastError(), "output launch");
}

void token_embedding(const void* table, const std::uint32_t* ids, std::uint32_t rows, std::uint16_t* out, cudaStream_t stream) {
    if (!rows) return;
    embedding_kernel<<<rows, 256U, 0, stream>>>(static_cast<const unsigned char*>(table), ids, out);
    check(cudaGetLastError(), "embedding launch");
}
void hyper_expand(const std::uint16_t* in, std::uint32_t rows, std::uint16_t* out, cudaStream_t stream) {
    const std::size_t total = std::size_t{rows} * kWide;
    if (!total) return;
    expand_kernel<<<blocks_for(total, 256U), 256U, 0, stream>>>(in, out, total);
    check(cudaGetLastError(), "expand launch");
}
void inject(const std::uint16_t* hyper, const std::uint16_t* hidden, const float* injection, std::uint32_t rows,
    std::uint16_t* out, cudaStream_t stream) {
    const std::size_t total = std::size_t{rows} * kWide;
    if (!total) return;
    const bool aligned = ((reinterpret_cast<std::uintptr_t>(hyper) | reinterpret_cast<std::uintptr_t>(hidden) |
                           reinterpret_cast<std::uintptr_t>(out)) & 15U) == 0U;
    if (aligned && rows <= 65535U && hc_inject_vec_enabled())
        inject_vec_kernel<<<dim3((kWide / 8U + 255U) / 256U, rows), 256U, 0, stream>>>(hyper, hidden, injection, out);
    else
        inject_kernel<<<blocks_for(total, 256U), 256U, 0, stream>>>(hyper, hidden, injection, out, total);
    check(cudaGetLastError(), "inject launch");
}
void combine(const std::uint16_t* shared, const std::uint16_t* routed, std::uint32_t rows, std::uint16_t* out, cudaStream_t stream) {
    const std::size_t total = std::size_t{rows} * kHidden;
    if (!total) return;
    combine_kernel<<<blocks_for(total, 256U), 256U, 0, stream>>>(shared, routed, out, total);
    check(cudaGetLastError(), "combine launch");
}
void route_logits(const std::uint16_t* hidden, const float* gate_inp, std::uint32_t rows, float* logits, cudaStream_t stream) {
    if (!rows) return;
    // 9..16 rows (core::decode_rows_max): <= 8-row pieces (hidden / logits rows are dense; each row's logits are its own)
    if (rows > core::kDecodePieceRows && rows <= core::decode_rows_max() && decode_rows_enabled()) {
        for (std::uint32_t r0 = 0; r0 < rows; r0 += core::kDecodePieceRows)
            route_logits(hidden + std::size_t{r0} * kHidden, gate_inp, std::min(rows - r0, core::kDecodePieceRows),
                         logits + std::size_t{r0} * kExperts, stream);
        return;
    }
    if (rows <= 8U && decode_rows_enabled()) {
        switch (rows) {
        case 1: route_rows_r<1>(hidden, gate_inp, logits, stream); break;
        case 2: route_rows_r<2>(hidden, gate_inp, logits, stream); break;
        case 3: route_rows_r<3>(hidden, gate_inp, logits, stream); break;
        case 4: route_rows_r<4>(hidden, gate_inp, logits, stream); break;
        case 5: route_rows_r<5>(hidden, gate_inp, logits, stream); break;
        case 6: route_rows_r<6>(hidden, gate_inp, logits, stream); break;
        case 7: route_rows_r<7>(hidden, gate_inp, logits, stream); break;
        default: route_rows_r<8>(hidden, gate_inp, logits, stream); break;
        }
        check(cudaGetLastError(), "route logits rows launch");
        return;
    }
    route_logits_kernel<<<dim3((rows + kRtRows - 1U) / kRtRows, kExperts / kRtExperts), 256U, 0, stream>>>(hidden, gate_inp, rows, logits);
    check(cudaGetLastError(), "route logits launch");
}
void route_select(const float* logits, std::uint32_t rows, std::uint32_t top_k, bool normalize, const am::Tables& tables,
    std::uint32_t* ids, float* weights, cudaStream_t stream) {
    if (top_k == 0U || top_k > 32U) throw std::invalid_argument("qwen_hc route_select: top_k must be 1..32");
    if (!rows) return;
    static const bool warp = strix::envcfg::flag("STRIX_QRTX_SELECT_WARP", true);
    if (warp) route_select_warp_kernel<<<rows, 256U, 0, stream>>>(logits, top_k, normalize, ids, weights, tables);
    else route_select_kernel<<<rows, 256U, 0, stream>>>(logits, top_k, normalize, ids, weights, tables);
    check(cudaGetLastError(), "route select launch");
}
void lm_head(const std::uint16_t* hidden, const void* output_q8, std::uint32_t vocab, float* logits, cudaStream_t stream) {
    if (!vocab) return;
    lm_head_kernel<<<(vocab + 7U) / 8U, 256U, 0, stream>>>(hidden, static_cast<const unsigned char*>(output_q8), vocab, logits);
    check(cudaGetLastError(), "lm head launch");
}
void lm_head_rows(const std::uint16_t* hidden, std::uint64_t hidden_stride, std::uint32_t rows, const void* output_q8,
    std::uint32_t vocab, float* logits, std::uint64_t logits_stride, cudaStream_t stream) {
    if (!vocab || !rows) return;
    const auto* w = static_cast<const unsigned char*>(output_q8);
    static const bool tile = strix::envcfg::flag("STRIX_QRTX_HEAD_ROWS_TILE", true);
    const auto go = [&](auto r_tag, auto rpw_tag) {
        constexpr unsigned R = decltype(r_tag)::value, RPW = decltype(rpw_tag)::value;
        lm_head_rows_kernel<R, RPW><<<(vocab + 8U * RPW - 1U) / (8U * RPW), 256U, 0, stream>>>(hidden, hidden_stride, w, vocab,
                                                                                            logits, logits_stride);
    };
    const auto pick = [&](auto r_tag) {
        if (tile) go(r_tag, std::integral_constant<unsigned, 8U>{});
        else go(r_tag, std::integral_constant<unsigned, 1U>{});
    };
    switch (rows) {
    case 1: pick(std::integral_constant<unsigned, 1U>{}); break;
    case 2: pick(std::integral_constant<unsigned, 2U>{}); break;
    case 3: pick(std::integral_constant<unsigned, 3U>{}); break;
    case 4: pick(std::integral_constant<unsigned, 4U>{}); break;
    case 5: pick(std::integral_constant<unsigned, 5U>{}); break;
    case 6: pick(std::integral_constant<unsigned, 6U>{}); break;
    case 7: pick(std::integral_constant<unsigned, 7U>{}); break;
    case 8: pick(std::integral_constant<unsigned, 8U>{}); break;
    default: throw std::invalid_argument("qwen_hc lm_head_rows: rows must be 1..8");
    }
    check(cudaGetLastError(), "lm head rows launch");
}
void first_max_rows(const float* logits, std::uint64_t stride, std::uint32_t rows, std::uint32_t count, std::uint32_t* out,
    cudaStream_t stream) {
    if (!rows) return;
    first_max_kernel<<<rows, 1024U, 0, stream>>>(logits, stride, count, out);
    check(cudaGetLastError(), "first max launch");
}
std::uint32_t first_max_token(const float* logits, std::size_t count) {  // select_greedy_token, src/serial_request.cpp:12-55
    if (count == 0U || count > 0x100000000ULL) throw std::invalid_argument("greedy selection vocabulary size is invalid");
    std::uint32_t best = 0;
    float best_value = 0.0F;
    bool initialized = false;
    for (std::size_t i = 0; i < count; ++i) {
        const float v = logits[i];
        if (!std::isfinite(v)) throw std::invalid_argument("greedy selection logits must be finite");
        if (!initialized || v > best_value) { initialized = true; best_value = v; best = static_cast<std::uint32_t>(i); }
    }
    return best;
}

}  // namespace strix::cuda::qwen_hc
