// CUDA dense A8 projections and shared expert, bit-identical to the HIP dense-a8-v1 prefill. See
// include/strix/cuda/qwen_dense.hpp for the contract; HIP references are given per kernel.
//
// ISA facts relied on (gfx1151 disassembly of the HIP translation units, engine bf60d9d2):
//  quantizer  hip_gdn_launcher.s, _ZN5strix7q8_rows24dense_a8_quantize_kernelINS_12_GLOBAL__N_116GdnInputRowsSiteILj256EEEEEvT_PaPf
//             (:257360): v_cmp_class_f32 0x207 (non-finite), v_max3_f32 |x| from 0, DPP row_xmask:1/2 quad max/or,
//             tiny = v_cmp_lt 0 / v_cmp_gt 0x4000000 (2^-119) (:257498-257499), IEEE v_div_scale/fmas/fixup amax/127
//             (:257598-257613) and x/delta (:257543-257558), v_rndne_f32 (:257560), v_maxmin_f32 -127/127 (:257561),
//             v_cvt_i32_f32 (:257563). float_denorm_mode_32 3 (IEEE denormals) in every kernel descriptor.
//  projection hip_gdn_launcher.s, ..project_rows_dense_a8_wmma_lds2_kernelINS_..GdnInputRowsSiteILj256EEELj8ELb0E.. (:274436):
//             v_wmma_i32_16x16x16_iu8 from the 0x4B400000 seed (:274851-274858), v_add_f32 0xcb400000 (exact float(dot)),
//             v_mul_f32 dw*delta, v_fmac_f32 acc += p*dot (:274854-274892), v_cndmask 0x7fc00000 canonical NaN (:279359...).
//  shared down store hip_shared_expert_launcher.s (:114776-114786): asm v_mul_f32 gate, value; then BF16 (v_add3 0x7fff).
//  activation hip_shared_expert_launcher.s, _ZN5strix12_GLOBAL__N_136rows_shared_activation_scalar_kernelEmmmmPKvjj (:165-462):
//             expf(-g) inlined (v_mul -log2e (:202), v_fma, v_fmamk lo, v_rndne, v_sub, v_add, v_exp, v_ldexp (:214),
//             selects 0x42ce8ed0 / 0xc2b17218) == amd_expf(-g) by sign symmetry; v_add 1.0 (:220); IEEE div g/den
//             (:221-236); v_mul u*q (:238); scalar dot v_dual_fmac_f32 sum = w * x + sum over features tid, tid+256, ...
//             ascending (:281); LDS tree strides 128..1, v_add_f32 (:301-401); 1/(1 + expf(-s)) (:416-457, IEEE div).
#include "strix/env_flags.hpp"
#include "strix/lever.hpp"
#include "../../core/imma_tile.cuh"
#include "strix/cuda/core/decode_rows.hpp"
#include "strix/cuda/qwen_dense.hpp"
#include "strix/cuda/core/nvfp4_a8.cuh"
#include "../../core/imma_dense.cuh"
#include "../../weight/arch/sm120_sources.cuh"

#include <unordered_map>

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace strix::cuda::qwen_dense {
#ifdef STRIX_QWEN_DENSE_TEST_DISPATCH
// Defined by the test executable; absent from production builds.
void test_dispatch(std::uint32_t rows, bool one_pass);
#endif
namespace {

// STRIX_QRTX_DECODE_KERNELS=0: keep the tiled kernels at rows <= 8 too (reference seam for the decode-shaped ones).
bool decode_rows_enabled() {
    static const bool on = strix::envcfg::flag("STRIX_QRTX_DECODE_KERNELS", true);
    return on;
}

// STRIX_QRTX_DENSE_TILE2=0: the first tiled kernel (project_kernel) instead of project_tile2_kernel. Bitwise: same chain per output.
inline constexpr strix::lever::Decl dense_tile2_lever{
    .name = "qwen_dense_tile2", .env = "STRIX_QRTX_DENSE_TILE2", .exactness = strix::lever::Exactness::bitwise,
    .default_on = true, .source = "tests/cuda/qwen_dense_tests.cu: byte equality vs project_kernel"};
bool dense_tile2_enabled() {
    static const bool on = strix::lever::select(dense_tile2_lever, {}).on;
    return on;
}

// STRIX_QRTX_DENSE_T2_REMAP=0: the first tile2 weight loader (17 lanes per row, two capped CTAs per SM). Bitwise.
inline constexpr strix::lever::Decl dense_t2_remap_lever{
    .name = "qwen_dense_t2_remap", .env = "STRIX_QRTX_DENSE_T2_REMAP", .exactness = strix::lever::Exactness::bitwise,
    .default_on = true, .source = "tests/cuda/qwen_dense_tests.cu --bench-shape: equal output hashes, 5 site shapes"};
bool dense_t2_remap_enabled() {
    static const bool on = strix::lever::select(dense_t2_remap_lever, {}).on;
    return on;
}

// STRIX_QRTX_DENSE_T2_A16=0: the remap loader (4-byte .ca copies) instead of the aligned-window loader (16-byte .cg
// copies of the 80-byte aligned window that holds a stage's 68 raw bytes). Bitwise: the same raw bytes reach the reader,
// which adds the window's word shift. Needs row_bytes % 16 == 0 and 16-aligned segments (launcher check).
inline constexpr strix::lever::Decl dense_t2_a16_lever{
    .name = "qwen_dense_t2_a16", .env = "STRIX_QRTX_DENSE_T2_A16", .exactness = strix::lever::Exactness::bitwise,
    .default_on = false, .source = "qwen_dense_tests --bench-shape 16-11: equal hashes, 2-stage ring +13 % slower (L2 bytes, see RTXFAST6R)"};
bool dense_t2_a16_enabled() {
    static const bool on = strix::lever::select(dense_t2_a16_lever, {}).on;
    return on;
}

// STRIX_QRTX_DENSE_T4=0: tile2 (2 K blocks per stage, 4-byte weight copies) instead of project_tile4_kernel (4 K blocks
// per stage, 8-byte weight copies). Bitwise: same per-output chain, blocks ascending. Needs the remap lever on.
inline constexpr strix::lever::Decl dense_t4_lever{
    .name = "qwen_dense_t4", .env = "STRIX_QRTX_DENSE_T4", .exactness = strix::lever::Exactness::bitwise,
    .default_on = false, .source = "tests/cuda/qwen_dense_tests.cu --bench-shape: equal output hashes, 5 site shapes"};
bool dense_t4_enabled() {
    static const bool on = strix::lever::select(dense_t4_lever, {}).on;
    return on;
}

__device__ __forceinline__ std::uint32_t bf16_bits(float v) {
    const std::uint32_t b = __float_as_uint(v);
    return (b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U;   // HIP float_to_bf16, no NaN special case
}
__device__ __forceinline__ float bf16_float(std::uint32_t h) { return __uint_as_float(h << 16U); }

// v_mul_f32 under IEEE mode: a NaN operand is returned quieted (src0 first) [assumption on RDNA3 NaN propagation];
// CUDA would return 0x7fffffff, whose BF16 rounding is 0x8000 (-0), so the NaN path is emulated explicitly.
__device__ __forceinline__ float amd_mul(float a, float b) {
    if (a != a) return __uint_as_float(__float_as_uint(a) | 0x00400000U);
    if (b != b) return __uint_as_float(__float_as_uint(b) | 0x00400000U);
    return __fmul_rn(a, b);
}

// ---------------------------------------------------------------------------------------------------------------------
// Quantizer: HIP dense_a8_quantize_kernel / dense_a8_quantize8 (src/hip_q8_rows_gemv.hpp :1107-1144). Four threads per
// 32-feature block, 8 features each; the quad max/or is order-free (fmaxf, |).
template <Input I>
__global__ __launch_bounds__(256) void quantize_kernel(std::uint32_t rows, std::uint32_t width, const void* input,
                                                       std::uint64_t stride, std::int8_t* codes, float* deltas,
                                                       std::uint32_t pitch) {
    const std::uint32_t groups = width / 8U;
    const std::uint64_t item = static_cast<std::uint64_t>(blockIdx.x) * 256U + threadIdx.x;
    const bool valid = item < static_cast<std::uint64_t>(rows) * groups;
    const std::uint32_t r = valid ? static_cast<std::uint32_t>(item / groups) : 0U;
    const std::uint32_t g = valid ? static_cast<std::uint32_t>(item % groups) : 0U;
    float x[8];
#pragma unroll
    for (unsigned l = 0; l < 8U; ++l) {
        const std::uint64_t at = static_cast<std::uint64_t>(r) * stride + 8U * g + l;
        if (!valid) x[l] = 0.0F;
        else if constexpr (I == Input::bf16) x[l] = bf16_float(static_cast<const std::uint16_t*>(input)[at]);
        else x[l] = static_cast<const float*>(input)[at];
    }
    int bad = 0;
    float amax = 0.0F;
#pragma unroll
    for (unsigned l = 0; l < 8U; ++l) { bad |= !isfinite(x[l]); amax = fmaxf(amax, fabsf(x[l])); }
    bad |= __shfl_xor_sync(0xffffffffU, bad, 1); bad |= __shfl_xor_sync(0xffffffffU, bad, 2);
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffU, amax, 1)); amax = fmaxf(amax, __shfl_xor_sync(0xffffffffU, amax, 2));
    const bool tiny = amax > 0.0F && amax < 0x1p-119F;
    const float delta = bad ? __uint_as_float(kCanonicalNan) : (amax == 0.0F || tiny) ? 1.0F : __fdiv_rn(amax, 127.0F);
    unsigned word[2] = {0U, 0U};
#pragma unroll
    for (unsigned l = 0; l < 8U; ++l) {
        const float rounded = bad || tiny ? 0.0F : rintf(__fdiv_rn(x[l], delta));
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        word[l / 4U] |= (static_cast<unsigned>(code) & 0xffU) << (8U * (l % 4U));
    }
    if (!valid) return;
    *reinterpret_cast<uint2*>(codes + static_cast<std::uint64_t>(r) * width + 8U * g) = make_uint2(word[0], word[1]);
    if (!(g & 3U)) deltas[static_cast<std::uint64_t>(g / 4U) * pitch + r] = delta;
}

// ---------------------------------------------------------------------------------------------------------------------
// Projection. CTA = 128 rows x 128 outputs, 8 warps (2 x 4), warp = 64 rows x 32 outputs = 4 x 4 IMMA m16n8k32 tiles.
// K = 32 = one Q8_0 block per IMMA, so the IMMA result is exactly dot_b (seeded with 0x4B400000 like the HIP WMMA; then
// float(dot) = bits - 1.5*2^23, one exact subtraction). Every thread owns 64 (row, output) accumulators and walks the
// blocks in ascending order: the HIP chain acc = fma(dw * delta, float(dot), acc), unchanged by tiling.
// Pipeline: stages of 4 blocks (codes 128 x 128 B, raw Q8_0 bytes 128 x 136 B, deltas 4 x 128), cp.async double buffer.
using KSite = core::dense_a8::Site<kMaxSegments>;
using KSeg = core::dense_a8::KSeg;
using NoCapture = core::dense_a8::NoCapture;
#ifdef STRIX_WFMT_CAPTURE
using PartialCapture = core::dense_a8::PartialCapture;
using core::dense_a8::PartialCaptureConfig;
using core::dense_a8::partial_capture_config;
#endif
bool rows_ksplit_enabled() {
    static const bool on = strix::envcfg::flag("STRIX_QRTX_ROWS_KSPLIT", true);
    return on;
}
// 9..16-row windows: one k-split pass with 16 padded rows (default) or STRIX_QRTX_DENSE_ROWS16=0 = two <= 8-row pieces
bool rows16_enabled() {
    static const bool on = strix::envcfg::flag("STRIX_QRTX_DENSE_ROWS16", true);
    return on;
}
// Decode rows (rows <= 8) policy of main's streamed form (#680, rtxmax2 receipts; kernels in core/imma_dense.cuh
// project_rows_cp_kernel / project_rows_ks_kernel<KO>). Selection (qwen_rows_glue_exact_test --time on the RTX PRO 6000,
// us/call old ks -> cp): small sites win (shared gate/up 2.7 MB 5.65 -> 3.76 at 1 row, shared down 1.4 MB 3.60 -> 2.98),
// large sites lose (gdn in 44 MB 36.1 -> 43.6 at 8 rows, gdn out 16.7 MB 15.2 -> 20.2 at 1 row) except K = 6144 at 8 rows
// (24.3 -> 22.5). STRIX_QRTX_ROWS_CP: unset/"auto" = by site (weight bytes <= STRIX_QRTX_ROWS_CP_MB, default 4, or
// K >= 6144 at 8 rows), 1 = every fitting site, 0 = never (old K-split kernel everywhere). Bits are identical in every mode.
int rows_cp_mode() {
    static const int mode = [] {
        const char* e = strix::envcfg::text("STRIX_QRTX_ROWS_CP");
        if (e == nullptr || *e == 0 || std::strcmp(e, "auto") == 0) return 2;
        return std::atoi(e) != 0 ? 1 : 0;
    }();
    return mode;
}
bool rows_cp_enabled() { return rows_cp_mode() != 0; }
bool rows_cp_wanted(const KSite& k, unsigned rows) {
    const int mode = rows_cp_mode();
    if (mode != 2) return mode == 1;
    static const double limit = [] {
        const char* e = strix::envcfg::text("STRIX_QRTX_ROWS_CP_MB");
        return (e != nullptr && *e != 0 ? std::atof(e) : 4.0) * 1048576.0;
    }();
    const double bytes = static_cast<double>(k.outputs) * (k.width / 32U) * 34.0;
    return bytes <= limit || (rows >= 8U && k.width >= 6144U);
}
// K-split outputs per CTA (#680): STRIX_QRTX_KSO = 2 / 4 / 8, unset = 4 at >= 128 blocks per row, else 8.
int rows_ks_out(std::uint32_t nblocks) {
    static const int ko = [] { const char* e = strix::envcfg::text("STRIX_QRTX_KSO"); return e ? std::atoi(e) : 0; }();
    return ko ? ko : (nblocks >= 128U ? 4 : 8);
}
struct DenseNumeric {
    using Site = KSite;
    static bool decode_rows_enabled() { return ::strix::cuda::qwen_dense::decode_rows_enabled(); }
    static bool rows_ksplit_enabled() { return ::strix::cuda::qwen_dense::rows_ksplit_enabled(); }
    static bool rows_cp_enabled() { return ::strix::cuda::qwen_dense::rows_cp_enabled(); }
    static bool rows_cp_wanted(const KSite& k, unsigned rows) { return ::strix::cuda::qwen_dense::rows_cp_wanted(k, rows); }
    static int rows_ks_out(std::uint32_t nblocks) { return ::strix::cuda::qwen_dense::rows_ks_out(nblocks); }
    static bool dense_tile2_enabled() { return ::strix::cuda::qwen_dense::dense_tile2_enabled(); }
    static bool dense_t2_remap_enabled() { return ::strix::cuda::qwen_dense::dense_t2_remap_enabled(); }
    template <Store S>
    __device__ __forceinline__ static void store(const KSite& site, std::uint32_t row, std::uint32_t n, float acc) {
    const float y = acc != acc ? __uint_as_float(kCanonicalNan) : acc;   // dense_a8_canonical (:1101)
    const KSeg& sg = site.seg[find_seg(site, n)];
    const std::uint64_t at = static_cast<std::uint64_t>(row) * sg.stride + (n - sg.begin);
    if constexpr (S == Store::f32) static_cast<float*>(sg.out)[at] = y;
    else if constexpr (S == Store::f32_round_bf16) static_cast<float*>(sg.out)[at] = bf16_float(bf16_bits(y));
    else if constexpr (S == Store::bf16) static_cast<std::uint16_t*>(sg.out)[at] = static_cast<std::uint16_t>(bf16_bits(y));
    else {
        const float gated = amd_mul(site.gate[static_cast<std::uint64_t>(row) * site.gate_stride], y);
        static_cast<std::uint16_t*>(sg.out)[at] = static_cast<std::uint16_t>(bf16_bits(gated));
    }
}
    static bool dense_t2_a16_enabled() { return ::strix::cuda::qwen_dense::dense_t2_a16_enabled(); }
    static bool dense_t4_enabled() { return ::strix::cuda::qwen_dense::dense_t4_enabled(); }
    // Diagnostics (microbench only, scheduling-only), as in #735's launcher: ring depth of the aligned-window
    // tile2 (2 default, 3 or 4) and the shared-memory carveout hint in percent (< 0: none).
    static int dense_t2_ring_stages() {
        static const int ns = [] { const char* e = strix::envcfg::text("STRIX_QRTX_DENSE_T2_NS"); return e && e[0] ? std::atoi(e) : 2; }();
        return ns;
    }
    static int dense_carveout_percent() {
        const char* cv = strix::envcfg::text("STRIX_QRTX_DENSE_CARVEOUT");
        return cv && cv[0] ? std::atoi(cv) : -1;
    }
};
// ---------------------------------------------------------------------------------------------------------------------
// rows_shared_activation_scalar_kernel (src/hip_shared_expert_launcher.hip:535-562), 256 threads, blockIdx.y = row.
__global__ __launch_bounds__(256) void shared_activation_kernel(float* workspace_base, std::uint64_t workspace_stride,
                                                                const std::uint16_t* input_base, std::uint64_t input_stride,
                                                                const float* scalar, std::uint32_t intermediate,
                                                                std::uint32_t hidden, amdmath::Tables tables) {
    __shared__ float scratch[256];
    float* workspace = workspace_base + blockIdx.y * workspace_stride;
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < intermediate) {
        const float gate = workspace[index];
        const float up = workspace[intermediate + index];
        const float den = __fadd_rn(1.0F, amdmath::amd_expf(tables, -gate));
        workspace[index] = __fmul_rn(__fdiv_rn(gate, den), up);
    }
    if (blockIdx.x != 0) return;
    const std::uint16_t* input = input_base + blockIdx.y * input_stride;
    float sum = 0.0F;
    for (std::uint32_t feature = threadIdx.x; feature < hidden; feature += 256U)
        sum = __fmaf_rn(scalar[feature], bf16_float(input[feature]), sum);   // v_dual_fmac_f32 (contracted in the ISA)
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = 128U; stride > 0U; stride >>= 1U) {   // HIP :555-558, v_add_f32 upper + lower
        if (threadIdx.x < stride) scratch[threadIdx.x] = __fadd_rn(scratch[threadIdx.x + stride], scratch[threadIdx.x]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        workspace[intermediate] = __fdiv_rn(1.0F, __fadd_rn(1.0F, amdmath::amd_expf(tables, -scratch[0])));
}





// TURBO onfly route: the shared nvfp4-a8-v1 GEMV bound through this site's segments and store forms.
namespace a8 = strix::cuda::core::nvfp4_a8;
struct K4Site {
    KSite k;
    const std::uint8_t* codes[kMaxSegments];
    const std::uint8_t* scales[kMaxSegments];
    float half_ts[kMaxSegments];
};
template <Store S> struct Nvfp4Policy {
    K4Site s;
    std::uint32_t row_base;
    __device__ __forceinline__ std::uint32_t outputs() const { return s.k.outputs; }
    __device__ __forceinline__ a8::Row operand(std::uint32_t n) const {
        const unsigned m = find_seg(s.k, n);
        const std::uint64_t i = n - s.k.seg[m].begin;
        return {s.codes[m] + i * (s.k.width / 2U), s.scales[m] + i * (s.k.width / 16U), s.half_ts[m]};
    }
    __device__ __forceinline__ void store(std::uint32_t r, std::uint32_t n, float y) const {
        DenseNumeric::store<S>(s.k, row_base + r, n, y);   // main's store_value (body moved with #713)
    }
};
// Launch shape (rows per warp chunk = blockIdx.y split, outputs per warp): parallelism only, the bits never change.
std::uint32_t onfly_rw_hook = 0, onfly_out_hook = 0;   // test / tuning hooks (0 = the measured table below)
template <Store S, std::uint32_t OUT, std::uint32_t RMAX>
cudaError_t launch_nvfp4_shape(cudaStream_t stream, const Nvfp4Policy<S>& p, std::uint32_t rows, const std::int8_t* codes,
                               const float* deltas, std::uint32_t pitch) {
    constexpr std::uint32_t kWarps = 8;
    const dim3 grid((p.s.k.outputs + OUT * kWarps - 1U) / (OUT * kWarps), (rows + RMAX - 1U) / RMAX);
    a8::rows_gemv_kernel<Nvfp4Policy<S>, OUT, RMAX, kWarps>
        <<<grid, kWarps * 32U, 0, stream>>>(p, rows, p.s.k.width, codes, p.s.k.width, deltas, pitch);
    return cudaGetLastError();
}
template <Store S, std::uint32_t OUT>
cudaError_t launch_nvfp4_rw(cudaStream_t stream, const Nvfp4Policy<S>& p, std::uint32_t rw, std::uint32_t rows,
                            const std::int8_t* codes, const float* deltas, std::uint32_t pitch) {
    switch (rw) {
    case 1: return launch_nvfp4_shape<S, OUT, 1>(stream, p, rows, codes, deltas, pitch);
    case 2: return launch_nvfp4_shape<S, OUT, 2>(stream, p, rows, codes, deltas, pitch);
    case 4: return launch_nvfp4_shape<S, OUT, 4>(stream, p, rows, codes, deltas, pitch);
    default: return launch_nvfp4_shape<S, OUT, 8>(stream, p, rows, codes, deltas, pitch);
    }
}
template <Store S> cudaError_t launch_nvfp4(cudaStream_t stream, const K4Site& site, std::uint32_t rows,
                                            const std::int8_t* codes, const float* deltas, std::uint32_t pitch) {
    if (rows > 65535U * 8U) return cudaErrorInvalidValue;
    std::uint32_t rw = onfly_rw_hook, out = onfly_out_hook;
    // Measured table (nvfp4_a8_bench, PRO sm_120 2.8 GHz, Flash-Next shapes, best of rows-per-warp 1/2/4/8 x outputs 1/2):
    // wide sites (>= 8192 outputs: GDN / QSA in) want min(pow2(rows), 8) rows per warp, one output per warp up to 2 rows;
    // narrow sites (GDN / QSA out, shared expert) want half the rows per warp and one output per warp.
    const bool wide = site.k.outputs >= 8192U;
    std::uint32_t p2 = 1;
    while (p2 < rows && p2 < 8U) p2 <<= 1U;
    if (rw == 0) rw = wide ? p2 : (rows <= 2U ? 1U : (p2 / 2U < 8U ? p2 / 2U : 8U));
    if (out == 0) out = wide && rows > 2U ? 2U : 1U;
    const Nvfp4Policy<S> p{site, 0};
    return out == 1U ? launch_nvfp4_rw<S, 1>(stream, p, rw, rows, codes, deltas, pitch)
                     : launch_nvfp4_rw<S, 2>(stream, p, rw, rows, codes, deltas, pitch);
}

std::unordered_map<const void*, OnflyImage>& onfly_images() {
    static std::unordered_map<const void*, OnflyImage> images;
    return images;
}
thread_local bool onfly_scope = false;

// Every segment registered with the site's shape -> the NVFP4 operands; any miss keeps the whole site on Q8_0.
bool onfly_site(const Site& site, const KSite& k, K4Site& out) {
    if (!onfly_scope || onfly_images().empty() || site.width % 32U != 0U) return false;
    out.k = k;
    for (std::uint32_t m = 0; m < site.segments; ++m) {
        const auto it = onfly_images().find(site.segment[m].weight);
        if (it == onfly_images().end() || it->second.width != site.width || it->second.outputs != site.segment[m].outputs)
            return false;
        out.codes[m] = it->second.codes;
        out.scales[m] = it->second.scales;
        out.half_ts[m] = 0.5F * it->second.tensor_scale;
    }
    return true;
}

}  // namespace

void onfly_register(const void* q8_weight, const OnflyImage& image) { onfly_images()[q8_weight] = image; }
void onfly_clear() { onfly_images().clear(); }
std::size_t onfly_count() { return onfly_images().size(); }
void onfly_set_launch(std::uint32_t rows_per_warp, std::uint32_t outputs_per_warp) {
    onfly_rw_hook = rows_per_warp;
    onfly_out_hook = outputs_per_warp;
}
OnflyScope::OnflyScope(bool on) : previous_(onfly_scope) { onfly_scope = on; }
OnflyScope::~OnflyScope() { onfly_scope = previous_; }

cudaError_t quantize(cudaStream_t stream, std::uint32_t rows, std::uint32_t width, Input type, const void* input,
                     std::uint64_t input_stride, std::int8_t* codes, float* deltas, std::uint32_t pitch) {
    if (!rows) return cudaSuccess;
    if (width == 0U || width % 64U != 0U || pitch < rows || !input || !codes || !deltas) return cudaErrorInvalidValue;
    const std::uint64_t items = static_cast<std::uint64_t>(rows) * (width / 8U);
    const unsigned blocks = static_cast<unsigned>((items + 255U) / 256U);
    if (type == Input::bf16) quantize_kernel<Input::bf16><<<blocks, 256, 0, stream>>>(rows, width, input, input_stride, codes, deltas, pitch);
    else quantize_kernel<Input::f32><<<blocks, 256, 0, stream>>>(rows, width, input, input_stride, codes, deltas, pitch);
    return cudaGetLastError();
}

template<class Capture>
cudaError_t project_observed(cudaStream_t stream, const Site& site, std::uint32_t rows, const std::int8_t* codes, const float* deltas,
                    std::uint32_t pitch) {
    if (!rows) return cudaSuccess;
    if (site.width == 0U || site.width % 64U != 0U || site.segments == 0U || site.segments > kMaxSegments || pitch < rows ||
        (reinterpret_cast<std::uintptr_t>(codes) & 15U) != 0U)
        return cudaErrorInvalidValue;
    if (site.store == Store::bf16_gated && !site.gate) return cudaErrorInvalidValue;
    KSite k{};
    std::uint32_t n = 0;
    for (std::uint32_t m = 0; m < site.segments; ++m) {
        const Segment& s = site.segment[m];
        if (!s.weight || !s.output || !s.outputs || (reinterpret_cast<std::uintptr_t>(s.weight) & 3U) != 0U) return cudaErrorInvalidValue;
        k.seg[m] = {static_cast<const unsigned char*>(s.weight), s.output, s.output_stride, n, n + s.outputs};
        n += s.outputs;
    }
    k.nseg = site.segments; k.width = site.width; k.outputs = n; k.gate = site.gate; k.gate_stride = site.gate_stride;
    if (K4Site k4{}; onfly_site(site, k, k4)) {
        switch (site.store) {
        case Store::f32: return launch_nvfp4<Store::f32>(stream, k4, rows, codes, deltas, pitch);
        case Store::f32_round_bf16: return launch_nvfp4<Store::f32_round_bf16>(stream, k4, rows, codes, deltas, pitch);
        case Store::bf16: return launch_nvfp4<Store::bf16>(stream, k4, rows, codes, deltas, pitch);
        case Store::bf16_gated: return launch_nvfp4<Store::bf16_gated>(stream, k4, rows, codes, deltas, pitch);
        }
        return cudaErrorInvalidValue;
    }
    const auto dispatch = [&](const KSite& ks, std::uint32_t r, const std::int8_t* c, const float* d) -> cudaError_t {
        switch (site.store) {
        case Store::f32: return core::dense_a8::launch_project<Store::f32, weight::Q8ImmaSource, DenseNumeric, Capture>(stream, ks, r, c, d, pitch);
        case Store::f32_round_bf16: return core::dense_a8::launch_project<Store::f32_round_bf16, weight::Q8ImmaSource, DenseNumeric, Capture>(stream, ks, r, c, d, pitch);
        case Store::bf16: return core::dense_a8::launch_project<Store::bf16, weight::Q8ImmaSource, DenseNumeric, Capture>(stream, ks, r, c, d, pitch);
        case Store::bf16_gated: return core::dense_a8::launch_project<Store::bf16_gated, weight::Q8ImmaSource, DenseNumeric, Capture>(stream, ks, r, c, d, pitch);
        }
        return cudaErrorInvalidValue;
    };
    // 9..16 rows (core::decode_rows_max): <= 8-row decode pieces. A piece sees rows r0.. through offset codes / deltas /
    // outputs / gates; each (row, output) chain is row-local, so the bits are the tiled kernel's.
    if (rows > core::kDecodePieceRows && rows <= core::decode_rows_max() && k.width <= 6144U && decode_rows_enabled() &&
        rows_ksplit_enabled() && rows16_enabled()) {
#ifdef STRIX_QWEN_DENSE_TEST_DISPATCH
        test_dispatch(rows, true);
#endif
        switch (site.store) {   // one weight pass for 9..16 rows
        case Store::f32: return core::dense_a8::launch_rows_ks<Store::f32, weight::Q8ImmaSource, DenseNumeric, Capture>(stream, k, rows, codes, deltas, pitch);
        case Store::f32_round_bf16: return core::dense_a8::launch_rows_ks<Store::f32_round_bf16, weight::Q8ImmaSource, DenseNumeric, Capture>(stream, k, rows, codes, deltas, pitch);
        case Store::bf16: return core::dense_a8::launch_rows_ks<Store::bf16, weight::Q8ImmaSource, DenseNumeric, Capture>(stream, k, rows, codes, deltas, pitch);
        case Store::bf16_gated: return core::dense_a8::launch_rows_ks<Store::bf16_gated, weight::Q8ImmaSource, DenseNumeric, Capture>(stream, k, rows, codes, deltas, pitch);
        }
        return cudaErrorInvalidValue;
    }
    if (rows > core::kDecodePieceRows && rows <= core::decode_rows_max() && k.width <= 6144U && decode_rows_enabled()) {
#ifdef STRIX_QWEN_DENSE_TEST_DISPATCH
        test_dispatch(rows, false);
#endif
        const std::uint64_t elem = site.store == Store::f32 || site.store == Store::f32_round_bf16 ? 4U : 2U;
        for (std::uint32_t r0 = 0; r0 < rows; r0 += core::kDecodePieceRows) {
            KSite piece = k;
            for (std::uint32_t m = 0; m < k.nseg; ++m)
                piece.seg[m].out = static_cast<unsigned char*>(k.seg[m].out) + std::uint64_t{r0} * k.seg[m].stride * elem;
            if (piece.gate) piece.gate += std::uint64_t{r0} * k.gate_stride;
            const cudaError_t e = dispatch(piece, std::min(rows - r0, core::kDecodePieceRows), codes + std::uint64_t{r0} * k.width, deltas + r0);
            if (e != cudaSuccess) return e;
        }
        return cudaSuccess;
    }
    return dispatch(k, rows, codes, deltas);
}

cudaError_t project(cudaStream_t stream, const Site& site, std::uint32_t rows, const std::int8_t* codes,
                    const float* deltas, std::uint32_t pitch) {
    return project_observed<NoCapture>(stream, site, rows, codes, deltas, pitch);
}
#ifdef STRIX_WFMT_CAPTURE
cudaError_t capture_project_partials(cudaStream_t stream, const Site& site, std::uint32_t rows,
                                    const std::int8_t* codes, const float* deltas, std::uint32_t pitch, float* partials) {
    unsigned outputs = 0;
    for (unsigned i = 0; i < site.segments; ++i) outputs += site.segment[i].outputs;
    const PartialCaptureConfig capture{partials, rows, outputs};
    const auto copied = cudaMemcpyToSymbol(partial_capture_config, &capture, sizeof(capture), 0, cudaMemcpyHostToDevice);
    if (copied != cudaSuccess) return copied;
    return project_observed<PartialCapture>(stream, site, rows, codes, deltas, pitch);
}
#endif

cudaError_t quantize_project(cudaStream_t stream, const Site& site, std::uint32_t rows, Input type, const void* input,
                             std::uint64_t input_stride, std::int8_t* codes, float* deltas, std::uint32_t pitch) {
    const cudaError_t e = quantize(stream, rows, site.width, type, input, input_stride, codes, deltas, pitch);
    return e != cudaSuccess ? e : project(stream, site, rows, codes, deltas, pitch);
}

cudaError_t shared_activation(cudaStream_t stream, std::uint32_t rows, float* workspace, std::uint64_t workspace_stride,
                              const std::uint16_t* input, std::uint64_t input_stride, const float* scalar,
                              std::uint32_t intermediate, std::uint32_t hidden, const amdmath::Tables& tables) {
    if (!rows) return cudaSuccess;
    if (!intermediate || workspace_stride < 2ULL * intermediate || !workspace || !input || !scalar) return cudaErrorInvalidValue;
    const dim3 grid((intermediate + 255U) / 256U, rows);
    shared_activation_kernel<<<grid, 256, 0, stream>>>(workspace, workspace_stride, input, input_stride, scalar, intermediate,
                                                       hidden, tables);
    return cudaGetLastError();
}

cudaError_t shared_expert(cudaStream_t stream, std::uint32_t rows, const SharedExpert& w, const std::uint16_t* input,
                          std::uint64_t input_stride, float* workspace, std::uint64_t workspace_stride, std::uint16_t* output,
                          std::uint64_t output_stride, std::int8_t* codes, float* deltas, std::uint32_t pitch,
                          const amdmath::Tables& tables) {
    Site gate_up{};
    gate_up.store = Store::f32;
    gate_up.width = w.hidden;
    gate_up.segments = 2;
    gate_up.segment[0] = {w.gate_q8, w.intermediate, workspace, workspace_stride};
    gate_up.segment[1] = {w.up_q8, w.intermediate, workspace + w.intermediate, workspace_stride};
    cudaError_t e = quantize_project(stream, gate_up, rows, Input::bf16, input, input_stride, codes, deltas, pitch);
    if (e != cudaSuccess) return e;
    e = shared_activation(stream, rows, workspace, workspace_stride, input, input_stride, w.scalar, w.intermediate, w.hidden, tables);
    if (e != cudaSuccess) return e;
    Site down{};
    down.store = Store::bf16_gated;
    down.width = w.intermediate;
    down.segments = 1;
    down.segment[0] = {w.down_q8, w.hidden, output, output_stride};
    down.gate = workspace + w.intermediate;   // the scalar overwrote up[0] (HIP site.gate[r], :1874)
    down.gate_stride = workspace_stride;
    return quantize_project(stream, down, rows, Input::f32, workspace, workspace_stride, codes, deltas, pitch);
}

}  // namespace strix::cuda::qwen_dense
