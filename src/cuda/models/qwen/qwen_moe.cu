// Exactness comments below describe the default FP32 route target. The explicit
// BF16 route-storage candidate is a separate, unqualified numerical target.
// CUDA routed-MoE prefill of the Qwen A8 profile, bit-identical to the HIP iu8 WMMA path (see include/strix/cuda/qwen_moe.hpp).
//
// Exactness notes (HIP sources at engine commit bf60d9d2; ISA = the gfx1151 disassembly):
// - The routed-MoE HIP TUs are built with -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt -fdenormal-fp-math=ieee
//   and #pragma clang fp contract(off) (hip_moe_q4k_a8_known_rows.hip:14, hip_moe_q4k_a8_probe.hip:12). ISA check of
//   project_prefill_wmma<{0,1},16> (hip_moe_q4k_a8_known_rows.s:193768 / :188834): its 10 v_fma_f32 + 6 v_fmac_f32 +
//   1 v_dual_fmac_f32 are exactly the expf sequence (1 fma + 1 fmac) and three correctly rounded divisions
//   (v_div_scale/v_rcp/3x v_fma/2x v_fmac/v_div_fmas/v_div_fixup): g/(1+e), x/127, hv/delta. The chain epilogue is v_mul/
//   v_sub/v_add only. down_prefill_wmma<{1,0},16> (:197936 / :200810) has no FMA at all; float(r) is v_add_f32 0xcb400000
//   (x + -1.5*2^23, the same value as the source subtraction). prefill_prologue<1> (hip_moe_q4k_a8_probe.s:7306): 6 v_fma +
//   4 v_fmac = two correctly rounded divisions, v_rndne for rintf, v_maxmin for the clamp.
// - The merge is inline asm v_fmac_f32 current, w[slot], dot[slot] in slot order from current = 0
//   (hip_moe_experts_launcher.s:3152 + 215..470): __fmaf_rn(w, dot, current).
// - The expf of the SwiGLU (hip_moe_q4k_a8_known_rows.s:193768+3836..3857) multiplies g by -log2e (0xbfb8aa3b), fma with
//   -log2e, fmac with -log2e_lo (0xb2a5705f), rndne, v_exp, ldexp, then the two range selects on g with the thresholds
//   negated: the same operations as amd_expf(-g) (negation is exact).
// - Integer dots (iu8 WMMA in HIP, mma.sync s8 here) are exact: |dot| <= 32*127*31 < 2^22, accumulated from the magic seed
//   0x4b400000 so that int_as_float(acc) - 1.5*2^23 is float(dot) exactly (group_float, known_rows.hip:730).
// - Gate/up per group: HIP i = float(sc)*float(r), u = d*i, j = float(mn)*s, v = dm*j (known_rows.hip:833-836). Here
//   u = (d*sc)*float(r) and v = (dm*mn)*s with d*sc and dm*mn precomputed: d is FP16 (11-bit significand) and sc, mn < 64,
//   so d*sc and dm*mn are exact in FP32, and sc*r (< 2^23), mn*s (< 2^18) are exact too: both sides are the exact product
//   d*sc*r rounded once (same for zero signs, infinities and NaN). Every other operation is HIP's, in HIP's order.
#include "strix/env_flags.hpp"
#include "../../core/moe_a8.cuh"
#include "strix/cuda/core/decode_rows.hpp"
#include "strix/cuda/qwen_moe.hpp"
#include "strix/cuda/qwen_quant.cuh"
#include "strix/cuda/qwen_route_bf16.hpp"

#include <cuda_fp16.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::cuda::qwen_moe {
namespace {
using namespace ::strix::cuda::core::moe_a8;

// Storage primitives only; the HIP DA16 arithmetic/certificate is not imported.
// CUDA spelling of the HIP width-limited xor shuffle used by the shared include (integer max: bit-neutral).
__device__ __forceinline__ unsigned qwen_moe_shfl_xor(unsigned v,int delta,int width) {
    return __shfl_xor_sync(0xffffffffU,v,delta,width);
}
#define __shfl_xor qwen_moe_shfl_xor
#include "../../../hip_moe_route_bf16.inc"
#undef __shfl_xor

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// ---------------------------------------------------------------------------------------------------------------------
// prologue: prefill_prologue<true> (probe.hip:159-173) with quantize_row<true> (probe.hip:41-73). One CTA per row; the
// group -> warp mapping differs from HIP (1024 threads) but each group is computed by one warp exactly as in HIP and the
// flags are an OR, so every byte is the same.
__global__ __launch_bounds__(256) void prologue_kernel(Rows v, std::uint32_t clear_words) {
    const std::uint32_t row = blockIdx.x, t = threadIdx.x, lane = t & 31U, wave = t >> 5;
    __shared__ __align__(16) signed char local_codes[hidden];
    __shared__ float local_deltas[hidden / 32U];
    __shared__ int local_sums[hidden / 32U];
    __shared__ unsigned route_bits, bad;
    if (t == 0) { route_bits = 0U; bad = 0U; }
    float* clear = at(v.accumulation, row);
    for (std::uint32_t i = t; i < clear_words; i += blockDim.x) clear[i] = 0.0F;
    __syncthreads();
    if (t < top_k && (at(v.routes, row)[t] >= experts || !finite(at(v.route_weights, row)[t]))) atomicOr(&route_bits, invalid_route);
    __syncthreads();
    if (route_bits == 0U) {
        const std::uint16_t* input = at(v.input, row);
        for (std::uint32_t group = wave; group < hidden / 32U; group += blockDim.x / 32U) {
            const std::uint32_t i = group * 32U + lane;
            float x = __uint_as_float(static_cast<unsigned>(input[i]) << 16U);
            if (!finite(x)) { atomicOr(&bad, nonfinite_activation); x = 0.0F; }
            float a = fabsf(x);
            for (int s = 16; s != 0; s /= 2) a = fmaxf(a, __shfl_xor_sync(0xffffffffU, a, s));
            float delta = a == 0.0F ? 1.0F : __fdiv_rn(a, 127.0F);
            if (!(delta > 0.0F) || !finite(delta)) { atomicOr(&bad, invalid_delta); delta = 1.0F; }
            const float rounded = rintf(__fdiv_rn(x, delta));
            const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
            local_codes[i] = static_cast<signed char>(code);
            int sum = code;
            for (int s = 16; s != 0; s /= 2) sum += __shfl_xor_sync(0xffffffffU, sum, s);
            if (lane == 0) { local_deltas[group] = delta; local_sums[group] = sum; }
        }
    }
    __syncthreads();
    const unsigned failure = route_bits | bad;
    if (route_bits == 0U && bad == 0U) {
        std::int8_t* codes = at(v.codes, row);
        for (std::uint32_t i = t; i < hidden / 16U; i += blockDim.x)
            reinterpret_cast<int4*>(codes)[i] = reinterpret_cast<const int4*>(local_codes)[i];
        for (std::uint32_t i = t; i < hidden / 32U; i += blockDim.x) { at(v.deltas, row)[i] = local_deltas[i]; at(v.sums, row)[i] = local_sums[i]; }
    }
    if (t == 0) *at(v.status, row) = failure;
}

// ---------------------------------------------------------------------------------------------------------------------
// csr: build_prefill_csr_waves (known_rows.hip:460-532) with cap 16, transcribed (32 waves of 32 lanes; the AMD
// ballot/shfl peer loop becomes __match_any_sync, and a __syncwarp separates the position reads from the leader updates,
// which the AMD wave executes in lockstep). Also emits this port's tiles (runs of <= 64 assignments of one expert).
__global__ __launch_bounds__(1024) void csr_kernel(Rows v, std::uint32_t rows) {
    constexpr unsigned Waves = 32U, cap = hip_batch, tcap = tile_rows;
    const Csr& csr = v.csr;
    const unsigned t = threadIdx.x, lane = t & 31U, wave = t / 32U, count = rows * top_k, span = (count + Waves - 1U) / Waves;
    const unsigned first = wave * span, last = min(count, first + span);
    __shared__ unsigned position[Waves][256], n_of[512], scan[4][512], hist[16], failure, status, duplicates, maximum;
    if (t == 0) { failure = 0U; status = 0U; duplicates = 0U; maximum = 0U; }
    if (t < 16U) hist[t] = 0U;
    for (unsigned k = t; k < Waves * 256U; k += 1024U) position[k / 256U][k % 256U] = 0U;
    __syncthreads();
    const auto expert_at = [&](unsigned a) { return at(v.routes, a / top_k)[a % top_k]; };
    for (unsigned a = first + lane; a < last; a += 32U) {
        const unsigned e = expert_at(a);
        if (e >= experts || !finite(at(v.route_weights, a / top_k)[a % top_k])) atomicOr(&failure, invalid_route);
        else atomicAdd(&position[wave][e / 2U], 1U << (16U * (e & 1U)));
    }
    for (unsigned r = t; r < rows; r += 1024U) { const unsigned s = *at(v.status, r); if (s) atomicOr(&status, s); }
    __syncthreads();
    const auto fail_summary = [&] {
        if (t < 32U) csr.summary[t] = t == 0U ? failure | status : t == 3U ? count : t == 6U ? rows : t == 7U ? 1U : 0U;
        if (t == 0U) *csr.tile_count = 0U;
    };
    if (failure) { fail_summary(); return; }
    if (t < 256U) {
        unsigned lo = 0U, hi = 0U;
        for (unsigned w = 0; w < Waves; ++w) { const unsigned c = position[w][t]; position[w][t] = lo | hi << 16U; lo += c & 65535U; hi += c >> 16U; }
        if (lo > 65535U || hi > 65535U) atomicOr(&failure, invalid_route);
        n_of[2U * t] = lo; n_of[2U * t + 1U] = hi;
    }
    __syncthreads();
    if (failure) { fail_summary(); return; }
    const unsigned n = t < 512U ? n_of[t] : 0U, nb = (n + cap - 1U) / cap, nt = (n + tcap - 1U) / tcap;
    if (t < 512U) { scan[0][t] = n; scan[1][t] = n != 0U; scan[2][t] = nb; scan[3][t] = nt; }
    __syncthreads();
    for (unsigned d = 1U; d < 512U; d <<= 1U) {
        unsigned x[4] = {};
        if (t >= d && t < 512U) for (unsigned k = 0; k < 4U; ++k) x[k] = scan[k][t - d];
        __syncthreads();
        if (t >= d && t < 512U) for (unsigned k = 0; k < 4U; ++k) scan[k][t] += x[k];
        __syncthreads();
    }
    for (unsigned step = first; step < last; step += 32U) {
        const unsigned a = step + lane;
        const bool valid = a < last;
        const unsigned e = valid ? expert_at(a) : 0xffffffffU;
        const unsigned vm = __ballot_sync(0xffffffffU, valid);
        const unsigned peers = __match_any_sync(0xffffffffU, e) & vm;
        unsigned slot = 0U;
        if (valid) slot = scan[0][e] - n_of[e] + ((position[wave][e / 2U] >> (16U * (e & 1U))) & 65535U);
        __syncwarp();
        if (valid) {
            const unsigned below = peers & ((1U << lane) - 1U);
            const unsigned at_pos = slot + static_cast<unsigned>(__popc(below));
            csr.rows[at_pos] = a / top_k; csr.slots[at_pos] = a % top_k;
            if (below == 0U) atomicAdd(&position[wave][e / 2U], static_cast<unsigned>(__popc(peers)) << (16U * (e & 1U)));
        }
        __syncwarp();
    }
    __syncthreads();
    if (t < experts) {
        const unsigned off = scan[0][t] - n, touched = scan[1][t] - (n != 0U), batch = scan[2][t] - nb, tile = scan[3][t] - nt;
        csr.offsets[t] = off;
        if (t == experts - 1U) csr.offsets[experts] = off + n;
        if (n) {
            csr.touched[touched] = t; atomicMax(&maximum, n);
            for (unsigned b = 0; b < nb; ++b) {
                csr.batch_experts[batch + b] = t; csr.batch_begins[batch + b] = off + b * cap;
                atomicAdd(&hist[8U + min(8U, min(cap, n - b * cap)) - 1U], 1U);
            }
            for (unsigned b = 0; b < nt; ++b) { csr.tile_experts[tile + b] = t; csr.tile_begins[tile + b] = off + b * tcap; }
            unsigned unique = 0U, previous = 0xffffffffU;
            for (unsigned k = 0; k < n; ++k) { const unsigned r = csr.rows[off + k]; unique += r != previous; previous = r; }
            atomicAdd(&hist[min(8U, unique) - 1U], 1U); atomicAdd(&duplicates, n - unique);
        }
    }
    __syncthreads();
    if (t == 0U) { csr.batch_begins[scan[2][511]] = count; csr.tile_begins[scan[3][511]] = count; *csr.tile_count = scan[3][511]; }
    if (t < 32U)
        csr.summary[t] = t == 0U ? status : t == 1U ? scan[1][511] : t == 2U ? scan[2][511] : t == 3U ? count : t == 4U ? duplicates
            : t == 5U ? maximum : t == 6U ? rows : t == 7U ? 1U : t >= 8U && t < 24U ? hist[t - 8U] : 0U;
}

// ---------------------------------------------------------------------------------------------------------------------
// project: project_prefill_wmma<Q5K,16> (known_rows.hip:860-909) for a tile of <= 64 assignments of one expert and 64
// outputs of both gate and up. 16 warps: warp w covers rows 16*(w%4).. and outputs 16*(w/4).. of both matrices
// (2 n8 tiles each). Per (row, output) element and matrix, four chains acc[p] (p = pair of groups 2p, 2p+1 of a K block),
// K blocks ascending, per pair (gate_up_chains :831-840):
//   u0 = (d*sc0)*float(r0), v0 = (dm*mn0)*s0, t0 = u0 - v0; u1, v1, t1 likewise; c0 = e0*t0, c1 = e1*t1;
//   pair = c0 + c1; acc[p] = acc[p] + pair;      fold (Chains::fold :782-785): (acc0 + acc1) + (acc2 + acc3).
// Pipeline: cp.async of K block b+1 (codes, raw weight blocks, deltas/sums) while block b computes.

// Serial down_activation_quantize of a tile's g/u values (Gt, Ut: [tile_rows][OUTS] FP32): shared by the exact and the moe_mma
// project kernels (code motion only: the exact kernel's bytes are unchanged).
template <unsigned OUTS, unsigned THREADS>
__device__ __forceinline__ void project_activation_tile(const float* Gt, const float* Ut, const Rows& v, const amdmath::Tables& tables,
                                                        const unsigned* row_of, const unsigned* slot_of, unsigned count, unsigned out0) {
    const Csr& csr = v.csr;
    const unsigned lane = threadIdx.x & 31U, warp = threadIdx.x >> 5;
    // Serial down_activation_quantize per (assignment, 32-output group), one warp per job (known_rows.hip:888-909).
    const bool write_intermediates = v.intermediates.base != nullptr;
    for (unsigned job = warp; job < count * (OUTS / 32U); job += THREADS / 32U) {
        const unsigned a = job / (OUTS / 32U), quad = job % (OUTS / 32U), r = row_of[a], slot = slot_of[a];
        const unsigned feature = out0 + 32U * quad + lane, at_slot = slot * 2U * ffn + feature;
        const float gv = Gt[a * OUTS + 32U * quad + lane], uv = Ut[a * OUTS + 32U * quad + lane];
        const float ex = amdmath::amd_expf(tables, -gv);
        const float hv = __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0F, ex)), uv);
        float* output = write_intermediates ? at(v.intermediates, r) : nullptr;
        if (output) output[at_slot + ffn] = canon(uv);
        const unsigned bad = __ballot_sync(0xffffffffU, !finite(gv) || !finite(uv) || !finite(hv));
        if (bad != 0U) {
            if (output) output[at_slot] = canon(gv);
            if (lane == 0) { atomicOr(at(v.status, r), projection_nonfinite); atomicOr(csr.summary, projection_nonfinite); }
            continue;
        }
        float x = fabsf(hv);
        for (int s = 16; s != 0; s /= 2) x = fmaxf(x, __shfl_xor_sync(0xffffffffU, x, s));
        const bool tiny = x > 0.0F && x < 0x1p-119F;
        const float delta = x == 0.0F || tiny ? 1.0F : __fdiv_rn(x, 127.0F);
        const float rounded = tiny ? 0.0F : rintf(__fdiv_rn(hv, delta));
        const int code = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, rounded)));
        int sum = code;
        for (int s = 16; s != 0; s /= 2) sum += __shfl_xor_sync(0xffffffffU, sum, s);
        if (output) output[at_slot] = hv;
        at(v.down_codes, r)[slot * ffn + feature] = static_cast<std::int8_t>(code);
        if (lane == 0) { at(v.down_deltas, r)[slot * (ffn / 32U) + feature / 32U] = delta; at(v.down_sums, r)[slot * (ffn / 32U) + feature / 32U] = sum; }
    }
}


struct MoeBinding {
 using Rows=qwen_moe::Rows; using Csr=qwen_moe::Csr; using Tables=amdmath::Tables;
 static constexpr unsigned hidden=qwen_moe::hidden,ffn=qwen_moe::ffn,projection_nonfinite=qwen_moe::projection_nonfinite;
 template<unsigned OUTS,unsigned THREADS>
 __device__ static void activation_tile(const float* g,const float* u,const Rows& v,const Tables& tables,
                                       const unsigned* rows,const unsigned* slots,unsigned count,unsigned out) {
     project_activation_tile<OUTS,THREADS>(g,u,v,tables,rows,slots,count,out);
 }
 template<bool BF> __device__ static void route_store(float* p,std::uint64_t index,float v) { moe_route_store<BF>(p,index,v); }
 __device__ static float native_activation(float g,float u) { return g/(1.0F+expf(-g))*u; }
};
using Q4Source=weight::OperandSource<weight::GgmlTag<12>,weight::SignedA8Tag,OperandArch,weight::CudaFourChainImma>;
using Q5Source=weight::OperandSource<weight::GgmlTag<13>,weight::SignedA8Tag,OperandArch,weight::CudaFourChainImma>;
using Q51Source=weight::OperandSource<weight::GgmlTag<7>,weight::SignedA8Tag,OperandArch,weight::CudaFourChainImma>;
using Q8Source=weight::OperandSource<weight::GgmlTag<8>,weight::SignedA8Tag,OperandArch,weight::CudaFourChainImma>;
using NativeSource=weight::OperandSource<weight::RuntimeGgmlTag,weight::NativeScalarTag,OperandArch,weight::CudaNativeFp32>;
bool decode_rows_enabled() { static const bool on=strix::envcfg::flag("STRIX_QRTX_DECODE_KERNELS",true); return on; }
template<class K> void set_smem(K kernel,unsigned bytes);

template<unsigned T> using GsqSource=weight::OperandSource<weight::GgmlTag<T>,weight::SignedA8Tag,OperandArch,weight::GsqIq3SA8Dot4Cuda>;
template<unsigned T> void launch_gsq_project(const Rows& v,unsigned rows,const Weights& w,const amdmath::Tables& tables,cudaStream_t stream) {
    using Source=GsqSource<T>;
    if(rows<=kRowsMax && decode_rows_enabled()) {
        project_rows_kernel<Source,MoeBinding><<<dim3(csr_sizes(rows).max_tiles,ffn/32),1024,0,stream>>>(w.gate,w.up,v,tables);
    } else {
        using L=ProjLayout<Source>;
        static_assert(L::total<=101376,"GSQ projection shared budget");
        static bool once=(set_smem(project_kernel<Source,MoeBinding>,L::total),true); (void)once;
        project_kernel<Source,MoeBinding><<<dim3(csr_sizes(rows).max_tiles,ffn/kProjOuts),kProjThreads,L::total,stream>>>(w.gate,w.up,v,tables);
    }
    check(cudaGetLastError(),"GSQ A8 gate/up");
}
template<unsigned T> void launch_gsq_down(const Rows& v,unsigned rows,const Weights& w,cudaStream_t stream) {
    using Source=GsqSource<T>;
    if(rows<=kRowsMax && decode_rows_enabled()) {
        down_rows_kernel<Source,MoeBinding><<<dim3(csr_sizes(rows).max_tiles,hidden/8),256,0,stream>>>(w.down,v);
    } else {
        using L=DownLayout<Source>;
        static bool once=(set_smem(down_kernel<Source,MoeBinding>,L::total),true); (void)once;
        down_kernel<Source,MoeBinding><<<dim3(csr_sizes(rows).max_tiles,hidden/kDownOuts),kDownThreads,L::total,stream>>>(w.down,v);
    }
    check(cudaGetLastError(),"GSQ A8 down");
}
void validate_expert_arithmetic(const Weights& w,bool moe_mma) {
    if(w.arithmetic==ExpertArithmetic::incumbent) return;
    if(w.arithmetic!=ExpertArithmetic::gsq_a8_dot4 || !gsq_types(w))
        throw std::invalid_argument("gsq-iq3s-a8-dot4-cuda-v1: unsupported expert storage");
    if(moe_mma) throw std::invalid_argument("gsq-iq3s-a8-dot4-cuda-v1: single-chain moe_mma composition unqualified");
}
// Generic expert kernels live once in the backend core.
// merge / commit (hip_moe_experts_launcher.hip:2660-2708).
template <bool RouteBF16 = false>
__global__ __launch_bounds__(256) void merge_kernel(Rows v) {
    const unsigned row = blockIdx.y;
    std::uint32_t* status = at(v.status, row);
    std::uint32_t* header = v.csr.summary;
    if (header[0] || *status) return;
    const unsigned out = blockIdx.x * blockDim.x + threadIdx.x;
    if (out >= hidden) return;
    const std::uint32_t* routes = at(v.routes, row);
    for (unsigned slot = 0; slot < top_k; ++slot) if (routes[slot] >= experts) return;
    const float* rawdots = at(v.rawdots, row);
    const float* w = at(v.route_weights, row);
    float current = 0.0F;
    if constexpr (RouteBF16) {
#pragma unroll
        for (unsigned slot = 0; slot < top_k; ++slot)
            current = __fmaf_rn(w[slot], moe_route_load<true>(rawdots,
                static_cast<std::uint64_t>(slot) * hidden + out), current);
    } else {
#pragma unroll
    for (unsigned slot = 0; slot < top_k; ++slot) current = __fmaf_rn(w[slot], rawdots[slot * hidden + out], current);
    }
    at(v.accumulation, row)[out] = canon(current);
    if ((__float_as_uint(current) & 0x7fffffffU) >= 0x7f7f8000U) { atomicOr(status, 8U); atomicOr(header, 8U); }
}

__global__ __launch_bounds__(256) void commit_kernel(Rows v, std::uint32_t rows, std::uint32_t* record) {
    std::uint32_t* header = v.csr.summary;
    const std::uint32_t failure = header[0];
    if (blockIdx.x == 0U && blockIdx.y == 0U) {
        __shared__ std::uint32_t first_bad;
        if (threadIdx.x == 0U) first_bad = rows;
        __syncthreads();
        for (std::uint32_t r = threadIdx.x; r < rows; r += blockDim.x)
            if (*at(v.status, r)) { atomicMin(&first_bad, r); break; }
        __syncthreads();
        if (threadIdx.x == 0U) {
            if (failure & 7U) { for (const unsigned word : {1U, 2U, 4U, 5U}) header[word] = 0U; for (unsigned word = 8U; word < 24U; ++word) header[word] = 0U; }
            if (first_bad < rows && record) {
                const std::uint32_t bits = *at(v.status, first_bad);
                if (bits != 0U && record[0] == 0U) { record[1] = bits; record[0] = 2U; }  // DeferredMoeDomain::q4k_a8
            }
        }
    }
    if (failure) return;
    const unsigned out = blockIdx.x * blockDim.x + threadIdx.x;
    if (out < hidden) {
        const std::uint32_t bits = __float_as_uint(at(v.accumulation, blockIdx.y)[out]);
        at(v.output, blockIdx.y)[out] = static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
    }
}

template <class K> void set_smem(K kernel, unsigned bytes) {
    check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)), "MoE smem attribute");
}

}  // namespace

CsrSizes csr_sizes(std::uint32_t rows) {
    const std::uint32_t a = rows * top_k;
    const auto bound = [&](std::uint32_t cap) { const std::uint32_t x = experts + (a + cap - 1U) / cap; return x < a ? x : a; };
    return {a, bound(hip_batch), bound(tile_rows)};
}

void enqueue_prologue(const Rows& v, std::uint32_t rows, std::uint32_t clear_words, cudaStream_t s) {
    prologue_kernel<<<rows, 256, 0, s>>>(v, clear_words);
    check(cudaGetLastError(), "MoE prologue");
}
void enqueue_csr(const Rows& v, std::uint32_t rows, cudaStream_t s) {
    if (rows * top_k > 65535U * 32U) throw std::invalid_argument("MoE CSR rows");
    csr_kernel<<<1, 1024, 0, s>>>(v, rows);
    check(cudaGetLastError(), "MoE CSR");
}
void enqueue_project(const Rows& v, std::uint32_t rows, const Weights& w, const amdmath::Tables& tables, cudaStream_t s, bool moe_mma) {
    validate_expert_arithmetic(w,moe_mma);
    if(w.arithmetic==ExpertArithmetic::gsq_a8_dot4) {
        switch(w.gate_type) {
        case 18: launch_gsq_project<18>(v,rows,w,tables,s); break;
        case 21: launch_gsq_project<21>(v,rows,w,tables,s); break;
        case 22: launch_gsq_project<22>(v,rows,w,tables,s); break;
        case 23: launch_gsq_project<23>(v,rows,w,tables,s); break;
        case 42: launch_gsq_project<42>(v,rows,w,tables,s); break;
        default: throw std::invalid_argument("GSQ A8 gate storage");
        }
        return;
    }

    // Decode-shaped rows: up to core::decode_rows_max() (9..16 rows: the tile loop below runs chunks of 8). Under the FAST
    // moe_mma form the cut stays at kRowsMax, so 9..16-row FAST launches keep the qualified tensor-core form.
    if (rows <= (moe_mma ? kRowsMax : core::decode_rows_max()) && decode_rows_enabled()) {
        const dim3 grid(csr_sizes(rows).max_tiles, ffn / 32U);
        if (w.gate_type == type_q5_k) project_rows_kernel<Q5Source, MoeBinding><<<grid, 1024, 0, s>>>(w.gate, w.up, v, tables);
        else if (w.gate_type == type_q4_k) project_rows_kernel<Q4Source, MoeBinding><<<grid, 1024, 0, s>>>(w.gate, w.up, v, tables);
        else throw std::invalid_argument("MoE gate type");
        check(cudaGetLastError(), "MoE gate/up rows");
        return;
    }
    if (moe_mma) {
        constexpr unsigned MT = MOE_FP_MT, NW = MOE_FP_NW;
        const dim3 grid(csr_sizes(rows).max_tiles, ffn / (16U * NW));
        if (w.gate_type == type_q5_k) {
            using L = FastProjLayout<Q5Source, MT, NW>;
            static bool once = (set_smem(project_fast_kernel<Q5Source, MoeBinding, MT, NW>, L::total), true); (void)once;
            project_fast_kernel<Q5Source, MoeBinding, MT, NW><<<grid, L::THREADS, L::total, s>>>(w.gate, w.up, v, tables);
        } else if (w.gate_type == type_q4_k) {
            using L = FastProjLayout<Q4Source, MT, NW>;
            static bool once = (set_smem(project_fast_kernel<Q4Source, MoeBinding, MT, NW>, L::total), true); (void)once;
            project_fast_kernel<Q4Source, MoeBinding, MT, NW><<<grid, L::THREADS, L::total, s>>>(w.gate, w.up, v, tables);
        } else throw std::invalid_argument("MoE gate type");
        check(cudaGetLastError(), "MoE gate/up moe_mma");
        return;
    }
    const dim3 grid(csr_sizes(rows).max_tiles, ffn / kProjOuts);
    if (w.gate_type == type_q5_k) {
        static bool once = (set_smem(project_kernel<Q5Source, MoeBinding>, ProjLayout<Q5Source>::total), true); (void)once;
        project_kernel<Q5Source, MoeBinding><<<grid, kProjThreads, ProjLayout<Q5Source>::total, s>>>(w.gate, w.up, v, tables);
    } else if (w.gate_type == type_q4_k) {
        static bool once = (set_smem(project_kernel<Q4Source, MoeBinding>, ProjLayout<Q4Source>::total), true); (void)once;
        project_kernel<Q4Source, MoeBinding><<<grid, kProjThreads, ProjLayout<Q4Source>::total, s>>>(w.gate, w.up, v, tables);
    } else throw std::invalid_argument("MoE gate type");
    check(cudaGetLastError(), "MoE gate/up");
}
template <bool RouteBF16>
void enqueue_down_storage(const Rows& v, std::uint32_t rows, const Weights& w, cudaStream_t s, bool moe_mma) {
    validate_expert_arithmetic(w,moe_mma);
    if(w.arithmetic==ExpertArithmetic::gsq_a8_dot4) {
        if(w.down_type==20) launch_gsq_down<20>(v,rows,w,s);
        else if(w.down_type==42) launch_gsq_down<42>(v,rows,w,s);
        else throw std::invalid_argument("GSQ A8 down storage");
        return;
    }

    // Decode-shaped rows: up to core::decode_rows_max() (9..16 rows: the tile loop below runs chunks of 8). Under the FAST
    // moe_mma form the cut stays at kRowsMax, so 9..16-row FAST launches keep the qualified tensor-core form.
    if (rows <= (moe_mma ? kRowsMax : core::decode_rows_max()) && decode_rows_enabled()) {
        const dim3 grid(csr_sizes(rows).max_tiles, hidden / 8U);
        if (w.down_type == type_q5_1) down_rows_kernel<Q51Source, MoeBinding, RouteBF16><<<grid, 256, 0, s>>>(w.down, v);
        else if (w.down_type == type_q8_0) down_rows_kernel<Q8Source, MoeBinding, RouteBF16><<<grid, 256, 0, s>>>(w.down, v);
        else throw std::invalid_argument("MoE down type");
        check(cudaGetLastError(), "MoE down rows");
        return;
    }
    if (moe_mma) {
        constexpr unsigned MT = MOE_FD_MT, NW = MOE_FD_NW;
        const dim3 grid(csr_sizes(rows).max_tiles, hidden / (16U * NW));
        if (w.down_type == type_q5_1) {
            using L = FastDownLayout<Q51Source, MT, NW>;
            static bool once = (set_smem(down_fast_kernel<Q51Source, MoeBinding, RouteBF16, MT, NW>, L::total), true); (void)once;
            down_fast_kernel<Q51Source, MoeBinding, RouteBF16, MT, NW><<<grid, L::THREADS, L::total, s>>>(w.down, v);
        } else if (w.down_type == type_q8_0) {
            using L = FastDownLayout<Q8Source, MT, NW>;
            static bool once = (set_smem(down_fast_kernel<Q8Source, MoeBinding, RouteBF16, MT, NW>, L::total), true); (void)once;
            down_fast_kernel<Q8Source, MoeBinding, RouteBF16, MT, NW><<<grid, L::THREADS, L::total, s>>>(w.down, v);
        } else throw std::invalid_argument("MoE down type");
        check(cudaGetLastError(), "MoE down moe_mma");
        return;
    }
    const dim3 grid(csr_sizes(rows).max_tiles, hidden / kDownOuts);
    if (w.down_type == type_q5_1) {
        static bool once = (set_smem(down_kernel<Q51Source, MoeBinding, RouteBF16>, DownLayout<Q51Source>::total), true); (void)once;
        down_kernel<Q51Source, MoeBinding, RouteBF16><<<grid, kDownThreads, DownLayout<Q51Source>::total, s>>>(w.down, v);
    } else if (w.down_type == type_q8_0) {
        static bool once = (set_smem(down_kernel<Q8Source, MoeBinding, RouteBF16>, DownLayout<Q8Source>::total), true); (void)once;
        down_kernel<Q8Source, MoeBinding, RouteBF16><<<grid, kDownThreads, DownLayout<Q8Source>::total, s>>>(w.down, v);
    } else throw std::invalid_argument("MoE down type");
    check(cudaGetLastError(), "MoE down");
}
namespace {
using namespace ::strix::cuda::core::moe_a8;
void validate_route_storage(const Rows& v, std::uint32_t rows) {
    if (v.route_bf16) qwen_route::validate_view(reinterpret_cast<std::uintptr_t>(v.rawdots.base),
        v.rawdots.stride, v.rawdot_capacity, rows, true);
}
} // namespace

void enqueue_down(const Rows& v, std::uint32_t rows, const Weights& w, cudaStream_t s, bool moe_mma) {
    validate_route_storage(v, rows);
    if (v.route_bf16 && (native_types(w) || w.arithmetic==ExpertArithmetic::gsq_a8_dot4))
        qwen_route::require_composition(true, qwen_route::Composition::other_weight_pack);
    if (v.route_bf16) enqueue_down_storage<true>(v, rows, w, s, moe_mma);
    else enqueue_down_storage<false>(v, rows, w, s, moe_mma);
}
void enqueue_merge(const Rows& v, std::uint32_t rows, cudaStream_t s) {
    validate_route_storage(v, rows);
    if (v.route_bf16) merge_kernel<true><<<dim3((hidden + 255U) / 256U, rows), 256, 0, s>>>(v);
    else merge_kernel<false><<<dim3((hidden + 255U) / 256U, rows), 256, 0, s>>>(v);
    check(cudaGetLastError(), "MoE merge");
}
void enqueue_commit(const Rows& v, std::uint32_t rows, std::uint32_t* record, cudaStream_t s) {
    commit_kernel<<<dim3((hidden + 255U) / 256U, rows), 256, 0, s>>>(v, rows, record);
    check(cudaGetLastError(), "MoE commit");
}
bool gsq_types(const Weights& w) {
    const bool gate=w.gate_type==18 || w.gate_type==21 || w.gate_type==22 || w.gate_type==23 || w.gate_type==42;
    const bool down=w.down_type==20 || w.down_type==42;
    return gate && down;
}
bool native_types(const Weights& w) {
    if(w.arithmetic==ExpertArithmetic::gsq_a8_dot4 && gsq_types(w)) return false;

    const bool a8_gate = w.gate_type == type_q4_k || w.gate_type == type_q5_k;
    const bool a8_down = w.down_type == type_q5_1 || w.down_type == type_q8_0;
    return !(a8_gate && a8_down);
}

bool enqueue_routed(const Rows& v, std::uint32_t rows, const Weights& w, const amdmath::Tables& tables, cudaStream_t s,
                    bool moe_mma) {
    validate_expert_arithmetic(w,moe_mma);
    validate_route_storage(v, rows);
    if (v.route_bf16 && (native_types(w) || w.arithmetic==ExpertArithmetic::gsq_a8_dot4))
        qwen_route::require_composition(true, qwen_route::Composition::other_weight_pack);
    enqueue_csr(v, rows, s);
    if (native_types(w)) {
        if (!v.intermediates.base) throw std::invalid_argument("MoE native expert types need Rows::intermediates");
        if (quant::geometry(w.gate_type).block_elements == 0 || quant::geometry(w.down_type).block_elements == 0)
            throw std::invalid_argument("MoE expert type without a decoder");
        const unsigned tiles = csr_sizes(rows).max_tiles;
        native_project_kernel<NativeSource,MoeBinding><<<dim3(tiles, ffn / 32U), 1024, 0, s>>>(w.gate, w.up, w.gate_type, v);
        check(cudaGetLastError(), "MoE native gate/up");
        native_down_kernel<NativeSource,MoeBinding><<<dim3(tiles, hidden / 32U), 1024, 0, s>>>(w.down, w.down_type, v);
        check(cudaGetLastError(), "MoE native down");
        return false;   // native experts: candidate MMA arithmetic did not run
    }
    enqueue_project(v, rows, w, tables, s, moe_mma);
    enqueue_down(v, rows, w, s, moe_mma);
    // enqueue_project / enqueue_down_storage run the rows kernels for at most kRowsMax rows with the decode kernels on: no MMA ran.
    return moe_mma && !(rows <= kRowsMax && decode_rows_enabled());
}
void enqueue_combine(const Rows& v, std::uint32_t rows, std::uint32_t* record, cudaStream_t s) {
    enqueue_merge(v, rows, s);
    enqueue_commit(v, rows, record, s);
}

bool enqueue(const Rows& v, std::uint32_t rows, std::uint32_t clear_words, const Weights& w, const amdmath::Tables& tables,
             std::uint32_t* record, cudaStream_t s, bool moe_mma) {
    validate_expert_arithmetic(w,moe_mma);
    validate_route_storage(v, rows);
    if (v.route_bf16 && (native_types(w) || w.arithmetic==ExpertArithmetic::gsq_a8_dot4))
        qwen_route::require_composition(true, qwen_route::Composition::other_weight_pack);
    enqueue_prologue(v, rows, clear_words, s);
    const bool dispatched = enqueue_routed(v, rows, w, tables, s, moe_mma);
    enqueue_combine(v, rows, record, s);
    return dispatched;
}

}  // namespace strix::cuda::qwen_moe
