#include "strix/env_flags.hpp"
#include "strix/ds41_prefill_tuning.hpp"
#include "strix/hip_check.hpp"
#pragma clang fp contract(off)
#include "strix/hip_ds41_prefill.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "moe_core/grouped_integer_decode.hpp"
#include "ds41_integer_operand.hpp"  // shared tiled core route (STRIX_DS41_PF_CORE=1, FAST_I4)

#include "strix/ds41_moe.hpp"
#include "strix/ds41_prefill.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace strix::ds41 {
namespace {

void check(hipError_t status, const char* what) {
    ::strix::hip_check_prefixed("ds41 prefill: ", status, what);
}

// ---- Format helpers (the decode kernels' integer forms, hip_ds41_formats.hip) ----

__device__ const unsigned long long prefill_iq2_grid[256] = {
#include "strix/ds41_iq2_xxs_grid.inc"
};
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
constexpr int dot_magic = 0x4b400000; // 1.5 * 2^23: bits + r are the float 1.5 * 2^23 + r for |r| < 2^22

template <WeightFormat Format> struct Raw;
template <> struct Raw<WeightFormat::iq2_xxs> { unsigned short d; unsigned long long aux; };
template <> struct Raw<WeightFormat::q2_k> { unsigned q[8]; unsigned short scales; unsigned dm; unsigned shift; };
template <> struct Raw<WeightFormat::mxfp4> { unsigned e; unsigned q[4]; };
template <> struct Raw<WeightFormat::q8_0> { unsigned short d; unsigned q[8]; };

struct Operands { v4i b0, b1; float scale, d, dmin; int m0, m1; unsigned dm_bits, sc_bits; };

// DS41 keeps its integer operand representation and launch policy at the adapter.
struct Ds41PrefillTraits {
    using WeightFormat = ::strix::ds41::WeightFormat;
    using Int4 = v4i;
    using Int8 = v8i;
    using OperandGroup = Operands;
    template <WeightFormat Format> using RawGroup = Raw<Format>;
    static constexpr int magic_bits = dot_magic;
    static constexpr auto iq2_xxs_block_bytes = ::strix::ds41::iq2_xxs_block_bytes;
    static constexpr auto q2_k_block_bytes = ::strix::ds41::q2_k_block_bytes;
    static constexpr auto mxfp4_block_bytes = ::strix::ds41::mxfp4_block_bytes;
    static constexpr auto q8_0_block_bytes = ::strix::ds41::q8_0_block_bytes;
    using Operations = strix::moe_core::GroupedIntegerDecodeOps<Ds41PrefillTraits>;
    template <WeightFormat Format> using WeightDecode = std::conditional_t<
        Format == WeightFormat::q2_k, strix::moe_core::GroupedQ2KWeightDecode<Ds41PrefillTraits>,
        std::conditional_t<Format == WeightFormat::iq2_xxs,
            strix::moe_core::GroupedIq2XxsWeightDecode<Ds41PrefillTraits>,
            std::conditional_t<Format == WeightFormat::q8_0,
                strix::moe_core::GroupedWeightDecode<strix::moe_core::GroupedWeightFormat::q8_0, Operations>,
                Operations>>>;
};
template <WeightFormat Format>
using Ds41PrefillCore = strix::moe_core::GroupedPrefillCore<Ds41PrefillTraits::WeightDecode<Format>>;
using Ds41Decode = Ds41PrefillTraits::Operations;
using SignEntry = Ds41Decode::SignEntry;

__device__ __forceinline__ float half_bits(unsigned short bits) { return Ds41Decode::half_bits(bits); }
__device__ __forceinline__ unsigned short load_u16(const unsigned char* p) { return Ds41Decode::load_u16(p); }
__device__ __forceinline__ unsigned byte_masks(unsigned bits4) { return Ds41Decode::byte_masks(bits4); }
__device__ __forceinline__ unsigned sign_byte(unsigned index) { return Ds41Decode::sign_byte(index); }
__device__ __forceinline__ int apply_signs(unsigned magnitudes, unsigned mask) { return Ds41Decode::apply_signs(magnitudes, mask); }
__device__ __forceinline__ float magic_float(int bits) { return Ds41Decode::magic_float(bits); }
__device__ __forceinline__ v8i group_dot(v4i a0, v4i b0, v4i a1, v4i b1) { return Ds41Decode::group_dot(a0, b0, a1, b1); }
template <WeightFormat Format>
__device__ __forceinline__ Raw<Format> load_raw(const unsigned char* row, unsigned g) { return Ds41PrefillCore<Format>::WeightDecode::template load_raw<Format>(row, g); }
__device__ __forceinline__ Operands decode(const Raw<WeightFormat::iq2_xxs>& raw, const unsigned long long* grid) { return Ds41PrefillCore<WeightFormat::iq2_xxs>::WeightDecode::decode(raw, grid); }
__device__ __forceinline__ Operands decode(const Raw<WeightFormat::mxfp4>& raw, const unsigned long long* grid) { return Ds41PrefillCore<WeightFormat::mxfp4>::WeightDecode::decode(raw, grid); }
__device__ __forceinline__ Operands decode(const Raw<WeightFormat::q8_0>& raw, const unsigned long long* grid) { return Ds41PrefillCore<WeightFormat::q8_0>::WeightDecode::decode(raw, grid); }
__device__ __forceinline__ Operands decode(const Raw<WeightFormat::q2_k>& raw, const unsigned long long* grid) { return Ds41PrefillCore<WeightFormat::q2_k>::WeightDecode::decode(raw, grid); }
__device__ __forceinline__ unsigned byte_masks_lean(unsigned bits4) { return Ds41Decode::byte_masks_lean(bits4); }
__device__ __forceinline__ Operands decode_lean(const Raw<WeightFormat::iq2_xxs>& raw, const unsigned long long* grid) { return Ds41PrefillCore<WeightFormat::iq2_xxs>::WeightDecode::decode_lean(raw, grid); }
__device__ __forceinline__ Operands decode_lean(const Raw<WeightFormat::q2_k>& raw, const unsigned long long* grid) { return Ds41PrefillCore<WeightFormat::q2_k>::WeightDecode::decode_lean(raw, grid); }
__device__ __forceinline__ void load_signs(SignEntry* signs) { return Ds41Decode::load_signs(signs); }
__device__ __forceinline__ Operands decode_signs(const Raw<WeightFormat::iq2_xxs>& raw, const unsigned long long* grid,
                                                 const SignEntry* signs) { return Ds41PrefillCore<WeightFormat::iq2_xxs>::WeightDecode::decode_signs(raw, grid, signs); }
template <int Form, WeightFormat Format>
__device__ __forceinline__ Operands decode_form(const Raw<Format>& raw, const unsigned long long* grid, const SignEntry* signs) { return Ds41PrefillCore<Format>::WeightDecode::template decode_form<Form, Format>(raw, grid, signs); }

#include "moe_core/hip_ds41_prefill_grouped.inc"

#include "hip_ds41_prefill_relaxed.inl"  // FAST lever ds41-moe-relaxed-v1 (not exact)

#include "moe_core/hip_ds41_prefill_variants.inc"

// ---- Router of T rows ----
//
// The decode router's declared order per (row, expert): lane l accumulates acc = acc + w * x over the
// elements l, l + 32, ... in ascending order from 0, the lanes meet in the butterfly (lane 0's value),
// then router_score. Here a block of 4 waves covers 16 rows x 16 experts (a wave 8 x 8); x and the
// weights are staged in LDS 256 elements at a time and every lane keeps the 64 chains of its tile, so
// each (row, expert) is formed by exactly the decode chain and tree.
constexpr unsigned prouter_rows = 16U, prouter_experts = 16U, prouter_chunk = 256U, prouter_threads = 128U, prouter_group = 4U;
__global__ __launch_bounds__(prouter_threads) void prefill_router_scores_kernel(RouterView view) {
    __shared__ alignas(16) float xs[prouter_rows][prouter_chunk];
    __shared__ alignas(16) float ws[prouter_experts][prouter_chunk];
    __shared__ float logits[prouter_rows][prouter_experts];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    const unsigned row0 = blockIdx.x * prouter_rows, expert0 = blockIdx.y * prouter_experts;
    const unsigned wr = (wave & 1U) * 8U, we = (wave >> 1U) * 8U;
    float acc[8][8];
#pragma unroll
    for (unsigned r = 0; r < 8U; ++r)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) acc[r][e] = 0.0F;
    for (unsigned c0 = 0; c0 < view.hidden; c0 += prouter_chunk) {
        const unsigned n = min(prouter_chunk, view.hidden - c0);
        if (c0) __syncthreads();
        if (view.wide_stage) {
            // the same values staged with 16-byte loads, prouter_group of them in flight per array before the LDS stores
            // (the element loop above waited for every load in turn: ~0.8 ms per call at any row count)
            constexpr unsigned quads = prouter_chunk / 4U, per_thread = prouter_rows * quads / prouter_threads;
            for (unsigned t0 = 0; t0 < per_thread; t0 += prouter_group) {
                float4 xv4[prouter_group], wv4[prouter_group];
#pragma unroll
                for (unsigned t = 0; t < prouter_group; ++t) {
                    const unsigned i = threadIdx.x + (t0 + t) * prouter_threads, r = i / quads, k = (i % quads) * 4U;
                    const unsigned row = row0 + r, expert = expert0 + r;
                    xv4[t] = row < view.rows && k < n ? *reinterpret_cast<const float4*>(view.x + static_cast<unsigned long long>(row) * view.hidden + c0 + k)
                                                      : float4{0.0F, 0.0F, 0.0F, 0.0F};
                    wv4[t] = expert < view.experts && k < n
                                 ? *reinterpret_cast<const float4*>(view.weight + static_cast<unsigned long long>(expert) * view.hidden + c0 + k)
                                 : float4{0.0F, 0.0F, 0.0F, 0.0F};
                }
#pragma unroll
                for (unsigned t = 0; t < prouter_group; ++t) {
                    const unsigned i = threadIdx.x + (t0 + t) * prouter_threads, r = i / quads, k = (i % quads) * 4U;
                    *reinterpret_cast<float4*>(&xs[r][k]) = xv4[t];
                    *reinterpret_cast<float4*>(&ws[r][k]) = wv4[t];
                }
            }
        } else {
            for (unsigned i = threadIdx.x; i < prouter_rows * prouter_chunk; i += prouter_threads) {
                const unsigned r = i / prouter_chunk, k = i % prouter_chunk;
                const unsigned row = row0 + r, expert = expert0 + r;
                xs[r][k] = row < view.rows && k < n ? view.x[static_cast<unsigned long long>(row) * view.hidden + c0 + k] : 0.0F;
                ws[r][k] = expert < view.experts && k < n ? view.weight[static_cast<unsigned long long>(expert) * view.hidden + c0 + k] : 0.0F;
            }
        }
        __syncthreads();
        for (unsigned j = 0; j < prouter_chunk / 32U; ++j) {
            const unsigned k = lane + 32U * j;
            if (k >= n) break; // n is a multiple of 32 whenever hidden is; the chain skips nothing
            float xv[8], wv[8];
#pragma unroll
            for (unsigned r = 0; r < 8U; ++r) xv[r] = xs[wr + r][k];
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) wv[e] = ws[we + e][k];
#pragma unroll
            for (unsigned r = 0; r < 8U; ++r)
#pragma unroll
                for (unsigned e = 0; e < 8U; ++e) acc[r][e] = acc[r][e] + wv[e] * xv[r];
        }
    }
#pragma unroll
    for (unsigned r = 0; r < 8U; ++r)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) {
            float v = acc[r][e];
#pragma unroll
            for (unsigned d = 16U; d; d /= 2U) v = v + __shfl_xor(v, static_cast<int>(d), 32);
            if (lane == 0U) logits[wr + r][we + e] = v;
        }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < prouter_rows * prouter_experts; i += prouter_threads) {
        const unsigned r = i / prouter_experts, e = i % prouter_experts, row = row0 + r, expert = expert0 + e;
        if (row < view.rows && expert < view.experts)
            view.scores[static_cast<unsigned long long>(row) * view.experts + expert] = router_score(logits[r][e]);
    }
}
// v2 of the same scores (STRIX_DS41_PREFILL_ROUTER_V2, default on): a block of 8 waves covers 32 rows x 32 experts (a
// wave 8 rows x 16 experts), x and the weights are staged 128 elements at a time and the next chunk's 16-byte loads are
// issued before the current chunk's products, so the loads overlap the FMAs instead of waiting behind two barriers.
// Lane l still owns the elements l, l + 32, ... of every (row, expert) of its tile, in ascending order from 0, and the
// lanes meet in the same butterfly: each score is formed by exactly the decode chain and tree, so every bit is equal.
// Needs hidden % 4 == 0 and 16-byte aligned x and weight (otherwise the kernel above runs).
constexpr unsigned prouter2_rows = 32U, prouter2_experts = 32U, prouter2_chunk = 128U, prouter2_threads = 256U;
constexpr unsigned prouter2_quads = prouter2_rows * prouter2_chunk / 4U / prouter2_threads; // 16-byte loads per thread per array
__global__ __launch_bounds__(prouter2_threads) void prefill_router_scores_v2_kernel(RouterView view) {
    __shared__ alignas(16) float xs[prouter2_rows][prouter2_chunk];
    __shared__ alignas(16) float ws[prouter2_experts][prouter2_chunk];
    __shared__ float logits[prouter2_rows][prouter2_experts];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    const unsigned row0 = blockIdx.y * prouter2_rows, expert0 = blockIdx.x * prouter2_experts;
    const unsigned wr = (wave & 3U) * 8U, we = (wave >> 2U) * 16U;
    float acc[8][16];
#pragma unroll
    for (unsigned r = 0; r < 8U; ++r)
#pragma unroll
        for (unsigned e = 0; e < 16U; ++e) acc[r][e] = 0.0F;
    float4 xv4[prouter2_quads], wv4[prouter2_quads];
    const auto load = [&](unsigned c0) {
        const unsigned n = (view.hidden - c0 < prouter2_chunk ? view.hidden - c0 : prouter2_chunk);
#pragma unroll
        for (unsigned t = 0; t < prouter2_quads; ++t) {
            const unsigned i = threadIdx.x + t * prouter2_threads, r = i / (prouter2_chunk / 4U), k = (i % (prouter2_chunk / 4U)) * 4U;
            const unsigned row = row0 + r, expert = expert0 + r;
            xv4[t] = row < view.rows && k < n ? *reinterpret_cast<const float4*>(view.x + static_cast<unsigned long long>(row) * view.hidden + c0 + k)
                                              : float4{0.0F, 0.0F, 0.0F, 0.0F};
            wv4[t] = expert < view.experts && k < n
                         ? *reinterpret_cast<const float4*>(view.weight + static_cast<unsigned long long>(expert) * view.hidden + c0 + k)
                         : float4{0.0F, 0.0F, 0.0F, 0.0F};
        }
    };
    load(0U);
    for (unsigned c0 = 0; c0 < view.hidden; c0 += prouter2_chunk) {
        const unsigned n = (view.hidden - c0 < prouter2_chunk ? view.hidden - c0 : prouter2_chunk);
        if (c0) __syncthreads();
#pragma unroll
        for (unsigned t = 0; t < prouter2_quads; ++t) {
            const unsigned i = threadIdx.x + t * prouter2_threads, r = i / (prouter2_chunk / 4U), k = (i % (prouter2_chunk / 4U)) * 4U;
            *reinterpret_cast<float4*>(&xs[r][k]) = xv4[t];
            *reinterpret_cast<float4*>(&ws[r][k]) = wv4[t];
        }
        __syncthreads();
        if (c0 + prouter2_chunk < view.hidden) load(c0 + prouter2_chunk);
        for (unsigned j = 0; j < prouter2_chunk / 32U; ++j) {
            const unsigned k = lane + 32U * j;
            if (k >= n) break; // n is a multiple of 32 (hidden is); the chain skips nothing
            float xv[8], wv[16];
#pragma unroll
            for (unsigned r = 0; r < 8U; ++r) xv[r] = xs[wr + r][k];
#pragma unroll
            for (unsigned e = 0; e < 16U; ++e) wv[e] = ws[we + e][k];
#pragma unroll
            for (unsigned r = 0; r < 8U; ++r)
#pragma unroll
                for (unsigned e = 0; e < 16U; ++e) acc[r][e] = acc[r][e] + wv[e] * xv[r];
        }
    }
#pragma unroll
    for (unsigned r = 0; r < 8U; ++r)
#pragma unroll
        for (unsigned e = 0; e < 16U; ++e) {
            float v = acc[r][e];
#pragma unroll
            for (unsigned d = 16U; d; d /= 2U) v = v + __shfl_xor(v, static_cast<int>(d), 32);
            if (lane == 0U) logits[wr + r][we + e] = v;
        }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < prouter2_rows * prouter2_experts; i += prouter2_threads) {
        const unsigned r = i / prouter2_experts, e = i % prouter2_experts, row = row0 + r, expert = expert0 + e;
        if (row < view.rows && expert < view.experts)
            view.scores[static_cast<unsigned long long>(row) * view.experts + expert] = router_score(logits[r][e]);
    }
}

// Top-k of one row per block, as the decode router's selection: repeated arg-max of score + bias
// (larger value, then lower id), weights from the unbiased scores in selection order.
__global__ __launch_bounds__(256) void prefill_router_select_kernel(RouterView view) {
    __shared__ float biased[1024];
    __shared__ float best_value[256];
    __shared__ unsigned best_index[256];
    __shared__ unsigned chosen[16];
    const unsigned r = blockIdx.x, t = threadIdx.x;
    const float* scores = view.scores + static_cast<unsigned long long>(r) * view.experts;
    const float* bias = view.image_rows && view.image_rows[r] ? view.bias_image : view.bias;
    for (unsigned e = t; e < view.experts; e += 256U) biased[e] = scores[e] + bias[e];
    __syncthreads();
    for (unsigned k = 0; k < view.top_k; ++k) {
        float v = 0.0F;
        unsigned index = view.experts;
        for (unsigned e = t; e < view.experts; e += 256U) {
            bool taken = false;
            for (unsigned j = 0; j < k; ++j) taken = taken || chosen[j] == e;
            if (taken) continue;
            if (index == view.experts || biased[e] > v) { v = biased[e]; index = e; }
        }
        best_value[t] = v; best_index[t] = index;
        __syncthreads();
        for (unsigned stride = 128U; stride; stride /= 2U) {
            if (t < stride) {
                const float ov = best_value[t + stride];
                const unsigned oi = best_index[t + stride];
                const bool better = oi != view.experts &&
                    (best_index[t] == view.experts || ov > best_value[t] || (ov == best_value[t] && oi < best_index[t]));
                if (better) { best_value[t] = ov; best_index[t] = oi; }
            }
            __syncthreads();
        }
        if (t == 0U) chosen[k] = best_index[0];
        __syncthreads();
    }
    if (t == 0U) {
        float sum = 0.0F;
        for (unsigned k = 0; k < view.top_k; ++k) sum = sum + scores[chosen[k]];
        const float denominator = sum + 1e-20F;
        for (unsigned k = 0; k < view.top_k; ++k) {
            const float s = scores[chosen[k]];
            view.ids[r * view.top_k + k] = chosen[k];
            view.weights[r * view.top_k + k] = (view.top_k > 1U ? s / denominator : s) * view.route_scale;
        }
    }
}

// ---- Routes on the device ----

// Experts [first, end) only (a rank's slice); expert indices below are relative to first.
__global__ __launch_bounds__(256) void route_count_kernel(const unsigned* ids, unsigned slots, unsigned first, unsigned end,
                                                          unsigned* counts) {
    const unsigned s = blockIdx.x * 256U + threadIdx.x;
    if (s >= slots) return;
    const unsigned id = ids[s];
    if (id >= first && id < end) atomicAdd(counts + (id - first), 1U);
}
// One block, histogram split (PrefillRoutesBuffers::sparse_max > 0): sparse experts (1..sparse_max entries) first, in
// ascending expert, with batches of 8 in the sparse arrays; then the dense experts with batches of batch_rows.
__global__ __launch_bounds__(1024) void route_scan_split_kernel(const unsigned* counts, unsigned experts, unsigned batch_rows,
                                                                PrefillRoutesBuffers out) {
    __shared__ unsigned scan[4][1024];
    const unsigned e = threadIdx.x;
    const unsigned c = e < experts ? counts[e] : 0U;
    const bool sparse = c > 0U && c <= out.sparse_max;
    const unsigned sc = sparse ? c : 0U, dc = sparse ? 0U : c, snb = (sc + 7U) / 8U, dnb = (dc + batch_rows - 1U) / batch_rows;
    scan[0][e] = sc; scan[1][e] = dc; scan[2][e] = snb; scan[3][e] = dnb;
    __syncthreads();
    for (unsigned stride = 1U; stride < 1024U; stride *= 2U) {
        unsigned add[4];
        for (unsigned a = 0; a < 4U; ++a) add[a] = e >= stride ? scan[a][e - stride] : 0U;
        __syncthreads();
        for (unsigned a = 0; a < 4U; ++a) scan[a][e] += add[a];
        __syncthreads();
    }
    const unsigned sparse_entries = scan[0][1023], dense_entries = scan[1][1023];
    if (e < experts) {
        if (sparse) {
            const unsigned begin = scan[0][e] - sc, first_batch = scan[2][e] - snb;
            out.begins[e] = begin;
            for (unsigned b = 0; b < snb; ++b) {
                out.sparse_expert[first_batch + b] = e;
                out.sparse_begin[first_batch + b] = begin + b * 8U;
            }
        } else {
            const unsigned begin = sparse_entries + scan[1][e] - dc, first_batch = scan[3][e] - dnb;
            out.begins[e] = begin;
            for (unsigned b = 0; b < dnb; ++b) {
                out.batch_expert[first_batch + b] = e;
                out.batch_begin[first_batch + b] = begin + b * batch_rows;
                out.batch_count[first_batch + b] = min(batch_rows, dc - b * batch_rows);
            }
        }
    }
    if (e == 0U) {
        out.begins[experts] = sparse_entries + dense_entries;
        *out.sparse_total = scan[2][1023];
        out.sparse_begin[scan[2][1023]] = sparse_entries;
        *out.batch_total = scan[3][1023];
        out.batch_begin[scan[3][1023]] = sparse_entries + dense_entries;
    }
}
// One block: begins (exclusive scan of the counts) and the batch descriptors in ascending expert.
__global__ __launch_bounds__(1024) void route_scan_kernel(const unsigned* counts, unsigned experts, unsigned batch_rows,
                                                          PrefillRoutesBuffers out) {
    __shared__ unsigned entries[1024], batches[1024];
    const unsigned e = threadIdx.x;
    const unsigned c = e < experts ? counts[e] : 0U, nb = (c + batch_rows - 1U) / batch_rows;
    entries[e] = c; batches[e] = nb;
    __syncthreads();
    for (unsigned stride = 1U; stride < 1024U; stride *= 2U) {
        const unsigned add_e = e >= stride ? entries[e - stride] : 0U, add_b = e >= stride ? batches[e - stride] : 0U;
        __syncthreads();
        entries[e] += add_e; batches[e] += add_b;
        __syncthreads();
    }
    if (e < experts) {
        const unsigned begin = entries[e] - c, first_batch = batches[e] - nb;
        out.begins[e] = begin;
        for (unsigned b = 0; b < nb; ++b) {
            out.batch_expert[first_batch + b] = e;
            out.batch_begin[first_batch + b] = begin + b * batch_rows;
            out.batch_count[first_batch + b] = min(batch_rows, c - b * batch_rows);
        }
    }
    if (e == 0U) {
        out.begins[experts] = entries[1023];
        *out.batch_total = batches[1023];
        out.batch_begin[batches[1023]] = entries[1023]; // the batch_begin array has max_batches + 1 entries
    }
}
// One block per expert: its slots in ascending order (ordered compaction over the id array).
__global__ __launch_bounds__(256) void route_fill_kernel(const unsigned* ids, unsigned slots, unsigned top_k, unsigned first,
                                                         PrefillRoutesBuffers out) {
    __shared__ unsigned wave_counts[8];
    const unsigned local = blockIdx.x, expert = first + local, wave = threadIdx.x / 32U, lane = threadIdx.x % 32U;
    unsigned at = out.begins[local];
    for (unsigned base = 0; base < slots; base += 256U) {
        const unsigned s = base + threadIdx.x;
        const bool match = s < slots && ids[s] == expert;
        const unsigned mask = static_cast<unsigned>(__ballot(match));
        if (lane == 0U) wave_counts[wave] = static_cast<unsigned>(__builtin_popcount(mask));
        __syncthreads();
        unsigned before = 0U, total = 0U;
        for (unsigned w = 0; w < 8U; ++w) { before += w < wave ? wave_counts[w] : 0U; total += wave_counts[w]; }
        if (match) {
            const unsigned index = at + before + static_cast<unsigned>(__builtin_popcount(mask & ((1U << lane) - 1U)));
            out.entry_rows[index] = s / top_k;
            out.entry_slots[index] = s;
        }
        at += total;
        __syncthreads();
    }
}
// Per row: slots in ascending expert id and how many ids are below split.
__global__ __launch_bounds__(256) void route_order_kernel(const unsigned* ids, unsigned rows, unsigned top_k, unsigned split,
                                                          PrefillRoutesBuffers out) {
    const unsigned row = blockIdx.x * 256U + threadIdx.x;
    if (row >= rows) return;
    unsigned expert[16], slot[16], low = 0U;
    for (unsigned k = 0; k < top_k; ++k) {
        const unsigned s = row * top_k + k, id = ids[s];
        unsigned at = k;
        while (at > 0U && expert[at - 1U] > id) { expert[at] = expert[at - 1U]; slot[at] = slot[at - 1U]; --at; }
        expert[at] = id; slot[at] = s;
        low += id < split ? 1U : 0U;
    }
    for (unsigned k = 0; k < top_k; ++k) out.order[row * top_k + k] = slot[k];
    out.low_counts[row] = low;
}

StageLayout stage_layout(unsigned width, unsigned batch_rows) {
    const unsigned groups = width / 32U, j = (groups + 31U) / 32U;
    return {batch_rows, j, stage_classes * j};
}
// STRIX_DS41_PREFILL_STAGE=1 keeps 16-row batches on the stage kernels (comparison only).
bool stage_forced() {
    static const bool forced = [] {
        const char* v = std::getenv("STRIX_DS41_PREFILL_STAGE");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return forced;
}
// Prefill kernel forms (moe_kernel_tuning().prefill, STRIX_DS41_MOE_PREFILL): 0 = the original decoders; 1 = the lean
// decoders (same integers, see decode_lean); 2 = lean with the IQ2_XXS sign table (decode_signs); 3 = the lean decoders
// with the activations double-buffered; 4 = the lean decoders with the merged stage (merged_stage); 6 = 1 with LDS-only
// chunk-loop barriers; 7 (default, -1) = 6 with a batched activation stage in gate/up (down runs form 6). Measured: the
// joint MoE layer -5.7 / -3.2 / -5.1 % at T 600 / 2048 / 8192, gate/up -8.8 %; same bits. =0 keeps the original form.
int prefill_form() {
    const int f = moe_kernel_tuning().prefill;
    return f < 0 ? 7 : f;
}
// STRIX_DS41_PREFILL_MOE_V2 (read per call; default on, =0 off): the split kernels v2 (split_gate_up_v2_kernel,
// split_down_v2_kernel) for IQ2_XXS gate/up and Q2_K down under the default form 7; the same bits as forms 7 / 6.
// STRIX_DS41_PREFILL_MOE_V2_MASK (expert override, attribution): a mask of v2_decode (2), v2_fma (4), v2_part (8), default
// v2_fma | v2_part (v2_decode measured slower than the lean decoder: opt-in only; v2_part is always added). STRIX_DS41_PREFILL_MOE_BAND: blocks in bands of N batches (0 or 1: batch-fastest order as before), default 4.
unsigned prefill_v2() {
    if (prefill_form() != 7) return 0U;
    const char* v = std::getenv("STRIX_DS41_PREFILL_MOE_V2");
    if (v && std::strcmp(v, "0") == 0) return 0U;
    const char* m = std::getenv("STRIX_DS41_PREFILL_MOE_V2_MASK");
    unsigned mask = v2_on | (m && *m ? static_cast<unsigned>(std::strtoul(m, nullptr, 0)) & (v2_decode | v2_fma | v2_part | v2_pin | v2_signs | v2_signs16 | v2_pipe) : v2_fma | v2_part);
    // STRIX_DS41_PREFILL_MOE_V2_SIGNS (=1 / =16, read per call; opt-in): v2_signs / v2_signs16, the IQ2_XXS gate/up sign
    // masks from an LDS table (8- / 16-byte entries). Same bits; measured slower (8K gate/up +7 % / +13 %): -19 % VALU but
    // 5x the LDS wait cycles (random-index table reads).
    if (!(m && *m)) {
        const char* sg = std::getenv("STRIX_DS41_PREFILL_MOE_V2_SIGNS");
        if (sg && std::strcmp(sg, "1") == 0) mask |= v2_signs;
        if (sg && std::strcmp(sg, "16") == 0) mask |= v2_signs16;
        // STRIX_DS41_PREFILL_MOE_V2_PIPE=1 (opt-in): v2_pipe, the IQ2_XXS decode's LDS reads one group ahead (same bits;
        // measured +10 % at 8K)
        const char* pp = std::getenv("STRIX_DS41_PREFILL_MOE_V2_PIPE");
        if (pp && std::strcmp(pp, "1") == 0) mask |= v2_pipe;
    }
    // STRIX_DS41_PREFILL_MOE_V2_PIN (=1 on, read per call; opt-in): v2_pin, the WMMAs on bank-pinned registers. Measured
    // (rank bench, same bits): routed gate/up and down neutral (+-1 %), the Q8_0 shared expert 3-14 % slower.
    if (!(m && *m)) {
        const char* pin = std::getenv("STRIX_DS41_PREFILL_MOE_V2_PIN");
        if (pin && std::strcmp(pin, "1") == 0) mask |= v2_pin;
    }
    // Only implemented combinations: the sign-table and pipelined decoders exist in the pinned form (they imply v2_pin),
    // and a partial last chunk always loads only its blocks (v2_part: the full-chunk load would read past the Q2_K
    // tensor's last row at width 2304).
    if (mask & (v2_signs | v2_signs16 | v2_pipe)) mask |= v2_pin;
    return mask | v2_part;
}
// STRIX_DS41_PREFILL_MOE_V2_Q8 (read per call; default on, =0 off; needs v2): the Q8_0 split kernels (the shared expert's
// gate/up and down) on split_gate_up_v2_kernel / split_down_v2_kernel with direct per-lane weight reads; the same bits.
bool prefill_v2_q8() {
    const char* v = std::getenv("STRIX_DS41_PREFILL_MOE_V2_Q8");
    return !(v && std::strcmp(v, "0") == 0);
}
unsigned prefill_band() {
    const char* v = std::getenv("STRIX_DS41_PREFILL_MOE_BAND");
    return v && *v ? static_cast<unsigned>(std::strtoul(v, nullptr, 10)) : 4U;
}
// STRIX_DS41_PREFILL_MOE_R2=1 (opt-in, read per call; needs v2): 32-row batches, IQ2_XXS gate/up on r2_gate_up_kernel and
// Q2_K down on split_down_v2_kernel in 16-row halves; the same bits.
bool prefill_r2() {
    const char* v = std::getenv("STRIX_DS41_PREFILL_MOE_R2");
    return v && std::strcmp(v, "1") == 0 && (prefill_v2() & v2_on) != 0U && !stage_forced();
}
template <WeightFormat Format, unsigned L>
void launch_split_gate_up_v2(const PrefillGateUpView& view, hipStream_t stream) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(split_gate_up_v2_kernel<Format, L>), dim3(view.routes.max_batches, view.inter / 32U), dim3(split_threads),
                       (v2_lds<Format, L>), stream, view, prefill_band());
}
template <WeightFormat Format, unsigned L>
void launch_split_down_v2(const PrefillDownView& view, hipStream_t stream) {
    const unsigned halves = view.routes.batch_rows == 32U ? 2U : 1U; // 32-row batches (R2) run as two 16-row blocks
    hipLaunchKernelGGL(HIP_KERNEL_NAME(split_down_v2_kernel<Format, L>), dim3(halves * view.routes.max_batches, view.outputs / 64U), dim3(split_threads),
                       (v2_lds<Format, L>), stream, view, prefill_band());
}
template <WeightFormat Format, int Form>
void launch_split_gate_up_form(const PrefillGateUpView& view, hipStream_t stream) {
    constexpr unsigned lds = split_lds_for<Format, stage_buffers_of<Form>>();
    hipLaunchKernelGGL(HIP_KERNEL_NAME(split_gate_up_kernel<Format, Form>), dim3(view.routes.max_batches, view.inter / 32U), dim3(split_threads), lds, stream, view);
}
template <WeightFormat Format>
void launch_split_gate_up(const PrefillGateUpView& view, hipStream_t stream) {
    if constexpr (Format == WeightFormat::iq2_xxs) {
        if (const unsigned m = prefill_v2(); (m & v2_on) && view.hidden % (split_chunk_groups * 32U) == 0U) {
            if (m & v2_decode) launch_split_gate_up_v2<Format, v2_on | v2_decode>(view, stream);
            else if ((m & (v2_pin | v2_pipe)) == (v2_pin | v2_pipe)) launch_split_gate_up_v2<Format, v2_on | v2_pin | v2_pipe>(view, stream);
            else if ((m & (v2_pin | v2_signs16)) == (v2_pin | v2_signs16)) launch_split_gate_up_v2<Format, v2_on | v2_pin | v2_signs16>(view, stream);
            else if ((m & (v2_pin | v2_signs)) == (v2_pin | v2_signs)) launch_split_gate_up_v2<Format, v2_on | v2_pin | v2_signs>(view, stream);
            else if (m & v2_pin) launch_split_gate_up_v2<Format, v2_on | v2_pin>(view, stream);
            else launch_split_gate_up_v2<Format, v2_on>(view, stream);
            return;
        }
    } else if constexpr (Format == WeightFormat::q8_0) {
        if (const unsigned m = prefill_v2(); (m & v2_on) && prefill_v2_q8()) {
            if (m & v2_pin) launch_split_gate_up_v2<Format, v2_on | v2_pin>(view, stream);
            else launch_split_gate_up_v2<Format, v2_on>(view, stream);
            return;
        }
    }
    switch (prefill_form()) {
    case 0: launch_split_gate_up_form<Format, 0>(view, stream); break;
    case 1: launch_split_gate_up_form<Format, 1>(view, stream); break;
    case 2: launch_split_gate_up_form<Format, 2>(view, stream); break;
    case 3: launch_split_gate_up_form<Format, 3>(view, stream); break;
    case 6: launch_split_gate_up_form<Format, 6>(view, stream); break;
    case 7: launch_split_gate_up_form<Format, 7>(view, stream); break;
    default: launch_split_gate_up_form<Format, 4>(view, stream); break;
    }
}
template <WeightFormat Format, int Form>
void launch_split_down_form(const PrefillDownView& view, hipStream_t stream) {
    constexpr unsigned lds = split_lds_for<Format, stage_buffers_of<Form>>();
    hipLaunchKernelGGL(HIP_KERNEL_NAME(split_down_kernel<Format, Form>), dim3(view.routes.max_batches, view.outputs / 64U), dim3(split_threads), lds, stream, view);
}
template <WeightFormat Format>
void launch_split_down(const PrefillDownView& view, hipStream_t stream) {
    if constexpr (Format == WeightFormat::q2_k) {
        if (const unsigned m = prefill_v2(); m & v2_on) {
            switch (m & (v2_fma | v2_pin)) {  // prefill_v2() always sets v2_part
            case v2_fma | v2_pin: launch_split_down_v2<Format, v2_on | v2_fma | v2_part | v2_pin>(view, stream); break;
            case v2_fma: launch_split_down_v2<Format, v2_on | v2_fma | v2_part>(view, stream); break;
            case v2_pin: launch_split_down_v2<Format, v2_on | v2_part | v2_pin>(view, stream); break;
            default: launch_split_down_v2<Format, v2_on | v2_part>(view, stream); break;
            }
            return;
        }
    } else if constexpr (Format == WeightFormat::q8_0) {
        if (const unsigned m = prefill_v2(); (m & v2_on) && prefill_v2_q8()) {
            if (m & v2_pin) launch_split_down_v2<Format, v2_on | v2_pin>(view, stream);
            else launch_split_down_v2<Format, v2_on>(view, stream);
            return;
        }
    }
    // the down formats (Q2_K, MXFP4, Q8_0) have no IQ2_XXS sign table: forms 1 and 2 are the same kernel
    switch (prefill_form()) {
    case 0: launch_split_down_form<Format, 0>(view, stream); break;
    case 1: case 2: launch_split_down_form<Format, 1>(view, stream); break;
    case 3: launch_split_down_form<Format, 3>(view, stream); break;
    case 6: launch_split_down_form<Format, 6>(view, stream); break;
    case 7: launch_split_down_form<Format, 6>(view, stream); break;  // the batched stage spills in the down kernel
    default: launch_split_down_form<Format, 4>(view, stream); break;
    }
}
void require_routes(const PrefillRoutesView& r) {
    if (!r.begins || !r.entry_rows || !r.entry_slots || !r.batch_expert || !r.batch_begin || !r.batch_count || !r.batch_total ||
        r.max_batches == 0U || (r.batch_rows != 16U && r.batch_rows != 32U))
        throw std::invalid_argument("ds41 prefill routes view is inconsistent");
}

// Shared tiled core route (opt-in): the same bits as the kernels above, one implementation for every MoE engine.
#include "ds41_prefill_adapter.inc"

template <WeightFormat Format, unsigned MT, unsigned J>
void launch_gate_up_j(const PrefillGateUpView& view, const StageLayout& layout, hipStream_t stream) {
    const unsigned bytes = std::max(layout.bytes(false), 2U * 32U * 64U * 4U);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(prefill_gate_up_kernel<Format, MT, J>), dim3(view.routes.max_batches, view.inter / 64U),
                       dim3(32U * prefill_waves), bytes, stream, view, layout);
}
template <WeightFormat Format, unsigned MT>
void launch_gate_up(const PrefillGateUpView& view, hipStream_t stream) {
    const auto layout = stage_layout(view.hidden, 16U * MT);
    switch (layout.groups_per_class) {
    case 1: launch_gate_up_j<Format, MT, 1>(view, layout, stream); break;
    case 2: launch_gate_up_j<Format, MT, 2>(view, layout, stream); break;
    case 3: launch_gate_up_j<Format, MT, 3>(view, layout, stream); break;
    case 4: launch_gate_up_j<Format, MT, 4>(view, layout, stream); break;
    default: launch_gate_up_j<Format, MT, 5>(view, layout, stream); break;
    }
}
template <WeightFormat Format, unsigned MT, unsigned J>
void launch_down_j(const PrefillDownView& view, const StageLayout& layout, hipStream_t stream) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(prefill_down_kernel<Format, MT, J>), dim3(view.routes.max_batches, view.outputs / 128U),
                       dim3(32U * prefill_waves), layout.bytes(Format == WeightFormat::q2_k), stream, view, layout);
}
template <WeightFormat Format, unsigned MT>
void launch_down(const PrefillDownView& view, hipStream_t stream) {
    const auto layout = stage_layout(view.width, 16U * MT);
    switch (layout.groups_per_class) {
    case 1: launch_down_j<Format, MT, 1>(view, layout, stream); break;
    case 2: launch_down_j<Format, MT, 2>(view, layout, stream); break;
    case 3: launch_down_j<Format, MT, 3>(view, layout, stream); break;
    case 4: launch_down_j<Format, MT, 4>(view, layout, stream); break;
    default: launch_down_j<Format, MT, 5>(view, layout, stream); break;
    }
}

} // namespace

void enqueue_build_prefill_routes(const std::uint32_t* ids, std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts,
                                  std::uint32_t split, std::uint32_t batch_rows, const PrefillRoutesBuffers& out, void* stream,
                                  std::uint32_t first, std::uint32_t end) {
    if (end == 0U) end = experts;
    if (first >= end || end > experts) throw std::invalid_argument("ds41 prefill route build: expert range");
    if (!ids || rows == 0U || top_k == 0U || top_k > 16U || experts == 0U || experts > 1024U ||
        (batch_rows != 8U && batch_rows != 16U && batch_rows != 32U) || !out.counts || !out.begins || !out.entry_rows || !out.entry_slots ||
        !out.batch_expert || !out.batch_begin || !out.batch_count || !out.batch_total || !out.order || !out.low_counts)
        throw std::invalid_argument("ds41 prefill route build is inconsistent");
    const auto queue = static_cast<hipStream_t>(stream);
    const unsigned slots = rows * top_k;
    const unsigned owned = end - first;
    check(hipMemsetAsync(out.counts, 0, owned * sizeof(std::uint32_t), queue), "route counts reset");
    hipLaunchKernelGGL(route_count_kernel, dim3((slots + 255U) / 256U), dim3(256), 0, queue, ids, slots, first, end, out.counts);
    if (out.sparse_max > 0U) {
        if (!out.sparse_expert || !out.sparse_begin || !out.sparse_total) throw std::invalid_argument("ds41 prefill route build: sparse arrays");
        hipLaunchKernelGGL(route_scan_split_kernel, dim3(1), dim3(1024), 0, queue, out.counts, owned, batch_rows, out);
    } else {
        hipLaunchKernelGGL(route_scan_kernel, dim3(1), dim3(1024), 0, queue, out.counts, owned, batch_rows, out);
    }
    hipLaunchKernelGGL(route_fill_kernel, dim3(owned), dim3(256), 0, queue, ids, slots, top_k, first, out);
    hipLaunchKernelGGL(route_order_kernel, dim3((rows + 255U) / 256U), dim3(256), 0, queue, ids, rows, top_k, split, out);
    check(hipGetLastError(), "route build launch");
}

PrefillRoutesView prefill_routes_view(const PrefillRoutesBuffers& b, std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts,
                                      std::uint32_t batch_rows) {
    PrefillRoutesView v;
    v.begins = b.begins; v.entry_rows = b.entry_rows; v.entry_slots = b.entry_slots; v.batch_expert = b.batch_expert;
    v.batch_begin = b.batch_begin; v.batch_count = b.batch_count; v.batch_total = b.batch_total;
    v.max_batches = prefill_max_batches(rows, top_k, experts, batch_rows);
    v.batch_rows = batch_rows;
    return v;
}

void enqueue_prefill_router(const RouterView& view, void* stream) {
    if (!view.x || !view.weight || !view.bias || !view.scores || !view.ids || !view.weights || view.rows == 0U || view.hidden == 0U ||
        view.hidden % 32U != 0U || view.experts == 0U || view.experts > 1024U || view.top_k == 0U || view.top_k > 16U ||
        view.top_k > view.experts || (view.image_rows && !view.bias_image))
        throw std::invalid_argument("ds41 prefill router view is inconsistent");
    const auto queue = static_cast<hipStream_t>(stream);
    static const bool wide = strix::envcfg::flag("STRIX_DS41_PREFILL_ROUTER_WIDE", true);
    RouterView v = view;
    v.wide_stage = wide && view.wide_stage && view.hidden % 4U == 0U && reinterpret_cast<std::uintptr_t>(view.x) % 16U == 0U &&
                   reinterpret_cast<std::uintptr_t>(view.weight) % 16U == 0U;
    // STRIX_DS41_PREFILL_ROUTER_V2 (read per call; default on, =0 off): the v2 scores kernel, the same bits
    const char* v2_env = std::getenv("STRIX_DS41_PREFILL_ROUTER_V2");
    const bool v2 = !(v2_env && v2_env[0] == '0') && v.wide_stage; // 16-byte staging admitted (alignment, hidden % 4)
    if (v2) {
        // experts fastest: the blocks of one row tile run together and share its activations in the L2
        hipLaunchKernelGGL(prefill_router_scores_v2_kernel, dim3((view.experts + prouter2_experts - 1U) / prouter2_experts,
                           (view.rows + prouter2_rows - 1U) / prouter2_rows), dim3(prouter2_threads), 0, queue, v);
    } else {
        hipLaunchKernelGGL(prefill_router_scores_kernel, dim3((view.rows + prouter_rows - 1U) / prouter_rows,
                           (view.experts + prouter_experts - 1U) / prouter_experts), dim3(prouter_threads), 0, queue, v);
    }
    hipLaunchKernelGGL(prefill_router_select_kernel, dim3(view.rows), dim3(256), 0, queue, view);
    check(hipGetLastError(), "prefill router launch");
}

// FAST lever ds41-moe-relaxed-v1: STRIX_DS41_FAST_MOE=1 (default off) or set_fast_moe(1 / 0; -1 = the environment)
int fast_moe_mode = -1;
void set_fast_moe(int mode) { fast_moe_mode = mode; }
// STRIX_DS41_FAST_MOE_ORDER (read per call; default on, =0 off): the relaxed kernels walk output blocks fastest (same bits;
// calls of more than 65535 batches keep the batch-fastest order, the grid y limit)
static bool fast_moe_order() {
    const char* e = std::getenv("STRIX_DS41_FAST_MOE_ORDER");
    return !(e && e[0] == '0');
}
bool fast_moe() {
    static const bool env = [] { const char* e = std::getenv("STRIX_DS41_FAST_MOE"); return e != nullptr && std::strcmp(e, "1") == 0; }();
    return fast_moe_mode < 0 ? env : fast_moe_mode != 0;
}
int fast_i4_mode = -1;
void set_fast_i4(int mode) { fast_i4_mode = mode; }
bool fast_i4() {
    const char* e = std::getenv("STRIX_DS41_FAST_I4");
    return fast_i4_mode < 0 ? (e && std::strcmp(e, "1") == 0) : fast_i4_mode != 0;
}
// STRIX_DS41_PF_CORE (resolved once per run, at the first product) or set_pf_core (harness override, may change between
// calls): which prefill products run on the shared tiled core (src/core) instead of the kernels of this file. 0 = none,
// moe = the routed and shared expert gate/up and down, dense = the dense Q8_0 x A8 products, 1 = both; a named product
// takes every format the core has (and rejects a width it cannot take). Unset: per form, the shared core where it is
// measured at least as fast as the kernel here (pf_core_for's argument at the call sites), the kernels of this file for
// everything else. Any other value, the empty one included, is rejected: a typo must not select a route silently. The
// same bits either way. FAST_I4 runs only on the shared core and implies it for the expert products.
enum : int { pf_core_auto = -1, pf_core_none = 0, pf_core_all = 1, pf_core_moe = 2, pf_core_dense_only = 3 };
int pf_core_policy_of(const char* raw) {
    static constexpr const char* names[] = {"0", "1", "moe", "dense"};
    for (int i = 0; i < 4; ++i)
        if (std::strcmp(raw, names[i]) == 0) return i;
    throw std::invalid_argument(std::string("STRIX_DS41_PF_CORE: expected 0, 1, moe or dense, got '") + raw + "'");
}
int pf_core_env() {
    static const int policy = [] {
        const char* raw = strix::envcfg::text("STRIX_DS41_PF_CORE");
        return raw ? pf_core_policy_of(raw) : int(pf_core_auto);
    }();
    return policy;
}
constexpr int pf_core_unset = -2;
int pf_core_override = pf_core_unset;
void set_pf_core(int mode) { pf_core_override = mode < 0 ? pf_core_unset : mode != 0 ? int(pf_core_all) : int(pf_core_none); }
void set_pf_core(const char* policy) { pf_core_override = policy ? pf_core_policy_of(policy) : int(pf_core_auto); }
// 1 = named on, 0 = named off, -1 = not named (decided per form)
int pf_core_request(bool dense) {
    const int p = pf_core_override != pf_core_unset ? pf_core_override : pf_core_env();
    if (p == pf_core_auto) return -1;
    return p == pf_core_all || p == (dense ? pf_core_dense_only : pf_core_moe) ? 1 : 0;
}
bool pf_core_for(bool faster_form) {
    const int r = pf_core_request(false);
    return r > 0 || (r < 0 && faster_form);
}
// The gate/up core loads whole 1024-element chunks of IQ2_XXS (no partial-chunk path): other widths keep the kernels below.
bool prefill_gate_up_on_core(WeightFormat format, std::uint32_t hidden, std::uint32_t batch_rows) {
    const bool faster_form = (format == WeightFormat::q8_0 || (format == WeightFormat::iq2_xxs && hidden % 1024U == 0U)) &&
                             batch_rows == 16U && !fast_moe() && !stage_forced() && hidden <= 5120U;
    return pf_core_for(faster_form) || fast_i4();
}
bool prefill_down_on_core(WeightFormat format, std::uint32_t width, std::uint32_t batch_rows) {
    const bool faster_form = (format == WeightFormat::q8_0 || format == WeightFormat::q2_k) && batch_rows == 16U &&
                             !fast_moe() && !stage_forced() && width <= 5120U;
    return pf_core_for(faster_form) || fast_i4();
}
bool pf_core() { return pf_core_request(false) > 0; }
bool pf_core_dense() { return pf_core_request(true) > 0; }
void enqueue_prefill_gate_up(WeightFormat format, const PrefillGateUpView& view, void* stream) {
    require_routes(view.routes);
    const auto elements = block_elements(format);
    if (format == WeightFormat::q2_k || !view.gate || !view.up || !view.codes || !view.deltas || !view.act_codes ||
        !view.act_deltas || !view.act_sums || (!view.gate_out) != (!view.up_out) || view.hidden == 0U ||
        view.hidden % elements != 0U || view.hidden % 32U != 0U || view.hidden > 32U * 32U * max_groups_per_class ||
        view.inter == 0U || view.inter % 64U != 0U || view.row_bytes != view.hidden / elements * block_bytes(format) ||
        reinterpret_cast<std::uintptr_t>(view.codes) % 16U != 0U)
        throw std::invalid_argument("ds41 prefill gate/up view is inconsistent");
    const auto queue = static_cast<hipStream_t>(stream);
    // Rank fixture at 2048 and 4096 rows in 16-row batches, shared core against the kernels below (same bits): Q8_0 (the
    // shared expert) -0.6..-2.2 % -> default; IQ2_XXS without the LDS weight tile (two workgroups per WGP) -7.1..-8.2 %
    // at 2048 / 4096 / 8192 rows -> default.
    // STRIX_DS41_PREFILL_STAGE=1 asks for the stage kernels of this file: only a named PF_CORE product overrides it.
    if (prefill_gate_up_on_core(format, view.hidden, view.routes.batch_rows)) {
        if (view.hidden > 5120U) throw std::invalid_argument("ds41 prefill gate/up view: shared core width");
        if (format == WeightFormat::iq2_xxs) launch_gate_up_core<WeightFormat::iq2_xxs>(view, queue);
        else if (format == WeightFormat::q8_0) launch_gate_up_core<WeightFormat::q8_0>(view, queue);
        else launch_gate_up_core<WeightFormat::mxfp4>(view, queue);
        check(hipGetLastError(), "prefill gate/up shared core");
        return;
    }
    if (fast_moe() && format == WeightFormat::iq2_xxs && view.hidden % 256U == 0U && view.inter % 128U == 0U &&
        view.routes.batch_rows <= relaxed_rows) {
        if (fast_moe_order() && view.routes.max_batches <= 65535U)  // grid y limit
            hipLaunchKernelGGL(HIP_KERNEL_NAME(relaxed_gate_up_kernel<WeightFormat::iq2_xxs, true>), dim3(view.inter / 128U, view.routes.max_batches),
                               dim3(relaxed_threads), 0, queue, view);
        else
            hipLaunchKernelGGL(HIP_KERNEL_NAME(relaxed_gate_up_kernel<WeightFormat::iq2_xxs, false>), dim3(view.routes.max_batches, view.inter / 128U),
                               dim3(relaxed_threads), 0, queue, view);
        check(hipGetLastError(), "prefill gate/up launch (relaxed)");
        return;
    }
    if (view.routes.batch_rows == 32U && format == WeightFormat::iq2_xxs && view.hidden % (split_chunk_groups * 32U) == 0U && prefill_r2()) {
        // STRIX_DS41_PREFILL_MOE_R2_GU=w: w2_gate_up_kernel (one pass, 32 waves) instead of r2_gate_up_kernel (same bits;
        // measured +30 % at 8K vs v2: one 32-wave workgroup per WGP, 192 VGPR with spills)
        if (const char* gu = std::getenv("STRIX_DS41_PREFILL_MOE_R2_GU"); gu && std::strcmp(gu, "w") == 0) {
            const unsigned m = prefill_v2();
            const dim3 grid(view.routes.max_batches, view.inter / 32U);
            if ((m & (v2_pin | v2_signs)) == (v2_pin | v2_signs))
                hipLaunchKernelGGL(HIP_KERNEL_NAME(w2_gate_up_kernel<WeightFormat::iq2_xxs, v2_on | v2_pin | v2_signs>), grid, dim3(w2_threads),
                                   w2_lds<WeightFormat::iq2_xxs>, queue, view, prefill_band());
            else if (m & v2_pin)
                hipLaunchKernelGGL(HIP_KERNEL_NAME(w2_gate_up_kernel<WeightFormat::iq2_xxs, v2_on | v2_pin>), grid, dim3(w2_threads),
                                   w2_lds<WeightFormat::iq2_xxs>, queue, view, prefill_band());
            else
                hipLaunchKernelGGL(HIP_KERNEL_NAME(w2_gate_up_kernel<WeightFormat::iq2_xxs, v2_on>), grid, dim3(w2_threads),
                                   w2_lds<WeightFormat::iq2_xxs>, queue, view, prefill_band());
            check(hipGetLastError(), "prefill gate/up launch");
            return;
        }
        hipLaunchKernelGGL(HIP_KERNEL_NAME(r2_gate_up_kernel<WeightFormat::iq2_xxs>), dim3(view.routes.max_batches, view.inter / 32U),
                           dim3(split_threads), r2_lds<WeightFormat::iq2_xxs>, queue, view, prefill_band());
        check(hipGetLastError(), "prefill gate/up launch");
        return;
    }
    if (view.routes.batch_rows == 16U && !stage_forced() && view.hidden % 256U == 0U) {
        if (format == WeightFormat::iq2_xxs) launch_split_gate_up<WeightFormat::iq2_xxs>(view, queue);
        else if (format == WeightFormat::q8_0) launch_split_gate_up<WeightFormat::q8_0>(view, queue);
        else launch_split_gate_up<WeightFormat::mxfp4>(view, queue);
        check(hipGetLastError(), "prefill gate/up launch");
        return;
    }
    const bool two = view.routes.batch_rows == 32U;
    if (format == WeightFormat::iq2_xxs) two ? launch_gate_up<WeightFormat::iq2_xxs, 2>(view, queue) : launch_gate_up<WeightFormat::iq2_xxs, 1>(view, queue);
    else if (format == WeightFormat::q8_0) two ? launch_gate_up<WeightFormat::q8_0, 2>(view, queue) : launch_gate_up<WeightFormat::q8_0, 1>(view, queue);
    else two ? launch_gate_up<WeightFormat::mxfp4, 2>(view, queue) : launch_gate_up<WeightFormat::mxfp4, 1>(view, queue);
    check(hipGetLastError(), "prefill gate/up launch");
}

void enqueue_prefill_down(WeightFormat format, const PrefillDownView& view, void* stream) {
    require_routes(view.routes);
    const auto elements = block_elements(format);
    if (format == WeightFormat::iq2_xxs || !view.weights || !view.codes || !view.deltas || !view.out || view.width == 0U ||
        view.width % elements != 0U || view.width % 32U != 0U || view.width > 32U * 32U * max_groups_per_class ||
        view.outputs == 0U || view.outputs % 128U != 0U || view.row_bytes != view.width / elements * block_bytes(format) ||
        reinterpret_cast<std::uintptr_t>(view.codes) % 16U != 0U)
        throw std::invalid_argument("ds41 prefill down view is inconsistent");
    const auto queue = static_cast<hipStream_t>(stream);
    // Rank fixture as in gate/up: Q8_0 (the shared expert) -6.0..-7.2 % -> default; Q2_K without the LDS weight tile
    // -2.4..-3.4 % at 2048 / 4096 / 8192 rows -> default. STRIX_DS41_PREFILL_STAGE=1 as in gate/up.
    if (prefill_down_on_core(format, view.width, view.routes.batch_rows)) {
        if (view.width > 5120U) throw std::invalid_argument("ds41 prefill down view: shared core width");
        if (format == WeightFormat::q2_k) launch_down_core<WeightFormat::q2_k>(view, queue);
        else if (format == WeightFormat::q8_0) launch_down_core<WeightFormat::q8_0>(view, queue);
        else launch_down_core<WeightFormat::mxfp4>(view, queue);
        check(hipGetLastError(), "prefill down shared core");
        return;
    }
    if (fast_moe() && format == WeightFormat::q2_k && view.width % 256U == 0U && view.outputs % 256U == 0U &&
        view.routes.batch_rows <= relaxed_rows) {
        // STRIX_DS41_FAST_MOE_DOWN_ROWS (read per call; default 32, =64 the previous row groups): the same bits either way
        const char* rows_env = std::getenv("STRIX_DS41_FAST_MOE_DOWN_ROWS");
        const bool rows32 = !(rows_env && std::strcmp(rows_env, "64") == 0) && view.routes.batch_rows <= 32U;
        const bool swap = fast_moe_order() && view.routes.max_batches <= 65535U;  // grid y limit
        const dim3 grid = swap ? dim3(view.outputs / 256U, view.routes.max_batches) : dim3(view.routes.max_batches, view.outputs / 256U);
        if (swap && rows32) hipLaunchKernelGGL(HIP_KERNEL_NAME(relaxed_down_kernel<WeightFormat::q2_k, true, 32U>), grid, dim3(relaxed_threads), 0, queue, view);
        else if (swap) hipLaunchKernelGGL(HIP_KERNEL_NAME(relaxed_down_kernel<WeightFormat::q2_k, true>), grid, dim3(relaxed_threads), 0, queue, view);
        else if (rows32) hipLaunchKernelGGL(HIP_KERNEL_NAME(relaxed_down_kernel<WeightFormat::q2_k, false, 32U>), grid, dim3(relaxed_threads), 0, queue, view);
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(relaxed_down_kernel<WeightFormat::q2_k, false>), grid, dim3(relaxed_threads), 0, queue, view);
        check(hipGetLastError(), "prefill down launch (relaxed)");
        return;
    }
    if (view.routes.batch_rows == 32U && format == WeightFormat::q2_k && view.width % 256U == 0U && prefill_r2()) {
        // STRIX_DS41_PREFILL_MOE_R2_DOWN=0: the v2 down kernel in 16-row halves instead of r2_down_kernel
        const char* rd = std::getenv("STRIX_DS41_PREFILL_MOE_R2_DOWN");
        if (rd && std::strcmp(rd, "0") == 0) launch_split_down<WeightFormat::q2_k>(view, queue);
        else hipLaunchKernelGGL(r2_down_kernel, dim3(view.routes.max_batches, view.outputs / 64U), dim3(split_threads), r2d_lds, queue, view,
                                prefill_band());
        check(hipGetLastError(), "prefill down launch");
        return;
    }
    if (view.routes.batch_rows == 16U && !stage_forced() && view.width % 256U == 0U) {
        if (format == WeightFormat::q2_k) launch_split_down<WeightFormat::q2_k>(view, queue);
        else if (format == WeightFormat::q8_0) launch_split_down<WeightFormat::q8_0>(view, queue);
        else launch_split_down<WeightFormat::mxfp4>(view, queue);
        check(hipGetLastError(), "prefill down launch");
        return;
    }
    const bool two = view.routes.batch_rows == 32U;
    if (format == WeightFormat::q2_k) two ? launch_down<WeightFormat::q2_k, 2>(view, queue) : launch_down<WeightFormat::q2_k, 1>(view, queue);
    else if (format == WeightFormat::q8_0) two ? launch_down<WeightFormat::q8_0, 2>(view, queue) : launch_down<WeightFormat::q8_0, 1>(view, queue);
    else two ? launch_down<WeightFormat::mxfp4, 2>(view, queue) : launch_down<WeightFormat::mxfp4, 1>(view, queue);
    check(hipGetLastError(), "prefill down launch");
}

namespace {
// Slot s's A8 activation (inter codes, inter / 32 deltas) at the down's stride: codes and deltas copied, the groups
// past inter zero (code 0, delta 0: each adds +0 to its lane class, so the product is the inter-wide one).
__global__ __launch_bounds__(256) void pad_a8_kernel(const std::int8_t* codes, const float* deltas, std::uint32_t inter,
                                                     std::uint32_t width, std::int8_t* out_codes, float* out_deltas) {
    const std::uint64_t s = blockIdx.x;
    const auto* src = codes + s * inter;
    auto* dst = out_codes + s * width;
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) dst[i] = i < inter ? src[i] : static_cast<std::int8_t>(0);
    for (unsigned g = threadIdx.x; g < width / 32U; g += blockDim.x)
        out_deltas[s * (width / 32U) + g] = g < inter / 32U ? deltas[s * (inter / 32U) + g] : 0.0F;
}
}  // namespace

void PrefillMoe::pad_activations(std::uint32_t rows, void* stream) {
    if (pad_codes_ == nullptr || rows == 0U) return;
    hipLaunchKernelGGL(pad_a8_kernel, dim3(rows * top_k_), dim3(256), 0, static_cast<hipStream_t>(stream), act_codes_, act_deltas_,
                       w_.inter, w_.down_width, pad_codes_, pad_deltas_);
    check(hipGetLastError(), "prefill MoE pad");
}

PrefillMoe::PrefillMoe(const PrefillMoeWeights& weights, std::uint32_t top_k, std::uint32_t experts, std::uint32_t split,
                       std::uint32_t chunk_rows, std::uint32_t first, std::uint32_t end)
    : w_(weights), top_k_(top_k), experts_(experts), split_(split), chunk_rows_(chunk_rows), first_(first), end_(end ? end : experts) {
    if (top_k == 0U || top_k > 16U || experts == 0U || chunk_rows == 0U || first_ >= end_ || end_ > experts || !w_.gate || !w_.up ||
        !w_.down || w_.hidden == 0U || w_.inter == 0U || (w_.down_width != 0U && (w_.down_width < w_.inter || w_.down_width % 32U != 0U)))
        throw std::invalid_argument("ds41 prefill MoE configuration is inconsistent");
    if (w_.down_width == 0U) w_.down_width = w_.inter;
    const std::uint64_t slots = static_cast<std::uint64_t>(chunk_rows) * top_k, owned = end_ - first_;
    // Batches of 8 (window path) bound the batch count of either path.
    const std::uint64_t batches = prefill_max_batches(chunk_rows, top_k, static_cast<std::uint32_t>(owned), 8U);
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_WINDOW_MEAN"); v && *v) window_mean_ = std::strtod(v, nullptr);
    if (const char* v = std::getenv("STRIX_DS41_PREFILL_SELECT"); v && *v) select_hist_ = std::strcmp(v, "mean") != 0;
    // One allocation, 256-byte aligned parts.
    std::uint64_t offset = 0;
    const auto take = [&](std::uint64_t bytes) { const auto at = offset; offset += (bytes + 255U) / 256U * 256U; return at; };
    const auto o_counts = take(owned * 4U), o_begins = take((owned + 1U) * 4U), o_rows = take(slots * 4U), o_slots = take(slots * 4U);
    const auto o_bexp = take(batches * 4U), o_bbeg = take((batches + 1U) * 4U), o_bcnt = take(batches * 4U), o_btot = take(4U);
    const auto o_order = take(slots * 4U), o_low = take(static_cast<std::uint64_t>(chunk_rows) * 4U);
    const auto o_sexp = take(batches * 4U), o_sbeg = take((batches + 1U) * 4U), o_stot = take(4U);
    const auto o_codes = take(slots * w_.inter), o_deltas = take(slots * w_.inter / 32U * 4U), o_sums = take(slots * w_.inter / 32U * 4U);
    const auto o_raw = take(slots * w_.hidden * 4U);
    const bool pad = w_.down_width != w_.inter;
    const auto o_pcodes = pad ? take(slots * w_.down_width) : 0U, o_pdeltas = pad ? take(slots * w_.down_width / 32U * 4U) : 0U;
    scratch_bytes_ = offset;
    check(hipMalloc(&memory_, offset), "prefill MoE scratch");
    auto* base = static_cast<unsigned char*>(memory_);
    const auto u32 = [&](std::uint64_t at) { return reinterpret_cast<std::uint32_t*>(base + at); };
    routes_ = {u32(o_counts), u32(o_begins), u32(o_rows), u32(o_slots), u32(o_bexp), u32(o_bbeg), u32(o_bcnt), u32(o_btot),
               u32(o_order), u32(o_low)};
    routes_.sparse_expert = u32(o_sexp);
    routes_.sparse_begin = u32(o_sbeg);
    routes_.sparse_total = u32(o_stot);
    act_codes_ = reinterpret_cast<std::int8_t*>(base + o_codes);
    act_deltas_ = reinterpret_cast<float*>(base + o_deltas);
    act_sums_ = reinterpret_cast<std::int32_t*>(base + o_sums);
    raw_ = reinterpret_cast<float*>(base + o_raw);
    if (pad) {
        pad_codes_ = reinterpret_cast<std::int8_t*>(base + o_pcodes);
        pad_deltas_ = reinterpret_cast<float*>(base + o_pdeltas);
    }
}
PrefillMoe::~PrefillMoe() {
    (void)hipFree(memory_);
    if (joint_) (void)hipFree(joint_);
}
void PrefillMoe::set_weights(const PrefillMoeWeights& weights) {
    if (!weights.gate || !weights.up || !weights.down || weights.hidden != w_.hidden || weights.inter != w_.inter ||
        (weights.down_width != 0U ? weights.down_width : weights.inter) != w_.down_width ||
        weights.gate_up_format != w_.gate_up_format || weights.down_format != w_.down_format)
        throw std::invalid_argument("ds41 prefill MoE: set_weights needs the same geometry and formats");
    const std::uint32_t down_width = w_.down_width;
    w_ = weights;
    w_.down_width = down_width;
}

void PrefillMoe::enqueue_experts(const PrefillMoeInputs& in, std::uint32_t chunk, std::uint32_t& row0, std::uint32_t& rows, void* stream) {
    if (!in.ids || !in.weights || !in.codes || !in.deltas || in.rows == 0U || chunk >= chunks(in.rows))
        throw std::invalid_argument("ds41 prefill MoE inputs are inconsistent");
    row0 = chunk * chunk_rows_;
    rows = std::min(chunk_rows_, in.rows - row0);
    experts_rows(in, row0, rows, stream);
}
// Routes and expert products of rows [row0, row0 + rows) of `in` (at most chunk_rows), by the selected path.
unsigned PrefillMoe::selected_batch_rows(unsigned /*rows*/) const {
    return prefill_r2() && w_.gate_up_format == WeightFormat::iq2_xxs && w_.down_format == WeightFormat::q2_k &&
                   w_.hidden % (split_chunk_groups * 32U) == 0U && w_.down_width % 256U == 0U ? 32U : 16U;
}
void PrefillMoe::experts_rows(const PrefillMoeInputs& in, std::uint32_t row0, std::uint32_t rows, void* stream) {
    const std::uint32_t owned = end_ - first_;
    const auto* ids = in.ids + static_cast<std::uint64_t>(row0) * top_k_;
    if (select_hist_ && !fast_i4()) {
        enqueue_experts_split(in, row0, rows, stream);
        return;
    }
    routes_.sparse_max = 0U;
    // FAST MoE: every entry takes the relaxed batch kernels, so its arithmetic does not depend on the chunking (rows per
    // call would otherwise move small chunks onto the exact window kernels)
    const bool window = !fast_moe() && !fast_i4() && static_cast<double>(rows) * top_k_ <= window_mean_ * owned;
    const std::uint32_t batch_rows = window ? 8U : selected_batch_rows(rows);
    enqueue_build_prefill_routes(ids, rows, top_k_, experts_, split_, batch_rows, routes_, stream, first_, end_);
    if (window) {
        // Each batch (<= 8 entries of one expert) is one window block of the decode kernels.
        RoutesView gr;
        gr.experts = routes_.batch_expert; gr.begins = routes_.batch_begin; gr.inputs = routes_.entry_rows; gr.outputs = routes_.entry_slots;
        gr.expert_count = prefill_max_batches(rows, top_k_, owned, 8U); gr.max_rows = 8U; gr.counts = routes_.batch_total;
        GateUpView gu;
        gu.gate = w_.gate; gu.up = w_.up; gu.expert_stride_bytes = w_.gate_row_bytes * w_.inter; gu.row_bytes = w_.gate_row_bytes;
        gu.hidden = w_.hidden; gu.inter = w_.inter; gu.routes = gr;
        gu.codes = in.codes + static_cast<std::uint64_t>(row0) * w_.hidden;
        gu.deltas = in.deltas + static_cast<std::uint64_t>(row0) * (w_.hidden / 32U);
        gu.limit = w_.limit; gu.act_codes = act_codes_; gu.act_deltas = act_deltas_; gu.act_sums = act_sums_;
        enqueue_gate_up(w_.gate_up_format, gu, stream);
        RoutesProductView dn;
        RoutesView dr = gr;
        dr.inputs = routes_.entry_slots;
        dn.weights = w_.down; dn.expert_stride_bytes = w_.down_row_bytes * w_.hidden; dn.row_bytes = w_.down_row_bytes;
        pad_activations(rows, stream);
        dn.width = w_.down_width; dn.outputs = w_.hidden; dn.routes = dr; dn.codes = down_codes(); dn.deltas = down_deltas(); dn.out = raw_;
        enqueue_routes_product(w_.down_format, dn, stream);
        return;
    }
    const auto view = prefill_routes_view(routes_, rows, top_k_, owned, batch_rows);
    PrefillGateUpView gu;
    gu.gate = w_.gate; gu.up = w_.up; gu.expert_stride_bytes = w_.gate_row_bytes * w_.inter; gu.row_bytes = w_.gate_row_bytes;
    gu.hidden = w_.hidden; gu.inter = w_.inter; gu.routes = view;
    gu.codes = in.codes + static_cast<std::uint64_t>(row0) * w_.hidden;
    gu.deltas = in.deltas + static_cast<std::uint64_t>(row0) * (w_.hidden / 32U);
    gu.limit = w_.limit; gu.act_codes = act_codes_; gu.act_deltas = act_deltas_; gu.act_sums = act_sums_;
    enqueue_prefill_gate_up(w_.gate_up_format, gu, stream);
    PrefillDownView dn;
    dn.weights = w_.down; dn.expert_stride_bytes = w_.down_row_bytes * w_.hidden; dn.row_bytes = w_.down_row_bytes;
    pad_activations(rows, stream);
    dn.width = w_.down_width; dn.outputs = w_.hidden; dn.routes = view; dn.codes = down_codes(); dn.deltas = down_deltas(); dn.out = raw_;
    enqueue_prefill_down(w_.down_format, dn, stream);
}
// Histogram selector: one route build with the sparse experts (<= window_mean entries) first; their batches of 8 run
// the decode window kernels and the dense experts' batches of 16 the WMMA kernels (each kernel exits at once when its
// list is empty). Every entry is the declared product whichever kernel runs it.
void PrefillMoe::enqueue_experts_split(const PrefillMoeInputs& in, std::uint32_t row0, std::uint32_t rows, void* stream) {
    if (fast_i4()) { experts_rows(in, row0, rows, stream); return; }
    const std::uint32_t owned = end_ - first_;
    const auto* ids = in.ids + static_cast<std::uint64_t>(row0) * top_k_;
    routes_.sparse_max = window_mean_ > 0.0 && !fast_moe() ? static_cast<std::uint32_t>(window_mean_) : 0U;   // FAST MoE: no sparse split
    const unsigned batch_rows = selected_batch_rows(rows);
    enqueue_build_prefill_routes(ids, rows, top_k_, experts_, split_, batch_rows, routes_, stream, first_, end_);
    const auto* codes = in.codes + static_cast<std::uint64_t>(row0) * w_.hidden;
    const auto* deltas = in.deltas + static_cast<std::uint64_t>(row0) * (w_.hidden / 32U);
    if (routes_.sparse_max > 0U) {
        RoutesView gr;
        gr.experts = routes_.sparse_expert; gr.begins = routes_.sparse_begin; gr.inputs = routes_.entry_rows; gr.outputs = routes_.entry_slots;
        gr.expert_count = prefill_max_batches(rows, top_k_, owned, 8U); gr.max_rows = 8U; gr.counts = routes_.sparse_total;
        GateUpView gu;
        gu.gate = w_.gate; gu.up = w_.up; gu.expert_stride_bytes = w_.gate_row_bytes * w_.inter; gu.row_bytes = w_.gate_row_bytes;
        gu.hidden = w_.hidden; gu.inter = w_.inter; gu.routes = gr; gu.codes = codes; gu.deltas = deltas;
        gu.limit = w_.limit; gu.act_codes = act_codes_; gu.act_deltas = act_deltas_; gu.act_sums = act_sums_;
        enqueue_gate_up(w_.gate_up_format, gu, stream);
        RoutesProductView dn;
        RoutesView dr = gr;
        dr.inputs = routes_.entry_slots;
        dn.weights = w_.down; dn.expert_stride_bytes = w_.down_row_bytes * w_.hidden; dn.row_bytes = w_.down_row_bytes;
        pad_activations(rows, stream);
        dn.width = w_.down_width; dn.outputs = w_.hidden; dn.routes = dr; dn.codes = down_codes(); dn.deltas = down_deltas(); dn.out = raw_;
        enqueue_routes_product(w_.down_format, dn, stream);
    }
    const auto view = prefill_routes_view(routes_, rows, top_k_, owned, batch_rows);
    PrefillGateUpView gu;
    gu.gate = w_.gate; gu.up = w_.up; gu.expert_stride_bytes = w_.gate_row_bytes * w_.inter; gu.row_bytes = w_.gate_row_bytes;
    gu.hidden = w_.hidden; gu.inter = w_.inter; gu.routes = view; gu.codes = codes; gu.deltas = deltas;
    gu.limit = w_.limit; gu.act_codes = act_codes_; gu.act_deltas = act_deltas_; gu.act_sums = act_sums_;
    enqueue_prefill_gate_up(w_.gate_up_format, gu, stream);
    PrefillDownView dn;
    dn.weights = w_.down; dn.expert_stride_bytes = w_.down_row_bytes * w_.hidden; dn.row_bytes = w_.down_row_bytes;
    pad_activations(rows, stream);
    dn.width = w_.down_width; dn.outputs = w_.hidden; dn.routes = view; dn.codes = down_codes(); dn.deltas = down_deltas(); dn.out = raw_;
    enqueue_prefill_down(w_.down_format, dn, stream);
}
void PrefillMoe::enqueue_combine(const PrefillMoeInputs& in, std::uint32_t chunk, float* y, void* stream) {
    if (first_ != 0U || end_ != experts_ || !in.shared || !y) throw std::invalid_argument("ds41 prefill MoE combine needs every expert");
    std::uint32_t row0 = 0, rows = 0;
    enqueue_experts(in, chunk, row0, rows, stream);
    CombineView c;
    c.raw = raw_; c.shared = in.shared + static_cast<std::uint64_t>(row0) * w_.hidden; c.weights = in.weights + static_cast<std::uint64_t>(row0) * top_k_;
    c.order = routes_.order; c.low_counts = routes_.low_counts; c.rows = rows; c.top_k = top_k_; c.hidden = w_.hidden;
    c.out = y + static_cast<std::uint64_t>(row0) * w_.hidden;
    strix::ds41::enqueue_combine(c, stream);
}
// Half sums of rows [first_row, first_row + rows) of the current chunk's routes into out[row - first_row].
void PrefillMoe::half_rows(const float* weights, std::uint32_t first_row, std::uint32_t rows, std::uint32_t half, float* out, void* stream) {
    if (rows == 0U) return;
    HalfSumView v;
    v.raw = raw_; v.weights = weights; v.order = routes_.order + static_cast<std::uint64_t>(first_row) * top_k_;
    v.low_counts = routes_.low_counts + first_row; v.rows = rows; v.top_k = top_k_; v.hidden = w_.hidden; v.half = half; v.out = out;
    enqueue_half_sum(v, stream);
}
void PrefillMoe::enqueue_half_joint(const PrefillMoeInputs& peer, const PrefillMoeInputs& own, std::uint32_t half, float* out_peer,
                                    float* out_own, void* stream, void* peer_done) {
    const bool covers = half == 0U ? first_ == 0U && end_ == split_ : first_ == split_ && end_ == experts_;
    if (half > 1U || !covers || (peer.rows && (!peer.ids || !peer.weights || !peer.codes || !peer.deltas || !out_peer)) ||
        (own.rows && (!own.ids || !own.weights || !own.codes || !own.deltas || !out_own)))
        throw std::invalid_argument("ds41 prefill MoE joint half: inconsistent inputs or owned range");
    const auto queue = static_cast<hipStream_t>(stream);
    const std::uint64_t H = w_.hidden, G = w_.hidden / 32U, K = top_k_, C = chunk_rows_;
    const std::uint64_t codes_at = 0, deltas_at = (C * H + 255U) / 256U * 256U, ids_at = deltas_at + (C * G * 4U + 255U) / 256U * 256U,
                        weights_at = ids_at + (C * K * 4U + 255U) / 256U * 256U, total = weights_at + C * K * 4U;
    if (!joint_) check(hipMalloc(&joint_, total), "prefill MoE joint scratch");
    auto* base = static_cast<unsigned char*>(joint_);
    PrefillMoeInputs j;
    j.codes = reinterpret_cast<const std::int8_t*>(base + codes_at);
    j.deltas = reinterpret_cast<const float*>(base + deltas_at);
    j.ids = reinterpret_cast<const std::uint32_t*>(base + ids_at);
    j.weights = reinterpret_cast<const float*>(base + weights_at);
    const std::uint32_t np = peer.rows, rows_all = peer.rows + own.rows;
    // copies rows [from, from + n) of `src` to joint row `to`
    const auto copy = [&](const PrefillMoeInputs& src, std::uint32_t from, std::uint32_t n, std::uint32_t to) {
        if (n == 0U) return;
        check(hipMemcpyAsync(base + codes_at + to * H, src.codes + from * H, n * H, hipMemcpyDeviceToDevice, queue), "joint codes");
        check(hipMemcpyAsync(base + deltas_at + to * G * 4U, src.deltas + from * G, n * G * 4U, hipMemcpyDeviceToDevice, queue), "joint deltas");
        check(hipMemcpyAsync(base + ids_at + to * K * 4U, src.ids + from * K, n * K * 4U, hipMemcpyDeviceToDevice, queue), "joint ids");
        check(hipMemcpyAsync(base + weights_at + to * K * 4U, src.weights + from * K, n * K * 4U, hipMemcpyDeviceToDevice, queue), "joint weights");
    };
    for (std::uint32_t r0 = 0; r0 < rows_all; r0 += chunk_rows_) {
        const std::uint32_t n = std::min(chunk_rows_, rows_all - r0);
        const std::uint32_t kp = r0 < np ? std::min(n, np - r0) : 0U;  // the chunk's peer rows come first
        copy(peer, r0, kp, 0U);
        copy(own, r0 + kp - np, n - kp, kp);
        j.rows = n;
        experts_rows(j, 0U, n, stream);
        half_rows(j.weights, 0U, kp, half, out_peer + static_cast<std::uint64_t>(r0) * H, stream);
        if (peer_done && kp > 0U && r0 + kp == np) check(hipEventRecord(static_cast<hipEvent_t>(peer_done), queue), "peer done");
        half_rows(j.weights, kp, n - kp, half, out_own + static_cast<std::uint64_t>(r0 + kp - np) * H, stream);
    }
    if (peer_done && np == 0U) check(hipEventRecord(static_cast<hipEvent_t>(peer_done), queue), "peer done");
}
void PrefillMoe::enqueue_half(const PrefillMoeInputs& in, std::uint32_t chunk, std::uint32_t half, float* out, void* stream) {
    const bool covers = half == 0U ? first_ == 0U && end_ == split_ : first_ == split_ && end_ == experts_;
    if (half > 1U || !covers || !out) throw std::invalid_argument("ds41 prefill MoE half: the owned experts must be that half");
    std::uint32_t row0 = 0, rows = 0;
    enqueue_experts(in, chunk, row0, rows, stream);
    HalfSumView v;
    v.raw = raw_; v.weights = in.weights + static_cast<std::uint64_t>(row0) * top_k_; v.order = routes_.order;
    v.low_counts = routes_.low_counts; v.rows = rows; v.top_k = top_k_; v.hidden = w_.hidden; v.half = half;
    v.out = out + static_cast<std::uint64_t>(row0) * w_.hidden;
    enqueue_half_sum(v, stream);
}

} // namespace strix::ds41
