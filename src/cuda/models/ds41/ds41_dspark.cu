// deepseek41 DSpark drafter on CUDA: the kernels of hip_ds41_dspark.hip (wave32 = one CUDA warp, same reduction
// trees) and the resident drafter, on the CUDA layer core for the operations it shares with the target.
#include "strix/cuda/ds41_dspark.hpp"

#include "strix/cuda/ds41_core.hpp"
#include "strix/cuda/ds41_moe.hpp"
#include "strix/ds41_scalar_codecs.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strix::ds41::cuda {
namespace {

using namespace strix::ds41::dspark;

__device__ __forceinline__ float f16_device(std::uint16_t bits) { return __half2float(__ushort_as_half(bits)); }
// Wave-order reduction: every lane holds its ascending partial sum; butterfly d = 16..1.
__device__ __forceinline__ float butterfly(float v) {
    for (int d = 16; d != 0; d /= 2) v = v + __shfl_xor_sync(0xffffffffU, v, d, 32);
    return v;
}
__device__ __forceinline__ float entry_value_device(const std::uint8_t* entry, unsigned i) {
    return decode_e4m3(entry[i]) * scale_value(entry[head_dim + i / 32U]);
}

__global__ __launch_bounds__(256) void main_hidden_kernel(const float* l37, const float* l38, const float* l39, unsigned stride, float* out) {
    const unsigned i = blockIdx.x * 256U + threadIdx.x, r = blockIdx.y;
    if (i >= main_width) return;
    const unsigned l = i / hidden, c = i % hidden;
    const float* s = (l == 0U ? l37 : l == 1U ? l38 : l39) + static_cast<std::size_t>(r) * stride;
    out[static_cast<std::size_t>(r) * main_width + i] = (((s[c] + s[hidden + c]) + s[2U * hidden + c]) + s[3U * hidden + c]) * 0.25F;
}

// Chunked sum (chunk_sum): warp w of the block sums chunk w; the chunk sums are added in order.
__device__ __forceinline__ float block_chunk_sum(float lane_acc, float* partial) {
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U, waves = blockDim.x / 32U;
    const float part = butterfly(lane_acc);
    if (lane == 0U) partial[wave] = part;
    __syncthreads();
    float total = partial[0];
    for (unsigned c = 1; c < waves; ++c) total = total + partial[c];
    __syncthreads();
    return total;
}

// One block per row, one warp per 640-value chunk of the row (width <= 640: one warp).
__global__ __launch_bounds__(1024) void rms_norm_kernel(const float* x, unsigned width, const float* weight, float* out) {
    __shared__ float partial[32];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    const float* row = x + static_cast<std::size_t>(blockIdx.x) * width;
    const unsigned first = wave * sum_chunk, end = min(width, first + sum_chunk);
    float acc = 0.0F;
    for (unsigned i = first + lane; i < end; i += 32U) acc = acc + row[i] * row[i];
    const float rstd = rstd_of(block_chunk_sum(acc, partial), width, norm_eps);
    float* y = out + static_cast<std::size_t>(blockIdx.x) * width;
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) y[i] = (row[i] * rstd) * weight[i];
}

// blockIdx.x < 24: dot of fn row j with the row's stream; == 24: sum of squares. One warp per 640-value chunk.
__global__ __launch_bounds__(1024) void hc_dot_kernel(const float* streams, const std::uint16_t* fn, float* scratch) {
    __shared__ float partial[32];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U, j = blockIdx.x, r = blockIdx.y;
    const float* x = streams + static_cast<std::size_t>(r) * stream_width;
    const unsigned first = wave * sum_chunk, end = first + sum_chunk;
    float acc = 0.0F;
    if (j < mix_count) {
        const std::uint16_t* f = fn + static_cast<std::size_t>(j) * stream_width;
        for (unsigned i = first + lane; i < end; i += 32U) acc = acc + f16_device(f[i]) * x[i];
    } else {
        for (unsigned i = first + lane; i < end; i += 32U) acc = acc + x[i] * x[i];
    }
    const float total = block_chunk_sum(acc, partial);
    if (threadIdx.x == 0U) scratch[r * (mix_count + 1U) + j] = total;
}
__global__ void sinkhorn_kernel(const float* scratch, const float* scale, const float* base, unsigned rows, float* mixes) {
    const unsigned r = threadIdx.x;
    if (r >= rows) return;
    const float* raw = scratch + r * (mix_count + 1U);
    const float rstd = rstd_of(raw[mix_count], stream_width, norm_eps);
    float m[mix_count];
    for (unsigned j = 0; j < mix_count; ++j) m[j] = raw[j] * rstd;
    sinkhorn_core(m, scale, base, mixes + r * mix_count);
}

__global__ __launch_bounds__(256) void hc_pre_kernel(const float* streams, const float* coefficients, unsigned stride, float* out) {
    const unsigned i = blockIdx.x * 256U + threadIdx.x, r = blockIdx.y;
    if (i >= hidden) return;
    const float* x = streams + static_cast<std::size_t>(r) * stream_width;
    const float* p = coefficients + static_cast<std::size_t>(r) * stride;
    out[static_cast<std::size_t>(r) * hidden + i] =
        ((p[0] * x[i] + p[1] * x[hidden + i]) + p[2] * x[2U * hidden + i]) + p[3] * x[3U * hidden + i];
}
__global__ __launch_bounds__(256) void hc_post_kernel(const float* x, const float* residual, const float* mixes, float* out) {
    const unsigned i = blockIdx.x * 256U + threadIdx.x, r = blockIdx.y;
    if (i >= hidden) return;
    const float* m = mixes + static_cast<std::size_t>(r) * mix_count;
    const float* s = residual + static_cast<std::size_t>(r) * stream_width;
    const float* c = m + 2U * hc;
    const float xi = x[static_cast<std::size_t>(r) * hidden + i];
    for (unsigned k = 0; k < hc; ++k)
        out[static_cast<std::size_t>(r) * stream_width + k * hidden + i] =
            m[hc + k] * xi + (((c[0U * hc + k] * s[i] + c[1U * hc + k] * s[hidden + i]) + c[2U * hc + k] * s[2U * hidden + i]) +
                              c[3U * hc + k] * s[3U * hidden + i]);
}

__global__ __launch_bounds__(256) void rope_kernel(float* x, unsigned row_width, unsigned head_width, const float* table, bool inverse) {
    const unsigned r = blockIdx.x, pairs = row_width / head_width * (rope_dim / 2U);
    const float* t = table + static_cast<std::size_t>(r) * rope_dim;
    for (unsigned item = threadIdx.x; item < pairs; item += 256U) {
        const unsigned h = item / (rope_dim / 2U), i = item % (rope_dim / 2U);
        float* tail = x + static_cast<std::size_t>(r) * row_width + static_cast<std::size_t>(h) * head_width + (head_width - rope_dim);
        rope_pair(tail[2U * i], tail[2U * i + 1U], t[i], t[rope_dim / 2U + i], inverse);
    }
}

__global__ __launch_bounds__(32) void fp8_entries_kernel(const float* x, std::uint8_t* entries, unsigned capacity, unsigned long long first) {
    const unsigned lane = threadIdx.x, g = blockIdx.x, r = blockIdx.y;
    const float v = x[static_cast<std::size_t>(r) * head_dim + 32U * g + lane];
    float amax = fabsf(v);
    for (int d = 16; d != 0; d /= 2) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffU, amax, d, 32));
    const std::uint8_t code = fp8_scale_code(amax);
    const unsigned long long slot = capacity != 0U ? (first + r) % capacity : r;
    std::uint8_t* entry = entries + slot * entry_bytes;
    entry[32U * g + lane] = fp8_element(v, scale_value(code));
    if (lane == 0U) entry[head_dim + g] = code;
}

constexpr unsigned max_keys = window + 8U;
// The parameters of a draft step that change from step to step, read by the kernels from device memory (written by the
// step's own copy from pinned memory), so one captured graph serves every step.
struct DraftParams {
    unsigned long long oldest;  // ring position of the first key
    unsigned ring_keys;         // ring keys of the step (then the block rows)
    unsigned keys;              // ring_keys + block rows
    unsigned extra;             // extra vocabulary rows in use
    unsigned pad[3];
};
// Decodes the keys of a block attention once (ring positions ascending, then the block rows) into FP32 rows of 512.
// Grid max_keys: blocks past the step's key count exit.
__global__ __launch_bounds__(256) void decode_keys_kernel(const std::uint8_t* ring, const DraftParams* prm, const std::uint8_t* block, float* keys) {
    const unsigned j = blockIdx.x, ring_keys = prm->ring_keys;
    if (j >= prm->keys) return;
    const unsigned long long oldest = prm->oldest;
    const std::uint8_t* k = j < ring_keys ? ring + ((oldest + j) % window) * entry_bytes : block + static_cast<std::size_t>(j - ring_keys) * entry_bytes;
    for (unsigned d = threadIdx.x; d < head_dim; d += 256U) keys[static_cast<std::size_t>(j) * head_dim + d] = entry_value_device(k, d);
}
// Block (head, row): 8 warps score the keys (warp w takes keys w, w + 8, ...), then every thread forms two output
// dimensions in key order.
__global__ __launch_bounds__(256) void block_attention_kernel(const float* q, const float* key_values, const DraftParams* prm, const float* sinks,
                                                              float* out) {
    __shared__ float scores[max_keys];
    __shared__ float weights[max_keys];
    __shared__ float total;
    const unsigned keys = prm->keys;
    const unsigned h = blockIdx.x, r = blockIdx.y, lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    const float* qh = q + (static_cast<std::size_t>(r) * heads + h) * head_dim;
    for (unsigned j = wave; j < keys; j += 8U) {
        const float* k = key_values + static_cast<std::size_t>(j) * head_dim;
        float acc = 0.0F;
        for (unsigned d = lane; d < head_dim; d += 32U) acc = acc + qh[d] * k[d];
        acc = butterfly(acc);
        if (lane == 0U) scores[j] = acc * attention_scale;
    }
    __syncthreads();
    float m = scores[0];
    for (unsigned j = 1; j < keys; ++j) m = scores[j] > m ? scores[j] : m;
    for (unsigned j = threadIdx.x; j < keys; j += 256U) weights[j] = declared_exp(scores[j] - m);
    __syncthreads();
    if (threadIdx.x == 0U) {
        float sum = 0.0F;
        for (unsigned j = 0; j < keys; ++j) sum = sum + weights[j];
        total = sum + declared_exp(sinks[h] - m);
    }
    __syncthreads();
    for (unsigned d = threadIdx.x; d < head_dim; d += 256U) {
        float acc = 0.0F;
        for (unsigned j = 0; j < keys; ++j) acc = acc + weights[j] * key_values[static_cast<std::size_t>(j) * head_dim + d];
        out[(static_cast<std::size_t>(r) * heads + h) * head_dim + d] = acc / total;
    }
}

__global__ __launch_bounds__(256) void embed_block_kernel(const std::uint16_t* table, const std::uint32_t* tokens, float* streams) {
    const unsigned i = blockIdx.x * 256U + threadIdx.x, r = blockIdx.y;
    if (i >= hidden) return;
    const float v = f16_device(table[static_cast<std::size_t>(tokens[r]) * hidden + i]);
    for (unsigned k = 0; k < hc; ++k) streams[static_cast<std::size_t>(r) * stream_width + k * hidden + i] = v;
}

__global__ __launch_bounds__(32) void confidence_kernel(const float* weight, const float* x, const float* markov, float* out) {
    const unsigned lane = threadIdx.x, r = blockIdx.x;
    const float* xr = x + static_cast<std::size_t>(r) * hidden;
    const float* er = markov + static_cast<std::size_t>(r) * markov_rank;
    float acc = 0.0F;
    for (unsigned j = lane; j < confidence_width; j += 32U) acc = acc + weight[j] * (j < hidden ? xr[j] : er[j - hidden]);
    acc = butterfly(acc);
    if (lane == 0U) out[r] = acc;
}

// ---- MXFP4 routed experts (product_mxfp4: per 32-weight group float(dot of doubled E2M1 codes with the A8 codes)
// times (scale * delta), lane class l = groups l, l + 32, ... from 0 in ascending order, then the butterfly) ----
constexpr unsigned mx_block = 17;
constexpr unsigned max_entries = 8;  // entries of one expert per pass (window rows <= 8)
// Four doubled E2M1 codes from the low nibbles of the four bytes of w (byte k -> int8 k): the magnitudes
// {0, 1, 2, 3, 4, 6, 8, 12} by one byte permute, then the bytes with the sign bit negated (e2m1_doubled).
__device__ __forceinline__ int e2m1x4_lo(std::uint32_t w) {
    const std::uint32_t sel = (w & 7U) | ((w >> 4U) & 0x70U) | ((w >> 8U) & 0x700U) | ((w >> 12U) & 0x7000U);
    const std::uint32_t mag = __byte_perm(0x03020100U, 0x0C080604U, sel);
    const std::uint32_t mask = ((w >> 3U) & 0x01010101U) * 0xFFU;
    return static_cast<int>((mag & ~mask) | (__vneg4(mag) & mask));
}
// The integer dot of one MXFP4 block (17 bytes: E8M0 then 16 bytes of two nibbles, element j in the low nibble of
// byte j, element j + 16 in the high nibble) with 32 A8 codes (16-byte aligned); the block's scale through `scale`.
// Integer products are exact, so any order gives the declared int32 dot.
__device__ __forceinline__ int mxfp4_dot(const std::uint8_t* block, const std::int8_t* x, float& scale) {
    scale = codec::e8m0_half(block[0]);
    std::uint32_t q[4];
#pragma unroll
    for (unsigned k = 0; k < 4U; ++k)
        q[k] = static_cast<std::uint32_t>(block[1U + 4U * k]) | (static_cast<std::uint32_t>(block[2U + 4U * k]) << 8U) |
               (static_cast<std::uint32_t>(block[3U + 4U * k]) << 16U) | (static_cast<std::uint32_t>(block[4U + 4U * k]) << 24U);
    const int4 x0 = reinterpret_cast<const int4*>(x)[0], x1 = reinterpret_cast<const int4*>(x)[1];
    int dot = 0;
    dot = __dp4a(e2m1x4_lo(q[0]), x0.x, dot);
    dot = __dp4a(e2m1x4_lo(q[1]), x0.y, dot);
    dot = __dp4a(e2m1x4_lo(q[2]), x0.z, dot);
    dot = __dp4a(e2m1x4_lo(q[3]), x0.w, dot);
    dot = __dp4a(e2m1x4_lo(q[0] >> 4U), x1.x, dot);
    dot = __dp4a(e2m1x4_lo(q[1] >> 4U), x1.y, dot);
    dot = __dp4a(e2m1x4_lo(q[2] >> 4U), x1.z, dot);
    dot = __dp4a(e2m1x4_lo(q[3] >> 4U), x1.w, dot);
    return dot;
}

// gate_out / up_out [slot][inter] of every routed entry. Grid (inter / 8, union slots), one warp per output neuron;
// empty union slots exit (routes padded to rows * top_k).
// Routed-expert usage of a draft step (diagnostics): counts[ids[i]] += 1 for every (row, k) slot.
__global__ void route_count_kernel(const std::uint32_t* __restrict__ ids, std::uint32_t n, std::uint32_t* __restrict__ counts) {
    const std::uint32_t i = threadIdx.x;
    if (i < n) atomicAdd(counts + ids[i], 1U);
}

__global__ __launch_bounds__(256) void mxfp4_gate_up_kernel(const std::uint8_t* gate, const std::uint8_t* up, unsigned long long expert_bytes,
                                                            unsigned long long row_bytes, unsigned width, unsigned inter, MoeRoutes routes,
                                                            const std::int8_t* codes, const float* deltas, unsigned x_stride,
                                                            float* gate_out, float* up_out, const unsigned* remap) {
    const unsigned j = blockIdx.y;
    if (j >= routes.counts[0]) return;
    const unsigned lane = threadIdx.x % 32U, o = blockIdx.x * 8U + threadIdx.x / 32U;
    if (o >= inter) return;
    const unsigned begin = routes.begins[j], end = routes.begins[j + 1U];
    const unsigned e = remap ? remap[routes.experts[j]] : routes.experts[j], groups = width / 32U;
    const std::uint8_t* grow = gate + e * expert_bytes + static_cast<unsigned long long>(o) * row_bytes;
    const std::uint8_t* urow = up + e * expert_bytes + static_cast<unsigned long long>(o) * row_bytes;
    for (unsigned i0 = begin; i0 < end; i0 += max_entries) {
        const unsigned n = min(max_entries, end - i0);
        float ga[max_entries], ua[max_entries];
#pragma unroll
        for (unsigned k = 0; k < max_entries; ++k) { ga[k] = 0.0F; ua[k] = 0.0F; }
        for (unsigned g = lane; g < groups; g += 32U) {
#pragma unroll
            for (unsigned k = 0; k < max_entries; ++k) {
                if (k >= n) break;
                const unsigned row = routes.rows[i0 + k];
                const std::int8_t* x = codes + static_cast<std::size_t>(row) * x_stride + 32U * g;
                const float delta = deltas[static_cast<std::size_t>(row) * (x_stride / 32U) + g];
                float sg, su;
                const int dg = mxfp4_dot(grow + mx_block * g, x, sg);
                const int du = mxfp4_dot(urow + mx_block * g, x, su);
                ga[k] = ga[k] + static_cast<float>(dg) * (sg * delta);
                ua[k] = ua[k] + static_cast<float>(du) * (su * delta);
            }
        }
#pragma unroll
        for (unsigned k = 0; k < max_entries; ++k) {
            if (k >= n) break;
            const float gv = butterfly(ga[k]), uv = butterfly(ua[k]);
            if (lane == 0U) {
                const unsigned slot = routes.slots[i0 + k];
                gate_out[static_cast<std::size_t>(slot) * inter + o] = gv;
                up_out[static_cast<std::size_t>(slot) * inter + o] = uv;
            }
        }
    }
}
// raw[slot][hidden] = down rows . act[slot] (unweighted).
__global__ __launch_bounds__(256) void mxfp4_down_kernel(const std::uint8_t* down, unsigned long long expert_bytes, unsigned long long row_bytes,
                                                         unsigned width, unsigned outputs, MoeRoutes routes, const std::int8_t* codes,
                                                         const float* deltas, float* raw, const unsigned* remap) {
    const unsigned j = blockIdx.y;
    if (j >= routes.counts[0]) return;
    const unsigned lane = threadIdx.x % 32U, o = blockIdx.x * 8U + threadIdx.x / 32U;
    if (o >= outputs) return;
    const unsigned begin = routes.begins[j], end = routes.begins[j + 1U];
    const unsigned e = remap ? remap[routes.experts[j]] : routes.experts[j], groups = width / 32U;
    const std::uint8_t* drow = down + e * expert_bytes + static_cast<unsigned long long>(o) * row_bytes;
    for (unsigned i0 = begin; i0 < end; i0 += max_entries) {
        const unsigned n = min(max_entries, end - i0);
        float acc[max_entries];
#pragma unroll
        for (unsigned k = 0; k < max_entries; ++k) acc[k] = 0.0F;
        for (unsigned g = lane; g < groups; g += 32U) {
#pragma unroll
            for (unsigned k = 0; k < max_entries; ++k) {
                if (k >= n) break;
                const unsigned slot = routes.slots[i0 + k];
                const std::int8_t* x = codes + static_cast<std::size_t>(slot) * width + 32U * g;
                float s;
                const int d = mxfp4_dot(drow + mx_block * g, x, s);
                acc[k] = acc[k] + static_cast<float>(d) * (s * deltas[static_cast<std::size_t>(slot) * groups + g]);
            }
        }
#pragma unroll
        for (unsigned k = 0; k < max_entries; ++k) {
            if (k >= n) break;
            const float v = butterfly(acc[k]);
            if (lane == 0U) raw[static_cast<std::size_t>(routes.slots[i0 + k]) * outputs + o] = v;
        }
    }
}

// ---- head: the Markov chain, one pass over the Markov head per step fused with a partial argmax ----
struct HeadView {
    const float* logits_core;
    unsigned core;
    unsigned core_base;
    const float* logits_extra;
    unsigned extra_stride;          // row stride of logits_extra (the extra capacity)
    const DraftParams* prm;         // prm->extra: extra rows in use
    unsigned extra_base;
    const std::uint32_t* extra_ids;
    const std::uint16_t* embed;
    const std::uint16_t* head;
    std::uint32_t* tokens;
    float* markov;
    float* best_values;
    std::uint32_t* best_ids;
};
constexpr unsigned markov_blocks = 512;
__device__ __forceinline__ float head_logit(const HeadView& v, unsigned step, unsigned j) {
    return j < v.core ? v.logits_core[static_cast<unsigned long long>(step) * v.core + j]
                      : v.logits_extra[static_cast<unsigned long long>(step) * v.extra_stride + (j - v.core)];
}
__device__ __forceinline__ unsigned head_id(const HeadView& v, unsigned j) {
    return j < v.core ? v.core_base + j : v.extra_ids[v.extra_base + (j - v.core)];
}
__device__ __forceinline__ void unpack8(const uint4 q, float (&out)[8]) {
    const unsigned w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (unsigned k = 0; k < 4U; ++k) {
        out[2U * k] = f16_device(static_cast<std::uint16_t>(w[k] & 0xffffU));
        out[2U * k + 1U] = f16_device(static_cast<std::uint16_t>(w[k] >> 16U));
    }
}
__global__ __launch_bounds__(256) void markov_pass_kernel(HeadView v, unsigned step) {
    __shared__ float values[8];
    __shared__ unsigned ids[8];
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    const std::uint16_t* erow = v.embed + static_cast<unsigned long long>(v.tokens[step]) * markov_rank;
    float e[8];
    unpack8(reinterpret_cast<const uint4*>(erow)[lane], e);
    if (blockIdx.x == 0U && wave == 0U)
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k) v.markov[static_cast<unsigned long long>(step) * markov_rank + 8U * lane + k] = e[k];
    float best = -__builtin_huge_valf();
    unsigned best_id = vocab;
    const unsigned size = v.core + v.prm->extra, waves = gridDim.x * 8U;
    for (unsigned j = blockIdx.x * 8U + wave; j < size; j += waves) {
        const unsigned id = head_id(v, j);
        float h[8];
        unpack8(reinterpret_cast<const uint4*>(v.head + static_cast<unsigned long long>(id) * markov_rank)[lane], h);
        float acc = 0.0F;
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k) acc = acc + h[k] * e[k];
        acc = butterfly(acc);
        const float value = head_logit(v, step, j) + acc;
        if (better(value, id, best, best_id)) { best = value; best_id = id; }
    }
    if (lane == 0U) { values[wave] = best; ids[wave] = best_id; }
    __syncthreads();
    if (threadIdx.x == 0U) {
        for (unsigned w = 1; w < 8U; ++w)
            if (better(values[w], ids[w], best, best_id)) { best = values[w]; best_id = ids[w]; }
        v.best_values[blockIdx.x] = best;
        v.best_ids[blockIdx.x] = best_id;
    }
}
__global__ __launch_bounds__(markov_blocks) void markov_pick_kernel(HeadView v, unsigned step) {
    __shared__ float values[markov_blocks];
    __shared__ unsigned ids[markov_blocks];
    const unsigned t = threadIdx.x;
    values[t] = v.best_values[t];
    ids[t] = v.best_ids[t];
    __syncthreads();
    for (unsigned h = markov_blocks / 2U; h != 0U; h /= 2U) {
        if (t < h && better(values[t + h], ids[t + h], values[t], ids[t])) { values[t] = values[t + h]; ids[t] = ids[t + h]; }
        __syncthreads();
    }
    if (t == 0U) v.tokens[step + 1U] = ids[0];
}
// Gathers Q8_0 head rows of extra vocabulary ids into a packed table (rows first .. first + count).
__global__ __launch_bounds__(256) void gather_rows_kernel(const std::uint8_t* table, unsigned long long row_bytes, const std::uint32_t* ids,
                                                          unsigned first, std::uint8_t* out) {
    const unsigned r = first + blockIdx.y;
    const std::uint8_t* src = table + static_cast<unsigned long long>(ids[r]) * row_bytes;
    std::uint8_t* dst = out + static_cast<unsigned long long>(r) * row_bytes;
    for (unsigned long long i = blockIdx.x * 256ULL + threadIdx.x; i < row_bytes; i += 256ULL * gridDim.x) dst[i] = src[i];
}

// The same Q8_0 rows from the tiled layout (Q8TiledBuffer::repack's inverse): block (group g, row r), lane k = code k.
// Row 16 t + m, group g: scale scales[(t G + g) 16 + m]; code k in lane 4 (m % 8) + k / 8, word (m >= 8) + 2 ((k % 8) / 4),
// byte k % 4 of the lane's 16 bytes.
__global__ __launch_bounds__(32) void gather_tiled_rows_kernel(Q8Tiled tiled, const std::uint32_t* ids, unsigned first, std::uint8_t* out) {
    const unsigned r = first + blockIdx.y, g = blockIdx.x, k = threadIdx.x, groups = tiled.width / 32U;
    const unsigned row = ids[r], t = row / 16U, m = row % 16U;
    const unsigned long long tg = static_cast<unsigned long long>(t) * groups + g;
    std::uint8_t* dst = out + static_cast<unsigned long long>(r) * groups * 34ULL + g * 34ULL;
    const unsigned lane = 4U * (m % 8U) + k / 8U, word = (m >= 8U ? 1U : 0U) + 2U * ((k % 8U) / 4U);
    dst[2U + k] = tiled.codes[(tg * 32ULL + lane) * 16ULL + word * 4ULL + (k % 4U)];
    if (k < 2U) {
        const std::uint16_t sc = tiled.scales[tg * 16ULL + m];
        dst[k] = static_cast<std::uint8_t>(k == 0U ? (sc & 0xffU) : (sc >> 8U));
    }
}

void launch_check(const char* what) { check(cudaGetLastError(), what, __FILE__, __LINE__); }

template <class T>
DeviceBuffer<T> upload_span(std::span<const T> s) {
    DeviceBuffer<T> b(s.size());
    if (!s.empty()) b.upload(s.data(), s.size());
    return b;
}

struct DeviceQ8 {
    DeviceBuffer<std::uint8_t> bytes;
    std::uint32_t width = 0, outputs = 0;
    DeviceQ8() = default;
    explicit DeviceQ8(const Q8Matrix& m) : bytes(upload_span(m.bytes)), width(m.width), outputs(m.outputs) {}
    [[nodiscard]] Q8Weights view() const noexcept { return {bytes.get(), outputs, width}; }
};
struct DeviceStage {
    DeviceBuffer<std::uint16_t> hc_attn_fn, hc_ffn_fn;
    DeviceBuffer<float> hc_attn_scale, hc_attn_base, hc_ffn_scale, hc_ffn_base, attn_norm, ffn_norm, sinks, q_norm, kv_norm;
    DeviceQ8 q_a, q_b, kv, out_a, out_b;
    DeviceBuffer<float> router, bias, bias_image;
    DeviceBuffer<std::uint8_t> routed_gate, routed_up, routed_down;
    std::uint64_t routed_gate_row = 0, routed_down_row = 0, routed_gate_expert = 0, routed_down_expert = 0;
    DeviceBuffer<std::uint32_t> remap;  // pruned stage: expert id -> compact index (empty: identity)
    DeviceBuffer<std::uint8_t> shared_gate, shared_up, shared_down;
    float limit = swiglu_limit;
    DeviceBuffer<std::uint8_t> ring;
};

}  // namespace

struct CudaDrafter::Impl {
    std::uint32_t max_rows = 128;
    Stream stream;
    cudaStream_t st = nullptr;
    DeviceQ8 main_proj, output_own;
    const std::uint8_t* output = nullptr;       // raw Q8_0 rows (null with a tiled head)
    std::uint64_t output_row_bytes = 0;
    Q8Tiled output_tiled{};                     // the head in the tiled layout (the target's, or the own copy tiled)
    Q8TiledBuffer output_tiled_own;
    DeviceBuffer<float> main_norm, final_norm, confidence;
    DeviceBuffer<std::uint16_t> markov_embed, markov_head, token_embd_own;
    const std::uint16_t* token_embd = nullptr;
    DeviceStage stage[stages];
    std::uint64_t next = 0, start = 0;  // ring positions [max(start, next - 128), next) hold appended rows
    // A8 scratch (the largest: max_rows x main_width).
    A8Buffer a8;
    // Append scratch.
    DeviceBuffer<float> x_main, proj, main_x, kv, kv_normed, rope_table;
    // Draft step buffers (5 rows).
    DeviceBuffer<std::uint32_t> tokens{block_size + 1U};
    DeviceBuffer<float> streams{static_cast<std::size_t>(block_size) * stream_width}, streams2{static_cast<std::size_t>(block_size) * stream_width};
    DeviceBuffer<float> attn_mix{block_size * mix_count}, ffn_mix{block_size * mix_count}, pre_mix{block_size * mix_count},
        identity{block_size * mix_count};
    DeviceBuffer<float> mix_scratch{block_size * (mix_count + 1U)};
    DeviceBuffer<float> x{static_cast<std::size_t>(block_size) * hidden}, xn{static_cast<std::size_t>(block_size) * hidden};
    DeviceBuffer<float> qa{static_cast<std::size_t>(block_size) * q_rank}, qan{static_cast<std::size_t>(block_size) * q_rank};
    DeviceBuffer<float> q{static_cast<std::size_t>(block_size) * q_width}, o{static_cast<std::size_t>(block_size) * q_width};
    DeviceBuffer<float> low{static_cast<std::size_t>(block_size) * o_width}, y{static_cast<std::size_t>(block_size) * hidden};
    DeviceBuffer<std::uint8_t> block_entries{static_cast<std::size_t>(block_size) * entry_bytes};
    DeviceBuffer<float> key_scratch{static_cast<std::size_t>(max_keys) * head_dim};
    // MoE.
    DeviceBuffer<float> scores{static_cast<std::size_t>(block_size) * experts}, route_weights{block_size * top_k};
    DeviceBuffer<std::uint32_t> ids{block_size * top_k};
    DeviceBuffer<std::uint32_t> r_experts{block_size * top_k}, r_begins{block_size * top_k + 1U}, r_rows{block_size * top_k},
        r_slots{block_size * top_k}, r_order{block_size * top_k}, r_low{block_size}, r_counts{2};
    DeviceBuffer<float> gate_out{static_cast<std::size_t>(block_size) * top_k * inter}, up_out{static_cast<std::size_t>(block_size) * top_k * inter};
    A8Buffer act;
    DeviceBuffer<float> raw{static_cast<std::size_t>(block_size) * top_k * hidden}, shared_out{static_cast<std::size_t>(block_size) * hidden};
    DeviceBuffer<std::uint8_t> shared_ws;
    // Head and draft vocabulary.
    std::uint32_t extra_capacity = 8192;
    std::uint32_t core = vocab;
    std::vector<std::uint32_t> extra_host;
    std::vector<std::uint8_t> member = std::vector<std::uint8_t>(vocab, 0U);
    DeviceBuffer<std::uint32_t> extra_ids{8192};
    DeviceBuffer<std::uint8_t> extra_rows;
    DeviceBuffer<float> logits{static_cast<std::size_t>(block_size) * vocab}, logits_extra{static_cast<std::size_t>(block_size) * 8192};
    DeviceBuffer<float> markov{static_cast<std::size_t>(block_size) * markov_rank};
    DeviceBuffer<float> best_values{markov_blocks};
    DeviceBuffer<std::uint32_t> best_ids{markov_blocks};
    DeviceBuffer<float> confidence_out{block_size};
    // Pinned staging (rope rows, tokens, outputs) and the event that guards its reuse.
    float* rope_host = nullptr;
    std::uint32_t* tokens_host = nullptr;
    float* conf_host = nullptr;
    std::uint32_t* ids_stage = nullptr;
    std::uint32_t ids_stage_capacity = 0;
    cudaEvent_t staged{}, ev_start{}, ev_end{}, producer_done{};
    std::size_t device_bytes = 0;
    DeviceBuffer<DraftParams> params_dev{1};
    DraftParams* params_host = nullptr;
    // The draft step as one CUDA graph (STRIX_DS41_CUDA_DSPARK_GRAPH=0: eager launches), captured again when the
    // vocabulary geometry changes (core rows, extra capacity, extras present).
    bool use_graph = true;
    cudaGraphExec_t graph = nullptr;
    std::uint32_t graph_core = 0, graph_capacity = 0;
    bool graph_extras = false, graph_stats = false;
    bool route_stats = false;
    DeviceBuffer<std::uint32_t> route_counts;  // [stage][expert] when route_stats

    Impl(const Weights& w, const CudaDrafterOptions& opt) : max_rows(opt.max_append_rows), st(stream.get()) {
        const std::size_t before = device_buffer_bytes().load();
        main_proj = DeviceQ8(w.main_proj);
        if (opt.output_tiled.codes) {
            if (opt.output_tiled.rows != vocab || opt.output_tiled.width != hidden) throw std::invalid_argument("ds41 cuda dspark: tiled head geometry");
            output_tiled = opt.output_tiled;
        } else if (opt.output) {
            output = opt.output;
        } else {
            output_own = DeviceQ8(w.output);
            output = output_own.bytes.get();
            if (opt.tile_head) {
                output_tiled_own.repack(output, vocab, hidden, st);
                STRIX_CUDA_CHECK(cudaStreamSynchronize(st));
                output_own = {};  // only the tiled copy stays
                output = nullptr;
                output_tiled = output_tiled_own.view();
            }
        }
        output_row_bytes = w.output.row_bytes();
        main_norm = upload_span(w.main_norm);
        final_norm = upload_span(w.final_norm);
        confidence = upload_span(w.confidence);
        markov_embed = upload_span(w.markov_embed.bits);
        markov_head = upload_span(w.markov_head.bits);
        if (opt.token_embd) {
            token_embd = opt.token_embd;
        } else {
            token_embd_own = upload_span(w.token_embd.bits);
            token_embd = token_embd_own.get();
        }
        for (std::uint32_t s = 0; s < stages; ++s) {
            const auto& h = w.stage[s];
            auto& d = stage[s];
            if (h.routed.gate_up_format != WeightFormat::mxfp4 || h.routed.down_format != WeightFormat::mxfp4 ||
                h.shared.gate_up_format != WeightFormat::q8_0 || h.shared.down_format != WeightFormat::q8_0 || h.routed.hidden != hidden ||
                h.routed.inter != inter)
                throw std::invalid_argument("ds41 cuda dspark: expected MXFP4 routed and Q8_0 shared experts");
            d.hc_attn_fn = upload_span(h.hc_attn_fn.bits); d.hc_ffn_fn = upload_span(h.hc_ffn_fn.bits);
            d.hc_attn_scale = upload_span(h.hc_attn_scale); d.hc_attn_base = upload_span(h.hc_attn_base);
            d.hc_ffn_scale = upload_span(h.hc_ffn_scale); d.hc_ffn_base = upload_span(h.hc_ffn_base);
            d.attn_norm = upload_span(h.attn_norm); d.ffn_norm = upload_span(h.ffn_norm); d.sinks = upload_span(h.sinks);
            d.q_norm = upload_span(h.q_norm); d.kv_norm = upload_span(h.kv_norm);
            d.q_a = DeviceQ8(h.q_a); d.q_b = DeviceQ8(h.q_b); d.kv = DeviceQ8(h.kv); d.out_a = DeviceQ8(h.out_a); d.out_b = DeviceQ8(h.out_b);
            d.router = upload_span(h.router);
            d.routed_gate_row = h.routed.gate_row_bytes(); d.routed_down_row = h.routed.down_row_bytes();
            d.routed_gate_expert = d.routed_gate_row * inter; d.routed_down_expert = d.routed_down_row * hidden;
            if (s >= opt.keep.size() || opt.keep[s].empty()) {
                d.bias = upload_span(h.bias); d.bias_image = upload_span(h.bias_image);
                d.routed_gate = upload_span(h.routed.gate); d.routed_up = upload_span(h.routed.up); d.routed_down = upload_span(h.routed.down);
            } else {  // pruned: compact the kept experts, a -1e30 selection bias for the others
                const auto& keep = opt.keep[s];
                std::vector<std::uint32_t> remap(experts, 0U);
                std::vector<std::uint8_t> kept(experts, 0U);
                if (keep.size() < top_k || keep.size() > experts) throw std::invalid_argument("ds41 cuda dspark: keep top_k..128 experts per stage");
                for (std::uint32_t i = 0; i < keep.size(); ++i) {
                    if (keep[i] >= experts || kept[keep[i]]) throw std::invalid_argument("ds41 cuda dspark: keep ids distinct and < 128");
                    kept[keep[i]] = 1U;
                    remap[keep[i]] = i;
                }
                const auto compact = [&](std::span<const std::uint8_t> all, std::uint64_t per) {
                    if (all.size() != per * experts) throw std::invalid_argument("ds41 cuda dspark: routed tensor size");
                    std::vector<std::uint8_t> v(per * keep.size());
                    for (std::uint32_t i = 0; i < keep.size(); ++i) std::memcpy(v.data() + i * per, all.data() + keep[i] * per, per);
                    return upload_span(std::span<const std::uint8_t>(v));
                };
                d.routed_gate = compact(h.routed.gate, d.routed_gate_expert);
                d.routed_up = compact(h.routed.up, d.routed_gate_expert);
                d.routed_down = compact(h.routed.down, d.routed_down_expert);
                const auto masked = [&](std::span<const float> b) {
                    std::vector<float> v(b.begin(), b.end());
                    for (std::size_t e = 0; e < v.size() && e < experts; ++e)
                        if (!kept[e]) v[e] = -1.0e30F;  // never in the top_k (finite: no inf arithmetic in the router)
                    return upload_span(std::span<const float>(v));
                };
                d.bias = masked(h.bias); d.bias_image = masked(h.bias_image);
                d.remap = upload_span(std::span<const std::uint32_t>(remap));
            }
            d.shared_gate = upload_span(h.shared.gate); d.shared_up = upload_span(h.shared.up); d.shared_down = upload_span(h.shared.down);
            d.limit = h.routed.limit;
            d.ring = DeviceBuffer<std::uint8_t>(static_cast<std::size_t>(window) * entry_bytes);
            STRIX_CUDA_CHECK(cudaMemsetAsync(d.ring.get(), 0, d.ring.size(), st));
        }
        a8.resize(std::max(max_rows, block_size), std::max(main_width, q_width));
        x_main.resize(static_cast<std::size_t>(max_rows) * main_width);
        proj.resize(static_cast<std::size_t>(max_rows) * hidden);
        main_x.resize(static_cast<std::size_t>(max_rows) * hidden);
        kv.resize(static_cast<std::size_t>(max_rows) * head_dim);
        kv_normed.resize(static_cast<std::size_t>(max_rows) * head_dim);
        rope_table.resize(static_cast<std::size_t>(max_rows) * rope_dim);
        act.resize(block_size * top_k, inter);
        shared_ws.resize(shared_expert_workspace_bytes(block_size, inter));
        extra_rows.resize(static_cast<std::size_t>(extra_capacity) * output_row_bytes);
        std::vector<float> ident(block_size * mix_count, 0.0F);
        for (std::uint32_t r = 0; r < block_size; ++r) ident[r * mix_count] = 1.0F;
        identity.upload(ident.data(), ident.size());
        STRIX_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&rope_host), static_cast<std::size_t>(max_rows) * rope_dim * sizeof(float)));
        STRIX_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&tokens_host), (block_size + 1U) * sizeof(std::uint32_t)));
        STRIX_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&conf_host), block_size * sizeof(float)));
        STRIX_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&params_host), sizeof(DraftParams)));
        *params_host = DraftParams{};
        if (const char* g = std::getenv("STRIX_DS41_CUDA_DSPARK_GRAPH"); g && g[0] == '0') use_graph = false;
        STRIX_CUDA_CHECK(cudaEventCreateWithFlags(&staged, cudaEventDisableTiming));
        STRIX_CUDA_CHECK(cudaEventCreateWithFlags(&producer_done, cudaEventDisableTiming));
        STRIX_CUDA_CHECK(cudaEventCreate(&ev_start));
        STRIX_CUDA_CHECK(cudaEventCreate(&ev_end));
        STRIX_CUDA_CHECK(cudaEventRecord(staged, st));
        STRIX_CUDA_CHECK(cudaStreamSynchronize(st));
        device_bytes = device_buffer_bytes().load() - before;
    }
    ~Impl() {
        (void)cudaStreamSynchronize(st);
        if (rope_host) cudaFreeHost(rope_host);
        if (tokens_host) cudaFreeHost(tokens_host);
        if (conf_host) cudaFreeHost(conf_host);
        if (params_host) cudaFreeHost(params_host);
        if (graph) cudaGraphExecDestroy(graph);
        if (ids_stage) cudaFreeHost(ids_stage);
        cudaEventDestroy(staged);
        cudaEventDestroy(producer_done);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_end);
    }

    // The pinned staging buffers are rewritten only after the copies of the previous use completed.
    void wait_staging() { STRIX_CUDA_CHECK(cudaEventSynchronize(staged)); }
    void staged_mark() { STRIX_CUDA_CHECK(cudaEventRecord(staged, st)); }

    A8Rows quantize(const float* xin, std::uint32_t rows, std::uint32_t width) {
        A8Rows out = a8.view();
        out.rows = rows;
        out.width = width;
        out.stride = width;
        QuantizeA8View v;
        v.x = xin; v.rows = rows; v.width = width; v.out = out;
        enqueue_quantize_a8(v, st);
        return out;
    }
    // y[rows][outputs] = Q8_0 linear of x (A8 per row).
    void q8_linear(const Q8Weights& w, const float* xin, std::uint32_t rows, float* out) {
        Q8A8View v;
        v.weight = w;
        v.count = w.rows;
        v.x = quantize(xin, rows, w.width);
        v.out = out;
        enqueue_q8_0_a8(v, st);
    }
    void upload_rope(std::uint64_t first, std::uint32_t rows) {
        wait_staging();
        for (std::uint32_t r = 0; r < rows; ++r) {
            const auto row = rope_row(first + r);
            std::copy(row.c.begin(), row.c.end(), rope_host + static_cast<std::size_t>(r) * rope_dim);
            std::copy(row.s.begin(), row.s.end(), rope_host + static_cast<std::size_t>(r) * rope_dim + rope_dim / 2U);
        }
        rope_table.upload_async(rope_host, static_cast<std::size_t>(rows) * rope_dim, st);
        staged_mark();
    }
    void rms(const float* xin, std::uint32_t rows, std::uint32_t width, const float* weight, float* out) {
        const unsigned waves = width <= sum_chunk ? 1U : width / sum_chunk;
        rms_norm_kernel<<<rows, 32U * waves, 0, st>>>(xin, width, weight, out);
        launch_check("dspark rms norm");
    }
    void hc_mixes(const float* s, std::uint32_t rows, const std::uint16_t* fn, const float* scale, const float* base, float* mixes) {
        hc_dot_kernel<<<dim3(mix_count + 1U, rows), 32U * (stream_width / sum_chunk), 0, st>>>(s, fn, mix_scratch.get());
        sinkhorn_kernel<<<1, rows, 0, st>>>(mix_scratch.get(), scale, base, rows, mixes);
        launch_check("dspark hc mixes");
    }
    void hc_pre(const float* s, std::uint32_t rows, const float* coefficients, float* out) {
        hc_pre_kernel<<<dim3((hidden + 255U) / 256U, rows), 256, 0, st>>>(s, coefficients, mix_count, out);
        launch_check("dspark hc pre");
    }
    void hc_post(const float* xin, const float* residual, std::uint32_t rows, const float* mixes, float* out) {
        hc_post_kernel<<<dim3((hidden + 255U) / 256U, rows), 256, 0, st>>>(xin, residual, mixes, out);
        launch_check("dspark hc post");
    }
    void rope(float* xin, std::uint32_t rows, std::uint32_t row_width, bool inverse) {
        rope_kernel<<<rows, 256, 0, st>>>(xin, row_width, head_dim, rope_table.get(), inverse);
        launch_check("dspark rope");
    }
    // kv entries of `rows` rows of xin (stage input) at positions first..: into the ring or `linear`.
    void kv_entries(DeviceStage& s, const float* xin, std::uint32_t rows, std::uint64_t first, std::uint8_t* linear) {
        q8_linear(s.kv.view(), xin, rows, kv.get());
        rms(kv.get(), rows, head_dim, s.kv_norm.get(), kv_normed.get());
        rope(kv_normed.get(), rows, head_dim, false);
        fp8_entries_kernel<<<dim3(head_dim / 32U, rows), 32, 0, st>>>(kv_normed.get(), linear ? linear : s.ring.get(), linear ? 0U : window,
                                                                      static_cast<unsigned long long>(linear ? 0U : first));
        launch_check("dspark fp8 entries");
    }
    void append_device(const float* mx, std::uint32_t rows) {
        upload_rope(next, rows);
        for (auto& s : stage) kv_entries(s, mx, rows, next, nullptr);
        next += rows;
    }
    void main_rows_device(const float* hidden_rows, std::uint32_t rows) {
        q8_linear(main_proj.view(), hidden_rows, rows, proj.get());
        rms(proj.get(), rows, hidden, main_norm.get(), main_x.get());
    }
    void moe(DeviceStage& s, const float* xin, float* out) {
        constexpr std::uint32_t rows = block_size;
        RouterView rv;
        rv.x = xin; rv.rows = rows; rv.weight = {s.router.get(), experts, hidden}; rv.bias = s.bias.get(); rv.bias_image = s.bias_image.get();
        rv.top_k = top_k; rv.route_scale = route_scale; rv.scores = scores.get(); rv.ids = ids.get(); rv.weights = route_weights.get();
        enqueue_router(rv, st);
        if (route_stats) {
            route_count_kernel<<<1, rows * top_k, 0, st>>>(ids.get(), rows * top_k, route_counts.get() + static_cast<std::size_t>(&s - stage) * experts);
            launch_check("dspark route counts");
        }
        const A8Rows xa = quantize(xin, rows, hidden);
        // Shared expert (Q8_0).
        SharedExpertView sv;
        sv.gate = {s.shared_gate.get(), inter, hidden}; sv.up = {s.shared_up.get(), inter, hidden}; sv.down = {s.shared_down.get(), hidden, inter};
        sv.limit = s.limit; sv.x = xa; sv.out = shared_out.get(); sv.workspace = shared_ws.get();
        enqueue_shared_expert(sv, st);
        // Routes of the window (union in ascending id, entries in ascending row; split = experts / 2).
        MoeRoutesView mr;
        mr.ids = ids.get(); mr.rows = rows; mr.top_k = top_k; mr.experts = experts; mr.split = experts / 2U;
        mr.out = MoeRoutes{r_experts.get(), r_begins.get(), r_rows.get(), r_slots.get(), r_order.get(), r_low.get(), r_counts.get()};
        enqueue_moe_routes(mr, st);
        mxfp4_gate_up_kernel<<<dim3(inter / 8U, rows * top_k), 256, 0, st>>>(
            s.routed_gate.get(), s.routed_up.get(), s.routed_gate_expert, s.routed_gate_row, hidden, inter, mr.out, xa.codes, xa.deltas, xa.stride,
            gate_out.get(), up_out.get(), s.remap.get());
        launch_check("dspark mxfp4 gate/up");
        SwigluA8View sw;
        sw.gate = gate_out.get(); sw.up = up_out.get(); sw.rows = rows * top_k; sw.inter = inter; sw.limit = s.limit; sw.a8 = act.view();
        enqueue_swiglu_a8(sw, st);
        mxfp4_down_kernel<<<dim3(hidden / 8U, rows * top_k), 256, 0, st>>>(s.routed_down.get(), s.routed_down_expert, s.routed_down_row, inter,
                                                                           hidden, mr.out, act.codes.get(), act.deltas.get(), raw.get(),
                                                                           s.remap.get());
        launch_check("dspark mxfp4 down");
        MoeCombineView c;
        c.raw = raw.get(); c.shared = shared_out.get(); c.weights = route_weights.get(); c.order = r_order.get(); c.low_counts = r_low.get();
        c.rows = rows; c.top_k = top_k; c.hidden = hidden; c.out = out;
        enqueue_moe_combine(c, st);
    }
    // Grouped output: per row and group g, A8 of o[g * 4096 ..] times rows g * 1024 .. of out_a, then out_b.
    void output_projection(DeviceStage& s, float* out) {
        constexpr std::uint32_t rows = block_size;
        Q8A8View v;
        v.weight = s.out_a.view();
        v.count = o_width;
        v.per_group = o_rank;
        v.width = o_group_width;
        A8Rows xa = quantize(o.get(), rows, q_width);  // groups of 32: the same codes as per-group rows of 4096
        xa.width = o_group_width;
        v.x = xa;
        v.out = low.get();
        enqueue_q8_0_a8(v, st);
        q8_linear(s.out_b.view(), low.get(), rows, out);
    }
    struct LocalRows { std::uint32_t core_count, extra_count; };
    [[nodiscard]] LocalRows local_rows() const { return {core, static_cast<std::uint32_t>(extra_host.size())}; }
    void reserve_extra(std::uint32_t needed) {
        if (needed <= extra_capacity) return;
        const std::uint32_t cap = needed + 16384U;
        DeviceBuffer<std::uint32_t> ids_new(cap);
        DeviceBuffer<std::uint8_t> rows_new(static_cast<std::size_t>(cap) * output_row_bytes);
        const auto have = static_cast<std::size_t>(extra_host.size());
        if (have) {
            STRIX_CUDA_CHECK(cudaMemcpyAsync(ids_new.get(), extra_ids.get(), have * 4U, cudaMemcpyDeviceToDevice, st));
            STRIX_CUDA_CHECK(cudaMemcpyAsync(rows_new.get(), extra_rows.get(), have * output_row_bytes, cudaMemcpyDeviceToDevice, st));
        }
        STRIX_CUDA_CHECK(cudaStreamSynchronize(st));
        extra_ids = std::move(ids_new);
        extra_rows = std::move(rows_new);
        logits_extra = DeviceBuffer<float>(static_cast<std::size_t>(block_size) * cap);
        extra_capacity = cap;
    }
    void push_extra(const std::vector<std::uint32_t>& fresh) {
        if (fresh.empty()) return;
        reserve_extra(static_cast<std::uint32_t>(extra_host.size() + fresh.size()));
        const auto first = static_cast<std::uint32_t>(extra_host.size());
        extra_host.insert(extra_host.end(), fresh.begin(), fresh.end());
        for (const auto id : fresh) member[id] = 1U;
        STRIX_CUDA_CHECK(cudaStreamSynchronize(st));  // the previous staged ids were copied
        if (fresh.size() > ids_stage_capacity) {
            if (ids_stage) STRIX_CUDA_CHECK(cudaFreeHost(ids_stage));
            ids_stage_capacity = static_cast<std::uint32_t>(fresh.size()) + 4096U;
            STRIX_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&ids_stage), ids_stage_capacity * sizeof(std::uint32_t)));
        }
        std::copy(fresh.begin(), fresh.end(), ids_stage);
        STRIX_CUDA_CHECK(cudaMemcpyAsync(extra_ids.get() + first, ids_stage, fresh.size() * 4U, cudaMemcpyHostToDevice, st));
        if (output_tiled.codes)
            gather_tiled_rows_kernel<<<dim3(output_tiled.width / 32U, static_cast<unsigned>(fresh.size())), 32, 0, st>>>(output_tiled, extra_ids.get(), first,
                                                                                                                 extra_rows.get());
        else
            gather_rows_kernel<<<dim3(8, static_cast<unsigned>(fresh.size())), 256, 0, st>>>(output, output_row_bytes, extra_ids.get(), first,
                                                                                            extra_rows.get());
        launch_check("dspark gather rows");
    }
    // Head logits of the vocabulary rows, then the Markov chain and the confidence.
    void head(std::uint32_t rows) {
        const A8Rows xa = quantize(xn.get(), rows, hidden);
        const auto [cc, ec] = local_rows();
        if (cc != 0U) {
            Q8A8View v;
            if (output_tiled.codes) v.tiled = output_tiled;
            else v.weight = {output, vocab, hidden};
            v.first = 0; v.count = cc; v.x = xa; v.out = logits.get();
            enqueue_q8_0_a8(v, st);
        }
        if (ec != 0U) {  // every row of the capacity (the rows past the extras in use are computed and never read)
            Q8A8View v;
            v.weight = {extra_rows.get(), extra_capacity, hidden};
            v.first = 0; v.count = extra_capacity; v.x = xa; v.out = logits_extra.get();
            enqueue_q8_0_a8(v, st);
        }
        HeadView h{logits.get(), cc, 0U, logits_extra.get(), extra_capacity, params_dev.get(), 0U, extra_ids.get(), markov_embed.get(),
                   markov_head.get(), tokens.get(), markov.get(), best_values.get(), best_ids.get()};
        for (std::uint32_t i = 0; i < rows; ++i) {
            markov_pass_kernel<<<markov_blocks, 256, 0, st>>>(h, i);
            markov_pick_kernel<<<1, markov_blocks, 0, st>>>(h, i);
        }
        launch_check("dspark markov");
        confidence_kernel<<<rows, 32, 0, st>>>(confidence.get(), x.get(), markov.get(), confidence_out.get());
        launch_check("dspark confidence");
    }
};

CudaDrafter::CudaDrafter(const Weights& weights, const CudaDrafterOptions& options) : impl_(std::make_unique<Impl>(weights, options)) {}
CudaDrafter::~CudaDrafter() = default;
std::uint64_t CudaDrafter::next() const noexcept { return impl_->next; }
std::size_t CudaDrafter::device_bytes() const noexcept { return impl_->device_bytes; }
cudaStream_t CudaDrafter::stream() const noexcept { return impl_->st; }

void CudaDrafter::set_vocabulary(const Vocab& vocabulary) {
    auto& m = *impl_;
    if (vocabulary.core > vocab || vocabulary.size() == 0U) throw std::invalid_argument("ds41 cuda dspark draft vocabulary");
    std::vector<std::uint8_t> seen(vocab, 0U);
    for (const auto id : vocabulary.extra) {
        if (id < vocabulary.core || id >= vocab || seen[id]) throw std::invalid_argument("ds41 cuda dspark draft vocabulary: bad extra id");
        seen[id] = 1U;
    }
    m.core = vocabulary.core;
    m.extra_host.clear();
    std::fill(m.member.begin(), m.member.end(), 0U);
    m.push_extra(vocabulary.extra);
}
Vocab CudaDrafter::vocabulary() const { return Vocab{impl_->core, impl_->extra_host}; }
std::size_t CudaDrafter::add_vocabulary(std::span<const std::uint32_t> ids) {
    auto& m = *impl_;
    std::vector<std::uint32_t> fresh;
    for (const auto id : ids)
        if (id < vocab && id >= m.core && !m.member[id]) {
            m.member[id] = 1U;
            fresh.push_back(id);
        }
    for (const auto id : fresh) m.member[id] = 0U;  // push_extra sets them again
    m.push_extra(fresh);
    return fresh.size();
}
void CudaDrafter::set_route_stats(bool on) {
    auto& m = *impl_;
    if (on && m.route_counts.size() == 0U) {
        m.route_counts.resize(std::size_t{stages} * experts);
        STRIX_CUDA_CHECK(cudaMemsetAsync(m.route_counts.get(), 0, m.route_counts.size() * sizeof(std::uint32_t), m.st));
    }
    m.route_stats = on;
}
std::vector<std::uint32_t> CudaDrafter::route_stats() const {
    STRIX_CUDA_CHECK(cudaStreamSynchronize(impl_->st));
    return impl_->route_counts.size() ? impl_->route_counts.download() : std::vector<std::uint32_t>{};
}
void CudaDrafter::clear_route_stats() {
    auto& m = *impl_;
    if (m.route_counts.size()) STRIX_CUDA_CHECK(cudaMemsetAsync(m.route_counts.get(), 0, m.route_counts.size() * sizeof(std::uint32_t), m.st));
}
void CudaDrafter::reset(std::uint64_t position) {
    STRIX_CUDA_CHECK(cudaStreamSynchronize(impl_->st));
    for (auto& s : impl_->stage) STRIX_CUDA_CHECK(cudaMemsetAsync(s.ring.get(), 0, s.ring.size(), impl_->st));
    STRIX_CUDA_CHECK(cudaStreamSynchronize(impl_->st));
    impl_->next = position;
    impl_->start = position;
}

CudaDrafterState CudaDrafter::export_state() const {
    auto& m=*impl_; STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
    CudaDrafterState s; s.start=m.start; s.next=m.next; s.vocabulary=vocabulary();
    for (const auto& stage:m.stage) s.rings.push_back(stage.ring.download());
    return s;
}
void CudaDrafter::import_state(const CudaDrafterState& s) {
    auto& m=*impl_;
    if(s.start>s.next || s.rings.size()!=stages || s.vocabulary.core>vocab || s.vocabulary.size()==0)
        throw std::invalid_argument("ds41 cuda drafter state geometry");
    std::vector<bool> seen(vocab);
    for(auto id:s.vocabulary.extra) {
        if(id<s.vocabulary.core || id>=vocab || seen[id]) throw std::invalid_argument("ds41 cuda drafter vocabulary");
        seen[id]=true;
    }
    for(unsigned i=0;i<stages;++i)
        if(s.rings[i].size()!=m.stage[i].ring.size()) throw std::invalid_argument("ds41 cuda drafter ring bytes");
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
    set_vocabulary(s.vocabulary);
    for(unsigned i=0;i<stages;++i)
        STRIX_CUDA_CHECK(cudaMemcpyAsync(m.stage[i].ring.get(),s.rings[i].data(),s.rings[i].size(),cudaMemcpyHostToDevice,m.st));
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
    m.start=s.start; m.next=s.next;
}

std::vector<std::uint8_t> CudaDrafter::ring(std::uint32_t s) const {
    STRIX_CUDA_CHECK(cudaStreamSynchronize(impl_->st));
    return impl_->stage[s].ring.download();
}

std::vector<float> CudaDrafter::main_rows(std::span<const float> main_hidden, std::uint32_t rows) {
    auto& m = *impl_;
    if (rows == 0U || rows > m.max_rows || main_hidden.size() != static_cast<std::size_t>(rows) * main_width)
        throw std::invalid_argument("ds41 cuda dspark main rows geometry");
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
    m.x_main.upload(main_hidden.data(), main_hidden.size());
    m.main_rows_device(m.x_main.get(), rows);
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
    return m.main_x.download(static_cast<std::size_t>(rows) * hidden);
}
void CudaDrafter::append(std::span<const float> main_x, std::uint32_t rows) {
    auto& m = *impl_;
    if (rows == 0U || rows > m.max_rows || main_x.size() != static_cast<std::size_t>(rows) * hidden)
        throw std::invalid_argument("ds41 cuda dspark append geometry");
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
    m.main_x.upload(main_x.data(), main_x.size());
    m.append_device(m.main_x.get(), rows);
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
}
void CudaDrafter::append_hidden(std::span<const float> main_hidden, std::uint32_t rows) {
    auto& m = *impl_;
    if (rows == 0U || rows > m.max_rows || main_hidden.size() != static_cast<std::size_t>(rows) * main_width)
        throw std::invalid_argument("ds41 cuda dspark append geometry");
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
    m.x_main.upload(main_hidden.data(), main_hidden.size());
    m.main_rows_device(m.x_main.get(), rows);
    m.append_device(m.main_x.get(), rows);
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
}
void CudaDrafter::append_hidden_device(const float* main_hidden, std::uint32_t rows, cudaStream_t producer) {
    auto& m = *impl_;
    if (rows == 0U || rows > m.max_rows || !main_hidden) throw std::invalid_argument("ds41 cuda dspark append geometry");
    STRIX_CUDA_CHECK(cudaEventRecord(m.producer_done, producer));
    STRIX_CUDA_CHECK(cudaStreamWaitEvent(m.st, m.producer_done, 0));
    m.main_rows_device(main_hidden, rows);
    m.append_device(m.main_x.get(), rows);
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));  // the input rows are consumed: the caller may overwrite them
}

Draft CudaDrafter::draft(std::uint32_t token, bool keep_logits, CudaDraftTimes* times) {
    using clock = std::chrono::steady_clock;
    auto& m = *impl_;
    constexpr std::uint32_t rows = block_size;
    if (token >= vocab) throw std::invalid_argument("ds41 cuda dspark draft token");
    const auto start = clock::now();
    const std::uint64_t first = m.next, oldest = std::max<std::uint64_t>(first > window ? first - window : 0U, m.start);
    const auto ring_keys = static_cast<std::uint32_t>(first - oldest);
    // host parameters of the step (pinned; read by the step's own copies)
    m.wait_staging();
    for (std::uint32_t r = 0; r < rows; ++r) {
        const auto row = rope_row(first + r);
        std::copy(row.c.begin(), row.c.end(), m.rope_host + static_cast<std::size_t>(r) * rope_dim);
        std::copy(row.s.begin(), row.s.end(), m.rope_host + static_cast<std::size_t>(r) * rope_dim + rope_dim / 2U);
    }
    m.tokens_host[0] = token;
    for (std::uint32_t r = 1; r < rows; ++r) m.tokens_host[r] = noise_token;
    m.tokens_host[rows] = 0U;
    m.params_host->oldest = oldest;
    m.params_host->ring_keys = ring_keys;
    m.params_host->keys = ring_keys + rows;
    m.params_host->extra = static_cast<unsigned>(m.extra_host.size());
    Draft out;
    const bool graphed = m.use_graph && !keep_logits;
    if (times) STRIX_CUDA_CHECK(cudaEventRecord(m.ev_start, m.st));
    if (graphed) {
        const bool extras = !m.extra_host.empty();
        if (!m.graph || m.graph_core != m.core || m.graph_capacity != m.extra_capacity || m.graph_extras != extras || m.graph_stats != m.route_stats) {
            if (m.graph) STRIX_CUDA_CHECK(cudaGraphExecDestroy(m.graph));
            m.graph = nullptr;
            cudaGraph_t g = nullptr;
            STRIX_CUDA_CHECK(cudaStreamBeginCapture(m.st, cudaStreamCaptureModeThreadLocal));
            enqueue_draft_step(m, false, out);
            STRIX_CUDA_CHECK(cudaStreamEndCapture(m.st, &g));
            STRIX_CUDA_CHECK(cudaGraphInstantiate(&m.graph, g, 0));
            STRIX_CUDA_CHECK(cudaGraphDestroy(g));
            m.graph_core = m.core;
            m.graph_capacity = m.extra_capacity;
            m.graph_extras = extras;
            m.graph_stats = m.route_stats;
        }
        STRIX_CUDA_CHECK(cudaGraphLaunch(m.graph, m.st));
    } else {
        enqueue_draft_step(m, keep_logits, out);
    }
    if (times) STRIX_CUDA_CHECK(cudaEventRecord(m.ev_end, m.st));
    m.staged_mark();
    STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
    std::copy(m.tokens_host, m.tokens_host + rows + 1U, out.tokens.begin());
    std::copy(m.conf_host, m.conf_host + rows, out.confidence.begin());
    const auto local = m.local_rows();
    out.candidates.fill(local.core_count + local.extra_count);
    if (keep_logits) {
        const auto cc = local.core_count, ec = local.extra_count, size = cc + ec;
        const auto core_logits = m.logits.download(static_cast<std::size_t>(rows) * cc);
        const auto extra_all = ec ? m.logits_extra.download(static_cast<std::size_t>(rows) * m.extra_capacity) : std::vector<float>{};
        out.logits.resize(static_cast<std::size_t>(rows) * size);
        for (std::uint32_t r = 0; r < rows; ++r) {
            std::copy_n(core_logits.begin() + static_cast<std::ptrdiff_t>(r) * cc, cc, out.logits.begin() + static_cast<std::ptrdiff_t>(r) * size);
            if (ec)
                std::copy_n(extra_all.begin() + static_cast<std::ptrdiff_t>(r) * m.extra_capacity, ec,
                            out.logits.begin() + static_cast<std::ptrdiff_t>(r) * size + cc);
        }
        out.head_in = m.x.download(static_cast<std::size_t>(rows) * hidden);
        out.markov = m.markov.download();
    }
    if (times) {
        STRIX_CUDA_CHECK(cudaEventElapsedTime(&times->device_ms, m.ev_start, m.ev_end));
        times->head_weight_bytes = (std::uint64_t(m.core) + (m.extra_host.empty() ? 0 : m.extra_capacity)) * m.output_row_bytes;
        times->wall_ms = std::chrono::duration<double, std::milli>(clock::now() - start).count();
    }
    return out;
}

// Everything a draft step enqueues on the drafter's stream (capturable: no host synchronisation; the host parameters
// come from the pinned buffers through the step's own copies). keep_logits (eager only) downloads every stage's streams.
void CudaDrafter::enqueue_draft_step(Impl& m, bool keep_logits, Draft& out) {
    constexpr std::uint32_t rows = block_size;
    m.rope_table.upload_async(m.rope_host, static_cast<std::size_t>(rows) * rope_dim, m.st);
    STRIX_CUDA_CHECK(cudaMemcpyAsync(m.params_dev.get(), m.params_host, sizeof(DraftParams), cudaMemcpyHostToDevice, m.st));
    m.tokens.upload_async(m.tokens_host, rows + 1U, m.st);
    embed_block_kernel<<<dim3((hidden + 255U) / 256U, rows), 256, 0, m.st>>>(m.token_embd, m.tokens.get(), m.streams.get());
    launch_check("dspark embed");
    const float* pre = m.identity.get();
    for (std::uint32_t si = 0; si < stages; ++si) {
        auto& s = m.stage[si];
        m.hc_mixes(m.streams.get(), rows, s.hc_attn_fn.get(), s.hc_attn_scale.get(), s.hc_attn_base.get(), m.attn_mix.get());
        m.hc_pre(m.streams.get(), rows, pre, m.x.get());
        m.rms(m.x.get(), rows, hidden, s.attn_norm.get(), m.xn.get());
        m.q8_linear(s.q_a.view(), m.xn.get(), rows, m.qa.get());
        m.rms(m.qa.get(), rows, q_rank, s.q_norm.get(), m.qan.get());
        m.q8_linear(s.q_b.view(), m.qan.get(), rows, m.q.get());
        m.rope(m.q.get(), rows, q_width, false);
        m.kv_entries(s, m.xn.get(), rows, 0U, m.block_entries.get());
        decode_keys_kernel<<<max_keys, 256, 0, m.st>>>(s.ring.get(), m.params_dev.get(), m.block_entries.get(), m.key_scratch.get());
        block_attention_kernel<<<dim3(heads, rows), 256, 0, m.st>>>(m.q.get(), m.key_scratch.get(), m.params_dev.get(), s.sinks.get(), m.o.get());
        launch_check("dspark block attention");
        m.rope(m.o.get(), rows, q_width, true);
        m.output_projection(s, m.y.get());
        m.hc_post(m.y.get(), m.streams.get(), rows, m.attn_mix.get(), m.streams2.get());
        m.hc_mixes(m.streams2.get(), rows, s.hc_ffn_fn.get(), s.hc_ffn_scale.get(), s.hc_ffn_base.get(), m.ffn_mix.get());
        m.hc_pre(m.streams2.get(), rows, m.attn_mix.get(), m.x.get());
        m.rms(m.x.get(), rows, hidden, s.ffn_norm.get(), m.xn.get());
        m.moe(s, m.xn.get(), m.y.get());
        m.hc_post(m.y.get(), m.streams2.get(), rows, m.ffn_mix.get(), m.streams.get());
        // The next stage's attention collapses with this FFN's pre coefficients.
        STRIX_CUDA_CHECK(cudaMemcpyAsync(m.pre_mix.get(), m.ffn_mix.get(), m.ffn_mix.size() * sizeof(float), cudaMemcpyDeviceToDevice, m.st));
        pre = m.pre_mix.get();
        if (keep_logits) {
            STRIX_CUDA_CHECK(cudaStreamSynchronize(m.st));
            out.stage_out[si] = m.streams.download();
        }
    }
    // Head: hc_pre with the last stage's FFN pre coefficients, final norm, target output head, Markov chain.
    m.hc_pre(m.streams.get(), rows, pre, m.x.get());
    m.rms(m.x.get(), rows, hidden, m.final_norm.get(), m.xn.get());
    m.head(rows);
    STRIX_CUDA_CHECK(cudaMemcpyAsync(m.tokens_host, m.tokens.get(), (rows + 1U) * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, m.st));
    STRIX_CUDA_CHECK(cudaMemcpyAsync(m.conf_host, m.confidence_out.get(), rows * sizeof(float), cudaMemcpyDeviceToHost, m.st));
}

}  // namespace strix::ds41::cuda
