// deepseek41 CUDA layer core: F16 projections (lane dot), embedding, Engram (gather, decode, projection, gated
// injection) and the greedy head selection. Bitwise the host reference of ds41_layer.hpp.

#include "ds41_cuda_common.cuh"

#include "strix/cuda/ds41_core.hpp"
#include "strix/ds41_scalar_codecs.hpp"

#include <cmath>

namespace strix::ds41::cuda {
namespace {

using detail::chunk;
using detail::full_mask;
using detail::require;

// ---- F16 rows ------------------------------------------------------------------------------------------------------
// One warp per output row. Segments of 256 weights are staged in shared memory by 16-byte loads; lane l then takes
// the segment's elements l, l + 32, ... so it keeps the lane class of the reference (elements l, l + 32, ... of the
// row, ascending, acc = acc + w * x from 0).
constexpr std::uint32_t kF16Warps = 8;

template <int R>
__global__ void __launch_bounds__(kF16Warps * 32)
f16_rows_kernel(const std::uint16_t* __restrict__ weight, std::uint32_t width, std::uint32_t first, std::uint32_t count,
                const float* __restrict__ x, std::uint32_t x_stride, float* __restrict__ out, std::uint32_t out_stride) {
    __shared__ __align__(16) std::uint16_t stage[kF16Warps][256];
    const std::uint32_t wib = threadIdx.x / 32U, lane = threadIdx.x & 31U, o = blockIdx.x * kF16Warps + wib;
    if (o >= count) return;
    const std::uint16_t* w = weight + std::size_t{first + o} * width;
    const bool aligned = (reinterpret_cast<std::uintptr_t>(w) & 15U) == 0;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0F;
    std::uint16_t* s = stage[wib];
    for (std::uint32_t s0 = 0; s0 < width; s0 += 256U) {
        const std::uint32_t len = width - s0 < 256U ? width - s0 : 256U;
        __syncwarp();
        if (aligned && len == 256U) reinterpret_cast<uint4*>(s)[lane] = __ldg(reinterpret_cast<const uint4*>(w + s0) + lane);
        else
            for (std::uint32_t i = lane; i < len; i += 32U) s[i] = w[s0 + i];
        __syncwarp();
        for (std::uint32_t i = lane; i < len; i += 32U) {
            const float wv = f16_to_float(s[i]);
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r] = acc[r] + wv * x[std::size_t(r) * x_stride + s0 + i];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float v = detail::butterfly(acc[r]);
        if (lane == 0) out[std::size_t(r) * out_stride + o] = v;
    }
}

template <int R>
void launch_f16(const F16RowsView& v, std::uint32_t x_stride, std::uint32_t out_stride, std::uint32_t r0, cudaStream_t stream) {
    f16_rows_kernel<R><<<(v.count + kF16Warps - 1U) / kF16Warps, kF16Warps * 32U, 0, stream>>>(
        v.weight.data, v.weight.width, v.first, v.count, v.x + std::size_t{r0} * x_stride, x_stride, v.out + std::size_t{r0} * out_stride,
        out_stride);
}

// ---- embedding -----------------------------------------------------------------------------------------------------
__global__ void embed_kernel(const std::uint16_t* __restrict__ table, std::uint32_t hidden, const std::uint32_t* __restrict__ tokens,
                             std::uint32_t rows, float* __restrict__ stream, float* __restrict__ pre_carry) {
    const std::size_t idx = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (idx >= std::size_t{rows} * hidden) return;
    const std::uint32_t r = static_cast<std::uint32_t>(idx / hidden), i = static_cast<std::uint32_t>(idx % hidden);
    const float v = f16_to_float(table[std::size_t{tokens[r]} * hidden + i]);
    for (int c = 0; c < 4; ++c) stream[(std::size_t{r} * 4U + c) * hidden + i] = v;
    if (pre_carry && i < 4U) pre_carry[std::size_t{r} * 4U + i] = i == 0 ? 1.0F : 0.0F;
}

// ---- Engram --------------------------------------------------------------------------------------------------------
__global__ void engram_gather_kernel(const std::uint8_t* __restrict__ table, const std::uint64_t* __restrict__ ids,
                                     std::uint32_t count, std::uint8_t* __restrict__ out) {
    const std::uint32_t k = blockIdx.x * (blockDim.x / 32U) + threadIdx.x / 32U, lane = threadIdx.x & 31U;
    if (k >= count) return;
    const auto* src = reinterpret_cast<const std::uint64_t*>(table + ids[k] * engram_row_bytes);
    auto* dst = reinterpret_cast<std::uint64_t*>(out + std::size_t{k} * engram_row_bytes);
    for (std::uint32_t i = lane; i < engram_row_bytes / 8U; i += 32U) dst[i] = src[i];
}

// emb[r][k * 256 + j] = round_to_bf16(e4m3(code j) * e8m0(scale j / 32)) of row (r, k).
__global__ void engram_decode_kernel(const std::uint8_t* __restrict__ rows, std::uint32_t n_rows, float* __restrict__ emb) {
    const std::size_t idx = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (idx >= std::size_t{n_rows} * engram_row_elements) return;
    const std::size_t row = idx / engram_row_elements, j = idx % engram_row_elements;
    const std::uint8_t* p = rows + row * engram_row_bytes;
    emb[idx] = round_to_bf16(codec::e4m3(p[j]) * codec::e8m0(p[256U + j / 32U]));
}

constexpr std::uint32_t kInjectThreads = 256, kMaxInjectChunks = 32;

// grid (rows, 4 copies): the three chunked dots of engram_gates for copy c (h.h, key.key, (h * (q * k)).key), the
// gate, then h = h + gate * value in place.
__global__ void __launch_bounds__(kInjectThreads)
engram_inject_kernel(float* stream, const float* __restrict__ kv, const float* __restrict__ qn, const float* __restrict__ kn,
                     std::uint32_t hidden, float eps, float inv_sqrt_hidden) {
    __shared__ float part[3][kMaxInjectChunks];
    __shared__ float gate_s;
    const std::uint32_t r = blockIdx.x, c = blockIdx.y, lane = threadIdx.x & 31U, warp = threadIdx.x / 32U, warps = blockDim.x / 32U;
    float* h = stream + (std::size_t{r} * 4U + c) * hidden;
    const float* key = kv + std::size_t{r} * 5U * hidden + std::size_t{c} * hidden;
    const float* value = kv + std::size_t{r} * 5U * hidden + 4U * std::size_t{hidden};
    const float* q = qn + std::size_t{c} * hidden;
    const float* k = kn + std::size_t{c} * hidden;
    const std::uint32_t chunks = (hidden + chunk - 1U) / chunk;
    for (std::uint32_t ch = warp; ch < chunks; ch += warps) {
        const std::uint32_t c0 = ch * chunk, len = hidden - c0 < chunk ? hidden - c0 : chunk;
        const float* hh = h + c0;
        const float* kk = key + c0;
        const float* qq = q + c0;
        const float* kq = k + c0;
        const float a = detail::lane_dot([&](std::uint32_t i) { return hh[i]; }, [&](std::uint32_t i) { return hh[i]; }, len, lane);
        const float b = detail::lane_dot([&](std::uint32_t i) { return kk[i]; }, [&](std::uint32_t i) { return kk[i]; }, len, lane);
        const float d = detail::lane_dot([&](std::uint32_t i) { return hh[i] * (qq[i] * kq[i]); }, [&](std::uint32_t i) { return kk[i]; },
                                         len, lane);
        if (lane == 0) { part[0][ch] = a; part[1][ch] = b; part[2][ch] = d; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float t[3] = {0.0F, 0.0F, 0.0F};
        for (int s = 0; s < 3; ++s)
            for (std::uint32_t ch = 0; ch < chunks; ++ch) t[s] = t[s] + part[s][ch];
        const float rh = declared_rsqrt(t[0] / static_cast<float>(hidden) + eps);
        const float rk = declared_rsqrt(t[1] / static_cast<float>(hidden) + eps);
        const float dot = (t[2] * (rh * rk)) * inv_sqrt_hidden;
        gate_s = engram_gate_of(dot);
    }
    __syncthreads();
    const float g = gate_s;
    for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) h[i] = h[i] + g * value[i];
}

std::size_t align256(std::size_t n) { return (n + 255U) / 256U * 256U; }

// ---- argmax --------------------------------------------------------------------------------------------------------
constexpr std::uint32_t kArgmaxThreads = 1024;

__device__ __forceinline__ void take_better(float& v, std::uint32_t& i, float ov, std::uint32_t oi) {
    if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
}

__global__ void __launch_bounds__(kArgmaxThreads)
argmax_kernel(const float* __restrict__ logits, std::uint32_t count, std::uint32_t stride, std::uint32_t first_index,
              ArgmaxCandidate* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ std::uint32_t si[32], sf[32];
    const std::uint32_t r = blockIdx.x, lane = threadIdx.x & 31U, warp = threadIdx.x / 32U;
    const float* l = logits + std::size_t{r} * stride;
    float v = -INFINITY;
    std::uint32_t idx = 0xffffffffU, finite = 1;
    for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        const float x = l[i];
        if (!isfinite(x)) { finite = 0; continue; }
        take_better(v, idx, x, first_index + i);
    }
    for (unsigned d = 16U; d; d >>= 1U) {
        const float ov = __shfl_xor_sync(full_mask, v, d);
        const std::uint32_t oi = __shfl_xor_sync(full_mask, idx, d);
        take_better(v, idx, ov, oi);
        finite &= __shfl_xor_sync(full_mask, finite, d);
    }
    if (lane == 0) { sv[warp] = v; si[warp] = idx; sf[warp] = finite; }
    __syncthreads();
    if (warp == 0) {
        const std::uint32_t warps = blockDim.x / 32U;
        v = lane < warps ? sv[lane] : -INFINITY;
        idx = lane < warps ? si[lane] : 0xffffffffU;
        finite = lane < warps ? sf[lane] : 1U;
        for (unsigned d = 16U; d; d >>= 1U) {
            const float ov = __shfl_xor_sync(full_mask, v, d);
            const std::uint32_t oi = __shfl_xor_sync(full_mask, idx, d);
            take_better(v, idx, ov, oi);
            finite &= __shfl_xor_sync(full_mask, finite, d);
        }
        if (lane == 0) out[r] = ArgmaxCandidate{v, idx, finite};
    }
}

}  // namespace

void enqueue_f16_rows(const F16RowsView& v, cudaStream_t stream) {
    const std::uint32_t x_stride = v.x_stride ? v.x_stride : v.weight.width, out_stride = v.out_stride ? v.out_stride : v.count;
    require(v.count > 0 && v.first + v.count <= v.weight.rows && v.weight.width > 0 && v.rows > 0, "F16 rows geometry");
    for (std::uint32_t r0 = 0; r0 < v.rows; r0 += 8U) {
        const std::uint32_t n = v.rows - r0 < 8U ? v.rows - r0 : 8U;
        switch (n) {
        case 1: launch_f16<1>(v, x_stride, out_stride, r0, stream); break;
        case 2: launch_f16<2>(v, x_stride, out_stride, r0, stream); break;
        case 3: launch_f16<3>(v, x_stride, out_stride, r0, stream); break;
        case 4: launch_f16<4>(v, x_stride, out_stride, r0, stream); break;
        case 5: launch_f16<5>(v, x_stride, out_stride, r0, stream); break;
        case 6: launch_f16<6>(v, x_stride, out_stride, r0, stream); break;
        case 7: launch_f16<7>(v, x_stride, out_stride, r0, stream); break;
        default: launch_f16<8>(v, x_stride, out_stride, r0, stream); break;
        }
        STRIX_CUDA_CHECK(cudaGetLastError());
    }
}

void enqueue_embed(const EmbedView& v, cudaStream_t stream) {
    require(v.rows > 0 && v.table.width >= 4U, "embedding geometry");
    const std::size_t n = std::size_t{v.rows} * v.table.width;
    embed_kernel<<<static_cast<unsigned>((n + 255U) / 256U), 256, 0, stream>>>(v.table.data, v.table.width, v.tokens, v.rows, v.stream,
                                                                               v.pre_carry);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_engram_gather(const std::uint8_t* table, const std::uint64_t* ids, std::uint32_t count, std::uint8_t* out,
                           cudaStream_t stream) {
    require(count > 0 && reinterpret_cast<std::uintptr_t>(table) % 8U == 0, "Engram gather geometry");
    engram_gather_kernel<<<(count + 7U) / 8U, 256, 0, stream>>>(table, ids, count, out);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

std::size_t engram_workspace_bytes(std::uint32_t rows, std::uint32_t cols, std::uint32_t hidden) {
    return align256(std::size_t{rows} * cols * engram_row_elements * 4U) + align256(std::size_t{rows} * 5U * hidden * 4U);
}

void enqueue_engram(const EngramView& v, cudaStream_t stream) {
    const std::uint32_t width = v.cols * engram_row_elements;
    require(v.rows > 0 && v.cols > 0 && v.hidden > 0 && (v.hidden + chunk - 1U) / chunk <= kMaxInjectChunks && v.kv.width == width &&
                v.kv.rows == 5U * v.hidden && v.workspace,
            "Engram geometry");
    auto* base = static_cast<std::uint8_t*>(v.workspace);
    float* emb = reinterpret_cast<float*>(base);
    float* kv = v.kv_out ? v.kv_out : reinterpret_cast<float*>(base + align256(std::size_t{v.rows} * width * 4U));
    const std::size_t n = std::size_t{v.rows} * width;
    engram_decode_kernel<<<static_cast<unsigned>((n + 255U) / 256U), 256, 0, stream>>>(v.row_bytes, v.rows * v.cols, emb);
    STRIX_CUDA_CHECK(cudaGetLastError());
    F16RowsView f;
    f.weight = v.kv;
    f.count = v.kv.rows;
    f.x = emb;
    f.rows = v.rows;
    f.out = kv;
    enqueue_f16_rows(f, stream);
    const float inv_sqrt_hidden = static_cast<float>(1.0 / std::sqrt(static_cast<double>(v.hidden)));
    engram_inject_kernel<<<dim3(v.rows, 4), kInjectThreads, 0, stream>>>(v.stream, kv, v.q_norm, v.k_norm, v.hidden, v.eps,
                                                                         inv_sqrt_hidden);
    STRIX_CUDA_CHECK(cudaGetLastError());
}

void enqueue_argmax(const ArgmaxView& v, cudaStream_t stream) {
    require(v.rows > 0 && v.count > 0, "argmax geometry");
    argmax_kernel<<<v.rows, kArgmaxThreads, 0, stream>>>(v.logits, v.count, v.stride ? v.stride : v.count, v.first_index,
                                                          static_cast<ArgmaxCandidate*>(v.out));
    STRIX_CUDA_CHECK(cudaGetLastError());
}

}  // namespace strix::ds41::cuda
