#pragma once
// Shared FAST wide-window attention over the dense_decode cache layout (BF16 [position][kv_head][Dim]): the
// dense_decode prep (per-head RMSNorm + NeoX partial RoPE + K/V -> BF16 cache, bitwise the decode stage), then the
// shared dense causal GQA flash prefill core (dense_flash_prefill.hpp, BF16 WMMA operands, FP32 online softmax)
// instead of one decode-form block per (row, head) that re-reads every key: for a window of n rows at position p
// the decode form reads n x heads x p keys, the flash form stages each 16-key tile once per (16 rows, KV group).
// The flash core needs V^T rows ([kv_head * Dim + d][position]); this adapter keeps a V^T copy of the cache and
// refreshes it incrementally from `from` (the caller tracks the first position whose V^T may be stale: any write by
// another path lowers it). Optional sigmoid output gate (q_stride == 2 * Dim) after the flash epilogue, same formula
// as dense_decode attend.
// Identity: FAST (bit-changing against dense_decode: BF16 Q and P operands, WMMA sum order, exp2 online softmax);
// callers use it only under their wide-window FAST policy and judge it on end metrics (KL / top-1 vs exact).
// Home: shared (attn_core): no model names; Dim / Group are template parameters (qwen35 binds Dim 256, Group 6).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "dense_decode.hpp"
#include "dense_flash_prefill.hpp"

namespace strix::attn_core::dense_decode_flash {

struct Args {
    const float* q;           // prep output [row][head][Dim] FP32
    float* out;               // [row][head][Dim] FP32
    unsigned* failure;
    const std::uint16_t* k;   // BF16 cache [position][kv_head][Dim]
    const std::uint16_t* vt;  // BF16 V^T [kv_head * Dim + d][vstride]
    unsigned rows, heads, kv_heads, pos0, vstride;
};

template <unsigned Dim> constexpr float inv_sqrt_dim() {
    static_assert(Dim == 64U || Dim == 128U || Dim == 256U || Dim == 512U, "instantiated head dims");
    return Dim == 64U ? 0.125F : Dim == 128U ? 0.088388347648318440F : Dim == 256U ? 0.0625F : 0.044194173824159220F;
}

template <unsigned Dim, unsigned Group>
struct Policy {
    using Args = dense_decode_flash::Args;
    using Shape = dense::Shape<Dim, Group, (Dim >= 128U ? Dim / 128U : 1U)>;
    static constexpr float kScaleLog2 = inv_sqrt_dim<Dim>() * 1.44269504088896341F;
    static constexpr tile::Operand kOp = tile::Operand::bf16;
    __device__ static bool valid(const Args& a) { return a.rows && a.heads == a.kv_heads * Group && a.pos0 + a.rows <= a.vstride; }
    __host__ __device__ static unsigned rows(const Args& a) { return a.rows; }
    __device__ static unsigned position(const Args& a) { return a.pos0; }
    __device__ static unsigned window(const Args&) { return 0U; }
    __device__ static const float* query(const Args& a, unsigned row, unsigned h) { return a.q + (std::size_t(row) * a.heads + h) * Dim; }
    __device__ static float* out(const Args& a, unsigned row, unsigned h) { return a.out + (std::size_t(row) * a.heads + h) * Dim; }
    __device__ static const std::uint16_t* key(const Args& a, unsigned p, unsigned kvh) {
        return a.k + (std::size_t(p) * a.kv_heads + kvh) * Dim;
    }
    __device__ static const std::uint16_t* value_t(const Args& a, unsigned kvh, unsigned d) {
        return a.vt + std::size_t(kvh * Dim + d) * a.vstride;
    }
    __device__ static unsigned* failure(const Args& a) { return a.failure; }
};

// V^T refresh of positions [from, to): 32 x 32 tiles through LDS (coalesced reads and writes). width = kv_heads * Dim.
template <int = 0>
__global__ void __launch_bounds__(256) vt_kernel(const std::uint16_t* v_cache, std::uint16_t* vt, unsigned width, unsigned from,
                                                 unsigned to, unsigned vstride) {
    __shared__ std::uint16_t t[32][33];
    const unsigned p0 = from + blockIdx.x * 32U, c0 = blockIdx.y * 32U, x = threadIdx.x % 32U, y = threadIdx.x / 32U;
    for (unsigned i = y; i < 32U; i += 8U) {
        const unsigned p = p0 + i;
        t[i][x] = p < to ? v_cache[std::size_t(p) * width + c0 + x] : std::uint16_t(0);
    }
    __syncthreads();
    for (unsigned i = y; i < 32U; i += 8U) {
        const unsigned p = p0 + x;
        if (p < to) vt[std::size_t(c0 + i) * vstride + p] = t[x][i];
    }
}

template <unsigned Dim>
__global__ void __launch_bounds__(256) gate_kernel(float* ctx, const float* q_in, unsigned q_stride, std::size_t count) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += std::size_t(gridDim.x) * blockDim.x) {
        const std::size_t rh = i / Dim, d = i % Dim;
        const float g = q_in[rh * q_stride + Dim + d];
        ctx[i] *= 1.0F / (1.0F + expf(-g));
    }
}

// Split-K count (grid z) for a window: fill ~160 work-groups (8 per WGP of 20) while every split keeps >= 32 key
// tiles, capped by the workspace the caller owns (gemma4 prefill uses the same rule).
inline unsigned splits_for(unsigned rows, unsigned heads, unsigned group, unsigned dim, unsigned context, std::size_t ws_floats) {
    const unsigned groups = (rows + 15U) / 16U * (heads / group), tiles = context / 16U;
    unsigned s = std::max(1U, std::min({8U, (160U + groups - 1U) / groups, tiles / 32U}));
    while (s > 1U && dense::workspace_floats(rows, heads, dim, s) > ws_floats) --s;
    return s;
}

// p, q_in .. ctx as dense_decode::launch; vt [kv_heads * Dim][vstride] (vstride >= max positions, multiple of 16);
// V^T of positions [vt_from, p.pos0 + p.rows) is refreshed (vt_from <= p.pos0).
template <unsigned Dim, unsigned Group>
inline void launch(const dense_decode::Params& p, const float* q_in, const float* k_in, const float* v_in, const float* q_gain,
                   const float* k_gain, float* q_scratch, std::uint16_t* k_cache, std::uint16_t* v_cache, std::uint16_t* vt,
                   unsigned vstride, unsigned vt_from, unsigned* failure, float* ws, std::size_t ws_floats, float* ctx,
                   hipStream_t stream) {
    dense_decode::prep_kernel<Dim><<<dim3(p.rows, p.heads + 2U * p.kv_heads), Dim, 0, stream>>>(p, q_in, k_in, v_in, q_gain, k_gain,
                                                                                              q_scratch, k_cache, v_cache);
    const unsigned to = p.pos0 + p.rows, width = p.kv_heads * Dim;
    vt_kernel<<<dim3((to - vt_from + 31U) / 32U, width / 32U), 256, 0, stream>>>(v_cache, vt, width, vt_from, to, vstride);
    const Args a{q_scratch, ctx, failure, k_cache, vt, p.rows, p.heads, p.kv_heads, p.pos0, vstride};
    const unsigned splits = ws ? splits_for(p.rows, p.heads, Group, Dim, to, ws_floats) : 1U;
    dense::launch<Policy<Dim, Group>>(a, p.heads, stream, splits > 1U ? ws : nullptr, splits);
    if (p.gated)
        gate_kernel<Dim><<<1024, 256, 0, stream>>>(ctx, q_in, p.q_stride, std::size_t(p.rows) * p.heads * Dim);
}

}  // namespace strix::attn_core::dense_decode_flash
