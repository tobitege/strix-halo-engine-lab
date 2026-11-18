#pragma once
// Shared dense decode attention (rows 1..16 per window, GQA any group, BF16 KV cache, FP32 math).
// Two kernels; every reduction has a fixed order independent of the row count, so row r of a window
// has the bits it has alone:
//   prep    per-head RMSNorm ((x*rsqrt(mean+eps)) * gain) + NeoX partial RoPE of q and k, K/V -> BF16 cache
//   attend  one block per (row, q head); 8 waves take positions w, w+8, ... with an online softmax, the
//           waves merge in wave order; optional sigmoid output gate; causal (row r sees keys 0..pos0+r)
// Layouts: q_in [row][head][q_stride] (q at +0, gate at +Dim when q_stride == 2*Dim);
//          k_in/v_in [row][kv_head][Dim]; cache [position][kv_head][Dim] bf16.
// Home: shared (attn_core): no model names; Dim is a template parameter (256 instantiated by qwen35).
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>

namespace strix::attn_core::dense_decode {

inline constexpr unsigned max_rows = 16U;
inline constexpr unsigned waves = 8U;

struct Params {
    unsigned rows, heads, kv_heads, q_stride, rot_dim, pos0;
    float eps, scale;
    double theta;
    bool gated;
};

__device__ __forceinline__ std::uint16_t to_bf16(float x) {
    const auto u = __builtin_bit_cast(std::uint32_t, x);
    if ((u & 0x7fffffffU) > 0x7f800000U)
        return std::uint16_t((u >> 16U) | 0x40U);
    return std::uint16_t((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U);
}
__device__ __forceinline__ float from_bf16(std::uint32_t h) { return __builtin_bit_cast(float, h << 16U); }

// grid (rows, heads + 2 * kv_heads): [0,heads) q, [heads, heads+kv) k, then v. block = Dim threads.
template <unsigned Dim>
__global__ void __launch_bounds__(Dim) prep_kernel(Params p, const float* q_in, const float* k_in, const float* v_in,
                                                   const float* q_gain, const float* k_gain, float* q_out,
                                                   std::uint16_t* k_cache, std::uint16_t* v_cache) {
    static_assert(Dim % 64U == 0U);
    __shared__ float red[Dim];
    __shared__ float vec[Dim];
    const unsigned row = blockIdx.x, slot = blockIdx.y, t = threadIdx.x;
    const unsigned pos = p.pos0 + row;
    if (slot >= p.heads + p.kv_heads) {
        const unsigned kh = slot - p.heads - p.kv_heads;
        v_cache[(std::size_t(pos) * p.kv_heads + kh) * Dim + t] = to_bf16(v_in[(std::size_t(row) * p.kv_heads + kh) * Dim + t]);
        return;
    }
    const bool is_q = slot < p.heads;
    const float x = is_q ? q_in[(std::size_t(row) * p.heads + slot) * p.q_stride + t]
                         : k_in[(std::size_t(row) * p.kv_heads + (slot - p.heads)) * Dim + t];
    red[t] = x * x;
    __syncthreads();
    for (unsigned s = Dim / 2U; s > 0U; s >>= 1U) {
        if (t < s)
            red[t] += red[t + s];
        __syncthreads();
    }
    const float r = 1.0F / sqrtf(red[0] / float(Dim) + p.eps);
    vec[t] = (x * r) * (is_q ? q_gain : k_gain)[t];
    __syncthreads();
    float y = vec[t];
    const unsigned half = p.rot_dim / 2U;
    if (t < p.rot_dim) {
        const unsigned i = t % half;
        const double ang = double(pos) * pow(p.theta, -2.0 * double(i) / double(p.rot_dim));
        const float c = float(cos(ang)), s = float(sin(ang));
        y = t < half ? vec[t] * c - vec[t + half] * s : vec[t] * c + vec[t - half] * s;
    }
    if (is_q)
        q_out[(std::size_t(row) * p.heads + slot) * Dim + t] = y;
    else
        k_cache[(std::size_t(pos) * p.kv_heads + (slot - p.heads)) * Dim + t] = to_bf16(y);
}

// grid (rows, heads), block = 32 * waves threads. ctx [row][head][Dim].
template <unsigned Dim>
__global__ void __launch_bounds__(32U * waves) attend_kernel(Params p, const float* q_n, const float* q_in,
                                                             const std::uint16_t* k_cache, const std::uint16_t* v_cache,
                                                             float* ctx) {
    constexpr unsigned per_lane = Dim / 32U;
    static_assert(per_lane % 8U == 0U);
    __shared__ float m_s[waves], l_s[waves];
    __shared__ float acc_s[waves][Dim];
    const unsigned row = blockIdx.x, head = blockIdx.y, wave = threadIdx.x / 32U, lane = threadIdx.x % 32U;
    const unsigned group = p.heads / p.kv_heads, kh = head / group;
    const unsigned last = p.pos0 + row;
    float q[per_lane];
    for (unsigned e = 0; e < per_lane; ++e)
        q[e] = q_n[(std::size_t(row) * p.heads + head) * Dim + lane * per_lane + e];
    float m = -INFINITY, l = 0.0F, acc[per_lane] = {};
    for (unsigned pos = wave; pos <= last; pos += waves) {
        const std::size_t at = (std::size_t(pos) * p.kv_heads + kh) * Dim + lane * per_lane;
        float s = 0.0F;
        for (unsigned e = 0; e < per_lane; ++e)
            s = __builtin_fmaf(q[e], from_bf16(k_cache[at + e]), s);
#pragma unroll
        for (unsigned o = 16U; o > 0U; o >>= 1U)
            s += __shfl_xor(s, int(o), 32);
        s *= p.scale;
        const float mn = fmaxf(m, s), corr = expf(m - mn), w = expf(s - mn);
        l = l * corr + w;
        for (unsigned e = 0; e < per_lane; ++e)
            acc[e] = acc[e] * corr + w * from_bf16(v_cache[at + e]);
        m = mn;
    }
    for (unsigned e = 0; e < per_lane; ++e)
        acc_s[wave][lane * per_lane + e] = acc[e];
    if (lane == 0) {
        m_s[wave] = m;
        l_s[wave] = l;
    }
    __syncthreads();
    for (unsigned d = threadIdx.x; d < Dim; d += 32U * waves) {
        float M = -INFINITY;
        for (unsigned w = 0; w < waves; ++w)
            M = fmaxf(M, m_s[w]);
        float L = 0.0F, O = 0.0F;
        for (unsigned w = 0; w < waves; ++w) {
            const float f = m_s[w] == -INFINITY ? 0.0F : expf(m_s[w] - M);
            L += l_s[w] * f;
            O += acc_s[w][d] * f;
        }
        float o = O / L;
        if (p.gated) {
            const float g = q_in[(std::size_t(row) * p.heads + head) * p.q_stride + Dim + d];
            o *= 1.0F / (1.0F + expf(-g));
        }
        ctx[(std::size_t(row) * p.heads + head) * Dim + d] = o;
    }
}

template <unsigned Dim>
inline void launch(const Params& p, const float* q_in, const float* k_in, const float* v_in, const float* q_gain,
                   const float* k_gain, float* q_scratch, std::uint16_t* k_cache, std::uint16_t* v_cache, float* ctx,
                   hipStream_t stream) {
    prep_kernel<Dim><<<dim3(p.rows, p.heads + 2U * p.kv_heads), Dim, 0, stream>>>(p, q_in, k_in, v_in, q_gain, k_gain,
                                                                                  q_scratch, k_cache, v_cache);
    attend_kernel<Dim><<<dim3(p.rows, p.heads), 32U * waves, 0, stream>>>(p, q_scratch, q_in, k_cache, v_cache, ctx);
}

} // namespace strix::attn_core::dense_decode
