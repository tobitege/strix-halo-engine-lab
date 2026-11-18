// Gated DeltaNet decode with deferred commit: causal depthwise conv (k=4) + SiLU, l2norm q/k, delta rule with sequential rows
// into a shadow state side, gated RMSNorm. See gdn.hpp for the contract.
#include "strix/cuda_core/gdn.hpp"

#include <cmath>

#include "../common.cuh"

namespace strix::cuda_core {
namespace {

// 128-thread block sum with the documented 256-thread order (warps 4..7 would hold zeros): butterfly per warp, then 0..3.
__device__ __forceinline__ float block_sum128(float v, float* smem) {
    v = warp_sum_xor(v);
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) smem[w] = v;
    __syncthreads();
    float s = __fadd_rn(__fadd_rn(__fadd_rn(smem[0], smem[1]), smem[2]), smem[3]);
    __syncthreads();
    return s;
}

// Block = 128 channels. Blocks [0, qk_blocks) are q/k heads (conv + silu + l2norm); the rest are v channels (conv + silu).
// tap order oldest first: ((0 + w0 s0) + w1 s1) + w2 s2) + w3 x. The conv state is read-only here.
__global__ void __launch_bounds__(128) conv_pre_kernel(const float* __restrict__ w, const float* __restrict__ conv_state,
                                                       const float* __restrict__ raw, float* qc, std::uint32_t channels,
                                                       std::uint32_t rows, std::uint32_t q_blocks, std::uint32_t qk_blocks,
                                                       float qscale, GdnSlotTable st) {
    __shared__ float red[4];
    pdl_trigger();
    if (st.row0) {  // multi-session: this block's group (blockIdx.y)
        pdl_wait();
        const std::uint32_t g = blockIdx.y;
        rows = st.nrows[g];
        conv_state += static_cast<std::size_t>(st.state_slot[g]) * st.conv_stride;
        raw += static_cast<std::size_t>(st.row0[g]) * channels;
        qc += static_cast<std::size_t>(st.row0[g]) * channels;
    }
    const std::uint32_t c = blockIdx.x * 128 + threadIdx.x;
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    float s0 = conv_state[c * 3], s1 = conv_state[c * 3 + 1], s2 = conv_state[c * 3 + 2];  // committed state: written only outside the graph
    pdl_wait();
    const bool qk = blockIdx.x < qk_blocks;
    const float scale = blockIdx.x < q_blocks ? qscale : 1.0f;
    for (std::uint32_t r = 0; r < rows; ++r) {
        const float x = raw[static_cast<std::size_t>(r) * channels + c];
        float sum = __fmul_rn(w0, s0);
        sum = __fadd_rn(sum, __fmul_rn(w1, s1));
        sum = __fadd_rn(sum, __fmul_rn(w2, s2));
        sum = __fadd_rn(sum, __fmul_rn(w3, x));
        float y = cc_silu(sum);
        s0 = s1;
        s1 = s2;
        s2 = x;
        if (qk) {
            const float ss = block_sum128(__fmul_rn(y, y), red);  // p = 0 + y*y is exact
            const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ss, 1e-6f)));
            y = __fmul_rn(__fmul_rn(y, inv), scale);
        }
        qc[static_cast<std::size_t>(r) * channels + c] = y;
    }
}


// Row-parallel twin of conv_pre_kernel for prefill chunks (BITWISE the same per-row arithmetic): block (128 channels, RB rows).
// The conv taps of row r are raw rows r-3..r-1 (negative = committed conv state).
constexpr std::uint32_t kConvRows = 32;
__global__ void __launch_bounds__(128) conv_prefill_kernel(const float* __restrict__ w, const float* __restrict__ conv_state,
                                                           const float* __restrict__ raw, float* qc, std::uint32_t channels,
                                                           std::uint32_t rows, std::uint32_t q_blocks, std::uint32_t qk_blocks,
                                                           float qscale) {
    __shared__ float red[4];
    const std::uint32_t c = blockIdx.x * 128 + threadIdx.x;
    const std::uint32_t r0 = blockIdx.y * kConvRows, r1 = min(rows, r0 + kConvRows);
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    float s[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int idx = static_cast<int>(r0) - 3 + i;
        s[i] = idx >= 0 ? raw[static_cast<std::size_t>(idx) * channels + c] : conv_state[c * 3 + (idx + 3)];
    }
    float s0 = s[0], s1 = s[1], s2 = s[2];
    const bool qk = blockIdx.x < qk_blocks;
    const float scale = blockIdx.x < q_blocks ? qscale : 1.0f;
    for (std::uint32_t r = r0; r < r1; ++r) {
        const float x = raw[static_cast<std::size_t>(r) * channels + c];
        float sum = __fmul_rn(w0, s0);
        sum = __fadd_rn(sum, __fmul_rn(w1, s1));
        sum = __fadd_rn(sum, __fmul_rn(w2, s2));
        sum = __fadd_rn(sum, __fmul_rn(w3, x));
        float y = cc_silu(sum);
        s0 = s1;
        s1 = s2;
        s2 = x;
        if (qk) {
            const float ss = block_sum128(__fmul_rn(y, y), red);
            const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(ss, 1e-6f)));
            y = __fmul_rn(__fmul_rn(y, inv), scale);
        }
        qc[static_cast<std::size_t>(r) * channels + c] = y;
    }
}

struct DeltaArgs {
    float* base;         // [2][heads][128][128]
    const float* qkvc;   // [rows][channels]
    const float* a_raw;  // [rows][heads]
    const float* b_raw;
    const float* ssm_a;
    const float* dt_bias;
    const float* gain;  // gated norm (window forward only)
    const float* z;     // [rows][v_width]
    float* out;         // [rows][v_width]
    float* conv;        // commit only
    const float* raw;   // commit only
};
struct CommitBatch {
    DeltaArgs a[kMaxGdnLayers];
};

template <bool OUT>
__device__ __forceinline__ void delta_body(const DeltaArgs& a, const GdnDims& d, std::uint32_t rows, std::uint32_t parity,
                                           std::uint32_t h, float eps) {
    constexpr int K = 128;
    __shared__ float ks[K], qs[K];
    __shared__ float red[4];
    const std::uint32_t t = threadIdx.x;
    const std::uint32_t kh = h % d.key_heads;
    const std::uint32_t channels = d.conv_channels();
    const std::size_t side = static_cast<std::size_t>(d.value_heads) * K * d.value_dim;
    const float* sp = a.base + parity * side + static_cast<std::size_t>(h) * K * d.value_dim + t;
    float* dp = a.base + (1 - parity) * side + static_cast<std::size_t>(h) * K * d.value_dim + t;
    float S[K];
    pdl_trigger();
#pragma unroll
    for (int k = 0; k < K; ++k) S[k] = sp[static_cast<std::size_t>(k) * d.value_dim];
    pdl_wait();
    const float dtb = a.dt_bias[h], sa = a.ssm_a[h];
    // row inputs are prefetched one row ahead (loads only: the arithmetic and its order are unchanged)
    float c_ks = 0.f, c_qs = 0.f, c_v = 0.f, c_b = 0.f, c_a = 0.f, c_z = 0.f;
    auto fetch = [&](std::uint32_t r) {
        const float* row = a.qkvc + static_cast<std::size_t>(r) * channels;
        c_ks = row[d.qk_width() + kh * K + t];
        if constexpr (OUT) {
            c_qs = row[kh * K + t];
            c_z = a.z[static_cast<std::size_t>(r) * d.v_width() + h * d.value_dim + t];
        }
        c_v = row[2 * d.qk_width() + h * d.value_dim + t];
        c_b = a.b_raw[static_cast<std::size_t>(r) * d.value_heads + h];
        c_a = a.a_raw[static_cast<std::size_t>(r) * d.value_heads + h];
    };
    if (rows > 0) fetch(0);
    for (std::uint32_t r = 0; r < rows; ++r) {
        ks[t] = c_ks;
        if constexpr (OUT) qs[t] = c_qs;
        const float vcol = c_v, zv = c_z;
        const float beta = cc_sigmoid(c_b);
        const float xin = __fadd_rn(c_a, dtb);
        const float sp_ = xin > 20.0f ? xin : cc_log1pf(cc_expf(xin));
        const float g = __fmul_rn(sa, sp_);
        const float decay = cc_expf(g);
        __syncthreads();
        if (r + 1 < rows) fetch(r + 1);
        float kv = 0.0f;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            S[k] = __fmul_rn(S[k], decay);
            kv = __fadd_rn(kv, __fmul_rn(S[k], ks[k]));
        }
        const float dl = __fmul_rn(__fsub_rn(vcol, kv), beta);
        // Register budget: S[] already holds 128 registers. Without this fence the compiler keeps the 128 ks[] values loaded by the
        // kv loop live into the update loop (256 values > 255 registers) and spills inside this serial per-row loop. The fence makes
        // the update loop re-read ks[] from shared memory (same values, no store in between): loads only, arithmetic order unchanged.
        asm volatile("" ::: "memory");
        float acc = 0.0f;
        if constexpr (OUT) {
#pragma unroll
            for (int k = 0; k < K; ++k) {
                S[k] = __fadd_rn(S[k], __fmul_rn(ks[k], dl));
                acc = __fadd_rn(acc, __fmul_rn(S[k], qs[k]));
            }
            const float ss = block_sum128(__fmul_rn(acc, acc), red);
            const float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, 128.0f), eps)));
            float v = __fmul_rn(acc, inv);
            v = __fmul_rn(v, a.gain[t]);
            v = __fmul_rn(v, cc_silu(zv));
            a.out[static_cast<std::size_t>(r) * d.v_width() + h * d.value_dim + t] = v;
        } else {
#pragma unroll
            for (int k = 0; k < K; ++k) S[k] = __fadd_rn(S[k], __fmul_rn(ks[k], dl));
        }
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < K; ++k) dp[static_cast<std::size_t>(k) * d.value_dim] = S[k];
}

// Multi-session adjustment of one group: offsets the row buffers and the state base, returns the group's rows and parity.
__device__ __forceinline__ void slot_adjust(DeltaArgs& a, const GdnDims& d, const GdnSlotTable& st, std::uint32_t g,
                                            std::uint32_t& rows, std::uint32_t& parity) {
    const std::size_t r0 = st.row0[g];
    const std::uint32_t ss = st.state_slot[g];
    rows = st.nrows[g];
    parity = st.parity[ss];
    a.base += static_cast<std::size_t>(ss) * st.delta_stride;
    a.qkvc += r0 * d.conv_channels();
    a.a_raw += r0 * d.value_heads;
    a.b_raw += r0 * d.value_heads;
    if (a.z) a.z += r0 * d.v_width();
    if (a.out) a.out += r0 * d.v_width();
}

__global__ void __launch_bounds__(128) delta_window_kernel(DeltaArgs a, GdnDims d, std::uint32_t rows,
                                                           const std::uint32_t* parity, float eps, GdnSlotTable st) {
    if (st.row0) {
        pdl_wait();
        std::uint32_t par = 0;
        slot_adjust(a, d, st, blockIdx.y, rows, par);
        delta_body<true>(a, d, rows, par, blockIdx.x, eps);
        return;
    }
    delta_body<true>(a, d, rows, *parity, blockIdx.x, eps);
}

// Partial accept for multi-session: re-apply the first accept[g] rows (0 < accept < nrows) of group g for layer blockIdx.y.
__global__ void __launch_bounds__(128) delta_apply_slots_kernel(CommitBatch b, GdnDims d, GdnSlotTable st, const std::uint32_t* accept) {
    const std::uint32_t g = blockIdx.z;
    const std::uint32_t k = accept[g];
    if (k == 0 || k >= st.nrows[g]) return;
    DeltaArgs a = b.a[blockIdx.y];
    std::uint32_t rows = 0, par = 0;
    slot_adjust(a, d, st, g, rows, par);
    delta_body<false>(a, d, k, par, blockIdx.x, 0.0f);
}

__global__ void __launch_bounds__(128) conv_commit_slots_kernel(CommitBatch b, std::uint32_t channels, GdnSlotTable st,
                                                                const std::uint32_t* accept) {
    const std::uint32_t g = blockIdx.z;
    const std::uint32_t k = accept ? accept[g] : st.nrows[g];
    if (k == 0) return;
    const DeltaArgs& a = b.a[blockIdx.y];
    float* conv = a.conv + static_cast<std::size_t>(st.state_slot[g]) * st.conv_stride;
    const float* raw = a.raw + static_cast<std::size_t>(st.row0[g]) * channels;
    const std::uint32_t c = blockIdx.x * 128 + threadIdx.x;
    const float s[3] = {conv[c * 3], conv[c * 3 + 1], conv[c * 3 + 2]};
    float n[3];
    for (int i = 0; i < 3; ++i) {
        const int idx = static_cast<int>(k) - 3 + i;
        n[i] = idx >= 0 ? raw[static_cast<std::size_t>(idx) * channels + c] : s[idx + 3];
    }
    conv[c * 3] = n[0];
    conv[c * 3 + 1] = n[1];
    conv[c * 3 + 2] = n[2];
}

__global__ void flip_slots_kernel(GdnSlotTable st, const std::uint32_t* accept) {
    const std::uint32_t g = threadIdx.x;
    const std::uint32_t k = accept ? accept[g] : st.nrows[g];
    if (k > 0) st.parity[st.state_slot[g]] ^= 1u;
}

__global__ void __launch_bounds__(128) delta_apply_kernel(CommitBatch b, GdnDims d, std::uint32_t rows,
                                                          const std::uint32_t* parity) {
    delta_body<false>(b.a[blockIdx.y], d, rows, *parity, blockIdx.x, 0.0f);
}

// conv state after accepting the first k raw rows: the last three of (s0, s1, s2, raw_0 .. raw_{k-1}).
__global__ void __launch_bounds__(128) conv_commit_kernel(CommitBatch b, std::uint32_t channels, std::uint32_t k) {
    const DeltaArgs& a = b.a[blockIdx.y];
    const std::uint32_t c = blockIdx.x * 128 + threadIdx.x;
    const float s[3] = {a.conv[c * 3], a.conv[c * 3 + 1], a.conv[c * 3 + 2]};
    float n[3];
    for (int i = 0; i < 3; ++i) {
        const int idx = static_cast<int>(k) - 3 + i;  // index into raw rows; negative = old state
        n[i] = idx >= 0 ? a.raw[static_cast<std::size_t>(idx) * channels + c] : s[idx + 3];
    }
    a.conv[c * 3] = n[0];
    a.conv[c * 3 + 1] = n[1];
    a.conv[c * 3 + 2] = n[2];
}

__global__ void flip_kernel(std::uint32_t* p) { *p ^= 1u; }

}  // namespace

cudaError_t gdn_window(const GdnDims& d, const GdnLayer& layer, const GdnState& state, const GdnWindow& win, const float* z,
                       float* out, std::uint32_t rows, float eps, const std::uint32_t* parity_dev, cudaStream_t stream) {
    if (d.key_dim != 128 || d.value_dim != 128 || d.conv_kernel != 4 || rows == 0 || rows > kMaxRows) return cudaErrorInvalidValue;
    const std::uint32_t C = d.conv_channels();
    const float qscale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d.key_dim)));
    if (auto e = launch_pdl(conv_pre_kernel, dim3(C / 128), dim3(128), stream, layer.conv_w, state.conv, win.raw, win.qkv_c, C, rows,
                            d.key_heads, 2 * d.key_heads, qscale, GdnSlotTable{});
        e != cudaSuccess)
        return e;
    DeltaArgs a{state.delta_base, win.qkv_c, win.a_raw, win.b_raw, layer.ssm_a, layer.dt_bias, layer.norm_gain, z, out,
                nullptr, nullptr};
    return launch_pdl(delta_window_kernel, dim3(d.value_heads), dim3(128), stream, a, d, rows, parity_dev, eps, GdnSlotTable{});
}

cudaError_t gdn_window_slots(const GdnDims& d, const GdnLayer& layer, const GdnState& state, const GdnWindow& win, const float* z,
                             float* out, std::uint32_t groups, const GdnSlotTable& t, float eps, cudaStream_t stream) {
    if (d.key_dim != 128 || d.value_dim != 128 || d.conv_kernel != 4 || groups == 0 || groups > kMaxRows || !t.row0) return cudaErrorInvalidValue;
    const std::uint32_t C = d.conv_channels();
    const float qscale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d.key_dim)));
    if (auto e = launch_pdl(conv_pre_kernel, dim3(C / 128, groups), dim3(128), stream, layer.conv_w, state.conv, win.raw, win.qkv_c, C,
                            0u, d.key_heads, 2 * d.key_heads, qscale, t);
        e != cudaSuccess)
        return e;
    DeltaArgs a{state.delta_base, win.qkv_c, win.a_raw, win.b_raw, layer.ssm_a, layer.dt_bias, layer.norm_gain, z, out,
                nullptr, nullptr};
    return launch_pdl(delta_window_kernel, dim3(d.value_heads, groups), dim3(128), stream, a, d, 0u, static_cast<const std::uint32_t*>(nullptr), eps, t);
}

cudaError_t gdn_commit_slots(const GdnDims& d, const GdnCommitLayer* layers, std::uint32_t n, std::uint32_t groups,
                             const GdnSlotTable& t, const std::uint32_t* accept, bool partial, cudaStream_t stream) {
    if (n == 0 || n > kMaxGdnLayers || groups == 0 || groups > kMaxRows || !t.row0 || (partial && !accept)) return cudaErrorInvalidValue;
    CommitBatch b{};
    for (std::uint32_t i = 0; i < n; ++i) {
        const auto& L = layers[i];
        b.a[i] = DeltaArgs{L.state.delta_base, L.win.qkv_c, L.win.a_raw, L.win.b_raw, L.layer.ssm_a, L.layer.dt_bias,
                           nullptr, nullptr, nullptr, L.state.conv, L.win.raw};
    }
    const std::uint32_t C = d.conv_channels();
    if (partial) {
        delta_apply_slots_kernel<<<dim3(d.value_heads, n, groups), 128, 0, stream>>>(b, d, t, accept);
        if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    }
    conv_commit_slots_kernel<<<dim3(C / 128, n, groups), 128, 0, stream>>>(b, C, t, accept);
    if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    flip_slots_kernel<<<1, groups, 0, stream>>>(t, accept);
    return cudaGetLastError();
}

cudaError_t gdn_commit(const GdnDims& d, const GdnCommitLayer* layers, std::uint32_t n, std::uint32_t window_rows,
                       std::uint32_t k, std::uint32_t* parity_dev, cudaStream_t stream) {
    if (n == 0 || n > kMaxGdnLayers || k > window_rows) return cudaErrorInvalidValue;
    if (k == 0) return cudaSuccess;
    CommitBatch b{};
    for (std::uint32_t i = 0; i < n; ++i) {
        const auto& L = layers[i];
        b.a[i] = DeltaArgs{L.state.delta_base, L.win.qkv_c, L.win.a_raw, L.win.b_raw, L.layer.ssm_a, L.layer.dt_bias,
                           nullptr, nullptr, nullptr, L.state.conv, L.win.raw};
    }
    const std::uint32_t C = d.conv_channels();
    if (k < window_rows) {
        delta_apply_kernel<<<dim3(d.value_heads, n), 128, 0, stream>>>(b, d, k, parity_dev);
        if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    }
    conv_commit_kernel<<<dim3(C / 128, n), 128, 0, stream>>>(b, C, k);
    if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    flip_kernel<<<1, 1, 0, stream>>>(parity_dev);
    return cudaGetLastError();
}

cudaError_t gdn_prefill_layer(const GdnDims& d, const GdnLayer& layer, const GdnState& state, const GdnWindow& win, const float* z,
                              float* out, std::uint32_t rows, float eps, const std::uint32_t* parity_dev, cudaStream_t stream) {
    if (d.key_dim != 128 || d.value_dim != 128 || d.conv_kernel != 4 || rows == 0) return cudaErrorInvalidValue;
    const std::uint32_t C = d.conv_channels();
    const float qscale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d.key_dim)));
    conv_prefill_kernel<<<dim3(C / 128, (rows + kConvRows - 1) / kConvRows), 128, 0, stream>>>(
        layer.conv_w, state.conv, win.raw, win.qkv_c, C, rows, d.key_heads, 2 * d.key_heads, qscale);
    if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    DeltaArgs a{state.delta_base, win.qkv_c, win.a_raw, win.b_raw, layer.ssm_a, layer.dt_bias, layer.norm_gain, z, out,
                nullptr, nullptr};
    delta_window_kernel<<<dim3(d.value_heads), 128, 0, stream>>>(a, d, rows, parity_dev, eps, GdnSlotTable{});
    if (auto e = cudaGetLastError(); e != cudaSuccess) return e;
    CommitBatch b{};
    b.a[0] = DeltaArgs{state.delta_base, win.qkv_c, win.a_raw, win.b_raw, layer.ssm_a, layer.dt_bias, nullptr, nullptr, nullptr, state.conv, win.raw};
    conv_commit_kernel<<<dim3(C / 128, 1), 128, 0, stream>>>(b, C, rows);
    return cudaGetLastError();
}

cudaError_t gdn_prefill_flip(std::uint32_t* parity_dev, cudaStream_t stream) {
    flip_kernel<<<1, 1, 0, stream>>>(parity_dev);
    return cudaGetLastError();
}

}  // namespace strix::cuda_core
